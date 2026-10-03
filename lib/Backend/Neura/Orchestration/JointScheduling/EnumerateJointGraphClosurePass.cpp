//===- EnumerateJointGraphClosurePass.cpp ----------------------*- C++ -*-===//
//
// C++/MLIR-only bounded graph-closure controller for ORBIT.
//
// This pass is intentionally a controller, not a second rewrite
// implementation.  It clones the current ModuleOp, invokes the source-owned
// materializers through their public pass factories, invokes the source-owned
// graph-facts pass again, and uses that pass's structural_key as the only
// semantic graph identity.  A failed or unsupported materialization is kept as
// an unknown action with its diagnostic; it is never silently discarded or
// promoted to an illegal action.
//
// The controller explores one-step actions and the two requested two-step
// orders (fuse -> tile and tile -> fuse).  It is deliberately bounded and
// fail-closed.  A successful run with unsupported actions, a resource bound,
// or an interrupted traversal remains incomplete.
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "ClosureLegalityPreflight.h"
#include "JointRewritePreflight.h"
#include "Backend/Neura/Orchestration/JointScheduling/GraphFactsIO.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;
namespace json = llvm::json;

namespace {

constexpr llvm::StringLiteral kSchema = "orbit-joint-graph-closure-v1";
constexpr llvm::StringLiteral kResumeJournalSchema =
    "orbit-joint-graph-closure-resume-v1";
constexpr llvm::StringLiteral kResumeSnapshotSchema =
    "orbit-joint-graph-closure-resume-snapshot-v1";
// Bump this whenever materializer legality, facts semantics, or action-path
// meaning changes.  A dirty implementation must never reuse an old unknown
// action or frontier merely because the source/options still match.
constexpr llvm::StringLiteral kRewriteImplementationContract =
    "orbit-joint-graph-rewrite-contract-v8";
constexpr std::array<int64_t, 4> kSupportedFactors = {1, 2, 4, 8};
constexpr std::array<int64_t, 3> kDefaultReplicaFactors = {1, 2, 4};
constexpr std::array<int64_t, 3> kDefaultTilingFactors = {1, 2, 4};

struct Facts {
  json::Object structuralKey;
  std::string structuralText;
  std::string factsPath;
  std::string factsText;
  bool kFactsComplete = false;
  bool siblingFactsComplete = false;
  bool siblingMaterializationComplete = false;
};

struct GraphArtifact {
  std::string graphId;
  std::string mlirPath;
  std::string factsPath;
};

struct Action {
  enum class Kind { Tile, KTile, Replica, ProducerConsumer, Sibling };

  Kind kind = Kind::Tile;
  std::string firstTask;
  std::string secondTask;
  int64_t axis = -1;
  int64_t factor = 1;
  std::string fusionMode;
  std::string kMode;

  std::string kindName() const {
    switch (kind) {
    case Kind::Tile:
      return "tiling";
    case Kind::KTile:
      return "k_tiling";
    case Kind::Replica:
      return "replica";
    case Kind::ProducerConsumer:
      return "producer_consumer_fusion";
    case Kind::Sibling:
      return "sibling_fusion";
    }
    return "unknown";
  }
};

struct Attempt {
  Action action;
  std::string order;
  std::string path;
};

// A lineage is an original task (or one of the original tasks in a fused
// group).  The source-owned tilers materialize one dimension at a time, so a
// branch must remember the factor already selected for each dimension instead
// of treating any tiling attribute as a blanket "already tiled" marker.
struct DimensionSelection {
  int64_t mFactor = 1;
  int64_t nFactor = 1;
  int64_t kFactor = 1;
  int64_t replicaFactor = 1;
};

using LineageState = std::map<std::string, DimensionSelection>;
using TaskLineageState = std::map<std::string, LineageState>;

static std::string quoteOption(StringRef value) {
  std::string quoted = "\"";
  for (char character : value) {
    if (character == '\\' || character == '"')
      quoted += '\\';
    quoted += character;
  }
  quoted += '"';
  return quoted;
}

static std::string jsonText(const json::Value &value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << value;
  stream.flush();
  return text;
}

static std::string jsonLine(json::Object object) {
  return jsonText(json::Value(std::move(object))) + "\n";
}

static json::Array factorArray(ArrayRef<int64_t> factors) {
  json::Array array;
  for (int64_t factor : factors)
    array.push_back(factor);
  return array;
}

static std::string actionText(const Action &action) {
  std::string text = action.kindName();
  text += ":" + action.firstTask;
  if (!action.secondTask.empty())
    text += ":" + action.secondTask;
  if (action.axis >= 0)
    text += ":axis=" + std::to_string(action.axis);
  text += ":factor=" + std::to_string(action.factor);
  if (!action.fusionMode.empty())
    text += ":mode=" + action.fusionMode;
  if (!action.kMode.empty())
    text += ":k-mode=" + action.kMode;
  return text;
}

// Frontier keys are deliberately length-prefixed rather than delimiter
// joined.  Task names and the printed module may contain arbitrary
// punctuation, so a delimiter-only key could merge distinct states.
static void appendKeyPart(std::string &key, StringRef value) {
  key += std::to_string(value.size());
  key += ":";
  key += value.str();
}

static std::string serializeLineageState(const TaskLineageState &state) {
  std::string text = "tasks=" + std::to_string(state.size()) + ";";
  for (const auto &taskEntry : state) {
    appendKeyPart(text, taskEntry.first);
    text += "members=" + std::to_string(taskEntry.second.size()) + ";";
    for (const auto &lineageEntry : taskEntry.second) {
      appendKeyPart(text, lineageEntry.first);
      const DimensionSelection &selection = lineageEntry.second;
      text += "m=" + std::to_string(selection.mFactor) + ";";
      text += "n=" + std::to_string(selection.nFactor) + ";";
      text += "k=" + std::to_string(selection.kFactor) + ";";
      text += "r=" + std::to_string(selection.replicaFactor) + ";";
    }
  }
  return text;
}

static json::Object lineageObject(const TaskLineageState &state) {
  json::Array tasks;
  for (const auto &taskEntry : state) {
    json::Array members;
    for (const auto &lineageEntry : taskEntry.second) {
      const DimensionSelection &selection = lineageEntry.second;
      members.push_back(json::Object{
          {"lineage", lineageEntry.first},
          {"m_factor", selection.mFactor},
          {"n_factor", selection.nFactor},
          {"k_factor", selection.kFactor},
          {"replica_factor", selection.replicaFactor}});
    }
    tasks.push_back(json::Object{
        {"task", taskEntry.first},
        {"members", std::move(members)}});
  }
  return json::Object{{"tasks", std::move(tasks)}};
}

static bool parseLineageObject(const json::Object &object,
                               TaskLineageState &state, std::string &error) {
  state.clear();
  const json::Array *tasks = object.getArray("tasks");
  if (!tasks) {
    error = "resume checkpoint lineage has no tasks array";
    return false;
  }
  for (const json::Value &taskValue : *tasks) {
    const json::Object *taskObject = taskValue.getAsObject();
    const auto taskName = taskObject ? taskObject->getString("task")
                                     : std::optional<StringRef>();
    const json::Array *members = taskObject
                                     ? taskObject->getArray("members")
                                     : nullptr;
    if (!taskName || !members || taskName->empty() ||
        state.count(taskName->str())) {
      error = "resume checkpoint lineage has an invalid task entry";
      return false;
    }
    LineageState lineage;
    for (const json::Value &memberValue : *members) {
      const json::Object *member = memberValue.getAsObject();
      const auto lineageName = member ? member->getString("lineage")
                                      : std::optional<StringRef>();
      const auto mFactor = member ? member->getInteger("m_factor")
                                  : std::optional<int64_t>();
      const auto nFactor = member ? member->getInteger("n_factor")
                                  : std::optional<int64_t>();
      const auto kFactor = member ? member->getInteger("k_factor")
                                  : std::optional<int64_t>();
      const auto replicaFactor = member
                                     ? member->getInteger("replica_factor")
                                     : std::optional<int64_t>();
      if (!lineageName || lineageName->empty() || lineage.count(lineageName->str()) ||
          !mFactor || !nFactor || !kFactor || !replicaFactor ||
          *mFactor <= 0 || *nFactor <= 0 || *kFactor <= 0 ||
          *replicaFactor <= 0) {
        error = "resume checkpoint lineage has an invalid member entry";
        return false;
      }
      lineage.emplace(lineageName->str(), DimensionSelection{
                                               *mFactor, *nFactor, *kFactor,
                                               *replicaFactor});
    }
    if (lineage.empty()) {
      error = "resume checkpoint lineage has an empty task lineage";
      return false;
    }
    state.emplace(taskName->str(), std::move(lineage));
  }
  return true;
}

static std::string printedModuleText(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

static SmallVector<std::string> taskNames(ModuleOp module,
                                          StringRef functionName) {
  std::string error;
  FailureOr<func::FuncOp> selected =
      selectTaskFunction(module, functionName, error);
  if (failed(selected))
    return {};
  SmallVector<std::string> names;
  selected->walk([&](TaskflowTaskOp task) {
    names.push_back(task.getTaskName().str());
  });
  return names;
}

static TaskLineageState initialTaskLineageState(ModuleOp module,
                                                StringRef functionName) {
  TaskLineageState state;
  std::string error;
  FailureOr<func::FuncOp> selected =
      selectTaskFunction(module, functionName, error);
  if (failed(selected))
    return state;
  selected->walk([&](TaskflowTaskOp task) {
    LineageState lineage;
    lineage.emplace(task.getTaskName().str(), DimensionSelection{});
    state.emplace(task.getTaskName().str(), std::move(lineage));
  });
  return state;
}

static bool lineageStateFor(const TaskLineageState &state, StringRef taskName,
                            LineageState &lineage) {
  auto found = state.find(taskName.str());
  if (found == state.end())
    return false;
  lineage = found->second;
  return true;
}

static bool allLineagesFree(const LineageState &lineage,
                            int64_t DimensionSelection::*member) {
  for (const auto &entry : lineage)
    if (entry.second.*member != 1)
      return false;
  return true;
}

static bool runPassOnClone(ModuleOp module, std::unique_ptr<Pass> pass,
                           std::string &reason, std::string &diagnostic) {
  PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(std::move(pass));
  LogicalResult runResult = failure();
  LogicalResult verifyResult = failure();
  {
    ScopedDiagnosticHandler capture(module.getContext(),
                                    [&](Diagnostic &entry) {
                                      if (diagnostic.empty()) {
                                        llvm::raw_string_ostream stream(
                                            diagnostic);
                                        entry.print(stream);
                                        stream.flush();
                                      }
                                      return success();
                                    });
    runResult = manager.run(module);
    if (succeeded(runResult))
      verifyResult = verify(module.getOperation());
  }
  if (failed(runResult)) {
    reason = "source_owned_materializer_rejected";
    if (diagnostic.empty())
      diagnostic = "source-owned pass failed without a diagnostic";
    return false;
  }
  if (failed(verifyResult)) {
    reason = "materialized_module_verifier_failed";
    return false;
  }
  return true;
}

static bool isFactsOutputResourceDiagnostic(StringRef diagnostic,
                                            StringRef factsPath) {
  std::string failedPrefix = "failed while writing " + factsPath.str() + ":";
  std::string temporaryPrefix = "cannot create temporary output " +
                                factsPath.str() + ".tmp-";
  std::string publishPrefix = "cannot publish " + factsPath.str() + ":";
  return diagnostic.contains(failedPrefix) ||
         diagnostic.contains(temporaryPrefix) ||
         diagnostic.contains(publishPrefix);
}

static bool runMaterializer(ModuleOp module, StringRef functionName,
                            const Action &action, std::string &reason,
                            std::string &diagnostic) {
  std::unique_ptr<Pass> pass;
  std::string options;
  switch (action.kind) {
  case Action::Kind::Tile:
    pass = mlir::amoeba::neura::createMaterializeNeuraJointRewritePass();
    options = "task-name=" + quoteOption(action.firstTask) +
               " tile-axis=" + std::to_string(action.axis) +
               " tile-factor=" + std::to_string(action.factor) +
               " fusion-mode=none";
    break;
  case Action::Kind::KTile:
    pass = mlir::amoeba::neura::createMaterializeNeuraKReductionPass();
    options = "function=" + quoteOption(functionName) +
              " task-name=" + quoteOption(action.firstTask) +
               " k-mode=" + action.kMode + " k-factor=" +
               std::to_string(action.factor);
    break;
  case Action::Kind::Replica:
    pass = mlir::amoeba::neura::createMaterializeJointTaskReplicasPass();
    options = "function=" + quoteOption(functionName) +
              " task=" + quoteOption(action.firstTask) +
               " replicas=" + std::to_string(action.factor) +
               " axis=" + std::to_string(action.axis);
    break;
  case Action::Kind::ProducerConsumer:
    pass = mlir::amoeba::neura::createMaterializeNeuraJointRewritePass();
    options = "first-task-name=" + quoteOption(action.firstTask) +
               " second-task-name=" + quoteOption(action.secondTask) +
               " fusion-mode=" + quoteOption(action.fusionMode);
    break;
  case Action::Kind::Sibling:
    pass = mlir::amoeba::neura::createMaterializeNeuraJointRewritePass();
    options = "first-task-name=" + quoteOption(action.firstTask) +
               " second-task-name=" + quoteOption(action.secondTask) +
               " fusion-mode=sibling";
    break;
  }
  if (!pass) {
    reason = "missing_source_owned_materializer";
    return false;
  }
  if (failed(pass->initializeOptions(options, [&](const llvm::Twine &message) {
        reason = message.str();
        return failure();
      }))) {
    if (reason.empty())
      reason = "source_owned_materializer_option_parse_failed";
    return false;
  }
  return runPassOnClone(module, std::move(pass), reason, diagnostic);
}

static bool extractFacts(ModuleOp module, StringRef functionName,
                         StringRef outputPath, uint64_t serial,
                         Facts &facts, std::string &reason,
                         std::string &diagnostic) {
  std::string factsPath = outputPath.str() + ".closure-facts-" +
                          std::to_string(serial) + ".json";
  auto pass = mlir::amoeba::neura::createExtractJointTaskGraphFactsPass();
  std::string options = "function=" + quoteOption(functionName) +
                        " output=" + quoteOption(factsPath) +
                        " fact-only=true max-tile-factor=8 "
                        "max-fission-actions-per-task=0";
  if (failed(pass->initializeOptions(options, [&](const llvm::Twine &message) {
        reason = message.str();
        return failure();
      }))) {
    if (reason.empty())
      reason = "facts_pass_option_parse_failed";
    return false;
  }
  if (!runPassOnClone(module, std::move(pass), reason, diagnostic)) {
    if (isFactsOutputResourceDiagnostic(diagnostic, factsPath))
      reason = "facts_output_resource_failure";
    llvm::sys::fs::remove(factsPath);
    return false;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(factsPath);
  if (!buffer) {
    reason = "facts_output_missing";
    llvm::sys::fs::remove(factsPath);
    return false;
  }
  llvm::Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    reason = "facts_output_invalid_json";
    llvm::consumeError(parsed.takeError());
    llvm::sys::fs::remove(factsPath);
    return false;
  }
  json::Object *root = parsed->getAsObject();
  json::Object *structural = root ? root->getObject("structural_key") : nullptr;
  if (!root || !structural) {
    reason = "facts_output_missing_structural_key";
    llvm::sys::fs::remove(factsPath);
    return false;
  }
  facts.structuralKey = *structural;
  facts.structuralText = jsonText(json::Value(json::Object(*structural)));
  facts.factsPath = factsPath;
  facts.factsText = (*buffer)->getBuffer().str();
  facts.kFactsComplete = root->getBoolean("k_reduction_tasks_complete")
                             .value_or(false);
  facts.siblingFactsComplete =
      root->getBoolean("sibling_fusion_candidates_complete").value_or(false);
  facts.siblingMaterializationComplete =
      root->getBoolean("sibling_fusion_materialization_complete")
          .value_or(false);
  return true;
}

static json::Object actionObject(const Action &action, StringRef status,
                                 StringRef reason, StringRef diagnostic,
                                 StringRef order, StringRef path) {
  json::Object object{{"schema", kSchema.str()},
                      {"record_type", "action"},
                      {"kind", action.kindName()},
                      {"status", status.str()},
                      {"reason", reason.str()},
                      {"diagnostic", diagnostic.str()},
                      {"order", order.str()},
                      {"path", path.str()},
                      {"first_task", action.firstTask},
                      {"factor", action.factor}};
  if (!action.secondTask.empty())
    object["second_task"] = action.secondTask;
  if (action.axis >= 0)
    object["axis"] = action.axis;
  if (!action.fusionMode.empty())
    object["fusion_mode"] = action.fusionMode;
  if (!action.kMode.empty())
    object["k_mode"] = action.kMode;
  return object;
}

static json::Object graphObject(const Facts &facts, StringRef path,
                                bool deduplicated, StringRef firstPath,
                                StringRef order, const GraphArtifact &artifact) {
  json::Object object{{"schema", kSchema.str()},
                      {"record_type", "graph"},
                      {"status", "legal"},
                      {"resource_placement_checked", false},
                      {"path", path.str()},
                      {"order", order.str()},
                      {"deduplicated", deduplicated},
                      {"k_facts_complete", facts.kFactsComplete},
                      {"sibling_facts_complete", facts.siblingFactsComplete},
                      {"sibling_materialization_complete",
                       facts.siblingMaterializationComplete},
                      {"graph_id", artifact.graphId},
                      {"mlir_path", artifact.mlirPath},
                      {"facts_path", artifact.factsPath},
                      {"structural_key", json::Value(
                                             json::Object(facts.structuralKey))}};
  if (!firstPath.empty())
    object["first_path"] = firstPath.str();
  return object;
}

static bool isTransientResourceBlocker(StringRef blocker) {
  return blocker == "max_actions_reached" ||
         blocker == "max_graphs_reached";
}

class EnumerateJointGraphClosurePass
    : public PassWrapper<EnumerateJointGraphClosurePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      EnumerateJointGraphClosurePass)

  EnumerateJointGraphClosurePass() = default;
  EnumerateJointGraphClosurePass(const EnumerateJointGraphClosurePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "enumerate-joint-graph-closure";
  }
  StringRef getDescription() const override {
    return "Enumerate bounded C++ MLIR joint graph rewrite closure";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    neura::NeuraDialect,
                    TaskflowDialect>();
  }

  Option<std::string> outputFile{*this, "output", llvm::cl::init("")};
  Option<std::string> functionName{*this, "function", llvm::cl::init("")};
  Option<std::string> stageName{*this, "stage", llvm::cl::init("full-joint")};
  Option<std::string> replicaFactors{*this, "replica-factors",
                                     llvm::cl::init("")};
  Option<std::string> tilingFactors{*this, "tiling-factors",
                                    llvm::cl::init("")};
  Option<bool> artifactBytecode{*this, "artifact-bytecode",
                               llvm::cl::init(false)};
  Option<std::string> factsCompression{*this, "facts-compression",
                                        llvm::cl::init("legacy-json")};
  Option<std::string> resumeJournal{*this, "resume-journal",
                                    llvm::cl::init("")};
  Option<int64_t> resumeSnapshotActions{
      *this, "resume-snapshot-actions", llvm::cl::init(256)};
  Option<std::string> resumeSnapshotCompression{
      *this, "resume-snapshot-compression", llvm::cl::init("legacy-json")};
  Option<int64_t> maxGraphs{*this, "max-graphs", llvm::cl::init(1024)};
  Option<int64_t> maxActions{*this, "max-actions", llvm::cl::init(10000)};

private:
  // A frontier entry owns the complete source-owned module for one rewrite
  // state.  The phase and enabled dimension flags are part of the state key:
  // equal modules reached in different approved orders can have different
  // legal successors.
  struct FrontierEntry {
    enum class Kind { Dimensions, FusionOnly, FusionPrefix, DimensionSuffix };

    OwningOpRef<ModuleOp> module;
    TaskLineageState lineage;
    std::string path;
    Kind kind = Kind::Dimensions;
    bool includeReplica = false;
    bool includeTiling = false;
    std::string structuralText;
  };

  struct FrontierCheckpointEntry {
    std::string structuralText;
    GraphArtifact artifact;
    std::string path;
    FrontierEntry::Kind kind = FrontierEntry::Kind::Dimensions;
    bool includeReplica = false;
    bool includeTiling = false;
    // This is the exact traversal module before persistGraphArtifact stamps
    // the independently stored artifact with its graph id.  It is the
    // source-owned witness used to restore the memo key, rather than the
    // printed artifact text.
    std::string moduleText;
    bool graphVariantPresent = false;
    std::string graphVariantValue;
    TaskLineageState lineage;
  };

  int stageLevel() const {
    StringRef stage = stageName.getValue();
    if (stage == "shape-only") return 1;
    if (stage == "shape-temporal") return 2;
    if (stage == "shape-temporal-replica") return 3;
    if (stage == "shape-temporal-replica-tiling") return 4;
    if (stage == "full-joint") return 5;
    return 0;
  }

  llvm::StringSet<> seenStructural;
  llvm::StringMap<std::string> firstPathByStructural;
  uint64_t actionAttempts = 0;
  uint64_t graphCount = 0;
  uint64_t unknownCount = 0;
  uint64_t incompleteFactsCount = 0;
  uint64_t rejectedCount = 0;
  uint64_t serial = 0;
  uint64_t nextGraphOrdinal = 1;
  bool budgetReached = false;
  std::string firstBlocker;
  std::string outputDirectory;
  std::string graphArtifactDirectory;
  llvm::StringMap<GraphArtifact> artifactByStructural;
  llvm::StringSet<> frontierMemo;
  uint64_t frontierMemoHits = 0;
  uint64_t frontierStatesEnqueued = 0;
  uint64_t provenIllegalCount = 0;
  uint64_t preflightGraphBuildCount = 0;
  SmallVector<int64_t, 4> replicaAlphabet;
  SmallVector<int64_t, 4> tilingAlphabet;
  bool resuming = false;
  std::string originalInputText;
  std::vector<std::string> resumeDataRecords;
  // Each entry carries an exact module witness; keep it out of SmallVector's
  // inline storage so checkpoint memory is explicitly heap-bounded.
  SmallVector<FrontierCheckpointEntry, 0> frontierLedger;
  SmallVector<FrontierEntry> restoredFrontier;
  llvm::StringSet<> attemptedActionPaths;
  llvm::StringSet<> recordedGraphPaths;
  // The journal contains the header metadata followed by exactly the
  // action/graph records that are eligible for the final closure JSONL.  A
  // checkpoint/footer may follow those records, so the byte count lets the
  // final atomic output stream only the record prefix without retaining all
  // record text in memory.
  std::string journalPath;
  std::string resumeSnapshotPath;
  uint64_t journalSegmentStart = 0;
  uint64_t journalHeaderBytes = 0;
  uint64_t journalDataBytes = 0;
  uint64_t journalRecordCount = 0;
  uint64_t journalBytes = 0;
  uint64_t snapshotSequence = 0;
  uint64_t lastSnapshotActionAttempts = 0;
  bool snapshotRequested = false;
  bool snapshotWriteFailed = false;
  bool journalWriteFailed = false;
  bool resourceFailure = false;
  bool outputWriteFailed = false;
  std::string journalError;

  json::Object closureHeaderObject(bool complete) const {
    return json::Object{
        {"schema", kSchema.str()},
        {"record_type", "header"},
        {"rewrite_implementation_contract",
         kRewriteImplementationContract.str()},
        {"function", functionName.getValue()},
        {"stage", stageName.getValue()},
        {"scope", "bounded_cross_task_rewrite_closure"},
        {"requires_downstream_4x4_schedule", true},
        {"source_owned_structural_identity", true},
        {"uses_sha256_identity", false},
        // factor_alphabet is retained as a compatibility alias for the
        // tiling alphabet. Replica factors are independent.
        {"factor_alphabet", factorArray(tilingAlphabet)},
        {"tiling_factor_alphabet", factorArray(tilingAlphabet)},
        {"replica_factor_alphabet", factorArray(replicaAlphabet)},
        {"orders", json::Array{"fuse_then_tile", "tile_then_fuse"}},
        {"graph_artifact_directory", graphArtifactDirectory},
        {"graph_facts_storage_format", factsCompression.getValue()},
        {"graph_artifact_format", artifactBytecode.getValue()
             ? "identity-text-variants-bytecode-v1" : "text"},
        {"coverage", "graph-rewrites-only; scheduling-coverage-delegated"},
        {"stage_coverage_note",
         "all stages delegate order and location to the selected downstream "
         "scheduler; graph closure proves rewrite coverage only"},
        {"complete", complete},
        {"status", complete ? "complete" : "incomplete"}};
  }

  json::Object closureFooterObject(bool complete,
                                   bool durableCheckpoint = false) const {
    json::Object object{
        {"schema", kSchema.str()},
        {"record_type", "footer"},
        {"rewrite_implementation_contract",
         kRewriteImplementationContract.str()},
        {"status", complete ? "complete" : "incomplete"},
        {"complete", complete},
        {"factor_alphabet", factorArray(tilingAlphabet)},
        {"tiling_factor_alphabet", factorArray(tilingAlphabet)},
        {"replica_factor_alphabet", factorArray(replicaAlphabet)},
        {"graph_scope_complete", complete},
        {"action_attempts", static_cast<int64_t>(actionAttempts)},
        {"graph_count", static_cast<int64_t>(graphCount)},
        {"unique_structural_keys",
         static_cast<int64_t>(seenStructural.size())},
        {"unknown_action_count", static_cast<int64_t>(unknownCount)},
        {"incomplete_facts_count",
         static_cast<int64_t>(incompleteFactsCount)},
        {"rejected_action_count", static_cast<int64_t>(rejectedCount)},
        {"proven_illegal_action_count",
         static_cast<int64_t>(provenIllegalCount)},
        {"frontier_states_enqueued",
         static_cast<int64_t>(frontierStatesEnqueued)},
        {"frontier_memo_hits", static_cast<int64_t>(frontierMemoHits)},
        {"frontier_memo_size", static_cast<int64_t>(frontierMemo.size())},
        {"preflight_graph_build_count",
         static_cast<int64_t>(preflightGraphBuildCount)},
        {"resource_bound_reached", budgetReached},
        {"resource_failure", resourceFailure},
        {"snapshot_write_failed", snapshotWriteFailed},
        {"journal_write_failed", journalWriteFailed},
        {"journal_error", journalError},
        {"first_blocker", firstBlocker.empty() ? "" : firstBlocker}};
    if (durableCheckpoint) {
      object["durable_checkpoint"] = true;
      object["journal_record_count"] =
          static_cast<int64_t>(journalRecordCount);
      object["journal_data_bytes"] = static_cast<int64_t>(journalDataBytes);
      object["journal_write_failed"] = journalWriteFailed;
      object["resource_failure"] = resourceFailure;
      object["journal_error"] = journalError;
    }
    return object;
  }

  std::string journalHeaderLine() const {
    return jsonLine(json::Object{
        {"schema", kSchema.str()},
        {"record_type", "journal_header"},
        {"journal_schema", "orbit-joint-graph-closure-journal-v1"},
        {"resume_schema", kResumeJournalSchema.str()},
        {"rewrite_implementation_contract",
         kRewriteImplementationContract.str()},
        {"function", functionName.getValue()},
        {"stage", stageName.getValue()},
        {"output", outputFile.getValue()},
        {"graph_artifact_directory", graphArtifactDirectory},
        {"source_module_text", originalInputText},
        {"options", json::Object{
            {"function", functionName.getValue()},
            {"stage", stageName.getValue()},
            {"replica_factors", replicaFactors.getValue()},
            {"tiling_factors", tilingFactors.getValue()},
            {"artifact_bytecode", artifactBytecode.getValue()},
            {"facts_compression", factsCompression.getValue()},
            {"resume_snapshot_actions", resumeSnapshotActions.getValue()},
            {"resume_snapshot_compression", resumeSnapshotCompression.getValue()},
            {"max_graphs", maxGraphs.getValue()},
            {"max_actions", maxActions.getValue()}}},
        {"complete", false},
        {"status", "incomplete"}});
  }

  void noteJournalFailure(StringRef error) {
    journalWriteFailed = true;
    resourceFailure = true;
    budgetReached = true;
    journalError = error.str();
    if (firstBlocker.empty())
      firstBlocker = "durable_journal_write_failed";
  }

  bool restoreJournalPrefix(std::string &error) {
    int descriptor = -1;
    std::error_code ec = llvm::sys::fs::openFileForReadWrite(
        journalPath, descriptor, llvm::sys::fs::CD_OpenExisting,
        llvm::sys::fs::OF_None);
    if (ec) {
      error = "cannot reopen durable journal for truncation " + journalPath +
              ": " + ec.message();
      return false;
    }
    std::error_code resizeError =
        llvm::sys::fs::resize_file(descriptor, journalBytes);
    std::error_code closeError = llvm::sys::fs::closeFile(descriptor);
    if (resizeError) {
      error = "cannot restore durable journal prefix " + journalPath +
              ": " + resizeError.message();
      if (closeError)
        error += "; close: " + closeError.message();
      return false;
    }
    if (closeError) {
      error = "cannot close durable journal after truncation " + journalPath +
              ": " + closeError.message();
      return false;
    }
    return true;
  }

  bool appendJournalText(StringRef text, bool isDataRecord,
                         std::string *errorOut = nullptr) {
    if (journalPath.empty()) {
      std::string error = "durable journal path is not initialized";
      noteJournalFailure(error);
      if (errorOut)
        *errorOut = error;
      return false;
    }
    std::error_code ec;
    llvm::raw_fd_ostream stream(
        journalPath, ec, llvm::sys::fs::CD_OpenAlways,
        llvm::sys::fs::FA_Write, llvm::sys::fs::OF_Append);
    if (ec) {
      std::string error = "cannot append durable journal " + journalPath +
                           ": " + ec.message();
      noteJournalFailure(error);
      if (errorOut)
        *errorOut = error;
      return false;
    }
    stream << text;
    stream.close();
    if (stream.has_error()) {
      std::string error = "failed appending durable journal " + journalPath +
                           ": " + stream.error().message();
      stream.clear_error();
      std::string restoreError;
      if (!restoreJournalPrefix(restoreError))
        error += "; " + restoreError;
      noteJournalFailure(error);
      if (errorOut)
        *errorOut = error;
      return false;
    }
    journalBytes += text.size();
    if (isDataRecord) {
      journalDataBytes += text.size();
      ++journalRecordCount;
    }
    return true;
  }

  bool initializeJournal(std::string &error) {
    journalPath = outputFile.getValue() + ".journal.jsonl";
    resumeSnapshotPath = outputFile.getValue() + ".resume.snapshot.json";
    llvm::sys::fs::file_status status;
    std::error_code ec = llvm::sys::fs::status(journalPath, status);
    if (ec && ec != std::make_error_code(std::errc::no_such_file_or_directory)) {
      error = "cannot inspect durable journal " + journalPath + ": " +
              ec.message();
      noteJournalFailure(error);
      return false;
    }
    // A prior run's sidecar is itself durable evidence.  Start a new
    // journal segment after it instead of truncating or deleting it; the
    // segment byte offset keeps final output streaming scoped to this run.
    if (!ec) {
      if (!llvm::sys::fs::is_regular_file(status)) {
        error = "durable journal path is not a regular file: " + journalPath;
        noteJournalFailure(error);
        return false;
      }
      journalBytes = status.getSize();
    }
    journalSegmentStart = journalBytes;
    if (!appendJournalText(journalHeaderLine(), /*isDataRecord=*/false,
                           &error))
      return false;
    journalHeaderBytes = journalBytes;
    return true;
  }

  bool appendJournalRecord(StringRef line) {
    return appendJournalText(line, /*isDataRecord=*/true);
  }

  json::Object resumeCheckpointObject() const {
    json::Array attempted;
    SmallVector<std::string> sortedAttempts;
    sortedAttempts.reserve(attemptedActionPaths.size());
    for (const auto &entry : attemptedActionPaths)
      sortedAttempts.push_back(entry.getKey().str());
    llvm::sort(sortedAttempts);
    for (const std::string &path : sortedAttempts)
      attempted.push_back(path);

    json::Array frontier;
    for (const FrontierCheckpointEntry &entry : frontierLedger) {
      frontier.push_back(json::Object{
          {"structural_text", entry.structuralText},
          {"graph_id", entry.artifact.graphId},
          {"mlir_path", entry.artifact.mlirPath},
          {"facts_path", entry.artifact.factsPath},
          {"path", entry.path},
          {"kind", frontierKindName(entry.kind).str()},
          {"include_replica", entry.includeReplica},
          {"include_tiling", entry.includeTiling},
          {"module_text", entry.moduleText},
          {"graph_variant_present", entry.graphVariantPresent},
          {"graph_variant_value", entry.graphVariantValue},
          {"lineage", lineageObject(entry.lineage)}});
    }
    return json::Object{
        {"schema", kSchema.str()},
        {"record_type", "resume_snapshot"},
        {"snapshot_schema", kResumeSnapshotSchema.str()},
        {"resume_schema", kResumeJournalSchema.str()},
        {"rewrite_implementation_contract",
         kRewriteImplementationContract.str()},
        {"complete", false},
        {"source_module_text", originalInputText},
        {"output", outputFile.getValue()},
        {"graph_artifact_directory", graphArtifactDirectory},
        {"facts_compression", factsCompression.getValue()},
        {"journal_segment_start", static_cast<int64_t>(journalSegmentStart)},
        {"journal_header_bytes", static_cast<int64_t>(journalHeaderBytes)},
        {"journal_data_bytes", static_cast<int64_t>(journalDataBytes)},
        {"journal_prefix_end",
         static_cast<int64_t>(journalHeaderBytes + journalDataBytes)},
        {"snapshot_sequence", static_cast<int64_t>(snapshotSequence)},
        {"resume_data_record_count", static_cast<int64_t>(journalRecordCount)},
        {"action_attempts", static_cast<int64_t>(actionAttempts)},
        {"graph_count", static_cast<int64_t>(graphCount)},
        {"unique_structural_keys",
         static_cast<int64_t>(seenStructural.size())},
        {"unknown_action_count", static_cast<int64_t>(unknownCount)},
        {"incomplete_facts_count", static_cast<int64_t>(incompleteFactsCount)},
        {"rejected_action_count", static_cast<int64_t>(rejectedCount)},
        {"proven_illegal_action_count",
         static_cast<int64_t>(provenIllegalCount)},
        {"frontier_states_enqueued",
         static_cast<int64_t>(frontierStatesEnqueued)},
        {"frontier_memo_hits", static_cast<int64_t>(frontierMemoHits)},
        {"frontier_memo_size", static_cast<int64_t>(frontierMemo.size())},
        {"preflight_graph_build_count",
         static_cast<int64_t>(preflightGraphBuildCount)},
        {"next_graph_ordinal", static_cast<int64_t>(nextGraphOrdinal)},
        {"next_serial", static_cast<int64_t>(serial)},
        {"journal_write_failed", journalWriteFailed},
        {"resource_failure", resourceFailure},
        {"snapshot_write_failed", snapshotWriteFailed},
        {"journal_error", journalError},
        {"first_blocker", firstBlocker.empty() ? "" : firstBlocker},
        {"pending_frontier_count",
         static_cast<int64_t>(frontierLedger.size())},
        {"frontier_memo_keys", [&]() {
           json::Array keys;
           SmallVector<std::string> sortedKeys;
           sortedKeys.reserve(frontierMemo.size());
           for (const auto &entry : frontierMemo)
             sortedKeys.push_back(entry.getKey().str());
           llvm::sort(sortedKeys);
           for (const std::string &key : sortedKeys)
             keys.push_back(key);
           return keys;
         }()},
        {"attempted_action_paths", std::move(attempted)},
        {"frontier", std::move(frontier)}};
  }

  void noteSnapshotFailure(StringRef error) {
    snapshotWriteFailed = true;
    resourceFailure = true;
    budgetReached = true;
    journalError = error.str();
    if (firstBlocker.empty())
      firstBlocker = "resume_snapshot_write_failed";
  }

  bool writeResumeSnapshot() {
    if (resumeSnapshotPath.empty() || journalPath.empty()) {
      noteSnapshotFailure("resume snapshot paths are not initialized");
      return false;
    }
    ++snapshotSequence;
    std::string error;
    // Snapshots contain full module/frontier witnesses. Compress their exact
    // JSON bytes to avoid needing two large uncompressed copies on disk during
    // atomic publication. A separate magic keeps snapshots distinct from
    // graph-facts artifacts, while sharing the checked lossless codec.
    SmallVector<uint8_t, 0> storedSnapshot;
    auto codec = resumeSnapshotCompression.getValue() == "zstd"
                     ? graph_facts_io::FactsCompression::Zstd
                     : graph_facts_io::FactsCompression::LegacyJson;
    if (!graph_facts_io::encodeGraphFacts(
            jsonLine(resumeCheckpointObject()), codec, storedSnapshot, error)) {
      noteSnapshotFailure(error);
      return false;
    }
    if (codec != graph_facts_io::FactsCompression::LegacyJson)
      std::memcpy(storedSnapshot.data(), "ORBITRS1", 8);
    if (!writeAtomically(
            resumeSnapshotPath,
            [&](llvm::raw_ostream &stream) {
              stream.write(reinterpret_cast<const char *>(storedSnapshot.data()),
                           storedSnapshot.size());
              return true;
            },
            error)) {
      if (error.empty())
        error = "atomic resume snapshot publication failed";
      noteSnapshotFailure(error);
      return false;
    }
    lastSnapshotActionAttempts = actionAttempts;
    snapshotRequested = false;
    return true;
  }

  json::Object resumeSnapshotMarkerObject() const {
    return json::Object{
        {"schema", kSchema.str()},
        {"record_type", "resume_snapshot_marker"},
        {"snapshot_schema", kResumeSnapshotSchema.str()},
        {"rewrite_implementation_contract",
         kRewriteImplementationContract.str()},
        {"snapshot_path", resumeSnapshotPath},
        {"snapshot_sequence", static_cast<int64_t>(snapshotSequence)},
        {"journal_segment_start", static_cast<int64_t>(journalSegmentStart)},
        {"journal_prefix_end",
         static_cast<int64_t>(journalHeaderBytes + journalDataBytes)}};
  }

  bool appendDurableCheckpoint() {
    if (!writeResumeSnapshot())
      return false;
    if (!appendJournalText(jsonLine(resumeSnapshotMarkerObject()),
                           /*isDataRecord=*/false))
      return false;
    return appendJournalText(
        jsonLine(closureFooterObject(/*complete=*/false,
                                     /*durableCheckpoint=*/true)),
        /*isDataRecord=*/false);
  }

  bool streamJournalRecords(llvm::raw_ostream &stream,
                            std::string &error) const {
    if (journalDataBytes == 0)
      return true;
    llvm::sys::fs::file_status status;
    std::error_code ec = llvm::sys::fs::status(journalPath, status);
    if (ec) {
      error = "cannot stat durable journal " + journalPath + ": " +
              ec.message();
      return false;
    }
    if (status.getSize() < journalHeaderBytes + journalDataBytes) {
      error = "durable journal is shorter than its committed record prefix";
      return false;
    }
    constexpr uint64_t kChunkBytes = 1 << 20;
    uint64_t offset = journalHeaderBytes;
    uint64_t remaining = journalDataBytes;
    while (remaining != 0) {
      uint64_t chunk = std::min<uint64_t>(remaining, kChunkBytes);
      auto buffer = llvm::MemoryBuffer::getFileSlice(journalPath, chunk,
                                                      offset);
      if (!buffer) {
        error = "cannot read durable journal record prefix " + journalPath;
        return false;
      }
      stream.write((*buffer)->getBufferStart(), chunk);
      offset += chunk;
      remaining -= chunk;
    }
    return true;
  }

  static bool parseResumeFrontierKind(StringRef name,
                                      FrontierEntry::Kind &kind) {
    if (name == "dimensions") {
      kind = FrontierEntry::Kind::Dimensions;
      return true;
    }
    if (name == "fusion_only") {
      kind = FrontierEntry::Kind::FusionOnly;
      return true;
    }
    if (name == "fusion_prefix") {
      kind = FrontierEntry::Kind::FusionPrefix;
      return true;
    }
    if (name == "dimension_suffix") {
      kind = FrontierEntry::Kind::DimensionSuffix;
      return true;
    }
    return false;
  }

  static bool parseRequiredInteger(const json::Object &object, StringRef key,
                                   int64_t &value, std::string &error) {
    auto parsed = object.getInteger(key);
    if (!parsed) {
      error = "resume journal is missing integer field " + key.str();
      return false;
    }
    value = *parsed;
    return true;
  }

  bool restoreResumeJournal(ModuleOp module, std::string &error) {
    resuming = true;
    const std::string path = resumeJournal.getValue();
    if (path.empty()) {
      error = "resume journal path is empty";
      return false;
    }
    const std::string expectedJournal = outputFile.getValue() + ".journal.jsonl";
    if (path != expectedJournal) {
      error = "resume journal must be the output's exact durable journal path";
      return false;
    }
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
      error = "cannot read resume journal " + path;
      return false;
    }

    json::Object currentHeader;
    std::optional<json::Object> currentCheckpoint;
    std::vector<std::string> currentData;
    bool loadedSnapshot = false;
    json::Object snapshot;
    uint64_t snapshotSegmentStart = 0;
    uint64_t snapshotPrefixEnd = 0;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> snapshotBuffer =
        llvm::MemoryBuffer::getFile(resumeSnapshotPath);
    if (snapshotBuffer) {
      StringRef bytes = (*snapshotBuffer)->getBuffer();
      std::string translated;
      if (bytes.starts_with("ORBITRS1")) {
        translated = bytes.str();
        std::memcpy(translated.data(), graph_facts_io::kCompressedFactsMagic, 8);
        bytes = translated;
      } else if (bytes.starts_with(graph_facts_io::kCompressedFactsMagic)) {
        error = "graph facts artifact cannot be used as a resume snapshot";
        return false;
      }
      std::string snapshotText;
      if (!graph_facts_io::decodeGraphFacts(bytes, snapshotText, error))
        return false;
      llvm::Expected<json::Value> parsed =
          json::parse(snapshotText);
      if (!parsed || !parsed->getAsObject()) {
        error = "resume snapshot is invalid JSON";
        if (!parsed)
          llvm::consumeError(parsed.takeError());
        return false;
      }
      snapshot = *parsed->getAsObject();
      if (snapshot.getString("record_type") !=
              std::optional<StringRef>("resume_snapshot") ||
          snapshot.getString("snapshot_schema") !=
              std::optional<StringRef>(kResumeSnapshotSchema)) {
        error = "resume snapshot schema is unsupported";
        return false;
      }
      int64_t segmentStart = 0;
      int64_t prefixEnd = 0;
      if (!parseRequiredInteger(snapshot, "journal_segment_start",
                                segmentStart, error) ||
          !parseRequiredInteger(snapshot, "journal_prefix_end", prefixEnd,
                                error) ||
          segmentStart < 0 || prefixEnd <= segmentStart) {
        error = "resume snapshot journal prefix is invalid";
        return false;
      }
      snapshotSegmentStart = static_cast<uint64_t>(segmentStart);
      snapshotPrefixEnd = static_cast<uint64_t>(prefixEnd);
      StringRef journalText = (*buffer)->getBuffer();
      if (snapshotPrefixEnd > journalText.size() ||
          snapshotSegmentStart >= snapshotPrefixEnd) {
        error = "resume snapshot journal prefix is not present";
        return false;
      }
      loadedSnapshot = true;
      currentCheckpoint = snapshot;
    } else if (snapshotBuffer.getError() !=
               std::make_error_code(std::errc::no_such_file_or_directory)) {
      error = "cannot read resume snapshot " + resumeSnapshotPath + ": " +
              snapshotBuffer.getError().message();
      return false;
    }
    StringRef remaining = (*buffer)->getBuffer();
    if (loadedSnapshot)
      remaining = remaining.substr(snapshotSegmentStart,
                                  snapshotPrefixEnd - snapshotSegmentStart);
    while (!remaining.empty()) {
      size_t newline = remaining.find('\n');
      StringRef line = newline == StringRef::npos
                           ? remaining
                           : remaining.take_front(newline);
      remaining = newline == StringRef::npos
                      ? StringRef()
                      : remaining.drop_front(newline + 1);
      if (line.trim().empty())
        continue;
      llvm::Expected<json::Value> parsed = json::parse(line);
      if (!parsed) {
        error = "resume journal contains invalid JSON";
        llvm::consumeError(parsed.takeError());
        return false;
      }
      json::Object *object = parsed->getAsObject();
      if (!object) {
        error = "resume journal contains a non-object record";
        return false;
      }
      auto type = object->getString("record_type");
      if (!type) {
        error = "resume journal record has no record_type";
        return false;
      }
      if (*type == "journal_header") {
        currentHeader = *object;
        if (!loadedSnapshot)
          currentCheckpoint.reset();
        currentData.clear();
      } else if (*type == "action" || *type == "graph") {
        if (currentHeader.empty()) {
          error = "resume journal data precedes its journal header";
          return false;
        }
        currentData.push_back(line.str());
      } else if (*type == "resume_checkpoint" && !loadedSnapshot) {
        if (currentHeader.empty()) {
          error = "resume checkpoint precedes its journal header";
          return false;
        }
        currentCheckpoint = *object;
      }
    }

    if (loadedSnapshot)
      currentCheckpoint = snapshot;
    if (currentHeader.empty() || !currentCheckpoint) {
      error = "resume journal has no frontier checkpoint; old or interrupted ledger is not resumable";
      return false;
    }
    if (currentHeader.getString("schema") != std::optional<StringRef>(kSchema) ||
        currentHeader.getString("journal_schema") !=
            std::optional<StringRef>("orbit-joint-graph-closure-journal-v1") ||
        currentHeader.getString("resume_schema") !=
            std::optional<StringRef>(kResumeJournalSchema) ||
        currentHeader.getString("rewrite_implementation_contract") !=
            std::optional<StringRef>(kRewriteImplementationContract)) {
      error = "resume journal schema is unsupported";
      return false;
    }
    if (currentHeader.getString("function") !=
            std::optional<StringRef>(functionName) ||
        currentHeader.getString("stage") != std::optional<StringRef>(stageName) ||
        currentHeader.getString("output") !=
            std::optional<StringRef>(outputFile.getValue()) ||
        currentHeader.getString("graph_artifact_directory") !=
            std::optional<StringRef>(graphArtifactDirectory)) {
      error = "resume journal input/output binding differs";
      return false;
    }
    auto sourceText = currentHeader.getString("source_module_text");
    if (!sourceText || *sourceText != originalInputText) {
      error = "resume journal source module text differs";
      return false;
    }
    const json::Object *options = currentHeader.getObject("options");
    if (!options || options->getString("function") !=
                        std::optional<StringRef>(functionName) ||
        options->getString("stage") != std::optional<StringRef>(stageName) ||
        options->getString("replica_factors") !=
            std::optional<StringRef>(replicaFactors.getValue()) ||
        options->getString("tiling_factors") !=
            std::optional<StringRef>(tilingFactors.getValue()) ||
        options->getBoolean("artifact_bytecode") !=
            std::optional<bool>(artifactBytecode.getValue()) ||
        options->getString("facts_compression") !=
            std::optional<StringRef>(factsCompression.getValue()) ||
        options->getString("resume_snapshot_compression") !=
            std::optional<StringRef>(resumeSnapshotCompression.getValue()) ||
        options->getInteger("resume_snapshot_actions") !=
            std::optional<int64_t>(resumeSnapshotActions.getValue())) {
      error = "resume journal pass options differ";
      return false;
    }
    int64_t savedMaxGraphs = 0;
    int64_t savedMaxActions = 0;
    if (!options || !parseRequiredInteger(*options, "max_graphs", savedMaxGraphs,
                                          error) ||
        !parseRequiredInteger(*options, "max_actions", savedMaxActions, error))
      return false;
    if (maxGraphs.getValue() < savedMaxGraphs ||
        maxActions.getValue() < savedMaxActions) {
      error = "resume journal cannot lower its original resource budgets";
      return false;
    }

    const json::Object &checkpoint = *currentCheckpoint;
    if (checkpoint.getString("schema") != std::optional<StringRef>(kSchema) ||
        checkpoint.getString("snapshot_schema") !=
            std::optional<StringRef>(kResumeSnapshotSchema) ||
        checkpoint.getString("resume_schema") !=
            std::optional<StringRef>(kResumeJournalSchema) ||
        checkpoint.getString("rewrite_implementation_contract") !=
            std::optional<StringRef>(kRewriteImplementationContract) ||
        checkpoint.getString("output") !=
            std::optional<StringRef>(outputFile.getValue()) ||
        checkpoint.getString("graph_artifact_directory") !=
            std::optional<StringRef>(graphArtifactDirectory) ||
        checkpoint.getString("source_module_text") !=
            std::optional<StringRef>(originalInputText) ||
        checkpoint.getString("facts_compression") !=
            std::optional<StringRef>(factsCompression.getValue())) {
      error = "resume checkpoint binding differs from current input/options";
      return false;
    }
    int64_t checkpointSegmentStart = 0;
    int64_t checkpointHeaderBytes = 0;
    int64_t checkpointDataBytes = 0;
    int64_t checkpointPrefixEnd = 0;
    if (!parseRequiredInteger(checkpoint, "journal_segment_start",
                              checkpointSegmentStart, error) ||
        !parseRequiredInteger(checkpoint, "journal_header_bytes",
                              checkpointHeaderBytes, error) ||
        !parseRequiredInteger(checkpoint, "journal_data_bytes",
                              checkpointDataBytes, error) ||
        !parseRequiredInteger(checkpoint, "journal_prefix_end",
                              checkpointPrefixEnd, error) ||
        checkpointSegmentStart < 0 || checkpointHeaderBytes <= checkpointSegmentStart ||
        checkpointDataBytes < 0 || checkpointPrefixEnd !=
            checkpointHeaderBytes + checkpointDataBytes ||
        (!loadedSnapshot ||
         static_cast<uint64_t>(checkpointSegmentStart) != snapshotSegmentStart) ||
        (!loadedSnapshot ||
         static_cast<uint64_t>(checkpointPrefixEnd) != snapshotPrefixEnd)) {
      error = "resume snapshot journal binding is inconsistent";
      return false;
    }
    if (auto savedSequence = checkpoint.getInteger("snapshot_sequence"))
      snapshotSequence = static_cast<uint64_t>(std::max<int64_t>(
          0, *savedSequence));
    if (checkpoint.getBoolean("journal_write_failed").value_or(true)) {
      error = "resume journal was itself write-failed and is not trustworthy";
      return false;
    }
    int64_t savedRecordCount = 0;
    if (!parseRequiredInteger(checkpoint, "resume_data_record_count",
                              savedRecordCount, error) ||
        savedRecordCount < 0 ||
        static_cast<size_t>(savedRecordCount) != currentData.size()) {
      error = "resume checkpoint data-record count is inconsistent";
      return false;
    }

    // Rebuild the exact action/graph ledger without applying any materializer.
    // The original data lines are copied into the new journal segment after its
    // header, so final closure output retains stable graph IDs and paths.
    resumeDataRecords = currentData;
    std::string ledgerFirstBlocker;
    auto noteLedgerBlocker = [&](StringRef reason, StringRef fallback) {
      if (ledgerFirstBlocker.empty())
        ledgerFirstBlocker = reason.empty() ? fallback.str() : reason.str();
    };
    for (const std::string &dataLine : currentData) {
      llvm::Expected<json::Value> parsed = json::parse(dataLine);
      if (!parsed) {
        error = "resume journal data record cannot be reparsed";
        llvm::consumeError(parsed.takeError());
        return false;
      }
      json::Object *object = parsed->getAsObject();
      if (!object)
        return false;
      auto type = object->getString("record_type");
      if (*type == "action") {
        auto status = object->getString("status");
        if (!status) {
          error = "resume action record has no status";
          return false;
        }
        if (*status != "legal" && *status != "proven_illegal") {
          if (auto reason = object->getString("reason"))
            noteLedgerBlocker(*reason, "unknown_action_record");
          else
            noteLedgerBlocker("", "unknown_action_record");
        }
        auto actionPath = object->getString("path");
        if (!actionPath || actionPath->empty()) {
          error = "resume action record has no path";
          return false;
        }
        attemptedActionPaths.insert(*actionPath);
        continue;
      }
      auto status = object->getString("status");
      if (*type != "graph")
        continue;
      if (!status) {
        error = "resume graph record has no status";
        return false;
      }
      if (*status != "legal") {
        if (auto reason = object->getString("reason"))
          noteLedgerBlocker(*reason, "unknown_graph_record");
        else
          noteLedgerBlocker("", "unknown_graph_record");
      } else if ((stageLevel() >= 4 &&
                  !object->getBoolean("k_facts_complete").value_or(false)) ||
                 (stageLevel() >= 5 &&
                  (!object->getBoolean("sibling_facts_complete")
                        .value_or(false) ||
                   !object->getBoolean("sibling_materialization_complete")
                        .value_or(false)))) {
        noteLedgerBlocker("source_owned_graph_facts_incomplete",
                          "source_owned_graph_facts_incomplete");
      }
      if (auto graphPath = object->getString("path"))
        recordedGraphPaths.insert(*graphPath);
      if (*status != "legal")
        continue;
      auto structural = object->getObject("structural_key");
      auto graphId = object->getString("graph_id");
      auto mlirPath = object->getString("mlir_path");
      auto factsPath = object->getString("facts_path");
      auto graphPath = object->getString("path");
      auto order = object->getString("order");
      if (!structural || !graphId || !mlirPath || !factsPath || !graphPath ||
          !order || graphId->empty() || mlirPath->empty() || factsPath->empty()) {
        error = "resume graph record is missing artifact binding";
        return false;
      }
      std::string structuralText =
          jsonText(json::Value(json::Object(*structural)));
      if (!seenStructural.insert(structuralText).second)
        continue;
      GraphArtifact artifact{graphId->str(), mlirPath->str(), factsPath->str()};
      firstPathByStructural[structuralText] = graphPath->str();
      artifactByStructural[structuralText] = artifact;
      int64_t graphOrdinal = 0;
      StringRef ordinalText = graphId->drop_front(6);
      if (graphId->starts_with("graph-") &&
          !ordinalText.getAsInteger(10, graphOrdinal))
        nextGraphOrdinal = std::max<uint64_t>(
            nextGraphOrdinal, static_cast<uint64_t>(graphOrdinal + 1));
      ++graphCount;
    }

    auto savedFirstBlocker = checkpoint.getString("first_blocker");
    if (!savedFirstBlocker) {
      error = "resume checkpoint has no first blocker binding";
      return false;
    }
    if (!ledgerFirstBlocker.empty()) {
      if (*savedFirstBlocker != ledgerFirstBlocker) {
        error = "resume checkpoint first blocker differs from the committed "
                "action/graph ledger";
        return false;
      }
      firstBlocker = ledgerFirstBlocker;
    } else if (savedFirstBlocker->empty() ||
               isTransientResourceBlocker(*savedFirstBlocker)) {
      // max-actions/max-graphs is a resumable budget boundary.  A larger
      // resume budget must not inherit it as a permanent semantic blocker.
      firstBlocker.clear();
    } else {
      error = "resume checkpoint first blocker has no committed ledger "
              "witness";
      return false;
    }

    auto readCounter = [&](StringRef key, uint64_t &target) {
      int64_t value = 0;
      if (!parseRequiredInteger(checkpoint, key, value, error) || value < 0)
        return false;
      target = static_cast<uint64_t>(value);
      return true;
    };
    uint64_t savedGraphCount = 0;
    uint64_t savedUniqueStructuralKeys = 0;
    if (!readCounter("action_attempts", actionAttempts) ||
        !readCounter("graph_count", savedGraphCount) ||
        !readCounter("unique_structural_keys", savedUniqueStructuralKeys) ||
        !readCounter("unknown_action_count", unknownCount) ||
        !readCounter("incomplete_facts_count", incompleteFactsCount) ||
        !readCounter("rejected_action_count", rejectedCount) ||
        !readCounter("proven_illegal_action_count", provenIllegalCount) ||
        !readCounter("frontier_states_enqueued", frontierStatesEnqueued) ||
        !readCounter("frontier_memo_hits", frontierMemoHits) ||
        !readCounter("preflight_graph_build_count", preflightGraphBuildCount) ||
        !readCounter("next_graph_ordinal", nextGraphOrdinal) ||
        !readCounter("next_serial", serial))
      return false;
    // The restored prefix is already covered by the sidecar.  Continue the
    // interval from that committed action count instead of immediately
    // rewriting the same frontier before another action completes.
    lastSnapshotActionAttempts = actionAttempts;
    graphCount = savedGraphCount;
    if (savedGraphCount != seenStructural.size() ||
        savedUniqueStructuralKeys != seenStructural.size()) {
      error = "resume checkpoint graph count disagrees with graph ledger";
      return false;
    }

    const json::Array *attempted = checkpoint.getArray("attempted_action_paths");
    if (!attempted) {
      error = "resume checkpoint has no attempted action ledger";
      return false;
    }
    const llvm::StringSet<> recordedActionPaths = attemptedActionPaths;
    for (const json::Value &value : *attempted) {
      auto pathValue = value.getAsString();
      if (!pathValue || pathValue->empty()) {
        error = "resume checkpoint has an invalid attempted action path";
        return false;
      }
      if (!recordedActionPaths.contains(*pathValue)) {
        error = "resume checkpoint attempted action is absent from the action ledger";
        return false;
      }
      attemptedActionPaths.insert(*pathValue);
    }
    if (attemptedActionPaths.size() != recordedActionPaths.size()) {
      error = "resume checkpoint attempted action ledger differs from records";
      return false;
    }

    const json::Array *frontier = checkpoint.getArray("frontier");
    if (!frontier) {
      error = "resume checkpoint has no graph/action frontier";
      return false;
    }
    int64_t savedPendingFrontierCount = 0;
    if (!parseRequiredInteger(checkpoint, "pending_frontier_count",
                              savedPendingFrontierCount, error) ||
        savedPendingFrontierCount < 0 ||
        static_cast<size_t>(savedPendingFrontierCount) != frontier->size()) {
      error = "resume checkpoint frontier count is inconsistent";
      return false;
    }
    const json::Array *memoKeys = checkpoint.getArray("frontier_memo_keys");
    if (!memoKeys) {
      error = "resume checkpoint has no frontier memo state";
      return false;
    }
    for (const json::Value &value : *memoKeys) {
      auto key = value.getAsString();
      if (!key || key->empty() || !frontierMemo.insert(*key).second) {
        error = "resume checkpoint frontier memo state is invalid";
        return false;
      }
    }
    int64_t savedMemoSize = 0;
    if (!parseRequiredInteger(checkpoint, "frontier_memo_size", savedMemoSize,
                              error) ||
        savedMemoSize < 0 ||
        static_cast<size_t>(savedMemoSize) != frontierMemo.size()) {
      error = "resume checkpoint frontier memo size is inconsistent";
      return false;
    }
    for (const json::Value &value : *frontier) {
      const json::Object *entry = value.getAsObject();
      if (!entry) {
        error = "resume checkpoint frontier entry is not an object";
        return false;
      }
      auto structuralText = entry->getString("structural_text");
      auto graphId = entry->getString("graph_id");
      auto mlirPath = entry->getString("mlir_path");
      auto factsPath = entry->getString("facts_path");
      auto pathValue = entry->getString("path");
      auto kindName = entry->getString("kind");
      auto includeReplica = entry->getBoolean("include_replica");
      auto includeTiling = entry->getBoolean("include_tiling");
      auto moduleText = entry->getString("module_text");
      auto graphVariantPresent = entry->getBoolean("graph_variant_present");
      auto graphVariantValue = entry->getString("graph_variant_value");
      const json::Object *lineageObjectValue = entry->getObject("lineage");
      if (!structuralText || !graphId || !mlirPath || !factsPath ||
          !pathValue || !kindName || !includeReplica || !includeTiling ||
          !moduleText || !graphVariantPresent || !graphVariantValue ||
          !lineageObjectValue || structuralText->empty() || pathValue->empty() ||
          (*graphVariantPresent && graphVariantValue->empty())) {
        error = "resume checkpoint frontier entry is incomplete";
        return false;
      }
      auto artifactIt = artifactByStructural.find(structuralText->str());
      if (artifactIt == artifactByStructural.end() ||
          artifactIt->second.graphId != graphId->str() ||
          artifactIt->second.mlirPath != mlirPath->str() ||
          artifactIt->second.factsPath != factsPath->str()) {
        error = "resume frontier artifact is not bound to its graph ledger";
        return false;
      }
      FrontierEntry::Kind kind;
      if (!parseResumeFrontierKind(*kindName, kind)) {
        error = "resume checkpoint frontier has an unknown phase";
        return false;
      }
      TaskLineageState lineage;
      if (!parseLineageObject(*lineageObjectValue, lineage, error))
        return false;
      std::string factsText;
      graph_facts_io::FactsCompression detectedCompression =
          graph_facts_io::FactsCompression::LegacyJson;
      if (!graph_facts_io::readGraphFactsFile(factsPath->str(), factsText,
                                              error, &detectedCompression)) {
        if (error.empty())
          error = "resume frontier facts artifact is missing or unreadable";
        return false;
      }
      if (detectedCompression != selectedFactsCompression()) {
        error = "resume frontier facts storage format differs from the "
                "checkpoint binding";
        return false;
      }
      llvm::Expected<json::Value> factsValue = json::parse(factsText);
      if (!factsValue) {
        error = "resume frontier facts artifact is invalid JSON";
        llvm::consumeError(factsValue.takeError());
        return false;
      }
      json::Object *factsRoot = factsValue->getAsObject();
      json::Object *factsStructural =
          factsRoot ? factsRoot->getObject("structural_key") : nullptr;
      if (!factsStructural ||
          jsonText(json::Value(json::Object(*factsStructural))) !=
              structuralText->str()) {
        error = "resume frontier facts structural key differs";
        return false;
      }
      std::string parseError;
      OwningOpRef<ModuleOp> artifactModule = parseSourceFile<ModuleOp>(
          mlirPath->str(), ParserConfig(module.getContext()));
      if (!artifactModule) {
        error = "resume frontier graph artifact is not parseable";
        return false;
      }
      FailureOr<func::FuncOp> artifactFunction =
          selectTaskFunction(*artifactModule, functionName, parseError);
      auto artifactVariant = failed(artifactFunction)
                                 ? StringAttr()
                                 : (*artifactFunction)->getAttrOfType<StringAttr>(
                                       "amoeba.graph_variant_id");
      if (failed(artifactFunction) || !artifactVariant ||
          artifactVariant.getValue() != graphId->str()) {
        error = "resume frontier graph artifact function binding differs";
        return false;
      }
      // persistGraphArtifact stamps only the independently written artifact
      // clone.  Restore the exact traversal-side attribute state before
      // comparing and memoizing it; the graph id must never leak into the
      // source-owned module used for subsequent rewrites.
      if (*graphVariantPresent)
        (*artifactFunction)->setAttr(
            "amoeba.graph_variant_id",
            StringAttr::get(artifactFunction->getContext(),
                            *graphVariantValue));
      else
        (*artifactFunction)->removeAttr("amoeba.graph_variant_id");
      if (printedModuleText(*artifactModule) != moduleText->str()) {
        error = "resume frontier graph artifact body differs from its exact "
                "traversal witness";
        return false;
      }
      restoredFrontier.push_back(FrontierEntry{
          std::move(artifactModule), std::move(lineage), pathValue->str(), kind,
          *includeReplica, *includeTiling, structuralText->str()});
      frontierLedger.push_back(FrontierCheckpointEntry{
          structuralText->str(), artifactIt->second, pathValue->str(), kind,
          *includeReplica, *includeTiling, moduleText->str(),
          *graphVariantPresent, graphVariantValue->str(),
          restoredFrontier.back().lineage});
      std::string restoredMemoKey = frontierStateKey(
          *structuralText, *restoredFrontier.back().module,
          restoredFrontier.back().lineage, kind, *includeReplica,
          *includeTiling);
      if (!frontierMemo.contains(restoredMemoKey)) {
        error = "resume frontier is absent from the durable memo state";
        return false;
      }
    }
    return true;
  }

  static bool parseFactorAlphabet(StringRef specification,
                                  ArrayRef<int64_t> defaults,
                                  StringRef optionName,
                                  SmallVectorImpl<int64_t> &alphabet,
                                  std::string &error) {
    alphabet.clear();
    if (specification.empty()) {
      alphabet.append(defaults.begin(), defaults.end());
      return true;
    }
    SmallVector<StringRef, 8> fields;
    specification.split(fields, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
    if (fields.empty()) {
      error = optionName.str() +
              " must contain at least one factor";
      return false;
    }
    for (StringRef field : fields) {
      field = field.trim();
      int64_t factor = 0;
      if (field.empty() || field.getAsInteger(10, factor) || factor <= 0) {
        error = optionName.str() +
                " must be a comma-separated list of positive integers from "
                "{1,2,4,8}";
        return false;
      }
      bool allowed = false;
      for (int64_t candidate : kSupportedFactors)
        allowed |= factor == candidate;
      if (!allowed) {
        error = optionName.str() +
                " contains a value outside {1,2,4,8}";
        return false;
      }
      for (int64_t existing : alphabet) {
        if (existing == factor) {
          error = optionName.str() + " must contain distinct values";
          return false;
        }
      }
      alphabet.push_back(factor);
    }
    return !alphabet.empty();
  }

  bool parseAlphabets(std::string &error) {
    if (!parseFactorAlphabet(replicaFactors.getValue(), kDefaultReplicaFactors,
                             "replica-factors", replicaAlphabet, error))
      return false;
    if (!parseFactorAlphabet(tilingFactors.getValue(), kDefaultTilingFactors,
                             "tiling-factors", tilingAlphabet, error))
      return false;
    return true;
  }
  bool canAttempt() {
    if (actionAttempts >= static_cast<uint64_t>(maxActions.getValue())) {
      budgetReached = true;
      if (firstBlocker.empty())
        firstBlocker = "max_actions_reached";
      return false;
    }
    ++actionAttempts;
    if (actionAttempts == static_cast<uint64_t>(maxActions.getValue())) {
      budgetReached = true;
      if (firstBlocker.empty())
        firstBlocker = "max_actions_reached";
    }
    return true;
  }

  void recordAction(const Action &action, StringRef status, StringRef reason,
                    StringRef diagnostic, StringRef order, StringRef path) {
    if (journalWriteFailed || resourceFailure)
      return;
    if (attemptedActionPaths.contains(path))
      return;
    if (!appendJournalRecord(jsonLine(actionObject(
            action, status, reason, diagnostic, order, path))))
      return;
    attemptedActionPaths.insert(path);
    if (resumeSnapshotActions.getValue() > 0 &&
        actionAttempts - lastSnapshotActionAttempts >=
            static_cast<uint64_t>(resumeSnapshotActions.getValue()))
      snapshotRequested = true;
    if (status == "proven_illegal") {
      ++provenIllegalCount;
      ++rejectedCount;
    } else if (status != "legal") {
      ++unknownCount;
      ++rejectedCount;
      noteFirstBlocker(reason);
    }
  }

  void noteFirstBlocker(StringRef reason) {
    if (!reason.empty() &&
        (firstBlocker.empty() || isTransientResourceBlocker(firstBlocker)))
      firstBlocker = reason.str();
  }

  graph_facts_io::FactsCompression selectedFactsCompression() const {
    if (factsCompression.getValue() == "zlib")
      return graph_facts_io::FactsCompression::Zlib;
    if (factsCompression.getValue() == "zstd")
      return graph_facts_io::FactsCompression::Zstd;
    return graph_facts_io::FactsCompression::LegacyJson;
  }

  bool persistGraphArtifact(ModuleOp module, Facts &facts,
                            GraphArtifact &artifact, bool identity) {
    artifact.graphId = identity ? "identity"
                                : "graph-" + std::to_string(nextGraphOrdinal++);
    artifact.mlirPath = graphArtifactDirectory + "/" + artifact.graphId +
                        ".mlir";
    artifact.factsPath = graphArtifactDirectory + "/" + artifact.graphId +
                         ".facts.json";

    std::string error;
    graph_facts_io::FactsCompression compression = selectedFactsCompression();
    SmallVector<uint8_t, 0> storedFacts;
    if (!graph_facts_io::encodeGraphFacts(facts.factsText, compression,
                                          storedFacts, error) ||
        !writeAtomically(
            artifact.factsPath,
            [&](llvm::raw_ostream &stream) {
              stream.write(reinterpret_cast<const char *>(storedFacts.data()),
                           storedFacts.size());
              return true;
            },
            error)) {
      outputWriteFailed = true;
      resourceFailure = true;
      budgetReached = true;
      noteFirstBlocker("graph_facts_artifact_write_failed");
      llvm::sys::fs::remove(facts.factsPath);
      return false;
    }
    llvm::sys::fs::remove(facts.factsPath);

    // Persisted graph artifacts are consumed as independent source-owned
    // variants.  The closure traversal starts from an identity module, so its
    // function attribute still says "identity" even after a legal rewrite.
    // Stamp only a print clone: mutating the traversal module would leak the
    // artifact label into later source-owned materialization and fact checks.
    OwningOpRef<ModuleOp> artifactModule = module.clone();
    std::string functionError;
    FailureOr<func::FuncOp> artifactFunction =
        selectTaskFunction(*artifactModule, functionName, functionError);
    if (failed(artifactFunction)) {
      resourceFailure = true;
      budgetReached = true;
      noteFirstBlocker("graph_variant_artifact_function_binding_failed");
      llvm::sys::fs::remove(artifact.factsPath);
      return false;
    }
    (*artifactFunction)->setAttr(
        "amoeba.graph_variant_id",
        StringAttr::get(artifactFunction->getContext(), artifact.graphId));
    if (!writeAtomically(
            artifact.mlirPath,
            [&](llvm::raw_ostream &stream) {
              // All graph consumers use MLIR's source parser, which detects
              // bytecode by its magic. Keep identity textual for the existing
              // exact input-reuse witness; serialize other complete modules
              // compactly without changing IR, graph identity, or coverage.
              if (artifactBytecode.getValue() && !identity)
                return succeeded(writeBytecodeToFile(
                    artifactModule->getOperation(), stream));
              artifactModule->print(stream);
              stream << "\n";
              return true;
            },
            error)) {
      resourceFailure = true;
      budgetReached = true;
      noteFirstBlocker("graph_mlir_artifact_write_failed");
      llvm::sys::fs::remove(artifact.factsPath);
      return false;
    }
    facts.factsPath = artifact.factsPath;
    return true;
  }

  bool recordGraph(ModuleOp module, StringRef order, StringRef path,
                   std::string *structuralTextOut = nullptr) {
    if (journalWriteFailed || resourceFailure)
      return false;
    if (recordedGraphPaths.contains(path))
      return true;
    Facts facts;
    std::string reason;
    std::string diagnostic;
    if (!extractFacts(module, functionName, outputFile, serial++, facts,
                      reason, diagnostic)) {
      if (reason == "facts_output_resource_failure") {
        resourceFailure = true;
        budgetReached = true;
        noteFirstBlocker(reason);
      }
      ++unknownCount;
      noteFirstBlocker(reason);
      bool appended = appendJournalRecord(jsonLine(json::Object{
          {"schema", kSchema.str()},
          {"record_type", "graph"},
          {"status", "unknown"},
          {"path", path.str()},
          {"order", order.str()},
          {"reason", reason},
          {"diagnostic", diagnostic}}));
      if (appended)
        recordedGraphPaths.insert(path);
      return false;
    }
    if ((stageLevel() >= 4 && !facts.kFactsComplete) ||
        (stageLevel() >= 5 && (!facts.siblingFactsComplete ||
                               !facts.siblingMaterializationComplete))) {
      ++incompleteFactsCount;
      noteFirstBlocker("source_owned_graph_facts_incomplete");
    }
    if (structuralTextOut)
      *structuralTextOut = facts.structuralText;
    auto found = firstPathByStructural.find(facts.structuralText);
    bool duplicate = found != firstPathByStructural.end();
    std::string firstPath = duplicate ? found->second : path.str();
    GraphArtifact artifact;
    if (duplicate) {
      artifact = artifactByStructural.lookup(facts.structuralText);
      llvm::sys::fs::remove(facts.factsPath);
    } else {
      bool identity = order == "identity" && path == "identity";
      if (!persistGraphArtifact(module, facts, artifact, identity)) {
        ++unknownCount;
        bool appended = appendJournalRecord(jsonLine(json::Object{
            {"schema", kSchema.str()},
            {"record_type", "graph"},
            {"status", "unknown"},
            {"path", path.str()},
            {"order", order.str()},
            {"reason", firstBlocker}}));
        if (appended)
          recordedGraphPaths.insert(path);
        return false;
      }
      firstPathByStructural[facts.structuralText] = path.str();
      artifactByStructural[facts.structuralText] = artifact;
      seenStructural.insert(facts.structuralText);
      ++graphCount;
      if (graphCount >= static_cast<uint64_t>(maxGraphs.getValue())) {
        budgetReached = true;
        if (firstBlocker.empty())
          firstBlocker = "max_graphs_reached";
      }
    }
    bool appended = appendJournalRecord(jsonLine(graphObject(
        facts, path, duplicate, firstPath, order, artifact)));
    if (appended)
      recordedGraphPaths.insert(path);
    return appended;
  }

  static void mergeDimensionSelection(DimensionSelection &target,
                                      const DimensionSelection &source) {
    target.mFactor = std::max(target.mFactor, source.mFactor);
    target.nFactor = std::max(target.nFactor, source.nFactor);
    target.kFactor = std::max(target.kFactor, source.kFactor);
    target.replicaFactor = std::max(target.replicaFactor,
                                    source.replicaFactor);
  }

  static LineageState mergeLineages(const LineageState &first,
                                    const LineageState &second) {
    LineageState merged = first;
    for (const auto &entry : second) {
      DimensionSelection &selection = merged[entry.first];
      mergeDimensionSelection(selection, entry.second);
    }
    return merged;
  }

  static void setDimensionSelection(LineageState &lineage,
                                    Action::Kind kind, int64_t axis,
                                    int64_t factor) {
    for (auto &entry : lineage) {
      if (kind == Action::Kind::Tile && axis == 0)
        entry.second.mFactor = factor;
      else if (kind == Action::Kind::Tile && axis == 1)
        entry.second.nFactor = factor;
      else if (kind == Action::Kind::KTile)
        entry.second.kFactor = factor;
      else if (kind == Action::Kind::Replica)
        entry.second.replicaFactor = factor;
    }
  }

  // Rebind logical state to the names emitted by the source-owned
  // materializer.  The materializers erase the selected task and create
  // derived tasks with explicit parent/fusion attributes; using those
  // source-owned links keeps the closure ledger independent of a guessed name
  // hash and lets different M/N dimensions compose on the same lineage.
  static bool rebindLineageState(ModuleOp module, StringRef functionName,
                                 const TaskLineageState &before,
                                 const Action &action,
                                 TaskLineageState &after) {
    after = before;
    LineageState first;
    if (!lineageStateFor(before, action.firstTask, first))
      return false;
    if (action.kind == Action::Kind::ProducerConsumer ||
        action.kind == Action::Kind::Sibling) {
      LineageState second;
      if (!lineageStateFor(before, action.secondTask, second))
        return false;
      LineageState merged = mergeLineages(first, second);
      after.erase(action.firstTask);
      after.erase(action.secondTask);
      std::string fusedName = action.firstTask + ".fuse." + action.secondTask;
      for (TaskflowTaskOp task : taskOps(module, functionName)) {
        if (task.getTaskName() == fusedName) {
          after[fusedName] = std::move(merged);
          return true;
        }
      }
      return false;
    }

    after.erase(action.firstTask);
    LineageState selected = first;
    setDimensionSelection(selected, action.kind, action.axis, action.factor);
    bool rebound = false;
    for (TaskflowTaskOp task : taskOps(module, functionName)) {
      bool child = false;
      if (action.kind == Action::Kind::Tile) {
        auto parent = task->getAttrOfType<StringAttr>(
            "amoeba.neura.tiling.parent_task");
        child = parent && parent.getValue() == action.firstTask;
      } else if (action.kind == Action::Kind::Replica) {
        auto parent =
            task->getAttrOfType<StringAttr>("amoeba.replica.parent_task");
        child = parent && parent.getValue() == action.firstTask;
      } else if (action.kind == Action::Kind::KTile) {
        child = task->hasAttr("amoeba.semantic.k_tiled") &&
                before.find(task.getTaskName().str()) == before.end();
      }
      if (child) {
        after[task.getTaskName().str()] = selected;
        rebound = true;
      }
    }
    return rebound;
  }

  bool trySingle(ModuleOp source, const Action &action, StringRef order,
                 StringRef path, const TaskLineageState &lineage,
                 OwningOpRef<ModuleOp> *result,
                 TaskLineageState *resultLineage,
                 std::string *resultStructuralText = nullptr,
                 JointRewritePreflightContext *preflightContext = nullptr) {
    if (attemptedActionPaths.contains(path))
      return false;
    if (!canAttempt())
      return false;
    LineageState selected;
    if (!lineageStateFor(lineage, action.firstTask, selected) ||
        ((action.kind == Action::Kind::ProducerConsumer ||
          action.kind == Action::Kind::Sibling) &&
         !lineageStateFor(lineage, action.secondTask, selected))) {
      recordAction(action, "unknown", "rewrite_lineage_not_rebindable",
                   "source-owned rewrite action has no tracked input lineage",
                   order, path);
      return false;
    }
    if (action.kind == Action::Kind::Tile ||
        action.kind == Action::Kind::KTile ||
        action.kind == Action::Kind::Replica) {
      TaskflowTaskOp dimensionTask;
      unsigned matches = 0;
      for (TaskflowTaskOp task : taskOps(source, functionName))
        if (task.getTaskName() == action.firstTask) {
          dimensionTask = task;
          ++matches;
        }
      if (matches == 1) {
        ClosureDimensionKind dimensionKind =
            action.kind == Action::Kind::Tile
                ? ClosureDimensionKind::Tile
                : action.kind == Action::Kind::KTile
                      ? ClosureDimensionKind::KTile
                      : ClosureDimensionKind::Replica;
        ClosureDimensionResult localLegality = preflightClosureDimension(
            dimensionTask, dimensionKind, action.axis, action.factor,
            action.kMode);
        if (localLegality.disposition ==
            ClosureDimensionDisposition::ProvenIllegal) {
          recordAction(action, "proven_illegal", localLegality.reason,
                       localLegality.diagnostic, order, path);
          return false;
        }
        if (localLegality.disposition ==
                ClosureDimensionDisposition::Unknown &&
            localLegality.skipMaterializer) {
          recordAction(action, "unknown", localLegality.reason,
                       localLegality.diagnostic, order, path);
          return false;
        }
      }
    }
    JointRewritePreflightKind preflightKind = JointRewritePreflightKind::Other;
    if (action.kind == Action::Kind::ProducerConsumer)
      preflightKind = JointRewritePreflightKind::ProducerConsumer;
    else if (action.kind == Action::Kind::Sibling)
      preflightKind = JointRewritePreflightKind::Sibling;
    if (preflightKind != JointRewritePreflightKind::Other) {
      std::optional<JointRewritePreflightContext> ephemeralContext;
      JointRewritePreflightContext *context = preflightContext;
      if (!context || !context->isFor(source)) {
        ephemeralContext.emplace(source, &preflightGraphBuildCount);
        context = &*ephemeralContext;
      }
      JointRewritePreflightResult preflight = context->evaluate(
          preflightKind, action.firstTask, action.secondTask);
      if (preflight.disposition ==
          JointRewritePreflightDisposition::ProvenIllegal) {
        recordAction(action, "proven_illegal", preflight.reason,
                     "exact typed data dependency and original storage proof",
                     order, path);
        return false;
      }
    }
    // An incomplete preflight never discards a possibly legal rewrite. The
    // source-owned materializer and re-extracted facts remain authoritative.
    OwningOpRef<ModuleOp> trial = source.clone();
    std::string reason;
    std::string diagnostic;
    bool success = runMaterializer(*trial, functionName, action, reason,
                                   diagnostic);
    if (!success) {
      recordAction(action, "unknown", reason, diagnostic, order, path);
      return false;
    }
    if (action.kind == Action::Kind::ProducerConsumer ||
        action.kind == Action::Kind::Sibling) {
      size_t sourceTaskCount = taskOps(source, functionName).size();
      size_t trialTaskCount = taskOps(*trial, functionName).size();
      if (trialTaskCount >= sourceTaskCount) {
        recordAction(action, "unknown", "fusion_did_not_reduce_task_count",
                     "source-owned fusion did not reduce the task count",
                     order, path);
        return false;
      }
    }
    TaskLineageState nextLineage;
    if (!rebindLineageState(*trial, functionName, lineage, action,
                            nextLineage)) {
      recordAction(action, "unknown", "rewrite_lineage_not_rebindable",
                   "source-owned rewrite produced no attributable child",
                   order, path);
      return false;
    }
    std::string structuralText;
    if (!recordGraph(*trial, order, path, &structuralText)) {
      recordAction(action, "unknown", "facts_reextract_failed", diagnostic,
                   order, path);
      return false;
    }
    recordAction(action, "legal", "", diagnostic, order, path);
    if (result)
      *result = std::move(trial);
    if (resultLineage)
      *resultLineage = std::move(nextLineage);
    if (resultStructuralText)
      *resultStructuralText = std::move(structuralText);
    return true;
  }

  static SmallVector<TaskflowTaskOp> taskOps(ModuleOp module,
                                              StringRef functionName) {
    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName, error);
    if (failed(selected))
      return {};
    SmallVector<TaskflowTaskOp> tasks;
    selected->walk([&](TaskflowTaskOp task) { tasks.push_back(task); });
    return tasks;
  }

  static unsigned outputRank(TaskflowTaskOp task) {
    if (task.getWillWrites().empty())
      return 0;
    auto type = dyn_cast<MemRefType>(task.getWillWrites().front().getType());
    return type ? type.getRank() : 0;
  }

  SmallVector<Action> tileActions(ModuleOp module,
                                  const TaskLineageState &lineage) const {
    SmallVector<Action> actions;
    for (TaskflowTaskOp taskOp : taskOps(module, functionName)) {
      const std::string task = taskOp.getTaskName().str();
      LineageState selected;
      bool lineageKnown = lineageStateFor(lineage, task, selected);
      for (int64_t factor : tilingAlphabet) {
        if (factor == 1)
          continue;
        if (!lineageKnown ||
            allLineagesFree(selected, &DimensionSelection::mFactor))
          actions.push_back(Action{Action::Kind::Tile, task, "", 0, factor,
                                   ""});
        if (!lineageKnown ||
            allLineagesFree(selected, &DimensionSelection::nFactor))
          actions.push_back(Action{Action::Kind::Tile, task, "", 1, factor,
                                   ""});
        if (!lineageKnown ||
            allLineagesFree(selected, &DimensionSelection::kFactor))
          for (StringRef kMode : {"sequential", "parallel-linear",
                                  "parallel-tree"})
            actions.push_back(Action{Action::Kind::KTile, task, "", -1,
                                     factor, "", kMode.str()});
      }
    }
    return actions;
  }

  SmallVector<Action> replicaActions(ModuleOp module,
                                     const TaskLineageState &lineage) const {
    SmallVector<Action> actions;
    for (TaskflowTaskOp taskOp : taskOps(module, functionName)) {
      const std::string task = taskOp.getTaskName().str();
      LineageState selected;
      bool lineageKnown = lineageStateFor(lineage, task, selected);
      if (lineageKnown &&
          !allLineagesFree(selected, &DimensionSelection::replicaFactor))
        continue;
      unsigned rank = outputRank(taskOp);
      // A non-memref task has no provable legal axis.  Probe axis 0 once so
      // the source-owned pass records its unsupported proof rather than
      // silently shrinking the declared action space.
      unsigned axisCount = rank == 0 ? 1 : rank;
      for (unsigned axis = 0; axis < axisCount; ++axis)
        for (int64_t factor : replicaAlphabet) {
          if (factor == 1)
            continue;
          actions.push_back(Action{Action::Kind::Replica, task, "", axis,
                                   factor, ""});
        }
    }
    return actions;
  }

  SmallVector<Action> fusionActions(ModuleOp module) const {
    SmallVector<Action> actions;
    SmallVector<std::string> tasks;
    // Include fused groups.  Source-owned fusion either rejects an
    // incompatible pair (recorded as unknown) or replaces two tasks with one;
    // the task-count guard in trySingle makes the recursive closure finite.
    for (TaskflowTaskOp task : taskOps(module, functionName))
      tasks.push_back(task.getTaskName().str());
    for (const std::string &first : tasks) {
      for (const std::string &second : tasks) {
        if (first == second)
          continue;
        actions.push_back(Action{Action::Kind::ProducerConsumer, first, second,
                                 -1, 1, "producer-consumer-retained"});
        actions.push_back(Action{Action::Kind::ProducerConsumer, first, second,
                                 -1, 1, "producer-consumer-forwarded"});
      }
    }
    for (size_t first = 0; first < tasks.size(); ++first)
      for (size_t second = first + 1; second < tasks.size(); ++second)
        actions.push_back(Action{Action::Kind::Sibling, tasks[first],
                                 tasks[second], -1, 1, "sibling"});
    return actions;
  }

  SmallVector<Action> dimensionActions(ModuleOp source,
                                       const TaskLineageState &lineage,
                                       bool includeReplica,
                                       bool includeTiling) const {
    SmallVector<Action> actions;
    if (includeReplica)
      actions.append(replicaActions(source, lineage));
    if (includeTiling)
      actions.append(tileActions(source, lineage));
    return actions;
  }

  static StringRef frontierKindName(FrontierEntry::Kind kind) {
    switch (kind) {
    case FrontierEntry::Kind::Dimensions:
      return "dimensions";
    case FrontierEntry::Kind::FusionOnly:
      return "fusion_only";
    case FrontierEntry::Kind::FusionPrefix:
      return "fusion_prefix";
    case FrontierEntry::Kind::DimensionSuffix:
      return "dimension_suffix";
    }
    return "unknown";
  }

  // structuralText is the source-owned facts key.  The complete printed
  // module is included as an exact equality guard because a structural key
  // alone does not establish equivalence for every future materializer input
  // or metadata attribute.  The key is stored directly in StringSet; no
  // cryptographic identity or digest is used.
  std::string frontierStateKey(StringRef structuralText, ModuleOp module,
                               const TaskLineageState &lineage,
                               FrontierEntry::Kind kind,
                               bool includeReplica,
                               bool includeTiling) const {
    std::string key = "structural=";
    appendKeyPart(key, structuralText);
    key += "module=";
    appendKeyPart(key, printedModuleText(module));
    key += "lineage=";
    appendKeyPart(key, serializeLineageState(lineage));
    key += "phase=";
    appendKeyPart(key, frontierKindName(kind));
    key += "include-replica=";
    appendKeyPart(key, includeReplica ? "1" : "0");
    key += "include-tiling=";
    appendKeyPart(key, includeTiling ? "1" : "0");
    return key;
  }

  bool capturePendingFrontierEntry(const FrontierEntry &entry) {
    GraphArtifact artifact = artifactByStructural.lookup(entry.structuralText);
    if (artifact.mlirPath.empty() || artifact.factsPath.empty()) {
      resourceFailure = true;
      budgetReached = true;
      if (firstBlocker.empty())
        firstBlocker = "frontier_graph_artifact_binding_missing";
      return false;
    }
    std::string functionError;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(*entry.module, functionName, functionError);
    if (failed(selected)) {
      resourceFailure = true;
      budgetReached = true;
      if (firstBlocker.empty())
        firstBlocker = "frontier_graph_artifact_function_binding_failed";
      return false;
    }
    StringAttr graphVariant =
        (*selected)->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
    frontierLedger.push_back(FrontierCheckpointEntry{
        entry.structuralText,
        artifact,
        entry.path,
        entry.kind,
        entry.includeReplica,
        entry.includeTiling,
        printedModuleText(*entry.module),
        static_cast<bool>(graphVariant),
        graphVariant ? graphVariant.getValue().str() : std::string(),
        entry.lineage});
    return true;
  }

  void clearPendingFrontierLedger() { frontierLedger.clear(); }

  bool maybeWritePendingSnapshot(SmallVector<FrontierEntry> &frontier,
                                 size_t firstPending,
                                 SmallVector<FrontierEntry> &next) {
    if (!snapshotRequested)
      return true;
    clearPendingFrontierLedger();
    for (size_t index = firstPending; index < frontier.size(); ++index)
      if (!capturePendingFrontierEntry(frontier[index]))
        return false;
    for (FrontierEntry &entry : next)
      if (!capturePendingFrontierEntry(entry))
        return false;
    return writeResumeSnapshot();
  }

  bool enqueueFrontier(SmallVector<FrontierEntry> &destination,
                       OwningOpRef<ModuleOp> module,
                       TaskLineageState lineage, StringRef path,
                       FrontierEntry::Kind kind, bool includeReplica,
                       bool includeTiling, StringRef structuralText) {
    std::string key = frontierStateKey(structuralText, *module, lineage, kind,
                                       includeReplica, includeTiling);
    if (!frontierMemo.insert(key).second) {
      ++frontierMemoHits;
      return false;
    }
    ++frontierStatesEnqueued;
    GraphArtifact artifact = artifactByStructural.lookup(structuralText);
    if (artifact.mlirPath.empty() || artifact.factsPath.empty()) {
      resourceFailure = true;
      budgetReached = true;
      if (firstBlocker.empty())
        firstBlocker = "frontier_graph_artifact_binding_missing";
      return false;
    }
    destination.push_back(FrontierEntry{
        std::move(module), std::move(lineage), path.str(), kind,
        includeReplica, includeTiling, structuralText.str()});
    return true;
  }

  // One breadth-first frontier covers both approved rewrite orders.  Root
  // dimension actions and root fusion actions are attempted before any child
  // state is expanded.  The exact frontier memo is applied only after the
  // action and graph ledger have been recorded.
  void exploreBreadthFirst(ModuleOp source, bool includeReplica,
                           bool includeTiling,
                           const TaskLineageState &lineage) {
    SmallVector<FrontierEntry> frontier;
    JointRewritePreflightContext rootPreflight(
        source, &preflightGraphBuildCount);

    // A resumed checkpoint is the queue prefix that was already enqueued by
    // the interrupted traversal.  Put it before newly regenerated root
    // actions: those actions were later in the original BFS queue and must
    // not receive earlier graph ordinals after a resume.
    for (FrontierEntry &entry : restoredFrontier)
      frontier.push_back(std::move(entry));
    restoredFrontier.clear();
    clearPendingFrontierLedger();

    for (const Action &action :
         dimensionActions(source, lineage, includeReplica, includeTiling)) {
      if (budgetReached)
        break;
      std::string path = "identity/" + actionText(action);
      OwningOpRef<ModuleOp> intermediate;
      TaskLineageState intermediateLineage;
      std::string intermediateStructuralText;
      if (!trySingle(source, action, "single", path, lineage, &intermediate,
                     &intermediateLineage, &intermediateStructuralText,
                     &rootPreflight))
        continue;
      enqueueFrontier(frontier, std::move(intermediate),
                      std::move(intermediateLineage), path,
                      FrontierEntry::Kind::Dimensions, includeReplica,
                      includeTiling, intermediateStructuralText);
    }

    // Keep the two root action spaces in the same depth-1 frontier.  A
    // max-actions/max-graphs interruption is still a resource interruption;
    // it never changes an unattempted root action into a legal rejection.
    if (stageLevel() >= 5) {
      for (const Action &action : fusionActions(source)) {
        if (budgetReached)
          break;
        std::string path = "identity/" + actionText(action);
        OwningOpRef<ModuleOp> intermediate;
        TaskLineageState intermediateLineage;
        std::string intermediateStructuralText;
        if (!trySingle(source, action, "fuse_then_tile", path, lineage,
                       &intermediate, &intermediateLineage,
                       &intermediateStructuralText, &rootPreflight))
          continue;
        enqueueFrontier(frontier, std::move(intermediate),
                        std::move(intermediateLineage), path,
                        FrontierEntry::Kind::FusionPrefix,
                        /*includeReplica=*/true, /*includeTiling=*/true,
                        intermediateStructuralText);
      }
    }

    SmallVector<FrontierEntry> noNextFrontier;
    if (!maybeWritePendingSnapshot(frontier, /*firstPending=*/0,
                                   noNextFrontier))
      return;
    if (budgetReached) {
      SmallVector<FrontierEntry *> pending;
      pending.reserve(frontier.size());
      for (FrontierEntry &entry : frontier)
        pending.push_back(&entry);
      clearPendingFrontierLedger();
      for (FrontierEntry *entry : pending)
        if (!capturePendingFrontierEntry(*entry))
          break;
      return;
    }

    while (!budgetReached && !frontier.empty()) {
      SmallVector<FrontierEntry> next;
      size_t interruptedIndex = frontier.size();
      for (size_t frontierIndex = 0; frontierIndex < frontier.size();
           ++frontierIndex) {
        if (budgetReached) {
          interruptedIndex = frontierIndex;
          break;
        }
        FrontierEntry &entry = frontier[frontierIndex];
        JointRewritePreflightContext entryPreflight(
            *entry.module, &preflightGraphBuildCount);

        if (entry.kind == FrontierEntry::Kind::Dimensions) {
          // The old tile -> fuse recursion offered fusion before another
          // dimension from the same state.  Preserve that local order while
          // advancing every state in this layer together.
          if (stageLevel() >= 5) {
            for (const Action &action : fusionActions(*entry.module)) {
              if (budgetReached)
                break;
              std::string path = entry.path + "/" + actionText(action);
              OwningOpRef<ModuleOp> intermediate;
              TaskLineageState intermediateLineage;
              std::string intermediateStructuralText;
              if (!trySingle(*entry.module, action, "tile_then_fuse", path,
                             entry.lineage, &intermediate,
                             &intermediateLineage,
                             &intermediateStructuralText, &entryPreflight))
                continue;
              enqueueFrontier(next, std::move(intermediate),
                              std::move(intermediateLineage), path,
                              FrontierEntry::Kind::FusionOnly,
                              /*includeReplica=*/false,
                              /*includeTiling=*/false,
                              intermediateStructuralText);
            }
          }
          if (budgetReached)
            break;
          for (const Action &action : dimensionActions(
                   *entry.module, entry.lineage, entry.includeReplica,
                   entry.includeTiling)) {
            if (budgetReached)
              break;
            std::string path = entry.path + "/" + actionText(action);
            OwningOpRef<ModuleOp> intermediate;
            TaskLineageState intermediateLineage;
            std::string intermediateStructuralText;
            if (!trySingle(*entry.module, action, "single", path,
                           entry.lineage, &intermediate, &intermediateLineage,
                           &intermediateStructuralText, &entryPreflight))
              continue;
            enqueueFrontier(next, std::move(intermediate),
                            std::move(intermediateLineage), path,
                            FrontierEntry::Kind::Dimensions,
                            entry.includeReplica, entry.includeTiling,
                            intermediateStructuralText);
          }
          if (!maybeWritePendingSnapshot(frontier, frontierIndex, next)) {
            interruptedIndex = frontierIndex;
            break;
          }
          if (budgetReached) {
            interruptedIndex = frontierIndex;
            break;
          }
          continue;
        }

        if (entry.kind == FrontierEntry::Kind::FusionOnly) {
          for (const Action &action : fusionActions(*entry.module)) {
            if (budgetReached)
              break;
            std::string path = entry.path + "/" + actionText(action);
            OwningOpRef<ModuleOp> intermediate;
            TaskLineageState intermediateLineage;
            std::string intermediateStructuralText;
            if (!trySingle(*entry.module, action, "tile_then_fuse", path,
                           entry.lineage, &intermediate,
                           &intermediateLineage,
                           &intermediateStructuralText, &entryPreflight))
              continue;
            enqueueFrontier(next, std::move(intermediate),
                            std::move(intermediateLineage), path,
                            FrontierEntry::Kind::FusionOnly,
                            /*includeReplica=*/false,
                            /*includeTiling=*/false,
                            intermediateStructuralText);
          }
          if (!maybeWritePendingSnapshot(frontier, frontierIndex, next)) {
            interruptedIndex = frontierIndex;
            break;
          }
          if (budgetReached) {
            interruptedIndex = frontierIndex;
            break;
          }
          continue;
        }

        // Fuse -> tile keeps dimension suffixes and further fusion prefixes
        // as separate phases.  Dimensions are offered before another fusion
        // prefix, matching the recursive source traversal.
        if (entry.kind == FrontierEntry::Kind::FusionPrefix) {
          for (const Action &action : dimensionActions(
                   *entry.module, entry.lineage, /*includeReplica=*/true,
                   /*includeTiling=*/true)) {
            if (budgetReached)
              break;
            std::string path = entry.path + "/" + actionText(action);
            OwningOpRef<ModuleOp> intermediate;
            TaskLineageState intermediateLineage;
            std::string intermediateStructuralText;
            if (!trySingle(*entry.module, action, "fuse_then_tile", path,
                           entry.lineage, &intermediate,
                           &intermediateLineage,
                           &intermediateStructuralText, &entryPreflight))
              continue;
            enqueueFrontier(next, std::move(intermediate),
                            std::move(intermediateLineage), path,
                            FrontierEntry::Kind::DimensionSuffix,
                            /*includeReplica=*/true,
                            /*includeTiling=*/true,
                            intermediateStructuralText);
          }
          if (budgetReached)
            break;
          for (const Action &action : fusionActions(*entry.module)) {
            if (budgetReached)
              break;
            std::string path = entry.path + "/" + actionText(action);
            OwningOpRef<ModuleOp> intermediate;
            TaskLineageState intermediateLineage;
            std::string intermediateStructuralText;
            if (!trySingle(*entry.module, action, "fuse_then_tile", path,
                           entry.lineage, &intermediate,
                           &intermediateLineage,
                           &intermediateStructuralText, &entryPreflight))
              continue;
            enqueueFrontier(next, std::move(intermediate),
                            std::move(intermediateLineage), path,
                            FrontierEntry::Kind::FusionPrefix,
                            /*includeReplica=*/true,
                            /*includeTiling=*/true,
                            intermediateStructuralText);
          }
          if (!maybeWritePendingSnapshot(frontier, frontierIndex, next)) {
            interruptedIndex = frontierIndex;
            break;
          }
          if (budgetReached) {
            interruptedIndex = frontierIndex;
            break;
          }
          continue;
        }

        // DimensionSuffix is terminal with respect to fusion: this preserves
        // the requested fuse -> tile order without introducing tile -> fuse.
        for (const Action &action : dimensionActions(
                 *entry.module, entry.lineage, /*includeReplica=*/true,
                 /*includeTiling=*/true)) {
          if (budgetReached)
            break;
          std::string path = entry.path + "/" + actionText(action);
          OwningOpRef<ModuleOp> intermediate;
          TaskLineageState intermediateLineage;
          std::string intermediateStructuralText;
          if (!trySingle(*entry.module, action, "fuse_then_tile", path,
                         entry.lineage, &intermediate, &intermediateLineage,
                         &intermediateStructuralText, &entryPreflight))
            continue;
          enqueueFrontier(next, std::move(intermediate),
                          std::move(intermediateLineage), path,
                          FrontierEntry::Kind::DimensionSuffix,
                          /*includeReplica=*/true, /*includeTiling=*/true,
                          intermediateStructuralText);
        }
        if (!maybeWritePendingSnapshot(frontier, frontierIndex, next)) {
          interruptedIndex = frontierIndex;
          break;
        }
        if (budgetReached) {
          interruptedIndex = frontierIndex;
          break;
        }
      }
      if (budgetReached) {
        if (interruptedIndex == frontier.size())
          interruptedIndex = frontier.empty() ? 0 : frontier.size() - 1;
        clearPendingFrontierLedger();
        for (size_t index = interruptedIndex; index < frontier.size();
             ++index)
          if (!capturePendingFrontierEntry(frontier[index]))
            break;
        for (FrontierEntry &entry : next)
          if (!capturePendingFrontierEntry(entry))
            break;
        return;
      }
      frontier = std::move(next);
    }
    // A non-resource exit has exhausted the frontier.  Do not let a prior
    // periodic snapshot's pending queue masquerade as the current resume
    // boundary when an unknown action later makes the closure incomplete.
    if (!budgetReached)
      clearPendingFrontierLedger();
  }

  // Enumerate the Cartesian product of the replica and M/N/K tiling choices
  // by applying each source-owned materializer to the current cloned module.
  // Each current task contributes at most one action of each dimension;
  // unsupported replica/tile composition remains visible as an unknown
  // materialization record rather than being called illegal by this
  // controller.
  void exploreDimensions(ModuleOp source, StringRef order,
                         StringRef pathPrefix, bool includeReplica,
                         bool includeTiling, bool allowFusionAfter,
                         bool hasDimensionAction,
                         const TaskLineageState &lineage) {
    if (budgetReached)
      return;

    // Every non-identity dimension state is a fusion input for the approved
    // tile -> fuse order.  The identity state is handled by the fusion-first
    // traversal below, avoiding duplicate direct-fusion records.
    if (allowFusionAfter && hasDimensionAction && stageLevel() >= 5)
      exploreFusionClosure(source, "tile_then_fuse", pathPrefix, lineage);

    for (const Action &action :
         dimensionActions(source, lineage, includeReplica, includeTiling)) {
      if (budgetReached)
        return;
      std::string path = pathPrefix.str() + "/" + actionText(action);
      OwningOpRef<ModuleOp> intermediate;
      TaskLineageState intermediateLineage;
      if (!trySingle(source, action, order, path, lineage, &intermediate,
                     &intermediateLineage))
        continue;
      exploreDimensions(*intermediate, order, path, includeReplica,
                        includeTiling, allowFusionAfter, true,
                        intermediateLineage);
    }
  }

  // Enumerate all compatible fusion combinations after a complete
  // replica/tiling prefix.  Fused groups remain eligible for another
  // source-owned fusion attempt; a legal fusion replaces two tasks with one,
  // and trySingle enforces that task-count decrease before recursing.
  void exploreFusionClosure(ModuleOp source, StringRef order,
                            StringRef pathPrefix,
                            const TaskLineageState &lineage) {
    for (const Action &action : fusionActions(source)) {
      if (budgetReached)
        return;
      std::string path = pathPrefix.str() + "/" + actionText(action);
      OwningOpRef<ModuleOp> intermediate;
      TaskLineageState intermediateLineage;
      if (!trySingle(source, action, order, path, lineage, &intermediate,
                     &intermediateLineage))
        continue;
      exploreFusionClosure(*intermediate, order, path, intermediateLineage);
    }
  }

  // The approved fuse -> tile traversal first materializes every compatible
  // fusion prefix, then re-enumerates all replica and M/N/K choices from that
  // canonical intermediate module.  Recursing over all remaining task pairs,
  // including fused groups, preserves compatible multi-fusion candidates
  // without inventing a third rewrite order.
  void exploreFusionThenDimensions(ModuleOp source, StringRef order,
                                   StringRef pathPrefix,
                                   const TaskLineageState &lineage) {
    for (const Action &action : fusionActions(source)) {
      if (budgetReached)
        return;
      std::string path = pathPrefix.str() + "/" + actionText(action);
      OwningOpRef<ModuleOp> intermediate;
      TaskLineageState intermediateLineage;
      if (!trySingle(source, action, order, path, lineage, &intermediate,
                     &intermediateLineage))
        continue;
      exploreDimensions(*intermediate, order, path, /*includeReplica=*/true,
                        /*includeTiling=*/true,
                        /*allowFusionAfter=*/false,
                        /*hasDimensionAction=*/false, intermediateLineage);
      exploreFusionThenDimensions(*intermediate, order, path,
                                  intermediateLineage);
    }
  }

  void writeOutput(ModuleOp module, bool complete) {
    // A resource/interruption checkpoint is appended before attempting the
    // final publication.  It is outside the committed record prefix, so the
    // final closure retains its exact header/action/graph/footer format while
    // the sidecar remains an honest durable ledger for a failed run.
    if (!complete || resourceFailure || journalWriteFailed)
      appendDurableCheckpoint();

    std::string error;
    if (!writeAtomically(
            outputFile,
            [&](llvm::raw_ostream &stream) {
              stream << jsonLine(closureHeaderObject(complete));
              if (!streamJournalRecords(stream, error))
                return false;
              stream << jsonLine(closureFooterObject(complete));
              return true;
            },
            error)) {
      resourceFailure = true;
      budgetReached = true;
      firstBlocker = "closure_output_write_failed";
      if (error.empty())
        error = "atomic closure output publication failed";
      // The first checkpoint may have been written before the final output
      // failed.  Append a second one with the publication error so recovery
      // does not mistake the durable prefix for a complete closure.
      journalError = error;
      appendDurableCheckpoint();
      module.emitError() << "cannot write graph closure output: " << error;
      signalPassFailure();
    }
  }

public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (outputFile.empty() || functionName.empty() || stageLevel() == 0 ||
        maxGraphs.getValue() <= 0 || maxActions.getValue() <= 0 ||
        resumeSnapshotActions.getValue() <= 0) {
      module.emitError("graph closure requires a known stage, function, output, "
                       "positive max-graphs/max-actions, and a positive "
                       "resume-snapshot-actions interval");
      return signalPassFailure();
    }
    if (factsCompression.getValue() != "legacy-json" &&
        factsCompression.getValue() != "zlib" &&
        factsCompression.getValue() != "zstd") {
      module.emitError("facts-compression must be legacy-json, zlib, or zstd");
      return signalPassFailure();
    }
    if (resumeSnapshotCompression.getValue() != "legacy-json" &&
        resumeSnapshotCompression.getValue() != "zstd") {
      module.emitError("resume-snapshot-compression must be legacy-json or zstd");
      return signalPassFailure();
    }
    std::string alphabetError;
    if (!parseAlphabets(alphabetError)) {
      module.emitError() << "invalid factor alphabet: " << alphabetError;
      return signalPassFailure();
    }
    llvm::SmallString<256> parent(outputFile);
    llvm::sys::path::remove_filename(parent);
    outputDirectory = parent.str().str();
    if (outputDirectory.empty())
      outputDirectory = ".";
    std::error_code ec = llvm::sys::fs::create_directories(outputDirectory);
    if (ec) {
      module.emitError() << "cannot create graph closure output directory: "
                         << ec.message();
      return signalPassFailure();
    }
    graphArtifactDirectory = outputFile.getValue() + ".graphs";
    ec = llvm::sys::fs::create_directories(graphArtifactDirectory);
    if (ec) {
      module.emitError() << "cannot create graph artifact directory: "
                         << ec.message();
      return signalPassFailure();
    }
    originalInputText = printedModuleText(module);
    resumeSnapshotPath = outputFile.getValue() + ".resume.snapshot.json";
    std::string journalError;
    if (!resumeJournal.getValue().empty() &&
        !restoreResumeJournal(module, journalError)) {
      module.emitError() << "cannot resume graph closure journal: "
                         << journalError;
      return signalPassFailure();
    }
    if (!initializeJournal(journalError)) {
      module.emitError() << "cannot initialize durable graph closure journal: "
                         << journalError;
      return signalPassFailure();
    }
    if (resuming) {
      for (const std::string &record : resumeDataRecords)
        if (!appendJournalRecord(record + "\n")) {
          module.emitError()
              << "cannot copy committed resume records into the new journal";
          return signalPassFailure();
        }
      resumeDataRecords.clear();
    }

    // The identity/no-op is a real graph candidate and is always retained.
    if (!recordGraph(module, "identity", "identity")) {
      writeOutput(module, false);
      return signalPassFailure();
    }
    if (budgetReached) {
      writeOutput(module, false);
      return;
    }

    // No-op is an explicit choice for every enabled dimension.  These rows
    // are semantic controls and do not consume the materializer budget.
    SmallVector<std::string> rootTasks = taskNames(module, functionName);
    for (const std::string &task : rootTasks) {
      for (int64_t axis = 0; axis < 2; ++axis) {
        if (stageLevel() >= 4) {
          Action noOp{Action::Kind::Tile, task, "", axis, 1, ""};
          std::string noOpPath = "identity/" + actionText(noOp);
          recordAction(noOp, "legal", "identity_no_op", "", "identity",
                       noOpPath);
          recordGraph(module, "identity", noOpPath);
        }
        if (stageLevel() >= 3) {
          Action replicaNoOp{Action::Kind::Replica, task, "", axis, 1, ""};
          std::string replicaNoOpPath =
              "identity/" + actionText(replicaNoOp);
          recordAction(replicaNoOp, "legal", "identity_no_op", "",
                       "identity", replicaNoOpPath);
          recordGraph(module, "identity", replicaNoOpPath);
        }
      }
      if (stageLevel() >= 4) {
        Action kNoOp{Action::Kind::KTile, task, "", -1, 1, ""};
        std::string kNoOpPath = "identity/" + actionText(kNoOp);
        recordAction(kNoOp, "legal", "identity_no_op", "", "identity",
                     kNoOpPath);
        recordGraph(module, "identity", kNoOpPath);
      }
    }
    if (stageLevel() >= 5) {
      Action fusionNoOp{Action::Kind::ProducerConsumer, "", "", -1, 1,
                        "none"};
      recordAction(fusionNoOp, "legal", "identity_no_op", "", "identity",
                   "identity/producer-consumer-fusion-none");
      recordGraph(module, "identity", "identity/producer-consumer-fusion-none");
      Action siblingNoOp{Action::Kind::Sibling, "", "", -1, 1, "none"};
      recordAction(siblingNoOp, "legal", "identity_no_op", "", "identity",
                   "identity/sibling-fusion-none");
      recordGraph(module, "identity", "identity/sibling-fusion-none");
    }

    TaskLineageState initialLineage =
        initialTaskLineageState(module, functionName);
    if (stageLevel() >= 3 && !budgetReached)
      exploreBreadthFirst(module, /*includeReplica=*/true,
                          /*includeTiling=*/stageLevel() >= 4,
                          initialLineage);

    // No arbitrary action-count or changed-task restriction is used to claim
    // legality.  A complete footer is emitted only when every generated
    // source-owned action and graph fact was handled without an unknown
    // materialization, incomplete fact set, or resource interruption.
    // Completion describes the C++ closure itself.  Stage 1/2 deliberately
    // retain the documented single-location/temporal coverage limitation;
    // that limitation is carried by the header and
    // requires_downstream_4x4_schedule, rather than falsely making an
    // otherwise exhaustive identity closure incomplete.
    bool complete = !budgetReached && unknownCount == 0 &&
                    incompleteFactsCount == 0;
    if (!complete && firstBlocker.empty())
      firstBlocker = "joint_candidate_space_not_proven_complete";
    writeOutput(module, complete);
    if (resourceFailure && !outputWriteFailed) {
      module.emitError() << "graph closure resource failure: "
                         << (this->journalError.empty() ? firstBlocker
                                                       : this->journalError);
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createEnumerateJointGraphClosurePass() {
  return std::make_unique<EnumerateJointGraphClosurePass>();
}
} // namespace mlir::amoeba::neura
