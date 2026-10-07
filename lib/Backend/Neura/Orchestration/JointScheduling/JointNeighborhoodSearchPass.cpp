//===- JointNeighborhoodSearchPass.cpp ----------------------*- C++ -*-===//
//
// A bounded, source-owned complete-program neighborhood search.  This pass
// deliberately keeps graph materialization and production scheduling at the
// existing C++ seams: the search itself only chooses a finite action, clones
// the resulting module, obtains source-owned structural facts, and invokes
// the production scheduler once for the resulting whole program.
//
// The pass is intentionally best-found.  Its output never claims exhaustive
// coverage or global optimality.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalMLPInference.h"
#include "AnalyticalTaskCostCatalog.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "Backend/Neura/Orchestration/JointScheduling/InterTaskNetwork.h"
#include "Backend/Neura/Orchestration/JointScheduling/ProveStaticActiveTransferShapesPass.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"
#include "Backend/Neura/Orchestration/JointScheduling/SourceIterationDomainPartitionProof.h"
#include "JointNeighborhoodActions.h"
#include "NeighborhoodReplaySelection.h"
#include "Backend/Neura/Orchestration/JointScheduling/GraphFactsIO.h"
#include "TaskflowFissionSourceReplay.h"
#include "ProductionSchedulerAdapter.h"
#include "Backend/Neura/Transforms/Optimizations/TaskflowFission.h"

#include "TaskflowDialect/TaskflowOps.h"
#include "NeuraDialect/NeuraOps.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;
namespace json = llvm::json;

namespace {

constexpr StringLiteral kSearchSchema = "orbit-neighborhood-search-v2";
constexpr StringLiteral kCheckpointSchema =
    "orbit-neighborhood-search-checkpoint-v6";
constexpr StringLiteral kDiagnostic23CheckpointSchema =
    "orbit-neighborhood-search-checkpoint-v7";
constexpr StringLiteral kFamilyFunnelSchema =
    "orbit-joint-neighborhood-family-funnel-v1";
constexpr StringLiteral kTypedActionHistorySchema =
    "orbit-joint-neighborhood-typed-actions-v1";
constexpr StringLiteral kSearchScope = "budgeted-complete-program-neighborhood";
constexpr StringLiteral kActiveTransferProofSchema =
    "orbit-static-active-transfer-proof-v1";
constexpr StringLiteral kActiveTransferWitnessEncoding =
    "exact-byte-interned-active-transfer-v1";
constexpr StringLiteral kDirectPerCgra2x2ArchitectureContract =
    "neura-architecture-v1:amoeba_4x4_cgra_2x2_context6";
constexpr StringLiteral kDiagnostic23ArchitectureContract =
    "neura-architecture-v1:amoeba_4x4_cgra_2x2_context6_ctrlmem23_diagnostic";
constexpr int64_t kDefaultRounds = 20;
constexpr int64_t kDefaultCandidates = 20000;
constexpr int64_t kDefaultBeam = 16;
constexpr int64_t kDefaultDiversity = 4;
// The artifact and validation contract reserve twelve CPUs for this scorer.
// Keep the option bounded even when a caller requests an accidentally larger
// worker pool; serial execution remains the exact default.
constexpr int64_t kMaxScoringWorkers = 12;
static constexpr int64_t kFormalMax4CgraRows[] = {1, 1, 2, 1, 3, 1, 2, 4};
static constexpr int64_t kFormalMax4CgraCols[] = {1, 2, 1, 3, 1, 4, 2, 1};

static bool cgraShapeToMapperTileShape(int64_t cgraRows, int64_t cgraCols,
                                       int64_t perCgraRows,
                                       int64_t perCgraCols,
                                       int64_t &mapperRows,
                                       int64_t &mapperCols) {
  if (cgraRows <= 0 || cgraCols <= 0 || perCgraRows <= 0 ||
      perCgraCols <= 0 ||
      cgraRows > std::numeric_limits<int64_t>::max() / perCgraRows ||
      cgraCols > std::numeric_limits<int64_t>::max() / perCgraCols)
    return false;
  mapperRows = cgraRows * perCgraRows;
  mapperCols = cgraCols * perCgraCols;
  return true;
}

static std::string jsonText(const json::Value &value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << value;
  stream.flush();
  return text;
}

static StringRef checkpointSchemaForCeiling(int64_t diagnosticIICeiling) {
  return diagnosticIICeiling == 23 ? StringRef(kDiagnostic23CheckpointSchema)
                                   : StringRef(kCheckpointSchema);
}

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

static bool parseActiveTransferArguments(StringRef text,
                                         SmallVectorImpl<unsigned> &arguments,
                                         std::string &error) {
  arguments.clear();
  if (text.trim().empty())
    return true;
  SmallVector<StringRef, 8> pieces;
  text.split(pieces, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  std::set<unsigned> seen;
  for (StringRef piece : pieces) {
    unsigned index = 0;
    piece = piece.trim();
    if (piece.empty() || piece.getAsInteger(10, index) ||
        !seen.insert(index).second) {
      error = "active-transfer-arguments must be an ordered comma-separated "
              "list of unique non-negative argument indices";
      arguments.clear();
      return false;
    }
    arguments.push_back(index);
  }
  return true;
}

static bool writeTextAtomically(StringRef path, StringRef text,
                                std::string &error) {
  return writeAtomically(
      path,
      [&](llvm::raw_ostream &stream) {
        stream << text;
        return true;
      },
      error);
}

static bool ensureDirectory(StringRef path, std::string &error) {
  if (path.empty()) {
    error = "neighborhood search output directory is empty";
    return false;
  }
  std::error_code ec = llvm::sys::fs::create_directories(path);
  if (ec) {
    error = "cannot create neighborhood search directory " + path.str() +
            ": " + ec.message();
    return false;
  }
  return true;
}

static std::string actionSignature(const NeighborhoodAction &action) {
  std::string text = action.family + ":" + action.label + ":" +
                     action.shapeTask + ":" + std::to_string(action.shapeRows) +
                     "x" + std::to_string(action.shapeCols) + ":reset=" +
                     (action.canonicalReset ? "1" : "0");
  for (const NeighborhoodPrimitive &primitive : action.primitives) {
    text += ":" + primitive.kind + ":" + primitive.firstTask + ":" +
            primitive.secondTask + ":" + primitive.mode + ":axis=" +
            std::to_string(primitive.axis) + ":factor=" +
            std::to_string(primitive.factor);
    for (unsigned node : primitive.leftNodes)
      text += ":left=" + std::to_string(node);
  }
  return text;
}

static std::string shapeSignature(ArrayRef<NeighborhoodShape> shapes) {
  std::string text;
  for (const NeighborhoodShape &shape : shapes) {
    // Tasks are in source order; names are path provenance rather than shape
    // semantics. Canonical graph facts bind the task/input identity.
    text += std::to_string(shape.rows) + "x" + std::to_string(shape.cols) + ";";
  }
  return text;
}

static bool parseShape(StringRef value, int64_t &rows, int64_t &cols) {
  size_t separator = value.find('x');
  if (separator == StringRef::npos)
    separator = value.find('X');
  if (separator == StringRef::npos)
    return false;
  StringRef rowText = value.take_front(separator);
  StringRef colText = value.drop_front(separator + 1);
  return !rowText.getAsInteger(10, rows) &&
         !colText.getAsInteger(10, cols) && rows > 0 && cols > 0;
}

static std::optional<std::pair<int64_t, int64_t>> taskShapeFromAttrs(
    taskflow::TaskflowTaskOp task) {
  if (auto shape = task->getAttrOfType<StringAttr>("cgra_shape")) {
    int64_t rows = 0, cols = 0;
    if (parseShape(shape.getValue(), rows, cols))
      return std::make_pair(rows, cols);
  }
  return std::nullopt;
}

static std::vector<NeighborhoodShape>
initialShapes(func::FuncOp function) {
  std::vector<NeighborhoodShape> shapes;
  function.walk([&](taskflow::TaskflowTaskOp task) {
    NeighborhoodShape shape;
    shape.task = task.getTaskName().str();
    if (auto parsed = taskShapeFromAttrs(task)) {
      shape.rows = parsed->first;
      shape.cols = parsed->second;
    }
    shapes.push_back(std::move(shape));
  });
  return shapes;
}

static const NeighborhoodShape *findShape(ArrayRef<NeighborhoodShape> shapes,
                                           StringRef task) {
  for (const NeighborhoodShape &shape : shapes)
    if (shape.task == task)
      return &shape;
  return nullptr;
}

static void synchronizeShapes(func::FuncOp function,
                              ArrayRef<NeighborhoodShape> prior,
                              std::vector<NeighborhoodShape> &result) {
  result.clear();
  function.walk([&](taskflow::TaskflowTaskOp task) {
    NeighborhoodShape shape;
    shape.task = task.getTaskName().str();
    if (const NeighborhoodShape *old = findShape(prior, shape.task)) {
      shape.rows = old->rows;
      shape.cols = old->cols;
    } else if (auto parsed = taskShapeFromAttrs(task)) {
      shape.rows = parsed->first;
      shape.cols = parsed->second;
    }
    if (shape.rows <= 0 || shape.cols <= 0) {
      shape.rows = 1;
      shape.cols = 1;
    }
    result.push_back(std::move(shape));
  });
}

static bool runPassOnModule(ModuleOp module, std::unique_ptr<Pass> pass,
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
    reason = reason.empty() ? "source_owned_pass_failed" : reason;
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

static bool extractStructuralFacts(ModuleOp module, StringRef functionName,
                                   StringRef outputDir, uint64_t serial,
                                   std::string &key, std::string &reason,
                                   std::string &diagnostic) {
  std::string factsPath = outputDir.str() + "/.facts-" +
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
  if (!runPassOnModule(module, std::move(pass), reason, diagnostic)) {
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
  llvm::Expected<json::Value> parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    reason = "facts_output_invalid_json";
    llvm::consumeError(parsed.takeError());
    llvm::sys::fs::remove(factsPath);
    return false;
  }
  json::Object *root = parsed->getAsObject();
  const json::Value *structural = root ? root->get("structural_key") : nullptr;
  if (!structural) {
    reason = "facts_output_missing_structural_key";
    llvm::sys::fs::remove(factsPath);
    return false;
  }
  key = jsonText(*structural);
  llvm::sys::fs::remove(factsPath);
  return true;
}

static std::optional<int64_t> durationFromCost(const TaskShapeCost &cost,
                                               int64_t tripCount,
                                               std::string &error) {
  if (!cost.supported)
    return std::nullopt;
  long double duration = static_cast<long double>(cost.startupCycles) +
                         static_cast<long double>(cost.predictedII) *
                             static_cast<long double>(tripCount - 1);
  if (!std::isfinite(duration) || duration <= 0.0L ||
      duration > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    error = "ML II/startup duration is outside scheduler range";
    return std::nullopt;
  }
  duration = std::ceil(duration);
  return std::max<int64_t>(1, static_cast<int64_t>(duration));
}

static bool sameTaskSet(ArrayRef<TaskMetadata> tasks,
                        ArrayRef<NeighborhoodShape> shapes) {
  if (tasks.size() != shapes.size()) return false;
  std::set<std::string> names;
  for (unsigned index = 0; index < tasks.size(); ++index)
    if (tasks[index].name != shapes[index].task ||
        !names.insert(shapes[index].task).second) return false;
  return true;
}

struct SearchState {
  std::string id;
  std::string factKey;
  std::string key;
  std::string parentId;
  std::string rejectReason;
  std::vector<std::string> path;
  struct TypedActionHistory {
    bool known = false;
    std::string canonicalFactKey;
    std::vector<NeighborhoodShape> initialShapes;
    std::vector<NeighborhoodAction> actions;
    // Source-side fission is replayed against the immutable prepared
    // Taskflow parent before any Neura neighborhood actions.
    std::vector<NeighborhoodAction> fissionActions;
    std::string unknownReason;
    std::vector<std::vector<uint64_t>> dependencies;
    std::map<std::string, uint64_t> taskProducers;
  } actionHistory;
  std::vector<NeighborhoodShape> shapes;
  OwningOpRef<ModuleOp> module;
  int64_t score = std::numeric_limits<int64_t>::max();
  ProductionScheduledResult schedule;
  SmallVector<TaskShapeChoice> choices;
  SmallVector<int64_t> durations;
  SmallVector<TaskShapeCost> costs;
  std::string costPath;
  std::vector<std::string> controlRoles;
  bool control = false;
  bool keyReserved = false;
  bool scored = false;
};

struct ControlRoleProvenance {
  std::string candidatePath;
  std::vector<std::string> path;
  SearchState::TypedActionHistory actionHistory;
};

struct ArchiveRecord {
  std::string id;
  std::string key;
  std::string factKey;
  std::string parentId;
  std::string candidatePath;
  std::vector<std::string> path;
  SearchState::TypedActionHistory actionHistory;
  std::vector<NeighborhoodShape> shapes;
  SmallVector<TaskShapeChoice> choices;
  SmallVector<int64_t> durations;
  SmallVector<TaskShapeCost> costs;
  ProductionScheduledResult schedule;
  std::string costPath;
  std::vector<std::string> controlRoles;
  std::map<std::string, ControlRoleProvenance> controlRoleProvenance;
  std::string graphId;
  bool control = false;
  std::vector<std::vector<std::string>> alternatePaths;
  int64_t score = std::numeric_limits<int64_t>::max();
  bool valid = false;
  std::string rejectReason;
  int64_t round = 0;
};

static json::Object shapeObject(const NeighborhoodShape &shape);
static json::Object typedActionObject(const NeighborhoodAction &action);
static bool parseShapeArray(const json::Array *array,
                            std::vector<NeighborhoodShape> &shapes);
static json::Array stringArray(ArrayRef<std::string> values);
static bool parseStringArray(const json::Array *array,
                             std::vector<std::string> &values);
static bool parseTypedAction(const json::Object &object,
                             NeighborhoodAction &action);

static json::Object typedActionHistoryObject(
    const SearchState::TypedActionHistory &history) {
  json::Array initialShapes, actions, fissionActions;
  for (const NeighborhoodShape &shape : history.initialShapes)
    initialShapes.push_back(shapeObject(shape));
  for (const NeighborhoodAction &action : history.actions)
    actions.push_back(typedActionObject(action));
  for (const NeighborhoodAction &action : history.fissionActions)
    fissionActions.push_back(typedActionObject(action));
  return json::Object{
      {"schema", kTypedActionHistorySchema}, {"known", history.known},
      {"canonicalFactKey", history.canonicalFactKey},
      {"initialShapes", std::move(initialShapes)},
      {"actions", std::move(actions)},
      {"fissionActions", std::move(fissionActions)},
      {"unknownReason", history.unknownReason}};
}

static json::Object controlRoleProvenanceObject(
    const ControlRoleProvenance &provenance) {
  return json::Object{
      {"candidate_path", provenance.candidatePath},
      {"action_path", stringArray(provenance.path)},
      {"action_history", typedActionHistoryObject(provenance.actionHistory)}};
}

static json::Object controlRoleProvenanceMapObject(
    const std::map<std::string, ControlRoleProvenance> &provenance) {
  json::Object object;
  for (const auto &[role, state] : provenance)
    object[role] = controlRoleProvenanceObject(state);
  return object;
}

static bool parseTypedActionHistoryObject(
    const json::Object &object,
    SearchState::TypedActionHistory &history) {
  auto schema = object.getString("schema");
  auto known = object.getBoolean("known");
  auto canonicalFactKey = object.getString("canonicalFactKey");
  const json::Array *initialShapes = object.getArray("initialShapes");
  const json::Array *actions = object.getArray("actions");
  const json::Array *fissionActions = object.getArray("fissionActions");
  if (!schema || *schema != kTypedActionHistorySchema || !known ||
      !canonicalFactKey || !initialShapes || !actions ||
      (object.get("fissionActions") && !fissionActions) ||
      !parseShapeArray(initialShapes, history.initialShapes))
    return false;
  history.known = *known;
  history.canonicalFactKey = canonicalFactKey->str();
  history.unknownReason = object.getString("unknownReason").value_or("").str();
  history.actions.clear();
  for (const json::Value &value : *actions) {
    const json::Object *actionObject = value.getAsObject();
    NeighborhoodAction action;
    if (!actionObject || !parseTypedAction(*actionObject, action) ||
        action.family == "fission")
      return false;
    history.actions.push_back(std::move(action));
  }
  history.fissionActions.clear();
  if (fissionActions)
    for (const json::Value &value : *fissionActions) {
      const json::Object *actionObject = value.getAsObject();
      NeighborhoodAction action;
      if (!actionObject || !parseTypedAction(*actionObject, action) ||
          action.family != "fission")
        return false;
      history.fissionActions.push_back(std::move(action));
    }
  if (history.known &&
      (history.canonicalFactKey.empty() || history.initialShapes.empty() ||
       !history.unknownReason.empty()))
    return false;
  if (!history.known) {
    if (history.unknownReason.empty())
      history.unknownReason = "typed_action_history_not_authenticated";
    history.canonicalFactKey.clear();
    history.initialShapes.clear();
    history.actions.clear();
    history.fissionActions.clear();
  }
  history.dependencies.clear();
  history.taskProducers.clear();
  return true;
}

static SearchState::TypedActionHistory portableActionHistory(
    const SearchState::TypedActionHistory &history) {
  SearchState::TypedActionHistory result;
  result.known = history.known;
  result.canonicalFactKey = history.canonicalFactKey;
  result.initialShapes = history.initialShapes;
  result.actions = history.actions;
  result.fissionActions = history.fissionActions;
  result.unknownReason = history.unknownReason;
  return result;
}

static void markActionHistoryUnknown(
    SearchState::TypedActionHistory &history, StringRef reason) {
  history.known = false;
  history.unknownReason = reason.empty()
                              ? "typed_action_history_not_authenticated"
                              : reason.str();
  history.canonicalFactKey.clear();
  history.initialShapes.clear();
  history.actions.clear();
  history.fissionActions.clear();
  history.dependencies.clear();
  history.taskProducers.clear();
}

static void importTypedActionHistory(
    const json::Object &record,
    SearchState::TypedActionHistory &history) {
  const json::Object *object = record.getObject("action_history");
  if (!object) {
    markActionHistoryUnknown(history,
                             "seed_missing_authenticated_typed_action_history");
    return;
  }
  SearchState::TypedActionHistory imported;
  if (!parseTypedActionHistoryObject(*object, imported)) {
    markActionHistoryUnknown(history,
                             "seed_typed_action_history_malformed");
    return;
  }
  if (!imported.known) {
    markActionHistoryUnknown(history, imported.unknownReason);
    return;
  }
  history = std::move(imported);
}

struct UnsupportedUnitQuery {
  std::string task;
  int64_t mapperRows = 0;
  int64_t mapperCols = 0;
  double lowerBound = 0.0;
  std::string reason;
};

static std::vector<int64_t> numericTieKey(ArrayRef<TaskShapeChoice> choices,
                                         const ProductionScheduledResult &schedule) {
  std::vector<int64_t> key;
  for (const auto &choice : choices) {
    key.push_back(choice.shape.rows); key.push_back(choice.shape.cols);
  }
  for (const auto &placement : schedule.placements) {
    key.push_back(placement.task); key.push_back(placement.row);
    key.push_back(placement.col); key.push_back(placement.start);
    key.push_back(placement.end); key.push_back(placement.idleCycles);
  }
  for (unsigned task : schedule.dispatch) key.push_back(task);
  return key;
}
static bool archiveLess(const ArchiveRecord &lhs, const ArchiveRecord &rhs) {
  if (lhs.valid != rhs.valid) return lhs.valid > rhs.valid;
  if (lhs.score != rhs.score) return lhs.score < rhs.score;
  auto left = numericTieKey(lhs.choices, lhs.schedule);
  auto right = numericTieKey(rhs.choices, rhs.schedule);
  if (left != right) return left < right;
  if (lhs.key != rhs.key) return lhs.key < rhs.key;
  return lhs.id < rhs.id;
}
static bool stateLess(const SearchState &lhs, const SearchState &rhs) {
  if (lhs.scored != rhs.scored) return lhs.scored > rhs.scored;
  if (lhs.score != rhs.score) return lhs.score < rhs.score;
  auto left = numericTieKey(lhs.choices, lhs.schedule);
  auto right = numericTieKey(rhs.choices, rhs.schedule);
  if (left != right) return left < right;
  if (lhs.key != rhs.key) return lhs.key < rhs.key;
  return lhs.id < rhs.id;
}

static json::Object shapeObject(const NeighborhoodShape &shape) {
  return json::Object{{"task", shape.task},
                      {"rows", shape.rows},
                      {"cols", shape.cols}};
}

static json::Array stringArray(ArrayRef<std::string> values) {
  json::Array result;
  for (StringRef value : values)
    result.push_back(value.str());
  return result;
}

static json::Array choiceRecords(ArrayRef<TaskShapeChoice> choices) {
  json::Array result;
  for (const TaskShapeChoice &choice : choices)
    result.push_back(json::Object{{"task", choice.task},
                                  {"rows", choice.shape.rows},
                                  {"cols", choice.shape.cols},
                                  {"mapper_tile_rows", choice.shape.mapperRows},
                                  {"mapper_tile_cols", choice.shape.mapperCols},
                                  {"trip_count", choice.tripCount}});
  return result;
}

static json::Array costRecords(ArrayRef<TaskShapeChoice> choices,
                               ArrayRef<TaskShapeCost> costs,
                               ArrayRef<int64_t> durations) {
  json::Array result;
  for (size_t index = 0; index < choices.size(); ++index) {
    json::Object record{{"task", choices[index].task},
                        {"mapper_tile_rows", choices[index].shape.mapperRows},
                        {"mapper_tile_cols", choices[index].shape.mapperCols},
                        {"trip_count", choices[index].tripCount},
                        {"predicted_duration", durations[index]}};
    if (index < costs.size()) {
      record["predicted_ii"] = costs[index].predictedII;
      record["startup_cycles"] = costs[index].startupCycles;
      record["support_status"] = costs[index].supported ? "supported"
                                                           : "unsupported";
    }
    result.push_back(std::move(record));
  }
  return result;
}

static json::Array scheduleRecords(ArrayRef<TaskShapeChoice> choices,
                                   ArrayRef<ExactSchedulePlacement> placements) {
  json::Array result;
  for (const ExactSchedulePlacement &placement : placements) {
    if (placement.task >= choices.size())
      continue;
    const TaskShapeChoice &choice = choices[placement.task];
    result.push_back(json::Object{{"task", choice.task},
                                  {"row", placement.row},
                                  {"col", placement.col},
                                  {"rows", choice.shape.rows},
                                  {"cols", choice.shape.cols},
                                  {"start_cycle", placement.start},
                                  {"end_cycle", placement.end},
                                  {"idle_cycles", placement.idleCycles}});
  }
  return result;
}

static json::Array indexArray(ArrayRef<unsigned> values,
                              ArrayRef<TaskShapeChoice> choices) {
  json::Array result;
  for (unsigned index : values)
    if (index < choices.size())
      result.push_back(choices[index].task);
  return result;
}

static json::Array uint64Array(ArrayRef<uint64_t> values) {
  json::Array result;
  for (uint64_t value : values)
    result.push_back(static_cast<int64_t>(value));
  return result;
}

static json::Object archiveObject(const ArchiveRecord &record) {
  json::Object object{
      {"record_type", "candidate"},
      {"schema", kSearchSchema},
      {"candidate_id", record.id},
      {"candidate_key", record.key},
      {"graph_facts_key", record.factKey},
      {"parent_candidate_id", record.parentId},
      {"candidate_path", record.candidatePath},
      {"action_path", stringArray(record.path)},
      {"action_history", typedActionHistoryObject(record.actionHistory)},
      {"alternate_action_paths", json::Array{}},
      {"shapes", json::Array{}},
      {"task_choices", choiceRecords(record.choices)},
      {"task_costs", costRecords(record.choices, record.costs,
                                  record.durations)},
      {"task_schedule", scheduleRecords(record.choices,
                                         record.schedule.placements)},
      {"dispatch_order", indexArray(record.schedule.dispatch,
                                     record.choices)},
      {"replayed_communication_edges",
       static_cast<int64_t>(record.schedule.replayedEdges)},
      {"communication_trace_known", record.schedule.replayedEdgesKnown},
      {"cost_catalogue_path", record.costPath},
      {"control_candidate", record.control},
      {"control_roles", stringArray(record.controlRoles)},
      {"control_role_provenance",
       controlRoleProvenanceMapObject(record.controlRoleProvenance)},
      {"graph_variant_id", record.graphId},
      {"predicted_whole_program_cycles", record.valid
                                             ? json::Value(record.score)
                                             : json::Value(nullptr)},
      {"valid", record.valid},
      {"reject_reason", record.rejectReason},
      {"round", record.round}};
  json::Array &shapes = *object.getArray("shapes");
  for (const NeighborhoodShape &shape : record.shapes)
    shapes.push_back(shapeObject(shape));
  json::Array &alternatePaths = *object.getArray("alternate_action_paths");
  for (ArrayRef<std::string> path : record.alternatePaths)
    alternatePaths.push_back(stringArray(path));
  return object;
}

static bool parseShapeArray(const json::Array *array,
                            std::vector<NeighborhoodShape> &shapes) {
  if (!array)
    return false;
  shapes.clear();
  for (const json::Value &value : *array) {
    const json::Object *object = value.getAsObject();
    auto task = object ? object->getString("task") : std::optional<StringRef>();
    auto rows = object ? object->getInteger("rows") : std::optional<int64_t>();
    auto cols = object ? object->getInteger("cols") : std::optional<int64_t>();
    if (!task || !rows || !cols || task->empty() || *rows <= 0 || *cols <= 0)
      return false;
    shapes.push_back({task->str(), *rows, *cols});
  }
  return true;
}

static bool parseShapeArrayFromObject(const json::Object &object,
                                      std::vector<NeighborhoodShape> &shapes) {
  if (parseShapeArray(object.getArray("shapes"), shapes))
    return true;
  if (parseShapeArray(object.getArray("task_shapes"), shapes))
    return true;
  return false;
}

static bool parseStringArray(const json::Array *array,
                             std::vector<std::string> &values) {
  if (!array)
    return false;
  values.clear();
  for (const json::Value &value : *array) {
    auto string = value.getAsString();
    if (!string)
      return false;
    values.push_back(string->str());
  }
  return true;
}

static bool parseUInt64Array(const json::Array *array,
                             std::vector<uint64_t> &values) {
  if (!array)
    return false;
  values.clear();
  for (const json::Value &value : *array) {
    auto integer = value.getAsInteger();
    if (!integer || *integer < 0)
      return false;
    values.push_back(static_cast<uint64_t>(*integer));
  }
  return true;
}

static std::optional<std::string> readCatalogGraphId(StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return std::nullopt;
  llvm::Expected<json::Value> parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return std::nullopt;
  json::Object *root = parsed->getAsObject();
  json::Object *metadata = root ? root->getObject("predictor_metadata") : nullptr;
  auto graph = metadata ? metadata->getString("source_graph_id")
                        : std::optional<StringRef>();
  return graph ? std::optional<std::string>(graph->str()) : std::nullopt;
}

static std::string decimal128(unsigned __int128 value) {
  if (value == 0)
    return "0";
  std::string text;
  while (value) {
    text.push_back(static_cast<char>('0' + value % 10));
    value /= 10;
  }
  std::reverse(text.begin(), text.end());
  return text;
}

static bool writeFiniteShapeDomain(StringRef path, func::FuncOp function,
                                   StringRef graphId, int64_t perCgraRows,
                                   int64_t perCgraCols, std::string &error) {
  if (perCgraRows <= 0 || perCgraCols <= 0) {
    error = "architecture has non-positive per-CGRA mapper dimensions";
    return false;
  }
  json::Array factors;
  unsigned __int128 candidateCount = 1;
  SmallVector<TaskMetadata> metadata;
  std::string metadataError;
  FailureOr<SmallVector<TaskMetadata>> collected =
      collectAnalyticalTaskMetadata(function, metadataError);
  if (failed(collected)) {
    error = metadataError;
    return false;
  }
  metadata = std::move(*collected);
  for (const TaskMetadata &task : metadata) {
    json::Array shapes;
    for (unsigned index = 0; index < 8; ++index) {
      int64_t mapperRows = 0, mapperCols = 0;
      if (!cgraShapeToMapperTileShape(
              kFormalMax4CgraRows[index], kFormalMax4CgraCols[index],
              perCgraRows, perCgraCols, mapperRows, mapperCols)) {
        error = "formal CGRA shape cannot be represented in mapper tiles";
        return false;
      }
      shapes.push_back(json::Object{{"kind", "rect"},
                                    {"rows", kFormalMax4CgraRows[index]},
                                    {"cols", kFormalMax4CgraCols[index]},
                                    {"cgra_count", kFormalMax4CgraRows[index] * kFormalMax4CgraCols[index]},
                                    {"cgra_shape",
                                     std::to_string(kFormalMax4CgraRows[index]) + "x" +
                                         std::to_string(kFormalMax4CgraCols[index])},
                                    {"mapper_tile_rows", mapperRows},
                                    {"mapper_tile_cols", mapperCols}});
    }
    factors.push_back(json::Object{{"task", task.name},
                                   {"trip_count", task.tripCount},
                                   {"shapes", std::move(shapes)}});
    candidateCount *= 8;
  }
  json::Object header{
      {"record_type", "space"},
      {"schema", kFactoredSpaceSchema},
      {"representation", "factored"},
      {"exact", true},
      {"function", function.getSymName()},
      {"graph_variant_id", graphId},
      {"search_scope", kFactoredSearchScope},
      {"shape_policy", kShapePolicy},
      {"shape_pruning_policy", kShapePruningPolicyNone},
      {"spatial_capacity_policy", kSpatialCapacityPolicy},
      {"max_cgras_per_task", 4},
      {"max_changed_tasks", -1},
      {"candidate_count", decimal128(candidateCount)},
      {"factors", std::move(factors)}};
  json::Object footer{{"record_type", "footer"},
                      {"schema", kFactoredSpaceSchema},
                      {"representation", "factored"},
                      {"ranked", false},
                      {"status", "unranked-factored-space"},
                      {"candidate_count", decimal128(candidateCount)}};
  std::string text = jsonText(json::Value(std::move(header))) + "\n" +
                     jsonText(json::Value(std::move(footer))) + "\n";
  return writeTextAtomically(path, text, error);
}

static bool runCurrentCostPredictor(
    ModuleOp module, func::FuncOp function, StringRef functionName,
    StringRef candidateId, StringRef outputDirectory, StringRef modelFile,
    StringRef cacheFile, StringRef checkpointDirectory,
    StringRef architectureContract, StringRef architecturePath,
    int64_t perCgraRows, int64_t perCgraCols,
    StringRef sourceRepository, StringRef sourceCommit,
    StringRef architectureTransferCatalog, StringRef modelNamespace,
    int64_t diagnosticIICeiling,
    bool allowUnsupportedAboveModelCeiling,
    std::string &costPath, std::string &error, bool &fatalFailure) {
  fatalFailure = false;
  if (modelFile.empty() || cacheFile.empty() || architectureContract.empty() ||
      architecturePath.empty() || sourceRepository.empty() ||
      sourceCommit.empty()) {
    error = "changed graph requires model/cache and complete cost provenance";
    fatalFailure = true;
    return false;
  }
  std::string spacesDirectory = outputDirectory.str() + "/spaces";
  if (!ensureDirectory(spacesDirectory, error)) {
    fatalFailure = true;
    return false;
  }
  costPath = outputDirectory.str() + "/costs-" + candidateId.str() + ".json";
  std::string spacePath = spacesDirectory + "/" + candidateId.str() + ".jsonl";
  if (!writeFiniteShapeDomain(spacePath, function, candidateId,
                              perCgraRows, perCgraCols, error)) {
    fatalFailure = true;
    return false;
  }
  std::string outputOptions =
      "function=" + quoteOption(functionName) +
      " space-file=" + quoteOption(spacePath) +
      " ensemble-file=" + quoteOption(modelFile) +
      " checkpoint-dir=" + quoteOption(checkpointDirectory) +
      " architecture-contract=" + quoteOption(architectureContract) +
      " architecture-path=" + quoteOption(architecturePath) +
      " source-git-repository=" + quoteOption(sourceRepository) +
      " source-git-commit=" + quoteOption(sourceCommit) +
      " graph-variant-id=" + quoteOption(candidateId) +
      " model-namespace=" + quoteOption(modelNamespace) +
      " cache=" + quoteOption(cacheFile) + " output=" +
      quoteOption(costPath);
  if (diagnosticIICeiling == 23)
    outputOptions += " diagnostic-ii-ceiling=23";
  if (allowUnsupportedAboveModelCeiling)
    outputOptions += " allow-unsupported-above-model-ceiling=true";
  if (!architectureTransferCatalog.empty())
    outputOptions += " architecture-transfer-catalog=" +
                     quoteOption(architectureTransferCatalog);
  auto pass = mlir::amoeba::neura::createPredictAnalyticalTaskCostCatalogPass();
  if (failed(pass->initializeOptions(outputOptions,
                                     [&](const llvm::Twine &message) {
                                       error = message.str();
                                       return failure();
                                     }))) {
    if (error.empty())
      error = "predictor option parsing failed";
    fatalFailure = true;
    return false;
  }
  std::string diagnostic;
  if (!runPassOnModule(module, std::move(pass), error, diagnostic)) {
    if (!diagnostic.empty())
      error += ":" + diagnostic;
    return false;
  }
  if (!llvm::sys::fs::exists(costPath)) {
    error = "predictor completed without writing its cost catalogue";
    fatalFailure = true;
    return false;
  }
  return true;
}

static bool requireCostIICeiling(const TaskShapeCost *cost,
                                int64_t expectedCeiling,
                                std::string &error) {
  if (!cost || cost->runtimeIICeiling == static_cast<double>(expectedCeiling))
    return true;
  error = "cost catalogue runtime II ceiling mismatch: expected=" +
          std::to_string(expectedCeiling) + ":actual=" +
          std::to_string(cost->runtimeIICeiling);
  return false;
}

static bool writeModule(ModuleOp module, StringRef path, std::string &error) {
  return writeAtomically(
      path,
      [&](llvm::raw_ostream &stream) {
        module.print(stream, OpPrintingFlags().printGenericOpForm());
        stream << "\n";
        return true;
      },
      error);
}


static bool readFileBytes(StringRef path, std::string &bytes,
                          std::string &error, bool optional = false) {
  if (path.empty() && optional) { bytes.clear(); return true; }
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read bound contract " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  bytes = (*buffer)->getBuffer().str();
  return true;
}

static bool fatalStorageError(StringRef text) {
  std::string lower = text.lower();
  return StringRef(lower).contains("no space") ||
         StringRef(lower).contains("cannot write") ||
         StringRef(lower).contains("cannot open") ||
         StringRef(lower).contains("atomic") ||
         StringRef(lower).contains("permission denied") ||
         StringRef(lower).contains("disk guard") ||
         StringRef(lower).contains("cannot create temporary output") ||
         StringRef(lower).contains("cannot publish") ||
         StringRef(lower).contains("failed while writing") ||
         StringRef(lower).contains("read-only file system") ||
         StringRef(lower).contains("input/output error") ||
         StringRef(lower).contains("write failure");
}

static json::Object actionObject(const NeighborhoodAction &action) {
  json::Array primitives;
  for (const auto &primitive : action.primitives) {
    json::Array leftNodes;
    for (unsigned node : primitive.leftNodes)
      leftNodes.push_back(static_cast<int64_t>(node));
    primitives.push_back(json::Object{{"kind", primitive.kind},
        {"first_task", primitive.firstTask}, {"second_task", primitive.secondTask},
        {"mode", primitive.mode}, {"axis", primitive.axis},
        {"factor", primitive.factor}, {"left_nodes", std::move(leftNodes)}});
  }
  return json::Object{{"family", action.family}, {"label", action.label},
      {"shape_task", action.shapeTask}, {"shape_rows", action.shapeRows},
      {"shape_cols", action.shapeCols}, {"canonical_reset", action.canonicalReset},
      {"primitives", std::move(primitives)}};
}

static json::Object typedActionObject(const NeighborhoodAction &action) {
  json::Array primitives;
  for (const auto &primitive : action.primitives) {
    json::Array leftNodes;
    for (unsigned node : primitive.leftNodes)
      leftNodes.push_back(static_cast<int64_t>(node));
    primitives.push_back(json::Object{
        {"kind", primitive.kind}, {"firstTask", primitive.firstTask},
        {"secondTask", primitive.secondTask}, {"axis", primitive.axis},
        {"factor", primitive.factor}, {"mode", primitive.mode},
        {"leftNodes", std::move(leftNodes)}});
  }
  return json::Object{
      {"family", action.family}, {"label", action.label},
      {"primitives", std::move(primitives)},
      {"shapeTask", action.shapeTask}, {"shapeRows", action.shapeRows},
      {"shapeCols", action.shapeCols},
      {"canonicalReset", action.canonicalReset}};
}

static bool parseTypedAction(const json::Object &object,
                             NeighborhoodAction &action) {
  auto family = object.getString("family");
  auto label = object.getString("label");
  auto shapeTask = object.getString("shapeTask");
  auto shapeRows = object.getInteger("shapeRows");
  auto shapeCols = object.getInteger("shapeCols");
  auto canonicalReset = object.getBoolean("canonicalReset");
  const json::Array *primitives = object.getArray("primitives");
  if (!family || !label || !shapeTask || !shapeRows || !shapeCols ||
      !canonicalReset || !primitives)
    return false;
  action.family = family->str();
  action.label = label->str();
  action.shapeTask = shapeTask->str();
  action.shapeRows = *shapeRows;
  action.shapeCols = *shapeCols;
  action.canonicalReset = *canonicalReset;
  action.primitives.clear();
  for (const json::Value &value : *primitives) {
    const json::Object *primitive = value.getAsObject();
    auto kind = primitive ? primitive->getString("kind") : std::nullopt;
    auto firstTask = primitive ? primitive->getString("firstTask")
                               : std::nullopt;
    auto secondTask = primitive ? primitive->getString("secondTask")
                                 : std::nullopt;
    auto mode = primitive ? primitive->getString("mode") : std::nullopt;
    auto axis = primitive ? primitive->getInteger("axis") : std::nullopt;
    auto factor = primitive ? primitive->getInteger("factor") : std::nullopt;
    const json::Array *leftNodes =
        primitive ? primitive->getArray("leftNodes") : nullptr;
    if (!kind || !firstTask || !secondTask || !mode || !axis || !factor)
      return false;
    if (primitive->get("leftNodes") && !leftNodes)
      return false;
    NeighborhoodPrimitive parsed{kind->str(), firstTask->str(),
                                 secondTask->str(), mode->str(), *axis,
                                 *factor};
    if (leftNodes)
      for (const json::Value &value : *leftNodes) {
        auto node = value.getAsInteger();
        if (!node || *node < 0 ||
            static_cast<uint64_t>(*node) >
                std::numeric_limits<unsigned>::max())
          return false;
        parsed.leftNodes.push_back(static_cast<unsigned>(*node));
      }
    const bool fission = StringRef(parsed.kind).lower() == "fission";
    if ((fission && parsed.leftNodes.empty()) ||
        (!fission && !parsed.leftNodes.empty()))
      return false;
    action.primitives.push_back(std::move(parsed));
  }
  return true;
}

static bool parseAction(const json::Object &object, NeighborhoodAction &action) {
  auto family = object.getString("family"), label = object.getString("label"),
       task = object.getString("shape_task");
  auto rows = object.getInteger("shape_rows"), cols = object.getInteger("shape_cols");
  auto reset = object.getBoolean("canonical_reset");
  auto primitives = object.getArray("primitives");
  if (!family || !label || !task || !rows || !cols || !reset || !primitives)
    return false;
  action.family = family->str(); action.label = label->str();
  action.shapeTask = task->str(); action.shapeRows = *rows;
  action.shapeCols = *cols; action.canonicalReset = *reset;
  for (const auto &value : *primitives) {
    const auto *primitive = value.getAsObject();
    if (!primitive) return false;
    auto kind = primitive->getString("kind"), first = primitive->getString("first_task"),
         second = primitive->getString("second_task"), mode = primitive->getString("mode");
    auto axis = primitive->getInteger("axis"), factor = primitive->getInteger("factor");
    const json::Array *leftNodes = primitive->getArray("left_nodes");
    if (!kind || !first || !second || !mode || !axis || !factor) return false;
    if (primitive->get("left_nodes") && !leftNodes)
      return false;
    NeighborhoodPrimitive parsed{kind->str(), first->str(), second->str(),
                                 mode->str(), *axis, *factor};
    if (leftNodes)
      for (const json::Value &nodeValue : *leftNodes) {
        auto node = nodeValue.getAsInteger();
        if (!node || *node < 0 ||
            static_cast<uint64_t>(*node) >
                std::numeric_limits<unsigned>::max())
          return false;
        parsed.leftNodes.push_back(static_cast<unsigned>(*node));
      }
    const bool fission = StringRef(parsed.kind).lower() == "fission";
    if ((fission && parsed.leftNodes.empty()) ||
        (!fission && !parsed.leftNodes.empty()))
      return false;
    action.primitives.push_back(std::move(parsed));
  }
  return true;
}

struct PendingNeighbor {
  uint64_t parent = 0;
  NeighborhoodAction action;
};

struct FamilyFunnelCounters {
  uint64_t menu = 0;
  uint64_t generated = 0;
  uint64_t attempted = 0;
  uint64_t materialized = 0;
  uint64_t reject = 0;
  uint64_t unique = 0;
  uint64_t duplicate = 0;
  uint64_t costPreparationAttempted = 0;
  uint64_t freshCostPrepared = 0;
  uint64_t freshCostScored = 0;
  uint64_t cacheReused = 0;
  uint64_t costShapeCacheHits = 0;
  uint64_t costShapeCacheMisses = 0;
  uint64_t schedulerCalls = 0;
  uint64_t schedulerPass = 0;
  uint64_t schedulerReject = 0;
  uint64_t archive = 0;
  uint64_t pendingUnattempted = 0;
  std::map<std::string, uint64_t> rejectReasons;
  std::map<std::string, uint64_t> pendingUnattemptedReasons;
  std::vector<std::string> pendingUnattemptedActionSignatures;
};

using FamilyFunnelRound = std::map<std::string, FamilyFunnelCounters>;
using FamilyFunnelByRound = std::map<int64_t, FamilyFunnelRound>;

struct CandidateCostCacheObservation {
  std::string graphCostKey;
  bool known = false;
  bool reused = false;
};

static json::Object familyFunnelCountersObject(
    const FamilyFunnelCounters &counters) {
  json::Object rejectReasons;
  for (const auto &[reason, count] : counters.rejectReasons)
    rejectReasons[reason] = static_cast<int64_t>(count);
  json::Object pendingReasons;
  for (const auto &[reason, count] : counters.pendingUnattemptedReasons)
    pendingReasons[reason] = static_cast<int64_t>(count);
  return json::Object{
      {"menu", static_cast<int64_t>(counters.menu)},
      {"generated", static_cast<int64_t>(counters.generated)},
      {"attempted", static_cast<int64_t>(counters.attempted)},
      {"materialized", static_cast<int64_t>(counters.materialized)},
      {"reject", static_cast<int64_t>(counters.reject)},
      {"unique", static_cast<int64_t>(counters.unique)},
      {"duplicate", static_cast<int64_t>(counters.duplicate)},
      {"cost_preparation_attempted",
       static_cast<int64_t>(counters.costPreparationAttempted)},
      {"fresh_cost_prepared",
       static_cast<int64_t>(counters.freshCostPrepared)},
      {"fresh_cost_scored", static_cast<int64_t>(counters.freshCostScored)},
      {"cache_reused", static_cast<int64_t>(counters.cacheReused)},
      {"cost_shape_cache_hits",
       static_cast<int64_t>(counters.costShapeCacheHits)},
      {"cost_shape_cache_misses",
       static_cast<int64_t>(counters.costShapeCacheMisses)},
      {"scheduler_calls", static_cast<int64_t>(counters.schedulerCalls)},
      {"scheduler_pass", static_cast<int64_t>(counters.schedulerPass)},
      {"scheduler_reject", static_cast<int64_t>(counters.schedulerReject)},
      {"archive", static_cast<int64_t>(counters.archive)},
      {"pending_unattempted",
       static_cast<int64_t>(counters.pendingUnattempted)},
      {"pending_unattempted_action_signatures",
       stringArray(counters.pendingUnattemptedActionSignatures)},
      {"pending_unattempted_reasons", std::move(pendingReasons)},
      {"reject_reasons", std::move(rejectReasons)}};
}

static bool parseFamilyFunnelCounters(const json::Object &object,
                                      FamilyFunnelCounters &counters) {
  auto readCount = [&](StringRef key, uint64_t &target) {
    auto value = object.getInteger(key);
    if (!value || *value < 0)
      return false;
    target = static_cast<uint64_t>(*value);
    return true;
  };
  if (!readCount("menu", counters.menu) ||
      !readCount("generated", counters.generated) ||
      !readCount("attempted", counters.attempted) ||
      !readCount("materialized", counters.materialized) ||
      !readCount("reject", counters.reject) ||
      !readCount("unique", counters.unique) ||
      !readCount("duplicate", counters.duplicate) ||
      !readCount("cost_preparation_attempted",
                 counters.costPreparationAttempted) ||
      !readCount("fresh_cost_prepared", counters.freshCostPrepared) ||
      !readCount("fresh_cost_scored", counters.freshCostScored) ||
      !readCount("cache_reused", counters.cacheReused) ||
      !readCount("cost_shape_cache_hits", counters.costShapeCacheHits) ||
      !readCount("cost_shape_cache_misses",
                 counters.costShapeCacheMisses) ||
      !readCount("scheduler_calls", counters.schedulerCalls) ||
      !readCount("scheduler_pass", counters.schedulerPass) ||
      !readCount("scheduler_reject", counters.schedulerReject) ||
      !readCount("archive", counters.archive) ||
      !readCount("pending_unattempted", counters.pendingUnattempted) ||
      !parseStringArray(object.getArray("pending_unattempted_action_signatures"),
                        counters.pendingUnattemptedActionSignatures))
    return false;
  const json::Object *reasons = object.getObject("reject_reasons");
  const json::Object *pendingReasons =
      object.getObject("pending_unattempted_reasons");
  if (!reasons || !pendingReasons)
    return false;
  for (const auto &[reasonRef, value] : *reasons) {
    auto count = value.getAsInteger();
    if (!count || *count < 0)
      return false;
    counters.rejectReasons.emplace(reasonRef.str(),
                                   static_cast<uint64_t>(*count));
  }
  for (const auto &[reasonRef, value] : *pendingReasons) {
    auto count = value.getAsInteger();
    if (!count || *count < 0)
      return false;
    counters.pendingUnattemptedReasons.emplace(
        reasonRef.str(), static_cast<uint64_t>(*count));
  }
  return true;
}

static json::Array typedActionArray(ArrayRef<NeighborhoodAction> actions) {
  json::Array result;
  for (const NeighborhoodAction &action : actions)
    result.push_back(typedActionObject(action));
  return result;
}

static std::set<std::string> typedHistoryFamilies(
    const SearchState::TypedActionHistory &history) {
  std::set<std::string> families;
  if (!history.known)
    return families;
  for (const NeighborhoodAction &action : history.fissionActions)
    families.insert(action.family);
  for (const NeighborhoodAction &action : history.actions)
    families.insert(action.family);
  return families;
}

static std::string familyWitnessFilename(StringRef candidateID) {
  auto safe = [](StringRef value) {
    std::string result;
    for (char character : value) {
      const bool alphaNumeric =
          (character >= 'a' && character <= 'z') ||
          (character >= 'A' && character <= 'Z') ||
          (character >= '0' && character <= '9');
      result.push_back(alphaNumeric || character == '-' || character == '_'
                           ? character
                           : '-');
    }
    return result.empty() ? std::string("unknown") : result;
  };
  return "family-best-witness-" + safe(candidateID) + ".json";
}

struct CostGuidedAction {
  uint64_t targetCycles = 0;
  NeighborhoodAction action;
  std::string signature;
};

struct TargetActionStream {
  std::string target;
  uint64_t targetCycles = 0;
  std::vector<CostGuidedAction> actions;
  size_t nextAction = 0;
};

struct ParentFamilyActionStream {
  uint64_t parent = 0;
  std::string family;
  std::vector<TargetActionStream> targets;
  size_t nextTarget = 0;

  bool take(PendingNeighbor &neighbor) {
    if (targets.empty())
      return false;
    for (size_t attempt = 0; attempt < targets.size(); ++attempt) {
      const size_t index = (nextTarget + attempt) % targets.size();
      TargetActionStream &target = targets[index];
      if (target.nextAction >= target.actions.size())
        continue;
      neighbor.parent = parent;
      neighbor.action = target.actions[target.nextAction++].action;
      nextTarget = (index + 1) % targets.size();
      return true;
    }
    return false;
  }
};

static std::string actionTargetKey(const NeighborhoodAction &action) {
  std::set<std::string> targets;
  if (!action.shapeTask.empty())
    targets.insert(action.shapeTask);
  for (const NeighborhoodPrimitive &primitive : action.primitives) {
    if (!primitive.firstTask.empty())
      targets.insert(primitive.firstTask);
    if (!primitive.secondTask.empty())
      targets.insert(primitive.secondTask);
  }
  if (targets.empty())
    return "<global>";
  std::string result;
  for (const std::string &target : targets) {
    if (!result.empty())
      result += "|";
    result += target;
  }
  return result;
}

static uint64_t currentTargetCycles(const SearchState &parent,
                                    const NeighborhoodAction &action) {
  std::set<std::string> targets;
  if (!action.shapeTask.empty())
    targets.insert(action.shapeTask);
  for (const NeighborhoodPrimitive &primitive : action.primitives) {
    if (!primitive.firstTask.empty())
      targets.insert(primitive.firstTask);
    if (!primitive.secondTask.empty())
      targets.insert(primitive.secondTask);
  }
  uint64_t priority = 0;
  for (const std::string &target : targets)
    for (unsigned index = 0; index < parent.choices.size() &&
                             index < parent.durations.size(); ++index)
      if (parent.choices[index].task == target)
        priority = std::max<uint64_t>(
            priority, static_cast<uint64_t>(
                          std::max<int64_t>(0, parent.durations[index])));
  return priority;
}

static uint64_t perRoundScoreQuota(uint64_t scoredCount,
                                   uint64_t maxCandidates,
                                   int64_t roundsRemaining) {
  if (roundsRemaining <= 0 || scoredCount >= maxCandidates)
    return 0;
  const uint64_t remaining = maxCandidates - scoredCount;
  const uint64_t divisor = static_cast<uint64_t>(roundsRemaining);
  return remaining / divisor + (remaining % divisor != 0 ? 1 : 0);
}

static bool ordinaryActionFamilyHasKnownTaskEffects(StringRef family) {
  return family == "shape" || family == "replica" || family == "tiling" ||
         family == "k-tiling" ||
         family == "producer-consumer-co-tiling" ||
         family == "producer-consumer-co-k-tiling" || family == "fusion" ||
         family == "fusion-plus-tiling" || family == "tiling-plus-fusion" ||
         family == "fusion-plus-shape" || family == "sibling-fusion";
}

static bool ordinaryFamilyAllowsPrimitiveKind(StringRef family,
                                              StringRef kind) {
  if (family == "replica")
    return kind == "replica";
  if (family == "tiling" || family == "producer-consumer-co-tiling" ||
      family == "fusion-plus-tiling" || family == "tiling-plus-fusion")
    return kind == "tile" ||
           ((family == "fusion-plus-tiling" ||
             family == "tiling-plus-fusion") &&
            kind == "fusion");
  if (family == "k-tiling" ||
      family == "producer-consumer-co-k-tiling")
    return kind == "k-tile";
  if (family == "fusion")
    return kind == "fusion";
  if (family == "fusion-plus-shape")
    return kind == "fusion";
  if (family == "sibling-fusion")
    return kind == "sibling-fusion";
  return false;
}

static std::string ordinaryActionFootprintFailure(
    const NeighborhoodAction &action, std::set<std::string> &footprint) {
  footprint.clear();
  if (action.canonicalReset || action.family == "canonical-reset")
    return "canonical_reset";
  if (action.family == "lineage-replacement")
    return "lineage_replacement";
  if (action.family == "fission")
    return "fission_in_ordinary_history";
  if (action.family == "identity")
    return "global_or_unknown_footprint";
  if (!ordinaryActionFamilyHasKnownTaskEffects(action.family))
    return "unknown_action_family";
  const bool shapeFamily = action.family == "shape";
  const bool fusionShapeFamily = action.family == "fusion-plus-shape";
  if ((shapeFamily || fusionShapeFamily) != !action.shapeTask.empty())
    return "unknown_footprint";
  if (shapeFamily && !action.primitives.empty())
    return "unknown_primitive_family";
  if (!action.shapeTask.empty())
    footprint.insert(action.shapeTask);
  bool sawTile = false;
  bool sawFusion = false;
  bool sawPrimitive = false;
  for (const NeighborhoodPrimitive &primitive : action.primitives) {
    const StringRef kind = primitive.kind;
    const bool fusionPrimitive = kind == "fusion" ||
                                 kind == "sibling-fusion";
    if (kind != "tile" && kind != "k-tile" && kind != "replica" &&
        !fusionPrimitive)
      return "unknown_primitive_kind";
    if (!ordinaryFamilyAllowsPrimitiveKind(action.family, kind))
      return "unknown_primitive_family";
    if (primitive.firstTask.empty())
      return "unknown_footprint";
    if (fusionPrimitive &&
        (primitive.secondTask.empty() ||
         primitive.secondTask == primitive.firstTask))
      return "unknown_footprint";
    if (!fusionPrimitive && !primitive.secondTask.empty())
      return "unknown_footprint";
    footprint.insert(primitive.firstTask);
    if (!primitive.secondTask.empty())
      footprint.insert(primitive.secondTask);
    sawPrimitive = true;
    sawTile |= kind == "tile";
    sawFusion |= fusionPrimitive;
  }
  if (action.family == "replica" &&
      (!sawPrimitive || action.primitives.size() != 1))
    return "unknown_footprint";
  if (action.family == "tiling" &&
      (!sawTile || action.primitives.size() != 1))
    return "unknown_footprint";
  if (action.family == "k-tiling" && action.primitives.size() != 1)
    return "unknown_footprint";
  if ((action.family == "producer-consumer-co-tiling" ||
       action.family == "producer-consumer-co-k-tiling") &&
      action.primitives.size() != 2)
    return "unknown_footprint";
  if ((action.family == "fusion" || action.family == "sibling-fusion") &&
      action.primitives.size() != 1)
    return "unknown_footprint";
  if (action.family == "fusion-plus-tiling" &&
      (action.primitives.size() != 2 || !sawTile || !sawFusion))
    return "unknown_footprint";
  if (action.family == "tiling-plus-fusion" &&
      (action.primitives.size() != 3 || !sawTile || !sawFusion))
    return "unknown_footprint";
  if (fusionShapeFamily &&
      (action.primitives.size() != 1 || !sawFusion))
    return "unknown_footprint";
  if (footprint.empty())
    return "global_or_unknown_footprint";
  return {};
}

static std::string fissionHistoryConflictReason(
    const SearchState::TypedActionHistory &history,
    const NeighborhoodAction &fission) {
  if (!history.known)
    return "history_unknown";
  if (fission.primitives.size() != 1 ||
      fission.primitives.front().firstTask.empty())
    return "malformed_fission_target";
  const std::string &target = fission.primitives.front().firstTask;
  for (const NeighborhoodAction &action : history.fissionActions)
    if (action.primitives.size() == 1 &&
        action.primitives.front().firstTask == target)
      return "target_already_fissioned";
  for (const NeighborhoodAction &action : history.actions) {
    std::set<std::string> footprint;
    std::string failure = ordinaryActionFootprintFailure(action, footprint);
    if (!failure.empty())
      return failure;
    if (footprint.count(target))
      return !action.shapeTask.empty() ||
                     StringRef(action.family).contains("shape")
                 ? "shape_target_overlap"
                 : "task_footprint_overlap";
  }
  return {};
}

static std::vector<PendingNeighbor> enumerateBalancedPendingNeighbors(
    ArrayRef<SearchState> beam, StringRef functionName, StringRef stage,
    unsigned round, unsigned maxPartitionFactor,
    ModuleOp preparedTaskflowSource, uint64_t maxFissionActionsPerTask,
    ArrayRef<NeighborhoodAction> canonicalFissionActions,
    FamilyFunnelRound &roundFunnel,
    std::string &enumerationError) {
  enumerationError.clear();
  const bool fissionStage =
      stage == "full-joint-fission" || stage == "s6" ||
      stage == "S6" ||
      stage == "shape-temporal-replica-tiling-fusion-fission";
  using TargetMap = std::map<std::string, TargetActionStream>;
  using FamilyMap = std::map<std::string, TargetMap>;
  std::vector<FamilyMap> grouped(beam.size());
  std::set<std::string> familyNames;

  for (auto indexed : llvm::enumerate(beam)) {
    std::string actionEnumerationError;
    auto actions = enumerateNeighborhoodActions(
        indexed.value().module.get(), functionName, indexed.value().shapes,
        stage, round, maxPartitionFactor, preparedTaskflowSource,
        maxFissionActionsPerTask, &actionEnumerationError, nullptr,
        canonicalFissionActions);
    if (!actionEnumerationError.empty()) {
      enumerationError = actionEnumerationError;
      return {};
    }
    for (const NeighborhoodAction &action : actions)
      ++roundFunnel[action.family].menu;
    std::vector<NeighborhoodAction> admitted;
    admitted.reserve(actions.size());
    for (NeighborhoodAction &action : actions) {
      if (fissionStage && action.family == "fission") {
        const std::string reason =
            fissionHistoryConflictReason(indexed.value().actionHistory,
                                         action);
        if (!reason.empty()) {
          ++roundFunnel[action.family].reject;
          ++roundFunnel[action.family].rejectReasons["menu:" + reason];
          continue;
        }
      }
      admitted.push_back(std::move(action));
    }
    FamilyMap &families = grouped[indexed.index()];
    for (NeighborhoodAction &action : admitted) {
      const std::string family = action.family;
      const std::string target = actionTargetKey(action);
      const uint64_t targetCycles =
          currentTargetCycles(indexed.value(), action);
      TargetActionStream &stream = families[family][target];
      stream.target = target;
      stream.targetCycles = std::max(stream.targetCycles, targetCycles);
      stream.actions.push_back(
          {targetCycles, std::move(action), {}});
      CostGuidedAction &queued = stream.actions.back();
      queued.signature = actionSignature(queued.action);
      familyNames.insert(family);
    }
  }

  std::vector<std::map<std::string, ParentFamilyActionStream>> streams(
      beam.size());
  for (size_t parent = 0; parent < grouped.size(); ++parent)
    for (auto &[family, targets] : grouped[parent]) {
      ParentFamilyActionStream &familyStream = streams[parent][family];
      familyStream.parent = parent;
      familyStream.family = family;
      for (auto &[target, targetStream] : targets) {
        llvm::sort(targetStream.actions, [](const CostGuidedAction &left,
                                             const CostGuidedAction &right) {
          if (left.targetCycles != right.targetCycles)
            return left.targetCycles > right.targetCycles;
          return left.signature < right.signature;
        });
        familyStream.targets.push_back(std::move(targetStream));
      }
      llvm::stable_sort(familyStream.targets,
                        [](const TargetActionStream &left,
                           const TargetActionStream &right) {
        if (left.targetCycles != right.targetCycles)
          return left.targetCycles > right.targetCycles;
        return left.target < right.target;
      });
    }

  std::vector<PendingNeighbor> pending;
  bool emitted = true;
  while (emitted) {
    emitted = false;
    for (const std::string &family : familyNames)
      for (size_t parent = 0; parent < streams.size(); ++parent) {
        auto familyIt = streams[parent].find(family);
        if (familyIt == streams[parent].end())
          continue;
        PendingNeighbor neighbor;
        if (familyIt->second.take(neighbor)) {
          ++roundFunnel[neighbor.action.family].generated;
          pending.push_back(std::move(neighbor));
          emitted = true;
        }
      }
  }
  return pending;
}

static std::map<std::string, uint64_t>
successfulActionFamilyScores(ArrayRef<ArchiveRecord> archive, int64_t round) {
  std::map<std::string, uint64_t> scores;
  for (const ArchiveRecord &record : archive) {
    if (!record.valid || record.control || record.parentId.empty() ||
        record.round != round || record.path.empty())
      continue;
    // Generated records append one source-owned action signature to their
    // path. Using that final signature also covers lineage replacement, whose
    // authenticated replay deliberately removes the replaced action history.
    StringRef signature(record.path.back());
    const size_t separator = signature.find(':');
    if (separator == StringRef::npos || separator == 0)
      continue;
    ++scores[signature.take_front(separator).str()];
  }
  return scores;
}

static std::map<std::string, uint64_t> pendingActionFamilyCounts(
    ArrayRef<PendingNeighbor> pending, size_t cursor) {
  std::map<std::string, uint64_t> counts;
  for (size_t index = cursor; index < pending.size(); ++index)
    ++counts[pending[index].action.family];
  return counts;
}

static void consumePendingActionFamily(
    std::map<std::string, uint64_t> &remaining, StringRef family) {
  auto found = remaining.find(family.str());
  if (found == remaining.end())
    return;
  if (--found->second == 0)
    remaining.erase(found);
}

static void prioritizeNextPendingActionFamily(
    std::vector<PendingNeighbor> &pending, size_t cursor,
    const std::map<std::string, uint64_t> &successfulScores,
    const std::map<std::string, uint64_t> &reservedScores,
    const std::map<std::string, uint64_t> &remainingFamilies) {
  if (cursor >= pending.size())
    return;

  std::map<std::string, size_t> firstByFamily;
  for (size_t index = cursor;
       index < pending.size() && firstByFamily.size() < remainingFamilies.size();
       ++index)
    if (remainingFamilies.count(pending[index].action.family))
      firstByFamily.try_emplace(pending[index].action.family, index);
  if (firstByFamily.empty())
    return;

  size_t selected = cursor;
  uint64_t selectedScore = std::numeric_limits<uint64_t>::max();
  for (const auto &[family, index] : firstByFamily) {
    const auto successful = successfulScores.find(family);
    const auto reserved = reservedScores.find(family);
    const uint64_t score =
        (successful == successfulScores.end() ? 0 : successful->second) +
        (reserved == reservedScores.end() ? 0 : reserved->second);
    if (score < selectedScore || (score == selectedScore && index < selected)) {
      selected = index;
      selectedScore = score;
    }
  }
  if (selected != cursor)
    std::rotate(pending.begin() + cursor, pending.begin() + selected,
                pending.begin() + selected + 1);
}

// A worker receives only immutable, source-owned score inputs.  In particular,
// it never receives the parent-context module, cost cache, catalogue path, or
// diagnostic handler.  The owner commits the result in pending-frontier order.
struct ParallelScoreJob {
  std::string moduleIR;
  std::string functionName;
  std::string candidateID;
  SmallVector<TaskShapeChoice> choices;
  SmallVector<int64_t> durations;
  bool fixedDispatch = false;
};

struct ParallelScoreResult {
  bool finished = false;
  bool fatal = false;
  bool scored = false;
  ProductionScheduledResult schedule;
  std::string error;
};

enum class ParallelBatchItemKind { Rejected, Duplicate, Score };

struct ParallelBatchItem {
  size_t pendingIndex = 0;
  std::string actionFamily;
  ParallelBatchItemKind kind = ParallelBatchItemKind::Rejected;
  size_t jobIndex = 0;
  bool materialized = false;
  bool materializationRejected = false;
  bool unique = false;
  bool duplicate = false;
  bool cacheReused = false;
  bool freshCostPrepared = false;
  bool freshCostScored = false;
  uint64_t costShapeCacheHits = 0;
  uint64_t costShapeCacheMisses = 0;
  SearchState state;
};

// A worker owns its context and exact serialized-module cache for the whole
// pass. The cache is keyed by complete IR bytes, so it memoizes only parsing
// and verification of byte-identical modules. Cache modules are declared after
// the context and are therefore destroyed before it during teardown.
constexpr size_t kPrivateModuleCacheCapacity = 16;

struct PrivateModuleCacheEntry {
  std::string moduleIR;
  OwningOpRef<ModuleOp> module;
};

struct PersistentParallelWorker {
  std::unique_ptr<MLIRContext> context;
  std::vector<PrivateModuleCacheEntry> moduleCache;
  uint64_t cacheHits = 0;
  uint64_t cacheMisses = 0;
};

struct PersistentParallelWorkerPool {
  std::vector<std::unique_ptr<PersistentParallelWorker>> workers;

  void ensure(unsigned count, const DialectRegistry &registry) {
    while (workers.size() < count) {
      auto worker = std::make_unique<PersistentParallelWorker>();
      worker->context = std::make_unique<MLIRContext>(
          registry, MLIRContext::Threading::DISABLED);
      workers.push_back(std::move(worker));
    }
  }
};

static bool runPrivateProductionScores(
    ArrayRef<ParallelScoreJob> jobs, const DialectRegistry &registry,
    unsigned requestedWorkers, PersistentParallelWorkerPool &workerPool,
    std::vector<ParallelScoreResult> &results, uint64_t &cacheHits,
    uint64_t &cacheMisses, std::string &error) {
  results.assign(jobs.size(), ParallelScoreResult{});
  if (jobs.empty())
    return true;

  const unsigned workerCount = std::max<unsigned>(
      1, std::min<unsigned>(requestedWorkers, jobs.size()));
  // Context construction and registry mutation stay on the owner thread.
  // Each worker then reuses exactly one disabled-threading context and its
  // bounded exact-IR cache across all batches in this pass.
  workerPool.ensure(workerCount, registry);
  std::vector<uint64_t> cacheHitsBefore(workerCount);
  std::vector<uint64_t> cacheMissesBefore(workerCount);
  for (unsigned index = 0; index < workerCount; ++index) {
    cacheHitsBefore[index] = workerPool.workers[index]->cacheHits;
    cacheMissesBefore[index] = workerPool.workers[index]->cacheMisses;
  }

  std::vector<std::thread> workers;
  workers.reserve(workerCount);
  for (unsigned worker = 0; worker < workerCount; ++worker) {
    workers.emplace_back([&, worker]() {
      PersistentParallelWorker &workerState = *workerPool.workers[worker];
      MLIRContext &context = *workerState.context;
      for (size_t index = worker; index < jobs.size(); index += workerCount) {
          const ParallelScoreJob &job = jobs[index];
          ParallelScoreResult &result = results[index];
          std::string diagnostic;
          std::string localError;
          ModuleOp module;
          FailureOr<func::FuncOp> selected = failure();
          {
            ScopedDiagnosticHandler capture(
                &context, [&](Diagnostic &entry) {
                  if (diagnostic.empty()) {
                    llvm::raw_string_ostream stream(diagnostic);
                    entry.print(stream);
                  }
                  return success();
                });
            for (PrivateModuleCacheEntry &entry : workerState.moduleCache) {
              if (entry.moduleIR == job.moduleIR) {
                module = entry.module.get();
                ++workerState.cacheHits;
                break;
              }
            }
            if (!module) {
              ++workerState.cacheMisses;
              OwningOpRef<ModuleOp> parsed =
                  parseSourceString<ModuleOp>(job.moduleIR, &context);
              if (!parsed || failed(verify(parsed->getOperation()))) {
                result.fatal = true;
                result.error = diagnostic.empty()
                                   ? "parallel worker could not parse or verify candidate"
                                   : diagnostic;
                result.finished = true;
                continue;
              }
              if (workerState.moduleCache.size() >=
                  kPrivateModuleCacheCapacity)
                workerState.moduleCache.erase(workerState.moduleCache.begin());
              workerState.moduleCache.push_back(
                  {job.moduleIR, std::move(parsed)});
              module = workerState.moduleCache.back().module.get();
            }
            selected = selectTaskFunction(module, job.functionName, localError);
            if (failed(selected)) {
              result.fatal = true;
              result.error = localError.empty()
                                 ? "parallel worker could not select candidate function"
                                 : localError;
              result.finished = true;
              continue;
            }
            if (!scheduleWithAmoebaProductionPass(
                    module, *selected, job.choices, job.durations,
                    job.candidateID, job.fixedDispatch, result.schedule,
                    localError)) {
              // A production scheduler rejection is a normal candidate
              // result.  Parse/selection failures above are infrastructure
              // failures and discard the whole unfinished batch.
              result.error = localError.empty()
                                 ? "production scheduler rejected candidate"
                                 : localError;
              result.finished = true;
              continue;
            }
            result.scored = true;
            result.finished = true;
          }
      }
    });
  }
  for (std::thread &worker : workers)
    worker.join();
  for (unsigned worker = 0; worker < workerCount; ++worker) {
    cacheHits += workerPool.workers[worker]->cacheHits - cacheHitsBefore[worker];
    cacheMisses +=
        workerPool.workers[worker]->cacheMisses - cacheMissesBefore[worker];
  }
  for (const ParallelScoreResult &result : results) {
    if (result.fatal) {
      error = result.error.empty() ? "parallel scoring worker failed" : result.error;
      return false;
    }
    if (!result.finished) {
      error = "parallel scoring worker did not complete its assigned job";
      return false;
    }
  }
  return true;
}

static bool parseArchiveObject(const json::Object &object, ArchiveRecord &record,
                               std::string &error) {
  auto id = object.getString("candidate_id"), key = object.getString("candidate_key"),
       fact = object.getString("graph_facts_key");
  auto valid = object.getBoolean("valid");
  if (!id || !key || !fact || !valid) {
    error = "checkpoint archive record lacks source-owned candidate identity";
    return false;
  }
  record.id = id->str(); record.key = key->str(); record.factKey = fact->str();
  record.valid = *valid;
  record.score = object.getInteger("predicted_whole_program_cycles").value_or(
      std::numeric_limits<int64_t>::max());
  record.candidatePath = object.getString("candidate_path").value_or("").str();
  record.parentId = object.getString("parent_candidate_id").value_or("").str();
  record.graphId = object.getString("graph_variant_id").value_or("").str();
  record.costPath = object.getString("cost_catalogue_path").value_or("").str();
  record.control = object.getBoolean("control_candidate").value_or(false);
  record.rejectReason = object.getString("reject_reason").value_or("").str();
  record.round = object.getInteger("round").value_or(0);
  if (!parseStringArray(object.getArray("action_path"), record.path) ||
      !parseShapeArray(object.getArray("shapes"), record.shapes)) {
    error = "checkpoint archive action or shape state is malformed";
    return false;
  }
  if (!parseStringArray(object.getArray("control_roles"),
                        record.controlRoles)) {
    error = "checkpoint archive control-role list is malformed";
    return false;
  }
  const json::Object *actionHistory = object.getObject("action_history");
  if (!actionHistory ||
      !parseTypedActionHistoryObject(*actionHistory, record.actionHistory)) {
    error = "checkpoint archive typed action history is malformed";
    return false;
  }
  const json::Object *roleProvenance =
      object.getObject("control_role_provenance");
  if (!roleProvenance) {
    error = "checkpoint archive control-role provenance is malformed";
    return false;
  }
  for (const auto &[roleRef, value] : *roleProvenance) {
    std::string role = roleRef.str();
    const json::Object *entry = value.getAsObject();
    auto candidatePath = entry ? entry->getString("candidate_path")
                               : std::optional<StringRef>();
    const json::Object *history = entry ? entry->getObject("action_history")
                                        : nullptr;
    ControlRoleProvenance provenance;
    if (!entry || !candidatePath || !history ||
        !parseStringArray(entry->getArray("action_path"), provenance.path) ||
        !parseTypedActionHistoryObject(*history, provenance.actionHistory) ||
        std::find(record.controlRoles.begin(), record.controlRoles.end(),
                  role) == record.controlRoles.end()) {
      error = "checkpoint archive control-role provenance entry is malformed";
      return false;
    }
    provenance.candidatePath = candidatePath->str();
    if (!record.controlRoleProvenance.emplace(std::move(role),
                                               std::move(provenance)).second) {
      error = "checkpoint archive repeats a control-role provenance entry";
      return false;
    }
  }
  if (record.controlRoleProvenance.size() != record.controlRoles.size()) {
    error = "checkpoint archive control roles and provenance differ";
    return false;
  }
  if (record.control != !record.controlRoles.empty()) {
    error = "checkpoint archive control marker and named roles differ";
    return false;
  }
  if (auto alternatives = object.getArray("alternate_action_paths"))
    for (const auto &value : *alternatives) {
      std::vector<std::string> path;
      if (!parseStringArray(value.getAsArray(), path)) return false;
      record.alternatePaths.push_back(std::move(path));
    }
  if (!record.valid) return true;
  auto choices = object.getArray("task_choices"), costs = object.getArray("task_costs"),
       schedule = object.getArray("task_schedule"), dispatch = object.getArray("dispatch_order");
  if (!choices || !costs || !schedule || !dispatch || choices->size() != costs->size()) {
    error = "checkpoint scored record lacks cost and production schedule witnesses";
    return false;
  }
  std::map<std::string, unsigned> indices;
  for (const auto &value : *choices) {
    const auto *choice = value.getAsObject();
    if (!choice) return false;
    auto task = choice->getString("task");
    auto rows = choice->getInteger("rows"), cols = choice->getInteger("cols"),
         mapperRows = choice->getInteger("mapper_tile_rows"),
         mapperCols = choice->getInteger("mapper_tile_cols"),
         trip = choice->getInteger("trip_count");
    if (!task || !rows || !cols || !mapperRows || !mapperCols || !trip ||
        !indices.emplace(task->str(), record.choices.size()).second) return false;
    record.choices.push_back({task->str(), *trip, {*rows, *cols, *mapperRows, *mapperCols}});
  }
  for (unsigned index = 0; index < costs->size(); ++index) {
    const auto *cost = (*costs)[index].getAsObject();
    if (!cost || cost->getString("task") != record.choices[index].task) return false;
    auto ii = cost->getNumber("predicted_ii"), startup = cost->getNumber("startup_cycles");
    auto duration = cost->getInteger("predicted_duration");
    if (!ii || !startup || !duration) return false;
    record.costs.push_back({*ii, *startup, true});
    record.durations.push_back(*duration);
  }
  for (const auto &value : *schedule) {
    const auto *entry = value.getAsObject();
    if (!entry) return false;
    auto task = entry->getString("task");
    auto row = entry->getInteger("row"), col = entry->getInteger("col"),
         start = entry->getInteger("start_cycle"), end = entry->getInteger("end_cycle");
    if (!task || !row || !col || !start || !end || !indices.count(task->str())) return false;
    ExactSchedulePlacement placement;
    placement.task = indices[task->str()]; placement.row = *row; placement.col = *col;
    placement.start = *start; placement.end = *end;
    placement.idleCycles = entry->getInteger("idle_cycles").value_or(0);
    record.schedule.placements.push_back(placement);
  }
  for (const auto &value : *dispatch) {
    auto task = value.getAsString();
    if (!task || !indices.count(task->str())) return false;
    record.schedule.dispatch.push_back(indices[task->str()]);
  }
  record.schedule.order = record.schedule.dispatch;
  record.schedule.makespan = record.score;
  record.schedule.replayedEdges = object.getInteger("replayed_communication_edges").value_or(0);
  record.schedule.replayedEdgesKnown = object.getBoolean("communication_trace_known").value_or(false);
  return true;
}

static ArchiveRecord stateRecord(const SearchState &state) {
  ArchiveRecord record;
  record.id = state.id; record.key = state.key; record.factKey = state.factKey;
  record.parentId = state.parentId; record.path = state.path;
  record.actionHistory = portableActionHistory(state.actionHistory);
  record.shapes = state.shapes; record.choices = state.choices;
  record.costs = state.costs; record.durations = state.durations;
  record.schedule = state.schedule; record.costPath = state.costPath;
  record.control = state.control; record.controlRoles = state.controlRoles;
  for (const std::string &role : record.controlRoles) {
    ControlRoleProvenance provenance;
    provenance.path = state.path;
    provenance.actionHistory =
        portableActionHistory(state.actionHistory);
    record.controlRoleProvenance.emplace(role, std::move(provenance));
  }
  record.score = state.score; record.valid = state.scored;
  record.rejectReason = state.rejectReason;
  return record;
}

static json::Object stateObject(const SearchState &state) {
  json::Object object = archiveObject(stateRecord(state));
  object["module_ir"] = neighborhoodReplaySourceText(state.module.get());
  object["key_reserved"] = state.keyReserved;
  return object;
}

static bool parseState(const json::Object &object, MLIRContext *context,
                       SearchState &state, std::string &error) {
  ArchiveRecord record;
  if (!parseArchiveObject(object, record, error)) return false;
  auto ir = object.getString("module_ir");
  if (!ir) { error = "checkpoint missing exact materialized module snapshot"; return false; }
  state.module = parseSourceString<ModuleOp>(*ir, context);
  if (!state.module) { error = "checkpoint materialized module snapshot does not parse"; return false; }
  state.id = record.id; state.key = record.key; state.factKey = record.factKey;
  state.parentId = record.parentId; state.path = std::move(record.path);
  state.actionHistory = std::move(record.actionHistory);
  state.shapes = std::move(record.shapes); state.choices = std::move(record.choices);
  state.costs = std::move(record.costs); state.durations = std::move(record.durations);
  state.schedule = std::move(record.schedule); state.costPath = record.costPath;
  state.control = record.control; state.controlRoles = std::move(record.controlRoles);
  state.score = record.score; state.scored = record.valid;
  state.keyReserved = object.getBoolean("key_reserved").value_or(true);
  return true;
}
static bool parseStageFixed(StringRef stage) {
  return stage == "S1" || stage == "s1" || stage == "shape" ||
         stage == "shape-only" ||
         stage.contains("fixed") || stage.contains("dependency-ready");
}

static bool parseCheckpointHeader(const json::Object &object, StringRef function,
                                  StringRef stage, StringRef sourceRepository,
                                  StringRef sourceCommit, StringRef architecture,
                                  StringRef protocol,
                                  int64_t diagnosticIICeiling,
                                  std::string &error) {
  auto schema = object.getString("schema");
  auto savedFunction = object.getString("function");
  auto savedStage = object.getString("stage");
  auto savedRepository = object.getString("source_repository");
  auto savedCommit = object.getString("source_commit");
  auto savedArchitecture = object.getString("architecture_path");
  auto savedProtocol = object.getString("protocol_path");
  std::optional<int64_t> savedDiagnosticIICeiling =
      object.getInteger("diagnostic_ii_ceiling");
  if (!schema || !savedFunction || !savedStage || !savedRepository ||
      !savedCommit || !savedArchitecture || !savedProtocol ||
      *schema != checkpointSchemaForCeiling(diagnosticIICeiling) ||
      (diagnosticIICeiling == 23 &&
       (!savedDiagnosticIICeiling || *savedDiagnosticIICeiling != 23)) ||
      (diagnosticIICeiling == 20 && savedDiagnosticIICeiling &&
       *savedDiagnosticIICeiling != 20) ||
      *savedFunction != function ||
      *savedStage != stage || *savedRepository != sourceRepository ||
      *savedCommit != sourceCommit || *savedArchitecture != architecture ||
      *savedProtocol != protocol) {
    error = "neighborhood checkpoint version/provenance binding mismatch";
    return false;
  }
  return true;
}

class SearchJointNeighborhoodPass
    : public PassWrapper<SearchJointNeighborhoodPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SearchJointNeighborhoodPass)

  SearchJointNeighborhoodPass() = default;
  SearchJointNeighborhoodPass(const SearchJointNeighborhoodPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override { return "search-joint-neighborhood"; }
  StringRef getDescription() const override {
    return "Budgeted cost-guided complete-program neighborhood search";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registerTaskflowFissionCanonicalLoweringDependencies(registry);
    // These passes run in nested managers while the outer manager has already
    // entered its execution context. Declare their dialects here so loading a
    // K-reduction or replica materializer never happens during execution.
    mlir::amoeba::neura::createExtractJointTaskGraphFactsPass()
        ->getDependentDialects(registry);
    mlir::amoeba::neura::createPredictAnalyticalTaskCostCatalogPass()
        ->getDependentDialects(registry);
    mlir::amoeba::neura::createMaterializeJointTaskReplicasPass()
        ->getDependentDialects(registry);
    mlir::amoeba::neura::createMaterializeNeuraJointRewritePass()
        ->getDependentDialects(registry);
    mlir::amoeba::neura::createMaterializeNeuraKReductionPass()
        ->getDependentDialects(registry);
  }

  Option<std::string> outputDir{*this, "output-dir", llvm::cl::init("")};
  Option<std::string> functionName{*this, "function", llvm::cl::init("")};
  Option<std::string> stage{*this, "stage", llvm::cl::init("")};
  Option<std::string> parentCostFile{*this, "parent-cost-file",
                                     llvm::cl::init("")};
  Option<std::string> architecturePath{*this, "architecture-path",
                                       llvm::cl::init("")};
  Option<std::string> sourceRepository{*this, "source-repository",
                                      llvm::cl::init("")};
  Option<std::string> sourceCommit{*this, "source-commit", llvm::cl::init("")};
  Option<std::string> modelPath{*this, "model", llvm::cl::init("")};
  Option<std::string> cachePath{*this, "cache", llvm::cl::init("")};
  Option<std::string> architectureContract{
      *this, "architecture-contract",
      llvm::cl::init("")};
  Option<std::string> architectureTransferCatalog{
      *this, "architecture-transfer-catalog", llvm::cl::init("")};
  Option<std::string> modelNamespace{
      *this, "model-namespace",
      llvm::cl::init("formal-max4-nohash-v2-exploratory")};
  Option<int64_t> diagnosticIICeiling{
      *this, "diagnostic-ii-ceiling",
      llvm::cl::desc("Training II ceiling 20 or explicit diagnostic runtime "
                     "ceiling 23 for the direct per-CGRA 2x2 model"),
      llvm::cl::init(20)};
  Option<std::string> seedManifest{*this, "seed-manifest", llvm::cl::init("")};
  Option<std::string> previousWinner{*this, "previous-winner",
                                     llvm::cl::init("")};
  Option<std::string> historicalNativeWinner{
      *this, "historical-native-winner", llvm::cl::init("")};
  Option<std::string> activeTransferArgumentsText{
      *this, "active-transfer-arguments",
      llvm::cl::desc("Ordered comma-separated memref arguments whose static "
                      "active-transfer proofs participate in search identity."),
      llvm::cl::init("")};
  Option<bool> activeTransferRequireProven{
      *this, "active-transfer-require-proven",
      llvm::cl::desc("Reject candidates unless every opted-in active-transfer "
                     "argument remains statically proven."),
      llvm::cl::init(true)};
  Option<bool> requireSourceIterationDomain{*this, "require-source-iteration-domain",
      llvm::cl::desc("Require complete source-domain proofs on every canonical task"),
      llvm::cl::init(false)};
  Option<bool> bootstrapSupportedShapes{
      *this, "bootstrap-supported-shapes",
      llvm::cl::desc("When a validated unit-shape query is explicitly outside "
                     "the model interval, initialize that task to the "
                     "minimum-area supported model shape"),
      llvm::cl::init(false)};
  Option<int64_t> maxRounds{*this, "max-rounds", llvm::cl::init(kDefaultRounds)};
  Option<int64_t> maxCandidates{*this, "max-candidates",
                                llvm::cl::init(kDefaultCandidates)};
  Option<int64_t> beamWidth{*this, "beam-width", llvm::cl::init(kDefaultBeam)};
  Option<int64_t> diversitySlots{*this, "diversity-slots",
                                 llvm::cl::init(kDefaultDiversity)};
  Option<std::string> checkpoint{*this, "checkpoint", llvm::cl::init("")};
  Option<bool> resume{*this, "resume", llvm::cl::init(false)};

  Option<std::string> protocolFile{*this, "protocol",
      llvm::cl::init("config/protocols/amoeba_input0_neighborhood_v3.json")};
  Option<std::string> sourceContractFile{*this, "source-contract-file", llvm::cl::init("")};
  Option<int64_t> checkpointActions{*this, "checkpoint-actions", llvm::cl::init(128)};
  Option<int64_t> pauseAfterCandidates{*this, "test-pause-after-candidates", llvm::cl::init(0)};
  Option<int64_t> minimumFreeBytes{*this, "minimum-free-bytes", llvm::cl::init(16777216)};
  // One preserves the pre-existing serial scorer exactly.  Values above one
  // parallelize only the production scheduler invocation after all candidate
  // preparation and identity decisions have been committed by the owner.
  Option<int64_t> scoringWorkers{*this, "scoring-workers", llvm::cl::init(1)};
  Option<int64_t> maxPartitionFactor{*this, "max-partition-factor", llvm::cl::init(4)};
  Option<std::string> preparedSourceFile{*this, "prepared-source-file",
                                         llvm::cl::init("")};
  Option<int64_t> maxFissionActionsPerTask{
      *this, "max-fission-actions-per-task", llvm::cl::init(64)};

  void runOnOperation() override {
    ModuleOp canonical = getOperation();
    std::string error;
    auto fatal = [&](StringRef message) {
      canonical.emitError() << message;
      signalPassFailure();
    };
    if (outputDir.empty() || functionName.empty() || stage.empty() ||
        maxRounds <= 0 || maxCandidates <= 0 || beamWidth <= 0 ||
        diversitySlots <= 0 || diversitySlots > beamWidth || checkpointActions <= 0 ||
        minimumFreeBytes < 0 || scoringWorkers <= 0 ||
        scoringWorkers > kMaxScoringWorkers ||
        (maxPartitionFactor != 4 && maxPartitionFactor != 8) ||
        maxFissionActionsPerTask <= 0 ||
        sourceContractFile.empty()) {
      fatal("neighborhood search requires positive budgets, valid beam/diversity limits, and source-contract-file");
      return;
    }
    if (diagnosticIICeiling != 20 && diagnosticIICeiling != 23) {
      fatal("diagnostic-ii-ceiling accepts only 20 (default) or 23");
      return;
    }
    if (diagnosticIICeiling == 23 &&
        (StringRef(modelNamespace.getValue()) !=
             kPerCgra2x2ModelNamespace ||
         !bootstrapSupportedShapes ||
         !architectureTransferCatalog.empty())) {
      fatal("diagnostic-ii-ceiling=23 requires the direct per-CGRA 2x2 "
            "model namespace, bootstrap-supported-shapes=true, and no "
            "architecture-transfer-catalog");
      return;
    }
    if (!stageNumber(stage)) { fatal("unknown cumulative ablation stage"); return; }
    const bool fissionStage = stageNumber(stage) == 6;
    if (fissionStage != !preparedSourceFile.empty()) {
      fatal(fissionStage
                ? "full-joint-fission requires prepared-source-file"
                : "prepared-source-file is valid only for full-joint-fission");
      return;
    }
    if (fissionStage) {
      std::string sourceBytes;
      if (!readFileBytes(preparedSourceFile, sourceBytes, error)) {
        fatal(error);
        return;
      }
      preparedTaskflowSource = parseSourceString<ModuleOp>(
          sourceBytes, canonical.getContext());
      if (!preparedTaskflowSource ||
          failed(verify(preparedTaskflowSource->getOperation()))) {
        fatal("prepared-source-file is not a verified ModuleOp");
        return;
      }
      preparedTaskflowSourceBytes = std::move(sourceBytes);
      if (failed(verifyTaskflowFissionCanonicalLowering(
              preparedTaskflowSource.get(), canonical, error))) {
        fatal("prepared source does not reproduce canonical Neura input: " +
              error);
        return;
      }
      preparedSourceCanonicalVerified = true;
      FailureOr<std::vector<NeighborhoodAction>> enumeratedFissions =
          enumerateTaskflowFissionNeighborhoodActions(
              preparedTaskflowSource.get(), functionName,
              static_cast<uint64_t>(maxFissionActionsPerTask.getValue()),
              fissionSupportDiagnostics, error);
      if (failed(enumeratedFissions)) {
        fatal(error.empty() ? "fission cut enumeration failed" : error);
        return;
      }
      canonicalFissionActions = std::move(*enumeratedFissions);
    }
    if (bootstrapSupportedShapes) {
      std::string protocolBytes;
      if (!readFileBytes(protocolFile, protocolBytes, error)) {
        fatal(error);
        return;
      }
      auto protocolValue = json::parse(protocolBytes);
      if (!protocolValue) {
        error = "bound protocol is not JSON: " +
                llvm::toString(protocolValue.takeError());
        fatal(error);
        return;
      }
      const json::Object *protocolObject = protocolValue->getAsObject();
      const json::Object *protocolSearch =
          protocolObject ? protocolObject->getObject("search") : nullptr;
      std::optional<StringRef> policy = protocolSearch
          ? protocolSearch->getString("supported_shape_bootstrap_policy")
          : std::nullopt;
      if (!policy || *policy != kSupportedShapeBootstrapPolicy) {
        fatal("supported-shape bootstrap requires the matching policy in the "
              "bound common protocol");
        return;
      }
    }
    if (!parseStageFixed(stage) && maxCandidates < 2) {
      fatal("S2-S5 candidate budget must accommodate identity and a neighbor"); return;
    }
    // Each ablation may start directly from its canonical identity. Importing
    // a previous measured winner is optional for historical warm-start runs.
    const std::string &schedulerArchitecture =
        ::mlir::amoeba::getNeuraArchitectureSpecFile();
    if (schedulerArchitecture.empty() || architecturePath.empty()) {
      fatal("neighborhood search requires the production scheduler architecture-spec and bound architecture-path");
      return;
    }
    if (schedulerArchitecture != architecturePath.getValue()) {
      std::string schedulerBytes, boundBytes;
      if (!readFileBytes(schedulerArchitecture, schedulerBytes, error) ||
          !readFileBytes(architecturePath, boundBytes, error)) {
        fatal(error); return;
      }
      if (schedulerBytes != boundBytes) {
        fatal("production scheduler architecture-spec differs from the bound task-cost architecture");
        return;
      }
    }
    const ::mlir::neura::Architecture &architecture =
        ::mlir::neura::getArchitecture();
    perCgraRows = architecture.getPerCgraRows();
    perCgraCols = architecture.getPerCgraColumns();
    if (perCgraRows <= 0 || perCgraCols <= 0) {
      fatal("architecture has non-positive per-CGRA mapper dimensions");
      return;
    }
    if (diagnosticIICeiling == 23 &&
        (perCgraRows != 2 || perCgraCols != 2)) {
      fatal("diagnostic-ii-ceiling=23 requires the validated 2x2 per-CGRA "
            "architecture");
      return;
    }
    interTaskNetworkSpecOverrideBytes.reset();
    StringRef networkOverridePath = getInterTaskNetworkSpecOverridePath();
    if (!networkOverridePath.empty()) {
      std::string networkBytes;
      if (!readFileBytes(networkOverridePath, networkBytes, error)) {
        fatal("cannot read inter-task network override: " + error);
        return;
      }
      FailureOr<std::optional<InterTaskNetworkSpec>> network =
          loadInterTaskNetworkSpec(networkOverridePath,
                                   /*require_network=*/true, error);
      if (failed(network) || !network->has_value()) {
        fatal("invalid inter-task network override: " +
              (error.empty() ? "network section is missing" : error));
        return;
      }
      if (!network->value().validateForGrid(architecture.getMultiCgraRows(),
                                            architecture.getMultiCgraColumns(),
                                            error)) {
        fatal("inter-task network override does not match the production "
              "CGRA grid: " + error);
        return;
      }
      interTaskNetworkSpecOverrideBytes = std::move(networkBytes);
    }
    if (architectureContract.empty()) {
      StringRef architectureId =
          llvm::sys::path::stem(architecturePath.getValue());
      if (architectureId.empty()) {
        fatal("cannot derive architecture contract from architecture-path");
        return;
      }
      effectiveArchitectureContract =
          "neura-architecture-v1:" + architectureId.str();
    } else {
      effectiveArchitectureContract = architectureContract.getValue();
    }
    if (diagnosticIICeiling == 23 &&
        StringRef(effectiveArchitectureContract) !=
            kDirectPerCgra2x2ArchitectureContract &&
        StringRef(effectiveArchitectureContract) !=
            kDiagnostic23ArchitectureContract) {
      fatal("diagnostic-ii-ceiling=23 requires the direct-model context6 "
            "architecture contract, optionally labeled for the II23 "
            "diagnostic architecture");
      return;
    }
    if (!ensureDirectory(outputDir, error) ||
        !ensureDirectory(outputDir + "/candidates", error) ||
        !ensureDirectory(outputDir + "/diagnostics", error)) {
      fatal(error);
      return;
    }
    const std::string candidatesDir = outputDir + "/candidates";
    const std::string checkpointPath = checkpoint.empty() ? outputDir + "/checkpoint.json" : checkpoint;
    journalPath = outputDir + "/archive.journal.jsonl";
    FailureOr<func::FuncOp> selected = selectTaskFunction(canonical, functionName, error);
    if (failed(selected)) { fatal(error); return; }
    if (!parseActiveTransferArguments(activeTransferArgumentsText.getValue(),
                                      activeTransferArguments, error)) {
      fatal(error);
      return;
    }
    for (unsigned index : activeTransferArguments) {
      if (index >= selected->getNumArguments() ||
          !isa<MemRefType>(selected->getArgument(index).getType())) {
        fatal("active-transfer-arguments contains a non-memref or out-of-range "
              "function argument");
        return;
      }
    }
    sourceDomainCanonical = *selected;
    sourceDomainEnabled = requireSourceIterationDomain;
    selected->walk([&](taskflow::TaskflowTaskOp task) {
      sourceDomainEnabled |= task->hasAttr(kSourceIterationDomainAttr);
    });
    if (sourceDomainEnabled &&
        failed(verifySourceIterationDomainPartition(*selected, *selected, error))) {
      fatal("canonical source-domain proof failed: " + error); return;
    }
    auto metadata = collectAnalyticalTaskMetadata(*selected, error);
    if (failed(metadata) || metadata->empty()) { fatal(error.empty() ? "function has no analytical tasks" : error); return; }
    TaskShapeCostCache costCache;
    bool costCacheLoaded = false;
    canonicalCostPath.clear();
    if (!parentCostFile.empty()) {
      auto graphId = readCatalogGraphId(parentCostFile);
      if (bootstrapSupportedShapes &&
          !mlir::amoeba::neura::verifyCurrentModelDomainCostCatalog(
              canonical, functionName, parentCostFile, error)) {
        fatal("parent model-domain cost proof does not match current source: " +
              error);
        return;
      }
      std::string expectedWitness =
          bootstrapSupportedShapes ? neighborhoodReplaySourceText(canonical)
                                  : std::string();
      if (!graphId || !costCache.load(parentCostFile, functionName, *metadata,
          sourceRepository, sourceCommit, architecturePath, *graphId, error,
          bootstrapSupportedShapes, expectedWitness)) {
        fatal("parent cost catalogue does not match current cost protocol: " + error); return;
      }
      costCacheLoaded = true;
      canonicalCostPath = parentCostFile;
    } else if (bootstrapSupportedShapes) {
      if (modelPath.empty()) {
        fatal("supported-shape bootstrap requires a validated parent "
              "catalogue or the C++ predictor model");
        return;
      }
      OwningOpRef<ModuleOp> predictorModule = canonical.clone();
      auto predictorFunction =
          selectTaskFunction(*predictorModule, functionName, error);
      if (failed(predictorFunction)) { fatal(error); return; }
      const std::string catalogueId = "canonical-supported-bootstrap";
      bool fatalPredictorFailure = false;
      if (!runCurrentCostPredictor(
              *predictorModule, *predictorFunction, functionName, catalogueId,
              outputDir, modelPath, cachePath, checkpointDirectoryForModel(),
              effectiveArchitectureContract, architecturePath, perCgraRows,
              perCgraCols, sourceRepository,
              sourceCommit, architectureTransferCatalog, modelNamespace,
              diagnosticIICeiling,
              /*allowUnsupportedAboveModelCeiling=*/true, canonicalCostPath,
              error, fatalPredictorFailure)) {
        fatal("cannot generate canonical supported-shape catalogue: " + error);
        return;
      }
      auto graphId = readCatalogGraphId(canonicalCostPath);
      std::string expectedWitness =
          neighborhoodReplaySourceText(*predictorModule);
      if (!graphId ||
          !mlir::amoeba::neura::verifyCurrentModelDomainCostCatalog(
              *predictorModule, functionName, canonicalCostPath, error) ||
          !costCache.load(
              canonicalCostPath, functionName, *metadata, sourceRepository,
              sourceCommit, architecturePath, *graphId, error,
              /*allowModelDomainUnsupported=*/true,
              expectedWitness)) {
        fatal("generated canonical model-domain catalogue failed source "
              "validation: " + error);
        return;
      }
      costCacheLoaded = true;
    }
    if (!costCacheLoaded && modelPath.empty()) { fatal("validated parent costs or C++ predictor required"); return; }
    canonicalBootstrapApplied = false;
    canonicalAllUnitCostStatus = "supported";
    canonicalAllUnitControlScored = false;
    canonicalAllUnitUnsupportedQueries.clear();
    canonicalIdentityShapes = initialShapes(*selected);
    bool sourceIdentityWasAllUnit = llvm::all_of(
        canonicalIdentityShapes, [](const NeighborhoodShape &shape) {
          return shape.rows == 1 && shape.cols == 1;
        });
    if (bootstrapSupportedShapes) {
      for (const TaskMetadata &task : *metadata) {
        auto shapeIt = llvm::find_if(canonicalIdentityShapes,
                                     [&](const NeighborhoodShape &shape) {
                                       return shape.task == task.name;
                                     });
        if (shapeIt == canonicalIdentityShapes.end() ||
            shapeIt->rows != 1 || shapeIt->cols != 1)
          continue;
        TaskShapeChoice unitChoice{
            task.name, task.tripCount,
            RectShape{1, 1, perCgraRows, perCgraCols}};
        std::string costError;
        const TaskShapeCost *unitCost = costCache.get(unitChoice, costError);
        if (!requireCostIICeiling(unitCost, diagnosticIICeiling, costError)) {
          fatal(costError);
          return;
        }
        if (!unitCost) {
          fatal("canonical unit-shape cost lookup failed: " + costError);
          return;
        }
        if (unitCost->supported)
          continue;
        if (!unitCost->modelDomainUnsupported) {
          fatal("canonical unit-shape query is unsupported without a "
                "validated model-domain proof for task=" + task.name);
          return;
        }
        canonicalBootstrapApplied = true;
        canonicalAllUnitCostStatus = kModelDomainUnsupportedStatus.str();
        canonicalAllUnitUnsupportedQueries.push_back(
            {task.name, perCgraRows, perCgraCols,
             unitCost->analyticalLowerBound,
             unitCost->unsupportedReason});
        bool foundSupported = false;
        for (unsigned index = 0; index < 8; ++index) {
          int64_t mapperRows = 0, mapperCols = 0;
          if (!cgraShapeToMapperTileShape(
                  kFormalMax4CgraRows[index], kFormalMax4CgraCols[index],
                  perCgraRows, perCgraCols, mapperRows, mapperCols)) {
            fatal("formal CGRA shape cannot be represented in mapper tiles");
            return;
          }
          TaskShapeChoice choice{
              task.name, task.tripCount,
              RectShape{kFormalMax4CgraRows[index],
                        kFormalMax4CgraCols[index], mapperRows,
                        mapperCols}};
          const TaskShapeCost *shapeCost = costCache.get(choice, costError);
          if (!requireCostIICeiling(shapeCost, diagnosticIICeiling,
                                   costError)) {
            fatal(costError);
            return;
          }
          if (!shapeCost) {
            fatal("canonical supported-shape lookup failed: " + costError);
            return;
          }
          if (!shapeCost->supported)
            continue;
          shapeIt->rows = kFormalMax4CgraRows[index];
          shapeIt->cols = kFormalMax4CgraCols[index];
          foundSupported = true;
          break;
        }
        if (!foundSupported) {
          fatal("task has no supported model shape for canonical identity: " +
                task.name);
          return;
        }
      }
      canonicalAllUnitControlScored = sourceIdentityWasAllUnit &&
                                      !canonicalBootstrapApplied;
    }
    std::string binding;
    historicalNativeWinnerBoundBytes.clear();
    if (!buildBinding(canonical, binding, error)) { fatal(error); return; }
    bindingPath = checkpointPath + ".binding.json";
    if (resume) {
      std::string savedBinding;
      if (!readFileBytes(bindingPath, savedBinding, error) || savedBinding != binding) {
        fatal(error.empty() ? "neighborhood exact byte/options binding mismatch" : error); return;
      }
    } else {
      if (llvm::sys::fs::exists(checkpointPath) || llvm::sys::fs::exists(journalPath)) {
        fatal("search output already contains continuation state; use resume or a new isolated directory"); return;
      }
      if (!writeTextAtomically(bindingPath, binding, error) ||
          !writeTextAtomically(journalPath, "", error)) { fatal(error); return; }
    }
    std::vector<ArchiveRecord> archive;
    rankedArchivePrefix.clear();
    std::set<std::string> seenKeys;
    std::vector<SearchState> beam, generated;
    std::vector<PendingNeighbor> pending;
    uint64_t pendingCursor = 0, nextSerial = 0, scoredCount = 0, rejectedCount = 0;
    int64_t round = 0;
    std::string canonicalFactKey, stopReason;
    if (resume) {
      if (!restoreCheckpoint(canonical, checkpointPath, binding, round, nextSerial,
          scoredCount, rejectedCount, canonicalFactKey, archive, beam, generated,
          pending, pendingCursor, seenKeys, error)) { fatal(error); return; }
      // The prefix is derived from the exact restored archive. Rebuilding it
      // once avoids storing a second authoritative ranking in the checkpoint.
      for (size_t index = 0; index < archive.size(); ++index)
        if (archive[index].valid) insertRankedArchiveIndex(index, archive);
      if (!writeFamilyFunnelSidecars(outputDir, checkpointPath, round, archive,
                                     beam, pending, pendingCursor, "",
                                     error)) {
        fatal(error);
        return;
      }
    } else {
      SearchState identity;
      identity.id = "neighborhood-" + std::to_string(nextSerial++);
      identity.module = canonical.clone(); identity.shapes = canonicalIdentityShapes;
      identity.control = true; identity.controlRoles.push_back("identity");
      if (!prepareInitialActiveTransferFacts(identity.module.get(), functionName,
                                             error)) {
        fatal(error);
        return;
      }
      if (!prepareCandidateKey(identity, functionName, outputDir, nextSerial, error) ||
          identity.key.empty()) {
        fatal(error.empty()
                  ? "canonical identity facts unknown: " +
                        identity.rejectReason
                  : error);
        return;
      }
      canonicalFactKey = identity.factKey;
      identity.actionHistory.known = true;
      identity.actionHistory.canonicalFactKey = canonicalFactKey;
      identity.actionHistory.initialShapes = canonicalIdentityShapes;
      seenKeys.insert(identity.key); identity.keyReserved = true;
      if (!evaluateCandidate(identity, canonical, functionName, stage, outputDir,
          candidatesDir, canonicalFactKey, costCacheLoaded, costCache, 0,
          scoredCount, /*chargeRoundQuota=*/false, nextSerial, error) ||
          !identity.scored) {
        fatal(error.empty() ? "canonical identity cost/schedule unknown: " + identity.rejectReason : error); return;
      }
      if (!registerCandidate(identity, archive, seenKeys, rejectedCount,
          candidatesDir, error)) { fatal(error); return; }
      beam.push_back(std::move(identity));
      auto ingest = [&](StringRef path, StringRef controlRole, bool required) -> bool {
        if (path.empty()) return !required;
        const bool strictHistoricalSelection =
            controlRole == "historical_measured_winner";
        std::string text;
        if (strictHistoricalSelection) {
          text = historicalNativeWinnerBoundBytes;
        } else if (!readFileBytes(path, text, error)) {
          if (!required) { error.clear(); return true; }
          return false;
        }
        if (strictHistoricalSelection) {
          auto validationBuffer = llvm::MemoryBuffer::getMemBuffer(text);
          unsigned nonemptyLines = 0;
          for (llvm::line_iterator lines(*validationBuffer, true);
               !lines.is_at_end(); ++lines) {
            StringRef line = *lines;
            if (line.trim().empty())
              continue;
            if (++nonemptyLines != 1) {
              error = "historical native winner must contain exactly one selection record";
              return false;
            }
            auto parsed = json::parse(line);
            if (!parsed) {
              llvm::consumeError(parsed.takeError());
              error = "historical native winner selection is malformed JSON";
              return false;
            }
            const json::Object *object = parsed->getAsObject();
            auto recordType = object
                                  ? object->getString("record_type")
                                  : std::optional<StringRef>();
            auto schema = object ? object->getString("schema")
                                 : std::optional<StringRef>();
            auto candidateId = object ? object->getString("candidate_id")
                                      : std::optional<StringRef>();
            auto declaredFunction = object ? object->getString("function")
                                           : std::optional<StringRef>();
            auto declaredRepository =
                object ? object->getString("source_repository")
                       : std::optional<StringRef>();
            auto declaredCommit = object ? object->getString("source_commit")
                                        : std::optional<StringRef>();
            bool bestFound = object &&
                             object->getBoolean("best_found").value_or(false);
            bool valid = object && object->getBoolean("valid").value_or(false);
            auto candidatePath = object ? object->getString("candidate_path")
                                        : std::optional<StringRef>();
            if (!object || !recordType || *recordType != "selection" ||
                !schema || *schema != kSearchSchema || !candidateId ||
                candidateId->empty() || !declaredFunction ||
                *declaredFunction != functionName.getValue() ||
                !declaredRepository ||
                *declaredRepository != sourceRepository.getValue() ||
                !declaredCommit || *declaredCommit != sourceCommit.getValue() ||
                !bestFound || !valid || !candidatePath ||
                candidatePath->empty()) {
              error = "historical native winner lacks a valid source-owned selection record";
              return false;
            }
          }
          if (nonemptyLines != 1) {
            error = "historical native winner selection file is empty";
            return false;
          }
        }
        auto buffer = llvm::MemoryBuffer::getMemBuffer(text);
        unsigned imported = 0;
        for (llvm::line_iterator lines(*buffer, true); !lines.is_at_end() && imported < 16; ++lines) {
          auto value = json::parse(*lines);
          if (!value) {
            llvm::consumeError(value.takeError());
            if (strictHistoricalSelection) {
              error = "historical native winner selection is malformed JSON";
              return false;
            }
            continue;
          }
          const auto *object = value->getAsObject();
          if (!object) {
            if (strictHistoricalSelection) {
              error = "historical native winner selection is not a JSON object";
              return false;
            }
            continue;
          }
          auto type = object->getString("record_type").value_or("");
          if (type != "selection" && type != "control" && type != "score") {
            if (strictHistoricalSelection) {
              error = "historical native winner record type is not selection";
              return false;
            }
            continue;
          }
          std::string sourcePath = object->getString("candidate_path").value_or(
              object->getString("mapper_replay_path").value_or(
                  object->getString("mlir_path").value_or(""))).str();
          if (required && sourcePath.empty()) {
            error = "previous measured winner selection lacks its complete source module";
            return false;
          }
          SearchState seed;
          seed.id = "neighborhood-seed-" + std::to_string(nextSerial++);
          seed.module = sourcePath.empty() ? canonical.clone() :
              parseSourceFile<ModuleOp>(sourcePath, canonical.getContext());
          if (!seed.module) {
            if (required) {
              error = strictHistoricalSelection
                          ? "historical native winner module missing: " + sourcePath
                          : "previous winner module missing: " + sourcePath;
              return false;
            }
            continue;
          }
          if (!validateStageSeed(seed.module.get(), stage,
                                 sourceDomainEnabled, error)) {
            if (required) return false;
            const std::string reason =
                "seed_stage_gate_rejected:" + error;
            appendRejected(archive, seed, reason, 0);
            ++rejectedCount;
            error.clear(); continue;
          }
          auto function = selectTaskFunction(seed.module.get(), functionName, error);
          if (failed(function)) {
            if (required)
              return false;
            const std::string reason =
                "seed_function_rejected:" +
                (error.empty() ? "unknown_function_failure" : error);
            appendRejected(archive, seed, reason, 0);
            ++rejectedCount;
            error.clear();
            return true;
          }
          if (!prepareInitialActiveTransferFacts(seed.module.get(), functionName,
                                                 error)) {
            if (required)
              return false;
            const std::string reason =
                "seed_active_transfer_rejected:" +
                (error.empty() ? "unknown_active_transfer_failure" : error);
            appendRejected(archive, seed, reason, 0);
            ++rejectedCount;
            error.clear();
            return true;
          }
          bool parsedShapes = parseShapeArrayFromObject(*object, seed.shapes);
          if (strictHistoricalSelection && !parsedShapes) {
            error = "historical native winner selection lacks complete task shapes";
            return false;
          }
          const auto *score = object->getObject("score_record");
          const auto *costs = score ? score->getArray("task_costs")
                                    : object->getArray("task_costs");
          std::vector<NeighborhoodShape> costShapes;
          std::string invalidMapperShapeReason;
          if (costs) for (const auto &entry : *costs) {
            const auto *cost = entry.getAsObject();
            if (!cost)
              continue;
            auto task = cost->getString("task");
            auto rows = cost->getInteger("mapper_tile_rows");
            auto cols = cost->getInteger("mapper_tile_cols");
            if (!task || !rows || !cols)
              continue;
            if (*rows <= 0 || *cols <= 0 || *rows % perCgraRows != 0 ||
                *cols % perCgraCols != 0) {
              invalidMapperShapeReason =
                  "seed_mapper_shape_not_aligned_to_architecture_cgra:" +
                  task->str() + ":mapper=" + std::to_string(*rows) + "x" +
                  std::to_string(*cols) + ":per-cgra=" +
                  std::to_string(perCgraRows) + "x" +
                  std::to_string(perCgraCols);
              break;
            }
            costShapes.push_back(
                {task->str(), *rows / perCgraRows, *cols / perCgraCols});
          }
          if (!invalidMapperShapeReason.empty()) {
            if (required) {
              error = invalidMapperShapeReason;
              return false;
            }
            appendRejected(archive, seed, invalidMapperShapeReason, 0);
            ++rejectedCount;
            error.clear();
            continue;
          }
          if (!parsedShapes)
            seed.shapes = std::move(costShapes);
          if (seed.shapes.empty()) {
            if (strictHistoricalSelection) {
              error = "historical native winner selection has no task shapes";
              return false;
            }
            seed.shapes = initialShapes(*function);
          }
          if (!parseStringArray(object->getArray("action_path"), seed.path))
            seed.path.clear();
          importTypedActionHistory(*object, seed.actionHistory);
          if (stageNumber(stage) != 6 &&
              !seed.actionHistory.fissionActions.empty()) {
            const std::string reason =
                "seed_stage_gate_rejected:fission_actions_require_full-joint-fission";
            if (required) {
              error = reason;
              return false;
            }
            appendRejected(archive, seed, reason, 0);
            ++rejectedCount;
            error.clear();
            return true;
          }
          seed.control = !controlRole.empty();
          if (seed.control) seed.controlRoles.push_back(controlRole.str());
          if (!prepareCandidateKey(seed, functionName, outputDir, nextSerial, error) || seed.key.empty()) {
            if (required) {
              if (error.empty())
                error = "previous winner canonical facts unknown: " +
                        seed.rejectReason;
              return false;
            }
            const std::string reason =
                "seed_candidate_facts_unknown:" +
                (seed.rejectReason.empty() ? error : seed.rejectReason);
            appendRejected(archive, seed, reason, 0);
            ++rejectedCount;
            error.clear(); continue;
          }
          if (!authenticateTypedActionHistory(seed, canonical, functionName,
                                              outputDir, nextSerial, error))
            return false;
          if (!seenKeys.insert(seed.key).second) {
            if (!mergeDuplicate(seed, canonical, functionName, archive,
                                outputDir, candidatesDir, nextSerial, error))
              return false;
            ++imported; continue;
          }
          seed.keyReserved = true;
          if (strictHistoricalSelection &&
              scoredCount >= static_cast<uint64_t>(maxCandidates)) {
            error = "distinct historical native winner exceeds max-candidates budget";
            return false;
          }
          const uint64_t reservedControlScores =
              (!previousWinner.empty() ? 1 : 0) +
              (!historicalNativeWinner.empty() ? 1 : 0);
          if (!required && scoredCount + reservedControlScores >=
                               uint64_t(maxCandidates)) {
            seenKeys.erase(seed.key); break;
          }
          if (!evaluateCandidate(seed, canonical, functionName, stage, outputDir,
              candidatesDir, canonicalFactKey, costCacheLoaded, costCache, 0,
              scoredCount, /*chargeRoundQuota=*/false, nextSerial, error))
            return false;
          if (required && !seed.scored) {
            error = (strictHistoricalSelection
                         ? "historical native winner could not be rescored: "
                         : "previous measured winner could not be scored: ") +
                    seed.rejectReason;
            return false;
          }
          if (!registerCandidate(seed, archive, seenKeys, rejectedCount, candidatesDir, error)) return false;
          if (seed.scored &&
              !persistBestFamilyWitnesses(seed, archive, outputDir, error))
            return false;
          if (seed.scored) beam.push_back(std::move(seed));
          ++imported;
          if (required) break;
        }
        if (required && !imported) { error = "previous winner file contains no legal complete selection"; return false; }
        return true;
      };
      if (!ingest(seedManifest, "", false) ||
          !ingest(previousWinner, "previous_stage_measured_winner", !previousWinner.empty()) ||
          !ingest(historicalNativeWinner, "historical_measured_winner",
                  !historicalNativeWinner.empty())) {
        fatal(error.empty() ? "seed import failed" : error); return;
      }
      beam = selectBeam(std::move(beam), beamWidth, diversitySlots);
      if (!retainCanonicalBeam(beam, archive, error)) { fatal(error); return; }
      roundScoredCount = 0;
      roundScoreLimit = perRoundScoreQuota(
          scoredCount, static_cast<uint64_t>(maxCandidates.getValue()),
          maxRounds.getValue() - round);
      if (!saveCheckpoint(checkpointPath, round, nextSerial, scoredCount, rejectedCount,
          canonicalFactKey, archive, beam, generated, pending, pendingCursor, "", error)) { fatal(error); return; }
    }
    std::map<std::string, uint64_t> successfulFamilyScores =
        successfulActionFamilyScores(archive, round);
    const std::map<std::string, uint64_t> noReservedFamilyScores;
    auto began = std::chrono::steady_clock::now();
    const int64_t previousElapsed = elapsedMilliseconds;
    uint64_t actionsSinceCheckpoint = 0;
    while (round < maxRounds && scoredCount < static_cast<uint64_t>(maxCandidates)) {
      if (beam.empty()) { stopReason = "no-new-legal-candidates"; break; }
      if (pending.empty()) {
        generated.clear(); pendingCursor = 0;
        std::string actionEnumerationError;
        pending = enumerateBalancedPendingNeighbors(
            beam, functionName, stage, static_cast<unsigned>(round),
            static_cast<unsigned>(maxPartitionFactor.getValue()),
            preparedTaskflowSource.get(),
            static_cast<uint64_t>(maxFissionActionsPerTask.getValue()),
            canonicalFissionActions, familyFunnelByRound[round],
            actionEnumerationError);
        if (!actionEnumerationError.empty()) {
          fatal(actionEnumerationError);
          return;
        }
        if (!saveCheckpoint(checkpointPath, round, nextSerial, scoredCount, rejectedCount,
            canonicalFactKey, archive, beam, generated, pending, pendingCursor, "", error)) { fatal(error); return; }
      }
      std::map<std::string, uint64_t> remainingPendingFamilies =
          pendingActionFamilyCounts(pending, pendingCursor);
      if (scoringWorkers <= 1) {
      while (pendingCursor < pending.size() &&
             scoredCount < static_cast<uint64_t>(maxCandidates) &&
             roundScoredCount < roundScoreLimit) {
        prioritizeNextPendingActionFamily(pending, pendingCursor,
                                          successfulFamilyScores,
                                          noReservedFamilyScores,
                                          remainingPendingFamilies);
        const auto &neighbor = pending[pendingCursor];
        const std::string actionFamily = neighbor.action.family;
        if (neighbor.parent >= beam.size()) { fatal("checkpoint pending parent outside beam"); return; }
        SearchState &parent = beam[neighbor.parent];
        FamilyFunnelCounters &funnel =
            familyFunnelByRound[round][actionFamily];
        SearchState child;
        child.id = "neighborhood-" + std::to_string(nextSerial++);
        child.parentId = parent.id; child.path = parent.path;
        child.path.push_back(actionSignature(neighbor.action));
        child.shapes = parent.shapes; child.module = parent.module->clone();
        std::string reason, diagnostic, materializeError;
        ++funnel.attempted;
        bool materialized = false;
        bool unique = false;
        bool duplicate = false;
        bool freshCostPrepared = false;
        bool cacheReused = false;
        bool freshCostScored = false;
        bool schedulerCalled = false;
        bool schedulerPassed = false;
        bool schedulerRejected = false;
        bool archived = false;
        bool expandCandidate = false;
        uint64_t costShapeCacheHits = 0;
        uint64_t costShapeCacheMisses = 0;
        std::string outcomeReason;
        NeighborMaterialization materialization = materializeNeighbor(
            child, parent, canonical, functionName, outputDir,
            neighbor.action, canonicalFactKey, nextSerial, reason, diagnostic,
            materializeError);
        if (materialization == NeighborMaterialization::Fatal) {
          fatal(materializeError);
          return;
        }
        if (materialization == NeighborMaterialization::Rejected) {
          outcomeReason =
              (reason.empty() ? "unsupported_or_unknown_action" : reason) +
              ":" + diagnostic;
          ++funnel.reject;
          ++funnel.rejectReasons["materialize:" +
                                 StringRef(outcomeReason)
                                     .take_front(StringRef(outcomeReason).find(':'))
                                     .str()];
          ++rejectedCount; appendRejected(archive, child,
                                           outcomeReason, round);
          archived = true;
        } else {
          materialized = true;
          ++funnel.materialized;
          if (!child.key.empty() && child.rejectReason.empty()) {
            if (!seenKeys.insert(child.key).second) {
              duplicate = true;
              ++funnel.duplicate;
              ++rejectedCount;
              if (!mergeDuplicate(child, canonical, functionName, archive,
                                  outputDir, candidatesDir, nextSerial, error)) {
                fatal(error);
                return;
              }
              archived = true;
            } else {
              unique = true;
              ++funnel.unique;
              child.keyReserved = true;
              CandidateCostCacheObservation cacheObservation =
                  observeCandidateCostCache(child, functionName,
                                            canonicalFactKey,
                                            costCacheLoaded);
              cacheReused = cacheObservation.known &&
                            cacheObservation.reused;
              ++funnel.costPreparationAttempted;
              if (cacheReused)
                ++funnel.cacheReused;
              const uint64_t oldCostHits = costCacheHits;
              const uint64_t oldCostMisses = costCacheMisses;
              const uint64_t oldSchedulerCalls = productionSchedulerCalls;
              if (!evaluateCandidate(child, canonical, functionName, stage, outputDir,
                  candidatesDir, canonicalFactKey, costCacheLoaded, costCache,
                  round, scoredCount, /*chargeRoundQuota=*/true, nextSerial,
                  error) ||
                  !registerCandidate(child, archive, seenKeys, rejectedCount, candidatesDir, error)) { fatal(error); return; }
              if (child.scored &&
                  !persistBestFamilyWitnesses(child, archive, outputDir,
                                              error)) {
                fatal(error);
                return;
              }
              archived = true;
              costShapeCacheHits = costCacheHits - oldCostHits;
              costShapeCacheMisses = costCacheMisses - oldCostMisses;
              funnel.costShapeCacheHits += costShapeCacheHits;
              funnel.costShapeCacheMisses += costShapeCacheMisses;
              const bool generatedFreshCatalogue =
                  cacheObservation.known && !cacheObservation.reused &&
                  !cacheObservation.graphCostKey.empty() &&
                  graphCostPaths.count(cacheObservation.graphCostKey);
              if (generatedFreshCatalogue) {
                freshCostPrepared = true;
                ++funnel.freshCostPrepared;
              }
              schedulerCalled = productionSchedulerCalls > oldSchedulerCalls;
              schedulerPassed = schedulerCalled && child.scored;
              schedulerRejected = schedulerCalled && !child.scored;
              if (schedulerCalled)
                ++funnel.schedulerCalls;
              if (schedulerPassed)
                ++funnel.schedulerPass;
              if (schedulerRejected) {
                ++funnel.schedulerReject;
                outcomeReason = child.rejectReason;
                ++funnel.reject;
                ++funnel.rejectReasons[
                    "scheduler:" +
                    StringRef(outcomeReason)
                        .take_front(StringRef(outcomeReason).find(':'))
                        .str()];
              } else if (!child.scored) {
                outcomeReason = child.rejectReason;
                if (outcomeReason.empty())
                  outcomeReason = "candidate_not_scored";
                ++funnel.reject;
                ++funnel.rejectReasons[
                    "candidate:" +
                    StringRef(outcomeReason)
                        .take_front(StringRef(outcomeReason).find(':'))
                        .str()];
              }
              freshCostScored = freshCostPrepared && schedulerCalled;
              if (freshCostScored)
                ++funnel.freshCostScored;
              archive.back().round = round;
              if (child.scored) {
                ++successfulFamilyScores[neighbor.action.family];
                expandCandidate = true;
              }
            }
          } else if (child.key.empty() && child.rejectReason.empty()) {
            outcomeReason =
                "active_transfer_proof_unsupported:proof preparation returned no candidate key";
            ++funnel.reject;
            ++funnel.rejectReasons["candidate:active_transfer_proof_unsupported"];
            ++rejectedCount;
            appendRejected(archive, child, outcomeReason, round);
            archived = true;
          } else if (child.key.empty()) {
            outcomeReason = child.rejectReason;
            ++funnel.reject;
            ++funnel.rejectReasons[
                "candidate:" +
                StringRef(outcomeReason)
                    .take_front(StringRef(outcomeReason).find(':'))
                    .str()];
            ++rejectedCount;
            appendRejected(archive, child, outcomeReason, round);
            archived = true;
          }
        }
        if (archived)
          ++funnel.archive;
        json::Object eventResult{
            {"attempted", true}, {"materialized", materialized},
            {"reject_reason", outcomeReason}, {"unique", unique},
            {"duplicate", duplicate},
            {"fresh_cost_prepared", freshCostPrepared},
            {"fresh_cost_scored", freshCostScored},
            {"cache_reused", cacheReused},
            {"cost_shape_cache_hits",
             static_cast<int64_t>(costShapeCacheHits)},
            {"cost_shape_cache_misses",
             static_cast<int64_t>(costShapeCacheMisses)},
            {"scheduler_calls", schedulerCalled ? 1 : 0},
            {"scheduler_pass", schedulerPassed},
            {"scheduler_reject", schedulerRejected},
            {"archive", archived}};
        appendFamilyFunnelCandidateEvent(
            round, actionFamily, child.id, child.parentId, neighbor.action,
            child.actionHistory, std::move(eventResult));
        if (expandCandidate) {
          generated.push_back(std::move(child));
          // Keep the expansion frontier bounded rather than serializing every
          // evaluated program in a round. Four diverse slots remain available
          // independently of the twelve cost-selected slots.
          generated = selectBeam(std::move(generated), beamWidth,
                                 diversitySlots);
        }
        ++pendingCursor; ++actionsSinceCheckpoint;
        consumePendingActionFamily(remainingPendingFamilies, actionFamily);
        bool pause = pauseAfterCandidates > 0 && scoredCount >= static_cast<uint64_t>(pauseAfterCandidates);
        if (actionsSinceCheckpoint >= static_cast<uint64_t>(checkpointActions) || pause) {
          if (!saveCheckpoint(checkpointPath, round, nextSerial, scoredCount, rejectedCount,
              canonicalFactKey, archive, beam, generated, pending, pendingCursor,
              pause ? "explicit-pause" : "", error)) { fatal(error); return; }
          actionsSinceCheckpoint = 0;
        }
        if (pause) { stopReason = "explicit-pause"; break; }
      }
      } else {
        // The worker path below keeps all action application, graph-fact and
        // catalogue work in this owner thread.  Only the already-prepared
        // production scheduler calls cross the context boundary.
        const unsigned requestedWorkers =
            static_cast<unsigned>(scoringWorkers.getValue());
        const DialectRegistry &workerRegistry =
            canonical.getContext()->getDialectRegistry();
        while (pendingCursor < pending.size() &&
               scoredCount < static_cast<uint64_t>(maxCandidates) &&
               roundScoredCount < roundScoreLimit) {
          const uint64_t scoreBudget =
              static_cast<uint64_t>(maxCandidates) - scoredCount;
          const uint64_t roundScoreBudget =
              roundScoreLimit - roundScoredCount;
          const uint64_t pauseBudget =
              pauseAfterCandidates > 0 &&
                      scoredCount < static_cast<uint64_t>(pauseAfterCandidates)
                  ? static_cast<uint64_t>(pauseAfterCandidates) - scoredCount
                  : scoreBudget;
          const uint64_t batchScoreLimit =
              std::min({scoreBudget, roundScoreBudget, pauseBudget});
          if (batchScoreLimit == 0)
            break;
          const uint64_t checkpointBudget =
              static_cast<uint64_t>(checkpointActions) -
              std::min<uint64_t>(actionsSinceCheckpoint,
                                 static_cast<uint64_t>(checkpointActions));
          const size_t batchItemLimit = std::max<size_t>(
              1, std::min<uint64_t>(requestedWorkers * 4ULL,
                                    std::max<uint64_t>(1, checkpointBudget)));

          std::vector<ParallelBatchItem> items;
          std::vector<ParallelScoreJob> jobs;
          std::map<std::string, uint64_t> reservedFamilyScores;
          items.reserve(batchItemLimit);
          jobs.reserve(std::min<uint64_t>(batchScoreLimit, batchItemLimit));
          uint64_t cursor = pendingCursor;
          const auto preparationBegan = std::chrono::steady_clock::now();
          while (cursor < pending.size() && jobs.size() < batchScoreLimit &&
                 items.size() < batchItemLimit) {
            prioritizeNextPendingActionFamily(pending, cursor,
                                              successfulFamilyScores,
                                              reservedFamilyScores,
                                              remainingPendingFamilies);
            const PendingNeighbor &neighbor = pending[cursor];
            if (neighbor.parent >= beam.size()) {
              fatal("checkpoint pending parent outside beam");
              return;
            }
            SearchState &parent = beam[neighbor.parent];
            FamilyFunnelCounters &funnel =
                familyFunnelByRound[round][neighbor.action.family];
            ++funnel.attempted;
            SearchState child;
            child.id = "neighborhood-" + std::to_string(nextSerial++);
            child.parentId = parent.id;
            child.path = parent.path;
            child.path.push_back(actionSignature(neighbor.action));
            child.shapes = parent.shapes;
            child.module = parent.module->clone();
            ParallelBatchItem item;
            item.pendingIndex = cursor;
            item.actionFamily = neighbor.action.family;
            item.state = std::move(child);
            std::string reason, diagnostic, materializeError;
            NeighborMaterialization materialization = materializeNeighbor(
                item.state, parent, canonical, functionName, outputDir,
                neighbor.action, canonicalFactKey, nextSerial, reason,
                diagnostic, materializeError);
            if (materialization == NeighborMaterialization::Fatal) {
              fatal(materializeError);
              return;
            }
            if (materialization == NeighborMaterialization::Rejected) {
              item.kind = ParallelBatchItemKind::Rejected;
              item.materializationRejected = true;
              item.state.rejectReason =
                  (reason.empty() ? "unsupported_or_unknown_action" : reason) +
                  ":" + diagnostic;
            } else {
              item.materialized = true;
              ++funnel.materialized;
              if (item.state.key.empty()) {
                item.kind = ParallelBatchItemKind::Rejected;
                item.state.rejectReason = "graph_facts_unknown";
              } else if (!seenKeys.insert(item.state.key).second) {
                // A duplicate whose key was reserved by an earlier item in
                // this batch is committed in order below, after that earlier
                // archive record exists.  Duplicates from an older batch can
                // be merged immediately at the same ordered commit point.
                item.kind = ParallelBatchItemKind::Duplicate;
                item.duplicate = true;
                ++funnel.duplicate;
              } else {
                item.state.keyReserved = true;
                item.unique = true;
                ++funnel.unique;
                CandidateCostCacheObservation cacheObservation =
                    observeCandidateCostCache(item.state, functionName,
                                              canonicalFactKey,
                                              costCacheLoaded);
                item.cacheReused = cacheObservation.known &&
                                   cacheObservation.reused;
                ++funnel.costPreparationAttempted;
                if (item.cacheReused)
                  ++funnel.cacheReused;
                const uint64_t oldCostHits = costCacheHits;
                const uint64_t oldCostMisses = costCacheMisses;
                if (!prepareParallelCandidate(
                        item.state, canonical, functionName, stage, outputDir,
                        canonicalFactKey, costCacheLoaded, costCache,
                        nextSerial, error)) {
                  fatal(error);
                  return;
                }
                funnel.costShapeCacheHits += costCacheHits - oldCostHits;
                funnel.costShapeCacheMisses += costCacheMisses - oldCostMisses;
                item.costShapeCacheHits = costCacheHits - oldCostHits;
                item.costShapeCacheMisses =
                    costCacheMisses - oldCostMisses;
                item.freshCostPrepared =
                    cacheObservation.known && !cacheObservation.reused &&
                    !cacheObservation.graphCostKey.empty() &&
                    graphCostPaths.count(cacheObservation.graphCostKey);
                if (item.freshCostPrepared)
                  ++funnel.freshCostPrepared;
                if (item.state.rejectReason.empty() &&
                    !item.state.choices.empty()) {
                  item.kind = ParallelBatchItemKind::Score;
                  ParallelScoreJob job;
                  job.moduleIR = neighborhoodReplaySourceText(
                      item.state.module.get());
                  job.functionName = functionName.getValue();
                  job.candidateID = item.state.id;
                  job.choices = item.state.choices;
                  job.durations = item.state.durations;
                  job.fixedDispatch = parseStageFixed(stage);
                  item.jobIndex = jobs.size();
                  jobs.push_back(std::move(job));
                  ++reservedFamilyScores[item.actionFamily];
                } else {
                  item.kind = ParallelBatchItemKind::Rejected;
                }
              }
            }
            items.push_back(std::move(item));
            consumePendingActionFamily(remainingPendingFamilies,
                                       neighbor.action.family);
            ++cursor;
          }
          parallelPreparationMilliseconds +=
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - preparationBegan)
                  .count();
          if (items.empty())
            break;
          parallelBatchItemSizes.push_back(items.size());
          parallelBatchScoreJobSizes.push_back(jobs.size());
          parallelScoredJobs += jobs.size();

          std::vector<ParallelScoreResult> results;
          std::string workerError;
          const auto scoringBegan = std::chrono::steady_clock::now();
          if (!runPrivateProductionScores(jobs, workerRegistry,
                                          requestedWorkers, parallelWorkerPool,
                                          results, parallelModuleCacheHits,
                                          parallelModuleCacheMisses,
                                          workerError)) {
            // No archive, ranking, budget, or frontier state has been
            // committed for this batch.  The last durable checkpoint remains
            // a safe replay cursor if the pass is resumed after this failure.
            fatal(workerError);
            return;
          }
          parallelScoringMilliseconds +=
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - scoringBegan)
                  .count();

          bool batchFailed = false;
          for (const ParallelBatchItem &item : items)
            if (item.kind == ParallelBatchItemKind::Score &&
                (item.jobIndex >= results.size() ||
                 !results[item.jobIndex].finished ||
                 results[item.jobIndex].fatal)) {
              batchFailed = true;
              workerError = item.jobIndex < results.size()
                                ? results[item.jobIndex].error
                                : "parallel scoring result index is invalid";
              break;
            }
          if (batchFailed) {
            fatal(workerError.empty() ? "parallel scoring worker failed"
                                      : workerError);
            return;
          }

          const auto commitBegan = std::chrono::steady_clock::now();
          for (ParallelBatchItem &item : items) {
            SearchState &child = item.state;
            FamilyFunnelCounters &funnel =
                familyFunnelByRound[round][item.actionFamily];
            bool schedulerPassed = false;
            bool schedulerRejected = false;
            if (item.kind == ParallelBatchItemKind::Rejected) {
              ++rejectedCount;
              const std::string rejectReason =
                  child.rejectReason.empty() ? "unsupported_or_unknown_action"
                                             : child.rejectReason;
              ++funnel.reject;
              ++funnel.rejectReasons[
                  (item.materializationRejected ? "materialize:" :
                                                   "candidate:") +
                  StringRef(rejectReason)
                      .take_front(StringRef(rejectReason).find(':'))
                      .str()];
              appendRejected(archive, child,
                             rejectReason,
                             round);
              ++funnel.archive;
            } else if (item.kind == ParallelBatchItemKind::Duplicate) {
              ++rejectedCount;
              if (!mergeDuplicate(child, canonical, functionName, archive,
                                  outputDir, candidatesDir, nextSerial, error)) {
                fatal(error);
                return;
              }
              ++funnel.archive;
            } else {
              ParallelScoreResult &result = results[item.jobIndex];
              // Every production invocation consumes one budget unit, even
              // when the unchanged scheduler rejects the candidate.
              ++scoredCount;
              ++productionSchedulerCalls;
              ++roundScoredCount;
              ++funnel.schedulerCalls;
              if (item.freshCostPrepared) {
                ++funnel.freshCostScored;
                item.freshCostScored = true;
              }
              if (result.scored) {
                child.schedule = std::move(result.schedule);
                child.score = child.schedule.makespan;
                child.scored = true;
                child.rejectReason.clear();
                schedulerPassed = true;
                ++funnel.schedulerPass;
                ++successfulFamilyScores[item.actionFamily];
              } else {
                child.rejectReason = "production_scheduler_rejected";
                if (!result.error.empty())
                  child.rejectReason += ":" + result.error;
                child.scored = false;
                schedulerRejected = true;
                ++funnel.schedulerReject;
                ++funnel.reject;
                ++funnel.rejectReasons[
                    "scheduler:production_scheduler_rejected"];
              }
              if (!registerCandidate(child, archive, seenKeys, rejectedCount,
                                     candidatesDir, error)) {
                fatal(error);
                return;
              }
              if (child.scored &&
                  !persistBestFamilyWitnesses(child, archive, outputDir,
                                              error)) {
                fatal(error);
                return;
              }
              archive.back().round = round;
              ++funnel.archive;
              if (child.scored) {
                generated.push_back(std::move(child));
                generated = selectBeam(std::move(generated), beamWidth,
                                       diversitySlots);
              }
            }
            const PendingNeighbor &attempt =
                pending[item.pendingIndex];
            json::Object eventResult{
                {"attempted", true},
                {"materialized", item.materialized},
                {"reject_reason", child.rejectReason},
                {"unique", item.unique}, {"duplicate", item.duplicate},
                {"fresh_cost_prepared", item.freshCostPrepared},
                {"fresh_cost_scored", item.freshCostScored},
                {"cache_reused", item.cacheReused},
                {"cost_shape_cache_hits",
                 static_cast<int64_t>(item.costShapeCacheHits)},
                {"cost_shape_cache_misses",
                 static_cast<int64_t>(item.costShapeCacheMisses)},
                {"scheduler_calls",
                 item.kind == ParallelBatchItemKind::Score ? 1 : 0},
                {"scheduler_pass", schedulerPassed},
                {"scheduler_reject", schedulerRejected},
                {"archive", true}};
            appendFamilyFunnelCandidateEvent(
                round, item.actionFamily, child.id, child.parentId,
                attempt.action, child.actionHistory, std::move(eventResult));
            ++actionsSinceCheckpoint;
          }
          parallelCommitMilliseconds +=
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - commitBegan)
                  .count();
          pendingCursor = cursor;
          const bool pause =
              pauseAfterCandidates > 0 &&
              scoredCount >= static_cast<uint64_t>(pauseAfterCandidates);
          if (actionsSinceCheckpoint >=
                  static_cast<uint64_t>(checkpointActions) ||
              pause) {
            if (!saveCheckpoint(
                    checkpointPath, round, nextSerial, scoredCount,
                    rejectedCount, canonicalFactKey, archive, beam, generated,
                    pending, pendingCursor, pause ? "explicit-pause" : "",
                    error)) {
              fatal(error);
              return;
            }
            actionsSinceCheckpoint = 0;
          }
          if (pause) {
            stopReason = "explicit-pause";
            break;
          }
        }
      }
      if (stopReason == "explicit-pause" ||
          scoredCount >= static_cast<uint64_t>(maxCandidates))
        break;
      for (size_t index = static_cast<size_t>(pendingCursor);
           index < pending.size(); ++index) {
        const PendingNeighbor &unattempted = pending[index];
        FamilyFunnelCounters &funnel =
            familyFunnelByRound[round][unattempted.action.family];
        ++funnel.pendingUnattempted;
        ++funnel.pendingUnattemptedReasons["round_score_quota"];
        funnel.pendingUnattemptedActionSignatures.push_back(
            actionSignature(unattempted.action));
        if (unattempted.parent >= beam.size()) {
          fatal("pending neighbor parent outside beam while recording funnel frontier");
          return;
        }
        appendPendingFamilyFunnelEvent(round, unattempted,
                                       beam[unattempted.parent],
                                       "round_score_quota");
      }
      ++round; pending.clear(); pendingCursor = 0;
      remainingPendingFamilies.clear();
      successfulFamilyScores.clear();
      roundScoredCount = 0;
      roundScoreLimit = perRoundScoreQuota(
          scoredCount, static_cast<uint64_t>(maxCandidates.getValue()),
          maxRounds.getValue() - round);
      if (generated.empty()) { stopReason = "no-new-legal-candidates"; break; }
      beam = selectBeam(std::move(generated), beamWidth, diversitySlots);
      if (!retainCanonicalBeam(beam, archive, error)) { fatal(error); return; }
      generated.clear();
      if (!saveCheckpoint(checkpointPath, round, nextSerial, scoredCount, rejectedCount,
          canonicalFactKey, archive, beam, generated, pending, pendingCursor, "", error)) { fatal(error); return; }
    }
    if (stopReason.empty()) stopReason = round >= maxRounds ? "max-rounds" : "max-unique-candidates";
    elapsedMilliseconds = previousElapsed + std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - began).count();
    if (!saveCheckpoint(checkpointPath, round, nextSerial, scoredCount, rejectedCount,
        canonicalFactKey, archive, beam, generated, pending, pendingCursor, stopReason, error)) { fatal(error); return; }
    if (!writeOutputs(outputDir, checkpointPath, functionName, stage,
        sourceRepository, sourceCommit, architecturePath, protocolFile,
        maxRounds, maxCandidates, beamWidth, diversitySlots, round,
        scoredCount, rejectedCount, costCacheLoaded, !previousWinner.empty(),
        !historicalNativeWinner.empty(),
        archive, stopReason, error)) { fatal(error); return; }
  }

private:
  int64_t perCgraRows = 0;
  int64_t perCgraCols = 0;
  std::string effectiveArchitectureContract;

  std::string checkpointDirectoryForModel() const {
    if (modelPath.empty())
      return {};
    StringRef parent = llvm::sys::path::parent_path(modelPath);
    return parent.empty() ? std::string(".") : parent.str();
  }

  static void appendRejected(std::vector<ArchiveRecord> &archive,
                             const SearchState &state, StringRef reason,
                             int64_t round) {
    ArchiveRecord record;
    record.id = state.id;
    record.parentId = state.parentId;
    record.key = state.key; record.factKey = state.factKey;
    record.path = state.path;
    record.actionHistory = portableActionHistory(state.actionHistory);
    markActionHistoryUnknown(record.actionHistory,
                             "rejected_candidate_history_not_materialized");
    record.shapes = state.shapes;
    record.rejectReason = reason.str();
    record.round = round;
    record.valid = false;
    archive.push_back(std::move(record));
  }

  static void appendAlternatePath(std::vector<ArchiveRecord> &archive,
                                  const SearchState &state) {
    for (ArchiveRecord &record : archive) {
      if (record.valid && record.key == state.key) {
        record.alternatePaths.push_back(state.path);
        return;
      }
    }
  }

  bool collectHistoryTaskNames(ModuleOp module, StringRef functionName,
                               std::vector<std::string> &names,
                               std::string &error) const {
    FailureOr<func::FuncOp> function =
        selectTaskFunction(module, functionName, error);
    if (failed(function))
      return false;
    FailureOr<SmallVector<TaskMetadata>> metadata =
        collectAnalyticalTaskMetadata(*function, error);
    if (failed(metadata))
      return false;
    names.clear();
    for (const TaskMetadata &task : *metadata)
      names.push_back(task.name);
    return !names.empty();
  }

  bool buildFissionCanonical(ModuleOp canonical,
                             ArrayRef<NeighborhoodAction> fissionActions,
                             OwningOpRef<ModuleOp> &effectiveCanonical,
                             std::string &reason, std::string &error) {
    reason.clear();
    error.clear();
    if (fissionActions.empty()) {
      effectiveCanonical = canonical.clone();
      if (!effectiveCanonical) {
        reason = "canonical_module_clone_failed";
        return false;
      }
      return true;
    }
    if (stageNumber(stage) != 6 || !preparedTaskflowSource) {
      reason = "fission_history_requires_full_joint_fission_source";
      return false;
    }
    std::string actionKey;
    SmallVector<mlir::amoeba::neura::TaskflowFissionAction> sourceActions;
    std::set<std::string> targets;
    for (const NeighborhoodAction &action : fissionActions) {
      if (action.family != "fission" || action.canonicalReset ||
          !action.shapeTask.empty() || action.shapeRows != 0 ||
          action.shapeCols != 0 || action.primitives.size() != 1) {
        reason = "malformed_typed_fission_action";
        return false;
      }
      const NeighborhoodPrimitive &primitive = action.primitives.front();
      if (primitive.kind != "fission" || primitive.firstTask.empty() ||
          !primitive.secondTask.empty() || !primitive.mode.empty() ||
          primitive.axis != 0 || primitive.factor != 1 ||
          primitive.leftNodes.empty() ||
          !targets.insert(primitive.firstTask).second) {
        reason = "malformed_or_repeated_typed_fission_target";
        return false;
      }
      for (size_t index = 1; index < primitive.leftNodes.size(); ++index)
        if (primitive.leftNodes[index - 1] >= primitive.leftNodes[index]) {
          reason = "fission_left_nodes_must_be_strictly_increasing";
          return false;
        }
      actionKey += actionSignature(action) + "\n";
      mlir::amoeba::neura::TaskflowFissionAction sourceAction;
      sourceAction.function = functionName.getValue();
      sourceAction.task = primitive.firstTask;
      sourceAction.leftNodes.append(primitive.leftNodes.begin(),
                                    primitive.leftNodes.end());
      sourceActions.push_back(std::move(sourceAction));
    }
    auto cached = effectiveFissionCanonicalText.find(actionKey);
    if (cached != effectiveFissionCanonicalText.end()) {
      effectiveCanonical = parseSourceString<ModuleOp>(
          cached->second, canonical.getContext());
      if (!effectiveCanonical || failed(verify(effectiveCanonical->getOperation()))) {
        reason = "cached_fission_canonical_does_not_parse";
        return false;
      }
      return true;
    }
    if (!preparedSourceCanonicalVerified ||
        failed(verifyTaskflowFissionCanonicalLowering(
            preparedTaskflowSource.get(), canonical, error))) {
      reason = "prepared_fission_source_no_longer_matches_canonical:" + error;
      error.clear();
      return false;
    }
    OwningOpRef<ModuleOp> sourceCandidate = preparedTaskflowSource->clone();
    for (const auto &action : sourceActions) {
      if (failed(mlir::amoeba::neura::materializeTaskflowFission(
              sourceCandidate.get(), action.function, action.task,
              action.leftNodes, error))) {
        reason = "typed_fission_source_materialization_failed:" + error;
        error.clear();
        return false;
      }
    }
    if (!sourceCandidate ||
        failed(mlir::amoeba::neura::verifyTaskflowFissionReplay(
            preparedTaskflowSource.get(), sourceCandidate.get(), sourceActions,
            error))) {
      reason = "typed_fission_source_replay_failed:" + error;
      error.clear();
      return false;
    }
    if (failed(lowerPreparedTaskflowFissionSource(
            sourceCandidate.get(), effectiveCanonical, error))) {
      reason = "typed_fission_canonical_lowering_failed:" + error;
      error.clear();
      return false;
    }
    if (!prepareInitialActiveTransferFacts(effectiveCanonical.get(),
                                           functionName, error)) {
      reason = "fission_active_transfer_seed_unproven:" + error;
      error.clear();
      return false;
    }
    std::string witness =
        printTaskflowFissionModuleWitness(effectiveCanonical.get());
    effectiveFissionCanonicalText.emplace(actionKey, witness);
    return true;
  }

  bool effectiveCanonicalForHistory(
      ModuleOp canonical, const SearchState::TypedActionHistory &history,
      OwningOpRef<ModuleOp> &effectiveCanonical, StringRef stageName,
      std::string &reason, std::string &error) {
    if (!history.known) {
      reason = "typed_action_history_not_authenticated";
      return false;
    }
    if (history.fissionActions.empty()) {
      if (stageNumber(stageName) == 6) {
        if (!preparedSourceCanonicalVerified || !preparedTaskflowSource) {
          reason = "prepared_source_no_longer_matches_canonical:" + error;
          error.clear();
          return false;
        }
      }
      effectiveCanonical = canonical.clone();
      if (!effectiveCanonical) {
        reason = "canonical_module_clone_failed";
        return false;
      }
      return true;
    }
    return buildFissionCanonical(canonical, history.fissionActions,
                                 effectiveCanonical, reason, error);
  }

  static void recordTypedAction(
      SearchState::TypedActionHistory &history,
      const NeighborhoodAction &action,
      ArrayRef<std::string> beforeNames,
      ArrayRef<std::string> afterNames) {
    if (action.canonicalReset) {
      history.actions.clear();
      history.dependencies.clear();
      history.taskProducers.clear();
    }

    std::set<uint64_t> dependencies;
    auto addDependency = [&](StringRef task) {
      auto producer = history.taskProducers.find(task.str());
      if (producer != history.taskProducers.end() &&
          producer->second < history.actions.size())
        dependencies.insert(producer->second);
    };
    if (!action.canonicalReset) {
      if (!action.shapeTask.empty())
        addDependency(action.shapeTask);
      for (const NeighborhoodPrimitive &primitive : action.primitives) {
        if (!primitive.firstTask.empty())
          addDependency(primitive.firstTask);
        if (!primitive.secondTask.empty())
          addDependency(primitive.secondTask);
      }
    }

    const uint64_t actionIndex = history.actions.size();
    history.actions.push_back(action);
    history.dependencies.emplace_back(dependencies.begin(), dependencies.end());

    if (action.canonicalReset)
      return;
    std::set<std::string> after(afterNames.begin(), afterNames.end());
    std::set<std::string> before(beforeNames.begin(), beforeNames.end());
    for (auto it = history.taskProducers.begin();
         it != history.taskProducers.end();) {
      if (!after.count(it->first))
        it = history.taskProducers.erase(it);
      else
        ++it;
    }
    for (const std::string &name : afterNames)
      if (!before.count(name))
        history.taskProducers[name] = actionIndex;

    // Some source-owned edits preserve a task name while changing its
    // lineage. Attribute that surviving name to the latest typed primitive;
    // replay, rather than attrs, remains the authority for this relation.
    for (const NeighborhoodPrimitive &primitive : action.primitives) {
      if (!primitive.firstTask.empty() && after.count(primitive.firstTask))
        history.taskProducers[primitive.firstTask] = actionIndex;
      if (!primitive.secondTask.empty() && after.count(primitive.secondTask))
        history.taskProducers[primitive.secondTask] = actionIndex;
    }
  }

  bool replayTypedActionHistory(
      const SearchState::TypedActionHistory &history, ModuleOp canonical,
      StringRef functionName, StringRef outputDirectory, uint64_t serial,
      SearchState &replayed, std::string &reason,
      std::string &fatalError) {
    reason.clear();
    fatalError.clear();
    if (!history.known || history.canonicalFactKey.empty() ||
        history.initialShapes.empty()) {
      reason = "typed_action_history_not_authenticated";
      return false;
    }
    OwningOpRef<ModuleOp> effectiveCanonical;
    if (!effectiveCanonicalForHistory(canonical, history, effectiveCanonical,
                                      stage, reason, fatalError)) {
      if (fatalError.empty() && reason.empty())
        reason = "typed_action_history_effective_canonical_unavailable";
      return false;
    }
    for (const NeighborhoodAction &fission : history.fissionActions) {
      if (stageNumber(stage) != 6 ||
          !llvm::any_of(canonicalFissionActions,
                        [&](const NeighborhoodAction &allowed) {
            return actionSignature(allowed) == actionSignature(fission);
          })) {
        reason = "typed_action_history_fission_action_not_in_complete_cut_set:" +
                 fission.label;
        return false;
      }
    }
    replayed = SearchState();
    replayed.module = effectiveCanonical->clone();
    replayed.shapes = history.initialShapes;
    replayed.actionHistory = portableActionHistory(history);
    for (const NeighborhoodShape &shape : replayed.shapes)
      if (shape.rows <= 0 || shape.cols <= 0 || shape.rows > 4 ||
          shape.cols > 4 || shape.rows * shape.cols > 4) {
        reason = "typed_action_history_initial_shape_outside_area_four";
        return false;
      }
    if (!prepareInitialActiveTransferFacts(replayed.module.get(), functionName,
                                           reason)) {
      reason = "typed_action_history_initial_proof_unsupported:" + reason;
      return false;
    }
    std::vector<std::string> names;
    std::string taskError;
    FailureOr<func::FuncOp> canonicalFunction =
        selectTaskFunction(replayed.module.get(), functionName, taskError);
    if (failed(canonicalFunction)) {
      reason = "typed_action_history_initial_function_missing:" + taskError;
      return false;
    }
    FailureOr<SmallVector<TaskMetadata>> canonicalMetadata =
        collectAnalyticalTaskMetadata(*canonicalFunction, taskError);
    if (!collectHistoryTaskNames(replayed.module.get(), functionName, names,
                                 taskError) ||
        failed(canonicalMetadata) ||
        !sameTaskSet(*canonicalMetadata, replayed.shapes)) {
      reason = "typed_action_history_initial_shapes_do_not_cover_canonical_tasks";
      if (!taskError.empty())
        reason += ":" + taskError;
      return false;
    }
    if (!prepareCandidateKey(replayed, functionName, outputDirectory, serial,
                             fatalError)) {
      if (!fatalStorageError(fatalError)) {
        reason = "typed_action_history_canonical_facts_unavailable:" +
                 fatalError;
        fatalError.clear();
      }
      return false;
    }
    const bool replayPendingFissionKey =
        !history.fissionActions.empty() &&
        history.canonicalFactKey == "source-replay-pending";
    if (replayed.key.empty() ||
        (!replayPendingFissionKey &&
         replayed.factKey != history.canonicalFactKey)) {
      reason = replayed.rejectReason.empty()
                   ? "typed_action_history_canonical_fact_key_mismatch"
                   : "typed_action_history_canonical_facts_unknown:" +
                         replayed.rejectReason;
      return false;
    }

    replayed.actionHistory = portableActionHistory(history);
    if (replayPendingFissionKey)
      replayed.actionHistory.canonicalFactKey = replayed.factKey;
    replayed.actionHistory.actions.clear();
    replayed.actionHistory.dependencies.clear();
    replayed.actionHistory.taskProducers.clear();
    for (const NeighborhoodAction &action : history.actions) {
      if (action.family == "lineage-replacement" ||
          action.family == "fission") {
        reason = "typed_action_history_contains_nonreplayable_lineage_replacement";
        return false;
      }
      std::vector<std::string> beforeNames;
      if (!collectHistoryTaskNames(replayed.module.get(), functionName,
                                   beforeNames, taskError)) {
        reason = "typed_action_history_task_set_unknown:" + taskError;
        return false;
      }
      std::string applyReason, diagnostic;
      if (!applyTrustedNeighborhoodAction(
              replayed.module.get(), effectiveCanonical.get(), functionName, action,
              replayed.shapes, applyReason, diagnostic,
              static_cast<unsigned>(maxPartitionFactor.getValue()))) {
        reason = "typed_action_history_replay_unsupported:";
        if (applyReason.empty())
          reason += "unknown_action";
        else
          reason += applyReason;
        if (!diagnostic.empty())
          reason += ":" + diagnostic;
        if (fatalStorageError(reason)) {
          fatalError = reason;
          reason.clear();
        }
        return false;
      }
      FailureOr<func::FuncOp> function =
          selectTaskFunction(replayed.module.get(), functionName, taskError);
      if (failed(function)) {
        reason = "typed_action_history_replay_function_missing:" + taskError;
        return false;
      }
      if (!refreshActiveTransferFactsAfterTrustedRewrite(*function, taskError)) {
        reason = "typed_action_history_replay_proof_unsupported:" + taskError;
        return false;
      }
      std::vector<NeighborhoodShape> priorShapes = replayed.shapes;
      synchronizeShapes(*function, priorShapes, replayed.shapes);
      replayed.key.clear();
      replayed.factKey.clear();
      replayed.rejectReason.clear();
      if (!prepareCandidateKey(replayed, functionName, outputDirectory, serial,
                               fatalError)) {
        if (!fatalStorageError(fatalError)) {
          reason = "typed_action_history_replay_facts_unavailable:" +
                   fatalError;
          fatalError.clear();
        }
        return false;
      }
      if (replayed.key.empty()) {
        reason = "typed_action_history_replay_facts_unknown:" +
                 replayed.rejectReason;
        return false;
      }
      std::vector<std::string> afterNames;
      if (!collectHistoryTaskNames(replayed.module.get(), functionName,
                                   afterNames, taskError)) {
        reason = "typed_action_history_result_task_set_unknown:" + taskError;
        return false;
      }
      recordTypedAction(replayed.actionHistory, action, beforeNames,
                        afterNames);
    }
    return true;
  }

  bool authenticateTypedActionHistory(SearchState &candidate,
                                      ModuleOp canonical,
                                      StringRef functionName,
                                      StringRef outputDirectory,
                                      uint64_t serial,
                                      std::string &fatalError) {
    if (!candidate.actionHistory.known)
      return true;
    SearchState replayed;
    std::string reason;
    if (!replayTypedActionHistory(candidate.actionHistory, canonical,
                                  functionName, outputDirectory, serial,
                                  replayed, reason, fatalError)) {
      if (!fatalError.empty())
        return false;
      markActionHistoryUnknown(candidate.actionHistory,
                               "typed_action_history_replay_failed:" + reason);
      return true;
    }
    if (replayed.key != candidate.key ||
        replayed.factKey != candidate.factKey ||
        shapeSignature(replayed.shapes) != shapeSignature(candidate.shapes) ||
        neighborhoodReplaySourceText(replayed.module.get()) !=
            neighborhoodReplaySourceText(candidate.module.get())) {
      markActionHistoryUnknown(candidate.actionHistory,
                               "typed_action_history_canonical_replay_mismatch");
      return true;
    }
    candidate.actionHistory = std::move(replayed.actionHistory);
    return true;
  }

  bool replayLineageReplacement(
      SearchState &child, const SearchState &parent, ModuleOp canonical,
      StringRef functionName, StringRef outputDirectory,
      const NeighborhoodAction &action, uint64_t serial,
      std::string &reason, std::string &fatalError) {
    if (!parent.actionHistory.known) {
      reason = "lineage_replacement_requires_authenticated_path_replay";
      return false;
    }
    if (action.primitives.size() != 1 ||
        action.primitives.front().firstTask.empty()) {
      reason = "lineage_replacement_target_unknown";
      return false;
    }
    const std::string &target = action.primitives.front().firstTask;
    auto producer = parent.actionHistory.taskProducers.find(target);
    if (producer == parent.actionHistory.taskProducers.end() ||
        producer->second >= parent.actionHistory.actions.size() ||
        parent.actionHistory.dependencies.size() !=
            parent.actionHistory.actions.size()) {
      reason = "lineage_replacement_target_has_no_authenticated_producer:" +
               target;
      return false;
    }
    std::set<uint64_t> removed{producer->second};
    bool changed = true;
    while (changed) {
      changed = false;
      for (uint64_t index = 0;
           index < parent.actionHistory.dependencies.size(); ++index) {
        if (removed.count(index))
          continue;
        for (uint64_t dependency :
             parent.actionHistory.dependencies[index]) {
          if (removed.count(dependency)) {
            removed.insert(index);
            changed = true;
            break;
          }
        }
      }
    }
    SearchState::TypedActionHistory filtered =
        portableActionHistory(parent.actionHistory);
    filtered.actions.clear();
    for (uint64_t index = 0; index < parent.actionHistory.actions.size();
         ++index)
      if (!removed.count(index))
        filtered.actions.push_back(parent.actionHistory.actions[index]);
    SearchState replayed;
    std::string replayReason;
    if (!replayTypedActionHistory(filtered, canonical, functionName,
                                  outputDirectory, serial, replayed,
                                  replayReason, fatalError)) {
      if (fatalError.empty())
        reason = "lineage_replacement_canonical_replay_unsupported:" +
                 replayReason;
      return false;
    }
    child.module = std::move(replayed.module);
    child.shapes = std::move(replayed.shapes);
    child.key = std::move(replayed.key);
    child.factKey = std::move(replayed.factKey);
    child.rejectReason.clear();
    child.actionHistory = std::move(replayed.actionHistory);
    return true;
  }

  enum class NeighborMaterialization { Applied, Rejected, Fatal };

  NeighborMaterialization materializeNeighbor(
      SearchState &child, const SearchState &parent, ModuleOp canonical,
      StringRef functionName, StringRef outputDirectory,
      const NeighborhoodAction &action, StringRef canonicalFactKey,
      uint64_t serial, std::string &reason, std::string &diagnostic,
      std::string &fatalError) {
    child.actionHistory = parent.actionHistory;
    if (action.family == "fission") {
      if (stageNumber(stage) != 6 || !preparedTaskflowSource ||
          !parent.actionHistory.known ||
          action.primitives.size() != 1 ||
          !llvm::any_of(canonicalFissionActions,
                        [&](const NeighborhoodAction &allowed) {
            return actionSignature(allowed) == actionSignature(action);
          })) {
        reason = "fission_requires_authenticated_source_history";
        return NeighborMaterialization::Rejected;
      }
      reason = fissionHistoryConflictReason(parent.actionHistory, action);
      if (!reason.empty()) {
        reason = "fission_history_conflict:" + reason;
        return NeighborMaterialization::Rejected;
      }
      for (const NeighborhoodAction &prior :
           parent.actionHistory.fissionActions)
        if (prior.primitives.size() == 1 &&
            prior.primitives.front().firstTask ==
                action.primitives.front().firstTask) {
            reason = "fission_target_was_already_split";
            return NeighborMaterialization::Rejected;
        }
      child.actionHistory.fissionActions.push_back(action);
      child.actionHistory.canonicalFactKey = "source-replay-pending";
      OwningOpRef<ModuleOp> effectiveCanonical;
      if (!buildFissionCanonical(canonical,
                                 child.actionHistory.fissionActions,
                                 effectiveCanonical, reason, fatalError)) {
        if (!fatalError.empty())
          return NeighborMaterialization::Fatal;
        return NeighborMaterialization::Rejected;
      }
      OwningOpRef<ModuleOp> priorCanonical;
      if (!buildFissionCanonical(canonical,
                                 parent.actionHistory.fissionActions,
                                 priorCanonical, reason, fatalError)) {
        if (!fatalError.empty())
          return NeighborMaterialization::Fatal;
        reason = "fission_prior_source_history_unavailable:" + reason;
        return NeighborMaterialization::Rejected;
      }
      FailureOr<func::FuncOp> priorFunction =
          selectTaskFunction(priorCanonical.get(), functionName, diagnostic);
      if (failed(priorFunction)) {
        reason = "fission_prior_source_function_missing:" + diagnostic;
        return NeighborMaterialization::Rejected;
      }
      FailureOr<SmallVector<TaskMetadata>> priorMetadata =
          collectAnalyticalTaskMetadata(*priorFunction, diagnostic);
      if (failed(priorMetadata) ||
          !sameTaskSet(*priorMetadata, parent.actionHistory.initialShapes)) {
        reason = "fission_parent_initial_shapes_do_not_cover_source_history";
        if (!diagnostic.empty())
          reason += ":" + diagnostic;
        return NeighborMaterialization::Rejected;
      }
      FailureOr<func::FuncOp> splitFunction =
          selectTaskFunction(effectiveCanonical.get(), functionName,
                             diagnostic);
      if (failed(splitFunction)) {
        reason = "fission_lowered_task_function_missing";
        return NeighborMaterialization::Rejected;
      }
      std::vector<NeighborhoodShape> rebasedInitialShapes =
          initialShapes(*splitFunction);
      if (rebasedInitialShapes.empty()) {
        reason = "fission_lowered_task_graph_is_empty";
        return NeighborMaterialization::Rejected;
      }
      std::map<std::string, NeighborhoodShape> priorInitialShapes;
      for (const NeighborhoodShape &shape :
           parent.actionHistory.initialShapes)
        if (!priorInitialShapes.emplace(shape.task, shape).second) {
          reason = "fission_parent_initial_shapes_repeat_task";
          return NeighborMaterialization::Rejected;
        }
      // Rebase only by stable task name. New fission children retain their
      // source-created 1x1 shapes, while unaffected bootstrap choices such as
      // Ray Task13 survive before ordinary typed actions are replayed.
      for (NeighborhoodShape &shape : rebasedInitialShapes) {
        auto prior = priorInitialShapes.find(shape.task);
        if (prior == priorInitialShapes.end()) {
          shape.rows = 1;
          shape.cols = 1;
        } else {
          shape.rows = prior->second.rows;
          shape.cols = prior->second.cols;
        }
      }
      FailureOr<SmallVector<TaskMetadata>> splitMetadata =
          collectAnalyticalTaskMetadata(*splitFunction, diagnostic);
      if (failed(splitMetadata) ||
          !sameTaskSet(*splitMetadata, rebasedInitialShapes)) {
        reason = "fission_rebased_initial_shapes_do_not_cover_task_set";
        if (!diagnostic.empty())
          reason += ":" + diagnostic;
        return NeighborMaterialization::Rejected;
      }
      child.actionHistory.initialShapes = std::move(rebasedInitialShapes);
      child.actionHistory.dependencies.clear();
      child.actionHistory.taskProducers.clear();
      SearchState replayed;
      if (!replayTypedActionHistory(child.actionHistory, canonical,
                                    functionName, outputDirectory, serial,
                                    replayed, reason, fatalError)) {
        if (!fatalError.empty())
          return NeighborMaterialization::Fatal;
        reason = "fission_complete_history_replay_failed:" + reason;
        return NeighborMaterialization::Rejected;
      }
      child.module = std::move(replayed.module);
      child.shapes = std::move(replayed.shapes);
      child.key = std::move(replayed.key);
      child.factKey = std::move(replayed.factKey);
      child.rejectReason.clear();
      child.actionHistory = std::move(replayed.actionHistory);
      return NeighborMaterialization::Applied;
    }
    if (action.family == "lineage-replacement") {
      if (!replayLineageReplacement(child, parent, canonical, functionName,
                                    outputDirectory, action, serial, reason,
                                    fatalError))
        return fatalError.empty() ? NeighborMaterialization::Rejected
                                  : NeighborMaterialization::Fatal;
      return NeighborMaterialization::Applied;
    }

    std::vector<std::string> beforeNames;
    if (!collectHistoryTaskNames(child.module.get(), functionName, beforeNames,
                                 diagnostic)) {
      reason = "action_source_task_set_unknown";
      return NeighborMaterialization::Rejected;
    }
    OwningOpRef<ModuleOp> effectiveCanonical;
    if (!effectiveCanonicalForHistory(canonical, parent.actionHistory,
                                      effectiveCanonical, stage, reason,
                                      fatalError)) {
      if (!fatalError.empty())
        return NeighborMaterialization::Fatal;
      return NeighborMaterialization::Rejected;
    }
    if (!applyTrustedNeighborhoodAction(
            child.module.get(), effectiveCanonical.get(), functionName, action,
            child.shapes, reason, diagnostic,
            static_cast<unsigned>(maxPartitionFactor.getValue()))) {
      if (fatalStorageError(diagnostic) || fatalStorageError(reason)) {
        fatalError = reason + ":" + diagnostic;
        return NeighborMaterialization::Fatal;
      }
      return NeighborMaterialization::Rejected;
    }
    FailureOr<func::FuncOp> function =
        selectTaskFunction(child.module.get(), functionName, diagnostic);
    if (failed(function)) {
      reason = "task_function_missing_after_action";
      return NeighborMaterialization::Rejected;
    }
    if (!refreshActiveTransferFactsAfterTrustedRewrite(*function, diagnostic)) {
      reason = "active_transfer_proof_unsupported:" + diagnostic;
      return NeighborMaterialization::Rejected;
    }
    std::vector<NeighborhoodShape> priorShapes = child.shapes;
    synchronizeShapes(*function, priorShapes, child.shapes);
    child.key.clear();
    child.factKey.clear();
    child.rejectReason.clear();
    if (!prepareCandidateKey(child, functionName, outputDirectory, serial,
                             fatalError))
      return NeighborMaterialization::Fatal;
    if (child.key.empty()) {
      reason = child.rejectReason.empty()
                   ? "graph_facts_unknown"
                   : child.rejectReason;
      return NeighborMaterialization::Rejected;
    }

    if (action.canonicalReset) {
      SearchState::TypedActionHistory resetHistory;
      resetHistory.known = true;
      resetHistory.fissionActions = parent.actionHistory.fissionActions;
      resetHistory.canonicalFactKey =
          parent.actionHistory.canonicalFactKey.empty()
              ? canonicalFactKey.str()
              : parent.actionHistory.canonicalFactKey;
      FailureOr<func::FuncOp> baselineFunction =
          selectTaskFunction(effectiveCanonical.get(), functionName,
                             diagnostic);
      if (failed(baselineFunction)) {
        reason = "action_reset_effective_canonical_function_missing";
        return NeighborMaterialization::Rejected;
      }
      resetHistory.initialShapes = initialShapes(*baselineFunction);
      child.actionHistory = std::move(resetHistory);
    }
    if (child.actionHistory.known) {
      std::vector<std::string> afterNames;
      if (!collectHistoryTaskNames(child.module.get(), functionName, afterNames,
                                   diagnostic)) {
        markActionHistoryUnknown(child.actionHistory,
                                 "typed_action_history_result_task_set_unknown");
      } else {
        recordTypedAction(child.actionHistory, action, beforeNames,
                          afterNames);
      }
    }
    return NeighborMaterialization::Applied;
  }

  bool prepareInitialActiveTransferFacts(ModuleOp module,
                                         StringRef functionName,
                                         std::string &error) const {
    if (activeTransferArguments.empty())
      return true;
    FailureOr<func::FuncOp> function =
        selectTaskFunction(module, functionName, error);
    if (failed(function))
      return false;
    for (unsigned index = 0; index < function->getNumArguments(); ++index) {
      if (!isa<MemRefType>(function->getArgument(index).getType()))
        continue;
      bool requested = llvm::is_contained(activeTransferArguments, index);
      if (hasStaticActiveTransferShapeFacts(*function, index)) {
        if (failed(verifyStaticActiveTransferShapeProof(*function, index,
                                                        &error))) {
          error = (Twine("active-transfer seed has stale or malformed facts "
                         "for argument ") +
                   Twine(index) + ": " + error)
                      .str();
          return false;
        }
      } else if (requested &&
                 failed(rederiveAndStoreStaticActiveTransferShapeProof(
                     *function, index, &error))) {
        error = (Twine("cannot derive active-transfer facts for argument ") +
                 Twine(index) + ": " + error)
                    .str();
        return false;
      }
    }
    return true;
  }

  bool refreshActiveTransferFactsAfterTrustedRewrite(func::FuncOp function,
                                                     std::string &error) const {
    if (activeTransferArguments.empty())
      return true;
    if (!function) {
      error = "active-transfer rewrite has no selected function";
      return false;
    }
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      if (!isa<MemRefType>(function.getArgument(index).getType()))
        continue;
      if (!llvm::is_contained(activeTransferArguments, index) &&
          !hasStaticActiveTransferShapeFacts(function, index))
        continue;
      if (failed(rederiveAndStoreStaticActiveTransferShapeProof(function, index,
                                                                &error))) {
        error = (Twine("cannot refresh active-transfer facts for argument ") +
                 Twine(index) + ": " + error)
                    .str();
        return false;
      }
    }
    return true;
  }

  bool applyTrustedNeighborhoodAction(ModuleOp cloned, ModuleOp canonical,
                                      StringRef selectedFunction,
                                      const NeighborhoodAction &action,
                                      std::vector<NeighborhoodShape> &shapes,
                                      std::string &reason,
                                      std::string &diagnostic,
                                      unsigned maxPartitionFactor) const {
    if (!sourceDomainEnabled)
      return applyActiveTransferNeighborhoodAction(cloned, canonical,
          selectedFunction, action, shapes, reason, diagnostic, maxPartitionFactor);
    std::string error;
    FailureOr<func::FuncOp> domainCanonical =
        selectTaskFunction(canonical, selectedFunction, error);
    auto child = selectTaskFunction(cloned, selectedFunction, error);
    if (failed(domainCanonical) || failed(child) ||
        failed(verifySourceIterationDomainPartition(
            *domainCanonical, *child, error))) {
      reason = "source_iteration_domain_seed_unproven"; diagnostic = error;
      return false;
    }
    auto suspend = [](ModuleOp module) {
      module.walk([&](taskflow::TaskflowTaskOp task) {
        task->removeAttr(kSourceIterationDomainAttr);
        task->removeAttr(kSourceIterationControlBindingAttr);
        task->removeAttr(kSourceIterationSourceControlBindingAttr);
        task->removeAttr(kSourceIterationPartitionProofAttr);
        task->removeAttr(kSourceIterationCapturePendingAttr);
      });
    };
    // Nested materializers see only current counters during the atomic edit;
    // no stale predecessor proof can authorize a half-built shard group.
    OwningOpRef<ModuleOp> actionCanonical = canonical.clone();
    suspend(cloned); suspend(*actionCanonical);
    if (!applyActiveTransferNeighborhoodAction(cloned, *actionCanonical,
          selectedFunction, action, shapes, reason, diagnostic, maxPartitionFactor))
      return false;
    child = selectTaskFunction(cloned, selectedFunction, error);
    if (failed(child) || failed(proveAndRefreshSourceIterationDomainPartition(
            *domainCanonical, *child, error))) {
      reason = "source_iteration_domain_rewrite_unproven"; diagnostic = error;
      return false;
    }
    return true;
  }

  bool applyActiveTransferNeighborhoodAction(ModuleOp cloned, ModuleOp canonical,
                                      StringRef selectedFunction,
                                      const NeighborhoodAction &action,
                                      std::vector<NeighborhoodShape> &shapes,
                                      std::string &reason,
                                      std::string &diagnostic,
                                      unsigned maxPartitionFactor) const {
    if (activeTransferArguments.empty())
      return applyNeighborhoodAction(cloned, canonical, selectedFunction,
                                     action, shapes, reason, diagnostic,
                                     maxPartitionFactor);

    std::string selectError;
    FailureOr<func::FuncOp> function =
        selectTaskFunction(cloned, selectedFunction, selectError);
    if (failed(function)) {
      reason = "active_transfer_seed_invalid";
      diagnostic = selectError;
      return false;
    }

    // Do not let an intermediate verifier mistake proof metadata from the
    // parent graph for facts about a graph partway through this atomic action.
    // Keep logical_transfer_shape intact: it is the conservative capacity
    // fallback used while completion joins are being rebuilt.
    std::set<unsigned> savedArguments;
    for (unsigned index = 0; index < function->getNumArguments(); ++index) {
      if (!hasStaticActiveTransferShapeFacts(*function, index))
        continue;
      std::string proofError;
      if (failed(verifyStaticActiveTransferShapeProof(*function, index,
                                                      &proofError))) {
        reason = "active_transfer_seed_invalid";
        diagnostic =
            (Twine("argument ") + Twine(index) + ": " + proofError).str();
        return false;
      }
      savedArguments.insert(index);
    }

    // Canonical-reset actions clone the canonical module inside the action
    // controller, so verify and suspend its derived facts too. The original
    // canonical module remains untouched and available to later actions.
    OwningOpRef<ModuleOp> proofFreeCanonical;
    if (action.canonicalReset) {
      FailureOr<func::FuncOp> canonicalFunction =
          selectTaskFunction(canonical, selectedFunction, selectError);
      if (failed(canonicalFunction)) {
        reason = "active_transfer_seed_invalid";
        diagnostic = selectError;
        return false;
      }
      for (unsigned index = 0; index < canonicalFunction->getNumArguments();
           ++index) {
        if (!hasStaticActiveTransferShapeFacts(*canonicalFunction, index))
          continue;
        std::string proofError;
        if (failed(verifyStaticActiveTransferShapeProof(*canonicalFunction,
                                                        index, &proofError))) {
          reason = "active_transfer_seed_invalid";
          diagnostic =
              (Twine("canonical argument ") + Twine(index) + ": " + proofError)
                  .str();
          return false;
        }
        savedArguments.insert(index);
      }
      proofFreeCanonical = canonical.clone();
      if (!proofFreeCanonical) {
        reason = "candidate_clone_failed";
        return false;
      }
      canonicalFunction = selectTaskFunction(proofFreeCanonical.get(),
                                             selectedFunction, selectError);
      if (failed(canonicalFunction)) {
        reason = "active_transfer_seed_invalid";
        diagnostic = selectError;
        return false;
      }
      for (unsigned index : savedArguments) {
        if (index >= canonicalFunction->getNumArguments())
          continue;
        canonicalFunction->removeArgAttr(index, "amoeba.active_transfer_shape");
        canonicalFunction->removeArgAttr(
            index, "amoeba.active_transfer_capacity_shape");
        canonicalFunction->removeArgAttr(index, "amoeba.active_transfer_proof");
      }
    }

    for (unsigned index : savedArguments) {
      if (index >= function->getNumArguments())
        continue;
      function->removeArgAttr(index, "amoeba.active_transfer_shape");
      function->removeArgAttr(index, "amoeba.active_transfer_capacity_shape");
      function->removeArgAttr(index, "amoeba.active_transfer_proof");
    }

    // Suspend facts through the complete action, including every primitive in
    // a combined edit, since its nested pass managers verify intermediate IR.
    ModuleOp canonicalForAction =
        proofFreeCanonical ? proofFreeCanonical.get() : canonical;
    if (!applyNeighborhoodAction(cloned, canonicalForAction, selectedFunction,
                                 action, shapes, reason, diagnostic,
                                 maxPartitionFactor))
      return false;

    function = selectTaskFunction(cloned, selectedFunction, selectError);
    if (failed(function)) {
      reason = "active_transfer_proof_unsupported";
      diagnostic = selectError;
      return false;
    }
    for (unsigned index : savedArguments) {
      if (index >= function->getNumArguments()) {
        reason = "active_transfer_proof_unsupported";
        diagnostic =
            "trusted neighborhood action changed the function signature";
        return false;
      }
      std::string proofError;
      if (failed(rederiveAndStoreStaticActiveTransferShapeProof(
              *function, index, &proofError))) {
        reason = "active_transfer_proof_unsupported";
        diagnostic = (Twine("cannot restore argument ") + Twine(index) +
                      " after trusted neighborhood action: " + proofError)
                         .str();
        return false;
      }
    }

    if (failed(verify(cloned.getOperation()))) {
      reason = "active_transfer_module_verification_failed";
      diagnostic =
          "module verification failed after restoring active-transfer facts";
      return false;
    }
    return true;
  }

  bool collectVerifiedActiveTransferSignature(func::FuncOp function,
                                              std::string &signature,
                                              std::string &error) const {
    signature.clear();
    if (activeTransferArguments.empty())
      return true;
    for (unsigned requested : activeTransferArguments) {
      if (!hasStaticActiveTransferShapeFacts(function, requested)) {
        error = (Twine("active-transfer proof is missing for argument ") +
                 Twine(requested))
                    .str();
        return false;
      }
    }
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      if (!hasStaticActiveTransferShapeFacts(function, index))
        continue;
      std::string verificationError;
      if (failed(verifyStaticActiveTransferShapeProof(function, index,
                                                      &verificationError))) {
        error = (Twine("active-transfer proof is stale for argument ") +
                 Twine(index) + ": " + verificationError)
                    .str();
        return false;
      }
      StaticActiveTransferShapeProof proof =
          analyzeStaticActiveTransferShape(function, index);
      if (activeTransferRequireProven &&
          llvm::is_contained(activeTransferArguments, index) && !proof.proven) {
        error =
            (Twine("active-transfer proof is unknown for required argument ") +
             Twine(index))
                .str();
        return false;
      }
      Attribute attribute =
          function.getArgAttr(index, "amoeba.active_transfer_proof");
      std::string proofText;
      llvm::raw_string_ostream proofStream(proofText);
      attribute.print(proofStream);
      proofStream.flush();
      signature += (Twine("argument=") + Twine(index) +
                    ";bytes=" + Twine(proofText.size()) + ":" + proofText + ";")
                       .str();
    }
    return true;
  }

  bool verifyActiveTransferWitnessInKey(StringRef signature, StringRef key,
                                        std::string &error) const {
    if (activeTransferArguments.empty())
      return true;
    if (signature.empty()) {
      error = "active-transfer replay has no verified proof signature";
      return false;
    }
    const std::string witnessBytes =
        kActiveTransferProofSchema.str() + "\n" + signature.str();
    auto witness = activeTransferWitnessIds.find(witnessBytes);
    if (witness == activeTransferWitnessIds.end()) {
      error = "active-transfer replay proof has no interned exact witness";
      return false;
    }
    const std::string keySuffix = "|active-transfer-proof:" + witness->second;
    if (!key.ends_with(keySuffix)) {
      error = "replay candidate key does not bind its active-transfer proof "
              "witness";
      return false;
    }
    auto path = witnessPaths.find(witness->second);
    std::string persistedBytes;
    if (path == witnessPaths.end() ||
        !readFileBytes(path->second, persistedBytes, error)) {
      if (error.empty())
        error = "active-transfer replay proof witness file is missing";
      return false;
    }
    if (persistedBytes != witnessBytes) {
      error = "active-transfer replay proof witness bytes differ from "
              "re-derived facts";
      return false;
    }
    return true;
  }

  bool prepareCandidateKey(SearchState &state, StringRef function,
                           StringRef outputDirectory, uint64_t serial,
                           std::string &error) {
    if (!state.key.empty())
      return true;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(state.module.get(), function, error);
    if (failed(selected)) {
      state.rejectReason = "task_function_missing";
      error.clear();
      return true;
    }
    if (sourceDomainEnabled) {
      func::FuncOp domainCanonical = sourceDomainCanonical;
      OwningOpRef<ModuleOp> effectiveCanonical;
      if (state.actionHistory.known &&
          !state.actionHistory.fissionActions.empty()) {
        ModuleOp originalCanonical =
            sourceDomainCanonical->getParentOfType<ModuleOp>();
        std::string replayReason;
        if (!effectiveCanonicalForHistory(originalCanonical,
                                          state.actionHistory,
                                          effectiveCanonical, stage,
                                          replayReason, error)) {
          state.rejectReason = "fission_canonical_replay_unproven:" +
                               replayReason;
          error.clear();
          return true;
        }
        FailureOr<func::FuncOp> selectedBaseline =
            selectTaskFunction(effectiveCanonical.get(), function, error);
        if (failed(selectedBaseline)) {
          state.rejectReason = "fission_source_function_missing:" + error;
          error.clear();
          return true;
        }
        domainCanonical = *selectedBaseline;
      }
      if (failed(verifySourceIterationDomainPartition(
              domainCanonical, *selected, error))) {
        state.rejectReason = "source_iteration_domain_unproven:" + error;
        error.clear(); return true;
      }
    }
    FailureOr<SmallVector<TaskMetadata>> metadata =
        collectAnalyticalTaskMetadata(*selected, error);
    if (failed(metadata)) {
      if (fatalStorageError(error))
        return false;
      state.rejectReason = "task_metadata_unknown";
      if (!error.empty())
        state.rejectReason += ":" + error;
      error.clear();
      return true;
    }
    if (!sameTaskSet(*metadata, state.shapes)) {
      state.rejectReason = "shape_state_does_not_cover_task_set";
      error.clear();
      return true;
    }
    std::string activeTransferSignature;
    if (!collectVerifiedActiveTransferSignature(*selected,
                                                activeTransferSignature,
                                                error)) {
      state.rejectReason = "active_transfer_proof_unsupported:" + error;
      error.clear();
      return true;
    }
    std::string factsDiagnostic;
    if (!extractStructuralFacts(state.module.get(), function, outputDirectory,
                                serial, state.factKey, state.rejectReason,
                                factsDiagnostic)) {
      if (fatalStorageError(factsDiagnostic) ||
          fatalStorageError(state.rejectReason)) {
        error = factsDiagnostic.empty() ? state.rejectReason : factsDiagnostic;
        return false;
      }
      if (state.rejectReason.empty())
        state.rejectReason = "graph_facts_unknown";
      if (!factsDiagnostic.empty())
        state.rejectReason += ":" + factsDiagnostic;
      error.clear();
      return true;
    }
    // Intern typed bodies and graph descriptors once. Archive rows and seen
    // keys refer to exact dictionary IDs, never repeated multi-KB IR strings.
    json::Array bodies, lineages;
    (*selected).walk([&](taskflow::TaskflowTaskOp task) {
      std::string ledger; llvm::raw_string_ostream ledgerStream(ledger);
      if (auto attr = task->getAttr("amoeba.neighborhood.partition_lineage.v1")) attr.print(ledgerStream);
      ledgerStream.flush(); lineages.push_back(ledger);
    });
    bool bodyFailure = false;
    (*selected).walk([&](::mlir::neura::KernelOp kernel) {
      if (bodyFailure) return;
      std::string body; llvm::raw_string_ostream stream(body);
      Operation *bodyClone = kernel->clone();
      bodyClone->removeAttr("kernel_symname"); bodyClone->removeAttr("sym_name");
      bodyClone->print(stream, OpPrintingFlags().printGenericOpForm()); stream.flush();
      bodyClone->destroy();
      std::string id;
      if (!internWitness(body, true, id, error)) { bodyFailure = true; return; }
      bodies.push_back(id);
    });
    if (bodyFailure) return false;
    std::string graphWitness = jsonText(json::Value(json::Object{
        {"canonical_structural_facts", state.factKey}, {"typed_kernel_body_ids", std::move(bodies)},
        {"partition_lineages", std::move(lineages)}}));
    if (!internWitness(graphWitness, false, state.factKey, error)) return false;
    std::string activeTransferWitnessID;
    if (!activeTransferSignature.empty()) {
      const std::string activeTransferWitness =
          kActiveTransferProofSchema.str() + "\n" + activeTransferSignature;
      if (!internActiveTransferWitness(activeTransferWitness,
                                       activeTransferWitnessID, error))
        return false;
    }
    state.key = state.factKey + "|" + shapeSignature(state.shapes);
    if (!activeTransferWitnessID.empty())
      state.key += "|active-transfer-proof:" + activeTransferWitnessID;
    (*selected)->setAttr("amoeba.graph_variant_id",
                        StringAttr::get(state.module->getContext(), state.factKey));
    return true;
  }

  CandidateCostCacheObservation observeCandidateCostCache(
      const SearchState &state, StringRef function,
      StringRef canonicalFactKey, bool costCacheLoaded) const {
    CandidateCostCacheObservation observation;
    if (state.factKey.empty())
      return observation;
    observation.known = true;
    if (state.factKey == canonicalFactKey && costCacheLoaded) {
      observation.reused = true;
      return observation;
    }
    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(state.module.get(), function, error);
    if (failed(selected)) {
      observation.known = false;
      return observation;
    }
    FailureOr<SmallVector<TaskMetadata>> metadata =
        collectAnalyticalTaskMetadata(*selected, error);
    if (failed(metadata)) {
      observation.known = false;
      return observation;
    }
    observation.graphCostKey = state.factKey;
    for (const TaskMetadata &task : *metadata)
      observation.graphCostKey += "|" + task.name;
    observation.reused = graphCostCaches.count(observation.graphCostKey) ||
                         graphCostPaths.count(observation.graphCostKey);
    return observation;
  }

  // Perform every mutable, source-owned part of candidate evaluation before
  // handing a job to a worker.  TaskShapeCostCache::get updates hit/miss
  // counters, and graph catalogue creation may publish files, so this helper
  // intentionally runs only on the owner thread.  It stops immediately before
  // the production scheduler call made by evaluateCandidate.
  bool prepareParallelCandidate(
      SearchState &state, ModuleOp canonical, StringRef function,
      StringRef stageName, StringRef outputDirectory,
      StringRef canonicalFactKey, bool costCacheLoaded,
      TaskShapeCostCache &costCache, uint64_t serial, std::string &error) {
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(state.module.get(), function, error);
    if (failed(selected))
      return false;
    FailureOr<SmallVector<TaskMetadata>> metadata =
        collectAnalyticalTaskMetadata(*selected, error);
    if (failed(metadata))
      return false;
    state.rejectReason.clear();
    state.scored = false;
    if (!sameTaskSet(*metadata, state.shapes)) {
      error.clear();
      state.rejectReason = "shape_state_does_not_cover_task_set";
      return true;
    }
    if (state.key.empty() &&
        !prepareCandidateKey(state, function, outputDirectory, serial, error))
      return false;
    if (state.key.empty()) {
      error.clear();
      return true;
    }

    state.choices.clear();
    state.durations.clear();
    state.costs.clear();
    state.schedule = {};
    TaskShapeCostCache *activeCostCache = nullptr;
    if (state.factKey == canonicalFactKey && costCacheLoaded) {
      activeCostCache = &costCache;
      state.costPath = canonicalCostPath;
    } else {
      // Catalogues bind source task names, while candidate dedup uses ordered
      // graph facts. Names can differ across equivalent rewrite paths, so bind
      // the memoization key to the current catalogue task vocabulary as well.
      std::string graphCostKey = state.factKey;
      for (const auto &task : *metadata)
        graphCostKey += "|" + task.name;
      auto cached = graphCostCaches.find(graphCostKey);
      if (cached == graphCostCaches.end()) {
        auto knownPath = graphCostPaths.find(graphCostKey);
        if (knownPath != graphCostPaths.end()) {
          state.costPath = knownPath->second;
        } else {
          if (!diskGuard(error))
            return false;
          OwningOpRef<ModuleOp> predictorModule = state.module->clone();
          auto predictorFunction =
              selectTaskFunction(*predictorModule, function, error);
          if (failed(predictorFunction))
            return false;
          const std::string catalogueId =
              "graph-catalogue-" + std::to_string(graphCostPaths.size());
          bool fatalPredictorFailure = false;
          if (!runCurrentCostPredictor(
                  *predictorModule, *predictorFunction, function, catalogueId,
                  outputDirectory, modelPath, cachePath,
                  checkpointDirectoryForModel(),
                  effectiveArchitectureContract, architecturePath,
                  perCgraRows, perCgraCols, sourceRepository, sourceCommit,
                  architectureTransferCatalog, modelNamespace,
                  diagnosticIICeiling,
                  bootstrapSupportedShapes, state.costPath, error,
                  fatalPredictorFailure)) {
            if (fatalPredictorFailure || fatalStorageError(error))
              return false;
            state.rejectReason = "current_graph_cost_unknown:" + error;
            error.clear();
            return true;
          }
          graphCostPaths[graphCostKey] = state.costPath;
        }
        auto graphId = readCatalogGraphId(state.costPath);
        auto oracle = std::make_unique<TaskShapeCostCache>();
        std::string expectedWitness =
            neighborhoodReplaySourceText(state.module.get());
        if (!graphId || !oracle->load(state.costPath, function, *metadata,
                                      sourceRepository, sourceCommit,
                                      architecturePath, *graphId, error,
                                      bootstrapSupportedShapes,
                                      expectedWitness)) {
          if (error.empty())
            error = "generated cost catalogue has no bound graph ID";
          error = "generated_cost_catalogue_binding_failed:" + error;
          return false;
        }
        cached = graphCostCaches.emplace(graphCostKey, std::move(oracle)).first;
      }
      state.costPath = graphCostPaths[graphCostKey];
      activeCostCache = cached->second.get();
    }

    std::map<std::string, const NeighborhoodShape *> shapeMap;
    for (const NeighborhoodShape &shape : state.shapes)
      shapeMap[shape.task] = &shape;
    for (const TaskMetadata &task : *metadata) {
      const NeighborhoodShape *shape = shapeMap[task.name];
      if (!shape || shape->rows <= 0 || shape->cols <= 0 ||
          shape->rows > std::numeric_limits<int64_t>::max() / shape->cols ||
          shape->rows * shape->cols > 4) {
        state.rejectReason = "invalid_or_oversized_shape";
        error.clear();
        return true;
      }
      int64_t mapperRows = 0, mapperCols = 0;
      if (!cgraShapeToMapperTileShape(shape->rows, shape->cols, perCgraRows,
                                      perCgraCols, mapperRows, mapperCols)) {
        state.rejectReason = "invalid_architecture_mapper_shape";
        error.clear();
        return true;
      }
      RectShape rectangle{shape->rows, shape->cols, mapperRows, mapperCols};
      TaskShapeChoice choice{task.name, task.tripCount, rectangle};
      std::string costError;
      const uint64_t oldHits = activeCostCache ? activeCostCache->hits() : 0;
      const uint64_t oldMisses = activeCostCache ? activeCostCache->misses() : 0;
      const TaskShapeCost *cost = activeCostCache
                                      ? activeCostCache->get(choice, costError)
                                      : nullptr;
      if (activeCostCache) {
        costCacheHits += activeCostCache->hits() - oldHits;
        costCacheMisses += activeCostCache->misses() - oldMisses;
      }
      if (cost && !requireCostIICeiling(cost, diagnosticIICeiling, error))
        return false;
      if (!cost || !cost->supported) {
        if (cost && cost->modelDomainUnsupported)
          state.rejectReason = "unsupported_model_domain_shape:" +
                               cost->unsupportedReason;
        else
          state.rejectReason = costError.empty()
                                   ? "unsupported_task_shape_cost"
                                   : "unsupported_task_shape_cost:" + costError;
        error.clear();
        return true;
      }
      std::optional<int64_t> duration =
          durationFromCost(*cost, task.tripCount, costError);
      if (!duration || *duration <= 0) {
        state.rejectReason = "invalid_task_shape_duration";
        error.clear();
        return true;
      }
      state.choices.push_back(choice);
      state.durations.push_back(*duration);
      state.costs.push_back(*cost);
    }
    if (state.choices.empty()) {
      state.rejectReason = "empty_candidate";
      error.clear();
    }
    (void)canonical;
    (void)stageName;
    return true;
  }

  bool evaluateCandidate(SearchState &state, ModuleOp canonical,
                         StringRef function, StringRef stageName,
                         StringRef outputDirectory, StringRef candidatesDirectory,
                         StringRef canonicalFactKey, bool costCacheLoaded,
                         TaskShapeCostCache &costCache,
                         int64_t round, uint64_t &scoredCount,
                         bool chargeRoundQuota,
                         uint64_t serial, std::string &error) {
    FailureOr<func::FuncOp> selected = selectTaskFunction(
        state.module.get(), function, error);
    if (failed(selected))
      return false;
    FailureOr<SmallVector<TaskMetadata>> metadata =
        collectAnalyticalTaskMetadata(*selected, error);
    if (failed(metadata))
      return false;
    if (!sameTaskSet(*metadata, state.shapes)) {
      error.clear();
      state.rejectReason = "shape_state_does_not_cover_task_set";
      state.scored = false;
      return true;
    }
    if (state.key.empty() &&
        !prepareCandidateKey(state, function, outputDirectory, serial, error)) {
      state.scored = false;
      return false;
    }
    if (state.key.empty()) {
      state.scored = false;
      error.clear();
      return true;
    }

    state.choices.clear();
    state.durations.clear();
    state.costs.clear();
    state.schedule = {};
    TaskShapeCostCache *activeCostCache = nullptr;
    if (state.factKey == canonicalFactKey && costCacheLoaded) {
      activeCostCache = &costCache; state.costPath = canonicalCostPath;
    } else {
      // Catalogues bind source task names, while candidate dedup uses ordered
      // graph facts. Names can differ across equivalent rewrite paths, so bind
      // the memoization key to the current catalogue task vocabulary as well.
      std::string graphCostKey = state.factKey;
      for (const auto &task : *metadata) graphCostKey += "|" + task.name;
      auto cached = graphCostCaches.find(graphCostKey);
      if (cached == graphCostCaches.end()) {
        auto knownPath = graphCostPaths.find(graphCostKey);
        if (knownPath != graphCostPaths.end()) state.costPath = knownPath->second;
        else {
          if (!diskGuard(error)) return false;
          OwningOpRef<ModuleOp> predictorModule = state.module->clone();
          auto predictorFunction = selectTaskFunction(*predictorModule, function, error);
          if (failed(predictorFunction)) return false;
          const std::string catalogueId = "graph-catalogue-" + std::to_string(graphCostPaths.size());
          bool fatalPredictorFailure = false;
          if (!runCurrentCostPredictor(*predictorModule, *predictorFunction,
                  function, catalogueId, outputDirectory, modelPath, cachePath,
                  checkpointDirectoryForModel(),
                  effectiveArchitectureContract, architecturePath,
                  perCgraRows, perCgraCols, sourceRepository, sourceCommit,
                  architectureTransferCatalog, modelNamespace,
                  diagnosticIICeiling,
                  bootstrapSupportedShapes, state.costPath, error,
                  fatalPredictorFailure)) {
            if (fatalPredictorFailure || fatalStorageError(error)) return false;
            state.rejectReason = "current_graph_cost_unknown:" + error;
            state.scored = false; error.clear(); return true;
          }
          graphCostPaths[graphCostKey] = state.costPath;
        }
        auto graphId = readCatalogGraphId(state.costPath);
        auto oracle = std::make_unique<TaskShapeCostCache>();
        std::string expectedWitness =
            neighborhoodReplaySourceText(state.module.get());
        if (!graphId || !oracle->load(state.costPath, function, *metadata,
            sourceRepository, sourceCommit, architecturePath, *graphId, error,
            bootstrapSupportedShapes, expectedWitness)) {
          if (error.empty()) error = "generated cost catalogue has no bound graph ID";
          error = "generated_cost_catalogue_binding_failed:" + error;
          state.scored = false; return false;
        }
        cached = graphCostCaches.emplace(graphCostKey, std::move(oracle)).first;
      }
      state.costPath = graphCostPaths[graphCostKey]; activeCostCache = cached->second.get();
    }

    std::map<std::string, const NeighborhoodShape *> shapeMap;
    for (const NeighborhoodShape &shape : state.shapes)
      shapeMap[shape.task] = &shape;
    for (const TaskMetadata &task : *metadata) {
      const NeighborhoodShape *shape = shapeMap[task.name];
      if (!shape || shape->rows <= 0 || shape->cols <= 0 ||
          shape->rows > std::numeric_limits<int64_t>::max() / shape->cols ||
          shape->rows * shape->cols > 4) {
        state.rejectReason = "invalid_or_oversized_shape";
        state.scored = false;
        error.clear();
        return true;
      }
      int64_t mapperRows = 0, mapperCols = 0;
      if (!cgraShapeToMapperTileShape(shape->rows, shape->cols, perCgraRows,
                                      perCgraCols, mapperRows, mapperCols)) {
        state.rejectReason = "invalid_architecture_mapper_shape";
        state.scored = false;
        error.clear();
        return true;
      }
      RectShape rectangle{shape->rows, shape->cols, mapperRows, mapperCols};
      TaskShapeChoice choice{task.name, task.tripCount, rectangle};
      std::string costError;
      const uint64_t oldHits = activeCostCache ? activeCostCache->hits() : 0;
      const uint64_t oldMisses = activeCostCache ? activeCostCache->misses() : 0;
      const TaskShapeCost *cost = activeCostCache
                                      ? activeCostCache->get(choice, costError)
                                      : nullptr;
      if (activeCostCache) {
        costCacheHits += activeCostCache->hits() - oldHits;
        costCacheMisses += activeCostCache->misses() - oldMisses;
      }
      if (cost && !requireCostIICeiling(cost, diagnosticIICeiling, error))
        return false;
      if (!cost || !cost->supported) {
        if (cost && cost->modelDomainUnsupported)
          state.rejectReason = "unsupported_model_domain_shape:" +
                               cost->unsupportedReason;
        else
          state.rejectReason = costError.empty()
                                   ? "unsupported_task_shape_cost"
                                   : "unsupported_task_shape_cost:" + costError;
        state.scored = false;
        error.clear();
        return true;
      }
      std::optional<int64_t> duration =
          durationFromCost(*cost, task.tripCount, costError);
      if (!duration || *duration <= 0) {
        state.rejectReason = "invalid_task_shape_duration";
        state.scored = false;
        error.clear();
        return true;
      }
      state.choices.push_back(choice);
      state.durations.push_back(*duration);
      state.costs.push_back(*cost);
    }
    if (state.choices.empty()) {
      state.rejectReason = "empty_candidate";
      state.scored = false;
      error.clear();
      return true;
    }
    bool fixedDispatch = parseStageFixed(stageName);
    std::string scheduleError;
    ++scoredCount;
    ++productionSchedulerCalls;
    if (chargeRoundQuota)
      ++roundScoredCount;
    if (!scheduleWithAmoebaProductionPass(
            state.module.get(), *selected, state.choices, state.durations,
            state.id, fixedDispatch, state.schedule, scheduleError)) {
      state.rejectReason = "production_scheduler_rejected";
      if (!scheduleError.empty())
        state.rejectReason += ":" + scheduleError;
      state.scored = false;
      error.clear();
      return true;
    }
    state.score = state.schedule.makespan;
    state.scored = true;
    (void)round;
    (void)candidatesDirectory;
    return true;
  }

  bool registerCandidate(SearchState &state,
                                std::vector<ArchiveRecord> &archive,
                                std::set<std::string> &seenKeys,
                                uint64_t &rejectedCount,
                                StringRef candidatesDirectory,
                                std::string &error) {
    if (!state.scored || state.key.empty()) {
      ++rejectedCount;
      appendRejected(archive, state,
                     state.rejectReason.empty() ? "candidate_unknown"
                                                : state.rejectReason,
                     0);
      return true;
    }
    if (!state.keyReserved && !seenKeys.insert(state.key).second) {
      ++rejectedCount;
      appendAlternatePath(archive, state);
      return true;
    }
    ArchiveRecord record;
    record.id = state.id;
    record.key = state.key;
    record.factKey = state.factKey;
    record.parentId = state.parentId;
    record.path = state.path;
    record.actionHistory = portableActionHistory(state.actionHistory);
    record.shapes = state.shapes;
    record.choices = state.choices;
    record.durations = state.durations;
    record.costs = state.costs;
    record.schedule = state.schedule;
    record.costPath = state.costPath;
    record.control = state.control;
    record.controlRoles = state.controlRoles;
    if (record.control && record.controlRoles.empty()) {
      error = "stage control candidate has no named control role";
      return false;
    }
    {
      std::set<std::string> uniqueRoles;
      for (const std::string &role : record.controlRoles)
        if (role.empty() || !uniqueRoles.insert(role).second) {
          error = "stage control candidate has an empty or duplicate control role";
          return false;
        }
    }
    auto function = selectTaskFunction(state.module.get(), "", error);
    if (failed(function)) return false;
    auto graph = (*function)->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
    record.graphId = graph ? graph.getValue().str() : state.id;
    record.score = state.score;
    record.valid = true;
    record.round = state.path.empty() ? 0 : 1;
    // Keep materialized IR bounded to candidates that can currently enter
    // native top-5.  A candidate that is outside this cutoff can never enter
    // it later because archive scores are immutable and only more rows are
    // added.  This avoids a full 20k-module snapshot while retaining every
    // final replay input.
    bool keepModule = state.control || rankedArchivePrefix.size() < 5;
    if (!keepModule) {
      keepModule = archiveLess(record, archive[rankedArchivePrefix.back()]);
    }
    if (keepModule) {
      std::string candidatePath =
          candidatesDirectory.str() + "/" + state.id + ".mlir";
      if (!diskGuard(error) || !writeModule(state.module.get(), candidatePath, error))
        return false;
      record.candidatePath = candidatePath;
    }
    for (const std::string &role : record.controlRoles) {
      ControlRoleProvenance provenance;
      provenance.candidatePath = record.candidatePath;
      provenance.path = state.path;
      provenance.actionHistory =
          portableActionHistory(state.actionHistory);
      record.controlRoleProvenance.emplace(role, std::move(provenance));
    }
    const std::vector<size_t> previousPrefix = rankedArchivePrefix;
    archive.push_back(std::move(record));
    insertRankedArchiveIndex(archive.size() - 1, archive);
    // Only a previous top-5 entry can lose a non-control replay snapshot.
    // Controls stay live regardless of their predicted rank.
    for (size_t index : previousPrefix) {
      auto &old = archive[index];
      if (old.control || old.candidatePath.empty() ||
          llvm::is_contained(rankedArchivePrefix, index)) continue;
      obsoleteCandidatePaths.push_back(old.candidatePath); old.candidatePath.clear();
      dirtyJournalIndices.insert(index);
    }
    return true;
  }

  void insertRankedArchiveIndex(size_t index,
                               const std::vector<ArchiveRecord> &archive) {
    auto position = std::lower_bound(
        rankedArchivePrefix.begin(), rankedArchivePrefix.end(), index,
        [&](size_t left, size_t right) {
          return archiveLess(archive[left], archive[right]);
        });
    rankedArchivePrefix.insert(position, index);
    if (rankedArchivePrefix.size() > 5) rankedArchivePrefix.pop_back();
  }

  bool retainCanonicalBeam(std::vector<SearchState> &beam,
                           ArrayRef<ArchiveRecord> archive,
                           std::string &error) {
    const ArchiveRecord *identity = nullptr;
    for (const auto &record : archive)
      if (record.valid && llvm::is_contained(record.controlRoles,
                                            std::string("identity"))) {
        identity = &record;
        break;
      }
    if (!identity || identity->candidatePath.empty()) {
      error = "canonical identity control lacks its immutable source snapshot";
      return false;
    }
    for (const auto &state : beam)
      if (state.key == identity->key) return true;
    SearchState state;
    state.module = parseSourceFile<ModuleOp>(identity->candidatePath, &getContext());
    if (!state.module) {
      error = "canonical identity source snapshot no longer parses";
      return false;
    }
    state.id = identity->id; state.key = identity->key;
    state.factKey = identity->factKey; state.parentId = identity->parentId;
    state.path = identity->path; state.shapes = identity->shapes;
    state.actionHistory = portableActionHistory(identity->actionHistory);
    state.choices = identity->choices; state.durations = identity->durations;
    state.costs = identity->costs; state.schedule = identity->schedule;
    state.costPath = identity->costPath; state.score = identity->score;
    state.control = identity->control; state.controlRoles = identity->controlRoles;
    state.keyReserved = true; state.scored = true;
    // The cost-selected prefix has width-diversity entries. When the beam is
    // full its last entry belongs to the diversity suffix, so reserving this
    // slot never evicts one of the twelve cost-selected candidates. The
    // identity is cloned from its archive witness without another score or
    // budget debit, and can replace or withdraw every historical decision.
    if (beam.size() >= static_cast<size_t>(beamWidth.getValue())) beam.pop_back();
    beam.push_back(std::move(state));
    return true;
  }

  static std::vector<SearchState> selectBeam(std::vector<SearchState> candidates,
                                             int64_t width,
                                             int64_t diversity) {
    std::stable_sort(candidates.begin(), candidates.end(), stateLess);
    std::vector<SearchState> selected;
    std::vector<size_t> indices;
    std::set<std::string> keys, configurations;
    const size_t costSlots = static_cast<size_t>(std::max<int64_t>(0, width - diversity));
    auto configuration = [](const SearchState &state) {
      return (state.path.empty() ? std::string("identity") : state.path.back()) +
             "|" + shapeSignature(state.shapes);
    };
    for (size_t index = 0; index < candidates.size() && indices.size() < costSlots; ++index) {
      if (!candidates[index].scored || candidates[index].key.empty() ||
          !keys.insert(candidates[index].key).second) continue;
      indices.push_back(index); configurations.insert(configuration(candidates[index]));
    }
    // Diversity selects complete legal candidates independently of immediate
    // improvement. Choose distinct action paths or parameter configurations;
    // this affects expansion only, never the global archive/native ranking.
    for (size_t index = 0; index < candidates.size() && indices.size() < size_t(width); ++index) {
      if (!candidates[index].scored || candidates[index].key.empty() ||
          keys.count(candidates[index].key) ||
          !configurations.insert(configuration(candidates[index])).second) continue;
      keys.insert(candidates[index].key); indices.push_back(index);
    }
    for (size_t index = 0; index < candidates.size() && indices.size() < size_t(width); ++index) {
      if (!candidates[index].scored || candidates[index].key.empty() ||
          !keys.insert(candidates[index].key).second) continue;
      indices.push_back(index);
    }
    // Decide all indices before moving. A moved-from state must never enter
    // the beam through a second traversal of the same vector.
    for (size_t index : indices) selected.push_back(std::move(candidates[index]));
    return selected;
  }

  static int stageNumber(StringRef value) {
    if (value == "S1" || value == "s1" || value == "shape-only" || value == "shape") return 1;
    if (value == "S2" || value == "s2" || value == "shape-temporal") return 2;
    if (value == "S3" || value == "s3" || value == "shape-temporal-replica") return 3;
    if (value == "S4" || value == "s4" || value == "shape-temporal-replica-tiling") return 4;
    if (value == "S5" || value == "s5" || value == "full-joint") return 5;
    if (value == "S6" || value == "s6" || value == "full-joint-fission" ||
        value == "shape-temporal-replica-tiling-fusion-fission") return 6;
    return 0;
  }

  static bool hasCompleteSharedReplicaOutputRegion(
      Operation *operation) {
    auto task = dyn_cast<taskflow::TaskflowTaskOp>(operation);
    if (!task)
      return false;
    auto parent = task->getAttrOfType<StringAttr>(
        "amoeba.replica.parent_task");
    auto id = task->getAttrOfType<IntegerAttr>("amoeba.replica.id");
    auto count = task->getAttrOfType<IntegerAttr>("amoeba.replica.count");
    auto lowers = task->getAttrOfType<ArrayAttr>(
        "amoeba.tiling.output_region_lowers");
    auto uppers = task->getAttrOfType<ArrayAttr>(
        "amoeba.tiling.output_region_uppers");
    return parent && !parent.getValue().empty() && id && count &&
           id.getInt() >= 0 && count.getInt() > 1 &&
           id.getInt() < count.getInt() && lowers && uppers &&
           !lowers.empty() && lowers.size() == uppers.size();
  }

  static bool validateStageSeed(
      ModuleOp module, StringRef stageName,
      bool sourceDomainWillAuthenticateReplicaRegions,
      std::string &error) {
    const int number = stageNumber(stageName);
    bool valid = number != 0;
    module.walk([&](Operation *operation) {
      for (const NamedAttribute &attribute : operation->getAttrs()) {
        StringRef name = attribute.getName().strref();
        const bool sharedReplicaOutputRegion =
            sourceDomainWillAuthenticateReplicaRegions &&
            (name == "amoeba.tiling.output_region_lowers" ||
             name == "amoeba.tiling.output_region_uppers") &&
            hasCompleteSharedReplicaOutputRegion(operation);
        const bool outsideStage =
            (number < 6 && name.starts_with("amoeba.fission.")) ||
            (number < 3 && name.starts_with("amoeba.replica.")) ||
            (number < 4 && (name.starts_with("amoeba.neura.tiling.") ||
                           (name.starts_with("amoeba.tiling.") &&
                            !sharedReplicaOutputRegion) ||
                           name.starts_with("amoeba.semantic.k_block") ||
                           name == "amoeba.semantic.K_range")) ||
            (number < 5 && (name.starts_with("amoeba.neura.fusion.") ||
                           name.starts_with("amoeba.fusion.")));
        if (outsideStage) {
          valid = false;
          if (error.empty()) {
            if (sourceDomainWillAuthenticateReplicaRegions &&
                (name == "amoeba.tiling.output_region_lowers" ||
                 name == "amoeba.tiling.output_region_uppers") &&
                !hasCompleteSharedReplicaOutputRegion(operation))
              error = "replica output-region annotation lacks complete parent/id/count metadata";
            else
              error = "seed contains forbidden rewrite annotation " +
                      name.str() + " for cumulative stage " +
                      stageName.str();
          }
        }
      }
    });
    if (!valid && error.empty())
      error = "seed contains a rewrite dimension outside this cumulative stage";
    return valid;
  }

  bool buildBinding(ModuleOp canonical, std::string &text, std::string &error) {
    json::Object files;
    auto bind = [&](StringRef role, StringRef path, bool optional = false,
                    std::string *boundBytes = nullptr) {
      std::string bytes;
      if (!readFileBytes(path, bytes, error, optional)) return false;
      if (boundBytes)
        *boundBytes = bytes;
      files[role] = json::Object{{"path", path}, {"exact_bytes", std::move(bytes)}};
      return true;
    };
    if (!bind("protocol", protocolFile) || !bind("source_and_model_contract", sourceContractFile) ||
        !bind("architecture", architecturePath) || !bind("ensemble", modelPath, true) ||
        !bind("parent_costs", parentCostFile, true) ||
        !bind("architecture_transfer", architectureTransferCatalog, true) ||
        !bind("seed_manifest", seedManifest, true) || !bind("previous_winner", previousWinner, true)) return false;
    if (stageNumber(stage) == 6) {
      std::string currentPreparedSourceBytes;
      if (!readFileBytes(preparedSourceFile, currentPreparedSourceBytes,
                         error) ||
          currentPreparedSourceBytes != preparedTaskflowSourceBytes) {
        if (error.empty())
          error = "prepared source bytes changed after canonical authentication";
        return false;
      }
      files["prepared_taskflow_source"] = json::Object{
          {"path", preparedSourceFile.getValue()},
          {"exact_bytes", preparedTaskflowSourceBytes}};
    }
    if (!historicalNativeWinner.empty() &&
        !bind("historical_native_winner", historicalNativeWinner,
              /*optional=*/false, &historicalNativeWinnerBoundBytes))
      return false;
    auto protocolBytes = files.getObject("protocol")->getString("exact_bytes");
    auto value = json::parse(*protocolBytes);
    if (!value) { llvm::consumeError(value.takeError()); error = "bound protocol is not JSON"; return false; }
    auto object = value->getAsObject();
    auto search = object ? object->getObject("search") : nullptr;
    std::optional<StringRef> supportedShapeBootstrapPolicy =
        search ? search->getString("supported_shape_bootstrap_policy")
               : std::nullopt;
    if (!search || search->getInteger("max_rounds") != maxRounds.getValue() ||
        search->getInteger("max_unique_complete_candidates_scored") != maxCandidates.getValue() ||
        search->getInteger("beam_width") != beamWidth.getValue() ||
        search->getInteger("diversity_min_slots") != diversitySlots.getValue() ||
        search->getInteger("max_partition_factor").value_or(4) !=
            maxPartitionFactor.getValue()) {
      error = "search budgets do not match the bound common protocol"; return false;
    }
    std::optional<int64_t> protocolDiagnosticIICeiling =
        search->getInteger("diagnostic_ii_ceiling");
    if ((diagnosticIICeiling == 23 &&
         (!protocolDiagnosticIICeiling ||
          *protocolDiagnosticIICeiling != diagnosticIICeiling.getValue())) ||
        (diagnosticIICeiling == 20 && protocolDiagnosticIICeiling &&
         *protocolDiagnosticIICeiling != diagnosticIICeiling.getValue())) {
      error = "diagnostic II ceiling does not match the bound common protocol";
      return false;
    }
    if (bootstrapSupportedShapes &&
        (!supportedShapeBootstrapPolicy ||
         *supportedShapeBootstrapPolicy != kSupportedShapeBootstrapPolicy)) {
      error = "supported-shape bootstrap requires the matching policy in the "
              "bound common protocol";
      return false;
    }
    if (stageNumber(stage) == 6 &&
        search->getInteger("max_fission_actions_per_task").value_or(-1) !=
            maxFissionActionsPerTask.getValue()) {
      error = "fission cut cap does not match the bound common protocol";
      return false;
    }
    if (supportedShapeBootstrapPolicy &&
        *supportedShapeBootstrapPolicy != kSupportedShapeBootstrapPolicy) {
      error = "bound protocol names an unknown supported-shape bootstrap "
              "policy";
      return false;
    }
    json::Object objectBinding{
        {"schema", "orbit-neighborhood-exact-binding-v1"},
        {"canonical_ir", neighborhoodReplaySourceText(canonical)},
        {"files", std::move(files)}, {"function", functionName.getValue()},
        {"stage", stage.getValue()}, {"source_repository", sourceRepository.getValue()},
        {"source_commit", sourceCommit.getValue()}, {"model_namespace", modelNamespace.getValue()},
        {"cache_path", cachePath.getValue()},
        {"architecture_contract", effectiveArchitectureContract},
        {"max_rounds", maxRounds.getValue()}, {"max_candidates", maxCandidates.getValue()},
        {"beam_width", beamWidth.getValue()}, {"diversity_slots", diversitySlots.getValue()},
        {"scoring_workers", scoringWorkers.getValue()},
        {"max_partition_factor", maxPartitionFactor.getValue()},
        {"tie_key_policy", "cost-numeric-shape-schedule-graph-key-v1"},
        {"search_contract", kSearchSchema},
        {"family_funnel_contract", kFamilyFunnelSchema},
        {"checkpoint_contract",
         checkpointSchemaForCeiling(diagnosticIICeiling.getValue())}};
    if (diagnosticIICeiling == 23)
      objectBinding["diagnostic_ii_ceiling"] = diagnosticIICeiling.getValue();
    if (stageNumber(stage) == 6) {
      objectBinding["prepared_source_file"] = preparedSourceFile.getValue();
      objectBinding["prepared_source_exact_bytes"] =
          preparedTaskflowSourceBytes;
      objectBinding["prepared_source_lowering_pipeline"] =
          taskflowFissionCanonicalLoweringPipeline();
      objectBinding["max_fission_actions_per_task"] =
          maxFissionActionsPerTask.getValue();
      objectBinding["fission_source_replay"] =
          "orbit-taskflow-fission-source-replay-v2-ordinary-suffix-rebase";
    }
    // Network costs are independent of mapper-II cache entries, but the
    // production schedule depends on this exact resource. Bind its bytes
    // without its path so identical network documents share the same search
    // continuation identity even when staged at different locations.
    if (interTaskNetworkSpecOverrideBytes)
      objectBinding["inter_task_network_spec_override"] = json::Object{
          {"exact_bytes", *interTaskNetworkSpecOverrideBytes}};
    if (bootstrapSupportedShapes) {
      objectBinding["bootstrap_supported_shapes"] = true;
      objectBinding["supported_shape_bootstrap_policy"] =
          kSupportedShapeBootstrapPolicy.str();
    }
    if (!activeTransferArguments.empty()) {
      json::Array arguments;
      for (unsigned index : activeTransferArguments)
        arguments.push_back(static_cast<int64_t>(index));
      objectBinding["active_transfer_proof"] = json::Object{
          {"schema", kActiveTransferProofSchema},
          {"arguments_option", activeTransferArgumentsText.getValue()},
          {"arguments", std::move(arguments)},
          {"require_proven", activeTransferRequireProven.getValue()},
          {"witness_encoding", kActiveTransferWitnessEncoding}};
    }
    if (!historicalNativeWinner.empty()) {
      json::Array requiredControlRoles;
      requiredControlRoles.push_back("identity");
      if (!previousWinner.empty())
        requiredControlRoles.push_back("previous_stage_measured_winner");
      requiredControlRoles.push_back("historical_measured_winner");
      objectBinding["historical_native_winner_option"] =
          historicalNativeWinner.getValue();
      objectBinding["required_control_roles"] =
          std::move(requiredControlRoles);
    }
    text = jsonText(json::Value(std::move(objectBinding))) + "\n";
    return true;
  }

  bool internWitnessInNamespace(
      StringRef bytes, StringRef idPrefix,
      std::map<std::string, std::string> &dictionary, std::string &id,
      std::string &error) {
    auto existing = dictionary.find(bytes.str());
    if (existing != dictionary.end()) { id = existing->second; return true; }
    id = idPrefix.str() + std::to_string(dictionary.size());
    const std::string directory = outputDir + "/witnesses";
    const std::string path = directory + "/" + id + ".txt";
    if (!ensureDirectory(directory, error) || !diskGuard(error)) return false;
    if (llvm::sys::fs::exists(path)) {
      std::string old;
      if (!readFileBytes(path, old, error)) return false;
      // A crash may leave an uncommitted immutable dictionary entry. It is
      // reusable only by direct byte equality, never by a numeric ID alone.
      if (old != bytes) {
        error = "immutable source witness file has different exact bytes";
        return false;
      }
    } else if (!writeTextAtomically(path, bytes, error)) return false;
    dictionary.emplace(bytes.str(), id); witnessPaths[id] = path;
    return true;
  }

  bool internWitness(StringRef bytes, bool body, std::string &id,
                     std::string &error) {
    auto &dictionary = body ? bodyWitnessIds : graphWitnessIds;
    return internWitnessInNamespace(bytes, body ? "body-" : "graph-",
                                    dictionary, id, error);
  }

  bool internActiveTransferWitness(StringRef bytes, std::string &id,
                                   std::string &error) {
    return internWitnessInNamespace(bytes, "active-transfer-",
                                    activeTransferWitnessIds, id, error);
  }

  bool diskGuard(std::string &error) const {
    std::error_code ec;
    auto space = std::filesystem::space(outputDir.getValue(), ec);
    if (ec || space.available < uintmax_t(minimumFreeBytes.getValue())) {
      error = ec ? "disk guard cannot query filesystem: " + ec.message() :
                   "disk guard: insufficient free bytes for durable search output";
      return false;
    }
    return true;
  }

  bool mergeDuplicate(SearchState &state, ModuleOp canonical,
                      StringRef functionName,
                      std::vector<ArchiveRecord> &archive,
                      StringRef outputDirectory,
                      StringRef candidatesDirectory, uint64_t &nextSerial,
                      std::string &error) {
    if (state.control != !state.controlRoles.empty()) {
      error = "stage control duplicate marker and named roles differ";
      return false;
    }
    {
      std::set<std::string> uniqueRoles;
      for (const std::string &role : state.controlRoles)
        if (role.empty() || !uniqueRoles.insert(role).second) {
          error = "stage control duplicate has an empty or repeated role";
          return false;
        }
    }
    for (size_t index = 0; index < archive.size(); ++index) {
      ArchiveRecord &record = archive[index];
      if (record.key != state.key || record.key.empty()) continue;
      if (state.control && !record.valid) {
        error = "required stage control duplicates a candidate without a legal production score";
        return false;
      }
      if (state.control &&
          !authenticateTypedActionHistory(state, canonical, functionName,
                                          outputDirectory, nextSerial, error))
        return false;
      if (record.path != state.path &&
          !llvm::is_contained(record.alternatePaths, state.path))
        record.alternatePaths.push_back(state.path);
      bool needsRoleSnapshot = false;
      for (const std::string &role : state.controlRoles)
        needsRoleSnapshot |= !record.controlRoleProvenance.count(role);
      std::string roleCandidatePath;
      if (needsRoleSnapshot) {
        if (state.controlRoles.empty()) {
          error = "stage control duplicate has no named control role";
          return false;
        }
        if (!diskGuard(error)) return false;
        for (;;) {
          roleCandidatePath =
              candidatesDirectory.str() + "/" + state.id + "-control-" +
              std::to_string(nextSerial++) + ".mlir";
          if (!llvm::sys::fs::exists(roleCandidatePath)) {
            if (!writeModule(state.module.get(), roleCandidatePath, error))
              return false;
            break;
          }
          OwningOpRef<ModuleOp> existing =
              parseSourceFile<ModuleOp>(roleCandidatePath,
                                        state.module.get().getContext());
          if (existing && neighborhoodReplaySourceText(*existing) ==
                              neighborhoodReplaySourceText(
                                  state.module.get()))
            break;
        }
      }
      for (const std::string &role : state.controlRoles) {
        if (record.controlRoleProvenance.count(role)) continue;
        if (!llvm::is_contained(record.controlRoles, role))
          record.controlRoles.push_back(role);
        ControlRoleProvenance provenance;
        provenance.candidatePath = roleCandidatePath;
        provenance.path = state.path;
        provenance.actionHistory =
            portableActionHistory(state.actionHistory);
        record.controlRoleProvenance.emplace(role, std::move(provenance));
      }
      record.control |= state.control;
      // The archive's primary history belongs to its original retained
      // representative. An equivalent duplicate may add a distinct control
      // role, but must not replace that representative's authenticated path.
      // Upgrade an unknown primary history only when its own immutable module
      // snapshot is retained and byte-equal to this authenticated candidate.
      if (!record.actionHistory.known && state.actionHistory.known &&
          !record.candidatePath.empty()) {
        OwningOpRef<ModuleOp> archived =
            parseSourceFile<ModuleOp>(record.candidatePath,
                                      state.module.get().getContext());
        if (archived &&
            neighborhoodReplaySourceText(*archived) ==
                neighborhoodReplaySourceText(state.module.get())) {
          if (!authenticateTypedActionHistory(state, canonical, functionName,
                                              outputDirectory, nextSerial,
                                              error))
            return false;
          if (state.actionHistory.known) {
            record.actionHistory = portableActionHistory(state.actionHistory);
            record.path = state.path;
          }
        }
      }
      dirtyJournalIndices.insert(index);
      ++duplicateCandidates;
      return true;
    }
    // Rejected actions with no canonical key do not reach this branch.
    error = "dedup key is reserved but absent from the archive";
    return false;
  }

  bool persistBestFamilyWitnesses(
      const SearchState &state, ArrayRef<ArchiveRecord> archive,
      StringRef outputDirectory, std::string &error) {
    if (!state.scored || !state.module)
      return true;
    std::set<std::string> candidateFamilies =
        typedHistoryFamilies(state.actionHistory);
    if (candidateFamilies.empty())
      return true;

    const ArchiveRecord *candidateRecord = nullptr;
    for (const ArchiveRecord &record : archive)
      if (record.id == state.id && record.valid) {
        candidateRecord = &record;
        break;
      }
    if (!candidateRecord)
      return true;

    std::vector<std::string> bestFamilies;
    for (const std::string &family : candidateFamilies) {
      const ArchiveRecord *best = nullptr;
      for (const ArchiveRecord &record : archive) {
        if (!record.valid)
          continue;
        std::set<std::string> recordFamilies =
            typedHistoryFamilies(record.actionHistory);
        if (!recordFamilies.count(family))
          continue;
        if (!best || archiveLess(record, *best))
          best = &record;
      }
      if (best != candidateRecord)
        continue;
      bestFamilies.push_back(family);
    }
    if (bestFamilies.empty())
      return true;

    std::string moduleIR = neighborhoodReplaySourceText(state.module.get());
    std::string costCatalogueJSON;
    const bool hasCostCatalogue =
        !state.costPath.empty() && llvm::sys::fs::exists(state.costPath);
    if (hasCostCatalogue &&
        !readFileBytes(state.costPath, costCatalogueJSON, error))
      return false;
    uint64_t operationCount = 0;
    state.module.get().walk([&](Operation *) { ++operationCount; });
    json::Array pathFamilies;
    for (const std::string &family : bestFamilies)
      pathFamilies.push_back(family);

    json::Object witnessBinding{
        {"stage", stage.getValue()},
        {"function", functionName.getValue()},
        {"source_repository", sourceRepository.getValue()},
        {"source_commit", sourceCommit.getValue()},
        {"architecture_path", architecturePath.getValue()},
        {"protocol_path", protocolFile.getValue()},
        {"source_contract_path", sourceContractFile.getValue()},
        {"prepared_source_path", preparedSourceFile.getValue()},
        {"source_binding_witness", bindingPath},
        {"checkpoint_contract",
         checkpointSchemaForCeiling(diagnosticIICeiling.getValue()).str()}};
    json::Object witness{
        {"schema", "orbit-joint-neighborhood-family-best-witness-v1"},
        {"typed_path_families", std::move(pathFamilies)},
        {"candidate_id", candidateRecord->id},
        {"parent_candidate_id", candidateRecord->parentId},
        {"candidate_key", candidateRecord->key},
        {"graph_facts_key", candidateRecord->factKey},
        {"graph_variant_id", candidateRecord->graphId},
        {"predicted_whole_program_cycles", candidateRecord->score},
        {"global_rank_recorded_in_family_funnel_summary", true},
        {"candidate_path_at_archive", candidateRecord->candidatePath},
        {"cost_catalogue_path_at_archive", candidateRecord->costPath},
        {"candidate_module_bytes", static_cast<int64_t>(moduleIR.size())},
        {"candidate_module_operation_count",
         static_cast<int64_t>(operationCount)},
        {"candidate_module_ir", std::move(moduleIR)},
        {"cost_catalogue_snapshot_available", hasCostCatalogue},
        {"cost_catalogue_snapshot_bytes",
         static_cast<int64_t>(costCatalogueJSON.size())},
        {"cost_catalogue_exact_json", std::move(costCatalogueJSON)},
        {"task_choices", choiceRecords(candidateRecord->choices)},
        {"task_costs", costRecords(candidateRecord->choices,
                                    candidateRecord->costs,
                                    candidateRecord->durations)},
        {"task_schedule", scheduleRecords(
                               candidateRecord->choices,
                               candidateRecord->schedule.placements)},
        {"action_path", stringArray(candidateRecord->path)},
        {"action_history",
         typedActionHistoryObject(candidateRecord->actionHistory)},
        {"binding", std::move(witnessBinding)}};
    const std::string diagnosticsDirectory =
        outputDirectory.str() + "/diagnostics";
    if (!ensureDirectory(diagnosticsDirectory, error) ||
        !diskGuard(error) ||
        !writeTextAtomically(
            diagnosticsDirectory + "/" +
                familyWitnessFilename(candidateRecord->id),
            jsonText(json::Value(std::move(witness))) + "\n", error))
      return false;
    return true;
  }

  void appendFamilyFunnelCandidateEvent(
      int64_t round, StringRef family, StringRef candidateID,
      StringRef parentID, const NeighborhoodAction &action,
      const SearchState::TypedActionHistory &history,
      json::Object result) {
    std::set<std::string> primitiveKinds;
    for (const NeighborhoodPrimitive &primitive : action.primitives)
      primitiveKinds.insert(primitive.kind);
    json::Array primitiveKindArray;
    for (const std::string &kind : primitiveKinds)
      primitiveKindArray.push_back(kind);
    json::Array initialShapes;
    if (history.known)
      for (const NeighborhoodShape &shape : history.initialShapes)
        initialShapes.push_back(shapeObject(shape));
    json::Object typedHistory{
        {"schema", kTypedActionHistorySchema},
        {"known", history.known},
        {"canonicalFactKey", history.canonicalFactKey},
        {"initialShapes", std::move(initialShapes)},
        {"fissionActions", typedActionArray(history.fissionActions)},
        {"actions", typedActionArray(history.actions)}};
    json::Object binding{
        {"stage", stage.getValue()},
        {"function", functionName.getValue()},
        {"source_repository", sourceRepository.getValue()},
        {"source_commit", sourceCommit.getValue()},
        {"architecture_path", architecturePath.getValue()},
        {"protocol_path", protocolFile.getValue()},
        {"source_contract_path", sourceContractFile.getValue()},
        {"prepared_source_path", preparedSourceFile.getValue()},
        {"source_binding_witness", bindingPath},
        {"checkpoint_contract",
         checkpointSchemaForCeiling(diagnosticIICeiling.getValue()).str()}};
    if (stageNumber(stage) == 6)
      binding["fission_source_replay"] =
          "orbit-taskflow-fission-source-replay-v2-ordinary-suffix-rebase";
    json::Object event{
        {"record_type", "candidate_attempt"},
        {"schema", kFamilyFunnelSchema},
        {"round", round},
        {"action_family", family},
        {"candidate_id", candidateID},
        {"parent_id", parentID},
        {"action", typedActionObject(action)},
        {"typed_primitive_kinds", std::move(primitiveKindArray)},
        {"typed_action_history", std::move(typedHistory)},
        {"binding", std::move(binding)},
        {"result", std::move(result)}};
    familyFunnelEvents.push_back(jsonText(json::Value(std::move(event))));
  }

  void appendPendingFamilyFunnelEvent(
      int64_t round, const PendingNeighbor &neighbor,
      const SearchState &parent, StringRef reason) {
    std::set<std::string> primitiveKinds;
    for (const NeighborhoodPrimitive &primitive : neighbor.action.primitives)
      primitiveKinds.insert(primitive.kind);
    json::Array kinds;
    for (const std::string &kind : primitiveKinds)
      kinds.push_back(kind);
    json::Array initialShapes;
    for (const NeighborhoodShape &shape : parent.actionHistory.initialShapes)
      initialShapes.push_back(shapeObject(shape));
    json::Object history{
        {"schema", kTypedActionHistorySchema},
        {"known", parent.actionHistory.known},
        {"canonicalFactKey", parent.actionHistory.canonicalFactKey},
        {"initialShapes", std::move(initialShapes)},
        {"fissionActions",
         typedActionArray(parent.actionHistory.fissionActions)},
        {"actions", typedActionArray(parent.actionHistory.actions)}};
    json::Object event{
        {"record_type", "pending_unattempted"},
        {"schema", kFamilyFunnelSchema},
        {"round", round},
        {"action_family", neighbor.action.family},
        {"candidate_id", ""},
        {"parent_id", parent.id},
        {"action", typedActionObject(neighbor.action)},
        {"typed_primitive_kinds", std::move(kinds)},
        {"typed_action_history", std::move(history)},
        {"status", "not_attempted_at_checkpoint"},
        {"pending_reason", reason.str()},
        {"source_binding_witness", bindingPath}};
    familyFunnelEvents.push_back(jsonText(json::Value(std::move(event))));
  }

  json::Array familyFunnelRoundsObject() const {
    json::Array rounds;
    for (const auto &[round, families] : familyFunnelByRound) {
      json::Object counts;
      for (const auto &[family, counters] : families)
        counts[family] = familyFunnelCountersObject(counters);
      rounds.push_back(json::Object{{"round", round},
                                    {"families", std::move(counts)}});
    }
    return rounds;
  }

  bool restoreFamilyFunnel(const json::Object &root, std::string &error) {
    const json::Array *rounds = root.getArray("family_funnel_rounds");
    const json::Array *events = root.getArray("family_funnel_events");
    if (!rounds || !events) {
      error = "checkpoint lacks family funnel continuation state";
      return false;
    }
    familyFunnelByRound.clear();
    for (const json::Value &value : *rounds) {
      const json::Object *roundObject = value.getAsObject();
      auto round = roundObject ? roundObject->getInteger("round")
                               : std::nullopt;
      const json::Object *families =
          roundObject ? roundObject->getObject("families") : nullptr;
      if (!round || *round < 0 || !families) {
        error = "checkpoint family funnel round is malformed";
        return false;
      }
      FamilyFunnelRound parsedFamilies;
      for (const auto &[familyRef, familyValue] : *families) {
        const json::Object *counts = familyValue.getAsObject();
        FamilyFunnelCounters parsed;
        if (!counts ||
            !parseFamilyFunnelCounters(*counts, parsed) ||
            !parsedFamilies.emplace(familyRef.str(), std::move(parsed)).second) {
          error = "checkpoint family funnel counters are malformed";
          return false;
        }
      }
      if (!familyFunnelByRound.emplace(*round, std::move(parsedFamilies)).second) {
        error = "checkpoint family funnel repeats a round";
        return false;
      }
    }
    familyFunnelEvents.clear();
    for (const json::Value &value : *events) {
      auto eventText = value.getAsString();
      if (!eventText) {
        error = "checkpoint family funnel event is malformed";
        return false;
      }
      auto parsed = json::parse(*eventText);
      const json::Object *eventObject =
          parsed ? parsed->getAsObject() : nullptr;
      auto schema = eventObject ? eventObject->getString("schema")
                                : std::nullopt;
      auto recordType = eventObject ? eventObject->getString("record_type")
                                    : std::nullopt;
      if (!eventObject || !schema || *schema != kFamilyFunnelSchema ||
          !recordType) {
        if (!parsed)
          llvm::consumeError(parsed.takeError());
        error = "checkpoint family funnel event schema is invalid";
        return false;
      }
      familyFunnelEvents.push_back(eventText->str());
    }
    return true;
  }

  bool writeFamilyFunnelSidecars(
      StringRef outputDirectory, StringRef checkpointPath,
      int64_t currentRound,
      ArrayRef<ArchiveRecord> archive, ArrayRef<SearchState> beam,
      ArrayRef<PendingNeighbor> pending, uint64_t pendingCursor,
      StringRef stopReason,
      std::string &error) {
    const std::string diagnosticsDirectory =
        outputDirectory.str() + "/diagnostics";
    if (!ensureDirectory(diagnosticsDirectory, error))
      return false;

    std::vector<const ArchiveRecord *> ranked;
    for (const ArchiveRecord &record : archive)
      if (record.valid)
        ranked.push_back(&record);
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const ArchiveRecord *left,
                        const ArchiveRecord *right) {
      return archiveLess(*left, *right);
    });
    struct PathPresence {
      std::set<std::string> archiveIDs;
      std::set<std::string> scoredIDs;
      std::set<std::string> passedIDs;
      std::set<std::string> beamIDs;
      std::map<uint64_t, const ArchiveRecord *> rankedRecords;
    };
    std::map<std::string, PathPresence> pathPresence;
    auto recordFamilies = [](const ArchiveRecord &record) {
      return typedHistoryFamilies(record.actionHistory);
    };
    for (const ArchiveRecord &record : archive) {
      for (const std::string &family : recordFamilies(record))
        pathPresence[family].archiveIDs.insert(record.id);
      const bool schedulerWasCalled =
          record.valid || StringRef(record.rejectReason).starts_with(
                              "production_scheduler_rejected");
      if (schedulerWasCalled)
        for (const std::string &family : recordFamilies(record))
          pathPresence[family].scoredIDs.insert(record.id);
    }
    for (size_t index = 0; index < ranked.size(); ++index) {
      const ArchiveRecord &record = *ranked[index];
      const uint64_t rank = index + 1;
      for (const std::string &family : recordFamilies(record)) {
        PathPresence &presence = pathPresence[family];
        presence.passedIDs.insert(record.id);
        presence.rankedRecords.emplace(rank, &record);
      }
    }
    for (const auto &[round, families] : familyFunnelByRound)
      for (const auto &[family, counters] : families)
        pathPresence.try_emplace(family);
    for (const SearchState &state : beam) {
      std::set<std::string> families =
          typedHistoryFamilies(state.actionHistory);
      for (const std::string &family : families)
        pathPresence[family].beamIDs.insert(state.id);
    }

    std::set<std::string> activeFamilyWitnessPaths;
    auto candidateRankObject = [](uint64_t rank,
                                  const ArchiveRecord &record,
                                  StringRef witnessPath) {
      json::Object candidate{
          {"candidate_id", record.id},
          {"score", record.score},
          {"global_rank", static_cast<int64_t>(rank)},
          {"candidate_path", record.candidatePath},
          {"cost_catalogue_path", record.costPath}};
      if (!witnessPath.empty())
        candidate["family_witness_path"] = witnessPath.str();
      return candidate;
    };
    json::Object pathPresenceRecords;
    for (const auto &[family, presence] : pathPresence) {
      json::Array archiveIDs, scoredIDs, passedIDs, beamIDs, ranks,
          top5Candidates;
      for (const std::string &id : presence.archiveIDs)
        archiveIDs.push_back(id);
      for (const std::string &id : presence.scoredIDs)
        scoredIDs.push_back(id);
      for (const std::string &id : presence.passedIDs)
        passedIDs.push_back(id);
      for (const std::string &id : presence.beamIDs)
        beamIDs.push_back(id);
      json::Value best = nullptr;
      bool haveBest = false;
      for (const auto &[rank, record] : presence.rankedRecords) {
        ranks.push_back(static_cast<int64_t>(rank));
        std::string witnessPath;
        if (!haveBest) {
          witnessPath = diagnosticsDirectory + "/" +
                        familyWitnessFilename(record->id);
          std::string witnessBytes;
          if (!readFileBytes(witnessPath, witnessBytes, error))
            return false;
          auto parsedWitness = json::parse(witnessBytes);
          const json::Object *witnessObject =
              parsedWitness ? parsedWitness->getAsObject() : nullptr;
          auto witnessSchema = witnessObject
                                   ? witnessObject->getString("schema")
                                   : std::nullopt;
          auto witnessCandidateID = witnessObject
                                        ? witnessObject->getString("candidate_id")
                                        : std::nullopt;
          const json::Array *witnessFamilies =
              witnessObject ? witnessObject->getArray("typed_path_families")
                            : nullptr;
          const auto body = witnessObject
                                ? witnessObject->getString("candidate_module_ir")
                                : std::nullopt;
          if (!witnessObject || !witnessSchema ||
              *witnessSchema !=
                  "orbit-joint-neighborhood-family-best-witness-v1" ||
              !witnessCandidateID || *witnessCandidateID != record->id ||
              !witnessFamilies || !body || body->empty() ||
              !llvm::any_of(*witnessFamilies, [&](const json::Value &value) {
                auto valueFamily = value.getAsString();
                return valueFamily && *valueFamily == family;
              })) {
            if (!parsedWitness)
              llvm::consumeError(parsedWitness.takeError());
            error = "family-best witness does not authenticate its ranked candidate";
            return false;
          }
          activeFamilyWitnessPaths.insert(witnessPath);
        }
        if (rank <= 5)
          top5Candidates.push_back(
              candidateRankObject(rank, *record, witnessPath));
        if (!haveBest) {
          best = candidateRankObject(rank, *record, witnessPath);
          haveBest = true;
        }
      }
      pathPresenceRecords[family] = json::Object{
          {"scored", static_cast<int64_t>(presence.scoredIDs.size())},
          {"scored_candidate_ids", std::move(scoredIDs)},
          {"passed", static_cast<int64_t>(presence.passedIDs.size())},
          {"passed_candidate_ids", std::move(passedIDs)},
          {"beam", static_cast<int64_t>(presence.beamIDs.size())},
          {"beam_candidate_ids", std::move(beamIDs)},
          {"archive", static_cast<int64_t>(presence.archiveIDs.size())},
          {"archive_candidate_ids", std::move(archiveIDs)},
          {"top5", static_cast<int64_t>(top5Candidates.size())},
          {"top5_candidates", std::move(top5Candidates)},
          {"global_ranks", std::move(ranks)}, {"best", std::move(best)}};
    }

    std::map<std::string, FamilyFunnelCounters> totals;
    auto addCounters = [](FamilyFunnelCounters &to,
                          const FamilyFunnelCounters &from) {
      to.menu += from.menu;
      to.generated += from.generated;
      to.attempted += from.attempted;
      to.materialized += from.materialized;
      to.reject += from.reject;
      to.unique += from.unique;
      to.duplicate += from.duplicate;
      to.costPreparationAttempted += from.costPreparationAttempted;
      to.freshCostPrepared += from.freshCostPrepared;
      to.freshCostScored += from.freshCostScored;
      to.cacheReused += from.cacheReused;
      to.costShapeCacheHits += from.costShapeCacheHits;
      to.costShapeCacheMisses += from.costShapeCacheMisses;
      to.schedulerCalls += from.schedulerCalls;
      to.schedulerPass += from.schedulerPass;
      to.schedulerReject += from.schedulerReject;
      to.archive += from.archive;
      to.pendingUnattempted += from.pendingUnattempted;
      for (const auto &[reason, count] : from.rejectReasons)
        to.rejectReasons[reason] += count;
      for (const auto &[reason, count] :
           from.pendingUnattemptedReasons)
        to.pendingUnattemptedReasons[reason] += count;
      to.pendingUnattemptedActionSignatures.insert(
          to.pendingUnattemptedActionSignatures.end(),
          from.pendingUnattemptedActionSignatures.begin(),
          from.pendingUnattemptedActionSignatures.end());
    };
    for (const auto &[round, families] : familyFunnelByRound)
      for (const auto &[family, counts] : families)
        addCounters(totals[family], counts);

    std::map<int64_t, FamilyFunnelRound> roundSnapshots;
    for (const auto &[round, families] : familyFunnelByRound)
      roundSnapshots[round] = families;
    FamilyFunnelRound &current = roundSnapshots[currentRound];
    for (auto &[family, counters] : current) {
      counters.pendingUnattempted = 0;
      counters.pendingUnattemptedReasons.clear();
      counters.pendingUnattemptedActionSignatures.clear();
    }
    for (size_t index = static_cast<size_t>(pendingCursor);
         index < pending.size(); ++index) {
      const PendingNeighbor &neighbor = pending[index];
      if (neighbor.parent >= beam.size())
        continue;
      FamilyFunnelCounters &counters =
          current[neighbor.action.family];
      ++counters.pendingUnattempted;
      const std::string pendingReason = stopReason.empty()
                                            ? "checkpoint_frontier_remaining"
                                            : stopReason.str();
      ++counters.pendingUnattemptedReasons[pendingReason];
      counters.pendingUnattemptedActionSignatures.push_back(
          actionSignature(neighbor.action));
      FamilyFunnelCounters &total = totals[neighbor.action.family];
      ++total.pendingUnattempted;
      ++total.pendingUnattemptedReasons[pendingReason];
      total.pendingUnattemptedActionSignatures.push_back(
          actionSignature(neighbor.action));
    }

    std::string jsonl;
    llvm::raw_string_ostream funnelStream(jsonl);
    for (const std::string &event : familyFunnelEvents)
      funnelStream << event << "\n";
    for (const auto &[round, families] : roundSnapshots)
      for (const auto &[family, counters] : families) {
        json::Object record = familyFunnelCountersObject(counters);
        record["record_type"] = "round_family";
        record["schema"] = kFamilyFunnelSchema;
        record["round"] = round;
        record["action_family"] = family;
        record["current_edge_semantics"] =
            "One count per current edge classified by its exact action family.";
        record["stage"] = stage.getValue();
        record["function"] = functionName.getValue();
        record["source_repository"] = sourceRepository.getValue();
        record["source_commit"] = sourceCommit.getValue();
        record["source_binding_witness"] = bindingPath;
        funnelStream << json::Value(std::move(record)) << "\n";
      }
    for (const auto &[family, presenceValue] : pathPresenceRecords) {
      funnelStream << json::Value(json::Object{
          {"record_type", "typed_path_presence"},
          {"schema", kFamilyFunnelSchema},
          {"through_round", currentRound},
          {"stage", stage.getValue()}, {"function", functionName.getValue()},
          {"action_family", family.str()}, {"presence", presenceValue},
          {"source_binding_witness", bindingPath}})
                   << "\n";
    }
    for (size_t index = static_cast<size_t>(pendingCursor);
         index < pending.size(); ++index) {
      const PendingNeighbor &neighbor = pending[index];
      if (neighbor.parent >= beam.size())
        continue;
      const SearchState &parent = beam[neighbor.parent];
      json::Array kinds;
      std::set<std::string> primitiveKinds;
      for (const NeighborhoodPrimitive &primitive :
           neighbor.action.primitives)
        primitiveKinds.insert(primitive.kind);
      for (const std::string &kind : primitiveKinds)
        kinds.push_back(kind);
      json::Object history{
          {"known", parent.actionHistory.known},
          {"fissionActions",
           typedActionArray(parent.actionHistory.fissionActions)},
          {"actions", typedActionArray(parent.actionHistory.actions)}};
      funnelStream << json::Value(json::Object{
          {"record_type", "pending_unattempted"},
          {"schema", kFamilyFunnelSchema},
          {"round", currentRound},
          {"action_family", neighbor.action.family},
          {"candidate_id", ""}, {"parent_id", parent.id},
          {"action", typedActionObject(neighbor.action)},
          {"typed_primitive_kinds", std::move(kinds)},
          {"typed_action_history", std::move(history)},
          {"status", "not_attempted_at_checkpoint"},
          {"pending_reason", stopReason.empty()
                                 ? "checkpoint_frontier_remaining"
                                 : stopReason.str()},
          {"source_binding_witness", bindingPath}})
                   << "\n";
    }
    funnelStream.flush();

    json::Object currentEdgeRecords;
    for (const auto &[family, counters] : totals)
      currentEdgeRecords[family] = familyFunnelCountersObject(counters);
    json::Object summary{
        {"schema", kFamilyFunnelSchema},
        {"stage", stage.getValue()},
        {"function", functionName.getValue()},
        {"source_repository", sourceRepository.getValue()},
        {"source_commit", sourceCommit.getValue()},
        {"architecture_path", architecturePath.getValue()},
        {"protocol_path", protocolFile.getValue()},
        {"source_contract_path", sourceContractFile.getValue()},
        {"prepared_source_path", preparedSourceFile.getValue()},
        {"source_binding_witness", bindingPath},
        {"checkpoint", checkpointPath.str()},
        {"max_rounds", maxRounds.getValue()},
        {"max_candidates", maxCandidates.getValue()},
        {"beam_width", beamWidth.getValue()},
        {"diversity_slots", diversitySlots.getValue()},
        {"checkpoint_stop_reason", stopReason},
        {"through_round", currentRound},
        {"current_edge_semantics",
         "menu counts are syntactic source inventory; generated and later counters classify current edges by exact action family. A current edge belongs to one exact family, so current-edge counts are additive across families. Typed full-path presence overlaps."},
        {"path_presence_semantics",
         "Only known histories contribute exact representative fissionActions and ordinary actions. Unknown histories and alternate paths are excluded. Candidate counts overlap across families and must not be summed."},
        {"current_edge_by_family", std::move(currentEdgeRecords)},
        {"typed_path_presence_by_family", std::move(pathPresenceRecords)}};
    if (stageNumber(stage) == 6)
      summary["fission_source_replay"] =
          "orbit-taskflow-fission-source-replay-v2-ordinary-suffix-rebase";
    const std::string summaryText =
        jsonText(json::Value(std::move(summary))) + "\n";
    if (!writeTextAtomically(diagnosticsDirectory + "/family-funnel.jsonl",
                             jsonl, error) ||
        !writeTextAtomically(
            diagnosticsDirectory + "/family-funnel-summary.json",
            summaryText, error))
      return false;
    std::error_code cleanupError;
    const std::string witnessPrefix = "family-best-witness-";
    for (std::filesystem::directory_iterator iterator(diagnosticsDirectory,
                                                      cleanupError),
         end;
         !cleanupError && iterator != end; iterator.increment(cleanupError)) {
      const std::string path = iterator->path().string();
      const std::string filename = iterator->path().filename().string();
      if (!StringRef(filename).starts_with(witnessPrefix) ||
          !StringRef(filename).ends_with(".json") ||
          activeFamilyWitnessPaths.count(path))
        continue;
      std::filesystem::remove(iterator->path(), cleanupError);
      if (cleanupError)
        break;
    }
    if (cleanupError) {
      error = "cannot retire stale family-best witness: " +
              cleanupError.message();
      return false;
    }
    return true;
  }

  bool saveCheckpoint(StringRef path, int64_t round, uint64_t nextSerial,
                      uint64_t scoredCount, uint64_t rejectedCount,
                      StringRef canonicalFactKey, const std::vector<ArchiveRecord> &archive,
                      ArrayRef<SearchState> beam, ArrayRef<SearchState> generated,
                      ArrayRef<PendingNeighbor> pending, uint64_t pendingCursor,
                      StringRef stopReason, std::string &error) {
    if (!diskGuard(error)) return false;
    // Archive is append-only with explicit indexed amendments. Checkpoints
    // reference one committed journal prefix instead of rewriting 20k rows
    // every 128 actions. A crash beyond that prefix is ignored on recovery.
    std::string additions;
    llvm::raw_string_ostream stream(additions);
    for (size_t index : dirtyJournalIndices)
      if (index < journalArchiveCount)
        stream << json::Value(json::Object{{"index", int64_t(index)},
                {"candidate", archiveObject(archive[index])}}) << "\n";
    for (size_t index = journalArchiveCount; index < archive.size(); ++index)
      stream << json::Value(json::Object{{"index", int64_t(index)},
              {"candidate", archiveObject(archive[index])}}) << "\n";
    stream.flush();
    if (!additions.empty()) {
      std::ofstream journal(journalPath, std::ios::binary | std::ios::app);
      if (!journal) { error = "cannot open archive journal for append"; return false; }
      journal.write(additions.data(), additions.size()); journal.flush();
      if (!journal) { error = "archive journal write failure"; return false; }
      journal.close();
      if (journal.fail()) { error = "archive journal close/write failure"; return false; }
    }
    const uint64_t newJournalBytes = journalBytes + additions.size();
    json::Array beams, generatedStates, neighbors, graphCosts, witnesses,
        funnelEvents;
    for (const auto &state : beam) beams.push_back(stateObject(state));
    for (const auto &state : generated) generatedStates.push_back(stateObject(state));
    for (const std::string &event : familyFunnelEvents)
      funnelEvents.push_back(event);
    for (const auto &neighbor : pending)
      neighbors.push_back(json::Object{{"parent", int64_t(neighbor.parent)},
                                       {"action", actionObject(neighbor.action)}});
    // Seen failed scoring keys survive restore too. Their canonical graph may
    // lack a score, so reconstructing seen solely from valid rows is unsafe.
    for (const auto &entry : graphCostPaths)
      graphCosts.push_back(json::Object{{"graph_key", entry.first}, {"path", entry.second}});
    for (const auto &entry : witnessPaths)
      witnesses.push_back(json::Object{{"id", entry.first}, {"path", entry.second}});
    json::Object root{
        {"schema", checkpointSchemaForCeiling(diagnosticIICeiling.getValue())},
        {"record_type", "checkpoint"},
        {"function", functionName.getValue()}, {"stage", stage.getValue()},
        {"source_repository", sourceRepository.getValue()}, {"source_commit", sourceCommit.getValue()},
        {"architecture_path", architecturePath.getValue()}, {"protocol_path", protocolFile.getValue()},
        {"binding_witness_path", bindingPath}, {"canonical_fact_key", canonicalFactKey},
        {"round", round}, {"next_serial", int64_t(nextSerial)},
        {"scored_candidates", int64_t(scoredCount)}, {"rejected_candidates", int64_t(rejectedCount)},
        {"round_scored_candidates", int64_t(roundScoredCount)},
        {"round_score_limit", int64_t(roundScoreLimit)},
        {"duplicate_candidates", int64_t(duplicateCandidates)},
        {"production_scheduler_calls", int64_t(productionSchedulerCalls)},
        {"scoring_workers", scoringWorkers.getValue()},
        {"max_partition_factor", maxPartitionFactor.getValue()},
        {"parallel_preparation_milliseconds", parallelPreparationMilliseconds},
        {"parallel_scoring_milliseconds", parallelScoringMilliseconds},
        {"parallel_commit_milliseconds", parallelCommitMilliseconds},
        {"parallel_module_cache_hits", int64_t(parallelModuleCacheHits)},
        {"parallel_module_cache_misses", int64_t(parallelModuleCacheMisses)},
        {"parallel_scored_jobs", int64_t(parallelScoredJobs)},
        {"parallel_batch_item_sizes", uint64Array(parallelBatchItemSizes)},
        {"parallel_batch_score_job_sizes", uint64Array(parallelBatchScoreJobSizes)},
        {"cost_cache_hits", int64_t(costCacheHits)}, {"cost_cache_misses", int64_t(costCacheMisses)},
        {"elapsed_milliseconds", elapsedMilliseconds},
        {"pending_action_cursor", int64_t(pendingCursor)}, {"pending_neighbors", std::move(neighbors)},
        {"beam", std::move(beams)}, {"generated", std::move(generatedStates)},
        {"witness_dictionary", std::move(witnesses)}, {"graph_cost_catalogues", std::move(graphCosts)},
        {"archive_journal_path", journalPath}, {"archive_journal_bytes", int64_t(newJournalBytes)},
        {"archive_count", int64_t(archive.size())}, {"stop_reason", stopReason},
        {"family_funnel_rounds", familyFunnelRoundsObject()},
        {"family_funnel_events", std::move(funnelEvents)},
        {"search_scope", kSearchScope}, {"best_found", true}, {"exhaustive", false}};
    if (diagnosticIICeiling == 23)
      root["diagnostic_ii_ceiling"] = diagnosticIICeiling.getValue();
    if (!writeTextAtomically(path, jsonText(json::Value(std::move(root))) + "\n", error)) return false;
    if (!writeFamilyFunnelSidecars(outputDir, path, round, archive, beam,
                                   pending, pendingCursor, stopReason, error))
      return false;
    journalBytes = newJournalBytes; journalArchiveCount = archive.size(); dirtyJournalIndices.clear();
    // Snapshot deletion happens only after a checkpoint binds the amended
    // archive. A failed checkpoint always leaves the previous replay inputs.
    for (const auto &obsolete : obsoleteCandidatePaths) {
      std::error_code ec = llvm::sys::fs::remove(obsolete);
      if (ec && ec != std::errc::no_such_file_or_directory) {
        error = "cannot retire superseded candidate snapshot: " + ec.message(); return false;
      }
    }
    obsoleteCandidatePaths.clear();
    return true;
  }

  bool restoreCheckpoint(ModuleOp canonical, StringRef path, StringRef binding,
      int64_t &round, uint64_t &nextSerial, uint64_t &scoredCount,
      uint64_t &rejectedCount, std::string &canonicalFactKey,
      std::vector<ArchiveRecord> &archive, std::vector<SearchState> &beam,
      std::vector<SearchState> &generated, std::vector<PendingNeighbor> &pending,
      uint64_t &pendingCursor, std::set<std::string> &seenKeys, std::string &error) {
    std::string bytes;
    if (!readFileBytes(path, bytes, error)) return false;
    auto parsed = json::parse(bytes);
    if (!parsed) { llvm::consumeError(parsed.takeError()); error = "invalid neighborhood checkpoint JSON"; return false; }
    const auto *root = parsed->getAsObject();
    if (!root || !parseCheckpointHeader(*root, functionName, stage, sourceRepository,
        sourceCommit, architecturePath, protocolFile, diagnosticIICeiling,
        error)) return false;
    if (!restoreFamilyFunnel(*root, error))
      return false;
    auto savedRound = root->getInteger("round"), serial = root->getInteger("next_serial"),
         count = root->getInteger("scored_candidates"), rejected = root->getInteger("rejected_candidates"),
         roundCount = root->getInteger("round_scored_candidates"),
         roundLimit = root->getInteger("round_score_limit"),
         cursor = root->getInteger("pending_action_cursor"), prefix = root->getInteger("archive_journal_bytes"),
         archiveCount = root->getInteger("archive_count");
    auto beamRecords = root->getArray("beam"), generatedRecords = root->getArray("generated"),
         neighbors = root->getArray("pending_neighbors");
    auto facts = root->getString("canonical_fact_key"), witness = root->getString("binding_witness_path"),
         savedJournal = root->getString("archive_journal_path");
    if (!savedRound || !serial || !count || !rejected || !roundCount ||
        !roundLimit || !cursor || !prefix || !archiveCount ||
        !beamRecords || !generatedRecords || !neighbors || !facts || !witness || !savedJournal ||
        *witness != bindingPath || *savedJournal != journalPath || *savedRound < 0 || *serial < 0 ||
        *count < 0 || *rejected < 0 || *roundCount < 0 || *roundLimit < 0 ||
        *roundCount > *roundLimit || *cursor < 0 || uint64_t(*cursor) > neighbors->size() ||
        *prefix < 0 || *archiveCount < 0) {
      error = "checkpoint lacks exact frontier/journal continuation state"; return false;
    }
    const uint64_t configuredMaxCandidates =
        static_cast<uint64_t>(maxCandidates.getValue());
    if (*savedRound > maxRounds.getValue() ||
        static_cast<uint64_t>(*count) > configuredMaxCandidates ||
        static_cast<uint64_t>(*roundCount) > static_cast<uint64_t>(*count)) {
      error = "checkpoint round/score counters exceed protocol budgets";
      return false;
    }
    const uint64_t scoresBeforeRound =
        static_cast<uint64_t>(*count) - static_cast<uint64_t>(*roundCount);
    const uint64_t expectedRoundLimit = perRoundScoreQuota(
        scoresBeforeRound, configuredMaxCandidates,
        maxRounds.getValue() - *savedRound);
    const uint64_t remainingScoreBound =
        configuredMaxCandidates - scoresBeforeRound;
    if (static_cast<uint64_t>(*roundLimit) != expectedRoundLimit ||
        static_cast<uint64_t>(*roundLimit) > remainingScoreBound ||
        (*savedRound == maxRounds.getValue() &&
         (*roundCount != 0 || *roundLimit != 0))) {
      error = "checkpoint per-round score quota differs from protocol-derived budget";
      return false;
    }
    if (!readFileBytes(journalPath, bytes, error) || bytes.size() < uint64_t(*prefix)) {
      if (error.empty()) error = "checkpoint committed archive journal prefix is truncated";
      return false;
    }
    const std::string committed = bytes.substr(0, *prefix);
    auto journalBuffer = llvm::MemoryBuffer::getMemBuffer(committed);
    for (llvm::line_iterator lines(*journalBuffer, true); !lines.is_at_end(); ++lines) {
      auto entry = json::parse(*lines);
      if (!entry) { llvm::consumeError(entry.takeError()); error = "invalid committed archive journal JSON"; return false; }
      auto object = entry->getAsObject();
      auto index = object ? object->getInteger("index") : std::nullopt;
      auto candidate = object ? object->getObject("candidate") : nullptr;
      if (!index || !candidate || *index < 0 || uint64_t(*index) > archive.size()) {
        error = "archive journal index is not an append or existing amendment"; return false;
      }
      ArchiveRecord record;
      if (!parseArchiveObject(*candidate, record, error)) return false;
      if (uint64_t(*index) == archive.size()) archive.push_back(std::move(record));
      else archive[*index] = std::move(record);
    }
    if (archive.size() != uint64_t(*archiveCount)) { error = "committed archive row count mismatch"; return false; }
    // Discard only this search's uncommitted journal suffix, after exact byte
    // binding and all committed rows have validated. No candidate traversal.
    if (bytes.size() != uint64_t(*prefix) && !writeTextAtomically(journalPath, committed, error)) return false;
    auto states = [&](const json::Array &records, std::vector<SearchState> &out) {
      for (const auto &value : records) {
        const auto *object = value.getAsObject(); SearchState state;
        if (!object || !parseState(*object, canonical.getContext(), state, error)) return false;
        if (!validateStageSeed(state.module.get(), stage,
                               sourceDomainEnabled, error))
          return false;
        out.push_back(std::move(state));
      }
      return true;
    };
    if (!states(*beamRecords, beam) || !states(*generatedRecords, generated)) return false;
    for (const auto &value : *neighbors) {
      const auto *object = value.getAsObject();
      auto parent = object ? object->getInteger("parent") : std::nullopt;
      auto action = object ? object->getObject("action") : nullptr;
      PendingNeighbor neighbor;
      if (!parent || *parent < 0 || uint64_t(*parent) >= beam.size() || !action ||
          !parseAction(*action, neighbor.action)) { error = "malformed exact pending neighbor"; return false; }
      neighbor.parent = *parent; pending.push_back(std::move(neighbor));
    }
    for (const auto &record : archive) if (!record.key.empty()) seenKeys.insert(record.key);
    const auto *witnesses = root->getArray("witness_dictionary");
    if (!witnesses) { error = "checkpoint missing exact graph/body/active-transfer witness dictionary"; return false; }
    for (const auto &value : *witnesses) {
      const auto *object = value.getAsObject();
      auto id = object ? object->getString("id") : std::nullopt;
      auto witnessPath = object ? object->getString("path") : std::nullopt;
      std::string witnessBytes;
      if (!id || !witnessPath || !readFileBytes(*witnessPath, witnessBytes, error) ||
          witnessPaths.count(id->str())) { if (error.empty()) error = "malformed witness dictionary"; return false; }
      std::map<std::string, std::string> *dictionary = nullptr;
      if (id->starts_with("body-"))
        dictionary = &bodyWitnessIds;
      else if (id->starts_with("active-transfer-"))
        dictionary = &activeTransferWitnessIds;
      else if (id->starts_with("graph-"))
        dictionary = &graphWitnessIds;
      if (!dictionary || !dictionary->emplace(witnessBytes, id->str()).second) {
        error = "duplicate exact dictionary witness with distinct ID"; return false;
      }
      witnessPaths[id->str()] = witnessPath->str();
    }
    for (size_t index = 0; index < archive.size(); ++index) {
      ArchiveRecord &record = archive[index];
      if (!record.actionHistory.known)
        continue;
      if (!record.valid || record.candidatePath.empty()) {
        markActionHistoryUnknown(
            record.actionHistory,
            "checkpoint_archive_history_has_no_retained_candidate_module");
        dirtyJournalIndices.insert(index);
        continue;
      }
      SearchState candidate;
      candidate.module = parseSourceFile<ModuleOp>(record.candidatePath,
                                                   canonical.getContext());
      if (!candidate.module) {
        error = "checkpoint retained archive candidate module is missing: " +
                record.candidatePath;
        return false;
      }
      candidate.shapes = record.shapes;
      candidate.key = record.key;
      candidate.factKey = record.factKey;
      candidate.actionHistory = record.actionHistory;
      candidate.key.clear();
      candidate.factKey.clear();
      const size_t oldWitnessCount = witnessPaths.size();
      if (!prepareCandidateKey(candidate, functionName, outputDir,
                               nextSerial, error) ||
          candidate.key != record.key || candidate.factKey != record.factKey ||
          witnessPaths.size() != oldWitnessCount) {
        if (error.empty())
          error = "restored archive module differs from exact source-owned candidate key";
        return false;
      }
      auto candidateFunction =
          selectTaskFunction(candidate.module.get(), functionName, error);
      if (failed(candidateFunction))
        return false;
      auto graph = (*candidateFunction)->getAttrOfType<StringAttr>(
          "amoeba.graph_variant_id");
      if (!graph || graph.getValue() != record.graphId) {
        error = "restored archive graph label differs from its exact candidate record";
        return false;
      }
      if (!authenticateTypedActionHistory(candidate, canonical, functionName,
                                          outputDir, nextSerial, error))
        return false;
      record.actionHistory = portableActionHistory(candidate.actionHistory);
      dirtyJournalIndices.insert(index);
    }
    for (size_t index = 0; index < archive.size(); ++index) {
      ArchiveRecord &record = archive[index];
      if (record.controlRoleProvenance.empty())
        continue;
      if (!record.valid || !record.control) {
        error = "checkpoint control-role provenance is attached to a non-control archive row";
        return false;
      }
      for (auto &[role, provenance] : record.controlRoleProvenance) {
        if (provenance.candidatePath.empty()) {
          error = "checkpoint control role has no retained exact source module: " +
                  role;
          return false;
        }
        SearchState candidate;
        candidate.module = parseSourceFile<ModuleOp>(
            provenance.candidatePath, canonical.getContext());
        if (!candidate.module) {
          error = "checkpoint retained control-role module is missing: " +
                  provenance.candidatePath;
          return false;
        }
        candidate.shapes = record.shapes;
        candidate.key = record.key;
        candidate.factKey = record.factKey;
        candidate.actionHistory = provenance.actionHistory;
        candidate.key.clear();
        candidate.factKey.clear();
        const size_t oldWitnessCount = witnessPaths.size();
        if (!prepareCandidateKey(candidate, functionName, outputDir,
                                 nextSerial, error) ||
            candidate.key != record.key ||
            candidate.factKey != record.factKey ||
            witnessPaths.size() != oldWitnessCount) {
          if (error.empty())
            error = "restored control-role module differs from exact source-owned candidate key";
          return false;
        }
        auto candidateFunction =
            selectTaskFunction(candidate.module.get(), functionName, error);
        if (failed(candidateFunction))
          return false;
        auto graph = (*candidateFunction)->getAttrOfType<StringAttr>(
            "amoeba.graph_variant_id");
        if (!graph || graph.getValue() != record.graphId) {
          error = "restored control-role graph label differs from its exact candidate record";
          return false;
        }
        const std::string previousHistory =
            jsonText(json::Value(typedActionHistoryObject(
                provenance.actionHistory)));
        if (!authenticateTypedActionHistory(candidate, canonical,
                                            functionName, outputDir,
                                            nextSerial, error))
          return false;
        provenance.actionHistory =
            portableActionHistory(candidate.actionHistory);
        if (previousHistory !=
            jsonText(json::Value(typedActionHistoryObject(
                provenance.actionHistory))))
          dirtyJournalIndices.insert(index);
      }
    }
    auto validateRestoredStates = [&](std::vector<SearchState> &states) {
      for (auto &state : states) {
        const std::string key = state.key, graphKey = state.factKey;
        const size_t oldWitnessCount = witnessPaths.size();
        state.key.clear(); state.factKey.clear();
        if (!prepareCandidateKey(state, functionName, outputDir, nextSerial,
                                 error) || state.key != key || state.factKey != graphKey ||
            witnessPaths.size() != oldWitnessCount) {
          if (error.empty()) error = "restored module differs from exact canonical graph/body/active-transfer/lineage witnesses";
          return false;
        }
      }
      return true;
    };
    if (!validateRestoredStates(beam) || !validateRestoredStates(generated)) return false;
    auto authenticateRestoredStates = [&](std::vector<SearchState> &states) {
      for (SearchState &state : states)
        if (!authenticateTypedActionHistory(state, canonical, functionName,
                                            outputDir, nextSerial, error))
          return false;
      return true;
    };
    if (!authenticateRestoredStates(beam) ||
        !authenticateRestoredStates(generated))
      return false;
    if (const auto *costs = root->getArray("graph_cost_catalogues"))
      for (const auto &value : *costs) {
        const auto *object = value.getAsObject();
        auto key = object ? object->getString("graph_key") : std::nullopt;
        auto costPath = object ? object->getString("path") : std::nullopt;
        if (!key || !costPath) { error = "malformed cached graph catalogue continuation"; return false; }
        graphCostPaths[key->str()] = costPath->str();
      }
    round = *savedRound; nextSerial = *serial; scoredCount = *count; rejectedCount = *rejected;
    roundScoredCount = *roundCount; roundScoreLimit = *roundLimit;
    pendingCursor = *cursor; canonicalFactKey = facts->str();
    journalBytes = *prefix; journalArchiveCount = archive.size();
    duplicateCandidates = root->getInteger("duplicate_candidates").value_or(0);
    productionSchedulerCalls = root->getInteger("production_scheduler_calls").value_or(0);
    costCacheHits = root->getInteger("cost_cache_hits").value_or(0);
    costCacheMisses = root->getInteger("cost_cache_misses").value_or(0);
    parallelPreparationMilliseconds =
        root->getInteger("parallel_preparation_milliseconds").value_or(0);
    parallelScoringMilliseconds =
        root->getInteger("parallel_scoring_milliseconds").value_or(0);
    parallelCommitMilliseconds =
        root->getInteger("parallel_commit_milliseconds").value_or(0);
    parallelModuleCacheHits =
        root->getInteger("parallel_module_cache_hits").value_or(0);
    parallelModuleCacheMisses =
        root->getInteger("parallel_module_cache_misses").value_or(0);
    parallelScoredJobs = root->getInteger("parallel_scored_jobs").value_or(0);
    if (const auto *sizes = root->getArray("parallel_batch_item_sizes")) {
      if (!parseUInt64Array(sizes, parallelBatchItemSizes)) {
        error = "checkpoint parallel batch item sizes are malformed";
        return false;
      }
    }
    if (const auto *sizes = root->getArray("parallel_batch_score_job_sizes")) {
      if (!parseUInt64Array(sizes, parallelBatchScoreJobSizes)) {
        error = "checkpoint parallel batch score job sizes are malformed";
        return false;
      }
    }
    elapsedMilliseconds = root->getInteger("elapsed_milliseconds").value_or(0);
    (void)binding;
    return true;
  }

  static json::Object scoreRecord(const ArchiveRecord &record, StringRef shapeId) {
    return json::Object{
        {"schema", "amoeba-exact-joint-task-scores-v1"}, {"record_type", "score"},
        {"candidate_id", record.id}, {"shape_candidate_id", shapeId.str()},
        {"graph_variant_id", record.graphId}, {"valid", true},
        {"predicted_whole_program_cycles", record.score},
        {"score_source", "ML-task-cost-production-scheduler-explicit-communication"},
        {"task_costs", costRecords(record.choices, record.costs, record.durations)},
        {"task_schedule", scheduleRecords(record.choices, record.schedule.placements)},
        {"dispatch_order", indexArray(record.schedule.dispatch, record.choices)},
        {"replayed_communication_edges", int64_t(record.schedule.replayedEdges)},
        {"communication_trace_known", record.schedule.replayedEdgesKnown},
        {"best_found", true}, {"certified", false}, {"exhaustive", false}};
  }

  bool replaySelection(const ArchiveRecord &record, StringRef function,
                       StringRef stageName, StringRef outputDirectory,
                       json::Object &selection, std::string &error) {
    if (record.candidatePath.empty()) { error = "retained native candidate has no materialized source module"; return false; }
    auto module = parseSourceFile<ModuleOp>(record.candidatePath, &getContext());
    if (!module) { error = "retained native candidate module does not parse"; return false; }
    auto selected = selectTaskFunction(*module, function, error);
    if (failed(selected)) return false;
    std::string activeTransferSignature;
    if (!collectVerifiedActiveTransferSignature(*selected,
                                                activeTransferSignature,
                                                error))
      return false;
    if (!verifyActiveTransferWitnessInKey(activeTransferSignature, record.key,
                                          error))
      return false;
    auto graph = (*selected)->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
    if (!graph || graph.getValue() != record.graphId) {
      error = "replay candidate graph label differs from saved C++ archive"; return false;
    }
    const std::string shapeId = "shape-" + record.id;
    const std::string replayDirectory = outputDirectory.str() + "/replay";
    if (!ensureDirectory(replayDirectory, error) || !diskGuard(error)) return false;
    const std::string shapePath = replayDirectory + "/" + record.id + "-shape.json";
    const std::string scorePath = replayDirectory + "/" + record.id + "-score.jsonl";
    Candidate candidate; candidate.id = shapeId; candidate.choices = record.choices;
    if (!writeNeighborhoodShapeSelection(*module, *selected, candidate, shapePath, error)) return false;
    int64_t iterations = 64;
    if (auto configured = (*selected)->getAttrOfType<IntegerAttr>("joint_scheduling_fixed_point_max_iterations"))
      iterations = configured.getInt();
    json::Object header{
        {"record_type", "header"}, {"schema", "amoeba-exact-joint-task-scores-v1"},
        {"function", function}, {"schedule_space", "production-scheduler"},
        {"temporal_model", "production-scheduler"},
        {"dispatch_policy", parseStageFixed(stageName) ? "fixed" : "critical-path"},
        {"start_policy", "production-spatial-temporal-scheduler"},
        {"temporal_search_scope", "production-task-scheduler-only"},
        {"production_fixed_point_max_iterations", iterations},
        {"search_scope", kSearchScope}, {"best_found", true}, {"exhaustive", false},
        {"source_repository", sourceRepository.getValue()}, {"source_commit", sourceCommit.getValue()},
        {"architecture_path", architecturePath.getValue()}, {"source_binding_witness", bindingPath}};
    json::Object footer{{"record_type", "footer"}, {"schema", "amoeba-exact-joint-task-scores-v1"},
                        {"scored_count", 1}, {"status", "best-found"}, {"certified", false}};
    std::string text = jsonText(json::Value(std::move(header))) + "\n" +
        jsonText(json::Value(scoreRecord(record, shapeId))) + "\n" +
        jsonText(json::Value(std::move(footer))) + "\n";
    if (!writeTextAtomically(scorePath, text, error)) return false;
    selection = archiveObject(record);
    selection["record_type"] = "selection";
    selection["shape_candidate_id"] = shapeId;
    selection["score_file"] = scorePath;
    selection["shape_manifest_file"] = shapePath;
    selection["score_record"] = scoreRecord(record, shapeId);
    selection["certified"] = false; selection["best_found"] = true;
    selection["source_binding_witness"] = bindingPath;
    selection["transformations"] = json::Object{
        {"shape", choiceRecords(record.choices)}, {"action_path", stringArray(record.path)},
        {"lineage_source", record.candidatePath}};
    return true;
  }

  bool writeOutputs(
      StringRef outputDirectory, StringRef checkpointPath, StringRef function,
      StringRef stageName, StringRef repository, StringRef commit,
      StringRef architecture, StringRef protocol, int64_t maxRoundsValue,
      int64_t maxCandidatesValue, int64_t beamWidthValue,
      int64_t diversityValue, int64_t round, uint64_t scoredCount,
      uint64_t rejectedCount, bool costCacheLoaded,
      bool previousWinnerRequested, bool historicalWinnerRequested,
      ArrayRef<ArchiveRecord> archive,
      StringRef stopReason, std::string &error) {
    std::vector<const ArchiveRecord *> ranked;
    json::Object reasons;
    uint64_t validCount = 0;
    for (const auto &record : archive) {
      if (record.valid) { ranked.push_back(&record); ++validCount; }
      else {
        StringRef reason = record.rejectReason;
        std::string category = reason.take_front(reason.find(':')).str();
        if (category.empty()) category = "unknown";
        reasons[category] = reasons.getInteger(category).value_or(0) + 1;
      }
    }
    reasons["canonical_duplicate"] = int64_t(duplicateCandidates);
    std::stable_sort(ranked.begin(), ranked.end(), [](auto *left, auto *right) {
      return archiveLess(*left, *right);
    });
    if (ranked.size() > 5) ranked.resize(5);
    std::string archiveText;
    llvm::raw_string_ostream archiveStream(archiveText);
    for (const auto &record : archive) archiveStream << json::Value(archiveObject(record)) << "\n";
    archiveStream.flush();
    if (!diskGuard(error) || !writeTextAtomically(outputDirectory.str() + "/archive.jsonl", archiveText, error)) return false;
    auto identityShapeRecords = [&]() {
      json::Array result;
      for (const NeighborhoodShape &shape : canonicalIdentityShapes)
        result.push_back(shapeObject(shape));
      return result;
    };
    auto unsupportedUnitRecords = [&]() {
      json::Array result;
      for (const UnsupportedUnitQuery &query :
           canonicalAllUnitUnsupportedQueries) {
        result.push_back(json::Object{
            {"task", query.task},
            {"mapper_tile_rows", query.mapperRows},
            {"mapper_tile_cols", query.mapperCols},
            {"status", kModelDomainUnsupportedStatus.str()},
            {"unsupported_reason", query.reason},
            {"analytical_lower_bound", query.lowerBound},
            {"model_interval_max_ii", kFormalMax4ModelCeilingII}});
      }
      return result;
    };
    json::Object metadata{
        {"schema", kSearchSchema}, {"search_scope", kSearchScope}, {"function", function},
        {"stage", stageName}, {"source_repository", repository}, {"source_commit", commit},
        {"schedule_space", "production-scheduler"},
        {"dispatch_policy", parseStageFixed(stageName) ? "fixed" : "critical-path"},
        {"start_policy", "production-spatial-temporal-scheduler"},
        {"architecture_path", architecture}, {"protocol_path", protocol},
        {"source_binding_witness", bindingPath}, {"max_rounds", maxRoundsValue},
        {"max_candidates", maxCandidatesValue}, {"beam_width", beamWidthValue},
        {"diversity_slots", diversityValue}, {"rounds", round}, {"rounds_completed", round},
        {"unique_complete_candidates_scored", int64_t(scoredCount)},
        {"unique_scored_candidates", int64_t(scoredCount)}, {"unique_valid_candidates", int64_t(validCount)},
        {"rejected_or_duplicate_candidates", int64_t(rejectedCount)},
        {"scoring_workers", scoringWorkers.getValue()},
        {"max_partition_factor", maxPartitionFactor.getValue()},
        {"parallel_phase_elapsed_milliseconds",
         json::Object{{"candidate_preparation", parallelPreparationMilliseconds},
                      {"worker_scoring", parallelScoringMilliseconds},
                      {"ordered_commit", parallelCommitMilliseconds}}},
        {"parallel_module_cache_hits", int64_t(parallelModuleCacheHits)},
        {"parallel_module_cache_misses", int64_t(parallelModuleCacheMisses)},
        {"parallel_scored_jobs", int64_t(parallelScoredJobs)},
        {"parallel_batch_item_sizes", uint64Array(parallelBatchItemSizes)},
        {"parallel_batch_score_job_sizes", uint64Array(parallelBatchScoreJobSizes)},
        {"cache_hits", int64_t(costCacheHits)}, {"cache_misses", int64_t(costCacheMisses)},
        {"reject_reasons", std::move(reasons)}, {"elapsed_seconds", elapsedMilliseconds / 1000.0},
        {"production_scheduler_calls", int64_t(productionSchedulerCalls)},
        {"graph_cost_catalogues", int64_t(graphCostPaths.size())},
        {"cost_catalogue_loaded", costCacheLoaded}, {"best_found", true}, {"exhaustive", false},
        {"stop_reason", stopReason}, {"checkpoint", checkpointPath},
        {"identity_control_retained", true}, {"previous_winner_requested", previousWinnerRequested},
        {"native_top5_required", true}, {"numeric_trace_sram_separate", true},
        {"family_funnel_schema", kFamilyFunnelSchema},
        {"family_funnel_path",
         outputDirectory.str() + "/diagnostics/family-funnel.jsonl"},
        {"family_funnel_summary_path",
         outputDirectory.str() + "/diagnostics/family-funnel-summary.json"},
        {"native_shortlist_count", int64_t(ranked.size())}};
    if (diagnosticIICeiling == 23)
      metadata["diagnostic_ii_ceiling"] = diagnosticIICeiling.getValue();
    if (stageNumber(stageName) == 6) {
      json::Array support;
      for (const std::string &entry : fissionSupportDiagnostics)
        support.push_back(entry);
      metadata["fission_dimension"] = json::Object{
          {"prepared_source_file", preparedSourceFile.getValue()},
          {"max_actions_per_task",
           maxFissionActionsPerTask.getValue()},
          {"complete_cut_action_count",
           static_cast<int64_t>(canonicalFissionActions.size())},
          {"task_support", std::move(support)},
          {"source_replay_verified", true},
          {"split_counter_domains", "retained-per-operation-not-disjoint"}};
    }
    if (bootstrapSupportedShapes) {
      metadata["supported_shape_bootstrap_policy"] =
          kSupportedShapeBootstrapPolicy.str();
      metadata["canonical_bootstrap_applied"] = canonicalBootstrapApplied;
      metadata["canonical_all_unit_cost_status"] =
          canonicalAllUnitCostStatus;
      metadata["canonical_all_unit_control_scored"] =
          canonicalAllUnitControlScored;
      metadata["canonical_actual_identity_shapes"] = identityShapeRecords();
      metadata["canonical_all_unit_unsupported_queries"] =
          unsupportedUnitRecords();
    }
    if (!activeTransferArguments.empty()) {
      json::Array arguments;
      for (unsigned index : activeTransferArguments)
        arguments.push_back(static_cast<int64_t>(index));
      metadata["active_transfer_proof"] = json::Object{
          {"schema", kActiveTransferProofSchema},
          {"arguments", std::move(arguments)},
          {"require_proven", activeTransferRequireProven.getValue()},
          {"witness_encoding", kActiveTransferWitnessEncoding}};
    }
    if (historicalWinnerRequested) {
      json::Array requiredControlRoles;
      requiredControlRoles.push_back("identity");
      if (previousWinnerRequested)
        requiredControlRoles.push_back("previous_stage_measured_winner");
      requiredControlRoles.push_back("historical_measured_winner");
      metadata["historical_native_winner_requested"] = true;
      metadata["required_control_roles"] = std::move(requiredControlRoles);
    }
    json::Object header = metadata; header["record_type"] = "header";
    std::string topText = jsonText(json::Value(std::move(header))) + "\n";
    for (size_t rank = 0; rank < ranked.size(); ++rank) {
      json::Object selection;
      if (!replaySelection(*ranked[rank], function, stageName, outputDirectory, selection, error)) return false;
      selection["rank"] = int64_t(rank);
      topText += jsonText(json::Value(std::move(selection))) + "\n";
    }
    json::Object footer = metadata; footer["record_type"] = "footer"; footer["status"] = "best-found";
    topText += jsonText(json::Value(std::move(footer))) + "\n";
    std::string controlsText;
    std::set<std::string> roles;
    for (const auto &record : archive) {
      if (!record.valid || !record.control) continue;
      for (const auto &role : record.controlRoles) {
        if (!roles.insert(role).second) { error = "duplicate stage control role"; return false; }
        auto provenance = record.controlRoleProvenance.find(role);
        if (provenance == record.controlRoleProvenance.end() ||
            provenance->second.candidatePath.empty()) {
          error = "stage control role lacks its retained exact candidate provenance: " +
                  role;
          return false;
        }
        ArchiveRecord roleRecord = record;
        roleRecord.candidatePath = provenance->second.candidatePath;
        roleRecord.path = provenance->second.path;
        roleRecord.actionHistory =
            portableActionHistory(provenance->second.actionHistory);
        roleRecord.controlRoles = {role};
        roleRecord.controlRoleProvenance.clear();
        roleRecord.controlRoleProvenance.emplace(role, provenance->second);
        json::Object selection;
        if (!replaySelection(roleRecord, function, stageName, outputDirectory,
                             selection, error))
          return false;
        selection["record_type"] = "control"; selection["control_role"] = role;
        if (bootstrapSupportedShapes && role == "identity") {
          selection["supported_shape_bootstrap_policy"] =
              kSupportedShapeBootstrapPolicy.str();
          selection["canonical_bootstrap_applied"] =
              canonicalBootstrapApplied;
          selection["canonical_all_unit_cost_status"] =
              canonicalAllUnitCostStatus;
          selection["canonical_all_unit_control_scored"] =
              canonicalAllUnitControlScored;
          selection["canonical_actual_identity_shapes"] =
              identityShapeRecords();
          selection["canonical_all_unit_unsupported_queries"] =
              unsupportedUnitRecords();
        }
        controlsText += jsonText(json::Value(std::move(selection))) + "\n";
      }
    }
    if (!roles.count("identity") ||
        (previousWinnerRequested &&
         !roles.count("previous_stage_measured_winner")) ||
        (historicalWinnerRequested &&
         !roles.count("historical_measured_winner"))) {
      error = "required stage native controls missing from C++ archive"; return false;
    }
    return writeTextAtomically(outputDirectory.str() + "/top5.jsonl", topText, error) &&
           writeTextAtomically(outputDirectory.str() + "/controls.jsonl", controlsText, error) &&
           writeTextAtomically(outputDirectory.str() + "/search-summary.json", jsonText(json::Value(std::move(metadata))) + "\n", error);
  }

  std::string journalPath, bindingPath, historicalNativeWinnerBoundBytes;
  std::string preparedTaskflowSourceBytes;
  OwningOpRef<ModuleOp> preparedTaskflowSource;
  std::vector<NeighborhoodAction> canonicalFissionActions;
  std::vector<std::string> fissionSupportDiagnostics;
  std::map<std::string, std::string> effectiveFissionCanonicalText;
  bool preparedSourceCanonicalVerified = false;
  std::optional<std::string> interTaskNetworkSpecOverrideBytes;
  uint64_t journalBytes = 0, journalArchiveCount = 0, duplicateCandidates = 0;
  uint64_t productionSchedulerCalls = 0, costCacheHits = 0, costCacheMisses = 0;
  uint64_t roundScoredCount = 0, roundScoreLimit = 0;
  int64_t elapsedMilliseconds = 0;
  int64_t parallelPreparationMilliseconds = 0;
  int64_t parallelScoringMilliseconds = 0;
  int64_t parallelCommitMilliseconds = 0;
  uint64_t parallelModuleCacheHits = 0;
  uint64_t parallelModuleCacheMisses = 0;
  uint64_t parallelScoredJobs = 0;
  func::FuncOp sourceDomainCanonical;
  bool sourceDomainEnabled = false;
  SmallVector<unsigned, 8> activeTransferArguments;
  std::vector<uint64_t> parallelBatchItemSizes;
  std::vector<uint64_t> parallelBatchScoreJobSizes;
  PersistentParallelWorkerPool parallelWorkerPool;
  std::set<size_t> dirtyJournalIndices;
  std::vector<std::string> obsoleteCandidatePaths;
  std::vector<size_t> rankedArchivePrefix;
  std::map<std::string, std::string> graphCostPaths;
  std::string canonicalCostPath;
  std::vector<NeighborhoodShape> canonicalIdentityShapes;
  std::vector<UnsupportedUnitQuery> canonicalAllUnitUnsupportedQueries;
  std::string canonicalAllUnitCostStatus = "supported";
  bool canonicalAllUnitControlScored = false;
  bool canonicalBootstrapApplied = false;
  std::map<std::string, std::string> bodyWitnessIds, graphWitnessIds;
  std::map<std::string, std::string> activeTransferWitnessIds, witnessPaths;
  std::map<std::string, std::unique_ptr<TaskShapeCostCache>> graphCostCaches;
  FamilyFunnelByRound familyFunnelByRound;
  std::vector<std::string> familyFunnelEvents;

};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createSearchJointNeighborhoodPass() {
  return std::make_unique<SearchJointNeighborhoodPass>();
}
} // namespace mlir::amoeba::neura
