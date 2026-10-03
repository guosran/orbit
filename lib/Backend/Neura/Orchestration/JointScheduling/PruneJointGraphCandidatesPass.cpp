//===- PruneJointGraphCandidatesPass.cpp ----------------------*- C++ -*-===//
//
// Post-enumeration, fail-open validation of persisted joint graph candidates.
//
// The closure pass deliberately enumerates the rewrite space before this pass
// runs.  This pass therefore never mutates a rewrite frontier and never makes
// task-count, operation-count, or probabilistic legality decision. It checks
// persisted artifacts and typed dependencies, then optionally reuses the exact
// scorer's proven cached-cost graph bound. A missing bound retains the graph.
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "Backend/Neura/Orchestration/JointScheduling/GraphFactsIO.h"
#include "NeuraDialect/NeuraDialect.h"
#include "TaskflowDialect/TaskflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;
namespace json = llvm::json;

namespace {

constexpr StringLiteral kSchema = "orbit-pruned-joint-graph-candidates-v1";
constexpr StringLiteral kClosureSchema = "orbit-joint-graph-closure-v1";
constexpr StringLiteral kProductionTemporalModel =
    "production-scheduler";
constexpr StringLiteral kProductionScheduleSpace =
    "production-scheduler";
constexpr StringLiteral kProductionTemporalScope =
    "production-task-scheduler-only";
constexpr StringLiteral kProductionStartPolicy =
    "production-spatial-temporal-scheduler";
constexpr StringLiteral kProductionSchedulerBackend =
    "orchestrate-tasks-on-accelerators";

static std::string jsonText(const json::Value &value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << value;
  stream.flush();
  return text;
}

static std::string jsonArrayText(const json::Array &array) {
  json::Array copy;
  copy.reserve(array.size());
  for (const json::Value &value : array)
    copy.push_back(value);
  return jsonText(json::Value(std::move(copy)));
}

static std::string quoteOption(StringRef value) {
  std::string result = "\"";
  for (char character : value) {
    if (character == '\\' || character == '"')
      result += '\\';
    result += character;
  }
  result += '"';
  return result;
}

static bool isProductionStageOne(StringRef stage) {
  return stage == "shape-only" || stage == "stage1" || stage == "stage-1";
}

struct GraphRow {
  std::string graphId;
  std::string status;
  std::string path;
  std::string order;
  std::string mlirPath;
  std::string factsPath;
  std::string structuralText;
  bool deduplicated = false;
  bool hasStructuralKey = false;
};

struct ClosureInput {
  bool readable = false;
  bool valid = true;
  bool sawHeader = false;
  bool sawFooter = false;
  bool headerComplete = false;
  bool footerComplete = false;
  bool headerResourceBound = false;
  bool footerResourceBound = false;
  bool upstreamComplete = false;
  bool upstreamResourceBound = false;
  // Older closure artifacts omit this field.  Preserve that compatibility,
  // but an explicit false must prevent an upstream-complete claim.
  std::optional<bool> graphScopeComplete;
  std::string upstreamStatus;
  std::string function;
  std::string stage;
  std::string graphArtifactDirectory;
  int64_t upstreamGraphCount = -1;
  int64_t upstreamUnknownActionCount = -1;
  int64_t upstreamIncompleteFactsCount = -1;
  std::string error;
  SmallVector<GraphRow> graphs;
};

enum class CandidateStatus { Retained, Pruned, BoundPruned, Unknown };

struct Verification {
  CandidateStatus status = CandidateStatus::Unknown;
  std::string reason;
  std::string diagnostic;
  std::string proofKind;
  int64_t nodeCount = 0;
  int64_t edgeCount = 0;
  int64_t topologicalCount = 0;
  bool eligibleForML = false;
  std::string boundStatus = "not-requested";
  std::string boundScorePath;
  std::string boundDiagnostic;
  int64_t graphLowerBound = 0;
  int64_t incumbentFifth = 0;
};

struct BoundInputs {
  const json::Object *costManifest = nullptr;
  const json::Object *stageManifest = nullptr;
  std::string incumbentPath;
  std::string stageManifestPath;
  std::string temporalModel;
  std::string scheduleSpace;
  std::string dispatchPolicy;
};

struct DagResult {
  bool valid = false;
  bool cycle = false;
  int64_t nodeCount = 0;
  int64_t edgeCount = 0;
  int64_t topologicalCount = 0;
  std::string error;
};

static StringRef statusText(CandidateStatus status) {
  switch (status) {
  case CandidateStatus::Retained:
    return "retained";
  case CandidateStatus::Pruned:
    return "pruned";
  case CandidateStatus::BoundPruned:
    return "bound-pruned";
  case CandidateStatus::Unknown:
    return "unknown";
  }
  llvm_unreachable("unknown candidate status");
}

static bool readJsonFile(StringRef path, json::Value &value,
                         std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read JSON file " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  llvm::Expected<json::Value> parsed =
      json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "cannot parse JSON file " + path.str() + ": " +
            llvm::toString(parsed.takeError());
    return false;
  }
  value = std::move(*parsed);
  return true;
}

static bool readFactsJsonFile(StringRef path, json::Value &value,
                               std::string &error) {
  std::string factsText;
  if (!graph_facts_io::readGraphFactsFile(path, factsText, error))
    return false;
  llvm::Expected<json::Value> parsed = json::parse(factsText);
  if (!parsed) {
    error = "cannot parse graph facts file " + path.str() + ": " +
            llvm::toString(parsed.takeError());
    return false;
  }
  value = std::move(*parsed);
  return true;
}

static bool getRequiredString(const json::Object &object, StringRef field,
                              std::string &value, std::string &error) {
  auto found = object.getString(field);
  if (!found || found->empty()) {
    error = "JSON object is missing non-empty " + field.str();
    return false;
  }
  value = found->str();
  return true;
}

static bool readClosure(StringRef path, ClosureInput &input) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    input.error = "cannot read closure " + path.str() + ": " +
                  buffer.getError().message();
    return false;
  }
  input.readable = true;
  llvm::MemoryBufferRef memory = (*buffer)->getMemBufferRef();
  llvm::line_iterator lines(memory, true);
  for (; !lines.is_at_end(); ++lines) {
    llvm::Expected<json::Value> parsed = json::parse(*lines);
    if (!parsed) {
      input.valid = false;
      input.error = "closure contains invalid JSON: " +
                    llvm::toString(parsed.takeError());
      continue;
    }
    json::Object *object = parsed->getAsObject();
    if (!object) {
      input.valid = false;
      input.error = "closure record is not a JSON object";
      continue;
    }
    auto recordType = object->getString("record_type");
    if (!recordType) {
      input.valid = false;
      input.error = "closure record has no record_type";
      continue;
    }
    if (*recordType == "header") {
      if (input.sawHeader) {
        input.valid = false;
        input.error = "closure has duplicate header";
        continue;
      }
      input.sawHeader = true;
      auto schema = object->getString("schema");
      if (!schema || *schema != kClosureSchema) {
        input.valid = false;
        input.error = "closure header schema is not recognized";
      }
      input.headerComplete = object->getBoolean("complete").value_or(false);
      if (auto scope = object->getBoolean("graph_scope_complete")) {
        if (!*scope || !input.graphScopeComplete)
          input.graphScopeComplete = *scope;
      }
      input.headerResourceBound =
          object->getBoolean("resource_bound_reached").value_or(false);
      if (auto status = object->getString("status"))
        input.upstreamStatus = status->str();
      if (auto function = object->getString("function"))
        input.function = function->str();
      if (auto stage = object->getString("stage"))
        input.stage = stage->str();
      if (auto directory = object->getString("graph_artifact_directory"))
        input.graphArtifactDirectory = directory->str();
      continue;
    }
    if (*recordType == "footer") {
      if (input.sawFooter) {
        input.valid = false;
        input.error = "closure has duplicate footer";
        continue;
      }
      input.sawFooter = true;
      auto schema = object->getString("schema");
      if (!schema || *schema != kClosureSchema) {
        input.valid = false;
        input.error = "closure footer schema is not recognized";
      }
      input.footerComplete = object->getBoolean("complete").value_or(false);
      if (auto scope = object->getBoolean("graph_scope_complete")) {
        if (!*scope || !input.graphScopeComplete)
          input.graphScopeComplete = *scope;
      }
      input.footerResourceBound =
          object->getBoolean("resource_bound_reached").value_or(false);
      if (auto status = object->getString("status"))
        input.upstreamStatus = status->str();
      if (auto count = object->getInteger("graph_count"))
        input.upstreamGraphCount = *count;
      if (auto count = object->getInteger("unknown_action_count"))
        input.upstreamUnknownActionCount = *count;
      if (auto count = object->getInteger("incomplete_facts_count"))
        input.upstreamIncompleteFactsCount = *count;
      continue;
    }
    if (*recordType != "graph")
      continue;

    GraphRow row;
    std::string fieldError;
    if (!getRequiredString(*object, "graph_id", row.graphId, fieldError) ||
        !getRequiredString(*object, "status", row.status, fieldError) ||
        !getRequiredString(*object, "path", row.path, fieldError) ||
        !getRequiredString(*object, "order", row.order, fieldError) ||
        !getRequiredString(*object, "mlir_path", row.mlirPath, fieldError) ||
        !getRequiredString(*object, "facts_path", row.factsPath, fieldError)) {
      // A malformed graph record is a per-candidate unknown.  Keep the
      // remainder of the closure available for validation; an incomplete or
      // malformed record must not turn otherwise well-formed graph rows into
      // unknown records.
      if (input.error.empty())
        input.error = fieldError;
      input.graphs.push_back(std::move(row));
      continue;
    }
    row.deduplicated = object->getBoolean("deduplicated").value_or(false);
    if (const json::Value *key = object->get("structural_key")) {
      row.structuralText = jsonText(*key);
      row.hasStructuralKey = true;
    }
    if (!row.hasStructuralKey) {
      if (input.error.empty())
        input.error = "closure graph is missing structural_key: " +
                      row.graphId;
    }
    input.graphs.push_back(std::move(row));
  }
  input.upstreamComplete = input.sawHeader && input.sawFooter &&
                           input.headerComplete && input.footerComplete &&
                           (!input.graphScopeComplete ||
                            *input.graphScopeComplete) &&
                           input.upstreamUnknownActionCount == 0 &&
                           input.upstreamIncompleteFactsCount == 0;
  input.upstreamResourceBound = input.headerResourceBound ||
                                input.footerResourceBound;
  if (!input.sawHeader || !input.sawFooter) {
    input.valid = false;
    input.error = "closure is missing header or footer";
  }
  return true;
}

static bool runFactsExtraction(ModuleOp module, StringRef functionName,
                               StringRef outputPath, std::string &error) {
  std::unique_ptr<Pass> pass =
      mlir::amoeba::neura::createExtractJointTaskGraphFactsPass();
  std::string options = "function=" + quoteOption(functionName) +
                        " output=" + quoteOption(outputPath) +
                        " fact-only=true max-tile-factor=8 "
                        "max-fission-actions-per-task=0";
  if (failed(pass->initializeOptions(options, [&](const llvm::Twine &message) {
        error = message.str();
        return failure();
      }))) {
    if (error.empty())
      error = "could not initialize graph facts pass";
    return false;
  }

  PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(std::move(pass));
  std::string diagnostic;
  LogicalResult result = failure();
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
    result = manager.run(module);
  }
  if (failed(result)) {
    error = diagnostic.empty() ? "graph facts extraction failed" : diagnostic;
    return false;
  }
  if (failed(verify(module))) {
    error = "persisted graph module failed MLIR verification";
    return false;
  }
  return true;
}

static bool compareFactsContract(const json::Value &persisted,
                                 const json::Value &extracted,
                                 StringRef closureStructural,
                                 std::string &error) {
  const json::Object *persistedRoot = persisted.getAsObject();
  const json::Object *extractedRoot = extracted.getAsObject();
  if (!persistedRoot || !extractedRoot) {
    error = "facts file root is not an object";
    return false;
  }
  const json::Value *persistedKey = persistedRoot->get("structural_key");
  const json::Value *extractedKey = extractedRoot->get("structural_key");
  if (!persistedKey || !extractedKey) {
    error = "facts file has no structural_key";
    return false;
  }
  const std::string persistedKeyText = jsonText(*persistedKey);
  if (persistedKeyText != closureStructural) {
    error = "closure structural_key differs from persisted facts";
    return false;
  }
  if (jsonText(*extractedKey) != persistedKeyText) {
    error = "persisted MLIR facts structural_key differs from closure facts";
    return false;
  }

  const json::Array *persistedTasks = persistedRoot->getArray("tasks");
  const json::Array *extractedTasks = extractedRoot->getArray("tasks");
  if (!persistedTasks || !extractedTasks ||
      persistedTasks->size() != extractedTasks->size()) {
    error = "persisted and re-extracted facts task arrays differ";
    return false;
  }
  for (auto [index, persistedValue] : llvm::enumerate(*persistedTasks)) {
    const json::Object *persistedTask = persistedValue.getAsObject();
    const json::Object *extractedTask = (*extractedTasks)[index].getAsObject();
    if (!persistedTask || !extractedTask ||
        persistedTask->getString("task_name") !=
            extractedTask->getString("task_name")) {
      error = "persisted and re-extracted task order differs";
      return false;
    }
  }
  const json::Array *persistedEdges = persistedRoot->getArray("dependencies");
  const json::Array *extractedEdges = extractedRoot->getArray("dependencies");
  if (!persistedEdges || !extractedEdges ||
      jsonArrayText(*persistedEdges) != jsonArrayText(*extractedEdges)) {
    error = "persisted and re-extracted dependency facts differ";
    return false;
  }
  return true;
}

static DagResult checkDag(const TaskEdgeGraph &graph) {
  DagResult result;
  auto tasks = graph.getTasks();
  result.nodeCount = static_cast<int64_t>(tasks.size());
  result.edgeCount = static_cast<int64_t>(graph.getEdges().size());
  DenseMap<Operation *, unsigned> ids;
  for (auto [index, task] : llvm::enumerate(tasks)) {
    TaskflowTaskOp mutableTask = task;
    ids[mutableTask.getOperation()] = index;
  }

  SmallVector<SmallVector<unsigned>> successors(tasks.size());
  SmallVector<unsigned> indegree(tasks.size(), 0);
  for (const TaskEdge &edge : graph.getEdges()) {
    TaskflowTaskOp producerTask = edge.producer;
    TaskflowTaskOp consumerTask = edge.consumer;
    auto producer = ids.find(producerTask.getOperation());
    auto consumer = ids.find(consumerTask.getOperation());
    if (producer == ids.end() || consumer == ids.end()) {
      result.error = "typed edge endpoint is outside the selected task graph";
      return result;
    }
    if (producer->second == consumer->second) {
      result.cycle = true;
      result.topologicalCount = 0;
      result.valid = true;
      return result;
    }
    successors[producer->second].push_back(consumer->second);
    ++indegree[consumer->second];
  }

  SmallVector<unsigned> ready;
  for (unsigned index = 0; index < indegree.size(); ++index)
    if (indegree[index] == 0)
      ready.push_back(index);
  while (!ready.empty()) {
    unsigned current = ready.pop_back_val();
    ++result.topologicalCount;
    for (unsigned successor : successors[current])
      if (--indegree[successor] == 0)
        ready.push_back(successor);
  }
  result.valid = true;
  result.cycle = result.topologicalCount != result.nodeCount;
  return result;
}

static bool isIdentityPath(const GraphRow &row) {
  return row.path == "identity" ||
         (row.order == "identity" &&
          StringRef(row.path).starts_with("identity/"));
}

static void applyCachedGraphBound(ModuleOp module, const GraphRow &row,
                                  const BoundInputs &inputs, uint64_t ordinal,
                                  StringRef outputFile, Verification &result) {
  if (!inputs.costManifest || !inputs.stageManifest)
    return;
  result.boundStatus = "missing-cost-coverage";
  const json::Object *cost = inputs.costManifest->getObject(row.graphId);
  if (!cost)
    return;
  auto candidates = cost->getString("candidates");
  auto catalog = cost->getString("cost_file");
  auto stage = inputs.stageManifest->getString("stage");
  auto repository = inputs.stageManifest->getString("source_repository");
  auto commit = inputs.stageManifest->getString("source_commit");
  if (!candidates || !catalog || !stage || !repository || !commit)
    return;
  const bool productionScheduler =
      inputs.temporalModel == kProductionTemporalModel;
  if (!inputs.temporalModel.empty() && !productionScheduler &&
      inputs.temporalModel != "dispatch-order")
    return;
  if (!productionScheduler && !inputs.scheduleSpace.empty() &&
      inputs.scheduleSpace != "exact")
    return;
  const StringRef boundScheduleSpace =
      productionScheduler
          ? StringRef(kProductionScheduleSpace)
          : StringRef("exact");
  const StringRef boundTemporalModel =
      productionScheduler ? StringRef(kProductionTemporalModel)
                           : StringRef("dispatch-order");
  const StringRef boundDispatchPolicy =
      productionScheduler
          ? (isProductionStageOne(*stage) ? StringRef("fixed")
                                         : StringRef("critical-path"))
          : StringRef("all-ready");
  if (productionScheduler &&
      inputs.scheduleSpace != kProductionScheduleSpace)
    return;
  if (!inputs.dispatchPolicy.empty() &&
      inputs.dispatchPolicy != boundDispatchPolicy)
    return;
  result.boundScorePath = outputFile.str() + ".bound-" +
                          std::to_string(ordinal) + ".jsonl";
  auto pass = mlir::amoeba::neura::createScoreExactJointTaskCandidatesPass();
  std::string options =
      "candidates=" + quoteOption(*candidates) +
      " cost-file=" + quoteOption(*catalog) +
      " output=" + quoteOption(result.boundScorePath) +
      " top-k=5 max-expanded-nodes=9223372036854775807 max-milliseconds=0"
      " max-makespan=0 dispatch-policy=" + boundDispatchPolicy.str() +
      " schedule-space=" + boundScheduleSpace.str() +
      " temporal-model=" + boundTemporalModel.str() +
      " graph-bound-only=true" +
      std::string(" graph-variant-id=") + quoteOption(row.graphId) +
      " source-git-repository=" + quoteOption(*repository) +
      " source-git-commit=" + quoteOption(*commit) +
      " stage=" + quoteOption(*stage) +
      " global-incumbent=" + quoteOption(inputs.incumbentPath) +
      " global-stage-manifest=" + quoteOption(inputs.stageManifestPath);
  if (failed(pass->initializeOptions(options, [&](const llvm::Twine &message) {
        result.boundDiagnostic = message.str();
        return failure();
      }))) {
    result.boundStatus = "bound-options-invalid";
    return;
  }
  PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(std::move(pass));
  LogicalResult ran = failure();
  {
    ScopedDiagnosticHandler capture(module.getContext(), [&](Diagnostic &entry) {
      if (result.boundDiagnostic.empty()) {
        llvm::raw_string_ostream stream(result.boundDiagnostic);
        entry.print(stream);
      }
      return success();
    });
    ran = manager.run(module);
  }
  if (failed(ran)) {
    result.boundStatus = "bound-proof-unavailable";
    return;
  }
  auto buffer = llvm::MemoryBuffer::getFile(result.boundScorePath);
  if (!buffer) {
    result.boundStatus = "bound-artifact-missing";
    return;
  }
  std::optional<json::Object> footer;
  std::optional<json::Object> header;
  llvm::line_iterator lines(**buffer, true);
  for (; !lines.is_at_end(); ++lines) {
    auto value = json::parse(*lines);
    if (!value) {
      llvm::consumeError(value.takeError());
      result.boundStatus = "bound-artifact-invalid";
      return;
    }
    const json::Object *object = value->getAsObject();
    if (!object || object->getString("schema") !=
                       std::optional<StringRef>("amoeba-exact-joint-task-scores-v1")) {
      result.boundStatus = "bound-artifact-invalid";
      return;
    }
    if (object->getString("record_type") ==
        std::optional<StringRef>("header"))
      header = *object;
    if (object->getString("record_type") == std::optional<StringRef>("footer"))
      footer = *object;
  }
  if (!header || !footer || !footer->getBoolean("graph_bound_only").value_or(false)) {
    result.boundStatus = "bound-artifact-invalid";
    return;
  }
  if (productionScheduler &&
      (header->getString("temporal_model") !=
           std::optional<StringRef>(kProductionTemporalModel) ||
       header->getString("schedule_space") !=
           std::optional<StringRef>(kProductionScheduleSpace) ||
       header->getBoolean("global_shape_location_temporal_space") !=
           std::optional<bool>(false) ||
       header->getBoolean("placement_coverage_limited") !=
           std::optional<bool>(true) ||
       header->getString("temporal_search_scope") !=
           std::optional<StringRef>(kProductionTemporalScope) ||
       header->getString("start_policy") !=
           std::optional<StringRef>(kProductionStartPolicy) ||
       header->getString("scheduler_backend") !=
           std::optional<StringRef>(kProductionSchedulerBackend) ||
       header->getBoolean("orbit_4x4_pruning").value_or(true) ||
       header->getString("shape_domain_pruning") !=
           std::optional<StringRef>("dependency-release-tail-cost-only") ||
       header->getString("dispatch_policy") !=
           std::optional<StringRef>(boundDispatchPolicy))) {
    result.boundStatus = "bound-artifact-invalid";
    return;
  }
  if (productionScheduler &&
      (footer->getString("temporal_model") !=
           std::optional<StringRef>(kProductionTemporalModel) ||
       footer->getString("schedule_space") !=
           std::optional<StringRef>(kProductionScheduleSpace) ||
       footer->getBoolean("global_shape_location_temporal_space") !=
           std::optional<bool>(false) ||
       footer->getBoolean("placement_coverage_limited") !=
           std::optional<bool>(true) ||
       footer->getString("temporal_search_scope") !=
           std::optional<StringRef>(kProductionTemporalScope) ||
       footer->getString("start_policy") !=
           std::optional<StringRef>(kProductionStartPolicy) ||
       footer->getString("scheduler_backend") !=
           std::optional<StringRef>(kProductionSchedulerBackend) ||
       footer->getBoolean("location_coverage_limited") !=
           std::optional<bool>(true) ||
       footer->getBoolean("temporal_coverage_limited") !=
           std::optional<bool>(true) ||
       footer->getBoolean("orbit_4x4_pruning").value_or(true) ||
       footer->getString("shape_domain_pruning") !=
           std::optional<StringRef>("dependency-release-tail-cost-only") ||
       footer->getString("dispatch_policy") !=
           std::optional<StringRef>(boundDispatchPolicy))) {
    result.boundStatus = "bound-artifact-invalid";
    return;
  }
  result.graphLowerBound =
      footer->getInteger("global_cutoff_graph_lower_bound").value_or(0);
  result.incumbentFifth = footer->getInteger("global_cutoff_cycles").value_or(0);
  const bool zeroExpansion =
      footer->getInteger("expanded_shape_nodes") == std::optional<int64_t>(0) &&
      footer->getInteger("expanded_schedule_nodes") == std::optional<int64_t>(0);
  const bool numericProof = result.incumbentFifth > 0 &&
                            result.graphLowerBound > result.incumbentFifth;
  const bool shapeDomainProof =
      footer->getBoolean("global_cutoff_shape_domain_contradiction")
          .value_or(false) &&
      footer->getString("global_cutoff_proof_kind") ==
          std::optional<StringRef>(
              "shape-domain-necessary-condition-contradiction") &&
      footer->getString("shape_domain_status") ==
          std::optional<StringRef>("whole-graph-above-global-fifth") &&
      footer->getInteger("global_cutoff_shape_domain_contradiction_task")
          .has_value();
  if (zeroExpansion &&
      footer->getString("status") == std::optional<StringRef>("globally-pruned") &&
      footer->getBoolean("global_cutoff_proven").value_or(false) &&
      footer->getBoolean("global_cutoff_graph_pruned").value_or(false) &&
      !footer->getBoolean("incomplete").value_or(true) &&
      result.incumbentFifth > 0 && (numericProof || shapeDomainProof)) {
    result.boundStatus = "proven-above-global-fifth";
    result.status = CandidateStatus::BoundPruned;
    result.reason = shapeDomainProof
                        ? "shape_domain_necessary_condition_above_global_fifth"
                        : "graph_lower_bound_above_global_fifth";
    result.proofKind = shapeDomainProof
                           ? "proven-shape-domain-global-cycle-cutoff"
                           : "proven-global-incumbent-cycle-cutoff";
    result.eligibleForML = false;
  } else if (zeroExpansion &&
             footer->getString("status") == std::optional<StringRef>("bound-only-retained")) {
    result.boundStatus = "retained-by-graph-bound";
  } else {
    result.boundStatus = "bound-proof-unavailable";
  }
}

static Verification verifyGraph(ModuleOp owner, StringRef functionName,
                                const GraphRow &row, uint64_t ordinal,
                                StringRef outputFile,
                                const BoundInputs &boundInputs) {
  Verification result;
  if (row.status != "legal") {
    result.reason = "upstream_graph_status_" + row.status;
    return result;
  }
  if (!row.hasStructuralKey) {
    result.reason = "missing_closure_structural_key";
    return result;
  }

  json::Value persistedFacts(nullptr);
  std::string error;
  if (!readFactsJsonFile(row.factsPath, persistedFacts, error)) {
    result.reason = "facts_read_failed";
    result.diagnostic = error;
    return result;
  }
  const json::Object *factsRoot = persistedFacts.getAsObject();
  if (!factsRoot) {
    result.reason = "facts_root_not_object";
    return result;
  }
  if (auto function = factsRoot->getString("function");
      !function || *function != functionName) {
    result.reason = "facts_function_mismatch";
    return result;
  }

  OwningOpRef<ModuleOp> parsed = parseSourceFile<ModuleOp>(
      row.mlirPath, ParserConfig(owner.getContext()));
  if (!parsed) {
    result.reason = "persisted_mlir_parse_failed";
    return result;
  }
  if (failed(verify(*parsed))) {
    result.reason = "persisted_mlir_verifier_failed";
    return result;
  }
  FailureOr<func::FuncOp> selected =
      selectTaskFunction(*parsed, functionName, error);
  if (failed(selected)) {
    result.reason = "persisted_mlir_function_missing";
    result.diagnostic = error;
    return result;
  }

  std::string extractedPath = outputFile.str() + ".facts-" +
                              std::to_string(ordinal) + ".json";
  if (!runFactsExtraction(*parsed, functionName, extractedPath, error)) {
    llvm::sys::fs::remove(extractedPath);
    result.reason = "facts_reextract_failed";
    result.diagnostic = error;
    return result;
  }
  json::Value extractedFacts(nullptr);
  if (!readJsonFile(extractedPath, extractedFacts, error)) {
    llvm::sys::fs::remove(extractedPath);
    result.reason = "reextracted_facts_read_failed";
    result.diagnostic = error;
    return result;
  }
  llvm::sys::fs::remove(extractedPath);
  if (!compareFactsContract(persistedFacts, extractedFacts,
                            row.structuralText, error)) {
    result.reason = "facts_structural_contract_mismatch";
    result.diagnostic = error;
    return result;
  }

  TaskEdgeGraphOptions options;
  options.require_payload = true;
  options.read_control_predecessors = true;
  FailureOr<TaskEdgeGraph> graph =
      buildTaskEdgeGraph(*selected, options, error);
  if (failed(graph)) {
    result.reason = "typed_edge_rebuild_failed";
    result.diagnostic = error;
    return result;
  }
  DagResult dag = checkDag(*graph);
  result.nodeCount = dag.nodeCount;
  result.edgeCount = dag.edgeCount;
  result.topologicalCount = dag.topologicalCount;
  if (!dag.valid) {
    result.reason = "typed_edge_dag_proof_failed";
    result.diagnostic = dag.error;
    return result;
  }
  if (dag.cycle) {
    // The identity/no-op graph is a required control candidate.  A malformed
    // identity artifact is retained as unknown under the fail-open contract;
    // only non-identity candidates with a proven typed dependency cycle are
    // pruned.
    if (isIdentityPath(row)) {
      result.reason = "identity_dependency_cycle_not_pruned";
      result.proofKind = "typed_dependency_cycle";
      return result;
    }
    result.status = CandidateStatus::Pruned;
    result.reason = "dependency_cycle";
    result.proofKind = "typed_dependency_cycle";
    return result;
  }
  result.status = CandidateStatus::Retained;
  result.reason = "typed_dependency_dag_verified";
  result.proofKind = "typed_dependency_dag";
  result.eligibleForML = true;
  applyCachedGraphBound(*parsed, row, boundInputs, ordinal, outputFile, result);
  return result;
}

static json::Object graphOutput(const GraphRow &row, const Verification &result,
                                bool duplicate, StringRef duplicateOf) {
  json::Object object{{"schema", kSchema.str()},
                      {"record_type", "graph"},
                      {"graph_id", row.graphId},
                      {"path", row.path},
                      {"order", row.order},
                      {"mlir_path", row.mlirPath},
                      {"facts_path", row.factsPath},
                      {"status", statusText(result.status).str()},
                      {"reason", result.reason},
                      {"proof_kind", result.proofKind},
                      {"bound_status", result.boundStatus},
                      {"eligible_for_ml", result.eligibleForML && !duplicate},
                      {"duplicate", duplicate},
                      {"dag_node_count", result.nodeCount},
                      {"dag_edge_count", result.edgeCount},
                      {"dag_topological_count", result.topologicalCount}};
  if (!result.diagnostic.empty())
    object["diagnostic"] = result.diagnostic;
  if (duplicate)
    object["duplicate_of_graph_id"] = duplicateOf.str();
  if (!result.boundScorePath.empty()) {
    object["bound_score_path"] = result.boundScorePath;
    object["graph_lower_bound"] = result.graphLowerBound;
    object["incumbent_fifth_cycles"] = result.incumbentFifth;
  }
  if (!result.boundDiagnostic.empty())
    object["bound_diagnostic"] = result.boundDiagnostic;
  return object;
}

class PruneJointGraphCandidatesPass
    : public PassWrapper<PruneJointGraphCandidatesPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PruneJointGraphCandidatesPass)

  PruneJointGraphCandidatesPass() = default;
  PruneJointGraphCandidatesPass(const PruneJointGraphCandidatesPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "prune-joint-graph-candidates";
  }
  StringRef getDescription() const override {
    return "Validate persisted joint graph candidates after rewrite enumeration";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    neura::NeuraDialect, TaskflowDialect>();
  }

  Option<std::string> closureFile{*this, "closure", llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output", llvm::cl::init("")};
  Option<std::string> functionName{*this, "function", llvm::cl::init("")};
  Option<std::string> costManifestFile{*this, "cost-manifest", llvm::cl::init("")};
  Option<std::string> globalIncumbentFile{*this, "global-incumbent", llvm::cl::init("")};
  Option<std::string> globalStageManifestFile{*this, "global-stage-manifest", llvm::cl::init("")};
  // The default keeps the historical exact bound contract.  New production
  // manifests may carry these fields themselves; explicit options take
  // precedence so stage drivers can bind the command body directly.
  Option<std::string> temporalModel{*this, "temporal-model", llvm::cl::init("")};
  Option<std::string> scheduleSpace{*this, "schedule-space", llvm::cl::init("")};
  Option<std::string> dispatchPolicy{*this, "dispatch-policy", llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp owner = getOperation();
    if (closureFile.empty() || outputFile.empty() || functionName.empty()) {
      owner.emitError("post-enumeration graph pruning requires closure, output, "
                      "and function options");
      return signalPassFailure();
    }

    ClosureInput input;
    readClosure(closureFile, input);
    if (!input.function.empty() && input.function != functionName) {
      input.valid = false;
      input.error = "closure function does not match the requested function";
    }
    BoundInputs boundInputs;
    json::Value costManifest(nullptr), stageManifest(nullptr);
    const bool boundsRequested = !costManifestFile.empty();
    if (boundsRequested) {
      std::string error;
      if (globalIncumbentFile.empty() || globalStageManifestFile.empty() ||
          !readJsonFile(costManifestFile, costManifest, error) ||
          !readJsonFile(globalStageManifestFile, stageManifest, error) ||
          !costManifest.getAsObject() || !stageManifest.getAsObject() ||
          stageManifest.getAsObject()->getString("function") !=
              std::optional<StringRef>(functionName.getValue()) ||
          stageManifest.getAsObject()->getString("stage") !=
              std::optional<StringRef>(input.stage)) {
        owner.emitError() << "cached graph bounds require matching cost/stage "
                            "manifests and a global incumbent: " << error;
        return signalPassFailure();
      }
      boundInputs.costManifest = costManifest.getAsObject();
      boundInputs.stageManifest = stageManifest.getAsObject();
      boundInputs.incumbentPath = globalIncumbentFile.getValue();
      boundInputs.stageManifestPath = globalStageManifestFile.getValue();
      if (!temporalModel.empty())
        boundInputs.temporalModel = temporalModel.getValue();
      else if (auto value = stageManifest.getAsObject()->getString("temporal_model"))
        boundInputs.temporalModel = value->str();
      if (!scheduleSpace.empty())
        boundInputs.scheduleSpace = scheduleSpace.getValue();
      else if (auto value = stageManifest.getAsObject()->getString("schedule_space"))
        boundInputs.scheduleSpace = value->str();
      if (!dispatchPolicy.empty())
        boundInputs.dispatchPolicy = dispatchPolicy.getValue();
      else if (auto value = stageManifest.getAsObject()->getString("dispatch_policy"))
        boundInputs.dispatchPolicy = value->str();
      if (boundInputs.temporalModel == kProductionTemporalModel &&
          boundInputs.scheduleSpace.empty())
        boundInputs.scheduleSpace = kProductionScheduleSpace.str();
    }

    struct Report {
      GraphRow row;
      Verification verification;
      bool duplicate = false;
      std::string duplicateOf;
    };
    std::vector<Report> reports;
    llvm::StringMap<unsigned> firstByStructural;
    SmallVector<std::string> eligiblePaths;
    SmallVector<std::string> eligibleIds;
    int64_t prunedCount = 0;
    int64_t retainedCount = 0;
    int64_t unknownCount = input.valid ? 0 : 1;
    int64_t duplicateCount = 0;
    int64_t boundPrunedCount = 0, boundUnavailableCount = 0;

    for (auto [ordinal, row] : llvm::enumerate(input.graphs)) {
      Report report;
      report.row = row;
      if (row.hasStructuralKey) {
        auto found = firstByStructural.find(row.structuralText);
        if (found != firstByStructural.end() &&
            row.graphId == reports[found->second].row.graphId &&
            row.mlirPath == reports[found->second].row.mlirPath &&
            row.factsPath == reports[found->second].row.factsPath) {
          report.duplicate = true;
          report.duplicateOf = reports[found->second].row.graphId;
          ++duplicateCount;
          report.verification = reports[found->second].verification;
          report.verification.eligibleForML = false;
        } else {
          firstByStructural[row.structuralText] = reports.size();
          report.verification = verifyGraph(owner, functionName, row, ordinal,
                                            outputFile, boundInputs);
        }
      } else {
        report.verification.reason = "missing_closure_structural_key";
      }

      switch (report.verification.status) {
      case CandidateStatus::Retained:
        ++retainedCount;
        if (!report.duplicate && report.verification.eligibleForML) {
          eligibleIds.push_back(row.graphId);
          eligiblePaths.push_back(row.mlirPath);
        }
        break;
      case CandidateStatus::Pruned:
        ++prunedCount;
        break;
      case CandidateStatus::BoundPruned:
        ++prunedCount;
        if (!report.duplicate)
          ++boundPrunedCount;
        break;
      case CandidateStatus::Unknown:
        ++unknownCount;
        break;
      }
      if (boundsRequested && !report.duplicate &&
          report.verification.status == CandidateStatus::Retained &&
          report.verification.boundStatus != "retained-by-graph-bound")
        ++boundUnavailableCount;
      reports.push_back(std::move(report));
    }

    bool complete = input.valid && input.upstreamComplete &&
                    !input.upstreamResourceBound && unknownCount == 0 &&
                    boundUnavailableCount == 0;
    // This describes whether this pass consumed the complete persisted input
    // artifact.  It is intentionally independent from upstreamComplete:
    // bounded closure enumeration can be fully checked here while remaining
    // incomplete as an all-candidate search.
    bool staticPruningComplete = input.readable && input.valid;
    std::string writeError;
    bool wrote = writeAtomically(
        outputFile,
        [&](llvm::raw_ostream &stream) {
          json::Object header{
              {"schema", kSchema.str()},
              {"record_type", "header"},
              {"function", functionName.getValue()},
              {"pre_ml", true},
              {"ml_inference_calls", 0},
              {"joint_search_nodes", 0},
              {"cached_graph_bounds_requested", boundsRequested},
              {"pruning_phase", "post-enumeration"},
              {"complete", complete},
              {"static_pruning_complete", staticPruningComplete},
              {"status", complete ? "complete" : "incomplete"},
              {"production_ready", false},
              {"upstream_complete", input.upstreamComplete},
              {"upstream_resource_bound_reached", input.upstreamResourceBound},
              {"upstream_status", input.upstreamStatus},
              {"upstream_stage", input.stage},
              {"graph_artifact_directory", input.graphArtifactDirectory},
              {"fail_open", true}};
          if (boundsRequested &&
              boundInputs.temporalModel == kProductionTemporalModel) {
            header["temporal_model"] = kProductionTemporalModel.str();
            header["schedule_space"] = kProductionScheduleSpace.str();
            header["temporal_search_scope"] = kProductionTemporalScope.str();
            header["start_policy"] = kProductionStartPolicy.str();
            header["scheduler_backend"] = kProductionSchedulerBackend.str();
            header["dispatch_policy"] = boundInputs.dispatchPolicy.empty()
                                              ? (isProductionStageOne(input.stage)
                                                     ? "fixed"
                                                     : "critical-path")
                                              : boundInputs.dispatchPolicy;
            header["global_shape_location_temporal_space"] = false;
            header["placement_coverage_limited"] = true;
            header["location_coverage_limited"] = true;
            header["temporal_coverage_limited"] = true;
            header["orbit_4x4_pruning"] = false;
            header["shape_domain_pruning"] =
                "dependency-release-tail-cost-only";
          }
          if (input.upstreamGraphCount >= 0)
            header["upstream_graph_count"] = input.upstreamGraphCount;
          if (input.upstreamUnknownActionCount >= 0)
            header["upstream_unknown_action_count"] =
                input.upstreamUnknownActionCount;
          if (input.upstreamIncompleteFactsCount >= 0)
            header["upstream_incomplete_facts_count"] =
                input.upstreamIncompleteFactsCount;
          writeJsonLine(stream, std::move(header));
          for (const Report &report : reports)
            writeJsonLine(stream, graphOutput(report.row, report.verification,
                                              report.duplicate,
                                              report.duplicateOf));
          json::Array retainedIds;
          for (const std::string &id : eligibleIds)
            retainedIds.push_back(id);
          json::Array retainedPaths;
          for (const std::string &path : eligiblePaths)
            retainedPaths.push_back(path);
          json::Object footer{
              {"schema", kSchema.str()},
              {"record_type", "footer"},
              {"complete", complete},
              {"static_pruning_complete", staticPruningComplete},
              {"status", complete ? "complete" : "incomplete"},
              {"production_ready", false},
              {"pre_ml", true},
              {"pruning_phase", "post-enumeration"},
              {"upstream_complete", input.upstreamComplete},
              {"upstream_resource_bound_reached", input.upstreamResourceBound},
              {"input_graph_count", static_cast<int64_t>(reports.size())},
              {"retained_graph_count", retainedCount},
              {"pruned_graph_count", prunedCount},
              {"bound_pruned_unique_graph_count", boundPrunedCount},
              {"bound_unavailable_unique_graph_count", boundUnavailableCount},
              {"bound_pruning_complete", !boundsRequested || boundUnavailableCount == 0},
              {"ml_inference_calls", 0},
              {"joint_search_nodes", 0},
              {"unknown_graph_count", unknownCount},
              {"duplicate_graph_count", duplicateCount},
              {"eligible_retained_graph_ids", std::move(retainedIds)},
              {"eligible_retained_paths", std::move(retainedPaths)},
              {"first_blocker", input.valid ? "" : input.error}};
          if (boundsRequested &&
              boundInputs.temporalModel == kProductionTemporalModel) {
            footer["temporal_model"] = kProductionTemporalModel.str();
            footer["schedule_space"] = kProductionScheduleSpace.str();
            footer["temporal_search_scope"] = kProductionTemporalScope.str();
            footer["start_policy"] = kProductionStartPolicy.str();
            footer["scheduler_backend"] = kProductionSchedulerBackend.str();
            footer["dispatch_policy"] = boundInputs.dispatchPolicy.empty()
                                             ? (isProductionStageOne(input.stage)
                                                    ? "fixed"
                                                    : "critical-path")
                                             : boundInputs.dispatchPolicy;
            footer["global_shape_location_temporal_space"] = false;
            footer["placement_coverage_limited"] = true;
            footer["location_coverage_limited"] = true;
            footer["temporal_coverage_limited"] = true;
            footer["orbit_4x4_pruning"] = false;
            footer["shape_domain_pruning"] =
                "dependency-release-tail-cost-only";
            footer["ranking_certificate"] =
                complete ? "proven-shape-space-under-production-scheduler"
                         : "best-found-uncertified";
          }
          writeJsonLine(stream, std::move(footer));
          return true;
        },
        writeError);
    if (!wrote) {
      owner.emitError() << "cannot write post-enumeration pruning output: "
                        << writeError;
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createPruneJointGraphCandidatesPass() {
  return std::make_unique<PruneJointGraphCandidatesPass>();
}
} // namespace mlir::amoeba::neura
