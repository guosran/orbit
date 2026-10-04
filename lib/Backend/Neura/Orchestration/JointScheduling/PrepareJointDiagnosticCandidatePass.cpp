//===- PrepareJointDiagnosticCandidatePass.cpp ---------------------------===//
//
// Emit one explicitly requested, source-bound complete-program candidate.
// This pass is deliberately diagnostic: it materializes no search space and
// makes no ranking or winner claim. Shape validation and cost provenance are
// still the same contracts used by the analytical candidate/replay passes,
// and the score is produced by the production scheduler adapter.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalTaskCostCatalog.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "NeighborhoodReplaySelection.h"
#include "ProductionSchedulerAdapter.h"

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;
namespace json = llvm::json;

namespace {

constexpr StringLiteral kScoreSchema = "amoeba-exact-joint-task-scores-v1";
constexpr StringLiteral kSearchScope =
    "budgeted-complete-program-neighborhood";
constexpr StringLiteral kDiagnosticTag =
    "explicit-complete-program-diagnostic";
constexpr StringLiteral kScoreSource =
    "ml-ii-startup-plus-production-explicit-communication";
constexpr StringLiteral kLatencyProvenance =
    "source-owned-ml-ii-startup-trip-count-contract";
constexpr StringLiteral kStartPolicy =
    "production-spatial-temporal-scheduler";
constexpr StringLiteral kTemporalSearchScope =
    "production-task-scheduler-only";

static std::string jsonText(const json::Value &value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << value;
  stream.flush();
  return text;
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
    error = "diagnostic candidate output directory is empty";
    return false;
  }
  std::error_code ec = llvm::sys::fs::create_directories(path);
  if (ec) {
    error = "cannot create diagnostic candidate output directory " +
            path.str() + ": " + ec.message();
    return false;
  }
  return true;
}

static bool readFileBytes(StringRef path, std::string &bytes,
                          std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read " + path.str() + ": " + buffer.getError().message();
    return false;
  }
  bytes = (*buffer)->getBuffer().str();
  return true;
}

static bool architectureBindingMatches(StringRef requested,
                                       std::string &error) {
  if (requested.empty()) {
    error = "architecture-path is required";
    return false;
  }
  const std::string &configured =
      ::mlir::amoeba::getNeuraArchitectureSpecFile();
  if (configured.empty()) {
    error = "the active scheduler has no architecture specification";
    return false;
  }
  if (configured == requested || samePath(configured, requested))
    return true;
  std::string configuredBytes;
  std::string requestedBytes;
  if (!readFileBytes(configured, configuredBytes, error) ||
      !readFileBytes(requested, requestedBytes, error))
    return false;
  if (configuredBytes != requestedBytes) {
    error = "architecture-path bytes differ from the active scheduler "
            "architecture";
    return false;
  }
  return true;
}

static bool parseShape(StringRef text, int64_t &rows, int64_t &cols) {
  size_t separator = text.find('x');
  if (separator == StringRef::npos)
    separator = text.find('X');
  if (separator == StringRef::npos)
    return false;
  StringRef rowText = text.take_front(separator).trim();
  StringRef colText = text.drop_front(separator + 1).trim();
  return !rowText.empty() && !colText.empty() &&
         !rowText.getAsInteger(10, rows) &&
         !colText.getAsInteger(10, cols) && rows > 0 && cols > 0;
}

static std::optional<int64_t> durationFromCost(const TaskShapeCost &cost,
                                               int64_t tripCount,
                                               std::string &error) {
  if (!cost.supported)
    return std::nullopt;
  if (tripCount <= 0) {
    error = "task trip count must be positive";
    return std::nullopt;
  }
  // This is the established source-owned duration contract: startup plus
  // the per-iteration II for every execution after the first, rounded up
  // once before it is passed to the production scheduler.
  long double duration = static_cast<long double>(cost.startupCycles) +
                         static_cast<long double>(cost.predictedII) *
                             static_cast<long double>(tripCount - 1);
  if (!std::isfinite(duration) || duration <= 0.0L ||
      duration > static_cast<long double>(
                     std::numeric_limits<int64_t>::max())) {
    error = "ML II/startup duration is outside scheduler range";
    return std::nullopt;
  }
  duration = std::ceil(duration);
  return std::max<int64_t>(1, static_cast<int64_t>(duration));
}

static bool parseShapeOverrides(
    StringRef text, ArrayRef<TaskMetadata> tasks,
    std::map<std::string, std::pair<int64_t, int64_t>> &overrides,
    std::string &error) {
  overrides.clear();
  if (text.trim().empty())
    return true;

  std::set<std::string> taskNames;
  for (const TaskMetadata &task : tasks)
    taskNames.insert(task.name);
  size_t begin = 0;
  while (begin <= text.size()) {
    size_t end = text.find(';', begin);
    if (end == StringRef::npos)
      end = text.size();
    StringRef entry = text.slice(begin, end).trim();
    if (entry.empty()) {
      error = "task-shapes contains an empty override";
      return false;
    }
    size_t separator = entry.find(':');
    if (separator == StringRef::npos ||
        entry.find(':', separator + 1) != StringRef::npos) {
      error = "task-shapes entry must have the form TaskName:RxC";
      return false;
    }
    StringRef task = entry.take_front(separator).trim();
    StringRef shape = entry.drop_front(separator + 1).trim();
    if (task.empty() || !taskNames.count(task.str())) {
      error = "task-shapes names an unknown task: " + task.str();
      return false;
    }
    int64_t rows = 0;
    int64_t cols = 0;
    if (!parseShape(shape, rows, cols)) {
      error = "task-shapes has an invalid rectangle for " + task.str();
      return false;
    }
    if (!overrides.emplace(task.str(), std::make_pair(rows, cols)).second) {
      error = "task-shapes contains a duplicate override for " + task.str();
      return false;
    }
    if (end == text.size())
      break;
    begin = end + 1;
  }
  return true;
}

static json::Array costRecords(ArrayRef<TaskShapeChoice> choices,
                               ArrayRef<TaskShapeCost> costs,
                               ArrayRef<int64_t> durations) {
  json::Array result;
  for (size_t index = 0; index < choices.size(); ++index) {
    json::Object entry{
        {"task", choices[index].task},
        {"mapper_tile_rows", choices[index].shape.mapperRows},
        {"mapper_tile_cols", choices[index].shape.mapperCols},
        {"trip_count", choices[index].tripCount},
        {"predicted_duration", durations[index]},
        {"predicted_ii", costs[index].predictedII},
        {"startup_cycles", costs[index].startupCycles},
        {"support_status",
         costs[index].supported ? "supported" : "unsupported"}};
    result.push_back(std::move(entry));
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
    result.push_back(json::Object{
        {"task", choice.task},
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

static json::Array dispatchRecords(ArrayRef<unsigned> dispatch,
                                   ArrayRef<TaskShapeChoice> choices) {
  json::Array result;
  for (unsigned task : dispatch)
    if (task < choices.size())
      result.push_back(choices[task].task);
  return result;
}

static json::Object makeScoreRecord(
    StringRef candidateId, StringRef graphId, int64_t makespan,
    ArrayRef<TaskShapeChoice> choices, ArrayRef<TaskShapeCost> costs,
    ArrayRef<int64_t> durations, const ProductionScheduledResult &schedule) {
  return json::Object{
      {"schema", kScoreSchema},
      {"record_type", "score"},
      {"candidate_id", candidateId},
      {"shape_candidate_id", candidateId},
      {"graph_variant_id", graphId},
      {"valid", true},
      {"predicted_whole_program_cycles", makespan},
      {"score_source", kScoreSource},
      {"task_latency_provenance", kLatencyProvenance},
      {"task_costs", costRecords(choices, costs, durations)},
      {"task_schedule", scheduleRecords(choices, schedule.placements)},
      {"dispatch_order", dispatchRecords(schedule.dispatch, choices)},
      {"replayed_communication_edges",
       static_cast<int64_t>(schedule.replayedEdges)},
      {"communication_trace_known", schedule.replayedEdgesKnown},
      {"best_found", true},
      {"certified", false},
      {"exhaustive", false},
      {"diagnostic_only", true},
      {"diagnostic_tag", kDiagnosticTag},
      {"search_winner_claim", false}};
}

class PrepareJointDiagnosticCandidatePass
    : public PassWrapper<PrepareJointDiagnosticCandidatePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      PrepareJointDiagnosticCandidatePass)

  PrepareJointDiagnosticCandidatePass() = default;
  PrepareJointDiagnosticCandidatePass(
      const PrepareJointDiagnosticCandidatePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "prepare-joint-diagnostic-candidate";
  }
  StringRef getDescription() const override {
    return "Emit one source-bound explicit joint scheduling diagnostic candidate";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, taskflow::TaskflowDialect>();
  }

  Option<std::string> functionName{*this, "function", llvm::cl::init("")};
  Option<std::string> parentCostFile{*this, "parent-cost-file",
                                     llvm::cl::init("")};
  Option<std::string> sourceRepository{*this, "source-repository",
                                       llvm::cl::init("")};
  Option<std::string> sourceCommit{*this, "source-commit",
                                   llvm::cl::init("")};
  Option<std::string> architecturePath{*this, "architecture-path",
                                       llvm::cl::init("")};
  Option<std::string> candidateId{*this, "candidate-id", llvm::cl::init("")};
  Option<std::string> outputDir{*this, "output-dir", llvm::cl::init("")};
  Option<std::string> dispatchPolicy{*this, "dispatch-policy",
                                     llvm::cl::init("fixed")};
  Option<std::string> taskShapes{*this, "task-shapes", llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    auto fail = [&](StringRef message) {
      module.emitError() << message;
      signalPassFailure();
    };

    if (functionName.empty() || parentCostFile.empty() ||
        sourceRepository.empty() || sourceCommit.empty() ||
        architecturePath.empty() || candidateId.empty() || outputDir.empty()) {
      fail("prepare-joint-diagnostic-candidate requires function, "
           "parent-cost-file, source-repository, source-commit, "
           "architecture-path, candidate-id, and output-dir");
      return;
    }
    if (dispatchPolicy != "fixed" && dispatchPolicy != "critical-path") {
      fail("dispatch-policy must be fixed or critical-path");
      return;
    }
    if (!architectureBindingMatches(architecturePath, error)) {
      fail(error);
      return;
    }

    auto selected = selectTaskFunction(module, functionName, error);
    if (failed(selected)) {
      fail(error.empty() ? "cannot select requested Taskflow function" : error);
      return;
    }
    func::FuncOp function = *selected;
    auto graph = function->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
    if (!graph || graph.getValue().empty()) {
      fail("selected function lacks a non-empty amoeba.graph_variant_id");
      return;
    }

    FailureOr<SmallVector<TaskMetadata>> metadataResult =
        collectAnalyticalTaskMetadata(function, error);
    if (failed(metadataResult)) {
      fail(error.empty() ? "cannot collect canonical task metadata" : error);
      return;
    }
    SmallVector<TaskMetadata> tasks = std::move(*metadataResult);
    if (tasks.empty()) {
      fail("selected function has no Taskflow tasks");
      return;
    }

    std::map<std::string, std::pair<int64_t, int64_t>> overrides;
    if (!parseShapeOverrides(taskShapes, tasks, overrides, error)) {
      fail(error);
      return;
    }

    const auto &architecture = ::mlir::neura::getArchitecture();
    SmallVector<RectShape> legalShapes = enumerateStaticRectShapes(
        architecture.getMultiCgraRows(), architecture.getMultiCgraColumns(),
        architecture.getPerCgraRows(), architecture.getPerCgraColumns(), 4);
    if (legalShapes.empty()) {
      fail("architecture has no legal area-four oriented rectangles");
      return;
    }

    Candidate candidate;
    candidate.id = candidateId;
    SmallVector<TaskShapeCost> costs;
    SmallVector<int64_t> durations;
    candidate.choices.reserve(tasks.size());
    costs.reserve(tasks.size());
    durations.reserve(tasks.size());

    // Load the complete parent catalogue before any output is published. This
    // catches stale source IDs, graph identity, architecture, Git provenance,
    // and missing shape bindings before a partial replay record can escape.
    TaskShapeCostCache costCache;
    if (!costCache.load(parentCostFile, functionName, tasks, sourceRepository,
                        sourceCommit, architecturePath, graph.getValue(),
                        error)) {
      fail(error);
      return;
    }

    for (const TaskMetadata &task : tasks) {
      int64_t rows = 1;
      int64_t cols = 1;
      auto override = overrides.find(task.name);
      if (override != overrides.end()) {
        rows = override->second.first;
        cols = override->second.second;
      }
      auto shape = llvm::find_if(legalShapes, [&](const RectShape &legal) {
        return legal.rows == rows && legal.cols == cols;
      });
      if (shape == legalShapes.end()) {
        error = "task-shapes selects an unsupported area-four oriented shape for " +
                task.name + ": " + std::to_string(rows) + "x" +
                std::to_string(cols);
        fail(error);
        return;
      }
      candidate.choices.push_back({task.name, task.tripCount, *shape});
      std::string lookupError;
      const TaskShapeCost *cost =
          costCache.get(candidate.choices.back(), lookupError);
      if (!cost) {
        error = lookupError;
        fail(error);
        return;
      }
      if (!cost->supported) {
        error = "requested task shape has an unsupported cost binding for " +
                task.name;
        fail(error);
        return;
      }
      auto duration = durationFromCost(*cost, task.tripCount, error);
      if (!duration) {
        if (error.empty())
          error = "requested task shape has no finite duration for " + task.name;
        fail(error);
        return;
      }
      costs.push_back(*cost);
      durations.push_back(*duration);
    }

    ProductionScheduledResult schedule;
    if (!scheduleWithAmoebaProductionPass(
            module, function, candidate.choices, durations, candidate.id,
            dispatchPolicy == "fixed", schedule, error)) {
      fail(error.empty() ? "production scheduler rejected diagnostic candidate"
                         : error);
      return;
    }
    if (schedule.makespan <= 0 || !schedule.replayedEdgesKnown ||
        schedule.placements.size() != candidate.choices.size() ||
        schedule.dispatch.size() != candidate.choices.size()) {
      fail("production scheduler returned an incomplete diagnostic schedule");
      return;
    }

    auto configuredIterations = function->getAttrOfType<IntegerAttr>(
        "joint_scheduling_fixed_point_max_iterations");
    const int64_t fixedPointIterations =
        configuredIterations ? configuredIterations.getInt() : 64;
    if (fixedPointIterations <= 0 ||
        fixedPointIterations > std::numeric_limits<int>::max()) {
      fail("joint_scheduling_fixed_point_max_iterations is outside the "
           "supported range");
      return;
    }

    // Build every output in memory before publishing any valid replay file.
    // In particular, all negative input cases above leave the output directory
    // untouched.
    if (!ensureDirectory(outputDir, error)) {
      fail(error);
      return;
    }
    const std::string candidatePath = outputDir + "/candidate.mlir";
    const std::string shapePath = outputDir + "/shape-selection.json";
    const std::string scorePath = outputDir + "/score.jsonl";
    const std::string selectionPath = outputDir + "/selection.jsonl";

    json::Object scoreRecord = makeScoreRecord(
        candidate.id, graph.getValue(), schedule.makespan, candidate.choices,
        costs, durations, schedule);
    json::Object header{
        {"record_type", "header"},
        {"schema", kScoreSchema},
        {"function", functionName},
        {"schedule_space", "production-scheduler"},
        {"temporal_model", "production-scheduler"},
        {"dispatch_policy", dispatchPolicy},
        {"start_policy", kStartPolicy},
        {"temporal_search_scope", kTemporalSearchScope},
        {"production_fixed_point_max_iterations", fixedPointIterations},
        {"search_scope", kSearchScope},
        {"best_found", true},
        {"exhaustive", false},
        {"source_repository", sourceRepository},
        {"source_commit", sourceCommit},
        {"architecture_path", architecturePath},
        {"source_binding_witness", candidatePath},
        {"diagnostic_only", true},
        {"diagnostic_tag", kDiagnosticTag},
        {"search_winner_claim", false}};
    json::Object footer{{"record_type", "footer"},
                        {"schema", kScoreSchema},
                        {"scored_count", int64_t(1)},
                        {"status", "diagnostic-candidate"},
                        {"certified", false},
                        {"diagnostic_only", true},
                        {"diagnostic_tag", kDiagnosticTag},
                        {"search_winner_claim", false}};
    std::string scoreText = jsonText(json::Value(std::move(header))) + "\n" +
                            jsonText(json::Value(json::Object(scoreRecord))) + "\n" +
                            jsonText(json::Value(std::move(footer))) + "\n";

    json::Object selection{
        {"record_type", "selection"},
        {"schema", "orbit-neighborhood-search-v1"},
        {"candidate_id", candidate.id},
        {"shape_candidate_id", candidate.id},
        {"function", functionName},
        {"graph_variant_id", graph.getValue()},
        {"rank", int64_t(0)},
        {"best_found", true},
        {"certified", false},
        {"valid", true},
        {"predicted_whole_program_cycles", schedule.makespan},
        {"candidate_path", candidatePath},
        {"mapper_replay_path", candidatePath},
        {"score_file", scorePath},
        {"score_file_path", scorePath},
        {"shape_manifest_file", shapePath},
        {"candidate_manifest", shapePath},
        {"score_record", json::Value(json::Object(scoreRecord))},
        {"source_repository", sourceRepository},
        {"source_commit", sourceCommit},
        {"architecture_path", architecturePath},
        {"search_scope", kSearchScope},
        {"diagnostic_only", true},
        {"diagnostic_tag", kDiagnosticTag},
        {"search_winner_claim", false}};
    json::Object candidateRecord = candidateJson(candidate);
    if (const json::Array *shapes = candidateRecord.getArray("task_shapes"))
      selection["task_shapes"] = json::Array(*shapes);
    json::Array seedShapes;
    for (const auto &choice : candidate.choices)
      seedShapes.push_back(json::Object{{"task", choice.task},
                                       {"rows", choice.shape.rows},
                                       {"cols", choice.shape.cols}});
    selection["shapes"] = std::move(seedShapes);

    std::string selectionText =
        jsonText(json::Value(std::move(selection))) + "\n";
    std::string sourceText = neighborhoodReplaySourceText(module);

    if (!writeTextAtomically(candidatePath, sourceText, error) ||
        !writeNeighborhoodShapeSelection(module, function, candidate, shapePath,
                                          error) ||
        !writeTextAtomically(scorePath, scoreText, error) ||
        !writeTextAtomically(selectionPath, selectionText, error)) {
      fail(error.empty() ? "cannot publish diagnostic candidate outputs" : error);
      return;
    }
  }
};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createPrepareJointDiagnosticCandidatePass() {
  return std::make_unique<PrepareJointDiagnosticCandidatePass>();
}
} // namespace mlir::amoeba::neura
