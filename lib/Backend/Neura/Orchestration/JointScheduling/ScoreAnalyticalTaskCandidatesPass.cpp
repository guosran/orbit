//===- ScoreAnalyticalTaskCandidatesPass.cpp -----------------------------===//
//
// Implements full-manifest scoring and deterministic top-k selection.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateManifest.h"
#include "AnalyticalTaskCostCatalog.h"

#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/AnalyticalBasedTaskOrchestration/AnalyticalBasedTaskOrchestration.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.h"

#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {

struct FactoredSpaceDescriptor {
  std::string function;
  std::string candidateCount;
  SmallVector<SmallVector<RectShape>> shapesByTask;
};

struct FactoredRankedCandidate {
  std::string id;
  std::string ordinal;
  double score = 0.0;
  llvm::json::Object scoreRecord;
};

static std::optional<std::string>
getDecimal(const llvm::json::Object &object, StringRef key,
           std::string &error) {
  if (std::optional<int64_t> numeric = object.getInteger(key)) {
    if (*numeric < 0) {
      error = "factored space count must be nonnegative: " + key.str();
      return std::nullopt;
    }
    return std::to_string(*numeric);
  }
  std::optional<StringRef> text = object.getString(key);
  if (!text || text->empty() ||
      llvm::any_of(*text, [](char value) {
        return value < '0' || value > '9';
      })) {
    error = "factored space count is not a decimal string: " + key.str();
    return std::nullopt;
  }
  StringRef normalized = text->ltrim('0');
  return (normalized.empty() ? StringRef("0") : normalized).str();
}

static int compareDecimal(StringRef lhs, StringRef rhs) {
  if (lhs.size() != rhs.size())
    return lhs.size() < rhs.size() ? -1 : 1;
  if (lhs == rhs)
    return 0;
  return lhs < rhs ? -1 : 1;
}

static void putDecimal(llvm::json::Object &object, StringRef key,
                       StringRef decimal) {
  uint64_t value = 0;
  if (!decimal.getAsInteger(10, value) &&
      value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    object[key] = static_cast<int64_t>(value);
  else
    object[key] = decimal.str();
}


static bool isFactoredSpaceDescriptor(StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return false;
  for (llvm::line_iterator lines(**buffer, /*SkipBlanks=*/true);
       !lines.is_at_end(); ++lines) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
    if (!parsed)
      return false;
    const llvm::json::Object *object = parsed->getAsObject();
    if (!object)
      return false;
    auto recordType = object->getString("record_type");
    auto schema = object->getString("schema");
    return recordType && schema && *recordType == "space" &&
           *schema == kFactoredSpaceSchema;
  }
  return false;
}

static bool parseFactoredSpaceDescriptor(
    StringRef path, func::FuncOp func, ArrayRef<TaskMetadata> tasks,
    StringRef expectedGraphId,
    const ::mlir::neura::Architecture &architecture,
    FactoredSpaceDescriptor &descriptor, std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read factored candidate space " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }

  std::optional<llvm::json::Object> headerStorage;
  std::optional<llvm::json::Object> footerStorage;
  size_t records = 0;
  for (llvm::line_iterator lines(**buffer, /*SkipBlanks=*/true);
       !lines.is_at_end(); ++lines) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
    if (!parsed) {
      error = "invalid factored candidate JSONL: " +
              llvm::toString(parsed.takeError());
      return false;
    }
    const llvm::json::Object *object = parsed->getAsObject();
    if (!object) {
      error = "factored candidate record is not an object";
      return false;
    }
    auto recordType = object->getString("record_type");
    if (!recordType) {
      error = "factored candidate record has no record_type";
      return false;
    }
    if (*recordType == "space" && !headerStorage && records == 0) {
      headerStorage = std::move(*object);
    } else if (*recordType == "footer" && headerStorage && !footerStorage &&
               records == 1) {
      footerStorage = std::move(*object);
    } else {
      error = "factored candidate space must contain exactly one header and "
              "one footer";
      return false;
    }
    ++records;
  }
  if (!headerStorage || !footerStorage || records != 2) {
    error = "factored candidate space is missing its header or footer";
    return false;
  }
  const llvm::json::Object &header = *headerStorage;
  const llvm::json::Object &footer = *footerStorage;

  auto schema = header.getString("schema");
  auto representation = header.getString("representation");
  auto exact = header.getBoolean("exact");
  auto function = header.getString("function");
  auto graphVariantId = header.getString("graph_variant_id");
  auto searchScope = header.getString("search_scope");
  auto shapePolicy = header.getString("shape_policy");
  auto capacityPolicy = header.getString("spatial_capacity_policy");
  auto pruningPolicy = header.getString("shape_pruning_policy");
  auto maxChangedTasks = header.getInteger("max_changed_tasks");
  auto maxCgras = header.getInteger("max_cgras_per_task");
  const llvm::json::Array *factors = header.getArray("factors");
  if (!schema || !representation || !exact || !function || !graphVariantId ||
      !searchScope || !shapePolicy || !capacityPolicy || !pruningPolicy ||
      !maxChangedTasks ||
      !maxCgras || !factors) {
    error = "factored candidate header is missing required fields";
    return false;
  }
  if (*schema != kFactoredSpaceSchema || *representation != "factored" ||
      !*exact || *function != func.getSymName() ||
      expectedGraphId.empty() || *graphVariantId != expectedGraphId ||
      *searchScope != kFactoredSearchScope || *shapePolicy != kShapePolicy ||
      *capacityPolicy != kSpatialCapacityPolicy ||
      *pruningPolicy != kShapePruningPolicyNone || *maxChangedTasks != -1 ||
      *maxCgras <= 0) {
    error = "factored candidate header does not match the exact shape-space "
            "contract";
    return false;
  }
  if (factors->size() != tasks.size()) {
    error = "factored candidate factors do not match current task count";
    return false;
  }
  if (*maxCgras > architecture.getMultiCgraRows() *
                      architecture.getMultiCgraColumns()) {
    error = "factored candidate max-cgras-per-task exceeds the architecture";
    return false;
  }

  SmallVector<RectShape> expectedShapes = enumerateStaticRectShapes(
      architecture.getMultiCgraRows(), architecture.getMultiCgraColumns(),
      architecture.getPerCgraRows(), architecture.getPerCgraColumns(),
      *maxCgras);
  if (expectedShapes.empty()) {
    error = "factored candidate shape alphabet is empty";
    return false;
  }
  descriptor.shapesByTask.clear();
  descriptor.shapesByTask.reserve(tasks.size());
  for (auto [taskIndex, factorValue] : llvm::enumerate(*factors)) {
    const llvm::json::Object *factor = factorValue.getAsObject();
    if (!factor) {
      error = "factored candidate factor is not an object";
      return false;
    }
    auto taskName = factor->getString("task");
    auto tripCount = factor->getInteger("trip_count");
    const llvm::json::Array *factorShapes = factor->getArray("shapes");
    if (!taskName || !tripCount || !factorShapes ||
        *taskName != tasks[taskIndex].name ||
        *tripCount != tasks[taskIndex].tripCount ||
        factorShapes->size() != expectedShapes.size()) {
      error = "factored candidate factor does not match current task metadata";
      return false;
    }
    SmallVector<RectShape> parsedShapes;
    parsedShapes.reserve(factorShapes->size());
    for (auto [shapeIndex, shapeValue] : llvm::enumerate(*factorShapes)) {
      const llvm::json::Object *shape = shapeValue.getAsObject();
      if (!shape) {
        error = "factored candidate shape is not an object";
        return false;
      }
      auto rows = shape->getInteger("rows");
      auto cols = shape->getInteger("cols");
      auto mapperRows = shape->getInteger("mapper_tile_rows");
      auto mapperCols = shape->getInteger("mapper_tile_cols");
      auto cgraCount = shape->getInteger("cgra_count");
      if (!rows || !cols || !mapperRows || !mapperCols || !cgraCount ||
          *rows != expectedShapes[shapeIndex].rows ||
          *cols != expectedShapes[shapeIndex].cols ||
          *mapperRows != expectedShapes[shapeIndex].mapperRows ||
          *mapperCols != expectedShapes[shapeIndex].mapperCols ||
          *cgraCount != expectedShapes[shapeIndex].cgraCount()) {
        error = "factored candidate shape alphabet is not canonical for the "
                "current architecture";
        return false;
      }
      parsedShapes.push_back(expectedShapes[shapeIndex]);
    }
    descriptor.shapesByTask.push_back(std::move(parsedShapes));
  }

  auto candidateCount = getDecimal(header, "candidate_count", error);
  auto footerCount = getDecimal(footer, "candidate_count", error);
  auto footerSchema = footer.getString("schema");
  auto ranked = footer.getBoolean("ranked");
  auto status = footer.getString("status");
  if (!candidateCount || !footerCount || !footerSchema || !ranked || !status ||
      *footerSchema != kFactoredSpaceSchema || *ranked ||
      *status != "unranked-factored-space") {
    if (error.empty())
      error = "factored candidate footer is not an unranked exact descriptor";
    return false;
  }
  ShapeCandidateSpaceCounts expected = countShapeCandidateSpace(
      descriptor.shapesByTask, kShapePruningPolicyNone, -1);
  if (expected.declared.empty() || *candidateCount != expected.declared ||
      *footerCount != expected.declared) {
    error = "factored candidate count does not match the canonical Cartesian "
            "space";
    return false;
  }
  descriptor.function = function->str();
  descriptor.candidateCount = *candidateCount;
  return true;
}

struct ScoreAnalyticalTaskCandidatesPass
    : public PassWrapper<ScoreAnalyticalTaskCandidatesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ScoreAnalyticalTaskCandidatesPass)

  ScoreAnalyticalTaskCandidatesPass() = default;
  ScoreAnalyticalTaskCandidatesPass(
      const ScoreAnalyticalTaskCandidatesPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "score-analytical-task-candidates";
  }
  StringRef getDescription() const override {
    return "Scores every frozen candidate through a shared task-shape ML "
           "cost cache, then emits deterministic top-k";
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Taskflow function; inferred when exactly one exists."),
      llvm::cl::init("")};
  Option<std::string> candidateFile{
      *this, "candidates", llvm::cl::desc("Frozen candidate JSONL path."),
      llvm::cl::init("")};
  Option<std::string> costFile{
      *this, "cost-file", llvm::cl::desc("Task-shape cost catalogue path."),
      llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output",
                                 llvm::cl::desc("Score JSONL output path."),
                                 llvm::cl::init("")};
  Option<int64_t> topK{
      *this, "top-k",
      llvm::cl::desc("Uses zero to select every valid candidate."),
      llvm::cl::init(1)};
  Option<int64_t> maxFactoredEvaluations{
      *this, "max-factored-evaluations",
      llvm::cl::desc("Stops factored traversal after this many candidates; "
                     "zero is unlimited."),
      llvm::cl::init(0)};
  Option<int64_t> factoredTimeBudgetMs{
      *this, "factored-time-budget-ms",
      llvm::cl::desc("Stops factored traversal after this many milliseconds; "
                     "zero is unlimited."),
      llvm::cl::init(0)};
  Option<std::string> communicationMode{
      *this, "communication-mode",
      llvm::cl::desc("Task communication mode: 'none' or 'explicit'."),
      llvm::cl::init("none")};
  Option<std::string> schedulingMode{
      *this, "scheduling-mode",
      llvm::cl::desc("Candidate scheduler: 'spatial' or 'spatial-temporal'."),
      llvm::cl::init("spatial-temporal")};
  Option<std::string> dispatchPolicy{
      *this, "dispatch-policy",
      llvm::cl::desc("Dependency-ready dispatch: 'fixed' or 'critical-path'."),
      llvm::cl::init("critical-path")};
  Option<std::string> graphVariantId{
      *this, "graph-variant-id",
      llvm::cl::desc("Optional semantic graph variant identity."),
      llvm::cl::init("")};
  Option<std::string> graphVariantSha256{
      *this, "graph-variant-sha256",
      llvm::cl::desc("Legacy file fingerprint; rejected by the no-SHA contract."),
      llvm::cl::init("")};
  Option<std::string> sourceRepository{
      *this, "source-git-repository",
      llvm::cl::desc("Git repository recorded by the cost catalogue."),
      llvm::cl::init("")};
  Option<std::string> sourceCommit{
      *this, "source-git-commit",
      llvm::cl::desc("Git commit recorded by the cost catalogue."),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selectedFunction =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selectedFunction)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp func = *selectedFunction;
    // File-content fingerprints are not part of the candidate identity
    // contract. Reject a legacy option instead of silently dropping it.
    if (!graphVariantSha256.getValue().empty()) {
      func.emitError() << "graph-variant-sha256 is unsupported by the no-SHA "
                          "candidate contract";
      return signalPassFailure();
    }
    if (candidateFile.getValue().empty() || costFile.getValue().empty() ||
        outputFile.getValue().empty() || topK.getValue() < 0 ||
        maxFactoredEvaluations.getValue() < 0 ||
        factoredTimeBudgetMs.getValue() < 0 ||
        samePath(candidateFile.getValue(), outputFile.getValue()) ||
        samePath(costFile.getValue(), outputFile.getValue()) ||
        samePath(candidateFile.getValue(), costFile.getValue())) {
      func.emitError() << "distinct candidates, cost-file, and output paths "
                          "and non-negative top-k are required";
      return signalPassFailure();
    }

    FailureOr<SmallVector<TaskMetadata>> taskMetadata =
        collectAnalyticalTaskMetadata(func, error);
    if (failed(taskMetadata)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    const bool factoredInput =
        isFactoredSpaceDescriptor(candidateFile.getValue());
    if (factoredInput && topK.getValue() == 0) {
      func.emitError()
          << "factored exact ranking requires a positive top-k; selecting "
             "every candidate would materialize the factored score space";
      return signalPassFailure();
    }
    FactoredSpaceDescriptor factoredDescriptor;

    // Bind the catalogue to explicit source and architecture provenance.
    // Candidate ordering and exact count are independently validated by
    // readCandidateManifest below; no file-content hash is required.
    StringRef architecturePath =
        ::mlir::amoeba::getNeuraArchitectureSpecFile();
    if (sourceRepository.getValue().empty() || sourceCommit.getValue().empty() ||
        architecturePath.empty()) {
      func.emitError()
          << "source-git-repository, source-git-commit, and "
             "--architecture-spec are required for cost provenance";
      return signalPassFailure();
    }
    const ::mlir::neura::Architecture &architecture =
        ::mlir::neura::getArchitecture();
    if (factoredInput && !parseFactoredSpaceDescriptor(
                             candidateFile.getValue(), func, *taskMetadata,
                             graphVariantId.getValue(), architecture,
                             factoredDescriptor, error)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    TaskShapeCostCache costs;
    if (!costs.load(costFile.getValue(), func.getSymName(), *taskMetadata,
                    sourceRepository.getValue(), sourceCommit.getValue(),
                    architecturePath, graphVariantId.getValue(), error)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    if (factoredInput &&
        costs.candidateCountDecimal() != factoredDescriptor.candidateCount) {
      func.emitError()
          << "factored candidate count does not match predictor catalogue "
             "candidate_count";
      return signalPassFailure();
    }
    if (!factoredInput && !costs.candidateCountFitsInt64()) {
      func.emitError()
          << "row candidate catalogue candidate_count exceeds signed 64-bit "
             "manifest contract";
      return signalPassFailure();
    }
    if (communicationMode.getValue() != "none" &&
        communicationMode.getValue() != "explicit") {
      func.emitError() << "unknown communication mode: "
                       << communicationMode.getValue();
      return signalPassFailure();
    }
    const bool communicationAware = communicationMode.getValue() == "explicit";
    if (dispatchPolicy.getValue() != "fixed" &&
        dispatchPolicy.getValue() != "critical-path") {
      func.emitError() << "unknown dispatch policy: "
                       << dispatchPolicy.getValue();
      return signalPassFailure();
    }
    taskflow::SchedulingMode mode;
    if (schedulingMode.getValue() == "spatial")
      mode = taskflow::SchedulingMode::Spatial;
    else if (schedulingMode.getValue() == "spatial-temporal")
      mode = taskflow::SchedulingMode::SpatialTemporal;
    else {
      func.emitError() << "unknown scheduling mode: "
                       << schedulingMode.getValue();
      return signalPassFailure();
    }
    FailureOr<std::optional<InterTaskNetworkSpec>> network =
        loadInterTaskNetworkSpec(::mlir::amoeba::getNeuraArchitectureSpecFile(),
                                 /*require_network=*/communicationAware, error);
    if (failed(network)) {
      func.emitError() << error;
      return signalPassFailure();
    }

    // Row manifests retain the historical full ranking. Factored descriptors
    // are traversed exactly but retain only the requested top-k records.
    SmallVector<RankedCandidate> ranked;
    SmallVector<FactoredRankedCandidate> factoredRanked;
    ManifestHeader manifestHeader;
    ManifestFooter manifestFooter;
    uint64_t scoredCount = 0;
    uint64_t validCount = 0;
    unsigned factoredCounterWidth = 64;
    if (factoredInput) {
      if (taskMetadata->size() >
          (std::numeric_limits<unsigned>::max() / 64u) - 1u) {
        func.emitError() << "factored task count exceeds exact counter width";
        return signalPassFailure();
      }
      factoredCounterWidth = std::max<unsigned>(
          64u, 64u * static_cast<unsigned>(taskMetadata->size() + 1));
    }
    llvm::APInt factoredScoredCount(factoredCounterWidth, 0);
    llvm::APInt factoredValidCount(factoredCounterWidth, 0);
    std::map<std::string, llvm::APInt> factoredRejectionCounts;
    bool factoredBudgetHit = false;
    bool factoredComplete = false;
    std::string factoredBudgetReason;
    const auto factoredStart = std::chrono::steady_clock::now();
    bool wrote = writeAtomically(
        outputFile.getValue(),
        [&](llvm::raw_ostream &os) {
          llvm::json::Object scoreHeader;
          scoreHeader["record_type"] = "header";
          scoreHeader["schema"] = kScoreSchema.str();
          scoreHeader["candidate_schema"] =
              (factoredInput ? kFactoredSpaceSchema : kCandidateSchema).str();
          scoreHeader["function"] = func.getSymName().str();
          scoreHeader["cost_namespace"] = costs.nameSpace().str();
          // Candidate IDs and the score model remain in the score header.
          // Provenance uses Git and declared architecture identifiers; no
          // file-content fingerprints are emitted.
          scoreHeader["source_repository"] = costs.sourceRepository().str();
          scoreHeader["source_commit"] = costs.sourceCommit().str();
          scoreHeader["architecture_path"] = costs.architecturePath().str();
          scoreHeader["architecture_schema"] = costs.architectureSchema().str();
          scoreHeader["source_graph_id"] = costs.sourceGraphId().str();
          scoreHeader["candidate_id_scheme"] =
              costs.candidateIdScheme().str();
          if (factoredInput)
            putDecimal(scoreHeader, "candidate_count",
                       factoredDescriptor.candidateCount);
          else
            scoreHeader["candidate_count"] =
                static_cast<int64_t>(costs.candidateCount());
          if (factoredInput) {
            scoreHeader["representation"] = "factored";
            scoreHeader["exact"] = true;
            scoreHeader["candidate_rows_materialized"] = false;
            scoreHeader["max_factored_evaluations"] =
                maxFactoredEvaluations.getValue();
            scoreHeader["factored_time_budget_ms"] =
                factoredTimeBudgetMs.getValue();
          }
          scoreHeader["score_model"] = kScoreModel.str();
          scoreHeader["communication_mode"] = communicationMode.getValue();
          scoreHeader["scheduling_mode"] = schedulingMode.getValue();
          scoreHeader["dispatch_policy"] = dispatchPolicy.getValue();
          if (!graphVariantId.getValue().empty())
            scoreHeader["graph_variant_id"] = graphVariantId.getValue();
          scoreHeader["mapper_success_probability"] =
              costs.mapperSuccessProbabilityRole().str();
          writeJsonLine(os, std::move(scoreHeader));

          manifestHeader.gridRows = architecture.getMultiCgraRows();
          manifestHeader.gridCols = architecture.getMultiCgraColumns();
          manifestHeader.perCgraRows = architecture.getPerCgraRows();
          manifestHeader.perCgraCols = architecture.getPerCgraColumns();
          auto noteScored = [&]() {
            if (factoredInput)
              factoredScoredCount += 1;
            else
              ++scoredCount;
          };
          auto noteFactoredRejection = [&](StringRef reason) {
            if (!factoredInput)
              return;
            auto found = factoredRejectionCounts.find(reason.str());
            if (found == factoredRejectionCounts.end())
              factoredRejectionCounts.emplace(
                  reason.str(), llvm::APInt(factoredCounterWidth, 1));
            else
              found->second += 1;
          };
          auto retainFactored = [&](llvm::json::Object score, StringRef id,
                                    StringRef ordinal, double makespan) {
            factoredRanked.push_back(
                {id.str(), ordinal.str(), makespan, std::move(score)});
            llvm::sort(factoredRanked,
                      [](const FactoredRankedCandidate &lhs,
                         const FactoredRankedCandidate &rhs) {
                        if (lhs.score != rhs.score)
                          return lhs.score < rhs.score;
                        return compareDecimal(lhs.ordinal, rhs.ordinal) < 0;
                      });
            if (factoredRanked.size() >
                static_cast<size_t>(topK.getValue()))
              factoredRanked.pop_back();
          };

          auto consume = [&](StringRef ordinal, const Candidate &candidate,
                             std::string &consumeError) {
            // duration(task, shape) = startup + II * (trip_count - 1)
            // score(candidate)      = production scheduler makespan
            bool valid = true;
            double bottleneck = 0.0;
            SmallVector<int64_t> schedulerDurations;
            std::string rejectReason;
            std::string rejectClass;
            llvm::json::Array taskCosts;
            for (const TaskShapeChoice &choice : candidate.choices) {
              const TaskShapeCost *cost = costs.get(choice, consumeError);
              if (!cost)
                return false;
              llvm::json::Object taskCost;
              taskCost["task"] = choice.task;
              taskCost["mapper_tile_rows"] = choice.shape.mapperRows;
              taskCost["mapper_tile_cols"] = choice.shape.mapperCols;
              taskCost["trip_count"] = choice.tripCount;
              if (!cost->supported) {
                valid = false;
                if (rejectReason.empty()) {
                  rejectReason = "UNSUPPORTED_TASK_SHAPE";
                  rejectClass = "definitely_illegal_shape";
                }
                taskCost["support_status"] = "unsupported";
              } else {
                taskCost["predicted_ii"] = cost->predictedII;
                taskCost["startup_cycles"] = cost->startupCycles;
                double duration = cost->startupCycles +
                                  cost->predictedII *
                                      static_cast<double>(choice.tripCount - 1);
                if (!std::isfinite(duration)) {
                  consumeError =
                      "task duration overflow for task=" + choice.task +
                      ", mapper_shape=rect-" +
                      std::to_string(choice.shape.mapperRows) + "x" +
                      std::to_string(choice.shape.mapperCols);
                  return false;
                }
                taskCost["support_status"] = "supported";
                taskCost["predicted_duration"] = duration;
                bottleneck = std::max(bottleneck, duration);
                double scheduledDuration = std::ceil(duration);
                if (scheduledDuration >
                    static_cast<double>(std::numeric_limits<int64_t>::max())) {
                  consumeError =
                      "task duration exceeds scheduler range for task=" +
                      choice.task;
                  return false;
                }
                schedulerDurations.push_back(std::max<int64_t>(
                    1, static_cast<int64_t>(scheduledDuration)));
              }
              taskCosts.push_back(std::move(taskCost));
            }

            llvm::json::Object score;
            score["record_type"] = "score";
            score["schema"] = kScoreSchema.str();
            score["candidate_id"] = candidate.id;
            score["valid"] = valid;
            score["task_costs"] = std::move(taskCosts);
            if (valid) {
              score["predicted_compute_bottleneck"] = bottleneck;
              // The clone receives the same fixed shapes and task priority as
              // final orchestration. Only its predicted durations differ.
              OwningOpRef<ModuleOp> clonedModule = module.clone();
              FailureOr<func::FuncOp> clonedFunction = selectTaskFunction(
                  *clonedModule, func.getSymName(), consumeError);
              if (failed(clonedFunction))
                return false;
              SmallVector<taskflow::TaskflowTaskOp> clonedTasks;
              clonedFunction->walk([&](taskflow::TaskflowTaskOp task) {
                clonedTasks.push_back(task);
              });
              if (clonedTasks.size() != candidate.choices.size() ||
                  clonedTasks.size() != schedulerDurations.size()) {
                consumeError = "candidate task count changed while cloning";
                return false;
              }
              OpBuilder builder(func.getContext());
              for (auto [task, choice, duration] : llvm::zip_equal(
                       clonedTasks, candidate.choices, schedulerDurations)) {
                task->setAttr("cgra_count", builder.getI32IntegerAttr(
                                                choice.shape.cgraCount()));
                task->setAttr(
                    "cgra_shape",
                    builder.getStringAttr(choice.shape.toCgraShapeAttrValue()));
                task->setAttr("amoeba.joint_shape_orientation_fixed",
                              builder.getUnitAttr());
                task->setAttr("est_latency",
                              builder.getI64IntegerAttr(duration));
              }
              taskflow::AnalyticalBasedTaskOrchestration strategy(
                  static_cast<int>(manifestHeader.gridRows),
                  static_cast<int>(manifestHeader.gridCols),
                  mode, /*communication_aware=*/communicationAware,
                  /*fixed_dispatch=*/dispatchPolicy.getValue() == "fixed");
              std::unique_ptr<InterTaskNetworkCommunicationModel>
                  communicationModel;
              if (communicationAware) {
                FailureOr<TaskEdgeGraph> edgeGraph = buildTaskEdgeGraph(
                    *clonedFunction,
                    TaskEdgeGraphOptions{/*require_payload=*/true,
                                         /*read_control_predecessors=*/true},
                    consumeError);
                if (failed(edgeGraph))
                  return false;
                communicationModel =
                    std::make_unique<InterTaskNetworkCommunicationModel>(
                        std::move(*edgeGraph), **network);
              }
              taskflow::TaskScheduler scheduler(
                  static_cast<int>(manifestHeader.gridRows),
                  static_cast<int>(manifestHeader.gridCols),
                  mode, taskflow::ShapeSelectionPolicy::FixedOrientation,
                  communicationModel.get());
              // A fixed-shape candidate can make the heuristic SRAM fixed
              // point oscillate even though its shape is individually legal.
              // Capture that one diagnostic locally so this candidate can be
              // censored while malformed graphs, placement failures, and
              // other scheduler invariants remain fail-closed below.
              std::string schedulerDiagnostic;
              bool schedulerAccepted = false;
              {
                ScopedDiagnosticHandler schedulerDiagnostics(
                    clonedFunction->getContext(),
                    [&](Diagnostic &diagnostic) {
                      if (diagnostic.getSeverity() ==
                          DiagnosticSeverity::Error)
                        schedulerDiagnostic = diagnostic.str();
                    });
                schedulerAccepted = scheduler.schedule(
                    *clonedFunction,
                    strategy.computeTaskPriority(*clonedFunction));
              }
              if (!schedulerAccepted) {
                if (StringRef(schedulerDiagnostic).contains(
                        "task placement and SRAM assignment did not converge")) {
                  score["valid"] = false;
                  score["reject_class"] = "heuristic_scheduler_failure";
                  score["reject_reason"] = "SCHEDULER_NO_CONVERGENCE";
                  score["scheduler_failure"] =
                      "task placement and SRAM assignment did not converge";
                  noteFactoredRejection("SCHEDULER_NO_CONVERGENCE");
                  if (!factoredInput)
                    writeJsonLine(os, std::move(score));
                  noteScored();
                  return true;
                }
                consumeError =
                    schedulingMode.getValue() +
                    " scheduler rejected candidate " +
                    candidate.id;
                if (!schedulerDiagnostic.empty())
                  consumeError += ": " + schedulerDiagnostic;
                return false;
              }
              int64_t makespan = scheduler.getScheduleMakespan();
              if (makespan <= 0) {
                consumeError = "scheduler produced an empty or overflowing "
                               "makespan for " +
                               candidate.id;
                return false;
              }
              score["predicted_scheduler_makespan"] = makespan;
              llvm::json::Array taskSchedule;
              std::string traceIdentity;
              llvm::raw_string_ostream identity(traceIdentity);
              identity << "joint-schedule-trace-v2\n"
                       << candidate.id << "\n"
                       << communicationMode.getValue() << "\n";
              for (taskflow::TaskflowTaskOp task : clonedTasks) {
                auto scheduleEntry = llvm::find_if(
                    scheduler.getScheduleEntries(), [&](const auto &entry) {
                      return entry.task == task.getOperation();
                    });
                if (scheduleEntry == scheduler.getScheduleEntries().end()) {
                  consumeError = "scheduler did not emit a complete task trace";
                  return false;
                }
                llvm::json::Object taskRecord;
                taskRecord["task"] = task.getTaskName().str();
                taskRecord["start_cycle"] = scheduleEntry->startCycle;
                taskRecord["end_cycle"] = scheduleEntry->endCycle;
                identity << "T|" << task.getTaskName().size() << ":"
                         << task.getTaskName() << "|"
                         << scheduleEntry->startCycle << "|"
                         << scheduleEntry->endCycle;
                llvm::json::Array cells;
                for (auto [row, col] : scheduleEntry->positions) {
                  llvm::json::Object cell;
                  cell["row"] = row;
                  cell["col"] = col;
                  identity << "|" << row << "," << col;
                  cells.push_back(std::move(cell));
                }
                identity << "\n";
                taskRecord["cgra_positions"] = std::move(cells);
                taskSchedule.push_back(std::move(taskRecord));
              }
              llvm::json::Array routes;
              llvm::json::Array dependencies;
              if (communicationModel) {
                for (const TaskEdge &edge :
                     communicationModel->getTypedEdges()) {
                  taskflow::TaskflowTaskOp producer = edge.producer;
                  taskflow::TaskflowTaskOp consumer = edge.consumer;
                  llvm::json::Object dependency;
                  dependency["producer"] = producer.getTaskName().str();
                  dependency["consumer"] = consumer.getTaskName().str();
                  dependency["kind"] = stringifyTaskEdgeKind(edge.kind).str();
                  dependency["producer_segment"] =
                      stringifyTaskResultSegment(edge.producer_segment).str();
                  dependency["producer_index"] = edge.producer_index;
                  dependency["consumer_segment"] =
                      stringifyTaskOperandSegment(edge.consumer_segment).str();
                  dependency["consumer_index"] = edge.consumer_index;
                  if (edge.payload_bits)
                    dependency["payload_bits"] =
                        static_cast<int64_t>(*edge.payload_bits);
                  identity << "D|" << producer.getTaskName().size() << ":"
                           << producer.getTaskName() << "|"
                           << consumer.getTaskName().size() << ":"
                           << consumer.getTaskName() << "|"
                           << stringifyTaskEdgeKind(edge.kind) << "|"
                           << stringifyTaskResultSegment(edge.producer_segment)
                           << "|" << edge.producer_index << "|"
                           << stringifyTaskOperandSegment(edge.consumer_segment)
                           << "|" << edge.consumer_index << "|";
                  if (edge.payload_bits)
                    identity << *edge.payload_bits;
                  else
                    identity << "-";
                  identity << "\n";
                  dependencies.push_back(std::move(dependency));
                }
                for (const auto &transfer :
                     communicationModel->getCommittedTransfers()) {
                  llvm::json::Object route;
                  route["producer"] =
                      cast<taskflow::TaskflowTaskOp>(transfer.producer)
                          .getTaskName()
                          .str();
                  route["consumer"] =
                      cast<taskflow::TaskflowTaskOp>(transfer.consumer)
                          .getTaskName()
                          .str();
                  if (transfer.payloadBits >
                          static_cast<uint64_t>(
                              std::numeric_limits<int64_t>::max()) ||
                      transfer.pathLatencyCycles >
                          static_cast<uint64_t>(
                              std::numeric_limits<int64_t>::max()) ||
                      transfer.bottleneckBandwidthBitsPerCycle >
                          static_cast<uint64_t>(
                              std::numeric_limits<int64_t>::max()) ||
                      transfer.transferCycles >
                          static_cast<uint64_t>(
                              std::numeric_limits<int64_t>::max())) {
                    consumeError =
                        "communication trace value exceeds signed 64-bit";
                    return false;
                  }
                  route["payload_bits"] =
                      static_cast<int64_t>(transfer.payloadBits);
                  route["path_latency_cycles"] =
                      static_cast<int64_t>(transfer.pathLatencyCycles);
                  route["bottleneck_bandwidth_bits_per_cycle"] =
                      static_cast<int64_t>(
                          transfer.bottleneckBandwidthBitsPerCycle);
                  route["transfer_cycles"] =
                      static_cast<int64_t>(transfer.transferCycles);
                  route["ready_cycle"] = transfer.readyCycle;
                  route["source_row"] = transfer.source.row;
                  route["source_col"] = transfer.source.column;
                  route["destination_row"] = transfer.destination.row;
                  route["destination_col"] = transfer.destination.column;
                  StringRef producerName =
                      cast<taskflow::TaskflowTaskOp>(transfer.producer)
                          .getTaskName();
                  StringRef consumerName =
                      cast<taskflow::TaskflowTaskOp>(transfer.consumer)
                          .getTaskName();
                  identity << "R|" << producerName.size() << ":" << producerName
                           << "|" << consumerName.size() << ":" << consumerName
                           << "|" << transfer.source.row << ","
                           << transfer.source.column << "|"
                           << transfer.destination.row << ","
                           << transfer.destination.column << "|"
                           << transfer.payloadBits << "|" << transfer.readyCycle
                           << "|" << transfer.pathLatencyCycles << "|"
                           << transfer.bottleneckBandwidthBitsPerCycle << "|"
                           << transfer.transferCycles;
                  llvm::json::Array edgeIndices;
                  for (uint32_t edgeIndex : transfer.edgeIndices) {
                    edgeIndices.push_back(edgeIndex);
                    identity << "|e" << edgeIndex;
                  }
                  route["edge_indices"] = std::move(edgeIndices);
                  llvm::json::Array links;
                  for (const auto &interval : transfer.links) {
                    llvm::json::Object link;
                    link["start_cycle"] = interval.startCycle;
                    link["end_cycle"] = interval.endCycle;
                    if (interval.localChannel) {
                      link["resource_kind"] = "local_channel";
                      link["row"] = interval.localCoordinate.row;
                      link["col"] = interval.localCoordinate.column;
                      identity << "|c" << interval.localCoordinate.row << ","
                               << interval.localCoordinate.column << ","
                               << interval.startCycle << ","
                               << interval.endCycle;
                    } else {
                      link["resource_kind"] = "network_link";
                      link["link_index"] = interval.linkIndex;
                      identity << "|l" << interval.linkIndex << ","
                               << interval.startCycle << ","
                               << interval.endCycle;
                    }
                    links.push_back(std::move(link));
                  }
                  identity << "\n";
                  route["links"] = std::move(links);
                  routes.push_back(std::move(route));
                }
              }
              llvm::json::Object trace;
              trace["candidate_id"] = candidate.id;
              trace["communication_mode"] = communicationMode.getValue();
              trace["task_schedule"] = std::move(taskSchedule);
              trace["dependencies"] = std::move(dependencies);
              trace["routes"] = std::move(routes);
              llvm::json::Value traceValue(std::move(trace));
              identity.flush();
              score["schedule_trace"] = std::move(traceValue);
              if (factoredInput) {
                retainFactored(std::move(score), candidate.id, ordinal,
                               static_cast<double>(makespan));
                factoredValidCount += 1;
              } else {
                uint64_t manifestIndex = 0;
                if (ordinal.getAsInteger(10, manifestIndex)) {
                  consumeError = "row candidate ordinal is not a uint64";
                  return false;
                }
                ranked.push_back({candidate.id, manifestIndex,
                                  static_cast<double>(makespan)});
                writeJsonLine(os, std::move(score));
                ++validCount;
              }
            } else {
              score["reject_reason"] = rejectReason;
              score["reject_class"] = rejectClass;
              noteFactoredRejection(rejectReason);
              if (!factoredInput)
                writeJsonLine(os, std::move(score));
            }
            noteScored();
            return true;
          };

          llvm::json::Array shortlist;
          if (factoredInput) {
            bool traversed = visitShapeCartesianProductExact(
                factoredDescriptor.shapesByTask,
                [&](const llvm::APInt &index,
                    ArrayRef<size_t> shapeIndices) {
                  if (maxFactoredEvaluations.getValue() > 0 &&
                      factoredScoredCount.uge(llvm::APInt(
                          factoredCounterWidth,
                          static_cast<uint64_t>(
                              maxFactoredEvaluations.getValue())))) {
                    factoredBudgetHit = true;
                    factoredBudgetReason = "max-factored-evaluations";
                    return false;
                  }
                  if (factoredTimeBudgetMs.getValue() > 0) {
                    auto elapsed = std::chrono::duration_cast<
                        std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - factoredStart)
                                       .count();
                    if (elapsed >= factoredTimeBudgetMs.getValue()) {
                      factoredBudgetHit = true;
                      factoredBudgetReason = "factored-time-budget-ms";
                      return false;
                    }
                  }
                  std::string ordinal = exactShapeIndexString(index);
                  Candidate candidate;
                  candidate.id = "candidate-" + ordinal;
                  for (auto [taskIndex, shapeIndex] :
                       llvm::enumerate(shapeIndices)) {
                    candidate.choices.push_back(
                        {(*taskMetadata)[taskIndex].name,
                         (*taskMetadata)[taskIndex].tripCount,
                         factoredDescriptor.shapesByTask[taskIndex][shapeIndex]});
                  }
                  return consume(ordinal, candidate, error);
                });
            if (!traversed && !factoredBudgetHit) {
              if (error.empty())
                error = "exact factored candidate traversal did not complete";
              return false;
            }
            llvm::APInt expectedCount(factoredCounterWidth,
                                      factoredDescriptor.candidateCount, 10);
            factoredComplete = !factoredBudgetHit;
            if (factoredComplete && factoredScoredCount != expectedCount) {
              error = "exact factored scorer did not visit every candidate";
              return false;
            }
            if (factoredComplete &&
                costs.coveredQueries() != costs.catalogQueries()) {
              error = "cost catalogue does not exactly cover factored task-shape "
                      "queries";
              return false;
            }
            if (factoredComplete && factoredValidCount.ult(llvm::APInt(
                    factoredCounterWidth,
                    static_cast<uint64_t>(topK.getValue())))) {
              error = "exact factored scorer found fewer valid candidates than "
                      "the requested top-k";
              return false;
            }
            llvm::sort(factoredRanked,
                      [](const FactoredRankedCandidate &lhs,
                         const FactoredRankedCandidate &rhs) {
                        if (lhs.score != rhs.score)
                          return lhs.score < rhs.score;
                        return compareDecimal(lhs.ordinal, rhs.ordinal) < 0;
                      });
            for (FactoredRankedCandidate &candidate : factoredRanked)
              writeJsonLine(os, std::move(candidate.scoreRecord));
            for (auto [index, candidate] : llvm::enumerate(factoredRanked)) {
              llvm::json::Object item;
              item["rank"] = static_cast<int64_t>(index);
              item["candidate_id"] = candidate.id;
              item["predicted_scheduler_makespan"] = candidate.score;
              shortlist.push_back(std::move(item));
            }
          } else {
            auto rowConsume = [&](uint64_t index, const Candidate &candidate,
                                  std::string &consumeError) {
              std::string ordinal = std::to_string(index);
              return consume(ordinal, candidate, consumeError);
            };
            if (!readCandidateManifest(
                    candidateFile.getValue(), *taskMetadata, func.getSymName(),
                    ::mlir::neura::getArchitecture(), rowConsume,
                    manifestHeader, manifestFooter, error))
              return false;
            if (manifestHeader.graphVariantId != graphVariantId.getValue() ||
                manifestHeader.graphVariantId != costs.sourceGraphId() ||
                manifestFooter.candidateCount != costs.candidateCount() ||
                scoredCount != manifestFooter.candidateCount) {
              error = "not every frozen candidate was scored under the bound "
                      "Git, graph, architecture, and candidate contract";
              return false;
            }
            if (costs.coveredQueries() != costs.catalogQueries()) {
              error = "cost catalogue does not exactly cover candidate manifest "
                      "task-shape queries";
              return false;
            }

            // Sort only after complete traversal. Numeric manifest order is the
            // deterministic tie-break, avoiding candidate-13 < candidate-5.
            llvm::sort(ranked, [](const RankedCandidate &lhs,
                                  const RankedCandidate &rhs) {
              if (lhs.score != rhs.score)
                return lhs.score < rhs.score;
              return lhs.manifestIndex < rhs.manifestIndex;
            });
            uint64_t selected =
                topK.getValue() == 0
                    ? ranked.size()
                    : std::min<uint64_t>(topK.getValue(), ranked.size());
            for (uint64_t index = 0; index < selected; ++index) {
              llvm::json::Object item;
              item["rank"] = static_cast<int64_t>(index);
              item["candidate_id"] = ranked[index].id;
              item["predicted_scheduler_makespan"] = ranked[index].score;
              shortlist.push_back(std::move(item));
            }
          }

          llvm::json::Object cacheStats;
          cacheStats["hits"] = static_cast<int64_t>(costs.hits());
          cacheStats["misses"] = static_cast<int64_t>(costs.misses());
          cacheStats["predictions"] =
              static_cast<int64_t>(costs.cachedPredictions());
          cacheStats["queries"] = static_cast<int64_t>(costs.coveredQueries());
          llvm::json::Object footer;
          footer["record_type"] = "footer";
          footer["schema"] = kScoreSchema.str();
          if (factoredInput) {
            putDecimal(footer, "candidate_count",
                       factoredDescriptor.candidateCount);
            std::string scoredDecimal = exactShapeIndexString(factoredScoredCount);
            std::string validDecimal = exactShapeIndexString(factoredValidCount);
            putDecimal(footer, "scored_count", scoredDecimal);
            putDecimal(footer, "valid_count", validDecimal);
            footer["representation"] = "factored";
            footer["exact"] = true;
            footer["candidate_rows_materialized"] = false;
            footer["explored_count"] = scoredDecimal;
            footer["pruned_count"] = 0;
            llvm::APInt expectedCount(factoredCounterWidth,
                                      factoredDescriptor.candidateCount, 10);
            llvm::APInt unexploredCount = expectedCount - factoredScoredCount;
            std::string unexploredDecimal =
                exactShapeIndexString(unexploredCount);
            putDecimal(footer, "unexplored_count", unexploredDecimal);
            putDecimal(footer, "unmeasured_count", unexploredDecimal);
            llvm::json::Array rejectionSummary;
            for (const auto &entry : factoredRejectionCounts) {
              llvm::json::Object rejection;
              rejection["reason"] = entry.first;
              std::string count = exactShapeIndexString(entry.second);
              putDecimal(rejection, "count", count);
              rejectionSummary.push_back(std::move(rejection));
            }
            footer["rejection_summary"] = std::move(rejectionSummary);
            // A production heuristic failure is not a proof that no legal
            // 4x4 placement exists for this shape combination. Keep its
            // diagnostic but never certify a global top five over it.
            auto unresolved =
                factoredRejectionCounts.find("SCHEDULER_NO_CONVERGENCE");
            const bool heuristicUnresolved =
                unresolved != factoredRejectionCounts.end();
            if (heuristicUnresolved)
              putDecimal(footer, "unresolved_candidate_count",
                         exactShapeIndexString(unresolved->second));
            if (factoredComplete && !heuristicUnresolved) {
              footer["enumeration_status"] = "complete";
              footer["evaluation_status"] = "complete";
              footer["incomplete"] = false;
              footer["top_k_certified"] = true;
              footer["ranking_certificate"] =
                  "exhaustive-factored-cartesian";
            } else if (factoredComplete) {
              footer["enumeration_status"] = "complete";
              footer["evaluation_status"] = "incomplete-heuristic";
              footer["incomplete"] = true;
              footer["top_k_certified"] = false;
              footer["ranking_status"] = "best-found-uncertified";
            } else {
              footer["enumeration_status"] = "incomplete-budget";
              footer["evaluation_status"] = "incomplete-budget";
              footer["incomplete"] = true;
              footer["budget_hit"] = true;
              footer["budget_reason"] = factoredBudgetReason;
              footer["ranking_status"] = "best-found-uncertified";
            }
          } else {
            footer["candidate_count"] =
                static_cast<int64_t>(manifestFooter.candidateCount);
            footer["scored_count"] = static_cast<int64_t>(scoredCount);
            footer["valid_count"] = static_cast<int64_t>(validCount);
          }
          footer["top_k_requested"] = topK.getValue();
          footer["shortlist"] = std::move(shortlist);
          footer["cache"] = std::move(cacheStats);
          writeJsonLine(os, std::move(footer));
          return true;
        },
        error);
    if (!wrote) {
      func.emitError() << error;
      return signalPassFailure();
    }
    if (factoredInput)
      llvm::errs() << "[JointScheduling] scored factored candidates with "
                   << costs.cachedPredictions()
                   << " cached task/shape predictions\n";
    else
      llvm::errs() << "[JointScheduling] scored all " << scoredCount
                   << " candidates with " << costs.cachedPredictions()
                   << " cached task/shape predictions\n";
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createScoreAnalyticalTaskCandidatesPass() {
  return std::make_unique<ScoreAnalyticalTaskCandidatesPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
