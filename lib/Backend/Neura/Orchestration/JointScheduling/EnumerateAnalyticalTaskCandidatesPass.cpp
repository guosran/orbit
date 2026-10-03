//===- EnumerateAnalyticalTaskCandidatesPass.cpp -------------------------===//
//
// Implements the pass that freezes the complete rectangular task-shape
// Cartesian product.
//
//===----------------------------------------------------------------------===//

#include "SpatialTaskCandidateSpace.h"

#include "Backend/Neura/NeuraBackendPasses.h"

#include "NeuraDialect/Architecture/Architecture.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {

struct EnumerateAnalyticalTaskCandidatesPass
    : public PassWrapper<EnumerateAnalyticalTaskCandidatesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      EnumerateAnalyticalTaskCandidatesPass)

  EnumerateAnalyticalTaskCandidatesPass() = default;
  EnumerateAnalyticalTaskCandidatesPass(
      const EnumerateAnalyticalTaskCandidatesPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "enumerate-analytical-task-candidates";
  }
  StringRef getDescription() const override {
    return "Freezes every rectangular task-shape candidate without scoring or "
           "running the mapper";
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Taskflow function; inferred when exactly one exists."),
      llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output",
                                 llvm::cl::desc("Candidate JSONL output path."),
                                 llvm::cl::init("")};
  Option<int64_t> maxCandidates{
      *this, "max-candidates",
      llvm::cl::desc("Fails rather than publishing a partial manifest."),
      llvm::cl::init(1000000)};
  Option<bool> factoredOutput{
      *this, "factored-output",
      llvm::cl::desc("Writes one exact factored space descriptor instead of candidate rows."),
      llvm::cl::init(false)};
  Option<int64_t> maxCgrasPerTask{
      *this, "max-cgras-per-task",
      llvm::cl::desc(
          "Limits each task footprint; zero uses the full physical grid."),
      llvm::cl::init(0)};
  Option<std::string> searchPolicy{
      *this, "search-policy",
      llvm::cl::desc(
          "Shape policy: complete-cartesian only; changed-task pruning was removed."),
      llvm::cl::init("")};
  Option<std::string> shapePruningPolicyOption{
      *this, "shape-pruning-policy",
      llvm::cl::desc(
          "Alias for search-policy for the shape-pruning contract."),
      llvm::cl::init("")};
  Option<int64_t> maxChangedTasks{
      *this, "max-changed-tasks",
      llvm::cl::desc(
          "Legacy pruning option; any value other than -1 is rejected."),
      llvm::cl::init(-1)};
  Option<std::string> graphVariantId{
      *this, "graph-variant-id",
      llvm::cl::desc("Optional semantic graph variant identity."),
      llvm::cl::init("")};
  Option<std::string> graphVariantSha256{
      *this, "graph-variant-sha256",
      llvm::cl::desc("Legacy file fingerprint; rejected by the no-SHA contract."),
      llvm::cl::init("")};

  void runOnOperation() override {
    // Selects the Taskflow function. Besides the candidate file, a successful
    // run attaches each canonical task-body hash to its source task so derived
    // artifacts can bind to the same body.
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selectedFunction =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selectedFunction)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp func = *selectedFunction;
    // Bind both factored and row manifests only after their atomic publication.
    // Failed enumeration must not leave an IR label paired with missing data.
    auto publishSemanticGraphLabel = [&]() {
      func->removeAttr("amoeba.source_task_body_sha256");
      func->removeAttr("amoeba.graph_variant_id");
      func->removeAttr("amoeba.graph_variant_sha256");
      if (!graphVariantId.getValue().empty())
        func->setAttr("amoeba.graph_variant_id",
                      StringAttr::get(func.getContext(), graphVariantId.getValue()));
    };
    if (outputFile.getValue().empty() || maxCandidates.getValue() <= 0 ||
        maxCgrasPerTask.getValue() < 0) {
      func.emitError() << "output, positive max-candidates, and nonnegative "
                          "max-cgras-per-task are required";
      return signalPassFailure();
    }
    std::string shapePruningPolicy = kShapePruningPolicyNone.str();
    if (!searchPolicy.getValue().empty() &&
        !shapePruningPolicyOption.getValue().empty() &&
        searchPolicy.getValue() != shapePruningPolicyOption.getValue()) {
      func.emitError() << "search-policy and shape-pruning-policy disagree";
      return signalPassFailure();
    }
    if (!searchPolicy.getValue().empty())
      shapePruningPolicy = searchPolicy.getValue();
    else if (!shapePruningPolicyOption.getValue().empty())
      shapePruningPolicy = shapePruningPolicyOption.getValue();
    if (shapePruningPolicy == "complete-cartesian")
      shapePruningPolicy = kShapePruningPolicyNone.str();
    if (shapePruningPolicy != kShapePruningPolicyNone) {
      func.emitError()
          << "shape pruning was removed; enumerate the complete Cartesian "
             "space or request --factored-output";
      return signalPassFailure();
    }
    if (maxChangedTasks.getValue() != -1) {
      func.emitError()
          << "max-changed-tasks is removed; complete shape search requires -1";
      return signalPassFailure();
    }
    // File-content fingerprints are not part of the candidate identity
    // contract. Reject a legacy option instead of silently dropping it.
    if (!graphVariantSha256.getValue().empty()) {
      func.emitError() << "graph-variant-sha256 is unsupported by the no-SHA "
                          "candidate contract";
      return signalPassFailure();
    }

    // Collects task metadata and builds the single-task shape alphabet from the
    // architecture values read by Neura's YAML loader. Dynamic or unresolved
    // trip counts are rejected by the static candidate-space contract.
    FailureOr<SmallVector<TaskMetadata>> taskMetadata =
        collectAnalyticalTaskMetadata(func, error);
    if (failed(taskMetadata)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    const ::mlir::neura::Architecture &architecture =
        ::mlir::neura::getArchitecture();
    const int64_t gridRows = architecture.getMultiCgraRows();
    const int64_t gridCols = architecture.getMultiCgraColumns();
    if (gridRows <= 0 || gridCols <= 0 ||
        gridRows > std::numeric_limits<int64_t>::max() / gridCols) {
      func.emitError() << "physical CGRA grid dimensions are invalid";
      return signalPassFailure();
    }
    const int64_t gridArea = gridRows * gridCols;
    const int64_t effectiveMaxCgrasPerTask =
        maxCgrasPerTask.getValue() == 0
            ? gridArea
            : std::min(maxCgrasPerTask.getValue(), gridArea);
    SmallVector<RectShape> shapes = enumerateStaticRectShapes(
        gridRows, gridCols, architecture.getPerCgraRows(),
        architecture.getPerCgraColumns(), effectiveMaxCgrasPerTask);
    if (shapes.empty()) {
      func.emitError() << "declared rectangular shape space is empty";
      return signalPassFailure();
    }
    SmallVector<SmallVector<RectShape>> shapesByTask(taskMetadata->size(),
                                                     shapes);
    // Count the exact unrestricted family before publishing anything. Every
    // individual rectangle fits the physical grid. Whole-program occupancy,
    // location, and temporal feasibility must be established by the exact
    // production scheduler before a candidate is admitted.
    ShapeCandidateSpaceCounts spaceCounts = countShapeCandidateSpace(
        shapesByTask, shapePruningPolicy, maxChangedTasks.getValue());
    if (spaceCounts.declared.empty()) {
      func.emitError() << "cannot count the declared task-shape space";
      return signalPassFailure();
    }
    uint64_t candidateCount = 0;
    const bool countFitsUint64 =
        !StringRef(spaceCounts.declared).getAsInteger(10, candidateCount);
    const bool exceedsRowBudget =
        !countFitsUint64 || candidateCount == 0 ||
        candidateCount > static_cast<uint64_t>(maxCandidates.getValue());
    if (candidateCount == 0 && countFitsUint64) {
      func.emitError() << "declared task-shape Cartesian product is empty";
      return signalPassFailure();
    }
    if (factoredOutput.getValue()) {
      // The factored descriptor is deliberately a distinct schema. Existing
      // row-oriented consumers must reject it until they implement direct
      // exact ranking; no partial candidate manifest is ever published.
      auto addDecimal = [](llvm::json::Object &object, StringRef key,
                           StringRef decimal) {
        uint64_t value = 0;
        if (!decimal.getAsInteger(10, value) &&
            value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
          object[key] = static_cast<int64_t>(value);
        else
          object[key] = decimal.str();
      };
      const std::string function = func.getSymName().str();
      bool wroteFactored = writeAtomically(
          outputFile.getValue(), [&](llvm::raw_ostream &os) {
            llvm::json::Object header;
            header["record_type"] = "space";
            header["schema"] = kFactoredSpaceSchema.str();
            header["representation"] = "factored";
            header["exact"] = true;
            header["function"] = function;
            if (!graphVariantId.getValue().empty())
              header["graph_variant_id"] = graphVariantId.getValue();
            header["search_scope"] = kFactoredSearchScope.str();
            header["shape_policy"] = kShapePolicy.str();
            header["spatial_capacity_policy"] = kSpatialCapacityPolicy.str();
            header["shape_pruning_policy"] = kShapePruningPolicyNone.str();
            header["max_changed_tasks"] = -1;
            header["max_cgras_per_task"] = effectiveMaxCgrasPerTask;
            addDecimal(header, "candidate_count", spaceCounts.declared);
            llvm::json::Array factors;
            for (auto [taskIndex, task] : llvm::enumerate(*taskMetadata)) {
              llvm::json::Object factor;
              factor["task"] = task.name;
              factor["trip_count"] = task.tripCount;
              llvm::json::Array factorShapes;
              for (const RectShape &shape : shapesByTask[taskIndex]) {
                llvm::json::Object shapeRecord;
                shapeRecord["kind"] = "rect";
                shapeRecord["rows"] = shape.rows;
                shapeRecord["cols"] = shape.cols;
                shapeRecord["cgra_count"] = shape.cgraCount();
                shapeRecord["cgra_shape"] = shape.toCgraShapeAttrValue();
                shapeRecord["mapper_tile_rows"] = shape.mapperRows;
                shapeRecord["mapper_tile_cols"] = shape.mapperCols;
                factorShapes.push_back(std::move(shapeRecord));
              }
              factor["shapes"] = std::move(factorShapes);
              factors.push_back(std::move(factor));
            }
            header["factors"] = std::move(factors);
            writeJsonLine(os, std::move(header));
            llvm::json::Object footer;
            footer["record_type"] = "footer";
            footer["schema"] = kFactoredSpaceSchema.str();
            footer["representation"] = "factored";
            footer["exact"] = true;
            addDecimal(footer, "candidate_count", spaceCounts.declared);
            footer["ranked"] = false;
            footer["status"] = "unranked-factored-space";
            writeJsonLine(os, std::move(footer));
            return true;
          },
          error);
      if (!wroteFactored) {
        func.emitError() << error;
        return signalPassFailure();
      }
      publishSemanticGraphLabel();
      llvm::errs() << "[JointScheduling] wrote exact factored task-shape "
                      "space (" << spaceCounts.declared
                   << " candidates) without candidate rows; direct ranking is "
                      "required before ablation" << "\n";
      return;
    }
    if (exceedsRowBudget) {
      func.emitError() << "complete task-shape Cartesian product exceeds "
                          "max-candidates=" << maxCandidates.getValue()
                       << "; refusing to publish a partial candidate manifest; "
                          "use --factored-output for an exact space descriptor";
      return signalPassFailure();
    }
    SmallVector<SmallVector<uint8_t>> usedCostQueries(taskMetadata->size());
    for (auto [taskIndex, used] : llvm::enumerate(usedCostQueries)) {
      used.assign(shapesByTask[taskIndex].size(), 0);
      for (auto [shapeIndex, shape] :
           llvm::enumerate(shapesByTask[taskIndex])) {
        (void)shape;
        // Complete search requires the predictor query for every legal task /
        // oriented-shape pair. No changed-task pruning is permitted.
        used[shapeIndex] = 1;
      }
    }

    const std::string function = func.getSymName().str();
    bool wrote = writeAtomically(
        outputFile.getValue(),
        [&](llvm::raw_ostream &os) {
          // Freezes the dimensions and semantic task names needed to
          // reconstruct the candidate space. Git commit and artifact-level
          // provenance are recorded by the caller; this manifest carries no
          // file-content identity. The cost-query list contains exactly every
          // task/shape pair referenced by the Cartesian product.
          llvm::json::Object architectureRecord;
          architectureRecord["grid_rows"] =
              int64_t{architecture.getMultiCgraRows()};
          architectureRecord["grid_cols"] =
              int64_t{architecture.getMultiCgraColumns()};
          architectureRecord["per_cgra_tile_rows"] =
              int64_t{architecture.getPerCgraRows()};
          architectureRecord["per_cgra_tile_cols"] =
              int64_t{architecture.getPerCgraColumns()};
          llvm::json::Array tasks;
          for (const TaskMetadata &task : *taskMetadata) {
            llvm::json::Object record;
            record["task"] = task.name;
            record["trip_count"] = task.tripCount;
            tasks.push_back(std::move(record));
          }
          llvm::json::Array costQueries;
          for (auto [taskIndex, task] : llvm::enumerate(*taskMetadata)) {
            for (auto [shapeIndex, shape] :
                 llvm::enumerate(shapesByTask[taskIndex])) {
              if (!usedCostQueries[taskIndex][shapeIndex])
                continue;
              llvm::json::Object query;
              query["task"] = task.name;
              query["mapper_tile_rows"] = shape.mapperRows;
              query["mapper_tile_cols"] = shape.mapperCols;
              costQueries.push_back(std::move(query));
            }
          }
          // Records the axes held constant by this static shape-selection
          // contract.
          llvm::json::Object fixedAxes;
          fixedAxes["fusion"] = "identity";
          fixedAxes["fission"] = "factor-1";
          fixedAxes["tiling"] = "factor-1";
          fixedAxes["placement"] = "not-enumerated-by-shape-space";
          fixedAxes["temporal_order"] = "not-enumerated-by-shape-space";
          fixedAxes["communication"] = "not-enumerated-by-shape-space";
          llvm::json::Object budget;
          budget["policy"] = kSpatialCandidateBudgetPolicy.str();
          budget["max_candidates"] = maxCandidates.getValue();
          budget["complete"] = true;
          llvm::json::Object header;
          header["record_type"] = "header";
          header["schema"] = kCandidateSchema.str();
          header["search_scope"] = kSearchScope.str();
          header["shape_policy"] = kShapePolicy.str();
          header["spatial_capacity_policy"] = kSpatialCapacityPolicy.str();
          header["shape_pruning_policy"] = shapePruningPolicy;
          header["max_changed_tasks"] = maxChangedTasks.getValue();
          auto addCount = [](llvm::json::Object &object, StringRef key,
                             StringRef decimal) {
            uint64_t value = 0;
            if (!decimal.getAsInteger(10, value) &&
                value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
              object[key] = static_cast<int64_t>(value);
            else
              object[key] = decimal.str();
          };
          addCount(header, "unrestricted_candidate_count",
                   spaceCounts.unrestricted);
          addCount(header, "declared_candidate_count", spaceCounts.declared);
          addCount(header, "excluded_candidate_count", spaceCounts.excluded);
          addCount(header, "unmeasured_candidate_count", spaceCounts.excluded);
          header["function"] = function;
          if (!graphVariantId.getValue().empty())
            header["graph_variant_id"] = graphVariantId.getValue();
          header["architecture"] = std::move(architectureRecord);
          header["max_cgras_per_task"] = effectiveMaxCgrasPerTask;
          header["candidate_budget"] = std::move(budget);
          header["tasks"] = std::move(tasks);
          header["cost_queries"] = std::move(costQueries);
          header["fixed_axes"] = std::move(fixedAxes);
          writeJsonLine(os, std::move(header));

          // Emits the complete task-major Cartesian product. This row mode is
          // intentionally bounded by max-candidates; use factored-output for
          // larger exact spaces instead of publishing a partial prefix.
          uint64_t emitted = 0;
          bool emittedAll = visitShapeCandidateSpace(
              shapesByTask, shapePruningPolicy, maxChangedTasks.getValue(),
              [&](uint64_t index, ArrayRef<size_t> shapeIndices) {
                Candidate candidate;
                candidate.id = makeSequentialCandidateId(index);
                for (auto [taskIndex, shapeIndex] :
                     llvm::enumerate(shapeIndices)) {
                  const TaskMetadata &task = (*taskMetadata)[taskIndex];
                  candidate.choices.push_back(
                      {task.name, task.tripCount,
                       shapesByTask[taskIndex][shapeIndex]});
                }
                writeJsonLine(os, candidateJson(candidate));
                emitted = index + 1;
                return true;
              });
          if (!emittedAll || emitted != candidateCount) {
            error = "internal candidate-count mismatch";
            return false;
          }

          // Closes the stream with its record count. The common reader
          // independently recomputes the exact Cartesian space and every ID.
          llvm::json::Object footer;
          footer["record_type"] = "footer";
          footer["schema"] = kCandidateSchema.str();
          footer["candidate_count"] = static_cast<int64_t>(emitted);
          footer["shape_pruning_policy"] = shapePruningPolicy;
          auto footerAddCount = [](llvm::json::Object &object, StringRef key,
                                   StringRef decimal) {
            uint64_t value = 0;
            if (!decimal.getAsInteger(10, value) &&
                value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
              object[key] = static_cast<int64_t>(value);
            else
              object[key] = decimal.str();
          };
          llvm::json::Object spaceSummary;
          spaceSummary["max_changed_tasks"] = maxChangedTasks.getValue();
          footerAddCount(spaceSummary, "unrestricted_candidate_count",
                         spaceCounts.unrestricted);
          footerAddCount(spaceSummary, "declared_candidate_count",
                         spaceCounts.declared);
          footerAddCount(spaceSummary, "excluded_candidate_count",
                         spaceCounts.excluded);
          footerAddCount(spaceSummary, "unmeasured_candidate_count",
                         spaceCounts.excluded);
          footer["space_summary"] = std::move(spaceSummary);
          writeJsonLine(os, std::move(footer));
          return true;
        },
        error);
    if (!wrote) {
      func.emitError() << error;
      return signalPassFailure();
    }

    publishSemanticGraphLabel();
    llvm::errs() << "[JointScheduling] enumerated all " << candidateCount
                 << " declared task-shape candidates (policy "
                 << shapePruningPolicy << ") into " << outputFile.getValue()
                 << "\n";
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createEnumerateAnalyticalTaskCandidatesPass() {
  return std::make_unique<EnumerateAnalyticalTaskCandidatesPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
