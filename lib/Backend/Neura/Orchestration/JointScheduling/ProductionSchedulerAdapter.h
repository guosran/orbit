//===- ProductionSchedulerAdapter.h -------------------------*- C++ -*-===//
//
// Invoke the source-owned production TaskScheduler for one fixed analytical
// shape candidate.  This adapter intentionally has no temporal search or
// placement search of its own: the production orchestration pass is the sole
// authority for dispatch, origins, overlap, communication readiness, and
// temporal reuse.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_PRODUCTION_SCHEDULER_ADAPTER_H
#define AMOEBA_PRODUCTION_SCHEDULER_ADAPTER_H

#include "AnalyticalTaskCandidateCommon.h"
#include "ExactScheduleSpace.h"
#include "SpatialTaskCandidateSpace.h"

#include "Backend/Neura/NeuraBackendPasses.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {

// The result is deliberately expressed in the fixed-schedule vocabulary used
// by the existing score/replay code.  `dispatch` contains indices into the
// input `choices` array.  `replayedEdges` counts committed production network
// transfer records' edge_indices union; it is marked unknown when the
// production pass did not emit its explicit communication trace.
struct ProductionScheduledResult {
  std::vector<ExactSchedulePlacement> placements;
  // Both names are kept at this narrow integration seam: `dispatch` is the
  // production terminology, while existing exact-score callers commonly use
  // `order` for the same task-index vector.
  std::vector<unsigned> dispatch;
  std::vector<unsigned> order;
  int64_t makespan = 0;
  uint64_t replayedEdges = 0;
  bool replayedEdgesKnown = false;
};

namespace detail {

inline bool productionAdapterFail(std::string &error, StringRef message) {
  error = message.str();
  return false;
}

inline void removeProductionCloneAttrs(func::FuncOp function) {
  // Exact schedule/replay metadata belongs to the ORBIT enumerator and must
  // never steer this production invocation.  Remove it only on the clone.
  for (StringRef attr : {
           StringRef("joint_scheduling_exact_dispatch_order"),
           StringRef("joint_scheduling_exact_replay_timing"),
           StringRef("joint_scheduling_predicted_makespan"),
           StringRef("joint_scheduling_actual_makespan"),
           StringRef("joint_scheduling_replay_verified"),
           StringRef("joint_scheduling_mapper_replay_completed"),
           StringRef("joint_scheduling_production_dispatch_order"),
           StringRef("joint_scheduling_production_schedule"),
           StringRef("joint_scheduling_communication_trace"),
           StringRef("joint_scheduling_dependency_trace"),
           StringRef("joint_scheduling_fixed_point_max_iterations")})
    function->removeAttr(attr);

  function.walk([](taskflow::TaskflowTaskOp task) {
    for (StringRef attr : {
             StringRef("amoeba.exact_schedule"),
             StringRef("amoeba.exact_replay_mapped_timing"),
             StringRef("profile_info"),
             StringRef("task_orchestration_info"),
             StringRef("amoeba.joint_shape_orientation_fixed"),
             StringRef("amoeba.selected_cgra_count"),
             StringRef("amoeba.selected_cgra_shape"),
             StringRef("amoeba.mapper_replay_verified")})
      task->removeAttr(attr);
  });
}

inline bool parseProductionRectangle(
    ArrayAttr cells, int64_t expectedRows, int64_t expectedCols,
    std::set<std::pair<int64_t, int64_t>> &parsed,
    int64_t &originRow, int64_t &originCol, std::string &error,
    StringRef what) {
  auto malformed = [&](StringRef suffix) {
    std::string message = what.str();
    message += suffix.str();
    return productionAdapterFail(error, message);
  };
  parsed.clear();
  originRow = std::numeric_limits<int64_t>::max();
  originCol = std::numeric_limits<int64_t>::max();
  int64_t maxRow = -1;
  int64_t maxCol = -1;
  if (!cells || expectedRows <= 0 || expectedCols <= 0 ||
      expectedRows > std::numeric_limits<int64_t>::max() / expectedCols)
    return malformed(" has an invalid rectangle");
  const int64_t area = expectedRows * expectedCols;
  if (static_cast<int64_t>(cells.size()) != area)
    return malformed(" does not contain the selected shape");
  for (Attribute attribute : cells) {
    auto position = dyn_cast<DictionaryAttr>(attribute);
    auto row = position ? position.getAs<IntegerAttr>("row")
                        : IntegerAttr();
    auto col = position ? position.getAs<IntegerAttr>("col")
                        : IntegerAttr();
    if (!row || !col || row.getInt() < 0 || col.getInt() < 0)
      return malformed(" contains a malformed coordinate");
    if (!parsed.insert({row.getInt(), col.getInt()}).second)
      return malformed(" contains a duplicate coordinate");
    originRow = std::min(originRow, row.getInt());
    originCol = std::min(originCol, col.getInt());
    maxRow = std::max(maxRow, row.getInt());
    maxCol = std::max(maxCol, col.getInt());
  }
  if (maxRow - originRow + 1 != expectedRows ||
      maxCol - originCol + 1 != expectedCols)
    return malformed(" changes the selected orientation");
  for (int64_t row = originRow; row <= maxRow; ++row)
    for (int64_t col = originCol; col <= maxCol; ++col)
      if (!parsed.count({row, col}))
        return malformed(" is not a complete rectangle");
  return true;
}

inline bool sameCoordinates(
    const std::set<std::pair<int64_t, int64_t>> &lhs,
    const std::set<std::pair<int64_t, int64_t>> &rhs) {
  return lhs == rhs;
}

} // namespace detail

// Run the production orchestration pass exactly once on a clone carrying one
// selected shape for every task.  The pass chooses origins, dispatch order,
// half-open temporal occupancy, SRAM placement, and explicit communication
// reservations.  A failed production heuristic is returned as an incomplete
// result; this function never substitutes an ORBIT placement certificate.
inline bool scheduleWithAmoebaProductionPass(
    ModuleOp sourceModule, func::FuncOp sourceFunction,
    ArrayRef<TaskShapeChoice> choices, ArrayRef<int64_t> predictedDurations,
    StringRef candidateId, bool fixedDispatch, ProductionScheduledResult &out,
    std::string &error) {
  out = {};
  error.clear();
  if (!sourceModule || !sourceFunction)
    return detail::productionAdapterFail(error,
                                         "production scheduler requires a module and function");
  if (sourceFunction->getParentOfType<ModuleOp>() != sourceModule)
    return detail::productionAdapterFail(
        error, "production scheduler function is outside the supplied module");
  if (candidateId.empty())
    return detail::productionAdapterFail(error,
                                         "production scheduler candidate id is empty");
  if (choices.empty() || choices.size() != predictedDurations.size())
    return detail::productionAdapterFail(
        error, "production scheduler shape and duration arrays disagree");

  SmallVector<taskflow::TaskflowTaskOp> sourceTasks;
  sourceFunction.walk([&](taskflow::TaskflowTaskOp task) {
    sourceTasks.push_back(task);
  });
  if (sourceTasks.size() != choices.size())
    return detail::productionAdapterFail(
        error, "production scheduler candidate task count disagrees with function");

  std::map<std::string, size_t> choiceIndices;
  for (size_t index = 0; index < choices.size(); ++index) {
    const TaskShapeChoice &choice = choices[index];
    if (choice.task.empty() || !choiceIndices.emplace(choice.task, index).second)
      return detail::productionAdapterFail(
          error, "production scheduler candidate has duplicate or empty task names");
    if (choice.shape.rows <= 0 || choice.shape.cols <= 0 ||
        choice.shape.rows > std::numeric_limits<int64_t>::max() /
                              choice.shape.cols)
      return detail::productionAdapterFail(
          error, "production scheduler candidate has an invalid rectangle shape");
    const int64_t cgraCount = choice.shape.rows * choice.shape.cols;
    if (cgraCount <= 0 ||
        cgraCount > std::numeric_limits<int32_t>::max() ||
        predictedDurations[index] <= 0)
      return detail::productionAdapterFail(
          error, "production scheduler candidate has an invalid count or duration");
  }
  for (taskflow::TaskflowTaskOp task : sourceTasks)
    if (!choiceIndices.count(task.getTaskName().str()))
      return detail::productionAdapterFail(
          error, "production scheduler candidate omits a function task");

  // Clone the source module so all symbol and architecture attributes remain
  // available, then run an anchored func pass on only the selected function.
  // This avoids running the orchestration pass on helper functions in the
  // module while preserving the production pass's own implementation.
  OwningOpRef<ModuleOp> clonedModule = sourceModule.clone();
  func::FuncOp clonedFunction;
  clonedModule->walk([&](func::FuncOp function) {
    if (!clonedFunction && function.getName() == sourceFunction.getName())
      clonedFunction = function;
  });
  if (!clonedFunction)
    return detail::productionAdapterFail(error,
                                         "production scheduler could not find cloned function");

  detail::removeProductionCloneAttrs(clonedFunction);
  OpBuilder builder(clonedFunction.getContext());
  // Fixed-orientation production candidates get a larger, still finite,
  // fixed-point budget. A caller may override it on the source function;
  // invalid overrides fail closed before the pass runs. Legacy callers that
  // invoke TaskScheduler directly retain its ten-iteration default.
  int64_t fixedPointIterations = 64;
  if (auto configured = sourceFunction->getAttrOfType<IntegerAttr>(
          "joint_scheduling_fixed_point_max_iterations")) {
    fixedPointIterations = configured.getInt();
    if (fixedPointIterations <= 0 ||
        fixedPointIterations > std::numeric_limits<int>::max())
      return detail::productionAdapterFail(
          error,
          "joint_scheduling_fixed_point_max_iterations is outside the supported range");
  }
  clonedFunction->setAttr(
      "joint_scheduling_fixed_point_max_iterations",
      builder.getI64IntegerAttr(fixedPointIterations));
  clonedFunction->setAttr("joint_scheduling_candidate_id",
                          builder.getStringAttr(candidateId));
  clonedFunction.walk([&](taskflow::TaskflowTaskOp task) {
    auto found = choiceIndices.find(task.getTaskName().str());
    if (found == choiceIndices.end())
      return;
    const size_t index = found->second;
    const TaskShapeChoice &choice = choices[index];
    task->setAttr("cgra_count",
                  builder.getI32IntegerAttr(static_cast<int32_t>(
                      choice.shape.rows * choice.shape.cols)));
    task->setAttr("cgra_shape",
                  builder.getStringAttr(choice.shape.toCgraShapeAttrValue()));
    task->setAttr("amoeba.joint_shape_orientation_fixed",
                  builder.getUnitAttr());
    task->setAttr("est_latency",
                  builder.getI64IntegerAttr(predictedDurations[index]));
  });

  std::unique_ptr<Pass> pass = createOrchestrateTasksOnAcceleratorsPass();
  std::string options =
      "scheduling-mode=spatial-temporal "
      "orchestration-strategy=analytical-based-task-orchestration "
      "communication-mode=explicit dispatch-policy=";
  options += fixedDispatch ? "fixed" : "critical-path";
  if (failed(pass->initializeOptions(options, [&](const llvm::Twine &message) {
        error = message.str();
        return failure();
      }))) {
    if (error.empty())
      error = "production scheduler pass option parsing failed";
    return false;
  }

  PassManager manager = PassManager::on<func::FuncOp>(clonedFunction.getContext());
  manager.enableVerifier(true);
  manager.addPass(std::move(pass));
  std::string diagnostic;
  LogicalResult runResult = failure();
  {
    ScopedDiagnosticHandler capture(clonedFunction.getContext(),
                                    [&](Diagnostic &entry) {
                                      if (diagnostic.empty()) {
                                        llvm::raw_string_ostream stream(diagnostic);
                                        entry.print(stream);
                                        stream.flush();
                                      }
                                      return success();
                                    });
    runResult = manager.run(clonedFunction.getOperation());
  }
  if (failed(runResult)) {
    error = diagnostic.empty() ? "production scheduler pass failed" : diagnostic;
    return false;
  }

  std::map<std::string, size_t> clonedTaskIndices;
  SmallVector<taskflow::TaskflowTaskOp> clonedTasks;
  clonedFunction.walk([&](taskflow::TaskflowTaskOp task) {
    clonedTaskIndices.emplace(task.getTaskName().str(), clonedTasks.size());
    clonedTasks.push_back(task);
  });
  if (clonedTasks.size() != choices.size() ||
      clonedTaskIndices.size() != clonedTasks.size())
    return detail::productionAdapterFail(
        error, "production scheduler output task set is incomplete or ambiguous");

  struct ParsedSchedule {
    ExactSchedulePlacement placement;
    std::set<std::pair<int64_t, int64_t>> cells;
  };
  std::map<std::string, ParsedSchedule> parsed;
  auto scheduleAttr = clonedFunction->getAttrOfType<ArrayAttr>(
      "joint_scheduling_production_schedule");
  if (!scheduleAttr || scheduleAttr.size() != choices.size())
    return detail::productionAdapterFail(
        error, "production scheduler did not emit one timing record per task");

  for (Attribute attribute : scheduleAttr) {
    auto record = dyn_cast<DictionaryAttr>(attribute);
    auto name = record ? record.getAs<StringAttr>("task") : StringAttr();
    auto start = record ? record.getAs<IntegerAttr>("start_cycle")
                        : IntegerAttr();
    auto end = record ? record.getAs<IntegerAttr>("end_cycle")
                      : IntegerAttr();
    auto cells = record ? record.getAs<ArrayAttr>("cgra_positions")
                        : ArrayAttr();
    if (!name || !start || !end || !cells || start.getInt() < 0 ||
        end.getInt() <= start.getInt())
      return detail::productionAdapterFail(
          error, "production scheduler emitted a malformed timing record");
    auto choice = choiceIndices.find(name.getValue().str());
    if (choice == choiceIndices.end() || parsed.count(name.getValue().str()))
      return detail::productionAdapterFail(
          error, "production scheduler timing record references an unknown task");
    const TaskShapeChoice &shapeChoice = choices[choice->second];
    std::set<std::pair<int64_t, int64_t>> scheduleCells;
    int64_t row = 0, col = 0;
    if (!detail::parseProductionRectangle(
            cells, shapeChoice.shape.rows, shapeChoice.shape.cols,
            scheduleCells, row, col, error, "production timing coordinates"))
      return false;
    const int64_t duration = predictedDurations[choice->second];
    if (end.getInt() - start.getInt() != duration)
      return detail::productionAdapterFail(
          error, "production scheduler timing does not preserve predicted duration");
    ParsedSchedule parsedRecord;
    parsedRecord.placement = {static_cast<unsigned>(choice->second),
                              static_cast<int>(row), static_cast<int>(col),
                              start.getInt(), end.getInt(), 0};
    parsedRecord.cells = std::move(scheduleCells);
    parsed.emplace(name.getValue().str(), std::move(parsedRecord));
  }

  // task_orchestration_info/profile_info are the established production
  // outputs.  Validate them against the new timing trace rather than deriving
  // origins or durations from the adapter's candidate data.
  for (taskflow::TaskflowTaskOp task : clonedTasks) {
    const std::string name = task.getTaskName().str();
    auto parsedRecord = parsed.find(name);
    if (parsedRecord == parsed.end())
      return detail::productionAdapterFail(error,
                                           "production scheduler omitted a task record");
    auto info = task->getAttrOfType<DictionaryAttr>("task_orchestration_info");
    auto infoCells = info ? info.getAs<ArrayAttr>("cgra_positions") : ArrayAttr();
    auto profile = task->getAttrOfType<DictionaryAttr>("profile_info");
    auto duration = profile ? profile.getAs<IntegerAttr>("duration")
                            : IntegerAttr();
    if (!info || !infoCells || !profile || !duration ||
        duration.getInt() != predictedDurations[choiceIndices[name]])
      return detail::productionAdapterFail(
          error, "production scheduler task metadata disagrees with selected duration");
    std::set<std::pair<int64_t, int64_t>> metadataCells;
    int64_t metadataRow = 0, metadataCol = 0;
    if (!detail::parseProductionRectangle(
            infoCells, choices[choiceIndices[name]].shape.rows,
            choices[choiceIndices[name]].shape.cols, metadataCells,
            metadataRow, metadataCol, error, "task_orchestration_info coordinates"))
      return false;
    if (!detail::sameCoordinates(metadataCells, parsedRecord->second.cells))
      return detail::productionAdapterFail(
          error, "production scheduler timing and placement coordinates disagree");
  }

  auto dispatchAttr = clonedFunction->getAttrOfType<ArrayAttr>(
      "joint_scheduling_production_dispatch_order");
  if (!dispatchAttr || dispatchAttr.size() != choices.size())
    return detail::productionAdapterFail(
        error, "production scheduler did not emit complete dispatch order");
  std::set<unsigned> seenDispatch;
  out.dispatch.reserve(dispatchAttr.size());
  out.order.reserve(dispatchAttr.size());
  for (Attribute attribute : dispatchAttr) {
    auto name = dyn_cast<StringAttr>(attribute);
    auto choice = name ? choiceIndices.find(name.getValue().str())
                       : choiceIndices.end();
    if (!name || choice == choiceIndices.end() ||
        !seenDispatch.insert(static_cast<unsigned>(choice->second)).second)
      return detail::productionAdapterFail(
          error, "production scheduler dispatch order is malformed");
    out.dispatch.push_back(static_cast<unsigned>(choice->second));
    out.order.push_back(static_cast<unsigned>(choice->second));
  }

  out.placements.resize(choices.size());
  int64_t maximumEnd = 0;
  for (const auto &choice : choiceIndices) {
    out.placements[choice.second] = parsed[choice.first].placement;
    maximumEnd = std::max(maximumEnd, parsed[choice.first].placement.end);
  }
  auto makespan = clonedFunction->getAttrOfType<IntegerAttr>(
      "joint_scheduling_predicted_makespan");
  if (!makespan || makespan.getInt() <= 0 || makespan.getInt() != maximumEnd)
    return detail::productionAdapterFail(
        error, "production scheduler makespan metadata is missing or inconsistent");
  out.makespan = makespan.getInt();

  auto communicationTrace = clonedFunction->getAttrOfType<ArrayAttr>(
      "joint_scheduling_communication_trace");
  auto dependencyTrace = clonedFunction->getAttrOfType<ArrayAttr>(
      "joint_scheduling_dependency_trace");
  if (!communicationTrace || !dependencyTrace)
    return detail::productionAdapterFail(
        error, "production scheduler omitted explicit communication evidence");
  std::set<int64_t> committedEdgeIndices;
  for (Attribute attribute : communicationTrace) {
    auto transfer = dyn_cast<DictionaryAttr>(attribute);
    auto edgeIndices = transfer ? transfer.getAs<ArrayAttr>("edge_indices")
                                : ArrayAttr();
    if (!transfer || !edgeIndices)
      return detail::productionAdapterFail(
          error, "production scheduler communication trace is malformed");
    for (Attribute edge : edgeIndices) {
      auto index = dyn_cast<IntegerAttr>(edge);
      if (!index || index.getInt() < 0)
        return detail::productionAdapterFail(
            error, "production scheduler communication edge index is malformed");
      committedEdgeIndices.insert(index.getInt());
    }
  }
  out.replayedEdges = committedEdgeIndices.size();
  out.replayedEdgesKnown = true;
  return true;
}

} // namespace mlir::amoeba::neura::joint_scheduling

#endif // AMOEBA_PRODUCTION_SCHEDULER_ADAPTER_H
