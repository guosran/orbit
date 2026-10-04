//===- OriginalAmoebaFixedDecisionRetimer.cpp ------------------*- C++ -*-===//
#include "OriginalAmoebaFixedDecisionRetimer.h"

#include <algorithm>
#include <limits>
#include <set>
#include <utility>

namespace mlir::amoeba::neura::joint_scheduling {
namespace {

bool isValidRectangle(const OriginalAmoebaFixedPlacement &placement) {
  if (placement.rows < 1 || placement.cols < 1 || placement.rows > 4 ||
      placement.cols > 4 || placement.rows * placement.cols > 4 ||
      placement.selectedRows < 1 || placement.selectedCols < 1 ||
      placement.selectedRows > 4 || placement.selectedCols > 4 ||
      placement.selectedRows * placement.selectedCols !=
          placement.rows * placement.cols)
    return false;
  return (placement.rows == placement.selectedRows &&
          placement.cols == placement.selectedCols) ||
         (placement.rows == placement.selectedCols &&
          placement.cols == placement.selectedRows);
}

bool overlapsSpatially(const OriginalAmoebaFixedPlacement &lhs,
                       const OriginalAmoebaFixedPlacement &rhs) {
  return lhs.row < rhs.row + rhs.rows && rhs.row < lhs.row + lhs.rows &&
         lhs.col < rhs.col + rhs.cols && rhs.col < lhs.col + lhs.cols;
}

bool checkedEnd(int64_t start, int64_t duration, int64_t &end) {
  if (start < 0 || duration < 1 ||
      start > std::numeric_limits<int64_t>::max() - duration)
    return false;
  end = start + duration;
  return true;
}

} // namespace

bool retimeOriginalAmoebaFixedDecisions(
    int gridRows, int gridCols,
    const std::vector<OriginalAmoebaRetimerTask> &tasks,
    const std::vector<int64_t> &durations,
    const std::vector<OriginalAmoebaFixedPlacement> &originalPlacements,
    const std::vector<unsigned> &originalDispatchOrder,
    FixedScheduleCommunication &communication,
    OriginalAmoebaFixedDecisionResult &result) {
  result = {};
  communication.resetReservations();
  auto reject = [&](std::string message, bool trialActive = false) {
    if (trialActive)
      communication.finishTrial(false);
    communication.resetReservations();
    result = {};
    result.rejection = std::move(message);
    return false;
  };

  if (gridRows != 4 || gridCols != 4 || tasks.empty() ||
      tasks.size() > std::numeric_limits<unsigned>::max() ||
      durations.size() != tasks.size() ||
      originalPlacements.size() != tasks.size() ||
      originalDispatchOrder.size() != tasks.size())
    return reject("original fixed-decision replay requires a complete 4x4 task "
                  "inventory");

  std::vector<const OriginalAmoebaFixedPlacement *> placementByTask(
      tasks.size(), nullptr);
  std::set<std::string> taskNames;
  for (unsigned task = 0; task < tasks.size(); ++task) {
    if (tasks[task].name.empty() ||
        !taskNames.insert(tasks[task].name).second || durations[task] < 1)
      return reject(
          "original fixed-decision task name or mapped duration is invalid");
    std::set<unsigned> declaredPredecessors;
    for (unsigned predecessor : tasks[task].predecessors)
      if (predecessor >= tasks.size() || predecessor == task ||
          !declaredPredecessors.insert(predecessor).second)
        return reject(
            "original fixed-decision task has an invalid predecessor");
  }

  for (const OriginalAmoebaFixedPlacement &placement : originalPlacements) {
    if (placement.task >= tasks.size() || placementByTask[placement.task])
      return reject(
          "original placement duplicates or references an unknown task");
    if (placement.activeReplicas != 1)
      return reject("original placement has unsupported active_replicas; "
                    "replicas are not materialized shards");
    if (!isValidRectangle(placement))
      return reject("selected and placed CGRA rectangles are invalid or have "
                    "different area");
    if (placement.row < 0 || placement.row >= gridRows || placement.col < 0 ||
        placement.col >= gridCols ||
        placement.rows > gridRows - placement.row ||
        placement.cols > gridCols - placement.col)
      return reject("original placement rectangle is outside the 4x4 grid");

    const size_t expectedCells =
        static_cast<size_t>(placement.rows * placement.cols);
    if (placement.occupiedCells.size() != expectedCells)
      return reject(
          "captured active occupancy does not fill the original rectangle");
    std::set<std::pair<int, int>> uniqueCells;
    for (const OriginalAmoebaOccupiedCell &cell : placement.occupiedCells) {
      if (cell.replicaId != 0 || cell.row < placement.row ||
          cell.row >= placement.row + placement.rows ||
          cell.col < placement.col ||
          cell.col >= placement.col + placement.cols ||
          !uniqueCells.emplace(cell.row, cell.col).second)
        return reject(
            "captured occupancy has an unsupported replica ID, "
            "duplicates a cell, or lies outside its original rectangle");
    }
    placementByTask[placement.task] = &placement;
  }
  for (const OriginalAmoebaFixedPlacement *placement : placementByTask)
    if (!placement)
      return reject("original placement inventory omits a task");

  std::vector<std::set<unsigned>> modelPredecessors(tasks.size());
  std::vector<std::vector<unsigned>> modelPredecessorOrder(tasks.size());
  for (unsigned task = 0; task < tasks.size(); ++task) {
    std::vector<unsigned> observed;
    std::string error;
    if (!communication.getPredecessors(task, observed, error))
      return reject(error.empty() ? "communication predecessor query failed"
                                  : std::move(error));
    modelPredecessorOrder[task] = observed;
    for (unsigned predecessor : observed)
      if (predecessor >= tasks.size() || predecessor == task ||
          !modelPredecessors[task].insert(predecessor).second)
        return reject(
            "communication graph has an invalid or duplicate predecessor");
    const std::set<unsigned> declared(tasks[task].predecessors.begin(),
                                      tasks[task].predecessors.end());
    if (declared != modelPredecessors[task] ||
        observed.size() != modelPredecessors[task].size())
      return reject(
          "declared task dependencies disagree with the communication graph");
  }

  std::vector<bool> visited(tasks.size(), false);
  for (unsigned task : originalDispatchOrder) {
    if (task >= tasks.size() || visited[task])
      return reject(
          "original dispatch order duplicates or references an unknown task");
    for (unsigned predecessor : modelPredecessors[task])
      if (!visited[predecessor])
        return reject("original dispatch order is not predecessor-ready");
    visited[task] = true;
  }
  if (std::find(visited.begin(), visited.end(), false) != visited.end())
    return reject("original dispatch order omits a task");

  OriginalAmoebaFixedDecisionResult candidate;
  candidate.dispatchOrder = originalDispatchOrder;
  candidate.preservedDecisions = originalPlacements;
  candidate.tasks.reserve(tasks.size());
  for (unsigned task = 0; task < tasks.size(); ++task) {
    const OriginalAmoebaFixedPlacement &placement = *placementByTask[task];
    ExactScheduleTask exactTask;
    exactTask.name = tasks[task].name;
    exactTask.rows = placement.rows;
    exactTask.cols = placement.cols;
    exactTask.duration = durations[task];
    exactTask.predecessors = tasks[task].predecessors;
    candidate.tasks.push_back(std::move(exactTask));
  }
  candidate.placements.resize(tasks.size());
  std::vector<bool> scheduled(tasks.size(), false);

  for (unsigned task : originalDispatchOrder) {
    const OriginalAmoebaFixedPlacement &destination = *placementByTask[task];
    int64_t dependencyReady = 0;
    std::string error;
    communication.beginTrial();

    for (unsigned predecessor : modelPredecessorOrder[task]) {
      const OriginalAmoebaFixedPlacement &source =
          *placementByTask[predecessor];
      const ExactSchedulePlacement &sourceSchedule =
          candidate.placements[predecessor];
      int64_t bestReady = std::numeric_limits<int64_t>::max();
      int bestSourceRow = -1;
      int bestSourceCol = -1;
      int bestDestinationRow = -1;
      int bestDestinationCol = -1;
      for (int sourceRow = source.row; sourceRow < source.row + source.rows;
           ++sourceRow)
        for (int sourceCol = source.col; sourceCol < source.col + source.cols;
             ++sourceCol)
          for (int destinationRow = destination.row;
               destinationRow < destination.row + destination.rows;
               ++destinationRow)
            for (int destinationCol = destination.col;
                 destinationCol < destination.col + destination.cols;
                 ++destinationCol) {
              int64_t endpointReady = 0;
              error.clear();
              if (!communication.getTransferReadyCycle(
                      predecessor, task, sourceRow, sourceCol, destinationRow,
                      destinationCol, sourceSchedule.end, false, endpointReady,
                      error))
                return reject(error.empty()
                                  ? "communication endpoint query failed"
                                  : std::move(error),
                              true);
              if (endpointReady < sourceSchedule.end)
                return reject(
                    "communication ready cycle precedes producer finish", true);
              // Match verifyAndScoreFixedSchedule: endpoint iteration order is
              // row-major and an equal ready time retains the first endpoint.
              if (endpointReady < bestReady) {
                bestReady = endpointReady;
                bestSourceRow = sourceRow;
                bestSourceCol = sourceCol;
                bestDestinationRow = destinationRow;
                bestDestinationCol = destinationCol;
              }
            }
      if (bestSourceRow < 0)
        return reject("communication transfer has no rectangle endpoint", true);

      int64_t reservedReady = 0;
      error.clear();
      if (!communication.getTransferReadyCycle(
              predecessor, task, bestSourceRow, bestSourceCol,
              bestDestinationRow, bestDestinationCol, sourceSchedule.end, true,
              reservedReady, error))
        return reject(error.empty()
                          ? "communication transfer reservation failed"
                          : std::move(error),
                      true);
      if (reservedReady < sourceSchedule.end)
        return reject(
            "reserved communication transfer precedes producer finish", true);
      dependencyReady = std::max(dependencyReady, reservedReady);
    }

    int64_t start = dependencyReady;
    int64_t end = 0;
    for (;;) {
      if (!checkedEnd(start, durations[task], end))
        return reject("fixed-decision replay time overflows int64", true);
      bool advanced = false;
      for (unsigned prior : originalDispatchOrder) {
        if (!scheduled[prior] ||
            !overlapsSpatially(destination, *placementByTask[prior]))
          continue;
        const ExactSchedulePlacement &priorSchedule =
            candidate.placements[prior];
        if (start < priorSchedule.end && priorSchedule.start < end) {
          // Advance to a resource-release event and retry every prior task.
          start = priorSchedule.end;
          advanced = true;
          break;
        }
      }
      if (!advanced)
        break;
    }

    ExactSchedulePlacement placement;
    placement.task = task;
    placement.row = destination.row;
    placement.col = destination.col;
    placement.start = start;
    placement.end = end;
    placement.idleCycles = start - dependencyReady;
    candidate.placements[task] = placement;
    scheduled[task] = true;
    communication.finishTrial(true);
  }

  candidate.replay = verifyAndScoreFixedSchedule(
      gridRows, gridCols, candidate.tasks, candidate.placements,
      candidate.dispatchOrder, communication);
  if (!candidate.replay.valid)
    return reject(candidate.replay.rejection.empty()
                      ? "fixed-decision replay failed final verification"
                      : candidate.replay.rejection);

  int64_t observedEnd = 0;
  uint64_t expectedEdges = 0;
  for (unsigned task = 0; task < tasks.size(); ++task) {
    observedEnd = std::max(observedEnd, candidate.placements[task].end);
    expectedEdges += tasks[task].predecessors.size();
  }
  if (candidate.replay.predictedMakespan != observedEnd ||
      candidate.replay.replayedEdges != expectedEdges)
    return reject("fixed-decision replay disagrees with verifier summary");

  candidate.valid = true;
  result = std::move(candidate);
  return true;
}

} // namespace mlir::amoeba::neura::joint_scheduling
