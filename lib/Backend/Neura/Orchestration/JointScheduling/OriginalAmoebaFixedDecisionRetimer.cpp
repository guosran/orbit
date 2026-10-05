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

bool validateCapturedReplicaCells(
    const OriginalAmoebaFixedPlacement &placement, int gridRows, int gridCols,
    bool allowOriginalF45ReplicaScaling,
    std::vector<FixedScheduleOccupiedCell> &sortedCells,
    std::string &rejection) {
  const unsigned replicas = placement.activeReplicas;
  if (replicas < 1 || replicas > 4) {
    rejection = "captured active_replicas must be in the range 1..4";
    return false;
  }
  if (replicas != 1 && !allowOriginalF45ReplicaScaling) {
    rejection = "captured active_replicas requires the explicit original F45 "
                "replica scaling estimate mode";
    return false;
  }

  const size_t cellsPerReplica =
      static_cast<size_t>(placement.selectedRows * placement.selectedCols);
  if (placement.occupiedCells.size() !=
      cellsPerReplica * static_cast<size_t>(replicas)) {
    rejection = "captured occupancy does not contain exactly the selected "
                "CGRA area for every replica";
    return false;
  }

  std::vector<std::set<FixedScheduleOccupiedCell>> replicaCells(replicas);
  std::set<FixedScheduleOccupiedCell> allCells;
  for (const OriginalAmoebaOccupiedCell &cell : placement.occupiedCells) {
    const FixedScheduleOccupiedCell coordinate{cell.row, cell.col};
    if (cell.replicaId >= replicas || cell.row < 0 || cell.row >= gridRows ||
        cell.col < 0 || cell.col >= gridCols) {
      rejection = "captured occupancy has an out-of-range replica ID or a "
                  "CGRA cell outside the grid";
      return false;
    }
    if (!replicaCells[cell.replicaId].insert(coordinate).second ||
        !allCells.insert(coordinate).second) {
      rejection = "captured replica occupancy duplicates or overlaps a CGRA "
                  "cell";
      return false;
    }
  }

  for (unsigned replica = 0; replica < replicas; ++replica) {
    const auto &cells = replicaCells[replica];
    if (cells.size() != cellsPerReplica) {
      rejection = "captured occupancy omits cells for a replica ID";
      return false;
    }
    int minRow = gridRows, maxRow = -1, minCol = gridCols, maxCol = -1;
    for (const FixedScheduleOccupiedCell &cell : cells) {
      minRow = std::min(minRow, cell.first);
      maxRow = std::max(maxRow, cell.first);
      minCol = std::min(minCol, cell.second);
      maxCol = std::max(maxCol, cell.second);
    }
    const int rows = maxRow - minRow + 1;
    const int cols = maxCol - minCol + 1;
    if (static_cast<size_t>(rows * cols) != cellsPerReplica ||
        !((rows == placement.selectedRows && cols == placement.selectedCols) ||
          (rows == placement.selectedCols && cols == placement.selectedRows))) {
      rejection = "captured replica cells do not form the selected shape or "
                  "its rotation";
      return false;
    }
    if (replica == 0 && (minRow != placement.row || minCol != placement.col ||
                         rows != placement.rows || cols != placement.cols)) {
      rejection = "replica 0 captured geometry disagrees with the primary "
                  "placement";
      return false;
    }
  }

  sortedCells.assign(allCells.begin(), allCells.end());
  return true;
}

bool shareCapturedCell(const std::vector<FixedScheduleOccupiedCell> &lhs,
                       const std::vector<FixedScheduleOccupiedCell> &rhs) {
  for (const FixedScheduleOccupiedCell &cell : lhs)
    if (std::binary_search(rhs.begin(), rhs.end(), cell))
      return true;
  return false;
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
    OriginalAmoebaFixedDecisionResult &result,
    bool allowOriginalF45ReplicaScaling) {
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
  FixedScheduleOccupiedCellInventory cellsByTask(tasks.size());
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
    if (!isValidRectangle(placement))
      return reject("selected and placed CGRA rectangles are invalid or have "
                    "different area");
    if (placement.row < 0 || placement.row >= gridRows || placement.col < 0 ||
        placement.col >= gridCols ||
        placement.rows > gridRows - placement.row ||
        placement.cols > gridCols - placement.col)
      return reject("original placement rectangle is outside the 4x4 grid");

    std::string occupancyRejection;
    if (!validateCapturedReplicaCells(
            placement, gridRows, gridCols, allowOriginalF45ReplicaScaling,
            cellsByTask[placement.task], occupancyRejection))
      return reject(std::move(occupancyRejection));
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
  candidate.usesOriginalF45ReplicaScalingEstimate =
      allowOriginalF45ReplicaScaling;
  candidate.dispatchOrder = originalDispatchOrder;
  candidate.preservedDecisions = originalPlacements;
  candidate.tasks.reserve(tasks.size());
  for (unsigned task = 0; task < tasks.size(); ++task) {
    const OriginalAmoebaFixedPlacement &placement = *placementByTask[task];
    ExactScheduleTask exactTask;
    exactTask.name = tasks[task].name;
    exactTask.rows = placement.rows;
    exactTask.cols = placement.cols;
    const unsigned activeReplicas = placement.activeReplicas;
    exactTask.duration = durations[task];
    if (allowOriginalF45ReplicaScaling && activeReplicas > 1)
      exactTask.duration = durations[task] / activeReplicas +
                           (durations[task] % activeReplicas != 0 ? 1 : 0);
    exactTask.predecessors = tasks[task].predecessors;
    candidate.tasks.push_back(std::move(exactTask));
  }
  candidate.placements.resize(tasks.size());
  std::vector<bool> scheduled(tasks.size(), false);

  for (unsigned task : originalDispatchOrder) {
    const OriginalAmoebaFixedPlacement &destination = *placementByTask[task];
    const int64_t duration = candidate.tasks[task].duration;
    int64_t dependencyReady = 0;
    std::string error;
    communication.beginTrial();

    for (unsigned predecessor : modelPredecessorOrder[task]) {
      const ExactSchedulePlacement &sourceSchedule =
          candidate.placements[predecessor];
      int64_t bestReady = std::numeric_limits<int64_t>::max();
      int bestSourceRow = -1;
      int bestSourceCol = -1;
      int bestDestinationRow = -1;
      int bestDestinationCol = -1;
      for (const FixedScheduleOccupiedCell &sourceCell :
           cellsByTask[predecessor]) {
        for (const FixedScheduleOccupiedCell &destinationCell :
             cellsByTask[task]) {
          const int sourceRow = sourceCell.first;
          const int sourceCol = sourceCell.second;
          const int destinationRow = destinationCell.first;
          const int destinationCol = destinationCell.second;
          int64_t endpointReady = 0;
          error.clear();
          if (!communication.getTransferReadyCycle(
                  predecessor, task, sourceRow, sourceCol, destinationRow,
                  destinationCol, sourceSchedule.end, false, endpointReady,
                  error))
            return reject(error.empty() ? "communication endpoint query failed"
                                        : std::move(error),
                          true);
          if (endpointReady < sourceSchedule.end)
            return reject("communication ready cycle precedes producer finish",
                          true);
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
      }
      if (bestSourceRow < 0)
        return reject("communication transfer has no occupied-cell endpoint",
                      true);

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
      if (!checkedEnd(start, duration, end))
        return reject("fixed-decision replay time overflows int64", true);
      bool advanced = false;
      for (unsigned prior : originalDispatchOrder) {
        if (!scheduled[prior] ||
            !shareCapturedCell(cellsByTask[task], cellsByTask[prior]))
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
      candidate.dispatchOrder, communication, &cellsByTask);
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
