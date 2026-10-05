//===- FixedScheduleVerifier.cpp - Fixed schedule resource/edge replay -----===//
#include "FixedScheduleVerifier.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {
namespace {
FixedScheduleScore reject(const char *reason) {
  FixedScheduleScore result;
  result.rejection = reason;
  return result;
}
FixedScheduleScore reject(std::string reason) {
  FixedScheduleScore result;
  result.rejection = std::move(reason);
  return result;
}
bool overlaps(int64_t firstStart, int64_t firstEnd, int64_t secondStart,
              int64_t secondEnd) {
  return firstStart < secondEnd && secondStart < firstEnd;
}
} // namespace

bool certifyExplicitCommunicationCoverage(
    const FixedScheduleCoverageEvidence &evidence, std::string &rejection) {
  rejection.clear();
  if (evidence.communicationMode != "explicit") {
    rejection = "inventory does not declare explicit communication";
    return false;
  }
  if (!evidence.enumerationComplete) {
    rejection = "communication-aware schedule inventory is incomplete";
    return false;
  }
  if (!evidence.dispatchOrderRecorded) {
    rejection = "schedule inventory omits recorded dispatch order";
    return false;
  }
  if (!evidence.communicationStartEventsProven) {
    rejection = "communication-ready start event coverage is unproven";
    return false;
  }
  return true;
}

FixedScheduleScore verifyAndScoreFixedSchedule(
    int gridRows, int gridCols, const std::vector<ExactScheduleTask> &tasks,
    const std::vector<ExactSchedulePlacement> &placements,
    const std::vector<unsigned> &taskOrder,
    FixedScheduleCommunication &communication,
    const FixedScheduleOccupiedCellInventory *occupiedCells) {
  if (gridRows != 4 || gridCols != 4 || tasks.empty() ||
      placements.size() != tasks.size() || taskOrder.size() != tasks.size())
    return reject("fixed schedule requires a complete 4x4 task inventory");
  if (occupiedCells && occupiedCells->size() != tasks.size())
    return reject("captured occupied-cell inventory omits or adds a task");
  std::vector<const ExactSchedulePlacement *> byTask(tasks.size(), nullptr);
  std::set<std::string> names;
  for (unsigned i = 0; i < tasks.size(); ++i) {
    const auto &task = tasks[i];
    if (task.name.empty() || !names.insert(task.name).second || task.rows < 1 ||
        task.cols < 1 || task.rows > 4 || task.cols > 4 ||
        task.rows * task.cols > 4 || task.duration < 1)
      return reject("invalid fixed task shape, name or predicted duration");
    std::set<unsigned> predecessors;
    for (unsigned predecessor : task.predecessors)
      if (predecessor >= tasks.size() || predecessor == i ||
          !predecessors.insert(predecessor).second)
        return reject("invalid or duplicate task predecessor");
  }
  for (const auto &placement : placements) {
    if (placement.task >= tasks.size() || byTask[placement.task])
      return reject("duplicate or unknown schedule task");
    const auto &task = tasks[placement.task];
    if (placement.row < 0 || placement.col < 0 ||
        placement.row + task.rows > gridRows ||
        placement.col + task.cols > gridCols || placement.start < 0 ||
        placement.start > std::numeric_limits<int64_t>::max() - task.duration ||
        placement.end != placement.start + task.duration)
      return reject(
          "schedule rectangle, interval or predicted duration is invalid");
    byTask[placement.task] = &placement;
  }
  for (const auto *placement : byTask)
    if (!placement)
      return reject("schedule omits a task");

  FixedScheduleOccupiedCellInventory cellInventory(tasks.size());
  std::vector<std::set<FixedScheduleOccupiedCell>> cellSets(tasks.size());
  for (unsigned task = 0; task < tasks.size(); ++task) {
    const ExactSchedulePlacement &placement = *byTask[task];
    const ExactScheduleTask &shape = tasks[task];
    auto &cells = cellInventory[task];
    if (occupiedCells) {
      cells = (*occupiedCells)[task];
      if (cells.empty())
        return reject("captured task occupancy contains no CGRA cells");
      for (const FixedScheduleOccupiedCell &cell : cells)
        if (cell.first < 0 || cell.first >= gridRows || cell.second < 0 ||
            cell.second >= gridCols || !cellSets[task].insert(cell).second)
          return reject("captured task occupancy has an out-of-bounds or "
                        "duplicate CGRA cell");
      for (int row = placement.row; row < placement.row + shape.rows; ++row)
        for (int col = placement.col; col < placement.col + shape.cols; ++col)
          if (!cellSets[task].count({row, col}))
            return reject("captured task occupancy omits a primary-rectangle "
                          "CGRA cell");
      std::sort(cells.begin(), cells.end());
    } else {
      for (int row = placement.row; row < placement.row + shape.rows; ++row)
        for (int col = placement.col; col < placement.col + shape.cols; ++col) {
          cells.emplace_back(row, col);
          cellSets[task].emplace(row, col);
        }
    }
  }

  for (unsigned i = 0; i < placements.size(); ++i)
    for (unsigned j = i + 1; j < placements.size(); ++j) {
      const auto &a = placements[i], &b = placements[j];
      if (!overlaps(a.start, a.end, b.start, b.end))
        continue;
      std::vector<FixedScheduleOccupiedCell> commonCells;
      std::set_intersection(
          cellInventory[a.task].begin(), cellInventory[a.task].end(),
          cellInventory[b.task].begin(), cellInventory[b.task].end(),
          std::back_inserter(commonCells));
      if (!commonCells.empty())
        return reject("simultaneous tasks overlap on captured CGRA cells");
    }
  std::vector<bool> visited(tasks.size(), false);
  for (unsigned index : taskOrder) {
    if (index >= tasks.size() || visited[index])
      return reject("task order duplicates or references an unknown task");
    for (unsigned predecessor : tasks[index].predecessors)
      if (!visited[predecessor] ||
          byTask[predecessor]->end > byTask[index]->start)
        return reject("task order or start violates a dependency");
    visited[index] = true;
  }
  if (std::find(visited.begin(), visited.end(), false) != visited.end())
    return reject("task order omits a task");

  FixedScheduleScore result;
  communication.resetReservations();
  for (unsigned index : taskOrder) {
    std::vector<unsigned> modelPredecessors;
    std::string error;
    if (!communication.getPredecessors(index, modelPredecessors, error))
      return reject(error.empty() ? "communication predecessor query failed"
                                  : std::move(error));
    std::set<unsigned> declared(tasks[index].predecessors.begin(),
                                tasks[index].predecessors.end());
    std::set<unsigned> observed(modelPredecessors.begin(),
                                modelPredecessors.end());
    if (declared != observed || observed.size() != modelPredecessors.size())
      return reject("schedule dependencies disagree with communication graph");

    int64_t ready = 0;
    communication.beginTrial();
    for (unsigned predecessor : modelPredecessors) {
      const auto &source = *byTask[predecessor];
      int64_t bestReady = std::numeric_limits<int64_t>::max();
      int bestSourceRow = -1, bestSourceCol = -1;
      int bestDestinationRow = -1, bestDestinationCol = -1;
      for (const FixedScheduleOccupiedCell &sourceCell :
           cellInventory[predecessor]) {
        for (const FixedScheduleOccupiedCell &destinationCell :
             cellInventory[index]) {
          const int sourceRow = sourceCell.first;
          const int sourceCol = sourceCell.second;
          const int destinationRow = destinationCell.first;
          const int destinationCol = destinationCell.second;
          int64_t endpointReady = 0;
          if (!communication.getTransferReadyCycle(
                  predecessor, index, sourceRow, sourceCol, destinationRow,
                  destinationCol, source.end, false, endpointReady, error)) {
            communication.finishTrial(false);
            return reject(error.empty() ? "communication transfer query failed"
                                        : std::move(error));
          }
          if (endpointReady < source.end) {
            communication.finishTrial(false);
            return reject(
                "communication model returns a cycle before producer finish");
          }
          if (endpointReady < bestReady) {
            bestReady = endpointReady;
            bestSourceRow = sourceRow;
            bestSourceCol = sourceCol;
            bestDestinationRow = destinationRow;
            bestDestinationCol = destinationCol;
          }
        }
      }
      if (bestSourceRow < 0) {
        communication.finishTrial(false);
        return reject("communication transfer has no occupied-cell endpoint");
      }
      if (!communication.getTransferReadyCycle(
              predecessor, index, bestSourceRow, bestSourceCol,
              bestDestinationRow, bestDestinationCol, source.end, true,
              bestReady, error)) {
        communication.finishTrial(false);
        return reject(error.empty()
                          ? "communication transfer reservation failed"
                          : std::move(error));
      }
      if (bestReady < source.end) {
        communication.finishTrial(false);
        return reject("communication reservation precedes producer finish");
      }
      ready = std::max(ready, bestReady);
      ++result.replayedEdges;
    }
    if (ready > byTask[index]->start) {
      communication.finishTrial(false);
      return reject("task starts before dependency/communication ready cycle");
    }
    communication.finishTrial(true);
    result.predictedMakespan =
        std::max(result.predictedMakespan, byTask[index]->end);
  }
  result.valid = true;
  return result;
}
} // namespace mlir::amoeba::neura::joint_scheduling
