//===- FixedScheduleVerifier.cpp - Fixed schedule resource/edge replay -----===//
#include "FixedScheduleVerifier.h"

#include <algorithm>
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
    FixedScheduleCommunication &communication) {
  if (gridRows != 4 || gridCols != 4 || tasks.empty() ||
      placements.size() != tasks.size() || taskOrder.size() != tasks.size())
    return reject("fixed schedule requires a complete 4x4 task inventory");
  std::vector<const ExactSchedulePlacement *> byTask(tasks.size(), nullptr);
  std::set<std::string> names;
  for (unsigned i = 0; i < tasks.size(); ++i) {
    const auto &task = tasks[i];
    if (task.name.empty() || !names.insert(task.name).second ||
        task.rows < 1 || task.cols < 1 || task.rows > 4 || task.cols > 4 ||
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
      return reject("schedule rectangle, interval or predicted duration is invalid");
    byTask[placement.task] = &placement;
  }
  for (const auto *placement : byTask)
    if (!placement)
      return reject("schedule omits a task");
  for (unsigned i = 0; i < placements.size(); ++i)
    for (unsigned j = i + 1; j < placements.size(); ++j) {
      const auto &a = placements[i], &b = placements[j];
      const auto &aShape = tasks[a.task], &bShape = tasks[b.task];
      if (overlaps(a.start, a.end, b.start, b.end) &&
          a.row < b.row + bShape.rows && b.row < a.row + aShape.rows &&
          a.col < b.col + bShape.cols && b.col < a.col + aShape.cols)
        return reject("simultaneous task rectangles overlap on the 4x4 grid");
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
      const auto &destination = *byTask[index];
      const auto &sourceShape = tasks[predecessor];
      const auto &destinationShape = tasks[index];
      int64_t bestReady = std::numeric_limits<int64_t>::max();
      int bestSourceRow = -1, bestSourceCol = -1;
      int bestDestinationRow = -1, bestDestinationCol = -1;
      for (int sourceRow = source.row;
           sourceRow < source.row + sourceShape.rows; ++sourceRow)
        for (int sourceCol = source.col;
             sourceCol < source.col + sourceShape.cols; ++sourceCol)
          for (int destinationRow = destination.row;
               destinationRow < destination.row + destinationShape.rows;
               ++destinationRow)
            for (int destinationCol = destination.col;
                 destinationCol < destination.col + destinationShape.cols;
                 ++destinationCol) {
              int64_t endpointReady = 0;
              if (!communication.getTransferReadyCycle(
                      predecessor, index, sourceRow, sourceCol,
                      destinationRow, destinationCol, source.end, false,
                      endpointReady, error)) {
                communication.finishTrial(false);
                return reject(error.empty() ? "communication transfer query failed"
                                            : std::move(error));
              }
              if (endpointReady < source.end) {
                communication.finishTrial(false);
                return reject("communication model returns a cycle before producer finish");
              }
              if (endpointReady < bestReady) {
                bestReady = endpointReady;
                bestSourceRow = sourceRow;
                bestSourceCol = sourceCol;
                bestDestinationRow = destinationRow;
                bestDestinationCol = destinationCol;
              }
            }
      if (bestSourceRow < 0) {
        communication.finishTrial(false);
        return reject("communication transfer has no rectangle endpoint");
      }
      if (!communication.getTransferReadyCycle(
              predecessor, index, bestSourceRow, bestSourceCol,
              bestDestinationRow, bestDestinationCol, source.end, true,
              bestReady, error)) {
        communication.finishTrial(false);
        return reject(error.empty() ? "communication transfer reservation failed"
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
