//===- OriginalAmoebaFixedDecisionRetimer.h ---------------------*- C++ -*-===//
#ifndef AMOEBA_ORIGINAL_AMOEBA_FIXED_DECISION_RETIMER_H
#define AMOEBA_ORIGINAL_AMOEBA_FIXED_DECISION_RETIMER_H

#include "FixedScheduleVerifier.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {

struct OriginalAmoebaRetimerTask {
  std::string name;
  std::vector<unsigned> predecessors;
};

struct OriginalAmoebaOccupiedCell {
  int row = 0;
  int col = 0;
  unsigned replicaId = 0;
};

// Geometry read from the original TaskScheduler trace. `selectedRows` and
// `selectedCols` retain the throughput-guided profile orientation; `rows`,
// `cols`, `row` and `col` describe replica 0's exact orientation and origin.
// `occupiedCells` preserves every captured (row, column, replica ID) tuple in
// source order; the retimer validates each replica's rectangle independently.
struct OriginalAmoebaFixedPlacement {
  unsigned task = 0;
  int row = 0;
  int col = 0;
  int rows = 1;
  int cols = 1;
  int selectedRows = 1;
  int selectedCols = 1;
  unsigned activeReplicas = 1;
  std::vector<OriginalAmoebaOccupiedCell> occupiedCells;
};

struct OriginalAmoebaFixedDecisionResult {
  bool valid = false;
  // True only when the caller explicitly selected the legacy F45 replica
  // duration estimate. This records the estimate mode, not child measurements.
  bool usesOriginalF45ReplicaScalingEstimate = false;
  std::string rejection;
  std::vector<ExactScheduleTask> tasks;
  std::vector<ExactSchedulePlacement> placements;
  std::vector<unsigned> dispatchOrder;
  std::vector<OriginalAmoebaFixedPlacement> preservedDecisions;
  // This is the explicit-communication replay of mapped durations. Original
  // TaskScheduler internal slots and pipeline_interval are separate metadata.
  FixedScheduleScore replay;
};

// Re-time captured AMOEBA decisions without choosing a new shape, origin or
// dispatch order. Durations are indexed by task and must be mapped latency
// cycles. With explicit F45 scaling, inputs are full-parent mapped cycles and
// the helper applies ceil(duration / activeReplicas) once. Otherwise inputs
// already describe each mapped task. Scaled internal timestamps are not an
// input. `occupiedCells` and `activeReplicas` must come from the actual old
// scheduler trace so unsupported replica layouts fail closed.
bool retimeOriginalAmoebaFixedDecisions(
    int gridRows, int gridCols,
    const std::vector<OriginalAmoebaRetimerTask> &tasks,
    const std::vector<int64_t> &durations,
    const std::vector<OriginalAmoebaFixedPlacement> &originalPlacements,
    const std::vector<unsigned> &originalDispatchOrder,
    FixedScheduleCommunication &communication,
    OriginalAmoebaFixedDecisionResult &result,
    bool allowOriginalF45ReplicaScaling = false);

} // namespace mlir::amoeba::neura::joint_scheduling
#endif // AMOEBA_ORIGINAL_AMOEBA_FIXED_DECISION_RETIMER_H
