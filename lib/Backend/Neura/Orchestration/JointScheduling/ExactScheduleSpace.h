//===- ExactScheduleSpace.h - Event-normalized fixed-shape schedules ------===//
//
// This independent search core enumerates dependency-ready orders, rectangle
// origins and resource-release start events. It does not rank schedules.
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_EXACT_SCHEDULE_SPACE_H
#define AMOEBA_EXACT_SCHEDULE_SPACE_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {

struct ExactScheduleTask {
  std::string name;
  int rows = 1;
  int cols = 1;
  int64_t duration = 1;
  std::vector<unsigned> predecessors;
};

struct ExactSchedulePlacement {
  unsigned task = 0;
  int row = 0;
  int col = 0;
  int64_t start = 0;
  int64_t end = 0;
  int64_t idleCycles = 0;
};

struct ExactScheduleLimits {
  uint64_t maxExpandedNodes = 100000;
  uint64_t maxMilliseconds = 0; // Zero disables the time limit.
  // Zero requests an inventory. A positive value permits only proven
  // makespan pruning, retaining ties at the incumbent's cutoff.
  uint64_t rankingTopK = 0;
  bool seedIncumbents = false;
  unsigned seedPreferences = 16;
  // Optional exact-state branch memoization.  This is an optimization only:
  // an unavailable canonical communication state disables the memo and never
  // truncates the underlying search.  A zero capacity means unbounded; the
  // production score pass supplies a finite capacity and records insert
  // skips separately.
  bool enableStateMemo = false;
  uint64_t stateMemoCapacity = 0;
  // The external top-K cutoff is safe for completed-subtree reuse only when
  // the caller proves it is monotonic non-increasing.  Generic callbacks do
  // not get this proof by default.
  bool externalUpperBoundMonotonic = false;
};

struct ExactScheduleResult {
  bool complete = false;
  std::string reason;
  uint64_t expandedNodes = 0;
  uint64_t acceptedPaths = 0;
  uint64_t uniqueSchedules = 0;
  uint64_t overlapRejectedBranches = 0;
  uint64_t outOfBoundsRejectedBranches = 0;
  uint64_t boundRejectedBranches = 0;
  uint64_t geometryTimeBoundRejectedBranches = 0;
  uint64_t seedSchedules = 0;
  bool stateMemoEnabled = false;
  bool stateMemoCanonicalStateAvailable = false;
  bool stateMemoExternalCutoffMonotonic = false;
  bool stateMemoCutoffViolation = false;
  uint64_t stateMemoEntries = 0;
  uint64_t stateMemoHits = 0;
  uint64_t stateMemoPrunedSubtrees = 0;
  uint64_t stateMemoCapacitySkips = 0;
  // Diagnostics for the fixed-shape optimistic end-vector memo.  These are
  // optimization telemetry only; they do not affect search completeness or
  // candidate ordering.
  bool optimisticEndsMemoEnabled = false;
  uint64_t optimisticEndsMemoComputations = 0;
  uint64_t optimisticEndsMemoHits = 0;
  uint64_t optimisticEndsMemoEntries = 0;
  uint64_t optimisticEndsMemoCapacitySkips = 0;
};

// Every emitted path records its dispatch order and fits the grid at every time point.
// A complete result covers event-normalized fixed-shape schedules only.
// Communication delays, SRAM byte capacity and the shape/graph search are
// separate gates. Returning false means malformed input or counter overflow.
bool enumerateExactScheduleSpace(
    int gridRows, int gridCols, const std::vector<ExactScheduleTask> &tasks,
    ExactScheduleLimits limits,
    const std::function<void(const std::vector<ExactSchedulePlacement> &,
                             const std::vector<unsigned> &)> &emit,
    ExactScheduleResult &result, std::string &error);

} // namespace mlir::amoeba::neura::joint_scheduling
#endif // AMOEBA_EXACT_SCHEDULE_SPACE_H
