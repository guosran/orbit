// Necessary timing conditions for rectangle pairs on a fixed grid.
#ifndef AMOEBA_RECTANGLE_PAIR_TIME_BOUNDS_H
#define AMOEBA_RECTANGLE_PAIR_TIME_BOUNDS_H

#include <algorithm>
#include <cstdint>

namespace mlir::amoeba::neura::joint_scheduling {

// Two individually fitting axis-aligned rectangles can be disjoint exactly
// when they can be stacked along at least one grid axis. Their areas alone
// cannot establish this: 1x4 and 4x1 intersect at every pair of 4x4 origins.
inline bool rectanglePairCanOverlapInTime(int gridRows, int gridCols,
                                         int rowsA, int colsA,
                                         int rowsB, int colsB) {
  return static_cast<int64_t>(rowsA) + rowsB <= gridRows ||
         static_cast<int64_t>(colsA) + colsB <= gridCols;
}

// If rectangles cannot be spatially disjoint, a feasible schedule must have
// A finish before B starts or B finish before A starts. Optimistic releases
// and deadlines make this a necessary condition, not a chosen dispatch order.
inline bool rectanglePairHasSerialWindow(int64_t releaseA, int64_t durationA,
                                        int64_t deadlineA, int64_t releaseB,
                                        int64_t durationB, int64_t deadlineB) {
  using Wide = __int128_t;
  const Wide endA = static_cast<Wide>(releaseA) + durationA;
  const Wide endB = static_cast<Wide>(releaseB) + durationB;
  const Wide endBA = std::max(static_cast<Wide>(releaseB), endA) + durationB;
  const Wide endAB = std::max(static_cast<Wide>(releaseA), endB) + durationA;
  return (endA <= deadlineA && endBA <= deadlineB) ||
         (endB <= deadlineB && endAB <= deadlineA);
}

} // namespace mlir::amoeba::neura::joint_scheduling
#endif
