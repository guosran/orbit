//===- SpatialTaskCandidateSpace.h -----------------------------*- C++ -*-===//
//
// Defines the fixed-orientation spatial shape alphabet and its Cartesian
// traversal. Resource feasibility is intentionally deferred to the production
// spatial-temporal scheduler, where non-overlapping tasks may reuse CGRAs.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_SPATIAL_TASK_CANDIDATE_SPACE_H
#define AMOEBA_SPATIAL_TASK_CANDIDATE_SPACE_H

#include "AnalyticalTaskCandidateCommon.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

inline constexpr llvm::StringLiteral kShapePolicy =
    "static-oriented-rectangles";
inline constexpr llvm::StringLiteral kSearchScope =
    "static-shape-cartesian-product";
inline constexpr llvm::StringLiteral kFactoredSearchScope =
    "static-shape-factored-cartesian-product";
inline constexpr llvm::StringLiteral kFactoredSpaceSchema =
    "amoeba-analytical-task-space";
inline constexpr llvm::StringLiteral kBoundedShapeSearchScope =
    "static-shape-bounded-changes-from-unit";
inline constexpr llvm::StringLiteral kSpatialCapacityPolicy =
    "per-task-fit-temporal-reuse-allowed";
inline constexpr llvm::StringLiteral kShapePruningPolicyNone = "none";
inline constexpr llvm::StringLiteral kShapePruningPolicyBoundedChangesFromUnit =
    "bounded-changes-from-unit";
inline constexpr llvm::StringLiteral kSpatialCandidateBudgetPolicy =
    "fail-before-publish";

// Stores one physical-CGRA rectangle and its corresponding mapper dimensions.
struct RectShape {
  int64_t rows = 1;
  int64_t cols = 1;
  int64_t mapperRows = 1;
  int64_t mapperCols = 1;

  int64_t cgraCount() const { return rows * cols; }
  std::string toCgraShapeAttrValue() const;
};

// Stores one task's shape choice within a program candidate.
struct TaskShapeChoice {
  std::string task;
  int64_t tripCount = 1;
  RectShape shape;
};

// Stores one ordered shape choice for every Taskflow task.
struct Candidate {
  std::string id;
  llvm::SmallVector<TaskShapeChoice> choices;
};

using ShapeIndexTupleConsumer =
    llvm::function_ref<bool(uint64_t, llvm::ArrayRef<size_t>)>;

// Exact mixed-radix indices are arbitrary-width because a legal program can
// have more than 64 bits of shape combinations. The callback receives only a
// borrowed index and the selected per-task shape indices; the traversal keeps
// no candidate rows in memory.
using ExactShapeIndexTupleConsumer =
    llvm::function_ref<bool(const llvm::APInt &, llvm::ArrayRef<size_t>)>;

// Exact decimal counts for the unrestricted and declared candidate families.
// The strings permit a large application graph's unrestricted Cartesian space
// to be reported without imposing an artificial uint64_t ceiling.
struct ShapeCandidateSpaceCounts {
  std::string unrestricted;
  std::string declared;
  std::string excluded;
};

llvm::SmallVector<RectShape> enumerateStaticRectShapes(int64_t gridRows,
                                                       int64_t gridCols,
                                                       int64_t perCgraRows,
                                                       int64_t perCgraCols,
                                                       int64_t maxCgrasPerTask);
// Visits the complete per-task shape Cartesian product. Each shape already
// fits the physical grid on its own; whole-program feasibility is determined
// later by the production spatial-temporal scheduler. `shapeIndices` follows
// task order and the candidate index is contiguous from zero.
bool visitShapeCartesianProduct(
    llvm::ArrayRef<llvm::SmallVector<RectShape>> shapesByTask,
    ShapeIndexTupleConsumer consume);

// Visits the complete Cartesian product without truncating its mixed-radix
// index to uint64_t. This is the exact/factored substrate used by callers
// that can rank candidates on the fly. It never applies a changed-task or
// candidate-count pruning rule.
bool visitShapeCartesianProductExact(
    llvm::ArrayRef<llvm::SmallVector<RectShape>> shapesByTask,
    ExactShapeIndexTupleConsumer consume);

// Converts an exact traversal index to its canonical candidate ID suffix.
std::string exactShapeIndexString(const llvm::APInt &index);

// Visits the complete family selected by the declared pruning policy. The
// bounded policy retains exactly those tuples with at most
// `maxChangedTasks` non-unit shapes, in the same task-major lexicographic
// order used by the unrestricted Cartesian traversal.
bool visitShapeCandidateSpace(
    llvm::ArrayRef<llvm::SmallVector<RectShape>> shapesByTask,
    llvm::StringRef shapePruningPolicy, int64_t maxChangedTasks,
    ShapeIndexTupleConsumer consume);

// Computes exact decimal counts for the unrestricted product, the declared
// policy family, and the candidates excluded by that policy.
ShapeCandidateSpaceCounts countShapeCandidateSpace(
    llvm::ArrayRef<llvm::SmallVector<RectShape>> shapesByTask,
    llvm::StringRef shapePruningPolicy, int64_t maxChangedTasks);
llvm::json::Object candidateJson(const Candidate &candidate);
} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_SPATIAL_TASK_CANDIDATE_SPACE_H
