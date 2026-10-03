//===- SpatialTaskCandidateSpace.cpp -----------------------------------===//
//
// Implements construction and traversal of the static rectangular
// task-shape candidate space.
//
//===----------------------------------------------------------------------===//

#include "SpatialTaskCandidateSpace.h"

#include "Backend/Neura/Orchestration/AnalyticalBasedTaskOrchestration/AnalyticalBasedTaskOrchestration.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

// Converts the physical CGRA rectangle into the string stored in the
// Taskflow `cgra_shape` attribute, such as `1x2`.
std::string RectShape::toCgraShapeAttrValue() const {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << llvm::formatv("{0}x{1}", rows, cols);
  return result;
}

// Enumerates every legal static physical rectangle and derives its mapper
// dimensions from the architecture getters. The deterministic order defines
// the mixed-radix alphabet for candidate IDs.
// TODO: Extend the analytical candidate schema and its consumers to represent
// non-rectangular shapes. The analytical search intentionally enumerates only
// fixed-orientation rectangles until that contract exists end to end.
SmallVector<RectShape> enumerateStaticRectShapes(int64_t gridRows,
                                                 int64_t gridCols,
                                                 int64_t perCgraRows,
                                                 int64_t perCgraCols,
                                                 int64_t maxCgrasPerTask) {
  // The candidate domain contains only concrete integer rectangles.
  SmallVector<RectShape> result;
  if (gridRows <= 0 || gridCols <= 0 || perCgraRows <= 0 || perCgraCols <= 0 ||
      maxCgrasPerTask <= 0)
    return result;

  const int64_t gridSize =
      gridRows > std::numeric_limits<int64_t>::max() / gridCols
          ? std::numeric_limits<int64_t>::max()
          : gridRows * gridCols;
  for (int64_t count = 1; count <= std::min(gridSize, maxCgrasPerTask);
       ++count) {
    for (const CgraShape &physicalShape :
         taskflow::AnalyticalBasedTaskOrchestration::getRectangularShapes(
             static_cast<int>(count), static_cast<int>(gridRows),
             static_cast<int>(gridCols))) {
      if (physicalShape.rows >
              std::numeric_limits<int64_t>::max() / perCgraRows ||
          physicalShape.cols >
              std::numeric_limits<int64_t>::max() / perCgraCols)
        continue;
      int64_t mapperRows = physicalShape.rows * perCgraRows;
      int64_t mapperCols = physicalShape.cols * perCgraCols;
      result.push_back(
          {physicalShape.rows, physicalShape.cols, mapperRows, mapperCols});
    }
  }
  return result;
}

static std::optional<unsigned> exactIndexBitWidth(size_t taskCount) {
  constexpr size_t bitsPerTask = 64;
  if (taskCount > (std::numeric_limits<unsigned>::max() / bitsPerTask) - 1)
    return std::nullopt;
  return std::max<unsigned>(64, static_cast<unsigned>(bitsPerTask *
                                                       (taskCount + 1)));
}

bool visitShapeCartesianProductExact(
    ArrayRef<SmallVector<RectShape>> shapesByTask,
    ExactShapeIndexTupleConsumer consume) {
  if (shapesByTask.empty() ||
      llvm::any_of(shapesByTask,
                   [](const auto &shapes) { return shapes.empty(); }))
    return true;
  std::optional<unsigned> width = exactIndexBitWidth(shapesByTask.size());
  if (!width)
    return false;

  SmallVector<size_t> selectedIndices;
  llvm::APInt index(*width, 0);
  std::function<bool(size_t)> visit = [&](size_t taskIndex) {
    if (taskIndex == shapesByTask.size()) {
      if (!consume(index, selectedIndices))
        return false;
      if (index.isMaxValue())
        return false;
      ++index;
      return true;
    }

    for (auto [shapeIndex, unused] : llvm::enumerate(shapesByTask[taskIndex])) {
      (void)unused;
      selectedIndices.push_back(shapeIndex);
      if (!visit(taskIndex + 1))
        return false;
      selectedIndices.pop_back();
    }
    return true;
  };
  return visit(0);
}

bool visitShapeCartesianProduct(ArrayRef<SmallVector<RectShape>> shapesByTask,
                                ShapeIndexTupleConsumer consume) {
  return visitShapeCartesianProductExact(
      shapesByTask, [&](const llvm::APInt &index,
                        ArrayRef<size_t> shapeIndices) {
        if (index.getActiveBits() > 64)
          return false;
        return consume(index.getZExtValue(), shapeIndices);
      });
}

std::string exactShapeIndexString(const llvm::APInt &index) {
  llvm::SmallString<64> text;
  index.toString(text, 10, false, false);
  return text.str().str();
}

static bool isUnitShape(const RectShape &shape) {
  return shape.rows == 1 && shape.cols == 1;
}

bool visitShapeCandidateSpace(ArrayRef<SmallVector<RectShape>> shapesByTask,
                              StringRef shapePruningPolicy,
                              int64_t maxChangedTasks,
                              ShapeIndexTupleConsumer consume) {
  if (shapePruningPolicy == kShapePruningPolicyNone)
    return visitShapeCartesianProduct(shapesByTask, consume);
  if (shapePruningPolicy != kShapePruningPolicyBoundedChangesFromUnit ||
      maxChangedTasks < 0)
    return false;
  if (shapesByTask.empty() ||
      llvm::any_of(shapesByTask,
                   [](const auto &shapes) { return shapes.empty(); }))
    return true;

  SmallVector<size_t> selectedIndices;
  uint64_t index = 0;
  std::function<bool(size_t, int64_t)> visit =
      [&](size_t taskIndex, int64_t changedTasks) {
        if (taskIndex == shapesByTask.size()) {
          if (!consume(index, selectedIndices))
            return false;
          if (index == std::numeric_limits<uint64_t>::max())
            return false;
          ++index;
          return true;
        }

        for (auto [shapeIndex, shape] :
             llvm::enumerate(shapesByTask[taskIndex])) {
          const int64_t nextChangedTasks =
              changedTasks + (isUnitShape(shape) ? 0 : 1);
          if (nextChangedTasks > maxChangedTasks)
            continue;
          selectedIndices.push_back(shapeIndex);
          if (!visit(taskIndex + 1, nextChangedTasks))
            return false;
          selectedIndices.pop_back();
        }
        return true;
      };
  return visit(0, 0);
}

static std::string decimalString(const llvm::APInt &value) {
  llvm::SmallString<64> text;
  value.toString(text, 10, false, false);
  return text.str().str();
}

ShapeCandidateSpaceCounts countShapeCandidateSpace(
    ArrayRef<SmallVector<RectShape>> shapesByTask,
    StringRef shapePruningPolicy, int64_t maxChangedTasks) {
  ShapeCandidateSpaceCounts result;
  if (shapePruningPolicy != kShapePruningPolicyNone &&
      shapePruningPolicy != kShapePruningPolicyBoundedChangesFromUnit)
    return result;
  if (shapePruningPolicy == kShapePruningPolicyBoundedChangesFromUnit &&
      maxChangedTasks < 0)
    return result;
  if (shapesByTask.empty() ||
      llvm::any_of(shapesByTask,
                   [](const auto &shapes) { return shapes.empty(); }))
    return result;

  // Every shape count is bounded by the architecture's finite grid. Giving
  // each task 64 bits in the accumulator is therefore sufficient for the
  // exact product even when it exceeds uint64_t.
  const unsigned width =
      64u * static_cast<unsigned>(shapesByTask.size() + 1);
  SmallVector<llvm::APInt> ways;
  ways.emplace_back(width, 1);
  for (ArrayRef<RectShape> shapes : shapesByTask) {
    uint64_t unitCount = 0;
    for (const RectShape &shape : shapes)
      unitCount += isUnitShape(shape);
    // The bounded policy is defined relative to one canonical unit shape.
    if (unitCount != 1)
      return {};
    const uint64_t nonUnitCount = shapes.size() - unitCount;
    SmallVector<llvm::APInt> next;
    next.reserve(ways.size() + 1);
    for (size_t index = 0; index <= ways.size(); ++index)
      next.emplace_back(width, 0);
    for (size_t changed = 0; changed < ways.size(); ++changed) {
      next[changed] += ways[changed] * llvm::APInt(width, unitCount);
      next[changed + 1] +=
          ways[changed] * llvm::APInt(width, nonUnitCount);
    }
    ways = std::move(next);
  }

  llvm::APInt unrestricted(width, 0);
  for (const llvm::APInt &count : ways)
    unrestricted += count;
  llvm::APInt declared(width, 0);
  if (shapePruningPolicy == kShapePruningPolicyNone) {
    declared = unrestricted;
  } else {
    const size_t limit = std::min<size_t>(
        static_cast<size_t>(maxChangedTasks), shapesByTask.size());
    for (size_t changed = 0; changed <= limit; ++changed)
      declared += ways[changed];
  }
  result.unrestricted = decimalString(unrestricted);
  result.declared = decimalString(declared);
  result.excluded = decimalString(unrestricted - declared);
  return result;
}

// Serializes the shape fields used by candidate records. The explicit tile
// dimensions are the only mapper-shape truth; a `rect-4x8` string is produced
// only for diagnostics when needed.
static llvm::json::Object shapeJson(const RectShape &shape) {
  llvm::json::Object object;
  object["kind"] = "rect";
  object["rows"] = shape.rows;
  object["cols"] = shape.cols;
  object["cgra_count"] = shape.cgraCount();
  object["cgra_shape"] = shape.toCgraShapeAttrValue();
  object["mapper_tile_rows"] = shape.mapperRows;
  object["mapper_tile_cols"] = shape.mapperCols;
  return object;
}

// Serializes one candidate while preserving task order. The candidate ID is a
// sequential index, so no task-body identity or file-derived metadata is
// required to interpret it within its validated manifest.
llvm::json::Object candidateJson(const Candidate &candidate) {
  llvm::json::Array choices;
  for (const TaskShapeChoice &choice : candidate.choices) {
    llvm::json::Object record;
    record["task"] = choice.task;
    record["trip_count"] = choice.tripCount;
    record["shape"] = shapeJson(choice.shape);
    choices.push_back(std::move(record));
  }
  llvm::json::Object record;
  record["record_type"] = "candidate";
  record["schema"] = kCandidateSchema.str();
  record["candidate_id"] = candidate.id;
  record["task_shapes"] = std::move(choices);
  return record;
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir
