//===- JointSchedulingCandidateOrder.h - Candidate tie ordering -----------===//
//
// Shared ordering for production shape/schedule candidate IDs.  Production
// IDs use arbitrary-precision decimal ordinals; legacy IDs deliberately keep
// their historical lexical ordering.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_JOINT_SCHEDULING_CANDIDATE_ORDER_H
#define AMOEBA_JOINT_SCHEDULING_CANDIDATE_ORDER_H

#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <optional>

namespace mlir::amoeba::neura::joint_scheduling {

inline constexpr llvm::StringLiteral kProductionNumericTieKeyPolicy =
    "predicted_whole_program_cycles,semantic_graph_id,numeric_shape_ordinal,"
    "numeric_schedule_ordinal,graph_variant_id";

struct ProductionCandidateIdOrdinals {
  llvm::StringRef shape;
  llvm::StringRef schedule;
};

inline bool isProductionDecimal(llvm::StringRef value) {
  if (value.empty())
    return false;
  for (char digit : value)
    if (digit < '0' || digit > '9')
      return false;
  return true;
}

inline llvm::StringRef normalizeProductionDecimal(llvm::StringRef value) {
  llvm::StringRef normalized = value.ltrim('0');
  return normalized.empty() ? llvm::StringRef("0") : normalized;
}

inline int compareNormalizedProductionDecimal(llvm::StringRef lhs,
                                              llvm::StringRef rhs) {
  lhs = normalizeProductionDecimal(lhs);
  rhs = normalizeProductionDecimal(rhs);
  if (lhs.size() != rhs.size())
    return lhs.size() < rhs.size() ? -1 : 1;
  if (lhs == rhs)
    return 0;
  return lhs < rhs ? -1 : 1;
}

inline std::optional<ProductionCandidateIdOrdinals>
parseProductionCandidateId(llvm::StringRef candidateId) {
  constexpr llvm::StringLiteral shapePrefix = "shape-";
  constexpr llvm::StringLiteral schedulePrefix = "/schedule-";
  if (!candidateId.starts_with(shapePrefix))
    return std::nullopt;

  llvm::StringRef suffix = candidateId.drop_front(shapePrefix.size());
  const std::size_t separator = suffix.find(schedulePrefix);
  if (separator == llvm::StringRef::npos)
    return std::nullopt;

  llvm::StringRef shape = suffix.substr(0, separator);
  llvm::StringRef schedule =
      suffix.drop_front(separator + schedulePrefix.size());
  if (!isProductionDecimal(shape) || !isProductionDecimal(schedule))
    return std::nullopt;
  return ProductionCandidateIdOrdinals{shape, schedule};
}

// Returns a strict weak ordering result: negative if lhs precedes rhs, zero
// if they are identical, and positive if rhs precedes lhs.  When both IDs are
// canonical production IDs, shape and schedule ordinals are compared as
// arbitrary-precision decimals.  Legacy IDs form a separate ordering category and retain their
// existing lexical order within that category.
inline int compareProductionCandidateIds(llvm::StringRef lhs,
                                         llvm::StringRef rhs) {
  const auto left = parseProductionCandidateId(lhs);
  const auto right = parseProductionCandidateId(rhs);
  // Keep one ordering category per ID format. A pairwise lexical fallback
  // between formats could form a cycle with the numeric production ordering.
  if (static_cast<bool>(left) != static_cast<bool>(right))
    return left ? -1 : 1;
  if (!left) {
    if (lhs == rhs)
      return 0;
    return lhs < rhs ? -1 : 1;
  }

  int shapeOrder = compareNormalizedProductionDecimal(left->shape,
                                                       right->shape);
  if (shapeOrder != 0)
    return shapeOrder;
  int scheduleOrder = compareNormalizedProductionDecimal(left->schedule,
                                                          right->schedule);
  if (scheduleOrder != 0)
    return scheduleOrder;

  // Canonical producer IDs have no leading zero aliases.  Keep a deterministic
  // lexical tie break if a malformed/noncanonical alias reaches this helper.
  if (lhs == rhs)
    return 0;
  return lhs < rhs ? -1 : 1;
}

} // namespace mlir::amoeba::neura::joint_scheduling

#endif // AMOEBA_JOINT_SCHEDULING_CANDIDATE_ORDER_H
