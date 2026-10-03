//===- ClosureLegalityPreflight.h -------------------------------*- C++ -*-===//
// Source-owned local legality facts for bounded graph-closure actions.
//
// These checks prove only facts that are visible in the current MLIR graph:
// an axis outside an actual output memref rank and the ordered cross-cell
// memory recurrence that the generic K materializer cannot preserve. Missing
// aliases, dynamic domains, software reductions, and unsupported kernel forms
// remain Unknown so the real materializer remains authoritative.
//===----------------------------------------------------------------------===//
#ifndef AMOEBA_CLOSURE_LEGALITY_PREFLIGHT_H
#define AMOEBA_CLOSURE_LEGALITY_PREFLIGHT_H

#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <cstdint>
#include <string>

namespace mlir::amoeba::neura::joint_scheduling {

using taskflow::TaskflowTaskOp;

enum class ClosureDimensionKind { Tile, KTile, Replica };
enum class ClosureDimensionDisposition { Proceed, ProvenIllegal, Unknown };

struct ClosureDimensionResult {
  ClosureDimensionDisposition disposition =
      ClosureDimensionDisposition::Proceed;
  std::string reason;
  std::string diagnostic;
  // Some source-owned materializers require a narrower ABI than the closure
  // action inventory.  A proven ABI mismatch is still Unknown (the action is
  // not proven illegal), but the known-buggy materializer must not be invoked
  // on it because it can erase a task while done-read results remain live.
  bool skipMaterializer = false;
};

struct ClosureDimensionFacts {
  bool outputRankKnown = false;
  int64_t outputRank = 0;
  bool kCandidate = false;
  bool kExtentKnown = false;
  int64_t kExtent = 0;
  bool crossCellK = false;
};

// This is the source-owned K classifier implemented by
// ExtractJointTaskGraphFactsPass.cpp.  Closure enumeration calls this wrapper
// instead of duplicating its recurrence/alias/counter proof.
struct ClosureStrictKClassification {
  bool potential = false;
  bool extentKnown = false;
  int64_t extent = 0;
  bool provenIllegal = false;
  std::string reason;
};

ClosureStrictKClassification
classifyClosureKLegality(TaskflowTaskOp task);

inline bool closureMemrefRank(Value storage, int64_t &rank) {
  if (!storage)
    return false;
  auto type = dyn_cast<MemRefType>(storage.getType());
  if (!type || !type.hasRank())
    return false;
  rank = type.getRank();
  return true;
}

inline ClosureDimensionFacts
inspectClosureTask(TaskflowTaskOp task,
                   const ClosureStrictKClassification *strictOverride = nullptr) {
  ClosureDimensionFacts facts;
  Value outputStorage = !task.getOriginalWriteMemrefs().empty()
                            ? task.getOriginalWriteMemrefs().front()
                            : (task.getWillWrites().empty()
                                   ? Value()
                                   : task.getWillWrites().front());
  facts.outputRankKnown = closureMemrefRank(outputStorage, facts.outputRank);

  ClosureStrictKClassification strict =
      strictOverride ? *strictOverride : classifyClosureKLegality(task);
  facts.kCandidate = strict.potential;
  facts.kExtentKnown = strict.extentKnown;
  facts.kExtent = strict.extent;
  facts.crossCellK = strict.provenIllegal &&
                     strict.reason ==
                         "memory-carried-cross-cell-feedback-order-not-preserved";
  return facts;
}

inline ClosureDimensionResult
preflightClosureDimension(TaskflowTaskOp task, ClosureDimensionKind kind,
                          int64_t axis, int64_t factor, StringRef kMode) {
  if (!task)
    return {ClosureDimensionDisposition::Unknown,
            "dimension_task_not_uniquely_identified", "task lookup failed"};
  if (factor < 1)
    return {ClosureDimensionDisposition::ProvenIllegal,
            "dimension_factor_invalid", "dimension factor must be positive"};
  if (factor == 1)
    return {};
  if (kind == ClosureDimensionKind::Tile)
    return {ClosureDimensionDisposition::Unknown, "tile_domain_unproven",
            "tile axes and extents are not proven by the local closure facts"};
  if (kind == ClosureDimensionKind::Replica) {
    Value outputStorage = !task.getOriginalWriteMemrefs().empty()
                              ? task.getOriginalWriteMemrefs().front()
                              : (task.getWillWrites().empty()
                                     ? Value()
                                     : task.getWillWrites().front());
    int64_t outputRank = 0;
    if (!closureMemrefRank(outputStorage, outputRank))
      return {ClosureDimensionDisposition::Unknown, "output_rank_unproven",
              "replica output memref rank is not proven"};
    if (axis < 0 || axis >= outputRank)
      return {ClosureDimensionDisposition::ProvenIllegal,
              "replica_axis_out_of_range",
              "selected replica axis is outside the actual output memref rank"};
    return {};
  }

  if (kind == ClosureDimensionKind::KTile &&
      !task.getDoneReads().empty())
    return {ClosureDimensionDisposition::Unknown,
            "k_task_has_done_read_results",
            "K materializer ABI does not preserve live done-read results",
            true};

  ClosureStrictKClassification strict = classifyClosureKLegality(task);
  if (strict.provenIllegal)
    return {ClosureDimensionDisposition::ProvenIllegal,
            strict.reason.empty() ? "k_rewrite_proven_illegal" : strict.reason,
            "source-owned strict K recurrence classifier rejected this rewrite"};
  if (!strict.potential)
    return {ClosureDimensionDisposition::Unknown, "k_domain_unproven",
            "a K reduction domain was not proven by the strict source-owned classifier"};
  if (!strict.extentKnown)
    return {ClosureDimensionDisposition::Unknown, "k_extent_unproven",
            "K reduction exists but its extent is not proven by the strict source-owned classifier"};
  if (factor > strict.extent)
    return {ClosureDimensionDisposition::ProvenIllegal,
            "k_factor_exceeds_extent",
            "selected K factor exceeds the proven K extent"};
  (void)kMode;
  return {};
}

} // namespace mlir::amoeba::neura::joint_scheduling

#endif
