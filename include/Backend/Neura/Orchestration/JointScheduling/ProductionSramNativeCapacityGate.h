//===- ProductionSramNativeCapacityGate.h --------------------*- C++ -*-===//
//
// Capacity classification for facts exported directly from a native
// Taskflow function.  This gate consumes the conservative, whole-schedule
// upper bound; it does not turn an upper-bound excess into an actual SRAM
// overflow without a source-owned residency/copy binding proof.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_PRODUCTION_SRAM_NATIVE_CAPACITY_GATE_H
#define AMOEBA_PRODUCTION_SRAM_NATIVE_CAPACITY_GATE_H

#include "Backend/Neura/Orchestration/JointScheduling/ProductionSramNativeUpperBound.h"

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <map>
#include <string>

namespace mlir::amoeba::neura::joint_scheduling {

enum class NativeSramCapacityStatus { Pass, Pending, Fail };

struct NativeSramCapacityGateResult {
  NativeSramCapacityStatus status = NativeSramCapacityStatus::Pending;
  std::string reason;
  std::map<int64_t, int64_t> capacitiesByCgra;
  std::map<int64_t, int64_t> conservativeUpperBoundByCgra;
  bool conservativeUpperBoundExceedsCapacity = false;
  /// True when the complete conservative whole-backing bound is within the
  /// supplied hardware capacity. This does not certify physical addresses,
  /// aliases, copies, or production readiness.
  bool conservativeCapacityProven = false;
  bool actualOverflowProven = false;
};

const char *nativeSramCapacityStatusName(NativeSramCapacityStatus status);

/// Classify the native export against one capacity for each target CGRA.
///
/// The capacity vector is the source-derived VectorCGRA configuration.  The
/// exporter charges each complete backing once per observed SRAM coordinate
/// under a whole-schedule/no-eviction policy. Therefore a complete indexed
/// storage-coverage proof and an upper bound at or below capacity are
/// sufficient for `pass` as a capacity-feasibility certificate. Physical
/// address, alias, copy, and residency certification remain separate fields
/// and do not gate this conservative capacity result. An upper-bound excess
/// is only `pending`: it may be avoided by streaming, eviction, or a copy
/// schedule, and this result contains no proof of actual residency overflow.
/// Malformed coordinates/capacities are `fail`.
NativeSramCapacityGateResult classifyNativeSramCapacity(
    const NativeSramUpperBoundResult &upperBound,
    llvm::ArrayRef<int64_t> capacityBytesByCgra, int64_t gridRows = 4,
    int64_t gridColumns = 4);

/// Annotate a native module with the same classification.  The owning
/// backend registers this factory and supplies the source-derived per-CGRA
/// capacity; no generic capacity default is installed here.
std::unique_ptr<mlir::Pass> createVerifyProductionSramNativeCapacityGatePass();

} // namespace mlir::amoeba::neura::joint_scheduling

#endif // AMOEBA_PRODUCTION_SRAM_NATIVE_CAPACITY_GATE_H
