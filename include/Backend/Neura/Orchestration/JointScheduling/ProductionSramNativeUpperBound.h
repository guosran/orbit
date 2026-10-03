//===- ProductionSramNativeUpperBound.h -----------------------*- C++ -*-===//
//
// Export a conservative storage upper bound from the native Taskflow IR.
//
// `task_orchestration_info.{read,write}_sram_locations` records where the
// production scheduler assigned a memory node.  It is not a byte-capacity or
// residency proof.  This interface therefore exports the bytes implied by
// the actual original memref operands.  The exporter itself has no capacity
// input; a separate source-configured capacity gate consumes these facts,
// while physical address, residency, and copy certification remain separate.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_PRODUCTION_SRAM_NATIVE_UPPER_BOUND_H
#define AMOEBA_PRODUCTION_SRAM_NATIVE_UPPER_BOUND_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {

/// One original memref backing observed at one or more scheduler SRAM
/// coordinates.  `bytes` is the complete backing extent used by the
/// whole-schedule/no-eviction upper bound, rather than a communication payload
/// size or a task tile size.
struct NativeSramBackingRecord {
  std::string id;
  std::string rootKind;
  int64_t bytes = 0;
  bool extentProven = false;
  bool aliasIdentityProven = true;
  bool copyBindingProven = false;
  uint64_t observedAccesses = 0;
  std::vector<int64_t> observedCgras;
};

/// One source-ordered original memref access paired with the SRAM coordinate
/// emitted for that operand.  Keeping these rows makes the exported backing
/// aggregation auditable against taskflow.task's original_*_memrefs lists.
struct NativeSramAccessRecord {
  std::string task;
  std::string direction;
  uint64_t operandIndex = 0;
  std::string backingId;
  int64_t cgra = -1;
  int64_t bytes = 0;
};

/// One native Neura indexed memory operation and the symbolic kernel input
/// that owns its backing. This is a capacity-coverage proof only: it records
/// the scheduler SRAM coordinates already assigned to the corresponding
/// original task operand, not a physical byte address or a copy schedule.
struct NativeSramIndexedAccessRecord {
  std::string task;
  std::string direction;
  uint64_t ordinal = 0;
  int64_t kernelInput = -1;
  std::string symbolicReference;
  std::string backingId;
  std::vector<int64_t> boundCgras;
  bool storageRootProven = false;
  bool schedulerBindingProven = false;
};

/// Result of exporting storage facts from one native function.  The exporter
/// intentionally has no hardware-capacity default.  A result may contain
/// useful per-CGRA required-byte records while still being `pending` because
/// a capacity input, residency interval, alias proof, or copy binding is
/// absent.  The capacity gate can certify the conservative bound without
/// certifying physical residency or copies.
struct NativeSramUpperBoundResult {
  std::string status = "pending";
  std::string reason;
  bool extentProofComplete = false;
  bool placementProofComplete = false;
  bool aliasProofComplete = false;
  bool copyBindingProofComplete = false;
  bool capacityProofComplete = false;
  /// Counts native indexed memory operations.  The current scheduler emits
  /// operand-to-CGRA coordinates, but no physical byte address or copy
  /// binding for these operations.
  uint64_t indexedLoadCount = 0;
  uint64_t indexedStoreCount = 0;
  uint64_t indexedAddressProofCount = 0;
  bool indexedAddressProofComplete = false;
  /// Counts indexed loads/stores whose symbolic kernel input resolves to an
  /// original task memory root and to that root's scheduler SRAM coordinates.
  /// This is sufficient for the conservative whole-backing capacity bound;
  /// it is independent of physical address/copy certification.
  uint64_t indexedStorageCoverageCount = 0;
  uint64_t indexedStorageCoverageFailureCount = 0;
  bool indexedStorageCoverageComplete = false;
  uint64_t taskCount = 0;
  uint64_t observedAccessCount = 0;
  std::map<int64_t, int64_t> requiredBytesByCgra;
  std::vector<NativeSramBackingRecord> backings;
  std::vector<NativeSramAccessRecord> accesses;
  std::vector<NativeSramIndexedAccessRecord> indexedAccesses;
};

/// Export actual taskflow original-memory operands and the SRAM locations
/// emitted by the production scheduler.  `gridRows` and `gridColumns` are
/// only coordinate validation parameters; they do not imply an SRAM byte
/// capacity.  The result is fail-closed and remains pending when any required
/// storage proof is missing.
NativeSramUpperBoundResult exportNativeSramUpperBound(
    func::FuncOp function, int64_t gridRows = 4, int64_t gridColumns = 4);

/// Annotate a module with the exporter result.  Registration belongs to the
/// owning backend; this private factory is provided so integration can add a
/// pass without making Python or a trace reader the source of storage facts.
std::unique_ptr<Pass> createExportProductionSramUpperBoundPass();

} // namespace mlir::amoeba::neura::joint_scheduling

#endif // AMOEBA_PRODUCTION_SRAM_NATIVE_UPPER_BOUND_H
