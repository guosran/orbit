//===- ProductionSramNativeCapacityGate.cpp ------------------*- C++ -*-===//
//
// Classify native conservative SRAM facts against a source-derived
// VectorCGRA capacity.  In particular, this file keeps the important
// distinction between an upper-bound excess and a proven physical overflow.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/Orchestration/JointScheduling/ProductionSramNativeCapacityGate.h"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {
namespace {

static void appendReason(NativeSramCapacityGateResult &result,
                         llvm::StringRef reason) {
  if (reason.empty())
    return;
  if (!result.reason.empty())
    result.reason += "; ";
  result.reason += reason.str();
}

static bool safeProduct(int64_t lhs, int64_t rhs, int64_t &out) {
  if (lhs <= 0 || rhs <= 0 ||
      lhs > std::numeric_limits<int64_t>::max() / rhs)
    return false;
  out = lhs * rhs;
  return true;
}

} // namespace

const char *nativeSramCapacityStatusName(NativeSramCapacityStatus status) {
  switch (status) {
  case NativeSramCapacityStatus::Pass:
    return "pass";
  case NativeSramCapacityStatus::Pending:
    return "pending";
  case NativeSramCapacityStatus::Fail:
    return "fail";
  }
  return "pending";
}

NativeSramCapacityGateResult classifyNativeSramCapacity(
    const NativeSramUpperBoundResult &upperBound,
    llvm::ArrayRef<int64_t> capacityBytesByCgra, int64_t gridRows,
    int64_t gridColumns) {
  NativeSramCapacityGateResult result;

  int64_t cgraCount = 0;
  if (!safeProduct(gridRows, gridColumns, cgraCount)) {
    result.status = NativeSramCapacityStatus::Fail;
    result.reason = "native SRAM gate requires positive, non-overflowing fabric dimensions";
    return result;
  }
  if (capacityBytesByCgra.size() != static_cast<size_t>(cgraCount)) {
    result.status = NativeSramCapacityStatus::Pending;
    result.reason = "source-derived VectorCGRA capacity must provide one byte value per target CGRA";
    return result;
  }
  for (int64_t cgra = 0; cgra < cgraCount; ++cgra) {
    int64_t capacity = capacityBytesByCgra[static_cast<size_t>(cgra)];
    if (capacity <= 0) {
      result.status = NativeSramCapacityStatus::Fail;
      result.reason = (llvm::Twine("capacity for CGRA ") + llvm::Twine(cgra) +
                       " must be positive")
                          .str();
      return result;
    }
    result.capacitiesByCgra.emplace(cgra, capacity);
  }

  if (upperBound.taskCount == 0 || upperBound.observedAccessCount == 0) {
    result.status = NativeSramCapacityStatus::Pending;
    result.reason = "native SRAM export contains no observed task memory accesses";
    return result;
  }
  if (!upperBound.extentProofComplete ||
      !upperBound.placementProofComplete) {
    result.status = NativeSramCapacityStatus::Pending;
    appendReason(result,
                 "native extent or production SRAM-coordinate proof is incomplete");
    if (!upperBound.reason.empty())
      appendReason(result, upperBound.reason);
    return result;
  }
  if (!upperBound.indexedStorageCoverageComplete) {
    result.status = NativeSramCapacityStatus::Pending;
    appendReason(result,
                 "native indexed memory coverage does not resolve every "
                 "Neura load/store to an original scheduled memory root");
    if (!upperBound.reason.empty())
      appendReason(result, upperBound.reason);
    return result;
  }

  for (const auto &entry : upperBound.requiredBytesByCgra) {
    int64_t cgra = entry.first;
    int64_t required = entry.second;
    if (cgra < 0 || cgra >= cgraCount || required < 0) {
      result.status = NativeSramCapacityStatus::Fail;
      result.reason = "native SRAM exporter emitted an invalid CGRA or byte requirement";
      return result;
    }
    result.conservativeUpperBoundByCgra[cgra] = required;
    int64_t capacity = result.capacitiesByCgra.at(cgra);
    if (required > capacity) {
      result.conservativeUpperBoundExceedsCapacity = true;
      appendReason(result,
                   (llvm::Twine("conservative whole-schedule upper bound on CGRA ") +
                    llvm::Twine(cgra) + " is " + llvm::Twine(required) +
                    " bytes, above derived capacity " + llvm::Twine(capacity) +
                    "; actual residency/copy overflow is not proven")
                       .str());
    }
  }

  if (result.conservativeUpperBoundExceedsCapacity) {
    result.status = NativeSramCapacityStatus::Pending;
    // The native exporter has no physical-copy or live-interval contract.  A
    // conservative full-backing charge is deliberately insufficient to fail
    // a schedule: a source-owned streaming/eviction proof may reduce peak
    // residency below this upper bound.
    result.actualOverflowProven = false;
    return result;
  }

  result.status = NativeSramCapacityStatus::Pass;
  result.conservativeCapacityProven = true;
  result.reason =
      "native whole-schedule conservative upper bound is within the derived "
      "VectorCGRA capacity; physical address/alias/copy certification remains "
      "separate and production_ready=false";
  return result;
}

namespace {

struct VerifyProductionSramNativeCapacityGatePass
    : public mlir::PassWrapper<VerifyProductionSramNativeCapacityGatePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      VerifyProductionSramNativeCapacityGatePass)

  VerifyProductionSramNativeCapacityGatePass() = default;
  VerifyProductionSramNativeCapacityGatePass(
      const VerifyProductionSramNativeCapacityGatePass &other)
      : PassWrapper(other) {}

  mlir::StringRef getArgument() const override {
    return "verify-production-sram-native-capacity-gate";
  }
  mlir::StringRef getDescription() const override {
    return "Classify native conservative SRAM facts against VectorCGRA capacity";
  }

  mlir::Pass::Option<int64_t> capacityBytesPerCgra{
      *this, "capacity-bytes-per-cgra",
      llvm::cl::desc("Source-derived VectorCGRA bytes per CGRA"),
      llvm::cl::init(0)};
  mlir::Pass::Option<int64_t> gridRows{
      *this, "grid-rows", llvm::cl::desc("Target fabric row count"),
      llvm::cl::init(4)};
  mlir::Pass::Option<int64_t> gridColumns{
      *this, "grid-columns", llvm::cl::desc("Target fabric column count"),
      llvm::cl::init(4)};
  mlir::Pass::Option<std::string> functionName{
      *this, "function", llvm::cl::desc("Optional taskflow function symbol"),
      llvm::cl::init("")};
  mlir::Pass::Option<std::string> outputFile{
      *this, "output", llvm::cl::desc("Native SRAM gate JSON evidence"),
      llvm::cl::init("")};

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *context = module.getContext();
    NativeSramUpperBoundResult upperBound;
    mlir::func::FuncOp selected;
    for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
      if (!functionName.getValue().empty() &&
          function.getSymName() != functionName.getValue())
        continue;
      bool hasTask = false;
      function.walk([&](taskflow::TaskflowTaskOp) { hasTask = true; });
      if (hasTask) {
        if (selected) {
          module.emitError("native SRAM gate requires exactly one selected taskflow function");
          return signalPassFailure();
        }
        selected = function;
      }
    }
    if (selected)
      upperBound = exportNativeSramUpperBound(
          selected, gridRows.getValue(), gridColumns.getValue());
    else
      upperBound.reason = functionName.getValue().empty()
                              ? "module has no taskflow function"
                              : (llvm::Twine("requested function '") +
                                 functionName.getValue() + "' has no taskflow function")
                                    .str();

    std::vector<int64_t> capacities;
    if (capacityBytesPerCgra.getValue() > 0 && gridRows.getValue() > 0 &&
        gridColumns.getValue() > 0 &&
        gridRows.getValue() <=
            std::numeric_limits<int64_t>::max() / gridColumns.getValue()) {
      int64_t count = gridRows.getValue() * gridColumns.getValue();
      capacities.assign(static_cast<size_t>(count),
                        capacityBytesPerCgra.getValue());
    }
    NativeSramCapacityGateResult result = classifyNativeSramCapacity(
        upperBound, capacities, gridRows.getValue(), gridColumns.getValue());
    module->setAttr("amoeba.joint_scheduling_sram_native_capacity_gate",
                    mlir::StringAttr::get(
                        context, nativeSramCapacityStatusName(result.status)));
    module->setAttr("amoeba.joint_scheduling_sram_native_capacity_gate_reason",
                    mlir::StringAttr::get(context, result.reason));
    module->setAttr(
        "amoeba.joint_scheduling_sram_native_capacity_upper_bound_exceeds",
        mlir::BoolAttr::get(context,
                            result.conservativeUpperBoundExceedsCapacity));
    module->setAttr(
        "amoeba.joint_scheduling_sram_native_capacity_proven",
        mlir::BoolAttr::get(context, result.conservativeCapacityProven));
    module->setAttr("amoeba.joint_scheduling_sram_native_capacity_actual_overflow",
                    mlir::BoolAttr::get(context, result.actualOverflowProven));
    if (!result.capacitiesByCgra.empty()) {
      llvm::SmallVector<mlir::Attribute> values;
      for (const auto &entry : result.capacitiesByCgra)
        values.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(context, 64), entry.second));
      module->setAttr("amoeba.joint_scheduling_sram_native_capacity_bytes_by_cgra",
                      mlir::ArrayAttr::get(context, values));
    }
    if (!result.conservativeUpperBoundByCgra.empty()) {
      llvm::SmallVector<mlir::Attribute> values;
      for (int64_t cgra = 0;
           cgra < static_cast<int64_t>(result.capacitiesByCgra.size()); ++cgra) {
        auto found = result.conservativeUpperBoundByCgra.find(cgra);
        int64_t value = found == result.conservativeUpperBoundByCgra.end()
                            ? 0
                            : found->second;
        values.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(context, 64), value));
      }
      module->setAttr(
          "amoeba.joint_scheduling_sram_native_capacity_upper_bound_by_cgra",
          mlir::ArrayAttr::get(context, values));
    }
    if (result.status == NativeSramCapacityStatus::Fail) {
      module.emitError() << result.reason;
      signalPassFailure();
    }
    if (!outputFile.getValue().empty()) {
      llvm::json::Object record;
      record["schema"] = "orbit-native-sram-capacity-gate-v1";
      record["status"] = nativeSramCapacityStatusName(result.status);
      record["reason"] = result.reason;
      record["policy"] = "conservative-full-backing-per-observed-sram-location";
      record["production_ready"] = false;
      record["actual_overflow_proven"] = result.actualOverflowProven;
      record["conservative_capacity_proven"] =
          result.conservativeCapacityProven;
      record["conservative_upper_bound_exceeds_capacity"] = result.conservativeUpperBoundExceedsCapacity;
      record["task_count"] = static_cast<int64_t>(upperBound.taskCount);
      record["observed_access_count"] = static_cast<int64_t>(upperBound.observedAccessCount);
      record["indexed_load_count"] = static_cast<int64_t>(upperBound.indexedLoadCount);
      record["indexed_store_count"] = static_cast<int64_t>(upperBound.indexedStoreCount);
      record["indexed_address_proof_count"] = static_cast<int64_t>(upperBound.indexedAddressProofCount);
      record["indexed_storage_coverage_count"] =
          static_cast<int64_t>(upperBound.indexedStorageCoverageCount);
      record["indexed_storage_coverage_failure_count"] =
          static_cast<int64_t>(upperBound.indexedStorageCoverageFailureCount);
      record["extent_proof_complete"] = upperBound.extentProofComplete;
      record["placement_proof_complete"] = upperBound.placementProofComplete;
      record["alias_proof_complete"] = upperBound.aliasProofComplete;
      record["copy_binding_proof_complete"] = upperBound.copyBindingProofComplete;
      record["indexed_address_proof_complete"] = upperBound.indexedAddressProofComplete;
      record["indexed_storage_coverage_complete"] =
          upperBound.indexedStorageCoverageComplete;
      if (selected) {
        record["function"] = selected.getSymName().str();
        if (auto graph = selected->getAttrOfType<mlir::StringAttr>("amoeba.graph_variant_id"))
          record["graph_variant_id"] = graph.getValue().str();
      }
      llvm::json::Object capacities, requirements;
      for (const auto &entry : result.capacitiesByCgra)
        capacities[std::to_string(entry.first)] = entry.second;
      for (const auto &entry : result.conservativeUpperBoundByCgra)
        requirements[std::to_string(entry.first)] = entry.second;
      record["capacity_bytes_by_cgra"] = std::move(capacities);
      record["conservative_upper_bound_bytes_by_cgra"] = std::move(requirements);
      llvm::json::Array indexedAccesses;
      for (const auto &access : upperBound.indexedAccesses) {
        llvm::json::Object indexed;
        indexed["task"] = access.task;
        indexed["direction"] = access.direction;
        indexed["ordinal"] = static_cast<int64_t>(access.ordinal);
        indexed["kernel_input"] = access.kernelInput;
        indexed["symbolic_reference"] = access.symbolicReference;
        indexed["backing_id"] = access.backingId;
        indexed["storage_root_proven"] = access.storageRootProven;
        indexed["scheduler_binding_proven"] =
            access.schedulerBindingProven;
        llvm::json::Array cgras;
        for (int64_t cgra : access.boundCgras)
          cgras.push_back(cgra);
        indexed["bound_cgras"] = std::move(cgras);
        indexedAccesses.push_back(std::move(indexed));
      }
      record["indexed_accesses"] = std::move(indexedAccesses);
      int fd = -1;
      llvm::SmallString<256> temporary;
      std::error_code error = llvm::sys::fs::createUniqueFile(
          outputFile.getValue() + ".partial-%%%%%%", fd, temporary);
      if (error) {
        module.emitError() << "cannot create SRAM evidence: " << error.message();
        return signalPassFailure();
      }
      {
        llvm::raw_fd_ostream stream(fd, true);
        stream << llvm::json::Value(std::move(record)) << "\n";
        stream.flush();
        if (stream.has_error()) {
          llvm::sys::fs::remove(temporary);
          module.emitError("cannot write SRAM evidence");
          return signalPassFailure();
        }
      }
      error = llvm::sys::fs::rename(temporary, outputFile.getValue());
      if (error) {
        llvm::sys::fs::remove(temporary);
        module.emitError() << "cannot publish SRAM evidence: " << error.message();
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createVerifyProductionSramNativeCapacityGatePass() {
  return std::make_unique<VerifyProductionSramNativeCapacityGatePass>();
}

} // namespace mlir::amoeba::neura::joint_scheduling
