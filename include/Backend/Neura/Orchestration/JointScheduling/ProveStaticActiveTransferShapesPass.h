//===- ProveStaticActiveTransferShapesPass.h -------------------*- C++ -*-===//
// Public proof record and verifier for opt-in active-transfer shape facts.
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_PROVE_STATIC_ACTIVE_TRANSFER_SHAPES_PASS_H
#define AMOEBA_PROVE_STATIC_ACTIVE_TRANSFER_SHAPES_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace mlir {
class Pass;
}

namespace mlir::amoeba::neura::joint_scheduling {

struct ActiveTransferAxisRange {
  int64_t lower = 0;
  int64_t upper = 0;
  std::string route;
  std::optional<int64_t> counterId;
  bool proven = false;
  std::string reason;
};

struct ActiveTransferAccessRecord {
  uint64_t ordinal = 0;
  std::string taskName;
  std::string operation;
  std::optional<unsigned> inputIndex;
  bool proven = false;
  std::string reason;
  llvm::SmallVector<ActiveTransferAxisRange, 4> axes;
};

struct StaticActiveTransferShapeProof {
  unsigned argumentIndex = 0;
  bool proven = false;
  bool hasCapacity = false;
  llvm::SmallVector<int64_t, 4> capacityShape;
  llvm::SmallVector<int64_t, 4> activeShape;
  llvm::SmallVector<std::string, 4> reasons;
  llvm::SmallVector<ActiveTransferAccessRecord, 16> accesses;
};

/// Re-derive a memref argument's access box from Taskflow and Neura IR.
/// Unsupported or ambiguous routes return a conservative unknown proof with
/// no activeShape; they do not mutate the function.
StaticActiveTransferShapeProof
analyzeStaticActiveTransferShape(func::FuncOp function, unsigned argumentIndex);

/// Serialize the complete deterministic proof record stored on the argument.
DictionaryAttr serializeStaticActiveTransferShapeProof(
    MLIRContext *context, func::FuncOp function,
    const StaticActiveTransferShapeProof &proof);

/// Recompute the proof and validate the function argument's active shape,
/// retained capacity, and detailed proof record. This is intended for any
/// future cost consumer; malformed or stale facts fail closed.
LogicalResult
verifyStaticActiveTransferShapeProof(func::FuncOp function,
                                     unsigned argumentIndex,
                                     std::string *errorMessage = nullptr);

/// Returns whether any derived active-transfer attribute is present. This
/// distinguishes absent metadata (which may be derived for a fresh source)
/// from supplied metadata that must be verified before it can be trusted.
bool hasStaticActiveTransferShapeFacts(func::FuncOp function,
                                       unsigned argumentIndex);

/// Re-derive and replace the derived active-transfer attributes for one
/// argument. Call only for a fresh source with no supplied proof facts or
/// after a trusted local rewrite; untrusted supplied facts must first pass
/// verifyStaticActiveTransferShapeProof instead of being silently repaired.
LogicalResult rederiveAndStoreStaticActiveTransferShapeProof(
    func::FuncOp function, unsigned argumentIndex,
    std::string *errorMessage = nullptr);

std::unique_ptr<Pass> createProveStaticActiveTransferShapesPass();

} // namespace mlir::amoeba::neura::joint_scheduling

#endif // AMOEBA_PROVE_STATIC_ACTIVE_TRANSFER_SHAPES_PASS_H
