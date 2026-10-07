//===- TaskflowFission.h ---------------------------------------*- C++ -*-===//
//
// Source-owned, replayable Taskflow hyperblock DFG fission.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_BACKEND_NEURA_TRANSFORMS_OPTIMIZATIONS_TASKFLOW_FISSION_H
#define AMOEBA_BACKEND_NEURA_TRANSFORMS_OPTIMIZATIONS_TASKFLOW_FISSION_H

#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace amoeba {
namespace neura {

/// One source-owned fission action. Function and task names resolve the exact
/// original Taskflow task in the immutable canonical parent module; leftNodes
/// are increasing ordinals in its hyperblock's non-terminator operation list.
struct TaskflowFissionAction {
  std::string function;
  std::string task;
  llvm::SmallVector<unsigned> leftNodes;
};

/// Enumerate every legal predecessor-closed DFG cut for a task. Unsupported
/// tasks return a successful empty list and a diagnostic in `error`. If the
/// legal result count would exceed `maxCuts`, return failure and publish no
/// partial list; a zero limit therefore permits only an empty result.
mlir::FailureOr<llvm::SmallVector<llvm::SmallVector<unsigned>>>
enumerateTaskflowFissionPartitions(taskflow::TaskflowTaskOp task,
                                   uint64_t maxCuts, std::string &error);

/// Materialize one validated source-side fission action in a module. The
/// selected task must be unique in the named function.
mlir::LogicalResult materializeTaskflowFission(
    mlir::ModuleOp module, llvm::StringRef function, llvm::StringRef task,
    llvm::ArrayRef<unsigned> leftNodes, std::string &error);

/// Prove that `candidate` is exactly the deterministic replay of `actions`
/// against the caller-pinned immutable canonical parent. Replay starts from a
/// clone of the parent and uses the same legality checker/materializer as the
/// public fission API; operation attributes are not accepted as authority.
/// Module comparison ignores debug locations but includes all operation
/// structure and attributes in generic MLIR form.
mlir::LogicalResult verifyTaskflowFissionReplay(
    mlir::ModuleOp canonicalParent, mlir::ModuleOp candidate,
    llvm::ArrayRef<TaskflowFissionAction> actions, std::string &error);

} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_BACKEND_NEURA_TRANSFORMS_OPTIMIZATIONS_TASKFLOW_FISSION_H
