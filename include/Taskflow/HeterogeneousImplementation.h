#ifndef ORBIT_TASKFLOW_HETEROGENEOUS_IMPLEMENTATION_H
#define ORBIT_TASKFLOW_HETEROGENEOUS_IMPLEMENTATION_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace orbit::taskflow {
// Lower an already materialized/checked execution-region child. Computation is
// cloned from the actual Taskflow bodies; no operation recipe is substituted.
// The child function's ranked memref ABI and symbol remain unchanged.
bool lowerHeterogeneousCPUImplementation(
    mlir::ModuleOp child, mlir::OwningOpRef<mlir::ModuleOp> &plainCPU,
    std::string &error);
} // namespace orbit::taskflow

#endif
