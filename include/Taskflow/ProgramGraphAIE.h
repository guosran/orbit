#ifndef ORBIT_TASKFLOW_PROGRAM_GRAPH_AIE_H
#define ORBIT_TASKFLOW_PROGRAM_GRAPH_AIE_H
#include "Taskflow/ProgramMapping.h"
#include "mlir/IR/BuiltinOps.h"
namespace orbit::taskflow {
// Independently compare a source-backed graph's direct AIR realization with
// final generic AIE. Attributes locate allocations; actual SSA, addresses,
// arithmetic, physical owners, DMA routes and locks must corroborate them.
// Unsupported control/topology fails closed. This does not certify timing or
// analytical configuration setup costs.
bool checkProgramGraphAIE(const ProgramGraph &, mlir::ModuleOp expectedAIR,
                          mlir::ModuleOp actualAIE, llvm::json::Value &receipt,
                          std::string &error);
} // namespace orbit::taskflow
#endif
