#ifndef ORBIT_TASKFLOW_PROGRAM_GRAPH_AIR_H
#define ORBIT_TASKFLOW_PROGRAM_GRAPH_AIR_H
#include "Taskflow/ProgramMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
namespace orbit::taskflow {
struct ProgramGraphAIRLoopPlan;
struct ProgramGraphAIRSelection;
// Consume a checked, source-backed graph directly. No mapping planner or
// optimizer is called. Independent physical cores share one configuration;
// logical groups on a core preserve their exact copies in one herd.
bool lowerProgramGraphToAIR(
    const ProgramGraph &, mlir::OwningOpRef<mlir::ModuleOp> &,
    llvm::json::Value &manifest, std::string &error,
    const ProgramGraphAIRLoopPlan *loopPlan = nullptr,
    const ProgramGraphAIRSelection *selection = nullptr);
// Reuses the established native source-body cloning/address translation.
// This helper never selects tiling, copies, allocations, or placement.
bool emitProgramGraphScalarInstance(mlir::OpBuilder &, const ProgramGraph &,
                                    const MappingTaskInstance &,
                                    llvm::ArrayRef<MappingRegion>,
                                    llvm::ArrayRef<mlir::Value>,
                                    std::string &error);
} // namespace orbit::taskflow
#endif
