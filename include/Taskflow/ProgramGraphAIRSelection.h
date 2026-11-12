#ifndef ORBIT_TASKFLOW_PROGRAM_GRAPH_AIR_SELECTION_H
#define ORBIT_TASKFLOW_PROGRAM_GRAPH_AIR_SELECTION_H
#include "Taskflow/ProgramMapping.h"
#include "mlir/IR/BuiltinOps.h"
namespace orbit::taskflow {
struct ProgramGraphAIRSelection {
  std::string graphId;
  size_t begin = 0, end = 0;
};
// Full source/graph validation precedes every selection proof. No local copy
// may have its creator and any consumer on opposite sides of a boundary.
bool checkProgramGraphAIRSelection(const ProgramGraph &,
                                   const ProgramGraphAIRSelection &,
                                   std::string &error);
bool checkProgramGraphAIRSelectionUnion(
    const ProgramGraph &, llvm::ArrayRef<ProgramGraphAIRSelection>,
    llvm::json::Value &receipt, std::string &error);
bool buildProgramGraphAIRSelections(const ProgramGraph &,
                                    int64_t maximumTransfers,
                                    std::vector<ProgramGraphAIRSelection> &,
                                    llvm::json::Value &receipt,
                                    std::string &error);
// Re-emits only selected source events, preserving the full ABI and static
// allocation slots. The caller runs every selection serially with wait2,
// syncing inputs once before the sequence and outputs once after it.
bool lowerProgramGraphAIRSelection(const ProgramGraph &,
                                   const ProgramGraphAIRSelection &,
                                   mlir::OwningOpRef<mlir::ModuleOp> &,
                                   llvm::json::Value &manifest,
                                   std::string &error);
} // namespace orbit::taskflow
#endif
