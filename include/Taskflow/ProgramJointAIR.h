#ifndef ORBIT_TASKFLOW_PROGRAM_JOINT_AIR_H
#define ORBIT_TASKFLOW_PROGRAM_JOINT_AIR_H
#include "Taskflow/ProgramJointSearch.h"
namespace orbit::taskflow {
bool configureProgramJointNPU(llvm::json::Value &, int64_t cores,
                              std::string &);
bool lowerProgramJointToAIR(mlir::ModuleOp, const llvm::json::Value &,
                            const ProgramJointDomain &,
                            const ProgramJointSolution &, const ProgramGraph &,
                            mlir::OwningOpRef<mlir::ModuleOp> &,
                            llvm::json::Value &, std::string &);
bool checkProgramJointAIE(mlir::ModuleOp, const llvm::json::Value &,
                          const ProgramJointDomain &,
                          const ProgramJointSolution &, const ProgramGraph &,
                          mlir::ModuleOp, llvm::json::Value &, std::string &);
} // namespace orbit::taskflow
#endif
