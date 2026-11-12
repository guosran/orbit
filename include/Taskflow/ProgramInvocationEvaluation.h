#ifndef ORBIT_TASKFLOW_PROGRAM_INVOCATION_EVALUATION_H
#define ORBIT_TASKFLOW_PROGRAM_INVOCATION_EVALUATION_H
#include "Taskflow/ProgramInvocationCost.h"
namespace orbit::taskflow {
struct ProgramInvocationModel {
  ProgramMechanismModel mechanisms;
  ProgramInvocationRuntimeContract runtime;
  std::string calibrationRuntimeIdentity;
  std::string completionPolicy = "post-device-overhead";
};
bool parseProgramInvocationModel(const llvm::json::Value &,
                                 ProgramInvocationModel &, std::string &error);
// Body graph events and invocation lifecycle have disjoint ownership. Prepared
// programs exclude physical context/ELF loading and logical group
// configuration.
bool evaluateProgramInvocationModel(const ProgramGraph &,
                                    const ProgramInvocationModel &,
                                    ProgramMechanismEvaluation &,
                                    llvm::json::Value &receipt,
                                    std::string &error);
} // namespace orbit::taskflow
#endif
