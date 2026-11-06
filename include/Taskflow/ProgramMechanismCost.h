#ifndef ORBIT_TASKFLOW_PROGRAM_MECHANISM_COST_H
#define ORBIT_TASKFLOW_PROGRAM_MECHANISM_COST_H
#include "Taskflow/ProgramTiming.h"
namespace orbit::taskflow {
// Backend-supplied parameters annotate the existing execution graph. They are
// not another planner. Strict mode never silently substitutes assumptions.
struct ProgramMechanismParameter {
  std::optional<double> value;
  std::string unit, kind, source; // calibrated, architecture, assumed
};
struct ProgramMechanismModel {
  std::string id, mode = "strict", boundary;
  std::map<std::string, ProgramMechanismParameter> parameters;
  std::vector<std::string> implementations;
};
struct ProgramMechanismEvaluation {
  ProgramTimingEvaluation timing;
  llvm::json::Array quantities, parameters;
  std::vector<std::string> assumptions;
  std::string mode, status;
};
bool evaluateProgramMechanismCost(const ProgramGraph &,
    const ProgramMechanismModel &, ProgramMechanismEvaluation &,
    std::string &error);
llvm::json::Value serializeProgramMechanismCost(const ProgramMechanismEvaluation &);
}
#endif
