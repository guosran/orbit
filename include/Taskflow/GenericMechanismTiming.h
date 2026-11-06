#ifndef ORBIT_TASKFLOW_GENERIC_MECHANISM_TIMING_H
#define ORBIT_TASKFLOW_GENERIC_MECHANISM_TIMING_H
#include "Taskflow/GenericMapping.h"
namespace orbit::taskflow {
// Adapter for kind=event-mechanism in the existing direct-AIR configuration.
// It uses public per-activity costs and ProgramTiming, never a whole-call fit.
// This function checks the public graph, not checkGenericMapping recursively.
bool evaluateGenericMechanismCost(const GenericAnalysis &,
    const GenericMappingPlan &, const llvm::json::Value &model,
    const llvm::json::Value &target, GenericDirectAIRCost &, std::string &error);
}
#endif
