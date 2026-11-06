#ifndef ORBIT_TASKFLOW_GENERIC_PROGRAM_TIMING_H
#define ORBIT_TASKFLOW_GENERIC_PROGRAM_TIMING_H
#include "Taskflow/GenericMapping.h"
#include "Taskflow/ProgramTiming.h"
namespace orbit::taskflow {
// Adapt the existing guarded direct-AIR implementation provider to the public
// event evaluator. No new graph, parameters, or planner are introduced. A full
// invocation cost owns its complete coverage once; internal durations remain
// unknown until a separately identifiable implementation model is supplied.
bool evaluateGenericProgramTiming(const GenericAnalysis &, const GenericMappingPlan &,
                                 llvm::json::Value &, std::string &error);
}
#endif
