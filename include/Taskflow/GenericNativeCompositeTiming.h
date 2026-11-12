#ifndef ORBIT_TASKFLOW_GENERIC_NATIVE_COMPOSITE_TIMING_H
#define ORBIT_TASKFLOW_GENERIC_NATIVE_COMPOSITE_TIMING_H
#include "Taskflow/GenericMapping.h"
#include "Taskflow/ProgramTiming.h"
namespace orbit::taskflow {
// Adapt the existing guarded native full-call linear/implementation composite
// provider to ProgramTiming. No new predictor, graph, or architecture is built.
// A different checked selected graph is feasible-but-unpriced: the native
// hardware-stage/control facts do not describe its executable implementation.
// true means structurally valid input, including UNKNOWN complete latency;
// false means malformed model or an invalid source/native/selected graph.
// The caller supplies a model/target calibrated for the SAME actual full-call
// boundary and runtime. No v29 pointwise coefficients or event fees are added.
bool evaluateGenericNativeCompositeTiming(
    const GenericAnalysis &, const GenericMappingPlan &,
    const ProgramGraph &selectedGraph, const llvm::json::Value &model,
    const llvm::json::Value &target, ProgramTimingEvaluation &,
    llvm::json::Value &receipt, std::string &error);
} // namespace orbit::taskflow
#endif
