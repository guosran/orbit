#ifndef ORBIT_TASKFLOW_PROGRAM_ALL_HETEROGENEOUS_COST_H
#define ORBIT_TASKFLOW_PROGRAM_ALL_HETEROGENEOUS_COST_H

#include "Taskflow/HeterogeneousSearch.h"
#include "Taskflow/ProgramAllSearch.h"

namespace orbit::taskflow {
// Shape-normalized source semantics for local calibration. Mapping attributes
// and extracted function ordinals are excluded; numeric contracts are retained.
std::string heterogeneousCostSourceFamily(mlir::ModuleOp);
// Export the exact selected parent/children and price nonoverlapping runtime
// phases. Calibration describes local implementations, never complete plans.
ProgramSearchEvaluation
evaluateProgramAllHeterogeneousCost(const ProgramAllHeterogeneousCandidate &,
                                    const llvm::json::Value &,
                                    HeterogeneousAIREmitter = {});
} // namespace orbit::taskflow
#endif
