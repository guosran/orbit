#ifndef ORBIT_TASKFLOW_PROGRAM_JOINT_MILP_H
#define ORBIT_TASKFLOW_PROGRAM_JOINT_MILP_H
#include "Taskflow/ProgramJointSearch.h"
namespace orbit::taskflow {
// Direct SCIP MILP over the same local choices and integer-nanosecond schedule
// as ProgramJointDomain. No complete-plan enumeration or pre-pricing.
// Optimal is relative to the declared, completely priced finite domain and
// SCIP's numerical tolerances, followed by independent integer checking.
// A checked baseline is submitted as a complete primal start (including derived
// auxiliaries), never fixed as variable bounds. Unary resources use conditional
// pairwise orders; higher capacities use exact integer start-anchor occupancy.
bool programJointMILPAvailable();
ProgramJointSolution solveProgramJointDomainMILP(
    const ProgramJointDomain &, const ConstraintSearchOptions & = {},
    const ProgramJointSolution *checkedBaseline = nullptr);
} // namespace orbit::taskflow
#endif
