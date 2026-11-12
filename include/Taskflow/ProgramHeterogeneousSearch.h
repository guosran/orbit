#ifndef ORBIT_TASKFLOW_PROGRAM_HETEROGENEOUS_SEARCH_H
#define ORBIT_TASKFLOW_PROGRAM_HETEROGENEOUS_SEARCH_H
#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/ProgramSearch.h"
namespace orbit::taskflow {
struct ProgramHeterogeneousSearchOptions {
  ProgramSearchOptions search;
  // Empty: {cpu,npu} for each task, with CPU as the all-zero baseline.
  // Otherwise one nonempty, duplicate-free row per original source task.
  std::vector<std::vector<std::string>> taskDeviceChoices;
  // Empty: {false,true} for each boundary. Otherwise one nonempty row per
  // boundary before tasks 1..N-1. Singleton rows freeze cuts or joins.
  std::vector<std::vector<bool>> boundaryChoices;
};
using ProgramHeterogeneousEvaluator =
    std::function<ProgramSearchEvaluation(const HeterogeneousMappingPlan &)>;
bool buildProgramHeterogeneousSearchSpace(
    const GenericAnalysis &, const ProgramHeterogeneousSearchOptions &,
    ProgramSearchSpace &, std::string &error);
std::string programHeterogeneousSearchSpaceIdentity(const GenericAnalysis &,
                                                    const ProgramSearchSpace &);
// Strict typed decoding of every dimension, binding and compatibility table.
// Device changes imply boundaries; an explicit cut there is a harmless alias.
// Mapping uses the existing public region/visibility/copy builder and checker.
bool materializeProgramHeterogeneousAssignment(const GenericAnalysis &,
                                               const ProgramSearchSpace &,
                                               const std::vector<int64_t> &,
                                               HeterogeneousMappingPlan &,
                                               std::string &error);
// A mapping the existing adapter cannot implement is UNKNOWN, not a proof of
// physical infeasibility. The caller alone supplies a complete cost/checker.
// result wraps its original JSON as cost_evaluation and appends heterogeneous
// source/choice/space/graph identities and the complete public graph.
ProgramSearchEvaluation evaluateProgramHeterogeneousAssignment(
    const GenericAnalysis &, const ProgramSearchSpace &,
    const std::vector<int64_t> &, const ProgramHeterogeneousEvaluator &);
ProgramSearchResult
searchProgramHeterogeneous(const GenericAnalysis &,
                           const ProgramHeterogeneousEvaluator &,
                           const ProgramHeterogeneousSearchOptions & = {});
} // namespace orbit::taskflow
#endif
