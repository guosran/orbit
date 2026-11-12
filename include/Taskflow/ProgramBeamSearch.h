#ifndef ORBIT_TASKFLOW_PROGRAM_BEAM_SEARCH_H
#define ORBIT_TASKFLOW_PROGRAM_BEAM_SEARCH_H
#include "Taskflow/ProgramSearch.h"
namespace orbit::taskflow {
// Explicit adapter over the public finite domain and evaluator. Reuses
// selectSearchBeam, but is not the legacy driver's default action generator.
// Neighbors change one dimension; coupled fields belong in compound choices.
ProgramSearchResult searchProgramSpaceBeam(const ProgramSearchSpace &,
                                           const ProgramSearchEvaluator &,
                                           const ProgramSearchOptions &,
                                           int64_t beamWidth, int64_t maxRounds,
                                           int64_t diversitySlots = 0);
} // namespace orbit::taskflow
#endif
