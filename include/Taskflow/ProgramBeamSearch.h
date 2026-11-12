#ifndef ORBIT_TASKFLOW_PROGRAM_BEAM_SEARCH_H
#define ORBIT_TASKFLOW_PROGRAM_BEAM_SEARCH_H
#include "Taskflow/ProgramSearch.h"
namespace orbit::taskflow {
// Unknown-cost states have no ranking score. A bounded FIFO frontier lets the
// search cross such states without claiming they improve the checked objective.
struct ProgramBeamExplorationOptions {
  // Reserved within beamWidth; zero restores scored-only exploration.
  int64_t unknownSlots = 1;
  // Queue payload is also bounded by ProgramSearchOptions::maxTupleCells.
  int64_t maxPendingStates = 256;
};
// Large searches can write every evaluation to a durable sink without keeping
// a second JSON copy of every assignment in RAM. This changes evidence
// storage only: visited assignments, beam ordering and convergence are intact.
struct ProgramBeamTraceOptions {
  bool retainInMemory = true;
  std::function<void(const llvm::json::Value &)> sink;
};
// Explicit adapter over the public finite domain and evaluator. Reuses
// selectSearchBeam, but is not the legacy driver's default action generator.
// Beam accepts +infinity wallSeconds, maxEvaluations=0 and maxRounds=-1
// to run until its checked neighborhood converges. These sentinels do not
// change the finite-domain CP adapter contract.
// Neighbors change one dimension; coupled fields belong in compound choices.
ProgramSearchResult
searchProgramSpaceBeam(const ProgramSearchSpace &,
                       const ProgramSearchEvaluator &,
                       const ProgramSearchOptions &, int64_t beamWidth,
                       int64_t maxRounds, int64_t diversitySlots = 0,
                       const ProgramBeamExplorationOptions &exploration = {},
                       const ProgramBeamTraceOptions &traceOptions = {});
} // namespace orbit::taskflow
#endif
