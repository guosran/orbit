#ifndef ORBIT_TASKFLOW_PROGRAM_TIMING_H
#define ORBIT_TASKFLOW_PROGRAM_TIMING_H
#include "Taskflow/ProgramMapping.h"
namespace orbit::taskflow {
// Cost annotations reference the existing graph; they do not carry computation,
// data or an alternative mapping. A composite exclusively owns all listed
// events.
struct ProgramActivityCost {
  std::string id, implementation, boundary, provenance, unknownReason;
  std::vector<size_t> events, regionEvents;
  // For unrefined native invocation boundaries only; a whole-region composite
  // explicitly owns submit/completion as well as all region member events.
  std::vector<size_t> executionRegions;
  std::map<std::string, int64_t> resourceDemands;
  std::optional<double> durationMs, lowerMs, upperMs;
  // Additional runtime work outside the graph requires explicit ordering by
  // cost ID. Its boundary must say which executor work is included.
  bool runtimeOnly = false;
  std::vector<std::string> predecessors;
};
struct ProgramTimingRequest {
  std::string graphId, modelIdentity;
  std::string boundary = "graph-events-and-region-lifecycle";
  bool serialize = false;
  std::vector<ProgramActivityCost> costs;
  // Optional permutation of explicitly supplied cost IDs. Generated unknown
  // activities follow in their established order. Does not change graph edges.
  std::vector<std::string> schedulePriority;
};
struct ProgramTimedActivity {
  std::string id, status, reason, boundary, provenance;
  std::vector<size_t> events, regionEvents, executionRegions;
  std::vector<std::string> predecessors, resourcePredecessors;
  std::vector<std::string> bufferPredecessors;
  std::map<std::string, int64_t> resourceDemands;
  std::optional<double> durationMs, startMs, endMs;
  double lowerStartMs = 0, lowerEndMs = 0;
};
struct ProgramTimingEvaluation {
  std::string graphId, modelIdentity, boundary;
  std::string policy = "existing-static-deterministic-topological-list";
  double knownWorkMs = 0, lowerBoundMs = 0;
  std::optional<double> upperBoundMs, makespanMs;
  bool knownCapacitiesVerified = false, allPhysicalCapacitiesKnown = false;
  std::vector<ProgramTimedActivity> activities;
  std::vector<std::string> unknowns, assumptions, criticalPath;
  std::vector<std::string> schedulePriority;
  std::vector<ProgramBufferOrdering> bufferOrdering;
  std::map<std::string, int64_t> peakMemoryBytes;
};
// Uses scheduleStaticPlan for resource calendars. Explicit integer resource
// shares reserve fixed capacity at the duration supplied by the implementation;
// adaptive fluid bandwidth sharing is not inferred. Bounds are conservative:
// precedence/aggregate resource work below, fully serialized work above.
bool evaluateProgramTiming(const ProgramGraph &, const ProgramTimingRequest &,
                           ProgramTimingEvaluation &, std::string &error);
llvm::json::Value serializeProgramTiming(const ProgramTimingEvaluation &);
} // namespace orbit::taskflow
#endif
