#ifndef ORBIT_TASKFLOW_CONSTRAINT_SEARCH_H
#define ORBIT_TASKFLOW_CONSTRAINT_SEARCH_H
#include <cstdint>
#include <string>
#include <vector>
namespace orbit::taskflow {
struct ConstraintPattern {
  int64_t id=0;
  std::vector<int64_t> tasks;
  int64_t tileIndex=0, costNs=0, bytes=0, inputCount=0, outputCount=0;
  std::vector<int64_t> exclusiveStorageIds;
};
struct ConstraintEdge { int64_t source=0,target=0; };
struct ConstraintSearchProblem {
  int64_t taskCount=0,tileCount=0;
  std::vector<ConstraintPattern> patterns;
  std::vector<ConstraintEdge> edges;
  std::vector<int64_t> tileBaseCostNs;
};
struct ConstraintSearchOptions {
  double wallSeconds=1;
  int workers=1,randomSeed=0;
  int64_t maxVariables=100000,maxConstraints=500000,maxPatterns=20000;
  int64_t maxTasks=512,maxTiles=16,maxInputEdges=65536,maxEdges=8192;
  int64_t maxMemberships=65536,maxIntersectionWork=2000000;
  int64_t maxPatternBytes=INT64_MAX,maxPatternInputs=INT64_MAX,maxPatternOutputs=INT64_MAX;
  std::vector<int64_t> hintPatternIndices;
};
enum class ConstraintSearchStatus { Unavailable,InvalidInput,ModelLimit,Unknown,Infeasible,Feasible,Optimal };
struct ConstraintSearchTelemetry {
  double modelBuildMs=0,solveMs=0,totalMs=0;
  int64_t variables=0,constraints=0,conflicts=0,branches=0,peakRssKiB=-1;
  int64_t objectiveNs=0;
  double bestBoundNs=0,relativeGap=0;
  bool hasObjective=false,budgetExhausted=false,objectiveExactForBackend=false;
};
struct ConstraintSearchResult {
  ConstraintSearchStatus status=ConstraintSearchStatus::Unknown;
  std::string message;
  std::vector<int64_t> selectedPatternIndices;
  int64_t selectedTileIndex=-1;
  ConstraintSearchTelemetry telemetry;
};
bool constraintSearchAvailable();
ConstraintSearchResult solveConstraintSearch(const ConstraintSearchProblem &,const ConstraintSearchOptions & = {});
const char *constraintSearchStatusName(ConstraintSearchStatus);
}
#endif
