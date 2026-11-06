#ifndef ORBIT_TASKFLOW_STATIC_MAPPING_H
#define ORBIT_TASKFLOW_STATIC_MAPPING_H
#include "Taskflow/StructuredProgram.h"
#include "llvm/Support/JSON.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace mlir {
class ModuleOp;
class Operation;
} // namespace mlir
namespace orbit::taskflow {
struct StaticAccess {
  std::string buffer, allocation, version;
  Range m, n;
};
struct StaticStep {
  std::string id, taskId;
  unsigned operationIndex = 0;
  Compute operation;
  std::vector<std::string> dependsOn;
  std::vector<StaticAccess> reads, writes;
  mlir::Operation *body = nullptr; // actual outer affine loop, never serialized
};
struct StaticAnalysis {
  Program program; // reconstructed from actual task body SSA/loops/accesses
  std::vector<StaticStep> steps;
  std::string bodyFingerprint;
};
struct StaticProvenance {
  std::string kind, source;
};
// Ownership is independent of accessibility and transfer connectivity.
struct StaticResourceDomain {
  std::string id, parent, kind;
  // Target-defined physical axes. Generic code validates types/references only.
  std::map<std::string,int64_t> coordinates;
};
struct StaticResource {
  std::string id, kind, device;
  int64_t capacity = 0;
  std::string domain;
  // Minimum reservation implications per unit; concurrent activities contend
  // through the referenced shared resource, independently of identifier names.
  std::map<std::string,int64_t> sharedResources;
};
struct StaticMemory {
  std::string id;
  int64_t capacityBytes = 0;
  std::vector<std::string> accessibleDevices;
  std::string domain, kind;
  // When present, this is the precise allow-list; device access is legacy.
  std::vector<std::string> accessibleResources;
  // Allocation byte quantum; default1 is payload accounting, not physical bank alignment.
  int64_t alignmentBytes = 1;
  // False means physical byte capacity is not modeled by this target. The
  // internal INT64_MAX ceiling is arithmetic accounting, never physical SRAM.
  bool capacityKnown = true;
};
struct StaticTransfer {
  std::string id, source, target, resource, memory, sourceLayout, targetLayout;
  std::string sourceDtype, targetDtype, sourceDevice, targetDevice;
  double latencyMs = 0, bandwidthBytesPerMs = 0;
  int64_t minBytes = 0, maxBytes = INT64_MAX, temporaryBytes = 0;
  StaticProvenance provenance;
  std::string domain;
};
struct StaticCost {
  std::string kind, scope;
  double fixedMs = 0, perElementMs = 0, perFlopMs = 0;
  StaticProvenance provenance;
  std::map<std::string, Range> shapeRange; // inclusive cost-validity bounds
};
struct StaticImplementation {
  std::string id, version, device, memory, layout, dtype, numericalContract;
  std::vector<std::string> operations;
  std::map<std::string, int64_t> resources, multiples;
  int64_t temporaryBytes = 0;
  StaticCost cost;
  bool fullKOnly = false, overwriteZeroAccumulator = false,
       allowAccumulate = true;
};
// Target-owned whole-region cost, never imported as decomposed compute latency.
struct StaticRegionImplementation {
  std::string id, version, pattern, mode, featureAxis, device, boundaryMemory;
  std::string layout, dtype, numericalContract, preparedState, modelId;
  std::map<std::string, int64_t> resources, workspaceBytes, multiples;
  std::map<std::string, Range> shapeRange;
  std::vector<std::string> trainingIds, excludedIds, powerOfTwo;
  StaticProvenance provenance;
  double fixedMs = 0, perMMs = 0, perHMs = 0, perMHMs = 0;
};
struct StaticRegionMatch {
  std::string id, implementationId;
  std::vector<std::string> coveredSteps;
  std::vector<StaticAccess> reads, writes;
  std::map<std::string, int64_t> shape;
};
struct StaticAssumptions {
  std::map<std::string, std::string> initialLocations, outputLocations,
      outputDevices;
  std::vector<std::string> allowedNumericalContracts;
  std::string overlap, programSourceId;
  std::string executionState = "cold_inputs_only";
  int64_t executions = 1, maxAssignments = 256, beamWidth = 16,
          maxRegionCells = 100000;
};
struct StaticConfig {
  std::vector<StaticResourceDomain> domains;
  std::vector<StaticResource> resources;
  std::vector<StaticMemory> memories;
  std::vector<StaticTransfer> links, conversions, synchronizations;
  std::vector<StaticImplementation> implementations;
  std::vector<StaticRegionImplementation> regionImplementations;
  StaticAssumptions assumptions;
  llvm::json::Value original = nullptr; // immutable identity/export only
};
struct StaticEvent {
  std::string id, kind, stepId, implementationId, taskId, device;
  unsigned operationIndex = 0;
  std::string link, conversion, synchronization, costBoundary, sourceDevice,
      targetDevice, memory;
  std::vector<std::string> dependsOn;
  std::vector<std::string> coveredSteps;
  std::string regionPattern, regionMode, modelId, preparedState;
  std::map<std::string, int64_t> regionShape;
  std::map<std::string, int64_t> resourceDemands;
  double durationMs = 0, startMs = 0, endMs = 0;
  int64_t bytes = 0;
  std::vector<StaticAccess> reads, writes;
  StaticProvenance provenance;
};
struct StaticAllocation {
  std::string id, buffer, memory, layout, dtype, createdBy, lifetime;
  Range m, n;
  int64_t bytes = 0;
  bool initial = false;
  std::vector<std::string> lastUsers;
  double startMs = 0, endMs = 0;
};
struct StaticTotals {
  double makespanMs = 0, computeMs = 0, copyMs = 0, convertMs = 0, syncMs = 0,
         regionWallMs = 0;
  std::map<std::string, int64_t> peakMemoryBytes;
  int64_t transferBytes = 0;
};
struct StaticOutput {
  std::string buffer, memory, device, allocation;
  std::vector<std::string> dependsOn;
};
struct StaticPlan {
  std::string identityScheme = "llvm-json-sorted-17g-sha256";
  std::string schema = "orbit.static.plan.v1", programId, candidateId,
              configurationId, planId;
  std::string predictionKind = "static_estimate",
              executionStatus = "not_executed";
  std::string schedulePolicy = "deterministic_topological_list", costBoundary;
  std::string storagePolicy =
                  "whole_buffer_per_location_layout_retained_for_single_graph",
              initialState = "cold_inputs_only";
  std::map<std::string, std::string> assignment;
  std::vector<StaticStep> steps;
  std::vector<StaticEvent> events;
  std::vector<StaticAllocation> allocations;
  std::vector<StaticAccess> outputs;
  std::vector<std::string> outputDependencies;
  std::map<std::string, std::string> outputConsumers;
  int64_t regionCellCount = 0;
  StaticTotals totals;
};
struct StaticAlternative {
  std::string status, reason;
  std::map<std::string, std::string> assignment;
  std::shared_ptr<StaticPlan> plan;
};
struct StaticMappingResult {
  std::string status;
  std::shared_ptr<StaticPlan> bestPlan;
  std::vector<StaticAlternative> alternatives;
  std::vector<std::string> reasons;
  bool exhaustiveAssignments = false, regionExactCover = false;
  uint64_t possibleAssignments = 0, evaluatedAssignments = 0, budget = 0;
};
// These functions validate topology/capacity without requiring calibrated costs.
bool validateMemoryArchitecture(const StaticConfig &, std::string &error);
bool memoryAccessible(const StaticConfig &, llvm::StringRef memory,
                      llvm::StringRef computeResource);
bool findMemoryPath(const StaticConfig &, llvm::StringRef source,
                    llvm::StringRef target, std::vector<std::string> &links,
                    std::string &error);
bool parseMemoryArchitecture(const llvm::json::Value &, StaticConfig &,
                             std::string &error);
llvm::json::Value serializeMemoryArchitecture(const StaticConfig &);
bool analyzeStaticProgram(mlir::ModuleOp, StaticAnalysis &, std::string &error);
bool parseStaticConfig(const llvm::json::Value &, StaticConfig &,
                       std::string &error);
bool parseStaticPlan(const llvm::json::Value &, StaticPlan &,
                     std::string &error);
llvm::json::Value serializeStaticAnalysis(const StaticAnalysis &);
llvm::json::Value serializeStaticPlan(const StaticPlan &,
                                      bool includeIdentity = true);
llvm::json::Value serializeStaticResult(const StaticMappingResult &);
std::string staticFingerprint(const llvm::json::Value &);
std::vector<StaticAccess> staticReads(const Compute &);
std::vector<StaticAccess> staticWrites(const Compute &);
bool staticImplementationSupports(const StaticAnalysis &, const StaticStep &,
                                  const StaticImplementation &,
                                  const StaticConfig &);
double staticImplementationCost(const Compute &, const StaticImplementation &);
bool buildStaticGraph(const StaticAnalysis &, const StaticConfig &,
                      const std::map<std::string, std::string> &, StaticPlan &,
                      std::string &error);
bool matchStaticRegions(const StaticAnalysis &, const StaticConfig &,
                        std::vector<StaticRegionMatch> &,
                        std::vector<std::string> &reasons);
double staticRegionCost(const StaticRegionMatch &,
                        const StaticRegionImplementation &);
bool buildStaticRegionGraph(const StaticAnalysis &, const StaticConfig &,
                            const std::vector<StaticRegionMatch> &,
                            StaticPlan &, std::string &error);
StaticMappingResult mapStaticRegions(const StaticAnalysis &,
                                     const StaticConfig &);
bool checkStaticRegionPlan(const StaticAnalysis &, const StaticConfig &,
                           const StaticPlan &,
                           std::vector<std::string> &errors);
bool scheduleStaticPlan(StaticPlan &, const StaticConfig &, std::string &error);
bool checkStaticPlan(const StaticAnalysis &, const StaticConfig &,
                     const StaticPlan &, std::vector<std::string> &errors);
StaticMappingResult mapStaticProgram(const StaticAnalysis &,
                                     const StaticConfig &);
} // namespace orbit::taskflow
#endif
