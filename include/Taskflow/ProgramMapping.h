#ifndef ORBIT_TASKFLOW_PROGRAM_MAPPING_H
#define ORBIT_TASKFLOW_PROGRAM_MAPPING_H
#include "Taskflow/StaticMapping.h"
#include "Taskflow/ComputeEstimate.h"
#include "Taskflow/GenericAnalysis.h"
#include <optional>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace orbit::taskflow {
enum class ProgramDataPolicy { PhaseLocal, RetainAcrossUsers };
llvm::StringRef programDataPolicyName(ProgramDataPolicy);
bool parseProgramDataPolicy(llvm::StringRef, ProgramDataPolicy &,
                            std::string &error);
// The native adapter reuses Static's target contracts. Unlike StaticAccess,
// these regions have arbitrary rank and are derived from real MLIR accesses.
struct MappingRegion {
  size_t storage = 0;
  std::vector<int64_t> offsets, sizes;
};
struct MappingAllocation {
  size_t storage = 0;
  std::string memory, key;
  int64_t bytes = 0;
  size_t beginPhase = 0, endPhase = 0;
  // Empty preserves legacy adapter addressing and canonical identities.
  // region-addressed uses logical offsets inside a complete storage allocation;
  // reusable-slot aliases all assigned regions at the same physical address.
  std::string addressing, spatialOwner;
};
struct ProgramExecutionGroup {
  size_t beginPhase = 0, endPhase = 0;
  std::string resource, memory;
  std::vector<size_t> dependencies;
  std::vector<size_t> memberEvents;
  std::string spatialOwner;
  // Source/control dependencies and target implementation serialization are
  // separate decisions. Both constrain execution; only dependencies are semantic.
  std::vector<size_t> implementationDependencies;
};
struct MappingVersion {
  size_t region = 0; // interned exact logical cell, arbitrary rank
  int64_t producer = -1; // event, -1 is a declared initial input
};
struct MappingCopy {
  size_t version = 0, allocation = 0;
  int64_t createdBy = -1;
  // Half-open validity in the graph's topological event witness. These ordinals
  // do not identify timed release activities or impose global completion order.
  size_t availableFrom = 0, releasedAfter = 0;
  std::vector<size_t> lastUsers;
  std::vector<size_t> users, consumerEdges;
  std::string layout = "row_major";
};
struct MappingConsumptionRequirement {
  std::string memory, layout = "row_major", dtype;
};
struct MappingDataEdge {
  int64_t producer = -1;
  size_t consumer = 0, region = 0, version = 0, copy = 0;
  MappingConsumptionRequirement requirement;
};
struct MappingOrderEdge {
  size_t producer = 0, consumer = 0;
  std::string reason;
};
struct MappingEvent {
  std::string kind; // compute, initialize, transfer
  size_t phase = 0, task = 0;
  std::string resource, link;
  std::vector<size_t> dependencies;
  std::vector<size_t> reads, writes; // location-copy indices
  std::vector<size_t> dataEdges, orderEdges;
  std::string memory;
  // Only copy events have a transfer region. Endpoints and versions are the
  // referenced read/write copies; this is the exact rectangular DMA payload.
  std::optional<MappingRegion> transferRegion;
  int64_t transferBytes = 0;
};
struct MappingTaskPlacement {
  size_t task = 0;
  std::string resource, memory, implementation;
  int64_t resourceCount = 1;
  std::vector<int64_t> resourceShape;
};
struct ProgramDataOrganizationChoice {
  ProgramDataPolicy policy = ProgramDataPolicy::PhaseLocal;
  std::vector<MappingTaskPlacement> placements;
  std::vector<MappingComputeEstimate> computeEstimates;
  std::optional<double> clockMHz;
  // Opt-in exact rectangular access partition; default preserves full-storage adapter.
  bool partitionExactRegions = false;
  // Keep source/data dependencies only; scheduling priority is chosen separately.
  // The default preserves historical whole-task serial plans.
  bool semanticDependenciesOnly = false;
};
// A call boundary is shared by all region inputs. Device stages and data
// transfers remain distinct events; submission cost belongs to this boundary.
struct MappingExecutionRegion {
  std::string id, placement, submissionResource;
  size_t beginEvent = 0, endEvent = 0; // half-open event coverage
  std::vector<size_t> deviceStages;
  int64_t submissionCount = 1;
  // Optional whole-program host-call refinement. The endpoint is accessible
  // to the invocation adapter, not implicitly to a device compute core.
  std::vector<size_t> sourceTasks, memberEvents, inputCopies, outputCopies, predecessorRegions;
  std::string implementation, endpointMemory;
  // Explicit invocation ABI policy, independent of target/device names.
  // Empty is accepted only as the legacy cpu/npu adapter spelling.
  std::string invocationLifecycle; // host-only | host-device
};
struct MappingRegionEvent {
  std::string kind; // submit, complete, visible-to-device, visible-to-host
  size_t region = 0;
  std::string resource;
  std::vector<size_t> dependencies, copies;
};
struct MappingIterationDomain {
  size_t loop = 0;
  int64_t lower = 0, upper = 0, step = 1;
};
struct MappingTaskInstance {
  size_t task = 0, event = 0, phase = 0, domainTask = 0;
  // A distributed initializer applies its actual body to the written regions
  // of a controlling tile, rather than reexecuting the full source loop nest.
  std::string kind = "source-task";
  std::vector<MappingIterationDomain> domain;
  std::string implementation = "scalar";
  int64_t lanes = 1, parallelLoop = -1;
  int64_t resourceCount = 1;
  std::vector<int64_t> resourceShape;
};
struct MappingScheduledEvent {
  size_t event = 0;
  std::vector<size_t> predecessors;
  std::optional<double> startMs, endMs;
};
struct MappingGraphSchedule {
  // A conservative serial realization of the dependency/resource partial
  // order. Unknown durations never become zero-duration measured work.
  std::string policy = "serial-event-order";
  std::vector<MappingScheduledEvent> events;
  std::optional<double> estimatedMakespanMs;
  std::vector<std::string> uncalibrated;
};
// Canonical public program/execution graph. The borrowed MLIR handles remain
// valid while its source ModuleOp lives. GenericAnalysis is the existing body,
// logical-storage, access and source-dependency table, not a JSON graph import.
// Native phase/stage fields are an adapter realization, checked against these
// common identities. The compatibility name below avoids a parallel ledger.
struct ProgramGraph {
  std::string targetIdentity;
  GenericAnalysis source;
  std::vector<MappingTaskInstance> instances;
  MappingGraphSchedule schedule;
  std::string allocationLifetime = "copy-events";
  std::map<size_t,std::string> inputLocations, outputLocations;
  std::vector<MappingExecutionRegion> executionRegions;
  std::vector<MappingRegionEvent> regionEvents;
  std::vector<ProgramExecutionGroup> executionGroups;
  // Capacities/topology are target facts. Copy placement/lifetimes and event
  // order are checked choices. No uncalibrated wall-clock schedule is invented.
  std::vector<StaticResourceDomain> domains;
  std::vector<StaticMemory> memories;
  std::vector<StaticResource> resources;
  std::vector<StaticTransfer> links;
  std::vector<MappingRegion> regions;
  std::vector<MappingAllocation> allocations;
  std::vector<MappingVersion> versions;
  std::vector<MappingCopy> copies;
  std::vector<MappingEvent> events;
  ProgramDataPolicy dataPolicy = ProgramDataPolicy::PhaseLocal;
  std::string bodyFingerprint;
  std::vector<MappingDataEdge> dataEdges;
  std::vector<MappingOrderEdge> orderEdges;
  std::vector<MappingTaskPlacement> placements;
  std::vector<MappingComputeEstimate> computeEstimates;
  std::optional<double> clockMHz;
  std::optional<double> estimatedMakespanMs;
  int64_t transferBytes = 0;
  std::map<std::string,int64_t> peakMemoryBytes;
};
using ProgramCommunicationPlan = ProgramGraph;
struct GenericAnalysis;
struct GenericMappingPlan;
// Builders call this once after constructing source-derived copies/events.
// With native=nullptr this is the existing finite whole-task adapter.
bool finalizeProgramGraph(const GenericAnalysis &, ProgramGraph &,
                          const GenericMappingPlan *native, std::string &error);
// Independent common validation; adapter checkers additionally prove their
// supported source-region and backend realization correspondence.
bool checkProgramGraph(const ProgramGraph &, std::string &error);
bool checkProgramGraphSource(const GenericAnalysis &, const ProgramGraph &,
                             std::string &error);
// Physical allocation reuse constraints, separate from logical consumption.
struct ProgramBufferOrdering {
  size_t allocation = 0, earlierCopy = 0, laterCopy = 0;
  size_t predecessor = 0, successor = 0;
  std::string reason;
};
bool programAllocationIsSlot(const ProgramGraph &, size_t allocation);
bool collectProgramBufferOrdering(const ProgramGraph &,
    std::vector<ProgramBufferOrdering> &, std::string &error);
llvm::json::Value serializeProgramGraph(const ProgramGraph &);
std::string programGraphIdentity(const ProgramGraph &);
std::string renderProgramGraphDOT(const ProgramGraph &);
llvm::json::Value summarizeProgramGraphCost(const ProgramGraph &);
bool parseWholeTaskProgramGraphConfig(const llvm::json::Value &, StaticConfig &,
    ProgramDataOrganizationChoice &, std::string &error);
bool buildProgramCommunication(const GenericAnalysis &, GenericMappingPlan &,
                               std::string &error);
bool checkProgramCommunication(const GenericAnalysis &,
                               const GenericMappingPlan &, std::string &error);
llvm::json::Value serializeProgramCommunication(const GenericMappingPlan &);
llvm::json::Value serializeProgramCommunication(const ProgramCommunicationPlan &);
bool planProgramDataOrganization(const GenericAnalysis &, const StaticConfig &,
    const ProgramDataOrganizationChoice &, ProgramCommunicationPlan &,
    std::string &error);
bool checkProgramDataOrganization(const GenericAnalysis &, const StaticConfig &,
    const ProgramDataOrganizationChoice &, const ProgramCommunicationPlan &,
    std::string &error);
} // namespace orbit::taskflow
#endif
