#ifndef ORBIT_TASKFLOW_PROGRAM_MAPPING_H
#define ORBIT_TASKFLOW_PROGRAM_MAPPING_H
#include "Taskflow/StaticMapping.h"
#include "Taskflow/ComputeEstimate.h"
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
};
struct MappingVersion {
  size_t region = 0; // interned exact logical cell, arbitrary rank
  int64_t producer = -1; // event, -1 is a declared initial input
};
struct MappingCopy {
  size_t version = 0, allocation = 0;
  int64_t createdBy = -1;
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
};
struct MappingTaskPlacement {
  size_t task = 0;
  std::string resource, memory, implementation;
};
struct ProgramDataOrganizationChoice {
  ProgramDataPolicy policy = ProgramDataPolicy::PhaseLocal;
  std::vector<MappingTaskPlacement> placements;
  std::vector<MappingComputeEstimate> computeEstimates;
  std::optional<double> clockMHz;
};
// A call boundary is shared by all region inputs. Device stages and data
// transfers remain distinct events; submission cost belongs to this boundary.
struct MappingExecutionRegion {
  std::string id, placement, submissionResource;
  size_t beginEvent = 0, endEvent = 0; // half-open event coverage
  std::vector<size_t> deviceStages;
  int64_t submissionCount = 1;
};
struct ProgramCommunicationPlan {
  std::vector<MappingExecutionRegion> executionRegions;
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
  std::optional<double> estimatedMakespanMs;
  int64_t transferBytes = 0;
  std::map<std::string,int64_t> peakMemoryBytes;
};
struct GenericAnalysis;
struct GenericMappingPlan;
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
