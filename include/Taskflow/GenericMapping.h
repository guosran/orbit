#ifndef ORBIT_TASKFLOW_GENERIC_MAPPING_H
#define ORBIT_TASKFLOW_GENERIC_MAPPING_H
#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/StaticMapping.h"
#include "Taskflow/ProgramMapping.h"
namespace orbit::taskflow {
// Search choices are separate from source access/dependence facts and from
// the AIR resource bindings below. Indices refer to analyzed source objects.
// Retained storages may be uniquely produced data, or immutable external input
// shared by at least two member readers of the exact same tile rectangle.
struct ProgramResidencyGroup {
  std::vector<size_t> tasks, retainedStorages;
  bool operator==(const ProgramResidencyGroup &other) const {
    return tasks==other.tasks && retainedStorages==other.retainedStorages;
  }
  bool operator!=(const ProgramResidencyGroup &other) const {return !(*this==other);}
};
struct ProgramMappingChoice {
  // Empty realization/grouping are accepted only as syntactic adapter-auto
  // inputs. Every checked plan stores their resolved canonical values.
  // phase-local is the legacy adapter-auto choice; explicit realizations are
  // tiled-phase-local, readonly-panel, and whole-program. The latter retains
  // graph intermediates/shared input copies in one capacity-checked program.
  std::string residency;
  int64_t storage = -1, reuseLoop = -1, panelLoop = -1;
  std::string grouping;
  // Public copy lifetime choice; empty is legacy input only.
  std::string dataOrganization;
  // tile-major executes corresponding pointwise instances of this closed task
  // group before advancing the spatial tile. Retained logical storages have
  // one shared local copy and no materialized host boundary inside the group.
  // shape-major reorders independent tiles only within each source task.
  // Empty vectors preserve the legacy ungrouped source action.
  std::vector<size_t> taskGroup, retainedStorages;
  // Scoped decisions. Legacy fields above denote one group when this is empty.
  // A logical storage may also have consumers outside its local residency scope.
  std::vector<ProgramResidencyGroup> residencyGroups;
};
std::vector<ProgramMappingChoice>
discoverGenericMappingChoices(const GenericAnalysis &,size_t maxResidencyProposals=8,
                              size_t maxResidencyGroups=4);
struct GenericTaskImplementation {
  size_t task = 0;
  std::string kind = "scalar";
  int64_t parallelLoop = -1, lanes = 1;
};
struct GenericReadOnlyRetention {
  size_t beginPhase = 0, endPhase = 0, storage = 0;
  std::vector<int64_t> offsets, sizes;
};
struct GenericLoopBounds {
  size_t loop = 0;
  int64_t lower = 0, upper = 0;
};
struct GenericTileRegion {
  size_t storage = 0;
  std::vector<int64_t> offsets, sizes;
  bool read = false, write = false;
};
struct GenericTilePhase {
  size_t task = 0;
  std::vector<GenericLoopBounds> loops;
  std::vector<size_t> dependencies;
  std::vector<GenericTileRegion> regions;
  int64_t computationEvent = -1;
};
struct GenericTileRetention {
  size_t beginPhase = 0, endPhase = 0, storage = 0;
  int64_t initializerTask = -1;
};
struct GenericResidencyScope {
  size_t beginPhase=0, endPhase=0;
  std::vector<size_t> tasks, retainedStorages, boundaryExports;
  // Selected implementation lifetime contract, not a semantic source edge.
  std::string inputReleasePolicy="group-tail"; // group-tail | last-reader
};
struct GenericTransfer {
  size_t storage = 0;
  std::string direction; // host_to_l1 or l1_to_host; explicit BO-to-local DMA
  int64_t bytes = 0;
  size_t phase = 0;
  std::vector<int64_t> offsets, sizes;
  int64_t event = -1;
  std::vector<size_t> sourceCopies, targetCopies;
};
struct GenericHardwareStage {
  size_t beginPhase = 0, endPhase = 0;
  bool legalizeChannels = false;
  size_t groupIndex = 0;
};
struct GenericAIRChannelAllocation {
  size_t stage = 0, storage = 0;
  std::string direction;
  int64_t tileChannel = 0;
  std::string logicalFifo;
  int64_t transferCount = 0, bytes = 0;
};
// Optional direct-AIR cost facts and provider. These do not alter compute
// semantics or the existing CGRA model. Operation keys are "name:dtype".
struct GenericCostFeatures {
  // Hash of the complete checked execution plan with all price metadata removed.
  std::string executionFingerprint;
  // Source SSA/access family with static extents/names erased, not a price key.
  std::string regionFamilyFingerprint, architectureFingerprint;
  std::map<std::string,int64_t> values, operations;
  std::vector<std::string> sourceFingerprints, implementations;
};
struct GenericDirectAIRCost {
  bool enabled = false;
  GenericCostFeatures features;
  llvm::json::Value model = nullptr, target = nullptr, eventTiming = nullptr;
  std::string status = "disabled", reason, modelDigest, featureDigest;
  std::optional<double> valueMs;
};
// AIR-specific realization of a common target-owned architecture. Direction
// strings in GenericTransfer remain backend channel roles, never locations.
struct GenericMemoryBinding {
  GenericMemoryBinding();
  StaticConfig architecture;
  std::string hostMemory = "system_memory_bo", localMemory = "aie_l1";
  std::string computeResource = "npu.compute", dmaResource = "npu.dma";
  std::string inputLink = "host_to_l1", outputLink = "l1_to_host";
  std::string submissionResource = "host-submission";
};
struct GenericTargetCapabilities {
  int64_t capacityBytes = 24576;
  int64_t maxGroupInputs=2,maxGroupOutputs=2;
  bool acquireAllInputsAtGroupEntry=true;
  std::vector<int64_t> tileAlignment{1,8,1};
  std::vector<GenericTaskImplementation> implementations{{0,"scalar",-1,1},{0,"scalar-product-vector-add-f32",-1,8}};
  std::string identity;
};
bool configureGenericMemory(const llvm::json::Value &, int64_t capacity,
                            GenericMemoryBinding &, std::string &error);
bool checkGenericMemory(const GenericMemoryBinding &, int64_t capacity,
                        std::string &error);
bool getGenericTargetCapabilities(const llvm::json::Value &,
    GenericTargetCapabilities &, std::string &error);
llvm::json::Value serializeGenericMemory(const GenericMemoryBinding &);
llvm::json::Value serializeGenericCapabilities(const GenericTargetCapabilities &);
bool checkGenericCapabilities(const GenericMemoryBinding &, const GenericTargetCapabilities &, std::string &error);
struct GenericMappingPlan {
  GenericMemoryBinding memoryBinding;
  GenericTargetCapabilities targetCapabilities;
  std::string strategy = "single-herd-resident", bodyFingerprint, planId;
  int64_t capacityBytes = 24576, residentBytes = 0;
  std::vector<size_t> orderedTasks;
  std::vector<GenericTransfer> transfers;
  std::vector<GenericTilePhase> phases;
  std::vector<GenericResidencyScope> residencyScopes;
  std::vector<GenericTileRetention> retainedRegions;
  std::vector<size_t> fusedInitializerTasks;
  std::vector<GenericHardwareStage> hardwareStages;
  bool moduleWideChannelConversion = false;
  std::vector<GenericAIRChannelAllocation> channelAllocations;
  ProgramMappingChoice choices;
  std::vector<std::vector<size_t>> taskLoopOrders;
  std::vector<GenericTaskImplementation> taskImplementations;
  std::vector<GenericReadOnlyRetention> readOnlyRetentions;
  std::vector<ProgramExecutionGroup> executionGroups;
  ProgramCommunicationPlan communication;
  std::string channelMergePolicy =
      "single-logical-fifo-per-stage-storage-direction";
  // Requested maximum iterations per loop; empty means automatic sizing.
  std::vector<int64_t> tileSizes;
  GenericDirectAIRCost directCost;
  bool hasCost = false;
  double estimatedMs = 0;
  double theoreticalLaunchMs = 0, theoreticalPerOperationMs = 0,
         theoreticalPerByteMs = 0;
  std::string costProvenance = "unknown: direct AIR path not calibrated";
};
// Resolved scope queries are shared with lowering; no backend placement inference.
const GenericResidencyScope *genericResidencyScope(const GenericMappingPlan &,size_t phase);
bool genericLocallyRetained(const GenericMappingPlan &,size_t phase,size_t storage);
std::vector<ProgramResidencyGroup> genericResidencyGroups(const ProgramMappingChoice &);
// Monotonic process/thread-local diagnostics only. They neither select plans
// nor alter candidate/cache identities. Snapshot deltas count actual entries,
// including the planner's internal final checker invocation.
struct GenericMappingStatistics { uint64_t plannerCalls=0, checkerCalls=0, costProviderCalls=0; };
GenericMappingStatistics genericMappingStatistics();
// Shared by the two native complete-cost providers; no nested double charge.
void recordGenericMappingCostProviderCall();
bool planGenericMapping(const GenericAnalysis &,
                        const llvm::json::Value &config, GenericMappingPlan &,
                        std::string &error);
bool checkGenericMapping(const GenericAnalysis &, const GenericMappingPlan &,
                         std::string &error);
llvm::json::Value serializeGenericMapping(const GenericAnalysis &,
                                          const GenericMappingPlan &);
// Native facts only; no fitting and no target queries. Full-call composite
// estimates must never be added to separately charged compute/link estimates.
bool collectGenericCostFeatures(const GenericAnalysis &, const GenericMappingPlan &,
                                GenericCostFeatures &, std::string &error);
llvm::json::Value serializeGenericCostFeatures(const GenericCostFeatures &);
bool evaluateGenericDirectAIRCost(const GenericCostFeatures &,
    const llvm::json::Value &model, const llvm::json::Value &target,
    GenericDirectAIRCost &, std::string &error);
// Native adapters for the shared version/copy model; no operator recognition.
bool deriveGenericPhaseRegions(const GenericAnalysis &, GenericTilePhase &,
                               int64_t &bytes, std::string &error);
void collectGenericAllocations(const GenericAnalysis &, const GenericMappingPlan &,
                               std::vector<MappingAllocation> &);
// Compatibility importer only: promote explicit source interface metadata to
// native function arguments. Never used by native MLIR analysis/code
// generation.
struct Program;
std::string emitNativeTaskflow(const Program &, std::string &error);
bool importNativeInterface(mlir::ModuleOp, const Program &, std::string &error);
llvm::json::Value evaluateGenericMLIR(llvm::StringRef,
                                      const llvm::json::Value &,
                                      std::string &error);
} // namespace orbit::taskflow
#endif
