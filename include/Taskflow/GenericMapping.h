#ifndef ORBIT_TASKFLOW_GENERIC_MAPPING_H
#define ORBIT_TASKFLOW_GENERIC_MAPPING_H
#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/StaticMapping.h"
#include "Taskflow/ProgramMapping.h"
namespace orbit::taskflow {
// Search choices are separate from source access/dependence facts and from
// the AIR resource bindings below. Indices refer to analyzed source objects.
struct ProgramMappingChoice {
  // Empty realization/grouping are accepted only as syntactic adapter-auto
  // inputs. Every checked plan stores their resolved canonical values.
  std::string residency;
  int64_t storage = -1, reuseLoop = -1, panelLoop = -1;
  std::string grouping;
  // Public copy lifetime choice; empty is legacy input only.
  std::string dataOrganization;
};
std::vector<ProgramMappingChoice>
discoverGenericMappingChoices(const GenericAnalysis &);
struct GenericTaskImplementation {
  size_t task = 0;
  std::string kind = "scalar";
  int64_t parallelLoop = -1, lanes = 1;
};
struct GenericReadOnlyRetention {
  size_t beginPhase = 0, endPhase = 0, storage = 0;
  std::vector<int64_t> offsets, sizes;
};
struct ProgramExecutionGroup {
  size_t beginPhase = 0, endPhase = 0;
  std::string resource, memory;
  std::vector<size_t> dependencies;
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
struct GenericMappingPlan {
  std::string strategy = "single-herd-resident", bodyFingerprint, planId;
  int64_t capacityBytes = 24576, residentBytes = 0;
  std::vector<size_t> orderedTasks;
  std::vector<GenericTransfer> transfers;
  std::vector<GenericTilePhase> phases;
  std::vector<GenericTileRetention> retainedRegions;
  std::vector<size_t> fusedInitializerTasks;
  std::vector<GenericHardwareStage> hardwareStages;
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
  bool hasCost = false;
  double estimatedMs = 0;
  double theoreticalLaunchMs = 0, theoreticalPerOperationMs = 0,
         theoreticalPerByteMs = 0;
  std::string costProvenance = "unknown: direct AIR path not calibrated";
};
bool planGenericMapping(const GenericAnalysis &,
                        const llvm::json::Value &config, GenericMappingPlan &,
                        std::string &error);
bool checkGenericMapping(const GenericAnalysis &, const GenericMappingPlan &,
                         std::string &error);
llvm::json::Value serializeGenericMapping(const GenericAnalysis &,
                                          const GenericMappingPlan &);
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
