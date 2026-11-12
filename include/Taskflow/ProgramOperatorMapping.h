#ifndef ORBIT_TASKFLOW_PROGRAM_OPERATOR_MAPPING_H
#define ORBIT_TASKFLOW_PROGRAM_OPERATOR_MAPPING_H
#include "Taskflow/ProgramOperatorSource.h"
#include "Taskflow/ProgramSearch.h"
namespace orbit::taskflow {
// Logical source instances stay unique even when a cover uses many cores or
// partitions work across replicas. The invocation variant owns its physical
// reservation once; charging it per covered source operator would double count.
// Backend registrations describe executable variants, rather than granting
// arbitrary shape/replica changes to an opaque implementation. Reservation is
// for the complete inclusive call: launch and transport must not be added
// again.
struct ProgramOperatorPackedFragment {
  std::string allocationId, memory, dtype;
  int64_t bytes = 0;
  std::vector<int64_t> sourceOffsets, sourceExtents;
  std::vector<int64_t> physicalShape, physicalOffsets, strides;
};
struct ProgramOperatorPackedInput {
  size_t sourceValue = 0;
  std::string packingIdentity;
  std::vector<ProgramOperatorPackedFragment> fragments;
};
struct ProgramOperatorWorkPartition {
  size_t outputValue = 0;
  int64_t replica = 0;
  std::vector<int64_t> offsets, extents;
};
struct ProgramOperatorExecutionVariant {
  std::string implementationId, artifactIdentity, entrypoint;
  // These identities are checked by the backend registration/replay bridge.
  // Common mapping carries them in candidate identity; it does not infer
  // compatible device contexts from identical operator names.
  std::string runtimeContextIdentity, dataflow, residencePolicy;
  std::map<std::string, int64_t> implementationParameters;
  std::map<std::string, int64_t> resourceDemands;
  std::vector<int64_t> resourceShape;
  std::vector<ProgramOperatorPackedInput> packedInputs;
  std::vector<ProgramOperatorWorkPartition> workPartitions;
  std::vector<std::string> unprovedConstraints;
  int64_t replicas = 1;
  std::string workDistribution = "single-invocation";
  std::map<std::string, int64_t> temporaryBytes;
  // Prepared immutable allocations are shared by identity, not per call.
  std::map<std::string, std::pair<std::string, int64_t>> residentAllocations;
};
struct ProgramOperatorMapping {
  std::string registrationIdentity, hostMemory;
  std::vector<ProgramOperatorExecutionVariant> variants; // event indexed
  std::vector<size_t> readyPriority; // permutation; never overrides dependence
};
struct ProgramOperatorMappingCandidate {
  ProgramOperatorSelection selection;
  ProgramOperatorMapping mapping;
};
enum class ProgramOperatorMappingActionKind {
  ReplaceExecutableCover,
  ReadyPriority,
  TransferRoute,
  IndependentReplica,
  IndependentResourceShape,
  IndependentResidence
};
struct ProgramOperatorMappingAction {
  ProgramOperatorMappingActionKind kind =
      ProgramOperatorMappingActionKind::ReadyPriority;
  size_t firstRegion = 0, regionCount = 0;
  std::vector<ProgramOperatorImplementation> replacements;
  std::vector<ProgramOperatorExecutionVariant> replacementVariants;
  std::vector<size_t> readyPriority;
};
// Structural replacement preserves the exact source-operator coverage. Only
// registered executable implementations grant fusion, fission or resource
// changes. The result still requires full physical materialization/checking.
bool applyProgramOperatorMappingAction(const ProgramOperatorMappingCandidate &,
                                       const ProgramOperatorMappingAction &,
                                       ProgramOperatorMappingCandidate &,
                                       std::string &error);
ProgramSearchEvaluation evaluateProgramOperatorMappingCandidate(
    std::shared_ptr<const ProgramOperatorSource>,
    const ProgramOperatorMappingCandidate &, const StaticConfig &,
    llvm::StringRef hostMemory, ProgramGraph &materialized);
bool parseProgramOperatorMappingRegistrations(
    const ProgramOperatorSource &, const llvm::json::Value &catalogue,
    const std::vector<ProgramOperatorImplementation> &,
    std::string &registrationIdentity,
    std::map<std::string, ProgramOperatorExecutionVariant> &,
    std::string &error);
// Immutable, source/ABI-certified schedule projection. It caches derived
// boundary-allocation identities, not whole-plan prices. Every assignment is
// still evaluated for joint resource/resident/scratch feasibility. The selected
// candidate must be fully materialized and checked before executable replay.
struct ProgramOperatorMappingDomain;
bool prepareProgramOperatorMappingDomain(
    const ProgramOperatorSource &,
    const std::vector<ProgramOperatorImplementation> &,
    const std::vector<ProgramOperatorChoiceGroup> &,
    const std::map<std::string, ProgramOperatorExecutionVariant> &,
    llvm::StringRef registrationIdentity, const StaticConfig &,
    llvm::StringRef hostMemory,
    std::shared_ptr<const ProgramOperatorMappingDomain> &, std::string &error);
ProgramSearchEvaluation
evaluateProgramOperatorMappingDomain(const ProgramOperatorMappingDomain &,
                                     const std::vector<size_t> &groupChoices);
// Deliberately retained slow invocation-by-invocation evaluator for
// differential tests and auditing captured assignments after evaluator
// optimizations.
ProgramSearchEvaluation evaluateProgramOperatorMappingDomainReference(
    const ProgramOperatorMappingDomain &,
    const std::vector<size_t> &groupChoices);
// Adds timing only after checking the source-bound physical graph. Scheduling
// uses its real data/control/implementation dependencies and target capacities.
// Unsupported transfers and unknown inclusive timings remain explicit errors.
bool scheduleProgramOperatorGraph(ProgramGraph &,
                                  const ProgramOperatorMapping &,
                                  std::string &error);
bool checkProgramOperatorMapping(const ProgramGraph &, std::string &error);
bool hasProgramOperatorPackedInputs(const ProgramOperatorMapping &);
bool checkProgramOperatorPackedGraph(const ProgramGraph &, std::string &error);
llvm::json::Value
serializeProgramOperatorMapping(const ProgramOperatorMapping &);
} // namespace orbit::taskflow
#endif
