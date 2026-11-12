#ifndef ORBIT_TASKFLOW_PROGRAM_REGION_COMPOSITE_COST_H
#define ORBIT_TASKFLOW_PROGRAM_REGION_COMPOSITE_COST_H
#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/ProgramInvocationCost.h"
namespace orbit::taskflow {
// Evidence about the compiled implementation, never a lookup key for its price.
// sourceFingerprint must match the separately materialized region source.
struct ProgramRegionImplementation {
  size_t region = 0;
  std::string implementation, sourceFingerprint, codeIdentity;
  std::string controlSequenceIdentity, controlFamily, provenance;
};
struct ProgramRegionCompositeFeatures {
  size_t region = 0;
  std::string sourceFamily, sourceFingerprint;
  ProgramRegionImplementation implementation;
  std::vector<size_t> events, regionEvents, sourceTasks;
  std::map<std::string, int64_t> resourceDemands;
  std::map<std::string, double> quantities;
  bool sourceWorkKnown = true;
  std::string unknownReason;
};
struct ProgramRegionCompositeFeatureSet {
  std::string graphId, identity;
  ProgramInvocationFeatures invocation;
  std::vector<ProgramRegionCompositeFeatures> regions;
};
struct ProgramRegionCompositeRange {
  double minimum = 0, maximum = 0;
};
struct ProgramRegionCompositeCalibration {
  std::string implementation, sourceFamily, controlFamily, provenance;
  // All eight features require explicit coefficients and calibrated ranges.
  // A consciously unused term may be architecture zero with provenance.
  std::map<std::string, ProgramMechanismParameter> coefficients;
  std::map<std::string, ProgramRegionCompositeRange> domain;
};
struct ProgramRegionCompositeModel {
  std::string id, mode = "strict", boundary, calibrationRuntimeIdentity;
  ProgramInvocationRuntimeContract runtime;
  std::vector<ProgramRegionCompositeCalibration> regions;
  // Global application reset/output-copy and host orchestration parameters;
  // control.per_call (ns) and control.per_region (ns/count) are both required.
  // Region call costs
  // already include launch, sync, setup, body, DMA, teardown and completion.
  std::map<std::string, ProgramMechanismParameter> parameters;
};
bool parseProgramRegionCompositeModel(const llvm::json::Value &,
                                      ProgramRegionCompositeModel &,
                                      std::string &error);
bool collectProgramRegionCompositeFeatures(
    const HeterogeneousMappingPlan &, const ProgramInvocationRuntimeContract &,
    const std::vector<ProgramRegionImplementation> &,
    ProgramRegionCompositeFeatureSet &, std::string &error);
llvm::json::Value serializeProgramRegionCompositeFeatures(
    const ProgramRegionCompositeFeatureSet &);
// Each graph body/lifecycle atom has exactly one region composite owner. No
// v28 body, setup, teardown, visibility or completion fees may be appended.
bool buildProgramRegionCompositeCosts(const HeterogeneousMappingPlan &,
                                      const ProgramRegionCompositeFeatureSet &,
                                      const ProgramRegionCompositeModel &,
                                      ProgramTimingRequest &,
                                      llvm::json::Value &receipt,
                                      std::string &error);
bool evaluateProgramRegionCompositeModel(
    const HeterogeneousMappingPlan &, const ProgramRegionCompositeModel &,
    const std::vector<ProgramRegionImplementation> &, ProgramTimingEvaluation &,
    llvm::json::Value &receipt, std::string &error);
} // namespace orbit::taskflow
#endif
