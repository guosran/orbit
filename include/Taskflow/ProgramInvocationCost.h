#ifndef ORBIT_TASKFLOW_PROGRAM_INVOCATION_COST_H
#define ORBIT_TASKFLOW_PROGRAM_INVOCATION_COST_H
#include "Taskflow/ProgramMechanismCost.h"
namespace orbit::taskflow {
// Prepared calls: ELF/context/BO allocation are outside this boundary. Device
// runtime-sequence setup/teardown remain inside every invocation. Unbound
// calls copy each staged ABI array into its BO; bound calls access mapped BOs.
// resetInputs is a distinct application input reset in scope, never an alias
// for the per-invocation unbound BO staging copy. It defaults to false.
struct ProgramInvocationRuntimeContract {
  std::string identity, hostWorkerResource;
  std::string state = "prepared-hot";
  bool boundBuffers = true, stageOutputsToDevice = true;
  bool resetInputs = false, copyFinalOutputs = true;
};
struct ProgramInvocationRegionFeatures {
  size_t region = 0, submit = SIZE_MAX, complete = SIZE_MAX;
  size_t visibleToDevice = SIZE_MAX, visibleToHost = SIZE_MAX;
  bool device = false;
  int64_t submissions = 0;
  std::vector<size_t> inputAllocations, outputAllocations, abiAllocations,
      scratchAllocations;
  std::vector<size_t> toDeviceAllocations;
  int64_t abiBytes = 0, toDeviceBytes = 0, fromDeviceBytes = 0,
          stagingBytes = 0;
};
struct ProgramInvocationFeatures {
  std::string graphId, identity;
  ProgramInvocationRuntimeContract runtime;
  std::vector<ProgramInvocationRegionFeatures> regions;
  std::vector<size_t> initialInputAllocations, finalOutputAllocations;
  int64_t initialInputBytes = 0, finalOutputBytes = 0;
};
bool collectProgramInvocationFeatures(const ProgramGraph &,
                                      const ProgramInvocationRuntimeContract &,
                                      ProgramInvocationFeatures &,
                                      std::string &error);
// Returns lifecycle-only coverage, plus physical device setup/teardown and
// ordered runtime-only application
// reset/final-copy activities. Concatenate with body costs built WITHOUT
// lifecycle or inferred logical-group configuration costs, then call the
// existing evaluateProgramTiming. The caller must order each region body after
// invocation.device_setup.R, its teardown after all body costs, and complete
// after invocation.device_teardown.R. Duplicate coverage is rejected there.
// modelRuntimeIdentity is explicit calibration-domain metadata and must match
// the collected runtime contract. Missing required coefficients stay unknown.
bool buildProgramInvocationCosts(const ProgramGraph &,
                                 const ProgramInvocationFeatures &,
                                 const ProgramMechanismModel &,
                                 const std::string &modelRuntimeIdentity,
                                 std::vector<ProgramActivityCost> &,
                                 llvm::json::Value &receipt,
                                 std::string &error);
llvm::json::Value
serializeProgramInvocationFeatures(const ProgramInvocationFeatures &);
} // namespace orbit::taskflow
#endif
