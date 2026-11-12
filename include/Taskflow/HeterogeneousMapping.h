#ifndef ORBIT_TASKFLOW_HETEROGENEOUS_MAPPING_H
#define ORBIT_TASKFLOW_HETEROGENEOUS_MAPPING_H

#include "Taskflow/ProgramMapping.h"

namespace orbit::taskflow {
struct HeterogeneousMappingChoice {
  std::vector<std::string> taskDevices; // exactly "cpu" or "npu"
  // Additional boundaries before these task indices. Device changes always
  // create a boundary; adjacent equal devices otherwise form one region.
  std::vector<size_t> cuts;
  // The parent owns host ABI copies and call lifecycle. A separately checked
  // implementation child owns its physical tiles, local buffers and DMA.
  // False retains the legacy full-tensor resident physical contract.
  bool implementationOwned = false;
};
struct HeterogeneousExecutionSlice {
  size_t beginTask = 0, endTask = 0;
  std::string device;
  std::vector<size_t> inputStorages, outputStorages, internalStorages;
  // Unaccessed readonly task state keeps its host identity across a cut. These
  // arguments carry ordering only, never a payload copy or device transfer.
  std::vector<size_t> orderOnlyStorages;
};
struct HeterogeneousMappingPlan {
  ProgramGraph graph; // the authoritative parent task/data graph
  HeterogeneousMappingChoice choice;
  std::vector<HeterogeneousExecutionSlice> slices;
  std::string sourceFingerprint, planId;
};
struct HeterogeneousRegionArgument {
  size_t parentStorage = 0, childArgument = 0, childStorage = 0;
  std::string role;       // input/output; no alias or inout ABI in this adapter
  bool orderOnly = false; // input identity with no actual region access
  int64_t inputCopy = -1, outputCopy = -1; // parent graph copy identities
};
struct HeterogeneousRegionProjection {
  size_t region = 0;
  std::string parentGraphId, parentSourceFingerprint;
  std::string childSourceFingerprint, childGraphId;
  // Child task index -> parent task index, preserving actual source order.
  std::vector<size_t> parentTasks;
  std::vector<HeterogeneousRegionArgument> arguments;
};
bool buildHeterogeneousMapping(const GenericAnalysis &,
                               const HeterogeneousMappingChoice &,
                               HeterogeneousMappingPlan &, std::string &error);
bool checkHeterogeneousMapping(const GenericAnalysis &,
                               const HeterogeneousMappingChoice &,
                               const HeterogeneousMappingPlan &,
                               std::string &error);
bool materializeHeterogeneousRegion(const HeterogeneousMappingPlan &,
                                    size_t region,
                                    mlir::OwningOpRef<mlir::ModuleOp> &,
                                    HeterogeneousRegionProjection &,
                                    std::string &error);
// Bind a separately checked implementation child graph; this does not plan,
// price, or lower it. Its native L1/DMA remains authoritative at child scope.
bool bindHeterogeneousRegionGraph(const HeterogeneousMappingPlan &,
                                  const ProgramGraph &child,
                                  HeterogeneousRegionProjection &,
                                  std::string &error);
struct GenericMappingPlan;
bool bindHeterogeneousRegionNativePlan(const HeterogeneousMappingPlan &,
                                       const GenericMappingPlan &,
                                       HeterogeneousRegionProjection &,
                                       std::string &error);
llvm::json::Value
serializeHeterogeneousMapping(const HeterogeneousMappingPlan &);
llvm::json::Value
serializeHeterogeneousRegionProjection(const HeterogeneousRegionProjection &);
} // namespace orbit::taskflow
#endif
