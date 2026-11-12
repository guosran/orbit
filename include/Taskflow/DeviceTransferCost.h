#ifndef ORBIT_TASKFLOW_DEVICE_TRANSFER_COST_H
#define ORBIT_TASKFLOW_DEVICE_TRANSFER_COST_H
#include "Taskflow/GenericMapping.h"

namespace orbit::taskflow {
struct DeviceTransferDescriptor {
  std::string id;
  int64_t offsetBytes = 0, elements = 0, repetitions = 1, elementBytes = 4;
};
struct DeviceTransferStreamBinding {
  size_t stage = 0, storage = 0;
  std::string direction; // host_to_l1 / l1_to_host
  std::vector<size_t> publicEvents;
  std::vector<DeviceTransferDescriptor> runtimeDescriptors, coreDescriptors;
};
struct DeviceTransferBackendBinding {
  std::string graphId, nativePlanId, artifactSHA256;
  std::string runtimeIdentity, toolchainIdentity, pathIdentity;
  std::string observerIdentity, conditionIdentity;
  std::string completionBoundary = "submit-to-successful-wait-return";
  std::vector<DeviceTransferStreamBinding> streams;
};
struct DeviceTransferFact {
  size_t event = 0, stage = 0, storage = 0;
  std::string direction, sourceMemory, targetMemory, resource, link;
  MappingRegion region;
  std::vector<size_t> sourceCopies, targetCopies, versions;
  int64_t requestedBytes = 0;
};
struct DeviceTransferFacts {
  DeviceTransferBackendBinding binding;
  std::string architectureIdentity, implementation = "identity-copy-f32";
  std::vector<DeviceTransferFact> transfers;
  std::vector<size_t> coveredComputeEvents;
  std::vector<std::string> reservationResources;
  int64_t publicTransferCount = 0, runtimeDescriptorCount = 0;
  int64_t coreTransactions = 0, requestedBytes = 0, descriptorBytes = 0;
  int64_t segmentCount = 0, tileElements = 0, elements = 0;
};
struct DeviceTransferCostDomain {
  std::string runtimeIdentity, toolchainIdentity, pathIdentity,
      observerIdentity;
  std::string conditionIdentity, architectureIdentity;
  std::string implementation = "identity-copy-f32";
  std::string completionBoundary = "submit-to-successful-wait-return";
  int64_t minBytes = 0, maxBytes = 0, minSegments = 0, maxSegments = 0;
  int64_t minTileElements = 0, maxTileElements = 0;
};
struct DeviceTransferCostRow {
  DeviceTransferCostDomain domain;
  std::optional<double> entryNs, segmentNs, byteNs;
  StaticProvenance provenance;
  std::string rawManifestSHA256;
};
struct DeviceTransferCostModel {
  std::string modelId, parameterFileSHA256;
  std::vector<DeviceTransferCostRow> rows;
};
struct DeviceTransferCostEvaluation {
  std::string status = "unknown", reason, modelId, parameterFileSHA256;
  std::optional<double> durationNs;
  std::optional<size_t> matchedRow;
  std::vector<size_t> coveredEvents;
  std::vector<std::string> reservationResources;
};
struct DeviceTransferComposition {
  std::vector<size_t> separatelyPricedEvents;
  bool submitAlreadyPriced = false, completionAlreadyPriced = false;
};
bool collectDeviceTransferFacts(const GenericAnalysis &,
                                const GenericMappingPlan &,
                                const DeviceTransferBackendBinding &,
                                DeviceTransferFacts &, std::string &error);
bool checkDeviceTransferFacts(const GenericAnalysis &,
                              const GenericMappingPlan &,
                              const DeviceTransferFacts &, std::string &error);
bool parseDeviceTransferCostModel(llvm::StringRef jsonBytes,
                                  DeviceTransferCostModel &,
                                  std::string &error);
// Rejects additive prices on any event already owned by this composite. The
// caller lists separately priced graph events; an empty list means none.
bool evaluateDeviceTransferCost(const DeviceTransferFacts &,
                                const DeviceTransferCostModel &,
                                const DeviceTransferComposition &,
                                DeviceTransferCostEvaluation &,
                                std::string &error);
llvm::json::Value
serializeDeviceTransferBackendBinding(const DeviceTransferBackendBinding &);
llvm::json::Value serializeDeviceTransferFacts(const DeviceTransferFacts &);
llvm::json::Value
serializeDeviceTransferCost(const DeviceTransferCostEvaluation &);
} // namespace orbit::taskflow
#endif
