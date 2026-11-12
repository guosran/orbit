#ifndef ORBIT_TASKFLOW_COMPUTE_ESTIMATE_H
#define ORBIT_TASKFLOW_COMPUTE_ESTIMATE_H
#include "llvm/Support/JSON.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace orbit::taskflow {
enum class MappingComputeTransportScope {
  Unspecified,
  ExcludesInterTaskTransfers,
  IncludesInterTaskTransfers
};
inline const char *
computeTransportScopeName(MappingComputeTransportScope scope) {
  switch (scope) {
  case MappingComputeTransportScope::ExcludesInterTaskTransfers:
    return "excludes_inter_task_transfers";
  case MappingComputeTransportScope::IncludesInterTaskTransfers:
    return "includes_inter_task_transfers";
  default:
    return "unspecified";
  }
}
// A backend publishes its already-computed task duration. This interface does
// not infer latency from II, retrain a model, or include inter-task transport.
// Source/provider conditions and charge ownership travel with the estimate.
struct MappingComputeEstimate {
  size_t task = 0;
  std::string resource, implementation, model, modelVersion, provenance,
      boundary;
  double duration = 0;
  std::string unit = "ms";
  std::vector<std::string> includedCosts, excludedCosts, requiredDataSupply;
  bool supplyConditionsVerified = false;
  MappingComputeTransportScope transportScope =
      MappingComputeTransportScope::Unspecified;
  int64_t resourceCount = 1;
  // Resource topology and implementation geometry can have different units.
  std::vector<int64_t> resourceShape, implementationShape;
  std::string resourceCountUnit = "execution_resource";
  std::string resourceShapeUnit = "provider_defined";
  std::string implementationShapeUnit = "provider_defined";
  std::optional<double> predictedII, startupCycles;
  std::optional<int64_t> iterations;
};
inline bool convertComputeEstimateToMs(const MappingComputeEstimate &estimate,
                                       std::optional<double> clockMHz,
                                       double &result, std::string &error) {
  if (!std::isfinite(estimate.duration) || estimate.duration <= 0 ||
      estimate.model.empty() || estimate.modelVersion.empty() ||
      estimate.provenance.empty() || estimate.boundary.empty() ||
      estimate.resource.empty() || estimate.resourceCount < 1) {
    error = "compute estimate requires positive finite duration and explicit "
            "provider/resource/boundary identity";
    return false;
  }
  if (estimate.transportScope !=
      MappingComputeTransportScope::ExcludesInterTaskTransfers) {
    error = "compute estimate must explicitly exclude separately charged "
            "inter-task transfers";
    return false;
  }
  if (!estimate.supplyConditionsVerified) {
    error = "compute estimate data-supply conditions are not verified";
    return false;
  }
  if (estimate.unit == "ms")
    result = estimate.duration;
  else if (estimate.unit == "cycles" && clockMHz && std::isfinite(*clockMHz) &&
           *clockMHz > 0)
    result = estimate.duration / *clockMHz / 1000.0;
  else {
    error = "compute estimate units require ms or cycles with explicit "
            "positive clock MHz";
    return false;
  }
  if (!std::isfinite(result) || result <= 0) {
    error = "compute estimate conversion is not finite and positive";
    return false;
  }
  return true;
}
inline llvm::json::Value
serializeMappingComputeEstimate(const MappingComputeEstimate &estimate) {
  auto strings = [](const std::vector<std::string> &values) {
    llvm::json::Array a;
    for (auto &v : values)
      a.push_back(v);
    return a;
  };
  llvm::json::Array shape;
  for (auto v : estimate.resourceShape)
    shape.push_back(v);
  llvm::json::Array implementationShape;
  for (auto v : estimate.implementationShape)
    implementationShape.push_back(v);
  return llvm::json::Object{
      {"task", int64_t(estimate.task)},
      {"resource", estimate.resource},
      {"implementation", estimate.implementation},
      {"model", estimate.model},
      {"model_version", estimate.modelVersion},
      {"provenance", estimate.provenance},
      {"duration", estimate.duration},
      {"unit", estimate.unit},
      {"boundary", estimate.boundary},
      {"included_costs", strings(estimate.includedCosts)},
      {"excluded_costs", strings(estimate.excludedCosts)},
      {"required_data_supply", strings(estimate.requiredDataSupply)},
      {"supply_conditions_verified", estimate.supplyConditionsVerified},
      {"transfer_accounting",
       computeTransportScopeName(estimate.transportScope)},
      {"resource_count", estimate.resourceCount},
      {"resource_count_unit", estimate.resourceCountUnit},
      {"resource_shape", std::move(shape)},
      {"resource_shape_unit", estimate.resourceShapeUnit},
      {"implementation_shape", std::move(implementationShape)},
      {"implementation_shape_unit", estimate.implementationShapeUnit},
      {"predicted_ii", estimate.predictedII
                           ? llvm::json::Value(*estimate.predictedII)
                           : llvm::json::Value(nullptr)},
      {"startup_cycles", estimate.startupCycles
                             ? llvm::json::Value(*estimate.startupCycles)
                             : llvm::json::Value(nullptr)},
      {"iterations", estimate.iterations
                         ? llvm::json::Value(*estimate.iterations)
                         : llvm::json::Value(nullptr)}};
}
} // namespace orbit::taskflow
#endif
