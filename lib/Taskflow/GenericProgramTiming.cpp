#include "Taskflow/GenericProgramTiming.h"
using namespace orbit::taskflow;
bool orbit::taskflow::evaluateGenericProgramTiming(
    const GenericAnalysis &analysis, const GenericMappingPlan &plan,
    llvm::json::Value &result, std::string &error) {
  if (!checkGenericMapping(analysis, plan, error))
    return false;
  if (!plan.directCost.eventTiming.getAsNull()) {
    result = llvm::json::Object{
        {"schema", "orbit.generic-program-timing.v18"},
        {"graph_id", programGraphIdentity(plan.communication)},
        {"native_plan_id", plan.planId},
        {"source_body_fingerprint", analysis.bodyFingerprint},
        {"event_mechanisms", llvm::json::Value(plan.directCost.eventTiming)},
        {"observation", "Explicit mechanism estimate on the checked execution "
                        "graph. Assumptions are not hardware measurements; old "
                        "whole-invocation composites are not added."}};
    return true;
  }
  const auto &graph = plan.communication;
  GenericCostFeatures features;
  if (!collectGenericCostFeatures(analysis, plan, features, error))
    return false;
  ProgramTimingRequest request;
  request.graphId = programGraphIdentity(graph);
  request.modelIdentity =
      plan.directCost.enabled ? plan.directCost.modelDigest : "unknown";
  request.boundary = "prepared-command-host-to-host";
  ProgramActivityCost invocation;
  invocation.id = "prepared-npu-invocation";
  invocation.implementation = "checked-direct-air-scalar-runtime-composite";
  invocation.boundary = request.boundary;
  invocation.provenance =
      "Existing native direct-AIR cost provider; owns sync, submission, "
      "configuration, DMA, local compute, locks, stalls and completion "
      "observation together. No independent DMA or compute price is added.";
  for (size_t i = 0; i < graph.events.size(); ++i) {
    invocation.events.push_back(i);
    const auto &event = graph.events[i];
    if (!event.resource.empty())
      invocation.resourceDemands[event.resource] = 1;
  }
  for (const auto &instance : graph.instances) {
    const auto &resource = graph.events[instance.event].resource;
    if (!resource.empty())
      invocation.resourceDemands[resource] = std::max(
          invocation.resourceDemands[resource], instance.resourceCount);
  }
  for (size_t i = 0; i < graph.regionEvents.size(); ++i) {
    invocation.regionEvents.push_back(i);
    const auto &resource = graph.regionEvents[i].resource;
    if (!resource.empty())
      invocation.resourceDemands[resource] = 1;
  }
  if (graph.regionEvents.empty())
    for (size_t i = 0; i < graph.executionRegions.size(); ++i) {
      invocation.executionRegions.push_back(i);
      if (!graph.executionRegions[i].submissionResource.empty())
        invocation
            .resourceDemands[graph.executionRegions[i].submissionResource] = 1;
    }
  invocation.durationMs = plan.directCost.valueMs;
  if (!invocation.durationMs)
    invocation.unknownReason =
        plan.directCost.enabled
            ? plan.directCost.reason
            : "No calibrated complete-invocation parameters. Native "
              "descriptors and scalar work are known; internal timing and "
              "overlap are unidentified.";
  request.costs.push_back(invocation);
  ProgramTimingEvaluation complete;
  if (!evaluateProgramTiming(graph, request, complete, error))
    return false;
  ProgramTimingRequest mechanisms;
  mechanisms.graphId = request.graphId;
  mechanisms.modelIdentity = request.modelIdentity;
  mechanisms.boundary = "internal-public-events-unidentified";
  ProgramTimingEvaluation internal;
  if (!evaluateProgramTiming(graph, mechanisms, internal, error))
    return false;
  llvm::json::Array terms;
  if (auto object = plan.directCost.model.getAsObject())
    if (auto entries = object->getArray("terms"))
      for (const auto &entry : *entries) {
        auto term = entry.getAsObject();
        auto feature = term ? term->getString("feature") : std::nullopt;
        auto coefficient =
            term ? term->getNumber("coefficient_ms") : std::nullopt;
        if (feature && coefficient && features.values.count(feature->str()))
          terms.push_back(llvm::json::Object{
              {"feature", *feature},
              {"native_quantity", features.values.at(feature->str())},
              {"coefficient_ms", *coefficient},
              {"contribution_ms",
               *coefficient * features.values.at(feature->str())},
              {"scope", "implementation-composite contribution; not an "
                        "independently measured event duration"}});
      }
  result = llvm::json::Object{
      {"schema", "orbit.generic-program-timing.v16"},
      {"graph_id", request.graphId},
      {"native_plan_id", plan.planId},
      {"source_body_fingerprint", analysis.bodyFingerprint},
      {"features", serializeGenericCostFeatures(features)},
      {"complete_invocation", serializeProgramTiming(complete)},
      {"internal_mechanisms", serializeProgramTiming(internal)},
      {"formula_contributions", std::move(terms)},
      {"physical_device_duration_ms", nullptr},
      {"physical_bandwidth_bytes_per_ms", nullptr},
      {"observation", "Host prepared bound-BO runner call; excludes caller "
                      "input reset, output poisoning and evidence IO. Actual "
                      "runtime/observer identity must match the frozen model."},
      {"limitation", "Internal public events preserve semantic order and "
                     "resources. Backend composite may coalesce transfers and "
                     "overlap internal channels; this invocation model cannot "
                     "identify their separate durations or certify overlap."}};
  return true;
}
