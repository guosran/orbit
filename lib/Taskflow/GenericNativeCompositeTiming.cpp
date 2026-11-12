#include "Taskflow/GenericNativeCompositeTiming.h"
#include <algorithm>
using namespace orbit::taskflow;
namespace {
using J = llvm::json::Value;
using O = llvm::json::Object;
bool fail(std::string &error, const std::string &message) {
  error = "native composite timing: " + message;
  return false;
}
} // namespace
bool orbit::taskflow::evaluateGenericNativeCompositeTiming(
    const GenericAnalysis &source, const GenericMappingPlan &native,
    const ProgramGraph &selected, const J &model, const J &target,
    ProgramTimingEvaluation &timed, J &receipt, std::string &error) {
  timed = {};
  receipt = nullptr;
  const auto *object = model.getAsObject();
  const auto kind = object ? object->getString("kind") : std::nullopt;
  if (!kind ||
      (*kind != "linear-composite" && *kind != "implementation-composite"))
    return fail(error, "only guarded linear-composite/implementation-composite "
                       "full-call providers are supported");
  if (!checkGenericMapping(source, native, error) ||
      !checkProgramGraphSource(source, selected, error) ||
      !checkProgramGraph(selected, error))
    return false;
  const std::string nativeId = programGraphIdentity(native.communication);
  const std::string selectedId = programGraphIdentity(selected);
  const bool matched = nativeId == selectedId;
  GenericDirectAIRCost cost;
  GenericCostFeatures features;
  std::string unknown;
  // Do not collect old stage/DMA features and re-label them as selected facts.
  if (matched) {
    if (!collectGenericCostFeatures(source, native, features, error) ||
        !evaluateGenericDirectAIRCost(features, model, target, cost, error))
      return false;
    if (!cost.valueMs)
      unknown = cost.reason;
  } else {
    unknown =
        "selected public graph differs from the checked native executable "
        "graph; "
        "native stage/DMA/control counts cannot price post-rebind choices";
  }
  ProgramTimingRequest request;
  request.graphId = selectedId;
  request.modelIdentity = staticFingerprint(model);
  request.boundary = "prepared-command-host-to-host";
  ProgramActivityCost invocation;
  invocation.id = "native.complete-invocation";
  invocation.implementation = "checked-native-full-call-composite";
  invocation.boundary = request.boundary;
  invocation.provenance =
      "Existing native source/control-count provider; actual calibration "
      "boundary/runtime "
      "must match. Owns full call once, including whatever "
      "reset/staging/submit/setup/body/DMA/"
      "teardown/wait/readback work the frozen measured full-call boundary "
      "declares; no additive event fees.";
  invocation.durationMs = matched ? cost.valueMs : std::nullopt;
  invocation.unknownReason = unknown;
  auto demand = [&](const std::string &resource, int64_t count) {
    if (!resource.empty())
      invocation.resourceDemands[resource] =
          std::max(invocation.resourceDemands[resource], count);
  };
  for (size_t i = 0; i < selected.events.size(); ++i) {
    invocation.events.push_back(i);
    demand(selected.events[i].resource, 1);
  }
  for (const auto &instance : selected.instances)
    demand(selected.events[instance.event].resource, instance.resourceCount);
  for (size_t i = 0; i < selected.regionEvents.size(); ++i) {
    invocation.regionEvents.push_back(i);
    demand(selected.regionEvents[i].resource, 1);
  }
  if (selected.regionEvents.empty())
    for (size_t i = 0; i < selected.executionRegions.size(); ++i) {
      invocation.executionRegions.push_back(i);
      demand(selected.executionRegions[i].submissionResource, 1);
    }
  request.costs.push_back(std::move(invocation));
  if (!evaluateProgramTiming(selected, request, timed, error))
    return false;
  receipt = O{
      {"schema", "orbit.generic-native-composite-timing.v30"},
      {"status", timed.makespanMs ? "complete-native-invocation-estimate"
                                  : "unknown-complete-native-invocation-cost"},
      {"native_graph_id", nativeId},
      {"selected_graph_id", selectedId},
      {"native_graph_matches_selected", matched},
      {"native_counts_applied_to_selected", matched},
      {"model_digest", staticFingerprint(model)},
      {"target_digest", staticFingerprint(target)},
      {"provider_kind", kind->str()},
      {"provider_status",
       matched ? cost.status : "not-evaluated-graph-mismatch"},
      {"unknown_reason", unknown},
      {"features",
       matched ? serializeGenericCostFeatures(features) : J(nullptr)},
      {"complete_invocation", serializeProgramTiming(timed)},
      {"exclusive_complete_invocation_owner", true},
      {"additive_body_or_lifecycle_costs", false},
      {"complete_plan_prepricing", false},
      {"physical_realization_checked", false},
      {"scope", "Existing calibrated native full-call predictor on exactly its "
                "checked native execution graph; "
                "code/control correspondence and heldout calibration validity "
                "require independent evidence; "
                "not a hardware optimum or event-level duration claim"}};
  return true;
}
