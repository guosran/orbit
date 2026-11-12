#include "Taskflow/ProgramInvocationEvaluation.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <cmath>
#include <set>
using namespace orbit::taskflow;
namespace {
bool fail(std::string &e, const std::string &s) {
  e = "program invocation model: " + s;
  return false;
}
} // namespace
bool orbit::taskflow::parseProgramInvocationModel(
    const llvm::json::Value &value, ProgramInvocationModel &out,
    std::string &error) {
  out = {};
  auto o = value.getAsObject();
  if (!o || o->getString("kind") != "region-invocation")
    return fail(error, "region-invocation model required");
  if (o->getString("schema") != "orbit.region-invocation.cost.v28" ||
      !o->getString("model_id") || o->getString("model_id")->empty())
    return fail(error,
                "versioned invocation schema and nonempty model_id required");
  const std::set<std::string> fields = {"kind",
                                        "schema",
                                        "model_id",
                                        "mode",
                                        "boundary",
                                        "parameters",
                                        "implementations",
                                        "runtime",
                                        "calibration_runtime_identity",
                                        "completion_policy"};
  for (auto &p : *o)
    if (!fields.count(p.first.str()))
      return fail(error, "unconsumed model field: " + p.first.str());
  auto r = o->getObject("runtime");
  auto parameters = o->getObject("parameters");
  auto implementations = o->getArray("implementations");
  if (!r || !parameters || !implementations)
    return fail(error, "runtime, parameters and implementations required");
  const std::set<std::string> runtimeFields = {"identity",
                                               "state",
                                               "host_worker_resource",
                                               "bound_buffers",
                                               "stage_outputs_to_device",
                                               "reset_inputs",
                                               "copy_final_outputs"};
  for (auto &p : *r)
    if (!runtimeFields.count(p.first.str()))
      return fail(error, "unconsumed runtime field: " + p.first.str());
  auto &runtime = out.runtime;
  runtime.identity = r->getString("identity").value_or("").str();
  runtime.hostWorkerResource =
      r->getString("host_worker_resource").value_or("").str();
  runtime.state = r->getString("state").value_or("").str();
  auto boolean = [&](const char *key, bool &target) {
    auto b = r->getBoolean(key);
    if (!b)
      return false;
    target = *b;
    return true;
  };
  if (!boolean("bound_buffers", runtime.boundBuffers) ||
      !boolean("stage_outputs_to_device", runtime.stageOutputsToDevice) ||
      !boolean("reset_inputs", runtime.resetInputs) ||
      !boolean("copy_final_outputs", runtime.copyFinalOutputs))
    return fail(error, "all runtime policy flags must be explicit booleans");
  out.calibrationRuntimeIdentity =
      o->getString("calibration_runtime_identity").value_or("").str();
  if (o->get("completion_policy") && !o->getString("completion_policy"))
    return fail(error, "completion_policy must be a string");
  out.completionPolicy =
      o->getString("completion_policy").value_or("post-device-overhead").str();
  if (out.completionPolicy != "post-device-overhead" &&
      out.completionPolicy != "polling-quantized")
    return fail(error, "unknown completion policy");
  if (runtime.identity.empty() || runtime.state != "prepared-hot" ||
      out.calibrationRuntimeIdentity != runtime.identity)
    return fail(
        error,
        "prepared-hot runtime and matching calibration identity required");
  auto &m = out.mechanisms;
  m.id = staticFingerprint(value);
  m.mode = o->getString("mode").value_or("").str();
  m.boundary = o->getString("boundary").value_or("").str();
  if (m.boundary.empty() || (m.mode != "strict" && m.mode != "estimate"))
    return fail(error, "explicit boundary and strict/estimate mode required");
  for (auto &v : *implementations) {
    auto s = v.getAsString();
    if (!s || s->empty())
      return fail(error, "implementation must be a string");
    m.implementations.push_back(s->str());
  }
  for (auto &entry : *parameters) {
    auto p = entry.second.getAsObject();
    auto provenance = p ? p->getObject("provenance") : nullptr;
    if (!p || !provenance || p->size() != 3 || provenance->size() != 2 ||
        !p->get("value") || !p->getString("unit") ||
        !provenance->getString("kind") || !provenance->getString("source"))
      return fail(error, "parameter requires value/unit/provenance: " +
                             entry.first.str());
    ProgramMechanismParameter x;
    if (!p->get("value")->getAsNull()) {
      x.value = p->getNumber("value");
      if (!x.value)
        return fail(error, "parameter value must be number or null");
    }
    x.unit = p->getString("unit")->str();
    x.kind = provenance->getString("kind")->str();
    x.source = provenance->getString("source")->str();
    m.parameters[entry.first.str()] = std::move(x);
  }
  return true;
}
bool orbit::taskflow::evaluateProgramInvocationModel(
    const ProgramGraph &graph, const ProgramInvocationModel &model,
    ProgramMechanismEvaluation &out, llvm::json::Value &receipt,
    std::string &error) {
  ProgramTimingRequest request;
  if (!buildProgramMechanismCosts(graph, model.mechanisms, out, request, error,
                                  false, false))
    return false;
  ProgramInvocationFeatures features;
  if (!collectProgramInvocationFeatures(graph, model.runtime, features, error))
    return false;
  std::vector<ProgramActivityCost> boundary;
  auto mechanisms = model.mechanisms;
  if (model.completionPolicy == "polling-quantized") {
    // A zero placeholder exposes the checked device critical interval. It is
    // replaced below with the residual detection delay before final scheduling.
    // No measured submit-to-completion duration is added to device work.
    mechanisms.parameters["invocation.device.complete.per_call"] = {
        0, "ns", "architecture",
        "temporary zero for conditional polling evaluation; not a free "
        "completion"};
  }
  if (!buildProgramInvocationCosts(graph, features, mechanisms,
                                   model.calibrationRuntimeIdentity, boundary,
                                   receipt, error))
    return false;
  // Input reset is application work before every body or lifecycle event.
  // The boundary builder cannot mutate costs owned by the body provider.
  if (std::any_of(boundary.begin(), boundary.end(), [](const auto &c) {
        return c.id == "invocation.input_reset";
      }))
    for (auto &c : request.costs)
      c.predecessors.push_back("invocation.input_reset");
  for (const auto &r : features.regions)
    if (r.device) {
      const auto &scope = graph.executionRegions[r.region];
      auto members = scope.memberEvents;
      if (members.empty())
        for (size_t i = scope.beginEvent; i < scope.endEvent; ++i)
          members.push_back(i);
      auto owns = [&](const ProgramActivityCost &c) {
        return std::any_of(c.events.begin(), c.events.end(), [&](size_t e) {
          return std::find(members.begin(), members.end(), e) != members.end();
        });
      };
      auto setup = "invocation.device_setup." + std::to_string(r.region);
      auto teardown = "invocation.device_teardown." + std::to_string(r.region);
      auto complete = "invocation.lifecycle." + std::to_string(r.complete);
      auto before = std::find_if(boundary.begin(), boundary.end(),
                                 [&](const auto &c) { return c.id == setup; });
      auto after =
          std::find_if(boundary.begin(), boundary.end(),
                       [&](const auto &c) { return c.id == teardown; });
      auto completion =
          std::find_if(boundary.begin(), boundary.end(),
                       [&](const auto &c) { return c.id == complete; });
      if (before == boundary.end() || after == boundary.end() ||
          completion == boundary.end())
        return fail(error,
                    "missing per-invocation physical setup/teardown owner");
      for (auto &c : request.costs)
        if (owns(c)) {
          c.predecessors.push_back(setup);
          after->predecessors.push_back(c.id);
        }
      completion->predecessors.push_back(teardown);
    }
  for (auto &c : boundary)
    request.costs.push_back(std::move(c));
  llvm::json::Array polling;
  if (model.completionPolicy == "polling-quantized") {
    ProgramTimingEvaluation provisional;
    if (!evaluateProgramTiming(graph, request, provisional, error))
      return false;
    auto parameter = [&](const char *key) -> std::optional<double> {
      auto p = model.mechanisms.parameters.find(key);
      if (p == model.mechanisms.parameters.end() || !p->second.value ||
          p->second.unit != "ns" ||
          (model.mechanisms.mode == "strict" && p->second.kind == "assumed"))
        return {};
      return p->second.value;
    };
    auto quantum = parameter("invocation.polling.quantum");
    auto probe = parameter("invocation.polling.probe");
    if (quantum && *quantum <= 0)
      return fail(error, "polling quantum must be positive");
    for (const auto &r : features.regions)
      if (r.device) {
        std::string submit = "invocation.lifecycle." + std::to_string(r.submit);
        std::string complete =
            "invocation.lifecycle." + std::to_string(r.complete);
        auto a = std::find_if(provisional.activities.begin(),
                              provisional.activities.end(),
                              [&](const auto &x) { return x.id == submit; });
        auto b = std::find_if(provisional.activities.begin(),
                              provisional.activities.end(),
                              [&](const auto &x) { return x.id == complete; });
        auto c = std::find_if(request.costs.begin(), request.costs.end(),
                              [&](const auto &x) { return x.id == complete; });
        if (c == request.costs.end() || a == provisional.activities.end() ||
            b == provisional.activities.end())
          return fail(error, "missing polling lifecycle owner");
        c->durationMs.reset();
        c->lowerMs = 0;
        c->unknownReason = "polling completion needs known body interval and "
                           "calibrated quantum/probe parameters";
        std::optional<double> body, residual, polls;
        if (a->endMs && b->endMs && quantum && probe) {
          body = std::max(0.0, (*b->endMs - *a->endMs) * 1e6);
          // Declared polling policy: the first probe is at submit return;
          // repeated probes follow the effective measured wakeup quantum. This
          // is a phase estimate, not a guarantee of OS wakeup or a device-cycle
          // measurement.
          polls = std::ceil(*body / *quantum);
          residual = std::max(0.0, *polls * *quantum - *body) + *probe;
          if (!std::isfinite(*residual))
            return fail(error, "polling duration overflow");
          c->durationMs = *residual / 1e6;
          c->lowerMs = c->durationMs;
          c->unknownReason.clear();
        }
        polling.push_back(llvm::json::Object{
            {"region", int64_t(r.region)},
            {"body_ns",
             body ? llvm::json::Value(*body) : llvm::json::Value(nullptr)},
            {"polls",
             polls ? llvm::json::Value(*polls) : llvm::json::Value(nullptr)},
            {"post_body_detection_ns", residual ? llvm::json::Value(*residual)
                                                : llvm::json::Value(nullptr)}});
      }
  }
  if (!evaluateProgramTiming(graph, request, out.timing, error))
    return false;
  bool assumed = !out.assumptions.empty();
  for (auto &p : model.mechanisms.parameters)
    if (p.second.kind == "assumed")
      assumed = true;
  out.status = !out.timing.makespanMs ? "unknown-required-mechanisms"
               : assumed ? "assumption-driven-invocation-estimate"
                         : "invocation-estimate-with-declared-provenance";
  if (auto o = receipt.getAsObject()) {
    (*o)["logical_group_configuration_charged"] = false;
    (*o)["configuration_scope"] =
        "host context/ELF creation outside boundary; device setup and teardown "
        "priced once per physical invocation; logical execution groups are not "
        "additional calls";
    (*o)["physical_setup_and_teardown_charged"] = true;
    (*o)["body_and_boundary_coverage"] =
        "exclusive disjoint graph events and region lifecycle events";
    (*o)["completion_policy"] = model.completionPolicy;
    (*o)["conditional_polling"] = std::move(polling);
    (*o)["polling_scope"] =
        "estimated first-probe phase at submit return and measured effective "
        "wakeup quantum; OS wakeup variation remains; complete owns detection "
        "tail only";
    (*o)["status"] = out.timing.makespanMs ? "complete-invocation-estimate"
                                           : "unknown-required-mechanisms";
  }
  return true;
}
