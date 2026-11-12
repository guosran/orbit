#include "Taskflow/ProgramRegionCompositeCost.h"
#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/HeterogeneousSearch.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>
using namespace orbit::taskflow;
namespace {
using J = llvm::json::Value;
using O = llvm::json::Object;
using A = llvm::json::Array;
const std::set<std::string> featureNames = {"call",
                                            "abi_bo_count",
                                            "abi_bytes",
                                            "source_operations",
                                            "source_memory_accesses",
                                            "task_count",
                                            "transfer_count",
                                            "requested_transfer_bytes"};
bool fail(std::string &e, const std::string &s) {
  e = "program region composite: " + s;
  return false;
}
bool nonnegativeFinite(double x) { return std::isfinite(x) && x >= 0; }
A indices(const std::vector<size_t> &v) {
  A a;
  for (auto x : v)
    a.push_back(int64_t(x));
  return a;
}
bool fields(const O &o, const std::set<std::string> &allowed, std::string &e) {
  for (auto &p : o)
    if (!allowed.count(p.first.str()))
      return fail(e, "unconsumed field: " + p.first.str());
  return true;
}
bool parameter(const J &v, ProgramMechanismParameter &p, std::string &e) {
  auto o = v.getAsObject();
  auto prov = o ? o->getObject("provenance") : nullptr;
  if (!o || !prov || o->size() != 3 || prov->size() != 2 || !o->get("value") ||
      !o->getString("unit") || !prov->getString("kind") ||
      !prov->getString("source"))
    return fail(e, "parameter requires explicit value/unit/provenance");
  if (!o->get("value")->getAsNull()) {
    p.value = o->getNumber("value");
    if (!p.value || !nonnegativeFinite(*p.value))
      return fail(e, "parameter must be nonnegative finite number or null");
  }
  p.unit = o->getString("unit")->str();
  p.kind = prov->getString("kind")->str();
  p.source = prov->getString("source")->str();
  if (p.source.empty() || (p.kind != "calibrated" && p.kind != "architecture" &&
                           p.kind != "assumed"))
    return fail(e, "parameter requires explicit valid provenance");
  return true;
}
std::string unit(const std::string &feature) {
  if (feature == "call")
    return "ns";
  if (feature == "abi_bytes" || feature == "requested_transfer_bytes")
    return "ns/byte";
  return "ns/count";
}
std::optional<double> coefficient(const ProgramMechanismParameter *p,
                                  const std::string &units,
                                  const std::string &mode,
                                  const std::string &key,
                                  std::string &unknown) {
  if (!p || !p->value) {
    unknown += "missing " + key + "; ";
    return {};
  }
  if (!nonnegativeFinite(*p->value) || p->source.empty() ||
      (p->kind != "calibrated" && p->kind != "architecture" &&
       p->kind != "assumed") ||
      p->unit != units) {
    unknown += "invalid parameter " + key + "; ";
    return {};
  }
  if (mode == "strict" && p->kind == "assumed") {
    unknown += "strict mode forbids assumed " + key + "; ";
    return {};
  }
  return p->value;
}
std::string id(size_t r) {
  return "invocation.region_composite." + std::to_string(r);
}
J implementationJSON(const ProgramRegionImplementation &i) {
  return O{{"region", int64_t(i.region)},
           {"implementation", i.implementation},
           {"source_fingerprint", i.sourceFingerprint},
           {"code_identity", i.codeIdentity},
           {"control_sequence_identity", i.controlSequenceIdentity},
           {"control_family", i.controlFamily},
           {"provenance", i.provenance}};
}
} // namespace
bool orbit::taskflow::parseProgramRegionCompositeModel(
    const J &v, ProgramRegionCompositeModel &m, std::string &e) {
  m = {};
  auto o = v.getAsObject();
  if (!o || o->getString("kind") != "region-composite" ||
      o->getString("schema") != "orbit.region-composite.cost.v29")
    return fail(e, "region-composite v29 schema required");
  if (!fields(*o,
              {"kind", "schema", "model_id", "mode", "boundary", "runtime",
               "calibration_runtime_identity", "region_models", "parameters"},
              e))
    return false;
  m.id = o->getString("model_id").value_or("").str();
  m.mode = o->getString("mode").value_or("").str();
  m.boundary = o->getString("boundary").value_or("").str();
  m.calibrationRuntimeIdentity =
      o->getString("calibration_runtime_identity").value_or("").str();
  auto runtime = o->getObject("runtime");
  auto rows = o->getArray("region_models");
  auto params = o->getObject("parameters");
  if (m.id.empty() || m.boundary.empty() ||
      (m.mode != "strict" && m.mode != "estimate") || !runtime || !rows ||
      !params)
    return fail(
        e, "explicit model "
           "identity/mode/boundary/runtime/region_models/parameters required");
  if (!fields(*runtime,
              {"identity", "state", "host_worker_resource", "bound_buffers",
               "stage_outputs_to_device", "reset_inputs", "copy_final_outputs"},
              e))
    return false;
  m.runtime.identity = runtime->getString("identity").value_or("").str();
  m.runtime.state = runtime->getString("state").value_or("").str();
  m.runtime.hostWorkerResource =
      runtime->getString("host_worker_resource").value_or("").str();
  const std::pair<const char *, bool *> runtimeFlags[] = {
      {"bound_buffers", &m.runtime.boundBuffers},
      {"stage_outputs_to_device", &m.runtime.stageOutputsToDevice},
      {"reset_inputs", &m.runtime.resetInputs},
      {"copy_final_outputs", &m.runtime.copyFinalOutputs}};
  for (const auto &item : runtimeFlags) {
    auto b = runtime->getBoolean(item.first);
    if (!b)
      return fail(e, "all runtime flags require explicit booleans");
    *item.second = *b;
  }
  if (m.runtime.identity.empty() || m.runtime.identity == "unbound" ||
      m.runtime.state != "prepared-hot" ||
      m.runtime.hostWorkerResource.empty() ||
      m.calibrationRuntimeIdentity != m.runtime.identity)
    return fail(e,
                "prepared-hot matching calibrated runtime identity required");
  std::set<std::tuple<std::string, std::string, std::string>> keys;
  for (const auto &v : *rows) {
    auto row = v.getAsObject();
    if (!row || !fields(*row,
                        {"implementation", "source_family", "control_family",
                         "provenance", "coefficients", "domain"},
                        e))
      return fail(e, "invalid regional calibration row");
    ProgramRegionCompositeCalibration r;
    r.implementation = row->getString("implementation").value_or("").str();
    r.sourceFamily = row->getString("source_family").value_or("").str();
    r.controlFamily = row->getString("control_family").value_or("").str();
    r.provenance = row->getString("provenance").value_or("").str();
    auto cs = row->getObject("coefficients");
    auto ds = row->getObject("domain");
    if (r.implementation.empty() || r.sourceFamily.empty() ||
        r.controlFamily.empty() || r.provenance.empty() || !cs || !ds ||
        !keys.emplace(r.implementation, r.sourceFamily, r.controlFamily).second)
      return fail(e, "unique semantic/implementation/control calibration and "
                     "provenance required");
    for (const auto &c : *cs) {
      if (!featureNames.count(c.first.str()))
        return fail(e, "unknown regional coefficient: " + c.first.str());
      ProgramMechanismParameter p;
      if (!parameter(c.second, p, e))
        return false;
      if (p.unit != unit(c.first.str()))
        return fail(e, "regional coefficient unit mismatch: " + c.first.str());
      r.coefficients[c.first.str()] = std::move(p);
    }
    for (const auto &d : *ds) {
      auto range = d.second.getAsObject();
      auto lo = range ? range->getNumber("min") : std::nullopt,
           hi = range ? range->getNumber("max") : std::nullopt;
      if (!featureNames.count(d.first.str()) || !range || range->size() != 2 ||
          !lo || !hi || !nonnegativeFinite(*lo) || !nonnegativeFinite(*hi) ||
          *lo > *hi)
        return fail(e, "invalid calibrated domain range");
      r.domain[d.first.str()] = {*lo, *hi};
    }
    m.regions.push_back(std::move(r));
  }
  const std::set<std::string> global = {
      "invocation.input_reset.per_call", "invocation.input_reset.per_bo",
      "invocation.input_reset.per_byte", "invocation.output_copy.per_call",
      "invocation.output_copy.per_bo",   "invocation.output_copy.per_byte",
      "invocation.control.per_call",     "invocation.control.per_region"};
  for (auto &entry : *params) {
    if (!global.count(entry.first.str()))
      return fail(
          e,
          "region composite forbids additive per-event/lifecycle parameter: " +
              entry.first.str());
    ProgramMechanismParameter p;
    if (!parameter(entry.second, p, e))
      return false;
    std::string units =
        llvm::StringRef(entry.first.str()).ends_with("per_byte") ? "ns/byte"
        : llvm::StringRef(entry.first.str()).ends_with("per_region")
            ? "ns/count"
            : "ns";
    if (p.unit != units)
      return fail(e, "global boundary parameter unit mismatch");
    m.parameters[entry.first.str()] = std::move(p);
  }
  m.id = staticFingerprint(v);
  return true;
}
J orbit::taskflow::serializeProgramRegionCompositeFeatures(
    const ProgramRegionCompositeFeatureSet &f) {
  A rows;
  for (auto &r : f.regions) {
    O quantities, resources;
    for (auto &x : r.quantities)
      quantities[x.first] = x.second;
    for (auto &x : r.resourceDemands)
      resources[x.first] = x.second;
    rows.push_back(O{{"region", int64_t(r.region)},
                     {"source_family", r.sourceFamily},
                     {"source_fingerprint", r.sourceFingerprint},
                     {"implementation", implementationJSON(r.implementation)},
                     {"events", indices(r.events)},
                     {"region_events", indices(r.regionEvents)},
                     {"source_tasks", indices(r.sourceTasks)},
                     {"resource_demands", std::move(resources)},
                     {"quantities", std::move(quantities)},
                     {"source_work_known", r.sourceWorkKnown},
                     {"unknown_reason", r.unknownReason}});
  }
  return O{{"schema", "orbit.program-region-composite-features.v29"},
           {"graph_id", f.graphId},
           {"identity", f.identity},
           {"invocation", serializeProgramInvocationFeatures(f.invocation)},
           {"regions", std::move(rows)}};
}
bool orbit::taskflow::collectProgramRegionCompositeFeatures(
    const HeterogeneousMappingPlan &plan,
    const ProgramInvocationRuntimeContract &runtime,
    const std::vector<ProgramRegionImplementation> &bindings,
    ProgramRegionCompositeFeatureSet &f, std::string &e) {
  f = {};
  const auto &g = plan.graph;
  if (!checkHeterogeneousMapping(g.source, plan.choice, plan, e) ||
      !collectProgramInvocationFeatures(g, runtime, f.invocation, e))
    return false;
  f.graphId = programGraphIdentity(g);
  if (bindings.size() != g.executionRegions.size())
    return fail(e, "one compiled implementation witness per region required");
  ProgramMechanismModel counting;
  counting.id = "source-work-features-only.v29";
  counting.mode = "strict";
  counting.boundary = "source counters only; no price or schedule used";
  for (const auto &i : g.instances)
    if (!llvm::is_contained(counting.implementations, i.implementation))
      counting.implementations.push_back(i.implementation);
  ProgramMechanismEvaluation counts;
  ProgramTimingRequest discarded;
  if (!buildProgramMechanismCosts(g, counting, counts, discarded, e, false,
                                  false))
    return false;
  std::vector<bool> seen(g.events.size()), seenLifecycle(g.regionEvents.size());
  for (size_t ri = 0; ri < g.executionRegions.size(); ++ri) {
    const auto &scope = g.executionRegions[ri];
    const ProgramRegionImplementation *binding = nullptr;
    for (const auto &b : bindings)
      if (b.region == ri) {
        if (binding)
          return fail(e, "duplicate implementation witness");
        binding = &b;
      }
    if (!binding || binding->implementation != scope.implementation ||
        binding->codeIdentity.empty() ||
        binding->controlSequenceIdentity.empty() ||
        binding->controlFamily.empty() || binding->provenance.empty())
      return fail(e,
                  "compiled implementation/control/source evidence required");
    mlir::OwningOpRef<mlir::ModuleOp> child;
    HeterogeneousRegionProjection projection;
    if (!materializeHeterogeneousRegion(plan, ri, child, projection, e))
      return false;
    GenericAnalysis childAnalysis;
    if (!analyzeGenericProgram(*child, childAnalysis, e))
      return false;
    if (binding->sourceFingerprint != childAnalysis.bodyFingerprint)
      return fail(
          e, "compiled implementation source differs from materialized region");
    ProgramRegionCompositeFeatures r;
    r.region = ri;
    r.sourceFingerprint = childAnalysis.bodyFingerprint;
    r.sourceFamily = heterogeneousProgramFamily(*child);
    r.implementation = *binding;
    r.sourceTasks = scope.sourceTasks;
    r.events = scope.memberEvents;
    if (r.events.empty())
      for (size_t i = scope.beginEvent; i < scope.endEvent; ++i)
        r.events.push_back(i);
    r.resourceDemands[runtime.hostWorkerResource] = 1;
    for (const auto &key : featureNames)
      r.quantities[key] = 0;
    r.quantities["call"] = 1;
    r.quantities["abi_bo_count"] =
        f.invocation.regions[ri].abiAllocations.size();
    r.quantities["abi_bytes"] = f.invocation.regions[ri].abiBytes;
    r.quantities["task_count"] = scope.sourceTasks.size();
    for (auto ei : r.events) {
      if (ei >= g.events.size() || seen[ei])
        return fail(e, "region body coverage must be a partition");
      seen[ei] = true;
      const auto &event = g.events[ei];
      int64_t demand = 1;
      for (const auto &i : g.instances)
        if (i.event == ei)
          demand = i.resourceCount;
      r.resourceDemands[event.resource] =
          std::max(r.resourceDemands[event.resource], demand);
      if (event.kind == "transfer") {
        r.quantities["transfer_count"] += 1;
        r.quantities["requested_transfer_bytes"] += event.transferBytes;
      } else {
        auto row = counts.quantities[ei].getAsObject();
        auto operations =
                 row ? row->getNumber("source_operations") : std::nullopt,
             accesses =
                 row ? row->getNumber("source_memory_accesses") : std::nullopt;
        if (!operations || !accesses) {
          r.sourceWorkKnown = false;
          r.unknownReason += "source work counter unsupported; ";
        } else {
          r.quantities["source_operations"] += *operations;
          r.quantities["source_memory_accesses"] += *accesses;
        }
      }
    }
    for (size_t i = 0; i < g.regionEvents.size(); ++i)
      if (g.regionEvents[i].region == ri) {
        if (seenLifecycle[i])
          return fail(e, "duplicate lifecycle coverage");
        seenLifecycle[i] = true;
        r.regionEvents.push_back(i);
        r.resourceDemands[g.regionEvents[i].resource] =
            std::max<int64_t>(r.resourceDemands[g.regionEvents[i].resource], 1);
      }
    for (auto &q : r.quantities)
      if (!nonnegativeFinite(q.second))
        return fail(e, "feature overflow");
    f.regions.push_back(std::move(r));
  }
  if (llvm::is_contained(seen, false) ||
      llvm::is_contained(seenLifecycle, false))
    return fail(e, "body/lifecycle atom is outside exact regional ownership");
  f.identity = staticFingerprint(serializeProgramRegionCompositeFeatures(f));
  return true;
}
bool orbit::taskflow::buildProgramRegionCompositeCosts(
    const HeterogeneousMappingPlan &plan,
    const ProgramRegionCompositeFeatureSet &f,
    const ProgramRegionCompositeModel &m, ProgramTimingRequest &q, J &receipt,
    std::string &e) {
  q = {};
  receipt = nullptr;
  std::vector<ProgramRegionImplementation> bindings;
  for (const auto &r : f.regions)
    bindings.push_back(r.implementation);
  ProgramRegionCompositeFeatureSet expected;
  if (!collectProgramRegionCompositeFeatures(plan, m.runtime, bindings,
                                             expected, e))
    return false;
  if (staticFingerprint(serializeProgramRegionCompositeFeatures(f)) !=
      staticFingerprint(serializeProgramRegionCompositeFeatures(expected)))
    return fail(
        e, "features or runtime contract differ from checked source/graph");
  if (m.id.empty() || m.boundary.empty() ||
      (m.mode != "strict" && m.mode != "estimate") ||
      m.calibrationRuntimeIdentity != m.runtime.identity)
    return fail(e, "explicit matched model/runtime required");
  q.graphId = f.graphId;
  q.modelIdentity = m.id;
  q.boundary = m.boundary;
  A rows;
  bool known = true;
  std::set<std::string> assumptions;
  for (const auto &r : f.regions) {
    ProgramActivityCost c;
    c.id = id(r.region);
    c.implementation = r.implementation.implementation;
    c.events = r.events;
    c.regionEvents = r.regionEvents;
    c.resourceDemands = r.resourceDemands;
    c.boundary = "exclusive complete prepared region call: "
                 "dispatch/sync/setup/body/DMA/teardown/completion; "
                 "application reset/output copy excluded";
    c.provenance = r.implementation.provenance +
                   "; control=" + r.implementation.controlSequenceIdentity +
                   "; code=" + r.implementation.codeIdentity;
    double ns = 0;
    std::string unknown = r.unknownReason;
    const ProgramRegionCompositeCalibration *row = nullptr;
    for (const auto &candidate : m.regions)
      if (candidate.implementation == r.implementation.implementation &&
          candidate.sourceFamily == r.sourceFamily &&
          candidate.controlFamily == r.implementation.controlFamily) {
        if (row)
          return fail(e, "ambiguous calibration row");
        row = &candidate;
      }
    if (!row)
      unknown += "no semantic/implementation/control-family calibrated model; ";
    else if (row->provenance.empty())
      return fail(e, "calibration provenance required");
    else
      for (const auto &key : featureNames) {
        auto quantity = r.quantities.at(key);
        auto range = row->domain.find(key);
        if (range == row->domain.end() ||
            !nonnegativeFinite(range->second.minimum) ||
            !nonnegativeFinite(range->second.maximum) ||
            range->second.minimum > range->second.maximum ||
            quantity < range->second.minimum ||
            quantity > range->second.maximum)
          unknown +=
              "feature outside calibrated interpolation domain: " + key + "; ";
        auto found = row->coefficients.find(key);
        auto p = coefficient(found == row->coefficients.end() ? nullptr
                                                              : &found->second,
                             unit(key), m.mode, key, unknown);
        if (p) {
          ns += quantity * *p;
          if (found->second.kind == "assumed")
            assumptions.insert(key + ": " + found->second.source);
        }
      }
    if (!nonnegativeFinite(ns))
      return fail(e, "regional cost overflow");
    if (unknown.empty())
      c.durationMs = ns / 1e6;
    else {
      c.unknownReason = unknown;
      known = false;
    }
    c.lowerMs = 0;
    rows.push_back(
        O{{"activity", c.id},
          {"region", int64_t(r.region)},
          {"predicted_ms", c.durationMs ? J(*c.durationMs) : J(nullptr)},
          {"known_affine_subtotal_ns", ns},
          {"unknown_reason", unknown},
          {"calibration_provenance", row ? row->provenance : ""}});
    q.costs.push_back(std::move(c));
  }
  auto global = [&](const std::string &name, int64_t buffers, int64_t bytes) {
    ProgramActivityCost c;
    c.id = "invocation." + name;
    c.runtimeOnly = true;
    c.resourceDemands[m.runtime.hostWorkerResource] = 1;
    c.boundary =
        "global application " + name + "; exclusive region call excluded";
    c.provenance = "model=" + m.id + "; runtime=" + m.runtime.identity;
    double ns = 0;
    std::string unknown;
    for (auto t : {std::make_pair(std::string("per_call"), int64_t(1)),
                   std::make_pair(std::string("per_bo"), buffers),
                   std::make_pair(std::string("per_byte"), bytes)}) {
      if (t.second == 0)
        continue;
      auto key = "invocation." + name + "." + t.first;
      auto found = m.parameters.find(key);
      auto p = coefficient(
          found == m.parameters.end() ? nullptr : &found->second,
          t.first == "per_byte" ? "ns/byte" : "ns", m.mode, key, unknown);
      if (p) {
        ns += t.second * *p;
        if (found->second.kind == "assumed")
          assumptions.insert(key + ": " + found->second.source);
      }
    }
    if (!nonnegativeFinite(ns))
      return fail(e, "global boundary cost overflow");
    if (unknown.empty())
      c.durationMs = ns / 1e6;
    else {
      c.unknownReason = unknown;
      known = false;
    }
    c.lowerMs = 0;
    if (name == "input_reset") {
      for (auto &region : q.costs)
        region.predecessors.push_back(c.id);
    } else {
      for (const auto &region : f.regions)
        c.predecessors.push_back(id(region.region));
      c.predecessors.push_back("invocation.control");
    }
    q.costs.push_back(std::move(c));
    return true;
  };
  if (m.runtime.resetInputs && !f.invocation.initialInputAllocations.empty() &&
      !global("input_reset", f.invocation.initialInputAllocations.size(),
              f.invocation.initialInputBytes))
    return false;
  // Region orchestration is scattered host work, aggregated once here for
  // the existing serial resident-region executor. It is not part of the timed
  // region spans and must not silently disappear or be absorbed into call fits.
  {
    ProgramActivityCost c;
    c.id = "invocation.control";
    c.runtimeOnly = true;
    c.resourceDemands[m.runtime.hostWorkerResource] = 1;
    c.boundary =
        "global host orchestration outside measured region/reset/output-copy "
        "spans; scattered work aggregated after serial regions";
    c.provenance = "model=" + m.id + "; runtime=" + m.runtime.identity;
    for (const auto &r : f.regions)
      c.predecessors.push_back(id(r.region));
    double ns = 0;
    std::string unknown;
    for (auto term : {std::make_pair(std::string("per_call"), double(1)),
                      std::make_pair(std::string("per_region"),
                                     double(f.regions.size()))}) {
      auto key = "invocation.control." + term.first;
      auto found = m.parameters.find(key);
      auto p = coefficient(
          found == m.parameters.end() ? nullptr : &found->second,
          term.first == "per_region" ? "ns/count" : "ns", m.mode, key, unknown);
      if (p) {
        ns += term.second * *p;
        if (found->second.kind == "assumed")
          assumptions.insert(key + ": " + found->second.source);
      }
    }
    if (!nonnegativeFinite(ns))
      return fail(e, "host control cost overflow");
    if (unknown.empty())
      c.durationMs = ns / 1e6;
    else {
      c.unknownReason = unknown;
      known = false;
    }
    c.lowerMs = 0;
    q.costs.push_back(std::move(c));
  }
  if (m.runtime.copyFinalOutputs &&
      !f.invocation.finalOutputAllocations.empty() &&
      !global("output_copy", f.invocation.finalOutputAllocations.size(),
              f.invocation.finalOutputBytes))
    return false;
  A assumed;
  for (auto &x : assumptions)
    assumed.push_back(x);
  receipt = O{{"schema", "orbit.program-region-composite-cost.v29"},
              {"status", known ? "complete-exclusive-region-model"
                               : "unknown-region-or-boundary-cost"},
              {"graph_id", f.graphId},
              {"features_id", f.identity},
              {"model_id", m.id},
              {"runtime_identity", m.runtime.identity},
              {"features", serializeProgramRegionCompositeFeatures(f)},
              {"regions", std::move(rows)},
              {"assumptions", std::move(assumed)},
              {"exclusive_body_and_lifecycle_ownership", true},
              {"additive_body_or_lifecycle_costs", false},
              {"control_identity_is_price_lookup_key", false},
              {"global_control_count", 1},
              {"global_control_regions", int64_t(f.regions.size())},
              {"global_control_scope",
               "scattered executor orchestration outside "
               "regional/reset/output-copy spans, aggregated after all serial "
               "resident regions and before output copy"},
              {"domain_policy",
               "closed per-feature calibrated ranges; no silent extrapolation"},
              {"scope",
               "complete prepared regional host call, including physical setup "
               "and teardown; application reset/output copy once outside "
               "regions; model prediction is not hardware optimality"}};
  return true;
}
bool orbit::taskflow::evaluateProgramRegionCompositeModel(
    const HeterogeneousMappingPlan &plan, const ProgramRegionCompositeModel &m,
    const std::vector<ProgramRegionImplementation> &bindings,
    ProgramTimingEvaluation &out, J &receipt, std::string &e) {
  ProgramRegionCompositeFeatureSet features;
  ProgramTimingRequest q;
  if (!collectProgramRegionCompositeFeatures(plan, m.runtime, bindings,
                                             features, e) ||
      !buildProgramRegionCompositeCosts(plan, features, m, q, receipt, e))
    return false;
  return evaluateProgramTiming(plan.graph, q, out, e);
}
