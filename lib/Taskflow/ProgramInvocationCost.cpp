#include "Taskflow/ProgramInvocationCost.h"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <cmath>
#include <set>
using namespace orbit::taskflow;
namespace {
using J = llvm::json::Value;
using O = llvm::json::Object;
using A = llvm::json::Array;
bool fail(std::string &e, const std::string &s) {
  e = "program invocation cost: " + s;
  return false;
}
bool add(int64_t &n, int64_t x) {
  if (x < 0 || n > INT64_MAX - x)
    return false;
  n += x;
  return true;
}
A ids(const std::vector<size_t> &v) {
  A a;
  for (auto x : v)
    a.push_back(int64_t(x));
  return a;
}
const StaticMemory *memory(const ProgramGraph &g, const std::string &id) {
  auto i = llvm::find_if(g.memories, [&](auto &m) { return m.id == id; });
  return i == g.memories.end() ? nullptr : &*i;
}
const StaticResource *resource(const ProgramGraph &g, const std::string &id) {
  auto i = llvm::find_if(g.resources, [&](auto &r) { return r.id == id; });
  return i == g.resources.end() ? nullptr : &*i;
}
std::string chip(const ProgramGraph &g, std::string owner) {
  std::set<std::string> seen;
  while (!owner.empty() && seen.insert(owner).second) {
    auto d = llvm::find_if(g.domains, [&](auto &d) { return d.id == owner; });
    if (d == g.domains.end())
      return {};
    if (d->kind == "device")
      return owner;
    owner = d->parent;
  }
  return {};
}
std::string activity(size_t event) {
  return "invocation.lifecycle." + std::to_string(event);
}
bool allocations(const ProgramGraph &g, const std::vector<size_t> &copies,
                 const std::string &endpoint, std::vector<size_t> &out,
                 std::string &e) {
  std::set<size_t> unique;
  for (auto ci : copies) {
    if (ci >= g.copies.size())
      return fail(e, "ABI copy outside graph");
    auto ai = g.copies[ci].allocation;
    const auto &a = g.allocations[ai];
    auto m = memory(g, a.memory);
    if (!m || m->kind != "system-bo" || a.memory != endpoint)
      return fail(
          e, "invocation ABI must reference its declared system-BO endpoint");
    unique.insert(ai);
  }
  out.assign(unique.begin(), unique.end());
  return true;
}
bool sum(const ProgramGraph &g, const std::vector<size_t> &allocations,
         int64_t &bytes, std::string &e) {
  bytes = 0;
  for (auto ai : allocations) {
    const auto &a = g.allocations[ai];
    auto type = mlir::dyn_cast<mlir::MemRefType>(
        g.source.storages[a.storage].root.getType());
    if (!type || !type.hasStaticShape() ||
        !type.getElementType().isIntOrFloat())
      return fail(e, "ABI buffer requires static scalar element byte size");
    int64_t full = (type.getElementTypeBitWidth() + 7) / 8;
    for (auto n : type.getShape())
      if (n <= 0 || full > INT64_MAX / n)
        return fail(e, "ABI storage byte size overflow");
      else
        full *= n;
    if (a.bytes != full)
      return fail(e, "partial/reusable host allocation needs an explicit "
                     "physical ABI refinement");
    if (!add(bytes, a.bytes))
      return fail(e, "ABI byte sum overflow");
  }
  return true;
}
} // namespace
J orbit::taskflow::serializeProgramInvocationFeatures(
    const ProgramInvocationFeatures &f) {
  A regions;
  for (const auto &r : f.regions)
    regions.push_back(
        O{{"region", int64_t(r.region)},
          {"device", r.device},
          {"submissions", r.submissions},
          {"submit", int64_t(r.submit)},
          {"complete", int64_t(r.complete)},
          {"visible_to_device", r.visibleToDevice == SIZE_MAX
                                    ? J(nullptr)
                                    : J(int64_t(r.visibleToDevice))},
          {"visible_to_host", r.visibleToHost == SIZE_MAX
                                  ? J(nullptr)
                                  : J(int64_t(r.visibleToHost))},
          {"input_allocations", ids(r.inputAllocations)},
          {"output_allocations", ids(r.outputAllocations)},
          {"abi_allocations", ids(r.abiAllocations)},
          {"scratch_allocations", ids(r.scratchAllocations)},
          {"to_device_allocations", ids(r.toDeviceAllocations)},
          {"abi_bytes", r.abiBytes},
          {"to_device_bytes", r.toDeviceBytes},
          {"from_device_bytes", r.fromDeviceBytes},
          {"unbound_staging_bytes", r.stagingBytes}});
  return O{
      {"schema", "orbit.program-invocation-features.v28"},
      {"identity", f.identity},
      {"graph_id", f.graphId},
      {"runtime", O{{"identity", f.runtime.identity},
                    {"state", f.runtime.state},
                    {"host_worker_resource", f.runtime.hostWorkerResource},
                    {"bound_buffers", f.runtime.boundBuffers},
                    {"stage_outputs_to_device", f.runtime.stageOutputsToDevice},
                    {"reset_inputs", f.runtime.resetInputs},
                    {"copy_final_outputs", f.runtime.copyFinalOutputs}}},
      {"regions", std::move(regions)},
      {"initial_input_allocations", ids(f.initialInputAllocations)},
      {"final_output_allocations", ids(f.finalOutputAllocations)},
      {"initial_input_bytes", f.initialInputBytes},
      {"final_output_bytes", f.finalOutputBytes}};
}
bool orbit::taskflow::collectProgramInvocationFeatures(
    const ProgramGraph &g, const ProgramInvocationRuntimeContract &runtime,
    ProgramInvocationFeatures &f, std::string &e) {
  f = {};
  if (g.operatorSource)
    return fail(
        e,
        "external implementation cover owns its invocation/transport contract");
  if (!checkProgramGraph(g, e))
    return false;
  if (runtime.identity.empty() || runtime.identity == "unbound" ||
      runtime.state != "prepared-hot" || g.executionRegions.empty() ||
      g.regionEvents.empty())
    return fail(e, "explicit prepared-hot runtime identity and refined "
                   "invocation regions required");
  auto worker = resource(g, runtime.hostWorkerResource);
  if (!worker || worker->domain.empty() || worker->kind != "compute" ||
      worker->capacity <= 0 || !chip(g, worker->domain).empty())
    return fail(e, "runtime worker must be a declared host compute resource");
  f.runtime = runtime;
  f.graphId = programGraphIdentity(g);
  for (size_t ri = 0; ri < g.executionRegions.size(); ++ri) {
    const auto &r = g.executionRegions[ri];
    ProgramInvocationRegionFeatures x;
    x.region = ri;
    x.submissions = r.submissionCount;
    if (x.submissions != 1)
      return fail(e, "one exact invocation per lifecycle region required; "
                     "split calls into explicit regions");
    if (r.invocationLifecycle == "host-device")
      x.device = true;
    else if (r.invocationLifecycle == "host-only")
      x.device = false;
    else if (r.invocationLifecycle.empty() &&
             (r.placement == "cpu" || r.placement == "npu"))
      x.device = r.placement == "npu";
    else
      return fail(e, "region has no supported invocation lifecycle");
    for (size_t ei = 0; ei < g.regionEvents.size(); ++ei) {
      const auto &v = g.regionEvents[ei];
      if (v.region != ri)
        continue;
      if (v.kind == "submit")
        x.submit = ei;
      else if (v.kind == "complete")
        x.complete = ei;
      else if (v.kind == "visible-to-device")
        x.visibleToDevice = ei;
      else if (v.kind == "visible-to-host")
        x.visibleToHost = ei;
      else
        return fail(e, "unknown lifecycle activity");
    }
    if (x.submit == SIZE_MAX || x.complete == SIZE_MAX ||
        (x.device &&
         (x.visibleToDevice == SIZE_MAX || x.visibleToHost == SIZE_MAX)))
      return fail(e, "incomplete invocation lifecycle");
    if (!allocations(g, r.inputCopies, r.endpointMemory, x.inputAllocations,
                     e) ||
        !allocations(g, r.outputCopies, r.endpointMemory, x.outputAllocations,
                     e))
      return false;
    std::set<size_t> abi(x.inputAllocations.begin(), x.inputAllocations.end());
    abi.insert(x.outputAllocations.begin(), x.outputAllocations.end());
    x.abiAllocations.assign(abi.begin(), abi.end());
    if (!sum(g, x.abiAllocations, x.abiBytes, e))
      return false;
    std::set<std::string> chips;
    auto members = r.memberEvents;
    if (members.empty())
      for (size_t i = r.beginEvent; i < r.endEvent; ++i)
        members.push_back(i);
    for (auto ei : members) {
      const auto &ev = g.events[ei];
      if (ev.kind == "compute" || ev.kind == "initialize") {
        auto rr = resource(g, ev.resource);
        if (!rr || rr->domain.empty())
          return fail(e, "compute resource lacks an explicit ownership domain");
        auto owner = chip(g, rr->domain);
        if (x.device == owner.empty())
          return fail(
              e, "CPU/device task is assigned to the wrong invocation region");
        if (!owner.empty())
          chips.insert(owner);
      } else if (ev.kind == "transfer") {
        std::set<std::string> endpoints;
        for (const auto *list : {&ev.reads, &ev.writes})
          for (auto ci : *list) {
            const auto &a = g.allocations[g.copies[ci].allocation];
            auto mm = memory(g, a.memory);
            if (mm && mm->kind != "system-bo") {
              auto owner = chip(g, mm->domain);
              if (owner.empty())
                return fail(e, "local transfer endpoint lacks a device owner");
              endpoints.insert(owner);
              chips.insert(owner);
            }
          }
        if (endpoints.size() > 1)
          return fail(
              e, "cross-chip local transfer has no invocation implementation");
        if (!x.device && !endpoints.empty())
          return fail(e, "CPU invocation owns device-local movement");
      }
    }
    if (x.device && chips.size() != 1)
      return fail(e, "one physical device per invocation region required");
    // Device-internal host hand-offs also occupy caller-bound BO arguments.
    // Prepared-hot scratch is initialized by graph producers each invocation;
    // its one-time BO initialization is outside this hot boundary. It still
    // contributes to launch argument binding, never to application live-in
    // data.
    if (x.device) {
      std::set<size_t> all(x.abiAllocations.begin(), x.abiAllocations.end());
      for (auto ei : members)
        for (const auto *list : {&g.events[ei].reads, &g.events[ei].writes})
          for (auto ci : *list) {
            auto ai = g.copies[ci].allocation;
            const auto &a = g.allocations[ai];
            auto mm = memory(g, a.memory);
            if (mm && mm->kind == "system-bo") {
              if (a.memory != r.endpointMemory)
                return fail(
                    e,
                    "device invocation host scratch uses another ABI endpoint");
              if (all.insert(ai).second)
                x.scratchAllocations.push_back(ai);
            }
          }
      x.abiAllocations.assign(all.begin(), all.end());
      std::sort(x.scratchAllocations.begin(), x.scratchAllocations.end());
      if (!sum(g, x.abiAllocations, x.abiBytes, e))
        return false;
    }
    if (x.device)
      for (auto ei : members)
        for (auto ci : g.events[ei].reads)
          if (llvm::is_contained(x.scratchAllocations,
                                 g.copies[ci].allocation)) {
            auto creator = g.copies[ci].createdBy;
            if (creator < 0 || !llvm::is_contained(members, size_t(creator)))
              return fail(
                  e, "scratch live-in is missing from the explicit region ABI");
          }
    if (x.device) {
      x.toDeviceAllocations = x.inputAllocations;
      if (runtime.stageOutputsToDevice) {
        std::set<size_t> all(x.toDeviceAllocations.begin(),
                             x.toDeviceAllocations.end());
        all.insert(x.outputAllocations.begin(), x.outputAllocations.end());
        x.toDeviceAllocations.assign(all.begin(), all.end());
      }
      if (!sum(g, x.toDeviceAllocations, x.toDeviceBytes, e) ||
          !sum(g, x.outputAllocations, x.fromDeviceBytes, e))
        return false;
      if (!runtime.boundBuffers)
        x.stagingBytes = x.toDeviceBytes;
    }
    for (auto ai : x.abiAllocations) {
      auto mm = memory(g, g.allocations[ai].memory);
      bool accessible =
          !mm->accessibleResources.empty()
              ? llvm::is_contained(mm->accessibleResources,
                                   runtime.hostWorkerResource)
              : llvm::is_contained(mm->accessibleDevices, worker->device);
      if (!accessible)
        return fail(
            e, "declared runtime worker cannot access a bound/staged ABI BO");
    }
    f.regions.push_back(std::move(x));
  }
  std::set<size_t> initial, final;
  for (const auto &c : g.copies) {
    const auto &a = g.allocations[c.allocation];
    const auto &s = g.source.storages[a.storage];
    auto m = memory(g, a.memory);
    if (!m || m->kind != "system-bo")
      continue;
    if (c.createdBy < 0 && !c.users.empty() && s.role == "input")
      initial.insert(c.allocation);
    auto location = g.outputLocations.find(a.storage);
    if (s.role == "output" && location != g.outputLocations.end() &&
        location->second == a.memory && c.releasedAfter == g.events.size())
      final.insert(c.allocation);
  }
  f.initialInputAllocations.assign(initial.begin(), initial.end());
  f.finalOutputAllocations.assign(final.begin(), final.end());
  if (!sum(g, f.initialInputAllocations, f.initialInputBytes, e) ||
      !sum(g, f.finalOutputAllocations, f.finalOutputBytes, e))
    return false;
  f.identity = staticFingerprint(serializeProgramInvocationFeatures(f));
  return true;
}
bool orbit::taskflow::buildProgramInvocationCosts(
    const ProgramGraph &g, const ProgramInvocationFeatures &f,
    const ProgramMechanismModel &m, const std::string &modelRuntimeIdentity,
    std::vector<ProgramActivityCost> &costs, J &receipt, std::string &e) {
  costs.clear();
  receipt = nullptr;
  ProgramInvocationFeatures expected;
  if (!collectProgramInvocationFeatures(g, f.runtime, expected, e))
    return false;
  if (staticFingerprint(serializeProgramInvocationFeatures(f)) !=
      staticFingerprint(serializeProgramInvocationFeatures(expected)))
    return fail(
        e, "invocation features changed or refer to another graph/runtime");
  if (modelRuntimeIdentity != f.runtime.identity)
    return fail(
        e, "parameter runtime identity differs from exact invocation contract");
  if (m.id.empty() || m.boundary.empty() ||
      (m.mode != "strict" && m.mode != "estimate"))
    return fail(e,
                "model identity, boundary and strict/estimate mode required");
  for (auto &[key, p] : m.parameters)
    if (key.empty() || p.source.empty() ||
        (p.kind != "calibrated" && p.kind != "architecture" &&
         p.kind != "assumed") ||
        (p.unit != "ns" && p.unit != "bytes/ns" && p.unit != "ns/byte") ||
        (p.value && (!std::isfinite(*p.value) || *p.value < 0)))
      return fail(e, "invalid parameter: " + key);
  A rows;
  std::set<std::string> assumptions;
  bool known = true;
  auto price = [&](ProgramActivityCost &c, const std::string &prefix,
                   int64_t calls, int64_t buffers, int64_t bytes, double &ns,
                   std::string &unknown) {
    for (auto term : {std::make_pair(std::string("per_call"), calls),
                      std::make_pair(std::string("per_bo"), buffers),
                      std::make_pair(std::string("per_byte"), bytes)}) {
      if (term.second == 0)
        continue;
      std::string key = "invocation." + prefix + "." + term.first;
      auto p = m.parameters.find(key);
      std::string unit = term.first == "per_byte" ? "ns/byte" : "ns";
      if (p == m.parameters.end() || !p->second.value)
        unknown += "missing " + key + "; ";
      else if (p->second.unit != unit)
        unknown += "unit mismatch " + key + "; ";
      else if (m.mode == "strict" && p->second.kind == "assumed")
        unknown += "strict mode forbids assumed " + key + "; ";
      else {
        ns += double(term.second) * *p->second.value;
        if (p->second.kind == "assumed")
          assumptions.insert(key + ": " + p->second.source);
      }
    }
    rows.push_back(O{{"activity", c.id},
                     {"mechanism", prefix},
                     {"calls", calls},
                     {"buffers", buffers},
                     {"bytes", bytes}});
  };
  auto finish = [&](ProgramActivityCost &c, double ns,
                    const std::string &unknown) {
    if (!std::isfinite(ns) || ns < 0)
      return fail(e, "invocation cost overflow");
    c.boundary = "prepared invocation API and physical device configuration "
                 "work; source body excluded";
    c.provenance = "runtime=" + f.runtime.identity + "; model=" + m.id +
                   "; mode=" + m.mode;
    c.lowerMs = ns / 1e6;
    if (unknown.empty())
      c.durationMs = ns / 1e6;
    else {
      c.unknownReason = unknown;
      known = false;
    }
    costs.push_back(std::move(c));
    return true;
  };
  for (size_t ei = 0; ei < g.regionEvents.size(); ++ei) {
    const auto &event = g.regionEvents[ei];
    const auto &r = f.regions[event.region];
    ProgramActivityCost c;
    c.id = activity(ei);
    c.regionEvents = {ei};
    c.resourceDemands[event.resource] = 1;
    c.resourceDemands[f.runtime.hostWorkerResource] = 1;
    double ns = 0;
    std::string unknown;
    if (event.kind == "visible-to-device") {
      if (!f.runtime.boundBuffers)
        price(c, "device.stage", 1, r.toDeviceAllocations.size(),
              r.stagingBytes, ns, unknown);
      price(c, "device.to_device", 1, r.toDeviceAllocations.size(),
            r.toDeviceBytes, ns, unknown);
    } else if (event.kind == "visible-to-host")
      price(c, "device.from_device", 1, r.outputAllocations.size(),
            r.fromDeviceBytes, ns, unknown);
    else if (event.kind == "submit")
      price(c, r.device ? "device.submit" : "host.submit", r.submissions,
            r.abiAllocations.size(), 0, ns, unknown);
    else if (event.kind == "complete")
      price(c, r.device ? "device.complete" : "host.complete", r.submissions, 0,
            0, ns, unknown);
    else
      return fail(e, "unsupported lifecycle cost kind");
    if (!finish(c, ns, unknown))
      return false;
  }
  // Physical invocation configuration is separate from logical group setup.
  // Preparation creates the ELF/context/BOs; it does not remove firmware work
  // embedded in the runtime sequence. The wrapper connects source body costs
  // after setup and before teardown, then complete after teardown.
  int64_t configuredRuntimeCount = 0;
  for (const auto &r : f.regions)
    if (r.device) {
      ++configuredRuntimeCount;
      std::map<std::string, int64_t> demands;
      const auto &region = g.executionRegions[r.region];
      auto members = region.memberEvents;
      if (members.empty())
        for (size_t i = region.beginEvent; i < region.endEvent; ++i)
          members.push_back(i);
      for (auto ei : members) {
        const auto &ev = g.events[ei];
        auto rr = resource(g, ev.resource);
        if (rr && !chip(g, rr->domain).empty() &&
            (rr->kind == "compute" || rr->kind == "transfer"))
          demands[rr->id] = 1;
      }
      if (demands.empty())
        return fail(
            e, "physical device setup has no owned compute/transfer resource");
      for (bool teardown : {false, true}) {
        ProgramActivityCost c;
        c.id = std::string(teardown ? "invocation.device_teardown."
                                    : "invocation.device_setup.") +
               std::to_string(r.region);
        c.runtimeOnly = true;
        c.resourceDemands = demands;
        c.predecessors = {teardown ? "invocation.device_setup." +
                                         std::to_string(r.region)
                                   : activity(r.submit)};
        double ns = 0;
        std::string unknown;
        price(c, teardown ? "device.teardown" : "device.setup", 1, 0, 0, ns,
              unknown);
        if (!finish(c, ns, unknown))
          return false;
      }
    }
  if (f.runtime.resetInputs && !f.initialInputAllocations.empty()) {
    ProgramActivityCost c;
    c.id = "invocation.input_reset";
    c.runtimeOnly = true;
    c.resourceDemands[f.runtime.hostWorkerResource] = 1;
    double ns = 0;
    std::string unknown;
    price(c, "input_reset", 1, f.initialInputAllocations.size(),
          f.initialInputBytes, ns, unknown);
    if (!finish(c, ns, unknown))
      return false;
    for (auto &lifecycle : costs)
      if (!lifecycle.regionEvents.empty())
        lifecycle.predecessors.push_back("invocation.input_reset");
  }
  if (f.runtime.copyFinalOutputs && !f.finalOutputAllocations.empty()) {
    ProgramActivityCost c;
    c.id = "invocation.output_copy";
    c.runtimeOnly = true;
    c.resourceDemands[f.runtime.hostWorkerResource] = 1;
    for (const auto &region : f.regions)
      c.predecessors.push_back(
          activity(region.device ? region.visibleToHost : region.complete));
    double ns = 0;
    std::string unknown;
    price(c, "output_copy", 1, f.finalOutputAllocations.size(),
          f.finalOutputBytes, ns, unknown);
    if (!finish(c, ns, unknown))
      return false;
  }
  A assumed;
  for (auto &s : assumptions)
    assumed.push_back(s);
  receipt =
      O{{"schema", "orbit.program-invocation-cost.v28"},
        {"graph_id", f.graphId},
        {"features_id", f.identity},
        {"runtime_identity", f.runtime.identity},
        {"model_id", m.id},
        {"status", known ? "complete-host-boundary-estimate"
                         : "unknown-required-host-mechanisms"},
        {"features", serializeProgramInvocationFeatures(f)},
        {"quantities", std::move(rows)},
        {"assumptions", std::move(assumed)},
        {"body_events_owned", false},
        {"logical_group_configuration_charged", false},
        {"configured_runtime_count", configuredRuntimeCount},
        {"physical_setup_activities", configuredRuntimeCount},
        {"physical_teardown_activities", configuredRuntimeCount},
        {"configuration_scope",
         "one explicit physical setup and teardown per device invocation; "
         "logical execution groups do not imply additional configurations"},
        {"configuration_phase_scope",
         "setup follows submit and precedes all region body costs; teardown "
         "follows all region body costs and precedes complete; wrapper "
         "supplies body and complete edges"},
        {"wait_scope",
         "complete prices only extra host polling/return overhead after device "
         "events; never full submit-to-completion wall time"},
        {"allocation_context_scope",
         "context, ELF, BO creation and initial scratch BO initialization "
         "excluded by prepared-hot contract"}};
  return true;
}
