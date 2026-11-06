#include "Taskflow/ProgramMechanismCost.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <cmath>
#include <set>
using namespace orbit::taskflow;
namespace {
using O = llvm::json::Object;
using A = llvm::json::Array;
using J = llvm::json::Value;
bool fail(std::string &error, const std::string &message) {
  error = "program mechanism cost: " + message;
  return false;
}
struct Work { double operations = 0, accesses = 0; };
bool instanceWork(const ProgramGraph &graph, const MappingTaskInstance &instance,
                  Work &work, std::string &error) {
  if (instance.task >= graph.source.tasks.size())
    return fail(error, "instance has no source task");
  const auto &task = graph.source.tasks[instance.task];
  bool valid = true;
  task.operation->walk([&](mlir::Operation *operation) {
    auto name = operation->getName().getStringRef();
    bool memory = name == "affine.load" || name == "affine.store" ||
                  name == "memref.load" || name == "memref.store";
    bool arithmetic = (name.starts_with("arith.") || name.starts_with("math.")) &&
                      name != "arith.constant";
    if (!memory && !arithmetic) return;
    double multiplicity = 1;
    for (size_t index = 0; index < task.loops.size(); ++index) {
      const auto &loop = task.loops[index];
      if (!loop.operation->isProperAncestor(operation)) continue;
      int64_t lower = loop.lower, upper = loop.upper, step = loop.step;
      for (const auto &bound : instance.domain) if (bound.loop == index) {
        lower = bound.lower; upper = bound.upper; step = bound.step;
      }
      if (step <= 0 || upper <= lower) { valid = false; return; }
      multiplicity *= double((upper - lower - 1) / step + 1);
    }
    if (!std::isfinite(multiplicity)) { valid = false; return; }
    (memory ? work.accesses : work.operations) += multiplicity;
  });
  if (!valid || !std::isfinite(work.accesses + work.operations))
    return fail(error, "source operation multiplicity invalid or overflowing");
  return true;
}
}

bool orbit::taskflow::evaluateProgramMechanismCost(const ProgramGraph &graph,
    const ProgramMechanismModel &model, ProgramMechanismEvaluation &result,
    std::string &error) {
  result = {};
  if (!checkProgramGraph(graph, error)) return false;
  if (model.id.empty() || model.boundary.empty() ||
      (model.mode != "strict" && model.mode != "estimate"))
    return fail(error, "explicit model identity, boundary and strict/estimate mode required");
  for (const auto &entry : model.parameters) {
    const auto &p = entry.second;
    if (entry.first.empty() || p.source.empty() ||
        (p.kind != "calibrated" && p.kind != "architecture" && p.kind != "assumed") ||
        (p.unit != "ns" && p.unit != "bytes/ns") ||
        (p.value && (!std::isfinite(*p.value) || *p.value < 0)))
      return fail(error, "invalid parameter units, provenance or value: " + entry.first);
    result.parameters.push_back(O{{"key", entry.first},
        {"value", p.value ? J(*p.value) : J(nullptr)}, {"unit", p.unit},
        {"kind", p.kind}, {"source", p.source}});
  }
  ProgramTimingRequest request;
  request.graphId = programGraphIdentity(graph);
  request.modelIdentity = model.id;
  request.boundary = model.boundary;
  result.mode = model.mode;
  std::set<std::string> assumptions;
  auto eventId = [](size_t i) { return "event." + std::to_string(i); };
  auto groupId = [](size_t i) { return "group.configure." + std::to_string(i); };
  auto parameter = [&](const std::string &specific, const std::string &fallback,
                       llvm::StringRef unit, std::string &unknown)
      -> std::optional<double> {
    auto found = model.parameters.find(specific);
    if (found == model.parameters.end()) found = model.parameters.find(fallback);
    if (found == model.parameters.end() || !found->second.value) {
      unknown += "missing parameter " + specific + "; "; return {};
    }
    const auto &p = found->second;
    if (p.unit != unit) { unknown += "parameter unit mismatch " + found->first + "; "; return {}; }
    if (p.kind == "assumed") {
      if (model.mode == "strict") {
        unknown += "strict mode forbids assumed parameter " + found->first + "; "; return {};
      }
      assumptions.insert(found->first + ": " + p.source);
    }
    return p.value;
  };
  auto addTerm = [&](double quantity, const std::string &specific,
                     const std::string &fallback, double &subtotal,
                     std::string &unknown) {
    if (quantity == 0) return;
    auto p = parameter(specific, fallback, "ns", unknown);
    if (p) subtotal += quantity * *p;
  };
  auto finish = [&](ProgramActivityCost &cost, double ns, const std::string &unknown) {
    cost.boundary = model.boundary;
    cost.provenance = "backend-supplied per-activity mechanism parameters; mode=" + model.mode;
    cost.lowerMs = ns / 1e6;
    if (unknown.empty()) cost.durationMs = ns / 1e6;
    else cost.unknownReason = unknown;
  };
  for (size_t i = 0; i < graph.events.size(); ++i) {
    const auto &event = graph.events[i];
    ProgramActivityCost cost; cost.id = eventId(i); cost.events = {i};
    cost.resourceDemands[event.resource] = 1;
    double ns = 0; std::string unknown; Work work; bool workKnown = true;
    if (event.kind == "transfer") {
      auto start = parameter("link." + event.link + ".startup", "communication.startup", "ns", unknown);
      auto bandwidth = parameter("link." + event.link + ".bandwidth", "communication.bandwidth", "bytes/ns", unknown);
      if (start) ns += *start;
      if (bandwidth && *bandwidth > 0) ns += double(event.transferBytes) / *bandwidth;
      else if (bandwidth) return fail(error, "bandwidth must be positive when used");
    } else {
      auto found = std::find_if(graph.instances.begin(), graph.instances.end(),
          [&](const auto &instance) { return instance.event == i; });
      if (found == graph.instances.end()) return fail(error, "compute activity has no actual source instance");
      cost.resourceDemands[event.resource] = found->resourceCount;
      if (found->kind != "source-task" && found->domainTask != found->task) {
        workKnown = false;
        unknown += "initializer domain belongs to another source task; operation multiplicity mapping is not proved by this provider; ";
      } else if (!instanceWork(graph, *found, work, error)) return false;
      bool covered = std::find(model.implementations.begin(), model.implementations.end(),
                               found->implementation) != model.implementations.end();
      if (!covered)
        unknown += "implementation outside declared mechanism coverage: " + found->implementation + "; ";
      // These are logical source operation/access quantities of this actual
      // implementation. Backend coefficients include loop/address/local effects;
      // no instruction simulator or ideal vector speedup is inferred.
      if (covered) {
        addTerm(1, "compute." + found->implementation + ".startup", "compute.startup", ns, unknown);
        addTerm(work.operations, "compute." + found->implementation + ".operation", "compute.operation", ns, unknown);
        addTerm(work.accesses, "compute." + found->implementation + ".memory_access", "compute.memory_access", ns, unknown);
      }
    }
    finish(cost, ns, unknown);
    result.quantities.push_back(O{{"activity", cost.id}, {"event", int64_t(i)},
        {"source_operations", workKnown ? J(work.operations) : J(nullptr)},
        {"source_memory_accesses", workKnown ? J(work.accesses) : J(nullptr)},
        {"requested_bytes", event.transferBytes}, {"known_subtotal_ns", ns},
        {"unknown_reason", unknown}});
    request.costs.push_back(std::move(cost));
  }
  for (size_t i = 0; i < graph.regionEvents.size(); ++i) {
    const auto &event = graph.regionEvents[i];
    ProgramActivityCost cost; cost.id = "region." + std::to_string(i);
    cost.regionEvents = {i}; cost.resourceDemands[event.resource] = 1;
    std::string key = event.kind == "submit" ? "host.submit" :
                      event.kind == "complete" ? "host.complete" : "host.visibility";
    double ns = 0; std::string unknown; addTerm(1, key, key, ns, unknown);
    finish(cost, ns, unknown); request.costs.push_back(std::move(cost));
  }
  // A configuration is an explicit public execution-group setup activity.
  // It is ordered before its group members and after only the declared parent
  // groups, never an invented global chain across independent resources.
  for (size_t i = 0; i < graph.executionGroups.size(); ++i) {
    const auto &group = graph.executionGroups[i];
    ProgramActivityCost setup; setup.id = groupId(i); setup.runtimeOnly = true;
    setup.resourceDemands[group.resource] = 1;
    double ns = 0; std::string unknown;
    addTerm(1, "configuration." + group.resource + ".startup", "configuration.startup", ns, unknown);
    for (const auto *parents : {&group.dependencies,&group.implementationDependencies}) for (auto parent : *parents) {
      if (parent >= graph.executionGroups.size()) return fail(error, "invalid group dependency");
      for (auto event : graph.executionGroups[parent].memberEvents)
        setup.predecessors.push_back(eventId(event));
    }
    for (size_t event : group.memberEvents) {
      if (event >= graph.events.size()) return fail(error, "group member outside graph");
      setup.resourceDemands[graph.events[event].resource] = 1;
      request.costs[event].predecessors.push_back(setup.id);
    }
    // Submission precedes device-side configuration, including the first group.
    for (size_t region = 0; region < graph.regionEvents.size(); ++region) {
      const auto &lifecycle = graph.regionEvents[region];
      if (lifecycle.kind != "submit") continue;
      const auto &scope = graph.executionRegions[lifecycle.region];
      bool sharesMembers = std::any_of(group.memberEvents.begin(), group.memberEvents.end(),
          [&](size_t event) {
            return scope.memberEvents.empty() ?
                (scope.beginEvent <= event && event < scope.endEvent) :
                std::find(scope.memberEvents.begin(), scope.memberEvents.end(), event) != scope.memberEvents.end();
          });
      if (sharesMembers) setup.predecessors.push_back("region." + std::to_string(region));
    }
    finish(setup, ns, unknown); request.costs.push_back(std::move(setup));
  }
  StaticConfig architecture;architecture.domains=graph.domains;architecture.resources=graph.resources;architecture.memories=graph.memories;architecture.links=graph.links;
  for(auto &cost:request.costs){std::map<std::string,int64_t> expanded;if(!programResourceDemands(architecture,cost.resourceDemands,expanded,error))return false;cost.resourceDemands=std::move(expanded);}
  for (const auto &text : assumptions) result.assumptions.push_back(text);
  if (!evaluateProgramTiming(graph, request, result.timing, error)) return false;
  result.status = result.timing.makespanMs ?
      (assumptions.empty() ? "mechanism-estimate-with-declared-provenance" : "assumption-driven-estimate") :
      "unknown-required-mechanisms";
  return true;
}

J orbit::taskflow::serializeProgramMechanismCost(const ProgramMechanismEvaluation &result) {
  A assumptions; for (const auto &text : result.assumptions) assumptions.push_back(text);
  return O{{"schema", "orbit.program-mechanism-evaluation.v18"},
      {"mode", result.mode}, {"status", result.status},
      {"timing", serializeProgramTiming(result.timing)},
      {"quantities", A(result.quantities)}, {"parameters", A(result.parameters)},
      {"assumptions", std::move(assumptions)},
      {"precision_claim", "Resource-constrained analytical estimate, not validated physical DMA/core timing. Logical transfer startup may differ from coalesced backend BD startup."}};
}
