#include "Taskflow/ProgramTiming.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
using namespace orbit::taskflow;
namespace {
bool fail(std::string &e, const std::string &s) {
  e = "program timing: " + s;
  return false;
}
bool valid(double x) { return std::isfinite(x) && x >= 0; }
using A = llvm::json::Array;
using O = llvm::json::Object;
using J = llvm::json::Value;
J opt(std::optional<double> x) { return x ? J(*x) : J(nullptr); }
A ids(const std::vector<size_t> &x) {
  A a;
  for (auto n : x)
    a.push_back(int64_t(n));
  return a;
}
A strings(const std::vector<std::string> &x) {
  A a;
  for (auto &n : x)
    a.push_back(n);
  return a;
}
} // namespace
bool orbit::taskflow::evaluateProgramTiming(const ProgramGraph &g,
                                            const ProgramTimingRequest &q,
                                            ProgramTimingEvaluation &r,
                                            std::string &error) {
  r = {};
  if (!checkProgramGraph(g, error))
    return false;
  if (q.graphId != programGraphIdentity(g) || q.modelIdentity.empty() ||
      q.boundary.empty())
    return fail(error, "graph identity/model/boundary required");
  r.graphId = q.graphId;
  r.modelIdentity = q.modelIdentity;
  r.boundary = q.boundary;
  r.allPhysicalCapacitiesKnown = true;
  for (auto &m : g.memories)
    if (!m.capacityKnown) {
      r.allPhysicalCapacitiesKnown = false;
      r.unknowns.push_back("physical memory capacity unknown: " + m.id);
    }
  auto costs = q.costs;
  size_t n = g.events.size(), atoms = n + g.regionEvents.size();
  std::vector<size_t> owner(atoms, SIZE_MAX),
      invocationOwner(g.executionRegions.size(), SIZE_MAX);
  std::map<std::string, size_t> names;
  std::map<std::string, int64_t> capacities;
  for (auto &x : g.resources)
    capacities[x.id] = x.capacity;
  for (size_t i = 0; i < costs.size(); ++i) {
    auto &c = costs[i];
    if (c.id.empty() || !names.emplace(c.id, i).second || c.boundary.empty() ||
        c.provenance.empty() ||
        (c.events.empty() && c.regionEvents.empty()) != c.runtimeOnly)
      return fail(
          error,
          "invalid cost identity, provenance, boundary or empty coverage");
    for (auto x : {c.durationMs, c.lowerMs, c.upperMs})
      if (x && !valid(*x))
        return fail(error, "invalid nonnegative finite duration");
    double lo = c.lowerMs.value_or(c.durationMs.value_or(0));
    auto hi = c.upperMs ? c.upperMs : c.durationMs;
    if ((hi && lo > *hi) ||
        (c.durationMs && (*c.durationMs < lo || (hi && *c.durationMs > *hi))))
      return fail(error, "inconsistent duration bounds");
    if (!c.durationMs && c.unknownReason.empty())
      return fail(error, "unknown duration requires reason");
    if (c.resourceDemands.empty())
      return fail(error, "cost needs explicit resource reservations");
    for (auto &d : c.resourceDemands)
      if (!capacities.count(d.first) || d.second <= 0 ||
          d.second > capacities[d.first])
        return fail(error, "unknown or over-capacity resource demand");
    for (bool region : {false, true})
      for (auto x : region ? c.regionEvents : c.events) {
        if (x >= (region ? g.regionEvents.size() : n))
          return fail(error, "coverage outside graph");
        size_t a = (region ? n : 0) + x;
        if (owner[a] != SIZE_MAX)
          return fail(error, "duplicate cost coverage");
        const auto &resource =
            region ? g.regionEvents[x].resource : g.events[x].resource;
        int64_t demand = 1;
        if (!region)
          for (auto &instance : g.instances)
            if (instance.event == x)
              demand = instance.resourceCount;
        auto reservation = c.resourceDemands.find(resource);
        if (reservation == c.resourceDemands.end() ||
            reservation->second < demand)
          return fail(error, "cost omits covered event resource reservation");
        owner[a] = i;
      }
  }
  for (size_t i = 0; i < costs.size(); ++i)
    for (auto region : costs[i].executionRegions) {
      if (!g.regionEvents.empty() || region >= g.executionRegions.size() ||
          invocationOwner[region] != SIZE_MAX)
        return fail(error,
                    "invalid or duplicate unrefined invocation coverage");
      auto &scope = g.executionRegions[region];
      if (scope.submissionCount <= 0)
        return fail(error,
                    "invocation coverage refers to non-submitted region");
      auto demand = costs[i].resourceDemands.find(scope.submissionResource);
      if (demand == costs[i].resourceDemands.end() || demand->second < 1)
        return fail(error, "invocation composite omits submission resource");
      auto members = scope.memberEvents;
      if (members.empty())
        for (size_t j = scope.beginEvent; j < scope.endEvent; ++j)
          members.push_back(j);
      for (auto member : members)
        if (owner[member] != i)
          return fail(error,
                      "invocation composite must own all region members");
      invocationOwner[region] = i;
    }
  if (g.regionEvents.empty())
    for (size_t region = 0; region < g.executionRegions.size(); ++region)
      if (g.executionRegions[region].submissionCount > 0 &&
          invocationOwner[region] == SIZE_MAX) {
        ProgramActivityCost c;
        c.id = "unpriced.invocation." + std::to_string(region);
        if (names.count(c.id))
          return fail(error, "reserved invocation identity collision");
        names[c.id] = costs.size();
        c.runtimeOnly = true;
        c.boundary = "unrefined-submit-and-completion";
        c.provenance = "checked-program-graph";
        c.unknownReason = "native invocation boundary has no exclusive "
                          "whole-region implementation cost";
        c.resourceDemands[g.executionRegions[region].submissionResource] = 1;
        costs.push_back(c);
      }
  for (size_t a = 0; a < atoms; ++a)
    if (owner[a] == SIZE_MAX) {
      ProgramActivityCost c;
      c.id = "unpriced." + std::to_string(a);
      if (names.count(c.id))
        return fail(error, "reserved unpriced identity collision");
      names[c.id] = costs.size();
      owner[a] = costs.size();
      c.boundary = "unpriced-public-event";
      c.provenance = "checked-program-graph";
      c.unknownReason = "no implementation cost covers this event";
      if (a < n) {
        c.events = {a};
        int64_t demand = 1;
        for (auto &instance : g.instances)
          if (instance.event == a)
            demand = instance.resourceCount;
        c.resourceDemands[g.events[a].resource] = demand;
      } else {
        c.regionEvents = {a - n};
        c.resourceDemands[g.regionEvents[a - n].resource] = 1;
      }
      costs.push_back(c);
    }
  StaticConfig architecture;
  architecture.domains = g.domains;
  architecture.resources = g.resources;
  architecture.memories = g.memories;
  architecture.links = g.links;
  for (auto &cost : costs) {
    std::map<std::string, int64_t> expanded;
    if (!programResourceDemands(architecture, cost.resourceDemands, expanded,
                                error))
      return false;
    cost.resourceDemands = std::move(expanded);
  }
  std::vector<std::set<size_t>> deps(costs.size());
  auto edge = [&](size_t a, size_t b) {
    if (owner[a] != owner[b])
      deps[owner[b]].insert(owner[a]);
  };
  for (size_t a = 0; a < n; ++a)
    for (auto p : g.events[a].dependencies)
      edge(p, a);
  // Respect the original schedule's chosen serialization when requested. The
  // original semantic dependency list is never mutated by evaluation.
  if (q.serialize)
    for (size_t a = 1; a < n; ++a)
      edge(a - 1, a);
  for (size_t a = 0; a < g.regionEvents.size(); ++a) {
    auto &e = g.regionEvents[a];
    for (auto p : e.dependencies)
      edge(n + p, n + a);
    auto &region = g.executionRegions[e.region];
    auto members = region.memberEvents;
    if (members.empty())
      for (size_t j = region.beginEvent; j < region.endEvent; ++j)
        members.push_back(j);
    if (e.kind == "submit")
      for (auto x : members)
        edge(n + a, x);
    if (e.kind == "complete")
      for (auto x : members)
        edge(x, n + a);
  }
  // All internal buffer edges contract to a self-edge when a validated
  // exclusive implementation owns every graph atom. Enumerating copy pairs
  // would only construct a quadratic relation that the owner projection drops.
  // Additional runtime-only activities do not own graph copies or atoms.
  const bool singleGraphOwner =
      !owner.empty() &&
      llvm::all_of(owner, [&](size_t x) { return x == owner.front(); });
  // Across owners preserve the full original physical buffer-reuse relation.
  if (!singleGraphOwner &&
      !collectProgramBufferOrdering(g, r.bufferOrdering, error))
    return false;
  std::vector<std::set<size_t>> bufferDeps(costs.size());
  for (const auto &order : r.bufferOrdering)
    if (owner[order.predecessor] != owner[order.successor]) {
      deps[owner[order.successor]].insert(owner[order.predecessor]);
      bufferDeps[owner[order.successor]].insert(owner[order.predecessor]);
    }
  for (const auto &group : g.executionGroups)
    for (const auto *parents :
         {&group.dependencies, &group.implementationDependencies})
      for (auto p : *parents)
        for (auto before : g.executionGroups[p].memberEvents)
          for (auto after : group.memberEvents)
            edge(before, after);
  for (size_t i = 0; i < costs.size(); ++i)
    for (auto &p : costs[i].predecessors) {
      auto it = names.find(p);
      if (it == names.end() || it->second == i)
        return fail(error, "invalid explicit cost predecessor");
      deps[i].insert(it->second);
    }
  StaticConfig cfg;
  cfg.resources = g.resources;
  cfg.assumptions.overlap = q.serialize ? "forbid" : "allow";
  std::vector<size_t> priority;
  if (!q.schedulePriority.empty()) {
    if (q.schedulePriority.size() != q.costs.size())
      return fail(error,
                  "schedule priority must permute all declared activities");
    std::set<std::string> seen;
    for (const auto &id : q.schedulePriority) {
      auto it = names.find(id);
      if (it == names.end() || it->second >= q.costs.size() ||
          !seen.insert(id).second)
        return fail(error, "unknown or duplicate schedule priority");
      priority.push_back(it->second);
    }
    for (size_t i = q.costs.size(); i < costs.size(); ++i)
      priority.push_back(i);
  } else
    for (size_t i = 0; i < costs.size(); ++i)
      priority.push_back(i);
  for (auto i : priority)
    r.schedulePriority.push_back(costs[i].id);
  auto scheduleOrdered = [&](StaticPlan &plan) {
    StaticPlan ordered = plan;
    ordered.events.clear();
    for (auto i : priority)
      ordered.events.push_back(plan.events[i]);
    if (!scheduleStaticPlan(ordered, cfg, error))
      return false;
    for (auto &event : ordered.events) {
      auto i = names.at(event.id);
      plan.events[i].startMs = event.startMs;
      plan.events[i].endMs = event.endMs;
    }
    plan.totals = ordered.totals;
    return true;
  };
  StaticPlan lower, exact;
  bool complete = true, upperKnown = true;
  double upper = 0;
  std::vector<double> lows;
  for (size_t i = 0; i < costs.size(); ++i) {
    auto &c = costs[i];
    StaticEvent e;
    e.id = c.id;
    e.kind = "compute";
    e.resourceDemands = c.resourceDemands;
    for (auto p : deps[i])
      e.dependsOn.push_back(costs[p].id);
    double lo = c.lowerMs.value_or(c.durationMs.value_or(0));
    lows.push_back(lo);
    e.durationMs = lo;
    lower.events.push_back(e);
    e.durationMs = c.durationMs.value_or(0);
    exact.events.push_back(e);
    if (c.durationMs) {
      r.knownWorkMs += *c.durationMs;
      if (!valid(r.knownWorkMs))
        return fail(error, "known work overflow");
    } else
      complete = false;
    auto hi = c.upperMs ? c.upperMs : c.durationMs;
    if (hi) {
      upper += *hi;
      if (!valid(upper))
        return fail(error, "upper bound overflow");
    } else
      upperKnown = false;
    ProgramTimedActivity out;
    out.id = c.id;
    out.boundary = c.boundary;
    out.provenance = c.provenance;
    out.events = c.events;
    out.regionEvents = c.regionEvents;
    out.executionRegions = c.executionRegions;
    out.predecessors = e.dependsOn;
    for (auto p : bufferDeps[i])
      out.bufferPredecessors.push_back(costs[p].id);
    out.resourceDemands = e.resourceDemands;
    out.durationMs = c.durationMs;
    out.status = c.durationMs ? "implementation-estimate" : "unknown";
    out.reason = c.unknownReason;
    if (!c.durationMs)
      r.unknowns.push_back(c.id + ": " + c.unknownReason);
    r.activities.push_back(out);
  }
  // Run existing calendars even for the diagnostic lower-duration realization,
  // but do not call that heuristic makespan a mathematical lower bound.
  if (!scheduleOrdered(lower))
    return false;
  std::vector<double> ends(costs.size(), -1);
  std::vector<bool> visiting(costs.size());
  std::function<bool(size_t)> visit = [&](size_t i) {
    if (ends[i] >= 0)
      return true;
    if (visiting[i])
      return false;
    visiting[i] = true;
    double t = 0;
    for (auto p : deps[i]) {
      if (!visit(p))
        return false;
      t = std::max(t, ends[p]);
    }
    ends[i] = t + lows[i];
    visiting[i] = false;
    return valid(ends[i]);
  };
  for (size_t i = 0; i < costs.size(); ++i) {
    if (!visit(i))
      return fail(error,
                  "coverage contracts cyclic or overflowing dependencies");
    r.lowerBoundMs = std::max(r.lowerBoundMs, ends[i]);
    r.activities[i].lowerStartMs = lower.events[i].startMs;
    r.activities[i].lowerEndMs = lower.events[i].endMs;
  }
  for (auto &capacity : capacities) {
    double work = 0;
    for (size_t i = 0; i < costs.size(); ++i) {
      auto d = costs[i].resourceDemands.find(capacity.first);
      if (d != costs[i].resourceDemands.end())
        work += lows[i] * double(d->second);
    }
    if (!valid(work))
      return fail(error, "resource work overflow");
    r.lowerBoundMs = std::max(r.lowerBoundMs, work / double(capacity.second));
  }
  if (upperKnown)
    r.upperBoundMs = upper;
  r.assumptions = {
      "cost coverage is exclusive; composite internal ordering is owned by "
      "implementation",
      "fixed resource shares use supplied implementation duration; dynamic "
      "bandwidth sharing is unknown",
      "lower bound is precedence/resource-work bound; upper bound is a "
      "feasible serial work bound before temporal memory validation",
      "graph boundary excludes unlisted executor work",
      "lower timeline substitutes duration lower bounds and is diagnostic, not "
      "predicted execution",
      "temporal memory reservation conservatively spans copy validity ordinal "
      "endpoints; these ordinals add no execution dependency"};
  if (singleGraphOwner)
    r.assumptions.push_back(
        "all graph atoms have one exclusive implementation owner; internal "
        "buffer-reuse edges contract within that implementation and are not "
        "enumerated");
  if (!complete)
    return true;
  if (!scheduleOrdered(exact))
    return false;
  r.makespanMs = exact.totals.makespanMs;
  for (size_t i = 0; i < costs.size(); ++i) {
    r.activities[i].startMs = exact.events[i].startMs;
    r.activities[i].endMs = exact.events[i].endMs;
  }
  // Explain calendar-induced waiting separately from semantic dependencies.
  for (size_t i = 0; i < costs.size(); ++i) {
    double ready = 0;
    for (auto p : deps[i])
      ready = std::max(ready, exact.events[p].endMs);
    auto &e = exact.events[i];
    if (e.startMs <= ready)
      continue;
    for (size_t j = 0; j < costs.size(); ++j)
      if (j != i && exact.events[j].endMs == e.startMs &&
          exact.events[j].durationMs > 0) {
        bool shares = q.serialize;
        for (auto &d : e.resourceDemands)
          shares |= exact.events[j].resourceDemands.count(d.first) > 0;
        if (shares)
          r.activities[i].resourcePredecessors.push_back(costs[j].id);
      }
  }
  if (!costs.empty()) {
    size_t current = 0;
    for (size_t i = 1; i < costs.size(); ++i)
      if (exact.events[i].endMs > exact.events[current].endMs)
        current = i;
    std::set<size_t> seen;
    while (seen.insert(current).second) {
      r.criticalPath.push_back(costs[current].id);
      size_t parent = SIZE_MAX;
      double latest = -1;
      for (auto p : deps[current])
        if (exact.events[p].endMs > latest) {
          parent = p;
          latest = exact.events[p].endMs;
        }
      for (auto &name : r.activities[current].resourcePredecessors) {
        auto p = names.at(name);
        if (exact.events[p].endMs > latest) {
          parent = p;
          latest = exact.events[p].endMs;
        }
      }
      if (parent == SIZE_MAX)
        break;
      current = parent;
    }
    std::reverse(r.criticalPath.begin(), r.criticalPath.end());
  }
  // A phase-pool implementation reuses different static stage pools inside
  // the exclusive whole-call interval. Their checked phase intervals describe
  // that implementation's internal allocation schedule; mapping every birth to
  // the composite start would invent simultaneous residency across all stages.
  // Recompute from allocations (including dormant templates), never cached
  // peaks.
  if (singleGraphOwner && g.allocationLifetime == "phase-pool") {
    struct PhaseDelta {
      int64_t starts = 0, ends = 0;
    };
    std::map<std::string, std::map<size_t, PhaseDelta>> phases;
    for (const auto &a : g.allocations) {
      if (a.bytes <= 0 || a.beginPhase >= a.endPhase)
        return fail(error, "invalid phase-pool allocation interval");
      auto &begin = phases[a.memory][a.beginPhase],
           &end = phases[a.memory][a.endPhase];
      if (begin.starts > INT64_MAX - a.bytes || end.ends > INT64_MAX - a.bytes)
        return fail(error, "phase-pool memory accounting overflow");
      begin.starts += a.bytes;
      end.ends += a.bytes;
    }
    for (const auto &m : g.memories) {
      int64_t live = 0, peak = 0;
      for (const auto &point : phases[m.id]) {
        // Half-open intervals: retire ending reservations before new starts.
        const auto &d = point.second;
        if (d.ends > live || d.starts > INT64_MAX - (live - d.ends))
          return fail(error, "phase-pool memory accounting overflow");
        live = live - d.ends + d.starts;
        peak = std::max(peak, live);
      }
      if (live != 0)
        return fail(error, "unbalanced phase-pool allocation intervals");
      r.peakMemoryBytes[m.id] = peak;
      if (m.capacityKnown && peak > m.capacityBytes)
        return fail(error, "phase-pool allocation capacity exceeded: " + m.id);
    }
    r.knownCapacitiesVerified = true;
    r.assumptions.push_back(
        "phase-pool peaks recomputed by half-open source-phase allocation "
        "sweep; internal stage ordering is owned by the exclusive "
        "implementation, not simultaneous allocation for its entire measured "
        "call");
    return true;
  }
  // Revalidate actual simultaneous allocation occupancy on the predicted
  // timeline. Distinct copies sharing an allocation are charged once;
  // allocation lifetime conservatively covers all version users, including
  // gaps.
  std::map<std::string, std::map<double, int64_t>> delta;
  for (size_t a = 0; a < g.allocations.size(); ++a) {
    auto &allocation = g.allocations[a];
    double begin = *r.makespanMs, end = 0;
    bool used = false;
    for (auto &c : g.copies)
      if (c.allocation == a) {
        used = true;
        double birth = c.createdBy < 0
                           ? 0
                           : exact.events[owner[size_t(c.createdBy)]].startMs;
        begin = std::min(begin, birth);
        end =
            std::max(end, c.releasedAfter == n
                              ? *r.makespanMs
                              : exact.events[owner[c.releasedAfter - 1]].endMs);
        for (auto u : c.users)
          end = std::max(end, exact.events[owner[u]].endMs);
      }
    if (!used) {
      begin = 0;
      end = *r.makespanMs;
    } // No copy witness: reserve the declared allocation conservatively.
    if (end > begin) {
      auto &d = delta[allocation.memory];
      if (d[begin] > INT64_MAX - allocation.bytes)
        return fail(error, "memory accounting overflow");
      d[begin] += allocation.bytes;
      d[end] -= allocation.bytes;
    }
  }
  for (auto &m : g.memories) {
    int64_t live = 0, peak = 0;
    for (auto &d : delta[m.id]) {
      if (d.second > 0 && live > INT64_MAX - d.second)
        return fail(error, "memory accounting overflow");
      live += d.second;
      peak = std::max(peak, live);
    }
    r.peakMemoryBytes[m.id] = peak;
    if (m.capacityKnown && peak > m.capacityBytes)
      return fail(error, "temporal allocation capacity exceeded: " + m.id);
  }
  r.knownCapacitiesVerified = true;
  return true;
}
J orbit::taskflow::serializeProgramTiming(const ProgramTimingEvaluation &r) {
  A activities, bufferOrdering;
  for (const auto &o : r.bufferOrdering)
    bufferOrdering.push_back(O{{"allocation", int64_t(o.allocation)},
                               {"earlier_copy", int64_t(o.earlierCopy)},
                               {"later_copy", int64_t(o.laterCopy)},
                               {"predecessor", int64_t(o.predecessor)},
                               {"successor", int64_t(o.successor)},
                               {"reason", o.reason}});
  O peak;
  for (auto &p : r.peakMemoryBytes)
    peak[p.first] = p.second;
  for (auto &x : r.activities) {
    O demands;
    for (auto &d : x.resourceDemands)
      demands[d.first] = d.second;
    activities.push_back(
        O{{"id", x.id},
          {"status", x.status},
          {"reason", x.reason},
          {"boundary", x.boundary},
          {"provenance", x.provenance},
          {"events", ids(x.events)},
          {"region_events", ids(x.regionEvents)},
          {"execution_regions", ids(x.executionRegions)},
          {"predecessors", strings(x.predecessors)},
          {"resource_predecessors", strings(x.resourcePredecessors)},
          {"buffer_predecessors", strings(x.bufferPredecessors)},
          {"resource_demands", std::move(demands)},
          {"duration_ms", opt(x.durationMs)},
          {"start_ms", opt(x.startMs)},
          {"end_ms", opt(x.endMs)},
          {"lower_duration_schedule_start_ms", x.lowerStartMs},
          {"lower_duration_schedule_end_ms", x.lowerEndMs}});
  }
  return O{{"schema", "orbit.program-timing.v16"},
           {"program_graph_id", r.graphId},
           {"model_identity", r.modelIdentity},
           {"boundary", r.boundary},
           {"policy", r.policy},
           {"known_work_ms", r.knownWorkMs},
           {"lower_bound_ms", r.lowerBoundMs},
           {"upper_bound_ms", opt(r.upperBoundMs)},
           {"makespan_ms", opt(r.makespanMs)},
           {"activities", std::move(activities)},
           {"unknowns", strings(r.unknowns)},
           {"critical_path", strings(r.criticalPath)},
           {"schedule_priority", strings(r.schedulePriority)},
           {"buffer_ordering", std::move(bufferOrdering)},
           {"assumptions", strings(r.assumptions)},
           {"peak_memory_bytes", std::move(peak)},
           {"known_capacities_verified", r.knownCapacitiesVerified},
           {"all_physical_capacities_known", r.allPhysicalCapacitiesKnown},
           {"upper_bound_scope", "time/resource-only serialized work; capacity "
                                 "validity reported separately"}};
}
