#include "Taskflow/GenericMapping.h"
#include "Taskflow/ProgramJointSearch.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
bool fail(std::string &e, const std::string &s) {
  e = "public joint plan: " + s;
  return false;
}
void canonical(std::vector<size_t> &v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}
int64_t aligned(int64_t n, int64_t a) {
  if (a < 1 || n < 1 || n > INT64_MAX - a)
    return -1;
  return ((n + a - 1) / a) * a;
}
// Checks raw decisions and costs without OR-Tools or a planner. The common
// graph checker separately proves source semantics and resource calendars.
bool decisions(const ProgramJointDomain &d, const ProgramJointSolution &s,
               std::string &e) {
  if (!s.hasPlan || s.variant < 0 || size_t(s.variant) >= d.variants.size() ||
      s.groups.empty())
    return fail(e, "missing selected variant/groups");
  if (s.makespanNs < 0 || s.makespanNs > programJointMaxExactNs ||
      d.submitNs < 0 || d.submitNs > programJointMaxExactNs ||
      d.completeNs < 0 || d.completeNs > programJointMaxExactNs ||
      d.configurationNs < 0 || d.configurationNs > programJointMaxExactNs)
    return fail(e, "unsafe exact-integer nanosecond boundary");
  const auto &v = d.variants[s.variant];
  size_t K = s.configurationStartNs.size();
  if (!K || K != s.configurationEndNs.size() || K > size_t(d.maxConfigurations))
    return fail(e, "invalid configuration count");
  std::map<std::string, std::map<int64_t, int64_t>> resources, memories;
  auto reservation = [&](auto &calendars, const std::string &id, int64_t begin,
                         int64_t end, int64_t amount) {
    if (end == begin)
      return true;
    auto &calendar = calendars[id];
    if (amount < 0 || calendar[begin] > INT64_MAX - amount ||
        calendar[end] < INT64_MIN + amount)
      return fail(e, "reservation arithmetic overflow");
    calendar[begin] += amount;
    calendar[end] -= amount;
    return true;
  };
  auto resource = [&](const std::string &id, int64_t begin, int64_t end) {
    std::map<std::string, int64_t> expanded;
    if (!programResourceDemands(d.target, {{id, 1}}, expanded, e))
      return false;
    for (auto &entry : expanded)
      if (!reservation(resources, entry.first, begin, end, entry.second))
        return false;
    return true;
  };
  std::vector<int> cover(v.graph.instances.size());
  std::vector<size_t> owner(cover.size(), SIZE_MAX),
      config(cover.size(), SIZE_MAX);
  std::vector<int64_t> inputs(K), outputs(K), maximum(K);
  std::vector<std::set<std::string>> cores(K);
  for (size_t k = 0; k < K; ++k) {
    auto begin = k ? s.configurationEndNs[k - 1] : d.submitNs;
    if (s.configurationStartNs[k] != begin || begin < 0 ||
        s.configurationEndNs[k] < begin ||
        s.configurationEndNs[k] > programJointMaxExactNs ||
        d.configurationNs > s.configurationEndNs[k] - begin)
      return fail(e, "invalid configuration barrier");
    maximum[k] = begin + d.configurationNs;
  }
  std::vector<int64_t> hostReady(v.producers.size(), -1);
  for (size_t n = 0; n < v.producers.size(); ++n)
    if (v.producers[n] < 0)
      hostReady[n] = d.submitNs;
  for (size_t si = 0; si < s.groups.size(); ++si) {
    const auto &a = s.groups[si];
    if (a.group >= d.groups.size() || a.place >= d.places.size() ||
        a.configuration >= K)
      return fail(e, "invalid decision index");
    const auto &g = d.groups[a.group];
    const auto &p = d.places[a.place];
    auto k = a.configuration;
    int64_t payload;
    std::string reason;
    if (g.variant != size_t(s.variant) ||
        !programJointGroupFits(d, a.group, a.place, payload, reason))
      return fail(e, "illegal group/place: " + reason);
    if (!cores[k].insert(p.binding.resource).second ||
        a.computeStartNs.size() != g.instances.size())
      return fail(e, "duplicate core or missing compute times");
    inputs[k] += g.inputs.size();
    outputs[k] += g.outputs.size();
    if (!reservation(memories, p.binding.memory, s.configurationStartNs[k],
                     s.configurationEndNs[k], payload))
      return false;
    auto begin = s.configurationStartNs[k] + d.configurationNs,
         end = s.configurationEndNs[k];
    int64_t previous = begin;
    for (size_t i = 0; i < g.instances.size(); ++i) {
      auto instance = g.instances[i];
      if (instance >= cover.size() || ++cover[instance] != 1)
        return fail(e, "instance cover overlaps");
      owner[instance] = si;
      config[instance] = k;
      auto start = a.computeStartNs[i], duration = v.computeNs[instance];
      if (start < previous || duration < 0 || start > end ||
          duration > end - start)
        return fail(e, "compute escapes its group/configuration");
      previous = start + duration;
      maximum[k] = std::max(maximum[k], previous);
      if (!resource(p.binding.resource, start, previous))
        return false;
    }
    for (bool input : {true, false}) {
      const auto &expected = input ? g.inputs : g.outputs;
      const auto &routes = input ? a.inputs : a.outputs;
      std::vector<size_t> actual;
      for (auto t : routes) {
        if (t.version >= v.valueBytes.size() || t.link >= d.target.links.size())
          return fail(e, "invalid version/route index");
        actual.push_back(t.version);
        auto &l = d.target.links[t.link];
        auto &allowed = input ? p.inputLinks : p.outputLinks;
        int64_t cost = 0;
        if (std::find(allowed.begin(), allowed.end(), t.link) ==
                allowed.end() ||
            l.source != (input ? d.hostMemory : p.binding.memory) ||
            l.target != (input ? p.binding.memory : d.hostMemory) ||
            !programJointTransferNs(d, s.variant, t.version, t.link, cost, e))
          return fail(e, "unsupported route endpoints/cost");
        if (t.startNs < begin || t.endNs < t.startNs || t.endNs > end ||
            t.endNs - t.startNs != cost)
          return fail(e, "transfer duration/configuration mismatch");
        if (input) {
          if (t.endNs > a.computeStartNs.front())
            return fail(e, "compute starts before all group inputs arrive");
        } else {
          if (t.startNs < previous || hostReady[t.version] >= 0)
            return fail(e, "output precedes compute or duplicate host export");
          hostReady[t.version] = t.endNs;
        }
        maximum[k] = std::max(maximum[k], t.endNs);
        if (!resource(l.resource, t.startNs, t.endNs))
          return false;
      }
      std::sort(actual.begin(), actual.end());
      if (actual != expected)
        return fail(e, "chosen boundary copies differ from selected group");
    }
  }
  for (auto &entry : resources) {
    auto target = programResource(d.target, entry.first);
    int64_t live = 0;
    for (auto delta : entry.second) {
      if (delta.second > 0 && live > INT64_MAX - delta.second)
        return fail(e, "resource occupancy overflow");
      live += delta.second;
      if (live < 0 || live > target->capacity)
        return fail(e, "overlapping exact integer resource reservations");
    }
    if (live)
      return fail(e, "unclosed resource reservation");
  }
  for (auto &entry : memories) {
    auto target = programMemory(d.target, entry.first);
    int64_t live = 0;
    for (auto delta : entry.second) {
      if (delta.second > 0 && live > INT64_MAX - delta.second)
        return fail(e, "memory occupancy overflow");
      live += delta.second;
      if (live < 0 || live > target->capacityBytes)
        return fail(e, "parallel static memory occupancy exceeds capacity");
    }
    if (live)
      return fail(e, "unclosed memory reservation");
  }
  if (std::find(cover.begin(), cover.end(), 0) != cover.end())
    return fail(e, "source instance omitted");
  for (size_t k = 0; k < K; ++k)
    if (cores[k].empty() || inputs[k] > d.maxConfigurationInputs ||
        outputs[k] > d.maxConfigurationOutputs ||
        maximum[k] != s.configurationEndNs[k])
      return fail(e, "empty/overcommitted configuration or false completion");
  if (s.configurationEndNs.back() > programJointMaxExactNs - d.completeNs ||
      s.makespanNs != s.configurationEndNs.back() + d.completeNs)
    return fail(e, "total model boundary/cost mismatch");
  for (const auto &a : s.groups)
    for (auto t : a.inputs)
      if (hostReady[t.version] < 0 || t.startNs < hostReady[t.version])
        return fail(e, "read before correct host version is available");
  for (size_t value = 0; value < v.producers.size(); ++value)
    if (v.producers[value] >= 0)
      for (auto reader : v.readers[value]) {
        auto producer = size_t(v.producers[value]);
        if (owner[reader] != owner[producer] &&
            config[reader] <= config[producer])
          return fail(e, "cross-group RAW requires a later configuration");
      }
  for (const auto &edge : v.graph.orderEdges) {
    auto pt = v.graph.events[edge.producer].task,
         ct = v.graph.events[edge.consumer].task;
    for (size_t p = 0; p < cover.size(); ++p)
      if (v.graph.instances[p].task == pt)
        for (size_t c = 0; c < cover.size(); ++c)
          if (v.graph.instances[c].task == ct && config[c] <= config[p])
            return fail(e, "pure source ordering violated");
  }
  return true;
}
} // namespace

bool orbit::taskflow::materializeProgramJointSolution(
    const ProgramJointDomain &d, const ProgramJointSolution &s, ProgramGraph &g,
    std::string &e) {
  if (!decisions(d, s, e))
    return false;
  g = {};
  const auto &v = d.variants[s.variant];
  const auto &source = v.graph.source;
  g.domains = d.target.domains;
  g.resources = d.target.resources;
  g.memories = d.target.memories;
  g.links = d.target.links;
  g.targetIdentity = v.graph.targetIdentity;
  g.regions = v.graph.regions;
  g.versions = v.graph.versions;
  g.dataPolicy = ProgramDataPolicy::RetainAcrossUsers;
  for (auto &version : g.versions)
    if (version.producer >= 0)
      version.producer = INT64_MAX;
  GenericMappingPlan adapter;
  adapter.strategy = "serial-tiled";
  adapter.memoryBinding.hostMemory = d.hostMemory;
  for (size_t t = 0; t < source.tasks.size(); ++t)
    adapter.taskImplementations.push_back({t, "scalar", -1, 1});
  auto host = programMemory(d.target, d.hostMemory);
  if (!host)
    return fail(e, "missing host memory");
  std::vector<size_t> hostAlloc(source.storages.size(), SIZE_MAX),
      hostCopy(g.versions.size(), SIZE_MAX),
      instanceEvent(v.graph.instances.size(), SIZE_MAX);
  auto allocateHost = [&](size_t storage) {
    if (hostAlloc[storage] != SIZE_MAX)
      return hostAlloc[storage];
    int64_t n =
        (mlir::cast<mlir::MemRefType>(source.storages[storage].root.getType())
             .getElementType()
             .getIntOrFloatBitWidth() +
         7) /
        8;
    for (auto x : source.storages[storage].shape) {
      if (x < 1 || n > INT64_MAX / x)
        return SIZE_MAX;
      n *= x;
    }
    MappingAllocation a;
    a.storage = storage;
    a.memory = d.hostMemory;
    a.key = "host-bo-" + std::to_string(storage);
    a.bytes = aligned(n, host->alignmentBytes);
    a.addressing = "region-addressed";
    a.spatialOwner = host->domain;
    auto id = g.allocations.size();
    g.allocations.push_back(a);
    return hostAlloc[storage] = id;
  };
  for (size_t value = 0; value < v.producers.size(); ++value)
    if (v.producers[value] < 0) {
      MappingCopy c;
      c.version = value;
      c.allocation = allocateHost(g.regions[g.versions[value].region].storage);
      hostCopy[value] = g.copies.size();
      g.copies.push_back(c);
    }
  std::vector<size_t> order(s.groups.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](auto a, auto b) {
    return std::tie(s.groups[a].configuration, s.groups[a].group) <
           std::tie(s.groups[b].configuration, s.groups[b].group);
  });
  std::vector<size_t> eventGroups, groupConfig;
  std::vector<MappingScheduledEvent> times;
  std::vector<double> durations;
  auto append = [&](MappingEvent ev, int64_t begin, int64_t end, size_t group) {
    canonical(ev.dependencies);
    auto id = g.events.size();
    g.events.push_back(std::move(ev));
    times.push_back({id, g.events.back().dependencies, double(begin) / 1e6,
                     double(end) / 1e6});
    durations.push_back(double(end - begin) / 1e6);
    eventGroups.push_back(group);
    return id;
  };
  for (auto index : order) {
    const auto &a = s.groups[index];
    const auto &mode = d.groups[a.group];
    const auto &place = d.places[a.place];
    auto gi = adapter.executionGroups.size();
    auto phaseBegin = adapter.phases.size(),
         phaseEnd = phaseBegin + mode.instances.size();
    ProgramExecutionGroup group;
    group.beginPhase = phaseBegin;
    group.endPhase = phaseEnd;
    group.resource = place.binding.resource;
    group.memory = place.binding.memory;
    group.spatialOwner = programResource(d.target, group.resource)->domain;
    groupConfig.push_back(a.configuration);
    std::map<size_t, size_t> localAlloc, localCopy;
    auto memory = programMemory(d.target, group.memory);
    for (auto value : mode.values) {
      MappingAllocation allocation;
      allocation.storage = g.regions[g.versions[value].region].storage;
      allocation.memory = group.memory;
      allocation.key =
          "group-" + std::to_string(gi) + "-version-" + std::to_string(value);
      allocation.bytes = aligned(v.valueBytes[value], memory->alignmentBytes);
      allocation.beginPhase = phaseBegin;
      allocation.endPhase = phaseEnd;
      allocation.spatialOwner = memory->domain;
      localAlloc[value] = g.allocations.size();
      g.allocations.push_back(allocation);
    }
    std::vector<size_t> inputEvents;
    for (auto t : a.inputs) {
      auto value = t.version;
      if (hostCopy[value] == SIZE_MAX)
        return fail(e, "producer group has not exported required version");
      auto id = g.events.size();
      MappingCopy c;
      c.version = value;
      c.allocation = localAlloc[value];
      c.createdBy = id;
      c.availableFrom = id;
      auto copy = g.copies.size();
      g.copies.push_back(c);
      localCopy[value] = copy;
      auto &l = d.target.links[t.link];
      MappingEvent ev;
      ev.kind = "transfer";
      ev.phase = phaseBegin;
      ev.task = v.graph.instances[mode.instances.front()].task;
      ev.resource = l.resource;
      ev.link = l.id;
      ev.reads = {hostCopy[value]};
      ev.writes = {copy};
      ev.transferRegion = g.regions[g.versions[value].region];
      ev.transferBytes = v.valueBytes[value];
      if (g.copies[hostCopy[value]].createdBy >= 0)
        ev.dependencies.push_back(g.copies[hostCopy[value]].createdBy);
      inputEvents.push_back(append(ev, t.startNs, t.endNs, gi));
    }
    size_t previous = SIZE_MAX;
    for (size_t i = 0; i < mode.instances.size(); ++i) {
      auto instance = mode.instances[i];
      const auto &reference = v.graph.instances[instance];
      GenericTilePhase phase;
      phase.task = reference.task;
      for (auto b : reference.domain)
        phase.loops.push_back({b.loop, b.lower, b.upper});
      adapter.phases.push_back(phase);
      auto id = g.events.size();
      MappingEvent ev;
      ev.kind = "compute";
      ev.phase = phaseBegin + i;
      ev.task = reference.task;
      ev.resource = group.resource;
      ev.memory = group.memory;
      ev.dependencies = inputEvents;
      if (previous != SIZE_MAX)
        ev.dependencies.push_back(previous);
      for (auto old : v.graph.events[reference.event].reads) {
        auto value = v.graph.copies[old].version;
        auto found = localCopy.find(value);
        if (found == localCopy.end())
          return fail(e, "missing selected local copy");
        auto copy = found->second;
        ev.reads.push_back(copy);
        if (g.copies[copy].createdBy >= 0)
          ev.dependencies.push_back(g.copies[copy].createdBy);
        ev.dataEdges.push_back(g.dataEdges.size());
        auto region = g.versions[value].region;
        g.dataEdges.push_back(
            {g.versions[value].producer,
             id,
             region,
             value,
             copy,
             {group.memory, "row_major",
              source.storages[g.regions[region].storage].dtype}});
      }
      for (auto old : v.graph.events[reference.event].writes) {
        auto value = v.graph.copies[old].version;
        g.versions[value].producer = id;
        MappingCopy copy;
        copy.version = value;
        copy.allocation = localAlloc[value];
        copy.createdBy = id;
        copy.availableFrom = id;
        localCopy[value] = g.copies.size();
        ev.writes.push_back(g.copies.size());
        g.copies.push_back(copy);
      }
      auto begin = a.computeStartNs[i];
      previous = append(ev, begin, begin + v.computeNs[instance], gi);
      instanceEvent[instance] = previous;
    }
    for (auto t : a.outputs) {
      auto value = t.version;
      auto id = g.events.size();
      MappingCopy copy;
      copy.version = value;
      copy.allocation =
          allocateHost(g.regions[g.versions[value].region].storage);
      copy.createdBy = id;
      copy.availableFrom = id;
      hostCopy[value] = g.copies.size();
      g.copies.push_back(copy);
      auto &l = d.target.links[t.link];
      MappingEvent ev;
      ev.kind = "transfer";
      ev.phase = phaseEnd - 1;
      ev.task = v.graph.instances[mode.instances.back()].task;
      ev.resource = l.resource;
      ev.link = l.id;
      ev.reads = {localCopy.at(value)};
      ev.writes = {hostCopy[value]};
      ev.dependencies = {previous,
                         size_t(g.copies[localCopy.at(value)].createdBy)};
      ev.transferRegion = g.regions[g.versions[value].region];
      ev.transferBytes = v.valueBytes[value];
      append(ev, t.startNs, t.endNs, gi);
    }
    for (auto &entry : localCopy)
      g.copies[entry.second].releasedAfter = g.events.size();
    adapter.executionGroups.push_back(group);
  }
  // Source pure order is represented once at the complete task boundary. Every
  // shard also obeys this order in the independently checked raw decisions.
  std::map<size_t, size_t> first, last;
  for (size_t i = 0; i < g.events.size(); ++i)
    if (g.events[i].kind == "compute") {
      first.emplace(g.events[i].task, i);
      last[g.events[i].task] = i;
    }
  for (const auto &edge : v.graph.orderEdges) {
    auto parent = v.graph.events[edge.producer].task,
         child = v.graph.events[edge.consumer].task;
    auto from = last[parent], to = first[child];
    if (from >= to)
      return fail(e, "non-topological pure source order");
    g.events[to].dependencies.push_back(from);
    g.events[to].orderEdges.push_back(g.orderEdges.size());
    g.orderEdges.push_back({from, to, "source pure ordering"});
  }
  // Configuration barriers are implementation edges. They are never presented
  // as additional semantic source dependencies.
  for (size_t i = 0; i < g.events.size(); ++i) {
    canonical(g.events[i].dependencies);
    for (auto parent : g.events[i].dependencies)
      if (eventGroups[parent] != eventGroups[i])
        adapter.executionGroups[eventGroups[i]].dependencies.push_back(
            eventGroups[parent]);
  }
  for (size_t gi = 0; gi < groupConfig.size(); ++gi)
    for (size_t pj = 0; pj < gi; ++pj)
      if (groupConfig[pj] + 1 == groupConfig[gi])
        adapter.executionGroups[gi].implementationDependencies.push_back(pj);
  std::vector<std::vector<size_t>> configEvents(s.configurationStartNs.size());
  for (size_t i = 0; i < g.events.size(); ++i)
    configEvents[groupConfig[eventGroups[i]]].push_back(i);
  for (size_t k = 1; k < configEvents.size(); ++k)
    for (auto ev : configEvents[k]) {
      for (auto prev : configEvents[k - 1])
        g.events[ev].dependencies.push_back(prev);
    }
  for (size_t i = 0; i < g.events.size(); ++i) {
    auto &ev = g.events[i];
    canonical(ev.dependencies);
    times[i].predecessors = ev.dependencies;
    for (auto c : ev.reads)
      g.copies[c].users.push_back(i);
    for (auto edge : ev.dataEdges)
      g.copies[g.dataEdges[edge].copy].consumerEdges.push_back(edge);
  }
  for (auto &group : adapter.executionGroups) {
    canonical(group.dependencies);
    canonical(group.implementationDependencies);
  }
  for (auto &copy : g.copies) {
    if (!copy.releasedAfter)
      copy.releasedAfter = g.events.size();
    if (!copy.users.empty())
      copy.lastUsers = {copy.users.back()};
  }
  for (auto &allocation : g.allocations)
    if (allocation.memory == d.hostMemory) {
      allocation.beginPhase = 0;
      allocation.endPhase = adapter.phases.size();
    }
  MappingExecutionRegion call;
  call.id = "joint-program";
  call.placement = "npu";
  call.endEvent = g.events.size();
  call.submissionCount = 1;
  for (const auto &r : d.target.resources)
    if (r.kind == "submission") {
      call.submissionResource = r.id;
      break;
    }
  if (call.submissionResource.empty())
    return fail(e, "target lacks submission resource");
  g.executionRegions = {call};
  g.schedule.policy = "dependency-order-witness";
  if (!finalizeProgramGraph(source, g, &adapter, e))
    return false;
  g.schedule.policy = "constraint-resource-schedule";
  g.schedule.modelIdentity = d.model.id;
  g.schedule.timingBoundary = d.model.boundary;
  g.schedule.events = std::move(times);
  g.schedule.eventDurationsMs = std::move(durations);
  g.schedule.uncalibrated.clear();
  g.schedule.estimatedMakespanMs = double(s.configurationEndNs.back()) / 1e6;
  for (size_t k = 0; k < s.configurationStartNs.size(); ++k) {
    MappingScheduledConfiguration c;
    c.startMs = double(s.configurationStartNs[k]) / 1e6;
    c.endMs = double(s.configurationEndNs[k]) / 1e6;
    c.setupMs = double(d.configurationNs) / 1e6;
    for (size_t gi = 0; gi < groupConfig.size(); ++gi)
      if (groupConfig[gi] == k)
        c.groups.push_back(gi);
    g.schedule.configurations.push_back(c);
  }
  g.transferBytes = 0;
  for (const auto &ev : g.events)
    g.transferBytes += ev.transferBytes;
  return true;
}

bool orbit::taskflow::checkProgramJointSolution(const ProgramJointDomain &d,
                                                const ProgramJointSolution &s,
                                                const ProgramGraph &g,
                                                std::string &e) {
  if (!decisions(d, s, e) || !checkProgramGraph(g, e))
    return false;
  ProgramGraph expected;
  if (!materializeProgramJointSolution(d, s, expected, e))
    return false;
  if (programGraphIdentity(expected) != programGraphIdentity(g))
    return fail(e, "public graph differs from chosen source-bound decisions");
  return true;
}

ProgramJointSolution
orbit::taskflow::baselineProgramJointDomain(const ProgramJointDomain &d) {
  ProgramJointSolution best;
  best.pricingComplete = d.costsComplete;
  best.message =
      "no independently checked serial baseline fits the finite backend domain";
  if (d.submitNs < 0 || d.submitNs > programJointMaxExactNs ||
      d.configurationNs < 0 || d.configurationNs > programJointMaxExactNs ||
      d.completeNs < 0 || d.completeNs > programJointMaxExactNs) {
    best.message = "unsafe exact-integer nanosecond boundary";
    return best;
  }
  for (size_t vi = 0; vi < d.variants.size(); ++vi) {
    const auto &v = d.variants[vi];
    ProgramJointSolution s;
    s.pricingComplete = d.costsComplete;
    s.variant = vi;
    s.hasPlan = true;
    s.selection = "baseline";
    s.status = ConstraintSearchStatus::Feasible;
    std::set<size_t> done;
    int64_t previous = d.submitNs;
    while (done.size() < v.graph.instances.size() &&
           s.groups.size() < size_t(d.maxConfigurations)) {
      size_t chosen = SIZE_MAX, place = 0;
      for (size_t gi = 0; gi < d.groups.size(); ++gi) {
        const auto &g = d.groups[gi];
        if (g.variant != vi ||
            g.inputs.size() > size_t(d.maxConfigurationInputs) ||
            g.outputs.size() > size_t(d.maxConfigurationOutputs))
          continue;
        bool legal = true;
        for (auto i : g.instances) {
          if (done.count(i)) {
            legal = false;
            break;
          }
          for (auto copy : v.graph.events[v.graph.instances[i].event].reads) {
            auto p = v.producers[v.graph.copies[copy].version];
            if (p >= 0 && !done.count(p) &&
                !std::binary_search(g.instances.begin(), g.instances.end(),
                                    size_t(p)))
              legal = false;
          }
        }
        for (const auto &edge : v.graph.orderEdges)
          for (auto i : g.instances)
            if (v.graph.instances[i].task == v.graph.events[edge.consumer].task)
              for (size_t p = 0; p < v.graph.instances.size(); ++p)
                if (v.graph.instances[p].task ==
                        v.graph.events[edge.producer].task &&
                    !done.count(p))
                  legal = false;
        if (!legal)
          continue;
        for (size_t pi = 0; pi < d.places.size(); ++pi) {
          int64_t n;
          std::string ignored;
          if (!programJointGroupFits(d, gi, pi, n, ignored))
            continue;
          // An unpriced placement stays in the domain, but cannot be used as a
          // checked finite horizon. Continue looking for a fully priced
          // baseline.
          bool priced = true;
          for (bool input : {true, false})
            for (auto value : (input ? g.inputs : g.outputs)) {
              bool found = false;
              for (auto li : (input ? d.places[pi].inputLinks
                                    : d.places[pi].outputLinks)) {
                int64_t ns;
                if (programJointTransferNs(d, vi, value, li, ns, ignored))
                  found = true;
              }
              priced &= found;
            }
          if (priced &&
              (chosen == SIZE_MAX ||
               g.instances.size() > d.groups[chosen].instances.size())) {
            chosen = gi;
            place = pi;
          }
        }
      }
      if (chosen == SIZE_MAX ||
          previous > programJointMaxExactNs - d.configurationNs)
        break;
      const auto &g = d.groups[chosen];
      ProgramJointSelection a;
      a.group = chosen;
      a.place = place;
      a.configuration = s.groups.size();
      s.configurationStartNs.push_back(previous);
      int64_t time = previous + d.configurationNs;
      bool valid = true;
      for (bool input : {true, false}) {
        if (!input)
          for (auto i : g.instances) {
            a.computeStartNs.push_back(time);
            if (v.computeNs[i] < 0 || v.computeNs[i] > programJointMaxExactNs ||
                time > programJointMaxExactNs - v.computeNs[i]) {
              valid = false;
              break;
            }
            time += v.computeNs[i];
            done.insert(i);
          }
        for (auto value : (input ? g.inputs : g.outputs)) {
          size_t link = SIZE_MAX;
          int64_t minimum = INT64_MAX;
          for (auto li : (input ? d.places[place].inputLinks
                                : d.places[place].outputLinks)) {
            int64_t ns;
            std::string ignored;
            if (programJointTransferNs(d, vi, value, li, ns, ignored) &&
                ns < minimum) {
              minimum = ns;
              link = li;
            }
          }
          if (link == SIZE_MAX || minimum < 0 ||
              minimum > programJointMaxExactNs ||
              time > programJointMaxExactNs - minimum) {
            valid = false;
            break;
          }
          ProgramJointTransfer t{value, link, time, time + minimum};
          (input ? a.inputs : a.outputs).push_back(t);
          time += minimum;
        }
      }
      if (!valid)
        break;
      s.configurationEndNs.push_back(time);
      previous = time;
      s.groups.push_back(a);
    }
    if (done.size() != v.graph.instances.size() ||
        previous > programJointMaxExactNs - d.completeNs)
      continue;
    s.makespanNs = previous + d.completeNs;
    ProgramGraph graph;
    std::string error;
    if (!materializeProgramJointSolution(d, s, graph, error) ||
        !checkProgramJointSolution(d, s, graph, error)) {
      best.message = "baseline rejected: " + error;
      continue;
    }
    if (!best.hasPlan || s.makespanNs < best.makespanNs)
      best = s;
  }
  return best;
}

j::Value
orbit::taskflow::serializeProgramJointSolution(const ProgramJointDomain &d,
                                               const ProgramJointSolution &s,
                                               const ProgramGraph *graph) {
  auto list = [](const auto &v) {
    j::Array out;
    for (auto x : v)
      out.push_back(int64_t(x));
    return out;
  };
  j::Array groups;
  for (auto &a : s.groups) {
    auto routes = [](const auto &items) {
      j::Array out;
      for (auto t : items)
        out.push_back(j::Object{{"version", int64_t(t.version)},
                                {"link", int64_t(t.link)},
                                {"start_ns", t.startNs},
                                {"end_ns", t.endNs}});
      return out;
    };
    groups.push_back(j::Object{{"group", int64_t(a.group)},
                               {"place", int64_t(a.place)},
                               {"configuration", int64_t(a.configuration)},
                               {"compute_start_ns", list(a.computeStartNs)},
                               {"inputs", routes(a.inputs)},
                               {"outputs", routes(a.outputs)}});
  }
  auto &t = s.telemetry;
  const bool priced = d.costsComplete && s.pricingComplete;
  const bool fullScope = d.complete && priced;
  const char *scope =
      fullScope ? "complete-known-domain" : "known-cost-generated-subset";
  j::Object result{
      {"schema", "orbit.program-joint.solution.v23"},
      {"domain_id", d.identity},
      {"source_fingerprint",
       d.variants.empty() ? "" : d.variants.front().graph.bodyFingerprint},
      {"model_id", d.model.id},
      {"status", constraintSearchStatusName(s.status)},
      {"message", s.message},
      {"selection", s.selection},
      {"has_plan", s.hasPlan},
      {"variant", s.variant},
      {"makespan_ns", s.makespanNs},
      {"groups", std::move(groups)},
      {"configuration_start_ns", list(s.configurationStartNs)},
      {"configuration_end_ns", list(s.configurationEndNs)},
      {"domain_complete", d.complete},
      {"domain_costs_complete", d.costsComplete},
      {"pricing_complete", s.pricingComplete},
      {"costs_complete", priced},
      {"host_capacity_scope",
       "caller-owned source BOs; local placement capacity only"},
      {"optimization_scope", scope},
      {"bound_scope", scope},
      {"full_domain_optimal",
       s.status == ConstraintSearchStatus::Optimal && fullScope},
      {"generation_ms", d.generationMs},
      {"model_build_ms", t.modelBuildMs},
      {"solve_ms", t.solveMs},
      {"total_ms", t.totalMs},
      {"variables", t.variables},
      {"constraints", t.constraints},
      {"conflicts", t.conflicts},
      {"branches", t.branches},
      {"peak_rss_kib", t.peakRssKiB},
      {"has_objective", t.hasObjective},
      {"objective_ns", t.objectiveNs},
      {"best_bound_ns", t.bestBoundNs},
      {"relative_gap", t.relativeGap},
      {"budget_exhausted", t.budgetExhausted},
      {"objective_exact_for_backend", false}};
  if (graph)
    result["program_graph"] = serializeProgramGraph(*graph);
  return result;
}

bool orbit::taskflow::parseProgramJointSolution(const j::Value &value,
                                                ProgramJointSolution &s,
                                                std::string &e) {
  s = {};
  auto o = value.getAsObject();
  if (!o || o->getString("schema") != "orbit.program-joint.solution.v23")
    return fail(e, "unsupported solution schema");
  auto integer = [&](const j::Object &object, const char *key, int64_t &n) {
    auto x = object.getInteger(key);
    if (!x || *x < 0)
      return false;
    n = *x;
    return true;
  };
  auto list = [&](const j::Object &object, const char *key,
                  std::vector<int64_t> &out) {
    auto a = object.getArray(key);
    if (!a)
      return false;
    for (auto &x : *a) {
      auto n = x.getAsInteger();
      if (!n || *n < 0)
        return false;
      out.push_back(*n);
    }
    return true;
  };
  if (!integer(*o, "variant", s.variant) ||
      !integer(*o, "makespan_ns", s.makespanNs) ||
      !list(*o, "configuration_start_ns", s.configurationStartNs) ||
      !list(*o, "configuration_end_ns", s.configurationEndNs))
    return fail(e, "invalid solution numbers");
  if (o->get("pricing_complete") && !o->getBoolean("pricing_complete"))
    return fail(e, "pricing_complete must be boolean");
  // Legacy receipts have only costs_complete. Preserve their conservative
  // coverage on replay rather than upgrading an unknown-cost solution.
  s.pricingComplete =
      o->getBoolean("pricing_complete")
          .value_or(o->getBoolean("costs_complete").value_or(true));
  s.hasPlan = o->getBoolean("has_plan").value_or(false);
  s.selection = o->getString("selection").value_or("none").str();
  s.message = o->getString("message").value_or("").str();
  auto status = o->getString("status");
  for (auto x :
       {ConstraintSearchStatus::Unavailable,
        ConstraintSearchStatus::InvalidInput,
        ConstraintSearchStatus::ModelLimit, ConstraintSearchStatus::Unknown,
        ConstraintSearchStatus::Infeasible, ConstraintSearchStatus::Feasible,
        ConstraintSearchStatus::Optimal})
    if (status == constraintSearchStatusName(x))
      s.status = x;
  auto groups = o->getArray("groups");
  if (!groups || groups->size() > 4096)
    return fail(e, "invalid selected group array");
  for (auto &entry : *groups) {
    auto a = entry.getAsObject();
    if (!a)
      return fail(e, "invalid selected group record");
    ProgramJointSelection selected;
    int64_t g, p, k;
    if (!integer(*a, "group", g) || !integer(*a, "place", p) ||
        !integer(*a, "configuration", k) ||
        !list(*a, "compute_start_ns", selected.computeStartNs))
      return fail(e, "invalid selected group fields");
    selected.group = g;
    selected.place = p;
    selected.configuration = k;
    for (bool input : {true, false}) {
      auto routes = a->getArray(input ? "inputs" : "outputs");
      if (!routes || routes->size() > 4096)
        return fail(e, "invalid chosen route array");
      for (auto &entry : *routes) {
        auto r = entry.getAsObject();
        ProgramJointTransfer t;
        int64_t version, link;
        if (!r || !integer(*r, "version", version) ||
            !integer(*r, "link", link) || !integer(*r, "start_ns", t.startNs) ||
            !integer(*r, "end_ns", t.endNs))
          return fail(e, "invalid chosen route");
        t.version = version;
        t.link = link;
        (input ? selected.inputs : selected.outputs).push_back(t);
      }
    }
    s.groups.push_back(selected);
  }
  return true;
}
