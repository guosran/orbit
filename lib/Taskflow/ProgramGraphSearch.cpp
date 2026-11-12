#include "Taskflow/ProgramMapping.h"
#include "Taskflow/ProgramTargetBinding.h"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>

using namespace orbit::taskflow;
namespace {
bool fail(std::string &error, const std::string &why) {
  error = "public graph search materialization: " + why;
  return false;
}
int64_t bytes(const ProgramGraph &g, size_t region) {
  const auto &r = g.regions[region];
  auto type =
      mlir::cast<mlir::MemRefType>(g.source.storages[r.storage].root.getType());
  int64_t n = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
  for (auto size : r.sizes) {
    if (size < 1 || n > INT64_MAX / size)
      return -1;
    n *= size;
  }
  return n;
}
void canonical(std::vector<size_t> &v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}
} // namespace

// Rebind the existing source/instance/version graph. No operator recipes,
// pointwise grid proof, source partition inference or new target model.
bool orbit::taskflow::rebindProgramGraph(
    const ProgramGraph &reference, const StaticConfig &target,
    const ProgramDataOrganizationChoice &choice, ProgramGraph &g,
    std::string &error) {
  if (!checkProgramGraph(reference, error) ||
      !validateMemoryArchitecture(target, error))
    return false;
  if (choice.placements.size() != reference.source.tasks.size())
    return fail(error, "one declared binding per source task required");
  std::vector<MappingTaskPlacement> places(choice.placements.size());
  std::set<size_t> seen;
  for (auto p : choice.placements) {
    auto resource = programResource(target, p.resource);
    auto memory = programMemory(target, p.memory);
    if (p.task >= places.size() || !seen.insert(p.task).second || !resource ||
        resource->kind != "compute" || !memory ||
        !memoryAccessible(target, p.memory, p.resource) ||
        p.resourceCount < 1 || p.resourceCount > resource->capacity)
      return fail(error, "invalid task resource/memory/count binding");
    int64_t n = 1;
    for (auto x : p.resourceShape) {
      if (x < 1 || n > INT64_MAX / x)
        return fail(error, "invalid resource geometry");
      n *= x;
    }
    if (!p.resourceShape.empty() && n != p.resourceCount)
      return fail(error, "resource geometry/count mismatch");
    places[p.task] = p;
  }
  g = reference;
  g.targetIdentity = staticFingerprint(serializeMemoryArchitecture(target));
  g.domains = target.domains;
  g.resources = target.resources;
  g.memories = target.memories;
  g.links = target.links;
  g.placements = places;
  g.dataPolicy = choice.policy;
  g.computeEstimates = choice.computeEstimates;
  g.clockMHz = choice.clockMHz;
  g.events.clear();
  g.copies.clear();
  g.allocations.clear();
  g.dataEdges.clear();
  g.orderEdges.clear();
  g.instances.clear();
  g.regionEvents.clear();
  g.executionGroups.clear();
  g.transferBytes = 0;
  g.peakMemoryBytes.clear();
  g.estimatedMakespanMs.reset();
  g.schedule = {};
  g.schedule.policy = "dependency-order-witness";
  g.allocationLifetime = "copy-events";
  for (auto &v : g.versions)
    v.producer = -2;
  std::vector<size_t> remap(reference.events.size(), SIZE_MAX),
      instanceAt(reference.events.size(), SIZE_MAX),
      oldGroup(reference.events.size(), SIZE_MAX);
  for (size_t i = 0; i < reference.instances.size(); ++i)
    instanceAt[reference.instances[i].event] = i;
  for (size_t i = 0; i < reference.executionGroups.size(); ++i) {
    const auto &old = reference.executionGroups[i];
    ProgramExecutionGroup group = old;
    group.memberEvents.clear();
    group.dependencies.clear();
    group.implementationDependencies.clear();
    bool first = true;
    for (auto event : old.memberEvents)
      if (instanceAt[event] != SIZE_MAX) {
        auto p = places[reference.events[event].task];
        if (first) {
          group.resource = p.resource;
          group.memory = p.memory;
          auto r = programResource(target, p.resource);
          group.spatialOwner = r->domain;
          first = false;
        } else if (group.resource != p.resource || group.memory != p.memory)
          return fail(error, "a selected execution group must have one "
                             "physical resource and local memory");
      }
    if (first)
      return fail(error, "execution group has no source instance");
    g.executionGroups.push_back(group);
    for (auto event : old.memberEvents)
      oldGroup[event] = i;
  }
  std::map<std::pair<size_t, std::string>, size_t> available, boundaryAlloc;
  std::vector<bool> imported;
  std::vector<size_t> initialCopies, outputCopies, eventOwners;
  std::vector<bool> transferUsed(choice.transferChoices.size());
  auto allocate = [&](size_t region, const std::string &memory, bool boundary) {
    auto m = programMemory(target, memory);
    if (!m)
      return SIZE_MAX;
    auto storage = g.regions[region].storage;
    if (boundary) {
      auto found = boundaryAlloc.find({storage, memory});
      if (found != boundaryAlloc.end())
        return found->second;
    }
    auto n = bytes(g, region);
    if (boundary) {
      auto ty = mlir::cast<mlir::MemRefType>(
          g.source.storages[storage].root.getType());
      n = (ty.getElementType().getIntOrFloatBitWidth() + 7) / 8;
      for (auto x : g.source.storages[storage].shape) {
        if (x < 1 || n > INT64_MAX / x)
          return SIZE_MAX;
        n *= x;
      }
    }
    if (n < 1 || m->alignmentBytes < 1)
      return SIZE_MAX;
    auto padding =
        (m->alignmentBytes - n % m->alignmentBytes) % m->alignmentBytes;
    if (n > INT64_MAX - padding)
      return SIZE_MAX;
    MappingAllocation a;
    a.storage = storage;
    a.memory = memory;
    a.key = "search-copy-" + std::to_string(g.allocations.size());
    a.bytes = n + padding;
    a.addressing = boundary ? "region-addressed" : "reusable-slot";
    a.spatialOwner = m->domain;
    a.beginPhase = 0;
    a.endPhase = reference.instances.size();
    auto id = g.allocations.size();
    g.allocations.push_back(a);
    if (boundary)
      boundaryAlloc[{storage, memory}] = id;
    return id;
  };
  auto copy = [&](size_t version, const std::string &memory, int64_t creator,
                  bool boundary, bool import) {
    auto allocation = allocate(g.versions[version].region, memory, boundary);
    if (allocation == SIZE_MAX)
      return SIZE_MAX;
    MappingCopy c;
    c.version = version;
    c.allocation = allocation;
    c.createdBy = creator;
    c.availableFrom = creator < 0 ? 0 : size_t(creator);
    auto id = g.copies.size();
    g.copies.push_back(c);
    imported.push_back(import);
    available[{version, memory}] = id;
    return id;
  };
  for (size_t version = 0; version < reference.versions.size(); ++version)
    if (reference.versions[version].producer < 0) {
      g.versions[version].producer = -1;
      std::set<std::string> locations;
      auto storage = g.regions[g.versions[version].region].storage;
      auto explicitLocation = target.assumptions.initialLocations.find(
          g.source.storages[storage].id);
      if (explicitLocation != target.assumptions.initialLocations.end())
        locations.insert(explicitLocation->second);
      else
        for (auto &c : reference.copies)
          if (c.version == version && c.createdBy < 0)
            locations.insert(reference.allocations[c.allocation].memory);
      if (locations.empty())
        return fail(error, "source input boundary has no declared location");
      for (auto &memory : locations) {
        auto id = copy(version, memory, -1, true, false);
        if (id == SIZE_MAX)
          return fail(error, "input allocation overflow");
        initialCopies.push_back(id);
        g.inputLocations[storage] = memory;
      }
    }
  auto read = [&](size_t c, size_t event, bool consumption) {
    auto &value = g.copies[c];
    auto &ev = g.events[event];
    ev.reads.push_back(c);
    value.users.push_back(event);
    if (value.createdBy >= 0)
      ev.dependencies.push_back(value.createdBy);
    if (consumption) {
      auto region = g.versions[value.version].region;
      auto storage = g.regions[region].storage;
      auto id = g.dataEdges.size();
      ev.dataEdges.push_back(id);
      value.consumerEdges.push_back(id);
      g.dataEdges.push_back({g.versions[value.version].producer,
                             event,
                             region,
                             value.version,
                             c,
                             {g.allocations[value.allocation].memory,
                              value.layout, g.source.storages[storage].dtype}});
    }
  };
  auto compatible = [&](const StaticTransfer &link, size_t version) {
    auto region = g.versions[version].region;
    auto dtype = g.source.storages[g.regions[region].storage].dtype;
    auto n = bytes(g, region);
    return n > 0 && n >= link.minBytes && n <= link.maxBytes &&
           link.temporaryBytes == 0 &&
           (link.sourceLayout.empty() || link.sourceLayout == "row_major") &&
           (link.targetLayout.empty() || link.targetLayout == "row_major") &&
           (link.sourceDtype.empty() || link.sourceDtype == dtype) &&
           (link.targetDtype.empty() || link.targetDtype == dtype);
  };
  std::function<bool(size_t, const std::string &, size_t, size_t, size_t &)>
      ensure;
  ensure = [&](size_t version, const std::string &memory, size_t task,
               size_t group, size_t &result) {
    if (g.versions[version].producer == -2)
      return fail(error, "read before its selected source producer");
    size_t forced = SIZE_MAX;
    auto storage = g.regions[g.versions[version].region].storage;
    for (size_t i = 0; i < choice.transferChoices.size(); ++i) {
      auto &t = choice.transferChoices[i];
      if (t.storage == storage && t.consumerTask == task &&
          t.targetMemory == memory) {
        if (forced != SIZE_MAX)
          return fail(error, "duplicate selected transfer requirement");
        forced = i;
      }
    }
    auto found = available.find({version, memory});
    if (forced == SIZE_MAX && found != available.end()) {
      result = found->second;
      return true;
    }
    size_t source = SIZE_MAX;
    std::vector<std::string> path;
    if (forced != SIZE_MAX) {
      const auto &t = choice.transferChoices[forced];
      transferUsed[forced] = true;
      auto f = available.find({version, t.sourceMemory});
      if (f == available.end())
        return fail(error, "selected source copy is not valid");
      source = f->second;
      path = t.links;
      std::string current = t.sourceMemory;
      for (auto &id : path) {
        auto l = programLink(target, id);
        if (!l || l->source != current || !compatible(*l, version))
          return fail(error,
                      "selected path violates topology/layout/dtype/payload");
        current = l->target;
      }
      if (current != memory)
        return fail(error, "selected path does not reach requested memory");
    } else {
      StaticConfig routes = target;
      routes.links.clear();
      for (auto &link : target.links)
        if (compatible(link, version))
          routes.links.push_back(link);
      for (auto &entry : available)
        if (entry.first.first == version) {
          std::vector<std::string> route;
          std::string ignored;
          if (findMemoryPath(routes, entry.first.second, memory, route,
                             ignored) &&
              (source == SIZE_MAX || route.size() < path.size())) {
            source = entry.second;
            path = std::move(route);
          }
        }
      if (source == SIZE_MAX)
        return fail(error, "no legal source-to-consumer path");
    }
    // Export transfers belong to the execution group which actually owns the
    // selected source copy, including a retained copy imported by a later
    // group. Do not infer ownership from the original logical producer alone.
    size_t owner = group;
    if (!path.empty() && owner == SIZE_MAX && !g.executionGroups.empty()) {
      auto creator = g.copies[source].createdBy;
      if (creator >= 0)
        owner = eventOwners[size_t(creator)];
      else if (g.versions[version].producer >= 0)
        owner = eventOwners[size_t(g.versions[version].producer)];
      if (owner == SIZE_MAX)
        return fail(error,
                    "output transfer has no source-copy execution owner");
    }
    for (auto &id : path) {
      auto link = programLink(target, id);
      MappingEvent ev;
      ev.kind = "transfer";
      ev.task = std::min(task, g.source.tasks.size() - 1);
      ev.phase = g.instances.size();
      ev.resource = link->resource;
      ev.memory = link->target;
      ev.link = id;
      ev.transferRegion = g.regions[g.versions[version].region];
      ev.transferBytes = bytes(g, g.versions[version].region);
      auto event = g.events.size();
      g.events.push_back(ev);
      eventOwners.push_back(owner);
      read(source, event, false);
      auto c = copy(version, link->target, event, task == g.source.tasks.size(),
                    true);
      if (c == SIZE_MAX)
        return fail(error, "transfer allocation overflow");
      g.events[event].writes.push_back(c);
      if (g.transferBytes > INT64_MAX - ev.transferBytes)
        return fail(error, "traffic overflow");
      g.transferBytes += ev.transferBytes;
      source = c;
      if (owner != SIZE_MAX)
        g.executionGroups[owner].memberEvents.push_back(event);
    }
    result = source;
    return true;
  };
  size_t previousGroup = SIZE_MAX;
  for (size_t old = 0; old < reference.events.size(); ++old)
    if (instanceAt[old] != SIZE_MAX) {
      auto instance = reference.instances[instanceAt[old]];
      auto binding = places[instance.task];
      auto group = oldGroup[old];
      if (choice.policy == ProgramDataPolicy::PhaseLocal &&
          (group == SIZE_MAX || group != previousGroup)) {
        for (auto it = available.begin(); it != available.end();)
          if (imported[it->second])
            it = available.erase(it);
          else
            ++it;
      }
      previousGroup = group;
      std::vector<size_t> inputs;
      for (auto c : reference.events[old].reads) {
        size_t selected;
        if (!ensure(reference.copies[c].version, binding.memory, instance.task,
                    group, selected))
          return false;
        inputs.push_back(selected);
      }
      MappingEvent ev = reference.events[old];
      ev.dependencies.clear();
      ev.reads.clear();
      ev.writes.clear();
      ev.dataEdges.clear();
      ev.orderEdges.clear();
      ev.resource = binding.resource;
      ev.memory = binding.memory;
      ev.phase = g.instances.size();
      auto event = g.events.size();
      g.events.push_back(ev);
      eventOwners.push_back(group);
      remap[old] = event;
      for (auto c : inputs)
        read(c, event, true);
      for (auto c : reference.events[old].writes) {
        auto version = reference.copies[c].version;
        g.versions[version].producer = event;
        auto selected = copy(version, binding.memory, event, false, false);
        if (selected == SIZE_MAX)
          return fail(error, "compute allocation overflow");
        g.events[event].writes.push_back(selected);
      }
      instance.event = event;
      instance.phase = ev.phase;
      instance.implementation = binding.implementation;
      instance.resourceCount = binding.resourceCount;
      instance.resourceShape = binding.resourceShape;
      g.instances.push_back(instance);
      if (group != SIZE_MAX)
        g.executionGroups[group].memberEvents.push_back(event);
    }
  for (auto &edge : reference.orderEdges) {
    if (edge.producer >= remap.size() || edge.consumer >= remap.size() ||
        remap[edge.producer] == SIZE_MAX || remap[edge.consumer] == SIZE_MAX)
      return fail(error, "order edge is not bound to a source instance");
    auto from = remap[edge.producer], to = remap[edge.consumer];
    g.events[to].dependencies.push_back(from);
    g.events[to].orderEdges.push_back(g.orderEdges.size());
    g.orderEdges.push_back({from, to, edge.reason});
  }
  std::map<size_t, size_t> finalVersion;
  for (size_t v = 0; v < g.versions.size(); ++v) {
    if (g.versions[v].producer == -2)
      return fail(error, "source version producer was lost");
    auto region = g.versions[v].region;
    auto it = finalVersion.find(region);
    if (it == finalVersion.end() || reference.versions[v].producer >
                                        reference.versions[it->second].producer)
      finalVersion[region] = v;
  }
  for (auto &entry : finalVersion) {
    auto storage = g.regions[entry.first].storage;
    if (g.source.storages[storage].role != "output")
      continue;
    std::string memory;
    auto declared =
        target.assumptions.outputLocations.find(g.source.storages[storage].id);
    if (declared != target.assumptions.outputLocations.end())
      memory = declared->second;
    else if (reference.outputLocations.count(storage))
      memory = reference.outputLocations.at(storage);
    else {
      for (auto &c : reference.copies)
        if (c.version == entry.second) {
          auto &a = reference.allocations[c.allocation];
          auto m = programMemory(target, a.memory);
          if (m && m->kind == "system-bo")
            memory = m->id;
        }
    }
    if (memory.empty())
      return fail(error, "final output has no declared boundary location");
    size_t c;
    if (!ensure(entry.second, memory, g.source.tasks.size(), SIZE_MAX, c))
      return false;
    outputCopies.push_back(c);
    g.outputLocations[storage] = memory;
  }
  for (size_t i = 0; i < transferUsed.size(); ++i)
    if (!transferUsed[i])
      return fail(error,
                  "selected transfer has no actual consumer requirement");
  for (size_t i = 0; i < g.copies.size(); ++i) {
    auto &c = g.copies[i];
    canonical(c.users);
    canonical(c.consumerEdges);
    if (!c.users.empty())
      c.lastUsers = {c.users.back()};
    c.releasedAfter =
        c.users.empty() ? c.availableFrom + 1 : c.users.back() + 1;
    if (std::find(outputCopies.begin(), outputCopies.end(), i) !=
        outputCopies.end())
      c.releasedAfter = g.events.size();
    auto &a = g.allocations[c.allocation];
    if (c.createdBy >= 0)
      a.beginPhase = g.events[c.createdBy].phase;
    a.endPhase =
        c.users.empty() ? a.beginPhase + 1 : g.events[c.users.back()].phase + 1;
  }
  for (auto &group : g.executionGroups) {
    canonical(group.memberEvents);
    if (group.memberEvents.empty())
      return fail(error, "empty selected group");
    group.beginPhase = SIZE_MAX;
    group.endPhase = 0;
    for (auto event : group.memberEvents) {
      group.beginPhase = std::min(group.beginPhase, g.events[event].phase);
      group.endPhase = std::max(group.endPhase, g.events[event].phase + 1);
    }
  }
  std::vector<size_t> eventGroup(g.events.size(), SIZE_MAX);
  for (size_t i = 0; i < g.executionGroups.size(); ++i)
    for (auto e : g.executionGroups[i].memberEvents)
      eventGroup[e] = i;
  for (size_t i = 0; i < g.events.size(); ++i) {
    canonical(g.events[i].dependencies);
    if (eventGroup[i] != SIZE_MAX)
      for (auto p : g.events[i].dependencies)
        if (eventGroup[p] != SIZE_MAX && eventGroup[p] != eventGroup[i])
          g.executionGroups[eventGroup[i]].dependencies.push_back(
              eventGroup[p]);
  }
  for (auto &group : g.executionGroups)
    canonical(group.dependencies);
  for (const auto &m : g.memories)
    for (size_t event = 0; event < g.events.size(); ++event) {
      int64_t live = 0;
      std::set<size_t> counted;
      for (auto &c : g.copies)
        if (c.availableFrom <= event && event < c.releasedAfter &&
            counted.insert(c.allocation).second) {
          auto &a = g.allocations[c.allocation];
          if (a.memory == m.id) {
            if (live > m.capacityBytes - a.bytes)
              return fail(error,
                          "selected copy lifetimes exceed memory capacity");
            live += a.bytes;
          }
        }
      g.peakMemoryBytes[m.id] = std::max(g.peakMemoryBytes[m.id], live);
    }
  if (g.executionRegions.size() > 1)
    return fail(error, "use the existing heterogeneous region evaluator for "
                       "multiple invocation regions");
  for (auto &region : g.executionRegions) {
    region.beginEvent = 0;
    region.endEvent = g.events.size();
    region.memberEvents.clear();
    for (size_t event = 0; event < g.events.size(); ++event)
      region.memberEvents.push_back(event);
    region.inputCopies = initialCopies;
    region.outputCopies = outputCopies;
    region.deviceStages.clear();
    for (size_t i = 0; i < g.executionGroups.size(); ++i)
      region.deviceStages.push_back(i);
    if (region.submissionCount > 0) {
      g.regionEvents = {
          {"visible-to-device",
           0,
           region.submissionResource,
           {},
           initialCopies},
          {"submit", 0, region.submissionResource, {0}, {}},
          {"complete", 0, region.submissionResource, {1}, {}},
          {"visible-to-host", 0, region.submissionResource, {2}, outputCopies}};
      region.invocationLifecycle = "host-device";
    }
  }
  return refreshProgramGraphSchedule(g, error) &&
         checkProgramGraphSource(reference.source, g, error) &&
         checkProgramGraph(g, error);
}
