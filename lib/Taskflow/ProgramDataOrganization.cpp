#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/ProgramMapping.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <tuple>
using namespace orbit::taskflow;
namespace {
bool reject(std::string &error, const std::string &message) {
  error = "program data organization: " + message;
  return false;
}
struct Effects {
  std::set<size_t> reads, writes;
};
bool sourceEffects(const GenericAnalysis &a, bool partition,
                   std::vector<Effects> &effects,
                   std::vector<MappingRegion> &regions, std::string &error) {
  std::vector<std::vector<std::set<int64_t>>> cuts(a.storages.size());
  for (size_t storage = 0; storage < a.storages.size(); ++storage) {
    for (auto extent : a.storages[storage].shape) {
      if (extent <= 0)
        return reject(error, "invalid source storage shape");
      cuts[storage].push_back({0, extent});
    }
  }
  for (const auto &task : a.tasks)
    for (const auto &access : task.accesses) {
      if (access.storage >= a.storages.size())
        return reject(error, "invalid source access storage");
      const auto &storage = a.storages[access.storage];
      if (!access.exactRegion || access.region.size() != storage.shape.size())
        return reject(
            error, "whole-task adapter requires exact source access coverage");
      for (size_t d = 0; d < access.region.size(); ++d) {
        const auto &r = access.region[d];
        if (r.lower < 0 || r.upper > storage.shape[d] || r.lower >= r.upper)
          return reject(error, "source access outside storage");
        if (!partition && (r.lower != 0 || r.upper != storage.shape[d]))
          return reject(
              error, "whole-task adapter requires full-storage source regions");
        cuts[access.storage][d].insert(r.lower);
        cuts[access.storage][d].insert(r.upper);
      }
    }
  regions.clear();
  effects.assign(a.tasks.size(), {});
  for (size_t storage = 0; storage < cuts.size(); ++storage) {
    size_t count = 1;
    for (const auto &axis : cuts[storage]) {
      if (count > 1048576 / (axis.size() - 1))
        return reject(error, "exact region host-plan expansion guard exceeded");
      count *= axis.size() - 1;
    }
    if (regions.size() > 1048576 - count)
      return reject(error, "exact region host-plan expansion guard exceeded");
    MappingRegion cell;
    cell.storage = storage;
    std::function<void(size_t)> emit = [&](size_t axis) {
      if (axis == cuts[storage].size()) {
        regions.push_back(cell);
        return;
      }
      const auto &bounds = cuts[storage][axis];
      for (auto lo = bounds.begin(), hi = std::next(lo); hi != bounds.end();
           ++lo, ++hi) {
        cell.offsets.push_back(*lo);
        cell.sizes.push_back(*hi - *lo);
        emit(axis + 1);
        cell.offsets.pop_back();
        cell.sizes.pop_back();
      }
    };
    emit(0);
  }
  for (size_t task = 0; task < a.tasks.size(); ++task)
    for (const auto &access : a.tasks[task].accesses)
      for (size_t cell = 0; cell < regions.size(); ++cell) {
        const auto &region = regions[cell];
        if (region.storage != access.storage)
          continue;
        bool covered = true;
        for (size_t d = 0; d < region.sizes.size(); ++d)
          covered &=
              access.region[d].lower <= region.offsets[d] &&
              region.offsets[d] + region.sizes[d] <= access.region[d].upper;
        if (covered) {
          if (access.read)
            effects[task].reads.insert(cell);
          if (access.write)
            effects[task].writes.insert(cell);
        }
      }
  return true;
}
int64_t storageBytes(const GenericStorage &s) {
  auto type = mlir::cast<mlir::MemRefType>(s.root.getType());
  int64_t n = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
  for (int64_t x : s.shape) {
    if (x <= 0 || n > INT64_MAX / x)
      return -1;
    n *= x;
  }
  return n;
}
int64_t regionBytes(const GenericAnalysis &a, const MappingRegion &region) {
  GenericStorage storage = a.storages[region.storage];
  storage.shape = region.sizes;
  return storageBytes(storage);
}
int64_t allocationBytes(const GenericAnalysis &a, const MappingRegion &region,
                        const StaticMemory &memory) {
  int64_t bytes = regionBytes(a, region), alignment = memory.alignmentBytes;
  if (bytes <= 0 || alignment <= 0)
    return -1;
  int64_t remainder = bytes % alignment,
          padding = remainder ? alignment - remainder : 0;
  if (bytes > INT64_MAX - padding)
    return -1;
  return bytes + padding;
}
bool placements(const GenericAnalysis &a, const StaticConfig &target,
                const ProgramDataOrganizationChoice &choice,
                std::vector<MappingTaskPlacement> &out, std::string &error) {
  if (choice.placements.size() != a.tasks.size())
    return reject(error, "one explicit placement per source task required");
  if (!validateMemoryArchitecture(target, error))
    return false;
  out.resize(a.tasks.size());
  std::set<size_t> seen;
  for (const auto &p : choice.placements) {
    if (p.task >= a.tasks.size() || !seen.insert(p.task).second)
      return reject(error, "invalid or duplicate task placement");
    auto resource =
        std::find_if(target.resources.begin(), target.resources.end(),
                     [&](const auto &r) { return r.id == p.resource; });
    auto memory = std::find_if(target.memories.begin(), target.memories.end(),
                               [&](const auto &m) { return m.id == p.memory; });
    if (resource == target.resources.end() || resource->capacity < 1 ||
        resource->kind != "compute" || memory == target.memories.end() ||
        memory->capacityBytes <= 0 ||
        !memoryAccessible(target, p.memory, p.resource))
      return reject(error, "task resource cannot access selected memory");
    if (p.resourceCount < 1 || p.resourceCount > resource->capacity)
      return reject(error, "placement resource count exceeds target capacity");
    if (!p.resourceShape.empty()) {
      int64_t product = 1;
      for (auto extent : p.resourceShape) {
        if (extent <= 0 || product > INT64_MAX / extent)
          return reject(error, "invalid placement resource shape");
        product *= extent;
      }
      if (product != p.resourceCount)
        return reject(error, "placement resource shape/count mismatch");
    }
    out[p.task] = p;
  }
  return true;
}
bool linkSupports(const StaticTransfer &l, const GenericStorage &s,
                  int64_t bytes) {
  return (l.sourceLayout.empty() || l.sourceLayout == "row_major") &&
         (l.targetLayout.empty() || l.targetLayout == "row_major") &&
         (l.sourceDtype.empty() || l.sourceDtype == s.dtype) &&
         (l.targetDtype.empty() || l.targetDtype == s.dtype) &&
         bytes >= l.minBytes && bytes <= l.maxBytes && l.temporaryBytes == 0;
}
const StaticTransfer *selectedLink(const StaticConfig &target,
                                   const std::string &id) {
  auto found = std::find_if(target.links.begin(), target.links.end(),
                            [&](const auto &l) { return l.id == id; });
  return found == target.links.end() ? nullptr : &*found;
}
const ProgramTransferChoice *
transferChoice(const ProgramDataOrganizationChoice &choice, size_t storage,
               size_t consumer, const std::string &target) {
  for (const auto &v : choice.transferChoices)
    if (v.storage == storage && v.consumerTask == consumer &&
        v.targetMemory == target)
      return &v;
  return nullptr;
}
bool validateTransferChoices(const GenericAnalysis &a,
                             const StaticConfig &target,
                             const ProgramDataOrganizationChoice &choice,
                             const std::vector<Effects> &effects,
                             const std::vector<MappingRegion> &regions,
                             std::string &error) {
  std::set<std::tuple<size_t, size_t, std::string>> seen;
  for (const auto &v : choice.transferChoices) {
    if (v.storage >= a.storages.size() || v.consumerTask > a.tasks.size() ||
        !seen.emplace(v.storage, v.consumerTask, v.targetMemory).second)
      return reject(error, "invalid or duplicate transfer choice");
    auto memory = [&](const std::string &id) {
      return std::any_of(target.memories.begin(), target.memories.end(),
                         [&](const auto &m) { return m.id == id; });
    };
    if (!memory(v.sourceMemory) || !memory(v.targetMemory))
      return reject(error, "transfer choice names unknown memory");
    bool demanded = false;
    if (v.consumerTask < a.tasks.size()) {
      for (auto cell : effects[v.consumerTask].reads)
        demanded |= regions[cell].storage == v.storage;
      auto place =
          std::find_if(choice.placements.begin(), choice.placements.end(),
                       [&](const auto &p) { return p.task == v.consumerTask; });
      demanded &=
          place != choice.placements.end() && place->memory == v.targetMemory;
    } else {
      auto output =
          target.assumptions.outputLocations.find(a.storages[v.storage].id);
      demanded = a.storages[v.storage].role == "output" &&
                 output != target.assumptions.outputLocations.end() &&
                 output->second == v.targetMemory;
    }
    if (!demanded)
      return reject(error,
                    "transfer choice has no matching consumption demand");
    std::string current = v.sourceMemory;
    std::set<std::string> visited{current};
    for (const auto &id : v.links) {
      auto *link = selectedLink(target, id);
      if (!link || link->source != current ||
          !visited.insert(link->target).second)
        return reject(
            error, "transfer choice has unknown, disconnected or cyclic links");
      current = link->target;
    }
    if (current != v.targetMemory)
      return reject(error, "transfer choice path does not reach target memory");
  }
  return true;
}
} // namespace
bool orbit::taskflow::planProgramDataOrganization(
    const GenericAnalysis &a, const StaticConfig &target,
    const ProgramDataOrganizationChoice &choice, ProgramCommunicationPlan &c,
    std::string &error) {
  c = {};
  c.dataPolicy = choice.policy;
  c.bodyFingerprint = a.bodyFingerprint;
  if (choice.semanticDependenciesOnly)
    c.schedule.policy = "dependency-order-witness";
  c.domains = target.domains;
  c.memories = target.memories;
  c.resources = target.resources;
  c.links = target.links;
  for (size_t storage = 0; storage < a.storages.size(); ++storage) {
    const auto &source = a.storages[storage];
    if (source.role == "input") {
      auto location = target.assumptions.initialLocations.find(source.id);
      if (location == target.assumptions.initialLocations.end())
        return reject(error, "missing declared input boundary location");
      c.inputLocations[storage] = location->second;
    } else if (source.role == "output") {
      auto location = target.assumptions.outputLocations.find(source.id);
      if (location == target.assumptions.outputLocations.end())
        return reject(error, "missing declared output boundary location");
      c.outputLocations[storage] = location->second;
    }
  }
  if (!placements(a, target, choice, c.placements, error))
    return false;
  std::vector<Effects> effects;
  if (!sourceEffects(a, choice.partitionExactRegions, effects, c.regions,
                     error))
    return false;
  if (!validateTransferChoices(a, target, choice, effects, c.regions, error))
    return false;
  c.computeEstimates = choice.computeEstimates;
  c.clockMHz = choice.clockMHz;
  std::map<size_t, double> computeMs;
  bool known = true;
  double duration = 0;
  for (const auto &estimate : choice.computeEstimates) {
    double ms;
    if (estimate.task >= a.tasks.size() || computeMs.count(estimate.task) ||
        estimate.resource != c.placements[estimate.task].resource ||
        estimate.implementation != c.placements[estimate.task].implementation ||
        estimate.resourceCount != c.placements[estimate.task].resourceCount ||
        !convertComputeEstimateToMs(estimate, choice.clockMHz, ms, error))
      return reject(error, "invalid compute estimate: " + error);
    auto resource =
        std::find_if(target.resources.begin(), target.resources.end(),
                     [&](const auto &r) { return r.id == estimate.resource; });
    if (resource == target.resources.end() ||
        estimate.resourceCount > resource->capacity)
      return reject(error, "compute estimate exceeds resource capacity");
    computeMs[estimate.task] = ms;
  }
  for (size_t s = 0; s < a.storages.size(); ++s) {
    if (storageBytes(a.storages[s]) < 0)
      return reject(error, "storage byte count overflow");
  }
  std::vector<int64_t> latest(c.regions.size(), -1);
  std::map<std::pair<size_t, std::string>, size_t> available;
  auto allocation = [&](size_t cell, const std::string &memory, size_t phase) {
    auto location = std::find_if(target.memories.begin(), target.memories.end(),
                                 [&](const auto &m) { return m.id == memory; });
    int64_t bytes = location == target.memories.end()
                        ? -1
                        : allocationBytes(a, c.regions[cell], *location);
    if (bytes <= 0) {
      reject(error, "invalid or overflowing aligned allocation size");
      return SIZE_MAX;
    }
    size_t i = c.allocations.size();
    c.allocations.push_back({c.regions[cell].storage, memory,
                             "copy-pool-" + std::to_string(i), bytes, phase,
                             phase + 1});
    return i;
  };
  auto create = [&](size_t version, size_t al, int64_t event) {
    size_t i = c.copies.size();
    MappingCopy copy;
    copy.version = version;
    copy.allocation = al;
    copy.createdBy = event;
    copy.availableFrom = event < 0 ? 0 : size_t(event);
    c.copies.push_back(copy);
    available[{c.versions[version].region, c.allocations[al].memory}] = i;
    if (event >= 0)
      c.events[event].writes.push_back(i);
    return i;
  };
  for (size_t s = 0; s < c.regions.size(); ++s)
    if (a.storages[c.regions[s].storage].role == "input") {
      auto loc = target.assumptions.initialLocations.find(
          a.storages[c.regions[s].storage].id);
      if (loc == target.assumptions.initialLocations.end())
        return reject(error, "missing initial input location");
      if (std::none_of(target.memories.begin(), target.memories.end(),
                       [&](const auto &m) { return m.id == loc->second; }))
        return reject(error, "unknown initial memory");
      size_t v = c.versions.size();
      c.versions.push_back({s, -1});
      latest[s] = v;
      size_t al = allocation(s, loc->second, 0);
      if (al == SIZE_MAX)
        return false;
      create(v, al, -1);
    }
  auto start = [&](std::string kind, size_t task, std::string memory,
                   std::string resource, std::string link) {
    MappingEvent e;
    e.kind = kind;
    e.task = task;
    e.phase = task;
    e.memory = memory;
    e.resource = resource;
    e.link = link;
    if (!choice.semanticDependenciesOnly && !c.events.empty())
      e.dependencies.push_back(c.events.size() - 1);
    c.events.push_back(e);
    return c.events.size() - 1;
  };
  auto read = [&](size_t copy, size_t event, bool data) {
    auto &v = c.copies[copy];
    c.events[event].reads.push_back(copy);
    v.users.push_back(event);
    v.lastUsers = {event};
    if (v.createdBy >= 0)
      c.events[event].dependencies.push_back(v.createdBy);
    if (data) {
      size_t edge = c.dataEdges.size();
      auto &version = c.versions[v.version];
      size_t s = c.regions[version.region].storage;
      c.dataEdges.push_back({version.producer,
                             event,
                             version.region,
                             v.version,
                             copy,
                             {c.allocations[v.allocation].memory, v.layout,
                              a.storages[s].dtype}});
      c.events[event].dataEdges.push_back(edge);
      v.consumerEdges.push_back(edge);
    }
  };
  std::function<bool(size_t, const std::string &, size_t, size_t &)> ensure;
  ensure = [&](size_t cell, const std::string &memory, size_t consumer,
               size_t &copy) {
    size_t task = std::min(consumer, a.tasks.size() - 1);
    size_t storage = c.regions[cell].storage;
    if (latest[cell] < 0)
      return reject(error, "source read has no initialized version");
    const auto *selected = transferChoice(choice, storage, consumer, memory);
    size_t source = SIZE_MAX;
    std::vector<std::string> path;
    int64_t b = regionBytes(a, c.regions[cell]);
    if (selected) {
      auto found = available.find({cell, selected->sourceMemory});
      if (found == available.end() ||
          c.copies[found->second].version != size_t(latest[cell]))
        return reject(
            error, "selected source memory has no current valid version copy");
      source = found->second;
      path = selected->links;
      for (const auto &id : path)
        if (!linkSupports(*selectedLink(target, id), a.storages[storage], b))
          return reject(
              error,
              "selected link does not support required payload/layout/dtype");
      if (path.empty()) {
        copy = source;
        return true;
      }
      if (choice.policy == ProgramDataPolicy::RetainAcrossUsers) {
        auto foundTarget = available.find({cell, memory});
        if (foundTarget != available.end() &&
            c.copies[foundTarget->second].version == size_t(latest[cell]))
          return reject(
              error,
              "selected transfer would duplicate a retained target copy");
      }
    } else {
      auto found = available.find({cell, memory});
      if (found != available.end() &&
          c.copies[found->second].version == size_t(latest[cell])) {
        copy = found->second;
        return true;
      }
      StaticConfig routes = target;
      routes.links.clear();
      // Every compatible parallel link remains in the connectivity domain.
      for (const auto &l : target.links)
        if (linkSupports(l, a.storages[storage], b))
          routes.links.push_back(l);
      for (const auto &entry : available)
        if (entry.first.first == cell &&
            c.copies[entry.second].version == size_t(latest[cell])) {
          std::vector<std::string> candidate;
          std::string ignored;
          if (findMemoryPath(routes, entry.first.second, memory, candidate,
                             ignored) &&
              (source == SIZE_MAX || candidate.size() < path.size())) {
            source = entry.second;
            path = std::move(candidate);
          }
        }
      if (source == SIZE_MAX || path.empty())
        return reject(error, "no legal transfer path for required copy");
    }
    for (const auto &id : path) {
      auto link = std::find_if(target.links.begin(), target.links.end(),
                               [&](const auto &l) { return l.id == id; });
      size_t event =
          start("transfer", task, link->target, link->resource, link->id);
      read(source, event, false);
      c.events[event].transferRegion = c.regions[cell];
      c.events[event].transferBytes = b;
      size_t al = allocation(cell, link->target, task);
      if (al == SIZE_MAX)
        return false;
      copy = create(latest[cell], al, event);
      if (c.transferBytes > INT64_MAX - b)
        return reject(error, "traffic overflow");
      c.transferBytes += b;
      if (std::isfinite(link->bandwidthBytesPerMs) &&
          std::isfinite(link->latencyMs) && link->bandwidthBytesPerMs > 0 &&
          link->latencyMs >= 0 && !link->provenance.source.empty())
        duration += link->latencyMs + b / link->bandwidthBytesPerMs;
      else
        known = false;
      source = copy;
    }
    return true;
  };
  for (size_t task = 0; task < a.tasks.size(); ++task) {
    for (size_t dep : a.tasks[task].dependencies)
      if (dep >= task)
        return reject(error,
                      "whole-task adapter requires source topological order");
    const auto &place = c.placements[task];
    std::vector<size_t> reads;
    for (size_t s : effects[task].reads) {
      if (latest[s] < 0 && effects[task].writes.count(s))
        return reject(
            error,
            "whole-task adapter does not resolve internally initialized "
            "read/write effects; provide a separate source initializer task");
      size_t copy;
      if (!ensure(s, place.memory, task, copy))
        return false;
      reads.push_back(copy);
    }
    size_t event = start("compute", task, place.memory, place.resource, "");
    for (size_t copy : reads)
      read(copy, event, true);
    for (size_t s : effects[task].writes) {
      size_t v = c.versions.size();
      c.versions.push_back({s, int64_t(event)});
      latest[s] = v;
      size_t al = allocation(s, place.memory, task);
      if (al == SIZE_MAX)
        return false;
      create(v, al, event);
    }
    if (computeMs.count(task))
      duration += computeMs[task];
    else
      known = false;
    for (size_t dep : a.tasks[task].dependencies) {
      bool raw = false;
      for (size_t id : c.events[event].dataEdges) {
        auto producer = c.dataEdges[id].producer;
        if (producer >= 0 && c.events[producer].task == dep)
          raw = true;
      }
      if (!raw) {
        auto prior = std::find_if(c.events.rbegin(), c.events.rend(),
                                  [&](const auto &e) {
                                    return e.kind == "compute" && e.task == dep;
                                  });
        if (prior == c.events.rend())
          return reject(error, "missing order predecessor");
        size_t predecessor = size_t(c.events.rend() - prior - 1);
        if (choice.semanticDependenciesOnly)
          c.events[event].dependencies.push_back(predecessor);
        c.events[event].orderEdges.push_back(c.orderEdges.size());
        c.orderEdges.push_back(
            {predecessor, event,
             "source task dependency without value consumption"});
      }
    }
    if (choice.policy == ProgramDataPolicy::PhaseLocal) {
      // Every imported location is task-local, including intermediate copies
      // along a multi-hop route. Only external and compute-produced copies
      // remain valid sources for a later task under this policy.
      for (auto it = available.begin(); it != available.end();) {
        auto creator = c.copies[it->second].createdBy;
        if (creator >= 0 && c.events[creator].kind == "transfer")
          it = available.erase(it);
        else
          ++it;
      }
    }
  }
  for (size_t s = 0; s < c.regions.size(); ++s)
    if (a.storages[c.regions[s].storage].role == "output") {
      auto dest = target.assumptions.outputLocations.find(
          a.storages[c.regions[s].storage].id);
      if (dest == target.assumptions.outputLocations.end())
        return reject(error, "missing output boundary location");
      size_t copy;
      if (!ensure(s, dest->second, a.tasks.size(), copy))
        return false;
    }
  for (auto &copy : c.copies) {
    copy.releasedAfter =
        copy.users.empty() ? copy.availableFrom + 1 : copy.users.back() + 1;
    auto &al = c.allocations[copy.allocation];
    if (copy.createdBy >= 0)
      al.beginPhase = c.events[copy.createdBy].phase;
    al.endPhase = copy.users.empty() ? al.beginPhase + 1
                                     : c.events[copy.users.back()].phase + 1;
    size_t cell = c.versions[copy.version].region;
    size_t storage = c.regions[cell].storage;
    if (a.storages[storage].role == "output" &&
        target.assumptions.outputLocations.at(a.storages[storage].id) ==
            al.memory &&
        copy.version == size_t(latest[cell])) {
      al.endPhase = a.tasks.size();
      copy.releasedAfter = c.events.size();
    }
  }
  for (auto &e : c.events) {
    std::sort(e.dependencies.begin(), e.dependencies.end());
    e.dependencies.erase(
        std::unique(e.dependencies.begin(), e.dependencies.end()),
        e.dependencies.end());
  }
  for (const auto &memory : target.memories)
    for (size_t event = 0; event < c.events.size(); ++event) {
      int64_t live = 0;
      std::set<size_t> counted;
      for (const auto &copy : c.copies)
        if (copy.availableFrom <= event && event < copy.releasedAfter &&
            counted.insert(copy.allocation).second) {
          const auto &al = c.allocations[copy.allocation];
          if (al.memory != memory.id)
            continue;
          if (al.bytes > INT64_MAX - live)
            return reject(error, "memory accounting overflow");
          live += al.bytes;
        }
      c.peakMemoryBytes[memory.id] =
          std::max(c.peakMemoryBytes[memory.id], live);
    }
  if (!choice.semanticDependenciesOnly && known && std::isfinite(duration))
    c.estimatedMakespanMs = duration;
  if (!finalizeProgramGraph(a, c, nullptr, error))
    return false;
  return checkProgramDataOrganization(a, target, choice, c, error);
}
bool orbit::taskflow::checkProgramDataOrganization(
    const GenericAnalysis &a, const StaticConfig &target,
    const ProgramDataOrganizationChoice &choice,
    const ProgramCommunicationPlan &c, std::string &error) {
  std::vector<Effects> effects;
  std::vector<MappingTaskPlacement> place;
  std::vector<MappingRegion> cells;
  if (!sourceEffects(a, choice.partitionExactRegions, effects, cells, error) ||
      !placements(a, target, choice, place, error))
    return false;
  if (!validateTransferChoices(a, target, choice, effects, cells, error))
    return false;
  for (const auto &storage : a.storages)
    if (storageBytes(storage) <= 0)
      return reject(error,
                    "source storage byte count is invalid or overflows int64");
  std::map<size_t, std::string> inputs, outputs;
  for (size_t storage = 0; storage < a.storages.size(); ++storage) {
    const auto &source = a.storages[storage];
    if (source.role == "input") {
      auto location = target.assumptions.initialLocations.find(source.id);
      if (location == target.assumptions.initialLocations.end())
        return reject(error, "target lacks source input boundary");
      inputs[storage] = location->second;
    } else if (source.role == "output") {
      auto location = target.assumptions.outputLocations.find(source.id);
      if (location == target.assumptions.outputLocations.end())
        return reject(error, "target lacks source output boundary");
      outputs[storage] = location->second;
    }
  }
  if (c.inputLocations != inputs || c.outputLocations != outputs)
    return reject(error, "whole-task input/output boundary differs from source "
                         "roles and target locations");
  if (c.schedule.policy != (choice.semanticDependenciesOnly
                                ? "dependency-order-witness"
                                : "serial-event-order"))
    return reject(
        error,
        "whole-task schedule policy differs from selected dependency contract");
  if (c.clockMHz != choice.clockMHz || c.allocationLifetime != "copy-events")
    return reject(
        error, "whole-task compute clock/allocation lifetime contract changed");
  if (c.executionRegions.size() != 1)
    return reject(error, "whole-task graph requires one host execution region");
  const auto &host = c.executionRegions.front();
  if (host.id != "host-program" || host.placement != "host" ||
      !host.submissionResource.empty() || host.beginEvent != 0 ||
      host.endEvent != c.events.size() || host.submissionCount != 0 ||
      !host.deviceStages.empty())
    return reject(
        error, "whole-task host region cannot fabricate a device submission");
  if (c.dataPolicy != choice.policy || c.bodyFingerprint != a.bodyFingerprint ||
      c.regions.size() != cells.size() || c.placements.size() != place.size())
    return reject(error, "source/policy/placement coverage mismatch");
  for (size_t t = 0; t < place.size(); ++t)
    if (c.placements[t].task != t ||
        c.placements[t].memory != place[t].memory ||
        c.placements[t].resource != place[t].resource ||
        c.placements[t].implementation != place[t].implementation ||
        c.placements[t].resourceCount != place[t].resourceCount ||
        c.placements[t].resourceShape != place[t].resourceShape)
      return reject(error, "source task placement changed");
  for (size_t s = 0; s < cells.size(); ++s)
    if (c.regions[s].storage != cells[s].storage ||
        c.regions[s].sizes != cells[s].sizes ||
        c.regions[s].offsets != cells[s].offsets)
      return reject(error, "logical region differs from source");
  StaticConfig copyTarget;
  copyTarget.memories = c.memories;
  copyTarget.resources = c.resources;
  copyTarget.links = c.links;
  copyTarget.domains = c.domains;
  if (staticFingerprint(serializeMemoryArchitecture(copyTarget)) !=
      staticFingerprint(serializeMemoryArchitecture(target)))
    return reject(error, "target architecture contract changed");
  // Target tables are contracts, not candidate-mutable resources.
  if (c.memories.size() != target.memories.size() ||
      c.resources.size() != target.resources.size() ||
      c.links.size() != target.links.size())
    return reject(error, "target contract coverage changed");
  for (size_t i = 0; i < c.memories.size(); ++i)
    if (c.memories[i].id != target.memories[i].id ||
        c.memories[i].capacityBytes != target.memories[i].capacityBytes ||
        c.memories[i].accessibleDevices != target.memories[i].accessibleDevices)
      return reject(error, "memory contract changed");
  for (size_t i = 0; i < c.resources.size(); ++i)
    if (c.resources[i].id != target.resources[i].id ||
        c.resources[i].kind != target.resources[i].kind ||
        c.resources[i].device != target.resources[i].device ||
        c.resources[i].capacity != target.resources[i].capacity)
      return reject(error, "resource contract changed");
  for (size_t i = 0; i < c.links.size(); ++i) {
    const auto &x = c.links[i], &y = target.links[i];
    if (x.id != y.id || x.source != y.source || x.target != y.target ||
        x.resource != y.resource || x.latencyMs != y.latencyMs ||
        x.bandwidthBytesPerMs != y.bandwidthBytesPerMs ||
        x.minBytes != y.minBytes || x.maxBytes != y.maxBytes ||
        x.sourceLayout != y.sourceLayout || x.targetLayout != y.targetLayout ||
        x.sourceDtype != y.sourceDtype || x.targetDtype != y.targetDtype ||
        x.sourceDevice != y.sourceDevice || x.targetDevice != y.targetDevice ||
        x.memory != y.memory || x.temporaryBytes != y.temporaryBytes ||
        x.provenance.kind != y.provenance.kind ||
        x.provenance.source != y.provenance.source)
      return reject(error, "link contract changed");
  }
  std::set<std::string> keys;
  for (const auto &al : c.allocations) {
    auto memory =
        std::find_if(target.memories.begin(), target.memories.end(),
                     [&](const auto &m) { return m.id == al.memory; });
    if (al.storage >= a.storages.size() || memory == target.memories.end() ||
        al.bytes <= 0 || !keys.insert(al.key).second ||
        al.beginPhase >= al.endPhase || al.endPhase > a.tasks.size())
      return reject(error, "invalid allocation region/size/location/lifetime");
  }
  std::vector<bool> seenCopies(c.copies.size()),
      seenVersions(c.versions.size()), seenTasks(a.tasks.size());
  std::vector<int64_t> latest(c.regions.size(), -1);
  std::vector<std::vector<size_t>> users(c.copies.size()),
      consumerEdges(c.copies.size());
  std::vector<size_t> creatorCount(c.allocations.size());
  for (size_t i = 0; i < c.versions.size(); ++i)
    if (c.versions[i].region >= c.regions.size() ||
        c.versions[i].producer < -1 ||
        c.versions[i].producer >= int64_t(c.events.size()))
      return reject(error, "version has invalid region or producer");
  for (size_t i = 0; i < c.copies.size(); ++i) {
    const auto &copy = c.copies[i];
    if (copy.version >= c.versions.size() ||
        copy.allocation >= c.allocations.size() || copy.createdBy < -1 ||
        copy.createdBy >= int64_t(c.events.size()) ||
        copy.layout != "row_major" || copy.availableFrom > copy.releasedAfter ||
        copy.releasedAfter > c.events.size() ||
        c.allocations[copy.allocation].storage !=
            c.regions[c.versions[copy.version].region].storage)
      return reject(error, "invalid copy version/allocation/layout/lifetime");
    auto memory = std::find_if(
        target.memories.begin(), target.memories.end(), [&](const auto &m) {
          return m.id == c.allocations[copy.allocation].memory;
        });
    if (c.allocations[copy.allocation].bytes !=
        allocationBytes(a, c.regions[c.versions[copy.version].region], *memory))
      return reject(error, "allocation byte size differs from exact region");
    ++creatorCount[copy.allocation];
    if (copy.createdBy < 0) {
      if (copy.createdBy != -1)
        return reject(error, "invalid external copy creator");
      auto cell = c.versions[copy.version].region;
      auto s = c.regions[cell].storage;
      auto loc = target.assumptions.initialLocations.find(a.storages[s].id);
      if (a.storages[s].role != "input" ||
          c.versions[copy.version].producer != -1 || latest[cell] >= 0 ||
          copy.availableFrom != 0 ||
          loc == target.assumptions.initialLocations.end() ||
          loc->second != c.allocations[copy.allocation].memory)
        return reject(error, "fabricated initial copy");
      latest[cell] = copy.version;
      seenCopies[i] = true;
      seenVersions[copy.version] = true;
    }
  }
  if (std::find(creatorCount.begin(), creatorCount.end(), size_t(0)) !=
          creatorCount.end() ||
      std::find_if(creatorCount.begin(), creatorCount.end(),
                   [](size_t n) { return n != 1; }) != creatorCount.end())
    return reject(
        error,
        "allocation must own exactly one created copy in whole-task adapter");
  size_t nextTask = 0, nextData = 0, nextOrder = 0;
  int64_t traffic = 0;
  double duration = 0;
  bool known = true;
  std::map<size_t, double> estimates;
  if (c.computeEstimates.size() != choice.computeEstimates.size())
    return reject(error, "compute estimate coverage changed");
  for (size_t i = 0; i < choice.computeEstimates.size(); ++i) {
    const auto &est = choice.computeEstimates[i];
    double ms;
    std::string lhs, rhs;
    llvm::raw_string_ostream lo(lhs), ro(rhs);
    lo << serializeMappingComputeEstimate(est);
    ro << serializeMappingComputeEstimate(c.computeEstimates[i]);
    if (lhs != rhs || est.task >= a.tasks.size() || estimates.count(est.task) ||
        est.resource != place[est.task].resource ||
        est.implementation != place[est.task].implementation ||
        est.resourceCount != place[est.task].resourceCount ||
        !convertComputeEstimateToMs(est, choice.clockMHz, ms, error))
      return reject(error, "compute estimate contract changed or invalid");
    auto resource =
        std::find_if(target.resources.begin(), target.resources.end(),
                     [&](const auto &r) { return r.id == est.resource; });
    if (resource == target.resources.end() ||
        est.resourceCount > resource->capacity)
      return reject(error, "compute estimate exceeds resource capacity");
    estimates[est.task] = ms;
  }
  for (size_t i = 0; i < c.events.size(); ++i) {
    const auto &e = c.events[i];
    if (e.task >= a.tasks.size() || e.phase != e.task ||
        (!choice.semanticDependenciesOnly && i &&
         std::find(e.dependencies.begin(), e.dependencies.end(), i - 1) ==
             e.dependencies.end()))
      return reject(error, "serial event schedule lacks previous completion");
    for (size_t dep : e.dependencies)
      if (dep >= i)
        return reject(error, "event dependence is not acyclic");
    for (size_t id : e.reads) {
      if (id >= c.copies.size() || !seenCopies[id])
        return reject(error, "read before copy creation");
      const auto &copy = c.copies[id];
      size_t cell = c.versions[copy.version].region;
      if (copy.version != size_t(latest[cell]) || copy.availableFrom > i ||
          copy.releasedAfter <= i ||
          (copy.createdBy >= 0 &&
           std::find(e.dependencies.begin(), e.dependencies.end(),
                     size_t(copy.createdBy)) == e.dependencies.end()))
        return reject(error, "stale version, use after release, or missing "
                             "copy creation dependency");
      users[id].push_back(i);
    }
    if (e.kind == "transfer") {
      if (e.reads.size() != 1 || e.writes.size() != 1 || !e.dataEdges.empty() ||
          !e.orderEdges.empty())
        return reject(error,
                      "transfer cannot fabricate source data/order edges");
      size_t id = e.writes[0];
      if (id >= c.copies.size())
        return reject(error, "invalid transfer target");
      const auto &src = c.copies[e.reads[0]], &dst = c.copies[id];
      auto cell = c.versions[src.version].region;
      auto s = c.regions[cell].storage;
      auto link = selectedLink(target, e.link);
      if (!e.transferRegion || e.transferRegion->storage != s ||
          e.transferRegion->offsets != c.regions[cell].offsets ||
          e.transferRegion->sizes != c.regions[cell].sizes ||
          e.transferBytes != regionBytes(a, c.regions[cell]))
        return reject(
            error,
            "whole-task transfer region/bytes differ from source demand");
      if (!link ||
          !linkSupports(*link, a.storages[s],
                        regionBytes(a, c.regions[cell])) ||
          link->source != c.allocations[src.allocation].memory ||
          link->target != c.allocations[dst.allocation].memory ||
          link->resource != e.resource || e.memory != link->target ||
          dst.version != src.version || dst.createdBy != int64_t(i) ||
          seenCopies[id])
        return reject(error,
                      "transfer source/destination/version/link mismatch");
      auto res =
          std::find_if(target.resources.begin(), target.resources.end(),
                       [&](const auto &r) { return r.id == e.resource; });
      if (res == target.resources.end() || res->capacity < 1)
        return reject(error, "missing transfer resource");
      seenCopies[id] = true;
      int64_t b = regionBytes(a, c.regions[cell]);
      if (traffic > INT64_MAX - b)
        return reject(error, "traffic overflow");
      traffic += b;
      if (std::isfinite(link->bandwidthBytesPerMs) &&
          std::isfinite(link->latencyMs) && link->bandwidthBytesPerMs > 0 &&
          link->latencyMs >= 0 && !link->provenance.source.empty())
        duration += link->latencyMs + b / link->bandwidthBytesPerMs;
      else
        known = false;
    } else if (e.kind == "compute") {
      if (e.transferRegion || e.transferBytes)
        return reject(error, "compute cannot carry transfer payload");
      if (e.task != nextTask++ || seenTasks[e.task] ||
          e.memory != place[e.task].memory ||
          e.resource != place[e.task].resource || !e.link.empty())
        return reject(error, "compute instance/placement mismatch");
      seenTasks[e.task] = true;
      std::set<size_t> reads, writes;
      std::set<size_t> raw;
      if (e.dataEdges.size() != e.reads.size())
        return reject(error, "missing value consumer edge");
      for (size_t k = 0; k < e.reads.size(); ++k) {
        size_t copyId = e.reads[k];
        const auto &copy = c.copies[copyId];
        const auto &v = c.versions[copy.version];
        size_t s = c.regions[v.region].storage;
        if (!reads.insert(v.region).second ||
            c.allocations[copy.allocation].memory != e.memory ||
            e.dataEdges[k] != nextData || nextData >= c.dataEdges.size())
          return reject(error, "compute reads wrong region/location");
        const auto &edge = c.dataEdges[nextData];
        if (edge.producer != v.producer || edge.consumer != i ||
            edge.region != v.region || edge.version != copy.version ||
            edge.copy != copyId || edge.requirement.memory != e.memory ||
            edge.requirement.layout != copy.layout ||
            edge.requirement.dtype != a.storages[s].dtype)
          return reject(error,
                        "consumer edge requirement/version/copy mismatch");
        consumerEdges[copyId].push_back(nextData++);
        if (v.producer >= 0)
          raw.insert(c.events[v.producer].task);
      }
      for (size_t id : e.writes) {
        if (id >= c.copies.size())
          return reject(error, "invalid compute result copy");
        const auto &copy = c.copies[id];
        const auto &v = c.versions[copy.version];
        if (seenCopies[id] || seenVersions[copy.version] ||
            !writes.insert(v.region).second || v.producer != int64_t(i) ||
            copy.createdBy != int64_t(i) ||
            c.allocations[copy.allocation].memory != e.memory)
          return reject(error,
                        "compute produces invalid version/location copy");
        seenCopies[id] = true;
        seenVersions[copy.version] = true;
        latest[v.region] = copy.version;
      }
      if (reads != effects[e.task].reads || writes != effects[e.task].writes)
        return reject(error, "compute effects differ from actual MLIR");
      size_t localOrder = 0;
      for (size_t dep : a.tasks[e.task].dependencies)
        if (!raw.count(dep)) {
          if (localOrder >= e.orderEdges.size() ||
              e.orderEdges[localOrder++] != nextOrder ||
              nextOrder >= c.orderEdges.size())
            return reject(error, "missing pure order edge");
          const auto &edge = c.orderEdges[nextOrder++];
          if (edge.consumer != i || edge.producer >= i ||
              c.events[edge.producer].kind != "compute" ||
              c.events[edge.producer].task != dep ||
              edge.reason != "source task dependency without value consumption")
            return reject(error, "invalid pure order edge");
        }
      if (localOrder != e.orderEdges.size())
        return reject(error, "fabricated pure order edge");
      if (estimates.count(e.task))
        duration += estimates[e.task];
      else
        known = false;
    } else
      return reject(error, "unsupported event kind");
    for (size_t id : e.writes)
      if (c.copies[id].availableFrom != i)
        return reject(error, "copy availability differs from creation event");
  }
  if (nextTask != a.tasks.size() || nextData != c.dataEdges.size() ||
      nextOrder != c.orderEdges.size() ||
      std::find(seenCopies.begin(), seenCopies.end(), false) !=
          seenCopies.end() ||
      std::find(seenVersions.begin(), seenVersions.end(), false) !=
          seenVersions.end())
    return reject(error, "missing/fabricated task, edge, version or copy");
  // Independently walk actual copy ancestry, not builder route diagnostics.
  auto verifyRoute = [&](size_t copyId, size_t consumer,
                         const std::string &memory) {
    const auto &terminal = c.copies[copyId];
    size_t storage = c.regions[c.versions[terminal.version].region].storage;
    auto *selected = transferChoice(choice, storage, consumer, memory);
    if (!selected)
      return true;
    for (auto id = selected->links.rbegin(); id != selected->links.rend();
         ++id) {
      const auto &copy = c.copies[copyId];
      if (copy.createdBy < 0)
        return reject(error, "selected route missing transfer event");
      const auto &event = c.events[copy.createdBy];
      if (event.kind != "transfer" || event.link != *id ||
          event.task != std::min(consumer, a.tasks.size() - 1) ||
          event.writes.size() != 1 || event.writes[0] != copyId ||
          event.reads.size() != 1)
        return reject(error,
                      "actual copy ancestry differs from selected route");
      copyId = event.reads[0];
    }
    if (c.allocations[c.copies[copyId].allocation].memory !=
            selected->sourceMemory ||
        c.copies[copyId].version != terminal.version)
      return reject(
          error,
          "actual route source differs from selected current-version memory");
    return true;
  };
  for (const auto &event : c.events)
    if (event.kind == "compute")
      for (auto copy : event.reads)
        if (!verifyRoute(copy, event.task, event.memory))
          return false;
  for (size_t cell = 0; cell < c.regions.size(); ++cell) {
    size_t storage = c.regions[cell].storage;
    auto location = outputs.find(storage);
    if (location == outputs.end())
      continue;
    if (!transferChoice(choice, storage, a.tasks.size(), location->second))
      continue;
    bool matched = false;
    for (size_t id = 0; id < c.copies.size(); ++id) {
      const auto &copy = c.copies[id];
      if (c.versions[copy.version].region != cell ||
          copy.version != size_t(latest[cell]) ||
          c.allocations[copy.allocation].memory != location->second ||
          copy.releasedAfter != c.events.size())
        continue;
      if (!verifyRoute(id, a.tasks.size(), location->second))
        return false;
      matched = true;
    }
    if (!matched)
      return reject(error, "selected output route has no final boundary copy");
  }
  std::set<std::pair<size_t, std::string>> retainedImports;
  for (size_t i = 0; i < c.copies.size(); ++i) {
    const auto &copy = c.copies[i];
    const auto &al = c.allocations[copy.allocation];
    size_t cell = c.versions[copy.version].region;
    size_t s = al.storage;
    auto expectedLast = users[i].empty() ? std::vector<size_t>{}
                                         : std::vector<size_t>{users[i].back()};
    size_t begin = copy.createdBy < 0 ? 0 : c.events[copy.createdBy].phase;
    size_t end =
        users[i].empty() ? begin + 1 : c.events[users[i].back()].phase + 1;
    size_t release =
        users[i].empty() ? copy.availableFrom + 1 : users[i].back() + 1;
    auto out = target.assumptions.outputLocations.find(a.storages[s].id);
    bool final = a.storages[s].role == "output" &&
                 out != target.assumptions.outputLocations.end() &&
                 out->second == al.memory &&
                 copy.version == size_t(latest[cell]);
    if (final) {
      end = a.tasks.size();
      release = c.events.size();
    }
    if (copy.createdBy >= 0 && c.events[copy.createdBy].kind == "transfer") {
      if (users[i].empty() && !final)
        return reject(error, "unneeded creation transfer");
      if (choice.policy == ProgramDataPolicy::RetainAcrossUsers &&
          !retainedImports.insert({copy.version, al.memory}).second)
        return reject(error,
                      "retained policy duplicates a valid location copy");
    }
    if (copy.users != users[i] || copy.lastUsers != expectedLast ||
        copy.consumerEdges != consumerEdges[i] ||
        copy.releasedAfter != release || al.beginPhase != begin ||
        al.endPhase != end)
      return reject(error, "copy users/release/allocation lifetime mismatch");
    if (choice.policy == ProgramDataPolicy::PhaseLocal && copy.createdBy >= 0 &&
        c.events[copy.createdBy].kind == "transfer") {
      // A transit copy can be consumed only by transfers, so compute-user
      // counting alone would silently permit retention across later tasks.
      // Bind all uses to the importing task, independently of the builder.
      std::set<size_t> consumers{c.events[copy.createdBy].task};
      for (auto u : users[i])
        consumers.insert(c.events[u].task);
      if (consumers.size() > 1)
        return reject(error, "phase-local imported copy retained across "
                             "task/transfer consumers");
    }
  }
  // Event intervals are the valid serial schedule's actual allocation lifetime;
  // phase bounds separately remain a conservative human-readable summary.
  std::map<std::string, int64_t> peaks;
  for (const auto &memory : target.memories)
    for (size_t event = 0; event < c.events.size(); ++event) {
      int64_t live = 0;
      std::set<size_t> counted;
      for (const auto &copy : c.copies)
        if (copy.availableFrom <= event && event < copy.releasedAfter &&
            counted.insert(copy.allocation).second) {
          const auto &al = c.allocations[copy.allocation];
          if (al.memory != memory.id)
            continue;
          if (memory.capacityBytes < 0 || live > memory.capacityBytes ||
              al.bytes > memory.capacityBytes - live)
            return reject(
                error,
                "memory capacity exceeded by event-interval copy lifetimes");
          live += al.bytes;
        }
      peaks[memory.id] = std::max(peaks[memory.id], live);
    }
  if (peaks != c.peakMemoryBytes)
    return reject(error, "event memory peak report mismatch");
  for (size_t cell = 0; cell < c.regions.size(); ++cell)
    if (a.storages[c.regions[cell].storage].role == "output") {
      size_t s = c.regions[cell].storage;
      auto loc = target.assumptions.outputLocations.find(a.storages[s].id);
      bool found = false;
      if (loc != target.assumptions.outputLocations.end())
        for (const auto &copy : c.copies)
          if (c.versions[copy.version].region == cell &&
              c.allocations[copy.allocation].memory == loc->second &&
              copy.version == size_t(latest[cell]) &&
              copy.releasedAfter == c.events.size())
            found = true;
      if (!found)
        return reject(error,
                      "final source output missing at declared boundary");
    }
  if (traffic != c.transferBytes ||
      (!choice.semanticDependenciesOnly && known && std::isfinite(duration)) !=
          c.estimatedMakespanMs.has_value() ||
      (c.estimatedMakespanMs && *c.estimatedMakespanMs != duration))
    return reject(error,
                  "traffic/full-graph cost differs from checked event charges");
  size_t instance = 0;
  for (size_t event = 0; event < c.events.size(); ++event)
    if (c.events[event].kind == "compute") {
      if (instance >= c.instances.size())
        return reject(error, "missing source task instance");
      const auto &value = c.instances[instance++];
      const auto &source = a.tasks[c.events[event].task];
      if (value.task != c.events[event].task || value.event != event ||
          value.phase != c.events[event].phase ||
          value.domainTask != value.task || value.kind != "source-task" ||
          value.implementation != place[value.task].implementation ||
          value.lanes != 1 || value.parallelLoop != -1 ||
          value.domain.size() != source.loops.size() ||
          value.resourceCount != place[value.task].resourceCount ||
          value.resourceShape != place[value.task].resourceShape)
        return reject(
            error, "whole-task instance source/domain/implementation mismatch");
      for (size_t loop = 0; loop < source.loops.size(); ++loop) {
        const auto &bound = value.domain[loop];
        const auto &expected = source.loops[loop];
        if (bound.loop != loop || bound.lower != expected.lower ||
            bound.upper != expected.upper || bound.step != expected.step)
          return reject(error,
                        "whole-task instance is not full source loop domain");
      }
    }
  if (instance != c.instances.size())
    return reject(error, "fabricated whole-task instance");
  return checkProgramGraphSource(a, c, error) && checkProgramGraph(c, error);
}
