#include "Taskflow/StaticMapping.h"
#include <algorithm>
#include <deque>
#include <functional>
#include <set>
#include <stdexcept>
#include <tuple>
using namespace orbit::taskflow;
namespace {
struct Cell {
  std::string buffer;
  int64_t m0, m1, n0, n1;
  bool operator<(const Cell &o) const {
    return std::tie(buffer, m0, m1, n0, n1) <
           std::tie(o.buffer, o.m0, o.m1, o.n0, o.n1);
  }
  bool operator==(const Cell &o) const {
    return buffer == o.buffer && m0 == o.m0 && m1 == o.m1 && n0 == o.n0 &&
           n1 == o.n1;
  }
};
struct State {
  std::string version, event, device;
};
void clean(std::vector<std::string> &v) {
  v.erase(std::remove(v.begin(), v.end(), ""), v.end());
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}
int64_t bytes(const Buffer &b) {
  if (b.rows <= 0 || b.cols <= 0 || b.rows > INT64_MAX / 4 / b.cols)
    throw std::runtime_error("buffer byte count overflow");
  return b.rows * b.cols * 4;
}
} // namespace
bool orbit::taskflow::buildStaticGraph(
    const StaticAnalysis &a, const StaticConfig &cfg,
    const std::map<std::string, std::string> &assignment, StaticPlan &p,
    std::string &error) {
  try {
    p = StaticPlan{};
    p.assignment = assignment;
    p.steps = a.steps;
    p.programId = staticFingerprint(printPlan(a.program));
    p.candidateId = a.program.candidateId;
    p.configurationId = staticFingerprint(cfg.original);
    p.initialState = "cold_inputs_only_visible_to_all_accessible_devices_at_"
                     "declared_initial_locations";
    p.costBoundary = "compute includes local memory access; explicit "
                     "copies/conversions include completion visibility; direct "
                     "cross-device sharing requires explicit sync";
    std::map<std::string, Buffer> buffers;
    std::map<std::string, const StaticImplementation *> impls;
    std::map<std::string, const StaticMemory *> memories;
    for (auto &b : a.program.buffers)
      buffers[b.id] = b;
    for (auto &i : cfg.implementations)
      impls[i.id] = &i;
    for (auto &m : cfg.memories)
      memories[m.id] = &m;
    std::map<std::string, std::pair<std::set<int64_t>, std::set<int64_t>>>
        bounds;
    for (auto &b : a.program.buffers)
      bounds[b.id] = {{0, b.rows}, {0, b.cols}};
    for (auto &s : a.steps) {
      auto accesses = s.reads;
      accesses.insert(accesses.end(), s.writes.begin(), s.writes.end());
      for (auto &r : accesses) {
        auto it = buffers.find(r.buffer);
        if (it == buffers.end() || r.m.lo < 0 || r.m.hi > it->second.rows ||
            r.n.lo < 0 || r.n.hi > it->second.cols || r.m.lo >= r.m.hi ||
            r.n.lo >= r.n.hi)
          throw std::runtime_error("invalid actual static access region");
        auto &b = bounds[r.buffer];
        b.first.insert(r.m.lo);
        b.first.insert(r.m.hi);
        b.second.insert(r.n.lo);
        b.second.insert(r.n.hi);
      }
    }
    std::map<std::string, std::vector<Cell>> cells;
    for (auto &b : bounds) {
      std::vector<int64_t> m(b.second.first.begin(), b.second.first.end()),
          n(b.second.second.begin(), b.second.second.end());
      if ((m.size() - 1) >
          uint64_t(cfg.assumptions.maxRegionCells) / (n.size() - 1))
        throw std::runtime_error("region cell budget exceeded");
      p.regionCellCount += (m.size() - 1) * (n.size() - 1);
      if (p.regionCellCount > cfg.assumptions.maxRegionCells)
        throw std::runtime_error("region cell budget exceeded");
      for (size_t i = 1; i < m.size(); ++i)
        for (size_t j = 1; j < n.size(); ++j)
          cells[b.first].push_back({b.first, m[i - 1], m[i], n[j - 1], n[j]});
    }
    auto covered = [&](const StaticAccess &r) {
      std::vector<Cell> out;
      for (auto &c : cells.at(r.buffer))
        if (r.m.lo <= c.m0 && c.m1 <= r.m.hi && r.n.lo <= c.n0 &&
            c.n1 <= r.n.hi)
          out.push_back(c);
      return out;
    };
    std::map<std::tuple<std::string, std::string, std::string, std::string>,
             std::string>
        allocKeys;
    std::map<std::string, size_t> allocIndex;
    std::map<Cell, std::string> current;
    std::map<std::pair<Cell, std::string>, State> copies;
    std::map<std::pair<Cell, std::string>, std::set<std::string>> readers;
    uint64_t serial = 0;
    auto allocation = [&](const std::string &b, const std::string &memory,
                          const std::string &layout = "row_major",
                          const std::string &dtype = "f32") {
      if (dtype != "f32" || !memories.count(memory))
        throw std::runtime_error("unsupported allocation dtype/memory");
      auto key = std::make_tuple(b, memory, layout, dtype);
      auto it = allocKeys.find(key);
      if (it != allocKeys.end())
        return it->second;
      StaticAllocation al;
      al.id = "alloc:" + std::to_string(p.allocations.size());
      al.buffer = b;
      al.memory = memory;
      al.layout = layout;
      al.dtype = dtype;
      al.m = {0, buffers.at(b).rows};
      al.n = {0, buffers.at(b).cols};
      al.bytes = bytes(buffers.at(b));
      al.lifetime = "graph";
      allocKeys[key] = al.id;
      allocIndex[al.id] = p.allocations.size();
      p.allocations.push_back(al);
      return al.id;
    };
    auto getAlloc = [&](const std::string &id) -> StaticAllocation & {
      return p.allocations.at(allocIndex.at(id));
    };
    auto entry = [&](const Cell &c, const std::string &aid,
                     const std::string &version) {
      return StaticAccess{c.buffer, aid, version, {c.m0, c.m1}, {c.n0, c.n1}};
    };
    auto emit = [&](StaticEvent ev) {
      if (ev.id.empty())
        ev.id = ev.kind + ":" + std::to_string(serial);
      ++serial;
      clean(ev.dependsOn);
      for (auto &r : ev.reads)
        getAlloc(r.allocation).lastUsers.push_back(ev.id);
      for (auto &w : ev.writes) {
        auto &al = getAlloc(w.allocation);
        al.lastUsers.push_back(ev.id);
        if (al.createdBy.empty() && !al.initial)
          al.createdBy = ev.id;
      }
      p.events.push_back(ev);
      return ev.id;
    };
    for (auto &b : a.program.buffers)
      if (b.role == "input") {
        auto it = cfg.assumptions.initialLocations.find(b.id);
        if (it == cfg.assumptions.initialLocations.end())
          throw std::runtime_error("missing input initial location " + b.id);
        auto aid = allocation(b.id, it->second);
        getAlloc(aid).initial = true;
        for (auto &c : cells[b.id]) {
          current[c] = "initial:" + b.id;
          copies[{c, aid}] = {current[c], "", ""};
        }
      }
    auto route = [&](std::string source, std::string target) {
      std::deque<std::pair<std::string, std::vector<const StaticTransfer *>>> q;
      q.push_back({source, {}});
      std::set<std::string> visited{source};
      while (!q.empty()) {
        auto node = q.front();
        q.pop_front();
        if (node.first == target)
          return node.second;
        for (auto &link : cfg.links)
          if (link.source == node.first && visited.insert(link.target).second) {
            auto next = node.second;
            next.push_back(&link);
            q.push_back({link.target, next});
          }
      }
      throw std::runtime_error("no communication path " + source + " -> " +
                               target);
    };
    auto conversion = [&](const StaticAllocation &s,
                          const StaticAllocation &t) -> const StaticTransfer * {
      for (auto &c : cfg.conversions)
        if (c.memory == s.memory && c.sourceLayout == s.layout &&
            c.targetLayout == t.layout && c.sourceDtype == s.dtype &&
            c.targetDtype == t.dtype)
          return &c;
      return nullptr;
    };
    auto transport = [&](const std::vector<Cell> &keys,
                         const std::string &source, const std::string &dest,
                         const std::string &kind, const StaticTransfer &d) {
      StaticEvent ev;
      ev.kind = kind;
      ev.resourceDemands = {{d.resource, 1}};
      ev.provenance = d.provenance;
      ev.costBoundary =
          "transfer_or_conversion_including_completion_visibility";
      if (kind == "copy")
        ev.link = d.id;
      else
        ev.conversion = d.id;
      for (auto &c : keys) {
        int64_t n = (c.m1 - c.m0) * (c.n1 - c.n0) * 4;
        if (ev.bytes > INT64_MAX - n)
          throw std::runtime_error("copy size overflow");
        ev.bytes += n;
        ev.reads.push_back(entry(c, source, current.at(c)));
        ev.writes.push_back(entry(c, dest, current.at(c)));
        ev.dependsOn.push_back(copies.at({c, source}).event);
        for (auto &r : readers[{c, dest}])
          ev.dependsOn.push_back(r);
      }
      if (ev.bytes < d.minBytes || ev.bytes > d.maxBytes)
        throw std::runtime_error("transfer outside cost-valid byte range");
      ev.durationMs = d.latencyMs + ev.bytes / d.bandwidthBytesPerMs;
      auto id = emit(ev);
      for (auto &c : keys) {
        copies[{c, dest}] = {current.at(c), id, ""};
        readers[{c, source}].insert(id);
      }
      if (d.temporaryBytes) {
        StaticAllocation tmp;
        tmp.id = "temporary:" + id;
        tmp.memory = getAlloc(dest).memory;
        tmp.layout = getAlloc(dest).layout;
        tmp.dtype = getAlloc(dest).dtype;
        tmp.bytes = d.temporaryBytes;
        tmp.createdBy = id;
        tmp.lastUsers = {id};
        tmp.lifetime = "event";
        allocIndex[tmp.id] = p.allocations.size();
        p.allocations.push_back(tmp);
      }
      return id;
    };
    std::map<std::string, std::string> syncCache;
    auto ensure = [&](const std::vector<Cell> &keys,
                      const std::string &targetId, const std::string &device) {
      auto target = getAlloc(targetId);
      std::map<std::string, std::vector<Cell>> groups;
      for (auto &c : keys) {
        if (!current.count(c))
          throw std::runtime_error("read of uninitialized region " + c.buffer);
        auto found = copies.find({c, targetId});
        if (found != copies.end() && found->second.version == current[c])
          continue;
        std::vector<std::string> valid;
        for (auto &copy : copies)
          if (copy.first.first == c && copy.second.version == current[c])
            valid.push_back(copy.first.second);
        std::sort(valid.begin(), valid.end(),
                  [&](const auto &x, const auto &y) {
                    auto ax = getAlloc(x), ay = getAlloc(y);
                    return std::make_tuple(ax.memory != target.memory,
                                           ax.layout != target.layout, x) <
                           std::make_tuple(ay.memory != target.memory,
                                           ay.layout != target.layout, y);
                  });
        std::string chosen;
        for (auto &aid : valid) {
          auto source = getAlloc(aid);
          if (source.layout != target.layout && !conversion(source, target))
            continue;
          try {
            route(source.memory, target.memory);
            chosen = aid;
            break;
          } catch (const std::runtime_error &) {
          }
        }
        if (chosen.empty())
          throw std::runtime_error("no valid communication/layout path for " +
                                   c.buffer);
        groups[chosen].push_back(c);
      }
      for (auto &g : groups) {
        std::string sourceId = g.first;
        auto source = getAlloc(sourceId);
        if (source.layout != target.layout) {
          auto d = conversion(source, target);
          auto converted = allocation(source.buffer, source.memory,
                                      target.layout, target.dtype);
          transport(g.second, sourceId, converted, "convert", *d);
          sourceId = converted;
          source = getAlloc(sourceId);
        }
        for (auto *l : route(source.memory, target.memory)) {
          auto dest =
              allocation(source.buffer, l->target, target.layout, target.dtype);
          transport(g.second, sourceId, dest, "copy", *l);
          sourceId = dest;
        }
      }
      std::vector<std::string> deps;
      std::map<std::string, std::vector<Cell>> syncGroups;
      for (auto &c : keys) {
        auto &s = copies.at({c, targetId});
        deps.push_back(s.event);
        if (!device.empty() && !s.device.empty() && s.device != device)
          syncGroups[s.device].push_back(c);
      }
      for (auto &g : syncGroups) {
        auto it = std::find_if(cfg.synchronizations.begin(),
                               cfg.synchronizations.end(), [&](auto &s) {
                                 return s.sourceDevice == g.first &&
                                        s.targetDevice == device &&
                                        s.memory == target.memory;
                               });
        if (it == cfg.synchronizations.end())
          throw std::runtime_error(
              "missing explicit shared-memory synchronization cost");
        std::string key = targetId + "|" + device;
        for (auto &c : g.second)
          key += "|" + c.buffer + ":" + std::to_string(c.m0) + ":" +
                 std::to_string(c.m1) + ":" + std::to_string(c.n0) + ":" +
                 std::to_string(c.n1) + ":" + current[c] + ":" +
                 copies.at({c, targetId}).event;
        if (!syncCache.count(key)) {
          StaticEvent ev;
          ev.kind = "sync";
          ev.synchronization = it->id;
          ev.sourceDevice = g.first;
          ev.targetDevice = device;
          ev.memory = target.memory;
          ev.resourceDemands = {{it->resource, 1}};
          ev.durationMs = it->latencyMs;
          ev.provenance = it->provenance;
          ev.costBoundary = "visibility_only";
          for (auto &c : g.second) {
            ev.dependsOn.push_back(copies.at({c, targetId}).event);
            ev.reads.push_back(entry(c, targetId, current[c]));
          }
          syncCache[key] = emit(ev);
        }
        deps.push_back(syncCache[key]);
      }
      clean(deps);
      return deps;
    };
    for (auto &s : a.steps) {
      auto chosen = assignment.find(s.id);
      if (chosen == assignment.end() || !impls.count(chosen->second))
        throw std::runtime_error("missing/unknown assigned implementation");
      auto &i = *impls[chosen->second];
      if (!staticImplementationSupports(a, s, i, cfg))
        throw std::runtime_error("unsupported assigned implementation");
      StaticEvent ev;
      ev.kind = "compute";
      ev.id = "compute:" + s.id;
      ev.stepId = s.id;
      ev.taskId = s.taskId;
      ev.operationIndex = s.operationIndex;
      ev.implementationId = i.id;
      ev.device = i.device;
      ev.resourceDemands = i.resources;
      ev.durationMs = staticImplementationCost(s.operation, i);
      ev.provenance = i.cost.provenance;
      ev.costBoundary = "compute_and_local_memory_access_only";
      for (auto &d : s.dependsOn)
        ev.dependsOn.push_back("compute:" + d);
      for (auto &r : s.reads) {
        auto aid = allocation(r.buffer, i.memory, i.layout, i.dtype);
        auto keys = covered(r);
        auto deps = ensure(keys, aid, i.device);
        ev.dependsOn.insert(ev.dependsOn.end(), deps.begin(), deps.end());
        for (auto &c : keys)
          ev.reads.push_back(entry(c, aid, current.at(c)));
      }
      for (auto &w : s.writes) {
        auto aid = allocation(w.buffer, i.memory, i.layout, i.dtype);
        for (auto &c : covered(w)) {
          for (auto &copy : copies)
            if (copy.first.first == c && current.count(c) &&
                copy.second.version == current[c])
              ev.dependsOn.push_back(copy.second.event);
          for (auto &r : readers)
            if (r.first.first == c)
              ev.dependsOn.insert(ev.dependsOn.end(), r.second.begin(),
                                  r.second.end());
          ev.writes.push_back(entry(c, aid, s.id));
        }
      }
      auto id = emit(ev);
      for (auto &r : ev.reads) {
        Cell c{r.buffer, r.m.lo, r.m.hi, r.n.lo, r.n.hi};
        readers[{c, r.allocation}].insert(id);
      }
      for (auto &w : ev.writes) {
        Cell c{w.buffer, w.m.lo, w.m.hi, w.n.lo, w.n.hi};
        current[c] = s.id;
        copies[{c, w.allocation}] = {s.id, id, i.device};
      }
      if (i.temporaryBytes) {
        StaticAllocation tmp;
        tmp.id = "temporary:" + s.id;
        tmp.memory = i.memory;
        tmp.layout = i.layout;
        tmp.dtype = i.dtype;
        tmp.bytes = i.temporaryBytes;
        tmp.createdBy = id;
        tmp.lastUsers = {id};
        tmp.lifetime = "event";
        allocIndex[tmp.id] = p.allocations.size();
        p.allocations.push_back(tmp);
      }
    }
    for (auto &b : a.program.outputs) {
      auto location = cfg.assumptions.outputLocations.find(b);
      if (location == cfg.assumptions.outputLocations.end())
        throw std::runtime_error("missing output location " + b);
      auto memory = memories.at(location->second);
      auto device = cfg.assumptions.outputDevices.find(b);
      std::string dev = device == cfg.assumptions.outputDevices.end()
                            ? (memory->accessibleDevices.size() == 1
                                   ? memory->accessibleDevices.front()
                                   : "")
                            : device->second;
      if (std::find(memory->accessibleDevices.begin(),
                    memory->accessibleDevices.end(),
                    dev) == memory->accessibleDevices.end())
        throw std::runtime_error(
            "explicit accessible output consumer required");
      p.outputConsumers[b] = dev;
      auto aid = allocation(b, memory->id);
      auto deps = ensure(cells.at(b), aid, dev);
      p.outputDependencies.insert(p.outputDependencies.end(), deps.begin(),
                                  deps.end());
      for (auto &c : cells.at(b))
        p.outputs.push_back(entry(c, aid, current.at(c)));
    }
    clean(p.outputDependencies);
    for (auto &al : p.allocations)
      clean(al.lastUsers);
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
}
