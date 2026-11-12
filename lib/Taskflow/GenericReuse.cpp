#include "Taskflow/GenericReuse.h"
#include "Taskflow/JSONUtils.h"
#include "Taskflow/NativeTransforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
using namespace orbit::taskflow;
namespace {
bool same(const std::vector<GenericInterval> &a,
          const std::vector<GenericInterval> &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].lower != b[i].lower || a[i].upper != b[i].upper)
      return false;
  return true;
}
llvm::json::Array indices(const std::vector<size_t> &v) {
  llvm::json::Array r;
  for (auto x : v)
    r.push_back(int64_t(x));
  return r;
}
llvm::json::Array sizes(const std::vector<int64_t> &v) {
  llvm::json::Array r;
  for (auto x : v)
    r.push_back(x);
  return r;
}
std::vector<std::vector<bool>> reachability(const GenericAnalysis &a) {
  size_t n = a.tasks.size();
  std::vector<std::vector<bool>> reach(n, std::vector<bool>(n));
  for (size_t t = 0; t < n; ++t)
    for (auto d : a.tasks[t].dependencies)
      if (d < n)
        reach[d][t] = true;
  for (size_t k = 0; k < n; ++k)
    for (size_t i = 0; i < n; ++i)
      if (reach[i][k])
        for (size_t j = 0; j < n; ++j)
          reach[i][j] = reach[i][j] || reach[k][j];
  return reach;
}
bool identityGrid(const GenericAnalysis &a, size_t task) {
  const auto &t = a.tasks[task];
  if (t.loops.empty() || t.loops.size() > 3)
    return false;
  for (const auto &l : t.loops)
    if (l.step != 1)
      return false;
  for (const auto &x : t.accesses) {
    if (!x.exactRegion || x.region.size() != t.loops.size() ||
        (x.read && x.write))
      return false;
    mlir::ValueRange operands;
    mlir::AffineMap map;
    if (auto op = mlir::dyn_cast<mlir::memref::LoadOp>(x.operation))
      operands = op.getIndices();
    else if (auto op = mlir::dyn_cast<mlir::memref::StoreOp>(x.operation))
      operands = op.getIndices();
    else if (auto op =
                 mlir::dyn_cast<mlir::affine::AffineLoadOp>(x.operation)) {
      operands = op.getMapOperands();
      map = op.getAffineMap();
    } else if (auto op =
                   mlir::dyn_cast<mlir::affine::AffineStoreOp>(x.operation)) {
      operands = op.getMapOperands();
      map = op.getAffineMap();
    } else
      return false;
    if (map && !map.isIdentity())
      return false;
    if (operands.size() != t.loops.size())
      return false;
    for (size_t i = 0; i < t.loops.size(); ++i)
      if (operands[i] != t.loops[i].induction)
        return false;
  }
  return true;
}
} // namespace
std::vector<GenericReuseOpportunity>
orbit::taskflow::discoverGenericReuseOpportunities(const GenericAnalysis &a) {
  std::vector<GenericReuseOpportunity> result;
  for (size_t storage = 0; storage < a.storages.size(); ++storage) {
    std::vector<std::pair<size_t, const GenericAccess *>> writers;
    for (size_t t = 0; t < a.tasks.size(); ++t)
      for (const auto &x : a.tasks[t].accesses)
        if (x.storage == storage && x.write)
          writers.emplace_back(t, &x);
    if (writers.empty() && a.storages[storage].role == "input") {
      for (size_t t = 0; t < a.tasks.size(); ++t)
        for (const auto &x : a.tasks[t].accesses)
          if (x.storage == storage && x.read && x.exactRegion &&
              x.readVersions.size() == 1) {
            auto found =
                std::find_if(result.begin(), result.end(), [&](const auto &o) {
                  return o.storage == storage && o.producer == -1 &&
                         o.version == x.readVersions.front() &&
                         same(o.region, x.region);
                });
            if (found == result.end()) {
              result.push_back(
                  {storage, x.region, x.readVersions.front(), -1, {t}});
            } else if (std::find(found->consumers.begin(),
                                 found->consumers.end(),
                                 t) == found->consumers.end())
              found->consumers.push_back(t);
          }
    } else if (writers.size() == 1 && writers.front().second->exactRegion) {
      const auto &w = writers.front();
      GenericReuseOpportunity o{storage,
                                w.second->region,
                                w.second->writeVersion,
                                int64_t(w.first),
                                {}};
      for (size_t t = 0; t < a.tasks.size(); ++t)
        for (const auto &x : a.tasks[t].accesses)
          if (t != w.first && x.storage == storage && x.read && x.exactRegion &&
              same(x.region, o.region) && x.readVersions.size() == 1 &&
              x.readVersions.front() == o.version) {
            if (std::find(o.consumers.begin(), o.consumers.end(), t) ==
                o.consumers.end())
              o.consumers.push_back(t);
          }
      if (!o.consumers.empty())
        result.push_back(std::move(o));
    }
  }
  result.erase(std::remove_if(result.begin(), result.end(),
                              [](const auto &o) {
                                return o.producer < 0 && o.consumers.size() < 2;
                              }),
               result.end());
  // Semantic ordering, rather than source numbering, controls bounded
  // proposals.
  auto names = [&](const auto &o) {
    std::vector<std::string> r;
    for (auto t : o.consumers)
      r.push_back(a.tasks[t].id);
    std::sort(r.begin(), r.end());
    return r;
  };
  std::sort(result.begin(), result.end(), [&](const auto &x, const auto &y) {
    return std::make_pair(a.storages[x.storage].id, names(x)) <
           std::make_pair(a.storages[y.storage].id, names(y));
  });
  for (auto &o : result)
    std::sort(o.consumers.begin(), o.consumers.end(), [&](size_t x, size_t y) {
      return a.tasks[x].id < a.tasks[y].id;
    });
  return result;
}
bool orbit::taskflow::deriveGenericResidencyOrder(
    const GenericAnalysis &a, const std::vector<ProgramResidencyGroup> &groups,
    std::vector<size_t> &order, std::string &error) {
  size_t n = a.tasks.size();
  std::vector<size_t> owner(n, size_t(-1));
  std::vector<std::vector<size_t>> nodes;
  for (const auto &g : groups) {
    if (g.tasks.size() < 2) {
      error = "residency contraction needs multiple tasks";
      return false;
    }
    size_t node = nodes.size();
    nodes.push_back(g.tasks);
    std::set<size_t> seen;
    for (auto t : g.tasks) {
      if (t >= n || owner[t] != size_t(-1)) {
        error = "invalid or overlapping residency contraction";
        return false;
      }
      owner[t] = node;
      for (auto d : a.tasks[t].dependencies)
        if (std::find(g.tasks.begin(), g.tasks.end(), d) != g.tasks.end() &&
            !seen.count(d)) {
          error = "residency group internal dependency order is invalid";
          return false;
        }
      seen.insert(t);
    }
  }
  for (size_t t = 0; t < n; ++t)
    if (owner[t] == size_t(-1)) {
      owner[t] = nodes.size();
      nodes.push_back({t});
    }
  std::vector<std::set<size_t>> parents(nodes.size());
  for (size_t t = 0; t < n; ++t)
    for (auto d : a.tasks[t].dependencies) {
      if (d >= n) {
        error = "source dependency references unknown task";
        return false;
      }
      if (owner[d] != owner[t])
        parents[owner[t]].insert(owner[d]);
    }
  std::set<size_t> done;
  order.clear();
  while (done.size() < nodes.size()) {
    size_t best = nodes.size(), key = n;
    for (size_t node = 0; node < nodes.size(); ++node)
      if (!done.count(node) &&
          std::all_of(parents[node].begin(), parents[node].end(),
                      [&](size_t p) { return done.count(p); })) {
        size_t first =
            *std::min_element(nodes[node].begin(), nodes[node].end());
        if (first < key) {
          best = node;
          key = first;
        }
      }
    if (best == nodes.size()) {
      error =
          "residency contraction creates dependency cycle (non-convex group)";
      order.clear();
      return false;
    }
    done.insert(best);
    order.insert(order.end(), nodes[best].begin(), nodes[best].end());
  }
  return true;
}
GenericReuseDiscovery orbit::taskflow::discoverGenericReuseProposals(
    const GenericAnalysis &a, const llvm::json::Value &config,
    size_t maxProposals, size_t maxGroups, size_t maxTileProposals) {
  GenericReuseDiscovery out;
  if (!a.function || !maxProposals || !maxGroups || !maxTileProposals)
    return out;
  auto module = a.function->getParentOfType<mlir::ModuleOp>();
  auto seeds = discoverGenericReuseOpportunities(a);
  auto reach = reachability(a);
  ProgramMappingChoice current;
  std::string error;
  readNativeMappingChoice(module, current, error);
  auto existing = current.grouping == "tile-major"
                      ? genericResidencyGroups(current)
                      : std::vector<ProgramResidencyGroup>{};
  std::vector<int64_t> initial;
  if (auto attr = module->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "orbit.air_tile_sizes"))
    initial.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
  if (initial.empty())
    if (auto *c = config.getAsObject())
      if (auto *g = c->getObject("generic_air"))
        if (auto *t = g->getArray("tile_sizes")) {
          initial.clear();
          for (const auto &x : *t)
            initial.push_back(x.getAsInteger().value_or(1));
        }
  if (initial.empty()) {
    initial = {1, 1, 1};
    for (const auto &t : a.tasks)
      for (size_t l = 0; l < t.loops.size() && l < 3; ++l)
        initial[l] = std::max(initial[l], t.loops[l].upper - t.loops[l].lower);
  }
  initial.resize(3, 1);
  std::vector<std::vector<int64_t>> tiles{initial};
  for (size_t axis = 0;
       axis < initial.size() && tiles.size() < maxTileProposals; ++axis)
    if (initial[axis] > 1) {
      auto next = initial;
      next[axis] = (next[axis] + 1) / 2;
      tiles.push_back(next);
    }
  while (tiles.size() < maxTileProposals) {
    auto next = tiles.back();
    bool shrunk = false;
    for (size_t axis = 0; axis < next.size(); ++axis)
      if (next[axis] > 1) {
        next[axis] = (next[axis] + 1) / 2;
        shrunk = true;
        break;
      }
    if (!shrunk || std::find(tiles.begin(), tiles.end(), next) != tiles.end())
      break;
    tiles.push_back(next);
  }
  // maxProposals bounds accepted results; independent attempt budget bounds
  // CPU.
  size_t attempts = 0,
         maxAttempts =
             std::max(size_t(16), maxProposals * maxTileProposals * 4);
  std::set<std::string> seen;
  for (const auto &seed : seeds) {
    std::vector<std::vector<size_t>> covers;
    if (seed.producer >= 0) {
      for (auto c : seed.consumers)
        covers.push_back({c});
    } else if (seed.consumers.size() >= 2)
      covers.push_back({seed.consumers[0], seed.consumers[1]});
    if (seed.consumers.size() > 1)
      covers.push_back(seed.consumers);
    for (const auto &cover : covers) {
      std::set<size_t> members(cover.begin(), cover.end());
      if (seed.producer >= 0)
        members.insert(size_t(seed.producer));
      bool changed = true;
      while (changed) {
        changed = false;
        for (size_t middle = 0; middle < a.tasks.size(); ++middle)
          if (!members.count(middle))
            for (auto from : members)
              for (auto to : members)
                if (reach[from][middle] && reach[middle][to]) {
                  members.insert(middle);
                  changed = true;
                }
      }
      ProgramResidencyGroup group;
      group.tasks.assign(members.begin(), members.end());
      group.retainedStorages = {seed.storage};
      // Every internal produced edge needs a retained local copy, including
      // intermediates introduced by convex closure.
      for (size_t s = 0; s < a.storages.size(); ++s) {
        size_t writers = 0, reads = 0;
        bool inside = false;
        for (size_t t = 0; t < a.tasks.size(); ++t) {
          bool w = false, r = false;
          for (const auto &x : a.tasks[t].accesses)
            if (x.storage == s) {
              w |= x.write;
              r |= x.read;
            }
          writers += w;
          inside |= w && members.count(t);
          reads += r && members.count(t);
        }
        if (((writers == 1 && inside && reads) ||
             (writers == 0 && a.storages[s].role == "input" && reads > 1)) &&
            std::find(group.retainedStorages.begin(),
                      group.retainedStorages.end(),
                      s) == group.retainedStorages.end())
          group.retainedStorages.push_back(s);
      }
      std::sort(group.retainedStorages.begin(), group.retainedStorages.end());
      std::vector<std::vector<ProgramResidencyGroup>> scopes{{group}};
      if (!existing.empty() && existing.size() < maxGroups) {
        bool disjoint = true;
        for (const auto &g : existing)
          for (auto t : g.tasks)
            disjoint &= !members.count(t);
        if (disjoint) {
          auto combined = existing;
          combined.push_back(group);
          std::sort(combined.begin(), combined.end(),
                    [](const auto &x, const auto &y) {
                      return x.tasks.front() < y.tasks.front();
                    });
          scopes.push_back(std::move(combined));
        }
      }
      for (const auto &groups : scopes)
        for (const auto &tile : tiles) {
          if (attempts >= maxAttempts)
            goto selectProposals;
          auto canonicalGroups = groups;
          std::vector<size_t> canonicalOrder;
          std::string orderError;
          if (deriveGenericResidencyOrder(a, canonicalGroups, canonicalOrder,
                                          orderError))
            std::sort(canonicalGroups.begin(), canonicalGroups.end(),
                      [&](const auto &x, const auto &y) {
                        return std::find(canonicalOrder.begin(),
                                         canonicalOrder.end(),
                                         x.tasks.front()) <
                               std::find(canonicalOrder.begin(),
                                         canonicalOrder.end(), y.tasks.front());
                      });
          ProgramMappingChoice mapping;
          mapping.residency = "tiled-phase-local";
          mapping.grouping = "tile-major";
          mapping.dataOrganization = "retain-across-users";
          mapping.residencyGroups = canonicalGroups;
          std::string key;
          for (const auto &g : groups) {
            for (auto t : g.tasks)
              key += std::to_string(t) + ",";
            key += "/";
            for (auto s : g.retainedStorages)
              key += std::to_string(s) + ",";
            key += ";";
          }
          for (auto t : tile)
            key += std::to_string(t) + ":";
          if (!seen.insert(key).second)
            continue;
          ++attempts;
          llvm::json::Array region;
          for (auto r : seed.region)
            region.push_back(
                llvm::json::Object{{"lower", r.lower}, {"upper", r.upper}});
          llvm::json::Object diagnostic{
              {"kind", "data-region-reuse"},
              {"storage", a.storages[seed.storage].id},
              {"storage_index", int64_t(seed.storage)},
              {"version", seed.version},
              {"producer", seed.producer},
              {"consumers", indices(cover)},
              {"all_consumers", indices(seed.consumers)},
              {"region", std::move(region)},
              {"dependency_convex_tasks", indices(group.tasks)},
              {"tile_sizes", sizes(tile)},
              {"propagation", "matching exact identity-grid rectangles; full "
                              "checked planner capacity/capability"},
              {"attempt", int64_t(attempts)}};
          bool supported =
              std::all_of(group.tasks.begin(), group.tasks.end(),
                          [&](size_t t) { return identityGrid(a, t); });
          if (!supported) {
            diagnostic["planner_supported"] = false;
            diagnostic["reason"] = "joint propagation supports exact "
                                   "identity-grid pointwise accesses only";
            out.rejected.push_back(std::move(diagnostic));
            continue;
          }
          mlir::OwningOpRef<mlir::ModuleOp> clone(
              mlir::cast<mlir::ModuleOp>(module->clone()));
          GenericAnalysis analysis;
          GenericMappingPlan plan;
          std::string failure;
          // Config tile sizes must not override this candidate's source
          // decision.
          auto trialConfig = copyJSONWithoutNestedFields(config, "generic_air",
                                                         {"tile_sizes"});
          bool legal =
              setNativeAIRTileSizes(*clone, tile[0], tile[1], tile[2],
                                    failure) &&
              setNativeMappingChoice(*clone, mapping, failure) &&
              analyzeGenericProgram(*clone, analysis, failure) &&
              planGenericMapping(analysis, trialConfig, plan, failure) &&
              checkGenericMapping(analysis, plan, failure);
          if (!legal) {
            diagnostic["planner_supported"] = false;
            diagnostic["reason"] = failure;
            out.rejected.push_back(std::move(diagnostic));
            continue;
          }
          int64_t bytes = 0;
          for (const auto &t : plan.transfers)
            bytes += t.bytes;
          diagnostic["planner_supported"] = true;
          diagnostic["memory"] = plan.memoryBinding.localMemory;
          diagnostic["conservative_peak_bytes"] = plan.residentBytes;
          diagnostic["boundary_transfer_bytes"] = bytes;
          diagnostic["transfer_count"] = int64_t(plan.transfers.size());
          diagnostic["configuration_count"] =
              int64_t(plan.executionGroups.size());
          diagnostic["parallelism_constraint"] =
              "selected group executes tile members serially on target compute "
              "resource";
          llvm::json::Array exports;
          for (const auto &s : plan.residencyScopes)
            exports.push_back(indices(s.boundaryExports));
          diagnostic["boundary_exports"] = std::move(exports);
          out.proposals.push_back(
              {mapping, tile, llvm::json::Value(std::move(diagnostic))});
        }
    }
  }
selectProposals:
  // Preserve scope/consumer coverage before adding another tile for one scope.
  // The expensive attempts are independently bounded above; this round-robin
  // archive admits different memory/transfer/configuration tradeoffs without
  // treating fewer bytes as a whole-program performance ordering.
  std::vector<std::vector<size_t>> buckets;
  std::vector<std::vector<ProgramResidencyGroup>> keys;
  for (size_t i = 0; i < out.proposals.size(); ++i) {
    auto found = std::find(keys.begin(), keys.end(),
                           out.proposals[i].mapping.residencyGroups);
    size_t k = found - keys.begin();
    if (found == keys.end()) {
      keys.push_back(out.proposals[i].mapping.residencyGroups);
      buckets.push_back({});
    }
    buckets[k].push_back(i);
  }
  std::vector<GenericReuseProposal> selected;
  for (size_t depth = 0; selected.size() < maxProposals; ++depth) {
    bool progress = false;
    for (const auto &bucket : buckets)
      if (depth < bucket.size() && selected.size() < maxProposals) {
        selected.push_back(std::move(out.proposals[bucket[depth]]));
        progress = true;
      }
    if (!progress)
      break;
  }
  out.proposals = std::move(selected);
  return out;
}

namespace {
using JV = llvm::json::Value;
using JO = llvm::json::Object;
using JA = llvm::json::Array;
struct ReuseScopeSet {
  std::vector<ProgramResidencyGroup> groups;
  JV provenance = nullptr;
};
struct ReuseTile {
  std::vector<int64_t> sizes;
  std::vector<std::string> origins;
};
struct ReuseCachedOutcome {
  bool legal = false;
  std::string reason;
  JV metrics = nullptr;
};
std::map<std::string, ReuseCachedOutcome> reuseTrialCache;
std::mutex reuseTrialMutex;
JV reuseGroupsJSON(const std::vector<ProgramResidencyGroup> &groups) {
  JA out;
  for (const auto &g : groups)
    out.push_back(JO{{"tasks", indices(g.tasks)},
                     {"retained_storages", indices(g.retainedStorages)}});
  return out;
}
std::string reuseScopeKey(const std::vector<ProgramResidencyGroup> &groups) {
  return staticFingerprint(reuseGroupsJSON(groups));
}
JV reuseMappingJSON(const ProgramMappingChoice &m) {
  return JO{{"residency", m.residency},
            {"grouping", m.grouping},
            {"data_organization", m.dataOrganization},
            {"groups", reuseGroupsJSON(genericResidencyGroups(m))}};
}
ProgramMappingChoice
reuseMapping(const std::vector<ProgramResidencyGroup> &groups) {
  ProgramMappingChoice m;
  m.residency = groups.empty() ? "phase-local" : "tiled-phase-local";
  m.grouping = groups.empty() ? "task-shape-stages" : "tile-major";
  m.dataOrganization = groups.empty() ? "phase-local" : "retain-across-users";
  m.residencyGroups = groups;
  return m;
}
bool reuseCanonicalScopes(const GenericAnalysis &a,
                          std::vector<ProgramResidencyGroup> &groups,
                          std::string &error) {
  std::vector<size_t> order;
  if (!deriveGenericResidencyOrder(a, groups, order, error))
    return false;
  std::sort(groups.begin(), groups.end(), [&](const auto &x, const auto &y) {
    return std::find(order.begin(), order.end(), x.tasks.front()) <
           std::find(order.begin(), order.end(), y.tasks.front());
  });
  return true;
}
bool reuseDisjoint(const std::vector<ProgramResidencyGroup> &groups,
                   const ProgramResidencyGroup &next) {
  for (const auto &g : groups) {
    for (auto t : next.tasks)
      if (std::find(g.tasks.begin(), g.tasks.end(), t) != g.tasks.end())
        return false;
    for (auto s : next.retainedStorages)
      if (std::find(g.retainedStorages.begin(), g.retainedStorages.end(), s) !=
          g.retainedStorages.end())
        return false;
  }
  return true;
}
std::vector<ReuseScopeSet> reuseScopeCatalogue(const GenericAnalysis &a,
                                               const GenericReuseOptions &o,
                                               bool &complete,
                                               GenericReuseStats &stats) {
  auto seeds = discoverGenericReuseOpportunities(a);
  stats.opportunities = seeds.size();
  auto reach = reachability(a);
  std::vector<ProgramResidencyGroup> singles;
  std::map<std::string, JV> provenance;
  for (const auto &seed : seeds) {
    std::vector<std::vector<size_t>> covers;
    if (seed.producer >= 0)
      for (auto c : seed.consumers)
        covers.push_back({c});
    else if (seed.consumers.size() > 1)
      covers.push_back({seed.consumers[0], seed.consumers[1]});
    if (seed.consumers.size() > 1)
      covers.push_back(seed.consumers);
    for (const auto &cover : covers) {
      std::set<size_t> members(cover.begin(), cover.end());
      if (seed.producer >= 0)
        members.insert(seed.producer);
      bool changed = true;
      while (changed) {
        changed = false;
        for (size_t k = 0; k < a.tasks.size(); ++k)
          if (!members.count(k))
            for (auto from : members)
              for (auto to : members)
                if (reach[from][k] && reach[k][to]) {
                  members.insert(k);
                  changed = true;
                }
      }
      ProgramResidencyGroup group;
      group.tasks.assign(members.begin(), members.end());
      group.retainedStorages = {seed.storage};
      for (size_t s = 0; s < a.storages.size(); ++s) {
        size_t writers = 0, readers = 0;
        bool inside = false;
        for (size_t t = 0; t < a.tasks.size(); ++t) {
          bool w = false, r = false;
          for (const auto &x : a.tasks[t].accesses)
            if (x.storage == s) {
              w |= x.write;
              r |= x.read;
            }
          writers += w;
          inside |= w && members.count(t);
          readers += r && members.count(t);
        }
        if (((writers == 1 && inside && readers) ||
             (writers == 0 && a.storages[s].role == "input" && readers > 1)) &&
            std::find(group.retainedStorages.begin(),
                      group.retainedStorages.end(),
                      s) == group.retainedStorages.end())
          group.retainedStorages.push_back(s);
      }
      std::sort(group.retainedStorages.begin(), group.retainedStorages.end());
      std::vector<ProgramResidencyGroup> gs{group};
      std::string error;
      if (!reuseCanonicalScopes(a, gs, error))
        continue;
      auto key = reuseScopeKey(gs);
      if (provenance.count(key))
        continue;
      JA region;
      for (auto r : seed.region)
        region.push_back(JO{{"lower", r.lower}, {"upper", r.upper}});
      provenance.emplace(key,
                         JO{{"storage", a.storages[seed.storage].id},
                            {"storage_index", int64_t(seed.storage)},
                            {"version", seed.version},
                            {"producer", seed.producer},
                            {"consumers", indices(cover)},
                            {"all_consumers", indices(seed.consumers)},
                            {"region", std::move(region)},
                            {"dependency_convex_tasks", indices(group.tasks)}});
      singles.push_back(std::move(group));
    }
  }
  // This catalogue is explicitly finite: source-backed singleton/full-consumer
  // scopes and every disjoint combination up to maxGroups, never all DAG
  // subsets.
  std::sort(singles.begin(), singles.end(), [](const auto &x, const auto &y) {
    return std::tie(x.tasks, x.retainedStorages) <
           std::tie(y.tasks, y.retainedStorages);
  });
  struct Entry {
    std::vector<ProgramResidencyGroup> groups;
    size_t next = 0;
  };
  std::vector<Entry> queue{{{}, 0}};
  std::set<std::string> seen;
  std::vector<ReuseScopeSet> result;
  if (o.catalogue || o.includeBaseline) {
    result.push_back({{}, JO{{"kind", "baseline"}}});
    seen.insert(reuseScopeKey({}));
  }
  for (size_t q = 0; q < queue.size(); ++q) {
    if (queue[q].groups.size() >= o.maxGroups)
      continue;
    auto parent = queue[q];
    for (size_t i = parent.next; i < singles.size(); ++i) {
      if (!reuseDisjoint(parent.groups, singles[i]))
        continue;
      auto groups = parent.groups;
      groups.push_back(singles[i]);
      std::string error;
      if (!reuseCanonicalScopes(a, groups, error))
        continue;
      auto key = reuseScopeKey(groups);
      if (!seen.insert(key).second)
        continue;
      if (result.size() >= o.maxScopeSets) {
        complete = false;
        continue;
      }
      JA seedsJSON;
      for (const auto &g : groups)
        seedsJSON.push_back(provenance.at(reuseScopeKey({g})));
      result.push_back(
          {groups,
           JO{{"kind", "data-region-reuse"},
              {"seeds", std::move(seedsJSON)},
              {"consumer_subset_policy",
               "singletons-and-full; immutable-input first-pair-and-full"}}});
      queue.push_back({std::move(groups), i + 1});
    }
  }
  stats.scopeSets = result.size();
  return result;
}
struct ReuseWorkingSetStage {
  int64_t bytesPerCell = 0;
  std::vector<int64_t> extents{1, 1, 1};
};
struct ReuseWorkingSetProof {
  bool proven = false;
  int64_t bytesPerCell = 0;
  std::vector<int64_t> extents{1, 1, 1};
  std::vector<ReuseWorkingSetStage> stages;
};
ReuseWorkingSetProof
reuseWorkingSet(const GenericAnalysis &a,
                const std::vector<ProgramResidencyGroup> &groups) {
  ReuseWorkingSetProof proof;
  std::set<size_t> covered;
  std::vector<std::vector<size_t>> stages;
  for (const auto &g : groups) {
    stages.push_back(g.tasks);
    covered.insert(g.tasks.begin(), g.tasks.end());
  }
  for (size_t t = 0; t < a.tasks.size(); ++t)
    if (!covered.count(t))
      stages.push_back({t});
  for (size_t storage = 0; storage < a.storages.size(); ++storage) {
    size_t writers = 0;
    for (const auto &t : a.tasks)
      for (const auto &x : t.accesses)
        if (x.storage == storage && x.write)
          ++writers;
    if (writers > 1)
      return proof;
  }
  for (const auto &stage : stages) {
    ReuseWorkingSetStage pool;
    std::set<size_t> storages;
    const auto &reference = a.tasks[stage.front()];
    int64_t coefficient = 0;
    for (auto t : stage) {
      if (!identityGrid(a, t))
        return proof;
      const auto &task = a.tasks[t];
      if (task.loops.size() != reference.loops.size())
        return proof;
      std::set<size_t> reads, writes;
      for (const auto &x : task.accesses) {
        storages.insert(x.storage);
        if (x.read)
          reads.insert(x.storage);
        if (x.write)
          writes.insert(x.storage);
      }
      for (auto s : reads)
        if (writes.count(s))
          return proof;
      for (size_t axis = 0; axis < task.loops.size(); ++axis) {
        const auto &loop = task.loops[axis];
        const auto &other = reference.loops[axis];
        if (loop.lower != other.lower || loop.upper != other.upper ||
            loop.step != other.step)
          return proof;
        proof.extents[axis] =
            std::max(proof.extents[axis], loop.upper - loop.lower);
        pool.extents[axis] = loop.upper - loop.lower;
      }
    }
    for (auto s : storages) {
      auto type =
          mlir::dyn_cast<mlir::MemRefType>(a.storages[s].root.getType());
      if (!type || !type.getLayout().isIdentity())
        return proof;
      auto bits = type.getElementType().getIntOrFloatBitWidth();
      if (!bits || bits % 8)
        return proof;
      coefficient += bits / 8;
    }
    pool.bytesPerCell = coefficient;
    proof.stages.push_back(pool);
    proof.bytesPerCell = std::max(proof.bytesPerCell, coefficient);
  }
  proof.proven = proof.bytesPerCell > 0;
  return proof;
}
void reuseAddTile(std::vector<ReuseTile> &tiles, std::vector<int64_t> tile,
                  const std::string &origin) {
  tile.resize(3, 1);
  auto found = std::find_if(tiles.begin(), tiles.end(),
                            [&](const auto &t) { return t.sizes == tile; });
  if (found == tiles.end())
    tiles.push_back({std::move(tile), {origin}});
  else if (std::find(found->origins.begin(), found->origins.end(), origin) ==
           found->origins.end())
    found->origins.push_back(origin);
}
std::vector<ReuseTile> reuseTileUniverse(
    const GenericAnalysis &a, const std::vector<ReuseScopeSet> &scopes,
    const GenericReuseOptions &o, const GenericTargetCapabilities &caps,
    std::vector<int64_t> anchor) {
  std::vector<int64_t> extents{1, 1, 1};
  for (const auto &t : a.tasks)
    for (size_t d = 0; d < t.loops.size() && d < 3; ++d)
      extents[d] = std::max(extents[d], t.loops[d].upper - t.loops[d].lower);
  anchor.resize(3, 1);
  std::vector<std::vector<ReuseTile>> families(6);
  reuseAddTile(families[0], anchor, "current-neighbor");
  if (o.tileGenerator == "historical-halving") {
    for (size_t axis = 0; axis < 3; ++axis)
      if (anchor[axis] > 1) {
        auto tile = anchor;
        tile[axis] = (tile[axis] + 1) / 2;
        reuseAddTile(families[0], tile, "halving");
      }
    while (families[0].size() < o.maxTileProposals) {
      auto tile = families[0].back().sizes;
      bool shrunk = false;
      for (auto &n : tile)
        if (n > 1) {
          n = (n + 1) / 2;
          shrunk = true;
          break;
        }
      auto count = families[0].size();
      if (shrunk)
        reuseAddTile(families[0], tile, "halving");
      if (!shrunk || count == families[0].size())
        break;
    }
  } else {
    // The supported pointwise pool needs at least one buffer per distinct
    // storage in each hardware stage. Shared inputs count once; all writes,
    // boundary exports and intermediates have distinct backing pools. This is a
    // necessary lower bound; extra templates/alignment may increase it. Full
    // planner confirmation remains mandatory; no monotonic-cost assumption is
    // made.
    for (const auto &scope : scopes) {
      auto proof = reuseWorkingSet(a, scope.groups);
      if (!proof.proven)
        continue;
      for (size_t axis = 0; axis < 3; ++axis)
        if (extents[axis] > 1) {
          int64_t bound = extents[axis];
          bool valid = true;
          for (const auto &stage : proof.stages) {
            int64_t denominator = stage.bytesPerCell;
            for (size_t other = 0; other < 3; ++other)
              if (other != axis) {
                auto n = std::min(anchor[other], stage.extents[other]);
                if (n <= 0 || denominator > INT64_MAX / n) {
                  valid = false;
                  break;
                }
                denominator *= n;
              }
            if (!valid || denominator <= 0) {
              valid = false;
              break;
            }
            if (stage.extents[axis] > caps.capacityBytes / denominator)
              bound = std::min(bound, caps.capacityBytes / denominator);
          }
          if (!valid)
            continue;
          if (bound < 1)
            continue;
          auto tile = anchor;
          tile[axis] = bound;
          reuseAddTile(families[1], tile, "capacity-boundary");
          auto alignment = caps.tileAlignment[axis];
          if (bound >= alignment) {
            tile[axis] = bound / alignment * alignment;
            reuseAddTile(families[3], tile, "target-alignment");
          }
          // Nearby exact bounds protect nonmonotone shape/configuration
          // effects.
          if (bound > 1) {
            tile[axis] = bound - 1;
            reuseAddTile(families[1], tile, "capacity-boundary");
          }
        }
      // Pair a source-domain axis with a capacity-derived other axis. This can
      // eliminate one tail/configuration family and is not obtainable merely
      // by shrinking the current rectangle along one axis.
      for (size_t fixed = 0; fixed < 3; ++fixed)
        if (extents[fixed] > 1)
          for (size_t axis = 0; axis < 3; ++axis)
            if (axis != fixed && extents[axis] > 1) {
              auto tile = anchor;
              tile[fixed] = extents[fixed];
              int64_t bound = extents[axis];
              bool valid = true;
              for (const auto &stage : proof.stages) {
                int64_t denominator = stage.bytesPerCell;
                for (size_t other = 0; other < 3; ++other)
                  if (other != axis) {
                    auto n = std::min(tile[other], stage.extents[other]);
                    if (n <= 0 || denominator > INT64_MAX / n) {
                      valid = false;
                      break;
                    }
                    denominator *= n;
                  }
                if (!valid || denominator <= 0) {
                  valid = false;
                  break;
                }
                if (stage.extents[axis] > caps.capacityBytes / denominator)
                  bound = std::min(bound, caps.capacityBytes / denominator);
              }
              if (!valid || bound <= 0)
                continue;
              tile[axis] = bound;
              reuseAddTile(families[1], tile, "capacity-boundary");
              auto aligned =
                  bound / caps.tileAlignment[axis] * caps.tileAlignment[axis];
              if (aligned > 0) {
                tile[axis] = aligned;
                reuseAddTile(families[3], tile, "target-alignment");
              }
              int64_t divisor = 1;
              for (int64_t d = 1; d * d <= extents[axis]; ++d)
                if (extents[axis] % d == 0) {
                  if (d <= bound)
                    divisor = std::max(divisor, d);
                  if (extents[axis] / d <= bound)
                    divisor = std::max(divisor, extents[axis] / d);
                }
              tile[axis] = divisor;
              reuseAddTile(families[2], tile, "divisor-tail");
            }
    }
    for (size_t axis = 0; axis < 3; ++axis)
      if (extents[axis] > 1) {
        auto tile = anchor;
        tile[axis] = extents[axis];
        reuseAddTile(families[2], tile, "divisor-tail");
        for (int64_t count = 2; count <= 8; ++count) {
          for (auto size :
               {extents[axis] / count, (extents[axis] + count - 1) / count})
            if (size > 0) {
              tile[axis] = size;
              reuseAddTile(families[2], tile, "divisor-tail");
            }
        }
        for (int64_t divisor = 1; divisor * divisor <= extents[axis]; ++divisor)
          if (extents[axis] % divisor == 0)
            for (auto size : {extents[axis] / divisor, divisor}) {
              tile[axis] = size;
              reuseAddTile(families[2], tile, "divisor-tail");
            }
        int64_t preferred = caps.tileAlignment[axis];
        for (auto delta : {-preferred, int64_t(-1), int64_t(1), preferred}) {
          auto n = anchor[axis] + delta;
          if (n > 0 && n <= extents[axis]) {
            tile = anchor;
            tile[axis] = n;
            reuseAddTile(families[4], tile, "current-neighbor");
          }
        }
        if (preferred > 0) {
          tile = anchor;
          tile[axis] =
              std::max(preferred, anchor[axis] / preferred * preferred);
          if (tile[axis] <= extents[axis])
            reuseAddTile(families[3], tile, "target-alignment");
        }
        if (anchor[axis] > 1) {
          tile = anchor;
          tile[axis] = (tile[axis] + 1) / 2;
          reuseAddTile(families[5], tile, "halving");
        }
      }
  }
  std::vector<ReuseTile> selected;
  // First reserve direct capacity boundaries for every distinct working-set
  // coefficient; then round-robin families rather than filling with halvings.
  for (const auto &tile : families[0])
    for (const auto &origin : tile.origins)
      reuseAddTile(selected, tile.sizes, origin);
  if (o.tileGenerator != "historical-halving") {
    for (const auto &tile : families[1])
      if (selected.size() <
          std::min(o.maxTileProposals,
                   std::max(size_t(2), o.maxTileProposals / 2)))
        for (const auto &origin : tile.origins)
          reuseAddTile(selected, tile.sizes, origin);
    for (size_t depth = 0; selected.size() < o.maxTileProposals; ++depth) {
      bool progress = false;
      for (size_t family = 2; family < families.size(); ++family)
        if (depth < families[family].size() &&
            selected.size() < o.maxTileProposals) {
          auto count = selected.size();
          for (const auto &origin : families[family][depth].origins)
            reuseAddTile(selected, families[family][depth].sizes, origin);
          progress |= count != selected.size();
        }
      if (!progress) {
        bool remains = false;
        for (size_t family = 2; family < families.size(); ++family)
          remains |= depth + 1 < families[family].size();
        if (!remains)
          break;
      }
    }
  }
  if (selected.size() > o.maxTileProposals)
    selected.resize(o.maxTileProposals);
  // Union provenance even when the same size was discovered by several rules.
  for (auto &tile : selected)
    for (const auto &family : families)
      for (const auto &other : family)
        if (tile.sizes == other.sizes)
          for (const auto &origin : other.origins)
            if (std::find(tile.origins.begin(), tile.origins.end(), origin) ==
                tile.origins.end())
              tile.origins.push_back(origin);
  return selected;
}
JV reuseTileJSON(const ReuseTile &tile) {
  JA origins;
  for (const auto &o : tile.origins)
    origins.push_back(o);
  return JO{{"tile_sizes", sizes(tile.sizes)}, {"origins", std::move(origins)}};
}
int64_t reusePeakBound(const ReuseWorkingSetProof &proof,
                       const std::vector<int64_t> &tile) {
  if (!proof.proven)
    return -1;
  int64_t peak = 0;
  for (const auto &stage : proof.stages) {
    int64_t bytes = stage.bytesPerCell;
    for (size_t d = 0; d < 3; ++d) {
      int64_t n = std::min(tile[d], stage.extents[d]);
      if (n <= 0 || bytes > INT64_MAX / n)
        return -1;
      bytes *= n;
    }
    peak = std::max(peak, bytes);
  }
  return peak;
}
} // namespace

void orbit::taskflow::clearGenericReuseCache() {
  std::lock_guard<std::mutex> guard(reuseTrialMutex);
  reuseTrialCache.clear();
}
size_t orbit::taskflow::genericReuseCacheSize() {
  std::lock_guard<std::mutex> guard(reuseTrialMutex);
  return reuseTrialCache.size();
}
llvm::json::Value
orbit::taskflow::serializeGenericReuseStats(const GenericReuseStats &s) {
  return JO{{"opportunities", int64_t(s.opportunities)},
            {"scope_sets", int64_t(s.scopeSets)},
            {"tile_trials", int64_t(s.tileTrials)},
            {"generation_attempts", int64_t(s.generationAttempts)},
            {"source_analyses", int64_t(s.sourceAnalyses)},
            {"planner_calls", int64_t(s.plannerCalls)},
            {"checker_calls", int64_t(s.checkerCalls)},
            {"cache_hits", int64_t(s.cacheHits)},
            {"capacity_screen_rejects", int64_t(s.capacityScreenRejects)},
            {"domain_exclusions", int64_t(s.domainExclusions)},
            {"legal", int64_t(s.legal)},
            {"rejected", int64_t(s.rejected)},
            {"elapsed_ms", s.elapsedMs}};
}
GenericReuseDiscovery orbit::taskflow::discoverGenericReuseConfigurations(
    const GenericAnalysis &a, const JV &config, const GenericReuseOptions &o) {
  GenericReuseDiscovery out;
  auto start = std::chrono::steady_clock::now();
  auto before = genericMappingStatistics();
  auto finish = [&]() {
    auto after = genericMappingStatistics();
    out.stats.plannerCalls = after.plannerCalls - before.plannerCalls;
    out.stats.checkerCalls = after.checkerCalls - before.checkerCalls;
    out.stats.elapsedMs = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count();
  };
  if (!a.function || !o.maxScopeSets || !o.maxTileProposals || !o.maxAttempts ||
      !o.maxGroups || !o.maxCatalogueConfigurations || !o.maxComputeInstances ||
      o.maxComputeInstances > 1000000) {
    out.stopReason = "zero discovery bound or unavailable source";
    finish();
    return out;
  }
  if (o.tileGenerator != "constraint" &&
      o.tileGenerator != "historical-halving") {
    out.stopReason = "unknown tile generator";
    finish();
    return out;
  }
  if (!o.requiredAlignment.empty() &&
      (o.requiredAlignment.size() != 3 ||
       std::any_of(o.requiredAlignment.begin(), o.requiredAlignment.end(),
                   [](auto n) { return n <= 0; }))) {
    out.stopReason = "required alignment must contain three positive integers";
    finish();
    return out;
  }
  GenericTargetCapabilities caps;
  std::string targetError;
  if (!getGenericTargetCapabilities(config, caps, targetError)) {
    out.stopReason = targetError;
    finish();
    return out;
  }
  bool scopeComplete = true;
  auto scopes = reuseScopeCatalogue(a, o, scopeComplete, out.stats);
  auto module = a.function->getParentOfType<mlir::ModuleOp>();
  std::vector<int64_t> anchor = o.currentTile;
  if (anchor.empty() && !o.catalogue)
    if (auto attr = module->getAttrOfType<mlir::DenseI64ArrayAttr>(
            "orbit.air_tile_sizes"))
      anchor.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
  if (anchor.empty()) {
    anchor = {1, 1, 1};
    for (const auto &t : a.tasks)
      for (size_t d = 0; d < t.loops.size() && d < 3; ++d)
        anchor[d] = std::max(anchor[d], t.loops[d].upper - t.loops[d].lower);
  }
  anchor.resize(3, 1);
  auto tiles = reuseTileUniverse(a, scopes, o, caps, anchor);
  out.stats.tileTrials = tiles.size();
  // Source semantics include the actual complete function text, access/version
  // facts and non-choice module attributes. Search labels/preferences are not
  // planner inputs; full target/model/version fields remain in the cache key.
  mlir::OwningOpRef<mlir::ModuleOp> semantic(
      mlir::cast<mlir::ModuleOp>(module->clone()));
  semantic->getOperation()->removeAttr("orbit.mapping");
  semantic->getOperation()->removeAttr("orbit.air_tile_sizes");
  std::string semanticText;
  llvm::raw_string_ostream semanticStream(semanticText);
  semantic->getOperation()->print(semanticStream);
  semanticStream.flush();
  JO keyedObject;
  if (auto *c = config.getAsObject())
    for (const auto &entry : *c) {
      if (entry.first == "native_auto_candidates" ||
          entry.first == "search_preferences" ||
          entry.first == "search_diversity_slots")
        continue;
      keyedObject[entry.first.str()] = JV(entry.second);
    }
  JV keyedConfig(std::move(keyedObject));
  JV semanticKey = JO{{"schema", "orbit.generic-reuse.cache.v21"},
                      {"semantic_source", semanticText},
                      {"source_analysis", serializeGenericAnalysis(a)},
                      {"target_model_config", JV(keyedConfig)},
                      {"optimizer_domain_max_compute_instances",
                       int64_t(o.maxComputeInstances)}};
  JA scopeJSON, tileJSON;
  for (const auto &s : scopes)
    scopeJSON.push_back(reuseGroupsJSON(s.groups));
  for (const auto &t : tiles)
    tileJSON.push_back(reuseTileJSON(t));
  out.catalogueId = staticFingerprint(
      JO{{"source_target_model", JV(semanticKey)},
         {"scopes", std::move(scopeJSON)},
         {"tiles", std::move(tileJSON)},
         {"generator", o.tileGenerator},
         {"required_alignment", sizes(o.requiredAlignment)},
         {"require_divisible", o.requireDivisible},
         {"max_compute_instances", int64_t(o.maxComputeInstances)}});
  out.catalogueComplete = scopeComplete;
  bool stop = false;
  size_t tested = 0;
  ProgramMappingChoice selected;
  std::string ignored;
  readNativeMappingChoice(module, selected, ignored);
  auto currentGroups = selected.grouping == "tile-major"
                           ? genericResidencyGroups(selected)
                           : std::vector<ProgramResidencyGroup>{};
  for (const auto &scope : scopes) {
    if (stop)
      break;
    if (!o.catalogue) {
      size_t removed = 0, added = 0;
      for (const auto &g : currentGroups)
        removed += std::find(scope.groups.begin(), scope.groups.end(), g) ==
                   scope.groups.end();
      for (const auto &g : scope.groups)
        added += std::find(currentGroups.begin(), currentGroups.end(), g) ==
                 currentGroups.end();
      if (std::max(removed, added) > 1)
        continue;
    }
    auto proof = reuseWorkingSet(a, scope.groups);
    auto mapping = reuseMapping(scope.groups);
    for (const auto &tile : tiles) {
      double elapsed = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start)
                           .count();
      auto counters = genericMappingStatistics();
      if (tested >= o.maxCatalogueConfigurations ||
          out.stats.generationAttempts >= o.maxAttempts ||
          (o.maxWallTimeMs > 0 && elapsed >= o.maxWallTimeMs)) {
        out.stopReason = tested >= o.maxCatalogueConfigurations
                             ? "max_catalogue_configurations"
                             : (out.stats.generationAttempts >= o.maxAttempts
                                    ? "max_generation_attempts"
                                    : "max_wall_time_ms");
        out.catalogueComplete = false;
        stop = true;
        break;
      }
      ++tested;
      ++out.stats.generationAttempts;
      std::string configurationId =
          staticFingerprint(JO{{"source_target_model", JV(semanticKey)},
                               {"mapping", reuseMappingJSON(mapping)},
                               {"tile_sizes", sizes(tile.sizes)}});
      JO diagnostic{
          {"kind", "constraint-region-reuse"},
          {"configuration_id", configurationId},
          {"catalogue_id", out.catalogueId},
          {"scope_key", reuseScopeKey(scope.groups)},
          {"source_opportunities", JV(scope.provenance)},
          {"groups", reuseGroupsJSON(scope.groups)},
          {"tile_provenance", reuseTileJSON(tile)},
          {"tile_sizes", sizes(tile.sizes)},
          {"working_set_proven", proof.proven},
          {"bytes_per_cell", proof.bytesPerCell},
          {"capacity_bytes", caps.capacityBytes},
          {"working_set_lower_bound_bytes", reusePeakBound(proof, tile.sizes)},
          {"constraint_semantics",
           "required_alignment/require_divisible are explicit optimizer-domain "
           "restrictions; target alignment is a preference"}};
      std::string invalid;
      std::vector<int64_t> extents{1, 1, 1};
      for (const auto &t : a.tasks)
        for (size_t d = 0; d < t.loops.size() && d < 3; ++d)
          extents[d] =
              std::max(extents[d], t.loops[d].upper - t.loops[d].lower);
      for (size_t d = 0; d < 3; ++d) {
        if (tile.sizes[d] <= 0 || tile.sizes[d] > extents[d])
          invalid = "tile outside positive source iteration extent";
        if (!o.requiredAlignment.empty() &&
            tile.sizes[d] % o.requiredAlignment[d])
          invalid = "tile violates explicit required alignment";
        if (o.requireDivisible && extents[d] % tile.sizes[d])
          invalid = "tile has tail forbidden by explicit require_divisible";
      }
      int64_t requestedInstances = 0;
      bool instanceCountKnown = true;
      for (const auto &task : a.tasks) {
        int64_t instances = 1;
        for (size_t axis = 0; axis < task.loops.size(); ++axis) {
          const auto &loop = task.loops[axis];
          if (axis >= 3 || loop.step <= 0 || loop.upper <= loop.lower ||
              tile.sizes[axis] <= 0) {
            instanceCountKnown = false;
            break;
          }
          int64_t iterations = (loop.upper - loop.lower - 1) / loop.step + 1;
          int64_t chunks = (iterations - 1) / tile.sizes[axis] + 1;
          if (instances > INT64_MAX / chunks) {
            instanceCountKnown = false;
            break;
          }
          instances *= chunks;
        }
        if (!instanceCountKnown || requestedInstances > INT64_MAX - instances) {
          instanceCountKnown = false;
          break;
        }
        requestedInstances += instances;
      }
      diagnostic["requested_compute_instances"] =
          instanceCountKnown ? JV(requestedInstances) : JV(nullptr);
      diagnostic["max_compute_instances"] = int64_t(o.maxComputeInstances);
      if (instanceCountKnown &&
          requestedInstances > int64_t(o.maxComputeInstances))
        invalid = "exact source task-tile count exceeds explicit "
                  "optimizer-domain max_compute_instances";
      if (!invalid.empty()) {
        diagnostic["domain_exclusion"] = true;
        diagnostic["planner_supported"] = false;
        diagnostic["reason"] = invalid;
        out.rejected.push_back(std::move(diagnostic));
        ++out.stats.rejected;
        ++out.stats.domainExclusions;
        continue;
      }
      diagnostic["domain_exclusion"] = false;
      ReuseCachedOutcome outcome;
      bool hit = false;
      {
        std::lock_guard<std::mutex> guard(reuseTrialMutex);
        auto found = reuseTrialCache.find(configurationId);
        if (found != reuseTrialCache.end()) {
          outcome = found->second;
          hit = true;
        }
      }
      if (hit) {
        ++out.stats.cacheHits;
        diagnostic["cache_hit"] = true;
      } else {
        diagnostic["cache_hit"] = false;
        auto peak = reusePeakBound(proof, tile.sizes);
        bool supported =
            std::all_of(a.tasks.begin(), a.tasks.end(), [&](const auto &t) {
              return identityGrid(a, &t - a.tasks.data());
            });
        if (!supported)
          outcome.reason = "constraint propagation supports exact "
                           "identity-grid pointwise accesses only";
        else if (proof.proven && peak > caps.capacityBytes) {
          outcome.reason = "proven distinct-buffer working-set lower bound "
                           "exceeds target capacity";
          ++out.stats.capacityScreenRejects;
        } else {
          if (counters.plannerCalls - before.plannerCalls >=
              o.maxPlannerCalls) {
            out.stopReason = "max_generator_planner_calls";
            out.catalogueComplete = false;
            stop = true;
            break;
          }
          mlir::OwningOpRef<mlir::ModuleOp> clone(
              mlir::cast<mlir::ModuleOp>(module->clone()));
          GenericAnalysis facts;
          GenericMappingPlan plan;
          JO trialObject;
          if (auto *c = config.getAsObject())
            for (const auto &entry : *c) {
              if (entry.first == "generic_air") {
                JO generic;
                if (auto *g = entry.second.getAsObject())
                  for (const auto &field : *g) {
                    if (field.first == "tile_sizes" ||
                        field.first == "cost_model" ||
                        field.first == "theoretical")
                      continue;
                    generic[field.first.str()] = JV(field.second);
                  }
                trialObject["generic_air"] = std::move(generic);
              } else
                trialObject[entry.first.str()] = JV(entry.second);
            }
          JV trialConfig(std::move(trialObject));
          bool valid =
              setNativeAIRTileSizes(*clone, tile.sizes[0], tile.sizes[1],
                                    tile.sizes[2], outcome.reason) &&
              setNativeMappingChoice(*clone, mapping, outcome.reason);
          if (valid) {
            ++out.stats.sourceAnalyses;
            valid = analyzeGenericProgram(*clone, facts, outcome.reason);
          }
          if (valid)
            valid =
                planGenericMapping(facts, trialConfig, plan, outcome.reason);
          if (valid && instanceCountKnown &&
              int64_t(plan.phases.size()) != requestedInstances) {
            valid = false;
            outcome.reason = "checked phase count disagrees with exact "
                             "identity-grid source task-tile count";
          }
          outcome.legal = valid;
          if (valid) {
            int64_t bytes = 0;
            for (const auto &t : plan.transfers)
              bytes += t.bytes;
            JA exports;
            for (const auto &s : plan.residencyScopes)
              exports.push_back(indices(s.boundaryExports));
            int64_t coverage = 0;
            for (const auto &s : scope.groups)
              coverage += s.tasks.size();
            outcome.metrics = JO{
                {"memory", plan.memoryBinding.localMemory},
                {"conservative_peak_bytes", plan.residentBytes},
                {"boundary_transfer_bytes", bytes},
                {"transfer_count", int64_t(plan.transfers.size())},
                {"configuration_count", int64_t(plan.executionGroups.size())},
                {"compute_instances", int64_t(plan.phases.size())},
                {"consumer_coverage", coverage},
                {"boundary_exports", std::move(exports)},
                {"full_planner_checked", true},
                {"score_evaluated", false}};
          }
        }
        {
          std::lock_guard<std::mutex> guard(reuseTrialMutex);
          if (reuseTrialCache.size() >= 8192)
            reuseTrialCache.erase(reuseTrialCache.begin());
          reuseTrialCache[configurationId] = outcome;
        }
      }
      diagnostic["planner_supported"] = outcome.legal;
      diagnostic["metrics"] = JV(outcome.metrics);
      if (!outcome.legal) {
        diagnostic["reason"] = outcome.reason;
        out.rejected.push_back(std::move(diagnostic));
        ++out.stats.rejected;
        continue;
      }
      ++out.stats.legal;
      out.proposals.push_back({mapping, tile.sizes, JV(std::move(diagnostic))});
    }
  }
  if (!scopeComplete && out.stopReason.empty())
    out.stopReason = "max_scope_sets";
  // Dominance is meaningful only for identical groups AND boundary exports.
  // Smaller local storage alone cannot dominate a different grouping or price.
  for (size_t i = 0; i < out.proposals.size(); ++i) {
    auto *di = out.proposals[i].diagnostics.getAsObject();
    auto *mi = di->getObject("metrics");
    bool dominated = false;
    for (size_t j = 0; j < out.proposals.size() && !dominated; ++j)
      if (i != j && out.proposals[i].mapping.residencyGroups ==
                        out.proposals[j].mapping.residencyGroups) {
        auto *dj = out.proposals[j].diagnostics.getAsObject();
        auto *mj = dj->getObject("metrics");
        if (*mi->get("boundary_exports") != *mj->get("boundary_exports"))
          continue;
        bool weak = true, strict = false;
        for (auto key :
             {"conservative_peak_bytes", "boundary_transfer_bytes",
              "transfer_count", "configuration_count", "compute_instances"}) {
          auto x = *mi->getInteger(key), y = *mj->getInteger(key);
          weak &= y <= x;
          strict |= y < x;
        }
        dominated = weak && strict;
      }
    (*di)["pareto_dominated_same_scope_boundary"] = dominated;
  }
  if (!o.catalogue && out.proposals.size() > o.maxProposals) {
    std::vector<GenericReuseProposal> kept;
    std::vector<std::string> scopeOrder;
    std::map<std::string, std::vector<size_t>> buckets;
    for (size_t i = 0; i < out.proposals.size(); ++i) {
      auto *diagnostic = out.proposals[i].diagnostics.getAsObject();
      std::string scope = diagnostic->getString("scope_key")->str();
      if (!buckets.count(scope))
        scopeOrder.push_back(scope);
      buckets[scope].push_back(i);
    }
    for (auto &bucket : buckets)
      std::stable_sort(
          bucket.second.begin(), bucket.second.end(), [&](size_t x, size_t y) {
            return out.proposals[x]
                       .diagnostics.getAsObject()
                       ->getBoolean("pareto_dominated_same_scope_boundary")
                       .value_or(false) <
                   out.proposals[y]
                       .diagnostics.getAsObject()
                       ->getBoolean("pareto_dominated_same_scope_boundary")
                       .value_or(false);
          });
    for (size_t depth = 0; kept.size() < o.maxProposals; ++depth) {
      bool progress = false;
      for (const auto &scope : scopeOrder)
        if (depth < buckets[scope].size() && kept.size() < o.maxProposals) {
          kept.push_back(std::move(out.proposals[buckets[scope][depth]]));
          progress = true;
        }
      if (!progress)
        break;
    }
    out.proposals = std::move(kept);
  }
  // An interrupted prefix is a different admitted catalogue, even when its
  // requested universe matches another run. Policy comparisons must bind the
  // admitted ID, never silently assume identical pools after different limits.
  JA admitted, rejectedIds;
  for (const auto &p : out.proposals)
    admitted.push_back(*p.diagnostics.getAsObject()->get("configuration_id"));
  for (const auto &r : out.rejected)
    if (auto *d = r.getAsObject())
      if (auto *id = d->get("configuration_id"))
        rejectedIds.push_back(*id);
  auto universeId = out.catalogueId;
  out.catalogueId = staticFingerprint(JO{{"universe_id", universeId},
                                         {"complete", out.catalogueComplete},
                                         {"admitted", std::move(admitted)},
                                         {"rejected", std::move(rejectedIds)}});
  for (auto &p : out.proposals) {
    auto *d = p.diagnostics.getAsObject();
    (*d)["universe_id"] = universeId;
    (*d)["catalogue_id"] = out.catalogueId;
  }
  for (auto &r : out.rejected)
    if (auto *d = r.getAsObject()) {
      (*d)["universe_id"] = universeId;
      (*d)["catalogue_id"] = out.catalogueId;
    }
  finish();
  return out;
}
