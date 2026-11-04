#include "Taskflow/StaticMapping.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <stdexcept>
#include <tuple>
using namespace orbit::taskflow;
namespace {
using Cell = std::tuple<std::string, int64_t, int64_t, int64_t, int64_t>;
using Tokens = std::map<Cell, std::string>;
bool near(double a, double b) {
  return std::isfinite(a) && std::isfinite(b) &&
         std::abs(a - b) <= 1e-8 * std::max({1., std::abs(a), std::abs(b)});
}
bool validNumber(double x) { return std::isfinite(x) && x >= 0; }
bool contains(const std::vector<std::string> &v, const std::string &s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}
bool provenance(const StaticProvenance &a, const StaticProvenance &b) {
  return a.kind == b.kind && a.source == b.source;
}
int64_t bytes(const Cell &c) {
  return (std::get<2>(c) - std::get<1>(c)) * (std::get<4>(c) - std::get<3>(c)) *
         4;
}
int64_t checkedAdd(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || b > INT64_MAX - a)
    throw std::runtime_error("byte/resource accounting overflow");
  return a + b;
}
struct Interval {
  double start, end;
  int64_t amount;
};
int64_t peak(const std::vector<Interval> &intervals) {
  std::set<double> times;
  for (auto &i : intervals) {
    times.insert(i.start);
    times.insert(i.end);
  }
  int64_t p = 0;
  for (double t : times) {
    int64_t n = 0;
    for (auto &i : intervals)
      if (i.start <= t && t < i.end)
        n = checkedAdd(n, i.amount);
    p = std::max(p, n);
  }
  return p;
}
std::pair<std::vector<StaticAccess>, std::vector<StaticAccess>>
semanticAccesses(const Compute &op) {
  std::vector<StaticAccess> r, w;
  auto access = [](const std::string &b, Range m, Range n) {
    StaticAccess a;
    a.buffer = b;
    a.m = m;
    a.n = n;
    return a;
  };
  if (op.kind == "matmul") {
    r.push_back(access(op.inputs.at(0), op.m, op.k));
    r.push_back(access(op.inputs.at(1), op.k, op.n));
    if (op.accumulate)
      r.push_back(access(op.output, op.m, op.n));
  } else if (op.kind == "relu" || op.kind == "swiglu")
    for (auto &b : op.inputs)
      r.push_back(access(b, op.m, op.n));
  else if (op.kind != "fill")
    throw std::runtime_error("unsupported semantic operation");
  w.push_back(access(op.output, op.m, op.n));
  return {r, w};
}
struct Content {
  std::string version, event, device;
  double end = 0;
};
} // namespace
bool orbit::taskflow::checkStaticPlan(const StaticAnalysis &analysis,
                                      const StaticConfig &config,
                                      const StaticPlan &plan,
                                      std::vector<std::string> &errors) {
  if (!config.regionImplementations.empty() ||
      std::any_of(plan.events.begin(), plan.events.end(),
                  [](const StaticEvent &e) { return e.kind == "region"; }))
    return checkStaticRegionPlan(analysis, config, plan, errors);
  auto require = [&](bool condition, const std::string &message) {
    if (!condition && errors.size() < 100)
      errors.push_back(message);
  };
  try {
    require(plan.identityScheme == "llvm-json-sorted-17g-sha256",
            "unsupported plan identity scheme");
    require(plan.schema == "orbit.static.plan.v1", "wrong plan schema");
    require(plan.programId == staticFingerprint(printPlan(analysis.program)),
            "program identity mismatch");
    require(plan.configurationId == staticFingerprint(config.original),
            "configuration identity mismatch");
    require(plan.planId == staticFingerprint(serializeStaticPlan(plan, false)),
            "plan identity mismatch");
    require(plan.candidateId == analysis.program.candidateId,
            "candidate identity mismatch");
    require(plan.predictionKind == "static_estimate" &&
                plan.executionStatus == "not_executed",
            "invalid static execution claim");
    require(plan.storagePolicy ==
                "whole_buffer_per_location_layout_retained_for_single_graph",
            "unknown storage/lifetime policy");
    require(plan.initialState == "cold_inputs_only_visible_to_all_accessible_"
                                 "devices_at_declared_initial_locations",
            "unsupported initial-state assumption");
    const auto &program = analysis.program;
    const auto &assumptions = config.assumptions;
    require(assumptions.programSourceId.empty() ||
                assumptions.programSourceId == program.sourceId,
            "numerical contract belongs to another program");
    require(assumptions.executions == 1, "only one static execution supported");
    require(assumptions.overlap == "explicit" ||
                assumptions.overlap == "forbid",
            "unknown overlap policy");
    std::map<std::string, const Buffer *> buffers;
    for (auto &b : program.buffers)
      require(buffers.emplace(b.id, &b).second, "duplicate buffer");
    std::map<std::string, const Task *> tasks;
    for (auto &t : program.tasks)
      require(tasks.emplace(t.id, &t).second, "duplicate task");
    std::map<std::string, const StaticEvent *> events;
    for (auto &e : plan.events)
      require(events.emplace(e.id, &e).second, "duplicate event");
    std::map<std::string, const StaticAllocation *> allocs;
    for (auto &a : plan.allocations)
      require(allocs.emplace(a.id, &a).second, "duplicate allocation");
    std::map<std::string, const StaticImplementation *> impls;
    for (auto &i : config.implementations)
      require(impls.emplace(i.id, &i).second, "duplicate implementation");
    std::map<std::string, const StaticResource *> resources;
    for (auto &r : config.resources) {
      require(resources.emplace(r.id, &r).second, "duplicate resource");
      require(r.capacity > 0, "invalid capacity");
    }
    std::map<std::string, const StaticMemory *> memories;
    for (auto &m : config.memories) {
      require(memories.emplace(m.id, &m).second, "duplicate memory");
      require(m.capacityBytes > 0, "invalid memory capacity");
    }
    std::map<std::string, const StaticTransfer *> links, conversions, syncs;
    for (auto &d : config.links)
      require(links.emplace(d.id, &d).second, "duplicate link");
    for (auto &d : config.conversions)
      require(conversions.emplace(d.id, &d).second, "duplicate conversion");
    for (auto &d : config.synchronizations)
      require(syncs.emplace(d.id, &d).second, "duplicate sync");
    // Reconstruct operation order and accesses from the authenticated program,
    // not from the candidate plan's steps, versions or producer annotations.
    std::vector<StaticStep> steps;
    std::set<std::string> done;
    while (done.size() < tasks.size()) {
      bool progress = false;
      for (auto &t : program.tasks)
        if (!done.count(t.id) &&
            std::all_of(t.dependsOn.begin(), t.dependsOn.end(),
                        [&](auto &d) { return done.count(d); })) {
          for (size_t i = 0; i < t.operations.size(); ++i) {
            StaticStep s;
            s.id = t.id + ":" + std::to_string(i);
            s.taskId = t.id;
            s.operationIndex = i;
            s.operation = t.operations[i];
            if (i)
              s.dependsOn.push_back(t.id + ":" + std::to_string(i - 1));
            else
              for (auto &d : t.dependsOn)
                s.dependsOn.push_back(
                    d + ":" +
                    std::to_string(tasks.at(d)->operations.size() - 1));
            steps.push_back(s);
          }
          done.insert(t.id);
          progress = true;
        }
      if (!progress)
        throw std::runtime_error("cyclic/missing source dependency");
    }
    std::map<std::string, const StaticStep *> stepById;
    for (auto &s : steps)
      stepById[s.id] = &s;
    require(plan.steps.size() == steps.size(),
            "plan step metadata coverage mismatch");
    std::set<std::string> metadataSteps;
    for (auto &s : plan.steps) {
      require(metadataSteps.insert(s.id).second, "duplicate step metadata");
      const auto &expected = *stepById.at(s.id);
      const auto &x = s.operation, &y = expected.operation;
      require(s.taskId == expected.taskId &&
                  s.operationIndex == expected.operationIndex &&
                  s.dependsOn == expected.dependsOn && x.kind == y.kind &&
                  x.inputs == y.inputs && x.output == y.output && x.m == y.m &&
                  x.n == y.n && (x.kind != "matmul" || x.k == y.k) &&
                  x.accumulate == y.accumulate,
              "plan step metadata differs from semantic program");
    }
    std::map<std::string, std::pair<std::set<int64_t>, std::set<int64_t>>> cuts;
    for (auto &b : program.buffers) {
      if (b.rows <= 0 || b.cols <= 0 || b.rows > INT64_MAX / b.cols / 4 ||
          b.dtype != "f32")
        throw std::runtime_error("unsupported/overflowing source storage");
      cuts[b.id] = {{0, b.rows}, {0, b.cols}};
    }
    for (auto &s : steps) {
      auto rw = semanticAccesses(s.operation);
      for (auto a : rw.first) {
        cuts.at(a.buffer).first.insert(a.m.lo);
        cuts.at(a.buffer).first.insert(a.m.hi);
        cuts.at(a.buffer).second.insert(a.n.lo);
        cuts.at(a.buffer).second.insert(a.n.hi);
      }
      for (auto a : rw.second) {
        cuts.at(a.buffer).first.insert(a.m.lo);
        cuts.at(a.buffer).first.insert(a.m.hi);
        cuts.at(a.buffer).second.insert(a.n.lo);
        cuts.at(a.buffer).second.insert(a.n.hi);
      }
    }
    std::map<std::string, std::vector<Cell>> cells;
    size_t totalCells = 0;
    for (auto &[b, xy] : cuts) {
      std::vector<int64_t> x(xy.first.begin(), xy.first.end()),
          y(xy.second.begin(), xy.second.end());
      if ((x.size() - 1) > size_t(assumptions.maxRegionCells) / (y.size() - 1))
        throw std::runtime_error("checker region budget exceeded");
      totalCells += (x.size() - 1) * (y.size() - 1);
      if (totalCells > size_t(assumptions.maxRegionCells))
        throw std::runtime_error("checker region budget exceeded");
      for (size_t i = 1; i < x.size(); ++i)
        for (size_t j = 1; j < y.size(); ++j)
          cells[b].emplace_back(b, x[i - 1], x[i], y[j - 1], y[j]);
    }
    require(plan.regionCellCount == int64_t(totalCells),
            "forged region partition count");
    auto covered = [&](const StaticAccess &a) {
      std::vector<Cell> result;
      const auto &b = *buffers.at(a.buffer);
      if (!(a.m.lo >= 0 && a.m.hi <= b.rows && a.m.lo < a.m.hi && a.n.lo >= 0 &&
            a.n.hi <= b.cols && a.n.lo < a.n.hi))
        throw std::runtime_error("access outside buffer");
      int64_t count = 0;
      for (auto &c : cells.at(a.buffer))
        if (a.m.lo <= std::get<1>(c) && std::get<2>(c) <= a.m.hi &&
            a.n.lo <= std::get<3>(c) && std::get<4>(c) <= a.n.hi) {
          result.push_back(c);
          count += bytes(c);
        }
      require(count == (a.m.hi - a.m.lo) * (a.n.hi - a.n.lo) * 4,
              "event region not source-cell aligned");
      return result;
    };
    Tokens semantic;
    for (auto &b : program.buffers)
      if (b.role == "input")
        for (auto &cell : cells[b.id])
          semantic[cell] = "initial:" + b.id;
    std::map<std::string, Tokens> expectedReads, expectedWrites;
    for (auto &s : steps) {
      auto rw = semanticAccesses(s.operation);
      for (auto &a : rw.first)
        for (auto &cell : covered(a)) {
          require(semantic.count(cell), "missing initialization/producer");
          expectedReads[s.id][cell] = semantic[cell];
        }
      for (auto &a : rw.second)
        for (auto &cell : covered(a)) {
          semantic[cell] = s.id;
          expectedWrites[s.id][cell] = s.id;
        }
    }
    std::map<std::string, const StaticEvent *> compute;
    for (auto &e : plan.events)
      if (e.kind == "compute")
        require(stepById.count(e.stepId) &&
                    compute.emplace(e.stepId, &e).second,
                "duplicate/unknown compute");
    require(compute.size() == steps.size(),
            "compute coverage missing fill/K/operation");
    std::map<std::string, std::string> actualAssignment;
    for (auto &[sid, e] : compute)
      actualAssignment[sid] = e->implementationId;
    require(actualAssignment == plan.assignment,
            "assignment differs from events");
    std::map<std::string, std::set<std::string>> ancestors;
    std::set<std::string> visiting;
    std::function<void(const std::string &)> visit =
        [&](const std::string &id) {
          if (ancestors.count(id))
            return;
          if (!visiting.insert(id).second)
            throw std::runtime_error("event dependency cycle");
          std::set<std::string> deps;
          for (auto &d : events.at(id)->dependsOn) {
            visit(d);
            deps.insert(d);
            deps.insert(ancestors[d].begin(), ancestors[d].end());
          }
          visiting.erase(id);
          ancestors[id] = std::move(deps);
        };
    double end = 0;
    for (auto &e : plan.events) {
      visit(e.id);
      require(validNumber(e.durationMs) && validNumber(e.startMs) &&
                  validNumber(e.endMs),
              "invalid event time");
      require(near(e.endMs - e.startMs, e.durationMs), "duration mismatch");
      end = std::max(end, e.endMs);
      for (auto &d : e.dependsOn)
        require(events.at(d)->endMs <= e.startMs + 1e-8,
                "dependency overlaps consumer");
    }
    require(near(plan.totals.makespanMs, end), "forged makespan");
    for (auto &s : steps)
      for (auto &d : s.dependsOn)
        require(ancestors.at(compute.at(s.id)->id).count(compute.at(d)->id),
                "missing semantic completion/order dependency");
    std::map<std::string, int64_t> sizes;
    std::set<std::tuple<std::string, std::string, std::string, std::string>>
        physicalKeys;
    for (auto &a : plan.allocations) {
      memories.at(a.memory);
      int64_t size = 0;
      if (a.lifetime == "event") {
        auto &e = *events.at(a.createdBy);
        if (e.kind == "compute")
          size = impls.at(e.implementationId)->temporaryBytes;
        else if (e.kind == "copy")
          size = links.at(e.link)->temporaryBytes;
        else if (e.kind == "convert")
          size = conversions.at(e.conversion)->temporaryBytes;
        else
          require(false, "invalid temporary owner");
        require(near(a.startMs, e.startMs) && near(a.endMs, e.endMs),
                "forged temporary lifetime");
      } else {
        auto &b = *buffers.at(a.buffer);
        size = b.rows * b.cols * 4;
        require(a.dtype == "f32" && a.m == Range{0, b.rows} &&
                    a.n == Range{0, b.cols},
                "forged physical buffer shape/dtype");
        require(a.lifetime == "graph" && near(a.startMs, 0) &&
                    near(a.endMs, end),
                "forged graph lifetime");
        require(
            physicalKeys.emplace(a.buffer, a.memory, a.layout, a.dtype).second,
            "duplicate physical replica identity");
        auto initial = assumptions.initialLocations.find(a.buffer);
        bool isInitial = b.role == "input" &&
                         initial != assumptions.initialLocations.end() &&
                         initial->second == a.memory &&
                         a.layout == "row_major" && a.dtype == "f32";
        require(a.initial == isInitial, "forged initial allocation state");
      }
      require(a.bytes == size && size >= 0, "forged allocation size");
      sizes[a.id] = size;
    }
    using Location = std::pair<std::string, Cell>;
    std::map<Location, std::vector<Content>> values;
    for (auto &a : plan.allocations)
      if (a.lifetime == "graph") {
        auto &b = *buffers.at(a.buffer);
        auto initial = assumptions.initialLocations.find(b.id);
        if (b.role == "input" &&
            initial != assumptions.initialLocations.end() &&
            initial->second == a.memory && a.layout == "row_major" &&
            a.dtype == "f32")
          for (auto &cell : cells[b.id])
            values[{a.id, cell}].push_back({"initial:" + b.id, "", "", 0});
      }
    std::vector<const StaticEvent *> ordered;
    for (auto &e : plan.events)
      ordered.push_back(&e);
    std::sort(ordered.begin(), ordered.end(), [&](auto l, auto r) {
      return std::make_tuple(l->startMs, ancestors[l->id].size(), l->id) <
             std::make_tuple(r->startMs, ancestors[r->id].size(), r->id);
    });
    auto visible = [&](const Content &v, const Cell &cell,
                       const std::string &aid, const std::string &target,
                       const std::set<std::string> &eligible) {
      for (auto &id : eligible) {
        auto &s = *events.at(id);
        if (s.kind != "sync" || s.sourceDevice != v.device ||
            s.targetDevice != target ||
            (!v.event.empty() && !ancestors[s.id].count(v.event)))
          continue;
        for (auto &r : s.reads)
          if (r.allocation == aid && r.version == v.version) {
            auto region = covered(r);
            if (std::find(region.begin(), region.end(), cell) != region.end())
              return true;
          }
      }
      return false;
    };
    for (auto ep : ordered) {
      const auto &e = *ep;
      Tokens reads, writes, content;
      std::map<Cell, const StaticAccess *> rentries, wentries;
      require(e.kind == "compute" || e.kind == "copy" || e.kind == "convert" ||
                  e.kind == "sync",
              "unknown event kind");
      require(
          e.costBoundary ==
              (e.kind == "compute" ? "compute_and_local_memory_access_only"
               : e.kind == "sync"
                   ? "visibility_only"
                   : "transfer_or_conversion_including_completion_visibility"),
          "event cost boundary mismatch");
      for (auto &r : e.reads) {
        require(allocs.at(r.allocation)->buffer == r.buffer,
                "logical/physical storage mismatch");
        for (auto &cell : covered(r)) {
          require(reads.emplace(cell, r.version).second,
                  "duplicate read region");
          rentries[cell] = &r;
        }
      }
      for (auto &w : e.writes) {
        require(allocs.at(w.allocation)->buffer == w.buffer,
                "logical/physical storage mismatch");
        for (auto &cell : covered(w)) {
          require(writes.emplace(cell, w.version).second,
                  "duplicate write region");
          wentries[cell] = &w;
        }
      }
      for (auto &[cell, access] : rentries) {
        auto &history = values[{access->allocation, cell}];
        const Content *chosen = nullptr;
        for (auto &v : history)
          if (v.end <= e.startMs + 1e-8 &&
              (v.event.empty() || ancestors[e.id].count(v.event))) {
            if (!chosen ||
                std::make_pair(v.end, v.event.empty()
                                          ? size_t(0)
                                          : ancestors[v.event].size() + 1) >
                    std::make_pair(chosen->end,
                                   chosen->event.empty()
                                       ? size_t(0)
                                       : ancestors[chosen->event].size() + 1))
              chosen = &v;
          }
        require(chosen, "read before valid production/transfer");
        if (chosen) {
          content[cell] = chosen->version;
          require(access->version == chosen->version, "forged read version");
          if (e.kind == "compute" && !chosen->device.empty() &&
              chosen->device != e.device)
            require(visible(*chosen, cell, access->allocation, e.device,
                            ancestors[e.id]),
                    "missing shared-memory visibility sync");
        }
      }
      int64_t scratch = 0;
      std::string scratchMemory;
      if (e.kind == "compute") {
        const auto &s = *stepById.at(e.stepId);
        const auto &op = s.operation;
        const auto &i = *impls.at(e.implementationId);
        const auto &cost = i.cost;
        require(e.taskId == s.taskId && e.operationIndex == s.operationIndex,
                "compute Taskflow operation binding mismatch");
        require(content == expectedReads[s.id] &&
                    writes == expectedWrites[s.id],
                "compute wrong semantic version/region");
        require(e.device == i.device && contains(i.operations, op.kind),
                "implementation mismatch");
        require(contains(assumptions.allowedNumericalContracts,
                         i.numericalContract),
                "numerical contract mismatch");
        int64_t m = op.m.hi - op.m.lo, n = op.n.hi - op.n.lo,
                k = op.kind == "matmul" ? op.k.hi - op.k.lo : 1;
        std::map<std::string, int64_t> dims{{"m", m}, {"n", n}, {"k", k}};
        for (auto &[axis, value] : dims) {
          auto bound = cost.shapeRange.at(axis);
          require(bound.lo <= value && value <= bound.hi,
                  "implementation outside cost range");
        }
        for (auto &[axis, multiple] : i.multiples)
          require(multiple > 0 && dims.at(axis) % multiple == 0,
                  "illegal physical shape");
        require(i.allowAccumulate || !op.accumulate,
                "implementation cannot accumulate");
        if (i.fullKOnly)
          require(op.kind == "matmul" &&
                      op.k == Range{0, buffers.at(op.inputs.at(0))->cols},
                  "implementation requires full K");
        if (i.overwriteZeroAccumulator) {
          require(s.operationIndex > 0, "overwrite requires initializer");
          if (s.operationIndex) {
            const auto &prev =
                tasks.at(s.taskId)->operations.at(s.operationIndex - 1);
            require(prev.kind == "fill" && prev.output == op.output &&
                        prev.m == op.m && prev.n == op.n,
                    "overwrite requires matching zero initializer");
          }
        }
        require(cost.kind == "affine" && cost.scope == "compute_only" &&
                    validNumber(cost.fixedMs) &&
                    validNumber(cost.perElementMs) &&
                    validNumber(cost.perFlopMs),
                "unknown/double-counted compute cost");
        require(provenance(e.provenance, cost.provenance),
                "forged compute provenance");
        double duration =
            cost.fixedMs + cost.perElementMs * double(m) * n +
            cost.perFlopMs * (op.kind == "matmul" ? 2. * m * n * k : 0.);
        require(near(e.durationMs, duration) &&
                    e.resourceDemands == i.resources,
                "forged compute cost/resources");
        for (auto *list : {&e.reads, &e.writes})
          for (auto &access : *list) {
            const auto &a = *allocs.at(access.allocation);
            require(a.memory == i.memory && a.layout == i.layout &&
                        a.dtype == i.dtype &&
                        contains(memories.at(a.memory)->accessibleDevices,
                                 e.device),
                    "compute storage/layout/access mismatch");
          }
        scratch = i.temporaryBytes;
        scratchMemory = i.memory;
        content = expectedWrites[s.id];
      } else if (e.kind == "copy" || e.kind == "convert") {
        require(!reads.empty() && reads == writes,
                "transfer changes region/version");
        const auto &d = *(e.kind == "copy" ? links.at(e.link)
                                           : conversions.at(e.conversion));
        int64_t size = 0;
        for (auto &[cell, v] : reads)
          size = checkedAdd(size, bytes(cell));
        require(e.bytes == size && size >= d.minBytes && size <= d.maxBytes,
                "forged transfer bytes/range");
        require(validNumber(d.latencyMs) && d.bandwidthBytesPerMs > 0 &&
                    std::isfinite(d.bandwidthBytesPerMs),
                "unknown transfer cost");
        require(near(e.durationMs,
                     d.latencyMs + double(size) / d.bandwidthBytesPerMs),
                "forged communication cost");
        require(e.resourceDemands ==
                        std::map<std::string, int64_t>{{d.resource, 1}} &&
                    provenance(e.provenance, d.provenance),
                "forged communication resources/provenance");
        for (auto &[cell, read] : rentries) {
          const auto &a = *allocs.at(read->allocation),
                     &b = *allocs.at(wentries.at(cell)->allocation);
          if (e.kind == "copy")
            require(a.memory == d.source && b.memory == d.target &&
                        a.layout == b.layout && a.dtype == b.dtype,
                    "wrong copy path/layout");
          else
            require(a.memory == d.memory && b.memory == d.memory &&
                        a.layout == d.sourceLayout &&
                        b.layout == d.targetLayout &&
                        a.dtype == d.sourceDtype && b.dtype == d.targetDtype,
                    "wrong conversion contract");
        }
        scratch = d.temporaryBytes;
        scratchMemory = e.kind == "copy" ? d.target : d.memory;
      } else if (e.kind == "sync") {
        const auto &d = *syncs.at(e.synchronization);
        require(e.writes.empty() && !e.reads.empty() && e.bytes == 0,
                "sync invents data/bytes");
        require(e.sourceDevice == d.sourceDevice &&
                    e.targetDevice == d.targetDevice && e.memory == d.memory,
                "wrong synchronization path");
        require(near(e.durationMs, d.latencyMs) &&
                    e.resourceDemands ==
                        std::map<std::string, int64_t>{{d.resource, 1}} &&
                    provenance(e.provenance, d.provenance),
                "forged sync cost/resources/provenance");
        for (auto &r : e.reads)
          require(allocs.at(r.allocation)->memory == d.memory,
                  "sync reads wrong memory");
      }
      int64_t scratchFound = 0;
      for (auto &a : plan.allocations)
        if (a.lifetime == "event" && a.createdBy == e.id) {
          scratchFound = checkedAdd(scratchFound, a.bytes);
          require(a.memory == scratchMemory, "scratch in wrong memory");
        }
      require(scratchFound == scratch, "scratch omitted/duplicated");
      for (auto &w : e.writes)
        for (auto &cell : covered(w))
          values[{w.allocation, cell}].push_back(
              {content.count(cell) ? content[cell] : "INVALID", e.id,
               e.kind == "compute" ? e.device : "", e.endMs});
    }
    // Check externally visible output cells rather than trusting output
    // metadata.
    std::set<std::string> outputDeps(plan.outputDependencies.begin(),
                                     plan.outputDependencies.end());
    for (auto &id : outputDeps)
      events.at(id);
    Tokens claimedOutputs;
    for (auto &o : plan.outputs)
      for (auto &cell : covered(o)) {
        require(claimedOutputs.emplace(cell, o.version).second,
                "duplicate output cell");
        require(contains(program.outputs, o.buffer), "nonpublic output");
        require(o.version == semantic.at(cell), "wrong final output version");
      }
    for (auto &bid : program.outputs) {
      auto location = assumptions.outputLocations.at(bid);
      auto &memory = *memories.at(location);
      auto it = assumptions.outputDevices.find(bid);
      std::string device;
      if (it != assumptions.outputDevices.end())
        device = it->second;
      else {
        require(memory.accessibleDevices.size() == 1,
                "ambiguous output consumer");
        device = memory.accessibleDevices.at(0);
      }
      require(contains(memory.accessibleDevices, device),
              "output device inaccessible");
      require(plan.outputConsumers.at(bid) == device,
              "forged output consumer metadata");
      for (auto &cell : cells[bid]) {
        require(claimedOutputs.count(cell), "missing public output cell");
        bool found = false;
        for (auto &o : plan.outputs) {
          auto region = covered(o);
          if (o.buffer != bid ||
              std::find(region.begin(), region.end(), cell) == region.end())
            continue;
          auto &a = *allocs.at(o.allocation);
          require(a.memory == location && a.layout == "row_major" &&
                      a.dtype == "f32",
                  "wrong output allocation");
          auto &history = values[{a.id, cell}];
          const Content *last = nullptr;
          for (auto &v : history)
            if (!last ||
                std::make_pair(v.end, v.event.empty()
                                          ? size_t(0)
                                          : ancestors[v.event].size() + 1) >
                    std::make_pair(last->end,
                                   last->event.empty()
                                       ? size_t(0)
                                       : ancestors[last->event].size() + 1))
              last = &v;
          if (last && last->version == semantic.at(cell) &&
              last->end <= end + 1e-8) {
            found = true;
            if (!last->device.empty() && last->device != device)
              require(visible(*last, cell, a.id, device, outputDeps),
                      "output lacks consumer visibility");
          }
        }
        require(found, "missing final output data/location");
      }
    }
    for (auto &[rid, r] : resources) {
      std::vector<Interval> iv;
      for (auto &e : plan.events) {
        auto it = e.resourceDemands.find(rid);
        iv.push_back({e.startMs, e.endMs,
                      it == e.resourceDemands.end() ? 0 : it->second});
      }
      require(peak(iv) <= r->capacity, "resource capacity exceeded " + rid);
    }
    for (auto &e : plan.events)
      for (auto &[rid, n] : e.resourceDemands)
        require(resources.count(rid) && n > 0, "invalid resource demand");
    if (assumptions.overlap == "forbid") {
      std::vector<Interval> iv;
      for (auto &e : plan.events)
        iv.push_back({e.startMs, e.endMs, 1});
      require(peak(iv) <= 1, "overlap prohibited");
    }
    std::map<std::string, int64_t> memoryPeaks;
    for (auto &[mid, m] : memories) {
      int64_t graphBytes = 0, floor = 0;
      std::vector<Interval> iv;
      std::map<std::string, int64_t> scratchByEvent;
      for (auto &a : plan.allocations)
        if (a.memory == mid) {
          if (a.lifetime == "event") {
            auto &e = *events.at(a.createdBy);
            iv.push_back({e.startMs, e.endMs, sizes[a.id]});
            scratchByEvent[a.createdBy] =
                checkedAdd(scratchByEvent[a.createdBy], sizes[a.id]);
            floor = std::max(floor, scratchByEvent[a.createdBy]);
          } else
            graphBytes = checkedAdd(graphBytes, sizes[a.id]);
        }
      auto p = checkedAdd(graphBytes, std::max(peak(iv), floor));
      memoryPeaks[mid] = p;
      require(p <= m->capacityBytes, "memory capacity exceeded " + mid);
    }
    require(memoryPeaks == plan.totals.peakMemoryBytes, "forged peak memory");
    double computeMs = 0, copyMs = 0, convertMs = 0, syncMs = 0;
    int64_t transferBytes = 0;
    for (auto &e : plan.events) {
      if (e.kind == "compute")
        computeMs += e.durationMs;
      else if (e.kind == "copy") {
        copyMs += e.durationMs;
        transferBytes = checkedAdd(transferBytes, e.bytes);
      } else if (e.kind == "convert")
        convertMs += e.durationMs;
      else if (e.kind == "sync")
        syncMs += e.durationMs;
    }
    require(near(computeMs, plan.totals.computeMs) &&
                near(copyMs, plan.totals.copyMs) &&
                near(convertMs, plan.totals.convertMs) &&
                near(syncMs, plan.totals.syncMs) &&
                transferBytes == plan.totals.transferBytes,
            "forged aggregate costs/bytes");
    std::map<Location, std::map<std::string, bool>> occupancy;
    for (auto &e : plan.events) {
      for (auto &r : e.reads)
        for (auto &cell : covered(r))
          occupancy[{r.allocation, cell}].emplace(e.id, false);
      for (auto &w : e.writes)
        for (auto &cell : covered(w))
          occupancy[{w.allocation, cell}][e.id] = true;
    }
    for (auto &[location, users] : occupancy)
      for (auto i = users.begin(); i != users.end(); ++i)
        for (auto j = std::next(i); j != users.end(); ++j)
          if (i->second || j->second)
            require(ancestors[i->first].count(j->first) ||
                        ancestors[j->first].count(i->first),
                    "unordered physical read/write hazard");
  } catch (const std::exception &e) {
    errors.push_back(std::string("malformed plan/config/source: ") + e.what());
  }
  return errors.empty();
}
