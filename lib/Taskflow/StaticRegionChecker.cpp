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
using Content = std::map<Cell, std::string>;
bool near(double x, double y) {
  return std::isfinite(x) && std::isfinite(y) &&
         std::abs(x - y) <= 1e-8 * std::max({1., std::abs(x), std::abs(y)});
}
bool overlap(const StaticAccess &a, const StaticAccess &b) {
  return a.buffer == b.buffer && a.m.lo < b.m.hi && b.m.lo < a.m.hi &&
         a.n.lo < b.n.hi && b.n.lo < a.n.hi;
}
int64_t add(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || b > INT64_MAX - a)
    throw std::runtime_error("storage accounting overflow");
  return a + b;
}
std::vector<StaticAccess> reads(const Compute &op) {
  std::vector<StaticAccess> r;
  if (op.kind == "matmul") {
    r.push_back({op.inputs.at(0), "", "", op.m, op.k});
    r.push_back({op.inputs.at(1), "", "", op.k, op.n});
    if (op.accumulate)
      r.push_back({op.output, "", "", op.m, op.n});
  } else if (op.kind == "relu" || op.kind == "swiglu")
    for (auto &b : op.inputs)
      r.push_back({b, "", "", op.m, op.n});
  else if (op.kind != "fill")
    throw std::runtime_error("unknown semantic operation");
  return r;
}
struct Interval {
  double start, end;
  int64_t amount;
};
int64_t peak(const std::vector<Interval> &iv) {
  std::set<double> times;
  for (auto &i : iv) {
    times.insert(i.start);
    times.insert(i.end);
  }
  int64_t p = 0;
  for (auto t : times) {
    int64_t use = 0;
    for (auto &i : iv)
      if (i.start <= t && t < i.end)
        use = add(use, i.amount);
    p = std::max(p, use);
  }
  return p;
}
} // namespace
// Independent verifier: does not invoke region matching, graph construction,
// mapper, scheduler or their cost helper. Source dependencies/versions and
// opaque boundaries are reconstructed here from the actual-body analysis.
bool orbit::taskflow::checkStaticRegionPlan(const StaticAnalysis &a,
                                            const StaticConfig &c,
                                            const StaticPlan &p,
                                            std::vector<std::string> &errors) {
  auto need = [&](bool ok, const std::string &s) {
    if (!ok && errors.size() < 100)
      errors.push_back(s);
  };
  try {
    need(p.identityScheme == "llvm-json-sorted-17g-sha256",
         "unknown identity scheme");
    need(p.schema == "orbit.static.plan.v1" &&
             p.programId == staticFingerprint(printPlan(a.program)) &&
             p.candidateId == a.program.candidateId,
         "source identity mismatch");
    need(p.configurationId == staticFingerprint(c.original) &&
             p.planId == staticFingerprint(serializeStaticPlan(p, false)),
         "plan/config identity mismatch");
    need(p.predictionKind == "static_estimate" &&
             p.executionStatus == "not_executed",
         "invalid execution claim");
    need(c.assumptions.executionState == "warm_contexts_weights" &&
             p.initialState == c.assumptions.executionState,
         "cold/hot boundary mismatch");
    need(c.assumptions.programSourceId.empty() ||
             c.assumptions.programSourceId == a.program.sourceId,
         "source-bound numerical contract mismatch");
    need(p.storagePolicy ==
             "whole_buffer_per_location_layout_retained_for_single_graph",
         "unsupported region storage policy");
    need(p.costBoundary ==
             "opaque_warm_host_to_host: activation conversion, staging, "
             "synchronization, submissions, intermediate handling and final "
             "conversion included; no duplicate internal communication",
         "forged opaque plan cost boundary");
    need(c.assumptions.executions == 1, "unsupported repeat model");
    std::map<std::string, const Buffer *> buffers;
    for (auto &b : a.program.buffers) {
      need(buffers.emplace(b.id, &b).second, "duplicate storage");
      if (b.rows <= 0 || b.cols <= 0 || b.rows > INT64_MAX / 4 / b.cols)
        throw std::runtime_error("source storage overflow");
    }
    std::map<std::string, const StaticStep *> steps;
    for (auto &s : a.steps)
      need(steps.emplace(s.id, &s).second, "duplicate source step");
    need(p.steps.size() == a.steps.size(), "missing source operation bindings");
    for (size_t i = 0; i < std::min(p.steps.size(), a.steps.size()); ++i) {
      auto &x = p.steps[i];
      auto &y = a.steps[i];
      need(x.id == y.id && x.taskId == y.taskId &&
               x.operationIndex == y.operationIndex &&
               x.dependsOn == y.dependsOn &&
               x.operation.kind == y.operation.kind &&
               x.operation.inputs == y.operation.inputs &&
               x.operation.output == y.operation.output &&
               x.operation.m == y.operation.m &&
               x.operation.n == y.operation.n &&
               x.operation.k == y.operation.k &&
               x.operation.accumulate == y.operation.accumulate,
           "forged source step binding");
    }
    std::map<std::string, const StaticRegionImplementation *> impls;
    for (auto &i : c.regionImplementations)
      need(impls.emplace(i.id, &i).second, "duplicate implementation");
    std::map<std::string, const StaticMemory *> memories;
    for (auto &m : c.memories)
      memories[m.id] = &m;
    std::map<std::string, const StaticResource *> resources;
    for (auto &r : c.resources)
      resources[r.id] = &r;
    std::map<std::string, const StaticEvent *> events;
    for (auto &e : p.events)
      need(events.emplace(e.id, &e).second, "duplicate event");
    std::map<std::string, const StaticAllocation *> allocs;
    std::set<std::tuple<std::string, std::string, std::string, std::string>>
        locations;
    for (auto &al : p.allocations) {
      need(allocs.emplace(al.id, &al).second, "duplicate allocation");
      memories.at(al.memory);
      need(al.bytes >= 0, "negative allocation");
      if (!al.buffer.empty()) {
        auto &b = *buffers.at(al.buffer);
        need(al.lifetime == "graph" && al.m == Range{0, b.rows} &&
                 al.n == Range{0, b.cols} && al.bytes == b.rows * b.cols * 4 &&
                 al.dtype == "f32" && al.layout == "row_major",
             "invalid external buffer allocation");
        need(
            locations.emplace(al.buffer, al.memory, al.layout, al.dtype).second,
            "duplicate physical storage");
        auto initial = c.assumptions.initialLocations.find(b.id);
        need(al.initial == (b.role == "input" &&
                            initial != c.assumptions.initialLocations.end() &&
                            initial->second == al.memory),
             "invented prepared input copy");
      } else
        need(al.lifetime == "event" && !al.initial,
             "invalid opaque workspace lifetime");
    }
    std::map<std::string, std::pair<std::set<int64_t>, std::set<int64_t>>> cuts;
    for (auto &b : a.program.buffers)
      cuts[b.id] = {{0, b.rows}, {0, b.cols}};
    for (auto &s : a.steps) {
      auto list = reads(s.operation);
      list.push_back(
          {s.operation.output, "", "", s.operation.m, s.operation.n});
      for (auto &r : list) {
        cuts.at(r.buffer).first.insert(r.m.lo);
        cuts.at(r.buffer).first.insert(r.m.hi);
        cuts.at(r.buffer).second.insert(r.n.lo);
        cuts.at(r.buffer).second.insert(r.n.hi);
      }
    }
    std::map<std::string, std::vector<Cell>> cells;
    int64_t cellCount = 0;
    for (auto &[id, xy] : cuts) {
      std::vector<int64_t> x(xy.first.begin(), xy.first.end()),
          y(xy.second.begin(), xy.second.end());
      if ((x.size() - 1) >
          uint64_t(c.assumptions.maxRegionCells) / (y.size() - 1))
        throw std::runtime_error("region cell budget");
      cellCount = add(cellCount, (x.size() - 1) * (y.size() - 1));
      if (cellCount > c.assumptions.maxRegionCells)
        throw std::runtime_error("region cell budget");
      for (size_t i = 1; i < x.size(); ++i)
        for (size_t j = 1; j < y.size(); ++j)
          cells[id].emplace_back(id, x[i - 1], x[i], y[j - 1], y[j]);
    }
    need(p.regionCellCount == cellCount, "forged region cell count");
    auto covered = [&](const StaticAccess &r) {
      auto &b = *buffers.at(r.buffer);
      if (r.m.lo < 0 || r.n.lo < 0 || r.m.hi > b.rows || r.n.hi > b.cols ||
          r.m.hi <= r.m.lo || r.n.hi <= r.n.lo)
        throw std::runtime_error("out-of-bounds access");
      std::vector<Cell> v;
      int64_t area = 0;
      for (auto &x : cells.at(r.buffer))
        if (r.m.lo <= std::get<1>(x) && std::get<2>(x) <= r.m.hi &&
            r.n.lo <= std::get<3>(x) && std::get<4>(x) <= r.n.hi) {
          v.push_back(x);
          area = add(area, (std::get<2>(x) - std::get<1>(x)) *
                               (std::get<4>(x) - std::get<3>(x)));
        }
      need(area == (r.m.hi - r.m.lo) * (r.n.hi - r.n.lo),
           "misaligned access region");
      return v;
    };
    Content state;
    for (auto &b : a.program.buffers)
      if (b.role == "input")
        for (auto &x : cells[b.id])
          state[x] = "initial:" + b.id;
    std::map<std::string, Content> before, after;
    for (auto &s : a.steps) {
      for (auto &r : reads(s.operation))
        for (auto &x : covered(r)) {
          need(state.count(x), "uninitialized source read");
          before[s.id][x] = state[x];
        }
      StaticAccess w{s.operation.output, "", "", s.operation.m, s.operation.n};
      for (auto &x : covered(w)) {
        state[x] = s.id;
        after[s.id][x] = s.id;
      }
    }
    std::map<std::string, std::string> owner, assignment;
    for (auto &e : p.events) {
      need(e.kind == "region",
           "mixed/internal communication double billing in opaque region plan");
      for (auto &sid : e.coveredSteps) {
        steps.at(sid);
        need(owner.emplace(sid, e.id).second,
             "overlapping implementation coverage");
        assignment[sid] = e.implementationId;
      }
    }
    need(owner.size() == steps.size() && assignment == p.assignment,
         "missing or duplicate operation coverage");
    std::map<std::string, std::set<std::string>> ancestors;
    std::set<std::string> active;
    std::function<void(const std::string &)> visit =
        [&](const std::string &id) {
          if (ancestors.count(id))
            return;
          if (!active.insert(id).second)
            throw std::runtime_error("contracted dependency cycle");
          std::set<std::string> all;
          for (auto &d : events.at(id)->dependsOn) {
            visit(d);
            all.insert(d);
            all.insert(ancestors.at(d).begin(), ancestors.at(d).end());
          }
          active.erase(id);
          ancestors[id] = all;
        };
    for (auto &e : p.events)
      visit(e.id);
    for (auto &s : a.steps)
      for (auto &dep : s.dependsOn)
        if (owner.at(s.id) != owner.at(dep))
          need(ancestors.at(owner.at(s.id)).count(owner.at(dep)),
               "missing external dependency");
    // Source order imposes physical RAW/WAR/WAW constraints even when no data
    // transfer is needed at the shared host boundary.
    for (size_t l = 0; l < a.steps.size(); ++l)
      for (size_t r = l + 1; r < a.steps.size(); ++r) {
        auto &x = a.steps[l];
        auto &y = a.steps[r];
        if (owner.at(x.id) == owner.at(y.id))
          continue;
        auto xr = reads(x.operation), yr = reads(y.operation);
        StaticAccess xw{x.operation.output, "", "", x.operation.m,
                        x.operation.n},
            yw{y.operation.output, "", "", y.operation.m, y.operation.n};
        bool hazard = overlap(xw, yw);
        for (auto &v : yr)
          hazard |= overlap(xw, v);
        for (auto &v : xr)
          hazard |= overlap(v, yw);
        if (hazard)
          need(ancestors.at(owner.at(y.id)).count(owner.at(x.id)),
               "missing external physical read/write dependency");
      }
    double end = 0, total = 0;
    std::map<std::string, std::map<Cell, std::pair<std::string, std::string>>>
        available;
    for (auto &al : p.allocations)
      if (al.initial)
        for (auto &x : cells.at(al.buffer))
          available[al.id][x] = {"initial:" + al.buffer, ""};
    std::vector<const StaticEvent *> ordered;
    for (auto &e : p.events)
      ordered.push_back(&e);
    std::stable_sort(ordered.begin(), ordered.end(), [&](auto x, auto y) {
      return std::make_pair(x->startMs, ancestors[x->id].size()) <
             std::make_pair(y->startMs, ancestors[y->id].size());
    });
    for (auto *ep : ordered) {
      auto &e = *ep;
      auto &i = *impls.at(e.implementationId);
      need(e.regionPattern == i.pattern && e.regionMode == i.mode &&
               e.modelId == i.modelId && e.preparedState == i.preparedState &&
               e.preparedState == c.assumptions.executionState,
           "implementation/model/state mismatch");
      need(e.costBoundary == "opaque_warm_host_to_host",
           "incorrect whole-region cost boundary");
      need(e.device == i.device && e.resourceDemands == i.resources,
           "incorrect conservative resource occupation");
      need(i.layout == "row_major" && i.dtype == "f32" && e.bytes == 0,
           "opaque boundary layout/dtype/traffic mismatch");
      need(std::find(c.assumptions.allowedNumericalContracts.begin(),
                     c.assumptions.allowedNumericalContracts.end(),
                     i.numericalContract) !=
               c.assumptions.allowedNumericalContracts.end(),
           "numerical contract not permitted");
      need(e.coveredSteps.size() == (i.pattern == "matmul_pair" ? 4u : 2u),
           "invalid region operation count");
      if (e.coveredSteps.size() != (i.pattern == "matmul_pair" ? 4u : 2u))
        continue;
      std::vector<const StaticStep *> mm;
      for (size_t j = 0; j < e.coveredSteps.size(); j += 2) {
        auto &f = *steps.at(e.coveredSteps[j]);
        auto &m = *steps.at(e.coveredSteps[j + 1]);
        auto &op = m.operation;
        auto &z = f.operation;
        need(z.kind == "fill" && op.kind == "matmul" && z.output == op.output &&
                 z.m == op.m && z.n == op.n && op.accumulate,
             "region does not preserve zero/full matmul semantics");
        need(m.taskId == f.taskId && m.operationIndex == f.operationIndex + 1,
             "fill not adjacent to matmul");
        need(buffers.at(op.inputs.at(1))->role == "input" &&
                 op.output != op.inputs.at(0) && op.output != op.inputs.at(1),
             "nonimmutable weight or aliased matmul target");
        need(op.k == Range{0, buffers.at(op.inputs.at(0))->cols},
             "region implementation requires full K");
        mm.push_back(&m);
      }
      auto &first = *mm.front();
      auto &last = *mm.back();
      auto &op = first.operation;
      std::map<std::string, int64_t> shape{
          {"m", op.m.hi - op.m.lo},
          {"k", op.k.hi - op.k.lo},
          {"n", op.n.hi - op.n.lo},
          {"h", i.featureAxis == "k" ? op.k.hi - op.k.lo : op.n.hi - op.n.lo}};
      std::vector<std::pair<const StaticStep *, StaticAccess>> external{
          {&first, {op.inputs[0], "", "", op.m, op.k}},
          {&first, {op.inputs[1], "", "", op.k, op.n}}};
      if (i.pattern == "matmul_pair") {
        auto &second = last.operation;
        need(i.mode == "independent" || i.mode == "chain",
             "unsupported pair mode");
        need(second.inputs.at(0) == op.output && second.m == op.m &&
                 second.k == op.n,
             "invalid chained data/region relation");
        shape["n"] = second.n.hi - second.n.lo;
        shape["h"] = op.n.hi - op.n.lo;
        external.push_back(
            {&last, {second.inputs.at(1), "", "", second.k, second.n}});
        StaticAccess hidden{op.output, "", "", op.m, op.n};
        need(std::find(a.program.outputs.begin(), a.program.outputs.end(),
                       op.output) == a.program.outputs.end(),
             "hidden intermediate is public output");
        for (auto &s : a.steps)
          if (owner.at(s.id) != e.id) {
            for (auto &r : reads(s.operation))
              need(!overlap(hidden, r),
                   "hidden intermediate has external consumer");
            StaticAccess w{s.operation.output, "", "", s.operation.m,
                           s.operation.n};
            need(!overlap(hidden, w),
                 "hidden intermediate has external writer");
          }
      } else
        need(i.pattern == "matmul_single" && i.mode == "independent",
             "unsupported standalone implementation");
      need(shape == e.regionShape, "forged shape features");
      for (auto &[axis, value] : shape) {
        auto range = i.shapeRange.at(axis);
        need(value >= range.lo && value <= range.hi,
             "out-of-domain calibration query");
      }
      for (auto &[axis, multiple] : i.multiples)
        need(multiple > 0 && shape.at(axis) % multiple == 0,
             "unsupported physical tile");
      for (auto &axis : i.powerOfTwo) {
        auto value = shape.at(axis);
        need(value > 0 && (value & (value - 1)) == 0,
             "unsupported non-power-of-two kernel dimension");
      }
      need(!i.modelId.empty() && !i.trainingIds.empty(),
           "missing calibration identity");
      for (auto &id : i.trainingIds)
        need(std::find(i.excludedIds.begin(), i.excludedIds.end(), id) ==
                 i.excludedIds.end(),
             "training/holdout identity leakage");
      double duration = i.fixedMs + i.perMMs * shape.at("m") +
                        i.perHMs * shape.at("h") +
                        i.perMHMs * shape.at("m") * shape.at("h");
      need(duration >= 0 && near(duration, e.durationMs) &&
               e.provenance.kind == i.provenance.kind &&
               e.provenance.source == i.provenance.source,
           "incorrect opaque wall cost/provenance");
      need(std::isfinite(e.startMs) && e.startMs >= 0 &&
               near(e.endMs - e.startMs, e.durationMs),
           "invalid schedule interval");
      for (auto &d : e.dependsOn)
        need(events.at(d)->endMs <= e.startMs + 1e-8,
             "dependency exceeds start");
      end = std::max(end, e.endMs);
      total += e.durationMs;
      Content expectedReads, expectedWrites = after.at(last.id);
      for (auto &[s, r] : external)
        for (auto &x : covered(r))
          expectedReads[x] = before.at(s->id).at(x);
      Content observedReads, observedWrites;
      for (auto &r : e.reads) {
        auto &al = *allocs.at(r.allocation);
        need(al.buffer == r.buffer && al.memory == i.boundaryMemory &&
                 al.layout == i.layout && al.dtype == i.dtype,
             "incorrect input location/layout/dtype");
        for (auto &x : covered(r)) {
          observedReads[x] = r.version;
          auto value = available.at(al.id).at(x);
          need(value.first == r.version, "incorrect or stale input version");
          if (!value.second.empty())
            need(ancestors.at(e.id).count(value.second),
                 "missing input producer dependency");
        }
      }
      for (auto &w : e.writes) {
        auto &al = *allocs.at(w.allocation);
        need(al.buffer == w.buffer && al.memory == i.boundaryMemory &&
                 al.layout == i.layout && al.dtype == i.dtype,
             "incorrect output location/layout/dtype");
        for (auto &x : covered(w)) {
          need(observedWrites.emplace(x, w.version).second,
               "duplicate output region");
          available[al.id][x] = {w.version, e.id};
        }
      }
      need(observedReads == expectedReads && observedWrites == expectedWrites,
           "region external accesses/version mismatch");
      std::map<std::string, int64_t> workspace;
      for (auto &al : p.allocations)
        if (al.lifetime == "event" && al.createdBy == e.id) {
          workspace[al.memory] = add(workspace[al.memory], al.bytes);
          need(near(al.startMs, e.startMs) && near(al.endMs, e.endMs),
               "workspace lifetime mismatch");
        }
      auto expectedWorkspace = i.workspaceBytes;
      for (auto it = expectedWorkspace.begin(); it != expectedWorkspace.end();)
        if (it->second == 0)
          it = expectedWorkspace.erase(it);
        else
          ++it;
      need(workspace == expectedWorkspace,
           "missing/duplicate opaque workspace");
    }
    need(near(p.totals.makespanMs, end) && near(p.totals.regionWallMs, total) &&
             p.totals.computeMs == 0 && p.totals.copyMs == 0 &&
             p.totals.syncMs == 0 && p.totals.convertMs == 0 &&
             p.totals.transferBytes == 0,
         "opaque wall mislabelled or double counted");
    for (auto &[id, r] : resources) {
      std::vector<Interval> iv;
      for (auto &e : p.events) {
        auto it = e.resourceDemands.find(id);
        iv.push_back({e.startMs, e.endMs,
                      it == e.resourceDemands.end() ? 0 : it->second});
      }
      need(peak(iv) <= r->capacity, "resource oversubscription");
    }
    for (auto &e : p.events)
      for (auto &[id, n] : e.resourceDemands)
        need(resources.count(id) && n > 0, "unknown resource demand");
    if (c.assumptions.overlap == "forbid") {
      std::vector<Interval> iv;
      for (auto &e : p.events)
        iv.push_back({e.startMs, e.endMs, 1});
      need(peak(iv) <= 1, "overlap forbidden");
    }
    std::map<std::string, int64_t> peaks;
    for (auto &m : c.memories) {
      int64_t retained = 0, floor = 0;
      std::vector<Interval> iv;
      std::map<std::string, int64_t> perEvent;
      for (auto &al : p.allocations)
        if (al.memory == m.id) {
          if (al.lifetime == "event") {
            events.at(al.createdBy);
            iv.push_back({al.startMs, al.endMs, al.bytes});
            perEvent[al.createdBy] = add(perEvent[al.createdBy], al.bytes);
          } else {
            retained = add(retained, al.bytes);
            need(near(al.startMs, 0) && near(al.endMs, end),
                 "external buffer lifetime mismatch");
          }
        }
      for (auto &x : perEvent)
        floor = std::max(floor, x.second);
      peaks[m.id] = add(retained, std::max(floor, peak(iv)));
      need(peaks[m.id] <= m.capacityBytes, "capacity exceeded");
    }
    need(peaks == p.totals.peakMemoryBytes, "incorrect memory peak");
    Content exported;
    for (auto &o : p.outputs) {
      auto &al = *allocs.at(o.allocation);
      need(std::find(a.program.outputs.begin(), a.program.outputs.end(),
                     o.buffer) != a.program.outputs.end(),
           "extra public output");
      need(al.buffer == o.buffer &&
               al.memory == c.assumptions.outputLocations.at(o.buffer),
           "incorrect final output location");
      for (auto &x : covered(o)) {
        need(exported.emplace(x, o.version).second, "duplicate public output");
        auto value = available.at(al.id).at(x);
        need(value.first == state.at(x) && o.version == state.at(x),
             "missing/stale final output");
        if (!value.second.empty())
          need(std::find(p.outputDependencies.begin(),
                         p.outputDependencies.end(),
                         value.second) != p.outputDependencies.end(),
               "missing final completion");
      }
    }
    for (auto &id : p.outputDependencies)
      events.at(id);
    for (auto &id : a.program.outputs) {
      auto it = c.assumptions.outputDevices.find(id);
      need(it == c.assumptions.outputDevices.end() || it->second == "cpu",
           "unsupported external output consumer");
      need(p.outputConsumers.at(id) == "cpu", "missing host output consumer");
    }
    Content expected;
    for (auto &id : a.program.outputs)
      for (auto &x : cells.at(id))
        expected[x] = state.at(x);
    need(exported == expected, "missing public output coverage");
  } catch (const std::exception &ex) {
    errors.push_back(std::string("malformed region plan/config: ") + ex.what());
  }
  return errors.empty();
}
