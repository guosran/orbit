#include "Taskflow/StaticMapping.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <tuple>
using namespace orbit::taskflow;
namespace {
bool fail(std::string &e, const std::string &s) {
  e = s;
  return false;
}
bool overlap(const StaticAccess &a, const StaticAccess &b) {
  return a.buffer == b.buffer && a.m.lo < b.m.hi && b.m.lo < a.m.hi &&
         a.n.lo < b.n.hi && b.n.lo < a.n.hi;
}
bool same(const Range &a, const Range &b) {
  return a.lo == b.lo && a.hi == b.hi;
}
void clean(std::vector<std::string> &a) {
  std::sort(a.begin(), a.end());
  a.erase(std::unique(a.begin(), a.end()), a.end());
}
struct Unit {
  const StaticStep *fill, *matmul;
};
const StaticRegionImplementation *implementation(const StaticConfig &c,
                                                 const std::string &id) {
  for (auto &i : c.regionImplementations)
    if (i.id == id)
      return &i;
  return nullptr;
}
bool scoped(const StaticRegionMatch &m, const StaticRegionImplementation &i) {
  for (auto &bound : i.shapeRange) {
    auto x = m.shape.find(bound.first);
    if (x == m.shape.end() || x->second < bound.second.lo ||
        x->second > bound.second.hi)
      return false;
  }
  for (auto &multiple : i.multiples) {
    auto x = m.shape.find(multiple.first);
    if (x == m.shape.end() || multiple.second <= 0 ||
        x->second % multiple.second)
      return false;
  }
  for (auto &axis : i.powerOfTwo) {
    auto x = m.shape.find(axis);
    if (x == m.shape.end() || x->second <= 0 || (x->second & (x->second - 1)))
      return false;
  }
  return true;
}
} // namespace
double orbit::taskflow::staticRegionCost(const StaticRegionMatch &m,
                                         const StaticRegionImplementation &i) {
  double rows = m.shape.at("m"), h = m.shape.at("h");
  return i.fixedMs + i.perMMs * rows + i.perHMs * h + i.perMHMs * rows * h;
}
bool orbit::taskflow::matchStaticRegions(
    const StaticAnalysis &a, const StaticConfig &c,
    std::vector<StaticRegionMatch> &matches,
    std::vector<std::string> &reasons) {
  matches.clear();
  std::map<std::string, const Buffer *> buffers;
  for (auto &b : a.program.buffers)
    buffers[b.id] = &b;
  if (c.assumptions.executionState != "warm_contexts_weights") {
    reasons.push_back("opaque region requires explicit warm_contexts_weights "
                      "execution state");
    return false;
  }
  std::vector<Unit> units;
  for (size_t index = 1; index < a.steps.size(); ++index) {
    auto &mm = a.steps[index];
    auto &fill = a.steps[index - 1];
    auto &op = mm.operation;
    auto &zero = fill.operation;
    if (op.kind != "matmul" || !op.accumulate || zero.kind != "fill" ||
        mm.taskId != fill.taskId ||
        mm.operationIndex != fill.operationIndex + 1 ||
        zero.output != op.output || !same(zero.m, op.m) || !same(zero.n, op.n))
      continue;
    if (op.inputs.size() != 2 || op.output == op.inputs[0] ||
        op.output == op.inputs[1] || op.k.lo != 0 ||
        op.k.hi != buffers.at(op.inputs[0])->cols ||
        buffers.at(op.inputs[1])->role != "input")
      continue;
    units.push_back({&fill, &mm});
  }
  auto accept = [&](StaticRegionMatch m, const StaticRegionImplementation &i) {
    if (!scoped(m, i)) {
      reasons.push_back(i.id +
                        ": out-of-domain shape or illegal physical multiple");
      return;
    }
    double cost = staticRegionCost(m, i);
    if (!std::isfinite(cost) || cost <= 0) {
      reasons.push_back(i.id + ": unknown/nonpositive model evaluation");
      return;
    }
    m.implementationId = i.id;
    m.id = "region:" + m.coveredSteps.front() + ":" + i.id;
    matches.push_back(std::move(m));
  };
  for (auto &i : c.regionImplementations) {
    if (i.preparedState != c.assumptions.executionState ||
        std::find(c.assumptions.allowedNumericalContracts.begin(),
                  c.assumptions.allowedNumericalContracts.end(),
                  i.numericalContract) ==
            c.assumptions.allowedNumericalContracts.end()) {
      reasons.push_back(i.id +
                        ": precision or prepared-state permission missing");
      continue;
    }
    if (i.pattern == "matmul_single")
      for (auto &u : units) {
        auto &op = u.matmul->operation;
        StaticRegionMatch m;
        m.coveredSteps = {u.fill->id, u.matmul->id};
        m.shape = {{"m", op.m.hi - op.m.lo},
                   {"k", op.k.hi - op.k.lo},
                   {"n", op.n.hi - op.n.lo}};
        m.shape["h"] = m.shape[i.featureAxis];
        m.reads = staticReads(op);
        m.reads.pop_back();
        m.writes = staticWrites(op);
        accept(m, i);
      }
    if (i.pattern == "matmul_pair")
      for (auto &u : units)
        for (auto &v : units) {
          auto &first = u.matmul->operation;
          auto &second = v.matmul->operation;
          if (u.matmul == v.matmul || first.output != second.inputs[0] ||
              first.output == second.output || !same(first.m, second.m) ||
              !same(first.n, second.k))
            continue;
          auto left = std::find_if(a.steps.begin(), a.steps.end(),
                                   [&](auto &s) { return &s == u.matmul; });
          auto right = std::find_if(a.steps.begin(), a.steps.end(),
                                    [&](auto &s) { return &s == v.fill; });
          if (left >= right)
            continue;
          StaticRegionMatch m;
          m.coveredSteps = {u.fill->id, u.matmul->id, v.fill->id, v.matmul->id};
          std::set<std::string> covered(m.coveredSteps.begin(),
                                        m.coveredSteps.end());
          auto middle = staticWrites(first).front();
          bool external =
              std::find(a.program.outputs.begin(), a.program.outputs.end(),
                        first.output) != a.program.outputs.end();
          for (auto &s : a.steps)
            if (!covered.count(s.id)) {
              auto uses = staticReads(s.operation);
              auto writes = staticWrites(s.operation);
              uses.insert(uses.end(), writes.begin(), writes.end());
              for (auto &r : uses)
                external |= overlap(r, middle);
            }
          if (external) {
            reasons.push_back(
                i.id +
                ": intermediate has external consumer/writer/public output");
            continue;
          }
          m.shape = {{"m", first.m.hi - first.m.lo},
                     {"k", first.k.hi - first.k.lo},
                     {"h", first.n.hi - first.n.lo},
                     {"n", second.n.hi - second.n.lo}};
          m.reads = staticReads(first);
          m.reads.pop_back();
          auto secondReads = staticReads(second);
          m.reads.push_back(secondReads[1]);
          m.writes = staticWrites(second);
          accept(m, i);
        }
  }
  clean(reasons);
  return !matches.empty();
}
bool orbit::taskflow::buildStaticRegionGraph(
    const StaticAnalysis &a, const StaticConfig &c,
    const std::vector<StaticRegionMatch> &selected, StaticPlan &p,
    std::string &e) {
  p = StaticPlan{};
  p.steps = a.steps;
  p.programId = staticFingerprint(printPlan(a.program));
  p.candidateId = a.program.candidateId;
  p.configurationId = staticFingerprint(c.original);
  p.initialState = "warm_contexts_weights";
  p.costBoundary =
      "opaque_warm_host_to_host: activation conversion, staging, "
      "synchronization, submissions, intermediate handling and final "
      "conversion included; no duplicate internal communication";
  std::map<std::string, size_t> owners;
  std::map<std::string, const StaticStep *> steps;
  for (auto &s : a.steps)
    steps[s.id] = &s;
  for (size_t j = 0; j < selected.size(); ++j)
    for (auto &id : selected[j].coveredSteps) {
      if (!steps.count(id) || !owners.emplace(id, j).second)
        return fail(e, "region exact-cover duplicate or unknown operation");
      p.assignment[id] = selected[j].implementationId;
    }
  if (owners.size() != a.steps.size())
    return fail(e, "region exact-cover omits computation");
  std::vector<StaticRegionMatch> valid;
  std::vector<std::string> why;
  matchStaticRegions(a, c, valid, why);
  for (auto &m : selected) {
    auto it = std::find_if(valid.begin(), valid.end(), [&](auto &v) {
      return m.id == v.id && m.implementationId == v.implementationId &&
             m.coveredSteps == v.coveredSteps && m.shape == v.shape;
    });
    if (it == valid.end())
      return fail(e, "selected region unsupported or out of model domain");
  }
  // One cold input location is not silently upgraded into the measured warm
  // state. The configuration explicitly selects pre-created contexts/weights;
  // f32 activations remain at host boundary and conversion is paid per event.
  std::string memory;
  for (auto &m : selected) {
    auto *i = implementation(c, m.implementationId);
    if (!i)
      return fail(e, "unknown region implementation");
    if (memory.empty())
      memory = i->boundaryMemory;
    else if (memory != i->boundaryMemory)
      return fail(e, "mixed opaque boundary memories require an explicit "
                     "external transport model");
  }
  std::map<std::string, size_t> allocationIndex;
  for (auto &b : a.program.buffers) {
    if (b.rows <= 0 || b.cols <= 0 || b.rows > INT64_MAX / 4 / b.cols)
      return fail(e, "region storage byte overflow");
    if (b.role == "input") {
      auto location = c.assumptions.initialLocations.find(b.id);
      if (location == c.assumptions.initialLocations.end() ||
          location->second != memory)
        return fail(e, "opaque input not at measured host boundary");
    }
    StaticAllocation al;
    al.id = "region-buffer:" + b.id;
    al.buffer = b.id;
    al.memory = memory;
    al.layout = "row_major";
    al.dtype = "f32";
    al.m = {0, b.rows};
    al.n = {0, b.cols};
    al.bytes = b.rows * b.cols * 4;
    al.lifetime = "graph";
    al.initial = b.role == "input";
    allocationIndex[b.id] = p.allocations.size();
    p.allocations.push_back(al);
  }
  // Partition at semantic access boundaries to preserve sliced regions and
  // exact producer versions without converting task ordering into copy bytes.
  using Cell = std::tuple<std::string, int64_t, int64_t, int64_t, int64_t>;
  std::map<std::string, std::pair<std::set<int64_t>, std::set<int64_t>>> bounds;
  for (auto &b : a.program.buffers)
    bounds[b.id] = {{0, b.rows}, {0, b.cols}};
  for (auto &s : a.steps) {
    auto accesses = staticReads(s.operation);
    auto w = staticWrites(s.operation);
    accesses.insert(accesses.end(), w.begin(), w.end());
    for (auto &r : accesses) {
      bounds[r.buffer].first.insert(r.m.lo);
      bounds[r.buffer].first.insert(r.m.hi);
      bounds[r.buffer].second.insert(r.n.lo);
      bounds[r.buffer].second.insert(r.n.hi);
    }
  }
  std::map<std::string, std::vector<Cell>> cells;
  for (auto &entry : bounds) {
    std::vector<int64_t> x(entry.second.first.begin(),
                           entry.second.first.end()),
        y(entry.second.second.begin(), entry.second.second.end());
    if (x.size() - 1 > uint64_t(c.assumptions.maxRegionCells) / (y.size() - 1))
      return fail(e, "region cell budget exceeded");
    p.regionCellCount += (x.size() - 1) * (y.size() - 1);
    if (p.regionCellCount > c.assumptions.maxRegionCells)
      return fail(e, "region cell budget exceeded");
    for (size_t i = 1; i < x.size(); ++i)
      for (size_t j = 1; j < y.size(); ++j)
        cells[entry.first].push_back(
            {entry.first, x[i - 1], x[i], y[j - 1], y[j]});
  }
  auto covered = [&](const StaticAccess &r) {
    std::vector<Cell> out;
    for (auto &v : cells[r.buffer])
      if (r.m.lo <= std::get<1>(v) && std::get<2>(v) <= r.m.hi &&
          r.n.lo <= std::get<3>(v) && std::get<4>(v) <= r.n.hi)
        out.push_back(v);
    return out;
  };
  std::map<Cell, std::string> version;
  std::map<std::string, std::vector<StaticAccess>> expectedReads,
      expectedWrites;
  auto access = [&](const Cell &v, const std::string &ver) {
    return StaticAccess{std::get<0>(v),
                        "region-buffer:" + std::get<0>(v),
                        ver,
                        {std::get<1>(v), std::get<2>(v)},
                        {std::get<3>(v), std::get<4>(v)}};
  };
  for (auto &b : a.program.buffers)
    if (b.role == "input")
      for (auto &cell : cells[b.id])
        version[cell] = "initial:" + b.id;
  for (auto &s : a.steps) {
    for (auto &r : staticReads(s.operation))
      for (auto &cell : covered(r)) {
        if (!version.count(cell))
          return fail(e, "semantic read is uninitialized");
        expectedReads[s.id].push_back(access(cell, version[cell]));
      }
    for (auto &w : staticWrites(s.operation))
      for (auto &cell : covered(w)) {
        version[cell] = s.id;
        expectedWrites[s.id].push_back(access(cell, s.id));
      }
  }
  for (size_t j = 0; j < selected.size(); ++j) {
    auto &m = selected[j];
    auto &i = *implementation(c, m.implementationId);
    StaticEvent event;
    event.id = m.id;
    event.kind = "region";
    event.implementationId = i.id;
    event.device = i.device;
    event.coveredSteps = m.coveredSteps;
    event.regionPattern = i.pattern;
    event.regionMode = i.mode;
    event.modelId = i.modelId;
    event.preparedState = i.preparedState;
    event.regionShape = m.shape;
    event.resourceDemands = i.resources;
    event.durationMs = staticRegionCost(m, i);
    event.provenance = i.provenance;
    event.costBoundary = "opaque_warm_host_to_host";
    std::set<std::string> inside(m.coveredSteps.begin(), m.coveredSteps.end());
    for (auto &sid : m.coveredSteps) {
      for (auto &dep : steps.at(sid)->dependsOn)
        if (!inside.count(dep))
          event.dependsOn.push_back(selected.at(owners.at(dep)).id);
      for (auto &r : expectedReads[sid])
        if (!inside.count(r.version)) {
          if (r.version.rfind("initial:", 0) != 0)
            event.dependsOn.push_back(selected.at(owners.at(r.version)).id);
          event.reads.push_back(r);
        }
    }
    // Deduplicate identical input cells reused internally, retaining each final
    // externally visible write only. Private intermediate storage is reserved.
    auto order = [](const StaticAccess &x, const StaticAccess &y) {
      return std::tie(x.buffer, x.m.lo, x.m.hi, x.n.lo, x.n.hi, x.version) <
             std::tie(y.buffer, y.m.lo, y.m.hi, y.n.lo, y.n.hi, y.version);
    };
    std::sort(event.reads.begin(), event.reads.end(), order);
    event.reads.erase(std::unique(event.reads.begin(), event.reads.end(),
                                  [&](auto &x, auto &y) {
                                    return !order(x, y) && !order(y, x);
                                  }),
                      event.reads.end());
    const auto &last = m.coveredSteps.back();
    event.writes = expectedWrites[last];
    for (auto &write : event.writes) {
      auto &al = p.allocations[allocationIndex.at(write.buffer)];
      if (al.createdBy.empty())
        al.createdBy = event.id;
      al.lastUsers.push_back(event.id);
    }
    for (auto &r : event.reads)
      p.allocations[allocationIndex.at(r.buffer)].lastUsers.push_back(event.id);
    for (auto &workspace : i.workspaceBytes) {
      StaticAllocation al;
      al.id = "workspace:" + event.id + ":" + workspace.first;
      al.memory = workspace.first;
      al.layout = "opaque_workspace";
      al.dtype = "u8";
      al.bytes = workspace.second;
      al.createdBy = event.id;
      al.lastUsers = {event.id};
      al.lifetime = "event";
      p.allocations.push_back(al);
    }
    clean(event.dependsOn);
    p.events.push_back(std::move(event));
  }
  // Preserve RAW/WAR/WAW ordering across covers, not only explicit task edges.
  for (size_t left = 0; left < a.steps.size(); ++left)
    for (size_t right = left + 1; right < a.steps.size(); ++right) {
      auto &x = a.steps[left];
      auto &y = a.steps[right];
      size_t ox = owners[x.id], oy = owners[y.id];
      if (ox == oy)
        continue;
      bool hazard = false;
      auto xr = staticReads(x.operation), xw = staticWrites(x.operation),
           yr = staticReads(y.operation), yw = staticWrites(y.operation);
      for (auto &w : xw) {
        for (auto &r : yr)
          hazard |= overlap(w, r);
        for (auto &v : yw)
          hazard |= overlap(w, v);
      }
      for (auto &r : xr)
        for (auto &w : yw)
          hazard |= overlap(r, w);
      if (hazard)
        p.events[oy].dependsOn.push_back(p.events[ox].id);
    }
  for (auto &event : p.events)
    clean(event.dependsOn);
  for (auto &bid : a.program.outputs) {
    auto location = c.assumptions.outputLocations.find(bid);
    if (location == c.assumptions.outputLocations.end() ||
        location->second != memory)
      return fail(e, "opaque output not at measured host boundary");
    auto device = c.assumptions.outputDevices.find(bid);
    if (device != c.assumptions.outputDevices.end() && device->second != "cpu")
      return fail(e, "opaque output consumer must be CPU at host boundary");
    p.outputConsumers[bid] = "cpu";
    for (auto &cell : cells[bid]) {
      auto final = version.find(cell);
      if (final == version.end() || !owners.count(final->second))
        return fail(e, "missing output production");
      auto eventId = selected[owners[final->second]].id;
      bool exposed = false;
      for (auto &w : p.events[owners[final->second]].writes)
        if (w.buffer == bid && w.m.lo == std::get<1>(cell) &&
            w.m.hi == std::get<2>(cell) && w.n.lo == std::get<3>(cell) &&
            w.n.hi == std::get<4>(cell))
          exposed = true;
      if (!exposed)
        return fail(e, "public output hidden by opaque implementation");
      p.outputs.push_back(access(cell, final->second));
      p.outputDependencies.push_back(eventId);
    }
  }
  clean(p.outputDependencies);
  for (auto &al : p.allocations)
    clean(al.lastUsers);
  return true;
}
StaticMappingResult orbit::taskflow::mapStaticRegions(const StaticAnalysis &a,
                                                      const StaticConfig &c) {
  StaticMappingResult result;
  result.regionExactCover = true;
  result.status = "unscored";
  result.budget = c.assumptions.maxAssignments;
  if (!c.assumptions.programSourceId.empty() &&
      c.assumptions.programSourceId != a.program.sourceId) {
    result.reasons = {"region numerical acceptance belongs to another source"};
    return result;
  }
  std::vector<StaticRegionMatch> matches;
  matchStaticRegions(a, c, matches, result.reasons);
  std::map<std::string, std::vector<size_t>> covering;
  for (size_t i = 0; i < matches.size(); ++i)
    for (auto &id : matches[i].coveredSteps)
      covering[id].push_back(i);
  for (auto &s : a.steps)
    if (covering[s.id].empty()) {
      result.reasons.push_back(
          "no calibrated region covers " + s.id +
          "; unknown/out-of-domain cost or unavailable implementation");
      return result;
    }
  std::set<std::string> covered;
  std::vector<size_t> selection;
  bool truncated = false;
  uint64_t visited = 0,
           nodeBudget = std::max<uint64_t>(
               1024,
               std::min<uint64_t>(
                   100000, std::min<uint64_t>(result.budget, 100000) * 128));
  std::function<void()> enumerate = [&]() {
    if (++visited > nodeBudget) {
      truncated = true;
      result.reasons.push_back(
          "exact-cover partial-state exploration budget exhausted");
      return;
    }
    if (result.evaluatedAssignments >= result.budget) {
      truncated = true;
      return;
    }
    const StaticStep *next = nullptr;
    for (auto &s : a.steps)
      if (!covered.count(s.id)) {
        next = &s;
        break;
      }
    if (!next) {
      ++result.evaluatedAssignments;
      StaticAlternative alt;
      std::vector<StaticRegionMatch> chosen;
      for (auto index : selection) {
        chosen.push_back(matches[index]);
        for (auto &id : matches[index].coveredSteps)
          alt.assignment[id] = matches[index].implementationId;
      }
      auto plan = std::make_shared<StaticPlan>();
      std::string error;
      if (!buildStaticRegionGraph(a, c, chosen, *plan, error)) {
        alt.status = "unsupported";
        alt.reason = error;
      } else if (!scheduleStaticPlan(*plan, c, error)) {
        alt.status = "infeasible";
        alt.reason = error;
      } else {
        plan->planId = staticFingerprint(serializeStaticPlan(*plan, false));
        std::vector<std::string> errors;
        if (!checkStaticPlan(a, c, *plan, errors)) {
          alt.status = "invalid_plan";
          for (auto &e : errors)
            alt.reason += e + "; ";
        } else {
          alt.status = "passed";
          alt.plan = plan;
          if (!result.bestPlan ||
              std::tie(plan->totals.makespanMs, plan->planId) <
                  std::tie(result.bestPlan->totals.makespanMs,
                           result.bestPlan->planId))
            result.bestPlan = plan;
        }
      }
      result.alternatives.push_back(std::move(alt));
      return;
    }
    for (auto index : covering[next->id]) {
      auto &m = matches[index];
      if (std::any_of(m.coveredSteps.begin(), m.coveredSteps.end(),
                      [&](auto &id) { return covered.count(id); }))
        continue;
      for (auto &id : m.coveredSteps)
        covered.insert(id);
      selection.push_back(index);
      enumerate();
      selection.pop_back();
      for (auto &id : m.coveredSteps)
        covered.erase(id);
      if (truncated)
        return;
    }
  };
  enumerate();
  result.exhaustiveAssignments = !truncated;
  result.possibleAssignments = result.evaluatedAssignments;
  result.status = result.bestPlan ? "passed" : "unscored";
  if (!result.bestPlan)
    for (auto &alt : result.alternatives)
      result.reasons.push_back(alt.reason);
  clean(result.reasons);
  return result;
}
