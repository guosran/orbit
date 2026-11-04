#include "Taskflow/StaticMapping.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <tuple>
using namespace orbit::taskflow;
namespace {
struct Interval {
  double start, end;
  int64_t units;
};
int64_t peak(const std::vector<Interval> &v) {
  std::set<double> times;
  for (auto &x : v) {
    times.insert(x.start);
    times.insert(x.end);
  }
  int64_t result = 0;
  for (double t : times) {
    int64_t count = 0;
    for (auto &x : v)
      if (x.start <= t && t < x.end) {
        if (x.units < 0 || x.units > INT64_MAX - count)
          return INT64_MAX;
        count += x.units;
      }
    result = std::max(result, count);
  }
  return result;
}
bool finiteTime(double x) { return std::isfinite(x) && x >= 0; }
} // namespace
bool orbit::taskflow::scheduleStaticPlan(StaticPlan &p, const StaticConfig &c,
                                         std::string &error) {
  std::map<std::string, size_t> ids;
  std::map<std::string, int64_t> capacities;
  for (auto &r : c.resources)
    capacities[r.id] = r.capacity;
  for (size_t i = 0; i < p.events.size(); ++i)
    if (!ids.emplace(p.events[i].id, i).second) {
      error = "duplicate event";
      return false;
    }
  std::set<std::string> scheduled;
  std::map<std::string, std::vector<Interval>> calendars;
  double serialEnd = 0;
  while (scheduled.size() < p.events.size()) {
    StaticEvent *chosen = nullptr;
    for (auto &e : p.events)
      if (!scheduled.count(e.id)) {
        bool ready = true;
        for (auto &dep : e.dependsOn) {
          if (!ids.count(dep)) {
            error = "unknown event dependency";
            return false;
          }
          if (!scheduled.count(dep))
            ready = false;
        }
        if (ready) {
          chosen = &e;
          break;
        }
      }
    if (!chosen) {
      error = "cyclic execution event dependency";
      return false;
    }
    auto &e = *chosen;
    if (!finiteTime(e.durationMs)) {
      error = "unknown or invalid event cost";
      return false;
    }
    double start = 0;
    for (auto &d : e.dependsOn)
      start = std::max(start, p.events[ids[d]].endMs);
    if (c.assumptions.overlap == "forbid")
      start = std::max(start, serialEnd);
    for (auto &[r, n] : e.resourceDemands)
      if (!capacities.count(r) || n <= 0 || n > capacities[r]) {
        error = "event resource capacity exceeded";
        return false;
      }
    for (;;) {
      std::vector<double> conflicts;
      if (e.durationMs > 0)
        for (auto &[rid, n] : e.resourceDemands) {
          auto &calendar = calendars[rid];
          std::set<double> points{start};
          for (auto &x : calendar) {
            if (start < x.start && x.start < start + e.durationMs)
              points.insert(x.start);
            if (start < x.end && x.end < start + e.durationMs)
              points.insert(x.end);
          }
          for (double t : points) {
            int64_t used = 0;
            for (auto &x : calendar)
              if (x.start <= t && t < x.end) {
                if (x.units > INT64_MAX - used) {
                  error = "resource accounting overflow";
                  return false;
                }
                used += x.units;
              }
            if (used > capacities[rid] - n) {
              for (auto &x : calendar)
                if (x.start <= t && t < x.end)
                  conflicts.push_back(x.end);
              break;
            }
          }
        }
      if (conflicts.empty())
        break;
      start = *std::min_element(conflicts.begin(), conflicts.end());
    }
    e.startMs = start;
    e.endMs = start + e.durationMs;
    if (!finiteTime(e.endMs)) {
      error = "schedule time overflow";
      return false;
    }
    scheduled.insert(e.id);
    serialEnd = std::max(serialEnd, e.endMs);
    for (auto &[r, n] : e.resourceDemands)
      calendars[r].push_back({e.startMs, e.endMs, n});
  }
  p.totals = {};
  p.totals.makespanMs = serialEnd;
  for (auto &e : p.events) {
    if (e.kind == "region")
      p.totals.regionWallMs += e.durationMs;
    else if (e.kind == "compute")
      p.totals.computeMs += e.durationMs;
    else if (e.kind == "copy") {
      p.totals.copyMs += e.durationMs;
      if (e.bytes < 0 || e.bytes > INT64_MAX - p.totals.transferBytes) {
        error = "transfer byte overflow";
        return false;
      }
      p.totals.transferBytes += e.bytes;
    } else if (e.kind == "convert")
      p.totals.convertMs += e.durationMs;
    else if (e.kind == "sync")
      p.totals.syncMs += e.durationMs;
    else {
      error = "unknown event kind";
      return false;
    }
  }
  for (auto &m : c.memories) {
    int64_t retained = 0, floor = 0;
    std::vector<Interval> temporary;
    std::map<std::string, int64_t> scratchByEvent;
    for (auto &a : p.allocations)
      if (a.memory == m.id) {
        if (a.bytes < 0) {
          error = "negative allocation";
          return false;
        }
        if (a.lifetime == "event") {
          if (!ids.count(a.createdBy)) {
            error = "unknown temporary owner";
            return false;
          }
          auto &e = p.events[ids[a.createdBy]];
          a.startMs = e.startMs;
          a.endMs = e.endMs;
          temporary.push_back({a.startMs, a.endMs, a.bytes});
          if (a.bytes > INT64_MAX - scratchByEvent[a.createdBy]) {
            error = "scratch byte overflow";
            return false;
          }
          scratchByEvent[a.createdBy] += a.bytes;
          floor = std::max(floor, scratchByEvent[a.createdBy]);
        } else {
          a.startMs = 0;
          a.endMs = serialEnd;
          if (a.bytes > INT64_MAX - retained) {
            error = "allocation byte overflow";
            return false;
          }
          retained += a.bytes;
        }
      }
    auto scratch = std::max(peak(temporary), floor);
    if (scratch == INT64_MAX || scratch > INT64_MAX - retained) {
      error = "memory peak overflow";
      return false;
    }
    auto total = retained + scratch;
    p.totals.peakMemoryBytes[m.id] = total;
    if (total > m.capacityBytes) {
      error = "memory " + m.id + ": peak " + std::to_string(total) +
              " exceeds capacity " + std::to_string(m.capacityBytes);
      return false;
    }
  }
  return true;
}
StaticMappingResult orbit::taskflow::mapStaticProgram(const StaticAnalysis &a,
                                                      const StaticConfig &c) {
  if (!c.regionImplementations.empty())
    return mapStaticRegions(a, c);

  StaticMappingResult result;
  result.status = "unsupported";
  result.budget = c.assumptions.maxAssignments;
  if (c.assumptions.maxAssignments < 1 || c.assumptions.beamWidth < 1) {
    result.reasons = {"invalid mapping budget"};
    return result;
  }
  if (!c.assumptions.programSourceId.empty() &&
      c.assumptions.programSourceId != a.program.sourceId) {
    result.reasons = {"numerical acceptance belongs to another program"};
    return result;
  }
  std::vector<std::vector<const StaticImplementation *>> choices;
  std::vector<std::string> ids;
  uint64_t count = 1;
  for (auto &s : a.steps) {
    std::vector<const StaticImplementation *> legal;
    for (auto &i : c.implementations)
      if (staticImplementationSupports(a, s, i, c))
        legal.push_back(&i);
    if (legal.empty()) {
      result.reasons = {"no supported implementation for " + s.id};
      return result;
    }
    std::sort(legal.begin(), legal.end(), [&](auto l, auto r) {
      double x = staticImplementationCost(s.operation, *l),
             y = staticImplementationCost(s.operation, *r);
      return x == y ? l->id < r->id : x < y;
    });
    if (count > UINT64_MAX / legal.size())
      count = UINT64_MAX;
    else
      count *= legal.size();
    choices.push_back(legal);
    ids.push_back(s.id);
  }
  result.possibleAssignments = count;
  result.exhaustiveAssignments = count <= result.budget;
  using Selection = std::vector<std::string>;
  std::vector<Selection> queue;
  if (result.exhaustiveAssignments) {
    Selection row;
    std::function<void(size_t)> enumerate = [&](size_t i) {
      if (i == choices.size()) {
        queue.push_back(row);
        return;
      }
      for (auto choice : choices[i]) {
        row.push_back(choice->id);
        enumerate(i + 1);
        row.pop_back();
      }
    };
    enumerate(0);
  } else {
    Selection row;
    for (auto &options : choices)
      row.push_back(options[0]->id);
    queue.push_back(row);
    std::set<std::string> devices;
    for (auto &i : c.implementations)
      devices.insert(i.device);
    for (auto &device : devices) {
      row.clear();
      for (auto &options : choices) {
        auto chosen = options[0];
        for (auto i : options)
          if (i->device == device) {
            chosen = i;
            break;
          }
        row.push_back(chosen->id);
      }
      queue.push_back(row);
    }
  }
  std::set<Selection> seen;
  while (!queue.empty() && seen.size() < result.budget) {
    auto batch = std::move(queue);
    queue.clear();
    std::vector<std::pair<double, Selection>> passed;
    std::vector<Selection> failed;
    for (auto &selection : batch) {
      if (seen.size() >= result.budget)
        break;
      if (!seen.insert(selection).second)
        continue;
      StaticAlternative alternative;
      for (size_t i = 0; i < ids.size(); ++i)
        alternative.assignment[ids[i]] = selection[i];
      auto plan = std::make_shared<StaticPlan>();
      std::string error;
      if (!buildStaticGraph(a, c, alternative.assignment, *plan, error)) {
        alternative.status = "unsupported";
        alternative.reason = error;
      } else if (!scheduleStaticPlan(*plan, c, error)) {
        alternative.status = "infeasible";
        alternative.reason = error;
      } else {
        plan->programId = staticFingerprint(printPlan(a.program));
        plan->candidateId = a.program.candidateId;
        plan->configurationId = staticFingerprint(c.original);
        plan->assignment = alternative.assignment;
        plan->steps = a.steps;
        plan->costBoundary =
            "compute_only includes local access; explicit copy/convert/sync "
            "exclude compute; single cold graph";
        plan->planId = staticFingerprint(serializeStaticPlan(*plan, false));
        std::vector<std::string> errors;
        if (!checkStaticPlan(a, c, *plan, errors)) {
          alternative.status = "invalid_plan";
          for (auto &e : errors)
            alternative.reason += e + "; ";
        } else {
          alternative.status = "passed";
          alternative.plan = plan;
          passed.emplace_back(plan->totals.makespanMs, selection);
          if (!result.bestPlan ||
              std::tie(plan->totals.makespanMs, plan->planId) <
                  std::tie(result.bestPlan->totals.makespanMs,
                           result.bestPlan->planId))
            result.bestPlan = plan;
        }
      }
      if (alternative.status != "passed")
        failed.push_back(selection);
      result.alternatives.push_back(std::move(alternative));
    }
    if (!result.exhaustiveAssignments) {
      std::sort(passed.begin(), passed.end());
      std::vector<Selection> frontier;
      for (auto &pair : passed) {
        if (frontier.size() >= size_t(c.assumptions.beamWidth))
          break;
        frontier.push_back(pair.second);
      }
      if (frontier.empty())
        for (auto &s : failed) {
          if (frontier.size() >= size_t(c.assumptions.beamWidth))
            break;
          frontier.push_back(s);
        }
      for (auto &selection : frontier)
        for (size_t i = 0; i < choices.size(); ++i)
          for (auto choice : choices[i]) {
            auto neighbor = selection;
            neighbor[i] = choice->id;
            if (!seen.count(neighbor))
              queue.push_back(std::move(neighbor));
          }
    }
  }
  result.evaluatedAssignments = seen.size();
  result.status = result.bestPlan ? "passed" : "unscored";
  if (!result.bestPlan) {
    std::set<std::string> reasons;
    for (auto &x : result.alternatives)
      reasons.insert(x.reason);
    result.reasons.assign(reasons.begin(), reasons.end());
  }
  return result;
}
