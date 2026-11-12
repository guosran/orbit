#include "Taskflow/ConstraintSearch.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sys/resource.h>
#ifdef ORBIT_HAS_ORTOOLS
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"
#include "ortools/util/sorted_interval_list.h"
#endif
namespace orbit::taskflow {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
} // namespace
const char *constraintSearchStatusName(ConstraintSearchStatus s) {
  switch (s) {
  case ConstraintSearchStatus::Unavailable:
    return "UNAVAILABLE";
  case ConstraintSearchStatus::InvalidInput:
    return "INVALID_INPUT";
  case ConstraintSearchStatus::ModelLimit:
    return "RESOURCE_LIMIT";
  case ConstraintSearchStatus::Unknown:
    return "UNKNOWN";
  case ConstraintSearchStatus::Infeasible:
    return "INFEASIBLE";
  case ConstraintSearchStatus::Feasible:
    return "FEASIBLE";
  case ConstraintSearchStatus::Optimal:
    return "OPTIMAL";
  }
  return "UNKNOWN";
}
bool constraintSearchAvailable() {
#ifdef ORBIT_HAS_ORTOOLS
  return true;
#else
  return false;
#endif
}
ConstraintSearchResult solveConstraintSearch(const ConstraintSearchProblem &p,
                                             const ConstraintSearchOptions &o) {
  auto start = Clock::now();
  ConstraintSearchResult r;
  auto finish = [&](ConstraintSearchStatus s, std::string m) {
    r.status = s;
    r.message = std::move(m);
    r.telemetry.totalMs = elapsed(start);
    rusage u{};
    if (!getrusage(RUSAGE_SELF, &u))
      r.telemetry.peakRssKiB = u.ru_maxrss;
    return r;
  };
#ifndef ORBIT_HAS_ORTOOLS
  (void)p;
  (void)o;
  return finish(ConstraintSearchStatus::Unavailable,
                "Built without ORBIT_ENABLE_ORTOOLS");
#else
  using namespace operations_research;
  using namespace operations_research::sat;
  if (p.taskCount <= 0 || p.tileCount <= 0 || !std::isfinite(o.wallSeconds) ||
      o.wallSeconds <= 0 || o.workers < 1 || o.workers > 16 ||
      o.randomSeed < 0 || o.maxVariables < 1 || o.maxConstraints < 1 ||
      o.maxPatterns < 1 || o.maxTasks < 1 || o.maxTiles < 1 ||
      o.maxInputEdges < 1 || o.maxEdges < 1 || o.maxMemberships < 1 ||
      o.maxIntersectionWork < 1 || o.maxPatternBytes < 0 ||
      o.maxPatternInputs < 0 || o.maxPatternOutputs < 0 ||
      (!p.tileBaseCostNs.empty() &&
       p.tileBaseCostNs.size() != size_t(p.tileCount)))
    return finish(ConstraintSearchStatus::InvalidInput,
                  "Invalid dimensions, limits or tile base costs");
  if (p.taskCount > o.maxTasks || p.tileCount > o.maxTiles ||
      p.edges.size() > size_t(o.maxInputEdges))
    return finish(ConstraintSearchStatus::ModelLimit,
                  "Task, tile or input-edge cap exceeded");
  std::set<std::pair<int64_t, int64_t>> edgePairs;
  for (auto e : p.edges) {
    if (e.source < 0 || e.target < 0 || e.source >= p.taskCount ||
        e.target >= p.taskCount || e.source == e.target)
      return finish(ConstraintSearchStatus::InvalidInput,
                    "Invalid source dependency");
    edgePairs.emplace(e.source, e.target);
  }
  if (edgePairs.size() > size_t(o.maxEdges))
    return finish(ConstraintSearchStatus::ModelLimit,
                  "Distinct dependency cap exceeded");
  std::vector<ConstraintEdge> edges;
  std::vector<std::vector<int64_t>> successors(p.taskCount);
  std::vector<int64_t> indegree(p.taskCount), coveringCounts(p.taskCount);
  int64_t memberships = 0;
  for (auto e : edgePairs) {
    edges.push_back({e.first, e.second});
    successors[e.first].push_back(e.second);
    ++indegree[e.second];
  }
  std::deque<int64_t> ready;
  for (int64_t i = 0; i < p.taskCount; ++i)
    if (!indegree[i])
      ready.push_back(i);
  int64_t visited = 0;
  while (!ready.empty()) {
    auto t = ready.front();
    ready.pop_front();
    ++visited;
    for (auto n : successors[t])
      if (!--indegree[n])
        ready.push_back(n);
  }
  if (visited != p.taskCount)
    return finish(ConstraintSearchStatus::InvalidInput,
                  "Source dependency graph contains a cycle");
  long double nv = static_cast<long double>(p.taskCount) + p.tileCount +
                   p.patterns.size() + edges.size();
  long double nc = static_cast<long double>(p.taskCount) + 1 +
                   2.L * p.patterns.size() + 2.L * edges.size();
  if (p.patterns.size() > size_t(o.maxPatterns) || nv > o.maxVariables ||
      nc > o.maxConstraints)
    return finish(ConstraintSearchStatus::ModelLimit,
                  "Model preflight limit exceeded");
  long double magnitude = 0;
  std::set<int64_t> ids, storages;
  for (auto cost : p.tileBaseCostNs)
    magnitude += std::abs(static_cast<long double>(cost));
  for (const auto &x : p.patterns) {
    if (x.tasks.empty() || x.tileIndex < 0 || x.tileIndex >= p.tileCount ||
        x.bytes < 0 || x.inputCount < 0 || x.outputCount < 0 ||
        !ids.insert(x.id).second)
      return finish(ConstraintSearchStatus::InvalidInput,
                    "Invalid or duplicate pattern");
    if (x.tasks.size() > size_t(o.maxMemberships - memberships))
      return finish(ConstraintSearchStatus::ModelLimit,
                    "Pattern membership cap exceeded");
    memberships += x.tasks.size();
    if (x.exclusiveStorageIds.size() > size_t(o.maxMemberships - memberships))
      return finish(ConstraintSearchStatus::ModelLimit,
                    "Exclusive-storage membership cap exceeded");
    memberships += x.exclusiveStorageIds.size();
    magnitude += std::abs(static_cast<long double>(x.costNs));
    nc += x.tasks.size() - 1;
    std::set<int64_t> members;
    for (auto t : x.tasks)
      if (t < 0 || t >= p.taskCount || !members.insert(t).second)
        return finish(ConstraintSearchStatus::InvalidInput,
                      "Invalid or repeated pattern task");
    for (auto t : x.tasks)
      ++coveringCounts[t];
    std::set<int64_t> ownStorages;
    for (auto s : x.exclusiveStorageIds) {
      if (s < 0 || !ownStorages.insert(s).second)
        return finish(ConstraintSearchStatus::InvalidInput,
                      "Invalid exclusive storage id");
      storages.insert(s);
    }
    if (elapsed(start) >= o.wallSeconds * 1000) {
      r.telemetry.budgetExhausted = true;
      r.telemetry.modelBuildMs = elapsed(start);
      return finish(ConstraintSearchStatus::Unknown,
                    "Input validation deadline");
    }
  }
  if (magnitude > std::numeric_limits<int64_t>::max() / 4)
    return finish(ConstraintSearchStatus::InvalidInput,
                  "Unsafe objective coefficient range");
  nc += storages.size();
  if (nc > o.maxConstraints)
    return finish(ConstraintSearchStatus::ModelLimit,
                  "Constraint preflight limit exceeded");
  int64_t intersectionWork = 0;
  for (auto e : edges)
    for (auto t : {e.source, e.target}) {
      if (coveringCounts[t] > o.maxIntersectionWork - intersectionWork)
        return finish(ConstraintSearchStatus::ModelLimit,
                      "Dependency intersection-work cap exceeded");
      intersectionWork += coveringCounts[t];
    }
  std::set<int64_t> hinted;
  for (auto i : o.hintPatternIndices)
    if (i < 0 || size_t(i) >= p.patterns.size() || !hinted.insert(i).second)
      return finish(ConstraintSearchStatus::InvalidInput, "Invalid hint");
  CpModelBuilder b;
  std::vector<BoolVar> selected, tiles;
  std::vector<IntVar> ranks;
  std::vector<std::vector<int64_t>> covering(p.taskCount);
  for (size_t i = 0; i < p.patterns.size(); ++i)
    selected.push_back(b.NewBoolVar());
  for (int64_t i = 0; i < p.tileCount; ++i)
    tiles.push_back(b.NewBoolVar());
  b.AddExactlyOne(tiles);
  for (int64_t i = 0; i < p.taskCount; ++i)
    ranks.push_back(b.NewIntVar(Domain(0, p.taskCount - 1)));
  LinearExpr objective;
  for (size_t i = 0; i < p.tileBaseCostNs.size(); ++i)
    objective += LinearExpr(tiles[i]) * p.tileBaseCostNs[i];
  auto expired = [&]() { return elapsed(start) >= o.wallSeconds * 1000; };
  for (size_t i = 0; i < p.patterns.size(); ++i) {
    const auto &x = p.patterns[i];
    b.AddImplication(selected[i], tiles[x.tileIndex]);
    if (x.bytes > o.maxPatternBytes || x.inputCount > o.maxPatternInputs ||
        x.outputCount > o.maxPatternOutputs)
      b.AddEquality(selected[i], 0);
    auto anchor = x.tasks.front();
    for (auto t : x.tasks) {
      covering[t].push_back(i);
      if (t != anchor)
        b.AddEquality(ranks[t], ranks[anchor]).OnlyEnforceIf(selected[i]);
    }
    objective += LinearExpr(selected[i]) * x.costNs;
    if (expired()) {
      r.telemetry.modelBuildMs = elapsed(start);
      r.telemetry.budgetExhausted = true;
      return finish(ConstraintSearchStatus::Unknown, "Pattern model deadline");
    }
  }
  for (const auto &indices : covering) {
    std::vector<BoolVar> choices;
    for (auto i : indices)
      choices.push_back(selected[i]);
    b.AddExactlyOne(choices);
  }
  std::map<int64_t, std::vector<BoolVar>> storageUsers;
  for (size_t i = 0; i < p.patterns.size(); ++i)
    for (auto s : p.patterns[i].exclusiveStorageIds)
      storageUsers[s].push_back(selected[i]);
  for (auto &entry : storageUsers)
    b.AddAtMostOne(entry.second);
  for (auto e : edges) {
    std::vector<int64_t> common;
    const auto &from = covering[e.source], &to = covering[e.target];
    std::set_intersection(from.begin(), from.end(), to.begin(), to.end(),
                          std::back_inserter(common));
    if (common.empty())
      b.AddGreaterOrEqual(ranks[e.target], LinearExpr(ranks[e.source]) + 1);
    else if (common.size() == 1)
      b.AddGreaterOrEqual(ranks[e.target],
                          LinearExpr(ranks[e.source]) + 1 -
                              LinearExpr(selected[common.front()]));
    else {
      BoolVar separate = b.NewBoolVar();
      LinearExpr same;
      for (auto i : common)
        same += selected[i];
      b.AddEquality(same + separate, 1);
      b.AddGreaterOrEqual(ranks[e.target],
                          LinearExpr(ranks[e.source]) + separate);
    }
    if (expired()) {
      r.telemetry.modelBuildMs = elapsed(start);
      r.telemetry.budgetExhausted = true;
      return finish(ConstraintSearchStatus::Unknown,
                    "Dependency model deadline");
    }
  }
  for (auto i : o.hintPatternIndices)
    b.AddHint(selected[i], 1);
  b.Minimize(objective);
  const CpModelProto proto = b.Build();
  r.telemetry.variables = proto.variables_size();
  r.telemetry.constraints = proto.constraints_size();
  r.telemetry.modelBuildMs = elapsed(start);
  if (r.telemetry.variables > o.maxVariables ||
      r.telemetry.constraints > o.maxConstraints)
    return finish(ConstraintSearchStatus::ModelLimit,
                  "Constructed model limit exceeded");
  double remaining = o.wallSeconds - elapsed(start) / 1000;
  if (remaining <= 0) {
    r.telemetry.budgetExhausted = true;
    return finish(ConstraintSearchStatus::Unknown, "Deadline before solve");
  }
  SatParameters params;
  params.set_max_time_in_seconds(remaining);
  params.set_num_search_workers(o.workers);
  params.set_random_seed(o.randomSeed);
  params.set_log_search_progress(false);
  Model model;
  model.Add(NewSatParameters(params));
  auto solveStart = Clock::now();
  auto response = SolveCpModel(proto, &model);
  r.telemetry.solveMs = elapsed(solveStart);
  r.telemetry.conflicts = response.num_conflicts();
  r.telemetry.branches = response.num_branches();
  if (response.status() == CpSolverStatus::MODEL_INVALID)
    return finish(ConstraintSearchStatus::InvalidInput,
                  response.solution_info());
  if (response.status() == CpSolverStatus::INFEASIBLE)
    return finish(
        ConstraintSearchStatus::Infeasible,
        "No cover satisfies common tile, resources and quotient ordering");
  if (response.status() != CpSolverStatus::FEASIBLE &&
      response.status() != CpSolverStatus::OPTIMAL) {
    r.telemetry.budgetExhausted = expired();
    return finish(ConstraintSearchStatus::Unknown, "No incumbent returned");
  }
  r.telemetry.hasObjective = true;
  r.telemetry.bestBoundNs = response.best_objective_bound();
  r.telemetry.budgetExhausted =
      response.status() == CpSolverStatus::FEASIBLE && expired();
  for (size_t i = 0; i < selected.size(); ++i)
    if (SolutionBooleanValue(response, selected[i]))
      r.selectedPatternIndices.push_back(i);
  for (size_t i = 0; i < tiles.size(); ++i)
    if (SolutionBooleanValue(response, tiles[i]))
      r.selectedTileIndex = i;
  if (!p.tileBaseCostNs.empty())
    r.telemetry.objectiveNs = p.tileBaseCostNs[r.selectedTileIndex];
  for (auto i : r.selectedPatternIndices)
    r.telemetry.objectiveNs += p.patterns[i].costNs;
  if (response.status() == CpSolverStatus::OPTIMAL)
    r.telemetry.bestBoundNs = double(r.telemetry.objectiveNs);
  r.telemetry.relativeGap =
      std::max(0., double(r.telemetry.objectiveNs) - r.telemetry.bestBoundNs) /
      std::max(1., std::abs(double(r.telemetry.objectiveNs)));
  return finish(response.status() == CpSolverStatus::OPTIMAL
                    ? ConstraintSearchStatus::Optimal
                    : ConstraintSearchStatus::Feasible,
                "Optimality applies to supplied additive surrogate; backend "
                "verification required");
#endif
}
} // namespace orbit::taskflow
