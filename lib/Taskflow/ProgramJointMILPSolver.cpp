#include "Taskflow/ProgramJointMILP.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <sys/resource.h>
#ifdef ORBIT_HAS_SCIP
#include "scip/cons_linear.h"
#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#endif

using namespace orbit::taskflow;
bool orbit::taskflow::programJointMILPAvailable() {
#ifdef ORBIT_HAS_SCIP
  return true;
#else
  return false;
#endif
}
#ifdef ORBIT_HAS_SCIP
namespace {
using Clock = std::chrono::steady_clock;
using Var = SCIP_VAR *;
struct Expr {
  std::map<Var, double> terms;
  Expr() = default;
  Expr(Var v, double c = 1) {
    if (v && c)
      terms[v] = c;
  }
  Expr &add(Var v, double c = 1) {
    if (v && c)
      terms[v] += c;
    return *this;
  }
};
// Every API failure is carried explicitly: LLVM builds may disable exceptions.
struct MILP {
  SCIP *scip = nullptr;
  const ConstraintSearchOptions &options;
  ConstraintSearchTelemetry &telemetry;
  Clock::time_point start;
  std::vector<Var> owned;
  // Complete primal start, including reified comparisons and max selectors.
  // These values seed SCIP; they never tighten a variable's feasible domain.
  std::map<Var, int64_t> warmValues;
  ConstraintSearchStatus failure = ConstraintSearchStatus::Unknown;
  std::string error;
  double horizon;
  MILP(const ConstraintSearchOptions &o, ConstraintSearchTelemetry &t,
       Clock::time_point s, int64_t h)
      : options(o), telemetry(t), start(s), horizon(h) {
    if (!api(SCIPcreate(&scip)) || !api(SCIPincludeDefaultPlugins(scip)) ||
        !api(SCIPcreateProbBasic(scip, "orbit-program-joint")))
      return;
    api(SCIPsetIntParam(scip, "display/verblevel", 0));
    api(SCIPsetRealParam(scip, "numerics/feastol", 1e-9));
    api(SCIPsetRealParam(scip, "limits/gap", 0));
    api(SCIPsetRealParam(scip, "limits/absgap", 0));
    api(SCIPsetIntParam(scip, "randomization/randomseedshift", o.randomSeed));
    api(SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE));
  }
  ~MILP() {
    if (scip) {
      for (auto v : owned)
        SCIPreleaseVar(scip, &v);
      SCIPfree(&scip);
    }
  }
  bool fail(ConstraintSearchStatus s, const std::string &e) {
    if (error.empty()) {
      failure = s;
      error = e;
    }
    return false;
  }
  bool api(SCIP_RETCODE rc) {
    return rc == SCIP_OKAY || fail(ConstraintSearchStatus::Unknown,
                                   "SCIP API/numerical failure (code " +
                                       std::to_string(int(rc)) + ")");
  }
  bool check() {
    if (!error.empty())
      return false;
    if (std::chrono::duration<double>(Clock::now() - start).count() >=
        options.wallSeconds) {
      telemetry.budgetExhausted = true;
      return fail(ConstraintSearchStatus::Unknown,
                  "total wall budget exhausted during MILP construction");
    }
    return true;
  }
  Var var(bool binary = false, double objective = 0) {
    if (!check())
      return nullptr;
    if (telemetry.variables >= options.maxVariables) {
      fail(ConstraintSearchStatus::ModelLimit, "MILP variable cap");
      return nullptr;
    }
    Var v = nullptr;
    if (!api(SCIPcreateVarBasic(
            scip, &v, "", 0, binary ? 1 : horizon, objective,
            binary ? SCIP_VARTYPE_BINARY : SCIP_VARTYPE_INTEGER)))
      return nullptr;
    owned.push_back(v);
    if (!api(SCIPaddVar(scip, v)))
      return nullptr;
    ++telemetry.variables;
    warmValues.emplace(v, 0);
    return v;
  }
  int64_t hint(Var variable) const {
    auto found = warmValues.find(variable);
    return found == warmValues.end() ? 0 : found->second;
  }
  void hint(Var variable, int64_t value) {
    if (!variable || !error.empty())
      return;
    if (value < 0 || value > horizon ||
        (SCIPvarGetType(variable) == SCIP_VARTYPE_BINARY && value > 1)) {
      fail(ConstraintSearchStatus::Unknown,
           "checked baseline hint outside MILP domain");
      return;
    }
    warmValues[variable] = value;
  }
  double hint(const Expr &expression) const {
    double value = 0;
    for (auto term : expression.terms)
      value += term.second * hint(term.first);
    return value;
  }
  bool submitPrimalStart() {
    if (!check())
      return false;
    SCIP_SOL *solution = nullptr;
    if (!api(SCIPcreateOrigSol(scip, &solution, nullptr)))
      return false;
    for (auto variable : owned) {
      if (!check() ||
          !api(SCIPsetSolVal(scip, solution, variable, hint(variable)))) {
        SCIPfreeSol(scip, &solution);
        return false;
      }
    }
    SCIP_Bool feasible = FALSE;
    if (!api(SCIPcheckSol(scip, solution, FALSE, TRUE, TRUE, TRUE, TRUE,
                          &feasible))) {
      SCIPfreeSol(scip, &solution);
      return false;
    }
    if (!feasible) {
      SCIPfreeSol(scip, &solution);
      return fail(
          ConstraintSearchStatus::Unknown,
          "checked baseline failed complete MILP primal-start validation");
    }
    SCIP_Bool stored = FALSE;
    if (!api(SCIPaddSolFree(scip, &solution, &stored))) {
      if (solution)
        SCIPfreeSol(scip, &solution);
      return false;
    }
    return stored || fail(ConstraintSearchStatus::Unknown,
                          "SCIP did not retain checked baseline primal start");
  }
  bool row(const Expr &e, double lower, double upper) {
    if (!check())
      return false;
    if (telemetry.constraints >= options.maxConstraints)
      return fail(ConstraintSearchStatus::ModelLimit, "MILP constraint cap");
    std::vector<Var> vars;
    std::vector<double> coeffs;
    // Keep integer coefficients and their absolute sum exactly representable.
    // SCIP uses binary64 even though the public contract uses integer ns.
    double absoluteSum = 0;
    for (auto entry : e.terms)
      if (entry.second) {
        absoluteSum += std::abs(entry.second);
        if (!std::isfinite(absoluteSum) || absoluteSum > 9007199254740991.0)
          return fail(ConstraintSearchStatus::ModelLimit,
                      "MILP row exceeds safe binary64 coefficient range");
        vars.push_back(entry.first);
        coeffs.push_back(entry.second);
      }
    SCIP_CONS *c = nullptr;
    if (!api(SCIPcreateConsBasicLinear(scip, &c, "", int(vars.size()),
                                       vars.data(), coeffs.data(), lower,
                                       upper)))
      return false;
    auto rc = SCIPaddCons(scip, c);
    SCIPreleaseCons(scip, &c);
    if (!api(rc))
      return false;
    ++telemetry.constraints;
    return true;
  }
  double inf() const { return SCIPinfinity(scip); }
  bool eq(const Expr &e, double value = 0) { return row(e, value, value); }
  bool ge(const Expr &e, double value = 0) { return row(e, value, inf()); }
  bool le(const Expr &e, double value = 0) { return row(e, -inf(), value); }
  // All guarded expressions used here have range [-H,H] before constants.
  void guardedGE(Expr e, double rhs, Var presence) {
    const double big = horizon + std::max(0.0, rhs);
    e.add(presence, -big);
    ge(e, rhs - big);
  }
  Var conjunction(const std::vector<Var> &xs) {
    auto z = var(true);
    Expr lower(z);
    for (auto x : xs) {
      le(Expr(z).add(x, -1));
      lower.add(x, -1);
    }
    ge(lower, 1 - double(xs.size()));
    hint(z, std::all_of(xs.begin(), xs.end(),
                        [&](Var x) { return hint(x) == 1; }));
    return z;
  }
};
struct Interval {
  Var presence, start, end;
  int64_t duration;
};
struct Route {
  size_t version, link;
  Var chosen, begin, end;
  int64_t duration;
};
struct Assignment {
  size_t group, place, configuration;
  Var chosen;
  std::vector<Var> compute;
  std::vector<Route> inputs, outputs;
};
struct Demand {
  Interval interval;
  int64_t demand;
};
// Optional valid inequality for a narrowly checked target class. It prices
// local groups only to bound the objective; no complete plan is pre-priced and
// no grouping, placement, route or configuration is removed.
bool addUnaryPortBound(const ProgramJointDomain &domain,
                       const std::vector<Assignment> &assignments,
                       const std::map<std::pair<std::string, size_t>,
                                      std::vector<Demand>> &resources,
                       const std::vector<Var> &selected, Var makespan,
                       MILP &builder) {
  if (!domain.costsComplete || domain.maxConfigurationInputs != 2 ||
      domain.maxConfigurationOutputs != 2 ||
      builder.telemetry.constraints >= builder.options.maxConstraints)
    return false;
  constexpr int64_t limit = 9007199254740991LL;
  auto add = [](int64_t &value, int64_t increment) {
    if (increment < 0 || value > limit - increment)
      return false;
    value += increment;
    return true;
  };
  std::set<Var> routeVariables;
  for (const auto &a : assignments)
    for (bool input : {true, false})
      for (const auto &r : (input ? a.inputs : a.outputs))
        if (r.duration > 0)
          routeVariables.insert(r.chosen);
  std::map<Var, std::set<std::string>> unary;
  for (const auto &entry : resources) {
    auto resource = programResource(domain.target, entry.first.first);
    if (!resource || resource->capacity != 1)
      continue;
    for (const auto &item : entry.second)
      if (routeVariables.count(item.interval.presence))
        unary[item.interval.presence].insert(entry.first.first);
  }
  bool first = true;
  std::set<std::string> common;
  for (auto variable : routeVariables) {
    const auto &current = unary[variable];
    if (first) {
      common = current;
      first = false;
    } else {
      std::set<std::string> intersection;
      std::set_intersection(common.begin(), common.end(), current.begin(),
                            current.end(),
                            std::inserter(intersection, intersection.end()));
      common = std::move(intersection);
    }
    if (common.empty())
      return false;
  }
  if (first)
    return false;
  struct GroupBound {
    bool present = false, unit = false;
    int64_t compute = 0, transfer = limit;
  };
  std::vector<GroupBound> groups(domain.groups.size());
  int64_t minimumCompute = limit, maximumOutput = 0;
  bool hasUnit = false;
  for (const auto &a : assignments) {
    if (!builder.check())
      return false;
    const auto &g = domain.groups[a.group];
    if (g.inputs.empty() || g.inputs.size() > 2 || g.outputs.empty() ||
        g.outputs.size() > 2)
      return false;
    auto &bound = groups[a.group];
    bound.present = true;
    bound.unit = g.inputs.size() == 1 && g.outputs.size() == 1;
    int64_t compute = 0, transfer = 0;
    for (auto i : g.instances)
      if (!add(compute, domain.variants[g.variant].computeNs[i]))
        return false;
    bound.compute = compute;
    for (bool input : {true, false})
      for (auto version : (input ? g.inputs : g.outputs)) {
        int64_t minimum = limit;
        for (const auto &r : (input ? a.inputs : a.outputs))
          if (r.version == version) {
            minimum = std::min(minimum, r.duration);
            if (bound.unit && !input)
              maximumOutput = std::max(maximumOutput, r.duration);
          }
        if (minimum == limit || !add(transfer, minimum))
          return false;
      }
    bound.transfer = std::min(bound.transfer, transfer);
    if (bound.unit) {
      hasUnit = true;
      minimumCompute = std::min(minimumCompute, compute);
    }
  }
  const int64_t gap =
      hasUnit ? std::max<int64_t>(0, minimumCompute - maximumOutput) : 0;
  Expr lower(makespan, 2);
  int64_t coefficientSum = 2;
  for (size_t g = 0; g < groups.size(); ++g) {
    const auto &bound = groups[g];
    if (!bound.present)
      continue;
    int64_t weight = bound.transfer;
    if (bound.unit) {
      if (weight > limit / 2)
        return false;
      weight *= 2;
      if (!add(weight, domain.configurationNs) || !add(weight, gap))
        return false;
    } else {
      if (!add(weight, bound.compute) || !add(weight, domain.configurationNs) ||
          weight > limit / 2)
        return false;
      weight *= 2;
    }
    if (!add(coefficientSum, weight))
      return false;
    lower.add(selected[g], -double(weight));
  }
  int64_t boundary = domain.submitNs;
  if (!add(boundary, domain.completeNs) || boundary > limit / 2)
    return false;
  // Every group has >=1 input/output. Thus only two (1,1) groups can
  // coexist; saturated-port groups are alone. For a pair, after its last
  // input there are >=Cmin compute ns, and only the other output (<=Omax)
  // can fill that transfer gap. Configuration length is consequently at
  // least S + sum(D) + max(0,Cmin-Omax). Lone groups satisfy the same
  // half-weight bound. Summing serial configurations proves this row.
  return builder.ge(lower, 2.0 * boundary);
}
} // namespace
#endif

ProgramJointSolution orbit::taskflow::solveProgramJointDomainMILP(
    const ProgramJointDomain &domain, const ConstraintSearchOptions &options,
    const ProgramJointSolution *checkedBaseline) {
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  auto elapsedMilliseconds = [&]() {
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
  };
  ConstraintSearchTelemetry telemetry;
  bool skippedPrice = false;
  auto baseline =
      checkedBaseline ? *checkedBaseline : baselineProgramJointDomain(domain);
  if (baseline.hasPlan) {
    ProgramGraph graph;
    std::string e;
    if (!materializeProgramJointSolution(domain, baseline, graph, e) ||
        !checkProgramJointSolution(domain, baseline, graph, e)) {
      baseline.hasPlan = false;
      baseline.message = "baseline independently rejected: " + e;
    }
  }
  auto finish = [&](ProgramJointSolution s) {
    s.pricingComplete =
        s.pricingComplete && domain.costsComplete && !skippedPrice;
    telemetry.totalMs = elapsedMilliseconds();
    struct rusage r;
    if (!getrusage(RUSAGE_SELF, &r))
      telemetry.peakRssKiB = r.ru_maxrss;
    s.telemetry = telemetry;
    return s;
  };
  auto fallback = [&](ConstraintSearchStatus status, const std::string &why) {
    auto s = baseline;
    s.status = status;
    s.message = why;
    s.selection = s.hasPlan ? "baseline" : "none";
    return finish(s);
  };
#ifndef ORBIT_HAS_SCIP
  (void)options;
  return fallback(ConstraintSearchStatus::Unavailable,
                  "built without ORBIT_HAS_SCIP; checked baseline only");
#else
  if (!std::isfinite(options.wallSeconds) || options.wallSeconds <= 0 ||
      options.workers < 1 || options.randomSeed < 0 ||
      options.maxVariables < 1 || options.maxConstraints < 1 ||
      options.maxPatterns < 1 || options.maxTasks < 1)
    return fallback(ConstraintSearchStatus::InvalidInput,
                    "invalid MILP options");
  // This backend uses sequential SCIPsolve, not SCIPsolveConcurrent. Reject an
  // unsupported thread request rather than silently changing benchmark setup.
  if (options.workers != 1)
    return fallback(ConstraintSearchStatus::InvalidInput,
                    "SCIP backend currently supports workers=1 only");
  if (!baseline.hasPlan)
    return fallback(ConstraintSearchStatus::Unknown,
                    "no independently checked finite horizon: " +
                        baseline.message);
  // Four horizon units bound every big-M expression. This deliberately rejects
  // unsupported numerical ranges rather than silently rounding nanoseconds.
  const int64_t horizonNs = baseline.makespanNs;
  if (horizonNs < 1 || horizonNs > ((int64_t(1) << 53) - 1) / 8)
    return fallback(ConstraintSearchStatus::Unknown,
                    "MILP integer horizon exceeds safe binary64 big-M range");
  const size_t variantCount = domain.variants.size(),
               groupCount = domain.groups.size(),
               placeCount = domain.places.size();
  if (!variantCount || !groupCount || !placeCount ||
      domain.maxConfigurations < 1 || domain.configurationNs < 0 ||
      domain.submitNs < 0 || domain.completeNs < 0 ||
      domain.configurationNs > horizonNs || domain.submitNs > horizonNs ||
      domain.completeNs > horizonNs)
    return fallback(ConstraintSearchStatus::InvalidInput,
                    "invalid joint domain/boundary");
  const size_t configurationCount = size_t(domain.maxConfigurations);
  if (variantCount > 8 || placeCount > 16 || configurationCount > 128 ||
      groupCount > size_t(options.maxPatterns))
    return fallback(ConstraintSearchStatus::ModelLimit,
                    "joint domain dimension cap");
  MILP builder(options, telemetry, start, horizonNs);
  auto failed = [&]() {
    telemetry.modelBuildMs = elapsedMilliseconds();
    return fallback(builder.failure, builder.error);
  };
  if (!builder.check())
    return failed();
  auto binary = [&]() { return builder.var(true); };
  auto time = [&]() { return builder.var(); };
  std::vector<Var> variant, selected, used, begin, end, nonzero;
  std::vector<std::vector<Expr>> completions(configurationCount);
  for (size_t v = 0; v < variantCount && builder.check(); ++v) {
    variant.push_back(binary());
    builder.hint(variant.back(), baseline.variant == int64_t(v));
  }
  Expr variantSum;
  for (auto x : variant)
    variantSum.add(x);
  builder.eq(variantSum, 1);
  for (size_t g = 0; g < groupCount && builder.check(); ++g) {
    selected.push_back(binary());
    builder.hint(selected.back(),
                 std::any_of(baseline.groups.begin(), baseline.groups.end(),
                             [&](const auto &a) { return a.group == g; }));
  }
  // Contiguous configuration barriers keep later phases from observing data
  // before the preceding phase has completed all compute and boundary copies.
  for (size_t k = 0; k < configurationCount && builder.check(); ++k) {
    used.push_back(binary());
    begin.push_back(time());
    end.push_back(time());
    nonzero.push_back(binary());
    const bool active = k < baseline.configurationEndNs.size();
    const int64_t startHint = active ? baseline.configurationStartNs[k]
                                     : baseline.configurationEndNs.back();
    const int64_t endHint = active ? baseline.configurationEndNs[k]
                                   : baseline.configurationEndNs.back();
    builder.hint(used[k], active);
    builder.hint(begin[k], startHint);
    builder.hint(end[k], endHint);
    builder.hint(nonzero[k], endHint > startHint);
    if (k) {
      builder.eq(Expr(begin[k]).add(end[k - 1], -1));
      builder.le(Expr(used[k]).add(used[k - 1], -1));
    } else
      builder.eq(Expr(begin[k]), domain.submitNs);
    builder.ge(Expr(end[k]).add(begin[k], -1));
    builder.le(Expr(end[k]).add(begin[k], -1).add(used[k], -double(horizonNs)));
    builder.ge(Expr(end[k]).add(begin[k], -1).add(nonzero[k], -1));
    builder.le(
        Expr(end[k]).add(begin[k], -1).add(nonzero[k], -double(horizonNs)));
    completions[k].push_back(
        Expr(begin[k]).add(used[k], domain.configurationNs));
  }
  if (!builder.check())
    return failed();
  std::vector<std::vector<std::vector<size_t>>> covers(variantCount),
      exporters(variantCount);
  std::vector<std::vector<Var>> ready(variantCount);
  std::vector<std::vector<Expr>> phases(variantCount);
  for (size_t v = 0; v < variantCount && builder.check(); ++v) {
    auto &x = domain.variants[v];
    size_t N = x.graph.instances.size(), M = x.producers.size();
    if (!N || N > size_t(options.maxTasks) || x.computeNs.size() != N ||
        x.readers.size() != M || x.valueBytes.size() != M)
      return fallback(ConstraintSearchStatus::InvalidInput,
                      "instance/version array mismatch or cap");
    covers[v].resize(N);
    exporters[v].resize(M);
    phases[v].resize(N);
    for (auto ns : x.computeNs)
      if (ns < 0 || ns > horizonNs)
        return fallback(ConstraintSearchStatus::Unknown,
                        "compute duration outside checked horizon");
    for (size_t z = 0; z < M && builder.check(); ++z) {
      if (x.producers[z] < -1 || x.producers[z] >= int64_t(N))
        return fallback(ConstraintSearchStatus::InvalidInput,
                        "invalid producer");
      for (auto i : x.readers[z])
        if (i >= N)
          return fallback(ConstraintSearchStatus::InvalidInput,
                          "invalid reader");
      ready[v].push_back(time());
      if (x.producers[z] < 0)
        builder.hint(ready[v].back(), domain.submitNs);
      if (baseline.variant == int64_t(v))
        for (const auto &a : baseline.groups)
          for (const auto &route : a.outputs)
            if (route.version == z)
              builder.hint(ready[v].back(), route.endNs);
      if (x.producers[z] < 0)
        builder.eq(Expr(ready[v].back()), domain.submitNs);
    }
  }
  if (!builder.check())
    return failed();
  for (size_t g = 0; g < groupCount && builder.check(); ++g) {
    auto &m = domain.groups[g];
    if (m.variant >= variantCount || m.instances.empty())
      return fallback(ConstraintSearchStatus::InvalidInput,
                      "invalid local group");
    std::set<size_t> unique;
    for (auto i : m.instances) {
      if (i >= covers[m.variant].size() || !unique.insert(i).second)
        return fallback(ConstraintSearchStatus::InvalidInput,
                        "invalid/duplicate group instance");
      covers[m.variant][i].push_back(g);
    }
    for (auto z : m.inputs)
      if (z >= ready[m.variant].size())
        return fallback(ConstraintSearchStatus::InvalidInput,
                        "invalid input version");
    for (auto z : m.outputs) {
      if (z >= ready[m.variant].size())
        return fallback(ConstraintSearchStatus::InvalidInput,
                        "invalid output version");
      exporters[m.variant][z].push_back(g);
    }
    builder.le(Expr(selected[g]).add(variant[m.variant], -1));
  }
  if (!builder.check())
    return failed();
  std::vector<Assignment> assignments;
  std::vector<Expr> groupChoices(groupCount), configChoices(configurationCount),
      inputCounts(configurationCount), outputCounts(configurationCount);
  std::map<std::pair<std::string, size_t>, Expr> coreChoices, memoryChoices;
  std::map<std::pair<std::string, size_t>, std::vector<Demand>> resources;
  auto reserve = [&](const std::string &id, size_t k, Interval interval) {
    if (!builder.check() || !interval.duration)
      return;
    std::map<std::string, int64_t> expanded;
    std::string e;
    if (!programResourceDemands(domain.target, {{id, 1}}, expanded, e)) {
      builder.fail(ConstraintSearchStatus::InvalidInput, e);
      return;
    }
    for (auto &r : expanded) {
      auto resource = programResource(domain.target, r.first);
      if (!resource || r.second < 0 || resource->capacity < 0 ||
          resource->capacity > ((int64_t(1) << 53) - 1) ||
          r.second > ((int64_t(1) << 53) - 1)) {
        builder.fail(ConstraintSearchStatus::InvalidInput,
                     "invalid resource capacity/demand");
        return;
      }
      if (r.second)
        resources[{r.first, k}].push_back({interval, r.second});
    }
  };
  // Each assignment activates a local group, placement and configuration
  // together. Every boundary value must choose a priced route when active;
  // skipping an unknown alternative only permits a known-subdomain result.
  for (size_t g = 0; g < groupCount && builder.check(); ++g) {
    auto &mode = domain.groups[g];
    auto v = mode.variant;
    auto &sourceVariant = domain.variants[v];
    for (auto z : mode.inputs)
      if (sourceVariant.producers[z] >= 0) {
        Expr e(selected[g]);
        for (auto producer : exporters[v][z])
          e.add(selected[producer], -1);
        builder.le(e);
      }
    for (size_t p = 0; p < placeCount && builder.check(); ++p) {
      int64_t bytes = 0;
      std::string e;
      if (!programJointGroupFits(domain, g, p, bytes, e))
        continue;
      auto memory =
          programMemory(domain.target, domain.places[p].binding.memory);
      if (!memory || memory->capacityBytes < 0 ||
          memory->capacityBytes > ((int64_t(1) << 53) - 1) || bytes < 0 ||
          bytes > ((int64_t(1) << 53) - 1))
        return fallback(ConstraintSearchStatus::InvalidInput,
                        "invalid memory reservation");
      for (size_t k = 0; k < configurationCount && builder.check(); ++k) {
        if (assignments.size() >= size_t(options.maxPatterns)) {
          builder.fail(ConstraintSearchStatus::ModelLimit,
                       "MILP assignment cap");
          break;
        }
        auto x = binary();
        Assignment a{g, p, k, x, {}, {}, {}};
        const ProgramJointSelection *baselineAssignment = nullptr;
        for (const auto &candidate : baseline.groups)
          if (candidate.group == g && candidate.place == p &&
              candidate.configuration == k)
            baselineAssignment = &candidate;
        builder.hint(x, baselineAssignment != nullptr);
        std::set<size_t> hintedInputVersions, hintedOutputVersions;
        groupChoices[g].add(x);
        configChoices[k].add(x);
        coreChoices[{domain.places[p].binding.resource, k}].add(x);
        inputCounts[k].add(x, mode.inputs.size());
        outputCounts[k].add(x, mode.outputs.size());
        builder.le(Expr(x).add(used[k], -1));
        auto occupied = builder.conjunction({x, nonzero[k]});
        memoryChoices[{domain.places[p].binding.memory, k}].add(occupied,
                                                                bytes);
        auto routes = [&](bool input) {
          for (auto z : (input ? mode.inputs : mode.outputs)) {
            if (!builder.check())
              break;
            Expr sum;
            sum.add(x, -1);
            for (auto li : (input ? domain.places[p].inputLinks
                                  : domain.places[p].outputLinks)) {
              if (!builder.check())
                break;
              if (li >= domain.target.links.size()) {
                builder.fail(ConstraintSearchStatus::InvalidInput,
                             "invalid route index");
                break;
              }
              int64_t duration = 0;
              std::string why;
              if (!programJointTransferNs(domain, v, z, li, duration, why)) {
                if (programJointTransferSupported(domain, v, z, li, why))
                  skippedPrice = true;
                continue;
              }
              if (duration < 0 || duration > horizonNs) {
                builder.fail(ConstraintSearchStatus::Unknown,
                             "route duration outside safe horizon");
                break;
              }
              auto r = binary(), rb = time(), re = time();
              const ProgramJointTransfer *routeHint = nullptr;
              auto &hintedVersions =
                  input ? hintedInputVersions : hintedOutputVersions;
              if (baselineAssignment && !hintedVersions.count(z))
                for (const auto &candidate :
                     (input ? baselineAssignment->inputs
                            : baselineAssignment->outputs))
                  if (candidate.version == z && candidate.link == li)
                    routeHint = &candidate;
              if (routeHint)
                hintedVersions.insert(z);
              builder.hint(r, routeHint != nullptr);
              builder.hint(rb, routeHint ? routeHint->startNs : 0);
              builder.hint(re, routeHint ? routeHint->endNs : 0);
              sum.add(r);
              builder.eq(Expr(re).add(rb, -1).add(r, -double(duration)));
              builder.le(Expr(rb).add(r, -double(horizonNs)));
              builder.le(Expr(re).add(r, -double(horizonNs)));
              builder.guardedGE(Expr(rb).add(begin[k], -1),
                                domain.configurationNs, r);
              builder.le(Expr(re).add(end[k], -1));
              completions[k].push_back(Expr(re));
              if (input)
                builder.guardedGE(Expr(rb).add(ready[v][z], -1), 0, r);
              else {
                builder.guardedGE(Expr(ready[v][z]).add(re, -1), 0, r);
                builder.guardedGE(Expr(re).add(ready[v][z], -1), 0, r);
              }
              reserve(domain.target.links[li].resource, k,
                      {r, rb, re, duration});
              (input ? a.inputs : a.outputs)
                  .push_back({z, li, r, rb, re, duration});
            }
            builder.eq(sum);
          }
        };
        routes(true);
        routes(false);
        std::vector<Var> finishes;
        for (size_t i = 0; i < mode.instances.size() && builder.check(); ++i) {
          auto instance = mode.instances[i];
          auto cb = time(), ce = time();
          auto duration = sourceVariant.computeNs[instance];
          builder.hint(cb, baselineAssignment
                               ? baselineAssignment->computeStartNs[i]
                               : 0);
          builder.hint(ce,
                       baselineAssignment
                           ? baselineAssignment->computeStartNs[i] + duration
                           : 0);
          a.compute.push_back(cb);
          finishes.push_back(ce);
          builder.eq(Expr(ce).add(cb, -1).add(x, -double(duration)));
          builder.le(Expr(cb).add(x, -double(horizonNs)));
          builder.le(Expr(ce).add(x, -double(horizonNs)));
          builder.guardedGE(Expr(cb).add(begin[k], -1), domain.configurationNs,
                            x);
          builder.le(Expr(ce).add(end[k], -1));
          if (i)
            builder.guardedGE(Expr(cb).add(finishes[i - 1], -1), 0, x);
          reserve(domain.places[p].binding.resource, k, {x, cb, ce, duration});
          phases[v][instance].add(x, k);
          completions[k].push_back(Expr(ce));
        }
        if (!builder.check())
          break;
        for (auto &r : a.inputs)
          builder.guardedGE(Expr(a.compute.front()).add(r.end, -1), 0,
                            r.chosen);
        for (auto &r : a.outputs)
          builder.guardedGE(Expr(r.begin).add(finishes.back(), -1), 0,
                            r.chosen);
        assignments.push_back(std::move(a));
      }
    }
    builder.eq(groupChoices[g].add(selected[g], -1));
  }
  if (!builder.check())
    return failed();
  // Exact cover applies only to the chosen source variant: each source
  // instance executes once, even when several overlapping groups are offered.
  int64_t intersectionWork = 0;
  for (size_t v = 0; v < variantCount && builder.check(); ++v) {
    for (size_t i = 0; i < covers[v].size() && builder.check(); ++i) {
      Expr cover;
      for (auto g : covers[v][i])
        cover.add(selected[g]);
      builder.eq(cover.add(variant[v], -1));
    }
    // A RAW dependency can remain in one configuration only when its producer
    // and consumer share the selected group. Otherwise the exported host
    // version must be available to a later configuration.
    std::set<std::pair<size_t, size_t>> raw;
    for (size_t z = 0; z < domain.variants[v].producers.size(); ++z)
      if (domain.variants[v].producers[z] >= 0)
        for (auto reader : domain.variants[v].readers[z])
          raw.emplace(size_t(domain.variants[v].producers[z]), reader);
    for (auto edge : raw) {
      if (!builder.check())
        break;
      if (edge.first == edge.second)
        continue;
      auto &left = covers[v][edge.first], &right = covers[v][edge.second];
      if (left.size() + right.size() >
          size_t(std::max<int64_t>(0, options.maxIntersectionWork -
                                          intersectionWork))) {
        builder.fail(ConstraintSearchStatus::ModelLimit,
                     "RAW intersection work cap");
        break;
      }
      intersectionWork += left.size() + right.size();
      std::vector<size_t> common;
      std::set_intersection(left.begin(), left.end(), right.begin(),
                            right.end(), std::back_inserter(common));
      Expr e = phases[v][edge.second];
      for (auto q : phases[v][edge.first].terms)
        e.add(q.first, -q.second);
      e.add(variant[v], -1);
      for (auto g : common)
        e.add(selected[g]);
      builder.ge(e);
    }
    for (auto edge : domain.variants[v].graph.orderEdges) {
      if (!builder.check())
        break;
      auto &graph = domain.variants[v].graph;
      if (edge.producer >= graph.events.size() ||
          edge.consumer >= graph.events.size())
        return fallback(ConstraintSearchStatus::InvalidInput,
                        "invalid order event");
      auto pt = graph.events[edge.producer].task,
           ct = graph.events[edge.consumer].task;
      for (size_t p = 0; p < covers[v].size() && builder.check(); ++p)
        if (graph.instances[p].task == pt)
          for (size_t c = 0; c < covers[v].size() && builder.check(); ++c)
            if (graph.instances[c].task == ct) {
              Expr e = phases[v][c];
              for (auto q : phases[v][p].terms)
                e.add(q.first, -q.second);
              e.add(variant[v], -1);
              builder.ge(e);
            }
    }
  }
  if (!builder.check())
    return failed();
  for (auto &entry : coreChoices)
    builder.le(entry.second, 1);
  for (auto &entry : memoryChoices)
    builder.le(entry.second,
               programMemory(domain.target, entry.first.first)->capacityBytes);
  for (size_t k = 0; k < configurationCount && builder.check(); ++k) {
    builder.ge(configChoices[k].add(used[k], -1));
    builder.le(inputCounts[k], domain.maxConfigurationInputs);
    builder.le(outputCounts[k], domain.maxConfigurationOutputs);
    Expr maxSelectors;
    bool hintedMaximum = false;
    for (auto candidate : completions[k]) {
      if (!builder.check())
        break;
      auto z = binary();
      const bool isMaximum =
          !hintedMaximum && builder.hint(candidate) == builder.hint(end[k]);
      builder.hint(z, isMaximum);
      hintedMaximum |= isMaximum;
      maxSelectors.add(z);
      Expr difference(end[k]);
      for (auto q : candidate.terms)
        difference.add(q.first, -q.second);
      builder.ge(difference);
      builder.le(difference.add(z, double(horizonNs)), double(horizonNs));
    }
    builder.eq(maxSelectors, 1);
  }
  // Exact cumulative capacity: integer start-anchor occupancy. Demand can only
  // increase at an active positive-length interval's start. Comparisons are
  // iff, including ties; end == start is outside the half-open interval. No
  // time grid.
  for (auto &entry : resources) {
    if (!builder.check())
      break;
    auto &items = entry.second;
    auto capacity = programResource(domain.target, entry.first.first)->capacity;
    if (capacity <= 1) {
      // Unary resources need only one order bit per unordered pair. If either
      // interval is absent, both big-M inequalities are redundant. Present
      // intervals may touch at endpoints; zero-length intervals were omitted
      // by reserve(). This preserves the full feasible domain.
      for (const auto &item : items)
        if (item.demand > capacity)
          builder.eq(Expr(item.interval.presence));
      if (capacity == 0)
        continue;
      for (size_t i = 0; i < items.size() && builder.check(); ++i)
        for (size_t j = i + 1; j < items.size() && builder.check(); ++j) {
          const auto &left = items[i].interval, &right = items[j].interval;
          auto order = binary();
          builder.hint(order,
                       builder.hint(left.end) <= builder.hint(right.start));
          builder.le(Expr(left.end)
                         .add(right.start, -1)
                         .add(left.presence, horizonNs)
                         .add(right.presence, horizonNs)
                         .add(order, horizonNs),
                     3.0 * horizonNs);
          builder.le(Expr(right.end)
                         .add(left.start, -1)
                         .add(left.presence, horizonNs)
                         .add(right.presence, horizonNs)
                         .add(order, -double(horizonNs)),
                     2.0 * horizonNs);
        }
      continue;
    }

    for (size_t i = 0; i < items.size() && builder.check(); ++i) {
      auto &anchor = items[i].interval;
      Expr load;
      for (size_t j = 0; j < items.size() && builder.check(); ++j) {
        if (i == j) {
          load.add(anchor.presence, items[j].demand);
          continue;
        }
        auto &other = items[j].interval;
        auto before = binary(), after = binary();
        builder.hint(before,
                     builder.hint(other.start) <= builder.hint(anchor.start));
        builder.hint(after,
                     builder.hint(other.end) > builder.hint(anchor.start));
        // before <=> other.start <= anchor.start
        builder.le(Expr(other.start)
                       .add(anchor.start, -1)
                       .add(before, double(horizonNs)),
                   double(horizonNs));
        builder.ge(Expr(other.start)
                       .add(anchor.start, -1)
                       .add(before, double(horizonNs) + 1),
                   1);
        // after <=> other.end >= anchor.start + 1
        builder.ge(Expr(other.end)
                       .add(anchor.start, -1)
                       .add(after, -double(horizonNs) - 1),
                   -double(horizonNs));
        builder.le(Expr(other.end)
                       .add(anchor.start, -1)
                       .add(after, -double(horizonNs)));
        auto overlaps = builder.conjunction(
            {anchor.presence, other.presence, before, after});
        load.add(overlaps, items[j].demand);
      }
      builder.le(load, capacity);
    }
  }
  if (!builder.check())
    return failed();
  auto makespan = builder.var(false, 1);
  builder.hint(makespan, baseline.makespanNs);
  builder.eq(Expr(makespan).add(end.back(), -1), domain.completeNs);
  if (!builder.check())
    return failed();
  const bool unaryPortBound = addUnaryPortBound(domain, assignments, resources,
                                                selected, makespan, builder);
  if (!builder.check())
    return failed();
  // Submit every original variable, including derived occupancy and max bits,
  // through SCIP's solution store. Do not fix bounds or restrict alternatives.
  if (!builder.submitPrimalStart())
    return failed();
  telemetry.modelBuildMs = elapsedMilliseconds();
  double remaining = options.wallSeconds - elapsedMilliseconds() / 1000;
  if (remaining <= 0) {
    telemetry.budgetExhausted = true;
    return fallback(ConstraintSearchStatus::Unknown,
                    "total wall budget exhausted before solve");
  }
  if (!builder.api(
          SCIPsetRealParam(builder.scip, "limits/time", remaining * .9)))
    return failed();
  auto solveStart = Clock::now();
  if (!builder.api(SCIPsolve(builder.scip)))
    return fallback(ConstraintSearchStatus::Unknown, builder.error);
  telemetry.solveMs =
      std::chrono::duration<double, std::milli>(Clock::now() - solveStart)
          .count();
  telemetry.branches = SCIPgetNNodes(builder.scip);
  auto status = SCIPgetStatus(builder.scip);
  auto sol = SCIPgetBestSol(builder.scip);
  telemetry.budgetExhausted =
      status == SCIP_STATUS_TIMELIMIT ||
      elapsedMilliseconds() >= options.wallSeconds * 1000;
  if (!sol) {
    if (status == SCIP_STATUS_INFEASIBLE)
      return fallback(ConstraintSearchStatus::InvalidInput,
                      "MILP rejects its independently checked baseline: "
                      "encoding/numerical mismatch");
    return fallback(
        ConstraintSearchStatus::Unknown,
        "SCIP has no incumbent; checked baseline retained (status " +
            std::to_string(int(status)) + ")");
  }
  bool numerical = false;
  auto value = [&](Var v) -> int64_t {
    double x = SCIPgetSolVal(builder.scip, sol, v);
    if (!std::isfinite(x) || x < -.00001 || x > double(horizonNs) + .00001 ||
        std::abs(x - std::round(x)) > .00001) {
      numerical = true;
      return 0;
    }
    return int64_t(std::llround(x));
  };
  ProgramJointSolution s;
  s.hasPlan = true;
  s.selection = "joint-milp";
  bool fullScope = domain.complete && domain.costsComplete && !skippedPrice;
  s.status = status == SCIP_STATUS_OPTIMAL && fullScope
                 ? ConstraintSearchStatus::Optimal
                 : ConstraintSearchStatus::Feasible;
  s.message = "direct SCIP integer-nanosecond joint MILP; independent schedule "
              "check; numerical optimality tolerances apply";
  if (!fullScope)
    s.message +=
        "; known priced subdomain only (no full-domain optimality proof)";
  if (unaryPortBound)
    s.message += "; guarded unary-port lower bound";
  s.message += "; checked baseline primal start accepted; SCIP status=" +
               std::to_string(int(status));
  s.makespanNs = value(makespan);
  for (size_t v = 0; v < variantCount; ++v)
    if (value(variant[v]))
      s.variant = v;
  for (size_t k = 0; k < configurationCount; ++k)
    if (value(used[k])) {
      s.configurationStartNs.push_back(value(begin[k]));
      s.configurationEndNs.push_back(value(end[k]));
    }
  for (auto &a : assignments)
    if (value(a.chosen)) {
      ProgramJointSelection x;
      x.group = a.group;
      x.place = a.place;
      x.configuration = a.configuration;
      for (auto cb : a.compute)
        x.computeStartNs.push_back(value(cb));
      for (bool input : {true, false})
        for (auto &r : (input ? a.inputs : a.outputs))
          if (value(r.chosen))
            (input ? x.inputs : x.outputs)
                .push_back({r.version, r.link, value(r.begin), value(r.end)});
      s.groups.push_back(std::move(x));
    }
  if (numerical)
    return fallback(
        ConstraintSearchStatus::Unknown,
        "SCIP incumbent fails integer-nanosecond numerical validation");
  // Integer-looking SCIP values are not a sufficient execution witness.
  // Rebuild and independently check source coverage, versions, costs and
  // resource calendars before accepting the incumbent.
  ProgramGraph graph;
  std::string error;
  if (!materializeProgramJointSolution(domain, s, graph, error) ||
      !checkProgramJointSolution(domain, s, graph, error))
    return fallback(ConstraintSearchStatus::Unknown,
                    "MILP incumbent independently rejected: " + error);
  telemetry.hasObjective = true;
  telemetry.objectiveNs = s.makespanNs;
  telemetry.bestBoundNs = SCIPgetDualbound(builder.scip);
  telemetry.relativeGap =
      std::isfinite(telemetry.bestBoundNs)
          ? std::max(0.0, double(s.makespanNs) - telemetry.bestBoundNs) /
                std::max(1.0, std::abs(double(s.makespanNs)))
          : 1;
  if (!std::isfinite(telemetry.bestBoundNs)) {
    telemetry.bestBoundNs = 0;
    s.status = ConstraintSearchStatus::Feasible;
    s.message += "; nonfinite dual bound";
  }
  return finish(s);
#endif
}
