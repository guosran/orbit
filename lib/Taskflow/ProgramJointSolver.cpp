#include "Taskflow/ProgramJointSearch.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <sys/resource.h>
#ifdef ORBIT_HAS_ORTOOLS
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"
#endif
using namespace orbit::taskflow;
ProgramJointSolution orbit::taskflow::solveProgramJointDomain(
    const ProgramJointDomain &domain, const ConstraintSearchOptions &options,
    const ProgramJointSolution *checkedBaseline) {
  using Clock = std::chrono::steady_clock;
  auto started = Clock::now();
  auto elapsedMilliseconds = [&]() {
    return std::chrono::duration<double, std::milli>(Clock::now() - started)
        .count();
  };
  auto baseline =
      checkedBaseline ? *checkedBaseline : baselineProgramJointDomain(domain);
  ConstraintSearchTelemetry telemetry;
  bool skippedPrice = false;
  if (checkedBaseline) {
    ProgramGraph graph;
    std::string error;
    if (!materializeProgramJointSolution(domain, baseline, graph, error) ||
        !checkProgramJointSolution(domain, baseline, graph, error)) {
      baseline.hasPlan = false;
      baseline.message = "supplied baseline rejected: " + error;
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
  auto fallback = [&](ConstraintSearchStatus status,
                      const std::string &message) {
    auto s = baseline;
    s.status = status;
    s.message = message;
    s.selection = s.hasPlan ? "baseline" : "none";
    return finish(s);
  };
#ifndef ORBIT_HAS_ORTOOLS
  (void)options;
  return fallback(ConstraintSearchStatus::Unavailable,
                  "built without ORBIT_ENABLE_ORTOOLS");
#else
  using namespace operations_research;
  using namespace operations_research::sat;
  // Missing prices can prevent a complete baseline without proving that the
  // source graph is infeasible. Preserve that distinction in the solve status.
  if (!baseline.hasPlan)
    return fallback(domain.costsComplete ? ConstraintSearchStatus::InvalidInput
                                         : ConstraintSearchStatus::Unknown,
                    baseline.message);
  if (!std::isfinite(options.wallSeconds) || options.wallSeconds <= 0 ||
      options.workers < 1 || options.workers > 16 || options.randomSeed < 0 ||
      options.maxVariables < 1 || options.maxConstraints < 1 ||
      options.maxPatterns < 1 || domain.maxConfigurations < 1)
    return fallback(ConstraintSearchStatus::InvalidInput,
                    "invalid solver budget/options");
  struct Stop {
    ConstraintSearchStatus status;
    std::string error;
  };
  try {
    CpModelBuilder builder;
    auto checkpoint = [&]() {
      telemetry.variables = builder.Proto().variables_size();
      telemetry.constraints = builder.Proto().constraints_size();
      if (elapsedMilliseconds() >= options.wallSeconds * 1000) {
        telemetry.budgetExhausted = true;
        throw Stop{ConstraintSearchStatus::Unknown,
                   "deadline exhausted during model construction; checked "
                   "baseline retained"};
      }
      if (builder.Proto().variables_size() > options.maxVariables ||
          builder.Proto().constraints_size() > options.maxConstraints)
        throw Stop{ConstraintSearchStatus::ModelLimit,
                   "joint model size limit; checked baseline retained"};
    };
    const int64_t horizonNs = baseline.makespanNs;
    // Objective-bound telemetry uses binary64. Above this limit, adjacent
    // integer nanoseconds cannot be distinguished reliably in both backends.
    if (horizonNs < 1 || horizonNs > programJointMaxExactNs)
      throw Stop{ConstraintSearchStatus::InvalidInput,
                 "unsafe checked-baseline horizon"};
    const size_t variantCount = domain.variants.size(),
                 groupCount = domain.groups.size(),
                 placeCount = domain.places.size(),
                 configurationCount = domain.maxConfigurations;
    if (groupCount > size_t(options.maxPatterns) || placeCount > 16 ||
        configurationCount > 128 || variantCount > 8)
      throw Stop{ConstraintSearchStatus::ModelLimit,
                 "joint domain dimension limit"};
    // Typed callers can extend a domain without refreshing costsComplete.
    // Dropping a supported but unpriced route only solves the priced subset;
    // it must never certify the complete domain's optimum.
    auto boolean = [&]() {
      checkpoint();
      return builder.NewBoolVar();
    };
    auto time = [&]() {
      checkpoint();
      return builder.NewIntVar(Domain(0, horizonNs));
    };
    struct Usage {
      std::vector<IntervalVar> intervals;
      std::vector<int64_t> demands;
    };
    std::map<std::string, Usage> resources, memories;
    auto reserve = [&](const std::string &id, IntervalVar interval) {
      std::map<std::string, int64_t> demands;
      std::string error;
      if (!programResourceDemands(domain.target, {{id, 1}}, demands, error))
        throw Stop{ConstraintSearchStatus::InvalidInput, error};
      for (auto &entry : demands) {
        resources[entry.first].intervals.push_back(interval);
        resources[entry.first].demands.push_back(entry.second);
      }
    };
    // Select one source variant, then cover its instances with local groups.
    // Groups are implementation choices, never pre-priced complete programs.
    std::vector<BoolVar> variant, selected, used;
    for (size_t v = 0; v < variantCount; ++v) {
      variant.push_back(boolean());
      builder.AddHint(variant.back(), baseline.variant == int64_t(v));
    }
    builder.AddExactlyOne(variant);
    for (size_t g = 0; g < groupCount; ++g) {
      selected.push_back(boolean());
      bool hint = false;
      for (auto &a : baseline.groups)
        hint |= a.group == g;
      builder.AddHint(selected.back(), hint);
    }
    // Configurations are serial barriers in this particular realization. All
    // selected groups in one configuration may overlap on distinct resources;
    // their local allocations remain live for the entire configuration.
    std::vector<IntVar> begin, end, length;
    std::vector<std::vector<LinearExpr>> completions(configurationCount);
    for (size_t k = 0; k < configurationCount; ++k) {
      used.push_back(boolean());
      begin.push_back(time());
      end.push_back(time());
      length.push_back(time());
      builder.AddEquality(length[k], LinearExpr(end[k]) - begin[k]);
      builder.AddEquality(begin[k], k ? LinearExpr(end[k - 1])
                                      : LinearExpr(domain.submitNs));
      if (k)
        builder.AddImplication(used[k], used[k - 1]);
      builder.AddEquality(end[k], begin[k]).OnlyEnforceIf(used[k].Not());
      completions[k].push_back(LinearExpr(begin[k]) +
                               LinearExpr(used[k]) * domain.configurationNs);
      builder.AddHint(used[k], k < baseline.configurationEndNs.size());
      builder.AddHint(begin[k], k < baseline.configurationStartNs.size()
                                    ? baseline.configurationStartNs[k]
                                    : baseline.configurationEndNs.back());
      builder.AddHint(end[k], k < baseline.configurationEndNs.size()
                                  ? baseline.configurationEndNs[k]
                                  : baseline.configurationEndNs.back());
    }
    std::vector<std::vector<std::vector<size_t>>> covers(variantCount),
        exporters(variantCount);
    std::vector<std::vector<IntVar>> phases(variantCount), ready(variantCount);
    std::vector<std::vector<LinearExpr>> phaseExpr(variantCount);
    for (size_t v = 0; v < variantCount; ++v) {
      auto &x = domain.variants[v];
      size_t N = x.graph.instances.size(), M = x.producers.size();
      if (N > size_t(options.maxTasks) || x.computeNs.size() != N ||
          x.readers.size() != M || x.valueBytes.size() != M)
        throw Stop{ConstraintSearchStatus::ModelLimit,
                   "instance/version domain limit"};
      covers[v].resize(N);
      exporters[v].resize(M);
      phaseExpr[v].resize(N);
      for (size_t i = 0; i < N; ++i)
        phases[v].push_back(
            builder.NewIntVar(Domain(0, configurationCount - 1)));
      for (size_t value = 0; value < M; ++value) {
        ready[v].push_back(time());
        if (x.producers[value] < 0)
          builder.AddEquality(ready[v].back(), domain.submitNs);
      }
    }
    for (size_t g = 0; g < groupCount; ++g) {
      auto &x = domain.groups[g];
      if (x.variant >= variantCount || x.instances.empty())
        throw Stop{ConstraintSearchStatus::InvalidInput, "invalid local mode"};
      for (auto i : x.instances) {
        if (i >= covers[x.variant].size())
          throw Stop{ConstraintSearchStatus::InvalidInput,
                     "invalid instance reference"};
        covers[x.variant][i].push_back(g);
      }
      for (auto value : x.outputs)
        exporters[x.variant][value].push_back(g);
      builder.AddImplication(selected[g], variant[x.variant]);
    }
    struct Route {
      size_t version, link;
      BoolVar chosen;
      IntVar begin, end;
    };
    struct Assignment {
      size_t group, place, configuration;
      BoolVar chosen;
      std::vector<IntVar> compute;
      std::vector<Route> inputs, outputs;
    };
    std::vector<Assignment> assignments;
    std::vector<std::vector<BoolVar>> groupChoices(groupCount),
        configChoices(configurationCount);
    std::vector<LinearExpr> inputCounts(configurationCount),
        outputCounts(configurationCount);
    std::map<std::pair<std::string, size_t>, std::vector<BoolVar>> coreChoices;
    // A group/place/configuration assignment activates compute, input and
    // output intervals together. An absent or unpriced route cannot become a
    // free transfer; only explicitly priced alternatives enter this model.
    for (size_t g = 0; g < groupCount; ++g) {
      auto &mode = domain.groups[g];
      auto v = mode.variant;
      auto &sourceVariant = domain.variants[v];
      for (auto value : mode.inputs)
        if (sourceVariant.producers[value] >= 0) {
          LinearExpr exported;
          for (auto p : exporters[v][value])
            exported += selected[p];
          builder.AddLessOrEqual(selected[g], exported);
        }
      for (size_t p = 0; p < placeCount; ++p) {
        int64_t bytes;
        std::string error;
        if (!programJointGroupFits(domain, g, p, bytes, error))
          continue;
        for (size_t k = 0; k < configurationCount; ++k) {
          checkpoint();
          if (assignments.size() >= size_t(options.maxPatterns))
            throw Stop{ConstraintSearchStatus::ModelLimit,
                       "group/place/configuration assignment limit"};
          auto x = boolean();
          Assignment a{g, p, k, x, {}, {}, {}};
          groupChoices[g].push_back(x);
          configChoices[k].push_back(x);
          coreChoices[{domain.places[p].binding.resource, k}].push_back(x);
          builder.AddImplication(x, used[k]);
          inputCounts[k] += LinearExpr(x) * int64_t(mode.inputs.size());
          outputCounts[k] += LinearExpr(x) * int64_t(mode.outputs.size());
          const ProgramJointSelection *hint = nullptr;
          for (auto &h : baseline.groups)
            if (h.group == g && h.place == p && h.configuration == k)
              hint = &h;
          builder.AddHint(x, hint ? 1 : 0);
          auto mi =
              builder.NewOptionalIntervalVar(begin[k], length[k], end[k], x);
          memories[domain.places[p].binding.memory].intervals.push_back(mi);
          memories[domain.places[p].binding.memory].demands.push_back(bytes);
          auto routes = [&](bool input) {
            for (auto value : (input ? mode.inputs : mode.outputs)) {
              LinearExpr sum;
              for (auto li : (input ? domain.places[p].inputLinks
                                    : domain.places[p].outputLinks)) {
                int64_t duration;
                std::string ignored;
                if (!programJointTransferNs(domain, v, value, li, duration,
                                            ignored)) {
                  if (programJointTransferSupported(domain, v, value, li,
                                                    ignored))
                    skippedPrice = true;
                  continue;
                }
                auto r = boolean();
                auto rb = time(), re = time();
                sum += r;
                auto interval =
                    builder.NewOptionalIntervalVar(rb, duration, re, r);
                builder.AddEquality(re, 0).OnlyEnforceIf(r.Not());
                completions[k].push_back(re);
                builder
                    .AddGreaterOrEqual(rb, LinearExpr(begin[k]) +
                                               domain.configurationNs)
                    .OnlyEnforceIf(r);
                builder.AddLessOrEqual(re, end[k]).OnlyEnforceIf(r);
                if (input)
                  builder.AddGreaterOrEqual(rb, ready[v][value])
                      .OnlyEnforceIf(r);
                else
                  builder.AddEquality(ready[v][value], re).OnlyEnforceIf(r);
                reserve(domain.target.links[li].resource, interval);
                (input ? a.inputs : a.outputs)
                    .push_back({value, li, r, rb, re});
                const ProgramJointTransfer *ht = nullptr;
                if (hint)
                  for (auto &h : (input ? hint->inputs : hint->outputs))
                    if (h.version == value && h.link == li)
                      ht = &h;
                builder.AddHint(r, ht ? 1 : 0);
                if (ht) {
                  builder.AddHint(rb, ht->startNs);
                  builder.AddHint(re, ht->endNs);
                }
              }
              builder.AddEquality(sum, x);
            }
          };
          routes(true);
          routes(false);
          std::vector<IntVar> ce;
          for (size_t i = 0; i < mode.instances.size(); ++i) {
            auto instance = mode.instances[i];
            auto cb = time(), finish = time();
            auto interval = builder.NewOptionalIntervalVar(
                cb, sourceVariant.computeNs[instance], finish, x);
            a.compute.push_back(cb);
            ce.push_back(finish);
            builder.AddEquality(finish, 0).OnlyEnforceIf(x.Not());
            completions[k].push_back(finish);
            builder
                .AddGreaterOrEqual(cb, LinearExpr(begin[k]) +
                                           domain.configurationNs)
                .OnlyEnforceIf(x);
            builder.AddLessOrEqual(finish, end[k]).OnlyEnforceIf(x);
            if (i)
              builder.AddGreaterOrEqual(cb, ce[i - 1]).OnlyEnforceIf(x);
            reserve(domain.places[p].binding.resource, interval);
            phaseExpr[v][instance] += LinearExpr(x) * int64_t(k);
            if (hint) {
              builder.AddHint(cb, hint->computeStartNs[i]);
              builder.AddHint(finish, hint->computeStartNs[i] +
                                          sourceVariant.computeNs[instance]);
            }
          }
          for (auto &r : a.inputs)
            builder.AddGreaterOrEqual(a.compute.front(), r.end)
                .OnlyEnforceIf(r.chosen);
          for (auto &r : a.outputs)
            builder.AddGreaterOrEqual(r.begin, ce.back())
                .OnlyEnforceIf(r.chosen);
          assignments.push_back(std::move(a));
        }
      }
      LinearExpr sum;
      for (auto x : groupChoices[g])
        sum += x;
      builder.AddEquality(sum, selected[g]);
    }
    for (size_t v = 0; v < variantCount; ++v) {
      for (size_t i = 0; i < covers[v].size(); ++i) {
        LinearExpr sum;
        for (auto g : covers[v][i])
          sum += selected[g];
        builder.AddEquality(sum, variant[v]);
        builder.AddEquality(phases[v][i], phaseExpr[v][i]);
      }
      // Internal RAW edges may stay within a selected group. Other RAW edges
      // require export/import through a later configuration; pure ordering
      // edges always retain the source barrier.
      std::set<std::pair<size_t, size_t>> raw;
      for (size_t value = 0; value < domain.variants[v].producers.size();
           ++value)
        if (domain.variants[v].producers[value] >= 0)
          for (auto reader : domain.variants[v].readers[value])
            raw.emplace(size_t(domain.variants[v].producers[value]), reader);
      int64_t intersectionWork = 0;
      for (auto edge : raw) {
        if (edge.first == edge.second)
          continue;
        auto &l = covers[v][edge.first], &r = covers[v][edge.second];
        intersectionWork += l.size() + r.size();
        if (intersectionWork > options.maxIntersectionWork)
          throw Stop{ConstraintSearchStatus::ModelLimit,
                     "RAW membership intersection limit"};
        std::vector<size_t> common;
        std::set_intersection(l.begin(), l.end(), r.begin(), r.end(),
                              std::back_inserter(common));
        LinearExpr together;
        for (auto g : common)
          together += selected[g];
        builder.AddGreaterOrEqual(phases[v][edge.second],
                                  LinearExpr(phases[v][edge.first]) +
                                      variant[v] - together);
      }
      for (auto edge : domain.variants[v].graph.orderEdges) {
        auto pt = domain.variants[v].graph.events[edge.producer].task,
             ct = domain.variants[v].graph.events[edge.consumer].task;
        for (size_t p = 0; p < covers[v].size(); ++p)
          if (domain.variants[v].graph.instances[p].task == pt)
            for (size_t c = 0; c < covers[v].size(); ++c)
              if (domain.variants[v].graph.instances[c].task == ct)
                builder.AddGreaterOrEqual(
                    phases[v][c], LinearExpr(phases[v][p]) + variant[v]);
      }
    }
    for (auto &entry : coreChoices)
      builder.AddAtMostOne(entry.second);
    for (size_t k = 0; k < configurationCount; ++k) {
      LinearExpr sum;
      for (auto x : configChoices[k])
        sum += x;
      builder.AddGreaterOrEqual(sum, used[k]);
      builder.AddLessOrEqual(inputCounts[k], domain.maxConfigurationInputs);
      builder.AddLessOrEqual(outputCounts[k], domain.maxConfigurationOutputs);
      builder.AddMaxEquality(end[k], completions[k]);
    }
    // Cumulative calendars include shared ancestors of compute/DMA resources.
    // Pairwise mutual exclusion would be incorrect for capacities above one.
    for (auto &entry : resources) {
      auto r = programResource(domain.target, entry.first);
      if (!r)
        throw Stop{ConstraintSearchStatus::InvalidInput, "unknown resource"};
      auto c = builder.AddCumulative(r->capacity);
      for (size_t i = 0; i < entry.second.intervals.size(); ++i)
        c.AddDemand(entry.second.intervals[i], entry.second.demands[i]);
    }
    for (auto &entry : memories) {
      auto c = builder.AddCumulative(
          programMemory(domain.target, entry.first)->capacityBytes);
      for (size_t i = 0; i < entry.second.intervals.size(); ++i)
        c.AddDemand(entry.second.intervals[i], entry.second.demands[i]);
    }
    auto makespan = time();
    builder.AddEquality(makespan, LinearExpr(end.back()) + domain.completeNs);
    builder.AddHint(makespan, horizonNs);
    builder.Minimize(makespan);
    checkpoint();
    auto proto = builder.Build();
    telemetry.variables = proto.variables_size();
    telemetry.constraints = proto.constraints_size();
    telemetry.modelBuildMs = elapsedMilliseconds();
    SatParameters parameters;
    // Reserve ten percent of the remaining backend budget for extracting and
    // independently checking the witness. Report actual solve time separately
    // from the caller's end-to-end wall limit.
    parameters.set_max_time_in_seconds(std::max(
        0.000001, (options.wallSeconds - elapsedMilliseconds() / 1000) * .9));
    parameters.set_num_search_workers(options.workers);
    parameters.set_random_seed(options.randomSeed);
    Model model;
    model.Add(NewSatParameters(parameters));
    auto solveStart = Clock::now();
    auto response = SolveCpModel(proto, &model);
    telemetry.solveMs =
        std::chrono::duration<double, std::milli>(Clock::now() - solveStart)
            .count();
    telemetry.branches = response.num_branches();
    telemetry.conflicts = response.num_conflicts();
    if (response.status() == CpSolverStatus::MODEL_INVALID)
      return fallback(ConstraintSearchStatus::InvalidInput,
                      response.solution_info());
    if (response.status() == CpSolverStatus::INFEASIBLE)
      return fallback(ConstraintSearchStatus::InvalidInput,
                      "encoding mismatch: finite model rejects its "
                      "independently checked baseline");
    if (response.status() != CpSolverStatus::FEASIBLE &&
        response.status() != CpSolverStatus::OPTIMAL) {
      telemetry.budgetExhausted = true;
      return fallback(
          ConstraintSearchStatus::Unknown,
          "no joint incumbent within budget; checked baseline retained");
    }
    ProgramJointSolution s;
    s.hasPlan = true;
    s.selection = "joint";
    const bool solverProvedOptimal =
        response.status() == CpSolverStatus::OPTIMAL;
    const bool fullDomainKnown =
        domain.complete && domain.costsComplete && !skippedPrice;
    s.status = solverProvedOptimal && fullDomainKnown
                   ? ConstraintSearchStatus::Optimal
                   : ConstraintSearchStatus::Feasible;
    s.message = "source-bound joint grouping, placement, copies, routes and "
                "configuration schedule";
    if (!fullDomainKnown)
      s.message += "; result covers only the generated, known-cost subset";
    s.makespanNs = SolutionIntegerValue(response, makespan);
    for (size_t v = 0; v < variantCount; ++v)
      if (SolutionBooleanValue(response, variant[v]))
        s.variant = v;
    for (size_t k = 0; k < configurationCount; ++k)
      if (SolutionBooleanValue(response, used[k])) {
        s.configurationStartNs.push_back(
            SolutionIntegerValue(response, begin[k]));
        s.configurationEndNs.push_back(SolutionIntegerValue(response, end[k]));
      }
    for (auto &a : assignments)
      if (SolutionBooleanValue(response, a.chosen)) {
        ProgramJointSelection x;
        x.group = a.group;
        x.place = a.place;
        x.configuration = a.configuration;
        for (auto cb : a.compute)
          x.computeStartNs.push_back(SolutionIntegerValue(response, cb));
        for (bool input : {true, false})
          for (auto &r : (input ? a.inputs : a.outputs))
            if (SolutionBooleanValue(response, r.chosen))
              (input ? x.inputs : x.outputs)
                  .push_back({r.version, r.link,
                              SolutionIntegerValue(response, r.begin),
                              SolutionIntegerValue(response, r.end)});
        s.groups.push_back(x);
      }
    telemetry.hasObjective = true;
    telemetry.objectiveNs = s.makespanNs;
    telemetry.bestBoundNs = response.best_objective_bound();
    telemetry.relativeGap =
        std::max(0.0, double(telemetry.objectiveNs) - telemetry.bestBoundNs) /
        std::max(1.0, std::abs(double(telemetry.objectiveNs)));
    telemetry.budgetExhausted = !solverProvedOptimal;
    ProgramGraph graph;
    std::string error;
    // Recompute the integer witness with the independent source/resource/cost
    // checker before accepting a solver objective or optimality declaration.
    if (!materializeProgramJointSolution(domain, s, graph, error) ||
        !checkProgramJointSolution(domain, s, graph, error))
      return fallback(ConstraintSearchStatus::InvalidInput,
                      "joint incumbent rejected: " + error);
    return finish(s);
  } catch (const Stop &stop) {
    telemetry.modelBuildMs = elapsedMilliseconds();
    return fallback(stop.status, stop.error);
  }
#endif
}
