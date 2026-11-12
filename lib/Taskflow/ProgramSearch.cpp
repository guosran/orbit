#include "Taskflow/ProgramSearch.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <utility>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif
#ifdef ORBIT_HAS_ORTOOLS
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"
#include "ortools/util/sorted_interval_list.h"
#endif

namespace orbit::taskflow {
ProgramSearchResult searchProgramSpace(const ProgramSearchSpace &space,
                                       const ProgramSearchEvaluator &evaluate,
                                       const ProgramSearchOptions &options) {
  using Clock = std::chrono::steady_clock;
  namespace j = llvm::json;
  const auto start = Clock::now();
  auto milliseconds = [](Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
  };
  auto remaining = [&]() {
    return options.wallSeconds - milliseconds(start) / 1000.;
  };
  ProgramSearchResult result;
  j::Array evaluations, solves, dimensions;
  int64_t checked = 0, feasible = 0, unknownCosts = 0, unknownFeasibility = 0;
  int64_t variableCount = 0, constraintCount = 0;
  double callbackMs = 0, solveMs = 0, buildMs = 0;
  double callbackDeadlineOverrunMs = 0;
  bool exhausted = false;
  std::vector<std::vector<int64_t>> exposures;
  int64_t selectorCount = 0, tupleCells = 0;
  auto finish = [&](ConstraintSearchStatus status, const std::string &reason) {
    result.status = status;
    auto &r = result.report;
    r["algorithm"] = "joint-cp-sat-branch-and-check";
    r["complete_plan_prepricing"] = false;
    r["objective"] = "full-checked-evaluator-cost-ms";
    r["solver_objective"] =
        "choice-coverage penalty only; not latency or a cost surrogate";
    r["proposal_policy"] =
        "minimize sum of prior exposure counts of selected local choices";
    r["status"] = constraintSearchStatusName(status);
    r["reason"] = reason;
    r["has_best"] = result.hasBest;
    r["finite_domain_exhausted"] = exhausted;
    r["all_feasibility_known"] = unknownFeasibility == 0;
    r["all_feasible_costs_known"] = unknownCosts == 0;
    r["finite_domain_optimality_proved"] =
        status == ConstraintSearchStatus::Optimal;
    r["evaluations"] = checked;
    r["feasible_evaluations"] = feasible;
    r["unknown_cost_evaluations"] = unknownCosts;
    r["unknown_feasibility_evaluations"] = unknownFeasibility;
    r["cache_hits"] = 0;
    r["cache_misses"] = checked;
    r["cache_policy"] = "evaluated assignments excluded; no repeat callbacks";
    r["model_variables"] = variableCount;
    r["model_constraints"] = constraintCount;
    r["model_build_ms"] = buildMs;
    r["solve_ms"] = solveMs;
    r["callback_ms"] = callbackMs;
    r["total_ms"] = milliseconds(start);
    r["wall_seconds"] = std::isfinite(options.wallSeconds)
                            ? j::Value(options.wallSeconds)
                            : j::Value(nullptr);
    r["budget_exhausted"] =
        std::isfinite(options.wallSeconds) && remaining() <= 0;
    r["callback_preemptible"] = false;
    r["deadline_overrun_ms"] =
        std::isfinite(options.wallSeconds)
            ? std::max(0., milliseconds(start) - options.wallSeconds * 1000.)
            : 0.;
    r["returned_after_deadline"] =
        std::isfinite(options.wallSeconds) && remaining() < 0;
    r["callback_deadline_overrun_ms"] = callbackDeadlineOverrunMs;
    r["max_evaluations"] = options.maxEvaluations;
    r["max_tuple_cells"] = options.maxTupleCells;
    r["compatibility_tuple_cells"] = tupleCells;
    r["dimensions"] = std::move(dimensions);
    j::Array coverage;
    for (size_t i = 0; i < exposures.size(); ++i) {
      j::Array counts;
      for (int64_t count : exposures[i])
        counts.push_back(count);
      coverage.push_back(j::Object{{"dimension", space.dimensions[i].id},
                                   {"choice_exposures", std::move(counts)}});
    }
    r["choice_coverage"] = std::move(coverage);
    r["evaluation_trace"] = std::move(evaluations);
    r["solve_trace"] = std::move(solves);
    if (result.hasBest) {
      j::Array assignment;
      for (auto value : result.bestAssignment)
        assignment.push_back(value);
      r["best_assignment"] = std::move(assignment);
      r["best_cost_ms"] = result.bestEvaluation.costMs
                              ? j::Value(*result.bestEvaluation.costMs)
                              : j::Value(nullptr);
    }
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
      r["process_peak_rss_kib"] = int64_t(usage.ru_maxrss / 1024);
#else
      r["process_peak_rss_kib"] = int64_t(usage.ru_maxrss);
#endif
    }
#endif
    // Include report construction in the observed soft-deadline accounting.
    const double finalMs = milliseconds(start);
    r["total_ms"] = finalMs;
    r["returned_after_deadline"] = std::isfinite(options.wallSeconds) &&
                                   finalMs > options.wallSeconds * 1000.;
    r["budget_exhausted"] = std::isfinite(options.wallSeconds) &&
                            finalMs >= options.wallSeconds * 1000.;
    r["deadline_overrun_ms"] =
        std::isfinite(options.wallSeconds)
            ? std::max(0., finalMs - options.wallSeconds * 1000.)
            : 0.;
    if (!std::isfinite(options.wallSeconds))
      r["wall_seconds"] = nullptr;
    return std::move(result);
  };

  if (!evaluate || !std::isfinite(options.wallSeconds) ||
      options.wallSeconds <= 0 || options.maxEvaluations <= 0 ||
      options.maxVariables <= 0 || options.maxConstraints <= 0 ||
      options.maxTupleCells <= 0 || options.workers <= 0 ||
      options.workers > 16 || options.seed < 0)
    return finish(ConstraintSearchStatus::InvalidInput,
                  "Invalid options or evaluator");
  // Bound all task/choice-indexed allocation and table payload before making
  // exposure arrays or allocating ordered ID-validation sets. Rejected input
  // domains have no model and need not invoke the baseline callback.
  if (space.dimensions.size() > static_cast<size_t>(options.maxVariables) ||
      space.constraints.size() > static_cast<size_t>(options.maxConstraints))
    return finish(ConstraintSearchStatus::ModelLimit,
                  "Domain dimensions exceed model limits");
  for (const auto &dimension : space.dimensions) {
    if (dimension.choices.size() <= 1)
      continue;
    if (dimension.choices.size() >
        static_cast<size_t>(options.maxVariables - selectorCount))
      return finish(ConstraintSearchStatus::ModelLimit,
                    "Choice selector count exceeds configured limit");
    selectorCount += static_cast<int64_t>(dimension.choices.size());
  }
  const int64_t baseConstraints =
      static_cast<int64_t>(space.constraints.size()) +
      (space.dimensions.empty() ? 0 : 1);
  if (static_cast<int64_t>(space.dimensions.size()) >
          options.maxVariables - selectorCount ||
      baseConstraints > options.maxConstraints ||
      selectorCount > (options.maxConstraints - baseConstraints) / 2)
    return finish(ConstraintSearchStatus::ModelLimit,
                  "Choice coverage model exceeds configured limit");
  for (const auto &constraint : space.constraints) {
    const size_t arity = constraint.dimensions.size();
    if (arity == 0)
      return finish(ConstraintSearchStatus::InvalidInput,
                    "Empty compatibility scope");
    if (arity > static_cast<size_t>(options.maxTupleCells) ||
        constraint.allowedTuples.size() >
            static_cast<size_t>((options.maxTupleCells - tupleCells) /
                                static_cast<int64_t>(arity)))
      return finish(ConstraintSearchStatus::ModelLimit,
                    "Compatibility tuple cell limit exceeded");
    tupleCells += static_cast<int64_t>(arity) *
                  static_cast<int64_t>(constraint.allowedTuples.size());
  }
  std::set<std::string> ids;
  for (const auto &dimension : space.dimensions) {
    if (dimension.id.empty() || !ids.insert(dimension.id).second ||
        dimension.choices.empty())
      return finish(ConstraintSearchStatus::InvalidInput,
                    "Dimension IDs must be unique and choice domains nonempty");
    std::set<std::string> choiceIds;
    for (const auto &choice : dimension.choices)
      if (choice.id.empty() || !choiceIds.insert(choice.id).second)
        return finish(
            ConstraintSearchStatus::InvalidInput,
            "Choice IDs must be nonempty and unique within a dimension");
    dimensions.push_back(
        j::Object{{"id", dimension.id},
                  {"scope", dimension.scope},
                  {"status", dimension.status},
                  {"reason", dimension.reason},
                  {"choices", int64_t(dimension.choices.size())}});
    if (remaining() <= 0)
      return finish(ConstraintSearchStatus::Unknown,
                    "Deadline during domain validation");
  }
  for (const auto &constraint : space.constraints) {
    if (constraint.dimensions.empty())
      return finish(ConstraintSearchStatus::InvalidInput,
                    "Empty compatibility scope");
    std::set<size_t> scope;
    for (size_t dimension : constraint.dimensions)
      if (dimension >= space.dimensions.size() ||
          !scope.insert(dimension).second)
        return finish(ConstraintSearchStatus::InvalidInput,
                      "Invalid compatibility scope");
    bool admitsBaseline = false;
    for (const auto &tuple : constraint.allowedTuples) {
      if (tuple.size() != constraint.dimensions.size())
        return finish(ConstraintSearchStatus::InvalidInput,
                      "Compatibility tuple arity mismatch");
      bool zero = true;
      for (size_t i = 0; i < tuple.size(); ++i) {
        if (tuple[i] < 0 ||
            static_cast<size_t>(tuple[i]) >=
                space.dimensions[constraint.dimensions[i]].choices.size())
          return finish(ConstraintSearchStatus::InvalidInput,
                        "Compatibility tuple outside domain");
        zero &= tuple[i] == 0;
      }
      admitsBaseline |= zero;
      if (remaining() <= 0)
        return finish(ConstraintSearchStatus::Unknown,
                      "Deadline during compatibility validation");
    }
    if (!admitsBaseline)
      return finish(
          ConstraintSearchStatus::InvalidInput,
          "All-zero baseline must be admitted by compatibility tables");
  }

  for (const auto &dimension : space.dimensions)
    exposures.emplace_back(dimension.choices.size(), 0);

  auto checkAssignment = [&](const std::vector<int64_t> &assignment) {
    for (size_t i = 0; i < assignment.size(); ++i)
      ++exposures[i][static_cast<size_t>(assignment[i])];
    const auto evaluationStart = Clock::now();
    auto evaluation = evaluate(assignment);
    const double duration = milliseconds(evaluationStart);
    callbackDeadlineOverrunMs = std::max(
        callbackDeadlineOverrunMs,
        std::max(0., milliseconds(start) - options.wallSeconds * 1000.));
    callbackMs += duration;
    ++checked;
    if (!evaluation.feasibilityKnown)
      ++unknownFeasibility;
    if (evaluation.feasible && evaluation.feasibilityKnown)
      ++feasible;
    if (evaluation.costMs &&
        (!std::isfinite(*evaluation.costMs) || *evaluation.costMs < 0)) {
      evaluation.costMs.reset();
      evaluation.message += "; invalid cost treated as unknown";
    }
    if (!evaluation.feasible)
      evaluation.costMs.reset();
    if (evaluation.feasible && evaluation.feasibilityKnown &&
        !evaluation.costMs)
      ++unknownCosts;
    j::Array choices;
    for (auto value : assignment)
      choices.push_back(value);
    evaluations.push_back(
        j::Object{{"assignment", std::move(choices)},
                  {"call", checked},
                  {"model_variables", variableCount},
                  {"model_constraints", constraintCount},
                  {"elapsed_ms", milliseconds(start)},
                  {"feasible", evaluation.feasible},
                  {"feasibility_known", evaluation.feasibilityKnown},
                  {"cost_ms", evaluation.costMs ? j::Value(*evaluation.costMs)
                                                : j::Value(nullptr)},
                  {"callback_ms", duration},
                  {"message", evaluation.message}});
    if (evaluation.feasible && evaluation.feasibilityKnown &&
        (!result.hasBest ||
         (evaluation.costMs &&
          (!result.bestEvaluation.costMs ||
           *evaluation.costMs < *result.bestEvaluation.costMs)))) {
      result.hasBest = true;
      result.bestAssignment = assignment;
      result.bestEvaluation = std::move(evaluation);
    }
  };
  std::vector<int64_t> baseline(space.dimensions.size(), 0);
  if (remaining() <= 0)
    return finish(ConstraintSearchStatus::Unknown,
                  "Deadline before baseline check");
  checkAssignment(baseline);

#ifndef ORBIT_HAS_ORTOOLS
  return finish(ConstraintSearchStatus::Unavailable,
                "Built without ORBIT_HAS_ORTOOLS; baseline checked only");
#else
  using namespace operations_research;
  using namespace operations_research::sat;
  if (remaining() <= 0)
    return finish(ConstraintSearchStatus::Unknown,
                  "Baseline callback exhausted soft deadline");
  if (space.dimensions.size() > static_cast<size_t>(options.maxVariables) ||
      space.constraints.size() >= static_cast<size_t>(options.maxConstraints))
    return finish(ConstraintSearchStatus::ModelLimit,
                  "Initial model exceeds configured limit");
  CpModelBuilder builder;
  std::vector<IntVar> variables;
  std::vector<std::vector<BoolVar>> selectors;
  auto buildStart = Clock::now();
  size_t buildOperations = 0;
  auto buildDeadlineReached = [&]() {
    if ((++buildOperations % 256) != 0 || remaining() > 0)
      return false;
    buildMs += milliseconds(buildStart);
    return true;
  };
  for (const auto &dimension : space.dimensions) {
    variables.push_back(builder.NewIntVar(
        Domain(0, static_cast<int64_t>(dimension.choices.size() - 1))));
    builder.AddHint(variables.back(), 0);
    selectors.emplace_back();
    if (dimension.choices.size() > 1)
      for (size_t c = 0; c < dimension.choices.size(); ++c) {
        BoolVar chosen = builder.NewBoolVar();
        builder.AddEquality(variables.back(), static_cast<int64_t>(c))
            .OnlyEnforceIf(chosen);
        builder.AddNotEqual(variables.back(), static_cast<int64_t>(c))
            .OnlyEnforceIf(chosen.Not());
        selectors.back().push_back(chosen);
        if (buildDeadlineReached())
          return finish(ConstraintSearchStatus::Unknown,
                        "Deadline during selector construction");
      }
    if (buildDeadlineReached())
      return finish(ConstraintSearchStatus::Unknown,
                    "Deadline during dimension construction");
  }
  for (const auto &constraint : space.constraints) {
    std::vector<IntVar> scope;
    for (size_t dimension : constraint.dimensions)
      scope.push_back(variables[dimension]);
    auto table = builder.AddAllowedAssignments(scope);
    for (const auto &tuple : constraint.allowedTuples) {
      table.AddTuple(tuple);
      if (buildDeadlineReached())
        return finish(ConstraintSearchStatus::Unknown,
                      "Deadline during compatibility construction");
    }
  }
  if (!variables.empty())
    builder.AddForbiddenAssignments(variables).AddTuple(baseline);
  buildMs += milliseconds(buildStart);
  auto exhaustedStatus = [&]() {
    if (unknownFeasibility || unknownCosts)
      return result.hasBest ? ConstraintSearchStatus::Feasible
                            : ConstraintSearchStatus::Unknown;
    return result.hasBest ? ConstraintSearchStatus::Optimal
                          : ConstraintSearchStatus::Infeasible;
  };
  if (variables.empty()) {
    exhausted = true;
    return finish(exhaustedStatus(),
                  "Empty-dimensional domain checked completely");
  }
  for (;;) {
    if (remaining() <= 0)
      return finish(ConstraintSearchStatus::Unknown,
                    "Shared wall quota exhausted");
    const auto snapshotStart = Clock::now();
    if (!variables.empty() &&
        checked > (std::numeric_limits<int64_t>::max() / 4) /
                      static_cast<int64_t>(variables.size()))
      return finish(ConstraintSearchStatus::ModelLimit,
                    "Coverage objective coefficient magnitude limit");
    const bool checkingExhaustion = checked >= options.maxEvaluations;
    if (checkingExhaustion)
      builder.ClearObjective();
    else {
      LinearExpr coveragePenalty;
      for (size_t i = 0; i < selectors.size(); ++i)
        for (size_t c = 0; c < selectors[i].size(); ++c) {
          coveragePenalty += LinearExpr(selectors[i][c]) * exposures[i][c];
          if (((++buildOperations) % 256) == 0 && remaining() <= 0) {
            buildMs += milliseconds(snapshotStart);
            return finish(ConstraintSearchStatus::Unknown,
                          "Deadline during coverage objective construction");
          }
        }
      builder.Minimize(coveragePenalty);
    }
    const CpModelProto &proto = builder.Build();
    variableCount = proto.variables_size();
    constraintCount = proto.constraints_size();
    buildMs += milliseconds(snapshotStart);
    if (variableCount > options.maxVariables ||
        constraintCount > options.maxConstraints)
      return finish(ConstraintSearchStatus::ModelLimit,
                    "Accumulated model limit reached");
    if (remaining() <= 0)
      return finish(ConstraintSearchStatus::Unknown,
                    "Deadline during model snapshot");
    SatParameters parameters;
    parameters.set_max_time_in_seconds(remaining());
    parameters.set_num_search_workers(options.workers);
    parameters.set_random_seed(options.seed);
    parameters.set_log_search_progress(false);
    Model model;
    model.Add(NewSatParameters(parameters));
    const auto solveStart = Clock::now();
    const auto response = SolveCpModel(proto, &model);
    const double duration = milliseconds(solveStart);
    solveMs += duration;
    solves.push_back(j::Object{
        {"call", int64_t(solves.size() + 1)},
        {"status", CpSolverStatus_Name(response.status())},
        {"purpose", checkingExhaustion ? "remaining-domain satisfiability"
                                       : "choice coverage"},
        {"solve_ms", duration},
        {"variables", variableCount},
        {"constraints", constraintCount},
        {"branches", response.num_branches()},
        {"conflicts", response.num_conflicts()},
        {"remaining_wall_ms", std::max(0., remaining() * 1000.)}});
    if (!checkingExhaustion && (response.status() == CpSolverStatus::FEASIBLE ||
                                response.status() == CpSolverStatus::OPTIMAL))
      solves.back().getAsObject()->operator[]("proposal_coverage_penalty") =
          response.objective_value();
    if (response.status() == CpSolverStatus::MODEL_INVALID)
      return finish(ConstraintSearchStatus::InvalidInput,
                    response.solution_info());
    if (response.status() == CpSolverStatus::INFEASIBLE) {
      exhausted = true;
      return finish(exhaustedStatus(), "Every admitted assignment checked");
    }
    if (response.status() != CpSolverStatus::FEASIBLE &&
        response.status() != CpSolverStatus::OPTIMAL)
      return finish(ConstraintSearchStatus::Unknown,
                    "CP-SAT returned no unchecked assignment");
    if (checked >= options.maxEvaluations)
      return finish(
          result.hasBest ? ConstraintSearchStatus::Feasible
                         : ConstraintSearchStatus::Unknown,
          "Evaluation quota reached; unexamined compatible assignments remain");
    if (remaining() <= 0)
      return finish(ConstraintSearchStatus::Unknown,
                    "Deadline before checker callback");
    std::vector<int64_t> assignment;
    for (IntVar variable : variables)
      assignment.push_back(SolutionIntegerValue(response, variable));
    checkAssignment(assignment);
    const auto exclusionStart = Clock::now();
    builder.AddForbiddenAssignments(variables).AddTuple(assignment);
    buildMs += milliseconds(exclusionStart);
  }
#endif
}
} // namespace orbit::taskflow
