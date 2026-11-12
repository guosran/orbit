#include "Taskflow/NPUConstraintSchedule.h"
#include "Taskflow/ProgramSearch.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <limits>
#include <thread>
using namespace orbit::taskflow;

int main() {
  unsigned checks = 0;
  auto require = [&](bool condition, const char *message) {
    ++checks;
    if (!condition)
      llvm::errs() << "FAILED: " << message << "\n";
    return condition;
  };
  ProgramSearchSpace space;
  for (const char *id : {"placement", "residency"}) {
    ProgramSearchDimension dimension;
    dimension.id = id;
    dimension.scope = "source";
    dimension.status = "searched";
    for (const char *choice : {"baseline", "alternative"}) {
      ProgramSearchChoice value;
      value.id = choice;
      value.binding = llvm::json::Object{{"selection", choice}};
      dimension.choices.push_back(std::move(value));
    }
    space.dimensions.push_back(std::move(dimension));
  }
  int calls = 0;
  auto evaluator = [&](const std::vector<int64_t> &a) {
    ++calls;
    ProgramSearchEvaluation evaluation;
    evaluation.feasible = true;
    // Both single changes are worse; only their joint selection improves.
    evaluation.costMs = a[0] && a[1] ? 1. : (!a[0] && !a[1] ? 10. : 20.);
    evaluation.result = llvm::json::Object{{"independent_checker", "passed"}};
    evaluation.sourceMLIR = "source identity";
    return evaluation;
  };
  ProgramSearchOptions options;
  options.wallSeconds = 5;
  auto result = searchProgramSpace(space, evaluator, options);
  if (!constraintSearchAvailable()) {
    if (!require(result.status == ConstraintSearchStatus::Unavailable &&
                     calls == 1 && result.hasBest &&
                     result.bestAssignment == std::vector<int64_t>({0, 0}),
                 "disabled build checks and retains only baseline"))
      return 1;
  } else {
    if (!require(result.status == ConstraintSearchStatus::Optimal &&
                     calls == 4 &&
                     result.bestAssignment == std::vector<int64_t>({1, 1}) &&
                     result.bestEvaluation.costMs == 1.,
                 "nonadditive joint optimum"))
      return 1;
    ProgramSearchConstraint constraint;
    constraint.dimensions = {0, 1};
    constraint.allowedTuples = {{0, 0}, {1, 1}};
    space.constraints.push_back(std::move(constraint));
    calls = 0;
    result = searchProgramSpace(space, evaluator, options);
    if (!require(result.status == ConstraintSearchStatus::Optimal && calls == 2,
                 "local compatibility excludes forbidden combinations"))
      return 1;
    options.maxEvaluations = 1;
    calls = 0;
    result = searchProgramSpace(space, evaluator, options);
    if (!require(result.status == ConstraintSearchStatus::Feasible &&
                     calls == 1 &&
                     result.bestAssignment == std::vector<int64_t>({0, 0}),
                 "evaluation cap preserves baseline without optimum claim"))
      return 1;
    if (!require(result.report.getArray("solve_trace")
                             ->back()
                             .getAsObject()
                             ->getString("purpose") ==
                         "remaining-domain satisfiability" &&
                     !result.report.getArray("solve_trace")
                          ->back()
                          .getAsObject()
                          ->get("proposal_coverage_penalty"),
                 "callback cap removes unneeded proposal optimization"))
      return 1;
    options.maxEvaluations = 2048;
    options.maxVariables = 1;
    calls = 0;
    result = searchProgramSpace(space, evaluator, options);
    if (!require(result.status == ConstraintSearchStatus::ModelLimit &&
                     !result.hasBest && calls == 0,
                 "oversized input domain rejected before indexed allocation or "
                 "callbacks"))
      return 1;
    options.maxVariables = 100000;
    options.maxTupleCells = 3;
    calls = 0;
    result = searchProgramSpace(space, evaluator, options);
    if (!require(result.status == ConstraintSearchStatus::ModelLimit &&
                     calls == 0,
                 "table payload cap counts tuple cells rather than one table "
                 "constraint"))
      return 1;
    options.maxTupleCells = 1000000;
    result = searchProgramSpace(
        space,
        [&](const std::vector<int64_t> &a) {
          auto e = evaluator(a);
          if (a[0])
            e.costMs.reset();
          return e;
        },
        options);
    if (!require(
            result.status == ConstraintSearchStatus::Feasible &&
                result.report.getBoolean("finite_domain_exhausted") == true &&
                result.report.getBoolean("finite_domain_optimality_proved") ==
                    false,
            "unknown full costs block finite optimality"))
      return 1;
    result = searchProgramSpace(
        space,
        [](const std::vector<int64_t> &) { return ProgramSearchEvaluation{}; },
        options);
    if (!require(result.status == ConstraintSearchStatus::Infeasible &&
                     !result.hasBest,
                 "infeasibility requires complete checked domain"))
      return 1;
    result = searchProgramSpace(
        space,
        [](const std::vector<int64_t> &) {
          ProgramSearchEvaluation e;
          e.costMs = std::numeric_limits<double>::infinity();
          return e;
        },
        options);
    bool sanitized = true;
    for (const auto &row : *result.report.getArray("evaluation_trace"))
      sanitized &= row.getAsObject()->get("cost_ms")->getAsNull().has_value();
    if (!require(
            sanitized && result.status == ConstraintSearchStatus::Infeasible,
            "infeasible provider rows cannot leak nonfinite costs into JSON"))
      return 1;
    ProgramSearchSpace wide;
    for (int i = 0; i < 8; ++i) {
      ProgramSearchDimension d;
      d.id = "binary-" + std::to_string(i);
      for (const char *name : {"zero", "one"}) {
        ProgramSearchChoice c;
        c.id = name;
        d.choices.push_back(std::move(c));
      }
      wide.dimensions.push_back(std::move(d));
    }
    options.maxEvaluations = 2;
    std::vector<std::vector<int64_t>> proposed;
    result = searchProgramSpace(
        wide,
        [&](const std::vector<int64_t> &a) {
          proposed.push_back(a);
          ProgramSearchEvaluation e;
          e.feasible = true;
          bool allOne = true;
          for (auto x : a)
            allOne &= x == 1;
          e.costMs = allOne ? 1. : 10.;
          return e;
        },
        options);
    if (!require(proposed.size() == 2 && result.bestEvaluation.costMs == 1. &&
                     result.status == ConstraintSearchStatus::Feasible,
                 "coverage proposal finds jointly improving all-one assignment "
                 "within two callbacks"))
      return 1;
    for (size_t i = 0; i < 8; ++i)
      if (!require(proposed[0][i] == 0 && proposed[1][i] == 1,
                   "all binary dimensions change before last-axis exhaustion"))
        return 1;
  }
  ProgramSearchSpace empty;
  options.wallSeconds = 0.001;
  result = searchProgramSpace(
      empty,
      [](const std::vector<int64_t> &) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        ProgramSearchEvaluation e;
        e.feasible = true;
        e.costMs = 1.;
        return e;
      },
      options);
  if (!require(
          result.hasBest &&
              result.report.getBoolean("budget_exhausted") == true &&
              result.report.getBoolean("returned_after_deadline") == true &&
              result.report.getNumber("deadline_overrun_ms").value_or(0) > 0 &&
              result.report.getNumber("callback_deadline_overrun_ms")
                      .value_or(0) > 0 &&
              (result.status == ConstraintSearchStatus::Unknown ||
               result.status == ConstraintSearchStatus::Unavailable),
          "nonpreemptive callback overrun remains visible and retains result"))
    return 1;
  // Invalid controls must be rejected before spending any evaluator work.
  for (int workers : {0, -1, 17, std::numeric_limits<int>::max()}) {
    ProgramSearchOptions invalid;
    invalid.workers = workers;
    const int before = calls;
    auto rejected = searchProgramSpace(space, evaluator, invalid);
    if (!require(rejected.status == ConstraintSearchStatus::InvalidInput &&
                     calls == before,
                 "worker range rejected before evaluator"))
      return 1;
  }
  {
    ProgramSearchOptions invalid;
    invalid.seed = -1;
    const int before = calls;
    auto rejected = searchProgramSpace(space, evaluator, invalid);
    if (!require(rejected.status == ConstraintSearchStatus::InvalidInput &&
                     calls == before,
                 "negative seed rejected before evaluator"))
      return 1;
    invalid.seed = std::numeric_limits<int>::max();
    invalid.maxEvaluations = 1;
    auto accepted = searchProgramSpace(space, evaluator, invalid);
    if (!require(accepted.hasBest && calls == before + 1 &&
                     accepted.status != ConstraintSearchStatus::InvalidInput,
                 "INT_MAX seed accepted with one worker and callback"))
      return 1;
  }
  if (constraintSearchAvailable()) {
    ConstraintSearchProblem cover;
    cover.taskCount = cover.tileCount = 1;
    ConstraintPattern singleton;
    singleton.tasks = {0};
    singleton.costNs = 1;
    cover.patterns.push_back(singleton);
    NPUScheduleProblem schedule;
    schedule.dmaResources.push_back({0, 2});
    schedule.tiles.push_back({0, 0, 2, 4, 0, 0, 0, 1});
    schedule.shards.push_back({0, 1, 1, 1, 4, 1, 1, {0}});
    for (int i = 0; i < 3; ++i) {
      ConstraintSearchOptions invalid;
      invalid.workers = i == 0 ? 0 : (i == 1 ? 17 : 1);
      invalid.randomSeed = i == 2 ? -1 : 0;
      // Invalid option values are rejected before constructing either model.
      auto coverResult = solveConstraintSearch(cover, invalid);
      auto scheduleResult = solveNPUConstraintSchedule(schedule, invalid);
      if (!require(coverResult.status == ConstraintSearchStatus::InvalidInput &&
                       scheduleResult.status ==
                           ConstraintSearchStatus::InvalidInput,
                   "typed cover/scheduler option bounds"))
        return 1;
    }
    ConstraintSearchOptions legal;
    legal.randomSeed = std::numeric_limits<int>::max();
    if (!require(solveConstraintSearch(cover, legal).status ==
                         ConstraintSearchStatus::Optimal &&
                     solveNPUConstraintSchedule(schedule, legal).status ==
                         ConstraintSearchStatus::Optimal,
                 "cover/scheduler INT_MAX seed with one worker"))
      return 1;
  }
  llvm::outs() << "{\"status\":\"passed\",\"checks\":" << checks << "}\n";
  return 0;
}
