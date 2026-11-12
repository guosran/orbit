#include "Taskflow/ProgramBeamSearch.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <limits>
#include <set>
#include <thread>
using namespace orbit::taskflow;

static ProgramSearchSpace binarySpace(size_t dimensions) {
  ProgramSearchSpace space;
  for (size_t i = 0; i < dimensions; ++i) {
    ProgramSearchDimension d;
    d.id = "dimension-" + std::to_string(i);
    d.status = "searched";
    d.scope = "contract";
    for (const char *id : {"baseline", "alternative"}) {
      ProgramSearchChoice c;
      c.id = id;
      d.choices.push_back(std::move(c));
    }
    space.dimensions.push_back(std::move(d));
  }
  return space;
}
int main() {
  unsigned checks = 0;
  auto require = [&](bool condition, const char *message) {
    ++checks;
    if (!condition)
      llvm::errs() << "FAILED: " << message << "\n";
    return condition;
  };
  auto space = binarySpace(3);
  ProgramSearchOptions options;
  options.wallSeconds = 5;
  int calls = 0;
  std::set<std::vector<int64_t>> seen;
  auto evaluator = [&](const std::vector<int64_t> &a) {
    ++calls;
    seen.insert(a);
    ProgramSearchEvaluation e;
    e.feasible = true;
    int sum = 0;
    for (auto v : a)
      sum += v;
    e.costMs = sum == 3 ? 1. : (sum == 0 ? 5. : 6.);
    e.result = llvm::json::Object{{"complete_checked_cost_ms", *e.costMs}};
    e.sourceMLIR = "checked actual source";
    return e;
  };
  auto r = searchProgramSpaceBeam(space, evaluator, options, 1, 10);
  if (!require(
          r.status == ConstraintSearchStatus::Feasible && r.hasBest &&
              r.bestEvaluation.costMs == 5. && calls == 4 && seen.size() == 4,
          "width-one beam gets stuck at checked nonadditive local optimum"))
    return 1;
  if (!require(r.report.getBoolean("finite_domain_optimality_proved") == false,
               "neighbor exhaustion never masquerades as finite-domain proof"))
    return 1;
  calls = 0;
  seen.clear();
  r = searchProgramSpaceBeam(space, evaluator, options, 4, 4, 1);
  if (!require(r.status == ConstraintSearchStatus::Feasible &&
                   r.bestEvaluation.costMs == 1. && calls == 8 &&
                   seen.size() == 8 && r.bestEvaluation.result.getAsObject() &&
                   !r.bestEvaluation.sourceMLIR.empty(),
               "wider existing selectSearchBeam crosses trap and keeps "
               "complete evidence"))
    return 1;
  if (!require(r.report.getArray("beam_selection_history")->size() > 1,
               "selection history retained"))
    return 1;
  options.maxEvaluations = 1;
  calls = 0;
  r = searchProgramSpaceBeam(space, evaluator, options, 4, 5);
  if (!require(calls == 1 && r.hasBest && r.bestEvaluation.costMs == 5. &&
                   r.report.getBoolean("finite_domain_optimality_proved") ==
                       false,
               "evaluation quota retains baseline without optimum claim"))
    return 1;
  options.maxEvaluations = 2048;
  calls = 0;
  r = searchProgramSpaceBeam(space, evaluator, options, 1, 0);
  if (!require(calls == 1 && r.status == ConstraintSearchStatus::Feasible,
               "zero rounds evaluates baseline only"))
    return 1;
  r = searchProgramSpaceBeam(
      space,
      [](const std::vector<int64_t> &) {
        ProgramSearchEvaluation e;
        e.feasible = true;
        return e;
      },
      options, 2, 3);
  if (!require(r.hasBest && !r.bestEvaluation.costMs &&
                   r.status == ConstraintSearchStatus::Feasible &&
                   r.report.getInteger("unknown_cost_evaluations").value_or(0) >
                       0,
               "unknown costs stay unknown while checked fallback survives"))
    return 1;
  auto paired = binarySpace(2);
  ProgramSearchConstraint table;
  table.dimensions = {0, 1};
  table.allowedTuples = {{0, 0}, {1, 1}};
  paired.constraints.push_back(std::move(table));
  calls = 0;
  r = searchProgramSpaceBeam(paired, evaluator, options, 2, 4);
  if (!require(calls == 1 &&
                   r.report.getInteger("compatibility_rejections") == 2 &&
                   r.status == ConstraintSearchStatus::Feasible,
               "legal single-dimension neighborhood reports absence of "
               "implicit repair"))
    return 1;
  auto compound = binarySpace(1);
  calls = 0;
  r = searchProgramSpaceBeam(
      compound,
      [&](const std::vector<int64_t> &a) {
        ++calls;
        ProgramSearchEvaluation e;
        e.feasible = true;
        e.costMs = a[0] ? 1. : 5.;
        return e;
      },
      options, 1, 3);
  if (!require(calls == 2 && r.bestEvaluation.costMs == 1.,
               "declared compound choice changes coupled fields atomically"))
    return 1;
  paired.constraints[0].allowedTuples = {{0, 0}, {2, 1}};
  calls = 0;
  r = searchProgramSpaceBeam(paired, evaluator, options, 2, 4);
  if (!require(r.status == ConstraintSearchStatus::InvalidInput && calls == 0,
               "out-of-domain table rejected before callback"))
    return 1;
  options.wallSeconds = .001;
  r = searchProgramSpaceBeam(
      binarySpace(0),
      [](const std::vector<int64_t> &) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        ProgramSearchEvaluation e;
        e.feasible = true;
        e.costMs = 1.;
        return e;
      },
      options, 1, 3);
  if (!require(
          r.hasBest && r.status == ConstraintSearchStatus::Unknown &&
              r.report.getBoolean("returned_after_deadline") == true &&
              r.report.getNumber("callback_deadline_overrun_ms").value_or(0) >
                  0,
          "shared soft deadline records nonpreemptive callback overshoot"))
    return 1;
  // Invalid controls must be rejected before spending any evaluator work.
  for (int workers : {0, -1, 17, std::numeric_limits<int>::max()}) {
    ProgramSearchOptions invalid;
    invalid.workers = workers;
    const int before = calls;
    auto rejected = searchProgramSpaceBeam(space, evaluator, invalid, 1, 1);
    if (!require(rejected.status == ConstraintSearchStatus::InvalidInput &&
                     calls == before,
                 "worker range rejected before evaluator"))
      return 1;
  }
  {
    ProgramSearchOptions invalid;
    invalid.seed = -1;
    const int before = calls;
    auto rejected = searchProgramSpaceBeam(space, evaluator, invalid, 1, 1);
    if (!require(rejected.status == ConstraintSearchStatus::InvalidInput &&
                     calls == before,
                 "negative seed rejected before evaluator"))
      return 1;
    invalid.seed = std::numeric_limits<int>::max();
    invalid.maxEvaluations = 1;
    auto accepted = searchProgramSpaceBeam(space, evaluator, invalid, 1, 1);
    if (!require(accepted.hasBest && calls == before + 1 &&
                     accepted.status != ConstraintSearchStatus::InvalidInput,
                 "INT_MAX seed accepted with one worker and callback"))
      return 1;
  }
  llvm::outs() << "{\"status\":\"passed\",\"checks\":" << checks << "}\n";
  return 0;
}
