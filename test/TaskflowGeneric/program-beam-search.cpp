#include "Taskflow/ProgramBeamSearch.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
#include <stdexcept>
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
  // Both priced endpoints are separated by checked-feasible unpriced states.
  // This is a graph-search contract, not an assumed cost for the middle states.
  options.wallSeconds = 5;
  auto barrier = binarySpace(4);
  calls = 0;
  seen.clear();
  auto partial = [&](const std::vector<int64_t> &a) {
    ++calls;
    seen.insert(a);
    ProgramSearchEvaluation e;
    e.feasible = true;
    const int sum = std::count(a.begin(), a.end(), 1);
    if (sum == 0 || sum == 4)
      e.costMs = sum == 0 ? 8. : 2.;
    return e;
  };
  r = searchProgramSpaceBeam(barrier, partial, options, 1, 30);
  if (!require(r.bestEvaluation.costMs == 2. && calls <= 16 &&
                   seen.size() == size_t(calls) &&
                   r.report.getInteger("unknown_expanded").value_or(0) > 0 &&
                   r.report.getBoolean("finite_domain_optimality_proved") ==
                       false,
               "bounded unpriced FIFO crosses a multi-step pricing barrier"))
    return 1;
  bool nullObserved = false;
  for (const auto &entry : *r.report.getArray("evaluation_trace")) {
    const auto *row = entry.getAsObject();
    if (!row->getNumber("cost_ms"))
      nullObserved = true;
  }
  if (!require(nullObserved,
               "unpriced callbacks retain null costs in evidence"))
    return 1;
  ProgramBeamExplorationOptions disabled;
  disabled.unknownSlots = 0;
  calls = 0;
  r = searchProgramSpaceBeam(barrier, partial, options, 1, 30, 0, disabled);
  if (!require(r.bestEvaluation.costMs == 8. && calls == 5,
               "disabled exploration reproduces the scored-only barrier"))
    return 1;
  ProgramBeamExplorationOptions bounded;
  bounded.maxPendingStates = 1;
  options.maxEvaluations = 6;
  calls = 0;
  r = searchProgramSpaceBeam(barrier, partial, options, 1, 30, 0, bounded);
  if (!require(calls <= 6 && r.report.getInteger("unknown_pending_peak") == 1 &&
                   r.report.getInteger("unknown_frontier_dropped").value_or(0) >
                       0,
               "frontier memory and evaluator quotas remain bounded"))
    return 1;
  options.maxEvaluations = 2048;
  for (bool unknownFeasibility : {false, true}) {
    calls = 0;
    r = searchProgramSpaceBeam(
        barrier,
        [&](const std::vector<int64_t> &a) {
          ++calls;
          ProgramSearchEvaluation e;
          const bool baseline = std::count(a.begin(), a.end(), 1) == 0;
          e.feasible = baseline || unknownFeasibility;
          e.feasibilityKnown = baseline || !unknownFeasibility;
          if (baseline)
            e.costMs = 8.;
          return e;
        },
        options, 1, 30);
    if (!require(
            calls == 5 && r.report.getInteger("unknown_enqueued") == 0,
            "unknown feasibility and infeasible states never enter frontier"))
      return 1;
  }
  auto fairSpace = binarySpace(3);
  ProgramSearchConstraint fork;
  fork.dimensions = {0, 1, 2};
  fork.allowedTuples = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                        {0, 0, 1}, {0, 1, 1}, {1, 1, 1}};
  fairSpace.constraints.push_back(std::move(fork));
  r = searchProgramSpaceBeam(
      fairSpace,
      [](const std::vector<int64_t> &a) {
        ProgramSearchEvaluation e;
        e.feasible = true;
        if (!a[2])
          e.costMs = a[1] ? 1. : (a[0] ? 4. : 8.);
        return e;
      },
      options, 1, 3);
  const auto *fairHistory = r.report.getArray("beam_selection_history");
  const auto *selected = (*fairHistory)[2].getAsObject()->getArray("selected");
  const auto *selectedAssignment =
      (*selected)[0].getAsObject()->getArray("assignment");
  if (!require(
          r.bestEvaluation.costMs == 1. &&
              (*selectedAssignment)[0].getAsInteger() == 1 &&
              (*selectedAssignment)[2].getAsInteger() == 0,
          "width-one FIFO preserves and fairly schedules pending priced work"))
    return 1;
  for (auto bad : {ProgramBeamExplorationOptions{-1, 256},
                   ProgramBeamExplorationOptions{2, 256},
                   ProgramBeamExplorationOptions{1, -1}}) {
    calls = 0;
    r = searchProgramSpaceBeam(barrier, partial, options, 1, 30, 0, bad);
    if (!require(r.status == ConstraintSearchStatus::InvalidInput && calls == 0,
                 "invalid exploration bounds rejected before callback"))
      return 1;
  }
  ProgramSearchOptions unlimited;
  unlimited.wallSeconds = std::numeric_limits<double>::infinity();
  unlimited.maxEvaluations = 0;
  auto converged =
      searchProgramSpaceBeam(binarySpace(3), evaluator, unlimited, 4, -1);
  if (!require(converged.hasBest && converged.bestEvaluation.costMs == 1. &&
                   converged.report.getInteger("evaluations") == 8 &&
                   converged.report.get("wall_seconds")->getAsNull() &&
                   converged.report.get("max_evaluations")->getAsNull() &&
                   converged.report.get("max_rounds")->getAsNull() &&
                   converged.report.getBoolean("budget_exhausted") == false,
               "unlimited beam terminates by checked-neighborhood convergence"))
    return 1;
  // Trace persistence must not change candidate order, FIFO exploration or
  // convergence. Compare the full semantic rows, excluding elapsed timings.
  auto semanticRow = [](const llvm::json::Value &row) {
    std::string text;
    llvm::raw_string_ostream output(text);
    for (const char *key : {"assignment", "call", "feasible",
                            "feasibility_known", "cost_ms", "message"})
      output << *row.getAsObject()->get(key) << "\n";
    output.flush();
    return text;
  };
  for (bool priced : {true, false}) {
    const auto &domain = priced ? space : barrier;
    ProgramSearchEvaluator objective = priced
                                           ? ProgramSearchEvaluator(evaluator)
                                           : ProgramSearchEvaluator(partial);
    auto retained = searchProgramSpaceBeam(domain, objective, unlimited, 4, -1);
    std::vector<std::string> written;
    ProgramBeamTraceOptions streamed;
    streamed.retainInMemory = false;
    streamed.sink = [&](const llvm::json::Value &row) {
      written.push_back(semanticRow(row));
    };
    auto disk = searchProgramSpaceBeam(domain, objective, unlimited, 4, -1, 0,
                                       {}, streamed);
    std::vector<std::string> expected;
    for (const auto &row : *retained.report.getArray("evaluation_trace"))
      expected.push_back(semanticRow(row));
    if (!require(expected == written && disk.status == retained.status &&
                     disk.bestAssignment == retained.bestAssignment &&
                     disk.bestEvaluation.costMs ==
                         retained.bestEvaluation.costMs &&
                     disk.report.getInteger("rounds") ==
                         retained.report.getInteger("rounds") &&
                     disk.report.getInteger("cache_hits") ==
                         retained.report.getInteger("cache_hits"),
                 "streaming preserves every priced/unpriced candidate and "
                 "convergence"))
      return 1;
    if (!require(
            disk.report.getArray("evaluation_trace")->empty() &&
                disk.report.getBoolean("evaluation_trace_streamed") == true &&
                disk.report.getBoolean("evaluation_trace_retained_in_memory") ==
                    false &&
                disk.report.getInteger("evaluation_trace_rows") ==
                    int64_t(written.size()),
            "streamed rows are complete without in-memory JSON duplication"))
      return 1;
  }
  ProgramBeamTraceOptions lost;
  lost.retainInMemory = false;
  calls = 0;
  auto missingSink =
      searchProgramSpaceBeam(space, evaluator, unlimited, 4, -1, 0, {}, lost);
  if (!require(missingSink.status == ConstraintSearchStatus::InvalidInput &&
                   calls == 0,
               "disabling retained evidence requires an explicit durable sink"))
    return 1;
  lost.sink = [](const llvm::json::Value &) {
    throw std::runtime_error("simulated full trace device");
  };
  bool failedSink = false;
  try {
    searchProgramSpaceBeam(space, evaluator, unlimited, 4, -1, 0, {}, lost);
  } catch (const std::runtime_error &) {
    failedSink = true;
  }
  if (!require(failedSink,
               "trace write failure cannot claim a completed search"))
    return 1;
  // Unequal domains cross byte boundaries and include zero-bit singleton
  // dimensions. Independently enumerate all states to catch packed-key aliasing
  // that would silently discard legitimate search assignments.
  for (const std::vector<size_t> cardinalities :
       {std::vector<size_t>{1, 3, 17, 1, 5, 2},
        std::vector<size_t>{1, 257, 1, 2}}) {
    ProgramSearchSpace unequal;
    std::set<std::vector<int64_t>> expected{{}}, actual;
    size_t bits = 0;
    for (size_t count : cardinalities) {
      ProgramSearchDimension dimension;
      dimension.id = "unequal-" + std::to_string(unequal.dimensions.size());
      for (size_t i = 0; i < count; ++i) {
        ProgramSearchChoice choice;
        choice.id = std::to_string(i);
        dimension.choices.push_back(std::move(choice));
      }
      unequal.dimensions.push_back(std::move(dimension));
      std::set<std::vector<int64_t>> next;
      for (auto prefix : expected)
        for (size_t i = 0; i < count; ++i) {
          auto assignment = prefix;
          assignment.push_back(i);
          next.insert(std::move(assignment));
        }
      expected = std::move(next);
      for (size_t maximum = count - 1; maximum; maximum >>= 1)
        ++bits;
    }
    auto packed = searchProgramSpaceBeam(
        unequal,
        [&](const std::vector<int64_t> &assignment) {
          actual.insert(assignment);
          ProgramSearchEvaluation evaluation;
          evaluation.feasible = true;
          evaluation.costMs = 1.;
          return evaluation;
        },
        unlimited, expected.size(), -1);
    if (!require(actual == expected &&
                     packed.report.getInteger("evaluations") ==
                         int64_t(expected.size()) &&
                     packed.report.getInteger("assignment_key_bytes") ==
                         int64_t((bits + 7) / 8),
                 "lossless packed deduplication visits every unequal-domain "
                 "state, including wide and fixed dimensions"))
      return 1;
  }
  {
    // This joint matrix cannot be represented as independent per-choice fees.
    // One coordinate pass reaches (0,1)=8; only later revisiting dimension0
    // reaches (1,1)=0. Initialization must not become a stopping criterion.
    auto jointSpace = binarySpace(2);
    auto jointCost = [](const std::vector<int64_t> &assignment) {
      const double costs[2][2] = {{10., 8.}, {11., 0.}};
      ProgramSearchEvaluation evaluation;
      evaluation.feasible = true;
      evaluation.costMs = costs[assignment[0]][assignment[1]];
      return evaluation;
    };
    std::vector<std::vector<int64_t>> initialized;
    auto coordinate = initializeProgramCoordinates(
        jointSpace, jointCost, {},
        [&](const auto &a, const auto &) { initialized.push_back(a); });
    if (!require(
            coordinate.valid && coordinate.hasPricedSeed &&
                coordinate.assignment == std::vector<int64_t>({0, 1}) &&
                coordinate.evaluation.costMs == 8. &&
                coordinate.startCostMs == 10. && coordinate.evaluations == 3 &&
                coordinate.improvedCoordinates == 1 &&
                initialized ==
                    std::vector<std::vector<int64_t>>({{0, 0}, {1, 0}, {0, 1}}),
            "one coordinate pass checks joint choices from a fixed "
            "per-dimension incumbent"))
      return 1;
    ProgramBeamInitialStates initializedSeed;
    initializedSeed.assignments.push_back(coordinate.assignment);
    auto closed = searchProgramSpaceBeam(jointSpace, jointCost, unlimited, 1,
                                         -1, 0, {}, {}, initializedSeed);
    if (!require(
            closed.hasBest && closed.bestEvaluation.costMs == 0. &&
                closed.bestAssignment == std::vector<int64_t>({1, 1}) &&
                closed.report.getInteger("evaluations").value_or(0) >= 4,
            "full neighborhood remains active after coordinate initialization"))
      return 1;
    auto guarded = initializeProgramCoordinates(jointSpace, [&](const auto &a) {
      auto evaluation = jointCost(a);
      if (a != std::vector<int64_t>({0, 0})) {
        evaluation.feasibilityKnown = false;
        evaluation.costMs = 0.;
      }
      return evaluation;
    });
    if (!require(guarded.hasPricedSeed && guarded.evaluation.costMs == 10. &&
                     guarded.improvedCoordinates == 0,
                 "unknown-feasibility trial is never an artificial zero-cost "
                 "improvement"))
      return 1;
  }
  {
    ProgramBeamInitialStates seeds;
    seeds.assignments = {{1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    std::vector<std::vector<int64_t>> warmCalls;
    auto warmEvaluator = [&](const std::vector<int64_t> &assignment) {
      warmCalls.push_back(assignment);
      ProgramSearchEvaluation evaluation;
      evaluation.feasible = true;
      evaluation.costMs =
          assignment == std::vector<int64_t>({1, 1, 1}) ? 1. : 5.;
      return evaluation;
    };
    auto warm = searchProgramSpaceBeam(binarySpace(3), warmEvaluator, unlimited,
                                       1, -1, 0, {}, {}, seeds);
    if (!require(warm.hasBest && warm.bestEvaluation.costMs == 1. &&
                     warm.report.getInteger("initial_states_reevaluated") ==
                         1 &&
                     warm.report.getInteger("initial_state_duplicates") == 2 &&
                     std::count(warmCalls.begin(), warmCalls.end(),
                                std::vector<int64_t>({1, 1, 1})) == 1,
                 "warm proposals use fresh callbacks, deduplicate and enter "
                 "initial beam"))
      return 1;
    for (size_t d = 0; d < 3; ++d)
      for (int64_t choice = 0; choice < 2; ++choice)
        if (!require(std::any_of(warmCalls.begin(), warmCalls.end(),
                                 [&](const auto &a) { return a[d] == choice; }),
                     "warm initialization leaves every local choice exposed"))
          return 1;
    auto reversedPrice = [&](const std::vector<int64_t> &assignment) {
      ProgramSearchEvaluation evaluation;
      evaluation.feasible = true;
      evaluation.costMs =
          assignment == std::vector<int64_t>({1, 1, 1}) ? 100. : 2.;
      return evaluation;
    };
    auto repriced = searchProgramSpaceBeam(binarySpace(3), reversedPrice,
                                           unlimited, 1, 0, 0, {}, {}, seeds);
    if (!require(
            repriced.hasBest && repriced.bestEvaluation.costMs == 2. &&
                repriced.bestAssignment == std::vector<int64_t>({0, 0, 0}),
            "historical winner loses when fresh evaluator reverses its price"))
      return 1;
    for (const auto &bad :
         std::vector<std::vector<int64_t>>{{1}, {2, 0, 0}, {-1, 0, 0}}) {
      ProgramBeamInitialStates invalid;
      invalid.assignments.push_back(bad);
      int invalidCalls = 0;
      auto rejectedSeed = searchProgramSpaceBeam(
          binarySpace(3),
          [&](const auto &a) {
            ++invalidCalls;
            return reversedPrice(a);
          },
          unlimited, 1, -1, 0, {}, {}, invalid);
      if (!require(rejectedSeed.status ==
                           ConstraintSearchStatus::InvalidInput &&
                       invalidCalls == 0,
                   "invalid seed is rejected before any baseline/evaluator "
                   "callback"))
        return 1;
    }
    auto constrained = binarySpace(2);
    ProgramSearchConstraint onlyBaseline;
    onlyBaseline.dimensions = {0, 1};
    onlyBaseline.allowedTuples = {{0, 0}};
    constrained.constraints.push_back(onlyBaseline);
    ProgramBeamInitialStates incompatible;
    incompatible.assignments = {{1, 1}};
    int incompatibleCalls = 0;
    auto incompatibleResult = searchProgramSpaceBeam(
        constrained,
        [&](const auto &a) {
          ++incompatibleCalls;
          return reversedPrice(a);
        },
        unlimited, 1, -1, 0, {}, {}, incompatible);
    if (!require(
            incompatibleResult.status == ConstraintSearchStatus::InvalidInput &&
                incompatibleCalls == 0,
            "compatibility-invalid warm state is rejected before callbacks"))
      return 1;
    std::vector<std::vector<int64_t>> oldOrder, emptyOrder;
    auto ordered = [&](auto &order) {
      return [&order](const std::vector<int64_t> &assignment) {
        order.push_back(assignment);
        ProgramSearchEvaluation evaluation;
        evaluation.feasible = true;
        evaluation.costMs = 1.;
        return evaluation;
      };
    };
    auto oldResult = searchProgramSpaceBeam(binarySpace(2), ordered(oldOrder),
                                            unlimited, 2, -1);
    auto emptyResult = searchProgramSpaceBeam(
        binarySpace(2), ordered(emptyOrder), unlimited, 2, -1, 0, {}, {}, {});
    if (!require(oldOrder == emptyOrder &&
                     oldResult.bestAssignment == emptyResult.bestAssignment &&
                     oldResult.status == emptyResult.status,
                 "default empty warm proposals preserve old evaluation order "
                 "and result"))
      return 1;
  }
  {
    std::vector<ProgramSearchSpace> malformed;
    auto invalid = binarySpace(2);
    invalid.dimensions[0].id.clear();
    malformed.push_back(invalid);
    invalid = binarySpace(2);
    invalid.dimensions[1].id = invalid.dimensions[0].id;
    malformed.push_back(invalid);
    invalid = binarySpace(2);
    invalid.dimensions[0].choices[0].id.clear();
    malformed.push_back(invalid);
    invalid = binarySpace(2);
    invalid.dimensions[0].choices[1].id = invalid.dimensions[0].choices[0].id;
    malformed.push_back(invalid);
    invalid = binarySpace(2);
    ProgramSearchConstraint duplicateScope;
    duplicateScope.dimensions = {0, 0};
    duplicateScope.allowedTuples = {{0, 0}};
    invalid.constraints.push_back(duplicateScope);
    malformed.push_back(invalid);
    for (const auto &domain : malformed) {
      int callbacks = 0;
      auto callback = [&](const auto &) {
        ++callbacks;
        ProgramSearchEvaluation evaluation;
        evaluation.feasible = true;
        evaluation.costMs = 1.;
        return evaluation;
      };
      auto coordinate = initializeProgramCoordinates(domain, callback);
      auto beam = searchProgramSpaceBeam(domain, callback, unlimited, 1, -1);
      if (!require(!coordinate.valid && !coordinate.attempted &&
                       !coordinate.completed &&
                       beam.status == ConstraintSearchStatus::InvalidInput &&
                       callbacks == 0,
                   "both public entries reject malformed domain "
                   "identities/scopes before callbacks"))
        return 1;
    }
    auto tinyWall = unlimited;
    tinyWall.wallSeconds = std::numeric_limits<double>::min();
    int deadlineCallbacks = 0;
    auto interrupted = searchProgramSpaceBeam(
        binarySpace(2),
        [&](const auto &) {
          ++deadlineCallbacks;
          return ProgramSearchEvaluation{};
        },
        tinyWall, 1, -1);
    if (!require(interrupted.status == ConstraintSearchStatus::Unknown &&
                     deadlineCallbacks == 0 &&
                     interrupted.report.getBoolean("budget_exhausted") == true,
                 "positive exhausted validation wall budget is Unknown before "
                 "callbacks"))
      return 1;
    auto unknown =
        initializeProgramCoordinates(binarySpace(1), [](const auto &) {
          ProgramSearchEvaluation evaluation;
          evaluation.feasible = true;
          return evaluation;
        });
    if (!require(unknown.valid && unknown.attempted && !unknown.completed &&
                     !unknown.hasPricedSeed && unknown.evaluations == 1 &&
                     !unknown.startCostMs && !unknown.evaluation.costMs,
                 "unknown start reports attempted but incomplete "
                 "initialization without zero cost"))
      return 1;
    auto quota = unlimited;
    quota.maxVariables = 1;
    int quotaCallbacks = 0;
    auto limited = searchProgramSpaceBeam(
        binarySpace(2),
        [&](const auto &) {
          ++quotaCallbacks;
          return ProgramSearchEvaluation{};
        },
        quota, 1, -1);
    if (!require(
            limited.status == ConstraintSearchStatus::ModelLimit &&
                quotaCallbacks == 0,
            "shared structural validation retains beam model quota rejection"))
      return 1;
  }
  unlimited.wallSeconds = std::numeric_limits<double>::quiet_NaN();
  auto invalidUnlimited =
      searchProgramSpaceBeam(binarySpace(1), evaluator, unlimited, 1, -1);
  if (!require(invalidUnlimited.status == ConstraintSearchStatus::InvalidInput,
               "unlimited sentinel does not admit NaN wall budget"))
    return 1;
  llvm::outs() << "{\"status\":\"passed\",\"checks\":" << checks << "}\n";
  return 0;
}
