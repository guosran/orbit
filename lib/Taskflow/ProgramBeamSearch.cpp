#include "Taskflow/ProgramBeamSearch.h"
#include "Taskflow/Search.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <string>
#include <utility>

namespace orbit::taskflow {
namespace {
struct BeamNode {
  std::vector<int64_t> assignment;
  std::optional<double> cost;
};
std::string nodeKey(const BeamNode &node) {
  std::string key;
  for (int64_t value : node.assignment)
    key += std::to_string(value) + ",";
  return key.empty() ? "empty" : key;
}
// Configuration diversity means different supports relative to the baseline,
// independent of device/operator names. Choices in one dimension share a class.
std::string nodeConfiguration(const BeamNode &node) {
  std::string key;
  for (size_t i = 0; i < node.assignment.size(); ++i)
    if (node.assignment[i] != 0)
      key += std::to_string(i) + ",";
  return key.empty() ? "baseline" : key;
}
llvm::json::Array assignmentJSON(const std::vector<int64_t> &assignment) {
  llvm::json::Array result;
  for (auto value : assignment)
    result.push_back(value);
  return result;
}
} // namespace

ProgramSearchResult
searchProgramSpaceBeam(const ProgramSearchSpace &space,
                       const ProgramSearchEvaluator &evaluate,
                       const ProgramSearchOptions &options, int64_t beamWidth,
                       int64_t maxRounds, int64_t diversitySlots) {
  using Clock = std::chrono::steady_clock;
  namespace j = llvm::json;
  const auto start = Clock::now();
  auto milliseconds = [](Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
  };
  auto expired = [&]() {
    return milliseconds(start) >= options.wallSeconds * 1000.;
  };
  ProgramSearchResult result;
  j::Array trace, history, dimensions;
  int64_t calls = 0, feasible = 0, unknownFeasibility = 0, unknownCosts = 0;
  int64_t duplicates = 0, rejected = 0, rounds = 0, tupleCells = 0;
  double callbackMs = 0, callbackOverrunMs = 0;
  auto finish = [&](ConstraintSearchStatus status, const std::string &reason) {
    result.status = status;
    auto &r = result.report;
    r["algorithm"] = "orbit-public-beam";
    r["implementation"] =
        "public-domain adapter using existing selectSearchBeam";
    r["legacy_default_search"] = false;
    r["complete_plan_prepricing"] = false;
    r["objective"] = "full-checked-evaluator-cost-ms";
    r["status"] = constraintSearchStatusName(status);
    r["reason"] = reason;
    r["has_best"] = result.hasBest;
    r["evaluations"] = calls;
    r["feasible_evaluations"] = feasible;
    r["unknown_feasibility_evaluations"] = unknownFeasibility;
    r["unknown_cost_evaluations"] = unknownCosts;
    r["all_feasibility_known"] = unknownFeasibility == 0;
    r["all_feasible_costs_known"] = unknownCosts == 0;
    r["finite_domain_optimality_proved"] = false;
    r["finite_domain_exhausted"] = false;
    r["neighborhood"] = "legal single-dimension changes; compound choices "
                        "permitted; no implicit compatibility repair";
    r["diversity_configuration"] = "support of nonbaseline dimensions";
    r["cache_hits"] = duplicates;
    r["cache_misses"] = calls;
    r["cache_policy"] =
        "global assignment deduplication; no complete-plan price catalogue";
    r["compatibility_rejections"] = rejected;
    r["beam_width"] = beamWidth;
    r["max_rounds"] = maxRounds;
    r["rounds"] = rounds;
    r["diversity_slots"] = diversitySlots;
    r["max_evaluations"] = options.maxEvaluations;
    r["callback_ms"] = callbackMs;
    r["solve_ms"] = 0.;
    r["model_build_ms"] = 0.;
    r["randomized"] = false;
    r["evaluation_workers"] = 1;
    r["callback_preemptible"] = false;
    r["callback_deadline_overrun_ms"] = callbackOverrunMs;
    r["compatibility_tuple_cells"] = tupleCells;
    r["dimensions"] = std::move(dimensions);
    r["evaluation_trace"] = std::move(trace);
    r["beam_selection_history"] = std::move(history);
    if (result.hasBest) {
      r["best_assignment"] = assignmentJSON(result.bestAssignment);
      r["best_cost_ms"] = result.bestEvaluation.costMs
                              ? j::Value(*result.bestEvaluation.costMs)
                              : j::Value(nullptr);
    }
    const double duration = milliseconds(start);
    r["total_ms"] = duration;
    r["wall_seconds"] = std::isfinite(options.wallSeconds)
                            ? j::Value(options.wallSeconds)
                            : j::Value(nullptr);
    r["budget_exhausted"] = std::isfinite(options.wallSeconds) && expired();
    r["returned_after_deadline"] = std::isfinite(options.wallSeconds) &&
                                   duration > options.wallSeconds * 1000.;
    r["deadline_overrun_ms"] =
        std::isfinite(options.wallSeconds)
            ? std::max(0., duration - options.wallSeconds * 1000.)
            : 0.;
    return std::move(result);
  };
  if (!evaluate || !std::isfinite(options.wallSeconds) ||
      options.wallSeconds <= 0 || options.maxEvaluations <= 0 ||
      options.maxVariables <= 0 || options.maxConstraints <= 0 ||
      options.maxTupleCells <= 0 || beamWidth <= 0 || options.workers <= 0 ||
      options.workers > 16 || options.seed < 0 || maxRounds < 0 ||
      diversitySlots < 0 || diversitySlots > beamWidth)
    return finish(ConstraintSearchStatus::InvalidInput,
                  "Invalid beam options or evaluator");
  if (space.dimensions.size() > static_cast<size_t>(options.maxVariables) ||
      space.constraints.size() > static_cast<size_t>(options.maxConstraints))
    return finish(ConstraintSearchStatus::ModelLimit,
                  "Input domain exceeds configured limits");
  std::set<std::string> ids;
  int64_t choices = 0;
  for (const auto &d : space.dimensions) {
    if (d.id.empty() || !ids.insert(d.id).second || d.choices.empty())
      return finish(ConstraintSearchStatus::InvalidInput,
                    "Empty choice domain or duplicate dimension ID");
    if (d.choices.size() > static_cast<size_t>(options.maxVariables - choices))
      return finish(ConstraintSearchStatus::ModelLimit,
                    "Choice domain exceeds configured limit");
    choices += static_cast<int64_t>(d.choices.size());
    std::set<std::string> choiceIds;
    for (const auto &c : d.choices)
      if (c.id.empty() || !choiceIds.insert(c.id).second)
        return finish(ConstraintSearchStatus::InvalidInput,
                      "Invalid choice IDs");
    dimensions.push_back(j::Object{{"id", d.id},
                                   {"scope", d.scope},
                                   {"status", d.status},
                                   {"reason", d.reason},
                                   {"choices", int64_t(d.choices.size())}});
    if (expired())
      return finish(ConstraintSearchStatus::Unknown,
                    "Deadline during domain validation");
  }
  for (const auto &c : space.constraints) {
    if (c.dimensions.empty())
      return finish(ConstraintSearchStatus::InvalidInput,
                    "Empty compatibility scope");
    std::set<size_t> scope;
    for (size_t d : c.dimensions)
      if (d >= space.dimensions.size() || !scope.insert(d).second)
        return finish(ConstraintSearchStatus::InvalidInput,
                      "Invalid compatibility scope");
    if (c.dimensions.size() > static_cast<size_t>(options.maxTupleCells) ||
        c.allowedTuples.size() >
            static_cast<size_t>((options.maxTupleCells - tupleCells) /
                                static_cast<int64_t>(c.dimensions.size())))
      return finish(ConstraintSearchStatus::ModelLimit,
                    "Compatibility tuple payload limit");
    tupleCells += static_cast<int64_t>(c.dimensions.size()) *
                  static_cast<int64_t>(c.allowedTuples.size());
    for (const auto &tuple : c.allowedTuples) {
      if (tuple.size() != c.dimensions.size())
        return finish(ConstraintSearchStatus::InvalidInput,
                      "Compatibility tuple arity mismatch");
      for (size_t i = 0; i < tuple.size(); ++i)
        if (tuple[i] < 0 ||
            static_cast<size_t>(tuple[i]) >=
                space.dimensions[c.dimensions[i]].choices.size())
          return finish(ConstraintSearchStatus::InvalidInput,
                        "Compatibility value outside domain");
      if (expired())
        return finish(ConstraintSearchStatus::Unknown,
                      "Deadline during table validation");
    }
  }
  auto compatible = [&](const std::vector<int64_t> &assignment) {
    for (const auto &c : space.constraints) {
      bool admits = false;
      for (const auto &tuple : c.allowedTuples) {
        bool matches = true;
        for (size_t i = 0; i < tuple.size(); ++i)
          matches &= assignment[c.dimensions[i]] == tuple[i];
        if (matches) {
          admits = true;
          break;
        }
      }
      if (!admits)
        return false;
    }
    return true;
  };
  std::set<std::vector<int64_t>> visited;
  auto check = [&](const std::vector<int64_t> &assignment) {
    visited.insert(assignment);
    const auto begin = Clock::now();
    auto evaluation = evaluate(assignment);
    const double duration = milliseconds(begin);
    callbackMs += duration;
    ++calls;
    callbackOverrunMs = std::max(
        callbackOverrunMs,
        std::max(0., milliseconds(start) - options.wallSeconds * 1000.));
    if (!evaluation.feasibilityKnown)
      ++unknownFeasibility;
    if (evaluation.costMs &&
        (!std::isfinite(*evaluation.costMs) || *evaluation.costMs < 0)) {
      evaluation.costMs.reset();
      evaluation.message += "; invalid cost treated as unknown";
    }
    if (!evaluation.feasible)
      evaluation.costMs.reset();
    if (evaluation.feasible && evaluation.feasibilityKnown) {
      ++feasible;
      if (!evaluation.costMs)
        ++unknownCosts;
    }
    trace.push_back(
        j::Object{{"assignment", assignmentJSON(assignment)},
                  {"call", calls},
                  {"feasible", evaluation.feasible},
                  {"feasibility_known", evaluation.feasibilityKnown},
                  {"cost_ms", evaluation.costMs ? j::Value(*evaluation.costMs)
                                                : j::Value(nullptr)},
                  {"callback_ms", duration},
                  {"elapsed_ms", milliseconds(start)},
                  {"message", evaluation.message}});
    BeamNode node{assignment, std::nullopt};
    if (evaluation.feasible && evaluation.feasibilityKnown) {
      node.cost = evaluation.costMs;
      if (!result.hasBest ||
          (evaluation.costMs &&
           (!result.bestEvaluation.costMs ||
            *evaluation.costMs < *result.bestEvaluation.costMs))) {
        result.hasBest = true;
        result.bestAssignment = assignment;
        result.bestEvaluation = std::move(evaluation);
      }
    }
    return node;
  };
  auto outcome = [&]() {
    return result.hasBest ? ConstraintSearchStatus::Feasible
                          : ConstraintSearchStatus::Unknown;
  };
  auto recordBeam = [&](int64_t round, const std::vector<BeamNode> &beam) {
    j::Array nodes;
    for (const auto &n : beam)
      nodes.push_back(j::Object{
          {"assignment", assignmentJSON(n.assignment)},
          {"cost_ms", n.cost ? j::Value(*n.cost) : j::Value(nullptr)}});
    history.push_back(j::Object{{"round", round},
                                {"evaluations", calls},
                                {"selected", std::move(nodes)}});
  };
  std::vector<int64_t> baseline(space.dimensions.size(), 0);
  if (!compatible(baseline))
    return finish(ConstraintSearchStatus::InvalidInput,
                  "All-zero baseline violates compatibility tables");
  if (expired())
    return finish(ConstraintSearchStatus::Unknown,
                  "Deadline before baseline callback");
  std::vector<BeamNode> beam{check(baseline)};
  recordBeam(0, beam);
  for (int64_t round = 1; round <= maxRounds; ++round) {
    if (expired())
      return finish(ConstraintSearchStatus::Unknown,
                    "Shared wall quota exhausted");
    if (calls >= options.maxEvaluations)
      return finish(outcome(), "Evaluation quota reached");
    std::vector<BeamNode> pool = beam;
    size_t generated = 0;
    for (const auto &parent : beam)
      for (size_t d = 0; d < space.dimensions.size(); ++d)
        for (size_t value = 0; value < space.dimensions[d].choices.size();
             ++value) {
          if (static_cast<int64_t>(value) == parent.assignment[d])
            continue;
          if (expired())
            return finish(ConstraintSearchStatus::Unknown,
                          "Deadline during neighborhood expansion");
          auto assignment = parent.assignment;
          assignment[d] = static_cast<int64_t>(value);
          if (visited.count(assignment)) {
            ++duplicates;
            continue;
          }
          if (!compatible(assignment)) {
            visited.insert(std::move(assignment));
            ++rejected;
            continue;
          }
          if (calls >= options.maxEvaluations)
            return finish(
                outcome(),
                "Evaluation quota reached during neighborhood expansion");
          pool.push_back(check(assignment));
          ++generated;
        }
    rounds = round;
    auto next = selectSearchBeam(
        std::move(pool), beamWidth, diversitySlots,
        [](const BeamNode &a, const BeamNode &b) {
          if (a.cost && b.cost && *a.cost != *b.cost)
            return *a.cost < *b.cost;
          if (bool(a.cost) != bool(b.cost))
            return bool(a.cost);
          return a.assignment < b.assignment;
        },
        [](const BeamNode &n) { return n.cost.has_value(); }, nodeKey,
        nodeConfiguration);
    recordBeam(round, next);
    if (generated == 0)
      return finish(outcome(), "No unchecked legal single-dimension neighbors; "
                               "no global optimality claim");
    if (next.empty())
      return finish(outcome(),
                    "No scored beam states; unknown costs are not zero");
    bool unchanged = next.size() == beam.size();
    if (unchanged)
      for (size_t i = 0; i < next.size(); ++i)
        unchanged &= next[i].assignment == beam[i].assignment;
    beam = std::move(next);
    if (unchanged)
      return finish(outcome(), "Beam unchanged after legal neighborhood "
                               "expansion; no global optimality claim");
  }
  return finish(outcome(), "Round quota reached; no global optimality claim");
}
} // namespace orbit::taskflow
