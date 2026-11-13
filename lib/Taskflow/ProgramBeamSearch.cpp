#include "Taskflow/ProgramBeamSearch.h"
#include "Taskflow/Search.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>

namespace orbit::taskflow {
namespace {
struct BeamNode {
  std::vector<int64_t> assignment;
  std::optional<double> cost;
  int64_t call = 0;
  bool knownFeasible = false;
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
// Validate semantic domain identity and table structure without invoking an
// evaluator. Both public entry points must reject malformed domains before a
// caller's callback can perform observable work. Resource quotas remain beam
// options and are checked separately before this structural walk.
enum class DomainValidation { Valid, Invalid, Interrupted };
DomainValidation
validateDomainStructure(const ProgramSearchSpace &space, std::string &error,
                        const std::function<bool()> &stopRequested = {}) {
  auto invalid = [&](const char *message) {
    error = message;
    return DomainValidation::Invalid;
  };
  auto interrupted = [&](const char *message) {
    if (!stopRequested || !stopRequested())
      return false;
    error = message;
    return true;
  };
  std::set<std::string> dimensionIds;
  for (const auto &dimension : space.dimensions) {
    if (interrupted("Input validation interrupted during dimension validation"))
      return DomainValidation::Interrupted;
    if (dimension.id.empty() || !dimensionIds.insert(dimension.id).second ||
        dimension.choices.empty())
      return invalid("Empty choice domain or duplicate dimension ID");
    std::set<std::string> choiceIds;
    for (const auto &choice : dimension.choices) {
      if (interrupted("Input validation interrupted during choice validation"))
        return DomainValidation::Interrupted;
      if (choice.id.empty() || !choiceIds.insert(choice.id).second)
        return invalid("Invalid choice IDs");
    }
  }
  for (const auto &constraint : space.constraints) {
    if (constraint.dimensions.empty())
      return invalid("Empty compatibility scope");
    std::set<size_t> scope;
    for (size_t d : constraint.dimensions)
      if (d >= space.dimensions.size() || !scope.insert(d).second)
        return invalid("Invalid compatibility scope");
    for (const auto &tuple : constraint.allowedTuples) {
      if (interrupted("Input validation interrupted during compatibility tuple "
                      "validation"))
        return DomainValidation::Interrupted;
      if (tuple.size() != constraint.dimensions.size())
        return invalid("Compatibility tuple arity mismatch");
      for (size_t i = 0; i < tuple.size(); ++i)
        if (tuple[i] < 0 ||
            size_t(tuple[i]) >=
                space.dimensions[constraint.dimensions[i]].choices.size())
          return invalid("Compatibility value outside domain");
      if (interrupted("Input validation interrupted during compatibility tuple "
                      "validation"))
        return DomainValidation::Interrupted;
    }
  }
  return DomainValidation::Valid;
}
} // namespace

ProgramCoordinateInitialization initializeProgramCoordinates(
    const ProgramSearchSpace &space, const ProgramSearchEvaluator &evaluate,
    const std::vector<std::vector<int64_t>> &starts,
    const std::function<void(const std::vector<int64_t> &,
                             const ProgramSearchEvaluation &)> &observed) {
  ProgramCoordinateInitialization result;
  auto invalid = [&](const char *message) {
    result.valid = false;
    result.message = message;
    return result;
  };
  if (!evaluate)
    return invalid("Coordinate initializer requires an evaluator");
  if (validateDomainStructure(space, result.message) !=
      DomainValidation::Valid) {
    result.valid = false;
    return result;
  }
  auto compatible = [&](const std::vector<int64_t> &assignment) {
    if (assignment.size() != space.dimensions.size())
      return false;
    for (size_t d = 0; d < assignment.size(); ++d)
      if (assignment[d] < 0 ||
          size_t(assignment[d]) >= space.dimensions[d].choices.size())
        return false;
    for (const auto &constraint : space.constraints) {
      bool admitted = false;
      for (const auto &tuple : constraint.allowedTuples) {
        bool matches = true;
        for (size_t i = 0; i < tuple.size(); ++i)
          matches &= assignment[constraint.dimensions[i]] == tuple[i];
        admitted |= matches;
      }
      if (!admitted)
        return false;
    }
    return true;
  };
  std::vector<std::vector<int64_t>> proposals{
      std::vector<int64_t>(space.dimensions.size(), 0)};
  proposals.insert(proposals.end(), starts.begin(), starts.end());
  for (const auto &assignment : proposals)
    if (!compatible(assignment))
      return invalid("Coordinate initial proposal violates the current domain");
  auto priced = [](const ProgramSearchEvaluation &evaluation) {
    return evaluation.feasible && evaluation.feasibilityKnown &&
           evaluation.costMs && std::isfinite(*evaluation.costMs) &&
           *evaluation.costMs >= 0;
  };
  auto check = [&](const std::vector<int64_t> &assignment) {
    auto evaluation = evaluate(assignment);
    ++result.evaluations;
    if (observed)
      observed(assignment, evaluation);
    return evaluation;
  };
  result.attempted = true;
  std::set<std::vector<int64_t>> initialSeen;
  for (const auto &assignment : proposals) {
    if (!initialSeen.insert(assignment).second)
      continue;
    auto evaluation = check(assignment);
    if (priced(evaluation) &&
        (!result.hasPricedSeed ||
         *evaluation.costMs < *result.evaluation.costMs)) {
      result.hasPricedSeed = true;
      result.assignment = assignment;
      result.evaluation = std::move(evaluation);
    }
  }
  if (!result.hasPricedSeed) {
    result.message = "No priced feasible start; ordinary beam exploration "
                     "remains responsible";
    return result;
  }
  result.startCostMs = result.evaluation.costMs;
  for (size_t d = 0; d < space.dimensions.size(); ++d) {
    // Every alternative starts from the same accepted joint state for this
    // coordinate. Commit once, after all choices; never accumulate trial edits.
    const auto incumbent = result.assignment;
    auto best = incumbent;
    auto bestEvaluation = result.evaluation;
    for (size_t choice = 0; choice < space.dimensions[d].choices.size();
         ++choice) {
      if (int64_t(choice) == incumbent[d])
        continue; // Already priced at this exact accepted state.
      auto trial = incumbent;
      trial[d] = int64_t(choice);
      if (!compatible(trial))
        continue;
      auto evaluation = check(trial);
      if (priced(evaluation) && *evaluation.costMs < *bestEvaluation.costMs) {
        best = std::move(trial);
        bestEvaluation = std::move(evaluation);
      }
    }
    if (best != incumbent) {
      ++result.improvedCoordinates;
      result.assignment = std::move(best);
      result.evaluation = std::move(bestEvaluation);
    }
  }
  result.completed = true;
  result.message = "One coordinate initialization pass; full uncapped beam and "
                   "neighborhood closure still required";
  return result;
}

ProgramSearchResult searchProgramSpaceBeam(
    const ProgramSearchSpace &space, const ProgramSearchEvaluator &evaluate,
    const ProgramSearchOptions &options, int64_t beamWidth, int64_t maxRounds,
    int64_t diversitySlots, const ProgramBeamExplorationOptions &exploration,
    const ProgramBeamTraceOptions &traceOptions,
    const ProgramBeamInitialStates &initialStates) {
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
  int64_t enqueued = 0, dropped = 0, expandedUnknown = 0, peakPending = 0;
  int64_t pendingLimit = 0;
  int64_t initialStatesEvaluated = 0, initialStateDuplicates = 0;
  std::deque<BeamNode> pending;
  std::set<int64_t> expanded;
  std::vector<unsigned> assignmentBitWidths;
  size_t assignmentKeyBytes = 0;
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
    r["initial_state_proposals"] = int64_t(initialStates.assignments.size());
    r["initial_states_reevaluated"] = initialStatesEvaluated;
    r["initial_state_duplicates"] = initialStateDuplicates;
    r["initial_state_costs_reused"] = false;
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
    r["assignment_key_encoding"] = "lossless fixed-width choice indices";
    r["assignment_key_bytes"] = int64_t(assignmentKeyBytes);
    r["compatibility_rejections"] = rejected;
    r["beam_width"] = beamWidth;
    r["max_rounds"] = maxRounds < 0 ? j::Value(nullptr) : j::Value(maxRounds);
    r["rounds"] = rounds;
    r["diversity_slots"] = diversitySlots;
    r["unknown_exploration_policy"] =
        "bounded FIFO of checked-feasible unpriced states";
    r["unknown_exploration_slots"] = exploration.unknownSlots;
    r["unknown_exploration_enabled"] = exploration.unknownSlots > 0;
    r["unknown_pending_limit"] = pendingLimit;
    r["unknown_pending_peak"] = peakPending;
    r["unknown_pending_remaining"] = int64_t(pending.size());
    r["unknown_enqueued"] = enqueued;
    r["unknown_frontier_dropped"] = dropped;
    r["unknown_expanded"] = expandedUnknown;
    r["unknown_cost_ranking_score"] = nullptr;
    r["frontier_fairness"] =
        "retain priced beam; alternate when exploration would occupy all slots";
    r["max_evaluations"] = options.maxEvaluations == 0
                               ? j::Value(nullptr)
                               : j::Value(options.maxEvaluations);
    r["callback_ms"] = callbackMs;
    r["solve_ms"] = 0.;
    r["model_build_ms"] = 0.;
    r["randomized"] = false;
    r["evaluation_workers"] = 1;
    r["callback_preemptible"] = false;
    r["callback_deadline_overrun_ms"] = callbackOverrunMs;
    r["compatibility_tuple_cells"] = tupleCells;
    r["dimensions"] = std::move(dimensions);
    r["evaluation_trace_retained_in_memory"] = traceOptions.retainInMemory;
    r["evaluation_trace_streamed"] = bool(traceOptions.sink);
    r["evaluation_trace_rows"] = calls;
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
  if (!evaluate || std::isnan(options.wallSeconds) ||
      options.wallSeconds <= 0 || options.maxEvaluations < 0 ||
      options.maxVariables <= 0 || options.maxConstraints <= 0 ||
      options.maxTupleCells <= 0 || beamWidth <= 0 || options.workers <= 0 ||
      options.workers > 16 || options.seed < 0 || maxRounds < -1 ||
      diversitySlots < 0 || diversitySlots > beamWidth ||
      exploration.unknownSlots < 0 || exploration.unknownSlots > beamWidth ||
      exploration.maxPendingStates < 0 ||
      (!traceOptions.retainInMemory && !traceOptions.sink))
    return finish(ConstraintSearchStatus::InvalidInput,
                  "Invalid beam options or evaluator");
  if (space.dimensions.size() > static_cast<size_t>(options.maxVariables) ||
      space.constraints.size() > static_cast<size_t>(options.maxConstraints))
    return finish(ConstraintSearchStatus::ModelLimit,
                  "Input domain exceeds configured limits");
  // Bound both the number of queued states and their assignment payload. This
  // auxiliary frontier shares the existing memory controls, not a hidden
  // budget.
  pendingLimit = std::min(exploration.maxPendingStates,
                          options.maxTupleCells /
                              std::max<int64_t>(1, space.dimensions.size()));
  int64_t choices = 0;
  for (const auto &dimension : space.dimensions) {
    if (dimension.choices.size() > size_t(options.maxVariables - choices))
      return finish(ConstraintSearchStatus::ModelLimit,
                    "Choice domain exceeds configured limit");
    choices += int64_t(dimension.choices.size());
    if (expired())
      return finish(ConstraintSearchStatus::Unknown,
                    "Deadline during domain quota validation");
  }
  for (const auto &constraint : space.constraints) {
    // Empty scopes are rejected by the shared structural checker below. Avoid
    // dividing by zero while checking this table's explicit resource bound.
    if (constraint.dimensions.empty())
      continue;
    if (constraint.dimensions.size() > size_t(options.maxTupleCells) ||
        constraint.allowedTuples.size() >
            size_t((options.maxTupleCells - tupleCells) /
                   int64_t(constraint.dimensions.size())))
      return finish(ConstraintSearchStatus::ModelLimit,
                    "Compatibility tuple payload limit");
    tupleCells += int64_t(constraint.dimensions.size()) *
                  int64_t(constraint.allowedTuples.size());
    if (expired())
      return finish(ConstraintSearchStatus::Unknown,
                    "Deadline during compatibility quota validation");
  }
  std::string domainError;
  auto validation = validateDomainStructure(space, domainError, expired);
  if (validation == DomainValidation::Interrupted)
    return finish(ConstraintSearchStatus::Unknown, domainError);
  if (validation == DomainValidation::Invalid)
    return finish(ConstraintSearchStatus::InvalidInput, domainError);
  for (const auto &dimension : space.dimensions) {
    dimensions.push_back(
        j::Object{{"id", dimension.id},
                  {"scope", dimension.scope},
                  {"status", dimension.status},
                  {"reason", dimension.reason},
                  {"choices", int64_t(dimension.choices.size())}});
    if (expired())
      return finish(ConstraintSearchStatus::Unknown,
                    "Deadline during domain validation");
  }
  if (expired())
    return finish(ConstraintSearchStatus::Unknown,
                  "Deadline during table validation");
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
  // Validate every proposal before any callback. An incompatible warm start
  // is an input error, not permission to repair a backend assignment silently.
  if (initialStates.assignments.size() >
      size_t(options.maxTupleCells /
             std::max<size_t>(1, space.dimensions.size())))
    return finish(
        ConstraintSearchStatus::ModelLimit,
        "Initial assignment payload exceeds configured resource limit");
  for (const auto &assignment : initialStates.assignments) {
    if (assignment.size() != space.dimensions.size())
      return finish(ConstraintSearchStatus::InvalidInput,
                    "Initial assignment arity differs from domain");
    for (size_t d = 0; d < assignment.size(); ++d)
      if (assignment[d] < 0 ||
          size_t(assignment[d]) >= space.dimensions[d].choices.size())
        return finish(ConstraintSearchStatus::InvalidInput,
                      "Initial assignment choice is outside domain");
    if (!compatible(assignment))
      return finish(ConstraintSearchStatus::InvalidInput,
                    "Initial assignment violates compatibility tables");
  }
  size_t totalBits = 0;
  for (const auto &dimension : space.dimensions) {
    size_t maximum = dimension.choices.size() - 1;
    unsigned bits = 0;
    while (maximum) {
      ++bits;
      maximum >>= 1;
    }
    if (totalBits > SIZE_MAX - bits - 7)
      return finish(ConstraintSearchStatus::ModelLimit,
                    "Assignment key size overflow");
    totalBits += bits;
    assignmentBitWidths.push_back(bits);
  }
  assignmentKeyBytes = (totalBits + 7) / 8;
  auto assignmentKey = [&](const std::vector<int64_t> &assignment) {
    // A fixed domain determines every field's width; singleton dimensions
    // require no bits. Keys encode the entire assignment without hashing it
    // down to a fingerprint. unordered_set resolves hash collisions by exact
    // byte equality, so neither deduplication nor search order changes.
    std::string key(assignmentKeyBytes, '\0');
    size_t offset = 0;
    for (size_t i = 0; i < assignment.size(); ++i) {
      uint64_t value = uint64_t(assignment[i]);
      unsigned remaining = assignmentBitWidths[i];
      while (remaining) {
        unsigned shift = offset % 8;
        unsigned take = std::min(remaining, 8 - shift);
        unsigned byte = uint8_t(key[offset / 8]);
        byte |= (value & ((1u << take) - 1)) << shift;
        key[offset / 8] = char(byte);
        value >>= take;
        offset += take;
        remaining -= take;
      }
    }
    return key;
  };
  std::unordered_set<std::string> visited;
  auto check = [&](const std::vector<int64_t> &assignment) {
    visited.insert(assignmentKey(assignment));
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
    j::Value traceRow =
        j::Object{{"assignment", assignmentJSON(assignment)},
                  {"call", calls},
                  {"feasible", evaluation.feasible},
                  {"feasibility_known", evaluation.feasibilityKnown},
                  {"cost_ms", evaluation.costMs ? j::Value(*evaluation.costMs)
                                                : j::Value(nullptr)},
                  {"callback_ms", duration},
                  {"elapsed_ms", milliseconds(start)},
                  {"message", evaluation.message}};
    // Emit before moving the row into the optional in-memory report. Sink
    // failures propagate to the caller; an incomplete trace is never reported
    // as a completed search. The sink must not influence the evaluator.
    if (traceOptions.sink)
      traceOptions.sink(traceRow);
    if (traceOptions.retainInMemory)
      trace.push_back(std::move(traceRow));
    BeamNode node{assignment, std::nullopt, calls,
                  evaluation.feasible && evaluation.feasibilityKnown};
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
    if (node.knownFeasible && !node.cost && exploration.unknownSlots > 0) {
      if (static_cast<int64_t>(pending.size()) < pendingLimit) {
        pending.push_back(node);
        ++enqueued;
        peakPending = std::max(peakPending, int64_t(pending.size()));
      } else {
        ++dropped;
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
          {"cost_ms", n.cost ? j::Value(*n.cost) : j::Value(nullptr)},
          {"role", n.cost ? "priced"
                          : (n.knownFeasible ? "unpriced-exploration"
                                             : "unranked-baseline")}});
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
  std::vector<BeamNode> pricedBeam;
  if (beam.front().cost)
    pricedBeam = beam;
  if (!initialStates.assignments.empty()) {
    std::vector<BeamNode> pool = beam;
    for (const auto &assignment : initialStates.assignments) {
      if (visited.count(assignmentKey(assignment))) {
        ++initialStateDuplicates;
        continue;
      }
      if (expired())
        return finish(ConstraintSearchStatus::Unknown,
                      "Deadline during initial assignment reevaluation");
      if (options.maxEvaluations > 0 && calls >= options.maxEvaluations)
        return finish(outcome(),
                      "Evaluation quota during initial reevaluation");
      pool.push_back(check(assignment));
      ++initialStatesEvaluated;
    }
    // Use precisely the usual pricing/diversity order. Unknown-cost proposals
    // retain the existing FIFO semantics; no historical score is imported.
    pricedBeam = selectSearchBeam(
        pool, beamWidth, diversitySlots,
        [](const BeamNode &a, const BeamNode &b) {
          if (a.cost && b.cost && *a.cost != *b.cost)
            return *a.cost < *b.cost;
          if (bool(a.cost) != bool(b.cost))
            return bool(a.cost);
          return a.assignment < b.assignment;
        },
        [](const BeamNode &n) { return n.cost.has_value(); }, nodeKey,
        nodeConfiguration);
    std::vector<BeamNode> unpriced;
    while (!pending.empty() &&
           int64_t(unpriced.size()) < exploration.unknownSlots) {
      unpriced.push_back(std::move(pending.front()));
      pending.pop_front();
    }
    int64_t pricedWidth = beamWidth - int64_t(unpriced.size());
    beam = selectSearchBeam(
        pricedBeam, pricedWidth, std::min(diversitySlots, pricedWidth),
        [](const BeamNode &a, const BeamNode &b) {
          if (*a.cost != *b.cost)
            return *a.cost < *b.cost;
          return a.assignment < b.assignment;
        },
        [](const BeamNode &n) { return n.cost.has_value(); }, nodeKey,
        nodeConfiguration);
    beam.insert(beam.end(), std::make_move_iterator(unpriced.begin()),
                std::make_move_iterator(unpriced.end()));
    if (beam.empty())
      beam.push_back(std::move(pool.front()));
  }
  recordBeam(0, beam);
  for (int64_t round = 1; maxRounds < 0 || round <= maxRounds; ++round) {
    if (expired())
      return finish(ConstraintSearchStatus::Unknown,
                    "Shared wall quota exhausted");
    if (options.maxEvaluations > 0 && calls >= options.maxEvaluations)
      return finish(outcome(), "Evaluation quota reached");
    std::vector<BeamNode> pool = pricedBeam;
    size_t generated = 0;
    for (const auto &parent : beam) {
      if (!expanded.insert(parent.call).second)
        continue;
      if (parent.knownFeasible && !parent.cost)
        ++expandedUnknown;
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
          auto key = assignmentKey(assignment);
          if (visited.count(key)) {
            ++duplicates;
            continue;
          }
          if (!compatible(assignment)) {
            visited.insert(std::move(key));
            ++rejected;
            continue;
          }
          if (options.maxEvaluations > 0 && calls >= options.maxEvaluations)
            return finish(
                outcome(),
                "Evaluation quota reached during neighborhood expansion");
          pool.push_back(check(assignment));
          ++generated;
        }
    }
    rounds = round;
    // FIFO exploration occupies slots within the advertised total beam width.
    // Keep pricing selection unchanged and never invent a surrogate unknown
    // cost.
    pricedBeam = selectSearchBeam(
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
    // Keep the ranked frontier even when an exploration slot temporarily
    // displaces it. If exploration consumes all slots, alternate with pending
    // priced work so width-one searches cannot starve either frontier.
    const bool pendingPriced = std::any_of(
        pricedBeam.begin(), pricedBeam.end(),
        [&](const BeamNode &node) { return !expanded.count(node.call); });
    int64_t unknownSlots = exploration.unknownSlots;
    if (pendingPriced && unknownSlots == beamWidth && round % 2 == 0)
      --unknownSlots;
    std::vector<BeamNode> unpriced;
    while (!pending.empty() && int64_t(unpriced.size()) < unknownSlots) {
      auto node = std::move(pending.front());
      pending.pop_front();
      if (!expanded.count(node.call))
        unpriced.push_back(std::move(node));
    }
    const int64_t pricedWidth = beamWidth - int64_t(unpriced.size());
    auto next = selectSearchBeam(
        pricedBeam, pricedWidth, std::min(diversitySlots, pricedWidth),
        [](const BeamNode &a, const BeamNode &b) {
          if (*a.cost != *b.cost)
            return *a.cost < *b.cost;
          return a.assignment < b.assignment;
        },
        [](const BeamNode &n) { return n.cost.has_value(); }, nodeKey,
        nodeConfiguration);
    next.insert(next.end(), std::make_move_iterator(unpriced.begin()),
                std::make_move_iterator(unpriced.end()));
    recordBeam(round, next);
    const bool hasUnexpanded =
        std::any_of(next.begin(), next.end(), [&](const BeamNode &node) {
          return !expanded.count(node.call);
        });
    if (generated == 0 && !hasUnexpanded)
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
    if (unchanged && !hasUnexpanded)
      return finish(outcome(), "Beam unchanged after legal neighborhood "
                               "expansion; no global optimality claim");
  }
  return finish(outcome(), "Round quota reached; no global optimality claim");
}
} // namespace orbit::taskflow
