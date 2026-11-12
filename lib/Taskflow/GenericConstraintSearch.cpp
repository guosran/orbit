#include "Taskflow/GenericConstraintSearch.h"
#include "Taskflow/ConstraintSearch.h"
#include "Taskflow/JSONUtils.h"
#include "Taskflow/NativeTransforms.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
using namespace orbit::taskflow;
namespace j = llvm::json;
GenericConstraintSearchResult orbit::taskflow::searchGenericConstraints(
    const GenericAnalysis &a, const j::Value &config,
    GenericReuseOptions domain, int64_t wallMs, uint64_t callBudget) {
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  GenericConstraintSearchResult out;
  auto before = genericMappingStatistics();
  double generationMs = 0, pricingMs = 0, recheckMs = 0,
         localPricingDeadline = wallMs;
  bool finalChecking = false, catalogueComplete = false;
  double maxPricingCallMs = 0;
  int64_t offeredLocalPlans = 0, localPrices = 0, priced = 0, rejected = 0;
  std::string error;
  auto ms = [&](auto t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
  };
  auto remaining = [&]() { return std::max(0., double(wallMs) - ms(start)); };
  auto calls = [&]() {
    return genericMappingStatistics().costProviderCalls -
           before.costProviderCalls;
  };
  auto canPrice = [&](uint64_t reserve = 2) {
    return remaining() > 0 &&
           (finalChecking || ms(start) < localPricingDeadline) &&
           calls() <= callBudget && callBudget - calls() >= reserve;
  };
  auto finish = [&](const char *status, const std::string &message) {
    auto after = genericMappingStatistics();
    out.report["status"] = status;
    out.report["message"] = message;
    out.report["generation_ms"] = generationMs;
    out.report["phase_budget_policy"] =
        "discovery <=25% soft; local pricing <=55% of remaining; split "
        "remaining between solve and final check";
    out.report["pricing_ms"] = pricingMs;
    out.report["max_complete_pricing_call_ms"] = maxPricingCallMs;
    out.report["offered_local_plans"] = offeredLocalPlans;
    out.report["fully_priced_local_plans"] = localPrices;
    out.report["pricing_complete"] = localPrices == offeredLocalPlans;
    out.report["domain_truncated"] =
        !catalogueComplete || localPrices != offeredLocalPlans;
    out.report["optimality_scope"] =
        "finite fully priced local modes and additive surrogate only";
    out.report["recheck_ms"] = recheckMs;
    out.report["total_ms"] = ms(start);
    out.report["wall_budget_ms"] = wallMs;
    out.report["budget_exhausted"] = remaining() <= 0;
    out.report["cost_provider_calls"] = int64_t(calls());
    out.report["cost_provider_call_budget"] = int64_t(callBudget);
    out.report["planner_calls"] =
        int64_t(after.plannerCalls - before.plannerCalls);
    out.report["checker_calls"] =
        int64_t(after.checkerCalls - before.checkerCalls);
    out.report["priced_local_plans"] = priced;
    out.report["rejected_prices"] = rejected;
    out.report["objective_exact_for_backend"] = false;
    out.report["last_error"] = error;
    return std::move(out);
  };
  if (wallMs <= 0 || a.tasks.empty() || callBudget < 4)
    return finish(
        "INVALID_INPUT",
        "Positive total time and at least four raw cost calls required");
  if (!constraintSearchAvailable())
    return finish("UNAVAILABLE", "Build with ORBIT_ENABLE_ORTOOLS");
  auto pricingConfig =
      copyJSONWithoutFields(config, {"native_auto_candidates"});
  if (!pricingConfig.getAsObject())
    return finish("INVALID_INPUT", "Configuration object required");
  auto source = a.function->getParentOfType<mlir::ModuleOp>();
  auto price = [&](const std::vector<int64_t> &tile,
                   const ProgramMappingChoice &mapping, int64_t &cost) {
    if (tile.size() != 3 || !canPrice())
      return false;
    auto t = Clock::now();
    mlir::OwningOpRef<mlir::ModuleOp> clone(
        mlir::cast<mlir::ModuleOp>(source->clone()));
    GenericAnalysis facts;
    GenericMappingPlan plan;
    std::string reason;
    bool ok =
        setNativeAIRTileSizes(*clone, tile[0], tile[1], tile[2], reason) &&
        setNativeMappingChoice(*clone, mapping, reason) &&
        analyzeGenericProgram(*clone, facts, reason) &&
        planGenericMapping(facts, pricingConfig, plan, reason) &&
        plan.hasCost && std::isfinite(plan.estimatedMs) &&
        plan.estimatedMs >= 0 &&
        plan.estimatedMs < double(INT64_MAX / 32) / 1e6;
    auto callMs = ms(t);
    pricingMs += callMs;
    maxPricingCallMs = std::max(maxPricingCallMs, callMs);
    if (ok) {
      cost = std::llround(plan.estimatedMs * 1e6);
      ++priced;
    } else {
      ++rejected;
      error = reason.empty() ? "Complete cost remains unknown" : reason;
    }
    return ok;
  };
  domain.catalogue = true;
  domain.includeBaseline = true;
  domain.maxGroups = 1;
  domain.maxWallTimeMs =
      std::min(domain.maxWallTimeMs > 0 ? domain.maxWallTimeMs : remaining(),
               remaining() * .25);
  auto t = Clock::now();
  auto discovered =
      discoverGenericReuseConfigurations(a, pricingConfig, domain);
  generationMs = ms(t);
  catalogueComplete = discovered.catalogueComplete;
  offeredLocalPlans = discovered.proposals.size();
  out.report["local_discovery"] = serializeGenericReuseStats(discovered.stats);
  out.report["local_catalogue_complete"] = discovered.catalogueComplete;
  out.report["local_stop_reason"] = discovered.stopReason;
  if (remaining() <= 0)
    return finish("UNKNOWN", "Local discovery exhausted total budget");
  std::optional<GenericReuseProposal> bestKnown;
  int64_t bestKnownCost = INT64_MAX;
  auto remember = [&](const std::vector<int64_t> &sizes,
                      const ProgramMappingChoice &mapping, int64_t cost) {
    if (cost >= bestKnownCost)
      return;
    bestKnownCost = cost;
    GenericReuseProposal p;
    p.tileSizes = sizes;
    p.mapping = mapping;
    p.diagnostics = j::Object{{"algorithm", "cp-sat"},
                              {"rechecked_ns", cost},
                              {"selection", "best-fully-priced-local-plan"},
                              {"objective_exact_for_backend", false}};
    bestKnown = std::move(p);
  };
  auto fallback = [&](const std::string &reason) {
    out.report["selection"] = "best-fully-priced-local-plan";
    out.report["rechecked_ns"] = bestKnownCost;
    out.proposal = std::move(bestKnown);
    return finish("CANDIDATE", reason);
  };
  localPricingDeadline = ms(start) + remaining() * .55;
  ConstraintSearchProblem problem;
  problem.taskCount = a.tasks.size();
  for (size_t child = 0; child < a.tasks.size(); ++child)
    for (auto parent : a.tasks[child].dependencies)
      problem.edges.push_back({int64_t(parent), int64_t(child)});
  struct Tile {
    std::vector<int64_t> sizes;
    ProgramMappingChoice baseline;
    int64_t cost;
  };
  std::vector<Tile> tiles;
  std::map<std::vector<int64_t>, int64_t> tileIds;
  std::vector<std::optional<ProgramResidencyGroup>> groups;
  for (auto &p : discovered.proposals) {
    if (!genericResidencyGroups(p.mapping).empty() ||
        tileIds.count(p.tileSizes) || !canPrice(4))
      continue;
    int64_t cost;
    if (!price(p.tileSizes, p.mapping, cost))
      continue;
    remember(p.tileSizes, p.mapping, cost);
    tileIds[p.tileSizes] = tiles.size();
    tiles.push_back({p.tileSizes, p.mapping, cost});
    problem.tileBaseCostNs.push_back(cost);
  }
  problem.tileCount = tiles.size();
  if (tiles.empty())
    return finish("UNKNOWN",
                  "No fully priced baseline; strict unknowns are preserved");
  for (size_t ti = 0; ti < tiles.size(); ++ti)
    for (size_t task = 0; task < a.tasks.size(); ++task) {
      ConstraintPattern p;
      p.id = problem.patterns.size();
      p.tasks = {int64_t(task)};
      p.tileIndex = ti;
      problem.patterns.push_back(p);
      groups.push_back(std::nullopt);
    }
  for (auto &p : discovered.proposals) {
    auto gs = genericResidencyGroups(p.mapping);
    auto found = tileIds.find(p.tileSizes);
    if (gs.size() != 1 || found == tileIds.end() || !canPrice(4))
      continue;
    int64_t cost;
    if (!price(p.tileSizes, p.mapping, cost))
      continue;
    remember(p.tileSizes, p.mapping, cost);
    ConstraintPattern x;
    x.id = problem.patterns.size();
    x.tileIndex = found->second;
    x.costNs = cost - tiles[x.tileIndex].cost;
    for (auto task : gs[0].tasks)
      x.tasks.push_back(task);
    for (auto storage : gs[0].retainedStorages)
      x.exclusiveStorageIds.push_back(storage);
    problem.patterns.push_back(x);
    groups.push_back(gs[0]);
  }
  localPrices = priced;
  out.report["best_local_ns"] = bestKnownCost;
  out.report["tiles"] = problem.tileCount;
  out.report["patterns"] = int64_t(problem.patterns.size());
  if (remaining() <= 0 || calls() + 2 > callBudget)
    return fallback("No composition budget remains; returning already checked "
                    "local incumbent");
  ConstraintSearchOptions solverOptions;
  solverOptions.wallSeconds = remaining() / 2000;
  auto solved = solveConstraintSearch(problem, solverOptions);
  auto &s = solved.telemetry;
  out.report["solver"] =
      j::Object{{"status", constraintSearchStatusName(solved.status)},
                {"model_build_ms", s.modelBuildMs},
                {"solve_ms", s.solveMs},
                {"variables", s.variables},
                {"constraints", s.constraints},
                {"branches", s.branches},
                {"conflicts", s.conflicts},
                {"peak_rss_kib", s.peakRssKiB},
                {"surrogate_ns",
                 s.hasObjective ? j::Value(s.objectiveNs) : j::Value(nullptr)},
                {"best_bound_ns",
                 s.hasObjective ? j::Value(s.bestBoundNs) : j::Value(nullptr)},
                {"relative_gap",
                 s.hasObjective ? j::Value(s.relativeGap) : j::Value(nullptr)}};
  if (solved.status != ConstraintSearchStatus::Optimal &&
      solved.status != ConstraintSearchStatus::Feasible)
    return fallback(std::string("Composition returned ") +
                    constraintSearchStatusName(solved.status) +
                    "; returning checked incumbent");
  const auto &tile = tiles.at(solved.selectedTileIndex);
  auto mapping = tile.baseline;
  mapping.taskGroup.clear();
  mapping.retainedStorages.clear();
  mapping.residencyGroups.clear();
  for (auto i : solved.selectedPatternIndices)
    if (groups.at(i))
      mapping.residencyGroups.push_back(*groups.at(i));
  if (!mapping.residencyGroups.empty()) {
    mapping.storage = mapping.reuseLoop = mapping.panelLoop = -1;
    mapping.residency = "tiled-phase-local";
    mapping.grouping = "tile-major";
    mapping.dataOrganization = "retain-across-users";
    std::vector<size_t> order;
    if (!deriveGenericResidencyOrder(a, mapping.residencyGroups, order, error))
      return fallback("Unsupported composition: " + error);
    std::sort(mapping.residencyGroups.begin(), mapping.residencyGroups.end(),
              [&](const auto &x, const auto &y) {
                return std::find(order.begin(), order.end(), x.tasks.front()) <
                       std::find(order.begin(), order.end(), y.tasks.front());
              });
  }
  finalChecking = true;
  int64_t checkedCost = 0;
  t = Clock::now();
  double oldPricing = pricingMs;
  bool checked = price(tile.sizes, mapping, checkedCost);
  recheckMs = ms(t);
  pricingMs = oldPricing;
  if (!checked)
    return fallback("Surrogate combination failed full planner/checker; "
                    "returning checked incumbent");
  if (remaining() <= 0)
    return fallback(
        "Final check exceeded soft deadline; returning checked incumbent");
  out.report["surrogate_ns"] = s.objectiveNs;
  out.report["rechecked_ns"] = checkedCost;
  out.report["surrogate_error_ns"] = double(checkedCost) - s.objectiveNs;
  if (checkedCost > bestKnownCost)
    return fallback(
        "Rechecked composition costs more than a checked local incumbent");
  out.report["selection"] = "rechecked-composition";
  GenericReuseProposal proposal;
  proposal.mapping = std::move(mapping);
  proposal.tileSizes = tile.sizes;
  proposal.diagnostics = j::Object{{"algorithm", "cp-sat"},
                                   {"surrogate_ns", s.objectiveNs},
                                   {"rechecked_ns", checkedCost},
                                   {"objective_exact_for_backend", false}};
  out.proposal = std::move(proposal);
  return finish("CANDIDATE",
                "Composed local modes and independently rechecked full plan");
}
