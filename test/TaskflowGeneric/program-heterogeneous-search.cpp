#include "Taskflow/ProgramHeterogeneousSearch.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <set>
using namespace orbit::taskflow;
namespace j = llvm::json;
int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "expected actual heterogeneous.mlir fixture\n";
    return 2;
  }
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.disableMultithreading();
  auto input = llvm::MemoryBuffer::getFile(argv[1]);
  if (!input)
    return 2;
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>((*input)->getBuffer(), &context);
  GenericAnalysis source;
  std::string error;
  if (!module || !analyzeGenericProgram(*module, source, error)) {
    llvm::errs() << error;
    return 2;
  }
  bool ok = true;
  int checks = 0;
  auto require = [&](bool condition, const char *label) {
    ++checks;
    if (!condition) {
      ok = false;
      llvm::errs() << label << ": " << error << "\n";
    }
  };
  ProgramHeterogeneousSearchOptions options;
  options.search.wallSeconds = 10;
  options.search.maxEvaluations = 64;
  ProgramSearchSpace space;
  require(buildProgramHeterogeneousSearchSpace(source, options, space, error),
          "typed domain builds");
  require(space.dimensions.size() == 5,
          "three task placements and two source boundaries");
  auto materialize = [&](std::vector<int64_t> assignment,
                         HeterogeneousMappingPlan &p) {
    error.clear();
    return materializeProgramHeterogeneousAssignment(source, space, assignment,
                                                     p, error);
  };
  HeterogeneousMappingPlan mixed, npu, split;
  require(materialize({0, 1, 0, 0, 0}, mixed) && mixed.slices.size() == 3,
          "CPU/NPU/CPU choices form three actual regions");
  require(mixed.graph.regionEvents.size() == 8,
          "CPU/NPU/CPU lifecycle is 2+4+2 atoms");
  require(materialize({1, 1, 1, 0, 0}, npu) && npu.slices.size() == 1,
          "joining adjacent NPU tasks changes the actual region graph");
  require(npu.graph.executionRegions[0].inputCopies.size() == 2 &&
              npu.graph.executionRegions[0].submissionCount == 1 &&
              npu.graph.regionEvents.size() == 4,
          "multiple input data edges share one launch");
  size_t transfers = 0;
  for (auto &e : npu.graph.events)
    transfers += e.kind == "transfer";
  require(transfers == 3, "two imports plus one export, no internal-region "
                          "host transfer or launch");
  require(materialize({1, 1, 1, 1, 1}, split) && split.slices.size() == 3 &&
              split.graph.transferBytes > npu.graph.transferBytes,
          "same-device cuts change calls and actual transfer copies");
  auto bad = space;
  bad.dimensions[1].id = bad.dimensions[0].id;
  HeterogeneousMappingPlan p;
  require(!materializeProgramHeterogeneousAssignment(source, bad,
                                                     {0, 0, 0, 0, 0}, p, error),
          "aliased dimensions rejected");
  bad = space;
  (*bad.dimensions[1].choices[0].binding.getAsObject())["task"] = 0;
  require(!materializeProgramHeterogeneousAssignment(source, bad,
                                                     {0, 0, 0, 0, 0}, p, error),
          "binding cannot silently target another task");
  bad = space;
  (*bad.dimensions[0].choices[1].binding.getAsObject())["unconsumed"] = true;
  require(!materializeProgramHeterogeneousAssignment(source, bad,
                                                     {0, 0, 0, 0, 0}, p, error),
          "unselected malformed choices also rejected");
  require(!materialize({0, 0, 0, 0, 2}, p), "out-of-range selection rejected");
  bad = space;
  bad.constraints.push_back({{0, 0}, {{0, 0}}});
  require(!materializeProgramHeterogeneousAssignment(source, bad,
                                                     {0, 0, 0, 0, 0}, p, error),
          "compatibility table cannot alias axes");
  auto frozen = options;
  frozen.taskDeviceChoices = {{"npu"}, {"npu"}, {"npu"}};
  frozen.boundaryChoices = {{false}, {true}};
  ProgramSearchSpace frozenSpace;
  require(buildProgramHeterogeneousSearchSpace(source, frozen, frozenSpace,
                                               error) &&
              materializeProgramHeterogeneousAssignment(
                  source, frozenSpace, {0, 0, 0, 0, 0}, p, error) &&
              p.slices.size() == 2,
          "frozen device/cut/join domains materialize exact boundaries");
  require(programHeterogeneousSearchSpaceIdentity(source, space) !=
              programHeterogeneousSearchSpaceIdentity(source, frozenSpace),
          "space identity includes domains and choices");
  // Explicit assumed test prices: whole graph compute + every transfer + one
  // launch per NPU region. These are not hardware-calibrated measurements.
  auto price = [&](double launch) {
    return ProgramHeterogeneousEvaluator(
        [&, launch](const HeterogeneousMappingPlan &plan) {
          ProgramSearchEvaluation e;
          e.feasible =
              checkHeterogeneousMapping(source, plan.choice, plan, error);
          if (!e.feasible) {
            e.feasibilityKnown = false;
            e.message = error;
            return e;
          }
          double cost = 0;
          for (auto &slice : plan.slices)
            cost += slice.device == "cpu"
                        ? .001 + .040 * (slice.endTask - slice.beginTask)
                        : launch + .002 * (slice.endTask - slice.beginTask);
          for (auto &event : plan.graph.events)
            if (event.kind == "transfer")
              cost += .001;
          e.costMs = cost;
          e.result = j::Object{
              {"test_price_kind", "explicit-assumed-complete-graph-cost"},
              {"marker", 17}};
          return e;
        });
  };
  auto low = price(.005), high = price(.200);
  double lowOracle = INFINITY, highOracle = INFINITY;
  std::set<std::string> uniqueGraphs;
  for (int mask = 0; mask < 32; ++mask) {
    std::vector<int64_t> a;
    for (int k = 0; k < 5; ++k)
      a.push_back((mask >> k) & 1);
    require(
        materialize(a, p),
        "every finite assignment materializes checked existing public graph");
    uniqueGraphs.insert(programGraphIdentity(p.graph));
    lowOracle = std::min(lowOracle, *low(p).costMs);
    highOracle = std::min(highOracle, *high(p).costMs);
  }
  require(uniqueGraphs.size() == 18,
          "32 assignments represent 18 actual region partitions including "
          "explicit-cut aliases");
  auto lowResult = searchProgramHeterogeneous(source, low, options),
       highResult = searchProgramHeterogeneous(source, high, options);
  if (constraintSearchAvailable()) {
    require(lowResult.status == ConstraintSearchStatus::Optimal &&
                highResult.status == ConstraintSearchStatus::Optimal,
            "known finite exhaustive domain proves oracle optimum");
    require(
        lowResult.hasBest && highResult.hasBest &&
            std::abs(*lowResult.bestEvaluation.costMs - lowOracle) < 1e-12 &&
            std::abs(*highResult.bestEvaluation.costMs - highOracle) < 1e-12,
        "solver matches independently enumerated full-cost oracle");
    HeterogeneousMappingPlan lowPlan, highPlan;
    require(materialize(lowResult.bestAssignment, lowPlan) &&
                materialize(highResult.bestAssignment, highPlan) &&
                lowPlan.choice.taskDevices ==
                    std::vector<std::string>({"npu", "npu", "npu"}) &&
                highPlan.choice.taskDevices ==
                    std::vector<std::string>({"cpu", "cpu", "cpu"}),
            "raising launch price changes optimal whole-region placement");
  } else
    require(lowResult.status == ConstraintSearchStatus::Unavailable &&
                lowResult.hasBest &&
                lowResult.bestAssignment ==
                    std::vector<int64_t>({0, 0, 0, 0, 0}),
            "solver-OFF keeps checked baseline and truthful UNAVAILABLE");
  auto *payload = lowResult.bestEvaluation.result.getAsObject();
  require(payload && payload->getObject("cost_evaluation") &&
              payload->getObject("cost_evaluation")->getInteger("marker") ==
                  17 &&
              payload->getObject("heterogeneous") &&
              payload->getObject("heterogeneous")->getString("space_id") ==
                  programHeterogeneousSearchSpaceIdentity(source, space),
          "caller payload and complete source/space/graph evidence preserved");
  auto budget = options;
  budget.search.maxEvaluations = 1;
  auto limited = searchProgramHeterogeneous(source, low, budget);
  require(limited.hasBest &&
              limited.bestAssignment == std::vector<int64_t>({0, 0, 0, 0, 0}) &&
              limited.status != ConstraintSearchStatus::Optimal,
          "finite evaluation cap preserves baseline without optimality claim");
  auto unknown = searchProgramHeterogeneous(
      source,
      [](const HeterogeneousMappingPlan &) {
        ProgramSearchEvaluation e;
        e.feasible = true;
        e.message = "complete invocation cost unavailable";
        return e;
      },
      options);
  require(unknown.status != ConstraintSearchStatus::Optimal &&
              !unknown.report.getBoolean("finite_domain_optimality_proved")
                   .value_or(false),
          "unknown complete cost blocks proof of optimality");
  std::string wide((*input)->getBuffer());
  for (size_t at = 0; (at = wide.find("f32", at)) != std::string::npos; at += 3)
    wide.replace(at, 3, "f64");
  auto wideModule = mlir::parseSourceString<mlir::ModuleOp>(wide, &context);
  GenericAnalysis wideSource;
  bool analyzed =
      wideModule && analyzeGenericProgram(*wideModule, wideSource, error);
  require(analyzed,
          "unsupported adapter dtype is still a valid analyzed source");
  if (analyzed) {
    ProgramSearchSpace wideSpace;
    require(buildProgramHeterogeneousSearchSpace(wideSource, options, wideSpace,
                                                 error),
            "typed search domain does not pretend unsupported dtype physically "
            "impossible");
    bool called = false;
    auto unsupported = evaluateProgramHeterogeneousAssignment(
        wideSource, wideSpace, {0, 0, 0, 0, 0},
        [&](const HeterogeneousMappingPlan &) {
          called = true;
          return ProgramSearchEvaluation{};
        });
    require(!called && !unsupported.feasibilityKnown && !unsupported.feasible &&
                unsupported.result.getAsObject()->getBoolean(
                    "physical_infeasibility_proved") == false,
            "existing adapter limitation is unknown with provenance, not "
            "proven infeasible");
  }
  auto noProvider = evaluateProgramHeterogeneousAssignment(source, space,
                                                           {0, 0, 0, 0, 0}, {});
  require(!noProvider.feasibilityKnown && !noProvider.feasible,
          "missing implementation provider remains unknown, not physical "
          "infeasibility");
  llvm::outs() << j::Value(j::Object{
                      {"status", ok ? "passed" : "failed"},
                      {"checks", checks},
                      {"solver_available", constraintSearchAvailable()},
                      {"raw_assignments", 32},
                      {"unique_region_graphs", int64_t(uniqueGraphs.size())},
                      {"scope", "CPU-only assumed complete region prices; no "
                                "hardware calibration or device execution"}})
               << "\n";
  return ok ? 0 : 1;
}
