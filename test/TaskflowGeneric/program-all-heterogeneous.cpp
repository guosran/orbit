#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramAllSearch.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>
#include <set>
using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
j::Value read(const std::string &path) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    llvm::report_fatal_error("combined test evidence missing");
  return llvm::cantFail(j::parse((*file)->getBuffer()));
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
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
  GenericAnalysis analysis;
  GenericMappingPlan reference;
  std::string error;
  if (!module || !analyzeGenericProgram(*module, analysis, error) ||
      !planGenericMapping(analysis, j::Object{}, reference, error)) {
    llvm::errs() << error << "\n";
    return 1;
  }
  int checks = 0;
  bool ok = true;
  auto require = [&](bool condition, const char *label) {
    ++checks;
    if (!condition) {
      ok = false;
      llvm::errs() << label << ": " << error << "\n";
    }
    error.clear();
  };
  ProgramDataOrganizationChoice publicChoice;
  publicChoice.policy = reference.communication.dataPolicy;
  for (size_t task = 0; task < analysis.tasks.size(); ++task)
    publicChoice.placements.push_back(
        {task, reference.memoryBinding.computeResource,
         reference.memoryBinding.localMemory, "scalar"});
  std::vector<int64_t> priority(analysis.tasks.size());
  std::iota(priority.begin(), priority.end(), 0);
  const auto target = reference.memoryBinding.architecture;
  auto build = [&](const HeterogeneousMappingChoice &choice,
                   ProgramAllHeterogeneousCandidate &candidate,
                   int64_t cap = 128) {
    return materializeProgramAllHeterogeneousCandidate(
        *module, j::Object{}, choice, target, publicChoice, priority, false,
        cap, candidate, error);
  };
  HeterogeneousMappingChoice mixed;
  mixed.taskDevices = {"cpu", "npu", "cpu"};
  ProgramAllHeterogeneousCandidate candidate;
  require(build(mixed, candidate) && candidate.regions.size() == 3,
          "CPU/NPU/CPU is one checked parent with three actual child "
          "implementations");
  if (!ok)
    return 1;
  require(candidate.parent.choice.implementationOwned &&
              !candidate.regions[0].hasNativePlan &&
              candidate.regions[1].hasNativePlan &&
              !candidate.regions[2].hasNativePlan,
          "only native child owns NPU L1 and DMA refinement");
  require(
      candidate.regions[1].projection.childGraphId ==
              programGraphIdentity(candidate.regions[1].native.communication) &&
          candidate.regions[1].projection.childGraphId ==
              programGraphIdentity(candidate.regions[1].selectedGraph),
      "projected child, selected graph and backend native plan are identical");
  auto sourcePriority = priority;
  std::reverse(priority.begin(), priority.end());
  require(!build(mixed, candidate) &&
              error.find("cross-region temporal priority") != std::string::npos,
          "cross-region priority cannot be silently replaced by source-order "
          "execution");
  priority = std::move(sourcePriority);
  require(setNativeAIRTileSizes(*module, 8, 1, 1, error),
          "select real native tile");
  require(
      build(mixed, candidate) &&
          candidate.regions[1].native.strategy == "serial-tiled" &&
          candidate.regions[1].native.phases.size() == 2,
      "CPU/NPU partition retains the selected two-tile native implementation");
  require(!build(mixed, candidate, 1) &&
              error.find("work cap") != std::string::npos,
          "unsupported materialization cap is retained rather than replacing "
          "selected tile");
  HeterogeneousMappingChoice allCPU;
  allCPU.taskDevices.assign(analysis.tasks.size(), "cpu");
  require(build(allCPU, candidate) && candidate.regions.size() == 1,
          "all-CPU has one complete call");
  allCPU.cuts = {1};
  require(
      build(allCPU, candidate) && candidate.regions.size() == 2,
      "same-device cut selects a second call without changing source tasks");
  module->getOperation()->setAttr(
      "orbit.program_partition_extent",
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 8));
  require(!build(mixed, candidate) &&
              error.find("assembly") != std::string::npos,
          "missing host partition assembly is explicit unsupported refinement");
  module->getOperation()->removeAttr("orbit.program_partition_extent");

  llvm::SmallString<128> directory;
  if (llvm::sys::fs::createUniqueDirectory("orbit-combined-search-test",
                                           directory))
    return 2;
  SearchOptions options;
  options.backend = "air";
  options.outputDir = directory.str().str() + "/discovery";
  j::Value config =
      j::Object{{"program_search", j::Object{{"heterogeneous", true},
                                             {"seconds", 10},
                                             {"max_evaluations", 1},
                                             {"max_values_per_dimension", 2},
                                             {"max_routes", 2},
                                             {"max_priorities", 1}}}};
  require(runProgramAllSearch((*input)->getBuffer(), options, config, error),
          "joined finite domain is generated without a runtime price provider");
  if (!ok)
    return 1;
  auto space = read(options.outputDir + "/space.json");
  j::Array dimensions;
  for (const auto &value : *space.getAsObject()->getArray("dimensions")) {
    auto &dimension = *value.getAsObject();
    auto id = dimension.getString("id")->str();
    const auto &first = dimension.getArray("choices")->front();
    j::Array choices{j::Value(*first.getAsObject()->get("binding"))};
    if (id == "device.task.0")
      choices.push_back(j::Object{{"task", 0}, {"device", "npu"}});
    if (id == "boundary.before.1")
      choices.push_back(j::Object{{"before_task", 1}, {"cut", true}});
    if (id == "tiling.m")
      choices.push_back(j::Object{{"tile_axis", 0}, {"tile", 8}});
    dimensions.push_back(
        j::Object{{"id", id}, {"choices", std::move(choices)}});
  }
  auto *search = config.getAsObject()->getObject("program_search");
  (*search)["dimensions"] = std::move(dimensions);
  (*search)["max_evaluations"] = 16;
  options.outputDir = directory.str().str() + "/joint";
  std::set<std::pair<bool, bool>> controls;
  size_t evaluated = 0;
  auto evaluate = [&](const ProgramAllHeterogeneousCandidate &actual) {
    ++evaluated;
    ProgramSearchEvaluation result;
    bool valid = true, hasNPU = false, tiled = false;
    double syntheticFullCalls = 0;
    for (const auto &region : actual.regions) {
      valid &= region.projection.parentGraphId ==
               programGraphIdentity(actual.parent.graph);
      valid &= region.projection.childGraphId ==
               programGraphIdentity(region.selectedGraph);
      if (region.hasNativePlan) {
        hasNPU = true;
        tiled |= region.native.phases.size() > 1;
        valid &= programGraphIdentity(region.native.communication) ==
                 programGraphIdentity(region.selectedGraph);
        valid &= checkGenericMapping(region.native.communication.source,
                                     region.native, error);
        syntheticFullCalls += 40. / region.selectedGraph.instances.size();
      } else
        syntheticFullCalls += 100. * region.projection.parentTasks.size();
    }
    controls.emplace(hasNPU, tiled);
    require(
        valid,
        "callback receives exact checked source/native/projection assignment");
    result.feasible = valid;
    result.feasibilityKnown = valid;
    // A synthetic complete-call cost is an explicit test fixture, never a
    // physical calibration. It is owned once per call, with global work once.
    result.costMs = syntheticFullCalls + 7 * actual.regions.size() + 3;
    j::Array childIDs;
    for (const auto &region : actual.regions)
      childIDs.push_back(programGraphIdentity(region.selectedGraph));
    result.result = j::Object{
        {"scope", "synthetic CPU-only complete-call contract"},
        {"candidate_id", actual.assignmentId},
        {"parent_graph_id", programGraphIdentity(actual.parent.graph)},
        {"region_graph_ids", std::move(childIDs)}};
    return result;
  };
  require(runProgramAllSearch((*input)->getBuffer(), options, config, error,
                              evaluate),
          "one C++ master searches native tiling and CPU/NPU cuts together");
  auto report = read(options.outputDir + "/search.json");
  auto *summary = report.getAsObject();
  require(summary->getBoolean("combined_cpu_npu_native") == true,
          "search result reports the joined domain explicitly");
  auto status = summary->getString("status");
  if (status != "UNAVAILABLE") {
    require(controls.count({false, false}) && controls.count({true, false}) &&
                controls.count({true, true}),
            "both native tiles are actually evaluated jointly with CPU/NPU "
            "placement");
    require(evaluated == 8 &&
                summary->getBoolean("finite_domain_exhausted") == true &&
                summary->getBoolean("finite_domain_optimality_proved") == true,
            "all eight complete assignments are checked by one master");
  }
  auto best = read(options.outputDir + "/best.json");
  require(best.getAsObject()->getObject("combined_candidate") &&
              best.getAsObject()->getObject("cost_evaluation") &&
              best.getAsObject()->getBoolean("backend_replanning_allowed") ==
                  false,
          "best evidence binds costs and all selected child plans to one "
          "assignment");
  // A finite price must not be accepted if the backend receipt describes a
  // different parent/child assignment, nor may invalid latency become zero.
  auto invalidConfiguration = config;
  j::Array singletonDimensions;
  for (const auto &value : *search->getArray("dimensions")) {
    const auto &dimension = *value.getAsObject();
    singletonDimensions.push_back(j::Object{
        {"id", dimension.getString("id")->str()},
        {"choices",
         j::Array{j::Value(dimension.getArray("choices")->front())}}});
  }
  auto *invalidSearch =
      invalidConfiguration.getAsObject()->getObject("program_search");
  (*invalidSearch)["dimensions"] = std::move(singletonDimensions);
  (*invalidSearch)["max_evaluations"] = 1;
  for (const auto &field : {"candidate_id", "parent_graph_id",
                            "region_graph_ids", "negative_latency"}) {
    options.outputDir = directory.str().str() + "/rejected-" + field;
    auto corrupt = [&](const ProgramAllHeterogeneousCandidate &actual) {
      auto result = evaluate(actual);
      if (std::string(field) == "negative_latency")
        result.costMs = -1;
      else
        (*result.result.getAsObject())[field] =
            std::string(field) == "region_graph_ids"
                ? j::Value(j::Array{"foreign-child"})
                : j::Value("foreign-assignment");
      return result;
    };
    require(runProgramAllSearch((*input)->getBuffer(), options,
                                invalidConfiguration, error, corrupt),
            "bad backend/cost receipt produces a retained unknown result");
    auto rejected = read(options.outputDir + "/search.json");
    require(rejected.getAsObject()->getBoolean("has_best") == false &&
                rejected.getAsObject()->getBoolean(
                    "finite_domain_optimality_proved") == false &&
                rejected.getAsObject()->getInteger(
                    "unknown_feasibility_evaluations") == 1,
            "foreign receipt or invalid price cannot produce an incumbent or "
            "optimum");
  }
  llvm::outs() << "program all heterogeneous checks: " << checks << "\n";
  return ok ? 0 : 1;
}
