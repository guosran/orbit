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

using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
j::Value read(const std::string &path) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    llvm::report_fatal_error("inactive-binding evidence missing");
  return llvm::cantFail(j::parse((*file)->getBuffer()));
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  auto source = llvm::MemoryBuffer::getFile(argv[1]);
  if (!source)
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.disableMultithreading();
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>((*source)->getBuffer(), &context);
  GenericAnalysis analysis;
  std::string error;
  if (!module || !analyzeGenericProgram(*module, analysis, error))
    return 2;
  llvm::SmallString<128> root;
  if (llvm::sys::fs::createUniqueDirectory("orbit-inactive-bindings", root))
    return 2;
  SearchOptions options;
  options.backend = "air";
  options.outputDir = root.str().str() + "/discovery";
  j::Value config =
      j::Object{{"program_search", j::Object{{"heterogeneous", true},
                                             {"seconds", 10},
                                             {"max_evaluations", 1},
                                             {"max_values_per_dimension", 3},
                                             {"max_routes", 2},
                                             {"max_priorities", 1}}}};
  (*config.getAsObject())["generic_air"] =
      j::Object{{"tile_sizes", j::Array{16, 1, 1}}};
  if (!runProgramAllSearch((*source)->getBuffer(), options, config, error)) {
    llvm::errs() << error;
    return 1;
  }
  auto space = read(options.outputDir + "/space.json");
  int checks = 0;
  auto require = [&](bool condition, const char *message) {
    ++checks;
    if (!condition)
      llvm::errs() << message << ": " << error << "\n";
    return condition;
  };
  // A resource demand above any actual target capacity, a mismatched shape,
  // and an undeclared NPU route destination must not constrain a CPU child.
  // The same bindings must still be ruled out when that child is NPU.
  for (bool customDeviceID : {false, true}) {
    j::Array dimensions;
    bool changedRoute = false;
    for (const auto &value : *space.getAsObject()->getArray("dimensions")) {
      const auto &dimension = *value.getAsObject();
      auto id = dimension.getString("id")->str();
      j::Value selected =
          *dimension.getArray("choices")->front().getAsObject()->get("binding");
      j::Array choices;
      if (id == "tiling.m")
        selected = j::Object{{"tile_axis", 0}, {"tile", 8}};
      if (id == "tiling.n")
        selected = j::Object{{"tile_axis", 1}, {"tile", 1}};
      if (id == "tiling.k")
        selected = j::Object{{"tile_axis", 2}, {"tile", 1}};
      if (id == "task.0.count")
        selected = j::Object{{"task", 0}, {"resource_count", 1000000}};
      if (id == "task.0.shape")
        selected = j::Object{{"task", 0}, {"resource_shape", j::Array{1}}};
      if (!changedRoute && id.rfind("task.0.data.", 0) == 0) {
        auto storage = id.substr(std::string("task.0.data.").size());
        storage = storage.substr(0, storage.find('.'));
        int64_t storageIndex = 0;
        if (llvm::StringRef(storage).getAsInteger(10, storageIndex))
          return 2;
        selected = j::Object{
            {"route", j::Object{{"storage", storageIndex},
                                {"consumer_task", 0},
                                {"source_memory", "inactive-source"},
                                {"target_memory", "inactive-destination"},
                                {"links", j::Array{}}}}};
        changedRoute = true;
      }
      if (customDeviceID && id == "device.task.1")
        selected = nullptr;
      if (id == "device.task.0") {
        if (customDeviceID)
          choices.push_back(nullptr);
        else {
          choices.push_back(j::Object{{"task", 0}, {"device", "cpu"}});
          choices.push_back(j::Object{{"task", 0}, {"device", "npu"}});
        }
      } else
        choices.push_back(std::move(selected));
      dimensions.push_back(
          j::Object{{"id", id}, {"choices", std::move(choices)}});
    }
    if (customDeviceID)
      dimensions.push_back(j::Object{
          {"id", "contract.actual-device-writer"},
          {"choices", j::Array{j::Object{{"task", 0}, {"device", "cpu"}},
                               j::Object{{"task", 0}, {"device", "npu"}},
                               j::Object{{"task", 1}, {"device", "npu"}}}}});
    if (!require(changedRoute, "fixture exercises a real source read route"))
      return 1;
    auto configuration = config;
    auto &search = *configuration.getAsObject()->getObject("program_search");
    search["dimensions"] = std::move(dimensions);
    search["max_evaluations"] = 4;
    options.outputDir =
        root.str().str() +
        (customDeviceID ? "/custom-device" : "/standard-device");
    int evaluated = 0;
    bool selectedNativeTileObserved = false;
    auto evaluate = [&](const ProgramAllHeterogeneousCandidate &candidate) {
      ++evaluated;
      ProgramSearchEvaluation result;
      result.feasible = true;
      result.feasibilityKnown = true;
      result.costMs = 7; // Assumed test oracle; no hardware pricing claim.
      j::Array children;
      for (const auto &region : candidate.regions) {
        result.feasible &= candidate.parent.choice.taskDevices[0] == "cpu";
        if (region.hasNativePlan)
          selectedNativeTileObserved |=
              region.native.tileSizes == std::vector<int64_t>({8, 1, 1}) &&
              region.native.phases.size() == 2;
        children.push_back(programGraphIdentity(region.selectedGraph));
      }
      result.result = j::Object{
          {"candidate_id", candidate.assignmentId},
          {"parent_graph_id", programGraphIdentity(candidate.parent.graph)},
          {"region_graph_ids", std::move(children)}};
      return result;
    };
    if (!require(runProgramAllSearch((*source)->getBuffer(), options,
                                     configuration, error, evaluate),
                 "inactive NPU bindings admit the checked CPU assignment"))
      return 1;
    auto report = read(options.outputDir + "/search.json");
    const auto &summary = *report.getAsObject();
    const int expected =
        customDeviceID && summary.getString("status") != "UNAVAILABLE" ? 2 : 1;
    if (!require(
            evaluated == expected && summary.getBoolean("has_best") == true &&
                summary.getInteger("feasible_evaluations") == expected,
            "only bindings active for their actual task constrain the oracle"))
      return 1;
    if (customDeviceID && summary.getString("status") != "UNAVAILABLE" &&
        !require(
            selectedNativeTileObserved,
            "selected source tile overrides inherited positive config tiles"))
      return 1;
    if (summary.getString("status") != "UNAVAILABLE" &&
        !require(summary.getBoolean("finite_domain_exhausted") == true &&
                     summary.getBoolean("finite_domain_optimality_proved") ==
                         true,
                 "active NPU binding is excluded without discarding CPU plan"))
      return 1;
  }
  llvm::outs() << "program inactive bindings checks: " << checks << "\n";
  return 0;
}
