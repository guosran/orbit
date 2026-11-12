#include "Taskflow/HeterogeneousSearch.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramAllSearch.h"
#include "Taskflow/ProgramMatrixImplementation.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
// Only the compiler-issued ABI bridge is exercised here. This synthetic AIR
// receipt never claims a compiled ELF, hardware validation or measured cost.
bool emit(const GenericAnalysis &source, const GenericMappingPlan &plan,
          const std::string &path, std::string &error) {
  j::Array ids, buffers;
  for (size_t storage = 0; storage < source.storages.size(); ++storage) {
    const auto &item = source.storages[storage];
    j::Array shape;
    for (auto extent : item.shape)
      shape.push_back(extent);
    ids.push_back(int64_t(storage));
    buffers.push_back(j::Object{{"name", item.id},
                                {"dtype", item.dtype},
                                {"role", item.role},
                                {"shape", std::move(shape)}});
  }
  std::error_code ec;
  llvm::raw_fd_ostream out(path + ".physical.json", ec);
  if (ec) {
    error = ec.message();
    return false;
  }
  out << llvm::formatv(
      "{0:2}",
      j::Value(j::Object{{"graph_id", programGraphIdentity(plan.communication)},
                         {"abi_storage_indices", std::move(ids)},
                         {"abi", j::Object{{"buffers", std::move(buffers)}}}}));
  return true;
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
  auto source = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!source)
    return 2;
  source->getOperation()->setAttr(
      "orbit.implementation",
      mlir::StringAttr::get(&context, programMatrixImplementation));
  j::Value config = j::Object{
      {"generic_air",
       j::Object{
           {"capabilities",
            j::Object{{"implementations",
                       j::Array{j::Object{{"kind", "scalar"}, {"lanes", 1}},
                                j::Object{{"kind", programMatrixImplementation},
                                          {"lanes", 1}}}}}}}}};
  GenericAnalysis analysis;
  GenericMappingPlan reference;
  std::string error;
  int checks = 0;
  auto require = [&](bool value, const char *label) {
    ++checks;
    if (!value) {
      llvm::errs() << label << ": " << error << "\n";
      std::exit(1);
    }
    error.clear();
  };
  require(distributeNativeInitialization(*source, error),
          "materialize source initializer before read/write matrix task "
          "projection");
  require(analyzeGenericProgram(*source, analysis, error) &&
              analysis.tasks.size() == 3,
          "analyze the actual BF16 matrix followed by FP32 relu");
  ProgramMatrixTaskContract matrix;
  require(analyzeProgramMatrixTask(analysis, 1, matrix, error),
          "first task is independently proved eligible for the matrix "
          "implementation");
  require(!analyzeProgramMatrixTask(analysis, 2, matrix, error),
          "relu source is independently proved ineligible for the matrix "
          "implementation");
  require(planGenericMapping(analysis, config, reference, error),
          "whole-source selected matrix implementation has a native witness");
  ProgramDataOrganizationChoice data;
  data.policy = reference.communication.dataPolicy;
  for (size_t task = 0; task < analysis.tasks.size(); ++task)
    data.placements.push_back({task, reference.memoryBinding.computeResource,
                               reference.memoryBinding.localMemory, "scalar"});
  llvm::SmallString<128> directory;
  if (llvm::sys::fs::createUniqueDirectory(
          "orbit-implementation-projection-test", directory))
    return 2;
  auto test = [&](std::vector<std::string> devices, const std::string &label) {
    HeterogeneousMappingChoice choice;
    choice.taskDevices = std::move(devices);
    ProgramAllHeterogeneousCandidate candidate;
    require(materializeProgramAllHeterogeneousCandidate(
                *source, config, choice, reference.memoryBinding.architecture,
                data, {0, 1, 2}, false, 128, candidate, error) &&
                candidate.regions.size() == 2,
            "joint candidate retains matrix/relu CPU/NPU device assignment");
    auto &matrixRegion = candidate.regions[0];
    auto &reluRegion = candidate.regions[1];
    require(
        matrixRegion.source->getOperation()
                ->getAttrOfType<mlir::StringAttr>("orbit.implementation")
                .getValue() == programMatrixImplementation,
        "eligible matrix child retains the selected implementation request");
    require(
        !reluRegion.source->getOperation()->hasAttr("orbit.implementation") &&
            reluRegion.source->getOperation()
                    ->getAttrOfType<mlir::StringAttr>(
                        "orbit.parent_implementation")
                    .getValue() == programMatrixImplementation,
        "ineligible epilogue projects the parent request as inactive "
        "provenance");
    require(source->getOperation()
                    ->getAttrOfType<mlir::StringAttr>("orbit.implementation")
                    .getValue() == programMatrixImplementation,
            "regional projection does not rewrite the parent implementation "
            "choice");
    for (const auto &region : candidate.regions) {
      require(
          (*region.source)
                  .lookupSymbol<mlir::func::FuncOp>(
                      "heterogeneous_region_" + std::to_string(region.region))
                  ->getAttrOfType<mlir::StringAttr>("orbit.matrix_contract")
                  .getValue() == programMatrixNumericalContract,
          "source numerical contract survives regional projection");
      if (!region.hasNativePlan)
        continue;
      require(checkGenericMapping(region.native.communication.source,
                                  region.native, error),
              "selected native graph remains independently checked");
      size_t selectedMatrix = 0;
      for (const auto &implementation : region.native.taskImplementations)
        selectedMatrix += implementation.kind == programMatrixImplementation;
      require(region.region == 0 ? selectedMatrix == 1 : selectedMatrix == 0,
              "eligible NPU matrix stays matrix; NPU epilogue uses actual "
              "scalar fallback");
    }
    std::vector<HeterogeneousSelectedRegion> selected;
    for (const auto &region : candidate.regions)
      selected.push_back(
          {*region.source, region.hasNativePlan ? &region.native : nullptr});
    j::Value manifest = nullptr;
    require(exportHeterogeneousProgram(
                *source, config, directory.str().str() + "/" + label, manifest,
                error, emit, &candidate.parent, &selected),
            "export consumes the exact typed parent and selected native child");
    const auto *regions = manifest.getAsObject()->getArray("regions");
    for (size_t region = 0; region < candidate.regions.size(); ++region)
      require((*regions)[region].getAsObject()->getString("child_graph_id") ==
                  programGraphIdentity(candidate.regions[region].selectedGraph),
              "backend receipt identifies the same selected child graph");
  };
  test({"cpu", "cpu", "npu"}, "matrix-cpu-relu-npu");
  test({"npu", "npu", "cpu"}, "matrix-npu-relu-cpu");
  llvm::outs() << "heterogeneous-implementation-projection: " << checks
               << " checks passed (CPU synthetic ABI only)\n";
  return 0;
}
