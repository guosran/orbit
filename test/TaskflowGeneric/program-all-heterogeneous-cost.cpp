#include "Taskflow/HeterogeneousImplementation.h"
#include "Taskflow/ProgramAllHeterogeneousCost.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
void write(const std::string &path, const j::Value &value) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec);
  if (ec)
    llvm::report_fatal_error("test evidence write failed");
  out << llvm::formatv("{0:2}", value);
}
j::Value phase(double value = 1, llvm::StringRef feature = "call") {
  // Synthetic ownership oracle only: these numbers are not hardware
  // calibration.
  return j::Object{
      {"coefficients_ns",
       j::Object{
           {feature,
            j::Object{{"value", value},
                      {"kind", value == 0 ? "architecture" : "calibrated"},
                      {"source",
                       "synthetic CPU contract test; no hardware claim"}}}}},
      {"ranges", j::Object{{feature, j::Array{0, 1000000000}}}}};
}
j::Value modelFor(const ProgramAllHeterogeneousCandidate &candidate,
                  const std::string &contractIdentity) {
  j::Array rows;
  std::set<std::string> seen;
  for (const auto &region : candidate.regions) {
    j::Array kinds;
    std::set<std::string> names;
    for (const auto &instance : region.selectedGraph.instances)
      names.insert(instance.implementation);
    for (const auto &name : names)
      kinds.push_back(name);
    auto device = candidate.parent.slices[region.region].device;
    auto family = heterogeneousCostSourceFamily(*region.source);
    auto key = device + family + staticFingerprint(j::Value(j::Array(kinds)));
    if (!seen.insert(key).second)
      continue;
    j::Object phases;
    for (const auto *name : {"body", "sync_to", "submit", "wait", "sync_from"})
      phases[name] = phase();
    rows.push_back(j::Object{{"device", device},
                             {"source_family", family},
                             {"implementations", std::move(kinds)},
                             {"phases", std::move(phases)}});
  }
  return j::Object{
      {"schema", "orbit.heterogeneous.edge-cost.v32"},
      {"runtime_identity", "orbit.serial-prepared-typed-native-wait2.v32"},
      {"execution_contract_identity", contractIdentity},
      {"application", j::Object{{"reset", phase(0)},
                                {"control", phase(0)},
                                {"copy", phase(0)}}},
      {"regions", std::move(rows)}};
}
// The synthetic emitter models a public graph whose real ABI includes private
// source temporaries. No compiler, ELF, XRT context or device is invoked.
bool emit(const GenericAnalysis &facts, const GenericMappingPlan &native,
          const std::string &path, std::string &) {
  j::Array ids, buffers;
  for (size_t index = 0; index < facts.storages.size(); ++index) {
    ids.push_back(int64_t(index));
    const auto &storage = facts.storages[index];
    j::Array shape;
    for (auto extent : storage.shape)
      shape.push_back(extent);
    buffers.push_back(j::Object{{"name", storage.id},
                                {"dtype", storage.dtype},
                                {"shape", std::move(shape)},
                                {"role", storage.role}});
  }
  write(path + ".physical.json",
        j::Object{{"graph_id", programGraphIdentity(native.communication)},
                  {"abi_storage_indices", std::move(ids)},
                  {"abi", j::Object{{"buffers", std::move(buffers)}}}});
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
  auto file = llvm::MemoryBuffer::getFile(argv[1]);
  if (!file)
    return 2;
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>((*file)->getBuffer(), &context);
  GenericAnalysis analysis;
  GenericMappingPlan reference;
  std::string error;
  if (!module || !analyzeGenericProgram(*module, analysis, error) ||
      !planGenericMapping(analysis, j::Object{}, reference, error))
    return 2;
  llvm::SmallString<128> directory;
  if (llvm::sys::fs::createUniqueDirectory("orbit-edge-cost-test", directory))
    return 2;
  const auto root = directory.str().str();
  j::Value contract = j::Object{
      {"runtime_identity", "orbit.serial-prepared-typed-native-wait2.v32"},
      {"provenance", "synthetic CPU test only; no hardware claim"}};
  write(root + "/contract.json", contract);
  j::Value config = j::Object{{"execution_contract", root + "/contract.json"}};
  ProgramDataOrganizationChoice data;
  data.policy = reference.communication.dataPolicy;
  for (size_t task = 0; task < analysis.tasks.size(); ++task)
    data.placements.push_back({task, reference.memoryBinding.computeResource,
                               reference.memoryBinding.localMemory, "scalar"});
  std::vector<int64_t> priority(analysis.tasks.size());
  std::iota(priority.begin(), priority.end(), 0);
  int checks = 0;
  auto require = [&](bool condition, const char *message) {
    ++checks;
    if (!condition) {
      llvm::errs() << message << ": " << error << "\n";
      std::exit(1);
    }
  };
  auto familyClone = mlir::OwningOpRef<mlir::ModuleOp>(module->clone());
  const auto sourceFamily = heterogeneousCostSourceFamily(*familyClone);
  familyClone->getOperation()->setAttr(
      "orbit.air_tile_sizes",
      mlir::DenseI64ArrayAttr::get(&context, {8, 1, 1}));
  familyClone->getOperation()->setAttr(
      "orbit.mapping", mlir::StringAttr::get(&context, "changed"));
  familyClone->getOperation()->setAttr(
      "orbit.implementation", mlir::StringAttr::get(&context, "scalar"));
  familyClone->getOperation()->setAttr(
      "orbit.parent_implementation",
      mlir::StringAttr::get(&context, "inactive"));
  familyClone->getOperation()->setAttr(
      "orbit.vector_lanes",
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 4));
  require(heterogeneousCostSourceFamily(*familyClone) == sourceFamily,
          "local source family ignores selected tile/mapping/implementation "
          "attributes");
  auto familyFunction = *familyClone->getOps<mlir::func::FuncOp>().begin();
  familyFunction->setAttr(
      "orbit.matrix_contract",
      mlir::StringAttr::get(&context, "bf16-f32-accumulate-v1"));
  const auto numericFamily = heterogeneousCostSourceFamily(*familyClone);
  familyFunction->removeAttr("orbit.matrix_contract");
  require(numericFamily != heterogeneousCostSourceFamily(*familyClone),
          "removing the source numerical contract changes local calibration "
          "family");
  ProgramAllHeterogeneousCandidate candidate;
  auto build = [&](std::vector<std::string> devices) {
    HeterogeneousMappingChoice choice;
    choice.taskDevices = std::move(devices);
    require(materializeProgramAllHeterogeneousCandidate(
                *module, config, choice, reference.memoryBinding.architecture,
                data, priority, false, 128, candidate, error),
            "typed candidate must materialize");
    candidate.evidenceDirectory = root + "/case-" + std::to_string(checks);
    config.getAsObject()->operator[]("heterogeneous_cost") =
        modelFor(candidate, staticFingerprint(contract));
  };
  auto evaluate = [&] {
    return evaluateProgramAllHeterogeneousCost(candidate, config, emit);
  };
  build({"npu", "npu", "npu"});
  auto result = evaluate();
  if (!result.costMs)
    llvm::errs() << result.message << "\n"
                 << llvm::formatv("{0:2}\n", result.result);
  require(
      result.costMs && std::abs(*result.costMs - 4e-6) < 1e-12,
      "one NPU region pays exactly four phases, no extra child compute fee");
  auto &features = *(*result.result.getAsObject()
                          ->getObject("features")
                          ->getArray("regions"))[0]
                        .getAsObject()
                        ->getObject("features");
  require(features.getNumber("abi_bo_count") == 5 &&
              features.getNumber("abi_bytes") == 320 &&
              features.getNumber("to_device_bytes") == 320 &&
              features.getNumber("from_device_bytes") == 64,
          "physical five-BO ABI includes T/U scratch beyond three projected "
          "public arguments");
  auto fiveConfigurations = [](const GenericAnalysis &facts,
                               const GenericMappingPlan &native,
                               const std::string &path, std::string &error) {
    if (!emit(facts, native, path, error))
      return false;
    auto file = llvm::MemoryBuffer::getFile(path + ".physical.json");
    if (!file)
      return false;
    auto physical = llvm::cantFail(j::parse((*file)->getBuffer()));
    (*physical.getAsObject())["physical_invocation_count"] = 5;
    write(path + ".physical.json", physical);
    return true;
  };
  auto splitCost = evaluateProgramAllHeterogeneousCost(candidate, config,
                                                       fiveConfigurations);
  auto limitedContract = contract;
  (*limitedContract.getAsObject())["max_prepared_npu_contexts"] = 4;
  write(root + "/contract.json", limitedContract);
  auto exhaustedContexts = evaluateProgramAllHeterogeneousCost(
      candidate, config, fiveConfigurations);
  require(!exhaustedContexts.costMs &&
              exhaustedContexts.message.find("prepared NPU context capacity") !=
                  std::string::npos,
          "physical context exhaustion rejects export before cost evaluation");
  (*limitedContract.getAsObject())["max_prepared_npu_contexts"] = 5;
  write(root + "/contract.json", limitedContract);
  auto exactContexts = evaluateProgramAllHeterogeneousCost(candidate, config,
                                                           fiveConfigurations);
  require(exactContexts.message.find("backend rejected") == std::string::npos,
          "exact context capacity remains structurally supported");
  write(root + "/contract.json", contract);
  require(
      !splitCost.costMs,
      "five physical configurations cannot borrow a single-launch phase price");
  const auto oneConfigurationModel = config;
  auto *splitPhases = (*config.getAsObject()
                            ->getObject("heterogeneous_cost")
                            ->getArray("regions"))[0]
                          .getAsObject()
                          ->getObject("phases");
  for (const auto *name : {"submit", "wait"}) {
    auto *phaseModel = splitPhases->getObject(name);
    (*phaseModel->getObject("coefficients_ns"))["configuration_count"] =
        j::Object{
            {"value", 2},
            {"kind", "calibrated"},
            {"source", "synthetic per-configuration test; no hardware claim"}};
    (*phaseModel->getObject("ranges"))["configuration_count"] = j::Array{1, 5};
  }
  splitCost = evaluateProgramAllHeterogeneousCost(candidate, config,
                                                  fiveConfigurations);
  require(splitCost.costMs && std::abs(*splitCost.costMs - 24e-6) < 1e-12,
          "five physical configurations price submit and wait count terms "
          "independently");
  const auto *splitFeatures = (*splitCost.result.getAsObject()
                                    ->getObject("features")
                                    ->getArray("regions"))[0]
                                  .getAsObject()
                                  ->getObject("features");
  require(splitFeatures->getNumber("configuration_count") == 5 &&
              splitCost.result.getAsObject()
                      ->getObject("features")
                      ->getObject("application")
                      ->getNumber("configuration_count") == 5,
          "regional and global control features preserve exact physical "
          "invocation count");
  require(splitFeatures->getNumber("configuration_argument_bindings") == 25,
          "five configurations each bind all five physical ABI arguments, "
          "including scratch independently of payload synchronization");
  const auto splitModel = config;
  for (const auto *name : {"submit", "wait"}) {
    auto *phases = (*config.getAsObject()
                         ->getObject("heterogeneous_cost")
                         ->getArray("regions"))[0]
                       .getAsObject()
                       ->getObject("phases");
    phases->getObject(name)
        ->getObject("coefficients_ns")
        ->erase("configuration_count");
    auto missingCount = evaluateProgramAllHeterogeneousCost(candidate, config,
                                                            fiveConfigurations);
    require(!missingCount.costMs, "missing either submission or completion "
                                  "configuration term stays unknown");
    config = splitModel;
  }
  config = oneConfigurationModel;
  auto *model = config.getAsObject()->getObject("heterogeneous_cost");
  auto *phases =
      (*model->getArray("regions"))[0].getAsObject()->getObject("phases");
  (*phases)["sync_to"] = phase(1, "abi_bytes");
  result = evaluate();
  require(result.costMs && std::abs(*result.costMs - 323e-6) < 1e-12,
          "physical ABI bytes, including private temporaries, drive phase "
          "prediction");
  build({"npu", "cpu", "npu"});
  (*limitedContract.getAsObject())["max_prepared_npu_contexts"] = 1;
  write(root + "/contract.json", limitedContract);
  auto sharedContexts = evaluate();
  require(!sharedContexts.costMs &&
              sharedContexts.message.find("prepared NPU context capacity") !=
                  std::string::npos,
          "context capacity is shared across separate NPU regions");
  write(root + "/contract.json", contract);
  result = evaluate();
  require(result.costMs && std::abs(*result.costMs - 9e-6) < 1e-12,
          "two NPU regions each launch/sync once and CPU region pays one body");
  auto *timing = result.result.getAsObject()->getObject("timing");
  auto *activities = timing->getArray("activities");
  require(activities && activities->size() == 12,
          "three global application phases occur only once");
  std::set<int64_t> events, lifecycle;
  bool unique = true;
  std::string previous;
  for (const auto &value : *activities) {
    const auto *activity = value.getAsObject();
    for (const auto &event : *activity->getArray("events"))
      unique &= events.insert(*event.getAsInteger()).second;
    for (const auto &event : *activity->getArray("region_events"))
      unique &= lifecycle.insert(*event.getAsInteger()).second;
    auto *predecessors = activity->getArray("predecessors");
    // Timing additionally preserves graph-derived data dependencies. Require
    // the explicit serial chain without forbidding those necessary edges.
    bool hasPrevious = previous.empty() && predecessors->empty();
    for (const auto &predecessor : *predecessors)
      hasPrevious |= predecessor.getAsString() == previous;
    unique &= hasPrevious;
    previous = activity->getString("id")->str();
  }
  if (!unique || events.size() != candidate.parent.graph.events.size() ||
      lifecycle.size() != candidate.parent.graph.regionEvents.size())
    llvm::errs() << "ownership unique=" << unique << " events=" << events.size()
                 << "/" << candidate.parent.graph.events.size()
                 << " lifecycle=" << lifecycle.size() << "/"
                 << candidate.parent.graph.regionEvents.size() << "\n"
                 << llvm::formatv("{0:2}\n", result.result);
  require(unique && events.size() == candidate.parent.graph.events.size() &&
              lifecycle.size() == candidate.parent.graph.regionEvents.size(),
          "every graph and lifecycle event has one owner with serial "
          "cross-region dependencies");
  const auto valid = config;
  auto unknown = [&](const char *label) {
    auto value = evaluate();
    require(value.feasible && value.feasibilityKnown && !value.costMs, label);
    config = valid;
  };
  config.getAsObject()->erase("heterogeneous_cost");
  unknown("missing runtime model preserves feasible but unpriced candidate");
  (*config.getAsObject()->getObject("heterogeneous_cost"))["runtime_identity"] =
      "foreign-runtime";
  unknown("foreign runtime cannot borrow measured phase prices");
  (*config.getAsObject()->getObject(
      "heterogeneous_cost"))["execution_contract_identity"] =
      "foreign-contract";
  unknown("foreign execution contract remains unknown");
  auto foreignContract = contract;
  (*foreignContract.getAsObject())["runtime_identity"] = "foreign-runtime";
  write(root + "/contract.json", foreignContract);
  (*config.getAsObject()->getObject(
      "heterogeneous_cost"))["execution_contract_identity"] =
      staticFingerprint(foreignContract);
  unknown("matching hash does not authorize a foreign runtime contract");
  write(root + "/contract.json", contract);
  auto mutateFirstPhase = [&]() -> j::Object & {
    return *(*config.getAsObject()
                  ->getObject("heterogeneous_cost")
                  ->getArray("regions"))[0]
                .getAsObject()
                ->getObject("phases");
  };
  mutateFirstPhase().erase("submit");
  unknown("missing launch price cannot become architectural zero");
  mutateFirstPhase()["sync_to"] =
      j::Object{{"coefficients_ns", j::Object{}}, {"ranges", j::Object{}}};
  unknown("empty coefficient table cannot price a phase");
  mutateFirstPhase()["wait"] = phase(-1);
  unknown("negative coefficient is not a feasible duration");
  auto bad = phase();
  (*bad.getAsObject()->getObject("ranges"))["call"] = j::Array{2, 3};
  mutateFirstPhase()["wait"] = std::move(bad);
  unknown("out-of-training-range features remain unknown");
  bad = phase();
  bad.getAsObject()->getObject("ranges")->erase("call");
  mutateFirstPhase()["wait"] = std::move(bad);
  unknown("missing calibration range remains unknown");
  bad = phase();
  (*bad.getAsObject()->getObject("coefficients_ns"))["call"]
      .getAsObject()
      ->erase("source");
  mutateFirstPhase()["wait"] = std::move(bad);
  unknown("coefficient without provenance remains unknown");
  bad = phase();
  (*bad.getAsObject()->getObject("coefficients_ns"))["call"].getAsObject()->
  operator[]("kind") = "assumed";
  mutateFirstPhase()["wait"] = std::move(bad);
  unknown("assumed coefficients cannot masquerade as calibrated architecture "
          "costs");
  mutateFirstPhase()["sync_to"] =
      phase(std::numeric_limits<double>::max(), "abi_bytes");
  unknown("overflowing feature times coefficient remains unknown");
  auto corruptPhysicalABI = [](const GenericAnalysis &facts,
                               const GenericMappingPlan &native,
                               const std::string &path, std::string &error) {
    if (!emit(facts, native, path, error))
      return false;
    auto file = llvm::MemoryBuffer::getFile(path + ".physical.json");
    if (!file)
      return false;
    auto physical = llvm::cantFail(j::parse((*file)->getBuffer()));
    auto *buffers =
        physical.getAsObject()->getObject("abi")->getArray("buffers");
    (*buffers)[0].getAsObject()->operator[]("dtype") = "bf16";
    write(path + ".physical.json", physical);
    return true;
  };
  auto corrupt = evaluateProgramAllHeterogeneousCost(candidate, config,
                                                     corruptPhysicalABI);
  require(!corrupt.costMs, "matching child graph ID cannot bless inconsistent "
                           "physical ABI geometry");
  auto noPhysicalReceipt =
      evaluateProgramAllHeterogeneousCost(candidate, config);
  require(!noPhysicalReceipt.costMs, "NPU public arguments without selected "
                                     "physical AIR ABI cannot be priced");
  build({"cpu", "cpu", "cpu"});
  auto cpuOnly = evaluateProgramAllHeterogeneousCost(candidate, config);
  require(
      cpuOnly.costMs && std::abs(*cpuOnly.costMs - 1e-6) < 1e-12,
      "CPU-only complete region needs no AIR receipt or NPU lifecycle fees");
  // Initialization distribution preserves readonly task-interface operands even
  // when the cloned initializer never loads them. The fused CPU region still
  // needs A/B for its consumer, so the complete region has a safe backing ABI.
  auto initialized = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @initialized(
      %a: memref<4xf32> {orbit.role = "input", orbit.name = "A"},
      %b: memref<4xf32> {orbit.role = "input", orbit.name = "B"},
      %y: memref<4xf32> {orbit.role = "output", orbit.name = "Y"})
      attributes {orbit.noalias} {
    %c = memref.alloc() {orbit.name = "C"} : memref<4xf32>
    %states:3 = taskflow.task @initialize
        will_reads(%a, %b : memref<4xf32>, memref<4xf32>)
        will_writes(%c : memref<4xf32>)
        [original_read_memrefs(%a, %b : memref<4xf32>, memref<4xf32>),
         original_write_memrefs(%c : memref<4xf32>)]
        : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>) {
    ^bb0(%unusedA: memref<4xf32>, %unusedB: memref<4xf32>, %out: memref<4xf32>):
      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to 4 {
        affine.store %zero, %out[%i] : memref<4xf32>
      }
      taskflow.yield done_reads(%unusedA, %unusedB : memref<4xf32>, memref<4xf32>) done_writes(%out : memref<4xf32>)
    }
    %done = taskflow.task @consume
        will_reads(%states#0, %states#1, %states#2 : memref<4xf32>, memref<4xf32>, memref<4xf32>)
        will_writes(%y : memref<4xf32>)
        [original_read_memrefs(%a, %b, %states#2 : memref<4xf32>, memref<4xf32>, memref<4xf32>),
         original_write_memrefs(%y : memref<4xf32>)]
        : (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) -> memref<4xf32> {
    ^bb0(%x: memref<4xf32>, %z: memref<4xf32>, %init: memref<4xf32>, %out: memref<4xf32>):
      affine.for %i = 0 to 4 {
        %av = affine.load %x[%i] : memref<4xf32>
        %bv = affine.load %z[%i] : memref<4xf32>
        %cv = affine.load %init[%i] : memref<4xf32>
        %sum = arith.addf %av, %bv : f32
        %result = arith.addf %sum, %cv : f32
        affine.store %result, %out[%i] : memref<4xf32>
      }
      taskflow.yield done_writes(%out : memref<4xf32>)
    }
    memref.dealloc %c : memref<4xf32>
    return
  }
}
)mlir",
                                                             &context);
  GenericAnalysis initializedSource;
  require(initialized &&
              analyzeGenericProgram(*initialized, initializedSource, error),
          "analyze initializer with unused readonly memory-state interface");
  HeterogeneousMappingChoice initializedChoice;
  initializedChoice.taskDevices = {"cpu", "cpu"};
  HeterogeneousMappingPlan initializedPlan;
  require(
      !buildHeterogeneousMapping(initializedSource, initializedChoice,
                                 initializedPlan, error) &&
          error.find("unused memory-state") != std::string::npos,
      "legacy resident mode still rejects order-only unused task arguments");
  error.clear();
  initializedChoice.implementationOwned = true;
  require(buildHeterogeneousMapping(initializedSource, initializedChoice,
                                    initializedPlan, error),
          "implementation-owned fused region preserves readonly initializer "
          "interface");
  mlir::OwningOpRef<mlir::ModuleOp> initializedChild;
  HeterogeneousRegionProjection initializedProjection;
  GenericAnalysis initializedChildSource;
  require(materializeHeterogeneousRegion(initializedPlan, 0, initializedChild,
                                         initializedProjection, error) &&
              analyzeGenericProgram(*initializedChild, initializedChildSource,
                                    error) &&
              initializedChildSource.tasks.size() == 2 &&
              initializedProjection.arguments.size() == 3,
          "complete CPU region extracts both tasks with A/B/Y backing ABI");
  mlir::OwningOpRef<mlir::ModuleOp> initializedCPU;
  require(lowerHeterogeneousCPUImplementation(*initializedChild, initializedCPU,
                                              error),
          "source-derived CPU lowering accepts safely backed unused readonly "
          "operands");
  initializedChoice.cuts = {1};
  require(buildHeterogeneousMapping(initializedSource, initializedChoice,
                                    initializedPlan, error),
          "explicit CPU cut retains source ordering without invented reads");
  require(materializeHeterogeneousRegion(initializedPlan, 0, initializedChild,
                                         initializedProjection, error),
          "initializer region extracts readonly order-only roots across cut");
  size_t orderOnlyCount = 0;
  for (const auto &binding : initializedProjection.arguments)
    if (binding.orderOnly) {
      ++orderOnlyCount;
      require(binding.role == "input" && binding.inputCopy == -1 &&
                  binding.outputCopy == -1,
              "order-only identity never owns a parent payload copy");
    }
  require(orderOnlyCount == 2,
          "only unused readonly A/B become explicit order-only arguments");
  require(
      analyzeGenericProgram(*initializedChild, initializedChildSource, error),
      "initializer order-only child remains a real analyzable source");
  for (const auto &binding : initializedProjection.arguments)
    if (binding.orderOnly)
      for (const auto &task : initializedChildSource.tasks)
        for (const auto &access : task.accesses)
          require(access.storage != binding.childStorage,
                  "order-only extraction does not invent payload accesses");
  require(materializeHeterogeneousRegion(initializedPlan, 1, initializedChild,
                                         initializedProjection, error) &&
              analyzeGenericProgram(*initializedChild, initializedChildSource,
                                    error),
          "consumer resolves external done_reads to original A/B roots");
  require(std::none_of(initializedProjection.arguments.begin(),
                       initializedProjection.arguments.end(),
                       [](const auto &argument) { return argument.orderOnly; }),
          "actual consumer reads retain normal payload ABI");
  j::Value initializedManifest = nullptr;
  require(exportHeterogeneousProgram(
              *initialized, config, root + "/init-cpu-cut", initializedManifest,
              error, {}, &initializedPlan),
          "cut CPU initializer/consumer export complete source-bound "
          "implementations");
  initializedChoice.taskDevices = {"npu", "cpu"};
  require(buildHeterogeneousMapping(initializedSource, initializedChoice,
                                    initializedPlan, error) &&
              exportHeterogeneousProgram(
                  *initialized, config, root + "/init-npu-cut",
                  initializedManifest, error, emit, &initializedPlan),
          "NPU initializer and CPU consumer export with exact physical "
          "order-only ABI");
  const auto *initializerABI =
      (*initializedManifest.getAsObject()->getArray("regions"))[0]
          .getAsObject()
          ->getArray("npu_abi");
  size_t physicalOrderOnly = 0;
  for (const auto &argument : *initializerABI)
    physicalOrderOnly +=
        argument.getAsObject()->getBoolean("order_only").value_or(false);
  require(
      physicalOrderOnly == 2,
      "actual NPU ABI preserves order-only flags for runtime sync exclusion");
  ProgramAllHeterogeneousCandidate initializedCandidate;
  ProgramDataOrganizationChoice initializedData;
  initializedData.policy = reference.communication.dataPolicy;
  for (size_t task = 0; task < initializedSource.tasks.size(); ++task)
    initializedData.placements.push_back(
        {task, reference.memoryBinding.computeResource,
         reference.memoryBinding.localMemory, "scalar"});
  require(materializeProgramAllHeterogeneousCandidate(
              *initialized, config, initializedChoice,
              reference.memoryBinding.architecture, initializedData, {0, 1},
              false, 128, initializedCandidate, error),
          "combined search materializes the same initializer/cut/order-only "
          "assignment");
  initializedCandidate.evidenceDirectory = root + "/init-priced";
  (*config.getAsObject())["heterogeneous_cost"] =
      modelFor(initializedCandidate, staticFingerprint(contract));
  auto initializedCost =
      evaluateProgramAllHeterogeneousCost(initializedCandidate, config, emit);
  require(initializedCost.costMs.has_value(),
          "typed initializer candidate has a complete disjoint phase price");
  const auto *initializedFeatures = (*initializedCost.result.getAsObject()
                                          ->getObject("features")
                                          ->getArray("regions"))[0]
                                        .getAsObject()
                                        ->getObject("features");
  require(initializedFeatures->getNumber("abi_bo_count") == 3 &&
              initializedFeatures->getNumber("abi_bytes") == 48 &&
              initializedFeatures->getNumber("to_device_bytes") == 16 &&
              initializedFeatures->getNumber("from_device_bytes") == 16 &&
              initializedFeatures->getNumber("dma_bytes") == 16,
          "order-only A/B retain BO identity while only C owns sync/DMA "
          "payload bytes");
  llvm::outs() << "program-all-heterogeneous-cost: " << checks
               << " checks passed (CPU synthetic only)\n";
  return 0;
}
