#include "Taskflow/GenericMapping.h"
#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramAllHeterogeneousCost.h"
#include "Taskflow/ProgramAllSearch.h"
#include "Taskflow/ProgramMatrixImplementation.h"
#ifdef ORBIT_ENABLE_AIR
#include "Taskflow/ProgramGraphAIR.h"
#include "Taskflow/ProgramMatrixRuntime.h"
#include "mlir/IR/Verifier.h"
#endif
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <numeric>
#include <set>

using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
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
// This ABI receipt tests exact selected-plan consumption on the host. It does
// not claim an ELF, hardware execution, or measured/calibrated prices.
bool emit(const GenericAnalysis &source, const GenericMappingPlan &plan,
          const std::string &path, std::string &error) {
  j::Array indices, buffers;
  for (size_t i = 0; i < source.storages.size(); ++i) {
    const auto &storage = source.storages[i];
    j::Array shape;
    for (auto extent : storage.shape)
      shape.push_back(extent);
    indices.push_back(int64_t(i));
    buffers.push_back(j::Object{{"name", storage.id},
                                {"dtype", storage.dtype},
                                {"role", storage.role},
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
                         {"abi_storage_indices", std::move(indices)},
                         {"abi", j::Object{{"buffers", std::move(buffers)}}}}));
  out.close();
  return !out.has_error();
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
  context.allowUnregisteredDialects();
  auto original = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!original)
    return 2;
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
  GenericAnalysis plain;
  require(analyzeGenericProgram(*original, plain, error) &&
              plain.tasks.size() == 3,
          "real source has two contractions and a scalar epilogue");
  const auto plainFingerprint = plain.bodyFingerprint;
  require(setNativeTaskImplementations(*original, 3, {{0, "scalar", 1}}, error),
          "set explicit scalar override");
  GenericAnalysis selected;
  require(analyzeGenericProgram(*original, selected, error) &&
              selected.bodyFingerprint == plainFingerprint,
          "module selectors do not change numeric source fingerprint");
  require(!setNativeTaskImplementations(
              *original, 3, {{0, "scalar", 1}, {0, "scalar", 1}}, error),
          "duplicate local selectors rejected");
  require(
      !setNativeTaskImplementations(*original, 3, {{3, "scalar", 1}}, error),
      "out of range local selector rejected");
  require(!setNativeTaskImplementations(
              *original, 3, {{0, "independent-lane-f32", 3}}, error),
          "non-power-of-two lanes rejected");
  require(setNativeTaskImplementations(
              *original, 3,
              {{0, programMatrixImplementation, 1}, {1, "scalar", 1}}, error),
          "set original-task choices before source initialization transform");
  require(distributeNativeInitialization(*original, error),
          "actual initializer transformation preserves and rebases selectors");
  GenericAnalysis analysis;
  require(analyzeGenericProgram(*original, analysis, error) &&
              analysis.tasks.size() == 5,
          "two generated scalar prologues are separate from original updates");
  auto numericOnly = mlir::OwningOpRef<mlir::ModuleOp>(original->clone());
  numericOnly->getOperation()->removeAttr("orbit.generated_initializer_tasks");
  numericOnly->getOperation()->removeAttr("orbit.task_implementations");
  GenericAnalysis numericAnalysis;
  require(analyzeGenericProgram(*numericOnly, numericAnalysis, error) &&
              numericAnalysis.bodyFingerprint == analysis.bodyFingerprint,
          "module initializer provenance preserves legacy numeric source "
          "fingerprint");
  std::vector<NativeTaskImplementationChoice> choices;
  require(readNativeTaskImplementations(*original, 5, choices, error) &&
              choices.size() == 4 && choices[0].kind == "scalar" &&
              choices[1].task == 1 &&
              choices[1].kind == programMatrixImplementation &&
              choices[2].kind == "scalar" && choices[3].task == 3,
          "local policy follows task operation identity through inserted "
          "initializers");
  j::Value config = j::Object{
      {"generic_air",
       j::Object{
           {"capabilities",
            j::Object{{"implementations",
                       j::Array{j::Object{{"kind", "scalar"}, {"lanes", 1}},
                                j::Object{{"kind", programMatrixImplementation},
                                          {"lanes", 1}}}}}}}}};
  const auto family = heterogeneousCostSourceFamily(*original);
  std::set<std::string> graphIDs;
  for (bool firstMatrix : {false, true})
    for (bool secondMatrix : {false, true}) {
      auto source = original->clone();
      mlir::OwningOpRef<mlir::ModuleOp> owned(source);
      const std::vector<NativeTaskImplementationChoice> localChoices{
          {1, firstMatrix ? programMatrixImplementation : "scalar", 1},
          {3, secondMatrix ? programMatrixImplementation : "scalar", 1}};
      require(setNativeImplementation(source, programMatrixImplementation, 1,
                                      error) &&
                  setNativeTaskImplementations(source, 5, localChoices, error),
              "local choices override module matrix policy independently");
      GenericAnalysis current;
      GenericMappingPlan plan;
      require(
          analyzeGenericProgram(source, current, error) &&
              planGenericMapping(current, config, plan, error),
          "each of the four same-region implementation combinations is legal");
      require(plan.taskImplementations[1].kind ==
                      (firstMatrix ? programMatrixImplementation : "scalar") &&
                  plan.taskImplementations[3].kind ==
                      (secondMatrix ? programMatrixImplementation : "scalar") &&
                  plan.taskImplementations[0].kind == "scalar" &&
                  plan.taskImplementations[2].kind == "scalar" &&
                  plan.taskImplementations[4].kind == "scalar",
              "selected implementations and generated initializer scalar "
              "policy realized");
      require(heterogeneousCostSourceFamily(source) == family,
              "source family excludes implementation choices");
      graphIDs.insert(programGraphIdentity(plan.communication));
#ifdef ORBIT_ENABLE_AIR
      mlir::OwningOpRef<mlir::ModuleOp> air;
      j::Value manifest = nullptr;
      require(
          lowerProgramGraphToAIR(plan.communication, air, manifest, error) &&
              mlir::succeeded(mlir::verify(*air)),
          "actual selected public graph lowers to valid AIR for every local "
          "mode combination");
      auto matrixReceipt = programMatrixAIRManifest(*air);
      auto calls = matrixReceipt.getAsObject()->getInteger("calls").value_or(0);
      require(calls == int64_t(firstMatrix) + int64_t(secondMatrix),
              "actual matrix call count follows the two independent local "
              "implementation selections");
#endif

      auto forged = plan;
      forged.taskImplementations[1].kind =
          firstMatrix ? "scalar" : programMatrixImplementation;
      require(!checkGenericMapping(current, forged, error),
              "independent checker rejects a forged otherwise legal local "
              "implementation");
    }
  require(graphIDs.size() == 4,
          "all four realized public graphs have distinct identities");
  require(setNativeTaskImplementations(
              *original, 5, {{4, programMatrixImplementation, 1}}, error),
          "source-ineligible selector is syntactically valid");
  auto explicitInit = mlir::OwningOpRef<mlir::ModuleOp>(original->clone());
  require(setNativeTaskImplementations(
              *explicitInit, 5, {{0, programMatrixImplementation, 1}}, error),
          "explicit initializer matrix selector passes syntax only");
  GenericAnalysis initAnalysis;
  GenericMappingPlan initPlan;
  require(analyzeGenericProgram(*explicitInit, initAnalysis, error) &&
              !planGenericMapping(initAnalysis, config, initPlan, error),
          "valid provenance cannot erase an explicit unsupported initializer "
          "request");
  GenericMappingPlan rejected;
  require(!planGenericMapping(analysis, config, rejected, error),
          "explicit scalar epilogue matrix request never silently falls back");
  // CPU selection is inactive, while a nonzero-index NPU selector must be
  // rebased to the extracted child task index, not copied as parent index 3.
  require(setNativeTaskImplementations(*original, 5,
                                       {{1, "scalar", 1},
                                        {3, programMatrixImplementation, 1},
                                        {4, programMatrixImplementation, 1}},
                                       error),
          "select CPU inactive epilogue request");
  HeterogeneousMappingChoice placement;
  placement.taskDevices = {"cpu", "cpu", "npu", "npu", "cpu"};
  placement.implementationOwned = true;
  HeterogeneousMappingPlan parent;
  require(buildHeterogeneousMapping(analysis, placement, parent, error),
          "construct actual CPU/NPU/CPU regions");
  mlir::OwningOpRef<mlir::ModuleOp> child;
  HeterogeneousRegionProjection projection;
  require(
      materializeHeterogeneousRegion(parent, 1, child, projection, error) &&
          readNativeTaskImplementations(*child, 2, choices, error) &&
          choices.size() == 1 && choices[0].task == 1 &&
          choices[0].kind == programMatrixImplementation,
      "nonzero parent implementation index correctly projected to NPU child");
  GenericAnalysis childAnalysis;
  require(analyzeGenericProgram(*child, childAnalysis, error) &&
              planGenericMapping(childAnalysis, config, rejected, error),
          "projected strict NPU request has a checked native plan");
  require(
      materializeHeterogeneousRegion(parent, 2, child, projection, error) &&
          readNativeTaskImplementations(*child, 1, choices, error) &&
          choices.empty(),
      "CPU child does not consume an ineligible NPU implementation selector");
  // The second contraction has K=8 while the first has K=16. Swapping modes
  // leaves the kind set unchanged but must change mode-resolved source work.
  llvm::SmallString<128> directory;
  require(!llvm::sys::fs::createUniqueDirectory(
              "orbit-task-implementation-cost", directory),
          "create fresh host evidence directory");
  j::Value contract = j::Object{
      {"runtime_identity", "orbit.serial-prepared-typed-native-wait2.v32"},
      {"provenance",
       "synthetic CPU contract only; no measured hardware price"}};
  const auto contractPath = directory.str().str() + "/contract.json";
  {
    std::error_code ec;
    llvm::raw_fd_ostream output(contractPath, ec);
    require(!ec, "open synthetic execution contract");
    output << llvm::formatv("{0:2}", contract);
  }
  (*config.getAsObject())["execution_contract"] = contractPath;
  std::vector<double> matrixWork, scalarWork;
  for (bool swap : {false, true}) {
    const std::vector<NativeTaskImplementationChoice> mixedChoices{
        {1, swap ? "scalar" : programMatrixImplementation, 1},
        {3, swap ? programMatrixImplementation : "scalar", 1},
        {4, "scalar", 1}};
    require(setNativeTaskImplementations(*original, 5, mixedChoices, error),
            "select mixed task modes for work collection");
    ProgramDataOrganizationChoice data;
    data.policy = rejected.communication.dataPolicy;
    for (size_t task = 0; task < 5; ++task)
      data.placements.push_back({task, rejected.memoryBinding.computeResource,
                                 rejected.memoryBinding.localMemory, "scalar"});
    HeterogeneousMappingChoice allNPU;
    allNPU.taskDevices.assign(5, "npu");
    ProgramAllHeterogeneousCandidate candidate;
    require(materializeProgramAllHeterogeneousCandidate(
                *original, config, allNPU, rejected.memoryBinding.architecture,
                data, {0, 1, 2, 3, 4}, false, 128, candidate, error),
            "materialize selected mixed implementations in the same real NPU "
            "region");
    candidate.evidenceDirectory =
        directory.str().str() + (swap ? "/swap" : "/original");
    require(!llvm::sys::fs::create_directories(candidate.evidenceDirectory),
            "create selected candidate evidence directory");
    auto priced = evaluateProgramAllHeterogeneousCost(candidate, config, emit);
    error = priced.message;
    require(priced.feasible && priced.feasibilityKnown && !priced.costMs,
            "untrained mixed implementation remains feasible with unknown "
            "full-call cost");
    const auto *featureRoot =
        priced.result.getAsObject()->getObject("features");
    const auto *features =
        featureRoot->getArray("regions")->front().getAsObject()->getObject(
            "features");
    auto matrix = features->getNumber(
        "implementation.aie2p-bf16-mmul-f32acc.lanes.1.source_operations");
    auto scalar =
        features->getNumber("implementation.scalar.lanes.1.source_operations");
    require(matrix && scalar,
            "actual work is separated by selected implementation");
    matrixWork.push_back(*matrix);
    scalarWork.push_back(*scalar);
    auto pricedConfig = j::Value(config);
    (*pricedConfig.getAsObject())["heterogeneous_cost"] =
        modelFor(candidate, staticFingerprint(contract));
    candidate.evidenceDirectory += "/aggregate-only";
    auto aggregate =
        evaluateProgramAllHeterogeneousCost(candidate, pricedConfig, emit);
    error = aggregate.message;
    require(aggregate.feasible && !aggregate.costMs,
            "an otherwise valid aggregate-only row cannot price new explicit "
            "mixed work");
    auto *model = pricedConfig.getAsObject()->getObject("heterogeneous_cost");
    auto *phases =
        model->getArray("regions")->front().getAsObject()->getObject("phases");
    auto *wait = phases->getObject("wait");
    for (const auto &feature : *features)
      if (feature.first.str().find("implementation.") == 0 &&
          feature.first.str().find(".source_operations") != std::string::npos &&
          feature.second.getAsNumber().value_or(0) > 0) {
        auto term = phase(0, feature.first);
        (*wait->getObject("coefficients_ns"))[feature.first] =
            j::Value(*term.getAsObject()
                          ->getObject("coefficients_ns")
                          ->get(feature.first));
        (*wait->getObject("ranges"))[feature.first] = j::Array{0, 1000000000};
      }
    candidate.evidenceDirectory += "/implementation-resolved";
    auto resolved =
        evaluateProgramAllHeterogeneousCost(candidate, pricedConfig, emit);
    error = resolved.message;
    require(resolved.feasible && bool(resolved.costMs),
            "explicit modeled per-implementation work enables finite synthetic "
            "pricing");
  }
  require(matrixWork[0] == 16384 && matrixWork[1] == 8192 &&
              scalarWork[0] == 8448 && scalarWork[1] == 16640,
          "independent arithmetic oracle: 4 operations per contraction "
          "iteration plus 256 relu ops");
  auto fission = original->clone();
  mlir::OwningOpRef<mlir::ModuleOp> fissionOwned(fission);
  const std::vector<NativeTaskImplementationChoice> scalarChoices{
      {1, "scalar", 1}, {3, "scalar", 1}};
  require(setNativeTaskImplementations(fission, 5, scalarChoices, error) &&
              splitNativeOuterLoops(fission, error),
          "range fission preserves local scalar choices");
  GenericAnalysis split;
  require(analyzeGenericProgram(fission, split, error) &&
              planGenericMapping(split, config, rejected, error),
          "fissioned scalar realization is rechecked on actual changed loop "
          "bodies");
  auto forgedInitializer = original->clone();
  mlir::OwningOpRef<mlir::ModuleOp> forgedOwned(forgedInitializer);
  GenericAnalysis forgedAnalysis;
  require(analyzeGenericProgram(forgedInitializer, forgedAnalysis, error),
          "analyze initializer-forgery case");
  forgedInitializer->setAttr("orbit.generated_initializer_tasks",
                             mlir::DenseI64ArrayAttr::get(&context, {4}));
  require(
      setNativeTaskImplementations(
          forgedInitializer, 5, {{4, programMatrixImplementation, 1}}, error),
      "select unsupported epilogue mode under forged initializer provenance");
  require(analyzeGenericProgram(forgedInitializer, forgedAnalysis, error) &&
              !planGenericMapping(forgedAnalysis, config, rejected, error),
          "initializer marker alone cannot suppress a strict implementation "
          "request");

  auto invalidMetadata = mlir::OwningOpRef<mlir::ModuleOp>(original->clone());
  GenericAnalysis metadataAnalysis;
  require(analyzeGenericProgram(*invalidMetadata, metadataAnalysis, error),
          "analyze initializer schema negative cases");
  std::vector<size_t> roles;
  invalidMetadata->getOperation()->setAttr(
      "orbit.generated_initializer_tasks",
      mlir::StringAttr::get(&context, "0"));
  require(!readNativeInitializerTasks(metadataAnalysis, roles, error),
          "initializer provenance rejects wrong table type");
  invalidMetadata->getOperation()->setAttr(
      "orbit.generated_initializer_tasks",
      mlir::DenseI64ArrayAttr::get(&context, {5}));
  require(!readNativeInitializerTasks(metadataAnalysis, roles, error),
          "initializer provenance rejects out-of-range task");
  invalidMetadata->getOperation()->setAttr(
      "orbit.generated_initializer_tasks",
      mlir::DenseI64ArrayAttr::get(&context, {0, 0}));
  require(!readNativeInitializerTasks(metadataAnalysis, roles, error),
          "initializer provenance rejects duplicate task indices");

  // Exercise the actual public C++ search adapter, including original-task
  // lineage through initialization. The oracle here is an explicitly assumed
  // nonadditive contract, never a hardware performance claim.
  auto input = llvm::MemoryBuffer::getFile(argv[1]);
  require(bool(input), "read immutable search source");
  SearchOptions options;
  options.backend = "air";
  options.outputDir = directory.str().str() + "/discovery";
  auto searchConfig = j::Value(config);
  (*searchConfig.getAsObject())["program_search"] =
      j::Object{{"heterogeneous", true}, {"max_evaluations", 1},
                {"seconds", 30},         {"max_values_per_dimension", 3},
                {"max_routes", 1},       {"max_priorities", 1}};
  auto oracle = [&](const ProgramAllHeterogeneousCandidate &candidate) {
    ProgramSearchEvaluation evaluation;
    evaluation.feasible = evaluation.feasibilityKnown = true;
    evaluation.costMs = 10;
    j::Array children;
    for (const auto &region : candidate.regions)
      children.push_back(programGraphIdentity(region.selectedGraph));
    evaluation.result = j::Object{
        {"candidate_id", candidate.assignmentId},
        {"parent_graph_id", programGraphIdentity(candidate.parent.graph)},
        {"region_graph_ids", std::move(children)}};
    return evaluation;
  };
  require(runProgramAllSearch((*input)->getBuffer(), options, searchConfig,
                              error, oracle),
          "discover real public native task-implementation dimensions");
  auto readJSON = [&](const std::string &path) {
    auto bytes = llvm::MemoryBuffer::getFile(path);
    return llvm::cantFail(j::parse((*bytes)->getBuffer()));
  };
  auto space = readJSON(options.outputDir + "/space.json");
  j::Array frozenDimensions;
  for (const auto &entry : *space.getAsObject()->getArray("dimensions")) {
    const auto *dimension = entry.getAsObject();
    const auto id = dimension->getString("id")->str();
    j::Value selected =
        *dimension->getArray("choices")->front().getAsObject()->get("binding");
    if (id == "source.initialization")
      selected = j::Object{{"distribute", true}};
    if (id.rfind("device.task.", 0) == 0)
      selected =
          j::Object{{"task", int64_t(id.back() - '0')}, {"device", "npu"}};
    j::Array values;
    if (id == "task.0.implementation" || id == "task.1.implementation") {
      const auto task = int64_t(id[5] - '0');
      values.push_back(j::Object{
          {"task", task}, {"implementation", "scalar"}, {"lanes", 1}});
      values.push_back(
          j::Object{{"task", task},
                    {"implementation", programMatrixImplementation},
                    {"lanes", 1}});
    } else
      values.push_back(std::move(selected));
    frozenDimensions.push_back(
        j::Object{{"id", id}, {"choices", std::move(values)}});
  }
  auto *search = searchConfig.getAsObject()->getObject("program_search");
  (*search)["dimensions"] = std::move(frozenDimensions);
  (*search)["max_evaluations"] = 8;
  options.outputDir = directory.str().str() + "/finite";
  std::set<std::pair<bool, bool>> visited;
  auto selectedOracle = [&](const ProgramAllHeterogeneousCandidate &candidate) {
    const auto &plan = candidate.regions.front().native;
    const bool first =
        plan.taskImplementations[1].kind == programMatrixImplementation;
    const bool second =
        plan.taskImplementations[3].kind == programMatrixImplementation;
    visited.insert({first, second});
    auto result = oracle(candidate);
    result.costMs = first && second ? 1 : 10;
    return result;
  };
  require(runProgramAllSearch((*input)->getBuffer(), options, searchConfig,
                              error, selectedOracle),
          "search jointly materializes both local implementation bindings");
  auto report = readJSON(options.outputDir + "/search.json");
  const bool unavailable =
      report.getAsObject()->getString("status") == "UNAVAILABLE";
  require(visited.size() == (unavailable ? 1 : 4),
          "ON exhausts four joint choices; OFF only evaluates baseline");
  require(report.getAsObject()->getBoolean("finite_domain_optimality_proved") ==
              !unavailable,
          "finite-domain proof requires actual solver enumeration");
  llvm::outs() << "task-implementations: " << checks
               << " checks passed (CPU only)\n";
}
