#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/GenericProgramTiming.h"
#include "Taskflow/HeterogeneousSearch.h"
#include "Taskflow/JSONUtils.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramAllHeterogeneousCost.h"
#include "Taskflow/ProgramAllSearch.h"
#include "Taskflow/ProgramHeterogeneousSearch.h"
#include "Taskflow/ProgramInvocationEvaluation.h"
#include "Taskflow/ProgramJointMILP.h"
#include "Taskflow/ProgramJointSearch.h"
#include "Taskflow/ProgramMatrixImplementation.h"
#include "Taskflow/ProgramPartition.h"
#include "Taskflow/ProgramRegionCompositeCost.h"
#include "Taskflow/Search.h"
#ifdef ORBIT_ENABLE_AIR
#include "Taskflow/AIRBackend.h"
#include "Taskflow/NPUIndependentShards.h"
#include "Taskflow/ProgramGraphAIE.h"
#include "Taskflow/ProgramGraphAIR.h"
#include "Taskflow/ProgramGraphAIRSelection.h"
#include "Taskflow/ProgramJointAIR.h"
#include "Taskflow/ProgramMatrixRuntime.h"
#endif
#include "Taskflow/StaticMapping.h"
#include "Taskflow/StructuredProgram.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#ifdef ORBIT_ENABLE_AIR
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#endif
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cmath>
using namespace orbit::taskflow;
namespace cl = llvm::cl;
static cl::opt<std::string>
    input(cl::Positional,
          cl::desc("source plan.json or authenticated Taskflow .mlir"),
          cl::Required);
static cl::opt<std::string>
    planFile("emit-plan", cl::desc("Write versioned execution plan"));
static cl::opt<std::string>
    mlirFile("emit-mlir", cl::desc("Write materialized canonical Taskflow"));
static cl::opt<int64_t> tm("tile-m", cl::init(0)), tn("tile-n", cl::init(0)),
    tk("tile-k", cl::init(0));
static cl::opt<bool> fuse("fuse", cl::init(false)),
    fission("fission", cl::init(false));
static cl::opt<bool>
    verifyOnly("verify-only", cl::init(false),
               cl::desc("Verify MLIR, including fail-closed legacy verifier, "
                        "without exchange export"));
static cl::opt<bool> search("search", cl::init(false)),
    searchResume("resume", cl::init(false));
static cl::opt<std::string> backend("backend", cl::init("npu")),
    evaluator("evaluator"), targetConfig("target-config"),
    searchOutput("search-output"), searchCheckpoint("checkpoint");
static cl::opt<int64_t> searchRounds("max-rounds", cl::init(2)),
    searchBudget("max-candidates", cl::init(9)),
    searchWidth("beam-width", cl::init(1));
static cl::opt<bool>
    staticMap("static-map", cl::init(false),
              cl::desc("Native C++ static mapping from actual Taskflow MLIR"));
static cl::opt<std::string> mappingOutput("mapping-output"),
    checkMapping("check-mapping"), analysisOutput("analysis-output");
static cl::opt<std::string> publicEvaluation(
    "public-program-evaluation",
    cl::desc("Replay a frozen complete public candidate, without search")),
    publicSpace("public-program-space"),
    publicAssignment("public-program-assignment"),
    publicLower("lower-public-program-to-air"),
    publicCheckAIE("check-public-program-aie"),
    publicRuntimeModel(
        "public-program-runtime-model",
        cl::desc(
            "Evaluate a new invocation model after exact frozen graph replay"));
static cl::opt<std::string> heterogeneousProgramSearch(
    "heterogeneous-program-search",
    cl::desc("Joint finite CPU/NPU placement and invocation-boundary search in "
             "a NEW directory")),
    heterogeneousProgramAssignment(
        "heterogeneous-program-assignment",
        cl::desc("Evaluate one assignment from the declared heterogeneous "
                 "search domain without solving"));
static cl::opt<bool> genericAnalyze("generic-analyze"),
    genericMap("generic-map"), lowerAIR("lower-to-air"),
    nativeInterface("native-interface"), airPassPipeline("air-pass-pipeline"),
    airApproximateExp("air-approximate-exp");
static cl::opt<std::string> programRuntimeContract(
    "realize-program-runtime",
    cl::desc(
        "Realize and check final generic NPU AIE using the sealed selected "
        "transfer contract; write --air-output and --implementation-output"));
static cl::opt<std::string> airOutput("air-output"), checkAIR("check-air"),
    checkAIE("check-aie"), implementationOutput("implementation-output"),
    implementationFactsOutput("implementation-facts-output");
static cl::opt<std::string>
    programTimingOutput("program-timing-output",
                        cl::desc("Export checked native features, composite "
                                 "timeline and unidentified internal events"));
static cl::opt<std::string> nativeCostFeaturesOutput(
    "native-cost-features-output",
    cl::desc("Export checked native cost quantities without constructing an "
             "event timing schedule"));
static cl::opt<bool> programGraph(
    "program-graph",
    cl::desc("Build and check the public task/data/communication graph"));
static cl::opt<std::string> programGraphOutput("program-graph-output"),
    programGraphDOT("program-graph-dot"),
    checkProgramGraphFile("check-program-graph");
static cl::opt<std::string> legalizeAIRChannels(
    "legalize-air-channels",
    cl::desc("Apply checked plan channel resources to converted generic AIR")),
    checkAIRChannels(
        "check-air-channels",
        cl::desc(
            "Check converted AIR channel resources against the native plan"));
static cl::opt<bool> airDistributeInitialization(
    "air-distribute-initialization",
    cl::desc(
        "Distribute provably independent initialization before AIR tiling"));
static cl::opt<std::string> residency(
    "residency",
    cl::desc(
        "Legacy target residency spelling: phase-local or readonly-panel")),
    dataOrganization(
        "data-organization",
        cl::desc(
            "Public data organization: phase-local or retain-across-users")),
    implementation(
        "implementation",
        cl::desc(
            "Explicit target implementation: scalar, independent-lane-f32, or "
            "scalar-product-vector-add-f32 (installed AIR: 8 lanes), or "
            "aie2p-bf16-mmul-f32acc (explicit BF16 source contract)"));
static cl::opt<int64_t> residentStorage("resident-storage", cl::init(-1)),
    reuseLoop("reuse-loop", cl::init(-1)),
    panelLoop("panel-loop", cl::init(-1)),
    vectorLanes("vector-lanes", cl::init(1));
static cl::opt<int64_t> programPartition(
    "program-partition", cl::init(0),
    cl::desc("Independent whole-program source-axis instances per command"));
static cl::opt<std::string>
    lowerProgram("lower-program-to-air",
                 cl::desc("Write checked partition templates and invocation "
                          "manifest to a NEW directory"));
static cl::opt<std::string>
    heterogeneousOutput("heterogeneous-output",
                        cl::desc("Export checked CPU/NPU regions and native "
                                 "implementations to a NEW directory"));
static cl::opt<std::string> checkHeterogeneous(
    "check-heterogeneous",
    cl::desc("Compare a static manifest with source/target reconstruction"));
static cl::opt<std::string>
    lowerMulticore("lower-multicore-to-air",
                   cl::desc("Write a CP-SAT-placed independent-shard AIR "
                            "program to a NEW directory"));
static cl::opt<std::string> checkMulticoreAIE(
    "check-multicore-aie",
    cl::desc(
        "Independently check final multi-core AIE for the source and target"));
static cl::opt<int64_t> npuShards("npu-shards", cl::init(2));
static cl::opt<std::string>
    jointSearch("joint-program-search",
                cl::desc("Joint public ProgramGraph optimization; write "
                         "evidence to a new directory"));
static cl::opt<std::string>
    jointSolver("joint-solver", cl::init("cp-sat"),
                cl::desc("Direct joint makespan solver: cp-sat or scip"));
static cl::opt<std::string>
    jointPlan("joint-program-plan",
              cl::desc("Recheck source-bound joint decisions from this JSON "
                       "file without solving"));
static cl::opt<std::string> checkJointAIE("check-joint-aie");
static cl::opt<bool> jointBaselineOnly("joint-baseline-only");
static cl::opt<bool> jointLower(
    "joint-lower-air",
    cl::desc(
        "Lower the chosen public joint graph to AIR in the search directory"));
static cl::opt<double> jointSeconds("joint-seconds", cl::init(5));
static cl::opt<int64_t> jointVariables("joint-max-variables", cl::init(100000));
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
                              "ORBIT independent structured Taskflow core\n");
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
#ifdef ORBIT_ENABLE_AIR
  registry.insert<mlir::vector::VectorDialect>();
  context.appendDialectRegistry(registry);
  if (!programRuntimeContract.empty() || lowerAIR || !publicLower.empty() ||
      !publicCheckAIE.empty() || jointLower || !checkJointAIE.empty() ||
      !lowerMulticore.empty() || !checkMulticoreAIE.empty() ||
      !lowerProgram.empty() || !heterogeneousOutput.empty() ||
      airPassPipeline || (!checkAIR.empty() || !checkAIE.empty()) ||
      !legalizeAIRChannels.empty() || !checkAIRChannels.empty()) {
    context.allowUnregisteredDialects();
    prepareGenericAIRContext(context);
  }
#endif
  auto bytes = llvm::MemoryBuffer::getFile(input);
  if (!bytes) {
    llvm::errs() << bytes.getError().message() << "\n";
    return 1;
  }
  Program p;
  std::string error;
  llvm::StringRef text = (*bytes)->getBuffer();
  auto fatal = [&](llvm::StringRef e) {
    llvm::errs() << "taskflow-opt: " << e << "\n";
    return 1;
  };
  auto parse = [&](llvm::StringRef s) {
    return mlir::parseSourceString<mlir::ModuleOp>(s, &context);
  };
  auto applyMappingOptions = [&](mlir::ModuleOp module) {
    if (programPartition) {
      if (programPartition < 1 || programPartition > INT32_MAX) {
        error = "program partition must be in [1,INT32_MAX]";
        return false;
      }
      module->setAttr("orbit.program_partition_extent",
                      mlir::Builder(module.getContext())
                          .getI64IntegerAttr(programPartition));
    }
    if (!residency.empty() || !dataOrganization.empty()) {
      ProgramMappingChoice choice;
      choice.dataOrganization = dataOrganization;
      choice.residency = residency;
      choice.storage = residentStorage;
      choice.reuseLoop = reuseLoop;
      choice.panelLoop = panelLoop;
      if (!setNativeMappingChoice(module, choice, error))
        return false;
    } else if (residentStorage != -1 || reuseLoop != -1 || panelLoop != -1) {
      error = "resident storage and loop choices require --data-organization "
              "or --residency";
      return false;
    }
    if (!implementation.empty()) {
      if (!setNativeImplementation(module, implementation, vectorLanes, error))
        return false;
    } else if (vectorLanes != 1) {
      error = "--vector-lanes requires --implementation";
      return false;
    }
    return normalizeNativeMappingChoice(module, error);
  };
  auto readJSON =
      [&](const std::string &path) -> llvm::Expected<llvm::json::Value> {
    auto b = llvm::MemoryBuffer::getFile(path);
    if (!b)
      return llvm::errorCodeToError(b.getError());
    return llvm::json::parse((*b)->getBuffer());
  };
  auto writeJSON = [&](const std::string &path,
                       const llvm::json::Value &value) {
    if (path.empty()) {
      llvm::outs() << llvm::formatv("{0:2}\n", value);
      return true;
    }
    std::error_code ec;
    llvm::raw_fd_ostream out(path, ec);
    if (ec) {
      error = ec.message();
      return false;
    }
    out << llvm::formatv("{0:2}\n", value);
    out.close();
    if (!out.has_error())
      return true;
    error = path + ": " + out.error().message();
    out.clear_error();
    return false;
  };
  if (!programRuntimeContract.empty()) {
#ifdef ORBIT_ENABLE_AIR
    if (airOutput.empty() || implementationOutput.empty())
      return fatal("runtime realization requires IR and proof output paths");
    auto contract = readJSON(programRuntimeContract);
    if (!contract)
      return fatal(llvm::toString(contract.takeError()));
    // AIE dialect properties are preserved by generic parsing; installed
    // aie-opt performs the registered dialect verification in the next stage.
    mlir::ParserConfig parserConfig(&context, /*verifyAfterParse=*/false);
    auto actual = mlir::parseSourceString<mlir::ModuleOp>(text, parserConfig);
    mlir::OwningOpRef<mlir::ModuleOp> realized;
    llvm::json::Value proof = nullptr;
    if (!actual || !realizeProgramMatrixRuntime(*contract, *actual, realized,
                                                proof, error))
      return fatal(error.empty() ? "invalid generic NPU AIE" : error);
    std::error_code ec;
    llvm::raw_fd_ostream out(airOutput, ec);
    if (ec)
      return fatal(ec.message());
    realized->print(out, mlir::OpPrintingFlags().printGenericOpForm());
    out << "\n";
    out.close();
    if (out.has_error()) {
      error = out.error().message();
      out.clear_error();
      return fatal(error);
    }
    return writeJSON(implementationOutput, proof) ? 0 : fatal(error);
#else
    return fatal("runtime realization requires ORBIT_ENABLE_AIR=ON");
#endif
  }
  HeterogeneousAIREmitter emitSelectedRegion;
#ifdef ORBIT_ENABLE_AIR
  emitSelectedRegion = [&](const GenericAnalysis &,
                           const GenericMappingPlan &plan,
                           const std::string &path, std::string &reason) {
    mlir::OwningOpRef<mlir::ModuleOp> air;
    llvm::json::Value physical = nullptr;
    if (!lowerProgramGraphToAIR(plan.communication, air, physical, reason))
      return false;
    auto writeText = [&](const std::string &name, auto printer) {
      std::error_code ec;
      llvm::raw_fd_ostream out(name, ec);
      if (ec) {
        reason = name + ": " + ec.message();
        return false;
      }
      printer(out);
      out.close();
      if (!out.has_error())
        return true;
      reason = name + ": " + out.error().message();
      out.clear_error();
      return false;
    };
    std::vector<int64_t> abiStorageIndices;
    for (const auto &entry :
         *physical.getAsObject()->getArray("abi_storage_indices"))
      abiStorageIndices.push_back(*entry.getAsInteger());
    auto runtimeContract = serializeProgramMatrixRuntimeContract(
        plan.communication, abiStorageIndices, reason);
    if (runtimeContract.getAsNull())
      return false;
    (*physical.getAsObject())["runtime_contract_sha256"] =
        staticFingerprint(runtimeContract);
    if (!writeJSON(path + ".runtime-contract.json", runtimeContract))
      return false;
    if (!writeText(path,
                   [&](auto &out) {
                     air->print(out,
                                mlir::OpPrintingFlags().printGenericOpForm());
                     out << "\n";
                   }) ||
        !writeJSON(path + ".physical.json", physical))
      return false;
    auto matrix = physical.getAsObject()->getObject("matrix_realization");
    if (matrix)
      if (auto kernels = matrix->getArray("kernels"))
        for (const auto &value : *kernels) {
          const auto &kernel = *value.getAsObject();
          const auto source = programMatrixKernelSource(
              *kernel.getInteger("m"), *kernel.getInteger("k"),
              *kernel.getInteger("n"));
          const auto filename = llvm::sys::path::parent_path(path).str() + "/" +
                                kernel.getString("symbol")->str() + ".cc";
          if (!writeText(filename, [&](auto &out) { out << source; }))
            return false;
        }
    // Bound each hardware configuration's static DMA descriptors. Cuts
    // preserve all local-copy lifetimes; the union proof owns full coverage.
    std::vector<ProgramGraphAIRSelection> selections;
    llvm::json::Value coverage = nullptr;
    if (!buildProgramGraphAIRSelections(plan.communication, 8, selections,
                                        coverage, reason))
      return false;
    (*physical.getAsObject())["physical_invocation_count"] =
        int64_t(selections.size());
    if (selections.size() > 1) {
      llvm::json::Array configurations;
      for (size_t i = 0; i < selections.size(); ++i) {
        mlir::OwningOpRef<mlir::ModuleOp> selected;
        llvm::json::Value receipt = nullptr;
        if (!lowerProgramGraphAIRSelection(plan.communication, selections[i],
                                           selected, receipt, reason))
          return false;
        auto contract = serializeProgramMatrixRuntimeContract(
            plan.communication, abiStorageIndices, selections[i], reason);
        if (contract.getAsNull())
          return false;
        const auto name =
            path + ".configuration-" + std::to_string(i) + ".mlir";
        (*receipt.getAsObject())["runtime_contract_sha256"] =
            staticFingerprint(contract);
        if (!writeText(name,
                       [&](auto &out) {
                         selected->print(
                             out, mlir::OpPrintingFlags().printGenericOpForm());
                         out << "\n";
                       }) ||
            !writeJSON(name + ".physical.json", receipt) ||
            !writeJSON(name + ".runtime-contract.json", contract))
          return false;
        configurations.push_back(llvm::json::Object{
            {"air", llvm::sys::path::filename(name).str()},
            {"begin_event", int64_t(selections[i].begin)},
            {"end_event_exclusive", int64_t(selections[i].end)},
            {"runtime_contract_sha256", staticFingerprint(contract)}});
      }
      (*physical.getAsObject())["configurations"] = std::move(configurations);
      (*physical.getAsObject())["configuration_coverage"] = std::move(coverage);
    }
    if (!writeJSON(path + ".physical.json", physical))
      return false;
    return true;
  };
#endif
  auto configureAllHeterogeneous = [&](SearchOptions &options) {
    if (targetConfig.empty())
      return true;
    auto configuration = readJSON(targetConfig);
    if (!configuration) {
      error = llvm::toString(configuration.takeError());
      return false;
    }
    const auto root = configuration->getAsObject();
    const auto controls = root ? root->getObject("program_search") : nullptr;
    if (!controls || !controls->getBoolean("heterogeneous").value_or(false))
      return true;
    if (!emitSelectedRegion) {
      error = "combined CPU/NPU search requires AIR support";
      return false;
    }
    options.heterogeneousEvaluator =
        [configuration = std::move(*configuration),
         emitter = emitSelectedRegion](
            const ProgramAllHeterogeneousCandidate &candidate) {
          return evaluateProgramAllHeterogeneousCost(candidate, configuration,
                                                     emitter);
        };
    return true;
  };
  auto emitGraph = [&](const ProgramGraph &graph) {
    if (!checkProgramGraph(graph, error))
      return false;
    auto view = serializeProgramGraph(graph);
    if (!checkProgramGraphFile.empty()) {
      auto expected = readJSON(checkProgramGraphFile);
      if (!expected) {
        error = llvm::toString(expected.takeError());
        return false;
      }
      if (staticFingerprint(*expected) != staticFingerprint(view)) {
        error =
            "program graph view differs from checked source/target/candidate";
        return false;
      }
    }
    if (!programGraphOutput.empty() && !writeJSON(programGraphOutput, view))
      return false;
    if (!programGraphDOT.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream output(programGraphDOT, ec);
      if (ec) {
        error = ec.message();
        return false;
      }
      output << renderProgramGraphDOT(graph);
    }
    return true;
  };
  if (!heterogeneousProgramSearch.empty() ||
      !heterogeneousProgramAssignment.empty()) {
    if (targetConfig.empty())
      return fatal("heterogeneous public search requires target configuration");
    if (!heterogeneousProgramSearch.empty() &&
        !heterogeneousProgramAssignment.empty())
      return fatal("choose search or single-assignment evaluation");
    auto config = readJSON(targetConfig);
    if (!config)
      return fatal(llvm::toString(config.takeError()));
    auto root = config->getAsObject();
    auto modelJSON = root ? root->get("program_invocation_model") : nullptr;
    if (!modelJSON)
      return fatal("program_invocation_model required");
    for (auto &entry : *root)
      if (entry.first != "program_invocation_model" &&
          entry.first != "heterogeneous_search" &&
          entry.first != "region_implementation_catalog")
        return fatal("unconsumed heterogeneous configuration field: " +
                     entry.first.str());
    bool composite =
        modelJSON->getAsObject() &&
        modelJSON->getAsObject()->getString("kind") == "region-composite";
    ProgramInvocationModel model;
    ProgramRegionCompositeModel regionModel;
    if (composite) {
      if (!parseProgramRegionCompositeModel(*modelJSON, regionModel, error))
        return fatal(error);
    } else if (!parseProgramInvocationModel(*modelJSON, model, error))
      return fatal(error);
    std::map<std::string, std::vector<ProgramRegionImplementation>>
        implementations;
    if (auto catalog = root->get("region_implementation_catalog")) {
      if (!composite)
        return fatal(
            "implementation catalogue requires region-composite model");
      auto entries = catalog->getAsArray();
      if (!entries)
        return fatal("implementation catalogue must be an array");
      for (auto &entry : *entries) {
        auto o = entry.getAsObject();
        auto graph = o ? o->getString("graph_id") : std::nullopt;
        auto rows = o ? o->getArray("implementations") : nullptr;
        if (!o || o->size() != 2 || !graph || graph->empty() || !rows ||
            implementations.count(graph->str()))
          return fatal(
              "unique graph-bound compiled implementation catalogue required");
        std::vector<ProgramRegionImplementation> values;
        for (auto &row : *rows) {
          auto r = row.getAsObject();
          auto index = r ? r->getInteger("region") : std::nullopt;
          if (!r || r->size() != 8 || !index || *index < 0)
            return fatal("invalid compiled region witness");
          ProgramRegionImplementation value;
          value.region = *index;
          const std::pair<const char *, std::string *> fields[] = {
              {"implementation", &value.implementation},
              {"source_fingerprint", &value.sourceFingerprint},
              {"code_identity", &value.codeIdentity},
              {"control_sequence_identity", &value.controlSequenceIdentity},
              {"control_family", &value.controlFamily},
              {"provenance", &value.provenance}};
          for (const auto &p : fields) {
            auto s = r->getString(p.first);
            if (!s || s->empty())
              return fatal("compiled witness strings must be explicit");
            *p.second = s->str();
          }
          // The eighth field binds the implementation producer, rather than a
          // price. Actual artifact files are verified by that producer/runner.
          if (!r->getString("artifact_receipt") ||
              r->getString("artifact_receipt")->empty())
            return fatal("compiled witness artifact receipt required");
          values.push_back(std::move(value));
        }
        implementations.emplace(graph->str(), std::move(values));
      }
    }
    auto module = parse(text);
    GenericAnalysis source;
    if (!module || mlir::failed(mlir::verify(*module)) ||
        !analyzeGenericProgram(*module, source, error))
      return fatal(error.empty() ? "valid Taskflow source required" : error);
    ProgramHeterogeneousSearchOptions options;
    auto controls = root->getObject("heterogeneous_search");
    if (root->get("heterogeneous_search") && !controls)
      return fatal("heterogeneous_search must be an object");
    if (controls)
      for (auto &entry : *controls) {
        auto key = entry.first;
        if (key == "seconds") {
          auto n = entry.second.getAsNumber();
          if (!n)
            return fatal("seconds must be numeric");
          options.search.wallSeconds = *n;
        } else if (key == "task_device_choices") {
          auto rows = entry.second.getAsArray();
          if (!rows)
            return fatal("task_device_choices must be an array");
          for (auto &row : *rows) {
            auto a = row.getAsArray();
            if (!a)
              return fatal("device choices need rows");
            std::vector<std::string> values;
            for (auto &v : *a) {
              auto s = v.getAsString();
              if (!s)
                return fatal("device choice must be a string");
              values.push_back(s->str());
            }
            options.taskDeviceChoices.push_back(std::move(values));
          }
        } else if (key == "boundary_choices") {
          auto rows = entry.second.getAsArray();
          if (!rows)
            return fatal("boundary_choices must be an array");
          for (auto &row : *rows) {
            auto a = row.getAsArray();
            if (!a)
              return fatal("boundary choices need rows");
            std::vector<bool> values;
            for (auto &v : *a) {
              auto b = v.getAsBoolean();
              if (!b)
                return fatal("boundary choice must be boolean");
              values.push_back(*b);
            }
            options.boundaryChoices.push_back(std::move(values));
          }
        } else {
          auto n = entry.second.getAsInteger();
          if (!n)
            return fatal("search integer control required");
          if (key == "max_evaluations")
            options.search.maxEvaluations = *n;
          else if (key == "max_variables")
            options.search.maxVariables = *n;
          else if (key == "max_constraints")
            options.search.maxConstraints = *n;
          else if (key == "max_tuple_cells")
            options.search.maxTupleCells = *n;
          else if (key == "workers") {
            if (*n < 1 || *n > 16)
              return fatal("workers out of range");
            options.search.workers = *n;
          } else if (key == "seed") {
            if (*n < 0 || *n > INT32_MAX)
              return fatal("seed out of range");
            options.search.seed = *n;
          } else
            return fatal("unconsumed heterogeneous search control: " +
                         key.str());
        }
      }
    ProgramSearchSpace space;
    if (!buildProgramHeterogeneousSearchSpace(source, options, space, error))
      return fatal(error);
    llvm::json::Array evaluations;
    auto evaluate = [&](const HeterogeneousMappingPlan &plan) {
      ProgramSearchEvaluation evaluated;
      ProgramMechanismEvaluation cost;
      llvm::json::Value receipt = nullptr;
      std::string why;
      if (composite) {
        auto found = implementations.find(programGraphIdentity(plan.graph));
        if (found == implementations.end()) {
          evaluated.feasibilityKnown = false;
          evaluated.message =
              "compiled region implementation evidence unavailable";
          return evaluated;
        }
        if (!evaluateProgramRegionCompositeModel(
                plan, regionModel, found->second, cost.timing, receipt, why)) {
          evaluated.feasibilityKnown = false;
          evaluated.message = why;
          return evaluated;
        }
        cost.mode = regionModel.mode;
        cost.status = cost.timing.makespanMs
                          ? "exclusive-region-composite-estimate"
                          : "unknown-required-regional-model";
      } else if (!evaluateProgramInvocationModel(plan.graph, model, cost,
                                                 receipt, why)) {
        evaluated.feasibilityKnown = false;
        evaluated.message = why;
        return evaluated;
      }
      evaluated.feasible = true;
      evaluated.costMs = cost.timing.makespanMs;
      evaluated.message =
          evaluated.costMs
              ? "checked region graph; complete declared invocation/body "
                "estimate"
              : "checked region graph; required mechanism remains unknown";
      evaluated.result = llvm::json::Object{
          {"estimated_ms", evaluated.costMs
                               ? llvm::json::Value(*evaluated.costMs)
                               : llvm::json::Value(nullptr)},
          {"mechanism", serializeProgramMechanismCost(cost)},
          {"invocation", std::move(receipt)},
          {"physical_realization_checked", false}};
      evaluations.push_back(llvm::json::Object{
          {"plan", serializeHeterogeneousMapping(plan)},
          {"evaluation", llvm::json::Value(evaluated.result)}});
      return evaluated;
    };
    if (!heterogeneousProgramAssignment.empty()) {
      auto value = readJSON(heterogeneousProgramAssignment);
      if (!value)
        return fatal(llvm::toString(value.takeError()));
      auto a = value->getAsArray();
      if (!a)
        return fatal("assignment must be an integer array");
      std::vector<int64_t> assignment;
      for (auto &v : *a) {
        auto n = v.getAsInteger();
        if (!n)
          return fatal("assignment must contain integers");
        assignment.push_back(*n);
      }
      auto result = evaluateProgramHeterogeneousAssignment(
          source, space, assignment, evaluate);
      llvm::json::Value out = llvm::json::Object{
          {"feasible", result.feasible},
          {"feasibility_known", result.feasibilityKnown},
          {"estimated_ms", result.costMs ? llvm::json::Value(*result.costMs)
                                         : llvm::json::Value(nullptr)},
          {"message", result.message},
          {"evaluation", std::move(result.result)}};
      if (!writeJSON(implementationOutput, out))
        return fatal(error);
      return 0;
    }
    if (llvm::sys::fs::exists(heterogeneousProgramSearch))
      return fatal("heterogeneous search output must be a new directory");
    if (auto ec = llvm::sys::fs::create_directories(heterogeneousProgramSearch))
      return fatal(ec.message());
    auto result = searchProgramHeterogeneous(source, evaluate, options);
    if (!writeJSON(heterogeneousProgramSearch + "/search.json",
                   llvm::json::Value(std::move(result.report))) ||
        !writeJSON(heterogeneousProgramSearch + "/evaluations.json",
                   llvm::json::Value(std::move(evaluations))) ||
        !writeJSON(heterogeneousProgramSearch + "/config.json", *config))
      return fatal(error);
    if (result.hasBest) {
      llvm::json::Array assignment;
      for (auto n : result.bestAssignment)
        assignment.push_back(n);
      if (!writeJSON(heterogeneousProgramSearch + "/best-assignment.json",
                     llvm::json::Value(std::move(assignment))) ||
          !writeJSON(heterogeneousProgramSearch + "/best.json",
                     result.bestEvaluation.result))
        return fatal(error);
    }
    return 0;
  }
  if (!publicEvaluation.empty()) {
    if (publicSpace.empty() || publicAssignment.empty() || targetConfig.empty())
      return fatal(
          "public replay requires space, assignment and target configuration");
    auto module = parse(text);
    if (!module || mlir::failed(mlir::verify(*module)))
      return fatal("valid materialized candidate source required");
    auto config = readJSON(targetConfig), space = readJSON(publicSpace),
         assignment = readJSON(publicAssignment),
         expected = readJSON(publicEvaluation);
    if (!config)
      return fatal(llvm::toString(config.takeError()));
    if (!space)
      return fatal(llvm::toString(space.takeError()));
    if (!assignment)
      return fatal(llvm::toString(assignment.takeError()));
    if (!expected)
      return fatal(llvm::toString(expected.takeError()));
    ProgramGraph graph;
    llvm::json::Value receipt = nullptr;
    if (!replayPublicProgramGraph(*module, *config, *space, *assignment,
                                  *expected, graph, receipt, error))
      return fatal(error);
    if (!emitGraph(graph))
      return fatal(error);
    if (!publicRuntimeModel.empty()) {
      if (!publicLower.empty() || !publicCheckAIE.empty())
        return fatal(
            "runtime re-evaluation is separate from lowering/checking");
      auto value = readJSON(publicRuntimeModel);
      if (!value)
        return fatal(llvm::toString(value.takeError()));
      ProgramInvocationModel model;
      ProgramMechanismEvaluation cost;
      llvm::json::Value invocation = nullptr;
      if (!parseProgramInvocationModel(*value, model, error) ||
          !evaluateProgramInvocationModel(graph, model, cost, invocation,
                                          error))
        return fatal(error);
      llvm::json::Value out = llvm::json::Object{
          {"replay", std::move(receipt)},
          {"mechanism", serializeProgramMechanismCost(cost)},
          {"invocation", std::move(invocation)},
          {"estimated_ms", cost.timing.makespanMs
                               ? llvm::json::Value(*cost.timing.makespanMs)
                               : llvm::json::Value(nullptr)},
          {"physical_time_optimality_proved", false}};
      if (!writeJSON(implementationOutput, out))
        return fatal(error);
      return 0;
    }
    if (!publicCheckAIE.empty()) {
#ifdef ORBIT_ENABLE_AIR
      auto actualBytes = llvm::MemoryBuffer::getFile(publicCheckAIE);
      if (!actualBytes)
        return fatal(actualBytes.getError().message());
      auto actual = parse((*actualBytes)->getBuffer());
      if (!actual)
        return fatal("invalid final generic AIE");
      mlir::OwningOpRef<mlir::ModuleOp> expectedAIR;
      llvm::json::Value manifest = nullptr;
      if (!lowerProgramGraphToAIR(graph, expectedAIR, manifest, error))
        return fatal(error);
      llvm::json::Value checked = nullptr;
      bool valid =
          checkProgramGraphAIE(graph, *expectedAIR, *actual, checked, error);
      if (auto o = checked.getAsObject()) {
        (*o)["valid"] = valid;
        (*o)["diagnostic"] = error;
      }
      if (!writeJSON(implementationOutput, checked))
        return fatal(error);
      return valid ? 0 : fatal(error);
#else
      return fatal("public AIE checking requires ORBIT_ENABLE_AIR=ON");
#endif
    }
    if (!publicLower.empty()) {
#ifdef ORBIT_ENABLE_AIR
      if (llvm::sys::fs::exists(publicLower))
        return fatal("public lowering output must be a new directory");
      if (auto ec = llvm::sys::fs::create_directories(publicLower))
        return fatal(ec.message());
      mlir::OwningOpRef<mlir::ModuleOp> air;
      llvm::json::Value manifest = nullptr;
      if (!lowerProgramGraphToAIR(graph, air, manifest, error))
        return fatal(error);
      std::error_code ec;
      llvm::raw_fd_ostream output(publicLower + "/program.air.mlir", ec);
      if (ec)
        return fatal(ec.message());
      air->print(output, mlir::OpPrintingFlags().printGenericOpForm());
      output << "\n";
      output.close();
      auto object = manifest.getAsObject();
      if (!object || !object->get("abi"))
        return fatal("public lowering omitted ABI manifest");
      if (!writeJSON(publicLower + "/manifest.json", manifest) ||
          !writeJSON(publicLower + "/abi.json", *object->get("abi")) ||
          !writeJSON(publicLower + "/replay.json", receipt))
        return fatal(error);
#else
      return fatal("public graph AIR replay requires ORBIT_ENABLE_AIR=ON");
#endif
    }
    if (!writeJSON(implementationOutput, receipt))
      return fatal(error);
    return 0;
  }
  const bool wantsGraph = !programTimingOutput.empty() || programGraph ||
                          !programGraphOutput.empty() ||
                          !programGraphDOT.empty() ||
                          !checkProgramGraphFile.empty();
  if (search && backend == "air" && !targetConfig.empty()) {
    auto cfg = readJSON(targetConfig);
    if (!cfg)
      return fatal(llvm::toString(cfg.takeError()));
    auto root = cfg->getAsObject();
    auto options = root ? root->getObject("native_auto_candidates") : nullptr;
    if (options && options->getString("algorithm") == "cp-sat-program") {
      if (options->getString("mode") != "joint")
        return fatal(
            "cp-sat-program requires native_auto_candidates.mode=joint");
      if (searchResume || !searchCheckpoint.empty())
        return fatal("public joint search currently writes a fresh finite "
                     "solve; staged-search resume is unsupported");
      for (auto &entry : *options)
        if (entry.first != "mode" && entry.first != "algorithm")
          return fatal("cp-sat-program domain and budgets use program_joint "
                       "and --joint-seconds, not legacy proposal fields");
      if (searchOutput.empty())
        return fatal("--search-output required for public joint search");
      jointSearch = searchOutput.getValue();
    }
  }
  if (!jointSearch.empty() || !jointPlan.empty() || !checkJointAIE.empty()) {
    const auto jointStarted = std::chrono::steady_clock::now();
    auto pipelineMilliseconds = [&]() {
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - jointStarted)
          .count();
    };
    if (jointSolver != "cp-sat" && jointSolver != "scip")
      return fatal("--joint-solver must be cp-sat or scip");
    if (!std::isfinite(jointSeconds) || jointSeconds <= 0)
      return fatal("--joint-seconds must be finite and positive");
    auto module = parse(text);
    if (!module || mlir::failed(mlir::verify(*module)))
      return fatal("valid native Taskflow source required");
    llvm::json::Value config = llvm::json::Object{};
    if (!targetConfig.empty()) {
      auto c = readJSON(targetConfig);
      if (!c)
        return fatal(llvm::toString(c.takeError()));
      config = std::move(*c);
    }
#ifdef ORBIT_ENABLE_AIR
    if (!configureProgramJointNPU(config, npuShards, error))
      return fatal(error);
#endif
    GenericAnalysis analysis;
    ProgramJointDomain domain;
    if (!analyzeGenericProgram(*module, analysis, error) ||
        !buildProgramJointDomain(analysis, config, domain, error))
      return fatal(error);
    auto baseline = baselineProgramJointDomain(domain);
    ProgramGraph baselineGraph;
    if (baseline.hasPlan &&
        (!materializeProgramJointSolution(domain, baseline, baselineGraph,
                                          error) ||
         !checkProgramJointSolution(domain, baseline, baselineGraph, error)))
      return fatal(error);
    const double preprocessingMs = pipelineMilliseconds();
    bool solverInvoked = false;
    auto serializeJointRecord = [&](const ProgramJointSolution &selected,
                                    const ProgramGraph *selectedGraph,
                                    llvm::StringRef solverName) {
      auto value =
          serializeProgramJointSolution(domain, selected, selectedGraph);
      auto &record = *value.getAsObject();
      record["solver"] = solverName.str();
      record["solver_invoked"] = solverInvoked;
      record["preprocessing_ms"] = preprocessingMs;
      record["pipeline_total_ms"] = pipelineMilliseconds();
      record["pipeline_total_scope"] = "elapsed through solution construction; "
                                       "final evidence writes excluded";
      record["pipeline_budget_seconds"] = double(jointSeconds);
      record["pipeline_budget_exhausted"] =
          pipelineMilliseconds() >= jointSeconds * 1000.;
      return value;
    };
    if (!jointSearch.empty()) {
      if (llvm::sys::fs::exists(jointSearch))
        return fatal("joint output directory must be new");
      if (auto ec = llvm::sys::fs::create_directories(jointSearch))
        return fatal(ec.message());
      if (!writeJSON(jointSearch + "/domain.json",
                     serializeProgramJointDomain(domain)) ||
          !writeJSON(jointSearch + "/baseline.json",
                     serializeJointRecord(
                         baseline, baseline.hasPlan ? &baselineGraph : nullptr,
                         "baseline")) ||
          !writeJSON(jointSearch + "/target.json", config))
        return fatal(error);
#ifdef ORBIT_ENABLE_AIR
      if (jointLower && baseline.hasPlan) {
        mlir::OwningOpRef<mlir::ModuleOp> air;
        llvm::json::Value manifest = nullptr;
        if (!lowerProgramJointToAIR(*module, config, domain, baseline,
                                    baselineGraph, air, manifest, error))
          return fatal("baseline cannot lower: " + error);
        std::error_code ec;
        llvm::raw_fd_ostream stream(jointSearch + "/baseline.air.mlir", ec);
        if (ec)
          return fatal(ec.message());
        air->print(stream, mlir::OpPrintingFlags().printGenericOpForm());
        stream << "\n";
        if (!writeJSON(jointSearch + "/baseline-air.json", manifest))
          return fatal(error);
      }
#endif
    }
    ProgramJointSolution solution;
    if (!jointPlan.empty()) {
      auto record = readJSON(jointPlan);
      if (!record)
        return fatal(llvm::toString(record.takeError()));
      auto object = record->getAsObject();
      if (!object || object->getString("domain_id") != domain.identity)
        return fatal("joint plan source/target/domain identity mismatch");
      if (!parseProgramJointSolution(*record, solution, error))
        return fatal(error);
    } else if (jointBaselineOnly)
      solution = baseline;
    else {
      ConstraintSearchOptions options;
      options.wallSeconds =
          std::max(1e-9, jointSeconds - pipelineMilliseconds() / 1000.);
      options.maxVariables = jointVariables;
      solverInvoked = true;
      solution = jointSolver == "scip"
                     ? solveProgramJointDomainMILP(domain, options, &baseline)
                     : solveProgramJointDomain(domain, options, &baseline);
    }
    ProgramGraph graph;
    if (solution.hasPlan &&
        (!materializeProgramJointSolution(domain, solution, graph, error) ||
         !checkProgramJointSolution(domain, solution, graph, error)))
      return fatal(error);
    // Own the conditional's string result. A StringRef would borrow the
    // temporary created by mixing string literals with jointSolver's string.
    const std::string solverName = !jointPlan.empty()  ? "replay"
                                   : jointBaselineOnly ? "baseline"
                                                       : jointSolver.getValue();
    if (!jointSearch.empty() &&
        !writeJSON(jointSearch + "/solution.json",
                   serializeJointRecord(solution,
                                        solution.hasPlan ? &graph : nullptr,
                                        solverName)))
      return fatal(error);
    if (!jointPlan.empty() && solution.hasPlan) {
      auto record = readJSON(jointPlan);
      auto stored =
          record ? record->getAsObject()->get("program_graph") : nullptr;
      if (!stored || staticFingerprint(*stored) !=
                         staticFingerprint(serializeProgramGraph(graph)))
        return fatal(
            "stored joint public graph differs from checked decisions");
    }
    if (!checkJointAIE.empty()) {
#ifdef ORBIT_ENABLE_AIR
      if (jointPlan.empty())
        return fatal("--check-joint-aie requires --joint-program-plan to check "
                     "frozen choices without solving");
      auto candidateBytes = llvm::MemoryBuffer::getFile(checkJointAIE);
      if (!candidateBytes)
        return fatal(candidateBytes.getError().message());
      auto actual = parse((*candidateBytes)->getBuffer());
      if (!actual)
        return fatal("valid generic final AIE required");
      llvm::json::Value receipt = nullptr;
      bool valid = checkProgramJointAIE(*module, config, domain, solution,
                                        graph, *actual, receipt, error);
      if (!writeJSON(implementationOutput, receipt))
        return fatal(error);
      return valid ? 0 : fatal(error);
#else
      return fatal("joint AIE checker requires ORBIT_ENABLE_AIR=ON");
#endif
    }
    if (jointLower) {
#ifdef ORBIT_ENABLE_AIR
      if (jointSearch.empty() || !solution.hasPlan)
        return fatal(
            "--joint-lower-air requires a search directory and checked plan");
      mlir::OwningOpRef<mlir::ModuleOp> air;
      llvm::json::Value manifest = nullptr;
      if (!lowerProgramJointToAIR(*module, config, domain, solution, graph, air,
                                  manifest, error))
        return fatal(error);
      std::error_code ec;
      llvm::raw_fd_ostream stream(jointSearch + "/program.air.mlir", ec);
      if (ec)
        return fatal(ec.message());
      air->print(stream, mlir::OpPrintingFlags().printGenericOpForm());
      stream << "\n";
      if (!writeJSON(jointSearch + "/manifest.json", manifest) ||
          !writeJSON(jointSearch + "/abi.json",
                     *manifest.getAsObject()->get("abi")) ||
          !writeJSON(jointSearch + "/target.json", config))
        return fatal(error);
#else
      return fatal("joint AIR lowering requires ORBIT_ENABLE_AIR=ON");
#endif
    }
    if (solution.hasPlan && !emitGraph(graph))
      return fatal(error);
    if (!mappingOutput.empty() &&
        !writeJSON(mappingOutput,
                   serializeJointRecord(solution,
                                        solution.hasPlan ? &graph : nullptr,
                                        solverName)))
      return fatal(error);
    if (!jointSearch.empty()) {
      // A separate receipt observes graph/lowering/evidence work after the
      // solution snapshot. External process wall time also includes this final
      // receipt write and process teardown; neither is solver search time.
      const double elapsedMs = pipelineMilliseconds();
      if (!writeJSON(
              jointSearch + "/pipeline.json",
              llvm::json::Object{
                  {"schema", "orbit.program-joint.pipeline.v31"},
                  {"solver", solverName},
                  {"preprocessing_ms", preprocessingMs},
                  {"backend_total_ms", solution.telemetry.totalMs},
                  {"model_build_ms", solution.telemetry.modelBuildMs},
                  {"solve_ms", solution.telemetry.solveMs},
                  {"elapsed_before_receipt_ms", elapsedMs},
                  {"budget_seconds", double(jointSeconds)},
                  {"budget_exhausted", elapsedMs >= jointSeconds * 1000.},
                  {"final_receipt_write_included", false}}))
        return fatal(error);
    }
    return solution.hasPlan ? 0 : fatal(solution.message);
  }
  if (!lowerMulticore.empty() || !checkMulticoreAIE.empty()) {
#ifdef ORBIT_ENABLE_AIR
    auto module = parse(text);
    if (!module || mlir::failed(mlir::verify(*module)))
      return fatal("valid Taskflow source required");
    llvm::json::Value config = llvm::json::Object{};
    if (!targetConfig.empty()) {
      auto c = readJSON(targetConfig);
      if (!c)
        return fatal(llvm::toString(c.takeError()));
      config = std::move(*c);
    }
    if (!checkMulticoreAIE.empty()) {
      auto candidateBytes = llvm::MemoryBuffer::getFile(checkMulticoreAIE);
      if (!candidateBytes)
        return fatal(candidateBytes.getError().message());
      auto actual = parse((*candidateBytes)->getBuffer());
      if (!actual)
        return fatal("valid final AIE module required");
      llvm::json::Value receipt = nullptr;
      bool valid = checkNPUIndependentAIE(*module, config, npuShards, *actual,
                                          receipt, error);
      if (!implementationOutput.empty() &&
          !writeJSON(implementationOutput, receipt))
        return fatal(error);
      return valid ? 0 : fatal(error);
    }
    if (llvm::sys::fs::exists(lowerMulticore))
      return fatal("multicore output directory must be new");
    mlir::OwningOpRef<mlir::ModuleOp> air;
    llvm::json::Value manifest = nullptr;
    if (!lowerNPUIndependentProgram(*module, config, npuShards, air, manifest,
                                    error))
      return fatal(error);
    if (auto ec = llvm::sys::fs::create_directories(lowerMulticore))
      return fatal(ec.message());
    auto writeModule = [&](const std::string &path, mlir::ModuleOp value,
                           bool generic) {
      std::error_code ec;
      llvm::raw_fd_ostream out(path, ec);
      if (ec) {
        error = ec.message();
        return false;
      }
      auto flags = mlir::OpPrintingFlags();
      if (generic)
        flags.printGenericOpForm();
      value.print(out, flags);
      out << "\n";
      return true;
    };
    if (!writeModule(lowerMulticore + "/source.mlir", *module, false) ||
        !writeModule(lowerMulticore + "/program.air.mlir", *air, true) ||
        !writeJSON(lowerMulticore + "/manifest.json", manifest) ||
        !writeJSON(lowerMulticore + "/target.json", config))
      return fatal(error);
    GenericAnalysis analysis;
    if (!analyzeGenericProgram(*module, analysis, error))
      return fatal(error);
    llvm::json::Array buffers, outputs;
    for (const auto &storage : analysis.storages) {
      if (!mlir::isa<mlir::BlockArgument>(storage.root))
        continue;
      llvm::json::Array shape;
      for (auto n : storage.shape)
        shape.push_back(n);
      buffers.push_back(llvm::json::Object{{"name", storage.id},
                                           {"role", storage.role},
                                           {"dtype", storage.dtype},
                                           {"shape", std::move(shape)}});
      if (storage.role == "output")
        outputs.push_back(storage.id);
    }
    if (!writeJSON(lowerMulticore + "/abi.json",
                   llvm::json::Object{{"buffers", std::move(buffers)},
                                      {"outputs", std::move(outputs)}}))
      return fatal(error);
    return 0;
#else
    return fatal("multicore lowering requires ORBIT_ENABLE_AIR=ON");
#endif
  }
  if (search && backend == "cpu-npu") {
    auto module = parse(text);
    if (!module)
      return fatal("native Taskflow source required");
    SearchOptions options{backend,      evaluator,        searchOutput,
                          targetConfig, searchCheckpoint, searchRounds,
                          searchBudget, searchWidth,      searchResume};
    if (!configureHeterogeneousSearch(*module, options, error))
      return fatal(error);
    return runStructuredSearch(Program{}, options, error) ? 0 : fatal(error);
  }
  if (!heterogeneousOutput.empty()) {
    if (llvm::sys::fs::exists(heterogeneousOutput))
      return fatal("heterogeneous output directory must be new");
    auto module = parse(text);
    if (!module || mlir::failed(mlir::verify(*module)))
      return fatal("valid Taskflow source required");
    llvm::json::Value config = llvm::json::Object{};
    if (!targetConfig.empty()) {
      auto c = readJSON(targetConfig);
      if (!c)
        return fatal(llvm::toString(c.takeError()));
      config = std::move(*c);
    }
    HeterogeneousMappingChoice originalChoice;
    GenericAnalysis originalSource;
    if (!readHeterogeneousChoice(*module, config, originalChoice, error) ||
        !analyzeGenericProgram(*module, originalSource, error))
      return fatal(error);
    if (airDistributeInitialization) {
      mlir::Builder builder(&context);
      for (size_t task = 0; task < originalSource.tasks.size(); ++task)
        originalSource.tasks[task].operation->setAttr(
            "orbit.search_origin", builder.getI64IntegerAttr(task));
      if (!distributeNativeInitialization(*module, error))
        return fatal(error);
      GenericAnalysis transformed;
      if (!analyzeGenericProgram(*module, transformed, error))
        return fatal(error);
      llvm::SmallVector<mlir::Attribute> devices;
      llvm::SmallVector<int64_t> cuts;
      int64_t previous = -1;
      for (size_t task = 0; task < transformed.tasks.size(); ++task) {
        auto origin =
            transformed.tasks[task].operation->getAttrOfType<mlir::IntegerAttr>(
                "orbit.search_origin");
        if (!origin || origin.getInt() < 0 ||
            size_t(origin.getInt()) >= originalChoice.taskDevices.size())
          return fatal("initialization transform lost device-choice lineage");
        devices.push_back(
            builder.getStringAttr(originalChoice.taskDevices[origin.getInt()]));
        if (task && previous != origin.getInt() &&
            llvm::is_contained(originalChoice.cuts, size_t(origin.getInt())))
          cuts.push_back(task);
        previous = origin.getInt();
      }
      module->getOperation()->setAttr("orbit.heterogeneous_devices",
                                      builder.getArrayAttr(devices));
      module->getOperation()->setAttr("orbit.heterogeneous_cuts",
                                      builder.getDenseI64ArrayAttr(cuts));
    }
    if (tm || tn || tk) {
      if (!setNativeAIRTileSizes(*module, tm, tn, tk, error))
        return fatal(error);
      // Explicit CLI tiles are the selected source decision. Remove inherited
      // target defaults before child planning, which otherwise overrides the
      // module attribute and silently exports a different physical program.
      config =
          copyJSONWithoutNestedFields(config, "generic_air", {"tile_sizes"});
    }
    if (!applyMappingOptions(*module))
      return fatal(error);
    HeterogeneousAIREmitter emitter;
#ifdef ORBIT_ENABLE_AIR
    emitter = [&](const GenericAnalysis &source, const GenericMappingPlan &plan,
                  const std::string &path, std::string &error) {
      mlir::OwningOpRef<mlir::ModuleOp> air;
      auto root = config.getAsObject();
      auto heterogeneous = root ? root->getObject("heterogeneous") : nullptr;
      if (heterogeneous &&
          heterogeneous->getBoolean("implementation_owned").value_or(false)) {
        return emitSelectedRegion(source, plan, path, error);
      } else if (!lowerGenericToAIR(source, plan, air, error))
        return false;
      std::error_code ec;
      llvm::raw_fd_ostream out(path, ec);
      if (ec) {
        error = ec.message();
        return false;
      }
      air->print(out, mlir::OpPrintingFlags().printGenericOpForm());
      out << "\n";
      out.close();
      if (!out.has_error())
        return true;
      error = path + ": " + out.error().message();
      out.clear_error();
      return false;
    };
#endif
    llvm::json::Value manifest = nullptr;
    if (!exportHeterogeneousProgram(*module, config, heterogeneousOutput,
                                    manifest, error, emitter))
      return fatal(error);
    if (!checkHeterogeneous.empty()) {
      auto expected = readJSON(checkHeterogeneous);
      if (!expected)
        return fatal(llvm::toString(expected.takeError()));
      if (staticFingerprint(*expected) != staticFingerprint(manifest))
        return fatal(
            "heterogeneous manifest differs from source/target reconstruction");
    }
    return 0;
  }
  if (airPassPipeline) {
    if (airApproximateExp || !targetConfig.empty())
      return fatal(
          "--air-pass-pipeline uses the default exact resident plan; use "
          "--lower-to-air for explicit target/approximation options");
#ifdef ORBIT_ENABLE_AIR
    auto module = parse(text);
    if (!module)
      return fatal("invalid input MLIR");
    mlir::PassManager pipeline(&context);
    pipeline.addPass(createTaskflowToAIRPass());
    if (mlir::failed(pipeline.run(*module)))
      return fatal("taskflow-to-air pipeline failed");
    if (airOutput.empty())
      return fatal("--air-output required");
    std::error_code ec;
    llvm::raw_fd_ostream output(airOutput, ec);
    if (ec)
      return fatal(ec.message());
    module->print(output, mlir::OpPrintingFlags().printGenericOpForm());
    output << "\n";
    return 0;
#else
    return fatal("configure ORBIT_ENABLE_AIR=ON for direct AIR pass");
#endif
  }
  if (search && backend == "air" && text.ltrim().starts_with("module")) {
    SearchOptions options{backend,      evaluator,        searchOutput,
                          targetConfig, searchCheckpoint, searchRounds,
                          searchBudget, searchWidth,      searchResume};
    options.nativeSourceMLIR = text.str();
    if (!configureAllHeterogeneous(options))
      return fatal(error);
    if (airDistributeInitialization || tm || tn || tk || !residency.empty() ||
        !dataOrganization.empty() || !implementation.empty() ||
        residentStorage != -1 || reuseLoop != -1 || panelLoop != -1 ||
        vectorLanes != 1 || programPartition) {
      auto module = parse(text);
      if (!module)
        return fatal("invalid input MLIR");
      if (airDistributeInitialization &&
          !distributeNativeInitialization(*module, error))
        return fatal(error);
      if ((tm || tn || tk) &&
          !setNativeAIRTileSizes(*module, tm, tn, tk, error))
        return fatal(error);
      if (!applyMappingOptions(*module))
        return fatal(error);
      options.nativeSourceMLIR.clear();
      llvm::raw_string_ostream os(options.nativeSourceMLIR);
      module->print(os);
    }
    return runStructuredSearch(Program{}, options, error) ? 0 : fatal(error);
  }
  if (wantsGraph || !nativeCostFeaturesOutput.empty() || genericAnalyze ||
      genericMap || lowerAIR || !lowerProgram.empty() ||
      (!checkAIR.empty() || !checkAIE.empty()) ||
      !legalizeAIRChannels.empty() || !checkAIRChannels.empty()) {
    auto module = parse(text);
    if (!module || mlir::failed(mlir::verify(*module)))
      return fatal("valid native Taskflow MLIR required");
    if (airDistributeInitialization &&
        !distributeNativeInitialization(*module, error))
      return fatal(error);
    if ((tm || tn || tk) && !setNativeAIRTileSizes(*module, tm, tn, tk, error))
      return fatal(error);
    if (!applyMappingOptions(*module))
      return fatal(error);
    if (!mlirFile.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream output(mlirFile, ec);
      if (ec)
        return fatal(ec.message());
      module->print(output);
      output << "\n";
    }
    GenericAnalysis analysis;
    if (!analyzeGenericProgram(*module, analysis, error))
      return fatal(error);
    if (!writeJSON(analysisOutput, serializeGenericAnalysis(analysis)))
      return fatal(error);
    if (wantsGraph || !nativeCostFeaturesOutput.empty() || genericMap ||
        lowerAIR || !lowerProgram.empty() ||
        (!checkAIR.empty() || !checkAIE.empty()) ||
        !legalizeAIRChannels.empty() || !checkAIRChannels.empty()) {
      llvm::json::Value config = llvm::json::Object{};
      if (!targetConfig.empty()) {
        auto value = readJSON(targetConfig);
        if (!value)
          return fatal(llvm::toString(value.takeError()));
        config = std::move(*value);
      }
      const auto *configuration = config.getAsObject();
      const auto *graphConfiguration =
          configuration ? configuration->getObject("program_graph") : nullptr;
      if (graphConfiguration &&
          graphConfiguration->getString("adapter") == "whole-task") {
        if (!nativeCostFeaturesOutput.empty())
          return fatal(
              "native cost features require the checked native adapter");
        if (!programTimingOutput.empty())
          return fatal(
              "--program-timing-output requires the native NPU adapter; use "
              "evaluateProgramTiming for whole-task backend annotations");
        if (lowerAIR || !lowerProgram.empty() ||
            (!checkAIR.empty() || !checkAIE.empty()) ||
            !legalizeAIRChannels.empty() || !checkAIRChannels.empty())
          return fatal(
              "whole-task public graph is a CPU planning adapter; direct AIR "
              "requires the checked native target realization");
        StaticConfig target;
        ProgramDataOrganizationChoice choice;
        ProgramGraph graph;
        if (!parseWholeTaskProgramGraphConfig(config, target, choice, error) ||
            !planProgramDataOrganization(analysis, target, choice, graph,
                                         error) ||
            !emitGraph(graph) ||
            !writeJSON(mappingOutput, serializeProgramGraph(graph)))
          return fatal(error);
        return 0;
      }
      if (graphConfiguration &&
          graphConfiguration->getString("adapter") != "native")
        return fatal("unsupported program graph adapter");
      if (module->getOperation()->hasAttr("orbit.program_partition_extent")) {
        if (!nativeCostFeaturesOutput.empty())
          return fatal("export native features separately for each checked "
                       "partition template");
        if (lowerAIR || (!checkAIR.empty() || !checkAIE.empty()) ||
            !legalizeAIRChannels.empty() || !checkAIRChannels.empty())
          return fatal("partitioned program requires --lower-program-to-air; a "
                       "single AIR command would ignore its schedule");
        auto chunk = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
            "orbit.program_partition_extent");
        ProgramPartitionPlan partition;
        if (!chunk ||
            !planProgramPartition(analysis, chunk.getInt(), partition, error))
          return fatal(error);
        auto evaluated = evaluatePartitionedGenericMLIR(*module, config, error);
        auto result = evaluated.getAsObject();
        auto best = result ? result->get("best_plan") : nullptr;
        if (!best || result->getString("status") == "unsupported")
          return fatal(error.empty()
                           ? "unsupported partition target implementation"
                           : error);
        if (!writeJSON(mappingOutput, *best))
          return fatal(error);
        if (lowerProgram.empty())
          return 0;
#ifdef ORBIT_ENABLE_AIR
        if (llvm::sys::fs::exists(lowerProgram))
          return fatal("use a new program output directory");
        if (auto ec = llvm::sys::fs::create_directories(lowerProgram))
          return fatal(ec.message());
        auto writeModule = [&](const std::string &path, mlir::ModuleOp value,
                               bool generic) {
          std::error_code ec;
          llvm::raw_fd_ostream out(path, ec);
          if (ec) {
            error = ec.message();
            return false;
          }
          auto flags = mlir::OpPrintingFlags();
          if (generic)
            flags.printGenericOpForm();
          value.print(out, flags);
          out << "\n";
          return true;
        };
        if (!writeModule(lowerProgram + "/taskflow.mlir", *module, false) ||
            !writeJSON(lowerProgram + "/config.json", config))
          return fatal(error);
        auto manifest = llvm::json::Value(*best);
        auto &manifestObject = *manifest.getAsObject();
        for (auto &value : *manifestObject.getArray("templates")) {
          auto &record = *value.getAsObject();
          auto extent = *record.getInteger("extent");
          auto prefix = lowerProgram + "/template-" + std::to_string(extent);
          if (auto ec = llvm::sys::fs::create_directories(prefix))
            return fatal(ec.message());
          mlir::OwningOpRef<mlir::ModuleOp> child, air;
          GenericAnalysis facts;
          GenericMappingPlan childPlan;
          if (!materializeProgramPartition(*module, partition, extent, child,
                                           error) ||
              !analyzeGenericProgram(*child, facts, error) ||
              !planGenericMapping(facts, config, childPlan, error) ||
              !lowerGenericToAIR(facts, childPlan, air, error,
                                 airApproximateExp))
            return fatal(error);
          if (llvm::any_of(childPlan.hardwareStages, [](const auto &stage) {
                return stage.legalizeChannels;
              }))
            return fatal(
                "partition template currently requires canonical direct-DMA "
                "stages; channel legalization must be explicit");
          if (!writeModule(prefix + "/taskflow.mlir", *child, false) ||
              !writeModule(prefix + "/program.air.mlir", *air, true) ||
              !writeJSON(prefix + "/facts.json",
                         serializeGenericAnalysis(facts)) ||
              !writeJSON(prefix + "/plan.json",
                         serializeGenericMapping(facts, childPlan)))
            return fatal(error);
          record["mapping_plan_id"] = childPlan.planId;
          llvm::json::Array buffers, outputs;
          for (bool temporary : {false, true})
            for (const auto &storage : facts.storages)
              if ((storage.role == "temporary") == temporary) {
                llvm::json::Array shape;
                for (auto size : storage.shape)
                  shape.push_back(size);
                buffers.push_back(
                    llvm::json::Object{{"name", storage.id},
                                       {"role", storage.role},
                                       {"dtype", storage.dtype},
                                       {"shape", std::move(shape)}});
                if (storage.role == "output")
                  outputs.push_back(storage.id);
              }
          if (!writeJSON(prefix + "/abi.json",
                         llvm::json::Object{{"buffers", std::move(buffers)},
                                            {"outputs", std::move(outputs)}}))
            return fatal(error);
        }
        llvm::json::Array invocation;
        for (int i = 0; i < argc; ++i)
          invocation.push_back(argv[i]);
        manifestObject["compiler_invocation"] = std::move(invocation);
        manifestObject["approximate_exp"] = bool(airApproximateExp);
        if (!writeJSON(lowerProgram + "/manifest.json", manifest))
          return fatal(error);
        return 0;
#else
        return fatal("partition AIR emission requires ORBIT_ENABLE_AIR=ON");
#endif
      }
      if (!lowerProgram.empty())
        return fatal(
            "--lower-program-to-air requires a program partition choice");
      GenericMappingPlan plan;
      if (!planGenericMapping(analysis, config, plan, error))
        return fatal(error);
      if (!nativeCostFeaturesOutput.empty()) {
        GenericCostFeatures features;
        if (!collectGenericCostFeatures(analysis, plan, features, error) ||
            !writeJSON(nativeCostFeaturesOutput,
                       serializeGenericCostFeatures(features)))
          return fatal(error);
      }
      if (wantsGraph && !emitGraph(plan.communication))
        return fatal(error);
      if (!programTimingOutput.empty()) {
        llvm::json::Value timing = nullptr;
        if (!evaluateGenericProgramTiming(analysis, plan, timing, error) ||
            !writeJSON(programTimingOutput, timing))
          return fatal(error);
      }
      if (!mappingOutput.empty() &&
          !writeJSON(mappingOutput, serializeGenericMapping(analysis, plan)))
        return fatal(error);
      if (!legalizeAIRChannels.empty() || !checkAIRChannels.empty()) {
#ifdef ORBIT_ENABLE_AIR
        if (!legalizeAIRChannels.empty() && !checkAIRChannels.empty())
          return fatal(
              "select either --legalize-air-channels or --check-air-channels");
        const std::string path = legalizeAIRChannels.empty()
                                     ? checkAIRChannels.getValue()
                                     : legalizeAIRChannels.getValue();
        auto candidateBytes = llvm::MemoryBuffer::getFile(path);
        if (!candidateBytes)
          return fatal(candidateBytes.getError().message());
        auto candidate = parse((*candidateBytes)->getBuffer());
        if (!candidate || mlir::failed(mlir::verify(*candidate)))
          return fatal("valid converted generic AIR required");
        if (!legalizeAIRChannels.empty() &&
            !legalizeGenericAIRChannels(analysis, plan, *candidate, error))
          return fatal(error);
        if (!checkGenericAIRChannels(analysis, plan, *candidate, error) ||
            mlir::failed(mlir::verify(*candidate)))
          return fatal(error);
        if (!legalizeAIRChannels.empty()) {
          if (airOutput.empty())
            return fatal("--air-output required");
          std::error_code ec;
          llvm::raw_fd_ostream output(airOutput, ec);
          if (ec)
            return fatal(ec.message());
          candidate->print(output,
                           mlir::OpPrintingFlags().printGenericOpForm());
          output << "\n";
        }
        return 0;
#else
        return fatal("AIR channel legalization requires ORBIT_ENABLE_AIR=ON");
#endif
      }
      if (!checkAIE.empty()) {
#ifdef ORBIT_ENABLE_AIR
        context.allowUnregisteredDialects();
        auto candidateBytes = llvm::MemoryBuffer::getFile(checkAIE);
        if (!candidateBytes)
          return fatal(candidateBytes.getError().message());
        auto candidate = parse((*candidateBytes)->getBuffer());
        if (!candidate)
          return fatal("invalid generic final-AIE MLIR");
        llvm::json::Value receipt = llvm::json::Object{};
        bool valid = checkGenericAIEImplementation(analysis, plan, *candidate,
                                                   receipt, error);
        if (auto *object = receipt.getAsObject()) {
          (*object)["valid"] = valid;
          (*object)["diagnostic"] = error;
        }
        if (!writeJSON(implementationOutput, std::move(receipt)) ||
            !writeJSON(implementationFactsOutput,
                       extractGenericAIRImplementation(*candidate)))
          return fatal(error);
        if (!valid)
          return fatal(error);
        return 0;
#else
        return fatal("AIE implementation checker requires ORBIT_ENABLE_AIR=ON");
#endif
      }
      if (!checkAIR.empty()) {
#ifdef ORBIT_ENABLE_AIR
        context.allowUnregisteredDialects();
        auto candidateBytes = llvm::MemoryBuffer::getFile(checkAIR);
        if (!candidateBytes)
          return fatal(candidateBytes.getError().message());
        auto candidate = parse((*candidateBytes)->getBuffer());
        if (!candidate)
          return fatal("invalid generic AIR MLIR");
        bool valid = checkGenericAIRMapping(analysis, plan, *candidate, error);
        // Keep the receipt alive through both writes, matching the named AIE
        // receipt above and making ownership apparent during review.
        llvm::json::Value receipt = llvm::json::Object{
            {"level", "AIR"}, {"valid", valid}, {"diagnostic", error}};
        if (!writeJSON(implementationOutput, receipt) ||
            !writeJSON(implementationFactsOutput,
                       extractGenericAIRImplementation(*candidate)))
          return fatal(error);
        if (!valid)
          return fatal(error);
        return 0;
#else
        return fatal("AIR mapping checker requires ORBIT_ENABLE_AIR=ON");
#endif
      }
      if (lowerAIR) {
#ifdef ORBIT_ENABLE_AIR
        mlir::OwningOpRef<mlir::ModuleOp> air;
        if (!lowerGenericToAIR(analysis, plan, air, error, airApproximateExp))
          return fatal(error);
        if (airOutput.empty())
          return fatal("--air-output required");
        std::error_code ec;
        llvm::raw_fd_ostream output(airOutput, ec);
        if (ec)
          return fatal(ec.message());
        air->print(output, mlir::OpPrintingFlags().printGenericOpForm());
        output << "\n";
#else
        return fatal("direct AIR backend not built: configure "
                     "ORBIT_ENABLE_AIR=ON; Core remains independent");
#endif
      }
    }
    return 0;
  }
  if (staticMap || !checkMapping.empty()) {
    auto module = parse(text);
    if (!module || mlir::failed(mlir::verify(*module)))
      return fatal("static planning requires valid Taskflow MLIR");
    StaticAnalysis analysis;
    StaticConfig config;
    if (!analyzeStaticProgram(*module, analysis, error))
      return fatal(error);
    auto cfg = readJSON(targetConfig);
    if (!cfg)
      return fatal(llvm::toString(cfg.takeError()));
    if (auto object = cfg->getAsObject())
      if (auto path = object->getString("mapping_config")) {
        auto nested = readJSON(path->str());
        if (!nested)
          return fatal(llvm::toString(nested.takeError()));
        cfg = std::move(nested);
      }
    if (!parseStaticConfig(*cfg, config, error))
      return fatal(error);
    if (!analysisOutput.empty() &&
        !writeJSON(analysisOutput, serializeStaticAnalysis(analysis)))
      return fatal(error);
    if (!checkMapping.empty()) {
      auto value = readJSON(checkMapping);
      StaticPlan plan;
      if (!value)
        return fatal(llvm::toString(value.takeError()));
      if (!parseStaticPlan(*value, plan, error))
        return fatal(error);
      std::vector<std::string> errors;
      bool passed = checkStaticPlan(analysis, config, plan, errors);
      llvm::json::Array messages;
      for (auto &s : errors)
        messages.push_back(s);
      if (!writeJSON(mappingOutput,
                     llvm::json::Object{{"passed", passed},
                                        {"errors", std::move(messages)}}))
        return fatal(error);
      return passed ? 0 : 1;
    }
    auto result = mapStaticProgram(analysis, config);
    if (!writeJSON(mappingOutput, serializeStaticResult(result)))
      return fatal(error);
    return result.status == "passed" ? 0 : 1;
  }
  if (text.ltrim().starts_with("{")) {
    auto json = llvm::json::parse(text);
    if (!json)
      return fatal(llvm::toString(json.takeError()));
    if (!parsePlan(*json, p, error))
      return fatal(error);
  } else {
    auto module = parse(text);
    if (!module || mlir::failed(mlir::verify(*module)))
      return fatal("MLIR verification failed");
    if (search && backend == "static" &&
        (evaluator.empty() || evaluator == "native")) {
      StaticAnalysis analysis;
      if (!analyzeStaticProgram(*module, analysis, error))
        return fatal(error);
      p = std::move(analysis.program);
    } else {
      auto witness =
          module->getOperation()->getAttrOfType<mlir::StringAttr>("orbit.plan");
      if (!witness) {
        if (verifyOnly)
          return 0;
        return fatal("unsupported MLIR: missing orbit.plan structured witness");
      }
      auto json = llvm::json::parse(witness.getValue());
      if (!json)
        return fatal(llvm::toString(json.takeError()));
      if (!parsePlan(*json, p, error))
        return fatal(error);
      auto expected = parse(emitTaskflow(p));
      if (!expected)
        return fatal("internal witness materialization failure");
      module->getOperation()->removeAttr("orbit.plan");
      if (!mlir::OperationEquivalence::isEquivalentTo(
              module->getOperation(), expected->getOperation(),
              mlir::OperationEquivalence::Flags::IgnoreLocations))
        return fatal("Taskflow body, slices, storage or dependencies disagree "
                     "with the exchange witness");
      if (verifyOnly)
        return 0;
    }
  }
  if (search) {
    SearchOptions options{backend,      evaluator,        searchOutput,
                          targetConfig, searchCheckpoint, searchRounds,
                          searchBudget, searchWidth,      searchResume};
    return runStructuredSearch(p, options, error) ? 0 : fatal(error);
  }
  if (!transform(p, {tm, tn, tk, fuse, fission}, error))
    return fatal(error);
  auto module = parse(emitTaskflow(p));
  if (!module || mlir::failed(mlir::verify(*module)))
    return fatal("materialized Taskflow verification failed");
  if (nativeInterface) {
    if (!importNativeInterface(*module, p, error))
      return fatal(error);
    if (mlirFile.empty())
      return fatal("--native-interface requires --emit-mlir");
    std::error_code ec;
    llvm::raw_fd_ostream output(mlirFile, ec);
    if (ec)
      return fatal(ec.message());
    module->print(output);
    output << "\n";
    return 0;
  }
  std::string json;
  llvm::raw_string_ostream js(json);
  js << llvm::formatv("{0:2}", printPlan(p));
  module->getOperation()->setAttr("orbit.plan",
                                  mlir::StringAttr::get(&context, json));
  auto output = [&](const std::string &path, auto printer) -> bool {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      error = ec.message();
      return false;
    }
    printer(os);
    return true;
  };
  if (!planFile.empty() &&
      !output(planFile, [&](auto &os) { os << json << "\n"; }))
    return fatal(error);
  if (!mlirFile.empty() && !output(mlirFile, [&](auto &os) {
        module->print(os);
        os << "\n";
      }))
    return fatal(error);
  if (planFile.empty() && mlirFile.empty()) {
    module->print(llvm::outs());
    llvm::outs() << "\n";
  }
  return 0;
}
