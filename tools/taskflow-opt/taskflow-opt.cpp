#include "Taskflow/GenericProgramTiming.h"
#include "Taskflow/Search.h"
#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramPartition.h"
#include "Taskflow/ProgramJointSearch.h"
#include "Taskflow/HeterogeneousSearch.h"
#ifdef ORBIT_ENABLE_AIR
#include "Taskflow/AIRBackend.h"
#include "Taskflow/NPUIndependentShards.h"
#include "Taskflow/ProgramJointAIR.h"
#endif
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Taskflow/StaticMapping.h"
#include "Taskflow/StructuredProgram.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#ifdef ORBIT_ENABLE_AIR
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#endif
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
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
static cl::opt<bool> genericAnalyze("generic-analyze"), genericMap("generic-map"), lowerAIR("lower-to-air"), nativeInterface("native-interface"), airPassPipeline("air-pass-pipeline"), airApproximateExp("air-approximate-exp");
static cl::opt<std::string> airOutput("air-output"), checkAIR("check-air"), checkAIE("check-aie"),
    implementationOutput("implementation-output"), implementationFactsOutput("implementation-facts-output");
static cl::opt<std::string> programTimingOutput("program-timing-output",
    cl::desc("Export checked native features, composite timeline and unidentified internal events"));
static cl::opt<bool> programGraph("program-graph",
    cl::desc("Build and check the public task/data/communication graph"));
static cl::opt<std::string> programGraphOutput("program-graph-output"),
    programGraphDOT("program-graph-dot"), checkProgramGraphFile("check-program-graph");
static cl::opt<std::string> legalizeAIRChannels("legalize-air-channels",
    cl::desc("Apply checked plan channel resources to converted generic AIR")),
    checkAIRChannels("check-air-channels",
    cl::desc("Check converted AIR channel resources against the native plan"));
static cl::opt<bool> airDistributeInitialization("air-distribute-initialization",
    cl::desc("Distribute provably independent initialization before AIR tiling"));
static cl::opt<std::string> residency("residency",
    cl::desc("Legacy target residency spelling: phase-local or readonly-panel")),
    dataOrganization("data-organization",
    cl::desc("Public data organization: phase-local or retain-across-users")),
    implementation("implementation",
    cl::desc("Explicit target implementation: scalar, independent-lane-f32, or scalar-product-vector-add-f32 (installed AIR: 8 lanes)"));
static cl::opt<int64_t> residentStorage("resident-storage", cl::init(-1)),
    reuseLoop("reuse-loop", cl::init(-1)), panelLoop("panel-loop", cl::init(-1)),
    vectorLanes("vector-lanes", cl::init(1));
static cl::opt<int64_t> programPartition("program-partition", cl::init(0),
    cl::desc("Independent whole-program source-axis instances per command"));
static cl::opt<std::string> lowerProgram("lower-program-to-air",
    cl::desc("Write checked partition templates and invocation manifest to a NEW directory"));
static cl::opt<std::string> heterogeneousOutput("heterogeneous-output",
    cl::desc("Export checked CPU/NPU regions and native implementations to a NEW directory"));
static cl::opt<std::string> checkHeterogeneous("check-heterogeneous",
    cl::desc("Compare a static manifest with source/target reconstruction"));
static cl::opt<std::string> lowerMulticore("lower-multicore-to-air",cl::desc("Write a CP-SAT-placed independent-shard AIR program to a NEW directory"));
static cl::opt<std::string> checkMulticoreAIE("check-multicore-aie",cl::desc("Independently check final multi-core AIE for the source and target"));
static cl::opt<int64_t> npuShards("npu-shards",cl::init(2));
static cl::opt<std::string> jointSearch("joint-program-search",cl::desc("Joint public ProgramGraph CP-SAT search; write evidence to a new directory"));
static cl::opt<std::string> jointPlan("joint-program-plan",cl::desc("Recheck source-bound joint decisions from this JSON file without solving"));
static cl::opt<std::string> checkJointAIE("check-joint-aie");
static cl::opt<bool> jointBaselineOnly("joint-baseline-only");
static cl::opt<bool> jointLower("joint-lower-air",cl::desc("Lower the chosen public joint graph to AIR in the search directory"));
static cl::opt<double> jointSeconds("joint-seconds",cl::init(5));
static cl::opt<int64_t> jointVariables("joint-max-variables",cl::init(100000));
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
                              "ORBIT independent structured Taskflow core\n");
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
#ifdef ORBIT_ENABLE_AIR
  registry.insert<mlir::vector::VectorDialect>();
  context.appendDialectRegistry(registry);
  if (lowerAIR || jointLower || !checkJointAIE.empty() || !lowerMulticore.empty() || !checkMulticoreAIE.empty() || !lowerProgram.empty() || !heterogeneousOutput.empty() || airPassPipeline || (!checkAIR.empty() || !checkAIE.empty()) ||
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
      if(programPartition<1 || programPartition>INT32_MAX) {
        error="program partition must be in [1,INT32_MAX]";return false;
      }
      module->setAttr("orbit.program_partition_extent",
          mlir::Builder(module.getContext()).getI64IntegerAttr(programPartition));
    }
    if (!residency.empty() || !dataOrganization.empty()) {
      ProgramMappingChoice choice;
      choice.dataOrganization=dataOrganization;
      choice.residency=residency; choice.storage=residentStorage;
      choice.reuseLoop=reuseLoop; choice.panelLoop=panelLoop;
      if (!setNativeMappingChoice(module,choice,error)) return false;
    } else if (residentStorage!=-1 || reuseLoop!=-1 || panelLoop!=-1) {
      error="resident storage and loop choices require --data-organization or --residency";return false;
    }
    if (!implementation.empty()) {
      if (!setNativeImplementation(module,implementation,vectorLanes,error))return false;
    } else if (vectorLanes!=1) {
      error="--vector-lanes requires --implementation";return false;
    }
    return normalizeNativeMappingChoice(module,error);
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
    return true;
  };
  auto emitGraph = [&](const ProgramGraph &graph) {
    if (!checkProgramGraph(graph,error)) return false;
    auto view=serializeProgramGraph(graph);
    if (!checkProgramGraphFile.empty()) {
      auto expected=readJSON(checkProgramGraphFile);
      if (!expected) { error=llvm::toString(expected.takeError());return false; }
      if (staticFingerprint(*expected)!=staticFingerprint(view)) {error="program graph view differs from checked source/target/candidate";return false;}
    }
    if (!programGraphOutput.empty() && !writeJSON(programGraphOutput,view)) return false;
    if (!programGraphDOT.empty()) {
      std::error_code ec;llvm::raw_fd_ostream output(programGraphDOT,ec);
      if(ec){error=ec.message();return false;}output<<renderProgramGraphDOT(graph);
    }
    return true;
  };
  const bool wantsGraph=!programTimingOutput.empty() || programGraph || !programGraphOutput.empty() || !programGraphDOT.empty() || !checkProgramGraphFile.empty();
  if(search&&backend=="air"&&!targetConfig.empty()){
    auto cfg=readJSON(targetConfig);if(!cfg)return fatal(llvm::toString(cfg.takeError()));auto root=cfg->getAsObject();auto options=root?root->getObject("native_auto_candidates"):nullptr;
    if(options&&options->getString("algorithm")=="cp-sat-program"){if(options->getString("mode")!="joint")return fatal("cp-sat-program requires native_auto_candidates.mode=joint");if(searchResume||!searchCheckpoint.empty())return fatal("public joint search currently writes a fresh finite solve; staged-search resume is unsupported");for(auto &entry:*options)if(entry.first!="mode"&&entry.first!="algorithm")return fatal("cp-sat-program domain and budgets use program_joint and --joint-seconds, not legacy proposal fields");if(searchOutput.empty())return fatal("--search-output required for public joint search");jointSearch=searchOutput.getValue();}
  }
  if(!jointSearch.empty() || !jointPlan.empty() || !checkJointAIE.empty()) {
    auto module=parse(text);if(!module||mlir::failed(mlir::verify(*module)))return fatal("valid native Taskflow source required");
    llvm::json::Value config=llvm::json::Object{};if(!targetConfig.empty()){auto c=readJSON(targetConfig);if(!c)return fatal(llvm::toString(c.takeError()));config=std::move(*c);}
#ifdef ORBIT_ENABLE_AIR
    if(!configureProgramJointNPU(config,npuShards,error))return fatal(error);
#endif
    GenericAnalysis analysis;ProgramJointDomain domain;if(!analyzeGenericProgram(*module,analysis,error)||!buildProgramJointDomain(analysis,config,domain,error))return fatal(error);
    auto baseline=baselineProgramJointDomain(domain);ProgramGraph baselineGraph;
    if(baseline.hasPlan&&(!materializeProgramJointSolution(domain,baseline,baselineGraph,error)||!checkProgramJointSolution(domain,baseline,baselineGraph,error)))return fatal(error);
    if(!jointSearch.empty()){
      if(llvm::sys::fs::exists(jointSearch))return fatal("joint output directory must be new");if(auto ec=llvm::sys::fs::create_directories(jointSearch))return fatal(ec.message());
      if(!writeJSON(jointSearch+"/domain.json",serializeProgramJointDomain(domain))||!writeJSON(jointSearch+"/baseline.json",serializeProgramJointSolution(domain,baseline,baseline.hasPlan?&baselineGraph:nullptr))||!writeJSON(jointSearch+"/target.json",config))return fatal(error);
#ifdef ORBIT_ENABLE_AIR
      if(jointLower&&baseline.hasPlan){mlir::OwningOpRef<mlir::ModuleOp> air;llvm::json::Value manifest=nullptr;if(!lowerProgramJointToAIR(*module,config,domain,baseline,baselineGraph,air,manifest,error))return fatal("baseline cannot lower: "+error);std::error_code ec;llvm::raw_fd_ostream stream(jointSearch+"/baseline.air.mlir",ec);if(ec)return fatal(ec.message());air->print(stream,mlir::OpPrintingFlags().printGenericOpForm());stream<<"\n";if(!writeJSON(jointSearch+"/baseline-air.json",manifest))return fatal(error);}
#endif
    }
    ProgramJointSolution solution;
    if(!jointPlan.empty()){auto record=readJSON(jointPlan);if(!record)return fatal(llvm::toString(record.takeError()));auto object=record->getAsObject();if(!object||object->getString("domain_id")!=domain.identity)return fatal("joint plan source/target/domain identity mismatch");if(!parseProgramJointSolution(*record,solution,error))return fatal(error);}
    else if(jointBaselineOnly)solution=baseline;
    else {ConstraintSearchOptions options;options.wallSeconds=jointSeconds;options.maxVariables=jointVariables;solution=solveProgramJointDomain(domain,options,&baseline);}
    ProgramGraph graph;if(solution.hasPlan&&(!materializeProgramJointSolution(domain,solution,graph,error)||!checkProgramJointSolution(domain,solution,graph,error)))return fatal(error);
    if(!jointSearch.empty()&&!writeJSON(jointSearch+"/solution.json",serializeProgramJointSolution(domain,solution,solution.hasPlan?&graph:nullptr)))return fatal(error);
    if(!jointPlan.empty()&&solution.hasPlan){auto record=readJSON(jointPlan);auto stored=record?record->getAsObject()->get("program_graph"):nullptr;if(!stored||staticFingerprint(*stored)!=staticFingerprint(serializeProgramGraph(graph)))return fatal("stored joint public graph differs from checked decisions");}
    if(!checkJointAIE.empty()){
#ifdef ORBIT_ENABLE_AIR
      if(jointPlan.empty())return fatal("--check-joint-aie requires --joint-program-plan to check frozen choices without solving");
      auto candidateBytes=llvm::MemoryBuffer::getFile(checkJointAIE);if(!candidateBytes)return fatal(candidateBytes.getError().message());auto actual=parse((*candidateBytes)->getBuffer());if(!actual)return fatal("valid generic final AIE required");llvm::json::Value receipt=nullptr;bool valid=checkProgramJointAIE(*module,config,domain,solution,graph,*actual,receipt,error);if(!writeJSON(implementationOutput,receipt))return fatal(error);return valid?0:fatal(error);
#else
      return fatal("joint AIE checker requires ORBIT_ENABLE_AIR=ON");
#endif
    }
    if(jointLower){
#ifdef ORBIT_ENABLE_AIR
      if(jointSearch.empty()||!solution.hasPlan)return fatal("--joint-lower-air requires a search directory and checked plan");
      mlir::OwningOpRef<mlir::ModuleOp> air;llvm::json::Value manifest=nullptr;if(!lowerProgramJointToAIR(*module,config,domain,solution,graph,air,manifest,error))return fatal(error);
      std::error_code ec;llvm::raw_fd_ostream stream(jointSearch+"/program.air.mlir",ec);if(ec)return fatal(ec.message());air->print(stream,mlir::OpPrintingFlags().printGenericOpForm());stream<<"\n";
      if(!writeJSON(jointSearch+"/manifest.json",manifest)||!writeJSON(jointSearch+"/abi.json",*manifest.getAsObject()->get("abi"))||!writeJSON(jointSearch+"/target.json",config))return fatal(error);
#else
      return fatal("joint AIR lowering requires ORBIT_ENABLE_AIR=ON");
#endif
    }
    if(solution.hasPlan&&!emitGraph(graph))return fatal(error);
    if(!mappingOutput.empty()&&!writeJSON(mappingOutput,serializeProgramJointSolution(domain,solution,solution.hasPlan?&graph:nullptr)))return fatal(error);
    return solution.hasPlan?0:fatal(solution.message);
  }
  if(!lowerMulticore.empty() || !checkMulticoreAIE.empty()) {
#ifdef ORBIT_ENABLE_AIR
    auto module=parse(text);if(!module || mlir::failed(mlir::verify(*module)))return fatal("valid Taskflow source required");
    llvm::json::Value config=llvm::json::Object{};if(!targetConfig.empty()){auto c=readJSON(targetConfig);if(!c)return fatal(llvm::toString(c.takeError()));config=std::move(*c);}
    if(!checkMulticoreAIE.empty()) {
      auto candidateBytes=llvm::MemoryBuffer::getFile(checkMulticoreAIE);if(!candidateBytes)return fatal(candidateBytes.getError().message());auto actual=parse((*candidateBytes)->getBuffer());if(!actual)return fatal("valid final AIE module required");llvm::json::Value receipt=nullptr;
      bool valid=checkNPUIndependentAIE(*module,config,npuShards,*actual,receipt,error);if(!implementationOutput.empty()&&!writeJSON(implementationOutput,receipt))return fatal(error);return valid?0:fatal(error);
    }
    if(llvm::sys::fs::exists(lowerMulticore))return fatal("multicore output directory must be new");
    mlir::OwningOpRef<mlir::ModuleOp> air;llvm::json::Value manifest=nullptr;if(!lowerNPUIndependentProgram(*module,config,npuShards,air,manifest,error))return fatal(error);
    if(auto ec=llvm::sys::fs::create_directories(lowerMulticore))return fatal(ec.message());
    auto writeModule=[&](const std::string &path,mlir::ModuleOp value,bool generic){std::error_code ec;llvm::raw_fd_ostream out(path,ec);if(ec){error=ec.message();return false;}auto flags=mlir::OpPrintingFlags();if(generic)flags.printGenericOpForm();value.print(out,flags);out<<"\n";return true;};
    if(!writeModule(lowerMulticore+"/source.mlir",*module,false)||!writeModule(lowerMulticore+"/program.air.mlir",*air,true)||!writeJSON(lowerMulticore+"/manifest.json",manifest)||!writeJSON(lowerMulticore+"/target.json",config))return fatal(error);
    GenericAnalysis analysis;if(!analyzeGenericProgram(*module,analysis,error))return fatal(error);llvm::json::Array buffers,outputs;for(const auto &storage:analysis.storages){if(!mlir::isa<mlir::BlockArgument>(storage.root))continue;llvm::json::Array shape;for(auto n:storage.shape)shape.push_back(n);buffers.push_back(llvm::json::Object{{"name",storage.id},{"role",storage.role},{"dtype",storage.dtype},{"shape",std::move(shape)}});if(storage.role=="output")outputs.push_back(storage.id);}
    if(!writeJSON(lowerMulticore+"/abi.json",llvm::json::Object{{"buffers",std::move(buffers)},{"outputs",std::move(outputs)}}))return fatal(error);return 0;
#else
    return fatal("multicore lowering requires ORBIT_ENABLE_AIR=ON");
#endif
  }
  if(search && backend=="cpu-npu") {
    auto module=parse(text);if(!module)return fatal("native Taskflow source required");
    SearchOptions options{backend,evaluator,searchOutput,targetConfig,searchCheckpoint,searchRounds,searchBudget,searchWidth,searchResume};
    if(!configureHeterogeneousSearch(*module,options,error))return fatal(error);
    return runStructuredSearch(Program{},options,error)?0:fatal(error);
  }
  if(!heterogeneousOutput.empty()) {
    if(llvm::sys::fs::exists(heterogeneousOutput))return fatal("heterogeneous output directory must be new");
    auto module=parse(text);if(!module || mlir::failed(mlir::verify(*module)))return fatal("valid Taskflow source required");
    llvm::json::Value config=llvm::json::Object{};
    if(!targetConfig.empty()){auto c=readJSON(targetConfig);if(!c)return fatal(llvm::toString(c.takeError()));config=std::move(*c);}
    HeterogeneousAIREmitter emitter;
#ifdef ORBIT_ENABLE_AIR
    emitter=[&](const GenericAnalysis &source,const GenericMappingPlan &plan,const std::string &path,std::string &error) {
      mlir::OwningOpRef<mlir::ModuleOp> air;
      if(!lowerGenericToAIR(source,plan,air,error))return false;
      std::error_code ec;llvm::raw_fd_ostream out(path,ec);if(ec){error=ec.message();return false;}
      air->print(out,mlir::OpPrintingFlags().printGenericOpForm());out<<"\n";return true;
    };
#endif
    llvm::json::Value manifest=nullptr;
    if(!exportHeterogeneousProgram(*module,config,heterogeneousOutput,manifest,error,emitter))return fatal(error);
    if(!checkHeterogeneous.empty()){
      auto expected=readJSON(checkHeterogeneous);if(!expected)return fatal(llvm::toString(expected.takeError()));
      if(staticFingerprint(*expected)!=staticFingerprint(manifest))return fatal("heterogeneous manifest differs from source/target reconstruction");
    }
    return 0;
  }
  if (airPassPipeline) {
    if (airApproximateExp || !targetConfig.empty()) return fatal("--air-pass-pipeline uses the default exact resident plan; use --lower-to-air for explicit target/approximation options");
#ifdef ORBIT_ENABLE_AIR
    auto module=parse(text);if(!module)return fatal("invalid input MLIR");
    mlir::PassManager pipeline(&context);pipeline.addPass(createTaskflowToAIRPass());
    if(mlir::failed(pipeline.run(*module)))return fatal("taskflow-to-air pipeline failed");
    if(airOutput.empty())return fatal("--air-output required");
    std::error_code ec;llvm::raw_fd_ostream output(airOutput,ec);if(ec)return fatal(ec.message());
    module->print(output,mlir::OpPrintingFlags().printGenericOpForm());output << "\n";return 0;
#else
    return fatal("configure ORBIT_ENABLE_AIR=ON for direct AIR pass");
#endif
  }
  if (search && backend == "air" && text.ltrim().starts_with("module")) {
    SearchOptions options{backend,evaluator,searchOutput,targetConfig,searchCheckpoint,searchRounds,searchBudget,searchWidth,searchResume};
    options.nativeSourceMLIR=text.str();
    if (airDistributeInitialization || tm || tn || tk || !residency.empty() || !dataOrganization.empty() ||
        !implementation.empty() || residentStorage!=-1 || reuseLoop!=-1 ||
        panelLoop!=-1 || vectorLanes!=1 || programPartition) {
      auto module=parse(text); if(!module)return fatal("invalid input MLIR");
      if (airDistributeInitialization && !distributeNativeInitialization(*module,error))return fatal(error);
      if ((tm || tn || tk) && !setNativeAIRTileSizes(*module,tm,tn,tk,error))return fatal(error);
      if (!applyMappingOptions(*module))return fatal(error);
      options.nativeSourceMLIR.clear(); llvm::raw_string_ostream os(options.nativeSourceMLIR);
      module->print(os);
    }
    return runStructuredSearch(Program{},options,error) ? 0 : fatal(error);
  }
  if (wantsGraph || genericAnalyze || genericMap || lowerAIR || !lowerProgram.empty() || (!checkAIR.empty() || !checkAIE.empty()) ||
      !legalizeAIRChannels.empty() || !checkAIRChannels.empty()) {
    auto module = parse(text);
    if (!module || mlir::failed(mlir::verify(*module))) return fatal("valid native Taskflow MLIR required");
    if (airDistributeInitialization &&
        !distributeNativeInitialization(*module, error)) return fatal(error);
    if ((tm || tn || tk) && !setNativeAIRTileSizes(*module, tm, tn, tk, error))
      return fatal(error);
    if (!applyMappingOptions(*module)) return fatal(error);
    if (!mlirFile.empty()) {
      std::error_code ec; llvm::raw_fd_ostream output(mlirFile, ec);
      if (ec) return fatal(ec.message());
      module->print(output); output << "\n";
    }
    GenericAnalysis analysis;
    if (!analyzeGenericProgram(*module, analysis, error)) return fatal(error);
    if (!writeJSON(analysisOutput, serializeGenericAnalysis(analysis))) return fatal(error);
    if (wantsGraph || genericMap || lowerAIR || !lowerProgram.empty() || (!checkAIR.empty() || !checkAIE.empty()) ||
        !legalizeAIRChannels.empty() || !checkAIRChannels.empty()) {
      llvm::json::Value config = llvm::json::Object{};
      if (!targetConfig.empty()) {auto value=readJSON(targetConfig); if(!value)return fatal(llvm::toString(value.takeError()));config=std::move(*value);}
      const auto *configuration=config.getAsObject();
      const auto *graphConfiguration=configuration?configuration->getObject("program_graph"):nullptr;
      if(graphConfiguration && graphConfiguration->getString("adapter")=="whole-task") {
        if(!programTimingOutput.empty())return fatal("--program-timing-output requires the native NPU adapter; use evaluateProgramTiming for whole-task backend annotations");
        if(lowerAIR || !lowerProgram.empty() || (!checkAIR.empty() || !checkAIE.empty()) || !legalizeAIRChannels.empty() || !checkAIRChannels.empty())
          return fatal("whole-task public graph is a CPU planning adapter; direct AIR requires the checked native target realization");
        StaticConfig target;ProgramDataOrganizationChoice choice;ProgramGraph graph;
        if(!parseWholeTaskProgramGraphConfig(config,target,choice,error) ||
           !planProgramDataOrganization(analysis,target,choice,graph,error) || !emitGraph(graph) ||
           !writeJSON(mappingOutput,serializeProgramGraph(graph)))return fatal(error);
        return 0;
      }
      if(graphConfiguration && graphConfiguration->getString("adapter")!="native")return fatal("unsupported program graph adapter");
      if(module->getOperation()->hasAttr("orbit.program_partition_extent")) {
        if(lowerAIR || (!checkAIR.empty() || !checkAIE.empty()) || !legalizeAIRChannels.empty() || !checkAIRChannels.empty())
          return fatal("partitioned program requires --lower-program-to-air; a single AIR command would ignore its schedule");
        auto chunk=module->getOperation()->getAttrOfType<mlir::IntegerAttr>("orbit.program_partition_extent");
        ProgramPartitionPlan partition;
        if(!chunk || !planProgramPartition(analysis,chunk.getInt(),partition,error))return fatal(error);
        auto evaluated=evaluatePartitionedGenericMLIR(*module,config,error);
        auto result=evaluated.getAsObject();auto best=result?result->get("best_plan"):nullptr;
        if(!best || result->getString("status")=="unsupported")return fatal(error.empty()?"unsupported partition target implementation":error);
        if(!writeJSON(mappingOutput,*best))return fatal(error);
        if(lowerProgram.empty())return 0;
#ifdef ORBIT_ENABLE_AIR
        if(llvm::sys::fs::exists(lowerProgram))return fatal("use a new program output directory");
        if(auto ec=llvm::sys::fs::create_directories(lowerProgram))return fatal(ec.message());
        auto writeModule=[&](const std::string &path,mlir::ModuleOp value,bool generic) {
          std::error_code ec;llvm::raw_fd_ostream out(path,ec);
          if(ec){error=ec.message();return false;}
          auto flags=mlir::OpPrintingFlags();if(generic)flags.printGenericOpForm();
          value.print(out,flags);out<<"\n";return true;
        };
        if(!writeModule(lowerProgram+"/taskflow.mlir",*module,false) ||
           !writeJSON(lowerProgram+"/config.json",config))return fatal(error);
        auto manifest=llvm::json::Value(*best);
        auto &manifestObject=*manifest.getAsObject();
        for(auto &value:*manifestObject.getArray("templates")) {
          auto &record=*value.getAsObject();auto extent=*record.getInteger("extent");
          auto prefix=lowerProgram+"/template-"+std::to_string(extent);
          if(auto ec=llvm::sys::fs::create_directories(prefix))return fatal(ec.message());
          mlir::OwningOpRef<mlir::ModuleOp> child,air;
          GenericAnalysis facts;GenericMappingPlan childPlan;
          if(!materializeProgramPartition(*module,partition,extent,child,error) ||
             !analyzeGenericProgram(*child,facts,error) ||
             !planGenericMapping(facts,config,childPlan,error) ||
             !lowerGenericToAIR(facts,childPlan,air,error,airApproximateExp))return fatal(error);
          if(llvm::any_of(childPlan.hardwareStages,[](const auto &stage){return stage.legalizeChannels;}))
            return fatal("partition template currently requires canonical direct-DMA stages; channel legalization must be explicit");
          if(!writeModule(prefix+"/taskflow.mlir",*child,false) ||
             !writeModule(prefix+"/program.air.mlir",*air,true) ||
             !writeJSON(prefix+"/facts.json",serializeGenericAnalysis(facts)) ||
             !writeJSON(prefix+"/plan.json",serializeGenericMapping(facts,childPlan)))return fatal(error);
          record["mapping_plan_id"]=childPlan.planId;
          llvm::json::Array buffers,outputs;
          for(bool temporary:{false,true})for(const auto &storage:facts.storages)
            if((storage.role=="temporary")==temporary) {
              llvm::json::Array shape;for(auto size:storage.shape)shape.push_back(size);
              buffers.push_back(llvm::json::Object{{"name",storage.id},{"role",storage.role},
                  {"dtype",storage.dtype},{"shape",std::move(shape)}});
              if(storage.role=="output")outputs.push_back(storage.id);
            }
          if(!writeJSON(prefix+"/abi.json",llvm::json::Object{{"buffers",std::move(buffers)},
              {"outputs",std::move(outputs)}}))return fatal(error);
        }
        llvm::json::Array invocation;for(int i=0;i<argc;++i)invocation.push_back(argv[i]);
        manifestObject["compiler_invocation"]=std::move(invocation);
        manifestObject["approximate_exp"]=bool(airApproximateExp);
        if(!writeJSON(lowerProgram+"/manifest.json",manifest))return fatal(error);
        return 0;
#else
        return fatal("partition AIR emission requires ORBIT_ENABLE_AIR=ON");
#endif
      }
      if(!lowerProgram.empty())return fatal("--lower-program-to-air requires a program partition choice");
      GenericMappingPlan plan;
      if (!planGenericMapping(analysis, config, plan, error)) return fatal(error);
      if(wantsGraph && !emitGraph(plan.communication))return fatal(error);
      if (!programTimingOutput.empty()) {
        llvm::json::Value timing=nullptr;
        if (!evaluateGenericProgramTiming(analysis,plan,timing,error) ||
            !writeJSON(programTimingOutput,timing))return fatal(error);
      }
      if (!writeJSON(mappingOutput, serializeGenericMapping(analysis,plan))) return fatal(error);
      if (!legalizeAIRChannels.empty() || !checkAIRChannels.empty()) {
#ifdef ORBIT_ENABLE_AIR
        if (!legalizeAIRChannels.empty() && !checkAIRChannels.empty())
          return fatal("select either --legalize-air-channels or --check-air-channels");
        const std::string path = legalizeAIRChannels.empty()
                                     ? checkAIRChannels.getValue()
                                     : legalizeAIRChannels.getValue();
        auto candidateBytes = llvm::MemoryBuffer::getFile(path);
        if (!candidateBytes) return fatal(candidateBytes.getError().message());
        auto candidate = parse((*candidateBytes)->getBuffer());
        if (!candidate || mlir::failed(mlir::verify(*candidate)))
          return fatal("valid converted generic AIR required");
        if (!legalizeAIRChannels.empty() &&
            !legalizeGenericAIRChannels(analysis, plan, *candidate, error))
          return fatal(error);
        if (!checkGenericAIRChannels(analysis, plan, *candidate, error) ||
            mlir::failed(mlir::verify(*candidate))) return fatal(error);
        if (!legalizeAIRChannels.empty()) {
          if (airOutput.empty()) return fatal("--air-output required");
          std::error_code ec; llvm::raw_fd_ostream output(airOutput, ec);
          if (ec) return fatal(ec.message());
          candidate->print(output, mlir::OpPrintingFlags().printGenericOpForm());
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
        auto candidateBytes=llvm::MemoryBuffer::getFile(checkAIE);if(!candidateBytes)return fatal(candidateBytes.getError().message());
        auto candidate=parse((*candidateBytes)->getBuffer());
        if(!candidate)return fatal("invalid generic final-AIE MLIR");
        llvm::json::Value receipt=llvm::json::Object{};
        bool valid=checkGenericAIEImplementation(analysis,plan,*candidate,receipt,error);
        if(auto *object=receipt.getAsObject()){(*object)["valid"]=valid;(*object)["diagnostic"]=error;}
        if(!writeJSON(implementationOutput,std::move(receipt)) ||
           !writeJSON(implementationFactsOutput,extractGenericAIRImplementation(*candidate)))return fatal(error);
        if(!valid)return fatal(error);
        return 0;
#else
        return fatal("AIE implementation checker requires ORBIT_ENABLE_AIR=ON");
#endif
      }
      if (!checkAIR.empty()) {
#ifdef ORBIT_ENABLE_AIR
        context.allowUnregisteredDialects();
        auto candidateBytes=llvm::MemoryBuffer::getFile(checkAIR);if(!candidateBytes)return fatal(candidateBytes.getError().message());
        auto candidate=parse((*candidateBytes)->getBuffer());
        if(!candidate)return fatal("invalid generic AIR MLIR");
        bool valid=checkGenericAIRMapping(analysis,plan,*candidate,error);
        if(!writeJSON(implementationOutput,llvm::json::Object{{"level","AIR"},{"valid",valid},{"diagnostic",error}}) ||
           !writeJSON(implementationFactsOutput,extractGenericAIRImplementation(*candidate)))return fatal(error);
        if(!valid)return fatal(error);
        return 0;
#else
        return fatal("AIR mapping checker requires ORBIT_ENABLE_AIR=ON");
#endif
      }
      if (lowerAIR) {
#ifdef ORBIT_ENABLE_AIR
        mlir::OwningOpRef<mlir::ModuleOp> air;
        if (!lowerGenericToAIR(analysis, plan, air, error, airApproximateExp)) return fatal(error);
        if (airOutput.empty()) return fatal("--air-output required");
        std::error_code ec;llvm::raw_fd_ostream output(airOutput,ec);if(ec)return fatal(ec.message());
        air->print(output,mlir::OpPrintingFlags().printGenericOpForm());output << "\n";
#else
        return fatal("direct AIR backend not built: configure ORBIT_ENABLE_AIR=ON; Core remains independent");
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
    if (!importNativeInterface(*module,p,error))return fatal(error);
    if (mlirFile.empty()) return fatal("--native-interface requires --emit-mlir");
    std::error_code ec;llvm::raw_fd_ostream output(mlirFile,ec);if(ec)return fatal(ec.message());
    module->print(output);output << "\n";return 0;
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
