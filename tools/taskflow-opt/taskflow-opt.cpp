#include "Taskflow/Search.h"
#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramPartition.h"
#ifdef ORBIT_ENABLE_AIR
#include "Taskflow/AIRBackend.h"
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
static cl::opt<std::string> airOutput("air-output"), checkAIR("check-air");
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
  if (lowerAIR || !lowerProgram.empty() || airPassPipeline || !checkAIR.empty() ||
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
  if (genericAnalyze || genericMap || lowerAIR || !lowerProgram.empty() || !checkAIR.empty() ||
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
    if (genericMap || lowerAIR || !lowerProgram.empty() || !checkAIR.empty() ||
        !legalizeAIRChannels.empty() || !checkAIRChannels.empty()) {
      llvm::json::Value config = llvm::json::Object{};
      if (!targetConfig.empty()) {auto value=readJSON(targetConfig); if(!value)return fatal(llvm::toString(value.takeError()));config=std::move(*value);}
      if(module->getOperation()->hasAttr("orbit.program_partition_extent")) {
        if(lowerAIR || !checkAIR.empty() || !legalizeAIRChannels.empty() || !checkAIRChannels.empty())
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
      if (!checkAIR.empty()) {
#ifdef ORBIT_ENABLE_AIR
        context.allowUnregisteredDialects();
        auto candidateBytes=llvm::MemoryBuffer::getFile(checkAIR);if(!candidateBytes)return fatal(candidateBytes.getError().message());
        auto candidate=parse((*candidateBytes)->getBuffer());
        if(!candidate || !checkGenericAIRMapping(analysis,plan,*candidate,error))return fatal(error);
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
