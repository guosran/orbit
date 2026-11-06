#include "Taskflow/HeterogeneousSearch.h"
#include "Taskflow/HeterogeneousImplementation.h"
#include "Taskflow/MechanismCost.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <set>
using namespace orbit::taskflow;
using namespace mlir;
namespace {
using J=llvm::json::Value;using O=llvm::json::Object;using A=llvm::json::Array;
bool fail(std::string &error,llvm::StringRef why){error=why.str();return false;}
bool write(const std::string &path,const J &value,std::string &error){std::error_code ec;llvm::raw_fd_ostream s(path,ec);if(ec)return fail(error,ec.message());s<<llvm::formatv("{0:2}",value)<<"\n";return true;}
bool moduleFile(const std::string &path,ModuleOp module,std::string &error){std::error_code ec;llvm::raw_fd_ostream s(path,ec);if(ec)return fail(error,ec.message());module.print(s);s<<"\n";return true;}
A ints(const std::vector<size_t> &v){A a;for(auto x:v)a.push_back(int64_t(x));return a;}
A dims(const std::vector<int64_t> &v){A a;for(auto x:v)a.push_back(x);return a;}
bool read(const std::string &path,J &out,std::string &error){auto b=llvm::MemoryBuffer::getFile(path);if(!b)return fail(error,b.getError().message());auto j=llvm::json::parse((*b)->getBuffer());if(!j)return fail(error,llvm::toString(j.takeError()));out=std::move(*j);return true;}
// Shape-independent family, preserving the actual SSA program, arithmetic,
// constants, access maps, ranks/layouts, dependency and loop structure. This
// clone is a fingerprint view only and is never analyzed or executed.
std::string family(ModuleOp source) {
  OwningOpRef<ModuleOp> normalized=source.clone();
  normalized->getOperation()->removeAttr("orbit.heterogeneous_devices");
  normalized->getOperation()->removeAttr("orbit.heterogeneous_cuts");
  auto type=[](Type t)->Type{if(auto m=dyn_cast<MemRefType>(t))return MemRefType::get(SmallVector<int64_t>(m.getRank(),1),m.getElementType(),m.getLayout(),m.getMemorySpace());return t;};
  normalized->walk([&](Operation *op){
    for(auto value:op->getResults())value.setType(type(value.getType()));
    for(auto &region:op->getRegions())for(auto &block:region)for(auto argument:block.getArguments())argument.setType(type(argument.getType()));
    if(auto loop=dyn_cast<affine::AffineForOp>(op))if(loop.hasConstantLowerBound() && loop.hasConstantUpperBound())loop.setConstantUpperBound(loop.getConstantLowerBound()+loop.getStepAsInt());
    if(auto function=dyn_cast<func::FuncOp>(op)){SmallVector<Type> inputs,outputs;auto signature=function.getFunctionType();for(auto t:signature.getInputs())inputs.push_back(type(t));for(auto t:signature.getResults())outputs.push_back(type(t));function.setFunctionType(FunctionType::get(source.getContext(),inputs,outputs));}
  });
  std::string text;llvm::raw_string_ostream out(text);normalized->print(out);out.flush();
  return staticFingerprint(O{{"normalization","static-shapes-and-loop-upper-bounds-only.v13"},{"source",text}});
}
double work(const GenericAnalysis &source,size_t first,size_t end,bool &supported){
  double total=0;supported=true;
  for(size_t i=first;i<end;++i)source.tasks[i].operation->walk([&](Operation *op){
    if(op->getName().getDialectNamespace()=="math")supported=false;
    if(op->getName().getDialectNamespace()!="arith" || op->getName().getStringRef()=="arith.constant")return;
    if(op->getNumResults()!=1 || !op->getResult(0).getType().isF32())return;
    if(op->getName().getStringRef()!="arith.addf" && op->getName().getStringRef()!="arith.mulf")supported=false;
    double n=1;for(const auto &loop:source.tasks[i].loops)if(loop.operation->isProperAncestor(op))n*=((loop.upper-loop.lower-1)/loop.step+1);
    total+=n;
  });
  return total;
}
std::optional<double> interpolate(const A *knots,double x,llvm::StringRef key){
  if(!knots || knots->size()<2)return {};
  std::vector<std::pair<double,double>> values;
  for(const auto &v:*knots){auto o=v.getAsObject();auto a=o?o->getNumber(key):std::nullopt,b=o?o->getNumber("ms"):std::nullopt;if(!a || !b || !std::isfinite(*a) || !std::isfinite(*b) || *a<0 || *b<0)return {};values.push_back({*a,*b});}
  std::sort(values.begin(),values.end());
  if(x<values.front().first || x>values.back().first)return {};
  for(size_t i=0;i<values.size();++i){if(x==values[i].first)return values[i].second;if(i && x<values[i].first){auto p=values[i-1],q=values[i];if(q.first<=p.first)return {};return p.second+(q.second-p.second)*(x-p.first)/(q.first-p.first);}}
  return {};
}
J price(const J &manifest,const J &config,std::string &error){
  auto cfg=config.getAsObject();auto path=cfg?cfg->getString("heterogeneous_model"):std::nullopt;J model=nullptr;
  O result{{"status","uncalibrated"},{"estimated_ms",nullptr},{"model_kind","historical-program-region-interpolation"},{"analytical",false},{"boundary","sum prepared regional composites plus input/output boundary once; no extra DMA or launch charge"}};
  if(!path)return result;
  if(!read(path->str(),model,error))return result;
  auto m=model.getAsObject();auto rows=m?m->getArray("region_models"):nullptr;
  if(!m || m->getString("schema")!="orbit.heterogeneous.region-cost.v13" || !rows){error="unsupported heterogeneous cost model schema";return result;}
  auto source=manifest.getAsObject();
  if(m->getString("program_family_key")!=source->getString("program_family_key") ||
     m->getString("execution_contract_identity")!=source->getString("execution_contract_identity") ||
     source->getString("execution_contract_identity")=="unbound") {
    result["reason"]="program semantics or target/runtime conditions outside frozen model contract";return result;
  }
  A parts;double sum=0;
  for(const auto &r:*manifest.getAsObject()->getArray("regions")){
    auto region=r.getAsObject();if(!region->getBoolean("model_operations_supported").value_or(false)){result["reason"]="source arithmetic is outside frozen model scope";return result;}
    std::optional<double> value;
    for(const auto &v:*rows){auto row=v.getAsObject();if(row && row->getString("status")=="calibrated" && row->getString("device")==region->getString("device") && row->getInteger("task_count")==int64_t(region->getArray("tasks")->size())){if(value){result["reason"]="duplicate regional cost model";return result;}value=interpolate(row->getArray("knots"),*region->getNumber("work_elements"),"work_elements");}}
    if(!value){result["reason"]="regional feature outside calibrated interpolation domain";return result;}
    sum+=*value;parts.push_back(O{{"region",*region->getInteger("index")},{"ms",*value},{"source","frozen-composite-interpolation"}});
  }
  double bytes=0;for(const auto &v:*manifest.getAsObject()->getArray("storage")){auto s=v.getAsObject();auto role=s->getString("role");if(role=="input" || role=="output")bytes+=*s->getInteger("bytes");}
  auto boundary=m->getObject("boundary_model");auto b=boundary?interpolate(boundary->getArray("knots"),bytes,"bytes"):std::nullopt;
  if(!b){result["reason"]="boundary feature outside calibrated interpolation domain";return result;}
  sum+=*b;if(!std::isfinite(sum) || sum<=0)return result;
  result["status"]="calibrated";result["estimated_ms"]=sum;result["regions"]=std::move(parts);result["boundary_ms"]=*b;result["model_identity"]=staticFingerprint(model);result["source"]="prediction; no hardware submission in evaluation";return result;
}
}
bool orbit::taskflow::readHeterogeneousChoice(ModuleOp module,const J &config,HeterogeneousMappingChoice &choice,std::string &error){
  GenericAnalysis source;if(!analyzeGenericProgram(module,source,error))return false;
  choice={};choice.taskDevices.assign(source.tasks.size(),"cpu");
  auto cfg=config.getAsObject();auto c=cfg?cfg->getObject("heterogeneous"):nullptr;
  if(c)if(auto a=c->getArray("task_devices")){choice.taskDevices.clear();for(const auto &v:*a){auto s=v.getAsString();if(!s)return fail(error,"task_devices must be strings");choice.taskDevices.push_back(s->str());}}
  if(c)if(auto a=c->getArray("cuts"))for(const auto &v:*a){auto n=v.getAsInteger();if(!n || *n<0)return fail(error,"cuts must be nonnegative task indices");choice.cuts.push_back(*n);}
  if(auto a=module->getAttrOfType<ArrayAttr>("orbit.heterogeneous_devices")){choice.taskDevices.clear();for(auto v:a){auto s=dyn_cast<StringAttr>(v);if(!s)return fail(error,"invalid heterogeneous device choice");choice.taskDevices.push_back(s.str());}}
  if(auto a=module->getAttrOfType<DenseI64ArrayAttr>("orbit.heterogeneous_cuts")){choice.cuts.clear();for(auto n:a.asArrayRef()){if(n<0)return fail(error,"negative region cut");choice.cuts.push_back(n);}}
  return true;
}
bool orbit::taskflow::exportHeterogeneousProgram(ModuleOp module,const J &config,const std::string &dir,J &manifest,std::string &error,HeterogeneousAIREmitter emitAIR){
  GenericAnalysis source;HeterogeneousMappingChoice choice;HeterogeneousMappingPlan plan;
  if(!analyzeGenericProgram(module,source,error) || !readHeterogeneousChoice(module,config,choice,error) || !buildHeterogeneousMapping(source,choice,plan,error) || !checkHeterogeneousMapping(source,choice,plan,error))return false;
  if(auto ec=llvm::sys::fs::create_directories(dir))return fail(error,ec.message());
  if(!moduleFile(dir+"/taskflow.mlir",module,error) || !write(dir+"/program-graph.json",serializeProgramGraph(plan.graph),error))return false;
  {std::error_code ec;llvm::raw_fd_ostream out(dir+"/program-graph.dot",ec);if(ec)return fail(error,ec.message());out<<renderProgramGraphDOT(plan.graph);}
  A regions;std::set<size_t> mappedStorage;
  for(size_t index=0;index<plan.slices.size();++index){
    const auto &slice=plan.slices[index];std::string prefix="region-"+std::to_string(index),path=dir+"/"+prefix;
    if(auto ec=llvm::sys::fs::create_directories(path))return fail(error,ec.message());
    OwningOpRef<ModuleOp> child;HeterogeneousRegionProjection projection;
    if(!materializeHeterogeneousRegion(plan,index,child,projection,error) || !moduleFile(path+"/taskflow.mlir",*child,error))return false;
    GenericAnalysis facts;if(!analyzeGenericProgram(*child,facts,error))return false;
    A abi,buffers,outputs;
    for(const auto &arg:projection.arguments){const auto &storage=source.storages[arg.parentStorage];int64_t bytes=4;for(auto n:storage.shape)bytes*=n;
      abi.push_back(O{{"argument",int64_t(arg.childArgument)},{"storage",int64_t(arg.parentStorage)},{"name",storage.id},{"role",arg.role},{"shape",dims(storage.shape)},{"dtype",storage.dtype},{"copy",arg.role=="input"?arg.inputCopy:arg.outputCopy}});
      buffers.push_back(O{{"name",storage.id},{"shape",dims(storage.shape)},{"dtype",storage.dtype},{"role",arg.role}});if(arg.role=="output")outputs.push_back(storage.id);if(slice.device=="npu")mappedStorage.insert(arg.parentStorage);
    }
    bool supported;double expandedWork=work(source,slice.beginTask,slice.endTask,supported);
    O record{{"index",int64_t(index)},{"id",plan.graph.executionRegions[index].id},{"device",slice.device},{"tasks",ints(projection.parentTasks)},
      {"source",prefix+"/taskflow.mlir"},{"function",facts.function.getName().str()},{"abi",J(A(abi))},{"work_elements",expandedWork},{"model_operations_supported",supported}};
    if(slice.device=="cpu"){
      HeterogeneousMappingChoice cpuChoice;cpuChoice.taskDevices.assign(facts.tasks.size(),"cpu");
      HeterogeneousMappingPlan cpuGraph;
      if(!buildHeterogeneousMapping(facts,cpuChoice,cpuGraph,error) ||
         !bindHeterogeneousRegionGraph(plan,cpuGraph.graph,projection,error) ||
         !write(path+"/program-graph.json",serializeProgramGraph(cpuGraph.graph),error))return false;
      OwningOpRef<ModuleOp> cpu;if(!lowerHeterogeneousCPUImplementation(*child,cpu,error) || !moduleFile(path+"/cpu.mlir",*cpu,error))return false;
      record["cpu_source"]=prefix+"/cpu.mlir";
      record["child_graph_id"]=projection.childGraphId;
    }else{
      GenericMappingPlan native;if(!planGenericMapping(facts,J(O{}),native,error))return false;
      if(native.strategy!="single-herd-resident")return fail(error,"heterogeneous NPU region requires resident native implementation; tiled refinement not implemented");
      if(!bindHeterogeneousRegionNativePlan(plan,native,projection,error))return false;
      if(!write(path+"/plan.json",serializeGenericMapping(facts,native),error) || !write(path+"/abi.json",O{{"buffers",std::move(buffers)},{"outputs",std::move(outputs)}},error))return false;
      if(emitAIR && !emitAIR(facts,native,path+"/program.air.mlir",error))return false;
      record["air"]=prefix+"/program.air.mlir";record["npu_abi_path"]=prefix+"/abi.json";record["npu_abi"]=std::move(abi);record["child_graph_id"]=projection.childGraphId;
    }
    if(!write(path+"/projection.json",serializeHeterogeneousRegionProjection(projection),error))return false;
    regions.push_back(std::move(record));
  }
  A storage;for(size_t index=0;index<source.storages.size();++index){const auto &s=source.storages[index];int64_t bytes=4;for(auto n:s.shape)bytes*=n;storage.push_back(O{{"index",int64_t(index)},{"name",s.id},{"shape",dims(s.shape)},{"dtype",s.dtype},{"role",s.role},{"bytes",bytes},{"realization",mappedStorage.count(index)?"xrt-mapped-bo":"numpy-host"}});}
  J graph=serializeProgramGraph(plan.graph);manifest=O{{"schema","orbit.heterogeneous.static.v13"},{"id",plan.planId},{"graph",J(graph)},{"graph_id",programGraphIdentity(plan.graph)},
    {"storage",std::move(storage)},{"regions",std::move(regions)},{"region_events",J(*graph.getAsObject()->get("region_events"))},
    {"numeric_contract",O{{"dtype","f32"},{"atol",1e-5},{"rtol",1e-5},{"cpu_contraction","off"},{"approximate_math",false}}},
    {"memory_contract","CPU directly accesses mapped system BOs used by NPU region ABI; NPU cores access checked child L1 via DMA; sync is distinct from host copying"}};
  auto &manifestObject=*manifest.getAsObject();
  manifestObject["program_family_key"]=family(module);
  J contract=nullptr;
  if(auto object=config.getAsObject())if(auto path=object->getString("execution_contract"))if(!read(path->str(),contract,error))return false;
  manifestObject["execution_contract_identity"]=contract.getAsObject()?staticFingerprint(contract):"unbound";
  manifestObject["execution_contract"]=std::move(contract);
  // Mechanism costs are derived from the checked graph and explicit installed
  // bound-BO realization, not inferred from regional elapsed time or work size.
  MechanismRuntimeContract runtime;
  runtime.identity=manifestObject.getString("execution_contract_identity")->str();
  MechanismFacts mechanismFacts;
  if(!collectMechanismFacts(plan.graph,runtime,mechanismFacts,error))return false;
  std::vector<MechanismParameter> mechanismParameters;
  if(auto object=config.getAsObject())if(auto path=object->getString("mechanism_parameters")){
    J parameters=nullptr;
    if(!read(path->str(),parameters,error) || !parseMechanismParameters(parameters,mechanismParameters,error))return false;
  }
  MechanismCostEvaluation mechanismEvaluation;
  if(!evaluateMechanismCost(mechanismFacts,mechanismParameters,mechanismEvaluation,error))return false;
  manifestObject["mechanism_facts"]=serializeMechanismFacts(mechanismFacts);
  manifestObject["mechanism_cost"]=serializeMechanismCost(mechanismEvaluation);
  if(!write(dir+"/mechanism-facts.json",serializeMechanismFacts(mechanismFacts),error) ||
     !write(dir+"/mechanism-cost.json",serializeMechanismCost(mechanismEvaluation),error))return false;
  J historicalCost=price(manifest,config,error);
  bool analyticalRequested=false;
  if(auto object=config.getAsObject())analyticalRequested=bool(object->getString("mechanism_parameters"));
  if(analyticalRequested){
    // Partial mechanism costs must never fall back to a complete empirical
    // regional score. Preserve that optional historical estimate separately.
    manifestObject["historical_empirical_cost"]=std::move(historicalCost);
    manifestObject["cost"]=O{{"status",mechanismEvaluation.totalMs?"calibrated":"uncalibrated"},
      {"estimated_ms",mechanismEvaluation.totalMs?J(*mechanismEvaluation.totalMs):J(nullptr)},
      {"known_subtotal_ms",mechanismEvaluation.knownSubtotalMs},
      {"model_kind","analytical-api-mechanisms"},{"analytical",true},
      {"reason","opaque compute/wait/runtime items remain unknown; partial subtotal is not a whole-program search score"}};
  }else manifestObject["cost"]=std::move(historicalCost);
  int64_t cpuCalls=0,npuCalls=0;for(const auto &slice:plan.slices)(slice.device=="cpu"?cpuCalls:npuCalls)++;
  (*manifest.getAsObject())["runtime_counts"]=O{{"cpu_calls",cpuCalls},{"npu_submissions",npuCalls},{"npu_device_stages",npuCalls},{"scope","one checked resident device stage per NPU region; input count does not multiply submissions"}};
  (*manifest.getAsObject())["target_memory_binding"]=O{{"system.shared","host-addressable system memory; exposed NPU ABI allocations use mapped XRT BOs, CPU-private temporaries remain compiled host allocations"},{"region-local-views","per-region allocation scopes of one serially reused NPU local tensor payload budget; not separate physical banks"},{"local_payload_budget_bytes",24576},{"physical_bank_capacity","not inferred; final AIE allocation/link checks authoritative"}};
  (*manifest.getAsObject())["implementation_capabilities"]=O{
    {"placements",A{"cpu","npu"}},{"region_actions",A{"merge-adjacent-same-device","split-region","place-task"}},
    {"npu_strategy","single-herd-resident"},{"precision","strict-f32; no approximate exp"},
    {"unsupported",A{"inout-alias-region-ABI","tiled-native-region-refinement","asynchronous-region-overlap"}}};
  if(!write(dir+"/manifest.json",manifest,error) || !write(dir+"/mapping.json",serializeHeterogeneousMapping(plan),error))return false;
  return true;
}
bool orbit::taskflow::configureHeterogeneousSearch(ModuleOp source,SearchOptions &search,std::string &error){
  GenericAnalysis analysis;if(!analyzeGenericProgram(source,analysis,error))return false;
  J configuration=O{};
  if(!search.targetConfigFile.empty() && !read(search.targetConfigFile,configuration,error))return false;
  HeterogeneousMappingChoice initial;
  if(!readHeterogeneousChoice(source,configuration,initial,error))return false;
  Builder initialBuilder(source.getContext());SmallVector<Attribute> initialDevices;
  for(auto &device:initial.taskDevices)initialDevices.push_back(initialBuilder.getStringAttr(device));
  source->setAttr("orbit.heterogeneous_devices",initialBuilder.getArrayAttr(initialDevices));
  SmallVector<int64_t> initialCuts;for(auto cut:initial.cuts)initialCuts.push_back(cut);
  source->setAttr("orbit.heterogeneous_cuts",initialBuilder.getDenseI64ArrayAttr(initialCuts));
  llvm::raw_string_ostream text(search.nativeSourceMLIR);source.print(text);text.flush();
  search.nativeBackendIdentity="orbit.cpu-npu.program-graph.v13.resident-f32.1";
  auto *context=source.getContext();
  search.nativeEvaluator=[context](llvm::StringRef text,const J &config,llvm::StringRef dir,std::string &error)->J{
    auto module=parseSourceString<ModuleOp>(text,context);J manifest=nullptr;
    if(!module || !exportHeterogeneousProgram(*module,config,dir.str(),manifest,error))return O{{"status","unsupported"},{"reason",error}};
    auto &m=*manifest.getAsObject();const auto &cost=*m.getObject("cost");
    return O{{"status","passed"},{"program_graph",J(*m.get("graph"))},{"program_graph_id",m.getString("graph_id")->str()},
      {"program_graph_dot",[&](){GenericAnalysis a;HeterogeneousMappingChoice c;HeterogeneousMappingPlan p;analyzeGenericProgram(*module,a,error);readHeterogeneousChoice(*module,config,c,error);buildHeterogeneousMapping(a,c,p,error);return renderProgramGraphDOT(p.graph);}()},
      {"best_plan",O{{"plan_id",m.getString("id")->str()},{"estimated_ms",J(*cost.get("estimated_ms"))},{"cost_status",cost.getString("status")->str()},{"cost_breakdown",J(O(cost))}}}};
  };
  search.nativeApplyAction=[](ModuleOp module,const SearchAction &action,std::string &error){
    GenericAnalysis source;if(!analyzeGenericProgram(module,source,error))return false;HeterogeneousMappingChoice choice;if(!readHeterogeneousChoice(module,J(O{}),choice,error))return false;
    if(action.backendAction=="placement") {if(action.backendParameters.size()!=2)return fail(error,"invalid placement action");auto task=action.backendParameters[0],device=action.backendParameters[1];if(task<0)choice.taskDevices.assign(source.tasks.size(),device?"npu":"cpu");else if(size_t(task)<choice.taskDevices.size())choice.taskDevices[task]=device?"npu":"cpu";else return fail(error,"placement task outside source");}
    else if(action.backendAction=="boundary") {size_t cut=action.backendParameters[0];auto it=std::find(choice.cuts.begin(),choice.cuts.end(),cut);if(it==choice.cuts.end())choice.cuts.push_back(cut);else choice.cuts.erase(it);std::sort(choice.cuts.begin(),choice.cuts.end());}
    else return fail(error,"unsupported heterogeneous source action");
    choice.cuts.erase(std::remove_if(choice.cuts.begin(),choice.cuts.end(),[&](size_t cut){return cut<choice.taskDevices.size() && cut>0 && choice.taskDevices[cut-1]!=choice.taskDevices[cut];}),choice.cuts.end());
    SmallVector<Attribute> devices;Builder b(module.getContext());for(auto &d:choice.taskDevices)devices.push_back(b.getStringAttr(d));module->setAttr("orbit.heterogeneous_devices",b.getArrayAttr(devices));SmallVector<int64_t> cuts;for(auto c:choice.cuts)cuts.push_back(c);module->setAttr("orbit.heterogeneous_cuts",b.getDenseI64ArrayAttr(cuts));return true;
  };
  for(int64_t task=-1;task<int64_t(analysis.tasks.size());++task)for(int64_t device=0;device<2;++device){SearchAction a;a.name="place-"+(task<0?std::string("program"):"task-"+std::to_string(task))+(device?"-npu":"-cpu");a.backendAction="placement";a.backendParameters={task,device};search.nativeActions.push_back(a);}
  for(size_t cut=1;cut<analysis.tasks.size();++cut){SearchAction a;a.name="toggle-region-boundary-"+std::to_string(cut);a.backendAction="boundary";a.backendParameters={int64_t(cut)};search.nativeActions.push_back(a);}
  return true;
}
