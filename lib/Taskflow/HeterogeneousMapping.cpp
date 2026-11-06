#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/GenericMapping.h"
#include <tuple>
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include <algorithm>
#include <map>
#include <set>

using namespace orbit::taskflow;
namespace {
bool fail(std::string &error,const std::string &message) {
  error="heterogeneous mapping: "+message;return false;
}
llvm::json::Array indices(const std::vector<size_t> &values) {
  llvm::json::Array result;for(auto value:values)result.push_back(int64_t(value));return result;
}
llvm::json::Value choiceJSON(const HeterogeneousMappingChoice &choice) {
  llvm::json::Array devices;for(const auto &device:choice.taskDevices)devices.push_back(device);
  return llvm::json::Object{{"task_devices",std::move(devices)},{"cuts",indices(choice.cuts)}};
}
bool build(const GenericAnalysis &source,const HeterogeneousMappingChoice &requested,
           HeterogeneousMappingPlan &plan,std::string &error) {
  plan={};plan.choice=requested;plan.sourceFingerprint=source.bodyFingerprint;
  if(source.tasks.empty() || requested.taskDevices.size()!=source.tasks.size())
    return fail(error,"one CPU/NPU device choice per actual source task required");
  for(const auto &device:requested.taskDevices)if(device!="cpu" && device!="npu")
    return fail(error,"device choice must be cpu or npu");
  std::set<size_t> cuts;
  for(auto cut:requested.cuts)if(cut==0 || cut>=source.tasks.size() || !cuts.insert(cut).second)
    return fail(error,"cuts must be unique boundaries strictly inside the task sequence");
  plan.choice.cuts.assign(cuts.begin(),cuts.end());
  for(const auto &storage:source.storages) {
    auto type=mlir::dyn_cast<mlir::MemRefType>(storage.root.getType());
    if(!type || !type.getElementType().isF32() || !type.hasStaticShape() ||
       !type.getLayout().isIdentity() || type.getMemorySpaceAsInt()!=0)
      return fail(error,"initial heterogeneous ABI requires static identity-layout host f32 memrefs");
  }
  std::map<size_t,size_t> writer;
  for(size_t t=0;t<source.tasks.size();++t) {
    const auto &task=source.tasks[t];std::set<size_t> accessed,written;
    for(const auto &access:task.accesses) {accessed.insert(access.storage);if(access.write)written.insert(access.storage);}
    for(auto storage:written)if(!writer.emplace(storage,t).second)
      return fail(error,"initial resident region adapter requires one source writer task per storage");
    for(auto storage:task.argumentStorages)if(!accessed.count(storage))
      return fail(error,"unused memory-state arguments need an explicit order-only ABI projection");
    for(auto result:task.operation->getResults())if(!mlir::isa<mlir::MemRefType>(result.getType()))
      return fail(error,"scalar values crossing task interfaces are outside the initial region ABI");
  }
  std::vector<size_t> taskRegion(source.tasks.size());
  for(size_t begin=0;begin<source.tasks.size();) {
    size_t end=begin+1;
    while(end<source.tasks.size() && requested.taskDevices[end]==requested.taskDevices[begin] && !cuts.count(end))++end;
    HeterogeneousExecutionSlice slice;slice.beginTask=begin;slice.endTask=end;slice.device=requested.taskDevices[begin];
    for(size_t t=begin;t<end;++t)taskRegion[t]=plan.slices.size();
    plan.slices.push_back(std::move(slice));begin=end;
  }
  auto local=[](size_t region){return "region."+std::to_string(region)+".local";};
  StaticConfig target;
  target.domains={{"host","","system"},{"npu","","device"}};
  target.resources={{"CPU.compute","compute","host",1,"host"},
                    {"NPU.hostcall","compute","host",1,"host"},
                    {"NPU.compute","compute","npu",1,"npu"},
                    {"NPU.dma","transfer","npu",1,"npu"},
                    {"CPU.submit","submission","host",1,"host"},
                    {"NPU.submit","submission","host",1,"host"}};
  StaticMemory shared;
  shared.id="system.shared";shared.capacityBytes=INT64_MAX;shared.capacityKnown=false;
  shared.domain="host";shared.kind="system-bo";
  shared.accessibleResources={"CPU.compute","NPU.hostcall"};target.memories={shared};
  for(size_t region=0;region<plan.slices.size();++region)if(plan.slices[region].device=="npu") {
    StaticMemory memory;memory.id=local(region);memory.capacityBytes=24576;
    memory.domain="npu";memory.kind="local-sram";memory.accessibleResources={"NPU.compute"};
    target.memories.push_back(memory);
    for(bool input:{true,false}) {
      StaticTransfer link;link.id=(input?"import.":"export.")+std::to_string(region);
      link.source=input?"system.shared":local(region);link.target=input?local(region):"system.shared";
      link.resource="NPU.dma";link.domain="npu";
      link.provenance={"target_contract","host BO/local tensor payload; timing uncalibrated; resident child graph refines allocation lifetime"};
      target.links.push_back(std::move(link));
    }
  }
  ProgramDataOrganizationChoice choice;choice.policy=ProgramDataPolicy::RetainAcrossUsers;
  for(size_t task=0;task<source.tasks.size();++task) {
    bool cpu=requested.taskDevices[task]=="cpu";
    choice.placements.push_back({task,cpu?"CPU.compute":"NPU.compute",cpu?"system.shared":local(taskRegion[task]),
                                cpu?"llvm-ciface-f32":"direct-air-hostcall"});
  }
  for(const auto &storage:source.storages) {
    if(storage.role=="input")target.assumptions.initialLocations[storage.id]="system.shared";
    if(storage.role=="output")target.assumptions.outputLocations[storage.id]="system.shared";
  }
  if(!planProgramDataOrganization(source,target,choice,plan.graph,error))return false;
  auto &graph=plan.graph;graph.executionRegions.clear();graph.regionEvents.clear();
  std::vector<size_t> eventRegion(graph.events.size()),computations(source.tasks.size(),SIZE_MAX);
  for(size_t index=0;index<graph.events.size();++index) {
    const auto &event=graph.events[index];
    if(event.kind=="compute") {eventRegion[index]=taskRegion[event.task];computations[event.task]=index;}
    else if(event.kind=="transfer") {
      const auto &from=graph.allocations[graph.copies[event.reads.front()].allocation].memory;
      const auto &to=graph.allocations[graph.copies[event.writes.front()].allocation].memory;
      size_t owner=SIZE_MAX;
      for(size_t region=0;region<plan.slices.size();++region)
        if(plan.slices[region].device=="npu" && (from==local(region) || to==local(region)))owner=region;
      if(owner==SIZE_MAX)return fail(error,"transfer has no concrete NPU region owner");
      eventRegion[index]=owner;
    } else return fail(error,"unsupported parent event kind");
  }
  // Replace the whole planner's incidental event serialization. A region may
  // own an export emitted immediately before a later task's import/compute.
  for(size_t index=0;index<graph.events.size();++index) {
    auto &event=graph.events[index];event.dependencies.clear();
    for(auto copy:event.reads)if(graph.copies[copy].createdBy>=0)
      event.dependencies.push_back(size_t(graph.copies[copy].createdBy));
    for(auto edge:event.orderEdges)event.dependencies.push_back(graph.orderEdges[edge].producer);
    if(event.kind=="compute")for(auto parent:source.tasks[event.task].dependencies)
      event.dependencies.push_back(computations[parent]);
    std::sort(event.dependencies.begin(),event.dependencies.end());
    event.dependencies.erase(std::unique(event.dependencies.begin(),event.dependencies.end()),event.dependencies.end());
    for(auto parent:event.dependencies)if(eventRegion[parent]>eventRegion[index])
      return fail(error,"parent dependency cannot realize the chosen serial region order");
  }
  for(size_t number=0;number<plan.slices.size();++number) {
    auto &slice=plan.slices[number];MappingExecutionRegion region;
    region.id="heterogeneous-region-"+std::to_string(number);region.placement=slice.device;
    bool cpu=slice.device=="cpu";
    region.implementation=cpu?"llvm-ciface-f32":"direct-air-hostcall";
    region.endpointMemory="system.shared";region.submissionCount=1;
    region.submissionResource=cpu?"CPU.submit":"NPU.submit";
    if(number)region.predecessorRegions.push_back(number-1);
    region.beginEvent=graph.events.size();region.endEvent=0;
    for(size_t event=0;event<graph.events.size();++event)if(eventRegion[event]==number) {
      region.memberEvents.push_back(event);region.beginEvent=std::min(region.beginEvent,event);region.endEvent=std::max(region.endEvent,event+1);
    }
    std::set<size_t> inputCopies,outputCopies,inputs,outputs,written;
    for(size_t task=slice.beginTask;task<slice.endTask;++task) {
      region.sourceTasks.push_back(task);const auto &event=graph.events[computations[task]];
      for(auto copyId:event.reads) {
        const auto &copy=graph.copies[copyId];const auto &version=graph.versions[copy.version];
        if(version.producer<0 || taskRegion[graph.events[size_t(version.producer)].task]!=number) {
          size_t hostCopy=copyId;
          if(!cpu) {
            if(copy.createdBy<0 || graph.events[size_t(copy.createdBy)].kind!="transfer")return fail(error,"NPU input has no checked host import");
            hostCopy=graph.events[size_t(copy.createdBy)].reads.front();
          }
          if(graph.allocations[graph.copies[hostCopy].allocation].memory!="system.shared")return fail(error,"region input ABI lacks host-visible copy");
          inputCopies.insert(hostCopy);inputs.insert(graph.regions[version.region].storage);
        }
      }
      for(auto copyId:event.writes) {
        const auto &copy=graph.copies[copyId];const auto &version=graph.versions[copy.version];
        size_t storage=graph.regions[version.region].storage;written.insert(storage);
        bool live=source.storages[storage].role=="output";
        for(auto user:copy.users)live|=eventRegion[user]!=number;
        // Exports are owned by this region, while their host-copy users belong
        // to later regions. Their existence also proves this is a live-out.
        size_t hostCopy=cpu?copyId:SIZE_MAX;
        for(auto user:copy.users)if(graph.events[user].kind=="transfer") {
          auto exported=graph.events[user].writes.front();
          if(graph.allocations[graph.copies[exported].allocation].memory=="system.shared") {hostCopy=exported;live=true;}
        }
        if(live) {
          if(hostCopy==SIZE_MAX)return fail(error,"NPU output ABI has no checked host export");
          outputCopies.insert(hostCopy);outputs.insert(storage);
        }
      }
    }
    for(auto storage:inputs)if(written.count(storage))
      return fail(error,"region has live-in/in-place writes; distinct noalias input/output ABI required");
    slice.inputStorages.assign(inputs.begin(),inputs.end());slice.outputStorages.assign(outputs.begin(),outputs.end());
    for(auto storage:written)if(!outputs.count(storage))slice.internalStorages.push_back(storage);
    region.inputCopies.assign(inputCopies.begin(),inputCopies.end());region.outputCopies.assign(outputCopies.begin(),outputCopies.end());
    graph.executionRegions.push_back(region);
    auto lifecycle=[&](const std::string &kind,const std::string &resource,const std::vector<size_t> &copies) {
      MappingRegionEvent event;event.kind=kind;event.region=number;event.resource=resource;event.copies=copies;
      if(!graph.regionEvents.empty())event.dependencies.push_back(graph.regionEvents.size()-1);
      graph.regionEvents.push_back(std::move(event));
    };
    if(!cpu)lifecycle("visible-to-device","NPU.hostcall",region.inputCopies);
    lifecycle("submit",region.submissionResource,{});
    lifecycle("complete",region.submissionResource,{});
    if(!cpu)lifecycle("visible-to-host","NPU.hostcall",region.outputCopies);
  }
  graph.targetIdentity=staticFingerprint(llvm::json::Object{
      {"kind","heterogeneous-physical-resident-v1"},
      {"memory",serializeMemoryArchitecture(target)},
      {"local_memory_scope","resident checked child graph refines full-call lifetimes"},
      {"cost_scope","prepared region composite owns visibility/submit/wait and child movement once"}});
  graph.schedule.policy="dependency-order-witness";graph.estimatedMakespanMs.reset();
  if(!finalizeProgramGraph(source,graph,nullptr,error) || !checkProgramGraph(graph,error))return false;
  plan.planId=staticFingerprint(llvm::json::Object{{"graph",programGraphIdentity(graph)},
                                               {"choice",choiceJSON(plan.choice)}});
  return true;
}
} // namespace

bool orbit::taskflow::buildHeterogeneousMapping(const GenericAnalysis &source,
    const HeterogeneousMappingChoice &choice,HeterogeneousMappingPlan &plan,std::string &error) {
  return build(source,choice,plan,error);
}
bool orbit::taskflow::checkHeterogeneousMapping(const GenericAnalysis &source,
    const HeterogeneousMappingChoice &choice,const HeterogeneousMappingPlan &plan,std::string &error) {
  if(!checkProgramGraphSource(source,plan.graph,error) || !checkProgramGraph(plan.graph,error))return false;
  HeterogeneousMappingPlan expected;if(!build(source,choice,expected,error))return false;
  if(staticFingerprint(serializeHeterogeneousMapping(plan))!=staticFingerprint(serializeHeterogeneousMapping(expected)))
    return fail(error,"parent graph, regions, source or choice differs from checked realization");
  return true;
}

bool orbit::taskflow::materializeHeterogeneousRegion(const HeterogeneousMappingPlan &plan,size_t number,
    mlir::OwningOpRef<mlir::ModuleOp> &output,HeterogeneousRegionProjection &projection,std::string &error) {
  output={};projection={};
  const auto &source=plan.graph.source;
  if(!checkHeterogeneousMapping(source,plan.choice,plan,error) || number>=plan.slices.size())
    return fail(error,"invalid parent plan or region index: "+error);
  const auto &slice=plan.slices[number];const auto &region=plan.graph.executionRegions[number];
  auto parentFunction=source.function;
  auto *context=parentFunction.getContext();mlir::OpBuilder builder(context);
  output=mlir::ModuleOp::create(parentFunction.getLoc());
  auto parent=source.function->getParentOfType<mlir::ModuleOp>();output->getOperation()->setAttrs(parent->getAttrs());
  projection.region=number;projection.parentGraphId=programGraphIdentity(plan.graph);
  projection.parentSourceFingerprint=source.bodyFingerprint;projection.parentTasks=region.sourceTasks;
  std::vector<size_t> arguments=slice.inputStorages;
  arguments.insert(arguments.end(),slice.outputStorages.begin(),slice.outputStorages.end());
  llvm::SmallVector<mlir::Type> types;
  for(auto storage:arguments)types.push_back(source.storages[storage].root.getType());
  auto function=mlir::func::FuncOp::create(parentFunction.getLoc(),"heterogeneous_region_"+std::to_string(number),
      builder.getFunctionType(types,{}));
  function->setAttr("orbit.noalias",builder.getUnitAttr());function->setAttr("llvm.emit_c_interface",builder.getUnitAttr());
  output->push_back(function);auto *block=function.addEntryBlock();builder.setInsertionPointToStart(block);
  mlir::IRMapping mapping;std::map<size_t,mlir::Value> roots;
  for(size_t argument=0;argument<arguments.size();++argument) {
    size_t storage=arguments[argument];std::string role=argument<slice.inputStorages.size()?"input":"output";
    function.setArgAttr(argument,"orbit.role",builder.getStringAttr(role));
    function.setArgAttr(argument,"orbit.name",builder.getStringAttr(source.storages[storage].id));
    roots[storage]=block->getArgument(argument);
    HeterogeneousRegionArgument binding;binding.parentStorage=storage;binding.childArgument=argument;binding.role=role;
    const auto &copies=role=="input"?region.inputCopies:region.outputCopies;
    for(auto copy:copies)if(plan.graph.regions[plan.graph.versions[plan.graph.copies[copy].version].region].storage==storage) {
      if(role=="input")binding.inputCopy=copy;else binding.outputCopy=copy;
    }
    projection.arguments.push_back(binding);
  }
  for(auto storage:slice.internalStorages) {
    auto allocation=builder.create<mlir::memref::AllocOp>(function.getLoc(),mlir::cast<mlir::MemRefType>(source.storages[storage].root.getType()));
    allocation->setAttr("orbit.name",builder.getStringAttr(source.storages[storage].id));roots[storage]=allocation.getResult();
  }
  for(const auto &entry:roots)mapping.map(source.storages[entry.first].root,entry.second);
  // External task state is only an interface reference here; the parent graph
  // and region lifecycle preserve its execution dependency. Selected task
  // result mappings are replaced by the real cloned results during cloning.
  for(size_t taskIndex=0;taskIndex<source.tasks.size();++taskIndex) {
    if(taskIndex>=slice.beginTask && taskIndex<slice.endTask)continue;
    const auto &task=source.tasks[taskIndex];
    auto &body=task.operation->getRegion(0).front();auto *yield=body.getTerminator();
    for(size_t result=0;result<task.operation->getNumResults();++result) {
      auto argument=mlir::dyn_cast<mlir::BlockArgument>(yield->getOperand(result));
      if(!argument || argument.getOwner()!=&body || argument.getArgNumber()>=task.argumentStorages.size())
        return fail(error,"task result is not a direct memory-state body argument");
      size_t storage=task.argumentStorages[argument.getArgNumber()];
      if(roots.count(storage))mapping.map(task.operation->getResult(result),roots.at(storage));
    }
  }
  for(auto &operation:parentFunction.getBody().front())if(mlir::isa<mlir::arith::ConstantOp>(operation))builder.clone(operation,mapping);
  for(size_t task=slice.beginTask;task<slice.endTask;++task) {
    for(auto operand:source.tasks[task].operation->getOperands())if(!mapping.contains(operand))
      return fail(error,"region extraction has an unsupported captured scalar/state operand");
    builder.clone(*source.tasks[task].operation,mapping);
  }
  for(auto storage:slice.internalStorages)builder.create<mlir::memref::DeallocOp>(function.getLoc(),roots.at(storage));
  builder.create<mlir::func::ReturnOp>(function.getLoc());
  if(mlir::failed(mlir::verify(*output)))return fail(error,"actual region Taskflow clone is invalid");
  GenericAnalysis child;if(!analyzeGenericProgram(*output,child,error))return false;
  projection.childSourceFingerprint=child.bodyFingerprint;
  for(auto &argument:projection.arguments) {
    auto found=std::find_if(child.storages.begin(),child.storages.end(),[&](const auto &storage){return storage.root==function.getArgument(argument.childArgument);});
    if(found==child.storages.end())return fail(error,"child argument has no analyzed logical storage");
    argument.childStorage=found-child.storages.begin();
  }
  return true;
}

static bool bindProjection(const HeterogeneousMappingPlan &plan,
    const ProgramGraph &child,HeterogeneousRegionProjection &projection,std::string &error) {
  mlir::OwningOpRef<mlir::ModuleOp> expectedModule;HeterogeneousRegionProjection expected;
  if(!materializeHeterogeneousRegion(plan,projection.region,expectedModule,expected,error) ||
     !checkProgramGraphSource(child.source,child,error) || !checkProgramGraph(child,error))return false;
  std::string old=projection.childGraphId;projection.childGraphId.clear();
  bool match=staticFingerprint(serializeHeterogeneousRegionProjection(projection))==
             staticFingerprint(serializeHeterogeneousRegionProjection(expected));
  projection.childGraphId=old;
  if(!match || child.source.bodyFingerprint!=expected.childSourceFingerprint || child.source.tasks.size()!=expected.parentTasks.size())
    return fail(error,"child graph source or parent-copy/ABI projection changed");
  for(const auto &argument:projection.arguments) {
    if(argument.childStorage>=child.source.storages.size())return fail(error,"child storage mapping outside graph");
    const auto &storage=child.source.storages[argument.childStorage];const auto &parent=plan.graph.source.storages[argument.parentStorage];
    if(storage.role!=argument.role || storage.dtype!=parent.dtype || storage.shape!=parent.shape)
      return fail(error,"child ABI source geometry/role differs from parent");
    const auto &locations=argument.role=="input"?child.inputLocations:child.outputLocations;
    if(!locations.count(argument.childStorage))return fail(error,"child graph lacks declared ABI boundary availability");
  }
  auto identity=programGraphIdentity(child);
  if(!old.empty() && old!=identity)return fail(error,"child graph identity changed after binding");
  projection.childGraphId=identity;return true;
}

bool orbit::taskflow::bindHeterogeneousRegionGraph(const HeterogeneousMappingPlan &plan,
    const ProgramGraph &child,HeterogeneousRegionProjection &projection,std::string &error) {
  if(projection.region>=plan.slices.size() || plan.slices[projection.region].device!="cpu")
    return fail(error,"NPU region requires the checked resident native-plan binding API");
  return bindProjection(plan,child,projection,error);
}
bool orbit::taskflow::bindHeterogeneousRegionNativePlan(const HeterogeneousMappingPlan &plan,
    const GenericMappingPlan &native,HeterogeneousRegionProjection &projection,std::string &error) {
  if(projection.region>=plan.slices.size() || plan.slices[projection.region].device!="npu")
    return fail(error,"native child binding requires an NPU execution region");
  if(native.strategy!="single-herd-resident")
    return fail(error,"initial heterogeneous NPU region supports resident native plans only; tiled refinement is unsupported");
  const auto &child=native.communication;
  if(!checkGenericMapping(child.source,native,error))return false;
  auto candidate=projection;
  if(!bindProjection(plan,child,candidate,error))return false;
  std::map<size_t,size_t> childToParent;
  for(size_t storage=0;storage<child.source.storages.size();++storage) {
    auto found=std::find_if(plan.graph.source.storages.begin(),plan.graph.source.storages.end(),
        [&](const auto &parent){return parent.id==child.source.storages[storage].id;});
    if(found==plan.graph.source.storages.end())return fail(error,"native child has an unbound logical storage");
    childToParent[storage]=found-plan.graph.source.storages.begin();
  }
  std::string local="region."+std::to_string(projection.region)+".local";
  std::map<size_t,int64_t> parentAllocations,childAllocations;
  for(const auto &allocation:plan.graph.allocations)if(allocation.memory==local)
    if(!parentAllocations.emplace(allocation.storage,allocation.bytes).second)
      return fail(error,"parent resident region has duplicate storage allocations");
  int64_t residentBytes=0;
  for(const auto &allocation:child.allocations)if(allocation.memory==native.memoryBinding.localMemory) {
    if(!childToParent.count(allocation.storage) ||
       !childAllocations.emplace(childToParent.at(allocation.storage),allocation.bytes).second ||
       allocation.bytes<=0 || residentBytes>24576-allocation.bytes)
      return fail(error,"child resident allocation union exceeds payload capacity or is not unique");
    residentBytes+=allocation.bytes;
  }
  if(parentAllocations!=childAllocations)return fail(error,"native child allocations differ from physical parent storage copies");
  using Transfer=std::tuple<size_t,std::string,int64_t>;
  std::map<Transfer,size_t> parentTransfers,childTransfers;
  const auto &region=plan.graph.executionRegions[projection.region];
  for(auto id:region.memberEvents) {
    const auto &event=plan.graph.events[id];if(event.kind!="transfer")continue;
    const auto &from=plan.graph.allocations[plan.graph.copies[event.reads.front()].allocation];
    const auto &to=plan.graph.allocations[plan.graph.copies[event.writes.front()].allocation];
    std::string direction;
    if(from.memory=="system.shared" && to.memory==local)direction="input";
    else if(from.memory==local && to.memory=="system.shared")direction="output";
    else return fail(error,"parent region DMA endpoint is not its host/local pair");
    ++parentTransfers[{from.storage,direction,event.transferBytes}];
  }
  for(const auto &event:child.events)if(event.kind=="transfer") {
    const auto &from=child.allocations[child.copies[event.reads.front()].allocation];
    const auto &to=child.allocations[child.copies[event.writes.front()].allocation];
    std::string direction;
    if(from.memory==native.memoryBinding.hostMemory && to.memory==native.memoryBinding.localMemory)direction="input";
    else if(from.memory==native.memoryBinding.localMemory && to.memory==native.memoryBinding.hostMemory)direction="output";
    else return fail(error,"child DMA endpoint differs from checked host/local binding");
    ++childTransfers[{childToParent.at(from.storage),direction,event.transferBytes}];
  }
  if(parentTransfers!=childTransfers)return fail(error,"native child DMA geometry/count differs from physical parent region");
  projection=std::move(candidate);return true;
}

llvm::json::Value orbit::taskflow::serializeHeterogeneousRegionProjection(const HeterogeneousRegionProjection &p) {
  llvm::json::Array arguments;
  for(const auto &argument:p.arguments)arguments.push_back(llvm::json::Object{
      {"parent_storage",int64_t(argument.parentStorage)},{"child_argument",int64_t(argument.childArgument)},
      {"child_storage",int64_t(argument.childStorage)},{"role",argument.role},
      {"input_copy",argument.inputCopy},{"output_copy",argument.outputCopy}});
  return llvm::json::Object{{"region",int64_t(p.region)},{"parent_graph_id",p.parentGraphId},
      {"parent_source_fingerprint",p.parentSourceFingerprint},{"child_source_fingerprint",p.childSourceFingerprint},
      {"child_graph_id",p.childGraphId},{"child_tasks_to_parent",indices(p.parentTasks)},
      {"arguments",std::move(arguments)},{"local_memory_authority","checked implementation child graph"},
      {"cost_ownership","prepared region composite includes child transfers, visibility and submit/wait once"}};
}
llvm::json::Value orbit::taskflow::serializeHeterogeneousMapping(const HeterogeneousMappingPlan &plan) {
  llvm::json::Array slices;
  for(const auto &slice:plan.slices)slices.push_back(llvm::json::Object{
      {"begin_task",int64_t(slice.beginTask)},{"end_task",int64_t(slice.endTask)},{"device",slice.device},
      {"input_storages",indices(slice.inputStorages)},{"output_storages",indices(slice.outputStorages)},
      {"internal_storages",indices(slice.internalStorages)}});
  return llvm::json::Object{{"schema","orbit.heterogeneous-mapping.v1"},{"plan_id",plan.planId},
      {"source_fingerprint",plan.sourceFingerprint},{"choice",choiceJSON(plan.choice)},
      {"program_graph",serializeProgramGraph(plan.graph)},{"slices",std::move(slices)},
      {"parent_memory_scope","physical host BO and per-region local tensor copies; serial regions reuse the target"},
      {"native_memory_scope","separately checked region child graph owns L1/DMA"},
      {"cost_scope","region composite; no duplicate parent charge for child movement"}};
}
