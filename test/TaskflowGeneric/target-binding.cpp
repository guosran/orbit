#include "Taskflow/ProgramTargetBinding.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/ProgramTiming.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
using namespace orbit::taskflow;
int main(){
 unsigned checks=0;bool ok=true;std::string error;
 auto require=[&](bool v,const char *s){if(v)++checks;else{ok=false;llvm::errs()<<s<<": "<<error<<"\n";}error.clear();};
 mlir::DialectRegistry registry;registry.insert<mlir::taskflow::TaskflowDialect,mlir::affine::AffineDialect,mlir::arith::ArithDialect,mlir::memref::MemRefDialect,mlir::func::FuncDialect>();mlir::MLIRContext context(registry);
 const char *source=R"mlir(module {func.func @copy(%a:memref<1x16xf32> {orbit.name="A",orbit.role="input"},%b:memref<1x16xf32> {orbit.name="B",orbit.role="output"}) attributes {orbit.noalias} {
 %r=taskflow.task @copy will_reads(%a:memref<1x16xf32>) will_writes(%b:memref<1x16xf32>) [original_read_memrefs(%a:memref<1x16xf32>),original_write_memrefs(%b:memref<1x16xf32>)] : (memref<1x16xf32>,memref<1x16xf32>)->memref<1x16xf32> {
 ^bb0(%x:memref<1x16xf32>,%y:memref<1x16xf32>): affine.for %i=0 to 1 {affine.for %j=0 to 16 {%v=affine.load %x[%i,%j]:memref<1x16xf32> affine.store %v,%y[%i,%j]:memref<1x16xf32>}} taskflow.yield done_writes(%y:memref<1x16xf32>)} return }})mlir";
 auto module=mlir::parseSourceString<mlir::ModuleOp>(source,&context);GenericAnalysis analysis;if(!module||!analyzeGenericProgram(*module,analysis,error)){llvm::errs()<<error;return 1;}
 for(bool shared:{false,true}){
 std::string p=shared?"fabric.":"island/";StaticConfig c;
 c.domains={{p+"root","","system"},{p+"bank",p+"root","cluster"},{p+"engine",p+"bank","processor",{{"socket",7},{"unit",3}}}};
 c.resources={{p+"alu","compute","arbitrary-device",1,p+"engine"},{p+"move","transfer","arbitrary-device",1,p+"bank"},{p+"submit","submission","host",1,p+"root"}};
 c.memories={{p+"external",4096,{"host"}},{p+"scratch",1024,{"arbitrary-device"}}};c.memories[0].domain=p+"root";c.memories[1].domain=shared?p+"bank":p+"engine";c.memories[1].accessibleResources={p+"alu"};
 StaticTransfer in;in.id=p+"ingress";in.source=c.memories[0].id;in.target=c.memories[1].id;in.resource=c.resources[1].id;c.links.push_back(in);std::swap(in.source,in.target);in.id=p+"egress";c.links.push_back(in);
 ProgramTargetBindingView binding{c.memories[0].id,c.memories[1].id,c.resources[0].id,c.resources[1].id,c.links[0].id,c.links[1].id,c.resources[2].id};
 require(checkProgramTargetBinding(c,binding,1024,error),"target-provided arbitrary identifiers and owners");
 StaticConfig roundtrip;require(parseMemoryArchitecture(serializeMemoryArchitecture(c),roundtrip,error)&&serializeMemoryArchitecture(roundtrip)==serializeMemoryArchitecture(c),"coordinates and ownership roundtrip");
 require(programDomainContains(c,p+"bank",p+"engine")&&!programDomainContains(c,"fabric",p+"engine"),"owner ancestry uses references not prefix spelling");
 c.assumptions.initialLocations={{"A",c.memories[0].id}};c.assumptions.outputLocations={{"B",c.memories[0].id}};ProgramDataOrganizationChoice choice;choice.semanticDependenciesOnly=true;choice.placements={{0,c.resources[0].id,c.memories[1].id,"scalar"}};ProgramGraph g;
 bool planned=planProgramDataOrganization(analysis,c,choice,g,error);require(planned,"public data organization planner migrates target topology");if(!planned)continue;
 ProgramTimingRequest q;q.graphId=programGraphIdentity(g);q.modelIdentity="synthetic-target-only";
 for(size_t i=0;i<g.events.size();++i){ProgramActivityCost a;a.id="e"+std::to_string(i);a.events={i};a.resourceDemands[g.events[i].resource]=1;a.durationMs=1;a.boundary="event";a.provenance="synthetic";q.costs.push_back(a);}ProgramTimingEvaluation r;require(evaluateProgramTiming(g,q,r,error)&&r.makespanMs&&*r.makespanMs==3,"target-independent public timing");
 auto ownerGraph=g;for(auto &a:ownerGraph.allocations){std::string owner;programMemoryOwner(c,a.memory,owner,error);a.spatialOwner=owner;}
 require(checkProgramGraph(ownerGraph,error),"allocation owner refers to actual selected memory instance");ownerGraph.allocations.front().spatialOwner=p+"engine";if(ownerGraph.allocations.front().memory!=c.memories[0].id)ownerGraph.allocations.front().spatialOwner=p+"root";require(!checkProgramGraph(ownerGraph,error),"existing but wrong memory owner rejected");
 auto nativeTarget=c;nativeTarget.memories[1].capacityBytes=65536;
 llvm::json::Value nativeConfig=llvm::json::Object{{"generic_air",llvm::json::Object{{"capacity_bytes",65536},{"memory_architecture",serializeMemoryArchitecture(nativeTarget)},{"memory_binding",llvm::json::Object{{"host_memory",c.memories[0].id},{"local_memory",c.memories[1].id},{"compute_resource",c.resources[0].id},{"dma_resource",c.resources[1].id},{"input_link",c.links[0].id},{"output_link",c.links[1].id},{"submission_resource",c.resources[2].id}}}}}};
 auto nativeAir=nativeConfig.getAsObject()->getObject("generic_air");(*nativeAir)["capabilities"]=llvm::json::Object{{"max_group_inputs",4},{"max_group_outputs",3},{"acquire_all_inputs_at_group_entry",false}};
 GenericTargetCapabilities caps;require(getGenericTargetCapabilities(nativeConfig,caps,error)&&caps.maxGroupInputs==4&&caps.maxGroupOutputs==3&&!caps.acquireAllInputsAtGroupEntry,"stream limits and acquisition policy are target capabilities");
 GenericMappingPlan native;bool nativePlanned=planGenericMapping(analysis,nativeConfig,native,error);require(nativePlanned,"native generic planner accepts renamed 64KiB target");
 if(nativePlanned){auto &ng=native.communication;ProgramTimingRequest nq;nq.graphId=programGraphIdentity(ng);nq.modelIdentity="synthetic-native-target";ProgramActivityCost whole;whole.id="call";whole.durationMs=2;whole.boundary="whole call";whole.provenance="synthetic";for(size_t i=0;i<ng.events.size();++i){whole.events.push_back(i);whole.resourceDemands[ng.events[i].resource]=1;}for(size_t i=0;i<ng.regionEvents.size();++i){whole.regionEvents.push_back(i);whole.resourceDemands[ng.regionEvents[i].resource]=1;}nq.costs={whole};require(evaluateProgramTiming(ng,nq,r,error)&&r.makespanMs&&*r.makespanMs==2,"renamed native target schedules public graph lifecycle");require(!ng.executionGroups.empty()&&ng.executionGroups.front().spatialOwner==p+"engine","native owner is target resource instance");require(ng.executionRegions.size()==1&&ng.executionRegions.front().placement=="arbitrary-device"&&ng.executionRegions.front().invocationLifecycle=="host-device","native invocation policy is independent of target device label");auto invalidLifecycle=ng;invalidLifecycle.executionRegions.front().invocationLifecycle="guess-from-name";require(!checkProgramGraph(invalidLifecycle,error),"unsupported invocation lifecycle rejected");auto missingVisibility=ng;missingVisibility.executionRegions.front().invocationLifecycle="host-only";require(!checkProgramGraph(missingVisibility,error),"device visibility events cannot masquerade as host-only invocation");}
 auto bad=c;bad.memories[1].accessibleResources={p+"missing"};require(!checkProgramTargetBinding(bad,binding,1024,error),"dangling access reference rejected");bad=c;bad.memories[1].domain="unbound";require(!checkProgramTargetBinding(bad,binding,1024,error),"unknown owner rejected");require(!checkProgramTargetBinding(c,binding,1025,error),"target capacity respected");
 // Separate DMA engines imply one shared bus unit per activity. The max closure
 // avoids charging the same minimum reservation twice within one activity.
 auto sg=g;sg.resources.push_back({p+"bus","shared","arbitrary-device",1,p+"bank"});sg.resources.push_back({p+"other-mover","transfer","arbitrary-device",1,p+"bank"});for(auto &resource:sg.resources)if(resource.id==p+"move"||resource.id==p+"other-mover")resource.sharedResources[p+"bus"]=1;
 require(finalizeProgramGraph(analysis,sg,nullptr,error),"shared target extension is checked");q.graphId=programGraphIdentity(sg);
 auto runtime=[&](std::string id,std::string resource){ProgramActivityCost a;a.id=id;a.runtimeOnly=true;a.durationMs=2;a.boundary="runtime";a.provenance="synthetic";a.resourceDemands[resource]=1;return a;};q.costs.push_back(runtime("left",p+"move"));q.costs.push_back(runtime("right",p+"other-mover"));require(evaluateProgramTiming(sg,q,r,error)&&r.makespanMs,"shared-resource timing evaluates");bool exclusive=true;for(auto &a:r.activities)for(auto &b:r.activities)if(a.id=="left"&&b.id=="right")exclusive=*a.endMs<=*b.startMs||*b.endMs<=*a.startMs;require(exclusive,"distinct movers contend on explicit shared bus");
 StaticConfig sc;sc.domains=sg.domains;sc.resources=sg.resources;sc.memories=sg.memories;sc.links=sg.links;std::map<std::string,int64_t> expanded;require(programResourceDemands(sc,{{p+"move",1},{p+"other-mover",1}},expanded,error)&&expanded[p+"bus"]==1,"shared closure max is idempotent per activity");sc.resources.back().sharedResources[p+"move"]=1;for(auto &v:sc.resources)if(v.id==p+"bus")v.sharedResources[p+"other-mover"]=1;require(!validateMemoryArchitecture(sc,error),"shared dependency cycle rejected");
 }
 llvm::outs()<<"{\"status\":\""<<(ok?"passed":"failed")<<"\",\"checks\":"<<checks<<"}\n";return ok?0:1;
}
