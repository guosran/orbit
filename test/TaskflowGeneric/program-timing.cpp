#include "Taskflow/ProgramTiming.h"
#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include <cmath>
#include <set>
using namespace orbit::taskflow;
#include "execution-synthetic.inc"
int main(int argc,char **argv){
 if(argc!=2 && argc!=3)return 2;
 mlir::DialectRegistry registry;registry.insert<mlir::taskflow::TaskflowDialect,mlir::affine::AffineDialect,mlir::arith::ArithDialect,mlir::memref::MemRefDialect,mlir::func::FuncDialect>();
 mlir::MLIRContext context(registry);std::string error;unsigned checks=0;bool ok=true;
 auto require=[&](bool v,const char *s){if(v)++checks;else{ok=false;llvm::errs()<<s<<": "<<error<<"\n";}};
 auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);GenericAnalysis source;
 if(!module||!analyzeGenericProgram(*module,source,error))return 1;
 HeterogeneousMappingChoice choice;choice.taskDevices={"cpu","npu","cpu"};HeterogeneousMappingPlan plan;
 if(!buildHeterogeneousMapping(source,choice,plan,error)){llvm::errs()<<error;return 1;}
 auto &g=plan.graph;ProgramTimingRequest q;q.graphId=programGraphIdentity(g);q.modelIdentity="synthetic-unit-only";ProgramTimingEvaluation r;
 require(evaluateProgramTiming(g,q,r,error)&&!r.makespanMs&&!r.upperBoundMs&&r.lowerBoundMs==0,"unpriced graph remains unknown with lower bound");
 require(r.activities.size()==g.events.size()+g.regionEvents.size()&&!r.unknowns.empty(),"unknown coverage retains graph event identities and reasons");
 for(size_t i=0;i<g.events.size()+g.regionEvents.size();++i){ProgramActivityCost c;c.id="a"+std::to_string(i);c.boundary="synthetic-event";c.provenance="unit-test-no-hardware";c.durationMs=1;
  if(i<g.events.size()){c.events={i};c.resourceDemands[g.events[i].resource]=1;}else{c.regionEvents={i-g.events.size()};c.resourceDemands[g.regionEvents[i-g.events.size()].resource]=1;}q.costs.push_back(c);
 }
 require(evaluateProgramTiming(g,q,r,error)&&r.makespanMs&&r.upperBoundMs,"fully covered graph produces timeline");
 require(r.knownWorkMs==q.costs.size()&&r.lowerBoundMs<=*r.makespanMs&&*r.makespanMs<=*r.upperBoundMs,"work and bounds distinct from makespan");
 for(size_t i=0;i<g.events.size();++i)for(auto p:g.events[i].dependencies)require(*r.activities[i].startMs>=*r.activities[p].endMs,"real data dependency completion");
 auto full=q;
 auto unknown=q;unknown.costs[0].durationMs.reset();unknown.costs[0].lowerMs=.5;unknown.costs[0].upperMs=2;unknown.costs[0].unknownReason="unidentified startup";
 require(evaluateProgramTiming(g,unknown,r,error)&&!r.makespanMs&&r.upperBoundMs&&r.lowerBoundMs>0,"bounded unknown keeps known work and useful bounds");
 require(r.knownWorkMs==q.costs.size()-1&&!r.activities[1].startMs,"unknown lower schedule is never promoted to exact timestamp");
 auto bad=q;bad.graphId="wrong";require(!evaluateProgramTiming(g,bad,r,error),"changed graph rejected");
 bad=q;bad.costs.push_back(q.costs.front());bad.costs.back().id="double";require(!evaluateProgramTiming(g,bad,r,error),"duplicate coverage rejected");
 bad=q;bad.costs[0].resourceDemands.begin()->second=INT64_MAX;require(!evaluateProgramTiming(g,bad,r,error),"over-capacity occupancy rejected");
 bad=q;bad.costs[0].resourceDemands.clear();bad.costs[0].resourceDemands["NPU.submit"]=1;require(!evaluateProgramTiming(g,bad,r,error),"replacement of covered resource rejected");
 bad=q;bad.costs[0].lowerMs=2;require(!evaluateProgramTiming(g,bad,r,error),"invalid bound rejected");
 bad=q;bad.costs[0].durationMs=std::nan("");require(!evaluateProgramTiming(g,bad,r,error),"NaN duration rejected");
 bad=q;bad.costs[0].predecessors={"missing"};require(!evaluateProgramTiming(g,bad,r,error),"unknown predecessor rejected");
 bad=q;bad.costs[0].predecessors={bad.costs.back().id};bad.costs.back().predecessors={bad.costs[0].id};require(!evaluateProgramTiming(g,bad,r,error),"cyclic cost binding rejected");
 // One guarded call owns all public activities, including its lifecycle. No
 // internal event duration is separately added to this composite.
 ProgramActivityCost composite;composite.id="whole";composite.boundary="synthetic-whole-graph";composite.provenance="unit-test";composite.durationMs=3;
 for(size_t i=0;i<g.events.size();++i){composite.events.push_back(i);composite.resourceDemands[g.events[i].resource]=1;}
 for(size_t i=0;i<g.regionEvents.size();++i){composite.regionEvents.push_back(i);composite.resourceDemands[g.regionEvents[i].resource]=1;}
 q.costs={composite};require(evaluateProgramTiming(g,q,r,error)&&r.makespanMs&&*r.makespanMs==3&&r.activities.size()==1,"composite owns interval once");
 bad=q;bad.costs.push_back(full.costs.front());require(!evaluateProgramTiming(g,bad,r,error),"composite plus internal price rejected");
 // A valid phase-index capacity proof does not authorize a composite
 // timeline that makes every distinct allocation live simultaneously.
 auto memoryGraph=g;bool capacityCase=false;
 for(auto &m:memoryGraph.memories){int64_t serialPeak=0,unionBytes=0;for(size_t e=0;e<g.events.size();++e){int64_t live=0;std::set<size_t> seen;for(auto &copy:g.copies)if(copy.availableFrom<=e&&e<copy.releasedAfter&&g.allocations[copy.allocation].memory==m.id&&seen.insert(copy.allocation).second)live+=g.allocations[copy.allocation].bytes;serialPeak=std::max(serialPeak,live);}
  std::set<size_t> seen;for(auto &copy:g.copies)if(g.allocations[copy.allocation].memory==m.id&&seen.insert(copy.allocation).second)unionBytes+=g.allocations[copy.allocation].bytes;
  if(serialPeak>0&&unionBytes>serialPeak){m.capacityKnown=true;m.capacityBytes=serialPeak;capacityCase=true;}
 }
 require(capacityCase,"fixture contains reusable temporal allocation capacity");
 if(capacityCase){auto capacityRequest=q;capacityRequest.graphId=programGraphIdentity(memoryGraph);require(checkProgramGraph(memoryGraph,error),"original event lifetime capacity remains legal");require(!evaluateProgramTiming(memoryGraph,capacityRequest,r,error)&&error.find("temporal allocation capacity exceeded")!=std::string::npos,"actual simultaneous allocation capacity independently rechecked");}
 // Nonconvex coverage: merging first/last lifecycle while leaving the middle
 // outside must fail the contracted DAG instead of silently losing dependence.
 bad=full;size_t first=g.events.size(),last=first+g.regionEvents.size()-1;
 bad.costs[first].regionEvents.push_back(g.regionEvents.size()-1);bad.costs.erase(bad.costs.begin()+last);
 require(!evaluateProgramTiming(g,bad,r,error),"nonconvex composite dependency cycle rejected");
 auto runtime=[&](const char *id,const char *resource,double t){ProgramActivityCost c;c.id=id;c.runtimeOnly=true;c.boundary="synthetic-independent-runtime";c.provenance="unit-test";c.resourceDemands[resource]=1;c.durationMs=t;return c;};
 q.costs.push_back(runtime("host","CPU.compute",5));q.costs.push_back(runtime("device","NPU.compute",7));
 require(evaluateProgramTiming(g,q,r,error)&&*r.makespanMs==10,"independent distinct resources overlap after shared composite reservation");
 require(*r.activities[1].startMs==3&&*r.activities[2].startMs==3,"resource calendars allow declared overlap");
 q.costs.back().resourceDemands={{"CPU.compute",1}};
 require(evaluateProgramTiming(g,q,r,error)&&*r.makespanMs==15,"same resource activities mutually exclude");
 require(r.activities.back().resourcePredecessors==std::vector<std::string>{"host"}&&r.criticalPath==std::vector<std::string>{"whole","host","device"},"resource wait determines explicit critical path");
 q.costs.back().resourceDemands={{"CPU.compute",1},{"NPU.compute",1}};
 require(evaluateProgramTiming(g,q,r,error)&&*r.activities.back().startMs==8,"shared resource contention applies alongside private engine");
 // Capacity-2 fixed shares can coexist; lowering resource reservations or
 // duration for fluid fair sharing is deliberately outside this interface.
 auto g2=g;g2.resources.push_back({"fixed-shares","transfer","npu",2,"npu"});
 // Changing architecture requires a newly finalized checked graph identity.
 require(finalizeProgramGraph(source,g2,nullptr,error),"resource extension is checked by existing graph finalizer");
 q.graphId=programGraphIdentity(g2);q.costs={composite,runtime("left","fixed-shares",5),runtime("right","fixed-shares",7)};
 require(evaluateProgramTiming(g2,q,r,error)&&*r.makespanMs==7,"declared fixed resource shares coexist");
 q.costs.back().resourceDemands["fixed-shares"]=2;
 require(evaluateProgramTiming(g2,q,r,error)&&*r.makespanMs==12,"full shared reservation excludes other shares");
 // Exercise the native tiled reduction path, including a distributed source
 // initializer and in-place old-version read/new-version write at K boundaries.
 std::string fixture=argv[1];fixture=fixture.substr(0,fixture.find_last_of('/')+1)+"gemm.mlir";
 auto gemm=mlir::parseSourceFile<mlir::ModuleOp>(fixture,&context);GenericAnalysis ga;GenericMappingPlan gp;
 llvm::json::Value cfg=llvm::json::Object{{"generic_air",llvm::json::Object{{"tile_sizes",llvm::json::Array{2,2,3}}}}};
 bool native=gemm&&distributeNativeInitialization(*gemm,error)&&analyzeGenericProgram(*gemm,ga,error)&&planGenericMapping(ga,cfg,gp,error);
 require(native,"real tiled GEMM initializer graph");
 if(native){auto &ng=gp.communication;
  require(std::count_if(ng.regionEvents.begin(),ng.regionEvents.end(),[](auto &e){return e.kind=="submit";})==1 && std::count_if(ng.regionEvents.begin(),ng.regionEvents.end(),[](auto &e){return e.kind=="complete";})==1,"one native invocation has exactly one submit and completion");
  require(!ng.executionGroups.empty(),"native execution groups are public and identity-bound");
  auto wrongOwner=ng;bool ownerChanged=false;for(auto &e:wrongOwner.events)if(e.kind!="transfer"&&!e.writes.empty()){auto &allocation=wrongOwner.allocations[wrongOwner.copies[e.writes.front()].allocation];allocation.spatialOwner="other.tile";ownerChanged=true;break;}
  require(ownerChanged&&!checkProgramGraph(wrongOwner,error),"group rejects output replica on conflicting spatial owner");
  auto missingGroup=ng;missingGroup.executionGroups.front().memberEvents.pop_back();require(!checkProgramGraph(missingGroup,error),"public groups must cover every physical event");
  auto changedGroup=ng;changedGroup.executionGroups.front().spatialOwner="other.tile";require(programGraphIdentity(changedGroup)!=programGraphIdentity(ng),"spatial group decisions participate in graph identity");
  ProgramTimingRequest nq;nq.graphId=programGraphIdentity(ng);nq.modelIdentity="synthetic-native-unit";
  bool initializer=false;for(auto &e:ng.events)initializer|=e.kind=="initialize";require(initializer,"distributed initialization remains an actual priced event");
  for(size_t i=0;i<ng.events.size();++i){ProgramActivityCost c;c.id="native-event-"+std::to_string(i);c.events={i};c.boundary="synthetic-native-event";c.provenance="unit-test";c.durationMs=1;c.resourceDemands[ng.events[i].resource]=1;nq.costs.push_back(c);}
  require(evaluateProgramTiming(ng,nq,r,error)&&!r.makespanMs&&r.knownWorkMs==ng.events.size(),"separate K-tile event costs preserve legal buffer reuse DAG and unknown invocation");
  ProgramActivityCost nc;nc.id="native";nc.boundary="synthetic-whole-call";nc.provenance="unit-test";nc.durationMs=4;
  for(size_t i=0;i<ng.events.size();++i){nc.events.push_back(i);nc.resourceDemands[ng.events[i].resource]=1;}
  nq.costs={nc};require(evaluateProgramTiming(ng,nq,r,error)&&!r.makespanMs,"native submit/completion cannot disappear from complete cost");
  for(size_t i=0;i<ng.regionEvents.size();++i){nc.regionEvents.push_back(i);nc.resourceDemands[ng.regionEvents[i].resource]=1;}
  nq.costs={nc};require(evaluateProgramTiming(ng,nq,r,error)&&r.makespanMs&&*r.makespanMs==4,"tiled K accumulation whole-call coverage is legal");
  require(r.knownCapacitiesVerified,"complete temporal schedule validates known capacities");
  auto badn=nq;badn.costs[0].events.pop_back();require(!evaluateProgramTiming(ng,badn,r,error),"whole invocation cannot omit member activity");
  badn=nq;badn.costs[0].regionEvents.push_back(nc.regionEvents.front());require(!evaluateProgramTiming(ng,badn,r,error),"duplicate invocation ownership rejected");
  auto small=ng;for(auto &m:small.memories)if(m.id==gp.memoryBinding.localMemory)m.capacityBytes=1;
  badn=nq;badn.graphId=programGraphIdentity(small);require(!evaluateProgramTiming(small,badn,r,error),"illegal underlying memory capacity is rejected before timing");
 }
 llvm::json::Object evidence;
 require(runExecutionSynthetic(context,checks,error,&evidence),"source-backed synthetic execution graph suite");
 if(argc==3){std::error_code ec;llvm::raw_fd_ostream stream(argv[2],ec,llvm::sys::fs::OF_Text);if(ec){llvm::errs()<<ec.message();return 1;}stream<<llvm::formatv("{0:2}\n",llvm::json::Value(std::move(evidence)));}
 require(serializeProgramTiming(r).getAsObject()!=nullptr,"time schedule is serializable with explicit units");
 llvm::outs()<<"{\"status\":\""<<(ok?"passed":"failed")<<"\",\"checks\":"<<checks<<",\"scope\":\"synthetic public graph scheduling; no hardware timing claim\"}\n";return ok?0:1;
}
