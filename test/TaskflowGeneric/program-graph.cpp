#include "Taskflow/ProgramMapping.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <algorithm>
using namespace orbit::taskflow;
static const char *source=R"mlir(
module {
 func.func @fanout(%a: memref<16xf32> {orbit.role="input",orbit.name="A"},
                   %b: memref<16xf32> {orbit.role="output",orbit.name="B"},
                   %c: memref<16xf32> {orbit.role="output",orbit.name="C"}) attributes {orbit.noalias} {
  %x = memref.alloc() : memref<16xf32>
  %xd = taskflow.task @produce will_reads(%a : memref<16xf32>) will_writes(%x : memref<16xf32>) [original_read_memrefs(%a : memref<16xf32>), original_write_memrefs(%x : memref<16xf32>)] : (memref<16xf32>,memref<16xf32>) -> memref<16xf32> {
  ^bb0(%in: memref<16xf32>,%out: memref<16xf32>):
   affine.for %i=0 to 16 {
    %v=affine.load %in[%i] : memref<16xf32>
    %two=arith.constant 2.0 : f32
    %r=arith.mulf %v,%two : f32
    affine.store %r,%out[%i] : memref<16xf32>
   }
   taskflow.yield done_writes(%out : memref<16xf32>)
  }
  %bd = taskflow.task @first will_reads(%xd : memref<16xf32>) will_writes(%b : memref<16xf32>) [original_read_memrefs(%xd : memref<16xf32>), original_write_memrefs(%b : memref<16xf32>)] : (memref<16xf32>,memref<16xf32>) -> memref<16xf32> {
  ^bb0(%in: memref<16xf32>,%out: memref<16xf32>):
   affine.for %i=0 to 16 {
    %v=affine.load %in[%i] : memref<16xf32>
    affine.store %v,%out[%i] : memref<16xf32>
   }
   taskflow.yield done_writes(%out : memref<16xf32>)
  }
  %cd = taskflow.task @second will_reads(%xd,%bd : memref<16xf32>,memref<16xf32>) will_writes(%c : memref<16xf32>) [original_read_memrefs(%xd,%bd : memref<16xf32>,memref<16xf32>), original_write_memrefs(%c : memref<16xf32>)] : (memref<16xf32>,memref<16xf32>,memref<16xf32>) -> memref<16xf32> {
  ^bb0(%in: memref<16xf32>,%orderOnly: memref<16xf32>,%out: memref<16xf32>):
   affine.for %i=0 to 16 {
    %v=affine.load %in[%i] : memref<16xf32>
    affine.store %v,%out[%i] : memref<16xf32>
   }
   taskflow.yield done_writes(%out : memref<16xf32>)
  }
  memref.dealloc %x : memref<16xf32>
  return
 }
})mlir";

int main(int argc,char **argv) {
  if(argc!=2){llvm::errs()<<"usage: taskflow-program-graph-test gemm-scale.mlir\n";return 2;}
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect,mlir::affine::AffineDialect,
      mlir::arith::ArithDialect,mlir::memref::MemRefDialect,mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);std::string error;unsigned checks=0;bool ok=true;
  auto require=[&](bool value,const char *name){if(!value){ok=false;llvm::errs()<<name<<": "<<error<<"\n";}else ++checks;};
  auto module=mlir::parseSourceString<mlir::ModuleOp>(source,&context);GenericAnalysis a;
  if(!module || !analyzeGenericProgram(*module,a,error)){llvm::errs()<<error;return 1;}
  StaticConfig target;target.resources={{"cpu","compute","cpu",1},{"accelerator","compute","accel",1},{"dma","transfer","system",1}};
  target.memories={{"dram",4096,{"cpu","accel"}},{"sram",192,{"accel"}}};
  StaticTransfer link;link.id="import";link.source="dram";link.target="sram";link.resource="dma";
  link.latencyMs=.1;link.bandwidthBytesPerMs=64;link.provenance={"synthetic","v11 CPU test only, not measured hardware"};
  target.links={link};target.assumptions.initialLocations["A"]="dram";
  target.assumptions.outputLocations["B"]="sram";target.assumptions.outputLocations["C"]="sram";
  ProgramDataOrganizationChoice choice;choice.placements={{0,"cpu","dram","scalar"},{1,"accelerator","sram","scalar"},{2,"accelerator","sram","scalar"}};
  for(size_t t=0;t<3;++t){MappingComputeEstimate e;e.task=t;e.resource=choice.placements[t].resource;e.implementation="scalar";
    e.duration=1;e.model="synthetic";e.modelVersion="v11";e.provenance="explicit CPU duration, no model training";
    e.boundary="compute only with supplied inputs";e.transportScope=MappingComputeTransportScope::ExcludesInterTaskTransfers;
    e.includedCosts={"body compute"};e.excludedCosts={"inter-task transfer"};e.supplyConditionsVerified=true;choice.computeEstimates.push_back(e);}
  ProgramGraph local,retained;
  if(!planProgramDataOrganization(a,target,choice,local,error)){llvm::errs()<<error;return 1;}
  choice.policy=ProgramDataPolicy::RetainAcrossUsers;
  if(!planProgramDataOrganization(a,target,choice,retained,error)){llvm::errs()<<error;return 1;}
  require(checkProgramGraph(local,error) && checkProgramGraphSource(a,local,error),"common check phase-local A");
  require(checkProgramGraph(retained,error) && checkProgramGraphSource(a,retained,error),"common check retained A");
  require(local.transferBytes==128 && retained.transferBytes==64 && retained.dataEdges.size()==3 && retained.orderEdges.size()==1,"source fanout/order/accounting");
  require(retained.source.tasks.size()==3 && retained.instances.size()==3 && retained.schedule.events.size()==retained.events.size(),"source/instance/schedule common coverage");
  require(!programGraphIdentity(retained).empty() && programGraphIdentity(local)!=programGraphIdentity(retained),"choice-bound graph identity");
  require(renderProgramGraphDOT(retained).find("digraph")!=std::string::npos,"native DOT export");
  require(serializeProgramGraph(retained).getAsObject() && summarizeProgramGraphCost(retained).getAsObject(),"native graph/cost JSON export");
  auto retainedCost=summarizeProgramGraphCost(retained);
  require(retainedCost.getAsObject()->getObject("peak_memory_bytes")->getInteger("sram")==192,
          "shared allocation counted once in exported capacity peak");
  auto reject=[&](const char *name,std::function<void(ProgramGraph &)> mutate){auto g=retained;mutate(g);error.clear();require(!checkProgramGraph(g,error),name);};
  reject("missing instance",[](auto &g){g.instances.pop_back();});
  reject("invalid instance event",[](auto &g){g.instances.front().event=g.events.size();});
  reject("invalid instance task",[](auto &g){g.instances.front().task=g.source.tasks.size();});
  reject("instance outside source domain",[](auto &g){++g.instances.front().domain.front().upper;});
  reject("missing scheduled event",[](auto &g){g.schedule.events.pop_back();});
  reject("invalid schedule event",[](auto &g){g.schedule.events.back().event=g.events.size();});
  reject("wrong schedule predecessors",[](auto &g){g.schedule.events.back().predecessors.clear();});
  reject("negative scheduled time",[](auto &g){g.schedule.events.front().startMs=-1;});
  reject("invalid copy version",[](auto &g){g.copies.front().version=g.versions.size();});
  reject("invalid copy allocation",[](auto &g){g.copies.front().allocation=g.allocations.size();});
  reject("negative logical region",[](auto &g){g.regions.front().offsets.front()=-1;});
  reject("missing pure order",[](auto &g){g.orderEdges.clear();});
  reject("shared memory capacity below live allocations",[](auto &g){g.memories.back().capacityBytes=1;});
  reject("copy released before use",[](auto &g){for(auto &c:g.copies)if(!c.users.empty()){c.releasedAfter=c.users.back();return;}});
  reject("source dependency facts stale",[](auto &g){g.source.tasks.back().dependencies.clear();});
  reject("fabricated transfer byte report",[](auto &g){++g.transferBytes;});
  reject("wrong transfer rectangle",[](auto &g){for(auto &e:g.events)if(e.kind=="transfer"){if(e.transferRegion)++e.transferRegion->sizes.front();return;}});
  reject("wrong transfer payload",[](auto &g){for(auto &e:g.events)if(e.kind=="transfer"){++e.transferBytes;return;}});
  auto incidence=[](ProgramGraph &g) {
    for(auto &e:g.events){e.dataEdges.clear();e.orderEdges.clear();}
    for(auto &c:g.copies){c.users.clear();c.lastUsers.clear();c.consumerEdges.clear();}
    for(size_t e=0;e<g.events.size();++e)for(size_t c:g.events[e].reads)g.copies[c].users.push_back(e);
    // This example's actual event DAG is serial; the final ordinal is its
    // unique maximal user. Lifetime bounds remain conservative and valid.
    for(auto &c:g.copies)if(!c.users.empty())c.lastUsers={c.users.back()};
    for(size_t i=0;i<g.dataEdges.size();++i){auto &d=g.dataEdges[i];g.events[d.consumer].dataEdges.push_back(i);g.copies[d.copy].consumerEdges.push_back(i);}
    for(size_t i=0;i<g.orderEdges.size();++i)g.events[g.orderEdges[i].consumer].orderEdges.push_back(i);
  };
  reject("coherent missing real source read",[&](auto &g){
    size_t last=g.events.size()-1;g.events[last].reads.clear();
    auto &edges=g.dataEdges;edges.erase(std::remove_if(edges.begin(),edges.end(),[&](const auto &d){return d.consumer==last;}),edges.end());
    // Preserve its producer ordering even after the forged deletion of RAW,
    // so rejection cannot rely only on a missing source dependency/incidence.
    g.orderEdges.push_back({0,last,"source task dependency without value consumption"});
    g.events[last].dependencies.push_back(0);std::sort(g.events[last].dependencies.begin(),g.events[last].dependencies.end());
    g.events[last].dependencies.erase(std::unique(g.events[last].dependencies.begin(),g.events[last].dependencies.end()),g.events[last].dependencies.end());
    g.schedule.events[last].predecessors=g.events[last].dependencies;incidence(g);
  });
  reject("coherent pure order forged as B data read",[&](auto &g){
    size_t last=g.events.size()-1,first=SIZE_MAX,copy=SIZE_MAX;
    for(size_t e=0;e<g.events.size();++e)if(g.events[e].kind=="compute" && g.events[e].task==1)first=e;
    for(size_t c=0;c<g.copies.size();++c)if(g.copies[c].createdBy==int64_t(first))copy=c;
    if(first==SIZE_MAX || copy==SIZE_MAX)return;
    g.events[last].reads.push_back(copy);auto &value=g.copies[copy];auto &version=g.versions[value.version];auto &allocation=g.allocations[value.allocation];
    auto &storage=g.source.storages[g.regions[version.region].storage];
    g.dataEdges.push_back({version.producer,last,version.region,value.version,copy,{allocation.memory,value.layout,storage.dtype}});
    auto &order=g.orderEdges;order.erase(std::remove_if(order.begin(),order.end(),[&](const auto &d){return d.producer==first && d.consumer==last;}),order.end());incidence(g);
  });
  auto unknown=choice;unknown.computeEstimates.clear();ProgramGraph unknownGraph;
  require(planProgramDataOrganization(a,target,unknown,unknownGraph,error) && checkProgramGraph(unknownGraph,error) && !unknownGraph.schedule.estimatedMakespanMs,"unknown durations stay null");
  auto changed=retained;changed.bodyFingerprint="stale-source";
  require(!checkProgramGraphSource(a,changed,error),"independent source identity reject");
  auto native=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);GenericAnalysis b;GenericMappingPlan base,panel;
  llvm::json::Value cfg=llvm::json::Object{{"generic_air",llvm::json::Object{{"tile_sizes",llvm::json::Array{2,2,4}}}}};
  if(!native || !distributeNativeInitialization(*native,error) || !analyzeGenericProgram(*native,b,error) || !planGenericMapping(b,cfg,base,error)){llvm::errs()<<error;return 1;}
  require(checkProgramGraph(base.communication,error) && checkProgramGraphSource(b,base.communication,error),"same common checker on native B");
  require(base.communication.instances.size()>b.tasks.size() && base.communication.executionRegions.size()==1 && base.communication.executionRegions.front().submissionCount==1,"native tiled instances share one program submit");
  bool found=false;
  for(const auto &c:discoverGenericMappingChoices(b))if(c.storage==1 && c.reuseLoop==0 && c.panelLoop==1){found=setNativeMappingChoice(*native,c,error);break;}
  if(!found || !analyzeGenericProgram(*native,b,error) || !planGenericMapping(b,cfg,panel,error)){llvm::errs()<<error;return 1;}
  require(checkProgramGraph(panel.communication,error) && checkProgramGraphSource(b,panel.communication,error),"same common checker retained B");
  auto nativeCost=summarizeProgramGraphCost(panel.communication);
  require(nativeCost.getAsObject()->getObject("peak_memory_bytes")->getInteger(panel.memoryBinding.localMemory)==panel.residentBytes,
          "native exported allocation peak equals checked local tensor pool");
  require(programGraphIdentity(base.communication)!=programGraphIdentity(panel.communication),"native choice changes common graph identity");
  require(base.communication.transferBytes>panel.communication.transferBytes,"native retention changes transfer cost facts");
  llvm::outs()<<llvm::json::Object{{"status",ok?"PASS":"FAIL"},{"checks",int64_t(checks)},{"hardware_executed",false},
    {"scope","independent canonical graph mutations over actual CPU fanout and native tiled GEMM-scale; synthetic CPU costs"},
    {"example_a",serializeProgramGraph(retained)},{"example_b",serializeProgramGraph(panel.communication)}}<<"\n";
  return ok?0:1;
}
