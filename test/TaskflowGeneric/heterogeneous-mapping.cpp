#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/GenericMapping.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
using namespace orbit::taskflow;
// Every access and dependence is obtained from this actual executable body.
static const char *sourceText=R"mlir(
module {
 func.func @pipeline(%a:memref<16xf32> {orbit.role="input",orbit.name="A"},
                     %b:memref<16xf32> {orbit.role="input",orbit.name="B"},
                     %c:memref<16xf32> {orbit.role="output",orbit.name="C"}) attributes {orbit.noalias} {
  %t=memref.alloc() {orbit.name="T"}:memref<16xf32>
  %u=memref.alloc() {orbit.name="U"}:memref<16xf32>
  %td=taskflow.task @add will_reads(%a,%b:memref<16xf32>,memref<16xf32>) will_writes(%t:memref<16xf32>) [original_read_memrefs(%a,%b:memref<16xf32>,memref<16xf32>),original_write_memrefs(%t:memref<16xf32>)] :(memref<16xf32>,memref<16xf32>,memref<16xf32>)->memref<16xf32> {
  ^bb0(%x:memref<16xf32>,%y:memref<16xf32>,%z:memref<16xf32>):
   affine.for %i=0 to 16 {
    %v=affine.load %x[%i]:memref<16xf32>
    %w=affine.load %y[%i]:memref<16xf32>
    %r=arith.addf %v,%w:f32
    affine.store %r,%z[%i]:memref<16xf32>
   }
   taskflow.yield done_writes(%z:memref<16xf32>)
  }
  %ud=taskflow.task @scale will_reads(%td:memref<16xf32>) will_writes(%u:memref<16xf32>) [original_read_memrefs(%td:memref<16xf32>),original_write_memrefs(%u:memref<16xf32>)] :(memref<16xf32>,memref<16xf32>)->memref<16xf32> {
  ^bb0(%x:memref<16xf32>,%z:memref<16xf32>):
   %two=arith.constant 2.0:f32
   affine.for %i=0 to 16 {
    %v=affine.load %x[%i]:memref<16xf32>
    %r=arith.mulf %v,%two:f32
    affine.store %r,%z[%i]:memref<16xf32>
   }
   taskflow.yield done_writes(%z:memref<16xf32>)
  }
  %cd=taskflow.task @finish will_reads(%ud,%b:memref<16xf32>,memref<16xf32>) will_writes(%c:memref<16xf32>) [original_read_memrefs(%ud,%b:memref<16xf32>,memref<16xf32>),original_write_memrefs(%c:memref<16xf32>)] :(memref<16xf32>,memref<16xf32>,memref<16xf32>)->memref<16xf32> {
  ^bb0(%x:memref<16xf32>,%y:memref<16xf32>,%z:memref<16xf32>):
   affine.for %i=0 to 16 {
    %v=affine.load %x[%i]:memref<16xf32>
    %w=affine.load %y[%i]:memref<16xf32>
    %r=arith.addf %v,%w:f32
    affine.store %r,%z[%i]:memref<16xf32>
   }
   taskflow.yield done_writes(%z:memref<16xf32>)
  }
  memref.dealloc %u:memref<16xf32>
  memref.dealloc %t:memref<16xf32>
  return
 }
})mlir";
int main() {
 mlir::DialectRegistry registry;
 registry.insert<mlir::taskflow::TaskflowDialect,mlir::affine::AffineDialect,
                 mlir::arith::ArithDialect,mlir::memref::MemRefDialect,mlir::func::FuncDialect>();
 mlir::MLIRContext context(registry);std::string error;unsigned checks=0;bool ok=true;
 auto require=[&](bool value,const char *label){if(value)++checks;else{ok=false;llvm::errs()<<label<<": "<<error<<"\n";}};
 auto module=mlir::parseSourceString<mlir::ModuleOp>(sourceText,&context);GenericAnalysis source;
 if(!module || !analyzeGenericProgram(*module,source,error)){llvm::errs()<<error;return 1;}
 HeterogeneousMappingPlan mixed;
 for(unsigned mask=0;mask<8;++mask) {
  HeterogeneousMappingChoice choice;
  for(unsigned task=0;task<3;++task)choice.taskDevices.push_back(mask&(1u<<task)?"npu":"cpu");
  HeterogeneousMappingPlan plan;
  bool built=buildHeterogeneousMapping(source,choice,plan,error);
  require(built,"all actual device placements build physical parent graph");if(!built)continue;
  require(checkHeterogeneousMapping(source,choice,plan,error),"independent parent/choice/common validation");
  require(plan.graph.instances.size()==3 && !plan.graph.estimatedMakespanMs,
          "same actual source tasks and no invented calibrated cost");
  for(size_t region=0;region<plan.slices.size();++region) {
   mlir::OwningOpRef<mlir::ModuleOp> child;HeterogeneousRegionProjection projection;
   bool materialized=materializeHeterogeneousRegion(plan,region,child,projection,error);
   require(materialized,"actual region body clones with parent-copy ABI");if(!materialized)continue;
   GenericAnalysis analysis;require(analyzeGenericProgram(*child,analysis,error),"child independently analyzes");
   require(analysis.tasks.size()==plan.slices[region].endTask-plan.slices[region].beginTask,
           "bijective actual source task projection");
   for(const auto &argument:projection.arguments) {
    auto copy=argument.role=="input"?argument.inputCopy:argument.outputCopy;
    require(copy>=0 && plan.graph.allocations[plan.graph.copies[size_t(copy)].allocation].memory=="system.shared",
            "child ABI binds host-visible endpoint copy");
   }
   if(plan.slices[region].device=="npu") {
    GenericMappingPlan native;
    bool planned=planGenericMapping(analysis,llvm::json::Object{},native,error);
    require(planned,"actual NPU child native plan");
    if(planned)require(bindHeterogeneousRegionNativePlan(plan,native,projection,error),
                       "resident child physical allocations and DMA refine parent exactly");
   } else {
    HeterogeneousMappingChoice cpu;cpu.taskDevices.assign(analysis.tasks.size(),"cpu");
    HeterogeneousMappingPlan childPlan;
    bool planned=buildHeterogeneousMapping(analysis,cpu,childPlan,error);
    require(planned,"actual CPU child graph");
    if(planned)require(bindHeterogeneousRegionGraph(plan,childPlan.graph,projection,error),"CPU graph projection binds");
   }
   require(!projection.childGraphId.empty(),"child identity recorded only after checked binding");
   auto corrupt=projection;corrupt.arguments.front().parentStorage=source.storages.size();
   require(!bindHeterogeneousRegionGraph(plan,plan.graph,corrupt,error),"wrong parent-copy/source projection rejected");
  }
  if(mask==2)mixed=plan;
 }
 if(mixed.slices.size()!=3){llvm::errs()<<"expected CPU NPU CPU regions";return 1;}
 require(mixed.graph.transferBytes==128,"CPU NPU CPU has one actual import and one export");
 require(mixed.graph.regionEvents.size()==8,"CPU/NPU lifecycle counts 2+4+2");
 auto reject=[&](const char *label,std::function<void(HeterogeneousMappingPlan &)> change){auto p=mixed;change(p);error.clear();require(!checkHeterogeneousMapping(source,mixed.choice,p,error),label);};
 reject("missing visibility synchronization",[](auto &p){p.graph.regionEvents.erase(p.graph.regionEvents.begin()+2);});
 reject("missing invocation",[](auto &p){p.graph.executionRegions[1].submissionCount=0;});
 reject("wrong input host copy",[](auto &p){p.graph.executionRegions[1].inputCopies.clear();});
 reject("wrong task region",[](auto &p){p.graph.executionRegions[1].sourceTasks={0};});
 reject("wrong physical transfer payload",[](auto &p){for(auto &e:p.graph.events)if(e.kind=="transfer"){++e.transferBytes;break;}});
 reject("wrong boundary source identity",[](auto &p){p.sourceFingerprint="stale";});
 reject("stale version at NPU read",[](auto &p){for(auto &event:p.graph.events)if(event.kind=="compute" && event.task==1){p.graph.copies[event.reads.front()].version=0;break;}});
 reject("inaccessible NPU local memory",[](auto &p){for(auto &memory:p.graph.memories)if(memory.kind=="local-sram")memory.accessibleResources={"CPU.compute"};});
 reject("missing source data consumption",[](auto &p){for(auto &event:p.graph.events)if(event.kind=="compute" && !event.dataEdges.empty()){event.dataEdges.pop_back();break;}});
 reject("missing physical DMA creation dependency",[](auto &p){for(auto &event:p.graph.events)if(event.kind=="transfer" && !event.dependencies.empty()){event.dependencies.clear();break;}});
 reject("local copy released before its last read",[](auto &p){for(auto &copy:p.graph.copies)if(!copy.users.empty() && p.graph.allocations[copy.allocation].memory!="system.shared"){copy.releasedAfter=copy.users.back();break;}});
 reject("local payload capacity cannot be relaxed by labels",[](auto &p){for(auto &memory:p.graph.memories)if(memory.kind=="local-sram")memory.capacityBytes=1;});
 reject("duplicate invocation lifecycle",[](auto &p){p.graph.regionEvents.push_back(p.graph.regionEvents.back());});
 reject("cyclic region predecessor",[](auto &p){p.graph.executionRegions[0].predecessorRegions={1};});
 reject("region omits a physical member",[](auto &p){p.graph.executionRegions[1].memberEvents.pop_back();});
 reject("physical member belongs to two regions",[](auto &p){p.graph.executionRegions[0].memberEvents.push_back(p.graph.executionRegions[1].memberEvents.front());});
 reject("target binding changes",[](auto &p){p.graph.targetIdentity="another-local-memory-contract";});
 HeterogeneousMappingChoice split;split.taskDevices={"npu","npu","npu"};split.cuts={1,2};
 HeterogeneousMappingPlan splitPlan;
 require(buildHeterogeneousMapping(source,split,splitPlan,error) && splitPlan.slices.size()==3,
         "explicit same-device cuts create distinct region local copies");
 HeterogeneousMappingChoice merged;merged.taskDevices=split.taskDevices;HeterogeneousMappingPlan mergedPlan;
 require(buildHeterogeneousMapping(source,merged,mergedPlan,error) && mergedPlan.slices.size()==1 &&
         mergedPlan.graph.transferBytes==192 && splitPlan.graph.transferBytes>mergedPlan.graph.transferBytes,
         "native region grouping changes actual transfer facts rather than only labels");
 split.cuts={0};require(!buildHeterogeneousMapping(source,split,splitPlan,error),"invalid cut rejected");
 split.cuts={1,1};require(!buildHeterogeneousMapping(source,split,splitPlan,error),"duplicate cut rejected");
 split.cuts.clear();split.taskDevices[0]="amoeba";require(!buildHeterogeneousMapping(source,split,splitPlan,error),"unsupported device rejected");

 // Real source-shape change: the five full local tensors total 40 KiB.
 // The event-live parent peak is smaller, so resident-child certification is
 // essential. Splitting into three calls gives 24/16/24 KiB local unions.
 std::string largeText(sourceText);
 for(size_t at=0;(at=largeText.find("16",at))!=std::string::npos;at+=4)
   largeText.replace(at,2,"2048");
 auto largeModule=mlir::parseSourceString<mlir::ModuleOp>(largeText,&context);
 GenericAnalysis largeSource;
 bool largeAnalyzed=largeModule && analyzeGenericProgram(*largeModule,largeSource,error);
 require(largeAnalyzed,"2048-point actual source independently analyzes");
 if(largeAnalyzed) {
  HeterogeneousMappingPlan largeMerged;
  bool parentBuilt=buildHeterogeneousMapping(largeSource,merged,largeMerged,error);
  require(parentBuilt,"event-live parent alone does not certify whole-call resident capacity");
  if(parentBuilt) {
   mlir::OwningOpRef<mlir::ModuleOp> child;HeterogeneousRegionProjection projection;
   bool extracted=materializeHeterogeneousRegion(largeMerged,0,child,projection,error);
   require(extracted,"over-resident-budget source region still extracts without fake body changes");
   if(extracted) {
    GenericAnalysis analysis;GenericMappingPlan native;
    bool planned=analyzeGenericProgram(*child,analysis,error) &&
                 planGenericMapping(analysis,llvm::json::Object{},native,error);
    require(planned,"native planner handles real over-budget child through its existing strategy");
    if(planned) {
     error.clear();
     require(!bindHeterogeneousRegionNativePlan(largeMerged,native,projection,error) &&
             error.find("resident")!=std::string::npos,
             "40 KiB single-region child cannot bind as a resident heterogeneous call");
     require(projection.childGraphId.empty(),"rejected resident binding does not acquire a child identity");
    }
   }
  }
  HeterogeneousMappingChoice largeCuts;largeCuts.taskDevices={"npu","npu","npu"};largeCuts.cuts={1,2};
  HeterogeneousMappingPlan bounded;
  bool boundedBuilt=buildHeterogeneousMapping(largeSource,largeCuts,bounded,error);
  require(boundedBuilt,"same 2048-point source with explicit call cuts builds");
  if(boundedBuilt)for(size_t region=0;region<3;++region) {
   mlir::OwningOpRef<mlir::ModuleOp> child;HeterogeneousRegionProjection projection;
   bool extracted=materializeHeterogeneousRegion(bounded,region,child,projection,error);
   require(extracted,"capacity-coupled region extracts actual source task");
   if(!extracted)continue;
   GenericAnalysis analysis;GenericMappingPlan native;
   bool planned=analyzeGenericProgram(*child,analysis,error) &&
                planGenericMapping(analysis,llvm::json::Object{},native,error);
   require(planned,"split child is planned by unchanged native planner");
   if(planned) {
    require(native.strategy=="single-herd-resident" && native.residentBytes==(region==1?16384:24576),
            "split child physical resident unions are exactly 24/16/24 KiB");
    require(bindHeterogeneousRegionNativePlan(bounded,native,projection,error),
            "explicit cuts legalize actual resident refinement at unchanged 24 KiB budget");
   }
  }
 }
 llvm::outs()<<"{\"status\":\""<<(ok?"passed":"failed")<<"\",\"checks\":"<<checks
             <<",\"scope\":\"CPU-only physical parent and checked region projection; no device execution\"}\n";
 return ok?0:1;
}
