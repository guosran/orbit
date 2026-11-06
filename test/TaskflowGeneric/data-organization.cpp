#include "Taskflow/ProgramMapping.h"
#include "Taskflow/GenericAnalysis.h"
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
// Every node and edge below is recovered from this executable Taskflow body.
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
int main() {
 mlir::DialectRegistry registry;
 registry.insert<mlir::taskflow::TaskflowDialect,mlir::affine::AffineDialect,
  mlir::arith::ArithDialect,mlir::memref::MemRefDialect,mlir::func::FuncDialect>();
 mlir::MLIRContext context(registry);
 auto module=mlir::parseSourceString<mlir::ModuleOp>(source,&context);
 GenericAnalysis a;std::string error;bool ok=true;unsigned checks=0;
 if(!module || !analyzeGenericProgram(*module,a,error)){llvm::errs()<<error<<"\n";return 1;}
 StaticConfig target;
 target.resources={{"cpu","compute","cpu",1},{"accelerator","compute","accel",1},{"dma","transfer","system",1}};
 target.memories={{"dram",4096,{"cpu","accel"}},{"sram",192,{"accel"}}};
 StaticTransfer in;in.id="import";in.source="dram";in.target="sram";in.resource="dma";
 in.latencyMs=0.1;in.bandwidthBytesPerMs=64;in.provenance={"synthetic","explicit CPU test parameters, not measured hardware"};
 target.links={in};target.assumptions.initialLocations["A"]="dram";
 target.assumptions.outputLocations["B"]="sram";target.assumptions.outputLocations["C"]="sram";
 ProgramDataOrganizationChoice choice;choice.placements={{0,"cpu","dram","scalar"},{1,"accelerator","sram","scalar"},{2,"accelerator","sram","scalar"}};
 for(size_t t=0;t<3;++t){MappingComputeEstimate estimate;estimate.task=t;estimate.resource=choice.placements[t].resource;estimate.implementation="scalar";
  estimate.duration=1;estimate.model="synthetic";estimate.modelVersion="1";estimate.provenance="CPU regression explicit duration";
  estimate.boundary="compute only, supplied inputs";estimate.transportScope=MappingComputeTransportScope::ExcludesInterTaskTransfers;estimate.includedCosts={"body compute"};estimate.excludedCosts={"inter-task transfer"};estimate.supplyConditionsVerified=true;choice.computeEstimates.push_back(estimate);}
 ProgramCommunicationPlan local,retained;
 if(!planProgramDataOrganization(a,target,choice,local,error)){llvm::errs()<<error<<"\n";return 1;}++checks;
 choice.policy=ProgramDataPolicy::RetainAcrossUsers;
 if(!planProgramDataOrganization(a,target,choice,retained,error)){llvm::errs()<<error<<"\n";return 1;}++checks;
 if(local.transferBytes!=128 || retained.transferBytes!=64 || !local.estimatedMakespanMs || !retained.estimatedMakespanMs ||
    *local.estimatedMakespanMs<=*retained.estimatedMakespanMs || retained.orderEdges.size()!=1 || retained.dataEdges.size()!=3) {
  llvm::errs()<<"wrong source-derived fanout traffic/cost/order edges\n";return 1;}++checks;
 size_t shared=SIZE_MAX;
 for(size_t i=0;i<retained.copies.size();++i)if(retained.copies[i].consumerEdges.size()==2)shared=i;
 if(shared==SIZE_MAX || retained.copies[shared].users.size()!=2 || retained.copies[shared].lastUsers.size()!=1 ||
    retained.allocations.size()+1!=local.allocations.size())return 1;++checks;
 auto reject=[&](const char *name,std::function<void(ProgramCommunicationPlan &)> mutate) {
   auto p=retained;mutate(p);error.clear();if(checkProgramDataOrganization(a,target,choice,p,error)){llvm::errs()<<"accepted "<<name<<"\n";ok=false;}else ++checks;
 };
 reject("early release",[&](auto &p){p.copies[shared].releasedAfter=p.copies[shared].users.back();});
 reject("wrong shared version",[&](auto &p){p.copies[shared].version=0;});
 reject("wrong region",[](auto &p){--p.regions.front().sizes.front();});
 reject("missing creation dependence",[&](auto &p){auto event=p.copies[shared].users.back();auto creator=size_t(p.copies[shared].createdBy);auto &deps=p.events[event].dependencies;deps.erase(std::remove(deps.begin(),deps.end(),creator),deps.end());});
 reject("missing consumer",[&](auto &p){p.copies[shared].consumerEdges.pop_back();});
 reject("missing user",[&](auto &p){p.copies[shared].users.pop_back();});
 reject("wrong consumption memory",[](auto &p){p.dataEdges.back().requirement.memory="dram";});
 reject("wrong consumption layout",[](auto &p){p.dataEdges.back().requirement.layout="transposed";});
 reject("missing order edge",[](auto &p){p.orderEdges.clear();});
 reject("order becomes transfer",[](auto &p){p.events[p.orderEdges.front().consumer].kind="transfer";});
 reject("duplicate creation cost",[](auto &p){p.transferBytes+=64;});
 reject("duplicate physical allocation",[&](auto &p){p.allocations.push_back(p.allocations[p.copies[shared].allocation]);});
 reject("source fingerprint",[](auto &p){p.bodyFingerprint="stale-source";});
 reject("unknown target resource",[](auto &p){p.resources.front().id="other";});
 reject("invented zero cost",[](auto &p){p.estimatedMakespanMs=0;});
 reject("old policy with shared copy",[](auto &p){p.dataPolicy=ProgramDataPolicy::PhaseLocal;});
 reject("invalid negative creator",[](auto &p){p.copies.front().createdBy=-2;});
 reject("invalid negative version producer",[](auto &p){p.versions.front().producer=-2;});
 reject("invalid version producer index",[](auto &p){p.versions.front().producer=p.events.size();});
 reject("invalid write index",[](auto &p){p.events.back().writes={p.copies.size()};});
 reject("invalid allocation index",[](auto &p){p.copies.front().allocation=p.allocations.size();});
 reject("invalid copy version index",[](auto &p){p.copies.front().version=p.versions.size();});
 reject("invalid event read index",[](auto &p){p.events.back().reads={p.copies.size()};});
 reject("wrong event memory peak",[](auto &p){++p.peakMemoryBytes["sram"];});
 auto small=target;small.memories.back().capacityBytes=127;
 ProgramCommunicationPlan rejected;
 if(planProgramDataOrganization(a,small,choice,rejected,error)){llvm::errs()<<"retention over capacity accepted\n";ok=false;}else ++checks;
 auto unknown=choice;unknown.computeEstimates.clear();ProgramCommunicationPlan noCost;
 if(!planProgramDataOrganization(a,target,unknown,noCost,error) || noCost.estimatedMakespanMs){llvm::errs()<<"unknown cost fabricated\n";ok=false;}else ++checks;
 auto unspecified=choice;unspecified.computeEstimates.front().transportScope=MappingComputeTransportScope::Unspecified;
 if(planProgramDataOrganization(a,target,unspecified,rejected,error)){llvm::errs()<<"unspecified transfer ownership accepted\n";ok=false;}else ++checks;
 auto bundled=choice;bundled.computeEstimates.front().transportScope=MappingComputeTransportScope::IncludesInterTaskTransfers;
 if(planProgramDataOrganization(a,target,bundled,rejected,error)){llvm::errs()<<"bundled communication double charged\n";ok=false;}else ++checks;
 auto cycles=choice;cycles.computeEstimates[0].unit="cycles";
 if(planProgramDataOrganization(a,target,cycles,rejected,error)){llvm::errs()<<"cycles without clock accepted\n";ok=false;}else ++checks;
 cycles.computeEstimates[0].duration=1000000;cycles.clockMHz=1000;ProgramCommunicationPlan cyclePlan;
 if(!planProgramDataOrganization(a,target,cycles,cyclePlan,error) || cyclePlan.estimatedMakespanMs!=retained.estimatedMakespanMs) {
   llvm::errs()<<"explicit clock cycles conversion differs from 1ms fullgraph cost: "<<error<<"\n";ok=false;
 }else ++checks;
 auto stale=retained;auto changedSource=std::string(source);auto offset=changedSource.find("arith.mulf");changedSource.replace(offset,10,"arith.addf");
 auto changed=mlir::parseSourceString<mlir::ModuleOp>(changedSource,&context);GenericAnalysis altered;
 if(!changed || !analyzeGenericProgram(*changed,altered,error) || checkProgramDataOrganization(altered,target,choice,stale,error)){llvm::errs()<<"structure/source copy reanalysis missing\n";ok=false;}else ++checks;
 // A real structural change is reanalyzed. The conservative whole-task
 // adapter rejects split partial accesses instead of reusing stale copies.
 auto split=mlir::parseSourceString<mlir::ModuleOp>(source,&context);GenericAnalysis splitFacts;
 if(!split || !splitNativeOuterLoops(*split,error) || !analyzeGenericProgram(*split,splitFacts,error) ||
    planProgramDataOrganization(splitFacts,target,choice,rejected,error) || error.find("full-storage source regions")==std::string::npos) {
   llvm::errs()<<"split whole-task boundary did not fail closed: "<<error<<"\n";ok=false;
 }else ++checks;
 // A shape change within the adapter's exact-region scope rebuilds its copies.
 auto smallerSource=std::string(source);size_t replaceAt=0;
 while((replaceAt=smallerSource.find("16",replaceAt))!=std::string::npos){smallerSource.replace(replaceAt,2,"8");++replaceAt;}
 auto smaller=mlir::parseSourceString<mlir::ModuleOp>(smallerSource,&context);GenericAnalysis smallFacts;ProgramCommunicationPlan rebuilt;
 if(!smaller || !analyzeGenericProgram(*smaller,smallFacts,error) ||
    !planProgramDataOrganization(smallFacts,target,choice,rebuilt,error) || rebuilt.transferBytes!=32 ||
    rebuilt.bodyFingerprint==retained.bodyFingerprint || rebuilt.regions.front().sizes.front()!=8) {
   llvm::errs()<<"shape reanalysis did not rebuild copies: "<<error<<"\n";ok=false;
 }else ++checks;
 // A real, statically valid MLIR type whose full payload exceeds int64.
 // No buffers or loop iterations are executed. Exercise the independent
 // checker directly with forged accounting, not merely builder rejection.
 auto hugeSource=std::string(source);
 auto replaceAll=[&](const std::string &from,const std::string &to){size_t p=0;while((p=hugeSource.find(from,p))!=std::string::npos){hugeSource.replace(p,from.size(),to);p+=to.size();}};
 replaceAll("16xf32","2000000000x2000000000xf32");
 replaceAll("affine.for %i=0 to 16 {","affine.for %i=0 to 2000000000 { affine.for %j=0 to 2000000000 {");
 replaceAll("[%i]","[%i,%j]");replaceAll("   }\n   taskflow.yield","   }\n   }\n   taskflow.yield");
 auto huge=mlir::parseSourceString<mlir::ModuleOp>(hugeSource,&context);GenericAnalysis hugeFacts;
 if(!huge || !analyzeGenericProgram(*huge,hugeFacts,error)){llvm::errs()<<"invalid overflow metadata fixture: "<<error<<"\n";ok=false;}
 else {
   auto forged=retained;forged.bodyFingerprint=hugeFacts.bodyFingerprint;
   for(size_t i=0;i<forged.regions.size();++i){forged.regions[i].offsets={0,0};forged.regions[i].sizes=hugeFacts.storages[i].shape;}
   for(auto &al:forged.allocations)al.bytes=-1;
   if(checkProgramDataOrganization(hugeFacts,target,choice,forged,error) || error.find("overflows int64")==std::string::npos) {
     llvm::errs()<<"independent checker accepted overflowing source bytes: "<<error<<"\n";ok=false;
   }else ++checks;
 }
 reject("negative allocation bytes",[](auto &p){p.allocations.front().bytes=-1;});
 // Target-owned sharing, resource access and multi-hop connectivity.
 auto arch=target;arch.domains={{"system","","system"},{"device","system","device"},{"tile0","device","tile"},{"tile1","device","tile"}};
 arch.resources[0].domain="system";arch.resources[1].domain="tile0";arch.resources[2].domain="device";
 arch.resources.push_back({"accelerator1","compute","accel",1,"tile1"});
 arch.memories[0].domain="system";arch.memories[0].kind="dram";arch.memories[0].accessibleResources={"cpu"};
 arch.memories[1].domain="device";arch.memories[1].kind="shared-sram";arch.memories[1].accessibleResources={"accelerator","accelerator1"};
 arch.memories.push_back({"local0",192,{"accel"},"tile0","local-sram",{"accelerator"}});
 arch.memories.push_back({"local1",192,{"accel"},"tile1","local-sram",{"accelerator1"}});
 auto edge=[&](const char *id,const char *from,const char *to){auto l=in;l.id=id;l.source=from;l.target=to;l.domain="device";return l;};
 arch.links={edge("to-shared","dram","sram"),edge("to-local0","sram","local0"),edge("to-local1","sram","local1")};
 auto require=[&](bool valid,const char *name){if(!valid){llvm::errs()<<"architecture: "<<name<<": "<<error<<"\n";ok=false;}else ++checks;};
 require(validateMemoryArchitecture(arch,error),"shared and two-local architecture");
 require(memoryAccessible(arch,"sram","accelerator1") && !memoryAccessible(arch,"local0","accelerator1"),"precise resource accessibility overrides same device");
 std::vector<std::string> path;
 require(findMemoryPath(arch,"dram","local1",path,error) && path==std::vector<std::string>{"to-shared","to-local1"},"two-hop reachability");
 require(!findMemoryPath(arch,"local1","local0",path,error),"directed unreachable path");
 auto malformed=arch;malformed.domains.front().parent="tile0";
 require(!validateMemoryArchitecture(malformed,error),"ownership cycle rejected");
 malformed=arch;malformed.memories.back().accessibleResources={"missing"};
 require(!validateMemoryArchitecture(malformed,error),"unknown accessor rejected");
 malformed=arch;malformed.links.back().resource="accelerator";
 require(!validateMemoryArchitecture(malformed,error),"compute resource cannot be DMA");
 StaticConfig roundtrip;require(parseMemoryArchitecture(serializeMemoryArchitecture(arch),roundtrip,error) &&
   staticFingerprint(serializeMemoryArchitecture(roundtrip))==staticFingerprint(serializeMemoryArchitecture(arch)),"architecture roundtrip");
 auto commonChoice=choice;commonChoice.placements.back().resource="accelerator1";commonChoice.computeEstimates.back().resource="accelerator1";
 ProgramCommunicationPlan sharedPlan,separatePlan,pathPlan;
 require(planProgramDataOrganization(a,arch,commonChoice,sharedPlan,error) && sharedPlan.transferBytes==64,"shared copy allocated/transferred once across compute resources");
 auto separate=arch;commonChoice.placements[1].memory="local0";commonChoice.placements[2].memory="local1";
 separate.assumptions.outputLocations["B"]="local0";separate.assumptions.outputLocations["C"]="local1";
 require(planProgramDataOrganization(a,separate,commonChoice,separatePlan,error) && separatePlan.transferBytes==192,"shared staging plus two local copies with actual path bytes");
 auto separateLocalChoice=commonChoice;separateLocalChoice.policy=ProgramDataPolicy::PhaseLocal;
 ProgramCommunicationPlan separateLocalPlan;
 require(planProgramDataOrganization(a,separate,separateLocalChoice,separateLocalPlan,error) && separateLocalPlan.transferBytes==256,
   "phase-local releases every imported multi-hop copy at task boundary");
 require(checkProgramDataOrganization(a,separate,separateLocalChoice,separateLocalPlan,error) &&
   checkProgramDataOrganization(a,separate,commonChoice,separatePlan,error),"both multi-hop policies pass independent checking");
 auto importCount=[&](const ProgramCommunicationPlan &p,const std::string &memory){size_t count=0;
   for(const auto &copy:p.copies)if(copy.createdBy>=0 && p.events[copy.createdBy].kind=="transfer" &&
       p.allocations[copy.allocation].memory==memory)++count;return count;};
 require(importCount(separatePlan,"sram")==1 && importCount(separateLocalPlan,"sram")==2 &&
   importCount(separatePlan,"local0")==1 && importCount(separatePlan,"local1")==1 &&
   importCount(separateLocalPlan,"local0")==1 && importCount(separateLocalPlan,"local1")==1,
   "only retained policy shares a transit creation across tasks");
 auto hiddenTransitRetention=separatePlan;hiddenTransitRetention.dataPolicy=ProgramDataPolicy::PhaseLocal;
 error.clear();
 require(!checkProgramDataOrganization(a,separate,separateLocalChoice,hiddenTransitRetention,error) &&
   error.find("across task/transfer consumers")!=std::string::npos,
   "independent checker rejects phase-local transit sharing with accurate lifetimes/costs");
 // The producer's DRAM version remains a source for both consumers. Removing
 // imported copies must not discard compute-created copies or fabricate a
 // new logical version to pay for the second route.
 require(separateLocalPlan.versions.size()==separatePlan.versions.size() &&
   separateLocalPlan.dataEdges.size()==separatePlan.dataEdges.size() &&
   separateLocalPlan.orderEdges.size()==separatePlan.orderEdges.size() &&
   separateLocalPlan.allocations.size()==separatePlan.allocations.size()+1,
   "phase-local route recreation preserves source versions and data/order edges");
 auto unc=arch;for(auto &l:unc.links){l.latencyMs=0;l.bandwidthBytesPerMs=0;l.provenance={};}
 auto sharedChoice=choice;sharedChoice.placements.back().resource="accelerator1";sharedChoice.computeEstimates.back().resource="accelerator1";
 require(planProgramDataOrganization(a,unc,sharedChoice,pathPlan,error) && !pathPlan.estimatedMakespanMs && pathPlan.transferBytes==64,"uncalibrated paths still check capacity and validity");
 malformed=arch;malformed.memories[1].capacityBytes=127;
 require(!planProgramDataOrganization(a,malformed,sharedChoice,pathPlan,error),"shared memory capacity limits residency");
 auto denied=sharedChoice;denied.placements.back().memory="local0";
 require(!planProgramDataOrganization(a,arch,denied,pathPlan,error),"same device cannot bypass local accessibility");
 auto tampered=sharedPlan;tampered.memories[1].accessibleResources={"accelerator"};
 require(!checkProgramDataOrganization(a,arch,sharedChoice,tampered,error),"access contract tamper rejected independently");
 // Both adapters now produce the canonical ProgramGraph, while their finite
 // source-domain/copy-policy checks remain independent.
 require(retained.instances.size()==3 && retained.source.function==a.function &&
   retained.executionRegions.size()==1 && retained.executionRegions.front().submissionCount==0 &&
   checkProgramGraph(retained,error),"whole-task result is a checked common source/instance graph");
 reject("common allocation lifetime",[](auto &p){p.allocationLifetime="phase-pool";});
 reject("common compute clock",[](auto &p){p.clockMHz=1000;});
 reject("common source binding",[](auto &p){p.source.bodyFingerprint="not-the-source";});
 reject("common instance domain",[](auto &p){--p.instances.front().domain.front().upper;});
 reject("common instance implementation",[](auto &p){p.instances.front().implementation="invented";});
 reject("common instance missing",[](auto &p){p.instances.pop_back();});
 reject("host graph fabricated submission",[](auto &p){p.executionRegions.front().submissionCount=1;});
 reject("common transfer bytes",[](auto &p){for(auto &e:p.events)if(e.kind=="transfer"){++e.transferBytes;break;}});
 reject("common transfer region",[](auto &p){for(auto &e:p.events)if(e.kind=="transfer"){--e.transferRegion->sizes.front();break;}});
 reject("common schedule missing event",[](auto &p){p.schedule.events.pop_back();});
 // Twelve logical bytes are allocated with a target-owned16-byte quantum.
 // Padding consumes memory but does not become fabricated DMA traffic.
 std::string alignedSource=source;size_t at=0;
 while((at=alignedSource.find("16",at))!=std::string::npos){alignedSource.replace(at,2,"3");++at;}
 auto alignedModule=mlir::parseSourceString<mlir::ModuleOp>(alignedSource,&context);GenericAnalysis alignedFacts;
 require(alignedModule && analyzeGenericProgram(*alignedModule,alignedFacts,error),"aligned small actual source");
 auto alignedTarget=target;alignedTarget.memories.back().alignmentBytes=16;
 ProgramCommunicationPlan alignedPlan;
 require(planProgramDataOrganization(alignedFacts,alignedTarget,choice,alignedPlan,error) && alignedPlan.transferBytes==12 &&
   alignedPlan.peakMemoryBytes["sram"]==48,"alignment rounds allocations without rounding transfer payload");
 bool alignedAllocations=true;
 for(const auto &allocation:alignedPlan.allocations)if(allocation.memory=="sram")alignedAllocations&=allocation.bytes==16;
 require(alignedAllocations && checkProgramDataOrganization(alignedFacts,alignedTarget,choice,alignedPlan,error),"independent checker accepts target-aligned allocations");
 auto unpadded=alignedPlan;
 for(auto &allocation:unpadded.allocations)if(allocation.memory=="sram"){allocation.bytes=12;break;}
 require(!checkProgramDataOrganization(alignedFacts,alignedTarget,choice,unpadded,error),"independent checker rejects logical-size unpadded allocation");
 auto alignedSmall=alignedTarget;alignedSmall.memories.back().capacityBytes=47;
 require(!planProgramDataOrganization(alignedFacts,alignedSmall,choice,pathPlan,error),"alignment padding can exceed otherwise sufficient logical capacity");
 StaticConfig alignedRoundtrip;
 require(parseMemoryArchitecture(serializeMemoryArchitecture(alignedTarget),alignedRoundtrip,error) &&
   alignedRoundtrip.memories.back().alignmentBytes==16,"nondefault alignment roundtrip");
 auto invalidAlignment=alignedTarget;invalidAlignment.memories.back().alignmentBytes=0;
 require(!validateMemoryArchitecture(invalidAlignment,error),"zero alignment invalid");
 invalidAlignment.memories.back().alignmentBytes=-1;
 require(!validateMemoryArchitecture(invalidAlignment,error),"negative alignment invalid");
 auto architectureJSON=serializeMemoryArchitecture(alignedTarget);
 (*architectureJSON.getAsObject()->getArray("memories"))[1].getAsObject()->operator[]("alignment_bytes")="sixteen";
 require(!parseMemoryArchitecture(architectureJSON,alignedRoundtrip,error),"noninteger alignment invalid");
 auto defaultArchitecture=serializeMemoryArchitecture(target);
 require(!(*defaultArchitecture.getAsObject()->getArray("memories"))[1].getAsObject()->get("alignment_bytes"),"default alignment preserves old architecture serialization");
 llvm::outs()<<llvm::json::Object{{"status",ok?"PASS":"FAIL"},{"checks",int64_t(checks)},
  {"scope","actual MLIR fanout on synthetic CPU target; no cross-stage AIR claim"},
  {"architecture",serializeMemoryArchitecture(arch)},{"shared_memory_example",serializeProgramCommunication(sharedPlan)},{"separate_local_example",serializeProgramCommunication(separatePlan)},
  {"separate_local_phase_policy_example",serializeProgramCommunication(separateLocalPlan)},
  {"source_facts",serializeGenericAnalysis(a)},{"phase_local",serializeProgramCommunication(local)},{"retain_across_users",serializeProgramCommunication(retained)}}<<"\n";
 return ok?0:1;
}
