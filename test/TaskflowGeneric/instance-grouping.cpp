#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
using namespace orbit::taskflow;
int main(int argc,char **argv) {
  if(argc!=2&&argc!=3)return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect,mlir::affine::AffineDialect,mlir::arith::ArithDialect,mlir::math::MathDialect,mlir::memref::MemRefDialect,mlir::func::FuncDialect,mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);
  GenericAnalysis facts;std::string error;unsigned checks=0;
  auto require=[&](bool value,const char *name){if(!value)llvm::errs()<<name<<": "<<error<<"\n";else ++checks;return value;};
  if(!require(module&&analyzeGenericProgram(*module,facts,error),"source"))return 1;
  llvm::json::Value config=llvm::json::Object{{"generic_air",llvm::json::Object{{"tile_sizes",llvm::json::Array{4,128}}}}};
  auto choices=discoverGenericMappingChoices(facts);ProgramMappingChoice tile,shape;
  for(auto choice:choices){if(choice.grouping=="tile-major")tile=choice;if(choice.grouping=="shape-major")shape=choice;}
  GenericMappingPlan materialized,retained;
  if(!require(setNativeMappingChoice(*module,shape,error)&&analyzeGenericProgram(*module,facts,error)&&planGenericMapping(facts,config,materialized,error),"materialized grouping"))return 1;
  if(!require(setNativeMappingChoice(*module,tile,error)&&analyzeGenericProgram(*module,facts,error)&&planGenericMapping(facts,config,retained,error),"retained grouping"))return 1;
  const auto &materializedEvents=materialized.communication.events;
  auto materializedHas=[&](size_t e,size_t parent){return llvm::is_contained(materializedEvents[e].dependencies,parent);};
  size_t firstCore=materialized.phases[0].computationEvent,nextCore=materialized.phases[1].computationEvent;
  size_t firstExport=SIZE_MAX,nextImport=SIZE_MAX;
  for(const auto &transfer:materialized.transfers){if(transfer.phase==0&&transfer.direction=="l1_to_host")firstExport=transfer.event;if(transfer.phase==1&&transfer.direction=="host_to_l1")nextImport=transfer.event;}
  if(!require(firstExport!=SIZE_MAX&&nextImport!=SIZE_MAX&&materializedHas(nextImport,firstCore)&&!materializedHas(nextImport,firstExport),"single-task input refill waits for reader, not unrelated output drain"))return 1;
  if(!require(materializedHas(nextCore,firstCore)&&materializedHas(nextCore,firstExport),"single-task core order and output slot reuse"))return 1;
  size_t groupEnd=materialized.executionGroups[0].endPhase,lastExport=SIZE_MAX,groupImport=SIZE_MAX;
  for(const auto &transfer:materialized.transfers){if(transfer.phase==groupEnd-1&&transfer.direction=="l1_to_host")lastExport=transfer.event;if(transfer.phase==groupEnd&&transfer.direction=="host_to_l1")groupImport=transfer.event;}
  if(!require(lastExport!=SIZE_MAX&&groupImport!=SIZE_MAX&&materializedHas(groupImport,lastExport),"configuration transition retains previous complete group frontier"))return 1;
  auto materializedBad=materialized;auto &badDeps=materializedBad.communication.events[nextImport].dependencies;badDeps.push_back(firstExport);std::sort(badDeps.begin(),badDeps.end());
  if(!require(!checkProgramCommunication(facts,materializedBad,error),"reject fabricated same-group DMA drain barrier"))return 1;
  if(!require(materialized.hardwareStages.size()==3&&retained.hardwareStages.size()==1,"configuration grouping"))return 1;
  if(!require(materialized.communication.transferBytes==196608&&retained.communication.transferBytes==98304,"actual traffic reduction"))return 1;
  if(!require(retained.residentBytes==8192&&retained.communication.allocations.size()==7,"working set and host boundary"))return 1;
  for(auto storage:tile.retainedStorages){
    for(auto &allocation:retained.communication.allocations)if(!require(allocation.storage!=storage||allocation.memory!=retained.memoryBinding.hostMemory,"local-only intermediate"))return 1;
    bool shared=false;for(auto &copy:retained.communication.copies)if(retained.communication.regions[retained.communication.versions[copy.version].region].storage==storage&&copy.users.size()==2)shared=true;
    if(!require(shared,"one copy consumed twice"))return 1;
  }
  const auto &events=retained.communication.events;
  size_t b=retained.phases[1].computationEvent,c=retained.phases[2].computationEvent;
  std::vector<size_t> exports;
  size_t nextInput=SIZE_MAX;
  for(const auto &transfer:retained.transfers){if(transfer.phase<3&&transfer.direction=="l1_to_host")exports.push_back(transfer.event);if(transfer.phase==3&&transfer.direction=="host_to_l1")nextInput=transfer.event;}
  auto has=[&](size_t event,size_t dependency){return llvm::is_contained(events[event].dependencies,dependency);};
  if(!require(exports.size()==2&&exports[0]>c&&exports[1]>c&&has(c,b),"compute group precedes deferred exports"))return 1;
  if(!require(has(exports[0],c)&&has(exports[1],c),"output slots released after last compute"))return 1;
  if(!require(nextInput!=SIZE_MAX&&has(nextInput,c)&&!has(nextInput,exports[0])&&!has(nextInput,exports[1]),"input may refill after core release before output drain"))return 1;
  size_t nextCompute=retained.phases[3].computationEvent;
  if(!require(has(nextCompute,exports[0])&&has(nextCompute,exports[1]),"next core block acquires both output slots"))return 1;
  auto rejected=[&](auto mutate,const char *name){auto changed=retained;changed.planId.clear();mutate(changed);return require(!checkGenericMapping(facts,changed,error),name);};
  if(!rejected([](auto &p){p.choices.retainedStorages.clear();},"source choice binding")||
     !rejected([](auto &p){std::swap(p.phases[0],p.phases[1]);},"producer before consumers")||
     !rejected([](auto &p){p.channelAllocations.push_back({});},"channel coverage")||
     !rejected([](auto &p){p.executionGroups.front().memberEvents.clear();},"group authority")||
     !rejected([](auto &p){p.communication.copies.back().releasedAfter=0;},"lifetime"))return 1;
  if(!rejected([&](auto &p){auto &deps=p.communication.events[exports[0]].dependencies;deps.erase(std::remove(deps.begin(),deps.end(),c),deps.end());},"missing common block release")||
     !rejected([&](auto &p){auto &deps=p.communication.events[nextInput].dependencies;deps.push_back(exports[0]);std::sort(deps.begin(),deps.end());},"fabricated full-block drain before refill"))return 1;
  if(!rejected([&](auto &p){auto &deps=p.communication.events[nextCompute].dependencies;deps.erase(std::remove(deps.begin(),deps.end(),exports[0]),deps.end());},"next core must acquire every output slot")||
     !rejected([&](auto &p){auto &deps=p.communication.events[nextCompute].dependencies;deps.erase(std::remove(deps.begin(),deps.end(),b),deps.end());},"slot reuse covers nonfinal ordinal consumer"))return 1;
  auto emptyGroup=retained;emptyGroup.choices.taskGroup.clear();
  if(!require(!checkProgramCommunication(facts,emptyGroup,error)&&error.find("nonempty task group")!=std::string::npos,"standalone communication API rejects empty group"))return 1;
  auto invalidChoice=[&](ProgramMappingChoice choice,const char *name){
    GenericMappingPlan plan;return require(setNativeMappingChoice(*module,choice,error)&&analyzeGenericProgram(*module,facts,error)&&!planGenericMapping(facts,config,plan,error),name);
  };
  auto partial=tile;partial.taskGroup.pop_back();GenericMappingPlan boundary;
  if(!require(setNativeMappingChoice(*module,partial,error)&&analyzeGenericProgram(*module,facts,error)&&planGenericMapping(facts,config,boundary,error)&&boundary.residencyScopes.size()==1&&!boundary.residencyScopes.front().boundaryExports.empty(),"partial scope preserves outside-consumer export"))return 1;
  auto duplicate=tile;duplicate.retainedStorages.push_back(duplicate.retainedStorages.front());if(!invalidChoice(duplicate,"duplicate retained logical storage"))return 1;
  auto external=tile;external.retainedStorages={0};if(!invalidChoice(external,"external source cannot silently lose ABI"))return 1;
  GenericMappingPlan oversized;
  llvm::json::Value tooLarge=llvm::json::Object{{"generic_air",llvm::json::Object{{"tile_sizes",llvm::json::Array{16,128}}}}};
  if(!require(setNativeMappingChoice(*module,tile,error)&&analyzeGenericProgram(*module,facts,error)&&!planGenericMapping(facts,tooLarge,oversized,error)&&error.find("capacity")!=std::string::npos,"closed group working-set capacity"))return 1;
  if(argc==3){
    auto ordered=mlir::parseSourceFile<mlir::ModuleOp>(argv[2],&context);
    if(!require(ordered&&analyzeGenericProgram(*ordered,facts,error),"ordered source"))return 1;
    llvm::json::Value tails=llvm::json::Object{{"generic_air",llvm::json::Object{{"tile_sizes",llvm::json::Array{4,48}}}}};
    GenericMappingPlan original,regrouped;
    if(!require(planGenericMapping(facts,tails,original,error),"tail baseline"))return 1;
    ProgramMappingChoice grouped;grouped.residency="tiled-phase-local";grouped.grouping="shape-major";grouped.dataOrganization="phase-local";
    if(!require(setNativeMappingChoice(*ordered,grouped,error)&&analyzeGenericProgram(*ordered,facts,error)&&planGenericMapping(facts,tails,regrouped,error),"tail shape grouping"))return 1;
    if(!require(original.hardwareStages.size()==12&&regrouped.hardwareStages.size()==6&&original.communication.transferBytes==regrouped.communication.transferBytes&&!regrouped.communication.orderEdges.empty(),"fair materialized grouping preserves order and traffic"))return 1;
    for(auto choice:discoverGenericMappingChoices(facts))if(choice.grouping=="tile-major")grouped=choice;
    if(!require(setNativeMappingChoice(*ordered,grouped,error)&&analyzeGenericProgram(*ordered,facts,error)&&!planGenericMapping(facts,tails,regrouped,error)&&error.find("pure-order")!=std::string::npos,"pure order barrier cannot become tile-major"))return 1;
  }
  llvm::outs()<<checks<<" instance grouping checks passed\n";
  return 0;
}
