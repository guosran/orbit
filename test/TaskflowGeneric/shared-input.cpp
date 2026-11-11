#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <set>
using namespace orbit::taskflow;
int main(int argc,char **argv) {
  if(argc!=2)return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect,mlir::affine::AffineDialect,mlir::arith::ArithDialect,mlir::math::MathDialect,mlir::memref::MemRefDialect,mlir::func::FuncDialect,mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);
  GenericAnalysis facts;std::string error;unsigned checks=0;
  auto require=[&](bool value,const char *name){if(!value)llvm::errs()<<name<<": "<<error<<"\n";else ++checks;return value;};
  if(!require(module&&analyzeGenericProgram(*module,facts,error),"source"))return 1;
  size_t input=SIZE_MAX;for(size_t s=0;s<facts.storages.size();++s)if(facts.storages[s].role=="input")input=s;
  ProgramMappingChoice choice;choice.residency="tiled-phase-local";choice.grouping="tile-major";choice.dataOrganization="retain-across-users";
  ProgramResidencyGroup group;for(size_t t=0;t<facts.tasks.size();++t)group.tasks.push_back(t);group.retainedStorages={input};choice.residencyGroups.push_back(group);
  llvm::json::Value config=llvm::json::Object{{"generic_air",llvm::json::Object{{"tile_sizes",llvm::json::Array{2,8}}}}};
  GenericMappingPlan plan;
  if(!require(setNativeMappingChoice(*module,choice,error)&&analyzeGenericProgram(*module,facts,error)&&planGenericMapping(facts,config,plan,error),"shared external plan"))return 1;
  if(!require(plan.residencyScopes.front().boundaryExports.empty()&&plan.residencyScopes.front().inputReleasePolicy=="last-reader","immutable input has no export and explicit final-reader release"))return 1;
  size_t blocks=plan.phases.size()/group.tasks.size(),imports=0;
  for(const auto &transfer:plan.transfers)if(transfer.storage==input&&transfer.direction=="host_to_l1")++imports;
  if(!require(imports==blocks,"one input transfer per complete or tail block"))return 1;
  const auto &graph=plan.communication;
  auto depends=[&](size_t event,size_t parent) {
    std::set<size_t> seen;std::vector<size_t> work{event};
    while(!work.empty()){size_t current=work.back();work.pop_back();if(current==parent)return true;if(!seen.insert(current).second)continue;for(auto dep:graph.events[current].dependencies)work.push_back(dep);}
    return false;
  };
  for(size_t begin=0;begin<plan.phases.size();begin+=group.tasks.size()) {
    auto first=plan.phases[begin].computationEvent,last=plan.phases[begin+group.tasks.size()-1].computationEvent;
    std::vector<size_t> x,y;
    for(auto copy:graph.events[first].reads)if(graph.regions[graph.versions[graph.copies[copy].version].region].storage==input)x.push_back(copy);
    for(auto copy:graph.events[last].reads)if(graph.regions[graph.versions[graph.copies[copy].version].region].storage==input)y.push_back(copy);
    if(!require(!x.empty()&&x==y,"all actual member reads share identical version/copy"))return 1;
    if(begin+group.tasks.size()<plan.phases.size()) {
      size_t next=SIZE_MAX;for(const auto &transfer:plan.transfers)if(transfer.phase==begin+group.tasks.size()&&transfer.storage==input&&transfer.direction=="host_to_l1")next=transfer.event;
      if(!require(next!=SIZE_MAX&&depends(next,size_t(last)),"slot refill waits for final reader"))return 1;
    }
  }
  auto reject=[&](auto mutate,const char *name){auto bad=plan;bad.planId.clear();mutate(bad);return require(!checkGenericMapping(facts,bad,error),name);};
  if(!reject([&](auto &p){auto transfer=*std::find_if(p.transfers.begin(),p.transfers.end(),[&](const auto &t){return t.storage==input&&t.direction=="host_to_l1";});transfer.phase++;p.transfers.push_back(transfer);},"reject duplicate shared delivery")||
     !reject([&](auto &p){p.transfers.erase(std::remove_if(p.transfers.begin(),p.transfers.end(),[&](const auto &t){return t.storage==input&&t.direction=="host_to_l1";}),p.transfers.end());},"reject missing input delivery")||
     !reject([](auto &p){p.residencyScopes.front().inputReleasePolicy="group-tail";},"reject fabricated release policy")||
     !reject([&](auto &p){auto next=std::find_if(p.transfers.begin(),p.transfers.end(),[&](const auto &t){return t.storage==input&&t.phase==group.tasks.size()&&t.direction=="host_to_l1";});auto &deps=p.communication.events[next->event].dependencies;auto last=p.phases[group.tasks.size()-1].computationEvent;deps.erase(std::remove(deps.begin(),deps.end(),size_t(last)),deps.end());},"reject early overwrite before last reader"))return 1;
  llvm::outs()<<checks<<" shared-input checks passed\n";
}
