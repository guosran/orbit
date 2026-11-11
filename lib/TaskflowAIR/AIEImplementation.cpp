#include "Taskflow/AIRBackend.h"
#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/ProgramTargetBinding.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <functional>
#include <set>
using namespace mlir;
using namespace orbit::taskflow;
namespace {
struct ValueLess { bool operator()(Value a,Value b) const {return std::less<const void *>{}(a.getAsOpaquePointer(),b.getAsOpaquePointer());} };
template<class T> T field(Operation *op,StringRef key){if(auto value=op->getAttrOfType<T>(key))return value;if(auto props=dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute()))return props.getAs<T>(key);return {};}
bool named(Operation *op,StringRef name){return op->getName().getStringRef()==name;}
int64_t integer(Operation *op,StringRef key,int64_t fallback=-1){auto attr=field<IntegerAttr>(op,key);return attr?attr.getInt():fallback;}
Value root(Value v){while(auto *op=v.getDefiningOp()){if(!isa<memref::ReinterpretCastOp,memref::CastOp>(op))break;v=op->getOperand(0);}return v;}
std::vector<std::set<size_t>> closure(std::vector<std::set<size_t>> edges){bool changed=true;while(changed){changed=false;for(auto &parents:edges){auto old=parents;for(auto p:old)parents.insert(edges[p].begin(),edges[p].end());changed|=parents!=old;}}return edges;}
}
// Checked subset: one scalar pointwise core per configuration, static rectangular
// tiles, one cyclic BD per boundary stream and complete runtime output awaits.
// No kernel/task/storage names, fixed task count, tensor size or arithmetic
// equation occurs in this extractor. Unknown topology fails closed.
bool orbit::taskflow::checkGenericAIEImplementation(const GenericAnalysis &a,
    const GenericMappingPlan &plan,ModuleOp module,llvm::json::Value &receipt,
    std::string &error) {
 auto reject=[&](const std::string &why){error="AIE implementation contract: "+why;return false;};
 if(!checkGenericMapping(a,plan,error))return false;
 if(failed(verify(module)))return reject("actual generic AIE contains invalid registered operations or SSA wiring");
 if(plan.strategy!="serial-tiled")return reject("only explicit serial-tiled groups are certified");
 for(const auto &task:a.tasks)for(const auto &access:task.accesses) {
   SmallVector<Value> indices;AffineMap map;
   if(auto load=dyn_cast<affine::AffineLoadOp>(access.operation)){map=load.getAffineMap();llvm::append_range(indices,load.getMapOperands());}
   else if(auto store=dyn_cast<affine::AffineStoreOp>(access.operation)){map=store.getAffineMap();llvm::append_range(indices,store.getMapOperands());}
   else if(auto load=dyn_cast<memref::LoadOp>(access.operation))llvm::append_range(indices,load.getIndices());
   else if(auto store=dyn_cast<memref::StoreOp>(access.operation))llvm::append_range(indices,store.getIndices());
   else return reject("unsupported non-scalar source access");
   if((map && !map.isIdentity()) || indices.size()!=task.loops.size())return reject("source address map is outside certified rectangular identity-grid subset");
   for(size_t d=0;d<indices.size();++d)if(indices[d]!=task.loops[d].induction)return reject("source access axis does not match its checked spatial loop");
 }

 const auto &g=plan.communication;size_t n=g.events.size();
 std::vector<std::set<size_t>> actual(n),declared(n);
 for(size_t i=0;i<n;++i)declared[i].insert(g.events[i].dependencies.begin(),g.events[i].dependencies.end());
 auto edge=[&](size_t p,size_t c){if(p!=c)actual[c].insert(p);};
 std::map<std::string,Operation *> devices;
 bool invalidWrapper=false;
 std::vector<std::pair<std::string,std::string>> configurations;
 module.walk([&](Operation *op){
  if(named(op,"aie.device"))if(auto s=field<StringAttr>(op,"sym_name"))devices[s.str()]=op;
  if(named(op,"aiex.configure")){
   auto symbol=field<FlatSymbolRefAttr>(op,"device");
   if(!symbol)symbol=field<FlatSymbolRefAttr>(op,"symbol");
   if(!symbol)symbol=field<FlatSymbolRefAttr>(op,"device_name");
   if(!symbol)symbol=field<FlatSymbolRefAttr>(op,"target");
   std::string device=symbol?symbol.getValue().str():"";std::string sequence;
   size_t runs=0;
   op->walk([&](Operation *run){if(named(run,"aiex.run")){++runs;auto ref=field<FlatSymbolRefAttr>(run,"runtime_sequence_symbol");if(ref)sequence=ref.getValue().str();else invalidWrapper=true;
     Operation *wrapper=op->getParentOp();if(!wrapper || !named(wrapper,"aie.runtime_sequence") || wrapper->getNumRegions()!=1 || !llvm::hasSingleElement(wrapper->getRegion(0)) || run->getNumOperands()!=wrapper->getRegion(0).front().getNumArguments()){invalidWrapper=true;return;}
     for(size_t i=0;i<run->getNumOperands();++i){auto arg=dyn_cast<BlockArgument>(run->getOperand(i));if(!arg || arg.getOwner()!=&wrapper->getRegion(0).front() || arg.getArgNumber()!=i)invalidWrapper=true;}
   }});
   if(runs!=1)invalidWrapper=true;
   // Attribute spelling belongs to installed AIE; require exactly one symbol
   // reference rather than guessing from a declaration string.
   if(device.empty())for(auto attr:op->getAttrs())if(auto ref=dyn_cast<FlatSymbolRefAttr>(attr.getValue()))device=ref.getValue().str();
   configurations.emplace_back(device,sequence);
  }
 });
 if(invalidWrapper)return reject("actual configure/run must execute once with unchanged complete caller ABI forwarding");
 if(configurations.size()!=g.executionGroups.size())return reject("configuration count differs from common execution groups");
 std::set<size_t> seenGroups;std::vector<size_t> priorTerminal;llvm::json::Array groupReceipts;
 for(auto &[symbol,sequence]:configurations){
  auto found=devices.find(symbol);if(found==devices.end())return reject("configure references missing actual device");
  Operation *device=found->second,*core=nullptr,*runtime=nullptr;size_t cores=0;
  device->walk([&](Operation *op){if(named(op,"aie.core")){core=op;++cores;}if(named(op,"aie.runtime_sequence")){auto s=field<StringAttr>(op,"sym_name");if(s && s.getValue()==sequence)runtime=op;}});
  if(cores!=1 || !runtime)return reject("requires one actual core and bound runtime sequence per configuration");
  std::map<size_t,std::map<size_t,Value>> phaseBuffers;std::map<size_t,std::set<size_t>> actualReads,actualWrites;
  std::map<size_t,Value> storageBuffers;llvm::DenseMap<Value,size_t> bufferStorage;
  std::set<size_t> phases;std::vector<size_t> actualPhaseOrder;Operation *firstView=nullptr;std::map<std::string,size_t> actualArithmetic;bool bad=false;size_t position=0,firstMemory=SIZE_MAX,lastMemory=0;
  std::map<Value,std::vector<std::pair<size_t,int64_t>>,ValueLess> coreLocks;
  std::vector<Operation *> coreLockUses;
  std::map<size_t,size_t> phaseLastMemory,storageLastMemory;
  core->walk<WalkOrder::PreOrder>([&](Operation *op){
   ++position;
   if(op->getNumResults() && isa<FloatType>(op->getResult(0).getType()) &&
       !named(op,"arith.constant") && (op->getName().getStringRef().starts_with("arith.") || op->getName().getStringRef().starts_with("math.")))
     ++actualArithmetic[op->getName().getStringRef().str()];
   if(named(op,"aie.use_lock")){if(op->getNumOperands()!=2){bad=true;return;}auto amount=op->getOperand(1).getDefiningOp<arith::ConstantOp>();auto value=amount?dyn_cast<IntegerAttr>(amount.getValue()):IntegerAttr{};if(!value || value.getInt()!=1)bad=true;coreLocks[op->getOperand(0)].push_back({position,integer(op,"action")});coreLockUses.push_back(op);}
   if(auto view=dyn_cast<memref::ReinterpretCastOp>(op)){
    if(!firstView)firstView=op;
    auto phase=field<IntegerAttr>(op,"orbit.phase");auto storage=field<StringAttr>(op,"orbit.storage");
    if(!phase || !storage){bad=true;return;}auto it=llvm::find_if(a.storages,[&](const auto &s){return s.id==storage.getValue();});if(it==a.storages.end()){bad=true;return;}
    size_t si=std::distance(a.storages.begin(),it),pi=phase.getInt();if(pi>=plan.phases.size()){bad=true;return;}
    Value buffer=root(view.getSource());auto *allocation=buffer.getDefiningOp();if(!allocation || !named(allocation,"aie.buffer")){bad=true;return;}
    auto region=llvm::find_if(plan.phases[pi].regions,[&](const auto &r){return r.storage==si;});
    if(region==plan.phases[pi].regions.end() || cast<MemRefType>(view.getType()).getShape()!=ArrayRef<int64_t>(region->sizes)){bad=true;return;}
    SmallVector<int64_t> denseStrides(region->sizes.size(),1);for(int64_t d=region->sizes.size()-2;d>=0;--d)denseStrides[d]=denseStrides[d+1]*region->sizes[d+1];
    if(!view.getOffsets().empty() || !view.getSizes().empty() || !view.getStrides().empty() ||
       view.getStaticOffsets()!=ArrayRef<int64_t>{0} || view.getStaticSizes()!=ArrayRef<int64_t>(region->sizes) || view.getStaticStrides()!=ArrayRef<int64_t>(denseStrides)){bad=true;return;}

    if(storageBuffers.count(si) && storageBuffers[si]!=buffer){bad=true;return;}
    if(bufferStorage.count(buffer) && bufferStorage[buffer]!=si){bad=true;return;}
    storageBuffers[si]=buffer;bufferStorage[buffer]=si;phaseBuffers[pi][si]=buffer;phases.insert(pi);
   }
   Value mem;bool write=false;
   if(auto load=dyn_cast<memref::LoadOp>(op))mem=load.getMemRef();
   if(auto store=dyn_cast<memref::StoreOp>(op)){mem=store.getMemRef();write=true;}
   if(mem){firstMemory=std::min(firstMemory,position);lastMemory=position;auto *view=mem.getDefiningOp();auto phase=view?field<IntegerAttr>(view,"orbit.phase"):IntegerAttr{};
    auto b=root(mem);if(!phase || !bufferStorage.count(b)){bad=true;return;}size_t pi=phase.getInt();phaseLastMemory[pi]=position;storageLastMemory[bufferStorage[b]]=position;if(actualPhaseOrder.empty() || actualPhaseOrder.back()!=pi)actualPhaseOrder.push_back(pi);(write?actualWrites:actualReads)[phase.getInt()].insert(bufferStorage[b]);}
  });
  if(bad || phases.empty())return reject("actual core view/access does not match a supported source region and unique buffer");
  for(auto *lockUse:coreLockUses)if(lockUse->getBlock()!=firstView->getBlock())return reject("actual core lock must occur once per block outside source computation loops");
  size_t begin=*phases.begin(),gi=SIZE_MAX;
  for(size_t i=0;i<g.executionGroups.size();++i)if(g.executionGroups[i].beginPhase==begin)gi=i;
  if(gi==SIZE_MAX || !seenGroups.insert(gi).second)return reject("actual core does not identify one unique common group");
  const auto &group=g.executionGroups[gi];size_t width=phases.size();
  auto *owner=programDomain(plan.memoryBinding.architecture,group.spatialOwner);
  auto *tile=core->getNumOperands()?core->getOperand(0).getDefiningOp():nullptr;
  if(!owner || !tile || !named(tile,"aie.tile") || !owner->coordinates.count("column") || !owner->coordinates.count("row") ||
     integer(tile,"col")!=owner->coordinates.at("column") || integer(tile,"row")!=owner->coordinates.at("row"))return reject("actual core position differs from target-owned mapped resource");
  size_t actualBufferCount=0;device->walk([&](Operation *op){if(named(op,"aie.buffer"))++actualBufferCount;});
  if(actualBufferCount!=storageBuffers.size())return reject("extra unbound actual local buffer; separate nonpayload accounting required");
  size_t expectedBuffers=0;
  for(const auto &allocation:g.allocations)if(allocation.memory==group.memory && allocation.beginPhase<group.endPhase && allocation.endPhase>group.beginPhase) {
    ++expectedBuffers;auto foundBuffer=storageBuffers.find(allocation.storage);if(foundBuffer==storageBuffers.end())return reject("missing actual mapped local allocation");
    auto type=dyn_cast<MemRefType>(foundBuffer->second.getType());auto *buffer=foundBuffer->second.getDefiningOp();
    if(!type || !type.hasStaticShape() || !type.getElementType().isIntOrFloat() ||
       type.getMemorySpaceAsInt()!=2 || !type.getLayout().isIdentity() ||
       type.getElementType()!=cast<MemRefType>(a.storages[allocation.storage].root.getType()).getElementType() ||
       type.getNumElements()*((type.getElementType().getIntOrFloatBitWidth()+7)/8)!=allocation.bytes ||
       buffer->getNumOperands()!=1 || buffer->getOperand(0)!=core->getOperand(0))return reject("actual local buffer size or owner differs from common allocation");
  }
  if(expectedBuffers!=storageBuffers.size())return reject("extra or aliased actual local allocation");
  if(actualPhaseOrder.size()!=width)return reject("actual task bodies interleave or repeat out of phase order");
  for(size_t i=0;i<width;++i)if(!phases.count(begin+i) || actualPhaseOrder[i]!=begin+i)return reject("actual core task body order differs from planned consecutive pattern");
  if((group.endPhase-begin)%width)return reject("actual repeated task pattern does not partition group");
  int64_t actualBlocks=1;
  for(Operation *parent=firstView->getParentOp();parent && parent!=core;parent=parent->getParentOp())
    if(auto loop=dyn_cast<scf::ForOp>(parent)) {
      auto lower=loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
      auto upper=loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
      auto step=loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
      if(!lower || !upper || !step || step.value()<=0)return reject("dynamic/unproved repeated core block count");
      actualBlocks *= (upper.value()-lower.value()+step.value()-1)/step.value();
    }
  if(actualBlocks!=int64_t((group.endPhase-begin)/width))return reject("actual core iteration count differs from planned block coverage");
  std::map<std::string,size_t> expectedArithmetic;
  for(auto pi:phases)a.tasks[plan.phases[pi].task].operation->walk([&](Operation *op) {
    if(op->getNumResults() && isa<FloatType>(op->getResult(0).getType()) && !named(op,"arith.constant") &&
       (op->getName().getStringRef().starts_with("arith.") || op->getName().getStringRef().starts_with("math.")))
      ++expectedArithmetic[op->getName().getStringRef().str()];
  });
  if(actualArithmetic!=expectedArithmetic)return reject("actual floating arithmetic implementation differs from source task pattern");
  for(auto pi:phases){std::set<size_t> reads,writes;for(const auto &r:plan.phases[pi].regions){if(r.read)reads.insert(r.storage);if(r.write)writes.insert(r.storage);}if(reads!=actualReads[pi] || writes!=actualWrites[pi])return reject("actual loads/stores differ from source phase access set");}
  bool unknownExpression=false;
  auto unsupportedExpression=[&](const char *reason){unknownExpression=true;return std::string(reason);};
  auto expression=[&](Value value,const GenericTask *task,auto &&self,size_t depth)->std::string {
    if(depth>64)return unsupportedExpression("unsupported-depth");
    Operation *op=value.getDefiningOp();if(!op)return unsupportedExpression("unsupported-argument");
    if(auto constant=dyn_cast<arith::ConstantOp>(op)) {std::string text;llvm::raw_string_ostream os(text);constant.getValue().print(os);return "constant("+text+")";}
    if(isa<memref::LoadOp,affine::AffineLoadOp>(op)) {
      if(task){for(const auto &access:task->accesses)if(access.operation==op && access.read)return "load("+std::to_string(access.storage)+")";}
      else {Value mem=cast<memref::LoadOp>(op).getMemRef();auto buffer=root(mem);auto it=bufferStorage.find(buffer);if(it!=bufferStorage.end())return "load("+std::to_string(it->second)+")";}
      return unsupportedExpression("unsupported-load");
    }
    if(!op->getName().getStringRef().starts_with("arith.") && !op->getName().getStringRef().starts_with("math."))return unsupportedExpression("unsupported-expression");
    std::string result=op->getName().getStringRef().str()+"(";
    for(auto operand:op->getOperands())result+=self(operand,task,self,depth+1)+",";return result+")";
  };
  std::map<size_t,std::multiset<std::pair<size_t,std::string>>> expectedExpressions,actualExpressions;
  for(auto pi:phases){const auto &task=a.tasks[plan.phases[pi].task];for(const auto &access:task.accesses)if(access.write){Value value;if(auto store=dyn_cast<affine::AffineStoreOp>(access.operation))value=store.getValueToStore();else if(auto store=dyn_cast<memref::StoreOp>(access.operation))value=store.getValue();if(!value)return reject("unsupported source write expression");expectedExpressions[pi].insert({access.storage,expression(value,&task,expression,0)});}}
  core->walk([&](memref::StoreOp store){auto *view=store.getMemRef().getDefiningOp();auto phase=view?field<IntegerAttr>(view,"orbit.phase"):IntegerAttr{};auto buffer=root(store.getMemRef());if(!phase || !bufferStorage.count(buffer)){bad=true;return;}actualExpressions[phase.getInt()].insert({bufferStorage[buffer],expression(store.getValue(),nullptr,expression,0)});});
  if(unknownExpression || bad || expectedExpressions!=actualExpressions)return reject("actual scalar arithmetic/dataflow/constants differ from source stored expressions");

  // Exact scalar address validation for the declared rectangular identity-grid
  // subset. Values are affine linear forms in the local iteration coordinates;
  // no attribute can replace the actual address operands.
  using Form=std::vector<int64_t>;
  auto linear=[&](Value value,const llvm::DenseMap<Value,Form> &variables,auto &&self,size_t depth)->Form {
    auto found=variables.find(value);if(found!=variables.end())return found->second;
    if(depth>64)return {};
    if(auto c=value.getDefiningOp<arith::ConstantIndexOp>()){Form f(variables.empty()?1:variables.begin()->second.size(),0);f.back()=c.value();return f;}
    auto *op=value.getDefiningOp();if(!op || op->getNumOperands()!=2)return {};
    auto x=self(op->getOperand(0),variables,self,depth+1),y=self(op->getOperand(1),variables,self,depth+1);if(x.empty() || x.size()!=y.size())return {};
    if(isa<arith::AddIOp,arith::SubIOp>(op)){for(size_t d=0;d<x.size();++d)x[d]+=isa<arith::AddIOp>(op)?y[d]:-y[d];return x;}
    if(isa<arith::MulIOp>(op)){bool xc=true,yc=true;for(size_t d=0;d+1<x.size();++d){xc&=x[d]==0;yc&=y[d]==0;}if(!xc && !yc)return {};if(xc)std::swap(x,y);for(auto &v:x)v*=y.back();return x;}
    return {};
  };
  core->walk([&](Operation *op){
    Value mem;SmallVector<Value> indices;
    if(auto load=dyn_cast<memref::LoadOp>(op)){mem=load.getMemRef();llvm::append_range(indices,load.getIndices());}
    if(auto store=dyn_cast<memref::StoreOp>(op)){mem=store.getMemRef();llvm::append_range(indices,store.getIndices());}
    if(!mem)return;auto *view=mem.getDefiningOp();auto phase=field<IntegerAttr>(view,"orbit.phase");if(!phase){bad=true;return;}
    const auto &task=a.tasks[plan.phases[phase.getInt()].task];const auto &bounds=plan.phases[phase.getInt()].loops;SmallVector<scf::ForOp> loops;
    size_t nonUnit=0;for(const auto &bound:bounds)if(bound.upper-bound.lower>1)++nonUnit;
    for(Operation *parent=op->getParentOp();parent && loops.size()<nonUnit;parent=parent->getParentOp())if(auto loop=dyn_cast<scf::ForOp>(parent))loops.push_back(loop);
    if(loops.size()!=nonUnit || indices.size()!=task.loops.size()){bad=true;return;}std::reverse(loops.begin(),loops.end());
    llvm::DenseMap<Value,Form> variables;size_t axis=0;
    for(size_t d=0;d<bounds.size();++d){auto expected=bounds[d];if(expected.upper-expected.lower==1)continue;auto loop=loops[axis++];auto lo=loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>(),hi=loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>(),step=loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
      if(!lo || !hi || !step || step.value()!=1 || hi.value()-lo.value()!=expected.upper-expected.lower){bad=true;return;}Form f(task.loops.size()+1,0);f[d]=1;f.back()=lo.value();variables[loop.getInductionVar()]=f;
    }
    for(size_t d=0;d<indices.size();++d){auto actualIndex=linear(indices[d],variables,linear,0);if(actualIndex.size()==1 && variables.empty())actualIndex.resize(indices.size()+1,0);Form expected(indices.size()+1,0);if(bounds[d].upper-bounds[d].lower>1)expected[d]=1;if(actualIndex!=expected){bad=true;return;}}
  });
  if(bad)return reject("actual scalar address map is not the complete supported local rectangular identity grid");
  std::set<size_t> inputs,outputs;size_t dmaCount=0;
  std::map<std::pair<bool,int64_t>,size_t> coreChannels;
  std::map<size_t,size_t> inputReleaseMember;
  std::set<int64_t> lockIDs;device->walk([&](Operation *lock){if(named(lock,"aie.lock") && lock->getNumOperands()==1 && lock->getOperand(0)==core->getOperand(0))if(!lockIDs.insert(integer(lock,"lockID")).second)bad=true;});
  if(bad)return reject("duplicate physical lock resource binding");
  device->walk([&](Operation *op){if(!named(op,"aie.dma_start"))return;if(op->getNumSuccessors()<1){bad=true;return;}
   Operation *bd=nullptr,*acquire=nullptr,*release=nullptr;size_t ordinal=0,bdPosition=0,acquirePosition=0,releasePosition=0;
   Block *bdBlock=op->getSuccessor(0);
   for(auto &item:*bdBlock){++ordinal;if(named(&item,"aie.dma_bd")){if(bd)bad=true;bd=&item;bdPosition=ordinal;}if(named(&item,"aie.use_lock")){auto action=integer(&item,"action");if(action==2){if(acquire)bad=true;acquire=&item;acquirePosition=ordinal;}else if(action==1){if(release)bad=true;release=&item;releasePosition=ordinal;}else bad=true;}}
   auto *tail=bdBlock->getTerminator();if(!tail || !named(tail,"aie.next_bd") || tail->getNumSuccessors()!=1 || tail->getSuccessor(0)!=bdBlock || !(acquirePosition<bdPosition && bdPosition<releasePosition))bad=true;
   if(!bd || !acquire || !release || !bd->getNumOperands()){bad=true;return;}
   Value buffer=root(bd->getOperand(0));if(!bufferStorage.count(buffer)){bad=true;return;}size_t si=bufferStorage[buffer];int64_t direction=integer(op,"channel_dir"),channel=integer(op,"channel");if(direction<0 || direction>1 || channel<0 || channel>1){bad=true;return;}bool incoming=direction==0;
   (incoming?inputs:outputs).insert(si);++dmaCount;
   if(!coreChannels.emplace(std::make_pair(incoming,integer(op,"channel")),si).second)bad=true;
   auto type=dyn_cast<MemRefType>(buffer.getType());if(!type || integer(bd,"static_offset")!=0 || integer(bd,"static_len")!=type.getNumElements()){bad=true;return;}
   // The DMA-ready release must be consumed by core; the free/ready acquire
   // must be released by core after every real block access.
   auto *freeOrReady=acquire->getOperand(0).getDefiningOp(),*readyOrFree=release->getOperand(0).getDefiningOp();
   if(!freeOrReady || !readyOrFree || !named(freeOrReady,"aie.lock") || !named(readyOrFree,"aie.lock") ||
      freeOrReady->getNumOperands()!=1 || readyOrFree->getNumOperands()!=1 ||
      freeOrReady->getOperand(0)!=core->getOperand(0) || readyOrFree->getOperand(0)!=core->getOperand(0) ||
      integer(freeOrReady,"init")!=(incoming?1:0) || integer(readyOrFree,"init")!=(incoming?0:1))bad=true;
   for(Operation *lockUse:{acquire,release}){auto amount=lockUse->getOperand(1).getDefiningOp<arith::ConstantOp>();auto value=amount?dyn_cast<IntegerAttr>(amount.getValue()):IntegerAttr{};if(!value || value.getInt()!=1)bad=true;}
   auto take=coreLocks.find(release->getOperand(0)),give=coreLocks.find(acquire->getOperand(0));
   if(take==coreLocks.end() || give==coreLocks.end() || take->second.size()!=1 || give->second.size()!=1 || take->second[0].second!=2 || give->second[0].second!=1 || take->second[0].first>=firstMemory || give->second[0].first<=storageLastMemory[si]) {bad=true;return;}
   if(!incoming && give->second[0].first<=lastMemory){bad=true;return;}
   if(incoming) {
     size_t completedPhase=SIZE_MAX;
     for(auto pi:actualPhaseOrder)if(phaseLastMemory[pi]<give->second[0].first)completedPhase=pi;
     if(completedPhase==SIZE_MAX){bad=true;return;}
     inputReleaseMember[si]=completedPhase-begin;
   }
  });
  if(bad || dmaCount!=inputs.size()+outputs.size())return reject("actual cyclic DMA/lock/buffer coverage is unsupported or incomplete");
  const auto *residencyScope=genericResidencyScope(plan,begin);
  std::string inputPolicy=residencyScope?residencyScope->inputReleasePolicy:"group-tail";
  for(auto si:inputs) {
    size_t expectedMember=width-1;
    if(inputPolicy=="last-reader") {
      expectedMember=SIZE_MAX;for(auto pi:actualPhaseOrder)if(actualReads[pi].count(si))expectedMember=pi-begin;
    } else if(inputPolicy!="group-tail")return reject("unknown public input-release refinement");
    if(inputReleaseMember[si]!=expectedMember)return reject("actual input-free lock release differs from explicit public per-buffer release policy");
  }

  std::vector<size_t> abiStorages;
  for(size_t si=0;si<a.storages.size();++si)if(isa<BlockArgument>(a.storages[si].root))abiStorages.push_back(si);
  for(size_t si=0;si<a.storages.size();++si)if(!isa<BlockArgument>(a.storages[si].root) && llvm::any_of(g.allocations,[&](const auto &allocation){return allocation.storage==si && allocation.memory==plan.memoryBinding.hostMemory;}))abiStorages.push_back(si);
  std::map<size_t,int64_t> requestedElements;
  std::map<size_t,std::map<int64_t,size_t>> requestedCoverage,runtimeCoverage;
  auto cover=[&](size_t si,ArrayRef<int64_t> sizes,ArrayRef<int64_t> strides,int64_t offset,auto &coverage) {
    int64_t count=1;for(auto size:sizes){if(size<=0 || count>1000000/size){bad=true;return;}count*=size;}
    if(sizes.size()!=strides.size()){bad=true;return;}
    std::function<void(size_t,int64_t)> visit=[&](size_t d,int64_t address){if(d==sizes.size()){++coverage[si][address];return;}for(int64_t i=0;i<sizes[d];++i)visit(d+1,address+i*strides[d]);};visit(0,offset);
  };
  for(const auto &e:g.events)if(e.phase>=group.beginPhase && e.phase<group.endPhase && e.kind=="transfer" && e.transferRegion) {
    size_t si=e.transferRegion->storage;auto type=cast<MemRefType>(a.storages[si].root.getType());
    requestedElements[si]+=e.transferBytes/((type.getElementType().getIntOrFloatBitWidth()+7)/8);
    SmallVector<int64_t> strides(type.getRank(),1);for(int64_t d=type.getRank()-2;d>=0;--d)strides[d]=strides[d+1]*type.getDimSize(d+1);
    int64_t offset=0;for(size_t d=0;d<strides.size();++d)offset+=e.transferRegion->offsets[d]*strides[d];
    cover(si,e.transferRegion->sizes,strides,offset,requestedCoverage);
  }
  std::map<size_t,int64_t> runtimeElements;std::set<Value,ValueLess> started,awaited;std::map<Value,size_t,ValueLess> configuredStorage;
  size_t starts=0,awaits=0,lastStart=0,firstAwait=SIZE_MAX,runtimePosition=0;
  std::map<std::string,size_t> shimStreams;
  device->walk([&](Operation *allocation){if(!named(allocation,"aie.shim_dma_allocation"))return;auto symbol=field<StringAttr>(allocation,"sym_name");if(!symbol || allocation->getNumOperands()!=1){bad=true;return;}
    bool incoming=integer(allocation,"channel_dir")==1;Value shim=allocation->getOperand(0);int64_t shimChannel=integer(allocation,"channel_index");size_t matches=0,storage=0;
    device->walk([&](Operation *flow){if(!named(flow,"aie.flow") || flow->getNumOperands()!=2)return;
      Value source=flow->getOperand(0),target=flow->getOperand(1);if((incoming?source:target)!=shim || (incoming?target:source)!=core->getOperand(0))return;
      if(integer(flow,incoming?"source_channel":"dest_channel")!=shimChannel)return;
      auto key=std::make_pair(incoming,integer(flow,incoming?"dest_channel":"source_channel"));auto channel=coreChannels.find(key);if(channel!=coreChannels.end()){storage=channel->second;++matches;}
    });
    if(matches!=1 || !shimStreams.emplace(symbol.str(),storage).second)bad=true;
  });
  if(bad)return reject("actual shim-flow/core-DMA routing is missing or ambiguous");
  runtime->walk([&](Operation *op){++runtimePosition;
   if(named(op,"aie.dma_bd")) {
    if(!op->getNumOperands()){bad=true;return;}auto arg=dyn_cast<BlockArgument>(op->getOperand(0));
    if(!arg || arg.getOwner()!=&runtime->getRegion(0).front() || arg.getArgNumber()>=abiStorages.size() || integer(op,"static_offset")<0){bad=true;return;}
    size_t si=abiStorages[arg.getArgNumber()];auto length=integer(op,"static_len");if(length<=0){bad=true;return;}
    runtimeElements[si]+=length;
    auto sizes=field<DenseI64ArrayAttr>(op,"static_sizes"),strides=field<DenseI64ArrayAttr>(op,"static_strides");
    if(sizes && strides)cover(si,sizes.asArrayRef(),strides.asArrayRef(),integer(op,"static_offset"),runtimeCoverage);
    else cover(si,ArrayRef<int64_t>{length},ArrayRef<int64_t>{1},integer(op,"static_offset"),runtimeCoverage);
    Operation *configure=op->getParentOp();if(!named(configure,"aiex.dma_configure_task_for") || configure->getNumResults()!=1){bad=true;return;}auto allocation=field<FlatSymbolRefAttr>(configure,"alloc");if(!allocation || !shimStreams.count(allocation.getValue().str()) || shimStreams[allocation.getValue().str()]!=si){bad=true;return;}configuredStorage[configure->getResult(0)]=si;
   }
   if(named(op,"aiex.dma_start_task")){++starts;lastStart=runtimePosition;if(op->getNumOperands()!=1 || !started.insert(op->getOperand(0)).second)bad=true;}
   if(named(op,"aiex.dma_await_task")){++awaits;firstAwait=std::min(firstAwait,runtimePosition);if(op->getNumOperands()!=1 || !awaited.insert(op->getOperand(0)).second)bad=true;}
  });
  if(runtimeElements!=requestedElements || runtimeCoverage!=requestedCoverage)return reject("actual runtime BD regions/bytes differ from public boundary coverage");
  for(auto &[token,storage]:configuredStorage)if(!started.count(token) || (outputs.count(storage)!=bool(awaited.count(token))))bad=true;
  if(bad)return reject("runtime starts/awaits do not cover the actual configured output buffers");
  if(starts!=dmaCount || awaits!=outputs.size() || lastStart>=firstAwait)return reject("runtime does not start every stream and await every output before return");
  std::vector<size_t> oldCompute,oldOutput;std::vector<size_t> terminal;
  for(size_t block=begin;block<group.endPhase;block+=width){std::vector<size_t> compute,in,out;std::set<size_t> inStorage,outStorage;
   for(size_t ei=0;ei<n;++ei){const auto &e=g.events[ei];if(e.phase<block || e.phase>=block+width)continue;if(e.kind=="compute")compute.push_back(ei);else if(e.kind=="transfer" && e.transferRegion){if(e.link==plan.memoryBinding.inputLink){in.push_back(ei);inStorage.insert(e.transferRegion->storage);}else if(e.link==plan.memoryBinding.outputLink){out.push_back(ei);outStorage.insert(e.transferRegion->storage);}else return reject("unknown boundary link");}else return reject("unsupported non-pointwise activity");}
   if(in.size()!=inStorage.size())return reject("shared boundary input is loaded more than once in one block");
   if(compute.size()!=width || inStorage!=inputs || outStorage!=outputs)return reject("actual stream set differs from necessary public boundary transfers");
   for(size_t j=1;j<compute.size();++j)edge(compute[j-1],compute[j]);
   for(auto t:in)edge(t,compute.front());for(auto t:out)edge(compute.back(),t);
   if(!oldCompute.empty())for(auto t:in){size_t storage=g.events[t].transferRegion->storage;auto release=inputReleaseMember.find(storage);if(release==inputReleaseMember.end() || release->second>=oldCompute.size())return reject("input refill has no actual prior free-lock completion member");edge(oldCompute[release->second],t);}
   for(auto old:oldOutput)edge(old,compute.front());
   if(block==begin){for(auto old:priorTerminal){for(auto t:in)edge(old,t);edge(old,compute.front());}}
   oldCompute=compute;oldOutput=out;terminal=out.empty()?std::vector<size_t>{compute.back()}:out;
  }
  for(auto dep:group.dependencies)for(auto child:group.memberEvents)declared[child].insert(g.executionGroups[dep].memberEvents.begin(),g.executionGroups[dep].memberEvents.end());
  // Resource/configuration ordering is visible to cost as implementation edges.
  for(auto dep:group.implementationDependencies)for(auto child:group.memberEvents)declared[child].insert(g.executionGroups[dep].memberEvents.begin(),g.executionGroups[dep].memberEvents.end());
  llvm::json::Array releases;for(auto &[storage,member]:inputReleaseMember)releases.push_back(llvm::json::Object{{"storage",int64_t(storage)},{"actual_release_after_pattern_member",int64_t(member)}});
  priorTerminal=terminal;groupReceipts.push_back(llvm::json::Object{{"group",int64_t(gi)},{"actual_device",symbol},{"source_pattern_width",int64_t(width)},{"boundary_streams",int64_t(dmaCount)},{"output_awaits",int64_t(awaits)},{"public_input_release_policy",inputPolicy},{"actual_input_releases",std::move(releases)}});
 }
 auto ac=closure(actual),dc=closure(declared);llvm::json::Array violations,missing;
 for(size_t c=0;c<n;++c){for(auto p:declared[c])if(!ac[c].count(p))violations.push_back(llvm::json::Object{{"parent",int64_t(p)},{"child",int64_t(c)}});for(auto p:actual[c])if(!dc[c].count(p))missing.push_back(llvm::json::Object{{"parent",int64_t(p)},{"child",int64_t(c)}});}
 bool ok=violations.empty() && missing.empty();receipt=llvm::json::Object{{"schema","orbit.aie.implementation-contract.v19"},{"status",ok?"passed":"failed"},{"scope","Actual single-core rectangular pointwise groups, SSA buffers, cyclic DMA/lock roles, configure/run and output awaits; static causality, not hardware counters or device timing."},{"groups",std::move(groupReceipts)},{"public_dependencies_not_implemented",std::move(violations)},{"actual_constraints_missing_in_plan",std::move(missing)},{"facts",extractGenericAIRImplementation(module)}};
 if(!ok)return reject("bidirectional causal coverage differs; see receipt");return true;
}
