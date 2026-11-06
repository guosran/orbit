#include "Taskflow/DeviceTransferAudit.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <map>
#include <set>
#include <limits>
using namespace orbit::taskflow;
using namespace mlir;
namespace {
bool fail(std::string &e,llvm::StringRef why) {e="device transfer final-IR audit: "+why.str();return false;}
Attribute property(Operation *op,llvm::StringRef key) {
  if(auto a=op->getAttr(key))return a;
  if(auto d=dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute()))return d.get(key);
  return {};
}
std::optional<int64_t> integer(Operation *op,llvm::StringRef key) {
  if(auto a=dyn_cast_or_null<IntegerAttr>(property(op,key)))return a.getInt();
  return std::nullopt;
}
std::string symbol(Operation *op,llvm::StringRef key="sym_name") {
  if(auto a=dyn_cast_or_null<StringAttr>(property(op,key)))return a.str();
  if(auto a=dyn_cast_or_null<FlatSymbolRefAttr>(property(op,key)))return a.getValue().str();
  return {};
}
bool named(Operation *op,llvm::StringRef name) {return op && op->getName().getStringRef()==name;}
std::optional<int64_t> constant(Value v) {
  auto op=v.getDefiningOp();if(!named(op,"arith.constant"))return std::nullopt;
  return integer(op,"value");
}
int64_t elements(Value v) {
  auto t=dyn_cast<MemRefType>(v.getType());
  if(!t || !t.hasStaticShape() || !t.getElementType().isF32())return -1;
  int64_t n=1;for(auto d:t.getShape()) {
    if(d<=0 || n>INT64_MAX/d)return -1;n*=d;
  }
  return n;
}
Value bufferRoot(Value v) {
  while(auto op=v.getDefiningOp()) {
    if(named(op,"aie.buffer"))return v;
    if(!named(op,"memref.reinterpret_cast") || op->getNumOperands()!=1)return {};
    auto offsets=dyn_cast_or_null<DenseI64ArrayAttr>(property(op,"static_offsets"));
    if(!offsets || offsets.size()!=1 || offsets[0]!=0)return {};
    auto sizes=dyn_cast_or_null<DenseI64ArrayAttr>(property(op,"static_sizes"));
    auto strides=dyn_cast_or_null<DenseI64ArrayAttr>(property(op,"static_strides"));
    auto type=dyn_cast<MemRefType>(v.getType());
    if(!sizes || !strides || !type || sizes.size()!=type.getRank() || strides.size()!=sizes.size())return {};
    int64_t stride=1;
    for(int64_t d=int64_t(sizes.size())-1;d>=0;--d) {
      if(sizes[d]!=type.getDimSize(d) || sizes[d]<=0 || strides[d]!=stride || stride>INT64_MAX/sizes[d])return {};
      stride*=sizes[d];
    }
    v=op->getOperand(0);
  }
  return {};
}
bool loopBounds(Operation *op,int64_t &upper) {
  if(!named(op,"scf.for") || op->getNumOperands()!=3 || op->getNumRegions()!=1 ||
     op->getRegion(0).empty() || op->getRegion(0).front().getNumArguments()!=1)return false;
  auto l=constant(op->getOperand(0)),u=constant(op->getOperand(1)),s=constant(op->getOperand(2));
  if(!l || !u || !s || *l!=0 || *u<=0 || *s!=1)return false;upper=*u;return true;
}
std::string tileName(Value tile) {
  auto op=tile.getDefiningOp();if(!named(op,"aie.tile"))return {};
  auto col=integer(op,"col"),row=integer(op,"row");if(!col || !row)return {};
  return "tile("+std::to_string(*col)+","+std::to_string(*row)+")";
}
bool staticBD(Operation *op,int64_t n,bool external) {
  auto len=integer(op,"static_len"),offset=integer(op,"static_offset");
  if(!len || *len!=n || !offset || *offset!=0 || op->getNumOperands()!=1)return false;
  if(external) {
    auto sizes=dyn_cast_or_null<DenseI64ArrayAttr>(property(op,"static_sizes"));
    auto strides=dyn_cast_or_null<DenseI64ArrayAttr>(property(op,"static_strides"));
    if(!sizes || !strides || sizes.size()!=1 || strides.size()!=1 || sizes[0]!=n || strides[0]!=1)return false;
  }
  return true;
}
}

bool orbit::taskflow::deriveDeviceTransferBackendBinding(const GenericAnalysis &a,
    const GenericMappingPlan &p,ModuleOp module,const DeviceTransferBackendBinding &metadata,
    DeviceTransferBackendBinding &output,llvm::json::Value &witness,std::string &error) {
  if(!checkGenericMapping(a,p,error))return false;
  if(!metadata.streams.empty())return fail(error,"input metadata must not supply descriptor decisions");
  if(a.tasks.size()!=1 || a.storages.size()!=2 || p.hardwareStages.size()!=1 ||
     p.choices.dataOrganization!="phase-local")return fail(error,"only one uniform phase-local copy stage is supported");
  std::string graphId=programGraphIdentity(p.communication);
  if((!metadata.graphId.empty() && metadata.graphId!=graphId) ||
     (!metadata.nativePlanId.empty() && metadata.nativePlanId!=p.planId))return fail(error,"metadata graph/plan binding differs");
  for(auto key:{"orbit.mapping_plan","orbit.program_graph","orbit.source_fingerprint"}) {
    auto value=module->getAttrOfType<StringAttr>(key);
    std::string expected=llvm::StringRef(key)=="orbit.mapping_plan"?p.planId:
       llvm::StringRef(key)=="orbit.program_graph"?graphId:a.bodyFingerprint;
    if(!value || value.getValue()!=expected)return fail(error,"final module source/graph/plan identity missing or different");
  }
  std::vector<size_t> abi;
  auto function=mlir::func::FuncOp(a.function);
  for(auto argument:function.getArguments()) {
    auto found=std::find_if(a.storages.begin(),a.storages.end(),[&](const auto &s){return s.root==argument;});
    if(found==a.storages.end())return fail(error,"external ABI is not two actual source storages");
    abi.push_back(size_t(found-a.storages.begin()));
  }
  if(abi.size()!=2)return fail(error,"external ABI must have exactly two buffers");
  for(auto storage:abi)if(a.storages[storage].dtype!="f32")return fail(error,"f32 buffers required");
  SmallVector<Operation *> cores,configs,bds,starts,flows,allocs,runs,loads,stores,loops;
  module.walk([&](Operation *op) {
    if(named(op,"aie.core"))cores.push_back(op);
    if(named(op,"aiex.dma_configure_task_for"))configs.push_back(op);
    if(named(op,"aie.dma_bd"))bds.push_back(op);
    if(named(op,"aie.dma_start"))starts.push_back(op);
    if(named(op,"aie.flow"))flows.push_back(op);
    if(named(op,"aie.shim_dma_allocation"))allocs.push_back(op);
    if(named(op,"aiex.run"))runs.push_back(op);
    if(named(op,"aie.memtile_dma"))cores.push_back(nullptr);
  });
  if(cores.size()!=1 || !cores[0] || configs.size()!=2 || bds.size()!=4 ||
     starts.size()!=2 || flows.size()!=2 || allocs.size()!=2 || runs.size()!=1)
    return fail(error,"expected one core, two runtime tasks/core BDs/flows and one main wrapper");
  Operation *core=cores.front();
  core->walk([&](Operation *op) {
    if(named(op,"memref.load"))loads.push_back(op);
    if(named(op,"memref.store"))stores.push_back(op);
    if(named(op,"scf.for"))loops.push_back(op);
  });
  if(loads.size()!=1 || stores.size()!=1 || loops.empty() || loops.size()>2 ||
     stores[0]->getOperand(0)!=loads[0]->getResult(0))return fail(error,"actual core must be one direct load-to-store copy");
  Operation *inner=loads[0]->getParentOp();int64_t chunk=0,repetitions=1;
  if(!loopBounds(inner,chunk) || stores[0]->getParentOp()!=inner)return fail(error,"copy inner loop is not zero-based unit-step");
  if(loops.size()==2 && !loopBounds(inner->getParentOp(),repetitions))return fail(error,"outer chunk loop is not zero-based unit-step");
  if(chunk>INT64_MAX/repetitions || chunk*repetitions>INT64_MAX/4)return fail(error,"copy extent overflow");
  int64_t total=chunk*repetitions;
  auto load=loads[0],store=stores[0];
  if(load->getNumOperands()+1!=store->getNumOperands())return fail(error,"copy access ranks differ");
  Value iv=inner->getRegion(0).front().getArgument(0);
  for(unsigned i=1;i<load->getNumOperands();++i) {
    auto index=load->getOperand(i);
    if(index!=store->getOperand(i+1) ||
       (i+1==load->getNumOperands()?index!=iv:constant(index)!=std::optional<int64_t>(0)))
      return fail(error,"core copy indices are not one contiguous identity range");
  }
  if(elements(load->getOperand(0))!=chunk || elements(store->getOperand(1))!=chunk)
    return fail(error,"core copy view extent differs from chunk loop");
  Value inputBuffer=bufferRoot(load->getOperand(0)),outputBuffer=bufferRoot(store->getOperand(1));
  if(!inputBuffer || !outputBuffer || inputBuffer==outputBuffer || elements(inputBuffer)!=chunk || elements(outputBuffer)!=chunk)
    return fail(error,"core copy buffers are not distinct exact f32 chunks");
  // No hidden executable arithmetic or additional side effect in the core.
  bool coreValid=true;
  core->walk([&](Operation *op) {
    auto n=op->getName().getStringRef();
    if(op==core || n=="arith.constant" || n=="cf.br" || n=="scf.for" || n=="scf.yield" ||
       n=="aie.use_lock" || n=="memref.reinterpret_cast" || n=="memref.load" || n=="memref.store")return;
    coreValid=false;
  });
  if(!coreValid)return fail(error,"core contains computation outside the exact copy implementation");
  Operation *sequence=configs[0]->getParentOp();
  if(!named(sequence,"aie.runtime_sequence") || configs[1]->getParentOp()!=sequence ||
     sequence->getNumRegions()!=1 || sequence->getRegion(0).front().getNumArguments()!=2)
    return fail(error,"runtime tasks must share one two-argument sequence");
  auto &block=sequence->getRegion(0).front();
  llvm::DenseMap<Value,unsigned> started,awaited,freed;
  for(Operation &op:block) {
    auto name=op.getName().getStringRef();
    if(name=="aiex.dma_configure_task_for")continue;
    if(op.getNumOperands()!=1 || (name!="aiex.dma_start_task" && name!="aiex.dma_await_task" && name!="aiex.dma_free_task"))
      return fail(error,"runtime contains unsupported operations outside the two DMA tasks");
    Value task=op.getOperand(0);auto defining=task.getDefiningOp();
    if(!defining || defining->getParentOp()!=sequence || !named(defining,"aiex.dma_configure_task_for"))
      return fail(error,"runtime start/await/free is not one inspected task");
    if(name=="aiex.dma_start_task")++started[task];
    else if(name=="aiex.dma_await_task") {
      if(started.size()!=2)return fail(error,"output await precedes both DMA starts");
      ++awaited[task];
    } else {if(awaited.empty())return fail(error,"input task freed before output await");++freed[task];}
  }
  for(auto config:configs)if(config->getNumResults()!=1 || started[config->getResult(0)]!=1)
    return fail(error,"each runtime task must be started exactly once");
  auto run=runs.front();
  if(run->getNumOperands()!=2 || symbol(run,"runtime_sequence_symbol")!=symbol(sequence))
    return fail(error,"main wrapper does not call the inspected runtime sequence");
  for(unsigned i=0;i<2;++i) {
    auto argument=dyn_cast<BlockArgument>(run->getOperand(i));
    if(!argument || argument.getArgNumber()!=i || elements(argument)!=total || elements(block.getArgument(i))!=total)
      return fail(error,"main wrapper ABI permutation/extent unsupported");
  }
  DeviceTransferBackendBinding result=metadata;result.graphId=graphId;result.nativePlanId=p.planId;
  llvm::json::Array paths;
  std::set<size_t> seenStorages;
  for(Operation *config:configs) {
    auto repeat=integer(config,"repeat_count");
    if((repeat && *repeat!=0) || config->getNumRegions()!=1)return fail(error,"runtime repeat/task geometry unsupported");
    SmallVector<Operation *> taskBDs;
    config->walk([&](Operation *op){if(named(op,"aie.dma_bd"))taskBDs.push_back(op);});
    if(taskBDs.size()!=1 || !staticBD(taskBDs[0],total,true))return fail(error,"runtime task must have one whole contiguous BD");
    auto argument=dyn_cast<BlockArgument>(taskBDs[0]->getOperand(0));
    if(!argument || argument.getOwner()!=&block || argument.getArgNumber()>=abi.size())return fail(error,"runtime BD not bound to its ABI argument");
    size_t storage=abi[argument.getArgNumber()];
    if(!seenStorages.insert(storage).second)return fail(error,"duplicate runtime storage stream");
    std::string allocation=symbol(config,"alloc");Operation *shim=nullptr;
    for(auto op:allocs)if(symbol(op)==allocation) {if(shim)return fail(error,"duplicate shim allocation symbol");shim=op;}
    if(!shim || shim->getNumOperands()!=1)return fail(error,"runtime allocation symbol is unresolved");
    auto dir=integer(shim,"channel_dir"),channel=integer(shim,"channel_index");
    if(!dir || !channel || (*dir!=0 && *dir!=1) || *channel<0)return fail(error,"invalid shim direction/channel");
    bool input=*dir==1;
    auto task=config->getResult(0);
    if(input?(awaited[task]!=0 || freed[task]!=1):(awaited[task]!=1 || freed[task]!=0))
      return fail(error,"expected successful output-token await and input-task free ownership");
    if(!input) {
      auto token=dyn_cast_or_null<BoolAttr>(property(config,"issue_token"));
      if(!token || !token.getValue())return fail(error,"output runtime task must issue completion token");
    }
    if(a.storages[storage].role!=(input?"input":"output"))return fail(error,"runtime direction differs from source ABI role");
    Value buffer=input?inputBuffer:outputBuffer;Operation *coreBD=nullptr,*coreStart=nullptr;
    for(auto bd:bds)if(bd->getNumOperands()==1 && bd->getOperand(0)==buffer) {
      if(coreBD)return fail(error,"ambiguous core buffer descriptors");coreBD=bd;
    }
    if(!coreBD || !staticBD(coreBD,chunk,false))return fail(error,"core BD length differs from actual copy loop");
    for(auto start:starts)if(start->getNumSuccessors()==2 && start->getSuccessor(0)==coreBD->getBlock()) {
      if(coreStart)return fail(error,"ambiguous core DMA start");coreStart=start;
    }
    auto coreDir=coreStart?integer(coreStart,"channel_dir"):std::nullopt;
    auto coreChannel=coreStart?integer(coreStart,"channel"):std::nullopt;
    if(!coreDir || !coreChannel || *coreChannel<0 || integer(coreStart,"repeat_count")!=std::optional<int64_t>(0) || *coreDir!=(input?0:1))return fail(error,"core DMA direction differs from copy buffer role");
    auto &last=coreBD->getBlock()->back();
    if(!named(&last,"aie.next_bd") || last.getNumSuccessors()!=1 || last.getSuccessor(0)!=coreBD->getBlock())
      return fail(error,"core chunk BD must be one self-cycling ring");
    Value coreTile=core->getOperand(0),shimTile=shim->getOperand(0);
    auto mem=coreStart->getParentOp();
    if(!named(mem,"aie.mem") || mem->getNumOperands()!=1 || mem->getOperand(0)!=coreTile)
      return fail(error,"core DMA engine is not on the copy core tile");
    if(buffer.getDefiningOp()->getOperand(0)!=coreTile || tileName(coreTile).empty() || tileName(shimTile).empty())
      return fail(error,"core buffer tile binding unsupported");
    unsigned matches=0;
    for(auto flow:flows) {
      if(flow->getNumOperands()!=2)continue;
      Value src=input?shimTile:coreTile,dst=input?coreTile:shimTile;
      if(flow->getOperand(0)==src && flow->getOperand(1)==dst &&
         integer(flow,"source_bundle")==std::optional<int64_t>(1) && integer(flow,"dest_bundle")==std::optional<int64_t>(1) &&
         integer(flow,"source_channel")==std::optional<int64_t>(input?*channel:*coreChannel) &&
         integer(flow,"dest_channel")==std::optional<int64_t>(input?*coreChannel:*channel))++matches;
    }
    if(matches!=1)return fail(error,"runtime shim and core BD have no unique DMA flow");
    DeviceTransferStreamBinding stream;stream.stage=0;stream.storage=storage;stream.direction=input?"host_to_l1":"l1_to_host";
    int64_t publicBytes=0;
    for(size_t i=0;i<p.communication.events.size();++i) {
      auto &event=p.communication.events[i];
      if(event.kind!="transfer" || !event.transferRegion || event.transferRegion->storage!=storage)continue;
      if(event.link!=(input?p.memoryBinding.inputLink:p.memoryBinding.outputLink))return fail(error,"public transfer link differs from direction");
      if(event.transferBytes<=0 || publicBytes>INT64_MAX-event.transferBytes)return fail(error,"public transfer byte overflow");
      publicBytes+=event.transferBytes;stream.publicEvents.push_back(i);
    }
    if(publicBytes!=total*4 || stream.publicEvents.size()!=size_t(repetitions))return fail(error,"public requests differ from final descriptor bytes/chunks");
    std::string rid="runtime:"+tileName(shimTile)+":"+(input?"MM2S":"S2MM")+":"+std::to_string(*channel);
    std::string cid="core:"+tileName(coreTile)+":"+(input?"S2MM":"MM2S")+":"+std::to_string(*coreChannel);
    stream.runtimeDescriptors.push_back({rid,0,total,1,4});
    stream.coreDescriptors.push_back({cid,0,chunk,repetitions,4});
    paths.push_back(llvm::json::Object{{"storage",int64_t(storage)},{"argument",int64_t(argument.getArgNumber())},
      {"allocation_symbol",allocation},{"runtime_resource",rid},{"core_resource",cid},
      {"runtime_task_repeat_count",repeat.value_or(0)},{"core_transactions",repetitions}});
    result.streams.push_back(std::move(stream));
  }
  std::sort(result.streams.begin(),result.streams.end(),[](const auto &x,const auto &y){return x.storage<y.storage;});
  DeviceTransferFacts facts;if(!collectDeviceTransferFacts(a,p,result,facts,error))return false;
  std::string text;llvm::raw_string_ostream print(text);module.print(print);print.flush();
  witness=llvm::json::Object{{"schema","orbit.device-transfer.final-ir-audit.v15"},{"graph_id",graphId},
    {"native_plan_id",p.planId},{"canonical_ir_id",staticFingerprint(llvm::json::Value(text))},
    {"elements",total},{"tile_elements",chunk},{"segments",repetitions},{"streams",std::move(paths)},
    {"physical_channel_overlap","possible; not measured or prohibited by serial logical phase order"},
    {"resource_scope","public DMA resource is an abstract provider, not proof of one physical engine"},
    {"scope","finite uniform copy geometry/connectivity; caller binds actual file/compile chain/ELF; not device traffic or trace"}};
  output=std::move(result);return true;
}
