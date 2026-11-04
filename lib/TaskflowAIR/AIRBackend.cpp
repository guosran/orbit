#include "Taskflow/AIRBackend.h"
#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/StaticMapping.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <map>
#include <set>
using namespace mlir;
using namespace orbit::taskflow;
namespace {
Operation *make(OpBuilder &b, StringRef name, ValueRange operands = {},
                TypeRange types = {}, bool region = false) {
  OperationState s(b.getUnknownLoc(), name);
  s.addOperands(operands);
  s.addTypes(types);
  if (region)
    s.addRegion();
  return b.create(s);
}
Operation *hierarchy(OpBuilder &b, StringRef name, StringRef symbol,
                     ValueRange sizes, ValueRange operands, bool asynchronous = false) {
  SmallVector<Value> all(sizes);
  llvm::append_range(all, operands);
  SmallVector<Type> results;
  if (asynchronous) results.push_back(OpaqueType::get(b.getStringAttr("air"), "async.token"));
  auto *op = make(b, name, all, results, true);
  op->setAttr("operandSegmentSizes",
              b.getDenseI32ArrayAttr(
                  {0, (int32_t)sizes.size(), (int32_t)operands.size()}));
  op->setAttr("sym_name", b.getStringAttr(symbol));
  auto *block = new Block;
  op->getRegion(0).push_back(block);
  for (unsigned i = 0; i < 2 * sizes.size(); ++i)
    block->addArgument(b.getIndexType(), b.getUnknownLoc());
  for (auto v : operands)
    block->addArgument(v.getType(), b.getUnknownLoc());
  return op;
}
Value dma(OpBuilder &b, Type token, Value dst, Value src, ValueRange deps,
          StringRef storage, StringRef direction) {
  SmallVector<Value> operands(deps);
  operands.push_back(dst);
  operands.push_back(src);
  auto *op = make(b, "air.dma_memcpy_nd", operands, {token});
  op->setAttr(
      "operandSegmentSizes",
      b.getDenseI32ArrayAttr({(int32_t)deps.size(), 1, 0, 0, 0, 1, 0, 0, 0}));
  for (auto prefix : {"src", "dst"}) {
    auto type =
        cast<MemRefType>((StringRef(prefix) == "src" ? src : dst).getType());
    SmallVector<int64_t> offsets(type.getRank(), 0), sizes(type.getShape()),
        strides(type.getRank(), 1);
    for (int64_t i = type.getRank() - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * sizes[i + 1];
    op->setAttr((Twine("static_") + prefix + "_offsets").str(),
                b.getDenseI64ArrayAttr(offsets));
    op->setAttr((Twine("static_") + prefix + "_sizes").str(),
                b.getDenseI64ArrayAttr(sizes));
    op->setAttr((Twine("static_") + prefix + "_strides").str(),
                b.getDenseI64ArrayAttr(strides));
  }
  op->setAttr("orbit.storage", b.getStringAttr(storage));
  op->setAttr("orbit.transfer", b.getStringAttr(direction));
  return op->getResult(0);
}

// Bind the actual serialized DMA operands to checked location-copy events.
// The public checker proves latest versions and complete region coverage; here
// the target adapter also checks the concrete BO/pool placement and capacity.
bool bindDMACopies(OpBuilder &b, const GenericAnalysis &a,
                   const GenericMappingPlan &plan, Operation *op,
                   Value external, Value local, const GenericTransfer &transfer,
                   std::string &error, bool annotate=true) {
  auto reject=[&](StringRef message){error=message.str();return false;};
  const auto &communication=plan.communication;
  if(transfer.event<0 || size_t(transfer.event)>=communication.events.size())
    return reject("AIR DMA has no checked public transfer event");
  const auto &event=communication.events[transfer.event];
  if(event.kind!="transfer" || event.link!=transfer.direction || event.reads!=transfer.sourceCopies || event.writes!=transfer.targetCopies)
    return reject("AIR DMA source/target copy IDs differ from the checked transfer event");
  Operation *allocation=local.getDefiningOp();
  while(allocation && isa<memref::CastOp,memref::ReinterpretCastOp>(allocation)) allocation=allocation->getOperand(0).getDefiningOp();
  if(!allocation || !isa<memref::AllocOp>(allocation))
    return reject("AIR DMA local operand is not a checked static pool");
  std::string key="storage-"+std::to_string(transfer.storage)+".data";
  if(auto attr=allocation->getAttrOfType<StringAttr>("orbit.tile_pool")) key=attr.str();
  auto checkCopies=[&](const std::vector<size_t> &copies,bool onHost) {
    if(copies.empty()) return false;
    Value actual=onHost?external:allocation->getResult(0);
    auto type=dyn_cast<MemRefType>(actual.getType());
    if(!type || !type.hasStaticShape() || type.getMemorySpaceAsInt()!=(onHost?0:2)) return false;
    int64_t bytes=type.getNumElements()*((type.getElementType().getIntOrFloatBitWidth()+7)/8);
    for(size_t id:copies) {
      if(id>=communication.copies.size()) return false;
      const auto &copy=communication.copies[id];
      if(copy.allocation>=communication.allocations.size() || copy.version>=communication.versions.size()) return false;
      const auto &place=communication.allocations[copy.allocation];
      const auto &version=communication.versions[copy.version];
      if(version.region>=communication.regions.size()) return false;
      const auto &cell=communication.regions[version.region];
      if(place.storage!=transfer.storage || cell.storage!=transfer.storage || place.memory!=(onHost?"system_memory_bo":"aie_l1") || place.bytes!=bytes || (!onHost && place.key!=key)) return false;
      if(place.beginPhase>transfer.phase || transfer.phase>=place.endPhase) return false;
      auto offsets=transfer.offsets;auto sizes=transfer.sizes;
      if(sizes.empty()) {offsets.assign(a.storages[transfer.storage].shape.size(),0);sizes=a.storages[transfer.storage].shape;}
      if(cell.sizes.size()!=sizes.size()) return false;
      for(size_t d=0;d<sizes.size();++d)
        if(cell.offsets[d]<offsets[d] || cell.offsets[d]+cell.sizes[d]>offsets[d]+sizes[d]) return false;
    }
    return true;
  };
  bool input=transfer.direction=="host_to_l1";
  if(!checkCopies(transfer.sourceCopies,input) || !checkCopies(transfer.targetCopies,!input))
    return reject("AIR DMA operands do not match checked copy region, memory, pool key or bytes");
  SmallVector<int64_t> reads(transfer.sourceCopies.begin(),transfer.sourceCopies.end()), writes(transfer.targetCopies.begin(),transfer.targetCopies.end());
  if(annotate) {
    op->setAttr("orbit.transfer_event",b.getI64IntegerAttr(transfer.event));
    op->setAttr("orbit.source_copies",b.getDenseI64ArrayAttr(reads));
    op->setAttr("orbit.target_copies",b.getDenseI64ArrayAttr(writes));
  } else {
    auto id=op->getAttrOfType<IntegerAttr>("orbit.transfer_event");
    auto source=op->getAttrOfType<DenseI64ArrayAttr>("orbit.source_copies");
    auto target=op->getAttrOfType<DenseI64ArrayAttr>("orbit.target_copies");
    if(!id || id.getInt()!=transfer.event || !source || source.asArrayRef()!=ArrayRef<int64_t>(reads) || !target || target.asArrayRef()!=ArrayRef<int64_t>(writes))
      return reject("AIR DMA public event/copy binding is missing or changed");
  }
  return true;
}

// A rectangular transfer uses the external tensor's physical strides and a
// densely packed local tile. Offsets are in elements, independently per side.
Value tileDMA(OpBuilder &b, Type token, Value external, Value local,
              const GenericTileRegion &r, bool input, ValueRange deps,
              StringRef storage, size_t phase, ArrayRef<Value> shifts = {},
              const GenericAnalysis *analysis=nullptr, const GenericMappingPlan *plan=nullptr, std::string *error=nullptr) {
  Value result = dma(b, token, input ? local : external,
                     input ? external : local, deps, storage,
                     input ? "host_to_l1" : "l1_to_host");
  Operation *op = result.getDefiningOp();
  for (bool ext : {false, true}) {
    StringRef prefix = (input == ext) ? "src" : "dst";
    auto type = cast<MemRefType>((ext ? external : local).getType());
    SmallVector<int64_t> strides(type.getRank(), 1);
    for (int64_t i = type.getRank() - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * type.getDimSize(i + 1);
    op->setAttr((Twine("static_") + prefix + "_offsets").str(),
                b.getDenseI64ArrayAttr(ext ? r.offsets : std::vector<int64_t>(r.sizes.size(), 0)));
    op->setAttr((Twine("static_") + prefix + "_sizes").str(), b.getDenseI64ArrayAttr(r.sizes));
    op->setAttr((Twine("static_") + prefix + "_strides").str(), b.getDenseI64ArrayAttr(strides));
  }
  if (!shifts.empty() && llvm::any_of(shifts, [](Value v) { return bool(v); })) {
    SmallVector<Value> dynamicOffsets;
    SmallVector<int64_t> offsets(r.offsets.begin(), r.offsets.end());
    for (size_t d = 0; d < shifts.size(); ++d) if (shifts[d]) {
      Value base = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), r.offsets[d]);
      dynamicOffsets.push_back(b.create<arith::AddIOp>(b.getUnknownLoc(), base, shifts[d]));
      offsets[d] = ShapedType::kDynamic;
    }
    SmallVector<Value> operands(deps);
    operands.push_back(input ? local : external);
    if (!input) llvm::append_range(operands, dynamicOffsets);
    operands.push_back(input ? external : local);
    if (input) llvm::append_range(operands, dynamicOffsets);
    op->setOperands(operands);
    int32_t n = dynamicOffsets.size();
    op->setAttr("operandSegmentSizes", b.getDenseI32ArrayAttr({(int32_t)deps.size(), 1, input ? 0 : n, 0, 0, 1, input ? n : 0, 0, 0}));
    op->setAttr(input ? "static_src_offsets" : "static_dst_offsets", b.getDenseI64ArrayAttr(offsets));
    // Dynamic offset expressions must dominate the asynchronous transfer.
    op->moveBefore(b.getInsertionBlock(), b.getInsertionPoint());
  }
  op->setAttr("orbit.phase", b.getI64IntegerAttr(phase));
  if(plan) {
    std::string direction=input?"host_to_l1":"l1_to_host";
    auto transfer=llvm::find_if(plan->transfers,[&](const auto &entry){return entry.phase==phase && entry.storage==r.storage && entry.direction==direction && entry.offsets==r.offsets && entry.sizes==r.sizes;});
    if(transfer==plan->transfers.end()) *error="AIR DMA has no exact public phase/region transfer";
    else bindDMACopies(b,*analysis,*plan,op,external,local,*transfer,*error);
  }
  return result;
}

bool emitVectorTileBody(OpBuilder &b, const GenericAnalysis &a,
                        const GenericTilePhase &phase, ArrayRef<Value> local,
                        ArrayRef<Value> loopShifts,
                        ArrayRef<std::vector<Value>> regionShifts,
                        const GenericTaskImplementation &implementation,
                        std::string &error) {
  const auto &task=a.tasks[phase.task];
  auto wrapper=b.create<scf::ExecuteRegionOp>(b.getUnknownLoc(),TypeRange{});
  auto *block=new Block();wrapper.getRegion().push_back(block);
  auto reject=[&](StringRef message){error=message.str();b.setInsertionPointAfter(wrapper);wrapper.erase();return false;};
  b.setInsertionPointToStart(block);
  IRMapping mapping;
  SmallVector<Value> vectorLocal(local.begin(),local.end());
  for(size_t i=0;i<vectorLocal.size();++i) if(vectorLocal[i]) {
    auto type=cast<MemRefType>(vectorLocal[i].getType());
    auto identity=MemRefType::get(type.getShape(),type.getElementType(),MemRefLayoutAttrInterface{},type.getMemorySpace());
    if(type!=identity) vectorLocal[i]=b.create<memref::CastOp>(wrapper.getLoc(),identity,vectorLocal[i]);
  }
  auto &body=task.operation->getRegion(0).front();
  for(size_t i=0;i<task.argumentStorages.size();++i)
    if(vectorLocal[task.argumentStorages[i]]) mapping.map(body.getArgument(i),vectorLocal[task.argumentStorages[i]]);
  for(auto &op:body.without_terminator()) b.clone(op,mapping);
  b.create<scf::YieldOp>(wrapper.getLoc());
  SmallVector<affine::AffineForOp> loops;
  wrapper.walk([&](affine::AffineForOp loop){loops.push_back(loop);});
  if(loops.size()!=task.loops.size() || implementation.parallelLoop<0 || size_t(implementation.parallelLoop)>=loops.size()) {
    return reject("independent-lane implementation requires the checked affine loop nest");
  }
  for(auto bound:phase.loops) {
    loops[bound.loop].setConstantLowerBound(bound.lower);
    loops[bound.loop].setConstantUpperBound(bound.upper);
  }
  // Upstream vectorization widens only the explicitly proved spatial loop.
  // Nested reduction loops retain their scalar induction and source order.
  llvm::DenseSet<Operation *> selected{loops[implementation.parallelLoop]};
  affine::vectorizeAffineLoops(wrapper,selected,{implementation.lanes},{0});
  loops.clear();
  wrapper.walk([&](affine::AffineForOp loop){loops.push_back(loop);});
  if(loops.size()!=task.loops.size() || loops[implementation.parallelLoop].getStepAsInt()!=implementation.lanes) {
    return reject("standard affine vectorizer cannot materialize the checked independent-lane implementation");
  }
  loops[implementation.parallelLoop]->setAttr("orbit.implementation",b.getStringAttr(implementation.kind));
  loops[implementation.parallelLoop]->setAttr("orbit.vector_lanes",b.getI64IntegerAttr(implementation.lanes));
  for(size_t i=0;i<loops.size();++i) if(!loopShifts.empty() && loopShifts[i]) {
    Value iv=loops[i].getInductionVar();
    OpBuilder shifted=OpBuilder::atBlockBegin(loops[i].getBody());
    Value global=shifted.create<arith::AddIOp>(wrapper.getLoc(),iv,loopShifts[i]);
    iv.replaceAllUsesExcept(global,global.getDefiningOp());
  }
  SmallVector<Operation *> accesses;
  wrapper.walk([&](Operation *op){
    if(isa<affine::AffineLoadOp,affine::AffineStoreOp,vector::TransferReadOp,vector::TransferWriteOp>(op)) accesses.push_back(op);
  });
  for(Operation *op:accesses) {
    OpBuilder ib(op);Value mem,value;SmallVector<Value> indices;
    auto appendAffine=[&](AffineMap map,ValueRange operands) {
      for(auto expr:map.getResults()) indices.push_back(ib.create<affine::AffineApplyOp>(op->getLoc(),AffineMap::get(map.getNumDims(),map.getNumSymbols(),expr),operands));
    };
    if(auto load=dyn_cast<affine::AffineLoadOp>(op)) {mem=load.getMemRef();appendAffine(load.getAffineMap(),load.getMapOperands());}
    else if(auto store=dyn_cast<affine::AffineStoreOp>(op)) {mem=store.getMemRef();value=store.getValueToStore();appendAffine(store.getAffineMap(),store.getMapOperands());}
    else if(auto load=dyn_cast<vector::TransferReadOp>(op)) {mem=load.getSource();llvm::append_range(indices,load.getIndices());}
    else if(auto store=dyn_cast<vector::TransferWriteOp>(op)) {mem=store.getSource();llvm::append_range(indices,store.getIndices());}
    auto region=llvm::find_if(phase.regions,[&](const auto &r){return vectorLocal[r.storage]==mem;});
    if(region==phase.regions.end()) {return reject("vectorized source access lost checked storage identity");}
    for(size_t d=0;d<indices.size();++d) {
      if(region->offsets[d]) indices[d]=ib.create<arith::SubIOp>(op->getLoc(),indices[d],ib.create<arith::ConstantIndexOp>(op->getLoc(),region->offsets[d]));
      if(!regionShifts.empty() && regionShifts[region->storage][d]) indices[d]=ib.create<arith::SubIOp>(op->getLoc(),indices[d],regionShifts[region->storage][d]);
    }
    if(auto load=dyn_cast<vector::TransferReadOp>(op)) {
      auto permutation=load.getPermutationMap();
      // An invariant read is one checked in-bounds scalar, replicated exactly.
      // Avoid load-vector/extract/broadcast: the installed target has no
      // lowering for aievec.broadcast, but supports scalar broadcast in f32.
      if(permutation.getNumResults()==1 && isa<AffineConstantExpr>(permutation.getResult(0))) {
        Value scalar=ib.create<memref::LoadOp>(op->getLoc(),mem,indices);
        auto broadcast=ib.create<vector::BroadcastOp>(op->getLoc(),load.getVectorType(),scalar);
        broadcast->setAttr("orbit.invariant_scalar_broadcast",ib.getUnitAttr());
        load.replaceAllUsesWith(broadcast.getResult());load.erase();
      } else load.getIndicesMutable().assign(indices);
    }
    else if(auto store=dyn_cast<vector::TransferWriteOp>(op)) store.getIndicesMutable().assign(indices);
    else {
      if(isa<affine::AffineLoadOp>(op)) {
        Value result=ib.create<memref::LoadOp>(op->getLoc(),mem,indices);op->getResult(0).replaceAllUsesWith(result);
      } else ib.create<memref::StoreOp>(op->getLoc(),value,mem,indices);
      op->erase();
    }
  }
  b.setInsertionPointAfter(wrapper);
  while(!isa<scf::YieldOp>(block->front())) block->front().moveBefore(wrapper);
  wrapper.erase();
  return true;
}

void emitTileBody(OpBuilder &b, const GenericAnalysis &a,
                  const GenericTilePhase &phase, ArrayRef<Value> local,
                  ArrayRef<Value> loopShifts = {},
                  ArrayRef<std::vector<Value>> regionShifts = {},
                  const GenericTaskImplementation *implementation = nullptr,
                  std::string *error = nullptr) {
  if(implementation && implementation->kind=="independent-lane-f32") {
    emitVectorTileBody(b,a,phase,local,loopShifts,regionShifts,*implementation,*error);return;
  }
  auto &task = a.tasks[phase.task];
  auto &body = task.operation->getRegion(0).front();
  IRMapping mapping;
  for (size_t i = 0; i < task.argumentStorages.size(); ++i)
    if (local[task.argumentStorages[i]])
      mapping.map(body.getArgument(i), local[task.argumentStorages[i]]);
  SmallVector<Operation *> clones;
  for (auto &op : body.without_terminator())
    clones.push_back(b.clone(op, mapping));
  llvm::DenseMap<Operation *, Operation *> correspondence;
  size_t top = 0;
  for (auto &source : body.without_terminator()) {
    SmallVector<Operation *> original, copied;
    source.walk<WalkOrder::PreOrder>([&](Operation *x) { original.push_back(x); });
    clones[top++]->walk<WalkOrder::PreOrder>([&](Operation *x) { copied.push_back(x); });
    for (size_t i = 0; i < original.size(); ++i) correspondence[original[i]] = copied[i];
  }
  // Restrict the actual source loops. Induction values retain their global
  // coordinates; only memory access indices are translated to tile origins.
  for (const auto &bound : phase.loops) {
    Operation *source = task.loops[bound.loop].operation;
    Value iv = mapping.lookup(task.loops[bound.loop].induction);
    Operation *loop = iv.getParentBlock()->getParentOp();
    if (auto l = dyn_cast<affine::AffineForOp>(loop)) {
      l.setConstantLowerBound(bound.lower);
      l.setConstantUpperBound(bound.upper);
    } else if (auto l = dyn_cast<scf::ForOp>(loop)) {
      OpBuilder lb(l);
      l.getLowerBoundMutable().assign(lb.create<arith::ConstantIndexOp>(l.getLoc(), bound.lower));
      l.getUpperBoundMutable().assign(lb.create<arith::ConstantIndexOp>(l.getLoc(), bound.upper));
    }
    if (!loopShifts.empty() && loopShifts[bound.loop]) {
      OpBuilder shiftedBuilder = OpBuilder::atBlockBegin(iv.getParentBlock());
      Value global = shiftedBuilder.create<arith::AddIOp>(loop->getLoc(), iv, loopShifts[bound.loop]);
      iv.replaceAllUsesExcept(global, global.getDefiningOp());
    }
    (void)source;
  }
  for (const auto &access : task.accesses) {
    Operation *op = correspondence.lookup(access.operation);
    auto region = llvm::find_if(phase.regions, [&](const auto &r) { return r.storage == access.storage; });
    OpBuilder ib(op);
    auto loc = op->getLoc();
    SmallVector<Value> indices;
    Value mem, value;
    if (auto l = dyn_cast<affine::AffineLoadOp>(op)) {
      mem = l.getMemRef();
      for (auto expr : l.getAffineMap().getResults()) {
        auto map = AffineMap::get(l.getAffineMap().getNumDims(), l.getAffineMap().getNumSymbols(), expr);
        indices.push_back(ib.create<affine::AffineApplyOp>(loc, map, l.getMapOperands()));
      }
    } else if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
      mem = st.getMemRef(); value = st.getValueToStore();
      for (auto expr : st.getAffineMap().getResults()) {
        auto map = AffineMap::get(st.getAffineMap().getNumDims(), st.getAffineMap().getNumSymbols(), expr);
        indices.push_back(ib.create<affine::AffineApplyOp>(loc, map, st.getMapOperands()));
      }
    } else if (auto l = dyn_cast<memref::LoadOp>(op)) {
      mem = l.getMemRef(); llvm::append_range(indices, l.getIndices());
    } else if (auto st = dyn_cast<memref::StoreOp>(op)) {
      mem = st.getMemRef(); value = st.getValue(); llvm::append_range(indices, st.getIndices());
    }
    for (size_t d = 0; d < indices.size(); ++d)
      if (region->offsets[d]) {
        Value offset = ib.create<arith::ConstantIndexOp>(loc, region->offsets[d]);
        indices[d] = ib.create<arith::SubIOp>(loc, indices[d], offset);
      }
    if (!regionShifts.empty())
      for (size_t d = 0; d < indices.size(); ++d)
        if (regionShifts[access.storage][d])
          indices[d] = ib.create<arith::SubIOp>(loc, indices[d], regionShifts[access.storage][d]);
    if (access.read) {
      Value result = ib.create<memref::LoadOp>(loc, mem, indices);
      op->getResult(0).replaceAllUsesWith(result);
    } else
      ib.create<memref::StoreOp>(loc, value, mem, indices);
    op->erase();
  }
}

} // namespace
static bool lowerGenericToAIRImpl(const GenericAnalysis &, const GenericMappingPlan &,
  OwningOpRef<ModuleOp> &, std::string &, bool,
  llvm::function_ref<LogicalResult(ModuleOp)>, bool = false);
bool orbit::taskflow::checkGenericAIRMapping(const GenericAnalysis &a,
                                             const GenericMappingPlan &plan,
                                             ModuleOp module,
                                             std::string &error) {
  auto reject = [&](StringRef reason) {
    error = reason.str();
    return false;
  };
  if (!checkGenericMapping(a, plan, error))
    return false;
  if (plan.strategy == "serial-tiled") {
    OwningOpRef<ModuleOp> expected;
    bool approximate = module->hasAttr("orbit.math_approximation");
    if (!lowerGenericToAIRImpl(a, plan, expected, error, approximate,
        [&](ModuleOp m) { PassManager pm(m.getContext()); pm.addPass(createLowerAffinePass()); return pm.run(m); }, true))
      return false;
    std::string actualText, expectedText;
    llvm::raw_string_ostream actualStream(actualText), expectedStream(expectedText);
    module.print(actualStream); expected->print(expectedStream);
    if (actualText != expectedText)
      return reject("AIR tiled operations differ from the checked source/phase/ABI/DMA schedule");
    return true;
  }
  auto identity = module->getAttrOfType<StringAttr>("orbit.mapping_plan");
  if (!identity || identity.getValue() != plan.planId)
    return reject("AIR mapping identity differs from checked plan");
  auto source = module->getAttrOfType<StringAttr>("orbit.source_fingerprint");
  if (!source || source.getValue() != a.bodyFingerprint)
    return reject("AIR source identity differs from analyzed body");
  std::map<std::string, Value> allocations;
  std::map<std::string, size_t> dmaCounts;
  std::vector<std::string> starts, ends;
  SmallVector<Operation *> taskStarts, taskEnds, inputDMAs, outputDMAs;
  Operation *herdOp = nullptr;
  SmallVector<Operation *> hierarchyOps;
  bool valid = true;
  unsigned herds = 0, segments = 0, launches = 0;
  module.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "air.herd") {
      ++herds;
      herdOp = op;
    }
    if (name == "air.herd" || name == "air.segment" || name == "air.launch")
      hierarchyOps.push_back(op);
    if (name == "air.segment")
      ++segments;
    if (name == "air.launch")
      ++launches;
    if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
      auto id = op->getAttrOfType<StringAttr>("orbit.storage");
      if (!id || !allocations.emplace(id.str(), alloc.getResult()).second)
        valid = false;
    }
    if (name == "air.wait_all") {
      if (auto id = op->getAttrOfType<StringAttr>("orbit.task")) {
        starts.push_back(id.str());
        taskStarts.push_back(op);
        auto t = llvm::find_if(a.tasks,
                               [&](const auto &t) { return t.id == id.str(); });
        auto deps =
            op->getAttrOfType<DenseI64ArrayAttr>("orbit.source_dependencies");
        if (t == a.tasks.end() || !deps ||
            deps.size() != t->dependencies.size())
          valid = false;
        else
          for (size_t i = 0; i < t->dependencies.size(); ++i)
            if (deps[i] != (int64_t)t->dependencies[i])
              valid = false;
      }
      if (auto id = op->getAttrOfType<StringAttr>("orbit.task_completion")) {
        ends.push_back(id.str());
        taskEnds.push_back(op);
      }
    }
    if (name != "air.dma_memcpy_nd")
      return;
    auto id = op->getAttrOfType<StringAttr>("orbit.storage");
    auto direction = op->getAttrOfType<StringAttr>("orbit.transfer");
    auto seg = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
    if (!id || !direction || !seg || seg.size() != 9) {
      valid = false;
      return;
    }
    auto storage = llvm::find_if(
        a.storages, [&](const auto &s) { return s.id == id.str(); });
    if (storage == a.storages.end()) {
      valid = false;
      return;
    }
    bool input = direction.getValue() == "host_to_l1";
    (input ? inputDMAs : outputDMAs).push_back(op);
    if ((input && storage->role != "input") ||
        (!input &&
         (direction.getValue() != "l1_to_host" || storage->role != "output"))) {
      valid = false;
      return;
    }
    if (seg[1] != 1 || seg[5] != 1 || seg[2] || seg[3] || seg[4] || seg[6] ||
        seg[7] || seg[8]) {
      valid = false;
      return;
    }
    Value dst = op->getOperand(seg[0]), src = op->getOperand(seg[0] + 1);
    Value local = input ? dst : src, external = input ? src : dst;
    size_t storageIndex=storage-a.storages.begin();
    auto transfer=llvm::find_if(plan.transfers,[&](const auto &entry){return entry.storage==storageIndex && entry.direction==direction.getValue();});
    OpBuilder bindingBuilder(op);
    if(transfer==plan.transfers.end() || !bindDMACopies(bindingBuilder,a,plan,op,external,local,*transfer,error,false)) valid=false;
    if (!allocations.count(id.str()) || allocations[id.str()] != local)
      valid = false;
    auto root = dyn_cast<BlockArgument>(storage->root);
    auto arg = dyn_cast<BlockArgument>(external);
    if (!root || !arg || arg.getArgNumber() != root.getArgNumber() + 4 ||
        arg.getType() != root.getType())
      valid = false;
    for (auto prefix : {"src", "dst"}) {
      auto offset = op->getAttrOfType<DenseI64ArrayAttr>(
          (Twine("static_") + prefix + "_offsets").str());
      auto sizes = op->getAttrOfType<DenseI64ArrayAttr>(
          (Twine("static_") + prefix + "_sizes").str());
      auto strides = op->getAttrOfType<DenseI64ArrayAttr>(
          (Twine("static_") + prefix + "_strides").str());
      if (!offset || !sizes || !strides ||
          offset.size() != storage->shape.size() ||
          sizes.asArrayRef() != ArrayRef<int64_t>(storage->shape) ||
          strides.size() != storage->shape.size()) {
        valid = false;
        continue;
      }
      int64_t stride = 1;
      for (int64_t i = storage->shape.size() - 1; i >= 0; --i) {
        if (offset[i] != 0 || strides[i] != stride)
          valid = false;
        stride *= storage->shape[i];
      }
    }
    ++dmaCounts[id.str() + ":" + direction.str()];
  });
  if (!valid || herds != 1 || segments != 1 || launches != 1)
    return reject("AIR storage, transfer, task dependency, or hierarchy "
                  "contract mismatch");
  if (taskStarts.size() != taskEnds.size())
    return reject("AIR task boundary coverage mismatch");
  auto sourceFunction = a.function;
  // Verify actual placement and ABI forwarding, independently of identity.
  for (Operation *op : hierarchyOps) {
    StringRef name = op->getName().getStringRef();
    bool segment = name == "air.segment";
    unsigned dims = segment ? 0 : 2;
    auto seg = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
    if (!seg || seg.size() != 3 || seg[0] != 0 || seg[1] != dims ||
        seg[2] != sourceFunction.getNumArguments() || op->getNumRegions() != 1 ||
        !llvm::hasSingleElement(op->getRegion(0)))
      return reject("AIR hierarchy placement or ABI mismatch");
    for (unsigned i = 0; i < dims; ++i) {
      auto c = op->getOperand(i).getDefiningOp<arith::ConstantIndexOp>();
      if (!c || c.value() != 1)
        return reject("AIR hierarchy must use the checked 1x1 placement");
    }
    StringRef expectedParent = name == "air.herd" ? "air.segment" :
                               segment ? "air.launch" : "func.func";
    if (op->getParentOp()->getName().getStringRef() != expectedParent)
      return reject("AIR hierarchy nesting mismatch");
    unsigned parentOffset = segment ? 4 : 0;
    for (unsigned i = 0; i < sourceFunction.getNumArguments(); ++i) {
      auto arg = dyn_cast<BlockArgument>(op->getOperand(dims + i));
      if (!arg || arg.getOwner() != op->getBlock() ||
          arg.getArgNumber() != parentOffset + i)
        return reject("AIR hierarchy external ABI forwarding mismatch");
    }
  }
  for (const auto &[id, value] : allocations)
    if (value.getDefiningOp()->getParentOp() != herdOp)
      return reject("AIR resident allocation is outside the checked herd");
  for (size_t i = 0; i < taskStarts.size(); ++i) {
    Operation *start = taskStarts[i];
    if (start->getParentOp() != herdOp || taskEnds[i]->getParentOp() != herdOp ||
        taskEnds[i]->getNumResults() != 1 || taskEnds[i]->getNumOperands() != 0)
      return reject("AIR task boundary placement mismatch");
    SmallVector<Value> expected;
    if (i)
      expected.push_back(taskEnds[i - 1]->getResult(0));
    else
      for (auto *dma : inputDMAs)
        expected.push_back(dma->getResult(0));
    if (start->getOperands() != ValueRange(expected))
      return reject("AIR task synchronization token mismatch");
  }
  for (auto *op : inputDMAs)
    if (op->getParentOp() != herdOp ||
        op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes")[0] != 0)
      return reject("AIR input DMA placement or dependencies mismatch");
  for (auto *op : outputDMAs)
    if (op->getParentOp() != herdOp || taskEnds.empty() ||
        op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes")[0] != 1 ||
        op->getOperand(0) != taskEnds.back()->getResult(0))
      return reject("AIR output DMA placement or dependencies mismatch");
  if (allocations.size() != a.storages.size())
    return reject("AIR missing or extra resident allocation");
  for (const auto &storage : a.storages) {
    auto found = allocations.find(storage.id);
    if (found == allocations.end())
      return reject("AIR resident storage missing");
    auto type = dyn_cast<MemRefType>(found->second.getType());
    auto original = cast<MemRefType>(storage.root.getType());
    auto memory = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
    if (type.getShape() != original.getShape() ||
        type.getElementType() != original.getElementType() || !memory ||
        memory.getInt() != 2)
      return reject(
          "AIR resident allocation shape, dtype or memory space mismatch");
  }
  if (dmaCounts.size() != plan.transfers.size())
    return reject("AIR transfer cardinality differs from communication plan");
  for (const auto &transfer : plan.transfers)
    if (dmaCounts[a.storages[transfer.storage].id + ":" + transfer.direction] !=
        1)
      return reject("AIR planned transfer missing or duplicated");
  if (starts.size() != plan.orderedTasks.size() || ends.size() != starts.size())
    return reject("AIR task coverage mismatch");
  for (size_t i = 0; i < starts.size(); ++i)
    if (starts[i] != a.tasks[plan.orderedTasks[i]].id || ends[i] != starts[i])
      return reject("AIR task order differs from checked schedule");
  return true;
}

namespace {
struct CheckedChannel {
  Operation *declaration = nullptr;
  size_t stage = 0, storage = 0;
  std::string direction;
  SmallVector<Operation *> local, external;
};
bool channelEndpoints(const GenericAnalysis &a, const GenericMappingPlan &plan,
                      ModuleOp module, std::vector<CheckedChannel> &channels,
                      std::string &error) {
  auto reject=[&](const std::string &message) {error=message;return false;};
  if (!checkGenericMapping(a,plan,error)) return false;
  auto source=module->getAttrOfType<StringAttr>("orbit.source_fingerprint");
  auto identity=module->getAttrOfType<StringAttr>("orbit.mapping_plan");
  if (!source || source.getValue()!=a.bodyFingerprint || !identity || identity.getValue()!=plan.planId)
    return reject("AIR channel source/plan identity differs from checked native mapping");
  std::map<std::string,Operation *> declarations;
  std::map<std::string,SmallVector<Operation *>> uses;
  bool valid=true;
  module.walk([&](Operation *op) {
    auto name=op->getName().getStringRef();
    if(name=="air.channel") {
      auto symbol=op->getAttrOfType<StringAttr>("sym_name");
      if(!symbol || !declarations.emplace(symbol.str(),op).second) valid=false;
    } else if(name=="air.channel.put" || name=="air.channel.get") {
      auto symbol=op->getAttrOfType<FlatSymbolRefAttr>("chan_name");
      if(!symbol) valid=false; else uses[symbol.getValue().str()].push_back(op);
    } else if(name=="air.dma_memcpy_nd") valid=false;
  });
  if(!valid || declarations.empty() || declarations.size()!=uses.size())
    return reject("channel legalization requires complete standard DMA-to-channel conversion");
  auto stageOf=[&](Operation *op,size_t &stage) {
    while(op && op->getName().getStringRef()!="air.launch") op=op->getParentOp();
    if(!op) return false;
    auto symbol=op->getAttrOfType<StringAttr>("sym_name");
    if(!symbol) return false;
    for(size_t i=0;i<plan.hardwareStages.size();++i) {
      std::string expected=i==0 ? "orbit_launch" : "orbit_launch_"+std::to_string(plan.hardwareStages[i].beginPhase);
      if(symbol.getValue()==expected) {stage=i;return true;}
    }
    return false;
  };
  std::function<bool(Value,size_t&)> externalStorage;
  externalStorage=[&](Value value,size_t &storage) {
    if(auto argument=dyn_cast<BlockArgument>(value)) {
      Operation *parent=argument.getOwner()->getParentOp();
      if(auto function=dyn_cast<func::FuncOp>(parent)) {
        size_t ordinal=argument.getArgNumber(),scratch=func::FuncOp(a.function).getNumArguments();
        for(size_t i=0;i<a.storages.size();++i) {
          auto root=dyn_cast<BlockArgument>(a.storages[i].root);
          size_t index=root ? root.getArgNumber() : scratch++;
          if(index==ordinal && value.getType()==a.storages[i].root.getType()) {storage=i;return true;}
        }
        return false;
      }
      auto name=parent->getName().getStringRef();
      if(name!="air.launch" && name!="air.segment" && name!="air.herd") return false;
      auto segments=parent->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
      if(!segments || segments.size()!=3 || argument.getArgNumber()<unsigned(2*segments[1])) return false;
      size_t index=segments[0]+segments[1]+argument.getArgNumber()-2*segments[1];
      return index<parent->getNumOperands() && externalStorage(parent->getOperand(index),storage);
    }
    if(auto cast=value.getDefiningOp<memref::CastOp>()) return externalStorage(cast.getSource(),storage);
    return false;
  };
  std::function<bool(Value,size_t&,Operation*&)> localStorage;
  localStorage=[&](Value value,size_t &storage,Operation *&root) {
    Operation *definition=value.getDefiningOp();
    if(!definition) return false;
    if(auto allocation=dyn_cast<memref::AllocOp>(definition)) {
      auto index=definition->getAttrOfType<IntegerAttr>("orbit.storage_index");
      if(!index || index.getInt()<0 || uint64_t(index.getInt())>=a.storages.size()) return false;
      storage=index.getInt();root=definition;return true;
    }
    if(isa<memref::ReinterpretCastOp,memref::CastOp>(definition)) {
      if(!localStorage(definition->getOperand(0),storage,root)) return false;
      if(auto id=definition->getAttrOfType<StringAttr>("orbit.storage"))
        return id.getValue()==a.storages[storage].id;
      return true;
    }
    if(definition->getName().getStringRef()=="air.execute") {
      auto result=dyn_cast<OpResult>(value);
      if(!result || !result.getResultNumber() || definition->getNumRegions()!=1 || !llvm::hasSingleElement(definition->getRegion(0))) return false;
      Operation *terminator=definition->getRegion(0).front().getTerminator();
      size_t index=result.getResultNumber()-1;
      return terminator->getName().getStringRef()=="air.execute_terminator" && index<terminator->getNumOperands() && localStorage(terminator->getOperand(index),storage,root);
    }
    return false;
  };
  auto checkedPool=[&](Operation *root,size_t storage,size_t stage) {
    auto key=root->getAttrOfType<StringAttr>("orbit.tile_pool");
    if(!key) return false;
    std::map<std::string,int64_t> expected;
    auto range=plan.hardwareStages[stage];
    std::string base="storage-"+std::to_string(storage)+".data";
    for(size_t pi=range.beginPhase;pi<range.endPhase;++pi)
      for(auto &region:plan.phases[pi].regions) if(region.storage==storage) {
        std::string name=base;
        if(region.read && !region.write) {
          name+=".input";
          for(auto n:region.sizes) name+="."+std::to_string(n);
          name+=".loops";
          for(auto bound:plan.phases[pi].loops) name+="."+std::to_string(bound.upper-bound.lower);
        }
        int64_t elements=1;for(auto n:region.sizes) elements*=n;
        expected[name]=std::max(expected[name],elements);
      }
    for(const auto &scope:plan.readOnlyRetentions) if(scope.storage==storage && range.beginPhase<=scope.beginPhase && scope.endPhase<=range.endPhase) {
      int64_t elements=1;for(auto n:scope.sizes) elements*=n;
      expected[base+".readonly-panel"]=std::max(expected[base+".readonly-panel"],elements);
    }
    auto type=cast<MemRefType>(root->getResult(0).getType());
    return expected.count(key.str()) && type.getRank()==1 && type.hasStaticShape() && type.getNumElements()==expected[key.str()] && type.getMemorySpaceAsInt()==2 && type.getElementType()==cast<MemRefType>(a.storages[storage].root.getType()).getElementType();
  };
  for(auto &[symbol,declaration]:declarations) {
    if(!uses.count(symbol)) return reject("AIR channel declaration has no paired endpoints");
    bool seenLocal=false,seenExternal=false; size_t storage=0,stage=0; std::string direction;
    SmallVector<Operation *> localEndpoints,externalEndpoints;
    for(Operation *op:uses[symbol]) {
      Value memref;
      for(Value operand:op->getOperands()) if(isa<MemRefType>(operand.getType())) {
        if(memref) return reject("AIR channel endpoint has multiple memrefs");
        memref=operand;
      }
      if(!memref) return reject("AIR channel endpoint has no memref");
      auto type=cast<MemRefType>(memref.getType());
      bool local=type.getMemorySpaceAsInt()==2;
      if(!local && type.getMemorySpaceAsInt()!=0) return reject("AIR channel endpoint memory space is unsupported");
      size_t actualStorage,actualStage; Operation *allocation=nullptr;
      if(!(local ? localStorage(memref,actualStorage,allocation) : externalStorage(memref,actualStorage)) || !stageOf(op,actualStage))
        return reject("AIR channel endpoint cannot be traced to checked storage and launch ABI");
      if(local && !checkedPool(allocation,actualStorage,actualStage))
        return reject("AIR channel local allocation differs from the checked storage template, shape or dtype");
      bool put=op->getName().getStringRef()=="air.channel.put";
      std::string actualDirection=(put!=local) ? "host_to_l1" : "l1_to_host";
      if(seenLocal || seenExternal) {
        if(storage!=actualStorage || stage!=actualStage || direction!=actualDirection)
          return reject("AIR channel paired endpoints disagree on storage, stage or direction");
      } else {storage=actualStorage;stage=actualStage;direction=actualDirection;}
      if(local) {seenLocal=true;localEndpoints.push_back(op);}
      else {seenExternal=true;externalEndpoints.push_back(op);}
    }
    if(!seenLocal || !seenExternal) return reject("AIR channel is missing a local or external endpoint");
    auto &range=plan.hardwareStages[stage];
    bool planned=llvm::any_of(plan.transfers,[&](const auto &transfer) {
      return range.beginPhase<=transfer.phase && transfer.phase<range.endPhase && transfer.storage==storage && transfer.direction==direction;
    });
    if(!planned) return reject("AIR channel has no matching checked stage transfer");
    channels.push_back({declaration,stage,storage,direction,localEndpoints,externalEndpoints});
  }
  return true;
}
// A stage contains one retained output scope. Compare the ordered static
// producer and consumer messages with actual checked source-phase transfers.
bool checkChannelStreams(const GenericMappingPlan &plan, ModuleOp module,
                         const std::vector<CheckedChannel> &channels,
                         std::string &error) {
  std::map<Operation *,size_t> order;size_t ordinal=0;
  module.walk([&](Operation *op){order[op]=ordinal++;});
  using Run=std::pair<int64_t,int64_t>;
  auto append=[](std::vector<Run> &runs,int64_t elements,int64_t count) {
    if(!runs.empty() && runs.back().first==elements) runs.back().second+=count;
    else runs.push_back({elements,count});
  };
  for(const auto &allocation:plan.channelAllocations) {
    std::vector<Operation *> locals,externals;
    for(const auto &channel:channels)
      if(channel.stage==allocation.stage && channel.storage==allocation.storage && channel.direction==allocation.direction) {
        locals.insert(locals.end(),channel.local.begin(),channel.local.end());
        externals.insert(externals.end(),channel.external.begin(),channel.external.end());
      }
    std::vector<Run> expected;int64_t messages=0,bytes=0;
    auto stage=plan.hardwareStages[allocation.stage];
    for(const auto &transfer:plan.transfers)
      if(stage.beginPhase<=transfer.phase && transfer.phase<stage.endPhase && transfer.storage==allocation.storage && transfer.direction==allocation.direction) {
        int64_t elements=1;for(auto n:transfer.sizes) elements*=n;
        append(expected,elements,1);++messages;bytes+=transfer.bytes;
      }
    if(messages!=allocation.transferCount || bytes!=allocation.bytes) {error="planned FIFO message or byte count differs from transfers";return false;}
    for(auto *endpoints:{&locals,&externals}) {
      std::sort(endpoints->begin(),endpoints->end(),[&](Operation *x,Operation *y){return order[x]<order[y];});
      std::vector<Run> actual;
      for(Operation *op:*endpoints) {
        int64_t count=1;
        for(Operation *parent=op->getParentOp();parent && parent->getName().getStringRef()!="air.launch";parent=parent->getParentOp()) {
          if(auto loop=dyn_cast<scf::ForOp>(parent)) {
            auto lower=loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
            auto upper=loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
            auto step=loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
            if(!lower || !upper || !step || step.value()<=0 || upper.value()<lower.value()) {error="FIFO repetition is not a static ordered loop";return false;}
            count*=(upper.value()-lower.value()+step.value()-1)/step.value();
          } else if(isa<scf::ParallelOp>(parent)) {error="FIFO has unresolved parallel repetition";return false;}
        }
        bool put=op->getName().getStringRef()=="air.channel.put";
        auto sizes=op->getAttrOfType<DenseI64ArrayAttr>(put?"static_src_sizes":"static_dst_sizes");
        int64_t elements=1;
        if(sizes && !sizes.empty()) for(auto n:sizes.asArrayRef()) {if(n<=0) {error="FIFO message has dynamic size";return false;}elements*=n;}
        else {
          MemRefType type;for(Value operand:op->getOperands()) if(auto t=dyn_cast<MemRefType>(operand.getType())) type=t;
          if(!type || !type.hasStaticShape()) {error="FIFO local message has no static shape";return false;}
          elements=type.getNumElements();
        }
        append(actual,elements,count);
      }
      if(actual!=expected) {error="FIFO stream order, message count or size differs from checked phase transfers";return false;}
    }
  }
  return true;
}
bool setAIRProperty(Operation *op,StringRef name,Attribute value) {
  auto properties=dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute());
  if(properties && properties.get(name)) {
    NamedAttrList attrs(properties);attrs.set(name,value);
    return succeeded(op->setPropertiesFromAttribute(attrs.getDictionary(op->getContext()),[&](){return op->emitError("invalid checked AIR property update");}));
  }
  op->setAttr(name,value);return true;
}
std::string channelFingerprint(ModuleOp module) {
  std::string value;
  llvm::raw_string_ostream stream(value);
  module.print(stream,OpPrintingFlags().printGenericOpForm());
  return orbit::taskflow::staticFingerprint(llvm::json::Value(value));
}
}
bool orbit::taskflow::legalizeGenericAIRChannels(const GenericAnalysis &a,
    const GenericMappingPlan &plan,ModuleOp module,std::string &error) {
  std::vector<CheckedChannel> channels;
  if(!channelEndpoints(a,plan,module,channels,error)) return false;
  if(module->hasAttr("orbit.channel_input_fingerprint")) {error="AIR channels are already legalized";return false;}
  for(auto &channel:channels) if(channel.declaration->hasAttr("air.tile_dma_channel")) {
    error="standard channel input must not contain unplanned tile DMA port pins";return false;
  }
  if(!checkChannelStreams(plan,module,channels,error)) return false;
  OpBuilder builder(module.getContext());
  std::string before=channelFingerprint(module);
  std::map<std::string,Operation *> merged;
  for(auto &channel:channels) if(plan.hardwareStages[channel.stage].legalizeChannels) {
    auto allocation=llvm::find_if(plan.channelAllocations,[&](const auto &entry) {return entry.stage==channel.stage && entry.storage==channel.storage && entry.direction==channel.direction;});
    if(allocation==plan.channelAllocations.end()) {error="missing planned logical FIFO";return false;}
    auto symbol=FlatSymbolRefAttr::get(module.getContext(),allocation->logicalFifo);
    for(Operation *endpoint:channel.local) if(!setAIRProperty(endpoint,"chan_name",symbol)) return false;
    for(Operation *endpoint:channel.external) if(!setAIRProperty(endpoint,"chan_name",symbol)) return false;
    auto found=merged.find(allocation->logicalFifo);
    if(found==merged.end()) {
      if(!setAIRProperty(channel.declaration,"sym_name",builder.getStringAttr(allocation->logicalFifo))) return false;
      merged[allocation->logicalFifo]=channel.declaration;
    } else {channel.declaration->erase();channel.declaration=found->second;}
  }
  channels.clear();if(!channelEndpoints(a,plan,module,channels,error)) return false;
  std::string fingerprint=channelFingerprint(module);
  for(auto &channel:channels) if(plan.hardwareStages[channel.stage].legalizeChannels) {
    auto allocation=llvm::find_if(plan.channelAllocations,[&](const auto &entry) {
      return entry.stage==channel.stage && entry.storage==channel.storage && entry.direction==channel.direction;
    });
    if(allocation==plan.channelAllocations.end()) {error="AIR channel lacks an explicit checked physical port assignment";return false;}
    channel.declaration->setAttr("air.tile_dma_channel",builder.getI32IntegerAttr(allocation->tileChannel));
  }
  module->setAttr("orbit.channel_pre_merge_fingerprint",builder.getStringAttr(before));
  module->setAttr("orbit.channel_input_fingerprint",builder.getStringAttr(fingerprint));
  module->setAttr("orbit.channel_legalization",builder.getStringAttr("checked-storage-compute-dma-ports-v1"));
  return checkGenericAIRChannels(a,plan,module,error);
}
bool orbit::taskflow::checkGenericAIRChannels(const GenericAnalysis &a,
    const GenericMappingPlan &plan,ModuleOp module,std::string &error) {
  std::vector<CheckedChannel> channels;
  if(!channelEndpoints(a,plan,module,channels,error)) return false;
  if(!checkChannelStreams(plan,module,channels,error)) return false;
  auto fingerprint=module->getAttrOfType<StringAttr>("orbit.channel_input_fingerprint");
  auto mode=module->getAttrOfType<StringAttr>("orbit.channel_legalization");
  if(!fingerprint || !mode || mode.getValue()!="checked-storage-compute-dma-ports-v1") {
    error="AIR channels lack checked standard-conversion provenance";return false;
  }
  for(auto &channel:channels) {
    auto pin=channel.declaration->getAttrOfType<IntegerAttr>("air.tile_dma_channel");
    if(plan.hardwareStages[channel.stage].legalizeChannels) {
      auto allocation=llvm::find_if(plan.channelAllocations,[&](const auto &entry) {
        return entry.stage==channel.stage && entry.storage==channel.storage && entry.direction==channel.direction;
      });
      if(allocation==plan.channelAllocations.end() || !pin || pin.getInt()!=allocation->tileChannel || channel.declaration->getAttrOfType<StringAttr>("sym_name").getValue()!=allocation->logicalFifo) {
        error="AIR channel compute DMA port differs from the checked resource plan";return false;
      }
    } else if(pin) {error="AIR channel has an unplanned compute DMA pin";return false;}
  }
  // Unregistered AIR operations carry opaque properties. MLIR's generic
  // Operation::clone does not retain those properties in this toolchain, so
  // fingerprint the original in place and restore only our added attributes.
  SmallVector<std::pair<Operation *,Attribute>> pins;
  for(auto &channel:channels) {
    Attribute value=channel.declaration->getAttr("air.tile_dma_channel");
    pins.push_back({channel.declaration,value});
    channel.declaration->removeAttr("air.tile_dma_channel");
  }
  auto before=module->getAttr("orbit.channel_pre_merge_fingerprint");
  module->removeAttr("orbit.channel_pre_merge_fingerprint");
  module->removeAttr("orbit.channel_input_fingerprint");
  module->removeAttr("orbit.channel_legalization");
  std::string actual=channelFingerprint(module);
  if(before) module->setAttr("orbit.channel_pre_merge_fingerprint",before);
  module->setAttr("orbit.channel_input_fingerprint",fingerprint);
  module->setAttr("orbit.channel_legalization",mode);
  for(auto &[operation,value]:pins) if(value) operation->setAttr("air.tile_dma_channel",value);
  if(actual!=fingerprint.getValue()) {
    error="AIR channel body or transfer changed after checked standard conversion";return false;
  }
  if(failed(verify(module))) {error="legalized AIR channel module failed MLIR verification";return false;}
  return true;
}

static bool lowerGenericToAIRImpl(const GenericAnalysis &a,
                                        const GenericMappingPlan &plan,
                                        OwningOpRef<ModuleOp> &output,
                                        std::string &error,
                                        bool approximateExp,
                                        llvm::function_ref<LogicalResult(ModuleOp)> normalize,
                                        bool skipCheck) {
  if (!checkGenericMapping(a, plan, error))
    return false;
  if(llvm::any_of(plan.taskImplementations,[](const auto &implementation){return implementation.kind=="independent-lane-f32";})) {
    error="independent-lane-f32 is unavailable on the validated AIR target: installed Peano cannot legalize strict f32 vector multiplication; BF16 emulation is not enabled";
    return false;
  }
  if (plan.strategy == "serial-tiled")
    for (size_t pi=0;pi<plan.phases.size();++pi)
      for (const auto &region:plan.phases[pi].regions)
        if (region.read && region.write && !llvm::any_of(plan.retainedRegions,[&](const auto &scope) {
              return scope.storage==region.storage && scope.beginPhase<=pi && pi<scope.endPhase && scope.initializerTask>=0;
            })) {
          error="direct AIR read/write tile requires a proved local source initializer; host feedback accumulation is not validated by this toolchain";
          return false;
        }
  auto function = a.function;
  auto *ctx = function.getContext();
  if (!ctx->allowsUnregisteredDialects()) {
    error = "direct AIR requires the caller to allow unregistered AIR dialect "
            "operations on the MLIR context before running passes";
    return false;
  }
  ctx->getOrLoadDialect<memref::MemRefDialect>();
  ctx->getOrLoadDialect<arith::ArithDialect>();
  ctx->getOrLoadDialect<scf::SCFDialect>();
  ctx->getOrLoadDialect<vector::VectorDialect>();
  OpBuilder b(ctx);
  auto loc = b.getUnknownLoc();
  for (auto &s : a.storages) {
    auto t = dyn_cast<MemRefType>(s.root.getType());
    if (!t || !t.getLayout().isIdentity() ||
        !(t.getElementType().isF32() || t.getElementType().isBF16() ||
          t.getElementType().isInteger(32))) {
      error = "direct AIR requires identity-layout f32/bf16/i32 memrefs";
      return false;
    }
  }
  for (auto &t : a.tasks) {
    auto &body = t.operation->getRegion(0).front();
    for (auto arg : body.getArguments().drop_front(t.argumentStorages.size()))
      for (auto &use : arg.getUses())
        if (use.getOwner() != body.getTerminator()) {
          error = "direct AIR does not yet lower used scalar task arguments";
          return false;
        }
  }
  output = ModuleOp::create(loc);
  b.setInsertionPointToEnd(output->getBody());
  SmallVector<Type> abiTypes(function.getArgumentTypes());
  if (plan.strategy == "serial-tiled")
    for (auto &s : a.storages)
      if (s.role == "temporary") abiTypes.push_back(s.root.getType());
  auto fn = b.create<func::FuncOp>(loc, function.getName(),
                                   b.getFunctionType(abiTypes, {}));
  auto *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  auto c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> args(entry->getArguments());
  auto *launch = hierarchy(b, "air.launch", "orbit_launch", {c1, c1}, args);
  b.setInsertionPointToStart(&launch->getRegion(0).front());
  SmallVector<Value> la(
      launch->getRegion(0).front().getArguments().drop_front(4));
  auto *segment = hierarchy(b, "air.segment", "orbit_segment", {}, la);
  b.setInsertionPointToStart(&segment->getRegion(0).front());
  auto one = b.create<arith::ConstantIndexOp>(loc, 1);
  auto *herd = hierarchy(b, "air.herd", "orbit_herd", {one, one},
                         segment->getRegion(0).front().getArguments(),
                         plan.strategy=="serial-tiled" && !plan.hardwareStages.empty() && plan.hardwareStages.front().legalizeChannels);
  herd->setAttr("orbit.mapping_strategy",
                b.getStringAttr(plan.strategy));
  b.setInsertionPointToStart(&herd->getRegion(0).front());
  auto ha = herd->getRegion(0).front().getArguments().drop_front(4);
  Type token = OpaqueType::get(b.getStringAttr("air"), "async.token");
  if (plan.strategy == "serial-tiled") {
    for (size_t stageIndex=0;stageIndex<plan.hardwareStages.size();++stageIndex) {
    size_t stageBegin=plan.hardwareStages[stageIndex].beginPhase;
    size_t stageEnd=plan.hardwareStages[stageIndex].endPhase;
    auto locallyInitialized = [&](size_t storage) {
      return llvm::any_of(plan.retainedRegions,[&](const auto &scope) {
        return scope.beginPhase>=stageBegin && scope.endPhase<=stageEnd && scope.storage==storage && scope.initializerTask>=0;
      });
    };
    SmallVector<Value> external(a.storages.size());
    size_t scratch = function.getNumArguments();
    for (size_t i = 0; i < a.storages.size(); ++i) {
      auto arg = dyn_cast<BlockArgument>(a.storages[i].root);
      external[i] = ha[arg ? arg.getArgNumber() : scratch++];
    }
    // Keep distinct storage pools in each hardware stage. Reuse them only
    // across serial tiles of that stage; host BOs carry stage boundaries.
    auto readonlyScopeOf=[&](size_t phase,size_t storage) -> const GenericReadOnlyRetention * {
      auto found=llvm::find_if(plan.readOnlyRetentions,[&](const auto &scope){return scope.storage==storage && scope.beginPhase<=phase && phase<scope.endPhase;});
      return found==plan.readOnlyRetentions.end()?nullptr:&*found;
    };
    auto storagePoolBase = [](size_t storage) { return "storage-" + std::to_string(storage) + ".data"; };
    auto storagePoolKey = [&](const GenericTileRegion &r, const GenericTilePhase &phase) {
      std::string key=storagePoolBase(r.storage);
      if (r.read && !r.write) {
        key += ".input";
        for (auto n:r.sizes) key += "." + std::to_string(n);
        key += ".loops";
        for (auto &bound:phase.loops) key += "." + std::to_string(bound.upper-bound.lower);
      }
      return key;
    };
    std::map<std::string, int64_t> poolElements;
    std::map<std::string, Type> poolTypes;
    std::map<std::string, size_t> poolStorages;
    std::map<std::string, Value> pools;
    for (size_t phaseIndex = stageBegin; phaseIndex < stageEnd; ++phaseIndex) {
      auto &phase = plan.phases[phaseIndex];
      std::map<std::string, int64_t> used;
      for (auto &r : phase.regions) {
        auto &storage = a.storages[r.storage];
        int64_t count = 1;
        for (auto n : r.sizes) count *= n;
        auto poolKey=storagePoolKey(r,phase);
        if(auto scope=readonlyScopeOf(phaseIndex,r.storage)) {
          poolKey=storagePoolBase(r.storage)+".readonly-panel";count=1;for(auto n:scope->sizes) count*=n;
        }
        used[poolKey] += count;
        poolStorages[poolKey] = r.storage;
        poolTypes[poolKey] = cast<MemRefType>(storage.root.getType()).getElementType();
        if (r.read && r.write && !locallyInitialized(r.storage)) {
          used[storagePoolBase(r.storage) + ".incoming"] += count;
          poolStorages[storagePoolBase(r.storage) + ".incoming"] = r.storage;
          poolTypes[storagePoolBase(r.storage) + ".incoming"] = poolTypes[storagePoolBase(r.storage)];
        }
      }
      for (auto &[dtype, count] : used) poolElements[dtype] = std::max(poolElements[dtype], count);
    }
    int64_t poolBytes = 0;
    for (auto &[dtype, count] : poolElements)
      poolBytes += count * ((poolTypes[dtype].getIntOrFloatBitWidth() + 7) / 8);
    if (poolBytes > plan.capacityBytes) {
      error = "AIR stage allocations exceed the checked local-memory capacity";
      return false;
    }
    for (auto &[dtype, count] : poolElements) {
      auto type = MemRefType::get({count}, poolTypes[dtype], MemRefLayoutAttrInterface{}, b.getI32IntegerAttr(2));
      auto alloc = b.create<memref::AllocOp>(loc, type);
      alloc->setAttr("orbit.tile_pool", b.getStringAttr(dtype));
      alloc->setAttr("orbit.storage_index", b.getI64IntegerAttr(poolStorages[dtype]));
      if (plan.hardwareStages[stageIndex].legalizeChannels)
        alloc->setAttr("orbit.lifetime", b.getStringAttr("static-hardware-stage-region"));
      pools[dtype] = alloc;
    }
    auto tileView = [&](const GenericTileRegion &r, size_t pi, StringRef key) -> Value {
      auto element = cast<MemRefType>(a.storages[r.storage].root.getType()).getElementType();
      SmallVector<int64_t> strides(r.sizes.size(), 1);
      for (int64_t d=r.sizes.size()-2;d>=0;--d) strides[d]=strides[d+1]*r.sizes[d+1];
      auto layout=StridedLayoutAttr::get(ctx,0,strides);
      auto type=MemRefType::get(r.sizes,element,layout,b.getI32IntegerAttr(2));
      auto view=b.create<memref::ReinterpretCastOp>(loc,type,pools[key.str()],int64_t(0),r.sizes,strides);
      view->setAttr("orbit.storage",b.getStringAttr(a.storages[r.storage].id));
      view->setAttr("orbit.phase",b.getI64IntegerAttr(pi));
      return view;
    };
    auto copyTile = [&](Value source, Value destination, const GenericTileRegion &r) {
      SmallVector<Value> indices;
      SmallVector<scf::ForOp> loops;
      for(auto size:r.sizes) {
        Value zero=b.create<arith::ConstantIndexOp>(loc,0), upper=b.create<arith::ConstantIndexOp>(loc,size), step=b.create<arith::ConstantIndexOp>(loc,1);
        auto loop=b.create<scf::ForOp>(loc,zero,upper,step);
        loop->setAttr("orbit.input_staging",b.getUnitAttr());
        loops.push_back(loop); indices.push_back(loop.getInductionVar());
        b.setInsertionPointToStart(loop.getBody());
      }
      Value value=b.create<memref::LoadOp>(loc,source,indices);
      b.create<memref::StoreOp>(loc,value,destination,indices);
      if (!loops.empty()) b.setInsertionPointAfter(loops.front());
    };
    struct Shifts { Value dependency; std::vector<Value> loops, retained; std::vector<std::vector<Value>> regions; std::vector<const GenericReadOnlyRetention *> readonly; std::vector<std::vector<Value>> readonlyShifts; };
    auto implementationOf=[&](size_t task) -> const GenericTaskImplementation * {
      auto found=llvm::find_if(plan.taskImplementations,[&](const auto &impl){return impl.task==task;});
      return found==plan.taskImplementations.end()?nullptr:&*found;
    };
    auto emitPhase = [&](size_t pi, const Shifts &shifts) {
      auto &phase = plan.phases[pi];
      SmallVector<Value> local(a.storages.size()), incoming;
      SmallVector<std::pair<Value,GenericTileRegion>> staged;
      for (auto &r : phase.regions) {
        if (shifts.retained[r.storage]) { local[r.storage]=shifts.retained[r.storage]; continue; }
        Value view=tileView(r,pi,storagePoolKey(r,phase));
        local[r.storage]=view;
        if (r.read) {
          Value input=view;
          if (r.write) {
            input=tileView(r,pi,storagePoolBase(r.storage) + ".incoming");
            staged.push_back({input,r});
          }
          incoming.push_back(tileDMA(b,token,external[r.storage],input,r,true,{shifts.dependency},a.storages[r.storage].id,pi,shifts.regions[r.storage],&a,&plan,&error));
        }
      }
      auto *start = make(b, "air.wait_all", incoming);
      start->setAttr("orbit.phase", b.getI64IntegerAttr(pi));
      SmallVector<int64_t> phaseDeps(phase.dependencies.begin(), phase.dependencies.end());
      start->setAttr("orbit.phase_dependencies", b.getDenseI64ArrayAttr(phaseDeps));
      start->setAttr("orbit.task", b.getStringAttr(a.tasks[phase.task].id));
      start->setAttr("orbit.computation_event",b.getI64IntegerAttr(phase.computationEvent));
      const auto &event=plan.communication.events[phase.computationEvent];
      SmallVector<int64_t> reads(event.reads.begin(),event.reads.end()), writes(event.writes.begin(),event.writes.end());
      start->setAttr("orbit.read_copies",b.getDenseI64ArrayAttr(reads));
      start->setAttr("orbit.write_copies",b.getDenseI64ArrayAttr(writes));
      for (auto &[input,r] : staged) copyTile(input,local[r.storage],r);
      GenericTilePhase effective=phase;
      auto effectiveShifts=shifts.regions;
      for(auto &r:effective.regions) if(shifts.readonly[r.storage]) {
        r.offsets=shifts.readonly[r.storage]->offsets;r.sizes=shifts.readonly[r.storage]->sizes;
        effectiveShifts[r.storage]=shifts.readonlyShifts[r.storage];
      }
      emitTileBody(b, a, effective, local, shifts.loops, effectiveShifts,implementationOf(phase.task),&error);
      SmallVector<Value> outgoing;
      for (auto &r : phase.regions)
        if (r.write && !shifts.retained[r.storage]) outgoing.push_back(tileDMA(b, token, external[r.storage], local[r.storage], r, false, {shifts.dependency}, a.storages[r.storage].id, pi, shifts.regions[r.storage],&a,&plan,&error));
      if (outgoing.empty()) outgoing.push_back(shifts.dependency);
      return make(b, "air.wait_all", outgoing, {token})->getResult(0);
    };
    // Compress only consecutive, exactly translation-equivalent source tiles.
    // Every candidate is compared against the checked finite schedule, so this
    // is schedule compression rather than a new mapping or arithmetic matcher.
    std::function<Value(size_t, size_t, size_t, const Shifts &)> emitRange;
    emitRange = [&](size_t begin, size_t end, size_t depth, const Shifts &inherited) -> Value {
      auto &first = plan.phases[begin];
      Shifts active=inherited;
      for(const auto &scope:plan.readOnlyRetentions)
        if(scope.beginPhase==begin && scope.endPhase==end && !active.retained[scope.storage]) {
          GenericTileRegion region{scope.storage,scope.offsets,scope.sizes,true,false};
          Value panel=tileView(region,begin,storagePoolBase(scope.storage)+".readonly-panel");
          Value imported=tileDMA(b,token,external[scope.storage],panel,region,true,{active.dependency},a.storages[scope.storage].id,begin,active.regions[scope.storage],&a,&plan,&error);
          imported.getDefiningOp()->setAttr("orbit.readonly_retained_begin",b.getI64IntegerAttr(begin));
          imported.getDefiningOp()->setAttr("orbit.readonly_retained_end",b.getI64IntegerAttr(end));
          make(b,"air.wait_all",{imported});
          active.dependency=imported;active.retained[scope.storage]=panel;
          active.readonly[scope.storage]=&scope;active.readonlyShifts[scope.storage]=active.regions[scope.storage];
        }
      SmallVector<size_t> imports;
      for (const auto &scope : plan.retainedRegions)
        if (scope.beginPhase==begin && scope.endPhase==end && !active.retained[scope.storage]) {
          auto region=llvm::find_if(first.regions,[&](const auto &r) {return r.storage==scope.storage;});
          Value output=tileView(*region,begin,storagePoolBase(scope.storage));
          if (scope.initializerTask >= 0) {
            GenericTilePhase initialization;
            initialization.task=scope.initializerTask;
            initialization.regions.push_back(*region);
            auto &initializer=a.tasks[scope.initializerTask];
            for(size_t l=0;l<initializer.loops.size();++l)
              initialization.loops.push_back({l,first.loops[l].lower,first.loops[l].upper});
            SmallVector<Value> initialized(a.storages.size());
            initialized[scope.storage]=output;
            emitTileBody(b,a,initialization,initialized,active.loops,active.regions,implementationOf(initialization.task),&error);
            output.getDefiningOp()->setAttr("orbit.initializer_task",b.getStringAttr(initializer.id));
          } else {
            Value input=tileView(*region,begin,storagePoolBase(scope.storage) + ".incoming");
            Value transferred=tileDMA(b,token,external[scope.storage],input,*region,true,{active.dependency},a.storages[scope.storage].id,begin,active.regions[scope.storage],&a,&plan,&error);
            make(b,"air.wait_all",{transferred});
            copyTile(input,output,*region);
            active.dependency=make(b,"air.wait_all",{transferred},{token})->getResult(0);
          }
          active.retained[scope.storage]=output;
          imports.push_back(scope.storage);
        }
      auto finishRetention = [&](Value complete) -> Value {
        SmallVector<Value> exported;
        for (auto storage : imports) {
          auto region=llvm::find_if(first.regions,[&](const auto &r) {return r.storage==storage;});
          Value result=tileDMA(b,token,external[storage],active.retained[storage],*region,false,{complete},a.storages[storage].id,end-1,active.regions[storage],&a,&plan,&error);
          result.getDefiningOp()->setAttr("orbit.retained_begin",b.getI64IntegerAttr(begin));
          result.getDefiningOp()->setAttr("orbit.retained_end",b.getI64IntegerAttr(end));
          exported.push_back(result);
        }
        return exported.empty() ? complete : make(b,"air.wait_all",exported,{token})->getResult(0);
      };
      if (depth == first.loops.size()) return finishRetention(emitPhase(begin, active));
      size_t scheduleLoop=plan.taskLoopOrders[first.task][depth];
      std::vector<std::pair<size_t,size_t>> children;
      for (size_t i = begin; i < end;) {
        size_t j = i + 1;
        auto bound = plan.phases[i].loops[scheduleLoop];
        while (j < end && plan.phases[j].loops[scheduleLoop].lower == bound.lower && plan.phases[j].loops[scheduleLoop].upper == bound.upper) ++j;
        children.push_back({i,j}); i=j;
      }
      Value completion = active.dependency;
      for (size_t child = 0; child < children.size();) {
        Shifts sequential = active; sequential.dependency = completion;
        size_t width = children[child].second - children[child].first;
        size_t count = 1;
        std::vector<int64_t> loopDelta(first.loops.size());
        std::vector<std::vector<int64_t>> regionDelta(a.storages.size());
        if (child + 1 < children.size()) {
          auto &x = plan.phases[children[child].first];
          auto &y = plan.phases[children[child+1].first];
          for (size_t l = 0; l < x.loops.size(); ++l) loopDelta[l] = y.loops[l].lower - x.loops[l].lower;
          for (size_t r = 0; r < x.regions.size(); ++r) {
            auto &xr=x.regions[r]; auto &yr=y.regions[r];
            if (xr.storage != yr.storage || xr.offsets.size() != yr.offsets.size()) continue;
            regionDelta[xr.storage].resize(xr.offsets.size());
            for (size_t d=0;d<xr.offsets.size();++d) regionDelta[xr.storage][d]=yr.offsets[d]-xr.offsets[d];
          }
          for (size_t next=child+1;next<children.size();++next) {
            bool equivalent=children[next].second-children[next].first==width;
            int64_t factor=next-child;
            for (size_t j=0;j<width && equivalent;++j) {
              auto &base=plan.phases[children[child].first+j];
              auto &candidate=plan.phases[children[next].first+j];
              equivalent &= base.task==candidate.task && base.loops.size()==candidate.loops.size() && base.regions.size()==candidate.regions.size();
              for (size_t l=0;l<base.loops.size() && equivalent;++l)
                equivalent &= candidate.loops[l].lower==base.loops[l].lower+factor*loopDelta[l] && candidate.loops[l].upper==base.loops[l].upper+factor*loopDelta[l];
              for (size_t r=0;r<base.regions.size() && equivalent;++r) {
                auto &br=base.regions[r];auto &cr=candidate.regions[r];
                equivalent &= br.storage==cr.storage && br.sizes==cr.sizes && br.read==cr.read && br.write==cr.write && regionDelta[br.storage].size()==br.offsets.size();
                for(size_t d=0;d<br.offsets.size() && equivalent;++d)
                  equivalent &= cr.offsets[d]==br.offsets[d]+factor*regionDelta[br.storage][d];
              }
            }
            if (!equivalent) break;
            ++count;
          }
        }
        if (count==1) completion = emitRange(children[child].first, children[child].second, depth+1, sequential);
        else {
          Value zero=b.create<arith::ConstantIndexOp>(loc,0), upper=b.create<arith::ConstantIndexOp>(loc,count), step=b.create<arith::ConstantIndexOp>(loc,1);
          auto loop=b.create<scf::ForOp>(loc,zero,upper,step,ValueRange{completion});
          loop->setAttr("orbit.compressed_phases",b.getI64IntegerAttr(count));
          b.setInsertionPointToStart(loop.getBody());
          Shifts shifts=sequential;
          shifts.dependency=loop.getRegionIterArgs()[0];
          auto shifted = [&](Value original,int64_t delta) -> Value {
            if (!delta) return original;
            Value scale=b.create<arith::ConstantIndexOp>(loc,delta);
            Value value=b.create<arith::MulIOp>(loc,loop.getInductionVar(),scale);
            if (original) value=b.create<arith::AddIOp>(loc,original,value);
            return value;
          };
          for(size_t l=0;l<loopDelta.size();++l) shifts.loops[l]=shifted(shifts.loops[l],loopDelta[l]);
          for(size_t r=0;r<regionDelta.size();++r)
            for(size_t d=0;d<regionDelta[r].size();++d) shifts.regions[r][d]=shifted(shifts.regions[r][d],regionDelta[r][d]);
          Value done = emitRange(children[child].first,children[child].second,depth+1,shifts);
          if (!loop.getBody()->empty() && isa<scf::YieldOp>(loop.getBody()->back())) loop.getBody()->back().erase();
          b.setInsertionPointToEnd(loop.getBody());
          b.create<scf::YieldOp>(loc, ValueRange{done});
          b.setInsertionPointAfter(loop);
          completion=loop.getResult(0);
        }
        child+=count;
      }
      return finishRetention(completion);
    };
    for (size_t begin=stageBegin;begin<stageEnd;) {
      size_t end=begin+1;
      while(end<stageEnd && plan.phases[end].task==plan.phases[begin].task) ++end;
      Shifts shifts;
      shifts.dependency = make(b, "air.wait_all", {}, {token})->getResult(0);
      shifts.loops.resize(a.tasks[plan.phases[begin].task].loops.size());
      shifts.retained.resize(a.storages.size());
      shifts.readonly.resize(a.storages.size());
      shifts.readonlyShifts.resize(a.storages.size());
      shifts.regions.resize(a.storages.size());
      for(size_t r=0;r<a.storages.size();++r) shifts.regions[r].resize(a.storages[r].shape.size());
      Value complete = emitRange(begin,end,0,shifts);
      make(b, "air.wait_all", {complete});
      begin=end;
    }
    // The target allocates these L1 pools statically for the whole stage.
    // Its channel dependency importer cannot reconnect an end-of-stage
    // deallocation through already channelized nested tile loops. On that
    // explicitly planned path, region exit owns the lifetime instead.
    if (!plan.hardwareStages[stageIndex].legalizeChannels)
      for (auto &[dtype, pool] : pools) b.create<memref::DeallocOp>(loc, pool);
    if (stageIndex + 1 < plan.hardwareStages.size()) {
      make(b, "air.herd_terminator");
      b.setInsertionPointAfter(herd);
      if (herd->getNumResults()) make(b, "air.wait_all", {herd->getResult(0)});
      make(b, "air.segment_terminator");
      b.setInsertionPointAfter(segment); make(b, "air.launch_terminator");
      b.setInsertionPointAfter(launch);
      std::string suffix = std::to_string(plan.hardwareStages[stageIndex+1].beginPhase);
      launch = hierarchy(b, "air.launch", "orbit_launch_" + suffix, {c1,c1}, args);
      b.setInsertionPointToStart(&launch->getRegion(0).front());
      SmallVector<Value> stageArgs(launch->getRegion(0).front().getArguments().drop_front(4));
      segment = hierarchy(b, "air.segment", "orbit_segment_" + suffix, {}, stageArgs);
      b.setInsertionPointToStart(&segment->getRegion(0).front());
      Value stageOne = b.create<arith::ConstantIndexOp>(loc, 1);
      herd = hierarchy(b, "air.herd", "orbit_herd_" + suffix, {stageOne,stageOne}, segment->getRegion(0).front().getArguments(), plan.hardwareStages[stageIndex+1].legalizeChannels);
      herd->setAttr("orbit.mapping_strategy", b.getStringAttr(plan.strategy));
      b.setInsertionPointToStart(&herd->getRegion(0).front());
      ha = herd->getRegion(0).front().getArguments().drop_front(4);
    }
    }
  } else {
  SmallVector<Value> local(a.storages.size());
  SmallVector<Value> pending;
  for (size_t i = 0; i < a.storages.size(); ++i) {
    auto &s = a.storages[i];
    auto t = cast<MemRefType>(s.root.getType());
    auto lt =
        MemRefType::get(t.getShape(), t.getElementType(),
                        MemRefLayoutAttrInterface{}, b.getI32IntegerAttr(2));
    auto alloc = b.create<memref::AllocOp>(loc, lt);
    alloc->setAttr("orbit.storage", b.getStringAttr(s.id));
    local[i] = alloc;
    if (s.role == "input") {
      auto ba = dyn_cast<BlockArgument>(s.root);
      if (!ba) {
        error = "input storage must be function argument";
        return false;
      }
      Value imported=dma(b,token,local[i],ha[ba.getArgNumber()],{},s.id,"host_to_l1");
      auto transfer=llvm::find_if(plan.transfers,[&](const auto &entry){return entry.storage==i && entry.direction=="host_to_l1";});
      if(transfer==plan.transfers.end() || !bindDMACopies(b,a,plan,imported.getDefiningOp(),ha[ba.getArgNumber()],local[i],*transfer,error)) return false;
      pending.push_back(imported);
    }
  }
  Value previous;
  for (size_t ti : plan.orderedTasks) {
    auto &t = a.tasks[ti];
    SmallVector<Value> deps;
    if (previous)
      deps.push_back(previous);
    else
      llvm::append_range(deps, pending);
    // AIR's dependency pass owns asynchronous loop construction.  A nested
    // pre-existing air.execute around an SCF nest is unsupported in the fixed
    // toolchain.  Synchronous task boundaries preserve the source ordering;
    // AIR then derives loop/DMA async edges from the real memory effects.
    auto *start = make(b, "air.wait_all", deps);
    start->setAttr("orbit.task", b.getStringAttr(t.id));
    SmallVector<int64_t> sourceDeps(t.dependencies.begin(),
                                    t.dependencies.end());
    start->setAttr("orbit.source_dependencies",
                   b.getDenseI64ArrayAttr(sourceDeps));
    IRMapping map;
    auto &body = t.operation->getRegion(0).front();
    for (size_t i = 0; i < t.argumentStorages.size(); ++i)
      map.map(body.getArgument(i), local[t.argumentStorages[i]]);
    for (auto &op : body.without_terminator()) {
      auto *clone = b.clone(op, map);
      clone->setAttr("orbit.task", b.getStringAttr(t.id));
    }
    auto *complete = make(b, "air.wait_all", {}, {token});
    complete->setAttr("orbit.task_completion", b.getStringAttr(t.id));
    previous = complete->getResult(0);
  }
  SmallVector<Value> final;
  for (size_t i = 0; i < a.storages.size(); ++i)
    if (a.storages[i].role == "output") {
      auto ba = dyn_cast<BlockArgument>(a.storages[i].root);
      if (!ba) {
        error = "output storage must be function argument";
        return false;
      }
      Value exported=dma(b,token,ha[ba.getArgNumber()],local[i],{previous},a.storages[i].id,"l1_to_host");
      auto transfer=llvm::find_if(plan.transfers,[&](const auto &entry){return entry.storage==i && entry.direction=="l1_to_host";});
      if(transfer==plan.transfers.end() || !bindDMACopies(b,a,plan,exported.getDefiningOp(),ha[ba.getArgNumber()],local[i],*transfer,error)) return false;
      final.push_back(exported);
    }
  if (final.empty() && previous)
    final.push_back(previous);
  make(b, "air.wait_all", final);
  for (auto v : local)
    b.create<memref::DeallocOp>(loc, v);
  }
  make(b, "air.herd_terminator");
  b.setInsertionPointAfter(herd);
  if (herd->getNumResults()) make(b, "air.wait_all", {herd->getResult(0)});
  make(b, "air.segment_terminator");
  b.setInsertionPointAfter(segment);
  make(b, "air.launch_terminator");
  b.setInsertionPointAfter(launch);
  b.create<func::ReturnOp>(loc);
  output->getOperation()->setAttr("orbit.source_fingerprint",
                                  b.getStringAttr(a.bodyFingerprint));
  output->getOperation()->setAttr("orbit.mapping_plan",
                                  b.getStringAttr(plan.planId));
  output->getOperation()->setAttr("orbit.data_organization",
                                  b.getStringAttr(plan.choices.dataOrganization));
  output->getOperation()->setAttr("orbit.data_organization_scope",
      b.getStringAttr(plan.choices.dataOrganization=="retain-across-users"
        ? "readonly users within a single execution group; no cross-stage SRAM retention"
        : "phase-local input copies; source output/reduction lifetime preserved"));
  output->getOperation()->setAttr("orbit.cost_status",
                                  b.getStringAttr("uncalibrated-direct-air"));
  if (approximateExp) {
    ctx->getOrLoadDialect<math::MathDialect>();
    llvm::SmallPtrSet<Operation *, 8> originalFloors;
    output->walk([&](math::FloorOp op) { originalFloors.insert(op); });
    RewritePatternSet patterns(ctx);
    populateMathPolynomialApproximationPatterns(patterns);
    llvm::erase_if(patterns.getNativePatterns(), [](const auto &pattern) {
      auto root = pattern->getRootKind();
      return !root || root->getStringRef() != "math.exp";
    });
    if (failed(applyPatternsGreedily(*output, std::move(patterns)))) {
      error = "explicit MLIR f32 exp polynomial approximation failed";
      return false;
    }
    RewritePatternSet unfused(ctx);
    populateExpandFmaFPattern(unfused);
    if (failed(applyPatternsGreedily(*output, std::move(unfused)))) {
      error = "explicit exp approximation FMA expansion failed";
      return false;
    }
    // The upstream exp pattern clamps x to [-87.8, 88.8] before
    // floor(x * log2(e) + .5). Only these newly generated, bounded floors
    // are expanded. The fixed MLIR provides a ceil expansion but no floor
    // expansion. Truncation and a negative-fraction correction are exact
    // over this range. Existing source math.floor operations are untouched.
    SmallVector<math::FloorOp> boundedFloors;
    output->walk([&](math::FloorOp op) {
      if (!originalFloors.contains(op))
        boundedFloors.push_back(op);
    });
    for (math::FloorOp op : boundedFloors) {
      OpBuilder floorBuilder(op);
      auto floorLoc = op.getLoc();
      Value input = op.getOperand();
      Value integer = floorBuilder.create<arith::FPToSIOp>(
          floorLoc, floorBuilder.getI32Type(), input);
      Value truncated = floorBuilder.create<arith::SIToFPOp>(
          floorLoc, input.getType(), integer);
      Value below = floorBuilder.create<arith::CmpFOp>(
          floorLoc, arith::CmpFPredicate::OLT, input, truncated);
      Value one = floorBuilder.create<arith::ConstantOp>(
          floorLoc, floorBuilder.getF32FloatAttr(1.0));
      Value corrected = floorBuilder.create<arith::SubFOp>(
          floorLoc, truncated, one);
      Value result = floorBuilder.create<arith::SelectOp>(
          floorLoc, below, corrected, truncated);
      op.replaceAllUsesWith(result);
      op.erase();
    }
    output->getOperation()->setAttr(
        "orbit.math_approximation",
        b.getStringAttr(
            "mlir-f32-cephes-exp; unfused-mul-add; bounded-floor-i32; finite-input-contract; flush-denormals; no-global-fastmath"));
  }
  if (failed(normalize(*output))) {
    error = "direct AIR standard affine-to-scf lowering failed";
    return false;
  }
  if (!error.empty()) return false;
  if (failed(verify(*output))) {
    error = "generated generic AIR module failed structural MLIR verification";
    return false;
  }
  if (!skipCheck && !checkGenericAIRMapping(a, plan, *output, error))
    return false;
  return true;
}

bool orbit::taskflow::lowerGenericToAIR(const GenericAnalysis &a,
                                        const GenericMappingPlan &plan,
                                        OwningOpRef<ModuleOp> &output,
                                        std::string &error,
                                        bool approximateExp) {
  return lowerGenericToAIRImpl(a, plan, output, error, approximateExp,
                              [&](ModuleOp module) {
    PassManager normalization(module.getContext());
    normalization.addPass(createLowerAffinePass());
    return normalization.run(module);
  });
}

namespace {
struct TaskflowToAIRPass
    : PassWrapper<TaskflowToAIRPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TaskflowToAIRPass)
  StringRef getArgument() const final { return "taskflow-to-air"; }
  StringRef getDescription() const final {
    return "Lower native Taskflow bodies with verified single-herd resident "
           "mapping to AIR";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect, func::FuncDialect, scf::SCFDialect,
                    math::MathDialect, cf::ControlFlowDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    GenericAnalysis analysis;
    GenericMappingPlan plan;
    std::string error;
    OwningOpRef<ModuleOp> result;
    if (!analyzeGenericProgram(getOperation(), analysis, error) ||
        !planGenericMapping(analysis, llvm::json::Object{}, plan, error) ||
        !lowerGenericToAIRImpl(analysis, plan, result, error, false,
                               [&](ModuleOp module) {
          OpPassManager normalization(ModuleOp::getOperationName());
          normalization.addPass(createLowerAffinePass());
          // Dynamic pipelines must run on an operation nested under this
          // pass's root. Temporarily attach the replacement module, then
          // detach it before replacing the original module body.
          getOperation().getBody()->push_back(module);
          LogicalResult status = runPipeline(normalization, module);
          module->remove();
          return status;
        })) {
      getOperation().emitError(error);
      signalPassFailure();
      return;
    }
    getOperation()->setAttrs(result->getOperation()->getAttrs());
    getOperation().getBodyRegion().takeBody(result->getBodyRegion());
  }
};
} // namespace
std::unique_ptr<Pass> orbit::taskflow::createTaskflowToAIRPass() {
  return std::make_unique<TaskflowToAIRPass>();
}
void orbit::taskflow::registerTaskflowToAIRPass() {
  PassRegistration<TaskflowToAIRPass>();
}
