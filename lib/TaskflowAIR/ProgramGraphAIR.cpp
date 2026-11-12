#include "Taskflow/ProgramGraphAIR.h"
#include "ProgramGraphAIRBF16RNE.h"
#include "Taskflow/AIRBackend.h"
#include "Taskflow/ProgramGraphAIRLoop.h"
#include "Taskflow/ProgramGraphAIRSelection.h"
#include "Taskflow/ProgramMatrixImplementation.h"
#include "Taskflow/ProgramTargetBinding.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
using namespace mlir;
using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
bool fail(std::string &e, const std::string &s) {
  e = "public graph AIR: " + s;
  return false;
}
Operation *op(OpBuilder &b, StringRef name, ValueRange values = {},
              TypeRange types = {}, bool region = false) {
  OperationState s(b.getUnknownLoc(), name);
  s.addOperands(values);
  s.addTypes(types);
  if (region)
    s.addRegion();
  return b.create(s);
}
Operation *hierarchy(OpBuilder &b, StringRef name, const std::string &symbol,
                     ValueRange sizes, ValueRange values) {
  SmallVector<Value> operands(sizes);
  llvm::append_range(operands, values);
  auto *o = op(b, name, operands, {}, true);
  o->setAttr("sym_name", b.getStringAttr(symbol));
  o->setAttr("operandSegmentSizes",
             b.getDenseI32ArrayAttr(
                 {0, int32_t(sizes.size()), int32_t(values.size())}));
  auto *block = new Block;
  o->getRegion(0).push_back(block);
  for (size_t i = 0; i < 2 * sizes.size(); ++i)
    block->addArgument(b.getIndexType(), b.getUnknownLoc());
  for (Value v : values)
    block->addArgument(v.getType(), b.getUnknownLoc());
  return o;
}
Block *body(Operation *o) { return &o->getRegion(0).front(); }
const StaticResource *resource(const ProgramGraph &g, const std::string &id) {
  auto it =
      llvm::find_if(g.resources, [&](const auto &r) { return r.id == id; });
  return it == g.resources.end() ? nullptr : &*it;
}
const StaticMemory *memory(const ProgramGraph &g, const std::string &id) {
  auto it =
      llvm::find_if(g.memories, [&](const auto &r) { return r.id == id; });
  return it == g.memories.end() ? nullptr : &*it;
}
const StaticResourceDomain *domain(const ProgramGraph &g,
                                   const std::string &id) {
  auto it = llvm::find_if(g.domains, [&](const auto &r) { return r.id == id; });
  return it == g.domains.end() ? nullptr : &*it;
}
int64_t elements(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (auto x : shape) {
    if (x < 1 || n > INT64_MAX / x)
      return -1;
    n *= x;
  }
  return n;
}
std::vector<int64_t> strides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> r(shape.size(), 1);
  for (int64_t i = int64_t(shape.size()) - 2; i >= 0; --i)
    r[i] = r[i + 1] * shape[i + 1];
  return r;
}
j::Array indices(const std::vector<size_t> &values) {
  j::Array out;
  for (auto v : values)
    out.push_back(int64_t(v));
  return out;
}
struct Core {
  std::string owner, memory, resource;
  int64_t col = 0, row = 2, bytes = 0;
  std::vector<size_t> events, groups, allocations;
};
// Require the exact cells for a source storage to form one rectangular view
// in one selected allocation. No union bounding-box approximation is allowed.
bool rectangle(const ProgramGraph &g, const std::vector<size_t> &copies,
               size_t storage, MappingRegion &out, size_t &allocation,
               std::string &e) {
  bool first = true;
  std::set<size_t> cells;
  int64_t volume = 0;
  std::vector<int64_t> upper;
  for (auto ci : copies) {
    const auto &c = g.copies[ci];
    const auto &a = g.allocations[c.allocation];
    if (a.storage != storage)
      continue;
    const auto &r = g.regions[g.versions[c.version].region];
    if (first) {
      out = r;
      allocation = c.allocation;
      upper = r.offsets;
      for (size_t d = 0; d < upper.size(); ++d)
        upper[d] += r.sizes[d];
      first = false;
    } else {
      if (allocation != c.allocation || r.sizes.size() != upper.size())
        return fail(e, "source storage spans different selected allocations");
      for (size_t d = 0; d < upper.size(); ++d) {
        out.offsets[d] = std::min(out.offsets[d], r.offsets[d]);
        upper[d] = std::max(upper[d], r.offsets[d] + r.sizes[d]);
      }
    }
    if (cells.insert(g.versions[c.version].region).second) {
      auto n = elements(r.sizes);
      if (n < 0 || volume > INT64_MAX - n)
        return fail(e, "rectangle volume overflow");
      volume += n;
    }
  }
  if (first)
    return fail(e, "source storage has no selected copy");
  for (size_t d = 0; d < upper.size(); ++d)
    out.sizes[d] = upper[d] - out.offsets[d];
  if (elements(out.sizes) != volume)
    return fail(e, "selected cells do not form one exact rectangular view");
  return true;
}
// A reusable slot begins at the origin of the rectangle installed by its
// producing event. A subrectangle needs a nonzero physical offset; do not
// silently reinterpret it at zero. This first backend rejects that refinement.
bool originContract(const ProgramGraph &g, const std::vector<size_t> &copies,
                    const MappingRegion &r, size_t ai, std::string &e) {
  for (auto ci : copies) {
    const auto &c = g.copies[ci];
    if (c.allocation != ai)
      continue;
    if (c.createdBy < 0)
      return fail(e, "local copy has no physical initialization event");
    MappingRegion installed;
    size_t producerAllocation = 0;
    if (!rectangle(g, g.events[c.createdBy].writes, r.storage, installed,
                   producerAllocation, e))
      return false;
    if (producerAllocation != ai || installed.offsets != r.offsets ||
        installed.sizes != r.sizes)
      return fail(e, "selected source view is a subrectangle of its slot; "
                     "physical offset refinement is unsupported");
  }
  return true;
}
Value view(OpBuilder &b, Value pool, const MappingRegion &r, Type element,
           size_t allocation, int64_t physicalOffset) {
  auto shape = llvm::to_vector(r.sizes);
  auto ss = strides(r.sizes);
  auto type = MemRefType::get(
      shape, element,
      StridedLayoutAttr::get(b.getContext(), physicalOffset, ss),
      b.getI32IntegerAttr(2));
  auto result = b.create<memref::ReinterpretCastOp>(
      b.getUnknownLoc(), type, pool, physicalOffset, shape, ss);
  result->setAttr("orbit.allocation", b.getI64IntegerAttr(allocation));
  result->setAttr("orbit.storage_index", b.getI64IntegerAttr(r.storage));
  result->setAttr("orbit.logical_offsets", b.getDenseI64ArrayAttr(r.offsets));
  result->setAttr("orbit.logical_sizes", b.getDenseI64ArrayAttr(r.sizes));
  return result;
}
} // namespace
bool orbit::taskflow::lowerProgramGraphToAIR(
    const ProgramGraph &g, OwningOpRef<ModuleOp> &out, j::Value &manifest,
    std::string &e, const ProgramGraphAIRLoopPlan *requestedLoops,
    const ProgramGraphAIRSelection *selection) {
  auto sourceFunction = g.source.function;
  e.clear();
  if (!checkProgramGraph(g, e) || !checkProgramGraphSource(g.source, g, e))
    return false;
  for (const auto &placement : g.placements)
    if (placement.resourceCount != 1 ||
        (!placement.resourceShape.empty() &&
         placement.resourceShape != std::vector<int64_t>{1}))
      return fail(e, "selected multi-resource count/shape is unsupported by "
                     "direct-core realization");
  // Installed AIR extraction changes repeated DMA FIFO order. No loop plan
  // is executable until final core/shim sequence coverage is independently
  // proven.
  if (requestedLoops)
    return fail(e, "experimental loop AIR is disabled: final DMA FIFO coverage "
                   "is unproved");
  const ProgramGraphAIRLoopPlan *loops = nullptr;
  if (selection && !checkProgramGraphAIRSelection(g, *selection, e))
    return false;
  std::map<std::string, Core> cores;
  std::vector<std::string> owner(g.events.size());
  // The physical backend assigns no source task, copy, route, or resource.
  for (size_t ai = 0; ai < g.allocations.size(); ++ai) {
    const auto &a = g.allocations[ai];
    auto *m = memory(g, a.memory);
    if (!m)
      return fail(e, "selected allocation memory is missing");
    if (m->kind == "system-bo")
      continue;
    if (m->kind != "local-sram" || a.addressing != "reusable-slot")
      return fail(
          e,
          "only explicit local-SRAM reusable-slot allocations are supported");
    auto *d = domain(g, m->domain);
    if (!d || !d->coordinates.count("column") || !d->coordinates.count("row"))
      return fail(e, "local allocation lacks physical coordinates");
    auto &c = cores[d->id];
    if (!c.memory.empty() && c.memory != m->id)
      return fail(e,
                  "multiple SRAM memories per physical core are unsupported");
    c.owner = d->id;
    c.memory = m->id;
    c.col = d->coordinates.at("column");
    c.row = d->coordinates.at("row");
    if (c.col < 0 || c.col > 3 || c.row != 2)
      return fail(e, "NPU2 physical coordinates outside direct-core subset");
    if (a.bytes < 1 || c.bytes > INT64_MAX - a.bytes)
      return fail(e, "static allocation bytes overflow");
    c.bytes += a.bytes;
    c.allocations.push_back(ai);
  }
  if (loops && cores.size() != 1)
    return fail(e, "loop realization requires one selected physical core");
  if (cores.empty())
    return fail(e, "no selected local compute core");
  if (selection && cores.size() != 1)
    return fail(e, "selected event ranges currently require one physical core");
  std::set<std::pair<int64_t, int64_t>> coordinates;
  for (auto &[id, c] : cores) {
    if (!coordinates.emplace(c.col, c.row).second)
      return fail(e, "different target owners alias physical coordinates");
    if (c.bytes > memory(g, c.memory)->capacityBytes)
      return fail(e,
                  "static selected allocation sum exceeds target SRAM "
                  "capacity; lifetime-based physical reuse is not implemented");
  }
  std::vector<const MappingTaskInstance *> instance(g.events.size(), nullptr);
  for (auto &i : g.instances) {
    if (i.resourceCount != 1 || (!i.resourceShape.empty() &&
                                 i.resourceShape != std::vector<int64_t>{1}))
      return fail(e, "selected instance resource count/shape is unsupported by "
                     "direct-core realization");
    if (i.event >= instance.size() || instance[i.event])
      return fail(e, "source instance event is not unique");
    instance[i.event] = &i;
  }
  for (size_t ei = 0; ei < g.events.size(); ++ei) {
    const auto &ev = g.events[ei];
    std::string own;
    if (ev.kind == "compute" || ev.kind == "initialize") {
      auto *r = resource(g, ev.resource);
      if (!r || r->kind != "compute" || !instance[ei])
        return fail(e, "compute event has no physical source binding");
      own = r->domain;
    } else if (ev.kind == "transfer") {
      if (!ev.transferRegion || ev.reads.empty() || ev.writes.empty())
        return fail(e, "transfer has no explicit selected endpoint copies");
      std::set<std::string> locals;
      bool hostSource = false, hostTarget = false;
      for (auto ci : ev.reads) {
        auto *m = memory(g, g.allocations[g.copies[ci].allocation].memory);
        if (m->kind == "system-bo")
          hostSource = true;
        else
          locals.insert(m->domain);
      }
      for (auto ci : ev.writes) {
        auto *m = memory(g, g.allocations[g.copies[ci].allocation].memory);
        if (m->kind == "system-bo")
          hostTarget = true;
        else
          locals.insert(m->domain);
      }
      if (locals.size() != 1 || hostSource == hostTarget)
        return fail(e, "transfer is not one direct host/core selected path");
      own = *locals.begin();
      auto link = llvm::find_if(g.links,
                                [&](const auto &l) { return l.id == ev.link; });
      if (link == g.links.end() || link->resource != ev.resource)
        return fail(e, "transfer link/resource mismatch");
    } else
      return fail(e, "unsupported event kind");
    auto it = cores.find(own);
    if (it == cores.end())
      return fail(e, "event has no selected local physical core");
    owner[ei] = own;
    it->second.events.push_back(ei);
    for (auto ci : ev.reads) {
      const auto &a = g.allocations[g.copies[ci].allocation];
      auto *m = memory(g, a.memory);
      if (m->kind != "system-bo" && m->domain != own)
        return fail(e, "event captures another core's selected copy");
    }
    for (auto ci : ev.writes) {
      const auto &a = g.allocations[g.copies[ci].allocation];
      auto *m = memory(g, a.memory);
      if (m->kind != "system-bo" && m->domain != own)
        return fail(e, "event writes another core's selected copy");
    }
  }
  for (size_t ei = 0; ei < g.events.size(); ++ei)
    for (auto pred : g.events[ei].dependencies)
      if (pred >= ei || owner[pred] != owner[ei])
        return fail(e, "cross-core dependencies require an explicit physical "
                       "synchronization implementation");
  for (size_t gi = 0; gi < g.executionGroups.size(); ++gi) {
    const auto &group = g.executionGroups[gi];
    auto it = cores.find(group.spatialOwner);
    if (it == cores.end())
      return fail(e, "execution group owner has no physical core");
    it->second.groups.push_back(gi);
    for (auto event : group.memberEvents)
      if (owner[event] != group.spatialOwner)
        return fail(e, "execution group crosses physical core owners");
  }
  if (selection)
    for (auto &[id, core] : cores)
      core.events.erase(std::remove_if(core.events.begin(), core.events.end(),
                                       [&](size_t event) {
                                         return event < selection->begin ||
                                                event >= selection->end;
                                       }),
                        core.events.end());
  auto *ctx = sourceFunction.getContext();
  ctx->loadDialect<memref::MemRefDialect, arith::ArithDialect,
                   func::FuncDialect, scf::SCFDialect>();
  prepareGenericAIRContext(*ctx);
  out = ModuleOp::create(sourceFunction.getLoc());
  OpBuilder b(ctx);
  auto loc = b.getUnknownLoc();
  SmallVector<Type> abiTypes(sourceFunction.getArgumentTypes());
  std::vector<int64_t> abi(g.source.storages.size(), -1);
  std::vector<size_t> abiStorage(abiTypes.size(), SIZE_MAX);
  for (size_t st = 0; st < g.source.storages.size(); ++st)
    if (auto arg = dyn_cast<BlockArgument>(g.source.storages[st].root)) {
      if (arg.getOwner() != &sourceFunction.getBody().front())
        return fail(e, "source ABI storage is not a direct function argument");
      abi[st] = arg.getArgNumber();
      abiStorage[arg.getArgNumber()] = st;
    }
  for (const auto &a : g.allocations)
    if (memory(g, a.memory)->kind == "system-bo") {
      auto type = cast<MemRefType>(g.source.storages[a.storage].root.getType());
      auto bytes = elements(g.source.storages[a.storage].shape) *
                   int64_t((type.getElementTypeBitWidth() + 7) / 8);
      if (a.bytes != bytes)
        return fail(e, "host allocation requires full source BO shape");
      if (abi[a.storage] < 0) {
        abi[a.storage] = abiTypes.size();
        abiTypes.push_back(type);
        abiStorage.push_back(a.storage);
      }
    }
  for (auto st : abiStorage)
    if (st == SIZE_MAX)
      return fail(e, "non-storage source ABI arguments are unsupported");
  auto f = func::FuncOp::create(loc, sourceFunction.getName(),
                                FunctionType::get(ctx, abiTypes, {}));
  f->setAttr("llvm.emit_c_interface", b.getUnitAttr());
  f->setAttr("orbit.noalias", b.getUnitAttr());
  out->push_back(f);
  auto *fb = f.addEntryBlock();
  for (size_t i = 0; i < abiStorage.size(); ++i) {
    auto st = abiStorage[i];
    f.setArgAttr(i, "orbit.name", b.getStringAttr(g.source.storages[st].id));
    f.setArgAttr(i, "orbit.role", b.getStringAttr(g.source.storages[st].role));
  }
  b.setInsertionPointToStart(fb);
  auto one = b.create<arith::ConstantIndexOp>(loc, 1);
  auto *launch = hierarchy(b, "air.launch", "orbit_public_graph", {one, one},
                           f.getArguments());
  // The sealed selected graph owns every DMA occurrence, including repeated
  // equal-address inputs. Per-channel folding can turn these into runtime
  // loops outside the static realization contract. Preserve the explicit feed;
  // final source-bound runtime coverage remains mandatory after extraction.
  launch->setAttr("air.preserve_shim_dma_order", b.getUnitAttr());
  b.setInsertionPointToStart(body(launch));
  auto *segment =
      hierarchy(b, "air.segment", "orbit_public_graph_configuration", {},
                body(launch)->getArguments().drop_front(4));
  b.setInsertionPointToStart(body(segment));
  auto token = OpaqueType::get(b.getStringAttr("air"), "async.token");
  j::Array coreRecords, eventRecords, allocationRecords, stagingRecords;
  int64_t localStagingBytes = 0;
  for (auto &[id, c] : cores) {
    b.setInsertionPointToEnd(body(segment));
    auto localOne = b.create<arith::ConstantIndexOp>(loc, 1);
    auto *herd =
        hierarchy(b, "air.herd", "orbit_public_core_" + std::to_string(c.col),
                  {localOne, localOne}, body(segment)->getArguments());
    herd->setAttr("x_loc", b.getI32IntegerAttr(c.col));
    herd->setAttr("y_loc", b.getI32IntegerAttr(c.row));
    herd->setAttr("orbit.spatial_owner", b.getStringAttr(id));
    b.setInsertionPointToStart(body(herd));
    std::map<size_t, Value> pools;
    std::map<size_t, int64_t> allocationOffsets;
    // Pack distinct logical allocations into disjoint slices, rather than
    // making each slice a separate DMA buffer/lock domain. No addresses alias;
    // this refinement preserves even simultaneously-live old versions.
    std::map<size_t, std::vector<size_t>> byStorage;
    for (auto ai : c.allocations)
      byStorage[g.allocations[ai].storage].push_back(ai);
    for (auto &[st, logical] : byStorage) {
      auto element = cast<MemRefType>(g.source.storages[st].root.getType())
                         .getElementType();
      auto itemBytes = int64_t((element.getIntOrFloatBitWidth() + 7) / 8);
      int64_t count = 0;
      SmallVector<int64_t> ids, offsets;
      for (auto ai : logical) {
        const auto &a = g.allocations[ai];
        if (a.bytes % itemBytes)
          return fail(
              e,
              "allocation byte size is not a whole number of source elements");
        ids.push_back(ai);
        offsets.push_back(count);
        allocationOffsets[ai] = count;
        count += a.bytes / itemBytes;
      }
      auto physicalPool = "public-storage-" + std::to_string(st);
      auto type = MemRefType::get({count}, element, MemRefLayoutAttrInterface{},
                                  b.getI32IntegerAttr(2));
      auto alloc = b.create<memref::AllocOp>(loc, type);
      alloc->setAttr("orbit.allocations", b.getDenseI64ArrayAttr(ids));
      alloc->setAttr("orbit.allocation_offsets_elements",
                     b.getDenseI64ArrayAttr(offsets));
      alloc->setAttr("orbit.storage_index", b.getI64IntegerAttr(st));
      alloc->setAttr("orbit.tile_pool", b.getStringAttr(physicalPool));
      if (logical.size() == 1)
        alloc->setAttr("orbit.allocation",
                       b.getI64IntegerAttr(logical.front()));
      for (auto ai : logical) {
        const auto &a = g.allocations[ai];
        pools[ai] = alloc;
        allocationRecords.push_back(
            j::Object{{"allocation", int64_t(ai)},
                      {"memory", a.memory},
                      {"key", a.key},
                      {"bytes", a.bytes},
                      {"column", c.col},
                      {"row", c.row},
                      {"physical_pool", physicalPool},
                      {"offset_elements", allocationOffsets[ai]}});
      }
    }
    // Express the selected transfer order in AIR. Installed extraction may
    // discard these async dependencies: final NPU runtime ordering must be
    // independently checked and realized before the ELF is executable. The
    // AIR token chain alone is not a completion-before-host-reload proof.
    Value previousDMA;
    auto emitEvent = [&](size_t ei,
                         ArrayRef<Value> dynamicHostOffsets) -> bool {
      if (previousDMA && previousDMA.getParentBlock() != b.getBlock())
        previousDMA = {};
      const auto &ev = g.events[ei];
      auto *marker = op(b, "air.wait_all");
      marker->setAttr("orbit.public_event", b.getI64IntegerAttr(ei));
      marker->setAttr("orbit.read_copies",
                      b.getDenseI64ArrayAttr(llvm::to_vector(llvm::map_range(
                          ev.reads, [](size_t i) { return int64_t(i); }))));
      marker->setAttr("orbit.write_copies",
                      b.getDenseI64ArrayAttr(llvm::to_vector(llvm::map_range(
                          ev.writes, [](size_t i) { return int64_t(i); }))));
      if (ev.kind == "transfer") {
        const auto &r = *ev.transferRegion;
        bool input =
            memory(g,
                   g.allocations[g.copies[ev.reads.front()].allocation].memory)
                ->kind == "system-bo";
        const auto &localCopies = input ? ev.writes : ev.reads;
        MappingRegion actual;
        size_t ai = 0;
        if (!rectangle(g, localCopies, r.storage, actual, ai, e))
          return false;
        if (actual.offsets != r.offsets || actual.sizes != r.sizes)
          return fail(e, "transfer selected local cells differ from exact "
                         "transfer rectangle");
        if (!originContract(g, localCopies, r, ai, e))
          return false;
        if (!pools.count(ai) || abi[r.storage] < 0)
          return fail(e, "transfer operand has no explicit pool or BO");
        auto element =
            cast<MemRefType>(g.source.storages[r.storage].root.getType())
                .getElementType();
        if (elements(r.sizes) >
            g.allocations[ai].bytes /
                int64_t((element.getIntOrFloatBitWidth() + 7) / 8))
          return fail(e, "transfer exceeds selected physical allocation");
        auto local = view(b, pools[ai], r, element, ai, allocationOffsets[ai]);
        Value host = body(herd)->getArgument(4 + abi[r.storage]);
        SmallVector<Value> dmaOperands;
        if (previousDMA)
          dmaOperands.push_back(previousDMA);
        if (input) {
          dmaOperands.push_back(local);
          dmaOperands.push_back(host);
          llvm::append_range(dmaOperands, dynamicHostOffsets);
        } else {
          dmaOperands.push_back(host);
          llvm::append_range(dmaOperands, dynamicHostOffsets);
          dmaOperands.push_back(local);
        }
        auto *dma = op(b, "air.dma_memcpy_nd", dmaOperands, {token});
        dma->setAttr(
            "operandSegmentSizes",
            b.getDenseI32ArrayAttr(
                {previousDMA ? 1 : 0, 1,
                 input ? 0 : int32_t(dynamicHostOffsets.size()), 0, 0, 1,
                 input ? int32_t(dynamicHostOffsets.size()) : 0, 0, 0}));
        previousDMA = dma->getResult(0);
        for (bool source : {true, false}) {
          bool onHost = source == input;
          std::string prefix = source ? "static_src_" : "static_dst_";
          dma->setAttr(
              prefix + "offsets",
              b.getDenseI64ArrayAttr(
                  onHost ? (dynamicHostOffsets.empty()
                                ? r.offsets
                                : std::vector<int64_t>(r.offsets.size(),
                                                       ShapedType::kDynamic))
                         : std::vector<int64_t>(r.sizes.size(), 0)));
          dma->setAttr(prefix + "sizes", b.getDenseI64ArrayAttr(r.sizes));
          dma->setAttr(
              prefix + "strides",
              b.getDenseI64ArrayAttr(strides(
                  onHost ? g.source.storages[r.storage].shape : r.sizes)));
        }
        dma->setAttr("orbit.public_event", b.getI64IntegerAttr(ei));
        dma->setAttr("orbit.storage",
                     b.getStringAttr(g.source.storages[r.storage].id));
        dma->setAttr("orbit.transfer",
                     b.getStringAttr(input ? "host_to_l1" : "l1_to_host"));
        op(b, "air.wait_all", dma->getResults());
      } else {
        const auto &inst = *instance[ei];
        std::vector<size_t> copies = ev.reads;
        copies.insert(copies.end(), ev.writes.begin(), ev.writes.end());
        std::set<size_t> storages;
        for (auto ci : copies)
          storages.insert(g.allocations[g.copies[ci].allocation].storage);
        SmallVector<Value> locals(g.source.storages.size());
        std::vector<MappingRegion> regions;
        for (auto st : storages) {
          bool hasRead = llvm::any_of(ev.reads, [&](size_t ci) {
            return g.allocations[g.copies[ci].allocation].storage == st;
          });
          bool hasWrite = llvm::any_of(ev.writes, [&](size_t ci) {
            return g.allocations[g.copies[ci].allocation].storage == st;
          });
          MappingRegion r, readRegion;
          size_t ai = 0, readAllocation = 0;
          if (!rectangle(g, hasWrite ? ev.writes : ev.reads, st, r, ai, e))
            return false;
          if (!originContract(g, hasWrite ? ev.writes : ev.reads, r, ai, e))
            return false;
          if (!pools.count(ai))
            return fail(e, "computation accesses a nonlocal allocation");
          auto element = cast<MemRefType>(g.source.storages[st].root.getType())
                             .getElementType();
          auto itemBytes = int64_t((element.getIntOrFloatBitWidth() + 7) / 8);
          if (elements(r.sizes) > g.allocations[ai].bytes / itemBytes)
            return fail(e, "source view exceeds selected physical allocation");
          locals[st] =
              view(b, pools[ai], r, element, ai, allocationOffsets[ai]);
          regions.push_back(r);
          if (hasRead && hasWrite) {
            if (!rectangle(g, ev.reads, st, readRegion, readAllocation, e) ||
                !originContract(g, ev.reads, readRegion, readAllocation, e))
              return false;
            if (readRegion.offsets != r.offsets || readRegion.sizes != r.sizes)
              return fail(e, "read-modify-write selected copies have different "
                             "source rectangles");
            if (!pools.count(readAllocation))
              return fail(e, "read-modify-write source copy is not core-local");
            if (elements(readRegion.sizes) >
                g.allocations[readAllocation].bytes / itemBytes)
              return fail(
                  e,
                  "old-version view exceeds its selected physical allocation");
            if (readAllocation != ai) {
              // Source RMW uses one memref. Preserve its old logical version in
              // the selected source allocation, initialize the distinct new
              // allocation bit-for-bit, then perform the original scalar
              // accumulation there.
              Value old =
                  view(b, pools[readAllocation], readRegion, element,
                       readAllocation, allocationOffsets[readAllocation]);
              SmallVector<Value> ivs;
              SmallVector<scf::ForOp> loops;
              for (auto extent : r.sizes) {
                auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
                auto upper = b.create<arith::ConstantIndexOp>(loc, extent);
                auto step = b.create<arith::ConstantIndexOp>(loc, 1);
                auto loop = b.create<scf::ForOp>(loc, zero, upper, step);
                loop->setAttr("orbit.local_version_staging",
                              b.getI64IntegerAttr(ei));
                loops.push_back(loop);
                ivs.push_back(loop.getInductionVar());
                b.setInsertionPointToStart(loop.getBody());
              }
              auto value = b.create<memref::LoadOp>(loc, old, ivs);
              b.create<memref::StoreOp>(loc, value, locals[st], ivs);
              if (!loops.empty())
                b.setInsertionPointAfter(loops.front());
              auto bytes = elements(r.sizes) * itemBytes;
              if (localStagingBytes > INT64_MAX - bytes)
                return fail(e, "local staging byte counter overflow");
              localStagingBytes += bytes;
              j::Array offsets, sizes;
              for (auto v : r.offsets)
                offsets.push_back(v);
              for (auto v : r.sizes)
                sizes.push_back(v);
              stagingRecords.push_back(
                  j::Object{{"event", int64_t(ei)},
                            {"storage", int64_t(st)},
                            {"source_allocation", int64_t(readAllocation)},
                            {"target_allocation", int64_t(ai)},
                            {"bytes", bytes},
                            {"offsets", std::move(offsets)},
                            {"sizes", std::move(sizes)}});
            }
          }
        }
        bool matrixHandled = false;
        if (!emitProgramMatrixInstance(b, g, inst, regions, locals,
                                       matrixHandled, e) ||
            (!matrixHandled &&
             !emitProgramGraphScalarInstance(b, g, inst, regions, locals, e)))
          return false;
      }
      eventRecords.push_back(j::Object{{"event", int64_t(ei)},
                                       {"column", c.col},
                                       {"row", c.row},
                                       {"reads", indices(ev.reads)},
                                       {"writes", indices(ev.writes)}});
      return true;
    };
    if (loops) {
      if (!emitProgramGraphAIRLoopEvents(b, *loops, emitEvent, e))
        return false;
    } else {
      for (auto ei : c.events)
        if (!emitEvent(ei, {}))
          return false;
    }
    SmallVector<Value> physicalPools;
    for (auto &[ai, pool] : pools)
      if (!llvm::is_contained(physicalPools, pool))
        physicalPools.push_back(pool);
    for (auto pool : physicalPools)
      b.create<memref::DeallocOp>(loc, pool);
    op(b, "air.herd_terminator");
    coreRecords.push_back(j::Object{{"owner", id},
                                    {"memory", c.memory},
                                    {"column", c.col},
                                    {"row", c.row},
                                    {"static_payload_bytes", c.bytes},
                                    {"logical_groups", indices(c.groups)},
                                    {"events", indices(c.events)}});
  }
  b.setInsertionPointToEnd(body(segment));
  op(b, "air.segment_terminator");
  b.setInsertionPointToEnd(body(launch));
  op(b, "air.launch_terminator");
  b.setInsertionPointToEnd(fb);
  b.create<func::ReturnOp>(loc);
  // Refine only the emitted scalar conversion. Source/selected graph stay
  // unchanged, and integer RNE does not modify the core rounding register.
  orbit::taskflow::detail::lowerScalarBF16RNE(*out);
  PassManager pm(ctx);
  pm.addPass(createLowerAffinePass());
  if (failed(pm.run(*out)) || failed(verify(*out)))
    return fail(e, "generated public graph AIR failed affine/SSA verification");
  out->getOperation()->setAttr("orbit.program_graph",
                               b.getStringAttr(programGraphIdentity(g)));
  j::Array buffers, outputs, abiIds;
  for (auto st : abiStorage) {
    j::Array shape;
    for (auto x : g.source.storages[st].shape)
      shape.push_back(x);
    auto &s = g.source.storages[st];
    buffers.push_back(j::Object{{"name", s.id},
                                {"role", s.role},
                                {"dtype", s.dtype},
                                {"shape", std::move(shape)}});
    abiIds.push_back(int64_t(st));
    if (s.role == "output")
      outputs.push_back(s.id);
  }
  manifest = j::Object{
      {"schema", "orbit.public-program-graph.air.v26"},
      {"graph_id", programGraphIdentity(g)},
      {"graph_copy_compute_realization", !selection},
      {"selected_event_range_realized", bool(selection)},
      {"analytical_group_setups_realized", false},
      {"final_aie_checked", false},
      {"physical_configurations", 1},
      {"physical_allocation_refinement",
       "distinct selected allocations packed into disjoint static slices per "
       "storage; no alias or lifetime reuse"},
      {"local_staging_payload_bytes", localStagingBytes},
      {"local_staging", std::move(stagingRecords)},
      {"analytical_local_staging_charged", false},
      {"dma_dependency_policy",
       "explicit per-core selected-transfer token chain"},
      {"additional_dma_serialization", true},
      {"matrix_realization", programMatrixAIRManifest(*out)},
      {"loop_realization",
       loops ? j::Value(loops->witness) : j::Value(nullptr)},
      {"cores", std::move(coreRecords)},
      {"events", std::move(eventRecords)},
      {"allocations", std::move(allocationRecords)},
      {"abi_storage_indices", std::move(abiIds)},
      {"abi", j::Object{{"buffers", std::move(buffers)},
                        {"outputs", std::move(outputs)}}},
      {"scope", "direct selected host/core paths; one herd per physical core; "
                "exact finite scalar source instances; static distinct "
                "allocations; conservative per-core event order; logical "
                "groups share configuration; no timed-dispatch claim"}};
  if (selection) {
    (*manifest.getAsObject())["selection"] =
        j::Object{{"begin_event", int64_t(selection->begin)},
                  {"end_event_exclusive", int64_t(selection->end)},
                  {"whole_graph_event_count", int64_t(g.events.size())},
                  {"requires_complete_union_and_serial_wait2", true}};
  }
  return true;
}
