#include "Taskflow/ProgramGraphAIE.h"
#include "Taskflow/ProgramGraphAIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
using namespace mlir;
using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
constexpr size_t traceLimit = 1000000, expressionLimit = 16384;
template <class T> T field(Operation *o, StringRef key) {
  if (auto a = o->getAttrOfType<T>(key))
    return a;
  if (auto p = dyn_cast_or_null<DictionaryAttr>(o->getPropertiesAsAttribute()))
    return p.getAs<T>(key);
  return {};
}
bool named(Operation *o, StringRef n) {
  return o && o->getName().getStringRef() == n;
}
int64_t integer(Operation *o, StringRef key, int64_t fallback = -1) {
  auto a = field<IntegerAttr>(o, key);
  return a ? a.getInt() : fallback;
}
template <class T> std::string printed(T v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  v.print(os);
  return s;
}
using Coordinate = std::pair<int64_t, int64_t>;
Coordinate coordinate(Value tile) {
  auto *o = tile.getDefiningOp();
  return named(o, "aie.tile") ? Coordinate(integer(o, "col"), integer(o, "row"))
                              : Coordinate(-1, -1);
}
Value root(Value v) {
  while (auto *o = v.getDefiningOp()) {
    if (!isa<memref::CastOp, memref::ReinterpretCastOp>(o))
      break;
    v = o->getOperand(0);
  }
  return v;
}
struct VLess {
  bool operator()(Value a, Value b) const {
    return std::less<const void *>{}(a.getAsOpaquePointer(),
                                     b.getAsOpaquePointer());
  }
};
struct LockUse {
  Value lock;
  int64_t action;
  size_t position;
};
struct Access {
  size_t allocation;
  bool write;
  int64_t offset;
  std::string expression;
  bool operator==(const Access &b) const {
    return allocation == b.allocation && write == b.write &&
           offset == b.offset && expression == b.expression;
  }
};
struct Address {
  size_t allocation;
  int64_t offset;
  std::vector<int64_t> shape, strides;
};
// Execute only a bounded static scalar subset. Memory is not evaluated with
// random inputs: an ordered load gets a fresh symbol and every store retains
// its exact expression in those symbols. Moving a load across a store changes
// the trace, even if the source arithmetic op multiset stays unchanged.
struct ScalarTrace {
  llvm::DenseMap<Value, int64_t> integers;
  llvm::DenseMap<Value, std::string> expressions;
  llvm::DenseMap<Value, Address> addresses;
  std::vector<Access> accesses;
  std::vector<LockUse> locks;
  std::vector<std::pair<size_t, size_t>> transferPositions;
  std::map<size_t, Value> buffers;
  std::map<size_t, int64_t> bases, limits;
  size_t steps = 0, loads = 0, expressionBytes = 0;
  std::string error;
  bool actual = false;
  bool fail(const std::string &s) {
    error = s;
    return false;
  }
  bool tick() {
    return ++steps <= traceLimit ||
           fail("static scalar trace exceeds one million operations");
  }
  bool number(Value v, int64_t &out) {
    auto i = integers.find(v);
    if (i != integers.end()) {
      out = i->second;
      return true;
    }
    if (auto c = v.getDefiningOp<arith::ConstantOp>())
      if (auto a = dyn_cast<IntegerAttr>(c.getValue())) {
        out = a.getInt();
        return true;
      }
    return fail("nonstatic scalar address/control operand");
  }
  bool address(Value v, Address &a) {
    auto it = addresses.find(v);
    if (it != addresses.end()) {
      a = it->second;
      return true;
    }
    auto *o = v.getDefiningOp();
    if (!o)
      return fail("unbound scalar buffer");
    if (auto cast = dyn_cast<memref::CastOp>(o)) {
      if (!address(cast.getSource(), a))
        return false;
      addresses[v] = a;
      return true;
    }
    if (auto view = dyn_cast<memref::ReinterpretCastOp>(o)) {
      if (!address(view.getSource(), a))
        return false;
      if (!view.getOffsets().empty() || !view.getSizes().empty() ||
          !view.getStrides().empty() || view.getStaticOffsets().size() != 1)
        return fail("dynamic reinterpret cast");
      a.offset = view.getStaticOffsets()[0];
      a.shape.assign(view.getStaticSizes().begin(),
                     view.getStaticSizes().end());
      a.strides.assign(view.getStaticStrides().begin(),
                       view.getStaticStrides().end());
      auto id = field<IntegerAttr>(o, "orbit.allocation");
      if (!id || id.getInt() < 0 || !buffers.count(id.getInt()) ||
          buffers[id.getInt()] != root(view.getSource()) ||
          a.offset < bases[id.getInt()])
        return fail("view allocation tag differs from actual SSA buffer/slice");
      a.allocation = id.getInt();
      a.offset -= bases[a.allocation];
      addresses[v] = a;
      return true;
    }
    if (!named(o, actual ? "aie.buffer" : "memref.alloc"))
      return fail("unsupported scalar buffer root");
    auto t = dyn_cast<MemRefType>(v.getType());
    if (!t || !t.hasStaticShape() || !t.getLayout().isIdentity())
      return fail("nonstatic/nonidentity scalar pool layout");
    bool known = false;
    for (auto &entry : buffers)
      known |= entry.second == v;
    if (!known)
      return fail("buffer has no checked slice provenance");
    a.allocation = SIZE_MAX;
    a.offset = 0;
    a.shape.assign(t.getShape().begin(), t.getShape().end());
    a.strides.assign(t.getRank(), 1);
    for (int64_t d = t.getRank() - 2; d >= 0; --d)
      a.strides[d] = a.strides[d + 1] * t.getDimSize(d + 1);
    addresses[v] = a;
    return true;
  }
  bool expression(Value v, std::string &out) {
    auto it = expressions.find(v);
    if (it != expressions.end()) {
      out = it->second;
      return true;
    }
    int64_t n;
    if (integers.count(v)) {
      out = "integer(" + std::to_string(integers[v]) + ":" +
            printed(v.getType()) + ")";
      return true;
    }
    if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
      out = "constant(" + printed(c.getValue()) + ")";
      expressions[v] = out;
      return true;
    }
    return fail("unsupported scalar SSA expression");
  }
  bool block(Block &b) {
    for (auto &o : b)
      if (!operation(&o))
        return false;
    return true;
  }
  bool operation(Operation *o) {
    if (!tick())
      return false;
    StringRef name = o->getName().getStringRef();
    if (auto loop = dyn_cast<scf::ForOp>(o)) {
      if (loop.getNumResults() || !loop.getInitArgs().empty())
        return fail("loop-carried SSA is outside finite trace subset");
      int64_t lo, hi, step;
      if (!number(loop.getLowerBound(), lo) ||
          !number(loop.getUpperBound(), hi) || !number(loop.getStep(), step) ||
          step <= 0 || hi < lo)
        return fail("nonstatic/negative scalar loop");
      __int128 trips = (__int128(hi) - lo + step - 1) / step;
      if (trips > traceLimit)
        return fail("scalar loop trip count exceeds trace cap");
      for (int64_t n = 0; n < int64_t(trips); ++n) {
        int64_t i = int64_t(__int128(lo) + __int128(n) * step);
        integers[loop.getInductionVar()] = i;
        if (!block(*loop.getBody()))
          return false;
      }
      return true;
    }
    if (isa<arith::ConstantOp>(o)) {
      auto c = cast<arith::ConstantOp>(o);
      if (auto a = dyn_cast<IntegerAttr>(c.getValue()))
        integers[c.getResult()] = a.getInt();
      expressions[c.getResult()] = "constant(" + printed(c.getValue()) + ")";
      return true;
    }
    if (isa<memref::AllocOp, memref::ReinterpretCastOp, memref::CastOp>(o)) {
      Address a;
      return address(o->getResult(0), a);
    }
    if (name == "air.dma_memcpy_nd") {
      auto event = field<IntegerAttr>(o, "orbit.public_event");
      if (!event || event.getInt() < 0)
        return fail("reference DMA lacks public event");
      transferPositions.push_back({size_t(event.getInt()), accesses.size()});
      return true;
    }
    if (isa<memref::DeallocOp, scf::YieldOp>(o) || name == "air.wait_all" ||
        name == "air.herd_terminator" || name == "aie.end")
      return true;
    if (name == "aie.use_lock") {
      int64_t amount;
      if (o->getNumOperands() != 2 || !number(o->getOperand(1), amount) ||
          amount != 1)
        return fail("nonunit actual lock amount");
      int64_t action = integer(o, "action");
      if (action != 1 && action != 2)
        return fail("unsupported actual lock action");
      locks.push_back({o->getOperand(0), action, accesses.size()});
      return true;
    }
    Value mem;
    SmallVector<Value> indices;
    bool write = false;
    Value stored;
    if (auto load = dyn_cast<memref::LoadOp>(o)) {
      mem = load.getMemRef();
      llvm::append_range(indices, load.getIndices());
    }
    if (auto store = dyn_cast<memref::StoreOp>(o)) {
      mem = store.getMemRef();
      llvm::append_range(indices, store.getIndices());
      stored = store.getValue();
      write = true;
    }
    if (mem) {
      Address a;
      if (!address(mem, a) || indices.size() != a.shape.size())
        return fail("scalar address rank mismatch");
      int64_t offset = a.offset;
      for (size_t d = 0; d < indices.size(); ++d) {
        int64_t n;
        if (!number(indices[d], n) || n < 0 || n >= a.shape[d] ||
            a.strides[d] < 0 ||
            n > INT64_MAX / std::max<int64_t>(1, a.strides[d]) ||
            offset > INT64_MAX - n * a.strides[d])
          return fail("scalar address outside view/overflow");
        offset += n * a.strides[d];
      }
      if (a.allocation == SIZE_MAX || !limits.count(a.allocation))
        return fail("scalar payload does not identify a checked slice");
      if (offset < 0 || offset >= limits[a.allocation])
        return fail("scalar access outside pool");
      std::string expr;
      if (write) {
        if (!expression(stored, expr))
          return false;
      } else {
        expr = "load" + std::to_string(loads++) + ":" +
               printed(o->getResult(0).getType());
        expressions[o->getResult(0)] = expr;
      }
      if (expr.size() > 64 * 1024 * 1024 - expressionBytes)
        return fail("scalar trace expression storage exceeds 64 MiB");
      expressionBytes += expr.size();
      accesses.push_back({a.allocation, write, offset, expr});
      return true;
    }
    if (name.starts_with("arith.") || name.starts_with("math.")) {
      if (o->getNumResults() != 1 || o->getNumRegions() ||
          o->getNumSuccessors())
        return fail("unsupported arithmetic shape");
      std::string expr =
          name.str() + "[" + printed(o->getResult(0).getType()) + "](";
      std::map<std::string, std::string> attrs;
      for (auto attr : o->getAttrs())
        if (!attr.getName().strref().starts_with("orbit."))
          attrs[attr.getName().str()] = printed(attr.getValue());
      if (auto props =
              dyn_cast_or_null<DictionaryAttr>(o->getPropertiesAsAttribute()))
        for (auto attr : props)
          attrs[attr.getName().str()] = printed(attr.getValue());
      for (auto &a : attrs)
        expr += a.first + "=" + a.second + ";";
      for (auto operand : o->getOperands()) {
        std::string x;
        if (!expression(operand, x))
          return false;
        expr += x + ",";
      }
      expr += ")";
      if (expr.size() > expressionLimit)
        return fail("scalar expression exceeds bounded trace size");
      expressions[o->getResult(0)] = expr;
      // Address arithmetic is interpreted exactly in signed 64-bit integers.
      if (o->getNumOperands() == 2 &&
          isa<IntegerType, IndexType>(o->getResult(0).getType())) {
        int64_t x, y;
        if (!number(o->getOperand(0), x) || !number(o->getOperand(1), y))
          return false;
        __int128 z = 0;
        if (name == "arith.addi")
          z = __int128(x) + y;
        else if (name == "arith.subi")
          z = __int128(x) - y;
        else if (name == "arith.muli")
          z = __int128(x) * y;
        else if (name == "arith.divsi" && y && !(x == INT64_MIN && y == -1))
          z = x / y;
        else if (name == "arith.remsi" && y && !(x == INT64_MIN && y == -1))
          z = x % y;
        else
          return fail("unsupported integer address arithmetic");
        if (z < INT64_MIN || z > INT64_MAX)
          return fail("integer address arithmetic overflow");
        integers[o->getResult(0)] = int64_t(z);
      } else if (name == "arith.index_cast" || name == "arith.index_castui") {
        int64_t n;
        if (!number(o->getOperand(0), n))
          return false;
        integers[o->getResult(0)] = n;
      }
      return true;
    }
    return fail("unsupported executable scalar operation " + name.str());
  }
};
struct Stream {
  Value buffer, free, ready;
  size_t allocation = 0;
  bool input = false;
  int64_t channel = 0;
};
struct Rectangle {
  size_t allocation = 0, storage = 0;
  bool input = false;
  std::vector<int64_t> addresses;
  bool operator==(const Rectangle &x) const {
    return allocation == x.allocation && storage == x.storage &&
           input == x.input && addresses == x.addresses;
  }
  bool operator<(const Rectangle &x) const {
    return std::tie(allocation, storage, input, addresses) <
           std::tie(x.allocation, x.storage, x.input, x.addresses);
  }
};
bool expandAddresses(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides,
                     int64_t offset, std::vector<int64_t> &out) {
  if (sizes.size() != strides.size() || offset < 0)
    return false;
  int64_t total = 1;
  for (auto n : sizes) {
    if (n <= 0 || total > int64_t(traceLimit) / n)
      return false;
    total *= n;
  }
  std::function<bool(size_t, int64_t)> rec = [&](size_t d, int64_t address) {
    if (d == sizes.size()) {
      out.push_back(address);
      return true;
    }
    for (int64_t i = 0; i < sizes[d]; ++i) {
      __int128 next = __int128(address) + __int128(i) * strides[d];
      if (next < 0 || next > INT64_MAX || !rec(d + 1, int64_t(next)))
        return false;
    }
    return true;
  };
  return rec(0, offset);
}
} // namespace

bool orbit::taskflow::checkProgramGraphAIE(const ProgramGraph &g,
                                           ModuleOp expectedAIR,
                                           ModuleOp actualAIE,
                                           j::Value &receipt,
                                           std::string &error) {
  error.clear();
  receipt = nullptr;
  auto reject = [&](const std::string &s) {
    error = "public graph final AIE: " + s;
    receipt = j::Object{{"schema", "orbit.public-program-graph.aie.v27"},
                        {"status", "failed"},
                        {"message", error},
                        {"graph_id", programGraphIdentity(g)}};
    return false;
  };
  if (g.operatorSource)
    return reject("external operator source requires its registered runtime "
                  "implementation");
  if (!expectedAIR || !actualAIE)
    return reject("missing expected AIR or final AIE");
  if (failed(verify(expectedAIR)) || failed(verify(actualAIE)))
    return reject("invalid module SSA/registered operation verification");
  OwningOpRef<ModuleOp> regenerated;
  j::Value manifest = nullptr;
  std::string detail;
  if (!lowerProgramGraphToAIR(g, regenerated, manifest, detail))
    return reject("cannot regenerate source-backed AIR: " + detail);
  if (printed(*regenerated) != printed(expectedAIR))
    return reject("expected AIR differs from regenerated frozen graph/source "
                  "realization");
  auto *m = manifest.getAsObject();
  auto *abi = m ? m->getArray("abi_storage_indices") : nullptr;
  auto *coreRecords = m ? m->getArray("cores") : nullptr;
  if (!abi || !coreRecords)
    return reject("missing regenerated physical manifest");
  std::vector<size_t> abiStorage;
  for (auto &v : *abi) {
    auto n = v.getAsInteger();
    if (!n || *n < 0 || size_t(*n) >= g.source.storages.size())
      return reject("bad regenerated ABI index");
    abiStorage.push_back(*n);
  }
  OwningOpRef<ModuleOp> normalized = cast<ModuleOp>(expectedAIR->clone());
  PassManager simplify(expectedAIR.getContext());
  simplify.addPass(createCanonicalizerPass());
  simplify.addPass(createCSEPass());
  if (failed(simplify.run(*normalized)))
    return reject("reference scalar canonicalization failed");
  std::map<Coordinate, Operation *> herds, cores;
  bool bad = false;
  normalized->walk([&](Operation *o) {
    if (named(o, "air.herd")) {
      Coordinate c(integer(o, "x_loc"), integer(o, "y_loc"));
      if (!herds.emplace(c, o).second)
        bad = true;
    }
  });
  Operation *device = nullptr, *runtime = nullptr;
  std::vector<Operation *> devices, configures, runs;
  actualAIE.walk([&](Operation *o) {
    if (named(o, "aie.device"))
      devices.push_back(o);
    if (named(o, "aiex.configure"))
      configures.push_back(o);
    if (named(o, "aiex.run"))
      runs.push_back(o);
    if (named(o, "aie.core")) {
      if (o->getNumOperands() != 1 ||
          !cores.emplace(coordinate(o->getOperand(0)), o).second)
        bad = true;
      auto *parent = o->getParentOp();
      if (device && device != parent)
        bad = true;
      device = parent;
    }
  });
  if (bad || !device || !named(device, "aie.device") ||
      cores.size() != herds.size())
    return reject("actual physical core ownership/count differs");
  if (integer(device, "device") != 8)
    return reject("actual target is not pinned NPU2");
  if (configures.empty()) {
    if (devices.size() != 1 || !runs.empty())
      return reject("unsupported unwrapped device structure");
    device->walk([&](Operation *o) {
      if (named(o, "aie.runtime_sequence")) {
        if (runtime)
          bad = true;
        runtime = o;
      }
    });
  } else {
    if (configures.size() != 1 || runs.size() != 1 || devices.size() != 2)
      return reject("requires exactly one physical configure/run");
    auto *configure = configures.front(), *run = runs.front();
    auto ds = field<StringAttr>(device, "sym_name");
    auto target = field<FlatSymbolRefAttr>(configure, "device");
    if (!target)
      target = field<FlatSymbolRefAttr>(configure, "symbol");
    if (!target)
      target = field<FlatSymbolRefAttr>(configure, "device_name");
    if (!target)
      target = field<FlatSymbolRefAttr>(configure, "target");
    auto sequence = field<FlatSymbolRefAttr>(run, "runtime_sequence_symbol");
    if (!ds || !target || target.getValue() != ds.getValue() || !sequence ||
        run->getParentOp() != configure)
      return reject("configure/run symbolic target differs");
    device->walk([&](Operation *o) {
      if (named(o, "aie.runtime_sequence")) {
        auto s = field<StringAttr>(o, "sym_name");
        if (s && s.getValue() == sequence.getValue()) {
          if (runtime)
            bad = true;
          runtime = o;
        }
      }
    });
    auto *wrapper = configure->getParentOp();
    if (!named(wrapper, "aie.runtime_sequence") ||
        wrapper->getNumRegions() != 1 ||
        !llvm::hasSingleElement(wrapper->getRegion(0)))
      return reject("unsupported invocation wrapper");
    auto &block = wrapper->getRegion(0).front();
    if (run->getNumOperands() != block.getNumArguments())
      return reject("incomplete caller ABI forwarding");
    for (size_t i = 0; i < run->getNumOperands(); ++i)
      if (run->getOperand(i) != block.getArgument(i))
        return reject("caller ABI reordered by run");
  }
  if (bad || !runtime || runtime->getNumRegions() != 1 ||
      !llvm::hasSingleElement(runtime->getRegion(0)) ||
      runtime->getRegion(0).front().getNumArguments() != abiStorage.size())
    return reject("actual runtime ABI differs");
  auto &runtimeBlock = runtime->getRegion(0).front();
  for (size_t i = 0; i < abiStorage.size(); ++i)
    if (runtimeBlock.getArgument(i).getType() !=
        g.source.storages[abiStorage[i]].root.getType())
      return reject("actual runtime BO shape/type differs");
  std::map<Coordinate, ScalarTrace> traces;
  std::map<Value, size_t, VLess> actualAllocations;
  std::map<size_t, Value> logicalBuffers;
  std::map<Value, Coordinate, VLess> bufferOwners;
  size_t scalarAccesses = 0;
  j::Array coreReceipts;
  for (auto &[xy, herd] : herds) {
    auto it = cores.find(xy);
    if (it == cores.end())
      return reject("selected physical core absent");
    auto *core = it->second;
    if (core->getNumRegions() != 1)
      return reject("unsupported core CFG");
    ScalarTrace expected, actual;
    actual.actual = true;
    herd->walk([&](memref::AllocOp pool) {
      auto ids = field<DenseI64ArrayAttr>(pool, "orbit.allocations"),
           offsets = field<DenseI64ArrayAttr>(
               pool, "orbit.allocation_offsets_elements");
      if (ids && offsets && ids.size() == offsets.size()) {
        for (size_t q = 0; q < ids.size(); ++q) {
          if (ids[q] < 0 || size_t(ids[q]) >= g.allocations.size() ||
              offsets[q] < 0 || expected.buffers.count(ids[q])) {
            bad = true;
            continue;
          }
          expected.buffers[ids[q]] = pool.getResult();
          expected.bases[ids[q]] = offsets[q];
        }
      } else if (auto id = field<IntegerAttr>(pool, "orbit.allocation")) {
        if (id.getInt() < 0 || size_t(id.getInt()) >= g.allocations.size() ||
            expected.buffers.count(id.getInt()))
          bad = true;
        else {
          expected.buffers[id.getInt()] = pool.getResult();
          expected.bases[id.getInt()] = 0;
        }
      } else
        bad = true;
    });
    for (auto &[ai, v] : expected.buffers) {
      auto type = cast<MemRefType>(v.getType());
      auto bytes = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
      if (g.allocations[ai].bytes % bytes) {
        bad = true;
        continue;
      }
      expected.limits[ai] = g.allocations[ai].bytes / bytes;
      actual.limits[ai] = expected.limits[ai];
      actual.bases[ai] = expected.bases[ai];
    }
    if (bad)
      return reject("invalid reference physical slice manifest");
    if (!expected.block(herd->getRegion(0).front()))
      return reject("reference scalar trace: " + expected.error);
    // Provenance is cross-checked at the pool AND at every source-backed view.
    core->walk([&](memref::ReinterpretCastOp view) {
      auto id = field<IntegerAttr>(view, "orbit.allocation");
      if (!id || id.getInt() < 0) {
        bad = true;
        return;
      }
      auto v = root(view.getSource());
      auto *pool = v.getDefiningOp();
      if (!named(pool, "aie.buffer")) {
        bad = true;
        return;
      }
      auto old = logicalBuffers.find(id.getInt());
      if (old != logicalBuffers.end() && old->second != v)
        bad = true;
      logicalBuffers[id.getInt()] = v;
      actualAllocations.emplace(v, id.getInt());
      bufferOwners[v] = xy;
    });
    if (bad)
      return reject("actual views do not bind unique selected buffers");
    for (auto &[ai, v] : logicalBuffers)
      if (bufferOwners[v] == xy) {
        auto *pool = v.getDefiningOp();
        auto found = expected.buffers.find(ai);
        if (found == expected.buffers.end())
          return reject("actual extra selected slice");
        auto t = dyn_cast<MemRefType>(v.getType()),
             et = cast<MemRefType>(found->second.getType());
        if (!t || t != et || pool->getNumOperands() != 1 ||
            pool->getOperand(0) != core->getOperand(0))
          return reject("actual pool type/physical owner mismatch");
        if (field<IntegerAttr>(pool, "address"))
          return reject(
              "explicit physical addresses require independent overlap proof");
        actual.buffers[ai] = v;
      }
    for (auto &[a, av] : actual.buffers)
      for (auto &[b, bv] : actual.buffers)
        if ((av == bv) != (expected.buffers[a] == expected.buffers[b]))
          return reject("physical pool alias partition differs from reference");
    for (auto &[a, v] : actual.buffers) {
      if (actual.bases[a] >
          cast<MemRefType>(v.getType()).getNumElements() - actual.limits[a])
        return reject("selected slice exceeds physical pool");
      for (auto &[b, w] : actual.buffers)
        if (a < b && v == w &&
            std::max(actual.bases[a], actual.bases[b]) <
                std::min(actual.bases[a] + actual.limits[a],
                         actual.bases[b] + actual.limits[b]))
          return reject("selected physical slices overlap");
    }
    if (actual.buffers.size() != expected.buffers.size())
      return reject("missing actual local pool");
    auto &region = core->getRegion(0);
    if (llvm::hasSingleElement(region)) {
      if (!actual.block(region.front()))
        return reject("actual scalar trace: " + actual.error);
    } else {
      if (std::distance(region.begin(), region.end()) != 2)
        return reject("unsupported actual core CFG (requires entry + one "
                      "self-loop block)");
      auto &entry = region.front();
      auto &cycle = *std::next(region.begin());
      auto *jump = entry.getTerminator(), *back = cycle.getTerminator();
      if (!named(jump, "cf.br") || !named(back, "cf.br") ||
          jump->getNumSuccessors() != 1 || back->getNumSuccessors() != 1 ||
          jump->getSuccessor(0) != &cycle || back->getSuccessor(0) != &cycle ||
          jump->getNumOperands() || back->getNumOperands() ||
          cycle.getNumArguments())
        return reject("core CFG is not a proved unconditional invocation loop");
      for (auto &op : entry)
        if (&op != jump && !actual.operation(&op))
          return reject(actual.error);
      for (auto &op : cycle)
        if (&op != back && !actual.operation(&op))
          return reject(actual.error);
    }
    if (expected.accesses != actual.accesses) {
      size_t q = 0;
      while (q < std::min(expected.accesses.size(), actual.accesses.size()) &&
             expected.accesses[q] == actual.accesses[q])
        ++q;
      std::string why = "actual scalar trace mismatch at " + std::to_string(q) +
                        " counts " + std::to_string(expected.accesses.size()) +
                        "/" + std::to_string(actual.accesses.size());
      if (q < expected.accesses.size())
        why += " expected=" + expected.accesses[q].expression;
      if (q < actual.accesses.size())
        why += " actual=" + actual.accesses[q].expression;
      return reject(why);
    }
    actual.transferPositions = expected.transferPositions;
    scalarAccesses += actual.accesses.size();
    if (scalarAccesses > traceLimit)
      return reject(
          "aggregate scalar access trace exceeds one million operations");
    coreReceipts.push_back(j::Object{
        {"column", xy.first},
        {"row", xy.second},
        {"allocations", int64_t(actual.buffers.size())},
        {"scalar_memory_operations", int64_t(actual.accesses.size())}});
    traces.emplace(xy, std::move(actual));
  }
  // Every actual buffer/lock/memory engine must have exactly the selected core
  // owner. Cross-owner references cannot hide behind valid allocation labels.
  size_t bufferCount = 0;
  std::map<Coordinate, std::set<int64_t>> lockIds;
  device->walk([&](Operation *o) {
    auto n = o->getName().getStringRef();
    if (n == "aie.external_buffer" && !o->use_empty())
      bad = true;
    if (n == "aie.buffer") {
      ++bufferCount;
      if (o->getNumResults() != 1 || !actualAllocations.count(o->getResult(0)))
        bad = true;
    }
    if (n == "aie.lock") {
      if (o->getNumOperands() != 1) {
        bad = true;
        return;
      }
      auto xy = coordinate(o->getOperand(0));
      auto id = integer(o, "lockID");
      if (!cores.count(xy) || id < 0 || id > 15 ||
          !lockIds[xy].insert(id).second)
        bad = true;
    }
    if (n == "aie.mem" || n == "aie.memTileDMA") {
      if (n != "aie.mem" || o->getNumOperands() != 1 ||
          !cores.count(coordinate(o->getOperand(0))))
        bad = true;
    }
    if (n.starts_with("aie.objectfifo"))
      bad = true;
  });
  actualAIE.walk([&](Operation *o) {
    StringRef n = o->getName().getStringRef();
    if (o == actualAIE.getOperation())
      return;
    bool inCore = false;
    for (auto *p = o; p; p = p->getParentOp())
      if (named(p, "aie.core"))
        inCore = true;
    if (inCore)
      return;
    auto *parent = o->getParentOp();
    if (n == "aie.device" && parent != actualAIE.getOperation())
      bad = true;
    if ((n == "aie.tile" || n == "aie.lock" || n == "aie.buffer" ||
         n == "aie.external_buffer" || n == "aie.mem" || n == "aie.core" ||
         n == "aie.flow" || n == "aie.packet_flow" ||
         n == "aie.shim_dma_allocation" || n == "aie.runtime_sequence") &&
        !named(parent, "aie.device"))
      bad = true;
    if ((n == "aie.packet_source" || n == "aie.packet_dest") &&
        !named(parent, "aie.packet_flow"))
      bad = true;
    if ((n == "aie.next_bd" || n == "aie.dma_start" || n == "aie.use_lock" ||
         n == "arith.constant") &&
        !named(parent, "aie.mem"))
      bad = true;
    if (n == "aie.dma_bd" && !named(parent, "aie.mem") &&
        !named(parent, "aiex.dma_configure_task_for"))
      bad = true;
    if (n.starts_with("aiex.dma_") && parent != runtime)
      bad = true;
    if (n == "aiex.configure" && !named(parent, "aie.runtime_sequence"))
      bad = true;
    if (n == "aiex.run" && !named(parent, "aiex.configure"))
      bad = true;
    static const std::set<std::string> allowed = {"aie.device",
                                                  "aie.tile",
                                                  "aie.lock",
                                                  "aie.buffer",
                                                  "aie.external_buffer",
                                                  "aie.mem",
                                                  "aie.dma_start",
                                                  "aie.dma_bd",
                                                  "aie.use_lock",
                                                  "aie.next_bd",
                                                  "aie.end",
                                                  "aie.flow",
                                                  "aie.packet_flow",
                                                  "aie.packet_source",
                                                  "aie.packet_dest",
                                                  "aie.shim_dma_allocation",
                                                  "aie.runtime_sequence",
                                                  "aiex.configure",
                                                  "aiex.run",
                                                  "aiex.dma_configure_task_for",
                                                  "aiex.dma_start_task",
                                                  "aiex.dma_await_task",
                                                  "aiex.dma_free_task",
                                                  "arith.constant"};
    if (!allowed.count(n.str()))
      bad = true;
  });
  if (bad || bufferCount != actualAllocations.size())
    return reject(
        "extra/aliased buffer or unsupported memory/lock/topology owner");
  // Read each actual DMA channel as a complete cyclic list of dense slices.
  // Packet attributes are opaque in this build: accept only the installed,
  // explicitly parsed packet_info spelling and packet type zero.
  auto attribute = [&](Operation *o, StringRef key) -> Attribute {
    if (auto a = o->getAttr(key))
      return a;
    if (auto p =
            dyn_cast_or_null<DictionaryAttr>(o->getPropertiesAsAttribute()))
      return p.get(key);
    return {};
  };
  auto packet = [&](Operation *o, int64_t &id) -> bool {
    auto a = attribute(o, "packet");
    id = -1;
    if (!a)
      return true;
    std::string text = printed(a);
    StringRef s(text);
    if (!s.consume_front("#aie.packet_info<pkt_type = 0, pkt_id = ") ||
        !s.consume_back(">") || s.getAsInteger(10, id) || id < 0 || id > 31)
      return false;
    return true;
  };
  using Channel = std::tuple<Coordinate, bool, int64_t>;
  struct PhysicalDMA {
    Stream stream;
    int64_t packet = -1;
  };
  std::vector<PhysicalDMA> descriptors;
  std::map<Channel, std::vector<size_t>> channels;
  std::set<Operation *> checkedBDs, checkedStarts, checkedDMALocks;
  device->walk([&](Operation *start) {
    if (!named(start, "aie.dma_start"))
      return;
    auto *mem = start->getParentOp();
    if (!named(mem, "aie.mem") || mem->getNumOperands() != 1 ||
        start->getNumSuccessors() != 2 ||
        integer(start, "repeat_count", 0) != 0) {
      bad = true;
      return;
    }
    auto xy = coordinate(mem->getOperand(0));
    auto direction = integer(start, "channel_dir"),
         channel = integer(start, "channel");
    if (!cores.count(xy) || direction < 0 || direction > 1 || channel < 0 ||
        channel > 1) {
      bad = true;
      return;
    }
    Channel key{xy, direction == 0, channel};
    if (channels.count(key)) {
      bad = true;
      return;
    }
    auto &sequence = channels[key];
    std::set<Block *> visited;
    Block *first = start->getSuccessor(0), *block = first;
    do {
      if (!visited.insert(block).second || visited.size() > traceLimit) {
        bad = true;
        return;
      }
      Operation *bd = nullptr, *take = nullptr, *give = nullptr;
      size_t ordinal = 0, ti = 0, bi = 0, gi = 0;
      for (auto &o : *block) {
        ++ordinal;
        if (named(&o, "aie.dma_bd")) {
          if (bd)
            bad = true;
          bd = &o;
          bi = ordinal;
        } else if (named(&o, "aie.use_lock")) {
          if (integer(&o, "action") == 2) {
            if (take)
              bad = true;
            take = &o;
            ti = ordinal;
          } else if (integer(&o, "action") == 1) {
            if (give)
              bad = true;
            give = &o;
            gi = ordinal;
          } else
            bad = true;
        } else if (!named(&o, "aie.next_bd") && !isa<arith::ConstantOp>(&o))
          bad = true;
      }
      auto *tail = block->getTerminator();
      if (!bd || !take || !give || !(ti < bi && bi < gi) ||
          !named(tail, "aie.next_bd") || tail->getNumSuccessors() != 1 ||
          bd->getNumOperands() != 1 || take->getNumOperands() != 2 ||
          give->getNumOperands() != 2 || !checkedBDs.insert(bd).second) {
        bad = true;
        return;
      }
      auto buffer = root(bd->getOperand(0));
      if (!actualAllocations.count(buffer) || bufferOwners[buffer] != xy) {
        bad = true;
        return;
      }
      auto &trace = traces.at(xy);
      size_t logical = SIZE_MAX;
      for (auto &[ai, pool] : trace.buffers)
        if (pool == buffer && trace.bases[ai] == integer(bd, "static_offset") &&
            trace.limits[ai] == integer(bd, "static_len")) {
          if (logical != SIZE_MAX)
            bad = true;
          logical = ai;
        }
      if (logical == SIZE_MAX || field<DenseI64ArrayAttr>(bd, "static_sizes") ||
          field<DenseI64ArrayAttr>(bd, "static_strides")) {
        bad = true;
        return;
      }
      for (auto *use : {take, give}) {
        checkedDMALocks.insert(use);
        auto c = use->getOperand(1).getDefiningOp<arith::ConstantOp>();
        auto amount = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr{};
        auto *lock = use->getOperand(0).getDefiningOp();
        if (!amount || amount.getInt() != 1 || !named(lock, "aie.lock") ||
            lock->getNumOperands() != 1 ||
            lock->getOperand(0) != mem->getOperand(0))
          bad = true;
      }
      bool input = direction == 0;
      auto *free = take->getOperand(0).getDefiningOp(),
           *ready = give->getOperand(0).getDefiningOp();
      if (integer(free, "init") != (input ? 1 : 0) ||
          integer(ready, "init") != (input ? 0 : 1))
        bad = true;
      int64_t id;
      if (!packet(bd, id)) {
        bad = true;
        return;
      }
      sequence.push_back(descriptors.size());
      descriptors.push_back({{buffer, take->getOperand(0), give->getOperand(0),
                              logical, input, channel},
                             id});
      block = tail->getSuccessor(0);
    } while (block != first);
    checkedStarts.insert(start);
  });
  // Each channel-start must actually be reachable from the DMA engine entry;
  // each core BD must belong to one of the checked cycles, not dead CFG text.
  device->walk([&](Operation *mem) {
    if (!named(mem, "aie.mem"))
      return;
    if (mem->getNumRegions() != 1 || mem->getRegion(0).empty()) {
      bad = true;
      return;
    }
    Block *b = &mem->getRegion(0).front();
    std::set<Block *> visited;
    size_t starts = 0;
    while (visited.insert(b).second) {
      auto *tail = b->getTerminator();
      if (named(tail, "aie.end"))
        break;
      if (!named(tail, "aie.dma_start") || !checkedStarts.count(tail) ||
          tail->getNumSuccessors() != 2) {
        bad = true;
        break;
      }
      ++starts;
      b = tail->getSuccessor(1);
    }
    size_t all = 0;
    mem->walk([&](Operation *o) {
      if (named(o, "aie.dma_start"))
        ++all;
      if (named(o, "aie.dma_bd") && !checkedBDs.count(o))
        bad = true;
    });
    if (starts != all)
      bad = true;
  });
  device->walk([&](Operation *o) {
    if (!named(o, "aie.use_lock"))
      return;
    bool inCore = false;
    for (auto *p = o; p; p = p->getParentOp())
      if (named(p, "aie.core"))
        inCore = true;
    if (!inCore && !checkedDMALocks.count(o))
      bad = true;
  });
  if (bad)
    return reject(
        "actual dense DMA slice/cycle/start/lock protocol is invalid");
  struct PacketRoute {
    Value source, target;
    int64_t sourceChannel = -1, targetChannel = -1;
  };
  std::map<int64_t, PacketRoute> packetRoutes;
  device->walk([&](Operation *o) {
    if (!named(o, "aie.packet_flow"))
      return;
    auto id = integer(o, "ID");
    if (id < 0 || id > 31 || o->getNumRegions() != 1 ||
        !llvm::hasSingleElement(o->getRegion(0))) {
      bad = true;
      return;
    }
    Operation *src = nullptr, *dst = nullptr;
    for (auto &item : o->getRegion(0).front()) {
      if (named(&item, "aie.packet_source")) {
        if (src)
          bad = true;
        src = &item;
      } else if (named(&item, "aie.packet_dest")) {
        if (dst)
          bad = true;
        dst = &item;
      } else if (!named(&item, "aie.end"))
        bad = true;
    }
    if (!src || !dst || src->getNumOperands() != 1 ||
        dst->getNumOperands() != 1 || integer(src, "bundle") != 1 ||
        integer(dst, "bundle") != 1 ||
        !packetRoutes
             .emplace(id, PacketRoute{src->getOperand(0), dst->getOperand(0),
                                      integer(src, "channel"),
                                      integer(dst, "channel")})
             .second)
      bad = true;
  });
  struct Shim {
    Value tile;
    bool input = false;
    int64_t channel = -1;
  };
  std::map<std::string, Shim> shims;
  device->walk([&](Operation *o) {
    if (!named(o, "aie.shim_dma_allocation"))
      return;
    auto symbol = field<StringAttr>(o, "sym_name");
    auto direction = integer(o, "channel_dir"),
         channel = integer(o, "channel_index");
    if (!symbol || o->getNumOperands() != 1 || direction < 0 || direction > 1 ||
        channel < 0 || channel > 1) {
      bad = true;
      return;
    }
    auto xy = coordinate(o->getOperand(0));
    auto plio = field<BoolAttr>(o, "plio");
    if (xy.second != 0 || xy.first < 0 || xy.first > 3 ||
        (plio && plio.getValue()) ||
        !shims
             .emplace(symbol.str(),
                      Shim{o->getOperand(0), direction == 1, channel})
             .second)
      bad = true;
  });
  // Locate candidates from actual physical routes. Packet IDs on host BDs and
  // core BDs, rather than allocation symbol names, select the multiplexed path.
  std::set<Operation *> usedFlows;
  std::set<int64_t> usedPackets;
  auto candidates = [&](const Shim &shim, int64_t hostPacket,
                        std::vector<size_t> &out) {
    for (auto &[channel, sequence] : channels) {
      auto xy = std::get<0>(channel);
      bool input = std::get<1>(channel);
      if (input != shim.input)
        continue;
      Value coreTile = cores.at(xy)->getOperand(0);
      for (auto di : sequence) {
        auto &d = descriptors[di];
        if (d.packet >= 0) {
          auto r = packetRoutes.find(d.packet);
          if (r == packetRoutes.end())
            continue;
          auto &route = r->second;
          if ((input ? route.source : route.target) != shim.tile ||
              (input ? route.target : route.source) != coreTile ||
              (input ? route.sourceChannel : route.targetChannel) !=
                  shim.channel ||
              (input ? route.targetChannel : route.sourceChannel) !=
                  std::get<2>(channel))
            continue;
          if (input ? hostPacket != d.packet : hostPacket != -1)
            continue;
          usedPackets.insert(d.packet);
          out.push_back(di);
        } else if (hostPacket == -1) {
          size_t matches = 0;
          device->walk([&](Operation *flow) {
            if (!named(flow, "aie.flow") || flow->getNumOperands() != 2)
              return;
            if (flow->getOperand(input ? 0 : 1) == shim.tile &&
                flow->getOperand(input ? 1 : 0) == coreTile &&
                integer(flow, input ? "source_channel" : "dest_channel") ==
                    shim.channel &&
                integer(flow, input ? "dest_channel" : "source_channel") ==
                    std::get<2>(channel) &&
                integer(flow, "source_bundle") == 1 &&
                integer(flow, "dest_bundle") == 1) {
              ++matches;
              usedFlows.insert(flow);
            }
          });
          if (matches == 1)
            out.push_back(di);
          else if (matches > 1)
            bad = true;
        }
      }
    }
  };
  if (bad)
    return reject("invalid/ambiguous actual packet or shim topology");
  std::multiset<Rectangle> requested, observed;
  std::map<std::pair<size_t, bool>, std::vector<Rectangle>> requestedOrder,
      observedOrder;
  std::map<size_t, std::pair<size_t, bool>> eventStream;
  size_t selectedElements = 0, actualElements = 0;
  for (auto &ev : g.events)
    if (ev.kind == "transfer") {
      if (!ev.transferRegion)
        return reject("selected transfer has no exact rectangle");
      const auto &r = *ev.transferRegion;
      bool input = false;
      size_t local = SIZE_MAX;
      for (auto ci : ev.reads) {
        size_t ai = g.copies[ci].allocation;
        if (logicalBuffers.count(ai)) {
          if (local != SIZE_MAX && local != ai)
            return reject("transfer spans local slices");
          local = ai;
        } else
          input = true;
      }
      for (auto ci : ev.writes) {
        size_t ai = g.copies[ci].allocation;
        if (logicalBuffers.count(ai)) {
          if (local != SIZE_MAX && local != ai)
            return reject("transfer spans local slices");
          local = ai;
        }
      }
      if (local == SIZE_MAX)
        return reject("selected transfer has no actual local slice");
      std::vector<int64_t> strides(g.source.storages[r.storage].shape.size(),
                                   1);
      for (int64_t d = int64_t(strides.size()) - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * g.source.storages[r.storage].shape[d + 1];
      int64_t offset = 0;
      for (size_t d = 0; d < strides.size(); ++d)
        offset += r.offsets[d] * strides[d];
      Rectangle rect{local, r.storage, input, {}};
      if (!expandAddresses(r.sizes, strides, offset, rect.addresses) ||
          rect.addresses.size() > traceLimit - selectedElements)
        return reject(
            "selected DMA address proof exceeds one million elements");
      selectedElements += rect.addresses.size();
      eventStream[&ev - g.events.data()] = {local, input};
      requestedOrder[{local, input}].push_back(rect);
      requested.insert(std::move(rect));
    }
  struct HostTask {
    Shim shim;
    int64_t packet = -1;
    size_t storage = 0;
    std::vector<int64_t> addresses;
    bool issued = false;
  };
  std::map<Value, HostTask, VLess> tasks;
  std::set<Value, VLess> started, awaited, freed;
  std::map<Channel, size_t> channelCursor;
  std::vector<size_t> observedDescriptors;
  std::map<Channel, std::vector<Value>> channelTasks;
  std::vector<std::pair<Value, bool>> hostActions;
  size_t position = 0, lastStart = 0, firstAwait = SIZE_MAX,
         lastOutputAwait = 0;
  std::vector<std::pair<Value, size_t>> frees;
  runtime->walk<WalkOrder::PreOrder>([&](Operation *o) {
    ++position;
    if (named(o, "aiex.dma_configure_task_for")) {
      auto ref = field<FlatSymbolRefAttr>(o, "alloc");
      if (!ref || !shims.count(ref.getValue().str()) ||
          o->getNumResults() != 1 || o->getNumRegions() != 1 ||
          !llvm::hasSingleElement(o->getRegion(0)) ||
          integer(o, "repeat_count", 0) != 0) {
        bad = true;
        return;
      }
      auto &block = o->getRegion(0).front();
      Operation *bd = nullptr;
      for (auto &item : block) {
        if (named(&item, "aie.dma_bd")) {
          if (bd)
            bad = true;
          bd = &item;
        } else if (!named(&item, "aie.end"))
          bad = true;
      }
      if (!bd || bd->getNumOperands() != 1) {
        bad = true;
        return;
      }
      auto arg = dyn_cast<BlockArgument>(bd->getOperand(0));
      if (!arg || arg.getOwner() != &runtimeBlock ||
          arg.getArgNumber() >= abiStorage.size()) {
        bad = true;
        return;
      }
      HostTask task;
      task.shim = shims.at(ref.getValue().str());
      task.storage = abiStorage[arg.getArgNumber()];
      auto issued = field<BoolAttr>(o, "issue_token");
      task.issued = issued && issued.getValue();
      auto sizes = field<DenseI64ArrayAttr>(bd, "static_sizes"),
           strides = field<DenseI64ArrayAttr>(bd, "static_strides");
      int64_t len = integer(bd, "static_len");
      bool ok =
          sizes && strides
              ? expandAddresses(sizes.asArrayRef(), strides.asArrayRef(),
                                integer(bd, "static_offset"), task.addresses)
              : expandAddresses({len}, {1}, integer(bd, "static_offset"),
                                task.addresses);
      if (!ok || len != int64_t(task.addresses.size()) ||
          task.addresses.size() > traceLimit - actualElements ||
          !packet(bd, task.packet)) {
        bad = true;
        return;
      }
      actualElements += task.addresses.size();
      auto type = cast<MemRefType>(arg.getType());
      for (auto a : task.addresses)
        if (a >= type.getNumElements())
          bad = true;
      tasks.emplace(o->getResult(0), std::move(task));
    }
    if (named(o, "aiex.dma_start_task")) {
      lastStart = position;
      if (o->getNumOperands() != 1 || !tasks.count(o->getOperand(0)) ||
          !started.insert(o->getOperand(0)).second) {
        bad = true;
        return;
      }
      auto &task = tasks.at(o->getOperand(0));
      std::vector<size_t> possible;
      candidates(task.shim, task.packet, possible);
      size_t selected = SIZE_MAX;
      for (auto di : possible) {
        auto &s = descriptors[di].stream;
        Channel channel{bufferOwners.at(s.buffer), s.input, s.channel};
        auto &seq = channels.at(channel);
        if (seq[channelCursor[channel] % seq.size()] == di) {
          if (selected != SIZE_MAX) {
            bad = true;
            return;
          }
          selected = di;
        }
      }
      if (selected == SIZE_MAX) {
        bad = true;
        return;
      }
      auto &s = descriptors[selected].stream;
      Channel channel{bufferOwners.at(s.buffer), s.input, s.channel};
      ++channelCursor[channel];
      observedDescriptors.push_back(selected);
      channelTasks[channel].push_back(o->getOperand(0));
      hostActions.push_back({o->getOperand(0), true});
      auto &trace = traces.at(bufferOwners.at(s.buffer));
      if (int64_t(task.addresses.size()) != trace.limits.at(s.allocation)) {
        bad = true;
        return;
      }
      Rectangle r{s.allocation, task.storage, s.input, task.addresses};
      observedOrder[{r.allocation, r.input}].push_back(r);
      observed.insert(std::move(r));
    }
    if (named(o, "aiex.dma_await_task")) {
      firstAwait = std::min(firstAwait, position);
      if (o->getNumOperands() != 1 || !started.count(o->getOperand(0)) ||
          !awaited.insert(o->getOperand(0)).second ||
          !tasks.at(o->getOperand(0)).issued) {
        bad = true;
        return;
      }
      hostActions.push_back({o->getOperand(0), false});
      if (!tasks.at(o->getOperand(0)).shim.input)
        lastOutputAwait = position;
    }
    if (named(o, "aiex.dma_free_task")) {
      if (o->getNumOperands() != 1 || !started.count(o->getOperand(0)) ||
          !freed.insert(o->getOperand(0)).second) {
        bad = true;
        return;
      }
      frees.push_back({o->getOperand(0), position});
    }
  });
  device->walk([&](Operation *o) {
    if (named(o, "aie.flow") && !usedFlows.count(o))
      bad = true;
  });
  if (usedPackets.size() != packetRoutes.size())
    bad = true;
  if (bad || firstAwait == SIZE_MAX || requested != observed ||
      requestedOrder != observedOrder)
    return reject("runtime BO addresses, stream cycle order or packet route "
                  "differ from selected copies");
  for (auto &[token, t] : tasks)
    if (!started.count(token) || (!t.shim.input && !awaited.count(token)))
      return reject("DMA task not started or selected output not awaited");
  for (auto &[token, p] : frees)
    if (!awaited.count(token) && p <= lastOutputAwait)
      return reject("input task freed before complete outputs");
  for (auto &[c, cursor] : channelCursor)
    if (cursor % channels.at(c).size())
      return reject("host invocation leaves a partial core DMA cycle");
  for (auto &[c, seq] : channels)
    if (!channelCursor[c])
      return reject("actual extra unused DMA channel");
  // Three invocations expose the boundary races of a continuously running
  // core. Per-actor vector clocks retain semaphore causality independently of
  // which enabled actor the progress witness happens to execute first.
  struct Action {
    Value lock;
    int64_t action = 0;
    size_t access = SIZE_MAX, descriptor = SIZE_MAX;
    int frame = 0;
    Value task;
  };
  struct Actor {
    std::vector<Action> actions;
    Coordinate core{-1, -1};
  };
  using Clock = std::vector<int64_t>;
  std::vector<Actor> actors;
  std::map<std::pair<size_t, bool>, size_t> dmaActor;
  for (auto &[xy, t] : traces) {
    Actor a;
    a.core = xy;
    for (int frame = 0; frame < 3; ++frame) {
      size_t next = 0;
      for (size_t p = 0; p <= t.accesses.size(); ++p) {
        while (next < t.locks.size() && t.locks[next].position == p) {
          auto &u = t.locks[next++];
          a.actions.push_back({u.lock, u.action, SIZE_MAX, SIZE_MAX, frame});
        }
        if (p < t.accesses.size())
          a.actions.push_back({{}, 0, p, SIZE_MAX, frame});
      }
      if (next != t.locks.size())
        return reject("invalid actual core lock position");
    }
    actors.push_back(std::move(a));
  }
  for (auto &[channel, sequence] : channels) {
    Actor a;
    size_t actor = actors.size();
    for (int frame = 0; frame < 3; ++frame)
      for (size_t n = 0; n < channelCursor[channel]; ++n) {
        size_t di = sequence[n % sequence.size()];
        auto &s = descriptors[di].stream;
        auto key = std::make_pair(s.allocation, s.input);
        auto old = dmaActor.find(key);
        if (old != dmaActor.end() && old->second != actor)
          return reject("logical slice DMA has multiple independent actors");
        dmaActor[key] = actor;
        a.actions.push_back({s.free, 2, SIZE_MAX, SIZE_MAX, frame});
        a.actions.push_back(
            {{}, 0, SIZE_MAX, di, frame, channelTasks.at(channel)[n]});
        a.actions.push_back({s.ready, 1, SIZE_MAX, SIZE_MAX, frame});
      }
    actors.push_back(std::move(a));
  }
  Actor host;
  for (int frame = 0; frame < 3; ++frame)
    for (auto [task, start] : hostActions)
      host.actions.push_back(
          {{}, start ? 3 : 4, SIZE_MAX, SIZE_MAX, frame, task});
  actors.push_back(std::move(host));
  size_t actions = 0, records = 0;
  for (auto &[c, n] : channelCursor)
    records += 3 * n;
  for (auto &[xy, t] : traces)
    records += 3 * t.accesses.size();
  if (actors.size() > 16 ||
      records > 8000000 / std::max<size_t>(1, actors.size()))
    return reject("causal proof exceeds eight million clock cells");
  struct Token {
    bool available = false;
    Clock clock;
  };
  std::map<Value, Token, VLess> tokens;
  std::map<Value, std::set<size_t>, VLess> takers, givers;
  for (size_t a = 0; a < actors.size(); ++a)
    for (auto &u : actors[a].actions) {
      if (++actions > 3 * traceLimit)
        return reject("causal progress proof exceeds three million actions");
      if (!u.lock)
        continue;
      auto init = integer(u.lock.getDefiningOp(), "init");
      if (init < 0 || init > 1)
        return reject("invalid initial semaphore token");
      tokens[u.lock] = {bool(init), Clock(actors.size(), 0)};
      (u.action == 2 ? takers : givers)[u.lock].insert(a);
    }
  for (auto &[lock, who] : takers)
    if (who.size() != 1 || givers[lock].size() != 1)
      return reject("semaphore has ambiguous actor ownership");
  for (auto &[xy, t] : traces)
    for (auto &u : t.locks) {
      auto *lock = u.lock.getDefiningOp();
      if (!tokens.count(u.lock) || !lock || lock->getNumOperands() != 1 ||
          coordinate(lock->getOperand(0)) != xy)
        return reject("unaccounted/cross-owner core lock");
    }
  std::map<Coordinate, std::vector<std::vector<Clock>>> memoryClocks;
  for (auto &[xy, t] : traces)
    memoryClocks[xy] = std::vector<std::vector<Clock>>(
        3, std::vector<Clock>(t.accesses.size()));
  struct TaskKeyLess {
    bool operator()(const std::pair<Value, int> &a,
                    const std::pair<Value, int> &b) const {
      VLess less;
      if (a.first != b.first)
        return less(a.first, b.first);
      return a.second < b.second;
    }
  };
  std::map<std::pair<Value, int>, Clock, TaskKeyLess> hostStarted,
      hostCompleted;
  std::map<std::pair<size_t, bool>, std::vector<Clock>> dmaClocks;
  std::vector<Clock> clocks(actors.size(), Clock(actors.size(), 0));
  std::vector<size_t> cursor(actors.size(), 0);
  size_t completed = 0;
  while (completed < actions) {
    bool progress = false;
    for (size_t a = 0; a < actors.size(); ++a) {
      if (cursor[a] == actors[a].actions.size())
        continue;
      auto &u = actors[a].actions[cursor[a]];
      if (u.lock && u.action == 2 && !tokens[u.lock].available)
        continue;
      auto task = std::make_pair(u.task, u.frame);
      if (u.action == 4 && !hostCompleted.count(task))
        continue;
      if (u.descriptor != SIZE_MAX && !hostStarted.count(task))
        continue;
      auto &clock = clocks[a];
      if (u.action == 4) {
        auto &dependency = hostCompleted.at(task);
        for (size_t k = 0; k < clock.size(); ++k)
          clock[k] = std::max(clock[k], dependency[k]);
      }
      if (u.descriptor != SIZE_MAX) {
        auto &dependency = hostStarted.at(task);
        for (size_t k = 0; k < clock.size(); ++k)
          clock[k] = std::max(clock[k], dependency[k]);
      }
      if (u.lock && u.action == 2) {
        auto &token = tokens[u.lock];
        for (size_t k = 0; k < clock.size(); ++k)
          clock[k] = std::max(clock[k], token.clock[k]);
        token.available = false;
      }
      ++clock[a];
      if (u.lock && u.action == 1) {
        auto &token = tokens[u.lock];
        if (token.available)
          return reject("ready/free semaphore accumulates duplicate tokens");
        token = {true, clock};
      }
      if (u.action == 3)
        hostStarted[task] = clock;
      if (u.descriptor != SIZE_MAX)
        hostCompleted[task] = clock;
      if (u.access != SIZE_MAX)
        memoryClocks.at(actors[a].core)[u.frame][u.access] = clock;
      if (u.descriptor != SIZE_MAX) {
        auto &s = descriptors[u.descriptor].stream;
        dmaClocks[{s.allocation, s.input}].push_back(clock);
      }
      ++cursor[a];
      ++completed;
      progress = true;
    }
    if (!progress)
      return reject("actual core/DMA protocol deadlocks before three complete "
                    "invocations");
  }
  auto before = [](const Clock &a, const Clock &b) {
    if (a.empty() || a.size() != b.size())
      return false;
    bool strict = false;
    for (size_t k = 0; k < a.size(); ++k) {
      if (a[k] > b[k])
        return false;
      strict |= a[k] < b[k];
    }
    return strict;
  };
  for (auto &[key, rectangles] : requestedOrder) {
    size_t ai = key.first;
    bool input = key.second;
    auto xy = bufferOwners.at(logicalBuffers.at(ai));
    auto &t = traces.at(xy);
    std::vector<size_t> boundaries;
    for (auto [event, p] : t.transferPositions)
      if (eventStream.count(event) && eventStream.at(event) == key)
        boundaries.push_back(p);
    size_t count = rectangles.size();
    auto found = dmaClocks.find(key);
    if (boundaries.size() != count || found == dmaClocks.end() ||
        found->second.size() != 3 * count)
      return reject(
          "logical transfer epoch missing from actual causal witness");
    auto &copies = found->second;
    for (size_t p = 0; p < t.accesses.size(); ++p) {
      auto &access = t.accesses[p];
      if (access.allocation != ai || (!input && !access.write))
        continue;
      auto &use = memoryClocks.at(xy)[1][p];
      size_t n = std::upper_bound(boundaries.begin(), boundaries.end(), p) -
                 boundaries.begin();
      if (input) {
        if (!n)
          return reject(
              "source input access precedes selected initial DMA event");
        size_t current = count + n - 1;
        if (!before(copies[current], use) || !before(use, copies[current + 1]))
          return reject("input DMA write/read/refill causality fails for "
                        "logical allocation " +
                        std::to_string(ai) + " access " + std::to_string(p));
      } else {
        size_t next = count + n;
        if (!before(copies[next - 1], use) || !before(use, copies[next]))
          return reject("output DMA previous-read/write/export causality fails "
                        "for logical allocation " +
                        std::to_string(ai) + " access " + std::to_string(p));
      }
    }
  }
  receipt = j::Object{
      {"schema", "orbit.public-program-graph.aie.v27"},
      {"status", "passed"},
      {"graph_id", programGraphIdentity(g)},
      {"source_backed_air_reconstructed", true},
      {"scalar_memory_operations", int64_t(scalarAccesses)},
      {"host_dma_rectangles", int64_t(observed.size())},
      {"physical_configurations", 1},
      {"causal_progress_actions", int64_t(actions)},
      {"cores", std::move(coreReceipts)},
      {"analytical_group_setups_realized", false},
      {"timed_dispatch_verified", false},
      {"scope",
       "Bounded exact scalar memory/expression traces; actual SSA allocation "
       "ownership, routed host/core dense cyclic DMA, rectangular host "
       "addresses and unit ready/free locks. Three-invocation semaphore "
       "vector-clock causality includes packed-slice prefill/drain and refill "
       "hazards. No device timing or analytical setup equivalence."}};
  return true;
}
