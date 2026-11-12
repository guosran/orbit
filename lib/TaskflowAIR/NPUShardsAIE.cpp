#include "Taskflow/AIRBackend.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/NPUIndependentShards.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <map>
#include <set>
using namespace mlir;
using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
bool named(Operation *o, StringRef n) {
  return o->getName().getStringRef() == n;
}
template <class T> T field(Operation *o, StringRef n) {
  if (auto a = o->getAttrOfType<T>(n))
    return a;
  if (auto p = dyn_cast_or_null<DictionaryAttr>(o->getPropertiesAsAttribute()))
    return p.getAs<T>(n);
  return {};
}
int64_t number(Operation *o, StringRef n, int64_t fallback = -1) {
  auto a = field<IntegerAttr>(o, n);
  return a ? a.getInt() : fallback;
}
bool setField(Operation *o, StringRef n, Attribute a) {
  auto p = dyn_cast_or_null<DictionaryAttr>(o->getPropertiesAsAttribute());
  if (p && p.get(n)) {
    NamedAttrList fields(p);
    fields.set(n, a);
    return succeeded(o->setPropertiesFromAttribute(
        fields.getDictionary(o->getContext()),
        [&]() { return o->emitError("invalid normalized AIE field"); }));
  }
  o->setAttr(n, a);
  return true;
}
bool reject(std::string &e, StringRef s) {
  e = "multicore AIE contract: " + s.str();
  return false;
}
Operation *findDevice(ModuleOp m, StringRef symbol) {
  Operation *answer = nullptr;
  m.walk([&](Operation *o) {
    if (named(o, "aie.device")) {
      auto s = field<StringAttr>(o, "sym_name");
      if (s && s.getValue() == symbol)
        answer = o;
    }
  });
  return answer;
}
} // namespace
bool orbit::taskflow::checkNPUIndependentAIE(ModuleOp source,
                                             const j::Value &config,
                                             int64_t requested, ModuleOp actual,
                                             j::Value &receipt,
                                             std::string &e) {
  receipt =
      j::Object{{"valid", false}, {"schema", "orbit.npu-independent-aie.v22"}};
  if (failed(verify(actual)))
    return reject(e, "invalid actual SSA/IR");
  OwningOpRef<ModuleOp> expectedAir;
  j::Value manifest = nullptr;
  if (!lowerNPUIndependentProgram(source, config, requested, expectedAir,
                                  manifest, e))
    return false;
  auto &m = *manifest.getAsObject();
  auto &shards = *m.getArray("shards");
  auto &placements = *m.getArray("placement");
  size_t count = shards.size();
  GenericAnalysis original;
  if (!analyzeGenericProgram(source, original, e))
    return false;
  // Bound exact output coverage and address arithmetic before multiplication.
  for (const auto &storage : original.storages)
    if (isa<BlockArgument>(storage.root)) {
      int64_t elements = 1;
      for (auto n : storage.shape) {
        if (n <= 0 || n > 1000000 / elements)
          return reject(
              e, "host ABI exceeds certified bounded static element domain");
        elements *= n;
      }
    }
  std::vector<size_t> abiStorages;
  for (size_t i = 0; i < original.storages.size(); ++i)
    if (isa<BlockArgument>(original.storages[i].root))
      abiStorages.push_back(i);
  std::map<std::pair<int64_t, int64_t>, size_t> coordinateOwners;
  for (auto &v : placements) {
    const auto &p = *v.getAsObject();
    coordinateOwners[{*p.getInteger("column"), *p.getInteger("row")}] =
        *p.getInteger("shard");
  }
  if (coordinateOwners.size() != count)
    return reject(e, "duplicate expected physical tile");
  Operation *configure = nullptr, *run = nullptr;
  size_t configurations = 0, runs = 0, devices = 0;
  actual.walk([&](Operation *o) {
    if (named(o, "aiex.configure")) {
      configure = o;
      ++configurations;
    }
    if (named(o, "aiex.run")) {
      run = o;
      ++runs;
    }
    if (named(o, "aie.device"))
      ++devices;
  });
  if (configurations != 1 || runs != 1 || devices != 2 ||
      run->getParentOp() != configure)
    return reject(e, "exactly one configuration and one complete run required");
  auto ref = field<FlatSymbolRefAttr>(configure, "symbol");
  auto sequence = field<FlatSymbolRefAttr>(run, "runtime_sequence_symbol");
  if (!ref || !sequence)
    return reject(e, "explicit device/runtime references required");
  auto *device = findDevice(actual, ref.getValue());
  // Installed mlir-aie AIEDevice::npu2 encodes as 8. This checker does not
  // silently reuse its four-column resource model for a different device.
  if (device && number(device, "device") != 8)
    return reject(e,
                  "independent shard resource model requires the npu2 target");
  if (!device || device->getNumRegions() != 1 ||
      !llvm::hasSingleElement(device->getRegion(0)))
    return reject(e, "missing concrete configured device");
  for (auto &top : actual.getBody()->getOperations())
    if (!named(&top, "aie.device"))
      return reject(e, "extra module execution entry");
  auto *wrapper = configure->getParentOp();
  if (!wrapper || !named(wrapper, "aie.runtime_sequence") ||
      wrapper->getNumRegions() != 1 ||
      !llvm::hasSingleElement(wrapper->getRegion(0)))
    return reject(e, "invalid host wrapper");
  auto *wrapperDevice = wrapper->getParentOp();
  if (!wrapperDevice || wrapperDevice == device ||
      !named(wrapperDevice, "aie.device") ||
      wrapperDevice->getNumRegions() != 1 ||
      !llvm::hasSingleElement(wrapperDevice->getRegion(0)))
    return reject(e, "invalid wrapper device");
  if (number(wrapperDevice, "device") != 8)
    return reject(e, "wrapper target differs from npu2");
  for (auto &o : wrapperDevice->getRegion(0).front())
    if (&o != wrapper && !named(&o, "aie.end"))
      return reject(e, "extra wrapper device side effect");
  for (auto &o : wrapper->getRegion(0).front())
    if (&o != configure)
      return reject(e, "extra host wrapper side effect");
  if (configure->getNumRegions() != 1 ||
      !llvm::hasSingleElement(configure->getRegion(0)))
    return reject(e, "malformed configure body");
  for (auto &o : configure->getRegion(0).front())
    if (&o != run)
      return reject(e, "extra configure side effect");
  llvm::DenseMap<Value, int64_t> tileOwners;
  std::set<size_t> observedCores;
  for (auto &o : device->getRegion(0).front())
    if (named(&o, "aie.core")) {
      if (o.getNumOperands() != 1)
        return reject(e, "malformed core");
      auto *tile = o.getOperand(0).getDefiningOp();
      if (!tile || !named(tile, "aie.tile"))
        return reject(e, "unbound core tile");
      auto found =
          coordinateOwners.find({number(tile, "col"), number(tile, "row")});
      if (found == coordinateOwners.end() ||
          !observedCores.insert(found->second).second)
        return reject(e,
                      "actual placement differs from selected physical tiles");
      tileOwners[o.getOperand(0)] = found->second;
    }
  if (observedCores.size() != count)
    return reject(e, "missing or duplicate physical cores");
  for (auto &o : device->getRegion(0).front())
    if (named(&o, "aie.tile") && number(&o, "row") == 0) {
      int64_t owner = -1;
      for (auto &entry : coordinateOwners)
        if (entry.first.first == number(&o, "col")) {
          if (owner >= 0)
            return reject(e,
                          "shared shim columns are outside independent subset");
          owner = entry.second;
        }
      if (owner < 0)
        return reject(e, "unbound extra shim tile");
      if (o.getNumResults() != 1)
        return reject(e, "malformed shim tile");
      tileOwners[o.getResult(0)] = owner;
    }
  std::vector<int64_t> topOwners;
  std::map<std::string, int64_t> streamOwners;
  std::set<std::tuple<int64_t, int64_t, int64_t>> channels;
  Operation *runtime = nullptr;
  for (auto &o : device->getRegion(0).front()) {
    int64_t owner = -1;
    if (named(&o, "aie.tile")) {
      if (o.getNumResults() != 1 || !tileOwners.count(o.getResult(0)))
        return reject(e, "unsupported memory/extra tile");
      owner = tileOwners[o.getResult(0)];
    } else if (named(&o, "aie.lock") || named(&o, "aie.buffer") ||
               named(&o, "aie.mem") || named(&o, "aie.core") ||
               named(&o, "aie.shim_dma_allocation")) {
      if (o.getNumOperands() != 1 || !tileOwners.count(o.getOperand(0)))
        return reject(e, "resource lacks a unique physical owner");
      owner = tileOwners[o.getOperand(0)];
      if (named(&o, "aie.shim_dma_allocation")) {
        auto s = field<StringAttr>(&o, "sym_name");
        auto *tile = o.getOperand(0).getDefiningOp();
        auto direction = number(&o, "channel_dir"),
             channel = number(&o, "channel_index");
        if (!s || number(tile, "row") != 0 || direction < 0 || direction > 1 ||
            channel < 0 || channel > 1 ||
            !channels.emplace(number(tile, "col"), direction, channel).second ||
            !streamOwners.emplace(s.str(), owner).second)
          return reject(e, "duplicate/invalid actual shim channel");
      }
      if (named(&o, "aie.lock") &&
          (number(&o, "lockID") < 0 || number(&o, "lockID") > 15))
        return reject(e, "unsupported physical lock index");
      if (named(&o, "aie.buffer") && field<IntegerAttr>(&o, "address"))
        return reject(
            e,
            "explicit buffer addresses require a separate non-overlap proof");
      if (named(&o, "aie.core") || named(&o, "aie.mem")) {
        bool unknown = false;
        bool core = named(&o, "aie.core");
        o.walk([&](Operation *nested) {
          if (nested == &o)
            return;
          auto n = nested->getName().getStringRef();
          bool allowed =
              core ? (n.starts_with("arith.") || n == "scf.for" ||
                      n == "scf.yield" || n == "cf.br" ||
                      n == "memref.reinterpret_cast" || n == "memref.load" ||
                      n == "memref.store" || n == "aie.use_lock")
                   : (n == "arith.constant" || n == "aie.dma_start" ||
                      n == "aie.use_lock" || n == "aie.dma_bd" ||
                      n == "aie.next_bd" || n == "aie.end");
          if (!allowed)
            unknown = true;
        });
        if (unknown)
          return reject(e, "unknown nested core/DMA side effect");
      }
    } else if (named(&o, "aie.flow")) {
      if (o.getNumOperands() != 2 || !tileOwners.count(o.getOperand(0)) ||
          !tileOwners.count(o.getOperand(1)) ||
          tileOwners[o.getOperand(0)] != tileOwners[o.getOperand(1)])
        return reject(e, "cross-owner/unknown flow");
      auto *a = o.getOperand(0).getDefiningOp(),
           *b = o.getOperand(1).getDefiningOp();
      if (!((number(a, "row") == 0 && number(b, "row") == 2) ||
            (number(a, "row") == 2 && number(b, "row") == 0)))
        return reject(e, "only direct shim/compute flows are supported");
      owner = tileOwners[o.getOperand(0)];
    } else if (named(&o, "aie.runtime_sequence")) {
      auto runtimeName = field<StringAttr>(&o, "sym_name");
      if (runtime || !runtimeName ||
          runtimeName.getValue() != sequence.getValue())
        return reject(e, "extra/mismatched runtime sequence");
      runtime = &o;
    } else if (named(&o, "aie.external_buffer")) {
      if (o.getNumOperands() || !o.use_empty())
        return reject(
            e, "external buffer introduces an unproved cross-owner reference");
    } else if (!named(&o, "aie.end"))
      return reject(e, "unknown device operation (packet/memtile topology is "
                       "not certified)");
    topOwners.push_back(owner);
  }
  // Unregistered AIE has no SymbolTable trait here. Check the complete
  // symbolic-reference vocabulary directly before projecting away resources.
  bool unknownSymbol = false;
  auto inspectSymbols = [&](Operation *op, DictionaryAttr attrs) {
    if (!attrs)
      return;
    for (auto entry : attrs) {
      auto value = entry.getValue();
      if (isa<ArrayAttr, DictionaryAttr>(value)) {
        // None of the certified fields nests symbol references.
        std::function<bool(Attribute)> contains = [&](Attribute a) {
          if (isa<SymbolRefAttr>(a))
            return true;
          if (auto d = dyn_cast<DictionaryAttr>(a)) {
            for (auto x : d)
              if (contains(x.getValue()))
                return true;
          }
          if (auto v = dyn_cast<ArrayAttr>(a)) {
            for (auto x : v)
              if (contains(x))
                return true;
          }
          return false;
        };
        if (contains(value))
          unknownSymbol = true;
      } else if (auto symbol = dyn_cast<SymbolRefAttr>(value)) {
        auto key = entry.getName().strref();
        bool allowed = (op == configure && key == "symbol" && symbol == ref) ||
                       (op == run && key == "runtime_sequence_symbol" &&
                        symbol == sequence) ||
                       (named(op, "aiex.dma_configure_task_for") &&
                        op->getParentOp() == runtime && key == "alloc" &&
                        isa<FlatSymbolRefAttr>(symbol) &&
                        streamOwners.count(symbol.getRootReference().str()));
        if (!allowed)
          unknownSymbol = true;
      }
    }
  };
  actual.walk([&](Operation *op) {
    inspectSymbols(op, op->getAttrDictionary());
    inspectSymbols(
        op, dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute()));
  });
  if (unknownSymbol)
    return reject(e, "unknown or cross-scope symbolic resource reference");
  if (!runtime || runtime->getNumRegions() != 1 ||
      !llvm::hasSingleElement(runtime->getRegion(0)) ||
      runtime->getRegion(0).front().getNumArguments() != abiStorages.size())
    return reject(e, "invalid complete runtime ABI");
  // Prove resource/SSA independence BEFORE removing anything for projection.
  llvm::DenseMap<Value, int64_t> ownedValues;
  size_t ordinal = 0;
  for (auto &o : device->getRegion(0).front()) {
    auto owner = topOwners[ordinal++];
    if (owner >= 0)
      for (auto v : o.getResults())
        ownedValues[v] = owner;
  }
  ordinal = 0;
  for (auto &top : device->getRegion(0).front()) {
    auto owner = topOwners[ordinal++];
    if (owner < 0)
      continue;
    bool bad = false;
    top.walk([&](Operation *o) {
      for (auto v : o->getOperands()) {
        if (auto a = dyn_cast<BlockArgument>(v)) {
          if (a.getOwner()->getParentOp() == &top ||
              top.isProperAncestor(a.getOwner()->getParentOp()))
            continue;
          bad = true;
          continue;
        }
        auto *definition = v.getDefiningOp();
        if (definition == &top || top.isProperAncestor(definition))
          continue;
        auto found = ownedValues.find(v);
        if (found == ownedValues.end() || found->second != owner)
          bad = true;
      }
    });
    if (bad)
      return reject(e, "cross-owner SSA/resource reference");
  }
  llvm::DenseMap<Value, int64_t> tokens;
  llvm::DenseMap<Value, std::vector<size_t>> starts, awaits, frees;
  llvm::DenseMap<Value, bool> inputTasks;
  std::vector<int64_t> runtimeOwners;
  std::vector<int64_t> bdCounts(count, 0);
  std::vector<std::map<int64_t, int64_t>> writes(abiStorages.size());
  size_t position = 0, lastStart = 0, firstAwait = SIZE_MAX, lastAwait = 0;
  for (auto &o : runtime->getRegion(0).front()) {
    ++position;
    int64_t owner = -1;
    if (named(&o, "aiex.dma_configure_task_for")) {
      auto alloc = field<FlatSymbolRefAttr>(&o, "alloc");
      if (!alloc || !streamOwners.count(alloc.getValue().str()) ||
          o.getNumResults() != 1 || o.getNumRegions() != 1 ||
          !llvm::hasSingleElement(o.getRegion(0)))
        return reject(e, "unbound DMA task");
      owner = streamOwners[alloc.getValue().str()];
      tokens[o.getResult(0)] = owner;
      size_t descriptors = 0;
      for (auto &bd : o.getRegion(0).front()) {
        if (named(&bd, "aie.end"))
          continue;
        if (!named(&bd, "aie.dma_bd") || bd.getNumOperands() != 1)
          return reject(e, "unknown DMA task side effect/BD chain");
        ++descriptors;
        ++bdCounts[owner];
        auto arg = dyn_cast<BlockArgument>(bd.getOperand(0));
        if (!arg || arg.getOwner() != &runtime->getRegion(0).front() ||
            arg.getArgNumber() >= abiStorages.size())
          return reject(e, "DMA BD does not reference caller buffer");
        auto ai = arg.getArgNumber();
        auto type = dyn_cast<MemRefType>(arg.getType());
        auto expected =
            cast<MemRefType>(original.storages[abiStorages[ai]].root.getType());
        if (type != expected)
          return reject(e, "actual host ABI differs from complete source");
        int64_t rowStride = 1;
        for (auto n : expected.getShape().drop_front())
          rowStride *= n;
        auto &shard = *shards[owner].getAsObject();
        int64_t lower = *shard.getInteger("row_offset") * rowStride,
                upper = lower + *shard.getInteger("row_count") * rowStride;
        auto offset = number(&bd, "static_offset"),
             length = number(&bd, "static_len");
        auto sizes = field<DenseI64ArrayAttr>(&bd, "static_sizes"),
             strides = field<DenseI64ArrayAttr>(&bd, "static_strides");
        bool contiguous =
            bool(sizes) && bool(strides) && sizes.size() == strides.size();
        int64_t elements = 1;
        if (contiguous)
          for (int64_t d = sizes.size() - 1; d >= 0; --d) {
            if (sizes[d] < 1 || sizes[d] > 1000000 / elements ||
                (sizes[d] > 1 && strides[d] != elements)) {
              contiguous = false;
              break;
            }
            elements *= sizes[d];
          }
        if (offset < lower || length <= 0 || !contiguous ||
            elements != length || length > 1000000 || offset > upper - length)
          return reject(
              e, "actual host BD is outside its exact contiguous shard region");
        bool input = original.storages[abiStorages[ai]].role == "input";
        if (!input && original.storages[abiStorages[ai]].role != "output")
          return reject(e,
                        "unexpected host temporary in independent shard ABI");
        inputTasks[o.getResult(0)] = input;
        if (original.storages[abiStorages[ai]].role == "output")
          for (int64_t k = offset; k < offset + length; ++k)
            if (++writes[ai][k] != 1)
              return reject(e, "duplicate host output write");
      }
      if (descriptors != 1)
        return reject(e,
                      "one static descriptor per independent stream required");
    } else if (named(&o, "aiex.dma_start_task") ||
               named(&o, "aiex.dma_await_task") ||
               named(&o, "aiex.dma_free_task")) {
      if (o.getNumOperands() != 1 || !tokens.count(o.getOperand(0)))
        return reject(e, "unbound DMA control token");
      auto token = o.getOperand(0);
      owner = tokens[token];
      if (named(&o, "aiex.dma_start_task")) {
        lastStart = position;
        starts[token].push_back(position);
      }
      if (named(&o, "aiex.dma_await_task")) {
        firstAwait = std::min(firstAwait, position);
        lastAwait = position;
        awaits[token].push_back(position);
      }
      if (named(&o, "aiex.dma_free_task"))
        frees[token].push_back(position);
    } else
      return reject(e, "unknown runtime side effect");
    runtimeOwners.push_back(owner);
  }
  if (lastStart >= firstAwait)
    return reject(e, "runtime waits before starting all independent streams");
  for (auto &entry : tokens) {
    auto token = entry.first;
    if (starts[token].size() != 1)
      return reject(e, "DMA task must start exactly once");
    if (inputTasks[token]) {
      if (!awaits[token].empty() || frees[token].size() != 1 ||
          frees[token][0] <= lastAwait)
        return reject(
            e, "input task resource freed before all shard outputs completed");
    } else if (awaits[token].size() != 1 ||
               awaits[token][0] <= starts[token][0] || !frees[token].empty())
      return reject(e, "output task requires exactly one completion await");
  }
  for (size_t i = 0; i < abiStorages.size(); ++i)
    if (original.storages[abiStorages[i]].role == "output") {
      auto type =
          cast<MemRefType>(original.storages[abiStorages[i]].root.getType());
      if (writes[i].size() != size_t(type.getNumElements()))
        return reject(e, "missing source output coverage");
      for (int64_t k = 0; k < type.getNumElements(); ++k)
        if (writes[i][k] != 1)
          return reject(e, "incorrect output partition coverage");
    }
  for (auto n : bdCounts)
    if (n > 8)
      return reject(e,
                    "independent-shard static shim descriptor budget exceeded");
  j::Array checks;
  for (size_t selected = 0; selected < count; ++selected) {
    auto &record = *shards[selected].getAsObject();
    auto child = mlir::parseSourceString<ModuleOp>(
        *record.getString("source_mlir"), source.getContext());
    if (!child)
      return reject(e, "cannot reconstruct checked source shard");
    GenericAnalysis analysis;
    GenericMappingPlan plan;
    if (!analyzeGenericProgram(*child, analysis, e) ||
        !planGenericMapping(analysis, config, plan, e))
      return false;
    // Round-trip unregistered AIE operations: this MLIR version's clone drops
    // opaque properties, including the device symbol and concrete DMA fields.
    std::string serialized;
    {
      llvm::raw_string_ostream stream(serialized);
      actual.print(stream, OpPrintingFlags().printGenericOpForm());
    }
    auto projection =
        parseSourceString<ModuleOp>(serialized, source.getContext());
    if (!projection)
      return reject(e, "cannot preserve actual AIE properties");
    auto *projected = findDevice(*projection, ref.getValue());
    if (!projected)
      return reject(e, "projection lost configured device");
    auto &block = projected->getRegion(0).front();
    Operation *projectedRuntime = nullptr;
    for (auto &o : block)
      if (named(&o, "aie.runtime_sequence"))
        projectedRuntime = &o;
    if (!projectedRuntime)
      return reject(e, "projection lost runtime sequence");
    SmallVector<Operation *> erase;
    size_t i = 0;
    for (auto &o : projectedRuntime->getRegion(0).front()) {
      if (runtimeOwners[i++] != int64_t(selected))
        erase.push_back(&o);
    }
    for (auto *o : llvm::reverse(erase)) {
      if (!o->use_empty())
        return reject(e, "DMA task ownership closure is incomplete");
      o->erase();
    }
    erase.clear();
    i = 0;
    for (auto &o : block) {
      auto owner = topOwners[i++];
      if (owner >= 0 && owner != int64_t(selected))
        erase.push_back(&o);
    }
    for (auto *o : llvm::reverse(erase)) {
      if (!o->use_empty())
        return reject(e, "tile ownership closure is incomplete");
      o->erase();
    }
    bool good = true;
    OpBuilder builder(source.getContext());
    projection->walk([&](Operation *o) {
      if (named(o, "aie.tile"))
        good &= setField(o, "col", builder.getI32IntegerAttr(0));
      if (named(o, "aie.runtime_sequence")) {
        auto *b = &o->getRegion(0).front();
        if (b->getNumArguments() != analysis.function.getNumArguments()) {
          good = false;
          return;
        }
        for (size_t ai = 0; ai < b->getNumArguments(); ++ai)
          b->getArgument(ai).setType(
              analysis.function.getArgument(ai).getType());
      }
      if (named(o, "aie.external_buffer")) {
        if (o->getNumResults() != 1) {
          good = false;
          return;
        }
        auto type = dyn_cast<MemRefType>(o->getResult(0).getType());
        if (!type || type.getRank() < 1) {
          good = false;
          return;
        }
        SmallVector<int64_t> shape(type.getShape());
        shape[0] = *record.getInteger("row_count");
        o->getResult(0).setType(MemRefType::get(shape, type.getElementType(),
                                                type.getLayout(),
                                                type.getMemorySpace()));
      }
    });
    projectedRuntime->walk([&](Operation *o) {
      if (!named(o, "aie.dma_bd"))
        return;
      auto arg = dyn_cast<BlockArgument>(o->getOperand(0));
      if (!arg) {
        good = false;
        return;
      }
      auto type = cast<MemRefType>(arg.getType());
      int64_t stride = 1;
      for (auto n : type.getShape().drop_front())
        stride *= n;
      int64_t delta = *record.getInteger("row_offset") * stride;
      auto old = field<IntegerAttr>(o, "static_offset");
      if (!old || old.getInt() < delta) {
        good = false;
        return;
      }
      good &= setField(o, "static_offset",
                       IntegerAttr::get(old.getType(), old.getInt() - delta));
    });
    if (!good || failed(verify(*projection)))
      return reject(e, "normalized isolated AIE failed SSA/ABI checks");
    j::Value check = nullptr;
    if (!checkGenericAIEImplementation(analysis, plan, *projection, check, e))
      return false;
    checks.push_back(std::move(check));
  }
  receipt = j::Object{
      {"schema", "orbit.npu-independent-aie.v22"},
      {"valid", true},
      {"actual_cores", int64_t(count)},
      {"configurations", 1},
      {"runs", 1},
      {"target", "npu2"},
      {"resource_ownership_proved_before_projection", true},
      {"complete_output_coverage", true},
      {"per_core_full_source_checks", std::move(checks)},
      {"scope",
       "direct independent shim/compute stream shards; static buffers and BDs; "
       "no packet/memtile/cross-core dependencies or measured overlap claim"}};
  return true;
}
