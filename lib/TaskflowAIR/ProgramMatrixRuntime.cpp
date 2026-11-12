#include "Taskflow/ProgramMatrixRuntime.h"
#include "Taskflow/ProgramGraphAIRSelection.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <functional>
#include <map>
#include <set>

using namespace mlir;
using namespace orbit::taskflow;
namespace {
template <class T> T field(Operation *op, StringRef key) {
  if (auto attr = op->getAttrOfType<T>(key)) {
    return attr;
  }
  if (auto props =
          dyn_cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute())) {
    return props.getAs<T>(key);
  }
  return {};
}
int64_t integer(Operation *op, StringRef key) {
  auto attr = field<IntegerAttr>(op, key);
  return attr ? attr.getInt() : -1;
}
bool named(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}
bool fail(std::string &error, const char *message) {
  error = std::string("matrix runtime: ") + message;
  return false;
}
std::string moduleFingerprint(ModuleOp module) {
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  module.print(stream, OpPrintingFlags().printGenericOpForm());
  stream << '\n';
  auto digest = llvm::SHA256::hash(ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  return llvm::toHex(digest, true);
}
struct RuntimeTask {
  Operation *configure = nullptr, *start = nullptr, *await = nullptr,
            *free = nullptr;
  size_t storage = 0, event = SIZE_MAX;
  std::vector<size_t> events;
  Operation *bd = nullptr;
  Attribute packet;
  bool input = false;
  std::vector<int64_t> addresses;
};
bool addresses(ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides,
               int64_t offset, std::vector<int64_t> &out) {
  if (sizes.size() != strides.size() || offset < 0) {
    return false;
  }
  out = {offset};
  for (size_t d = 0; d < sizes.size(); ++d) {
    if (sizes[d] <= 0 || strides[d] <= 0 ||
        out.size() > 1048576 / uint64_t(sizes[d])) {
      return false;
    }
    std::vector<int64_t> expanded;
    for (auto base : out) {
      for (int64_t i = 0; i < sizes[d]; ++i) {
        if (i > (INT64_MAX - base) / strides[d]) {
          return false;
        }
        expanded.push_back(base + i * strides[d]);
      }
    }
    out = std::move(expanded);
  }
  return true;
}
bool hostInput(const ProgramGraph &g, const MappingEvent &event) {
  auto allocation = g.copies[event.reads.front()].allocation;
  auto memory = llvm::find_if(g.memories, [&](const auto &m) {
    return m.id == g.allocations[allocation].memory;
  });
  return memory != g.memories.end() && memory->kind == "system-bo";
}
struct TransferSpec {
  size_t event = 0, storage = 0;
  bool input = false;
  std::vector<int64_t> addresses;
  int64_t offset = 0, localAllocation = -1, localOffset = -1;
  std::vector<int64_t> sizes, strides;
};
struct RuntimeSpec {
  std::string graphId, sourceFingerprint, contractHash;
  int64_t selectionBegin = -1, selectionEnd = -1;
  std::vector<int64_t> abi;
  std::vector<std::string> types;
  std::vector<TransferSpec> transfers;
  std::vector<std::vector<size_t>> dependencies;
  std::vector<std::pair<size_t, size_t>> feedback;
};
int64_t packetNumber(Attribute packet) {
  if (!packet)
    return -1;
  std::string text;
  llvm::raw_string_ostream stream(text);
  packet.print(stream);
  StringRef value(text);
  auto at = value.find("pkt_id");
  if (at == StringRef::npos)
    return -1;
  value = value.drop_front(at + 6);
  at = value.find('=');
  if (at == StringRef::npos)
    return -1;
  value = value.drop_front(at + 1).trim();
  auto end = value.find_first_of(",>");
  if (end != StringRef::npos)
    value = value.take_front(end);
  int64_t result = -1;
  if (value.trim().getAsInteger(10, result))
    return -1;
  return result;
}
bool routedStorage(const RuntimeSpec &spec, const RuntimeTask &task,
                   ModuleOp module,
                   std::map<std::pair<const void *, int64_t>, size_t> &fifo,
                   std::string &error, size_t singleEvent) {
  auto symbol = field<SymbolRefAttr>(task.configure, "alloc");
  Operation *shim = nullptr;
  module.walk([&](Operation *op) {
    if (named(op, "aie.shim_dma_allocation") &&
        field<StringAttr>(op, "sym_name") &&
        field<StringAttr>(op, "sym_name").getValue() ==
            symbol.getLeafReference())
      shim = op;
  });
  if (!shim || shim->getNumOperands() != 1)
    return fail(error, "missing concrete shim endpoint");
  Value destination;
  int64_t channel = -1;
  size_t routes = 0;
  auto shimChannel = integer(shim, "channel_index");
  auto packet = packetNumber(task.packet);
  int64_t routePacket = task.input ? packet : -1;
  module.walk([&](Operation *op) {
    if (task.input && packet >= 0 && named(op, "aie.packet_flow") &&
        integer(op, "ID") == packet && op->getNumRegions() == 1) {
      Operation *source = nullptr, *sink = nullptr;
      size_t sources = 0, sinks = 0;
      op->walk([&](Operation *inner) {
        if (named(inner, "aie.packet_source")) {
          source = inner;
          ++sources;
        }
        if (named(inner, "aie.packet_dest")) {
          sink = inner;
          ++sinks;
        }
      });
      if (sources == 1 && sinks == 1 && source->getNumOperands() == 1 &&
          sink->getNumOperands() == 1 &&
          source->getOperand(0) == shim->getOperand(0) &&
          integer(source, "bundle") == 1 && integer(sink, "bundle") == 1 &&
          integer(source, "channel") == shimChannel) {
        destination = sink->getOperand(0);
        channel = integer(sink, "channel");
        ++routes;
      }
    }
    if (!task.input && named(op, "aie.packet_flow") &&
        op->getNumRegions() == 1) {
      Operation *source = nullptr, *sink = nullptr;
      size_t sources = 0, sinks = 0;
      op->walk([&](Operation *inner) {
        if (named(inner, "aie.packet_source")) {
          source = inner;
          ++sources;
        }
        if (named(inner, "aie.packet_dest")) {
          sink = inner;
          ++sinks;
        }
      });
      if (sources == 1 && sinks == 1 && source->getNumOperands() == 1 &&
          sink->getNumOperands() == 1 &&
          sink->getOperand(0) == shim->getOperand(0) &&
          integer(source, "bundle") == 1 && integer(sink, "bundle") == 1 &&
          integer(sink, "channel") == shimChannel) {
        destination = source->getOperand(0);
        channel = integer(source, "channel");
        routePacket = integer(op, "ID");
        ++routes;
      }
    }
    if (task.input && !task.packet && named(op, "aie.flow") &&
        op->getNumOperands() == 2 && op->getOperand(0) == shim->getOperand(0) &&
        integer(op, "source_bundle") == 1 && integer(op, "dest_bundle") == 1 &&
        integer(op, "source_channel") == shimChannel) {
      destination = op->getOperand(1);
      channel = integer(op, "dest_channel");
      ++routes;
    }
    if (!task.input && named(op, "aie.flow") && op->getNumOperands() == 2 &&
        op->getOperand(1) == shim->getOperand(0) &&
        integer(op, "dest_bundle") == 1 && integer(op, "source_bundle") == 1 &&
        integer(op, "dest_channel") == shimChannel) {
      destination = op->getOperand(0);
      channel = integer(op, "source_channel");
      ++routes;
    }
  });
  if (routes != 1 || channel < 0 || channel > 1)
    return fail(error, "shim has no unique direct DMA route to checked core");
  std::vector<Operation *> descriptors;
  size_t starts = 0;
  bool malformed = false;
  module.walk([&](Operation *memory) {
    if (!named(memory, "aie.mem") || memory->getNumOperands() != 1 ||
        memory->getOperand(0) != destination || memory->getNumRegions() != 1)
      return;
    for (auto &block : memory->getRegion(0))
      for (auto &op : block) {
        if (!named(&op, "aie.dma_start") ||
            integer(&op, "channel") != channel ||
            integer(&op, "channel_dir") != (task.input ? 0 : 1))
          continue;
        ++starts;
        if (op.getNumSuccessors() != 2) {
          malformed = true;
          continue;
        }
        auto *current = op.getSuccessor(0);
        std::set<Block *> visited;
        while (visited.insert(current).second) {
          for (auto &inner : *current)
            if (named(&inner, "aie.dma_bd")) {
              descriptors.push_back(&inner);
            }
          if (current->empty()) {
            malformed = true;
            break;
          }
          auto &terminator = current->back();
          if (named(&terminator, "aie.end"))
            break;
          if (!named(&terminator, "aie.next_bd") ||
              terminator.getNumSuccessors() != 1) {
            malformed = true;
            break;
          }
          current = terminator.getSuccessor(0);
        }
        if (task.input && current != op.getSuccessor(0))
          malformed = true;
      }
  });
  if (malformed || starts != 1 || descriptors.empty())
    return fail(error, "routed core DMA chain is unsupported or absent");
  for (auto event : task.events) {
    if (event != singleEvent)
      continue;
    auto expected = llvm::find_if(spec.transfers, [&](const auto &entry) {
      return entry.event == event;
    });
    if (expected == spec.transfers.end())
      return fail(error, "routed transfer has no selected source binding");
    Value buffer;
    size_t aliases = 0;
    bool invalid = false;
    auto hostType = dyn_cast<MemRefType>(task.bd->getOperand(0).getType());
    module.walk([&](Operation *view) {
      if (!named(view, "memref.reinterpret_cast") ||
          integer(view, "orbit.allocation") != expected->localAllocation)
        return;
      auto offsets = field<DenseI64ArrayAttr>(view, "static_offsets");
      if (integer(view, "orbit.storage_index") != int64_t(expected->storage) ||
          view->getNumOperands() < 1 || !offsets || offsets.size() != 1 ||
          offsets[0] != expected->localOffset) {
        invalid = true;
        return;
      }
      auto base = view->getOperand(0);
      auto *definition = base.getDefiningOp();
      auto type = dyn_cast<MemRefType>(base.getType());
      if (!definition || !named(definition, "aie.buffer") ||
          definition->getNumOperands() != 1 ||
          definition->getOperand(0) != destination || !hostType || !type ||
          type.getElementType() != hostType.getElementType() ||
          expected->localOffset >
              type.getNumElements() - int64_t(expected->addresses.size()) ||
          (buffer && buffer != base)) {
        invalid = true;
        return;
      }
      buffer = base;
      ++aliases;
    });
    if (invalid || !aliases)
      return fail(
          error,
          "selected local allocation has no consistent checked storage view");
    auto key = std::make_pair(destination.getAsOpaquePointer(),
                              channel + (task.input ? 0 : 2));
    size_t &position = fifo[key];
    Operation *bd = descriptors[position % descriptors.size()];
    ++position;
    {
      if (packetNumber(field<Attribute>(bd, "packet")) != routePacket)
        return fail(error, "selected host transfer does not match the next "
                           "routed core DMA BD packet");
      if (bd->getNumOperands() != 1 || bd->getOperand(0) != buffer ||
          integer(bd, "static_offset") != expected->localOffset ||
          integer(bd, "static_len") != int64_t(expected->addresses.size()))
        return fail(error, "routed core BD does not target selected storage, "
                           "dtype, offset and extent");
    }
  }
  return true;
}
bool inventory(const RuntimeSpec &spec, ModuleOp module,
               std::vector<RuntimeTask> &tasks,
               std::map<size_t, size_t> &eventTask, std::string &error,
               bool requirePacketProof = true) {
  auto graphId = module->getAttrOfType<StringAttr>("orbit.program_graph");
  if (!graphId || graphId.getValue() != spec.graphId)
    return fail(error,
                "final AIE selected graph identity does not match contract");
  std::map<std::string, bool> shims;
  std::map<std::string, Operation *> shimDefinitions;
  bool bad = false;
  module.walk([&](Operation *op) {
    if (named(op, "aie.mem") &&
        (op->getNumRegions() != 1 || op->getRegion(0).getBlocks().size() > 16))
      bad = true;
    if (named(op, "aie.lock")) {
      auto id = integer(op, "lockID");
      if (id < 0 || id >= 16)
        bad = true;
    }
  });
  if (bad)
    return fail(error,
                "actual core DMA block or lock ID exceeds NPU2 static limit");
  module.walk([&](Operation *op) {
    if (!named(op, "aie.shim_dma_allocation")) {
      return;
    }
    auto name = field<StringAttr>(op, "sym_name");
    auto direction = integer(op, "channel_dir");
    if (name)
      shimDefinitions[name.str()] = op;
    if (!name || direction < 0 || direction > 1 ||
        !shims.emplace(name.str(), direction == 1).second) {
      bad = true;
    }
  });
  if (bad || shims.empty()) {
    return fail(error, "ambiguous shim allocation symbols or directions");
  }
  size_t sequences = 0, wrappers = 0;
  Operation *bodySequence = nullptr, *wrapperConfigure = nullptr,
            *wrapperRun = nullptr;
  module.walk([&](Operation *sequence) {
    if (!named(sequence, "aie.runtime_sequence") || bad) {
      return;
    }
    if (sequence->getNumRegions() != 1 ||
        !llvm::hasSingleElement(sequence->getRegion(0))) {
      bad = true;
      return;
    }
    auto &block = sequence->getRegion(0).front();
    if (block.getNumArguments() != spec.abi.size()) {
      bad = true;
      return;
    }
    for (size_t i = 0; i < spec.abi.size(); ++i) {
      std::string type;
      llvm::raw_string_ostream ts(type);
      block.getArgument(i).getType().print(ts);
      if (type != spec.types[i]) {
        bad = true;
      }
    }
    bool wrapper = llvm::any_of(
        block, [](Operation &op) { return named(&op, "aiex.configure"); });
    if (wrapper) {
      ++wrappers;
      if (!llvm::hasSingleElement(block)) {
        bad = true;
        return;
      }
      auto &configure = block.front();
      if (configure.getNumRegions() != 1 ||
          !llvm::hasSingleElement(configure.getRegion(0)) ||
          !llvm::hasSingleElement(configure.getRegion(0).front())) {
        bad = true;
        return;
      }
      auto &run = configure.getRegion(0).front().front();
      wrapperConfigure = &configure;
      wrapperRun = &run;
      if (!named(&run, "aiex.run") || run.getNumOperands() != spec.abi.size()) {
        bad = true;
        return;
      }
      for (size_t i = 0; i < spec.abi.size(); ++i) {
        if (run.getOperand(i) != block.getArgument(i)) {
          bad = true;
        }
      }
      return;
    }
    ++sequences;
    bodySequence = sequence;
    llvm::DenseMap<Value, size_t> handles;
    SmallVector<Operation *> runtimeOps;
    std::map<Operation *, int64_t> repetitions;
    for (Operation &outer : block) {
      if (!requirePacketProof && named(&outer, "arith.constant"))
        continue;
      if (!requirePacketProof && named(&outer, "scf.for")) {
        // Read-only finite inventory of an installed compiler's repeated feed.
        // No IR is cloned or executed: the selected transfer stream below must
        // independently prove every expanded address before reconstruction.
        auto constant = [](Value value) -> std::optional<int64_t> {
          auto *definition = value.getDefiningOp();
          if (!definition || !named(definition, "arith.constant") ||
              !value.getType().isIndex())
            return {};
          auto attr = field<IntegerAttr>(definition, "value");
          return attr ? std::optional<int64_t>(attr.getInt()) : std::nullopt;
        };
        if (outer.getNumOperands() != 3 || outer.getNumResults() ||
            outer.getNumRegions() != 1 ||
            !llvm::hasSingleElement(outer.getRegion(0))) {
          bad = true;
          continue;
        }
        auto lower = constant(outer.getOperand(0));
        auto upper = constant(outer.getOperand(1));
        auto step = constant(outer.getOperand(2));
        auto &body = outer.getRegion(0).front();
        if (!lower || !upper || !step || *lower < 0 || *upper <= *lower ||
            *step <= 0 || body.getNumArguments() != 1 ||
            !body.getArgument(0).use_empty() ||
            body.getOperations().size() != 4) {
          bad = true;
          continue;
        }
        int64_t count = (*upper - *lower - 1) / *step + 1;
        auto it = body.begin();
        Operation *configure = &*it++, *start = &*it++, *free = &*it++,
                  *yield = &*it++;
        if (count > 1024 || !named(configure, "aiex.dma_configure_task_for") ||
            configure->getNumResults() != 1 ||
            !named(start, "aiex.dma_start_task") ||
            !named(free, "aiex.dma_free_task") || !named(yield, "scf.yield") ||
            yield->getNumOperands() || start->getNumOperands() != 1 ||
            free->getNumOperands() != 1 ||
            start->getOperand(0) != configure->getResult(0) ||
            free->getOperand(0) != configure->getResult(0) ||
            std::distance(configure->getResult(0).use_begin(),
                          configure->getResult(0).use_end()) != 2) {
          bad = true;
          continue;
        }
        repetitions[configure] = count;
        runtimeOps.append({configure, start, free});
      } else
        runtimeOps.push_back(&outer);
    }
    for (Operation *operation : runtimeOps) {
      Operation &op = *operation;
      if (named(&op, "aiex.dma_configure_task_for")) {
        if (op.getNumResults() != 1 || op.getNumOperands() != 0 ||
            op.getNumRegions() != 1 ||
            !llvm::hasSingleElement(op.getRegion(0))) {
          bad = true;
          continue;
        }
        if (auto repeat = field<IntegerAttr>(&op, "repeat_count")) {
          if (repeat.getInt() != 0) {
            bad = true;
            continue;
          }
        }
        auto symbol = field<SymbolRefAttr>(&op, "alloc");
        if (!symbol || !shims.count(symbol.getLeafReference().str())) {
          bad = true;
          continue;
        }
        Operation *bd = nullptr;
        for (Operation &inner : op.getRegion(0).front()) {
          if (named(&inner, "aie.dma_bd") && !bd) {
            bd = &inner;
          } else if (!named(&inner, "aie.end")) {
            bad = true;
          }
        }
        if (!bd || bd->getNumOperands() != 1) {
          bad = true;
          continue;
        }
        auto argument = dyn_cast<BlockArgument>(bd->getOperand(0));
        if (!argument || argument.getOwner() != &block) {
          bad = true;
          continue;
        }
        RuntimeTask task;
        task.configure = &op;
        task.bd = bd;
        task.packet = field<Attribute>(bd, "packet");
        task.storage = spec.abi[argument.getArgNumber()];
        task.input = shims[symbol.getLeafReference().str()];
        int64_t length = integer(bd, "static_len");
        auto sizes = field<DenseI64ArrayAttr>(bd, "static_sizes");
        auto strides = field<DenseI64ArrayAttr>(bd, "static_strides");
        bool valid =
            sizes && strides && !sizes.empty()
                ? addresses(sizes.asArrayRef(), strides.asArrayRef(),
                            integer(bd, "static_offset"), task.addresses)
                : addresses({length}, {1}, integer(bd, "static_offset"),
                            task.addresses);
        if (!valid || length != int64_t(task.addresses.size()) ||
            (!task.input &&
             (!field<BoolAttr>(&op, "issue_token") ||
              !field<BoolAttr>(&op, "issue_token").getValue()))) {
          bad = true;
          continue;
        }
        if (auto repeat = repetitions.find(&op); repeat != repetitions.end()) {
          if (!task.input ||
              task.addresses.size() > size_t(1000000 / repeat->second)) {
            bad = true;
            continue;
          }
          const auto one = task.addresses;
          for (int64_t i = 1; i < repeat->second; ++i)
            task.addresses.insert(task.addresses.end(), one.begin(), one.end());
        }
        handles.insert({op.getResult(0), tasks.size()});
        tasks.push_back(std::move(task));
      } else if (named(&op, "aiex.dma_start_task") ||
                 named(&op, "aiex.dma_await_task") ||
                 named(&op, "aiex.dma_free_task")) {
        if (op.getNumOperands() != 1 || !handles.count(op.getOperand(0))) {
          bad = true;
          continue;
        }
        RuntimeTask &task = tasks[handles[op.getOperand(0)]];
        Operation **slot = named(&op, "aiex.dma_start_task")   ? &task.start
                           : named(&op, "aiex.dma_await_task") ? &task.await
                                                               : &task.free;
        if (*slot) {
          bad = true;
        }
        *slot = &op;
      } else {
        bad = true;
      }
    }
  });
  if (bad || sequences != 1 || wrappers != 1) {
    return fail(error,
                "unsupported runtime operation, loop, wrapper or descriptor");
  }
  auto configSymbol = field<SymbolRefAttr>(wrapperConfigure, "symbol");
  auto runSymbol = field<SymbolRefAttr>(wrapperRun, "runtime_sequence_symbol");
  auto sequenceName = field<StringAttr>(bodySequence, "sym_name");
  auto deviceName = field<StringAttr>(bodySequence->getParentOp(), "sym_name");
  if (!configSymbol || !runSymbol || !sequenceName || !deviceName ||
      configSymbol.getRootReference() != deviceName.getValue() ||
      runSymbol.getLeafReference() != sequenceName.getValue()) {
    return fail(
        error,
        "wrapper does not invoke the checked configuration and sequence");
  }
  std::map<std::pair<size_t, bool>, std::vector<const TransferSpec *>> streams;
  std::map<std::pair<size_t, bool>, size_t> cursors;
  for (const auto &transfer : spec.transfers)
    streams[{transfer.storage, transfer.input}].push_back(&transfer);
  std::map<std::pair<const void *, int64_t>, size_t> fifo;
  for (auto &task : tasks) {
    if (!task.start || !task.configure->isBeforeInBlock(task.start) ||
        (!task.input && !task.await) ||
        (task.await && !task.start->isBeforeInBlock(task.await)) ||
        (task.free && !task.start->isBeforeInBlock(task.free)) ||
        (task.free && task.await && !task.await->isBeforeInBlock(task.free))) {
      return fail(error, "task start/completion/free coverage is invalid");
    }
    auto key = std::make_pair(task.storage, task.input);
    auto &stream = streams[key];
    auto &cursor = cursors[key];
    size_t consumed = 0;
    // A compiler may coalesce adjacent transfers on one FIFO. Accept only an
    // exact concatenation of complete selected transfers, in their source-
    // checked stream order. Repeated addresses are retained, never set-unioned.
    while (consumed < task.addresses.size()) {
      if (cursor >= stream.size())
        return fail(error, "host BD has extra elements beyond selected FIFO");
      const auto &expected = *stream[cursor];
      if (expected.addresses.size() > task.addresses.size() - consumed ||
          !std::equal(expected.addresses.begin(), expected.addresses.end(),
                      task.addresses.begin() + consumed))
        return fail(
            error,
            "host BD does not preserve exact selected FIFO address order");
      consumed += expected.addresses.size();
      task.events.push_back(expected.event);
      if (!eventTask.emplace(expected.event, &task - tasks.data()).second)
        return fail(
            error, "selected transfer appears in multiple runtime descriptors");
      ++cursor;
    }
    if (task.events.empty())
      return fail(error, "empty runtime transfer task");
    task.event = task.events.front();

    if (requirePacketProof && task.input) {
      if (task.events.size() != 1 || !task.await ||
          !field<BoolAttr>(task.configure, "issue_token") ||
          !field<BoolAttr>(task.configure, "issue_token").getValue())
        return fail(error, "packet input requires one selected transfer and "
                           "explicit completion");
      // A missing packet is legal only for an independently checked unique
      // circuit route; deleting a packet on a packet-flow feed still rejects.
      bool foundConsumer = false, invalidConsumer = false;
      module.walk([&](Operation *op) {
        if (!named(op, "aie.dma_bd") ||
            field<Attribute>(op, "packet") != task.packet)
          return;
        Operation *parent = op->getParentOp();
        while (parent && !named(parent, "aie.mem"))
          parent = parent->getParentOp();
        if (!parent)
          return;
        foundConsumer = true;
        if (integer(op, "static_len") != int64_t(task.addresses.size()))
          invalidConsumer = true;
      });
      if (task.packet && (!foundConsumer || invalidConsumer))
        return fail(error, "host packet payload does not equal every matching "
                           "core BD payload");
    }
  }
  if (eventTask.size() != spec.transfers.size())
    return fail(
        error, "actual runtime does not cover every selected transfer exactly");
  if (requirePacketProof) {
    std::set<std::pair<const void *, int64_t>> outputFifos;
    for (const auto &task : tasks) {
      if (task.input)
        continue;
      auto symbol = field<SymbolRefAttr>(task.configure, "alloc");
      auto *shim = shimDefinitions.at(symbol.getLeafReference().str());
      if (shim->getNumOperands() != 1 ||
          !outputFifos
               .emplace(shim->getOperand(0).getAsOpaquePointer(),
                        integer(shim, "channel_index"))
               .second)
        return fail(
            error,
            "multiple output tasks on one physical shim FIFO are unsupported");
    }
    const RuntimeTask *previous = nullptr;
    for (const auto &transfer : spec.transfers) {
      const auto &task = tasks[eventTask.at(transfer.event)];
      if (!routedStorage(spec, task, module, fifo, error, transfer.event))
        return false;
      if (!transfer.input)
        continue;
      if (previous &&
          (!previous->await || !previous->await->isBeforeInBlock(task.start)))
        return fail(
            error,
            "packet input FIFO order lacks completion-before-next-start");
      previous = &task;
    }
  }
  if (eventTask.size() != spec.transfers.size()) {
    return fail(
        error, "actual runtime does not cover every selected transfer exactly");
  }
  return true;
}
bool dependencies(const RuntimeSpec &spec,
                  const std::vector<RuntimeTask> &tasks,
                  const std::map<size_t, size_t> &index,
                  std::vector<std::pair<size_t, size_t>> &edges,
                  std::string &error) {
  for (auto [producerEvent, consumerEvent] : spec.feedback) {
    auto p = index.find(producerEvent), c = index.find(consumerEvent);
    if (p == index.end() || c == index.end() || tasks[p->second].input ||
        !tasks[c->second].input || !tasks[p->second].await ||
        !tasks[p->second].start->isBeforeInBlock(tasks[c->second].start)) {
      return fail(error, "host reload producer has no preceding export task");
    }
    edges.emplace_back(p->second, c->second);
  }
  return true;
}
bool realizePacketInputs(const RuntimeSpec &spec,
                         const std::vector<RuntimeTask> &tasks,
                         const std::map<size_t, size_t> &index,
                         std::string &error) {
  if (tasks.empty())
    return fail(error, "no runtime tasks to realize");
  Operation *sequence = tasks.front().configure;
  while (sequence && !named(sequence, "aie.runtime_sequence"))
    sequence = sequence->getParentOp();
  if (!sequence)
    return fail(error, "runtime task has no owning sequence");
  auto *old = &sequence->getRegion(0).front();
  auto *block = new Block;
  for (auto arg : old->getArguments())
    block->addArgument(arg.getType(), arg.getLoc());
  sequence->getRegion(0).push_back(block);
  OpBuilder b(sequence->getContext());
  b.setInsertionPointToStart(block);
  auto create = [&](StringRef name, ValueRange operands, TypeRange results,
                    DictionaryAttr attrs, Attribute props, bool region) {
    OperationState state(b.getUnknownLoc(), name);
    state.addOperands(operands);
    state.addTypes(results);
    if (attrs)
      state.addAttributes(attrs.getValue());
    state.propertiesAttr = props;
    if (region)
      state.addRegion();
    return b.create(state);
  };
  auto unary = [&](StringRef name, Value handle) {
    return create(name, {handle}, {}, {}, {}, false);
  };
  auto configure = [&](const RuntimeTask &original,
                       const TransferSpec *transfer) -> Value {
    auto originalProps = dyn_cast_or_null<DictionaryAttr>(
        original.configure->getPropertiesAsAttribute());
    if (!originalProps)
      return {};
    NamedAttrList props(originalProps);
    props.set("issue_token", b.getBoolAttr(true));
    props.set("repeat_count", b.getI32IntegerAttr(0));
    auto *task = create("aiex.dma_configure_task_for", {},
                        original.configure->getResultTypes(),
                        original.configure->getAttrDictionary(),
                        props.getDictionary(b.getContext()), true);
    auto *body = new Block;
    task->getRegion(0).push_back(body);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(body);
    auto oldArgument = cast<BlockArgument>(original.bd->getOperand(0));
    NamedAttrList bdProps(
        cast<DictionaryAttr>(original.bd->getPropertiesAsAttribute()));
    if (transfer) {
      bdProps.set("static_len",
                  b.getI32IntegerAttr(transfer->addresses.size()));
      bdProps.set("static_offset", b.getI32IntegerAttr(transfer->offset));
      bdProps.set("static_sizes", b.getDenseI64ArrayAttr(transfer->sizes));
      bdProps.set("static_strides", b.getDenseI64ArrayAttr(transfer->strides));
    }
    create("aie.dma_bd", {block->getArgument(oldArgument.getArgNumber())}, {},
           original.bd->getAttrDictionary(),
           bdProps.getDictionary(b.getContext()), false);
    create("aie.end", {}, {}, {}, {}, false);
    return task->getResult(0);
  };
  std::map<size_t, Value> outputHandles;
  std::set<size_t> completedOutputs, startedInputs;
  for (size_t i = 0; i < tasks.size(); ++i)
    if (!tasks[i].input) {
      Value handle = configure(tasks[i], nullptr);
      if (!handle)
        return fail(error, "output task properties are unavailable");
      unary("aiex.dma_start_task", handle);
      outputHandles[i] = handle;
    }
  auto awaitOutput = [&](size_t taskIndex) {
    if (completedOutputs.count(taskIndex))
      return true;
    // An output descriptor may aggregate multiple selected exports. Waiting
    // early is legal only when all input ancestors of every export were issued.
    std::set<size_t> visited;
    SmallVector<size_t> pending(tasks[taskIndex].events.begin(),
                                tasks[taskIndex].events.end());
    while (!pending.empty()) {
      size_t event = pending.pop_back_val();
      if (!visited.insert(event).second)
        continue;
      auto mapped = index.find(event);
      if (mapped != index.end() && tasks[mapped->second].input &&
          !startedInputs.count(event))
        return false;
      llvm::append_range(pending, spec.dependencies[event]);
    }
    unary("aiex.dma_await_task", outputHandles.at(taskIndex));
    completedOutputs.insert(taskIndex);
    return true;
  };
  for (const auto &transfer : spec.transfers) {
    if (!transfer.input)
      continue;
    if (transfer.offset > INT32_MAX || transfer.addresses.size() > INT32_MAX)
      return fail(error,
                  "selected packet descriptor exceeds static integer range");
    for (auto [producer, consumer] : spec.feedback)
      if (consumer == transfer.event)
        if (!awaitOutput(index.at(producer)))
          return fail(
              error,
              "packet input would await output before its input ancestors");
    Value handle = configure(tasks[index.at(transfer.event)], &transfer);
    if (!handle)
      return fail(error, "input task properties are unavailable");
    unary("aiex.dma_start_task", handle);
    // Packets share a core DMA FIFO. Complete each entire selected packet
    // before a packet from another shim channel can enter that FIFO.
    unary("aiex.dma_await_task", handle);
    unary("aiex.dma_free_task", handle);
    startedInputs.insert(transfer.event);
  }
  for (const auto &[task, handle] : outputHandles) {
    if (!awaitOutput(task))
      return fail(error, "output completion has unissued input ancestors");
    unary("aiex.dma_free_task", handle);
  }
  old->erase();
  return true;
}
} // namespace

// Reject the old numerically unverified repeated-feed configuration even if
// its address/route/FIFO certificate passes. Contracts do not carry a trusted
// per-target lock-lifetime proof, so this guard intentionally also rejects
// equal host streams aimed at distinct targets. The source-bound selection
// checker can use target identity to construct safe separate configurations.
static bool checkTransferIsolation(const RuntimeSpec &spec,
                                   std::string &error) {
  // Multiple selected outputs can share a physical buffer even on distinct
  // routes. Address/route/FIFO coverage cannot establish that each output
  // observes its own value before the next write. Keep one host output until
  // an independent core-buffer lock-lifetime proof is available.
  if (llvm::count_if(spec.transfers,
                     [](const auto &transfer) { return !transfer.input; }) > 1)
    return fail(error, "multiple host output transfers in one configuration "
                       "are unsupported: core output lock lifetime is not "
                       "certified; isolate at a checked local-copy frontier");
  std::set<std::pair<size_t, std::vector<int64_t>>> inputs;
  for (const auto &transfer : spec.transfers)
    if (transfer.input &&
        !inputs.emplace(transfer.storage, transfer.addresses).second)
      return fail(error, "repeated host input region in one configuration is "
                         "unsupported: core lock/output lifetime is not "
                         "certified; isolate at a checked local-copy frontier");
  return true;
}
static bool checkRuntime(const RuntimeSpec &spec, ModuleOp module,
                         llvm::json::Value &receipt, std::string &error) {
  std::vector<RuntimeTask> tasks;
  std::map<size_t, size_t> index;
  std::vector<std::pair<size_t, size_t>> edges;
  if (!inventory(spec, module, tasks, index, error) ||
      !checkTransferIsolation(spec, error) ||
      !dependencies(spec, tasks, index, edges, error)) {
    return false;
  }
  llvm::json::Array checks;
  for (auto [producer, consumer] : edges) {
    if (!tasks[producer].await->isBeforeInBlock(tasks[consumer].start)) {
      return fail(error,
                  "host reload can prefetch before selected export completes");
    }
    checks.push_back(
        llvm::json::Object{{"producer_event", int64_t(tasks[producer].event)},
                           {"consumer_event", int64_t(tasks[consumer].event)}});
  }
  receipt = llvm::json::Object{
      {"schema", "orbit.matrix-runtime-ordering.v32"},
      {"program_graph_id", spec.graphId},
      {"runtime_contract_sha256", spec.contractHash},
      {"source_body_fingerprint", spec.sourceFingerprint},
      {"generic_module_sha256", moduleFingerprint(module)},
      {"selected_transfer_count", int64_t(index.size())},
      {"runtime_descriptor_task_count", int64_t(tasks.size())},
      {"fifo_coverage", "exact ordered concatenation of complete selected "
                        "transfers, preserving repeated addresses"},
      {"host_reload_completion_edges", std::move(checks)},
      {"core_numeric_checked", false},
      {"core_dma_route_and_storage_checked", true},
      {"core_dma_entry_and_fifo_order_checked", true},
      {"output_fifo_unique_task_checked", true},
      {"core_lock_protocol_checked", false},
      {"kernel_arithmetic_checked", false},
      {"input_packet_boundaries_checked", true},
      {"repeated_host_input_regions_isolated", true},
      {"maximum_host_output_transfers_per_configuration", 1},
      {"input_packet_completion_order_checked", true},
      {"input_circuit_route_and_storage_checked", true},
      {"scope", "static exact host BD coverage and completion-before-reload; "
                "no kernel arithmetic or timing certificate"}};
  if (spec.selectionBegin >= 0) {
    (*receipt.getAsObject())["event_selection"] =
        llvm::json::Array{spec.selectionBegin, spec.selectionEnd};
    (*receipt.getAsObject())["requires_complete_union_and_serial_wait2"] = true;
  }
  return true;
}

static bool realizeRuntime(const RuntimeSpec &spec, ModuleOp input,
                           OwningOpRef<ModuleOp> &output,
                           llvm::json::Value &receipt, std::string &error) {
  // This MLIR version drops unregistered operation properties in clone().
  // Round-trip generic syntax so AIE inherent properties remain intact.
  std::string serialized;
  llvm::raw_string_ostream stream(serialized);
  input.print(stream, OpPrintingFlags().printGenericOpForm());
  output = parseSourceString<ModuleOp>(
      serialized, ParserConfig(input.getContext(), /*verifyAfterParse=*/false));
  if (!output)
    return fail(error, "cannot preserve generic AIE properties");
  std::vector<RuntimeTask> tasks;
  std::map<size_t, size_t> index;
  std::vector<std::pair<size_t, size_t>> edges;
  if (!inventory(spec, *output, tasks, index, error, false) ||
      !dependencies(spec, tasks, index, edges, error)) {
    output = {};
    return false;
  }
  bool hasInputs =
      llvm::any_of(tasks, [](const auto &task) { return task.input; });
  if (hasInputs) {
    if (!realizePacketInputs(spec, tasks, index, error)) {
      output = {};
      return false;
    }
    tasks.clear();
    index.clear();
    edges.clear();
    if (!inventory(spec, *output, tasks, index, error) ||
        !dependencies(spec, tasks, index, edges, error)) {
      output = {};
      return false;
    }
  }
  for (auto [producer, consumer] : edges) {
    Operation *await = tasks[producer].await;
    Operation *start = tasks[consumer].start;
    if (await->isBeforeInBlock(start)) {
      continue;
    }
    // Every ancestor transfer must already have started before the new wait;
    // otherwise the controller would block before issuing a needed input.
    std::set<size_t> visited;
    SmallVector<size_t> pending(tasks[producer].events.begin(),
                                tasks[producer].events.end());
    while (!pending.empty()) {
      size_t event = pending.pop_back_val();
      if (!visited.insert(event).second) {
        continue;
      }
      auto mapped = index.find(event);
      if (mapped != index.end() &&
          !tasks[mapped->second].start->isBeforeInBlock(start)) {
        return fail(error,
                    "moving host await would block an unissued producer input");
      }
      llvm::append_range(pending, spec.dependencies[event]);
    }
    await->moveBefore(start);
  }
  if (!checkRuntime(spec, *output, receipt, error)) {
    output = {};
    return false;
  }
  size_t feedLoops = 0;
  input.walk([&](Operation *op) {
    if (!named(op, "scf.for"))
      return;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp())
      if (named(parent, "aie.runtime_sequence")) {
        ++feedLoops;
        break;
      }
  });
  (*receipt.getAsObject())["constant_feed_loops_expanded"] = int64_t(feedLoops);
  (*receipt.getAsObject())["final_runtime_has_loops"] = false;
  (*receipt.getAsObject())["input_generic_module_sha256"] =
      moduleFingerprint(input);
  (*receipt.getAsObject())["realization"] =
      "exact per-selected input packets with serialized completion; existing "
      "output payloads and source copies preserved";
  return true;
}

namespace {
std::string jsonHash(const llvm::json::Value &value) {
  std::string bytes = llvm::formatv("{0}", value).str();
  auto digest = llvm::SHA256::hash(ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  return llvm::toHex(digest, true);
}
bool integers(const llvm::json::Array *array, std::vector<int64_t> &out) {
  if (!array)
    return false;
  for (const auto &entry : *array) {
    auto value = entry.getAsInteger();
    if (!value || *value < 0)
      return false;
    out.push_back(*value);
  }
  return true;
}
bool parseContract(const llvm::json::Value &value, RuntimeSpec &spec,
                   std::string &error) {
  auto *root = value.getAsObject();
  if (!root ||
      root->getString("schema") != "orbit.matrix-runtime-contract.v32" ||
      !root->getString("program_graph_id") ||
      !root->getString("source_body_fingerprint"))
    return fail(error, "invalid sealed runtime contract header");
  spec.graphId = root->getString("program_graph_id")->str();
  spec.sourceFingerprint = root->getString("source_body_fingerprint")->str();
  if (spec.graphId.size() != 64 || spec.sourceFingerprint.size() != 64)
    return fail(error, "invalid source or graph identity");
  spec.contractHash = jsonHash(value);
  auto *abi = root->getArray("abi");
  auto *events = root->getArray("event_dependencies");
  auto *transfers = root->getArray("transfers");
  auto *feedback = root->getArray("feedback");
  if (!abi || abi->empty() || !events || events->empty() || !transfers ||
      !feedback)
    return fail(error, "runtime contract is missing required tables");
  std::set<int64_t> ids;
  for (const auto &entry : *abi) {
    auto *row = entry.getAsObject();
    auto storage = row ? row->getInteger("storage") : std::nullopt;
    auto type = row ? row->getString("type") : std::nullopt;
    if (!storage || *storage < 0 || !type || type->empty() ||
        !ids.insert(*storage).second)
      return fail(error, "runtime contract ABI is invalid");
    spec.abi.push_back(*storage);
    spec.types.push_back(type->str());
  }
  for (size_t i = 0; i < events->size(); ++i) {
    std::vector<int64_t> deps;
    if (!integers((*events)[i].getAsArray(), deps))
      return fail(error, "invalid runtime event dependencies");
    std::set<size_t> unique;
    for (auto dep : deps)
      if (uint64_t(dep) >= events->size() || size_t(dep) == i ||
          !unique.insert(dep).second)
        return fail(error, "runtime event dependency is invalid");
    spec.dependencies.emplace_back(deps.begin(), deps.end());
  }
  if (auto *selection = root->getArray("event_selection")) {
    std::vector<int64_t> bounds;
    if (!integers(selection, bounds) || bounds.size() != 2 ||
        bounds[0] >= bounds[1] || uint64_t(bounds[1]) > events->size())
      return fail(error, "invalid runtime contract event selection");
    spec.selectionBegin = bounds[0];
    spec.selectionEnd = bounds[1];
  }
  // Validate acyclicity independently; serialized event order is not assumed.
  std::vector<unsigned> states(events->size());
  std::function<bool(size_t)> visit = [&](size_t i) {
    if (states[i] == 1)
      return false;
    if (states[i] == 2)
      return true;
    states[i] = 1;
    for (auto dep : spec.dependencies[i])
      if (!visit(dep))
        return false;
    states[i] = 2;
    return true;
  };
  for (size_t i = 0; i < states.size(); ++i)
    if (!visit(i))
      return fail(error, "runtime contract dependency cycle");
  std::map<size_t, bool> directions;
  for (const auto &entry : *transfers) {
    auto *row = entry.getAsObject();
    if (!row)
      return fail(error, "invalid runtime transfer row");
    auto event = row->getInteger("event"), storage = row->getInteger("storage"),
         offset = row->getInteger("offset");
    auto input = row->getBoolean("host_input");
    std::vector<int64_t> sizes, strides;
    if (!event || *event < 0 || uint64_t(*event) >= events->size() ||
        !storage || !ids.count(*storage) || !offset || !input ||
        !integers(row->getArray("sizes"), sizes) || sizes.empty() ||
        !integers(row->getArray("strides"), strides) ||
        !directions.emplace(*event, *input).second)
      return fail(error, "runtime contract transfer fields are invalid");
    if (spec.selectionBegin >= 0 &&
        (*event < spec.selectionBegin || *event >= spec.selectionEnd))
      return fail(error,
                  "runtime transfer lies outside selected configuration");
    if (!spec.transfers.empty() &&
        size_t(*event) <= spec.transfers.back().event)
      return fail(error,
                  "runtime contract transfers are not in selected event order");
    auto localAllocation = row->getInteger("local_allocation");
    auto localOffset = row->getInteger("local_offset_elements");
    if (!localAllocation || *localAllocation < 0 || !localOffset ||
        *localOffset < 0)
      return fail(error,
                  "source-bound local allocation and offset are required");
    TransferSpec transfer;
    transfer.event = *event;
    transfer.storage = *storage;
    transfer.input = *input;
    transfer.offset = *offset;
    transfer.localAllocation = *localAllocation;
    transfer.localOffset = *localOffset;
    transfer.sizes = sizes;
    transfer.strides = strides;
    if (!addresses(sizes, strides, *offset, transfer.addresses))
      return fail(error, "runtime contract transfer descriptor is invalid");
    spec.transfers.push_back(std::move(transfer));
  }
  std::set<std::pair<size_t, size_t>> unique;
  for (const auto &entry : *feedback) {
    std::vector<int64_t> edge;
    if (!integers(entry.getAsArray(), edge) || edge.size() != 2 ||
        !directions.count(edge[0]) || !directions.count(edge[1]) ||
        directions[edge[0]] || !directions[edge[1]] ||
        !unique.emplace(edge[0], edge[1]).second)
      return fail(error, "invalid runtime feedback dependency");
    std::set<size_t> ancestors;
    SmallVector<size_t> pending{size_t(edge[1])};
    while (!pending.empty()) {
      size_t event = pending.pop_back_val();
      if (ancestors.insert(event).second)
        llvm::append_range(pending, spec.dependencies[event]);
    }
    if (!ancestors.count(edge[0]))
      return fail(error, "feedback producer is not a selected graph ancestor");
    spec.feedback.emplace_back(edge[0], edge[1]);
  }
  return true;
}
} // namespace

llvm::json::Value orbit::taskflow::serializeProgramMatrixRuntimeContract(
    const ProgramGraph &g, ArrayRef<int64_t> abi, std::string &error) {
  if (!checkProgramGraph(g, error) ||
      !checkProgramGraphSource(g.source, g, error))
    return nullptr;
  llvm::json::Array abiRows, events, transfers, feedback;
  std::set<int64_t> ids;
  for (auto storage : abi) {
    if (storage < 0 || uint64_t(storage) >= g.source.storages.size() ||
        !ids.insert(storage).second) {
      fail(error, "invalid ABI storage binding");
      return nullptr;
    }
    std::string type;
    llvm::raw_string_ostream stream(type);
    g.source.storages[storage].root.getType().print(stream);
    abiRows.push_back(llvm::json::Object{{"storage", storage}, {"type", type}});
  }
  for (size_t ei = 0; ei < g.events.size(); ++ei) {
    const auto &event = g.events[ei];
    llvm::json::Array deps;
    for (auto dep : event.dependencies)
      deps.push_back(int64_t(dep));
    events.push_back(std::move(deps));
    if (event.kind != "transfer")
      continue;
    const auto &region = *event.transferRegion;
    if (!ids.count(region.storage)) {
      fail(error, "selected transfer storage is absent from ABI");
      return nullptr;
    }
    const auto &shape = g.source.storages[region.storage].shape;
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t d = int64_t(shape.size()) - 2; d >= 0; --d)
      strides[d] = strides[d + 1] * shape[d + 1];
    int64_t offset = 0;
    llvm::json::Array sizesJSON, stridesJSON;
    for (size_t d = 0; d < shape.size(); ++d) {
      offset += strides[d] * region.offsets[d];
      sizesJSON.push_back(region.sizes[d]);
      stridesJSON.push_back(strides[d]);
    }
    bool input = hostInput(g, event);
    const auto &localCopies = input ? event.writes : event.reads;
    std::set<size_t> allocations;
    for (auto copy : localCopies)
      allocations.insert(g.copies[copy].allocation);
    if (allocations.size() != 1) {
      fail(error, "transfer does not name one physical local slot");
      return nullptr;
    }
    size_t local = *allocations.begin();
    const auto &allocation = g.allocations[local];
    auto type =
        cast<MemRefType>(g.source.storages[region.storage].root.getType());
    int64_t itemBytes = (type.getElementTypeBitWidth() + 7) / 8,
            localOffset = 0;
    for (size_t ai = 0; ai < local; ++ai)
      if (g.allocations[ai].storage == region.storage &&
          g.allocations[ai].memory == allocation.memory)
        localOffset += g.allocations[ai].bytes / itemBytes;
    transfers.push_back(
        llvm::json::Object{{"event", int64_t(ei)},
                           {"storage", int64_t(region.storage)},
                           {"host_input", input},
                           {"local_allocation", int64_t(local)},
                           {"local_offset_elements", localOffset},
                           {"offset", offset},
                           {"sizes", std::move(sizesJSON)},
                           {"strides", std::move(stridesJSON)}});
    if (input) {
      std::set<int64_t> producers;
      for (auto copy : event.reads)
        if (g.copies[copy].createdBy >= 0)
          producers.insert(g.copies[copy].createdBy);
      for (auto producer : producers)
        feedback.push_back(llvm::json::Array{producer, int64_t(ei)});
    }
  }
  llvm::json::Value result = llvm::json::Object{
      {"schema", "orbit.matrix-runtime-contract.v32"},
      {"program_graph_id", programGraphIdentity(g)},
      {"source_body_fingerprint", g.source.bodyFingerprint},
      {"abi", std::move(abiRows)},
      {"event_dependencies", std::move(events)},
      {"transfers", std::move(transfers)},
      {"feedback", std::move(feedback)},
      {"trust_boundary",
       "exported only after source and selected graph checks; caller must bind "
       "this exact contract hash to its physical receipt"}};
  RuntimeSpec validated;
  if (!parseContract(result, validated, error))
    return nullptr;
  return result;
}

bool orbit::taskflow::checkProgramMatrixRuntime(
    const llvm::json::Value &contract, ModuleOp module,
    llvm::json::Value &receipt, std::string &error) {
  RuntimeSpec spec;
  return parseContract(contract, spec, error) &&
         checkRuntime(spec, module, receipt, error);
}
bool orbit::taskflow::realizeProgramMatrixRuntime(
    const llvm::json::Value &contract, ModuleOp input,
    OwningOpRef<ModuleOp> &output, llvm::json::Value &receipt,
    std::string &error) {
  RuntimeSpec spec;
  return parseContract(contract, spec, error) &&
         realizeRuntime(spec, input, output, receipt, error);
}
bool orbit::taskflow::checkProgramMatrixRuntime(const ProgramGraph &g,
                                                ArrayRef<int64_t> abi,
                                                ModuleOp module,
                                                llvm::json::Value &receipt,
                                                std::string &error) {
  auto contract = serializeProgramMatrixRuntimeContract(g, abi, error);
  return !contract.getAsNull() &&
         checkProgramMatrixRuntime(contract, module, receipt, error);
}
bool orbit::taskflow::realizeProgramMatrixRuntime(const ProgramGraph &g,
                                                  ArrayRef<int64_t> abi,
                                                  ModuleOp input,
                                                  OwningOpRef<ModuleOp> &output,
                                                  llvm::json::Value &receipt,
                                                  std::string &error) {
  auto contract = serializeProgramMatrixRuntimeContract(g, abi, error);
  return !contract.getAsNull() &&
         realizeProgramMatrixRuntime(contract, input, output, receipt, error);
}

llvm::json::Value orbit::taskflow::serializeProgramMatrixRuntimeContract(
    const ProgramGraph &g, ArrayRef<int64_t> abi,
    const ProgramGraphAIRSelection &selection, std::string &error) {
  if (!checkProgramGraphAIRSelection(g, selection, error))
    return nullptr;
  auto result = serializeProgramMatrixRuntimeContract(g, abi, error);
  if (result.getAsNull())
    return nullptr;
  auto *root = result.getAsObject();
  llvm::json::Array transfers, feedback;
  for (const auto &row : *root->getArray("transfers")) {
    auto event = *row.getAsObject()->getInteger("event");
    if (uint64_t(event) >= selection.begin && uint64_t(event) < selection.end)
      transfers.push_back(row);
  }
  for (const auto &row : *root->getArray("feedback")) {
    const auto *edge = row.getAsArray();
    auto producer = *(*edge)[0].getAsInteger(),
         consumer = *(*edge)[1].getAsInteger();
    if (uint64_t(producer) >= selection.begin &&
        uint64_t(producer) < selection.end &&
        uint64_t(consumer) >= selection.begin &&
        uint64_t(consumer) < selection.end)
      feedback.push_back(row);
  }
  (*root)["transfers"] = std::move(transfers);
  (*root)["feedback"] = std::move(feedback);
  (*root)["event_selection"] =
      llvm::json::Array{int64_t(selection.begin), int64_t(selection.end)};
  (*root)["cross_configuration_dependencies"] =
      "complete union executes serially with native wait2; no private "
      "intermediate syncTo";
  RuntimeSpec validated;
  if (!parseContract(result, validated, error))
    return nullptr;
  return result;
}
