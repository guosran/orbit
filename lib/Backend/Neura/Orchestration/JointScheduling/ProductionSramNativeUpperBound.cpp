//===- ProductionSramNativeUpperBound.cpp ---------------------*- C++ -*-===//
//
// Export conservative storage facts from the native Taskflow IR.  This file
// deliberately does not infer a byte capacity from architecture context
// counts or from scheduler coordinates.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/Orchestration/JointScheduling/ProductionSramNativeUpperBound.h"

#include "Backend/Neura/Orchestration/JointScheduling/TaskGraphRewriteLegality.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace mlir::amoeba::neura::joint_scheduling {
namespace {

constexpr StringLiteral kLogicalTransferShapeAttr =
    "amoeba.logical_transfer_shape";
constexpr StringLiteral kStorageResidencyAttr =
    "joint_scheduling_storage_residency";
constexpr StringLiteral kStorageCopyBindingAttr =
    "joint_scheduling_storage_copy_binding";
constexpr StringLiteral kStorageAddressBindingAttr =
    "joint_scheduling_storage_address_binding";

struct ExtentFact {
  bool proven = false;
  int64_t bytes = 0;
  std::string error;
};

struct ResolvedBacking {
  Value root;
  std::string error;
};

struct ExportState {
  struct TaskStorageCoordinates {
    llvm::DenseMap<Value, SmallVector<int64_t, 4>> reads;
    llvm::DenseMap<Value, SmallVector<int64_t, 4>> writes;
  };

  NativeSramUpperBoundResult result;
  func::FuncOp function;
  int64_t gridRows = 0;
  int64_t gridColumns = 0;
  llvm::DenseMap<Value, unsigned> backingIndices;
  std::vector<std::set<int64_t>> backingCgras;
  std::set<std::pair<unsigned, int64_t>> chargedBackingCgras;
  std::vector<std::string> reasons;
  bool sawStorageResidencyContract = false;
  bool sawStorageCopyBindingContract = false;
  llvm::DenseMap<Operation *, TaskStorageCoordinates> taskStorageCoordinates;
  llvm::DenseSet<Operation *> indexedOperationsSeen;
  uint64_t nextIndexedAccessOrdinal = 0;

  void addReason(StringRef reason) {
    if (reason.empty())
      return;
    if (std::find(reasons.begin(), reasons.end(), reason) == reasons.end())
      reasons.push_back(reason.str());
  }

};

static bool safeMultiply(uint64_t lhs, uint64_t rhs, uint64_t &out) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  out = lhs * rhs;
  return true;
}

static std::optional<uint64_t> scalarBitWidth(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type)) {
    unsigned width = integer.getWidth();
    return width == 0 ? std::nullopt
                      : std::optional<uint64_t>(static_cast<uint64_t>(width));
  }
  if (auto floating = dyn_cast<FloatType>(type)) {
    unsigned width = floating.getWidth();
    return width == 0 ? std::nullopt
                      : std::optional<uint64_t>(static_cast<uint64_t>(width));
  }
  if (auto complex = dyn_cast<ComplexType>(type)) {
    auto width = scalarBitWidth(complex.getElementType());
    if (!width || *width > std::numeric_limits<uint64_t>::max() / 2)
      return std::nullopt;
    return *width * 2;
  }
  if (auto vector = dyn_cast<VectorType>(type)) {
    auto width = scalarBitWidth(vector.getElementType());
    if (!width)
      return std::nullopt;
    uint64_t count = 1;
    for (int64_t extent : vector.getShape()) {
      if (extent <= 0 || !safeMultiply(count, static_cast<uint64_t>(extent),
                                        count))
        return std::nullopt;
    }
    if (!safeMultiply(count, *width, count))
      return std::nullopt;
    return count;
  }
  // Index width is target-dependent.  A host-side native artifact does not
  // establish its storage width, so it must not silently be charged as i64.
  return std::nullopt;
}

static bool parseShapeAttribute(Attribute attribute,
                                SmallVectorImpl<int64_t> &extents,
                                std::string &error) {
  extents.clear();
  if (auto dense = dyn_cast<DenseI64ArrayAttr>(attribute)) {
    extents.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
    return true;
  }
  auto array = dyn_cast<ArrayAttr>(attribute);
  if (!array) {
    error = "amoeba.logical_transfer_shape must be a DenseI64ArrayAttr";
    return false;
  }
  for (Attribute element : array) {
    auto integer = dyn_cast<IntegerAttr>(element);
    if (!integer || !integer.getType().isInteger(64)) {
      error = "amoeba.logical_transfer_shape entries must be i64 integers";
      return false;
    }
    extents.push_back(integer.getInt());
  }
  return true;
}

static std::optional<func::FuncOp> getOwningFunction(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || !argument.getOwner())
    return std::nullopt;
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front())
    return std::nullopt;
  return function;
}

static ExtentFact computeMemrefBytes(Value root, func::FuncOp function) {
  ExtentFact fact;
  auto type = dyn_cast<MemRefType>(root.getType());
  if (!type) {
    fact.error = "backing root is not a ranked memref";
    return fact;
  }
  if (!type.getLayout().isIdentity()) {
    fact.error = "backing memref has a non-identity layout without a checked "
                 "strided byte bound";
    return fact;
  }

  SmallVector<int64_t> extents;
  if (type.hasStaticShape()) {
    extents.append(type.getShape().begin(), type.getShape().end());
  } else {
    Attribute shapeAttr;
    if (auto argument = dyn_cast<BlockArgument>(root)) {
      if (auto owner = getOwningFunction(root))
        shapeAttr = owner->getArgAttr(argument.getArgNumber(),
                                      kLogicalTransferShapeAttr);
    } else if (Operation *defining = root.getDefiningOp()) {
      shapeAttr = defining->getAttr(kLogicalTransferShapeAttr);
    }
    if (!shapeAttr) {
      fact.error = "dynamic memref has no amoeba.logical_transfer_shape "
                   "extent contract";
      return fact;
    }
    if (!parseShapeAttribute(shapeAttr, extents, fact.error))
      return fact;
  }

  if (extents.size() != static_cast<size_t>(type.getRank())) {
    fact.error = "logical_transfer_shape rank does not match backing memref";
    return fact;
  }

  uint64_t elements = 1;
  for (auto indexed : llvm::enumerate(extents)) {
    int64_t extent = indexed.value();
    if (extent <= 0) {
      fact.error = "memref backing extent must be positive";
      return fact;
    }
    int64_t staticExtent = type.getDimSize(indexed.index());
    if (!ShapedType::isDynamic(staticExtent) && staticExtent != extent) {
      fact.error = "logical_transfer_shape disagrees with a static memref "
                   "dimension";
      return fact;
    }
    if (!safeMultiply(elements, static_cast<uint64_t>(extent), elements)) {
      fact.error = "memref backing element count overflows uint64";
      return fact;
    }
  }

  auto elementBits = scalarBitWidth(type.getElementType());
  if (!elementBits || *elementBits == 0) {
    fact.error = "memref element type has no fixed byte width";
    return fact;
  }
  // Round up sub-byte elements.  Charging a full byte is conservative for
  // packed i1/i4 storage and avoids turning a bit count into a byte proof.
  uint64_t elementBytes = (*elementBits + 7) / 8;
  uint64_t bytes = 0;
  if (!safeMultiply(elements, elementBytes, bytes) ||
      bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    fact.error = "memref backing byte extent overflows int64";
    return fact;
  }
  fact.proven = true;
  fact.bytes = static_cast<int64_t>(bytes);
  return fact;
}

static FailureOr<Value> resolveBackingRoot(Value value,
                                           llvm::DenseSet<Operation *> &visited,
                                           std::string &error) {
  Operation *defining = value.getDefiningOp();
  if (!defining)
    return value;
  if (!visited.insert(defining).second) {
    error = "memref backing resolver found a forwarding cycle";
    return failure();
  }

  if (auto channel = dyn_cast<taskflow::TaskflowChannelOp>(defining))
    return resolveBackingRoot(channel.getSource(), visited, error);
  if (auto join = dyn_cast<taskflow::TaskflowJoinOp>(defining))
    return resolveBackingRoot(join.getBase(), visited, error);
  if (auto cast = dyn_cast<memref::CastOp>(defining))
    return resolveBackingRoot(cast.getSource(), visited, error);
  if (auto subview = dyn_cast<memref::SubViewOp>(defining))
    return resolveBackingRoot(subview.getSource(), visited, error);
  if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(defining))
    return resolveBackingRoot(reinterpret.getSource(), visited, error);

  if (auto task = dyn_cast<taskflow::TaskflowTaskOp>(defining)) {
    auto terminator = dyn_cast<taskflow::TaskflowYieldOp>(
        task.getBody().front().getTerminator());
    if (!terminator) {
      error = "task result has no taskflow.yield terminator";
      return failure();
    }
    auto resolveResult = [&](ValueRange results, ValueRange yielded)
        -> FailureOr<Value> {
      auto found = llvm::find(results, value);
      if (found == results.end())
        return failure();
      unsigned index = static_cast<unsigned>(found - results.begin());
      if (index >= yielded.size())
        return failure();
      Value state = yielded[index];
      auto argument = dyn_cast<BlockArgument>(state);
      unsigned stateInputs = task.getWillReads().size() +
                             task.getWillWrites().size();
      if (!argument || argument.getOwner() != &task.getBody().front() ||
          argument.getArgNumber() >= stateInputs) {
        error = "task result does not resolve to an original memref operand";
        return failure();
      }
      return resolveBackingRoot(task->getOperand(argument.getArgNumber()),
                                visited, error);
    };

    if (auto root = resolveResult(task.getDoneWrites(), terminator.getDoneWrites());
        succeeded(root))
      return *root;
    if (auto root = resolveResult(task.getDoneReads(), terminator.getDoneReads());
        succeeded(root))
      return *root;
    if (error.empty())
      error = "task result is not a classified memref state";
    return failure();
  }

  // Alloc/alloca/get_global and function arguments are storage roots.  Other
  // defining operations are deliberately unsupported: treating an arbitrary
  // alias-producing op as a fresh allocation could undercount bytes.
  if (isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(defining))
    return value;
  error = (Twine("unsupported memref backing producer '") +
           defining->getName().getStringRef() + "'")
              .str();
  return failure();
}

static std::string rootKind(Value root) {
  if (auto argument = dyn_cast<BlockArgument>(root))
    return (Twine("function_argument_") + Twine(argument.getArgNumber())).str();
  if (Operation *defining = root.getDefiningOp())
    return defining->getName().getStringRef().str();
  return "unknown";
}

static std::string backingId(func::FuncOp function, Value root,
                             unsigned sequence) {
  std::string functionName = function.getSymName().str();
  if (functionName.empty())
    functionName = "anonymous";
  if (auto argument = dyn_cast<BlockArgument>(root))
    return (Twine(functionName) + ".arg" + Twine(argument.getArgNumber()))
        .str();
  return (Twine(functionName) + ".backing" + Twine(sequence)).str();
}

static bool parseLocation(Attribute attribute, int64_t rows, int64_t columns,
                          int64_t &cgra, std::string &error) {
  auto dictionary = dyn_cast<DictionaryAttr>(attribute);
  auto row = dictionary ? dictionary.getAs<IntegerAttr>("row")
                        : IntegerAttr();
  auto column = dictionary ? dictionary.getAs<IntegerAttr>("col")
                           : IntegerAttr();
  if (!row || !column) {
    error = "SRAM location must contain integer row and col fields";
    return false;
  }
  if (row.getInt() < 0 || row.getInt() >= rows || column.getInt() < 0 ||
      column.getInt() >= columns) {
    error = "SRAM location is outside the configured fabric bounds";
    return false;
  }
  cgra = row.getInt() * columns + column.getInt();
  return true;
}

static unsigned getOrCreateBacking(ExportState &state, Value root,
                                    ExtentFact extent) {
  auto found = state.backingIndices.find(root);
  if (found != state.backingIndices.end())
    return found->second;
  unsigned index = state.result.backings.size();
  NativeSramBackingRecord record;
  record.rootKind = rootKind(root);
  record.id = backingId(state.function, root, index);
  record.bytes = extent.bytes;
  record.extentProven = extent.proven;
  state.result.backings.push_back(std::move(record));
  state.backingCgras.emplace_back();
  state.backingIndices.try_emplace(root, index);
  return index;
}

static void recordBackingAccess(ExportState &state, Value root,
                                int64_t cgra, StringRef taskName,
                                StringRef direction, unsigned operandIndex) {
  ExtentFact extent = computeMemrefBytes(root, state.function);
  unsigned backing = getOrCreateBacking(state, root, extent);
  NativeSramBackingRecord &record = state.result.backings[backing];
  ++record.observedAccesses;
  ++state.result.observedAccessCount;
  NativeSramAccessRecord access;
  access.task = taskName.str();
  access.direction = direction.str();
  access.operandIndex = operandIndex;
  access.backingId = record.id;
  access.cgra = cgra;
  access.bytes = extent.bytes;
  state.result.accesses.push_back(std::move(access));
  if (!extent.proven) {
    state.result.extentProofComplete = false;
    state.addReason((Twine("task ") + taskName + " " + direction + " operand " +
                     Twine(operandIndex) + " backing " + record.id + ": " +
                     extent.error)
                        .str());
    return;
  }
  if (!state.backingCgras[backing].insert(cgra).second)
    return;
  record.observedCgras.push_back(cgra);
  std::sort(record.observedCgras.begin(), record.observedCgras.end());
  state.chargedBackingCgras.insert({backing, cgra});
  int64_t &required = state.result.requiredBytesByCgra[cgra];
  if (required > std::numeric_limits<int64_t>::max() - extent.bytes) {
    state.addReason((Twine("backing byte sum overflows int64 on CGRA ") +
                     Twine(cgra))
                        .str());
    state.result.extentProofComplete = false;
    return;
  }
  required += extent.bytes;
}

static void recordTaskStorageCoordinate(ExportState &state,
                                        taskflow::TaskflowTaskOp task,
                                        Value root, int64_t cgra,
                                        StringRef direction) {
  auto &coordinates = state.taskStorageCoordinates[task.getOperation()];
  auto &byRoot = direction == "read" ? coordinates.reads : coordinates.writes;
  auto &cgras = byRoot[root];
  if (!llvm::is_contained(cgras, cgra))
    cgras.push_back(cgra);
}

static void processMemoryOperands(ExportState &state,
                                  taskflow::TaskflowTaskOp task,
                                  ValueRange memrefs, ArrayAttr locations,
                                  StringRef direction) {
  StringRef taskName = task.getTaskName();
  if (!locations) {
    state.result.placementProofComplete = false;
    state.addReason((Twine("task ") + taskName + " is missing " + direction +
                     "_sram_locations in task_orchestration_info")
                        .str());
    return;
  }
  if (locations.size() != memrefs.size()) {
    state.result.placementProofComplete = false;
    state.addReason((Twine("task ") + taskName + " " + direction +
                     "_sram_locations count " + Twine(locations.size()) +
                     " does not match original_" + direction + "_memrefs count " +
                     Twine(memrefs.size()))
                        .str());
  }
  unsigned count = std::min<unsigned>(locations.size(), memrefs.size());
  for (unsigned index = 0; index < count; ++index) {
    int64_t cgra = -1;
    std::string locationError;
    if (!parseLocation(locations[index], state.gridRows, state.gridColumns,
                       cgra, locationError)) {
      state.result.placementProofComplete = false;
      state.addReason((Twine("task ") + taskName + " " + direction +
                       "_sram_locations[" + Twine(index) + "]: " +
                       locationError)
                          .str());
      continue;
    }
    llvm::DenseSet<Operation *> visited;
    std::string rootError;
    FailureOr<Value> root =
        resolveBackingRoot(memrefs[index], visited, rootError);
    if (failed(root)) {
      state.result.extentProofComplete = false;
      state.addReason((Twine("task ") + taskName + " " + direction +
                       " original memref[" + Twine(index) + "]: " +
                       (rootError.empty() ? "cannot resolve backing root"
                                           : rootError))
                          .str());
      continue;
    }
    recordTaskStorageCoordinate(state, task, *root, cgra, direction);
    recordBackingAccess(state, *root, cgra, taskName, direction, index);
  }
}

static std::optional<unsigned>
parseKernelInputReference(Attribute attribute) {
  auto text = dyn_cast_or_null<StringAttr>(attribute);
  if (!text)
    return std::nullopt;
  StringRef value = text.getValue();
  if (!value.consume_front("%input") || value.empty())
    return std::nullopt;
  unsigned index = 0;
  if (value.getAsInteger(10, index))
    return std::nullopt;
  return index;
}

// Prove symbolic storage provenance through the value forms emitted by the
// Neura dataflow lowering.  A phi/select is accepted only when every data arm
// names the same kernel input; choosing one arm would undercount a possible
// memory root.  This deliberately does not infer provenance from arbitrary
// operation attributes.
static bool valueReferencesKernelInput(Value value, unsigned inputIndex,
                                       ::mlir::neura::KernelOp kernel,
                                       DenseSet<Value> &visiting,
                                       unsigned depth = 0) {
  if (!value || depth >= 32 || !visiting.insert(value).second)
    return false;
  if (inputIndex >= kernel.getInputs().size()) {
    visiting.erase(value);
    return false;
  }

  bool result = value == kernel.getInputs()[inputIndex];
  if (!result) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      result = argument.getOwner() == &kernel.getBody().front() &&
               argument.getArgNumber() == inputIndex;
    } else if (Operation *definition = value.getDefiningOp()) {
      StringRef name = definition->getName().getStringRef();
      if (name == "neura.constant") {
        if (auto reference =
                parseKernelInputReference(definition->getAttr("value")))
          result = *reference == inputIndex;
      } else if (name == "neura.grant_once" ||
                 name == "neura.grant_always") {
        // Neura's folded grant form keeps the value in constant_value and
        // has no SSA operand.  Require both representations to agree when
        // present; an invalid reference must never fall back to the operand.
        Attribute folded = definition->getAttr("constant_value");
        auto reference = parseKernelInputReference(folded);
        bool hasOperand = definition->getNumOperands() == 1;
        bool operandMatches = hasOperand && valueReferencesKernelInput(
            definition->getOperand(0), inputIndex, kernel, visiting, depth + 1);
        result = definition->getNumOperands() <= 1 &&
                 (folded ? reference && *reference == inputIndex &&
                               (!hasOperand || operandMatches)
                         : operandMatches);
      } else if (name == "neura.data_mov" ||
                 name == "neura.grant_predicate" ||
                 name == "neura.ctrl_mov" || name == "neura.cast" ||
                 name == "neura.phi_start") {
        result = definition->getNumOperands() >= 1 &&
                 valueReferencesKernelInput(definition->getOperand(0),
                                            inputIndex, kernel, visiting,
                                            depth + 1);
      } else if (name == "neura.phi" || name == "neura.sel") {
        unsigned first = name == "neura.sel" ? 1u : 0u;
        result = first < definition->getNumOperands();
        for (unsigned operand = first;
             result && operand < definition->getNumOperands(); ++operand) {
          DenseSet<Value> armVisiting = visiting;
          result = valueReferencesKernelInput(
              definition->getOperand(operand), inputIndex, kernel,
              armVisiting, depth + 1);
        }
      }
    }
  }
  visiting.erase(value);
  return result;
}

static std::optional<unsigned>
traceKernelInput(Value value, ::mlir::neura::KernelOp kernel) {
  std::optional<unsigned> found;
  for (unsigned index = 0; index < kernel.getInputs().size(); ++index) {
    DenseSet<Value> visiting;
    if (!valueReferencesKernelInput(value, index, kernel, visiting))
      continue;
    if (found)
      return std::nullopt;
    found = index;
  }
  return found;
}

static FailureOr<Value>
resolveKernelInputBacking(taskflow::TaskflowTaskOp task,
                         ::mlir::neura::KernelOp kernel, unsigned inputIndex,
                         DenseSet<Operation *> &visited, std::string &error) {
  if (inputIndex >= kernel.getInputs().size()) {
    error = "indexed operation references a kernel input outside kernel.inputs";
    return failure();
  }
  Value input = kernel.getInputs()[inputIndex];
  if (auto argument = dyn_cast<BlockArgument>(input)) {
    if (argument.getOwner() != &task.getBody().front() ||
        argument.getArgNumber() >= task->getNumOperands()) {
      error = "kernel input is not a task body operand";
      return failure();
    }
    input = task->getOperand(argument.getArgNumber());
  }
  if (!isa<MemRefType>(input.getType())) {
    error = "indexed kernel input is not a memref backing";
    return failure();
  }
  FailureOr<Value> root = resolveBackingRoot(input, visited, error);
  if (failed(root))
    return failure();
  if (!isa<MemRefType>((*root).getType())) {
    error = "indexed kernel input resolved to a non-memref backing";
    return failure();
  }
  return root;
}

static void inspectIndexedAccesses(ExportState &state,
                                   taskflow::TaskflowTaskOp task) {
  task.walk([&](Operation *operation) {
    StringRef name = operation->getName().getStringRef();
    bool isLoad = name == "neura.load_indexed";
    bool isStore = name == "neura.store_indexed";
    if (!isLoad && !isStore)
      return;
    if (!state.indexedOperationsSeen.insert(operation).second)
      return;
    if (isLoad)
      ++state.result.indexedLoadCount;
    else
      ++state.result.indexedStoreCount;

    // mapping_locs describe the mapped compute operation, not a byte address
    // in a VectorCGRA SRAM. Only a source-owned address-binding dictionary can
    // establish the physical address range consumed by this access.
    if (operation->getAttr(kStorageAddressBindingAttr))
      ++state.result.indexedAddressProofCount;

    const uint64_t ordinal = state.nextIndexedAccessOrdinal++;
    StringRef direction = isLoad ? StringRef("read") : StringRef("write");
    StringRef referenceAttr = isLoad ? StringRef("lhs_value")
                                     : StringRef("rhs_value");
    NativeSramIndexedAccessRecord record;
    record.task = task.getTaskName().str();
    record.direction = direction.str();
    record.ordinal = ordinal;

    auto failCoverage = [&](StringRef reason) {
      ++state.result.indexedStorageCoverageFailureCount;
      state.addReason((Twine("task ") + task.getTaskName() + " indexed " +
                       direction + " #" + Twine(ordinal) + ": " + reason)
                          .str());
      state.result.indexedAccesses.push_back(std::move(record));
    };

    auto kernel = operation->getParentOfType<::mlir::neura::KernelOp>();
    if (!kernel) {
      failCoverage("operation is outside a neura.kernel");
      return;
    }

    Attribute reference = operation->getAttr(referenceAttr);
    std::optional<unsigned> foldedInput;
    if (reference) {
      if (auto text = dyn_cast<StringAttr>(reference))
        record.symbolicReference = text.getValue().str();
      else
        record.symbolicReference = "<non-string>";
      foldedInput = parseKernelInputReference(reference);
      if (!foldedInput) {
        failCoverage((Twine("invalid ") + referenceAttr +
                      " kernel-input reference")
                         .str());
        return;
      }
    }

    Value base;
    if (auto load = dyn_cast<::mlir::neura::LoadIndexedOp>(operation))
      base = load.getBase();
    else if (auto store = dyn_cast<::mlir::neura::StoreIndexedOp>(operation))
      base = store.getBase();

    std::optional<unsigned> baseInput;
    if (base) {
      baseInput = traceKernelInput(base, kernel);
      if (!baseInput) {
        failCoverage("explicit indexed base has ambiguous or unresolved kernel-input provenance");
        return;
      }
      if (foldedInput && *foldedInput != *baseInput) {
        failCoverage("folded kernel-input reference disagrees with explicit indexed base");
        return;
      }
    }

    std::optional<unsigned> input = foldedInput ? foldedInput : baseInput;
    if (!input) {
      failCoverage("indexed operation has no symbolic or explicit memory base");
      return;
    }
    if (record.symbolicReference.empty())
      record.symbolicReference = (Twine("%input") + Twine(*input)).str();
    if (*input >= kernel.getInputs().size()) {
      failCoverage("kernel-input reference is outside kernel.inputs");
      return;
    }
    record.kernelInput = static_cast<int64_t>(*input);

    DenseSet<Operation *> visited;
    std::string rootError;
    FailureOr<Value> root = resolveKernelInputBacking(
        task, kernel, *input, visited, rootError);
    if (failed(root)) {
      failCoverage(rootError.empty() ? "cannot resolve indexed backing root"
                                     : StringRef(rootError));
      return;
    }
    record.storageRootProven = true;

    auto taskCoordinates = state.taskStorageCoordinates.find(task.getOperation());
    if (taskCoordinates == state.taskStorageCoordinates.end()) {
      failCoverage("task has no scheduler storage-coordinate records");
      return;
    }
    const auto &byRoot = isLoad ? taskCoordinates->second.reads
                                : taskCoordinates->second.writes;
    auto coordinates = byRoot.find(*root);
    if (coordinates == byRoot.end() || coordinates->second.empty()) {
      failCoverage((Twine("indexed backing root is omitted from the task's scheduled ") +
                    direction + " memory operands")
                       .str());
      return;
    }
    auto backing = state.backingIndices.find(*root);
    if (backing == state.backingIndices.end()) {
      failCoverage("indexed backing root was not exported from original task operands");
      return;
    }
    record.backingId = state.result.backings[backing->second].id;
    record.boundCgras.assign(coordinates->second.begin(),
                             coordinates->second.end());
    record.schedulerBindingProven = true;
    ++state.result.indexedStorageCoverageCount;
    state.result.indexedAccesses.push_back(std::move(record));
  });
}

static void compareAliasProof(ExportState &state) {
  state.result.aliasProofComplete = true;
  for (unsigned lhs = 0; lhs < state.result.backings.size(); ++lhs) {
    Value lhsRoot;
    for (const auto &entry : state.backingIndices)
      if (entry.second == lhs) {
        lhsRoot = entry.first;
        break;
      }
    if (!lhsRoot)
      continue;
    for (unsigned rhs = lhs + 1; rhs < state.result.backings.size(); ++rhs) {
      Value rhsRoot;
      for (const auto &entry : state.backingIndices)
        if (entry.second == rhs) {
          rhsRoot = entry.first;
          break;
        }
      if (!rhsRoot || provesDistinctTaskStorage(lhsRoot, rhsRoot))
        continue;
      bool firstUnprovenPair = state.result.aliasProofComplete;
      state.result.aliasProofComplete = false;
      state.result.backings[lhs].aliasIdentityProven = false;
      state.result.backings[rhs].aliasIdentityProven = false;
      if (firstUnprovenPair)
        state.addReason((Twine("missing alias proof between backing ") +
                         state.result.backings[lhs].id + " and " +
                         state.result.backings[rhs].id +
                         "; other potentially-aliasing roots are summarized")
                            .str());
    }
  }
}

static std::string joinReasons(const std::vector<std::string> &reasons) {
  std::string joined;
  for (const std::string &reason : reasons) {
    if (!joined.empty())
      joined += "; ";
    joined += reason;
  }
  return joined;
}

static void exportFunction(ExportState &state) {
  state.result.extentProofComplete = true;
  state.result.placementProofComplete = true;
  state.result.aliasProofComplete = true;
  state.result.copyBindingProofComplete = false;
  state.result.capacityProofComplete = false;
  state.result.indexedAddressProofComplete = true;
  state.result.indexedStorageCoverageComplete = true;

  state.function.walk([&](taskflow::TaskflowTaskOp task) {
    ++state.result.taskCount;
    auto info = task->getAttrOfType<DictionaryAttr>("task_orchestration_info");
    if (!info) {
      state.result.placementProofComplete = false;
      state.addReason((Twine("task ") + task.getTaskName() +
                       " is missing task_orchestration_info")
                          .str());
      // Still inspect indexed references so an otherwise malformed schedule
      // cannot appear to have complete native memory-root coverage.
      inspectIndexedAccesses(state, task);
      return;
    }
    if (task->hasAttr(kStorageResidencyAttr))
      state.sawStorageResidencyContract = true;
    if (task->hasAttr(kStorageCopyBindingAttr))
      state.sawStorageCopyBindingContract = true;
    processMemoryOperands(
        state, task, task.getOriginalReadMemrefs(),
        info.getAs<ArrayAttr>("read_sram_locations"), "read");
    processMemoryOperands(
        state, task, task.getOriginalWriteMemrefs(),
        info.getAs<ArrayAttr>("write_sram_locations"), "write");
    inspectIndexedAccesses(state, task);
  });

  // A native function may contain a kernel outside a taskflow task. Such an
  // access has no scheduler-owned SRAM coordinate and must be diagnosed as
  // uncovered rather than silently omitted from the capacity certificate.
  state.function.walk([&](Operation *operation) {
    StringRef name = operation->getName().getStringRef();
    bool isLoad = name == "neura.load_indexed";
    bool isStore = name == "neura.store_indexed";
    if ((!isLoad && !isStore) ||
        state.indexedOperationsSeen.contains(operation))
      return;
    if (isLoad)
      ++state.result.indexedLoadCount;
    else
      ++state.result.indexedStoreCount;
    NativeSramIndexedAccessRecord record;
    record.task = "<outside-task>";
    record.direction = isLoad ? "read" : "write";
    record.ordinal = state.nextIndexedAccessOrdinal++;
    ++state.result.indexedStorageCoverageFailureCount;
    state.result.indexedAccesses.push_back(std::move(record));
    state.addReason("native indexed memory operation is outside a taskflow "
                    "task and has no scheduler SRAM coordinate");
  });

  compareAliasProof(state);

  uint64_t indexedAccesses = state.result.indexedLoadCount +
                             state.result.indexedStoreCount;
  state.result.indexedStorageCoverageComplete =
      indexedAccesses == state.result.indexedStorageCoverageCount &&
      state.result.indexedStorageCoverageFailureCount == 0;
  if (!state.result.indexedStorageCoverageComplete)
    state.addReason(
        "native indexed memory references are not all covered by original "
        "task memory roots and scheduler SRAM coordinates");
  state.result.indexedAddressProofComplete =
      indexedAccesses == state.result.indexedAddressProofCount;
  if (indexedAccesses != 0 && !state.result.indexedAddressProofComplete)
    state.addReason(
        "native neura.load_indexed/store_indexed operations have no source-verified physical SRAM address binding");

  // The native scheduler's coordinate assignment is not a proof that a
  // buffer stayed resident or that a copy was bound to one physical SRAM.
  // Keep all byte records, but leave the status pending until a storage
  // residency/copy contract is exported by a source-owned pass.
  if (!state.sawStorageResidencyContract)
    state.addReason("missing source-verified storage_residency intervals");
  if (!state.sawStorageCopyBindingContract)
    state.addReason("missing source-verified storage copy_binding records");
  state.addReason("per-CGRA SRAM byte capacity is not supplied by the native "
                  "artifact");

  state.result.status = "pending";
  state.result.reason = joinReasons(state.reasons);
}

static ArrayAttr makeI64Array(MLIRContext *context,
                              const std::map<int64_t, int64_t> &values,
                              int64_t count) {
  SmallVector<Attribute> attrs;
  attrs.reserve(static_cast<size_t>(std::max<int64_t>(0, count)));
  for (int64_t index = 0; index < count; ++index) {
    auto found = values.find(index);
    int64_t value = found == values.end() ? 0 : found->second;
    attrs.push_back(IntegerAttr::get(IntegerType::get(context, 64), value));
  }
  return ArrayAttr::get(context, attrs);
}

static ArrayAttr makeBackingRecords(MLIRContext *context,
                                    ArrayRef<NativeSramBackingRecord> records) {
  Builder builder(context);
  SmallVector<Attribute> attrs;
  attrs.reserve(records.size());
  for (const NativeSramBackingRecord &record : records) {
    SmallVector<NamedAttribute> fields;
    fields.push_back(builder.getNamedAttr("id",
                                         builder.getStringAttr(record.id)));
    fields.push_back(builder.getNamedAttr(
        "root_kind", builder.getStringAttr(record.rootKind)));
    fields.push_back(builder.getNamedAttr(
        "bytes", builder.getI64IntegerAttr(record.bytes)));
    fields.push_back(builder.getNamedAttr(
        "extent_proven", builder.getBoolAttr(record.extentProven)));
    fields.push_back(builder.getNamedAttr(
        "alias_identity_proven",
        builder.getBoolAttr(record.aliasIdentityProven)));
    fields.push_back(builder.getNamedAttr(
        "copy_binding_proven", builder.getBoolAttr(record.copyBindingProven)));
    fields.push_back(builder.getNamedAttr(
        "observed_accesses",
        builder.getI64IntegerAttr(static_cast<int64_t>(record.observedAccesses))));
    SmallVector<Attribute> cgras;
    for (int64_t cgra : record.observedCgras)
      cgras.push_back(builder.getI64IntegerAttr(cgra));
    fields.push_back(builder.getNamedAttr("observed_cgras",
                                         builder.getArrayAttr(cgras)));
    attrs.push_back(builder.getDictionaryAttr(fields));
  }
  return builder.getArrayAttr(attrs);
}

static ArrayAttr makeAccessRecords(MLIRContext *context,
                                   ArrayRef<NativeSramAccessRecord> accesses) {
  Builder builder(context);
  SmallVector<Attribute> attrs;
  attrs.reserve(accesses.size());
  for (const NativeSramAccessRecord &access : accesses) {
    attrs.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("task", builder.getStringAttr(access.task)),
        builder.getNamedAttr("direction",
                            builder.getStringAttr(access.direction)),
        builder.getNamedAttr(
            "operand_index",
            builder.getI64IntegerAttr(static_cast<int64_t>(access.operandIndex))),
        builder.getNamedAttr("backing_id",
                            builder.getStringAttr(access.backingId)),
        builder.getNamedAttr("cgra", builder.getI64IntegerAttr(access.cgra)),
        builder.getNamedAttr("bytes", builder.getI64IntegerAttr(access.bytes)),
    }));
  }
  return builder.getArrayAttr(attrs);
}

static ArrayAttr makeIndexedAccessRecords(
    MLIRContext *context,
    ArrayRef<NativeSramIndexedAccessRecord> accesses) {
  Builder builder(context);
  SmallVector<Attribute> attrs;
  attrs.reserve(accesses.size());
  for (const NativeSramIndexedAccessRecord &access : accesses) {
    SmallVector<NamedAttribute> fields;
    fields.push_back(builder.getNamedAttr(
        "task", builder.getStringAttr(access.task)));
    fields.push_back(builder.getNamedAttr(
        "direction", builder.getStringAttr(access.direction)));
    fields.push_back(builder.getNamedAttr(
        "ordinal", builder.getI64IntegerAttr(
                        static_cast<int64_t>(access.ordinal))));
    fields.push_back(builder.getNamedAttr(
        "kernel_input", builder.getI64IntegerAttr(access.kernelInput)));
    fields.push_back(builder.getNamedAttr(
        "symbolic_reference",
        builder.getStringAttr(access.symbolicReference)));
    fields.push_back(builder.getNamedAttr(
        "backing_id", builder.getStringAttr(access.backingId)));
    fields.push_back(builder.getNamedAttr(
        "storage_root_proven",
        builder.getBoolAttr(access.storageRootProven)));
    fields.push_back(builder.getNamedAttr(
        "scheduler_binding_proven",
        builder.getBoolAttr(access.schedulerBindingProven)));
    SmallVector<Attribute> cgras;
    for (int64_t cgra : access.boundCgras)
      cgras.push_back(builder.getI64IntegerAttr(cgra));
    fields.push_back(builder.getNamedAttr("bound_cgras",
                                         builder.getArrayAttr(cgras)));
    attrs.push_back(builder.getDictionaryAttr(fields));
  }
  return builder.getArrayAttr(attrs);
}

struct ExportProductionSramNativeUpperBoundPass
    : public PassWrapper<ExportProductionSramNativeUpperBoundPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ExportProductionSramNativeUpperBoundPass)

  ExportProductionSramNativeUpperBoundPass() = default;
  ExportProductionSramNativeUpperBoundPass(
      const ExportProductionSramNativeUpperBoundPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "export-production-sram-native-upper-bound";
  }
  StringRef getDescription() const override {
    return "Export conservative native Taskflow SRAM byte requirements";
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Optional func.func symbol to inspect"),
      llvm::cl::init("")};
  Option<int64_t> gridRows{
      *this, "grid-rows", llvm::cl::desc("Fabric row count"),
      llvm::cl::init(4)};
  Option<int64_t> gridColumns{
      *this, "grid-columns", llvm::cl::desc("Fabric column count"),
      llvm::cl::init(4)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    NativeSramUpperBoundResult aggregate;
    aggregate.status = "pending";
    aggregate.extentProofComplete = true;
    aggregate.placementProofComplete = true;
    aggregate.aliasProofComplete = true;
    aggregate.copyBindingProofComplete = false;
    aggregate.capacityProofComplete = false;
    aggregate.indexedAddressProofComplete = true;
    aggregate.indexedStorageCoverageComplete = true;
    std::vector<std::string> reasons;
    bool foundFunction = false;

    auto merge = [&](func::FuncOp function) {
      foundFunction = true;
      NativeSramUpperBoundResult current =
          exportNativeSramUpperBound(function, gridRows.getValue(),
                                     gridColumns.getValue());
      aggregate.taskCount += current.taskCount;
      aggregate.observedAccessCount += current.observedAccessCount;
      aggregate.extentProofComplete &= current.extentProofComplete;
      aggregate.placementProofComplete &= current.placementProofComplete;
      aggregate.aliasProofComplete &= current.aliasProofComplete;
      aggregate.copyBindingProofComplete &= current.copyBindingProofComplete;
      aggregate.indexedLoadCount += current.indexedLoadCount;
      aggregate.indexedStoreCount += current.indexedStoreCount;
      aggregate.indexedAddressProofCount += current.indexedAddressProofCount;
      aggregate.indexedAddressProofComplete &=
          current.indexedAddressProofComplete;
      aggregate.indexedStorageCoverageCount +=
          current.indexedStorageCoverageCount;
      aggregate.indexedStorageCoverageFailureCount +=
          current.indexedStorageCoverageFailureCount;
      aggregate.indexedStorageCoverageComplete &=
          current.indexedStorageCoverageComplete;
      for (const auto &entry : current.requiredBytesByCgra) {
        int64_t &value = aggregate.requiredBytesByCgra[entry.first];
        if (value > std::numeric_limits<int64_t>::max() - entry.second)
          reasons.push_back("module SRAM byte requirement overflows int64");
        else
          value += entry.second;
      }
      aggregate.backings.insert(aggregate.backings.end(),
                               current.backings.begin(), current.backings.end());
      aggregate.accesses.insert(aggregate.accesses.end(),
                               current.accesses.begin(), current.accesses.end());
      aggregate.indexedAccesses.insert(
          aggregate.indexedAccesses.end(), current.indexedAccesses.begin(),
          current.indexedAccesses.end());
      if (!current.reason.empty())
        reasons.push_back((Twine(function.getSymName()) + ": " +
                           current.reason)
                              .str());
    };

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (!functionName.getValue().empty() &&
          function.getSymName() != functionName.getValue())
        continue;
      bool hasTask = false;
      function.walk([&](taskflow::TaskflowTaskOp) { hasTask = true; });
      if (hasTask)
        merge(function);
    }
    if (!foundFunction)
      reasons.push_back(functionName.getValue().empty()
                            ? "module has no func.func with taskflow tasks"
                            : (Twine("requested function '") +
                               functionName.getValue() +
                               "' has no taskflow tasks")
                                  .str());
    if (!foundFunction)
      aggregate.indexedStorageCoverageComplete = false;
    aggregate.reason = joinReasons(reasons);
    module->setAttr("amoeba.joint_scheduling_sram_conservative_upper_bound_status",
                    StringAttr::get(context, "pending"));
    module->setAttr("amoeba.joint_scheduling_sram_conservative_upper_bound_reason",
                    StringAttr::get(context, aggregate.reason));
    module->setAttr(
        "amoeba.joint_scheduling_sram_conservative_upper_bound_policy",
        StringAttr::get(
            context,
            "whole_schedule_no_eviction_full_backing_per_observed_sram_location"));
    int64_t cgraCount =
        gridRows > 0 && gridColumns > 0 &&
                gridRows <= std::numeric_limits<int64_t>::max() / gridColumns
            ? gridRows * gridColumns
            : 0;
    if (cgraCount > 0)
      module->setAttr("amoeba.joint_scheduling_sram_conservative_required_bytes_by_cgra",
                      makeI64Array(context, aggregate.requiredBytesByCgra,
                                   cgraCount));
    module->setAttr("amoeba.joint_scheduling_sram_conservative_backings",
                    makeBackingRecords(context, aggregate.backings));
    module->setAttr("amoeba.joint_scheduling_sram_conservative_accesses",
                    makeAccessRecords(context, aggregate.accesses));
    module->setAttr("amoeba.joint_scheduling_sram_conservative_extent_proven",
                    BoolAttr::get(context, aggregate.extentProofComplete));
    module->setAttr("amoeba.joint_scheduling_sram_conservative_placement_proven",
                    BoolAttr::get(context, aggregate.placementProofComplete));
    module->setAttr("amoeba.joint_scheduling_sram_conservative_alias_proven",
                    BoolAttr::get(context, aggregate.aliasProofComplete));
    module->setAttr("amoeba.joint_scheduling_sram_conservative_copy_binding_proven",
                    BoolAttr::get(context, aggregate.copyBindingProofComplete));
    module->setAttr("amoeba.joint_scheduling_sram_conservative_task_count",
                    IntegerAttr::get(IntegerType::get(context, 64),
                                     aggregate.taskCount));
    module->setAttr("amoeba.joint_scheduling_sram_conservative_observed_access_count",
                    IntegerAttr::get(IntegerType::get(context, 64),
                                     aggregate.observedAccessCount));
    module->setAttr("amoeba.joint_scheduling_sram_indexed_load_count",
                    IntegerAttr::get(IntegerType::get(context, 64),
                                     aggregate.indexedLoadCount));
    module->setAttr("amoeba.joint_scheduling_sram_indexed_store_count",
                    IntegerAttr::get(IntegerType::get(context, 64),
                                     aggregate.indexedStoreCount));
    module->setAttr("amoeba.joint_scheduling_sram_indexed_address_proof_count",
                    IntegerAttr::get(IntegerType::get(context, 64),
                                     aggregate.indexedAddressProofCount));
    module->setAttr("amoeba.joint_scheduling_sram_indexed_address_proven",
                    BoolAttr::get(context, aggregate.indexedAddressProofComplete));
    module->setAttr(
        "amoeba.joint_scheduling_sram_indexed_storage_coverage_count",
        IntegerAttr::get(IntegerType::get(context, 64),
                         aggregate.indexedStorageCoverageCount));
    module->setAttr(
        "amoeba.joint_scheduling_sram_indexed_storage_coverage_failures",
        IntegerAttr::get(IntegerType::get(context, 64),
                         aggregate.indexedStorageCoverageFailureCount));
    module->setAttr(
        "amoeba.joint_scheduling_sram_indexed_storage_coverage_proven",
        BoolAttr::get(context, aggregate.indexedStorageCoverageComplete));
    module->setAttr("amoeba.joint_scheduling_sram_indexed_accesses",
                    makeIndexedAccessRecords(context,
                                             aggregate.indexedAccesses));
  }
};

} // namespace

NativeSramUpperBoundResult exportNativeSramUpperBound(
    func::FuncOp function, int64_t gridRows, int64_t gridColumns) {
  ExportState state;
  state.function = function;
  state.gridRows = gridRows;
  state.gridColumns = gridColumns;
  state.result.extentProofComplete = true;
  state.result.placementProofComplete = true;
  state.result.aliasProofComplete = true;
  state.result.indexedAddressProofComplete = false;
  if (!function) {
    state.result.reason = "native SRAM exporter requires a func.func";
    return state.result;
  }
  if (gridRows <= 0 || gridColumns <= 0) {
    state.result.placementProofComplete = false;
    state.addReason("native SRAM exporter requires positive fabric dimensions");
    state.result.status = "pending";
    state.result.reason = joinReasons(state.reasons);
    return state.result;
  }
  if (gridRows > std::numeric_limits<int64_t>::max() / gridColumns) {
    state.result.placementProofComplete = false;
    state.addReason("native SRAM exporter fabric dimensions overflow int64");
    state.result.status = "pending";
    state.result.reason = joinReasons(state.reasons);
    return state.result;
  }
  exportFunction(state);
  return state.result;
}

std::unique_ptr<Pass> createExportProductionSramUpperBoundPass() {
  return std::make_unique<ExportProductionSramNativeUpperBoundPass>();
}

} // namespace mlir::amoeba::neura::joint_scheduling
