#ifndef AMOEBA_REPLICA_OUTPUT_COORDINATE_PROOF_H
#define AMOEBA_REPLICA_OUTPUT_COORDINATE_PROOF_H

// Shared proof for logical output-coordinate permutations.  It derives the
// mapping from the Taskflow and Neura IR itself.  Replica metadata is checked
// against the proof only after this derivation; it is never used as the proof
// source.

#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace mlir::amoeba::neura::joint_scheduling {

using namespace mlir::taskflow;

struct ReplicaOutputCoordinateProof {
  // A constant output dimension has no counter route.  Keep an explicit
  // sentinel in the legacy counter-map vectors so existing callers can still
  // compare the complete rank-sized map; outputConstantAxes carries the
  // authenticated value for those sentinel entries.
  static constexpr unsigned kConstantAxis = std::numeric_limits<unsigned>::max();

  bool proven = false;
  bool selectedAxisIndependent = false;
  bool sawOutputLoad = false;
  bool sawOutputStore = false;
  unsigned outputInput = std::numeric_limits<unsigned>::max();
  SmallVector<unsigned> outputCounterAxes;
  SmallVector<SmallVector<unsigned>> outputAccessCounterAxes;
  SmallVector<std::optional<int64_t>> outputConstantAxes;
  SmallVector<SmallVector<std::optional<int64_t>>>
      outputAccessConstantAxes;
  SmallVector<unsigned> auxiliaryInputIndices;
  SmallVector<int64_t> taskLowers;
  SmallVector<int64_t> taskUppers;
  SmallVector<int64_t> kernelLowers;
  SmallVector<int64_t> kernelUppers;
  SmallVector<int64_t> producedLowers;
  SmallVector<int64_t> producedShape;
  SmallVector<int64_t> callerShape;
  std::string reason;
};

namespace detail {

inline void proofFail(ReplicaOutputCoordinateProof &proof,
                      llvm::StringRef reason) {
  if (proof.reason.empty())
    proof.reason = reason.str();
  proof.proven = false;
}

inline std::optional<unsigned> parseInputReference(mlir::Attribute attribute) {
  auto text = mlir::dyn_cast_or_null<mlir::StringAttr>(attribute);
  if (!text || !text.getValue().starts_with("%input"))
    return std::nullopt;
  llvm::StringRef suffix = text.getValue().drop_front(6);
  unsigned value = 0;
  if (suffix.empty() || suffix.getAsInteger(10, value))
    return std::nullopt;
  return value;
}

inline bool fitsSignedWidth(int64_t value, uint64_t width) {
  if (width == 0 || width > 64)
    return false;
  if (width == 64)
    return true;
  int64_t maximum = (int64_t(1) << (width - 1)) - 1;
  int64_t minimum = -maximum - 1;
  return value >= minimum && value <= maximum;
}

inline bool fitsUnsignedWidth(int64_t value, uint64_t width) {
  if (value < 0 || width == 0 || width > 64)
    return false;
  if (width == 64)
    return true;
  return static_cast<uint64_t>(value) <= ((uint64_t(1) << width) - 1);
}

inline std::optional<int64_t> integerAttributeValue(IntegerAttr attribute,
                                                     Type type) {
  const llvm::APInt &value = attribute.getValue();
  if (isa<IndexType>(type)) {
    if (!value.isSignedIntN(64))
      return std::nullopt;
    return value.getSExtValue();
  }
  auto integerType = dyn_cast<IntegerType>(type);
  if (!integerType || value.getBitWidth() > 64)
    return std::nullopt;
  if (integerType.isSigned()) {
    if (!value.isSignedIntN(64))
      return std::nullopt;
    return value.getSExtValue();
  }
  if (integerType.isUnsigned()) {
    // This proof stores coordinates in int64, so larger unsigned literals
    // cannot be represented without changing its domain.
    if (!value.isIntN(63))
      return std::nullopt;
    return static_cast<int64_t>(value.getZExtValue());
  }
  // Keep the existing signed interpretation for signless literal attributes,
  // but reject a value that does not fit the proof's int64 domain.
  if (!value.isSignedIntN(64))
    return std::nullopt;
  return value.getSExtValue();
}

inline std::optional<int64_t>
input0StaticIndexBound(mlir::func::FuncOp function,
                       mlir::BlockArgument argument) {
  using namespace mlir;
  if (!function || argument.getOwner() != &function.getBody().front() ||
      argument.getArgNumber() >= function.getNumArguments() ||
      !isa<IndexType>(argument.getType()) ||
      !function->getAttrOfType<UnitAttr>("amoeba.input0_caller_noalias_proven"))
    return std::nullopt;

  auto callerBound =
      function->getAttrOfType<IntegerAttr>("amoeba.input0_caller_static_bound");
  auto caller =
      function->getAttrOfType<StringAttr>("amoeba.input0_caller_function");
  auto evidence = function->getAttrOfType<StringAttr>(
      "amoeba.input0_caller_evidence_path");
  auto proofMode = function->getAttrOfType<StringAttr>(
      "amoeba.input0_caller_noalias_proof_mode");
  std::string boundName = "amoeba.static_bound.arg." +
                          std::to_string(argument.getArgNumber());
  auto sourceBound = function->getAttrOfType<IntegerAttr>(boundName);
  if (!callerBound || !caller || caller.getValue().empty() || !evidence ||
      evidence.getValue().empty() || !proofMode ||
      (proofMode.getValue() != "in-module-caller" &&
       proofMode.getValue() != "prepared-external-caller") ||
      !sourceBound || callerBound.getInt() <= 0 ||
      sourceBound.getInt() != callerBound.getInt())
    return std::nullopt;

  DataLayout layout = DataLayout::closest(function);
  std::optional<uint64_t> width =
      layout.getTypeIndexBitwidth(IndexType::get(function.getContext()));
  if (!width || !fitsSignedWidth(callerBound.getInt(), *width))
    return std::nullopt;
  return callerBound.getInt();
}

inline std::optional<int64_t> checkedStaticIntegerBinary(
    mlir::Operation *operation, int64_t lhs, int64_t rhs, bool subtract) {
  using namespace mlir;
  if (!operation || operation->getNumOperands() != 2 ||
      operation->getNumResults() != 1)
    return std::nullopt;
  Type type = operation->getResult(0).getType();
  if (operation->getOperand(0).getType() != type ||
      operation->getOperand(1).getType() != type)
    return std::nullopt;

  uint64_t width = 0;
  bool signedDomain = false;
  if (isa<IndexType>(type)) {
    DataLayout layout = DataLayout::closest(operation);
    std::optional<uint64_t> indexWidth = layout.getTypeIndexBitwidth(type);
    if (!indexWidth)
      return std::nullopt;
    width = *indexWidth;
    signedDomain = true;
  } else if (auto integerType = dyn_cast<IntegerType>(type)) {
    width = integerType.getWidth();
    if (integerType.isSigned())
      signedDomain = true;
    else if (!integerType.isUnsigned())
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  auto fits = [&](int64_t value) {
    return signedDomain ? fitsSignedWidth(value, width)
                        : fitsUnsignedWidth(value, width);
  };
  if (!fits(lhs) || !fits(rhs))
    return std::nullopt;
  int64_t result = 0;
  if (subtract ? __builtin_sub_overflow(lhs, rhs, &result)
               : __builtin_add_overflow(lhs, rhs, &result))
    return std::nullopt;
  if (!fits(result))
    return std::nullopt;
  return result;
}

inline std::optional<int64_t>
staticIndex(mlir::Value value, mlir::taskflow::TaskflowTaskOp task,
            mlir::neura::KernelOp kernel,
            llvm::SmallDenseSet<mlir::Value, 16> &visiting) {
  using namespace mlir;
  if (!value || !visiting.insert(value).second)
    return std::nullopt;
  // Keep this set as an active recursion stack. A global visited set would
  // reject valid repeated operands such as `%x + %x` after the lhs walk.
  auto eraseActive = llvm::make_scope_exit([&] { visiting.erase(value); });
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integerAttributeValue(integer, constant.getType());
  if (auto constant = value.getDefiningOp<::mlir::neura::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant->getAttr("value"))) {
      if (!integer.getValue().isSignedIntN(64))
        return std::nullopt;
      return integer.getValue().getSExtValue();
    }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *parent = argument.getOwner()->getParentOp();
    if (kernel && argument.getOwner() == &kernel.getBody().front() &&
        argument.getArgNumber() < kernel.getInputs().size())
      return staticIndex(kernel.getInputs()[argument.getArgNumber()], task,
                         kernel, visiting);
    if (task && argument.getOwner() == &task.getBody().front() &&
        argument.getArgNumber() < task->getNumOperands())
      return staticIndex(task->getOperand(argument.getArgNumber()), task,
                         kernel, visiting);
    if (auto function = dyn_cast_or_null<func::FuncOp>(parent)) {
      if (argument.getOwner() != &function.getBody().front())
        return std::nullopt;
      return input0StaticIndexBound(function, argument);
    }
    return std::nullopt;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
    // A literal integer to index conversion is exact for the static bound
    // contract.  Do not extend this allowance to a dynamic SSA cast (or an
    // index-to-narrow-integer cast): those require width/signedness/range
    // proof and must remain unknown.
    if (!cast.getType().isIndex() ||
        !cast.getIn().getDefiningOp<arith::ConstantOp>())
      return std::nullopt;
    return staticIndex(cast.getIn(), task, kernel, visiting);
  }
  // Predicate/data forwarding preserves a literal coordinate.  Follow only
  // the same transparent routes accepted for authenticated counter indices;
  // arithmetic and opaque selects remain unknown.
  StringRef name = definition->getName().getStringRef();
  if ((name == "neura.data_mov" || name == "neura.grant_predicate") &&
      definition->getNumOperands() >= 1)
    return staticIndex(definition->getOperand(0), task, kernel, visiting);
  if (name == "neura.phi" && definition->getNumOperands() > 0) {
    // A phi has independent incoming paths.  Keep cycle detection local to
    // each path so two alternatives that forward the same literal (through
    // distinct grant_predicate operations) still authenticate as one value.
    llvm::SmallDenseSet<Value, 16> firstVisiting = visiting;
    auto first = staticIndex(definition->getOperand(0), task, kernel,
                             firstVisiting);
    if (!first)
      return std::nullopt;
    for (Value operand : definition->getOperands().drop_front()) {
      llvm::SmallDenseSet<Value, 16> branchVisiting = visiting;
      auto candidate = staticIndex(operand, task, kernel, branchVisiting);
      if (!candidate || *candidate != *first)
        return std::nullopt;
    }
    return first;
  }
  // Arithmetic is accepted only from two independently resolved SSA values
  // with identical operation/result types. The operation's actual index
  // width (or explicit signed/unsigned integer width) must represent both
  // operands and the mathematical result, so modular wrap and narrow casts
  // cannot enter this proof.
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = staticIndex(add.getLhs(), task, kernel, visiting);
    auto rhs = staticIndex(add.getRhs(), task, kernel, visiting);
    if (!lhs || !rhs)
      return std::nullopt;
    return checkedStaticIntegerBinary(add, *lhs, *rhs, false);
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    auto lhs = staticIndex(sub.getLhs(), task, kernel, visiting);
    auto rhs = staticIndex(sub.getRhs(), task, kernel, visiting);
    if (!lhs || !rhs)
      return std::nullopt;
    return checkedStaticIntegerBinary(sub, *lhs, *rhs, true);
  }
  if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
    SmallVector<Attribute> constants;
    constants.reserve(apply->getNumOperands());
    for (Value operand : apply->getOperands()) {
      auto constant = staticIndex(operand, task, kernel, visiting);
      if (!constant)
        return std::nullopt;
      constants.push_back(IntegerAttr::get(IndexType::get(value.getContext()),
                                           *constant));
    }
    SmallVector<Attribute> folded;
    if (failed(apply.getAffineMap().constantFold(constants, folded)) ||
        folded.size() != 1)
      return std::nullopt;
    auto integer = dyn_cast<IntegerAttr>(folded.front());
    if (!integer || !integer.getValue().isSignedIntN(64))
      return std::nullopt;
    return integer.getValue().getSExtValue();
  }
  return std::nullopt;
}

inline std::optional<int64_t>
staticIndex(mlir::Value value, mlir::taskflow::TaskflowTaskOp task,
            mlir::neura::KernelOp kernel = {}) {
  llvm::SmallDenseSet<mlir::Value, 16> visiting;
  return staticIndex(value, task, kernel, visiting);
}

inline std::optional<int64_t>
staticCounterAttribute(mlir::Operation *operation, llvm::StringRef name,
                       mlir::Value fallback,
                       mlir::taskflow::TaskflowTaskOp task,
                       mlir::neura::KernelOp kernel) {
  using namespace mlir;
  std::optional<int64_t> attributeValue;
  if (Attribute attribute = operation->getAttr(name)) {
    if (auto integer = dyn_cast<IntegerAttr>(attribute)) {
      attributeValue = integer.getInt();
    } else {
      auto input = parseInputReference(attribute);
      if (!input || *input >= kernel.getInputs().size())
        return std::nullopt;
      attributeValue =
          staticIndex(kernel.getInputs()[*input], task, kernel);
      if (!attributeValue)
        return std::nullopt;
    }
  }

  std::optional<int64_t> operandValue;
  if (fallback) {
    operandValue = staticIndex(fallback, task, kernel);
    // A present SSA bound that cannot be resolved is not made trustworthy by
    // a matching-looking folded attribute.
    if (!operandValue)
      return std::nullopt;
  }
  if (attributeValue && operandValue && *attributeValue != *operandValue)
    return std::nullopt;
  if (attributeValue)
    return attributeValue;
  return operandValue;
}

inline std::optional<unsigned>
resolveKernelInput(mlir::Value value, mlir::neura::KernelOp kernel,
                  unsigned depth = 0) {
  using namespace mlir;
  if (!value || depth >= 32)
    return std::nullopt;
  if (auto argument = dyn_cast<BlockArgument>(value))
    if (argument.getOwner() == &kernel.getBody().front() &&
        argument.getArgNumber() < kernel.getInputs().size())
      return argument.getArgNumber();
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  // Only the Neura constant op owns the semantic `value` attribute.  An
  // arbitrary dataflow op may carry a same-named annotation, but that is not
  // an authenticated kernel-input route.
  if (name == "neura.constant")
    if (auto input = parseInputReference(definition->getAttr("value")))
      return *input < kernel.getInputs().size() ? input : std::nullopt;
  if ((name == "neura.data_mov" || name == "neura.grant_predicate") &&
      definition->getNumOperands() >= 1)
    return resolveKernelInput(definition->getOperand(0), kernel, depth + 1);
  return std::nullopt;
}

inline std::optional<unsigned>
accessInput(mlir::Operation *operation, mlir::Value base,
            mlir::neura::KernelOp kernel) {
  bool isLoad = isa<::mlir::neura::LoadIndexedOp>(operation);
  bool isStore = isa<::mlir::neura::StoreIndexedOp>(operation);
  if (isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(operation))
    return resolveKernelInput(base, kernel);
  if (!isLoad && !isStore)
    return std::nullopt;

  // Folded memory-base references have one canonical slot per indexed op:
  // lhs_value for loads and rhs_value for stores.  The opposite slot may be
  // a folded data value (not a base), but it must still be syntactically and
  // numerically valid when present; malformed or out-of-range references are
  // never ignored.
  StringRef expectedName = isLoad ? StringRef("lhs_value")
                                 : StringRef("rhs_value");
  StringRef oppositeName = isLoad ? StringRef("rhs_value")
                                  : StringRef("lhs_value");
  auto parseFolded = [&](StringRef name) -> std::optional<unsigned> {
    Attribute attribute = operation->getAttr(name);
    if (!attribute)
      return std::nullopt;
    auto input = parseInputReference(attribute);
    if (!input || *input >= kernel.getInputs().size())
      return std::nullopt;
    return input;
  };
  Attribute expectedAttribute = operation->getAttr(expectedName);
  Attribute oppositeAttribute = operation->getAttr(oppositeName);
  std::optional<unsigned> expected;
  if (expectedAttribute) {
    expected = parseFolded(expectedName);
    if (!expected)
      return std::nullopt;
  }
  if (oppositeAttribute && !parseFolded(oppositeName))
    return std::nullopt;

  std::optional<unsigned> actual;
  if (base) {
    actual = resolveKernelInput(base, kernel);
    if (!actual)
      return std::nullopt;
  }
  if (actual && expected && *actual != *expected)
    return std::nullopt;
  if (actual)
    return actual;
  return expected;
}

inline std::optional<unsigned>
counterRoute(mlir::Value value,
             llvm::ArrayRef<mlir::neura::CounterOp> counters,
             unsigned depth = 0) {
  using namespace mlir;
  if (!value || depth >= 32)
    return std::nullopt;
  for (auto [index, counterRef] : llvm::enumerate(counters)) {
    ::mlir::neura::CounterOp counter = counterRef;
    if (value == counter.getCurrentIndex())
      return static_cast<unsigned>(index);
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if ((name == "neura.data_mov" || name == "neura.grant_predicate") &&
      definition->getNumOperands() >= 1)
    return counterRoute(definition->getOperand(0), counters, depth + 1);
  if (name == "neura.phi" && definition->getNumOperands() > 0) {
    auto first = counterRoute(definition->getOperand(0), counters, depth + 1);
    if (!first)
      return std::nullopt;
    for (Value operand : definition->getOperands().drop_front()) {
      auto candidate = counterRoute(operand, counters, depth + 1);
      if (!candidate || *candidate != *first)
        return std::nullopt;
    }
    return first;
  }
  // Casts, constants, selects, reserve/ctrl feedback, and opaque arithmetic
  // are deliberately not accepted as coordinate proofs.
  return std::nullopt;
}

inline bool isNoAliasFunctionArgument(mlir::Value value) {
  using namespace mlir;
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || !argument.getOwner())
    return false;
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  return function && argument.getOwner() == &function.getBody().front() &&
         function.getArgAttr(argument.getArgNumber(), "amoeba.noalias");
}

inline bool provesDistinctStorage(mlir::Value lhs, mlir::Value rhs) {
  using namespace mlir;
  if (!lhs || !rhs || lhs == rhs)
    return false;
  Operation *lhsDefinition = lhs.getDefiningOp();
  Operation *rhsDefinition = rhs.getDefiningOp();
  bool lhsAllocation = lhsDefinition &&
                       isa<memref::AllocOp, memref::AllocaOp>(lhsDefinition);
  bool rhsAllocation = rhsDefinition &&
                       isa<memref::AllocOp, memref::AllocaOp>(rhsDefinition);
  if (lhsAllocation && rhsAllocation)
    return lhsDefinition != rhsDefinition;
  if (lhsAllocation && isNoAliasFunctionArgument(rhs))
    return true;
  if (rhsAllocation && isNoAliasFunctionArgument(lhs))
    return true;
  return isNoAliasFunctionArgument(lhs) && isNoAliasFunctionArgument(rhs);
}

inline std::optional<mlir::Value>
taskRootForKernelInput(unsigned input, mlir::neura::KernelOp kernel,
                       mlir::taskflow::TaskflowTaskOp task) {
  using namespace mlir;
  if (input >= kernel.getInputs().size() || !task.getBody().hasOneBlock())
    return std::nullopt;
  Value value = kernel.getInputs()[input];
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() != &task.getBody().front() ||
        argument.getArgNumber() >= task->getNumOperands())
      return std::nullopt;
    unsigned index = argument.getArgNumber();
    if (index < task.getWillReads().size() &&
        index < task.getOriginalReadMemrefs().size())
      return task.getOriginalReadMemrefs()[index];
    index -= task.getWillReads().size();
    if (index < task.getWillWrites().size() &&
        index < task.getOriginalWriteMemrefs().size())
      return task.getOriginalWriteMemrefs()[index];
    return std::nullopt;
  }
  for (Value root : task.getOriginalReadMemrefs())
    if (root == value)
      return root;
  for (Value root : task.getOriginalWriteMemrefs())
    if (root == value)
      return root;
  return std::nullopt;
}

inline std::optional<llvm::SmallVector<int64_t>>
callerShape(mlir::Value value, unsigned depth = 0) {
  using namespace mlir;
  if (!value || depth >= 64)
    return std::nullopt;
  // Static views and verified completion-state forwarding preserve storage;
  // obtain its capacity from the original allocation or function argument,
  // never from a cast that merely asserts a static view of a dynamic value.
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return callerShape(cast.getSource(), depth + 1);
  if (auto channel = value.getDefiningOp<taskflow::TaskflowChannelOp>())
    return callerShape(channel.getSource(), depth + 1);
  if (auto join = value.getDefiningOp<taskflow::TaskflowJoinOp>())
    return callerShape(join.getBase(), depth + 1);
  if (auto join = value.getDefiningOp<taskflow::TaskflowReadCompletionJoinOp>())
    return callerShape(join.getBaseState(), depth + 1);
  if (auto producer = value.getDefiningOp<taskflow::TaskflowTaskOp>()) {
    if (!producer.getBody().hasOneBlock())
      return std::nullopt;
    auto yield = dyn_cast<taskflow::TaskflowYieldOp>(
        producer.getBody().front().getTerminator());
    if (!yield)
      return std::nullopt;
    Value forwarded;
    auto read = llvm::find(producer.getDoneReads(), value);
    auto write = llvm::find(producer.getDoneWrites(), value);
    if (read != producer.getDoneReads().end()) {
      unsigned index = read - producer.getDoneReads().begin();
      if (index >= yield.getDoneReads().size())
        return std::nullopt;
      forwarded = yield.getDoneReads()[index];
    } else if (write != producer.getDoneWrites().end()) {
      unsigned index = write - producer.getDoneWrites().begin();
      if (index >= yield.getDoneWrites().size())
        return std::nullopt;
      forwarded = yield.getDoneWrites()[index];
    } else {
      return std::nullopt;
    }
    auto argument = dyn_cast<BlockArgument>(forwarded);
    if (!argument || argument.getOwner() != &producer.getBody().front() ||
        argument.getArgNumber() >= producer->getNumOperands())
      return std::nullopt;
    if (read != producer.getDoneReads().end()) {
      if (argument.getArgNumber() >= producer.getWillReads().size())
        return std::nullopt;
    } else if (argument.getArgNumber() !=
               producer.getWillReads().size() +
                   (write - producer.getDoneWrites().begin())) {
      return std::nullopt;
    }
    return callerShape(producer->getOperand(argument.getArgNumber()), depth + 1);
  }
  // Private, statically allocated intermediates carry their storage extent
  // directly in the allocation type. The current output rectangle is still
  // derived from counters and indexed accesses and checked within that extent.
  if (value.getDefiningOp<memref::AllocOp>() ||
      value.getDefiningOp<memref::AllocaOp>()) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (type && type.hasStaticShape() &&
        llvm::all_of(type.getShape(), [](int64_t extent) { return extent > 0; }))
      return llvm::SmallVector<int64_t>(type.getShape().begin(),
                                       type.getShape().end());
    return std::nullopt;
  }
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || !argument.getOwner())
    return std::nullopt;
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front())
    return std::nullopt;
  Attribute attr = function.getArgAttr(argument.getArgNumber(),
                                      "amoeba.logical_transfer_shape");
  // A fully static function-argument type independently fixes its storage
  // capacity. Dynamic arguments still require the explicit caller contract;
  // a malformed present contract must not fall back to the type.
  if (!attr) {
    auto type = dyn_cast<MemRefType>(argument.getType());
    if (type && type.hasStaticShape() &&
        llvm::all_of(type.getShape(), [](int64_t extent) { return extent > 0; }))
      return llvm::SmallVector<int64_t>(type.getShape().begin(),
                                      type.getShape().end());
    return std::nullopt;
  }
  if (auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(attr))
    return llvm::SmallVector<int64_t>(dense.asArrayRef().begin(),
                                      dense.asArrayRef().end());
  auto array = dyn_cast_or_null<ArrayAttr>(attr);
  if (!array)
    return std::nullopt;
  llvm::SmallVector<int64_t> shape;
  for (Attribute element : array) {
    auto integer = dyn_cast<IntegerAttr>(element);
    if (!integer)
      return std::nullopt;
    shape.push_back(integer.getInt());
  }
  return shape;
}

inline std::optional<llvm::SmallVector<unsigned>>
parseCounterMap(mlir::Attribute attr) {
  using namespace mlir;
  llvm::SmallVector<unsigned> result;
  if (auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(attr)) {
    for (int64_t value : dense.asArrayRef()) {
      if (value < 0 || value > std::numeric_limits<unsigned>::max())
        return std::nullopt;
      result.push_back(static_cast<unsigned>(value));
    }
    return result;
  }
  auto array = dyn_cast_or_null<ArrayAttr>(attr);
  if (!array)
    return std::nullopt;
  for (Attribute element : array) {
    auto integer = dyn_cast<IntegerAttr>(element);
    if (!integer || integer.getInt() < 0 ||
        integer.getInt() > std::numeric_limits<unsigned>::max())
      return std::nullopt;
    result.push_back(static_cast<unsigned>(integer.getInt()));
  }
  return result;
}

} // namespace detail

inline ReplicaOutputCoordinateProof
analyzeReplicaOutputCoordinates(mlir::taskflow::TaskflowTaskOp task,
                                int64_t requestedAxis = -1,
                                unsigned outputWriteIndex = 0) {
  using namespace mlir;
  using namespace detail;
  ReplicaOutputCoordinateProof proof;
  if (!task || !task.getBody().hasOneBlock() ||
      task.getWillWrites().empty() ||
      task.getOriginalWriteMemrefs().size() != task.getWillWrites().size() ||
      outputWriteIndex >= task.getWillWrites().size()) {
    proofFail(proof, "requires a valid Taskflow output and write provenance");
    return proof;
  }
  Value outputState = task.getWillWrites()[outputWriteIndex];
  Value outputRoot = task.getOriginalWriteMemrefs()[outputWriteIndex];
  auto outputType = dyn_cast<MemRefType>(outputState.getType());
  auto rootType = dyn_cast<MemRefType>(outputRoot.getType());
  if (!outputType || !rootType || outputType.getRank() != rootType.getRank() ||
      outputType.getRank() < 1 || outputType.getRank() > 3) {
    proofFail(proof, "output state/storage rank is not supported");
    return proof;
  }
  for (auto [index, otherRoot] : llvm::enumerate(
           task.getOriginalWriteMemrefs())) {
    if (index == outputWriteIndex)
      continue;
    if (!provesDistinctStorage(outputRoot, otherRoot)) {
      proofFail(proof,
                "output write storage may alias the selected output slot");
      return proof;
    }
  }
  const unsigned rank = static_cast<unsigned>(rootType.getRank());
  auto shape = callerShape(outputRoot);
  if (!shape || shape->size() != rank ||
      llvm::any_of(*shape, [](int64_t extent) { return extent <= 0; })) {
    proofFail(proof, "missing positive logical transfer shape");
    return proof;
  }
  auto staticDimensionsAgree = [&](MemRefType type) {
    if (!type || type.getRank() != static_cast<int64_t>(shape->size()))
      return false;
    for (auto [dimension, extent] : llvm::enumerate(*shape)) {
      int64_t declared = type.getDimSize(dimension);
      if (!ShapedType::isDynamic(declared) && declared != extent)
        return false;
    }
    return true;
  };
  if (!staticDimensionsAgree(outputType) ||
      !staticDimensionsAgree(rootType)) {
    proofFail(proof,
              "logical transfer shape disagrees with a static memref extent");
    return proof;
  }
  proof.callerShape = *shape;

  SmallVector<TaskflowCounterOp> taskCounters;
  for (Operation &operation : task.getBody().front())
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      taskCounters.push_back(counter);
  if (taskCounters.size() < rank) {
    proofFail(proof, "too few Taskflow counters");
    return proof;
  }
  llvm::sort(taskCounters, [](TaskflowCounterOp lhs, TaskflowCounterOp rhs) {
    auto lhsId = lhs->getAttrOfType<IntegerAttr>("counter_id");
    auto rhsId = rhs->getAttrOfType<IntegerAttr>("counter_id");
    if (!lhsId || !rhsId)
      return static_cast<bool>(lhsId);
    return lhsId.getInt() < rhsId.getInt();
  });
  for (auto [index, counter] : llvm::enumerate(taskCounters)) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    auto lower = staticIndex(counter.getLowerBound(), task);
    auto upper = staticIndex(counter.getUpperBound(), task);
    auto step = staticIndex(counter.getStep(), task);
    bool parentOK = index == 0
                        ? !counter.getParentIndex()
                        : counter.getParentIndex() ==
                              taskCounters[index - 1].getCounterIndex();
    if (!id || id.getInt() != static_cast<int64_t>(index) || !lower ||
        !upper || !step || *upper <= *lower || *step != 1 || !parentOK) {
      proofFail(proof, "Taskflow counter IDs, parent chain, or bounds invalid");
      return proof;
    }
    proof.taskLowers.push_back(*lower);
    proof.taskUppers.push_back(*upper);
  }

  ::mlir::neura::KernelOp kernel;
  unsigned kernelCount = 0;
  for (Operation &operation : task.getBody().front())
    if (auto candidate = dyn_cast<::mlir::neura::KernelOp>(&operation)) {
      kernel = candidate;
      ++kernelCount;
    }
  if (kernelCount != 1 || !kernel || !kernel.getBody().hasOneBlock()) {
    proofFail(proof, "requires one one-block Neura kernel");
    return proof;
  }
  SmallVector<::mlir::neura::CounterOp> kernelCounters;
  for (Operation &operation : kernel.getBody().front())
    if (auto counter = dyn_cast<::mlir::neura::CounterOp>(&operation))
      kernelCounters.push_back(counter);
  llvm::sort(kernelCounters, [](::mlir::neura::CounterOp lhs,
                                ::mlir::neura::CounterOp rhs) {
    auto lhsId = lhs->getAttrOfType<IntegerAttr>("counter_id");
    auto rhsId = rhs->getAttrOfType<IntegerAttr>("counter_id");
    if (!lhsId || !rhsId)
      return static_cast<bool>(lhsId);
    return lhsId.getInt() < rhsId.getInt();
  });
  if (kernelCounters.size() != taskCounters.size()) {
    proofFail(proof, "Taskflow and Neura counter counts differ");
    return proof;
  }
  for (auto [index, counter] : llvm::enumerate(kernelCounters)) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    auto lower = staticCounterAttribute(counter.getOperation(),
                                        "lower_bound_value",
                                        counter.getLowerBound(), task, kernel);
    auto upper = staticCounterAttribute(counter.getOperation(),
                                        "upper_bound_value",
                                        counter.getUpperBound(), task, kernel);
    auto step = staticCounterAttribute(counter.getOperation(), "step_value",
                                       counter.getStep(), task, kernel);
    if (!id || id.getInt() != static_cast<int64_t>(index) || !lower ||
        !upper || !step || *lower != proof.taskLowers[index] ||
        *upper != proof.taskUppers[index] || *step != 1) {
      proofFail(proof, "Neura counter bounds do not match Taskflow bounds");
      return proof;
    }
    proof.kernelLowers.push_back(*lower);
    proof.kernelUppers.push_back(*upper);
  }

  size_t expectedBodyArguments = task.getWillReads().size() +
                                 task.getWillWrites().size() +
                                 task.getValueInputs().size();
  if (task.getBody().front().getNumArguments() != expectedBodyArguments) {
    proofFail(proof, "replica-coordinate-body-args: " +
                         std::to_string(task.getBody().front().getNumArguments()) +
                         ", expected " + std::to_string(expectedBodyArguments) +
                         ")");
    return proof;
  }
  if (task.getWillReads().size() != task.getOriginalReadMemrefs().size()) {
    proofFail(proof, "replica-coordinate-read-roots: " +
                         std::to_string(task.getOriginalReadMemrefs().size()) +
                         ", read states " +
                         std::to_string(task.getWillReads().size()) + ")");
    return proof;
  }
  for (auto [index, input] : llvm::enumerate(kernel.getInputs())) {
    if (input == outputState || input == outputRoot) {
      if (proof.outputInput != std::numeric_limits<unsigned>::max() &&
          proof.outputInput != index) {
        proofFail(proof, "kernel has duplicate output inputs");
        return proof;
      }
      proof.outputInput = index;
      continue;
    }
    if (auto argument = dyn_cast<BlockArgument>(input))
      if (argument.getOwner() == &task.getBody().front() &&
          argument.getArgNumber() < task->getNumOperands()) {
        unsigned operand = argument.getArgNumber();
        Value root;
        if (operand < task.getWillReads().size() &&
            operand < task.getOriginalReadMemrefs().size())
          root = task.getOriginalReadMemrefs()[operand];
        else {
          operand -= task.getWillReads().size();
          if (operand < task.getWillWrites().size() &&
              operand < task.getOriginalWriteMemrefs().size())
            root = task.getOriginalWriteMemrefs()[operand];
        }
        if (root == outputRoot) {
          if (proof.outputInput != std::numeric_limits<unsigned>::max() &&
              proof.outputInput != index) {
            proofFail(proof, "kernel has duplicate output inputs");
            return proof;
          }
          proof.outputInput = index;
        }
      }
  }
  if (proof.outputInput == std::numeric_limits<unsigned>::max()) {
    proofFail(proof, "cannot resolve the output kernel input");
    return proof;
  }

  llvm::SmallDenseSet<unsigned, 4> outputInputIndices;
  for (unsigned input = 0; input < kernel.getInputs().size(); ++input) {
    auto root = taskRootForKernelInput(input, kernel, task);
    if (!root)
      continue;
    for (Value output : task.getOriginalWriteMemrefs())
      if (*root == output)
        outputInputIndices.insert(input);
  }
  if (!outputInputIndices.contains(proof.outputInput)) {
    proofFail(proof, "selected output kernel input has no write provenance");
    return proof;
  }

  auto checkAuxiliary = [&](unsigned input) -> bool {
    auto root = taskRootForKernelInput(input, kernel, task);
    return root && *root != outputRoot &&
           llvm::is_contained(task.getOriginalReadMemrefs(), *root) &&
           provesDistinctStorage(*root, outputRoot);
  };
  bool sawUnknownMemory = false;
  auto checkOutputIndices = [&](Operation *operation, ValueRange indices) {
    if (indices.size() != rank) {
      proofFail(proof, "output access does not have a complete index vector");
      return;
    }
    SmallVector<unsigned> mapping;
    SmallVector<std::optional<int64_t>> constants;
    llvm::SmallDenseSet<unsigned, 4> used;
    for (auto [dimension, index] : llvm::enumerate(indices)) {
      auto route = counterRoute(index, kernelCounters);
      if (route) {
        if (!used.insert(*route).second) {
          proofFail(proof,
                    "output index routes are not distinct authenticated "
                    "counters");
          return;
        }
        mapping.push_back(*route);
        constants.push_back(std::nullopt);
        continue;
      }
      auto constant = staticIndex(index, task, kernel);
      if (rank != 3 || !constant || *constant < 0 ||
          static_cast<size_t>(dimension) >= proof.callerShape.size() ||
          *constant >= proof.callerShape[dimension]) {
        proofFail(proof,
                  "output index is neither an authenticated counter nor a "
                  "bounded constant");
        return;
      }
      mapping.push_back(ReplicaOutputCoordinateProof::kConstantAxis);
      constants.push_back(*constant);
    }
    if (proof.outputCounterAxes.empty()) {
      proof.outputCounterAxes = mapping;
      proof.outputConstantAxes = constants;
    }
    proof.outputAccessCounterAxes.push_back(std::move(mapping));
    proof.outputAccessConstantAxes.push_back(std::move(constants));
    SmallVector<unsigned> &current = proof.outputAccessCounterAxes.back();
    SmallVector<std::optional<int64_t>> &currentConstants =
        proof.outputAccessConstantAxes.back();
    if (current.size() != proof.outputCounterAxes.size() ||
        currentConstants.size() != proof.outputConstantAxes.size()) {
      proofFail(proof, "output access rank changes");
      return;
    }
    if (rank == 3 &&
        (!llvm::equal(current, proof.outputCounterAxes) ||
         !llvm::equal(currentConstants, proof.outputConstantAxes))) {
      proofFail(proof,
                "rank-3 output accesses do not preserve the authenticated "
                "coordinate map");
      return;
    }
    int64_t axis = requestedAxis;
    if (axis < 0) {
      TaskflowTaskOp parentTask = operation->getParentOfType<TaskflowTaskOp>();
      if (auto outputAxisAttr = parentTask->getAttrOfType<IntegerAttr>(
              "amoeba.replica.output_shard_axis"))
        axis = outputAxisAttr.getInt();
      else if (auto legacyShardAxisAttr =
                   parentTask->getAttrOfType<IntegerAttr>(
                       "amoeba.replica.shard_axis"))
        axis = legacyShardAxisAttr.getInt();
    }
    if (axis >= 0 &&
        (axis >= static_cast<int64_t>(current.size()) ||
         current[axis] == ReplicaOutputCoordinateProof::kConstantAxis ||
         current[axis] != proof.outputCounterAxes[axis]))
      proofFail(proof, "output access crosses the selected shard coordinate");
  };

  kernel.walk([&](Operation *operation) {
    if (!proof.reason.empty() || operation == kernel.getOperation())
      return;
    if (auto load = dyn_cast<::mlir::neura::LoadIndexedOp>(operation)) {
      auto input = accessInput(operation, load.getBase(), kernel);
      if (!input) {
        proofFail(proof, "indexed load base is not an authenticated input");
        return;
      }
      if (*input == proof.outputInput) {
        checkOutputIndices(operation, load.getIndices());
        proof.sawOutputLoad = proof.reason.empty();
      } else {
        if (outputInputIndices.contains(*input)) {
          proofFail(proof, "output slot load crosses another output storage");
          return;
        }
        if (!checkAuxiliary(*input)) {
          proofFail(proof, "auxiliary indexed load may alias output storage");
          return;
        }
        if (!llvm::is_contained(proof.auxiliaryInputIndices, *input))
          proof.auxiliaryInputIndices.push_back(*input);
      }
      return;
    }
    if (auto store = dyn_cast<::mlir::neura::StoreIndexedOp>(operation)) {
      auto input = accessInput(operation, store.getBase(), kernel);
      if (!input) {
        proofFail(proof, "indexed store base is not an authenticated input");
        return;
      }
      if (*input != proof.outputInput) {
        if (outputInputIndices.contains(*input))
          return;
        proofFail(proof, "auxiliary store is outside the read-only contract");
        return;
      }
      checkOutputIndices(operation, store.getIndices());
      proof.sawOutputStore = proof.reason.empty();
      return;
    }
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      auto input = accessInput(operation, load.getMemRef(), kernel);
      if (!input) {
        proofFail(proof, "memref load base is not an authenticated input");
        return;
      }
      if (*input == proof.outputInput) {
        checkOutputIndices(operation, load.getIndices());
        proof.sawOutputLoad = proof.reason.empty();
      } else {
        if (outputInputIndices.contains(*input)) {
          proofFail(proof, "output slot load crosses another output storage");
          return;
        }
        if (!checkAuxiliary(*input)) {
          proofFail(proof, "auxiliary memref load may alias output storage");
          return;
        }
        if (!llvm::is_contained(proof.auxiliaryInputIndices, *input))
          proof.auxiliaryInputIndices.push_back(*input);
      }
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      auto input = accessInput(operation, store.getMemRef(), kernel);
      if (!input) {
        proofFail(proof, "memref store base is not an authenticated input");
        return;
      }
      if (*input != proof.outputInput) {
        if (outputInputIndices.contains(*input))
          return;
        proofFail(proof, "auxiliary memref store is outside the read-only contract");
        return;
      }
      checkOutputIndices(operation, store.getIndices());
      proof.sawOutputStore = proof.reason.empty();
      return;
    }
    StringRef name = operation->getName().getStringRef();
    if (name == "neura.load" || name == "neura.store") {
      proofFail(proof, "unclassified memory operation in Neura output proof");
      sawUnknownMemory = true;
      return;
    }
  });
  (void)sawUnknownMemory;
  if (!proof.reason.empty() || !proof.sawOutputStore ||
      proof.outputCounterAxes.size() != rank) {
    if (proof.reason.empty())
      proofFail(proof, "output store or complete map is missing");
    return proof;
  }
  llvm::SmallDenseSet<unsigned, 4> mapped;
  proof.producedLowers.resize(rank);
  proof.producedShape.resize(rank);
  for (auto [dimension, counter] : llvm::enumerate(proof.outputCounterAxes)) {
    if (counter == ReplicaOutputCoordinateProof::kConstantAxis) {
      if (dimension >= proof.outputConstantAxes.size() ||
          !proof.outputConstantAxes[dimension]) {
        proofFail(proof, "constant output dimension lacks an authenticated value");
        return proof;
      }
      proof.producedLowers[dimension] = *proof.outputConstantAxes[dimension];
      proof.producedShape[dimension] = 1;
      continue;
    }
    if (counter >= proof.taskUppers.size() || !mapped.insert(counter).second) {
      proofFail(proof, "output counter map is not one-to-one");
      return proof;
    }
    int64_t lower = proof.taskLowers[counter];
    int64_t upper = proof.taskUppers[counter];
    if (lower < 0 || upper <= lower ||
        upper > proof.callerShape[dimension]) {
      proofFail(proof, "produced output region exceeds caller shape");
      return proof;
    }
    // Bounds are now nonnegative and limited by the positive caller extent,
    // so this subtraction cannot overflow signed int64.
    proof.producedLowers[dimension] = lower;
    proof.producedShape[dimension] = upper - lower;
  }
  if (auto attr = task->getAttr("amoeba.replica.output_counter_axes")) {
    auto declared = parseCounterMap(attr);
    if (!declared || *declared != proof.outputCounterAxes) {
      proofFail(proof, "declared output counter map does not match IR proof");
      return proof;
    }
  }
  if (Attribute attr = task->getAttr("amoeba.replica.output_constant_axes")) {
    SmallVector<int64_t> declared;
    if (auto dense = dyn_cast<DenseI64ArrayAttr>(attr)) {
      declared.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
    } else if (auto array = dyn_cast<ArrayAttr>(attr)) {
      for (Attribute element : array) {
        auto integer = dyn_cast<IntegerAttr>(element);
        if (!integer) {
          proofFail(proof,
                    "declared output constant map is not an integer array");
          return proof;
        }
        declared.push_back(integer.getInt());
      }
    } else {
      proofFail(proof, "declared output constant map has an unsupported type");
      return proof;
    }
    if (declared.size() != proof.outputConstantAxes.size()) {
      proofFail(proof, "declared output constant map has the wrong rank");
      return proof;
    }
    for (auto [dimension, value] : llvm::enumerate(declared)) {
      bool isUnmapped =
          value == std::numeric_limits<int64_t>::min();
      if (isUnmapped != !proof.outputConstantAxes[dimension] ||
          (!isUnmapped &&
           value != *proof.outputConstantAxes[dimension])) {
        proofFail(proof,
                  "declared output constant map does not match IR proof");
        return proof;
      }
    }
  }
  int64_t axis = requestedAxis;
  if (axis < 0) {
    if (auto outputAxisAttr = task->getAttrOfType<IntegerAttr>(
            "amoeba.replica.output_shard_axis"))
      axis = outputAxisAttr.getInt();
    else if (auto legacyShardAxisAttr = task->getAttrOfType<IntegerAttr>(
                 "amoeba.replica.shard_axis"))
      axis = legacyShardAxisAttr.getInt();
  }
  if (axis >= 0) {
    if (axis >= static_cast<int64_t>(rank)) {
      proofFail(proof, "selected shard axis is outside output rank");
      return proof;
    }
    for (ArrayRef<unsigned> access : proof.outputAccessCounterAxes)
      if (access[axis] != proof.outputCounterAxes[axis]) {
        proofFail(proof, "output access crosses the selected shard coordinate");
        return proof;
      }
    proof.selectedAxisIndependent = true;
  }
  proof.proven = true;
  return proof;
}

} // namespace mlir::amoeba::neura::joint_scheduling

#endif
