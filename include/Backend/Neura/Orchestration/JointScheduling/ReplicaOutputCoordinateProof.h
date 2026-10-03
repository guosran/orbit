#ifndef AMOEBA_REPLICA_OUTPUT_COORDINATE_PROOF_H
#define AMOEBA_REPLICA_OUTPUT_COORDINATE_PROOF_H

// Shared proof for logical output-coordinate permutations.  It derives the
// mapping from the Taskflow and Neura IR itself.  Replica metadata is checked
// against the proof only after this derivation; it is never used as the proof
// source.

#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace mlir::amoeba::neura::joint_scheduling {

using namespace mlir::taskflow;

struct ReplicaOutputCoordinateProof {
  bool proven = false;
  bool selectedAxisIndependent = false;
  bool sawOutputLoad = false;
  bool sawOutputStore = false;
  unsigned outputInput = std::numeric_limits<unsigned>::max();
  SmallVector<unsigned> outputCounterAxes;
  SmallVector<SmallVector<unsigned>> outputAccessCounterAxes;
  SmallVector<unsigned> auxiliaryInputIndices;
  SmallVector<int64_t> taskLowers;
  SmallVector<int64_t> taskUppers;
  SmallVector<int64_t> kernelLowers;
  SmallVector<int64_t> kernelUppers;
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

inline std::optional<int64_t>
staticIndex(mlir::Value value, mlir::taskflow::TaskflowTaskOp task,
            mlir::neura::KernelOp kernel,
            llvm::SmallDenseSet<mlir::Value, 16> &visiting) {
  using namespace mlir;
  if (!value || !visiting.insert(value).second)
    return std::nullopt;
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
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
      std::string name = "amoeba.static_bound.arg." +
                         std::to_string(argument.getArgNumber());
      if (auto bound = function->getAttrOfType<IntegerAttr>(name))
        return bound.getInt();
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
  // Arithmetic on a bound/index is not folded here.  Even checked int64
  // arithmetic is insufficient to prove the source integer width and
  // signedness semantics, so a non-constant arithmetic route stays unknown.
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
callerShape(mlir::Value value) {
  using namespace mlir;
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || !argument.getOwner())
    return std::nullopt;
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front())
    return std::nullopt;
  Attribute attr = function.getArgAttr(argument.getArgNumber(),
                                      "amoeba.logical_transfer_shape");
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
      outputType.getRank() < 1 || outputType.getRank() > 2) {
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
    llvm::SmallDenseSet<unsigned, 4> used;
    for (Value index : indices) {
      auto route = counterRoute(index, kernelCounters);
      if (!route || !used.insert(*route).second) {
        proofFail(proof, "output index route is not an authenticated counter");
        return;
      }
      mapping.push_back(*route);
    }
    if (proof.outputCounterAxes.empty())
      proof.outputCounterAxes = mapping;
    proof.outputAccessCounterAxes.push_back(std::move(mapping));
    SmallVector<unsigned> &current = proof.outputAccessCounterAxes.back();
    if (current.size() != proof.outputCounterAxes.size()) {
      proofFail(proof, "output access rank changes");
      return;
    }
    int64_t axis = requestedAxis;
    if (axis < 0)
      if (auto attr = operation->getParentOfType<TaskflowTaskOp>()
                          ->getAttrOfType<IntegerAttr>(
                              "amoeba.replica.shard_axis"))
        axis = attr.getInt();
    if (axis >= 0 &&
        (axis >= static_cast<int64_t>(current.size()) ||
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
    StringRef name = operation->getName().getStringRef();
    if (name == "neura.load" || name == "neura.store" ||
        name == "memref.load" || name == "memref.store") {
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
  proof.producedShape.resize(rank);
  for (auto [dimension, counter] : llvm::enumerate(proof.outputCounterAxes)) {
    if (counter >= proof.taskUppers.size() || !mapped.insert(counter).second) {
      proofFail(proof, "output counter map is not one-to-one");
      return proof;
    }
    int64_t extent = proof.taskUppers[counter] - proof.taskLowers[counter];
    if (extent <= 0 || extent > proof.callerShape[dimension]) {
      proofFail(proof, "produced output region exceeds caller shape");
      return proof;
    }
    proof.producedShape[dimension] = extent;
  }
  if (auto attr = task->getAttr("amoeba.replica.output_counter_axes")) {
    auto declared = parseCounterMap(attr);
    if (!declared || *declared != proof.outputCounterAxes) {
      proofFail(proof, "declared output counter map does not match IR proof");
      return proof;
    }
  }
  int64_t axis = requestedAxis;
  if (axis < 0)
    if (auto attr = task->getAttrOfType<IntegerAttr>(
            "amoeba.replica.shard_axis"))
      axis = attr.getInt();
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
