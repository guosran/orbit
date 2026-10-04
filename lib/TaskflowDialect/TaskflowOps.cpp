#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "Backend/Neura/Orchestration/JointScheduling/ReplicaOutputCoordinateProof.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

static std::optional<int64_t> constantIndex(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  return std::nullopt;
}

static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
regionFromOutputProof(const mlir::amoeba::neura::joint_scheduling::
                          ReplicaOutputCoordinateProof &proof,
                      MemRefType baseType) {
  if (!baseType || !baseType.hasStaticShape() ||
      proof.producedLowers.size() != static_cast<size_t>(baseType.getRank()) ||
      proof.producedShape.size() != proof.producedLowers.size())
    return failure();

  SmallVector<int64_t> lower(proof.producedLowers.begin(),
                             proof.producedLowers.end());
  SmallVector<int64_t> upper(lower);
  for (auto [dimension, extent] : llvm::enumerate(proof.producedShape)) {
    if (extent <= 0 || lower[dimension] < 0 ||
        lower[dimension] > std::numeric_limits<int64_t>::max() - extent)
      return failure();
    upper[dimension] = lower[dimension] + extent;
    if (upper[dimension] > baseType.getShape()[dimension])
      return failure();
  }
  return std::make_pair(std::move(lower), std::move(upper));
}

static LogicalResult verifyReplicaShardMetadata(
    TaskflowTaskOp task,
    const mlir::amoeba::neura::joint_scheduling::
        ReplicaOutputCoordinateProof &proof,
    int64_t selectedAxis) {
  if (selectedAxis < 0 ||
      selectedAxis >= static_cast<int64_t>(proof.outputCounterAxes.size()))
    return failure();
  auto outputAxis = task->getAttrOfType<IntegerAttr>(
      "amoeba.replica.output_shard_axis");
  auto shardAxis = task->getAttrOfType<IntegerAttr>(
      "amoeba.replica.shard_axis");
  if ((outputAxis && outputAxis.getInt() != selectedAxis) ||
      (!outputAxis && (!shardAxis || shardAxis.getInt() != selectedAxis)))
    return failure();

  unsigned shardCounter = proof.outputCounterAxes[selectedAxis];
  if (shardCounter ==
          mlir::amoeba::neura::joint_scheduling::
              ReplicaOutputCoordinateProof::kConstantAxis ||
      shardCounter >= proof.taskLowers.size() ||
      shardCounter >= proof.taskUppers.size() || !shardAxis ||
      (outputAxis &&
       shardAxis.getInt() != static_cast<int64_t>(shardCounter)))
    return failure();
  auto shardLower = task->getAttrOfType<IntegerAttr>(
      "amoeba.replica.shard_lower");
  auto shardUpper = task->getAttrOfType<IntegerAttr>(
      "amoeba.replica.shard_upper");
  if (!shardLower || !shardUpper ||
      shardLower.getInt() != proof.taskLowers[shardCounter] ||
      shardUpper.getInt() != proof.taskUppers[shardCounter])
    return failure();
  return success();
}

static Value stripTaskflowMemrefCasts(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>())
    value = cast.getSource();
  return value;
}

// Resolve a Taskflow dependence state to the exact storage value it forwards.
// Unsupported body aliases deliberately fail closed; this is used by the read
// completion barrier to ensure that its base state and original-root witness
// name the same memory object.
static FailureOr<Value> resolveTaskflowStorageRoot(Value value,
                                                   DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return failure();
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return resolveTaskflowStorageRoot(cast.getSource(), visited);
  if (auto channel = value.getDefiningOp<TaskflowChannelOp>())
    return resolveTaskflowStorageRoot(channel.getSource(), visited);
  if (auto join = value.getDefiningOp<TaskflowJoinOp>())
    return resolveTaskflowStorageRoot(join.getBase(), visited);
  if (auto join = value.getDefiningOp<TaskflowReadCompletionJoinOp>()) {
    if (failed(verify(join.getOperation())))
      return failure();
    return resolveTaskflowStorageRoot(join.getBaseState(), visited);
  }

  auto task = value.getDefiningOp<TaskflowTaskOp>();
  if (!task)
    return value;
  auto yield =
      dyn_cast<TaskflowYieldOp>(task.getBody().front().getTerminator());
  if (!yield)
    return failure();
  unsigned bodyArgument = task.getBody().front().getNumArguments();
  auto read = llvm::find(task.getDoneReads(), value);
  if (read != task.getDoneReads().end()) {
    size_t resultIndex = read - task.getDoneReads().begin();
    if (resultIndex >= yield.getDoneReads().size())
      return failure();
    auto argument = dyn_cast<BlockArgument>(yield.getDoneReads()[resultIndex]);
    if (!argument || argument.getOwner() != &task.getBody().front() ||
        argument.getArgNumber() >= task.getWillReads().size())
      return failure();
    bodyArgument = argument.getArgNumber();
  } else {
    auto write = llvm::find(task.getDoneWrites(), value);
    if (write == task.getDoneWrites().end())
      return failure();
    size_t resultIndex = write - task.getDoneWrites().begin();
    if (resultIndex >= yield.getDoneWrites().size())
      return failure();
    auto argument = dyn_cast<BlockArgument>(yield.getDoneWrites()[resultIndex]);
    if (!argument || argument.getOwner() != &task.getBody().front() ||
        argument.getArgNumber() < task.getWillReads().size() ||
        argument.getArgNumber() >=
            task.getWillReads().size() + task.getWillWrites().size())
      return failure();
    bodyArgument = argument.getArgNumber();
  }
  if (bodyArgument >= task->getNumOperands())
    return failure();
  return resolveTaskflowStorageRoot(task->getOperand(bodyArgument), visited);
}

static bool taskReadResultInput(TaskflowTaskOp task, Value state,
                                unsigned &resultIndex, unsigned &inputIndex) {
  if (!task || !task.getBody().hasOneBlock())
    return false;
  auto result = llvm::find(task.getDoneReads(), state);
  auto yield =
      dyn_cast<TaskflowYieldOp>(task.getBody().front().getTerminator());
  if (result == task.getDoneReads().end() || !yield)
    return false;
  resultIndex = static_cast<unsigned>(result - task.getDoneReads().begin());
  if (resultIndex >= yield.getDoneReads().size())
    return false;
  auto argument = dyn_cast<BlockArgument>(yield.getDoneReads()[resultIndex]);
  if (!argument || argument.getOwner() != &task.getBody().front() ||
      argument.getArgNumber() >= task.getWillReads().size() ||
      task.getOriginalReadMemrefs().size() != task.getWillReads().size())
    return false;
  inputIndex = argument.getArgNumber();
  return task.getWillReads()[inputIndex].getType() == state.getType();
}

static bool hasOnlyJoinUseThroughCasts(Value state, TaskflowJoinOp join) {
  Operation *expectedOwner = join.getOperation();
  while (auto cast = state.getDefiningOp<memref::CastOp>()) {
    if (!llvm::hasSingleElement(state.getUses()) ||
        state.use_begin()->getOwner() != expectedOwner)
      return false;
    expectedOwner = cast.getOperation();
    state = cast.getSource();
  }
  return llvm::hasSingleElement(state.getUses()) &&
         state.use_begin()->getOwner() == expectedOwner;
}

static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
stateRegion(Value state, Value expectedBase, int64_t selectedAxis = -1) {
  // Compatible memref casts change the view type, not the storage or the
  // completion token. Do not follow subviews or other alias-producing ops.
  auto stripCasts = [](Value value) {
    while (auto cast = value.getDefiningOp<memref::CastOp>())
      value = cast.getSource();
    return value;
  };
  state = stripCasts(state);
  Value originalBase = stripCasts(expectedBase);
  if (auto join = state.getDefiningOp<TaskflowJoinOp>()) {
    if (stripCasts(join.getBase()) != originalBase)
      return failure();
    return std::make_pair(SmallVector<int64_t>(join.getRegionLower()),
                          SmallVector<int64_t>(join.getRegionUpper()));
  }
  auto baseType = dyn_cast<MemRefType>(expectedBase.getType());
  auto task = state.getDefiningOp<TaskflowTaskOp>();
  if (!task || !llvm::is_contained(task.getDoneWrites(), state) ||
      task.getDoneWrites().size() != task.getWillWrites().size() ||
      task.getWillWrites().size() != task.getOriginalWriteMemrefs().size())
    return failure();
  auto stateIt = llvm::find(task.getDoneWrites(), state);
  if (stateIt == task.getDoneWrites().end())
    return failure();
  const size_t writeIndex = stateIt - task.getDoneWrites().begin();
  if (stripCasts(task.getWillWrites()[writeIndex]) != originalBase)
    return failure();
  // The write dependence state can be a prior task's completion token while
  // the original-write operand remains its storage root. Preserve the exact
  // state-version check above and independently authenticate that root.
  DenseSet<Value> baseVisited, writeVisited;
  FailureOr<Value> baseRoot =
      resolveTaskflowStorageRoot(originalBase, baseVisited);
  FailureOr<Value> writeRoot = resolveTaskflowStorageRoot(
      task.getOriginalWriteMemrefs()[writeIndex], writeVisited);
  if (failed(baseRoot) || failed(writeRoot) || *baseRoot != *writeRoot)
    return failure();

  // Multi-output post-Neura tasks carry one exact rectangle per completion
  // state.  A standalone join verifier must rederive that rectangle from the
  // indexed accesses for this write slot; materializer metadata is only a
  // witness which must agree with the independent proof.
  auto lowers = task->getAttrOfType<ArrayAttr>(
      "amoeba.tiling.output_region_lowers");
  auto uppers = task->getAttrOfType<ArrayAttr>(
      "amoeba.tiling.output_region_uppers");
  if (task.getWillWrites().size() > 1) {
    auto proof =
        mlir::amoeba::neura::joint_scheduling::analyzeReplicaOutputCoordinates(
            task, selectedAxis, static_cast<unsigned>(writeIndex));
    if (!proof.proven) {
      task.emitError() << "authenticated multi-output coordinate proof failed: "
                       << proof.reason;
      return failure();
    }
    if (selectedAxis >= 0 &&
        task->hasAttr("amoeba.replica.output_counter_axes") &&
        failed(verifyReplicaShardMetadata(task, proof, selectedAxis)))
      return failure();
    auto actual = regionFromOutputProof(proof, baseType);
    if (failed(actual))
      return failure();
    if (lowers || uppers) {
      if (!lowers || !uppers ||
          lowers.size() != task.getWillWrites().size() ||
          uppers.size() != lowers.size())
        return failure();
      auto lower = dyn_cast<DenseI64ArrayAttr>(lowers[writeIndex]);
      auto upper = dyn_cast<DenseI64ArrayAttr>(uppers[writeIndex]);
      if (!lower || !upper ||
          static_cast<size_t>(lower.size()) != actual->first.size() ||
          static_cast<size_t>(upper.size()) != actual->second.size() ||
          !llvm::equal(lower.asArrayRef(), actual->first) ||
          !llvm::equal(upper.asArrayRef(), actual->second))
        return failure();
    }
    return std::move(*actual);
  }

  if (task.getWillWrites().size() != 1)
    return failure();
  // Replica tasks carry a copied output-counter map only after the private
  // materializer has proved it from the actual Neura indexed accesses.  The
  // join verifier must independently re-derive that map; accepting the attr
  // by itself would let a forged permutation make a non-disjoint join look
  // legal.
  if (task->hasAttr("amoeba.replica.output_counter_axes")) {
    if (!baseType)
      return failure();
    auto proof =
        mlir::amoeba::neura::joint_scheduling::analyzeReplicaOutputCoordinates(
            task, selectedAxis);
    if (!proof.proven) {
      task.emitError() << "authenticated output-coordinate proof failed: "
                       << proof.reason;
      return failure();
    }
    if (proof.outputCounterAxes.size() !=
        static_cast<size_t>(baseType.getRank()))
      return failure();
    if (failed(verifyReplicaShardMetadata(task, proof, selectedAxis)))
      return failure();
    auto actual = regionFromOutputProof(proof, baseType);
    if (failed(actual))
      return failure();
    for (size_t dimension = 0; dimension < actual->first.size(); ++dimension)
      if (actual->second[dimension] <= actual->first[dimension])
        return failure();
    return std::move(*actual);
}
  SmallVector<int64_t> lower;
  SmallVector<int64_t> upper;
  SmallVector<TaskflowCounterOp> counters;
  for (Operation &operation : task.getBody().front())
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      counters.push_back(counter);
  size_t outputRank = baseType ? static_cast<size_t>(baseType.getRank()) : 0;
  if (!baseType || counters.size() < outputRank)
    return failure();
  // A reduction kernel may retain a K counter in addition to the output
  // counters.  Completion regions describe the written memref, so ignore
  // those trailing reduction counters when recovering the tile region.
  for (auto [index, counter] : llvm::enumerate(counters)) {
    std::optional<int64_t> lb = constantIndex(counter.getLowerBound());
    std::optional<int64_t> ub = constantIndex(counter.getUpperBound());
    std::optional<int64_t> step = constantIndex(counter.getStep());
    if (!lb || !ub || !step || *step != 1 || *ub <= *lb ||
        (index == 0 && counter.getParentIndex()) ||
        (index != 0 &&
         counter.getParentIndex() != counters[index - 1].getCounterIndex()))
      return failure();
    if (index < outputRank) {
      lower.push_back(*lb);
      upper.push_back(*ub);
    }
  }
  if (lower.empty())
    return failure();
  return std::make_pair(std::move(lower), std::move(upper));
}

} // namespace

LogicalResult TaskflowTaskOp::verify() {
  if (!getBody().hasOneBlock())
    return emitOpError("requires exactly one body block");
  Block &body = getBody().front();
  SmallVector<Value> inputs(getWillReads());
  inputs.append(getWillWrites().begin(), getWillWrites().end());
  inputs.append(getValueInputs().begin(), getValueInputs().end());
  if (body.getNumArguments() != inputs.size())
    return emitOpError("body argument count must match execution inputs; "
                       "original memrefs are provenance only");
  for (auto [argument, input] : llvm::zip(body.getArguments(), inputs))
    if (argument.getType() != input.getType())
      return emitOpError("body argument type does not match execution input");
  auto yield = dyn_cast<TaskflowYieldOp>(body.getTerminator());
  if (!yield || yield->getNumOperands() != getNumResults())
    return emitOpError("yield arity must match task results");
  for (auto [result, yielded] : llvm::zip(getResults(), yield->getOperands()))
    if (result.getType() != yielded.getType())
      return emitOpError("yield type does not match task result");
  return success();
}

LogicalResult TaskflowJoinOp::verify() {
  if (getTileStates().size() < 2)
    return emitOpError("requires at least two tile states");
  auto type = dyn_cast<MemRefType>(getBase().getType());
  if (!type || !type.hasStaticShape() || getJoined().getType() != type)
    return emitOpError(
        "requires one static memref type for base, states, and result");
  int64_t axis = getAxis();
  ArrayRef<int64_t> parentLower = getRegionLower();
  ArrayRef<int64_t> parentUpper = getRegionUpper();
  if (axis < 0 || axis >= type.getRank() ||
      parentLower.size() != static_cast<size_t>(type.getRank()) ||
      parentUpper.size() != static_cast<size_t>(type.getRank()))
    return emitOpError("has invalid axis or parent region rank");
  for (int64_t dim = 0; dim < type.getRank(); ++dim)
    if (parentLower[dim] < 0 || parentUpper[dim] <= parentLower[dim] ||
        parentUpper[dim] > type.getShape()[dim])
      return emitOpError("parent region is outside the static memref shape");

  SmallVector<std::pair<int64_t, int64_t>> intervals;
  llvm::SmallDenseSet<Value, 8> uniqueStates;
  for (Value state : getTileStates()) {
    if (state.getType() != type || !uniqueStates.insert(state).second)
      return emitOpError("requires distinct tile states of the base type");
    auto region = stateRegion(state, getBase(), axis);
    if (failed(region) || region->first.size() != parentLower.size() ||
        region->second.size() != parentUpper.size())
      return emitOpError("cannot prove a tile state's static region and base");
    for (int64_t dim = 0; dim < type.getRank(); ++dim) {
      if (dim == axis)
        continue;
      if (region->first[dim] != parentLower[dim] ||
          region->second[dim] != parentUpper[dim])
        return emitOpError("tile regions differ outside the selected axis");
    }
    intervals.push_back({region->first[axis], region->second[axis]});
  }
  llvm::sort(intervals);
  int64_t cursor = parentLower[axis];
  for (auto [lower, upper] : intervals) {
    if (lower != cursor || upper <= lower)
      return emitOpError("tile regions overlap or leave a gap");
    cursor = upper;
  }
  if (cursor != parentUpper[axis])
    return emitOpError("tile regions do not exactly cover the parent region");
  return success();
}

LogicalResult TaskflowReadCompletionJoinOp::verify() {
  if (getTileStates().size() < 2)
    return emitOpError("requires every state from at least two replicas");
  auto baseType = dyn_cast<MemRefType>(getBaseState().getType());
  auto rootType = dyn_cast<MemRefType>(getOriginalRoot().getType());
  if (!baseType || !rootType || getJoined().getType() != baseType ||
      rootType.getRank() != baseType.getRank() ||
      rootType.getElementType() != baseType.getElementType())
    return emitOpError("requires a compatible read-state, root, and result");
  if (!getOperation()->hasAttr("amoeba.semantic.completion_only") ||
      !getOperation()->hasAttr("amoeba.replica.completion_only"))
    return emitOpError("requires completion-only replica markers");

  auto writeJoin = getWriteCompletion().getDefiningOp<TaskflowJoinOp>();
  if (!writeJoin || writeJoin.getJoined() != getWriteCompletion() ||
      !writeJoin->hasAttr("amoeba.semantic.completion_only") ||
      !writeJoin->hasAttr("amoeba.replica.completion_only") ||
      failed(mlir::verify(writeJoin.getOperation())) ||
      writeJoin->getParentRegion() != getOperation()->getParentRegion() ||
      writeJoin.getTileStates().size() != getTileStates().size())
    return emitOpError(
        "must reference the verified output join for the same replica group");

  DenseMap<Operation *, std::pair<int64_t, int64_t>> outputReplicas;
  for (Value state : writeJoin.getTileStates()) {
    if (!hasOnlyJoinUseThroughCasts(state, writeJoin))
      return emitOpError("output completion state has an unexpected SSA user");
    TaskflowTaskOp task =
        stripTaskflowMemrefCasts(state).getDefiningOp<TaskflowTaskOp>();
    if (!task ||
        !llvm::is_contained(task.getDoneWrites(),
                            stripTaskflowMemrefCasts(state)) ||
        task->getParentRegion() != getOperation()->getParentRegion())
      return emitOpError("output join must contain direct replica done-writes");
    auto id = task->getAttrOfType<IntegerAttr>("amoeba.replica.id");
    auto count = task->getAttrOfType<IntegerAttr>("amoeba.replica.count");
    auto parent = task->getAttrOfType<StringAttr>("amoeba.replica.parent_task");
    if (!id || !count || !parent || parent.getValue().empty() ||
        id.getInt() < 0 || id.getInt() >= count.getInt() ||
        count.getInt() != static_cast<int64_t>(getTileStates().size()) ||
        !outputReplicas
             .insert(
                 std::make_pair(task.getOperation(),
                                std::make_pair(id.getInt(), count.getInt())))
             .second)
      return emitOpError("output join has incomplete replica identity");
  }

  DenseSet<Operation *> readReplicas;
  DenseSet<int64_t> readIds;
  std::optional<unsigned> commonResultIndex;
  std::optional<unsigned> commonInputIndex;
  std::optional<std::string> commonParent;
  DenseSet<int64_t> readCounts;
  for (Value state : getTileStates()) {
    if (state.getType() != baseType ||
        !llvm::hasSingleElement(state.getUses()) ||
        state.use_begin()->getOwner() != getOperation())
      return emitOpError("requires unique direct read-completion states");
    TaskflowTaskOp task = state.getDefiningOp<TaskflowTaskOp>();
    unsigned resultIndex = 0, inputIndex = 0;
    if (!task || task->getParentRegion() != getOperation()->getParentRegion() ||
        !taskReadResultInput(task, state, resultIndex, inputIndex) ||
        task.getDoneReads().size() != 1 || resultIndex != 0 ||
        task.getWillReads()[inputIndex] != getBaseState() ||
        task.getOriginalReadMemrefs()[inputIndex] != getOriginalRoot() ||
        !readReplicas.insert(task.getOperation()).second)
      return emitOpError(
          "read state does not preserve one exact will_reads version and root");
    auto output = outputReplicas.find(task.getOperation());
    auto id = task->getAttrOfType<IntegerAttr>("amoeba.replica.id");
    auto count = task->getAttrOfType<IntegerAttr>("amoeba.replica.count");
    auto parent = task->getAttrOfType<StringAttr>("amoeba.replica.parent_task");
    if (output == outputReplicas.end() || !id || !count || !parent ||
        output->second.first != id.getInt() ||
        output->second.second != count.getInt() ||
        !readIds.insert(id.getInt()).second)
      return emitOpError(
          "read completion and output joins contain different replica tasks");
    if (!commonResultIndex)
      commonResultIndex = resultIndex;
    if (!commonInputIndex)
      commonInputIndex = inputIndex;
    if (!commonParent)
      commonParent = parent.getValue().str();
    if (*commonResultIndex != resultIndex || *commonInputIndex != inputIndex ||
        *commonParent != parent.getValue().str())
      return emitOpError(
          "replicas disagree on read-result slot, input slot, or parent");
    readCounts.insert(static_cast<int64_t>(task.getDoneReads().size()));
  }
  if (readReplicas.size() != outputReplicas.size() ||
      readIds.size() != getTileStates().size() || readCounts.size() != 1)
    return emitOpError(
        "read completion does not cover the complete output replica set");

  DenseSet<Value> baseVisited, rootVisited;
  FailureOr<Value> baseRoot =
      resolveTaskflowStorageRoot(getBaseState(), baseVisited);
  FailureOr<Value> originalRoot =
      resolveTaskflowStorageRoot(getOriginalRoot(), rootVisited);
  if (failed(baseRoot) || failed(originalRoot) || *baseRoot != *originalRoot)
    return emitOpError(
        "base read state and original storage root do not resolve identically");
  return success();
}

//===----------------------------------------------------------------------===//
// TaskflowTaskOp
//===----------------------------------------------------------------------===//

ParseResult TaskflowTaskOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parses optional task name: @Task_0.
  StringAttr task_name;
  if (succeeded(parser.parseOptionalSymbolName(task_name))) {
    result.addAttribute("task_name", task_name);
  }

  // Parses will_reads(%arg0, %arg1 : memref<?xi32>, memref<?xi32>).
  SmallVector<OpAsmParser::UnresolvedOperand> read_operands;
  SmallVector<Type> read_types;
  if (succeeded(parser.parseOptionalKeyword("will_reads"))) {
    if (parser.parseLParen() || parser.parseOperandList(read_operands) ||
        parser.parseColonTypeList(read_types) || parser.parseRParen())
      return failure();
  }

  // Parses will_writes(%arg5 : memref<?xi32>).
  SmallVector<OpAsmParser::UnresolvedOperand> write_operands;
  SmallVector<Type> write_types;
  if (succeeded(parser.parseOptionalKeyword("will_writes"))) {
    if (parser.parseLParen() || parser.parseOperandList(write_operands) ||
        parser.parseColonTypeList(write_types) || parser.parseRParen())
      return failure();
  }

  // Parses value_inputs: value_inputs(%scalar : i32).
  SmallVector<OpAsmParser::UnresolvedOperand> value_operands;
  SmallVector<Type> value_types;
  if (succeeded(parser.parseOptionalKeyword("value_inputs"))) {
    if (parser.parseLParen() || parser.parseOperandList(value_operands) ||
        parser.parseColonTypeList(value_types) || parser.parseRParen())
      return failure();
  }

  // Parses original memrefs with explicit types:
  // [original_read_memrefs(%arg0, %arg1 : type1, type2),
  // original_write_memrefs(%arg5 : type3)].
  SmallVector<OpAsmParser::UnresolvedOperand> original_read_operands;
  SmallVector<Type> original_read_types;
  SmallVector<OpAsmParser::UnresolvedOperand> original_write_operands;
  SmallVector<Type> original_write_types;

  if (succeeded(parser.parseOptionalLSquare())) {
    // original_read_memrefs with types.
    if (succeeded(parser.parseOptionalKeyword("original_read_memrefs"))) {
      if (parser.parseLParen() ||
          parser.parseOperandList(original_read_operands) ||
          parser.parseColonTypeList(original_read_types) ||
          parser.parseRParen())
        return failure();
    }

    // optional comma.
    (void)parser.parseOptionalComma();

    // original_write_memrefs with types.
    if (succeeded(parser.parseOptionalKeyword("original_write_memrefs"))) {
      if (parser.parseLParen() ||
          parser.parseOperandList(original_write_operands) ||
          parser.parseColonTypeList(original_write_types) ||
          parser.parseRParen())
        return failure();
    }

    if (parser.parseRSquare())
      return failure();
  }

  // Validates operand/type count match.
  if (read_operands.size() != read_types.size() ||
      write_operands.size() != write_types.size() ||
      value_operands.size() != value_types.size() ||
      original_read_operands.size() != original_read_types.size() ||
      original_write_operands.size() != original_write_types.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "operand and type count mismatch");
  }

  // Resolves all operands.
  if (parser.resolveOperands(read_operands, read_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(write_operands, write_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(value_operands, value_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(original_read_operands, original_read_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(original_write_operands, original_write_types,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  // Parses optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parses function type: : (...) -> (...).
  FunctionType func_type;
  if (parser.parseColon() || parser.parseType(func_type))
    return failure();

  // Adds result types.
  result.addTypes(func_type.getResults());

  // Parses region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*args=*/{}, /*argTypes=*/{})) {
    return failure();
  }

  // Adds operand segment sizes.
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(read_operands.size()),
           static_cast<int32_t>(write_operands.size()),
           static_cast<int32_t>(value_operands.size()),
           static_cast<int32_t>(original_read_operands.size()),
           static_cast<int32_t>(original_write_operands.size())}));

  // Adds result segment sizes. Read/write outputs are inferred from the
  // terminator, because multiple will_writes states may map to a single
  // done_writes state.
  size_t num_read_outputs = 0;
  size_t num_write_outputs = 0;
  if (!body->empty()) {
    if (auto yield_op =
            dyn_cast<TaskflowYieldOp>(body->front().getTerminator())) {
      num_read_outputs = yield_op.getDoneReads().size();
      num_write_outputs = yield_op.getDoneWrites().size();
    }
  }

  size_t num_value_outputs = 0;
  size_t total_memref_results = 0;
  for (Type t : func_type.getResults()) {
    if (isa<MemRefType>(t)) {
      total_memref_results++;
    } else {
      num_value_outputs++;
    }
  }
  if (total_memref_results != num_read_outputs + num_write_outputs) {
    return parser.emitError(parser.getCurrentLocation(),
                            "taskflow.yield memref result count does not "
                            "match task function result type");
  }
  result.addAttribute("resultSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(num_read_outputs),
                           static_cast<int32_t>(num_write_outputs),
                           static_cast<int32_t>(num_value_outputs)}));

  return success();
}

void TaskflowTaskOp::print(OpAsmPrinter &printer) {
  // Prints task name.
  printer << " @" << getTaskName();

  // Prints will_reads.
  if (!getWillReads().empty()) {
    printer << " will_reads(";
    llvm::interleaveComma(getWillReads(), printer);
    printer << " : ";
    llvm::interleaveComma(getWillReads().getTypes(), printer);
    printer << ")";
  }

  // Prints will_writes.
  if (!getWillWrites().empty()) {
    printer << " will_writes(";
    llvm::interleaveComma(getWillWrites(), printer);
    printer << " : ";
    llvm::interleaveComma(getWillWrites().getTypes(), printer);
    printer << ")";
  }

  // Prints value_inputs.
  if (!getValueInputs().empty()) {
    printer << " value_inputs(";
    llvm::interleaveComma(getValueInputs(), printer);
    printer << " : ";
    llvm::interleaveComma(getValueInputs().getTypes(), printer);
    printer << ")";
  }

  // Prints original memrefs with types.
  if (!getOriginalReadMemrefs().empty() || !getOriginalWriteMemrefs().empty()) {
    printer << " [";

    if (!getOriginalReadMemrefs().empty()) {
      printer << "original_read_memrefs(";
      llvm::interleaveComma(getOriginalReadMemrefs(), printer);
      printer << " : ";
      llvm::interleaveComma(getOriginalReadMemrefs().getTypes(), printer);
      printer << ")";
    }

    if (!getOriginalReadMemrefs().empty() && !getOriginalWriteMemrefs().empty())
      printer << ", ";

    if (!getOriginalWriteMemrefs().empty()) {
      printer << "original_write_memrefs(";
      llvm::interleaveComma(getOriginalWriteMemrefs(), printer);
      printer << " : ";
      llvm::interleaveComma(getOriginalWriteMemrefs().getTypes(), printer);
      printer << ")";
    }

    printer << "]";
  }

  // Prints attributes (skip operandSegmentSizes, resultSegmentSizes,
  // task_name).
  SmallVector<StringRef> elidedAttrs = {"operandSegmentSizes",
                                        "resultSegmentSizes", "task_name"};
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // Prints function type.
  printer << " : (";
  llvm::interleaveComma(llvm::concat<const Type>(getWillReads().getTypes(),
                                                 getWillWrites().getTypes(),
                                                 getValueInputs().getTypes()),
                        printer);
  printer << ") -> (";
  llvm::interleaveComma(llvm::concat<const Type>(getDoneReads().getTypes(),
                                                 getDoneWrites().getTypes(),
                                                 getValueOutputs().getTypes()),
                        printer);
  printer << ")";

  // Prints region.
  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/true);
}

//===----------------------------------------------------------------------===//
// TaskflowYieldOp
//===----------------------------------------------------------------------===//

ParseResult TaskflowYieldOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> read_operands;
  SmallVector<Type> read_types;
  SmallVector<OpAsmParser::UnresolvedOperand> write_operands;
  SmallVector<Type> write_types;
  SmallVector<OpAsmParser::UnresolvedOperand> value_operands;
  SmallVector<Type> value_types;

  // Parses done_reads (WAR dependency passthrough).
  if (succeeded(parser.parseOptionalKeyword("done_reads"))) {
    if (parser.parseLParen() || parser.parseOperandList(read_operands) ||
        parser.parseColonTypeList(read_types) || parser.parseRParen())
      return failure();
  }

  // Parses done_writes.
  if (succeeded(parser.parseOptionalKeyword("done_writes"))) {
    if (parser.parseLParen() || parser.parseOperandList(write_operands) ||
        parser.parseColonTypeList(write_types) || parser.parseRParen())
      return failure();
  }

  // Parses values.
  if (succeeded(parser.parseOptionalKeyword("values"))) {
    if (parser.parseLParen() || parser.parseOperandList(value_operands) ||
        parser.parseColonTypeList(value_types) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(read_operands, read_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(write_operands, write_types,
                             parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperands(value_operands, value_types,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(read_operands.size()),
                           static_cast<int32_t>(write_operands.size()),
                           static_cast<int32_t>(value_operands.size())}));

  return success();
}

void TaskflowYieldOp::print(OpAsmPrinter &printer) {
  if (!getDoneReads().empty()) {
    printer << " done_reads(";
    llvm::interleaveComma(getDoneReads(), printer);
    printer << " : ";
    llvm::interleaveComma(getDoneReads().getTypes(), printer);
    printer << ")";
  }

  if (!getDoneWrites().empty()) {
    printer << " done_writes(";
    llvm::interleaveComma(getDoneWrites(), printer);
    printer << " : ";
    llvm::interleaveComma(getDoneWrites().getTypes(), printer);
    printer << ")";
  }

  if (!getValueResults().empty()) {
    printer << " values(";
    llvm::interleaveComma(getValueResults(), printer);
    printer << " : ";
    llvm::interleaveComma(getValueResults().getTypes(), printer);
    printer << ")";
  }
}
