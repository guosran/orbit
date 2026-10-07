//===- FissionTaskPass.cpp - Canonical hyperblock DFG fission ---*- C++ -*-===//
//
// Splits a hyperblock DFG at a stable predecessor-closed node partition and
// materializes every live scalar cut edge in a static tensor indexed by the
// unchanged counter domain.
// The two derived tasks therefore retain canonical counter/hyperblock form,
// and their temporary write/read state is an explicit Taskflow RAW edge.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"
#include "Backend/Neura/Transforms/Optimizations/TaskflowFission.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

constexpr StringLiteral kParentTaskAttr = "amoeba.fission.parent_task";
constexpr StringLiteral kLeftNodesAttr = "amoeba.fission.left_nodes";
constexpr StringLiteral kPartIndexAttr = "amoeba.fission.part_index";
constexpr StringLiteral kControlPredecessorsAttr =
    "amoeba.control_predecessors";
constexpr StringLiteral kNoAliasAttr = "amoeba.noalias";
constexpr StringLiteral kSourceIterationDomainAttr =
    "amoeba.source_iteration_domain";
constexpr StringLiteral kSourceIterationCapturePendingAttr =
    "amoeba.source_iteration_capture_pending";
constexpr StringLiteral kSourceIterationControlBindingAttr =
    "amoeba.source_iteration_control_binding";
constexpr StringLiteral kSourceIterationSourceControlBindingAttr =
    "amoeba.source_iteration_source_control_binding";
constexpr StringLiteral kDlpReplicableAttr = "dlp_replicable";
constexpr StringLiteral kRuntimeManageableAttr = "runtime_managable";

struct FissionPlan {
  TaskflowTaskOp task;
  SmallVector<TaskflowCounterOp> counters;
  TaskflowHyperblockOp hyperblock;
  TaskflowYieldOp taskYield;
  SmallVector<Operation *> nodes;
  SmallVector<Value> interfaceValues;
  DenseMap<Value, unsigned> producers;
  SmallVector<SmallVector<unsigned>> predecessors;
  SmallVector<int64_t> shape;
  SmallVector<unsigned> leftNodes;
  SmallVector<unsigned> rightNodes;
};

static std::optional<int64_t> constantIndex(Value value, TaskflowTaskOp task) {
  DenseSet<Value> active;
  return mlir::amoeba::neura::joint_scheduling::sourceStaticIndex(value, task,
                                                                active);
}

// The source capture is pending until the fixed Taskflow-to-Neura lowering
// binds it to the current control/body. Fission may duplicate that pending
// capture only when its represented axes are independently rederived from the
// exact Taskflow counter bounds that remain in both children. Internal axes
// and already-bound certificates are outside this canonical source form.
static LogicalResult validatePendingSourceIterationDomain(
    TaskflowTaskOp task, ArrayRef<TaskflowCounterOp> counters,
    std::string &error) {
  bool hasDomain = task->hasAttr(kSourceIterationDomainAttr);
  bool hasPending = task->hasAttr(kSourceIterationCapturePendingAttr);
  if (!hasDomain && !hasPending)
    return success();
  if (!isa_and_nonnull<DictionaryAttr>(
          task->getAttr(kSourceIterationDomainAttr)) ||
      !isa_and_nonnull<UnitAttr>(
          task->getAttr(kSourceIterationCapturePendingAttr)) ||
      task->hasAttr(kSourceIterationControlBindingAttr) ||
      task->hasAttr(kSourceIterationSourceControlBindingAttr)) {
    error = "fission requires an unbound pending source iteration-domain "
            "capture";
    return failure();
  }

  FailureOr<mlir::amoeba::neura::joint_scheduling::SourceIterationDomainInfo>
      source = mlir::amoeba::neura::joint_scheduling::parseSourceIterationDomain(
          task, error);
  if (failed(source))
    return failure();
  auto &info = *source;
  if (!info.complete || info.axes.size() != counters.size() ||
      info.internalMultiplicity != 1) {
    error = "fission source iteration-domain capture does not describe only "
            "the current Taskflow counters";
    return failure();
  }

  int64_t representedMultiplicity = 1;
  for (auto [index, counterHandle] : llvm::enumerate(counters)) {
    TaskflowCounterOp counter = counterHandle;
    std::optional<int64_t> lower = constantIndex(counter.getLowerBound(), task);
    std::optional<int64_t> upper = constantIndex(counter.getUpperBound(), task);
    std::optional<int64_t> step = constantIndex(counter.getStep(), task);
    const auto &axis = info.axes[index];
    if (!lower || !upper || !step || *step <= 0 || *upper <= *lower ||
        (*lower < 0 &&
         *upper > std::numeric_limits<int64_t>::max() + *lower)) {
      error = "fission cannot rederive a static captured Taskflow counter axis";
      return failure();
    }
    int64_t distance = *upper - *lower;
    int64_t extent = 1 + (distance - 1) / *step;
    if (!axis.representedByTaskflow || axis.ordinal != index ||
        axis.lower != *lower || axis.upper != *upper || axis.step != *step ||
        axis.extent != extent || axis.expandedInsideMapperFiring ||
        axis.parentCounterOrdinal != -1 || axis.carriedValues != 0 ||
        axis.resultUses != 0 ||
        !mlir::amoeba::neura::joint_scheduling::checkedMultiply(
            representedMultiplicity, extent, representedMultiplicity)) {
      error = "fission source iteration-domain capture disagrees with the "
              "current Taskflow counter chain";
      return failure();
    }
  }
  if (representedMultiplicity != info.representedMultiplicity ||
      representedMultiplicity != info.sourceMultiplicity) {
    error = "fission source iteration-domain multiplicity does not match "
            "the current Taskflow counter chain";
    return failure();
  }
  return success();
}

static bool isStructuralOrSafeTaskAttribute(NamedAttribute attribute) {
  StringRef name = attribute.getName().strref();
  if (name == kDlpReplicableAttr || name == kRuntimeManageableAttr)
    return isa<BoolAttr>(attribute.getValue());
  if (name == kSourceIterationDomainAttr)
    return isa<DictionaryAttr>(attribute.getValue());
  if (name == kSourceIterationCapturePendingAttr)
    return isa<UnitAttr>(attribute.getValue());
  return name == "task_name" || name == "operandSegmentSizes" ||
         name == "resultSegmentSizes" || name == "operand_segment_sizes" ||
         name == "result_segment_sizes";
}

static bool isStorableScalar(Type type) { return type.isIntOrIndexOrFloat(); }

static bool hasDirectIndices(ValueRange indices, Block &hyperblockBody) {
  if (indices.size() != hyperblockBody.getNumArguments())
    return false;
  for (auto [index, value] : llvm::enumerate(indices))
    if (value != hyperblockBody.getArgument(index))
      return false;
  return true;
}

static Value canonicalStorageRoot(Value value) {
  DenseSet<Operation *> visited;
  while (auto cast = value.getDefiningOp<memref::CastOp>()) {
    if (!visited.insert(cast.getOperation()).second)
      return Value();
    value = cast.getSource();
  }
  return value;
}

static bool isNoAliasFunctionArgument(Value value) {
  auto argument = dyn_cast<BlockArgument>(canonicalStorageRoot(value));
  if (!argument)
    return false;
  auto function =
      dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front())
    return false;
  Attribute proof = function.getArgAttr(argument.getArgNumber(), kNoAliasAttr);
  if (auto boolean = dyn_cast_or_null<BoolAttr>(proof))
    return boolean.getValue();
  return isa_and_nonnull<UnitAttr>(proof);
}

static bool provesDistinctStorage(Value lhs, Value rhs) {
  lhs = canonicalStorageRoot(lhs);
  rhs = canonicalStorageRoot(rhs);
  if (!lhs || !rhs || lhs == rhs)
    return false;
  Operation *lhsDefinition = lhs.getDefiningOp();
  Operation *rhsDefinition = rhs.getDefiningOp();
  bool lhsAllocation =
      lhsDefinition && isa<memref::AllocOp, memref::AllocaOp>(lhsDefinition);
  bool rhsAllocation =
      rhsDefinition && isa<memref::AllocOp, memref::AllocaOp>(rhsDefinition);
  if (lhsAllocation && rhsAllocation)
    return lhsDefinition != rhsDefinition;
  if (lhsAllocation && isNoAliasFunctionArgument(rhs))
    return true;
  if (rhsAllocation && isNoAliasFunctionArgument(lhs))
    return true;
  return isNoAliasFunctionArgument(lhs) && isNoAliasFunctionArgument(rhs);
}

static bool isNamedControlPredecessor(TaskflowTaskOp task) {
  auto function = task->getParentOfType<func::FuncOp>();
  if (!function || task->hasAttr(kControlPredecessorsAttr))
    return true;
  bool referenced = false;
  function.walk([&](TaskflowTaskOp consumer) {
    Attribute attribute = consumer->getAttr(kControlPredecessorsAttr);
    if (!attribute)
      return;
    auto names = dyn_cast<ArrayAttr>(attribute);
    if (!names) {
      referenced = true;
      return;
    }
    for (Attribute nameAttribute : names) {
      StringRef name;
      if (auto string = dyn_cast<StringAttr>(nameAttribute))
        name = string.getValue();
      else if (auto symbol = dyn_cast<FlatSymbolRefAttr>(nameAttribute))
        name = symbol.getValue();
      else {
        referenced = true;
        return;
      }
      referenced |= name == task.getTaskName();
    }
  });
  return referenced;
}

static bool hasFissionTaskNameCollision(TaskflowTaskOp task) {
  auto function = task->getParentOfType<func::FuncOp>();
  if (!function)
    return true;
  std::string prefixName = (Twine(task.getTaskName()) + ".split.0").str();
  std::string suffixName = (Twine(task.getTaskName()) + ".split.1").str();
  bool collision = false;
  function.walk([&](TaskflowTaskOp other) {
    collision |= other != task && (other.getTaskName() == prefixName ||
                                   other.getTaskName() == suffixName);
  });
  return collision;
}

static FailureOr<FissionPlan> analyzeFission(TaskflowTaskOp task,
                                             std::string &error) {
  FissionPlan plan;
  plan.task = task;
  if (!task || !task.getBody().hasOneBlock()) {
    error = "fission requires a selected single-block task";
    return failure();
  }
  if (isNamedControlPredecessor(task)) {
    error = "fission rejects tasks named by unsupported control edges";
    return failure();
  }
  if (hasFissionTaskNameCollision(task)) {
    error = "fission-derived task name collides with an existing task";
    return failure();
  }
  if (!task.getDoneReads().empty() || task.getWillWrites().size() != 1 ||
      task.getDoneWrites().size() != 1 || !task.getValueOutputs().empty() ||
      task.getOriginalWriteMemrefs().size() != 1 ||
      task.getOriginalReadMemrefs().size() != task.getWillReads().size()) {
    error = "fission requires one tensor write result, matching read "
            "provenance, and no read/value result";
    return failure();
  }
  for (NamedAttribute attribute : task->getAttrs())
    if (!isStructuralOrSafeTaskAttribute(attribute)) {
      error = "fission rejects unsupported task state attribute " +
              attribute.getName().str();
      return failure();
    }
  for (Value input : task.getOriginalReadMemrefs())
    for (Value output : task.getOriginalWriteMemrefs())
      if (!provesDistinctStorage(input, output)) {
        error = "fission requires a proof that each read input is disjoint "
                "from the task output";
        return failure();
      }

  for (Operation &operation : task.getBody().front()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      plan.counters.push_back(counter);
    } else if (auto hyperblock = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      if (plan.hyperblock) {
        error = "fission requires exactly one hyperblock";
        return failure();
      }
      plan.hyperblock = hyperblock;
    } else if (auto yield = dyn_cast<TaskflowYieldOp>(&operation)) {
      plan.taskYield = yield;
    } else if (!isa<arith::ConstantOp>(&operation)) {
      error = "fission requires canonical counters and one hyperblock";
      return failure();
    }
  }
  if (!plan.hyperblock || !plan.taskYield || plan.counters.empty() ||
      !plan.hyperblock.getIterArgs().empty() ||
      plan.hyperblock.getNumResults() != 0 ||
      plan.hyperblock.getIndices().size() != plan.counters.size()) {
    error = "fission rejects recurrence or a non-canonical counter domain";
    return failure();
  }
  Block &taskBody = task.getBody().front();
  size_t expectedArguments = task.getWillReads().size() +
                             task.getWillWrites().size() +
                             task.getValueInputs().size();
  if (taskBody.getNumArguments() != expectedArguments ||
      !plan.taskYield.getDoneReads().empty() ||
      plan.taskYield.getDoneWrites().size() != 1 ||
      plan.taskYield.getDoneWrites().front() !=
          taskBody.getArgument(task.getWillReads().size()) ||
      !plan.taskYield.getValueResults().empty()) {
    error = "fission requires task operands and yield to directly mirror "
            "tensor state";
    return failure();
  }
  for (auto [index, counter] : llvm::enumerate(plan.counters)) {
    std::optional<int64_t> lower = constantIndex(counter.getLowerBound(), task);
    std::optional<int64_t> upper = constantIndex(counter.getUpperBound(), task);
    std::optional<int64_t> step = constantIndex(counter.getStep(), task);
    if (!lower || !upper || !step || *lower != 0 || *step != 1 ||
        *upper <= *lower || (index == 0 && counter.getParentIndex()) ||
        (index != 0 && counter.getParentIndex() !=
                           plan.counters[index - 1].getCounterIndex()) ||
        plan.hyperblock.getIndices()[index] != counter.getCounterIndex()) {
      error = "fission requires a zero-based static unit-step counter chain";
      return failure();
    }
    plan.shape.push_back(*upper);
  }
  if (failed(validatePendingSourceIterationDomain(task, plan.counters, error)))
    return failure();

  Block &hyperblockBody = plan.hyperblock.getBody().front();
  DenseSet<Value> arguments(hyperblockBody.getArguments().begin(),
                            hyperblockBody.getArguments().end());
  DenseSet<Value> scaffoldConstants;
  for (Operation &operation : taskBody)
    if (isa<arith::ConstantOp>(&operation))
      for (Value result : operation.getResults())
        scaffoldConstants.insert(result);
  for (Operation &operation : hyperblockBody.without_terminator()) {
    if (operation.getNumRegions() != 0 ||
        (!isa<memref::LoadOp, memref::StoreOp>(&operation) &&
         (!isPure(&operation) || !isMemoryEffectFree(&operation)))) {
      error = "fission rejects nested regions and unknown effects";
      return failure();
    }
    if (auto load = dyn_cast<memref::LoadOp>(&operation)) {
      bool knownRead = false;
      for (size_t index = 0; index < task.getWillReads().size(); ++index)
        knownRead |= load.getMemRef() == taskBody.getArgument(index);
      if (!knownRead) {
        error = "fission requires loads from declared tensor read inputs";
        return failure();
      }
    }
    if (auto store = dyn_cast<memref::StoreOp>(&operation)) {
      if (store.getMemRef() !=
              taskBody.getArgument(task.getWillReads().size()) ||
          !hasDirectIndices(store.getIndices(), hyperblockBody)) {
        error = "fission requires direct counter-indexed tensor stores";
        return failure();
      }
    }
    unsigned nodeIndex = plan.nodes.size();
    for (Value operand : operation.getOperands()) {
      if (arguments.contains(operand) || scaffoldConstants.contains(operand))
        continue;
      if (auto argument = dyn_cast<BlockArgument>(operand)) {
        if (argument.getOwner() != &taskBody) {
          error = "fission rejects a value captured from outside the task";
          return failure();
        }
        continue;
      }
      auto producer = plan.producers.find(operand);
      if (producer == plan.producers.end() || producer->second >= nodeIndex) {
        error = "fission rejects a backward or unknown DFG edge";
        return failure();
      }
    }
    for (Value result : operation.getResults())
      plan.producers[result] = nodeIndex;
    plan.nodes.push_back(&operation);
  }
  if (plan.nodes.size() < 2) {
    error = "fission requires at least two hyperblock DFG operations";
    return failure();
  }
  plan.predecessors.resize(plan.nodes.size());
  for (auto [index, node] : llvm::enumerate(plan.nodes)) {
    for (Value operand : node->getOperands()) {
      auto producer = plan.producers.find(operand);
      if (producer != plan.producers.end())
        plan.predecessors[index].push_back(producer->second);
    }
    llvm::sort(plan.predecessors[index]);
    plan.predecessors[index].erase(std::unique(plan.predecessors[index].begin(),
                                               plan.predecessors[index].end()),
                                   plan.predecessors[index].end());
  }
  return plan;
}

static bool taskArgumentAvailableInStage(Value value, TaskflowTaskOp task,
                                         bool prefix) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &task.getBody().front())
    return false;
  unsigned index = argument.getArgNumber();
  if (index < task.getWillReads().size())
    return true;
  index -= task.getWillReads().size();
  if (index < task.getWillWrites().size())
    return !prefix;
  index -= task.getWillWrites().size();
  return index < task.getValueInputs().size();
}

static LogicalResult validatePartition(FissionPlan &plan,
                                       ArrayRef<unsigned> leftNodes,
                                       std::string &error) {
  plan.leftNodes.clear();
  plan.rightNodes.clear();
  plan.interfaceValues.clear();
  plan.leftNodes.append(leftNodes.begin(), leftNodes.end());
  if (leftNodes.empty() || leftNodes.size() >= plan.nodes.size()) {
    error =
        "left-nodes must select a non-empty proper hyperblock DFG partition";
    return failure();
  }
  SmallVector<bool> isLeft(plan.nodes.size(), false);
  unsigned previous = 0;
  for (auto [position, index] : llvm::enumerate(leftNodes)) {
    if (index >= plan.nodes.size() || (position != 0 && index <= previous)) {
      error = "left-nodes must be unique increasing hyperblock node ordinals";
      return failure();
    }
    previous = index;
    isLeft[index] = true;
  }
  Block &taskBody = plan.task.getBody().front();
  for (unsigned index = 0; index < plan.nodes.size(); ++index) {
    if (!isLeft[index]) {
      plan.rightNodes.push_back(index);
      continue;
    }
    Operation *node = plan.nodes[index];
    if (isa<memref::StoreOp>(node)) {
      error = "fission rejects a cut after an externally visible memory write";
      return failure();
    }
    for (Value operand : node->getOperands()) {
      if (auto argument = dyn_cast<BlockArgument>(operand)) {
        if (argument.getOwner() == &plan.hyperblock.getBody().front())
          continue;
        if (argument.getOwner() != &taskBody ||
            !taskArgumentAvailableInStage(operand, plan.task, true)) {
          error = "fission prefix depends on task state unavailable before "
                  "the output write";
          return failure();
        }
      }
      auto producer = plan.producers.find(operand);
      if (producer != plan.producers.end() && !isLeft[producer->second]) {
        error = "left-nodes must be predecessor closed";
        return failure();
      }
    }
  }
  if (!llvm::any_of(plan.rightNodes, [&](unsigned index) {
        return isa<memref::StoreOp>(plan.nodes[index]);
      })) {
    error = "fission suffix must retain the tensor write";
    return failure();
  }
  for (unsigned index : plan.rightNodes)
    for (Value operand : plan.nodes[index]->getOperands()) {
      if (auto argument = dyn_cast<BlockArgument>(operand)) {
        if (argument.getOwner() != &plan.hyperblock.getBody().front() &&
            (argument.getOwner() != &taskBody ||
             !taskArgumentAvailableInStage(operand, plan.task, false))) {
          error = "fission suffix depends on task state outside its inputs";
          return failure();
        }
      }
      auto producer = plan.producers.find(operand);
      if (producer != plan.producers.end() && isLeft[producer->second]) {
        if (!isStorableScalar(operand.getType())) {
          error = "fission cut edge has a non-storable value type";
          return failure();
        }
        if (!llvm::is_contained(plan.interfaceValues, operand))
          plan.interfaceValues.push_back(operand);
      }
    }
  if (plan.interfaceValues.empty()) {
    error = "fission cut has no live DFG edge";
    return failure();
  }
  return success();
}

static void setLineage(TaskflowTaskOp source, TaskflowTaskOp target,
                       OpBuilder &builder, ArrayRef<unsigned> leftNodes,
                       unsigned partIndex) {
  for (StringRef name : {kDlpReplicableAttr, kRuntimeManageableAttr,
                         kSourceIterationDomainAttr,
                         kSourceIterationCapturePendingAttr})
    if (Attribute value = source->getAttr(name))
      target->setAttr(name, value);
  target->setAttr(kParentTaskAttr, builder.getStringAttr(source.getTaskName()));
  SmallVector<int64_t> ordinals(leftNodes.begin(), leftNodes.end());
  target->setAttr(kLeftNodesAttr, builder.getDenseI64ArrayAttr(ordinals));
  target->setAttr(kPartIndexAttr, builder.getI64IntegerAttr(partIndex));
}

static void cloneScaffold(FissionPlan &plan, TaskflowTaskOp target,
                          IRMapping &mapping,
                          SmallVectorImpl<TaskflowCounterOp> &counters,
                          OpBuilder &builder) {
  Block &sourceBody = plan.task.getBody().front();
  for (Operation &operation : sourceBody) {
    if (isa<TaskflowHyperblockOp, TaskflowYieldOp>(&operation))
      continue;
    Operation *clone = builder.clone(operation, mapping);
    if (auto counter = dyn_cast<TaskflowCounterOp>(clone))
      counters.push_back(counter);
  }
}

static TaskflowHyperblockOp
createHyperblock(FissionPlan &plan, ArrayRef<TaskflowCounterOp> counters,
                 IRMapping &mapping, OpBuilder &builder) {
  SmallVector<Value> indices;
  llvm::transform(
      counters, std::back_inserter(indices),
      [](TaskflowCounterOp counter) { return counter.getCounterIndex(); });
  auto hyperblock = builder.create<TaskflowHyperblockOp>(
      plan.task.getLoc(), TypeRange{}, indices, ValueRange{});
  Block *body = &hyperblock.getBody().emplaceBlock();
  for (BlockArgument sourceArgument :
       plan.hyperblock.getBody().front().getArguments()) {
    BlockArgument argument =
        body->addArgument(sourceArgument.getType(), plan.task.getLoc());
    mapping.map(sourceArgument, argument);
  }
  return hyperblock;
}

static TaskflowTaskOp createPrefix(FissionPlan &plan, OpBuilder &builder,
                                   ValueRange temporaries) {
  SmallVector<Type> resultTypes(temporaries.getTypes());
  SmallVector<Value> originalWrites(temporaries);
  auto prefix = builder.create<TaskflowTaskOp>(
      plan.task.getLoc(), TypeRange{}, resultTypes, TypeRange{},
      plan.task.getWillReads(), temporaries, plan.task.getValueInputs(),
      builder.getStringAttr(
          (Twine(plan.task.getTaskName()) + ".split.0").str()),
      plan.task.getOriginalReadMemrefs(), originalWrites);
  Block *body = new Block();
  prefix.getBody().push_back(body);
  IRMapping mapping;
  SmallVector<Value> operands(plan.task.getWillReads());
  llvm::append_range(operands, temporaries);
  llvm::append_range(operands, plan.task.getValueInputs());
  for (Value operand : operands)
    body->addArgument(operand.getType(), plan.task.getLoc());
  Block &sourceBody = plan.task.getBody().front();
  for (size_t index = 0; index < plan.task.getWillReads().size(); ++index)
    mapping.map(sourceBody.getArgument(index), body->getArgument(index));
  size_t sourceValueOffset =
      plan.task.getWillReads().size() + plan.task.getWillWrites().size();
  size_t targetValueOffset =
      plan.task.getWillReads().size() + temporaries.size();
  for (size_t index = 0; index < plan.task.getValueInputs().size(); ++index)
    mapping.map(sourceBody.getArgument(sourceValueOffset + index),
                body->getArgument(targetValueOffset + index));

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  SmallVector<TaskflowCounterOp> counters;
  cloneScaffold(plan, prefix, mapping, counters, bodyBuilder);
  auto hyperblock = createHyperblock(plan, counters, mapping, bodyBuilder);
  Block &hyperblockBody = hyperblock.getBody().front();
  OpBuilder hyperblockBuilder = OpBuilder::atBlockEnd(&hyperblockBody);
  for (unsigned index : plan.leftNodes)
    hyperblockBuilder.clone(*plan.nodes[index], mapping);
  for (auto [index, interfaceValue] : llvm::enumerate(plan.interfaceValues)) {
    Value temporary =
        body->getArgument(plan.task.getWillReads().size() + index);
    hyperblockBuilder.create<memref::StoreOp>(
        plan.task.getLoc(), mapping.lookup(interfaceValue), temporary,
        hyperblockBody.getArguments());
  }
  hyperblockBuilder.create<TaskflowHyperblockYieldOp>(plan.task.getLoc());
  bodyBuilder.setInsertionPointToEnd(body);
  SmallVector<Value> yields;
  for (size_t index = 0; index < temporaries.size(); ++index)
    yields.push_back(
        body->getArgument(plan.task.getWillReads().size() + index));
  bodyBuilder.create<TaskflowYieldOp>(plan.task.getLoc(), ValueRange{}, yields,
                                      ValueRange{});
  setLineage(plan.task, prefix, builder, plan.leftNodes, 0);
  return prefix;
}

static TaskflowTaskOp createSuffix(FissionPlan &plan, OpBuilder &builder,
                                   TaskflowTaskOp prefix) {
  SmallVector<Value> reads(prefix.getDoneWrites());
  llvm::append_range(reads, plan.task.getWillReads());
  SmallVector<Value> originalReads(prefix.getWillWrites());
  llvm::append_range(originalReads, plan.task.getOriginalReadMemrefs());
  auto suffix = builder.create<TaskflowTaskOp>(
      plan.task.getLoc(), TypeRange{}, plan.task.getDoneWrites().getTypes(),
      TypeRange{}, reads, plan.task.getWillWrites(), plan.task.getValueInputs(),
      builder.getStringAttr(
          (Twine(plan.task.getTaskName()) + ".split.1").str()),
      originalReads, plan.task.getOriginalWriteMemrefs());
  Block *body = new Block();
  suffix.getBody().push_back(body);
  for (Value operand : suffix->getOperands()) {
    if (body->getNumArguments() == reads.size() +
                                       plan.task.getWillWrites().size() +
                                       plan.task.getValueInputs().size())
      break;
    body->addArgument(operand.getType(), plan.task.getLoc());
  }
  IRMapping mapping;
  Block &sourceBody = plan.task.getBody().front();
  size_t temporaryCount = prefix.getDoneWrites().size();
  for (size_t index = 0; index < plan.task.getWillReads().size(); ++index)
    mapping.map(sourceBody.getArgument(index),
                body->getArgument(temporaryCount + index));
  size_t sourceWriteOffset = plan.task.getWillReads().size();
  size_t targetWriteOffset = reads.size();
  mapping.map(sourceBody.getArgument(sourceWriteOffset),
              body->getArgument(targetWriteOffset));
  size_t sourceValueOffset =
      sourceWriteOffset + plan.task.getWillWrites().size();
  size_t targetValueOffset =
      targetWriteOffset + plan.task.getWillWrites().size();
  for (size_t index = 0; index < plan.task.getValueInputs().size(); ++index)
    mapping.map(sourceBody.getArgument(sourceValueOffset + index),
                body->getArgument(targetValueOffset + index));

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  SmallVector<TaskflowCounterOp> counters;
  cloneScaffold(plan, suffix, mapping, counters, bodyBuilder);
  auto hyperblock = createHyperblock(plan, counters, mapping, bodyBuilder);
  Block &hyperblockBody = hyperblock.getBody().front();
  OpBuilder hyperblockBuilder = OpBuilder::atBlockEnd(&hyperblockBody);
  for (auto [index, interfaceValue] : llvm::enumerate(plan.interfaceValues)) {
    Value loaded = hyperblockBuilder.create<memref::LoadOp>(
        plan.task.getLoc(), body->getArgument(index),
        hyperblockBody.getArguments());
    mapping.map(interfaceValue, loaded);
  }
  for (unsigned index : plan.rightNodes)
    hyperblockBuilder.clone(*plan.nodes[index], mapping);
  hyperblockBuilder.create<TaskflowHyperblockYieldOp>(plan.task.getLoc());
  bodyBuilder.setInsertionPointToEnd(body);
  Value output = body->getArgument(reads.size());
  bodyBuilder.create<TaskflowYieldOp>(plan.task.getLoc(), ValueRange{},
                                      ValueRange{output}, ValueRange{});
  setLineage(plan.task, suffix, builder, plan.leftNodes, 1);
  return suffix;
}

static LogicalResult rewriteFission(FissionPlan &plan, func::FuncOp function) {
  std::string prefixName = (Twine(plan.task.getTaskName()) + ".split.0").str();
  std::string suffixName = (Twine(plan.task.getTaskName()) + ".split.1").str();
  bool collision = false;
  function.walk([&](TaskflowTaskOp task) {
    collision |= task != plan.task && (task.getTaskName() == prefixName ||
                                       task.getTaskName() == suffixName);
  });
  if (collision) {
    plan.task.emitError(
        "fission-derived task name collides with an existing task");
    return failure();
  }

  OpBuilder builder(plan.task);
  SmallVector<Value> temporaries;
  for (Value interfaceValue : plan.interfaceValues) {
    auto type = MemRefType::get(plan.shape, interfaceValue.getType());
    temporaries.push_back(
        builder.create<memref::AllocOp>(plan.task.getLoc(), type));
  }
  TaskflowTaskOp prefix = createPrefix(plan, builder, temporaries);
  TaskflowTaskOp suffix = createSuffix(plan, builder, prefix);
  for (auto [oldResult, newResult] :
       llvm::zip(plan.task.getDoneWrites(), suffix.getDoneWrites()))
    oldResult.replaceAllUsesWith(newResult);
  plan.task.erase();
  return success();
}

} // namespace

static func::FuncOp findUniqueFunction(ModuleOp module, StringRef functionName,
                                       std::string &error) {
  if (functionName.empty()) {
    error = "fission requires a non-empty function name";
    return {};
  }
  Operation *symbol = SymbolTable::lookupSymbolIn(module, functionName);
  auto function = dyn_cast_or_null<func::FuncOp>(symbol);
  if (!function) {
    error = "fission function symbol was not found or is not func.func";
    return {};
  }
  return function;
}

static TaskflowTaskOp findUniqueTask(func::FuncOp function, StringRef taskName,
                                     std::string &error) {
  SmallVector<TaskflowTaskOp> matches;
  function.walk([&](TaskflowTaskOp task) {
    if (task.getTaskName() == taskName)
      matches.push_back(task);
  });
  if (matches.size() != 1) {
    error = "fission requires one task with the selected exact task name";
    return {};
  }
  return matches.front();
}

static std::string genericModuleBytes(ModuleOp module) {
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  OpPrintingFlags flags;
  flags.printGenericOpForm();
  module.print(stream, flags);
  stream.flush();
  return bytes;
}

FailureOr<SmallVector<SmallVector<unsigned>>>
mlir::amoeba::neura::enumerateTaskflowFissionPartitions(TaskflowTaskOp task,
                                                        uint64_t maxCuts,
                                                        std::string &error) {
  error.clear();
  FailureOr<FissionPlan> analyzed = analyzeFission(task, error);
  if (failed(analyzed))
    return SmallVector<SmallVector<unsigned>>{};
  FissionPlan plan = std::move(*analyzed);

  SmallVector<SmallVector<unsigned>> partitions;
  SmallVector<unsigned> leftNodes;
  SmallVector<bool> selected(plan.nodes.size(), false);
  bool overflow = false;
  std::string rejectedCut;
  std::function<void(unsigned)> enumerate = [&](unsigned nodeIndex) {
    if (overflow)
      return;
    if (nodeIndex == plan.nodes.size()) {
      std::string validationError;
      if (failed(validatePartition(plan, leftNodes, validationError))) {
        if (rejectedCut.empty())
          rejectedCut = std::move(validationError);
        return;
      }
      if (partitions.size() == maxCuts) {
        overflow = true;
        return;
      }
      partitions.push_back(leftNodes);
      return;
    }

    enumerate(nodeIndex + 1);
    if (overflow || isa<memref::StoreOp>(plan.nodes[nodeIndex]))
      return;
    if (!llvm::all_of(plan.predecessors[nodeIndex], [&](unsigned predecessor) {
          return selected[predecessor];
        }))
      return;
    bool prefixInputsAvailable =
        llvm::all_of(plan.nodes[nodeIndex]->getOperands(), [&](Value operand) {
          auto argument = dyn_cast<BlockArgument>(operand);
          if (!argument ||
              argument.getOwner() == &plan.hyperblock.getBody().front())
            return true;
          return argument.getOwner() == &plan.task.getBody().front() &&
                 taskArgumentAvailableInStage(operand, plan.task, true);
        });
    if (!prefixInputsAvailable)
      return;
    selected[nodeIndex] = true;
    leftNodes.push_back(nodeIndex);
    enumerate(nodeIndex + 1);
    leftNodes.pop_back();
    selected[nodeIndex] = false;
  };
  enumerate(0);
  if (overflow) {
    error = "legal fission cut-set count exceeds the requested maximum";
    return failure();
  }
  std::sort(
      partitions.begin(), partitions.end(),
      [](const SmallVector<unsigned> &lhs, const SmallVector<unsigned> &rhs) {
        return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                            rhs.end());
      });
  if (partitions.empty())
    error = rejectedCut.empty() ? "task has no legal non-empty DFG cut"
                                : rejectedCut;
  return partitions;
}

LogicalResult mlir::amoeba::neura::materializeTaskflowFission(
    ModuleOp module, StringRef functionName, StringRef taskName,
    ArrayRef<unsigned> leftNodes, std::string &error) {
  error.clear();
  if (!module) {
    error = "fission requires a module";
    return failure();
  }
  func::FuncOp function = findUniqueFunction(module, functionName, error);
  if (!function)
    return failure();
  TaskflowTaskOp task = findUniqueTask(function, taskName, error);
  if (!task)
    return failure();
  FailureOr<FissionPlan> analyzed = analyzeFission(task, error);
  if (failed(analyzed))
    return failure();
  FissionPlan plan = std::move(*analyzed);
  if (failed(validatePartition(plan, leftNodes, error)))
    return failure();
  if (failed(rewriteFission(plan, function))) {
    if (error.empty())
      error = "fission materialization failed";
    return failure();
  }
  return success();
}

LogicalResult mlir::amoeba::neura::verifyTaskflowFissionReplay(
    ModuleOp canonicalParent, ModuleOp candidate,
    ArrayRef<TaskflowFissionAction> actions, std::string &error) {
  error.clear();
  if (!canonicalParent || !candidate) {
    error = "fission replay requires both parent and candidate modules";
    return failure();
  }
  if (failed(verify(canonicalParent))) {
    error = "fission replay parent module does not verify";
    return failure();
  }
  if (failed(verify(candidate))) {
    error = "fission replay candidate module does not verify";
    return failure();
  }

  OwningOpRef<ModuleOp> replay = canonicalParent.clone();
  SmallVector<std::pair<std::string, std::string>> seenActions;
  for (const TaskflowFissionAction &action : actions) {
    std::pair<std::string, std::string> key{action.function, action.task};
    if (llvm::is_contained(seenActions, key)) {
      error = "fission replay rejects repeated actions for one parent task";
      return failure();
    }
    seenActions.push_back(std::move(key));
    func::FuncOp parentFunction =
        findUniqueFunction(canonicalParent, action.function, error);
    if (!parentFunction || !findUniqueTask(parentFunction, action.task, error))
      return failure();
    if (failed(materializeTaskflowFission(*replay, action.function, action.task,
                                          action.leftNodes, error)))
      return failure();
  }

  if (genericModuleBytes(*replay) != genericModuleBytes(candidate)) {
    error = "candidate module is not the exact source fission replay";
    return failure();
  }
  return success();
}

namespace {

struct FissionTaskPass
    : public PassWrapper<FissionTaskPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FissionTaskPass)
  FissionTaskPass() = default;
  FissionTaskPass(const FissionTaskPass &other) : PassWrapper(other) {}
  StringRef getArgument() const override { return "fission-task"; }
  StringRef getDescription() const override {
    return "Split a canonical hyperblock DFG through static cut tensors";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }
  Option<std::string> taskName{*this, "task-name",
                               llvm::cl::desc("Exact Taskflow task name"),
                               llvm::cl::init("")};
  Option<std::string> leftNodes{
      *this, "left-nodes",
      llvm::cl::desc("Colon-separated predecessor-closed DFG node ordinals"),
      llvm::cl::init("")};

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isDeclaration())
      return;
    SmallVector<TaskflowTaskOp> matches;
    function.walk([&](TaskflowTaskOp task) {
      if (task.getTaskName() == taskName)
        matches.push_back(task);
    });
    SmallVector<unsigned> parsedLeftNodes;
    std::stringstream stream(leftNodes.getValue());
    std::string token;
    while (std::getline(stream, token, ':')) {
      if (token.empty() || !llvm::all_of(token, llvm::isDigit)) {
        function.emitError("fission-task left-nodes must be colon-separated "
                           "non-negative integers");
        return signalPassFailure();
      }
      uint64_t ordinal = 0;
      if (StringRef(token).getAsInteger(10, ordinal) ||
          ordinal > std::numeric_limits<unsigned>::max()) {
        function.emitError("fission-task left-nodes ordinal is out of range");
        return signalPassFailure();
      }
      parsedLeftNodes.push_back(static_cast<unsigned>(ordinal));
    }
    if (taskName.empty() || parsedLeftNodes.empty() || matches.size() != 1) {
      function.emitError()
          << "fission-task requires one task-name and non-empty left-nodes";
      return signalPassFailure();
    }
    std::string error;
    FailureOr<FissionPlan> plan = analyzeFission(matches.front(), error);
    if (failed(plan)) {
      matches.front().emitError() << error;
      return signalPassFailure();
    }
    FissionPlan fissionPlan = std::move(*plan);
    if (failed(validatePartition(fissionPlan, parsedLeftNodes, error))) {
      matches.front().emitError() << error;
      return signalPassFailure();
    }
    if (failed(rewriteFission(fissionPlan, function)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createFissionTaskPass() {
  return std::make_unique<FissionTaskPass>();
}
} // namespace mlir::amoeba::neura
