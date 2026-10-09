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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  SmallVector<unsigned> prefixOutputs;
  SmallVector<unsigned> suffixOutputs;
  SmallVector<unsigned> prefixCarries;
  SmallVector<unsigned> suffixCarries;
  scf::IfOp initialization;
  arith::CmpIOp initializationCondition;
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
  if (!task.getDoneReads().empty() || task.getWillWrites().empty() ||
      task.getDoneWrites().size() != task.getWillWrites().size() || !task.getValueOutputs().empty() ||
      task.getOriginalWriteMemrefs().size() != task.getWillWrites().size() ||
      task.getOriginalReadMemrefs().size() != task.getWillReads().size()) {
    error = "fission requires tensor write results, matching read/write "
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

  for (auto [index, output] : llvm::enumerate(task.getOriginalWriteMemrefs()))
    for (Value other : task.getOriginalWriteMemrefs().drop_front(index + 1))
      if (!provesDistinctStorage(output, other)) {
        error = "fission requires pairwise disjoint output storage";
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
      plan.hyperblock.getIndices().size() != plan.counters.size()) {
    error = "fission rejects recurrence or a non-canonical counter domain";
    return failure();
  }
  auto hyperYield = dyn_cast<TaskflowHyperblockYieldOp>(
      plan.hyperblock.getBody().front().getTerminator());
  if (!hyperYield || !plan.hyperblock->use_empty() ||
      plan.hyperblock.getNumResults() != plan.hyperblock.getIterArgs().size() ||
      hyperYield.getIterArgsNext().size() != plan.hyperblock.getIterArgs().size() ||
      !llvm::equal(hyperYield.getResults(), hyperYield.getIterArgsNext())) {
    error = "fission requires internal recurrence results identical to next-state values";
    return failure();
  }
  Block &taskBody = task.getBody().front();
  size_t expectedArguments = task.getWillReads().size() +
                             task.getWillWrites().size() +
                             task.getValueInputs().size();
  if (taskBody.getNumArguments() != expectedArguments ||
      !plan.taskYield.getDoneReads().empty() ||
      plan.taskYield.getDoneWrites().size() != task.getWillWrites().size() ||
      !llvm::equal(plan.taskYield.getDoneWrites(),
                   taskBody.getArguments().slice(task.getWillReads().size(),
                                                task.getWillWrites().size())) ||
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
  // Recognize the source's joint reduction reset as an effect scaffold.
  // It may be duplicated only after filtering disjoint output stores by owner;
  // no predicate, side effect or recurrence update is dropped.
  for (Operation &operation : hyperblockBody.without_terminator()) {
    auto conditional = dyn_cast<scf::IfOp>(&operation);
    if (!conditional) continue;
    auto compare = conditional.getCondition().getDefiningOp<arith::CmpIOp>();
    if (plan.initialization || conditional.getNumResults() ||
        !conditional.getElseRegion().empty() || !compare ||
        !compare.getResult().hasOneUse() ||
        !llvm::all_of(compare.getOperands(), [&](Value value) {
          return scaffoldConstants.contains(value) ||
                 llvm::is_contained(hyperblockBody.getArguments().take_front(plan.counters.size()), value);
        })) {
      error = "fission cannot partition conditional effects outside a counter-only output reset";
      return failure();
    }
    for (Operation &effect : conditional.getThenRegion().front().without_terminator()) {
      auto store = dyn_cast<memref::StoreOp>(&effect);
      auto outputs = taskBody.getArguments().slice(task.getWillReads().size(), task.getWillWrites().size());
      auto stored = store ? dyn_cast<BlockArgument>(store.getValueToStore()) : BlockArgument();
      if (!store || !llvm::is_contained(outputs, store.getMemRef()) ||
          !(scaffoldConstants.contains(store.getValueToStore()) ||
            (stored && stored.getOwner() == &taskBody &&
             stored.getArgNumber() >= task.getWillReads().size() + task.getWillWrites().size())) ||
          !llvm::all_of(store.getIndices(), [&](Value index) {
            return llvm::is_contained(hyperblockBody.getArguments().take_front(plan.counters.size()), index);
          })) {
        error = "fission conditional reset must only store invariant initial values to declared outputs";
        return failure();
      }
    }
    // The reset must precede the computational body. Its condition is the only
    // permitted preceding hyperblock operation, so re-emission preserves order.
    if (&hyperblockBody.front() != compare.getOperation() ||
        compare->getNextNode() != conditional.getOperation()) {
      error = "fission output reset must precede the complete DFG body";
      return failure();
    }
    plan.initialization = conditional;
    plan.initializationCondition = compare;
  }
  for (Operation &operation : hyperblockBody.without_terminator()) {
    if (&operation == plan.initialization.getOperation() ||
        &operation == plan.initializationCondition.getOperation()) continue;
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
      auto outputs = taskBody.getArguments().slice(task.getWillReads().size(),
                                                   task.getWillWrites().size());
      // A permutation is injective over the original Cartesian counter domain
      // and preserves source indices verbatim (e.g. Radar's output[j][i]).
      DenseSet<Value> indexSet(store.getIndices().begin(), store.getIndices().end());
      if (!llvm::is_contained(outputs, store.getMemRef()) ||
          store.getIndices().empty() ||
          indexSet.size() != store.getIndices().size() ||
          !llvm::all_of(store.getIndices(), [&](Value index) {
            return llvm::is_contained(hyperblockBody.getArguments().take_front(plan.counters.size()), index);
          }) ||
          (plan.hyperblock.getIterArgs().empty() && indexSet.size() != plan.counters.size())) {
        error = "fission requires stores to declared outputs indexed by a permutation of counters";
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
    return true; // Output ownership is checked for the complete partition.
  index -= task.getWillWrites().size();
  return index < task.getValueInputs().size();
}

static LogicalResult validatePartition(FissionPlan &plan,
                                       ArrayRef<unsigned> leftNodes,
                                       std::string &error) {
  plan.leftNodes.clear();
  plan.rightNodes.clear();
  plan.interfaceValues.clear();
  plan.prefixOutputs.clear();
  plan.suffixOutputs.clear();
  plan.prefixCarries.clear();
  plan.suffixCarries.clear();
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
  // Every public output has exactly one owner. Splitting writes to the same
  // tensor across stages would reorder observable effects and is rejected.
  SmallVector<int> outputOwners(plan.task.getWillWrites().size(), -1);
  for (auto [index, node] : llvm::enumerate(plan.nodes)) {
    auto store = dyn_cast<memref::StoreOp>(node);
    if (!store) continue;
    unsigned output = cast<BlockArgument>(store.getMemRef()).getArgNumber() -
                      plan.task.getWillReads().size();
    int owner = isLeft[index] ? 0 : 1;
    if (outputOwners[output] != -1 && outputOwners[output] != owner) {
      error = "fission output writes must have one partition owner";
      return failure();
    }
    outputOwners[output] = owner;
  }
  for (auto [index, owner] : llvm::enumerate(outputOwners)) {
    if (owner == -1) {
      error = "fission declared output has no owned store";
      return failure();
    }
    (owner == 0 ? plan.prefixOutputs : plan.suffixOutputs).push_back(index);
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
  // Loop-carried cycles must stay wholly within one stage. In particular a
  // left-to-right cut is not sufficient proof for a reverse recurrence edge.
  auto yield = cast<TaskflowHyperblockYieldOp>(plan.hyperblock.getBody().front().getTerminator());
  for (auto [index, next] : llvm::enumerate(yield.getIterArgsNext())) {
    auto producer = plan.producers.find(next);
    if (producer == plan.producers.end()) {
      error = "fission cannot assign an unproduced recurrence update";
      return failure();
    }
    bool owner = isLeft[producer->second];
    Value carried = plan.hyperblock.getBody().front().getArgument(plan.counters.size() + index);
    for (OpOperand &use : carried.getUses()) {
      auto found = llvm::find(plan.nodes, use.getOwner());
      if (found == plan.nodes.end() || isLeft[std::distance(plan.nodes.begin(), found)] != owner) {
        error = "fission partition crosses a loop-carried dependence";
        return failure();
      }
    }
    (owner ? plan.prefixCarries : plan.suffixCarries).push_back(index);
  }
  // A stage must not capture an output owned by its sibling through a pure
  // address operation or another operand outside the recognized store path.
  for (auto [index, node] : llvm::enumerate(plan.nodes))
    for (Value operand : node->getOperands()) {
      auto argument = dyn_cast<BlockArgument>(operand);
      if (!argument || argument.getOwner() != &taskBody) continue;
      unsigned ordinal = argument.getArgNumber();
      if (ordinal >= plan.task.getWillReads().size() &&
          ordinal < plan.task.getWillReads().size() + plan.task.getWillWrites().size() &&
          outputOwners[ordinal - plan.task.getWillReads().size()] != (isLeft[index] ? 0 : 1)) {
        error = "fission stage references an output owned by the other partition";
        return failure();
      }
    }
  if (plan.interfaceValues.empty() && plan.prefixOutputs.empty()) {
    error = "fission cut has neither a live DFG edge nor an independently owned output";
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
                 IRMapping &mapping, OpBuilder &builder,
                 ArrayRef<unsigned> carries) {
  SmallVector<Value> indices;
  llvm::transform(
      counters, std::back_inserter(indices),
      [](TaskflowCounterOp counter) { return counter.getCounterIndex(); });
  SmallVector<Value> initial;
  SmallVector<Type> results;
  for (unsigned index : carries) {
    initial.push_back(mapping.lookupOrDefault(plan.hyperblock.getIterArgs()[index]));
    results.push_back(plan.hyperblock.getResult(index).getType());
  }
  auto hyperblock = builder.create<TaskflowHyperblockOp>(
      plan.task.getLoc(), results, indices, initial);
  Block *body = &hyperblock.getBody().emplaceBlock();
  SmallVector<BlockArgument> arguments;
  auto sourceArguments = plan.hyperblock.getBody().front().getArguments();
  llvm::append_range(arguments, sourceArguments.take_front(plan.counters.size()));
  for (unsigned index : carries) arguments.push_back(sourceArguments[plan.counters.size() + index]);
  for (BlockArgument sourceArgument : arguments) {
    BlockArgument argument =
        body->addArgument(sourceArgument.getType(), plan.task.getLoc());
    mapping.map(sourceArgument, argument);
  }
  return hyperblock;
}

// A stage owns its original output states and, for the prefix, cut tensors.
// Only actual cut tensors create prefix->suffix dependencies. Independent
// disjoint-output branches have no manufactured global completion join.
static TaskflowTaskOp createStage(FissionPlan &plan, OpBuilder &builder,
                                  ValueRange temporaries, TaskflowTaskOp prefix,
                                  bool isPrefix) {
  auto &owned = isPrefix ? plan.prefixOutputs : plan.suffixOutputs;
  SmallVector<Value> reads, originalReads, writes, originalWrites;
  if (!isPrefix) {
    llvm::append_range(reads, prefix.getDoneWrites().take_front(temporaries.size()));
    llvm::append_range(originalReads, temporaries);
  }
  llvm::append_range(reads, plan.task.getWillReads());
  llvm::append_range(originalReads, plan.task.getOriginalReadMemrefs());
  if (isPrefix) {
    llvm::append_range(writes, temporaries);
    llvm::append_range(originalWrites, temporaries);
  }
  for (unsigned index : owned) {
    writes.push_back(plan.task.getWillWrites()[index]);
    originalWrites.push_back(plan.task.getOriginalWriteMemrefs()[index]);
  }
  SmallVector<Type> resultTypes;
  for (Value write : writes) resultTypes.push_back(write.getType());
  auto stage = builder.create<TaskflowTaskOp>(
      plan.task.getLoc(), TypeRange{}, resultTypes, TypeRange{}, reads, writes,
      plan.task.getValueInputs(), builder.getStringAttr(
          (Twine(plan.task.getTaskName()) + (isPrefix ? ".split.0" : ".split.1")).str()),
      originalReads, originalWrites);
  Block *body = new Block();
  stage.getBody().push_back(body);
  SmallVector<Value> operands(reads);
  llvm::append_range(operands, writes);
  llvm::append_range(operands, plan.task.getValueInputs());
  for (Value operand : operands) body->addArgument(operand.getType(), plan.task.getLoc());
  IRMapping mapping;
  Block &source = plan.task.getBody().front();
  unsigned inputOffset = isPrefix ? 0 : temporaries.size();
  for (unsigned index = 0; index < plan.task.getWillReads().size(); ++index)
    mapping.map(source.getArgument(index), body->getArgument(inputOffset + index));
  unsigned writeOffset = reads.size() + (isPrefix ? temporaries.size() : 0);
  for (auto [position, index] : llvm::enumerate(owned))
    mapping.map(source.getArgument(plan.task.getWillReads().size() + index),
                body->getArgument(writeOffset + position));
  unsigned sourceValues = plan.task.getWillReads().size() + plan.task.getWillWrites().size();
  for (unsigned index = 0; index < plan.task.getValueInputs().size(); ++index)
    mapping.map(source.getArgument(sourceValues + index),
                body->getArgument(reads.size() + writes.size() + index));
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  SmallVector<TaskflowCounterOp> counters;
  cloneScaffold(plan, stage, mapping, counters, bodyBuilder);
  auto &carries = isPrefix ? plan.prefixCarries : plan.suffixCarries;
  auto hyperblock = createHyperblock(plan, counters, mapping, bodyBuilder, carries);
  Block &hyperblockBody = hyperblock.getBody().front();
  OpBuilder kernelBuilder = OpBuilder::atBlockEnd(&hyperblockBody);
  if (plan.initialization) {
    Operation *compare = kernelBuilder.clone(*plan.initializationCondition, mapping);
    auto conditional = kernelBuilder.create<scf::IfOp>(plan.task.getLoc(), compare->getResult(0), false);
    OpBuilder resetBuilder = conditional.getThenBodyBuilder();
    for (Operation &effect : plan.initialization.getThenRegion().front().without_terminator()) {
      auto store = cast<memref::StoreOp>(effect);
      unsigned output = cast<BlockArgument>(store.getMemRef()).getArgNumber() - plan.task.getWillReads().size();
      if (llvm::is_contained(owned, output)) resetBuilder.clone(effect, mapping);
    }
  }
  if (!isPrefix)
    for (auto [index, interfaceValue] : llvm::enumerate(plan.interfaceValues)) {
      Value loaded = kernelBuilder.create<memref::LoadOp>(
          plan.task.getLoc(), body->getArgument(index), hyperblockBody.getArguments().take_front(plan.counters.size()));
      mapping.map(interfaceValue, loaded);
    }
  for (unsigned index : isPrefix ? plan.leftNodes : plan.rightNodes)
    kernelBuilder.clone(*plan.nodes[index], mapping);
  if (isPrefix)
    for (auto [index, interfaceValue] : llvm::enumerate(plan.interfaceValues))
      kernelBuilder.create<memref::StoreOp>(
          plan.task.getLoc(), mapping.lookup(interfaceValue),
          body->getArgument(reads.size() + index), hyperblockBody.getArguments().take_front(plan.counters.size()));
  auto originalYield = cast<TaskflowHyperblockYieldOp>(plan.hyperblock.getBody().front().getTerminator());
  SmallVector<Value> next;
  for (unsigned index : carries) next.push_back(mapping.lookupOrDefault(originalYield.getIterArgsNext()[index]));
  kernelBuilder.create<TaskflowHyperblockYieldOp>(plan.task.getLoc(), next, next);
  bodyBuilder.setInsertionPointToEnd(body);
  SmallVector<Value> yields;
  for (unsigned index = 0; index < writes.size(); ++index)
    yields.push_back(body->getArgument(reads.size() + index));
  bodyBuilder.create<TaskflowYieldOp>(plan.task.getLoc(), ValueRange{}, yields, ValueRange{});
  setLineage(plan.task, stage, builder, plan.leftNodes, isPrefix ? 0 : 1);
  return stage;
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
  TaskflowTaskOp prefix = createStage(plan, builder, temporaries, {}, true);
  TaskflowTaskOp suffix = createStage(plan, builder, temporaries, prefix, false);
  for (auto [position, index] : llvm::enumerate(plan.prefixOutputs))
    plan.task.getDoneWrites()[index].replaceAllUsesWith(
        prefix.getDoneWrites()[temporaries.size() + position]);
  for (auto [position, index] : llvm::enumerate(plan.suffixOutputs))
    plan.task.getDoneWrites()[index].replaceAllUsesWith(suffix.getDoneWrites()[position]);
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
    if (overflow)
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
                    func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
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
