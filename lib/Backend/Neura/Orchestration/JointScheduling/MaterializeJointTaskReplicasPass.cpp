//===- MaterializeJointTaskReplicasPass.cpp -------------------------------===//
//
// Materializes a bounded, proved execution-shard experiment. This is a
// resource transformation: both tasks have the complete original DFG, while
// their root counter domains and writes are disjoint.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/ReplicaOutputCoordinateProof.h"
#include "Backend/Neura/Orchestration/JointScheduling/SourceIterationDomainPartitionProof.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskGraphRewriteLegality.h"
#include "Backend/Neura/Orchestration/JointScheduling/ProveStaticActiveTransferShapesPass.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {

static std::optional<unsigned> parseKernelInputReference(Attribute attribute) {
  if (!attribute)
    return std::nullopt;
  auto text = dyn_cast<StringAttr>(attribute);
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

std::optional<int64_t> constantIndex(Value value) {
  if (!value)
    return std::nullopt;
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  return std::nullopt;
}

// Resolve a bound through the value-input ABI of a Taskflow task.  A
// symbol-dynamic counter is admitted only when its value input has already
// been specialized to a literal at the function scope.  In particular, a
// function argument (the unspecialized AMOEBA-Test seq_len) is not treated as
// a compile-time value merely because its current benchmark row is 320.
static std::optional<int64_t>
compileTimeIndex(Value value, TaskflowTaskOp task,
                 neura::KernelOp kernel = neura::KernelOp()) {
  if (!value)
    return std::nullopt;
  if (auto constant = constantIndex(value))
    return constant;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return compileTimeIndex(cast.getIn(), task, kernel);
  if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
    SmallVector<Attribute> constants;
    constants.reserve(apply->getNumOperands());
    for (Value operand : apply->getOperands()) {
      auto constant = compileTimeIndex(operand, task, kernel);
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

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (kernel && argument.getOwner() == &kernel.getBody().front()) {
      unsigned index = argument.getArgNumber();
      if (index >= kernel.getInputs().size())
        return std::nullopt;
      return compileTimeIndex(kernel.getInputs()[index], task);
    }
    if (argument.getOwner() == &task.getBody().front()) {
      unsigned firstValueInput = task.getWillReads().size() +
                                 task.getWillWrites().size();
      if (argument.getArgNumber() < firstValueInput)
        return std::nullopt;
      unsigned index = argument.getArgNumber() - firstValueInput;
      if (index >= task.getValueInputs().size())
        return std::nullopt;
      return compileTimeIndex(task.getValueInputs()[index], task);
    }
    if (auto function = dyn_cast_or_null<func::FuncOp>(
            argument.getOwner()->getParentOp()))
      return mlir::amoeba::neura::joint_scheduling::detail::
          input0StaticIndexBound(function, argument);
  }
  return std::nullopt;
}

static std::optional<int64_t>
compileTimeCounterBound(neura::CounterOp counter, StringRef attributeName,
                        Value operand, TaskflowTaskOp task,
                        neura::KernelOp kernel = neura::KernelOp()) {
  if (operand)
    if (auto value = compileTimeIndex(operand, task, kernel))
      return value;

  if (auto integer = counter->getAttrOfType<IntegerAttr>(attributeName))
    return integer.getInt();

  if (auto input = parseKernelInputReference(
          counter->getAttr(attributeName))) {
    if (!kernel || *input >= kernel.getInputs().size())
      return std::nullopt;
    return compileTimeIndex(kernel.getInputs()[*input], task, kernel);
  }
  return std::nullopt;
}

// The historical producer validators intentionally retain the permissive
// operand-then-folded-attribute lookup above.  The post-Neura in-place
// replica proof has a different contract: when a counter carries both an SSA
// bound operand and a folded *_value/input reference, both are independent
// claims about the same domain and must be resolved and compared.  In
// particular, a valid folded attr must never mask an unresolved or conflicting
// actual operand.  Keep this helper separate so Radar's legacy validator does
// not silently change while the LU/replica proof remains fail-closed.
static std::optional<int64_t>
compileTimeCounterBoundWithAgreement(
    neura::CounterOp counter, StringRef attributeName, Value operand,
    TaskflowTaskOp task, neura::KernelOp kernel = neura::KernelOp()) {
  std::optional<int64_t> operandValue;
  if (operand) {
    operandValue = compileTimeIndex(operand, task, kernel);
    if (!operandValue)
      return std::nullopt;
  }

  std::optional<int64_t> attributeValue;
  if (Attribute attribute = counter->getAttr(attributeName)) {
    if (auto integer = dyn_cast<IntegerAttr>(attribute)) {
      attributeValue = integer.getInt();
    } else if (auto input = parseKernelInputReference(attribute)) {
      if (!kernel || *input >= kernel.getInputs().size())
        return std::nullopt;
      attributeValue = compileTimeIndex(kernel.getInputs()[*input], task,
                                        kernel);
      if (!attributeValue)
        return std::nullopt;
    } else {
      return std::nullopt;
    }
  }

  if (operandValue && attributeValue && *operandValue != *attributeValue)
    return std::nullopt;
  if (attributeValue)
    return attributeValue;
  return operandValue;
}

static constexpr StringLiteral kSemanticIncomingEdgesAttr =
    "amoeba.semantic.incoming_edges";

static Value stripMemrefCasts(Value value);

// The cumulative ablation enumerates these execution-shard factors explicitly.
// Keep the materializer fail-closed for every other factor: a clone count is a
// semantic transformation, not a generic way to manufacture unsupported
// partitioning proofs. Factor one is the identity candidate and is handled as
// a no-op by the pass driver below.
static bool isSupportedReplicaCount(int64_t count) {
  return count == 1 || count == 2 || count == 4 || count == 8;
}

struct ActiveTransferRewriteSnapshot {
  unsigned argument = 0;
  Attribute logicalShape;
  Attribute capacityShape;
  Attribute activeShape;
  Attribute noAlias;
};

static LogicalResult captureActiveTransferRewriteFacts(
    func::FuncOp function,
    SmallVectorImpl<ActiveTransferRewriteSnapshot> &facts,
    SmallVectorImpl<std::pair<std::string, Attribute>> &callerFacts) {
  for (NamedAttribute attribute : function->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name.starts_with("amoeba.input0_caller_") ||
        name.starts_with("amoeba.static_bound.arg."))
      callerFacts.emplace_back(name.str(), attribute.getValue());
  }
  for (unsigned argument = 0; argument < function.getNumArguments();
       ++argument) {
    if (!hasStaticActiveTransferShapeFacts(function, argument))
      continue;
    std::string error;
    if (failed(
            verifyStaticActiveTransferShapeProof(function, argument, &error))) {
      function.emitError() << "cannot enter source replica rewrite with an "
                              "invalid active-transfer proof on argument "
                           << argument << ": " << error;
      return failure();
    }
    StaticActiveTransferShapeProof proof =
        analyzeStaticActiveTransferShape(function, argument);
    if (llvm::any_of(proof.accesses, [](const ActiveTransferAccessRecord &access) {
          return !access.proven;
        })) {
      function.emitError()
          << "cannot enter source replica rewrite with an unproved memory "
             "access on argument "
          << argument;
      return failure();
    }
    facts.push_back(
        {argument,
         function.getArgAttr(argument, "amoeba.logical_transfer_shape"),
         function.getArgAttr(argument, "amoeba.active_transfer_capacity_shape"),
         function.getArgAttr(argument, "amoeba.active_transfer_shape"),
         function.getArgAttr(argument, "amoeba.noalias")});
  }
  return success();
}

static LogicalResult refreshActiveTransferProofsAfterSourceRewrite(
    func::FuncOp function, ArrayRef<ActiveTransferRewriteSnapshot> facts,
    ArrayRef<std::pair<std::string, Attribute>> callerFacts) {
  for (const ActiveTransferRewriteSnapshot &fact : facts) {
    std::string error;
    if (failed(rederiveAndStoreStaticActiveTransferShapeProof(
            function, fact.argument, &error))) {
      function.emitError() << "trusted source rewrite could not re-prove "
                              "active-transfer facts for argument "
                           << fact.argument << ": " << error;
      return failure();
    }
    StaticActiveTransferShapeProof proof =
        analyzeStaticActiveTransferShape(function, fact.argument);
    if (llvm::any_of(proof.accesses, [](const ActiveTransferAccessRecord &access) {
          return !access.proven;
        })) {
      function.emitError()
          << "trusted source rewrite left an unproved memory access on "
             "argument "
          << fact.argument;
      return failure();
    }
    if (function.getArgAttr(fact.argument, "amoeba.logical_transfer_shape") !=
            fact.logicalShape ||
        function.getArgAttr(fact.argument,
                            "amoeba.active_transfer_capacity_shape") !=
            fact.capacityShape ||
        function.getArgAttr(fact.argument, "amoeba.active_transfer_shape") !=
            fact.activeShape ||
        function.getArgAttr(fact.argument, "amoeba.noalias") != fact.noAlias) {
      function.emitError()
          << "source replica rewrite changed caller capacity, active access "
             "shape, logical footprint, or no-alias root for argument "
          << fact.argument;
      return failure();
    }
    if (failed(verifyStaticActiveTransferShapeProof(function, fact.argument,
                                                    &error))) {
      function.emitError() << "refreshed active-transfer proof is not "
                              "self-consistent for argument "
                           << fact.argument << ": " << error;
      return failure();
    }
  }
  SmallVector<std::pair<std::string, Attribute>> refreshedCallerFacts;
  for (NamedAttribute attribute : function->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name.starts_with("amoeba.input0_caller_") ||
        name.starts_with("amoeba.static_bound.arg."))
      refreshedCallerFacts.emplace_back(name.str(), attribute.getValue());
  }
  llvm::sort(refreshedCallerFacts, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  SmallVector<std::pair<std::string, Attribute>> sortedCallerFacts(
      callerFacts.begin(), callerFacts.end());
  llvm::sort(sortedCallerFacts, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  if (!llvm::equal(refreshedCallerFacts, sortedCallerFacts)) {
    function.emitError(
        "source replica rewrite changed caller-bound static evidence");
    return failure();
  }
  return success();
}

static constexpr StringLiteral kOriginalAmoebaDecisionModeAttr =
    "amoeba.original_amoeba.fixed_decision_materialization";
static constexpr StringLiteral kOriginalAmoebaReplicaDecisionAttr =
    "amoeba.original_amoeba.replica_decision";
static constexpr StringLiteral kOriginalAmoebaReplicaPlacementsAttr =
    "amoeba.original_amoeba.replica_placements";
static constexpr StringLiteral kOriginalAmoebaPartitionRealizationAttr =
    "amoeba.original_amoeba.source_partition_realization";
static constexpr StringLiteral kOriginalAmoebaFixedDecisionFlag =
    "original-amoeba-fixed-decision";

static bool isOriginalAmoebaFixedDecisionTask(func::FuncOp function,
                                              TaskflowTaskOp task) {
  if (!function || !task)
    return false;
  StringRef symbol = function.getSymName();
  StringRef name = task.getTaskName();
  // Keep the baseline scope guard at the stable Itanium symbol prefix. Exact
  // f45 schedule records and all source/no-alias/partition proofs remain the
  // authority for whether an individual task can be materialized.
  return (symbol.starts_with("_Z8gcn_funci") &&
          (name == "Task_16" || name == "Task_17")) ||
         (symbol.starts_with("_Z11harris_funci") &&
          (name == "Task_19" || name == "Task_21")) ||
         (symbol.starts_with("_Z7lu_funci") && name == "Task_6");
}

static bool isOriginalAmoebaFixedDecisionMode(TaskflowTaskOp task) {
  auto mode = task ? task->getAttrOfType<BoolAttr>(
                         kOriginalAmoebaDecisionModeAttr)
                   : BoolAttr();
  return mode && mode.getValue();
}

static bool supportsTaskReplicaCount(TaskflowTaskOp task, int64_t count) {
  return isSupportedReplicaCount(count) ||
         (isOriginalAmoebaFixedDecisionMode(task) && count == 3);
}

static std::optional<int64_t> integerField(DictionaryAttr dictionary,
                                           StringRef name) {
  auto value = dictionary ? dictionary.getAs<IntegerAttr>(name) : IntegerAttr();
  if (!value)
    return std::nullopt;
  return value.getInt();
}

static std::optional<StringRef> stringField(DictionaryAttr dictionary,
                                            StringRef name) {
  auto value = dictionary ? dictionary.getAs<StringAttr>(name) : StringAttr();
  if (!value)
    return std::nullopt;
  return value.getValue();
}

static std::optional<std::pair<int64_t, int64_t>> parseCgraShape(
    StringRef shape) {
  auto parts = shape.split('x');
  int64_t rows = 0;
  int64_t cols = 0;
  if (parts.first.empty() || parts.second.empty() || parts.second.contains('x') ||
      parts.first.getAsInteger(10, rows) ||
      parts.second.getAsInteger(10, cols) || rows < 1 || cols < 1 ||
      rows > 4 || cols > 4 || rows * cols > 4)
    return std::nullopt;
  return std::make_pair(rows, cols);
}

// Read a fixed active-replica decision only from the original TaskScheduler
// record.  The profile's composed shape remains a separate field: each replica
// gets the exact placed orientation recorded in replica_shapes.
static LogicalResult readOriginalAmoebaReplicaDecision(
    TaskflowTaskOp task, int64_t &replicaCount,
    SmallVectorImpl<DictionaryAttr> &replicaShapes) {
  auto fail = [&](StringRef reason) {
    task.emitError() << "original AMOEBA fixed-decision materialization: "
                     << reason;
    return failure();
  };
  auto active = task->getAttrOfType<IntegerAttr>("active_replicas");
  auto info = task->getAttrOfType<DictionaryAttr>(
      "amoeba.task_scheduler_schedule_info");
  auto infoActive = integerField(info, "active_replicas");
  auto infoTask = stringField(info, "task_name");
  auto shapes = info ? info.getAs<ArrayAttr>("replica_shapes") : ArrayAttr();
  auto placements = info ? info.getAs<ArrayAttr>("placements") : ArrayAttr();
  if (!active || !info || !infoActive || !infoTask || !shapes || !placements ||
      *infoActive != active.getInt() || *infoTask != task.getTaskName() ||
      active.getInt() < 2 || active.getInt() > 4 ||
      shapes.size() != static_cast<size_t>(active.getInt()) || placements.empty())
    return fail("trace does not bind a valid 2-4 replica inventory");

  auto composed = task->getAttrOfType<StringAttr>("composed_cgra_shape");
  auto composedCount = task->getAttrOfType<IntegerAttr>("composed_cgra_count");
  auto parsedComposed = composed ? parseCgraShape(composed.getValue())
                                 : std::nullopt;
  if (!composed || !composedCount || !parsedComposed ||
      composedCount.getInt() != parsedComposed->first * parsedComposed->second)
    return fail("selected composed shape/count is invalid");

  SmallVector<DictionaryAttr> byId(active.getInt());
  std::vector<unsigned> cellCounts(active.getInt(), 0);
  std::set<std::pair<int64_t, int64_t>> uniqueCells;
  for (Attribute attribute : shapes) {
    auto shape = dyn_cast<DictionaryAttr>(attribute);
    auto id = integerField(shape, "replica_id");
    auto count = integerField(shape, "cgra_count");
    auto row = integerField(shape, "row");
    auto col = integerField(shape, "col");
    auto rows = integerField(shape, "placement_rows");
    auto cols = integerField(shape, "placement_cols");
    auto shapeText = stringField(shape, "shape");
    auto parsed = shapeText ? parseCgraShape(*shapeText) : std::nullopt;
    if (!shape || !id || !count || !row || !col || !rows || !cols ||
        !shapeText || !parsed || *id < 0 || *id >= active.getInt() ||
        byId[*id] || *count != *rows * *cols || *rows != parsed->first ||
        *cols != parsed->second || *count != composedCount.getInt() ||
        ((*rows != parsedComposed->first || *cols != parsedComposed->second) &&
         (*rows != parsedComposed->second || *cols != parsedComposed->first)) ||
        *row < 0 || *col < 0 || *row + *rows > 4 || *col + *cols > 4)
      return fail("replica shape disagrees with the selected area or placement");
    byId[*id] = shape;
  }
  for (Attribute attribute : placements) {
    auto cell = dyn_cast<DictionaryAttr>(attribute);
    auto id = integerField(cell, "replica_id");
    auto row = integerField(cell, "row");
    auto col = integerField(cell, "col");
    auto context = integerField(cell, "context_id");
    auto start = integerField(cell, "scheduler_start_time");
    auto end = integerField(cell, "scheduler_end_time");
    auto duration = integerField(cell, "scheduler_duration");
    if (!cell || !id || !row || !col || !context || !start || !end ||
        !duration || *id < 0 || *id >= active.getInt() || *row < 0 ||
        *row >= 4 || *col < 0 || *col >= 4 || *context < 0 || *start < 0 ||
        *end <= *start || *duration != *end - *start ||
        !uniqueCells.emplace(*row, *col).second || !byId[*id])
      return fail("placement cell lacks a valid original replica binding");
    auto baseRow = integerField(byId[*id], "row");
    auto baseCol = integerField(byId[*id], "col");
    auto rows = integerField(byId[*id], "placement_rows");
    auto cols = integerField(byId[*id], "placement_cols");
    if (!baseRow || !baseCol || !rows || !cols || *row < *baseRow ||
        *row >= *baseRow + *rows || *col < *baseCol || *col >= *baseCol + *cols)
      return fail("placed cell lies outside its recorded replica rectangle");
    ++cellCounts[*id];
  }
  for (int64_t id = 0; id < active.getInt(); ++id) {
    auto count = integerField(byId[id], "cgra_count");
    if (!count || cellCounts[id] != static_cast<unsigned>(*count))
      return fail("replica placement does not cover its recorded CGRA shape");
    replicaShapes.push_back(byId[id]);
  }
  replicaCount = active.getInt();
  return success();
}

// A completion-only join forwards each state producer to every consumer of
// the joined value.  Rebuilding only the parent annotation is insufficient:
// facts extraction expands the typed join into one Taskflow edge per state.
// Recover those producers from the join itself, and publish annotations only
// after the join verifier, replica metadata, SSA uses, and typed edge graph
// all agree.  This keeps a malformed/deleted state or a wrong consumer
// fail-closed instead of inventing an edge from a task-name string.
//
// A source-owned tile may itself be one state of an outer completion-only
// join.  The recursive proof below admits that composition only when every
// join is a verified, same-storage, acyclic completion tree. Direct states
// must be task done-writes with a matching output root; replica-marked joins
// additionally require a complete, unique replica set. The final consumer
// edges are still taken from TaskEdgeGraph, so both RAW and WAW endpoints are
// preserved exactly as the graph derives them.
static bool hasVerifiedReadCompletionFanIn(TaskflowTaskOp task,
                                           TaskflowJoinOp outputJoin) {
  if (task.getDoneReads().empty())
    return true;
  if (task.getDoneReads().size() != 1)
    return false;
  Value state = task.getDoneReads().front();
  if (!llvm::hasSingleElement(state.getUses()))
    return false;
  auto join = dyn_cast<TaskflowReadCompletionJoinOp>(
      state.use_begin()->getOwner());
  return join && join.getWriteCompletion() == outputJoin.getJoined() &&
         llvm::is_contained(join.getTileStates(), state) &&
         succeeded(verify(join.getOperation()));
}

static LogicalResult appendTypedCompletionJoinIncomingEdges(
    TaskflowTaskOp consumer, Value input,
    SmallVectorImpl<Attribute> &incoming, OpBuilder &builder) {
  auto reject = [&](const Twine &reason) {
    consumer.emitError(reason);
    return failure();
  };

  // Dynamic output states are represented by a memref.cast around the
  // completion join.  The cast is only a type view; the join remains the
  // source-owned typed dependency and must be validated as such.
  Value joinValue = input;
  while (auto cast = joinValue.getDefiningOp<memref::CastOp>())
    joinValue = cast.getSource();
  auto join = joinValue.getDefiningOp<TaskflowJoinOp>();
  if (!join || join.getJoined() != joinValue ||
      !join->hasAttr("amoeba.semantic.completion_only"))
    return reject("replica consumer has an unproved completion join input");
  if (!join->hasAttr("amoeba.replica.completion_only") &&
      !join->hasAttr("amoeba.neura.joint_rewrite"))
    return reject("replica consumer has an unclassified completion join");
  if (failed(verify(join.getOperation())))
    return reject("replica consumer has an invalid completion join");

  auto joinedInput = llvm::find(consumer.getWillReads(), input);
  if (joinedInput == consumer.getWillReads().end() ||
      llvm::count(consumer.getWillReads(), input) != 1)
    return reject("replica consumer does not have one completion join input");
  const uint32_t joinedInputIndex =
      static_cast<uint32_t>(joinedInput - consumer.getWillReads().begin());

  auto function = consumer->getParentOfType<func::FuncOp>();
  if (!function || function != join->getParentOfType<func::FuncOp>())
    return reject("completion join and consumer are in different functions");

  std::string graphError;
  FailureOr<TaskEdgeGraph> graph = buildTaskEdgeGraph(
      function, TaskEdgeGraphOptions{/*require_payload=*/false,
                                     /*read_control_predecessors=*/true},
      graphError);
  if (failed(graph))
    return reject("cannot prove completion join incoming edges: " +
                  Twine(graphError));

  Value rootBase = stripMemrefCasts(join.getBase());
  struct Leaf {
    TaskflowTaskOp task;
    unsigned resultIndex = 0;
  };
  SmallVector<Leaf> leaves;
  llvm::SmallPtrSet<Operation *, 32> visitedJoins;
  llvm::SmallPtrSet<Operation *, 32> visitedLeaves;
  bool sawReplicaJoin = false;

  auto stateHasOnlyJoinUse = [&](Value state, TaskflowJoinOp owner) {
    Value current = state;
    Operation *expectedOwner = owner.getOperation();
    while (auto cast = current.getDefiningOp<memref::CastOp>()) {
      if (!llvm::hasSingleElement(current.getUses()) ||
          current.use_begin()->getOwner() != expectedOwner)
        return false;
      expectedOwner = cast.getOperation();
      current = cast.getSource();
    }
    return llvm::hasSingleElement(current.getUses()) &&
           current.use_begin()->getOwner() == expectedOwner;
  };

  auto visitJoin = [&](auto &&self, TaskflowJoinOp current,
                       unsigned depth) -> LogicalResult {
    if (!current || depth > 32 ||
        !visitedJoins.insert(current.getOperation()).second)
      return reject("completion join graph is cyclic or repeated");
    if (!current->hasAttr("amoeba.semantic.completion_only") ||
        (!current->hasAttr("amoeba.replica.completion_only") &&
         !current->hasAttr("amoeba.neura.joint_rewrite")) ||
        failed(verify(current.getOperation())))
      return reject("completion join graph contains an unproved join");
    if (stripMemrefCasts(current.getBase()) != rootBase)
      return reject("completion join graph mixes storage bases");

    const bool replicaJoin = current->hasAttr("amoeba.replica.completion_only");
    if (replicaJoin)
      sawReplicaJoin = true;
    llvm::SmallDenseSet<int64_t, 8> replicaIds;
    llvm::SmallPtrSet<Operation *, 8> directReplicaTasks;
    std::string parentName;
    int64_t replicaCount = -1;
    for (Value state : current.getTileStates()) {
      if (!stateHasOnlyJoinUse(state, current))
        return reject("completion join state has an unexpected SSA user");
      Value stateValue = stripMemrefCasts(state);
      if (auto child = stateValue.getDefiningOp<TaskflowJoinOp>()) {
        if (child.getJoined() != stateValue ||
            failed(self(self, child, depth + 1)))
          return failure();
        continue;
      }
      auto task = stateValue.getDefiningOp<TaskflowTaskOp>();
      if (!task)
        return reject("completion join state is not a done-write result");
      auto result = llvm::find(task.getDoneWrites(), stateValue);
      if (result == task.getDoneWrites().end())
        return reject("completion join state is not a done-write result");
      const unsigned resultIndex =
          static_cast<unsigned>(result - task.getDoneWrites().begin());
      if (task.getWillWrites().size() != task.getDoneWrites().size() ||
          task.getWillWrites().size() != task.getOriginalWriteMemrefs().size() ||
          resultIndex >= task.getWillWrites().size())
        return reject("completion join state has an incomplete output ABI");
      FailureOr<Value> taskOutputRoot =
          resolveTaskflowMemoryRoot(task.getOriginalWriteMemrefs()[resultIndex]);
      FailureOr<Value> joinOutputRoot = resolveTaskflowMemoryRoot(rootBase);
      if (stripMemrefCasts(task.getWillWrites()[resultIndex]) != rootBase ||
          failed(taskOutputRoot) || failed(joinOutputRoot) ||
          *taskOutputRoot != *joinOutputRoot ||
          !hasVerifiedReadCompletionFanIn(task, current) ||
          !task.getValueOutputs().empty() ||
          !visitedLeaves.insert(task.getOperation()).second)
        return reject("completion join state is not a unique output task");
      if (replicaJoin) {
        auto parent = task->getAttrOfType<StringAttr>(
            "amoeba.replica.parent_task");
        auto id = task->getAttrOfType<IntegerAttr>("amoeba.replica.id");
        auto count = task->getAttrOfType<IntegerAttr>("amoeba.replica.count");
        auto axis = task->getAttrOfType<IntegerAttr>(
            "amoeba.replica.shard_axis");
        auto outputAxis = task->getAttrOfType<IntegerAttr>(
            "amoeba.replica.output_shard_axis");
        if (!parent || !id || !count || !axis || parent.getValue().empty() ||
            id.getInt() < 0 ||
            id.getInt() >= static_cast<int64_t>(current.getTileStates().size()) ||
            (outputAxis ? outputAxis.getInt() : axis.getInt()) !=
                static_cast<int64_t>(current.getAxis()) ||
            !replicaIds.insert(id.getInt()).second ||
            !directReplicaTasks.insert(task.getOperation()).second)
          return reject("completion join state is not a unique replica");
        if (replicaCount < 0)
          replicaCount = count.getInt();
        if (count.getInt() != replicaCount ||
            replicaCount != static_cast<int64_t>(current.getTileStates().size()))
          return reject("completion join replica metadata disagrees with states");
        if (parentName.empty())
          parentName = parent.getValue().str();
        if (parent.getValue() != parentName)
          return reject("completion join mixes replica parent tasks");
      }
      leaves.push_back({task, resultIndex});
    }
    if (replicaJoin &&
        (replicaCount < 2 || replicaIds.size() != current.getTileStates().size() ||
         directReplicaTasks.size() != current.getTileStates().size()))
      return reject("completion join does not contain its complete replica set");
    return success();
  };

  if (failed(visitJoin(visitJoin, join, 0)) || !sawReplicaJoin ||
      leaves.empty())
    return failure();

  const bool expectsWrite = llvm::any_of(
      consumer.getWillWrites(), [&](Value value) {
        return stripMemrefCasts(value) == joinValue;
      });
  for (Leaf &leaf : leaves) {
    SmallVector<const TaskEdge *, 2> incomingEdges;
    for (const TaskEdge &edge : graph->getEdges()) {
      if (edge.origin != TaskEdgeOrigin::Taskflow ||
          edge.producer != leaf.task || edge.consumer != consumer ||
          edge.producer_segment != TaskResultSegment::DoneWrites ||
          edge.producer_index != leaf.resultIndex ||
          edge.scope != TaskEdgeScope::TensorWide)
        continue;
      bool matchingRead =
          edge.kind == TaskEdgeKind::Raw &&
          edge.consumer_segment == TaskOperandSegment::WillReads &&
          edge.consumer_index == joinedInputIndex;
      bool matchingWrite =
          edge.kind == TaskEdgeKind::Waw &&
          edge.consumer_segment == TaskOperandSegment::WillWrites &&
          edge.consumer_index < consumer.getWillWrites().size() &&
          stripMemrefCasts(consumer.getWillWrites()[edge.consumer_index]) ==
              joinValue;
      if (!matchingRead && !matchingWrite)
        return reject("completion join has an unproved read/write edge");
      if (llvm::any_of(incomingEdges, [&](const TaskEdge *previous) {
            return previous->kind == edge.kind;
          }))
        return reject("completion join edge kind is ambiguous");
      incomingEdges.push_back(&edge);
    }
    bool hasRaw = llvm::any_of(incomingEdges, [](const TaskEdge *edge) {
      return edge->kind == TaskEdgeKind::Raw;
    });
    bool hasWaw = llvm::any_of(incomingEdges, [](const TaskEdge *edge) {
      return edge->kind == TaskEdgeKind::Waw;
    });
    if (!hasRaw || (expectsWrite != hasWaw))
      return reject("completion join leaf lacks the required typed RAW/WAW edge");
    for (const TaskEdge *edge : incomingEdges) {
      std::string encoded =
          (leaf.task.getTaskName() + "|producer_consumer|tensor_wide").str();
      // A linked read-completion join can add a WAR edge from the same
      // producer to this consumer. Qualify the done-write edge even when it
      // is the only edge in this completion tree's local result segment.
      encoded += "|" + stringifyTaskEdgeKind(edge->kind).str();
      auto annotation = builder.getStringAttr(encoded);
      if (!llvm::is_contained(incoming, annotation))
        incoming.push_back(annotation);
    }
  }
  return success();
}

static bool hasBalancedShards(int64_t extent, int64_t count) {
  return extent > 0 && count > 1 && extent % count == 0;
}

static std::pair<int64_t, int64_t>
balancedShardBounds(int64_t extent, int64_t shard, int64_t count) {
  const int64_t width = extent / count;
  return {shard * width, (shard + 1) * width};
}

static LogicalResult rejectGemm(TaskflowTaskOp source, StringRef reason) {
  source.emitError() << reason;
  return failure();
}

static bool isPureArith(Operation *operation) {
  return operation && operation->getNumRegions() == 0 &&
         operation->getDialect() &&
         operation->getDialect()->getNamespace() == "arith" &&
         isPure(operation) && isMemoryEffectFree(operation);
}

static bool isReadOnlyTaskInput(Value value, func::FuncOp function) {
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner() == &function.getBody().front() &&
         function.getArgAttr(argument.getArgNumber(), "amoeba.noalias");
}

// Prove the exact producer shape that can be split without changing the
// reduction.  The proof is intentionally narrow: static M/N counters, one
// static unit-step K reduction, read-only inputs, and one direct output store.
// This keeps arbitrary task bodies out of the execution-shard transform.
static FailureOr<TaskflowTaskOp>
validateGemmProducer(func::FuncOp function, TaskflowTaskOp source,
                     int64_t axis, int64_t replicas) {
  if (!source || source.getWillReads().size() != 3 ||
      source.getWillWrites().size() != 1 ||
      source.getOriginalReadMemrefs().size() != 3 ||
      source.getOriginalWriteMemrefs().size() != 1 ||
      source.getDoneReads().size() != 0 || source.getValueOutputs().size() != 0)
    return rejectGemm(source, "GEMM producer must have three reads, one write, "
                              "and no read/value outputs");

  Value output = source.getWillWrites().front();
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!outputType || !outputType.hasStaticShape() || outputType.getRank() != 2 ||
      axis < 0 || axis >= outputType.getRank() ||
      !isSupportedReplicaCount(replicas) || replicas < 2 ||
      !hasBalancedShards(outputType.getShape()[axis], replicas))
    return rejectGemm(source,
                      "GEMM producer needs a static rank-2 output with a "
                      "shard axis divisible by factor 2, 4, or 8");

  if (source.getOriginalWriteMemrefs().front() != output ||
      llvm::is_contained(source.getOriginalReadMemrefs(), output) ||
      llvm::is_contained(source.getWillReads(), output))
    return rejectGemm(source, "GEMM producer output must not alias an input");
  for (auto [index, input] : llvm::enumerate(source.getWillReads()))
    if (source.getOriginalReadMemrefs()[index] != input ||
        !isReadOnlyTaskInput(input, function))
      return rejectGemm(source, "GEMM producer inputs must be noalias function "
                              "arguments with matching original memrefs");

  // The producer result must feed exactly one task.  Channels, returns, and
  // multiple consumers are deliberately rejected because the transform only
  // knows how to replace one direct RAW edge with one completion join.
  Value produced = source.getDoneWrites().front();
  TaskflowTaskOp consumer;
  unsigned resultUses = 0;
  for (OpOperand &use : produced.getUses()) {
    auto candidate = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (!candidate || !llvm::is_contained(candidate.getWillReads(), produced))
      return rejectGemm(source, "GEMM producer must have one direct task "
                              "consumer and no aliases");
    if (consumer && consumer != candidate)
      return rejectGemm(source, "GEMM producer has multiple downstream tasks");
    consumer = candidate;
    ++resultUses;
  }
  if (!consumer || resultUses != 1)
    return rejectGemm(source, "GEMM producer must have exactly one downstream "
                              "consumer");
  unsigned consumerInputs = 0;
  for (Value input : consumer.getWillReads())
    consumerInputs += input == produced;
  if (consumerInputs != 1 ||
      llvm::count(consumer.getOriginalReadMemrefs(), output) != 1)
    return rejectGemm(source, "GEMM producer consumer edge must be unique");
  auto consumerLowers = consumer->getAttrOfType<ArrayAttr>(
      kTilingInputRegionLowersAttr);
  auto consumerUppers = consumer->getAttrOfType<ArrayAttr>(
      kTilingInputRegionUppersAttr);
  auto consumerReasons = consumer->getAttrOfType<ArrayAttr>(
      kTilingInputRegionReasonsAttr);
  if ((consumerLowers || consumerUppers || consumerReasons) &&
      (!consumerLowers || !consumerUppers || !consumerReasons ||
       consumerLowers.size() != consumer.getWillReads().size() ||
       consumerUppers.size() != consumerLowers.size() ||
       consumerReasons.size() != consumerLowers.size()))
    return rejectGemm(source, "GEMM producer consumer has incomplete "
                              "input-region metadata");

  auto allocation = output.getDefiningOp<memref::AllocOp>();
  if (!allocation)
    return rejectGemm(source, "GEMM producer output must be a private alloc");
  for (OpOperand &use : output.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == source.getOperation() || owner == consumer.getOperation() ||
        isa<memref::DeallocOp>(owner))
      continue;
    return rejectGemm(source, "GEMM producer output has an alias or an "
                              "additional user");
  }

  Block &body = source.getBody().front();
  SmallVector<TaskflowCounterOp> counters;
  TaskflowHyperblockOp hyperblock;
  unsigned hyperblockCount = 0;
  for (Operation &operation : body) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      counters.push_back(counter);
      continue;
    }
    if (auto block = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      hyperblock = block;
      ++hyperblockCount;
      continue;
    }
    if (!isa<arith::ConstantOp, arith::ConstantIndexOp, TaskflowYieldOp>(
            &operation))
      return rejectGemm(source, "GEMM producer body contains an unsupported "
                              "top-level operation");
  }
  if (counters.size() != 2 || hyperblockCount != 1)
    return rejectGemm(source, "GEMM producer needs exactly two outer counters "
                              "and one hyperblock");
  for (auto [index, counter] : llvm::enumerate(counters)) {
    if ((index == 0) != !counter.getParentIndex() ||
        (index != 0 &&
         counter.getParentIndex() != counters[index - 1].getCounterIndex()) ||
        constantIndex(counter.getLowerBound()) != std::optional<int64_t>(0) ||
        constantIndex(counter.getUpperBound()) !=
            std::optional<int64_t>(outputType.getShape()[index]) ||
        constantIndex(counter.getStep()) != std::optional<int64_t>(1))
      return rejectGemm(source, "GEMM producer outer bounds must be static "
                              "unit-step M/N counters");
  }
  if (hyperblock.getIndices().size() != counters.size() ||
      !hyperblock.getBody().hasOneBlock())
    return rejectGemm(source, "GEMM producer hyperblock must use both outer "
                              "counters");
  for (auto [index, value] : llvm::enumerate(hyperblock.getIndices()))
    if (value != counters[index].getCounterIndex())
      return rejectGemm(source, "GEMM producer hyperblock indices must be the "
                              "outer counters");

  Block &kernel = hyperblock.getBody().front();
  if (kernel.getNumArguments() != 2)
    return rejectGemm(source, "GEMM producer hyperblock must have M/N indices");
  unsigned stores = 0;
  unsigned reductions = 0;
  memref::StoreOp store;
  scf::ForOp reduction;
  for (Operation &operation : kernel.without_terminator()) {
    if (auto candidate = dyn_cast<memref::StoreOp>(&operation)) {
      store = candidate;
      ++stores;
    } else if (auto candidate = dyn_cast<scf::ForOp>(&operation)) {
      reduction = candidate;
      ++reductions;
    } else if (!isa<memref::LoadOp>(&operation) &&
               !isPureArith(&operation)) {
      return rejectGemm(source, "GEMM producer kernel has an unsupported "
                              "side effect");
    }
  }
  if (stores != 1 || reductions != 1)
    return rejectGemm(source, "GEMM producer needs one K reduction and one "
                              "direct output store");
  unsigned outputArgument = source.getWillReads().size();
  if (store.getMemRef() != body.getArgument(outputArgument) ||
      store.getIndices().size() != kernel.getNumArguments() ||
      store.getValue() != reduction.getResult(0))
    return rejectGemm(source, "GEMM producer store must use the output and "
                              "the reduction result");
  for (auto [index, value] : llvm::enumerate(store.getIndices()))
    if (value != kernel.getArgument(index))
      return rejectGemm(source, "GEMM producer store must be indexed directly "
                              "by M/N counters");

  if (reduction.getInitArgs().size() != 1 || reduction.getNumResults() != 1 ||
      !reduction.getBody() ||
      constantIndex(reduction.getLowerBound()) != std::optional<int64_t>(0) ||
      !constantIndex(reduction.getUpperBound()) ||
      *constantIndex(reduction.getUpperBound()) <= 0 ||
      constantIndex(reduction.getStep()) != std::optional<int64_t>(1))
    return rejectGemm(source, "GEMM producer K loop must have static positive "
                              "unit-step bounds and one accumulator");
  Block &reductionBody = *reduction.getBody();
  if (reductionBody.getNumArguments() != 2)
    return rejectGemm(source, "GEMM producer K loop must have one induction "
                              "and one accumulator argument");
  auto yield = dyn_cast<scf::YieldOp>(reductionBody.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return rejectGemm(source, "GEMM producer K loop must yield one accumulator");
  auto add = yield.getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!add || (add.getOperand(0) != reductionBody.getArgument(1) &&
               add.getOperand(1) != reductionBody.getArgument(1)))
    return rejectGemm(source, "GEMM producer K loop must use an add reduction");
  Value product = add.getOperand(0) == reductionBody.getArgument(1)
                      ? add.getOperand(1)
                      : add.getOperand(0);
  if (!product.getDefiningOp<arith::MulIOp>())
    return rejectGemm(source, "GEMM producer K loop must reduce a multiply");
  unsigned arithOps = 0;
  unsigned loads = 0;
  bool inductionUsedByLoad = false;
  for (Operation &operation : reductionBody.without_terminator()) {
    if (auto load = dyn_cast<memref::LoadOp>(&operation)) {
      ++loads;
      inductionUsedByLoad |=
          llvm::is_contained(load.getIndices(), reduction.getInductionVar());
      continue;
    }
    if (isPureArith(&operation)) {
      ++arithOps;
      continue;
    }
    return rejectGemm(source, "GEMM producer K loop contains an unsafe "
                              "operation");
  }
  if (loads < 2 || arithOps != 2 || !inductionUsedByLoad)
    return rejectGemm(source, "GEMM producer K loop is not a simple multiply "
                              "and add reduction");

  bool badLoad = false;
  source.walk([&](memref::LoadOp load) {
    auto argument = dyn_cast<BlockArgument>(load.getMemRef());
    if (!argument || argument.getOwner() != &body ||
        argument.getArgNumber() >= source.getWillReads().size())
      badLoad = true;
  });
  if (badLoad)
    return rejectGemm(source, "GEMM producer loads must come only from "
                              "read-only inputs");
  return consumer;
}

static LogicalResult makeConsumerTensorWideForReplicas(TaskflowTaskOp consumer,
                                                       Value joined,
                                                       StringRef parent,
                                                       OpBuilder &builder,
                                                       int64_t replicas,
                                                       bool preserveOtherProducerEdges = false) {
  SmallVector<Attribute> incoming;
  if (auto previous = consumer->getAttrOfType<ArrayAttr>(
          kSemanticIncomingEdgesAttr))
    for (Attribute entry : previous) {
      auto text = dyn_cast<StringAttr>(entry);
      if (!text) {
        consumer.emitError("replica consumer has a non-string semantic edge");
        return failure();
      }
      if (!text.getValue().starts_with((Twine(parent) + "|").str()))
        incoming.push_back(entry);
    }
  Value joinedBase = joined;
  while (auto cast = joinedBase.getDefiningOp<memref::CastOp>())
    joinedBase = cast.getSource();
  if (auto completionJoin = joinedBase.getDefiningOp<TaskflowJoinOp>();
      completionJoin && completionJoin->hasAttr(
                             "amoeba.semantic.completion_only") &&
      (completionJoin->hasAttr("amoeba.replica.completion_only") ||
       completionJoin->hasAttr("amoeba.neura.joint_rewrite"))) {
    if (failed(appendTypedCompletionJoinIncomingEdges(consumer, joined,
                                                       incoming, builder)))
      return failure();
  } else {
    for (int64_t replicaId = 0; replicaId < replicas; ++replicaId)
      incoming.push_back(builder.getStringAttr(
          (Twine(parent) + ".replica." + Twine(replicaId) +
           "|producer_consumer|tensor_wide")
              .str()));
  }
  if (preserveOtherProducerEdges) {
    std::string graphError;
    auto function = consumer->getParentOfType<func::FuncOp>();
    FailureOr<TaskEdgeGraph> graph = buildTaskEdgeGraph(
        function, TaskEdgeGraphOptions{/*require_payload=*/false,
                                       /*read_control_predecessors=*/true},
        graphError);
    if (failed(graph)) {
      consumer.emitError("cannot prove preserved replica consumer edges: " +
                         Twine(graphError));
      return failure();
    }
    auto namesJoinedValue = [&](Value value) {
      if (value == joined)
        return true;
      Value base = value;
      while (auto cast = base.getDefiningOp<memref::CastOp>())
        base = cast.getSource();
      Value joinedBase = joined;
      while (auto cast = joinedBase.getDefiningOp<memref::CastOp>())
        joinedBase = cast.getSource();
      return base == joinedBase;
    };
    for (Value input : consumer.getWillReads()) {
      if (namesJoinedValue(input))
        continue;
      auto producer = input.getDefiningOp<TaskflowTaskOp>();
      if (producer)
        continue;
      Value joinInput = input;
      while (auto cast = joinInput.getDefiningOp<memref::CastOp>())
        joinInput = cast.getSource();
      if (auto previousJoin = joinInput.getDefiningOp<TaskflowJoinOp>()) {
        if (!previousJoin->hasAttr("amoeba.semantic.completion_only") ||
            (!previousJoin->hasAttr("amoeba.replica.completion_only") &&
             !previousJoin->hasAttr("amoeba.neura.joint_rewrite"))) {
          consumer.emitError("replica consumer has an unproved non-replica join");
          return failure();
        }
        if (failed(appendTypedCompletionJoinIncomingEdges(
                consumer, input, incoming, builder)))
          return failure();
      } else if (!isa<BlockArgument>(input) &&
                 !input.getDefiningOp<memref::AllocOp>() &&
                 !input.getDefiningOp<memref::AllocaOp>()) {
        consumer.emitError("replica consumer has an unclassified input producer");
        return failure();
      }
    }
    // Annotation coverage is a contract for every actual Taskflow edge, not
    // just data inputs. Independent done-read tokens in will_writes retain
    // WAR order even when their producers provide no will_reads operand.
    SmallVector<Attribute> completeIncoming;
    SmallVector<std::pair<Operation *, TaskEdgeKind>, 8> annotatedPairs;
    for (const TaskEdge &edge : graph->getEdges()) {
      if (edge.origin != TaskEdgeOrigin::Taskflow || edge.consumer != consumer)
        continue;
      auto producer = edge.producer;
      auto pair = std::make_pair(producer.getOperation(), edge.kind);
      if (llvm::is_contained(annotatedPairs, pair)) {
        consumer.emitError("preserved replica consumer has multiple "
                           "Taskflow edges with the same producer and kind");
        return failure();
      }
      annotatedPairs.push_back(pair);
      completeIncoming.push_back(builder.getStringAttr(
          (producer.getTaskName() + "|producer_consumer|" +
           stringifyTaskEdgeScope(edge.scope) + "|" +
           stringifyTaskEdgeKind(edge.kind)).str()));
    }
    incoming = std::move(completeIncoming);
  }
  consumer->setAttr(kSemanticIncomingEdgesAttr,
                    builder.getArrayAttr(incoming));
  auto lowers = consumer->getAttrOfType<ArrayAttr>(
      kTilingInputRegionLowersAttr);
  auto uppers = consumer->getAttrOfType<ArrayAttr>(
      kTilingInputRegionUppersAttr);
  auto reasons = consumer->getAttrOfType<ArrayAttr>(
      kTilingInputRegionReasonsAttr);
  if (!lowers && !uppers && !reasons)
    return success();
  if (!lowers || !uppers || !reasons ||
      lowers.size() != consumer.getWillReads().size() ||
      uppers.size() != lowers.size() || reasons.size() != lowers.size())
    {
      consumer.emitError("replica consumer has incomplete input-region "
                         "metadata");
      return failure();
    }
  SmallVector<Attribute> newLowers(lowers.begin(), lowers.end());
  SmallVector<Attribute> newUppers(uppers.begin(), uppers.end());
  SmallVector<Attribute> newReasons(reasons.begin(), reasons.end());
  bool foundJoinedInput = false;
  for (auto [index, input] : llvm::enumerate(consumer.getWillReads())) {
    Value inputBase = input;
    while (auto cast = inputBase.getDefiningOp<memref::CastOp>())
      inputBase = cast.getSource();
    Value joinedBase = joined;
    while (auto cast = joinedBase.getDefiningOp<memref::CastOp>())
      joinedBase = cast.getSource();
    if (input != joined && inputBase != joinedBase)
      continue;
    foundJoinedInput = true;
    newLowers[index] = builder.getUnitAttr();
    newUppers[index] = builder.getUnitAttr();
    newReasons[index] = builder.getStringAttr("replica_completion");
  }
  if (!foundJoinedInput)
    {
      consumer.emitError("replica consumer does not use the completion join");
      return failure();
    }
  consumer->setAttr(kTilingInputRegionLowersAttr,
                    builder.getArrayAttr(newLowers));
  consumer->setAttr(kTilingInputRegionUppersAttr,
                    builder.getArrayAttr(newUppers));
  consumer->setAttr(kTilingInputRegionReasonsAttr,
                    builder.getArrayAttr(newReasons));
  return success();
}

// A source done-read may feed downstream writers. Replacing it with a typed
// read-completion join must retain one WAR edge from every replica to each
// writer, with annotations checked against the derived TaskEdgeGraph.
static LogicalResult makeReadCompletionConsumerEdges(
    TaskflowTaskOp consumer, Value joined, StringRef parent,
    OpBuilder &builder) {
  auto reject = [&](const Twine &reason) {
    consumer.emitError(reason);
    return failure();
  };
  auto join = joined.getDefiningOp<TaskflowReadCompletionJoinOp>();
  if (!join || join.getJoined() != joined || failed(verify(join.getOperation())))
    return reject("replica writer has an unverified read-completion join");
  auto joinedWrite = llvm::find(consumer.getWillWrites(), joined);
  if (joinedWrite == consumer.getWillWrites().end() ||
      llvm::count(consumer.getWillWrites(), joined) != 1)
    return reject("replica writer must consume one read-completion token");
  unsigned writeIndex =
      static_cast<unsigned>(joinedWrite - consumer.getWillWrites().begin());
  auto function = consumer->getParentOfType<func::FuncOp>();
  if (!function || function != join->getParentOfType<func::FuncOp>())
    return reject("read-completion join and writer are in different functions");
  std::string graphError;
  FailureOr<TaskEdgeGraph> graph = buildTaskEdgeGraph(
      function, TaskEdgeGraphOptions{/*require_payload=*/false,
                                     /*read_control_predecessors=*/true},
      graphError);
  if (failed(graph))
    return reject("cannot prove read-completion WAR edges: " +
                  Twine(graphError));

  SmallVector<Attribute> incoming;
  if (auto previous = consumer->getAttrOfType<ArrayAttr>(
          kSemanticIncomingEdgesAttr)) {
    for (Attribute entry : previous) {
      auto text = dyn_cast<StringAttr>(entry);
      if (!text)
        return reject("replica writer has a non-string semantic edge");
      if (!text.getValue().starts_with((Twine(parent) + "|").str()))
        incoming.push_back(entry);
    }
  }

  DenseSet<Operation *> leaves;
  for (Value state : join.getTileStates()) {
    auto producer = state.getDefiningOp<TaskflowTaskOp>();
    if (!producer)
      return reject("read-completion join contains an unclassified replica leaf");
    auto result = llvm::find(producer.getDoneReads(), state);
    if (result == producer.getDoneReads().end() ||
        !leaves.insert(producer.getOperation()).second)
      return reject("read-completion join contains a duplicate or non-read result");
    unsigned resultIndex =
        static_cast<unsigned>(result - producer.getDoneReads().begin());
    SmallVector<const TaskEdge *, 2> matchingEdges;
    for (const TaskEdge &edge : graph->getEdges())
      if (edge.origin == TaskEdgeOrigin::Taskflow &&
          edge.producer == producer && edge.consumer == consumer &&
          edge.producer_segment == TaskResultSegment::DoneReads &&
          edge.producer_index == resultIndex && edge.kind == TaskEdgeKind::War &&
          edge.consumer_segment == TaskOperandSegment::WillWrites &&
          edge.consumer_index == writeIndex &&
          edge.scope == TaskEdgeScope::TensorWide)
        matchingEdges.push_back(&edge);
    if (matchingEdges.size() != 1)
      return reject("read-completion leaf lacks one typed tensor-wide WAR edge");
    incoming.push_back(builder.getStringAttr(
        (producer.getTaskName() +
         "|producer_consumer|tensor_wide|war")
            .str()));
  }
  consumer->setAttr(kSemanticIncomingEdgesAttr,
                    builder.getArrayAttr(incoming));
  return success();
}

static LogicalResult rejectLlama(TaskflowTaskOp source, StringRef reason) {
  source.emitError() << reason;
  return failure();
}


static LogicalResult rejectRadar(TaskflowTaskOp source, StringRef reason) {
  source.emitError() << reason;
  return failure();
}

// The specialized validators below prove workload-specific storage and
// reduction contracts.  Task names are only local compiler labels: LU, GCN,
// Harris, and Radar all legitimately use Task_0.  Bind the specialized paths
// to the source function's mangled workload symbol as well, so a generic
// Task_0 cannot accidentally enter the LLaMA or Radar proof.
static bool isLlamaFunction(func::FuncOp function) {
  return function && function.getSymName().starts_with("_Z10llama_func");
}

static bool isRadarFunction(func::FuncOp function) {
  return function && function.getSymName().starts_with("_Z10radar_func");
}

static bool isRadarNoAliasArgument(Value value, func::FuncOp function) {
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner() == &function.getBody().front() &&
         function.getArgAttr(argument.getArgNumber(), "amoeba.noalias");
}

static FailureOr<TaskflowTaskOp>
validateRadarLegacyProducer(func::FuncOp function, TaskflowTaskOp source,
                      int64_t axis, int64_t replicas) {
  if (axis != 0 || source.getTaskName() != "Task_0")
    return rejectRadar(source, "Radar replica requires Task_0 and chirp axis");
  if (!isSupportedReplicaCount(replicas) || replicas < 2 ||
      !hasBalancedShards(32, replicas))
    return rejectRadar(source,
                       "Radar replica factor must be 2, 4, or 8 and divide "
                       "the 32-chirp domain");
  if (auto dlp = source->getAttrOfType<BoolAttr>("dlp_replicable");
      dlp && !dlp.getValue())
    return rejectRadar(source,
                       "Radar replica rejected explicit dlp_replicable=false");
  if (auto incoming =
          source->getAttrOfType<ArrayAttr>(kSemanticIncomingEdgesAttr);
      incoming && !incoming.empty())
    return rejectRadar(source,
                       "Radar Task_0 must not have an incoming semantic edge");
  if (source.getWillReads().size() != 1 ||
      source.getWillWrites().size() != 1 ||
      source.getOriginalReadMemrefs().size() != 1 ||
      source.getOriginalWriteMemrefs().size() != 1 ||
      source.getDoneWrites().size() != 1 || !source.getDoneReads().empty() ||
      !source.getValueOutputs().empty() || source.getValueInputs().size() != 3)
    return rejectRadar(source,
                       "Radar Task_0 requires one input, one output, and "
                       "three value inputs");

  Value input = source.getWillReads().front();
  Value output = source.getWillWrites().front();
  auto inputType = dyn_cast<MemRefType>(input.getType());
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!inputType || inputType.getRank() != 2 ||
      inputType.getElementType() !=
          IntegerType::get(source.getContext(), 32) ||
      inputType.isDynamicDim(1) || inputType.getShape()[1] != 256 ||
      !outputType || outputType.getRank() != 1 ||
      outputType.getElementType() !=
          IntegerType::get(source.getContext(), 32) ||
      (!inputType.hasStaticShape() || inputType.getShape()[0] != 32 ||
       !outputType.hasStaticShape()) ||
      outputType.getShape()[0] != 32 ||
      source.getOriginalReadMemrefs().front() != input ||
      source.getOriginalWriteMemrefs().front() != output || input == output ||
      !isRadarNoAliasArgument(input, function) ||
      !isRadarNoAliasArgument(output, function))
    return rejectRadar(source,
                       "Radar Task_0 needs distinct noalias i32 input and "
                       "per-chirp output memrefs");

  // The producer's storage value is also named by the direct consumer's
  // original_read_memrefs list.  That is the explicit producer/consumer
  // contract we validate below; every other use would either introduce an
  // alias or hide an additional writer.
  for (OpOperand &use : output.getUses()) {
    if (use.getOwner() == source.getOperation())
      continue;
    auto task = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (task && task.getTaskName() == "Task_2" &&
        llvm::is_contained(task.getOriginalReadMemrefs(), output) &&
        !llvm::is_contained(task.getWillWrites(), output))
      continue;
    return rejectRadar(source,
                       "Radar Task_0 output has an alias or extra writer");
  }
  for (OpOperand &use : input.getUses()) {
    auto task = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (!task || !llvm::is_contained(task.getWillReads(), input) ||
        llvm::is_contained(task.getWillWrites(), input))
      return rejectRadar(source,
                         "Radar Task_0 input has an alias or possible writer");
  }

  Value produced = source.getDoneWrites().front();
  TaskflowTaskOp consumer;
  unsigned directUses = 0;
  for (OpOperand &use : produced.getUses()) {
    auto candidate = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (!candidate || candidate.getTaskName() != "Task_2" ||
        !llvm::is_contained(candidate.getWillReads(), produced))
      return rejectRadar(source,
                         "Radar Task_0 result must feed only Task_2");
    consumer = candidate;
    ++directUses;
  }
  if (!consumer || directUses != 1 ||
      llvm::count(consumer.getWillReads(), produced) != 1 ||
      llvm::count(consumer.getOriginalReadMemrefs(), output) != 1)
    return rejectRadar(source,
                       "Radar Task_0 requires one direct Task_2 RAW edge");
  if (auto incoming =
          consumer->getAttrOfType<ArrayAttr>(kSemanticIncomingEdgesAttr))
    for (Attribute edge : incoming)
      if (auto text = dyn_cast<StringAttr>(edge);
          text && text.getValue().contains("tile_local"))
        return rejectRadar(source,
                           "Radar Task_0 cannot replace a tile-local edge");

  Block &body = source.getBody().front();
  SmallVector<TaskflowCounterOp> counters;
  neura::KernelOp kernel;
  unsigned kernels = 0;
  for (Operation &operation : body) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      counters.push_back(counter);
      continue;
    }
    if (auto candidate = dyn_cast<neura::KernelOp>(&operation)) {
      kernel = candidate;
      ++kernels;
      continue;
    }
    if (!isa<arith::ConstantOp, arith::ConstantIndexOp, TaskflowYieldOp>(
            &operation))
      return rejectRadar(source,
                         "Radar Task_0 body contains an unsupported op");
  }
  if (counters.size() != 2 || kernels != 1 || !kernel)
    return rejectRadar(source,
                       "Radar Task_0 requires two counters and one Neura "
                       "kernel");
  if (counters[0].getParentIndex() ||
      counters[1].getParentIndex() != counters[0].getCounterIndex())
    return rejectRadar(source,
                       "Radar Task_0 counters must form a chirp/range chain");
  auto checkTaskCounter = [&](TaskflowCounterOp counter, int64_t lower,
                              std::optional<int64_t> upper, StringRef name) {
    if (constantIndex(counter.getLowerBound()) != lower ||
        constantIndex(counter.getStep()) != 1 ||
        (upper && compileTimeIndex(counter.getUpperBound(), source) != upper))
      return rejectRadar(
          source,
          (Twine("Radar Task_0 ") + name +
           " counter is not a compile-time unit-step domain")
              .str());
    return success();
  };
  if (failed(checkTaskCounter(counters[0], 0, 32, "chirp")) ||
      failed(checkTaskCounter(counters[1], 0, 64, "range")))
    return failure();
  if (compileTimeIndex(source.getValueInputs()[0], source) !=
          std::optional<int64_t>(64) ||
      compileTimeIndex(source.getValueInputs()[1], source) !=
          std::optional<int64_t>(0) ||
      compileTimeIndex(source.getValueInputs()[2], source) !=
          std::optional<int64_t>(64))
    return rejectRadar(source,
                       "Radar Task_0 value inputs must bind range_bins=64, "
                       "zero accumulator, and divisor=64");
  if (auto trip = source->getAttrOfType<IntegerAttr>("trip_count");
      trip && trip.getInt() != 32 * 64)
    return rejectRadar(source,
                       "Radar Task_0 trip_count must be 32*64 for input0");

  const unsigned firstWrite = source.getWillReads().size();
  const unsigned firstValue =
      source.getWillReads().size() + source.getWillWrites().size();
  if (kernel.getInputs().size() != 5 ||
      kernel.getIterArgsInit().size() != 1 ||
      kernel.getInputs()[0] != body.getArgument(0) ||
      kernel.getInputs()[1] != body.getArgument(firstValue + 2) ||
      kernel.getInputs()[2] != body.getArgument(firstValue) ||
      kernel.getInputs()[3] != body.getArgument(firstWrite) ||
      kernel.getInputs()[4] != body.getArgument(firstValue))
    return rejectRadar(source,
                       "Radar Task_0 Neura kernel inputs are not the canonical "
                       "input/divisor/range/output ABI");
  Block &kernelBody = kernel.getBody().front();
  if (kernelBody.getNumArguments() != 6 ||
      kernel.getIterArgsInit().front() != body.getArgument(firstValue + 1))
    return rejectRadar(
        source, "Radar Task_0 Neura kernel has an invalid accumulator ABI");
  SmallVector<neura::CounterOp> neuraCounters;
  kernel.walk([&](neura::CounterOp counter) { neuraCounters.push_back(counter); });
  if (neuraCounters.size() != 2)
    return rejectRadar(source,
                       "Radar Task_0 Neura kernel requires chirp and range "
                       "counters");
  llvm::sort(neuraCounters, [](neura::CounterOp lhs, neura::CounterOp rhs) {
    auto lhsId = lhs->getAttrOfType<IntegerAttr>("counter_id");
    auto rhsId = rhs->getAttrOfType<IntegerAttr>("counter_id");
    if (!lhsId || !rhsId)
      return static_cast<bool>(lhsId);
    return lhsId.getInt() < rhsId.getInt();
  });
  for (auto [index, counter] : llvm::enumerate(neuraCounters)) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    int64_t upper = index == 0 ? 32 : 64;
    if (!id || id.getInt() != static_cast<int64_t>(index) ||
        compileTimeIndex(counter.getLowerBound(), source, kernel) != 0 ||
        compileTimeIndex(counter.getUpperBound(), source, kernel) != upper ||
        compileTimeIndex(counter.getStep(), source, kernel) != 1)
      return rejectRadar(source,
                         "Radar Task_0 Neura counters are not 0..32/64 "
                         "compile-time unit-step domains");
  }
  Value kernelInput = kernelBody.getArgument(0);
  Value kernelOutput = kernelBody.getArgument(3);
  Value kernelChirp = neuraCounters[0].getResult();
  Value kernelRange = neuraCounters[1].getResult();
  unsigned loads = 0, stores = 0;
  bool badMemory = false;
  memref::LoadOp load;
  memref::StoreOp store;
  kernel.walk([&](memref::LoadOp candidate) {
    load = candidate;
    ++loads;
    badMemory |= candidate.getMemRef() != kernelInput ||
                 candidate.getIndices().size() != 2 ||
                 candidate.getIndices()[0] != kernelChirp ||
                 candidate.getIndices()[1] != kernelRange;
  });
  kernel.walk([&](memref::StoreOp candidate) {
    store = candidate;
    ++stores;
    badMemory |= candidate.getMemRef() != kernelOutput ||
                 candidate.getIndices().size() != 1 ||
                 candidate.getIndices()[0] != kernelChirp;
  });
  kernel.walk([&](Operation *operation) {
    if (operation == kernel.getOperation() ||
        isa<neura::CounterOp, memref::LoadOp, memref::StoreOp, scf::IfOp,
            scf::YieldOp, neura::YieldOp, arith::ConstantOp,
            arith::ConstantIndexOp>(operation))
      return;
    if (!isMemoryEffectFree(operation))
      badMemory = true;
  });
  if (loads != 1 || stores != 1 || badMemory || !load || !store)
    return rejectRadar(source,
                       "Radar Task_0 Neura DFG must load input[chirp,range] "
                       "and store output[chirp] exactly once");
  return consumer;
}




static LogicalResult materializeRadarLegacyProducer(func::FuncOp function,
                                              TaskflowTaskOp source,
                                              int64_t axis, int64_t replicas) {
  FailureOr<TaskflowTaskOp> validated =
      validateRadarLegacyProducer(function, source, axis, replicas);
  if (failed(validated))
    return failure();
  TaskflowTaskOp consumer = *validated;
  constexpr int64_t chirps = 32;
  constexpr int64_t rangeBins = 64;
  const std::string sourceName = source.getTaskName().str();
  OpBuilder builder(source);
  SmallVector<Value> states;
  for (int64_t replicaId = 0; replicaId < replicas; ++replicaId) {
    const auto [lower, upper] =
        balancedShardBounds(chirps, replicaId, replicas);
    auto replica = cast<TaskflowTaskOp>(builder.clone(*source.getOperation()));
    replica->setAttr("task_name",
                     builder.getStringAttr(sourceName + ".replica." +
                                           std::to_string(replicaId)));
    replica->setAttr("amoeba.replica.parent_task",
                     builder.getStringAttr(sourceName));
    replica->setAttr("amoeba.replica.id", builder.getI64IntegerAttr(replicaId));
    replica->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicas));
    replica->setAttr("amoeba.replica.shard_axis", builder.getI64IntegerAttr(0));
    replica->setAttr("amoeba.replica.shard_lower",
                     builder.getI64IntegerAttr(lower));
    replica->setAttr("amoeba.replica.shard_upper",
                     builder.getI64IntegerAttr(upper));
    replica->removeAttr(kSemanticIncomingEdgesAttr);
    replica->setAttr(
        kTilingOutputRegionLowersAttr,
        builder.getArrayAttr({builder.getDenseI64ArrayAttr({lower})}));
    replica->setAttr(
        kTilingOutputRegionUppersAttr,
        builder.getArrayAttr({builder.getDenseI64ArrayAttr({upper})}));

    Block &replicaBody = replica.getBody().front();
    SmallVector<TaskflowCounterOp> clonedCounters;
    neura::KernelOp clonedKernel;
    for (Operation &operation : replicaBody) {
      if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
        clonedCounters.push_back(counter);
      else if (auto candidate = dyn_cast<neura::KernelOp>(&operation))
        clonedKernel = candidate;
    }
    if (clonedCounters.size() != 2 || !clonedKernel ||
        !clonedCounters[1].getCounterIndex().use_empty())
      return rejectRadar(source,
                         "Radar replica clone lost the dead range scaffold "
                         "or complete Neura kernel");
    // The source range Taskflow counter is only a dead scheduling scaffold;
    // the complete range reduction remains in the Neura DFG below.  Remove
    // that scaffold so the completion state has the rank-one chirp region
    // required by the rank-one dc_i memref and its completion join.
    clonedCounters[1].erase();
    clonedCounters.pop_back();

    OpBuilder counterBuilder(clonedCounters[0]);
    Value lowerValue =
        counterBuilder.create<arith::ConstantIndexOp>(source.getLoc(), lower);
    Value upperValue =
        counterBuilder.create<arith::ConstantIndexOp>(source.getLoc(), upper);
    clonedCounters[0].getLowerBoundMutable().assign(lowerValue);
    clonedCounters[0].getUpperBoundMutable().assign(upperValue);
    clonedCounters[0].setCounterDynamismAttr(
        builder.getStringAttr("constant_bound"));
    SmallVector<neura::CounterOp> clonedNeuraCounters;
    clonedKernel.walk(
        [&](neura::CounterOp counter) { clonedNeuraCounters.push_back(counter); });
    for (neura::CounterOp counter : clonedNeuraCounters) {
      auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
      if (!id || id.getInt() != 0)
        continue;
      OpBuilder kernelBuilder =
          OpBuilder::atBlockBegin(&clonedKernel.getBody().front());
      Value kernelLower =
          kernelBuilder.create<arith::ConstantIndexOp>(source.getLoc(), lower);
      Value kernelUpper =
          kernelBuilder.create<arith::ConstantIndexOp>(source.getLoc(), upper);
      counter.getLowerBoundMutable().assign(kernelLower);
      counter.getUpperBoundMutable().assign(kernelUpper);
      counter->setAttr("counter_dynamism",
                       builder.getStringAttr("constant_bound"));
    }
    int64_t shardTripCount = (upper - lower) * rangeBins;
    replica->setAttr("trip_count",
                     builder.getI64IntegerAttr(shardTripCount));
    if (replica->getAttrOfType<IntegerAttr>("amoeba.selected_trip_count"))
      replica->setAttr("amoeba.selected_trip_count",
                       builder.getI64IntegerAttr(shardTripCount));
    states.push_back(replica.getDoneWrites().front());
  }

  auto outputType = cast<MemRefType>(source.getWillWrites().front().getType());
  auto join = builder.create<TaskflowJoinOp>(
      source.getLoc(), outputType, states, source.getWillWrites().front(),
      builder.getI64IntegerAttr(0), builder.getDenseI64ArrayAttr({0}),
      builder.getDenseI64ArrayAttr({chirps}));
  join->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
  join->setAttr("amoeba.replica.completion_only", builder.getUnitAttr());
  source.getDoneWrites().front().replaceAllUsesWith(join.getJoined());
  if (failed(makeConsumerTensorWideForReplicas(consumer, join.getJoined(),
                                               sourceName, builder, replicas)))
    return failure();
  source.erase();
  function->setAttr("amoeba.replica.materialized_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.parent_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicas));
  function->setAttr("amoeba.replica.shard_axis", builder.getI64IntegerAttr(0));
  function->setAttr("amoeba.replica.total_trip_count",
                    builder.getI64IntegerAttr(chirps * rangeBins));
  function->setAttr("amoeba.replica.shard_trip_count",
                    builder.getI64IntegerAttr((chirps / replicas) * rangeBins));
  return success();
}



// The prepared input keeps caller-owned memrefs dynamically shaped while its
// source-owned preparation records the exact transfer shape on the function
// argument.  Radar's row split can consume that contract because its memory
// proof below names every access coordinate explicitly.  A dynamic type is
// accepted only when the contract agrees with all fixed dimensions; a static
// type remains valid without the auxiliary attribute for legacy prepared
// inputs.
static std::optional<SmallVector<int64_t>> replicaCallerShape(Value value);

static bool radarShapeMatches(Value value, MemRefType type,
                              ArrayRef<int64_t> expected) {
  if (!type || type.getRank() != static_cast<int64_t>(expected.size()))
    return false;
  for (auto [dimension, extent] : llvm::enumerate(expected)) {
    int64_t declared = type.getDimSize(dimension);
    if (!ShapedType::isDynamic(declared) && declared != extent)
      return false;
  }
  if (type.hasStaticShape())
    return llvm::equal(type.getShape(), expected);
  auto contract = replicaCallerShape(value);
  return contract && llvm::equal(*contract, expected);
}

// Radar's Task_0/Task_1 producers are deliberately narrow producer proofs. Their outer domain is
// the 32 independent chirps and its inner domain is the statically bound
// input row (64 for the canonical input). The output is one scalar per chirp,
// so splitting the outer domain creates disjoint output regions while
// preserving the complete reduction body in each clone.
static FailureOr<TaskflowTaskOp>
validateRadarNormalizedProducer(func::FuncOp function, TaskflowTaskOp source,
                      int64_t axis, int64_t replicas) {
  if (axis != 0 || (source.getTaskName() != "Task_0" &&
                    source.getTaskName() != "Task_1"))
    return rejectRadar(source,
                       "Radar replica requires Task_0/Task_1 and chirp axis");
  if (!isSupportedReplicaCount(replicas) || replicas < 2 ||
      !hasBalancedShards(32, replicas))
    return rejectRadar(source,
                       "Radar replica factor must be 2, 4, or 8 and divide "
                       "the 32-chirp domain");
  if (auto dlp = source->getAttrOfType<BoolAttr>("dlp_replicable");
      dlp && !dlp.getValue())
    return rejectRadar(source,
                       "Radar replica rejected explicit dlp_replicable=false");
  if (auto incoming =
          source->getAttrOfType<ArrayAttr>(kSemanticIncomingEdgesAttr);
      incoming && !incoming.empty())
    return rejectRadar(source,
                       "Radar producer must not have an incoming semantic edge");
  if (source.getWillReads().size() != 1 ||
      source.getWillWrites().size() != 1 ||
      source.getOriginalReadMemrefs().size() != 1 ||
      source.getOriginalWriteMemrefs().size() != 1 ||
      source.getDoneWrites().size() != 1 || !source.getDoneReads().empty() ||
      !source.getValueOutputs().empty() || source.getValueInputs().size() != 3)
    return rejectRadar(source,
                       "Radar producer requires one input, one output, and "
                       "three value inputs");

  Value input = source.getWillReads().front();
  Value output = source.getWillWrites().front();
  auto inputType = dyn_cast<MemRefType>(input.getType());
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!inputType || inputType.getRank() != 2 ||
      inputType.getElementType() !=
          IntegerType::get(source.getContext(), 32) ||
      inputType.isDynamicDim(1) || inputType.getShape()[1] != 256 ||
      !outputType || outputType.getRank() != 1 ||
      outputType.getElementType() !=
          IntegerType::get(source.getContext(), 32) ||
      !radarShapeMatches(input, inputType, {32, 256}) ||
      !radarShapeMatches(output, outputType, {32}) ||
      source.getOriginalReadMemrefs().front() != input ||
      source.getOriginalWriteMemrefs().front() != output || input == output ||
      !isRadarNoAliasArgument(input, function) ||
      !isRadarNoAliasArgument(output, function))
    return rejectRadar(source,
                       "Radar producer needs distinct noalias i32 input and "
                       "per-chirp output memrefs");

  // The producer's storage value is also named by the direct consumer's
  // original_read_memrefs list.  That is the explicit producer/consumer
  // contract we validate below; every other use would either introduce an
  // alias or hide an additional writer.
  for (OpOperand &use : output.getUses()) {
    if (use.getOwner() == source.getOperation())
      continue;
    auto task = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (task && task.getTaskName() == "Task_2" &&
        llvm::is_contained(task.getOriginalReadMemrefs(), output) &&
        !llvm::is_contained(task.getWillWrites(), output))
      continue;
    return rejectRadar(source,
                       "Radar producer output has an alias or extra writer");
  }
  for (OpOperand &use : input.getUses()) {
    auto task = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (!task || !llvm::is_contained(task.getWillReads(), input) ||
        llvm::is_contained(task.getWillWrites(), input))
      return rejectRadar(source,
                         "Radar producer input has an alias or possible writer");
  }

  Value produced = source.getDoneWrites().front();
  TaskflowTaskOp consumer;
  unsigned directUses = 0;
  for (OpOperand &use : produced.getUses()) {
    auto candidate = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (!candidate || candidate.getTaskName() != "Task_2" ||
        !llvm::is_contained(candidate.getWillReads(), produced))
      return rejectRadar(source,
                         "Radar producer result must feed only Task_2");
    consumer = candidate;
    ++directUses;
  }
  if (!consumer || directUses != 1 ||
      llvm::count(consumer.getWillReads(), produced) != 1 ||
      llvm::count(consumer.getOriginalReadMemrefs(), output) != 1)
    return rejectRadar(source,
                       "Radar producer requires one direct Task_2 RAW edge");
  if (auto incoming =
          consumer->getAttrOfType<ArrayAttr>(kSemanticIncomingEdgesAttr))
    for (Attribute edge : incoming)
      if (auto text = dyn_cast<StringAttr>(edge);
          text && text.getValue().contains("tile_local"))
        return rejectRadar(source,
                           "Radar producer cannot replace a tile-local edge");

  Block &body = source.getBody().front();
  SmallVector<TaskflowCounterOp> counters;
  neura::KernelOp kernel;
  unsigned kernels = 0;
  for (Operation &operation : body) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      counters.push_back(counter);
      continue;
    }
    if (auto candidate = dyn_cast<neura::KernelOp>(&operation)) {
      kernel = candidate;
      ++kernels;
      continue;
    }
    if (!isa<arith::ConstantOp, arith::ConstantIndexOp, TaskflowYieldOp>(
            &operation))
      return rejectRadar(source,
                         "Radar Task_0 body contains an unsupported op");
  }
  if ((counters.size() != 1 && counters.size() != 2) || kernels != 1 ||
      !kernel)
    return rejectRadar(source,
                       "Radar producer requires one or two counters and one "
                       "Neura kernel");
  if (counters[0].getParentIndex())
    return rejectRadar(source,
                       "Radar producer chirp counter must be a root counter");
  if (counters.size() == 2 &&
      counters[1].getParentIndex() != counters[0].getCounterIndex())
    return rejectRadar(
        source, "Radar producer counters must form a chirp/range chain");
  auto checkTaskCounter = [&](TaskflowCounterOp counter, int64_t lower,
                              std::optional<int64_t> upper, StringRef name) {
    if (constantIndex(counter.getLowerBound()) != lower ||
        constantIndex(counter.getStep()) != 1 ||
        (upper && compileTimeIndex(counter.getUpperBound(), source) != upper))
      return rejectRadar(
          source,
          (Twine("Radar Task_0 ") + name +
           " counter is not a compile-time unit-step domain")
              .str());
    return success();
  };
  if (failed(checkTaskCounter(counters[0], 0, 32, "chirp")) ||
      (counters.size() == 2 &&
       failed(checkTaskCounter(counters[1], 0, 64, "range"))))
    return failure();
  if (counters.size() == 2 && !counters[1].getCounterIndex().use_empty())
    return rejectRadar(source,
                       "Radar producer range scaffold must be dead before "
                       "row replication");
  if (compileTimeIndex(source.getValueInputs()[0], source) !=
          std::optional<int64_t>(64) ||
      compileTimeIndex(source.getValueInputs()[1], source) !=
          std::optional<int64_t>(0) ||
      compileTimeIndex(source.getValueInputs()[2], source) !=
          std::optional<int64_t>(64))
    return rejectRadar(source,
                       "Radar Task_0 value inputs must bind range_bins=64, "
                       "zero accumulator, and divisor=64");
  if (auto trip = source->getAttrOfType<IntegerAttr>("trip_count");
      trip && trip.getInt() != 32 * 64)
    return rejectRadar(source,
                       "Radar Task_0 trip_count must be 32*64 for input0");

  const unsigned firstWrite = source.getWillReads().size();
  const unsigned firstValue =
      source.getWillReads().size() + source.getWillWrites().size();
  if (kernel.getInputs().size() != 5 || !kernel.getIterArgsInit().empty() ||
      kernel.getInputs()[0] != body.getArgument(0) ||
      kernel.getInputs()[1] != body.getArgument(firstValue) ||
      kernel.getInputs()[2] != body.getArgument(firstValue + 1) ||
      kernel.getInputs()[3] != body.getArgument(firstValue + 2) ||
      kernel.getInputs()[4] != body.getArgument(firstWrite))
    return rejectRadar(
        source,
        "Radar producer Neura kernel inputs are not the canonical "
        "input/range/accumulator/divisor/output ABI");
  Block &kernelBody = kernel.getBody().front();
  if (kernelBody.getNumArguments() != kernel.getInputs().size())
    return rejectRadar(
        source, "Radar producer Neura kernel has an invalid accumulator ABI");
  SmallVector<neura::CounterOp> neuraCounters;
  kernel.walk([&](neura::CounterOp counter) { neuraCounters.push_back(counter); });
  if (neuraCounters.size() != 1)
    return rejectRadar(source,
                       "Radar producer Neura kernel requires exactly one "
                       "chirp counter and an explicit range recurrence");
  auto counterId = neuraCounters.front()->getAttrOfType<IntegerAttr>(
      "counter_id");
  if (!counterId || counterId.getInt() != 0 ||
      compileTimeCounterBound(neuraCounters.front(), "lower_bound_value",
                              neuraCounters.front().getLowerBound(), source,
                              kernel) != 0 ||
      compileTimeCounterBound(neuraCounters.front(), "upper_bound_value",
                              neuraCounters.front().getUpperBound(), source,
                              kernel) != 32 ||
      compileTimeCounterBound(neuraCounters.front(), "step_value",
                              neuraCounters.front().getStep(), source,
                              kernel) != 1)
    return rejectRadar(
        source,
        "Radar producer Neura chirp counter is not a 0..32 unit-step domain");
  Value kernelInput = kernelBody.getArgument(0);
  Value kernelOutput = kernelBody.getArgument(4);
  Value kernelChirp = neuraCounters.front().getResult();
  unsigned loads = 0, stores = 0;
  bool badMemory = false;
  neura::LoadIndexedOp load;
  neura::StoreIndexedOp store;
  kernel.walk([&](neura::LoadIndexedOp candidate) {
    load = candidate;
    ++loads;
    auto inputIndex = parseKernelInputReference(candidate->getAttr("lhs_value"));
    badMemory |=
        (!inputIndex || *inputIndex != 0) ||
        (candidate.getBase() && candidate.getBase() != kernelInput) ||
        candidate.getIndices().size() != 2;
  });
  kernel.walk([&](neura::StoreIndexedOp candidate) {
    store = candidate;
    ++stores;
    auto outputIndex =
        parseKernelInputReference(candidate->getAttr("rhs_value"));
    badMemory |=
        (!outputIndex || *outputIndex != 4) ||
        (candidate.getBase() && candidate.getBase() != kernelOutput) ||
        candidate.getIndices().size() != 1;
  });
  kernel.walk([&](Operation *operation) {
    if (operation == kernel.getOperation())
      return;
    StringRef name = operation->getName().getStringRef();
    if (name == "neura.counter" || name == "neura.reserve" ||
        name == "neura.phi" || name == "neura.ctrl_mov" ||
        name == "neura.grant_predicate" || name == "neura.constant" ||
        name == "neura.cast" || name == "neura.icmp" ||
        name == "neura.not" || name == "neura.add" ||
        name == "neura.div" || name == "neura.load_indexed" ||
        name == "neura.store_indexed" || name == "neura.yield")
      return;
    badMemory = true;
  });
  if (loads != 1 || stores != 1 || badMemory || !load || !store ||
      (load.getBase() && load.getBase() != kernelInput) ||
      load.getIndices().size() != 2 ||
      (store.getBase() && store.getBase() != kernelOutput) ||
      store.getIndices().size() != 1 ||
      parseKernelInputReference(load->getAttr("lhs_value")) !=
          std::optional<unsigned>(0) ||
      parseKernelInputReference(store->getAttr("rhs_value")) !=
          std::optional<unsigned>(4))
    return rejectRadar(source,
                       "Radar producer Neura DFG must load input[chirp,range] "
                       "and store output[chirp] exactly once");

  // The normalized Radar kernel has one explicit Neura chirp counter and
  // carries the range index and accumulator through three local reserve/phi
  // states.  Prove that the indexed accesses use exactly those states and
  // that every feedback value is reinitialized inside the cloned kernel.
  auto isNamed = [](Value value, StringRef name) {
    return value && value.getDefiningOp() &&
           value.getDefiningOp()->getName().getStringRef() == name;
  };
  auto isReserve = [&](Value value) { return isNamed(value, "neura.reserve"); };
  auto isFoldedKernelInput = [](Value value, unsigned index) {
    Operation *definition = value ? value.getDefiningOp() : nullptr;
    if (!definition || definition->getName().getStringRef() !=
                           "neura.constant")
      return false;
    return parseKernelInputReference(definition->getAttr("value")) ==
           std::optional<unsigned>(index);
  };
  Value outerCurrent;
  Value rangePhi;
  Value rangeIndex;
  Value accumulatorPhi;
  unsigned reserveCount = 0;
  unsigned phiCount = 0;
  kernelBody.walk([&](Operation *operation) {
    StringRef name = operation->getName().getStringRef();
    if (name == "neura.reserve")
      ++reserveCount;
    if (name != "neura.phi" || operation->getNumOperands() != 2 ||
        operation->getNumResults() != 1)
      return;
    ++phiCount;
    if (operation->getOperand(1) == kernelChirp &&
        isReserve(operation->getOperand(0)))
      outerCurrent = operation->getResult(0);
    if (isFoldedKernelInput(operation->getOperand(1), 2) &&
        isReserve(operation->getOperand(0)))
      accumulatorPhi = operation->getResult(0);
  });
  if (!outerCurrent || !accumulatorPhi || reserveCount != 3 || phiCount != 3)
    return rejectRadar(source,
                       "Radar producer range/accumulator feedback is not "
                       "the reset local form");
  auto loadOuter = load.getIndices()[0].getDefiningOp();
  auto loadRange = load.getIndices()[1].getDefiningOp();
  auto storeOuter = store.getIndices()[0].getDefiningOp();
  if (!loadOuter || !loadRange || !storeOuter ||
      loadOuter->getName().getStringRef() != "neura.grant_predicate" ||
      loadRange->getName().getStringRef() != "neura.grant_predicate" ||
      storeOuter->getName().getStringRef() != "neura.grant_predicate" ||
      loadOuter->getNumOperands() != 2 || loadRange->getNumOperands() != 2 ||
      storeOuter->getNumOperands() != 2 ||
      loadOuter->getOperand(0) != outerCurrent ||
      loadRange->getOperand(1) != loadOuter->getOperand(1) ||
      storeOuter->getOperand(0) != outerCurrent)
    return rejectRadar(source,
                       "Radar producer indexed accesses do not share the "
                       "same chirp predicate");
  Value predicate = loadOuter->getOperand(1);
  rangeIndex = loadRange->getOperand(0);
  auto rangeCast = rangeIndex.getDefiningOp();
  if (!rangeCast || rangeCast->getName().getStringRef() != "neura.cast" ||
      rangeCast->getNumOperands() != 1 ||
      rangeCast->getAttrOfType<StringAttr>("cast_type") == nullptr ||
      rangeCast->getAttrOfType<StringAttr>("cast_type").getValue() !=
          "int_to_index")
    return rejectRadar(source,
                       "Radar producer range index is not the reset phi "
                       "state");
  rangePhi = rangeCast->getOperand(0);
  auto rangePhiOp = rangePhi.getDefiningOp();
  if (!rangePhiOp || rangePhiOp->getName().getStringRef() != "neura.phi" ||
      rangePhiOp->getNumOperands() != 2 ||
      !isReserve(rangePhiOp->getOperand(0)))
    return rejectRadar(source,
                       "Radar producer range index lacks a reserve/phi "
                       "initialization");
  auto initialCast = rangePhiOp->getOperand(1).getDefiningOp();
  auto initialConstant = initialCast && initialCast->getNumOperands() == 1
                             ? initialCast->getOperand(0).getDefiningOp()
                             : nullptr;
  auto zero = initialConstant
                  ? initialConstant->getAttrOfType<IntegerAttr>("value")
                  : IntegerAttr();
  if (!initialCast || initialCast->getName().getStringRef() != "neura.cast" ||
      initialCast->getAttrOfType<StringAttr>("cast_type") == nullptr ||
      initialCast->getAttrOfType<StringAttr>("cast_type").getValue() !=
          "index_to_int" || !initialConstant ||
      initialConstant->getName().getStringRef() != "neura.constant" || !zero ||
      zero.getInt() != 0)
    return rejectRadar(source,
                       "Radar producer range phi is not initialized to zero");
  auto predicateOp = predicate.getDefiningOp();
  if (!predicateOp || predicateOp->getName().getStringRef() != "neura.icmp" ||
      predicateOp->getNumOperands() != 1 ||
      predicateOp->getOperand(0) != rangeIndex ||
      predicateOp->getAttrOfType<StringAttr>("cmpType") == nullptr ||
      predicateOp->getAttrOfType<StringAttr>("cmpType").getValue() != "slt" ||
      predicateOp->getAttrOfType<StringAttr>("rhs_value") == nullptr ||
      predicateOp->getAttrOfType<StringAttr>("rhs_value").getValue() !=
          "%input1")
    return rejectRadar(source,
                       "Radar producer range predicate is not input1-bounded");
  auto storePredicate = storeOuter->getOperand(1).getDefiningOp();
  if (!storePredicate || storePredicate->getName().getStringRef() !=
                             "neura.not" ||
      storePredicate->getNumOperands() != 1 ||
      storePredicate->getOperand(0) != predicate)
    return rejectRadar(source,
                       "Radar producer store is not guarded by the inverse "
                       "range predicate");
  Value rangeIncrement;
  Value accumulatorUpdate;
  // Check the complete feedback wiring. Merely finding an add and initialized
  // phis does not prove that the ctrl_mov targets reset the same local state.
  for (Operation &operation : kernelBody) {
    StringRef name = operation.getName().getStringRef();
    if (name == "neura.add" && operation.getNumOperands() == 1) {
      Operation *rangeUpdate = operation.getOperand(0).getDefiningOp();
      auto increment = operation.getAttrOfType<IntegerAttr>("rhs_value");
      if (rangeUpdate &&
          rangeUpdate->getName().getStringRef() == "neura.grant_predicate" &&
          rangeUpdate->getNumOperands() == 2 &&
          rangeUpdate->getOperand(0) == rangeIndex &&
          rangeUpdate->getOperand(1) == predicate && increment &&
          increment.getInt() == 1)
        rangeIncrement = operation.getResult(0);
    }
    if (name == "neura.add" && operation.getNumOperands() == 2) {
      Value other;
      if (operation.getOperand(0) == load.getResult())
        other = operation.getOperand(1);
      else if (operation.getOperand(1) == load.getResult())
        other = operation.getOperand(0);
      Operation *grant = other ? other.getDefiningOp() : nullptr;
      if (grant && grant->getName().getStringRef() == "neura.grant_predicate" &&
          grant->getNumOperands() == 2 &&
          grant->getOperand(0) == accumulatorPhi &&
          grant->getOperand(1) == predicate)
        accumulatorUpdate = operation.getResult(0);
    }
  }
  if (!rangeIncrement || !accumulatorUpdate)
    return rejectRadar(source,
                       "Radar producer feedback does not advance the range "
                       "and accumulator from the current load");
  Value outerReserve = outerCurrent.getDefiningOp()->getOperand(0);
  Value rangeReserve = rangePhiOp->getOperand(0);
  Value accumulatorReserve = accumulatorPhi.getDefiningOp()->getOperand(0);
  unsigned feedbackCount = 0;
  bool outerFeedback = false, rangeFeedback = false, accumulatorFeedback = false;
  for (Operation &operation : kernelBody) {
    if (operation.getName().getStringRef() != "neura.ctrl_mov")
      continue;
    ++feedbackCount;
    if (operation.getNumOperands() != 2)
      return rejectRadar(source, "Radar producer has an invalid feedback ABI");
    Value from = operation.getOperand(0), to = operation.getOperand(1);
    if (to == outerReserve && from == loadOuter->getResult(0)) {
      outerFeedback = true;
      continue;
    }
    if (to == accumulatorReserve && from == accumulatorUpdate) {
      accumulatorFeedback = true;
      continue;
    }
    Operation *cast = from.getDefiningOp();
    auto castType = cast ? cast->getAttrOfType<StringAttr>("cast_type")
                         : StringAttr();
    if (to == rangeReserve && cast &&
        cast->getName().getStringRef() == "neura.cast" &&
        cast->getNumOperands() == 1 && castType &&
        castType.getValue() == "index_to_int" &&
        cast->getOperand(0) == rangeIncrement) {
      rangeFeedback = true;
      continue;
    }
    return rejectRadar(source,
                       "Radar producer feedback crosses its proved local state");
  }
  if (feedbackCount != 3 || !outerFeedback || !rangeFeedback ||
      !accumulatorFeedback)
    return rejectRadar(source,
                       "Radar producer feedback does not cover each local state");
  Operation *divide = store.getValue().getDefiningOp();
  Operation *finalAccumulator =
      divide && divide->getNumOperands() == 1
          ? divide->getOperand(0).getDefiningOp()
          : nullptr;
  if (!divide || divide->getName().getStringRef() != "neura.div" ||
      parseKernelInputReference(divide->getAttr("rhs_value")) !=
          std::optional<unsigned>(3) ||
      !finalAccumulator ||
      finalAccumulator->getName().getStringRef() != "neura.grant_predicate" ||
      finalAccumulator->getNumOperands() != 2 ||
      finalAccumulator->getOperand(0) != accumulatorPhi ||
      finalAccumulator->getOperand(1) != storeOuter->getOperand(1))
    return rejectRadar(source,
                       "Radar producer store is not the completed row accumulator");
  return consumer;
}


static LogicalResult materializeRadarNormalizedProducer(func::FuncOp function,
                                              TaskflowTaskOp source,
                                              int64_t axis, int64_t replicas) {
  FailureOr<TaskflowTaskOp> validated =
      validateRadarNormalizedProducer(function, source, axis, replicas);
  if (failed(validated))
    return failure();
  TaskflowTaskOp consumer = *validated;
  constexpr int64_t chirps = 32;
  constexpr int64_t rangeBins = 64;
  const std::string sourceName = source.getTaskName().str();
  OpBuilder builder(source);
  SmallVector<Value> states;
  for (int64_t replicaId = 0; replicaId < replicas; ++replicaId) {
    const auto [lower, upper] =
        balancedShardBounds(chirps, replicaId, replicas);
    auto replica = cast<TaskflowTaskOp>(builder.clone(*source.getOperation()));
    replica->setAttr("task_name",
                     builder.getStringAttr(sourceName + ".replica." +
                                           std::to_string(replicaId)));
    replica->setAttr("amoeba.replica.parent_task",
                     builder.getStringAttr(sourceName));
    replica->setAttr("amoeba.replica.id", builder.getI64IntegerAttr(replicaId));
    replica->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicas));
    replica->setAttr("amoeba.replica.shard_axis", builder.getI64IntegerAttr(0));
    replica->setAttr("amoeba.replica.shard_lower",
                     builder.getI64IntegerAttr(lower));
    replica->setAttr("amoeba.replica.shard_upper",
                     builder.getI64IntegerAttr(upper));
    replica->removeAttr(kSemanticIncomingEdgesAttr);
    replica->setAttr(
        kTilingOutputRegionLowersAttr,
        builder.getArrayAttr({builder.getDenseI64ArrayAttr({lower})}));
    replica->setAttr(
        kTilingOutputRegionUppersAttr,
        builder.getArrayAttr({builder.getDenseI64ArrayAttr({upper})}));

    Block &replicaBody = replica.getBody().front();
    SmallVector<TaskflowCounterOp> clonedCounters;
    neura::KernelOp clonedKernel;
    for (Operation &operation : replicaBody) {
      if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
        clonedCounters.push_back(counter);
      else if (auto candidate = dyn_cast<neura::KernelOp>(&operation))
        clonedKernel = candidate;
    }
    if ((clonedCounters.size() != 1 && clonedCounters.size() != 2) ||
        !clonedKernel ||
        (clonedCounters.size() == 2 &&
         !clonedCounters[1].getCounterIndex().use_empty()))
      return rejectRadar(source,
                         "Radar replica clone lost the chirp scaffold or "
                         "complete Neura kernel");
    // The source range Taskflow counter is only a dead scheduling scaffold;
    // the complete range reduction remains in the Neura DFG below.  Remove
    // that scaffold so the completion state has the rank-one chirp region
    // required by the rank-one dc_i memref and its completion join.
    if (clonedCounters.size() == 2) {
      clonedCounters[1].erase();
      clonedCounters.pop_back();
    }

    OpBuilder counterBuilder(clonedCounters[0]);
    Value lowerValue =
        counterBuilder.create<arith::ConstantIndexOp>(source.getLoc(), lower);
    Value upperValue =
        counterBuilder.create<arith::ConstantIndexOp>(source.getLoc(), upper);
    clonedCounters[0].getLowerBoundMutable().assign(lowerValue);
    clonedCounters[0].getUpperBoundMutable().assign(upperValue);
    clonedCounters[0].setCounterDynamismAttr(
        builder.getStringAttr("constant_bound"));
    SmallVector<neura::CounterOp> clonedNeuraCounters;
    clonedKernel.walk(
        [&](neura::CounterOp counter) { clonedNeuraCounters.push_back(counter); });
    for (neura::CounterOp counter : clonedNeuraCounters) {
      auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
      if (!id || id.getInt() != 0)
        continue;
      OpBuilder kernelBuilder =
          OpBuilder::atBlockBegin(&clonedKernel.getBody().front());
      Value kernelLower =
          kernelBuilder.create<arith::ConstantIndexOp>(source.getLoc(), lower);
      Value kernelUpper =
          kernelBuilder.create<arith::ConstantIndexOp>(source.getLoc(), upper);
      counter.getLowerBoundMutable().assign(kernelLower);
      counter.getUpperBoundMutable().assign(kernelUpper);
      counter->setAttr("counter_dynamism",
                       builder.getStringAttr("constant_bound"));
      if (counter->hasAttr("lower_bound_value"))
        counter->setAttr("lower_bound_value", builder.getIndexAttr(lower));
      if (counter->hasAttr("upper_bound_value"))
        counter->setAttr("upper_bound_value", builder.getIndexAttr(upper));
      if (counter->hasAttr("step_value"))
        counter->setAttr("step_value", builder.getIndexAttr(1));
    }
    // `trip_count` is the canonical analytical Taskflow execution count.  A
    // prepared Radar input without that attribute derives 32 executions from
    // its outer counter; do not silently replace that contract with the
    // Neura body's inner 64-element recurrence.  When a source explicitly
    // carries the full work count, preserve the established work-count
    // materialization and shard it consistently.
    if (source->hasAttr("trip_count")) {
      int64_t shardTripCount = (upper - lower) * rangeBins;
      replica->setAttr("trip_count",
                       builder.getI64IntegerAttr(shardTripCount));
      if (replica->getAttrOfType<IntegerAttr>("amoeba.selected_trip_count"))
        replica->setAttr("amoeba.selected_trip_count",
                         builder.getI64IntegerAttr(shardTripCount));
    } else {
      replica->removeAttr("trip_count");
      replica->removeAttr("amoeba.selected_trip_count");
    }
    states.push_back(replica.getDoneWrites().front());
  }

  auto outputType = cast<MemRefType>(source.getWillWrites().front().getType());
  SmallVector<int64_t> outputShape;
  if (outputType.hasStaticShape())
    outputShape.append(outputType.getShape().begin(), outputType.getShape().end());
  else
    outputShape = {chirps};
  auto staticOutputType = MemRefType::get(
      outputShape, outputType.getElementType(), outputType.getLayout(),
      outputType.getMemorySpace());
  Value outputStorage = source.getWillWrites().front();
  if (outputStorage.getType() != staticOutputType) {
    if (!memref::CastOp::areCastCompatible({outputStorage.getType()},
                                            {staticOutputType}))
      return rejectRadar(source,
                         "Radar producer output has no compatible static "
                         "completion view");
    outputStorage = builder.create<memref::CastOp>(
        source.getLoc(), staticOutputType, outputStorage);
  }
  SmallVector<Value> staticStates;
  staticStates.reserve(states.size());
  for (Value state : states) {
    if (state.getType() == staticOutputType) {
      staticStates.push_back(state);
      continue;
    }
    if (!memref::CastOp::areCastCompatible({state.getType()},
                                            {staticOutputType}))
      return rejectRadar(source,
                         "Radar producer replica state has no compatible "
                         "static completion view");
    staticStates.push_back(builder.create<memref::CastOp>(
        source.getLoc(), staticOutputType, state));
  }
  auto join = builder.create<TaskflowJoinOp>(
      source.getLoc(), staticOutputType, staticStates, outputStorage,
      builder.getI64IntegerAttr(0), builder.getDenseI64ArrayAttr({0}),
      builder.getDenseI64ArrayAttr({chirps}));
  const bool sourceHasExplicitTripCount = source->hasAttr("trip_count");
  join->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
  join->setAttr("amoeba.replica.completion_only", builder.getUnitAttr());
  Value replacement = join.getJoined();
  if (replacement.getType() != source.getDoneWrites().front().getType())
    replacement = builder.create<memref::CastOp>(
        source.getLoc(), source.getDoneWrites().front().getType(), replacement);
  source.getDoneWrites().front().replaceAllUsesWith(replacement);
  if (failed(makeConsumerTensorWideForReplicas(consumer, replacement,
                                               sourceName, builder, replicas,
                                               true)))
    return failure();
  source.erase();
  function->setAttr("amoeba.replica.materialized_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.parent_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicas));
  function->setAttr("amoeba.replica.shard_axis", builder.getI64IntegerAttr(0));
  const int64_t canonicalTripCount = sourceHasExplicitTripCount
                                         ? chirps * rangeBins
                                         : chirps;
  function->setAttr("amoeba.replica.total_trip_count",
                    builder.getI64IntegerAttr(canonicalTripCount));
  function->setAttr("amoeba.replica.shard_trip_count",
                    builder.getI64IntegerAttr(canonicalTripCount / replicas));
  return success();
}

static bool hasRadarLegacyAbi(TaskflowTaskOp source) {
  if (!source)
    return false;
  SmallVector<TaskflowCounterOp> counters;
  neura::KernelOp kernel;
  unsigned kernels = 0;
  for (Operation &operation : source.getBody().front()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      counters.push_back(counter);
      continue;
    }
    if (auto candidate = dyn_cast<neura::KernelOp>(&operation)) {
      kernel = candidate;
      ++kernels;
    }
  }
  if (counters.size() != 2 || kernels != 1 || !kernel ||
      kernel.getIterArgsInit().size() != 1)
    return false;
  SmallVector<neura::CounterOp> neuraCounters;
  kernel.walk([&](neura::CounterOp counter) { neuraCounters.push_back(counter); });
  return neuraCounters.size() == 2;
}

static bool hasRadarNormalizedAbi(TaskflowTaskOp source) {
  if (!source)
    return false;
  SmallVector<TaskflowCounterOp> counters;
  neura::KernelOp kernel;
  unsigned kernels = 0;
  for (Operation &operation : source.getBody().front()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      counters.push_back(counter);
      continue;
    }
    if (auto candidate = dyn_cast<neura::KernelOp>(&operation)) {
      kernel = candidate;
      ++kernels;
    }
  }
  if ((counters.size() != 1 && counters.size() != 2) || kernels != 1 ||
      !kernel || !kernel.getIterArgsInit().empty())
    return false;
  SmallVector<neura::CounterOp> neuraCounters;
  kernel.walk([&](neura::CounterOp counter) { neuraCounters.push_back(counter); });
  return neuraCounters.size() == 1;
}

static LogicalResult materializeRadarProducer(func::FuncOp function,
                                              TaskflowTaskOp source,
                                              int64_t axis, int64_t replicas) {
  if (hasRadarLegacyAbi(source))
    return materializeRadarLegacyProducer(function, source, axis, replicas);
  if (hasRadarNormalizedAbi(source))
    return materializeRadarNormalizedProducer(function, source, axis, replicas);
  return rejectRadar(source,
                     "Radar producer has neither the legacy two-counter "
                     "iter-arg ABI nor the normalized one-counter ABI");
}

static bool isLlamaReadArgument(Value value, func::FuncOp function);

// The normalized source graph still contains a Taskflow hyperblock.  Prove
// this exact two-input integer GEMM before making independent M shards.  In
// particular, the sole reduction must start at zero for every (M,N) point;
// splitting a loop-carried K reduction would not have this property.
static FailureOr<TaskflowTaskOp>
validateLlamaSourceProducer(func::FuncOp function, TaskflowTaskOp source,
                            int64_t axis, int64_t replicas) {
  if (axis != 0 || source.getTaskName() != "Task_0.k.0")
    return rejectLlama(source,
                       "source LLaMA replica requires Task_0.k.0 and M axis");
  if (!isSupportedReplicaCount(replicas) || replicas < 2 ||
      !hasBalancedShards(320, replicas))
    return rejectLlama(source,
                       "source LLaMA replica factor must be 2, 4, or 8 and "
                       "divide the 320-row M domain");
  if (auto dlp = source->getAttrOfType<BoolAttr>("dlp_replicable");
      dlp && !dlp.getValue())
    return rejectLlama(source,
                       "source LLaMA replica rejected explicit "
                       "dlp_replicable=false");
  auto incoming = source->getAttrOfType<ArrayAttr>(kSemanticIncomingEdgesAttr);
  if (!incoming || !incoming.empty())
    return rejectLlama(source,
                       "source LLaMA replica requires an empty incoming "
                       "semantic edge list");
  auto kind = source->getAttrOfType<StringAttr>("amoeba.semantic.task_kind");
  auto policy = source->getAttrOfType<StringAttr>("amoeba.semantic.k_policy");
  auto mRange =
      source->getAttrOfType<DenseI64ArrayAttr>("amoeba.semantic.M_range");
  auto nRange =
      source->getAttrOfType<DenseI64ArrayAttr>("amoeba.semantic.N_range");
  auto kRange =
      source->getAttrOfType<DenseI64ArrayAttr>("amoeba.semantic.K_range");
  auto exactRange = [](DenseI64ArrayAttr attr, int64_t upper) {
    return attr && attr.asArrayRef().size() == 2 && attr.asArrayRef()[0] == 0 &&
           attr.asArrayRef()[1] == upper;
  };
  if (!kind || kind.getValue() != "gemm_producer" || !policy ||
      policy.getValue() != "sequential" || !exactRange(mRange, 320) ||
      !exactRange(nRange, 256) || !exactRange(kRange, 256))
    return rejectLlama(source, "source LLaMA requires normalized sequential "
                               "M=320 N=256 K=256 GEMM metadata");
  if (source.getWillReads().size() != 2 || source.getWillWrites().size() != 1 ||
      source.getDoneWrites().size() != 1 ||
      source.getOriginalReadMemrefs().size() != 2 ||
      source.getOriginalWriteMemrefs().size() != 1 ||
      source.getValueInputs().size() != 2 || !source.getDoneReads().empty() ||
      !source.getValueOutputs().empty())
    return rejectLlama(source, "source LLaMA replica requires two reads, one "
                               "write, and exactly two value inputs");
  Value read0 = source.getWillReads()[0];
  Value read1 = source.getWillReads()[1];
  if (read0 == read1 || source.getOriginalReadMemrefs()[0] != read0 ||
      source.getOriginalReadMemrefs()[1] != read1 ||
      !isLlamaReadArgument(read0, function) ||
      !isLlamaReadArgument(read1, function) ||
      !cast<MemRefType>(read0.getType()).isDynamicDim(0) ||
      !cast<MemRefType>(read1.getType()).isDynamicDim(0) ||
      compileTimeIndex(source.getValueInputs()[0], source) != 320 ||
      compileTimeIndex(source.getValueInputs()[1], source) != 0)
    return rejectLlama(source, "source LLaMA inputs must be distinct i32 "
                               "rank-two function arguments with M=320 and "
                               "zero initial accumulator");
  for (Value input : {read0, read1})
    for (OpOperand &use : input.getUses()) {
      auto task = dyn_cast<TaskflowTaskOp>(use.getOwner());
      if (!task || !llvm::is_contained(task.getWillReads(), input) ||
          llvm::is_contained(task.getWillWrites(), input))
        return rejectLlama(source, "source LLaMA read input has an alias or "
                                   "a possible function-level writer");
    }

  Value output = source.getWillWrites().front();
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!outputType || !outputType.hasStaticShape() ||
      outputType.getShape() != ArrayRef<int64_t>({512, 256}) ||
      !outputType.getElementType().isInteger(32) ||
      !output.getDefiningOp<memref::AllocaOp>() ||
      source.getOriginalWriteMemrefs().front() != output)
    return rejectLlama(source, "source LLaMA output must be a private "
                               "512x256xi32 alloca");
  Value produced = source.getDoneWrites().front();
  TaskflowTaskOp consumer;
  unsigned directUses = 0;
  for (OpOperand &use : produced.getUses()) {
    auto candidate = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (!candidate || candidate.getTaskName() != "Task_3" ||
        !llvm::is_contained(candidate.getWillReads(), produced))
      return rejectLlama(source, "source LLaMA result must feed only Task_3");
    consumer = candidate;
    ++directUses;
  }
  if (!consumer || directUses != 1 ||
      llvm::count(consumer.getWillReads(), produced) != 1 ||
      consumer.getOriginalReadMemrefs().empty() ||
      consumer.getOriginalReadMemrefs().front() != output)
    return rejectLlama(source,
                       "source LLaMA requires one direct Task_3 RAW edge");
  for (OpOperand &use : output.getUses()) {
    Operation *owner = use.getOwner();
    if (owner != source.getOperation() && owner != consumer.getOperation())
      return rejectLlama(source,
                         "source LLaMA output has an alias or extra user");
  }
  if (auto incoming =
          consumer->getAttrOfType<ArrayAttr>(kSemanticIncomingEdgesAttr))
    for (Attribute edge : incoming)
      if (auto text = dyn_cast<StringAttr>(edge);
          text && text.getValue().contains("tile_local"))
        return rejectLlama(source,
                           "source LLaMA cannot replace a tile-local edge");

  Block &body = source.getBody().front();
  SmallVector<TaskflowCounterOp> counters;
  TaskflowHyperblockOp hyperblock;
  unsigned hyperblocks = 0;
  for (Operation &operation : body) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      counters.push_back(counter);
    else if (auto candidate = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      hyperblock = candidate;
      ++hyperblocks;
    } else if (!isa<arith::ConstantOp, arith::ConstantIndexOp, TaskflowYieldOp>(
                   &operation))
      return rejectLlama(source,
                         "source LLaMA task body has an unsupported op");
  }
  if (counters.size() != 2 || hyperblocks != 1 ||
      counters[0].getParentIndex() ||
      counters[1].getParentIndex() != counters[0].getCounterIndex() ||
      hyperblock.getIndices().size() != 2 ||
      hyperblock.getIndices()[0] != counters[0].getCounterIndex() ||
      hyperblock.getIndices()[1] != counters[1].getCounterIndex() ||
      !hyperblock.getBody().hasOneBlock())
    return rejectLlama(source, "source LLaMA needs exactly M/N counters and "
                               "one matching hyperblock");
  for (auto [index, counter] : llvm::enumerate(counters))
    if (constantIndex(counter.getLowerBound()) != 0 ||
        compileTimeIndex(counter.getUpperBound(), source) !=
            (index == 0 ? 320 : 256) ||
        constantIndex(counter.getStep()) != 1)
      return rejectLlama(source, "source LLaMA M/N counters must be 0..320/256 "
                                 "with unit steps");

  Block &block = hyperblock.getBody().front();
  if (block.getNumArguments() != 2)
    return rejectLlama(source, "source LLaMA hyperblock needs M/N indices");
  scf::ForOp reduction;
  memref::StoreOp store;
  unsigned operations = 0;
  unsigned reductions = 0;
  unsigned stores = 0;
  for (Operation &operation : block.without_terminator()) {
    ++operations;
    if (auto candidate = dyn_cast<scf::ForOp>(&operation)) {
      reduction = candidate;
      ++reductions;
    } else if (auto candidate = dyn_cast<memref::StoreOp>(&operation)) {
      store = candidate;
      ++stores;
    } else if (!isa<arith::ConstantOp, arith::ConstantIndexOp>(&operation))
      return rejectLlama(source,
                         "source LLaMA hyperblock has an unsupported op");
  }
  if (reductions != 1 || stores != 1 || operations < 2 ||
      reduction.getInitArgs().size() != 1 ||
      reduction.getInitArgs()[0] != body.getArgument(4) ||
      reduction.getNumResults() != 1 ||
      constantIndex(reduction.getLowerBound()) != 0 ||
      constantIndex(reduction.getUpperBound()) != 256 ||
      constantIndex(reduction.getStep()) != 1 ||
      store.getMemRef() != body.getArgument(2) ||
      store.getIndices().size() != 2 ||
      store.getIndices()[0] != block.getArgument(0) ||
      store.getIndices()[1] != block.getArgument(1) ||
      store.getValue() != reduction.getResult(0))
    return rejectLlama(source, "source LLaMA needs a full K=256 reduction "
                               "and one direct M/N output store");
  Block &loop = *reduction.getBody();
  if (loop.getNumArguments() != 2)
    return rejectLlama(source, "source LLaMA reduction needs one accumulator");
  SmallVector<memref::LoadOp> loads;
  arith::MulIOp multiply;
  arith::AddIOp add;
  unsigned loopOps = 0;
  for (Operation &operation : loop.without_terminator()) {
    ++loopOps;
    if (auto candidate = dyn_cast<memref::LoadOp>(&operation))
      loads.push_back(candidate);
    else if (auto candidate = dyn_cast<arith::MulIOp>(&operation))
      multiply = candidate;
    else if (auto candidate = dyn_cast<arith::AddIOp>(&operation))
      add = candidate;
    else
      return rejectLlama(source,
                         "source LLaMA K loop contains an unsupported op");
  }
  auto yield = dyn_cast<scf::YieldOp>(loop.getTerminator());
  if (loopOps != 4 || loads.size() != 2 || !multiply || !add || !yield ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != add.getResult() ||
      add.getLhs() != loop.getArgument(1) ||
      add.getRhs() != multiply.getResult() ||
      multiply.getLhs() != loads[0].getResult() ||
      multiply.getRhs() != loads[1].getResult() ||
      loads[0].getMemRef() != body.getArgument(0) ||
      loads[1].getMemRef() != body.getArgument(1) ||
      loads[0].getIndices().size() != 2 || loads[1].getIndices().size() != 2 ||
      loads[0].getIndices()[0] != block.getArgument(0) ||
      loads[0].getIndices()[1] != loop.getArgument(0) ||
      loads[1].getIndices()[0] != loop.getArgument(0) ||
      loads[1].getIndices()[1] != block.getArgument(1))
    return rejectLlama(source, "source LLaMA K loop must compute "
                               "acc + read0[M,K] * read1[K,N]");
  return consumer;
}

static LogicalResult materializeLlamaSourceProducer(func::FuncOp function,
                                                    TaskflowTaskOp source,
                                                    int64_t axis, int64_t replicas) {
  FailureOr<TaskflowTaskOp> validated =
      validateLlamaSourceProducer(function, source, axis, replicas);
  if (failed(validated))
    return failure();
  TaskflowTaskOp consumer = *validated;
  constexpr int64_t seqLen = 320;
  constexpr int64_t hidden = 256;
  constexpr int64_t reduction = 256;
  const std::string sourceName = source.getTaskName().str();
  OpBuilder builder(source);
  SmallVector<Value> states;
  for (int64_t replicaId = 0; replicaId < replicas; ++replicaId) {
    const auto [lower, upper] =
        balancedShardBounds(seqLen, replicaId, replicas);
    auto replica = cast<TaskflowTaskOp>(builder.clone(*source.getOperation()));
    replica->setAttr("task_name",
                     builder.getStringAttr(sourceName + ".replica." +
                                           std::to_string(replicaId)));
    replica->setAttr("amoeba.replica.parent_task",
                     builder.getStringAttr(sourceName));
    replica->setAttr("amoeba.replica.id", builder.getI64IntegerAttr(replicaId));
    replica->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicas));
    replica->setAttr("amoeba.replica.shard_axis", builder.getI64IntegerAttr(0));
    replica->setAttr("amoeba.replica.shard_lower",
                     builder.getI64IntegerAttr(lower));
    replica->setAttr("amoeba.replica.shard_upper",
                     builder.getI64IntegerAttr(upper));
    replica->setAttr("amoeba.semantic.M_range",
                     builder.getDenseI64ArrayAttr({lower, upper}));
    const int64_t shardTripCount = (upper - lower) * hidden * reduction;
    replica->setAttr("trip_count",
                     builder.getI64IntegerAttr(shardTripCount));
    replica->setAttr("amoeba.selected_trip_count",
                     builder.getI64IntegerAttr(shardTripCount));
    replica->removeAttr(kSemanticIncomingEdgesAttr);
    replica->setAttr(
        kTilingOutputRegionLowersAttr,
        builder.getArrayAttr({builder.getDenseI64ArrayAttr({lower, 0})}));
    replica->setAttr(
        kTilingOutputRegionUppersAttr,
        builder.getArrayAttr({builder.getDenseI64ArrayAttr({upper, 256})}));

    TaskflowCounterOp mCounter;
    for (Operation &operation : replica.getBody().front())
      if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
        mCounter = counter;
        break;
      }
    OpBuilder counterBuilder(mCounter);
    Value lowerValue =
        counterBuilder.create<arith::ConstantIndexOp>(source.getLoc(), lower);
    Value upperValue =
        counterBuilder.create<arith::ConstantIndexOp>(source.getLoc(), upper);
    mCounter.getLowerBoundMutable().assign(lowerValue);
    mCounter.getUpperBoundMutable().assign(upperValue);
    mCounter.setCounterDynamismAttr(builder.getStringAttr("constant_bound"));
    states.push_back(replica.getDoneWrites().front());
  }
  auto output = cast<MemRefType>(source.getWillWrites().front().getType());
  auto join = builder.create<TaskflowJoinOp>(
      source.getLoc(), output, states, source.getWillWrites().front(),
      builder.getI64IntegerAttr(0), builder.getDenseI64ArrayAttr({0, 0}),
      builder.getDenseI64ArrayAttr({320, 256}));
  join->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
  join->setAttr("amoeba.replica.completion_only", builder.getUnitAttr());
  source.getDoneWrites().front().replaceAllUsesWith(join.getJoined());
  if (failed(makeConsumerTensorWideForReplicas(consumer, join.getJoined(),
                                               sourceName, builder, replicas,
                                               true)))
    return failure();
  source.erase();
  function->setAttr("amoeba.replica.materialized_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.parent_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicas));
  function->setAttr("amoeba.replica.shard_axis", builder.getI64IntegerAttr(0));
  function->setAttr("amoeba.replica.total_trip_count",
                    builder.getI64IntegerAttr(seqLen * hidden * reduction));
  function->setAttr("amoeba.replica.shard_trip_count",
                    builder.getI64IntegerAttr((seqLen / replicas) * hidden *
                                               reduction));
  return success();
}

static bool isLlamaReadArgument(Value value, func::FuncOp function) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &function.getBody().front())
    return false;
  auto type = dyn_cast<MemRefType>(argument.getType());
  return type && type.getRank() == 2 && type.getElementType().isInteger(32) &&
         type.getShape()[1] == 256;
}

// A later canonical matmul producer (currently Task_3 in the prepared LLaMA
// graph) consumes the private results of earlier canonical producers instead
// of function arguments.  Admit that form only when the complete storage
// provenance is explicit: the read is a direct Taskflow done-write result,
// its original memref is the producer's one private alloca, the producer has
// one write and no read/value outputs, and the result has exactly this one
// consumer.  A join, channel, block argument, or extra alias remains
// unsupported rather than being treated as read-only by assumption.
static bool isPrivateLlamaProducerRead(Value input, Value original,
                                       TaskflowTaskOp consumer) {
  auto producer = input.getDefiningOp<TaskflowTaskOp>();
  if (!producer || !llvm::is_contained(producer.getDoneWrites(), input) ||
      producer.getWillWrites().size() != 1 ||
      producer.getWillWrites().front() != original ||
      producer.getOriginalWriteMemrefs().size() != 1 ||
      producer.getOriginalWriteMemrefs().front() != original ||
      !producer.getDoneReads().empty() || !producer.getValueOutputs().empty() ||
      llvm::count(consumer.getOriginalReadMemrefs(), original) != 1)
    return false;
  auto type = dyn_cast<MemRefType>(original.getType());
  auto allocation = original.getDefiningOp<memref::AllocaOp>();
  if (!type || !type.hasStaticShape() || type.getRank() != 2 ||
      type.getShape()[0] != 512 || type.getShape()[1] != 256 || !allocation)
    return false;
  unsigned consumers = 0;
  for (OpOperand &use : input.getUses()) {
    if (use.getOwner() != consumer.getOperation())
      return false;
    ++consumers;
  }
  if (consumers != 1)
    return false;
  for (Operation *owner : original.getUsers())
    if (owner != producer.getOperation() && owner != consumer.getOperation() &&
        !isa<memref::DeallocOp>(owner))
      return false;
  return true;
}


// A replicated canonical producer publishes one completion-only join for its
// original private output.  The join is a storage-preserving barrier: it
// carries the exact original allocation as its base, its states cover the
// declared region without overlap, and no writer or alias may bypass that
// base.  Treating an arbitrary join as a read-only input would lose the
// producer/storage dependency, so every part of this proof is checked here.
static bool isLlamaCompletionJoinRead(Value input, Value original,
                                      TaskflowTaskOp consumer) {
  auto join = input.getDefiningOp<TaskflowJoinOp>();
  if (!join || join.getJoined() != input || join.getBase() != original ||
      !join->hasAttr("amoeba.replica.completion_only") ||
      !join->hasAttr("amoeba.semantic.completion_only"))
    return false;

  auto originalType = dyn_cast<MemRefType>(original.getType());
  if (!originalType || !originalType.hasStaticShape() ||
      originalType.getRank() != 2 || originalType.getShape()[0] != 512 ||
      originalType.getShape()[1] != 256 ||
      !original.getDefiningOp<memref::AllocaOp>())
    return false;

  // Task_0/1/2 canonical outputs expose a logical 320x256 producer domain in
  // their 512x256 private storage.  The join verifier proves exact coverage
  // of this region; the explicit comparison prevents accepting a join that
  // merely happens to have the right storage type.
  if (join.getAxis() > 1 ||
      join.getRegionLower() != ArrayRef<int64_t>({0, 0}) ||
      join.getRegionUpper() != ArrayRef<int64_t>({320, 256}) ||
      failed(verify(join.getOperation())))
    return false;

  unsigned joinedUses = 0;
  for (OpOperand &use : input.getUses()) {
    if (use.getOwner() != consumer.getOperation())
      return false;
    ++joinedUses;
  }
  if (joinedUses != 1 || llvm::count(consumer.getWillReads(), input) != 1 ||
      consumer.getOriginalReadMemrefs().size() !=
          consumer.getWillReads().size())
    return false;

  const size_t stateCount = join.getTileStates().size();
  if (stateCount < 2)
    return false;
  llvm::SmallDenseSet<int64_t, 8> replicaIds;
  llvm::SmallDenseSet<Operation *, 8> stateOperations;
  std::string parentName;
  int64_t replicaCount = -1;
  for (Value state : join.getTileStates()) {
    auto replica = state.getDefiningOp<TaskflowTaskOp>();
    auto parent = replica
                      ? replica->getAttrOfType<StringAttr>(
                            "amoeba.replica.parent_task")
                      : StringAttr();
    auto id = replica ? replica->getAttrOfType<IntegerAttr>(
                            "amoeba.replica.id")
                      : IntegerAttr();
    auto count = replica ? replica->getAttrOfType<IntegerAttr>(
                              "amoeba.replica.count")
                         : IntegerAttr();
    auto axis = replica ? replica->getAttrOfType<IntegerAttr>(
                             "amoeba.replica.shard_axis")
                        : IntegerAttr();
    if (!replica || !llvm::is_contained(replica.getDoneWrites(), state) ||
        replica.getWillWrites().size() != 1 ||
        replica.getWillWrites().front() != original ||
        replica.getOriginalWriteMemrefs().size() != 1 ||
        replica.getOriginalWriteMemrefs().front() != original ||
        !replica.getDoneReads().empty() || !replica.getValueOutputs().empty() ||
        llvm::is_contained(replica.getWillReads(), original) || !parent || !id ||
        !count || !axis || parent.getValue().empty() || id.getInt() < 0 ||
        id.getInt() >= static_cast<int64_t>(stateCount) ||
        axis.getInt() != static_cast<int64_t>(join.getAxis()) ||
        !replicaIds.insert(id.getInt()).second)
      return false;
    if (replicaCount < 0)
      replicaCount = count.getInt();
    if (count.getInt() != replicaCount ||
        replicaCount != static_cast<int64_t>(stateCount))
      return false;
    if (parentName.empty())
      parentName = parent.getValue().str();
    if (parent.getValue() != parentName)
      return false;
    stateOperations.insert(replica.getOperation());

    unsigned stateUses = 0;
    for (OpOperand &use : state.getUses()) {
      if (use.getOwner() != join.getOperation())
        return false;
      ++stateUses;
    }
    if (stateUses != 1)
      return false;
  }
  if (replicaIds.size() != stateCount || replicaCount < 2)
    return false;

  // The storage root may be written only by the covered replica states and
  // named by this join.  Deallocation is the sole non-compute exception.
  for (Operation *owner : original.getUsers()) {
    if (owner == join.getOperation() || isa<memref::DeallocOp>(owner))
      continue;
    if (auto task = dyn_cast<TaskflowTaskOp>(owner)) {
      if (stateOperations.contains(task.getOperation()) &&
          task.getWillWrites().size() == 1 &&
          task.getWillWrites().front() == original)
        continue;
      // Taskflow keeps the mapper-visible storage identity in the consumer's
      // original_read_memrefs operand list. The one consumer being proved
      // may therefore name the producer allocation even though its execution
      // input is the completion join. This is a read-only provenance use,
      // not an alias or a writer, and is allowed only for that consumer.
      if (task == consumer &&
          llvm::is_contained(task.getOriginalReadMemrefs(), original) &&
          !llvm::is_contained(task.getOriginalWriteMemrefs(), original) &&
          !llvm::is_contained(task.getWillWrites(), original))
        continue;
    }
    return false;
  }
  return true;
}

// Reverse-order composition (Task_3 first, then Task_0/1/2) leaves one
// producer done-write state with several read-only Task_3 replicas.  Accept
// that fanout only when it is the complete, source-owned replica set and all
// of those consumers are covered by one verified completion join.  This
// keeps a second writer, an alias, or an incomplete replica set fail-closed.
static bool collectLlamaProducerConsumers(
    TaskflowTaskOp source, Value output,
    SmallVectorImpl<TaskflowTaskOp> &consumers) {
  if (source.getDoneWrites().size() != 1)
    return false;
  Value produced = source.getDoneWrites().front();
  for (OpOperand &use : produced.getUses()) {
    auto candidate = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (!candidate || !llvm::is_contained(candidate.getWillReads(), produced) ||
        llvm::count(candidate.getWillReads(), produced) != 1 ||
        candidate.getWillReads().size() !=
            candidate.getOriginalReadMemrefs().size() ||
        candidate.getDoneReads().size() != 0 ||
        candidate.getValueOutputs().size() != 0)
      return false;
    unsigned inputIndex = 0;
    while (inputIndex < candidate.getWillReads().size() &&
           candidate.getWillReads()[inputIndex] != produced)
      ++inputIndex;
    if (inputIndex == candidate.getWillReads().size() ||
        candidate.getOriginalReadMemrefs()[inputIndex] != output ||
        candidate.getOriginalWriteMemrefs().size() !=
            candidate.getWillWrites().size() ||
        llvm::is_contained(candidate.getWillWrites(), output))
      return false;
    consumers.push_back(candidate);
  }
  if (consumers.empty())
    return false;
  if (consumers.size() == 1)
    return true;

  llvm::SmallDenseSet<int64_t, 8> replicaIds;
  std::string parentName;
  int64_t replicaCount = -1;
  TaskflowJoinOp completionJoin;
  Value consumerBase;
  llvm::SmallDenseSet<Value, 8> consumerStates;
  for (TaskflowTaskOp consumer : consumers) {
    auto parent = consumer->getAttrOfType<StringAttr>(
        "amoeba.replica.parent_task");
    auto id = consumer->getAttrOfType<IntegerAttr>("amoeba.replica.id");
    auto count = consumer->getAttrOfType<IntegerAttr>("amoeba.replica.count");
    if (!parent || !id || !count || parent.getValue() != "Task_3" ||
        id.getInt() < 0 || id.getInt() >= static_cast<int64_t>(consumers.size()) ||
        !replicaIds.insert(id.getInt()).second ||
        consumer.getDoneWrites().size() != 1 ||
        consumer.getWillWrites().size() != 1 ||
        consumer.getOriginalWriteMemrefs().size() != 1 ||
        consumer.getOriginalWriteMemrefs().front() !=
            consumer.getWillWrites().front())
      return false;
    if (replicaCount < 0)
      replicaCount = count.getInt();
    if (count.getInt() != replicaCount ||
        replicaCount != static_cast<int64_t>(consumers.size()))
      return false;
    if (parentName.empty())
      parentName = parent.getValue().str();
    if (parent.getValue() != parentName)
      return false;

    Value state = consumer.getDoneWrites().front();
    consumerStates.insert(state);
    TaskflowJoinOp candidateJoin;
    unsigned stateUses = 0;
    for (OpOperand &use : state.getUses()) {
      auto join = dyn_cast<TaskflowJoinOp>(use.getOwner());
      if (!join)
        return false;
      candidateJoin = join;
      ++stateUses;
    }
    if (stateUses != 1 || !candidateJoin)
      return false;
    if (!completionJoin)
      completionJoin = candidateJoin;
    if (completionJoin != candidateJoin)
      return false;
    if (!consumerBase)
      consumerBase = consumer.getWillWrites().front();
    if (consumerBase != consumer.getWillWrites().front())
      return false;
  }
  if (replicaIds.size() != consumers.size() || replicaCount < 2 ||
      !completionJoin || completionJoin.getBase() != consumerBase ||
      !completionJoin->hasAttr("amoeba.replica.completion_only") ||
      !completionJoin->hasAttr("amoeba.semantic.completion_only") ||
      failed(verify(completionJoin.getOperation())) ||
      completionJoin.getTileStates().size() != consumers.size())
    return false;
  for (Value state : completionJoin.getTileStates())
    if (!consumerStates.contains(state))
      return false;

  // The consumer output root must not be written or aliased by an operation
  // outside the complete replica set and its completion join.
  for (Operation *owner : consumerBase.getUsers()) {
    if (owner == completionJoin.getOperation() ||
        isa<memref::DeallocOp>(owner))
      continue;
    auto task = dyn_cast<TaskflowTaskOp>(owner);
    if (task && llvm::is_contained(consumers, task) &&
        task.getWillWrites().size() == 1 &&
        task.getWillWrites().front() == consumerBase)
      continue;
    // A downstream task may retain the same mapper-visible output allocation
    // in original_read_memrefs while consuming only the completion result.
    // Admit that provenance use only when it is read-only and explicitly
    // names this join; any other storage owner remains an alias/writer.
    if (task &&
        llvm::is_contained(task.getOriginalReadMemrefs(), consumerBase) &&
        !llvm::is_contained(task.getOriginalWriteMemrefs(), consumerBase) &&
        !llvm::is_contained(task.getWillWrites(), consumerBase) &&
        llvm::is_contained(task.getWillReads(), completionJoin.getJoined()))
      continue;
    return false;
  }
  for (OpOperand &use : completionJoin.getJoined().getUses()) {
    auto downstream = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (!downstream || !llvm::is_contained(downstream.getWillReads(),
                                           completionJoin.getJoined()) ||
        llvm::is_contained(downstream.getWillWrites(), consumerBase))
      return false;
  }
  return true;
}



// Converted Neura DFGs may carry a memory base through data-movement and
// predicate operations, or fold it into lhs_value/rhs_value as %inputN. Walk
// only the value chain needed to identify the kernel input; this keeps the
// replica proof independent of the particular dataflow lowering sequence.
static std::optional<unsigned>
traceKernelInput(Value value, neura::KernelOp kernel,
                 llvm::SmallDenseSet<Value, 16> &visited) {
  if (!value || !visited.insert(value).second)
    return std::nullopt;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() == &kernel.getBody().front() &&
        argument.getArgNumber() < kernel.getInputs().size())
      return argument.getArgNumber();
    return std::nullopt;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  if (auto constant = dyn_cast<neura::ConstantOp>(definition))
    if (auto index = parseKernelInputReference(constant->getAttr("value")))
      return index;
  for (Value operand : definition->getOperands())
    if (auto index = traceKernelInput(operand, kernel, visited))
      return index;
  return std::nullopt;
}

static std::optional<unsigned> traceKernelInput(Value value,
                                                neura::KernelOp kernel) {
  llvm::SmallDenseSet<Value, 16> visited;
  return traceKernelInput(value, kernel, visited);
}

static std::optional<unsigned>
getFoldedKernelInput(Operation *operation, StringRef attributeName) {
  return parseKernelInputReference(operation->getAttr(attributeName));
}

// The canonical LLaMA matmul producer is emitted by the AMOEBA-Test pipeline
// after the affine task has been converted to a mapped Neura kernel.  Its
// physical result is
// rank-two, while the kernel still carries a dead third Taskflow counter for
// the K loop.  The K loop is represented by the complete Neura DFG and must
// remain there; only the dead Taskflow scaffold is removed from each clone so
// that the completion join can prove a rank-two M/N region.
// The Neura dialect does not attach MemoryEffectOpInterface to its internal
// arithmetic/control dataflow operations.  These listed operations are cloned
// with the full kernel and do not name external memref storage.  Indexed
// loads/stores are checked separately against exact kernel input roles.
static bool isLlamaInternalDFGOp(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  return name == "neura.constant" || name == "neura.counter" ||
         name == "neura.grant_once" || name == "neura.reserve" ||
         name == "neura.phi_start" || name == "neura.phi" ||
         name == "neura.icmp" ||
         name == "neura.grant_predicate" || name == "neura.mul" ||
         name == "neura.not" || name == "neura.add" ||
         name == "neura.ctrl_mov" || name == "neura.extract_predicate" ||
         name == "neura.return_value";
}

// DataMovOp is a routing wrapper in a Neura kernel, but it has no memory-effect
// interface.  Treat it as an internal forwarding operation only for the exact
// well-typed, attribute-free form emitted by the canonical kernel conversion.
// Requiring the definition and use to stay in the kernel's single body block
// keeps this special case from forwarding values across region/control edges.
static bool isLlamaIdentityDataMove(Operation *operation,
                                    Block &kernelBody) {
  if (!isa<neura::DataMovOp>(operation) ||
      operation->getBlock() != &kernelBody ||
      operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
      operation->getNumRegions() != 0 || !operation->getAttrs().empty())
    return false;

  Value input = operation->getOperand(0);
  Type inputType = input.getType();
  auto dataType = dyn_cast<neura::PredicatedValue>(inputType);
  if (!dataType || inputType != operation->getResult(0).getType() ||
      !dataType.getPredicateType().isInteger(1))
    return false;

  if (auto argument = dyn_cast<BlockArgument>(input))
    return argument.getOwner() == &kernelBody;
  Operation *definition = input.getDefiningOp();
  return definition && definition->getBlock() == &kernelBody;
}

// A completion join relies on disjoint output cells, not just on different
// counter bounds. Predication forwards its first value; both alternatives of
// an index phi must retain the same coordinate. Arithmetic and opaque chains
// cannot establish this ownership proof.
static bool isReplicaOutputCoordinate(Value value, neura::CounterOp counter,
                                      unsigned depth = 0) {
  if (value == counter.getCurrentIndex())
    return true;
  if (!value || depth >= 64)
    return false;
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return false;
  StringRef name = operation->getName().getStringRef();
  if ((name == "neura.grant_predicate" || name == "neura.data_mov") &&
      operation->getNumOperands() >= 1)
    return isReplicaOutputCoordinate(operation->getOperand(0), counter,
                                     depth + 1);
  if (name == "neura.phi" && operation->getNumOperands() == 2)
    return isReplicaOutputCoordinate(operation->getOperand(0), counter,
                                     depth + 1) &&
           isReplicaOutputCoordinate(operation->getOperand(1), counter,
                                     depth + 1);
  return false;
}

static FailureOr<TaskflowTaskOp>
validateLlamaProducer(func::FuncOp function, TaskflowTaskOp source,
                      int64_t axis, int64_t replicas) {
  if (axis < 0 || axis > 1)
    return rejectLlama(source,
                       "canonical LLaMA replica supports only the M or N axis");
  const StringRef shardName = axis == 0 ? "M" : "N";
  // The generic classifier marks a reduction-bearing hyperblock as not DLP
  // replicable because it cannot prove arbitrary loop-carried state. This
  // canonical Task_0 path has a narrower proof below: the K state remains in
  // each complete Neura DFG, while only the independent M domain is split.
  // Therefore an absent generic dlp_replicable tag is expected here. If a
  // caller supplied an explicit negative tag, preserve fail-closed behavior.
  if (auto dlp = source->getAttrOfType<BoolAttr>("dlp_replicable"); dlp &&
      !dlp.getValue())
    return rejectLlama(source,
                       "LLaMA Task_0 replica rejected explicit "
                       "dlp_replicable=false");
  auto partitionable =
      source->getAttrOfType<DenseI32ArrayAttr>("dlp_partitionable_counter_ids");
  if (partitionable &&
      (partitionable.asArrayRef().size() != 2 ||
       partitionable.asArrayRef()[0] != 0 ||
       partitionable.asArrayRef()[1] != 1))
    return rejectLlama(
        source,
        "LLaMA Task_0 has an invalid partitionable counter declaration; "
        "expected M/N counters [0, 1]");

  if (source.getWillReads().size() != 2 ||
      source.getWillWrites().size() != 1 ||
      source.getOriginalReadMemrefs().size() != 2 ||
      source.getOriginalWriteMemrefs().size() != 1 ||
      source.getDoneReads().size() != 0 || source.getValueOutputs().size() != 0)
    return rejectLlama(source,
                       "LLaMA Task_0 replica requires two reads, one write, "
                       "and no read or value outputs");
  if (source.getWillReads()[0] == source.getWillReads()[1] ||
      source.getOriginalReadMemrefs()[0] ==
          source.getOriginalReadMemrefs()[1])
    return rejectLlama(source,
                       "canonical LLaMA inputs must be distinct storage values");
  for (auto [index, input] : llvm::enumerate(source.getWillReads()))
    if (!((source.getOriginalReadMemrefs()[index] == input &&
           isLlamaReadArgument(input, function)) ||
          isPrivateLlamaProducerRead(
              input, source.getOriginalReadMemrefs()[index], source) ||
          isLlamaCompletionJoinRead(
              input, source.getOriginalReadMemrefs()[index], source)))
      return rejectLlama(
          source,
          "canonical LLaMA inputs must be read-only function arguments or "
          "verified completion-only private producer results");

  Block &body = source.getBody().front();
  unsigned readCount = source.getWillReads().size();
  unsigned writeCount = source.getWillWrites().size();
  if (body.getNumArguments() !=
      readCount + writeCount + source.getValueInputs().size())
    return rejectLlama(
        source,
        "LLaMA Task_0 body arguments do not match its Taskflow operand "
        "segments");
  Value taskRead0 = body.getArgument(0);
  Value taskRead1 = body.getArgument(1);
  Value taskOutput = body.getArgument(readCount);
  unsigned seqValueIndex = std::numeric_limits<unsigned>::max();
  unsigned zeroValueIndex = std::numeric_limits<unsigned>::max();
  for (auto [index, value] : llvm::enumerate(source.getValueInputs())) {
    if (value.getType().isIndex())
      seqValueIndex = seqValueIndex == std::numeric_limits<unsigned>::max()
                          ? index
                          : std::numeric_limits<unsigned>::max();
    else if (value.getType().isInteger(32))
      zeroValueIndex = zeroValueIndex == std::numeric_limits<unsigned>::max()
                           ? index
                           : std::numeric_limits<unsigned>::max();
  }
  if (seqValueIndex == std::numeric_limits<unsigned>::max() ||
      zeroValueIndex == std::numeric_limits<unsigned>::max())
    return rejectLlama(
        source,
        "LLaMA Task_0 requires one index bound and one i32 initial "
        "accumulator value input");
  Value taskSeq = body.getArgument(readCount + writeCount + seqValueIndex);
  Value taskZero = body.getArgument(readCount + writeCount + zeroValueIndex);
  if (compileTimeIndex(source.getValueInputs()[zeroValueIndex], source) != 0)
    return rejectLlama(source,
                       "LLaMA Task_0 initial accumulator must be zero");

  Value output = source.getWillWrites().front();
  auto outputType = dyn_cast<MemRefType>(output.getType());
  auto allocation = output.getDefiningOp<memref::AllocaOp>();
  const int64_t logicalN =
      outputType && outputType.hasStaticShape() && outputType.getRank() == 2 &&
              outputType.getShape()[1] == 512
          ? 320
          : 256;
  if (!outputType || !outputType.hasStaticShape() || outputType.getRank() != 2 ||
      outputType.getShape()[0] != 512 ||
      (outputType.getShape()[1] != 256 && outputType.getShape()[1] != 512) ||
      !allocation || source.getOriginalWriteMemrefs().front() != output ||
      llvm::is_contained(source.getOriginalReadMemrefs(), output) ||
      llvm::is_contained(source.getWillReads(), output))
    return rejectLlama(source,
                       "LLaMA Task_0 output must be a private memref.alloca "
                       "of canonical 512x256 or 512x512 shape with no input "
                       "alias");
  const int64_t shardExtent = axis == 0 ? 320 : logicalN;
  if (!isSupportedReplicaCount(replicas) || replicas < 2 ||
      !hasBalancedShards(shardExtent, replicas))
    return rejectLlama(source,
                       (Twine("canonical LLaMA replica factor must be 2, 4, "
                              "or 8 and divide the ") + shardName +
                        " domain")
                           .str());
  SmallVector<TaskflowTaskOp> consumers;
  if (!collectLlamaProducerConsumers(source, output, consumers))
    return rejectLlama(
        source,
        "LLaMA producer consumers must be one direct edge or a complete "
        "read-only Task_3 replica set");
  for (OpOperand &use : output.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == source.getOperation() || isa<memref::DeallocOp>(owner))
      continue;
    if (auto consumer = dyn_cast<TaskflowTaskOp>(owner)) {
      if (llvm::is_contained(consumers, consumer) &&
          llvm::is_contained(consumer.getOriginalReadMemrefs(), output) &&
          !llvm::is_contained(consumer.getOriginalWriteMemrefs(), output))
        continue;
    }
    return rejectLlama(source,
                       "LLaMA Task_0 output has an alias or an additional "
                       "consumer");
  }

  // The source state may be consumed by one direct Task_3 edge or by a
  // complete read-only Task_3 replica set proven above.  A channel, a
  // tile-local semantic edge, or an unclassified input still invalidates the
  // completion-only replacement below.
  TaskflowTaskOp consumer = consumers.front();
  for (TaskflowTaskOp candidate : consumers)
    if (auto incoming = candidate->getAttrOfType<ArrayAttr>(
            kSemanticIncomingEdgesAttr))
      for (Attribute edge : incoming)
        if (auto text = dyn_cast<StringAttr>(edge);
            text && text.getValue().contains("tile_local"))
          return rejectLlama(
              source,
              "LLaMA producer cannot rewrite an existing tile-local consumer "
              "edge");
  for (Operation *owner : output.getUsers())
    if (owner != source.getOperation() && !isa<memref::DeallocOp>(owner)) {
      auto task = dyn_cast<TaskflowTaskOp>(owner);
      if (!task || !llvm::is_contained(consumers, task))
        return rejectLlama(
            source, "replica output has an additional storage alias or consumer");
    }

  SmallVector<TaskflowCounterOp> counters;
  neura::KernelOp kernel;
  unsigned kernels = 0;
  for (Operation &operation : body) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      counters.push_back(counter);
      continue;
    }
    if (auto candidate = dyn_cast<neura::KernelOp>(&operation)) {
      kernel = candidate;
      ++kernels;
      continue;
    }
    if (!isa<arith::ConstantOp, arith::ConstantIndexOp, TaskflowYieldOp>(
            &operation))
      return rejectLlama(
          source,
          "LLaMA Task_0 body must contain only constants, three counters, "
          "one Neura kernel, and a yield");
  }
  if (counters.size() != 3 || kernels != 1 || !kernel)
    return rejectLlama(source,
                       "LLaMA Task_0 requires exactly three Taskflow counters "
                       "and one Neura kernel");
  if (counters[0].getParentIndex() ||
      counters[1].getParentIndex() != counters[0].getCounterIndex() ||
      counters[2].getParentIndex() != counters[1].getCounterIndex())
    return rejectLlama(source,
                       "LLaMA Task_0 counters must form an M/N/K parent chain");
  auto checkTaskCounter = [&](TaskflowCounterOp counter, int64_t upper,
                              StringRef name) -> LogicalResult {
    if (constantIndex(counter.getLowerBound()) !=
            std::optional<int64_t>(0) ||
        compileTimeIndex(counter.getUpperBound(), source) !=
            std::optional<int64_t>(upper) ||
        constantIndex(counter.getStep()) != std::optional<int64_t>(1))
      return rejectLlama(source, (Twine("LLaMA Task_0 ") + name +
                                  " counter is not a compile-time 0..bound "
                                  "unit-step domain")
                                 .str());
    return success();
  };
  if (failed(checkTaskCounter(counters[0], 320, "M")) ||
      failed(checkTaskCounter(counters[1], logicalN, "N")) ||
      failed(checkTaskCounter(counters[2], 256, "K")))
    return failure();
  if (!counters[2].getCounterIndex().use_empty())
    return rejectLlama(
        source,
        "LLaMA Task_0 K Taskflow counter is live; refusing to discard a "
        "non-dead execution dependency");

  SmallVector<neura::CounterOp> neuraCounters;
  kernel.walk([&](neura::CounterOp counter) { neuraCounters.push_back(counter); });
  if (neuraCounters.size() != 3)
    return rejectLlama(source,
                       "LLaMA Task_0 Neura kernel must contain complete M/N/K "
                       "counter DFG");
  llvm::sort(neuraCounters, [](neura::CounterOp lhs, neura::CounterOp rhs) {
    auto lhsId = lhs->getAttrOfType<IntegerAttr>("counter_id");
    auto rhsId = rhs->getAttrOfType<IntegerAttr>("counter_id");
    // Keep malformed counters in a deterministic order so the subsequent
    // validation emits the precise missing/invalid-ID diagnostic instead of
    // dereferencing a null attribute while sorting.
    if (!lhsId || !rhsId)
      return static_cast<bool>(lhsId);
    return lhsId.getInt() < rhsId.getInt();
  });
  for (auto [index, counter] : llvm::enumerate(neuraCounters)) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    if (!id || id.getInt() != static_cast<int64_t>(index))
      return rejectLlama(source,
                         "LLaMA Task_0 Neura counter IDs must be 0,1,2");
    int64_t expected = index == 0 ? 320 : (index == 1 ? logicalN : 256);
    if (compileTimeCounterBound(counter, "lower_bound_value",
                                counter.getLowerBound(), source, kernel) !=
            std::optional<int64_t>(0) ||
        compileTimeCounterBound(counter, "upper_bound_value",
                                counter.getUpperBound(), source, kernel) !=
            std::optional<int64_t>(expected) ||
        compileTimeCounterBound(counter, "step_value", counter.getStep(),
                                source, kernel) != std::optional<int64_t>(1))
      return rejectLlama(
          source,
          "LLaMA Task_0 Neura M/N/K bounds must be compile-time constants");
  }

  SmallVector<Value> kernelRoles{taskOutput, taskRead0, taskRead1, taskSeq};
  SmallVector<unsigned> roleInputIndices(
      kernelRoles.size(), std::numeric_limits<unsigned>::max());
  for (auto [index, input] : llvm::enumerate(kernel.getInputs())) {
    auto role = llvm::find(kernelRoles, input);
    if (role != kernelRoles.end()) {
      unsigned roleIndex = role - kernelRoles.begin();
      if (roleInputIndices[roleIndex] !=
          std::numeric_limits<unsigned>::max())
        return rejectLlama(source,
                           "LLaMA Task_0 Neura kernel repeats a Taskflow "
                           "operand");
      roleInputIndices[roleIndex] = index;
      continue;
    }
    if (input != taskZero ||
        llvm::count(kernel.getInputs(), taskZero) != 1)
      return rejectLlama(
          source,
          "LLaMA Task_0 Neura kernel has an unexpected input outside its "
          "Taskflow operand roles");
  }
  if (llvm::any_of(roleInputIndices, [](unsigned index) {
        return index == std::numeric_limits<unsigned>::max();
      }) ||
      kernel.getIterArgsInit().size() != 1 ||
      kernel.getIterArgsInit().front() != taskZero)
    return rejectLlama(
        source,
        "LLaMA Task_0 Neura kernel inputs must identify output, read0, "
        "read1, seq, and the zero iter-arg initializer");
  unsigned zeroInputCount = llvm::count(kernel.getInputs(), taskZero);
  if (zeroInputCount > 1 ||
      (zeroInputCount == 1 && kernel.getInputs().front() != taskZero) ||
      (kernel.getInputs().size() != kernelRoles.size() &&
       zeroInputCount != 1))
    return rejectLlama(source,
                       "LLaMA Task_0 Neura kernel input ABI has an invalid "
                       "leading zero role");
  if (kernel.getInputs().size() != kernelRoles.size() + zeroInputCount)
    return rejectLlama(source,
                       "LLaMA Task_0 Neura kernel has extra input operands");
  if (kernel.getBody().empty() ||
      kernel.getBody().front().getNumArguments() !=
          kernel.getInputs().size() + kernel.getIterArgsInit().size())
    return rejectLlama(
        source,
        "LLaMA Task_0 Neura kernel body arguments do not match its input and "
        "iter-arg ABI");
  Block &kernelBody = kernel.getBody().front();
  Value kernelOutput = kernelBody.getArgument(roleInputIndices[0]);
  Value kernelRead0 = kernelBody.getArgument(roleInputIndices[1]);
  Value kernelRead1 = kernelBody.getArgument(roleInputIndices[2]);
  unsigned stores = 0;
  bool loadedRead0 = false;
  bool loadedRead1 = false;
  bool unsafeMemory = false;
  bool unsafeOutputCoordinates = false;
  auto checkOutputCoordinates = [&](ValueRange indices) {
    unsafeOutputCoordinates |=
        indices.size() != 2 ||
        !isReplicaOutputCoordinate(indices[0], neuraCounters[0]) ||
        !isReplicaOutputCoordinate(indices[1], neuraCounters[1]);
  };
  kernel.walk([&](Operation *operation) {
    if (operation == kernel.getOperation())
      return;
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      ++stores;
      unsafeMemory |= store.getMemRef() != kernelOutput;
      checkOutputCoordinates(store.getIndices());
      return;
    }
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      if (load.getMemRef() == kernelRead0)
        loadedRead0 = true;
      else if (load.getMemRef() == kernelRead1)
        loadedRead1 = true;
      else
        unsafeMemory = true;
      return;
    }
    if (auto store = dyn_cast<neura::StoreIndexedOp>(operation)) {
      ++stores;
      std::optional<unsigned> input =
          getFoldedKernelInput(operation, "rhs_value");
      if (!input && store.getBase())
        input = traceKernelInput(store.getBase(), kernel);
      unsafeMemory |= !input || *input != roleInputIndices[0];
      checkOutputCoordinates(store.getIndices());
      return;
    }
    if (auto load = dyn_cast<neura::LoadIndexedOp>(operation)) {
      std::optional<unsigned> input =
          getFoldedKernelInput(operation, "lhs_value");
      if (!input && load.getBase())
        input = traceKernelInput(load.getBase(), kernel);
      if (input && *input == roleInputIndices[1])
        loadedRead0 = true;
      else if (input && *input == roleInputIndices[2])
        loadedRead1 = true;
      else
        unsafeMemory = true;
      return;
    }
    if (isa<neura::DataMovOp>(operation)) {
      if (!isLlamaIdentityDataMove(operation, kernelBody))
        unsafeMemory = true;
      return;
    }
    if (isa<scf::IfOp, scf::YieldOp, neura::YieldOp>(operation) ||
        isLlamaInternalDFGOp(operation))
      return;
    if (!isMemoryEffectFree(operation)) {
      unsafeMemory = true;
    }
  });
  // Loads and stores above establish the intended dataflow.  Reject any
  // other operation with an unmodeled memory effect (copy, alloc, call, or
  // opaque dialect op); otherwise a hidden alias could invalidate the
  // completion-only join even when all visible loads/stores look safe.
  kernel.walk([&](Operation *operation) {
    if (operation == kernel.getOperation() ||
        isa<memref::LoadOp, memref::StoreOp, neura::LoadIndexedOp,
            neura::StoreIndexedOp, scf::IfOp, scf::YieldOp,
            neura::YieldOp>(operation) ||
        isLlamaInternalDFGOp(operation))
      return;
    if (isa<neura::DataMovOp>(operation)) {
      if (!isLlamaIdentityDataMove(operation, kernelBody))
        unsafeMemory = true;
      return;
    }
    if (!isMemoryEffectFree(operation)) {
      unsafeMemory = true;
    }
  });
  if (stores == 0 || !loadedRead0 || !loadedRead1 || unsafeMemory)
    return rejectLlama(
        source,
        (Twine("LLaMA ") + source.getTaskName() +
         " Neura DFG must store only the private output and load both "
         "read-only inputs")
            .str());
  if (unsafeOutputCoordinates)
    return rejectLlama(source,
        "replica output stores do not preserve independent M/N cell ownership");

  return consumer;
}

static LogicalResult materializeLlamaProducer(func::FuncOp function,
                                              TaskflowTaskOp source,
                                              int64_t axis, int64_t replicas) {
  FailureOr<TaskflowTaskOp> validated =
      validateLlamaProducer(function, source, axis, replicas);
  if (failed(validated))
    return failure();
  SmallVector<TaskflowTaskOp> consumers;
  Value sourceOutput = source.getWillWrites().front();
  if (!collectLlamaProducerConsumers(source, sourceOutput, consumers))
    return rejectLlama(source, "LLaMA producer consumers changed after validation");
  auto output = cast<MemRefType>(source.getWillWrites().front().getType());
  constexpr int64_t seqLen = 320;
  constexpr int64_t reduction = 256;
  const int64_t hidden = output.getShape()[1] == 512 ? 320 : 256;
  const std::string sourceName = source.getTaskName().str();

  OpBuilder builder(source);
  SmallVector<Value> states;
  for (int64_t replicaId = 0; replicaId < replicas; ++replicaId) {
    const int64_t shardExtent = axis == 0 ? seqLen : hidden;
    const int64_t otherExtent = axis == 0 ? hidden : seqLen;
    const auto [lower, upper] =
        balancedShardBounds(shardExtent, replicaId, replicas);
    auto replica = cast<TaskflowTaskOp>(builder.clone(*source.getOperation()));
    replica->setAttr("task_name",
                     builder.getStringAttr(sourceName + ".replica." +
                                           std::to_string(replicaId)));
    replica->setAttr("amoeba.replica.parent_task",
                     builder.getStringAttr(sourceName));
    replica->setAttr("amoeba.replica.id", builder.getI64IntegerAttr(replicaId));
    replica->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicas));
    replica->setAttr("amoeba.replica.shard_axis",
                     builder.getI64IntegerAttr(axis));
    replica->setAttr("amoeba.replica.shard_lower",
                     builder.getI64IntegerAttr(lower));
    replica->setAttr("amoeba.replica.shard_upper",
                     builder.getI64IntegerAttr(upper));
    replica->removeAttr(kSemanticIncomingEdgesAttr);
    replica->removeAttr("amoeba.tiling.parent_task");

    SmallVector<int64_t> regionLower{0, 0};
    SmallVector<int64_t> regionUpper{seqLen, hidden};
    regionLower[axis] = lower;
    regionUpper[axis] = upper;
    replica->setAttr(
        kTilingOutputRegionLowersAttr,
        builder.getArrayAttr({builder.getDenseI64ArrayAttr(regionLower)}));
    replica->setAttr(
        kTilingOutputRegionUppersAttr,
        builder.getArrayAttr({builder.getDenseI64ArrayAttr(regionUpper)}));

    Block &replicaBody = replica.getBody().front();
    SmallVector<TaskflowCounterOp> clonedCounters;
    neura::KernelOp clonedKernel;
    for (Operation &operation : replicaBody) {
      if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
        clonedCounters.push_back(counter);
      else if (auto candidate = dyn_cast<neura::KernelOp>(&operation))
        clonedKernel = candidate;
    }
    // Preserve the complete source counter scaffold. Completion regions are
    // derived from the leading output counters; the trailing K counter still
    // authenticates the full mapper firing count and source partition ledger.
    if (clonedCounters.size() != 3 || !clonedKernel ||
        !clonedCounters[2].getCounterIndex().use_empty())
      return rejectLlama(source,
                         "LLaMA Task_0 clone lost the proved dead K counter "
                         "or complete Neura kernel");

    TaskflowCounterOp selectedCounter = clonedCounters[axis];
    OpBuilder counterBuilder(selectedCounter);
    Value lowerValue = counterBuilder.create<arith::ConstantIndexOp>(
        source.getLoc(), lower);
    Value upperValue = counterBuilder.create<arith::ConstantIndexOp>(
        source.getLoc(), upper);
    selectedCounter.getLowerBoundMutable().assign(lowerValue);
    selectedCounter.getUpperBoundMutable().assign(upperValue);
    selectedCounter.setCounterDynamismAttr(
        builder.getStringAttr("constant_bound"));
    // The join verifier recovers each task state's complete rank-two region
    // from Taskflow counter operands.  The canonical M bound may still be a
    // value-input block argument even though validation proved it is the
    // literal 320.  When N is the selected axis, materialize that unchanged
    // M bound as a constant so the join can prove the full M extent without
    // weakening the runtime domain.
    if (axis == 1) {
      OpBuilder mBoundBuilder(clonedCounters[0]);
      Value fullMLower = mBoundBuilder.create<arith::ConstantIndexOp>(
          source.getLoc(), 0);
      Value fullMUpper = mBoundBuilder.create<arith::ConstantIndexOp>(
          source.getLoc(), seqLen);
      clonedCounters[0].getLowerBoundMutable().assign(fullMLower);
      clonedCounters[0].getUpperBoundMutable().assign(fullMUpper);
      clonedCounters[0].setCounterDynamismAttr(
          builder.getStringAttr("constant_bound"));
    } else {
      // Task_3 carries the full N extent as the same value input as M.  The
      // completion join needs a static region for the unchanged dimension;
      // validation has already proved this value is exactly logicalN, so
      // replace only the cloned scaffold operand with that constant.
      OpBuilder nBoundBuilder(clonedCounters[1]);
      Value fullNLower = nBoundBuilder.create<arith::ConstantIndexOp>(
          source.getLoc(), 0);
      Value fullNUpper = nBoundBuilder.create<arith::ConstantIndexOp>(
          source.getLoc(), hidden);
      clonedCounters[1].getLowerBoundMutable().assign(fullNLower);
      clonedCounters[1].getUpperBoundMutable().assign(fullNUpper);
      clonedCounters[1].setCounterDynamismAttr(
          builder.getStringAttr("constant_bound"));
    }

    SmallVector<neura::CounterOp> clonedNeuraCounters;
    clonedKernel.walk(
        [&](neura::CounterOp counter) { clonedNeuraCounters.push_back(counter); });
    for (neura::CounterOp counter : clonedNeuraCounters) {
      auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
      if (!id || id.getInt() != axis)
        continue;
      OpBuilder kernelBuilder =
          OpBuilder::atBlockBegin(&clonedKernel.getBody().front());
      Value kernelLower = kernelBuilder.create<arith::ConstantIndexOp>(
          source.getLoc(), lower);
      Value kernelUpper = kernelBuilder.create<arith::ConstantIndexOp>(
          source.getLoc(), upper);
      bool hasFoldedBounds =
          counter->hasAttr("lower_bound_value") ||
          counter->hasAttr("upper_bound_value") ||
          counter->hasAttr("step_value");
      counter.getLowerBoundMutable().assign(kernelLower);
      counter.getUpperBoundMutable().assign(kernelUpper);
      counter->setAttr("counter_dynamism",
                       builder.getStringAttr("constant_bound"));
      if (hasFoldedBounds) {
        counter->setAttr("lower_bound_value", builder.getIndexAttr(lower));
        counter->setAttr("upper_bound_value", builder.getIndexAttr(upper));
        counter->setAttr("step_value", builder.getIndexAttr(1));
      }
    }

    // The Taskflow and Neura K domains both retain the complete reduction.
    // Record all source work in the mapper firing contract for this shard.
    int64_t shardTripCount = (upper - lower) * otherExtent * reduction;
    replica->setAttr("trip_count", builder.getI64IntegerAttr(shardTripCount));
    replica->setAttr("amoeba.selected_trip_count",
                     builder.getI64IntegerAttr(shardTripCount));
    replica->setAttr("amoeba.replica.total_trip_count",
                     builder.getI64IntegerAttr(seqLen * hidden * reduction));
    replica->setAttr("amoeba.replica.shard_trip_count",
                     builder.getI64IntegerAttr(shardTripCount));
    states.push_back(replica.getDoneWrites().front());
  }

  auto join = builder.create<TaskflowJoinOp>(
      source.getLoc(), output, states, source.getWillWrites().front(),
      builder.getI64IntegerAttr(axis),
      builder.getDenseI64ArrayAttr({0, 0}),
      builder.getDenseI64ArrayAttr({seqLen, hidden}));
  join->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
  join->setAttr("amoeba.replica.completion_only", builder.getUnitAttr());
  source.getDoneWrites().front().replaceAllUsesWith(join.getJoined());
  for (TaskflowTaskOp consumer : consumers)
    if (failed(makeConsumerTensorWideForReplicas(
            consumer, join.getJoined(), sourceName, builder, replicas, true)))
      return failure();
  source.erase();
  function->setAttr("amoeba.replica.materialized_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.parent_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicas));
  function->setAttr("amoeba.replica.shard_axis",
                    builder.getI64IntegerAttr(axis));
  function->setAttr("amoeba.replica.total_trip_count",
                    builder.getI64IntegerAttr(seqLen * hidden * reduction));
  function->setAttr("amoeba.replica.shard_trip_count",
                    builder.getI64IntegerAttr(
                        ((axis == 0 ? seqLen : hidden) / replicas) *
                        (axis == 0 ? hidden : seqLen) * reduction));
  return success();
}

static LogicalResult materializeGemmProducer(func::FuncOp function,
                                             TaskflowTaskOp source,
                                             int64_t axis, int64_t replicas) {
  const std::string sourceName = source.getTaskName().str();
  FailureOr<TaskflowTaskOp> validated =
      validateGemmProducer(function, source, axis, replicas);
  if (failed(validated))
    return failure();
  TaskflowTaskOp consumer = *validated;
  auto output = cast<MemRefType>(source.getWillWrites().front().getType());
  Block &body = source.getBody().front();
  SmallVector<TaskflowCounterOp> counters;
  for (Operation &operation : body)
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      counters.push_back(counter);

  int64_t reductionTripCount = 0;
  source.walk([&](scf::ForOp loop) {
    if (reductionTripCount != 0)
      return;
    if (auto upper = constantIndex(loop.getUpperBound()))
      reductionTripCount = *upper;
  });
  if (reductionTripCount <= 0)
    return rejectGemm(source,
                      "GEMM producer needs a positive constant K trip count");
  int64_t totalTripCount = output.getShape()[0] * output.getShape()[1] *
                           reductionTripCount;
  if (auto trip = source->getAttrOfType<IntegerAttr>("trip_count"))
    totalTripCount = trip.getInt();
  if (totalTripCount <= 0 || totalTripCount % replicas != 0)
    return rejectGemm(source,
                      "GEMM producer trip_count must divide evenly by the "
                      "replica factor");
  const int64_t shardTripCount = totalTripCount / replicas;

  OpBuilder builder(source);
  SmallVector<Value> states;
  for (int64_t replicaId = 0; replicaId < replicas; ++replicaId) {
    auto replica = cast<TaskflowTaskOp>(builder.clone(*source.getOperation()));
    const auto [lower, upper] =
        balancedShardBounds(output.getShape()[axis], replicaId, replicas);
    replica->setAttr(
        "task_name",
        builder.getStringAttr(sourceName + ".replica." +
                              std::to_string(replicaId)));
    replica->setAttr("amoeba.replica.parent_task",
                     builder.getStringAttr(sourceName));
    replica->setAttr("amoeba.replica.id",
                     builder.getI64IntegerAttr(replicaId));
    replica->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicas));
    replica->setAttr("amoeba.replica.shard_axis",
                     builder.getI64IntegerAttr(axis));
    replica->setAttr("amoeba.replica.shard_lower",
                     builder.getI64IntegerAttr(lower));
    replica->setAttr("amoeba.replica.shard_upper",
                     builder.getI64IntegerAttr(upper));
    if (source->getAttr("trip_count"))
      replica->setAttr("trip_count",
                       builder.getI64IntegerAttr(shardTripCount));
    if (source->getAttr("amoeba.selected_trip_count"))
      replica->setAttr("amoeba.selected_trip_count",
                       builder.getI64IntegerAttr(shardTripCount));
    replica->setAttr("amoeba.replica.total_trip_count",
                     builder.getI64IntegerAttr(totalTripCount));
    replica->setAttr("amoeba.replica.shard_trip_count",
                     builder.getI64IntegerAttr(shardTripCount));
    replica->removeAttr(kSemanticIncomingEdgesAttr);
    replica->removeAttr("amoeba.tiling.parent_task");
    SmallVector<int64_t> regionLower(output.getRank(), 0);
    SmallVector<int64_t> regionUpper(output.getShape().begin(),
                                     output.getShape().end());
    regionLower[axis] = lower;
    regionUpper[axis] = upper;
    replica->setAttr(
        kTilingOutputRegionLowersAttr,
        builder.getArrayAttr({builder.getDenseI64ArrayAttr(regionLower)}));
    replica->setAttr(
        kTilingOutputRegionUppersAttr,
        builder.getArrayAttr({builder.getDenseI64ArrayAttr(regionUpper)}));
    SmallVector<TaskflowCounterOp> clonedCounters;
    for (Operation &operation : replica.getBody().front())
      if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
        clonedCounters.push_back(counter);
    OpBuilder counterBuilder(clonedCounters[axis]);
    Value lowerValue = counterBuilder.create<arith::ConstantIndexOp>(
        source.getLoc(), lower);
    Value upperValue = counterBuilder.create<arith::ConstantIndexOp>(
        source.getLoc(), upper);
    clonedCounters[axis].getLowerBoundMutable().assign(lowerValue);
    clonedCounters[axis].getUpperBoundMutable().assign(upperValue);
    states.push_back(replica.getDoneWrites().front());
  }

  auto join = builder.create<TaskflowJoinOp>(
      source.getLoc(), output, states, source.getWillWrites().front(),
      builder.getI64IntegerAttr(axis),
      builder.getDenseI64ArrayAttr(SmallVector<int64_t>(output.getRank(), 0)),
      builder.getDenseI64ArrayAttr(
          SmallVector<int64_t>(output.getShape().begin(), output.getShape().end())));
  join->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
  join->setAttr("amoeba.replica.completion_only", builder.getUnitAttr());
  source.getDoneWrites().front().replaceAllUsesWith(join.getJoined());
  if (failed(makeConsumerTensorWideForReplicas(consumer, join.getJoined(),
                                               sourceName, builder, replicas)))
    return failure();
  source.erase();
  function->setAttr("amoeba.replica.materialized_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.parent_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicas));
  function->setAttr("amoeba.replica.shard_axis",
                    builder.getI64IntegerAttr(axis));
  function->setAttr("amoeba.replica.total_trip_count",
                    builder.getI64IntegerAttr(totalTripCount));
  function->setAttr("amoeba.replica.shard_trip_count",
                    builder.getI64IntegerAttr(shardTripCount));
  return success();
}

// A small, workload-independent proof for an in-place Neura task.  The
// source graph currently contains one real non-LLaMA example (GCN Task_27):
// its task result is a dynamic view of a caller argument, and its kernel
// loads and stores that same root at one counter coordinate.  This path uses
// only IR contracts (the logical transfer shape, the Taskflow/Neura counter
// domains, and the indexed memory effects); it deliberately does not infer
// noalias from two unrelated SSA values or from a benchmark name.
struct ReplicaNeuraInPlacePlan {
  TaskflowTaskOp task;
  neura::KernelOp kernel;
  SmallVector<TaskflowCounterOp> taskCounters;
  SmallVector<neura::CounterOp> kernelCounters;
  SmallVector<int64_t> taskLowers;
  SmallVector<int64_t> taskUppers;
  SmallVector<int64_t> kernelLowers;
  SmallVector<int64_t> kernelUppers;
  SmallVector<int64_t> outputShape;
  SmallVector<unsigned> outputCounterAxes;
  SmallVector<std::optional<int64_t>> outputConstantAxes;
  SmallVector<int64_t> producedLowers;
  SmallVector<int64_t> producedShape;
  int64_t totalTripCount = 0;
  int64_t axis = -1;
  int64_t factor = 0;
  unsigned outputReadIndex = 0;
};

static LogicalResult rejectReplicaNeura(TaskflowTaskOp task,
                                        StringRef reason) {
  task.emitError() << reason;
  return failure();
}

static std::optional<SmallVector<int64_t>>
replicaCallerShape(Value value) {
  if (value.getDefiningOp<memref::AllocOp>() ||
      value.getDefiningOp<memref::AllocaOp>()) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (type && type.hasStaticShape() &&
        llvm::all_of(type.getShape(), [](int64_t extent) { return extent > 0; }))
      return SmallVector<int64_t>(type.getShape().begin(), type.getShape().end());
    return std::nullopt;
  }
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || !argument.getOwner())
    return std::nullopt;
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front())
    return std::nullopt;
  Attribute attribute = function.getArgAttr(
      argument.getArgNumber(), "amoeba.logical_transfer_shape");
  if (!attribute)
    return std::nullopt;
  SmallVector<int64_t> shape;
  if (auto dense = dyn_cast<DenseI64ArrayAttr>(attribute)) {
    shape.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
    return shape;
  }
  auto array = dyn_cast<ArrayAttr>(attribute);
  if (!array)
    return std::nullopt;
  for (Attribute element : array) {
    auto integer = dyn_cast<IntegerAttr>(element);
    if (!integer || !integer.getType().isInteger(64))
      return std::nullopt;
    shape.push_back(integer.getInt());
  }
  return shape;
}

static LogicalResult validateReplicaShape(Value value, MemRefType type,
                                          SmallVectorImpl<int64_t> &shape) {
  auto contract = replicaCallerShape(value);
  if (!contract || contract->size() != static_cast<size_t>(type.getRank()))
    return failure();
  for (auto [dimension, extent] : llvm::enumerate(*contract)) {
    if (extent <= 0 ||
        (!ShapedType::isDynamic(type.getDimSize(dimension)) &&
         type.getDimSize(dimension) != extent))
      return failure();
  }
  shape.append(contract->begin(), contract->end());
  return success();
}

static std::optional<unsigned>
replicaKernelInputIndex(neura::KernelOp kernel, TaskflowTaskOp task,
                        Value root) {
  if (!kernel.getBody().hasOneBlock() || !task.getBody().hasOneBlock())
    return std::nullopt;
  Block &taskBody = task.getBody().front();
  for (auto [index, input] : llvm::enumerate(kernel.getInputs())) {
    if (input == root)
      return static_cast<unsigned>(index);
    auto argument = dyn_cast<BlockArgument>(input);
    if (!argument || argument.getOwner() != &taskBody ||
        argument.getArgNumber() >= task->getNumOperands())
      continue;
    if (task->getOperand(argument.getArgNumber()) == root)
      return static_cast<unsigned>(index);
  }
  return std::nullopt;
}

static std::optional<unsigned>
replicaAccessInputIndex(Operation *operation, neura::KernelOp kernel,
                        StringRef foldedBaseAttribute, Value base) {
  if (auto input = parseKernelInputReference(
          operation->getAttr(foldedBaseAttribute)))
    return input;
  if (!base || !kernel.getBody().hasOneBlock())
    return std::nullopt;
  Block &body = kernel.getBody().front();
  for (auto [index, argument] : llvm::enumerate(body.getArguments()))
    if (base == argument && index < kernel.getInputs().size())
      return static_cast<unsigned>(index);
  return std::nullopt;
}

static std::optional<Value>
replicaKernelInputRoot(neura::KernelOp kernel, TaskflowTaskOp task,
                       unsigned inputIndex) {
  if (inputIndex >= kernel.getInputs().size() ||
      !task.getBody().hasOneBlock())
    return std::nullopt;
  Value input = kernel.getInputs()[inputIndex];
  for (Value root : task.getOriginalReadMemrefs())
    if (input == root)
      return root;
  for (Value root : task.getOriginalWriteMemrefs())
    if (input == root)
      return root;
  auto argument = dyn_cast<BlockArgument>(input);
  if (!argument || argument.getOwner() != &task.getBody().front() ||
      argument.getArgNumber() >= task->getNumOperands())
    return std::nullopt;
  unsigned operand = argument.getArgNumber();
  if (operand < task.getWillReads().size() &&
      operand < task.getOriginalReadMemrefs().size())
    return task.getOriginalReadMemrefs()[operand];
  operand -= task.getWillReads().size();
  if (operand < task.getWillWrites().size() &&
      operand < task.getOriginalWriteMemrefs().size())
    return task.getOriginalWriteMemrefs()[operand];
  return std::nullopt;
}

static std::optional<unsigned>
replicaCounterIndex(Value value, ArrayRef<neura::CounterOp> counters,
                    unsigned depth = 0) {
  if (!value || depth >= 16)
    return std::nullopt;
  for (auto [index, counterRef] : llvm::enumerate(counters)) {
    neura::CounterOp counter = counterRef;
    if (value == counter.getCurrentIndex())
      return static_cast<unsigned>(index);
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (name == "neura.data_mov" && definition->getNumOperands() == 1)
    return replicaCounterIndex(definition->getOperand(0), counters, depth + 1);
  if (name == "neura.grant_predicate" &&
      definition->getNumOperands() == 2)
    return replicaCounterIndex(definition->getOperand(0), counters, depth + 1);
  if (name != "neura.phi" || definition->getNumOperands() == 0)
    return std::nullopt;
  auto first = replicaCounterIndex(definition->getOperand(0), counters,
                                   depth + 1);
  if (!first)
    return std::nullopt;
  for (Value operand : definition->getOperands().drop_front()) {
    auto candidate = replicaCounterIndex(operand, counters, depth + 1);
    if (!candidate || *candidate != *first)
      return std::nullopt;
  }
  return first;
}

static bool replicaForwardsCounterIndex(Value value, Value expected,
                                        unsigned depth = 0) {
  if (value == expected)
    return true;
  if (!value || depth >= 8)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getNumOperands() != 1)
    return false;
  StringRef name = definition->getName().getStringRef();
  if (name != "neura.data_mov" && name != "neura.ctrl_mov")
    return false;
  return replicaForwardsCounterIndex(definition->getOperand(0), expected,
                                     depth + 1);
}

static bool replicaMentionsKernelInput(Operation *operation,
                                       unsigned inputIndex) {
  for (StringRef attribute : {StringRef("lhs_value"),
                              StringRef("rhs_value"), StringRef("value")})
    if (auto input = parseKernelInputReference(operation->getAttr(attribute));
        input && *input == inputIndex)
      return true;
  return false;
}

static bool replicaForwardsKernelArgument(Value value, BlockArgument argument,
                                          unsigned depth = 0) {
  if (value == argument)
    return true;
  if (!value || depth >= 8)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getNumOperands() != 1)
    return false;
  StringRef name = definition->getName().getStringRef();
  if (name != "neura.data_mov" && name != "neura.ctrl_mov")
    return false;
  return replicaForwardsKernelArgument(definition->getOperand(0), argument,
                                       depth + 1);
}

static bool replicaKnownPureNeuraOp(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  return name == "neura.constant" || name == "neura.counter" ||
         name == "neura.grant_once" || name == "neura.reserve" ||
         name == "neura.phi_start" || name == "neura.phi" ||
         name == "neura.icmp" || name == "neura.grant_predicate" ||
         name == "neura.mul" || name == "neura.div" || name == "neura.add" ||
         name == "neura.sub" || name == "neura.cast" || name == "neura.not" ||
         name == "neura.sel" ||
         name == "neura.ctrl_mov" || name == "neura.data_mov" ||
         name == "neura.extract_predicate" || name == "neura.return_value" ||
         name == "neura.yield";
}

// A few converted kernels fold a memory argument into a typed neura.constant
// instead of retaining the kernel block argument as the base SSA value.  That
// folded address is safe only when its ABI type and original storage root are
// authenticated, and when the value can reach an indexed memory access
// through the two explicit predicate/data forwarding operations.  In
// particular, an arithmetic/phi/opaque user is an address escape even if the
// operation happens to carry a Pure trait.
static bool verifyFoldedOutputAddressUsers(
    Value value, Type expectedType, unsigned outputInput, unsigned rank,
    llvm::SmallDenseSet<Value, 16> &visited, unsigned &indexedUses) {
  if (!value || value.getType() != expectedType ||
      !visited.insert(value).second)
    return false;

  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (auto grant = dyn_cast<neura::GrantPredicateOp>(owner)) {
      if (grant->getNumOperands() != 2 || grant->getOperand(0) != value ||
          grant->getNumResults() != 1 ||
          grant->getResult(0).getType() != expectedType ||
          !verifyFoldedOutputAddressUsers(grant->getResult(0), expectedType,
                                           outputInput, rank, visited,
                                           indexedUses))
        return false;
      continue;
    }
    if (isa<neura::DataMovOp>(owner)) {
      if (owner->getNumOperands() != 1 || owner->getOperand(0) != value ||
          owner->getNumResults() != 1 ||
          owner->getResult(0).getType() != expectedType ||
          !verifyFoldedOutputAddressUsers(owner->getResult(0), expectedType,
                                           outputInput, rank, visited,
                                           indexedUses))
        return false;
      continue;
    }
    if (auto load = dyn_cast<neura::LoadIndexedOp>(owner)) {
      if (load.getBase() != value || load.getIndices().size() != rank)
        return false;
      if (auto input = parseKernelInputReference(load->getAttr("lhs_value"));
          input && *input != outputInput)
        return false;
      ++indexedUses;
      continue;
    }
    if (auto store = dyn_cast<neura::StoreIndexedOp>(owner)) {
      if (store.getBase() != value || store.getIndices().size() != rank)
        return false;
      // The output-address folded constant is the base of this access.  The
      // historical StoreIndexed spelling may carry a lhs_value annotation for
      // the stored value, so only a present rhs_value annotation authenticates
      // the base and may constrain it.
      if (auto input = parseKernelInputReference(store->getAttr("rhs_value"));
          input && *input != outputInput)
        return false;
      ++indexedUses;
      continue;
    }
    // No unknown operation may receive an authenticated output address.
    return false;
  }
  return indexedUses != 0;
}

static bool authenticateFoldedOutputAddress(
    Operation *operation, ReplicaNeuraInPlacePlan &plan, unsigned outputInput,
    BlockArgument outputArgument, Value outputRoot, unsigned rank) {
  auto constant = dyn_cast<neura::ConstantOp>(operation);
  if (!constant || constant->getNumResults() != 1)
    return false;
  auto foldedInput = parseKernelInputReference(constant->getAttr("value"));
  if (!foldedInput || *foldedInput != outputInput ||
      outputInput >= plan.kernel.getInputs().size())
    return false;

  auto constantType = dyn_cast<neura::PredicatedValue>(
      constant->getResult(0).getType());
  auto argumentType = dyn_cast<neura::PredicatedValue>(outputArgument.getType());
  Type abiType = plan.kernel.getInputs()[outputInput].getType();
  auto rootType = dyn_cast<MemRefType>(outputRoot.getType());
  if (!constantType || !argumentType || !rootType ||
      constant->getResult(0).getType() != outputArgument.getType() ||
      constantType.getValueType() != argumentType.getValueType() ||
      constantType.getPredicateType() != argumentType.getPredicateType() ||
      constantType.getValueType() != abiType || abiType != outputRoot.getType())
    return false;

  auto authenticatedRoot =
      replicaKernelInputRoot(plan.kernel, plan.task, outputInput);
  if (plan.task.getWillWrites().size() != 1)
    return false;
  FailureOr<Value> outputStateRoot =
      resolveTaskflowMemoryRoot(plan.task.getWillWrites().front());
  FailureOr<Value> originalOutputRoot =
      resolveTaskflowMemoryRoot(outputRoot);
  if (!authenticatedRoot || *authenticatedRoot != outputRoot ||
      failed(outputStateRoot) || failed(originalOutputRoot) ||
      *outputStateRoot != *originalOutputRoot ||
      plan.task.getOriginalWriteMemrefs().size() != 1 ||
      plan.task.getOriginalWriteMemrefs().front() != outputRoot ||
      llvm::count(plan.task.getOriginalReadMemrefs(), outputRoot) != 1)
    return false;
  if (isTaskFunctionArgument(outputRoot) &&
      !isNoAliasTaskFunctionArgument(outputRoot))
    return false;
  for (Value root : plan.task.getOriginalReadMemrefs())
    if (root != outputRoot && !provesDistinctTaskStorage(root, outputRoot))
      return false;

  llvm::SmallDenseSet<Value, 16> visited;
  unsigned indexedUses = 0;
  return verifyFoldedOutputAddressUsers(
      constant->getResult(0), constant->getResult(0).getType(), outputInput,
      rank, visited, indexedUses);
}

// Accept only the counter-zero guard used to initialize independent source
// cells.  The conditional itself is not treated as effect-free: its regions
// remain in the kernel walk below, where every memory operation is checked
// against the authenticated root and coordinate map and every unknown effect
// remains rejected.
static bool
isVerifiedReplicaCounterZeroIf(scf::IfOp ifOp, ReplicaNeuraInPlacePlan &plan,
                               unsigned selectedCounter,
                               StringRef *rejectionReason = nullptr) {
  auto reject = [&](StringRef reason) {
    if (rejectionReason)
      *rejectionReason = reason;
    return false;
  };
  if (!ifOp)
    return reject("operation is not scf.if");
  if (ifOp.getNumResults() != 0)
    return reject("scf.if has results");
  if (!ifOp.getCondition().getType().isInteger(1))
    return reject("scf.if condition is not i1");
  if (selectedCounter >= plan.kernelCounters.size())
    return reject("selected output axis has no authenticated kernel counter");
  auto condition = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!condition)
    return reject("scf.if condition is not arith.cmpi");
  if (condition.getPredicate() != arith::CmpIPredicate::eq)
    return reject("scf.if comparison predicate is not equal");

  auto exactCounterThroughRepresentableCast =
      [&](auto &&self, Value value, unsigned depth) -> std::optional<unsigned> {
    if (depth > 8)
      return std::nullopt;
    if (auto direct = replicaCounterIndex(value, plan.kernelCounters))
      return direct;
    Operation *definition = value.getDefiningOp();
    if (!definition || definition->getNumOperands() != 1 ||
        definition->getNumResults() != 1 ||
        (!isa<arith::IndexCastOp, arith::IndexCastUIOp>(definition)))
      return std::nullopt;
    auto counter = self(self, definition->getOperand(0), depth + 1);
    if (!counter || *counter >= plan.kernelLowers.size() ||
        *counter >= plan.kernelUppers.size() ||
        plan.kernelLowers[*counter] < 0 ||
        plan.kernelUppers[*counter] <= plan.kernelLowers[*counter])
      return std::nullopt;
    auto destination =
        dyn_cast<IntegerType>(definition->getResult(0).getType());
    if (!destination)
      return definition->getResult(0).getType().isIndex() ? counter
                                                          : std::nullopt;
    unsigned width = destination.getWidth();
    if (width == 0 || width > 64)
      return std::nullopt;
    bool unsignedCast = isa<arith::IndexCastUIOp>(definition);
    uint64_t maxValue = 0;
    if (unsignedCast) {
      maxValue = width == 64 ? std::numeric_limits<uint64_t>::max()
                             : (uint64_t{1} << width) - 1;
    } else {
      maxValue =
          width == 64
              ? static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
              : (uint64_t{1} << (width - 1)) - 1;
    }
    if (static_cast<uint64_t>(plan.kernelUppers[*counter] - 1) > maxValue)
      return std::nullopt;
    return counter;
  };
  auto lhsCounter = exactCounterThroughRepresentableCast(
      exactCounterThroughRepresentableCast, condition.getLhs(), 0);
  auto rhsCounter = exactCounterThroughRepresentableCast(
      exactCounterThroughRepresentableCast, condition.getRhs(), 0);
  auto lhsConstant = mlir::amoeba::neura::joint_scheduling::detail::staticIndex(
      condition.getLhs(), plan.task, plan.kernel);
  auto rhsConstant = mlir::amoeba::neura::joint_scheduling::detail::staticIndex(
      condition.getRhs(), plan.task, plan.kernel);
  std::optional<unsigned> guardCounter;
  if (lhsCounter && rhsConstant && *rhsConstant == 0 && !rhsCounter)
    guardCounter = lhsCounter;
  else if (rhsCounter && lhsConstant && *lhsConstant == 0 && !lhsCounter)
    guardCounter = rhsCounter;
  if (!guardCounter)
    return reject("scf.if equality does not compare one representable kernel "
                  "counter with literal zero");
  if (*guardCounter == selectedCounter)
    return reject("zero-initialization guard uses the selected shard counter");
  if (*guardCounter >= plan.kernelLowers.size() ||
      *guardCounter >= plan.kernelUppers.size())
    return reject("zero-initialization guard counter has no proved bounds");
  if (plan.kernelLowers[*guardCounter] != 0)
    return reject("zero-initialization guard counter does not start at zero");
  if (plan.kernelUppers[*guardCounter] <= 0)
    return reject("zero-initialization guard counter has an empty range");

  auto validateYieldRegion = [](Region &region, bool allowEmpty, StringRef arm,
                                StringRef &reason) {
    if (region.empty()) {
      if (allowEmpty)
        return true;
      reason = "then initialization region is empty";
      return false;
    }
    if (!llvm::hasSingleElement(region)) {
      reason = arm == "then" ? "then initialization region has multiple blocks"
                             : "else region has multiple blocks";
      return false;
    }
    Block &block = region.front();
    if (block.getNumArguments() != 0) {
      reason = arm == "then" ? "then initialization region has block arguments"
                             : "else region has block arguments";
      return false;
    }
    auto yield = dyn_cast<scf::YieldOp>(block.getTerminator());
    if (!yield || yield.getNumOperands() != 0) {
      reason = arm == "then" ? "then initialization region lacks an empty yield"
                             : "else region lacks an empty yield";
      return false;
    }
    if (allowEmpty && block.getOperations().size() != 1) {
      reason = "else region is not empty";
      return false;
    }
    if (!allowEmpty && block.getOperations().size() <= 1) {
      reason = "then initialization region has no initialization operation";
      return false;
    }
    return true;
  };
  // The initialization arm may contain memory operations.  They are not
  // trusted by this structural check: the enclosing kernel walk below still
  // proves every nested load/store against the selected output root and exact
  // counter-coordinate map, and rejects every other effect.
  StringRef regionRejection;
  if (!validateYieldRegion(ifOp.getThenRegion(), /*allowEmpty=*/false, "then",
                           regionRejection))
    return reject(regionRejection);
  if (!validateYieldRegion(ifOp.getElseRegion(), /*allowEmpty=*/true, "else",
                           regionRejection))
    return reject(regionRejection);
  return true;
}

static LogicalResult
proveReplicaNeuraInPlaceRegion(ReplicaNeuraInPlacePlan &plan,
                               Value output) {
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!outputType || outputType.getRank() < 1 || outputType.getRank() > 3)
    return rejectReplicaNeura(
        plan.task,
        "in-place Neura replica requires a rank-1, rank-2, or rank-3 output");
  const unsigned rank = static_cast<unsigned>(outputType.getRank());
  if (plan.taskCounters.size() < rank || plan.kernelCounters.size() < rank)
    return rejectReplicaNeura(
        plan.task, "in-place Neura replica has too few counters for its output "
                   "dimensions");

  auto outputInput = replicaKernelInputIndex(plan.kernel, plan.task, output);
  if (!outputInput || *outputInput >= plan.kernel.getInputs().size())
    return rejectReplicaNeura(
        plan.task, "in-place Neura replica cannot resolve its output kernel "
                   "input");
  Block &taskBody = plan.task.getBody().front();
  unsigned outputInputCount = 0;
  for (auto [index, input] : llvm::enumerate(plan.kernel.getInputs())) {
    bool namesOutput = input == output;
    if (auto argument = dyn_cast<BlockArgument>(input))
      if (argument.getOwner() == &taskBody &&
          argument.getArgNumber() < plan.task->getNumOperands())
        namesOutput |= plan.task->getOperand(argument.getArgNumber()) == output;
    if (namesOutput) {
      ++outputInputCount;
      if (index != *outputInput)
        return rejectReplicaNeura(
            plan.task, "in-place Neura replica rejects duplicate output kernel "
                       "inputs");
    }
  }
  if (outputInputCount != 1 ||
      *outputInput >= plan.kernel.getBody().front().getNumArguments())
    return rejectReplicaNeura(
        plan.task, "in-place Neura replica requires one output kernel argument");

  Block &kernelBody = plan.kernel.getBody().front();
  BlockArgument outputArgument = kernelBody.getArgument(*outputInput);
  Value outputRoot = plan.task.getOriginalWriteMemrefs().front();
  auto checkIndices = [&](Operation *operation, ValueRange indices) {
    if (indices.size() != rank)
      return rejectReplicaNeura(
          plan.task, "in-place Neura replica requires complete output indices");
    SmallVector<unsigned> mapping;
    SmallVector<std::optional<int64_t>> constants;
    llvm::SmallDenseSet<unsigned, 4> used;
    for (auto [dimension, index] : llvm::enumerate(indices)) {
      auto counter = replicaCounterIndex(index, plan.kernelCounters);
      if (counter) {
        if (!used.insert(*counter).second)
          return rejectReplicaNeura(
              plan.task, "in-place Neura replica output indices are not the "
                         "restricted counter permutation");
        mapping.push_back(*counter);
        constants.push_back(std::nullopt);
        continue;
      }
      auto constant =
          mlir::amoeba::neura::joint_scheduling::detail::staticIndex(
              index, plan.task, plan.kernel);
      if (outputType.getRank() != 3 || !constant || *constant < 0 ||
          static_cast<size_t>(dimension) >= plan.outputShape.size() ||
          *constant >= plan.outputShape[dimension]) {
        plan.task.emitError()
            << "in-place Neura replica cannot resolve output index at "
            << dimension << " in " << operation->getName().getStringRef();
        return failure();
      }
      mapping.push_back(ReplicaOutputCoordinateProof::kConstantAxis);
      constants.push_back(*constant);
    }
    if (plan.outputCounterAxes.empty()) {
      plan.outputCounterAxes = mapping;
      plan.outputConstantAxes = constants;
    } else {
      bool coordinateMapMismatch =
          plan.outputCounterAxes.size() != mapping.size() ||
          plan.outputConstantAxes.size() != constants.size();
      if (!coordinateMapMismatch && outputType.getRank() == 3)
        coordinateMapMismatch =
            !llvm::equal(plan.outputCounterAxes, mapping) ||
            !llvm::equal(plan.outputConstantAxes, constants);
      if (!coordinateMapMismatch)
        coordinateMapMismatch =
            plan.axis < 0 ||
            plan.axis >= static_cast<int64_t>(mapping.size()) ||
            plan.outputCounterAxes[plan.axis] != mapping[plan.axis];
      if (coordinateMapMismatch)
        return rejectReplicaNeura(
            plan.task,
            "in-place Neura replica output accesses do not preserve the "
            "authenticated shard coordinate map");
    }
    if (plan.axis < 0 ||
        plan.axis >= static_cast<int64_t>(mapping.size()) ||
        mapping[plan.axis] == ReplicaOutputCoordinateProof::kConstantAxis)
      return rejectReplicaNeura(
          plan.task, "in-place Neura replica selected shard coordinate is not "
                     "an authenticated counter");
    return success();
  };
  auto validateAuxiliaryInput = [&](std::optional<unsigned> input) {
    if (!input)
      return rejectReplicaNeura(
          plan.task, "in-place Neura replica has an unproved auxiliary input");
    auto root = replicaKernelInputRoot(plan.kernel, plan.task, *input);
    if (!root || *root == outputRoot ||
        !llvm::is_contained(plan.task.getOriginalReadMemrefs(), *root) ||
        !provesDistinctTaskStorage(*root, outputRoot))
      return rejectReplicaNeura(
          plan.task, "in-place Neura replica auxiliary input may alias its "
                     "output root");
    return success();
  };

  bool sawLoad = false;
  bool sawStore = false;
  LogicalResult result = success();
  plan.kernel.walk([&](Operation *operation) {
    if (failed(result) || operation == plan.kernel.getOperation())
      return;
    if (isa<neura::ReserveOp, neura::CtrlMovOp, neura::PhiStartOp>(operation)) {
      // Coordinate-local loads/stores alone do not prove independence when
      // a carried scalar can pass a value from one cell to the next.
      result = rejectReplicaNeura(
          plan.task, "in-place Neura replica has unproved carried value feedback");
      return;
    }
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      auto input = replicaAccessInputIndex(operation, plan.kernel, "lhs_value",
                                           load.getMemRef());
      if (!input)
        input = traceKernelInput(load.getMemRef(), plan.kernel);
      if (!input)
        result = rejectReplicaNeura(
            plan.task, "in-place Neura replica has an unproved input load");
      else if (*input == *outputInput) {
        if (failed(checkIndices(operation, load.getIndices())))
          result = failure();
        else
          sawLoad = true;
      } else if (failed(validateAuxiliaryInput(input)))
        result = failure();
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      auto input = replicaAccessInputIndex(operation, plan.kernel, "rhs_value",
                                           store.getMemRef());
      if (!input)
        input = traceKernelInput(store.getMemRef(), plan.kernel);
      if (!input)
        result = rejectReplicaNeura(
            plan.task, "in-place Neura replica has an unproved output store");
      else if (*input == *outputInput) {
        if (failed(checkIndices(operation, store.getIndices())))
          result = failure();
        else
          sawStore = true;
      } else {
        result = rejectReplicaNeura(
            plan.task, "in-place Neura replica has an auxiliary memory store");
      }
      return;
    }
    if (auto load = dyn_cast<neura::LoadIndexedOp>(operation)) {
      auto input = replicaAccessInputIndex(operation, plan.kernel, "lhs_value",
                                           load.getBase());
      if (!input)
        input = traceKernelInput(load.getBase(), plan.kernel);
      if (!input)
        result = rejectReplicaNeura(
            plan.task, "in-place Neura replica has an unproved indexed load");
      else if (*input == *outputInput) {
        if (failed(checkIndices(operation, load.getIndices())))
          result = failure();
        else
          sawLoad = true;
      } else if (failed(validateAuxiliaryInput(input)))
        result = failure();
      return;
    }
    if (auto store = dyn_cast<neura::StoreIndexedOp>(operation)) {
      auto input = replicaAccessInputIndex(operation, plan.kernel, "rhs_value",
                                           store.getBase());
      if (!input)
        input = traceKernelInput(store.getBase(), plan.kernel);
      if (!input)
        result = rejectReplicaNeura(
            plan.task, "in-place Neura replica has an unproved indexed store");
      else if (*input == *outputInput) {
        if (failed(checkIndices(operation, store.getIndices())))
          result = failure();
        else
          sawStore = true;
      } else {
        result = rejectReplicaNeura(
            plan.task, "in-place Neura replica has an auxiliary indexed store");
      }
      return;
    }
    if (replicaMentionsKernelInput(operation, *outputInput)) {
      if (authenticateFoldedOutputAddress(operation, plan, *outputInput,
                                          outputArgument, outputRoot, rank))
        return;
      result = rejectReplicaNeura(
          plan.task, "in-place Neura replica rejects an unknown output memory "
                     "effect");
      return;
    }
    for (Value operand : operation->getOperands())
      if (replicaForwardsKernelArgument(operand, outputArgument)) {
        result = rejectReplicaNeura(
            plan.task, "in-place Neura replica rejects an unknown output alias");
        return;
      }
    auto conditional = dyn_cast<scf::IfOp>(operation);
    unsigned selectedCounter =
        plan.axis >= 0 &&
                plan.axis < static_cast<int64_t>(plan.outputCounterAxes.size())
            ? plan.outputCounterAxes[plan.axis]
            : std::numeric_limits<unsigned>::max();
    StringRef guardRejection;
    bool verifiedInitializationGuard = false;
    if (conditional) {
      if (selectedCounter == ReplicaOutputCoordinateProof::kConstantAxis) {
        guardRejection =
            "selected output axis has no authenticated kernel counter";
      } else {
        verifiedInitializationGuard = isVerifiedReplicaCounterZeroIf(
            conditional, plan, selectedCounter, &guardRejection);
      }
    }
    if (!replicaKnownPureNeuraOp(operation) && !verifiedInitializationGuard &&
        (!isPure(operation) || !isMemoryEffectFree(operation))) {
      std::string reason =
          "in-place Neura replica contains unclassified kernel operation " +
          operation->getName().getStringRef().str();
      if (conditional) {
        reason += "; zero-initialization guard rejected: ";
        reason += guardRejection.empty()
                      ? "guard did not satisfy the counter-zero proof"
                      : guardRejection.str();
      }
      result = rejectReplicaNeura(plan.task, reason);
    }
  });
  if (failed(result))
    return failure();
  if (!sawLoad || !sawStore)
    return rejectReplicaNeura(
        plan.task, "in-place Neura replica requires an indexed load/store pair");
  return success();
}

static LogicalResult
synchronizeReplicaKernelInputs(neura::KernelOp kernel) {
  if (!kernel.getBody().hasOneBlock())
    return rejectReplicaNeura(
        kernel->getParentOfType<TaskflowTaskOp>(),
        "replica kernel input synchronization requires one block");
  Block &body = kernel.getBody().front();
  auto inputs = kernel.getInputs();
  auto iterArgs = kernel.getIterArgsInit();
  if (body.getNumArguments() != inputs.size() + iterArgs.size())
    return failure();
  for (auto [index, input] : llvm::enumerate(inputs)) {
    BlockArgument argument = body.getArgument(index);
    Type expected = input.getType();
    if (argument.getType() == expected)
      continue;
    if (isa<neura::PredicatedValue>(expected))
      return failure();
    auto payload = dyn_cast<neura::PredicatedValue>(argument.getType());
    if (!payload) {
      Type original = argument.getType();
      if (!argument.use_empty()) {
        if (!isa<MemRefType>(expected) || !isa<MemRefType>(original) ||
            !memref::CastOp::areCastCompatible({expected}, {original}))
          return failure();
        SmallVector<OpOperand *> uses;
        for (OpOperand &use : argument.getUses())
          uses.push_back(&use);
        argument.setType(expected);
        OpBuilder castBuilder = OpBuilder::atBlockBegin(&body);
        Value oldView = castBuilder.create<memref::CastOp>(
            kernel.getLoc(), original, argument);
        for (OpOperand *use : uses)
          use->set(oldView);
      } else {
        argument.setType(expected);
      }
      continue;
    }
    if (!isa<MemRefType>(expected) && payload.getValueType() != expected)
      return failure();
    if (!argument.use_empty())
      return failure();
    argument.setType(neura::PredicatedValue::get(
        kernel.getContext(), expected, payload.getPredicateType()));
  }
  return success();
}

static LogicalResult updateReplicaKernelBounds(neura::KernelOp kernel,
                                               ArrayRef<int64_t> lowers,
                                               ArrayRef<int64_t> uppers) {
  SmallVector<neura::CounterOp> counters;
  for (Operation &operation : kernel.getBody().front())
    if (auto counter = dyn_cast<neura::CounterOp>(&operation))
      counters.push_back(counter);
  if (counters.size() != lowers.size() || counters.size() != uppers.size())
    return failure();
  for (auto [index, counter] : llvm::enumerate(counters)) {
    OpBuilder builder(counter);
    Value lower = builder.create<arith::ConstantIndexOp>(kernel.getLoc(),
                                                          lowers[index]);
    Value upper = builder.create<arith::ConstantIndexOp>(kernel.getLoc(),
                                                          uppers[index]);
    counter.getLowerBoundMutable().assign(lower);
    counter.getUpperBoundMutable().assign(upper);
    counter.getStepMutable().assign(
        builder.create<arith::ConstantIndexOp>(kernel.getLoc(), 1));
    if (counter->hasAttr("lower_bound_value"))
      counter->setAttr("lower_bound_value", builder.getIndexAttr(lowers[index]));
    if (counter->hasAttr("upper_bound_value"))
      counter->setAttr("upper_bound_value", builder.getIndexAttr(uppers[index]));
    if (counter->hasAttr("step_value"))
      counter->setAttr("step_value", builder.getIndexAttr(1));
    if (counter->hasAttr("counter_dynamism"))
      counter->setAttr("counter_dynamism",
                       builder.getStringAttr("constant_bound"));
  }
  return success();
}

static void copyReplicaTaskAttributes(TaskflowTaskOp source,
                                      TaskflowTaskOp target, StringRef name,
                                      OpBuilder &builder) {
  for (NamedAttribute attribute : source->getAttrs()) {
    StringRef key = attribute.getName().strref();
    if (key == "operandSegmentSizes" || key == "resultSegmentSizes" ||
        key == "task_name")
      continue;
    target->setAttr(attribute.getName(), attribute.getValue());
  }
  target->setAttr("task_name", builder.getStringAttr(name));
}

static FailureOr<TaskflowTaskOp>
createReplicaNeuraPart(OpBuilder &builder,
                       const ReplicaNeuraInPlacePlan &plan, int64_t part,
                       int64_t lower, int64_t upper, Value outputStorage) {
  TaskflowTaskOp source = plan.task;
  std::string name =
      (Twine(source.getTaskName()) + ".replica." + Twine(part)).str();
  SmallVector<Value> reads(source.getWillReads());
  SmallVector<Value> writes(source.getWillWrites());
  SmallVector<Value> originalWrites(source.getOriginalWriteMemrefs());
  if (!outputStorage || writes.size() != 1 || originalWrites.size() != 1)
    return failure();
  if (plan.outputReadIndex >= reads.size() ||
      source.getOriginalReadMemrefs()[plan.outputReadIndex] !=
          source.getOriginalWriteMemrefs().front())
    return failure();
  // Preserve the source read-state version when it is distinct from the write
  // state, even though both name the same storage root.  When they are the
  // exact same SSA state, use the static output view for both task inputs.
  if (reads[plan.outputReadIndex] == source.getWillWrites().front())
    reads[plan.outputReadIndex] = outputStorage;
  writes.front() = outputStorage;
  SmallVector<Type> writeTypes;
  writeTypes.push_back(outputStorage.getType());
  TaskflowTaskOp partTask = builder.create<TaskflowTaskOp>(
      source.getLoc(), source.getDoneReads().getTypes(), writeTypes,
      source.getValueOutputs().getTypes(), reads, writes,
      source.getValueInputs(), builder.getStringAttr(name),
      source.getOriginalReadMemrefs(), originalWrites);
  Block *body = new Block();
  partTask.getBody().push_back(body);
  IRMapping mapping;
  Block &sourceBody = source.getBody().front();
  for (auto [index, operand] : llvm::enumerate(source->getOperands())) {
    if (index >= sourceBody.getNumArguments())
      break;
    BlockArgument argument =
        body->addArgument(partTask->getOperand(index).getType(), source.getLoc());
    mapping.map(sourceBody.getArgument(index), argument);
  }
  Operation *terminator = sourceBody.getTerminator();
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  for (Operation &operation : sourceBody) {
    if (&operation == terminator)
      break;
    bodyBuilder.clone(operation, mapping);
  }
  SmallVector<TaskflowCounterOp> taskCounters;
  neura::KernelOp kernel;
  for (Operation &operation : *body) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      taskCounters.push_back(counter);
    else if (auto candidate = dyn_cast<neura::KernelOp>(&operation))
      kernel = candidate;
  }
  if (taskCounters.size() != plan.taskCounters.size() || !kernel)
    return failure();
  if (failed(synchronizeReplicaKernelInputs(kernel)))
    return failure();
  SmallVector<int64_t> taskLowers(plan.taskLowers);
  SmallVector<int64_t> taskUppers(plan.taskUppers);
  if (plan.axis < 0 ||
      plan.axis >= static_cast<int64_t>(plan.outputCounterAxes.size()))
    return failure();
  unsigned counterAxis = plan.outputCounterAxes[plan.axis];
  if (counterAxis >= taskLowers.size())
    return failure();
  taskLowers[counterAxis] = lower;
  taskUppers[counterAxis] = upper;
  for (auto [index, counter] : llvm::enumerate(taskCounters)) {
    OpBuilder counterBuilder(counter);
    counter.getLowerBoundMutable().assign(
        counterBuilder.create<arith::ConstantIndexOp>(source.getLoc(),
                                                       taskLowers[index]));
    counter.getUpperBoundMutable().assign(
        counterBuilder.create<arith::ConstantIndexOp>(source.getLoc(),
                                                       taskUppers[index]));
    counter.getStepMutable().assign(
        counterBuilder.create<arith::ConstantIndexOp>(source.getLoc(), 1));
    counter->setAttr("counter_dynamism",
                     counterBuilder.getStringAttr("constant_bound"));
  }
  SmallVector<int64_t> kernelLowers(plan.kernelLowers);
  SmallVector<int64_t> kernelUppers(plan.kernelUppers);
  if (counterAxis >= kernelLowers.size())
    return failure();
  kernelLowers[counterAxis] = lower;
  kernelUppers[counterAxis] = upper;
  if (failed(updateReplicaKernelBounds(kernel, kernelLowers, kernelUppers)))
    return failure();
  if (!terminator)
    return failure();
  bodyBuilder.setInsertionPointToEnd(body);
  bodyBuilder.clone(*terminator, mapping);
  copyReplicaTaskAttributes(source, partTask, name, builder);
  partTask->setAttr("amoeba.replica.parent_task",
                    builder.getStringAttr(source.getTaskName()));
  partTask->setAttr("amoeba.replica.id", builder.getI64IntegerAttr(part));
  partTask->setAttr("amoeba.replica.count",
                    builder.getI64IntegerAttr(plan.factor));
  partTask->setAttr("amoeba.replica.shard_axis",
                    builder.getI64IntegerAttr(counterAxis));
  partTask->setAttr("amoeba.replica.output_shard_axis",
                    builder.getI64IntegerAttr(plan.axis));
  SmallVector<int64_t> outputCounterAxes;
  outputCounterAxes.reserve(plan.outputCounterAxes.size());
  for (unsigned counter : plan.outputCounterAxes)
    outputCounterAxes.push_back(static_cast<int64_t>(counter));
  // This declaration is only an authenticated copy of the mapping proved
  // from the source Neura indexed accesses above.  The join verifier and
  // facts extractor independently re-derive the mapping before accepting it.
  partTask->setAttr("amoeba.replica.output_counter_axes",
                    builder.getDenseI64ArrayAttr(outputCounterAxes));
  SmallVector<int64_t> outputConstantAxes;
  outputConstantAxes.reserve(plan.outputConstantAxes.size());
  for (std::optional<int64_t> constant : plan.outputConstantAxes)
    outputConstantAxes.push_back(
        constant ? *constant : std::numeric_limits<int64_t>::min());
  if (llvm::any_of(plan.outputConstantAxes,
                   [](std::optional<int64_t> value) {
                     return value.has_value();
                   }))
    partTask->setAttr("amoeba.replica.output_constant_axes",
                      builder.getDenseI64ArrayAttr(outputConstantAxes));
  partTask->setAttr("amoeba.replica.shard_lower",
                    builder.getI64IntegerAttr(lower));
  partTask->setAttr("amoeba.replica.shard_upper",
                    builder.getI64IntegerAttr(upper));
  partTask->setAttr("trip_count",
                    builder.getI64IntegerAttr(plan.totalTripCount / plan.factor));
  partTask->setAttr("amoeba.replica.total_trip_count",
                    builder.getI64IntegerAttr(plan.totalTripCount));
  partTask->setAttr("amoeba.replica.shard_trip_count",
                    builder.getI64IntegerAttr(plan.totalTripCount / plan.factor));
  partTask->removeAttr(kSemanticIncomingEdgesAttr);
  partTask->removeAttr("amoeba.tiling.parent_task");
  SmallVector<int64_t> regionLower(plan.producedLowers.begin(),
                                   plan.producedLowers.end());
  if (plan.producedShape.size() != plan.outputShape.size())
    return failure();
  if (regionLower.size() != plan.outputShape.size())
    return failure();
  SmallVector<int64_t> regionUpper(regionLower);
  for (auto [dimension, extent] : llvm::enumerate(plan.producedShape)) {
    if (extent <= 0 || regionLower[dimension] >
                           std::numeric_limits<int64_t>::max() - extent)
      return failure();
    regionUpper[dimension] = regionLower[dimension] + extent;
  }
  regionLower[plan.axis] = lower;
  regionUpper[plan.axis] = upper;
  // The authenticated output-coordinate proof covers every load of the
  // recurrence state as well as its stores.  That read stays inside this
  // shard.  Describe every input: auxiliary inputs retain unknown regions
  // and their tensor-wide transfer contract.  This lets the existing memory
  // ordering analysis prove sibling recurrence states disjoint without
  // dropping any unknown auxiliary dependency.
  SmallVector<Attribute> inputLowers, inputUppers, inputReasons;
  Value outputRoot = source.getOriginalWriteMemrefs().front();
  for (Value root : source.getOriginalReadMemrefs()) {
    if (root == outputRoot) {
      inputLowers.push_back(builder.getDenseI64ArrayAttr(regionLower));
      inputUppers.push_back(builder.getDenseI64ArrayAttr(regionUpper));
      inputReasons.push_back(builder.getStringAttr(
          "replica-authenticated-output-recurrence-shard"));
    } else {
      inputLowers.push_back(builder.getUnitAttr());
      inputUppers.push_back(builder.getUnitAttr());
      inputReasons.push_back(builder.getStringAttr(
          "replica-auxiliary-input-region-unknown"));
    }
  }
  partTask->setAttr(kTilingInputRegionLowersAttr,
                    builder.getArrayAttr(inputLowers));
  partTask->setAttr(kTilingInputRegionUppersAttr,
                    builder.getArrayAttr(inputUppers));
  partTask->setAttr(kTilingInputRegionReasonsAttr,
                    builder.getArrayAttr(inputReasons));
  partTask->setAttr(
      kTilingOutputRegionLowersAttr,
      builder.getArrayAttr({builder.getDenseI64ArrayAttr(regionLower)}));
  partTask->setAttr(
      kTilingOutputRegionUppersAttr,
      builder.getArrayAttr({builder.getDenseI64ArrayAttr(regionUpper)}));
  return partTask;
}

static FailureOr<unsigned>
getReadInputIndexForResult(TaskflowTaskOp task, unsigned resultIndex) {
  if (!task || !task.getBody().hasOneBlock() ||
      resultIndex >= task.getDoneReads().size())
    return failure();
  auto yield = dyn_cast<TaskflowYieldOp>(task.getBody().front().getTerminator());
  if (!yield || resultIndex >= yield.getDoneReads().size())
    return failure();
  auto argument = dyn_cast<BlockArgument>(yield.getDoneReads()[resultIndex]);
  if (!argument || argument.getOwner() != &task.getBody().front() ||
      argument.getArgNumber() >= task.getWillReads().size() ||
      task.getOriginalReadMemrefs().size() != task.getWillReads().size() ||
      task.getWillReads()[argument.getArgNumber()].getType() !=
          task.getDoneReads()[resultIndex].getType())
    return failure();
  return argument.getArgNumber();
}

static FailureOr<ReplicaNeuraInPlacePlan>
analyzeReplicaNeuraInPlace(TaskflowTaskOp task, int64_t axis, int64_t factor) {
  if (!task || !task.getBody().hasOneBlock() ||
      task.getWillReads().empty() || task.getWillWrites().size() != 1 ||
      task.getOriginalReadMemrefs().size() != task.getWillReads().size() ||
      task.getOriginalWriteMemrefs().size() != 1 ||
      task.getDoneWrites().size() != 1 || task.getDoneReads().size() > 1 ||
      !task.getValueOutputs().empty())
    return rejectReplicaNeura(
        task, "in-place Neura replica requires one output and at most one "
              "sparse read-completion result");
  if (!task.getDoneReads().empty() &&
      failed(getReadInputIndexForResult(task, 0)))
    return rejectReplicaNeura(
        task, "in-place Neura read-completion result must preserve one exact "
              "will_reads state");
  Value outputRoot = task.getOriginalWriteMemrefs().front();
  if (llvm::count(task.getOriginalReadMemrefs(), outputRoot) != 1)
    return rejectReplicaNeura(
        task, "in-place Neura replica requires one exact read/write storage root");
  FailureOr<Value> outputRootIdentity = resolveTaskflowMemoryRoot(outputRoot);
  bool foundOutputRead = false;
  unsigned outputReadIndex = 0;
  for (auto [index, root] : llvm::enumerate(task.getOriginalReadMemrefs())) {
    if (root == outputRoot) {
      FailureOr<Value> readRoot =
          resolveTaskflowMemoryRoot(task.getWillReads()[index]);
      if (failed(readRoot) || failed(outputRootIdentity) ||
          *readRoot != *outputRootIdentity || foundOutputRead)
        return rejectReplicaNeura(
            task, "in-place Neura replica read state does not resolve to its "
                  "matched output storage root");
      outputReadIndex = static_cast<unsigned>(index);
      foundOutputRead = true;
      continue;
    }
    if (!provesDistinctTaskStorage(root, outputRoot))
      return rejectReplicaNeura(
          task, "in-place Neura replica auxiliary read may alias its output "
                "root");
  }
  if (!foundOutputRead)
    return rejectReplicaNeura(
        task, "in-place Neura replica has no matched input state for its "
              "output storage root");
  if (axis < 0 || factor < 2 || !supportsTaskReplicaCount(task, factor))
    return rejectReplicaNeura(
        task, "in-place Neura replica requires a supported factor and axis");

  // Authenticate the original source before any replica planning can lead to
  // a clone.  The shared proof rejects spoofed folded input annotations,
  // unsupported integer/index casts, malformed memory bases, and static-shape
  // disagreements.  The plan below must agree with this source-derived map
  // and produced shape; it may not reconstruct a weaker proof from a mutated
  // clone operand.
  ReplicaOutputCoordinateProof sourceOutputProof =
      analyzeReplicaOutputCoordinates(task, axis);
  if (!sourceOutputProof.proven) {
    task.emitError() << "in-place Neura replica source output-coordinate proof "
                        "failed: "
                     << sourceOutputProof.reason;
    return failure();
  }

  ReplicaNeuraInPlacePlan plan;
  plan.task = task;
  plan.axis = axis;
  plan.factor = factor;
  plan.outputReadIndex = outputReadIndex;
  for (Operation &operation : task.getBody().front()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      plan.taskCounters.push_back(counter);
      continue;
    }
    if (auto kernel = dyn_cast<neura::KernelOp>(&operation)) {
      if (plan.kernel)
        return rejectReplicaNeura(task,
                                  "in-place Neura replica requires one kernel");
      plan.kernel = kernel;
      continue;
    }
    if (isa<affine::AffineApplyOp, arith::AddIOp, arith::SubIOp>(&operation)) {
      if (operation.getNumResults() == 1 &&
          operation.getResult(0).getType().isIndex() &&
          compileTimeIndex(operation.getResult(0), task))
        continue;
      return rejectReplicaNeura(
          task, "in-place Neura replica has an unresolved task-level index "
                "expression");
    }
    if (!isa<arith::ConstantOp, arith::ConstantIndexOp, TaskflowYieldOp>(
            &operation))
      return rejectReplicaNeura(
          task, "in-place Neura replica body has an unsupported top-level op");
  }
  if (!plan.kernel)
    return rejectReplicaNeura(task, "in-place Neura replica requires one kernel");
  auto outputType = dyn_cast<MemRefType>(task.getWillWrites().front().getType());
  Value root = task.getOriginalWriteMemrefs().front();
  auto rootType = dyn_cast<MemRefType>(root.getType());
  if (!outputType || !rootType || outputType.getRank() != rootType.getRank() ||
      outputType.getElementType() != rootType.getElementType())
    return rejectReplicaNeura(task,
                              "in-place Neura replica has incompatible output "
                              "and storage types");
  if (failed(validateReplicaShape(root, rootType, plan.outputShape)))
    return rejectReplicaNeura(
        task, "in-place Neura replica requires the exact logical transfer "
              "shape contract on its caller storage root");
  if (plan.outputShape.size() != static_cast<size_t>(outputType.getRank()) ||
      plan.outputShape.size() < 1 || plan.outputShape.size() > 3 ||
      axis >= static_cast<int64_t>(plan.outputShape.size()))
    return rejectReplicaNeura(
        task, "in-place Neura replica selected axis is outside its output "
              "shape");
  if (!plan.kernel.getIterArgsInit().empty() || plan.kernel.getNumResults())
    return rejectReplicaNeura(task, "in-place Neura replica has unproved carried kernel state");
  if (plan.taskCounters.size() < plan.outputShape.size() ||
      !plan.kernel.getBody().hasOneBlock() ||
      plan.kernel.getBody().front().getNumArguments() !=
          plan.kernel.getInputs().size() + plan.kernel.getIterArgsInit().size())
    return rejectReplicaNeura(
        task, "in-place Neura replica counter or kernel ABI is incomplete");
  for (auto [index, counter] : llvm::enumerate(plan.taskCounters)) {
    auto lower = compileTimeIndex(counter.getLowerBound(), task);
    auto upper = compileTimeIndex(counter.getUpperBound(), task);
    auto step = compileTimeIndex(counter.getStep(), task);
    bool parentMismatch =
        index == 0
            ? static_cast<bool>(counter.getParentIndex())
            : counter.getParentIndex() !=
                  plan.taskCounters[index - 1].getCounterIndex();
    if (!lower || !upper || !step || *lower < 0 || *upper <= *lower ||
        *step != 1 || parentMismatch)
      return rejectReplicaNeura(
          task, "in-place Neura replica Taskflow counter domains are not "
                "nonnegative static unit-step ranges");
    plan.taskLowers.push_back(*lower);
    plan.taskUppers.push_back(*upper);
  }
  for (Operation &operation : plan.kernel.getBody().front())
    if (auto counter = dyn_cast<neura::CounterOp>(&operation))
      plan.kernelCounters.push_back(counter);
  if (plan.kernelCounters.size() != plan.taskCounters.size())
    return rejectReplicaNeura(
        task, "in-place Neura replica Taskflow and Neura counters differ");
  // Radar's specialized producer path retains the historical
  // operand-then-attribute lookup.  Every generic post-Neura in-place proof,
  // including LU Task_6, must authenticate both representations when they
  // are present and reject a conflict instead of accepting the folded value.
  func::FuncOp parentFunction = task->getParentOfType<func::FuncOp>();
  bool preserveLegacyCounterLookup = isRadarFunction(parentFunction);
  auto resolveReplicaCounterBound = [&](neura::CounterOp counter,
                                        StringRef attributeName,
                                        Value operand) {
    if (preserveLegacyCounterLookup)
      return compileTimeCounterBound(counter, attributeName, operand, task,
                                     plan.kernel);
    return compileTimeCounterBoundWithAgreement(
        counter, attributeName, operand, task, plan.kernel);
  };
  for (auto [index, counter] : llvm::enumerate(plan.kernelCounters)) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    auto lower = resolveReplicaCounterBound(counter, "lower_bound_value",
                                            counter.getLowerBound());
    auto upper = resolveReplicaCounterBound(counter, "upper_bound_value",
                                            counter.getUpperBound());
    auto step = resolveReplicaCounterBound(counter, "step_value",
                                           counter.getStep());
    if (!id || id.getInt() != static_cast<int64_t>(index) || !lower ||
        !upper || !step || *lower != plan.taskLowers[index] ||
        *upper != plan.taskUppers[index] || *step != 1)
      return rejectReplicaNeura(
          task, "in-place Neura replica Neura counter bounds are not the "
              "Taskflow domains");
    plan.kernelLowers.push_back(*lower);
    plan.kernelUppers.push_back(*upper);
  }
  plan.outputCounterAxes = sourceOutputProof.outputCounterAxes;
  plan.outputConstantAxes = sourceOutputProof.outputConstantAxes;
  if (failed(proveReplicaNeuraInPlaceRegion(
          plan, task.getWillWrites().front())))
    return failure();
  if (plan.outputCounterAxes.size() != plan.outputShape.size() ||
      plan.outputConstantAxes.size() != plan.outputShape.size())
    return rejectReplicaNeura(
        task, "in-place Neura replica did not prove every output dimension "
              "as a counter or bounded constant");
  llvm::SmallDenseSet<unsigned, 4> mappedCounters;
  plan.producedLowers.resize(plan.outputShape.size());
  plan.producedShape.resize(plan.outputShape.size());
  for (auto [dimension, counter] : llvm::enumerate(plan.outputCounterAxes)) {
    if (counter == ReplicaOutputCoordinateProof::kConstantAxis) {
      if (!plan.outputConstantAxes[dimension] ||
          *plan.outputConstantAxes[dimension] < 0 ||
          *plan.outputConstantAxes[dimension] >= plan.outputShape[dimension])
        return rejectReplicaNeura(
            task, "in-place Neura replica constant output dimension is out "
                  "of bounds");
      plan.producedLowers[dimension] = *plan.outputConstantAxes[dimension];
      plan.producedShape[dimension] = 1;
      continue;
    }
    if (counter >= plan.taskUppers.size() ||
        !mappedCounters.insert(counter).second)
      return rejectReplicaNeura(
          task, "in-place Neura replica output counter permutation is not "
                "one-to-one");
    int64_t extent = plan.taskUppers[counter] - plan.taskLowers[counter];
    if (extent <= 0 || extent > plan.outputShape[dimension])
      return rejectReplicaNeura(
          task, "in-place Neura replica produced region exceeds its caller "
                "shape");
    plan.producedLowers[dimension] = plan.taskLowers[counter];
    plan.producedShape[dimension] = extent;
  }
  if (!llvm::equal(sourceOutputProof.outputCounterAxes,
                   plan.outputCounterAxes) ||
      !llvm::equal(sourceOutputProof.outputConstantAxes,
                   plan.outputConstantAxes) ||
      !llvm::equal(sourceOutputProof.producedLowers, plan.producedLowers) ||
      !llvm::equal(sourceOutputProof.producedShape, plan.producedShape))
    return rejectReplicaNeura(
        task, "in-place Neura replica plan disagrees with the source "
              "output-coordinate proof");
  if (plan.outputCounterAxes[axis] ==
          ReplicaOutputCoordinateProof::kConstantAxis ||
      !hasBalancedShards(plan.producedShape[axis], factor))
    return rejectReplicaNeura(
        task, "in-place Neura replica produced region is not divisible on "
              "its selected axis");
  plan.totalTripCount = 1;
  for (auto [index, lower] : llvm::enumerate(plan.taskLowers)) {
    int64_t extent = plan.taskUppers[index] - lower;
    if (extent <= 0 ||
        extent > std::numeric_limits<int64_t>::max() / plan.totalTripCount)
      return rejectReplicaNeura(task,
                                "in-place Neura replica trip count overflows");
    plan.totalTripCount *= extent;
  }
  return plan;
}

static LogicalResult
materializeReplicaNeuraInPlace(func::FuncOp function,
                               ReplicaNeuraInPlacePlan &plan) {
  TaskflowTaskOp source = plan.task;
  SmallVector<ActiveTransferRewriteSnapshot> activeTransferFacts;
  SmallVector<std::pair<std::string, Attribute>> callerFacts;
  if (failed(captureActiveTransferRewriteFacts(function, activeTransferFacts,
                                               callerFacts)))
    return failure();
  std::string sourceName = source.getTaskName().str();
  // Capture every downstream Taskflow consumer before replacing the source
  // done-write.  The completion join is a typed dependency, so its incoming
  // RAW/WAW annotations must be rebuilt for each generated replica rather
  // than left pointing at the erased parent task.
  SmallVector<TaskflowTaskOp> consumers;
  SmallVector<TaskflowTaskOp> readConsumers;
  Value sourceDone = source.getDoneWrites().front();
  for (OpOperand &use : sourceDone.getUses()) {
    auto consumer = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (!consumer || !llvm::is_contained(consumer.getWillReads(), sourceDone) ||
        llvm::count(consumer.getWillReads(), sourceDone) != 1)
      return rejectReplicaNeura(
          source, "in-place Neura replica has an unclassified done-write "
                  "consumer");
    consumers.push_back(consumer);
  }
  if (!source.getDoneReads().empty()) {
    if (source.getDoneReads().size() != 1 ||
        failed(getReadInputIndexForResult(source, 0)))
      return rejectReplicaNeura(
          source, "in-place Neura replica supports one passthrough done-read "
                  "result");
    Value sourceReadDone = source.getDoneReads().front();
    for (OpOperand &use : sourceReadDone.getUses()) {
      auto consumer = dyn_cast<TaskflowTaskOp>(use.getOwner());
      if (!consumer ||
          !llvm::is_contained(consumer.getWillWrites(), sourceReadDone) ||
          llvm::count(consumer.getWillWrites(), sourceReadDone) != 1)
        return rejectReplicaNeura(
            source, "in-place Neura done-read must feed a unique downstream "
                    "writer operand");
      if (!llvm::is_contained(readConsumers, consumer))
        readConsumers.push_back(consumer);
    }
  }
  for (int64_t part = 0; part < plan.factor; ++part) {
    std::string name = (Twine(sourceName) + ".replica." + Twine(part)).str();
    bool collision = false;
    function.walk([&](TaskflowTaskOp task) {
      collision |= task != source && task.getTaskName() == name;
    });
    if (collision)
      return rejectReplicaNeura(
          source, "in-place Neura replica name collides with an existing task");
  }
  auto originalType =
      dyn_cast<MemRefType>(source.getWillWrites().front().getType());
  if (!originalType)
    return rejectReplicaNeura(source,
                              "in-place Neura replica output is not a memref");
  auto staticType =
      MemRefType::get(plan.outputShape, originalType.getElementType(),
                      originalType.getLayout(), originalType.getMemorySpace());
  OpBuilder builder(source);
  Value outputStorage = builder.create<memref::CastOp>(
      source.getLoc(), staticType, source.getWillWrites().front());
  SmallVector<Value> states;
  if (plan.axis < 0 ||
      plan.axis >= static_cast<int64_t>(plan.outputCounterAxes.size()))
    return rejectReplicaNeura(
        source, "in-place Neura replica selected axis has no counter map");
  unsigned counterAxis = plan.outputCounterAxes[plan.axis];
  if (counterAxis >= plan.taskUppers.size())
    return rejectReplicaNeura(
        source, "in-place Neura replica selected counter is out of range");
  int64_t extent = plan.taskUppers[counterAxis] - plan.taskLowers[counterAxis];
  int64_t base = extent / plan.factor;
  int64_t remainder = extent % plan.factor;
  int64_t cursor = plan.taskLowers[counterAxis];
  SmallVector<TaskflowTaskOp> replicaTasks;
  for (int64_t part = 0; part < plan.factor; ++part) {
    int64_t width = base + (part < remainder ? 1 : 0);
    FailureOr<TaskflowTaskOp> replica =
        createReplicaNeuraPart(builder, plan, part, cursor, cursor + width,
                               source.getWillWrites().front());
    if (failed(replica))
      return failure();
    replicaTasks.push_back(*replica);
    // Preserve the parent's mapper-visible task and kernel operand types.
    // The static view exists only outside the kernel for completion-region
    // verification; shard counters are the only body specialization.
    states.push_back(builder.create<memref::CastOp>(
        source.getLoc(), staticType, replica->getDoneWrites().front()));
    cursor += width;
  }
  SmallVector<int64_t> regionLower(plan.producedLowers.begin(),
                                   plan.producedLowers.end());
  if (regionLower.size() != plan.outputShape.size() ||
      plan.producedShape.size() != plan.outputShape.size())
    return failure();
  SmallVector<int64_t> regionUpper(regionLower);
  for (auto [dimension, extent] : llvm::enumerate(plan.producedShape)) {
    if (extent <= 0 ||
        regionLower[dimension] > std::numeric_limits<int64_t>::max() - extent)
      return failure();
    regionUpper[dimension] = regionLower[dimension] + extent;
  }
  auto join = builder.create<TaskflowJoinOp>(
      source.getLoc(), staticType, states, outputStorage,
      builder.getI64IntegerAttr(plan.axis),
      builder.getDenseI64ArrayAttr(regionLower),
      builder.getDenseI64ArrayAttr(regionUpper));
  join->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
  join->setAttr("amoeba.replica.completion_only", builder.getUnitAttr());
  Value readReplacement;
  if (source.getDoneReads().size() == 1) {
    FailureOr<unsigned> readInputIndex = getReadInputIndexForResult(source, 0);
    if (failed(readInputIndex) || replicaTasks.size() != states.size())
      return rejectReplicaNeura(
          source, "in-place Neura replica lost its exact done-read input slot");
    SmallVector<Value> readStates;
    readStates.reserve(replicaTasks.size());
    Value baseReadState = replicaTasks.front().getWillReads()[*readInputIndex];
    Value originalReadRoot = source.getOriginalReadMemrefs()[*readInputIndex];
    for (TaskflowTaskOp replica : replicaTasks) {
      if (replica.getWillReads()[*readInputIndex] != baseReadState ||
          replica.getOriginalReadMemrefs()[*readInputIndex] !=
              originalReadRoot ||
          replica.getDoneReads().size() != 1)
        return rejectReplicaNeura(
            source, "in-place Neura replica read versions or roots diverged");
      readStates.push_back(replica.getDoneReads().front());
    }
    auto readJoin = builder.create<TaskflowReadCompletionJoinOp>(
        source.getLoc(), source.getDoneReads().front().getType(), readStates,
        baseReadState, originalReadRoot, join.getJoined());
    readJoin->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
    readJoin->setAttr("amoeba.replica.completion_only", builder.getUnitAttr());
    readReplacement = readJoin.getJoined();
    source.getDoneReads().front().replaceAllUsesWith(readReplacement);
  }
  Value replacement = builder.create<memref::CastOp>(
      source.getLoc(), source.getDoneWrites().front().getType(),
      join.getJoined());
  source.getDoneWrites().front().replaceAllUsesWith(replacement);
  source.erase();
  function->setAttr("amoeba.replica.materialized_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.count",
                    builder.getI64IntegerAttr(plan.factor));
  function->setAttr("amoeba.replica.shard_axis",
                    builder.getI64IntegerAttr(plan.axis));
  function->setAttr("amoeba.replica.total_trip_count",
                    builder.getI64IntegerAttr(plan.totalTripCount));
  function->setAttr(
      "amoeba.replica.shard_trip_count",
      builder.getI64IntegerAttr(plan.totalTripCount / plan.factor));
  // The function-wide active-transfer record binds all observed accesses, so
  // cloning a task makes only that detailed access witness stale. Re-derive
  // the complete proof after the graph reaches its final task/join/SSA shape,
  // while requiring every source caller fact and both transfer boxes to stay
  // bit-for-bit fixed. The subsequent TaskEdgeGraph checks then consume the
  // refreshed, source-equivalent record rather than a stale proof.
  if (failed(refreshActiveTransferProofsAfterSourceRewrite(
          function, activeTransferFacts, callerFacts)))
    return failure();
  if (readReplacement)
    for (TaskflowTaskOp consumer : readConsumers)
      if (failed(makeReadCompletionConsumerEdges(consumer, readReplacement,
                                                 sourceName, builder)))
        return failure();
  for (TaskflowTaskOp consumer : consumers)
    if (failed(makeConsumerTensorWideForReplicas(
            consumer, replacement, sourceName, builder, plan.factor, true)))
      return failure();
  return success();
}

// The post-Neura tiler is the source-owned proof for a dynamic M/N task.  A
// generic execution replica must consume that proof rather than re-proving
// storage from the Taskflow SSA state: for a producer/consumer task the
// will-read value is intentionally the producer's done-write, while
// original_read_memrefs retains the caller-visible storage root.  Run the
// tiler on a module clone, then translate only its already-proved tile shell
// into the replica vocabulary.  No operation body or counter is reconstructed
// here.
static std::string quotePassOption(StringRef value) {
  std::string quoted = "\"";
  for (char character : value) {
    if (character == '\\' || character == '"')
      quoted += '\\';
    quoted += character;
  }
  quoted += '"';
  return quoted;
}

static LogicalResult runSourceNeuraTiler(
    ModuleOp module, StringRef taskName, int64_t axis, int64_t factor,
    OwningOpRef<ModuleOp> &trial, std::string &diagnostic) {
  trial = module.clone();
  std::unique_ptr<Pass> pass =
      mlir::amoeba::neura::createMaterializeNeuraJointRewritePass();
  std::string options = "task-name=" + quotePassOption(taskName) +
                        " tile-axis=" + std::to_string(axis) +
                        " tile-factor=" + std::to_string(factor) +
                        " fusion-mode=none";
  std::string optionError;
  if (failed(pass->initializeOptions(
          options, [&](const llvm::Twine &message) {
            optionError = message.str();
            return failure();
          }))) {
    diagnostic = optionError.empty() ? "source tiler option parsing failed"
                                     : optionError;
    return failure();
  }

  PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(std::move(pass));
  LogicalResult result = failure();
  {
    ScopedDiagnosticHandler capture(module.getContext(),
                                    [&](Diagnostic &entry) {
                                      if (diagnostic.empty()) {
                                        llvm::raw_string_ostream stream(
                                            diagnostic);
                                        entry.print(stream);
                                        stream.flush();
                                      }
                                      return success();
                                    });
    result = manager.run(*trial);
  }
  if (failed(result)) {
    if (diagnostic.empty())
      diagnostic = "source-owned post-Neura tiler failed";
    return failure();
  }
  if (failed(verify(trial->getOperation()))) {
    if (diagnostic.empty())
      diagnostic = "source-owned post-Neura tiler produced invalid IR";
    return failure();
  }
  return success();
}

static bool inferReplicaTripCount(TaskflowTaskOp task, int64_t &tripCount) {
  if (!task || !task.getBody().hasOneBlock())
    return false;
  tripCount = 1;
  bool sawCounter = false;
  for (Operation &operation : task.getBody().front()) {
    auto counter = dyn_cast<TaskflowCounterOp>(&operation);
    if (!counter)
      continue;
    auto lower = compileTimeIndex(counter.getLowerBound(), task);
    auto upper = compileTimeIndex(counter.getUpperBound(), task);
    auto step = compileTimeIndex(counter.getStep(), task);
    if (!lower || !upper || !step || *step != 1 || *upper <= *lower)
      return false;
    const int64_t extent = *upper - *lower;
    if (tripCount > std::numeric_limits<int64_t>::max() / extent)
      return false;
    tripCount *= extent;
    sawCounter = true;
  }
  return sawCounter && tripCount > 0;
}

// A generic M/N replica is independent only when the source-owned tiler has
// a stateless kernel body to clone.  A carried Neura iter-arg or reserve is a
// feedback state; treating different counter ranges as independent in that
// case would manufacture a completion certificate.  The specialized LLaMA
// and in-place paths run before this generic path and retain their narrower
// feedback proofs.
static bool hasUnprovedGenericReplicaFeedback(TaskflowTaskOp task) {
  bool feedback = false;
  task.walk([&](neura::KernelOp kernel) {
    feedback |= !kernel.getIterArgsInit().empty() || kernel.getNumResults() != 0;
  });
  task.walk([&](Operation *operation) {
    feedback |= isa<neura::ReserveOp>(operation);
  });
  return feedback;
}

static Value stripMemrefCasts(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>())
    value = cast.getSource();
  return value;
}

// Completion values may be forwarded through a verified completion-only join
// that was already present when a nested replica was materialized. Follow
// only those joins, with a bounded visited set and a storage-root check; an
// arbitrary user or an unclassified join remains a hard failure.
static bool isCompletionValueRecursive(
    Value value, Value joined,
    llvm::SmallPtrSetImpl<Operation *> &visitedJoins) {
  value = stripMemrefCasts(value);
  joined = stripMemrefCasts(joined);
  if (!value || !joined)
    return false;
  if (value == joined)
    return true;
  auto join = value.getDefiningOp<TaskflowJoinOp>();
  if (!join || !join->hasAttr("amoeba.semantic.completion_only") ||
      (!join->hasAttr("amoeba.replica.completion_only") &&
       !join->hasAttr("amoeba.neura.joint_rewrite")) ||
      failed(verify(join.getOperation())))
    return false;
  Value anchorRoot = joined;
  if (auto anchorJoin = joined.getDefiningOp<TaskflowJoinOp>())
    anchorRoot = anchorJoin.getBase();
  if (stripMemrefCasts(join.getBase()) != stripMemrefCasts(anchorRoot))
    return false;
  if (!visitedJoins.insert(join.getOperation()).second)
    return false;
  for (Value state : join.getTileStates())
    if (isCompletionValueRecursive(state, joined, visitedJoins))
      return true;
  return false;
}

static bool isCompletionValue(Value value, Value joined) {
  llvm::SmallPtrSet<Operation *, 32> visitedJoins;
  return isCompletionValueRecursive(value, joined, visitedJoins);
}

static LogicalResult collectCompletionConsumers(
    Value value, Value joined, SmallVectorImpl<TaskflowTaskOp> &consumers) {
  llvm::SmallPtrSet<Operation *, 32> visitedJoins;
  auto storageRoot = [](Value candidate) {
    candidate = stripMemrefCasts(candidate);
    if (auto join = candidate.getDefiningOp<TaskflowJoinOp>())
      return stripMemrefCasts(join.getBase());
    return candidate;
  };
  const Value joinedRoot = storageRoot(joined);
  auto collect = [&](auto &&self, Value current, Value anchor)
      -> LogicalResult {
    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (auto task = dyn_cast<TaskflowTaskOp>(owner)) {
        if (!llvm::is_contained(task.getWillReads(), current) ||
            !isCompletionValue(current, anchor))
          return failure();
        if (!llvm::is_contained(consumers, task))
          consumers.push_back(task);
        continue;
      }
      if (auto cast = dyn_cast<memref::CastOp>(owner)) {
        if (cast.getSource() != current ||
            failed(self(self, cast.getResult(), anchor)))
          return failure();
        continue;
      }
      auto nextJoin = dyn_cast<TaskflowJoinOp>(owner);
      if (!nextJoin || !llvm::is_contained(nextJoin.getTileStates(), current) ||
          !nextJoin->hasAttr("amoeba.semantic.completion_only") ||
          (!nextJoin->hasAttr("amoeba.replica.completion_only") &&
           !nextJoin->hasAttr("amoeba.neura.joint_rewrite")) ||
          storageRoot(nextJoin.getJoined()) != joinedRoot ||
          !visitedJoins.insert(nextJoin.getOperation()).second ||
          failed(verify(nextJoin.getOperation())))
        return failure();
      if (failed(self(self, nextJoin.getJoined(), nextJoin.getJoined())))
        return failure();
    }
    return success();
  };
  return collect(collect, value, joined);
}

// The source tiler uses a static view for its completion join when the caller
// storage is dynamically shaped.  Replica cost inheritance compares the
// mapper-visible task shell and Neura kernel types with the canonical task,
// whose storage remains the dynamic caller memref.  Restore that shell type
// on each generated tile while retaining a static cast at the completion
// join.  This keeps the join's exact-region proof intact and does not alter
// the cloned body or its counter domains.
static LogicalResult restoreDynamicReplicaOutputShell(
    TaskflowTaskOp tile, TaskflowJoinOp completionJoin,
    std::string &diagnostic) {
  if (!tile || !completionJoin || !tile.getBody().hasOneBlock() ||
      tile.getWillWrites().size() != 1 ||
      tile.getOriginalWriteMemrefs().size() != 1 ||
      tile.getDoneWrites().size() != 1) {
    diagnostic = "generated tile has an incomplete output shell";
    return failure();
  }
  Value staticWrite = tile.getWillWrites().front();
  auto staticType = dyn_cast<MemRefType>(staticWrite.getType());
  Value dynamicStorage = stripMemrefCasts(staticWrite);
  auto dynamicType = dyn_cast<MemRefType>(dynamicStorage.getType());
  if (!staticType || !staticType.hasStaticShape() || !dynamicType) {
    diagnostic = "generated tile output does not have a memref shell";
    return failure();
  }
  // A statically shaped source task already has the same mapper-visible shell
  // as its canonical parent.  Keep its direct join states and preserve the
  // established static-output behavior unchanged.
  if (dynamicStorage == staticWrite || dynamicType == staticType)
    return success();
  if (dynamicType.hasStaticShape() ||
      !memref::CastOp::areCastCompatible({dynamicType}, {staticType}) ||
      stripMemrefCasts(tile.getOriginalWriteMemrefs().front()) !=
          dynamicStorage) {
    diagnostic = "generated tile output does not have one compatible dynamic "
                 "caller storage root";
    return failure();
  }

  Value oldDone = tile.getDoneWrites().front();
  unsigned joinState = 0;
  bool foundJoinState = false;
  for (auto [index, state] : llvm::enumerate(completionJoin.getTileStates())) {
    if (state != oldDone)
      continue;
    if (foundJoinState) {
      diagnostic = "generated tile completion state is repeated in the join";
      return failure();
    }
    joinState = index;
    foundJoinState = true;
  }
  if (!foundJoinState) {
    diagnostic = "generated tile completion state is not owned by its join";
    return failure();
  }
  for (OpOperand &use : oldDone.getUses()) {
    if (use.getOwner() != completionJoin.getOperation()) {
      diagnostic = "generated tile completion state has an unclassified use";
      return failure();
    }
  }

  Block &body = tile.getBody().front();
  const unsigned outputOperand = tile.getWillReads().size();
  if (outputOperand >= body.getNumArguments() ||
      body.getArgument(outputOperand).getType() != staticType) {
    diagnostic = "generated tile output body argument does not match its "
                 "static shell type";
    return failure();
  }

  // Keep the original dynamic storage in both the execution and provenance
  // operand segments.  The static join state is rebuilt below from the
  // resulting dynamic done-write.
  const unsigned originalWriteOperand =
      tile.getWillReads().size() + tile.getWillWrites().size() +
      tile.getValueInputs().size() + tile.getOriginalReadMemrefs().size();
  tile->setOperand(outputOperand, dynamicStorage);
  tile->setOperand(originalWriteOperand, dynamicStorage);
  body.getArgument(outputOperand).setType(dynamicType);
  oldDone.setType(dynamicType);

  bool matchedKernelOutput = false;
  LogicalResult kernelResult = success();
  tile.walk([&](neura::KernelOp kernel) {
    if (failed(kernelResult))
      return;
    if (!kernel.getBody().hasOneBlock()) {
      diagnostic = "generated tile kernel has no single body block";
      kernelResult = failure();
      return;
    }
    auto inputs = kernel.getInputs();
    for (auto [index, input] : llvm::enumerate(inputs)) {
      if (input != body.getArgument(outputOperand))
        continue;
      if (matchedKernelOutput || index >= kernel.getBody().front().getNumArguments()) {
        diagnostic = "generated tile output has an ambiguous kernel input";
        kernelResult = failure();
        return;
      }
      BlockArgument kernelArgument = kernel.getBody().front().getArgument(index);
      Type expected = input.getType();
      if (auto payload = dyn_cast<neura::PredicatedValue>(
              kernelArgument.getType())) {
        kernelArgument.setType(neura::PredicatedValue::get(
            kernel.getContext(), expected, payload.getPredicateType()));
      } else if (isa<MemRefType>(kernelArgument.getType())) {
        kernelArgument.setType(expected);
      } else {
        diagnostic = "generated tile output kernel argument is not a memref "
                     "or predicated memref";
        kernelResult = failure();
        return;
      }
      matchedKernelOutput = true;
    }
  });
  if (failed(kernelResult))
    return failure();
  if (!matchedKernelOutput) {
    diagnostic = "generated tile output is not bound to a kernel input";
    return failure();
  }

  OpBuilder castBuilder(completionJoin);
  Value staticState = castBuilder.create<memref::CastOp>(
      tile.getLoc(), staticType, oldDone);
  completionJoin->setOperand(joinState, staticState);
  return success();
}

static LogicalResult convertNeuraTilesToReplicas(
    StringRef functionSymbol, StringRef sourceName, int64_t axis,
    int64_t factor, OwningOpRef<ModuleOp> &trial,
    std::string &diagnostic) {
  func::FuncOp function;
  trial->walk([&](func::FuncOp candidate) {
    if (candidate.getName() == functionSymbol)
      function = candidate;
  });
  if (!function) {
    diagnostic = "source-owned tile clone lost the selected function";
    return failure();
  }

  const std::string tilePrefix =
      (Twine(sourceName) + ".tile." + Twine(axis) + ".").str();
  const std::string replicaPrefix =
      (Twine(sourceName) + ".replica.").str();
  SmallVector<TaskflowTaskOp> tiles;
  std::map<int64_t, TaskflowTaskOp> tileById;
  std::map<std::string, std::string> renamedTasks;
  function.walk([&](TaskflowTaskOp task) {
    auto parent = task->getAttrOfType<StringAttr>(
        "amoeba.neura.tiling.parent_task");
    auto tileAxis = task->getAttrOfType<IntegerAttr>(
        "amoeba.neura.tiling.axis");
    auto part = task->getAttrOfType<IntegerAttr>(
        "amoeba.neura.tiling.part_index");
    if (!parent || !tileAxis || !part || parent.getValue() != sourceName ||
        tileAxis.getInt() != axis)
      return;
    if (task.getTaskName() !=
            (tilePrefix + std::to_string(part.getInt())) ||
        part.getInt() < 0 || part.getInt() >= factor ||
        !tileById.emplace(part.getInt(), task).second) {
      diagnostic = "source-owned tiler returned an ambiguous tile identity";
      return;
    }
    tiles.push_back(task);
  });
  if (!diagnostic.empty() || tiles.size() != static_cast<size_t>(factor) ||
      tileById.size() != static_cast<size_t>(factor)) {
    if (diagnostic.empty())
      diagnostic = "source-owned tiler did not return the requested tile set";
    return failure();
  }

  for (int64_t id = 0; id < factor; ++id) {
    TaskflowTaskOp tile = tileById.find(id)->second;
    const std::string oldName = tile.getTaskName().str();
    const std::string newName = replicaPrefix + std::to_string(id);
    bool collision = false;
    function.walk([&](TaskflowTaskOp task) {
      if (!llvm::is_contained(tiles, task) && task.getTaskName() == newName)
        collision = true;
    });
    if (collision) {
      diagnostic = "replica task name collides with an existing task";
      return failure();
    }
    renamedTasks.emplace(oldName, newName);
  }

  // A module can already contain a completion join for a different tiled
  // task.  Select only a rewrite join whose states are exactly the generated
  // tile done-writes; unrelated joins remain untouched.
  TaskflowJoinOp completionJoin;
  llvm::SmallDenseSet<Operation *, 8> stateTasks;
  function.walk([&](TaskflowJoinOp candidate) {
    if (!candidate->hasAttr("amoeba.neura.joint_rewrite") ||
        candidate.getTileStates().size() != tiles.size())
      return;
    llvm::SmallDenseSet<Operation *, 8> candidateTasks;
    for (Value state : candidate.getTileStates()) {
      auto task = state.getDefiningOp<TaskflowTaskOp>();
      if (!task || !llvm::is_contained(tiles, task) ||
          !llvm::is_contained(task.getDoneWrites(), state) ||
          !candidateTasks.insert(task.getOperation()).second)
        return;
    }
    if (candidateTasks.size() != tiles.size())
      return;
    if (completionJoin) {
      diagnostic = "source-owned tiler returned multiple joins for the "
                   "generated tile set";
      return;
    }
    completionJoin = candidate;
    stateTasks = std::move(candidateTasks);
  });
  if (!diagnostic.empty() || !completionJoin ||
      completionJoin.getAxis() != static_cast<uint64_t>(axis) ||
      !completionJoin->hasAttr("amoeba.semantic.completion_only")) {
    if (diagnostic.empty())
      diagnostic = "source-owned tiler returned no unique completion join for "
                   "the generated tile set";
    return failure();
  }
  // TaskflowJoinOp::verify proves static base/region identity, exact
  // half-open coverage, non-overlap, and one state per tile.  Keep that
  // source-owned verifier in the adapter rather than reproducing a weaker
  // region test here.
  if (failed(verify(completionJoin.getOperation()))) {
    diagnostic = "generated tile completion join failed region verification";
    return failure();
  }

  // Keep the child tile shells structurally comparable with their canonical
  // dynamic-output task.  The completion join continues to consume static
  // cast states, so its half-open region and coverage proof remain unchanged.
  for (TaskflowTaskOp tile : tiles) {
    if (failed(restoreDynamicReplicaOutputShell(tile, completionJoin,
                                                diagnostic)))
      return failure();
    // The source tiler can leave an identity memref.cast on an output base.
    // Fold only casts whose source/result types are identical; the shared
    // output proof then authenticates the same base input and every index.
    SmallVector<memref::CastOp> identityCasts;
    tile.walk([&](memref::CastOp cast) {
      if (cast.getSource().getType() == cast.getType())
        identityCasts.push_back(cast);
    });
    for (memref::CastOp cast : identityCasts) {
      cast.getResult().replaceAllUsesWith(cast.getSource());
      cast.erase();
    }
  }

  OpBuilder builder(function.getContext());
  SmallVector<int64_t> childTripCounts;
  childTripCounts.reserve(tiles.size());
  int64_t totalTripCount = 0;
  for (int64_t id = 0; id < factor; ++id) {
    TaskflowTaskOp tile = tileById.find(id)->second;
    int64_t childTripCount = 0;
    if (!inferReplicaTripCount(tile, childTripCount) ||
        childTripCount > std::numeric_limits<int64_t>::max() - totalTripCount)
      return failure();
    totalTripCount += childTripCount;
    childTripCounts.push_back(childTripCount);
  }
  if (totalTripCount <= 0)
    return failure();

  // Rewrite any typed semantic annotation that names a generated tile before
  // changing the task attributes themselves.  Downstream annotations that
  // refer to the original parent are rebuilt from the typed completion join
  // below, so this map only handles genuine tile-to-tile references.
  SmallVector<TaskflowTaskOp> allTasks;
  function.walk([&](TaskflowTaskOp task) { allTasks.push_back(task); });
  for (TaskflowTaskOp task : allTasks) {
    auto incoming = task->getAttrOfType<ArrayAttr>(kSemanticIncomingEdgesAttr);
    if (!incoming)
      continue;
    SmallVector<Attribute> rebuilt;
    for (Attribute entry : incoming) {
      auto text = dyn_cast<StringAttr>(entry);
      if (!text)
        return failure();
      StringRef encoded = text.getValue();
      size_t separator = encoded.find('|');
      if (separator == StringRef::npos)
        return failure();
      StringRef producer = encoded.take_front(separator);
      auto renamed = renamedTasks.find(producer.str());
      if (renamed == renamedTasks.end()) {
        rebuilt.push_back(entry);
        continue;
      }
      rebuilt.push_back(builder.getStringAttr(
          renamed->second + encoded.drop_front(producer.size()).str()));
    }
    task->setAttr(kSemanticIncomingEdgesAttr, builder.getArrayAttr(rebuilt));
  }

  for (int64_t id = 0; id < factor; ++id) {
    TaskflowTaskOp tile = tileById.find(id)->second;
    SmallVector<StringRef> remove;
    for (NamedAttribute attribute : tile->getAttrs()) {
      StringRef name = attribute.getName().strref();
      if (name.starts_with("amoeba.neura.tiling.") ||
          name == "amoeba.neura.joint_rewrite")
        remove.push_back(name);
    }
    for (StringRef name : remove)
      tile->removeAttr(name);
    tile->setAttr("task_name",
                  builder.getStringAttr(replicaPrefix + std::to_string(id)));
    tile->setAttr("amoeba.replica.parent_task",
                  builder.getStringAttr(sourceName));
    tile->setAttr("amoeba.replica.id", builder.getI64IntegerAttr(id));
    tile->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(factor));
    tile->setAttr("amoeba.replica.shard_axis", builder.getI64IntegerAttr(axis));
    // For this legacy tiler, the selected tile axis is both the source
    // counter ordinal and output dimension.  Record both meanings explicitly
    // so source-partition validation never has to infer them from task names
    // or scheduler metadata.
    tile->setAttr("amoeba.replica.output_shard_axis",
                  builder.getI64IntegerAttr(axis));
    // The source tiler metadata was removed above, so recover the selected
    // counter interval directly from the cloned body.  This is the same
    // static domain already checked by the source pass.
    SmallVector<TaskflowCounterOp> counters;
    for (Operation &operation : tile.getBody().front())
      if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
        counters.push_back(counter);
    if (axis < 0 || axis >= static_cast<int64_t>(counters.size()))
      return failure();
    auto shardLower = compileTimeIndex(counters[axis].getLowerBound(), tile);
    auto shardUpper = compileTimeIndex(counters[axis].getUpperBound(), tile);
    if (!shardLower || !shardUpper || *shardLower >= *shardUpper)
      return failure();
    tile->setAttr("amoeba.replica.shard_lower",
                  builder.getI64IntegerAttr(*shardLower));
    tile->setAttr("amoeba.replica.shard_upper",
                  builder.getI64IntegerAttr(*shardUpper));
    tile->setAttr("trip_count",
                  builder.getI64IntegerAttr(childTripCounts[id]));
    tile->setAttr("amoeba.replica.total_trip_count",
                  builder.getI64IntegerAttr(totalTripCount));
    tile->setAttr("amoeba.replica.shard_trip_count",
                  builder.getI64IntegerAttr(childTripCounts[id]));
    if (tile->hasAttr("amoeba.selected_trip_count"))
      tile->setAttr("amoeba.selected_trip_count",
                    builder.getI64IntegerAttr(childTripCounts[id]));
  }

  completionJoin->removeAttr("amoeba.neura.joint_rewrite");
  completionJoin->setAttr("amoeba.replica.completion_only",
                           builder.getUnitAttr());
  SmallVector<TaskflowTaskOp> consumers;
  if (failed(collectCompletionConsumers(completionJoin.getJoined(),
                                        completionJoin.getJoined(),
                                        consumers))) {
    diagnostic = "completion join has an unclassified downstream user";
    return failure();
  }
  for (TaskflowTaskOp consumer : consumers) {
    Value joinedInput;
    for (Value input : consumer.getWillReads())
      if (isCompletionValue(input, completionJoin.getJoined())) {
        if (joinedInput) {
          diagnostic = "consumer has multiple completion-join inputs";
          return failure();
        }
        joinedInput = input;
      }
    if (!joinedInput ||
        failed(makeConsumerTensorWideForReplicas(
            consumer, joinedInput, sourceName, builder, factor, true))) {
      diagnostic = "completion join semantic edges failed replica conversion";
      return failure();
    }
  }

  function->removeAttr("amoeba.neura.joint_rewrite");
  function->setAttr("amoeba.replica.materialized_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.parent_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(factor));
  function->setAttr("amoeba.replica.shard_axis",
                    builder.getI64IntegerAttr(axis));
  function->setAttr("amoeba.replica.total_trip_count",
                    builder.getI64IntegerAttr(totalTripCount));
  if (llvm::all_of(childTripCounts, [&](int64_t count) {
        return count == childTripCounts.front();
      }))
    function->setAttr("amoeba.replica.shard_trip_count",
                      builder.getI64IntegerAttr(childTripCounts.front()));
  else
    function->removeAttr("amoeba.replica.shard_trip_count");

  if (failed(verify(trial->getOperation()))) {
    diagnostic = "replica-converted module failed verification";
    return failure();
  }
  return success();
}

static LogicalResult materializeGenericNeuraReplica(
    func::FuncOp originalFunction, TaskflowTaskOp source, int64_t axis,
    int64_t factor) {
  ModuleOp originalModule = originalFunction->getParentOfType<ModuleOp>();
  if (!originalModule)
    return failure();
  std::string diagnostic;
  OwningOpRef<ModuleOp> trial;
  if (failed(runSourceNeuraTiler(originalModule, source.getTaskName(), axis,
                                 factor, trial, diagnostic)) ||
      failed(convertNeuraTilesToReplicas(
          originalFunction.getName(), source.getTaskName(), axis, factor,
          trial, diagnostic))) {
    source.emitError() << "source-owned post-Neura replica materialization "
                          "failed: "
                       << diagnostic;
    return failure();
  }
  // Commit only after the source-owned pass, replica conversion, typed edge
  // checks, and verifier have all succeeded on the clone.  Clone operations
  // once more into the original module so no operation in the committed IR
  // retains ownership by the trial module.
  SmallVector<Operation *> committedOperations;
  for (Operation &operation : trial->getBody()->getOperations())
    committedOperations.push_back(operation.clone());
  originalModule.getBody()->clear();
  for (Operation *operation : committedOperations)
    originalModule.getBody()->push_back(operation);
  if (failed(verify(originalModule.getOperation()))) {
    originalModule.emitError(
        "source-owned post-Neura replica commit failed verification");
    return failure();
  }
  return success();
}

static bool isInPlaceNeuraReplicaCandidate(TaskflowTaskOp source) {
  bool hasKernel = false;
  source.walk([&](neura::KernelOp) { hasKernel = true; });
  return hasKernel && !source.getWillReads().empty() &&
         source.getWillWrites().size() == 1 &&
         source.getOriginalReadMemrefs().size() == source.getWillReads().size() &&
         source.getOriginalWriteMemrefs().size() == 1 &&
         llvm::count(source.getOriginalReadMemrefs(),
                     source.getOriginalWriteMemrefs().front()) == 1;
}

// The original TaskScheduler specifies how many replicas run, their shapes,
// placement cells, dispatch ordering, and timing.  It does not specify which
// source counter each replica computes.  Choose that partition here, after
// authenticating the exact output-coordinate map and requiring one exact,
// evenly divisible counter interval.  Candidate source tiling is performed
// on private clones so a failed axis cannot partially rewrite the module.
static LogicalResult chooseOriginalAmoebaAxis(
    func::FuncOp function, TaskflowTaskOp source, int64_t factor,
    int64_t &outputAxis, ReplicaOutputCoordinateProof &selectedProof,
    std::string &error) {
  bool inPlace = isInPlaceNeuraReplicaCandidate(source);
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (!module) {
    error = "original AMOEBA realization has no parent module";
    return failure();
  }

  SmallVector<std::pair<int64_t, ReplicaOutputCoordinateProof>, 1> candidates;
  SmallVector<std::string, 3> rejectedAxes;
  auto recordAxisRejection = [&](int64_t axis, StringRef reason) {
    std::string detail = reason.str();
    constexpr size_t kMaxAxisDiagnosticLength = 240;
    if (detail.size() > kMaxAxisDiagnosticLength)
      detail.resize(kMaxAxisDiagnosticLength);
    rejectedAxes.push_back(
        (Twine("axis ") + Twine(axis) + ": " + detail).str());
  };
  for (int64_t axis = 0; axis < 3; ++axis) {
    ReplicaOutputCoordinateProof proof =
        analyzeReplicaOutputCoordinates(source, axis);
    if (!proof.proven) {
      if (proof.reason.empty())
        recordAxisRejection(axis, "output-coordinate proof failed");
      else
        recordAxisRejection(axis, proof.reason);
      continue;
    }
    if (!proof.selectedAxisIndependent) {
      recordAxisRejection(axis,
                          "selected output coordinate is not axis-independent");
      continue;
    }
    if (axis >= static_cast<int64_t>(proof.outputCounterAxes.size())) {
      recordAxisRejection(axis, "output-coordinate proof has no selected axis");
      continue;
    }
    unsigned counter = proof.outputCounterAxes[axis];
    if (counter == ReplicaOutputCoordinateProof::kConstantAxis ||
        counter >= proof.taskLowers.size() ||
        counter >= proof.taskUppers.size() ||
        !hasBalancedShards(
            proof.taskUppers[counter] - proof.taskLowers[counter], factor)) {
      recordAxisRejection(
          axis, "selected counter has no exact balanced source partition");
      continue;
    }

    bool materializable = false;
    std::string ignoredDiagnostic;
    SmallVector<std::string, 2> capturedDiagnostics;
    auto captureDiagnostic = [&](Diagnostic &diagnostic) {
      std::string detail;
      for (const DiagnosticArgument &argument : diagnostic.getArguments()) {
        // Keep only textual reasons. Operation and attribute arguments can
        // print the full rejected IR, which is not useful in this summary.
        if (argument.getKind() !=
            DiagnosticArgument::DiagnosticArgumentKind::String)
          continue;
        StringRef text = argument.getAsString();
        if (text.empty())
          continue;
        if (!detail.empty())
          detail += ' ';
        detail.append(text.data(), text.size());
        if (detail.size() >= 240) {
          detail.resize(240);
          break;
        }
      }
      if (!detail.empty())
        capturedDiagnostics.push_back(std::move(detail));
      return success();
    };
    if (inPlace) {
      func::FuncOp clonedFunction = cast<func::FuncOp>(function->clone());
      OwningOpRef<func::FuncOp> ownedFunction(clonedFunction);
      TaskflowTaskOp clonedTask;
      clonedFunction.walk([&](TaskflowTaskOp task) {
        if (task.getTaskName() == source.getTaskName())
          clonedTask = task;
      });
      if (clonedTask) {
        ScopedDiagnosticHandler capture(function.getContext(),
                                        captureDiagnostic);
        materializable =
            succeeded(analyzeReplicaNeuraInPlace(clonedTask, axis, factor));
      }
    } else {
      OwningOpRef<ModuleOp> trial;
      ScopedDiagnosticHandler capture(function.getContext(), captureDiagnostic);
      materializable =
          succeeded(runSourceNeuraTiler(module, source.getTaskName(), axis,
                                        factor, trial, ignoredDiagnostic)) &&
          succeeded(convertNeuraTilesToReplicas(
              function.getName(), source.getTaskName(), axis, factor, trial,
              ignoredDiagnostic));
    }
    if (materializable)
      candidates.emplace_back(axis, std::move(proof));
    else if (!ignoredDiagnostic.empty())
      recordAxisRejection(axis, ignoredDiagnostic);
    else if (!capturedDiagnostics.empty())
      recordAxisRejection(axis, capturedDiagnostics.front());
    else
      recordAxisRejection(axis, "source replica materialization was rejected");
  }
  if (candidates.size() != 1) {
    error = candidates.empty()
                ? "no output axis has a source-proved exact partition and a "
                  "legal replica materialization"
                : "more than one output axis has a legal fixed-decision "
                  "source partition; refusing an ambiguous realization";
    if (candidates.empty() && !rejectedAxes.empty()) {
      error += "; axis diagnostics: ";
      llvm::interleave(
          rejectedAxes, [&](const std::string &reason) { error += reason; },
          [&] { error += "; "; });
    }
    return failure();
  }
  outputAxis = candidates.front().first;
  selectedProof = std::move(candidates.front().second);
  return success();
}

static std::optional<int64_t> positiveSourceInternalMultiplicity(
    TaskflowTaskOp task) {
  auto domain = task->getAttrOfType<DictionaryAttr>(
      kSourceIterationDomainAttr);
  auto complete = domain ? domain.getAs<BoolAttr>("complete") : BoolAttr();
  auto multiplicity =
      domain ? domain.getAs<IntegerAttr>("internal_multiplicity")
             : IntegerAttr();
  if (!complete || !complete.getValue() || !multiplicity ||
      multiplicity.getInt() <= 0)
    return std::nullopt;
  return multiplicity.getInt();
}

static LogicalResult attachOriginalAmoebaRealization(
    func::FuncOp function, StringRef parentName, int64_t factor,
    int64_t outputAxis, const ReplicaOutputCoordinateProof &sourceProof,
    ArrayRef<DictionaryAttr> replicaShapes, DictionaryAttr schedulerRecord,
    TaskflowTaskOp canonicalSource) {
  auto fail = [&](StringRef reason) {
    function.emitError() << "original AMOEBA source realization: " << reason;
    return failure();
  };
  if (!function || !schedulerRecord ||
      replicaShapes.size() != static_cast<size_t>(factor) ||
      outputAxis < 0 ||
      outputAxis >= static_cast<int64_t>(sourceProof.outputCounterAxes.size()))
    return fail("decision and output-coordinate bindings are incomplete");
  unsigned selectedCounter = sourceProof.outputCounterAxes[outputAxis];
  if (selectedCounter == ReplicaOutputCoordinateProof::kConstantAxis ||
      selectedCounter >= sourceProof.taskLowers.size() ||
      selectedCounter >= sourceProof.taskUppers.size())
    return fail("selected output axis has no authenticated source counter");
  std::optional<int64_t> internalMultiplicity =
      positiveSourceInternalMultiplicity(canonicalSource);
  if (!internalMultiplicity)
    return fail("canonical task lacks a complete source-domain witness");

  OpBuilder builder(function.getContext());
  SmallVector<NamedAttribute> domainFields;
  domainFields.push_back(builder.getNamedAttr(
      "schema", builder.getStringAttr("amoeba-source-domain-v1")));
  SmallVector<Attribute> originalAxes;
  for (auto [ordinal, lower] : llvm::enumerate(sourceProof.taskLowers)) {
    int64_t upper = sourceProof.taskUppers[ordinal];
    if (upper <= lower)
      return fail("source task has an invalid counter interval");
    originalAxes.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
        builder.getNamedAttr("lower", builder.getI64IntegerAttr(lower)),
        builder.getNamedAttr("upper", builder.getI64IntegerAttr(upper)),
        builder.getNamedAttr("step", builder.getI64IntegerAttr(1))}));
  }

  SmallVector<TaskflowTaskOp> children;
  function.walk([&](TaskflowTaskOp task) {
    auto parent = task->getAttrOfType<StringAttr>("amoeba.replica.parent_task");
    if (parent && parent.getValue() == parentName)
      children.push_back(task);
  });
  if (children.size() != static_cast<size_t>(factor))
    return fail("expanded task count differs from the original replica count");
  SmallVector<TaskflowTaskOp> byId(factor);
  for (TaskflowTaskOp child : children) {
    auto id = child->getAttrOfType<IntegerAttr>("amoeba.replica.id");
    auto count = child->getAttrOfType<IntegerAttr>("amoeba.replica.count");
    if (!id || !count || id.getInt() < 0 || id.getInt() >= factor ||
        count.getInt() != factor || byId[id.getInt()])
      return fail("expanded task ids are duplicate or disagree with f45 count");
    byId[id.getInt()] = child;
  }

  SmallVector<Attribute> replicaManifest;
  SmallVector<Attribute> allPlacementRecords;
  if (auto placements = schedulerRecord.getAs<ArrayAttr>("placements"))
    allPlacementRecords.append(placements.begin(), placements.end());
  else
    return fail("original decision has no physical placements");

  for (int64_t replicaId = 0; replicaId < factor; ++replicaId) {
    TaskflowTaskOp child = byId[replicaId];
    if (!child)
      return fail("expanded task group is missing an original replica id");
    SmallVector<int64_t> childLowers;
    SmallVector<int64_t> childUppers;
    SmallVector<TaskflowCounterOp> counters;
    for (Operation &operation : child.getBody().front())
      if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
        counters.push_back(counter);
    llvm::sort(counters, [](TaskflowCounterOp lhs, TaskflowCounterOp rhs) {
      auto left = lhs->getAttrOfType<IntegerAttr>("counter_id");
      auto right = rhs->getAttrOfType<IntegerAttr>("counter_id");
      return left && right && left.getInt() < right.getInt();
    });
    if (counters.size() != sourceProof.taskLowers.size())
      return fail("expanded shard changed the number of source counters");
    int64_t sourceWork = *internalMultiplicity;
    SmallVector<Attribute> partitionAxes;
    for (auto [ordinal, counter] : llvm::enumerate(counters)) {
      auto lower = compileTimeIndex(counter.getLowerBound(), child);
      auto upper = compileTimeIndex(counter.getUpperBound(), child);
      auto step = compileTimeIndex(counter.getStep(), child);
      if (!lower || !upper || !step || *step != 1 || *upper <= *lower)
        return fail("expanded shard has an unresolved source counter interval");
      childLowers.push_back(*lower);
      childUppers.push_back(*upper);
      int64_t extent = *upper - *lower;
      if (sourceWork > std::numeric_limits<int64_t>::max() / extent)
        return fail("expanded shard source-work count overflows");
      sourceWork *= extent;
      partitionAxes.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
          builder.getNamedAttr("lower", builder.getI64IntegerAttr(*lower)),
          builder.getNamedAttr("upper", builder.getI64IntegerAttr(*upper)),
          builder.getNamedAttr("step", builder.getI64IntegerAttr(*step))}));
    }
    for (unsigned ordinal = 0; ordinal < childLowers.size(); ++ordinal) {
      if (ordinal == selectedCounter)
        continue;
      if (childLowers[ordinal] != sourceProof.taskLowers[ordinal] ||
          childUppers[ordinal] != sourceProof.taskUppers[ordinal])
        return fail("materializer changed a nonselected source counter");
    }
    int64_t expectedWidth =
        (sourceProof.taskUppers[selectedCounter] -
         sourceProof.taskLowers[selectedCounter]) /
        factor;
    int64_t expectedLower =
        sourceProof.taskLowers[selectedCounter] + replicaId * expectedWidth;
    if (childLowers[selectedCounter] != expectedLower ||
        childUppers[selectedCounter] != expectedLower + expectedWidth)
      return fail("expanded intervals do not preserve ordered equal shards");

    SmallVector<Attribute> placements;
    for (Attribute attribute : allPlacementRecords) {
      auto placement = dyn_cast<DictionaryAttr>(attribute);
      auto id = placement ? placement.getAs<IntegerAttr>("replica_id")
                          : IntegerAttr();
      if (!id)
        return fail("original placement has no replica id");
      if (id.getInt() == replicaId)
        placements.push_back(attribute);
    }
    if (placements.empty())
      return fail("original replica has no placement cells");
    replicaManifest.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("replica_id", builder.getI64IntegerAttr(replicaId)),
        builder.getNamedAttr("task_name", builder.getStringAttr(
                                               child.getTaskName())),
        builder.getNamedAttr("replica_shape", replicaShapes[replicaId]),
        builder.getNamedAttr("placements", builder.getArrayAttr(placements)),
        builder.getNamedAttr("partition_bounds",
                             builder.getArrayAttr(partitionAxes)),
        builder.getNamedAttr("source_work_count",
                             builder.getI64IntegerAttr(sourceWork))}));

    SmallVector<NamedAttribute> replicaDecisionFields;
    replicaDecisionFields.push_back(builder.getNamedAttr(
        "scheduler_record", schedulerRecord));
    replicaDecisionFields.push_back(builder.getNamedAttr(
        "replica_shape", replicaShapes[replicaId]));
    replicaDecisionFields.push_back(builder.getNamedAttr(
        "placements", builder.getArrayAttr(placements)));
    child->setAttr(kOriginalAmoebaReplicaDecisionAttr,
                   builder.getDictionaryAttr(replicaDecisionFields));
    child->setAttr(kOriginalAmoebaReplicaPlacementsAttr,
                   builder.getArrayAttr(placements));

    SmallVector<NamedAttribute> realizationFields;
    realizationFields.push_back(builder.getNamedAttr(
        "schema", builder.getStringAttr(
                      "amoeba-source-certified-replica-realization-v1")));
    realizationFields.push_back(builder.getNamedAttr(
        "status", builder.getStringAttr(
                      "verified-source-certified-realization-v1")));
    realizationFields.push_back(builder.getNamedAttr(
        "origin", builder.getStringAttr(
                      "orbit-native-materializer-selected-counter-axis-after-f45")));
    realizationFields.push_back(builder.getNamedAttr(
        "f45_axis_and_bounds_present", builder.getBoolAttr(false)));
    realizationFields.push_back(builder.getNamedAttr(
        "selected_output_axis", builder.getI64IntegerAttr(outputAxis)));
    realizationFields.push_back(builder.getNamedAttr(
        "selected_counter_ordinal",
        builder.getI64IntegerAttr(selectedCounter)));
    realizationFields.push_back(builder.getNamedAttr(
        "original_replica_count", builder.getI64IntegerAttr(factor)));
    realizationFields.push_back(builder.getNamedAttr(
        "replica_id", builder.getI64IntegerAttr(replicaId)));
    realizationFields.push_back(builder.getNamedAttr(
        "original_domain", builder.getArrayAttr(originalAxes)));
    realizationFields.push_back(builder.getNamedAttr(
        "partition_bounds", builder.getArrayAttr(partitionAxes)));
    realizationFields.push_back(builder.getNamedAttr(
        "source_work_count", builder.getI64IntegerAttr(sourceWork)));
    realizationFields.push_back(builder.getNamedAttr(
        "internal_multiplicity",
        builder.getI64IntegerAttr(*internalMultiplicity)));
    realizationFields.push_back(builder.getNamedAttr(
        "decision_binding", builder.getStringAttr(
                                "amoeba.task_scheduler_schedule_info")));
    child->setAttr(kOriginalAmoebaPartitionRealizationAttr,
                   builder.getDictionaryAttr(realizationFields));

    // These are parent-firing costs, not the current shard mapper result.
    // Drop them so no consumer can accidentally treat a full-task profile as
    // a per-replica measurement. The fresh profile evidence is imported only
    // by the fixed-decision cost adapter.
    child->removeAttr("profile_info");
    child->removeAttr("task_orchestration_info");
  }

  constexpr StringLiteral kDecisionManifestAttr =
      "amoeba.original_amoeba.materialized_decisions";
  SmallVector<Attribute> decisions;
  if (auto existing = function->getAttrOfType<ArrayAttr>(kDecisionManifestAttr))
    decisions.append(existing.begin(), existing.end());
  if (llvm::any_of(decisions, [&](Attribute attribute) {
        auto decision = dyn_cast<DictionaryAttr>(attribute);
        auto parent = decision ? decision.getAs<StringAttr>("parent_task")
                               : StringAttr();
        return parent && parent.getValue() == parentName;
      }))
    return fail("parent decision was already recorded in the function manifest");
  decisions.push_back(builder.getDictionaryAttr({
      builder.getNamedAttr("schema", builder.getStringAttr(
                                         "amoeba-original-fixed-decision-realization-v1")),
      builder.getNamedAttr("parent_task", builder.getStringAttr(parentName)),
      builder.getNamedAttr("original_replica_count",
                           builder.getI64IntegerAttr(factor)),
      builder.getNamedAttr("selected_output_axis",
                           builder.getI64IntegerAttr(outputAxis)),
      builder.getNamedAttr("selected_counter_ordinal",
                           builder.getI64IntegerAttr(selectedCounter)),
      builder.getNamedAttr("original_domain", builder.getArrayAttr(originalAxes)),
      builder.getNamedAttr("scheduler_record", schedulerRecord),
      builder.getNamedAttr("replicas", builder.getArrayAttr(replicaManifest))}));
  function->setAttr(kDecisionManifestAttr, builder.getArrayAttr(decisions));
  // Keep the legacy scalar summary for existing tooling. The append-only
  // per-parent manifest above preserves all groups in a multi-task function.
  function->setAttr("amoeba.original_amoeba.materialized_parent_task",
                    builder.getStringAttr(parentName));
  function->setAttr("amoeba.original_amoeba.materialized_count",
                    builder.getI64IntegerAttr(factor));
  function->setAttr("amoeba.original_amoeba.selected_output_axis",
                    builder.getI64IntegerAttr(outputAxis));
  function->setAttr("amoeba.original_amoeba.selected_counter_ordinal",
                    builder.getI64IntegerAttr(selectedCounter));
  return success();
}

static LogicalResult materializeOriginalAmoebaFixedDecision(
    ModuleOp module, func::FuncOp function, func::FuncOp canonicalFunction,
    TaskflowTaskOp source) {
  auto reject = [&](StringRef reason) {
    source.emitError() << "original AMOEBA fixed-decision materialization: "
                       << reason;
    return failure();
  };
  if (!isOriginalAmoebaFixedDecisionTask(function, source))
    return reject("only the pinned GCN Task_16/Task_17, Harris "
                  "Task_19/Task_21, and LU Task_6 baseline tasks are "
                  "supported");
  const std::string functionSymbol = function.getSymName().str();
  const std::string parentName = source.getTaskName().str();

  int64_t factor = 0;
  SmallVector<DictionaryAttr> replicaShapes;
  if (failed(readOriginalAmoebaReplicaDecision(source, factor,
                                                replicaShapes)))
    return failure();
  if (factor < 2 || factor > 4)
    return reject("original fixed replica count must be between two and four");
  auto schedulerRecord = source->getAttrOfType<DictionaryAttr>(
      "amoeba.task_scheduler_schedule_info");
  if (!schedulerRecord)
    return reject("original TaskScheduler decision record is missing");

  TaskflowTaskOp canonicalSource;
  canonicalFunction.walk([&](TaskflowTaskOp task) {
    if (task.getTaskName() == parentName)
      canonicalSource = task;
  });
  if (!canonicalSource)
    return reject("cannot retain an immutable canonical source task");

  OpBuilder builder(source.getContext());
  source->setAttr(kOriginalAmoebaDecisionModeAttr, builder.getBoolAttr(true));
  int64_t outputAxis = -1;
  ReplicaOutputCoordinateProof sourceProof;
  std::string error;
  if (failed(chooseOriginalAmoebaAxis(function, source, factor, outputAxis,
                                      sourceProof, error)))
    return reject(error);

  if (isInPlaceNeuraReplicaCandidate(source)) {
    FailureOr<ReplicaNeuraInPlacePlan> plan =
        analyzeReplicaNeuraInPlace(source, outputAxis, factor);
    if (failed(plan) || failed(materializeReplicaNeuraInPlace(function, *plan)))
      return failure();
  } else {
    if (hasUnprovedGenericReplicaFeedback(source))
      return reject("source tiler cannot authenticate kernel feedback state");
    if (failed(materializeGenericNeuraReplica(function, source, outputAxis,
                                               factor)))
      return failure();
  }

  FailureOr<func::FuncOp> materializedFunction = selectTaskFunction(
      module, functionSymbol, error);
  if (failed(materializedFunction))
    return reject("cannot reselect the expanded function after materialization");
  func::FuncOp currentFunction = *materializedFunction;
  if (failed(attachOriginalAmoebaRealization(
          currentFunction, parentName, factor, outputAxis,
          sourceProof, replicaShapes, schedulerRecord, canonicalSource)))
    return failure();
  // Include the source-certified realization metadata in the final exact task
  // control binding.  The full graph proof still rechecks every existing and
  // newly materialized replica group before refreshing any binding.
  if (failed(proveAndRefreshSourceIterationDomainPartition(
          canonicalFunction, currentFunction, error))) {
    currentFunction.emitError()
        << "expanded source partition is not authenticated: " << error;
    return failure();
  }
  currentFunction->setAttr(kOriginalAmoebaDecisionModeAttr,
                           builder.getBoolAttr(true));
  if (failed(verify(module.getOperation()))) {
    currentFunction.emitError(
        "fixed-decision source-certified realization failed module verification");
    return failure();
  }
  return success();
}

// Fresh Radar Task_2/Task_3 kernels are stateless post-Neura tasks with two
// independently written outputs.  The source-owned generic M/N tiler is
// intentionally one-output today, so this narrow path performs the same
// source-owned proof directly and leaves all other workloads on the existing
// dispatch paths.  Counter id 0 is the 64-element range axis and id 1 is the
// 32-element chirp axis; both output stores use the explicit permutation
// [counter-1, counter-0].
static LogicalResult rejectRadarMultiOutput(TaskflowTaskOp source,
                                            StringRef reason) {
  source.emitError() << reason;
  return failure();
}

static bool isRadarMultiOutputTask(TaskflowTaskOp task) {
  return task && (task.getTaskName() == "Task_2" ||
                  task.getTaskName() == "Task_3") &&
         task.getWillWrites().size() == 2 &&
         task.getDoneWrites().size() == 2 &&
         task.getOriginalWriteMemrefs().size() == 2;
}

static LogicalResult validateRadarMultiOutputProducer(
    func::FuncOp function, TaskflowTaskOp source, int64_t axis,
    int64_t replicas, SmallVectorImpl<TaskflowTaskOp> &consumers) {
  if (!isRadarFunction(function) || !isRadarMultiOutputTask(source))
    return failure();
  if (axis < 0 || axis > 1 || (replicas != 2 && replicas != 4))
    return rejectRadarMultiOutput(
        source, "Radar multi-output replica requires axis 0/1 and factor 2/4");
  if (auto dlp = source->getAttrOfType<BoolAttr>("dlp_replicable");
      dlp && !dlp.getValue())
    return rejectRadarMultiOutput(
        source, "Radar multi-output replica rejected dlp_replicable=false");
  if (!source.getDoneReads().empty() || !source.getValueOutputs().empty() ||
      source.getOriginalReadMemrefs().size() != source.getWillReads().size())
    return rejectRadarMultiOutput(
        source, "Radar multi-output replica requires complete read provenance "
                "and no completion/value output");

  SmallVector<Value> outputs(source.getWillWrites());
  SmallVector<Value> outputRoots(source.getOriginalWriteMemrefs());
  llvm::SmallDenseSet<Value, 4> distinctOutputs;
  for (auto [index, output] : llvm::enumerate(outputs)) {
    auto type = dyn_cast<MemRefType>(output.getType());
    if (!type || type.getRank() != 2 ||
        type.getElementType() != IntegerType::get(source.getContext(), 32) ||
        type.isDynamicDim(1) || type.getShape()[1] != 256 ||
        !radarShapeMatches(output, type, {32, 256}) ||
        outputRoots[index] != output || !isRadarNoAliasArgument(output, function) ||
        !distinctOutputs.insert(output).second)
      return rejectRadarMultiOutput(
          source, "Radar multi-output writes need two distinct noalias "
                  "memref<?x256xi32> roots");
  }
  llvm::SmallDenseSet<Value, 16> readRoots;
  for (auto [index, root] : llvm::enumerate(source.getOriginalReadMemrefs())) {
    if (!isRadarNoAliasArgument(root, function) ||
        !readRoots.insert(root).second || distinctOutputs.contains(root))
      return rejectRadarMultiOutput(
          source, "Radar multi-output read storage is not a distinct noalias "
                  "read-only root");
    if (llvm::is_contained(source.getOriginalWriteMemrefs(), root))
      return rejectRadarMultiOutput(
          source, "Radar multi-output input/output storage aliases");
  }

  Block &body = source.getBody().front();
  SmallVector<TaskflowCounterOp> taskCounters;
  SmallVector<neura::KernelOp> kernels;
  for (Operation &operation : body) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      taskCounters.push_back(counter);
    else if (auto kernel = dyn_cast<neura::KernelOp>(&operation))
      kernels.push_back(kernel);
    else if (!isa<arith::ConstantOp, arith::ConstantIndexOp, TaskflowYieldOp>(
                 &operation))
      return rejectRadarMultiOutput(
          source, "Radar multi-output task has an unsupported top-level op");
  }
  if (taskCounters.size() != 2 || kernels.size() != 1 ||
      !kernels.front().getBody().hasOneBlock())
    return rejectRadarMultiOutput(
        source, "Radar multi-output task requires two counters and one kernel");
  for (int64_t id = 0; id < 2; ++id) {
    unsigned matches = 0;
    for (TaskflowCounterOp counter : taskCounters)
      if (auto counterId = counter->getAttrOfType<IntegerAttr>("counter_id");
          counterId && counterId.getInt() == id)
        ++matches;
    if (matches != 1)
      return rejectRadarMultiOutput(
          source, "Radar multi-output Taskflow counter ids are ambiguous");
  }
  TaskflowCounterOp rootCounter;
  TaskflowCounterOp chirpCounter;
  for (TaskflowCounterOp counter : taskCounters) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    if (!id)
      return rejectRadarMultiOutput(
          source, "Radar multi-output Taskflow counter id is missing");
    if (id.getInt() == 0)
      rootCounter = counter;
    else
      chirpCounter = counter;
  }
  if (!rootCounter || !chirpCounter || rootCounter.getParentIndex() ||
      chirpCounter.getParentIndex() != rootCounter.getCounterIndex() ||
      compileTimeIndex(rootCounter.getLowerBound(), source) != 0 ||
      compileTimeIndex(rootCounter.getUpperBound(), source) != 64 ||
      compileTimeIndex(rootCounter.getStep(), source) != 1 ||
      compileTimeIndex(chirpCounter.getLowerBound(), source) != 0 ||
      compileTimeIndex(chirpCounter.getUpperBound(), source) != 32 ||
      compileTimeIndex(chirpCounter.getStep(), source) != 1)
    return rejectRadarMultiOutput(
        source, "Radar multi-output Taskflow bounds are not range64/chirp32");

  neura::KernelOp kernel = kernels.front();
  if (!kernel.getIterArgsInit().empty() || kernel.getNumResults() != 0 ||
      hasUnprovedGenericReplicaFeedback(source))
    return rejectRadarMultiOutput(
        source, "Radar multi-output kernel carries feedback state");
  SmallVector<neura::CounterOp> neuraCounters;
  kernel.walk([&](neura::CounterOp counter) { neuraCounters.push_back(counter); });
  if (neuraCounters.size() != 2)
    return rejectRadarMultiOutput(
        source, "Radar multi-output kernel requires two Neura counters");
  SmallVector<neura::CounterOp> neuraById(2);
  for (neura::CounterOp counter : neuraCounters) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    if (!id || id.getInt() < 0 || id.getInt() > 1 || neuraById[id.getInt()])
      return rejectRadarMultiOutput(
          source, "Radar multi-output Neura counter ids are ambiguous");
    neuraById[id.getInt()] = counter;
  }
  for (int64_t id = 0; id < 2; ++id) {
    auto lower = compileTimeCounterBound(
        neuraById[id], "lower_bound_value", neuraById[id].getLowerBound(),
        source, kernel);
    auto upper = compileTimeCounterBound(
        neuraById[id], "upper_bound_value", neuraById[id].getUpperBound(),
        source, kernel);
    auto step = compileTimeCounterBound(
        neuraById[id], "step_value", neuraById[id].getStep(), source, kernel);
    if (lower != 0 || upper != (id == 0 ? 64 : 32) || step != 1)
      return rejectRadarMultiOutput(
          source, "Radar multi-output Neura bounds disagree with "
                  "range64/chirp32");
  }

  Block &kernelBody = kernel.getBody().front();
  if (kernelBody.getNumArguments() != kernel.getInputs().size())
    return rejectRadarMultiOutput(
        source, "Radar multi-output kernel input/block ABI is inconsistent");
  SmallVector<unsigned> outputInputIndices;
  outputInputIndices.reserve(outputs.size());
  for (unsigned writeIndex = 0; writeIndex < outputs.size(); ++writeIndex) {
    unsigned bodyIndex = source.getWillReads().size() + writeIndex;
    if (bodyIndex >= body.getNumArguments())
      return rejectRadarMultiOutput(source,
                                    "Radar multi-output output body argument "
                                    "is missing");
    auto found = llvm::find(kernel.getInputs(), body.getArgument(bodyIndex));
    if (found == kernel.getInputs().end())
      return rejectRadarMultiOutput(
          source, "Radar multi-output kernel lost an output input binding");
    outputInputIndices.push_back(found - kernel.getInputs().begin());
  }

  llvm::SmallDenseSet<unsigned, 4> storedOutputs;
  unsigned loads = 0;
  bool badAccess = false;
  kernel.walk([&](neura::LoadIndexedOp load) {
    if (load.getIndices().size() != 1 && load.getIndices().size() != 2)
      return;
    ++loads;
    if (load.getIndices().size() == 2) {
      if (load.getIndices()[0] != neuraById[1].getResult() ||
          load.getIndices()[1] != neuraById[0].getResult())
        badAccess = true;
    } else if (load.getIndices()[0] != neuraById[1].getResult()) {
      badAccess = true;
    }
    auto input = parseKernelInputReference(load->getAttr("lhs_value"));
    if (!input || *input >= kernel.getInputs().size() ||
        (load.getBase() && load.getBase() != kernel.getInputs()[*input]))
      badAccess = true;
  });
  unsigned stores = 0;
  kernel.walk([&](neura::StoreIndexedOp store) {
    ++stores;
    if (store.getIndices().size() != 2 ||
        store.getIndices()[0] != neuraById[1].getResult() ||
        store.getIndices()[1] != neuraById[0].getResult()) {
      badAccess = true;
      return;
    }
    auto output = parseKernelInputReference(store->getAttr("rhs_value"));
    if (!output || !llvm::is_contained(outputInputIndices, *output) ||
        (store.getBase() && store.getBase() != kernel.getInputs()[*output])) {
      badAccess = true;
      return;
    }
    storedOutputs.insert(*output);
  });
  if (badAccess || loads == 0 ||
      stores != outputs.size() || storedOutputs.size() != outputs.size())
    return rejectRadarMultiOutput(
        source, "Radar multi-output indexed accesses do not prove the "
                "[counter1,counter0] permutation for both outputs");
  bool carriesState = false;
  kernel.walk([&](Operation *operation) {
    StringRef name = operation->getName().getStringRef();
    if (name == "neura.reserve" || name == "neura.phi" ||
        name == "neura.ctrl_mov")
      carriesState = true;
  });
  if (carriesState)
    return rejectRadarMultiOutput(source,
                                  "Radar multi-output kernel contains carry state");

  // Every completion result must have only ordinary Taskflow consumers, and
  // each consumer must name the corresponding storage root in provenance.
  for (auto [writeIndex, state] : llvm::enumerate(source.getDoneWrites())) {
    for (OpOperand &use : state.getUses()) {
      auto consumer = dyn_cast<TaskflowTaskOp>(use.getOwner());
      if (!consumer || !llvm::is_contained(consumer.getWillReads(), state))
        return rejectRadarMultiOutput(
            source, "Radar multi-output completion has an unclassified user");
      unsigned readIndex =
          llvm::find(consumer.getWillReads(), state) - consumer.getWillReads().begin();
      if (readIndex >= consumer.getOriginalReadMemrefs().size() ||
          consumer.getOriginalReadMemrefs()[readIndex] != outputRoots[writeIndex] ||
          llvm::is_contained(consumer.getWillWrites(), outputRoots[writeIndex]))
        return rejectRadarMultiOutput(
            source, "Radar multi-output completion consumer provenance is "
                    "missing or writable");
      if (!llvm::is_contained(consumers, consumer))
        consumers.push_back(consumer);
    }
  }
  if (consumers.empty())
    return rejectRadarMultiOutput(
        source, "Radar multi-output task has no downstream completion consumer");
  return success();
}

static LogicalResult setRadarMultiOutputConsumerEdges(
    TaskflowTaskOp consumer, ArrayRef<Value> joinedValues, StringRef parent,
    OpBuilder &builder) {
  SmallVector<Attribute> incoming;
  bool hadPreviousAnnotations = false;
  if (auto previous = consumer->getAttrOfType<ArrayAttr>(
          kSemanticIncomingEdgesAttr))
    hadPreviousAnnotations = true;
  if (auto previous = consumer->getAttrOfType<ArrayAttr>(
          kSemanticIncomingEdgesAttr))
    for (Attribute entry : previous) {
      auto text = dyn_cast<StringAttr>(entry);
      if (!text)
        return failure();
      if (!text.getValue().starts_with((Twine(parent) + "|").str()))
        incoming.push_back(entry);
    }
  for (Value joined : joinedValues) {
    // The current semantic annotation schema identifies only producer task
    // and edge kind.  A two-output replica has two distinct RAW endpoints
    // from the same producer, so publishing one source-only annotation would
    // be ambiguous.  Still run the typed endpoint proof here; the actual
    // Taskflow graph remains the authoritative RAW/WAW dependency record, and
    // leave fresh consumers without a lossy semantic attribute until the
    // endpoint-indexed schema is available to the caller.
    SmallVector<Attribute> validatedEndpointAnnotations;
    if (failed(appendTypedCompletionJoinIncomingEdges(
            consumer, joined, validatedEndpointAnnotations, builder)))
      return failure();
  }
  if (hadPreviousAnnotations)
    consumer->setAttr(kSemanticIncomingEdgesAttr, builder.getArrayAttr(incoming));
  return success();
}

static LogicalResult materializeRadarMultiOutputReplica(
    func::FuncOp function, TaskflowTaskOp source, int64_t axis,
    int64_t replicas) {
  SmallVector<TaskflowTaskOp> consumers;
  if (failed(validateRadarMultiOutputProducer(function, source, axis, replicas,
                                              consumers)))
    return failure();

  constexpr int64_t counterExtents[2] = {64, 32};
  constexpr int64_t outputExtents[2] = {32, 64};
  constexpr int64_t outputCounterAxis[2] = {1, 0};
  const std::string sourceName = source.getTaskName().str();
  OpBuilder builder(source);
  SmallVector<SmallVector<Value>> states(source.getWillWrites().size());
  const int64_t splitExtent = counterExtents[axis];
  const int64_t splitOutputAxis = outputCounterAxis[axis];
  const int64_t staticOutputShape[2] = {32, 256};

  for (int64_t replicaId = 0; replicaId < replicas; ++replicaId) {
    const auto [lower, upper] =
        balancedShardBounds(splitExtent, replicaId, replicas);
    auto replica = cast<TaskflowTaskOp>(builder.clone(*source.getOperation()));
    replica->setAttr("task_name",
                     builder.getStringAttr(sourceName + ".replica." +
                                           std::to_string(replicaId)));
    replica->setAttr("amoeba.replica.parent_task",
                     builder.getStringAttr(sourceName));
    replica->setAttr("amoeba.replica.id", builder.getI64IntegerAttr(replicaId));
    replica->setAttr("amoeba.replica.count",
                     builder.getI64IntegerAttr(replicas));
    replica->setAttr("amoeba.replica.shard_axis",
                     builder.getI64IntegerAttr(axis));
    // The selected shard is a Neura/task counter axis.  Completion joins
    // expose the corresponding output dimension, which is permuted for this
    // Radar kernel ([counter-1,counter-0]); keep both authenticated axes
    // explicit instead of conflating the two in the typed-edge verifier.
    replica->setAttr("amoeba.replica.output_shard_axis",
                     builder.getI64IntegerAttr(splitOutputAxis));
    // Preserve the authenticated output-coordinate permutation explicitly on
    // each clone.  The validator above proves both stores use
    // [counter-1,counter-0]; this attribute records that proof for downstream
    // facts/acceptance consumers instead of leaving the mapping implicit in
    // the cloned Neura body.
    replica->setAttr(
        "amoeba.replica.output_counter_axes",
        builder.getDenseI64ArrayAttr(SmallVector<int64_t>{1, 0}));
    replica->setAttr("amoeba.replica.shard_lower",
                     builder.getI64IntegerAttr(lower));
    replica->setAttr("amoeba.replica.shard_upper",
                     builder.getI64IntegerAttr(upper));
    replica->setAttr("trip_count", builder.getI64IntegerAttr(
                                      (upper - lower) * counterExtents[1 - axis]));
    if (source->hasAttr("amoeba.selected_trip_count"))
      replica->setAttr("amoeba.selected_trip_count",
                       builder.getI64IntegerAttr((upper - lower) *
                                                 counterExtents[1 - axis]));
    replica->setAttr("amoeba.replica.total_trip_count",
                     builder.getI64IntegerAttr(counterExtents[0] *
                                               counterExtents[1]));
    replica->setAttr("amoeba.replica.shard_trip_count",
                     builder.getI64IntegerAttr((upper - lower) *
                                               counterExtents[1 - axis]));
    replica->removeAttr(kSemanticIncomingEdgesAttr);

    SmallVector<TaskflowCounterOp> clonedTaskCounters;
    for (Operation &operation : replica.getBody().front())
      if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
        clonedTaskCounters.push_back(counter);
    TaskflowCounterOp selectedTaskCounter;
    for (TaskflowCounterOp counter : clonedTaskCounters)
      if (auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
          id && id.getInt() == axis)
        selectedTaskCounter = counter;
    if (!selectedTaskCounter)
      return rejectRadarMultiOutput(source,
                                    "Radar replica lost selected Taskflow "
                                    "counter");
    OpBuilder counterBuilder(selectedTaskCounter);
    Value lowerValue =
        counterBuilder.create<arith::ConstantIndexOp>(source.getLoc(), lower);
    Value upperValue =
        counterBuilder.create<arith::ConstantIndexOp>(source.getLoc(), upper);
    selectedTaskCounter.getLowerBoundMutable().assign(lowerValue);
    selectedTaskCounter.getUpperBoundMutable().assign(upperValue);
    selectedTaskCounter.setCounterDynamismAttr(
        builder.getStringAttr("constant_bound"));

    neura::KernelOp clonedKernel;
    for (Operation &operation : replica.getBody().front())
      if (auto kernel = dyn_cast<neura::KernelOp>(&operation))
        clonedKernel = kernel;
    if (!clonedKernel)
      return rejectRadarMultiOutput(source, "Radar replica lost Neura kernel");
    SmallVector<neura::CounterOp> clonedNeuraCounters;
    clonedKernel.walk([&](neura::CounterOp counter) {
      clonedNeuraCounters.push_back(counter);
    });
    for (neura::CounterOp counter : clonedNeuraCounters) {
      auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
      if (!id || id.getInt() != axis)
        continue;
      OpBuilder kernelBuilder(counter);
      Value kernelLower =
          kernelBuilder.create<arith::ConstantIndexOp>(source.getLoc(), lower);
      Value kernelUpper =
          kernelBuilder.create<arith::ConstantIndexOp>(source.getLoc(), upper);
      counter.getLowerBoundMutable().assign(kernelLower);
      counter.getUpperBoundMutable().assign(kernelUpper);
      counter->setAttr("counter_dynamism",
                       builder.getStringAttr("constant_bound"));
      if (counter->hasAttr("lower_bound_value"))
        counter->setAttr("lower_bound_value", builder.getIndexAttr(lower));
      if (counter->hasAttr("upper_bound_value"))
        counter->setAttr("upper_bound_value", builder.getIndexAttr(upper));
      if (counter->hasAttr("step_value"))
        counter->setAttr("step_value", builder.getIndexAttr(1));
    }

    SmallVector<int64_t> regionLower = {0, 0};
    SmallVector<int64_t> regionUpper = {outputExtents[0], outputExtents[1]};
    regionLower[splitOutputAxis] = lower;
    regionUpper[splitOutputAxis] = upper;
    SmallVector<Attribute> lowerAttrs;
    SmallVector<Attribute> upperAttrs;
    for (size_t output = 0; output < source.getWillWrites().size(); ++output) {
      lowerAttrs.push_back(builder.getDenseI64ArrayAttr(regionLower));
      upperAttrs.push_back(builder.getDenseI64ArrayAttr(regionUpper));
      states[output].push_back(replica.getDoneWrites()[output]);
    }
    replica->setAttr(kTilingOutputRegionLowersAttr,
                     builder.getArrayAttr(lowerAttrs));
    replica->setAttr(kTilingOutputRegionUppersAttr,
                     builder.getArrayAttr(upperAttrs));
  }

  builder.setInsertionPointAfter(source);
  SmallVector<Value> joinedValues;
  for (size_t output = 0; output < source.getWillWrites().size(); ++output) {
    auto originalType = cast<MemRefType>(source.getWillWrites()[output].getType());
    auto staticType = MemRefType::get(
        ArrayRef<int64_t>(staticOutputShape), originalType.getElementType(),
        originalType.getLayout(), originalType.getMemorySpace());
    if (!memref::CastOp::areCastCompatible({source.getWillWrites()[output].getType()},
                                            {staticType}))
      return rejectRadarMultiOutput(source,
                                    "Radar multi-output output lacks a "
                                    "compatible static completion view");
    Value base = builder.create<memref::CastOp>(source.getLoc(), staticType,
                                                source.getWillWrites()[output]);
    SmallVector<Value> staticStates;
    for (Value state : states[output]) {
      if (!memref::CastOp::areCastCompatible({state.getType()}, {staticType}))
        return rejectRadarMultiOutput(source,
                                      "Radar multi-output state lacks a "
                                      "compatible static completion view");
      staticStates.push_back(
          builder.create<memref::CastOp>(source.getLoc(), staticType, state));
    }
    SmallVector<int64_t> parentLower = {0, 0};
    SmallVector<int64_t> parentUpper = {outputExtents[0], outputExtents[1]};
    auto join = builder.create<TaskflowJoinOp>(
        source.getLoc(), staticType, staticStates, base,
        builder.getI64IntegerAttr(splitOutputAxis),
        builder.getDenseI64ArrayAttr(parentLower),
        builder.getDenseI64ArrayAttr(parentUpper));
    join->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
    join->setAttr("amoeba.replica.completion_only", builder.getUnitAttr());
    // Taskflow consumers retain the dynamic state type carried by the source
    // result.  Keep the static join result for its verifier, then use a
    // compatible view when replacing the old done-write operand.
    Value replacement = builder.create<memref::CastOp>(
        source.getLoc(), source.getDoneWrites()[output].getType(),
        join.getJoined());
    joinedValues.push_back(replacement);
    source.getDoneWrites()[output].replaceAllUsesWith(replacement);
  }

  for (TaskflowTaskOp consumer : consumers) {
    SmallVector<Value> inputs;
    for (Value joined : joinedValues)
      for (OpOperand &use : joined.getUses())
        if (use.getOwner() == consumer.getOperation() &&
            !llvm::is_contained(inputs, joined))
          inputs.push_back(joined);
    if (inputs.empty() || failed(setRadarMultiOutputConsumerEdges(
                                  consumer, inputs, sourceName, builder)))
      return rejectRadarMultiOutput(
          source, "Radar multi-output completion edge annotation failed");
  }
  source.erase();
  function->setAttr("amoeba.replica.materialized_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.parent_task",
                    builder.getStringAttr(sourceName));
  function->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicas));
  function->setAttr("amoeba.replica.shard_axis", builder.getI64IntegerAttr(axis));
  function->setAttr("amoeba.replica.total_trip_count",
                    builder.getI64IntegerAttr(counterExtents[0] *
                                              counterExtents[1]));
  function->setAttr("amoeba.replica.shard_trip_count",
                    builder.getI64IntegerAttr((counterExtents[0] /
                                               (axis == 0 ? replicas : 1)) *
                                              (axis == 1 ? counterExtents[1] /
                                                               replicas
                                                         : counterExtents[1])));
  return success();
}


struct MaterializeJointTaskReplicasPass
    : PassWrapper<MaterializeJointTaskReplicasPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeJointTaskReplicasPass)

  MaterializeJointTaskReplicasPass() = default;
  MaterializeJointTaskReplicasPass(const MaterializeJointTaskReplicasPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "materialize-joint-task-replicas";
  }
  StringRef getDescription() const override {
    return "Clone a proved terminal task into balanced full DFG shards";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, neura::NeuraDialect,
                    TaskflowDialect>();
  }

  Option<std::string> functionName{*this, "function",
                                   llvm::cl::init("")};
  Option<std::string> taskName{*this, "task", llvm::cl::init("consumer")};
  Option<int64_t> replicaCount{*this, "replicas", llvm::cl::init(2)};
  Option<int64_t> shardAxis{*this, "axis", llvm::cl::init(0)};
  Option<bool> originalAmoebaFixedDecision{
      *this, kOriginalAmoebaFixedDecisionFlag,
      llvm::cl::desc("realize only the replica count and geometry already "
                     "recorded by the original AMOEBA scheduler"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selected)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp func = *selected;
    if (!originalAmoebaFixedDecision &&
        !isSupportedReplicaCount(replicaCount.getValue())) {
      func.emitError("replicas must be one of the proved factors {1, 2, 4, 8}");
      return signalPassFailure();
    }
    TaskflowTaskOp source;
    bool duplicate = false;
    func.walk([&](TaskflowTaskOp task) {
      if (task.getTaskName() != taskName.getValue())
        return;
      if (source)
        duplicate = true;
      source = task;
    });
    if (!source || duplicate) {
      func.emitError("replica task name must identify exactly one task");
      return signalPassFailure();
    }
    if (originalAmoebaFixedDecision) {
      if (!isOriginalAmoebaFixedDecisionTask(func, source)) {
        source.emitError("selected seed task is outside the pinned original "
                         "AMOEBA fixed-decision baseline scope");
        return signalPassFailure();
      }

      const std::string functionSymbol = func.getSymName().str();
      const std::string seedName = source.getTaskName().str();
      func::FuncOp canonicalFunction = cast<func::FuncOp>(func->clone());
      OwningOpRef<func::FuncOp> canonicalOwner(canonicalFunction);
      SmallVector<std::string> decisionTasks;
      std::set<std::string> seenDecisionNames;
      bool unsupportedDecision = false;
      func.walk([&](TaskflowTaskOp task) {
        auto active = task->getAttrOfType<IntegerAttr>("active_replicas");
        auto scheduler = task->getAttrOfType<DictionaryAttr>(
            "amoeba.task_scheduler_schedule_info");
        auto schedulerActive = integerField(scheduler, "active_replicas");
        bool activeAboveOne = active && active.getInt() > 1;
        bool schedulerAboveOne = schedulerActive && *schedulerActive > 1;
        if (!activeAboveOne && !schedulerAboveOne)
          return;
        if (!active || !schedulerActive ||
            active.getInt() != *schedulerActive) {
          task.emitError("original AMOEBA multi-replica decision has "
                         "unbound active_replicas fields");
          unsupportedDecision = true;
          return;
        }
        if (!isOriginalAmoebaFixedDecisionTask(func, task)) {
          task.emitError("original AMOEBA fixed-decision materialization found "
                         "an unsupported task with active_replicas > 1");
          unsupportedDecision = true;
          return;
        }
        std::string name = task.getTaskName().str();
        if (!seenDecisionNames.insert(name).second) {
          task.emitError("original AMOEBA fixed-decision task name is "
                         "ambiguous in the selected function");
          unsupportedDecision = true;
          return;
        }
        decisionTasks.push_back(std::move(name));
      });
      if (unsupportedDecision)
        return signalPassFailure();
      if (!llvm::is_contained(decisionTasks, seedName)) {
        source.emitError("selected seed task is not an original active replica "
                         "decision");
        return signalPassFailure();
      }
      if (decisionTasks.empty()) {
        func.emitError("original AMOEBA fixed-decision function has no active "
                       "multi-replica tasks");
        return signalPassFailure();
      }

      for (const std::string &decisionTaskName : decisionTasks) {
        FailureOr<func::FuncOp> current =
            selectTaskFunction(module, functionSymbol, error);
        if (failed(current)) {
          module.emitError() << error;
          return signalPassFailure();
        }
        TaskflowTaskOp currentSource;
        bool duplicateCurrentSource = false;
        current->walk([&](TaskflowTaskOp task) {
          if (task.getTaskName() != decisionTaskName)
            return;
          if (currentSource)
            duplicateCurrentSource = true;
          currentSource = task;
        });
        if (!currentSource || duplicateCurrentSource) {
          current->emitError() << "fixed-decision source task "
                               << decisionTaskName
                               << " disappeared or became ambiguous";
          return signalPassFailure();
        }
        if (failed(materializeOriginalAmoebaFixedDecision(
                module, *current, canonicalFunction, currentSource)))
          return signalPassFailure();
      }
      return;
    }
    if (replicaCount.getValue() == 1)
      return;
    bool inPlaceNeuraCandidate =
        source.getWillReads().size() == 1 &&
        source.getWillWrites().size() == 1 &&
        source.getOriginalReadMemrefs().size() == 1 &&
        source.getOriginalWriteMemrefs().size() == 1 &&
        source.getWillReads().front() == source.getWillWrites().front() &&
        source.getOriginalReadMemrefs().front() ==
            source.getOriginalWriteMemrefs().front();
    // A post-Neura in-place task may carry auxiliary read-only inputs in
    // addition to its output state.  Route that form through the same
    // authenticated coordinate proof as the single-state form.  The proof
    // checks the shared output state/root, every auxiliary root's disjointness,
    // exact Taskflow/Neura counter bounds, and all indexed output accesses;
    // this predicate only prevents the generic source-owned tiler from
    // silently applying its identity-coordinate assumption before those
    // checks run.  Specialized LLaMA/Radar producer paths above retain
    // precedence.
    bool hasNeuraKernelForInPlace = false;
    source.walk([&](neura::KernelOp) { hasNeuraKernelForInPlace = true; });
    bool multiReadInPlaceNeuraCandidate =
        hasNeuraKernelForInPlace && source.getWillReads().size() > 1 &&
        source.getWillWrites().size() == 1 &&
        source.getOriginalReadMemrefs().size() == source.getWillReads().size() &&
        source.getOriginalWriteMemrefs().size() == 1 &&
        llvm::count(source.getOriginalReadMemrefs(),
                    source.getOriginalWriteMemrefs().front()) == 1;
    inPlaceNeuraCandidate |= multiReadInPlaceNeuraCandidate;
    bool hasSupportedReplicaReadCompletion =
        source.getDoneReads().empty() ||
        (multiReadInPlaceNeuraCandidate &&
         source.getDoneReads().size() == 1);
    // Fresh Radar Task_2/Task_3 are the intentionally narrow two-output
    // post-Neura ABI.  Dispatch before the historical one-output guard so the
    // validator can authenticate both output coordinates and downstream
    // completion edges instead of silently routing this source into the
    // generic single-state implementation.
    if (isRadarFunction(func) && isRadarMultiOutputTask(source)) {
      if (failed(materializeRadarMultiOutputReplica(
              func, source, shardAxis.getValue(), replicaCount.getValue())))
        return signalPassFailure();
      return;
    }
    if (source->hasAttr("amoeba.replica.id") ||
        source->hasAttr("amoeba.tiling.parent_task") ||
        source.getWillWrites().size() != 1 ||
        source.getOriginalWriteMemrefs().size() != 1 ||
        source.getDoneWrites().size() != 1 ||
        !hasSupportedReplicaReadCompletion ||
        !source.getValueOutputs().empty()) {
      func.emitError("replica source must be one unsharded terminal "
                     "elementwise consumer with one output and a supported "
                     "read-completion shape");
      return signalPassFailure();
    }
    auto taskKind =
        source->getAttrOfType<StringAttr>("amoeba.semantic.task_kind");
    if (source.getTaskName() == "Task_0.k.0") {
      if (!isLlamaFunction(func)) {
        source.emitError("specialized Task_0.k.0 replica requires the pinned "
                         "LLaMA function");
        return signalPassFailure();
      }
      if (failed(materializeLlamaSourceProducer(func, source,
                                                shardAxis.getValue(),
                                                replicaCount.getValue())))
        return signalPassFailure();
      return;
    }
    if (taskKind && taskKind.getValue() == "gemm_producer") {
      if (failed(materializeGemmProducer(func, source, shardAxis.getValue(),
                                            replicaCount.getValue())))
        return signalPassFailure();
      return;
    }
    auto sourceOutputType = dyn_cast<MemRefType>(source.getWillWrites().front().getType());
    bool canonicalMatmulProducer =
        sourceOutputType && sourceOutputType.hasStaticShape() &&
        sourceOutputType.getRank() == 2 &&
        sourceOutputType.getShape()[0] == 512 &&
        (sourceOutputType.getShape()[1] == 256 ||
         sourceOutputType.getShape()[1] == 512) &&
        source.getWillReads().size() == 2 &&
        source.getWillWrites().front().getDefiningOp<memref::AllocaOp>();
    if ((isRadarFunction(func) &&
         (source.getTaskName() == "Task_0" ||
          source.getTaskName() == "Task_1")) ||
        (canonicalMatmulProducer && isLlamaFunction(func))) {
      LogicalResult result =
          isRadarFunction(func)
              ? materializeRadarProducer(func, source, shardAxis.getValue(),
                                         replicaCount.getValue())
              : materializeLlamaProducer(func, source, shardAxis.getValue(),
                                            replicaCount.getValue());
      if (failed(result))
        return signalPassFailure();
      return;
    }
    if (inPlaceNeuraCandidate) {
      FailureOr<ReplicaNeuraInPlacePlan> plan = analyzeReplicaNeuraInPlace(
          source, shardAxis.getValue(), replicaCount.getValue());
      if (failed(plan) || failed(materializeReplicaNeuraInPlace(func, *plan)))
        return signalPassFailure();
      return;
    }
    Value output = source.getWillWrites().front();
    bool matchingOriginalReads =
        source.getOriginalReadMemrefs().size() == source.getWillReads().size();
    if (matchingOriginalReads)
      for (auto [index, read] : llvm::enumerate(source.getWillReads()))
        matchingOriginalReads &=
            source.getOriginalReadMemrefs()[index] == read;
    // A post-Neura producer/consumer task deliberately carries a Taskflow
    // state in will_reads while original_read_memrefs names the storage root.
    // Let the source-owned M/N materializer prove that relationship and
    // convert its complete tile result transactionally.  The legacy generic
    // shell below remains available only for the historical equal-state form.
    if (!matchingOriginalReads ||
        source.getOriginalWriteMemrefs().front() !=
            source.getWillWrites().front()) {
      if (hasUnprovedGenericReplicaFeedback(source)) {
        source.emitError("generic post-Neura replica has unproved carried "
                         "feedback state");
        return signalPassFailure();
      }
      if (failed(materializeGenericNeuraReplica(
              func, source, shardAxis.getValue(), replicaCount.getValue())))
        return signalPassFailure();
      return;
    }
    if (!matchingOriginalReads ||
        llvm::is_contained(source.getWillReads(), output) ||
        llvm::is_contained(source.getOriginalReadMemrefs(), output)) {
      source.emitError("generic replica output or input memref aliases are "
                       "not proved disjoint");
      return signalPassFailure();
    }
    auto outputType = dyn_cast<MemRefType>(output.getType());
    // Equal-state tasks normally use the legacy static terminal shell below.
    // Dynamic caller storage and nonterminal completion values are different:
    // their source-owned M/N tiling proof must establish the exact transfer
    // shape and preserve the downstream completion edge.  Route only the
    // stateless equal-state form here; specialized producers and the
    // producer/consumer mismatch path above have already taken precedence,
    // while carried iter-args/reserve state remains fail-closed.
    bool hasNonReturnCompletionUse = llvm::any_of(
        source.getDoneWrites().front().getUsers(), [](Operation *user) {
          return !isa<func::ReturnOp>(user);
        });
    bool dynamicOrNonterminal =
        (outputType && !outputType.hasStaticShape()) ||
        hasNonReturnCompletionUse;
    bool hasNeuraKernel = false;
    source.walk([&](neura::KernelOp) { hasNeuraKernel = true; });
    if (hasNeuraKernel && dynamicOrNonterminal && matchingOriginalReads &&
        source.getOriginalWriteMemrefs().front() ==
            source.getWillWrites().front()) {
      if (hasUnprovedGenericReplicaFeedback(source)) {
        source.emitError("generic equal-state replica has unproved carried "
                         "feedback state");
        return signalPassFailure();
      }
      if (failed(materializeGenericNeuraReplica(
              func, source, shardAxis.getValue(), replicaCount.getValue())))
        return signalPassFailure();
      return;
    }
    auto outputArg = dyn_cast<BlockArgument>(output);
    bool noaliasArgument =
        outputArg && outputArg.getOwner() == &func.getBody().front() &&
        func.getArgAttr(outputArg.getArgNumber(), "amoeba.noalias");
    bool privateAllocation = output.getDefiningOp<memref::AllocOp>() &&
                             llvm::all_of(output.getUsers(), [&](Operation *user) {
                               return user == source.getOperation();
                             });
    if (!outputType || !outputType.hasStaticShape() ||
        shardAxis.getValue() < 0 || shardAxis.getValue() >= outputType.getRank() ||
        outputType.getShape()[shardAxis.getValue()] < 2 ||
        !isSupportedReplicaCount(replicaCount.getValue()) ||
        replicaCount.getValue() < 2 ||
        !hasBalancedShards(outputType.getShape()[shardAxis.getValue()],
                           replicaCount.getValue()) ||
        (!noaliasArgument && !privateAllocation)) {
      source.emitError("replica output must be a noalias static memref "
                       "argument or private allocation with an axis divisible "
                       "by factor 2, 4, or 8");
      return signalPassFailure();
    }
    if (!llvm::all_of(source.getDoneWrites().front().getUsers(),
                      [](Operation *user) { return isa<func::ReturnOp>(user); }) ||
        source.getDoneWrites().front().use_empty()) {
      source.emitError("replica source must be terminal in the function");
      return signalPassFailure();
    }
    Block &body = source.getBody().front();
    SmallVector<TaskflowCounterOp> counters;
    TaskflowHyperblockOp hyperblock;
    unsigned hyperblockCount = 0;
    bool unsupportedTopLevel = false;
    for (Operation &operation : body) {
      if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
        counters.push_back(counter);
      } else if (auto block = dyn_cast<TaskflowHyperblockOp>(&operation)) {
        hyperblock = block;
        ++hyperblockCount;
      } else if (!isa<arith::ConstantOp, TaskflowYieldOp>(&operation)) {
        unsupportedTopLevel = true;
      }
    }
    auto dimensions = outputType.getShape();
    bool completeCounters = counters.size() ==
                            static_cast<size_t>(outputType.getRank());
    if (completeCounters)
      for (auto [index, counter] : llvm::enumerate(counters))
        completeCounters &=
            (index == 0 ? !counter.getParentIndex()
                        : counter.getParentIndex() ==
                              counters[index - 1].getCounterIndex()) &&
            constantIndex(counter.getLowerBound()) == 0 &&
            constantIndex(counter.getUpperBound()) == dimensions[index] &&
            constantIndex(counter.getStep()) == 1;
    if (!completeCounters || unsupportedTopLevel || hyperblockCount != 1) {
      source.emitError("replica source needs one full unit-step static "
                       "counter domain matching its output");
      return signalPassFailure();
    }
    unsigned stores = 0;
    source.walk([&](memref::StoreOp) { ++stores; });
    bool directHyperblock = hyperblock.getIndices().size() == counters.size() &&
                            hyperblock.getBody().hasOneBlock();
    if (directHyperblock)
      for (auto [index, value] : llvm::enumerate(hyperblock.getIndices()))
        directHyperblock &= value == counters[index].getCounterIndex();
    if (stores != 1 || !directHyperblock) {
      source.emitError("replica source needs one direct elementwise store body");
      return signalPassFailure();
    }
    memref::StoreOp store;
    source.walk([&](memref::StoreOp op) { store = op; });
    Block &hyperBody = hyperblock.getBody().front();
    bool independentBody = store.getMemRef() ==
                               body.getArgument(source.getWillReads().size()) &&
                           store.getIndices().size() == counters.size();
    if (independentBody)
      for (auto [index, value] : llvm::enumerate(store.getIndices()))
        independentBody &= value == hyperBody.getArgument(index);
    for (Operation &operation : hyperBody.without_terminator()) {
      if (auto load = dyn_cast<memref::LoadOp>(&operation)) {
        auto argument = dyn_cast<BlockArgument>(load.getMemRef());
        independentBody &= argument && argument.getOwner() == &body &&
                           argument.getArgNumber() < source.getWillReads().size();
        independentBody &= load.getIndices().size() == counters.size();
        if (independentBody)
          for (auto [index, value] : llvm::enumerate(load.getIndices()))
            independentBody &= value == hyperBody.getArgument(index);
      } else if (!isa<memref::StoreOp>(&operation)) {
        independentBody &= operation.getNumRegions() == 0 &&
                           operation.getDialect()->getNamespace() == "arith";
      }
    }
    if (!independentBody) {
      source.emitError("replica store/load body is not an independent "
                       "elementwise DFG");
      return signalPassFailure();
    }
    int64_t totalTripCount = 1;
    for (int64_t dimension : dimensions)
      totalTripCount *= dimension;
    if (auto trip = source->getAttrOfType<IntegerAttr>("trip_count"))
      totalTripCount = trip.getInt();
    if (totalTripCount <= 0 || totalTripCount % replicaCount.getValue() != 0) {
      source.emitError("replica trip_count must divide evenly by the factor");
      return signalPassFailure();
    }
    const int64_t shardTripCount =
        totalTripCount / replicaCount.getValue();
    // The existing output-region contract is used only to prove disjoint
    // memory order. Input-region hints are removed, so producer fanout remains
    // tensor-wide and no semantic tile-local edge is introduced.
    OpBuilder builder(source);
    SmallVector<Value> states;
    const int64_t axis = shardAxis.getValue();
    for (int64_t index = 0; index < replicaCount.getValue(); ++index) {
      auto replica = cast<TaskflowTaskOp>(builder.clone(*source.getOperation()));
      const auto [lower, upper] =
          balancedShardBounds(dimensions[axis], index, replicaCount.getValue());
      replica->setAttr("task_name", builder.getStringAttr(
          (taskName.getValue() + ".replica." + std::to_string(index))));
      replica->setAttr("amoeba.replica.parent_task",
                       builder.getStringAttr(taskName.getValue()));
      replica->setAttr("amoeba.replica.id", builder.getI64IntegerAttr(index));
      replica->setAttr("amoeba.replica.count", builder.getI64IntegerAttr(replicaCount.getValue()));
      replica->setAttr("amoeba.replica.shard_axis",
                       builder.getI64IntegerAttr(axis));
      replica->setAttr("amoeba.replica.shard_lower",
                       builder.getI64IntegerAttr(lower));
      replica->setAttr("amoeba.replica.shard_upper",
                       builder.getI64IntegerAttr(upper));
      replica->setAttr("trip_count",
                       builder.getI64IntegerAttr(shardTripCount));
      if (source->getAttr("amoeba.selected_trip_count"))
        replica->setAttr("amoeba.selected_trip_count",
                         builder.getI64IntegerAttr(shardTripCount));
      replica->setAttr("amoeba.replica.total_trip_count",
                       builder.getI64IntegerAttr(totalTripCount));
      replica->setAttr("amoeba.replica.shard_trip_count",
                       builder.getI64IntegerAttr(shardTripCount));
      for (StringRef attr : {StringRef(kTilingInputRegionLowersAttr),
                             StringRef(kTilingInputRegionUppersAttr),
                             StringRef(kTilingInputRegionReasonsAttr),
                             StringRef("amoeba.semantic.incoming_edges")})
        replica->removeAttr(attr);
      SmallVector<int64_t> regionLower(dimensions.size(), 0);
      SmallVector<int64_t> regionUpper(dimensions.begin(), dimensions.end());
      regionLower[axis] = lower;
      regionUpper[axis] = upper;
      replica->setAttr(kTilingOutputRegionLowersAttr,
                       builder.getArrayAttr(
                           {builder.getDenseI64ArrayAttr(regionLower)}));
      replica->setAttr(kTilingOutputRegionUppersAttr,
                       builder.getArrayAttr(
                           {builder.getDenseI64ArrayAttr(regionUpper)}));
      SmallVector<TaskflowCounterOp> clonedCounters;
      for (Operation &operation : replica.getBody().front())
        if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
          clonedCounters.push_back(counter);
      TaskflowCounterOp selectedCounter = clonedCounters[axis];
      OpBuilder inBody(selectedCounter);
      Value lowerValue = inBody.create<arith::ConstantIndexOp>(source.getLoc(),
                                                                lower);
      Value upperValue = inBody.create<arith::ConstantIndexOp>(source.getLoc(),
                                                                upper);
      selectedCounter.getLowerBoundMutable().assign(lowerValue);
      selectedCounter.getUpperBoundMutable().assign(upperValue);
      states.push_back(replica.getDoneWrites().front());
    }
    auto join = builder.create<TaskflowJoinOp>(
        source.getLoc(), outputType, states, output,
        builder.getI64IntegerAttr(axis),
        builder.getDenseI64ArrayAttr(
            SmallVector<int64_t>(dimensions.size(), 0)),
        builder.getDenseI64ArrayAttr(dimensions));
    join->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
    join->setAttr("amoeba.replica.completion_only", builder.getUnitAttr());
    source.getDoneWrites().front().replaceAllUsesWith(join.getJoined());
    source.erase();
    func->setAttr("amoeba.replica.materialized_task",
                  builder.getStringAttr(taskName.getValue()));
    func->setAttr("amoeba.replica.count",
                  builder.getI64IntegerAttr(replicaCount.getValue()));
    func->setAttr("amoeba.replica.shard_axis", builder.getI64IntegerAttr(axis));
    func->setAttr("amoeba.replica.total_trip_count",
                  builder.getI64IntegerAttr(totalTripCount));
    func->setAttr("amoeba.replica.shard_trip_count",
                  builder.getI64IntegerAttr(shardTripCount));
  }
};

} // namespace

namespace mlir::amoeba::neura {

std::unique_ptr<Pass> createMaterializeJointTaskReplicasPass() {
  return std::make_unique<MaterializeJointTaskReplicasPass>();
}

} // namespace mlir::amoeba::neura
