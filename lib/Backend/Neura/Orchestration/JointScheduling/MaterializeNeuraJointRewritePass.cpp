//===- MaterializeNeuraJointRewritePass.cpp --------------------*- C++ -*-===//
//
// Post-conversion ORBIT rewrite slice.
//
// The Taskflow-to-Neura conversion intentionally retains taskflow.task and
// its outer counter chain.  This pass consumes that mixed IR directly.  It
// clones complete task/kernel instances, narrows the selected static M/N
// counter in both the retained Taskflow shell and the neura.kernel body, and
// joins the resulting completion states.  The mapper consequently still sees
// one neura.kernel per output task.
//
// K reduction remains rejected until its state contract can be represented
// explicitly.  Producer-consumer fusion is implemented below as a retained
// mode and a separately selectable forwarded mode with strict storage and
// index proofs.  This pass never silently turns a task into a Python-side
// rewrite or a fission operation.
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskGraphRewriteLegality.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

constexpr StringLiteral kParentTaskAttr = "amoeba.neura.tiling.parent_task";
constexpr StringLiteral kAxisAttr = "amoeba.neura.tiling.axis";
constexpr StringLiteral kFactorAttr = "amoeba.neura.tiling.factor";
constexpr StringLiteral kPartIndexAttr = "amoeba.neura.tiling.part_index";
constexpr StringLiteral kOriginalRangeAttr =
    "amoeba.neura.tiling.original_range";
constexpr StringLiteral kDerivedRangeAttr = "amoeba.neura.tiling.derived_range";
constexpr StringLiteral kRewriteAttr = "amoeba.neura.joint_rewrite";
constexpr StringLiteral kSemanticIncomingEdgesAttr =
    "amoeba.semantic.incoming_edges";
constexpr StringLiteral kTilingOutputRegionLowersAttr =
    "amoeba.tiling.output_region_lowers";
constexpr StringLiteral kTilingOutputRegionUppersAttr =
    "amoeba.tiling.output_region_uppers";

static LogicalResult reject(Operation *operation, const Twine &message) {
  operation->emitError(message);
  return failure();
}

namespace joint_scheduling = mlir::amoeba::neura::joint_scheduling;

struct SemanticIncomingAnnotation {
  std::string producer;
  std::string role;
  std::string scope;
  std::string kind;
};

struct SemanticIncomingRecord {
  std::string consumer;
  SemanticIncomingAnnotation annotation;
};

struct SemanticIncomingSnapshot {
  std::set<std::string> taskNames;
  std::set<std::string> tasksWithAttribute;
  std::vector<SemanticIncomingRecord> records;
};

static bool parseSemanticIncomingEdge(StringRef encoded, StringRef &producer,
                                      StringRef &role, StringRef &scope,
                                      StringRef &kind) {
  SmallVector<StringRef, 4> fields;
  encoded.split(fields, '|');
  if (fields.size() != 3 && fields.size() != 4)
    return false;
  if (llvm::any_of(fields, [](StringRef field) { return field.empty(); }))
    return false;
  producer = fields[0];
  role = fields[1];
  scope = fields[2];
  kind = fields.size() == 4 ? fields[3] : StringRef();
  return kind.empty() || kind == "raw" || kind == "war" || kind == "waw" ||
         kind == "value" || kind == "control";
}

static bool sameTask(TaskflowTaskOp lhs, TaskflowTaskOp rhs) {
  return lhs && rhs && lhs.getOperation() == rhs.getOperation();
}

static StringRef taskNameFromOperation(TaskflowTaskOp task) {
  auto name = task.getOperation()->getAttrOfType<StringAttr>("task_name");
  return name ? name.getValue() : StringRef();
}

// Read and validate the semantic contract before a rewrite mutates any SSA.
// The typed graph is the authority for endpoint and scope identity: an
// annotation is never allowed to manufacture a dependency or to hide an
// ambiguous pair of Taskflow edges.
static LogicalResult
collectSemanticIncomingSnapshot(func::FuncOp function,
                                SemanticIncomingSnapshot &snapshot) {
  std::string error;
  joint_scheduling::TaskEdgeGraphOptions options;
  FailureOr<joint_scheduling::TaskEdgeGraph> graph =
      joint_scheduling::buildTaskEdgeGraph(function, options, error);
  if (failed(graph))
    return reject(function, "cannot snapshot semantic incoming edges: " +
                               Twine(error));

  for (TaskflowTaskOp task : graph->getTasks())
    snapshot.taskNames.insert(task.getTaskName().str());

  for (TaskflowTaskOp consumer : graph->getTasks()) {
    Attribute attribute = consumer->getAttr(kSemanticIncomingEdgesAttr);
    if (!attribute)
      continue;
    snapshot.tasksWithAttribute.insert(consumer.getTaskName().str());
    auto annotations = dyn_cast<ArrayAttr>(attribute);
    if (!annotations)
      return reject(consumer,
                    "amoeba.semantic.incoming_edges must be an array "
                    "attribute");

    std::set<std::tuple<std::string, std::string, std::string, std::string>> seen;
    for (Attribute entry : annotations) {
      auto string = dyn_cast<StringAttr>(entry);
      if (!string)
        return reject(consumer,
                      "amoeba.semantic.incoming_edges entries must be "
                      "strings");
      StringRef producer;
      StringRef role;
      StringRef scope;
      StringRef kind;
      if (!parseSemanticIncomingEdge(string.getValue(), producer, role,
                                     scope, kind))
        return reject(
            consumer,
            "amoeba.semantic.incoming_edges entries must have the form "
            "source|role|scope or source|role|scope|kind with non-empty fields");
      if (!seen.insert({producer.str(), role.str(), scope.str(), kind.str()}).second)
        return reject(consumer,
                      "duplicate semantic incoming edge annotation: " +
                          Twine(string.getValue()));

      size_t matchingEdges = 0;
      const joint_scheduling::TaskEdge *matchingEdge = nullptr;
      for (const joint_scheduling::TaskEdge &edge : graph->getEdges()) {
        if (edge.origin != joint_scheduling::TaskEdgeOrigin::Taskflow ||
            !sameTask(edge.consumer, consumer) ||
            taskNameFromOperation(edge.producer) != producer ||
            (!kind.empty() && kind != joint_scheduling::stringifyTaskEdgeKind(edge.kind)))
          continue;
        ++matchingEdges;
        matchingEdge = &edge;
      }
      if (matchingEdges == 0)
        return reject(consumer,
                      "semantic incoming edge does not match an actual "
                      "Taskflow edge: " +
                          Twine(string.getValue()));
      if (matchingEdges != 1)
        return reject(consumer,
                      "semantic incoming edge source is ambiguous for "
                      "actual Taskflow edges: " +
                          Twine(string.getValue()));
      if (scope != joint_scheduling::stringifyTaskEdgeScope(
                     matchingEdge->scope))
        return reject(consumer,
                      "semantic incoming edge scope disagrees with the "
                      "actual Taskflow edge: " +
                          Twine(string.getValue()));
      snapshot.records.push_back(
          {consumer.getTaskName().str(),
           {producer.str(), role.str(), scope.str(),
            joint_scheduling::stringifyTaskEdgeKind(matchingEdge->kind).str()}});
    }

    // A semantic attribute is a complete annotation of the Taskflow input
    // edges.  This catches the stale/partial arrays that otherwise survive a
    // clone and only fail much later in facts extraction.
    for (const joint_scheduling::TaskEdge &edge : graph->getEdges()) {
      if (edge.origin != joint_scheduling::TaskEdgeOrigin::Taskflow ||
          !sameTask(edge.consumer, consumer))
        continue;
      size_t matchingAnnotations = 0;
      for (const SemanticIncomingRecord &record : snapshot.records)
        if (record.consumer == consumer.getTaskName() &&
            record.annotation.producer == taskNameFromOperation(edge.producer) &&
            record.annotation.kind == joint_scheduling::stringifyTaskEdgeKind(edge.kind))
          ++matchingAnnotations;
      if (matchingAnnotations != 1)
        return reject(
            consumer,
            "missing semantic incoming edge annotation for actual "
            "Taskflow edge from " +
                Twine(taskNameFromOperation(edge.producer)));
    }
  }
  return success();
}

static const std::vector<std::string> *
findSemanticOrigins(
    StringRef taskName, const SemanticIncomingSnapshot &snapshot,
    const std::map<std::string, std::vector<std::string>> &origins,
    std::vector<std::string> &identityOrigin) {
  auto found = origins.find(taskName.str());
  if (found != origins.end())
    return &found->second;
  if (snapshot.taskNames.find(taskName.str()) == snapshot.taskNames.end())
    return nullptr;
  identityOrigin.clear();
  identityOrigin.push_back(taskName.str());
  return &identityOrigin;
}

static bool hasProvenTileLocalTransfer(
    const joint_scheduling::TaskEdge &edge) {
  if (edge.scope != joint_scheduling::TaskEdgeScope::TileLocal ||
      edge.transfer_region_lower.empty() ||
      edge.transfer_region_lower.size() != edge.transfer_region_upper.size())
    return false;
  for (auto [lower, upper] :
       llvm::zip(edge.transfer_region_lower, edge.transfer_region_upper))
    if (lower >= upper)
      return false;
  // TaskEdgeContract emits transfer regions only after proving the
  // producer output and consumer input regions have the same original
  // memref root and a non-empty intersection.  Requiring that payload here
  // permits the sound tensor-wide -> tile-local narrowing caused by a legal
  // M/N tile while keeping arbitrary scope relabeling fail-closed.
  return true;
}

// Rebuild semantic incoming edges after clones or fusion have changed task
// names.  `origins` records only the newly-created tasks; all other tasks are
// identity mappings.  External edges are retained when an old endpoint is in
// the corresponding lineage.  An edge internal to a fused task has no final
// Taskflow edge and therefore disappears naturally.  Ambiguous role/scope
// mappings fail closed instead of choosing a heuristic label.
static LogicalResult remapSemanticIncomingEdges(
    func::FuncOp function, const SemanticIncomingSnapshot &snapshot,
    const std::map<std::string, std::vector<std::string>> &origins) {
  std::string error;
  joint_scheduling::TaskEdgeGraphOptions options;
  FailureOr<joint_scheduling::TaskEdgeGraph> graph =
      joint_scheduling::buildTaskEdgeGraph(function, options, error);
  if (failed(graph))
    return reject(function, "cannot rebuild semantic incoming edges: " +
                               Twine(error));

  std::map<std::string, std::vector<SemanticIncomingAnnotation>> rebuilt;
  std::map<std::string, std::vector<std::string>> taskOrigins;
  for (TaskflowTaskOp task : graph->getTasks()) {
    std::vector<std::string> identity;
    const std::vector<std::string> *lineage =
        findSemanticOrigins(task.getTaskName(), snapshot, origins, identity);
    if (!lineage)
      return reject(task,
                    "rewritten task has no semantic origin: " +
                        Twine(task.getTaskName()));
    taskOrigins.emplace(task.getTaskName().str(), *lineage);
    for (const std::string &oldTask : *lineage)
      if (snapshot.tasksWithAttribute.find(oldTask) !=
          snapshot.tasksWithAttribute.end())
        rebuilt[task.getTaskName().str()];
  }

  std::map<std::pair<std::string, std::string>, size_t> edgePairCounts;
  for (const joint_scheduling::TaskEdge &edge : graph->getEdges())
    if (edge.origin == joint_scheduling::TaskEdgeOrigin::Taskflow)
      ++edgePairCounts[{taskNameFromOperation(edge.producer).str(),
                        taskNameFromOperation(edge.consumer).str()}];

  for (const joint_scheduling::TaskEdge &edge : graph->getEdges()) {
    if (edge.origin != joint_scheduling::TaskEdgeOrigin::Taskflow)
      continue;
    // Tasks without the optional semantic contract retain their ordinary
    // typed dependencies; there is no role/scope fact to remap for them.
    if (rebuilt.find(taskNameFromOperation(edge.consumer).str()) ==
        rebuilt.end())
      continue;
    auto producerLineage =
        taskOrigins.find(taskNameFromOperation(edge.producer).str());
    auto consumerLineage =
        taskOrigins.find(taskNameFromOperation(edge.consumer).str());
    if (producerLineage == taskOrigins.end() ||
        consumerLineage == taskOrigins.end())
      return reject(edge.consumer,
                    "semantic edge endpoint has no rewrite lineage");

    std::vector<const SemanticIncomingRecord *> matches;
    for (const SemanticIncomingRecord &record : snapshot.records) {
      if (!llvm::is_contained(consumerLineage->second, record.consumer))
        continue;
      if (!llvm::is_contained(producerLineage->second,
                              record.annotation.producer))
        continue;
      if (record.annotation.kind != joint_scheduling::stringifyTaskEdgeKind(edge.kind))
        continue;
      matches.push_back(&record);
    }
    if (matches.empty())
      return reject(edge.consumer,
                    "cannot remap semantic incoming edge from " +
                        Twine(taskNameFromOperation(edge.producer)) + " to " +
                        Twine(taskNameFromOperation(edge.consumer)));

    std::optional<StringRef> role;
    std::optional<StringRef> scope;
    for (const SemanticIncomingRecord *match : matches) {
      if (role && *role != match->annotation.role)
        return reject(edge.consumer,
                      "conflicting semantic roles after rewrite for edge " +
                          Twine(taskNameFromOperation(edge.producer)) +
                              " -> " +
                          Twine(taskNameFromOperation(edge.consumer)));
      if (scope && *scope != match->annotation.scope)
        return reject(edge.consumer,
                      "conflicting semantic scopes after rewrite for edge " +
                          Twine(taskNameFromOperation(edge.producer)) +
                              " -> " +
                          Twine(taskNameFromOperation(edge.consumer)));
      role = match->annotation.role;
      scope = match->annotation.scope;
    }
    StringRef actualScope =
        joint_scheduling::stringifyTaskEdgeScope(edge.scope);
    bool exactScope = scope && *scope == actualScope;
    bool provenTileNarrowing =
        scope && *scope == "tensor_wide" && actualScope == "tile_local" &&
        hasProvenTileLocalTransfer(edge);
    if (!exactScope && !provenTileNarrowing)
      return reject(edge.consumer,
                    "semantic scope disagrees with the rewritten Taskflow "
                    "edge");
    // A typed tile-local transfer is a sound narrowing of the old
    // tensor-wide contract.  Persist the actual post-rewrite scope so the
    // final typed-contract validation sees the derived scope rather than the
    // stale pre-rewrite spelling.
    std::string remappedScope =
        provenTileNarrowing ? actualScope.str() : scope->str();
    size_t sameKindCount = 0;
    for (const joint_scheduling::TaskEdge &candidate : graph->getEdges())
      if (candidate.origin == joint_scheduling::TaskEdgeOrigin::Taskflow &&
          sameTask(candidate.producer, edge.producer) &&
          sameTask(candidate.consumer, edge.consumer) &&
          candidate.kind == edge.kind)
        ++sameKindCount;
    if (sameKindCount != 1)
      return reject(edge.consumer,
                    "rewritten Taskflow edge kind is ambiguous for semantic "
                    "annotation");
    std::string remappedKind =
        edgePairCounts[{taskNameFromOperation(edge.producer).str(),
                        taskNameFromOperation(edge.consumer).str()}] == 1
            ? std::string()
            : joint_scheduling::stringifyTaskEdgeKind(edge.kind).str();

    auto &annotations = rebuilt[taskNameFromOperation(edge.consumer).str()];
    bool duplicate = false;
    for (const SemanticIncomingAnnotation &annotation : annotations)
      if (annotation.producer == taskNameFromOperation(edge.producer) &&
          annotation.role == *role && annotation.scope == remappedScope &&
          annotation.kind == remappedKind)
        duplicate = true;
    if (!duplicate)
      annotations.push_back({taskNameFromOperation(edge.producer).str(),
                             role->str(),
                             remappedScope, remappedKind});
  }

  OpBuilder builder(function.getContext());
  for (TaskflowTaskOp task : graph->getTasks()) {
    auto rebuiltIt = rebuilt.find(task.getTaskName().str());
    if (rebuiltIt == rebuilt.end())
      continue;
    llvm::sort(rebuiltIt->second, [](const SemanticIncomingAnnotation &lhs,
                                    const SemanticIncomingAnnotation &rhs) {
      return std::tie(lhs.producer, lhs.role, lhs.scope, lhs.kind) <
             std::tie(rhs.producer, rhs.role, rhs.scope, rhs.kind);
    });
    SmallVector<Attribute> attributes;
    for (const SemanticIncomingAnnotation &annotation : rebuiltIt->second)
      attributes.push_back(builder.getStringAttr(
          (Twine(annotation.producer) + "|" + annotation.role + "|" +
           annotation.scope +
           (annotation.kind.empty() ? std::string() : "|" + annotation.kind))
              .str()));
    task->setAttr(kSemanticIncomingEdgesAttr,
                  builder.getArrayAttr(attributes));
  }

  // Run the same typed contract over the final graph, including empty arrays
  // left after internal fusion.  This is the final guard against stale source
  // names copied by createPart/createPostNeura*.
  SemanticIncomingSnapshot verified;
  return collectSemanticIncomingSnapshot(function, verified);
}

static std::optional<int64_t> constantIndex(Value value) {
  // Converted Neura counters may carry a compile-time bound only in their
  // metadata (for example, `lower_bound_value`) and consequently expose an
  // empty optional operand.  Value::getDefiningOp() asserts on such an empty
  // Value in this LLVM build; fail closed and let the caller inspect the
  // metadata instead of crashing the optimizer.
  if (!value)
    return std::nullopt;
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return constant.value();
  return std::nullopt;
}

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

// Dataflow lowering folds loop-carried initial values into operation
// attributes as `%iter_arg_initN`.  Keep this parser separate from the kernel
// input parser: the two namespaces have different rebasing rules when two
// kernels are concatenated.
static std::optional<unsigned>
parseKernelIterArgReference(Attribute attribute) {
  if (!attribute)
    return std::nullopt;
  auto text = dyn_cast<StringAttr>(attribute);
  if (!text)
    return std::nullopt;
  StringRef value = text.getValue();
  if (!value.consume_front("%iter_arg_init") || value.empty())
    return std::nullopt;
  unsigned index = 0;
  if (value.getAsInteger(10, index))
    return std::nullopt;
  return index;
}

// Return whether an attribute is in one of the folded Neura value
// namespaces.  A malformed spelling in either namespace is rejected by the
// stateful fusion path instead of being copied with an accidental source
// index.
static bool isFoldedKernelReference(Attribute attribute, StringRef prefix) {
  auto text = dyn_cast_or_null<StringAttr>(attribute);
  return text && text.getValue().starts_with(prefix);
}

// Resolve a compile-time index through the Taskflow value-input ABI.  The
// Taskflow-to-Neura conversion used by the application fixtures folds some
// counter bounds into attributes and keeps others as SSA operands.  Both
// encodings are accepted only when they resolve to an actual literal.
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

  // Application frontends sometimes materialize a static bound as an
  // arith.addi of a task value input and a literal (Radar uses 64 + (-2)).
  // Fold only when both operands are already proven constants by the paths
  // above. In particular, this must not turn a symbolic block argument into
  // a static bound merely because it participates in an addi.
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = compileTimeIndex(add.getLhs(), task, kernel);
    auto rhs = compileTimeIndex(add.getRhs(), task, kernel);
    if (!lhs || !rhs)
      return std::nullopt;
    if ((*rhs > 0 && *lhs > std::numeric_limits<int64_t>::max() - *rhs) ||
        (*rhs < 0 && *lhs < std::numeric_limits<int64_t>::min() - *rhs))
      return std::nullopt;
    int64_t result = *lhs + *rhs;

    // MLIR integer arithmetic is width-specific. Do not accept a wrapped iN
    // result as a static loop bound; that would make a malformed or
    // overflowing source look like a legal positive interval. Index values
    // are checked against int64_t above, which is the representation used by
    // this pass for compile-time bounds.
    if (auto integer = dyn_cast<IntegerType>(add.getType())) {
      unsigned width = integer.getWidth();
      if (width == 0 || width > 64)
        return std::nullopt;
      if (width < 64) {
        int64_t maximum = (int64_t(1) << (width - 1)) - 1;
        int64_t minimum = -maximum - 1;
        if (result < minimum || result > maximum)
          return std::nullopt;
      }
    } else if (!isa<IndexType>(add.getType())) {
      return std::nullopt;
    }
    return result;
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
  }
  return std::nullopt;
}

static std::optional<int64_t>
compileTimeCounterBound(neura::CounterOp counter, StringRef attributeName,
                        Value operand, TaskflowTaskOp task,
                        neura::KernelOp kernel = neura::KernelOp()) {
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
    } else {
      auto input = parseKernelInputReference(attribute);
      if (!input || !kernel || *input >= kernel.getInputs().size())
        return std::nullopt;
      attributeValue = compileTimeIndex(kernel.getInputs()[*input], task, kernel);
      if (!attributeValue)
        return std::nullopt;
    }
  }
  if (operandValue && attributeValue && *operandValue != *attributeValue)
    return std::nullopt;
  return attributeValue ? attributeValue : operandValue;
}

static std::optional<SmallVector<int64_t>>
callerMemrefShape(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || !argument.getOwner())
    return std::nullopt;
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front())
    return std::nullopt;
  Attribute attribute =
      function.getArgAttr(argument.getArgNumber(),
                          "amoeba.logical_transfer_shape");
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

static LogicalResult validateCallerShape(Value value, MemRefType type,
                                         SmallVectorImpl<int64_t> &shape) {
  auto contract = callerMemrefShape(value);
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

static bool isPrivateStorageRoot(Value value) {
  // Forwarding removes the only memory writes that make an intermediate
  // observable.  A function argument may alias caller-visible storage even
  // when no other SSA use is present, so require a private allocation root.
  Operation *definition = value.getDefiningOp();
  return definition && isa<memref::AllocOp, memref::AllocaOp>(definition);
}

static bool provesDistinctStorage(Value lhs, Value rhs) {
  // Use the shared legality contract.  The old local copy required an
  // explicit noalias attribute even when one root was a fresh alloc/alloca,
  // which disagreed with graph-facts and the other rewrite passes.
  return mlir::amoeba::neura::joint_scheduling::
      provesDistinctTaskStorage(lhs, rhs);
}

static std::optional<int64_t> counterId(Operation *operation) {
  if (auto attr = operation->getAttrOfType<IntegerAttr>("counter_id"))
    return attr.getInt();
  return std::nullopt;
}

struct NeuraTilePlan {
  TaskflowTaskOp task;
  neura::KernelOp kernel;
  SmallVector<TaskflowCounterOp> taskCounters;
  SmallVector<neura::CounterOp> kernelCounters;
  SmallVector<int64_t> taskCounterLowers;
  SmallVector<int64_t> taskCounterUppers;
  SmallVector<int64_t> kernelCounterLowers;
  SmallVector<int64_t> kernelCounterUppers;
  int64_t axis = -1;
  int64_t factor = 0;
  int64_t lower = 0;
  int64_t upper = 0;
  // Dynamic caller memrefs may participate only through an exact shape
  // contract.  Tiles use a static view of that proven shape so the existing
  // completion-join verifier can continue to enforce disjoint coverage.
  SmallVector<int64_t> outputShape;
  bool outputNeedsStaticView = false;
  // A narrow exception for an in-place rank-1 task. Such a task is legal only
  // when every read and write names the same output root at the selected
  // counter index; ordinary input/output aliasing remains rejected below.
  bool inPlaceSelectedIndex = false;
};

static bool forwardsCounterIndex(Value value, Value expected,
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
  return forwardsCounterIndex(definition->getOperand(0), expected, depth + 1);
}

static std::optional<unsigned>
kernelInputIndex(neura::KernelOp kernel, TaskflowTaskOp task, Value root) {
  if (!kernel.getBody().hasOneBlock() || !task.getBody().hasOneBlock())
    return std::nullopt;
  Block &taskBody = task.getBody().front();
  for (auto [index, input] : llvm::enumerate(kernel.getInputs())) {
    if (input == root)
      return static_cast<unsigned>(index);
    // Taskflow-to-Neura keeps kernel inputs as Taskflow region arguments.
    // Resolve that ABI edge back to the task operand before comparing it with
    // the original storage root. No other forwarding form is accepted here.
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
accessInputIndex(Operation *operation, neura::KernelOp kernel,
                 StringRef foldedBaseAttribute, Value base) {
  if (auto input = parseKernelInputReference(
          operation->getAttr(foldedBaseAttribute)))
    return input;
  if (!base || !kernel.getBody().hasOneBlock())
    return std::nullopt;
  Block &body = kernel.getBody().front();
  for (auto [index, argument] : llvm::enumerate(body.getArguments()))
    if (base == argument)
      return static_cast<unsigned>(index);
  return std::nullopt;
}

static bool mentionsKernelInput(Operation *operation, unsigned inputIndex) {
  for (StringRef attribute : {StringRef("lhs_value"),
                              StringRef("rhs_value")})
    if (auto input = parseKernelInputReference(operation->getAttr(attribute));
        input && *input == inputIndex)
      return true;
  return false;
}

static bool forwardsKernelArgument(Value value, BlockArgument argument,
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
  return forwardsKernelArgument(definition->getOperand(0), argument,
                                depth + 1);
}

static LogicalResult proveInPlaceSelectedRegion(NeuraTilePlan &plan,
                                               Value outputRoot) {
  auto outputType = dyn_cast<MemRefType>(outputRoot.getType());
  if (!outputType || outputType.getRank() < 1 || outputType.getRank() > 2 ||
      plan.axis < 0 || plan.axis >= outputType.getRank())
    return reject(plan.task,
                  "in-place tiling supports only rank-1/rank-2 output and a "
                  "selected output axis");
  unsigned outputRank = static_cast<unsigned>(outputType.getRank());
  if (plan.taskCounters.size() < outputRank ||
      plan.kernelCounters.size() < outputRank)
    return reject(plan.task,
                  "in-place tiling requires one counter per output dimension");

  auto outputInput = kernelInputIndex(plan.kernel, plan.task,
                                      plan.task.getWillWrites().front());
  if (!outputInput || *outputInput >= plan.kernel.getInputs().size())
    return reject(plan.task,
                  "in-place tiling cannot resolve the output kernel input");

  // A second kernel input carrying the same root would make an access path
  // indistinguishable from the selected output path.  Reject it rather than
  // assuming that the two input positions are semantically interchangeable.
  unsigned outputInputCount = 0;
  Block &taskBody = plan.task.getBody().front();
  for (auto [index, input] : llvm::enumerate(plan.kernel.getInputs())) {
    bool namesOutput = input == plan.task.getWillWrites().front();
    if (auto argument = dyn_cast<BlockArgument>(input))
      if (argument.getOwner() == &taskBody &&
          argument.getArgNumber() < plan.task->getNumOperands())
        namesOutput |= plan.task->getOperand(argument.getArgNumber()) ==
                       plan.task.getWillWrites().front();
    if (namesOutput) {
      ++outputInputCount;
      if (index != *outputInput)
        return reject(plan.task,
                      "in-place tiling rejects duplicate output kernel inputs");
    }
  }
  if (outputInputCount != 1)
    return reject(plan.task,
                  "in-place tiling requires one output kernel input");

  Block &kernelBody = plan.kernel.getBody().front();
  if (*outputInput >= kernelBody.getNumArguments())
    return reject(plan.task,
                  "in-place tiling cannot resolve the output kernel argument");
  BlockArgument outputArgument = kernelBody.getArgument(*outputInput);

  auto checkIndices = [&](Operation *operation, ValueRange indices) {
    if (indices.size() != outputRank) {
      if (outputRank == 1)
        return reject(
            plan.task,
            isa<memref::LoadOp, neura::LoadIndexedOp>(operation)
                ? "rank-1 in-place tiling requires every load to read the "
                  "output root at the selected counter index"
                : "rank-1 in-place tiling requires every store to write the "
                  "output root at the selected counter index");
      return reject(plan.task,
                    "in-place tiling requires every output access to use "
                    "the complete output rank");
    }
    for (auto [dimension, index] : llvm::enumerate(indices)) {
      if (dimension >= plan.kernelCounters.size() ||
          !forwardsCounterIndex(
              index, plan.kernelCounters[dimension].getCurrentIndex())) {
        if (outputRank == 1)
          return reject(
              plan.task,
              isa<memref::LoadOp, neura::LoadIndexedOp>(operation)
                  ? "rank-1 in-place tiling requires every load to read the "
                    "output root at the selected counter index"
                  : "rank-1 in-place tiling requires every store to write the "
                    "output root at the selected counter index");
        return reject(plan.task,
                      "in-place tiling requires every output access to use "
                      "the matching counter indices");
      }
    }
    return success();
  };

  bool sawLoad = false;
  bool sawStore = false;
  LogicalResult result = success();
  plan.kernel.walk([&](Operation *operation) {
    if (failed(result))
      return;
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      auto input = accessInputIndex(operation, plan.kernel, "lhs_value",
                                    load.getMemRef());
      if (!input && outputRank == 2) {
        result = reject(plan.task,
                        "in-place tiling cannot prove a rank-2 load storage "
                        "root");
        return;
      }
      if (!input || *input == *outputInput) {
        if (failed(checkIndices(operation, load.getIndices())))
          result = failure();
        else
          sawLoad = true;
      } else if (outputRank == 1) {
        result = reject(
            plan.task,
            "rank-1 in-place tiling requires every load to read the output "
            "root at the selected counter index");
      }
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      auto input = accessInputIndex(operation, plan.kernel, "rhs_value",
                                    store.getMemRef());
      if (!input && outputRank == 2) {
        result = reject(plan.task,
                        "in-place tiling cannot prove a rank-2 store storage "
                        "root");
        return;
      }
      if (!input || *input == *outputInput) {
        if (failed(checkIndices(operation, store.getIndices())))
          result = failure();
        else
          sawStore = true;
      } else if (outputRank == 1) {
        result = reject(
            plan.task,
            "rank-1 in-place tiling requires every store to write the output "
            "root at the selected counter index");
      }
      return;
    }
    if (auto load = dyn_cast<neura::LoadIndexedOp>(operation)) {
      auto input = accessInputIndex(operation, plan.kernel, "lhs_value",
                                    load.getBase());
      if (!input && outputRank == 2) {
        result = reject(plan.task,
                        "in-place tiling cannot prove a rank-2 load storage "
                        "root");
        return;
      }
      if (!input || *input == *outputInput) {
        if (failed(checkIndices(operation, load.getIndices())))
          result = failure();
        else
          sawLoad = true;
      } else if (outputRank == 1) {
        result = reject(
            plan.task,
            "rank-1 in-place tiling requires every load to read the output "
            "root at the selected counter index");
      }
      return;
    }
    if (auto store = dyn_cast<neura::StoreIndexedOp>(operation)) {
      auto input = accessInputIndex(operation, plan.kernel, "rhs_value",
                                    store.getBase());
      if (!input && outputRank == 2) {
        result = reject(plan.task,
                        "in-place tiling cannot prove a rank-2 store storage "
                        "root");
        return;
      }
      if (!input || *input == *outputInput) {
        if (failed(checkIndices(operation, store.getIndices())))
          result = failure();
        else
          sawStore = true;
      } else if (outputRank == 1) {
        result = reject(
            plan.task,
            "rank-1 in-place tiling requires every store to write the output "
            "root at the selected counter index");
      }
      return;
    }

    // Folded input references on any other op are unclassified memory effects.
    // Do not infer that they are harmless from the operation name.  Walking
    // the complete kernel region is intentional: canonical LLaMA writes are
    // nested below scf.if, unlike the original rank-1 fixture.
    if (mentionsKernelInput(operation, *outputInput)) {
      result = reject(plan.task,
                      "in-place tiling rejects an unknown output memory "
                      "effect");
      return;
    }
    // Explicit uses of the output block argument through an unrecognized op
    // (for example a cast/subview) are ambiguous because the resulting alias
    // may be accessed later.  Reject them instead of claiming tile isolation.
    for (Value operand : operation->getOperands())
      if (forwardsKernelArgument(operand, outputArgument)) {
        result = reject(plan.task,
                        "in-place tiling rejects an unknown output alias or "
                        "memory effect");
        return;
      }
  });
  if (failed(result))
    return failure();
  if (!sawLoad || !sawStore)
    return reject(
        plan.task,
        outputRank == 1
            ? "rank-1 in-place tiling requires an explicit load/store pair on "
              "the selected counter"
            : "in-place tiling requires an explicit output load/store pair on "
              "the selected output region");
  plan.inPlaceSelectedIndex = true;
  return success();
}

// Resolve a memref base through the small set of value-forwarding operations
// emitted by the Taskflow-to-Neura conversion.  A phi/selection is
// deliberately not followed: two storage roots would make the write
// footprint ambiguous, even when the operation currently happens to select
// one branch at runtime.
static std::optional<unsigned>
traceKernelStorageInput(Value value, neura::KernelOp kernel,
                        unsigned depth = 0) {
  if (!value || depth >= 64)
    return std::nullopt;
  for (auto [index, input] : llvm::enumerate(kernel.getInputs()))
    if (value == input)
      return static_cast<unsigned>(index);
  if (auto argument = dyn_cast<BlockArgument>(value))
    if (argument.getOwner() == &kernel.getBody().front() &&
        argument.getArgNumber() < kernel.getInputs().size())
      return argument.getArgNumber();

  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (name == "neura.constant") {
    auto input = parseKernelInputReference(definition->getAttr("value"));
    if (!input || *input >= kernel.getInputs().size())
      return std::nullopt;
    return input;
  }
  if ((name == "neura.data_mov" || name == "neura.ctrl_mov" ||
       name == "neura.grant_predicate" || name == "neura.cast") &&
      definition->getNumOperands() >= 1)
    return traceKernelStorageInput(definition->getOperand(0), kernel,
                                   depth + 1);
  if (isa<memref::CastOp>(definition))
    return traceKernelStorageInput(definition->getOperand(0), kernel,
                                   depth + 1);
  return std::nullopt;
}

// A folded storage reference and an SSA base are two independent pieces of
// provenance.  Require them to agree when both are present; accepting either
// one in isolation would let a stale folded %inputN annotation hide an alias.
static std::optional<unsigned>
resolveKernelStorageInput(Operation *operation, neura::KernelOp kernel,
                          StringRef foldedAttribute, Value base) {
  std::optional<unsigned> folded =
      parseKernelInputReference(operation->getAttr(foldedAttribute));
  if (folded && *folded >= kernel.getInputs().size())
    return std::nullopt;
  if (!base)
    return folded;
  std::optional<unsigned> traced = traceKernelStorageInput(base, kernel);
  if (!traced || (folded && *folded != *traced))
    return std::nullopt;
  return traced;
}

static bool isProvenOutputCoordinate(Value value, neura::CounterOp counter,
                                     unsigned depth = 0) {
  if (value == counter.getCurrentIndex())
    return true;
  if (!value || depth >= 64)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  StringRef name = definition->getName().getStringRef();
  if ((name == "neura.grant_predicate" || name == "neura.data_mov" ||
       name == "neura.ctrl_mov") && definition->getNumOperands() >= 1)
    return isProvenOutputCoordinate(definition->getOperand(0), counter,
                                    depth + 1);
  if (name == "neura.phi" && definition->getNumOperands() == 2)
    return isProvenOutputCoordinate(definition->getOperand(0), counter,
                                    depth + 1) &&
           isProvenOutputCoordinate(definition->getOperand(1), counter,
                                    depth + 1);
  return false;
}

static bool hasFoldedStorageReference(Operation *operation) {
  for (StringRef attribute : {StringRef("lhs_value"),
                              StringRef("rhs_value")})
    if (isFoldedKernelReference(operation->getAttr(attribute), "%input"))
      return true;
  return false;
}

// Neura DFG operations do not all implement MemoryEffectOpInterface.  Keep a
// source-owned allowlist for their data-only operations and use MLIR's
// effect-free proof for ordinary arith/memref cast operations.  Any other
// operation is rejected below, including an opaque operation with an
// unclassified memory effect.
static bool isKnownOutputProofOperation(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (name == "neura.constant" || name == "neura.counter" ||
      name == "neura.grant_once" || name == "neura.reserve" ||
      name == "neura.phi_start" || name == "neura.phi" ||
      name == "neura.icmp" || name == "neura.grant_predicate" ||
      name == "neura.mul" || name == "neura.sub" || name == "neura.add" ||
      name == "neura.not" || name == "neura.sel" || name == "neura.cast" ||
      name == "neura.ctrl_mov" || name == "neura.data_mov" ||
      name == "neura.extract_predicate" || name == "neura.return_value" ||
      name == "neura.yield" || name == "scf.if" || name == "scf.for" ||
      name == "scf.yield" || name == "scf.condition")
    return true;
  if (name == "memref.cast")
    return true;
  return operation->getDialect() &&
         operation->getDialect()->getNamespace() == "arith" &&
         operation->getNumRegions() == 0 && isMemoryEffectFree(operation);
}

static LogicalResult proveNonInPlaceOutputRegion(NeuraTilePlan &plan,
                                                 Value outputRoot) {
  auto outputType = dyn_cast<MemRefType>(outputRoot.getType());
  if (!outputType || outputType.getRank() < 1 ||
      outputType.getRank() > static_cast<int64_t>(plan.kernelCounters.size()))
    return reject(plan.task,
                  "post-Neura output footprint requires a ranked output whose "
                  "leading dimensions have hardware counters");
  auto outputInput = kernelInputIndex(plan.kernel, plan.task, outputRoot);
  if (!outputInput || *outputInput >= plan.kernel.getInputs().size())
    return reject(plan.task,
                  "post-Neura output footprint cannot resolve the output "
                  "kernel input");

  auto outputIndicesMatch = [&](ValueRange indices) {
    if (indices.size() != static_cast<size_t>(outputType.getRank()))
      return false;
    for (auto [dimension, index] : llvm::enumerate(indices))
      if (dimension >= plan.kernelCounters.size() ||
          !isProvenOutputCoordinate(index, plan.kernelCounters[dimension]))
        return false;
    return true;
  };

  bool sawOutputStore = false;
  LogicalResult result = success();
  plan.kernel.walk([&](Operation *operation) {
    if (failed(result) || operation == plan.kernel.getOperation())
      return;

    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      auto input = traceKernelStorageInput(load.getMemRef(), plan.kernel);
      if (!input) {
        result = reject(plan.task,
                        "post-Neura output footprint cannot classify a memref "
                        "load base");
        return;
      }
      if (*input == *outputInput && !outputIndicesMatch(load.getIndices()))
        result = reject(plan.task,
                        "post-Neura output load does not use leading hardware "
                        "counter coordinates");
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      auto input = traceKernelStorageInput(store.getMemRef(), plan.kernel);
      if (!input) {
        result = reject(plan.task,
                        "post-Neura output footprint cannot classify a memref "
                        "store base");
        return;
      }
      if (*input != *outputInput) {
        result = reject(plan.task,
                        "post-Neura output footprint rejects a store to an "
                        "input or aliased storage root");
        return;
      }
      if (!outputIndicesMatch(store.getIndices()))
        result = reject(plan.task,
                        "post-Neura output store does not use leading hardware "
                        "counter coordinates");
      else
        sawOutputStore = true;
      return;
    }
    if (auto load = dyn_cast<neura::LoadIndexedOp>(operation)) {
      auto input = resolveKernelStorageInput(operation, plan.kernel, "lhs_value",
                                             load.getBase());
      if (!input) {
        result = reject(plan.task,
                        "post-Neura output footprint cannot classify an indexed "
                        "load base or folded input");
        return;
      }
      if (*input == *outputInput && !outputIndicesMatch(load.getIndices()))
        result = reject(plan.task,
                        "post-Neura indexed output load does not use leading "
                        "hardware counter coordinates");
      return;
    }
    if (auto store = dyn_cast<neura::StoreIndexedOp>(operation)) {
      // With an explicit destination base, rhs_value is a second address
      // identity rather than a harmless hint.  Reject the contradiction.
      if (store.getBase() && store->hasAttr("rhs_value")) {
        result = reject(plan.task,
                        "post-Neura indexed output store has conflicting SSA "
                        "and folded destination identities");
        return;
      }
      auto input = resolveKernelStorageInput(operation, plan.kernel, "rhs_value",
                                             store.getBase());
      if (!input) {
        result = reject(plan.task,
                        "post-Neura output footprint cannot classify an indexed "
                        "store base or folded input");
        return;
      }
      if (*input != *outputInput) {
        result = reject(plan.task,
                        "post-Neura output footprint rejects an indexed store "
                        "to an input or aliased storage root");
        return;
      }
      if (!outputIndicesMatch(store.getIndices()))
        result = reject(plan.task,
                        "post-Neura indexed output store does not use leading "
                        "hardware counter coordinates");
      else
        sawOutputStore = true;
      return;
    }

    if (isKnownOutputProofOperation(operation))
      return;
    if (hasFoldedStorageReference(operation)) {
      result = reject(plan.task,
                      "post-Neura output footprint contains an unclassified "
                      "folded storage reference");
      return;
    }
    for (Value operand : operation->getOperands()) {
      if (traceKernelStorageInput(operand, plan.kernel)) {
        result = reject(plan.task,
                        "post-Neura output footprint contains an unknown "
                        "storage alias or memory effect");
        return;
      }
    }
    if (!isMemoryEffectFree(operation))
      result = reject(plan.task,
                      "post-Neura output footprint contains an unknown memory "
                      "effect");
  });
  if (failed(result))
    return failure();
  if (!sawOutputStore)
    return reject(plan.task,
                  "post-Neura output footprint requires a proven output store");
  return success();
}

static FailureOr<NeuraTilePlan>
analyzeTask(TaskflowTaskOp task, int64_t axis, int64_t factor) {
  if (!task || !task.getBody().hasOneBlock())
    return failure();
  if (task.getWillWrites().size() != 1 || task.getDoneWrites().size() != 1 ||
      task.getOriginalWriteMemrefs().size() != 1 ||
      !task.getDoneReads().empty() || !task.getValueOutputs().empty() ||
      task.getOriginalReadMemrefs().size() != task.getWillReads().size())
    return reject(task,
                  "post-Neura tiling requires one output state and no yielded "
                  "read/value state");
  if (axis < 0 || axis > 1 || factor < 2)
    return reject(task, "post-Neura tiling requires an M/N axis (0 or 1) and "
                        "factor >= 2");

  NeuraTilePlan plan;
  plan.task = task;
  plan.axis = axis;
  plan.factor = factor;
  for (Operation &operation : task.getBody().front()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      plan.taskCounters.push_back(counter);
      continue;
    }
    if (auto kernel = dyn_cast<neura::KernelOp>(&operation)) {
      if (plan.kernel)
        return reject(task, "post-Neura tiling requires exactly one "
                            "neura.kernel per task");
      plan.kernel = kernel;
      continue;
    }
    if (isa<TaskflowYieldOp, arith::ConstantOp>(&operation))
      continue;
    // Frontends retain literal-resolvable index additions for stencil bounds
    // (e.g. Harris height - 1). The whole source expression is cloned; only
    // the independently checked counter interval is changed. An unresolved
    // input, non-index arithmetic, or signed overflow remains unsupported.
    if (auto add = dyn_cast<arith::AddIOp>(&operation))
      if (add.getType().isIndex() && compileTimeIndex(add.getResult(), task))
        continue;
    return reject(task, "post-Neura tiling accepts only constants, a static "
                        "counter chain, one neura.kernel, and taskflow.yield");
  }
  if (!plan.kernel || plan.taskCounters.empty() ||
      axis >= static_cast<int64_t>(plan.taskCounters.size()))
    return reject(task, "post-Neura tiling could not find the selected counter");
  if (!plan.kernel.getBody().hasOneBlock())
    return reject(plan.kernel,
                  "post-Neura tiling requires a single-block kernel");

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
      return reject(counter, "post-Neura tiling requires constant unit-step "
                            "ordered counters");
    plan.taskCounterLowers.push_back(*lower);
    plan.taskCounterUppers.push_back(*upper);
  }

  for (Operation &operation : plan.kernel.getBody().front())
    if (auto counter = dyn_cast<neura::CounterOp>(&operation))
      plan.kernelCounters.push_back(counter);
  if (plan.kernelCounters.size() != plan.taskCounters.size())
    return reject(plan.kernel,
                  "post-Neura tiling requires one neura.counter per retained "
                  "Taskflow counter");
  for (auto [index, counter] : llvm::enumerate(plan.kernelCounters)) {
    if (auto id = counterId(counter.getOperation()); id && *id != static_cast<int64_t>(index))
      return reject(counter, "Neura counter IDs do not match the Taskflow "
                            "counter chain");
    auto lower = compileTimeCounterBound(
        counter, "lower_bound_value", counter.getLowerBound(), task,
        plan.kernel);
    auto upper = compileTimeCounterBound(
        counter, "upper_bound_value", counter.getUpperBound(), task,
        plan.kernel);
    auto step = compileTimeCounterBound(counter, "step_value",
                                       counter.getStep(), task, plan.kernel);
    auto taskLower = compileTimeIndex(
        plan.taskCounters[index].getLowerBound(), task);
    auto taskUpper = compileTimeIndex(
        plan.taskCounters[index].getUpperBound(), task);
    auto taskStep = compileTimeIndex(plan.taskCounters[index].getStep(), task);
    if (!lower || !upper || !step || !taskLower || !taskUpper || !taskStep ||
        *lower != *taskLower || *upper != *taskUpper || *step != *taskStep ||
        *step != 1)
      return reject(counter, "Neura and Taskflow counter bounds are not the "
                            "same static domain");
    plan.kernelCounterLowers.push_back(*lower);
    plan.kernelCounterUppers.push_back(*upper);
  }

  SmallVector<Value> states(task.getWillReads());
  states.push_back(task.getWillWrites().front());
  for (Value state : states) {
    auto type = dyn_cast<MemRefType>(state.getType());
    if (!type || !type.getLayout().isIdentity() ||
        type.getRank() > static_cast<int64_t>(plan.taskCounters.size()))
      return reject(task, "post-Neura tiling requires identity memrefs with "
                          "rank no greater than the retained counter rank");
  }
  SmallVector<Value> roots(task.getOriginalReadMemrefs());
  roots.push_back(task.getOriginalWriteMemrefs().front());
  Value outputRoot = roots.back();
  bool hasInPlaceRoot = llvm::any_of(
      ArrayRef<Value>(roots).drop_back(),
      [&](Value root) { return root == outputRoot; });
  for (auto [state, root] : llvm::zip(states, roots)) {
    if (state.getType() != root.getType())
      return reject(task, "post-Neura task state and original storage types "
                          "differ");
    auto stateType = dyn_cast<MemRefType>(state.getType());
    if (!stateType)
      return reject(task, "post-Neura tiling requires ranked memref states");

    SmallVector<int64_t> provenShape;
    if (!stateType.hasStaticShape()) {
      if (failed(validateCallerShape(root, stateType, provenShape)))
        return reject(task,
                      "dynamic memref state requires an exact caller shape "
                      "contract matching its rank and static dimensions");
      if (state == task.getWillWrites().front()) {
        if (stateType.getRank() != 2 &&
            !(stateType.getRank() == 1 && hasInPlaceRoot && axis == 0))
          return reject(task,
                        "dynamic output tiling requires one rank-2 output or "
                        "a proven rank-1 in-place output");
        plan.outputShape = provenShape;
        plan.outputNeedsStaticView = true;
      }
    }

    // The output's first two dimensions are the selected M/N domains even
    // when the kernel retains a trailing K counter.  Check them explicitly;
    // the general rank-equals-counter check below intentionally does not make
    // that assumption for input tensors with permuted GEMM indexing.
    if (state == task.getWillWrites().front()) {
      for (int64_t dimension = 0; dimension < stateType.getRank();
           ++dimension) {
        auto lower = compileTimeIndex(
            plan.taskCounters[dimension].getLowerBound(), task);
        auto upper = compileTimeIndex(
            plan.taskCounters[dimension].getUpperBound(), task);
        int64_t extent = stateType.getShape()[dimension];
        if (ShapedType::isDynamic(extent)) {
          if (provenShape.empty() ||
              dimension >= static_cast<int64_t>(provenShape.size()))
            return reject(task, "dynamic output lacks a proven dimension");
          extent = provenShape[dimension];
        }
        if (!lower || !upper || *lower < 0 || *upper <= *lower ||
            *upper > extent)
          return reject(task,
                        "output counter domain is outside the proven shape");
      }
    }

    // When the memref rank matches the retained counter rank, every counter
    // interval must fit the real caller extent.  Dynamic dimensions use the
    // checked contract above rather than the unknown MLIR type bound.
    if (stateType.getRank() ==
        static_cast<int64_t>(plan.taskCounters.size())) {
      for (auto [counterAxis, counter] : llvm::enumerate(plan.taskCounters)) {
        auto lower = compileTimeIndex(counter.getLowerBound(), task);
        auto upper = compileTimeIndex(counter.getUpperBound(), task);
        int64_t extent = stateType.getShape()[counterAxis];
        if (ShapedType::isDynamic(extent)) {
          if (provenShape.empty())
            if (failed(validateCallerShape(root, stateType, provenShape)))
              return reject(task,
                            "dynamic counter domain lacks a caller shape "
                            "proof");
          extent = provenShape[counterAxis];
        }
        if (!lower || !upper || *lower < 0 || *upper <= *lower ||
            *upper > extent)
          return reject(task,
                        "retained counter domain is outside the proven "
                        "memref shape");
      }
    }
  }
  // Tiling needs the output region to be independent of every input.  Two
  // read-only operands may alias without changing the result, and requiring
  // pairwise no-alias metadata for them rejects ordinary GEMM signatures.
  // A function-local alloca/alloc is fresh storage and is therefore distinct
  // from caller-owned input arguments even when the frontend did not attach
  // an amoeba.noalias attribute.
  if (hasInPlaceRoot) {
    if (auto outputType = dyn_cast<MemRefType>(outputRoot.getType());
        !outputType || outputType.getRank() < 1 || outputType.getRank() > 2)
      return reject(task,
                    "in-place tiling is restricted to rank-1 or rank-2 output");
    if (failed(proveInPlaceSelectedRegion(plan, outputRoot)))
      return failure();
  } else {
    for (Value inputRoot : ArrayRef<Value>(roots).drop_back())
      if (!provesDistinctStorage(inputRoot, outputRoot) &&
          !isPrivateStorageRoot(outputRoot))
        return reject(task, "post-Neura tiling requires output storage distinct "
                            "from every input (or amoeba.noalias arguments)");
    if (failed(proveNonInPlaceOutputRegion(plan, outputRoot)))
      return failure();
  }

  auto selectedLower =
      compileTimeIndex(plan.taskCounters[axis].getLowerBound(), task);
  auto selectedUpper =
      compileTimeIndex(plan.taskCounters[axis].getUpperBound(), task);
  if (!selectedLower || !selectedUpper)
    return reject(task, "selected tile axis lacks a compile-time bound");
  plan.lower = *selectedLower;
  plan.upper = *selectedUpper;
  if (factor > plan.upper - plan.lower)
    return reject(task, "post-Neura tiling factor must not exceed the selected "
                        "static extent");
  return plan;
}

static void copyTaskAttributes(TaskflowTaskOp source, TaskflowTaskOp target,
                               StringRef taskName, OpBuilder &builder) {
  for (NamedAttribute attribute : source->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name == "operandSegmentSizes" || name == "resultSegmentSizes" ||
        name == "task_name" || name.starts_with("amoeba.neura.tiling."))
      continue;
    target->setAttr(attribute.getName(), attribute.getValue());
  }
  target->setAttr("task_name", builder.getStringAttr(taskName));
}

static LogicalResult updateKernelBounds(neura::KernelOp kernel,
                                        ArrayRef<int64_t> lowers,
                                        ArrayRef<int64_t> uppers) {
  SmallVector<neura::CounterOp> counters;
  for (Operation &operation : kernel.getBody().front())
    if (auto counter = dyn_cast<neura::CounterOp>(&operation))
      counters.push_back(counter);
  if (lowers.size() != counters.size() || uppers.size() != counters.size())
    return reject(kernel, "cloned kernel lost a Neura counter");
  for (auto [index, counter] : llvm::enumerate(counters)) {
    OpBuilder builder(counter);
    Value lowerValue =
        builder.create<arith::ConstantIndexOp>(kernel.getLoc(), lowers[index]);
    Value upperValue =
        builder.create<arith::ConstantIndexOp>(kernel.getLoc(), uppers[index]);
    counter.getLowerBoundMutable().assign(lowerValue);
    counter.getUpperBoundMutable().assign(upperValue);
    // Some converted kernels retain folded bound attributes even though the
    // SSA operands were omitted.  Keep those attributes synchronized with
    // the cloned tile domain for consumers that inspect metadata.
    if (counter->hasAttr("lower_bound_value"))
      counter->setAttr("lower_bound_value", builder.getIndexAttr(lowers[index]));
    if (counter->hasAttr("upper_bound_value"))
      counter->setAttr("upper_bound_value", builder.getIndexAttr(uppers[index]));
    if (counter->hasAttr("step_value"))
      counter->setAttr("step_value", builder.getIndexAttr(1));
    if (counter->hasAttr("counter_dynamism"))
      counter->setAttr("counter_dynamism", builder.getStringAttr("constant_bound"));
  }
  return success();
}

// A static output view changes the raw Taskflow kernel input type.  The
// converted kernel block argument may carry the same payload inside
// !neura.data, so keep that payload synchronized before a native mapper sees
// the cloned kernel.  Kernel iter-args occupy a suffix after the ordinary
// inputs; validate that suffix without rewriting it.  Direct users of an old
// input argument type cannot be rewritten generically here, so reject those
// cases instead of publishing an internally inconsistent kernel signature.
static LogicalResult
synchronizeKernelInputPayloads(neura::KernelOp kernel) {
  if (!kernel.getBody().hasOneBlock())
    return reject(kernel,
                  "post-Neura tiling requires a single-block kernel for "
                  "input payload synchronization");

  Block &body = kernel.getBody().front();
  auto inputs = kernel.getInputs();
  auto iterArgs = kernel.getIterArgsInit();
  size_t inputCount = inputs.size();
  size_t iterArgCount = iterArgs.size();
  if (body.getNumArguments() != inputCount + iterArgCount)
    return reject(kernel,
                  "post-Neura tiling kernel input, iter-arg, and block "
                  "argument counts differ");

  for (auto [index, argument] : llvm::enumerate(inputs)) {
    BlockArgument blockArgument = body.getArgument(index);
    Type expected = argument.getType();
    // Existing plain-typed kernels are already consistent.  This is the
    // representation used by the LLaMA post-Neura M/N kernels.
    if (blockArgument.getType() == expected)
      continue;

    if (isa<neura::PredicatedValue>(expected))
      return reject(kernel,
                    "post-Neura tiling kernel input unexpectedly carries a "
                    "nested Neura data type");
    auto payload = dyn_cast<neura::PredicatedValue>(blockArgument.getType());
    if (!payload) {
      // A proven static view may replace a dynamic memref input while the
      // cloned body still uses its original dynamic type.  Keep those uses
      // behind a local cast; the task/kernel interface exposes the exact
      // static view, and the body retains its source access semantics.
      Type original = blockArgument.getType();
      if (!blockArgument.use_empty()) {
        if (!isa<MemRefType>(expected) || !isa<MemRefType>(original) ||
            !memref::CastOp::areCastCompatible({expected}, {original}))
          return reject(kernel,
                        "post-Neura tiling cannot retarget a plain kernel "
                        "input with incompatible direct users");
        SmallVector<OpOperand *> oldUses;
        for (OpOperand &use : blockArgument.getUses())
          oldUses.push_back(&use);
        blockArgument.setType(expected);
        OpBuilder castBuilder = OpBuilder::atBlockBegin(&body);
        Value oldView = castBuilder.create<memref::CastOp>(
            kernel.getLoc(), original, blockArgument);
        for (OpOperand *use : oldUses)
          use->set(oldView);
      } else {
        blockArgument.setType(expected);
      }
      continue;
    }
    if (payload.getValueType() == expected)
      continue;
    if (!blockArgument.use_empty())
      return reject(kernel,
                    "post-Neura tiling cannot retarget a kernel input "
                    "payload with direct users");
    blockArgument.setType(neura::PredicatedValue::get(
        kernel.getContext(), expected, payload.getPredicateType()));
  }

  // Iter args are not rewritten by the output static-view conversion.  A
  // predicated kernel may wrap them in !neura.data, while plain kernels keep
  // the raw type; both are valid only when their payload matches the init
  // value exactly.
  for (auto [index, iterArg] : llvm::enumerate(iterArgs)) {
    BlockArgument blockArgument = body.getArgument(inputCount + index);
    Type expected = iterArg.getType();
    if (blockArgument.getType() == expected)
      continue;
    auto payload = dyn_cast<neura::PredicatedValue>(blockArgument.getType());
    if (!payload || payload.getValueType() != expected)
      return reject(kernel,
                    "post-Neura tiling kernel iter-arg payload does not "
                    "match its init value");
  }
  return success();
}

static LogicalResult setProvenOutputRegionAttrs(OpBuilder &builder,
                                                TaskflowTaskOp part,
                                                const NeuraTilePlan &plan,
                                                int64_t partLower,
                                                int64_t partUpper) {
  auto outputType = dyn_cast<MemRefType>(part.getWillWrites().front().getType());
  if (!outputType || outputType.getRank() < 1 ||
      outputType.getRank() > static_cast<int64_t>(plan.taskCounterLowers.size()) ||
      plan.axis < 0 || plan.axis >= outputType.getRank())
    return reject(part,
                  "post-Neura tiling cannot publish an exact output region for "
                  "the cloned part");

  SmallVector<int64_t> lowers;
  SmallVector<int64_t> uppers;
  lowers.reserve(outputType.getRank());
  uppers.reserve(outputType.getRank());
  for (int64_t dimension = 0; dimension < outputType.getRank(); ++dimension) {
    int64_t lower = plan.taskCounterLowers[dimension];
    int64_t upper = plan.taskCounterUppers[dimension];
    if (dimension == plan.axis) {
      lower = partLower;
      upper = partUpper;
    }
    int64_t extent = outputType.getDimSize(dimension);
    if (ShapedType::isDynamic(extent)) {
      if (dimension >= static_cast<int64_t>(plan.outputShape.size()))
        return reject(part,
                      "post-Neura tiling cannot validate a dynamic output "
                      "region against the caller shape");
      extent = plan.outputShape[dimension];
    }
    if (lower < 0 || upper <= lower || upper > extent)
      return reject(part,
                    "post-Neura tiling derived output region is outside the "
                    "proven caller shape");
    lowers.push_back(lower);
    uppers.push_back(upper);
  }
  SmallVector<Attribute> lowerAttrs;
  SmallVector<Attribute> upperAttrs;
  lowerAttrs.push_back(builder.getDenseI64ArrayAttr(lowers));
  upperAttrs.push_back(builder.getDenseI64ArrayAttr(uppers));
  part->setAttr(kTilingOutputRegionLowersAttr,
                builder.getArrayAttr(lowerAttrs));
  part->setAttr(kTilingOutputRegionUppersAttr,
                builder.getArrayAttr(upperAttrs));
  return success();
}

static FailureOr<TaskflowTaskOp>
createPart(OpBuilder &builder, const NeuraTilePlan &plan, int64_t partIndex,
          int64_t partLower, int64_t partUpper, Value outputStorage) {
  TaskflowTaskOp source = plan.task;
  std::string name = (Twine(source.getTaskName()) + ".tile." +
                      Twine(plan.axis) + "." + Twine(partIndex))
                         .str();
  SmallVector<Value> writes(source.getWillWrites());
  SmallVector<Value> originalWrites(source.getOriginalWriteMemrefs());
  if (plan.outputNeedsStaticView) {
    if (!outputStorage || writes.size() != 1 || originalWrites.size() != 1)
      return reject(source, "dynamic-output tile lacks its static storage view");
    writes.front() = outputStorage;
    originalWrites.front() = outputStorage;
  }
  SmallVector<Type> writeTypes;
  writeTypes.reserve(writes.size());
  for (Value write : writes)
    writeTypes.push_back(write.getType());
  TaskflowTaskOp part = builder.create<TaskflowTaskOp>(
      source.getLoc(), source.getDoneReads().getTypes(), writeTypes,
      source.getValueOutputs().getTypes(), source.getWillReads(), writes,
      source.getValueInputs(), builder.getStringAttr(name),
      source.getOriginalReadMemrefs(), originalWrites);
  Block *body = new Block();
  part.getBody().push_back(body);
  IRMapping mapping;
  Block &sourceBody = source.getBody().front();
  for (auto [index, operand] : llvm::enumerate(source->getOperands())) {
    if (index >= sourceBody.getNumArguments())
      break;
    // The dynamic-output path replaces the corresponding write operand with
    // a proven static view.  Body arguments must follow the new task operand
    // types so TaskflowTaskOp verification remains meaningful.
    BlockArgument argument =
        body->addArgument(part->getOperand(index).getType(), source.getLoc());
    mapping.map(sourceBody.getArgument(index), argument);
  }
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  Operation *sourceTerminator = sourceBody.getTerminator();
  for (Operation &operation : sourceBody) {
    if (&operation == sourceTerminator)
      break;
    bodyBuilder.clone(operation, mapping);
  }

  SmallVector<TaskflowCounterOp> clonedTaskCounters;
  neura::KernelOp clonedKernel;
  for (Operation &operation : *body) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      clonedTaskCounters.push_back(counter);
    if (auto kernel = dyn_cast<neura::KernelOp>(&operation))
      clonedKernel = kernel;
  }
  if (clonedTaskCounters.size() != plan.taskCounters.size() || !clonedKernel)
    return reject(source, "post-Neura task clone lost its counter/kernel body");
  if (failed(synchronizeKernelInputPayloads(clonedKernel)))
    return failure();
  SmallVector<int64_t> taskLowers(plan.taskCounterLowers);
  SmallVector<int64_t> taskUppers(plan.taskCounterUppers);
  taskLowers[plan.axis] = partLower;
  taskUppers[plan.axis] = partUpper;
  for (auto [index, counter] : llvm::enumerate(clonedTaskCounters)) {
    OpBuilder counterBuilder(counter);
    Value lowerValue = counterBuilder.create<arith::ConstantIndexOp>(
        source.getLoc(), taskLowers[index]);
    Value upperValue = counterBuilder.create<arith::ConstantIndexOp>(
        source.getLoc(), taskUppers[index]);
    counter.getLowerBoundMutable().assign(lowerValue);
    counter.getUpperBoundMutable().assign(upperValue);
  }
  SmallVector<int64_t> kernelLowers(plan.kernelCounterLowers);
  SmallVector<int64_t> kernelUppers(plan.kernelCounterUppers);
  kernelLowers[plan.axis] = partLower;
  kernelUppers[plan.axis] = partUpper;
  if (failed(updateKernelBounds(clonedKernel, kernelLowers, kernelUppers)))
    return failure();

  if (!sourceTerminator)
    return reject(source, "post-Neura task has no taskflow.yield terminator");
  bodyBuilder.setInsertionPointToEnd(body);
  bodyBuilder.clone(*sourceTerminator, mapping);

  copyTaskAttributes(source, part, name, builder);
  // Current work is derived from every retained Taskflow counter after the
  // selected M/N interval is changed. This includes an unchanged trailing K
  // reduction counter; historical replica total/shard attributes remain
  // attached to their earlier ledger step and are not rewritten here.
  auto counterVolume = [](ArrayRef<int64_t> lowers,
                          ArrayRef<int64_t> uppers)
      -> std::optional<int64_t> {
    if (lowers.empty() || lowers.size() != uppers.size())
      return std::nullopt;
    __int128 product = 1;
    for (auto [lower, upper] : llvm::zip(lowers, uppers)) {
      if (lower < 0 || upper <= lower)
        return std::nullopt;
      product *= static_cast<__int128>(upper - lower);
      if (product <= 0 ||
          product > static_cast<__int128>(std::numeric_limits<int64_t>::max()))
        return std::nullopt;
    }
    return static_cast<int64_t>(product);
  };
  auto parentWork = counterVolume(plan.taskCounterLowers,
                                  plan.taskCounterUppers);
  auto childWork = counterVolume(taskLowers, taskUppers);
  if (!parentWork || !childWork)
    return reject(source, "post-Neura tiling cannot derive an overflow-safe "
                         "current M/N[/K] work volume");
  auto updateCurrentWork = [&](StringRef attribute) -> LogicalResult {
    Attribute raw = source->getAttr(attribute);
    if (!raw)
      return success();
    auto count = dyn_cast<IntegerAttr>(raw);
    if (!count || count.getInt() != *parentWork)
      return reject(source, "post-Neura tiling current trip-count metadata "
                           "does not match its actual Taskflow counter volume");
    part->setAttr(attribute, builder.getI64IntegerAttr(*childWork));
    return success();
  };
  if (failed(updateCurrentWork("trip_count")) ||
      failed(updateCurrentWork("amoeba.selected_trip_count")))
    return failure();
  part->setAttr(kParentTaskAttr, builder.getStringAttr(source.getTaskName()));
  part->setAttr(kAxisAttr, builder.getI64IntegerAttr(plan.axis));
  part->setAttr(kFactorAttr, builder.getI64IntegerAttr(plan.factor));
  part->setAttr(kPartIndexAttr, builder.getI64IntegerAttr(partIndex));
  part->setAttr(kOriginalRangeAttr,
                builder.getDenseI64ArrayAttr({plan.lower, plan.upper}));
  part->setAttr(kDerivedRangeAttr,
                builder.getDenseI64ArrayAttr({partLower, partUpper}));
  part->setAttr(kRewriteAttr, builder.getStringAttr("post-neura-mn-tiling"));
  auto outputType =
      dyn_cast<MemRefType>(part.getWillWrites().front().getType());
  bool publishRankOneInPlaceRegion =
      plan.inPlaceSelectedIndex && outputType && outputType.getRank() == 1;
  if ((!plan.inPlaceSelectedIndex || publishRankOneInPlaceRegion) &&
      failed(setProvenOutputRegionAttrs(builder, part, plan, partLower,
                                        partUpper)))
    return failure();
  return part;
}

static LogicalResult rewriteTask(NeuraTilePlan &plan, func::FuncOp function) {
  SemanticIncomingSnapshot semanticSnapshot;
  if (failed(collectSemanticIncomingSnapshot(function, semanticSnapshot)))
    return failure();
  std::string sourceName = plan.task.getTaskName().str();
  std::map<std::string, std::vector<std::string>> semanticOrigins;
  for (int64_t part = 0; part < plan.factor; ++part) {
    std::string name = (Twine(plan.task.getTaskName()) + ".tile." +
                        Twine(plan.axis) + "." + Twine(part))
                           .str();
    bool collision = false;
    function.walk([&](TaskflowTaskOp task) {
      collision |= task != plan.task && task.getTaskName() == name;
    });
    if (collision)
      return reject(plan.task,
                    "post-Neura tiling-derived task name collides with an "
                    "existing task");
  }

  OpBuilder builder(plan.task);
  Value originalOutput = plan.task.getWillWrites().front();
  Value outputJoinBase = originalOutput;
  auto outputType = dyn_cast<MemRefType>(originalOutput.getType());
  if (!outputType)
    return reject(plan.task, "post-Neura tiling output is not a memref");
  if (plan.outputNeedsStaticView) {
    if (plan.outputShape.empty())
      return reject(plan.task,
                    "dynamic output requires an exact caller shape");
    auto staticType = MemRefType::get(
        plan.outputShape, outputType.getElementType(), outputType.getLayout(),
        outputType.getMemorySpace());
    outputJoinBase = builder.create<memref::CastOp>(
        plan.task.getLoc(), staticType, originalOutput);
    outputType = staticType;
  }
  SmallVector<Value> tileStates;
  int64_t extent = plan.upper - plan.lower;
  int64_t base = extent / plan.factor;
  int64_t remainder = extent % plan.factor;
  int64_t cursor = plan.lower;
  for (int64_t part = 0; part < plan.factor; ++part) {
    int64_t width = base + (part < remainder ? 1 : 0);
    FailureOr<TaskflowTaskOp> tile =
        createPart(builder, plan, part, cursor, cursor + width,
                   outputJoinBase);
    if (failed(tile))
      return failure();
    semanticOrigins[tile->getTaskName().str()] = {sourceName};
    tileStates.push_back(tile->getDoneWrites().front());
    cursor += width;
  }

  if (!outputType.hasStaticShape() || plan.axis >= outputType.getRank())
    return reject(plan.task, "post-Neura tiling requires a static output view "
                            "whose rank contains the selected axis");
  SmallVector<int64_t> regionLower;
  SmallVector<int64_t> regionUpper;
  // The completion join describes the output region, not the complete loop
  // band.  In a GEMM the K counter is carried by the kernel but is absent from
  // the rank-two output memref, so only the output's leading counter domains
  // belong in the join metadata.
  for (int64_t dimension = 0; dimension < outputType.getRank(); ++dimension) {
    TaskflowCounterOp counter = plan.taskCounters[dimension];
    auto lower = compileTimeIndex(counter.getLowerBound(), plan.task);
    auto upper = compileTimeIndex(counter.getUpperBound(), plan.task);
    if (!lower || !upper)
      return reject(counter, "post-Neura tiling lost a static counter domain "
                            "while constructing the completion join");
    regionLower.push_back(*lower);
    regionUpper.push_back(*upper);
  }
  auto join = builder.create<TaskflowJoinOp>(
      plan.task.getLoc(), outputType, tileStates, outputJoinBase,
      builder.getI64IntegerAttr(plan.axis),
      builder.getDenseI64ArrayAttr(regionLower),
      builder.getDenseI64ArrayAttr(regionUpper));
  join->setAttr("amoeba.semantic.completion_only", builder.getUnitAttr());
  join->setAttr(kRewriteAttr, builder.getStringAttr("post-neura-mn-tiling"));
  Value replacement = join.getJoined();
  if (plan.outputNeedsStaticView)
    replacement = builder.create<memref::CastOp>(
        plan.task.getLoc(), originalOutput.getType(), replacement);
  plan.task.getDoneWrites().front().replaceAllUsesWith(replacement);
  plan.task.erase();
  function->setAttr(kRewriteAttr,
                    builder.getStringAttr("post-neura-mn-tiling"));
  if (failed(remapSemanticIncomingEdges(function, semanticSnapshot,
                                         semanticOrigins)))
    return failure();
  return success();
}


struct NeuraFusionPlan {
  TaskflowTaskOp producer;
  TaskflowTaskOp consumer;
  neura::KernelOp producerKernel;
  neura::KernelOp consumerKernel;
  SmallVector<TaskflowChannelOp> channels;
  Value producerResult;
  Value consumerInput;
  Value intermediateRoot;
  BlockArgument producerKernelIntermediate;
  BlockArgument consumerKernelIntermediate;
  memref::StoreOp producerStore;
  memref::LoadOp consumerLoad;
  neura::StoreIndexedOp producerIndexedStore;
  neura::LoadIndexedOp consumerIndexedLoad;
  SmallVector<TaskflowCounterOp> producerTaskCounters;
  SmallVector<TaskflowCounterOp> consumerTaskCounters;
  SmallVector<neura::CounterOp> producerKernelCounters;
  SmallVector<neura::CounterOp> consumerKernelCounters;
  bool forwarded = false;
};

// A sibling fusion has no producer-consumer intermediate.  The two kernels
// consume the same loop point and are emitted sequentially in one Neura
// kernel; every output from both source tasks remains a result of the fused
// Taskflow task.  Keep this plan separate from NeuraFusionPlan because the
// latter intentionally models one RAW edge and has special forwarded-store
// handling throughout its builder.
struct NeuraSiblingFusionPlan {
  TaskflowTaskOp first;
  TaskflowTaskOp second;
  neura::KernelOp firstKernel;
  neura::KernelOp secondKernel;
  SmallVector<TaskflowCounterOp> firstTaskCounters;
  SmallVector<TaskflowCounterOp> secondTaskCounters;
  SmallVector<neura::CounterOp> firstKernelCounters;
  SmallVector<neura::CounterOp> secondKernelCounters;
};

static SmallVector<TaskflowCounterOp>
collectTaskCounters(TaskflowTaskOp task) {
  SmallVector<TaskflowCounterOp> result;
  if (!task || !task.getBody().hasOneBlock())
    return result;
  for (Operation &operation : task.getBody().front())
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      result.push_back(counter);
  return result;
}

static SmallVector<neura::CounterOp>
collectKernelCounters(neura::KernelOp kernel) {
  SmallVector<neura::CounterOp> result;
  if (!kernel || !kernel.getBody().hasOneBlock())
    return result;
  for (Operation &operation : kernel.getBody().front())
    if (auto counter = dyn_cast<neura::CounterOp>(&operation))
      result.push_back(counter);
  return result;
}

static bool matchingStaticCounters(ArrayRef<TaskflowCounterOp> lhs,
                                   ArrayRef<TaskflowCounterOp> rhs,
                                   TaskflowTaskOp lhsTask,
                                   TaskflowTaskOp rhsTask) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t index = 0; index < lhs.size(); ++index) {
    TaskflowCounterOp lhsCounter = lhs[index];
    TaskflowCounterOp rhsCounter = rhs[index];
    auto lhsLower = compileTimeIndex(lhsCounter.getLowerBound(), lhsTask);
    auto rhsLower = compileTimeIndex(rhsCounter.getLowerBound(), rhsTask);
    auto lhsUpper = compileTimeIndex(lhsCounter.getUpperBound(), lhsTask);
    auto rhsUpper = compileTimeIndex(rhsCounter.getUpperBound(), rhsTask);
    auto lhsStep = compileTimeIndex(lhsCounter.getStep(), lhsTask);
    auto rhsStep = compileTimeIndex(rhsCounter.getStep(), rhsTask);
    auto lhsId = counterId(lhsCounter.getOperation());
    auto rhsId = counterId(rhsCounter.getOperation());
    if ((lhsId && *lhsId != static_cast<int64_t>(index)) ||
        (rhsId && *rhsId != static_cast<int64_t>(index)) ||
        (lhsId && rhsId && *lhsId != *rhsId) ||
        !lhsLower || !rhsLower || !lhsUpper || !rhsUpper || !lhsStep ||
        !rhsStep || *lhsLower != *rhsLower || *lhsUpper != *rhsUpper ||
        *lhsStep != *rhsStep || *lhsStep != 1)
      return false;
  }
  return true;
}

static bool matchingStaticKernelCounters(ArrayRef<neura::CounterOp> lhs,
                                         ArrayRef<neura::CounterOp> rhs,
                                         TaskflowTaskOp lhsTask,
                                         TaskflowTaskOp rhsTask,
                                         neura::KernelOp lhsKernel,
                                         neura::KernelOp rhsKernel) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t index = 0; index < lhs.size(); ++index) {
    neura::CounterOp lhsCounter = lhs[index];
    neura::CounterOp rhsCounter = rhs[index];
    if (auto id = counterId(lhsCounter.getOperation());
        id && *id != static_cast<int64_t>(index))
      return false;
    if (auto id = counterId(rhsCounter.getOperation());
        id && *id != static_cast<int64_t>(index))
      return false;
    auto lhsLower = compileTimeCounterBound(lhsCounter, "lower_bound_value",
                                            lhsCounter.getLowerBound(), lhsTask,
                                            lhsKernel);
    auto rhsLower = compileTimeCounterBound(rhsCounter, "lower_bound_value",
                                            rhsCounter.getLowerBound(), rhsTask,
                                            rhsKernel);
    auto lhsUpper = compileTimeCounterBound(lhsCounter, "upper_bound_value",
                                            lhsCounter.getUpperBound(), lhsTask,
                                            lhsKernel);
    auto rhsUpper = compileTimeCounterBound(rhsCounter, "upper_bound_value",
                                            rhsCounter.getUpperBound(), rhsTask,
                                            rhsKernel);
    auto lhsStep = compileTimeCounterBound(lhsCounter, "step_value",
                                            lhsCounter.getStep(), lhsTask,
                                            lhsKernel);
    auto rhsStep = compileTimeCounterBound(rhsCounter, "step_value",
                                            rhsCounter.getStep(), rhsTask,
                                            rhsKernel);
    if (!lhsLower || !rhsLower || !lhsUpper || !rhsUpper || !lhsStep ||
        !rhsStep || *lhsLower != *rhsLower || *lhsUpper != *rhsUpper ||
        *lhsStep != *rhsStep || *lhsStep != 1)
      return false;
  }
  return true;
}

// Collect the loop-carried state slots that contribute to a dataflow value.
// The traversal is deliberately conservative: a block argument from a nested
// or external region, or a malformed folded reference, is not silently
// classified as independent state.
static bool collectKernelStateDependencies(
    Value value, neura::KernelOp kernel, unsigned iterArgCount,
    const llvm::SmallDenseMap<Value, unsigned, 8> &reserveOwners,
    llvm::SmallDenseSet<unsigned, 8> &dependencies,
    llvm::SmallDenseSet<Value, 32> &visited) {
  if (!value || !visited.insert(value).second)
    return true;
  Block &body = kernel.getBody().front();
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() != &body)
      return false;
    unsigned inputCount = kernel.getInputs().size();
    if (argument.getArgNumber() >= inputCount) {
      unsigned state = argument.getArgNumber() - inputCount;
      if (state >= iterArgCount)
        return false;
      dependencies.insert(state);
    }
    return true;
  }
  if (auto owner = reserveOwners.find(value); owner != reserveOwners.end()) {
    dependencies.insert(owner->second);
    return true;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition || !kernel->isAncestor(definition))
    return false;
  for (NamedAttribute attribute : definition->getAttrs()) {
    Attribute raw = attribute.getValue();
    if (isFoldedKernelReference(raw, "%input")) {
      auto input = parseKernelInputReference(raw);
      if (!input || *input >= kernel.getInputs().size())
        return false;
    }
    if (isFoldedKernelReference(raw, "%iter_arg_init")) {
      auto state = parseKernelIterArgReference(raw);
      if (!state || *state >= iterArgCount)
        return false;
      dependencies.insert(*state);
    }
  }
  for (Value operand : definition->getOperands())
    if (!collectKernelStateDependencies(operand, kernel, iterArgCount,
                                         reserveOwners, dependencies, visited))
      return false;
  return true;
}

// Prove that each carried slot has its own reserve/phi recurrence and that no
// operation combines two slots.  This is the source-owned condition that
// permits concatenating two stateful kernels: cloning the complete region
// keeps each reserve and phi graph separate, while a cross-slot recurrence is
// rejected before any rewrite is attempted.
static bool hasIndependentKernelState(neura::KernelOp kernel) {
  if (!kernel || !kernel.getBody().hasOneBlock())
    return false;
  unsigned inputCount = kernel.getInputs().size();
  unsigned iterArgCount = kernel.getIterArgsInit().size();
  Block &body = kernel.getBody().front();
  if (body.getNumArguments() != inputCount + iterArgCount)
    return false;
  auto yield = dyn_cast<neura::YieldOp>(body.getTerminator());
  if (!yield || (yield.getIterArgsNext().size() != 0 &&
                 yield.getIterArgsNext().size() != iterArgCount) ||
      (yield.getResults().size() != 0 &&
       yield.getResults().size() != kernel.getNumResults()))
    return false;
  if (iterArgCount == 0)
    return true;

  llvm::SmallDenseMap<Value, unsigned, 8> reserveOwners;
  llvm::SmallDenseSet<unsigned, 8> initializedStates;
  for (Operation &operation : body) {
    auto phiStart = dyn_cast<neura::PhiStartOp>(&operation);
    if (!phiStart)
      continue;
    llvm::SmallDenseSet<unsigned, 8> dependencies;
    llvm::SmallDenseSet<Value, 32> visited;
    if (!collectKernelStateDependencies(phiStart.getInitValue(), kernel,
                                         iterArgCount, reserveOwners,
                                         dependencies, visited) ||
        dependencies.size() != 1)
      return false;
    unsigned state = *dependencies.begin();
    if (!initializedStates.insert(state).second)
      return false;
    if (!phiStart.getReserved().getDefiningOp<neura::ReserveOp>())
      return false;
    auto [owner, inserted] = reserveOwners.try_emplace(phiStart.getReserved(),
                                                        state);
    if (!inserted && owner->second != state)
      return false;
  }
  if (initializedStates.size() != iterArgCount)
    return false;

  for (Operation &operation : body) {
    if (isa<neura::YieldOp>(&operation))
      continue;
    llvm::SmallDenseSet<unsigned, 8> dependencies;
    for (Value operand : operation.getOperands()) {
      llvm::SmallDenseSet<Value, 32> visited;
      if (!collectKernelStateDependencies(operand, kernel, iterArgCount,
                                           reserveOwners, dependencies,
                                           visited))
        return false;
    }
    if (dependencies.size() > 1)
      return false;
    if (auto ctrl = dyn_cast<neura::CtrlMovOp>(&operation)) {
      (void)ctrl;
      if (operation.getNumOperands() != 2)
        return false;
      auto target = reserveOwners.find(operation.getOperand(1));
      if (target == reserveOwners.end())
        return false;
      if (!dependencies.empty() && *dependencies.begin() != target->second)
        return false;
    }
  }
  return true;
}

// The fact extractor permits unrelated operations between two siblings when
// their memory effects are known not to conflict.  Recheck that proof here;
// this post-conversion pass must never move a task across an operation whose
// storage aliasing it cannot prove.
static bool siblingCanMoveFirstToSecond(TaskflowTaskOp first,
                                        TaskflowTaskOp second) {
  if (!first || !second || first == second ||
      first->hasAttr("amoeba.control_predecessors") ||
      second->hasAttr("amoeba.control_predecessors") ||
      first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second))
    return false;
  auto completeMemoryProvenance = [](TaskflowTaskOp task) {
    return task.getOriginalReadMemrefs().size() == task.getWillReads().size() &&
           task.getOriginalWriteMemrefs().size() == task.getWillWrites().size();
  };
  if (!completeMemoryProvenance(first) || !completeMemoryProvenance(second))
    return false;
  SmallVector<Value> firstReads(first.getOriginalReadMemrefs().begin(),
                                first.getOriginalReadMemrefs().end());
  SmallVector<Value> firstWrites(first.getOriginalWriteMemrefs().begin(),
                                 first.getOriginalWriteMemrefs().end());
  for (Operation *operation = first->getNextNode();
       operation && operation != second.getOperation();
       operation = operation->getNextNode()) {
    if (isa<TaskflowChannelOp, arith::ConstantOp, memref::AllocOp,
            memref::AllocaOp>(operation))
      continue;
    auto crossed = dyn_cast<TaskflowTaskOp>(operation);
    if (!crossed || crossed->hasAttr("amoeba.control_predecessors") ||
        !completeMemoryProvenance(crossed))
      return false;
    for (Value result : first->getResults())
      for (OpOperand &use : result.getUses())
        if (use.getOwner() == crossed.getOperation())
          return false;
    // Shared reads preserve memory values under this movement.  Only a
    // pair containing a write needs a storage-disjointness proof.
    for (Value lhs : firstReads)
      for (Value rhs : crossed.getOriginalWriteMemrefs())
        if (!provesDistinctStorage(lhs, rhs))
          return false;
    for (Value lhs : firstWrites)
      for (Value rhs : crossed.getOriginalReadMemrefs())
        if (!provesDistinctStorage(lhs, rhs))
          return false;
    for (Value lhs : firstWrites)
      for (Value rhs : crossed.getOriginalWriteMemrefs())
        if (!provesDistinctStorage(lhs, rhs))
          return false;
  }
  return true;
}

static FailureOr<NeuraSiblingFusionPlan>
analyzePostNeuraSiblingFusion(ModuleOp module, StringRef firstName,
                              StringRef secondName) {
  SmallVector<TaskflowTaskOp> firstMatches;
  SmallVector<TaskflowTaskOp> secondMatches;
  module.walk([&](TaskflowTaskOp task) {
    if (task.getTaskName() == firstName)
      firstMatches.push_back(task);
    if (task.getTaskName() == secondName)
      secondMatches.push_back(task);
  });
  if (firstMatches.size() != 1 || secondMatches.size() != 1) {
    module.emitError()
        << "post-Neura sibling fusion names must each identify exactly one "
           "task";
    return failure();
  }
  TaskflowTaskOp first = firstMatches.front();
  TaskflowTaskOp second = secondMatches.front();
  if (!siblingCanMoveFirstToSecond(first, second))
    return reject(second, "post-Neura sibling fusion requires an ordered pair "
                           "with proven nonconflicting intervening effects");

  auto rejectTask = [&](TaskflowTaskOp task, StringRef reason)
      -> FailureOr<NeuraSiblingFusionPlan> {
    return reject(task, (Twine("post-Neura sibling fusion ") + reason).str());
  };
  if (!first.getBody().hasOneBlock() || !second.getBody().hasOneBlock())
    return rejectTask(second, "requires one body block per task");
  if (first.getDoneReads().size() != 0 || second.getDoneReads().size() != 0)
    return rejectTask(second, "requires no done-read completion states");
  if (first.getValueOutputs().size() != 0 || second.getValueOutputs().size() != 0)
    return rejectTask(second, "requires no value outputs");
  auto dependsOn = [](TaskflowTaskOp consumer, TaskflowTaskOp producer) {
    auto depends = [&](Value input) {
      Value source = input;
      while (auto channel = source.getDefiningOp<TaskflowChannelOp>())
        source = channel.getSource();
      return source.getDefiningOp<TaskflowTaskOp>() == producer;
    };
    for (Value input : consumer.getWillReads())
      if (depends(input))
        return true;
    for (Value input : consumer.getValueInputs())
      if (depends(input))
        return true;
    return false;
  };
  if (dependsOn(second, first) || dependsOn(first, second))
    return rejectTask(second, "rejects an inter-task dependency");
  if (first.getWillWrites().empty() || second.getWillWrites().empty() ||
      first.getWillWrites().size() != first.getDoneWrites().size() ||
      second.getWillWrites().size() != second.getDoneWrites().size() ||
      first.getOriginalReadMemrefs().size() != first.getWillReads().size() ||
      second.getOriginalReadMemrefs().size() != second.getWillReads().size() ||
      first.getOriginalWriteMemrefs().size() != first.getWillWrites().size() ||
      second.getOriginalWriteMemrefs().size() != second.getWillWrites().size())
    return rejectTask(second, "requires complete memory-only input/output provenance");
  if (!matchingStaticCounters(collectTaskCounters(first),
                              collectTaskCounters(second), first, second) ||
      collectTaskCounters(first).empty())
    return rejectTask(second, "requires matching static Taskflow counter domains");

  // Sibling fusion is a sequential execution in one kernel.  Shared reads
  // are legal; any uncertain read/write or write/write alias would change the
  // original independent-task semantics and therefore rejects the action.
  SmallVector<Value> reads;
  SmallVector<Value> writes;
  SmallVector<Value> allWrites;
  auto appendRoots = [](SmallVectorImpl<Value> &out, ValueRange values) {
    for (Value value : values)
      if (!llvm::is_contained(out, value))
        out.push_back(value);
  };
  appendRoots(reads, first.getOriginalReadMemrefs());
  appendRoots(reads, second.getOriginalReadMemrefs());
  appendRoots(writes, first.getOriginalWriteMemrefs());
  appendRoots(writes, second.getOriginalWriteMemrefs());
  llvm::append_range(allWrites, first.getOriginalWriteMemrefs());
  llvm::append_range(allWrites, second.getOriginalWriteMemrefs());
  for (size_t lhs = 0; lhs < allWrites.size(); ++lhs)
    for (size_t rhs = lhs + 1; rhs < allWrites.size(); ++rhs)
      if (!provesDistinctStorage(allWrites[lhs], allWrites[rhs]))
        return rejectTask(second, "rejects ambiguous output aliasing");
  for (Value read : reads)
    for (Value write : writes)
      if (!provesDistinctStorage(read, write))
        return rejectTask(second, "rejects read/write aliasing");

  neura::KernelOp firstKernel;
  neura::KernelOp secondKernel;
  for (Operation &operation : first.getBody().front())
    if (auto kernel = dyn_cast<neura::KernelOp>(&operation)) {
      if (firstKernel)
        return rejectTask(first, "requires one first neura.kernel");
      firstKernel = kernel;
    }
  for (Operation &operation : second.getBody().front())
    if (auto kernel = dyn_cast<neura::KernelOp>(&operation)) {
      if (secondKernel)
        return rejectTask(second, "requires one second neura.kernel");
      secondKernel = kernel;
    }
  if (!firstKernel || !secondKernel || !firstKernel.getBody().hasOneBlock() ||
      !secondKernel.getBody().hasOneBlock())
    return rejectTask(second, "requires two one-block kernels");
  if (!hasIndependentKernelState(firstKernel) ||
      !hasIndependentKernelState(secondKernel))
    return rejectTask(
        second,
        "requires independently carried reserve/phi state with no "
        "cross-slot recurrence");
  auto isStructuralKernelAttribute = [](StringRef name) {
    return name == "operandSegmentSizes" || name == "resultSegmentSizes";
  };
  for (NamedAttribute attribute : firstKernel->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (isStructuralKernelAttribute(name))
      continue;
    if (secondKernel->getAttr(name) != attribute.getValue())
      return rejectTask(second,
                        "requires matching kernel execution attributes");
  }
  for (NamedAttribute attribute : secondKernel->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (isStructuralKernelAttribute(name))
      continue;
    if (firstKernel->getAttr(name) != attribute.getValue())
      return rejectTask(second,
                        "requires matching kernel execution attributes");
  }

  SmallVector<TaskflowCounterOp> firstTaskCounters = collectTaskCounters(first);
  SmallVector<TaskflowCounterOp> secondTaskCounters = collectTaskCounters(second);
  SmallVector<neura::CounterOp> firstKernelCounters =
      collectKernelCounters(firstKernel);
  SmallVector<neura::CounterOp> secondKernelCounters =
      collectKernelCounters(secondKernel);
  if (firstKernelCounters.size() != firstTaskCounters.size() ||
      secondKernelCounters.size() != secondTaskCounters.size() ||
      !matchingStaticKernelCounters(firstKernelCounters, secondKernelCounters,
                                    first, second, firstKernel, secondKernel))
    return rejectTask(second, "requires matching static Neura counter domains");

  NeuraSiblingFusionPlan plan;
  plan.first = first;
  plan.second = second;
  plan.firstKernel = firstKernel;
  plan.secondKernel = secondKernel;
  plan.firstTaskCounters = std::move(firstTaskCounters);
  plan.secondTaskCounters = std::move(secondTaskCounters);
  plan.firstKernelCounters = std::move(firstKernelCounters);
  plan.secondKernelCounters = std::move(secondKernelCounters);
  return plan;
}

static std::optional<Value> peelIdentityDataMov(Value value,
                                                Block *accessBlock) {
  while (auto move = value.getDefiningOp<neura::DataMovOp>()) {
    Operation *operation = move.getOperation();
    if (operation->getBlock() != accessBlock ||
        operation->getNumOperands() != 1 ||
        operation->getNumResults() != 1 || operation->getNumRegions() != 0 ||
        operation->getNumSuccessors() != 0 ||
        !operation->getAttrs().empty() || operation->getResult(0) != value ||
        operation->getOperand(0).getType() != operation->getResult(0).getType())
      return std::nullopt;
    // A DataMov may be peeled only when it preserves both payload and the
    // Neura validity type, and is in the same guarded block as the access.
    value = operation->getOperand(0);
  }
  return value;
}

static bool equivalentCounterIndex(Value lhs, Value rhs,
                                   ArrayRef<neura::CounterOp> lhsCounters,
                                   ArrayRef<neura::CounterOp> rhsCounters,
                                   Block *lhsAccessBlock,
                                   Block *rhsAccessBlock) {
  auto peeledLhs = peelIdentityDataMov(lhs, lhsAccessBlock);
  auto peeledRhs = peelIdentityDataMov(rhs, rhsAccessBlock);
  if (!peeledLhs || !peeledRhs)
    return false;
  lhs = *peeledLhs;
  rhs = *peeledRhs;
  if (lhs == rhs)
    return true;
  for (size_t index = 0; index < lhsCounters.size(); ++index) {
    neura::CounterOp lhsCounter = lhsCounters[index];
    neura::CounterOp rhsCounter = rhsCounters[index];
    if (lhs == lhsCounter.getCurrentIndex() &&
        rhs == rhsCounter.getCurrentIndex())
      return true;
  }
  auto lhsConstant = constantIndex(lhs);
  auto rhsConstant = constantIndex(rhs);
  return lhsConstant && rhsConstant && *lhsConstant == *rhsConstant;
}

static bool equivalentAccessIndices(ValueRange lhs, ValueRange rhs,
                                    ArrayRef<neura::CounterOp> lhsCounters,
                                    ArrayRef<neura::CounterOp> rhsCounters,
                                    Block *lhsAccessBlock,
                                    Block *rhsAccessBlock) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (!equivalentCounterIndex(left, right, lhsCounters, rhsCounters,
                                lhsAccessBlock, rhsAccessBlock))
      return false;
  return true;
}

static BlockArgument kernelMemrefArgument(neura::KernelOp kernel,
                                          Value taskBodyArgument) {
  for (auto [index, input] : llvm::enumerate(kernel.getInputs()))
    if (input == taskBodyArgument && index < kernel.getBody().front().getNumArguments())
      return kernel.getBody().front().getArgument(index);
  return BlockArgument();
}

static bool hasOnlyAllowedIntermediateUses(neura::KernelOp kernel,
                                           BlockArgument intermediate,
                                           memref::StoreOp expectedStore,
                                           memref::LoadOp expectedLoad) {
  if (!intermediate)
    return false;
  bool valid = true;
  kernel.walk([&](Operation *operation) {
    if (!valid || operation == kernel.getOperation())
      return;
    for (Value operand : operation->getOperands()) {
      if (operand != intermediate)
        continue;
      if (operation != expectedStore.getOperation() &&
          operation != expectedLoad.getOperation())
        valid = false;
    }
  });
  return valid;
}

static FailureOr<NeuraFusionPlan>
analyzePostNeuraFusion(ModuleOp module, StringRef producerName,
                       StringRef consumerName, StringRef mode) {
  bool forwarded = mode == "producer-consumer-forwarded";
  if (!forwarded && mode != "producer-consumer-retained") {
    module.emitError()
        << "post-Neura fusion-mode must be producer-consumer-retained or "
           "producer-consumer-forwarded";
    return failure();
  }
  SmallVector<TaskflowTaskOp> producers;
  SmallVector<TaskflowTaskOp> consumers;
  module.walk([&](TaskflowTaskOp task) {
    if (task.getTaskName() == producerName)
      producers.push_back(task);
    if (task.getTaskName() == consumerName)
      consumers.push_back(task);
  });
  if (producers.size() != 1 || consumers.size() != 1) {
    module.emitError()
        << "post-Neura fusion names must each identify exactly one task";
    return failure();
  }
  TaskflowTaskOp producer = producers.front();
  TaskflowTaskOp consumer = consumers.front();
  if (producer == consumer || producer->getBlock() != consumer->getBlock() ||
      !producer->isBeforeInBlock(consumer))
    return reject(consumer, "post-Neura fusion requires an ordered task pair "
                            "in one block");

  SmallVector<TaskflowChannelOp> channels;
  Value producerResult;
  Value consumerInput;
  for (Value input : consumer.getWillReads()) {
    Value source = input;
    SmallVector<TaskflowChannelOp> path;
    while (auto channel = source.getDefiningOp<TaskflowChannelOp>()) {
      if (!source.hasOneUse())
        return reject(consumer, "post-Neura fusion requires an exclusive "
                                "channel path");
      path.push_back(channel);
      source = channel.getSource();
    }
    if (!llvm::is_contained(producer.getDoneWrites(), source))
      continue;
    if (producerResult) {
      return reject(consumer, "post-Neura fusion requires exactly one RAW "
                              "producer-to-consumer value");
    }
    producerResult = source;
    consumerInput = input;
    channels = std::move(path);
  }
  if (!producerResult)
    return reject(consumer, "post-Neura fusion found no producer-consumer RAW "
                            "edge");
  if (!producerResult.hasOneUse() || !consumerInput.hasOneUse())
    return reject(consumer, "post-Neura fusion requires a single-use "
                            "intermediate state");

  // The fused operation is placed at the consumer.  Moving the producer
  // there is legal only when every crossed task has complete memory
  // provenance and all read/write or write/write pairs are proved disjoint.
  // Keep shared reads, private allocations and constant scaffolding; reject
  // control predecessors, unknown effects and producer-result consumers.
  if (!siblingCanMoveFirstToSecond(producer, consumer))
    return reject(consumer, "post-Neura fusion cannot prove nonconflicting "
                            "intervening effects");
  for (Value result : producer.getDoneReads())
    if (!result.use_empty())
      return reject(producer, "post-Neura fusion currently requires no "
                              "producer done-read state");
  if (!producer.getValueOutputs().empty() || !consumer.getDoneReads().empty() ||
      !consumer.getValueOutputs().empty())
    return reject(consumer, "post-Neura fusion requires memory-only task "
                            "states");
  if (producer.getWillWrites().size() != 1 ||
      consumer.getWillWrites().size() != 1 ||
      producer.getOriginalWriteMemrefs().size() != 1 ||
      consumer.getOriginalWriteMemrefs().size() != 1)
    return reject(consumer, "post-Neura fusion requires one producer and one "
                            "consumer output state");

  SmallVector<TaskflowCounterOp> producerTaskCounters =
      collectTaskCounters(producer);
  SmallVector<TaskflowCounterOp> consumerTaskCounters =
      collectTaskCounters(consumer);
  if (!matchingStaticCounters(producerTaskCounters, consumerTaskCounters,
                              producer, consumer) ||
      producerTaskCounters.empty())
    return reject(consumer, "post-Neura fusion requires matching static "
                            "Taskflow counter domains");

  neura::KernelOp producerKernel;
  neura::KernelOp consumerKernel;
  for (Operation &operation : producer.getBody().front())
    if (auto kernel = dyn_cast<neura::KernelOp>(&operation)) {
      if (producerKernel)
        return reject(producer, "post-Neura fusion requires one producer "
                                "neura.kernel");
      producerKernel = kernel;
    }
  for (Operation &operation : consumer.getBody().front())
    if (auto kernel = dyn_cast<neura::KernelOp>(&operation)) {
      if (consumerKernel)
        return reject(consumer, "post-Neura fusion requires one consumer "
                                "neura.kernel");
      consumerKernel = kernel;
    }
  if (!producerKernel || !consumerKernel ||
      !producerKernel.getBody().hasOneBlock() ||
      !consumerKernel.getBody().hasOneBlock() ||
      !producerKernel.getIterArgsInit().empty() ||
      !consumerKernel.getIterArgsInit().empty() ||
      !producerKernel.getOutputs().empty() ||
      !consumerKernel.getOutputs().empty())
    return reject(consumer, "post-Neura fusion requires one stateless kernel "
                            "per task");

  SmallVector<neura::CounterOp> producerKernelCounters =
      collectKernelCounters(producerKernel);
  SmallVector<neura::CounterOp> consumerKernelCounters =
      collectKernelCounters(consumerKernel);
  if (!matchingStaticKernelCounters(producerKernelCounters,
                                    consumerKernelCounters, producer, consumer,
                                    producerKernel, consumerKernel) ||
      producerKernelCounters.size() != producerTaskCounters.size())
    return reject(consumer, "post-Neura fusion requires matching static Neura "
                            "counter domains");

  Block &producerBody = producer.getBody().front();
  Block &consumerBody = consumer.getBody().front();
  BlockArgument producerIntermediateTaskArg =
      producerBody.getArgument(producer.getWillReads().size());
  unsigned consumerIntermediateIndex = 0;
  bool foundInput = false;
  for (auto [index, input] : llvm::enumerate(consumer.getWillReads())) {
    Value source = input;
    while (auto channel = source.getDefiningOp<TaskflowChannelOp>())
      source = channel.getSource();
    if (source == producerResult) {
      consumerIntermediateIndex = index;
      foundInput = true;
      break;
    }
  }
  if (!foundInput || !producerIntermediateTaskArg ||
      consumerIntermediateIndex >= consumerBody.getNumArguments())
    return reject(consumer, "post-Neura fusion cannot locate intermediate "
                            "task body arguments");
  BlockArgument consumerIntermediateTaskArg =
      consumerBody.getArgument(consumerIntermediateIndex);
  BlockArgument producerKernelIntermediate =
      kernelMemrefArgument(producerKernel, producerIntermediateTaskArg);
  BlockArgument consumerKernelIntermediate =
      kernelMemrefArgument(consumerKernel, consumerIntermediateTaskArg);
  if (!producerKernelIntermediate || !consumerKernelIntermediate)
    return reject(consumer, "post-Neura fusion requires the intermediate "
                            "memref to be an explicit Neura kernel input");

  Value intermediateRoot = producer.getOriginalWriteMemrefs().front();
  if (consumer.getOriginalReadMemrefs().size() <= consumerIntermediateIndex ||
      consumer.getOriginalReadMemrefs()[consumerIntermediateIndex] !=
          intermediateRoot)
    return reject(consumer, "post-Neura fusion requires matching original "
                            "intermediate storage provenance");
  if (forwarded && !isPrivateStorageRoot(intermediateRoot))
    return reject(consumer, "post-Neura forwarded fusion requires private "
                            "intermediate storage (memref.alloc/alloca)");
  // A retained fused kernel interleaves producer and consumer accesses at
  // each counter point.  Require every distinct input/output root touched by
  // this pair to be disjoint: a caller-visible argument is accepted only
  // under the explicit per-argument amoeba.noalias contract.  In particular,
  // the C signature alone does not establish this property.
  SmallVector<Value> storageRoots;
  auto appendRoots = [&](ValueRange values) {
    for (Value value : values)
      if (!llvm::is_contained(storageRoots, value))
        storageRoots.push_back(value);
  };
  appendRoots(producer.getOriginalReadMemrefs());
  appendRoots(producer.getOriginalWriteMemrefs());
  appendRoots(consumer.getOriginalReadMemrefs());
  appendRoots(consumer.getOriginalWriteMemrefs());
  for (size_t lhs = 0; lhs < storageRoots.size(); ++lhs)
    for (size_t rhs = lhs + 1; rhs < storageRoots.size(); ++rhs)
      if (!provesDistinctStorage(storageRoots[lhs], storageRoots[rhs]))
        return reject(consumer, "post-Neura fusion requires pairwise distinct "
                                "input/output storage roots");

  // The intermediate must not be observed by any operation outside this
  // producer/consumer pair.  This catches function returns, other tasks, and
  // ad-hoc side effects before any state is removed.
  for (OpOperand &use : intermediateRoot.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == producer.getOperation() || owner == consumer.getOperation())
      continue;
    bool channelUse = isa<TaskflowChannelOp>(owner);
    if (!channelUse)
      return reject(consumer, "post-Neura fusion rejects an externally "
                              "observed intermediate storage");
  }
  module.walk([&](TaskflowTaskOp task) {
    if (task == producer || task == consumer)
      return;
    for (Value root : task.getOriginalReadMemrefs())
      if (root == intermediateRoot)
        task.emitError("post-Neura fusion rejects an intermediate read by "
                       "another task");
    for (Value root : task.getOriginalWriteMemrefs())
      if (root == intermediateRoot)
        task.emitError("post-Neura fusion rejects an intermediate write by "
                       "another task");
  });
  bool otherDiagnostic = false;
  module.walk([&](TaskflowTaskOp task) {
    if (task == producer || task == consumer)
      return;
    for (Value root : task.getOriginalReadMemrefs())
      otherDiagnostic |= root == intermediateRoot;
    for (Value root : task.getOriginalWriteMemrefs())
      otherDiagnostic |= root == intermediateRoot;
  });
  if (otherDiagnostic)
    return failure();

  memref::StoreOp producerStore;
  memref::LoadOp consumerLoad;
  neura::StoreIndexedOp producerIndexedStore;
  neura::LoadIndexedOp consumerIndexedLoad;
  SmallVector<memref::StoreOp> producerStores;
  SmallVector<memref::LoadOp> consumerLoads;
  SmallVector<neura::StoreIndexedOp> producerIndexedStores;
  SmallVector<neura::LoadIndexedOp> consumerIndexedLoads;
  producerKernel.walk([&](memref::StoreOp store) {
    if (store.getMemRef() == producerKernelIntermediate)
      producerStores.push_back(store);
  });
  consumerKernel.walk([&](memref::LoadOp load) {
    if (load.getMemRef() == consumerKernelIntermediate)
      consumerLoads.push_back(load);
  });
  auto producerIntermediateIndex = kernelInputIndex(
      producerKernel, producer, producerIntermediateTaskArg);
  auto consumerIntermediateKernelIndex = kernelInputIndex(
      consumerKernel, consumer, consumerIntermediateTaskArg);
  if (!producerIntermediateIndex || !consumerIntermediateKernelIndex)
    return reject(consumer, "post-Neura fusion cannot resolve indexed "
                            "intermediate storage inputs");
  producerKernel.walk([&](neura::StoreIndexedOp store) {
    if (accessInputIndex(store, producerKernel, "rhs_value",
                         store.getBase()) == producerIntermediateIndex)
      producerIndexedStores.push_back(store);
  });
  consumerKernel.walk([&](neura::LoadIndexedOp load) {
    if (accessInputIndex(load, consumerKernel, "lhs_value",
                         load.getBase()) == consumerIntermediateKernelIndex)
      consumerIndexedLoads.push_back(load);
  });
  bool directMemref = producerStores.size() == 1 &&
                      consumerLoads.size() == 1 &&
                      producerIndexedStores.empty() &&
                      consumerIndexedLoads.empty();
  bool directIndexed = producerIndexedStores.size() == 1 &&
                       consumerIndexedLoads.size() == 1 &&
                       producerStores.empty() && consumerLoads.empty();
  if (!directMemref && !directIndexed)
    return reject(consumer, "post-Neura fusion requires exactly one matching "
                            "intermediate store and load");
  if (directIndexed) {
    // Caller-visible Radar scratch cannot be forwarded: both the write and
    // the read remain in the fused body.  Matching per-iteration indices
    // guarantees that interleaving keeps the original RAW order.
    if (forwarded)
      return reject(consumer, "post-Neura indexed forwarded fusion requires "
                              "a separate private-storage proof");
    producerIndexedStore = producerIndexedStores.front();
    consumerIndexedLoad = consumerIndexedLoads.front();
    if (producerIndexedStore->getBlock() != &producerKernel.getBody().front() ||
        consumerIndexedLoad->getBlock() != &consumerKernel.getBody().front())
      return reject(consumer, "post-Neura indexed fusion requires direct "
                              "kernel-body accesses");
    if (!equivalentAccessIndices(producerIndexedStore.getIndices(),
                                 consumerIndexedLoad.getIndices(),
                                 producerKernelCounters,
                                 consumerKernelCounters,
                                 producerIndexedStore->getBlock(),
                                 consumerIndexedLoad->getBlock()))
      return reject(consumer, "post-Neura indexed fusion rejects mismatched "
                              "producer-store/consumer-load indices");

    // The single matched pair must be the entire intermediate-memory effect.
    // Folded %inputN references are not SSA uses, so scan their attributes
    // explicitly instead of relying on BlockArgument::getUses().
    auto onlyExpectedAccess = [&](neura::KernelOp kernel, unsigned inputIndex,
                                  Operation *expected) {
      bool valid = true;
      kernel.walk([&](Operation *operation) {
        if (!valid || operation == kernel.getOperation())
          return;
        if (auto load = dyn_cast<neura::LoadIndexedOp>(operation)) {
          if (accessInputIndex(operation, kernel, "lhs_value", load.getBase()) ==
              inputIndex && operation != expected)
            valid = false;
          return;
        }
        if (auto store = dyn_cast<neura::StoreIndexedOp>(operation)) {
          if (accessInputIndex(operation, kernel, "rhs_value", store.getBase()) ==
              inputIndex && operation != expected)
            valid = false;
          return;
        }
        if (mentionsKernelInput(operation, inputIndex))
          valid = false;
        for (Value operand : operation->getOperands())
          if (forwardsKernelArgument(
                  operand, kernel.getBody().front().getArgument(inputIndex)))
            valid = false;
      });
      return valid;
    };
    if (!onlyExpectedAccess(producerKernel, *producerIntermediateIndex,
                            producerIndexedStore) ||
        !onlyExpectedAccess(consumerKernel, *consumerIntermediateKernelIndex,
                            consumerIndexedLoad))
      return reject(consumer, "post-Neura indexed fusion rejects extra "
                              "intermediate storage users");
  } else {
    producerStore = producerStores.front();
    consumerLoad = consumerLoads.front();
    // The fused builder skips only direct children of each kernel body.
    if (producerStore->getBlock() != &producerKernel.getBody().front() ||
        consumerLoad->getBlock() != &consumerKernel.getBody().front())
      return reject(consumer, "post-Neura fusion requires direct kernel-body "
                              "intermediate store/load operations");
    if (!hasOnlyAllowedIntermediateUses(producerKernel,
                                        producerKernelIntermediate,
                                        producerStore, memref::LoadOp()) ||
        !hasOnlyAllowedIntermediateUses(consumerKernel,
                                        consumerKernelIntermediate,
                                        memref::StoreOp(), consumerLoad))
      return reject(consumer, "post-Neura fusion found an unsupported use of "
                              "the intermediate memref");
    if (!equivalentAccessIndices(producerStore.getIndices(),
                                 consumerLoad.getIndices(),
                                 producerKernelCounters,
                                 consumerKernelCounters,
                                 producerStore->getBlock(),
                                 consumerLoad->getBlock()))
      return reject(consumer, "post-Neura fusion rejects mismatched "
                              "producer-store/consumer-load indices");
  }

  NeuraFusionPlan plan;
  plan.producer = producer;
  plan.consumer = consumer;
  plan.producerKernel = producerKernel;
  plan.consumerKernel = consumerKernel;
  plan.channels = std::move(channels);
  plan.producerResult = producerResult;
  plan.consumerInput = consumerInput;
  plan.intermediateRoot = intermediateRoot;
  plan.producerKernelIntermediate = producerKernelIntermediate;
  plan.consumerKernelIntermediate = consumerKernelIntermediate;
  plan.producerStore = producerStore;
  plan.consumerLoad = consumerLoad;
  plan.producerIndexedStore = producerIndexedStore;
  plan.consumerIndexedLoad = consumerIndexedLoad;
  plan.producerTaskCounters = std::move(producerTaskCounters);
  plan.consumerTaskCounters = std::move(consumerTaskCounters);
  plan.producerKernelCounters = std::move(producerKernelCounters);
  plan.consumerKernelCounters = std::move(consumerKernelCounters);
  plan.forwarded = forwarded;
  return plan;
}

static void appendUniqueValue(SmallVectorImpl<Value> &values, Value value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

static Value resolveThroughFusion(Value value, TaskflowTaskOp task) {
  for (auto [index, result] : llvm::enumerate(task.getDoneReads()))
    if (value == result)
      return task.getWillReads()[index];
  for (auto [index, result] : llvm::enumerate(task.getDoneWrites()))
    if (value == result)
      return task.getWillWrites()[index];
  return value;
}

static Value resolveFusionInput(Value value, TaskflowTaskOp producer) {
  return resolveThroughFusion(value, producer);
}

static FailureOr<TaskflowTaskOp>
createPostNeuraFusedTask(const NeuraFusionPlan &plan, OpBuilder &builder,
                         StringRef fusedName) {
  TaskflowTaskOp producer = plan.producer;
  TaskflowTaskOp consumer = plan.consumer;
  // Generated MLIR Op wrappers expose non-const accessors even when the
  // operation itself is only inspected. Keep local wrapper copies so this
  // builder remains const-correct with respect to the analysis plan.
  neura::KernelOp producerKernel = plan.producerKernel;
  neura::KernelOp consumerKernel = plan.consumerKernel;
  memref::StoreOp producerStore = plan.producerStore;
  memref::LoadOp consumerLoad = plan.consumerLoad;
  SmallVector<Value> fusedReads;
  for (Value value : producer.getWillReads())
    appendUniqueValue(fusedReads, value);
  for (Value value : consumer.getWillReads()) {
    Value source = value;
    while (auto channel = source.getDefiningOp<TaskflowChannelOp>())
      source = channel.getSource();
    if (source != plan.producerResult)
      appendUniqueValue(fusedReads, resolveFusionInput(value, producer));
  }
  SmallVector<Value> fusedWrites;
  if (!plan.forwarded)
    for (Value value : producer.getWillWrites())
      appendUniqueValue(fusedWrites, value);
  for (Value value : consumer.getWillWrites())
    appendUniqueValue(fusedWrites, resolveFusionInput(value, producer));
  SmallVector<Value> fusedValues;
  for (Value value : producer.getValueInputs())
    appendUniqueValue(fusedValues, value);
  for (Value value : consumer.getValueInputs())
    appendUniqueValue(fusedValues, resolveFusionInput(value, producer));

  SmallVector<Value> originalReads;
  for (Value value : producer.getOriginalReadMemrefs())
    appendUniqueValue(originalReads, value);
  for (Value value : consumer.getOriginalReadMemrefs())
    if (value != plan.intermediateRoot)
      appendUniqueValue(originalReads, value);
  SmallVector<Value> originalWrites;
  if (!plan.forwarded)
    for (Value value : producer.getOriginalWriteMemrefs())
      appendUniqueValue(originalWrites, value);
  for (Value value : consumer.getOriginalWriteMemrefs())
    appendUniqueValue(originalWrites, value);

  SmallVector<Type> writeTypes;
  if (!plan.forwarded)
    llvm::append_range(writeTypes, producer.getDoneWrites().getTypes());
  llvm::append_range(writeTypes, consumer.getDoneWrites().getTypes());
  TaskflowTaskOp fused = builder.create<TaskflowTaskOp>(
      consumer.getLoc(), TypeRange{}, writeTypes, TypeRange{}, fusedReads,
      fusedWrites, fusedValues, builder.getStringAttr(fusedName),
      originalReads, originalWrites);
  Block *body = new Block();
  fused.getBody().push_back(body);
  for (Value value : fusedReads)
    body->addArgument(value.getType(), fused.getLoc());
  for (Value value : fusedWrites)
    body->addArgument(value.getType(), fused.getLoc());
  for (Value value : fusedValues)
    body->addArgument(value.getType(), fused.getLoc());

  llvm::SmallDenseMap<Value, Value> outerValues;
  auto mapTaskInputs = [&](TaskflowTaskOp task, bool isProducer) {
    Block &taskBody = task.getBody().front();
    for (auto [index, value] : llvm::enumerate(task.getWillReads())) {
      Value source = value;
      while (auto channel = source.getDefiningOp<TaskflowChannelOp>())
        source = channel.getSource();
      if (!isProducer && source == plan.producerResult && plan.forwarded)
        continue;
      Value mapped = resolveFusionInput(value, producer);
      auto it = llvm::find(fusedReads, mapped);
      if (it == fusedReads.end())
        continue;
      outerValues[taskBody.getArgument(index)] =
          body->getArgument(std::distance(fusedReads.begin(), it));
    }
    unsigned writeBase = fusedReads.size();
    for (auto [index, value] : llvm::enumerate(task.getWillWrites())) {
      Value mapped = resolveFusionInput(value, producer);
      auto it = llvm::find(fusedWrites, mapped);
      if (it == fusedWrites.end())
        continue;
      unsigned position = std::distance(fusedWrites.begin(), it);
      outerValues[taskBody.getArgument(task.getWillReads().size() + index)] =
          body->getArgument(writeBase + position);
    }
    unsigned valueBase = fusedReads.size() + fusedWrites.size();
    for (auto [index, value] : llvm::enumerate(task.getValueInputs())) {
      Value mapped = resolveFusionInput(value, producer);
      auto it = llvm::find(fusedValues, mapped);
      if (it == fusedValues.end())
        continue;
      outerValues[taskBody.getArgument(task.getWillReads().size() +
                                       task.getWillWrites().size() + index)] =
          body->getArgument(valueBase + std::distance(fusedValues.begin(), it));
    }
  };
  mapTaskInputs(producer, true);
  mapTaskInputs(consumer, false);
  if (!plan.forwarded) {
    for (auto [index, input] : llvm::enumerate(consumer.getWillReads())) {
      Value source = input;
      while (auto channel = source.getDefiningOp<TaskflowChannelOp>())
        source = channel.getSource();
      if (source != plan.producerResult ||
          index >= consumer.getBody().front().getNumArguments())
        continue;
      Value mapped = resolveFusionInput(input, producer);
      auto it = llvm::find(fusedWrites, mapped);
      if (it != fusedWrites.end()) {
        unsigned position = std::distance(fusedWrites.begin(), it);
        outerValues[consumer.getBody().front().getArgument(index)] =
            body->getArgument(fusedReads.size() + position);
      }
    }
  }
  (void)outerValues;

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  IRMapping outerMapping;
  for (BlockArgument argument : producer.getBody().front().getArguments()) {
    if (auto mapped = outerValues.lookup(argument))
      outerMapping.map(argument, mapped);
  }
  for (Operation &operation : producer.getBody().front()) {
    if (isa<neura::KernelOp, TaskflowYieldOp>(&operation))
      continue;
    bodyBuilder.clone(operation, outerMapping);
  }

  auto mapOuterValue = [&](Value value) -> Value {
    if (auto mapped = outerMapping.lookupOrNull(value))
      return mapped;
    if (auto mapped = outerValues.lookup(value))
      return mapped;
    return value;
  };

  // A dynamic Taskflow counter bound can also be an operation result in the
  // task body (Radar's `%upper = arith.addi %bound, %delta`). The producer
  // body is cloned above, but the consumer body is erased below and is not
  // cloned because its counter domain was already proven equivalent. Map any
  // consumer kernel operand that names a counter bound back to the matching
  // producer bound before constructing the fused kernel. Leaving the original
  // consumer result in `kernelInputs` would create a dangling use when the
  // consumer task is erased.
  auto mapKernelInput = [&](Value value, bool isProducer) -> Value {
    if (isProducer)
      return mapOuterValue(value);
    for (size_t index = 0; index < plan.consumerTaskCounters.size();
         ++index) {
      if (index >= plan.producerTaskCounters.size())
        break;
      TaskflowCounterOp consumerCounter = plan.consumerTaskCounters[index];
      TaskflowCounterOp producerCounter = plan.producerTaskCounters[index];
      auto consumerId = counterId(consumerCounter.getOperation());
      auto producerId = counterId(producerCounter.getOperation());
      if ((consumerId && *consumerId != static_cast<int64_t>(index)) ||
          (producerId && *producerId != static_cast<int64_t>(index)) ||
          (consumerId && producerId && *consumerId != *producerId))
        continue;
      if (value == consumerCounter.getLowerBound())
        return mapOuterValue(producerCounter.getLowerBound());
      if (value == consumerCounter.getUpperBound())
        return mapOuterValue(producerCounter.getUpperBound());
      if (value == consumerCounter.getStep())
        return mapOuterValue(producerCounter.getStep());
    }
    return mapOuterValue(value);
  };

  auto belongsToErasedTask = [](Value value, TaskflowTaskOp task) {
    Operation *definition = value.getDefiningOp();
    if (!definition) {
      if (auto argument = dyn_cast<BlockArgument>(value))
        if (Block *owner = argument.getOwner())
          definition = owner->getParentOp();
    }
    while (definition) {
      if (definition == task.getOperation())
        return true;
      definition = definition->getParentOp();
    }
    return false;
  };

  SmallVector<Value> kernelInputs;
  SmallVector<Type> kernelInputBodyTypes;
  bool compatibleKernelInputTypes = true;
  auto appendKernelInputs = [&](neura::KernelOp kernel,
                                BlockArgument intermediate,
                                bool skipIntermediate, bool isProducer)
      -> LogicalResult {
    for (auto [index, input] : llvm::enumerate(kernel.getInputs())) {
      bool isIntermediate =
          index < kernel.getBody().front().getNumArguments() &&
          kernel.getBody().front().getArgument(index) == intermediate;
      if (skipIntermediate && isIntermediate)
        continue;
      Value mapped = mapKernelInput(input, isProducer);
      if (!mapped || belongsToErasedTask(mapped, plan.producer) ||
          belongsToErasedTask(mapped, plan.consumer))
        return reject(consumer,
                      "post-Neura fusion left a kernel input defined in an "
                      "erased task");
      Type bodyType = kernel.getBody().front().getArgument(index).getType();
      auto existing = llvm::find(kernelInputs, mapped);
      if (existing == kernelInputs.end()) {
        kernelInputs.push_back(mapped);
        kernelInputBodyTypes.push_back(bodyType);
      } else if (kernelInputBodyTypes[std::distance(kernelInputs.begin(),
                                                   existing)] != bodyType) {
        compatibleKernelInputTypes = false;
      }
    }
    return success();
  };
  if (failed(appendKernelInputs(producerKernel,
                                plan.producerKernelIntermediate,
                                plan.forwarded, true)) ||
      failed(appendKernelInputs(consumerKernel,
                                plan.consumerKernelIntermediate,
                                plan.forwarded, false)))
    return failure();
  if (!compatibleKernelInputTypes)
    return reject(consumer, "post-Neura fusion kernel inputs have incompatible "
                            "body payload types");
  auto fusedKernel = bodyBuilder.create<neura::KernelOp>(
      consumer.getLoc(), TypeRange{}, kernelInputs, ValueRange{},
      producerKernel.getCgraIdAttr(),
      producerKernel.getKernelNameAttr(),
      producerKernel.getAcceleratorAttr(),
      producerKernel.getKernelMetadataAttr());
  Block *kernelBody = new Block();
  fusedKernel.getBody().push_back(kernelBody);
  llvm::SmallDenseMap<Value, BlockArgument> kernelInputArgs;
  for (auto [index, input] : llvm::enumerate(kernelInputs))
    kernelInputArgs[input] = kernelBody->addArgument(
        kernelInputBodyTypes[index], fusedKernel.getLoc());
  IRMapping kernelMapping;
  auto mapKernelBlockArgs = [&](neura::KernelOp kernel,
                                BlockArgument intermediate,
                                bool skipIntermediate, bool isProducer) {
    for (auto [index, input] : llvm::enumerate(kernel.getInputs())) {
      bool isIntermediate =
          index < kernel.getBody().front().getNumArguments() &&
          kernel.getBody().front().getArgument(index) == intermediate;
      if (skipIntermediate && isIntermediate)
        continue;
      Value mappedOuter = mapKernelInput(input, isProducer);
      auto it = kernelInputArgs.find(mappedOuter);
      if (it == kernelInputArgs.end())
        return false;
      kernelMapping.map(kernel.getBody().front().getArgument(index), it->second);
    }
    return true;
  };
  if (!mapKernelBlockArgs(producerKernel,
                          plan.producerKernelIntermediate, plan.forwarded,
                          true) ||
      !mapKernelBlockArgs(consumerKernel,
                          plan.consumerKernelIntermediate, plan.forwarded,
                          false))
    return reject(consumer, "post-Neura fusion failed to map kernel operands");

  // Folded Neura inputs use string attributes such as lhs_value="%input0"
  // and upper_bound_value="%input4". Translate them through the same guarded
  // v4 kernel-input map used for SSA operands; otherwise merging two kernels
  // would silently retarget Radar's indexed storage and symbolic bound.
  auto remapFoldedInputs = [&](Operation *source, Operation *cloned,
                               neura::KernelOp sourceKernel,
                               bool isProducer) {
    for (NamedAttribute attribute : source->getAttrs()) {
      auto inputIndex = parseKernelInputReference(attribute.getValue());
      if (!inputIndex)
        continue;
      if (*inputIndex >= sourceKernel.getInputs().size())
        return false;
      Value mapped = mapKernelInput(sourceKernel.getInputs()[*inputIndex],
                                    isProducer);
      if (!mapped || belongsToErasedTask(mapped, plan.producer) ||
          belongsToErasedTask(mapped, plan.consumer))
        return false;
      auto found = llvm::find(kernelInputs, mapped);
      if (found == kernelInputs.end())
        return false;
      unsigned newIndex = std::distance(kernelInputs.begin(), found);
      cloned->setAttr(attribute.getName(),
                      builder.getStringAttr((Twine("%input") +
                                             Twine(newIndex)).str()));
    }
    return true;
  };
  SmallVector<neura::CounterOp> clonedCounters;
  OpBuilder kernelBuilder = OpBuilder::atBlockEnd(kernelBody);
  for (Operation &operation : producerKernel.getBody().front()) {
    if (isa<neura::YieldOp>(&operation))
      continue;
    if (auto store = dyn_cast<memref::StoreOp>(&operation))
      if (plan.forwarded && store == producerStore)
        continue;
    Operation *cloned = kernelBuilder.clone(operation, kernelMapping);
    if (!remapFoldedInputs(&operation, cloned, producerKernel, true))
      return reject(consumer, "post-Neura fusion cannot remap producer "
                              "folded input reference");
    if (auto counter = dyn_cast<neura::CounterOp>(cloned))
      clonedCounters.push_back(counter);
  }
  if (clonedCounters.size() != plan.producerKernelCounters.size())
    return reject(consumer, "post-Neura fusion lost the producer counter chain");
  Value forwardedValue;
  if (plan.forwarded)
    forwardedValue = kernelMapping.lookupOrDefault(
        producerStore.getValueToStore());
  for (Operation &operation : consumerKernel.getBody().front()) {
    if (isa<neura::YieldOp>(&operation))
      continue;
    if (auto counter = dyn_cast<neura::CounterOp>(&operation)) {
      auto id = counterId(counter.getOperation());
      if (!id || *id < 0 || *id >= static_cast<int64_t>(clonedCounters.size()))
        return reject(consumer, "post-Neura fusion cannot map consumer counter");
      kernelMapping.map(counter.getCurrentIndex(),
                        clonedCounters[*id].getCurrentIndex());
      continue;
    }
    if (plan.forwarded) {
      if (auto load = dyn_cast<memref::LoadOp>(&operation))
        if (load == consumerLoad) {
          if (!forwardedValue)
            return reject(consumer, "post-Neura fusion has no forwarded value");
          kernelMapping.map(load.getResult(), forwardedValue);
          continue;
        }
    }
    Operation *cloned = kernelBuilder.clone(operation, kernelMapping);
    if (!remapFoldedInputs(&operation, cloned, consumerKernel, false))
      return reject(consumer, "post-Neura fusion cannot remap consumer "
                              "folded input reference");
  }
  kernelBuilder.setInsertionPointToEnd(kernelBody);
  kernelBuilder.create<neura::YieldOp>(consumer.getLoc());

  SmallVector<Value> writeYields;
  if (!plan.forwarded)
    writeYields.push_back(body->getArgument(fusedReads.size()));
  writeYields.push_back(body->getArgument(fusedReads.size() + fusedWrites.size() - 1));
  bodyBuilder.setInsertionPointToEnd(body);
  bodyBuilder.create<TaskflowYieldOp>(consumer.getLoc(), ValueRange{}, writeYields,
                                      ValueRange{});
  for (NamedAttribute attribute : consumer->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name == "operandSegmentSizes" || name == "resultSegmentSizes" ||
        name == "task_name" || name.starts_with("amoeba.neura.fusion."))
      continue;
    fused->setAttr(attribute.getName(), attribute.getValue());
  }
  fused->setAttr("task_name", builder.getStringAttr(fusedName));
  fused->setAttr(kRewriteAttr,
                 builder.getStringAttr(plan.forwarded
                                           ? "post-neura-producer-consumer-forwarded"
                                           : "post-neura-producer-consumer-retained"));
  fused->setAttr("amoeba.neura.fusion.mode",
                 builder.getStringAttr(plan.forwarded ? "forwarded" : "retained"));
  fused->setAttr("amoeba.neura.fusion.eliminated_stores",
                 builder.getI64IntegerAttr(plan.forwarded ? 1 : 0));
  fused->setAttr("amoeba.neura.fusion.eliminated_loads",
                 builder.getI64IntegerAttr(plan.forwarded ? 1 : 0));
  return fused;
}

static FailureOr<TaskflowTaskOp>
createPostNeuraSiblingTask(const NeuraSiblingFusionPlan &plan,
                           OpBuilder &builder, StringRef fusedName) {
  TaskflowTaskOp first = plan.first;
  TaskflowTaskOp second = plan.second;
  neura::KernelOp firstKernel = plan.firstKernel;
  neura::KernelOp secondKernel = plan.secondKernel;

  SmallVector<Value> fusedReads;
  SmallVector<Value> fusedWrites;
  SmallVector<Value> fusedValues;
  SmallVector<Value> originalReads;
  SmallVector<Value> originalWrites;
  for (Value value : first.getWillReads())
    appendUniqueValue(fusedReads, value);
  for (Value value : second.getWillReads())
    appendUniqueValue(fusedReads, value);
  for (Value value : first.getWillWrites())
    appendUniqueValue(fusedWrites, value);
  for (Value value : second.getWillWrites())
    appendUniqueValue(fusedWrites, value);
  for (Value value : first.getValueInputs())
    appendUniqueValue(fusedValues, value);
  for (Value value : second.getValueInputs())
    appendUniqueValue(fusedValues, value);
  for (Value value : first.getOriginalReadMemrefs())
    appendUniqueValue(originalReads, value);
  for (Value value : second.getOriginalReadMemrefs())
    appendUniqueValue(originalReads, value);
  for (Value value : first.getOriginalWriteMemrefs())
    appendUniqueValue(originalWrites, value);
  for (Value value : second.getOriginalWriteMemrefs())
    appendUniqueValue(originalWrites, value);

  SmallVector<Type> writeTypes;
  llvm::append_range(writeTypes, first.getDoneWrites().getTypes());
  llvm::append_range(writeTypes, second.getDoneWrites().getTypes());
  TaskflowTaskOp fused = builder.create<TaskflowTaskOp>(
      second.getLoc(), TypeRange{}, writeTypes, TypeRange{}, fusedReads,
      fusedWrites, fusedValues, builder.getStringAttr(fusedName), originalReads,
      originalWrites);
  Block *body = new Block();
  fused.getBody().push_back(body);
  for (Value value : fusedReads)
    body->addArgument(value.getType(), fused.getLoc());
  for (Value value : fusedWrites)
    body->addArgument(value.getType(), fused.getLoc());
  for (Value value : fusedValues)
    body->addArgument(value.getType(), fused.getLoc());

  IRMapping firstTaskMapping;
  IRMapping secondTaskMapping;
  auto mapTaskInputs = [&](TaskflowTaskOp task, IRMapping &mapping) {
    Block &taskBody = task.getBody().front();
    for (auto [index, value] : llvm::enumerate(task.getWillReads())) {
      auto found = llvm::find(fusedReads, value);
      if (found == fusedReads.end())
        return false;
      mapping.map(taskBody.getArgument(index),
                  body->getArgument(found - fusedReads.begin()));
    }
    unsigned writeBase = fusedReads.size();
    for (auto [index, value] : llvm::enumerate(task.getWillWrites())) {
      auto found = llvm::find(fusedWrites, value);
      if (found == fusedWrites.end())
        return false;
      mapping.map(taskBody.getArgument(task.getWillReads().size() + index),
                  body->getArgument(writeBase + found - fusedWrites.begin()));
    }
    unsigned valueBase = fusedReads.size() + fusedWrites.size();
    for (auto [index, value] : llvm::enumerate(task.getValueInputs())) {
      auto found = llvm::find(fusedValues, value);
      if (found == fusedValues.end())
        return false;
      mapping.map(taskBody.getArgument(task.getWillReads().size() +
                                       task.getWillWrites().size() + index),
                  body->getArgument(valueBase + found - fusedValues.begin()));
    }
    return true;
  };
  if (!mapTaskInputs(first, firstTaskMapping) ||
      !mapTaskInputs(second, secondTaskMapping)) {
    fused.erase();
    return reject(second, "post-Neura sibling fusion could not map task inputs");
  }

  // Kernel results are normally consumed only by the dataflow return path,
  // but a prepared task may contain a shell operation that uses one of them.
  // Defer that operation (and its transitive users) until the fused kernel
  // results have been created, so those users can be mapped instead of
  // retaining a dangling value from an erased source task.
  auto collectDeferredKernelUsers =
      [](TaskflowTaskOp task, neura::KernelOp kernel,
         SmallVectorImpl<Operation *> &deferred) {
        std::set<Operation *> deferredDefinitions;
        for (Operation &operation : task.getBody().front()) {
          if (isa<neura::KernelOp, TaskflowYieldOp>(&operation))
            continue;
          bool dependsOnKernelResult = false;
          for (Value operand : operation.getOperands()) {
            if (llvm::is_contained(kernel.getResults(), operand) ||
                (operand.getDefiningOp() &&
                 deferredDefinitions.find(operand.getDefiningOp()) !=
                     deferredDefinitions.end())) {
              dependsOnKernelResult = true;
              break;
            }
          }
          if (dependsOnKernelResult) {
            deferred.push_back(&operation);
            deferredDefinitions.insert(&operation);
          }
        }
      };
  SmallVector<Operation *> deferredFirst;
  SmallVector<Operation *> deferredSecond;
  collectDeferredKernelUsers(first, firstKernel, deferredFirst);
  collectDeferredKernelUsers(second, secondKernel, deferredSecond);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  auto cloneTaskShell = [&](TaskflowTaskOp task, IRMapping &mapping,
                            ArrayRef<TaskflowCounterOp> canonicalCounters,
                            bool canonical,
                            ArrayRef<Operation *> deferred) -> LogicalResult {
    unsigned clonedCounterCount = 0;
    for (Operation &operation : task.getBody().front()) {
      if (isa<neura::KernelOp, TaskflowYieldOp>(&operation))
        continue;
      if (llvm::is_contained(deferred, &operation))
        continue;
      if (isa<TaskflowCounterOp>(&operation) && !canonical)
        continue;
      bodyBuilder.clone(operation, mapping);
      if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
        if (clonedCounterCount >= canonicalCounters.size())
          return reject(second, "post-Neura sibling fusion has invalid task counter id");
        if (!mapping.lookupOrNull(counter.getCounterIndex()))
          return reject(second, "post-Neura sibling fusion lost a task counter");
        ++clonedCounterCount;
      }
    }
    return success();
  };
  if (failed(cloneTaskShell(first, firstTaskMapping, plan.firstTaskCounters,
                            true, deferredFirst))) {
    fused.erase();
    return failure();
  }
  // Reuse the first task's loop counters.  The source proof established equal
  // static domains, so cloning a second counter chain would create a second
  // set of hardware counters rather than a fused loop point.
  for (auto [index, counter] : llvm::enumerate(plan.secondTaskCounters)) {
    if (index >= plan.firstTaskCounters.size()) {
      fused.erase();
      return reject(second, "post-Neura sibling fusion has mismatched task counters");
    }
    TaskflowCounterOp firstCounter = plan.firstTaskCounters[index];
    Value mapped = firstTaskMapping.lookupOrNull(firstCounter.getCounterIndex());
    if (!mapped) {
      fused.erase();
      return reject(second, "post-Neura sibling fusion cannot map task counters");
    }
    TaskflowCounterOp secondCounter = counter;
    secondTaskMapping.map(secondCounter.getCounterIndex(), mapped);
  }
  if (failed(cloneTaskShell(second, secondTaskMapping, plan.firstTaskCounters,
                            false, deferredSecond))) {
    fused.erase();
    return failure();
  }

  auto mapBodyValue = [](IRMapping &mapping, Value value) -> Value {
    if (Value mapped = mapping.lookupOrNull(value))
      return mapped;
    return value;
  };
  SmallVector<Value> kernelInputs;
  SmallVector<Type> kernelInputBodyTypes;
  auto appendKernelInputs = [&](neura::KernelOp kernel,
                                IRMapping &taskMapping) -> LogicalResult {
    for (auto [index, input] : llvm::enumerate(kernel.getInputs())) {
      Value mapped = mapBodyValue(taskMapping, input);
      if (!mapped || !mapped.getType())
        return reject(second, "post-Neura sibling fusion has an unmapped kernel input");
      auto found = llvm::find(kernelInputs, mapped);
      Type bodyType = kernel.getBody().front().getArgument(index).getType();
      if (found == kernelInputs.end()) {
        kernelInputs.push_back(mapped);
        kernelInputBodyTypes.push_back(bodyType);
      } else if (kernelInputBodyTypes[found - kernelInputs.begin()] != bodyType) {
        return reject(second, "post-Neura sibling fusion has incompatible kernel input types");
      }
    }
    return success();
  };
  if (failed(appendKernelInputs(firstKernel, firstTaskMapping)) ||
      failed(appendKernelInputs(secondKernel, secondTaskMapping))) {
    fused.erase();
    return failure();
  }

  SmallVector<Value> kernelIterArgs;
  auto appendKernelIterArgs = [&](neura::KernelOp kernel,
                                  IRMapping &taskMapping) -> LogicalResult {
    for (Value initial : kernel.getIterArgsInit()) {
      Value mapped = mapBodyValue(taskMapping, initial);
      if (!mapped || !mapped.getType())
        return reject(second,
                      "post-Neura sibling fusion has an unmapped iter-arg "
                      "initializer");
      kernelIterArgs.push_back(mapped);
    }
    return success();
  };
  if (failed(appendKernelIterArgs(firstKernel, firstTaskMapping)) ||
      failed(appendKernelIterArgs(secondKernel, secondTaskMapping))) {
    fused.erase();
    return failure();
  }

  SmallVector<Type> kernelResultTypes;
  llvm::append_range(kernelResultTypes, firstKernel.getResultTypes());
  llvm::append_range(kernelResultTypes, secondKernel.getResultTypes());
  auto fusedKernel = bodyBuilder.create<neura::KernelOp>(
      second.getLoc(), kernelResultTypes, kernelInputs, kernelIterArgs,
      firstKernel.getCgraIdAttr(), firstKernel.getKernelNameAttr(),
      firstKernel.getAcceleratorAttr(), firstKernel.getKernelMetadataAttr());
  for (NamedAttribute attribute : firstKernel->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name == "operandSegmentSizes" || name == "resultSegmentSizes")
      continue;
    fusedKernel->setAttr(attribute.getName(), attribute.getValue());
  }
  Block *kernelBody = new Block();
  fusedKernel.getBody().push_back(kernelBody);
  llvm::SmallDenseMap<Value, BlockArgument> kernelInputArgs;
  for (auto [index, input] : llvm::enumerate(kernelInputs))
    kernelInputArgs[input] =
        kernelBody->addArgument(kernelInputBodyTypes[index], fusedKernel.getLoc());
  SmallVector<BlockArgument> kernelIterArgArgs;
  bool predicatedKernel = false;
  if (auto mode = firstKernel->getAttrOfType<StringAttr>("dataflow_mode"))
    predicatedKernel = mode.getValue() == "predicate";
  for (Value initial : kernelIterArgs) {
    Type bodyType = initial.getType();
    // Post-dataflow kernels carry !neura.data block arguments for every
    // state slot, including the iter-arg suffix.  The source LLaMA kernels
    // may still print their original raw iter-arg arguments, but a newly
    // constructed fused kernel is consumed by the host replay contract,
    // which requires the predicated payload wrapper here.
    if (predicatedKernel)
      bodyType = neura::PredicatedValue::get(
          fusedKernel.getContext(), bodyType,
          IntegerType::get(fusedKernel.getContext(), 1));
    kernelIterArgArgs.push_back(
        kernelBody->addArgument(bodyType, fusedKernel.getLoc()));
  }
  IRMapping firstKernelMapping;
  IRMapping secondKernelMapping;
  auto mapKernelBlockArgs = [&](neura::KernelOp kernel,
                                IRMapping &taskMapping,
                                IRMapping &kernelMapping,
                                unsigned iterOffset) -> LogicalResult {
    for (auto [index, input] : llvm::enumerate(kernel.getInputs())) {
      Value mappedOuter = mapBodyValue(taskMapping, input);
      auto found = kernelInputArgs.find(mappedOuter);
      if (found == kernelInputArgs.end())
        return reject(second, "post-Neura sibling fusion failed to map kernel input");
      kernelMapping.map(kernel.getBody().front().getArgument(index),
                        found->second);
    }
    for (auto [index, initial] : llvm::enumerate(kernel.getIterArgsInit())) {
      unsigned fusedIndex = iterOffset + index;
      if (fusedIndex >= kernelIterArgArgs.size())
        return reject(second,
                      "post-Neura sibling fusion lost an iter-arg block "
                      "mapping");
      kernelMapping.map(
          kernel.getBody().front().getArgument(kernel.getInputs().size() +
                                               index),
          kernelIterArgArgs[fusedIndex]);
    }
    return success();
  };
  if (failed(mapKernelBlockArgs(firstKernel, firstTaskMapping,
                                firstKernelMapping, 0)) ||
      failed(mapKernelBlockArgs(
          secondKernel, secondTaskMapping, secondKernelMapping,
          firstKernel.getIterArgsInit().size()))) {
    fused.erase();
    return failure();
  }
  for (auto [index, result] : llvm::enumerate(firstKernel.getResults()))
    firstTaskMapping.map(result, fusedKernel.getResult(index));
  unsigned secondResultOffset = firstKernel.getNumResults();
  for (auto [index, result] : llvm::enumerate(secondKernel.getResults()))
    secondTaskMapping.map(result, fusedKernel.getResult(secondResultOffset +
                                                         index));

  auto remapFoldedInputs = [&](Operation *source, Operation *cloned,
                               neura::KernelOp sourceKernel,
                               IRMapping &taskMapping,
                               unsigned iterArgOffset) -> LogicalResult {
    for (NamedAttribute attribute : source->getAttrs()) {
      Attribute raw = attribute.getValue();
      auto inputIndex = parseKernelInputReference(raw);
      auto iterArgIndex = parseKernelIterArgReference(raw);
      bool mentionsInput = isFoldedKernelReference(raw, "%input");
      bool mentionsIterArg =
          isFoldedKernelReference(raw, "%iter_arg_init");
      if (mentionsInput && !inputIndex)
        return failure();
      if (mentionsIterArg && !iterArgIndex)
        return failure();
      if (!inputIndex && !iterArgIndex)
        continue;
      if (inputIndex) {
        if (*inputIndex >= sourceKernel.getInputs().size())
          return failure();
        Value mappedOuter = mapBodyValue(
            taskMapping, sourceKernel.getInputs()[*inputIndex]);
        auto found = kernelInputArgs.find(mappedOuter);
        if (found == kernelInputArgs.end())
          return failure();
        cloned->setAttr(
            attribute.getName(),
            builder.getStringAttr(
                (Twine("%input") +
                 Twine(std::distance(kernelInputs.begin(),
                                     llvm::find(kernelInputs, mappedOuter))))
                    .str()));
        continue;
      }
      if (*iterArgIndex >= sourceKernel.getIterArgsInit().size())
        return failure();
      unsigned fusedIterArg = iterArgOffset + *iterArgIndex;
      if (fusedIterArg >= kernelIterArgs.size())
        return failure();
      cloned->setAttr(
          attribute.getName(),
          builder.getStringAttr((Twine("%iter_arg_init") +
                                 Twine(fusedIterArg))
                                    .str()));
    }
    return success();
  };

  SmallVector<Value> firstCounterValues;
  OpBuilder kernelBuilder = OpBuilder::atBlockEnd(kernelBody);
  unsigned firstCounterIndex = 0;
  for (Operation &operation : firstKernel.getBody().front()) {
    if (isa<neura::YieldOp>(&operation))
      continue;
    Operation *cloned = kernelBuilder.clone(operation, firstKernelMapping);
    if (auto counter = dyn_cast<neura::CounterOp>(&operation)) {
      if (firstCounterIndex >= plan.firstKernelCounters.size()) {
        fused.erase();
        return reject(second, "post-Neura sibling fusion has invalid kernel counter id");
      }
      Value mapped =
          firstKernelMapping.lookupOrNull(counter.getCurrentIndex());
      if (!mapped) {
        fused.erase();
        return reject(second, "post-Neura sibling fusion lost a kernel counter");
      }
      firstCounterValues.push_back(mapped);
      ++firstCounterIndex;
    }
    if (failed(remapFoldedInputs(&operation, cloned, firstKernel,
                                 firstTaskMapping, 0))) {
      fused.erase();
      return reject(second, "post-Neura sibling fusion cannot remap first folded input");
    }
  }
  if (firstCounterValues.size() != plan.firstKernelCounters.size()) {
    fused.erase();
    return reject(second, "post-Neura sibling fusion lost the first counter chain");
  }
  for (auto [index, counter] : llvm::enumerate(plan.secondKernelCounters)) {
    if (index >= plan.firstKernelCounters.size() ||
        counter->getNumOperands() !=
            plan.firstKernelCounters[index]->getNumOperands()) {
      fused.erase();
      return reject(second,
                    "post-Neura sibling fusion has mismatched counter "
                    "operands");
    }
    neura::CounterOp secondCounter = counter;
    secondKernelMapping.map(secondCounter.getCurrentIndex(),
                            firstCounterValues[index]);
    neura::CounterOp firstCounter = plan.firstKernelCounters[index];
    for (auto [operandIndex, secondOperand] :
         llvm::enumerate(secondCounter->getOperands())) {
      Value firstOperand = firstCounter->getOperand(operandIndex);
      auto firstBound = compileTimeIndex(firstOperand, first, firstKernel);
      auto secondBound =
          compileTimeIndex(secondOperand, second, secondKernel);
      if (!firstBound || !secondBound || *firstBound != *secondBound) {
        fused.erase();
        return reject(second,
                      "post-Neura sibling fusion requires equal counter "
                      "operand bounds");
      }
      Value mappedOperand =
          firstKernelMapping.lookupOrNull(firstOperand);
      if (!mappedOperand)
        mappedOperand = firstTaskMapping.lookupOrNull(firstOperand);
      if (!mappedOperand) {
        fused.erase();
        return reject(second,
                      "post-Neura sibling fusion cannot map shared counter "
                      "operand");
      }
      secondKernelMapping.map(secondOperand, mappedOperand);
    }
  }
  for (Operation &operation : secondKernel.getBody().front()) {
    if (isa<neura::YieldOp>(&operation) || isa<neura::CounterOp>(&operation))
      continue;
    Operation *cloned = kernelBuilder.clone(operation, secondKernelMapping);
    if (failed(remapFoldedInputs(&operation, cloned, secondKernel,
                                 secondTaskMapping,
                                 firstKernel.getIterArgsInit().size()))) {
      fused.erase();
      return reject(second, "post-Neura sibling fusion cannot remap second folded input");
    }
  }
  kernelBuilder.setInsertionPointToEnd(kernelBody);
  SmallVector<Value> iterArgsNext;
  SmallVector<Value> kernelResults;
  auto appendKernelYield = [&](neura::KernelOp kernel,
                               IRMapping &mapping) -> LogicalResult {
    auto yield = dyn_cast<neura::YieldOp>(kernel.getBody().front().getTerminator());
    if (!yield)
      return failure();
    if (!yield.getIterArgsNext().empty() &&
        yield.getIterArgsNext().size() != kernel.getIterArgsInit().size())
      return failure();
    if (!yield.getResults().empty() &&
        yield.getResults().size() != kernel.getNumResults())
      return failure();
    auto mapYieldValue = [&](Value value) -> FailureOr<Value> {
      if (Value mapped = mapping.lookupOrNull(value))
        return mapped;
      // A missing mapping for a source-kernel block argument or a value
      // defined anywhere inside that kernel would retain an SSA value owned
      // by the task being erased.  Fail closed instead of using
      // IRMapping::lookupOrDefault, which silently carries that stale value
      // into the fused kernel.  Values defined outside the source kernel are
      // intentionally allowed to remain external operands.
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (argument.getOwner()->getParentOp() == kernel.getOperation())
          return failure();
      } else if (Operation *definition = value.getDefiningOp()) {
        for (Operation *owner = definition; owner;
             owner = owner->getParentOp())
          if (owner == kernel.getOperation())
            return failure();
      }
      return value;
    };
    for (Value value : yield.getIterArgsNext()) {
      FailureOr<Value> mapped = mapYieldValue(value);
      if (failed(mapped))
        return failure();
      iterArgsNext.push_back(*mapped);
    }
    for (Value value : yield.getResults()) {
      FailureOr<Value> mapped = mapYieldValue(value);
      if (failed(mapped))
        return failure();
      kernelResults.push_back(*mapped);
    }
    return success();
  };
  if (failed(appendKernelYield(firstKernel, firstKernelMapping)) ||
      failed(appendKernelYield(secondKernel, secondKernelMapping))) {
    fused.erase();
    return reject(second,
                  "post-Neura sibling fusion cannot preserve kernel yield "
                  "operands");
  }
  kernelBuilder.create<neura::YieldOp>(second.getLoc(), iterArgsNext,
                                        kernelResults);

  // Clone shell users of kernel results after the fused kernel exists.  The
  // task mappings now contain the corresponding fused result values.
  for (Operation *operation : deferredFirst)
    bodyBuilder.clone(*operation, firstTaskMapping);
  for (Operation *operation : deferredSecond)
    bodyBuilder.clone(*operation, secondTaskMapping);

  bodyBuilder.setInsertionPointToEnd(body);
  SmallVector<Value> writeYields;
  unsigned writeBase = fusedReads.size();
  for (unsigned index = 0; index < fusedWrites.size(); ++index)
    writeYields.push_back(body->getArgument(writeBase + index));
  bodyBuilder.create<TaskflowYieldOp>(second.getLoc(), ValueRange{}, writeYields,
                                      ValueRange{});
  for (NamedAttribute attribute : second->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name == "operandSegmentSizes" || name == "resultSegmentSizes" ||
        name == "task_name" || name.starts_with("amoeba.neura.fusion."))
      continue;
    fused->setAttr(attribute.getName(), attribute.getValue());
  }
  fused->setAttr("task_name", builder.getStringAttr(fusedName));
  fused->setAttr(kRewriteAttr,
                 builder.getStringAttr("post-neura-sibling-fusion"));
  fused->setAttr("amoeba.neura.fusion.mode",
                 builder.getStringAttr("sibling"));
  fused->setAttr("amoeba.neura.fusion.eliminated_stores",
                 builder.getI64IntegerAttr(0));
  fused->setAttr("amoeba.neura.fusion.eliminated_loads",
                 builder.getI64IntegerAttr(0));
  fused->setAttr("amoeba.neura.fusion.sibling_first",
                 builder.getStringAttr(first.getTaskName()));
  fused->setAttr("amoeba.neura.fusion.sibling_second",
                 builder.getStringAttr(second.getTaskName()));
  return fused;
}

static LogicalResult fusePostNeuraSiblingPair(ModuleOp module,
                                                StringRef firstName,
                                                StringRef secondName) {
  FailureOr<NeuraSiblingFusionPlan> plan =
      analyzePostNeuraSiblingFusion(module, firstName, secondName);
  if (failed(plan))
    return failure();
  func::FuncOp function = plan->first->getParentOfType<func::FuncOp>();
  if (!function || function != plan->second->getParentOfType<func::FuncOp>())
    return reject(plan->second, "post-Neura sibling fusion requires one function");
  SemanticIncomingSnapshot semanticSnapshot;
  if (failed(collectSemanticIncomingSnapshot(function, semanticSnapshot)))
    return failure();
  std::string firstSourceName = plan->first.getTaskName().str();
  std::string secondSourceName = plan->second.getTaskName().str();
  std::string fusedName =
      (Twine(firstName) + ".fuse." + Twine(secondName)).str();
  bool collision = false;
  function.walk([&](TaskflowTaskOp task) {
    collision |= task != plan->first && task != plan->second &&
                 task.getTaskName() == fusedName;
  });
  if (collision)
    return reject(plan->second,
                  "post-Neura sibling fusion-derived task name collides with "
                  "an existing task");
  OpBuilder builder(plan->second);
  FailureOr<TaskflowTaskOp> fused =
      createPostNeuraSiblingTask(*plan, builder, fusedName);
  if (failed(fused))
    return failure();
  unsigned firstWrites = plan->first.getDoneWrites().size();
  for (auto [oldValue, newValue] :
       llvm::zip(plan->first.getDoneWrites(), fused->getDoneWrites()))
    oldValue.replaceAllUsesWith(newValue);
  for (auto [oldValue, newValue] : llvm::zip(
           plan->second.getDoneWrites(),
           fused->getDoneWrites().drop_front(firstWrites)))
    oldValue.replaceAllUsesWith(newValue);
  plan->second.getBody().dropAllReferences();
  plan->second.erase();
  plan->first.getBody().dropAllReferences();
  plan->first.erase();
  std::map<std::string, std::vector<std::string>> semanticOrigins;
  semanticOrigins[fusedName] = {firstSourceName, secondSourceName};
  if (failed(remapSemanticIncomingEdges(function, semanticSnapshot,
                                         semanticOrigins)))
    return failure();
  return success();
}

static LogicalResult fusePostNeuraPair(ModuleOp module, StringRef producerName,
                                        StringRef consumerName,
                                        StringRef mode) {
  FailureOr<NeuraFusionPlan> plan =
      analyzePostNeuraFusion(module, producerName, consumerName, mode);
  if (failed(plan))
    return failure();
  func::FuncOp function = plan->producer->getParentOfType<func::FuncOp>();
  if (!function || function != plan->consumer->getParentOfType<func::FuncOp>())
    return reject(plan->consumer, "post-Neura fusion requires one function");
  SemanticIncomingSnapshot semanticSnapshot;
  if (failed(collectSemanticIncomingSnapshot(function, semanticSnapshot)))
    return failure();
  std::string producerSourceName = plan->producer.getTaskName().str();
  std::string consumerSourceName = plan->consumer.getTaskName().str();
  std::string fusedName =
      (Twine(producerName) + ".fuse." + Twine(consumerName)).str();
  bool collision = false;
  function.walk([&](TaskflowTaskOp task) {
    collision |= task != plan->producer && task != plan->consumer &&
                 task.getTaskName() == fusedName;
  });
  if (collision)
    return reject(plan->consumer, "post-Neura fusion-derived task name collides "
                                 "with an existing task");
  OpBuilder builder(plan->consumer);
  FailureOr<TaskflowTaskOp> fused =
      createPostNeuraFusedTask(*plan, builder, fusedName);
  if (failed(fused))
    return failure();
  unsigned producerWrites = plan->producer.getDoneWrites().size();
  if (!plan->forwarded) {
    for (auto [oldValue, newValue] :
         llvm::zip(plan->producer.getDoneWrites(), fused->getDoneWrites()))
      oldValue.replaceAllUsesWith(newValue);
    for (auto [oldValue, newValue] : llvm::zip(
             plan->consumer.getDoneWrites(),
             fused->getDoneWrites().drop_front(producerWrites)))
      oldValue.replaceAllUsesWith(newValue);
  } else {
    for (auto [oldValue, newValue] :
         llvm::zip(plan->consumer.getDoneWrites(), fused->getDoneWrites()))
      oldValue.replaceAllUsesWith(newValue);
  }
  // The erased consumer can contain a shell expression used by its kernel.
  // Drop every nested operand reference before Region destruction.
  plan->consumer.getBody().dropAllReferences();
  plan->consumer.erase();
  for (TaskflowChannelOp channel : plan->channels)
    if (channel && channel->use_empty())
      channel.erase();
  plan->producer.getBody().dropAllReferences();
  plan->producer.erase();
  std::map<std::string, std::vector<std::string>> semanticOrigins;
  semanticOrigins[fusedName] = {producerSourceName, consumerSourceName};
  if (failed(remapSemanticIncomingEdges(function, semanticSnapshot,
                                         semanticOrigins)))
    return failure();
  return success();
}

struct MaterializeNeuraJointRewritePass
    : public PassWrapper<MaterializeNeuraJointRewritePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializeNeuraJointRewritePass)

  MaterializeNeuraJointRewritePass() = default;
  MaterializeNeuraJointRewritePass(const MaterializeNeuraJointRewritePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "materialize-neura-joint-rewrite";
  }
  StringRef getDescription() const override {
    return "Materialize post-Neura M/N tiling or proven producer-consumer/sibling fusion";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, neura::NeuraDialect,
                    TaskflowDialect>();
  }

  Option<std::string> taskName{
      *this, "task-name", llvm::cl::desc("Exact Taskflow task name"),
      llvm::cl::init("")};
  Option<int64_t> tileAxis{
      *this, "tile-axis", llvm::cl::desc("Static M/N counter axis (0 or 1)"),
      llvm::cl::init(-1)};
  Option<int64_t> tileFactor{
      *this, "tile-factor", llvm::cl::desc("Number of disjoint domain parts"),
      llvm::cl::init(0)};
  Option<std::string> fusionMode{
      *this, "fusion-mode", llvm::cl::desc("Post-Neura fusion mode"),
      llvm::cl::init("none")};
  Option<std::string> firstTaskName{
      *this, "first-task-name", llvm::cl::desc("Producer task name"),
      llvm::cl::init("")};
  Option<std::string> secondTaskName{
      *this, "second-task-name", llvm::cl::desc("Consumer task name"),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (fusionMode != "none") {
      if (firstTaskName.empty() || secondTaskName.empty()) {
        module.emitError()
            << "post-Neura fusion requires --first-task-name and "
               "--second-task-name";
        signalPassFailure();
        return;
      }
      if (fusionMode == "sibling") {
        if (failed(fusePostNeuraSiblingPair(module, firstTaskName,
                                            secondTaskName)))
          signalPassFailure();
      } else if (failed(fusePostNeuraPair(module, firstTaskName,
                                          secondTaskName, fusionMode))) {
        signalPassFailure();
      }
      return;
    }
    if (taskName.empty()) {
      module.emitError()
          << "post-Neura joint rewrite requires --task-name";
      signalPassFailure();
      return;
    }
    SmallVector<TaskflowTaskOp> matches;
    module.walk([&](TaskflowTaskOp task) {
      if (task.getTaskName() == taskName)
        matches.push_back(task);
    });
    if (matches.size() != 1) {
      module.emitError()
          << "post-Neura joint rewrite task name must select exactly one "
             "task (found "
          << matches.size() << ")";
      signalPassFailure();
      return;
    }
    FailureOr<NeuraTilePlan> plan =
        analyzeTask(matches.front(), tileAxis, tileFactor);
    if (failed(plan) || failed(rewriteTask(*plan,
                                            matches.front()
                                                ->getParentOfType<func::FuncOp>())))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createMaterializeNeuraJointRewritePass() {
  return std::make_unique<MaterializeNeuraJointRewritePass>();
}
} // namespace mlir::amoeba::neura
