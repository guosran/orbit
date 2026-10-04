//===- ExtractJointTaskGraphFactsPass.cpp ---------------------*- C++ -*-===//
//
// Emits deterministic Taskflow graph facts and conservative legal rewrite
// actions. The graph-variant driver invokes this pass again after every
// rewrite; no legality decision is carried across a changed graph.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "ClosureLegalityPreflight.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskGraphRewriteLegality.h"
#include "NeuraDialect/NeuraOps.h"
#include "Backend/Neura/Orchestration/JointScheduling/ReplicaOutputCoordinateProof.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;
namespace json = llvm::json;

namespace {

constexpr StringLiteral kFactsSchema = "amoeba-joint-task-graph-facts-v1";

struct EdgeFact {
  unsigned source = 0;
  unsigned target = 0;
  std::string kind;
  std::string producerSegment;
  uint32_t producerIndex = 0;
  std::string consumerSegment;
  uint32_t consumerIndex = 0;
  std::string origin;
  std::string scope;
  std::string semanticRole;
  std::string semanticScope;
  SmallVector<int64_t> regionLower;
  SmallVector<int64_t> regionUpper;
  std::optional<uint64_t> payloadBits;

  bool operator<(const EdgeFact &other) const {
    return std::tie(source, target, kind, producerSegment, producerIndex,
                    consumerSegment, consumerIndex, origin, scope, semanticRole,
                    semanticScope, regionLower, regionUpper, payloadBits) <
           std::tie(other.source, other.target, other.kind,
                    other.producerSegment, other.producerIndex,
                    other.consumerSegment, other.consumerIndex, other.origin,
                    other.scope, other.semanticRole, other.semanticScope,
                    other.regionLower, other.regionUpper, other.payloadBits);
  }
  bool operator==(const EdgeFact &other) const {
    return source == other.source && target == other.target &&
           kind == other.kind && producerSegment == other.producerSegment &&
           producerIndex == other.producerIndex &&
           consumerSegment == other.consumerSegment &&
           consumerIndex == other.consumerIndex && origin == other.origin &&
           scope == other.scope && semanticRole == other.semanticRole &&
           semanticScope == other.semanticScope &&
           regionLower == other.regionLower &&
           regionUpper == other.regionUpper && payloadBits == other.payloadBits;
  }
};

constexpr StringLiteral kSemanticIncomingEdgesAttr =
    "amoeba.semantic.incoming_edges";

static std::optional<int64_t>
constantIndexImpl(Value value, DenseSet<Value> &visiting) {
  if (!value || !visiting.insert(value).second)
    return std::nullopt;

  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();

  // Application frontends keep some proven bounds as a sum of a forwarded
  // compile-time value and a literal (for example Radar's 64 + (-2)). Only
  // fold when both operands are independently proven and the result cannot
  // wrap in the operation's integer type.
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = constantIndexImpl(add.getLhs(), visiting);
    auto rhs = constantIndexImpl(add.getRhs(), visiting);
    if (!lhs || !rhs ||
        (*rhs > 0 && *lhs > std::numeric_limits<int64_t>::max() - *rhs) ||
        (*rhs < 0 && *lhs < std::numeric_limits<int64_t>::min() - *rhs))
      return std::nullopt;
    int64_t result = *lhs + *rhs;
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

  // A Taskflow region argument is an ABI forwarding edge: its operand in the
  // task op is the value that was supplied by the caller. Following this edge
  // proves LLaMA's symbol-dynamic counter is bounded by the compile-time
  // constant %c320.
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *parent = argument.getOwner()->getParentOp();
    if (auto kernel = dyn_cast<mlir::neura::KernelOp>(parent)) {
      if (argument.getOwner() == &kernel.getBody().front() &&
          argument.getArgNumber() < kernel.getInputs().size())
        return constantIndexImpl(kernel.getInputs()[argument.getArgNumber()],
                                 visiting);
      return std::nullopt;
    }
    if (auto task = dyn_cast<TaskflowTaskOp>(parent)) {
      if (argument.getArgNumber() < task->getNumOperands())
        return constantIndexImpl(task->getOperand(argument.getArgNumber()),
                                 visiting);
      return std::nullopt;
    }
    if (auto function = dyn_cast<func::FuncOp>(parent)) {
      if (argument.getOwner() != &function.getBody().front())
        return std::nullopt;
      std::string attribute =
          "amoeba.static_bound.arg." +
          std::to_string(argument.getArgNumber());
      if (auto bound = function->getAttrOfType<IntegerAttr>(attribute))
        return bound.getInt() > 0 ? std::optional<int64_t>(bound.getInt())
                                  : std::nullopt;
    }
    return std::nullopt;
  }

  // Only transparent integer casts are followed beyond the proven add above.
  if (Operation *definition = value.getDefiningOp()) {
    StringRef name = definition->getName().getStringRef();
    if ((name == "arith.index_cast" || name == "arith.extsi" ||
         name == "arith.extui" || name == "arith.trunci") &&
        definition->getNumOperands() == 1)
      return constantIndexImpl(definition->getOperand(0), visiting);
  }
  return std::nullopt;
}

static std::optional<int64_t> constantIndex(Value value) {
  DenseSet<Value> visiting;
  return constantIndexImpl(value, visiting);
}

static FailureOr<SmallVector<std::pair<int64_t, int64_t>>>
loopDomain(TaskflowTaskOp task) {
  SmallVector<std::pair<int64_t, int64_t>> domain;
  SmallVector<TaskflowCounterOp> counters;
  task.walk([&](TaskflowCounterOp counter) { counters.push_back(counter); });
  if (!counters.empty()) {
    for (TaskflowCounterOp counter : counters) {
      std::optional<int64_t> lower = constantIndex(counter.getLowerBound());
      std::optional<int64_t> upper = constantIndex(counter.getUpperBound());
      std::optional<int64_t> step = constantIndex(counter.getStep());
      if (!lower || !upper || !step || *step != 1 || *upper <= *lower)
        return failure();
      domain.push_back({*lower, *upper});
    }
    return domain;
  }
  task.walk([&](scf::ForOp loop) {
    std::optional<int64_t> lower = constantIndex(loop.getLowerBound());
    std::optional<int64_t> upper = constantIndex(loop.getUpperBound());
    std::optional<int64_t> step = constantIndex(loop.getStep());
    if (lower && upper && step && *step == 1 && *upper > *lower)
      domain.push_back({*lower, *upper});
  });
  if (domain.empty()) {
    task.walk([&](affine::AffineForOp loop) {
      if (loop.hasConstantBounds() && loop.getStepAsInt() == 1 &&
          loop.getConstantUpperBound() > loop.getConstantLowerBound())
        domain.push_back(
            {loop.getConstantLowerBound(), loop.getConstantUpperBound()});
    });
  }
  if (domain.empty())
    return failure();
  return domain;
}

struct KReductionProof {
  bool potential = false;
  std::optional<int64_t> extent;
  std::string unprovenReason;
  bool provenIllegal = false;
};

enum class MemoryFeedbackKind { None, Canonical, OrderedCrossCell };

// Taskflow's read/write lists may contain task-result values rather than the
// original memref SSA value. Resolve that one level of provenance before
// classifying memory feedback so a distinct result token for the same backing
// buffer cannot be mistaken for an independent input.
static bool representsOriginalStorage(Value value, Value storage) {
  if (value == storage)
    return true;
  auto producer = value.getDefiningOp<TaskflowTaskOp>();
  if (!producer)
    return false;
  auto doneWrites = producer.getDoneWrites();
  auto originalWrites = producer.getOriginalWriteMemrefs();
  if (doneWrites.size() != originalWrites.size())
    return false;
  for (auto [doneWrite, originalWrite] :
       llvm::zip(doneWrites, originalWrites))
    if (doneWrite == value && originalWrite == storage)
      return true;
  return false;
}

static bool taskBodyValueRepresentsStorage(Value value, TaskflowTaskOp task,
                                           Value storage) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &task.getBody().front())
    return false;
  unsigned index = argument.getArgNumber();
  if (index < task.getWillReads().size())
    return representsOriginalStorage(task.getWillReads()[index], storage);
  index -= task.getWillReads().size();
  if (index < task.getWillWrites().size())
    return representsOriginalStorage(task.getWillWrites()[index], storage);
  return false;
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

static bool valueReferencesKernelInput(Value value, unsigned inputIndex,
                                       neura::KernelOp kernel,
                                       DenseSet<Value> &visiting,
                                       unsigned depth = 0) {
  if (!value || depth >= 24 || !visiting.insert(value).second)
    return false;
  bool result = false;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    result = argument.getOwner() == &kernel.getBody().front() &&
             argument.getArgNumber() == inputIndex;
  } else if (Operation *definition = value.getDefiningOp()) {
    StringRef name = definition->getName().getStringRef();
    // Only a Neura constant may carry a folded kernel-input identity. An
    // arbitrary operation's `value` attribute is metadata, not provenance.
    if (name == "neura.constant") {
      if (auto reference =
              parseKernelInputReference(definition->getAttr("value")))
        result = *reference == inputIndex;
    } else if (name == "neura.grant_predicate" ||
               name == "neura.ctrl_mov" || name == "neura.cast") {
      result = definition->getNumOperands() >= 1 &&
               valueReferencesKernelInput(definition->getOperand(0),
                                          inputIndex, kernel, visiting,
                                          depth + 1);
    } else if (name == "neura.phi" || name == "neura.sel") {
      unsigned first = name == "neura.sel" ? 1u : 0u;
      result = first < definition->getNumOperands();
      // A phi/select proves storage provenance only when every data arm
      // resolves to this same kernel input. Copy the path state per arm so a
      // shared arm is not mistaken for a cycle on a later arm.
      for (unsigned operand = first;
           result && operand < definition->getNumOperands(); ++operand) {
        DenseSet<Value> armVisiting = visiting;
        result = valueReferencesKernelInput(
            definition->getOperand(operand), inputIndex, kernel,
            armVisiting, depth + 1);
      }
    }
  }
  visiting.erase(value);
  return result;
}

static bool valueReferencesKernelInput(Value value, unsigned inputIndex,
                                       neura::KernelOp kernel) {
  DenseSet<Value> visited;
  return valueReferencesKernelInput(value, inputIndex, kernel, visited);
}

static bool reachesValue(Value value, Value target, DenseSet<Value> &visited,
                         unsigned depth = 0) {
  if (value == target)
    return true;
  if (!value || depth >= 12 || !visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;

  // Predicated index values carry a data operand and a control predicate.
  // Traversing every operand would make an M index appear to reach K through
  // the predicate's icmp. Only follow data operands in the source-owned
  // transparent index forwarding operations.
  StringRef name = definition->getName().getStringRef();
  if (name == "neura.grant_predicate" || name == "neura.ctrl_mov" ||
      name == "neura.cast") {
    return definition->getNumOperands() >= 1 &&
           reachesValue(definition->getOperand(0), target, visited,
                        depth + 1);
  }
  if (name == "neura.phi") {
    for (Value operand : definition->getOperands())
      if (reachesValue(operand, target, visited, depth + 1))
        return true;
  }
  if (name == "neura.sel") {
    for (unsigned index = 1; index < definition->getNumOperands(); ++index)
      if (reachesValue(definition->getOperand(index), target, visited,
                       depth + 1))
        return true;
  }
  return false;
}

static std::optional<int64_t>
sourceConstantIndex(Value value, DenseSet<Value> &visited,
                    unsigned depth = 0) {
  if (!value || depth >= 16 || !visited.insert(value).second)
    return std::nullopt;
  if (auto literal = constantIndex(value))
    return literal;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (name == "neura.constant") {
    Attribute constant = definition->getAttr("value");
    // Integer/index constants are source-owned fixed indices. Pointer-like
    // constants such as %input1 are represented by StringAttr and are not
    // accepted here.
    if (auto integer = dyn_cast_or_null<IntegerAttr>(constant))
      return integer.getInt();
    return std::nullopt;
  }
  if (name == "neura.grant_predicate" || name == "neura.ctrl_mov")
    return definition->getNumOperands() >= 1
               ? sourceConstantIndex(definition->getOperand(0), visited,
                                     depth + 1)
               : std::nullopt;
  if (name == "neura.cast")
    return definition->getNumOperands() == 1
               ? sourceConstantIndex(definition->getOperand(0), visited,
                                     depth + 1)
               : std::nullopt;
  if (name == "neura.phi" || name == "neura.sel") {
    if (definition->getNumOperands() < (name == "neura.sel" ? 3u : 1u))
      return std::nullopt;
    unsigned first = name == "neura.sel" ? 1u : 0u;
    std::optional<int64_t> common;
    for (unsigned index = first; index < definition->getNumOperands(); ++index) {
      // Each phi arm is an independent forwarding path. Reuse the parent
      // cycle set, but do not let visiting a shared constant on one arm make
      // the same constant on another arm look cyclic.
      DenseSet<Value> branchVisited = visited;
      auto branch = sourceConstantIndex(definition->getOperand(index),
                                        branchVisited, depth + 1);
      // A predicated constant index is proven only when every data arm is the
      // same integer. Merely finding a constant arm is existential and would
      // incorrectly accept an index that changes at runtime.
      if (!branch)
        return std::nullopt;
      if (!common)
        common = branch;
      else if (*common != *branch)
        return std::nullopt;
    }
    return common;
  }
  return std::nullopt;
}

enum class SourceIndexRole { Unknown, M, N, K, Constant };

static SourceIndexRole sourceIndexRole(Value value, Value m, Value n, Value k,
                                       DenseSet<Value> &visited,
                                       unsigned depth = 0) {
  if (!value || depth >= 16 || !visited.insert(value).second)
    return SourceIndexRole::Unknown;
  if (value == m)
    return SourceIndexRole::M;
  if (value == n)
    return SourceIndexRole::N;
  if (value == k)
    return SourceIndexRole::K;
  DenseSet<Value> constantVisited;
  if (sourceConstantIndex(value, constantVisited).has_value())
    return SourceIndexRole::Constant;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return SourceIndexRole::Unknown;
  StringRef name = definition->getName().getStringRef();
  if (name == "neura.grant_predicate" || name == "neura.ctrl_mov" ||
      name == "neura.cast") {
    if (definition->getNumOperands() < 1)
      return SourceIndexRole::Unknown;
    DenseSet<Value> operandVisited = visited;
    return sourceIndexRole(definition->getOperand(0), m, n, k,
                           operandVisited, depth + 1);
  }
  if (name == "neura.phi" || name == "neura.sel") {
    if (definition->getNumOperands() < (name == "neura.sel" ? 3u : 1u))
      return SourceIndexRole::Unknown;
    unsigned first = name == "neura.sel" ? 1u : 0u;
    SourceIndexRole common = SourceIndexRole::Unknown;
    for (unsigned index = first; index < definition->getNumOperands(); ++index) {
      DenseSet<Value> branchVisited = visited;
      SourceIndexRole branch = sourceIndexRole(
          definition->getOperand(index), m, n, k, branchVisited, depth + 1);
      if (branch == SourceIndexRole::Unknown)
        return SourceIndexRole::Unknown;
      if (common == SourceIndexRole::Unknown)
        common = branch;
      else if (common != branch)
        return SourceIndexRole::Unknown;
    }
    return common;
  }
  return SourceIndexRole::Unknown;
}

static SourceIndexRole sourceIndexRole(Value value, Value m, Value n, Value k) {
  DenseSet<Value> visited;
  return sourceIndexRole(value, m, n, k, visited);
}

static bool isOutputCellIndexVector(ValueRange indices, Value m, Value n,
                                    Value k) {
  bool hasM = false;
  bool hasN = false;
  for (Value index : indices) {
    switch (sourceIndexRole(index, m, n, k)) {
    case SourceIndexRole::K:
    case SourceIndexRole::Unknown:
      return false;
    case SourceIndexRole::M:
      if (hasM)
        return false;
      hasM = true;
      continue;
    case SourceIndexRole::N:
      if (hasN)
        return false;
      hasN = true;
      continue;
    case SourceIndexRole::Constant:
      // GCN's output is a fixed feature slice, e.g. [0, M, N]. Constant
      // leading indices are harmless and are retained by the materializer.
      continue;
    }
  }
  return hasM && hasN;
}

static bool isKZeroPredicate(Value predicate, Value m, Value n, Value k) {
  Operation *definition = predicate.getDefiningOp();
  if (!definition || definition->getName().getStringRef() != "neura.icmp" ||
      definition->getNumOperands() < 1)
    return false;
  auto cmpType = definition->getAttrOfType<StringAttr>("cmpType");
  auto rhs = definition->getAttrOfType<IntegerAttr>("rhs_value");
  return cmpType && cmpType.getValue() == "eq" && rhs && rhs.getInt() == 0 &&
         sourceIndexRole(definition->getOperand(0), m, n, k) ==
             SourceIndexRole::K;
}

static bool isKZeroGuarded(Value value, Value m, Value n, Value k) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  StringRef name = definition->getName().getStringRef();
  if (name != "neura.grant_predicate" && name != "neura.ctrl_mov")
    return false;
  return definition->getNumOperands() >= 2 &&
         isKZeroPredicate(definition->getOperand(1), m, n, k);
}

static bool isOrderedCrossCellIndexVector(ValueRange indices, Value m,
                                           Value n, Value k) {
  bool hasK = false;
  bool hasOuter = false;
  for (Value index : indices) {
    switch (sourceIndexRole(index, m, n, k)) {
    case SourceIndexRole::K:
      if (hasK)
        return false;
      hasK = true;
      continue;
    case SourceIndexRole::M:
    case SourceIndexRole::N:
      if (hasOuter)
        return false;
      hasOuter = true;
      continue;
    case SourceIndexRole::Constant:
      continue;
    case SourceIndexRole::Unknown:
      return false;
    }
  }
  return hasK && hasOuter;
}

static MemoryFeedbackKind classifyMemoryFeedback(TaskflowTaskOp task) {
  if (task.getWillReads().empty() || task.getWillWrites().size() != 1 ||
      task.getBody().empty())
    return MemoryFeedbackKind::None;

  Value outputRoot = task.getWillWrites().front();
  if (task.getOriginalReadMemrefs().size() != task.getWillReads().size() ||
      task.getOriginalWriteMemrefs().size() != 1)
    return MemoryFeedbackKind::None;
  Value outputStorage = task.getOriginalWriteMemrefs().front();
  unsigned outputReadCount = 0;
  for (auto [index, read] : llvm::enumerate(task.getWillReads())) {
    // Compare both the task-level value and its declared source storage. A
    // second SSA value that resolves to the same output buffer is an alias,
    // not a second independent input, and is rejected below.
    if (read == outputRoot ||
        representsOriginalStorage(read, outputStorage) ||
        task.getOriginalReadMemrefs()[index] == outputStorage)
      ++outputReadCount;
  }
  if (outputReadCount != 1)
    return MemoryFeedbackKind::None;

  SmallVector<TaskflowCounterOp> taskCounters;
  task.walk([&](TaskflowCounterOp counter) { taskCounters.push_back(counter); });
  if (taskCounters.size() != 3)
    return MemoryFeedbackKind::None;
  for (auto [index, counter] : llvm::enumerate(taskCounters)) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    auto lower = constantIndex(counter.getLowerBound());
    auto upper = constantIndex(counter.getUpperBound());
    auto step = constantIndex(counter.getStep());
    auto hierarchy = counter->getAttrOfType<StringAttr>("counter_hierarchy");
    StringRef expectedHierarchy = index == 0 ? "root" :
                                  (index == 1 ? "relay" : "leaf");
    bool parentOK = index == 0
                        ? !counter.getParentIndex()
                        : counter.getParentIndex() ==
                              taskCounters[index - 1].getCounterIndex();
    if (!id || id.getInt() != static_cast<int64_t>(index) || !lower ||
        !upper || !step || *lower != 0 || *upper <= *lower || *step != 1 ||
        !hierarchy || hierarchy.getValue() != expectedHierarchy || !parentOK)
      return MemoryFeedbackKind::None;
  }

  SmallVector<neura::KernelOp> kernels;
  task.walk([&](neura::KernelOp kernel) { kernels.push_back(kernel); });
  if (kernels.size() != 1 || !kernels.front().getBody().hasOneBlock() ||
      !kernels.front().getIterArgsInit().empty())
    return MemoryFeedbackKind::None;
  neura::KernelOp kernel = kernels.front();
  SmallVector<neura::CounterOp> kernelCounters;
  kernel.walk([&](neura::CounterOp counter) { kernelCounters.push_back(counter); });
  if (kernelCounters.size() != 3)
    return MemoryFeedbackKind::None;
  for (auto [index, counter] : llvm::enumerate(kernelCounters)) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    // Folded attributes are only a cache of an omitted constant operand. If
    // the Neura counter still has an SSA bound operand, prove that operand
    // itself and compare it with the Taskflow counter; otherwise a stale
    // optional attribute could certify a different loop extent.
    auto resolveCounterBound = [&](Value operand, StringRef attrName)
        -> std::optional<int64_t> {
      std::optional<int64_t> actual;
      if (operand)
        actual = constantIndex(operand);

      std::optional<int64_t> folded;
      if (auto integer = counter->getAttrOfType<IntegerAttr>(attrName))
        folded = integer.getInt();
      else if (auto reference = parseKernelInputReference(
                   counter->getAttr(attrName))) {
        if (*reference < kernel.getInputs().size())
          folded = constantIndex(kernel.getInputs()[*reference]);
      }
      // A present SSA bound must resolve and agree with any folded attribute.
      // Only zero-operand predicate counters may rely on folded bounds alone.
      if (operand && !actual)
        return std::nullopt;
      if (actual && folded && *actual != *folded)
        return std::nullopt;
      return actual ? actual : folded;
    };
    std::optional<int64_t> lower =
        resolveCounterBound(counter.getLowerBound(), "lower_bound_value");
    std::optional<int64_t> upper =
        resolveCounterBound(counter.getUpperBound(), "upper_bound_value");
    std::optional<int64_t> step =
        resolveCounterBound(counter.getStep(), "step_value");
    auto hierarchy = counter.getCounterHierarchyAttr();
    StringRef expectedHierarchy = index == 0 ? "root" :
                                  (index == 1 ? "relay" : "leaf");
    auto taskUpper = constantIndex(taskCounters[index].getUpperBound());
    bool hierarchyOK = hierarchy && hierarchy.getValue() == expectedHierarchy;
    // Converted source graphs encode symbolic kernel bounds as StringAttr
    // (%inputN). Those are checked through the Taskflow ABI below when they
    // are available; numeric attributes are checked here directly.
    bool numericBoundsOK = lower && upper && step && *lower == 0 &&
                           *step == 1 && taskUpper && *upper == *taskUpper;
    auto lowerAttr = counter->getAttrOfType<IntegerAttr>("lower_bound_value");
    auto upperAttr = counter->getAttrOfType<IntegerAttr>("upper_bound_value");
    auto stepAttr = counter->getAttrOfType<IntegerAttr>("step_value");
    bool foldedAttrsMatch =
        (!lowerAttr || (lower && lowerAttr.getInt() == *lower)) &&
        (!upperAttr || (upper && upperAttr.getInt() == *upper)) &&
        (!stepAttr || (step && stepAttr.getInt() == *step));
    if (!id || id.getInt() != static_cast<int64_t>(index) || !hierarchyOK ||
        !numericBoundsOK || !foldedAttrsMatch)
      return MemoryFeedbackKind::None;
  }

  Block &taskBody = task.getBody().front();
  unsigned writeArgument = task.getWillReads().size();
  if (writeArgument >= taskBody.getNumArguments())
    return MemoryFeedbackKind::None;
  Value outputArgument = taskBody.getArgument(writeArgument);
  std::optional<unsigned> outputInput;
  unsigned outputInputAliases = 0;
  for (auto [index, input] : llvm::enumerate(kernel.getInputs()))
    if (input == outputArgument ||
        taskBodyValueRepresentsStorage(input, task, outputStorage)) {
      outputInput = index;
      ++outputInputAliases;
    }
  if (!outputInput)
    return MemoryFeedbackKind::None;
  if (outputInputAliases != 1)
    return MemoryFeedbackKind::None;

  Value m = kernelCounters[0].getCurrentIndex();
  Value n = kernelCounters[1].getCurrentIndex();
  Value k = kernelCounters[2].getCurrentIndex();
  SmallVector<neura::LoadIndexedOp> outputLoads;
  SmallVector<neura::StoreIndexedOp> outputStores;
  bool sawMultiply = false;
  bool sawSubtract = false;
  unsigned nonOutputStores = 0;
  bool invalidIndexedAccess = false;
  kernel.walk([&](Operation *operation) {
    if (invalidIndexedAccess)
      return;
    StringRef name = operation->getName().getStringRef();
    sawMultiply |= name == "neura.mul";
    sawSubtract |= name == "neura.sub";
    if (auto load = dyn_cast<neura::LoadIndexedOp>(operation)) {
      auto input = parseKernelInputReference(load->getAttr("lhs_value"));
      bool baseIsOutput =
          load.getBase() &&
          valueReferencesKernelInput(load.getBase(), *outputInput, kernel);
      if (!input || *input >= kernel.getInputs().size()) {
        invalidIndexedAccess = true;
        return;
      }
      if (*input == *outputInput)
        outputLoads.push_back(load);
      else if (baseIsOutput)
        invalidIndexedAccess = true;
      return;
    }
    if (auto store = dyn_cast<neura::StoreIndexedOp>(operation)) {
      // An explicit base and folded rhs_value provide two address identities.
      // Accepting either one would let contradictory output/input provenance
      // bypass the definite alias proof.
      if (store.getBase() && store->hasAttr("rhs_value")) {
        invalidIndexedAccess = true;
        return;
      }
      bool baseIsOutput =
          store.getBase() &&
          valueReferencesKernelInput(store.getBase(), *outputInput, kernel);
      auto input = parseKernelInputReference(store->getAttr("rhs_value"));
      // As in the materializer, an explicit-base K==0 store records its
      // value in lhs_value; the base is the authoritative destination.
      if (baseIsOutput) {
        outputStores.push_back(store);
      } else if (!input || *input >= kernel.getInputs().size()) {
        invalidIndexedAccess = true;
      } else if (*input == *outputInput) {
        outputStores.push_back(store);
      } else {
        ++nonOutputStores;
      }
    }
  });
  unsigned recurrenceStoreCount = 0;
  unsigned initialStoreCount = 0;
  for (neura::StoreIndexedOp store : outputStores) {
    Operation *valueDefinition = store.getValue().getDefiningOp();
    if (valueDefinition &&
        (valueDefinition->getName().getStringRef() == "neura.add" ||
         valueDefinition->getName().getStringRef() == "neura.sub"))
      ++recurrenceStoreCount;
    else if (isKZeroGuarded(store.getValue(), m, n, k))
      ++initialStoreCount;
  }
  bool storeShapeOK =
      (outputStores.size() == 1 && recurrenceStoreCount == 1 &&
       initialStoreCount == 0) ||
      (outputStores.size() == 2 && recurrenceStoreCount == 1 &&
       initialStoreCount == 1);
  if (invalidIndexedAccess || outputLoads.empty() || outputLoads.size() > 2 ||
      !storeShapeOK ||
      nonOutputStores > 1 || !sawMultiply)
    return MemoryFeedbackKind::None;

  bool sawMatchingAccumulator = false;
  bool sawCrossCellLoad = false;
  unsigned sameCellLoadCount = 0;
  unsigned crossCellLoadCount = 0;
  for (neura::LoadIndexedOp load : outputLoads) {
    bool sameCell = isOutputCellIndexVector(load.getIndices(), m, n, k);
    bool crossCell = outputLoads.size() == 2 &&
                     isOrderedCrossCellIndexVector(load.getIndices(), m, n, k);
    if (!sameCell && !crossCell)
      return MemoryFeedbackKind::None;
    sawCrossCellLoad |= crossCell;
    sameCellLoadCount += sameCell;
    crossCellLoadCount += crossCell;
    for (neura::StoreIndexedOp store : outputStores) {
      if (!isOutputCellIndexVector(store.getIndices(), m, n, k) ||
          load.getIndices() != store.getIndices())
        continue;
      for (Operation *user : load.getResult().getUsers()) {
        StringRef name = user->getName().getStringRef();
        if (name != "neura.add" && name != "neura.sub")
          continue;
        if (store.getValue() != user->getResult(0) ||
            user->getNumOperands() != 2)
          continue;
        unsigned accumulatorCount = 0;
        Value other;
        for (Value operand : user->getOperands()) {
          if (operand == load.getResult())
            ++accumulatorCount;
          else
            other = operand;
        }
        Operation *otherDef = other ? other.getDefiningOp() : nullptr;
        if (accumulatorCount != 1 || !otherDef ||
            otherDef->getName().getStringRef() != "neura.mul" ||
            (name == "neura.sub" && user->getOperand(0) != load.getResult()))
          continue;
        sawMatchingAccumulator = true;
      }
    }
  }
  if (!sawMatchingAccumulator || sameCellLoadCount != 1 ||
      crossCellLoadCount != (outputLoads.size() == 2 ? 1u : 0u))
    return MemoryFeedbackKind::None;

  if (outputLoads.size() == 1)
    return MemoryFeedbackKind::Canonical;
  // LU's second output load reads output[leaf, root], so it cannot be
  // replicated or reduced as an unordered partial. It is still a legal
  // sequential K state machine when the original counter order is retained.
  return sawCrossCellLoad && sawSubtract
             ? MemoryFeedbackKind::OrderedCrossCell
             : MemoryFeedbackKind::None;
}

// K actions require a reduction proof, rather than merely a third tensor
// dimension. Prefer the source-owned K_range annotation when present. For
// converted input-0 graphs, the equivalent proof is a three-counter task with
// a loop-carried kernel value; the last counter is then the K extent. Any
// unresolved candidate is reported as incomplete by the caller.
static KReductionProof proveKReduction(TaskflowTaskOp task) {
  if (auto range = task->getAttrOfType<DenseI64ArrayAttr>(
          "amoeba.semantic.K_range")) {
    if (range.size() == 2 && range[0] < range[1])
      return {true, range[1] - range[0]};
    return {true, std::nullopt};
  }
  if (auto range = task->getAttrOfType<DenseI64ArrayAttr>(
          "amoeba.semantic.k_range")) {
    if (range.size() == 2 && range[0] < range[1])
      return {true, range[1] - range[0]};
    return {true, std::nullopt};
  }

  bool hasLoopCarriedKernelValue = false;
  task.walk([&](mlir::neura::KernelOp kernel) {
    hasLoopCarriedKernelValue |= !kernel.getIterArgsInit().empty();
  });
  if (hasLoopCarriedKernelValue) {
    FailureOr<SmallVector<std::pair<int64_t, int64_t>>> domain =
        loopDomain(task);
    SmallVector<TaskflowCounterOp> counters;
    task.walk([&](TaskflowCounterOp counter) { counters.push_back(counter); });
    if (succeeded(domain) && counters.size() >= 3 && domain->size() >= 3) {
      int64_t lower = domain->back().first;
      int64_t upper = domain->back().second;
      if (upper > lower)
        return {true, upper - lower};
    }
    // A loop-carried kernel with a dynamic or non-GEMM counter chain is a
    // possible reduction, but it has no proof that can be used for K search.
    if (counters.size() >= 3)
      return {true, std::nullopt};
  }

  bool sawScfReduction = false;
  std::optional<int64_t> scfExtent;
  bool scfExtentConflict = false;
  task.walk([&](scf::ForOp loop) {
    if (loop.getInitArgs().empty())
      return;
    sawScfReduction = true;
    std::optional<int64_t> lower = constantIndex(loop.getLowerBound());
    std::optional<int64_t> upper = constantIndex(loop.getUpperBound());
    std::optional<int64_t> step = constantIndex(loop.getStep());
    if (!lower || !upper || !step || *step != 1 || *upper <= *lower) {
      scfExtentConflict = true;
      return;
    }
    int64_t extent = *upper - *lower;
    if (scfExtent && *scfExtent != extent)
      scfExtentConflict = true;
    scfExtent = extent;
  });
  if (sawScfReduction)
    return {true, scfExtentConflict ? std::nullopt : scfExtent};

  // A three-counter kernel can accumulate through a memory location rather
  // than an SSA iter arg. Recognize only the source-owned indexed recurrence;
  // unrelated read/write feedback remains incomplete. Cross-cell recurrences
  // are intentionally not admitted: K materialization clones the complete
  // M/N body per K block, which changes source order for a value whose address
  // contains both K and an outer counter. A future ordered schedule proof may
  // add this case back; the generic K proof cannot.
  SmallVector<TaskflowCounterOp> counters;
  task.walk([&](TaskflowCounterOp counter) { counters.push_back(counter); });
  if (counters.size() >= 3)
    for (Value read : task.getWillReads())
      for (Value write : task.getWillWrites())
        if (read == write) {
          MemoryFeedbackKind kind = classifyMemoryFeedback(task);
          if (kind != MemoryFeedbackKind::None) {
            if (kind == MemoryFeedbackKind::OrderedCrossCell)
              return {true, std::nullopt,
                      "memory-carried-cross-cell-feedback-order-not-preserved",
                      true};
            FailureOr<SmallVector<std::pair<int64_t, int64_t>>> domain =
                loopDomain(task);
            if (succeeded(domain) && domain->size() >= 3) {
              auto [lower, upper] = domain->back();
              if (upper > lower)
                return {true, upper - lower};
            }
            return {true, std::nullopt,
                    "memory-feedback-K-bound-unproven"};
          }
          return {true, std::nullopt,
                  "memory-carried-read-write-feedback-needs-K-proof"};
        }
  return {};
}

static Value peelChannel(Value value) {
  while (auto channel = value.getDefiningOp<TaskflowChannelOp>())
    value = channel.getSource();
  return value;
}

static void
sourceTaskIndices(Value value, const DenseMap<Operation *, unsigned> &indices,
                  SmallVectorImpl<std::pair<unsigned, std::string>> &sources,
                  DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  if (auto channel = value.getDefiningOp<TaskflowChannelOp>()) {
    sourceTaskIndices(channel.getSource(), indices, sources, visited);
    return;
  }
  if (auto join = value.getDefiningOp<TaskflowReadCompletionJoinOp>()) {
    for (Value state : join.getTileStates())
      sourceTaskIndices(state, indices, sources, visited);
    return;
  }
  if (auto join = value.getDefiningOp<TaskflowJoinOp>()) {
    for (Value state : join.getTileStates())
      sourceTaskIndices(state, indices, sources, visited);
    return;
  }
  auto task = value.getDefiningOp<TaskflowTaskOp>();
  if (!task)
    return;
  auto found = indices.find(task.getOperation());
  if (found == indices.end())
    return;
  std::string kind;
  if (llvm::is_contained(task.getDoneWrites(), value))
    kind = "raw";
  else if (llvm::is_contained(task.getDoneReads(), value))
    kind = "war";
  else if (llvm::is_contained(task.getValueOutputs(), value))
    kind = "value";
  else
    return;
  sources.push_back({found->second, std::move(kind)});
}

// Mixed Taskflow/Neura facts deliberately avoid candidate identity and
// cost-query requirements. Counter extents are emitted only when their SSA
// provenance reaches a constant or an explicit static-bound function
// attribute; unresolved symbol-dynamic bounds remain unknown.
static FailureOr<SmallVector<TaskMetadata>>
collectFactOnlyMetadata(func::FuncOp function, std::string &error) {
  SmallVector<TaskMetadata> tasks;
  llvm::StringMap<bool> names;
  function.walk([&](TaskflowTaskOp task) {
    std::string name = task.getTaskName().str();
    if (!names.try_emplace(name, true).second) {
      error = "duplicate task name " + name;
      return WalkResult::interrupt();
    }
    TaskMetadata metadata;
    metadata.op = task;
    metadata.name = std::move(name);
    std::string domainError;
    FailureOr<TaskIterationDomainMetadata> domain =
        resolveTaskIterationDomain(task, domainError);
    if (succeeded(domain)) {
      metadata.tripCount = domain->effectiveMapperFiringCount;
      metadata.taskflowTripCount = domain->taskflowTripCount;
      metadata.sourceIterationMultiplicity = domain->internalMultiplicity;
      metadata.sourceIterationWorkCount = domain->sourceIterationWorkCount;
      metadata.sourceIterationDomainCertified = domain->sourceCertified;
      metadata.sourceIterationDomainComplete = domain->complete;
      metadata.tripCountKnown = domain->countKnown;
      metadata.sourceIterationDomainStatus = domain->status;
      metadata.sourceIterationDomainReason = domain->reason;
      metadata.expandedInternalExtents =
          std::move(domain->expandedInternalExtents);
    } else {
      // Fact-only extraction preserves the diagnostic task record but never
      // substitutes one for an unproved or stale source workload count.
      metadata.tripCount = 0;
      metadata.taskflowTripCount = 0;
      metadata.sourceIterationWorkCount = 0;
      metadata.tripCountKnown = false;
      metadata.sourceIterationDomainStatus = "unsupported-or-unproven";
      metadata.sourceIterationDomainReason = std::move(domainError);
    }
    tasks.push_back(std::move(metadata));
    return WalkResult::advance();
  });
  if (tasks.empty() && error.empty())
    error = "function contains no taskflow.task operations";
  return error.empty() ? FailureOr<SmallVector<TaskMetadata>>(std::move(tasks))
                        : FailureOr<SmallVector<TaskMetadata>>(failure());
}

static FailureOr<SmallVector<EdgeFact, 0>>
collectEdges(func::FuncOp function,
             const DenseMap<Operation *, unsigned> &indices,
             std::string &error) {
  SmallVector<EdgeFact, 0> edges;
  TaskEdgeGraphOptions options;
  FailureOr<TaskEdgeGraph> graph = buildTaskEdgeGraph(function, options, error);
  if (failed(graph))
    return failure();
  for (const TaskEdge &edge : graph->getEdges()) {
    TaskflowTaskOp producerTask = edge.producer;
    TaskflowTaskOp consumerTask = edge.consumer;
    auto producer = indices.find(producerTask.getOperation());
    auto consumer = indices.find(consumerTask.getOperation());
    if (producer == indices.end() || consumer == indices.end()) {
      error = "typed task edge references a task outside graph metadata";
      return failure();
    }
    edges.push_back({producer->second, consumer->second,
                     stringifyTaskEdgeKind(edge.kind).str(),
                     stringifyTaskResultSegment(edge.producer_segment).str(),
                     edge.producer_index,
                     stringifyTaskOperandSegment(edge.consumer_segment).str(),
                     edge.consumer_index,
                     stringifyTaskEdgeOrigin(edge.origin).str(),
                     stringifyTaskEdgeScope(edge.scope).str(), "", "",
                     edge.transfer_region_lower, edge.transfer_region_upper,
                     edge.payload_bits});
  }
  llvm::sort(edges);
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
}

static bool parseSemanticIncomingEdge(StringRef encoded, StringRef &source,
                                      StringRef &role, StringRef &scope,
                                      StringRef &kind) {
  SmallVector<StringRef, 4> fields;
  encoded.split(fields, '|');
  if (fields.size() != 3 && fields.size() != 4)
    return false;
  if (llvm::any_of(fields, [](StringRef field) { return field.empty(); }))
    return false;
  source = fields[0];
  role = fields[1];
  scope = fields[2];
  kind = fields.size() == 4 ? fields[3] : StringRef();
  return kind.empty() || kind == "raw" || kind == "war" || kind == "waw" ||
         kind == "value" || kind == "control";
}

// Semantic annotations describe the producer-facing meaning of an incoming
// edge.  They are deliberately checked against the already-derived Taskflow
// graph so they cannot manufacture dependencies or silently relabel a missing
// one.  The generic edge kind remains raw/war/waw/value/control; semantic_role
// and semantic_scope are additional facts for clients that understand the
// producer's algorithm.
static bool validateSemanticIncomingEdges(
    func::FuncOp function, const DenseMap<Operation *, unsigned> &indices,
    MutableArrayRef<EdgeFact> edges, std::string &error) {
  llvm::StringMap<SmallVector<unsigned>> taskNameIds;
  DenseMap<unsigned, TaskflowTaskOp> tasksById;
  function.walk([&](TaskflowTaskOp task) {
    auto found = indices.find(task.getOperation());
    if (found != indices.end()) {
      taskNameIds[task.getTaskName()].push_back(found->second);
      tasksById[found->second] = task;
    }
  });

  for (TaskflowTaskOp consumer : function.getOps<TaskflowTaskOp>()) {
    auto targetIt = indices.find(consumer.getOperation());
    if (targetIt == indices.end()) {
      error = "semantic edge annotation references a task outside graph "
              "metadata";
      return false;
    }
    unsigned target = targetIt->second;
    Attribute attribute = consumer->getAttr(kSemanticIncomingEdgesAttr);
    if (!attribute)
      continue;
    auto annotations = dyn_cast<ArrayAttr>(attribute);
    if (!annotations) {
      error = "amoeba.semantic.incoming_edges must be an array attribute";
      return false;
    }

    DenseSet<unsigned> annotatedEdges;
    for (Attribute entry : annotations) {
      auto string = dyn_cast<StringAttr>(entry);
      if (!string) {
        error = "amoeba.semantic.incoming_edges entries must be strings";
        return false;
      }
      StringRef sourceName;
      StringRef role;
      StringRef scope;
      StringRef kind;
      if (!parseSemanticIncomingEdge(string.getValue(), sourceName, role,
                                     scope, kind)) {
        error = "amoeba.semantic.incoming_edges entries must have the form "
                "source|role|scope or source|role|scope|kind with non-empty fields";
        return false;
      }
      auto sourceIds = taskNameIds.lookup(sourceName);
      if (sourceIds.size() != 1) {
        error = sourceIds.empty()
                    ? "semantic incoming edge source task does not exist: " +
                          sourceName.str()
                    : "semantic incoming edge source task is ambiguous: " +
                          sourceName.str();
        return false;
      }
      unsigned source = sourceIds.front();
      SmallVector<unsigned> matches;
      for (auto [edgeIndex, edge] : llvm::enumerate(edges))
        if (edge.target == target && edge.source == source &&
            edge.origin == "taskflow" &&
            (kind.empty() || edge.kind == kind))
          matches.push_back(edgeIndex);
      if (matches.empty()) {
        error = "semantic incoming edge does not match an actual Taskflow "
                "edge: " +
                string.getValue().str();
        return false;
      }
      if (matches.size() != 1) {
        error = "semantic incoming edge source is ambiguous for actual "
                "Taskflow edges: " +
                string.getValue().str();
        return false;
      }
      unsigned edgeIndex = matches.front();
      if (!annotatedEdges.insert(edgeIndex).second) {
        error = "duplicate semantic incoming edge annotation: " +
                string.getValue().str();
        return false;
      }
      EdgeFact &edge = edges[edgeIndex];
      if (scope != edge.scope) {
        error = "semantic incoming edge scope disagrees with the actual "
                "Taskflow edge: " +
                string.getValue().str();
        return false;
      }
      edge.semanticRole = role.str();
      edge.semanticScope = scope.str();
    }

    for (auto [edgeIndex, edge] : llvm::enumerate(edges)) {
      if (edge.target != target || edge.origin != "taskflow")
        continue;
      if (!annotatedEdges.contains(edgeIndex)) {
        auto sourceTask = tasksById.find(edge.source);
        error = "missing semantic incoming edge annotation for actual "
                "Taskflow edge from " +
                (sourceTask == tasksById.end()
                     ? std::to_string(edge.source)
                     : sourceTask->second.getTaskName().str());
        return false;
      }
    }
  }
  return true;
}

static bool sameLoopDomain(TaskflowTaskOp first, TaskflowTaskOp second) {
  FailureOr<SmallVector<std::pair<int64_t, int64_t>>> firstDomain =
      loopDomain(first);
  FailureOr<SmallVector<std::pair<int64_t, int64_t>>> secondDomain =
      loopDomain(second);
  return succeeded(firstDomain) && succeeded(secondDomain) &&
         *firstDomain == *secondDomain;
}

static bool canMoveFirstToSecond(TaskflowTaskOp first, TaskflowTaskOp second) {
  if (!first || !second || first == second ||
      first->hasAttr("amoeba.control_predecessors") ||
      second->hasAttr("amoeba.control_predecessors") ||
      isNamedControlPredecessor(first) || isNamedControlPredecessor(second) ||
      first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second))
    return false;
  llvm::SmallPtrSet<Operation *, 8> intervening;
  for (Operation *operation = first->getNextNode();
       operation && operation != second.getOperation();
       operation = operation->getNextNode()) {
    if (isa<TaskflowChannelOp>(operation))
      continue;
    auto crossed = dyn_cast<TaskflowTaskOp>(operation);
    if (!crossed)
      return false;
    if (crossed->hasAttr("amoeba.control_predecessors") ||
        hasConflictingOriginalMemoryEffects(first, crossed))
      return false;
    intervening.insert(operation);
  }
  for (Value result : first->getResults())
    for (OpOperand &use : result.getUses())
      if (intervening.contains(use.getOwner()))
        return false;
  return true;
}

static bool isExclusiveChannelPath(Value producerResult, Value consumerInput) {
  Value current = consumerInput;
  while (auto channel = current.getDefiningOp<TaskflowChannelOp>()) {
    if (!current.hasOneUse())
      return false;
    current = channel.getSource();
  }
  return current == producerResult && producerResult.hasOneUse();
}

static bool safeProducerConsumerForwarding(TaskflowTaskOp producer,
                                           TaskflowTaskOp consumer,
                                           Value producerResult,
                                           Value consumerInput) {
  if (!producer.getBody().hasOneBlock() || !consumer.getBody().hasOneBlock() ||
      producer.getOriginalWriteMemrefs().size() !=
          producer.getWillWrites().size() ||
      consumer.getOriginalReadMemrefs().size() !=
          consumer.getWillReads().size() ||
      consumer.getOriginalWriteMemrefs().size() !=
          consumer.getWillWrites().size())
    return false;
  auto result = llvm::find(producer.getDoneWrites(), producerResult);
  auto input = llvm::find(consumer.getWillReads(), consumerInput);
  if (result == producer.getDoneWrites().end() ||
      input == consumer.getWillReads().end())
    return false;
  unsigned resultIndex = result - producer.getDoneWrites().begin();
  unsigned consumerReadIndex = input - consumer.getWillReads().begin();
  auto producerYield =
      dyn_cast<TaskflowYieldOp>(producer.getBody().front().getTerminator());
  if (!producerYield || resultIndex >= producerYield.getDoneWrites().size())
    return false;
  auto producerArgument =
      dyn_cast<BlockArgument>(producerYield.getDoneWrites()[resultIndex]);
  unsigned firstWrite = producer.getWillReads().size();
  if (!producerArgument ||
      producerArgument.getOwner() != &producer.getBody().front() ||
      producerArgument.getArgNumber() < firstWrite ||
      producerArgument.getArgNumber() >=
          firstWrite + producer.getWillWrites().size())
    return false;
  unsigned producerWriteIndex = producerArgument.getArgNumber() - firstWrite;
  Value intermediateRoot =
      producer.getOriginalWriteMemrefs()[producerWriteIndex];
  if (consumer.getOriginalReadMemrefs()[consumerReadIndex] !=
          intermediateRoot ||
      !isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
          intermediateRoot.getDefiningOp()))
    return false;
  for (OpOperand &use : intermediateRoot.getUses())
    if (use.getOwner() != producer.getOperation() &&
        use.getOwner() != consumer.getOperation() &&
        !isa<memref::DeallocOp>(use.getOwner()))
      return false;
  if (llvm::is_contained(producer.getOriginalReadMemrefs(), intermediateRoot) ||
      llvm::is_contained(consumer.getOriginalWriteMemrefs(), intermediateRoot))
    return false;
  for (auto [readIndex, root] :
       llvm::enumerate(consumer.getOriginalReadMemrefs()))
    if (readIndex != consumerReadIndex && root == intermediateRoot)
      return false;
  for (auto [writeIndex, root] :
       llvm::enumerate(producer.getOriginalWriteMemrefs())) {
    if (writeIndex == producerWriteIndex || root != intermediateRoot)
      continue;
    BlockArgument argument = producer.getBody().front().getArgument(
        producer.getWillReads().size() + writeIndex);
    if (!argument.use_empty())
      return false;
  }

  TaskflowHyperblockOp producerHyperblock;
  TaskflowHyperblockOp consumerHyperblock;
  for (Operation &operation : producer.getBody().front())
    if (auto hyperblock = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      if (producerHyperblock)
        return false;
      producerHyperblock = hyperblock;
    }
  for (Operation &operation : consumer.getBody().front())
    if (auto hyperblock = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      if (consumerHyperblock)
        return false;
      consumerHyperblock = hyperblock;
    }
  if (!producerHyperblock || !consumerHyperblock)
    return false;
  Block &producerKernel = producerHyperblock.getBody().front();
  Block &consumerKernel = consumerHyperblock.getBody().front();
  auto directPositions = [](ValueRange indices,
                            Block &body) -> FailureOr<SmallVector<unsigned>> {
    SmallVector<unsigned> positions;
    llvm::SmallDenseSet<unsigned, 4> unique;
    for (Value index : indices) {
      auto argument = dyn_cast<BlockArgument>(index);
      if (!argument || argument.getOwner() != &body ||
          !unique.insert(argument.getArgNumber()).second)
        return failure();
      positions.push_back(argument.getArgNumber());
    }
    return positions;
  };

  memref::StoreOp producerStore;
  for (OpOperand &use : producerArgument.getUses()) {
    if (isa<TaskflowYieldOp>(use.getOwner()))
      continue;
    auto store = dyn_cast<memref::StoreOp>(use.getOwner());
    if (!store || store->getBlock() != &producerKernel || producerStore)
      return false;
    producerStore = store;
  }
  if (!producerStore)
    return false;
  FailureOr<SmallVector<unsigned>> producerIndices =
      directPositions(producerStore.getIndices(), producerKernel);
  if (failed(producerIndices))
    return false;

  BlockArgument consumerArgument =
      consumer.getBody().front().getArgument(consumerReadIndex);
  bool sawLoad = false;
  for (OpOperand &use : consumerArgument.getUses()) {
    auto load = dyn_cast<memref::LoadOp>(use.getOwner());
    if (!load || load->getBlock() != &consumerKernel)
      return false;
    FailureOr<SmallVector<unsigned>> consumerIndices =
        directPositions(load.getIndices(), consumerKernel);
    if (failed(consumerIndices) || *consumerIndices != *producerIndices)
      return false;
    sawLoad = true;
  }
  return sawLoad;
}

static bool directProducerConsumer(TaskflowTaskOp first,
                                   TaskflowTaskOp second) {
  if (!canMoveFirstToSecond(first, second) || !sameLoopDomain(first, second))
    return false;
  unsigned edges = 0;
  Value producerIntermediate;
  Value consumerIntermediate;
  for (Value input : second.getWillReads()) {
    Value source = peelChannel(input);
    if (llvm::is_contained(first.getDoneWrites(), source) &&
        isExclusiveChannelPath(source, input)) {
      ++edges;
      producerIntermediate = source;
      consumerIntermediate = input;
    }
  }
  if (edges != 1)
    return false;
  for (Value result : first.getDoneWrites())
    if (result != producerIntermediate && !result.use_empty())
      return false;
  for (Value result : first.getValueOutputs())
    if (!result.use_empty())
      return false;
  return safeProducerConsumerForwarding(first, second, producerIntermediate,
                                        consumerIntermediate);
}

static bool independentSiblings(TaskflowTaskOp first, TaskflowTaskOp second) {
  if (!canMoveFirstToSecond(first, second) || !sameLoopDomain(first, second))
    return false;
  for (Value operand : second->getOperands())
    if (peelChannel(operand).getDefiningOp<TaskflowTaskOp>() == first)
      return false;
  for (Value operand : first->getOperands())
    if (peelChannel(operand).getDefiningOp<TaskflowTaskOp>() == second)
      return false;
  llvm::SmallPtrSet<Value, 8> firstInputs;
  for (Value input : first.getWillReads())
    firstInputs.insert(peelChannel(input));
  for (Value input : first.getValueInputs())
    firstInputs.insert(peelChannel(input));
  for (Value input : second.getWillReads())
    if (firstInputs.contains(peelChannel(input)))
      return true;
  for (Value input : second.getValueInputs())
    if (firstInputs.contains(peelChannel(input)))
      return true;
  return false;
}

struct DirectRawConnection {
  unsigned producerWriteIndex = 0;
  unsigned consumerReadIndex = 0;
};

static std::optional<DirectRawConnection>
directRawProducerConsumer(TaskflowTaskOp producer, TaskflowTaskOp consumer) {
  if (!producer || !consumer || producer == consumer ||
      !producer.getBody().hasOneBlock() || !consumer.getBody().hasOneBlock() ||
      producer->getBlock() != consumer->getBlock() ||
      !producer->isBeforeInBlock(consumer) ||
      producer.getOriginalWriteMemrefs().size() !=
          producer.getWillWrites().size() ||
      consumer.getOriginalReadMemrefs().size() !=
          consumer.getWillReads().size())
    return std::nullopt;
  std::optional<DirectRawConnection> connection;
  for (auto [inputIndex, input] : llvm::enumerate(consumer.getWillReads())) {
    Value source = peelChannel(input);
    auto result = llvm::find(producer.getDoneWrites(), source);
    if (result == producer.getDoneWrites().end())
      continue;
    unsigned resultIndex = result - producer.getDoneWrites().begin();
    auto taskYield =
        dyn_cast<TaskflowYieldOp>(producer.getBody().front().getTerminator());
    if (!taskYield || resultIndex >= taskYield.getDoneWrites().size())
      return std::nullopt;
    auto argument =
        dyn_cast<BlockArgument>(taskYield.getDoneWrites()[resultIndex]);
    unsigned firstWrite = producer.getWillReads().size();
    if (!argument || argument.getOwner() != &producer.getBody().front() ||
        argument.getArgNumber() < firstWrite)
      return std::nullopt;
    unsigned writeIndex = argument.getArgNumber() - firstWrite;
    if (writeIndex >= producer.getOriginalWriteMemrefs().size() ||
        producer.getOriginalWriteMemrefs()[writeIndex] !=
            consumer.getOriginalReadMemrefs()[inputIndex])
      return std::nullopt;
    if (connection)
      return std::nullopt;
    connection =
        DirectRawConnection{writeIndex, static_cast<unsigned>(inputIndex)};
  }
  return connection;
}

struct CanonicalTask {
  SmallVector<TaskflowCounterOp> counters;
  TaskflowHyperblockOp hyperblock;
  SmallVector<int64_t> lower;
  SmallVector<int64_t> upper;
};

enum class CotilingAxisCompatibility : uint8_t {
  Compatible,
  Incompatible,
  Unknown,
};

static FailureOr<std::optional<unsigned>>
indexCounterAxis(Value value, Block &hyperblockBody) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() != &hyperblockBody)
      return failure();
    return std::optional<unsigned>(argument.getArgNumber());
  }
  if (constantIndex(value))
    return std::optional<unsigned>();
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    FailureOr<std::optional<unsigned>> lhs =
        indexCounterAxis(add.getLhs(), hyperblockBody);
    FailureOr<std::optional<unsigned>> rhs =
        indexCounterAxis(add.getRhs(), hyperblockBody);
    if (failed(lhs) || failed(rhs) || (lhs->has_value() && rhs->has_value()))
      return failure();
    return lhs->has_value() ? *lhs : *rhs;
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    FailureOr<std::optional<unsigned>> lhs =
        indexCounterAxis(sub.getLhs(), hyperblockBody);
    FailureOr<std::optional<unsigned>> rhs =
        indexCounterAxis(sub.getRhs(), hyperblockBody);
    if (failed(lhs) || failed(rhs) || rhs->has_value())
      return failure();
    return *lhs;
  }
  return failure();
}

static FailureOr<unsigned> producerStorageDimension(TaskflowTaskOp task,
                                                    CanonicalTask &canonical,
                                                    unsigned writeIndex,
                                                    unsigned counterAxis) {
  if (writeIndex >= task.getWillWrites().size())
    return failure();
  Block &taskBody = task.getBody().front();
  BlockArgument output =
      taskBody.getArgument(task.getWillReads().size() + writeIndex);
  std::optional<unsigned> storageDimension;
  bool valid = true;
  bool sawStore = false;
  task.walk([&](memref::StoreOp store) {
    if (!valid || store.getMemRef() != output)
      return;
    if (store->getBlock() != &canonical.hyperblock.getBody().front()) {
      valid = false;
      return;
    }
    sawStore = true;
    std::optional<unsigned> useDimension;
    for (auto [dimension, index] : llvm::enumerate(store.getIndices())) {
      auto argument = dyn_cast<BlockArgument>(index);
      if (!argument ||
          argument.getOwner() != &canonical.hyperblock.getBody().front()) {
        valid = false;
        return;
      }
      if (argument.getArgNumber() == counterAxis) {
        if (useDimension) {
          valid = false;
          return;
        }
        useDimension = dimension;
      }
    }
    if (!valid || !useDimension ||
        (storageDimension && *storageDimension != *useDimension)) {
      valid = false;
      return;
    }
    storageDimension = useDimension;
  });
  if (!valid || !sawStore || !storageDimension)
    return failure();
  return *storageDimension;
}

static FailureOr<std::optional<unsigned>>
consumerStorageDimension(TaskflowTaskOp task, CanonicalTask &canonical,
                         unsigned readIndex, unsigned counterAxis) {
  if (readIndex >= task.getWillReads().size())
    return failure();
  BlockArgument input = task.getBody().front().getArgument(readIndex);
  std::optional<unsigned> storageDimension;
  bool sawLoad = false;
  bool valid = true;
  bool selectedAxisMissing = false;
  task.walk([&](memref::LoadOp load) {
    if (!valid || load.getMemRef() != input)
      return;
    if (load->getBlock() != &canonical.hyperblock.getBody().front()) {
      valid = false;
      return;
    }
    sawLoad = true;
    std::optional<unsigned> useDimension;
    for (auto [dimension, index] : llvm::enumerate(load.getIndices())) {
      FailureOr<std::optional<unsigned>> projected =
          indexCounterAxis(index, canonical.hyperblock.getBody().front());
      if (failed(projected)) {
        valid = false;
        return;
      }
      if (*projected && **projected == counterAxis) {
        if (useDimension) {
          valid = false;
          return;
        }
        useDimension = dimension;
      }
    }
    if (!useDimension) {
      selectedAxisMissing = true;
      return;
    }
    if (storageDimension && *storageDimension != *useDimension) {
      valid = false;
      return;
    }
    storageDimension = useDimension;
  });
  if (!valid || !sawLoad)
    return failure();
  if (selectedAxisMissing)
    return std::optional<unsigned>();
  return storageDimension;
}

static CotilingAxisCompatibility cotilingAxisCompatibility(
    TaskflowTaskOp producer, CanonicalTask &producerCanonical,
    unsigned producerAxis, TaskflowTaskOp consumer,
    CanonicalTask &consumerCanonical, unsigned consumerAxis,
    DirectRawConnection connection) {
  FailureOr<unsigned> producerDimension = producerStorageDimension(
      producer, producerCanonical, connection.producerWriteIndex, producerAxis);
  FailureOr<std::optional<unsigned>> consumerDimension =
      consumerStorageDimension(consumer, consumerCanonical,
                               connection.consumerReadIndex, consumerAxis);
  if (failed(producerDimension) || failed(consumerDimension))
    return CotilingAxisCompatibility::Unknown;
  if (!*consumerDimension || *producerDimension != **consumerDimension)
    return CotilingAxisCompatibility::Incompatible;
  return CotilingAxisCompatibility::Compatible;
}

static FailureOr<CanonicalTask> canonicalTask(TaskflowTaskOp task) {
  if (!task.getBody().hasOneBlock())
    return failure();
  CanonicalTask canonical;
  for (Operation &operation : task.getBody().front()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      canonical.counters.push_back(counter);
      continue;
    }
    if (auto hyperblock = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      if (canonical.hyperblock)
        return failure();
      canonical.hyperblock = hyperblock;
      continue;
    }
    if (!isa<TaskflowYieldOp, arith::ConstantOp>(&operation))
      return failure();
  }
  if (!canonical.hyperblock || canonical.counters.empty() ||
      !canonical.hyperblock.getIterArgs().empty() ||
      canonical.hyperblock.getNumResults() != 0 ||
      canonical.hyperblock.getIndices().size() != canonical.counters.size())
    return failure();
  for (auto [index, counter] : llvm::enumerate(canonical.counters)) {
    std::optional<int64_t> lower = constantIndex(counter.getLowerBound());
    std::optional<int64_t> upper = constantIndex(counter.getUpperBound());
    std::optional<int64_t> step = constantIndex(counter.getStep());
    if (!lower || !upper || !step || *step != 1 || *upper <= *lower ||
        (index == 0 && counter.getParentIndex()) ||
        (index != 0 && counter.getParentIndex() !=
                           canonical.counters[index - 1].getCounterIndex()) ||
        canonical.hyperblock.getIndices()[index] != counter.getCounterIndex())
      return failure();
    canonical.lower.push_back(*lower);
    canonical.upper.push_back(*upper);
  }
  return canonical;
}

static bool isStorableScalar(Type type) { return type.isIntOrIndexOrFloat(); }

static bool hasDirectIndices(ValueRange indices, Block &hyperblockBody) {
  return indices.size() == hyperblockBody.getNumArguments() &&
         llvm::equal(indices, hyperblockBody.getArguments());
}

static FailureOr<SmallVector<SmallVector<unsigned>>>
legalFissionPartitions(TaskflowTaskOp task, CanonicalTask &canonical,
                       uint64_t maxActions, std::string &error) {
  SmallVector<SmallVector<unsigned>> partitions;
  if (isNamedControlPredecessor(task) || !task.getDoneReads().empty() ||
      task.getWillWrites().size() != 1 || task.getDoneWrites().size() != 1 ||
      !task.getValueOutputs().empty() ||
      task.getOriginalWriteMemrefs().size() != 1)
    return partitions;
  for (NamedAttribute attribute : task->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name != "task_name" && name != "operandSegmentSizes" &&
        name != "resultSegmentSizes" && name != "operand_segment_sizes" &&
        name != "result_segment_sizes" &&
        !name.starts_with("amoeba.fission.") &&
        !name.starts_with("amoeba.fusion.") &&
        !name.starts_with("amoeba.tiling."))
      return partitions;
  }
  if (llvm::any_of(canonical.lower, [](int64_t lower) { return lower != 0; }))
    return partitions;
  Block &taskBody = task.getBody().front();
  size_t expectedArguments = task.getWillReads().size() +
                             task.getWillWrites().size() +
                             task.getValueInputs().size();
  auto taskYield = dyn_cast<TaskflowYieldOp>(taskBody.getTerminator());
  if (taskBody.getNumArguments() != expectedArguments || !taskYield ||
      !taskYield.getDoneReads().empty() ||
      taskYield.getDoneWrites().size() != 1 ||
      taskYield.getDoneWrites().front() !=
          taskBody.getArgument(task.getWillReads().size()) ||
      !taskYield.getValueResults().empty())
    return partitions;

  Block &body = canonical.hyperblock.getBody().front();
  SmallVector<Operation *> nodes;
  for (Operation &operation : body.without_terminator()) {
    if (operation.getNumRegions() != 0)
      return partitions;
    if (!isa<memref::LoadOp, memref::StoreOp>(&operation) &&
        (!isPure(&operation) || !isMemoryEffectFree(&operation)))
      return partitions;
    if (auto load = dyn_cast<memref::LoadOp>(&operation)) {
      bool knownRead = false;
      for (size_t index = 0; index < task.getWillReads().size(); ++index)
        knownRead |= load.getMemRef() == taskBody.getArgument(index);
      if (!knownRead || !hasDirectIndices(load.getIndices(), body))
        return partitions;
    }
    if (auto store = dyn_cast<memref::StoreOp>(&operation))
      if (store.getMemRef() !=
              taskBody.getArgument(task.getWillReads().size()) ||
          !hasDirectIndices(store.getIndices(), body))
        return partitions;
    nodes.push_back(&operation);
  }
  if (nodes.size() < 2)
    return partitions;
  DenseSet<Value> arguments(body.getArguments().begin(),
                            body.getArguments().end());
  DenseMap<Value, unsigned> producers;
  SmallVector<SmallVector<unsigned>> predecessors(nodes.size());
  SmallVector<bool> prefixAvailable(nodes.size(), true);
  for (auto [index, node] : llvm::enumerate(nodes)) {
    for (Value operand : node->getOperands()) {
      if (arguments.contains(operand))
        continue;
      if (isa<BlockArgument>(operand)) {
        if (!isKnownFissionBlockArgument(operand, task, body))
          return partitions;
        prefixAvailable[index] &=
            isAvailableInFissionPrefix(operand, task, body);
        continue;
      }
      auto producer = producers.find(operand);
      if (producer == producers.end() || producer->second >= index)
        return partitions;
      predecessors[index].push_back(producer->second);
    }
    llvm::sort(predecessors[index]);
    predecessors[index].erase(
        std::unique(predecessors[index].begin(), predecessors[index].end()),
        predecessors[index].end());
    for (Value result : node->getResults())
      producers[result] = index;
  }

  SmallVector<bool> selected(nodes.size(), false);
  SmallVector<unsigned> leftNodes;
  bool overflow = false;
  std::function<void(unsigned)> enumerate = [&](unsigned nodeIndex) {
    if (overflow)
      return;
    if (nodeIndex != nodes.size()) {
      enumerate(nodeIndex + 1);
      if (prefixAvailable[nodeIndex] &&
          !isa<memref::StoreOp>(nodes[nodeIndex]) &&
          llvm::all_of(predecessors[nodeIndex], [&](unsigned predecessor) {
            return selected[predecessor];
          })) {
        selected[nodeIndex] = true;
        leftNodes.push_back(nodeIndex);
        enumerate(nodeIndex + 1);
        leftNodes.pop_back();
        selected[nodeIndex] = false;
      }
      return;
    }
    if (leftNodes.empty() || leftNodes.size() == nodes.size())
      return;
    bool crosses = false;
    for (unsigned index = 0; index < nodes.size(); ++index) {
      if (selected[index])
        continue;
      for (Value operand : nodes[index]->getOperands()) {
        auto found = producers.find(operand);
        if (found != producers.end() && selected[found->second]) {
          if (!isStorableScalar(operand.getType()))
            return;
          crosses = true;
        }
      }
    }
    if (!crosses)
      return;
    if (partitions.size() == maxActions) {
      overflow = true;
      return;
    }
    partitions.push_back(leftNodes);
  };
  enumerate(0);
  if (overflow) {
    error = "legal fission cut-set count exceeds max-fission-actions-per-task";
    return failure();
  }
  llvm::sort(partitions);
  return partitions;
}

static bool isNoAliasFunctionArgument(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return false;
  auto function =
      dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
  return function && argument.getOwner() == &function.getBody().front() &&
         function.getArgAttr(argument.getArgNumber(), "amoeba.noalias");
}

static bool provesDistinctStorage(Value lhs, Value rhs) {
  if (lhs == rhs)
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

static bool tilingContract(TaskflowTaskOp task, CanonicalTask &canonical,
                           size_t axis, std::string *reason = nullptr) {
  auto reject = [&](StringRef message) {
    if (reason)
      *reason = message.str();
    return false;
  };
  if (isNamedControlPredecessor(task))
    return reject("task_has_control_predecessor");
  if (task.getWillWrites().empty())
    return reject("no_writable_output");
  if (task.getWillWrites().size() != task.getDoneWrites().size() ||
      task.getWillWrites().size() != task.getOriginalWriteMemrefs().size())
    return reject("incomplete_write_provenance");
  if (!task.getDoneReads().empty())
    return reject("read_completion_result");
  if (!task.getValueOutputs().empty())
    return reject("value_output");
  if (task.getOriginalReadMemrefs().size() != task.getWillReads().size())
    return reject("incomplete_read_provenance");
  if (axis >= canonical.counters.size())
    return reject("axis_out_of_range");
  for (NamedAttribute attribute : task->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name != "operandSegmentSizes" && name != "resultSegmentSizes" &&
        name != "operand_segment_sizes" && name != "result_segment_sizes" &&
        name != "task_name" && !name.starts_with("amoeba.tiling.") &&
        !name.starts_with("amoeba.fusion.") &&
        !name.starts_with("amoeba.fission."))
      return reject("unsupported_task_attribute");
  }
  Block &taskBody = task.getBody().front();
  Block &kernelBody = canonical.hyperblock.getBody().front();
  auto yield = dyn_cast<TaskflowYieldOp>(taskBody.getTerminator());
  if (!yield || yield.getDoneWrites().size() != task.getWillWrites().size())
    return reject("noncanonical_write_yield");
  unsigned firstWriteArgument = task.getWillReads().size();
  SmallVector<MemRefType> outputTypes(task.getWillWrites().size());
  SmallVector<Value> writeArguments(task.getWillWrites().size());
  SmallVector<bool> mappedWrites(task.getWillWrites().size(), false);
  for (Value yielded : yield.getDoneWrites()) {
    auto argument = dyn_cast<BlockArgument>(yielded);
    if (!argument || argument.getOwner() != &taskBody ||
        argument.getArgNumber() < firstWriteArgument ||
        argument.getArgNumber() >=
            firstWriteArgument + task.getWillWrites().size())
      return reject("noncanonical_write_yield");
    unsigned writeIndex = argument.getArgNumber() - firstWriteArgument;
    auto type = dyn_cast<MemRefType>(argument.getType());
    if (mappedWrites[writeIndex] || !type || !type.hasStaticShape() ||
        !type.getLayout().isIdentity() ||
        task.getOriginalWriteMemrefs()[writeIndex].getType() != type)
      return reject("unsupported_output_layout_or_provenance");
    mappedWrites[writeIndex] = true;
    outputTypes[writeIndex] = type;
    writeArguments[writeIndex] = argument;
  }
  for (auto [lhsIndex, lhs] : llvm::enumerate(task.getOriginalWriteMemrefs()))
    for (Value rhs : task.getOriginalWriteMemrefs().drop_front(lhsIndex + 1))
      if (!provesDistinctStorage(lhs, rhs))
        return reject("ambiguous_output_alias");

  SmallVector<std::optional<unsigned>> readOutput(task.getWillReads().size());
  llvm::SmallPtrSet<Value, 8> readOnlyInputs;
  for (auto [readIndex, root] :
       llvm::enumerate(task.getOriginalReadMemrefs())) {
    Value argument = taskBody.getArgument(readIndex);
    if (argument.getType() != root.getType())
      return reject("mismatched_input_provenance");
    for (auto [writeIndex, outputRoot] :
         llvm::enumerate(task.getOriginalWriteMemrefs())) {
      if (root == outputRoot) {
        readOutput[readIndex] = writeIndex;
        break;
      }
      if (!provesDistinctStorage(root, outputRoot))
        return reject("ambiguous_input_output_alias");
    }
    if (!readOutput[readIndex])
      readOnlyInputs.insert(argument);
  }

  SmallVector<SmallVector<Value>> expectedOutputIndices(outputTypes.size());
  SmallVector<SmallVector<unsigned>> outputCounterAxes(outputTypes.size());
  SmallVector<bool> sawStore(outputTypes.size(), false);
  bool valid = true;
  auto checkOutputIndices = [&](unsigned outputIndex, ValueRange indices) {
    MemRefType outputType = outputTypes[outputIndex];
    if (!valid || indices.size() != static_cast<size_t>(outputType.getRank())) {
      valid = false;
      return;
    }
    if (expectedOutputIndices[outputIndex].empty()) {
      llvm::SmallDenseSet<unsigned, 4> uniqueAxes;
      for (Value index : indices) {
        auto found = llvm::find(kernelBody.getArguments(), index);
        if (found == kernelBody.getArguments().end()) {
          valid = false;
          return;
        }
        unsigned counterAxis =
            static_cast<unsigned>(found - kernelBody.getArguments().begin());
        if (!uniqueAxes.insert(counterAxis).second) {
          valid = false;
          return;
        }
        expectedOutputIndices[outputIndex].push_back(index);
        outputCounterAxes[outputIndex].push_back(counterAxis);
      }
      return;
    }
    if (!llvm::equal(indices, expectedOutputIndices[outputIndex]))
      valid = false;
  };
  task.walk([&](Operation *operation) {
    if (!valid || operation == task.getOperation() ||
        isa<TaskflowYieldOp, TaskflowCounterOp, TaskflowHyperblockOp,
            TaskflowHyperblockYieldOp, arith::ConstantOp>(operation))
      return;
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      auto readArgument = dyn_cast<BlockArgument>(load.getMemRef());
      if (readArgument && readArgument.getOwner() == &taskBody &&
          readArgument.getArgNumber() < readOutput.size() &&
          readOutput[readArgument.getArgNumber()]) {
        checkOutputIndices(*readOutput[readArgument.getArgNumber()],
                           load.getIndices());
        return;
      }
      auto outputArgument = llvm::find(writeArguments, load.getMemRef());
      if (outputArgument != writeArguments.end()) {
        checkOutputIndices(outputArgument - writeArguments.begin(),
                           load.getIndices());
      } else if (!readOnlyInputs.contains(load.getMemRef())) {
        valid = false;
      }
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      auto outputArgument = llvm::find(writeArguments, store.getMemRef());
      if (outputArgument == writeArguments.end()) {
        valid = false;
        return;
      }
      unsigned outputIndex = outputArgument - writeArguments.begin();
      checkOutputIndices(outputIndex, store.getIndices());
      sawStore[outputIndex] = true;
      return;
    }
    valid &= operation->getNumRegions() == 0 && isMemoryEffectFree(operation);
  });
  if (!valid)
    return reject("unsupported_or_nonrectangular_access");
  for (auto [outputIndex, outputType] : llvm::enumerate(outputTypes)) {
    if (!sawStore[outputIndex] ||
        !llvm::is_contained(outputCounterAxes[outputIndex], axis))
      return reject("selected_axis_not_in_every_output");
    for (auto [dimension, counterAxis] :
         llvm::enumerate(outputCounterAxes[outputIndex]))
      if (canonical.lower[counterAxis] < 0 ||
          canonical.upper[counterAxis] > outputType.getShape()[dimension])
        return reject("output_region_out_of_bounds");
  }
  return true;
}

struct SiblingMaterializationStatus {
  StringRef status;
  StringRef reason;
};

// The source facts pass proves the pair-level Taskflow sibling contract, but
// post-Neura materialization has a second capability contract. Keep this
// check in C++ so a facts-listed stateful pair is never published as a
// proved-legal post-Neura rewrite. The pair remains in the inventory as
// unknown until the materializer grows the stateful-kernel protocol or a
// dedicated dry run proves it.
static SiblingMaterializationStatus
siblingMaterializationStatus(TaskflowTaskOp first, TaskflowTaskOp second) {
  for (TaskflowTaskOp task : {first, second}) {
    if (!task || !task.getBody().hasOneBlock())
      return {"unknown", "post_neura_kernel_contract_unproven"};
    SmallVector<neura::KernelOp> taskKernels;
    for (Operation &operation : task.getBody().front())
      if (auto kernel = dyn_cast<neura::KernelOp>(&operation))
        taskKernels.push_back(kernel);
    if (taskKernels.size() != 1 ||
        !taskKernels.front().getBody().hasOneBlock())
      return {"unknown", "post_neura_kernel_contract_unproven"};
    neura::KernelOp kernel = taskKernels.front();
    if (!kernel.getIterArgsInit().empty() || !kernel.getOutputs().empty())
      return {"unknown", "needs-stateful-kernel-support"};
  }
  // Stateless one-block kernels still need the actual post-Neura materializer
  // dry run for body/index/storage checks. Do not call them legal here.
  return {"unknown", "post_neura_materializer_dry_run_required"};
}

static json::Object fusionAction(StringRef mode, StringRef first,
                                 StringRef second,
                                 StringRef status = "unknown",
                                 StringRef reason =
                                     "post_neura_materializer_dry_run_required") {
  return json::Object{{"kind", "fusion"},
                      {"mode", mode},
                      {"first_task", first},
                      {"second_task", second},
                      {"status", status},
                      {"reason", reason}};
}

static json::Object siblingFusionCandidate(TaskflowTaskOp first,
                                           TaskflowTaskOp second,
                                           StringRef firstName,
                                           StringRef secondName) {
  SiblingMaterializationStatus materialization =
      siblingMaterializationStatus(first, second);
  return json::Object{
      {"first_task", firstName},
      {"second_task", secondName},
      {"proof", "source-owned-sibling-contract"},
      {"source_status", "legal"},
      {"materialization_status", materialization.status},
      {"status", materialization.status},
      {"reason", materialization.reason}};
}

static json::Object cotilingAction(StringRef producer, StringRef consumer,
                                   int64_t producerAxis, int64_t consumerAxis,
                                   int64_t factor) {
  return json::Object{{"kind", "cotiling"},
                      {"producer_task", producer},
                      {"consumer_task", consumer},
                      {"producer_axis", producerAxis},
                      {"consumer_axis", consumerAxis},
                      {"factor", factor}};
}

static json::Array integerArray(ArrayRef<int64_t> values) {
  json::Array result;
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

static std::optional<json::Object> tileDescriptor(TaskflowTaskOp task) {
  auto parent = task->getAttrOfType<StringAttr>("amoeba.tiling.parent_task");
  if (!parent)
    return std::nullopt;
  auto axis = task->getAttrOfType<IntegerAttr>("amoeba.tiling.axis");
  auto factor = task->getAttrOfType<IntegerAttr>("amoeba.tiling.factor");
  auto part = task->getAttrOfType<IntegerAttr>("amoeba.tiling.part_index");
  auto step = task->getAttrOfType<IntegerAttr>("amoeba.tiling.step");
  auto original =
      task->getAttrOfType<DenseI64ArrayAttr>("amoeba.tiling.original_range");
  auto derived =
      task->getAttrOfType<DenseI64ArrayAttr>("amoeba.tiling.derived_range");
  if (!axis || !factor || !part || !step || !original || !derived)
    return std::nullopt;

  json::Array inputs;
  auto inputLowers =
      task->getAttrOfType<ArrayAttr>(kTilingInputRegionLowersAttr);
  auto inputUppers =
      task->getAttrOfType<ArrayAttr>(kTilingInputRegionUppersAttr);
  auto inputReasons =
      task->getAttrOfType<ArrayAttr>(kTilingInputRegionReasonsAttr);
  if (inputLowers && inputUppers && inputReasons &&
      inputLowers.size() == task.getWillReads().size() &&
      inputUppers.size() == inputLowers.size() &&
      inputReasons.size() == inputLowers.size()) {
    for (size_t index = 0; index < inputLowers.size(); ++index) {
      json::Object input{{"operand_index", static_cast<int64_t>(index)}};
      if (auto reason = dyn_cast<StringAttr>(inputReasons[index]))
        input["reason"] = reason.getValue().str();
      auto lower = dyn_cast<DenseI64ArrayAttr>(inputLowers[index]);
      auto upper = dyn_cast<DenseI64ArrayAttr>(inputUppers[index]);
      if (lower && upper) {
        input["scope"] = "tile_local";
        input["lower"] = integerArray(lower.asArrayRef());
        input["upper"] = integerArray(upper.asArrayRef());
      } else {
        input["scope"] = "tensor_wide";
      }
      inputs.push_back(std::move(input));
    }
  }

  json::Array outputs;
  auto outputAxes =
      task->getAttrOfType<DenseI64ArrayAttr>("amoeba.tiling.output_axes");
  auto outputResults = task->getAttrOfType<DenseI64ArrayAttr>(
      "amoeba.tiling.output_result_indices");
  auto outputLowers =
      task->getAttrOfType<ArrayAttr>(kTilingOutputRegionLowersAttr);
  auto outputUppers =
      task->getAttrOfType<ArrayAttr>(kTilingOutputRegionUppersAttr);
  if (outputAxes && outputResults && outputLowers && outputUppers &&
      outputAxes.size() == static_cast<int64_t>(task.getWillWrites().size()) &&
      outputResults.size() == outputAxes.size() &&
      outputLowers.size() == task.getWillWrites().size() &&
      outputUppers.size() == outputLowers.size()) {
    for (size_t index = 0; index < outputLowers.size(); ++index) {
      auto lower = dyn_cast<DenseI64ArrayAttr>(outputLowers[index]);
      auto upper = dyn_cast<DenseI64ArrayAttr>(outputUppers[index]);
      if (!lower || !upper)
        continue;
      outputs.push_back(
          json::Object{{"write_index", static_cast<int64_t>(index)},
                       {"result_index", outputResults[index]},
                       {"axis", outputAxes[index]},
                       {"lower", integerArray(lower.asArrayRef())},
                       {"upper", integerArray(upper.asArrayRef())}});
    }
  }
  return json::Object{{"source_task", parent.getValue()},
                      {"axis", axis.getInt()},
                      {"partition_count", factor.getInt()},
                      {"partition_index", part.getInt()},
                      {"step", step.getInt()},
                      {"original_range", integerArray(original.asArrayRef())},
                      {"derived_range", integerArray(derived.asArrayRef())},
                      {"inputs", std::move(inputs)},
                      {"outputs", std::move(outputs)}};
}

// Post-Neura tiling has its own metadata contract. It does not carry the
// pre-Neura tile's input/output region descriptors or step.
static std::optional<json::Object>
neuraTileDescriptor(TaskflowTaskOp task, std::string &error) {
  bool hasMetadata = false;
  for (NamedAttribute attribute : task->getAttrs())
    hasMetadata |= attribute.getName().strref().starts_with(
        "amoeba.neura.tiling.");
  if (!hasMetadata)
    return std::nullopt;

  auto parent =
      task->getAttrOfType<StringAttr>("amoeba.neura.tiling.parent_task");
  auto axis = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.axis");
  auto factor = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.factor");
  auto part =
      task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.part_index");
  auto original = task->getAttrOfType<DenseI64ArrayAttr>(
      "amoeba.neura.tiling.original_range");
  auto derived = task->getAttrOfType<DenseI64ArrayAttr>(
      "amoeba.neura.tiling.derived_range");
  if (!parent || !axis || !factor || !part || !original || !derived ||
      parent.getValue().empty() || axis.getInt() < 0 || factor.getInt() < 2 ||
      part.getInt() < 0 || part.getInt() >= factor.getInt() ||
      original.size() != 2 || derived.size() != 2 ||
      original[0] >= original[1] || derived[0] < original[0] ||
      derived[0] >= derived[1] || derived[1] > original[1]) {
    error = "incomplete or invalid amoeba.neura.tiling metadata";
    return std::nullopt;
  }
  return json::Object{{"source_task", parent.getValue()},
                      {"axis", axis.getInt()},
                      {"partition_count", factor.getInt()},
                      {"partition_index", part.getInt()},
                      {"original_range", integerArray(original.asArrayRef())},
                      {"derived_range", integerArray(derived.asArrayRef())}};
}

// The graph structural key must distinguish retained from forwarded
// post-Neura fusion.  Those modes have different memory effects even when
// their Taskflow edges are otherwise identical.
static std::optional<json::Object>
neuraFusionDescriptor(TaskflowTaskOp task, std::string &error) {
  bool hasMetadata = false;
  for (NamedAttribute attribute : task->getAttrs())
    hasMetadata |= attribute.getName().strref().starts_with(
        "amoeba.neura.fusion.");
  if (!hasMetadata)
    return std::nullopt;
  auto mode = task->getAttrOfType<StringAttr>("amoeba.neura.fusion.mode");
  auto loads = task->getAttrOfType<IntegerAttr>(
      "amoeba.neura.fusion.eliminated_loads");
  auto stores = task->getAttrOfType<IntegerAttr>(
      "amoeba.neura.fusion.eliminated_stores");
  auto siblingFirst = task->getAttrOfType<StringAttr>(
      "amoeba.neura.fusion.sibling_first");
  auto siblingSecond = task->getAttrOfType<StringAttr>(
      "amoeba.neura.fusion.sibling_second");
  if (!mode || !loads || !stores) {
    error = "incomplete or invalid amoeba.neura.fusion metadata";
    return std::nullopt;
  }
  if (mode.getValue() == "sibling") {
    if (loads.getInt() != 0 || stores.getInt() != 0 ||
        !siblingFirst || !siblingSecond ||
        siblingFirst.getValue().empty() || siblingSecond.getValue().empty() ||
        siblingFirst.getValue() == siblingSecond.getValue()) {
      error = "incomplete or invalid sibling amoeba.neura.fusion metadata";
      return std::nullopt;
    }
    return json::Object{{"mode", "sibling"},
                        {"eliminated_loads", 0},
                        {"eliminated_stores", 0},
                        {"sibling_first", siblingFirst.getValue().str()},
                        {"sibling_second", siblingSecond.getValue().str()}};
  }
  if ((mode.getValue() != "retained" && mode.getValue() != "forwarded") ||
      siblingFirst || siblingSecond ||
      loads.getInt() != (mode.getValue() == "forwarded" ? 1 : 0) ||
      stores.getInt() != (mode.getValue() == "forwarded" ? 1 : 0)) {
    error = "incomplete or invalid amoeba.neura.fusion metadata";
    return std::nullopt;
  }
  return json::Object{{"mode", mode.getValue().str()},
                      {"eliminated_loads", loads.getInt()},
                      {"eliminated_stores", stores.getInt()}};
}

struct ExtractJointTaskGraphFactsPass
    : public PassWrapper<ExtractJointTaskGraphFactsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractJointTaskGraphFactsPass)

  ExtractJointTaskGraphFactsPass() = default;
  ExtractJointTaskGraphFactsPass(const ExtractJointTaskGraphFactsPass &other)
      : PassWrapper(other) {}
  StringRef getArgument() const override {
    return "extract-joint-task-graph-facts";
  }
  StringRef getDescription() const override {
    return "Extract graph facts and conservative legal rewrite actions";
  }
  Option<std::string> functionName{*this, "function",
                                   llvm::cl::desc("Taskflow function"),
                                   llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output",
                                 llvm::cl::desc("Graph facts JSON output"),
                                 llvm::cl::init("")};
  Option<bool> factOnly{
      *this, "fact-only",
      llvm::cl::desc("Emit structural facts for mixed Taskflow/Neura IR; "
                     "do not require a Taskflow hyperblock or emit rewrite "
                     "actions"),
      llvm::cl::init(false)};
  Option<int64_t> maxTileFactor{
      *this, "max-tile-factor",
      llvm::cl::desc("Largest generated task count per tiling action"),
      llvm::cl::init(8)};
  Option<int64_t> maxFissionActionsPerTask{
      *this, "max-fission-actions-per-task",
      llvm::cl::desc("Fail above this many legal DFG cut sets per task; zero "
                     "disables fission action generation"),
      llvm::cl::init(0)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName, error);
    if (failed(selected) || outputFile.empty() || maxTileFactor < 2 ||
        maxFissionActionsPerTask < 0) {
      module.emitError()
          << (error.empty() ? "graph facts require output, max-tile-factor >= "
                              "2, and a nonnegative fission action bound"
                            : error);
      return signalPassFailure();
    }
    FailureOr<SmallVector<TaskMetadata>> metadata =
        factOnly ? collectFactOnlyMetadata(*selected, error)
                 : collectAnalyticalTaskMetadata(*selected, error);
    if (failed(metadata)) {
      selected->emitError() << error;
      return signalPassFailure();
    }
    DenseMap<Operation *, unsigned> indices;
    for (auto [index, task] : llvm::enumerate(*metadata))
      indices[task.op.getOperation()] = index;
    FailureOr<SmallVector<EdgeFact, 0>> edges =
        collectEdges(*selected, indices, error);
    if (failed(edges)) {
      selected->emitError() << error;
      return signalPassFailure();
    }
    if (!validateSemanticIncomingEdges(*selected, indices, *edges, error)) {
      selected->emitError() << error;
      return signalPassFailure();
    }
    DenseMap<Value, unsigned> externalValueIds;
    unsigned nextExternalValueId = 0;
    for (BlockArgument argument : selected->getArguments())
      externalValueIds[argument] = nextExternalValueId++;
    for (Operation &operation : selected->getBody().front()) {
      if (isa<TaskflowTaskOp, TaskflowChannelOp, TaskflowJoinOp,
              TaskflowReadCompletionJoinOp>(&operation))
        continue;
      for (Value result : operation.getResults())
        externalValueIds.try_emplace(result, nextExternalValueId++);
    }

    json::Array taskRecords;
    json::Array shapeNeeds;
    json::Array structuralNodes;
    json::Array kReductionTasks;
    json::Array kReductionUnprovenTasks;
    json::Array kReductionIllegalTasks;
    bool kReductionFactsComplete = true;
    json::Array siblingFusionCandidates;
    json::Array siblingFusionScannedPairs;
    bool siblingFusionFactsComplete = true;
    bool siblingFusionMaterializationComplete = true;
    json::Array actions;
    json::Array actionDiagnostics;
    int64_t consideredTiling = 0;
    int64_t acceptedTiling = 0;
    int64_t consideredFusion = 0;
    int64_t acceptedFusion = 0;
    int64_t consideredCotiling = 0;
    int64_t acceptedCotiling = 0;
    llvm::StringMap<int64_t> tilingRejections;
    llvm::StringMap<int64_t> fusionRejections;
    llvm::StringMap<int64_t> cotilingRejections;
    for (auto [index, task] : llvm::enumerate(*metadata)) {
      FailureOr<CanonicalTask> canonical = canonicalTask(task.op);
      if (failed(canonical) && !factOnly) {
        task.op.emitError("graph enumeration requires the canonical static "
                          "counter-chain plus one hyperblock phase");
        return signalPassFailure();
      }
      json::Array dimensions;
      FailureOr<SmallVector<std::pair<int64_t, int64_t>>> domain =
          loopDomain(task.op);
      if (succeeded(domain))
        for (auto [lower, upper] : *domain)
          dimensions.push_back(upper - lower);
      json::Object taskRecord{{"task_name", task.name},
                              {"task_index", static_cast<int64_t>(index)},
                              {"cost_query_supported", !factOnly},
                              {"native_mapper_input_supported", !factOnly}};
      taskRecord["source_iteration_domain_status"] =
          task.sourceIterationDomainStatus;
      taskRecord["source_iteration_domain_certified"] =
          task.sourceIterationDomainCertified;
      taskRecord["source_iteration_domain_complete"] =
          task.sourceIterationDomainComplete;
      taskRecord["trip_count_known"] = task.tripCountKnown;
      taskRecord["taskflow_trip_count"] =
          task.tripCountKnown ? json::Value(task.taskflowTripCount)
                              : json::Value(nullptr);
      taskRecord["expanded_internal_extents"] =
          integerArray(task.expandedInternalExtents);
      taskRecord["source_iteration_work_count"] =
          task.tripCountKnown ? json::Value(task.sourceIterationWorkCount)
                              : json::Value(nullptr);
      taskRecord["effective_mapper_firing_count"] =
          task.tripCountKnown ? json::Value(task.tripCount)
                              : json::Value(nullptr);
      if (!task.sourceIterationDomainReason.empty())
        taskRecord["source_iteration_domain_reason"] =
            task.sourceIterationDomainReason;
      bool hasNeuraKernel = false;
      task.op.walk([&](neura::KernelOp) { hasNeuraKernel = true; });
      if (hasNeuraKernel) {
        int64_t requestedAxis = -1;
        // Facts describe the output coordinate being sharded.  A rank-3
        // output may map that dimension to a different Taskflow counter, and
        // Radar's legacy shard_axis is explicitly a counter axis.
        auto outputAxis = task.op->getAttrOfType<IntegerAttr>(
            "amoeba.replica.output_shard_axis");
        auto counterAxis = task.op->getAttrOfType<IntegerAttr>(
            "amoeba.replica.shard_axis");
        bool invalidAxisMetadata =
            (task.op->hasAttr("amoeba.replica.output_shard_axis") &&
             !outputAxis) ||
            (task.op->hasAttr("amoeba.replica.shard_axis") &&
             !counterAxis) ||
            (outputAxis && !counterAxis);
        if (outputAxis)
          requestedAxis = outputAxis.getInt();
        else if (counterAxis)
          requestedAxis = counterAxis.getInt();
        ReplicaOutputCoordinateProof outputProof =
            analyzeReplicaOutputCoordinates(task.op, requestedAxis);
        if (invalidAxisMetadata) {
          outputProof.proven = false;
          outputProof.selectedAxisIndependent = false;
          outputProof.reason = "replica shard-axis metadata is incomplete";
        }
        if (outputProof.proven && outputAxis && counterAxis) {
          if (requestedAxis < 0 ||
              requestedAxis >= static_cast<int64_t>(
                                   outputProof.outputCounterAxes.size()) ||
              outputProof.outputCounterAxes[requestedAxis] ==
                  ReplicaOutputCoordinateProof::kConstantAxis ||
              outputProof.outputCounterAxes[requestedAxis] !=
                  static_cast<unsigned>(counterAxis.getInt())) {
            outputProof.proven = false;
            outputProof.selectedAxisIndependent = false;
            outputProof.reason =
                "replica output shard axis does not map to its counter axis";
          }
        }
        if (outputProof.proven) {
          SmallVector<int64_t> mapping;
          for (unsigned counter : outputProof.outputCounterAxes)
            mapping.push_back(static_cast<int64_t>(counter));
          json::Array accesses;
          for (ArrayRef<unsigned> access : outputProof.outputAccessCounterAxes) {
            SmallVector<int64_t> encoded;
            for (unsigned counter : access)
              encoded.push_back(static_cast<int64_t>(counter));
            accesses.push_back(integerArray(encoded));
          }
          SmallVector<int64_t> auxiliary;
          for (unsigned input : outputProof.auxiliaryInputIndices)
            auxiliary.push_back(static_cast<int64_t>(input));
          taskRecord["neura_output_coordinate_status"] = "proven";
          taskRecord["neura_output_counter_map"] = integerArray(mapping);
          taskRecord["neura_output_counter_accesses"] = std::move(accesses);
          taskRecord["neura_output_produced_shape"] =
              integerArray(outputProof.producedShape);
          taskRecord["neura_output_caller_shape"] =
              integerArray(outputProof.callerShape);
          taskRecord["neura_auxiliary_read_kernel_inputs"] =
              integerArray(auxiliary);
          taskRecord["neura_auxiliary_read_count"] =
              static_cast<int64_t>(auxiliary.size());
          taskRecord["neura_selected_axis_independent"] =
              outputProof.selectedAxisIndependent;
          if (requestedAxis >= 0)
            taskRecord["neura_selected_axis"] = requestedAxis;
        } else {
          // Unsupported or malformed post-Neura routes remain facts-unknown;
          // they never become a legal action merely because a metadata map is
          // present.  Keep the reason for closure diagnostics.
          taskRecord["neura_output_coordinate_status"] = "unknown";
          taskRecord["neura_output_coordinate_reason"] = outputProof.reason;
        }
      }
      if (auto role =
              task.op->getAttrOfType<StringAttr>("amoeba.semantic.task_role"))
        taskRecord["semantic_role"] = role.getValue().str();
      if (auto kind =
              task.op->getAttrOfType<StringAttr>("amoeba.semantic.task_kind"))
        taskRecord["semantic_kind"] = kind.getValue().str();
      auto range =
          task.op->getAttrOfType<DenseI64ArrayAttr>("amoeba.semantic.K_range");
      if (!range)
        range = task.op->getAttrOfType<DenseI64ArrayAttr>(
            "amoeba.semantic.k_range");
      if (range)
        taskRecord["semantic_k_range"] = integerArray(range.asArrayRef());
      auto outputLowers =
          task.op->getAttrOfType<ArrayAttr>(kTilingOutputRegionLowersAttr);
      auto outputUppers =
          task.op->getAttrOfType<ArrayAttr>(kTilingOutputRegionUppersAttr);
      if (outputLowers && outputUppers &&
          outputLowers.size() == outputUppers.size()) {
        json::Array regions;
        for (auto [lowerAttr, upperAttr] :
             llvm::zip(outputLowers, outputUppers)) {
          auto lower = dyn_cast<DenseI64ArrayAttr>(lowerAttr);
          auto upper = dyn_cast<DenseI64ArrayAttr>(upperAttr);
          if (!lower || !upper)
            continue;
          regions.push_back(
              json::Object{{"lower", integerArray(lower.asArrayRef())},
                           {"upper", integerArray(upper.asArrayRef())}});
        }
        taskRecord["output_regions"] = std::move(regions);
      }
      if (std::optional<json::Object> descriptor = tileDescriptor(task.op))
        taskRecord["tile"] = std::move(*descriptor);
      std::string neuraTileError;
      std::optional<json::Object> neuraTile =
          neuraTileDescriptor(task.op, neuraTileError);
      if (!neuraTileError.empty()) {
        task.op.emitError() << neuraTileError;
        return signalPassFailure();
      }
      if (neuraTile)
        taskRecord["neura_tile"] = std::move(*neuraTile);
      std::string neuraFusionError;
      std::optional<json::Object> neuraFusion =
          neuraFusionDescriptor(task.op, neuraFusionError);
      if (!neuraFusionError.empty()) {
        task.op.emitError() << neuraFusionError;
        return signalPassFailure();
      }
      if (neuraFusion)
        taskRecord["neura_fusion"] = std::move(*neuraFusion);
      ClosureStrictKClassification strictK =
          classifyClosureKLegality(task.op);
      ClosureDimensionFacts localLegality =
          inspectClosureTask(task.op, &strictK);
      json::Object localLegalityRecord{
          {"output_rank_proven", localLegality.outputRankKnown},
          {"output_rank",
           localLegality.outputRank},
          {"k_candidate", localLegality.kCandidate},
          {"k_extent_proven", localLegality.kExtentKnown},
          {"cross_cell_k_order", localLegality.crossCellK}};
      if (localLegality.kExtentKnown)
        localLegalityRecord["k_extent"] = localLegality.kExtent;
      taskRecord["source_owned_legality_facts"] =
          std::move(localLegalityRecord);
      if (strictK.potential) {
        if (strictK.provenIllegal) {
          std::string reason = strictK.reason.empty()
                                   ? "proven-illegal-K-recurrence"
                                   : strictK.reason;
          taskRecord["k_reduction_illegal_reason"] = reason;
          kReductionIllegalTasks.push_back(
              json::Object{{"task_name", task.name}, {"reason", reason}});
        } else if (strictK.extentKnown && strictK.extent > 1) {
          kReductionTasks.push_back(json::Object{
              {"static_extent", strictK.extent}, {"task_name", task.name}});
          taskRecord["k_extent_proven"] = strictK.extent;
        } else {
          kReductionFactsComplete = false;
          std::string reason = strictK.reason.empty()
                                   ? "potential-reduction-without-proven-K-extent"
                                   : strictK.reason;
          taskRecord["k_reduction_unproven_reason"] = reason;
          kReductionUnprovenTasks.push_back(
              json::Object{{"task_name", task.name}, {"reason", reason}});
        }
      }
      if (succeeded(domain)) {
        json::Array provenDimensions;
        for (auto [lower, upper] : *domain)
          provenDimensions.push_back(upper - lower);
        taskRecord["static_dims"] = std::move(provenDimensions);
      }
      taskRecords.push_back(std::move(taskRecord));
      shapeNeeds.push_back(json::Object{
          {"task_name", task.name}, {"static_dims", std::move(dimensions)}});
      json::Array inputBindings;
      auto addBinding = [&](StringRef segment, Value input) {
        SmallVector<std::pair<unsigned, std::string>> sources;
        DenseSet<Value> visited;
        sourceTaskIndices(input, indices, sources, visited);
        if (!sources.empty()) {
          inputBindings.push_back((segment + ":dependency").str());
          return;
        }
        Value root = peelChannel(input);
        auto found = externalValueIds.find(root);
        if (found == externalValueIds.end()) {
          inputBindings.push_back((segment + ":unknown").str());
          return;
        }
        inputBindings.push_back(
            (segment + ":external:" + Twine(found->second)).str());
      };
      for (Value input : task.op.getWillReads())
        addBinding("read", input);
      for (Value input : task.op.getWillWrites())
        addBinding("write", input);
      for (Value input : task.op.getValueInputs())
        addBinding("value", input);
      json::Array structuralDimensions;
      if (succeeded(domain))
        for (auto [lower, upper] : *domain)
          structuralDimensions.push_back(upper - lower);
      json::Object structuralNode{
          {"static_dims", std::move(structuralDimensions)},
          {"trip_count", task.tripCountKnown
                              ? json::Value(task.tripCount)
                              : json::Value(nullptr)},
          {"input_bindings", std::move(inputBindings)}};
      structuralNode["trip_count_known"] = task.tripCountKnown;
      structuralNode["source_iteration_domain_status"] =
          task.sourceIterationDomainStatus;
      structuralNode["source_iteration_domain_certified"] =
          task.sourceIterationDomainCertified;
      structuralNode["source_iteration_domain_complete"] =
          task.sourceIterationDomainComplete;
      structuralNode["taskflow_trip_count"] =
          task.tripCountKnown ? json::Value(task.taskflowTripCount)
                              : json::Value(nullptr);
      structuralNode["expanded_internal_extents"] =
          integerArray(task.expandedInternalExtents);
      structuralNode["source_iteration_work_count"] =
          task.tripCountKnown ? json::Value(task.sourceIterationWorkCount)
                              : json::Value(nullptr);
      structuralNode["effective_mapper_firing_count"] =
          task.tripCountKnown ? json::Value(task.tripCount)
                              : json::Value(nullptr);
      if (!task.sourceIterationDomainReason.empty())
        structuralNode["source_iteration_domain_reason"] =
            task.sourceIterationDomainReason;
      if (std::optional<json::Object> descriptor = tileDescriptor(task.op))
        structuralNode["tile"] = std::move(*descriptor);
      if (std::optional<json::Object> descriptor =
              neuraTileDescriptor(task.op, neuraTileError))
        structuralNode["neura_tile"] = std::move(*descriptor);
      if (std::optional<json::Object> descriptor =
              neuraFusionDescriptor(task.op, neuraFusionError))
        structuralNode["neura_fusion"] = std::move(*descriptor);
      structuralNodes.push_back(std::move(structuralNode));

      // A post-conversion mixed graph has a neura.kernel in place of the
      // Taskflow hyperblock. Its typed Taskflow edges and task metadata are
      // still authoritative, but the pre-conversion rewrite contracts cannot
      // be applied to it. Fact-only mode intentionally emits those facts and
      // leaves action generation to a Neura-level MLIR pass.
      if (factOnly)
        continue;

      if (maxFissionActionsPerTask > 0) {
        FailureOr<SmallVector<SmallVector<unsigned>>> fissionPartitions =
            legalFissionPartitions(task.op, *canonical,
                                   maxFissionActionsPerTask, error);
        if (failed(fissionPartitions)) {
          task.op.emitError() << error;
          return signalPassFailure();
        }
        for (ArrayRef<unsigned> leftNodes : *fissionPartitions) {
          json::Array ordinals;
          for (unsigned ordinal : leftNodes)
            ordinals.push_back(static_cast<int64_t>(ordinal));
          actions.push_back(json::Object{{"kind", "fission"},
                                         {"task_name", task.name},
                                         {"left_nodes", std::move(ordinals)}});
        }
      }
      for (size_t axis = 0; axis < canonical->counters.size(); ++axis) {
        ++consideredTiling;
        std::string rejectionReason;
        bool accepted =
            tilingContract(task.op, *canonical, axis, &rejectionReason);
        if (accepted) {
          ++acceptedTiling;
          int64_t extent = canonical->upper[axis] - canonical->lower[axis];
          int64_t limit = std::min<int64_t>(maxTileFactor, extent);
          for (int64_t factor = 2; factor <= limit; ++factor)
            actions.push_back(json::Object{{"kind", "tiling"},
                                           {"task_name", task.name},
                                           {"axis", static_cast<int64_t>(axis)},
                                           {"factor", factor}});
        } else {
          ++tilingRejections[rejectionReason];
        }
      }
    }
    if (factOnly) {
      // Fact-only mode still enumerates the complete sibling pair space. A
      // pair with an unresolved domain is left out only with an explicit
      // incomplete marker; it is never silently treated as illegal.
      for (size_t firstIndex = 0; firstIndex < metadata->size(); ++firstIndex)
        for (size_t secondIndex = firstIndex + 1;
             secondIndex < metadata->size(); ++secondIndex) {
          TaskflowTaskOp first = (*metadata)[firstIndex].op;
          TaskflowTaskOp second = (*metadata)[secondIndex].op;
          json::Object scanned{
              {"first_task", (*metadata)[firstIndex].name},
              {"second_task", (*metadata)[secondIndex].name}};
          if (failed(loopDomain(first)) || failed(loopDomain(second))) {
            siblingFusionFactsComplete = false;
            siblingFusionMaterializationComplete = false;
            scanned["source_status"] = "unknown";
            scanned["materialization_status"] = "unknown";
            scanned["status"] = "unknown";
            scanned["reason"] = "missing_static_loop_domain";
            siblingFusionScannedPairs.push_back(std::move(scanned));
            continue;
          }
          if (!independentSiblings(first, second)) {
            scanned["source_status"] = "illegal";
            scanned["materialization_status"] = "not_applicable";
            scanned["status"] = "illegal";
            scanned["reason"] = "sibling_contract_rejected";
            siblingFusionScannedPairs.push_back(std::move(scanned));
            continue;
          }
          SiblingMaterializationStatus materialization =
              siblingMaterializationStatus(first, second);
          json::Object candidate = siblingFusionCandidate(
              first, second, (*metadata)[firstIndex].name,
              (*metadata)[secondIndex].name);
          if (materialization.status != "legal")
            siblingFusionMaterializationComplete = false;
          scanned["source_status"] = "legal";
          scanned["materialization_status"] = materialization.status;
          scanned["status"] = materialization.status;
          scanned["reason"] = materialization.reason;
          siblingFusionScannedPairs.push_back(std::move(scanned));
          siblingFusionCandidates.push_back(std::move(candidate));
        }
    }

    if (!factOnly)
      for (size_t firstIndex = 0; firstIndex < metadata->size(); ++firstIndex)
        for (size_t secondIndex = firstIndex + 1;
             secondIndex < metadata->size(); ++secondIndex) {
        TaskflowTaskOp first = (*metadata)[firstIndex].op;
        TaskflowTaskOp second = (*metadata)[secondIndex].op;
        ++consideredFusion;
        bool producerConsumer = directProducerConsumer(first, second);
        if (producerConsumer) {
          ++acceptedFusion;
          actions.push_back(fusionAction("producer_consumer",
                                         (*metadata)[firstIndex].name,
                                         (*metadata)[secondIndex].name));
        } else {
          ++fusionRejections["producer_consumer_contract_rejected"];
        }
        ++consideredFusion;
        bool siblings = independentSiblings(first, second);
        if (siblings) {
          ++acceptedFusion;
          actions.push_back(fusionAction("sibling",
                                         (*metadata)[firstIndex].name,
                                         (*metadata)[secondIndex].name));
        } else {
          ++fusionRejections["sibling_contract_rejected"];
        }
        std::optional<DirectRawConnection> rawConnection =
            directRawProducerConsumer(first, second);
        FailureOr<CanonicalTask> firstCanonical = canonicalTask(first);
        FailureOr<CanonicalTask> secondCanonical = canonicalTask(second);
        // Equal factors do not require equal loop bounds. Interior stencil
        // domains intentionally differ from their producers, and layout
        // changes can map a producer dimension to a differently numbered
        // consumer counter. After both rewrites, the typed-edge builder proves
        // exact read/write intersections or conservatively expands the join.
        size_t producerAxes =
            succeeded(firstCanonical) ? firstCanonical->counters.size() : 0;
        size_t consumerAxes =
            succeeded(secondCanonical) ? secondCanonical->counters.size() : 0;
        for (size_t producerAxis = 0; producerAxis < producerAxes;
             ++producerAxis)
          for (size_t consumerAxis = 0; consumerAxis < consumerAxes;
               ++consumerAxis) {
            ++consideredCotiling;
            std::string rejectionReason;
            if (!rawConnection) {
              rejectionReason = "no_direct_raw_connection";
            } else if (!tilingContract(first, *firstCanonical, producerAxis,
                                       &rejectionReason)) {
              rejectionReason = "producer_" + rejectionReason;
            } else if (!tilingContract(second, *secondCanonical, consumerAxis,
                                       &rejectionReason)) {
              rejectionReason = "consumer_" + rejectionReason;
            } else if (cotilingAxisCompatibility(
                           first, *firstCanonical, producerAxis, second,
                           *secondCanonical, consumerAxis, *rawConnection) ==
                       CotilingAxisCompatibility::Incompatible) {
              rejectionReason = "shared_storage_axis_mismatch";
            }
            if (!rejectionReason.empty()) {
              ++cotilingRejections[rejectionReason];
              continue;
            }
            ++acceptedCotiling;
            int64_t producerExtent = firstCanonical->upper[producerAxis] -
                                     firstCanonical->lower[producerAxis];
            int64_t consumerExtent = secondCanonical->upper[consumerAxis] -
                                     secondCanonical->lower[consumerAxis];
            int64_t limit = std::min({static_cast<int64_t>(maxTileFactor),
                                      producerExtent, consumerExtent});
            for (int64_t factor = 2; factor <= limit; ++factor)
              actions.push_back(cotilingAction(
                  (*metadata)[firstIndex].name, (*metadata)[secondIndex].name,
                  producerAxis, consumerAxis, factor));
          }
        }

    auto addActionSummary = [&](StringRef kind, int64_t considered,
                                int64_t accepted,
                                const llvm::StringMap<int64_t> &rejections) {
      SmallVector<StringRef> reasons;
      for (const auto &entry : rejections)
        reasons.push_back(entry.getKey());
      llvm::sort(reasons);
      json::Array reasonRecords;
      for (StringRef reason : reasons)
        reasonRecords.push_back(json::Object{
            {"reason", reason}, {"count", rejections.lookup(reason)}});
      StringRef summary = reasons.empty()       ? "none"
                          : reasons.size() == 1 ? reasons.front()
                                                : "multiple";
      actionDiagnostics.push_back(
          json::Object{{"kind", kind},
                       {"considered", considered},
                       {"accepted", accepted},
                       {"rejected", considered - accepted},
                       {"rejection_reason", summary},
                       {"rejection_reasons", std::move(reasonRecords)}});
    };
    addActionSummary("tiling", consideredTiling, acceptedTiling,
                     tilingRejections);
    addActionSummary("fusion", consideredFusion, acceptedFusion,
                     fusionRejections);
    addActionSummary("cotiling", consideredCotiling, acceptedCotiling,
                     cotilingRejections);

    json::Array edgeRecords;
    json::Array structuralEdges;
    for (const EdgeFact &edge : *edges) {
      json::Object edgeRecord{
          {"source_task", (*metadata)[edge.source].name},
          {"target_task", (*metadata)[edge.target].name},
          {"kind", edge.kind},
          {"producer_segment", edge.producerSegment},
          {"producer_index", edge.producerIndex},
          {"consumer_segment", edge.consumerSegment},
          {"consumer_index", edge.consumerIndex},
          {"origin", edge.origin},
          {"scope", edge.scope},
          {"semantic_role", edge.semanticRole.empty()
                                ? json::Value(nullptr)
                                : json::Value(edge.semanticRole)},
          {"semantic_scope", edge.semanticScope.empty()
                                 ? json::Value(nullptr)
                                 : json::Value(edge.semanticScope)},
          {"region_lower", nullptr},
          {"region_upper", nullptr}};
      json::Object structuralEdge{
          {"source", static_cast<int64_t>(edge.source)},
          {"target", static_cast<int64_t>(edge.target)},
          {"kind", edge.kind},
          {"producer_segment", edge.producerSegment},
          {"producer_index", edge.producerIndex},
          {"consumer_segment", edge.consumerSegment},
          {"consumer_index", edge.consumerIndex},
          {"origin", edge.origin},
          {"scope", edge.scope},
          {"semantic_role", edge.semanticRole.empty()
                                ? json::Value(nullptr)
                                : json::Value(edge.semanticRole)},
          {"semantic_scope", edge.semanticScope.empty()
                                 ? json::Value(nullptr)
                                 : json::Value(edge.semanticScope)},
          {"region_lower", nullptr},
          {"region_upper", nullptr}};
      if (edge.payloadBits) {
        edgeRecord["payload_bits"] = static_cast<int64_t>(*edge.payloadBits);
        edgeRecord["payload_bytes"] = static_cast<int64_t>(
            *edge.payloadBits / 8 + (*edge.payloadBits % 8 != 0));
        structuralEdge["payload_bits"] =
            static_cast<int64_t>(*edge.payloadBits);
      } else {
        edgeRecord["payload_bits"] = nullptr;
        edgeRecord["payload_bytes"] = nullptr;
        structuralEdge["payload_bits"] = nullptr;
      }
      if (!edge.regionLower.empty()) {
        edgeRecord["region_lower"] = integerArray(edge.regionLower);
        edgeRecord["region_upper"] = integerArray(edge.regionUpper);
        structuralEdge["region_lower"] = integerArray(edge.regionLower);
        structuralEdge["region_upper"] = integerArray(edge.regionUpper);
      }
      edgeRecords.push_back(std::move(edgeRecord));
      structuralEdges.push_back(std::move(structuralEdge));
    }
    json::Array completionJoins;
    DenseMap<Operation *, int64_t> completionJoinIndices;
    int64_t joinIndex = 0;
    selected->walk([&](TaskflowJoinOp join) {
      completionJoinIndices[join.getOperation()] = joinIndex;
      completionJoins.push_back(json::Object{
          {"join_index", joinIndex++},
          {"axis", join.getAxis()},
          {"region_lower", integerArray(join.getRegionLower())},
          {"region_upper", integerArray(join.getRegionUpper())},
          {"producer_tiles", static_cast<int64_t>(join.getTileStates().size())},
          {"users",
           static_cast<int64_t>(std::distance(join.getJoined().use_begin(),
                                              join.getJoined().use_end()))}});
    });
    json::Array readCompletionJoins;
    int64_t readJoinIndex = 0;
    selected->walk([&](TaskflowReadCompletionJoinOp join) {
      auto completion =
          join.getWriteCompletion().getDefiningOp<TaskflowJoinOp>();
      int64_t completionIndex = -1;
      if (completion)
        if (auto found = completionJoinIndices.find(completion.getOperation());
            found != completionJoinIndices.end())
          completionIndex = found->second;
      auto task = join.getTileStates().front().getDefiningOp<TaskflowTaskOp>();
      unsigned readResult = 0;
      if (task) {
        auto result = llvm::find(task.getDoneReads(),
                                 join.getTileStates().front());
        if (result != task.getDoneReads().end())
          readResult = result - task.getDoneReads().begin();
      }
      readCompletionJoins.push_back(json::Object{
          {"read_join_index", readJoinIndex++},
          {"read_result_index", static_cast<int64_t>(readResult)},
          {"replica_read_states",
           static_cast<int64_t>(join.getTileStates().size())},
          {"linked_output_completion_join", completionIndex},
          {"users", static_cast<int64_t>(std::distance(
                         join.getJoined().use_begin(),
                         join.getJoined().use_end()))}});
    });
    json::Object root{
        {"schema", kFactsSchema},
        {"mode", factOnly ? "fact_only" : "canonical"},
        {"function", selected->getSymName()},
        {"tasks", std::move(taskRecords)},
        {"dependencies", std::move(edgeRecords)},
        {"shape_needs", std::move(shapeNeeds)},
        {"k_reduction_tasks", std::move(kReductionTasks)},
        {"k_reduction_tasks_complete", kReductionFactsComplete},
        {"k_reduction_unproven_tasks", std::move(kReductionUnprovenTasks)},
        {"k_reduction_illegal_tasks", std::move(kReductionIllegalTasks)},
        {"sibling_fusion_candidates", std::move(siblingFusionCandidates)},
        {"sibling_fusion_scanned_pairs", std::move(siblingFusionScannedPairs)},
        {"sibling_fusion_candidates_complete", siblingFusionFactsComplete},
        {"sibling_fusion_materialization_complete",
         siblingFusionMaterializationComplete},
        {"completion_joins", std::move(completionJoins)},
        {"read_completion_joins", std::move(readCompletionJoins)},
        {"action_diagnostics", std::move(actionDiagnostics)},
        {"structural_key", json::Object{{"nodes", std::move(structuralNodes)},
                                        {"edges", std::move(structuralEdges)}}},
        {"legal_actions", std::move(actions)}};
    if (!writeAtomically(
            outputFile.getValue(),
            [&](llvm::raw_ostream &os) {
              os << json::Value(std::move(root)) << "\n";
              return true;
            },
            error)) {
      selected->emitError() << error;
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::amoeba::neura::joint_scheduling {

ClosureStrictKClassification
classifyClosureKLegality(TaskflowTaskOp task) {
  KReductionProof proof = proveKReduction(task);
  return {proof.potential, proof.extent.has_value(),
          proof.extent.value_or(0), proof.provenIllegal,
          proof.unprovenReason};
}

} // namespace mlir::amoeba::neura::joint_scheduling

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createExtractJointTaskGraphFactsPass() {
  return std::make_unique<ExtractJointTaskGraphFactsPass>();
}
} // namespace mlir::amoeba::neura
