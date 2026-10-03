//===- JointNeighborhoodActions.cpp --------------------------------------===//
// Bounded, source-owned local actions for the ORBIT neighborhood search.
//
// This file deliberately contains a small action controller only.  It does
// not build a graph closure and it does not rank candidates.  Every rewrite
// is delegated to the existing Neura materializer pass and is committed only
// after the complete trial module verifies.
//===----------------------------------------------------------------------===//

#include "JointNeighborhoodActions.h"

#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <set>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {

struct ShapeKey {
  int64_t rows = 1;
  int64_t cols = 1;
  bool operator==(const ShapeKey &other) const {
    return rows == other.rows && cols == other.cols;
  }
};

static bool isLegalArea4Shape(int64_t rows, int64_t cols) {
  if (rows <= 0 || cols <= 0 || rows > 4 / cols)
    return false;
  // The local search uses oriented rectangles.  The explicit test also keeps
  // malformed caller-provided shape state from leaking into a candidate.
  return (rows == 1 && (cols == 1 || cols == 2 || cols == 3 || cols == 4)) ||
         (cols == 1 && (rows == 2 || rows == 3 || rows == 4)) ||
         (rows == 2 && cols == 2);
}

static std::string lower(StringRef value) {
  std::string result = value.str();
  for (char &character : result)
    character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  return result;
}

static unsigned stageNumber(StringRef stage) {
  std::string value = lower(stage);
  // These are the protocol's explicit names.  Keep them ahead of substring
  // matching so a future descriptive stage name cannot silently lose a
  // dimension.
  if (value == "s5" || value == "full-joint" ||
      value == "shape-temporal-replica-tiling-fusion")
    return 5;
  if (value == "s4" || value == "shape-temporal-replica-tiling")
    return 4;
  if (value == "s3" || value == "shape-temporal-replica")
    return 3;
  if (value == "s2" || value == "shape-temporal" ||
      value == "shape-temporal-critical-path" ||
      value == "shape-temporal-critical")
    return 2;
  if (value == "s1" || value == "shape" || value == "shape-only" ||
      value == "shape-temporal-fixed-dispatch")
    return 1;
  if (value.find("s5") != std::string::npos ||
      value.find("fusion") != std::string::npos)
    return 5;
  if (value.find("s4") != std::string::npos ||
      value.find("til") != std::string::npos)
    return 4;
  if (value.find("s3") != std::string::npos ||
      value.find("replica") != std::string::npos)
    return 3;
  if (value.find("s2") != std::string::npos ||
      value.find("critical") != std::string::npos)
    return 2;
  return 1;
}

struct DependencyPair {
  std::string producer;
  std::string consumer;
};

static TaskflowTaskOp definingTask(Value value) {
  while (auto channel = value.getDefiningOp<TaskflowChannelOp>())
    value = channel.getSource();
  return value.getDefiningOp<TaskflowTaskOp>();
}

static SmallVector<DependencyPair> dependencyPairs(ModuleOp module,
                                                   StringRef functionName) {
  SmallVector<DependencyPair> result;
  std::string error;
  FailureOr<func::FuncOp> function =
      selectTaskFunction(module, functionName, error);
  if (failed(function))
    return result;
  function->walk([&](TaskflowTaskOp consumer) {
    SmallVector<std::string> producers;
    auto collect = [&](ValueRange values) {
      for (Value value : values) {
        TaskflowTaskOp producer = definingTask(value);
        if (producer && producer != consumer &&
            !llvm::is_contained(producers, producer.getTaskName().str()))
          producers.push_back(producer.getTaskName().str());
      }
    };
    collect(consumer.getWillReads());
    collect(consumer.getValueInputs());
    llvm::sort(producers);
    for (const std::string &producer : producers)
      result.push_back({producer, consumer.getTaskName().str()});
  });
  llvm::sort(result, [](const DependencyPair &lhs, const DependencyPair &rhs) {
    if (lhs.producer != rhs.producer)
      return lhs.producer < rhs.producer;
    return lhs.consumer < rhs.consumer;
  });
  return result;
}

static bool hasTask(ModuleOp module, StringRef functionName, StringRef name) {
  std::string error;
  FailureOr<func::FuncOp> function =
      selectTaskFunction(module, functionName, error);
  if (failed(function))
    return false;
  bool found = false;
  function->walk([&](TaskflowTaskOp task) {
    if (task.getTaskName() == name)
      found = true;
  });
  return found;
}

static TaskflowTaskOp findTask(ModuleOp module, StringRef functionName,
                               StringRef name) {
  std::string error;
  FailureOr<func::FuncOp> function =
      selectTaskFunction(module, functionName, error);
  if (failed(function))
    return nullptr;
  TaskflowTaskOp result;
  function->walk([&](TaskflowTaskOp task) {
    if (!result && task.getTaskName() == name)
      result = task;
  });
  return result;
}

static SmallVector<std::string> taskNames(ModuleOp module,
                                           StringRef functionName) {
  SmallVector<std::string> result;
  std::string error;
  FailureOr<func::FuncOp> function =
      selectTaskFunction(module, functionName, error);
  if (failed(function))
    return result;
  function->walk([&](TaskflowTaskOp task) {
    result.push_back(task.getTaskName().str());
  });
  llvm::sort(result);
  return result;
}

static void addUniqueShape(SmallVectorImpl<ShapeKey> &shapes, int64_t rows,
                           int64_t cols) {
  ShapeKey candidate{rows, cols};
  if (isLegalArea4Shape(rows, cols) &&
      !llvm::is_contained(shapes, candidate))
    shapes.push_back(candidate);
}

static void addUniqueAction(std::vector<NeighborhoodAction> &actions,
                            NeighborhoodAction action) {
  // Labels are stable path identities.  Keeping this check local to the
  // enumerator prevents the same pair/order from consuming beam slots twice.
  if (llvm::any_of(actions, [&](const NeighborhoodAction &existing) {
        return existing.label == action.label;
      }))
    return;
  actions.push_back(std::move(action));
}

static NeighborhoodPrimitive tilePrimitive(StringRef task, int64_t axis,
                                            int64_t factor) {
  NeighborhoodPrimitive primitive;
  primitive.kind = "tile";
  primitive.firstTask = task.str();
  primitive.mode = "none";
  primitive.axis = axis;
  primitive.factor = factor;
  return primitive;
}

static NeighborhoodPrimitive kTilePrimitive(StringRef task, StringRef mode,
                                             int64_t factor) {
  NeighborhoodPrimitive primitive;
  primitive.kind = "k-tile";
  primitive.firstTask = task.str();
  primitive.mode = mode.str();
  primitive.axis = 2;
  primitive.factor = factor;
  return primitive;
}

static NeighborhoodPrimitive replicaPrimitive(StringRef task, int64_t axis,
                                               int64_t factor) {
  NeighborhoodPrimitive primitive;
  primitive.kind = "replica";
  primitive.firstTask = task.str();
  primitive.axis = axis;
  primitive.factor = factor;
  return primitive;
}

static NeighborhoodPrimitive fusionPrimitive(StringRef first, StringRef second,
                                              StringRef mode) {
  NeighborhoodPrimitive primitive;
  primitive.kind = mode == "sibling" ? "sibling-fusion" : "fusion";
  primitive.firstTask = first.str();
  primitive.secondTask = second.str();
  primitive.mode = mode.str();
  primitive.factor = 1;
  return primitive;
}

static std::string fusedTaskName(StringRef first, StringRef second) {
  return (Twine(first) + ".fuse." + Twine(second)).str();
}

static void initialShapes(ModuleOp module, StringRef functionName,
                          std::vector<NeighborhoodShape> &result) {
  result.clear();
  for (const std::string &name : taskNames(module, functionName))
    result.push_back(NeighborhoodShape{name, 1, 1});
}

static const NeighborhoodShape *findShape(ArrayRef<NeighborhoodShape> shapes,
                                          StringRef task) {
  for (const NeighborhoodShape &shape : shapes)
    if (shape.task == task)
      return &shape;
  return nullptr;
}

static void inheritShapeForName(StringRef name,
                                ArrayRef<NeighborhoodShape> oldShapes,
                                NeighborhoodShape &shape) {
  shape.task = name.str();
  shape.rows = 1;
  shape.cols = 1;

  // Materializers encode ancestry in their source-owned task names.  If a
  // derived task has exactly one obvious parent, carry that parent's shape;
  // otherwise the safe default is 1x1 and the next local action may choose a
  // new oriented shape explicitly.
  auto inheritPrefix = [&](StringRef marker) {
    size_t position = name.find(marker);
    if (position == StringRef::npos)
      return false;
    StringRef parent = name.take_front(position);
    if (const NeighborhoodShape *old = findShape(oldShapes, parent)) {
      shape.rows = old->rows;
      shape.cols = old->cols;
      return true;
    }
    return false;
  };
  if (inheritPrefix(".tile.") || inheritPrefix(".replica."))
    return;
  size_t fusion = name.find(".fuse.");
  if (fusion != StringRef::npos) {
    StringRef first = name.take_front(fusion);
    StringRef rest = name.drop_front(fusion + 6);
    const NeighborhoodShape *left = findShape(oldShapes, first);
    const NeighborhoodShape *right = findShape(oldShapes, rest);
    if (left && right && left->rows == right->rows &&
        left->cols == right->cols) {
      shape.rows = left->rows;
      shape.cols = left->cols;
    } else if (left) {
      shape.rows = left->rows;
      shape.cols = left->cols;
    } else if (right) {
      shape.rows = right->rows;
      shape.cols = right->cols;
    }
  }
}

static void synchronizeShapes(ModuleOp module, StringRef functionName,
                              ArrayRef<NeighborhoodShape> oldShapes,
                              std::vector<NeighborhoodShape> &result) {
  result.clear();
  for (const std::string &name : taskNames(module, functionName)) {
    if (const NeighborhoodShape *old = findShape(oldShapes, name)) {
      result.push_back(*old);
      continue;
    }
    NeighborhoodShape inherited;
    inheritShapeForName(name, oldShapes, inherited);
    result.push_back(std::move(inherited));
  }
}

static bool runMaterializer(ModuleOp module, StringRef functionName,
                            const NeighborhoodPrimitive &primitive,
                            std::string &reason, std::string &diagnostic) {
  std::unique_ptr<Pass> pass;
  std::string options;
  std::string kind = lower(primitive.kind);
  if (kind == "tile" || kind == "tiling") {
    if (primitive.axis < 0 || primitive.axis > 1 ||
        (primitive.factor != 2 && primitive.factor != 4)) {
      reason = "unsupported_tiling_factor_or_axis";
      return false;
    }
    pass = mlir::amoeba::neura::createMaterializeNeuraJointRewritePass();
    options = "task-name=\"" + primitive.firstTask + "\" tile-axis=" +
              std::to_string(primitive.axis) + " tile-factor=" +
              std::to_string(primitive.factor) + " fusion-mode=none";
  } else if (kind == "k-tile" || kind == "k_tiling" || kind == "ktile") {
    if (primitive.factor != 2 && primitive.factor != 4) {
      reason = "unsupported_k_tiling_factor";
      return false;
    }
    std::string mode = lower(primitive.mode);
    if (mode.empty())
      mode = "sequential";
    if (mode != "sequential" && mode != "parallel-linear" &&
        mode != "parallel-tree") {
      reason = "unsupported_k_reduction_mode";
      return false;
    }
    pass = mlir::amoeba::neura::createMaterializeNeuraKReductionPass();
    options = "function=\"" + functionName.str() + "\" task-name=\"" +
              primitive.firstTask + "\" k-mode=" + mode + " k-factor=" +
              std::to_string(primitive.factor);
  } else if (kind == "replica" || kind == "replication") {
    if (primitive.axis < 0 || primitive.axis > 1 ||
        (primitive.factor != 2 && primitive.factor != 4)) {
      reason = "unsupported_replica_factor_or_axis";
      return false;
    }
    pass = mlir::amoeba::neura::createMaterializeJointTaskReplicasPass();
    options = "function=\"" + functionName.str() + "\" task=\"" +
              primitive.firstTask + "\" replicas=" +
              std::to_string(primitive.factor) + " axis=" +
              std::to_string(primitive.axis);
  } else if (kind == "fusion" || kind == "producer-consumer" ||
             kind == "producer_consumer" || kind == "sibling-fusion" ||
             kind == "sibling") {
    std::string mode = lower(primitive.mode);
    if (kind == "sibling-fusion" || kind == "sibling")
      mode = "sibling";
    if (mode != "sibling" && mode != "producer-consumer-retained" &&
        mode != "producer-consumer-forwarded") {
      reason = "unsupported_fusion_mode";
      return false;
    }
    pass = mlir::amoeba::neura::createMaterializeNeuraJointRewritePass();
    options = "first-task-name=\"" + primitive.firstTask +
              "\" second-task-name=\"" + primitive.secondTask +
              "\" fusion-mode=" + mode;
  } else if (kind == "noop" || kind == "identity") {
    return true;
  } else {
    reason = "unsupported_neighborhood_action_kind";
    return false;
  }

  if (!pass) {
    reason = "missing_source_owned_materializer";
    return false;
  }
  if (failed(pass->initializeOptions(options, [&](const llvm::Twine &message) {
        reason = message.str();
        return failure();
      }))) {
    if (reason.empty())
      reason = "source_owned_materializer_option_parse_failed";
    return false;
  }

  PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(std::move(pass));
  LogicalResult runResult = failure();
  LogicalResult verifyResult = failure();
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
    runResult = manager.run(module);
    if (succeeded(runResult))
      verifyResult = verify(module.getOperation());
  }
  if (failed(runResult)) {
    reason = "source_owned_materializer_rejected";
    if (diagnostic.empty())
      diagnostic = "source-owned materializer failed without a diagnostic";
    return false;
  }
  if (failed(verifyResult)) {
    reason = "materialized_module_verifier_failed";
    return false;
  }
  return true;
}

static bool copyModuleBody(ModuleOp destination, ModuleOp source,
                           std::string &reason) {
  if (!destination || !source || destination.getOperation() == source.getOperation()) {
    reason = "invalid_module_reset_target";
    return false;
  }
  SmallVector<Operation *> operations;
  for (Operation &operation : source.getBody()->getOperations())
    operations.push_back(operation.clone());
  destination->setAttrs(source->getAttrs());
  destination.getBody()->clear();
  for (Operation *operation : operations)
    destination.getBody()->push_back(operation);
  if (failed(verify(destination.getOperation()))) {
    reason = "committed_module_verifier_failed";
    return false;
  }
  return true;
}

static bool isShapePrimitive(StringRef kind) {
  std::string value = lower(kind);
  return value == "shape" || value == "shape-choice" || value == "shape_choice";
}

static bool actionHasDuplicateLineageFactor(
    ArrayRef<NeighborhoodPrimitive> primitives, std::string &reason) {
  // A local composite may contain two edits to the same original dimension.
  // Reject only a provable product above four; all other unsupported cases are
  // left to the source materializer and reported as unknown there.
  // Tiling partitions the original task domain as one area budget across M,
  // N, and K choices.  Replication is a separate budget by design, so a
  // tile factor and a replica factor do not multiply each other here.
  std::map<std::pair<std::string, std::string>, int64_t> products;
  for (const NeighborhoodPrimitive &primitive : primitives) {
    std::string kind = lower(primitive.kind);
    std::string family;
    if (kind == "tile" || kind == "tiling" || kind == "k-tile" ||
        kind == "k_tiling" || kind == "ktile")
      family = "tiling";
    else if (kind == "replica" || kind == "replication")
      family = "replica";
    else
      continue;
    if (primitive.factor <= 0) {
      reason = "invalid_non_positive_action_factor";
      return false;
    }
    auto key = std::make_pair(primitive.firstTask, family);
    int64_t &product = products[key];
    if (product == 0)
      product = 1;
    if (product > 4 / primitive.factor) {
      reason = "cumulative_original_domain_factor_exceeds_four";
      return false;
    }
    product *= primitive.factor;
  }
  return true;
}

// This ledger is emitted only after a source materializer has changed the
// actual typed counter domains. Names are never a partition-factor proof.
constexpr StringLiteral kPartitionLineage =
    "amoeba.neighborhood.partition_lineage.v1";

struct CounterDomain {
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 1;
  bool operator==(const CounterDomain &other) const {
    return lower == other.lower && upper == other.upper && step == other.step;
  }
};
using CounterDomains = SmallVector<CounterDomain>;

struct PartitionStep {
  std::string family;
  int64_t axis = 0;
  int64_t factor = 1;
  int64_t part = 0;
  CounterDomains before;
  CounterDomains after;
};
struct RootLineage {
  std::string root;
  CounterDomains original;
  SmallVector<PartitionStep, 2> steps;
};
using PartitionLineage = SmallVector<RootLineage, 2>;

static std::optional<int64_t> staticIndex(Value value, TaskflowTaskOp task,
                                         neura::KernelOp kernel = nullptr,
                                         unsigned depth = 0) {
  if (!value || depth > 32)
    return std::nullopt;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return staticIndex(cast.getIn(), task, kernel, depth + 1);
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = staticIndex(add.getLhs(), task, kernel, depth + 1);
    auto rhs = staticIndex(add.getRhs(), task, kernel, depth + 1);
    if (!lhs || !rhs ||
        (*rhs > 0 && *lhs > std::numeric_limits<int64_t>::max() - *rhs) ||
        (*rhs < 0 && *lhs < std::numeric_limits<int64_t>::min() - *rhs))
      return std::nullopt;
    int64_t sum = *lhs + *rhs;
    if (auto type = dyn_cast<IntegerType>(add.getType())) {
      unsigned width = type.getWidth();
      if (width == 0 || width > 64)
        return std::nullopt;
      if (width < 64) {
        int64_t high = (int64_t(1) << (width - 1)) - 1;
        if (sum < -high - 1 || sum > high)
          return std::nullopt;
      }
    } else if (!add.getType().isIndex()) {
      return std::nullopt;
    }
    return sum;
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    unsigned index = argument.getArgNumber();
    if (kernel && kernel.getBody().hasOneBlock() &&
        argument.getOwner() == &kernel.getBody().front()) {
      if (index >= kernel.getInputs().size())
        return std::nullopt;
      return staticIndex(kernel.getInputs()[index], task, nullptr, depth + 1);
    }
    if (task && task.getBody().hasOneBlock() &&
        argument.getOwner() == &task.getBody().front()) {
      unsigned first = task.getWillReads().size() + task.getWillWrites().size();
      if (index < first || index - first >= task.getValueInputs().size())
        return std::nullopt;
      return staticIndex(task.getValueInputs()[index - first], task, nullptr,
                         depth + 1);
    }
  }
  return std::nullopt;
}

static std::optional<int64_t> staticKernelBound(
    neura::CounterOp counter, StringRef name, Value operand,
    TaskflowTaskOp task, neura::KernelOp kernel) {
  std::optional<int64_t> fromOperand;
  if (operand) {
    fromOperand = staticIndex(operand, task, kernel);
    if (!fromOperand)
      return std::nullopt;
  }
  std::optional<int64_t> fromAttribute;
  if (Attribute attribute = counter->getAttr(name)) {
    if (auto integer = dyn_cast<IntegerAttr>(attribute)) {
      fromAttribute = integer.getInt();
    } else if (auto text = dyn_cast<StringAttr>(attribute)) {
      StringRef spelling = text.getValue();
      unsigned input = 0;
      if (!spelling.consume_front("%input") || spelling.empty() ||
          spelling.getAsInteger(10, input) || input >= kernel.getInputs().size())
        return std::nullopt;
      fromAttribute = staticIndex(kernel.getInputs()[input], task, kernel);
      if (!fromAttribute)
        return std::nullopt;
    } else {
      return std::nullopt;
    }
  }
  if (fromOperand && fromAttribute && *fromOperand != *fromAttribute)
    return std::nullopt;
  return fromAttribute ? fromAttribute : fromOperand;
}

static bool collectTypedDomains(TaskflowTaskOp task, CounterDomains &domains,
                                std::string &reason) {
  domains.clear();
  if (!task || !task.getBody().hasOneBlock()) {
    reason = "lineage_proof_unknown";
    return false;
  }
  SmallVector<TaskflowCounterOp> taskCounters;
  SmallVector<neura::CounterOp> kernelCounters;
  neura::KernelOp kernel;
  for (Operation &operation : task.getBody().front()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(operation))
      taskCounters.push_back(counter);
    if (auto found = dyn_cast<neura::KernelOp>(operation)) {
      if (kernel) {
        reason = "lineage_proof_unknown";
        return false;
      }
      kernel = found;
    }
  }
  if (!kernel || !kernel.getBody().hasOneBlock() || taskCounters.empty()) {
    reason = "lineage_proof_unknown";
    return false;
  }
  for (Operation &operation : kernel.getBody().front())
    if (auto counter = dyn_cast<neura::CounterOp>(operation))
      kernelCounters.push_back(counter);
  if (taskCounters.size() != kernelCounters.size()) {
    reason = "lineage_proof_unknown";
    return false;
  }
  for (auto [index, counter] : llvm::enumerate(taskCounters)) {
    auto low = staticIndex(counter.getLowerBound(), task);
    auto high = staticIndex(counter.getUpperBound(), task);
    auto step = staticIndex(counter.getStep(), task);
    bool chainMatches = index == 0
                            ? !counter.getParentIndex()
                            : counter.getParentIndex() ==
                                  taskCounters[index - 1].getCounterIndex();
    if (!low || !high || !step || *low < 0 || *high <= *low || *step != 1 ||
        !chainMatches) {
      reason = "lineage_proof_unknown";
      return false;
    }
    for (Operation *operation : {counter.getOperation(),
                                 kernelCounters[index].getOperation()}) {
      if (auto id = operation->getAttrOfType<IntegerAttr>("counter_id"))
        if (id.getInt() != static_cast<int64_t>(index)) {
          reason = "lineage_proof_unknown";
          return false;
        }
    }
    neura::CounterOp hardware = kernelCounters[index];
    auto hardwareLow = staticKernelBound(hardware, "lower_bound_value",
                                        hardware.getLowerBound(), task, kernel);
    auto hardwareHigh = staticKernelBound(hardware, "upper_bound_value",
                                         hardware.getUpperBound(), task, kernel);
    auto hardwareStep = staticKernelBound(hardware, "step_value",
                                         hardware.getStep(), task, kernel);
    if (!hardwareLow || !hardwareHigh || !hardwareStep ||
        *hardwareLow != *low || *hardwareHigh != *high ||
        *hardwareStep != *step) {
      reason = "lineage_proof_unknown";
      return false;
    }
    domains.push_back({*low, *high, *step});
  }
  return true;
}

static bool checkedMultiply4(int64_t &value, int64_t factor,
                             StringRef boundReason, std::string &reason) {
  if ((factor != 2 && factor != 4) || value < 1 || value > 4 / factor) {
    reason = boundReason.str();
    return false;
  }
  value *= factor;
  return true;
}

static bool partitionMatches(const PartitionStep &step) {
  if ((step.family != "tiling" && step.family != "replica") ||
      (step.factor != 2 && step.factor != 4) || step.part < 0 ||
      step.part >= step.factor || step.axis < 0 ||
      static_cast<size_t>(step.axis) >= step.before.size() ||
      step.after.size() != step.before.size())
    return false;
  for (auto [axis, source] : llvm::enumerate(step.before)) {
    if (source.lower < 0 || source.upper <= source.lower || source.step != 1)
      return false;
    if (static_cast<int64_t>(axis) != step.axis) {
      if (!(source == step.after[axis]))
        return false;
      continue;
    }
    int64_t extent = source.upper - source.lower;
    if (extent < step.factor)
      return false;
    int64_t quotient = extent / step.factor;
    int64_t remainder = extent % step.factor;
    // Both products are bounded by extent; the resulting range is bounded by
    // the already authenticated positive int64 source interval.
    int64_t low = source.lower + quotient * step.part +
                  std::min(step.part, remainder);
    int64_t high = low + quotient + (step.part < remainder ? 1 : 0);
    if (!(step.after[axis] == CounterDomain{low, high, 1}))
      return false;
  }
  return true;
}

static DenseI64ArrayAttr encodeDomains(MLIRContext *context,
                                      ArrayRef<CounterDomain> domains) {
  SmallVector<int64_t> values;
  for (const CounterDomain &domain : domains)
    values.append({domain.lower, domain.upper, domain.step});
  return DenseI64ArrayAttr::get(context, values);
}

static bool decodeDomains(Attribute attribute, CounterDomains &domains) {
  auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(attribute);
  if (!dense || dense.size() == 0 || dense.size() % 3)
    return false;
  domains.clear();
  ArrayRef<int64_t> values = dense.asArrayRef();
  for (size_t index = 0; index < values.size(); index += 3) {
    if (values[index] < 0 || values[index + 1] <= values[index] ||
        values[index + 2] != 1)
      return false;
    domains.push_back({values[index], values[index + 1], values[index + 2]});
  }
  return true;
}

static ArrayAttr encodeLineage(MLIRContext *context,
                               ArrayRef<RootLineage> lineage) {
  Builder builder(context);
  SmallVector<Attribute> roots;
  for (const RootLineage &root : lineage) {
    SmallVector<Attribute> steps;
    for (const PartitionStep &step : root.steps) {
      NamedAttrList fields;
      fields.set("family", builder.getStringAttr(step.family));
      fields.set("axis", builder.getI64IntegerAttr(step.axis));
      fields.set("factor", builder.getI64IntegerAttr(step.factor));
      fields.set("part", builder.getI64IntegerAttr(step.part));
      fields.set("before", encodeDomains(context, step.before));
      fields.set("after", encodeDomains(context, step.after));
      steps.push_back(fields.getDictionary(context));
    }
    NamedAttrList fields;
    fields.set("root", builder.getStringAttr(root.root));
    fields.set("original", encodeDomains(context, root.original));
    fields.set("steps", builder.getArrayAttr(steps));
    roots.push_back(fields.getDictionary(context));
  }
  return builder.getArrayAttr(roots);
}

static bool decodeLineage(Attribute attribute, PartitionLineage &lineage) {
  auto roots = dyn_cast_or_null<ArrayAttr>(attribute);
  if (!roots || roots.empty())
    return false;
  lineage.clear();
  std::set<std::string> seenRoots;
  for (Attribute entry : roots) {
    auto fields = dyn_cast<DictionaryAttr>(entry);
    if (!fields)
      return false;
    auto rootName = fields.getAs<StringAttr>("root");
    auto steps = fields.getAs<ArrayAttr>("steps");
    RootLineage root;
    if (!rootName || rootName.getValue().empty() || !steps ||
        !seenRoots.insert(rootName.getValue().str()).second ||
        !decodeDomains(fields.get("original"), root.original))
      return false;
    root.root = rootName.getValue().str();
    for (Attribute stepEntry : steps) {
      auto stepFields = dyn_cast<DictionaryAttr>(stepEntry);
      if (!stepFields)
        return false;
      auto family = stepFields.getAs<StringAttr>("family");
      auto axis = stepFields.getAs<IntegerAttr>("axis");
      auto factor = stepFields.getAs<IntegerAttr>("factor");
      auto part = stepFields.getAs<IntegerAttr>("part");
      PartitionStep step;
      if (!family || !axis || !factor || !part ||
          !axis.getType().isInteger(64) || !factor.getType().isInteger(64) ||
          !part.getType().isInteger(64) ||
          !decodeDomains(stepFields.get("before"), step.before) ||
          !decodeDomains(stepFields.get("after"), step.after))
        return false;
      step.family = family.getValue().str();
      step.axis = axis.getInt();
      step.factor = factor.getInt();
      step.part = part.getInt();
      root.steps.push_back(std::move(step));
    }
    lineage.push_back(std::move(root));
  }
  return true;
}

static bool authenticateLineage(ModuleOp canonical, StringRef functionName,
                                 TaskflowTaskOp task,
                                 PartitionLineage &lineage,
                                 std::string &reason) {
  CounterDomains actual;
  if (!collectTypedDomains(task, actual, reason))
    return false;
  if (Attribute ledger = task->getAttr(kPartitionLineage)) {
    if (!decodeLineage(ledger, lineage)) {
      reason = "lineage_proof_unknown";
      return false;
    }
  } else {
    // Legacy rewritten candidates without a complete chain remain usable as
    // archive seeds. Further partition edits are unsupported; their most
    // recent metadata does not prove cumulative factor history.
    if (task->hasAttr("amoeba.neura.tiling.parent_task") ||
        task->hasAttr("amoeba.replica.parent_task") ||
        task->hasAttr("amoeba.tiling.parent_task") ||
        task->hasAttr("amoeba.semantic.k_tiled")) {
      reason = "lineage_proof_unknown";
      return false;
    }
    TaskflowTaskOp original =
        findTask(canonical, functionName, task.getTaskName());
    CounterDomains originalDomains;
    if (!original || !collectTypedDomains(original, originalDomains, reason) ||
        originalDomains != actual) {
      reason = "lineage_proof_unknown";
      return false;
    }
    lineage = {{task.getTaskName().str(), originalDomains, {}}};
  }
  for (const RootLineage &root : lineage) {
    CounterDomains originalDomains;
    TaskflowTaskOp original = findTask(canonical, functionName, root.root);
    if (!original || !collectTypedDomains(original, originalDomains, reason) ||
        originalDomains != root.original) {
      reason = "lineage_proof_unknown";
      return false;
    }
    CounterDomains current = root.original;
    int64_t tiling = 1;
    int64_t replica = 1;
    for (const PartitionStep &step : root.steps) {
      if (step.before != current || !partitionMatches(step)) {
        reason = "lineage_proof_unknown";
        return false;
      }
      int64_t &product = step.family == "tiling" ? tiling : replica;
      if (!checkedMultiply4(product, step.factor,
                            step.family == "tiling"
                                ? "cumulative_original_tiling_factor_exceeds_four"
                                : "cumulative_original_replica_factor_exceeds_four",
                            reason))
        return false;
      current = step.after;
    }
    if (current != actual) {
      reason = "lineage_proof_unknown";
      return false;
    }
  }
  return true;
}

static std::string partitionFamily(StringRef kind) {
  std::string value = lower(kind);
  if (value == "tile" || value == "tiling" || value == "k-tile" ||
      value == "k_tiling" || value == "ktile")
    return "tiling";
  if (value == "replica" || value == "replication")
    return "replica";
  return {};
}

static bool isFusion(StringRef kind) {
  std::string value = lower(kind);
  return value == "fusion" || value == "producer-consumer" ||
         value == "producer_consumer" || value == "sibling-fusion" ||
         value == "sibling";
}

static bool checkCumulativeLineageFactor(
    ModuleOp module, ModuleOp canonical, StringRef functionName,
    const NeighborhoodPrimitive &primitive, PartitionLineage &lineage,
    std::string &reason) {
  std::string family = partitionFamily(primitive.kind);
  if (family.empty())
    return true;
  TaskflowTaskOp task = findTask(module, functionName, primitive.firstTask);
  if (!task) {
    reason = "action_target_task_missing";
    return false;
  }
  if (!authenticateLineage(canonical, functionName, task, lineage, reason))
    return false;
  for (const RootLineage &root : lineage) {
    int64_t product = 1;
    for (const PartitionStep &step : root.steps)
      if (step.family == family &&
          !checkedMultiply4(product, step.factor,
                            "lineage_proof_unknown", reason))
        return false;
    if (!checkedMultiply4(product, primitive.factor,
                          family == "tiling"
                              ? "cumulative_original_tiling_factor_exceeds_four"
                              : "cumulative_original_replica_factor_exceeds_four",
                          reason))
      return false;
  }
  return true;
}

static bool recordPartitionLineage(
    ModuleOp module, StringRef functionName,
    const NeighborhoodPrimitive &primitive, ArrayRef<RootLineage> sourceLineage,
    ArrayRef<std::string> oldNames, std::string &reason) {
  std::string family = partitionFamily(primitive.kind);
  if (family.empty())
    return true;
  if (sourceLineage.empty()) {
    reason = "lineage_proof_unknown";
    return false;
  }
  CounterDomains before = sourceLineage.front().steps.empty()
                              ? sourceLineage.front().original
                              : sourceLineage.front().steps.back().after;
  for (const RootLineage &root : sourceLineage) {
    CounterDomains rootCurrent = root.steps.empty() ? root.original
                                                  : root.steps.back().after;
    if (rootCurrent != before) {
      reason = "lineage_proof_unknown";
      return false;
    }
  }
  std::set<int64_t> seenParts;
  for (const std::string &name : taskNames(module, functionName)) {
    if (llvm::is_contained(oldNames, name))
      continue;
    TaskflowTaskOp task = findTask(module, functionName, name);
    StringRef kind = primitive.kind;
    bool kTile = kind == "k-tile" || kind == "k_tiling" || kind == "ktile";
    StringRef parentAttribute = family == "replica"
                                   ? "amoeba.replica.parent_task"
                                   : "amoeba.neura.tiling.parent_task";
    auto parent = task->getAttrOfType<StringAttr>(parentAttribute);
    if (!kTile && (!parent || parent.getValue() != primitive.firstTask))
      continue;
    StringRef partAttribute = family == "replica"
                                 ? "amoeba.replica.id"
                                 : (kTile ? "amoeba.semantic.k_block_index"
                                          : "amoeba.neura.tiling.part_index");
    StringRef factorAttribute = family == "replica"
                                   ? "amoeba.replica.count"
                                   : (kTile ? "amoeba.semantic.k_block_count"
                                            : "amoeba.neura.tiling.factor");
    auto part = task->getAttrOfType<IntegerAttr>(partAttribute);
    auto factor = task->getAttrOfType<IntegerAttr>(factorAttribute);
    if (!part || !factor || factor.getInt() != primitive.factor)
      continue;
    CounterDomains after;
    if (!collectTypedDomains(task, after, reason))
      return false;
    PartitionStep step{family, primitive.axis, primitive.factor,
                       part.getInt(), before, after};
    if (!partitionMatches(step) || !seenParts.insert(step.part).second) {
      reason = "lineage_proof_unknown";
      return false;
    }
    // Authenticate source metadata against the actual selected interval.
    if (!kTile) {
      StringRef axisAttribute = family == "replica"
                                    ? "amoeba.replica.shard_axis"
                                    : "amoeba.neura.tiling.axis";
      auto axis = task->getAttrOfType<IntegerAttr>(axisAttribute);
      if (!axis || axis.getInt() != primitive.axis) {
        reason = "lineage_proof_unknown";
        return false;
      }
    }
    if (family == "tiling") {
      auto matchesRange = [&](StringRef attribute, const CounterDomain &domain) {
        auto range = task->getAttrOfType<DenseI64ArrayAttr>(attribute);
        return range && range.size() == 2 &&
               range.asArrayRef()[0] == domain.lower &&
               range.asArrayRef()[1] == domain.upper;
      };
      if ((kTile && !matchesRange("amoeba.semantic.K_range", after[step.axis])) ||
          (!kTile &&
           (!matchesRange("amoeba.neura.tiling.original_range", before[step.axis]) ||
            !matchesRange("amoeba.neura.tiling.derived_range", after[step.axis])))) {
        reason = "lineage_proof_unknown";
        return false;
      }
    }
    if (family == "replica") {
      auto low = task->getAttrOfType<IntegerAttr>("amoeba.replica.shard_lower");
      auto high = task->getAttrOfType<IntegerAttr>("amoeba.replica.shard_upper");
      if (!low || !high || low.getInt() != after[step.axis].lower ||
          high.getInt() != after[step.axis].upper) {
        reason = "lineage_proof_unknown";
        return false;
      }
    }
    PartitionLineage derived(sourceLineage.begin(), sourceLineage.end());
    for (RootLineage &root : derived)
      root.steps.push_back(step);
    task->setAttr(kPartitionLineage, encodeLineage(task.getContext(), derived));
  }
  if (seenParts.size() != static_cast<size_t>(primitive.factor)) {
    reason = "lineage_proof_unknown";
    return false;
  }
  return true;
}

static bool recordFusionLineage(ModuleOp module, ModuleOp canonical,
                                 StringRef functionName,
                                 const NeighborhoodPrimitive &primitive,
                                 ArrayRef<RootLineage> merged,
                                 std::string &reason) {
  TaskflowTaskOp fused = findTask(module, functionName,
      fusedTaskName(primitive.firstTask, primitive.secondTask));
  if (!fused)
    return true; // Unsupported fusion naming cannot grant a factor proof.
  fused->removeAttr(kPartitionLineage);
  if (merged.empty())
    return true;
  PartitionLineage unique;
  for (const RootLineage &root : merged) {
    auto existing = llvm::find_if(unique, [&](const RootLineage &other) {
      return other.root == root.root;
    });
    if (existing != unique.end()) {
      // Two paths with distinct partitions of the same original root cannot
      // be represented by this rectangular ledger. Preserve the candidate,
      // but subsequent partition actions remain unsupported.
      if (encodeLineage(module.getContext(), {*existing}) !=
          encodeLineage(module.getContext(), {root}))
        return true;
    } else {
      unique.push_back(root);
    }
  }
  fused->setAttr(kPartitionLineage, encodeLineage(module.getContext(), unique));
  PartitionLineage authenticated;
  std::string ignored;
  if (!authenticateLineage(canonical, functionName, fused, authenticated, ignored))
    fused->removeAttr(kPartitionLineage);
  return true;
}
// The tiler deliberately emits full completion joins. This narrowly scoped
// composite connects matching shards only after all join uses and domains
// have been authenticated, then asks the existing fusion proof to validate
// every affected kernel pair. It never guesses input-coordinate locality.
static bool repairCoTiledDependency(ModuleOp module, StringRef functionName,
                                    const NeighborhoodAction &action,
                                    std::string &reason,
                                    std::string &diagnostic) {
  if (action.primitives.size() < 3)
    return false;
  const NeighborhoodPrimitive &producerEdit = action.primitives[0];
  const NeighborhoodPrimitive &consumerEdit = action.primitives[1];
  if (lower(producerEdit.kind) != "tile" ||
      lower(consumerEdit.kind) != "tile" ||
      producerEdit.axis != consumerEdit.axis ||
      producerEdit.factor != consumerEdit.factor) {
    reason = "co_tiling_dependency_proof_unknown";
    return false;
  }
  std::map<int64_t, TaskflowTaskOp> producers, consumers;
  for (const std::string &name : taskNames(module, functionName)) {
    TaskflowTaskOp task = findTask(module, functionName, name);
    auto parent = task->getAttrOfType<StringAttr>("amoeba.neura.tiling.parent_task");
    auto part = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.part_index");
    auto axis = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.axis");
    auto factor = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.factor");
    if (!parent || !part || !axis || !factor ||
        axis.getInt() != producerEdit.axis ||
        factor.getInt() != producerEdit.factor)
      continue;
    if (parent.getValue() == producerEdit.firstTask)
      producers.emplace(part.getInt(), task);
    if (parent.getValue() == consumerEdit.firstTask)
      consumers.emplace(part.getInt(), task);
  }
  if (producers.size() != static_cast<size_t>(producerEdit.factor) ||
      consumers.size() != producers.size()) {
    reason = "co_tiling_dependency_proof_unknown";
    return false;
  }
  TaskflowJoinOp producerJoin;
  SmallVector<std::pair<TaskflowTaskOp, unsigned>> replacements;
  for (int64_t part = 0; part < producerEdit.factor; ++part) {
    auto p = producers.find(part), c = consumers.find(part);
    if (p == producers.end() || c == consumers.end() ||
        p->second.getDoneWrites().size() != 1 ||
        p->second.getOriginalWriteMemrefs().size() != 1) {
      reason = "co_tiling_dependency_proof_unknown";
      return false;
    }
    CounterDomains pDomains, cDomains;
    if (!collectTypedDomains(p->second, pDomains, reason) ||
        !collectTypedDomains(c->second, cDomains, reason) ||
        pDomains != cDomains) {
      reason = "co_tiling_dependency_proof_unknown";
      return false;
    }
    unsigned matches = 0;
    for (auto [index, input] : llvm::enumerate(c->second.getWillReads())) {
      auto join = input.getDefiningOp<TaskflowJoinOp>();
      if (!join ||
          !llvm::is_contained(join.getTileStates(), p->second.getDoneWrites()[0]))
        continue;
      if ((producerJoin && producerJoin != join) ||
          join.getAxis() != static_cast<uint64_t>(producerEdit.axis) ||
          join.getTileStates().size() != producers.size() ||
          input.getType() != p->second.getDoneWrites()[0].getType() ||
          index >= c->second.getOriginalReadMemrefs().size() ||
          c->second.getOriginalReadMemrefs()[index] !=
              p->second.getOriginalWriteMemrefs()[0]) {
        reason = "co_tiling_dependency_proof_unknown";
        return false;
      }
      producerJoin = join;
      replacements.push_back({c->second, static_cast<unsigned>(index)});
      ++matches;
    }
    if (matches != 1) {
      reason = "co_tiling_dependency_proof_unknown";
      return false;
    }
  }
  if (!producerJoin ||
      producerJoin.getJoined().getUses().empty()) {
    reason = "co_tiling_dependency_proof_unknown";
    return false;
  }
  // No caller, channel, extra task, or value-input use may escape the proof.
  if (static_cast<size_t>(std::distance(producerJoin.getJoined().use_begin(),
                                        producerJoin.getJoined().use_end())) !=
      replacements.size()) {
    reason = "co_tiling_dependency_has_external_use";
    return false;
  }
  for (OpOperand &use : producerJoin.getJoined().getUses()) {
    auto consumer = dyn_cast<TaskflowTaskOp>(use.getOwner());
    if (!consumer || !llvm::any_of(replacements, [&](const auto &replacement) {
          return replacement.first == consumer &&
                 use.getOperandNumber() ==
                     consumer.getWillReads().getBeginOperandIndex() +
                         replacement.second;
        })) {
      reason = "co_tiling_dependency_has_external_use";
      return false;
    }
  }
  for (int64_t part = 0; part < producerEdit.factor; ++part) {
    TaskflowTaskOp consumer = consumers.at(part);
    auto replacement = llvm::find_if(replacements, [&](const auto &entry) {
      return entry.first == consumer;
    });
    consumer.getWillReadsMutable().slice(replacement->second, 1)
        .assign(producers.at(part).getDoneWrites()[0]);
  }
  producerJoin.erase();

  // Rebind complete semantic annotations to actual typed edges. Roles remain
  // source annotations, while endpoints, scope and kind come from the graph.
  std::string error;
  auto function = selectTaskFunction(module, functionName, error);
  TaskEdgeGraphOptions options;
  auto graph = succeeded(function) ? buildTaskEdgeGraph(*function, options, error)
                                   : FailureOr<TaskEdgeGraph>(failure());
  if (failed(graph)) {
    reason = "co_tiling_dependency_proof_unknown";
    diagnostic = error;
    return false;
  }
  for (const auto &[part, consumer] : consumers) {
    auto old = consumer->getAttrOfType<ArrayAttr>("amoeba.semantic.incoming_edges");
    if (!old)
      continue;
    std::map<std::string, std::string> roles;
    for (Attribute attribute : old) {
      auto text = dyn_cast<StringAttr>(attribute);
      SmallVector<StringRef> fields;
      if (!text) {
        reason = "co_tiling_semantic_annotation_unknown";
        return false;
      }
      text.getValue().split(fields, '|');
      if (fields.size() < 3 || fields.size() > 4 || fields[1].empty()) {
        reason = "co_tiling_semantic_annotation_unknown";
        return false;
      }
      roles.emplace(fields[0].str(), fields[1].str());
    }
    SmallVector<Attribute> rebuilt;
    std::set<std::pair<std::string, std::string>> seen;
    for (const TaskEdge &edge : graph->getEdges()) {
      if (edge.consumer != consumer || edge.origin != TaskEdgeOrigin::Taskflow)
        continue;
      TaskflowTaskOp edgeProducer = edge.producer;
      std::string producer = edgeProducer.getTaskName().str();
      auto role = roles.find(producer);
      std::string kind = stringifyTaskEdgeKind(edge.kind).str();
      if (role == roles.end() || !seen.insert({producer, kind}).second) {
        reason = "co_tiling_semantic_annotation_unknown";
        return false;
      }
      rebuilt.push_back(StringAttr::get(module.getContext(),
          producer + "|" + role->second + "|" +
          stringifyTaskEdgeScope(edge.scope).str() + "|" + kind));
    }
    consumer->setAttr("amoeba.semantic.incoming_edges",
                      ArrayAttr::get(module.getContext(), rebuilt));
  }
  if (failed(verify(module.getOperation()))) {
    reason = "co_tiling_dependency_verifier_failed";
    return false;
  }
  // Existing body/address/store-load legality is the final authority. All
  // repaired pairs must admit fusion, even though this local action commits
  // only its requested pair. Other preflight modules are discarded.
  for (int64_t part = 0; part < producerEdit.factor; ++part) {
    OwningOpRef<ModuleOp> proof = module.clone();
    NeighborhoodPrimitive fusion = fusionPrimitive(
        producers.at(part).getTaskName(), consumers.at(part).getTaskName(),
        action.primitives[2].mode);
    std::string proofReason, proofDiagnostic;
    if (!runMaterializer(*proof, functionName, fusion, proofReason,
                           proofDiagnostic)) {
      reason = "co_tiling_dependency_existing_fusion_proof_rejected";
      diagnostic = proofDiagnostic;
      return false;
    }
  }
  return true;
}

} // namespace

std::vector<NeighborhoodAction>
mlir::amoeba::neura::joint_scheduling::enumerateNeighborhoodActions(
    ModuleOp module, StringRef functionName, ArrayRef<NeighborhoodShape> shapes,
    StringRef stage, unsigned round) {
  std::vector<NeighborhoodAction> actions;
  SmallVector<std::string> names = taskNames(module, functionName);
  if (names.empty())
    return actions;
  const unsigned stageId = stageNumber(stage);
  const bool seedRound = round == 0;

  NeighborhoodAction identity;
  identity.family = "identity";
  identity.label = "identity";
  addUniqueAction(actions, std::move(identity));

  NeighborhoodAction reset;
  reset.family = "canonical-reset";
  reset.label = "canonical-reset";
  reset.canonicalReset = true;
  addUniqueAction(actions, std::move(reset));

  // A lineage withdrawal/replacement is intentionally represented as an
  // explicit path action.  The search controller can replay the parent's
  // canonical path while omitting this lineage; applyNeighborhoodAction
  // refuses to guess that edit when it has no path history of its own.
  for (const std::string &name : names) {
    NeighborhoodAction replacement;
    replacement.family = "lineage-replacement";
    replacement.label = "replace-lineage:" + name;
    replacement.canonicalReset = true;
    NeighborhoodPrimitive primitive;
    primitive.kind = "lineage-reset";
    primitive.firstTask = name;
    replacement.primitives.push_back(std::move(primitive));
    addUniqueAction(actions, std::move(replacement));
  }

  SmallVector<ShapeKey> shapeChoices;
  addUniqueShape(shapeChoices, 1, 1);
  addUniqueShape(shapeChoices, 1, 2);
  addUniqueShape(shapeChoices, 2, 1);
  if (!seedRound) {
    addUniqueShape(shapeChoices, 1, 3);
    addUniqueShape(shapeChoices, 3, 1);
    addUniqueShape(shapeChoices, 1, 4);
    addUniqueShape(shapeChoices, 2, 2);
    addUniqueShape(shapeChoices, 4, 1);
  }
  // Round zero is the deliberately narrow seed neighborhood.  Existing
  // larger decisions remain available through identity/history candidates,
  // and are reopened only in later rounds.
  if (!seedRound)
    for (const NeighborhoodShape &shape : shapes)
      addUniqueShape(shapeChoices, shape.rows, shape.cols);

  for (const std::string &name : names) {
    const NeighborhoodShape *current = findShape(shapes, name);
    for (const ShapeKey &choice : shapeChoices) {
      if (current && current->rows == choice.rows && current->cols == choice.cols)
        continue;
      NeighborhoodAction action;
      action.family = "shape";
      action.shapeTask = name;
      action.shapeRows = choice.rows;
      action.shapeCols = choice.cols;
      action.label = (Twine("shape:") + name + ":" +
                      Twine(choice.rows) + "x" + Twine(choice.cols))
                         .str();
      addUniqueAction(actions, std::move(action));
    }
  }

  if (stageId >= 3) {
    SmallVector<int64_t> replicaFactors;
    replicaFactors.push_back(2);
    if (!seedRound)
      replicaFactors.push_back(4);
    for (const std::string &name : names) {
      for (int64_t factor : replicaFactors) {
        for (int64_t axis = 0; axis <= 1; ++axis) {
          NeighborhoodAction action;
          action.family = "replica";
          action.primitives.push_back(replicaPrimitive(name, axis, factor));
          action.label = (Twine("replica:") + name + ":axis=" +
                          Twine(axis) + ":factor=" + Twine(factor))
                             .str();
          addUniqueAction(actions, std::move(action));
        }
      }
    }
  }

  if (stageId >= 4) {
    SmallVector<int64_t> tileFactors;
    tileFactors.push_back(2);
    if (!seedRound)
      tileFactors.push_back(4);
    for (const std::string &name : names) {
      for (int64_t factor : tileFactors) {
        for (int64_t axis = 0; axis <= 1; ++axis) {
          NeighborhoodAction action;
          action.family = "tiling";
          action.primitives.push_back(tilePrimitive(name, axis, factor));
          action.label = (Twine("tile:") + name + ":axis=" + Twine(axis) +
                          ":factor=" + Twine(factor))
                             .str();
          addUniqueAction(actions, std::move(action));
        }
        for (StringRef mode : {StringRef("sequential"),
                               StringRef("parallel-linear"),
                               StringRef("parallel-tree")}) {
          NeighborhoodAction action;
          action.family = "k-tiling";
          action.primitives.push_back(kTilePrimitive(name, mode, factor));
          action.label = (Twine("k-tile:") + name + ":mode=" + mode +
                          ":factor=" + Twine(factor))
                             .str();
          addUniqueAction(actions, std::move(action));
        }
      }
    }

    // Coupled edits follow actual SSA dependencies. Each action is finite;
    // unsupported axes/policies are reported by the source materializers.
    for (const DependencyPair &edge : dependencyPairs(module, functionName)) {
      for (int64_t axis = 0; axis <= 1; ++axis) {
        NeighborhoodAction coTile;
        coTile.family = "producer-consumer-co-tiling";
        coTile.primitives.push_back(tilePrimitive(edge.producer, axis, 2));
        coTile.primitives.push_back(tilePrimitive(edge.consumer, axis, 2));
        coTile.label = (Twine("co-tile:") + edge.producer + ":" +
                        edge.consumer + ":axis=" + Twine(axis)).str();
        addUniqueAction(actions, std::move(coTile));
      }
      for (StringRef mode : {StringRef("sequential"),
                             StringRef("parallel-linear"),
                             StringRef("parallel-tree")}) {
        NeighborhoodAction coTile;
        coTile.family = "producer-consumer-co-k-tiling";
        coTile.primitives.push_back(kTilePrimitive(edge.producer, mode, 2));
        coTile.primitives.push_back(kTilePrimitive(edge.consumer, mode, 2));
        coTile.label = (Twine("co-k-tile:") + edge.producer + ":" +
                        edge.consumer + ":mode=" + mode).str();
        addUniqueAction(actions, std::move(coTile));
      }
    }
  }

  if (stageId >= 5 && names.size() >= 2) {
    // Producer/consumer actions must follow an authenticated SSA edge.  Task
    // names are sorted for deterministic output, so using names[0]/names[1]
    // here would reverse the fixture's consumer/tensor pair.
    SmallVector<DependencyPair> edges = dependencyPairs(module, functionName);
    for (const DependencyPair &edge : edges) {
      const std::string &producer = edge.producer;
      const std::string &consumer = edge.consumer;
      std::string fused = fusedTaskName(producer, consumer);

      for (StringRef mode : {StringRef("producer-consumer-retained"),
                             StringRef("producer-consumer-forwarded")}) {
        NeighborhoodAction action;
        action.family = "fusion";
        action.primitives.push_back(fusionPrimitive(producer, consumer, mode));
        action.label = (Twine("fuse:") + producer + ":" + consumer + ":" +
                        mode)
                           .str();
        addUniqueAction(actions, std::move(action));
      }

      // Complete local path: fuse the authenticated edge, then tile the
      // actual source-owned fused task name.
      NeighborhoodAction fuseTile;
      fuseTile.family = "fusion-plus-tiling";
      fuseTile.primitives.push_back(
          fusionPrimitive(producer, consumer, "producer-consumer-forwarded"));
      fuseTile.primitives.push_back(tilePrimitive(fused, 0, 2));
      fuseTile.label = (Twine("fuse-then-tile:") + producer + ":" + consumer)
                           .str();
      addUniqueAction(actions, std::move(fuseTile));

      // Complete local path in the opposite order.  Both tasks are tiled and
      // the fusion targets corresponding derived shards, so this remains the
      // same producer/consumer edge rather than an unrelated pair or an
      // obsolete pre-tile task name.
      std::string producerShard = producer + ".tile.0.0";
      std::string consumerShard = consumer + ".tile.0.0";
      NeighborhoodAction tileFuse;
      tileFuse.family = "tiling-plus-fusion";
      tileFuse.primitives.push_back(tilePrimitive(producer, 0, 2));
      tileFuse.primitives.push_back(tilePrimitive(consumer, 0, 2));
      tileFuse.primitives.push_back(fusionPrimitive(
          producerShard, consumerShard, "producer-consumer-forwarded"));
      tileFuse.label = (Twine("tile-then-fuse:") + producer + ":" + consumer)
                           .str();
      addUniqueAction(actions, std::move(tileFuse));

      NeighborhoodAction coTile;
      coTile.family = "producer-consumer-co-tiling";
      coTile.primitives.push_back(tilePrimitive(producer, 0, 2));
      coTile.primitives.push_back(tilePrimitive(consumer, 0, 2));
      coTile.label = (Twine("co-tile:") + producer + ":" + consumer).str();
      addUniqueAction(actions, std::move(coTile));

      NeighborhoodAction fusionShape;
      fusionShape.family = "fusion-plus-shape";
      fusionShape.primitives.push_back(
          fusionPrimitive(producer, consumer, "producer-consumer-retained"));
      fusionShape.shapeTask = fused;
      fusionShape.shapeRows = 1;
      fusionShape.shapeCols = 2;
      fusionShape.label =
          (Twine("fuse-plus-shape:") + fused + ":1x2").str();
      addUniqueAction(actions, std::move(fusionShape));
    }

    // Sibling fusion is a separate legal source-owned mode.  Enumerate each
    // unordered pair once; the materializer remains the legality oracle.
    for (size_t i = 0; i < names.size(); ++i) {
      for (size_t j = i + 1; j < names.size(); ++j) {
        NeighborhoodAction sibling;
        sibling.family = "sibling-fusion";
        sibling.primitives.push_back(
            fusionPrimitive(names[i], names[j], "sibling"));
        sibling.label = (Twine("sibling-fuse:") + names[i] + ":" + names[j])
                            .str();
        addUniqueAction(actions, std::move(sibling));
      }
    }
  }

  return actions;
}

bool mlir::amoeba::neura::joint_scheduling::applyNeighborhoodAction(
    ModuleOp cloned, ModuleOp canonical, StringRef functionName,
    const NeighborhoodAction &action, std::vector<NeighborhoodShape> &shapes,
    std::string &reason, std::string &diagnostic) {
  reason.clear();
  diagnostic.clear();
  if (!cloned || !canonical) {
    reason = "missing_candidate_module";
    return false;
  }
  if (cloned.getOperation() == canonical.getOperation()) {
    reason = "candidate_and_canonical_must_be_distinct";
    return false;
  }
  if (action.family == "lineage-replacement") {
    reason = "lineage_replacement_requires_canonical_path_replay";
    diagnostic = action.shapeTask.empty() ? action.label : action.shapeTask;
    return false;
  }
  if (!actionHasDuplicateLineageFactor(action.primitives, reason))
    return false;

  // Work on a private operation clone.  A failed or unsupported action can
  // never leave half of a rewrite in the caller's predecessor.
  OwningOpRef<ModuleOp> trial = action.canonicalReset ? canonical.clone()
                                                       : cloned.clone();
  if (!trial) {
    reason = "candidate_clone_failed";
    return false;
  }
  std::vector<NeighborhoodShape> nextShapes;
  if (action.canonicalReset)
    initialShapes(*trial, functionName, nextShapes);
  else
    synchronizeShapes(*trial, functionName, shapes, nextShapes);

  for (auto [primitiveIndex, primitive] : llvm::enumerate(action.primitives)) {
    if (primitiveIndex == 2 && action.family == "tiling-plus-fusion" &&
        !repairCoTiledDependency(*trial, functionName, action, reason, diagnostic))
      return false;
    if (isShapePrimitive(primitive.kind)) {
      reason = "shape_primitive_requires_action_shape_fields";
      return false;
    }
    if (lower(primitive.kind) == "noop" || lower(primitive.kind) == "identity")
      continue;
    if (!hasTask(*trial, functionName, primitive.firstTask)) {
      reason = "action_target_task_missing";
      diagnostic = primitive.firstTask;
      return false;
    }
    PartitionLineage sourceLineage;
    if (!checkCumulativeLineageFactor(*trial, canonical, functionName,
                                      primitive, sourceLineage, reason))
      return false;
    PartitionLineage fusionLineage;
    if (isFusion(primitive.kind)) {
      PartitionLineage first, second;
      std::string firstReason, secondReason;
      if (authenticateLineage(canonical, functionName,
                              findTask(*trial, functionName, primitive.firstTask),
                              first, firstReason) &&
          authenticateLineage(canonical, functionName,
                              findTask(*trial, functionName, primitive.secondTask),
                              second, secondReason)) {
        fusionLineage.append(first.begin(), first.end());
        fusionLineage.append(second.begin(), second.end());
      }
    }
    SmallVector<std::string> namesBefore = taskNames(*trial, functionName);
    if ((lower(primitive.kind) == "fusion" ||
         lower(primitive.kind) == "producer-consumer" ||
         lower(primitive.kind) == "producer_consumer" ||
         lower(primitive.kind) == "sibling-fusion" ||
         lower(primitive.kind) == "sibling") &&
        !hasTask(*trial, functionName, primitive.secondTask)) {
      reason = "fusion_second_task_missing";
      diagnostic = primitive.secondTask;
      return false;
    }
    if (!runMaterializer(*trial, functionName, primitive, reason, diagnostic))
      return false;
    if (!recordPartitionLineage(*trial, functionName, primitive, sourceLineage,
                                 namesBefore, reason))
      return false;
    if (isFusion(primitive.kind) &&
        !recordFusionLineage(*trial, canonical, functionName, primitive,
                              fusionLineage, reason))
      return false;
    std::vector<NeighborhoodShape> refreshed;
    synchronizeShapes(*trial, functionName, nextShapes, refreshed);
    nextShapes = std::move(refreshed);
  }

  if (!action.shapeTask.empty()) {
    if (!isLegalArea4Shape(action.shapeRows, action.shapeCols)) {
      reason = "unsupported_oriented_shape";
      return false;
    }
    if (!hasTask(*trial, functionName, action.shapeTask)) {
      reason = "shape_target_task_missing_after_rewrite";
      diagnostic = action.shapeTask;
      return false;
    }
    bool replaced = false;
    for (NeighborhoodShape &shape : nextShapes) {
      if (shape.task == action.shapeTask) {
        shape.rows = action.shapeRows;
        shape.cols = action.shapeCols;
        replaced = true;
        break;
      }
    }
    if (!replaced)
      nextShapes.push_back(NeighborhoodShape{action.shapeTask, action.shapeRows,
                                             action.shapeCols});
  }

  if (failed(verify(trial->getOperation()))) {
    reason = "candidate_module_verifier_failed";
    return false;
  }
  if (!copyModuleBody(cloned, *trial, reason))
    return false;
  shapes = std::move(nextShapes);
  return true;
}
