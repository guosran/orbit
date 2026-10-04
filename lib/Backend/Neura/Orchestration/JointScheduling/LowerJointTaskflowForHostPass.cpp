//===- LowerJointTaskflowForHostPass.cpp ----------------------*- C++ -*-===//
//
// Lowers the executable subset of ORBIT Taskflow IR to ordinary host IR.
// Taskflow tasks are dependency-ordered and inlined, counters/hyperblocks are
// represented by nested scf.for operations, and proven completion joins are
// lowered to aliases of their base memref.  The pass is deliberately
// fail-closed: a body contract or an operation shape that cannot be lowered is
// diagnosed before any rewrite is made.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <cstring>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

constexpr StringLiteral kErrorPrefix = "[ORBIT_HOST_LOWERING] ";
constexpr StringLiteral kHostTaskNameAttr = "amoeba.host.task_name";
constexpr StringLiteral kHostTaskCompletionAttr = "amoeba.host.task_completion";
constexpr StringLiteral kHostTraceSymbolAttr = "amoeba.host.trace_symbol";
// This marker is emitted only by the control-flow producer after it proves a
// source entry boundary and an ordered hardware-counter packet.  Host replay
// must never infer once-only initialization from a folded zero or textual
// prefix in an already-flattened kernel.
constexpr StringLiteral kSourceOwnedOnceInitAttr =
    "amoeba.source_owned_once_init";

static LogicalResult reject(Operation *operation, const Twine &message) {
  operation->emitError() << kErrorPrefix << message;
  return failure();
}

static bool isTaskflowOperation(Operation *operation) {
  Dialect *dialect = operation->getDialect();
  return dialect && dialect->getNamespace() == "taskflow";
}

// The generated ORBIT bodies use arithmetic, memref access, and structured
// control flow.  Checking the dialect namespace keeps this whitelist useful as
// new arith/memref operations are introduced, while regions are still walked
// recursively below so an unrelated operation cannot hide inside a loop.
static bool isSupportedHostOperation(Operation *operation) {
  if (isa<scf::ForOp, scf::IfOp, scf::YieldOp>(operation))
    return true;
  Dialect *dialect = operation->getDialect();
  if (!dialect)
    return false;
  StringRef name = dialect->getNamespace();
  return name == "arith" || name == "memref";
}

static LogicalResult validateHostRegion(Region &region, Operation *owner,
                                        bool allowHyperblockYield = false) {
  // An scf.if without results may omit its else block. The MLIR verifier
  // checks that this is legal; an absent else executes no operations.
  if (auto conditional = dyn_cast<scf::IfOp>(owner))
    if (&region == &conditional.getElseRegion() && region.empty())
      return success();
  if (!region.hasOneBlock())
    return reject(owner, "HOST_LOWERING_UNSUPPORTED_SHAPE: expected one block");

  Block &block = region.front();
  for (Operation &operation : block) {
    if (allowHyperblockYield && isa<TaskflowHyperblockYieldOp>(&operation) &&
        &operation == block.getTerminator())
      continue;
    if (!isSupportedHostOperation(&operation))
      return reject(&operation, "HOST_LOWERING_UNSUPPORTED_OPERATION: " +
                                    operation.getName().getStringRef());
    for (Region &nested : operation.getRegions())
      if (failed(validateHostRegion(nested, &operation)))
        return failure();
  }
  return success();
}

static LogicalResult validateCounter(TaskflowCounterOp counter) {
  if (!counter.getCounterIndex().getType().isIndex() ||
      !counter.getLowerBound().getType().isIndex() ||
      !counter.getUpperBound().getType().isIndex() ||
      !counter.getStep().getType().isIndex())
    return reject(counter,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: counter bounds and "
                  "result must have index type");
  return success();
}

static LogicalResult validateCounter(neura::CounterOp counter) {
  if (!counter.getCurrentIndex().getType().isIndex() ||
      !counter.getLowerBound() || !counter.getUpperBound() ||
      !counter.getStep() || !counter.getLowerBound().getType().isIndex() ||
      !counter.getUpperBound().getType().isIndex() ||
      !counter.getStep().getType().isIndex())
    return reject(counter,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: neura.counter bounds "
                  "and result must have index type");
  return success();
}

// Post-Neura kernels use !neura.data<T, i1> values.  The dataflow lowering
// stores a payload and a validity predicate in each value; host replay keeps
// those two pieces explicit while it reconstructs ordinary SCF/memref IR.
struct HostDataValue {
  Value value;
  Value predicate;
};

using HostDataMap = llvm::DenseMap<Value, HostDataValue>;

static bool isPredicatedType(Type type) {
  return isa<neura::PredicatedValue>(type);
}

static Type payloadType(Type type) {
  if (auto predicated = dyn_cast<neura::PredicatedValue>(type))
    return predicated.getValueType();
  return type;
}

static Value createHostBool(OpBuilder &builder, Location loc, bool value) {
  return builder.create<arith::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(value));
}

static Value createHostInteger(OpBuilder &builder, Location loc, Type type,
                               const APInt &value) {
  return builder.create<arith::ConstantOp>(
      loc, type, builder.getIntegerAttr(type, value));
}

static FailureOr<HostDataValue>
lookupHostData(Value source, const HostDataMap &values, IRMapping &mapping,
               Operation *owner) {
  if (auto found = values.find(source); found != values.end())
    return found->second;
  Value mapped = mapping.lookupOrNull(source);
  if (!mapped)
    return reject(owner,
                  "HOST_LOWERING_BODY_CONTRACT: dataflow value was not "
                  "mapped into host IR");
  return HostDataValue{mapped, nullptr};
}

static FailureOr<HostDataValue>
resolveHostDataAttribute(Attribute attribute, Type expectedType,
                         ArrayRef<HostDataValue> inputs,
                         ArrayRef<HostDataValue> iterArgs,
                         Value truePredicate, OpBuilder &builder,
                         Location loc, Operation *owner) {
  if (!attribute)
    return reject(owner, "HOST_LOWERING_BODY_CONTRACT: missing dataflow attribute");
  Type rawType = payloadType(expectedType);
  if (auto text = dyn_cast<StringAttr>(attribute)) {
    StringRef value = text.getValue();
    auto resolveIndexed = [&](StringRef prefix,
                              ArrayRef<HostDataValue> source)
        -> FailureOr<HostDataValue> {
      if (!value.consume_front(prefix))
        return failure();
      unsigned index = 0;
      if (value.empty() || value.getAsInteger(10, index) || index >= source.size())
        return failure();
      return source[index];
    };
    if (auto resolved = resolveIndexed("%input", inputs); succeeded(resolved))
      return *resolved;
    if (auto resolved = resolveIndexed("%iter_arg_init", iterArgs);
        succeeded(resolved))
      return *resolved;
    return reject(owner,
                  "HOST_LOWERING_BODY_CONTRACT: unsupported dataflow value "
                  "reference " + value);
  }
  if (auto integer = dyn_cast<IntegerAttr>(attribute)) {
    if (!rawType.isIndex() && !isa<IntegerType>(rawType))
      return reject(owner,
                    "HOST_LOWERING_UNSUPPORTED_SHAPE: integer dataflow "
                    "constant has a non-integer payload type");
    Value constant = createHostInteger(builder, loc, rawType, integer.getValue());
    return HostDataValue{constant, truePredicate};
  }
  return reject(owner,
                "HOST_LOWERING_UNSUPPORTED_OPERATION: dataflow constant "
                "attribute is not an input reference or integer");
}

// Store/load destination attributes are references to a kernel input, but
// their memref type is carried by that referenced value rather than by the
// operation itself.  Resolve those references without inventing a type.
static FailureOr<HostDataValue>
resolveHostDataReference(Attribute attribute, ArrayRef<HostDataValue> inputs,
                         ArrayRef<HostDataValue> iterArgs,
                         Operation *owner) {
  auto text = dyn_cast<StringAttr>(attribute);
  if (!text)
    return reject(owner,
                  "HOST_LOWERING_UNSUPPORTED_OPERATION: dataflow memory "
                  "reference must be a string input reference");
  StringRef value = text.getValue();
  auto resolve = [&](StringRef prefix, ArrayRef<HostDataValue> source)
      -> FailureOr<HostDataValue> {
    StringRef suffix = value;
    if (!suffix.consume_front(prefix))
      return failure();
    unsigned index = 0;
    if (suffix.empty() || suffix.getAsInteger(10, index) ||
        index >= source.size())
      return failure();
    return source[index];
  };
  if (auto result = resolve("%input", inputs); succeeded(result))
    return *result;
  if (auto result = resolve("%iter_arg_init", iterArgs); succeeded(result))
    return *result;
  return reject(owner,
                "HOST_LOWERING_BODY_CONTRACT: unsupported dataflow memory "
                "reference " + value);
}

static FailureOr<Value> createHostZero(OpBuilder &builder, Location loc,
                                       Type type, Operation *owner) {
  if (!type.isIndex() && !isa<IntegerType, FloatType>(type))
    return reject(owner,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: host dataflow loop "
                  "state requires an integer or floating-point payload");
  return builder.create<arith::ConstantOp>(loc, type,
                                           builder.getZeroAttr(type))
      .getResult();
}

static Value andHostPredicates(OpBuilder &builder, Location loc,
                               Value lhs, Value rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return builder.create<arith::AndIOp>(loc, lhs, rhs);
}

static Value orHostPredicates(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs) {
  return builder.create<arith::OrIOp>(loc, lhs, rhs);
}

static FailureOr<arith::CmpIPredicate>
parseHostCmpPredicate(StringRef predicate) {
  if (predicate == "eq")
    return arith::CmpIPredicate::eq;
  if (predicate == "ne")
    return arith::CmpIPredicate::ne;
  if (predicate == "slt")
    return arith::CmpIPredicate::slt;
  if (predicate == "sle")
    return arith::CmpIPredicate::sle;
  if (predicate == "sgt")
    return arith::CmpIPredicate::sgt;
  if (predicate == "sge")
    return arith::CmpIPredicate::sge;
  if (predicate == "ult")
    return arith::CmpIPredicate::ult;
  if (predicate == "ule")
    return arith::CmpIPredicate::ule;
  if (predicate == "ugt")
    return arith::CmpIPredicate::ugt;
  if (predicate == "uge")
    return arith::CmpIPredicate::uge;
  return failure();
}

// Mapper register moves preserve the dataflow value and its validity. Match
// semantic recurrence patterns through these routing operations as well.
static Value unwrapHostDataMoves(Value value) {
  while (auto move = value.getDefiningOp<neura::DataMovOp>()) {
    if (move->getOperand(0).getType() != move->getResult(0).getType())
      break;
    value = move->getOperand(0);
  }
  return value;
}

static bool isKnownHostDataflowOperation(Operation *operation) {
  return isa<neura::ConstantOp, neura::CounterOp, neura::ReserveOp,
              neura::GrantOnceOp, neura::PhiStartOp, neura::PhiOp,
              neura::GrantPredicateOp, neura::ExtractPredicateOp,
              neura::NotOp, neura::ICmpOp, neura::SelOp, neura::AddOp,
              neura::SubOp, neura::MulOp,
              neura::DivOp, neura::CastOp, neura::LoadIndexedOp,
              neura::StoreIndexedOp, neura::CtrlMovOp, neura::DataMovOp,
              neura::ReturnValueOp, neura::YieldOp, arith::ConstantOp>(
      operation);
}

static bool isPredicatedKernel(neura::KernelOp kernel) {
  if (auto mode = kernel->getAttrOfType<StringAttr>("dataflow_mode"))
    if (mode.getValue() == "predicate")
      return true;
  if (kernel.getBody().empty())
    return false;
  for (BlockArgument argument : kernel.getBody().front().getArguments())
    if (isPredicatedType(argument.getType()))
      return true;
  return llvm::any_of(kernel.getResultTypes(), isPredicatedType);
}

static LogicalResult validateNeuraKernel(neura::KernelOp kernel) {
  if (!kernel.getBody().hasOneBlock())
    return reject(kernel,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: neura.kernel body must "
                  "have one block");
  Block &body = kernel.getBody().front();
  auto yield = dyn_cast<neura::YieldOp>(body.getTerminator());
  if (!yield)
    return reject(kernel,
                  "HOST_LOWERING_BODY_CONTRACT: neura.kernel must end in "
                  "neura.yield");
  if (body.getNumArguments() != kernel.getInputs().size() +
                                    kernel.getIterArgsInit().size())
    return reject(kernel,
                  "HOST_LOWERING_BODY_CONTRACT: neura.kernel block "
                  "arguments do not match inputs and iter_args_init");
  const bool predicated = isPredicatedKernel(kernel);
  if (yield.getIterArgsNext().size() != kernel.getIterArgsInit().size() ||
      yield.getResults().size() != kernel.getNumResults())
    if (!predicated)
      return reject(yield,
                    "HOST_LOWERING_BODY_CONTRACT: neura.kernel yield/result "
                    "count mismatch");
  if (predicated) {
    // transform-ctrl-to-data-flow moves kernel results into one or more
    // neura.return_value operations and leaves an empty neura.yield
    // terminator.  Requiring the old yield segment counts here rejects the
    // actual post-dataflow IR before the host replay lowering gets a chance
    // to consume it.
    if (!yield.getIterArgsNext().empty() || !yield.getResults().empty())
      return reject(yield,
                    "HOST_LOWERING_BODY_CONTRACT: predicated neura.kernel "
                    "must return values through neura.return_value");
    unsigned returnedValues = 0;
    for (Operation &operation : body.without_terminator())
      if (auto returned = dyn_cast<neura::ReturnValueOp>(&operation))
        returnedValues += returned.getValues().size();
    if (returnedValues != kernel.getNumResults())
      return reject(kernel,
                    "HOST_LOWERING_BODY_CONTRACT: predicated neura.kernel "
                    "return_value count does not match kernel results");
  }
  unsigned inputCount = kernel.getInputs().size();
  for (auto [index, argument] : llvm::enumerate(body.getArguments())) {
    Value expected = index < inputCount
                         ? kernel.getInputs()[index]
                         : kernel.getIterArgsInit()[index - inputCount];
    if ((!predicated && argument.getType() != expected.getType()) ||
        (predicated &&
         (!isPredicatedType(argument.getType()) ||
          payloadType(argument.getType()) != expected.getType())))
      return reject(kernel,
                    "HOST_LOWERING_BODY_CONTRACT: neura.kernel argument "
                    "type does not match its input");
  }
  if (predicated) {
    SmallVector<neura::CounterOp> counters;
    for (Operation &operation : body.without_terminator()) {
      if (auto counter = dyn_cast<neura::CounterOp>(&operation)) {
        if (!counter->getAttr("lower_bound_value") ||
            !counter->getAttr("upper_bound_value") ||
            !counter->getAttr("step_value"))
          return reject(counter,
                        "HOST_LOWERING_UNSUPPORTED_SHAPE: predicated "
                        "neura.counter requires bound attributes");
        counters.push_back(counter);
        continue;
      }
      if (isa<arith::ConstantOp>(&operation))
        continue;
      if (!isKnownHostDataflowOperation(&operation))
        return reject(&operation,
                      "HOST_LOWERING_UNSUPPORTED_OPERATION: unsupported "
                      "post-Neura dataflow operation " +
                          operation.getName().getStringRef());
    }
    if (counters.empty())
      return reject(kernel,
                    "HOST_LOWERING_UNSUPPORTED_SHAPE: predicated "
                    "neura.kernel requires at least one counter");
    return success();
  }
  for (auto [next, initial] :
       llvm::zip(yield.getIterArgsNext(), kernel.getIterArgsInit()))
    if (next.getType() != initial.getType())
      return reject(yield,
                    "HOST_LOWERING_BODY_CONTRACT: neura.kernel iter_arg "
                    "yield type does not match its initial value");
  for (auto [result, output] : llvm::zip(yield.getResults(), kernel.getResults()))
    if (result.getType() != output.getType())
      return reject(yield,
                    "HOST_LOWERING_BODY_CONTRACT: neura.kernel result type "
                    "does not match its output");

  SmallVector<neura::CounterOp> counters;
  for (Operation &operation : body.without_terminator()) {
    if (auto counter = dyn_cast<neura::CounterOp>(&operation)) {
      if (failed(validateCounter(counter)))
        return failure();
      counters.push_back(counter);
      continue;
    }
    // Counter bounds are internalized as arith constants by the
    // Taskflow-to-Neura conversion. They are cloned into the host function
    // before the loop nest and therefore do not execute as kernel operations.
    if (isa<arith::ConstantOp>(&operation))
      continue;
    for (Region &region : operation.getRegions())
      if (failed(validateHostRegion(region, &operation)))
        return failure();
    if (!isSupportedHostOperation(&operation))
      return reject(&operation, "HOST_LOWERING_UNSUPPORTED_OPERATION: " +
                                    operation.getName().getStringRef() +
                                    " (post-Neura dataflow kernels must be "
                                    "lowered before host replay)");
  }
  if (counters.empty())
    return reject(kernel,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: neura.kernel requires "
                  "at least one static counter");
  for (neura::CounterOp counter : counters) {
    if (!counter.getLowerBound() || !counter.getUpperBound() ||
        !counter.getStep())
      return reject(counter,
                    "HOST_LOWERING_UNSUPPORTED_SHAPE: neura.counter requires "
                    "explicit lower, upper, and step operands");
  }
  return success();
}

static LogicalResult validateHyperblock(TaskflowHyperblockOp hyperblock,
                                        ArrayRef<TaskflowCounterOp> counters) {
  if (hyperblock.getIndices().empty() || !hyperblock.getIterArgs().empty() ||
      !hyperblock.getOutputs().empty())
    return reject(hyperblock,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock requires "
                  "counter indices and no iter_args/results");
  if (!hyperblock.getBody().hasOneBlock())
    return reject(hyperblock,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock body must "
                  "have one block");
  Block &body = hyperblock.getBody().front();
  if (body.getNumArguments() != hyperblock.getIndices().size())
    return reject(hyperblock,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock index/body "
                  "argument count mismatch");
  if (!isa<TaskflowHyperblockYieldOp>(body.getTerminator()))
    return reject(hyperblock,
                  "HOST_LOWERING_BODY_CONTRACT: hyperblock must end in "
                  "taskflow.hyperblock.yield");
  auto yield = cast<TaskflowHyperblockYieldOp>(body.getTerminator());
  if (!yield.getIterArgsNext().empty() || !yield.getResults().empty())
    return reject(yield, "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock yield "
                         "cannot carry values");

  llvm::DenseSet<Value> usedCounters;
  for (auto [index, blockArgument] :
       llvm::zip(hyperblock.getIndices(), body.getArguments())) {
    auto counter = index.getDefiningOp<TaskflowCounterOp>();
    if (!counter || !llvm::is_contained(counters, counter))
      return reject(hyperblock,
                    "HOST_LOWERING_BODY_CONTRACT: hyperblock index is not a "
                    "taskflow.counter result");
    if (!index.getType().isIndex() || !blockArgument.getType().isIndex())
      return reject(hyperblock,
                    "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock indices "
                    "must have index type");
    usedCounters.insert(index);

    for (OpOperand &use : index.getUses()) {
      Operation *user = use.getOwner();
      if (user == hyperblock.getOperation())
        continue;
      if (auto child = dyn_cast<TaskflowCounterOp>(user)) {
        if (child.getParentIndex() == index &&
            llvm::is_contained(counters, child))
          continue;
      }
      if (user->getParentOfType<TaskflowHyperblockOp>() != hyperblock)
        return reject(user,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: counter escapes its "
                      "hyperblock");
    }
  }
  if (usedCounters.size() != counters.size())
    return reject(hyperblock,
                  "HOST_LOWERING_BODY_CONTRACT: every counter must drive the "
                  "hyperblock exactly once");
  for (auto [position, index] : llvm::enumerate(hyperblock.getIndices())) {
    auto counter = index.getDefiningOp<TaskflowCounterOp>();
    Value parent = counter.getParentIndex();
    if ((position == 0 && parent) ||
        (position != 0 && parent != hyperblock.getIndices()[position - 1]))
      return reject(counter, "HOST_LOWERING_UNSUPPORTED_SHAPE: counter "
                             "parent must follow hyperblock nesting order");
  }
  if (failed(validateHostRegion(hyperblock.getBody(), hyperblock,
                                /*allowHyperblockYield=*/true)))
    return failure();
  return success();
}

static LogicalResult validateTask(TaskflowTaskOp task) {
  if (!task.getBody().hasOneBlock())
    return reject(task, "HOST_LOWERING_BODY_CONTRACT: task body must have one "
                        "block");
  Block &body = task.getBody().front();
  SmallVector<Value> inputs;
  llvm::append_range(inputs, task.getWillReads());
  llvm::append_range(inputs, task.getWillWrites());
  llvm::append_range(inputs, task.getValueInputs());
  if (body.getNumArguments() != inputs.size())
    return reject(task,
                  "HOST_LOWERING_BODY_CONTRACT: body arguments must match "
                  "will_reads, will_writes, and value_inputs; "
                  "original_* operands are not body arguments");
  for (auto [argument, input] : llvm::zip(body.getArguments(), inputs))
    if (argument.getType() != input.getType())
      return reject(task,
                    "HOST_LOWERING_BODY_CONTRACT: body argument type does "
                    "not match its task input");

  auto yield = dyn_cast<TaskflowYieldOp>(body.getTerminator());
  if (!yield)
    return reject(task, "HOST_LOWERING_BODY_CONTRACT: task must end in "
                        "taskflow.yield");
  if (yield.getDoneReads().size() != task.getDoneReads().size() ||
      yield.getDoneWrites().size() != task.getDoneWrites().size() ||
      yield.getValueResults().size() != task.getValueOutputs().size())
    return reject(yield,
                  "HOST_LOWERING_BODY_CONTRACT: yield/result segment count "
                  "mismatch");
  auto checkTypes = [&](ValueRange values, ValueRange results) {
    for (auto [value, result] : llvm::zip(values, results))
      if (value.getType() != result.getType())
        return failure();
    return success();
  };
  if (failed(checkTypes(yield.getDoneReads(), task.getDoneReads())) ||
      failed(checkTypes(yield.getDoneWrites(), task.getDoneWrites())) ||
      failed(checkTypes(yield.getValueResults(), task.getValueOutputs())))
    return reject(yield,
                  "HOST_LOWERING_BODY_CONTRACT: yielded value type does not "
                  "match the task result");

  SmallVector<TaskflowCounterOp> counters;
  TaskflowHyperblockOp hyperblock;
  neura::KernelOp kernel;
  for (Operation &operation : body.without_terminator()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      if (failed(validateCounter(counter)))
        return failure();
      counters.push_back(counter);
      continue;
    }
    if (auto candidate = dyn_cast<TaskflowHyperblockOp>(&operation)) {
      if (hyperblock)
        return reject(candidate, "HOST_LOWERING_UNSUPPORTED_SHAPE: only one "
                                 "hyperblock per task is supported");
      hyperblock = candidate;
      continue;
    }
    if (auto candidate = dyn_cast<neura::KernelOp>(&operation)) {
      if (kernel)
        return reject(candidate, "HOST_LOWERING_UNSUPPORTED_SHAPE: only one "
                                 "neura.kernel per task is supported");
      kernel = candidate;
      continue;
    }
    if (isTaskflowOperation(&operation))
      return reject(&operation,
                    "HOST_LOWERING_UNSUPPORTED_OPERATION: unexpected "
                    "Taskflow operation in task body");
    if (isa<TaskflowCounterOp, TaskflowHyperblockOp>(&operation))
      continue;
    for (Region &region : operation.getRegions())
      if (failed(validateHostRegion(region, &operation)))
        return failure();
    if (!isSupportedHostOperation(&operation))
      return reject(&operation, "HOST_LOWERING_UNSUPPORTED_OPERATION: " +
                                    operation.getName().getStringRef());
  }
  if (!counters.empty() && !hyperblock && !kernel)
    return reject(task, "HOST_LOWERING_UNSUPPORTED_SHAPE: counters require a "
                        "hyperblock or neura.kernel");
  if (hyperblock && kernel)
    return reject(task,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: task cannot contain both "
                  "a hyperblock and neura.kernel");
  if (hyperblock && failed(validateHyperblock(hyperblock, counters)))
    return failure();
  if (kernel && failed(validateNeuraKernel(kernel)))
    return failure();
  return success();
}

static void collectTaskDependencies(
    Value value, const llvm::DenseMap<Operation *, unsigned> &taskIndices,
    llvm::DenseSet<unsigned> &dependencies) {
  if (auto task = value.getDefiningOp<TaskflowTaskOp>()) {
    auto found = taskIndices.find(task.getOperation());
    if (found != taskIndices.end())
      dependencies.insert(found->second);
    return;
  }
  if (auto channel = value.getDefiningOp<TaskflowChannelOp>()) {
    collectTaskDependencies(channel.getSource(), taskIndices, dependencies);
    return;
  }
  if (auto join = value.getDefiningOp<TaskflowReadCompletionJoinOp>()) {
    for (Value state : join.getTileStates())
      collectTaskDependencies(state, taskIndices, dependencies);
    return;
  }
  if (auto join = value.getDefiningOp<TaskflowJoinOp>()) {
    for (Value state : join.getTileStates())
      collectTaskDependencies(state, taskIndices, dependencies);
  }
}

static FailureOr<SmallVector<TaskflowTaskOp>>
dependencyOrder(func::FuncOp function) {
  SmallVector<TaskflowTaskOp> tasks;
  function.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });
  llvm::DenseMap<Operation *, unsigned> taskIndices;
  for (auto [index, task] : llvm::enumerate(tasks))
    taskIndices[task.getOperation()] = index;

  SmallVector<llvm::SmallDenseSet<unsigned>> dependencies(tasks.size());
  for (auto [index, task] : llvm::enumerate(tasks)) {
    llvm::DenseSet<unsigned> found;
    for (Value value : task.getWillReads())
      collectTaskDependencies(value, taskIndices, found);
    for (Value value : task.getWillWrites())
      collectTaskDependencies(value, taskIndices, found);
    for (Value value : task.getValueInputs())
      collectTaskDependencies(value, taskIndices, found);
    found.erase(index);
    dependencies[index].insert(found.begin(), found.end());
  }

  SmallVector<unsigned> state(tasks.size(), 0);
  SmallVector<TaskflowTaskOp> order;
  std::function<LogicalResult(unsigned)> visit = [&](unsigned index) {
    if (state[index] == 2)
      return success();
    if (state[index] == 1)
      return reject(tasks[index],
                    "HOST_LOWERING_DEPENDENCY_CYCLE: task dependencies are "
                    "not acyclic");
    state[index] = 1;
    for (unsigned dependency : dependencies[index])
      if (failed(visit(dependency)))
        return failure();
    state[index] = 2;
    order.push_back(tasks[index]);
    return success();
  };
  for (unsigned index = 0; index < tasks.size(); ++index)
    if (failed(visit(index)))
      return failure();
  return order;
}

static FailureOr<Value> mappedValue(IRMapping &mapping, Value value,
                                    Operation *owner) {
  Value mapped = mapping.lookupOrNull(value);
  if (!mapped)
    return reject(owner,
                  "HOST_LOWERING_BODY_CONTRACT: body value was not mapped "
                  "into the host function");
  return mapped;
}

static LogicalResult lowerHyperblock(TaskflowHyperblockOp hyperblock,
                                     IRMapping &mapping, OpBuilder &builder,
                                     Operation *&completionSite) {
  Block &sourceBody = hyperblock.getBody().front();
  SmallVector<TaskflowCounterOp> counters;
  for (Value index : hyperblock.getIndices()) {
    auto counter = index.getDefiningOp<TaskflowCounterOp>();
    if (!counter)
      return reject(hyperblock,
                    "HOST_LOWERING_BODY_CONTRACT: missing counter definition");
    counters.push_back(counter);
  }

  OpBuilder::InsertionGuard guard(builder);
  // The active insertion point is the parent task in the host function.
  // Creating loops inside the source task region would erase them with it.
  SmallVector<scf::ForOp> loops;
  for (TaskflowCounterOp counter : counters) {
    FailureOr<Value> lower =
        mappedValue(mapping, counter.getLowerBound(), counter.getOperation());
    FailureOr<Value> upper =
        mappedValue(mapping, counter.getUpperBound(), counter.getOperation());
    FailureOr<Value> step =
        mappedValue(mapping, counter.getStep(), counter.getOperation());
    if (failed(lower) || failed(upper) || failed(step))
      return failure();
    scf::ForOp loop =
        builder.create<scf::ForOp>(hyperblock.getLoc(), *lower, *upper, *step);
    loops.push_back(loop);
    builder.setInsertionPointToStart(loop.getBody());
    mapping.map(counter.getCounterIndex(), loop.getInductionVar());
  }
  if (loops.empty())
    return reject(hyperblock,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: hyperblock has no "
                  "counter loop");

  for (auto [argument, loop] : llvm::zip(sourceBody.getArguments(), loops))
    mapping.map(argument, loop.getInductionVar());

  builder.setInsertionPointToStart(loops.back().getBody());
  for (Operation &operation : sourceBody.without_terminator()) {
    for (Value operand : operation.getOperands())
      if (!mapping.lookupOrNull(operand))
        return reject(&operation,
                      "HOST_LOWERING_BODY_CONTRACT: hyperblock operand was "
                      "not mapped");
    builder.clone(operation, mapping);
  }
  // The body of every nested loop needs its own terminator.  The innermost
  // loop already contains the cloned hyperblock body and receives its yield
  // first; outer loops only contain the next inner loop and need an additional
  // yield after it.
  for (scf::ForOp loop : loops) {
    if (loop.getBody()->getTerminator())
      continue;
    OpBuilder::InsertionGuard yieldGuard(builder);
    builder.setInsertionPointToEnd(loop.getBody());
    builder.create<scf::YieldOp>(hyperblock.getLoc());
  }
  completionSite = loops.front().getOperation();
  hyperblock.erase();
  return success();
}

static FailureOr<Value> mappedValue(IRMapping &mapping, Value value,
                                    Operation *owner);

struct HostCounterBounds {
  HostDataValue lower;
  HostDataValue upper;
  HostDataValue step;
};

struct HostLoopState {
  Type type;
  int iterArg = -1;
  Value reserve;
  bool output = false;
};

// Lowers the post-Neura predicated form emitted by
// transform-ctrl-to-data-flow.  This is kept separate from the ordinary
// kernel lowering because the predicated form has no SSA operands on its
// neura.yield; its dataflow values are explicitly materialized as host
// payload/predicate pairs instead.
static LogicalResult lowerPredicatedNeuraKernel(
    neura::KernelOp kernel, IRMapping &mapping, OpBuilder &builder,
    Operation *&completionSite) {
  Block &sourceBody = kernel.getBody().front();
  SmallVector<neura::CounterOp> counters;
  SmallVector<neura::ReserveOp> reserves;
  for (Operation &operation : sourceBody.without_terminator()) {
    if (auto counter = dyn_cast<neura::CounterOp>(&operation))
      counters.push_back(counter);
    else if (auto reserve = dyn_cast<neura::ReserveOp>(&operation))
      reserves.push_back(reserve);
  }
  if (counters.empty())
    return reject(kernel,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: predicated neura.kernel "
                  "has no counter loop");

  OpBuilder::InsertionGuard guard(builder);
  Location loc = kernel.getLoc();
  Value truePredicate = createHostBool(builder, loc, true);

  SmallVector<HostDataValue> inputValues;
  for (Value input : kernel.getInputs()) {
    FailureOr<Value> mapped = mappedValue(mapping, input, kernel);
    if (failed(mapped))
      return failure();
    inputValues.push_back({*mapped, truePredicate});
  }
  SmallVector<HostDataValue> iterArgValues;
  for (Value initial : kernel.getIterArgsInit()) {
    FailureOr<Value> mapped = mappedValue(mapping, initial, kernel);
    if (failed(mapped))
      return failure();
    iterArgValues.push_back({*mapped, truePredicate});
  }

  // memref.dim is safe to use for a dynamic input, but materializing it in
  // every loop iteration lowers to a loop-local alloca in the LLVM runner.
  // Cache each dynamic extent once before constructing the SCF loops and use
  // the cached value in all subsequent bounds guards.
  DenseMap<Value, DenseMap<unsigned, Value>> dynamicExtents;
  auto cacheDynamicExtents = [&](ArrayRef<HostDataValue> data) {
    for (HostDataValue value : data) {
      auto memref = dyn_cast<MemRefType>(value.value.getType());
      if (!memref)
        continue;
      for (unsigned dimension = 0; dimension < memref.getRank(); ++dimension)
        if (memref.isDynamicDim(dimension))
          dynamicExtents[value.value][dimension] = builder.create<memref::DimOp>(
              loc, value.value, dimension);
    }
  };
  cacheDynamicExtents(inputValues);
  cacheDynamicExtents(iterArgValues);

  DenseMap<Value, HostCounterBounds> counterBounds;
  for (neura::CounterOp counter : counters) {
    auto resolveBound = [&](StringRef name) -> FailureOr<HostDataValue> {
      Attribute attr = counter->getAttr(name);
      if (!attr)
        return reject(counter,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: predicated counter "
                      "is missing " + name + " attribute");
      return resolveHostDataAttribute(attr, builder.getIndexType(),
                                      inputValues, iterArgValues,
                                      truePredicate, builder, loc, counter);
    };
    FailureOr<HostDataValue> lower = resolveBound("lower_bound_value");
    FailureOr<HostDataValue> upper = resolveBound("upper_bound_value");
    FailureOr<HostDataValue> step = resolveBound("step_value");
    if (failed(lower) || failed(upper) || failed(step))
      return failure();
    if (!lower->value.getType().isIndex() ||
        !upper->value.getType().isIndex() || !step->value.getType().isIndex())
      return reject(counter,
                    "HOST_LOWERING_UNSUPPORTED_SHAPE: predicated counter "
                    "bounds must have index payloads");
    counterBounds[counter.getResult()] = {*lower, *upper, *step};
  }

  // Some prepared reduction kernels have one or two explicit Taskflow
  // counters and carry an additional software-loop index in a Neura reserve.
  // The extra loop is materialized only after proving the complete
  // reserve/phi/compare/increment/control cycle.  In particular, an
  // unrelated add-one value is not a recurrence merely because it feeds a
  // ctrl_mov, and a repeated store target is not a seed without an explicit
  // initialization predicate.
  struct SequentialRecurrence {
    Value reserve;
    Value compareLhs;
    Value compareResult;
    Attribute bound;
    neura::ICmpOp compare;
    Operation *updateOperation;
    SmallVector<Value, 4> routePredicates;
    SmallVector<Value, 2> cycleReserves;
    SmallVector<Value, 4> seedPredicates;
    SmallVector<Value, 2> negativePredicates;
  };
  SmallVector<SequentialRecurrence> sequentialRecurrences;
  bool hasSequentialReduction = false;
  HostDataValue sequentialReductionUpper;
  SmallVector<Value> sequentialReductionReserves;
  Attribute sequentialReductionBoundAttr;
  if (counters.size() == 1 || counters.size() == 2) {
    auto isSupportedLoopBound = [](Attribute bound) {
      if (auto text = dyn_cast<StringAttr>(bound))
        return text.getValue().starts_with("%input");
      return isa<IntegerAttr>(bound);
    };

    // A software-loop seed must be compiler-owned zero data.  The only
    // accepted paths are constants/casts, grant_predicate value routing, and
    // the initial (non-reserve) operand of a two-input phi.  Keep the
    // predicates encountered on this path: a nested child is reset only when
    // its zero seed is routed from the parent's own guard.
    struct SeedEvidence {
      SmallVector<Value, 4> predicates;
    };
    llvm::SmallPtrSet<Value, 16> zeroSeedVisiting;
    std::function<bool(Value, SeedEvidence &)> proveZeroSeed;
    proveZeroSeed = [&](Value value, SeedEvidence &evidence) {
      value = unwrapHostDataMoves(value);
      if (!value || !zeroSeedVisiting.insert(value).second)
        return false;
      auto clearVisiting = llvm::make_scope_exit(
          [&] { zeroSeedVisiting.erase(value); });
      if (auto constant = value.getDefiningOp<neura::ConstantOp>()) {
        auto integer = dyn_cast<IntegerAttr>(constant.getValueAttr());
        return integer && integer.getValue().isZero();
      }
      if (auto cast = value.getDefiningOp<neura::CastOp>()) {
        // A compiler-owned zero may cross the index/int boundary used by a
        // reserve recurrence's seed.  Unlike route data, this conversion is
        // value-preserving for zero and remains explicit in the SSA chain.
        return proveZeroSeed(cast.getInput(), evidence);
      }
      if (auto grant = value.getDefiningOp<neura::GrantPredicateOp>()) {
        unsigned oldSize = evidence.predicates.size();
        evidence.predicates.push_back(
            unwrapHostDataMoves(grant.getPredicate()));
        if (proveZeroSeed(grant.getValue(), evidence))
          return true;
        evidence.predicates.resize(oldSize);
        return false;
      }
      if (auto phi = value.getDefiningOp<neura::PhiOp>())
        return phi->getNumOperands() == 2 &&
               proveZeroSeed(phi->getOperand(1), evidence);
      return false;
    };

    // Follow only compiler-owned register/control routing from the
    // increment back to the compare's index.  A Phi may be a reserve cycle
    // (reserve, routed value) only when a negative completion predicate is
    // already active; otherwise that branch can bypass the recurrence's own
    // guard.  A non-reserve merge must prove the own guard on every incoming
    // branch.  Positive application predicates are accepted only when the
    // merge proves an exhaustive complementary partition.
    struct RouteEvidence {
      SmallVector<Value, 4> predicates;
      SmallVector<Value, 2> cycleReserves;
      SmallVector<Value, 4> unpartitionedPositivePredicates;
      SmallVector<Value, 2> negativePredicates;
      unsigned negativeGuardCount = 0;
      bool allActivePathsGuarded = false;
      bool guardedReserveBranch = false;
    };
    auto isPotentialRecurrenceCompare = [&](Value value) {
      value = unwrapHostDataMoves(value);
      auto cmp = value.getDefiningOp<neura::ICmpOp>();
      if (!cmp || cmp.getCmpType() != "slt")
        return false;
      Attribute bound = cmp->getAttr("rhs_value");
      if (!bound || !isSupportedLoopBound(bound))
        return false;
      auto indexCast =
          unwrapHostDataMoves(cmp.getLhs()).getDefiningOp<neura::CastOp>();
      auto phi = indexCast
                     ? unwrapHostDataMoves(indexCast.getInput())
                           .getDefiningOp<neura::PhiOp>()
                     : nullptr;
      if (!indexCast || indexCast.getCastType() != "int_to_index" || !phi ||
          phi->getNumOperands() != 2 ||
          !unwrapHostDataMoves(phi->getOperand(0))
               .getDefiningOp<neura::ReserveOp>())
        return false;
      SeedEvidence seedEvidence;
      zeroSeedVisiting.clear();
      return proveZeroSeed(phi->getOperand(1), seedEvidence);
    };
    llvm::SmallPtrSet<Value, 32> routeVisiting;
    std::function<bool(Value, Value, Value, Value, RouteEvidence &)> proveRoute;
    proveRoute = [&](Value value, Value anchor, Value reserve, Value ownCompare,
                     RouteEvidence &evidence) -> bool {
      value = unwrapHostDataMoves(value);
      anchor = unwrapHostDataMoves(anchor);
      reserve = unwrapHostDataMoves(reserve);
      ownCompare = unwrapHostDataMoves(ownCompare);
      if (!value)
        return false;
      if (value == anchor) {
        evidence.allActivePathsGuarded = false;
        return true;
      }
      if (!routeVisiting.insert(value).second)
        return false;
      auto clearVisiting = llvm::make_scope_exit(
          [&] { routeVisiting.erase(value); });
      if (auto grant = value.getDefiningOp<neura::GrantPredicateOp>()) {
        Value predicate = unwrapHostDataMoves(grant.getPredicate());
        RouteEvidence oldEvidence = evidence;
        evidence.predicates.push_back(predicate);
        if (predicate == ownCompare) {
          if (proveRoute(grant.getValue(), anchor, reserve, ownCompare,
                         evidence)) {
            evidence.allActivePathsGuarded = true;
            return true;
          }
        } else {
          bool isNegativeRecurrenceGuard = false;
          if (predicate.getDefiningOp<neura::NotOp>())
            isNegativeRecurrenceGuard = isPotentialRecurrenceCompare(
                predicate.getDefiningOp<neura::NotOp>().getInput());
          if (isNegativeRecurrenceGuard) {
            ++evidence.negativeGuardCount;
            evidence.negativePredicates.push_back(predicate);
          }
          if (proveRoute(grant.getValue(), anchor, reserve, ownCompare,
                         evidence)) {
            if (!isNegativeRecurrenceGuard)
              evidence.unpartitionedPositivePredicates.push_back(predicate);
            return true;
          }
        }
        evidence = std::move(oldEvidence);
        return false;
      }
      if (auto cast = value.getDefiningOp<neura::CastOp>()) {
        if (payloadType(cast.getInput().getType()) !=
            payloadType(cast.getResult().getType()))
          return false;
        return proveRoute(cast.getInput(), anchor, reserve, ownCompare,
                          evidence);
      }
      if (auto phi = value.getDefiningOp<neura::PhiOp>()) {
        if (phi->getNumOperands() != 2)
          return false;
        Value first = unwrapHostDataMoves(phi->getOperand(0));
        Value second = unwrapHostDataMoves(phi->getOperand(1));
        RouteEvidence baseEvidence = evidence;
        RouteEvidence firstEvidence = baseEvidence;
        RouteEvidence secondEvidence = baseEvidence;
        if (first.getDefiningOp<neura::ReserveOp>()) {
          // The reserve edge is an active path too.  It may omit the own
          // compare only when a completion guard is already on this route;
          // relation validation later ties that guard to a proven child.
          if (evidence.negativeGuardCount == 0)
            return false;
          firstEvidence.cycleReserves.push_back(first);
          firstEvidence.guardedReserveBranch = true;
        } else if (!proveRoute(first, anchor, reserve, ownCompare,
                               firstEvidence)) {
          return false;
        }
        if (!proveRoute(second, anchor, reserve, ownCompare, secondEvidence))
          return false;
        bool firstGuarded = baseEvidence.allActivePathsGuarded ||
                            firstEvidence.allActivePathsGuarded ||
                            firstEvidence.guardedReserveBranch;
        bool secondGuarded = baseEvidence.allActivePathsGuarded ||
                             secondEvidence.allActivePathsGuarded ||
                             secondEvidence.guardedReserveBranch;
        if (!firstGuarded || !secondGuarded)
          return false;

        auto hasComplement = [](Value lhs, Value rhs) {
          lhs = unwrapHostDataMoves(lhs);
          rhs = unwrapHostDataMoves(rhs);
          if (auto notOp = lhs.getDefiningOp<neura::NotOp>())
            return unwrapHostDataMoves(notOp.getInput()) == rhs;
          if (auto notOp = rhs.getDefiningOp<neura::NotOp>())
            return unwrapHostDataMoves(notOp.getInput()) == lhs;
          return false;
        };
        SmallVector<Value, 4> firstPositive(
            firstEvidence.unpartitionedPositivePredicates.begin() +
                baseEvidence.unpartitionedPositivePredicates.size(),
            firstEvidence.unpartitionedPositivePredicates.end());
        SmallVector<Value, 4> secondPositive(
            secondEvidence.unpartitionedPositivePredicates.begin() +
                baseEvidence.unpartitionedPositivePredicates.size(),
            secondEvidence.unpartitionedPositivePredicates.end());
        // A pair of single-predicate branches is the only partition we can
        // discharge from a local complement check.  Matching each predicate
        // independently would incorrectly accept (p && q) || (!p && !q),
        // which leaves the mixed assignments uncovered.  A future extension
        // may prove a larger Boolean partition explicitly; until then keep
        // multi-guard routes fail-closed.
        if (firstPositive.size() > 1 || secondPositive.size() > 1)
          return false;
        bool complementaryPartition =
            firstPositive.empty() == secondPositive.empty() &&
            llvm::all_of(firstPositive, [&](Value predicate) {
              return llvm::any_of(secondPositive, [&](Value other) {
                return hasComplement(predicate, other);
              });
            }) &&
            llvm::all_of(secondPositive, [&](Value predicate) {
              return llvm::any_of(firstPositive, [&](Value other) {
                return hasComplement(predicate, other);
              });
            });
        if (!complementaryPartition) {
          return false;
        }

        evidence = baseEvidence;
        evidence.predicates.append(
            firstEvidence.predicates.begin() + baseEvidence.predicates.size(),
            firstEvidence.predicates.end());
        evidence.predicates.append(
            secondEvidence.predicates.begin() + baseEvidence.predicates.size(),
            secondEvidence.predicates.end());
        evidence.cycleReserves.append(
            firstEvidence.cycleReserves.begin() +
                baseEvidence.cycleReserves.size(),
            firstEvidence.cycleReserves.end());
        evidence.cycleReserves.append(
            secondEvidence.cycleReserves.begin() +
                baseEvidence.cycleReserves.size(),
            secondEvidence.cycleReserves.end());
        evidence.negativePredicates.append(
            firstEvidence.negativePredicates.begin() +
                baseEvidence.negativePredicates.size(),
            firstEvidence.negativePredicates.end());
        evidence.negativePredicates.append(
            secondEvidence.negativePredicates.begin() +
                baseEvidence.negativePredicates.size(),
            secondEvidence.negativePredicates.end());
        evidence.negativeGuardCount =
            baseEvidence.negativeGuardCount +
            (firstEvidence.negativeGuardCount -
             baseEvidence.negativeGuardCount) +
            (secondEvidence.negativeGuardCount -
             baseEvidence.negativeGuardCount);
        evidence.allActivePathsGuarded = firstGuarded && secondGuarded;
        evidence.guardedReserveBranch =
            baseEvidence.guardedReserveBranch ||
            firstEvidence.guardedReserveBranch ||
            secondEvidence.guardedReserveBranch;
        // The complementary branch pair discharges every positive guard
        // found below this merge.  Any positive guard added above the merge
        // remains in this vector and must be discharged by a later merge.
        evidence.unpartitionedPositivePredicates =
            baseEvidence.unpartitionedPositivePredicates;
        return true;
      }
      return false;
    };

    for (Operation &operation : sourceBody.without_terminator()) {
      auto cmp = dyn_cast<neura::ICmpOp>(&operation);
      if (!cmp || cmp.getCmpType() != "slt")
        continue;
      Attribute bound = cmp->getAttr("rhs_value");
      if (!bound || !isSupportedLoopBound(bound))
        continue;
      auto indexCast =
          unwrapHostDataMoves(cmp.getLhs()).getDefiningOp<neura::CastOp>();
      auto phi = indexCast
                     ? unwrapHostDataMoves(indexCast.getInput())
                           .getDefiningOp<neura::PhiOp>()
                     : nullptr;
      if (!indexCast || indexCast.getCastType() != "int_to_index" || !phi ||
          phi->getNumOperands() != 2)
        continue;
      Value reserve = unwrapHostDataMoves(phi->getOperand(0));
      if (!reserve.getDefiningOp<neura::ReserveOp>())
        continue;
      SeedEvidence seedEvidence;
      zeroSeedVisiting.clear();
      if (!proveZeroSeed(phi->getOperand(1), seedEvidence))
        continue;

      SmallVector<neura::CtrlMovOp> matchingUpdates;
      for (Operation &candidate : sourceBody.without_terminator()) {
        auto ctrl = dyn_cast<neura::CtrlMovOp>(&candidate);
        if (ctrl && unwrapHostDataMoves(ctrl.getTarget()) == reserve)
          matchingUpdates.push_back(ctrl);
      }
      if (matchingUpdates.empty())
        continue;
      if (matchingUpdates.size() != 1)
        return reject(kernel,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: recurrence reserve "
                      "has multiple control updates");
      neura::CtrlMovOp ctrl = matchingUpdates.front();
      auto cast =
          unwrapHostDataMoves(ctrl.getValue()).getDefiningOp<neura::CastOp>();
      auto add = cast
                     ? unwrapHostDataMoves(cast.getInput())
                           .getDefiningOp<neura::AddOp>()
                     : nullptr;
      auto increment = add
                           ? add->getAttrOfType<IntegerAttr>("rhs_value")
                           : IntegerAttr();
      if (!cast || cast.getCastType() != "index_to_int" || !add ||
          add->getNumOperands() != 1 || add->getAttr("lhs_value") ||
          !increment || !payloadType(add.getResult().getType()).isIndex())
        return reject(kernel,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: carried update must "
                      "be a single index-valued add");
      if (!increment.getValue().isOne())
        return reject(kernel,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: carried increment "
                      "must be a unit step");

      RouteEvidence routeEvidence;
      routeVisiting.clear();
      if (!proveRoute(add->getOperand(0), cmp.getLhs(), reserve,
                      cmp.getResult(), routeEvidence))
        return reject(kernel,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: carried increment "
                      "does not route through its compare predicate");
      if (!routeEvidence.allActivePathsGuarded ||
          !routeEvidence.unpartitionedPositivePredicates.empty())
        return reject(kernel,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: carried increment "
                      "has an unguarded or non-exhaustive route");
      sequentialRecurrences.push_back(
          {reserve, unwrapHostDataMoves(cmp.getLhs()), cmp.getResult(), bound,
           cmp, ctrl.getOperation(), std::move(routeEvidence.predicates),
           std::move(routeEvidence.cycleReserves),
           std::move(seedEvidence.predicates),
           std::move(routeEvidence.negativePredicates)});
    }

    if (!sequentialRecurrences.empty()) {
      // A nested completion guard need not target the child index reserve
      // directly.  The generated dataflow commonly carries the parent's
      // coordinate through a separate reserve while the child is active.
      // Prove that handoff by following the compiler-owned route of that
      // reserve's control update and requiring the child compare predicate.
      llvm::SmallPtrSet<Value, 32> predicateRouteVisiting;
      std::function<bool(Value, Value)> carriesPredicate;
      carriesPredicate = [&](Value value, Value wanted) {
        value = unwrapHostDataMoves(value);
        wanted = unwrapHostDataMoves(wanted);
        if (!value)
          return false;
        if (!predicateRouteVisiting.insert(value).second)
          return false;
        auto clearVisiting = llvm::make_scope_exit(
            [&] { predicateRouteVisiting.erase(value); });
        if (auto grant = value.getDefiningOp<neura::GrantPredicateOp>()) {
          if (unwrapHostDataMoves(grant.getPredicate()) == wanted)
            return true;
          return carriesPredicate(grant.getValue(), wanted);
        }
        if (auto cast = value.getDefiningOp<neura::CastOp>()) {
          if (payloadType(cast.getInput().getType()) !=
              payloadType(cast.getResult().getType()))
            return false;
          return carriesPredicate(cast.getInput(), wanted);
        }
        if (auto phi = value.getDefiningOp<neura::PhiOp>()) {
          if (phi->getNumOperands() != 2)
            return false;
          // Every incoming arm must carry the handoff predicate.  Accepting
          // one guarded arm is unsound when the other arm can update the
          // parent coordinate without proving the child boundary.
          return carriesPredicate(phi->getOperand(0), wanted) &&
                 carriesPredicate(phi->getOperand(1), wanted);
        }
        return false;
      };

      // Each route must contain the recurrence's positive guard.  A negative
      // route predicate is a parent/child boundary and must name another
      // proven recurrence, prove a guarded parent-coordinate handoff, and
      // reset that child through the parent's own compare guard.
      SmallVector<int, 4> parent(sequentialRecurrences.size(), -1);
      SmallVector<int, 4> child(sequentialRecurrences.size(), -1);
      for (unsigned index = 0; index < sequentialRecurrences.size(); ++index) {
        SequentialRecurrence &recurrence = sequentialRecurrences[index];
        bool ownGuard = false;
        SmallVector<int, 4> negativeChildren;
        for (Value predicate : recurrence.routePredicates) {
          predicate = unwrapHostDataMoves(predicate);
          if (predicate == recurrence.compareResult) {
            ownGuard = true;
            continue;
          }
          if (!llvm::is_contained(recurrence.negativePredicates, predicate))
            continue;
          auto notOp = predicate.getDefiningOp<neura::NotOp>();
          if (!notOp)
            return reject(kernel,
                          "HOST_LOWERING_UNSUPPORTED_SHAPE: positive route "
                          "guard was not discharged by an exhaustive merge");
          Value childCompare = unwrapHostDataMoves(notOp.getInput());
          int childIndex = -1;
          for (unsigned candidate = 0;
               candidate < sequentialRecurrences.size(); ++candidate)
            if (sequentialRecurrences[candidate].compareResult == childCompare)
              childIndex = static_cast<int>(candidate);
          if (childIndex < 0 || childIndex == static_cast<int>(index))
            return reject(kernel,
                          "HOST_LOWERING_UNSUPPORTED_SHAPE: recurrence route "
                          "has a negative guard without a proven child");
          if (recurrence.cycleReserves.empty())
            return reject(kernel,
                          "HOST_LOWERING_UNSUPPORTED_SHAPE: negative child "
                          "guard has no guarded parent-coordinate handoff");
          for (Value cycleReserve : recurrence.cycleReserves) {
            SmallVector<neura::CtrlMovOp, 2> cycleUpdates;
            for (Operation &candidate : sourceBody.without_terminator()) {
              auto ctrl = dyn_cast<neura::CtrlMovOp>(&candidate);
              if (ctrl && unwrapHostDataMoves(ctrl.getTarget()) == cycleReserve)
                cycleUpdates.push_back(ctrl);
            }
            if (cycleUpdates.size() != 1) {
              return reject(
                  kernel,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: guarded parent "
                  "coordinate reserve has an ambiguous control update");
            }
            predicateRouteVisiting.clear();
            if (!carriesPredicate(cycleUpdates.front().getValue(),
                                  childCompare))
              return reject(
                  kernel,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: parent-coordinate "
                  "handoff is not guarded by the child compare");
          }
          if (!llvm::is_contained(
                  sequentialRecurrences[childIndex].seedPredicates,
                  recurrence.compareResult))
            return reject(kernel,
                          "HOST_LOWERING_UNSUPPORTED_SHAPE: child reserve "
                          "is not reset from the parent's compare guard");
          negativeChildren.push_back(childIndex);
          if (child[index] >= 0 && child[index] != childIndex)
            return reject(kernel,
                          "HOST_LOWERING_UNSUPPORTED_SHAPE: recurrence has "
                          "multiple child reset guards");
          if (parent[childIndex] >= 0 && parent[childIndex] !=
                                             static_cast<int>(index))
            return reject(kernel,
                          "HOST_LOWERING_UNSUPPORTED_SHAPE: recurrence child "
                          "has multiple parents");
          child[index] = childIndex;
          parent[childIndex] = static_cast<int>(index);
        }
        if (!recurrence.cycleReserves.empty() && negativeChildren.empty())
          return reject(kernel,
                        "HOST_LOWERING_UNSUPPORTED_SHAPE: reserve cycle is "
                        "missing its child completion guard");
        if (!ownGuard)
          return reject(kernel,
                        "HOST_LOWERING_UNSUPPORTED_SHAPE: recurrence route "
                        "does not carry its positive compare guard");
        if (sequentialReductionBoundAttr &&
            sequentialReductionBoundAttr != recurrence.bound)
          return reject(kernel,
                        "HOST_LOWERING_UNSUPPORTED_SHAPE: multiple carried "
                        "reduction bounds require separate loops");
        sequentialReductionBoundAttr = recurrence.bound;
      }

      // More than one recurrence is accepted only as one explicit parent /
      // child chain.  Independent add-one cycles have no sufficient flattened
      // loop budget and therefore remain fail-closed.
      unsigned roots = 0;
      int root = -1;
      for (unsigned index = 0; index < parent.size(); ++index)
        if (parent[index] < 0) {
          ++roots;
          root = static_cast<int>(index);
        }
      if (sequentialRecurrences.size() > 1 && roots != 1)
        return reject(kernel,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: carried recurrences "
                      "are not one parent/child chain");
      if (roots == 1) {
        unsigned visited = 0;
        for (int index = root; index >= 0; index = child[index])
          ++visited;
        if (visited != sequentialRecurrences.size())
          return reject(kernel,
                        "HOST_LOWERING_UNSUPPORTED_SHAPE: carried recurrence "
                        "chain is incomplete");
      }

      for (SequentialRecurrence &recurrence : sequentialRecurrences) {
        if (!llvm::is_contained(sequentialReductionReserves,
                                recurrence.reserve))
          sequentialReductionReserves.push_back(recurrence.reserve);
      }
      FailureOr<HostDataValue> bound = resolveHostDataAttribute(
          sequentialReductionBoundAttr, builder.getIndexType(), inputValues,
          iterArgValues, truePredicate, builder, loc, kernel);
      if (failed(bound))
        return failure();
      if (!bound->value.getType().isIndex())
        return reject(kernel,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: sequential reduction "
                      "bound must have index payload");
      hasSequentialReduction = true;
      sequentialReductionUpper = *bound;
    }
  }

  /*
   * The old implementation of this block intentionally remains absent: all
   * recurrence acceptance above is tied to the route/seed proofs, rather
   * than to store counts or an unrelated add-one value.
   */

  // A reserve paired with grant_once(%iter_arg_initN) is the loop-carried
  // state for kernel iter_arg N.  Other reserves are still carried, since
  // post-dataflow kernels use them for software loop state even when the
  // kernel has no public iter_args/results.
  DenseMap<Value, unsigned> reserveToIterArg;
  for (Operation &operation : sourceBody.without_terminator()) {
    auto phiStart = dyn_cast<neura::PhiStartOp>(&operation);
    if (!phiStart)
      continue;
    auto grant = unwrapHostDataMoves(phiStart.getInitValue()).getDefiningOp<neura::GrantOnceOp>();
    if (!grant)
      continue;
    auto text = grant->getAttrOfType<StringAttr>("constant_value");
    if (!text)
      continue;
    StringRef name = text.getValue();
    if (!name.starts_with("%iter_arg_init"))
      continue;
    name = name.drop_front(strlen("%iter_arg_init"));
    unsigned index = 0;
    if (!name.empty() && !name.getAsInteger(10, index) &&
        index < kernel.getIterArgsInit().size())
      reserveToIterArg[unwrapHostDataMoves(phiStart.getReserved())] = index;
  }

  SmallVector<HostLoopState> states;
  SmallVector<unsigned> iterArgSlots(kernel.getIterArgsInit().size(), 0);
  for (auto [index, initial] : llvm::enumerate(kernel.getIterArgsInit())) {
    iterArgSlots[index] = states.size();
    states.push_back({initial.getType(), static_cast<int>(index), Value(),
                      /*output=*/false});
  }
  DenseMap<Value, unsigned> stateSlots;
  for (auto [reserve, iterArg] : reserveToIterArg)
    stateSlots[reserve] = iterArgSlots[iterArg];
  for (neura::ReserveOp reserve : reserves) {
    if (stateSlots.contains(reserve.getResult()))
      continue;
    unsigned slot = states.size();
    stateSlots[reserve.getResult()] = slot;
    states.push_back({payloadType(reserve.getResult().getType()), -1,
                      reserve.getResult(), /*output=*/false});
  }
  SmallVector<unsigned> outputSlots;
  for (Value result : kernel.getResults()) {
    unsigned slot = states.size();
    outputSlots.push_back(slot);
    states.push_back({result.getType(), -1, Value(), /*output=*/true});
  }

  auto isKnownRecurrenceControl = [&](Value predicate) {
    predicate = unwrapHostDataMoves(predicate);
    for (const SequentialRecurrence &recurrence : sequentialRecurrences) {
      if (predicate == recurrence.compareResult)
        return true;
      if (auto notOp = predicate.getDefiningOp<neura::NotOp>())
        if (unwrapHostDataMoves(notOp.getInput()) == recurrence.compareResult)
          return true;
    }
    return false;
  };

  // State validity is packet-scoped only when every incoming route carries a
  // proven recurrence guard.  A carried value with an unrelated predicate is
  // deliberately left on the ordinary OR-preserving path.
  std::function<bool(Value)> hasPacketScopedRoute;
  hasPacketScopedRoute = [&](Value value) -> bool {
    value = unwrapHostDataMoves(value);
    if (!value)
      return false;
    llvm::SmallPtrSet<Value, 32> visiting;
    std::function<bool(Value)> visit = [&](Value current) -> bool {
      current = unwrapHostDataMoves(current);
      if (!current || !visiting.insert(current).second)
        return false;
      auto clearVisiting = llvm::make_scope_exit(
          [&] { visiting.erase(current); });
      if (auto grant = current.getDefiningOp<neura::GrantPredicateOp>()) {
        if (isKnownRecurrenceControl(grant.getPredicate()))
          return true;
        return visit(grant.getValue());
      }
      // A recurrence control update is commonly routed through a unit-step
      // add before its index-to-int cast (for example, %70 -> %71 -> %30 in
      // the LU reduction kernel).  The add preserves packet validity only
      // when its one explicit operand carries the proven guard and the other
      // operand is an integer attribute; reject folded/data-dependent forms.
      if (auto add = current.getDefiningOp<neura::AddOp>()) {
        if (add->getNumOperands() != 1 || add->getAttr("lhs_value") ||
            !add->getAttrOfType<IntegerAttr>("rhs_value") ||
            payloadType(add->getOperand(0).getType()) !=
                payloadType(add.getResult().getType()))
          return false;
        return visit(add->getOperand(0));
      }
      if (auto cast = current.getDefiningOp<neura::CastOp>()) {
        StringRef castType = cast.getCastType();
        bool indexIntCast = castType == "index_to_int" ||
                            castType == "int_to_index";
        if (!indexIntCast &&
            payloadType(cast.getInput().getType()) !=
                payloadType(cast.getResult().getType()))
          return false;
        return visit(cast.getInput());
      }
      if (auto phi = current.getDefiningOp<neura::PhiOp>()) {
        if (phi->getNumOperands() != 2)
          return false;
        // A reserve operand denotes the carried value before this packet and
        // therefore cannot by itself prove packet scope.
        for (Value operand : phi->getOperands())
          if (unwrapHostDataMoves(operand).getDefiningOp<neura::ReserveOp>())
            return false;
        return visit(phi->getOperand(0)) && visit(phi->getOperand(1));
      }
      return false;
    };
    return visit(value);
  };

  // A seed store is recognized only from an explicit equality-to-zero test
  // of a compiler-owned counter.  This intentionally excludes repeated-target
  // and source-order heuristics.
  auto isInitialControlPredicate = [&](Value predicate) {
    predicate = unwrapHostDataMoves(predicate);
    auto cmp = predicate.getDefiningOp<neura::ICmpOp>();
    if (!cmp || cmp.getCmpType() != "eq")
      return false;
    auto rhs = cmp->getAttrOfType<IntegerAttr>("rhs_value");
    if (!rhs || !rhs.getValue().isZero())
      return false;
    return !!unwrapHostDataMoves(cmp.getLhs())
                 .getDefiningOp<neura::CounterOp>();
  };
  llvm::SmallPtrSet<Value, 16> initialRouteVisiting;
  std::function<bool(Value)> hasInitialControlRoute =
      [&](Value value) -> bool {
    value = unwrapHostDataMoves(value);
    if (!value || !initialRouteVisiting.insert(value).second)
      return false;
    auto clearVisiting = llvm::make_scope_exit(
        [&] { initialRouteVisiting.erase(value); });
    if (auto grant = value.getDefiningOp<neura::GrantPredicateOp>()) {
      if (isInitialControlPredicate(grant.getPredicate()))
        return true;
      return hasInitialControlRoute(grant.getValue());
    }
    if (auto cast = value.getDefiningOp<neura::CastOp>()) {
      if (payloadType(cast.getInput().getType()) !=
          payloadType(cast.getResult().getType()))
        return false;
      return hasInitialControlRoute(cast.getInput());
    }
    return false;
  };

  // Constant folding can remove the value operand of a store while retaining
  // it as lhs_value.  A folded store is a seed only under the narrow
  // compiler-owned initialization contract: its source reference resolves to
  // an explicit zero arith.constant, its destination is a direct kernel-input
  // memref reference, every index is a direct compiler-owned constant/counter,
  // and the store is reached before any dataflow state or predicate operation
  // is introduced.  The last two conditions are the initialization boundary;
  // source order alone or a repeated destination is not a seed proof.
  auto isCompilerOwnedZero = [&](Value value) {
    value = unwrapHostDataMoves(value);
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValueAttr());
      return integer && integer.getValue().isZero() &&
             (isa<IntegerType>(value.getType()) || value.getType().isIndex());
    }
    return false;
  };
  llvm::SmallPtrSet<Value, 32> initializerIndexVisiting;
  std::function<bool(Value)> isCompilerOwnedInitializerIndex;
  isCompilerOwnedInitializerIndex = [&](Value value) {
    value = unwrapHostDataMoves(value);
    if (!value || !initializerIndexVisiting.insert(value).second)
      return false;
    auto clearVisiting = llvm::make_scope_exit(
        [&] { initializerIndexVisiting.erase(value); });
    if (auto counter = value.getDefiningOp<neura::CounterOp>())
      return llvm::is_contained(counters, counter);
    if (auto constant = value.getDefiningOp<neura::ConstantOp>())
      return isa<IntegerAttr>(constant.getValueAttr());
    if (auto cast = value.getDefiningOp<neura::CastOp>())
      return isCompilerOwnedInitializerIndex(cast.getInput());
    return false;
  };
  auto hasSourceOwnedOnceProof = [&](neura::StoreIndexedOp store) {
    if (store->hasAttr(kSourceOwnedOnceInitAttr))
      return true;
    Value value = unwrapHostDataMoves(store.getValue());
    auto grant = value ? value.getDefiningOp<neura::GrantOnceOp>() : nullptr;
    return grant && grant->hasAttr(kSourceOwnedOnceInitAttr);
  };
  auto isFoldedZeroInitializer = [&](neura::StoreIndexedOp store) {
    Attribute lhsAttribute = store->getAttr("lhs_value");
    auto lhs = dyn_cast_or_null<StringAttr>(lhsAttribute);
    if (!lhs || store->getAttr("rhs_value") || !store.getBase() ||
        !isa<MemRefType>(payloadType(store.getBase().getType())) ||
        store.getIndices().empty())
      return false;
    StringRef reference = lhs.getValue();
    if (!reference.starts_with("%input"))
      return false;
    reference = reference.drop_front(strlen("%input"));
    unsigned inputIndex = 0;
    if (reference.empty() || reference.getAsInteger(10, inputIndex) ||
        inputIndex >= inputValues.size() ||
        !isCompilerOwnedZero(inputValues[inputIndex].value))
      return false;
    initializerIndexVisiting.clear();
    if (!llvm::all_of(store.getIndices(), [&](Value index) {
      initializerIndexVisiting.clear();
      return isCompilerOwnedInitializerIndex(index);
    }))
      return false;

    // The folded base must retain an explicit kernel-input reference.  A
    // carried memref or an SSA value produced by the recurrence cannot prove
    // that this store belongs to the source-owned initialization phase.
    Value base = unwrapHostDataMoves(store.getBase());
    auto baseConstant = base.getDefiningOp<neura::ConstantOp>();
    auto baseReference = baseConstant
                             ? dyn_cast<StringAttr>(
                                   baseConstant.getValueAttr())
                             : StringAttr();
    if (!baseReference || !baseReference.getValue().starts_with("%input"))
      return false;
    StringRef baseText = baseReference.getValue().drop_front(strlen("%input"));
    unsigned baseIndex = 0;
    if (baseText.empty() || baseText.getAsInteger(10, baseIndex) ||
        baseIndex >= inputValues.size() ||
        !isa<MemRefType>(payloadType(inputValues[baseIndex].value.getType())))
      return false;

    // Only constants, counters, and their value-preserving casts may precede
    // this store.  In particular, a grant/phi/control route before the store
    // would allow a loop-internal reset to masquerade as a one-time seed.
    for (Operation &prior : sourceBody.without_terminator()) {
      if (&prior == store.getOperation())
        return true;
      if (isa<neura::ConstantOp, neura::CounterOp, arith::ConstantOp>(
              &prior))
        continue;
      if (auto cast = dyn_cast<neura::CastOp>(&prior)) {
        initializerIndexVisiting.clear();
        if (isCompilerOwnedInitializerIndex(cast.getInput()))
          continue;
      }
      return false;
    }
    return false;
  };
  DenseSet<Operation *> provenSeedStores;
  bool seedCandidateSeen = false;
  if (hasSequentialReduction) {
    for (Operation &operation : sourceBody.without_terminator()) {
      auto store = dyn_cast<neura::StoreIndexedOp>(&operation);
      if (!store)
        continue;
      initialRouteVisiting.clear();
      bool explicitCounterSeed =
          !store->getAttr("lhs_value") &&
          hasInitialControlRoute(store.getValue());
      bool foldedZeroCandidate = isFoldedZeroInitializer(store);
      Value storeValue = unwrapHostDataMoves(store.getValue());
      bool grantOnceCandidate =
          storeValue && storeValue.getDefiningOp<neura::GrantOnceOp>();
      if (explicitCounterSeed || foldedZeroCandidate || grantOnceCandidate)
        seedCandidateSeen = true;
      bool sourceOwnedSeed = hasSourceOwnedOnceProof(store);
      if (!sourceOwnedSeed) {
        if (explicitCounterSeed || foldedZeroCandidate || grantOnceCandidate)
          return reject(
              kernel,
              "HOST_LOWERING_BODY_CONTRACT: seed candidate lacks "
              "compiler-owned once-init provenance");
        continue;
      }
      bool foldedZeroSeed = foldedZeroCandidate;
      // The producer marker is the source CFG proof.  The route/zero checks
      // remain useful diagnostics, but neither can authorize an unmarked
      // seed after flattening has erased its source boundary.
      if (!explicitCounterSeed && !foldedZeroSeed && !sourceOwnedSeed)
        continue;
      bool precedesAllRecurrenceUpdates = llvm::all_of(
          sequentialRecurrences, [&](const SequentialRecurrence &recurrence) {
            return store->isBeforeInBlock(recurrence.updateOperation);
          });
      if (precedesAllRecurrenceUpdates)
        provenSeedStores.insert(store.getOperation());
    }
  }
  if (hasSequentialReduction && seedCandidateSeen &&
      provenSeedStores.empty())
    return reject(kernel,
                  "HOST_LOWERING_BODY_CONTRACT: sequential reduction has "
                  "no compiler-owned once-init provenance");

  DenseSet<unsigned> packetScopedStateSlots;

  SmallVector<Value> initialStateValues;
  for (HostLoopState &state : states) {
    if (state.iterArg >= 0) {
      initialStateValues.push_back(iterArgValues[state.iterArg].value);
      initialStateValues.push_back(iterArgValues[state.iterArg].predicate);
      continue;
    }
    FailureOr<Value> zero = createHostZero(builder, loc, state.type, kernel);
    if (failed(zero))
      return failure();
    initialStateValues.push_back(*zero);
    initialStateValues.push_back(createHostBool(builder, loc, false));
  }

  SmallVector<scf::ForOp> loops;
  SmallVector<HostDataValue> innermostBounds;
  for (auto [position, counter] : llvm::enumerate(counters)) {
    HostCounterBounds bounds = counterBounds.lookup(counter.getResult());
    SmallVector<Value> initialValues;
    if (!hasSequentialReduction && position + 1 == counters.size())
      initialValues = initialStateValues;
    scf::ForOp loop = builder.create<scf::ForOp>(
        loc, bounds.lower.value, bounds.upper.value, bounds.step.value,
        initialValues);
    loops.push_back(loop);
    builder.setInsertionPointToStart(loop.getBody());
    if (position + 1 == counters.size()) {
      innermostBounds = {bounds.lower, bounds.upper, bounds.step};
      // State values are paired as payload, predicate in the SCF region.
      // Their mapping is installed after all counters have been materialized.
    }
  }
  if (hasSequentialReduction) {
    Value reductionStep = builder.create<arith::ConstantIndexOp>(loc, 1);
    // The flattened replay loop covers the lexicographic product of the
    // proven software recurrences.  A depth-one recurrence needs N body
    // iterations plus its terminating packet; depth two needs N*N+N plus
    // that packet.  Build the geometric sum from the number of distinct
    // reserve recurrences instead of assuming the LU shape.
    Value reductionUpper = sequentialReductionUpper.value;
    for (unsigned depth = 1; depth < sequentialReductionReserves.size();
         ++depth) {
      Value product = builder.create<arith::MulIOp>(
          loc, reductionUpper, sequentialReductionUpper.value);
      reductionUpper = builder.create<arith::AddIOp>(
          loc, product, sequentialReductionUpper.value);
    }
    reductionUpper = builder.create<arith::AddIOp>(
        loc, reductionUpper, reductionStep);
    HostCounterBounds bounds{
        {builder.create<arith::ConstantIndexOp>(loc, 0), truePredicate},
        {reductionUpper, sequentialReductionUpper.predicate},
        {reductionStep, truePredicate}};
    scf::ForOp loop = builder.create<scf::ForOp>(
        loc, bounds.lower.value, bounds.upper.value, bounds.step.value,
        initialStateValues);
    loops.push_back(loop);
    innermostBounds = {bounds.lower, bounds.upper, bounds.step};
    builder.setInsertionPointToStart(loop.getBody());
  }
  if (loops.empty())
    return reject(kernel,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: predicated kernel has "
                  "no materialized loop");

  scf::ForOp innermost = loops.back();
  Value firstIteration = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, innermost.getInductionVar(),
      innermostBounds.front().value);
  SmallVector<HostDataValue> currentStates;
  auto regionArgs = innermost.getRegionIterArgs();
  if (regionArgs.size() != states.size() * 2)
    return reject(kernel,
                  "HOST_LOWERING_BODY_CONTRACT: predicated loop state "
                  "argument count mismatch");
  for (unsigned index = 0; index < states.size(); ++index)
    currentStates.push_back({regionArgs[index * 2], regionArgs[index * 2 + 1]});

  HostDataMap values;
  unsigned inputCount = kernel.getInputs().size();
  for (auto [index, argument] : llvm::enumerate(sourceBody.getArguments())) {
    if (index < inputCount)
      values[argument] = inputValues[index];
    else
      values[argument] = currentStates[iterArgSlots[index - inputCount]];
  }
  for (auto [index, counter] : llvm::enumerate(counters)) {
    values[counter.getResult()] =
        {loops[index].getInductionVar(), truePredicate};
  }
  for (auto [reserve, slot] : stateSlots)
    values[reserve] = currentStates[slot];

  auto valid = [&](HostDataValue value) -> Value {
    return value.predicate ? value.predicate : truePredicate;
  };
  auto resolveOperand = [&](Value source, Operation *owner)
      -> FailureOr<HostDataValue> {
    return lookupHostData(source, values, mapping, owner);
  };
  // Constant folding removes a source operand while retaining its original
  // semantic slot in lhs_value/rhs_value. A remaining operand is not always
  // the lhs (sub/div/comparison make that distinction observable).
  auto resolveBinary = [&](Operation *owner)
      -> FailureOr<std::pair<HostDataValue, HostDataValue>> {
    Attribute lhsAttribute = owner->getAttr("lhs_value");
    Attribute rhsAttribute = owner->getAttr("rhs_value");
    unsigned expectedOperands = !lhsAttribute + !rhsAttribute;
    if (owner->getNumOperands() != expectedOperands)
      return reject(owner, "HOST_LOWERING_BODY_CONTRACT: folded binary source slots do not match operands");
    unsigned operand = 0;
    Type type = expectedOperands ? payloadType(owner->getOperand(0).getType())
                                 : payloadType(owner->getResult(0).getType());
    FailureOr<HostDataValue> lhs = lhsAttribute
        ? resolveHostDataAttribute(lhsAttribute, type, inputValues, iterArgValues,
                                   truePredicate, builder, loc, owner)
        : resolveOperand(owner->getOperand(operand++), owner);
    FailureOr<HostDataValue> rhs = rhsAttribute
        ? resolveHostDataAttribute(rhsAttribute, type, inputValues, iterArgValues,
                                   truePredicate, builder, loc, owner)
        : resolveOperand(owner->getOperand(operand++), owner);
    if (failed(lhs) || failed(rhs))
      return failure();
    return std::make_pair(*lhs, *rhs);
  };

  DenseMap<unsigned, HostDataValue> pendingUpdates;
  unsigned returnIndex = 0;
  for (Operation &operation : sourceBody.without_terminator()) {
    if (auto constant = dyn_cast<neura::ConstantOp>(&operation)) {
      FailureOr<HostDataValue> value = resolveHostDataAttribute(
          constant.getValueAttr(), payloadType(constant.getResult().getType()),
          inputValues, iterArgValues, truePredicate, builder, loc, constant);
      if (failed(value))
        return failure();
      values[constant.getResult()] = *value;
      continue;
    }
    if (isa<neura::CounterOp, neura::ReserveOp>(&operation))
      continue;
    if (auto arithConstant = dyn_cast<arith::ConstantOp>(&operation)) {
      Operation *clone = builder.clone(*arithConstant);
      values[arithConstant.getResult()] = {clone->getResult(0), truePredicate};
      continue;
    }
    if (auto move = dyn_cast<neura::DataMovOp>(&operation)) {
      if (move->getOperand(0).getType() != move->getResult(0).getType())
        return reject(move, "HOST_LOWERING_BODY_CONTRACT: data_mov must preserve its payload and predicate type");
      FailureOr<HostDataValue> source = resolveOperand(move->getOperand(0), move);
      if (failed(source))
        return failure();
      // Register routing preserves both payload and predicate validity.
      values[move->getResult(0)] = *source;
      continue;
    }
    if (auto grant = dyn_cast<neura::GrantOnceOp>(&operation)) {
      HostDataValue source;
      if (Value operand = grant.getValue()) {
        FailureOr<HostDataValue> resolved = resolveOperand(operand, grant);
        if (failed(resolved))
          return failure();
        source = *resolved;
      } else if (grant.getConstantValue().has_value()) {
        FailureOr<HostDataValue> resolved = resolveHostDataAttribute(
            *grant.getConstantValue(), payloadType(grant.getResult().getType()),
            inputValues, iterArgValues, truePredicate, builder, loc, grant);
        if (failed(resolved))
          return failure();
        source = *resolved;
      } else {
        return reject(grant,
                      "HOST_LOWERING_BODY_CONTRACT: grant_once requires a "
                      "value operand or constant_value");
      }
      values[grant.getResult()] =
          {source.value, builder.create<arith::AndIOp>(loc, valid(source),
                                                        firstIteration)};
      continue;
    }
    if (auto phiStart = dyn_cast<neura::PhiStartOp>(&operation)) {
      FailureOr<HostDataValue> init =
          resolveOperand(phiStart.getInitValue(), phiStart);
      FailureOr<HostDataValue> reserved =
          resolveOperand(phiStart.getReserved(), phiStart);
      if (failed(init) || failed(reserved))
        return failure();
      Value initValid = valid(*init);
      Value reservedValid = valid(*reserved);
      values[phiStart.getResult()] = {
          builder.create<arith::SelectOp>(loc, initValid, init->value,
                                           reserved->value),
          builder.create<arith::OrIOp>(loc, initValid, reservedValid)};
      continue;
    }
    if (auto phi = dyn_cast<neura::PhiOp>(&operation)) {
      if (phi->getNumOperands() < 2)
        return reject(phi,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: dataflow phi needs "
                      "at least two operands");
      FailureOr<HostDataValue> first = resolveOperand(phi->getOperand(0), phi);
      FailureOr<HostDataValue> second = resolveOperand(phi->getOperand(1), phi);
      if (failed(first) || failed(second))
        return failure();
      Value firstValid = valid(*first);
      Value secondValid = valid(*second);
      values[phi.getResult()] = {
          builder.create<arith::SelectOp>(loc, firstValid, first->value,
                                           second->value),
          builder.create<arith::OrIOp>(loc, firstValid, secondValid)};
      continue;
    }
    if (auto grant = dyn_cast<neura::GrantPredicateOp>(&operation)) {
      FailureOr<HostDataValue> source =
          resolveOperand(grant.getValue(), grant);
      FailureOr<HostDataValue> predicate =
          resolveOperand(grant.getPredicate(), grant);
      if (failed(source) || failed(predicate))
        return failure();
      Value predicateValue = predicate->value;
      if (!predicateValue.getType().isSignlessInteger(1))
        return reject(grant,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: grant_predicate "
                      "requires an i1 payload");
      Value combined = builder.create<arith::AndIOp>(
          loc, valid(*source),
          builder.create<arith::AndIOp>(loc, valid(*predicate), predicateValue));
      values[grant.getResult()] = {source->value, combined};
      continue;
    }
    if (auto extract = dyn_cast<neura::ExtractPredicateOp>(&operation)) {
      FailureOr<HostDataValue> source =
          resolveOperand(extract.getInput(), extract);
      if (failed(source))
        return failure();
      Value predicateValue = valid(*source);
      if (auto counter = extract.getInput().getDefiningOp<neura::CounterOp>()) {
        auto found = counterBounds.find(counter.getResult());
        if (found == counterBounds.end())
          return reject(extract,
                        "HOST_LOWERING_BODY_CONTRACT: counter bounds were "
                        "not materialized before extract_predicate");
        Value next = builder.create<arith::AddIOp>(
            loc, source->value, found->second.step.value);
        predicateValue = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, next, found->second.upper.value);
      }
      values[extract.getPredicate()] = {predicateValue, truePredicate};
      continue;
    }
    if (auto notOp = dyn_cast<neura::NotOp>(&operation)) {
      FailureOr<HostDataValue> source = resolveOperand(notOp.getInput(), notOp);
      if (failed(source))
        return failure();
      if (!source->value.getType().isSignlessInteger(1))
        return reject(notOp,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: neura.not requires "
                      "an i1 payload");
      Value inverted = builder.create<arith::XOrIOp>(
          loc, source->value, createHostBool(builder, loc, true));
      values[notOp.getOutput()] = {inverted, valid(*source)};
      continue;
    }
    if (auto cmp = dyn_cast<neura::ICmpOp>(&operation)) {
      StringRef predicate = cmp.getCmpType();
      FailureOr<arith::CmpIPredicate> parsedPredicate =
          parseHostCmpPredicate(predicate);
      if (failed(parsedPredicate))
        return reject(cmp,
                      "HOST_LOWERING_UNSUPPORTED_OPERATION: unsupported "
                      "integer comparison predicate " + predicate);
      auto operands = resolveBinary(cmp);
      if (failed(operands))
        return failure();
      auto [lhs, rhs] = *operands;
      if (lhs.value.getType() != rhs.value.getType())
        return reject(cmp, "HOST_LOWERING_BODY_CONTRACT: comparison payload types differ");
      Value result = builder.create<arith::CmpIOp>(
          loc, *parsedPredicate, lhs.value, rhs.value);
      values[cmp.getResult()] = {
          result, builder.create<arith::AndIOp>(loc, valid(lhs), valid(rhs))};
      continue;
    }
    if (auto select = dyn_cast<neura::SelOp>(&operation)) {
      FailureOr<HostDataValue> condition = resolveOperand(select.getCond(), select);
      FailureOr<HostDataValue> ifTrue = resolveOperand(select.getIfTrue(), select);
      FailureOr<HostDataValue> ifFalse = resolveOperand(select.getIfFalse(), select);
      if (failed(condition) || failed(ifTrue) || failed(ifFalse))
        return failure();
      if (!condition->value.getType().isSignlessInteger(1) ||
          ifTrue->value.getType() != ifFalse->value.getType() ||
          ifTrue->value.getType() != payloadType(select.getResult().getType()))
        return reject(select,
                      "HOST_LOWERING_BODY_CONTRACT: sel requires an i1 condition "
                      "and matching branch/result payload types");
      Value selected = builder.create<arith::SelectOp>(
          loc, condition->value, ifTrue->value, ifFalse->value);
      Value selectedPredicate = builder.create<arith::SelectOp>(
          loc, condition->value, valid(*ifTrue), valid(*ifFalse));
      // Only the selected branch needs a valid packet. Requiring both would
      // suppress values produced by mutually exclusive dataflow branches.
      values[select.getResult()] = {
          selected, builder.create<arith::AndIOp>(
                        loc, valid(*condition), selectedPredicate)};
      continue;
    }
    if (auto add = dyn_cast<neura::AddOp>(&operation);
        add || isa<neura::SubOp, neura::MulOp, neura::DivOp>(&operation)) {
      Value resultValue;
      StringRef name = operation.getName().getStringRef();
      auto operands = resolveBinary(&operation);
      if (failed(operands))
        return failure();
      auto [lhsData, rhsData] = *operands;
      if (lhsData.value.getType() != rhsData.value.getType() ||
          lhsData.value.getType() != payloadType(operation.getResult(0).getType()) ||
          (!lhsData.value.getType().isIndex() &&
           !isa<IntegerType>(lhsData.value.getType())))
        return reject(&operation,
                      "HOST_LOWERING_BODY_CONTRACT: integer arithmetic requires "
                      "matching operand/result payload types");
      Value arithmeticPredicate = builder.create<arith::AndIOp>(
          loc, valid(lhsData), valid(rhsData));
      if (name == "neura.add")
        resultValue = builder.create<arith::AddIOp>(loc, lhsData.value,
                                                    rhsData.value);
      else if (name == "neura.sub")
        resultValue = builder.create<arith::SubIOp>(loc, lhsData.value,
                                                    rhsData.value);
      else if (name == "neura.mul")
        resultValue = builder.create<arith::MulIOp>(loc, lhsData.value,
                                                    rhsData.value);
      else {
        if (!rhsData.value.getType().isIndex() &&
            !isa<IntegerType>(rhsData.value.getType()))
          return reject(&operation,
                        "HOST_LOWERING_UNSUPPORTED_SHAPE: dataflow div "
                        "requires an integer or index divisor");
        FailureOr<Value> zero =
            createHostZero(builder, loc, rhsData.value.getType(), &operation);
        if (failed(zero))
          return failure();
        Value nonzero = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, rhsData.value, *zero);
        arithmeticPredicate = builder.create<arith::AndIOp>(
            loc, arithmeticPredicate, nonzero);

        // arith.divsi is not speculative-safe for a zero divisor. Keep the
        // operation in the valid/nonzero branch and provide an invalid zero
        // payload on the else path; consumers propagate the predicate.
        scf::IfOp guarded = builder.create<scf::IfOp>(
            loc, TypeRange{lhsData.value.getType()}, arithmeticPredicate,
            /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard divGuard(builder);
          builder.setInsertionPointToStart(&guarded.getThenRegion().front());
          Value quotient = builder.create<arith::DivSIOp>(
              loc, lhsData.value, rhsData.value);
          builder.create<scf::YieldOp>(loc, quotient);
          builder.setInsertionPointToStart(&guarded.getElseRegion().front());
          FailureOr<Value> invalid = createHostZero(
              builder, loc, lhsData.value.getType(), &operation);
          if (failed(invalid))
            return failure();
          builder.create<scf::YieldOp>(loc, *invalid);
        }
        resultValue = guarded.getResult(0);
      }
      values[operation.getResult(0)] = {resultValue, arithmeticPredicate};
      continue;
    }
    if (auto cast = dyn_cast<neura::CastOp>(&operation)) {
      FailureOr<HostDataValue> source = resolveOperand(cast.getInput(), cast);
      if (failed(source))
        return failure();
      Type resultType = payloadType(cast.getResult().getType());
      Value result;
      StringRef castType = cast.getCastType();
      if (castType == "index_to_int" || castType == "int_to_index")
        result = builder.create<arith::IndexCastOp>(loc, resultType,
                                                    source->value);
      else if (castType == "extui") {
        auto sourceInteger = dyn_cast<IntegerType>(source->value.getType());
        auto resultInteger = dyn_cast<IntegerType>(resultType);
        if (!sourceInteger || !resultInteger ||
            sourceInteger.getWidth() >= resultInteger.getWidth())
          return reject(cast,
                        "HOST_LOWERING_BODY_CONTRACT: extui requires a wider "
                        "integer result");
        result = builder.create<arith::ExtUIOp>(loc, resultType, source->value);
      }
      else
        return reject(cast,
                      "HOST_LOWERING_UNSUPPORTED_OPERATION: unsupported "
                      "dataflow cast type " + castType);
      values[cast.getResult()] = {result, valid(*source)};
      continue;
    }
    if (auto load = dyn_cast<neura::LoadIndexedOp>(&operation)) {
      HostDataValue base;
      if (Value baseOperand = load.getBase()) {
        FailureOr<HostDataValue> resolved =
            resolveOperand(baseOperand, load);
        if (failed(resolved))
          return failure();
        base = *resolved;
      } else {
        Attribute attr = load->getAttr("lhs_value");
        if (!attr)
          return reject(load,
                        "HOST_LOWERING_BODY_CONTRACT: load_indexed has no "
                        "base or lhs_value reference");
        FailureOr<HostDataValue> resolved = resolveHostDataReference(
            attr, inputValues, iterArgValues, load);
        if (failed(resolved))
          return failure();
        base = *resolved;
      }
      auto memref = dyn_cast<MemRefType>(base.value.getType());
      if (!memref)
        return reject(load,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: load base is not a "
                      "memref");
      if (memref.getRank() != load.getIndices().size())
        return reject(load,
                      "HOST_LOWERING_BODY_CONTRACT: load index rank does not "
                      "match its memref rank");
      Type loadType = payloadType(load.getResult().getType());
      if (memref.getElementType() != loadType)
        return reject(load,
                      "HOST_LOWERING_BODY_CONTRACT: load result type does "
                      "not match its memref element type");
      SmallVector<Value> indices;
      Value predicate = valid(base);
      for (auto [dimension, source] : llvm::enumerate(load.getIndices())) {
        FailureOr<HostDataValue> index = resolveOperand(source, load);
        if (failed(index))
          return failure();
        if (!index->value.getType().isIndex())
          return reject(load,
                        "HOST_LOWERING_UNSUPPORTED_SHAPE: load index is not "
                        "an index payload");
        indices.push_back(index->value);
        predicate = builder.create<arith::AndIOp>(loc, predicate, valid(*index));
        Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value lower = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, index->value, zero);
        Value extent;
        if (memref.isDynamicDim(dimension)) {
          auto baseExtents = dynamicExtents.find(base.value);
          if (baseExtents == dynamicExtents.end())
            return reject(load,
                          "HOST_LOWERING_BODY_CONTRACT: dynamic load base "
                          "has no cached extent");
          auto extentIt = baseExtents->second.find(dimension);
          if (extentIt == baseExtents->second.end())
            return reject(load,
                          "HOST_LOWERING_BODY_CONTRACT: dynamic load "
                          "dimension has no cached extent");
          extent = extentIt->second;
        } else
          extent = builder.create<arith::ConstantIndexOp>(
              loc, memref.getDimSize(dimension));
        Value upper = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, index->value, extent);
        predicate = builder.create<arith::AndIOp>(
            loc, predicate, builder.create<arith::AndIOp>(loc, lower, upper));
      }
      // A predicate-invalid load must not touch memory.  The else value is
      // deliberately marked invalid below, so its payload cannot be used as
      // a valid computation result.  Bounds are included in the guard even
      // for valid predicates because post-dataflow indices are ordinary SSA
      // values and memref.load has no implicit bounds check.
      scf::IfOp guarded = builder.create<scf::IfOp>(
          loc, TypeRange{loadType}, predicate, /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard loadGuard(builder);
        builder.setInsertionPointToStart(&guarded.getThenRegion().front());
        Value loaded = builder.create<memref::LoadOp>(
            loc, loadType, base.value, indices);
        builder.create<scf::YieldOp>(loc, loaded);
        builder.setInsertionPointToStart(&guarded.getElseRegion().front());
        FailureOr<Value> zero = createHostZero(builder, loc, loadType, load);
        if (failed(zero))
          return failure();
        builder.create<scf::YieldOp>(loc, *zero);
      }
      values[load.getResult()] = {guarded.getResult(0), predicate};
      continue;
    }
    if (auto store = dyn_cast<neura::StoreIndexedOp>(&operation)) {
      FailureOr<HostDataValue> source;
      if (Attribute sourceAttr = store->getAttr("lhs_value"))
        source = resolveHostDataReference(sourceAttr, inputValues,
                                          iterArgValues, store);
      else
        source = resolveOperand(store.getValue(), store);
      if (failed(source))
        return failure();
      HostDataValue base;
      if (Attribute baseAttr = store->getAttr("rhs_value")) {
        FailureOr<HostDataValue> resolved = resolveHostDataReference(
            baseAttr, inputValues, iterArgValues, store);
        if (failed(resolved))
          return failure();
        base = *resolved;
      } else if (Value baseOperand = store.getBase()) {
        FailureOr<HostDataValue> resolved =
            resolveOperand(baseOperand, store);
        if (failed(resolved))
          return failure();
        base = *resolved;
      } else {
        return reject(store,
                      "HOST_LOWERING_BODY_CONTRACT: store_indexed has no "
                      "base or rhs_value memory reference");
      }
      auto memref = dyn_cast<MemRefType>(base.value.getType());
      if (!memref)
        return reject(store,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: store base is not a "
                      "memref");
      if (memref.getRank() != store.getIndices().size())
        return reject(store,
                      "HOST_LOWERING_BODY_CONTRACT: store index rank does "
                      "not match its memref rank");
      if (memref.getElementType() != source->value.getType())
        return reject(store,
                      "HOST_LOWERING_BODY_CONTRACT: store value type does "
                      "not match its memref element type");
      SmallVector<Value> indices;
      Value predicate = builder.create<arith::AndIOp>(loc, valid(*source),
                                                      valid(base));
      for (auto [dimension, sourceIndex] :
           llvm::enumerate(store.getIndices())) {
        FailureOr<HostDataValue> index =
            resolveOperand(sourceIndex, store);
        if (failed(index))
          return failure();
        if (!index->value.getType().isIndex())
          return reject(store,
                        "HOST_LOWERING_UNSUPPORTED_SHAPE: store index is not "
                        "an index payload");
        indices.push_back(index->value);
        predicate = builder.create<arith::AndIOp>(loc, predicate,
                                                  valid(*index));
        Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value lower = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, index->value, zero);
        Value extent;
        if (memref.isDynamicDim(dimension)) {
          auto baseExtents = dynamicExtents.find(base.value);
          if (baseExtents == dynamicExtents.end())
            return reject(store,
                          "HOST_LOWERING_BODY_CONTRACT: dynamic store base "
                          "has no cached extent");
          auto extentIt = baseExtents->second.find(dimension);
          if (extentIt == baseExtents->second.end())
            return reject(store,
                          "HOST_LOWERING_BODY_CONTRACT: dynamic store "
                          "dimension has no cached extent");
          extent = extentIt->second;
        } else
          extent = builder.create<arith::ConstantIndexOp>(
              loc, memref.getDimSize(dimension));
        Value upper = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, index->value, extent);
        predicate = builder.create<arith::AndIOp>(
            loc, predicate, builder.create<arith::AndIOp>(loc, lower, upper));
      }
      // Only an explicit counter==0 grant route is a seed boundary.  Stores
      // with the same destination but no such SSA/control proof remain live
      // on every valid packet.
      if (hasSequentialReduction && provenSeedStores.contains(store.getOperation()))
        predicate = builder.create<arith::AndIOp>(loc, predicate,
                                                  firstIteration);
      scf::IfOp conditional = builder.create<scf::IfOp>(loc, predicate);
      OpBuilder::InsertionGuard storeGuard(builder);
      builder.setInsertionPointToStart(&conditional.getThenRegion().front());
      builder.create<memref::StoreOp>(loc, source->value, base.value, indices);
      continue;
    }
    if (auto ctrl = dyn_cast<neura::CtrlMovOp>(&operation)) {
      FailureOr<HostDataValue> source = resolveOperand(ctrl.getValue(), ctrl);
      if (failed(source))
        return failure();
      auto found = stateSlots.find(ctrl.getTarget());
      if (found == stateSlots.end())
        return reject(ctrl,
                      "HOST_LOWERING_BODY_CONTRACT: ctrl_mov target is not "
                      "a loop-carried reserve");
      if (pendingUpdates.contains(found->second))
        return reject(ctrl,
                      "HOST_LOWERING_UNSUPPORTED_SHAPE: loop-carried reserve "
                      "has multiple control updates");
      pendingUpdates[found->second] = *source;
      bool packetScoped = hasPacketScopedRoute(ctrl.getValue());
      if (packetScoped)
        packetScopedStateSlots.insert(found->second);
      continue;
    }
    if (auto returned = dyn_cast<neura::ReturnValueOp>(&operation)) {
      for (Value sourceValue : returned.getValues()) {
        if (returnIndex >= outputSlots.size())
          return reject(returned,
                        "HOST_LOWERING_BODY_CONTRACT: too many return values");
        FailureOr<HostDataValue> source =
            resolveOperand(sourceValue, returned);
        if (failed(source))
          return failure();
        unsigned slot = outputSlots[returnIndex++];
        Value sourceValid = valid(*source);
        bool packetScoped = hasPacketScopedRoute(sourceValue);
        currentStates[slot] = {
            builder.create<arith::SelectOp>(loc, sourceValid, source->value,
                                             currentStates[slot].value),
            packetScoped
                ? sourceValid
                : builder.create<arith::OrIOp>(loc, sourceValid,
                                               valid(currentStates[slot]))};
      }
      continue;
    }
    if (isa<neura::YieldOp>(&operation))
      continue;
    return reject(&operation,
                  "HOST_LOWERING_UNSUPPORTED_OPERATION: unsupported "
                  "post-Neura dataflow operation " +
                      operation.getName().getStringRef());
  }
  if (returnIndex != outputSlots.size())
    return reject(kernel,
                  "HOST_LOWERING_BODY_CONTRACT: missing predicated return "
                  "value");

  SmallVector<Value> nextStateValues;
  for (unsigned index = 0; index < states.size(); ++index) {
    HostDataValue next = currentStates[index];
    if (auto update = pendingUpdates.find(index); update != pendingUpdates.end()) {
      Value updateValid = valid(update->second);
      next = {builder.create<arith::SelectOp>(
                  loc, updateValid, update->second.value, currentStates[index].value),
              packetScopedStateSlots.contains(index)
                  ? updateValid
                  : builder.create<arith::OrIOp>(
                        loc, updateValid, valid(currentStates[index]))};
    } else if (packetScopedStateSlots.contains(index)) {
      // A packet-scoped state with no update in this packet is invalid, while
      // retaining its payload for a later valid select.  Independent carried
      // state takes the old OR-preserving path above.
      next.predicate = createHostBool(builder, loc, false);
    }
    nextStateValues.push_back(next.value);
    nextStateValues.push_back(valid(next));
  }
  builder.setInsertionPointToEnd(innermost.getBody());
  if (!innermost.getBody()->empty() &&
      innermost.getBody()->back().hasTrait<OpTrait::IsTerminator>())
    innermost.getBody()->back().erase();
  builder.setInsertionPointToEnd(innermost.getBody());
  builder.create<scf::YieldOp>(loc, nextStateValues);
  for (scf::ForOp loop : loops) {
    if (!loop.getBody()->empty() &&
        loop.getBody()->back().hasTrait<OpTrait::IsTerminator>())
      continue;
    builder.setInsertionPointToEnd(loop.getBody());
    builder.create<scf::YieldOp>(loc);
  }

  if (innermost.getNumResults() != states.size() * 2)
    return reject(kernel,
                  "HOST_LOWERING_BODY_CONTRACT: predicated loop result "
                  "count mismatch");
  for (auto [result, slot] : llvm::zip(kernel.getResults(), outputSlots)) {
    mapping.map(result, innermost.getResult(slot * 2));
  }
  completionSite = loops.front().getOperation();
  kernel.erase();
  return success();
}

// Lowers a Taskflow-to-Neura kernel before the later Neura dataflow
// conversion. At this point the kernel already owns neura.counter operations,
// while its computation is still ordinary arith/memref/SCF host IR. This is
// the executable bridge used by native replay; a kernel containing dataflow
// values is rejected by validateNeuraKernel instead of being approximated.
static LogicalResult lowerNeuraKernel(neura::KernelOp kernel,
                                      IRMapping &mapping, OpBuilder &builder,
                                      Operation *&completionSite) {
  if (isPredicatedKernel(kernel))
    return lowerPredicatedNeuraKernel(kernel, mapping, builder,
                                      completionSite);
  Block &sourceBody = kernel.getBody().front();
  SmallVector<neura::CounterOp> counters;
  for (Operation &operation : sourceBody.without_terminator())
    if (auto counter = dyn_cast<neura::CounterOp>(&operation))
      counters.push_back(counter);
  if (counters.empty())
    return reject(kernel,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: neura.kernel has no "
                  "counter loop");

  OpBuilder::InsertionGuard guard(builder);
  // Constants internalized by Taskflow-to-Neura live in the isolated kernel
  // region. Clone them into the host function before resolving loop bounds.
  for (Operation &operation : sourceBody.without_terminator())
    if (isa<arith::ConstantOp>(&operation))
      builder.clone(operation, mapping);

  unsigned inputCount = kernel.getInputs().size();
  for (auto [index, argument] : llvm::enumerate(sourceBody.getArguments())) {
    Value source = index < inputCount
                       ? kernel.getInputs()[index]
                       : kernel.getIterArgsInit()[index - inputCount];
    FailureOr<Value> mapped = mappedValue(mapping, source, kernel);
    if (failed(mapped))
      return failure();
    mapping.map(argument, *mapped);
  }

  SmallVector<scf::ForOp> loops;
  for (auto [position, counter] : llvm::enumerate(counters)) {
    FailureOr<Value> lower = mappedValue(mapping, counter.getLowerBound(), counter);
    FailureOr<Value> upper = mappedValue(mapping, counter.getUpperBound(), counter);
    FailureOr<Value> step = mappedValue(mapping, counter.getStep(), counter);
    if (failed(lower) || failed(upper) || failed(step))
      return failure();
    SmallVector<Value> initialValues;
    if (position + 1 == counters.size())
      for (Value initial : kernel.getIterArgsInit()) {
        FailureOr<Value> mapped = mappedValue(mapping, initial, kernel);
        if (failed(mapped))
          return failure();
        initialValues.push_back(*mapped);
      }
    scf::ForOp loop = builder.create<scf::ForOp>(
        kernel.getLoc(), *lower, *upper, *step, initialValues);
    loops.push_back(loop);
    builder.setInsertionPointToStart(loop.getBody());
    mapping.map(counter.getCurrentIndex(), loop.getInductionVar());
  }
  if (loops.empty())
    return reject(kernel,
                  "HOST_LOWERING_UNSUPPORTED_SHAPE: neura.kernel has no "
                  "counter loop");

  if (!kernel.getIterArgsInit().empty())
    for (auto [position, argument] :
         llvm::enumerate(sourceBody.getArguments().drop_front(inputCount)))
      mapping.map(argument, loops.back().getRegionIterArgs()[position]);

  builder.setInsertionPointToStart(loops.back().getBody());
  for (Operation &operation : sourceBody.without_terminator()) {
    if (isa<arith::ConstantOp, neura::CounterOp>(&operation))
      continue;
    for (Value operand : operation.getOperands())
      if (!mapping.lookupOrNull(operand))
        return reject(&operation,
                      "HOST_LOWERING_BODY_CONTRACT: neura.kernel operand was "
                      "not mapped into host IR");
    builder.clone(operation, mapping);
  }

  auto sourceYield = dyn_cast<neura::YieldOp>(sourceBody.getTerminator());
  if (!sourceYield)
    return reject(kernel,
                  "HOST_LOWERING_BODY_CONTRACT: neura.kernel terminator must "
                  "be neura.yield");
  if (!sourceYield.getIterArgsNext().empty()) {
    SmallVector<Value> nextValues;
    for (Value next : sourceYield.getIterArgsNext()) {
      FailureOr<Value> mapped = mappedValue(mapping, next, sourceYield);
      if (failed(mapped))
        return failure();
      nextValues.push_back(*mapped);
    }
    OpBuilder::InsertionGuard yieldGuard(builder);
    builder.setInsertionPointToEnd(loops.back().getBody());
    builder.create<scf::YieldOp>(kernel.getLoc(), nextValues);
  }

  if (!kernel.getIterArgsInit().empty()) {
    if (kernel.getNumResults() != loops.back().getNumResults())
      return reject(kernel,
                    "HOST_LOWERING_BODY_CONTRACT: neura.kernel loop/result "
                    "count mismatch");
    for (auto [result, loopResult] :
         llvm::zip(kernel.getResults(), loops.back().getResults()))
      mapping.map(result, loopResult);
  } else {
    for (auto [result, yielded] :
         llvm::zip(kernel.getResults(), sourceYield.getResults())) {
      FailureOr<Value> mapped = mappedValue(mapping, yielded, sourceYield);
      if (failed(mapped))
        return failure();
      mapping.map(result, *mapped);
    }
  }

  for (scf::ForOp loop : loops) {
    if (loop.getBody()->getTerminator())
      continue;
    OpBuilder::InsertionGuard yieldGuard(builder);
    builder.setInsertionPointToEnd(loop.getBody());
    builder.create<scf::YieldOp>(kernel.getLoc());
  }
  completionSite = loops.front().getOperation();
  kernel.erase();
  return success();
}

static void markCompletionSite(TaskflowTaskOp task, Operation *completionSite,
                               OpBuilder &builder) {
  if (!completionSite)
    return;
  completionSite->setAttr(kHostTaskNameAttr,
                          builder.getStringAttr(task.getTaskName()));
  completionSite->setAttr(kHostTaskCompletionAttr,
                          UnitAttr::get(task.getContext()));
  // Preserve semantic provenance at the host completion site.  This lets a
  // host interpreter observe partial buffers, state-carry blocks, and numeric
  // reduction tasks without retaining a non-executable Taskflow marker op.
  for (NamedAttribute attribute : task->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name.starts_with("amoeba.semantic.") ||
        name.starts_with("amoeba.tiling."))
      completionSite->setAttr(name, attribute.getValue());
  }
}

static LogicalResult lowerTask(TaskflowTaskOp task, unsigned ordinal,
                               bool captureTaskStates) {
  Block &taskBody = task.getBody().front();
  SmallVector<Value> inputs;
  llvm::append_range(inputs, task.getWillReads());
  llvm::append_range(inputs, task.getWillWrites());
  llvm::append_range(inputs, task.getValueInputs());

  IRMapping mapping;
  for (auto [argument, input] : llvm::zip(taskBody.getArguments(), inputs))
    mapping.map(argument, input);

  OpBuilder builder(task.getContext());
  builder.setInsertionPoint(task);
  TaskflowYieldOp yield = cast<TaskflowYieldOp>(taskBody.getTerminator());
  Operation *completionSite = nullptr;
  SmallVector<Operation *> bodyOperations;
  for (Operation &operation : taskBody.without_terminator())
    bodyOperations.push_back(&operation);
  for (Operation *operation : bodyOperations) {
    if (isa<TaskflowCounterOp>(operation))
      continue;
    if (auto hyperblock = dyn_cast<TaskflowHyperblockOp>(operation)) {
      if (failed(lowerHyperblock(hyperblock, mapping, builder, completionSite)))
        return failure();
      builder.setInsertionPoint(task);
      continue;
    }
    if (auto kernel = dyn_cast<neura::KernelOp>(operation)) {
      if (failed(lowerNeuraKernel(kernel, mapping, builder, completionSite)))
        return failure();
      builder.setInsertionPoint(task);
      continue;
    }
    for (Value operand : operation->getOperands())
      if (!mapping.lookupOrNull(operand))
        return reject(operation,
                      "HOST_LOWERING_BODY_CONTRACT: task operand was not "
                      "mapped into the host function");
    completionSite = builder.clone(*operation, mapping);
    builder.setInsertionPoint(task);
  }

  SmallVector<Value> replacements;
  auto appendMapped = [&](ValueRange values) -> LogicalResult {
    for (Value value : values) {
      FailureOr<Value> mapped = mappedValue(mapping, value, task);
      if (failed(mapped))
        return failure();
      replacements.push_back(*mapped);
    }
    return success();
  };
  if (failed(appendMapped(yield.getDoneReads())) ||
      failed(appendMapped(yield.getDoneWrites())) ||
      failed(appendMapped(yield.getValueResults())))
    return failure();
  if (replacements.size() != task->getNumResults())
    return reject(task, "HOST_LOWERING_BODY_CONTRACT: replacement result count "
                        "does not match task results");
  markCompletionSite(task, completionSite, builder);
  if (captureTaskStates) {
    if (!completionSite || yield.getDoneWrites().empty())
      return reject(task, "HOST_LOWERING_TRACE_UNSUPPORTED: task has no "
                          "completion site or written state");
    ModuleOp module = task->getParentOfType<ModuleOp>();
    auto function = task->getParentOfType<func::FuncOp>();
    OpBuilder traceBuilder(task.getContext());
    traceBuilder.setInsertionPointAfter(completionSite);
    for (auto [writeIndex, written] : llvm::enumerate(yield.getDoneWrites())) {
      Value state = mapping.lookupOrNull(written);
      auto type =
          dyn_cast_or_null<MemRefType>(state ? state.getType() : Type());
      if (!type || !type.hasStaticShape())
        return reject(task, "HOST_LOWERING_TRACE_UNSUPPORTED: written state "
                            "must be a static memref");
      std::string symbol = ("__orbit_trace_" + function.getSymName() + "_" +
                            Twine(ordinal) + "_" + Twine(writeIndex))
                               .str();
      OpBuilder globalBuilder(task.getContext());
      globalBuilder.setInsertionPointToStart(module.getBody());
      auto global = globalBuilder.create<memref::GlobalOp>(
          task.getLoc(), globalBuilder.getStringAttr(symbol), StringAttr(),
          TypeAttr::get(type), UnitAttr::get(task.getContext()), UnitAttr(),
          IntegerAttr());
      auto snapshot = traceBuilder.create<memref::GetGlobalOp>(
          task.getLoc(), type, global.getName());
      auto copy = traceBuilder.create<memref::CopyOp>(task.getLoc(), state,
                                                      snapshot.getResult());
      copy->setAttr(kHostTraceSymbolAttr, traceBuilder.getStringAttr(symbol));
      copy->setAttr("amoeba.host.trace_task_index",
                    traceBuilder.getI64IntegerAttr(ordinal));
      copy->setAttr("amoeba.host.trace_write_index",
                    traceBuilder.getI64IntegerAttr(writeIndex));
      traceBuilder.setInsertionPointAfter(copy);
    }
  }
  task->replaceAllUsesWith(replacements);
  task.erase();
  return success();
}

static LogicalResult eliminateAliases(func::FuncOp function) {
  SmallVector<TaskflowJoinOp> joins;
  SmallVector<TaskflowReadCompletionJoinOp> readJoins;
  SmallVector<TaskflowChannelOp> channels;
  function.walk([&](TaskflowJoinOp join) { joins.push_back(join); });
  function.walk([&](TaskflowReadCompletionJoinOp join) {
    readJoins.push_back(join);
  });
  function.walk(
      [&](TaskflowChannelOp channel) { channels.push_back(channel); });

  // Verification is intentionally done before the first alias rewrite.  A
  // completion join is only an alias after the existing Taskflow verifier has
  // proved exact, disjoint region coverage.
  for (TaskflowJoinOp join : joins)
    if (failed(verify(join.getOperation())))
      return reject(join,
                    "HOST_LOWERING_UNVERIFIED_JOIN: completion join proof "
                    "did not verify");
  for (TaskflowReadCompletionJoinOp join : readJoins)
    if (failed(verify(join.getOperation())))
      return reject(join,
                    "HOST_LOWERING_UNVERIFIED_READ_JOIN: read completion "
                    "proof did not verify");
  for (TaskflowReadCompletionJoinOp join : readJoins) {
    join.getJoined().replaceAllUsesWith(join.getBaseState());
    join.erase();
  }
  for (TaskflowJoinOp join : joins) {
    join.getJoined().replaceAllUsesWith(join.getBase());
    join.erase();
  }
  for (TaskflowChannelOp channel : channels) {
    channel.getTarget().replaceAllUsesWith(channel.getSource());
    channel.erase();
  }
  return success();
}

static LogicalResult validateFunction(func::FuncOp function) {
  SmallVector<TaskflowTaskOp> tasks;
  function.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });
  for (TaskflowTaskOp task : tasks)
    if (failed(validateTask(task)))
      return failure();
  // Verify aliases before lowering.  This also diagnoses malformed joins and
  // channels without relying on a later pass to notice the problem.
  bool valid = true;
  function.walk([&](TaskflowJoinOp join) {
    if (failed(verify(join.getOperation()))) {
      join.emitError() << kErrorPrefix << "HOST_LOWERING_UNVERIFIED_JOIN: "
                       << "completion join proof did not verify";
      valid = false;
    }
  });
  function.walk([&](TaskflowReadCompletionJoinOp join) {
    if (failed(verify(join.getOperation()))) {
      join.emitError() << kErrorPrefix
                       << "HOST_LOWERING_UNVERIFIED_READ_JOIN: "
                       << "read completion proof did not verify";
      valid = false;
    }
  });
  return valid ? success() : failure();
}

static LogicalResult lowerFunction(func::FuncOp function,
                                   bool captureTaskStates) {
  if (failed(validateFunction(function)))
    return failure();
  FailureOr<SmallVector<TaskflowTaskOp>> order = dependencyOrder(function);
  if (failed(order))
    return failure();
  if (failed(eliminateAliases(function)))
    return failure();
  for (auto [ordinal, task] : llvm::enumerate(*order))
    if (failed(lowerTask(task, ordinal, captureTaskStates)))
      return failure();

  bool leftover = false;
  function.walk([&](Operation *operation) {
    if (isTaskflowOperation(operation))
      leftover = true;
  });
  if (leftover)
    return reject(function,
                  "HOST_LOWERING_UNSUPPORTED_OPERATION: Taskflow operation "
                  "remains after lowering");
  return verify(function.getOperation());
}

static LogicalResult attachInput0NumericHarness(ModuleOp module,
                                                func::FuncOp program,
                                                StringRef workload) {
  int64_t tag = llvm::StringSwitch<int64_t>(workload)
                    .Case("gcn", 1).Case("harris", 2).Case("radar", 3)
                    .Case("lu", 4).Case("raytracing", 5).Default(0);
  if (!tag || !program.getSymName().contains((workload + "_func").str()) ||
      program.getNumArguments() < 2 ||
      !program.getArgument(0).getType().isSignlessInteger(32) ||
      program.getNumResults() != 0 || module.lookupSymbol("main"))
    return reject(program, "HOST_NUMERIC_INPUT0_CONTRACT: unsupported program signature");
  auto bound = program->getAttrOfType<IntegerAttr>("amoeba.static_bound.arg.0");
  if (!bound || bound.getInt() <= 0 || bound.getInt() > INT32_MAX)
    return reject(program, "HOST_NUMERIC_INPUT0_CONTRACT: missing positive input-0 scalar bound");
  for (unsigned index = 1; index < program.getNumArguments(); ++index) {
    auto type = dyn_cast<MemRefType>(program.getArgument(index).getType());
    auto shape = program.getArgAttrOfType<DenseI64ArrayAttr>(index,
                                                          "amoeba.logical_transfer_shape");
    if (!type || !type.getElementType().isSignlessInteger(32) ||
        !type.getLayout().isIdentity() || !shape || shape.size() != type.getRank())
      return reject(program, "HOST_NUMERIC_INPUT0_CONTRACT: missing contiguous i32 buffer extent");
    for (auto [dimension, size] : llvm::enumerate(shape.asArrayRef()))
      if (size <= 0 || (!type.isDynamicDim(dimension) &&
                       type.getDimSize(dimension) != size))
        return reject(program, "HOST_NUMERIC_INPUT0_CONTRACT: buffer extent disagrees with its type");
  }
  OpBuilder builder(module.getContext());
  Location loc = program.getLoc();
  Type i64 = builder.getI64Type();
  Type unranked = UnrankedMemRefType::get(builder.getI32Type(), 0);
  auto declare = [&](StringRef name, TypeRange inputs, TypeRange results) {
    auto function = func::FuncOp::create(loc, name,
                                        builder.getFunctionType(inputs, results));
    function.setPrivate();
    function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    module.push_back(function);
    return function;
  };
  for (StringRef name : {"orbit_numeric_initialize", "orbit_numeric_snapshot",
                         "orbit_numeric_check"})
    if (module.lookupSymbol(name))
      return reject(program, "HOST_NUMERIC_INPUT0_CONTRACT: conflicting reference symbol");
  auto initialize = declare("orbit_numeric_initialize", {i64, i64, unranked}, {});
  auto snapshot = declare("orbit_numeric_snapshot", {i64, i64}, {});
  auto check = declare("orbit_numeric_check", {i64}, {i64});
  auto main = func::FuncOp::create(loc, "main", builder.getFunctionType({}, {i64}));
  module.push_back(main);
  builder.setInsertionPointToStart(main.addEntryBlock());
  Value programTag = builder.create<arith::ConstantIntOp>(loc, tag, 64);
  Value scalar = builder.create<arith::ConstantIntOp>(loc, bound.getInt(), 32);
  SmallVector<Value> arguments{scalar}, allocations;
  for (unsigned index = 1; index < program.getNumArguments(); ++index) {
    auto type = cast<MemRefType>(program.getArgument(index).getType());
    auto shape = program.getArgAttrOfType<DenseI64ArrayAttr>(index,
                                                          "amoeba.logical_transfer_shape");
    SmallVector<Value> dynamicSizes;
    for (unsigned dimension = 0; dimension < type.getRank(); ++dimension)
      if (type.isDynamicDim(dimension))
        dynamicSizes.push_back(builder.create<arith::ConstantIndexOp>(
            loc, shape[dimension]));
    Value buffer = builder.create<memref::AllocOp>(loc, type, dynamicSizes);
    allocations.push_back(buffer);
    arguments.push_back(buffer);
    Value unrankedBuffer = builder.create<memref::CastOp>(loc, unranked, buffer);
    Value argumentIndex = builder.create<arith::ConstantIntOp>(loc, index, 64);
    builder.create<func::CallOp>(loc, initialize,
                                ValueRange{programTag, argumentIndex, unrankedBuffer});
  }
  Value scalar64 = builder.create<arith::ExtSIOp>(loc, i64, scalar);
  builder.create<func::CallOp>(loc, snapshot, ValueRange{programTag, scalar64});
  builder.create<func::CallOp>(loc, program, arguments);
  Value mismatches = builder.create<func::CallOp>(loc, check,
                                                ValueRange{programTag}).getResult(0);
  for (Value buffer : allocations)
    builder.create<memref::DeallocOp>(loc, buffer);
  builder.create<func::ReturnOp>(loc, mismatches);
  return verify(module.getOperation());
}

struct LowerJointTaskflowToHostSCFPass
    : public PassWrapper<LowerJointTaskflowToHostSCFPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerJointTaskflowToHostSCFPass)

  LowerJointTaskflowToHostSCFPass() = default;
  LowerJointTaskflowToHostSCFPass(const LowerJointTaskflowToHostSCFPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "lower-joint-taskflow-to-host-scf";
  }
  StringRef getDescription() const override {
    return "Lower executable ORBIT Taskflow graphs to host SCF";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect,
                    func::FuncDialect>();
    registry.insert<neura::NeuraDialect>();
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Only lower this function; empty lowers every Taskflow "
                     "function."),
      llvm::cl::init("")};
  Option<bool> captureTaskStates{
      *this, "capture-task-states",
      llvm::cl::desc("Snapshot each task's written memrefs for host numeric "
                     "auditing"),
      llvm::cl::init(false)};

  Option<std::string> hostHarness{
      *this, "host-harness",
      llvm::cl::desc("Host MLIR numeric harness; import nonconflicting functions "
                     "after lowering, retaining the current program body."),
      llvm::cl::init("")};
  Option<std::string> numericProgram{
      *this, "input0-numeric-program",
      llvm::cl::desc("Build a C++ reference numeric entry point for the selected input-0 program."),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!hostHarness.getValue().empty() && !numericProgram.getValue().empty()) {
      module.emitError("HOST_NUMERIC_HARNESS_INVALID: select one harness source");
      return signalPassFailure();
    }
    SmallVector<func::FuncOp> functions;
    module.walk([&](func::FuncOp function) {
      if (functionName.getValue().empty() ||
          function.getSymName() == functionName.getValue()) {
        bool hasTaskflow = false;
        function.walk([&](Operation *operation) {
          if (isTaskflowOperation(operation))
            hasTaskflow = true;
        });
        if (hasTaskflow)
          functions.push_back(function);
      }
    });
    if (!functionName.getValue().empty() && functions.empty()) {
      module.emitError() << kErrorPrefix << "HOST_LOWERING_FUNCTION_NOT_FOUND: "
                         << "no Taskflow function named "
                         << functionName.getValue();
      return signalPassFailure();
    }
    for (func::FuncOp function : functions)
      if (failed(lowerFunction(function, captureTaskStates)))
        return signalPassFailure();
    if (!numericProgram.getValue().empty()) {
      if (functions.size() != 1 ||
          failed(attachInput0NumericHarness(module, functions.front(),
                                           numericProgram.getValue())))
        return signalPassFailure();
    }
    if (!hostHarness.getValue().empty()) {
      OwningOpRef<ModuleOp> harness = parseSourceFile<ModuleOp>(
          hostHarness.getValue(), module.getContext());
      if (!harness) {
        module.emitError("HOST_NUMERIC_HARNESS_INVALID: cannot parse host harness");
        return signalPassFailure();
      }
      SmallVector<Operation *> imports;
      for (Operation &operation : harness->getBody()->getOperations()) {
        auto function = dyn_cast<func::FuncOp>(operation);
        if (!function) {
          module.emitError("HOST_NUMERIC_HARNESS_INVALID: only function symbols are supported");
          return signalPassFailure();
        }
        if (Operation *existing = SymbolTable::lookupSymbolIn(
                module, function.getSymName())) {
          auto program = dyn_cast<func::FuncOp>(existing);
          if (!program || program.getFunctionType() != function.getFunctionType() ||
              !llvm::is_contained(functions, program)) {
            module.emitError("HOST_NUMERIC_HARNESS_INVALID: conflicting symbol or program signature");
            return signalPassFailure();
          }
          // The fixture's old program body never replaces the selected graph.
          continue;
        }
        imports.push_back(&operation);
      }
      for (Operation *operation : imports)
        module.getBody()->push_back(operation->clone());
      if (failed(verify(module.getOperation())))
        return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createLowerJointTaskflowToHostSCFPass() {
  return std::make_unique<LowerJointTaskflowToHostSCFPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
