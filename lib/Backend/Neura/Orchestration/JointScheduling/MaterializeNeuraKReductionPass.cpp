//===- MaterializeNeuraKReductionPass.cpp ----------------------*- C++ -*-===//
//
// Post-Neura C++ MLIR materialization of the canonical static i32 GEMM K
// reduction.  This pass consumes the mixed Taskflow + Neura IR produced by
// --convert-taskflow-to-neura.  It never invokes the pre-Neura K passes and it
// never represents arithmetic reduction with a completion join.
//
// The accepted source is deliberately narrow: one producer, one elementwise
// consumer, static 2-D i32 tensors, a static M/N counter chain, and one
// scf.for K accumulator in the producer neura.kernel.  Unsupported shapes,
// aliases, dynamic bounds, and already rewritten graphs fail closed.
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

constexpr StringLiteral kSchemaAttr = "amoeba.semantic.schema";
constexpr StringLiteral kKindAttr = "amoeba.semantic.task_kind";
constexpr StringLiteral kRoleAttr = "amoeba.semantic.task_role";
constexpr StringLiteral kKPolicyAttr = "amoeba.semantic.k_policy";
constexpr StringLiteral kTopologyAttr =
    "amoeba.semantic.reduction_topology";
constexpr StringLiteral kKRangeAttr = "amoeba.semantic.K_range";
constexpr StringLiteral kMRangeAttr = "amoeba.semantic.M_range";
constexpr StringLiteral kNRangeAttr = "amoeba.semantic.N_range";
constexpr StringLiteral kInitialCOwnershipAttr =
    "amoeba.semantic.initial_C_ownership";
constexpr StringLiteral kFinalCOwnershipAttr =
    "amoeba.semantic.final_C_ownership";
constexpr StringLiteral kNumericReductionAttr =
    "amoeba.semantic.numeric_reduction";
constexpr StringLiteral kReductionFinalAttr =
    "amoeba.semantic.reduction_final";
constexpr StringLiteral kPrivatePartialAttr =
    "amoeba.semantic.private_partial";
constexpr StringLiteral kRewriteSchemaAttr =
    "amoeba.semantic.rewrite_schema";
constexpr StringLiteral kRewriteSchema = "orbit-joint-semantic-rewrite-v1";
constexpr StringLiteral kIncomingEdgesAttr = "amoeba.semantic.incoming_edges";
constexpr StringLiteral kInputRegionLowersAttr =
    "amoeba.tiling.input_region_lowers";
constexpr StringLiteral kInputRegionUppersAttr =
    "amoeba.tiling.input_region_uppers";
constexpr StringLiteral kInputRegionReasonsAttr =
    "amoeba.tiling.input_region_reasons";
constexpr StringLiteral kOutputRegionLowersAttr =
    "amoeba.tiling.output_region_lowers";
constexpr StringLiteral kOutputRegionUppersAttr =
    "amoeba.tiling.output_region_uppers";

static LogicalResult reject(Operation *operation, const Twine &message) {
  operation->emitError(message);
  return failure();
}

static std::optional<int64_t> constantIndex(Value value) {
  if (!value)
    return std::nullopt;
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return constant.value();
  return std::nullopt;
}

static bool isNoAliasArgument(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return false;
  auto function = dyn_cast_or_null<func::FuncOp>(
      argument.getOwner()->getParentOp());
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
  if (lhsAllocation && isNoAliasArgument(rhs))
    return true;
  if (rhsAllocation && isNoAliasArgument(lhs))
    return true;
  return isNoAliasArgument(lhs) && isNoAliasArgument(rhs);
}

// A Taskflow result is a dataflow token, while original_write_memrefs records
// the backing storage.  For application Task_3, require the producer result
// to resolve to the exact declared storage instead of accepting matching
// types alone. Direct storage values are accepted for small focused fixtures;
// task results must have an unambiguous done-writes/original-write pairing.
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
       llvm::zip(doneWrites, originalWrites)) {
    if (doneWrite == value && originalWrite == storage)
      return true;
  }
  return false;
}

static bool taskBodyValueRepresentsStorage(Value value, TaskflowTaskOp task,
                                           Value storage) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &task.getBody().front())
    return false;
  unsigned index = argument.getArgNumber();
  if (index < task.getWillReads().size())
    return representsOriginalStorage(task.getWillReads()[index], storage) ||
           (index < task.getOriginalReadMemrefs().size() &&
            task.getOriginalReadMemrefs()[index] == storage);
  index -= task.getWillReads().size();
  if (index < task.getWillWrites().size())
    return representsOriginalStorage(task.getWillWrites()[index], storage);
  return false;
}

static bool isStructuralAttribute(StringRef name) {
  return name == "task_name" || name == "operandSegmentSizes" ||
         name == "resultSegmentSizes" || name == "operand_segment_sizes" ||
         name == "result_segment_sizes";
}

static StringRef taskKind(TaskflowTaskOp task) {
  if (auto attr = task->getAttrOfType<StringAttr>(kKindAttr))
    return attr.getValue();
  return task.getTaskName().starts_with("consumer")
             ? StringRef("elementwise_consumer")
             : StringRef("gemm_producer");
}

static StringRef taskRole(TaskflowTaskOp task) {
  if (auto attr = task->getAttrOfType<StringAttr>(kRoleAttr))
    return attr.getValue();
  return taskKind(task) == "elementwise_consumer" ? StringRef("consumer")
                                                  : StringRef("producer");
}

static void copyTaskAttributes(TaskflowTaskOp source, TaskflowTaskOp target,
                               StringRef name) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef attrName = attr.getName().strref();
    if (isStructuralAttribute(attrName))
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
  target->setAttr("task_name", StringAttr::get(target.getContext(), name));
}

struct StaticRegion {
  SmallVector<int64_t> lower;
  SmallVector<int64_t> upper;
};

static StaticRegion fullRegion(int64_t M, int64_t N) {
  return {{0, 0}, {M, N}};
}

static void setRegionContracts(OpBuilder &builder, TaskflowTaskOp task,
                               ArrayRef<StaticRegion> inputRegions,
                               ArrayRef<StaticRegion> outputRegions,
                               StringRef reason) {
  SmallVector<Attribute> lowers;
  SmallVector<Attribute> uppers;
  SmallVector<Attribute> reasons;
  for (const StaticRegion &region : inputRegions) {
    lowers.push_back(builder.getDenseI64ArrayAttr(region.lower));
    uppers.push_back(builder.getDenseI64ArrayAttr(region.upper));
    reasons.push_back(builder.getStringAttr(reason));
  }
  task->setAttr(kInputRegionLowersAttr, builder.getArrayAttr(lowers));
  task->setAttr(kInputRegionUppersAttr, builder.getArrayAttr(uppers));
  task->setAttr(kInputRegionReasonsAttr, builder.getArrayAttr(reasons));
  lowers.clear();
  uppers.clear();
  for (const StaticRegion &region : outputRegions) {
    lowers.push_back(builder.getDenseI64ArrayAttr(region.lower));
    uppers.push_back(builder.getDenseI64ArrayAttr(region.upper));
  }
  task->setAttr(kOutputRegionLowersAttr, builder.getArrayAttr(lowers));
  task->setAttr(kOutputRegionUppersAttr, builder.getArrayAttr(uppers));
}

static void setIncomingEdges(
    OpBuilder &builder, TaskflowTaskOp task,
    ArrayRef<std::tuple<StringRef, StringRef, StringRef>> edges) {
  SmallVector<Attribute> attrs;
  for (auto [source, role, scope] : edges)
    attrs.push_back(builder.getStringAttr(
        (Twine(source) + "|" + role + "|" + scope).str()));
  task->setAttr(kIncomingEdgesAttr, builder.getArrayAttr(attrs));
}

static void markTask(OpBuilder &builder, TaskflowTaskOp task, StringRef kind,
                     StringRef role, StringRef policy, StringRef topology) {
  task->setAttr(kSchemaAttr,
                builder.getStringAttr("orbit-joint-taskflow-materialization-v1"));
  task->setAttr(kKindAttr, builder.getStringAttr(kind));
  task->setAttr(kRoleAttr, builder.getStringAttr(role));
  task->setAttr(kKPolicyAttr, builder.getStringAttr(policy));
  task->setAttr(kTopologyAttr, builder.getStringAttr(topology));
  task->setAttr(kRewriteSchemaAttr, builder.getStringAttr(kRewriteSchema));
}

static void setRangeAttrs(OpBuilder &builder, TaskflowTaskOp task,
                          ArrayRef<int64_t> mRange, ArrayRef<int64_t> nRange,
                          ArrayRef<int64_t> kRange) {
  task->setAttr(kMRangeAttr, builder.getDenseI64ArrayAttr(mRange));
  task->setAttr(kNRangeAttr, builder.getDenseI64ArrayAttr(nRange));
  task->setAttr(kKRangeAttr, builder.getDenseI64ArrayAttr(kRange));
}

static void addTaskYield(Block *body, Value output) {
  OpBuilder builder = OpBuilder::atBlockEnd(body);
  builder.create<TaskflowYieldOp>(output.getLoc(), ValueRange{},
                                  ValueRange{output}, ValueRange{});
}

static void createCounterConstants(OpBuilder &builder, Location loc,
                                   int64_t lower, int64_t upper, Value &lb,
                                   Value &ub, Value &step) {
  lb = builder.create<arith::ConstantIndexOp>(loc, lower);
  ub = builder.create<arith::ConstantIndexOp>(loc, upper);
  step = builder.create<arith::ConstantIndexOp>(loc, 1);
}

static SmallVector<TaskflowCounterOp>
createTaskflowCounters(OpBuilder &builder, Location loc, int64_t M, int64_t N) {
  Value mLower, mUpper, mStep;
  Value nLower, nUpper, nStep;
  createCounterConstants(builder, loc, 0, M, mLower, mUpper, mStep);
  auto m = builder.create<TaskflowCounterOp>(
      loc, builder.getIndexType(), Value{}, mLower, mUpper, mStep, StringAttr{},
      StringAttr{}, IntegerAttr{});
  createCounterConstants(builder, loc, 0, N, nLower, nUpper, nStep);
  auto n = builder.create<TaskflowCounterOp>(
      loc, builder.getIndexType(), m.getCounterIndex(), nLower, nUpper, nStep,
      StringAttr{}, StringAttr{}, IntegerAttr{});
  return {m, n};
}

static std::optional<unsigned> findInput(TaskflowTaskOp task, Value value) {
  for (auto indexed : llvm::enumerate(task.getWillReads()))
    if (indexed.value() == value)
      return indexed.index();
  return std::nullopt;
}

static BlockArgument kernelMemrefArgument(neura::KernelOp kernel,
                                          Value taskArgument) {
  for (auto [index, input] : llvm::enumerate(kernel.getInputs()))
    if (input == taskArgument && index < kernel.getBody().front().getNumArguments())
      return kernel.getBody().front().getArgument(index);
  return BlockArgument();
}

static bool equivalentIndex(Value lhs, Value rhs,
                            ArrayRef<neura::CounterOp> counters) {
  if (lhs == rhs)
    return true;
  for (neura::CounterOp counter : counters)
    if (lhs == counter.getCurrentIndex() && rhs == counter.getCurrentIndex())
      return true;
  auto lhsConstant = constantIndex(lhs);
  auto rhsConstant = constantIndex(rhs);
  return lhsConstant && rhsConstant && *lhsConstant == *rhsConstant;
}

static bool equivalentIndices(ValueRange lhs, ValueRange rhs,
                              ArrayRef<neura::CounterOp> counters) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (!equivalentIndex(left, right, counters))
      return false;
  return true;
}

struct KSource {
  func::FuncOp function;
  TaskflowTaskOp producer;
  TaskflowTaskOp consumer;
  neura::KernelOp producerKernel;
  Value A;
  Value B;
  Value C;
  Value T;
  Value E;
  Value D;
  BlockArgument taskA;
  BlockArgument taskB;
  BlockArgument taskC;
  BlockArgument taskT;
  BlockArgument kernelA;
  BlockArgument kernelB;
  BlockArgument kernelC;
  BlockArgument kernelT;
  SmallVector<TaskflowCounterOp> taskCounters;
  SmallVector<neura::CounterOp> kernelCounters;
  scf::ForOp kLoop;
  memref::LoadOp initialLoad;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  MemRefType tensorType;
};

static FailureOr<func::FuncOp> selectFunction(ModuleOp module,
                                              StringRef functionName) {
  SmallVector<func::FuncOp> matches;
  module.walk([&](func::FuncOp function) {
    bool hasTask = false;
    function.walk([&](TaskflowTaskOp) { hasTask = true; });
    if (hasTask && (functionName.empty() || function.getSymName() == functionName))
      matches.push_back(function);
  });
  if (matches.size() != 1) {
    module.emitError() << "post-Neura K reduction requires exactly one "
                          "Taskflow function (found "
                       << matches.size() << ")";
    return failure();
  }
  return matches.front();
}

static FailureOr<KSource> analyzeSource(ModuleOp module, StringRef functionName) {
  FailureOr<func::FuncOp> function = selectFunction(module, functionName);
  if (failed(function))
    return failure();
  SmallVector<TaskflowTaskOp> producers;
  SmallVector<TaskflowTaskOp> consumers;
  (*function).walk([&](TaskflowTaskOp task) {
    if (taskRole(task) == "producer" && !task->hasAttr(kPrivatePartialAttr))
      producers.push_back(task);
    if (taskRole(task) == "consumer")
      consumers.push_back(task);
  });
  if (producers.size() != 1 || consumers.size() != 1) {
    (*function).emitError()
        << "post-Neura K reduction requires one producer and one consumer "
           "(found "
        << producers.size() << " producer(s), " << consumers.size()
        << " consumer(s))";
    return failure();
  }
  TaskflowTaskOp producer = producers.front();
  TaskflowTaskOp consumer = consumers.front();
  if (producer->getBlock() != consumer->getBlock() ||
      !producer->isBeforeInBlock(consumer))
    return reject(consumer, "post-Neura K reduction requires ordered producer "
                           "and consumer tasks");
  if (!producer.getBody().hasOneBlock() || !consumer.getBody().hasOneBlock())
    return reject(producer, "post-Neura K reduction requires single-block tasks");
  if (producer.getWillReads().size() != 3 ||
      producer.getWillWrites().size() != 1 ||
      producer.getDoneReads().size() != 0 ||
      producer.getDoneWrites().size() != 1 ||
      producer.getValueInputs().size() != 0 ||
      producer.getOriginalReadMemrefs().size() != 3 ||
      producer.getOriginalWriteMemrefs().size() != 1)
    return reject(producer, "post-Neura K reduction expects A/B/C and one T state");
  if (consumer.getWillReads().size() != 2 ||
      consumer.getWillWrites().size() != 1 ||
      consumer.getDoneReads().size() != 0 ||
      consumer.getDoneWrites().size() != 1 ||
      consumer.getValueInputs().size() != 0 ||
      consumer.getOriginalReadMemrefs().size() != 2 ||
      consumer.getOriginalWriteMemrefs().size() != 1)
    return reject(consumer, "post-Neura K reduction expects T/E and one D state");
  if (producer.getDoneWrites().front().getUses().begin() ==
      producer.getDoneWrites().front().getUses().end())
    return reject(producer, "producer T state has no consumer use");
  Value produced = producer.getDoneWrites().front();
  auto consumerIndex = findInput(consumer, produced);
  if (!consumerIndex || !produced.hasOneUse())
    return reject(consumer, "producer T must have one direct consumer use");
  if (consumer.getWillReads()[*consumerIndex].getDefiningOp<TaskflowChannelOp>())
    return reject(consumer, "post-Neura K reduction rejects channel-wrapped T state");

  KSource result;
  result.function = *function;
  result.producer = producer;
  result.consumer = consumer;
  result.A = producer.getWillReads()[0];
  result.B = producer.getWillReads()[1];
  result.C = producer.getWillReads()[2];
  result.T = producer.getWillWrites().front();
  result.E = consumer.getWillReads()[*consumerIndex == 0 ? 1 : 0];
  result.D = consumer.getWillWrites().front();
  result.tensorType = dyn_cast<MemRefType>(result.T.getType());
  if (!result.tensorType || !result.tensorType.hasStaticShape() ||
      !result.tensorType.getLayout().isIdentity() ||
      result.tensorType.getRank() != 2 ||
      !result.tensorType.getElementType().isInteger(32))
    return reject(producer, "post-Neura K reduction requires static 2-D i32 T");
  for (Value value : {result.A, result.B, result.C, result.T, result.E,
                      result.D}) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type != result.tensorType)
      return reject(producer,
                    "post-Neura K reduction requires matching static tensor types");
  }
  SmallVector<Value> storage{result.A, result.B, result.C, result.T, result.E,
                              result.D};
  for (size_t i = 0; i < storage.size(); ++i)
    for (size_t j = i + 1; j < storage.size(); ++j)
      if (!provesDistinctStorage(storage[i], storage[j]))
        return reject(producer,
                      "post-Neura K reduction requires distinct noalias or "
                      "allocation roots");

  Block &producerBody = producer.getBody().front();
  if (producerBody.getNumArguments() != 4)
    return reject(producer, "post-Neura K reduction expects four producer body arguments");
  result.taskA = producerBody.getArgument(0);
  result.taskB = producerBody.getArgument(1);
  result.taskC = producerBody.getArgument(2);
  result.taskT = producerBody.getArgument(3);
  result.taskCounters.clear();
  for (Operation &operation : producerBody)
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
      result.taskCounters.push_back(counter);
  if (result.taskCounters.size() != 2)
    return reject(producer, "post-Neura K reduction requires two outer M/N counters");
  for (auto [index, counter] : llvm::enumerate(result.taskCounters)) {
    auto lower = constantIndex(counter.getLowerBound());
    auto upper = constantIndex(counter.getUpperBound());
    auto step = constantIndex(counter.getStep());
    if (!lower || !upper || !step || *lower != 0 || *upper <= *lower ||
        *step != 1 || (index == 0 ? static_cast<bool>(counter.getParentIndex())
                                  : !counter.getParentIndex()))
      return reject(counter, "post-Neura K reduction requires static unit-step M/N counters");
    if (index == 0)
      result.M = *upper;
    else
      result.N = *upper;
  }

  for (Operation &operation : producerBody) {
    if (isa<TaskflowCounterOp, arith::ConstantOp, TaskflowYieldOp>(&operation))
      continue;
    if (auto kernel = dyn_cast<neura::KernelOp>(&operation)) {
      if (result.producerKernel)
        return reject(producer, "post-Neura K reduction requires one neura.kernel");
      result.producerKernel = kernel;
      continue;
    }
    return reject(&operation, "post-Neura K reduction found unsupported producer shell operation");
  }
  if (!result.producerKernel || !result.producerKernel.getBody().hasOneBlock() ||
      !result.producerKernel.getIterArgsInit().empty() ||
      !result.producerKernel.getOutputs().empty())
    return reject(producer, "post-Neura K reduction requires one stateless kernel");
  result.kernelA = kernelMemrefArgument(result.producerKernel, result.taskA);
  result.kernelB = kernelMemrefArgument(result.producerKernel, result.taskB);
  result.kernelC = kernelMemrefArgument(result.producerKernel, result.taskC);
  result.kernelT = kernelMemrefArgument(result.producerKernel, result.taskT);
  if (!result.kernelA || !result.kernelB || !result.kernelC || !result.kernelT)
    return reject(result.producerKernel,
                  "post-Neura K reduction requires A/B/C/T kernel inputs");

  for (Operation &operation : result.producerKernel.getBody().front())
    if (auto counter = dyn_cast<neura::CounterOp>(&operation))
      result.kernelCounters.push_back(counter);
  if (result.kernelCounters.size() != 2)
    return reject(result.producerKernel,
                  "post-Neura K reduction requires two Neura M/N counters");
  for (auto [index, counter] : llvm::enumerate(result.kernelCounters)) {
    auto lower = constantIndex(counter.getLowerBound());
    auto upper = constantIndex(counter.getUpperBound());
    auto step = constantIndex(counter.getStep());
    auto outerLower = constantIndex(result.taskCounters[index].getLowerBound());
    auto outerUpper = constantIndex(result.taskCounters[index].getUpperBound());
    auto outerStep = constantIndex(result.taskCounters[index].getStep());
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    if (!lower || !upper || !step || !outerLower || !outerUpper || !outerStep ||
        !id || id.getInt() != static_cast<int64_t>(index) ||
        *lower != *outerLower || *upper != *outerUpper || *step != *outerStep ||
        *step != 1)
      return reject(counter, "post-Neura K reduction requires matching static Neura counters");
  }

  SmallVector<scf::ForOp> loops;
  result.producerKernel.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  if (loops.size() != 1)
    return reject(result.producerKernel,
                  "post-Neura K reduction requires one K scf.for");
  result.kLoop = loops.front();
  auto kLower = constantIndex(result.kLoop.getLowerBound());
  auto kUpper = constantIndex(result.kLoop.getUpperBound());
  auto kStep = constantIndex(result.kLoop.getStep());
  if (!kLower || !kUpper || !kStep || *kLower != 0 || *kUpper <= *kLower ||
      *kStep != 1 || result.kLoop.getNumRegionIterArgs() != 1 ||
      !result.kLoop.getResult(0).getType().isInteger(32))
    return reject(result.kLoop,
                  "post-Neura K reduction requires one static i32 accumulator");
  result.K = *kUpper;

  SmallVector<memref::LoadOp> cLoads;
  SmallVector<memref::LoadOp> aLoads;
  SmallVector<memref::LoadOp> bLoads;
  SmallVector<memref::StoreOp> tStores;
  result.producerKernel.walk([&](memref::LoadOp load) {
    if (load.getMemRef() == result.kernelC)
      cLoads.push_back(load);
    if (load.getMemRef() == result.kernelA)
      aLoads.push_back(load);
    if (load.getMemRef() == result.kernelB)
      bLoads.push_back(load);
  });
  result.producerKernel.walk([&](memref::StoreOp store) {
    if (store.getMemRef() == result.kernelT)
      tStores.push_back(store);
  });
  if (cLoads.size() != 1 || aLoads.size() != 1 || bLoads.size() != 1 ||
      tStores.size() != 1 || cLoads.front().getResult() !=
          result.kLoop.getInitArgs().front())
    return reject(result.producerKernel,
                  "post-Neura K reduction requires canonical C/A/B loads and T store");
  result.initialLoad = cLoads.front();
  if (!equivalentIndices(result.initialLoad.getIndices(),
                         {result.kernelCounters[0].getCurrentIndex(),
                          result.kernelCounters[1].getCurrentIndex()},
                         result.kernelCounters))
    return reject(result.initialLoad,
                  "initial C load must use the Neura M/N counters");
  for (OpOperand &use : result.kernelC.getUses())
    if (use.getOwner() != result.initialLoad.getOperation())
      return reject(result.producerKernel,
                    "C may only feed the canonical initial accumulator load");
  if (producer->hasAttr("amoeba.semantic.k_tiled") ||
      producer->hasAttr("amoeba.semantic.parallel_partials_pending"))
    return reject(producer, "post-Neura K reduction refuses an already tiled producer");
  return result;
}

static LogicalResult updateKLoopBounds(neura::KernelOp kernel, int64_t lower,
                                       int64_t upper) {
  SmallVector<scf::ForOp> loops;
  kernel.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  if (loops.size() != 1)
    return reject(kernel, "K clone must contain one scf.for");
  OpBuilder builder(kernel.getContext());
  builder.setInsertionPointToStart(&kernel.getBody().front());
  Value lowerValue = builder.create<arith::ConstantIndexOp>(kernel.getLoc(), lower);
  Value upperValue = builder.create<arith::ConstantIndexOp>(kernel.getLoc(), upper);
  Value stepValue = builder.create<arith::ConstantIndexOp>(kernel.getLoc(), 1);
  loops.front().getLowerBoundMutable().assign(lowerValue);
  loops.front().getUpperBoundMutable().assign(upperValue);
  loops.front().getStepMutable().assign(stepValue);
  return success();
}

static FailureOr<neura::KernelOp>
clonePartialKernel(OpBuilder &builder, KSource &source,
                   IRMapping &outerMapping, int64_t kLower, int64_t kUpper) {
  neura::KernelOp oldKernel = source.producerKernel;
  memref::LoadOp initialLoad = source.initialLoad;
  SmallVector<Value> inputs;
  SmallVector<unsigned> keptInputIndices;
  for (auto [index, input] : llvm::enumerate(oldKernel.getInputs())) {
    if (input == source.taskC)
      continue;
    Value mapped = outerMapping.lookupOrDefault(input);
    if (!mapped)
      return reject(oldKernel, "partial kernel input has no cloned task argument");
    inputs.push_back(mapped);
    keptInputIndices.push_back(index);
  }
  auto kernel = builder.create<neura::KernelOp>(
      oldKernel.getLoc(), oldKernel.getResultTypes(), inputs, ValueRange{},
      oldKernel.getCgraIdAttr(), oldKernel.getKernelNameAttr(),
      oldKernel.getAcceleratorAttr(), oldKernel.getKernelMetadataAttr());
  Block *newBlock = new Block();
  kernel.getBody().push_back(newBlock);
  IRMapping mapping;
  for (unsigned index : keptInputIndices) {
    BlockArgument newArg = newBlock->addArgument(
        oldKernel.getBody().front().getArgument(index).getType(),
        oldKernel.getLoc());
    mapping.map(oldKernel.getBody().front().getArgument(index), newArg);
  }
  OpBuilder blockBuilder = OpBuilder::atBlockBegin(newBlock);
  Value zero = blockBuilder.create<arith::ConstantIntOp>(oldKernel.getLoc(), 0, 32);
  SmallVector<scf::ForOp> clonedLoops;
  OpBuilder cloneBuilder = OpBuilder::atBlockEnd(newBlock);
  for (Operation &operation : oldKernel.getBody().front()) {
    if (isa<neura::YieldOp>(&operation))
      continue;
    if (&operation == initialLoad.getOperation()) {
      mapping.map(initialLoad.getResult(), zero);
      continue;
    }
    Operation *cloned = cloneBuilder.clone(operation, mapping);
    if (auto loop = dyn_cast<scf::ForOp>(cloned))
      clonedLoops.push_back(loop);
  }
  if (clonedLoops.size() != 1)
    return reject(kernel, "partial kernel clone must contain one K loop");
  cloneBuilder.setInsertionPointToEnd(newBlock);
  cloneBuilder.create<neura::YieldOp>(oldKernel.getLoc(), ValueRange{},
                                      ValueRange{});
  if (failed(updateKLoopBounds(kernel, kLower, kUpper)))
    return failure();
  return kernel;
}

static FailureOr<TaskflowTaskOp>
createProducerClone(OpBuilder &builder, KSource &source, Value carry,
                  Value output, ArrayRef<Value> roots, StringRef name,
                  int64_t kLower, int64_t kUpper, unsigned blockIndex,
                  unsigned blockCount, StringRef policy, bool partial,
                  StringRef topology) {
  SmallVector<Value> reads;
  if (partial)
    reads = {source.A, source.B};
  else
    reads = {source.A, source.B, carry};
  TaskflowTaskOp task = builder.create<TaskflowTaskOp>(
      source.producer.getLoc(), TypeRange{}, TypeRange{output.getType()},
      TypeRange{}, reads, ValueRange{output}, ValueRange{},
      builder.getStringAttr(name), roots, ValueRange{output});
  Block *body = new Block();
  task.getBody().push_back(body);
  for (Value read : reads)
    body->addArgument(read.getType(), source.producer.getLoc());
  body->addArgument(output.getType(), source.producer.getLoc());

  Block &sourceBody = source.producer.getBody().front();
  IRMapping outerMapping;
  outerMapping.map(source.taskA, body->getArgument(0));
  outerMapping.map(source.taskB, body->getArgument(1));
  if (!partial)
    outerMapping.map(source.taskC, body->getArgument(2));
  outerMapping.map(source.taskT, body->getArgument(partial ? 2 : 3));
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  neura::KernelOp clonedKernel;
  for (Operation &operation : sourceBody) {
    if (isa<TaskflowYieldOp>(&operation))
      continue;
    if (auto kernel = dyn_cast<neura::KernelOp>(&operation)) {
      if (partial) {
        FailureOr<neura::KernelOp> partialKernel =
            clonePartialKernel(bodyBuilder, source, outerMapping, kLower, kUpper);
        if (failed(partialKernel))
          return failure();
        clonedKernel = *partialKernel;
      } else {
        Operation *cloned = bodyBuilder.clone(*kernel, outerMapping);
        clonedKernel = dyn_cast<neura::KernelOp>(cloned);
        if (!clonedKernel)
          return reject(task, "producer clone lost its neura.kernel");
      }
      continue;
    }
    bodyBuilder.clone(operation, outerMapping);
  }
  if (!clonedKernel)
    return reject(task, "producer clone has no neura.kernel");
  if (!partial && failed(updateKLoopBounds(clonedKernel, kLower, kUpper)))
    return failure();
  addTaskYield(body, body->getArgument(partial ? 2 : 3));

  copyTaskAttributes(source.producer, task, name);
  markTask(builder, task, "gemm_producer", "producer", policy, topology);
  task->setAttr("amoeba.semantic.k_tiled", builder.getUnitAttr());
  task->setAttr("amoeba.semantic.k_block_index",
                builder.getI64IntegerAttr(blockIndex));
  task->setAttr("amoeba.semantic.k_block_count",
                builder.getI64IntegerAttr(blockCount));
  task->setAttr("amoeba.semantic.output_visibility",
                builder.getStringAttr(partial ? "private_partial"
                                               : "global_materialized"));
  if (!partial) {
    task->setAttr(kInitialCOwnershipAttr,
                  builder.getStringAttr(blockIndex == 0 ? "initial_C_once"
                                                        : "state_carry"));
    task->setAttr(kFinalCOwnershipAttr,
                  builder.getStringAttr(blockIndex + 1 == blockCount
                                            ? "external_final_C"
                                            : "state_carry"));
  } else {
    task->setAttr(kPrivatePartialAttr, builder.getUnitAttr());
    task->setAttr("amoeba.semantic.partial_buffer",
                  builder.getStringAttr(name));
    task->setAttr("amoeba.semantic.logical_k_block",
                  builder.getI64IntegerAttr(blockIndex));
  }
  setRangeAttrs(builder, task, {0, source.M}, {0, source.N},
                {kLower, kUpper});
  SmallVector<StaticRegion> inputRegions;
  inputRegions.push_back({{0, kLower}, {source.M, kUpper}});
  inputRegions.push_back({{kLower, 0}, {kUpper, source.N}});
  if (!partial)
    inputRegions.push_back(fullRegion(source.M, source.N));
  setRegionContracts(builder, task, inputRegions,
                     {fullRegion(source.M, source.N)},
                     partial ? "private_partial" : "semantic_k_tile");
  return task;
}

static FailureOr<TaskflowTaskOp>
createReductionTask(OpBuilder &builder, KSource &source,
                  ArrayRef<Value> reads, ArrayRef<Value> roots, Value output,
                  StringRef name, StringRef topology, bool finalReduction,
                  ArrayRef<std::tuple<StringRef, StringRef, StringRef>> edges) {
  if (reads.empty() || reads.size() != roots.size())
    return reject(source.producer,
                  "numeric reduction reads and roots must be non-empty and aligned");
  TaskflowTaskOp task = builder.create<TaskflowTaskOp>(
      source.producer.getLoc(), TypeRange{}, TypeRange{output.getType()},
      TypeRange{}, reads, ValueRange{output}, ValueRange{},
      builder.getStringAttr(name), roots, ValueRange{output});
  Block *body = new Block();
  task.getBody().push_back(body);
  for (Value read : reads)
    body->addArgument(read.getType(), source.producer.getLoc());
  BlockArgument outputArg = body->addArgument(output.getType(), source.producer.getLoc());
  OpBuilder tb = OpBuilder::atBlockEnd(body);
  createTaskflowCounters(tb, source.producer.getLoc(), source.M, source.N);
  SmallVector<Value> kernelInputs;
  for (unsigned index = 0; index < reads.size(); ++index)
    kernelInputs.push_back(body->getArgument(index));
  // The host lowering contract requires every stateless kernel block argument
  // to correspond to a kernel input.  Numeric reduction kernels write their
  // output directly, so pass the Taskflow write memref as the final input
  // rather than creating an unmatched block argument.
  kernelInputs.push_back(outputArg);
  auto kernel = tb.create<neura::KernelOp>(
      source.producer.getLoc(), TypeRange{}, kernelInputs, ValueRange{},
      IntegerAttr{}, StringAttr{}, StringAttr{}, DictionaryAttr{});
  Block *kernelBody = new Block();
  kernel.getBody().push_back(kernelBody);
  SmallVector<BlockArgument> kernelArgs;
  for (Value input : kernelInputs)
    kernelArgs.push_back(kernelBody->addArgument(input.getType(), kernel.getLoc()));
  OpBuilder kb = OpBuilder::atBlockBegin(kernelBody);
  Value mLower, mUpper, mStep, nLower, nUpper, nStep;
  createCounterConstants(kb, kernel.getLoc(), 0, source.M, mLower, mUpper, mStep);
  auto m = kb.create<neura::CounterOp>(
      kernel.getLoc(), kb.getIndexType(), mLower, mUpper, mStep,
      kb.getStringAttr("root"), kb.getStringAttr("constant_bound"),
      kb.getI32IntegerAttr(0));
  createCounterConstants(kb, kernel.getLoc(), 0, source.N, nLower, nUpper, nStep);
  auto n = kb.create<neura::CounterOp>(
      kernel.getLoc(), kb.getIndexType(), nLower, nUpper, nStep,
      kb.getStringAttr("leaf"), kb.getStringAttr("constant_bound"),
      kb.getI32IntegerAttr(1));
  Value accumulator;
  for (unsigned index = 0; index < reads.size(); ++index) {
    Value loaded = kb.create<memref::LoadOp>(
        source.producer.getLoc(), kernelArgs[index],
        ValueRange{m.getCurrentIndex(), n.getCurrentIndex()});
    if (!accumulator)
      accumulator = loaded;
    else
      accumulator = kb.create<arith::AddIOp>(source.producer.getLoc(),
                                             accumulator, loaded);
  }
  kb.create<memref::StoreOp>(source.producer.getLoc(), accumulator,
                             kernelArgs.back(),
                             ValueRange{m.getCurrentIndex(), n.getCurrentIndex()});
  kb.setInsertionPointToEnd(kernelBody);
  kb.create<neura::YieldOp>(source.producer.getLoc(), ValueRange{}, ValueRange{});
  addTaskYield(body, outputArg);

  markTask(builder, task, "reduction", "reduction", "parallel", topology);
  task->setAttr(kNumericReductionAttr, builder.getUnitAttr());
  if (finalReduction)
    task->setAttr(kReductionFinalAttr, builder.getUnitAttr());
  else
    task->setAttr(kPrivatePartialAttr, builder.getUnitAttr());
  task->setAttr("amoeba.semantic.output_visibility",
                builder.getStringAttr("global_materialized"));
  setRangeAttrs(builder, task, {0, source.M}, {0, source.N}, {0, source.K});
  SmallVector<StaticRegion> regions(reads.size(), fullRegion(source.M, source.N));
  setRegionContracts(builder, task, regions, {fullRegion(source.M, source.N)},
                     "numeric_reduction");
  setIncomingEdges(builder, task, edges);
  return task;
}

static void markConsumer(OpBuilder &builder, TaskflowTaskOp consumer,
                         const KSource &source, StringRef policy,
                         StringRef topology, StringRef inputSource) {
  markTask(builder, consumer, "elementwise_consumer", "consumer", policy,
           topology);
  setRangeAttrs(builder, consumer, {0, source.M}, {0, source.N},
                {0, source.K});
  setRegionContracts(builder, consumer,
                     {fullRegion(source.M, source.N), fullRegion(source.M, source.N)},
                     {fullRegion(source.M, source.N)}, "consumer_tile");
  setIncomingEdges(builder, consumer, {{inputSource, "producer_consumer", "tile_local"}});
}

static FailureOr<int64_t> resolveWidth(int64_t K, int64_t tileSize,
                                       int64_t factor, bool sequential) {
  int64_t width = tileSize;
  if (width == 0 && factor != 0) {
    if (factor < 1)
      return failure();
    width = (K + factor - 1) / factor;
  }
  if (width == 0)
    width = std::max<int64_t>(1, K / 2);
  if (width < 1 || width > K || (sequential && width == K))
    return failure();
  return width;
}

static LogicalResult runSequential(ModuleOp module, StringRef functionName,
                                   int64_t tileSize, int64_t factor) {
  FailureOr<KSource> source = analyzeSource(module, functionName);
  if (failed(source))
    return failure();
  FailureOr<int64_t> width = resolveWidth(source->K, tileSize, factor, true);
  if (failed(width))
    return reject(source->producer,
                  "post-Neura sequential K tiling requires width in [1,K)");
  unsigned blockCount = static_cast<unsigned>((source->K + *width - 1) / *width);
  OpBuilder builder(source->producer);
  Value carry = source->C;
  SmallVector<TaskflowTaskOp> blocks;
  for (unsigned index = 0; index < blockCount; ++index) {
    int64_t lower = index * *width;
    int64_t upper = std::min<int64_t>(source->K, lower + *width);
    SmallVector<Value> roots{source->A, source->B,
                             index == 0 ? source->C : source->T};
    std::string name =
        (Twine(source->producer.getTaskName()) + ".k." + Twine(index)).str();
    FailureOr<TaskflowTaskOp> block = createProducerClone(
        builder, *source, carry, source->T, roots, name, lower, upper, index,
        blockCount, "sequential", false, "none");
    if (failed(block))
      return failure();
    if (index != 0)
      setIncomingEdges(builder, *block,
                       {{blocks.back().getTaskName(), "state_carry", "tile_local"}});
    else
      setIncomingEdges(builder, *block, {});
    blocks.push_back(*block);
    carry = block->getDoneWrites().front();
  }
  auto input = findInput(source->consumer, source->producer.getDoneWrites().front());
  if (!input)
    return reject(source->consumer, "consumer has no T input to rewire");
  source->consumer.getWillReadsMutable().slice(*input, 1).assign(carry);
  OpBuilder consumerBuilder(source->consumer);
  markConsumer(consumerBuilder, source->consumer, *source, "sequential", "none",
               blocks.back().getTaskName());
  source->function->setAttr(kKPolicyAttr, consumerBuilder.getStringAttr("sequential"));
  source->function->setAttr(kTopologyAttr, consumerBuilder.getStringAttr("none"));
  source->function->setAttr(kRewriteSchemaAttr,
                             consumerBuilder.getStringAttr(kRewriteSchema));
  source->producer.erase();
  return success();
}

static LogicalResult runParallel(ModuleOp module, StringRef functionName,
                                 int64_t tileSize, int64_t factor,
                                 StringRef topology) {
  FailureOr<KSource> source = analyzeSource(module, functionName);
  if (failed(source))
    return failure();
  FailureOr<int64_t> width = resolveWidth(source->K, tileSize, factor, false);
  if (failed(width))
    return reject(source->producer,
                  "post-Neura parallel K tiling requires width in [1,K]");
  unsigned blockCount = static_cast<unsigned>((source->K + *width - 1) / *width);
  if (blockCount < 2)
    return reject(source->producer,
                  "post-Neura parallel K tiling requires at least two K blocks");
  OpBuilder builder(source->producer);
  SmallVector<Value> states;
  SmallVector<Value> roots;
  SmallVector<TaskflowTaskOp> partialTasks;
  for (unsigned index = 0; index < blockCount; ++index) {
    int64_t lower = index * *width;
    int64_t upper = std::min<int64_t>(source->K, lower + *width);
    auto alloc = builder.create<memref::AllocOp>(source->producer.getLoc(),
                                                 source->tensorType);
    alloc->setAttr(kPrivatePartialAttr, builder.getUnitAttr());
    alloc->setAttr("amoeba.semantic.logical_k_block",
                   builder.getI64IntegerAttr(index));
    std::string name = (Twine("partial_k.") + Twine(index)).str();
    FailureOr<TaskflowTaskOp> partial = createProducerClone(
        builder, *source, Value{}, alloc.getMemref(),
        {source->A, source->B}, name, lower, upper, index, blockCount,
        "parallel", true, topology);
    if (failed(partial))
      return failure();
    setIncomingEdges(builder, *partial, {});
    states.push_back(partial->getDoneWrites().front());
    roots.push_back(alloc.getMemref());
    partialTasks.push_back(*partial);
  }

  TaskflowTaskOp finalReduction;
  if (topology == "linear") {
    SmallVector<Value> reads{source->C};
    SmallVector<Value> reductionRoots{source->C};
    reads.append(states.begin(), states.end());
    reductionRoots.append(roots.begin(), roots.end());
    SmallVector<std::tuple<StringRef, StringRef, StringRef>> edges;
    for (TaskflowTaskOp task : partialTasks)
      edges.push_back({task.getTaskName(), "partial_reduction", "tile_local"});
    FailureOr<TaskflowTaskOp> reduction = createReductionTask(
        builder, *source, reads, reductionRoots, source->T, "reduction.final",
        "linear", true, edges);
    if (failed(reduction))
      return failure();
    finalReduction = *reduction;
  } else {
    unsigned level = 0;
    while (states.size() > 1) {
      SmallVector<Value> nextStates;
      SmallVector<Value> nextRoots;
      SmallVector<TaskflowTaskOp> nextTasks;
      for (unsigned index = 0; index < states.size(); index += 2) {
        if (index + 1 == states.size()) {
          nextStates.push_back(states[index]);
          nextRoots.push_back(roots[index]);
          nextTasks.push_back(partialTasks[index]);
          continue;
        }
        auto alloc = builder.create<memref::AllocOp>(source->producer.getLoc(),
                                                     source->tensorType);
        alloc->setAttr(kPrivatePartialAttr, builder.getUnitAttr());
        std::string name =
            (Twine("reduction.tree.") + Twine(level) + "." +
             Twine(index / 2)).str();
        SmallVector<std::tuple<StringRef, StringRef, StringRef>> edges{
            {partialTasks[index].getTaskName(), "partial_reduction", "tile_local"},
            {partialTasks[index + 1].getTaskName(), "partial_reduction", "tile_local"}};
        FailureOr<TaskflowTaskOp> reduction = createReductionTask(
            builder, *source, {states[index], states[index + 1]},
            {roots[index], roots[index + 1]}, alloc.getMemref(), name,
            "balanced_tree", false, edges);
        if (failed(reduction))
          return failure();
        nextStates.push_back(reduction->getDoneWrites().front());
        nextRoots.push_back(alloc.getMemref());
        nextTasks.push_back(*reduction);
      }
      states = std::move(nextStates);
      roots = std::move(nextRoots);
      partialTasks = std::move(nextTasks);
      ++level;
    }
    SmallVector<Value> reads{source->C, states.front()};
    SmallVector<Value> reductionRoots{source->C, roots.front()};
    SmallVector<std::tuple<StringRef, StringRef, StringRef>> edges{
        {partialTasks.front().getTaskName(), "partial_reduction", "tile_local"}};
    FailureOr<TaskflowTaskOp> reduction = createReductionTask(
        builder, *source, reads, reductionRoots, source->T, "reduction.final",
        "balanced_tree", true, edges);
    if (failed(reduction))
      return failure();
    finalReduction = *reduction;
  }
  auto input = findInput(source->consumer, source->producer.getDoneWrites().front());
  if (!input)
    return reject(source->consumer, "consumer has no T input to rewire");
  source->consumer.getWillReadsMutable()
      .slice(*input, 1)
      .assign(finalReduction.getDoneWrites().front());
  OpBuilder consumerBuilder(source->consumer);
  markConsumer(consumerBuilder, source->consumer, *source, "parallel", topology,
               finalReduction.getTaskName());
  source->function->setAttr(kKPolicyAttr, consumerBuilder.getStringAttr("parallel"));
  source->function->setAttr(kTopologyAttr, consumerBuilder.getStringAttr(topology));
  source->function->setAttr(kRewriteSchemaAttr,
                             consumerBuilder.getStringAttr(kRewriteSchema));
  source->producer.erase();
  return success();
}


// Memory-carried reductions retain the output buffer as the reduction state.
// Unlike the canonical producer/consumer adapter above, no scalar SSA
// iter_arg is available: each K clone runs the original M/N body, and later
// clones read the previous clone's output state from the same memory root.
// This adapter is sequential-only. A parallel split would require a private
// partial buffer and a proof that every output load is independent of K.
// Ordered LU-style cross-cell loads are rejected because the generic K block
// clone cannot preserve their M/N/K iteration order.
struct MemoryFeedbackKSource {
  func::FuncOp function;
  TaskflowTaskOp task;
  neura::KernelOp kernel;
  SmallVector<TaskflowCounterOp> taskCounters;
  SmallVector<neura::CounterOp> kernelCounters;
  unsigned outputReadIndex = 0;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  bool orderedCrossCell = false;
  // Predicate-form kernels keep the complete Neura dataflow graph in each
  // clone. Their counter bounds live in folded attributes rather than SSA
  // operands, so they use a separate bound updater and recurrence proof.
  bool predicateForm = false;
};

static std::optional<int64_t>
memoryTaskConstantIndex(Value value, TaskflowTaskOp task) {
  if (auto literal = constantIndex(value))
    return literal;
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &task.getBody().front())
    return std::nullopt;
  unsigned firstValueInput = task.getWillReads().size() +
                              task.getWillWrites().size();
  if (argument.getArgNumber() < firstValueInput)
    return std::nullopt;
  unsigned index = argument.getArgNumber() - firstValueInput;
  if (index >= task.getValueInputs().size())
    return std::nullopt;
  return constantIndex(task.getValueInputs()[index]);
}

static std::optional<int64_t>
memoryKernelConstantIndex(Value value, TaskflowTaskOp task,
                         neura::KernelOp kernel) {
  if (auto literal = memoryTaskConstantIndex(value, task))
    return literal;
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &kernel.getBody().front() ||
      argument.getArgNumber() >= kernel.getInputs().size())
    return std::nullopt;
  return memoryTaskConstantIndex(kernel.getInputs()[argument.getArgNumber()],
                                 task);
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

// Predicate-form Neura counters commonly have no SSA bound operands. The
// folded %inputN attributes are only accepted after resolving that input back
// through the kernel and Taskflow ABI, so a stale attribute cannot certify a
// different bound.
static std::optional<int64_t>
predicateKernelConstantIndex(neura::CounterOp counter, StringRef attrName,
                             Value operand, TaskflowTaskOp task,
                             neura::KernelOp kernel) {
  std::optional<int64_t> actual;
  if (operand)
    actual = memoryKernelConstantIndex(operand, task, kernel);

  std::optional<int64_t> folded;
  Attribute attribute = counter->getAttr(attrName);
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attribute))
    folded = integer.getInt();
  else if (auto reference = parseKernelInputReference(attribute)) {
    if (*reference >= kernel.getInputs().size())
      return std::nullopt;
    folded = memoryKernelConstantIndex(kernel.getInputs()[*reference], task,
                                       kernel);
  }
  // A real SSA operand is authoritative. If it cannot be resolved, or if a
  // folded attribute disagrees with it, fail closed instead of certifying a
  // stale bound. Predicate counters with no operands may use the folded form.
  if (operand && !actual)
    return std::nullopt;
  if (actual && folded && *actual != *folded)
    return std::nullopt;
  return actual ? actual : folded;
}

static bool predicateValueReferencesKernelInput(
    Value value, unsigned inputIndex, neura::KernelOp kernel,
    DenseSet<Value> &visiting, unsigned depth = 0) {
  if (!value || depth >= 24 || !visiting.insert(value).second)
    return false;
  bool result = false;
  if (auto argument = dyn_cast<BlockArgument>(value))
    result = argument.getOwner() == &kernel.getBody().front() &&
             argument.getArgNumber() == inputIndex;
  else if (Operation *definition = value.getDefiningOp()) {
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
               predicateValueReferencesKernelInput(
                   definition->getOperand(0), inputIndex, kernel, visiting,
                   depth + 1);
    } else if (name == "neura.phi" || name == "neura.sel") {
      unsigned first = name == "neura.sel" ? 1u : 0u;
      result = first < definition->getNumOperands();
      // A phi/select is definite storage provenance only when every data arm
      // resolves to this same kernel input. Use a path-local visiting set so
      // a shared arm is not mistaken for a cycle on a later arm.
      for (unsigned operand = first;
           result && operand < definition->getNumOperands(); ++operand) {
        DenseSet<Value> armVisiting = visiting;
        result = predicateValueReferencesKernelInput(
            definition->getOperand(operand), inputIndex, kernel,
            armVisiting, depth + 1);
      }
    }
  }
  visiting.erase(value);
  return result;
}

static bool predicateValueReferencesKernelInput(Value value, unsigned inputIndex,
                                                neura::KernelOp kernel) {
  DenseSet<Value> visited;
  return predicateValueReferencesKernelInput(value, inputIndex, kernel,
                                             visited);
}

enum class MemoryIndexRole { Unknown, M, N, K, Constant };

static MemoryIndexRole memoryIndexRole(Value value, Value m, Value n, Value k,
                                       DenseSet<Value> &visited,
                                       unsigned depth = 0) {
  if (!value || depth >= 16 || !visited.insert(value).second)
    return MemoryIndexRole::Unknown;
  if (value == m)
    return MemoryIndexRole::M;
  if (value == n)
    return MemoryIndexRole::N;
  if (value == k)
    return MemoryIndexRole::K;
  if (constantIndex(value))
    return MemoryIndexRole::Constant;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return MemoryIndexRole::Unknown;
  StringRef name = definition->getName().getStringRef();
  if (name == "neura.constant" &&
      isa<IntegerAttr>(definition->getAttr("value")))
    return MemoryIndexRole::Constant;
  if (name == "neura.grant_predicate" || name == "neura.ctrl_mov" ||
      name == "neura.cast") {
    if (definition->getNumOperands() < 1)
      return MemoryIndexRole::Unknown;
    DenseSet<Value> operandVisited = visited;
    return memoryIndexRole(definition->getOperand(0), m, n, k,
                           operandVisited, depth + 1);
  }
  if (name == "neura.phi" || name == "neura.sel") {
    if (definition->getNumOperands() < (name == "neura.sel" ? 3u : 1u))
      return MemoryIndexRole::Unknown;
    unsigned first = name == "neura.sel" ? 1u : 0u;
    MemoryIndexRole common = MemoryIndexRole::Unknown;
    for (unsigned index = first; index < definition->getNumOperands(); ++index) {
      DenseSet<Value> branchVisited = visited;
      MemoryIndexRole branch = memoryIndexRole(
          definition->getOperand(index), m, n, k, branchVisited, depth + 1);
      if (branch == MemoryIndexRole::Unknown)
        return MemoryIndexRole::Unknown;
      if (common == MemoryIndexRole::Unknown)
        common = branch;
      else if (common != branch)
        return MemoryIndexRole::Unknown;
    }
    return common;
  }
  return MemoryIndexRole::Unknown;
}

static MemoryIndexRole memoryIndexRole(Value value, Value m, Value n, Value k) {
  DenseSet<Value> visited;
  return memoryIndexRole(value, m, n, k, visited);
}

static bool memoryCellIndices(ValueRange indices, Value m, Value n, Value k) {
  bool hasM = false;
  bool hasN = false;
  for (Value index : indices) {
    switch (memoryIndexRole(index, m, n, k)) {
    case MemoryIndexRole::K:
    case MemoryIndexRole::Unknown:
      return false;
    case MemoryIndexRole::M:
      if (hasM)
        return false;
      hasM = true;
      continue;
    case MemoryIndexRole::N:
      if (hasN)
        return false;
      hasN = true;
      continue;
    case MemoryIndexRole::Constant:
      continue;
    }
  }
  return hasM && hasN;
}

static bool memoryCrossCellIndices(ValueRange indices, Value m, Value n,
                                    Value k) {
  bool hasK = false;
  bool hasOuter = false;
  for (Value index : indices) {
    switch (memoryIndexRole(index, m, n, k)) {
    case MemoryIndexRole::K:
      if (hasK)
        return false;
      hasK = true;
      continue;
    case MemoryIndexRole::M:
    case MemoryIndexRole::N:
      if (hasOuter)
        return false;
      hasOuter = true;
      continue;
    case MemoryIndexRole::Constant:
      continue;
    case MemoryIndexRole::Unknown:
      return false;
    }
  }
  return hasK && hasOuter;
}

static bool memoryIsZeroComparison(Value condition, Value k) {
  auto compare = condition.getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::eq)
    return false;
  return (compare.getLhs() == k && constantIndex(compare.getRhs()) == 0) ||
         (compare.getRhs() == k && constantIndex(compare.getLhs()) == 0);
}

static bool predicateIsKZero(Value condition, Value m, Value n, Value k) {
  Operation *definition = condition.getDefiningOp();
  if (!definition || definition->getName().getStringRef() != "neura.icmp" ||
      definition->getNumOperands() < 1)
    return false;
  auto cmpType = definition->getAttrOfType<StringAttr>("cmpType");
  auto rhs = definition->getAttrOfType<IntegerAttr>("rhs_value");
  return cmpType && cmpType.getValue() == "eq" && rhs && rhs.getInt() == 0 &&
         memoryIndexRole(definition->getOperand(0), m, n, k) ==
             MemoryIndexRole::K;
}

static bool predicateIsKZeroGuarded(Value value, Value m, Value n, Value k) {
  Operation *definition = value.getDefiningOp();
  if (!definition ||
      (definition->getName().getStringRef() != "neura.grant_predicate" &&
       definition->getName().getStringRef() != "neura.ctrl_mov") ||
      definition->getNumOperands() < 2)
    return false;
  return predicateIsKZero(definition->getOperand(1), m, n, k);
}

static LogicalResult validatePredicateMemoryFeedbackK(
    MemoryFeedbackKSource &source, Value outputStorage,
    unsigned outputInput) {
  neura::KernelOp kernel = source.kernel;
  Value m = source.kernelCounters[0].getCurrentIndex();
  Value n = source.kernelCounters[1].getCurrentIndex();
  Value k = source.kernelCounters[2].getCurrentIndex();

  SmallVector<neura::LoadIndexedOp> outputLoads;
  SmallVector<neura::StoreIndexedOp> outputStores;
  unsigned nonOutputStores = 0;
  Operation *invalidAccess = nullptr;
  bool sawMultiply = false;
  kernel.walk([&](Operation *operation) {
    if (invalidAccess || operation == kernel.getOperation())
      return;
    StringRef name = operation->getName().getStringRef();
    sawMultiply |= name == "neura.mul";
    if (auto load = dyn_cast<neura::LoadIndexedOp>(operation)) {
      auto input = parseKernelInputReference(load->getAttr("lhs_value"));
      if (!input || *input >= kernel.getInputs().size()) {
        invalidAccess = operation;
        return;
      }
      if (*input == outputInput)
        outputLoads.push_back(load);
      else if (load.getBase() &&
               predicateValueReferencesKernelInput(load.getBase(), outputInput,
                                                   kernel))
        invalidAccess = operation;
      return;
    }
    if (auto store = dyn_cast<neura::StoreIndexedOp>(operation)) {
      // Predicate lowering uses two StoreIndexed forms.  The recurrent
      // store has no explicit base and records the destination in
      // rhs_value; the K==0 initialization has an explicit output base and
      // records its value in lhs_value.  Classify the latter by tracing its
      // base, rather than treating the folded value attribute as a storage
      // identity.
      // An explicit base and a folded rhs_value are two address sources. Do
      // not guess which one is authoritative when both are present.
      if (store.getBase() && store->hasAttr("rhs_value")) {
        invalidAccess = operation;
        return;
      }
      bool baseIsOutput =
          store.getBase() &&
          predicateValueReferencesKernelInput(store.getBase(), outputInput,
                                               kernel);
      auto input = parseKernelInputReference(store->getAttr("rhs_value"));
      if (baseIsOutput) {
        outputStores.push_back(store);
        return;
      }
      if (!input || *input >= kernel.getInputs().size()) {
        invalidAccess = operation;
        return;
      }
      if (*input == outputInput)
        outputStores.push_back(store);
      else {
        ++nonOutputStores;
        if (store.getBase() &&
            predicateValueReferencesKernelInput(store.getBase(), outputInput,
                                                kernel))
          invalidAccess = operation;
      }
      return;
    }
    if (isa<memref::LoadOp, memref::StoreOp, neura::LoadOp, neura::StoreOp>(
            operation))
      invalidAccess = operation;
  });
  if (invalidAccess)
    return reject(invalidAccess,
                  "predicate memory-feedback K has unclassified indexed memory access");
  if (outputLoads.size() != 1 || outputStores.empty() || outputStores.size() > 2 ||
      nonOutputStores != 0 || !sawMultiply)
    return reject(kernel,
                  "predicate memory-feedback K recurrence shape differs");

  unsigned recurrenceStoreCount = 0;
  unsigned initialStoreCount = 0;
  for (neura::StoreIndexedOp store : outputStores) {
    if (!memoryCellIndices(store.getIndices(), m, n, k))
      return reject(store,
                    "predicate memory-feedback K store coordinates are not [M,N]");
    Operation *valueDefinition = store.getValue().getDefiningOp();
    if (valueDefinition &&
        (valueDefinition->getName().getStringRef() == "neura.add" ||
         valueDefinition->getName().getStringRef() == "neura.sub"))
      ++recurrenceStoreCount;
    else if (predicateIsKZeroGuarded(store.getValue(), m, n, k))
      ++initialStoreCount;
    else
      return reject(store,
                    "predicate memory-feedback K store is neither recurrence nor K==0 initialization");
  }
  bool storeShapeOK =
      (outputStores.size() == 1 && recurrenceStoreCount == 1 &&
       initialStoreCount == 0) ||
      (outputStores.size() == 2 && recurrenceStoreCount == 1 &&
       initialStoreCount == 1);
  if (!storeShapeOK)
    return reject(kernel,
                  "predicate memory-feedback K requires one recurrence and optional K==0 store");

  if (!memoryCellIndices(outputLoads.front().getIndices(), m, n, k))
    return reject(outputLoads.front(),
                  "predicate memory-feedback K output load coordinates are not [M,N]");
  bool sawMatchingAccumulator = false;
  for (neura::StoreIndexedOp store : outputStores) {
    if (!memoryCellIndices(store.getIndices(), m, n, k) ||
        store.getIndices() != outputLoads.front().getIndices())
      continue;
    for (Operation *user : outputLoads.front().getResult().getUsers()) {
      StringRef name = user->getName().getStringRef();
      if ((name != "neura.add" && name != "neura.sub") ||
          store.getValue() != user->getResult(0) ||
          user->getNumOperands() != 2)
        continue;
      unsigned accumulatorCount = 0;
      Value other;
      for (Value operand : user->getOperands()) {
        if (operand == outputLoads.front().getResult())
          ++accumulatorCount;
        else
          other = operand;
      }
      Operation *otherDefinition = other ? other.getDefiningOp() : nullptr;
      if (accumulatorCount != 1 || !otherDefinition ||
          otherDefinition->getName().getStringRef() != "neura.mul" ||
          (name == "neura.sub" &&
           user->getOperand(0) != outputLoads.front().getResult()))
        continue;
      sawMatchingAccumulator = true;
    }
  }
  if (!sawMatchingAccumulator)
    return reject(kernel,
                  "predicate memory-feedback K lacks same-cell accumulator recurrence");

  // The proof deliberately admits only a same-[M,N]-cell state carry. Any
  // second output read, or any coordinate containing K/another outer index,
  // would change the original M/N/K order when the whole kernel is cloned.
  if (outputLoads.size() != 1)
    return reject(kernel,
                  "predicate memory-feedback K has duplicate output reads");
  (void)outputStorage;
  return success();
}

static void setMemoryFeedbackRegions(OpBuilder &builder,
                                     TaskflowTaskOp task) {
  SmallVector<Attribute> inputLowers;
  SmallVector<Attribute> inputUppers;
  SmallVector<Attribute> inputReasons;
  for (Value input : task.getWillReads()) {
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!type || !type.hasStaticShape()) {
      inputLowers.push_back(builder.getUnitAttr());
      inputUppers.push_back(builder.getUnitAttr());
      inputReasons.push_back(builder.getStringAttr("dynamic_external"));
      continue;
    }
    SmallVector<int64_t> lower(type.getRank(), 0);
    SmallVector<int64_t> upper(type.getShape().begin(), type.getShape().end());
    inputLowers.push_back(builder.getDenseI64ArrayAttr(lower));
    inputUppers.push_back(builder.getDenseI64ArrayAttr(upper));
    inputReasons.push_back(builder.getStringAttr("memory_feedback"));
  }
  task->setAttr(kInputRegionLowersAttr, builder.getArrayAttr(inputLowers));
  task->setAttr(kInputRegionUppersAttr, builder.getArrayAttr(inputUppers));
  task->setAttr(kInputRegionReasonsAttr, builder.getArrayAttr(inputReasons));

  SmallVector<Attribute> outputLowers;
  SmallVector<Attribute> outputUppers;
  for (Value output : task.getWillWrites()) {
    auto type = dyn_cast<MemRefType>(output.getType());
    if (!type || !type.hasStaticShape()) {
      outputLowers.push_back(builder.getUnitAttr());
      outputUppers.push_back(builder.getUnitAttr());
      continue;
    }
    SmallVector<int64_t> lower(type.getRank(), 0);
    SmallVector<int64_t> upper(type.getShape().begin(), type.getShape().end());
    outputLowers.push_back(builder.getDenseI64ArrayAttr(lower));
    outputUppers.push_back(builder.getDenseI64ArrayAttr(upper));
  }
  task->setAttr(kOutputRegionLowersAttr, builder.getArrayAttr(outputLowers));
  task->setAttr(kOutputRegionUppersAttr, builder.getArrayAttr(outputUppers));
}

static FailureOr<MemoryFeedbackKSource>
analyzeMemoryFeedbackK(ModuleOp module, StringRef functionName,
                       StringRef selectedTaskName) {
  FailureOr<func::FuncOp> selected = selectFunction(module, functionName);
  if (failed(selected))
    return failure();
  func::FuncOp function = *selected;
  SmallVector<TaskflowTaskOp> matches;
  function.walk([&](TaskflowTaskOp task) {
    if (task.getTaskName() == selectedTaskName)
      matches.push_back(task);
  });
  if (matches.size() != 1)
    return reject(function, (Twine("memory-feedback K requires exactly one ") +
                             selectedTaskName)
                                .str());
  TaskflowTaskOp task = matches.front();
  if (task->hasAttr("amoeba.semantic.k_tiled") ||
      task->hasAttr("amoeba.neura.tiling.parent_task") ||
      task->hasAttr("amoeba.neura.replication.parent_task"))
    return reject(task, "memory-feedback K refuses an already rewritten task");
  if (!task.getBody().hasOneBlock() || task.getWillReads().empty() ||
      task.getWillWrites().size() != 1 || task.getDoneWrites().size() != 1 ||
      task.getBody().front().getNumArguments() !=
          task.getWillReads().size() + task.getWillWrites().size() +
              task.getValueInputs().size())
    return reject(task, "memory-feedback K task ABI differs");

  unsigned outputReadIndex = 0;
  while (outputReadIndex < task.getWillReads().size() &&
         task.getWillReads()[outputReadIndex] != task.getWillWrites().front())
    ++outputReadIndex;
  if (outputReadIndex == task.getWillReads().size() ||
      llvm::count(task.getWillReads(), task.getWillWrites().front()) != 1)
    return reject(task, "memory-feedback K requires one output read/write state");
  if (task.getOriginalReadMemrefs().size() != task.getWillReads().size() ||
      task.getOriginalWriteMemrefs().size() != 1)
    return reject(task,
                  "memory-feedback K output read/write provenance is incomplete");
  Value outputStorage = task.getOriginalWriteMemrefs().front();
  unsigned aliasedOutputReads = 0;
  for (auto [index, read] : llvm::enumerate(task.getWillReads()))
    aliasedOutputReads +=
        representsOriginalStorage(read, outputStorage) ||
        task.getOriginalReadMemrefs()[index] == outputStorage;
  if (aliasedOutputReads != 1)
    return reject(task,
                  "memory-feedback K output read/write has an ambiguous alias");

  SmallVector<TaskflowCounterOp> taskCounters;
  neura::KernelOp kernel;
  Block &taskBody = task.getBody().front();
  for (Operation &operation : taskBody) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      taskCounters.push_back(counter);
      continue;
    }
    if (auto candidate = dyn_cast<neura::KernelOp>(&operation)) {
      if (kernel)
        return reject(task, "memory-feedback K requires one neura.kernel");
      kernel = candidate;
      continue;
    }
    if (!isa<arith::ConstantOp, arith::ConstantIndexOp, TaskflowYieldOp>(
            &operation))
      return reject(&operation, "memory-feedback K task shell differs");
  }
  if (taskCounters.size() != 3 || !kernel || !kernel.getBody().hasOneBlock() ||
      !kernel.getIterArgsInit().empty() || kernel.getNumResults() != 0)
    return reject(task, "memory-feedback K requires three counters and a stateless kernel");
  bool predicateForm =
      kernel->getAttrOfType<StringAttr>("dataflow_mode") &&
      kernel->getAttrOfType<StringAttr>("dataflow_mode").getValue() ==
          "predicate";
  for (auto [index, counter] : llvm::enumerate(taskCounters)) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    auto lower = memoryTaskConstantIndex(counter.getLowerBound(), task);
    auto step = memoryTaskConstantIndex(counter.getStep(), task);
    auto upper = memoryTaskConstantIndex(counter.getUpperBound(), task);
    bool parentOK = index == 0
                        ? !counter.getParentIndex()
                        : counter.getParentIndex() ==
                              taskCounters[index - 1].getCounterIndex();
    if (!id || id.getInt() != static_cast<int64_t>(index) || !lower ||
        !upper || !step || *lower != 0 || *upper <= *lower || *step != 1 ||
        !parentOK)
      return reject(counter, "memory-feedback K Taskflow counters differ");
  }
  MemoryFeedbackKSource result;
  result.function = function;
  result.task = task;
  result.kernel = kernel;
  result.taskCounters = taskCounters;
  result.outputReadIndex = outputReadIndex;
  result.predicateForm = predicateForm;
  auto mExtent = memoryTaskConstantIndex(taskCounters[0].getUpperBound(), task);
  auto nExtent = memoryTaskConstantIndex(taskCounters[1].getUpperBound(), task);
  auto kExtent = memoryTaskConstantIndex(taskCounters[2].getUpperBound(), task);
  if (!mExtent || !nExtent || !kExtent)
    return reject(task, "memory-feedback K counter bounds are not constants");
  result.M = *mExtent;
  result.N = *nExtent;
  result.K = *kExtent;

  unsigned outputArgument = task.getWillReads().size();
  Value outputValue = taskBody.getArgument(outputArgument);
  std::optional<unsigned> outputInput;
  unsigned outputInputAliases = 0;
  for (auto [index, input] : llvm::enumerate(kernel.getInputs()))
    if (input == outputValue ||
        taskBodyValueRepresentsStorage(input, task, outputStorage)) {
      outputInput = index;
      ++outputInputAliases;
    }
  if (!outputInput)
    return reject(kernel, "memory-feedback K cannot resolve output kernel input");
  if (outputInputAliases != 1)
    return reject(kernel,
                  "memory-feedback K output kernel input has an ambiguous alias");

  SmallVector<neura::CounterOp> kernelCounters;
  for (Operation &operation : kernel.getBody().front())
    if (auto counter = dyn_cast<neura::CounterOp>(&operation))
      kernelCounters.push_back(counter);
  if (kernelCounters.size() != 3)
    return reject(kernel, "memory-feedback K requires three Neura counters");
  for (auto [index, counter] : llvm::enumerate(kernelCounters)) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    auto resolveBound = [&](StringRef attrName, Value operand)
        -> std::optional<int64_t> {
      if (predicateForm)
        return predicateKernelConstantIndex(counter, attrName, operand, task,
                                            kernel);
      return memoryKernelConstantIndex(operand, task, kernel);
    };
    auto lower = resolveBound("lower_bound_value", counter.getLowerBound());
    auto step = resolveBound("step_value", counter.getStep());
    auto upper = resolveBound("upper_bound_value", counter.getUpperBound());
    auto hierarchy = counter.getCounterHierarchyAttr();
    StringRef expectedHierarchy = index == 0 ? "root" :
                                  (index == 1 ? "relay" : "leaf");
    bool hierarchyOK = hierarchy && hierarchy.getValue() == expectedHierarchy;
    if (!id || id.getInt() != static_cast<int64_t>(index) || !lower ||
        !upper || !step || *lower != 0 || *upper <= *lower || *step != 1 ||
        !hierarchyOK || *upper !=
                          *memoryTaskConstantIndex(
                              taskCounters[index].getUpperBound(), task))
      return reject(counter, "memory-feedback K Neura counters differ");
    result.kernelCounters.push_back(counter);
  }

  if (predicateForm) {
    if (failed(validatePredicateMemoryFeedbackK(result, outputStorage,
                                                *outputInput)))
      return failure();
    return result;
  }

  Block &kernelBody = kernel.getBody().front();
  Value output = kernelBody.getArgument(*outputInput);
  // Every memory effect in this state machine must be either an input load or
  // the one output update/initialization store. In particular, silently
  // ignoring a store to an unrelated buffer would make the facts and the
  // materializer prove different programs.
  DenseSet<Value> allowedMemrefs;
  for (auto [index, input] : llvm::enumerate(kernel.getInputs()))
    if (isa<MemRefType>(input.getType()))
      allowedMemrefs.insert(kernelBody.getArgument(index));
  bool unsupportedMemoryEffect = false;
  kernel.walk([&](Operation *operation) {
    if (unsupportedMemoryEffect || operation == kernel.getOperation())
      return;
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      if (!allowedMemrefs.contains(load.getMemRef()))
        unsupportedMemoryEffect = true;
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      if (store.getMemRef() != output)
        unsupportedMemoryEffect = true;
      return;
    }
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(operation)) {
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      if (!instances.empty())
        unsupportedMemoryEffect = true;
    }
  });
  if (unsupportedMemoryEffect)
    return reject(kernel,
                  "memory-feedback K has unsupported extra memory effects");
  Value m = result.kernelCounters[0].getCurrentIndex();
  Value n = result.kernelCounters[1].getCurrentIndex();
  Value k = result.kernelCounters[2].getCurrentIndex();
  SmallVector<memref::LoadOp> outputLoads;
  SmallVector<memref::StoreOp> outputStores;
  SmallVector<memref::StoreOp> directOutputStores;
  SmallVector<arith::MulIOp> multiplies;
  scf::IfOp initial;
  for (Operation &operation : kernelBody) {
    if (auto load = dyn_cast<memref::LoadOp>(&operation)) {
      if (load.getMemRef() == output)
        outputLoads.push_back(load);
      continue;
    }
    if (auto store = dyn_cast<memref::StoreOp>(&operation)) {
      if (store.getMemRef() == output) {
        outputStores.push_back(store);
        directOutputStores.push_back(store);
      }
      continue;
    }
    if (auto mul = dyn_cast<arith::MulIOp>(&operation)) {
      multiplies.push_back(mul);
      continue;
    }
    if (auto candidate = dyn_cast<scf::IfOp>(&operation)) {
      if (initial)
        return reject(kernel, "memory-feedback K has multiple initial stores");
      initial = candidate;
    }
  }
  // The initialization store is nested in scf.if and therefore absent from
  // the direct list above. Find it and include it in the total output effect.
  SmallVector<memref::StoreOp> nestedOutputStores;
  kernel.walk([&](memref::StoreOp store) {
    if (store.getMemRef() == output && store->getBlock() != &kernelBody)
      nestedOutputStores.push_back(store);
  });
  outputStores.append(nestedOutputStores.begin(), nestedOutputStores.end());
  if (outputLoads.empty() || outputLoads.size() > 2 ||
      directOutputStores.size() != 1 || nestedOutputStores.size() > 1 ||
      (!initial && !nestedOutputStores.empty()) || multiplies.empty())
    return reject(kernel, "memory-feedback K output recurrence shape differs");
  // Some legal source tasks receive an already initialized output state from a
  // predecessor (LLaMA Task_8 and GCN tasks), so no in-kernel K==0 store is
  // required. If an initialization branch exists, validate it exactly and
  // retain it in block zero only.
  if (initial) {
    if (nestedOutputStores.size() != 1 ||
        !memoryIsZeroComparison(initial.getCondition(), k) ||
        !initial.getThenRegion().hasOneBlock() || !initial.getElseRegion().empty())
      return reject(initial, "memory-feedback K initialization must be K==0");
    Block &thenBody = initial.getThenRegion().front();
    auto initialStore = nestedOutputStores.front();
    if (!memoryCellIndices(initialStore.getIndices(), m, n, k) ||
        !isa<scf::YieldOp>(thenBody.getTerminator()) ||
        std::distance(thenBody.begin(), thenBody.end()) != 2 ||
        &*thenBody.begin() != initialStore.getOperation())
      return reject(initial, "memory-feedback K initial output cell differs");
    for (Value result : initial->getResults())
      if (!result.use_empty())
        return reject(initial,
                      "memory-feedback K initialization has a live result");
  }

  memref::StoreOp updateStore = directOutputStores.front();
  if (!memoryCellIndices(updateStore.getIndices(), m, n, k))
    return reject(updateStore, "memory-feedback K update cell differs");
  Operation *update = updateStore.getValue().getDefiningOp();
  if (!update || (!isa<arith::AddIOp, arith::SubIOp>(update)))
    return reject(updateStore, "memory-feedback K update must be add/sub");
  memref::LoadOp accumulator;
  for (memref::LoadOp load : outputLoads)
    if (load.getIndices() == updateStore.getIndices() &&
        llvm::is_contained(update->getOperands(), load.getResult()))
      accumulator = load;
  if (!accumulator)
    return reject(updateStore, "memory-feedback K update lacks accumulator load");
  if (update->getNumOperands() != 2)
    return reject(updateStore, "memory-feedback K update has extra operands");
  unsigned accumulatorCount = 0;
  Value product;
  for (Value operand : update->getOperands()) {
    if (operand == accumulator.getResult())
      ++accumulatorCount;
    else
      product = operand;
  }
  Operation *productDef = product ? product.getDefiningOp() : nullptr;
  if (accumulatorCount != 1 || !productDef ||
      !isa<arith::MulIOp>(productDef) ||
      (isa<arith::SubIOp>(update) &&
       update->getOperand(0) != accumulator.getResult()))
    return reject(updateStore,
                  "memory-feedback K update is not accumulator plus product");
  bool sawCrossCellLoad = false;
  unsigned sameCellLoadCount = 0;
  unsigned crossCellLoadCount = 0;
  for (memref::LoadOp load : outputLoads) {
    bool sameCell = memoryCellIndices(load.getIndices(), m, n, k);
    bool crossCell = outputLoads.size() == 2 &&
                     memoryCrossCellIndices(load.getIndices(), m, n, k);
    if (!sameCell && !crossCell)
      return reject(load,
                    "memory-feedback K output load uses an unproved index");
    sawCrossCellLoad |= crossCell;
    sameCellLoadCount += sameCell;
    crossCellLoadCount += crossCell;
  }
  if (sameCellLoadCount != 1 ||
      crossCellLoadCount != (outputLoads.size() == 2 ? 1u : 0u))
    return reject(kernel,
                  "memory-feedback K has duplicate or missing output reads");
  if (outputLoads.size() == 2) {
    if (!isa<arith::SubIOp>(update) || !sawCrossCellLoad)
      return reject(kernel,
                    "memory-feedback K cross-cell recurrence is not LU subtraction");
    bool differs = false;
    for (memref::LoadOp load : outputLoads)
      differs |= load.getIndices() != accumulator.getIndices();
    if (!differs)
      return reject(kernel, "memory-feedback K second output load is not cross-cell");
    // K materialization clones the complete M/N body for each K block. A
    // cross-cell output address containing K and an outer counter therefore
    // changes the original M/N/K iteration order. Without a separate ordered
    // schedule proof, retaining this recurrence would be unsound.
    return reject(kernel,
                  "memory-feedback K cross-cell recurrence order is not preserved");
  }
  result.kernelCounters = kernelCounters;
  return result;
}

static void updateMemoryCounterBounds(TaskflowCounterOp taskCounter,
                                      neura::CounterOp kernelCounter,
                                      TaskflowTaskOp task,
                                      neura::KernelOp kernel, int64_t lower,
                                      int64_t upper) {
  OpBuilder taskBuilder(taskCounter);
  Value taskLower = taskBuilder.create<arith::ConstantIndexOp>(
      taskCounter.getLoc(), lower);
  Value taskUpper = taskBuilder.create<arith::ConstantIndexOp>(
      taskCounter.getLoc(), upper);
  taskCounter.getLowerBoundMutable().assign(taskLower);
  taskCounter.getUpperBoundMutable().assign(taskUpper);
  taskCounter->setAttr("counter_dynamism",
                       taskBuilder.getStringAttr("constant_bound"));
  OpBuilder kernelBuilder(kernelCounter);
  Value kernelLower = kernelBuilder.create<arith::ConstantIndexOp>(
      kernelCounter.getLoc(), lower);
  Value kernelUpper = kernelBuilder.create<arith::ConstantIndexOp>(
      kernelCounter.getLoc(), upper);
  kernelCounter.getLowerBoundMutable().assign(kernelLower);
  kernelCounter.getUpperBoundMutable().assign(kernelUpper);
  kernelCounter->setAttr("counter_dynamism",
                         kernelBuilder.getStringAttr("constant_bound"));
  (void)task;
  (void)kernel;
}

static void updatePredicateCounterBounds(TaskflowCounterOp taskCounter,
                                         neura::CounterOp kernelCounter,
                                         TaskflowTaskOp task,
                                         neura::KernelOp kernel, int64_t lower,
                                         int64_t upper) {
  // Taskflow retains SSA bounds, while predicate Neura counters retain their
  // folded bounds as attributes.  Update both views in every clone so the
  // facts pass can compare the advertised Neura range with its Taskflow
  // source rather than trusting a stale symbolic %inputN attribute.
  updateMemoryCounterBounds(taskCounter, kernelCounter, task, kernel, lower,
                            upper);
  OpBuilder builder(kernelCounter);
  kernelCounter->setAttr("lower_bound_value", builder.getIndexAttr(lower));
  kernelCounter->setAttr("upper_bound_value", builder.getIndexAttr(upper));
  kernelCounter->setAttr("step_value", builder.getIndexAttr(1));
  kernelCounter->setAttr("counter_dynamism",
                         builder.getStringAttr("constant_bound"));
}

static void addMemoryFeedbackYield(Block *body, TaskflowYieldOp originalYield,
                                   IRMapping &mapping, Value output) {
  SmallVector<Value> doneReads;
  for (Value value : originalYield.getDoneReads())
    doneReads.push_back(mapping.lookupOrDefault(value));
  SmallVector<Value> valueOutputs;
  for (Value value : originalYield.getValueResults())
    valueOutputs.push_back(mapping.lookupOrDefault(value));
  OpBuilder builder = OpBuilder::atBlockEnd(body);
  builder.create<TaskflowYieldOp>(output.getLoc(), doneReads,
                                  ValueRange{output}, valueOutputs);
}

static FailureOr<TaskflowTaskOp>
cloneMemoryFeedbackSequentialBlock(OpBuilder &builder,
                                   MemoryFeedbackKSource &source,
                                   Value previousState, unsigned index,
                                   unsigned blockCount, int64_t lower,
                                   int64_t upper) {
  TaskflowTaskOp original = source.task;
  SmallVector<Value> reads(original.getWillReads());
  if (previousState)
    reads[source.outputReadIndex] = previousState;
  Value output = original.getWillWrites().front();
  std::string name =
      (Twine(original.getTaskName()) + ".k." + Twine(index)).str();
  auto task = builder.create<TaskflowTaskOp>(
      original.getLoc(), original.getDoneReads().getTypes(),
      original.getDoneWrites().getTypes(), original.getValueOutputs().getTypes(),
      reads, ValueRange{output}, original.getValueInputs(),
      builder.getStringAttr(name), original.getOriginalReadMemrefs(),
      original.getOriginalWriteMemrefs());
  Block *body = new Block();
  task.getBody().push_back(body);
  for (Value read : reads)
    body->addArgument(read.getType(), original.getLoc());
  body->addArgument(output.getType(), original.getLoc());
  for (Value input : original.getValueInputs())
    body->addArgument(input.getType(), original.getLoc());

  Block &oldBody = original.getBody().front();
  auto originalYield = dyn_cast<TaskflowYieldOp>(oldBody.getTerminator());
  if (!originalYield)
    return reject(original, "memory-feedback K source lacks taskflow.yield");
  IRMapping mapping;
  for (unsigned input = 0; input < reads.size(); ++input)
    mapping.map(oldBody.getArgument(input), body->getArgument(input));
  unsigned writeArgument = reads.size();
  mapping.map(oldBody.getArgument(original.getWillReads().size()),
              body->getArgument(writeArgument));
  unsigned valueOffset = original.getWillReads().size() +
                         original.getWillWrites().size();
  for (unsigned input = 0; input < original.getValueInputs().size(); ++input)
    mapping.map(oldBody.getArgument(valueOffset + input),
                body->getArgument(writeArgument + 1 + input));
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  for (Operation &operation : oldBody.without_terminator())
    bodyBuilder.clone(operation, mapping);
  addMemoryFeedbackYield(body, originalYield, mapping,
                          body->getArgument(writeArgument));

  TaskflowCounterOp taskK;
  neura::KernelOp kernel;
  for (Operation &operation : *body) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
      if (id && id.getInt() == 2)
        taskK = counter;
    } else if (auto candidate = dyn_cast<neura::KernelOp>(&operation)) {
      kernel = candidate;
    }
  }
  if (!taskK || !kernel || !kernel.getBody().hasOneBlock())
    return reject(task, "memory-feedback K clone lost counter or kernel");
  SmallVector<neura::CounterOp> kernelCounters;
  for (Operation &operation : kernel.getBody().front())
    if (auto counter = dyn_cast<neura::CounterOp>(&operation))
      kernelCounters.push_back(counter);
  if (kernelCounters.size() != 3)
    return reject(kernel, "memory-feedback K clone lost Neura counters");
  if (source.predicateForm)
    updatePredicateCounterBounds(taskK, kernelCounters[2], task, kernel, lower,
                                 upper);
  else
    updateMemoryCounterBounds(taskK, kernelCounters[2], task, kernel, lower,
                              upper);
  if (previousState) {
    // If the source had a K==0 initialization, only the first clone keeps it.
    // Tasks whose output state is initialized by a predecessor have no branch
    // here; their state is carried solely by the previous output token.
    for (Operation &operation : kernel.getBody().front())
      if (auto initial = dyn_cast<scf::IfOp>(&operation)) {
        for (Value result : initial->getResults())
          if (!result.use_empty())
            return reject(initial,
                          "memory-feedback K cannot erase a live initialization result");
        initial.erase();
        break;
      }
  }

  copyTaskAttributes(original, task, name);
  // A memory-carried K clone is a serialized state machine. Preserve no
  // source replication/runtime scheduling claims on any clone, including the
  // first block, because duplicating it would duplicate the carried state.
  task->removeAttr("dlp_replicable");
  task->removeAttr("runtime_managable");
  markTask(builder, task, "memory_feedback_producer", "producer", "sequential",
           source.orderedCrossCell ? "ordered_cross_cell" : "none");
  task->setAttr("amoeba.semantic.k_tiled", builder.getUnitAttr());
  task->setAttr("amoeba.tiling.parent_task",
                builder.getStringAttr(original.getTaskName()));
  task->setAttr("amoeba.semantic.k_source",
                builder.getStringAttr("memory_feedback"));
  task->setAttr("amoeba.semantic.output_visibility",
                builder.getStringAttr("global_materialized"));
  task->setAttr("amoeba.semantic.k_parallel_legal", builder.getBoolAttr(false));
  task->setAttr("amoeba.semantic.k_block_index",
                builder.getI64IntegerAttr(index));
  task->setAttr("amoeba.semantic.k_block_count",
                builder.getI64IntegerAttr(blockCount));
  task->setAttr(kInitialCOwnershipAttr,
                builder.getStringAttr(index == 0 ? "initial_C_once"
                                                  : "state_carry"));
  task->setAttr(kFinalCOwnershipAttr,
                builder.getStringAttr(index + 1 == blockCount
                                          ? "external_final_C"
                                          : "state_carry"));
  setRangeAttrs(builder, task, {0, source.M}, {0, source.N},
                {lower, upper});
  setMemoryFeedbackRegions(builder, task);
  SmallVector<std::tuple<StringRef, StringRef, StringRef>> edges;
  llvm::SmallPtrSet<Operation *, 4> producers;
  for (unsigned read = 0; read < original.getWillReads().size(); ++read) {
    if (read == source.outputReadIndex)
      continue;
    if (auto producer = original.getWillReads()[read]
                            .getDefiningOp<TaskflowTaskOp>())
      if (producers.insert(producer.getOperation()).second)
        edges.push_back({producer.getTaskName(), "producer_consumer",
                         "tensor_wide"});
  }
  if (previousState) {
    std::string previousName =
        (Twine(original.getTaskName()) + ".k." + Twine(index - 1)).str();
    edges.push_back({previousName, "state_carry", "tile_local"});
  }
  // Predicate kernels retain multiple result-token reads from the same
  // producer.  The compact source|role|scope annotation cannot represent
  // that multiplicity without making the facts validator ambiguous; the
  // typed Taskflow graph remains authoritative for these clones.
  if (source.predicateForm)
    task->removeAttr(kIncomingEdgesAttr);
  else
    setIncomingEdges(builder, task, edges);
  return task;
}

// A memory-feedback task may carry WAR tokens in done_reads in addition to
// the output state in done_writes.  Those tokens remain live after the source
// task is replaced, so every result segment must be forwarded by the final K
// block before the source operation can be erased.  Keep this check explicit:
// a newly introduced result segment must fail closed rather than reintroduce
// the operation-lifetime abort that motivated this pass.
static LogicalResult replaceMemoryFeedbackResults(TaskflowTaskOp source,
                                                   TaskflowTaskOp finalBlock) {
  if (source.getDoneReads().size() != finalBlock.getDoneReads().size() ||
      source.getDoneWrites().size() != finalBlock.getDoneWrites().size() ||
      source.getValueOutputs().size() != finalBlock.getValueOutputs().size())
    return reject(source,
                  "memory-feedback K clone result segments differ from source");

  auto replace = [](ValueRange oldValues, ValueRange newValues) {
    for (auto [oldValue, newValue] : llvm::zip(oldValues, newValues))
      oldValue.replaceAllUsesWith(newValue);
  };
  replace(source.getDoneReads(), finalBlock.getDoneReads());
  replace(source.getDoneWrites(), finalBlock.getDoneWrites());
  replace(source.getValueOutputs(), finalBlock.getValueOutputs());

  for (Value result : source->getResults())
    if (!result.use_empty())
      return reject(source,
                    "memory-feedback K source result remains live after rewrite");
  return success();
}

static LogicalResult runMemoryFeedbackSequentialK(ModuleOp module,
                                                   StringRef functionName,
                                                   StringRef taskName,
                                                   int64_t tileSize,
                                                   int64_t factor) {
  FailureOr<MemoryFeedbackKSource> source =
      analyzeMemoryFeedbackK(module, functionName, taskName);
  if (failed(source))
    return failure();
  FailureOr<int64_t> width = resolveWidth(source->K, tileSize, factor, true);
  if (failed(width))
    return reject(source->task,
                  "memory-feedback sequential K requires a width in [1,K)");
  unsigned blockCount = static_cast<unsigned>(
      (source->K + *width - 1) / *width);
  if (blockCount < 2)
    return reject(source->task,
                  "memory-feedback sequential K requires at least two blocks");
  for (unsigned index = 0; index < blockCount; ++index) {
    std::string name =
        (Twine(source->task.getTaskName()) + ".k." + Twine(index)).str();
    bool collision = false;
    source->function.walk([&](TaskflowTaskOp task) {
      collision |= task != source->task && task.getTaskName() == name;
    });
    if (collision)
      return reject(source->task, "memory-feedback K derived task name collides");
  }
  OpBuilder builder(source->task);
  Value previousState;
  SmallVector<TaskflowTaskOp> blocks;
  for (unsigned index = 0; index < blockCount; ++index) {
    int64_t lower = index * *width;
    int64_t upper = std::min<int64_t>(source->K, lower + *width);
    FailureOr<TaskflowTaskOp> block = cloneMemoryFeedbackSequentialBlock(
        builder, *source, previousState, index, blockCount, lower, upper);
    if (failed(block))
      return failure();
    blocks.push_back(*block);
    previousState = block->getDoneWrites().front();
  }
  if (failed(replaceMemoryFeedbackResults(source->task, blocks.back())))
    return failure();
  source->function->setAttr(kKPolicyAttr,
                            builder.getStringAttr("sequential"));
  source->function->setAttr(kRewriteSchemaAttr,
                            builder.getStringAttr(kRewriteSchema));
  source->task.erase();
  return success();
}

// The AMOEBA-Test LLaMA input-0 Q/K/V graph uses a retained third counter
// for K, unlike the canonical scf.for fixture above.  This matcher is
// deliberately specific to the checked Task_1/Task_2/Task_3 caller
// contracts. It does not infer a GEMM from an arbitrary three-counter
// kernel. Each task is accepted only when its exact operand types, bounds,
// recurrence, and storage ownership match the contract below.
struct LlamaApplicationKSource {
  func::FuncOp function;
  TaskflowTaskOp task;
  neura::KernelOp kernel;
  TaskflowCounterOp taskK;
  neura::CounterOp kernelK;
  arith::AddIOp accumulator;
  scf::IfOp initialStore;
  std::string taskName;
  int64_t M = 320;
  int64_t N = 256;
  int64_t K = 256;
  bool relayUsesOuterBound = false;
};

static bool hasExactCallerShape(func::FuncOp function, Value value,
                                unsigned argumentNumber,
                                ArrayRef<int64_t> expected) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &function.getBody().front() ||
      argument.getArgNumber() != argumentNumber)
    return false;
  auto shape = dyn_cast_or_null<DenseI64ArrayAttr>(
      function.getArgAttr(argumentNumber, "amoeba.logical_transfer_shape"));
  return shape && shape.asArrayRef() == expected;
}

static bool hasIndices(ValueRange actual, Value first, Value second) {
  return actual.size() == 2 && actual[0] == first && actual[1] == second;
}

static FailureOr<LlamaApplicationKSource>
analyzeLlamaApplicationK(ModuleOp module, StringRef functionName,
                         StringRef selectedTaskName) {
  FailureOr<func::FuncOp> selected = selectFunction(module, functionName);
  if (failed(selected))
    return failure();
  func::FuncOp function = *selected;
  if (function.getSymName() !=
      "_Z10llama_funciPA256_KiS1_S1_S1_S1_S1_S1_PA256_i")
    return reject(function, "application K requires the pinned LLaMA function");
  auto bound = function->getAttrOfType<IntegerAttr>("amoeba.static_bound.arg.0");
  if (!bound || bound.getInt() != 320)
    return reject(function, "application K requires input-0 bound 320");

  bool task3 = selectedTaskName == "Task_3";
  unsigned secondInputArgument = 0;
  int64_t expectedN = 0;
  if (selectedTaskName == "Task_1") {
    secondInputArgument = 3;
    expectedN = 256;
  } else if (selectedTaskName == "Task_2") {
    secondInputArgument = 4;
    expectedN = 256;
  } else if (task3) {
    // Task_3 is the QK^T product: both output dimensions are the static
    // caller bound, while K remains the retained 256-wide counter.
    expectedN = 320;
  } else {
    return reject(function,
                  (Twine("application K does not support task ") +
                   selectedTaskName + "; expected Task_1, Task_2, or Task_3")
                      .str());
  }

  SmallVector<TaskflowTaskOp> matches;
  function.walk([&](TaskflowTaskOp task) {
    if (task.getTaskName() == selectedTaskName)
      matches.push_back(task);
  });
  if (matches.size() != 1)
    return reject(function,
                  (Twine("application K requires exactly one ") +
                   selectedTaskName)
                      .str());
  TaskflowTaskOp task = matches.front();
  if (task->hasAttr("amoeba.semantic.k_tiled") ||
      task->hasAttr("amoeba.neura.tiling.parent_task") ||
      task->hasAttr("amoeba.neura.replication.parent_task"))
    return reject(task, "application K refuses an already rewritten task");
  if (!task.getBody().hasOneBlock() || task.getWillReads().size() != 2 ||
      task.getWillWrites().size() != 1 || task.getValueInputs().size() != 2 ||
      !task.getDoneReads().empty() || task.getDoneWrites().size() != 1 ||
      !task.getValueOutputs().empty() ||
      task.getOriginalReadMemrefs().size() != 2 ||
      task.getOriginalWriteMemrefs().size() != 1 ||
      task.getBody().front().getNumArguments() != 5)
    return reject(task, "application K operand ABI differs");
  Value A = task.getWillReads()[0];
  Value B = task.getWillReads()[1];
  Value T = task.getWillWrites()[0];
  auto i32 = IntegerType::get(module.getContext(), 32);
  auto inputType = MemRefType::get({512, 256}, i32);
  // Task_3 has a 512-wide physical output buffer but a 320-wide logical N
  // bound. Task_1/Task_2 use a 256-wide physical output for N=256.
  int64_t outputStorageN = task3 ? 512 : expectedN;
  auto outputType = MemRefType::get({512, outputStorageN}, i32);
  bool callerInputsMatch =
      !task3 && hasExactCallerShape(function, A, 1, {512, 256}) &&
      hasExactCallerShape(function, B, secondInputArgument, {256, 256});
  bool staticTask3Storage =
      task3 && dyn_cast<MemRefType>(A.getType()) == inputType &&
      dyn_cast<MemRefType>(B.getType()) == inputType &&
      isa<memref::AllocaOp>(task.getOriginalReadMemrefs()[0].getDefiningOp()) &&
      isa<memref::AllocaOp>(task.getOriginalReadMemrefs()[1].getDefiningOp()) &&
      provesDistinctStorage(task.getOriginalReadMemrefs()[0],
                            task.getOriginalReadMemrefs()[1]) &&
      representsOriginalStorage(A, task.getOriginalReadMemrefs()[0]) &&
      representsOriginalStorage(B, task.getOriginalReadMemrefs()[1]);
  bool originalInputsMatch =
      task3 ? staticTask3Storage
            : (task.getOriginalReadMemrefs()[0] == A &&
               task.getOriginalReadMemrefs()[1] == B);
  if (!originalInputsMatch || task.getOriginalWriteMemrefs()[0] != T ||
      !isa_and_nonnull<memref::AllocaOp>(T.getDefiningOp()) ||
      (!callerInputsMatch && !staticTask3Storage) ||
      dyn_cast<MemRefType>(T.getType()) != outputType ||
      constantIndex(task.getValueInputs()[0]) != 320 ||
      constantIndex(task.getValueInputs()[1]) != 0)
    return reject(task, "application K lacks exact disjoint storage, caller "
                        "shapes, bound, or zero initial state");

  Block &body = task.getBody().front();
  SmallVector<TaskflowCounterOp> counters;
  neura::KernelOp kernel;
  for (Operation &operation : body) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      counters.push_back(counter);
      continue;
    }
    if (auto candidate = dyn_cast<neura::KernelOp>(&operation)) {
      if (kernel)
        return reject(task, "application K has multiple kernels");
      kernel = candidate;
      continue;
    }
    if (!isa<arith::ConstantOp, TaskflowYieldOp>(&operation))
      return reject(&operation, "application K shell differs");
  }
  if (counters.size() != 3 || !kernel || !kernel.getBody().hasOneBlock() ||
      counters[0].getParentIndex() ||
      counters[1].getParentIndex() != counters[0].getCounterIndex() ||
      counters[2].getParentIndex() != counters[1].getCounterIndex() ||
      constantIndex(counters[0].getLowerBound()) != 0 ||
      counters[0].getUpperBound() != body.getArgument(3) ||
      constantIndex(counters[0].getStep()) != 1 ||
      constantIndex(counters[1].getLowerBound()) != 0 ||
      (task3 ? counters[1].getUpperBound() != body.getArgument(3)
              : constantIndex(counters[1].getUpperBound()) != 256) ||
      constantIndex(counters[1].getStep()) != 1 ||
      constantIndex(counters[2].getLowerBound()) != 0 ||
      constantIndex(counters[2].getUpperBound()) != 256 ||
      constantIndex(counters[2].getStep()) != 1)
    return reject(task, "application K must have exact M/N/K counters");
  for (auto [index, counter] : llvm::enumerate(counters)) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    if (!id || id.getInt() != static_cast<int64_t>(index))
      return reject(counter, "application K Taskflow counter IDs differ");
  }
  if (kernel.getInputs().size() != 5 || kernel.getIterArgsInit().size() != 1 ||
      kernel.getNumResults() != 1 ||
      kernel.getInputs()[0] != body.getArgument(4) ||
      kernel.getInputs()[1] != body.getArgument(2) ||
      kernel.getInputs()[2] != body.getArgument(0) ||
      kernel.getInputs()[3] != body.getArgument(1) ||
      kernel.getInputs()[4] != body.getArgument(3) ||
      kernel.getIterArgsInit()[0] != body.getArgument(4) ||
      kernel.getBody().front().getNumArguments() != 6)
    return reject(kernel, "application K kernel ABI differs");
  auto taskYield = dyn_cast<TaskflowYieldOp>(body.getTerminator());
  if (!taskYield || taskYield.getDoneWrites().size() != 1 ||
      taskYield.getDoneWrites()[0] != body.getArgument(2))
    return reject(task, "application K yield differs");

  Block &kernelBody = kernel.getBody().front();
  Value zero = kernelBody.getArgument(0);
  Value output = kernelBody.getArgument(1);
  Value inputA = kernelBody.getArgument(2);
  Value inputB = kernelBody.getArgument(3);
  Value outerBound = kernelBody.getArgument(4);
  Value acc = kernelBody.getArgument(5);
  SmallVector<neura::CounterOp> neuraCounters;
  SmallVector<memref::LoadOp> loads;
  SmallVector<memref::StoreOp> stores;
  arith::CmpIOp cmp;
  arith::MulIOp mul;
  arith::AddIOp add;
  scf::IfOp init;
  for (Operation &operation : kernelBody) {
    if (auto item = dyn_cast<neura::CounterOp>(&operation))
      neuraCounters.push_back(item);
    else if (auto item = dyn_cast<memref::LoadOp>(&operation))
      loads.push_back(item);
    else if (auto item = dyn_cast<memref::StoreOp>(&operation))
      stores.push_back(item);
    else if (auto item = dyn_cast<arith::CmpIOp>(&operation)) {
      if (cmp)
        return reject(kernel, "application K has extra comparison");
      cmp = item;
    } else if (auto item = dyn_cast<arith::MulIOp>(&operation)) {
      if (mul)
        return reject(kernel, "application K has extra multiply");
      mul = item;
    } else if (auto item = dyn_cast<arith::AddIOp>(&operation)) {
      if (add)
        return reject(kernel, "application K has extra add");
      add = item;
    } else if (auto item = dyn_cast<scf::IfOp>(&operation)) {
      if (init)
        return reject(kernel, "application K has extra conditional");
      init = item;
    } else if (!isa<arith::ConstantOp, neura::YieldOp>(&operation))
      return reject(&operation, "application K kernel body differs");
  }
  if (neuraCounters.size() != 3 || loads.size() != 2 || stores.size() != 1 ||
      !cmp || !mul || !add || !init || !init.getElseRegion().empty() ||
      !init.getThenRegion().hasOneBlock())
    return reject(kernel, "application K recurrence shape differs");
  for (auto [index, counter] : llvm::enumerate(neuraCounters)) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    if (!id || id.getInt() != static_cast<int64_t>(index) ||
        constantIndex(counter.getLowerBound()) != 0 ||
        constantIndex(counter.getStep()) != 1 ||
        ((index == 0 || (task3 && index == 1))
             ? counter.getUpperBound() != outerBound
             : constantIndex(counter.getUpperBound()) != 256))
      return reject(counter, "application K Neura counter bounds differ");
  }
  Value m = neuraCounters[0].getCurrentIndex();
  Value n = neuraCounters[1].getCurrentIndex();
  Value k = neuraCounters[2].getCurrentIndex();
  if (cmp.getPredicate() != arith::CmpIPredicate::eq ||
      cmp.getLhs() != k || constantIndex(cmp.getRhs()) != 0 ||
      init.getCondition() != cmp.getResult() ||
      loads[0].getMemRef() != inputA ||
      !hasIndices(loads[0].getIndices(), m, k) ||
      loads[1].getMemRef() != inputB ||
      !(task3 ? hasIndices(loads[1].getIndices(), n, k)
              : hasIndices(loads[1].getIndices(), k, n)) ||
      mul.getLhs() != loads[0].getResult() ||
      mul.getRhs() != loads[1].getResult() ||
      add.getLhs() != acc || add.getRhs() != mul.getResult() ||
      stores[0].getMemRef() != output ||
      stores[0].getValue() != add.getResult() ||
      !hasIndices(stores[0].getIndices(), m, n))
    return reject(kernel, "application K GEMM recurrence differs");
  Block &thenBody = init.getThenRegion().front();
  auto initStore = dyn_cast<memref::StoreOp>(&thenBody.front());
  if (!initStore || !isa<scf::YieldOp>(thenBody.getTerminator()) ||
      std::next(thenBody.begin()) != std::prev(thenBody.end()) ||
      initStore.getValue() != zero || initStore.getMemRef() != output ||
      !hasIndices(initStore.getIndices(), m, n))
    return reject(init, "application K initial store differs");
  auto yield = dyn_cast<neura::YieldOp>(kernelBody.getTerminator());
  if (!yield || yield.getIterArgsNext().size() != 1 ||
      yield.getResults().size() != 1 ||
      yield.getIterArgsNext()[0] != add.getResult() ||
      yield.getResults()[0] != add.getResult())
    return reject(kernel, "application K scalar yield differs");
  return LlamaApplicationKSource{function, task, kernel, counters[2],
                                 neuraCounters[2], add, init,
                                 selectedTaskName.str(), 320, expectedN, 256,
                                 task3};
}

static FailureOr<TaskflowTaskOp>
cloneLlamaSequentialBlock(OpBuilder &builder, LlamaApplicationKSource &source,
                          Value previousState, unsigned index,
                          unsigned blockCount, int64_t lower, int64_t upper) {
  TaskflowTaskOp original = source.task;
  TaskflowTaskOp outerProducerA;
  TaskflowTaskOp outerProducerB;
  if (source.relayUsesOuterBound) {
    outerProducerA = original.getWillReads()[0].getDefiningOp<TaskflowTaskOp>();
    outerProducerB = original.getWillReads()[1].getDefiningOp<TaskflowTaskOp>();
    if (!outerProducerA || !outerProducerB)
      return reject(original,
                    "application K Task_3 requires Taskflow producer results");
  }
  SmallVector<Value> reads(original.getWillReads());
  SmallVector<Value> originalReads(original.getOriginalReadMemrefs());
  if (previousState) {
    reads.push_back(previousState);
    originalReads.push_back(original.getWillWrites()[0]);
  }
  std::string name =
      (Twine(original.getTaskName()) + ".k." + Twine(index)).str();
  Value output = original.getWillWrites()[0];
  auto task = builder.create<TaskflowTaskOp>(
      original.getLoc(), TypeRange{}, TypeRange{output.getType()}, TypeRange{},
      reads, ValueRange{output}, original.getValueInputs(),
      builder.getStringAttr(name), originalReads,
      original.getOriginalWriteMemrefs());
  Block *body = new Block();
  task.getBody().push_back(body);
  for (Value read : reads)
    body->addArgument(read.getType(), original.getLoc());
  body->addArgument(output.getType(), original.getLoc());
  for (Value input : original.getValueInputs())
    body->addArgument(input.getType(), original.getLoc());
  IRMapping mapping;
  Block &oldBody = original.getBody().front();
  mapping.map(oldBody.getArgument(0), body->getArgument(0));
  mapping.map(oldBody.getArgument(1), body->getArgument(1));
  unsigned shift = previousState ? 1 : 0;
  mapping.map(oldBody.getArgument(2), body->getArgument(2 + shift));
  mapping.map(oldBody.getArgument(3), body->getArgument(3 + shift));
  mapping.map(oldBody.getArgument(4), body->getArgument(4 + shift));
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  for (Operation &operation : oldBody)
    bodyBuilder.clone(operation, mapping);

  TaskflowCounterOp taskK;
  neura::KernelOp kernel;
  for (Operation &operation : *body) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&operation)) {
      auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
      if (id && id.getInt() == 2)
        taskK = counter;
    } else if (auto candidate = dyn_cast<neura::KernelOp>(&operation))
      kernel = candidate;
  }
  if (!taskK || !kernel)
    return reject(task, "application K clone lost K counter or kernel");
  OpBuilder taskBuilder(taskK);
  Value taskLower =
      taskBuilder.create<arith::ConstantIndexOp>(taskK.getLoc(), lower);
  Value taskUpper =
      taskBuilder.create<arith::ConstantIndexOp>(taskK.getLoc(), upper);
  taskK.getLowerBoundMutable().assign(taskLower);
  taskK.getUpperBoundMutable().assign(taskUpper);

  neura::CounterOp kernelK;
  arith::AddIOp accumulator;
  scf::IfOp initialStore;
  for (Operation &operation : kernel.getBody().front()) {
    if (auto counter = dyn_cast<neura::CounterOp>(&operation)) {
      auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
      if (id && id.getInt() == 2)
        kernelK = counter;
    } else if (auto add = dyn_cast<arith::AddIOp>(&operation))
      accumulator = add;
    else if (auto conditional = dyn_cast<scf::IfOp>(&operation))
      initialStore = conditional;
  }
  if (!kernelK || !accumulator || !initialStore)
    return reject(task, "application K clone lost verified recurrence");
  OpBuilder kernelBuilder(kernelK);
  Value kernelLower =
      kernelBuilder.create<arith::ConstantIndexOp>(kernelK.getLoc(), lower);
  Value kernelUpper =
      kernelBuilder.create<arith::ConstantIndexOp>(kernelK.getLoc(), upper);
  kernelK.getLowerBoundMutable().assign(kernelLower);
  kernelK.getUpperBoundMutable().assign(kernelUpper);
  kernelK->setAttr("counter_dynamism",
                   kernelBuilder.getStringAttr("constant_bound"));
  if (kernelK->hasAttr("lower_bound_value"))
    kernelK->setAttr("lower_bound_value",
                     kernelBuilder.getIndexAttr(lower));
  if (kernelK->hasAttr("upper_bound_value"))
    kernelK->setAttr("upper_bound_value",
                     kernelBuilder.getIndexAttr(upper));
  taskK->setAttr("counter_dynamism",
                 taskBuilder.getStringAttr("constant_bound"));

  if (previousState) {
    // The first K iteration of each later block loads the preceding block's
    // per-(M,N) value. Subsequent iterations use the kernel's scalar iter_arg.
    // The load stays inside the K body so initialization is per output cell,
    // rather than once before the M/N loops in the host lowering.
    initialStore.erase();
    Block &kernelBody = kernel.getBody().front();
    SmallVector<neura::CounterOp> counters;
    for (Operation &operation : kernelBody)
      if (auto counter = dyn_cast<neura::CounterOp>(&operation))
        counters.push_back(counter);
    if (counters.size() != 3)
      return reject(kernel, "application K clone lost M/N counters");
    OpBuilder atAdd(accumulator);
    Value cell = atAdd.create<memref::LoadOp>(
        accumulator.getLoc(), kernelBody.getArgument(1),
        ValueRange{counters[0].getCurrentIndex(),
                   counters[1].getCurrentIndex()});
    Value isFirst = atAdd.create<arith::CmpIOp>(
        accumulator.getLoc(), arith::CmpIPredicate::eq,
        kernelK.getCurrentIndex(), kernelLower);
    Value startingValue = atAdd.create<arith::SelectOp>(
        accumulator.getLoc(), isFirst, cell, kernelBody.getArgument(5));
    accumulator->setOperand(0, startingValue);
  }

  copyTaskAttributes(original, task, name);
  markTask(builder, task, "gemm_producer", "producer", "sequential", "none");
  task->setAttr("amoeba.semantic.k_tiled", builder.getUnitAttr());
  task->setAttr("amoeba.semantic.k_block_index",
                builder.getI64IntegerAttr(index));
  task->setAttr("amoeba.semantic.k_block_count",
                builder.getI64IntegerAttr(blockCount));
  task->setAttr("amoeba.semantic.initial_accumulator",
                builder.getStringAttr(index == 0 ? "zero_once" : "state_carry"));
  setRangeAttrs(builder, task, {0, source.M}, {0, source.N}, {lower, upper});
  SmallVector<StaticRegion> inputRegions{
      {{0, lower}, {source.M, upper}},
      source.relayUsesOuterBound ? StaticRegion{{0, lower},
                                                {source.N, upper}}
                                 : StaticRegion{{lower, 0},
                                                {upper, source.N}}};
  if (previousState)
    inputRegions.push_back(fullRegion(source.M, source.N));
  setRegionContracts(builder, task, inputRegions,
                     {fullRegion(source.M, source.N)}, "semantic_k_tile");
  // Task_1 and Task_2 receive dynamically typed external operands, even
  // though the application matcher checked their caller-shape attributes.
  // Task_3 receives statically typed producer results, so retain its exact
  // regions for contract checking. In both cases retain an exact state-carry
  // region for the internal producer-consumer edge.
  if (!source.relayUsesOuterBound) {
    auto lowerAttrs = task->getAttrOfType<ArrayAttr>(kInputRegionLowersAttr);
    auto upperAttrs = task->getAttrOfType<ArrayAttr>(kInputRegionUppersAttr);
    auto reasonAttrs = task->getAttrOfType<ArrayAttr>(kInputRegionReasonsAttr);
    SmallVector<Attribute> lowerValues(lowerAttrs.begin(), lowerAttrs.end());
    SmallVector<Attribute> upperValues(upperAttrs.begin(), upperAttrs.end());
    SmallVector<Attribute> reasonValues(reasonAttrs.begin(), reasonAttrs.end());
    for (unsigned input = 0; input < 2; ++input) {
      lowerValues[input] = builder.getUnitAttr();
      upperValues[input] = builder.getUnitAttr();
      reasonValues[input] = builder.getStringAttr("dynamic_external");
    }
    task->setAttr(kInputRegionLowersAttr, builder.getArrayAttr(lowerValues));
    task->setAttr(kInputRegionUppersAttr, builder.getArrayAttr(upperValues));
    task->setAttr(kInputRegionReasonsAttr, builder.getArrayAttr(reasonValues));
  }
  // Task_3 retains its two real producer-result operands on every K shard.
  // The source tasks have no output-region contract in the pinned application
  // graph, so these edges are tensor-wide; the exact A[m,k]/B[n,k] transfer
  // regions remain on this clone's input-region attributes above.  Later
  // shards also consume the previous shard's exact state-carry tile.
  SmallVector<std::tuple<StringRef, StringRef, StringRef>> incomingEdges;
  if (source.relayUsesOuterBound) {
    incomingEdges.push_back(
        {outerProducerA.getTaskName(), "producer_consumer", "tensor_wide"});
    incomingEdges.push_back(
        {outerProducerB.getTaskName(), "producer_consumer", "tensor_wide"});
  }
  std::string previousTaskName;
  if (previousState) {
    previousTaskName =
        (Twine(original.getTaskName()) + ".k." + Twine(index - 1)).str();
    incomingEdges.push_back({previousTaskName, "state_carry", "tile_local"});
  }
  setIncomingEdges(builder, task, incomingEdges);
  return task;
}

static LogicalResult runLlamaApplicationSequentialK(ModuleOp module,
                                                       StringRef functionName,
                                                       StringRef taskName,
                                                       int64_t tileSize,
                                                       int64_t factor) {
  FailureOr<LlamaApplicationKSource> source =
      analyzeLlamaApplicationK(module, functionName, taskName);
  if (failed(source))
    return failure();
  FailureOr<int64_t> width = resolveWidth(source->K, tileSize, factor, true);
  if (failed(width) || *width != 64)
    return reject(source->task,
                  (Twine("LLaMA ") + taskName +
                   " sequential K currently requires width 64")
                      .str());
  unsigned blockCount = static_cast<unsigned>(source->K / *width);
  for (unsigned index = 0; index < blockCount; ++index) {
    std::string name =
        (Twine(source->task.getTaskName()) + ".k." + Twine(index)).str();
    bool collision = false;
    source->function.walk([&](TaskflowTaskOp task) {
      collision |= task != source->task && task.getTaskName() == name;
    });
    if (collision)
      return reject(source->task, "application K derived task name collides");
  }
  OpBuilder builder(source->task);
  Value previousState;
  TaskflowTaskOp finalBlock;
  for (unsigned index = 0; index < blockCount; ++index) {
    FailureOr<TaskflowTaskOp> block = cloneLlamaSequentialBlock(
        builder, *source, previousState, index, blockCount,
        index * *width, (index + 1) * *width);
    if (failed(block))
      return failure();
    finalBlock = *block;
    previousState = finalBlock.getDoneWrites().front();
  }
  source->task.getDoneWrites().front().replaceAllUsesWith(previousState);
  source->function->setAttr(kKPolicyAttr, builder.getStringAttr("sequential"));
  source->function->setAttr(kRewriteSchemaAttr,
                             builder.getStringAttr(kRewriteSchema));
  source->task.erase();
  return success();
}

} // namespace

namespace mlir::amoeba::neura {

struct MaterializeNeuraKReductionPass
    : public PassWrapper<MaterializeNeuraKReductionPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeNeuraKReductionPass)
  MaterializeNeuraKReductionPass() = default;
  MaterializeNeuraKReductionPass(const MaterializeNeuraKReductionPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "materialize-neura-k-reduction";
  }
  StringRef getDescription() const override {
    return "Materialize canonical post-Neura static i32 GEMM K reduction";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, ::mlir::neura::NeuraDialect,
                    scf::SCFDialect, TaskflowDialect>();
  }

  Option<std::string> functionName{
      *this, "function", llvm::cl::desc("Taskflow function name"),
      llvm::cl::init("")};
  Option<std::string> kMode{
      *this, "k-mode",
      llvm::cl::desc("sequential, parallel-linear, or parallel-tree"),
      llvm::cl::init("sequential")};
  Option<int64_t> tileSize{
      *this, "tile-size", llvm::cl::desc("K block width"),
      llvm::cl::init(0)};
  Option<int64_t> kFactor{
      *this, "k-factor", llvm::cl::desc("Number of K blocks"),
      llvm::cl::init(0)};
  Option<std::string> taskName{
      *this, "task-name",
      llvm::cl::desc("Select the checked LLaMA Task_1/Task_2/Task_3 application recurrence"),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LogicalResult result = failure();
    if (!taskName.empty()) {
      if ((taskName == "Task_1" || taskName == "Task_2" ||
           taskName == "Task_3") &&
          kMode == "sequential")
        result = runLlamaApplicationSequentialK(module, functionName, taskName,
                                                tileSize, kFactor);
      else if (kMode == "sequential")
        // Memory-carried Taskflow reductions (LLaMA Task_8, LU Task_6,
        // GCN's in-place reduction tasks) have no SSA iter_arg. Route them
        // through the strict state-carry adapter, which rejects any graph
        // that does not prove the canonical or ordered recurrence shape.
        result = runMemoryFeedbackSequentialK(module, functionName, taskName,
                                               tileSize, kFactor);
      else {
        module.emitError()
            << "application/memory-feedback K supports only sequential mode";
        result = failure();
      }
    } else if (kMode == "sequential")
      result = runSequential(module, functionName, tileSize, kFactor);
    else if (kMode == "parallel-linear")
      result = runParallel(module, functionName, tileSize, kFactor, "linear");
    else if (kMode == "parallel-tree")
      result = runParallel(module, functionName, tileSize, kFactor,
                           "balanced_tree");
    else {
      module.emitError()
          << "post-Neura K reduction mode must be sequential, "
             "parallel-linear, or parallel-tree";
      result = failure();
    }
    if (failed(result))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createMaterializeNeuraKReductionPass() {
  return std::make_unique<MaterializeNeuraKReductionPass>();
}

} // namespace mlir::amoeba::neura
