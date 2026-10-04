//===- FissionRayCarriedReductionPass.cpp --------------------*- C++ -*-===//
//
// Source-owned fission for the exact Ray Task_13 carried min/index reduction.
// The pass partitions the original lane range into two ordered stages while
// preserving every outer firing, the carried state, and the final stores.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {

constexpr StringLiteral kCallerNoAliasAttr =
    "amoeba.input0_caller_noalias_proven";
constexpr StringLiteral kCallerStaticBoundAttr =
    "amoeba.input0_caller_static_bound";
constexpr StringLiteral kCallerProofModeAttr =
    "amoeba.input0_caller_noalias_proof_mode";
constexpr StringLiteral kCallerEvidencePathAttr =
    "amoeba.input0_caller_evidence_path";
constexpr StringLiteral kPreparedInputPathAttr =
    "amoeba.input0_caller_prepared_input_path";
constexpr StringLiteral kLogicalTransferShapeAttr =
    "amoeba.logical_transfer_shape";
constexpr StringLiteral kFissionSchemaAttr = "amoeba.fission.schema";
constexpr StringLiteral kFissionSourceTaskAttr = "amoeba.fission.source_task";
constexpr StringLiteral kFissionStageAttr = "amoeba.fission.stage";
constexpr StringLiteral kFissionIntervalAttr = "amoeba.fission.lane_interval";
constexpr StringLiteral kFissionParentWorkAttr =
    "amoeba.fission.parent_source_work_count";
constexpr StringLiteral kFissionStageWorkAttr =
    "amoeba.fission.stage_source_work_count";
constexpr StringLiteral kFissionStateAttr = "amoeba.fission.state_link";
constexpr StringLiteral kFissionSchema = "ray-carried-min-index-fission-v1";

struct RayReductionFissionPlan {
  TaskflowTaskOp task;
  func::FuncOp function;
  affine::AffineForOp outer;
  affine::AffineForOp inner;
  affine::AffineLoadOp seedLoad;
  arith::CmpIOp seedCompare;
  arith::SelectOp seedSelect;
  affine::AffineStoreOp indexStore;
  affine::AffineStoreOp bestStore;
  int64_t outerTripCount = 0;
  int64_t laneCount = 0;
  int64_t splitAt = 0;
};

static bool isFunctionArgument(Value value, func::FuncOp function,
                               unsigned &argumentIndex) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || function.getBody().empty() ||
      argument.getOwner() != &function.getBody().front())
    return false;
  argumentIndex = argument.getArgNumber();
  return true;
}

static bool hasFunctionArgumentAttribute(Value value, func::FuncOp function,
                                         StringRef name) {
  unsigned index = 0;
  return isFunctionArgument(value, function, index) &&
         static_cast<bool>(function.getArgAttr(index, name));
}

static bool hasLogicalShape(Value value, func::FuncOp function,
                            ArrayRef<int64_t> expected) {
  unsigned index = 0;
  if (!isFunctionArgument(value, function, index))
    return false;
  auto shape = dyn_cast_or_null<DenseI64ArrayAttr>(
      function.getArgAttr(index, kLogicalTransferShapeAttr));
  return shape && shape.asArrayRef() == expected;
}

static bool taskResultHasSourceRoot(Value state, Value sourceRoot) {
  if (state == sourceRoot)
    return true;
  auto producer = state.getDefiningOp<TaskflowTaskOp>();
  if (!producer)
    return false;
  ValueRange results = producer.getDoneWrites();
  ValueRange roots = producer.getOriginalWriteMemrefs();
  for (auto [index, result] : llvm::enumerate(results))
    if (result == state && index < roots.size() && roots[index] == sourceRoot)
      return true;
  return false;
}

static bool isCmp(arith::CmpIOp compare, arith::CmpIPredicate predicate,
                  Value lhs, Value rhs) {
  return compare && compare.getPredicate() == predicate &&
         compare.getLhs() == lhs && compare.getRhs() == rhs;
}

static FailureOr<RayReductionFissionPlan>
analyzeRayReductionFission(TaskflowTaskOp task, func::FuncOp function,
                           int64_t splitAt, std::string &error) {
  auto reject = [&](StringRef reason) -> FailureOr<RayReductionFissionPlan> {
    error = reason.str();
    return failure();
  };
  if (!function->hasAttr(kCallerNoAliasAttr))
    return reject("requires a source-owned caller noalias proof");
  auto proofMode = function->getAttrOfType<StringAttr>(kCallerProofModeAttr);
  auto evidencePath =
      function->getAttrOfType<StringAttr>(kCallerEvidencePathAttr);
  auto preparedInputPath =
      function->getAttrOfType<StringAttr>(kPreparedInputPathAttr);
  if (!proofMode || proofMode.getValue() != "prepared-external-caller" ||
      !evidencePath || evidencePath.getValue().empty() || !preparedInputPath ||
      preparedInputPath.getValue().empty())
    return reject("requires prepared-external-caller proof provenance");
  auto staticBound =
      function->getAttrOfType<IntegerAttr>(kCallerStaticBoundAttr);
  if (!staticBound || staticBound.getInt() <= 0)
    return reject("requires a positive caller-proven outer bound");

  if (!task.getDoneReads().empty() || task.getWillReads().size() != 2 ||
      task.getWillWrites().size() != 2 || task.getDoneWrites().size() != 2 ||
      task.getValueInputs().size() != 4 || !task.getValueOutputs().empty() ||
      task.getOriginalReadMemrefs().size() != 2 ||
      task.getOriginalWriteMemrefs().size() != 2 ||
      !task.getBody().hasOneBlock())
    return reject(
        "requires two reads, two writes, four scalar inputs, and one block");
  for (NamedAttribute attribute : task->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name != "task_name" && name != "operandSegmentSizes" &&
        name != "resultSegmentSizes" && name != "operand_segment_sizes" &&
        name != "result_segment_sizes")
      return reject("refuses to discard unsupported source task metadata");
  }

  Block &taskBody = task.getBody().front();
  if (taskBody.getNumArguments() != 8)
    return reject(
        "task execution arguments do not match the Ray reduction interface");
  for (unsigned index = 0; index < 8; ++index) {
    Value expected = index < 2   ? task.getWillReads()[index]
                     : index < 4 ? task.getWillWrites()[index - 2]
                                 : task.getValueInputs()[index - 4];
    if (taskBody.getArgument(index).getType() != expected.getType())
      return reject("task execution argument types are inconsistent");
  }
  if (!task.getValueInputs()[0].getType().isa<IndexType>() ||
      !task.getValueInputs()[1].getType().isInteger(32) ||
      !task.getValueInputs()[2].getType().isInteger(32) ||
      !task.getValueInputs()[3].getType().isInteger(32))
    return reject("requires index bound and three i32 reduction values");

  SmallVector<Value> sourceRoots(task.getOriginalReadMemrefs());
  llvm::append_range(sourceRoots, task.getOriginalWriteMemrefs());
  llvm::SmallDenseSet<Value, 4> uniqueRoots;
  for (Value root : sourceRoots) {
    if (!uniqueRoots.insert(root).second ||
        !hasFunctionArgumentAttribute(root, function, "amoeba.noalias"))
      return reject(
          "caller proof does not establish four disjoint source buffers");
  }
  const int64_t outerBound = staticBound.getInt();
  if (!hasLogicalShape(sourceRoots[0], function, {outerBound}) ||
      !hasLogicalShape(sourceRoots[1], function, {outerBound, 8}) ||
      !hasLogicalShape(sourceRoots[2], function, {outerBound}) ||
      !hasLogicalShape(sourceRoots[3], function, {outerBound}))
    return reject(
        "caller-proven logical buffer shapes do not match the Ray reduction");
  if (!taskResultHasSourceRoot(task.getWillReads()[0], sourceRoots[0]) ||
      !taskResultHasSourceRoot(task.getWillReads()[1], sourceRoots[1]) ||
      task.getWillReads()[0].getType() != sourceRoots[0].getType() ||
      task.getWillReads()[1].getType() != sourceRoots[1].getType() ||
      task.getWillWrites()[0] != sourceRoots[2] ||
      task.getWillWrites()[1] != sourceRoots[3])
    return reject(
        "Taskflow memory states do not bind to the proved source buffers");

  auto inputType0 = dyn_cast<MemRefType>(task.getWillReads()[0].getType());
  auto inputType1 = dyn_cast<MemRefType>(task.getWillReads()[1].getType());
  auto outputType0 = dyn_cast<MemRefType>(task.getWillWrites()[0].getType());
  auto outputType1 = dyn_cast<MemRefType>(task.getWillWrites()[1].getType());
  if (!inputType0 || !inputType1 || !outputType0 || !outputType1 ||
      inputType0.getRank() != 1 || inputType1.getRank() != 2 ||
      inputType1.getDimSize(1) != 8 || outputType0.getRank() != 1 ||
      outputType1.getRank() != 1 ||
      !inputType0.getElementType().isInteger(32) ||
      !inputType1.getElementType().isInteger(32) ||
      !outputType0.getElementType().isInteger(32) ||
      !outputType1.getElementType().isInteger(32))
    return reject(
        "Taskflow state types do not match the rank-one/rank-two i32 buffers");
  for (MemRefType type : {inputType0, inputType1, outputType0, outputType1}) {
    Attribute space = type.getMemorySpace();
    auto numericSpace = dyn_cast_or_null<IntegerAttr>(space);
    if (!type.getLayout().isIdentity() ||
        (space && (!numericSpace || numericSpace.getInt() != 0)))
      return reject("requires identity buffer layouts in the default memory "
                    "space");
  }

  if (task.getNumResults() != 2 || taskBody.empty() ||
      !isa<TaskflowYieldOp>(taskBody.getTerminator()))
    return reject("task has no two-state Taskflow result yield");
  auto taskYield = cast<TaskflowYieldOp>(taskBody.getTerminator());
  if (!taskYield.getDoneReads().empty() ||
      taskYield.getDoneWrites().size() != 2 ||
      taskYield.getDoneWrites()[0] != taskBody.getArgument(2) ||
      taskYield.getDoneWrites()[1] != taskBody.getArgument(3) ||
      !taskYield.getValueResults().empty())
    return reject("task yield does not return both original output states");

  if (std::distance(taskBody.begin(), taskBody.end()) != 2)
    return reject("task body must contain only one outer reduction loop");
  auto outer = dyn_cast<affine::AffineForOp>(&taskBody.front());
  if (!outer || outer.getNumResults() != 0 || outer.getNumIterOperands() != 0 ||
      !outer.hasConstantLowerBound() || outer.getConstantLowerBound() != 0 ||
      outer.getStepAsInt() != 1)
    return reject("requires one zero-based, unit-step outer affine loop");
  int64_t outerLower = 0;
  int64_t outerUpper = 0;
  int64_t outerStep = 0;
  std::optional<int64_t> tripCount =
      sourceAffineTripCount(outer, task, outerLower, outerUpper, outerStep);
  if (!tripCount || outerLower != 0 || outerStep != 1 ||
      *tripCount != outerBound || outerUpper != outerBound ||
      outerBound > std::numeric_limits<int64_t>::max() / 8)
    return reject(
        "outer loop extent is not bound to the caller-proven source domain");

  Block &outerBody = *outer.getBody();
  if (std::distance(outerBody.begin(), outerBody.end()) != 7)
    return reject("outer loop is not the exact seed/reduction/store sequence");
  auto operation = outerBody.begin();
  auto seedLoad = dyn_cast<affine::AffineLoadOp>(&*operation++);
  auto seedCompare = dyn_cast<arith::CmpIOp>(&*operation++);
  auto seedSelect = dyn_cast<arith::SelectOp>(&*operation++);
  auto inner = dyn_cast<affine::AffineForOp>(&*operation++);
  auto indexStore = dyn_cast<affine::AffineStoreOp>(&*operation++);
  auto bestStore = dyn_cast<affine::AffineStoreOp>(&*operation++);
  if (!seedLoad || !seedCompare || !seedSelect || !inner || !indexStore ||
      !bestStore || !isa<affine::AffineYieldOp>(outerBody.getTerminator()) ||
      seedLoad.getMemRef() != taskBody.getArgument(0) ||
      seedLoad.getIndices().size() != 1 ||
      seedLoad.getIndices().front() != outer.getInductionVar() ||
      !isCmp(seedCompare, arith::CmpIPredicate::sgt, seedLoad.getResult(),
             taskBody.getArgument(5)) ||
      seedSelect.getCondition() != seedCompare.getResult() ||
      seedSelect.getTrueValue() != seedLoad.getResult() ||
      seedSelect.getFalseValue() != taskBody.getArgument(6))
    return reject(
        "seed must select input0 when it exceeds the proved threshold");

  if (inner.getNumResults() != 2 || inner.getNumIterOperands() != 2 ||
      inner.getInits()[0] != seedSelect.getResult() ||
      inner.getInits()[1] != taskBody.getArgument(7) ||
      !inner.hasConstantLowerBound() || inner.getConstantLowerBound() != 0 ||
      !inner.hasConstantUpperBound() || inner.getConstantUpperBound() != 8 ||
      inner.getStepAsInt() != 1)
    return reject(
        "inner loop must carry min/index across the complete [0,8) lane range");
  if (splitAt <= 0 || splitAt >= 8)
    return reject(
        "split-at must lie strictly inside the source lane range [0,8)");

  Block &innerBody = *inner.getBody();
  if (std::distance(innerBody.begin(), innerBody.end()) != 5)
    return reject("inner loop is not the exact load/filter/update sequence");
  operation = innerBody.begin();
  auto laneCast = dyn_cast<arith::IndexCastOp>(&*operation++);
  auto matrixLoad = dyn_cast<affine::AffineLoadOp>(&*operation++);
  auto candidateCompare = dyn_cast<arith::CmpIOp>(&*operation++);
  auto branch = dyn_cast<scf::IfOp>(&*operation++);
  auto innerYield = dyn_cast<affine::AffineYieldOp>(innerBody.getTerminator());
  if (!laneCast || !matrixLoad || !candidateCompare || !branch || !innerYield ||
      laneCast.getIn() != inner.getInductionVar() ||
      !laneCast.getResult().getType().isInteger(32) ||
      matrixLoad.getMemRef() != taskBody.getArgument(1) ||
      matrixLoad.getIndices().size() != 2 ||
      matrixLoad.getIndices()[0] != outer.getInductionVar() ||
      matrixLoad.getIndices()[1] != inner.getInductionVar() ||
      !isCmp(candidateCompare, arith::CmpIPredicate::sgt,
             matrixLoad.getResult(), taskBody.getArgument(5)) ||
      branch.getCondition() != candidateCompare.getResult() ||
      branch.getNumResults() != 2 || innerYield.getNumOperands() != 2 ||
      innerYield.getOperand(0) != branch.getResult(0) ||
      innerYield.getOperand(1) != branch.getResult(1))
    return reject("inner lane operations do not preserve thresholded min/index "
                  "semantics");

  if (!branch.getThenRegion().hasOneBlock() ||
      !branch.getElseRegion().hasOneBlock())
    return reject(
        "reduction branch must have explicit one-block then and else regions");
  Block &thenBody = branch.getThenRegion().front();
  Block &elseBody = branch.getElseRegion().front();
  if (std::distance(thenBody.begin(), thenBody.end()) != 4 ||
      std::distance(elseBody.begin(), elseBody.end()) != 1)
    return reject(
        "reduction branch contains unsupported effects or operations");
  auto thenOperation = thenBody.begin();
  auto updateCompare = dyn_cast<arith::CmpIOp>(&*thenOperation++);
  auto bestSelect = dyn_cast<arith::SelectOp>(&*thenOperation++);
  auto indexSelect = dyn_cast<arith::SelectOp>(&*thenOperation++);
  auto thenYield = dyn_cast<scf::YieldOp>(thenBody.getTerminator());
  auto elseYield = dyn_cast<scf::YieldOp>(elseBody.getTerminator());
  auto carried = inner.getRegionIterArgs();
  if (!updateCompare || !bestSelect || !indexSelect || !thenYield ||
      !elseYield ||
      !isCmp(updateCompare, arith::CmpIPredicate::slt, matrixLoad.getResult(),
             carried[0]) ||
      bestSelect.getCondition() != updateCompare.getResult() ||
      bestSelect.getTrueValue() != matrixLoad.getResult() ||
      bestSelect.getFalseValue() != carried[0] ||
      indexSelect.getCondition() != updateCompare.getResult() ||
      indexSelect.getTrueValue() != laneCast.getResult() ||
      indexSelect.getFalseValue() != carried[1] ||
      thenYield.getNumOperands() != 2 ||
      thenYield.getOperand(0) != bestSelect.getResult() ||
      thenYield.getOperand(1) != indexSelect.getResult() ||
      elseYield.getNumOperands() != 2 ||
      elseYield.getOperand(0) != carried[0] ||
      elseYield.getOperand(1) != carried[1])
    return reject("carried state must update only for a strictly smaller "
                  "qualifying lane");

  auto indexStoreIndices = indexStore.getIndices();
  auto bestStoreIndices = bestStore.getIndices();
  if (indexStore.getMemRef() != taskBody.getArgument(2) ||
      bestStore.getMemRef() != taskBody.getArgument(3) ||
      indexStore.getValueToStore() != inner.getResult(1) ||
      bestStore.getValueToStore() != inner.getResult(0) ||
      indexStoreIndices.size() != 1 || bestStoreIndices.size() != 1 ||
      indexStoreIndices.front() != outer.getInductionVar() ||
      bestStoreIndices.front() != outer.getInductionVar())
    return reject("reduction epilogue must store the carried index and best "
                  "value by row");

  RayReductionFissionPlan plan;
  plan.task = task;
  plan.function = function;
  plan.outer = outer;
  plan.inner = inner;
  plan.seedLoad = seedLoad;
  plan.seedCompare = seedCompare;
  plan.seedSelect = seedSelect;
  plan.indexStore = indexStore;
  plan.bestStore = bestStore;
  plan.outerTripCount = *tripCount;
  plan.laneCount = 8;
  plan.splitAt = splitAt;
  return plan;
}

static void mapTaskWritesAndValues(TaskflowTaskOp source, Block &targetBody,
                                   ArrayRef<unsigned> writeMap,
                                   unsigned valueOffset, IRMapping &mapping) {
  Block &sourceBody = source.getBody().front();
  unsigned sourceWriteOffset = source.getWillReads().size();
  for (auto [index, targetIndex] : llvm::enumerate(writeMap))
    mapping.map(sourceBody.getArgument(sourceWriteOffset + index),
                targetBody.getArgument(targetIndex));
  unsigned sourceValueOffset =
      sourceWriteOffset + source.getWillWrites().size();
  for (unsigned index = 0; index < source.getValueInputs().size(); ++index)
    mapping.map(sourceBody.getArgument(sourceValueOffset + index),
                targetBody.getArgument(valueOffset + index));
}

static void addStageAttributes(TaskflowTaskOp task, OpBuilder &builder,
                               StringRef sourceName, unsigned stage,
                               int64_t lower, int64_t upper, int64_t parentWork,
                               int64_t stageWork) {
  task->setAttr(kFissionSchemaAttr, builder.getStringAttr(kFissionSchema));
  task->setAttr(kFissionSourceTaskAttr, builder.getStringAttr(sourceName));
  task->setAttr(kFissionStageAttr, builder.getI64IntegerAttr(stage));
  task->setAttr(kFissionIntervalAttr,
                builder.getDenseI64ArrayAttr({lower, upper}));
  task->setAttr(kFissionParentWorkAttr, builder.getI64IntegerAttr(parentWork));
  task->setAttr(kFissionStageWorkAttr, builder.getI64IntegerAttr(stageWork));
  task->setAttr(kFissionStateAttr,
                builder.getStringAttr("taskflow-raw:row-index,row-best"));
}

static LogicalResult buildStageBody(
    TaskflowTaskOp source, TaskflowTaskOp target,
    affine::AffineForOp sourceOuter, affine::AffineForOp sourceInner,
    affine::AffineLoadOp sourceSeedLoad, arith::CmpIOp sourceSeedCompare,
    arith::SelectOp sourceSeedSelect, affine::AffineStoreOp sourceIndexStore,
    affine::AffineStoreOp sourceBestStore, unsigned stage, int64_t splitAt) {
  Block *body = new Block();
  target.getBody().push_back(body);
  SmallVector<Value> executionInputs(target.getWillReads());
  llvm::append_range(executionInputs, target.getWillWrites());
  llvm::append_range(executionInputs, target.getValueInputs());
  for (Value input : executionInputs)
    body->addArgument(input.getType(), source.getLoc());

  IRMapping mapping;
  if (stage == 0) {
    mapping.map(source.getBody().front().getArgument(0), body->getArgument(0));
    mapping.map(source.getBody().front().getArgument(1), body->getArgument(1));
    mapTaskWritesAndValues(source, *body, {2, 3}, 4, mapping);
  } else {
    // Source input zero is no longer needed after stage zero. The remaining
    // matrix input is target read argument two, after the two state arrays.
    mapping.map(source.getBody().front().getArgument(1), body->getArgument(2));
    mapTaskWritesAndValues(source, *body, {3, 4}, 5, mapping);
  }

  auto remapOperands = [&](ValueRange values) {
    SmallVector<Value> mapped;
    mapped.reserve(values.size());
    for (Value value : values)
      mapped.push_back(mapping.lookupOrDefault(value));
    return mapped;
  };
  SmallVector<Value> lowerOperands =
      remapOperands(sourceOuter.getLowerBoundOperands());
  SmallVector<Value> upperOperands =
      remapOperands(sourceOuter.getUpperBoundOperands());
  OpBuilder taskBodyBuilder = OpBuilder::atBlockEnd(body);
  affine::AffineForOp outer = taskBodyBuilder.create<affine::AffineForOp>(
      source.getLoc(), lowerOperands, sourceOuter.getLowerBoundMap(),
      upperOperands, sourceOuter.getUpperBoundMap(),
      sourceOuter.getStepAsInt());
  mapping.map(sourceOuter.getInductionVar(), outer.getInductionVar());

  Block &outerBody = *outer.getBody();
  OpBuilder loopBuilder = OpBuilder::atBlockBegin(&outerBody);
  if (stage == 0) {
    loopBuilder.clone(*sourceSeedLoad, mapping);
    loopBuilder.clone(*sourceSeedCompare, mapping);
    loopBuilder.clone(*sourceSeedSelect, mapping);
  } else {
    Value scratchIndex = body->getArgument(0);
    Value scratchBest = body->getArgument(1);
    Value best = loopBuilder.create<affine::AffineLoadOp>(
        source.getLoc(), scratchBest, ValueRange{outer.getInductionVar()});
    Value index = loopBuilder.create<affine::AffineLoadOp>(
        source.getLoc(), scratchIndex, ValueRange{outer.getInductionVar()});
    mapping.map(sourceInner.getInits()[0], best);
    mapping.map(sourceInner.getInits()[1], index);
  }

  auto inner =
      cast<affine::AffineForOp>(loopBuilder.clone(*sourceInner, mapping));
  if (stage == 0) {
    inner.setConstantLowerBound(0);
    inner.setConstantUpperBound(splitAt);
  } else {
    inner.setConstantLowerBound(splitAt);
    inner.setConstantUpperBound(8);
  }
  loopBuilder.clone(*sourceIndexStore, mapping);
  loopBuilder.clone(*sourceBestStore, mapping);

  OpBuilder taskBuilder = OpBuilder::atBlockEnd(body);
  SmallVector<Value> outputs;
  unsigned writeOffset = target.getWillReads().size();
  for (unsigned index = 0; index < target.getWillWrites().size(); ++index)
    outputs.push_back(body->getArgument(writeOffset + index));
  taskBuilder.create<TaskflowYieldOp>(source.getLoc(), ValueRange{}, outputs,
                                      ValueRange{});
  return success();
}

static LogicalResult rewriteRayReductionFission(RayReductionFissionPlan &plan,
                                                StringRef taskName) {
  TaskflowTaskOp source = plan.task;
  func::FuncOp function = plan.function;
  std::string prefixName = (Twine(taskName) + ".fission.0").str();
  bool collision = false;
  function.walk([&](TaskflowTaskOp task) {
    collision |= task != source && task.getTaskName() == prefixName;
  });
  if (collision) {
    source.emitError(
        "fission-derived task name collides with an existing task");
    return failure();
  }

  OpBuilder builder(source);
  auto scratchType =
      MemRefType::get({plan.outerTripCount}, builder.getI32Type());
  Value rowIndex =
      builder.create<memref::AllocOp>(source.getLoc(), scratchType);
  Value rowBest = builder.create<memref::AllocOp>(source.getLoc(), scratchType);
  SmallVector<Value> scratch{rowIndex, rowBest};
  SmallVector<Type> prefixResultTypes{scratchType, scratchType};
  TaskflowTaskOp prefix = builder.create<TaskflowTaskOp>(
      source.getLoc(), TypeRange{}, prefixResultTypes, TypeRange{},
      source.getWillReads(), scratch, source.getValueInputs(),
      builder.getStringAttr(prefixName), source.getOriginalReadMemrefs(),
      scratch);

  SmallVector<Value> suffixReads(prefix.getDoneWrites());
  suffixReads.push_back(source.getWillReads()[1]);
  SmallVector<Value> suffixOriginalReads{rowIndex, rowBest,
                                         source.getOriginalReadMemrefs()[1]};
  SmallVector<Type> suffixResultTypes(source.getDoneWrites().getTypes());
  TaskflowTaskOp suffix = builder.create<TaskflowTaskOp>(
      source.getLoc(), TypeRange{}, suffixResultTypes, TypeRange{}, suffixReads,
      source.getWillWrites(), source.getValueInputs(),
      builder.getStringAttr(taskName), suffixOriginalReads,
      source.getOriginalWriteMemrefs());

  if (failed(buildStageBody(source, prefix, plan.outer, plan.inner,
                            plan.seedLoad, plan.seedCompare, plan.seedSelect,
                            plan.indexStore, plan.bestStore,
                            /*stage=*/0, plan.splitAt)) ||
      failed(buildStageBody(source, suffix, plan.outer, plan.inner,
                            plan.seedLoad, plan.seedCompare, plan.seedSelect,
                            plan.indexStore, plan.bestStore,
                            /*stage=*/1, plan.splitAt)))
    return failure();

  int64_t parentWork = plan.outerTripCount * plan.laneCount;
  addStageAttributes(prefix, builder, taskName, 0, 0, plan.splitAt, parentWork,
                     plan.outerTripCount * plan.splitAt);
  addStageAttributes(suffix, builder, taskName, 1, plan.splitAt, plan.laneCount,
                     parentWork,
                     plan.outerTripCount * (plan.laneCount - plan.splitAt));

  for (auto [oldResult, newResult] :
       llvm::zip(source.getDoneWrites(), suffix.getDoneWrites()))
    oldResult.replaceAllUsesWith(newResult);
  source.erase();
  return success();
}

struct FissionRayCarriedReductionPass
    : public PassWrapper<FissionRayCarriedReductionPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FissionRayCarriedReductionPass)
  FissionRayCarriedReductionPass() = default;
  FissionRayCarriedReductionPass(const FissionRayCarriedReductionPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final {
    return "fission-ray-carried-reduction";
  }
  StringRef getDescription() const final {
    return "Split the exact Ray Task_13 min/index recurrence with owned state";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, arith::ArithDialect, func::FuncDialect,
                memref::MemRefDialect, scf::SCFDialect, TaskflowDialect>();
  }

  Option<std::string> taskName{*this, "task-name",
                               llvm::cl::desc("Exact source task name"),
                               llvm::cl::init("Task_13")};
  Option<int64_t> splitAt{*this, "split-at",
                          llvm::cl::desc("Exclusive prefix lane bound"),
                          llvm::cl::init(4)};

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    SmallVector<TaskflowTaskOp> matches;
    function.walk([&](TaskflowTaskOp task) {
      if (task.getTaskName() == taskName)
        matches.push_back(task);
    });
    if (matches.empty())
      return;
    if (matches.size() != 1) {
      function.emitError()
          << "fission-ray-carried-reduction requires exactly one task named "
          << taskName;
      return signalPassFailure();
    }

    std::string error;
    FailureOr<RayReductionFissionPlan> plan =
        analyzeRayReductionFission(matches.front(), function, splitAt, error);
    if (failed(plan)) {
      matches.front().emitError() << "fission-ray-carried-reduction: " << error;
      return signalPassFailure();
    }
    if (failed(rewriteRayReductionFission(*plan, taskName)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::amoeba::neura::createFissionRayCarriedReductionPass() {
  return std::make_unique<FissionRayCarriedReductionPass>();
}
