#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "Backend/Neura/NeuraBackendPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <cstddef>
#include <utility>
using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {

//==============================================================================
// Innermost Loop Detection (for non-counter mode).
//==============================================================================

// Checks if an affine.for loop is innermost (has no nested loops).
static bool isInnermostLoop(affine::AffineForOp for_op) {
  bool has_nested_loops = false;
  for_op.getBody()->walk([&](affine::AffineForOp) { has_nested_loops = true; });
  return !has_nested_loops;
}

//==============================================================================
// Perfect Loop Band Detection.
//==============================================================================

// A perfect loop band is a sequence of perfectly nested loops where each loop
// (except the innermost) has exactly one child loop and no other operations
// (no prologue/epilogue).
struct PerfectLoopBand {
  // Outer to inner loop order.
  SmallVector<affine::AffineForOp> loops;

  bool isEmpty() const { return loops.empty(); }
  size_t getDepth() const { return loops.size(); }
};

// Detects the maximal perfect loop band starting from the given loop.
// Returns the sequence of perfectly nested loops.
static PerfectLoopBand detectPerfectLoopBand(affine::AffineForOp start_loop) {
  PerfectLoopBand band;
  affine::AffineForOp current_loop = start_loop;

  while (current_loop) {
    band.loops.push_back(current_loop);

    // Checks the body of current loop.
    Block &body = current_loop.getRegion().front();

    // Counts non-trivial operations (excluding yield).
    affine::AffineForOp nested_loop = nullptr;
    size_t num_loops = 0;
    size_t num_other_ops = 0;

    for (Operation &op : body) {
      if (auto for_op = dyn_cast<affine::AffineForOp>(&op)) {
        nested_loop = for_op;
        num_loops++;
      } else if (!(isa<affine::AffineYieldOp>(&op) &&
                   op.getNumOperands() == 0)) {
        num_other_ops++;
      }
    }

    // Perfect nesting condition: exactly 1 nested loop, no other operations.
    if (num_loops == 1 && num_other_ops == 0) {
      // Continues to next level.
      current_loop = nested_loop;
    } else {
      break; // Not perfect anymore.
    }
  }

  return band;
}

//==============================================================================
// Counter Creation.
//==============================================================================

struct CounterInfo {
  affine::AffineForOp loop;
  // The index value from taskflow.counter
  Value counter_index;
};

// Resolves an affine bound (lower or upper) to an SSA value.
// const_cache maps integer values to already-created arith.constant ops so
// that multiple loops with the same constant bound (e.g. 0, 64, 1) share a
// single definition instead of producing redundant constants.
static Value resolveAffineBound(OpBuilder &builder, Location loc, AffineMap map,
                                ValueRange operands,
                                DenseMap<int64_t, Value> &const_cache) {
  // Constant: affine_map<() -> (C)>
  if (map.isSingleConstant()) {
    int64_t val = map.getSingleConstantResult();
    auto [it, inserted] = const_cache.try_emplace(val, Value{});
    if (inserted)
      it->second = builder.create<arith::ConstantIndexOp>(loc, val);
    return it->second;
  }

  // Simple symbol passthrough: affine_map<(d0) -> (d0)>
  if (map.getNumResults() == 1) {
    if (auto sym = dyn_cast<AffineSymbolExpr>(map.getResult(0))) {
      return operands[map.getNumDims() + sym.getPosition()];
    }
    if (auto dim = dyn_cast<AffineDimExpr>(map.getResult(0))) {
      return operands[dim.getPosition()];
    }
  }

  // General case: materializes as affine.apply.
  return builder.create<affine::AffineApplyOp>(loc, map, operands);
}

// Creates a chain of taskflow.counter operations for a perfect loop band.
// Returns counter info for each loop level.
static SmallVector<CounterInfo>
createCounterChain(OpBuilder &builder, Location loc,
                   const PerfectLoopBand &band) {
  SmallVector<CounterInfo> counters;
  Value parent_counter = nullptr;
  // Cache of integer value → arith.constant op, shared across all loop levels
  // so that identical constants (0, 1, 64, …) are emitted only once.
  DenseMap<int64_t, Value> const_cache;

  for (affine::AffineForOp loop : band.loops) {
    CounterInfo info;
    info.loop = loop;

    // Gets loop bounds.
    Value lb = resolveAffineBound(builder, loc, loop.getLowerBoundMap(),
                                  loop.getLowerBoundOperands(), const_cache);
    Value up = resolveAffineBound(builder, loc, loop.getUpperBoundMap(),
                                  loop.getUpperBoundOperands(), const_cache);

    // Step is always a compile-time constant; reuse from cache.
    int64_t step_val = loop.getStepAsInt();
    auto [step_it, step_inserted] = const_cache.try_emplace(step_val, Value{});
    if (step_inserted)
      step_it->second = builder.create<arith::ConstantIndexOp>(loc, step_val);
    Value step = step_it->second;

    // Creates counter.

    auto counter_op = builder.create<TaskflowCounterOp>(
        loc,
        /*counter_index_type*/ builder.getIndexType(),
        /*parent_index*/ parent_counter,
        /*lower_bound*/ lb,
        /*upper_bound*/ up,
        /*step*/ step,
        /*counter_hierarchy*/ StringAttr{},
        /*counter_dynamism*/ StringAttr{},
        /*counter_id*/ IntegerAttr{});

    info.counter_index = counter_op.getCounterIndex();
    parent_counter = info.counter_index;
    counters.push_back(info);
  }

  return counters;
}

//==============================================================================
// Hyperblock Creation (with counter support).
//==============================================================================

// Analyzes which loop induction variables are actually used in the loop body.
// Returns indices of loops whose induction variables are used.
static SmallVector<size_t> analyzeUsedLoopIndices(const PerfectLoopBand &band) {
  SmallVector<size_t> used_indices;

  // Gets the deepest perfect loop's body.
  affine::AffineForOp deepest_loop = band.loops.back();
  Block &body = deepest_loop.getRegion().front();

  // Collects all values used in the body.
  DenseSet<Value> used_values;
  body.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      used_values.insert(operand);
    }
  });

  // Checks which loop induction variables are used.
  for (size_t i = 0; i < band.loops.size(); ++i) {
    affine::AffineForOp loop = band.loops[i];
    Value induction_var = loop.getInductionVar();
    if (used_values.contains(induction_var)) {
      used_indices.push_back(i);
    }
  }

  return used_indices;
}

// Clones the body of the deepest perfect loop in the perfect band into a
// hyperblock. Handles iter_args (reduction variables) by:
// 1. Adding iter_args initial values as hyperblock inputs
// 2. Mapping iter_args to hyperblock block arguments
// 3. Returning reduction results as hyperblock outputs
static TaskflowHyperblockOp
createHyperblockFromLoopBody(OpBuilder &builder, Location loc,
                             const PerfectLoopBand &band,
                             const SmallVector<CounterInfo> &counters) {
  // Gets the deepest perfect loop in the perfect nested band.
  affine::AffineForOp deepest_perfect_loop = band.loops.back();
  Block &loop_body = deepest_perfect_loop.getRegion().front();

  // Analyzes which loop indices are actually used.
  SmallVector<size_t> used_loop_indices = analyzeUsedLoopIndices(band);

  // Checks if the deepest loop has iter_args (reduction variables).
  bool has_iter_args = deepest_perfect_loop.getNumIterOperands() > 0;
  SmallVector<Value> iter_args_init_values = {};
  SmallVector<Type> iter_args_types = {};

  if (has_iter_args) {
    for (Value init_val : deepest_perfect_loop.getInits()) {
      iter_args_init_values.push_back(init_val);
      iter_args_types.push_back(init_val.getType());
    }
  }

  // Builds trigger values (only for USED counter indices)
  SmallVector<Value> trigger_values;
  for (size_t idx : used_loop_indices) {
    trigger_values.push_back(counters[idx].counter_index);
  }

  // Determines hyperblock result types (from iter_args if present).
  SmallVector<Type> result_types = {};
  if (has_iter_args) {
    result_types = iter_args_types;
  }

  // Creates hyperblock operation with iter_args as inputs.
  auto hyperblock_op = builder.create<TaskflowHyperblockOp>(
      loc, result_types, trigger_values, iter_args_init_values);

  // Builds block arguments:
  // 1. Counter indices (only for USED loop levels).
  // 2. Iter args values (passed through hyperblock invocation).
  SmallVector<Type> arg_types;
  SmallVector<Location> arg_locs;

  // Adds counter index arguments (only for used indices).
  for (size_t i = 0; i < used_loop_indices.size(); ++i) {
    arg_types.push_back(builder.getIndexType());
    arg_locs.push_back(loc);
  }

  // Adds iter_args as hyperblock block arguments.
  if (has_iter_args) {
    for (Type ty : iter_args_types) {
      arg_types.push_back(ty);
      arg_locs.push_back(loc);
    }
  }

  Block *hyperblock_body = &hyperblock_op.getBody().emplaceBlock();
  hyperblock_body->addArguments(arg_types, arg_locs);

  OpBuilder body_builder = OpBuilder::atBlockBegin(hyperblock_body);
  IRMapping mapper;

  // Maps USED loop induction variables to hyperblock arguments.
  for (size_t i = 0; i < used_loop_indices.size(); ++i) {
    size_t loop_idx = used_loop_indices[i];
    affine::AffineForOp loop = band.loops[loop_idx];
    mapper.map(loop.getInductionVar(), hyperblock_body->getArgument(i));
  }

  // Maps iter_args to hyperblock block arguments (after counter indices).
  if (has_iter_args) {
    for (size_t i = 0; i < iter_args_types.size(); ++i) {
      size_t arg_idx = used_loop_indices.size() + i;
      mapper.map(deepest_perfect_loop.getRegionIterArgs()[i],
                 hyperblock_body->getArgument(arg_idx));
    }
  }

  // Clones all operations from the deepest perfect loop's body.
  SmallVector<Value> yield_operands;

  for (Operation &op : loop_body) {
    // Handles affine.yield with operands (reduction results).
    if (auto yield_op = dyn_cast<affine::AffineYieldOp>(&op)) {
      if (yield_op.getNumOperands() > 0) {
        // Maps the yielded values for hyperblock's return.
        for (Value yielded : yield_op.getOperands()) {
          Value mapped = mapper.lookupOrDefault(yielded);
          yield_operands.push_back(mapped);
        }
      }
      continue; // Skips the yield itself.
    }

    // Clones operation (including nested affine.for with iter_args).
    Operation *cloned = body_builder.clone(op, mapper);

    // Updates mapper with cloned operation results.
    for (size_t i = 0; i < op.getNumResults(); ++i) {
      mapper.map(op.getResult(i), cloned->getResult(i));
    }
  }

  // Adds terminator with reduction results (if any).
  if (has_iter_args) {
    body_builder.create<TaskflowHyperblockYieldOp>(
        loc,
        /*iter_args_next=*/yield_operands, // No iter_args_next for final
                                           // iteration
        /*results=*/yield_operands);       // Reduction results
  } else {
    body_builder.create<TaskflowHyperblockYieldOp>(loc);
  }

  // Converts affine operations to standard/scf operations.
  MLIRContext *context = hyperblock_op.getContext();
  RewritePatternSet patterns(context);
  populateAffineToStdConversionPatterns(patterns);

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         func::FuncDialect, TaskflowDialect, scf::SCFDialect>();
  target.addIllegalOp<affine::AffineLoadOp, affine::AffineStoreOp,
                      affine::AffineForOp, affine::AffineIfOp,
                      affine::AffineYieldOp>();

  if (failed(
          applyPartialConversion(hyperblock_op, target, std::move(patterns)))) {
    llvm::errs()
        << "Error: Failed to convert affine operations to standard/scf\n";
    return nullptr;
  }

  return hyperblock_op;
}

//==============================================================================
// Hyperblock Creation (without counter support - innermost loop only).
//==============================================================================

// Wraps an innermost affine.for loop in a hyperblock without using counters.
// The entire loop is moved into the hyperblock, and affine ops are converted
// to standard/scf ops. If the loop has results (iter_args), they are returned
// through the hyperblock.
//
// Returns the created hyperblock operation, or nullptr on failure.
static TaskflowHyperblockOp
createHyperblockFromInnermostLoop(OpBuilder &builder, Location loc,
                                  affine::AffineForOp innermost_loop) {
  // Checks if the loop has results (iter_args/reduction variables).
  bool has_results = innermost_loop.getNumResults() > 0;
  SmallVector<Type> result_types;

  if (has_results) {
    for (Value result : innermost_loop.getResults()) {
      result_types.push_back(result.getType());
    }
  }

  // Creates hyperblock with NO triggers and NO iter_args.
  // The entire affine.for will be moved inside, so all loop parameters
  // (bounds, step, iter_args) remain inside the loop itself.
  auto hyperblock_op = builder.create<TaskflowHyperblockOp>(
      loc, result_types, /*triggers=*/ValueRange{},
      /*iter_args=*/ValueRange{});

  // Creates hyperblock body block with NO arguments.
  // The loop is self-contained and doesn't need external inputs.
  Block *hyperblock_body = &hyperblock_op.getBody().emplaceBlock();

  // Moves the entire innermost loop INTO the hyperblock.
  // After this, the loop is inside the hyperblock region.
  innermost_loop->moveBefore(hyperblock_body, hyperblock_body->end());

  // Converts affine operations to standard/scf operations.
  // This must be done BEFORE creating the yield, because:
  // 1. affine.for will be replaced by scf.for
  // 2. We need to reference the scf.for's results in the yield
  MLIRContext *context = hyperblock_op.getContext();
  RewritePatternSet patterns(context);
  populateAffineToStdConversionPatterns(patterns);

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         func::FuncDialect, TaskflowDialect, scf::SCFDialect>();
  target.addIllegalOp<affine::AffineLoadOp, affine::AffineStoreOp,
                      affine::AffineForOp, affine::AffineIfOp,
                      affine::AffineYieldOp>();

  if (failed(
          applyPartialConversion(hyperblock_op, target, std::move(patterns)))) {
    llvm::errs()
        << "Error: Failed to convert affine operations to standard/scf\n";
    return nullptr;
  }

  // After conversion, the affine.for has been replaced by scf.for.
  // We need to find the converted scf.for to get its results.
  OpBuilder body_builder = OpBuilder::atBlockEnd(hyperblock_body);

  if (has_results) {
    // Finds the scf.for that replaced our affine.for.
    // Since we moved only one loop into the hyperblock, there should be
    // exactly one scf.for at the top level.
    scf::ForOp scf_loop = nullptr;
    for (Operation &op : hyperblock_body->getOperations()) {
      if (auto for_op = dyn_cast<scf::ForOp>(&op)) {
        scf_loop = for_op;
        break;
      }
    }

    if (!scf_loop) {
      llvm::errs() << "Error: Could not find converted scf.for in hyperblock\n";
      return nullptr;
    }

    // Gets results from the converted scf.for.
    SmallVector<Value> loop_results;
    for (size_t i = 0; i < scf_loop.getNumResults(); ++i) {
      loop_results.push_back(scf_loop.getResult(i));
    }

    // Creates hyperblock terminator that yields the loop's results.
    body_builder.create<TaskflowHyperblockYieldOp>(
        loc, /*iter_args_next=*/ValueRange{},
        /*results=*/loop_results);
  } else {
    // No results - just create empty yield.
    body_builder.create<TaskflowHyperblockYieldOp>(loc);
  }

  return hyperblock_op;
}

//============================================================================
// Task Transformation.
//===========================================================================
static void writeSourceIterationDomain(TaskflowTaskOp task,
                                       SourceIterationDomainInfo info,
                                       bool capturePending) {
  OpBuilder builder(task.getContext());
  task->setAttr(kSourceIterationDomainAttr,
                makeSourceIterationDomainAttr(builder, info));
  if (capturePending)
    task->setAttr(kSourceIterationCapturePendingAttr, builder.getUnitAttr());
}

static void markSourceDomainUnsupported(SourceIterationDomainInfo &info,
                                        StringRef reason) {
  info.complete = false;
  info.reason = reason.str();
}

/// affine::loopUnrollFull may promote a single-iteration affine loop by
/// replacing its IV with an arith.constant inserted at the containing
/// function entry. That insertion point is outside TaskflowTaskOp's isolated
/// region, so uses in the task would become illegal captures. Re-materialize
/// only those leaked index constants in the task; reject any other leaked
/// value instead of silently weakening region isolation.
static LogicalResult localizeUnrolledLoopConstants(TaskflowTaskOp task) {
  auto isInsideTask = [&](Operation *operation) {
    for (Operation *parent = operation; parent;
         parent = parent->getParentOp()) {
      if (parent == task.getOperation())
        return true;
    }
    return false;
  };

  SmallVector<Value> escapedConstants;
  LogicalResult result = success();
  task.walk([&](Operation *operation) {
    if (operation == task.getOperation())
      return;
    for (Value operand : operation->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (!definition) {
        auto argument = dyn_cast<BlockArgument>(operand);
        if (argument && !isInsideTask(argument.getOwner()->getParentOp())) {
          operation->emitError("internal-loop unrolling introduced an external block argument");
          result = failure();
        }
        continue;
      }
      if (isInsideTask(definition)) continue;
      if (isa<arith::ConstantIndexOp>(definition)) {
        escapedConstants.push_back(operand);
        continue;
      }
      operation->emitError(
          "internal-loop unrolling introduced a non-constant value outside "
          "the isolated task region");
      result = failure();
    }
  });
  if (failed(result))
    return failure();

  if (escapedConstants.empty())
    return success();

  Block &taskBody = task.getBody().front();
  OpBuilder builder(&taskBody, taskBody.begin());
  DenseMap<Value, Value> localized;
  for (Value escaped : escapedConstants) {
    auto constant = cast<arith::ConstantIndexOp>(escaped.getDefiningOp());
    Value local = localized.lookup(escaped);
    if (!local) {
      local = builder.create<arith::ConstantIndexOp>(constant.getLoc(),
                                                      constant.value());
      localized.try_emplace(escaped, local);
    }
    escaped.replaceUsesWithIf(local, [&](OpOperand &use) {
      return use.getOwner() != task.getOperation() &&
             isInsideTask(use.getOwner());
    });
  }
  return success();
}

static LogicalResult captureAndExpandSourceIterationDomain(
    TaskflowTaskOp task, const PerfectLoopBand &band) {
  if (task->hasAttr(kSourceIterationDomainAttr)) {
    bool hasAffineLoop = false;
    task.walk([&](affine::AffineForOp) { hasAffineLoop = true; });
    if (hasAffineLoop) {
      task.emitError("refusing to replace an imported source iteration-domain "
                     "certificate while source loops remain");
      return failure();
    }
    return success();
  }

  SourceIterationDomainInfo info =
      certifySourceIterationDomain(task, band.loops);
  if (!info.complete) {
    writeSourceIterationDomain(task, std::move(info),
                               /*capturePending=*/false);
    return success();
  }

  // An explicit legacy count is valid only when it describes the source axes
  // that are about to become the current Taskflow counter chain. Internal
  // carried axes are expanded inside one mapper firing and therefore do not
  // increase this count.
  for (StringRef attribute : {StringRef("trip_count"),
                              StringRef("amoeba.selected_trip_count")}) {
    if (!task->hasAttr(attribute))
      continue;
    auto count = task->getAttrOfType<IntegerAttr>(attribute);
    if (!count || count.getInt() != info.representedMultiplicity) {
      markSourceDomainUnsupported(
          info, "explicit mapper-firing count disagrees with source Taskflow axes");
      writeSourceIterationDomain(task, std::move(info),
                                 /*capturePending=*/false);
      return success();
    }
  }

  auto internalAxis = llvm::find_if(info.axes, [](const auto &axis) {
    return !axis.representedByTaskflow;
  });
  if (internalAxis != info.axes.end()) {
    // Persist the source-derived bounds/carry contract before the loop is
    // rewritten. This incomplete snapshot cannot be consumed for cost; it is
    // replaced only in this invocation after loopUnrollFull succeeds.
    SourceIterationDomainInfo pending = info;
    pending.complete = false;
    pending.reason = "source proof captured; internal expansion is pending";
    writeSourceIterationDomain(task, std::move(pending),
                               /*capturePending=*/false);
    if (internalAxis->extent > 8) {
      markSourceDomainUnsupported(
          info, "internal source loop exceeds the supported full-unroll bound");
      writeSourceIterationDomain(task, std::move(info),
                                 /*capturePending=*/false);
      return success();
    }

    SmallVector<affine::AffineForOp> internalLoops;
    llvm::DenseSet<Operation *> represented;
    for (affine::AffineForOp loop : band.loops)
      represented.insert(loop.getOperation());
    task.walk([&](affine::AffineForOp loop) {
      if (!represented.contains(loop.getOperation()))
        internalLoops.push_back(loop);
    });
    if (internalLoops.size() != 1 ||
        failed(affine::loopUnrollFull(internalLoops.front()))) {
      markSourceDomainUnsupported(
          info, "internal carried source loop could not be fully unrolled");
      writeSourceIterationDomain(task, std::move(info),
                                 /*capturePending=*/false);
      return success();
    }
    if (failed(localizeUnrolledLoopConstants(task)))
      return failure();
    internalAxis->expandedInsideMapperFiring = true;
  }

  writeSourceIterationDomain(task, std::move(info),
                             /*capturePending=*/true);
  return success();
}

static LogicalResult transformTaskWithCounter(TaskflowTaskOp task_op) {
  Location loc = task_op.getLoc();
  Block &task_body = task_op.getBody().front();

  // Finds all top-level loops in the task.
  SmallVector<affine::AffineForOp> top_level_loops;
  for (Operation &op : task_body) {
    if (auto for_op = dyn_cast<affine::AffineForOp>(&op)) {
      top_level_loops.push_back(for_op);
    }
  }

  if (top_level_loops.empty()) {
    if (task_op->hasAttr(kSourceIterationDomainAttr))
      return success();
    SourceIterationDomainInfo info =
        certifySourceIterationDomain(task_op, {});
    bool complete = info.complete;
    writeSourceIterationDomain(task_op, std::move(info), complete);
    return success();
  }

  if (task_op->hasAttr(kSourceIterationDomainAttr)) {
    task_op.emitError("refusing to recertify source iteration domain while "
                      "affine loops remain");
    return failure();
  }
  if (top_level_loops.size() != 1) {
    SourceIterationDomainInfo info =
        certifySourceIterationDomain(task_op, {});
    markSourceDomainUnsupported(info,
                                "task has multiple top-level affine loops");
    writeSourceIterationDomain(task_op, std::move(info),
                               /*capturePending=*/false);
    return success();
  }

  PerfectLoopBand sourceBand = detectPerfectLoopBand(top_level_loops.front());
  if (failed(captureAndExpandSourceIterationDomain(task_op, sourceBand)))
    return failure();
  auto domain = task_op->getAttrOfType<DictionaryAttr>(
      kSourceIterationDomainAttr);
  auto complete = domain ? domain.getAs<BoolAttr>("complete") : BoolAttr{};
  if (!complete || !complete.getValue())
    return success();

  OpBuilder builder(&task_body, task_body.begin());

  // Stores mapping from loop results to hyperblock results.
  DenseMap<Value, Value> loop_result_to_hyperblock_result;

  // Processes each top-level loop.
  for (affine::AffineForOp top_loop : top_level_loops) {
    llvm::errs() << "\n[ConstructHyperblock] Processing top-level loop\n";

    // Step 1: Re-detects the maximal band after any supported carried-loop
    // expansion. The source domain was captured before the expansion above.
    PerfectLoopBand band = detectPerfectLoopBand(top_loop);
    llvm::errs() << "  Detected perfect loop band of depth " << band.getDepth()
                 << "\n";

    // Step 2: Creates counter chain for the perfect band.
    builder.setInsertionPoint(top_loop);
    SmallVector<CounterInfo> counters = createCounterChain(builder, loc, band);
    llvm::errs() << "  Created " << counters.size() << " counters\n";

    // Step 3: Creates hyperblock from deepest loop's body.
    TaskflowHyperblockOp hyperblock_op =
        createHyperblockFromLoopBody(builder, loc, band, counters);
    llvm::errs() << "  Created hyperblock with "
                 << hyperblock_op.getBody().front().getOperations().size()
                 << " operations\n";

    assert(hyperblock_op && "Hyperblock creation failed");

    // If the loop has results (iter_args), map them to hyperblock results.
    if (top_loop.getNumResults() > 0) {
      llvm::errs() << "  Mapping " << top_loop.getNumResults()
                   << " loop results to hyperblock outputs\n";

      for (size_t i = 0; i < top_loop.getNumResults(); ++i) {
        loop_result_to_hyperblock_result[top_loop.getResult(i)] =
            hyperblock_op.getResult(i);
      }
    }
  }

  // Replaces loop results with hyperblock results BEFORE erasing loops.
  for (auto [loop_result, hb_result] : loop_result_to_hyperblock_result) {
    loop_result.replaceAllUsesWith(hb_result);
  }

  // Step 4: Erases all original loops.
  for (affine::AffineForOp loop : top_level_loops) {
    // Ensures no uses remain.
    assert(loop->use_empty() && "Loop still has uses before erasing");
    loop->erase();
  }

  return success();
}

static LogicalResult transformTaskWithoutCounter(TaskflowTaskOp task_op) {
  if (!task_op->hasAttr(kSourceIterationDomainAttr)) {
    SourceIterationDomainInfo info =
        certifySourceIterationDomain(task_op, {});
    if (!info.axes.empty() || !info.complete)
      markSourceDomainUnsupported(
          info, "source loops are not represented by Taskflow counters");
    bool complete = info.complete;
    writeSourceIterationDomain(task_op, std::move(info), complete);
  }

  auto existingDomain = task_op->getAttrOfType<DictionaryAttr>(
      kSourceIterationDomainAttr);
  auto existingComplete =
      existingDomain ? existingDomain.getAs<BoolAttr>("complete") : BoolAttr{};
  if (!existingComplete || !existingComplete.getValue())
    return success();

  Location loc = task_op.getLoc();

  llvm::errs()
      << "\n[ConstructHyperblock] Processing WITHOUT counter support\n";
  llvm::errs() << "  Will wrap only innermost loops as hyperblocks\n";

  // Collects ALL innermost loops (including nested ones).
  SmallVector<affine::AffineForOp> innermost_loops;
  task_op.walk([&](affine::AffineForOp for_op) {
    if (isInnermostLoop(for_op)) {
      innermost_loops.push_back(for_op);
    }
  });

  llvm::errs() << "  Found " << innermost_loops.size() << " innermost loops\n";

  OpBuilder builder(task_op.getContext());

  // Processes each innermost loop.
  for (affine::AffineForOp innermost_loop : innermost_loops) {
    llvm::errs() << "  Processing innermost loop\n";

    bool has_results = innermost_loop.getNumResults() > 0;

    // Saves the USERS of loop results, not the results themselves.
    // And replaces these uses after creating the hyperblock.
    SmallVector<OpOperand *> result_uses;
    if (has_results) {
      for (Value result : innermost_loop.getResults()) {
        for (OpOperand &use : result.getUses()) {
          result_uses.push_back(&use);
        }
      }
    }

    // Inserts hyperblock BEFORE the innermost loop.
    builder.setInsertionPoint(innermost_loop);

    // Creates hyperblock from the innermost loop.
    TaskflowHyperblockOp hyperblock_op =
        createHyperblockFromInnermostLoop(builder, loc, innermost_loop);

    if (!hyperblock_op) {
      llvm::errs()
          << "Error: Failed to create hyperblock from innermost loop\n";
      return failure();
    }

    llvm::errs() << "  Created hyperblock with "
                 << hyperblock_op.getBody().front().getOperations().size()
                 << " operations\n";

    // Replaces uses of the original loop results with hyperblock results.
    // And iterates through the saved uses and update them.
    if (has_results) {
      llvm::errs() << "  Replacing " << result_uses.size()
                   << " uses with hyperblock outputs\n";

      size_t result_idx = 0;

      for (OpOperand *use : result_uses) {
        // Updates this use to point to the hyperblock result.
        use->set(hyperblock_op.getResult(result_idx++));
      }
    }

    llvm::errs() << "  Wrapped innermost loop in hyperblock\n";
  }

  return success();
}

struct ConstructHyperblockFromTaskPass
    : public PassWrapper<ConstructHyperblockFromTaskPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstructHyperblockFromTaskPass)

  StringRef getArgument() const final {
    return "construct-hyperblock-from-task";
  }

  StringRef getDescription() const final {
    return "Constructs hyperblocks from taskflow tasks by detecting perfect "
           "nested loop bands.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<mlir::taskflow::TaskflowDialect, affine::AffineDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();
    const neura::Architecture &arch = neura::getArchitecture();
    bool supports_counter = arch.canSupportCounter();

    // Walks through all TaskflowTaskOp in the function.
    func_op.walk([&](TaskflowTaskOp task_op) {
      if (supports_counter) {
        if (failed(transformTaskWithCounter(task_op))) {
          signalPassFailure();
          return WalkResult::interrupt();
        }
      } else {
        if (failed(transformTaskWithoutCounter(task_op))) {
          signalPassFailure();
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::amoeba::neura::createConstructHyperblockFromTaskPass() {
  return std::make_unique<ConstructHyperblockFromTaskPass>();
}
