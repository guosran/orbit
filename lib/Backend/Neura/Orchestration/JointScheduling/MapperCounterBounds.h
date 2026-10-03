// Normalize literal counter bounds at the shared mapper/feature boundary.
// Only arith index literals are metadata here. Neura constants retain their
// routed predicate/iteration semantics. Reject drift before removing SSA.
#ifndef AMOEBA_JOINT_SCHEDULING_MAPPER_COUNTER_BOUNDS_H
#define AMOEBA_JOINT_SCHEDULING_MAPPER_COUNTER_BOUNDS_H

#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::amoeba::neura::joint_scheduling {

inline LogicalResult prepareMapperCounterBounds(func::FuncOp wrapper) {
  bool invalid = false;
  llvm::SmallPtrSet<Operation *, 16> literals;
  wrapper.walk([&](::mlir::neura::CounterOp counter) {
    if (invalid)
      return;
    const Value bounds[] = {counter.getLowerBound(), counter.getUpperBound(),
                            counter.getStep()};
    const StringRef names[] = {"lower_bound_value", "upper_bound_value",
                               "step_value"};
    SmallVector<Value> remaining;
    SmallVector<int32_t> segments;
    bool changed = false;
    for (unsigned index = 0; index < 3; ++index) {
      Value bound = bounds[index];
      Operation *producer = bound ? bound.getDefiningOp() : nullptr;
      IntegerAttr literal;
      if (producer && isa<arith::ConstantOp>(producer))
        literal = producer->getAttrOfType<IntegerAttr>("value");
      if (literal) {
        if (!bound.getType().isIndex() || !literal.getType().isIndex()) {
          counter.emitError("mapper counter literal and operand must have index type");
          invalid = true;
          return;
        }
        if (!literal.getValue().isSignedIntN(64)) {
          counter.emitError("mapper counter literal exceeds signed 64-bit bounds");
          invalid = true;
          return;
        }
        Attribute folded = counter->getAttr(names[index]);
        if (folded) {
          auto integer = dyn_cast<IntegerAttr>(folded);
          if (!integer || integer.getType() != literal.getType() ||
              !integer.getValue().isSignedIntN(64) ||
              integer.getInt() != literal.getInt()) {
            counter.emitError("mapper counter operand disagrees with folded ")
                << names[index];
            invalid = true;
            return;
          }
        } else {
          counter->setAttr(names[index], literal);
        }
        segments.push_back(0);
        literals.insert(producer);
        changed = true;
      } else {
        segments.push_back(bound ? 1 : 0);
        if (bound)
          remaining.push_back(bound);
      }
    }
    if (changed) {
      counter->setOperands(remaining);
      counter->setAttr("operandSegmentSizes",
                       OpBuilder(counter).getDenseI32ArrayAttr(segments));
    }
  });
  if (invalid)
    return failure();
  for (Operation *literal : literals)
    if (literal->use_empty())
      literal->erase();
  // Repeated M/N rewrites can leave the previous literal bound producers
  // behind after replacing their last operand use. These pure, unused index
  // constants do not belong to the routed body either.
  SmallVector<arith::ConstantOp> deadBounds;
  wrapper.walk([&](arith::ConstantOp constant) {
    if (constant.getType().isIndex() && constant->use_empty())
      deadBounds.push_back(constant);
  });
  for (arith::ConstantOp constant : deadBounds)
    constant.erase();

  WalkResult result = wrapper.walk([&](Operation *operation) {
    StringRef dialect = operation->getName().getDialectNamespace();
    if (dialect == "arith" || dialect == "memref" || dialect == "scf" ||
        dialect == "cf") {
      operation->emitError("mapper body requires prepared Neura dataflow; "
                           "unsupported residual operation ")
          << operation->getName();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace mlir::amoeba::neura::joint_scheduling
#endif
