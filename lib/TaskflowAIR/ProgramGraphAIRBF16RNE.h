#ifndef ORBIT_TASKFLOW_AIR_PROGRAM_GRAPH_BF16_RNE_H
#define ORBIT_TASKFLOW_AIR_PROGRAM_GRAPH_BF16_RNE_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace orbit::taskflow::detail {
// The installed AIE lowering uses ambient core rounding for scalar fptrunc.
// Refine only default/explicit RNE scalar f32->bf16 in the emitted AIR clone.
// Integer arithmetic avoids changing crRnd or other accumulator conversions.
inline void lowerScalarBF16RNE(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::arith::TruncFOp> conversions;
  module.walk([&](mlir::arith::TruncFOp conversion) {
    if (!conversion.getIn().getType().isF32() ||
        !conversion.getOut().getType().isBF16())
      return;
    if (conversion.getRoundingmode().value_or(
            mlir::arith::RoundingMode::to_nearest_even) !=
        mlir::arith::RoundingMode::to_nearest_even)
      return;
    conversions.push_back(conversion);
  });
  for (auto conversion : conversions) {
    mlir::OpBuilder builder(conversion);
    const auto location = conversion.getLoc();
    auto constant = [&](int32_t value) -> mlir::Value {
      return builder.create<mlir::arith::ConstantIntOp>(location, value, 32);
    };
    auto shift = constant(16);
    auto bits = builder.create<mlir::arith::BitcastOp>(
        location, builder.getI32Type(), conversion.getIn());
    auto high = builder.create<mlir::arith::ShRUIOp>(location, bits, shift);
    auto odd = builder.create<mlir::arith::AndIOp>(location, high, constant(1));
    auto bias =
        builder.create<mlir::arith::AddIOp>(location, constant(32767), odd);
    // No overflow flags: wrapping i32 arithmetic preserves both signs,
    // subnormal ties, signed zero and finite overflow to infinity.
    auto rounded = builder.create<mlir::arith::AddIOp>(location, bits, bias);
    auto roundedHigh =
        builder.create<mlir::arith::ShRUIOp>(location, rounded, shift);
    auto finite = builder.create<mlir::arith::TruncIOp>(
        location, builder.getI16Type(), roundedHigh);
    auto absolute = builder.create<mlir::arith::AndIOp>(location, bits,
                                                        constant(0x7fffffff));
    auto isNaN = builder.create<mlir::arith::CmpIOp>(
        location, mlir::arith::CmpIPredicate::ugt, absolute,
        constant(0x7f800000));
    auto nanHigh = builder.create<mlir::arith::TruncIOp>(
        location, builder.getI16Type(), high);
    auto quietBit =
        builder.create<mlir::arith::ConstantIntOp>(location, 64, 16);
    // A NaN must never become infinity. Keep sign and upper payload and quiet
    // signaling NaNs; this does not model observable floating exception flags.
    auto nan = builder.create<mlir::arith::OrIOp>(location, nanHigh, quietBit);
    auto selected =
        builder.create<mlir::arith::SelectOp>(location, isNaN, nan, finite);
    auto result = builder.create<mlir::arith::BitcastOp>(
        location, conversion.getOut().getType(), selected);
    conversion.getOut().replaceAllUsesWith(result);
    conversion.erase();
  }
}
} // namespace orbit::taskflow::detail
#endif
