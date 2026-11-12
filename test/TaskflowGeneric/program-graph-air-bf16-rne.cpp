#include "../../lib/TaskflowAIR/ProgramGraphAIRBF16RNE.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <utility>

int main() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  const char *source = R"mlir(
module {
  func.func @default_rne(%x: f32) -> bf16 {
    %r = arith.truncf %x : f32 to bf16
    return %r : bf16
  }
  func.func @explicit_rne(%x: f32) -> bf16 {
    %r = arith.truncf %x to_nearest_even : f32 to bf16
    return %r : bf16
  }
  func.func @explicit_rtz(%x: f32) -> bf16 {
    %r = arith.truncf %x toward_zero : f32 to bf16
    return %r : bf16
  }
  func.func @vector_untouched(%x: vector<2xf32>) -> vector<2xbf16> {
    %r = arith.truncf %x : vector<2xf32> to vector<2xbf16>
    return %r : vector<2xbf16>
  }
  func.func @other_width_untouched(%x: f64) -> f32 {
    %r = arith.truncf %x : f64 to f32
    return %r : f32
  }
}
)mlir";
  auto original = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!original)
    return 1;
  mlir::OwningOpRef<mlir::ModuleOp> emitted = original->clone();
  orbit::taskflow::detail::lowerScalarBF16RNE(*emitted);
  size_t before = 0, after = 0;
  original->walk([&](mlir::arith::TruncFOp) { ++before; });
  emitted->walk([&](mlir::arith::TruncFOp) { ++after; });
  if (before != 5 || after != 3 || mlir::failed(mlir::verify(*emitted)))
    return 2;
  // Constant folding evaluates the actual generated IR, rather than a second
  // copy of the rounding formula. Cover signs, ties, zero, subnormals and NaNs.
  const std::pair<uint32_t, uint16_t> cases[] = {
      {0x00000000, 0x0000}, {0x80000000, 0x8000}, {0x3f808000, 0x3f80},
      {0x3f818000, 0x3f82}, {0xbf808000, 0xbf80}, {0xbf818000, 0xbf82},
      {0x00008000, 0x0000}, {0x00008001, 0x0001}, {0x80008000, 0x8000},
      {0x80008001, 0x8001}, {0x7f7fffff, 0x7f80}, {0xff7fffff, 0xff80},
      {0x7f800000, 0x7f80}, {0xff800000, 0xff80}, {0x7f800001, 0x7fc0},
      {0xff800001, 0xffc0},
  };
  for (const auto &[input, expected] : cases) {
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module.getBody());
    auto function = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "fold",
        builder.getFunctionType({}, {builder.getBF16Type()}));
    builder.setInsertionPointToStart(function.addEntryBlock());
    llvm::APFloat value(llvm::APFloat::IEEEsingle(), llvm::APInt(32, input));
    auto constant = builder.create<mlir::arith::ConstantFloatOp>(
        builder.getUnknownLoc(), value, builder.getF32Type());
    auto converted = builder.create<mlir::arith::TruncFOp>(
        builder.getUnknownLoc(), builder.getBF16Type(), constant);
    auto returned = builder.create<mlir::func::ReturnOp>(
        builder.getUnknownLoc(), mlir::ValueRange{converted});
    orbit::taskflow::detail::lowerScalarBF16RNE(module);
    mlir::PassManager passes(&context);
    passes.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(passes.run(module)))
      return 3;
    auto folded =
        returned.getOperand(0).getDefiningOp<mlir::arith::ConstantOp>();
    if (!folded || mlir::cast<mlir::FloatAttr>(folded.getValue())
                           .getValue()
                           .bitcastToAPInt()
                           .getZExtValue() != expected)
      return 4;
    module.erase();
  }
  llvm::outs() << "BF16 RNE trigger and generated-IR constant checks PASS\n";
}
