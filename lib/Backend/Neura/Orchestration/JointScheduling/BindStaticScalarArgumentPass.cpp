//===- BindStaticScalarArgumentPass.cpp -----------------------------------===//
// Specialize one scalar function argument for a compile-time-known benchmark
// input while keeping the function ABI intact. The caller must prove that it
// passes the same scalar value when checking source equivalence.
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <string>

using namespace mlir;

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createBindStaticScalarArgumentPass();
}

namespace {

struct BindStaticScalarArgumentPass
    : PassWrapper<BindStaticScalarArgumentPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BindStaticScalarArgumentPass)

  BindStaticScalarArgumentPass() = default;
  BindStaticScalarArgumentPass(const BindStaticScalarArgumentPass &other)
      : PassWrapper<BindStaticScalarArgumentPass, OperationPass<ModuleOp>>(
            other) {}

  StringRef getArgument() const override { return "bind-static-scalar-argument"; }
  StringRef getDescription() const override {
    return "Bind a verified compile-time scalar function argument";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }

  Option<std::string> functionName{
      *this, "function", llvm::cl::desc("Exact function symbol; inferred if unique"),
      llvm::cl::init("")};
  Option<int64_t> argumentIndex{
      *this, "argument", llvm::cl::desc("Zero-based scalar argument index"),
      llvm::cl::init(-1)};
  Option<int64_t> scalarValue{
      *this, "value", llvm::cl::desc("Compile-time signed scalar value"),
      llvm::cl::init(0)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> matches;
    module.walk([&](func::FuncOp function) {
      if (functionName.empty() || function.getSymName() == functionName)
        matches.push_back(function);
    });
    if (matches.size() != 1) {
      module.emitError("static scalar binding requires exactly one selected function");
      signalPassFailure();
      return;
    }
    func::FuncOp function = matches.front();
    if (function.isDeclaration() || argumentIndex < 0 ||
        argumentIndex >= static_cast<int64_t>(function.getNumArguments())) {
      function.emitError("static scalar binding has no body or invalid argument index");
      signalPassFailure();
      return;
    }
    BlockArgument argument = function.getArgument(argumentIndex);
    Type type = argument.getType();
    if (auto integer = dyn_cast<IntegerType>(type)) {
      if (!llvm::APInt(64, scalarValue, true).isSignedIntN(integer.getWidth())) {
        function.emitError("static scalar binding value exceeds argument width");
        signalPassFailure();
        return;
      }
    } else if (!type.isIndex()) {
      function.emitError("static scalar binding requires an integer or index argument");
      signalPassFailure();
      return;
    }
    if (argument.use_empty()) {
      function.emitError("static scalar binding argument is unused");
      signalPassFailure();
      return;
    }
    OpBuilder builder(function.getContext());
    builder.setInsertionPointToStart(&function.getBody().front());
    Value constant;
    if (type.isIndex())
      constant = builder.create<arith::ConstantIndexOp>(
          function.getLoc(), scalarValue).getResult();
    else
      constant = builder.create<arith::ConstantIntOp>(
          function.getLoc(), scalarValue,
          cast<IntegerType>(type)).getResult();
    argument.replaceAllUsesWith(constant);
    std::string attribute =
        "amoeba.static_bound.arg." + std::to_string(argumentIndex);
    function->setAttr(attribute, builder.getI64IntegerAttr(scalarValue));
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::amoeba::neura::createBindStaticScalarArgumentPass() {
  return std::make_unique<BindStaticScalarArgumentPass>();
}
