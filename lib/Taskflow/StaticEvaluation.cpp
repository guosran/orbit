#include "Taskflow/StaticEvaluation.h"
#include "Taskflow/StaticMapping.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
using namespace orbit::taskflow;
llvm::json::Value orbit::taskflow::evaluateStaticMLIR(
    llvm::StringRef text, const llvm::json::Value &config, std::string &error) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  StaticAnalysis analysis;
  StaticConfig parsed;
  if (!module || mlir::failed(mlir::verify(*module)))
    error = "static evaluation MLIR verification failed";
  else if (analyzeStaticProgram(*module, analysis, error) &&
           parseStaticConfig(config, parsed, error))
    return serializeStaticResult(mapStaticProgram(analysis, parsed));
  return llvm::json::Object{{"status", "unsupported"},
                            {"best_plan", nullptr},
                            {"reasons", llvm::json::Array{error}}};
}
