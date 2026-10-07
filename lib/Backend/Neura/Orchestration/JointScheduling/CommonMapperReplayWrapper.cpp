//===- CommonMapperReplayWrapper.cpp --------------------------*- C++ -*-===//

#include "CommonMapperReplayWrapper.h"

#include "MapperCounterBounds.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::amoeba::neura::joint_scheduling {

func::FuncOp buildCommonMapperReplayWrapper(taskflow::TaskflowTaskOp task,
                                           ModuleOp owner,
                                           llvm::StringRef symbolName,
                                           std::string &error) {
  auto fail = [&](llvm::StringRef message) {
    error = message.str();
    return func::FuncOp();
  };
  if (!task || !owner || symbolName.empty())
    return fail("mapper replay wrapper requires a task, owner module, and symbol");

  SmallVector<::mlir::neura::KernelOp> kernels;
  task.walk([&](::mlir::neura::KernelOp kernel) { kernels.push_back(kernel); });
  if (kernels.size() != 1)
    return fail("per-task mapper replay requires exactly one neura.kernel");
  ::mlir::neura::KernelOp kernel = kernels.front();
  if (kernel.getBody().empty())
    return fail("cannot map an empty neura.kernel");

  SmallVector<Type> wrapperResultTypes;
  bool sawYield = false;
  for (Block &block : kernel.getBody()) {
    auto yield = dyn_cast<::mlir::neura::YieldOp>(block.getTerminator());
    if (!yield)
      continue;
    SmallVector<Type> blockResultTypes;
    for (Value result : yield.getResults())
      blockResultTypes.push_back(result.getType());
    if (!sawYield) {
      wrapperResultTypes = std::move(blockResultTypes);
      sawYield = true;
    } else if (wrapperResultTypes != blockResultTypes) {
      return fail("mapper replay kernel yields inconsistent result types");
    }
  }
  if (!sawYield)
    return fail("mapper replay kernel has no terminal neura.yield");

  MLIRContext *context = task.getContext();
  Location location = task.getLoc();
  OpBuilder builder(context);
  builder.setInsertionPointToStart(owner.getBody());
  SmallVector<Type> argumentTypes;
  for (BlockArgument argument : kernel.getBody().front().getArguments())
    argumentTypes.push_back(argument.getType());
  auto wrapper = builder.create<func::FuncOp>(
      location, symbolName,
      builder.getFunctionType(argumentTypes, wrapperResultTypes));
  wrapper->setAttr("accelerator", builder.getStringAttr("neura"));
  IRMapping mapping;
  kernel.getBody().cloneInto(&wrapper.getBody(), mapping);
  for (Block &block : wrapper.getBody()) {
    if (auto yield = dyn_cast<::mlir::neura::YieldOp>(block.getTerminator())) {
      builder.setInsertionPoint(yield);
      SmallVector<Value> returnValues;
      for (Value result : yield.getResults()) {
        // InsertDataMovPass only wraps Neura consumers. Give the mapper a
        // routed producer for an operand-bearing func.return as well.
        auto move = builder.create<::mlir::neura::DataMovOp>(
            location, result.getType(), result);
        returnValues.push_back(move.getResult());
      }
      builder.create<func::ReturnOp>(location, returnValues);
      yield.erase();
    }
  }

  if (failed(prepareMapperCounterBounds(wrapper))) {
    wrapper.erase();
    return fail("invalid counter metadata or unprepared mapper body");
  }
  return wrapper;
}

std::string printCommonMapperReplayWrapper(func::FuncOp wrapper) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  wrapper.print(stream, OpPrintingFlags().useLocalScope());
  stream.flush();
  return text;
}

} // namespace mlir::amoeba::neura::joint_scheduling
