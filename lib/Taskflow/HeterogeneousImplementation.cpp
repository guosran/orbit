#include "Taskflow/HeterogeneousImplementation.h"
#include "Taskflow/GenericAnalysis.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace orbit::taskflow;

bool orbit::taskflow::lowerHeterogeneousCPUImplementation(
    ModuleOp child, OwningOpRef<ModuleOp> &plainCPU, std::string &error) {
  plainCPU = OwningOpRef<ModuleOp>();
  GenericAnalysis source;
  if (!analyzeGenericProgram(child, source, error))
    return false;
  auto fail = [&](StringRef reason) {
    error = "heterogeneous CPU implementation: " + reason.str();
    return false;
  };
  if (source.tasks.empty())
    return fail("a nonempty checked Taskflow child is required");
  for (const auto &storage : source.storages) {
    auto type = dyn_cast<MemRefType>(storage.root.getType());
    if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
        (!type.getElementType().isF32() && !type.getElementType().isBF16()))
      return fail(
          "only static identity-layout ranked f32/bf16 memrefs are supported");
  }

  OwningOpRef<ModuleOp> candidate = cast<ModuleOp>(child->clone());
  SmallVector<::mlir::taskflow::TaskflowTaskOp> tasks;
  candidate->walk(
      [&](::mlir::taskflow::TaskflowTaskOp task) { tasks.push_back(task); });
  if (tasks.size() != source.tasks.size())
    return fail("source task correspondence changed while cloning");

  for (auto task : tasks) {
    Region &region = task->getRegion(0);
    if (!llvm::hasSingleElement(region))
      return fail("task body must have exactly one block");
    Block &body = region.front();
    Operation *yield = body.getTerminator();
    if (!isa<::mlir::taskflow::TaskflowYieldOp>(yield) ||
        yield->getNumOperands() != task->getNumResults())
      return fail("task result/yield correspondence is unsupported");

    SmallVector<Value> arguments;
    arguments.append(task.getWillReads().begin(), task.getWillReads().end());
    arguments.append(task.getWillWrites().begin(), task.getWillWrites().end());
    arguments.append(task.getValueInputs().begin(),
                     task.getValueInputs().end());
    if (arguments.size() != body.getNumArguments())
      return fail("task body arguments do not match its actual interface");
    IRMapping mapping;
    for (auto [argument, value] : llvm::zip(body.getArguments(), arguments)) {
      if (argument.getType() != value.getType())
        return fail("task argument type differs from its actual operand");
      mapping.map(argument, value);
    }
    OpBuilder builder(task);
    for (Operation &operation : body.without_terminator())
      builder.clone(operation, mapping);
    SmallVector<Value> results;
    for (Value value : yield->getOperands())
      results.push_back(mapping.lookupOrDefault(value));
    task->replaceAllUsesWith(results);
    task.erase();
  }

  bool taskflowRemains = false;
  candidate->walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() == "taskflow")
      taskflowRemains = true;
  });
  if (taskflowRemains)
    return fail("a Taskflow operation remains after actual-body extraction");
  auto function =
      candidate->lookupSymbol<func::FuncOp>(source.function.getName());
  if (!function)
    return fail("child function symbol was not preserved");
  function->setAttr("llvm.emit_c_interface", UnitAttr::get(child.getContext()));
  (*candidate)
      ->setAttr("orbit.cpu.source_fingerprint",
                StringAttr::get(child.getContext(), source.bodyFingerprint));
  (*candidate)
      ->setAttr("orbit.cpu.lowering",
                StringAttr::get(child.getContext(), "actual-task-body-inline"));
  if (failed(verify(*candidate)))
    return fail("extracted CPU module failed MLIR verification");
  plainCPU = std::move(candidate);
  return true;
}
