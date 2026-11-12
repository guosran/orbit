#include "Taskflow/AIRBackend.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
using namespace mlir;
namespace orbit::taskflow {
namespace {
std::string typeText(Type type) {
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os);
  return s;
}
llvm::json::Array shapeJSON(ArrayRef<int64_t> shape) {
  llvm::json::Array a;
  for (auto x : shape)
    a.push_back(x);
  return a;
}
std::string attrText(Attribute attr) {
  std::string s;
  llvm::raw_string_ostream os(s);
  attr.print(os);
  return s;
}
} // namespace
// This deliberately reads operations and their SSA/control-flow wiring. orbit.*
// attributes are recorded separately as proposed joins; they are excluded from
// the structural facts and cannot establish actual buffer/DMA/compute coverage.
llvm::json::Value extractGenericAIRImplementation(ModuleOp module) {
  llvm::DenseMap<Operation *, int64_t> operationIDs;
  llvm::DenseMap<Value, int64_t> values;
  int64_t nextValue = 0, nextOperation = 0;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    operationIDs[op] = nextOperation++;
    for (auto result : op->getResults())
      values[result] = nextValue++;
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (auto argument : block.getArguments())
          values[argument] = nextValue++;
  });
  llvm::json::Array operations, buffers, transfers, computations, groups,
      synchronizations;
  std::map<std::string, int64_t> counts;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    std::string name = op->getName().getStringRef().str();
    ++counts[name];
    llvm::json::Array inputs, outputs, regions, successors;
    for (auto operand : op->getOperands())
      inputs.push_back(values.lookup(operand));
    for (auto result : op->getResults())
      outputs.push_back(
          llvm::json::Object{{"value", values.lookup(result)},
                             {"type", typeText(result.getType())}});
    for (auto *successor : op->getSuccessors()) {
      llvm::json::Array blockOps;
      for (auto &child : *successor)
        blockOps.push_back(operationIDs.lookup(&child));
      successors.push_back(std::move(blockOps));
    }
    for (auto &region : op->getRegions()) {
      llvm::json::Array blocks;
      for (auto &block : region) {
        llvm::json::Array arguments, children;
        for (auto arg : block.getArguments())
          arguments.push_back(
              llvm::json::Object{{"value", values.lookup(arg)},
                                 {"type", typeText(arg.getType())}});
        for (auto &child : block)
          children.push_back(operationIDs.lookup(&child));
        blocks.push_back(
            llvm::json::Object{{"arguments", std::move(arguments)},
                               {"operations", std::move(children)}});
      }
      regions.push_back(std::move(blocks));
    }
    llvm::json::Object attributes, bindings;
    for (auto attr : op->getAttrs()) {
      if (attr.getName().strref().starts_with("orbit."))
        bindings[attr.getName().strref()] = attrText(attr.getValue());
      else
        attributes[attr.getName().strref()] = attrText(attr.getValue());
    }
    int64_t id = operationIDs.lookup(op),
            parent =
                op->getParentOp() ? operationIDs.lookup(op->getParentOp()) : -1;
    operations.push_back(llvm::json::Object{
        {"id", id},
        {"name", name},
        {"parent", parent},
        {"operands", std::move(inputs)},
        {"results", std::move(outputs)},
        {"attributes", std::move(attributes)},
        {"properties", op->getPropertiesAsAttribute()
                           ? attrText(op->getPropertiesAsAttribute())
                           : ""},
        {"regions", std::move(regions)},
        {"successors", std::move(successors)}});
    if (name == "memref.alloc" || name == "aie.buffer" ||
        name == "aie.external_buffer") {
      llvm::json::Object buffer{{"operation", id},
                                {"parent", parent},
                                {"proposed_bindings", std::move(bindings)}};
      if (op->getNumResults())
        if (auto type = dyn_cast<MemRefType>(op->getResult(0).getType())) {
          buffer["value"] = values.lookup(op->getResult(0));
          buffer["shape"] = shapeJSON(type.getShape());
          buffer["type"] = typeText(type);
          buffer["memory_space"] =
              type.getMemorySpace() ? attrText(type.getMemorySpace()) : "0";
          buffer["layout"] = attrText(type.getLayout());
          if (type.hasStaticShape() && type.getElementType().isIntOrFloat())
            buffer["bytes"] =
                type.getNumElements() *
                ((type.getElementType().getIntOrFloatBitWidth() + 7) / 8);
          else
            buffer["bytes"] = nullptr;
        }
      buffers.push_back(std::move(buffer));
    }
    if (name == "air.dma_memcpy_nd" || name == "air.channel.put" ||
        name == "air.channel.get" || name == "aie.dma_bd" ||
        name == "aiex.npu.dma_memcpy_nd")
      transfers.push_back(
          llvm::json::Object{{"operation", id},
                             {"parent", parent},
                             {"proposed_bindings", std::move(bindings)}});
    if (name == "memref.load" || name == "memref.store" ||
        name == "vector.transfer_read" || name == "vector.transfer_write" ||
        (StringRef(name).starts_with("arith.") && name != "arith.constant"))
      computations.push_back(id);
    if (name == "air.herd" || name == "aie.core" || name == "aiex.configure" ||
        name == "aiex.run")
      groups.push_back(llvm::json::Object{
          {"operation", id}, {"proposed_bindings", std::move(bindings)}});
    if (name == "air.wait_all" || name == "aie.use_lock" ||
        name == "aie.lock" || name == "aiex.dma_await_task" ||
        name == "scf.yield")
      synchronizations.push_back(id);
  });
  llvm::json::Object histogram;
  for (auto &[name, count] : counts)
    histogram[name] = count;
  return llvm::json::Object{
      {"schema", "orbit.mlir.implementation-facts.v19"},
      {"scope",
       "Actual operation structure, SSA values, CFG successors, buffer "
       "types/layouts, DMA operands and synchronization operations; proposed "
       "orbit bindings are separate and are not coverage proof."},
      {"operations", std::move(operations)},
      {"buffers", std::move(buffers)},
      {"transfers", std::move(transfers)},
      {"computations", std::move(computations)},
      {"groups", std::move(groups)},
      {"synchronizations", std::move(synchronizations)},
      {"operation_counts", std::move(histogram)}};
}
} // namespace orbit::taskflow
