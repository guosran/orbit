//===- BindSourceIterationDomainPass.cpp ----------------------*- C++ -*-===//
//
// Consumes source iteration-domain captures after the trusted canonical
// Taskflow-to-Neura control lowering and binds them to the current kernel body.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/Orchestration/JointScheduling/SourceIterationDomainPartitionProof.h"
#include "mlir/Parser/Parser.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "NeuraDialect/NeuraDialect.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {

struct BindSourceIterationDomainPass
    : PassWrapper<BindSourceIterationDomainPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BindSourceIterationDomainPass)

  StringRef getArgument() const final { return "bind-source-iteration-domain"; }
  StringRef getDescription() const final {
    return "Bind captured source iteration domains to canonical Neura control bodies";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, taskflow::TaskflowDialect,
                    neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool failedBinding = false;
    module.walk([&](TaskflowTaskOp task) {
      if (failedBinding)
        return WalkResult::interrupt();
      auto domain = task->getAttrOfType<DictionaryAttr>(
          kSourceIterationDomainAttr);
      if (!domain) {
        task.emitError("source iteration-domain binding requires an explicit "
                       "source capture from construct-hyperblock-from-task");
        failedBinding = true;
        return WalkResult::interrupt();
      }
      auto complete = domain.getAs<BoolAttr>("complete");
      if (!complete) {
        task.emitError("source iteration-domain certificate has no complete field");
        failedBinding = true;
        return WalkResult::interrupt();
      }
      // Unsupported source bodies remain in the module and are recorded as
      // unknown by fact extraction. They cannot acquire a numeric cost proof.
      if (!complete.getValue())
        return WalkResult::advance();

      std::string error;
      FailureOr<SourceIterationDomainInfo> info =
          parseSourceIterationDomain(task, error);
      if (failed(info)) {
        task.emitError() << error;
        failedBinding = true;
        return WalkResult::interrupt();
      }
      const bool capturePending =
          task->hasAttr(kSourceIterationCapturePendingAttr);
      auto existing = task->getAttrOfType<StringAttr>(
          kSourceIterationControlBindingAttr);
      auto sourceBinding = task->getAttrOfType<StringAttr>(
          kSourceIterationSourceControlBindingAttr);
      std::string expected = currentSourceIterationControlBinding(
          task, sourceIterationDomainCanonicalWitness(*info));
      if (expected.empty()) {
        task.emitError("source iteration-domain certificate has no exact "
                       "task control/body witness");
        failedBinding = true;
        return WalkResult::interrupt();
      }

      if (existing) {
        if (capturePending || existing.getValue() != expected ||
            !sourceBinding || sourceBinding.getValue().empty()) {
          task.emitError("source iteration-domain binding is stale; imported "
                         "proof attributes cannot be refreshed");
          failedBinding = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (!capturePending) {
        task.emitError("missing source iteration-domain body binding without a "
                       "pending source capture; refusing to repair imported IR");
        failedBinding = true;
        return WalkResult::interrupt();
      }

      OpBuilder builder(task.getContext());
      task->setAttr(kSourceIterationControlBindingAttr,
                    builder.getStringAttr(expected));
      task->setAttr(kSourceIterationSourceControlBindingAttr,
                    builder.getStringAttr(expected));
      task->removeAttr(kSourceIterationCapturePendingAttr);
      return WalkResult::advance();
    });
    if (failedBinding)
      signalPassFailure();
  }
};

struct VerifySourceIterationDomainPartitionsPass
    : PassWrapper<VerifySourceIterationDomainPartitionsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifySourceIterationDomainPartitionsPass)
  VerifySourceIterationDomainPartitionsPass() = default;
  VerifySourceIterationDomainPartitionsPass(const VerifySourceIterationDomainPartitionsPass &other)
      : PassWrapper(other) {}
  StringRef getArgument() const final { return "verify-source-iteration-domain-partitions"; }
  StringRef getDescription() const final {
    return "Re-prove complete source-domain coverage against an immutable canonical module";
  }
  Option<std::string> parentModule{*this, "parent-module", llvm::cl::init("")};
  Option<std::string> functionName{*this, "function", llvm::cl::init("")};
  void getDependentDialects(DialectRegistry &registry) const override {
    // Verification may deterministically replay source-owned tiling or fusion
    // in a nested pass manager. Register the same dialect closure as those
    // materializers before the outer pass can run on a threaded context.
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect, memref::MemRefDialect, scf::SCFDialect,
                    TaskflowDialect, neura::NeuraDialect>();
  }
  void runOnOperation() override {
    std::string error;
    if (parentModule.empty() || functionName.empty()) {
      getOperation().emitError("source-domain verification requires parent-module and function");
      return signalPassFailure();
    }
    OwningOpRef<ModuleOp> parent = parseSourceFile<ModuleOp>(parentModule.getValue(), ParserConfig(getOperation().getContext()));
    if (!parent) {
      getOperation().emitError("cannot read immutable canonical source-domain module");
      return signalPassFailure();
    }
    auto canonical = selectTaskFunction(*parent, functionName, error);
    auto child = selectTaskFunction(getOperation(), functionName, error);
    if (failed(canonical) || failed(child) ||
        failed(verifySourceIterationDomainPartition(*canonical, *child, error))) {
      getOperation().emitError() << "source iteration-domain partition is unproven: " << error;
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::amoeba::neura::createBindSourceIterationDomainPass() {
  return std::make_unique<BindSourceIterationDomainPass>();
}

std::unique_ptr<Pass>
mlir::amoeba::neura::createVerifySourceIterationDomainPartitionsPass() {
  return std::make_unique<VerifySourceIterationDomainPartitionsPass>();
}
