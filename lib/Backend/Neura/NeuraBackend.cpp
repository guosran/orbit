//===- NeuraBackend.cpp - Neura backend integration ----------------------===//

#include "Backend/Neura/NeuraBackend.h"
#include "Backend/Neura/Orchestration/JointScheduling/ReplicaOutputCoordinateProof.h"
#include "Taskflow/OutputCoordinateProof.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/ProductionSramNativeUpperBound.h"
#include "Backend/Neura/Orchestration/JointScheduling/ProductionSramNativeCapacityGate.h"

#include "Conversion/NeuraConversionPasses.h"
#include "Conversion/RepairLuAffineDeterminantCarryPass.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/Util/ArchParser.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"

namespace {
// The common program may contain sequential loop nests (fill then GEMM, or
// sequential K chunks). Neura's counter materializer needs one nest per task.
// This adapter partitions only this closed structured subset into target
// implementation stages. The public candidate itself is not rewritten.
struct PrepareStructuredTaskflowForNeura
    : mlir::PassWrapper<PrepareStructuredTaskflowForNeura,
                        mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareStructuredTaskflowForNeura)
  llvm::StringRef getArgument() const final {
    return "prepare-structured-taskflow-for-neura";
  }
  llvm::StringRef getDescription() const final {
    return "Adapt sequential structured loop nests into dependency-ordered Neura stages";
  }
  void runOnOperation() final {
    using namespace mlir;
    using namespace mlir::taskflow;
    SmallVector<TaskflowTaskOp> tasks;
    getOperation().walk([&](TaskflowTaskOp t) { tasks.push_back(t); });
    for (auto task : tasks) {
      SmallVector<affine::AffineForOp> loops;
      for (Operation &op : task.getBody().front().without_terminator()) {
        if (auto loop = dyn_cast<affine::AffineForOp>(op)) loops.push_back(loop);
        else if (!isa<arith::ConstantOp>(op)) {
          task.emitError("Neura structured adapter supports only constants and affine loop nests");
          return signalPassFailure();
        }
      }
      if (loops.size() <= 1) continue;
      // An affine loop carrying results to another nest needs a value-state
      // fission proof; do not guess it. Memref writes are chained explicitly.
      for (auto loop : loops) {
        if (loop->getNumResults()) {
          task.emitError("Neura structured adapter does not support cross-nest SSA values");
          return signalPassFailure();
        }
      }
      if (task.getWillWrites().size() != 1 || task.getDoneWrites().size() != 1 ||
          !task.getDoneReads().empty()) {
        task.emitError("Neura structured adapter requires one output state and no read-completion results");
        return signalPassFailure();
      }
      auto yield = cast<TaskflowYieldOp>(task.getBody().front().getTerminator());
      auto write = dyn_cast<BlockArgument>(yield.getDoneWrites().front());
      if (!write || write.getOwner() != &task.getBody().front() ||
          write.getArgNumber() != task.getWillReads().size()) {
        task.emitError("Neura structured adapter requires identity output storage forwarding");
        return signalPassFailure();
      }
      TaskflowTaskOp previous;
      for (size_t i = 0; i < loops.size(); ++i) {
        IRMapping mapping;
        auto stage = cast<TaskflowTaskOp>(task->clone(mapping));
        task->getBlock()->getOperations().insert(task->getIterator(), stage);
        stage->setAttr("task_name", StringAttr::get(task.getContext(),
            task->getAttrOfType<StringAttr>("task_name").getValue().str() +
            "_neura_stage_" + std::to_string(i)));
        stage->setAttr("orbit.neura.source_task", task->getAttr("task_name"));
        stage->setAttr("orbit.neura.stage", IntegerAttr::get(
            IntegerType::get(task.getContext(), 64), i));
        if (previous)
          stage->setOperand(stage.getWillReads().size(), previous.getDoneWrites().front());
        size_t ordinal = 0;
        for (Operation &op : llvm::make_early_inc_range(stage.getBody().front()))
          if (isa<affine::AffineForOp>(op) && ordinal++ != i) op.erase();
        previous = stage;
      }
      task->replaceAllUsesWith(previous->getResults());
      task.erase();
    }
  }
};
} // namespace

using mlir::neura::Architecture;
using mlir::neura::util::ArchParser;

const Architecture &mlir::neura::getArchitecture() {
  static Architecture architecture = []() {
    ArchParser parser(mlir::amoeba::getNeuraArchitectureSpecFile());
    auto result = parser.getArchitecture();
    if (failed(result))
      llvm::report_fatal_error("[neura-backend] Failed to get architecture.");
    return std::move(*result);
  }();
  return architecture;
}

const std::string &mlir::neura::getLatencySpecFile() {
  return mlir::amoeba::getNeuraLatencySpecFile();
}

void mlir::amoeba::registerNeuraBackend(DialectRegistry &registry) {
  // Keep the standalone Taskflow verifier backend-free. Register the existing
  // authenticated Neura proof only in an executable that links this backend.
  mlir::taskflow::registerOutputCoordinateProofProvider(
      [](mlir::taskflow::TaskflowTaskOp task, int64_t axis,
         unsigned outputIndex) -> mlir::taskflow::OutputCoordinateProof {
        auto legacy =
            mlir::amoeba::neura::joint_scheduling::analyzeReplicaOutputCoordinates(
                task, axis, outputIndex);
        mlir::taskflow::OutputCoordinateProof proof;
        proof.proven = legacy.proven;
        proof.outputCounterAxes = std::move(legacy.outputCounterAxes);
        proof.taskLowers = std::move(legacy.taskLowers);
        proof.taskUppers = std::move(legacy.taskUppers);
        proof.producedLowers = std::move(legacy.producedLowers);
        proof.producedShape = std::move(legacy.producedShape);
        proof.reason = std::move(legacy.reason);
        return proof;
      });
  registry.insert<mlir::neura::NeuraDialect>();
  mlir::registerPass([] { return std::make_unique<PrepareStructuredTaskflowForNeura>(); });
  mlir::neura::registerPasses();
  mlir::registerNeuraConversionPasses();
  mlir::amoeba::neura::registerNeuraBackendPasses();
  mlir::registerPass([] {
    return mlir::amoeba::neura::joint_scheduling::createExportProductionSramUpperBoundPass();
  });
  mlir::registerPass([] {
    return mlir::amoeba::neura::joint_scheduling::createVerifyProductionSramNativeCapacityGatePass();
  });
  mlir::registerPass([] {
    return mlir::createRepairLuAffineDeterminantCarryPass();
  });
  mlir::amoeba::neura::registerTaskflowConversionPassPipeline();
}
