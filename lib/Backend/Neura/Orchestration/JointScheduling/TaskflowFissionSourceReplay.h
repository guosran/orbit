#ifndef ORBIT_TASKFLOW_FISSION_SOURCE_REPLAY_H
#define ORBIT_TASKFLOW_FISSION_SOURCE_REPLAY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::amoeba::neura::joint_scheduling {
// The input is after construct-hyperblock/classify, before Neura conversion.
llvm::StringRef taskflowFissionCanonicalLoweringPipeline();
void registerTaskflowFissionCanonicalLoweringDependencies(
    DialectRegistry &registry);
mlir::LogicalResult lowerPreparedTaskflowFissionSource(
    mlir::ModuleOp source, mlir::OwningOpRef<mlir::ModuleOp> &lowered,
    std::string &error);
mlir::LogicalResult verifyTaskflowFissionCanonicalLowering(
    mlir::ModuleOp source, mlir::ModuleOp canonicalNeura, std::string &error);
std::string printTaskflowFissionModuleWitness(mlir::ModuleOp module);
} // namespace mlir::amoeba::neura::joint_scheduling

#endif
