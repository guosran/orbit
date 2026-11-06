#ifndef ORBIT_TASKFLOW_DEVICE_TRANSFER_AUDIT_H
#define ORBIT_TASKFLOW_DEVICE_TRANSFER_AUDIT_H
#include "Taskflow/DeviceTransferCost.h"
namespace orbit::taskflow {
// The caller binds parsed genericFinalAIE bytes, compile-command chain and ELF
// hash in its receipt. This checks the finite final-IR geometry/connectivity;
// it does not authenticate an ELF or infer device traffic from IR.
bool deriveDeviceTransferBackendBinding(const GenericAnalysis &,
    const GenericMappingPlan &, mlir::ModuleOp genericFinalAIE,
    const DeviceTransferBackendBinding &identityMetadata,
    DeviceTransferBackendBinding &output, llvm::json::Value &witness,
    std::string &error);
} // namespace orbit::taskflow
#endif
