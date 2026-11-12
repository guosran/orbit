#ifndef ORBIT_TASKFLOW_PROGRAM_MATRIX_RUNTIME_H
#define ORBIT_TASKFLOW_PROGRAM_MATRIX_RUNTIME_H

#include "Taskflow/ProgramMapping.h"
#include "mlir/IR/BuiltinOps.h"

namespace orbit::taskflow {
struct ProgramGraphAIRSelection;
// Export only after validating the complete source-bound selected graph. The
// caller binds the exact canonical JSON hash to its physical receipt; loading a
// contract is not a substitute for that source-validation provenance.
llvm::json::Value
serializeProgramMatrixRuntimeContract(const ProgramGraph &,
                                      llvm::ArrayRef<int64_t> abiStorageIndices,
                                      std::string &error);
bool realizeProgramMatrixRuntime(const llvm::json::Value &contract,
                                 mlir::ModuleOp input,
                                 mlir::OwningOpRef<mlir::ModuleOp> &output,
                                 llvm::json::Value &receipt,
                                 std::string &error);
bool checkProgramMatrixRuntime(const llvm::json::Value &contract,
                               mlir::ModuleOp input, llvm::json::Value &receipt,
                               std::string &error);
llvm::json::Value serializeProgramMatrixRuntimeContract(
    const ProgramGraph &, llvm::ArrayRef<int64_t> abiStorageIndices,
    const ProgramGraphAIRSelection &, std::string &error);
// Consume generic-printed final NPU AIE after extraction. Packet inputs become
// one BD per selected transfer, with serialized completion. Selected BO copies,
// output payloads and core IR are preserved. Recompile directly with aiecc;
// rerunning aircc would erase this realization.
bool realizeProgramMatrixRuntime(const ProgramGraph &,
                                 llvm::ArrayRef<int64_t> abiStorageIndices,
                                 mlir::ModuleOp input,
                                 mlir::OwningOpRef<mlir::ModuleOp> &output,
                                 llvm::json::Value &receipt,
                                 std::string &error);
// Read-only host coverage, packet boundary, concrete DMA route/storage and
// FIFO-order checks. Core lock protocol and kernel arithmetic remain outside
// this certificate; final compilation and numerical validation are required.
bool checkProgramMatrixRuntime(const ProgramGraph &,
                               llvm::ArrayRef<int64_t> abiStorageIndices,
                               mlir::ModuleOp, llvm::json::Value &receipt,
                               std::string &error);
} // namespace orbit::taskflow
#endif
