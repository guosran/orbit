#ifndef ORBIT_TASKFLOW_PROGRAM_MATRIX_IMPLEMENTATION_H
#define ORBIT_TASKFLOW_PROGRAM_MATRIX_IMPLEMENTATION_H

#include "Taskflow/ProgramMapping.h"
#include "mlir/IR/Builders.h"

namespace orbit::taskflow {
// A numerical contract for a new BF16 workload, not a replacement for strict
// source-ordered FP32 arithmetic. Inputs are finite BF16 with abs(value)<=1;
// the FP32 accumulator may use the matrix kernel's reduction order.
inline constexpr const char *programMatrixNumericalContract =
    "bf16-f32-accumulate-v1";
inline constexpr const char *programMatrixImplementation =
    "aie2p-bf16-mmul-f32acc";

struct ProgramMatrixTaskContract {
  size_t task = 0, a = 0, b = 0, c = 0;
  int64_t m = 0, n = 0, k = 0;
  bool initializes = false;
};
struct ProgramMatrixInstanceContract {
  ProgramMatrixTaskContract task;
  int64_t m = 0, n = 0, k = 0;
  int64_t lowerM = 0, lowerN = 0, lowerK = 0;
};

// Match the complete source body and exact affine indexing, rather than a
// task name, attribute assertion, or operation count alone.
bool analyzeProgramMatrixTask(const GenericAnalysis &, size_t task,
                              ProgramMatrixTaskContract &, std::string &error);
bool analyzeProgramMatrixInstance(const ProgramGraph &,
                                  const MappingTaskInstance &,
                                  ProgramMatrixInstanceContract &,
                                  std::string &error);

// Realize only an explicitly selected implementation. Unhandled scalar tasks
// retain the original source emitter. Uses the caller's selected local views;
// never changes placement, copies, transfer events, or source partitioning.
bool emitProgramMatrixInstance(mlir::OpBuilder &, const ProgramGraph &,
                               const MappingTaskInstance &,
                               llvm::ArrayRef<MappingRegion>,
                               llvm::ArrayRef<mlir::Value>, bool &handled,
                               std::string &error);

// Generate the exact external kernel TU named by the emitted AIR declaration.
// The caller compiles it for aie2p and binds the resulting object before aircc.
std::string programMatrixKernelSymbol(int64_t m, int64_t k, int64_t n);
std::string programMatrixKernelSource(int64_t m, int64_t k, int64_t n);
llvm::json::Value programMatrixAIRManifest(mlir::ModuleOp);
} // namespace orbit::taskflow
#endif
