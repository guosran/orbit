#ifndef ORBIT_TASKFLOW_NATIVE_TRANSFORMS_H
#define ORBIT_TASKFLOW_NATIVE_TRANSFORMS_H
#include "mlir/IR/BuiltinOps.h"
#include <string>
namespace orbit::taskflow {
struct ProgramMappingChoice;
bool hasSplittableNativeLoops(mlir::ModuleOp module);
// Clone/split each eligible task-body outer affine loop into two contiguous
// sequential ranges. Mathematics and Taskflow task dependencies are unchanged.
bool splitNativeOuterLoops(mlir::ModuleOp module, std::string &error);
// Distribute a write-only prologue from an independent rectangular loop nest.
// This preserves each output element's update order; no arithmetic is matched.
bool distributeNativeInitialization(mlir::ModuleOp module, std::string &error);
// Tiling an explicit whole-program realization switches it back to
// tiled-phase-local; readonly-panel scope composes with ordinary tiling.
bool setNativeAIRTileSizes(mlir::ModuleOp module, int64_t m, int64_t n,
                           int64_t k, std::string &error);
// Materialize a whole-program mapping choice; legality is checked against
// source accesses and target capabilities by the shared planner.
// Read syntactically canonical public policy and target realization without
// mutating source. Legality/capacity remain the shared planner's responsibility.
bool readNativeMappingChoice(mlir::ModuleOp, ProgramMappingChoice &,
                             std::string &error);
// Canonicalize legacy and public spellings before candidate identity/caching.
bool normalizeNativeMappingChoice(mlir::ModuleOp, std::string &error);
bool setNativeMappingChoice(mlir::ModuleOp module,
                            const ProgramMappingChoice &, std::string &error);
bool setNativeImplementation(mlir::ModuleOp module, const std::string &kind,
                             int64_t lanes, std::string &error);
} // namespace orbit::taskflow
#endif
