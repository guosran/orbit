#ifndef ORBIT_TASKFLOW_NATIVE_TRANSFORMS_H
#define ORBIT_TASKFLOW_NATIVE_TRANSFORMS_H
#include "mlir/IR/BuiltinOps.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace orbit::taskflow {
struct ProgramMappingChoice;
struct GenericAnalysis;
// Validate compiler initializer provenance against actual constant-store
// source. Selection metadata stays outside the function's numeric source
// fingerprint.
bool readNativeInitializerTasks(const GenericAnalysis &analysis,
                                std::vector<size_t> &tasks, std::string &error);
// Sparse input policy. Unlike GenericTaskImplementation, this does not contain
// a derived parallel loop. An explicit scalar entry overrides the module
// policy.
struct NativeTaskImplementationChoice {
  size_t task = 0;
  std::string kind = "scalar";
  int64_t lanes = 1;
};
bool validateNativeImplementation(const std::string &kind, int64_t lanes,
                                  std::string &error);
bool readNativeTaskImplementations(
    mlir::ModuleOp module, size_t taskCount,
    std::vector<NativeTaskImplementationChoice> &choices, std::string &error);
bool setNativeTaskImplementations(
    mlir::ModuleOp module, size_t taskCount,
    const std::vector<NativeTaskImplementationChoice> &choices,
    std::string &error);
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
// mutating source. Legality/capacity remain the shared planner's
// responsibility.
bool readNativeMappingChoice(mlir::ModuleOp, ProgramMappingChoice &,
                             std::string &error);
// Canonicalize legacy and public spellings before candidate identity/caching.
bool normalizeNativeMappingChoice(mlir::ModuleOp, std::string &error);
bool setNativeMappingChoice(mlir::ModuleOp module, const ProgramMappingChoice &,
                            std::string &error);
bool setNativeImplementation(mlir::ModuleOp module, const std::string &kind,
                             int64_t lanes, std::string &error);
} // namespace orbit::taskflow
#endif
