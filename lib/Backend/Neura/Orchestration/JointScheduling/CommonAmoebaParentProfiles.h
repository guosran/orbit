//===- CommonAmoebaParentProfiles.h ---------------------------*- C++ -*-===//

#ifndef AMOEBA_COMMON_AMOEBA_PARENT_PROFILES_H
#define AMOEBA_COMMON_AMOEBA_PARENT_PROFILES_H

#include "AnalyticalTaskCandidateCommon.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {

inline constexpr llvm::StringLiteral kCommonAmoebaParentProfileProvenance =
    "source-owned-common-parent-profile-only-mapped-duration-v1";
inline constexpr llvm::StringLiteral
    kCommonAmoebaParentProfileBindingSchema =
        "orbit-original-f45-common-parent-profile-binding-v1";
inline constexpr llvm::StringLiteral
    kCommonAmoebaParentProfileDurationFormula =
        "ceil(structural_startup_cycles + compiled_ii * "
        "(sample_trip_count - 1))";

struct CommonAmoebaMapperShapeProfile {
  std::string shape;
  int64_t cgraCount = 0;
  int64_t mapperTileRows = 0;
  int64_t mapperTileCols = 0;
  int64_t compiledII = 0;
  int64_t steps = 0;
  int64_t materializedOperationCount = 0;
  int64_t sampleTripCount = 0;
  int64_t estimatedLatency = 0;
  int64_t stepsFormulaEstimatedLatency = 0;
  int64_t sourceIterationWorkCount = 0;
  std::string sourceIterationDomainStatus;
  double structuralStartupCycles = 0.0;
  std::string preMapperWrapperBytes;
};

struct CommonAmoebaTaskProfiles {
  std::string task;
  int64_t sampleTripCount = 0;
  int64_t sourceIterationWorkCount = 0;
  std::string sourceIterationDomainStatus;
  std::map<std::string, CommonAmoebaMapperShapeProfile> profilesByShape;
};

struct CommonAmoebaParentProfiles {
  std::string profilePath;
  std::string canonicalModulePath;
  std::string function;
  std::string candidateId;
  std::string candidateScope;
  std::string profileProvenance;
  std::string canonicalModuleWitness;
  std::string canonicalFunctionWitness;
  std::string architectureSpecPath;
  std::string architectureSpecText;
  int64_t gridRows = 0;
  int64_t gridCols = 0;
  int64_t perCgraTileRows = 0;
  int64_t perCgraTileCols = 0;
  std::vector<CommonAmoebaTaskProfiles> tasks;
  std::map<std::string, unsigned> taskIndices;
};

struct CommonAmoebaF45TaskChoice {
  std::string task;
  std::string selectedShape;
  int64_t selectedCgraCount = 0;
  int64_t mapperTileRows = 0;
  int64_t mapperTileCols = 0;
  int64_t activeReplicas = 0;
  int64_t actualTripCount = 0;
  int64_t compiledII = 0;
  int64_t steps = 0;
  int64_t materializedOperationCount = 0;
  int64_t sourceSchedulerDuration = 0;
};

// Validates the native MapJoint parent profile against the pre-F45 canonical
// module, the current F45-scheduled module's semantic projection, and the
// current source task inventory. Only an explicit allowlist of known scheduler
// and source-certificate attrs is ignored during semantic projection; task
// kernels are also checked byte-for-byte against the shared pre-mapper wrapper.
bool verifyCommonAmoebaParentProfiles(
    llvm::StringRef profilePath, llvm::StringRef canonicalModulePath,
    llvm::StringRef expectedFunction, ModuleOp currentModule,
    func::FuncOp currentFunction, llvm::ArrayRef<TaskMetadata> currentTasks,
    llvm::ArrayRef<CommonAmoebaF45TaskChoice> f45Choices,
    int64_t runtimeIICeiling, CommonAmoebaParentProfiles &result,
    std::string &error);

// Rebind only after verifyCommonAmoebaParentProfiles has authenticated the
// pre-F45 canonical module, current semantic projection, and exact mapper
// wrappers. This refreshes stale imported control-binding strings caused by
// the validated F45 scheduling attrs; it does not change source-domain bounds
// or partition evidence.
bool refreshCommonAmoebaSourceIterationBindings(func::FuncOp function,
                                                std::string &error);

} // namespace mlir::amoeba::neura::joint_scheduling

#endif // AMOEBA_COMMON_AMOEBA_PARENT_PROFILES_H
