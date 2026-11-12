#ifndef ORBIT_TASKFLOW_HETEROGENEOUS_SEARCH_H
#define ORBIT_TASKFLOW_HETEROGENEOUS_SEARCH_H
#include "Taskflow/GenericMapping.h"
#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/Search.h"
namespace orbit::taskflow {
// Existing shape-only normalization, shared by calibration and execution.
std::string heterogeneousProgramFamily(mlir::ModuleOp);
using HeterogeneousAIREmitter =
    std::function<bool(const GenericAnalysis &, const GenericMappingPlan &,
                       const std::string &, std::string &)>;
bool readHeterogeneousChoice(mlir::ModuleOp, const llvm::json::Value &,
                             HeterogeneousMappingChoice &, std::string &);
// Borrow the same region IR that owns the selected graph's source witness.
// A textually equal clone cannot substitute for its pointer-bound proof.
struct HeterogeneousSelectedRegion {
  mlir::ModuleOp source;
  const GenericMappingPlan *native = nullptr;
};
bool exportHeterogeneousProgram(
    mlir::ModuleOp, const llvm::json::Value &, const std::string &directory,
    llvm::json::Value &manifest, std::string &, HeterogeneousAIREmitter = {},
    const HeterogeneousMappingPlan *selectedParent = nullptr,
    const std::vector<HeterogeneousSelectedRegion> *selectedRegions = nullptr);
bool configureHeterogeneousSearch(mlir::ModuleOp, SearchOptions &,
                                  std::string &);
} // namespace orbit::taskflow
#endif
