#ifndef ORBIT_TASKFLOW_HETEROGENEOUS_SEARCH_H
#define ORBIT_TASKFLOW_HETEROGENEOUS_SEARCH_H
#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/Search.h"
namespace orbit::taskflow {
using HeterogeneousAIREmitter = std::function<bool(const GenericAnalysis &,
    const GenericMappingPlan &, const std::string &, std::string &)>;
bool readHeterogeneousChoice(mlir::ModuleOp, const llvm::json::Value &,
    HeterogeneousMappingChoice &, std::string &);
bool exportHeterogeneousProgram(mlir::ModuleOp, const llvm::json::Value &,
    const std::string &directory, llvm::json::Value &manifest,
    std::string &, HeterogeneousAIREmitter = {});
bool configureHeterogeneousSearch(mlir::ModuleOp, SearchOptions &, std::string &);
} // namespace orbit::taskflow
#endif
