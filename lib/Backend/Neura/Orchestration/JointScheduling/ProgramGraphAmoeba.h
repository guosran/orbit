#ifndef AMOEBA_PROGRAM_GRAPH_BACKEND_H
#define AMOEBA_PROGRAM_GRAPH_BACKEND_H
#include "Taskflow/Search.h"
#include "Taskflow/AmoebaMapping.h"
#include "AnalyticalTaskCostCatalog.h"
#include "mlir/IR/BuiltinOps.h"
namespace mlir::amoeba::neura::joint_scheduling {
struct ProgramGraphAmoebaOptions {
  std::string function,architecture,architectureContract,model,cache,sourceRepository,sourceCommit,architectureTransferCatalog,modelNamespace;
  MLIRContext *context=nullptr;
};
bool configureProgramGraphAmoebaSearch(ModuleOp,::orbit::taskflow::SearchOptions &,
    const ProgramGraphAmoebaOptions &,std::string &);
// Calls the same unchanged predictor/cache/duration logic as original search.
bool predictProgramGraphTaskCosts(ModuleOp,func::FuncOp,
    const ProgramGraphAmoebaOptions &,StringRef graphId,StringRef output,
    ArrayRef<::orbit::taskflow::AmoebaTaskShape>,int64_t peRows,int64_t peColumns,
    std::vector<TaskShapeChoice> &,std::vector<TaskShapeCost> &,std::vector<int64_t> &,std::string &);
}
#endif
