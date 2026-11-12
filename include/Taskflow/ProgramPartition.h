#ifndef ORBIT_TASKFLOW_PROGRAM_PARTITION_H
#define ORBIT_TASKFLOW_PROGRAM_PARTITION_H
#include "Taskflow/GenericAnalysis.h"
namespace orbit::taskflow {
// A whole-program fission choice: independent source-axis instances execute
// in bounded commands. This is not a new compute graph or backend scheduler.
struct ProgramPartitionPlan {
  std::string sourceFingerprint, planId;
  int64_t extent = 0, chunk = 0;
  std::vector<int64_t> storageAxes; // -1: replicated immutable input
};
bool planProgramPartition(const GenericAnalysis &, int64_t chunk,
                          ProgramPartitionPlan &, std::string &error);
bool materializeProgramPartition(mlir::ModuleOp, const ProgramPartitionPlan &,
                                 int64_t extent,
                                 mlir::OwningOpRef<mlir::ModuleOp> &,
                                 std::string &error);
llvm::json::Value serializeProgramPartition(const GenericAnalysis &,
                                            const ProgramPartitionPlan &);
llvm::json::Value evaluatePartitionedGenericMLIR(mlir::ModuleOp,
                                                 const llvm::json::Value &,
                                                 std::string &error);
// Checked aggregation of repeated complete-template metrics. Unknown costs
// pass zero here; the evaluator retains a separate uncalibrated/null status.
bool accumulateProgramPartitionMetrics(int64_t transferBytes, int64_t count,
                                       double estimatedMs, int64_t &totalBytes,
                                       double &totalMs, std::string &error);
} // namespace orbit::taskflow
#endif
