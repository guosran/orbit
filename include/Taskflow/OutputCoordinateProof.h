#ifndef ORBIT_TASKFLOW_OUTPUT_COORDINATE_PROOF_H
#define ORBIT_TASKFLOW_OUTPUT_COORDINATE_PROOF_H
#include "TaskflowDialect/TaskflowOps.h"
#include "llvm/ADT/SmallVector.h"
#include <limits>
#include <string>
namespace mlir::taskflow {
// A backend-owned proof service. Core never trusts replica metadata as proof.
struct OutputCoordinateProof {
  static constexpr unsigned kConstantAxis =
      std::numeric_limits<unsigned>::max();
  bool proven = false;
  llvm::SmallVector<unsigned> outputCounterAxes;
  llvm::SmallVector<int64_t> taskLowers, taskUppers, producedLowers,
      producedShape;
  std::string reason;
};
using OutputCoordinateProofProvider = OutputCoordinateProof (*)(TaskflowTaskOp,
                                                                int64_t,
                                                                unsigned);
void registerOutputCoordinateProofProvider(
    OutputCoordinateProofProvider provider);
OutputCoordinateProof analyzeOutputCoordinates(TaskflowTaskOp task,
                                               int64_t axis,
                                               unsigned output = 0);
} // namespace mlir::taskflow
#endif
