#ifndef AMOEBA_PROGRAM_GRAPH_COMMUNICATION_H
#define AMOEBA_PROGRAM_GRAPH_COMMUNICATION_H
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
namespace orbit::taskflow {
struct ProgramGraph;
}
namespace mlir::amoeba::neura::joint_scheduling {
// Synchronous, thread-local object binding, not a serialized graph import.
// The source graph and its owning MLIR module must outlive this scope.
class ScopedProgramGraphCommunication {
  const ::orbit::taskflow::ProgramGraph *previous;

public:
  explicit ScopedProgramGraphCommunication(
      const ::orbit::taskflow::ProgramGraph *);
  ~ScopedProgramGraphCommunication();
  ScopedProgramGraphCommunication(const ScopedProgramGraphCommunication &) =
      delete;
  ScopedProgramGraphCommunication &
  operator=(const ScopedProgramGraphCommunication &) = delete;
};
const ::orbit::taskflow::ProgramGraph *activeProgramGraphCommunication();
FailureOr<TaskEdgeGraph> buildTaskEdgeGraphFromProgramGraph(
    func::FuncOp, const ::orbit::taskflow::ProgramGraph &, std::string &);
} // namespace mlir::amoeba::neura::joint_scheduling
#endif
