#ifndef ORBIT_TASKFLOW_PROGRAM_GRAPH_AIR_LOOP_H
#define ORBIT_TASKFLOW_PROGRAM_GRAPH_AIR_LOOP_H
#include "Taskflow/ProgramMapping.h"
#include "mlir/IR/Builders.h"
#include <functional>
#include <map>
namespace orbit::taskflow {
struct ProgramGraphAIRLoopOffset {
  std::vector<int64_t> base, row, column;
};
struct ProgramGraphAIRLoopGroup {
  int64_t rows = 0, columns = 0;
  std::vector<size_t> events;
  std::vector<std::vector<size_t>> iterations;
  std::map<size_t, ProgramGraphAIRLoopOffset> hostOffsets;
};
struct ProgramGraphAIRLoopPlan {
  std::string graphId;
  std::vector<ProgramGraphAIRLoopGroup> groups;
  llvm::json::Value witness = nullptr;
};
// No optimization is performed: prove finite lexical expansion equals every
// selected event, source instance, copy version and physical allocation slot.
bool buildProgramGraphAIRLoopPlan(const ProgramGraph &,
                                  ProgramGraphAIRLoopPlan &,
                                  std::string &error);
using ProgramGraphAIRLoopEmitter =
    std::function<bool(size_t, llvm::ArrayRef<mlir::Value>)>;
// The caller's exact emitter consumes a representative event. Only host DMA
// offsets vary; source arithmetic is proven invariant under tile translation.
bool emitProgramGraphAIRLoopEvents(mlir::OpBuilder &,
                                   const ProgramGraphAIRLoopPlan &,
                                   ProgramGraphAIRLoopEmitter,
                                   std::string &error);
} // namespace orbit::taskflow
#endif
