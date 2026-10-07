//===- CommonMapperReplayWrapper.h ----------------------------*- C++ -*-===//
// Shared source-owned pre-mapper wrapper construction for native replay.
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_COMMON_MAPPER_REPLAY_WRAPPER_H
#define AMOEBA_COMMON_MAPPER_REPLAY_WRAPPER_H

#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <string>

namespace mlir::amoeba::neura::joint_scheduling {

// Builds the exact pre-mapper function recorded by source-owned common parent
// profile export. `owner` keeps the function and cloned kernel alive. The
// caller chooses the symbol name; profile export uses
// `__joint_task_mapper_replay__`.
func::FuncOp buildCommonMapperReplayWrapper(taskflow::TaskflowTaskOp task,
                                           ModuleOp owner,
                                           llvm::StringRef symbolName,
                                           std::string &error);

// Print the wrapper with the same flags used when MapJointSchedulingTasksPass
// records pre_mapper_wrapper_bytes.
std::string printCommonMapperReplayWrapper(func::FuncOp wrapper);

} // namespace mlir::amoeba::neura::joint_scheduling

#endif // AMOEBA_COMMON_MAPPER_REPLAY_WRAPPER_H
