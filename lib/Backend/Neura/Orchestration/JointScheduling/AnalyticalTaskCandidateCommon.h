//===- AnalyticalTaskCandidateCommon.h -----------------------*- C++ -*-===//
//
// Shared task metadata and file/JSON helpers for analytical candidate spaces.
// Candidate-space implementations provide their own shape and traversal
// types, so a future temporal space can reuse these utilities independently.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_ANALYTICAL_TASK_CANDIDATE_COMMON_H
#define AMOEBA_ANALYTICAL_TASK_CANDIDATE_COMMON_H

#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

inline constexpr llvm::StringLiteral kCandidateSchema =
    "amoeba-analytical-task-candidates";
// Stores immutable identity and the compile-time trip count from one Taskflow
// task. Tasks without a Taskflow counter represent one execution.
struct TaskMetadata {
  taskflow::TaskflowTaskOp op;
  std::string name;
  // Number of scheduled mapper firings. This intentionally excludes internal
  // loops that the source-owned certificate proves fully expanded into each
  // firing; their source work is reported separately below.
  int64_t tripCount = 1;
  int64_t taskflowTripCount = 1;
  int64_t sourceIterationMultiplicity = 1;
  int64_t sourceIterationWorkCount = 1;
  bool sourceIterationDomainCertified = false;
  bool sourceIterationDomainComplete = false;
  bool tripCountKnown = true;
  std::string sourceIterationDomainStatus = "legacy-unproved";
  std::string sourceIterationDomainReason;
  SmallVector<int64_t> expandedInternalExtents;
};

struct TaskIterationDomainMetadata {
  bool sourceCertified = false;
  bool complete = false;
  bool countKnown = false;
  int64_t taskflowTripCount = 1;
  int64_t internalMultiplicity = 1;
  int64_t sourceIterationWorkCount = 1;
  int64_t effectiveMapperFiringCount = 1;
  std::string status;
  std::string reason;
  SmallVector<int64_t> expandedInternalExtents;
};

// Recomputes current Taskflow extents and validates the source-domain
// certificate/binding. A certified task reports both total source work and
// mapper firings after proven internal full expansion.
FailureOr<TaskIterationDomainMetadata>
resolveTaskIterationDomain(taskflow::TaskflowTaskOp task,
                           std::string &error);

// Internal metadata view for the replica-partition proof. It validates the
// source certificate and current counter arithmetic while allowing the
// current body binding to be stale until the complete child group has been
// compared with its canonical source. Ordinary scoring must use the strict
// overload above.
FailureOr<TaskIterationDomainMetadata>
resolveTaskIterationDomainForSourcePartitionProof(
    taskflow::TaskflowTaskOp task, std::string &error);

FailureOr<llvm::SmallVector<TaskMetadata>>
collectAnalyticalTaskMetadata(func::FuncOp func, std::string &error);

FailureOr<llvm::SmallVector<TaskMetadata>>
collectAnalyticalTaskMetadataForSourcePartitionProof(func::FuncOp func,
                                                     std::string &error);


FailureOr<func::FuncOp> selectTaskFunction(ModuleOp module,
                                           llvm::StringRef requested,
                                           std::string &error);

std::string makeSequentialCandidateId(uint64_t index);

// Writes one JSON object as one JSONL record.
void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object);

// Publishes a complete output atomically so consumers never read a partial
// candidate manifest.
bool writeAtomically(llvm::StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> writeBody,
                     std::string &error);

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_TASK_CANDIDATE_COMMON_H
