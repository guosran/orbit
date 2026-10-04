//===- OriginalAmoebaProfileCostAdapter.h ----------------------*- C++ -*-===//

#ifndef AMOEBA_ORIGINAL_AMOEBA_PROFILE_COST_ADAPTER_H
#define AMOEBA_ORIGINAL_AMOEBA_PROFILE_COST_ADAPTER_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace mlir {
class Pass;
class Operation;
namespace amoeba::neura::joint_scheduling {

inline constexpr llvm::StringLiteral kOriginalAmoebaProfileBindingSchema =
    "amoeba-original-profile-body-binding-v1";

struct OriginalAmoebaProfileBodyEvidence {
  std::string task;
  int64_t staticTripCount = 0;
  std::string taskSignature;
  std::string counterSignature;
  std::string kernelBindingSignature;
  std::string normalizedMapperBody;
};

struct OriginalAmoebaProfileBodyExport {
  std::string function;
  std::string functionSignature;
  std::string functionResultSignature;
  std::map<std::string, OriginalAmoebaProfileBodyEvidence> tasks;
};

bool readOriginalAmoebaProfileBodyExport(
    llvm::StringRef path, llvm::StringRef expectedFunction,
    llvm::ArrayRef<std::string> expectedTasks,
    OriginalAmoebaProfileBodyExport &result, std::string &error);

// Rebuilds the exact pre-mapper wrapper used by the original TaskProfiler from
// a current task's neura.kernel and returns the locally-scoped printed
// __task_profile__ function after the archived lowering pipeline.
bool computeOriginalAmoebaNormalizedMapperBody(
    Operation *task, std::string &normalizedMapperBody, std::string &error);

} // namespace amoeba::neura::joint_scheduling
namespace amoeba::neura {
std::unique_ptr<Pass> createOriginalAmoebaProfileCostAdapterPass();
} // namespace amoeba::neura
} // namespace mlir

#endif // AMOEBA_ORIGINAL_AMOEBA_PROFILE_COST_ADAPTER_H
