#ifndef ORBIT_TASKFLOW_PROGRAM_TARGET_BINDING_H
#define ORBIT_TASKFLOW_PROGRAM_TARGET_BINDING_H
#include "Taskflow/StaticMapping.h"
#include "llvm/ADT/StringRef.h"
namespace orbit::taskflow {
// Borrowed references into the existing target/configuration binding. This is
// not another architecture, placement policy, or serialized configuration.
struct ProgramTargetBindingView {
  llvm::StringRef hostMemory, localMemory, computeResource, dmaResource;
  llvm::StringRef inputLink, outputLink, submissionResource;
};
const StaticResourceDomain *programDomain(const StaticConfig &, llvm::StringRef);
const StaticResource *programResource(const StaticConfig &, llvm::StringRef);
const StaticMemory *programMemory(const StaticConfig &, llvm::StringRef);
const StaticTransfer *programLink(const StaticConfig &, llvm::StringRef);
bool programOwnerExists(const StaticConfig &, llvm::StringRef);
// Includes equality; unknown owners are never related by their spelling.
bool programDomainContains(const StaticConfig &, llvm::StringRef ancestor,
                           llvm::StringRef descendant);
bool programResourceOwner(const StaticConfig &, llvm::StringRef resource,
                          std::string &owner, std::string &error);
bool programMemoryOwner(const StaticConfig &, llvm::StringRef memory,
                        std::string &owner, std::string &error);
// Idempotent minimum-reservation closure: multiple implications use max, not
// sum. An implementation can explicitly request a greater shared reservation.
bool programResourceDemands(const StaticConfig &,
    const std::map<std::string,int64_t> &base,
    std::map<std::string,int64_t> &expanded, std::string &error);
// Common reference/access/capacity checking only. Device-specific memory kinds,
// positions, supported implementations and lowering limits belong to the backend.
bool checkProgramTargetBinding(const StaticConfig &, ProgramTargetBindingView,
                               int64_t payloadCapacity, std::string &error);
} // namespace orbit::taskflow
#endif
