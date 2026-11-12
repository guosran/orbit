#ifndef ORBIT_TASKFLOW_SYNC_BOUNDARY_MAPPING_H
#define ORBIT_TASKFLOW_SYNC_BOUNDARY_MAPPING_H

#include "Taskflow/MechanismCost.h"
#include "Taskflow/SyncState.h"

namespace orbit::taskflow {
// Runtime evidence, not an inferred allocation/page size. Logical tensor bytes
// occupy [storageOffset, storageOffset + tensorBytes) in the observed BO. The
// initial adapter requires a distinct physical BO per storage (no
// suballocation).
struct SyncRuntimeBOBinding {
  size_t storage = 0;
  std::string physicalId;
  uint64_t generation = 0;
  int64_t observedExtentBytes = 0, storageOffset = 0;
  std::vector<size_t> copies;
  std::string extentEvidence;
};
struct SyncBoundaryObservation {
  SyncObservation observation;
  int64_t mechanismItem = -1; // -1: output poison outside the measured call
  std::vector<size_t> copies, publicEvents, regionEvents;
};
struct SyncBoundaryQuery {
  size_t mechanismItem = 0;
  int64_t region = -1, storage = -1;
  SyncDirection direction = SyncDirection::ToDevice;
  std::vector<size_t> copies, publicEvents, regionEvents, sourceTasks;
  SyncQueryResult result;
};
struct SyncBoundaryTrace {
  std::string graphIdentity, factsIdentity;
  std::string origin = "static-success-path-projection";
  std::vector<std::string> requiredPredicates = {
      "all-declared-waits-succeed", "no-unrecorded-host-or-device-access",
      "runtime-observed-bo-extents-and-aliases-match-bindings"};
  std::vector<SyncRuntimeBOBinding> bindings;
  std::vector<SyncBoundaryObservation> observations;
  std::vector<SyncBoundaryQuery> queries;
};
// Projects a successful serial execution of the checked source/runtime
// contract. This is NOT a claim that these observations were measured on a
// device. Wait success is an explicit scenario assumption. Unobserved BO
// padding stays unknown.
bool collectSyncBoundaryQueries(const ProgramGraph &, const MechanismFacts &,
                                const std::vector<SyncRuntimeBOBinding> &,
                                const SyncQueryContext &, SyncBoundaryTrace &,
                                std::string &error);
llvm::json::Value serializeSyncBoundaryTrace(const SyncBoundaryTrace &);
} // namespace orbit::taskflow
#endif
