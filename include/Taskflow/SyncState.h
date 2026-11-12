#ifndef ORBIT_TASKFLOW_SYNC_STATE_H
#define ORBIT_TASKFLOW_SYNC_STATE_H

#include "llvm/Support/JSON.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace orbit::taskflow {
struct PhysicalBORange {
  std::string bo;
  uint64_t generation = 0;
  int64_t offset = 0, bytes = 0;
};
enum class SyncDirection { ToDevice, FromDevice };
enum class SyncObservationKind {
  HostRead,
  HostWrite,
  SyncBegin,
  SyncComplete,
  DeviceSubmit,
  WaitSuccess,
  WaitFailure,
  UnknownMutation
};
struct SyncDeclaredAccess {
  PhysicalBORange range;
  bool read = false, write = false;
};
struct SyncObservation {
  // Positive, strictly increasing IDs establish order, never elapsed timing.
  uint64_t id = 0;
  SyncObservationKind kind = SyncObservationKind::UnknownMutation;
  PhysicalBORange range;
  SyncDirection direction = SyncDirection::ToDevice;
  // A sync begin/complete pair shares an operation ID. Device submit/wait
  // pairs share a command ID. Success is not inferred from submission alone.
  std::string operation;
  std::vector<SyncDeclaredAccess> accesses;
};
enum class SyncCallForm { ImplicitFullBuffer, ExplicitRange };
enum class SyncObserverBoundary {
  MinimalAPI,
  DetailedCompatibility,
  BatchOuter
};
struct SyncQueryContext {
  std::optional<SyncCallForm> callForm;
  std::optional<SyncObserverBoundary> observerBoundary;
  std::string runtimeIdentity, observerIdentity;
};
struct SyncRangeState {
  PhysicalBORange range;
  bool known = false, hostVisible = false, deviceVisible = false;
  uint64_t contentEpoch = 0;
  uint64_t hostWriteEpoch = 0, hostReadEpoch = 0;
  uint64_t deviceReadEpoch = 0, deviceWriteEpoch = 0;
  uint64_t toCompleteEpoch = 0, fromCompleteEpoch = 0;
  std::vector<uint64_t> evidence;
};
struct SyncQueryResult {
  PhysicalBORange range;
  SyncQueryContext context;
  bool known = false, homogeneous = false, pending = false;
  std::string state, reason;
  std::vector<size_t> copies; // logical evidence aliases, not BO identity
  std::vector<SyncRangeState> segments;
};
class SyncState {
public:
  SyncState();
  ~SyncState();
  SyncState(SyncState &&) noexcept;
  SyncState &operator=(SyncState &&) noexcept;
  SyncState(const SyncState &) = delete;
  SyncState &operator=(const SyncState &) = delete;
  bool registerPhysicalBO(const std::string &id, uint64_t generation,
                          int64_t totalBytes, const std::vector<size_t> &copies,
                          std::string &error);
  bool observe(const SyncObservation &, std::string &error);
  bool query(const PhysicalBORange &, const SyncQueryContext &,
             SyncQueryResult &, std::string &error) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
const char *syncCallFormName(SyncCallForm);
const char *syncObserverBoundaryName(SyncObserverBoundary);
llvm::json::Value serializeSyncQuery(const SyncQueryResult &);
} // namespace orbit::taskflow
#endif
