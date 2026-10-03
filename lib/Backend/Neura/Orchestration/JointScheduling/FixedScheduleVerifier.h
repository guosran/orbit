//===- FixedScheduleVerifier.h - Score one fixed 4x4 schedule -*- C++ -*-===//
#ifndef AMOEBA_FIXED_SCHEDULE_VERIFIER_H
#define AMOEBA_FIXED_SCHEDULE_VERIFIER_H

#include "ExactScheduleSpace.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {

// A bridge to the production TaskCommunicationModel. Querying with reserve=false
// compares rectangle endpoint pairs; reserve=true commits the chosen pair in
// the current trial, exactly as production TaskScheduler does.
class FixedScheduleCommunication {
public:
  // A checkpoint is valid only for the provider instance that created it and
  // the provider type tag it assigned.  The payload is deliberately opaque so
  // checkpoint users never need RTTI or knowledge of production internals.
  struct ReservationCheckpoint {
    const FixedScheduleCommunication *owner = nullptr;
    uint64_t typeTag = 0;
    std::shared_ptr<const void> state;
  };

  virtual ~FixedScheduleCommunication() = default;
  virtual bool getPredecessors(unsigned consumer,
                               std::vector<unsigned> &predecessors,
                               std::string &error) const = 0;
  virtual void resetReservations() = 0;
  virtual void beginTrial() = 0;
  virtual void finishTrial(bool commit) = 0;
  // A completed-subtree memo may reuse a branch only when the communication
  // implementation exposes every mutable field that can affect a future
  // transfer query.  Legacy/custom models fail closed by default.
  virtual bool supportsCanonicalReservationState() const { return false; }
  virtual bool getCanonicalReservationState(std::string &state,
                                            std::string &error) const {
    state.clear();
    error = "communication model does not expose canonical reservation state";
    return false;
  }
  // Optional exact reservation save/restore. Providers that cannot expose
  // every future-affecting field fail closed and retain prefix replay.
  virtual bool supportsReservationCheckpoint() const { return false; }
  virtual bool captureReservationCheckpoint(
      ReservationCheckpoint &checkpoint, std::string &error) const {
    checkpoint = {};
    error = "communication model does not expose reservation checkpoints";
    return false;
  }
  virtual bool restoreReservationCheckpoint(
      const ReservationCheckpoint &checkpoint, std::string &error) {
    (void)checkpoint;
    error = "communication model does not expose reservation checkpoints";
    return false;
  }
  // A lower bound valid for every endpoint, release cycle and reservation
  // state. Models without such a proof retain the conservative zero bound.
  virtual bool getTransferLowerBound(unsigned, unsigned, int64_t &cycles,
                                    std::string &) const {
    cycles = 0;
    return true;
  }
  // Optional endpoint-conditioned lower bound.  The default delegates to the
  // universal edge bound, so existing mock models retain identical behavior.
  // A model may return cycles=0 for an unavailable/disconnected endpoint;
  // endpoint legality is still decided by getTransferReadyCycle().
  virtual bool getTransferLowerBoundForEndpoints(
      unsigned producer, unsigned consumer, int sourceRow, int sourceCol,
      int destinationRow, int destinationCol, int64_t &cycles,
      std::string &error) const {
    (void)sourceRow;
    (void)sourceCol;
    (void)destinationRow;
    (void)destinationCol;
    return getTransferLowerBound(producer, consumer, cycles, error);
  }
  // A fixed-shape search may memoize the minimum of the endpoint-conditioned
  // lower bounds for one source/destination rectangle pair.  This is safe
  // only when the provider proves that getTransferLowerBoundForEndpoints()
  // depends on the immutable task/architecture/payload model and the queried
  // endpoints, never on committed or trial reservations.  Custom providers
  // fail closed by default; their existing endpoint-query path is unchanged.
  virtual bool supportsStaticEndpointLowerBoundCache() const { return false; }
  virtual bool getTransferReadyCycle(unsigned producer, unsigned consumer,
                                     int sourceRow, int sourceCol,
                                     int destinationRow, int destinationCol,
                                     int64_t producerFinish, bool reserve,
                                     int64_t &readyCycle,
                                     std::string &error) = 0;
};

// Inventory-level evidence is distinct from validity of one fixed schedule.
// The current exact-schedule enumerator emits communication_mode=none, omits
// dispatch order, and does not enumerate communication-ready start events.
// Its records must therefore fail this gate even if one schedule replays.
struct FixedScheduleCoverageEvidence {
  std::string communicationMode;
  bool enumerationComplete = false;
  bool dispatchOrderRecorded = false;
  bool communicationStartEventsProven = false;
};

// Only a future source-owned, complete communication-aware inventory can pass.
// The caller must supply recorded evidence from that inventory, not infer it
// from a successful per-candidate replay.
bool certifyExplicitCommunicationCoverage(
    const FixedScheduleCoverageEvidence &evidence, std::string &rejection);

struct FixedScheduleScore {
  bool valid = false;
  std::string rejection;
  int64_t predictedMakespan = 0;
  uint64_t replayedEdges = 0;
};

// Validate one complete fixed-shape schedule and its explicit replay order.
// ExactScheduleTask::duration must be a latency derived from ML-predicted II,
// startup and trip count by the caller; measured mapper costs are not input.
// Failure/rejection never certifies that the encompassing search is complete.
// Communication event-normalization of the search is a separate proof gate.
FixedScheduleScore verifyAndScoreFixedSchedule(
    int gridRows, int gridCols, const std::vector<ExactScheduleTask> &tasks,
    const std::vector<ExactSchedulePlacement> &placements,
    const std::vector<unsigned> &taskOrder,
    FixedScheduleCommunication &communication);

} // namespace mlir::amoeba::neura::joint_scheduling
#endif // AMOEBA_FIXED_SCHEDULE_VERIFIER_H
