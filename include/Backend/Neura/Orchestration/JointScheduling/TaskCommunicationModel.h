//===- TaskCommunicationModel.h -----------------------------*- C++ -*-===//
//
// Opt-in communication model that adapts the explicit task-edge and
// architecture-level network contracts to TaskScheduler.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_JOINT_SCHEDULING_TASK_COMMUNICATION_MODEL_H
#define AMOEBA_JOINT_SCHEDULING_TASK_COMMUNICATION_MODEL_H

#include "Backend/Neura/Orchestration/JointScheduling/InterTaskNetwork.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "Backend/Neura/Orchestration/orchestration_utils.h"

#include "llvm/ADT/DenseMap.h"

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

// Adapts one strict typed task-edge graph and one validated directed network
// to the scheduler's coordinate-level transfer query.  A task-level edge
// does not identify a tile inside a multi-CGRA placement, so the scheduler
// uses the cheapest endpoint pair. Data payloads between the same task pair
// are added exactly, and each directed link is reserved until its transfer
// completes. Routing therefore minimizes load-aware arrival time.
class InterTaskNetworkCommunicationModel final
    : public ::mlir::taskflow::TaskCommunicationModel {
public:
  InterTaskNetworkCommunicationModel(TaskEdgeGraph edge_graph,
                                     InterTaskNetworkSpec network)
      : edge_graph_(std::move(edge_graph)), network_(std::move(network)),
        committedLinkIntervals_(network_.getLinks().size()),
        trialLinkIntervals_(network_.getLinks().size()),
        committedLocalIntervals_(network_.getRows() * network_.getColumns()),
        trialLocalIntervals_(network_.getRows() * network_.getColumns()) {}

  bool getTaskPredecessors(Operation *consumer,
                           llvm::SmallVectorImpl<Operation *> &predecessors,
                           std::string &error) const override;
  bool getTransferLowerBound(Operation *producer, Operation *consumer,
                             int64_t &cycles,
                             std::string &error) const override;

  // Returns a lower bound for this concrete source/destination endpoint pair.
  // A disconnected remote pair is reported as a successful zero lower bound:
  // endpoint legality remains the ready-cycle/replay check's responsibility,
  // and a missing route must never become an unjustified branch rejection.
  bool getTransferLowerBoundForEndpoints(Operation *producer,
                                         Operation *consumer, int source_row,
                                         int source_col, int destination_row,
                                         int destination_col, int64_t &cycles,
                                         std::string &error) const override;

  void resetReservations() override;
  void beginTrial() override;
  void finishTrial(bool commit) override;
  // Serializes exactly the mutable reservation state used by routeTransfer:
  // committed and trial local/link half-open intervals, including each
  // resource index. Transfer history is intentionally excluded because it is
  // diagnostic only and is not consulted by future route selection.
  bool supportsCanonicalReservationState() const override { return true; }
  bool getCanonicalReservationState(std::string &state,
                                    std::string &error) const override;
  bool supportsStaticEndpointLowerBoundCache() const override { return true; }
  bool supportsReservationCheckpoint() const override { return true; }
  bool captureReservationCheckpoint(
      ::mlir::taskflow::TaskCommunicationReservationCheckpoint &checkpoint,
      std::string &error) const override;
  bool restoreReservationCheckpoint(
      const ::mlir::taskflow::TaskCommunicationReservationCheckpoint
          &checkpoint,
      std::string &error) override;
  bool getTransferReadyCycle(Operation *producer, Operation *consumer,
                             int source_row, int source_col,
                             int destination_row, int destination_col,
                             int64_t producer_finish, bool reserve,
                             int64_t &ready_cycle, std::string &error) override;

  struct LinkInterval {
    uint32_t linkIndex = 0;
    int64_t startCycle = 0;
    int64_t endCycle = 0;
    bool localChannel = false;
    TaskNetworkCoordinate localCoordinate;
  };
  struct TransferRecord {
    Operation *producer = nullptr;
    Operation *consumer = nullptr;
    TaskNetworkCoordinate source;
    TaskNetworkCoordinate destination;
    uint64_t payloadBits = 0;
    uint64_t pathLatencyCycles = 0;
    uint64_t bottleneckBandwidthBitsPerCycle = 0;
    uint64_t transferCycles = 0;
    int64_t readyCycle = 0;
    llvm::SmallVector<uint32_t> edgeIndices;
    llvm::SmallVector<LinkInterval> links;
  };
  llvm::ArrayRef<TransferRecord> getCommittedTransfers() const {
    return committedTransfers_;
  }
  llvm::ArrayRef<TaskEdge> getTypedEdges() const {
    return edge_graph_.getEdges();
  }
  const InterTaskNetworkSpec &getNetworkSpec() const { return network_; }

private:
  // These caches belong to one immutable edge-graph/network model instance.
  // Reservations do not affect either the payload facts or the optimistic
  // endpoint metrics.  They are mutable only so const lower-bound queries can
  // memoize their first result; they are intentionally not shared globally.
  struct EndpointPathMetrics {
    bool computed = false;
    bool reachable = false;
    uint64_t latency_cycles = 0;
    uint64_t bandwidth_bits_per_cycle = 0;
  };

  struct TaskPairPayloadCache {
    bool computed = false;
    bool found = false;
    bool valid = true;
    bool unknown_payload = false;
    bool overflowed = false;
    uint64_t payload_bits = 0;
    llvm::SmallVector<uint32_t> edge_indices;
  };

  struct EndpointBoundCacheEntry {
    bool computed = false;
    bool success = false;
    int64_t cycles = 0;
    std::string error;
  };

  struct TaskPairCache {
    TaskPairPayloadCache payload;
    llvm::SmallVector<EndpointBoundCacheEntry> endpoint_bounds;
  };

  using Interval = std::pair<int64_t, int64_t>;
  struct ReservationCheckpointState {
    llvm::SmallVector<llvm::SmallVector<Interval>> committedLinkIntervals;
    llvm::SmallVector<llvm::SmallVector<Interval>> trialLinkIntervals;
    llvm::SmallVector<llvm::SmallVector<Interval>> committedLocalIntervals;
    llvm::SmallVector<llvm::SmallVector<Interval>> trialLocalIntervals;
    llvm::SmallVector<TransferRecord> committedTransfers;
    llvm::SmallVector<TransferRecord> trialTransfers;
  };

  const TaskPairPayloadCache *
  getTaskPairPayloadCache(Operation *producer, Operation *consumer) const;
  bool getEndpointPathMetrics(int source_row, int source_col,
                              int destination_row, int destination_col,
                              EndpointPathMetrics &metrics) const;
  bool getCachedEndpointLowerBound(Operation *producer, Operation *consumer,
                                   int source_row, int source_col,
                                   int destination_row, int destination_col,
                                   int64_t &cycles, std::string &error) const;

  bool routeTransfer(Operation *producer, Operation *consumer,
                     TaskNetworkCoordinate source,
                     TaskNetworkCoordinate destination, uint64_t payload_bits,
                     llvm::ArrayRef<uint32_t> edge_indices,
                     int64_t producer_finish, bool reserve,
                     int64_t &ready_cycle, std::string &error);

  TaskEdgeGraph edge_graph_;
  InterTaskNetworkSpec network_;
  static constexpr uint64_t kReservationCheckpointTypeTag =
      0x9d8f6f3f7a4c2711ULL;
  llvm::SmallVector<llvm::SmallVector<Interval>> committedLinkIntervals_;
  llvm::SmallVector<llvm::SmallVector<Interval>> trialLinkIntervals_;
  llvm::SmallVector<llvm::SmallVector<Interval>> committedLocalIntervals_;
  llvm::SmallVector<llvm::SmallVector<Interval>> trialLocalIntervals_;
  llvm::SmallVector<TransferRecord> committedTransfers_;
  llvm::SmallVector<TransferRecord> trialTransfers_;
  mutable llvm::SmallVector<EndpointPathMetrics> endpointPathCache_;
  mutable llvm::DenseMap<Operation *,
                         llvm::DenseMap<Operation *, TaskPairCache>>
      taskPairCache_;
};

// Builds the strict communication model for the current Neura architecture
// and checks that the network section dimensions match the scheduler grid.
FailureOr<std::unique_ptr<InterTaskNetworkCommunicationModel>>
createInterTaskNetworkCommunicationModel(func::FuncOp function, int grid_rows,
                                         int grid_columns, std::string &error);

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_JOINT_SCHEDULING_TASK_COMMUNICATION_MODEL_H
