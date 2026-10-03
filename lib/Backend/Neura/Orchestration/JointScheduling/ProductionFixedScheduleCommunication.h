//===- ProductionFixedScheduleCommunication.h - Task network bridge -------===//
#ifndef AMOEBA_PRODUCTION_FIXED_SCHEDULE_COMMUNICATION_H
#define AMOEBA_PRODUCTION_FIXED_SCHEDULE_COMMUNICATION_H

#include "FixedScheduleVerifier.h"

#include "Backend/Neura/Orchestration/orchestration_utils.h"

#include <string>
#include <vector>

namespace mlir {
class Operation;
namespace amoeba::neura::joint_scheduling {

// The task index order must match ExactScheduleTask indices. The supplied
// production model must be created with require_payload=true and a validated
// architecture network, e.g. createInterTaskNetworkCommunicationModel().
class ProductionFixedScheduleCommunication final
    : public FixedScheduleCommunication {
public:
  ProductionFixedScheduleCommunication(
      taskflow::TaskCommunicationModel &model,
      std::vector<Operation *> taskOperations);

  bool getPredecessors(unsigned consumer,
                       std::vector<unsigned> &predecessors,
                       std::string &error) const override;
  void resetReservations() override { model_.resetReservations(); }
  void beginTrial() override { model_.beginTrial(); }
  void finishTrial(bool commit) override { model_.finishTrial(commit); }
  bool supportsCanonicalReservationState() const override;
  bool getCanonicalReservationState(std::string &state,
                                    std::string &error) const override;
  bool supportsReservationCheckpoint() const override;
  bool captureReservationCheckpoint(
      ReservationCheckpoint &checkpoint, std::string &error) const override;
  bool restoreReservationCheckpoint(
      const ReservationCheckpoint &checkpoint,
      std::string &error) override;
  bool getTransferLowerBound(unsigned producer, unsigned consumer,
                             int64_t &cycles, std::string &error) const override;
  bool getTransferLowerBoundForEndpoints(
      unsigned producer, unsigned consumer, int sourceRow, int sourceCol,
      int destinationRow, int destinationCol, int64_t &cycles,
      std::string &error) const override;
  bool supportsStaticEndpointLowerBoundCache() const override;
  bool getTransferReadyCycle(unsigned producer, unsigned consumer,
                             int sourceRow, int sourceCol, int destinationRow,
                             int destinationCol, int64_t producerFinish,
                             bool reserve, int64_t &readyCycle,
                             std::string &error) override;

private:
  taskflow::TaskCommunicationModel &model_;
  std::vector<Operation *> operations_;
  std::string validationError_;
};

} // namespace amoeba::neura::joint_scheduling
} // namespace mlir
#endif // AMOEBA_PRODUCTION_FIXED_SCHEDULE_COMMUNICATION_H
