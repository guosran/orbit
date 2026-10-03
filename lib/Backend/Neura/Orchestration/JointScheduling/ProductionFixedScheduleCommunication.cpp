//===- ProductionFixedScheduleCommunication.cpp - Production network bridge ===//
#include "ProductionFixedScheduleCommunication.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <memory>
#include <set>
#include <utility>

namespace mlir::amoeba::neura::joint_scheduling {
namespace {
constexpr uint64_t kProductionReservationCheckpointTypeTag =
    0x2e5a8bb4c1d90763ULL;

struct ProductionReservationCheckpointState {
  taskflow::TaskCommunicationReservationCheckpoint model;
};
} // namespace

ProductionFixedScheduleCommunication::ProductionFixedScheduleCommunication(
    taskflow::TaskCommunicationModel &model,
    std::vector<Operation *> taskOperations)
    : model_(model), operations_(std::move(taskOperations)) {
  std::set<Operation *> seen;
  for (Operation *operation : operations_)
    if (!operation || !seen.insert(operation).second)
      validationError_ = "fixed schedule Taskflow operations must be unique and nonnull";
}

bool ProductionFixedScheduleCommunication::supportsCanonicalReservationState()
    const {
  return validationError_.empty() && model_.supportsCanonicalReservationState();
}

bool ProductionFixedScheduleCommunication::getCanonicalReservationState(
    std::string &state, std::string &error) const {
  state.clear();
  if (!validationError_.empty()) {
    error = validationError_;
    return false;
  }
  return model_.getCanonicalReservationState(state, error);
}

bool ProductionFixedScheduleCommunication::supportsReservationCheckpoint()
    const {
  return validationError_.empty() && model_.supportsReservationCheckpoint();
}

bool ProductionFixedScheduleCommunication::captureReservationCheckpoint(
    ReservationCheckpoint &checkpoint, std::string &error) const {
  checkpoint = {};
  if (!validationError_.empty()) {
    error = validationError_;
    return false;
  }
  if (!model_.supportsReservationCheckpoint()) {
    error = "fixed schedule communication has no reservation checkpoint";
    return false;
  }
  auto state = std::make_shared<ProductionReservationCheckpointState>();
  if (!model_.captureReservationCheckpoint(state->model, error))
    return false;
  if (state->model.owner != &model_ || state->model.typeTag == 0 ||
      !state->model.state) {
    error = "fixed schedule communication returned an invalid reservation checkpoint";
    return false;
  }
  checkpoint.owner = this;
  checkpoint.typeTag = kProductionReservationCheckpointTypeTag;
  checkpoint.state = std::move(state);
  return true;
}

bool ProductionFixedScheduleCommunication::restoreReservationCheckpoint(
    const ReservationCheckpoint &checkpoint, std::string &error) {
  error.clear();
  if (!validationError_.empty()) {
    error = validationError_;
    return false;
  }
  if (checkpoint.owner != this) {
    error = "fixed schedule reservation checkpoint belongs to another model";
    return false;
  }
  if (checkpoint.typeTag != kProductionReservationCheckpointTypeTag) {
    error = "fixed schedule reservation checkpoint has an unknown type";
    return false;
  }
  if (!checkpoint.state) {
    error = "fixed schedule reservation checkpoint has no state";
    return false;
  }
  const auto state = std::static_pointer_cast<
      const ProductionReservationCheckpointState>(checkpoint.state);
  if (!state) {
    error = "fixed schedule reservation checkpoint has invalid state";
    return false;
  }
  return model_.restoreReservationCheckpoint(state->model, error);
}

bool ProductionFixedScheduleCommunication::getPredecessors(
    unsigned consumer, std::vector<unsigned> &predecessors,
    std::string &error) const {
  predecessors.clear();
  if (!validationError_.empty()) {
    error = validationError_;
    return false;
  }
  if (consumer >= operations_.size() || !operations_[consumer]) {
    error = "fixed schedule consumer has no Taskflow operation";
    return false;
  }
  llvm::SmallVector<Operation *> operations;
  if (!model_.getTaskPredecessors(operations_[consumer], operations, error))
    return false;
  for (Operation *operation : operations) {
    const auto it = std::find(operations_.begin(), operations_.end(), operation);
    if (it == operations_.end() || !operation) {
      error = "communication predecessor is absent from fixed schedule";
      return false;
    }
    predecessors.push_back(static_cast<unsigned>(it - operations_.begin()));
  }
  return true;
}

bool ProductionFixedScheduleCommunication::getTransferLowerBound(
    unsigned producer, unsigned consumer, int64_t &cycles, std::string &error) const {
  cycles = 0;
  if (producer >= operations_.size() || consumer >= operations_.size()) {
    error = "lower bound has an unknown task"; return false;
  }
  return model_.getTransferLowerBound(operations_[producer], operations_[consumer], cycles, error);
}

bool ProductionFixedScheduleCommunication::getTransferLowerBoundForEndpoints(
    unsigned producer, unsigned consumer, int sourceRow, int sourceCol,
    int destinationRow, int destinationCol, int64_t &cycles,
    std::string &error) const {
  cycles = 0;
  if (producer >= operations_.size() || consumer >= operations_.size()) {
    error = "endpoint lower bound has an unknown task";
    return false;
  }
  return model_.getTransferLowerBoundForEndpoints(
      operations_[producer], operations_[consumer], sourceRow, sourceCol,
      destinationRow, destinationCol, cycles, error);
}

bool ProductionFixedScheduleCommunication::supportsStaticEndpointLowerBoundCache()
    const {
  // The provider itself must certify that its endpoint lower bound ignores
  // committed/trial reservations. Legacy/custom models fail closed.
  return validationError_.empty() &&
         model_.supportsStaticEndpointLowerBoundCache();
}

bool ProductionFixedScheduleCommunication::getTransferReadyCycle(
    unsigned producer, unsigned consumer, int sourceRow, int sourceCol,
    int destinationRow, int destinationCol, int64_t producerFinish,
    bool reserve, int64_t &readyCycle, std::string &error) {
  if (!validationError_.empty()) {
    error = validationError_;
    return false;
  }
  if (producer >= operations_.size() || consumer >= operations_.size() ||
      !operations_[producer] || !operations_[consumer]) {
    error = "fixed schedule transfer has an unknown Taskflow operation";
    return false;
  }
  return model_.getTransferReadyCycle(
      operations_[producer], operations_[consumer], sourceRow, sourceCol,
      destinationRow, destinationCol, producerFinish, reserve, readyCycle,
      error);
}
} // namespace mlir::amoeba::neura::joint_scheduling
