//===- CommunicationExactScheduleSpace.h - Explicit-network schedules ----===//
#ifndef AMOEBA_COMMUNICATION_EXACT_SCHEDULE_SPACE_H
#define AMOEBA_COMMUNICATION_EXACT_SCHEDULE_SPACE_H

#include "ExactScheduleSpace.h"
#include "FixedScheduleVerifier.h"

namespace mlir::amoeba::neura::joint_scheduling {

enum class CommunicationStartPolicy { EventNormalized, AllInteger, SeedOnly, OrderDerived };
enum class CommunicationDispatchPolicy { AllReady, CanonicalSmallestReady };

struct CommunicationScheduleCertificate {
  // Applies only to one fixed graph and one fixed task-shape assignment.
  // AllInteger with H enumerates every legal integer start through H.
  bool topFiveCertifiedWithinFixedShapeGraph = false;
  int64_t fifthPredictedCycles = 0;
  uint64_t distinctCandidates = 0;
};

// EventNormalized searches communication-ready and resource-release events,
// but does not certify unrestricted temporal placement. AllInteger searches
// every integer start through maxMakespan, with a finite and exact bound.
bool enumerateCommunicationScheduleSpace(
    int gridRows, int gridCols, const std::vector<ExactScheduleTask> &tasks,
    ExactScheduleLimits limits, CommunicationStartPolicy startPolicy,
    int64_t maxMakespan, FixedScheduleCommunication &communication,
    const std::function<void(const std::vector<ExactSchedulePlacement> &,
                             const std::vector<unsigned> &,
                             const FixedScheduleScore &)> &emit,
    ExactScheduleResult &result,
    CommunicationScheduleCertificate &certificate, std::string &error,
    CommunicationDispatchPolicy dispatchPolicy =
        CommunicationDispatchPolicy::AllReady,
    const std::function<int64_t()> &externalUpperBound = {});

} // namespace mlir::amoeba::neura::joint_scheduling
#endif // AMOEBA_COMMUNICATION_EXACT_SCHEDULE_SPACE_H
