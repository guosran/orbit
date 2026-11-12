#ifndef ORBIT_TASKFLOW_NPU_CONSTRAINT_SCHEDULE_H
#define ORBIT_TASKFLOW_NPU_CONSTRAINT_SCHEDULE_H
#include "Taskflow/ConstraintSearch.h"
namespace orbit::taskflow {
struct NPUDmaResource {
  int64_t id = 0, maxStaticBds = 0;
};
struct NPUPhysicalTile {
  int64_t id = 0, column = 0, row = 0, staticBytesAvailable = 0;
  int64_t inputDmaResource = -1, outputDmaResource = -1, inputChannel = 0,
          outputChannel = 1;
};
struct NPUIndependentShard {
  int64_t id = 0, inputNs = 0, computeNs = 0, outputNs = 0, staticBytes = 0,
          inputBdCount = 0, outputBdCount = 0;
  std::vector<int64_t> candidateTileIndices;
};
struct NPUScheduleProblem {
  std::vector<NPUDmaResource> dmaResources;
  std::vector<NPUPhysicalTile> tiles;
  std::vector<NPUIndependentShard> shards;
  // The native first implementation places one independently lowered herd on
  // each selected tile. Generic offline schedules may explicitly allow sharing.
  bool exclusiveTiles = true;
};
struct NPUShardSchedule {
  int64_t shardIndex = -1, tileIndex = -1, column = -1, row = -1,
          inputChannel = -1, outputChannel = -1;
  int64_t inputStartNs = 0, inputEndNs = 0, computeStartNs = 0,
          computeEndNs = 0, outputStartNs = 0, outputEndNs = 0;
};
struct NPUScheduleResult {
  ConstraintSearchStatus status = ConstraintSearchStatus::Unknown;
  std::string message;
  std::vector<NPUShardSchedule> shards;
  int64_t makespanNs = 0;
  ConstraintSearchTelemetry telemetry;
};
NPUScheduleResult
solveNPUConstraintSchedule(const NPUScheduleProblem &,
                           const ConstraintSearchOptions & = {});
} // namespace orbit::taskflow
#endif
