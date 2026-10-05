#!/usr/bin/env python3
"""Compile and run focused fixed-decision replica geometry/replay checks."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


HARNESS = r'''#include "OriginalAmoebaFixedDecisionRetimer.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir::amoeba::neura::joint_scheduling;

namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct EndpointQuery {
  int sourceRow;
  int sourceCol;
  int destinationRow;
  int destinationCol;
  bool reserve;
};

class RecordingCommunication final : public FixedScheduleCommunication {
public:
  explicit RecordingCommunication(std::vector<std::vector<unsigned>> edges,
                                  int64_t transferLatency = 0)
      : predecessors(std::move(edges)), latency(transferLatency) {}

  bool getPredecessors(unsigned consumer,
                       std::vector<unsigned> &result,
                       std::string &error) const override {
    if (consumer >= predecessors.size()) {
      error = "unknown consumer";
      return false;
    }
    result = predecessors[consumer];
    return true;
  }

  void resetReservations() override { queries.clear(); }
  void beginTrial() override {}
  void finishTrial(bool) override {}

  bool getTransferReadyCycle(unsigned, unsigned, int sourceRow,
                             int sourceCol, int destinationRow,
                             int destinationCol, int64_t producerFinish,
                             bool reserve, int64_t &readyCycle,
                             std::string &) override {
    queries.push_back({sourceRow, sourceCol, destinationRow, destinationCol,
                       reserve});
    readyCycle = producerFinish + latency;
    return true;
  }

  std::vector<EndpointQuery> queries;

private:
  std::vector<std::vector<unsigned>> predecessors;
  int64_t latency;
};

OriginalAmoebaFixedPlacement replicatedPlacement(unsigned task = 0) {
  OriginalAmoebaFixedPlacement placement;
  placement.task = task;
  placement.row = 0;
  placement.col = 0;
  placement.rows = 2;
  placement.cols = 1;
  placement.selectedRows = 2;
  placement.selectedCols = 1;
  placement.activeReplicas = 2;
  // Deliberately not row-major: the returned trace must retain this order,
  // while transfer replay must visit captured endpoints in row-major order.
  placement.occupiedCells = {{1, 0, 0}, {0, 0, 0}, {0, 3, 1}, {0, 2, 1}};
  return placement;
}

OriginalAmoebaFixedPlacement singletonPlacement(unsigned task, int row,
                                                int col) {
  OriginalAmoebaFixedPlacement placement;
  placement.task = task;
  placement.row = row;
  placement.col = col;
  placement.rows = 1;
  placement.cols = 1;
  placement.selectedRows = 1;
  placement.selectedCols = 1;
  placement.activeReplicas = 1;
  placement.occupiedCells = {{row, col, 0}};
  return placement;
}

bool runRetimer(const std::vector<OriginalAmoebaRetimerTask> &tasks,
                const std::vector<int64_t> &durations,
                const std::vector<OriginalAmoebaFixedPlacement> &placements,
                const std::vector<unsigned> &order,
                RecordingCommunication &communication,
                OriginalAmoebaFixedDecisionResult &result,
                bool allowScaling) {
  return retimeOriginalAmoebaFixedDecisions(
      4, 4, tasks, durations, placements, order, communication, result,
      allowScaling);
}

void expectRejected(const OriginalAmoebaFixedPlacement &placement,
                    const char *expectedReason) {
  std::vector<OriginalAmoebaRetimerTask> tasks = {{"task", {}}};
  RecordingCommunication communication(
      std::vector<std::vector<unsigned>>{{}});
  OriginalAmoebaFixedDecisionResult result;
  const bool accepted = runRetimer(tasks, {11}, {placement}, {0},
                                   communication, result, true);
  require(!accepted, "malformed replica geometry unexpectedly passed");
  require(result.rejection.find(expectedReason) != std::string::npos,
          "replica geometry rejected for an unexpected reason");
}

void expectVerifierRejected(const FixedScheduleOccupiedCellInventory &cells,
                            const char *expectedReason) {
  std::vector<ExactScheduleTask> tasks = {{"task", 2, 1, 6, {}}};
  std::vector<ExactSchedulePlacement> placements = {{0, 0, 0, 0, 6, 0}};
  RecordingCommunication communication(
      std::vector<std::vector<unsigned>>{{}});
  const FixedScheduleScore score = verifyAndScoreFixedSchedule(
      4, 4, tasks, placements, {0}, communication, &cells);
  require(!score.valid, "malformed full-cell verifier inventory passed");
  require(score.rejection.find(expectedReason) != std::string::npos,
          "full-cell inventory rejected for an unexpected reason");
}

void testMultiReplicaOccupancyAndRotation() {
  std::vector<OriginalAmoebaRetimerTask> tasks = {
      {"replicated", {}}, {"uses-secondary-cell", {}}};
  auto primary = replicatedPlacement();
  auto secondary = singletonPlacement(1, 0, 2);
  const auto originalCellSequence = primary.occupiedCells;
  RecordingCommunication communication(
      std::vector<std::vector<unsigned>>{{}, {}});
  OriginalAmoebaFixedDecisionResult result;
  require(runRetimer(tasks, {11, 3}, {primary, secondary}, {0, 1},
                     communication, result, true),
          "valid rotated multi-replica geometry was rejected");
  require(result.tasks[0].duration == 6,
          "multi-replica duration did not use ceil(parent duration / count)");
  require(result.tasks[1].duration == 3,
          "singleton duration changed under replica estimate mode");
  require(result.placements[0].end == 6 && result.placements[1].start >= 6,
          "secondary-replica cell was not reserved for the complete task");
  require(result.usesOriginalF45ReplicaScalingEstimate,
          "explicit estimate mode was not recorded in the result");
  require(result.preservedDecisions[0].occupiedCells.size() ==
              originalCellSequence.size() &&
              std::equal(result.preservedDecisions[0].occupiedCells.begin(),
                         result.preservedDecisions[0].occupiedCells.end(),
                         originalCellSequence.begin(),
                         [](const auto &a, const auto &b) {
                           return a.row == b.row && a.col == b.col &&
                                  a.replicaId == b.replicaId;
                         }),
          "source cell inventory order was changed in preserved decisions");

  std::vector<ExactScheduleTask> exactTasks = {
      {"replicated", 2, 1, 6, {}}, {"uses-secondary-cell", 1, 1, 3, {}}};
  std::vector<ExactSchedulePlacement> overlapping = {
      {0, 0, 0, 0, 6, 0}, {1, 0, 2, 0, 3, 0}};
  FixedScheduleOccupiedCellInventory inventory = {
      {{0, 0}, {0, 2}, {0, 3}, {1, 0}}, {{0, 2}}};
  RecordingCommunication verifierCommunication(
      std::vector<std::vector<unsigned>>{{}, {}});
  const FixedScheduleScore captured = verifyAndScoreFixedSchedule(
      4, 4, exactTasks, overlapping, {0, 1}, verifierCommunication,
      &inventory);
  require(!captured.valid &&
              captured.rejection.find("captured CGRA cells") !=
                  std::string::npos,
          "full-cell verifier accepted an overlap on replica 1");

  const FixedScheduleScore rectangleOnly = verifyAndScoreFixedSchedule(
      4, 4, exactTasks, overlapping, {0, 1}, verifierCommunication);
  require(rectangleOnly.valid,
          "default rectangle verifier behavior changed without an inventory");

  expectVerifierRejected({{{0, 0}, {0, 0}, {0, 2}}}, "duplicate CGRA cell");
  expectVerifierRejected({{{0, 0}, {1, 0}, {4, 0}}},
                         "out-of-bounds");
  expectVerifierRejected({{{1, 0}, {0, 2}}}, "omits a primary-rectangle");
}

void testGeometryTamperingAndDefaultGate() {
  auto placement = replicatedPlacement();
  placement.occupiedCells[2].replicaId = 0;
  placement.occupiedCells[3].replicaId = 0;
  expectRejected(placement, "omits cells for a replica ID");

  placement = replicatedPlacement();
  placement.occupiedCells[1].row = placement.occupiedCells[0].row;
  placement.occupiedCells[1].col = placement.occupiedCells[0].col;
  expectRejected(placement, "duplicates or overlaps");

  placement = replicatedPlacement();
  placement.occupiedCells[2].replicaId = 2;
  expectRejected(placement, "out-of-range replica ID");

  placement = replicatedPlacement();
  placement.occupiedCells[2].row = 4;
  expectRejected(placement, "outside the grid");

  placement = replicatedPlacement();
  placement.occupiedCells[2].row = 0;
  placement.occupiedCells[2].col = 0;
  expectRejected(placement, "duplicates or overlaps");

  placement = replicatedPlacement();
  placement.occupiedCells[2] = {0, 2, 1};
  placement.occupiedCells[3] = {1, 3, 1};
  expectRejected(placement, "do not form the selected shape");

  std::vector<OriginalAmoebaRetimerTask> tasks = {{"task", {}}};
  RecordingCommunication communication(
      std::vector<std::vector<unsigned>>{{}});
  OriginalAmoebaFixedDecisionResult result;
  require(!retimeOriginalAmoebaFixedDecisions(
              4, 4, tasks, {11}, {replicatedPlacement()}, {0},
              communication, result),
          "default mode accepted a multi-replica inventory");
  require(result.rejection.find("explicit original F45") !=
              std::string::npos,
          "default multi-replica gate lacks its explicit-mode diagnostic");

  RecordingCommunication singletonCommunication(
      std::vector<std::vector<unsigned>>{{}});
  OriginalAmoebaFixedDecisionResult singletonResult;
  require(retimeOriginalAmoebaFixedDecisions(
              4, 4, tasks, {11}, {singletonPlacement(0, 2, 2)}, {0},
              singletonCommunication, singletonResult),
          "default singleton path was rejected");
  require(singletonResult.tasks[0].duration == 11 &&
              !singletonResult.usesOriginalF45ReplicaScalingEstimate,
          "default singleton duration or estimate marker changed");
}

void testCommunicationVisitsAllReplicaCellsInOrder() {
  std::vector<OriginalAmoebaRetimerTask> tasks = {
      {"producer", {}}, {"consumer", {0}}};
  auto producer = replicatedPlacement();
  auto consumer = singletonPlacement(1, 3, 3);
  RecordingCommunication communication(
      std::vector<std::vector<unsigned>>{{}, {0}}, 3);
  OriginalAmoebaFixedDecisionResult result;
  require(runRetimer(tasks, {11, 2}, {producer, consumer}, {0, 1},
                     communication, result, true),
          "valid communicated replica geometry was rejected");
  require(result.placements[0].end == 6 && result.placements[1].start == 9,
          "network replay did not use the F45 estimate and edge delay");
  require(communication.queries.size() == 5,
          "final communication replay did not query all captured endpoints");
  const std::vector<std::pair<int, int>> expectedSourceCells = {
      {0, 0}, {0, 2}, {0, 3}, {1, 0}};
  for (size_t index = 0; index < expectedSourceCells.size(); ++index) {
    const auto &query = communication.queries[index];
    require(query.sourceRow == expectedSourceCells[index].first &&
                query.sourceCol == expectedSourceCells[index].second &&
                query.destinationRow == 3 && query.destinationCol == 3 &&
                !query.reserve,
            "network replay endpoints are incomplete or not row-major");
  }
  require(communication.queries.back().reserve,
          "network replay did not reserve its selected endpoint");
}

} // namespace

int main() {
  try {
    testMultiReplicaOccupancyAndRotation();
    testGeometryTamperingAndDefaultGate();
    testCommunicationVisitsAllReplicaCellsInOrder();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "original F45 replica scaling checks passed\n";
  return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--include-dir", action="append", type=Path, default=[])
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()

    joint_dir = (
        args.source_root
        / "lib/Backend/Neura/Orchestration/JointScheduling"
    )
    sources = [
        joint_dir / "OriginalAmoebaFixedDecisionRetimer.cpp",
        joint_dir / "FixedScheduleVerifier.cpp",
    ]
    for source in sources:
        if not source.is_file():
            parser.error(f"missing source file: {source}")

    with tempfile.TemporaryDirectory(prefix="original-f45-replica-check-") as temp:
        temp_path = Path(temp)
        harness = temp_path / "replica_checks.cpp"
        binary = temp_path / "replica_checks"
        harness.write_text(HARNESS)
        command = [
            *shlex.split(args.cxx),
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(joint_dir),
            *[arg for path in args.include_dir for arg in ("-I", str(path))],
            str(harness),
            *(str(source) for source in sources),
            "-o",
            str(binary),
        ]
        compiled = subprocess.run(command, text=True, capture_output=True)
        if compiled.returncode:
            sys.stderr.write(compiled.stdout)
            sys.stderr.write(compiled.stderr)
            return compiled.returncode
        checked = subprocess.run([str(binary)], text=True, capture_output=True)
        sys.stdout.write(checked.stdout)
        sys.stderr.write(checked.stderr)
        return checked.returncode


if __name__ == "__main__":
    raise SystemExit(main())
