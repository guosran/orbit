//===- EnumerateExactTaskSchedulesPass.cpp - Fixed-shape 4x4 inventory -----===//

#include "AnalyticalTaskCandidateCommon.h"
#include "CommunicationExactScheduleSpace.h"
#include "ProductionFixedScheduleCommunication.h"
#include "ExactScheduleSpace.h"

#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.h"

#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {
struct EnumerateExactTaskSchedulesPass
    : PassWrapper<EnumerateExactTaskSchedulesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnumerateExactTaskSchedulesPass)
  EnumerateExactTaskSchedulesPass() = default;
  EnumerateExactTaskSchedulesPass(const EnumerateExactTaskSchedulesPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "enumerate-exact-task-schedules";
  }
  StringRef getDescription() const override {
    return "Enumerate event-normalized legal 4x4 placements for fixed shapes";
  }

  Option<std::string> functionName{
      *this, "function", llvm::cl::desc("Unique Taskflow function."),
      llvm::cl::init("")};
  Option<std::string> outputFile{
      *this, "output", llvm::cl::desc("Schedule inventory JSONL path."),
      llvm::cl::init("")};
  Option<int64_t> maxExpandedNodes{
      *this, "max-expanded-nodes",
      llvm::cl::desc("Branch budget; exhaustion reports incomplete."),
      llvm::cl::init(100000)};
  Option<int64_t> maxMilliseconds{
      *this, "max-milliseconds",
      llvm::cl::desc("Optional wall time budget; exhaustion reports incomplete."),
      llvm::cl::init(0)};

  Option<std::string> communicationMode{
      *this, "communication-mode",
      llvm::cl::desc("none or explicit production inter-task network."),
      llvm::cl::init("none")};
  Option<std::string> startPolicy{
      *this, "start-policy",
      llvm::cl::desc("event-normalized or all-integer."),
      llvm::cl::init("event-normalized")};
  Option<int64_t> maxMakespan{
      *this, "max-makespan",
      llvm::cl::desc("Inclusive whole-program cycle horizon for all-integer."),
      llvm::cl::init(0)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    if (outputFile.getValue().empty() || maxExpandedNodes.getValue() < 1 ||
        maxMilliseconds.getValue() < 0 ||
        (communicationMode.getValue() != "none" &&
         communicationMode.getValue() != "explicit") ||
        (startPolicy.getValue() != "event-normalized" &&
         startPolicy.getValue() != "all-integer") ||
        maxMakespan.getValue() < 0 ||
        (startPolicy.getValue() == "all-integer" &&
         (maxMakespan.getValue() == 0 ||
          communicationMode.getValue() != "explicit")) ||
        (startPolicy.getValue() == "event-normalized" &&
         maxMakespan.getValue() != 0)) {
      module.emitError() << "output, positive budget, communication mode and compatible start policy/horizon required";
      return signalPassFailure();
    }
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selected)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp function = *selected;
    FailureOr<TaskEdgeGraph> graph = buildTaskEdgeGraph(
        function, TaskEdgeGraphOptions{/*require_payload=*/false,
                                       /*read_control_predecessors=*/true},
        error);
    if (failed(graph)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    std::vector<ExactScheduleTask> tasks;
    std::vector<Operation *> taskOperations;
    llvm::DenseMap<Operation *, unsigned> indexes;
    std::set<std::string> names;
    for (taskflow::TaskflowTaskOp task : graph->getTasks()) {
      ExactScheduleTask item;
      item.name = task.getTaskName().str();
      if (!names.insert(item.name).second) {
        function.emitError() << "task names must be unique";
        return signalPassFailure();
      }
      auto shape = task->getAttrOfType<StringAttr>("cgra_shape");
      auto duration = task->getAttrOfType<IntegerAttr>("est_latency");
      if (!shape || !duration || duration.getInt() < 1 ||
          !task->hasAttr("amoeba.joint_shape_orientation_fixed")) {
        task.emitError() << "exact schedule inventory requires fixed cgra_shape "
                            "and positive predicted est_latency";
        return signalPassFailure();
      }
      StringRef text = shape.getValue();
      auto parts = text.split('x');
      if (parts.first.empty() || parts.second.empty() ||
          parts.first.getAsInteger(10, item.rows) ||
          parts.second.getAsInteger(10, item.cols) ||
          item.rows < 1 || item.cols < 1 || item.rows > 4 || item.cols > 4 ||
          item.rows * item.cols > 4) {
        task.emitError() << "invalid fixed rectangle or more than four CGRAs";
        return signalPassFailure();
      }
      if (auto count = task->getAttrOfType<IntegerAttr>("cgra_count"))
        if (count.getInt() != item.rows * item.cols) {
          task.emitError() << "cgra_count disagrees with fixed rectangle";
          return signalPassFailure();
        }
      item.duration = duration.getInt();
      indexes[task.getOperation()] = tasks.size();
      taskOperations.push_back(task.getOperation());
      tasks.push_back(std::move(item));
    }
    for (TaskEdge edge : graph->getEdges()) {
      auto producerIt = indexes.find(edge.producer.getOperation());
      auto consumerIt = indexes.find(edge.consumer.getOperation());
      if (producerIt == indexes.end() || consumerIt == indexes.end()) {
        function.emitError() << "typed task edge endpoint is absent from graph";
        return signalPassFailure();
      }
      unsigned producer = producerIt->second;
      unsigned consumer = consumerIt->second;
      auto &predecessors = tasks[consumer].predecessors;
      if (std::find(predecessors.begin(), predecessors.end(), producer) ==
          predecessors.end())
        predecessors.push_back(producer);
    }
    llvm::json::Array scheduleRecords;
    ExactScheduleResult result;
    ExactScheduleLimits limits{
        static_cast<uint64_t>(maxExpandedNodes.getValue()),
        static_cast<uint64_t>(maxMilliseconds.getValue())};
    auto appendRecord =
        [&](const std::vector<ExactSchedulePlacement> &schedule,
            const std::vector<unsigned> &dispatchOrder,
            const std::optional<FixedScheduleScore> &score) {
          llvm::json::Array entries;
          llvm::json::Array order;
          for (unsigned taskIndex : dispatchOrder)
            order.push_back(tasks[taskIndex].name);
          for (const auto &placement : schedule) {
            llvm::json::Object entry;
            entry["task"] = tasks[placement.task].name;
            entry["row"] = placement.row;
            entry["col"] = placement.col;
            entry["rows"] = tasks[placement.task].rows;
            entry["cols"] = tasks[placement.task].cols;
            entry["start_cycle"] = placement.start;
            entry["end_cycle"] = placement.end;
            entries.push_back(std::move(entry));
          }
          llvm::json::Object record;
          record["record_type"] = "schedule";
          record["schedule_id"] =
              "schedule-" + std::to_string(scheduleRecords.size());
          record["task_schedule"] = std::move(entries);
          record["dispatch_order"] = std::move(order);
          if (score) {
            record["predicted_whole_program_cycles"] =
                score->predictedMakespan;
            record["replayed_communication_edges"] =
                static_cast<int64_t>(score->replayedEdges);
            record["score_source"] =
                "fixed-est-latency-plus-production-explicit-network";
            record["task_latency_provenance"] =
                "caller-supplied-unverified";
          }
          scheduleRecords.push_back(std::move(record));
        };
    bool accepted = false;
    CommunicationScheduleCertificate certificate;
    if (communicationMode.getValue() == "explicit") {
      auto network = createInterTaskNetworkCommunicationModel(
          function, 4, 4, error);
      if (failed(network)) {
        function.emitError() << error;
        return signalPassFailure();
      }
      ProductionFixedScheduleCommunication bridge(**network, taskOperations);
      accepted = enumerateCommunicationScheduleSpace(
          4, 4, tasks, limits,
          startPolicy.getValue() == "all-integer"
              ? CommunicationStartPolicy::AllInteger
              : CommunicationStartPolicy::EventNormalized,
          maxMakespan.getValue(), bridge,
          [&](const std::vector<ExactSchedulePlacement> &schedule,
              const std::vector<unsigned> &dispatchOrder,
              const FixedScheduleScore &score) {
            appendRecord(schedule, dispatchOrder, score);
          },
          result, certificate, error);
    } else {
      accepted = enumerateExactScheduleSpace(
          4, 4, tasks, limits,
          [&](const std::vector<ExactSchedulePlacement> &schedule,
              const std::vector<unsigned> &dispatchOrder) {
            appendRecord(schedule, dispatchOrder, std::nullopt);
          },
          result, error);
    }
    if (!accepted) {
      function.emitError() << error;
      return signalPassFailure();
    }
    if (!writeAtomically(outputFile.getValue(),
                         [&](llvm::raw_ostream &out) {
                           llvm::json::Object header;
                           header["record_type"] = "header";
                           header["schema"] = "amoeba-exact-task-schedules-v1";
                           header["function"] = function.getSymName().str();
                           header["grid_rows"] = 4;
                           header["grid_cols"] = 4;
                           header["max_cgras_per_task"] = 4;
                           header["search_scope"] =
                               startPolicy.getValue() == "all-integer"
                                   ? "fixed-shape-graph-bounded-all-integer"
                                   : "fixed-shapes-event-normalized";
                           header["start_policy"] = startPolicy.getValue();
                           header["max_makespan"] = maxMakespan.getValue();
                           header["communication_mode"] =
                               communicationMode.getValue();
                           header["ranked"] =
                               communicationMode.getValue() == "explicit";
                           header["communication_ready_event_enumerated"] =
                               communicationMode.getValue() == "explicit";
                           header["bounded_integer_start_coverage"] =
                               startPolicy.getValue() == "all-integer";
                           header["communication_start_events_proven"] = false;
                           header["full_legal_schedule_space_certified"] = false;
                           header["dispatch_order_recorded"] = true;
                           writeJsonLine(out, std::move(header));
                           for (auto &value : scheduleRecords)
                             writeJsonLine(out, std::move(*value.getAsObject()));
                           llvm::json::Object footer;
                           footer["record_type"] = "footer";
                           footer["status"] = !result.complete
                               ? "incomplete"
                               : startPolicy.getValue() == "all-integer"
                                   ? "bounded_makespan_complete"
                                   : communicationMode.getValue() == "explicit"
                                       ? "event_normalized_complete"
                                       : "complete";
                           footer["reason"] = result.reason;
                           footer["expanded_nodes"] =
                               static_cast<int64_t>(result.expandedNodes);
                           footer["accepted_paths"] =
                               static_cast<int64_t>(result.acceptedPaths);
                           footer["unique_schedules"] =
                               static_cast<int64_t>(result.uniqueSchedules);
                           footer["overlap_rejected_branches"] =
                               static_cast<int64_t>(
                                   result.overlapRejectedBranches);
                           footer["out_of_bounds_rejected_branches"] =
                               static_cast<int64_t>(
                                   result.outOfBoundsRejectedBranches);
                           footer["top_k_certified"] = false;
                           footer["top_five_certified_within_fixed_shape_graph"] =
                               certificate.topFiveCertifiedWithinFixedShapeGraph;
                           footer["distinct_schedule_candidates"] =
                               static_cast<int64_t>(certificate.distinctCandidates);
                           if (certificate.topFiveCertifiedWithinFixedShapeGraph)
                             footer["fifth_predicted_cycles"] =
                                 certificate.fifthPredictedCycles;
                           writeJsonLine(out, std::move(footer));
                           return true;
                         },
                         error)) {
      function.emitError() << error;
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::amoeba::neura::createEnumerateExactTaskSchedulesPass() {
  return std::make_unique<EnumerateExactTaskSchedulesPass>();
}
