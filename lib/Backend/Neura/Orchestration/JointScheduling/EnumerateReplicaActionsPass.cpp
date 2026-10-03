//===- EnumerateReplicaActionsPass.cpp ------------------------------------===//
// Conservative per-task execution-shard inventory. A successful dry run of
// the source-owned materializer is the only nonidentity legality witness.
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {
constexpr llvm::StringLiteral kSchema = "orbit-replica-actions-v1";
constexpr std::array<int64_t, 4> kFactors = {1, 2, 4, 8};

static std::string quoteOption(StringRef value) {
  std::string quoted = "\"";
  for (char character : value) {
    if (character == '\\' || character == '"')
      quoted += '\\';
    quoted += character;
  }
  quoted += '"';
  return quoted;
}

// The current C++ materializer requires equal shards in its generic path.
// A remainder shard could still be legal in the experiment, so this is an
// unsupported capability, never a semantic illegality proof.
static bool genericStaticUnevenShard(TaskflowTaskOp task, int64_t axis,
                                     int64_t factor) {
  auto kind = task->getAttrOfType<StringAttr>("amoeba.semantic.task_kind");
  if (task.getTaskName() == "Task_0" ||
      task.getTaskName() == "Task_0.k.0" ||
      (kind && kind.getValue() == "gemm_producer") ||
      task.getWillWrites().size() != 1)
    return false;
  auto output = dyn_cast<MemRefType>(task.getWillWrites().front().getType());
  return output && axis >= 0 && axis < output.getRank() &&
         !output.isDynamicDim(axis) && output.getDimSize(axis) % factor != 0;
}

static bool dryRunMaterializer(ModuleOp module, StringRef function,
                               StringRef task, int64_t axis, int64_t factor,
                               std::string &reason,
                               std::string &diagnostic) {
  OwningOpRef<ModuleOp> trial = module.clone();
  auto pass = mlir::amoeba::neura::createMaterializeJointTaskReplicasPass();
  std::string options = "function=" + quoteOption(function) +
                        " task=" + quoteOption(task) +
                        " replicas=" + std::to_string(factor) +
                        " axis=" + std::to_string(axis);
  if (failed(pass->initializeOptions(
          options, [&](const llvm::Twine &message) {
            reason = message.str();
            return failure();
          }))) {
    reason = "materializer_option_parse_failed";
    return false;
  }
  PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(std::move(pass));
  LogicalResult runResult = failure();
  LogicalResult verifyResult = failure();
  {
    ScopedDiagnosticHandler capture(module.getContext(),
                                    [&](Diagnostic &entry) {
                                      if (diagnostic.empty()) {
                                        llvm::raw_string_ostream os(diagnostic);
                                        entry.print(os);
                                      }
                                      return success();
                                    });
    runResult = manager.run(*trial);
    if (succeeded(runResult))
      verifyResult = verify(trial->getOperation());
  }
  if (failed(runResult)) {
    reason = "materializer_dry_run_failed";
    return false;
  }
  if (failed(verifyResult)) {
    reason = "materialized_module_verifier_failed";
    return false;
  }
  std::string selectionError;
  FailureOr<func::FuncOp> selected =
      selectTaskFunction(*trial, function, selectionError);
  if (failed(selected)) {
    reason = "materialized_function_missing";
    return false;
  }
  auto materializedTask = (*selected)->getAttrOfType<StringAttr>(
      "amoeba.replica.materialized_task");
  auto materializedCount = (*selected)->getAttrOfType<IntegerAttr>(
      "amoeba.replica.count");
  if (!materializedTask || materializedTask.getValue() != task ||
      !materializedCount || materializedCount.getInt() != factor) {
    reason = "materializer_success_without_replica_witness";
    return false;
  }
  reason = "materializer_dry_run_verified";
  return true;
}

class EnumerateReplicaActionsPass
    : public PassWrapper<EnumerateReplicaActionsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnumerateReplicaActionsPass)

  EnumerateReplicaActionsPass() = default;
  EnumerateReplicaActionsPass(const EnumerateReplicaActionsPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override { return "enumerate-replica-actions"; }
  StringRef getDescription() const override {
    return "Inventory every Taskflow task's {1,2,4,8} replica choices with "
           "C++ materializer dry-run proofs";
  }

  Option<std::string> outputFile{*this, "output", llvm::cl::init("")};
  Option<std::string> functionName{*this, "function", llvm::cl::init("")};
  Option<int64_t> maxActionRecords{*this, "max-action-records",
                                   llvm::cl::init(100000)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (outputFile.empty() || maxActionRecords <= 0) {
      module.emitError("replica inventory requires output and positive "
                       "max-action-records");
      return signalPassFailure();
    }
    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName, error);
    if (failed(selected)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    SmallVector<TaskflowTaskOp> tasks;
    llvm::StringSet<> names;
    selected->walk([&](TaskflowTaskOp task) {
      if (!names.insert(task.getTaskName()).second)
        error = "duplicate Taskflow task name: " + task.getTaskName().str();
      tasks.push_back(task);
    });
    if (!error.empty() || tasks.empty()) {
      selected->emitError() << (error.empty() ? "no Taskflow tasks" : error);
      return signalPassFailure();
    }
    struct TaskAxes {
      TaskflowTaskOp task;
      int64_t axisCount;
    };
    SmallVector<TaskAxes> axes;
    int64_t declared = 0;
    for (TaskflowTaskOp task : tasks) {
      int64_t rank = 0;
      for (Value output : task.getWillWrites())
        if (auto type = dyn_cast<MemRefType>(output.getType()))
          rank = std::max<int64_t>(rank, type.getRank());
      // An unrecognized output still gets a recorded unknown probe on axis 0.
      int64_t axisCount = std::max<int64_t>(1, rank);
      if (axisCount > 64 || declared > INT64_MAX - 1 - 3 * axisCount) {
        selected->emitError("replica action inventory size overflow");
        return signalPassFailure();
      }
      axes.push_back({task, axisCount});
      declared += 1 + 3 * axisCount;
    }
    const bool bounded = declared >= maxActionRecords;
    const int64_t limit = std::min<int64_t>(declared, maxActionRecords);
    int64_t emitted = 0, legal = 0, illegal = 0, unknown = 0;
    const std::string function = selected->getSymName().str();
    auto emit = [&](llvm::raw_ostream &os, TaskflowTaskOp task, int64_t axis,
                    int64_t factor, StringRef status, StringRef reason,
                    StringRef diagnostic) {
      if (emitted >= limit)
        return;
      writeJsonLine(os, llvm::json::Object{
          {"schema", kSchema.str()}, {"record_type", "action"},
          {"kind", "replica"}, {"task", task.getTaskName().str()},
          {"axis", axis}, {"factor", factor},
          {"status", status.str()}, {"reason", reason.str()},
          {"diagnostic", diagnostic.str()}});
      ++emitted;
      if (status == "legal") ++legal;
      else if (status == "illegal") ++illegal;
      else ++unknown;
    };
    if (!writeAtomically(outputFile,
          [&](llvm::raw_ostream &os) {
            writeJsonLine(os, llvm::json::Object{
                {"schema", kSchema.str()}, {"record_type", "header"},
                {"function", function},
                {"task_count", static_cast<int64_t>(tasks.size())},
                {"factor_alphabet", llvm::json::Array{1, 2, 4, 8}},
                {"declared_action_count", declared},
                {"status", "incomplete"}, {"complete", false},
                {"scope", "per_task_actions_only"}});
            for (TaskAxes entry : axes) {
              emit(os, entry.task, -1, 1, "legal", "identity_no_op", "");
              for (int64_t axis = 0; axis < entry.axisCount; ++axis) {
                for (int64_t factor : kFactors) {
                  if (factor == 1)
                    continue;
                  if (emitted >= limit)
                    break;
                  std::string proof, diagnostic;
                  bool proved = dryRunMaterializer(module, function,
                      entry.task.getTaskName(), axis, factor, proof, diagnostic);
                  if (!proved &&
                      genericStaticUnevenShard(entry.task, axis, factor))
                    proof = "needs_uneven_shard_materializer";
                  emit(os, entry.task, axis, factor,
                       proved ? "legal" : "unknown", proof, diagnostic);
                }
              }
            }
            writeJsonLine(os, llvm::json::Object{
                {"schema", kSchema.str()}, {"record_type", "footer"},
                {"status", "incomplete"}, {"complete", false},
                {"declared_action_count", declared},
                {"emitted_action_count", emitted},
                {"omitted_action_count", declared - emitted},
                {"legal_action_count", legal},
                {"illegal_action_count", illegal},
                {"unknown_action_count", unknown},
                {"resource_bound_reached", bounded},
                {"all_task_factor_pairs_recorded", !bounded},
                {"first_blocker", bounded ? "max_action_records_reached" :
                    "joint_graph_replica_closure_not_enumerated"}});
            return true;
          }, error)) {
      selected->emitError() << error;
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createEnumerateReplicaActionsPass() {
  return std::make_unique<EnumerateReplicaActionsPass>();
}
} // namespace mlir::amoeba::neura
