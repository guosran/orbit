//===- EnumerateApplicationRewriteActionsPass.cpp -------------------------===//
// Source-owned, conservative action alphabet for complete Taskflow programs.
// This pass is a fail-closed inventory, not a semantic rewrite executor.
// Existing canonical 8x8 GEMM passes cannot prove general AMOEBA workloads.
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {
constexpr llvm::StringLiteral kSchema = "orbit-application-rewrite-actions-v2";
constexpr std::array<int64_t, 4> kFactors = {1, 2, 4, 8};
constexpr std::array<llvm::StringLiteral, 2> kOutputAxes = {"M", "N"};
constexpr std::array<llvm::StringLiteral, 3> kKPolicies = {
    "sequential_state_carry",
    "parallel_private_partials_linear_reduction",
    "parallel_private_partials_balanced_tree_reduction"};
constexpr std::array<llvm::StringLiteral, 2> kProducerConsumerModes = {
    "producer-consumer-forwarded", "producer-consumer-retained"};

class EnumerateApplicationRewriteActionsPass
    : public PassWrapper<EnumerateApplicationRewriteActionsPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      EnumerateApplicationRewriteActionsPass)

  EnumerateApplicationRewriteActionsPass() = default;
  EnumerateApplicationRewriteActionsPass(
      const EnumerateApplicationRewriteActionsPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "enumerate-application-rewrite-actions";
  }
  StringRef getDescription() const override {
    return "Inventory M/N/K and fusion actions without pretending unsupported "
           "mixed Taskflow/Neura rewrites are illegal";
  }

  Option<std::string> outputFile{*this, "output",
      llvm::cl::desc("action inventory JSONL output"), llvm::cl::init("")};
  Option<std::string> functionName{*this, "function",
      llvm::cl::desc("function containing the complete program"),
      llvm::cl::init("")};
  Option<int64_t> maxActionRecords{*this, "max-action-records",
      llvm::cl::desc("resource bound; a reached bound remains incomplete"),
      llvm::cl::init(100000)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (outputFile.empty() || maxActionRecords <= 0) {
      module.emitError("application rewrite inventory requires output and "
                       "positive max-action-records");
      return signalPassFailure();
    }
    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName, error);
    if (failed(selected)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    SmallVector<std::string> tasks;
    SmallVector<TaskflowTaskOp> taskOps;
    llvm::StringSet<> names;
    selected->walk([&](TaskflowTaskOp task) {
      std::string name = task.getTaskName().str();
      if (!names.insert(name).second)
        error = "duplicate Taskflow task name: " + name;
      tasks.push_back(std::move(name));
      taskOps.push_back(task);
    });
    if (!error.empty() || tasks.empty()) {
      selected->emitError() << (error.empty() ? "no Taskflow tasks" : error);
      return signalPassFailure();
    }
    bool mixedNeura = false;
    selected->walk([&](Operation *operation) {
      mixedNeura |= operation->getName().getStringRef().starts_with("neura.");
    });

    const int64_t taskCount = static_cast<int64_t>(tasks.size());
    llvm::DenseMap<Operation *, size_t> taskIndices;
    for (auto indexed : llvm::enumerate(taskOps))
      taskIndices[indexed.value().getOperation()] = indexed.index();
    std::string edgeError;
    TaskEdgeGraphOptions edgeOptions;
    edgeOptions.require_payload = false;
    FailureOr<TaskEdgeGraph> edgeGraph =
        buildTaskEdgeGraph(*selected, edgeOptions, edgeError);
    const bool dependencyFactsComplete = succeeded(edgeGraph);
    if (!dependencyFactsComplete && edgeError.empty())
      edgeError = "typed_edge_graph_unavailable";
    std::vector<std::vector<bool>> anyEdge(
        tasks.size(), std::vector<bool>(tasks.size(), false));
    std::vector<std::vector<bool>> dataEdge(
        tasks.size(), std::vector<bool>(tasks.size(), false));
    if (dependencyFactsComplete)
      for (TaskEdge edge : edgeGraph->getEdges()) {
        auto producer = taskIndices.find(edge.producer.getOperation());
        auto consumer = taskIndices.find(edge.consumer.getOperation());
        if (producer == taskIndices.end() || consumer == taskIndices.end()) {
          edgeError = "typed edge endpoint is outside task inventory";
          break;
        }
        anyEdge[producer->second][consumer->second] = true;
        if (edge.kind == TaskEdgeKind::Raw ||
            edge.kind == TaskEdgeKind::Value)
          dataEdge[producer->second][consumer->second] = true;
      }
    const bool preciseDependencyFacts = dependencyFactsComplete && edgeError.empty();
    const char *unknownReason =
        mixedNeura ? "mixed_taskflow_neura_application_proof_unavailable" :
                     "generic_source_rewrite_proof_unavailable";
    // The action alphabet is finite and factored. Compute its exact count
    // before writing, avoiding arithmetic overflow and partial publication.
    if (taskCount > 1000) {
      selected->emitError("task count exceeds action inventory index width");
      return signalPassFailure();
    }
    const int64_t tilingCount =
        taskCount * static_cast<int64_t>(kFactors.size()) *
        (static_cast<int64_t>(kOutputAxes.size()) +
         static_cast<int64_t>(kKPolicies.size()));
    const int64_t pairCount = taskCount * (taskCount - 1) / 2;
    const int64_t fusionCount = 2 * taskCount * (taskCount - 1) + pairCount;
    const int64_t total = 2 + tilingCount + fusionCount;
    const bool bounded = total >= maxActionRecords;
    const int64_t limit = std::min<int64_t>(total, maxActionRecords);
    int64_t emitted = 0;
    int64_t legal = 0;
    int64_t unknown = 0;
    int64_t illegal = 0;
    auto emit = [&](llvm::raw_ostream &os, llvm::json::Object row,
                    StringRef status, StringRef reason) {
      if (emitted >= limit)
        return;
      row["schema"] = kSchema.str();
      row["record_type"] = "action";
      row["status"] = status.str();
      row["reason"] = reason.str();
      writeJsonLine(os, std::move(row));
      ++emitted;
      if (status == "legal")
        ++legal;
      else if (status == "illegal")
        ++illegal;
      else
        ++unknown;
    };
    if (!writeAtomically(outputFile,
          [&](llvm::raw_ostream &os) {
            writeJsonLine(os, llvm::json::Object{
                {"schema", kSchema.str()}, {"record_type", "header"},
                {"function", selected->getSymName().str()},
                {"task_count", taskCount}, {"mixed_taskflow_neura", mixedNeura},
                {"complete", false}, {"status", "incomplete"},
                {"dependency_facts_complete", preciseDependencyFacts},
                {"dependency_error", preciseDependencyFacts ? "" : edgeError},
                {"factor_alphabet", llvm::json::Array{1, 2, 4, 8}},
                {"tiling_axes", llvm::json::Array{"M", "N", "K"}},
                {"k_policies", llvm::json::Array{
                    "sequential_state_carry",
                    "parallel_private_partials_linear_reduction",
                    "parallel_private_partials_balanced_tree_reduction"}},
                {"declared_action_count", total},
                {"semantic_graph_deduplication", "not_performed"}});
            emit(os, llvm::json::Object{{"kind", "graph_noop"}},
                 "legal", "identity_no_op");
            emit(os, llvm::json::Object{{"kind", "fusion"},
                                        {"fusion_mode", "none"}},
                 "legal", "identity_no_op");
            for (const std::string &task : tasks) {
              for (StringRef axis : kOutputAxes)
                for (int64_t factor : kFactors)
                  emit(os, llvm::json::Object{{"kind", "tiling"},
                                               {"task", task},
                                               {"axis", axis.str()},
                                               {"factor", factor}},
                       factor == 1 ? "legal" : "unknown",
                       factor == 1 ? "identity_no_op" : unknownReason);
              for (StringRef policy : kKPolicies)
                for (int64_t factor : kFactors)
                  emit(os, llvm::json::Object{{"kind", "tiling"},
                                               {"task", task},
                                               {"axis", "K"},
                                               {"k_policy", policy.str()},
                                               {"factor", factor}},
                       factor == 1 ? "legal" : "unknown",
                       factor == 1 ? "identity_no_op" : unknownReason);
            }
            for (size_t first = 0; first < tasks.size(); ++first) {
              for (size_t second = 0; second < tasks.size(); ++second) {
                if (first == second)
                  continue;
                for (StringRef mode : kProducerConsumerModes) {
                  const bool provenNoDataEdge =
                      preciseDependencyFacts && !dataEdge[first][second];
                  emit(os, llvm::json::Object{{"kind", "fusion"},
                                               {"fusion_mode", mode.str()},
                                               {"first_task", tasks[first]},
                                               {"second_task", tasks[second]}},
                       provenNoDataEdge ? "illegal" : "unknown",
                       provenNoDataEdge ? "no_direct_data_dependency" :
                                          unknownReason);
                }
                if (first < second) {
                  const bool provenDependent =
                      preciseDependencyFacts &&
                      (anyEdge[first][second] || anyEdge[second][first]);
                  emit(os, llvm::json::Object{{"kind", "fusion"},
                                               {"fusion_mode", "sibling"},
                                               {"first_task", tasks[first]},
                                               {"second_task", tasks[second]}},
                       provenDependent ? "illegal" : "unknown",
                       provenDependent ? "siblings_have_ordering_dependency" :
                                         unknownReason);
                }
              }
            }
            // This compact Cartesian descriptor names both rewrite orders.
            // It does not imply that their materialization or graph closure
            // has occurred. Each sequence remains unknown until C++ passes
            // prove and record its actual intermediate graph.
            writeJsonLine(os, llvm::json::Object{
                {"schema", kSchema.str()}, {"record_type", "order_space"},
                {"orders", llvm::json::Array{"fuse_then_tile",
                                              "tile_then_fuse"}},
                {"fusion_action_count", fusionCount + 1},
                {"tiling_action_count", tilingCount},
                {"declared_order_paths", 2 * (fusionCount + 1) * tilingCount},
                {"paths_materialized", false},
                {"status", "unknown"},
                {"reason", "intermediate_graph_legality_unproven"}});
            writeJsonLine(os, llvm::json::Object{
                {"schema", kSchema.str()}, {"record_type", "footer"},
                {"status", "incomplete"}, {"complete", false},
                {"declared_action_count", total},
                {"emitted_action_count", emitted},
                {"omitted_action_count", total - emitted},
                {"legal_no_op_count", legal},
                {"unknown_action_count", unknown},
                {"illegal_action_count", illegal},
                {"resource_bound_reached", bounded},
                {"dependency_facts_complete", preciseDependencyFacts},
                {"first_blocker", bounded ?
                     "max_action_records_reached" :
                     "nonidentity_action_proofs_and_graph_closure_missing"}});
            return true;
          }, error)) {
      selected->emitError() << error;
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createEnumerateApplicationRewriteActionsPass() {
  return std::make_unique<EnumerateApplicationRewriteActionsPass>();
}
} // namespace mlir::amoeba::neura
