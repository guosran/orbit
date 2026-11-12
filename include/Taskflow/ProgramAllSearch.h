#ifndef ORBIT_TASKFLOW_PROGRAM_ALL_SEARCH_H
#define ORBIT_TASKFLOW_PROGRAM_ALL_SEARCH_H
#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/ProgramSearch.h"
#include "Taskflow/Search.h"
namespace orbit::taskflow {
// Sources own the MLIR handles borrowed by their checked child graphs. A
// backend must lower these selected plans, rather than plan their source again
// with defaults and silently replace a searched tile/copy/resource decision.
struct ProgramAllHeterogeneousRegion {
  size_t region = 0;
  mlir::OwningOpRef<mlir::ModuleOp> source;
  HeterogeneousRegionProjection projection;
  GenericMappingPlan native;
  ProgramGraph selectedGraph;
  bool hasNativePlan = false;
  llvm::json::Value receipt = nullptr;
};
struct ProgramAllHeterogeneousCandidate {
  mlir::ModuleOp source; // borrowed for the duration of the evaluator call
  HeterogeneousMappingPlan parent;
  std::vector<ProgramAllHeterogeneousRegion> regions;
  std::string assignmentId, evidenceDirectory;
  llvm::json::Value assignment = nullptr, receipt = nullptr;
};
using ProgramAllHeterogeneousEvaluator = std::function<ProgramSearchEvaluation(
    const ProgramAllHeterogeneousCandidate &)>;
// A priced evaluator result must return candidate_id, parent_graph_id and
// region_graph_ids (in region order). Search verifies these against the checked
// typed candidate before accepting a cost; missing/mismatched receipts are
// UNKNOWN. Region costs own body, transfer and invocation effects exclusively.
// This builder never prices or lowers. Unsupported refinements return false;
// search treats that as UNKNOWN, not mathematical infeasibility. The parent
// owns region ABI/visibility and each checked child owns its native L1/DMA.
bool materializeProgramAllHeterogeneousCandidate(
    mlir::ModuleOp, const llvm::json::Value &configuration,
    const HeterogeneousMappingChoice &, const StaticConfig &,
    const ProgramDataOrganizationChoice &, const std::vector<int64_t> &priority,
    bool reuseSlots, int64_t maxSourceInstances,
    ProgramAllHeterogeneousCandidate &, std::string &error);
// Existing source actions and public graph choices share one finite search.
// No architecture, operator semantics, or cost predictor is defined here.
bool runProgramAllSearch(llvm::StringRef source, const SearchOptions &,
                         const llvm::json::Value &configuration,
                         std::string &error);
bool runProgramAllSearch(llvm::StringRef source, const SearchOptions &,
                         const llvm::json::Value &configuration,
                         std::string &error,
                         const ProgramAllHeterogeneousEvaluator &);
// Reconstruct a frozen source-bound candidate; exact JSON identities must
// match. This verifies one supplied assignment and never searches for a
// replacement.
bool replayPublicProgramGraph(mlir::ModuleOp source,
                              const llvm::json::Value &configuration,
                              const llvm::json::Value &space,
                              const llvm::json::Value &assignment,
                              const llvm::json::Value &expectedEvaluation,
                              ProgramGraph &graph, llvm::json::Value &receipt,
                              std::string &error);
} // namespace orbit::taskflow
#endif
