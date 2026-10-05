//===- AnalyticalTaskCostCatalog.h ---------------------------*- C++ -*-===//
//
// Declares the validated task-shape ML cost oracle used by joint scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_ANALYTICAL_TASK_COST_CATALOG_H
#define AMOEBA_ANALYTICAL_TASK_COST_CATALOG_H

#include "SpatialTaskCandidateSpace.h"

#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

inline constexpr llvm::StringLiteral kCostSchema = "amoeba-task-shape-cost";
inline constexpr llvm::StringLiteral kScoreSchema =
    "amoeba-analytical-task-scores";
inline constexpr llvm::StringLiteral kScoreModel =
    "spatial-temporal-scheduler-makespan";
inline constexpr llvm::StringLiteral kCostProvenanceSchema =
    "orbit-cost-provenance-v1";
inline constexpr llvm::StringLiteral kArchitectureSchema =
    "neura-architecture-v1";
inline constexpr llvm::StringLiteral kCandidateIdScheme =
    "candidate-<sequential-index>";
inline constexpr llvm::StringLiteral kModelDomainUnsupportedStatus =
    "unsupported-model-domain";
inline constexpr llvm::StringLiteral kModelDomainUnsupportedReason =
    "analytical-lower-bound-exceeds-model-ceiling";
inline constexpr llvm::StringLiteral kSupportedShapeBootstrapPolicy =
    "minimum-area-supported-model-shape-v1";
inline constexpr double kFormalMax4ModelCeilingII = 20.0;
inline constexpr std::array<std::pair<int64_t, int64_t>, 8>
    kFormalMax4CostShapes = {{{4, 4}, {4, 8}, {8, 4}, {4, 12},
                              {12, 4}, {4, 16}, {8, 8}, {16, 4}}};

// Stores one predictor result. Unsupported queries remain explicit catalogue
// entries so every candidate can be visited and audited.
struct TaskShapeCost {
  double predictedII = 0.0;
  double startupCycles = 0.0;
  bool supported = false;
  bool modelDomainUnsupported = false;
  double analyticalLowerBound = 0.0;
  double modelIntervalMaxII = 0.0;
  double runtimeIICeiling = kFormalMax4ModelCeilingII;
  std::string unsupportedReason;
};

struct RankedCandidate {
  std::string id;
  uint64_t manifestIndex = 0;
  double score = 0.0;
};

using CostQueryKey = std::tuple<std::string, int64_t, int64_t>;

// The task name is deliberately part of the catalogue key.  It is not a
// substitute for checking that every named task in the current IR has an
// explicit cost entry; that check is performed against source_task_ids.
//
// PredictionCacheKey is the reusable prediction identity.  The source task ID
// and oriented mapper dimensions bind the spatial query.  A task's trip count
// is not included because the predictor returns the per-iteration II/startup
// pair; the scorer applies each task's current trip count afterward.
using PredictionCacheKey =
    std::tuple<std::string, int64_t, int64_t>;

// Loads one complete external predictor catalogue and memoizes lookups by the
// stable (source task, oriented mapper shape) identity. The per-task query
// table is retained separately so stale or extra catalogue records cannot be
// hidden by two tasks sharing a body.
class TaskShapeCostCache {
public:
  bool load(llvm::StringRef path, llvm::StringRef expectedFunction,
            llvm::ArrayRef<TaskMetadata> expectedTasks,
            llvm::StringRef expectedSourceRepository,
            llvm::StringRef expectedSourceCommit,
            llvm::StringRef expectedArchitecturePath,
            llvm::StringRef expectedGraphId, std::string &error,
            bool allowModelDomainUnsupported = false,
            llvm::StringRef expectedCanonicalModuleWitness = {});
  const TaskShapeCost *get(const TaskShapeChoice &choice, std::string &error);

  llvm::StringRef nameSpace() const { return namespace_; }
  llvm::StringRef sourceRepository() const { return sourceRepository_; }
  llvm::StringRef sourceCommit() const { return sourceCommit_; }
  llvm::StringRef architecturePath() const { return architecturePath_; }
  llvm::StringRef architectureSchema() const { return architectureSchema_; }
  llvm::StringRef sourceGraphId() const { return sourceGraphId_; }
  llvm::StringRef candidateIdScheme() const { return candidateIdScheme_; }
  // Factored candidate spaces use an arbitrary-width canonical decimal
  // count. Row manifests remain bounded by their signed 64-bit JSON field;
  // callers must check candidateCountFitsInt64() before using candidateCount.
  llvm::StringRef candidateCountDecimal() const {
    return candidateCountDecimal_;
  }
  bool candidateCountFitsInt64() const { return candidateCountFitsInt64_; }
  uint64_t candidateCount() const { return candidateCount_; }
  llvm::StringRef mapperSuccessProbabilityRole() const {
    return mapperSuccessProbabilityRole_;
  }
  double runtimeIICeiling() const { return runtimeIICeiling_; }
  bool hasDiagnosticOverride() const { return diagnosticOverride_; }
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  uint64_t cachedPredictions() const { return predictionCache_.size(); }
  uint64_t coveredQueries() const { return coveredQueries_.size(); }
  uint64_t catalogQueries() const { return catalog_.size(); }

private:
  std::string namespace_;
  std::string sourceRepository_;
  std::string sourceCommit_;
  std::string architecturePath_;
  std::string architectureSchema_;
  std::string sourceGraphId_;
  std::string candidateIdScheme_;
  std::string candidateCountDecimal_;
  bool candidateCountFitsInt64_ = false;
  uint64_t candidateCount_ = 0;
  std::string mapperSuccessProbabilityRole_;
  double runtimeIICeiling_ = kFormalMax4ModelCeilingII;
  bool diagnosticOverride_ = false;
  // Current-IR task IDs. This set is populated only after the catalogue's
  // source_task_ids have been checked against the IR.
  std::set<std::string> taskIds_;
  std::map<CostQueryKey, TaskShapeCost> catalog_;
  std::map<PredictionCacheKey, TaskShapeCost> predictionCache_;
  std::set<CostQueryKey> coveredQueries_;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
};

bool samePath(llvm::StringRef lhs, llvm::StringRef rhs);

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_TASK_COST_CATALOG_H
