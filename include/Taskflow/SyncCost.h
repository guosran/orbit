#ifndef ORBIT_TASKFLOW_SYNC_COST_H
#define ORBIT_TASKFLOW_SYNC_COST_H
#include "Taskflow/SyncState.h"

namespace orbit::taskflow {
struct SyncCostRequest {
  SyncQueryResult query;
  std::string site;
  std::string conditionIdentity;
  int64_t argumentOrdinal = -1, count = 1;
  // Actual observed BO API extent: distinguishes whole-explicit from a region.
  int64_t observedExtentBytes = 0;
  SyncDirection direction = SyncDirection::ToDevice;
};
struct SyncCostDomain {
  std::string site, callForm, observerBoundary, direction;
  std::string runtimeIdentity, observerIdentity;
  std::string conditionIdentity;
  int64_t argumentOrdinal = -1;
  std::vector<std::string> allowedStates;
  int64_t minBytes = 0, maxBytes = 0, minCount = 1, maxCount = 1;
  int64_t alignmentBytes = 1;
};
struct SyncCostRow {
  SyncCostDomain domain;
  std::string formula, provenance, rawManifestSHA256;
  double entryNs = 0, callNs = 0, cachelineNs = 0;
};
struct SyncAPICostModel {
  std::string modelId, parameterFileSHA256;
  std::vector<SyncCostRow> rows;
};
struct SyncAPICostEvaluation {
  std::string modelId, parameterFileSHA256;
  std::string status = "unknown", reason, callForm;
  std::optional<size_t> matchedRow;
  std::optional<double> durationNs;
};
// Identity is SHA256 of exact input bytes, not a recursively hashed model_id.
// model_id is a declared label. Only this API scope is priced; compute, wait,
// internal DMA and physical DDR/cache traffic are explicitly outside it.
bool parseSyncAPICostModel(llvm::StringRef jsonBytes, SyncAPICostModel &,
                           std::string &error);
bool evaluateSyncAPICost(const SyncAPICostModel &, const SyncCostRequest &,
                         SyncAPICostEvaluation &, std::string &error);
llvm::json::Value serializeSyncAPICostModel(const SyncAPICostModel &);
llvm::json::Value serializeSyncCostRequest(const SyncCostRequest &);
llvm::json::Value serializeSyncAPICostEvaluation(const SyncAPICostEvaluation &);
} // namespace orbit::taskflow
#endif
