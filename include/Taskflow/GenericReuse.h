#ifndef ORBIT_TASKFLOW_GENERIC_REUSE_H
#define ORBIT_TASKFLOW_GENERIC_REUSE_H
#include "Taskflow/GenericMapping.h"
namespace orbit::taskflow {
// Analysis facts, not selected copies. Exact source rectangles and versions
// identify seeds independently of the current source topological arrangement.
struct GenericReuseOpportunity {
  size_t storage = 0;
  std::vector<GenericInterval> region;
  std::string version;
  int64_t producer = -1;
  std::vector<size_t> consumers;
};
struct GenericReuseProposal {
  ProgramMappingChoice mapping;
  std::vector<int64_t> tileSizes;
  llvm::json::Value diagnostics = nullptr;
};
// Raw work counters for one discovery call. Planner/checker counters include
// cache misses only; screening and cached outcomes are reported separately.
struct GenericReuseStats {
  uint64_t opportunities = 0, scopeSets = 0, tileTrials = 0,
           generationAttempts = 0;
  uint64_t sourceAnalyses = 0, plannerCalls = 0, checkerCalls = 0;
  uint64_t cacheHits = 0, capacityScreenRejects = 0, domainExclusions = 0,
           legal = 0, rejected = 0;
  double elapsedMs = 0;
};
struct GenericReuseOptions {
  std::string tileGenerator = "constraint"; // constraint | historical-halving
  bool catalogue = false, includeBaseline = false;
  // The catalogue anchor is supplied once from the root source. Empty uses
  // full source extents in catalogue mode, current source tile otherwise.
  std::vector<int64_t> currentTile;
  // Explicit optimizer-domain restrictions, not reinterpretations of the
  // backend's tileAlignment preferences. Empty means no hard alignment.
  std::vector<int64_t> requiredAlignment;
  bool requireDivisible = false;
  size_t maxProposals = 8, maxGroups = 4, maxTileProposals = 12;
  size_t maxAttempts = 8192, maxScopeSets = 64,
         maxCatalogueConfigurations = 1024;
  size_t maxPlannerCalls = 1024;
  // Common optimizer-domain work bound, never a hardware legality claim.
  size_t maxComputeInstances = 256;
  double maxWallTimeMs = 0; // zero disables this local deadline
};
struct GenericReuseDiscovery {
  std::vector<GenericReuseProposal> proposals;
  // Bounded rejected attempts remain diagnostic evidence; they are not plans.
  llvm::json::Array rejected;
  GenericReuseStats stats;
  bool catalogueComplete = false;
  std::string catalogueId, stopReason;
};
std::vector<GenericReuseOpportunity>
discoverGenericReuseOpportunities(const GenericAnalysis &);
// Contract selected groups, topologically order the quotient, then expand each
// group without changing source task indices or bodies. Reject contraction
// cycles, overlapping groups and unsatisfied internal dependencies.
bool deriveGenericResidencyOrder(const GenericAnalysis &,
                                 const std::vector<ProgramResidencyGroup> &,
                                 std::vector<size_t> &, std::string &error);
// Joint, bounded region/group/tile proposals. Uses the existing native source
// transforms and checked planner for physical capacity and backend capability.
GenericReuseDiscovery discoverGenericReuseProposals(
    const GenericAnalysis &, const llvm::json::Value &config,
    size_t maxProposals = 8, size_t maxGroups = 4, size_t maxTileProposals = 4);
// Canonical bounded scope × shared-tile catalogue, or a local neighborhood.
// No timing/ranking decision is made here. Capacity-proven screening precedes
// the existing full mapping checker; unsupported proofs use bounded probes.
GenericReuseDiscovery
discoverGenericReuseConfigurations(const GenericAnalysis &,
                                   const llvm::json::Value &config,
                                   const GenericReuseOptions &);
llvm::json::Value serializeGenericReuseStats(const GenericReuseStats &);
// The bounded process-local cache stores checked outcome/metrics JSON, never
// MLIR pointers or a checked plan transferred to another source instance.
void clearGenericReuseCache();
size_t genericReuseCacheSize();
} // namespace orbit::taskflow
#endif
