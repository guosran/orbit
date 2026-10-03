//===- ScoreExactJointTaskCandidatesPass.cpp ------------------------------===//
//
// Ranks one fixed semantic graph over the complete factored task shape space.
// Production mode invokes Amoeba orchestration once per shape tuple; legacy
// diagnostic modes retain their explicit schedule spaces.
//
// This pass is fail-closed and does not call a mapper. A production result
// certifies shape ranking under the existing Amoeba scheduler, with omitted
// shapes justified only by cost lower bounds. It does not claim exhaustive
// placement or dispatch-order coverage.
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalTaskCostCatalog.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "CommunicationExactScheduleSpace.h"
#include "ExactScoreCache.h"
#include "ShapeDomainBounds.h"
#include "ProductionFixedScheduleCommunication.h"
#include "ProductionSchedulerAdapter.h"
#include "SpatialTaskCandidateSpace.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.h"
#include "Backend/Neura/Orchestration/JointScheduling/JointSchedulingCandidateOrder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {

struct FactoredSpaceDescriptor {
  std::string function;
  std::string candidateCount;
  unsigned maxCgrasPerTask = 0;
  SmallVector<SmallVector<RectShape>> shapesByTask;
};

struct TieRestrictedCriticalPathResult {
  // `valid` is false when the ordinal/radix/domain/graph inputs are not
  // sufficient for a proof.  Callers must fail open in that case.
  bool valid = false;
  bool hasEligibleCase = false;
  int64_t lowerBound = std::numeric_limits<int64_t>::max();
};

struct RankedSchedule {
  std::string key;
  std::string shapeOrdinal;
  uint64_t scheduleOrdinal = 0;
  int64_t cycles = 0;
  llvm::json::Object record;
};

// A frontier snapshot is a source-owned continuation token for the production
// shape DFS.  It contains the active decision stack and the incumbent rows
// already produced by this exact pass; it is never a warm-start score file.
// The binding witness is persisted separately as full length-prefixed bytes so
// repeated snapshots remain small while resume still compares every input byte.
struct ShapeFrontierFrameSnapshot {
  size_t depth = 0;
  unsigned task = 0;
  size_t chosenIndex = 0;
  uint64_t nextChoice = 0;
  std::string ordinal;
};

struct ShapeFrontierSnapshot {
  bool resumeEligible = false;
  std::string reason;
  std::string bindingWitnessPath;
  bool canonicalIncumbentSeedEnabled = false;
  std::string canonicalSeedContinuationState;
  uint64_t canonicalSeedSchedulerCalls = 0;
  uint64_t canonicalSeedAccepted = 0;
  uint64_t canonicalSeedNextTask = 0;
  uint64_t canonicalSeedNextSmaller = 0;
  std::string candidateCount;
  std::vector<unsigned> decisionOrder;
  std::vector<ShapeFrontierFrameSnapshot> frames;
  std::vector<RankedSchedule> ranked;
  uint64_t scheduleSerial = 0;
  uint64_t expandedNodes = 0;
  uint64_t expandedShapeNodes = 0;
  uint64_t productionSchedulerCalls = 0;
  uint64_t validShapes = 0;
  uint64_t unsupportedShapes = 0;
  int64_t shapeDomainBound = 0;
  std::vector<std::string> warmupSolvedShapes;
  std::string visitedShapes;
  std::string boundPrunedShapes;
  int64_t savedMaxShapeCandidates = 0;
  int64_t savedMaxExpandedNodes = 0;
  int64_t savedMaxMilliseconds = 0;
};

enum class DispatchPolicy { Fixed, AllReady, CriticalPath };

// Compute an optimistic critical-path bound for the subset of a shape
// subtree whose canonical numeric ordinal is strictly smaller than
// `incumbentOrdinal`.  The production DFS may decide shape variables in a
// non-canonical order, so this helper partitions the canonical ordinal
// comparison into first-difference cases.  For case k, canonical tasks before
// k equal the incumbent digit, task k is smaller, and later tasks are free.
// Every complete ordinal below the incumbent belongs to exactly one case.
//
// This helper deliberately reasons only about source-owned duration minima and
// proven dependency communication lower bounds.  It does not infer placement,
// dispatch, resource occupancy, or scheduler behavior.  Unknown cost rows and
// malformed inputs make the result invalid, preserving the caller's fail-open
// policy.
static TieRestrictedCriticalPathResult computeTieRestrictedCriticalPath(
    StringRef incumbentOrdinal, unsigned counterWidth,
    const FactoredSpaceDescriptor &descriptor,
    const std::vector<std::vector<ShapeDomainOption>> &domains,
    const SmallVector<size_t> &selectedShapes,
    const std::vector<uint8_t> &selectedFlags,
    const std::vector<std::vector<unsigned>> &predecessors,
    const std::vector<std::vector<int64_t>> &edgeLowerBounds) {
  TieRestrictedCriticalPathResult result;
  const size_t taskCount = descriptor.shapesByTask.size();
  if (taskCount == 0 || domains.size() != taskCount ||
      selectedShapes.size() != taskCount || selectedFlags.size() != taskCount ||
      predecessors.size() != taskCount || edgeLowerBounds.size() != taskCount ||
      counterWidth == 0 || incumbentOrdinal.empty() ||
      llvm::any_of(incumbentOrdinal, [](char value) {
        return value < '0' || value > '9';
      }) || descriptor.candidateCount.empty() ||
      llvm::any_of(descriptor.candidateCount, [](char value) {
        return value < '0' || value > '9';
      }))
    return result;
  if (llvm::APInt::getBitsNeeded(incumbentOrdinal, 10) > counterWidth ||
      llvm::APInt::getBitsNeeded(descriptor.candidateCount, 10) > counterWidth)
    return result;

  llvm::APInt target(counterWidth, incumbentOrdinal, 10);
  llvm::APInt candidateCount(counterWidth, descriptor.candidateCount, 10);
  if (candidateCount.isZero() || target.uge(candidateCount))
    return result;

  std::vector<size_t> targetDigits(taskCount, 0);
  llvm::APInt remainder = target;
  llvm::APInt product(counterWidth, 1);
  for (size_t task = taskCount; task > 0; --task) {
    const size_t index = task - 1;
    const size_t radix = descriptor.shapesByTask[index].size();
    if (radix == 0 || radix > std::numeric_limits<uint64_t>::max())
      return result;
    const llvm::APInt radixValue(counterWidth, static_cast<uint64_t>(radix));
    const llvm::APInt digit = remainder.urem(radixValue);
    if (digit.getActiveBits() > sizeof(size_t) * 8)
      return result;
    targetDigits[index] = static_cast<size_t>(digit.getZExtValue());
    remainder = remainder.udiv(radixValue);
    bool overflow = false;
    product = product.umul_ov(radixValue, overflow);
    if (overflow)
      return result;
  }
  if (!remainder.isZero() || product != candidateCount)
    return result;

  // Validate every domain and currently selected option before considering
  // cases.  A stale selected/domain relationship is not evidence that no tie
  // can win, and malformed graph/domain records must fail open.
  for (size_t task = 0; task < taskCount; ++task) {
    if (domains[task].empty() || selectedFlags[task] > 1)
      return result;
    std::set<size_t> ordinals;
    for (const ShapeDomainOption &option : domains[task]) {
      if (option.originalOrdinal >= descriptor.shapesByTask[task].size() ||
          option.duration < 0 || !ordinals.insert(option.originalOrdinal).second)
        return result;
    }
    if (!selectedFlags[task])
      continue;
    bool found = false;
    for (const ShapeDomainOption &option : domains[task]) {
      if (option.originalOrdinal == selectedShapes[task]) {
        found = true;
        break;
      }
    }
    if (!found)
      return result;
  }

  for (size_t task = 0; task < taskCount; ++task) {
    if (edgeLowerBounds[task].size() != taskCount)
      return result;
    std::set<unsigned> seen;
    for (unsigned predecessor : predecessors[task]) {
      if (predecessor >= taskCount || predecessor == task ||
          !seen.insert(predecessor).second)
        return result;
    }
    for (size_t predecessor = 0; predecessor < taskCount; ++predecessor) {
      const int64_t bound = edgeLowerBounds[task][predecessor];
      if (bound < 0 ||
          (!seen.count(static_cast<unsigned>(predecessor)) && bound != 0))
        return result;
    }
  }

  for (size_t firstDifference = 0; firstDifference < taskCount;
       ++firstDifference) {
    std::vector<int64_t> caseDurations(taskCount, 0);
    bool caseEligible = true;
    for (size_t task = 0; task < taskCount && caseEligible; ++task) {
      bool hasOption = false;
      int64_t minimum = std::numeric_limits<int64_t>::max();
      for (const ShapeDomainOption &option : domains[task]) {
        const size_t optionOrdinal = option.originalOrdinal;
        if (task < firstDifference && optionOrdinal != targetDigits[task])
          continue;
        if (task == firstDifference && optionOrdinal >= targetDigits[task])
          continue;
        if (selectedFlags[task] && optionOrdinal != selectedShapes[task])
          continue;
        hasOption = true;
        minimum = std::min(minimum, optimisticShapeDuration(option));
      }
      if (!hasOption) {
        caseEligible = false;
        break;
      }
      caseDurations[task] = minimum;
    }
    if (!caseEligible)
      continue;
    result.hasEligibleCase = true;

    std::vector<int64_t> ends(taskCount, -1);
    std::vector<uint8_t> visiting(taskCount, 0);
    bool graphValid = true;
    std::function<bool(unsigned)> computeEnd = [&](unsigned task) {
      if (task >= taskCount || edgeLowerBounds[task].size() != taskCount) {
        graphValid = false;
        return false;
      }
      if (ends[task] >= 0)
        return true;
      if (visiting[task]) {
        graphValid = false;
        return false;
      }
      visiting[task] = 1;
      int64_t start = 0;
      for (unsigned predecessor : predecessors[task]) {
        if (predecessor >= taskCount || predecessor == task ||
            edgeLowerBounds[task][predecessor] < 0 ||
            !computeEnd(predecessor)) {
          graphValid = false;
          return false;
        }
        const int64_t predecessorEnd = ends[predecessor];
        const int64_t delay = edgeLowerBounds[task][predecessor];
        if (predecessorEnd < 0 ||
            predecessorEnd > std::numeric_limits<int64_t>::max() - delay) {
          graphValid = false;
          return false;
        }
        start = std::max(start, predecessorEnd + delay);
      }
      const int64_t duration = caseDurations[task];
      if (duration < 0 ||
          start > std::numeric_limits<int64_t>::max() - duration) {
        graphValid = false;
        return false;
      }
      visiting[task] = 0;
      ends[task] = start + duration;
      return true;
    };
    for (unsigned task = 0; task < taskCount; ++task)
      if (!computeEnd(task))
        graphValid = false;
    if (!graphValid)
      return TieRestrictedCriticalPathResult{};

    int64_t caseLowerBound = 0;
    for (int64_t end : ends) {
      if (end < 0)
        return TieRestrictedCriticalPathResult{};
      caseLowerBound = std::max(caseLowerBound, end);
    }
    result.lowerBound = std::min(result.lowerBound, caseLowerBound);
  }

  result.valid = true;
  return result;
}

// This contract is part of the exact score-cache identity.  A cache entry
// produced before the production scheduler's fixed-point budget changed must
// not be presented as a complete result under the current scheduler.
static constexpr llvm::StringLiteral kProductionSchedulerContract =
    "production-scheduler-v2-fixed-point64-explicit-communication";
static constexpr llvm::StringLiteral kProductionFixedPointIterationAttr =
    "joint_scheduling_fixed_point_max_iterations";
static constexpr int64_t kDefaultProductionFixedPointIterations = 64;

static bool getProductionFixedPointIterations(func::FuncOp function,
                                              int64_t &iterations,
                                              std::string &error) {
  iterations = kDefaultProductionFixedPointIterations;
  if (auto configured = function->getAttrOfType<IntegerAttr>(
          kProductionFixedPointIterationAttr)) {
    iterations = configured.getInt();
    if (iterations <= 0 || iterations > std::numeric_limits<int>::max()) {
      error = kProductionFixedPointIterationAttr.str() +
              " must be in the range [1, " +
              std::to_string(std::numeric_limits<int>::max()) + "]";
      return false;
    }
  }
  return true;
}

static std::optional<DispatchPolicy>
parseDispatchPolicy(StringRef value) {
  if (value == "fixed")
    return DispatchPolicy::Fixed;
  if (value == "all-ready")
    return DispatchPolicy::AllReady;
  if (value == "critical-path")
    return DispatchPolicy::CriticalPath;
  return std::nullopt;
}

static StringRef scoreStartPolicy(StringRef temporalModel, StringRef scheduleSpace) {
  if (scheduleSpace == "production-scheduler")
    return "production-spatial-temporal-scheduler";
  if (scheduleSpace != "exact")
    return "single-greedy-placement";
  return temporalModel == "dispatch-order" ? "dependency-and-resource-ready" : "all-integer";
}

static StringRef scoreTemporalScope(StringRef temporalModel) {
  if (temporalModel == "production-scheduler")
    return "production-task-scheduler-only";
  return temporalModel == "dispatch-order" ? "task-dispatch-order-only" : "all-integer-diagnostic";
}

static StringRef scoreRankingCertificate(StringRef temporalModel, StringRef scheduleSpace) {
  if (scheduleSpace == "production-scheduler")
    return "proven-shape-space-under-production-scheduler";
  if (scheduleSpace != "exact")
    return "proven-shape-ranking-at-single-heuristic-placement";
  return temporalModel == "dispatch-order" ? "proven-shape-location-dispatch-order-branch-and-bound" : "proven-shape-and-schedule-branch-and-bound";
}

static void addProductionCoverage(llvm::json::Object &record,
                                  StringRef temporalModel,
                                  StringRef scheduleSpace,
                                  StringRef dispatchPolicy,
                                  func::FuncOp function) {
  if (scheduleSpace != "production-scheduler") return;
  record["temporal_model"] = temporalModel.str();
  record["schedule_space"] = scheduleSpace.str();
  record["start_policy"] = scoreStartPolicy(temporalModel, scheduleSpace).str();
  record["temporal_search_scope"] = scoreTemporalScope(temporalModel).str();
  record["scheduler_backend"] = "orchestrate-tasks-on-accelerators";
  auto iterations = function->getAttrOfType<IntegerAttr>(
      "joint_scheduling_fixed_point_max_iterations");
  record["production_fixed_point_max_iterations"] =
      iterations ? iterations.getInt() : int64_t{64};
  record["production_scheduler_contract"] = kProductionSchedulerContract.str();
  record["tie_key_policy"] = kProductionNumericTieKeyPolicy.str();
  record["dispatch_policy"] = dispatchPolicy.str();
  record["dispatch_policy_scope"] = dispatchPolicy == "fixed"
      ? "canonical-smallest-ready-task" : "production-critical-path-dependency-ready";
  record["cycle_starts_are_search_dimensions"] = false;
  record["global_shape_location_temporal_space"] = false;
  record["placement_coverage_limited"] = true;
  record["location_coverage_limited"] = true;
  record["temporal_coverage_limited"] = true;
  record["orbit_4x4_pruning"] = false;
  record["shape_domain_pruning"] = record.getBoolean("ranking_pruning").value_or(true)
      ? "dependency-release-tail-cost-only" : "disabled";
}

static bool isCanonicalFixedDispatch(
    ArrayRef<unsigned> order,
    ArrayRef<ExactScheduleTask> tasks) {
  std::vector<bool> scheduled(tasks.size(), false);
  for (unsigned current : order) {
    if (current >= tasks.size() || scheduled[current])
      return false;
    unsigned smallestReady = tasks.size();
    for (unsigned candidate = 0; candidate < tasks.size(); ++candidate) {
      if (scheduled[candidate])
        continue;
      bool ready = true;
      for (unsigned predecessor : tasks[candidate].predecessors) {
        if (predecessor >= tasks.size() || !scheduled[predecessor]) {
          ready = false;
          break;
        }
      }
      if (ready) {
        smallestReady = candidate;
        break;
      }
    }
    if (smallestReady == tasks.size() || current != smallestReady)
      return false;
    scheduled[current] = true;
  }
  return llvm::all_of(scheduled, [](bool value) { return value; });
}

static std::optional<std::string>
getDecimal(const llvm::json::Object &object, StringRef key,
           std::string &error) {
  if (std::optional<int64_t> value = object.getInteger(key)) {
    if (*value < 0) {
      error = "negative factored count for " + key.str();
      return std::nullopt;
    }
    return std::to_string(*value);
  }
  std::optional<StringRef> text = object.getString(key);
  if (!text || text->empty() ||
      llvm::any_of(*text, [](char c) { return c < '0' || c > '9'; })) {
    error = "factored count is not a decimal string for " + key.str();
    return std::nullopt;
  }
  StringRef normalized = text->ltrim('0');
  return (normalized.empty() ? StringRef("0") : normalized).str();
}

static int compareDecimal(StringRef lhs, StringRef rhs) {
  if (lhs.size() != rhs.size())
    return lhs.size() < rhs.size() ? -1 : 1;
  if (lhs == rhs)
    return 0;
  return lhs < rhs ? -1 : 1;
}

static void putDecimal(llvm::json::Object &object, StringRef key,
                       StringRef value) {
  uint64_t numeric = 0;
  if (!value.getAsInteger(10, numeric) &&
      numeric <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    object[key] = static_cast<int64_t>(numeric);
  else
    object[key] = value.str();
}

// A global incumbent is deliberately a small, source-owned certificate rather
// than a caller-supplied numeric cutoff.  It is read from a complete C++ score
// artifact (or from the complete global ranker output) and is checked against
// the current function/stage/source/architecture before it can affect search.
struct GlobalIncumbent {
  bool requested = false;
  bool valid = false;
  // This is the completeness of the evidence closure itself.  A known legal
  // member remains a valid numeric incumbent when its source closure is
  // bounded/incomplete, but that evidence must never be presented as a proof
  // that the whole stage closure was enumerated.
  bool evidenceClosureComplete = false;
  // A seed-only score artifact proves only that its emitted rows replay as
  // feasible schedules.  It is deliberately never treated as a local or
  // stage-complete ranking certificate.
  bool evidenceSeedOnly = false;
  std::string path;
  std::string stage;
  std::string function;
  std::string sourceRepository;
  std::string sourceCommit;
  std::string architecturePath;
  std::string architectureSchema;
  std::string costNamespace;
  std::string closurePath;
  std::string keyPolicy =
      "predicted_whole_program_cycles,semantic_graph_id,candidate_id,graph_variant_id";
  int64_t fifthCycles = 0;
  unsigned candidateCount = 0;
  std::string error;
};

struct IncumbentRow {
  int64_t cycles = 0;
  std::string semanticGraphId;
  std::string candidateId;
  std::string graphVariantId;
  std::string scorePath;
  llvm::json::Object scoreRecord;
};

// A closure row is the source-owned binding for one graph variant.  The
// graph id alone is insufficient for an incumbent: the score command must be
// tied to the exact materialized module and facts that were admitted by the
// closure pass.
struct ClosureGraphBinding {
  std::string path;
  std::string graphVariantId;
  std::string mlirPath;
  std::string factsPath;
  bool ambiguous = false;
};

// Closure JSON is shared by every per-graph score invocation in one process.
// Keep only one parsed representation per thread, keyed by the complete
// request identity and exact closure bytes.  The cached values are plain
// strings/maps: no MLIR operation or parser-owned pointer may outlive a call.
struct ClosureGraphParseCache {
  bool initialized = false;
  bool parseValid = false;
  std::string path;
  std::string expectedFunction;
  std::string expectedStage;
  std::string bytes;
  std::map<std::string, ClosureGraphBinding> graphBindings;
  bool closureComplete = false;
  std::string error;
};

static thread_local ClosureGraphParseCache closureGraphParseCache;

static bool closureGraphParseCacheMatches(StringRef path,
                                          StringRef expectedFunction,
                                          StringRef expectedStage,
                                          StringRef bytes) {
  const ClosureGraphParseCache &cache = closureGraphParseCache;
  return cache.initialized && cache.path == path &&
         cache.expectedFunction == expectedFunction &&
         cache.expectedStage == expectedStage && cache.bytes == bytes;
}

static void cacheClosureGraphParseFailure(StringRef path,
                                          StringRef expectedFunction,
                                          StringRef expectedStage,
                                          StringRef bytes, StringRef error) {
  ClosureGraphParseCache replacement;
  replacement.initialized = true;
  replacement.parseValid = false;
  replacement.path = path.str();
  replacement.expectedFunction = expectedFunction.str();
  replacement.expectedStage = expectedStage.str();
  replacement.bytes = bytes.str();
  replacement.error = error.str();
  closureGraphParseCache = std::move(replacement);
}

// Parsing is cacheable, but these paths are external mutable inputs. Recheck
// their existence on every positive cache hit so deleting an artifact cannot
// turn a cached closure into a valid incumbent.
static bool revalidateClosureGraphArtifacts(
    const std::map<std::string, ClosureGraphBinding> &graphBindings,
    std::string &error) {
  for (const auto &entry : graphBindings) {
    const ClosureGraphBinding &binding = entry.second;
    if (!llvm::sys::fs::exists(binding.mlirPath) ||
        !llvm::sys::fs::exists(binding.factsPath)) {
      error = "incumbent closure graph lacks complete persisted artifacts";
      return false;
    }
  }
  return true;
}

struct CostCatalogBinding {
  std::string nameSpace;
  std::string model;
  std::string modelSchema;
  std::string featureContractId;
  std::string featureExtractor;
  std::string predictorSource;
  std::string provenanceSchema;
  std::string architectureSchema;
  std::string architecturePath;
  std::string sourceRepository;
  std::string sourceCommit;
  std::string sourceGraphId;
};

static bool parsePositiveCycle(const llvm::json::Object &object,
                               int64_t &cycles) {
  std::optional<int64_t> value =
      object.getInteger("predicted_whole_program_cycles");
  if (!value || *value <= 0)
    return false;
  cycles = *value;
  return true;
}

static bool jsonDecimalIsZero(const llvm::json::Object &object,
                              StringRef key) {
  if (std::optional<int64_t> value = object.getInteger(key))
    return *value == 0;
  std::optional<StringRef> text = object.getString(key);
  return text && !text->empty() &&
         llvm::all_of(*text, [](char value) { return value == '0'; });
}

static bool jsonDecimalEquals(const llvm::json::Object &object, StringRef key,
                              uint64_t expected) {
  if (std::optional<int64_t> value = object.getInteger(key))
    return *value >= 0 && static_cast<uint64_t>(*value) == expected;
  std::optional<StringRef> text = object.getString(key);
  uint64_t value = 0;
  return text && !text->empty() && !text->getAsInteger(10, value) &&
         value == expected;
}

static std::optional<uint64_t>
jsonDecimalValue(const llvm::json::Object &object, StringRef key) {
  if (std::optional<int64_t> value = object.getInteger(key)) {
    if (*value < 0)
      return std::nullopt;
    return static_cast<uint64_t>(*value);
  }
  std::optional<StringRef> text = object.getString(key);
  uint64_t value = 0;
  if (!text || text->empty() || text->getAsInteger(10, value))
    return std::nullopt;
  return value;
}

// A global cutoff is sound for the current graph when every selected row names
// a legal graph member in the source-owned closure.  The closure may be
// bounded/incomplete: that only means the incumbent is an observed upper
// bound, never a claim that the whole stage closure was enumerated.  The
// closure pass uses graph_id in its persisted JSONL; graph_variant_id is
// accepted as a forward compatible alias.  This check intentionally validates
// the materialized MLIR/facts artifacts as well as footer counters, so a stale
// path-only certificate cannot become an incumbent for a different graph set.
static bool loadLegalClosureGraphIds(
  StringRef path, StringRef expectedFunction, StringRef expectedStage,
    std::map<std::string, ClosureGraphBinding> &graphBindings,
    bool &closureComplete,
    std::string &error) {
  graphBindings.clear();
  closureComplete = false;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read incumbent graph closure " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  std::string closureBytes = (*buffer)->getBuffer().str();
  if (closureGraphParseCacheMatches(path, expectedFunction, expectedStage,
                                    closureBytes)) {
    if (!closureGraphParseCache.parseValid) {
      error = closureGraphParseCache.error;
      return false;
    }
    graphBindings = closureGraphParseCache.graphBindings;
    closureComplete = closureGraphParseCache.closureComplete;
    return revalidateClosureGraphArtifacts(graphBindings, error);
  }
  auto cacheParseFailure = [&](StringRef failure) {
    cacheClosureGraphParseFailure(path, expectedFunction, expectedStage,
                                  closureBytes, failure);
    error = failure.str();
    return false;
  };
  llvm::MemoryBufferRef closureBuffer(closureBytes, path);
  llvm::line_iterator lines(closureBuffer, true);
  bool sawHeader = false;
  bool sawFooter = false;
  bool headerComplete = false;
  unsigned graphRows = 0;
  std::set<std::string> structuralKeys;
  std::optional<llvm::json::Object> footer;
  for (; !lines.is_at_end(); ++lines) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
    if (!parsed) {
      return cacheParseFailure("invalid incumbent closure JSON: " +
                               llvm::toString(parsed.takeError()));
    }
    const llvm::json::Object *object = parsed->getAsObject();
    if (!object) {
      return cacheParseFailure("incumbent closure record is not an object");
    }
    if (object->getString("schema") !=
        std::optional<StringRef>("orbit-joint-graph-closure-v1")) {
      return cacheParseFailure("incumbent closure has the wrong schema");
    }
    std::optional<StringRef> type = object->getString("record_type");
    if (!type) {
      return cacheParseFailure("incumbent closure record has no record_type");
    }
    if (!sawHeader) {
      if (*type != "header") {
        return cacheParseFailure("incumbent closure does not start with a header");
      }
      std::optional<StringRef> status = object->getString("status");
      std::optional<bool> complete = object->getBoolean("complete");
      if (object->getString("function") !=
              std::optional<StringRef>(expectedFunction) ||
          object->getString("stage") != std::optional<StringRef>(expectedStage) ||
          !status || (*status != "complete" && *status != "incomplete") ||
          !complete || ((*status == "complete") != *complete)) {
        return cacheParseFailure(
            "incumbent closure header is not a valid requested stage");
      }
      headerComplete = *complete;
      sawHeader = true;
      continue;
    }
    if (sawFooter) {
      return cacheParseFailure("incumbent closure has records after its footer");
    }
    if (*type == "footer") {
      footer = *object;
      sawFooter = true;
      continue;
    }
    if (*type != "graph")
      continue; // action rows are checked by the complete/unknown counters.
    // Resource-bounded closure runs persist rejected/unknown attempts with a
    // graph record type but without structural_key.  They are blockers for
    // stage completeness, not legal graph members, and are ignored here.
    if (object->getString("status") !=
            std::optional<StringRef>("legal") ||
        !object->getObject("structural_key"))
      continue;
    std::optional<StringRef> id = object->getString("graph_variant_id");
    if (!id)
      id = object->getString("graph_id");
    std::optional<StringRef> pathField = object->getString("path");
    std::optional<StringRef> mlirPath = object->getString("mlir_path");
    std::optional<StringRef> factsPath = object->getString("facts_path");
    if (!id || id->empty() || !pathField || pathField->empty() ||
        !mlirPath || mlirPath->empty() || !factsPath || factsPath->empty() ||
        // These fields prove rewrite-space coverage, not feasibility of an
        // already materialized graph. An incomplete closure can still provide
        // a verified feasible incumbent. Complete-closure claims retain the
        // strict rewrite-facts requirement and footer consistency checks.
        (headerComplete &&
         (((expectedStage == "shape-temporal-replica-tiling" ||
            expectedStage == "full-joint") &&
           !object->getBoolean("k_facts_complete").value_or(false)) ||
          (expectedStage == "full-joint" &&
           (!object->getBoolean("sibling_facts_complete").value_or(false) ||
            !object->getBoolean("sibling_materialization_complete").value_or(false)))))) {
      return cacheParseFailure(
          "incumbent closure graph lacks complete persisted artifacts");
    }
    ClosureGraphBinding binding;
    binding.path = pathField->str();
    binding.graphVariantId = id->str();
    binding.mlirPath = mlirPath->str();
    binding.factsPath = factsPath->str();
    auto [bindingIt, inserted] =
        graphBindings.emplace(id->str(), std::move(binding));
    if (!inserted) {
      // Older closure artifacts can repeat the identity record for the same
      // persisted artifact while walking no-op actions.  Preserve those
      // records, but mark an identity with different artifacts ambiguous so
      // an incumbent referring to it is rejected below.
      if (bindingIt->second.mlirPath != mlirPath->str() ||
          bindingIt->second.factsPath != factsPath->str())
        bindingIt->second.ambiguous = true;
    }
    std::string structuralText;
    llvm::raw_string_ostream structuralStream(structuralText);
    llvm::json::Object structuralObject =
        *object->getObject("structural_key");
    structuralStream << llvm::json::Value(std::move(structuralObject));
    structuralStream.flush();
    if (structuralKeys.insert(std::move(structuralText)).second)
      ++graphRows;
  }
  if (!sawHeader || !sawFooter || !footer) {
    return cacheParseFailure("incumbent closure has no footer");
  }
  const llvm::json::Object &footerObject = *footer;
  std::optional<StringRef> footerStatus = footerObject.getString("status");
  std::optional<bool> footerComplete = footerObject.getBoolean("complete");
  const bool completeFooter =
      footerStatus && *footerStatus == "complete" && footerComplete &&
      *footerComplete && footerObject.getBoolean("graph_scope_complete")
          .value_or(false) &&
      !footerObject.getBoolean("resource_bound_reached").value_or(true) &&
      jsonDecimalIsZero(footerObject, "unknown_action_count") &&
      jsonDecimalIsZero(footerObject, "incomplete_facts_count");
  const bool boundedFooter =
      footerStatus && *footerStatus == "incomplete" && footerComplete &&
      !*footerComplete &&
      !footerObject.getBoolean("graph_scope_complete").value_or(false);
  std::optional<uint64_t> footerGraphCount =
      jsonDecimalValue(footerObject, "graph_count");
  std::optional<uint64_t> footerStructuralCount =
      jsonDecimalValue(footerObject, "unique_structural_keys");
  if ((!completeFooter && !boundedFooter) ||
      headerComplete != completeFooter || !footerGraphCount ||
      *footerGraphCount != graphRows || !footerStructuralCount ||
      *footerStructuralCount == 0) {
    return cacheParseFailure(
        "incumbent closure footer is not a valid graph member proof");
  }
  if (graphBindings.empty())
    return cacheParseFailure("incumbent closure contains no legal graph bindings");
  closureComplete = completeFooter;
  ClosureGraphParseCache replacement;
  replacement.initialized = true;
  replacement.parseValid = true;
  replacement.path = path.str();
  replacement.expectedFunction = expectedFunction.str();
  replacement.expectedStage = expectedStage.str();
  replacement.bytes = std::move(closureBytes);
  replacement.graphBindings = graphBindings;
  replacement.closureComplete = closureComplete;
  closureGraphParseCache = std::move(replacement);
  return revalidateClosureGraphArtifacts(graphBindings, error);
}

static bool readStageManifestClosure(StringRef manifestPath,
                                     StringRef expectedFunction,
                                     StringRef expectedStage,
                                     StringRef expectedRepository,
                                     StringRef expectedCommit,
                                     StringRef expectedArchitecture,
                                     StringRef expectedArchitectureSchema,
                                     StringRef expectedCostNamespace,
                                     std::string &closurePath,
                                     std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(manifestPath);
  if (!buffer) {
    error = "cannot read global stage manifest " + manifestPath.str() + ": " +
            buffer.getError().message();
    return false;
  }
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "invalid global stage manifest: " + llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root || root->getString("schema") !=
                   std::optional<StringRef>("orbit-global-stage-ranker-input-v1")) {
    error = "global stage manifest has the wrong schema";
    return false;
  }
  auto require = [&](StringRef key, StringRef expected) {
    return root->getString(key) == std::optional<StringRef>(expected);
  };
  if (!require("function", expectedFunction) || !require("stage", expectedStage) ||
      !require("source_repository", expectedRepository) ||
      !require("source_commit", expectedCommit) ||
      !require("architecture_path", expectedArchitecture)) {
    error = "global stage manifest provenance disagrees with the score request";
    return false;
  }
  if (auto architectureSchema = root->getString("architecture_schema")) {
    if (*architectureSchema != expectedArchitectureSchema) {
      error = "global stage manifest architecture schema disagrees with the score request";
      return false;
    }
  }
  if (auto costNamespace = root->getString("cost_namespace")) {
    if (*costNamespace != expectedCostNamespace ||
        root->getString("cost_provenance_schema") !=
            std::optional<StringRef>("orbit-cost-provenance-v1")) {
      error = "global stage manifest cost provenance disagrees with the score request";
      return false;
    }
  }
  std::optional<StringRef> raw = root->getString("graph_closure_record_path");
  if (!raw)
    raw = root->getString("graph_closure_path");
  if (!raw || raw->empty()) {
    error = "global stage manifest has no graph closure path";
    return false;
  }
  llvm::SmallString<256> resolved;
  if (llvm::sys::path::is_absolute(*raw))
    resolved = *raw;
  else {
    resolved = manifestPath;
    llvm::sys::path::remove_filename(resolved);
    llvm::sys::path::append(resolved, *raw);
  }
  if (std::error_code ec = llvm::sys::fs::make_absolute(resolved)) {
    error = "cannot resolve global stage closure path: " + ec.message();
    return false;
  }
  llvm::sys::path::remove_dots(resolved, true);
  closurePath = resolved.str().str();
  return true;
}

static bool validateIncumbentBindings(
    func::FuncOp currentFunction, ArrayRef<TaskMetadata> currentTasks,
    const TaskShapeCostCache &currentCosts, StringRef currentCandidatePath,
    StringRef currentCostPath, StringRef expectedDispatchPolicy,
    StringRef expectedTemporalModel, StringRef expectedScheduleSpace,
    ArrayRef<IncumbentRow> rows,
    const std::map<std::string, ClosureGraphBinding> &graphBindings,
    std::string &error);

static bool loadGlobalIncumbent(StringRef path, StringRef expectedFunction,
                                StringRef expectedStage,
                                StringRef expectedRepository,
                                StringRef expectedCommit,
                                StringRef expectedArchitecture,
                                StringRef expectedArchitectureSchema,
                                StringRef expectedCostNamespace,
                                StringRef stageManifestPath,
                                func::FuncOp currentFunction,
                                ArrayRef<TaskMetadata> currentTasks,
                                const TaskShapeCostCache &currentCosts,
                                StringRef currentCandidatePath,
                                StringRef currentCostPath,
                                StringRef expectedDispatchPolicy,
                                StringRef expectedTemporalModel,
                                StringRef expectedScheduleSpace,
                                GlobalIncumbent &incumbent) {
  incumbent.path = path.str();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    incumbent.error = "cannot read global incumbent " + path.str() + ": " +
                      buffer.getError().message();
    return false;
  }

  llvm::line_iterator lines(**buffer, true);
  std::optional<llvm::json::Object> header;
  std::optional<llvm::json::Object> footer;
  std::vector<IncumbentRow> rows;
  std::string schema;
  bool sawHeader = false;
  bool sawFooter = false;
  unsigned lineNumber = 0;
  for (; !lines.is_at_end(); ++lines) {
    ++lineNumber;
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
    if (!parsed) {
      incumbent.error = "invalid global incumbent JSON at line " +
                        std::to_string(lineNumber) + ": " +
                        llvm::toString(parsed.takeError());
      return false;
    }
    const llvm::json::Object *object = parsed->getAsObject();
    if (!object) {
      incumbent.error = "global incumbent record is not an object";
      return false;
    }
    std::optional<StringRef> recordSchema = object->getString("schema");
    std::optional<StringRef> type = object->getString("record_type");
    if (!recordSchema || !type) {
      incumbent.error = "global incumbent record has no schema/record_type";
      return false;
    }
    if (!sawHeader) {
      if (*type != "header") {
        incumbent.error = "global incumbent does not start with a header";
        return false;
      }
      schema = recordSchema->str();
      if (schema != "amoeba-exact-joint-task-scores-v1" &&
          schema != "orbit-global-stage-top5-v1") {
        incumbent.error = "global incumbent has an unsupported C++ schema";
        return false;
      }
      header = *object;
      sawHeader = true;
      continue;
    }
    if (sawFooter) {
      incumbent.error = "global incumbent has records after its footer";
      return false;
    }
    if (*type == "footer") {
      if (*recordSchema != schema) {
        incumbent.error = "global incumbent footer schema disagrees";
        return false;
      }
      footer = *object;
      sawFooter = true;
      continue;
    }
    const bool exactSchema = schema == "amoeba-exact-joint-task-scores-v1";
    const bool globalSchema = schema == "orbit-global-stage-top5-v1";
    if ((exactSchema && *type != "score") ||
        (globalSchema && *type != "selection")) {
      incumbent.error = "global incumbent contains an unexpected record type";
      return false;
    }
    std::optional<StringRef> candidateId = object->getString("candidate_id");
    int64_t cycles = 0;
    if (!candidateId || candidateId->empty() ||
        !parsePositiveCycle(*object, cycles)) {
      incumbent.error = "global incumbent contains an invalid/duplicate candidate";
      return false;
    }
    if (exactSchema &&
        object->getBoolean("valid").value_or(false) == false) {
      incumbent.error = "global incumbent contains an uncertified score row";
      return false;
    }
    if (globalSchema &&
        !object->getBoolean("certified").value_or(false) &&
        !object->getBoolean("feasible").value_or(false)) {
      incumbent.error = "global incumbent contains an uncertified selection";
      return false;
    }
    IncumbentRow row;
    row.cycles = cycles;
    row.candidateId = candidateId->str();
    row.scorePath = path.str();
    if (globalSchema) {
      auto semantic = object->getString("semantic_graph_id");
      auto graphVariant = object->getString("graph_variant_id");
      if (!semantic || semantic->empty() || !graphVariant ||
          graphVariant->empty()) {
        incumbent.error = "global incumbent selection lacks graph identity";
        return false;
      }
      row.semanticGraphId = semantic->str();
      row.graphVariantId = graphVariant->str();
      auto scorePath = object->getString("score_file_path");
      if (!scorePath || scorePath->empty()) {
        incumbent.error = "global incumbent selection has no score_file_path";
        return false;
      }
      row.scorePath = scorePath->str();
      const llvm::json::Object *scoreRecord =
          object->getObject("score_record");
      if (!scoreRecord || !scoreRecord->getBoolean("valid").value_or(false) ||
          scoreRecord->getString("schema") !=
              std::optional<StringRef>("amoeba-exact-joint-task-scores-v1") ||
          scoreRecord->getString("candidate_id") !=
              std::optional<StringRef>(*candidateId) ||
          scoreRecord->getInteger("predicted_whole_program_cycles") !=
              std::optional<int64_t>(cycles) ||
          !scoreRecord->getArray("task_costs") ||
          scoreRecord->getArray("task_costs")->empty() ||
          !scoreRecord->getArray("task_schedule") ||
          scoreRecord->getArray("task_schedule")->empty() ||
          scoreRecord->getString("score_source") != std::optional<StringRef>(
              "ml-ii-startup-plus-production-explicit-communication") ||
          scoreRecord->getString("task_latency_provenance") !=
              std::optional<StringRef>(
                  "source-owned-ml-ii-startup-trip-count-contract")) {
        incumbent.error = "global incumbent selection lacks a complete score proof";
        return false;
      }
      row.scoreRecord = *scoreRecord;
    } else {
      row.graphVariantId = "";
      row.semanticGraphId = "";
      if (!object->getBoolean("valid").value_or(false) ||
          object->getString("schema") !=
              std::optional<StringRef>("amoeba-exact-joint-task-scores-v1") ||
          !object->getArray("task_costs") ||
          object->getArray("task_costs")->empty() ||
          !object->getArray("task_schedule") ||
          object->getArray("task_schedule")->empty() ||
          object->getString("score_source") != std::optional<StringRef>(
              "ml-ii-startup-plus-production-explicit-communication") ||
          object->getString("task_latency_provenance") !=
              std::optional<StringRef>(
                  "source-owned-ml-ii-startup-trip-count-contract")) {
        incumbent.error = "exact incumbent row lacks a complete score proof";
        return false;
      }
      row.scoreRecord = *object;
    }
    rows.push_back(std::move(row));
  }

  if (!header || !footer || rows.size() < 5) {
    incumbent.error = "global incumbent requires a header, footer and at least five rows";
    return false;
  }
  const llvm::json::Object &headerObject = *header;
  const llvm::json::Object &footerObject = *footer;
  auto requireHeader = [&](StringRef key, StringRef expected,
                           StringRef label) -> bool {
    std::optional<StringRef> actual = headerObject.getString(key);
    if (!actual || *actual != expected) {
      incumbent.error = "global incumbent " + label.str() + " disagrees";
      return false;
    }
    return true;
  };
  if (!requireHeader("function", expectedFunction, "function") ||
      !requireHeader("source_repository", expectedRepository, "source repository") ||
      !requireHeader("source_commit", expectedCommit, "source commit") ||
      !requireHeader("architecture_path", expectedArchitecture,
                     "architecture path"))
    return false;
  // Older exact score artifacts predate these two namespace fields.  They are
  // accepted for compatibility, but when present they must bind to the
  // current source-owned architecture/cost catalogue; the target catalogue
  // values are always copied into the generated cutoff footer below.
  if (auto architectureSchema = headerObject.getString("architecture_schema")) {
    if (*architectureSchema != expectedArchitectureSchema) {
      incumbent.error = "global incumbent architecture schema disagrees";
      return false;
    }
  }
  if (auto costNamespace = headerObject.getString("cost_namespace")) {
    if (*costNamespace != expectedCostNamespace ||
        headerObject.getString("cost_provenance_schema") !=
            std::optional<StringRef>("orbit-cost-provenance-v1")) {
      incumbent.error = "global incumbent cost provenance disagrees";
      return false;
    }
  } else if (headerObject.getString("cost_provenance_schema")) {
    incumbent.error = "global incumbent has cost provenance without a namespace";
    return false;
  }
  const bool exactSchema = schema == "amoeba-exact-joint-task-scores-v1";
  if (auto stageHeader = headerObject.getString("stage")) {
    if (*stageHeader != expectedStage) {
      incumbent.error = "global incumbent stage disagrees";
      return false;
    }
  } else if (!exactSchema || stageManifestPath.empty()) {
    incumbent.error = "global incumbent has no C++ stage identity";
    return false;
  }
  std::map<std::string, ClosureGraphBinding> graphBindings;
  std::string closurePath;
  if (auto closure = headerObject.getString("graph_closure_record_path"))
    closurePath = closure->str();
  else if (!stageManifestPath.empty() &&
           !readStageManifestClosure(stageManifestPath, expectedFunction,
                                     expectedStage, expectedRepository,
                                     expectedCommit, expectedArchitecture,
                                     expectedArchitectureSchema,
                                     expectedCostNamespace,
                                     closurePath, incumbent.error))
    return false;
  if (closurePath.empty()) {
    incumbent.error = "global incumbent has no graph closure binding";
    return false;
  }
  incumbent.closurePath = closurePath;
  bool closureComplete = false;
  if (!loadLegalClosureGraphIds(closurePath, expectedFunction, expectedStage,
                                graphBindings, closureComplete,
                                incumbent.error))
    return false;
  incumbent.evidenceClosureComplete = closureComplete;
  if (auto predictor = headerObject.getString("predictor_source")) {
    if (*predictor != "ml-ii-startup-predictor") {
      incumbent.error = "global incumbent predictor source is not source-owned ML";
      return false;
    }
  } else {
    incumbent.error = "global incumbent has no predictor source";
    return false;
  }
  for (IncumbentRow &row : rows) {
    if (exactSchema) {
      auto graph = headerObject.getString("graph_variant_id");
      if (!graph || graph->empty()) {
        incumbent.error = "exact incumbent has no graph identity";
        return false;
      }
      row.graphVariantId = graph->str();
      row.semanticGraphId = headerObject.getString("semantic_graph_id")
                                .value_or(*graph)
                                .str();
    }
    if (!graphBindings.count(row.graphVariantId)) {
      incumbent.error = "global incumbent candidate is outside the complete closure";
      return false;
    }
  }
  if (!validateIncumbentBindings(
          currentFunction, currentTasks, currentCosts, currentCandidatePath,
          currentCostPath, expectedDispatchPolicy, expectedTemporalModel,
          expectedScheduleSpace, rows, graphBindings, incumbent.error))
    return false;
  std::set<std::tuple<int64_t, std::string, std::string, std::string>> keys;
  for (const IncumbentRow &row : rows)
    if (!keys.emplace(row.cycles, row.semanticGraphId, row.candidateId,
                      row.graphVariantId)
             .second) {
      incumbent.error = "global incumbent repeats a stable candidate tuple";
      return false;
    }
  auto topK = headerObject.getInteger("top_k");
  if (!topK)
    topK = headerObject.getInteger("top_k_requested");
  if (!topK || *topK < 5) {
    incumbent.error = "global incumbent header top_k is smaller than five";
    return false;
  }
  auto footerStatus = footerObject.getString("status");
  auto footerComplete = footerObject.getBoolean("complete");
  auto footerIncomplete = footerObject.getBoolean("incomplete");
  auto footerCertified = footerObject.getBoolean("top_k_certified");
  const bool exactCompleteFooter =
      footerStatus && *footerStatus == "complete" && footerCertified &&
      *footerCertified && footerIncomplete && !*footerIncomplete;
  const bool exactSeedFooter =
      exactSchema && footerStatus && *footerStatus == "seed-only" &&
      footerIncomplete && *footerIncomplete && footerCertified &&
      !*footerCertified && !footerComplete.value_or(false) &&
      footerObject.getBoolean("seed_only").value_or(false) &&
      footerObject.getString("ranking_certificate") ==
          std::optional<StringRef>("feasible-seed-only");
  const bool globalCompleteFooter =
      footerStatus && *footerStatus == "complete" && footerComplete &&
      *footerComplete && footerCertified && *footerCertified &&
      footerObject.getBoolean("stage_certified").value_or(false);
  const bool globalSeedFooter =
      !exactSchema && footerStatus && *footerStatus == "seed-only" &&
      footerIncomplete && *footerIncomplete &&
      !footerComplete.value_or(false) && footerCertified && !*footerCertified &&
      footerObject.getBoolean("seed_only").value_or(false) &&
      footerObject.getBoolean("feasible_only").value_or(false) &&
      footerObject.getString("ranking_certificate") ==
          std::optional<StringRef>("feasible-seed-only") &&
      footerObject.getInteger("seed_feasible_count").value_or(0) >= 5;
  const bool completeFooter = exactSchema
                                  ? (exactCompleteFooter || exactSeedFooter)
                                  : (globalCompleteFooter || globalSeedFooter);
  if (!completeFooter) {
    incumbent.error =
        "global incumbent footer is neither a complete C++ certificate nor a "
        "validated seed-only artifact";
    return false;
  }
  incumbent.evidenceSeedOnly = exactSeedFooter;
  if (globalSeedFooter)
    incumbent.evidenceSeedOnly = true;
  if (exactSeedFooter && !headerObject.getString("stage")) {
    incumbent.error = "global incumbent seed-only artifact has no stage identity";
    return false;
  }
  if ((exactSeedFooter || globalSeedFooter) !=
      headerObject.getBoolean("seed_only").value_or(false)) {
    incumbent.error = "global incumbent seed-only header/footer disagree";
    return false;
  }
  if (schema == "amoeba-exact-joint-task-scores-v1") {
    auto certificate = footerObject.getString("ranking_certificate");
    if (exactSeedFooter)
      certificate = std::nullopt;
    if (!exactSeedFooter &&
        (!certificate ||
        (*certificate != "proven-shape-and-schedule-branch-and-bound" &&
         *certificate != "exhaustive-shape-plus-bounded-exact-schedule" &&
         *certificate != "proven-shape-location-dispatch-order-branch-and-bound" &&
         *certificate != "proven-shape-ranking-at-single-heuristic-placement" &&
         *certificate != "proven-shape-space-under-production-scheduler"))) {
      incumbent.error = "global incumbent exact score has no accepted ranking certificate";
      return false;
    }
  } else if (!globalSeedFooter &&
             !headerObject.getBoolean("stage_certified").value_or(false)) {
    incumbent.error = "global incumbent ranking output is not stage-certified";
    return false;
  }
  if (exactSchema) {
    if (footerObject.getInteger("scored_count").value_or(-1) !=
        static_cast<int64_t>(rows.size())) {
      incumbent.error = "exact incumbent footer counts disagree with score rows";
      return false;
    }
    if (exactSeedFooter) {
      if (footerObject.getInteger("seed_feasible_count").value_or(-1) !=
              static_cast<int64_t>(rows.size()) ||
          footerObject.getInteger("top_k_requested").value_or(0) < 5) {
        incumbent.error =
            "seed-only incumbent footer does not account for all feasible rows";
        return false;
      }
    } else if (footerObject.getInteger("unsupported_shape_candidates").value_or(-1) !=
                   0 ||
               footerObject.getInteger("unmeasured_shape_candidates").value_or(-1) !=
                   0 ||
               footerObject.getInteger("selected_count").value_or(5) !=
                   static_cast<int64_t>(rows.size())) {
      // Exact score files use scored_count and the full shape coverage fields;
      // selected_count is optional in older artifacts.
      incumbent.error = "exact incumbent footer counts disagree with score rows";
      return false;
    }
  } else if (globalSeedFooter) {
    if (!jsonDecimalEquals(footerObject, "selected_count", rows.size()) ||
        !jsonDecimalEquals(footerObject, "seed_feasible_count", rows.size()) ||
        !headerObject.getBoolean("seed_only").value_or(false)) {
      incumbent.error = "global seed-only incumbent counts or markers disagree";
      return false;
    }
    std::string countError;
    auto headerCount = getDecimal(headerObject, "scored_candidate_count", countError);
    auto footerCount = getDecimal(footerObject, "scored_candidate_count", countError);
    if (!headerCount || !footerCount || *headerCount != *footerCount ||
        footerObject.getInteger("candidate_rows_considered").value_or(0) <
            static_cast<int64_t>(rows.size())) {
      incumbent.error = "global seed-only incumbent candidate counts disagree";
      return false;
    }
  } else {
    std::string countError;
    auto headerCount = getDecimal(headerObject, "scored_candidate_count", countError);
    auto footerCount = getDecimal(footerObject, "scored_candidate_count", countError);
    if (!headerCount || !footerCount || *headerCount != *footerCount) {
      incumbent.error = "global incumbent candidate counts disagree";
      return false;
    }
    if (!jsonDecimalEquals(footerObject, "selected_count", rows.size()) ||
        !jsonDecimalEquals(footerObject, "candidate_rows_considered",
                           rows.size()) ||
        !jsonDecimalIsZero(footerObject, "unmeasured_candidate_count") ||
        !jsonDecimalIsZero(footerObject, "unmeasured_graph_variant_count")) {
      incumbent.error = "global incumbent candidate counts are incomplete";
      return false;
    }
  }
  std::sort(rows.begin(), rows.end(), [](const IncumbentRow &lhs,
                                         const IncumbentRow &rhs) {
    if (lhs.cycles != rhs.cycles)
      return lhs.cycles < rhs.cycles;
    if (lhs.semanticGraphId != rhs.semanticGraphId)
      return lhs.semanticGraphId < rhs.semanticGraphId;
    int candidateOrder =
        compareProductionCandidateIds(lhs.candidateId, rhs.candidateId);
    if (candidateOrder != 0)
      return candidateOrder < 0;
    return lhs.graphVariantId < rhs.graphVariantId;
  });
  incumbent.stage = expectedStage.str();
  incumbent.function = expectedFunction.str();
  incumbent.sourceRepository = expectedRepository.str();
  incumbent.sourceCommit = expectedCommit.str();
  incumbent.architecturePath = expectedArchitecture.str();
  incumbent.architectureSchema = expectedArchitectureSchema.str();
  incumbent.costNamespace = expectedCostNamespace.str();
  incumbent.fifthCycles = rows[4].cycles;
  incumbent.candidateCount = static_cast<unsigned>(rows.size());
  incumbent.valid = true;
  return true;
}

inline constexpr llvm::StringLiteral kMlPredictorSource =
    "ml-ii-startup-predictor";

static bool hasTrustedMlPredictorSource(StringRef path, std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read cost catalogue " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "invalid cost catalogue JSON: " + llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  const llvm::json::Object *metadata =
      root ? root->getObject("predictor_metadata") : nullptr;
  std::optional<StringRef> source =
      metadata ? metadata->getString("predictor_source") : std::nullopt;
  if (!source || *source != kMlPredictorSource) {
    error = "exact joint ranking requires cost catalogue "
            "predictor_source=\"" + kMlPredictorSource.str() +
            "\"; measured mapper caches and legacy unmarked catalogues "
            "cannot be reported as ML predictions";
    return false;
  }
  return true;
}

static bool isFactoredSpace(StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return false;
  llvm::line_iterator lines(**buffer, true);
  if (lines.is_at_end())
    return false;
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
  if (!parsed)
    return false;
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return false;
  auto type = object->getString("record_type");
  auto schema = object->getString("schema");
  return type && schema && *type == "space" && *schema == kFactoredSpaceSchema;
}

// Parse and revalidate the exact factored shape descriptor emitted by
// EnumerateAnalyticalTaskCandidatesPass.  In particular, no row prefix is
// accepted as a substitute for the full Cartesian alphabet.
static bool parseFactoredSpace(
    StringRef path, func::FuncOp function, ArrayRef<TaskMetadata> metadata,
    StringRef expectedGraphId, const ::mlir::neura::Architecture &architecture,
    FactoredSpaceDescriptor &descriptor, std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read factored candidate space " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  llvm::line_iterator lines(**buffer, true);
  std::optional<llvm::json::Object> headerStorage;
  std::optional<llvm::json::Object> footerStorage;
  unsigned records = 0;
  for (; !lines.is_at_end(); ++lines) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
    if (!parsed) {
      error = "invalid factored JSONL: " + llvm::toString(parsed.takeError());
      return false;
    }
    const llvm::json::Object *object = parsed->getAsObject();
    if (!object) {
      error = "factored record is not an object";
      return false;
    }
    auto type = object->getString("record_type");
    if (!type) {
      error = "factored record has no record_type";
      return false;
    }
    if (*type == "space" && records == 0)
      headerStorage = std::move(*object);
    else if (*type == "footer" && records == 1)
      footerStorage = std::move(*object);
    else {
      error = "factored space must contain one header and one footer";
      return false;
    }
    ++records;
  }
  if (!headerStorage || !footerStorage || records != 2) {
    error = "factored space is missing header or footer";
    return false;
  }
  const llvm::json::Object &header = *headerStorage;
  const llvm::json::Object &footer = *footerStorage;
  auto schema = header.getString("schema");
  auto representation = header.getString("representation");
  auto exact = header.getBoolean("exact");
  auto functionName = header.getString("function");
  auto graphId = header.getString("graph_variant_id");
  auto scope = header.getString("search_scope");
  auto shapePolicy = header.getString("shape_policy");
  auto capacityPolicy = header.getString("spatial_capacity_policy");
  auto pruning = header.getString("shape_pruning_policy");
  auto maxChanged = header.getInteger("max_changed_tasks");
  auto maxCgras = header.getInteger("max_cgras_per_task");
  const llvm::json::Array *factors = header.getArray("factors");
  if (!schema || !representation || !exact || !functionName || !graphId ||
      !scope || !shapePolicy || !capacityPolicy || !pruning || !maxChanged ||
      !maxCgras || !factors) {
    error = "factored header is missing required fields";
    return false;
  }
  if (*schema != kFactoredSpaceSchema || *representation != "factored" ||
      !*exact || *functionName != function.getSymName() || expectedGraphId.empty() ||
      *graphId != expectedGraphId || *scope != kFactoredSearchScope ||
      *shapePolicy != kShapePolicy || *capacityPolicy != kSpatialCapacityPolicy ||
      *pruning != kShapePruningPolicyNone || *maxChanged != -1 ||
      *maxCgras < 1 || *maxCgras > 4 || architecture.getMultiCgraRows() != 4 ||
      architecture.getMultiCgraColumns() != 4) {
    error = "factored space does not match the complete 4x4/at-most-four-CGRA contract";
    return false;
  }
  if (factors->size() != metadata.size()) {
    error = "factored task count does not match current IR";
    return false;
  }

  SmallVector<RectShape> expected = enumerateStaticRectShapes(
      architecture.getMultiCgraRows(), architecture.getMultiCgraColumns(),
      architecture.getPerCgraRows(), architecture.getPerCgraColumns(),
      static_cast<unsigned>(*maxCgras));
  if (expected.empty()) {
    error = "architecture has no legal per-task shapes";
    return false;
  }
  descriptor.shapesByTask.clear();
  descriptor.shapesByTask.reserve(metadata.size());
  for (auto [taskIndex, value] : llvm::enumerate(*factors)) {
    const llvm::json::Object *factor = value.getAsObject();
    if (!factor) {
      error = "factored task factor is not an object";
      return false;
    }
    auto taskName = factor->getString("task");
    auto tripCount = factor->getInteger("trip_count");
    const llvm::json::Array *shapes = factor->getArray("shapes");
    if (!taskName || !tripCount || !shapes || *taskName != metadata[taskIndex].name ||
        *tripCount != metadata[taskIndex].tripCount ||
        shapes->size() != expected.size()) {
      error = "factored task factor does not match canonical metadata";
      return false;
    }
    SmallVector<RectShape> parsed;
    parsed.reserve(shapes->size());
    for (auto [shapeIndex, shapeValue] : llvm::enumerate(*shapes)) {
      const llvm::json::Object *shape = shapeValue.getAsObject();
      if (!shape) {
        error = "factored shape is not an object";
        return false;
      }
      auto rows = shape->getInteger("rows");
      auto cols = shape->getInteger("cols");
      auto mapperRows = shape->getInteger("mapper_tile_rows");
      auto mapperCols = shape->getInteger("mapper_tile_cols");
      auto cgraCount = shape->getInteger("cgra_count");
      if (!rows || !cols || !mapperRows || !mapperCols || !cgraCount ||
          *rows != expected[shapeIndex].rows || *cols != expected[shapeIndex].cols ||
          *mapperRows != expected[shapeIndex].mapperRows ||
          *mapperCols != expected[shapeIndex].mapperCols ||
          *cgraCount != expected[shapeIndex].cgraCount()) {
        error = "factored shape alphabet is not canonical";
        return false;
      }
      parsed.push_back(expected[shapeIndex]);
    }
    descriptor.shapesByTask.push_back(std::move(parsed));
  }
  auto count = getDecimal(header, "candidate_count", error);
  auto footerCount = getDecimal(footer, "candidate_count", error);
  auto footerSchema = footer.getString("schema");
  auto ranked = footer.getBoolean("ranked");
  auto status = footer.getString("status");
  if (!count || !footerCount || !footerSchema || !ranked || !status ||
      *footerSchema != kFactoredSpaceSchema || *ranked ||
      *status != "unranked-factored-space") {
    if (error.empty())
      error = "factored footer is not an exact unranked descriptor";
    return false;
  }
  ShapeCandidateSpaceCounts expectedCounts = countShapeCandidateSpace(
      descriptor.shapesByTask, kShapePruningPolicyNone, -1);
  if (expectedCounts.declared.empty() || *count != expectedCounts.declared ||
      *footerCount != expectedCounts.declared) {
    error = "factored candidate count does not match Cartesian product";
    return false;
  }
  descriptor.function = functionName->str();
  descriptor.candidateCount = *count;
  descriptor.maxCgrasPerTask = static_cast<unsigned>(*maxCgras);
  return true;
}

static std::optional<int64_t>
predictDuration(const TaskShapeCost &cost, int64_t tripCount, std::string &error) {
  if (!cost.supported)
    return std::nullopt;
  long double duration = static_cast<long double>(cost.startupCycles) +
                         static_cast<long double>(cost.predictedII) *
                             static_cast<long double>(tripCount - 1);
  if (!std::isfinite(duration) || duration <= 0.0L ||
      duration > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    error = "ML II/startup duration is outside the scheduler range";
    return std::nullopt;
  }
  long double rounded = std::ceil(duration);
  if (rounded > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    error = "rounded ML II/startup duration overflows int64";
    return std::nullopt;
  }
  return std::max<int64_t>(1, static_cast<int64_t>(rounded));
}

static llvm::json::Array makeTaskCosts(ArrayRef<TaskShapeChoice> choices,
                                        ArrayRef<TaskShapeCost> costs) {
  llvm::json::Array records;
  for (auto [index, choice] : llvm::enumerate(choices)) {
    llvm::json::Object record;
    record["task"] = choice.task;
    record["mapper_tile_rows"] = choice.shape.mapperRows;
    record["mapper_tile_cols"] = choice.shape.mapperCols;
    record["trip_count"] = choice.tripCount;
    record["predicted_ii"] = costs[index].predictedII;
    record["startup_cycles"] = costs[index].startupCycles;
    record["predicted_duration"] =
        static_cast<double>(costs[index].startupCycles) +
        costs[index].predictedII * static_cast<double>(choice.tripCount - 1);
    record["support_status"] = "supported";
    records.push_back(std::move(record));
  }
  return records;
}

static bool readExactCacheInput(StringRef path, std::string &contents,
                                std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read score cache input " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  contents = (*buffer)->getBuffer().str();
  return true;
}

static void appendExactCacheField(std::string &key, StringRef name,
                                  StringRef value) {
  // Length-prefixing makes the structural key unambiguous even when an input
  // file contains newlines, NULs, or arbitrary JSON whitespace.
  key += name.str();
  key += ":";
  key += std::to_string(value.size());
  key += "\n";
  key.append(value.data(), value.size());
  key += "\n";
}

static void appendExactCacheOption(std::string &key, StringRef name,
                                   int64_t value) {
  appendExactCacheField(key, name, std::to_string(value));
}

static void appendExactCacheOption(std::string &key, StringRef name,
                                   bool value) {
  appendExactCacheField(key, name, value ? "true" : "false");
}

static bool makeExactScoreCacheKey(
    func::FuncOp function, ArrayRef<TaskMetadata> metadata,
    ArrayRef<std::vector<unsigned>> predecessors,
    const FactoredSpaceDescriptor &descriptor, const TaskShapeCostCache &costs,
    StringRef candidateFile, StringRef costFile,
    StringRef dispatchPolicy, StringRef temporalModel, StringRef scheduleSpace,
    int64_t topK, int64_t maxShapeCandidates, int64_t maxExpandedNodes,
    int64_t maxMilliseconds, int64_t maxMakespan, int64_t stateMemoCapacity,
    bool horizonProven, bool includeProductionSchedulerContract,
    std::string &key, std::string &error) {
  // This is a cache format/version identity, not a cryptographic digest. The
  // complete key bytes are persisted and compared byte-for-byte by the cache.
  key = "orbit-exact-score-cache-v1\n";
  appendExactCacheField(key, "implementation",
                        "score-exact-joint-task-candidates-v1");
  appendExactCacheField(key, "function", function.getSymName());
  appendExactCacheField(key, "source-repository", costs.sourceRepository());
  appendExactCacheField(key, "source-commit", costs.sourceCommit());
  appendExactCacheField(key, "candidate-id-scheme", costs.candidateIdScheme());
  appendExactCacheField(key, "candidate-count", costs.candidateCountDecimal());
  appendExactCacheField(key, "architecture-schema", costs.architectureSchema());
  appendExactCacheField(key, "predictor-source", kMlPredictorSource);
  std::string contents;
  if (!readExactCacheInput(candidateFile, contents, error))
    return false;
  appendExactCacheField(key, "candidate-file-content", contents);
  if (!readExactCacheInput(costFile, contents, error))
    return false;
  appendExactCacheField(key, "cost-file-content", contents);
  if (!readExactCacheInput(costs.architecturePath(), contents, error))
    return false;
  appendExactCacheField(key, "architecture-file-content", contents);

  std::string functionText;
  llvm::raw_string_ostream functionStream(functionText);
  function.print(functionStream);
  functionStream.flush();
  appendExactCacheField(key, "function-ir", functionText);

  std::string taskText;
  for (auto [index, task] : llvm::enumerate(metadata)) {
    taskText += std::to_string(index);
    taskText += ":";
    taskText += task.name;
    taskText += ":";
    taskText += std::to_string(task.tripCount);
    taskText += ":";
    std::vector<unsigned> ordered = predecessors[index];
    llvm::sort(ordered);
    for (unsigned predecessor : ordered) {
      taskText += std::to_string(predecessor);
      taskText += ",";
    }
    taskText += "\n";
  }
  appendExactCacheField(key, "task-identifiers-and-edges", taskText);

  std::string shapeText;
  for (auto [taskIndex, shapes] : llvm::enumerate(descriptor.shapesByTask)) {
    shapeText += std::to_string(taskIndex);
    shapeText += ":";
    for (const RectShape &shape : shapes) {
      shapeText += std::to_string(shape.rows);
      shapeText += ",";
      shapeText += std::to_string(shape.cols);
      shapeText += ",";
      shapeText += std::to_string(shape.mapperRows);
      shapeText += ",";
      shapeText += std::to_string(shape.mapperCols);
      shapeText += ";";
    }
    shapeText += "\n";
  }
  appendExactCacheField(key, "factored-shape-descriptor", shapeText);
  appendExactCacheField(key, "factored-candidate-count",
                        descriptor.candidateCount);
  appendExactCacheOption(key, "max-cgras-per-task",
                         static_cast<int64_t>(descriptor.maxCgrasPerTask));
  appendExactCacheOption(key, "grid-rows", static_cast<int64_t>(4));
  appendExactCacheOption(key, "grid-cols", static_cast<int64_t>(4));
  appendExactCacheOption(key, "top-k", topK);
  appendExactCacheOption(key, "max-shape-candidates", maxShapeCandidates);
  appendExactCacheOption(key, "max-expanded-nodes", maxExpandedNodes);
  appendExactCacheOption(key, "max-milliseconds", maxMilliseconds);
  appendExactCacheOption(key, "max-makespan", maxMakespan);
  appendExactCacheOption(key, "state-memo-capacity", stateMemoCapacity);
  appendExactCacheField(key, "dispatch-policy", dispatchPolicy);
  appendExactCacheField(key, "temporal-model", temporalModel);
  appendExactCacheField(key, "schedule-space", scheduleSpace);
  appendExactCacheOption(key, "horizon-proven", horizonProven);
  appendExactCacheField(key, "communication-mode", "explicit");
  if (scheduleSpace == "production-scheduler" &&
      includeProductionSchedulerContract) {
    int64_t fixedPointIterations = 0;
    if (!getProductionFixedPointIterations(function, fixedPointIterations,
                                            error))
      return false;
    appendExactCacheField(key, "production-scheduler-contract",
                          kProductionSchedulerContract);
    appendExactCacheOption(key, "production-fixed-point-max-iterations",
                           fixedPointIterations);
  }
  return true;
}

// Build the immutable binding for a resumable shape DFS.  Search-resource
// limits are intentionally zeroed in the embedded score key: increasing a
// resume budget must continue the same semantic search, while changing any
// source/catalogue/architecture/scheduler/objective input must invalidate it.
// Every optional input that can affect traversal order is included both by
// path and, when present, by its complete file bytes.  This is a provenance
// witness, not a hash.
static bool makeShapeFrontierBinding(
    func::FuncOp function, ArrayRef<TaskMetadata> metadata,
    ArrayRef<std::vector<unsigned>> predecessors,
    const FactoredSpaceDescriptor &descriptor, const TaskShapeCostCache &costs,
    StringRef candidateFile, StringRef costFile, StringRef dispatchPolicy,
    StringRef temporalModel, StringRef scheduleSpace, int64_t topK,
    int64_t maxMakespan, int64_t stateMemoCapacity, bool horizonProven,
    StringRef sourceRepository, StringRef sourceCommit, StringRef graphVariantId,
    StringRef stage, StringRef warmStartPath, StringRef globalIncumbentPath,
    StringRef globalManifestPath, bool rankingPruning, bool graphBoundOnly,
    bool seedOnly, bool canonicalIncumbentSeed, int64_t shardCount,
    int64_t shardIndex,
    std::string &binding, std::string &error) {
  std::string scoreKey;
  if (!makeExactScoreCacheKey(
          function, metadata, predecessors, descriptor, costs, candidateFile,
          costFile, dispatchPolicy, temporalModel, scheduleSpace, topK,
          /*maxShapeCandidates=*/0, /*maxExpandedNodes=*/0,
          /*maxMilliseconds=*/0, maxMakespan, stateMemoCapacity, horizonProven,
          /*includeProductionSchedulerContract=*/true, scoreKey, error))
    return false;
  binding = "orbit-shape-frontier-binding-v1\n";
  appendExactCacheField(binding, "score-search-key", scoreKey);
  // The score key already binds the exact function operation.  A resumable
  // traversal also depends on module-level attributes and symbol context, so
  // retain the complete owning module text as a separate witness.  This is
  // deliberately full bytes rather than a digest: the loader compares the
  // persisted witness byte-for-byte before restoring a frontier.
  if (ModuleOp module = function->getParentOfType<ModuleOp>()) {
    std::string moduleText;
    llvm::raw_string_ostream moduleStream(moduleText);
    module.print(moduleStream);
    moduleStream.flush();
    appendExactCacheField(binding, "module-ir", moduleText);
  } else {
    error = "cannot bind shape frontier without an owning module";
    return false;
  }
  appendExactCacheField(binding, "frontier-implementation",
                        "score-exact-joint-task-candidates-frontier-v1");
  appendExactCacheField(binding, "source-repository", sourceRepository);
  appendExactCacheField(binding, "source-commit", sourceCommit);
  appendExactCacheField(binding, "graph-variant-id", graphVariantId);
  appendExactCacheField(binding, "stage", stage);
  appendExactCacheField(binding, "dispatch-policy", dispatchPolicy);
  appendExactCacheField(binding, "temporal-model", temporalModel);
  appendExactCacheField(binding, "schedule-space", scheduleSpace);
  appendExactCacheOption(binding, "top-k", topK);
  appendExactCacheOption(binding, "max-makespan", maxMakespan);
  appendExactCacheOption(binding, "state-memo-capacity", stateMemoCapacity);
  appendExactCacheOption(binding, "horizon-proven", horizonProven);
  appendExactCacheOption(binding, "ranking-pruning", rankingPruning);
  appendExactCacheOption(binding, "graph-bound-only", graphBoundOnly);
  appendExactCacheOption(binding, "seed-only", seedOnly);
  appendExactCacheOption(binding, "canonical-incumbent-seed",
                         canonicalIncumbentSeed);
  appendExactCacheField(
      binding, "canonical-seed-continuation-contract",
      "canonical-incumbent-seed-continuation-v1");
  appendExactCacheOption(binding, "shape-shard-count", shardCount);
  appendExactCacheOption(binding, "shape-shard-index", shardIndex);
  appendExactCacheField(binding, "candidate-file-path", candidateFile);
  appendExactCacheField(binding, "cost-file-path", costFile);
  appendExactCacheField(binding, "architecture-file-path",
                        costs.architecturePath());
  appendExactCacheField(binding, "source-graph-id", costs.sourceGraphId());
  appendExactCacheField(binding, "cost-namespace", costs.nameSpace());
  appendExactCacheField(binding, "candidate-id-scheme",
                        costs.candidateIdScheme());

  auto appendOptionalFile = [&](StringRef name, StringRef path) {
    appendExactCacheField(binding, (name + "-path").str(), path);
    if (path.empty()) {
      appendExactCacheField(binding, (name + "-content").str(), "");
      return true;
    }
    std::string contents;
    if (!readExactCacheInput(path, contents, error))
      return false;
    appendExactCacheField(binding, (name + "-content").str(), contents);
    return true;
  };
  if (!appendOptionalFile("warm-start", warmStartPath) ||
      !appendOptionalFile("global-incumbent", globalIncumbentPath) ||
      !appendOptionalFile("global-stage-manifest", globalManifestPath))
    return false;
  return true;
}

static bool writeShapeFrontierBinding(StringRef path, StringRef binding,
                                      std::string &error) {
  return writeAtomically(
      path,
      [&](llvm::raw_ostream &stream) {
        stream << binding;
        return true;
      },
      error);
}

struct PeerScoreCommand {
  std::string modulePath;
  std::string architecturePath;
  std::string candidatePath;
  std::string costPath;
  std::string scorePath;
  std::map<std::string, std::string> options;
  std::optional<double> startedUnix;
  bool hasExitCode = false;
  int64_t exitCode = 0;
};

enum class PeerProbeStatus { Imported, Pending, Rejected };

static bool readJsonObject(StringRef path, llvm::json::Object &object,
                           std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read peer JSON " + path.str();
    return false;
  }
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "peer JSON is invalid: " + path.str();
    return false;
  }
  const llvm::json::Object *parsedObject = parsed->getAsObject();
  if (!parsedObject) {
    error = "peer JSON is not an object: " + path.str();
    return false;
  }
  object = *parsedObject;
  return true;
}

static bool parsePeerScoreOptions(StringRef text,
                                  std::map<std::string, std::string> &options,
                                  std::string &error) {
  SmallVector<StringRef> fields;
  text.split(fields, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef field : fields) {
    size_t equals = field.find('=');
    if (equals == StringRef::npos || equals == 0) {
      error = "peer score option has no key/value: " + field.str();
      return false;
    }
    options[field.take_front(equals).str()] =
        field.drop_front(equals + 1).str();
  }
  return true;
}

static bool readPeerScoreCommand(StringRef path, PeerScoreCommand &command,
                                 std::string &error) {
  llvm::json::Object root;
  if (!readJsonObject(path, root, error))
    return false;
  const llvm::json::Array *argv = root.getArray("argv");
  if (!argv || argv->size() < 3) {
    error = "peer score command has no complete argv";
    return false;
  }
  if (std::optional<int64_t> exitCode = root.getInteger("exit_code")) {
    command.hasExitCode = true;
    command.exitCode = *exitCode;
  }
  command.startedUnix = root.getNumber("started_unix");
  for (auto [index, value] : llvm::enumerate(*argv)) {
    std::optional<StringRef> text = value.getAsString();
    if (!text)
      continue;
    if (index == 1)
      command.modulePath = text->str();
    if (text->starts_with("--architecture-spec="))
      command.architecturePath =
          text->drop_front(strlen("--architecture-spec=")).str();
    if (text->starts_with("--score-exact-joint-task-candidates=")) {
      StringRef optionsText =
          text->drop_front(strlen("--score-exact-joint-task-candidates="));
      if (!parsePeerScoreOptions(optionsText, command.options, error))
        return false;
    }
  }
  auto getOption = [&](StringRef name, std::string &value) {
    auto it = command.options.find(name.str());
    if (it == command.options.end())
      return false;
    value = it->second;
    return true;
  };
  if (command.modulePath.empty() || command.architecturePath.empty() ||
      !getOption("candidates", command.candidatePath) ||
      !getOption("cost-file", command.costPath) ||
      !getOption("output", command.scorePath)) {
    error =
        "peer score command is missing module, architecture, input, or output binding";
    return false;
  }
  return true;
}

static std::string resolvePeerVariantDirectory(StringRef root,
                                               StringRef graphVariantId) {
  SmallVector<std::string> candidates;
  candidates.push_back((Twine(root) + "/variants/" + graphVariantId).str());
  candidates.push_back((Twine(root) + "/" + graphVariantId).str());
  candidates.push_back(root.str());
  for (const std::string &candidate : candidates) {
    if (llvm::sys::fs::exists(candidate + "/score.command.json") ||
        llvm::sys::fs::exists(candidate + "/scores.jsonl"))
      return candidate;
  }
  return {};
}

static bool sameFileBytes(StringRef lhs, StringRef rhs, std::string &error) {
  std::string left, right;
  if (!readExactCacheInput(lhs, left, error) ||
      !readExactCacheInput(rhs, right, error))
    return false;
  if (left != right) {
    error = "peer input bytes differ: " + lhs.str() + " versus " + rhs.str();
    return false;
  }
  return true;
}

static bool peerFunctionMatches(func::FuncOp current, StringRef peerModule,
                                std::string &error) {
  MLIRContext *context = current.getContext();
  OwningOpRef<ModuleOp> parsed =
      parseSourceFile<ModuleOp>(peerModule, ParserConfig(context));
  if (!parsed) {
    error = "cannot parse peer module: " + peerModule.str();
    return false;
  }
  FailureOr<func::FuncOp> peerFunction =
      selectTaskFunction(*parsed, current.getSymName(), error);
  if (failed(peerFunction))
    return false;
  std::string currentText, peerText;
  llvm::raw_string_ostream currentStream(currentText), peerStream(peerText);
  current.print(currentStream);
  (*peerFunction).print(peerStream);
  currentStream.flush();
  peerStream.flush();
  if (currentText != peerText) {
    error = "peer module function IR differs from current function";
    return false;
  }
  return true;
}

static bool peerScoreHeaderFooter(StringRef score, llvm::json::Object &header,
                                  llvm::json::Object &footer,
                                  std::string &error) {
  bool sawHeader = false, sawFooter = false;
  llvm::MemoryBufferRef buffer(score, "peer-score");
  llvm::line_iterator lines(buffer, true);
  for (; !lines.is_at_end(); ++lines) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
    if (!parsed) {
      error = "peer score contains invalid JSON";
      return false;
    }
    const llvm::json::Object *object = parsed->getAsObject();
    if (!object) {
      error = "peer score contains a non-object record";
      return false;
    }
    std::optional<StringRef> type = object->getString("record_type");
    if (!type) {
      error = "peer score record has no record_type";
      return false;
    }
    if (*type == "header") {
      if (sawHeader || sawFooter) {
        error = "peer score has duplicate or late header";
        return false;
      }
      header = *object;
      sawHeader = true;
    } else if (*type == "footer") {
      if (!sawHeader || sawFooter) {
        error = "peer score has invalid footer placement";
        return false;
      }
      footer = *object;
      sawFooter = true;
    } else if (!sawHeader || sawFooter || *type != "score") {
      error = "peer score has an invalid record sequence";
      return false;
    }
  }
  if (!sawHeader || !sawFooter) {
    error = "peer score has no complete header/footer";
    return false;
  }
  return true;
}

static bool requirePeerString(const llvm::json::Object &object, StringRef name,
                              StringRef expected, std::string &error) {
  std::optional<StringRef> value = object.getString(name);
  if (!value || *value != expected) {
    error = "peer score binding differs at " + name.str();
    return false;
  }
  return true;
}

static bool requirePeerInteger(const llvm::json::Object &object, StringRef name,
                               int64_t expected, std::string &error) {
  std::optional<int64_t> value = object.getInteger(name);
  if (!value || *value != expected) {
    error = "peer score binding differs at " + name.str();
    return false;
  }
  return true;
}

static bool requirePeerCount(const llvm::json::Object &object, StringRef name,
                             StringRef expected, std::string &error) {
  if (object.getString(name))
    return requirePeerString(object, name, expected, error);
  if (std::optional<int64_t> value = object.getInteger(name)) {
    int64_t parsed = 0;
    if (!expected.getAsInteger(10, parsed) && *value == parsed)
      return true;
  }
  error = "peer score binding differs at " + name.str();
  return false;
}

static bool requirePeerBool(const llvm::json::Object &object, StringRef name,
                            bool expected, std::string &error) {
  std::optional<bool> value = object.getBoolean(name);
  if (!value || *value != expected) {
    error = "peer score binding differs at " + name.str();
    return false;
  }
  return true;
}

static bool samePeerOption(const PeerScoreCommand &command, StringRef name,
                           StringRef expected, std::string &error) {
  auto it = command.options.find(name.str());
  if (it == command.options.end()) {
    error = "peer score command omits semantic option " + name.str();
    return false;
  }
  if (it->second != expected) {
    error = "peer score command differs at " + name.str();
    return false;
  }
  return true;
}

static bool peerInputWasStableBeforeStart(StringRef path, double startedUnix,
                                          std::string &error) {
  llvm::sys::fs::file_status status;
  if (std::error_code ec = llvm::sys::fs::status(path, status)) {
    error = "cannot stat peer input " + path.str() + ": " + ec.message();
    return false;
  }
  using namespace std::chrono;
  const auto startedNanos = duration_cast<nanoseconds>(
      duration<double>(startedUnix));
  const llvm::sys::TimePoint<> startedTime(startedNanos);
  if (status.getLastModificationTime() > startedTime) {
    error = "peer input was modified after score.command started: " +
            path.str();
    return false;
  }
  return true;
}

static bool readCostCatalogBinding(StringRef path, CostCatalogBinding &binding,
                                   std::string &error) {
  llvm::json::Object root;
  if (!readJsonObject(path, root, error))
    return false;
  const llvm::json::Object *metadata = root.getObject("predictor_metadata");
  if (!metadata) {
    error = "cost catalogue has no predictor_metadata: " + path.str();
    return false;
  }
  auto requiredRoot = [&](StringRef key, std::string &destination) {
    auto value = root.getString(key);
    if (!value || value->empty()) {
      error = "cost catalogue is missing " + key.str() + ": " + path.str();
      return false;
    }
    destination = value->str();
    return true;
  };
  auto requiredMetadata = [&](StringRef key, std::string &destination) {
    auto value = metadata->getString(key);
    if (!value || value->empty()) {
      error = "cost catalogue metadata is missing " + key.str() + ": " +
              path.str();
      return false;
    }
    destination = value->str();
    return true;
  };
  return requiredRoot("namespace", binding.nameSpace) &&
         requiredMetadata("model", binding.model) &&
         requiredMetadata("model_schema", binding.modelSchema) &&
         requiredMetadata("feature_contract_id", binding.featureContractId) &&
         requiredMetadata("feature_extractor", binding.featureExtractor) &&
         requiredMetadata("predictor_source", binding.predictorSource) &&
         requiredMetadata("provenance_schema", binding.provenanceSchema) &&
         requiredMetadata("architecture_schema", binding.architectureSchema) &&
         requiredMetadata("architecture_path", binding.architecturePath) &&
         requiredMetadata("source_repository", binding.sourceRepository) &&
         requiredMetadata("source_commit", binding.sourceCommit) &&
         requiredMetadata("source_graph_id", binding.sourceGraphId);
}

static bool compareCostCatalogBindings(const CostCatalogBinding &incumbent,
                                       const CostCatalogBinding &current,
                                       StringRef currentArchitecture,
                                       std::string &error) {
  auto compare = [&](StringRef label, StringRef lhs, StringRef rhs) {
    if (lhs != rhs) {
      error = "incumbent/current cost catalogue mismatch at " + label.str();
      return false;
    }
    return true;
  };
  if (!compare("namespace", incumbent.nameSpace, current.nameSpace) ||
      !compare("model", incumbent.model, current.model) ||
      !compare("model_schema", incumbent.modelSchema, current.modelSchema) ||
      !compare("feature_contract_id", incumbent.featureContractId,
               current.featureContractId) ||
      !compare("feature_extractor", incumbent.featureExtractor,
               current.featureExtractor) ||
      !compare("predictor_source", incumbent.predictorSource,
               current.predictorSource) ||
      !compare("provenance_schema", incumbent.provenanceSchema,
               current.provenanceSchema) ||
      !compare("architecture_schema", incumbent.architectureSchema,
               current.architectureSchema))
    return false;
  if (incumbent.predictorSource != kMlPredictorSource ||
      incumbent.provenanceSchema != kCostProvenanceSchema) {
    error = "incumbent cost catalogue provenance is not source-owned";
    return false;
  }
  if (!samePath(incumbent.architecturePath, currentArchitecture) ||
      !samePath(current.architecturePath, currentArchitecture)) {
    error = "cost catalogue architecture path is not the current architecture";
    return false;
  }
  if (incumbent.sourceRepository != current.sourceRepository ||
      incumbent.sourceCommit != current.sourceCommit) {
    error = "incumbent/current cost catalogue source provenance differs";
    return false;
  }
  return true;
}

static bool parseCandidateOrdinal(StringRef candidateId, StringRef label,
                                  StringRef candidateCount,
                                  std::string &error) {
  constexpr StringLiteral prefix = "candidate-";
  if (!candidateId.starts_with(prefix)) {
    error = label.str() + " has no canonical candidate ordinal";
    return false;
  }
  StringRef suffix = candidateId.drop_front(prefix.size());
  StringRef normalized = suffix.ltrim('0');
  if (normalized.empty())
    normalized = "0";
  if (suffix.empty() || llvm::any_of(suffix, [](char value) {
        return value < '0' || value > '9';
      }) ||
      compareDecimal(normalized, candidateCount) >= 0) {
    error = label.str() + " is outside the bound candidate descriptor";
    return false;
  }
  return true;
}

struct WarmStartShapeHint {
  llvm::APInt ordinal;
  SmallVector<size_t> shapeIndices;
  std::string canonicalOrdinal;
};

// A warm-start file contributes only candidate ordinals.  Its old scores,
// schedules, timing fields, and scheduler contract are deliberately ignored.
// Decode each ordinal against the current factored alphabet so a row can never
// smuggle in a partial or foreign tuple.
static bool loadWarmStartShapeHints(
    StringRef path, const FactoredSpaceDescriptor &descriptor,
    unsigned counterWidth, std::vector<WarmStartShapeHint> &hints,
    std::string &error) {
  hints.clear();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read warm-start shape hints " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }

  llvm::MemoryBufferRef input(**buffer);
  llvm::line_iterator lines(input, true);
  std::set<std::string> seenOrdinals;
  bool sawScore = false;
  bool sawHeader = false;
  bool sawFooter = false;
  uint64_t lineNumber = 0;
  auto fail = [&](StringRef reason) {
    error = path.str() + ": line " + std::to_string(lineNumber) + ": " +
            reason.str();
    return false;
  };

  for (; !lines.is_at_end(); ++lines) {
    ++lineNumber;
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
    if (!parsed)
      return fail("invalid JSON: " + llvm::toString(parsed.takeError()));
    const llvm::json::Object *object = parsed->getAsObject();
    if (!object)
      return fail("record is not a JSON object");
    std::optional<StringRef> type = object->getString("record_type");
    if (!type)
      return fail("record has no record_type");
    if (*type == "header") {
      if (sawHeader || sawFooter || sawScore)
        return fail("header is out of order or duplicated");
      sawHeader = true;
      continue;
    }
    if (*type == "footer") {
      if (sawFooter)
        return fail("footer is duplicated");
      sawFooter = true;
      continue;
    }
    if (*type != "score")
      return fail("unsupported record_type; expected header, score, or footer");
    if (sawFooter)
      return fail("score appears after footer");
    sawScore = true;

    std::optional<StringRef> candidateId =
        object->getString("shape_candidate_id");
    if (!candidateId || candidateId->empty())
      return fail("score row has no shape_candidate_id");
    if (!parseCandidateOrdinal(*candidateId, "shape_candidate_id",
                               descriptor.candidateCount, error)) {
      error = path.str() + ": line " + std::to_string(lineNumber) + ": " +
              error;
      return false;
    }
    constexpr StringLiteral prefix = "candidate-";
    StringRef suffix = candidateId->drop_front(prefix.size());
    StringRef normalized = suffix.ltrim('0');
    if (normalized.empty())
      normalized = "0";
    llvm::APInt ordinal(counterWidth, normalized, 10);
    std::string canonicalOrdinal = exactShapeIndexString(ordinal);
    if (!seenOrdinals.insert(canonicalOrdinal).second)
      return fail("duplicate shape_candidate_id after decimal normalization");

    llvm::APInt remaining = ordinal;
    SmallVector<size_t> shapeIndices(descriptor.shapesByTask.size(), 0);
    for (size_t task = descriptor.shapesByTask.size(); task > 0; --task) {
      const size_t taskIndex = task - 1;
      const size_t radix = descriptor.shapesByTask[taskIndex].size();
      if (radix == 0)
        return fail("current factored descriptor has an empty task alphabet");
      const llvm::APInt radixValue(counterWidth,
                                   static_cast<uint64_t>(radix));
      const llvm::APInt remainder = remaining.urem(radixValue);
      if (remainder.getActiveBits() > 64)
        return fail("decoded shape index exceeds host index width");
      const uint64_t shapeIndex = remainder.getZExtValue();
      if (shapeIndex >= radix)
        return fail("decoded shape index is outside the current alphabet");
      shapeIndices[taskIndex] = static_cast<size_t>(shapeIndex);
      remaining = remaining.udiv(radixValue);
    }
    if (!remaining.isZero())
      return fail("shape_candidate_id does not decode as a full current tuple");
    hints.push_back({std::move(ordinal), std::move(shapeIndices),
                     std::move(canonicalOrdinal)});
  }
  if (!sawScore)
    return fail("warm-start file contains no score rows");
  return true;
}

static bool nearlyEqualCost(double lhs, double rhs) {
  const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
  return std::fabs(lhs - rhs) <= 1.0e-9 * scale;
}

static bool validateScoreRecordAgainstCatalog(
    const llvm::json::Object &record, StringRef label,
    const FactoredSpaceDescriptor &descriptor,
    ArrayRef<TaskMetadata> metadata, TaskShapeCostCache &costs,
    std::string &error) {
  auto shapeCandidateId = record.getString("shape_candidate_id");
  if (!shapeCandidateId ||
      !parseCandidateOrdinal(*shapeCandidateId, label, descriptor.candidateCount,
                             error))
    return false;
  const llvm::json::Array *taskCosts = record.getArray("task_costs");
  const llvm::json::Array *schedule = record.getArray("task_schedule");
  if (!taskCosts || taskCosts->size() != metadata.size() || !schedule ||
      schedule->empty()) {
    error = label.str() + " does not contain one cost row per peer task";
    return false;
  }
  std::map<std::string, unsigned> taskIndices;
  for (auto [index, task] : llvm::enumerate(metadata))
    if (!taskIndices.emplace(task.name, static_cast<unsigned>(index)).second) {
      error = "peer task metadata contains duplicate task IDs";
      return false;
    }
  std::set<std::string> seenTasks;
  for (const llvm::json::Value &value : *taskCosts) {
    const llvm::json::Object *taskCost = value.getAsObject();
    if (!taskCost) {
      error = label.str() + " has a non-object task cost";
      return false;
    }
    auto taskName = taskCost->getString("task");
    auto mapperRows = taskCost->getInteger("mapper_tile_rows");
    auto mapperCols = taskCost->getInteger("mapper_tile_cols");
    auto tripCount = taskCost->getInteger("trip_count");
    auto predictedII = taskCost->getNumber("predicted_ii");
    auto startupCycles = taskCost->getNumber("startup_cycles");
    auto predictedDuration = taskCost->getNumber("predicted_duration");
    auto supportStatus = taskCost->getString("support_status");
    if (!taskName || !mapperRows || !mapperCols || !tripCount ||
        !predictedII || !startupCycles || !predictedDuration ||
        !supportStatus || *mapperRows <= 0 || *mapperCols <= 0 ||
        *tripCount <= 0 || *supportStatus != "supported") {
      error = label.str() + " has an incomplete task cost row";
      return false;
    }
    auto taskIt = taskIndices.find(taskName->str());
    if (taskIt == taskIndices.end() ||
        !seenTasks.insert(taskName->str()).second) {
      error = label.str() + " task costs do not exactly cover peer tasks";
      return false;
    }
    const unsigned taskIndex = taskIt->second;
    if (*tripCount != metadata[taskIndex].tripCount) {
      error = label.str() + " task cost trip count disagrees with peer IR";
      return false;
    }
    std::optional<RectShape> matchingShape;
    for (const RectShape &shape : descriptor.shapesByTask[taskIndex])
      if (shape.mapperRows == *mapperRows && shape.mapperCols == *mapperCols) {
        matchingShape = shape;
        break;
      }
    if (!matchingShape) {
      error = label.str() + " task cost shape is outside the peer descriptor";
      return false;
    }
    TaskShapeChoice choice{taskName->str(), *tripCount, *matchingShape};
    const TaskShapeCost *actual = costs.get(choice, error);
    if (!actual || !actual->supported ||
        !nearlyEqualCost(actual->predictedII, *predictedII) ||
        !nearlyEqualCost(actual->startupCycles, *startupCycles)) {
      if (error.empty())
        error = label.str() + " task cost row differs from the peer catalogue";
      return false;
    }
    const double expectedDuration =
        actual->startupCycles +
        actual->predictedII * static_cast<double>(*tripCount - 1);
    if (!nearlyEqualCost(expectedDuration, *predictedDuration)) {
      error = label.str() + " task cost duration disagrees with II/startup";
      return false;
    }
  }
  if (seenTasks.size() != metadata.size()) {
    error = label.str() + " task costs omit a peer task";
    return false;
  }
  for (const llvm::json::Value &value : *schedule) {
    const llvm::json::Object *placement = value.getAsObject();
    if (!placement || !placement->getString("task") ||
        placement->getInteger("row").value_or(-1) < 0 ||
        placement->getInteger("col").value_or(-1) < 0 ||
        placement->getInteger("rows").value_or(0) <= 0 ||
        placement->getInteger("cols").value_or(0) <= 0 ||
        placement->getInteger("row").value_or(0) +
                placement->getInteger("rows").value_or(0) >
            4 ||
        placement->getInteger("col").value_or(0) +
                placement->getInteger("cols").value_or(0) >
            4) {
      error = label.str() + " contains an out-of-grid task placement";
      return false;
    }
  }
  return true;
}

static bool parseFrontierUnsigned(const llvm::json::Object &object,
                                  StringRef key, uint64_t &value,
                                  std::string &error) {
  std::string decimalError;
  std::optional<std::string> decimal = getDecimal(object, key, decimalError);
  if (!decimal) {
    error = "frontier field " + key.str() + " is invalid: " + decimalError;
    return false;
  }
  if (llvm::StringRef(*decimal).getAsInteger(10, value)) {
    error = "frontier field " + key.str() + " exceeds uint64";
    return false;
  }
  return true;
}

// Parse only an incomplete snapshot emitted by this pass.  In particular, a
// JSON file that merely contains top-five rows is not a frontier and is
// rejected before any row can influence the new search.
static bool loadShapeFrontierSnapshot(
    StringRef path, StringRef expectedBinding,
    const FactoredSpaceDescriptor &descriptor, ArrayRef<TaskMetadata> metadata,
    TaskShapeCostCache &costs, StringRef expectedFunction,
    StringRef expectedGraphVariantId, StringRef expectedSourceRepository,
    StringRef expectedSourceCommit, StringRef expectedDispatchPolicy,
    StringRef expectedTemporalModel, StringRef expectedScheduleSpace,
    int64_t expectedTopK, bool expectedCanonicalIncumbentSeed,
    ShapeFrontierSnapshot &snapshot, std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read shape frontier " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "invalid shape frontier JSON: " + llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    error = "shape frontier root is not an object";
    return false;
  }
  if (root->getString("record_type") !=
          std::optional<StringRef>("shape-frontier") ||
      root->getString("schema") !=
          std::optional<StringRef>("amoeba-shape-frontier-v1") ||
      root->getString("status") !=
          std::optional<StringRef>("incomplete") ||
      root->getString("phase") != std::optional<StringRef>("shape-search") ||
      root->getBoolean("resume_eligible") != std::optional<bool>(true)) {
    error = "shape frontier is not an eligible incomplete source-owned snapshot";
    return false;
  }
  auto bindingPath = root->getString("binding_witness_path");
  if (!bindingPath || bindingPath->empty()) {
    error = "shape frontier has no binding witness path";
    return false;
  }
  if (*bindingPath != path.str() + ".binding") {
    error = "shape frontier binding witness path is not adjacent to snapshot";
    return false;
  }
  std::string persistedBinding;
  if (!readExactCacheInput(*bindingPath, persistedBinding, error))
    return false;
  if (persistedBinding != expectedBinding) {
    error = "shape frontier binding witness differs byte-for-byte";
    return false;
  }
  auto requireStaticString = [&](StringRef key, StringRef expected) {
    auto value = root->getString(key);
    return value && *value == expected;
  };
  auto topK = root->getInteger("top_k");
  if (!topK || *topK != expectedTopK ||
      !requireStaticString("function", expectedFunction) ||
      !requireStaticString("graph_variant_id", expectedGraphVariantId) ||
      !requireStaticString("source_repository", expectedSourceRepository) ||
      !requireStaticString("source_commit", expectedSourceCommit) ||
      !requireStaticString("dispatch_policy", expectedDispatchPolicy) ||
      !requireStaticString("temporal_model", expectedTemporalModel) ||
      !requireStaticString("schedule_space", expectedScheduleSpace) ||
      !requireStaticString("production_scheduler_contract",
                           kProductionSchedulerContract)) {
    error = "shape frontier static option fields differ from current invocation";
    return false;
  }
  auto canonicalSeedEnabled =
      root->getBoolean("canonical_incumbent_seed_enabled");
  auto canonicalSeedContinuation =
      root->getString("canonical_seed_continuation_state");
  if (!canonicalSeedEnabled ||
      *canonicalSeedEnabled != expectedCanonicalIncumbentSeed ||
      !canonicalSeedContinuation ||
      (*canonicalSeedContinuation !=
           (expectedCanonicalIncumbentSeed ? "complete" : "disabled"))) {
    error =
        "shape frontier canonical incumbent seed binding/continuation state "
        "does not match current options";
    return false;
  }
  uint64_t canonicalSeedCalls = 0, canonicalSeedAccepted = 0,
           canonicalSeedNextTask = 0, canonicalSeedNextSmaller = 0;
  if (!parseFrontierUnsigned(*root, "canonical_seed_scheduler_calls",
                             canonicalSeedCalls, error) ||
      !parseFrontierUnsigned(*root, "canonical_seed_accepted",
                             canonicalSeedAccepted, error) ||
      !parseFrontierUnsigned(*root, "canonical_seed_next_task",
                             canonicalSeedNextTask, error) ||
      !parseFrontierUnsigned(*root, "canonical_seed_next_smaller",
                             canonicalSeedNextSmaller, error))
    return false;
  if ((!expectedCanonicalIncumbentSeed &&
       (canonicalSeedCalls != 0 || canonicalSeedAccepted != 0 ||
        canonicalSeedNextTask != 0 || canonicalSeedNextSmaller != 0)) ||
      (expectedCanonicalIncumbentSeed &&
       (canonicalSeedAccepted > canonicalSeedCalls ||
        canonicalSeedNextTask != metadata.size() ||
        canonicalSeedNextSmaller != 0))) {
    error = "shape frontier canonical seed continuation counters are invalid";
    return false;
  }
  auto candidateCount = root->getString("candidate_count");
  if (!candidateCount || *candidateCount != descriptor.candidateCount) {
    error = "shape frontier candidate count differs from current descriptor";
    return false;
  }
  auto implementation = root->getString("implementation_contract");
  if (!implementation ||
      *implementation !=
          "score-exact-joint-task-candidates-frontier-v1") {
    error = "shape frontier implementation contract is unsupported";
    return false;
  }
  auto reason = root->getString("reason");
  if (!reason || (*reason != "periodic" && *reason != "before-child" &&
                  *reason != "max-shape-candidates" &&
                  *reason != "max-expanded-nodes" &&
                  *reason != "max-milliseconds")) {
    error = "shape frontier has an unsupported checkpoint reason";
    return false;
  }
  auto decisionOrder = root->getArray("decision_order");
  if (!decisionOrder || decisionOrder->size() != metadata.size()) {
    error = "shape frontier decision order is incomplete";
    return false;
  }
  snapshot.decisionOrder.reserve(decisionOrder->size());
  std::set<unsigned> seenTasks;
  for (const llvm::json::Value &value : *decisionOrder) {
    std::optional<int64_t> task = value.getAsInteger();
    if (!task || *task < 0 ||
        static_cast<size_t>(*task) >= metadata.size() ||
        !seenTasks.insert(static_cast<unsigned>(*task)).second) {
      error = "shape frontier decision order is not a permutation";
      return false;
    }
    snapshot.decisionOrder.push_back(static_cast<unsigned>(*task));
  }

  auto frames = root->getArray("frames");
  if (!frames || frames->empty() || frames->size() > metadata.size() + 1) {
    error = "shape frontier has an invalid DFS frame stack";
    return false;
  }
  snapshot.frames.reserve(frames->size());
  for (size_t position = 0; position < frames->size(); ++position) {
    const llvm::json::Object *frame = (*frames)[position].getAsObject();
    if (!frame) {
      error = "shape frontier frame is not an object";
      return false;
    }
    uint64_t depth = 0, task = 0, chosen = 0, next = 0;
    if (!parseFrontierUnsigned(*frame, "depth", depth, error) ||
        !parseFrontierUnsigned(*frame, "task", task, error) ||
        !parseFrontierUnsigned(*frame, "chosen_index", chosen, error) ||
        !parseFrontierUnsigned(*frame, "next_choice", next, error))
      return false;
    auto ordinal = frame->getString("ordinal");
    if (!ordinal || ordinal->empty() ||
        llvm::any_of(*ordinal, [](char c) { return c < '0' || c > '9'; })) {
      error = "shape frontier frame ordinal is not decimal";
      return false;
    }
    if (depth != position || task >= metadata.size() + 1 ||
        (depth < metadata.size() && task != snapshot.decisionOrder[depth]) ||
        (depth == metadata.size() && task != metadata.size())) {
      error = "shape frontier frame depth/task does not match decision order";
      return false;
    }
    if (chosen > std::numeric_limits<size_t>::max() ||
        next > std::numeric_limits<uint64_t>::max()) {
      error = "shape frontier frame index overflows host width";
      return false;
    }
    snapshot.frames.push_back({static_cast<size_t>(depth),
                               static_cast<unsigned>(task),
                               static_cast<size_t>(chosen), next,
                               ordinal->str()});
  }

  auto ranked = root->getArray("ranked");
  if (!ranked) {
    error = "shape frontier has no incumbent rows";
    return false;
  }
  std::set<std::string> seenKeys;
  for (const llvm::json::Value &value : *ranked) {
    const llvm::json::Object *row = value.getAsObject();
    if (!row) {
      error = "shape frontier ranked row is not an object";
      return false;
    }
    auto key = row->getString("key");
    auto shapeOrdinal = row->getString("shape_ordinal");
    auto candidateId = row->getString("candidate_id");
    std::string cyclesError;
    std::optional<std::string> scheduleOrdinal =
        getDecimal(*row, "schedule_ordinal", cyclesError);
    std::optional<int64_t> cycles = row->getInteger("cycles");
    const llvm::json::Object *record = row->getObject("record");
    if (!key || !shapeOrdinal || !candidateId || !scheduleOrdinal || !cycles ||
        *cycles < 0 || !record || !seenKeys.insert(key->str()).second) {
      error = "shape frontier ranked row is incomplete or duplicated";
      return false;
    }
    if (!parseCandidateOrdinal("candidate-" + shapeOrdinal->str(),
                               "shape frontier shape ordinal",
                               descriptor.candidateCount, error))
      return false;
    uint64_t schedule = 0;
    if (llvm::StringRef(*scheduleOrdinal).getAsInteger(10, schedule)) {
      error = "shape frontier schedule ordinal exceeds uint64";
      return false;
    }
    auto recordShape = record->getString("shape_candidate_id");
    auto recordCandidate = record->getString("candidate_id");
    auto recordCycles = record->getInteger("predicted_whole_program_cycles");
    if (!recordShape || !recordCandidate || !recordCycles ||
        *recordCycles != *cycles ||
        *recordShape != "candidate-" + shapeOrdinal->str() ||
        *recordCandidate != *candidateId ||
        *candidateId != "shape-" + shapeOrdinal->str() + "/schedule-" +
                             *scheduleOrdinal) {
      error = "shape frontier ranked row identity/cycle fields disagree";
      return false;
    }
    llvm::json::Object recordCopy = *record;
    if (!validateScoreRecordAgainstCatalog(
            recordCopy, "shape frontier ranked row", descriptor, metadata,
            costs, error))
      return false;
    snapshot.ranked.push_back({key->str(), shapeOrdinal->str(), schedule,
                               *cycles, std::move(recordCopy)});
  }
  uint64_t serial = 0, expanded = 0, expandedShape = 0, schedulerCalls = 0,
           validShapes = 0, unsupportedShapes = 0;
  if (!parseFrontierUnsigned(*root, "schedule_serial", serial, error) ||
      !parseFrontierUnsigned(*root, "expanded_nodes", expanded, error) ||
      !parseFrontierUnsigned(*root, "expanded_shape_nodes", expandedShape,
                             error) ||
      !parseFrontierUnsigned(*root, "production_scheduler_calls",
                             schedulerCalls, error) ||
      !parseFrontierUnsigned(*root, "valid_shapes", validShapes, error) ||
      !parseFrontierUnsigned(*root, "unsupported_shapes", unsupportedShapes,
                             error))
    return false;
  if (canonicalSeedCalls > schedulerCalls) {
    error = "shape frontier canonical seed calls exceed scheduler calls";
    return false;
  }
  const llvm::json::Array *warmupSolved =
      root->getArray("warmup_solved_shapes");
  if (!warmupSolved) {
    error = "shape frontier omits the warmup solved-shape set";
    return false;
  }
  std::set<std::string> uniqueWarmupSolved;
  for (const llvm::json::Value &value : *warmupSolved) {
    auto shape = value.getAsString();
    if (!shape || shape->empty() ||
        llvm::any_of(*shape, [](char c) { return c < '0' || c > '9'; }) ||
        !uniqueWarmupSolved.insert(shape->str()).second) {
      error = "shape frontier warmup solved-shape set is malformed";
      return false;
    }
    if (!parseCandidateOrdinal("candidate-" + shape->str(),
                               "shape frontier warmup solved shape",
                               descriptor.candidateCount, error))
      return false;
    snapshot.warmupSolvedShapes.push_back(shape->str());
  }
  auto visited = root->getString("visited_shapes");
  auto boundPruned = root->getString("bound_pruned_shapes");
  if (!visited || !boundPruned || visited->empty() || boundPruned->empty() ||
      llvm::any_of(*visited, [](char c) { return c < '0' || c > '9'; }) ||
      llvm::any_of(*boundPruned, [](char c) { return c < '0' || c > '9'; })) {
    error = "shape frontier counters are not decimal";
    return false;
  }
  auto savedShapeLimit = root->getInteger("saved_max_shape_candidates");
  auto savedNodeLimit = root->getInteger("saved_max_expanded_nodes");
  auto savedTimeLimit = root->getInteger("saved_max_milliseconds");
  auto savedShapeDomainBound = root->getInteger("shape_domain_bound_cycles");
  if (!savedShapeLimit || !savedNodeLimit || !savedTimeLimit ||
      !savedShapeDomainBound || *savedShapeDomainBound < 0 ||
      *savedShapeLimit < 0 || *savedNodeLimit <= 0 || *savedTimeLimit < 0) {
    error = "shape frontier saved resource limits are invalid";
    return false;
  }
  if (!snapshot.ranked.empty() &&
      *savedShapeDomainBound < snapshot.ranked.back().cycles) {
    error = "shape frontier domain bound is below its incumbent cutoff";
    return false;
  }
  snapshot.resumeEligible = true;
  snapshot.reason = reason->str();
  snapshot.bindingWitnessPath = bindingPath->str();
  snapshot.canonicalIncumbentSeedEnabled = *canonicalSeedEnabled;
  snapshot.canonicalSeedContinuationState = canonicalSeedContinuation->str();
  snapshot.canonicalSeedSchedulerCalls = canonicalSeedCalls;
  snapshot.canonicalSeedAccepted = canonicalSeedAccepted;
  snapshot.canonicalSeedNextTask = canonicalSeedNextTask;
  snapshot.canonicalSeedNextSmaller = canonicalSeedNextSmaller;
  snapshot.candidateCount = candidateCount->str();
  snapshot.scheduleSerial = serial;
  snapshot.expandedNodes = expanded;
  snapshot.expandedShapeNodes = expandedShape;
  snapshot.productionSchedulerCalls = schedulerCalls;
  snapshot.validShapes = validShapes;
  snapshot.unsupportedShapes = unsupportedShapes;
  snapshot.shapeDomainBound = *savedShapeDomainBound;
  snapshot.visitedShapes = visited->str();
  snapshot.boundPrunedShapes = boundPruned->str();
  snapshot.savedMaxShapeCandidates = *savedShapeLimit;
  snapshot.savedMaxExpandedNodes = *savedNodeLimit;
  snapshot.savedMaxMilliseconds = *savedTimeLimit;
  return true;
}

// Re-run the source-owned fixed-schedule verifier for every incumbent row.
// Checking rectangle bounds alone would allow a forged seed to hide temporal
// overlap or a communication-ready violation.  The peer module supplies the
// real Taskflow communication model; the cost catalogue supplies the exact
// rounded ML II/startup duration used by the scorer.
static bool validateScoreRecordCommunication(
    const llvm::json::Object &record, StringRef label,
    const FactoredSpaceDescriptor &descriptor,
    ArrayRef<TaskMetadata> metadata,
    const std::vector<std::vector<unsigned>> &predecessors,
    TaskShapeCostCache &costs,
    ProductionFixedScheduleCommunication &communication,
    func::FuncOp sourceFunction, StringRef scheduleSpace, StringRef dispatchPolicy,
    std::string &error) {
  const llvm::json::Array *taskCosts = record.getArray("task_costs");
  const llvm::json::Array *schedule = record.getArray("task_schedule");
  const llvm::json::Array *dispatchOrder = record.getArray("dispatch_order");
  if (!taskCosts || taskCosts->size() != metadata.size() || !schedule ||
      schedule->size() != metadata.size() || !dispatchOrder ||
      dispatchOrder->size() != metadata.size()) {
    error = label.str() + " does not contain a complete fixed schedule/order";
    return false;
  }
  if (predecessors.size() != metadata.size()) {
    error = label.str() + " communication predecessor inventory is incomplete";
    return false;
  }

  std::map<std::string, unsigned> taskIndices;
  for (auto [index, task] : llvm::enumerate(metadata))
    if (!taskIndices.emplace(task.name, static_cast<unsigned>(index)).second) {
      error = label.str() + " communication task names are not unique";
      return false;
    }

  std::vector<ExactScheduleTask> tasks(metadata.size());
  std::vector<bool> taskCostSeen(metadata.size(), false);
  for (const llvm::json::Value &value : *taskCosts) {
    const llvm::json::Object *taskCost = value.getAsObject();
    if (!taskCost) {
      error = label.str() + " has a non-object task cost";
      return false;
    }
    auto taskName = taskCost->getString("task");
    auto mapperRows = taskCost->getInteger("mapper_tile_rows");
    auto mapperCols = taskCost->getInteger("mapper_tile_cols");
    if (!taskName || !mapperRows || !mapperCols || *mapperRows <= 0 ||
        *mapperCols <= 0) {
      error = label.str() + " has an incomplete mapper shape";
      return false;
    }
    auto taskIt = taskIndices.find(taskName->str());
    if (taskIt == taskIndices.end() || taskCostSeen[taskIt->second]) {
      error = label.str() + " task costs do not exactly cover communication tasks";
      return false;
    }
    const unsigned taskIndex = taskIt->second;
    taskCostSeen[taskIndex] = true;
    std::optional<RectShape> matchingShape;
    for (const RectShape &shape : descriptor.shapesByTask[taskIndex])
      if (shape.mapperRows == *mapperRows && shape.mapperCols == *mapperCols) {
        matchingShape = shape;
        break;
      }
    if (!matchingShape) {
      error = label.str() + " task cost mapper shape is outside the catalogue";
      return false;
    }
    TaskShapeChoice choice{taskName->str(), metadata[taskIndex].tripCount,
                           *matchingShape};
    const TaskShapeCost *actual = costs.get(choice, error);
    if (!actual || !actual->supported)
      return false;
    auto duration = predictDuration(*actual, choice.tripCount, error);
    if (!duration)
      return false;
    ExactScheduleTask &task = tasks[taskIndex];
    task.name = taskName->str();
    task.rows = static_cast<int>(matchingShape->rows);
    task.cols = static_cast<int>(matchingShape->cols);
    task.duration = *duration;
    task.predecessors = predecessors[taskIndex];
  }
  if (llvm::any_of(taskCostSeen, [](bool seen) { return !seen; })) {
    error = label.str() + " task costs omit a communication task";
    return false;
  }

  std::vector<ExactSchedulePlacement> placements;
  placements.reserve(schedule->size());
  std::vector<bool> placementSeen(metadata.size(), false);
  for (const llvm::json::Value &value : *schedule) {
    const llvm::json::Object *placement = value.getAsObject();
    auto taskName = placement ? placement->getString("task") : std::nullopt;
    auto row = placement ? placement->getInteger("row") : std::nullopt;
    auto col = placement ? placement->getInteger("col") : std::nullopt;
    auto rows = placement ? placement->getInteger("rows") : std::nullopt;
    auto cols = placement ? placement->getInteger("cols") : std::nullopt;
    auto start = placement ? placement->getInteger("start_cycle") : std::nullopt;
    auto end = placement ? placement->getInteger("end_cycle") : std::nullopt;
    if (!placement || !taskName || !row || !col || !rows || !cols || !start ||
        !end) {
      error = label.str() + " has an incomplete communication placement";
      return false;
    }
    auto taskIt = taskIndices.find(taskName->str());
    if (taskIt == taskIndices.end() || placementSeen[taskIt->second]) {
      error = label.str() + " schedule does not cover each task exactly once";
      return false;
    }
    const unsigned taskIndex = taskIt->second;
    placementSeen[taskIndex] = true;
    if (*rows != tasks[taskIndex].rows || *cols != tasks[taskIndex].cols) {
      error = label.str() + " schedule shape disagrees with task cost shape";
      return false;
    }
    ExactSchedulePlacement item;
    item.task = taskIndex;
    item.row = static_cast<int>(*row);
    item.col = static_cast<int>(*col);
    item.start = *start;
    item.end = *end;
    item.idleCycles = placement->getInteger("idle_cycles").value_or(0);
    placements.push_back(item);
  }
  if (llvm::any_of(placementSeen, [](bool seen) { return !seen; })) {
    error = label.str() + " schedule omits a communication task";
    return false;
  }

  std::vector<unsigned> order;
  order.reserve(dispatchOrder->size());
  std::vector<bool> orderSeen(metadata.size(), false);
  for (const llvm::json::Value &value : *dispatchOrder) {
    auto taskName = value.getAsString();
    if (!taskName) {
      error = label.str() + " dispatch order contains a non-task name";
      return false;
    }
    auto taskIt = taskIndices.find(taskName->str());
    if (taskIt == taskIndices.end() || orderSeen[taskIt->second]) {
      error = label.str() + " dispatch order does not cover each task once";
      return false;
    }
    orderSeen[taskIt->second] = true;
    order.push_back(taskIt->second);
  }
  if (llvm::any_of(orderSeen, [](bool seen) { return !seen; })) {
    error = label.str() + " dispatch order omits a task";
    return false;
  }

  if (scheduleSpace == "production-scheduler") {
    SmallVector<TaskShapeChoice> choices;
    SmallVector<int64_t> durations;
    for (unsigned index = 0; index < tasks.size(); ++index) {
      auto shape = llvm::find_if(descriptor.shapesByTask[index], [&](const RectShape &s) {
        return s.rows == tasks[index].rows && s.cols == tasks[index].cols;
      });
      if (shape == descriptor.shapesByTask[index].end()) {
        error = label.str() + " production shape is absent from its descriptor";
        return false;
      }
      choices.push_back({metadata[index].name, metadata[index].tripCount, *shape});
      durations.push_back(tasks[index].duration);
    }
    ProductionScheduledResult actual;
    auto candidate = record.getString("shape_candidate_id");
    if (!candidate || !scheduleWithAmoebaProductionPass(
            sourceFunction->getParentOfType<ModuleOp>(), sourceFunction,
            choices, durations, *candidate, dispatchPolicy == "fixed", actual, error))
      return false;
    if (actual.order != order || actual.placements.size() != placements.size() ||
        record.getInteger("predicted_whole_program_cycles") != std::optional<int64_t>(actual.makespan) ||
        record.getInteger("replayed_communication_edges") != std::optional<int64_t>(actual.replayedEdges)) {
      error = label.str() + " production scheduler replay disagrees with score";
      return false;
    }
    for (const auto &expected : placements) {
      auto found = llvm::find_if(actual.placements, [&](const auto &p) { return p.task == expected.task; });
      if (found == actual.placements.end() || found->row != expected.row ||
          found->col != expected.col || found->start != expected.start || found->end != expected.end) {
        error = label.str() + " production scheduler replay placement disagrees with score";
        return false;
      }
    }
    return true;
  }
  FixedScheduleScore replay = verifyAndScoreFixedSchedule(
      4, 4, tasks, placements, order, communication);
  if (!replay.valid) {
    error = label.str() + " communication replay rejected the schedule: " +
            replay.rejection;
    return false;
  }
  auto predicted = record.getInteger("predicted_whole_program_cycles");
  auto replayedEdges = record.getInteger("replayed_communication_edges");
  if (!predicted || *predicted != replay.predictedMakespan || !replayedEdges ||
      *replayedEdges < 0 ||
      static_cast<uint64_t>(*replayedEdges) != replay.replayedEdges) {
    error = label.str() +
            " communication replay result disagrees with the score record";
    return false;
  }
  return true;
}

// A feasible incumbent is the same five peer witnesses across graph-bound
// invocations. Cache its expensive MLIR/scheduler validation only after a
// successful proof, with every external input byte and policy in the key.
// File existence and score-start stability are checked while building each
// key, including hits. No operation/context pointer or negative proof is kept.
static thread_local std::string validatedIncumbentBindingsKey;

static bool buildIncumbentBindingsKey(
    func::FuncOp currentFunction, const TaskShapeCostCache &currentCosts,
    const CostCatalogBinding &binding, StringRef dispatchPolicy,
    StringRef temporalModel, StringRef scheduleSpace,
    ArrayRef<IncumbentRow> rows,
    const std::map<std::string, ClosureGraphBinding> &graphBindings,
    const std::map<std::string, std::string> &scorePaths,
    std::string &key, std::string &error) {
  key.clear();
  appendExactCacheField(key, "contract", "validated-incumbent-bindings-v1");
  appendExactCacheField(key, "function", currentFunction.getSymName());
  appendExactCacheField(key, "dispatch", dispatchPolicy);
  appendExactCacheField(key, "temporal", temporalModel);
  appendExactCacheField(key, "space", scheduleSpace);
  if (scheduleSpace == "production-scheduler") {
    int64_t fixedPointIterations = 0;
    if (!getProductionFixedPointIterations(currentFunction,
                                            fixedPointIterations, error))
      return false;
    appendExactCacheField(key, "production-scheduler-contract",
                          kProductionSchedulerContract);
    appendExactCacheOption(key, "production-fixed-point-max-iterations",
                           fixedPointIterations);
  }
  // sourceGraphId belongs to the current graph, and is not compared against
  // the peer graph. Include precisely the common catalogue binding instead.
  for (StringRef field : {StringRef(binding.nameSpace), StringRef(binding.model),
       StringRef(binding.modelSchema), StringRef(binding.featureContractId),
       StringRef(binding.featureExtractor), StringRef(binding.predictorSource),
       StringRef(binding.provenanceSchema), StringRef(binding.architectureSchema),
       StringRef(binding.architecturePath), StringRef(binding.sourceRepository),
       StringRef(binding.sourceCommit), currentCosts.nameSpace(),
       currentCosts.architectureSchema(), currentCosts.architecturePath(),
       currentCosts.sourceRepository(), currentCosts.sourceCommit()})
    appendExactCacheField(key, "binding", field);
  for (const IncumbentRow &row : rows) {
    appendExactCacheField(key, "graph", row.graphVariantId);
    appendExactCacheField(key, "semantic-graph", row.semanticGraphId);
    appendExactCacheField(key, "candidate", row.candidateId);
    appendExactCacheOption(key, "cycles", row.cycles);
    appendExactCacheField(key, "score-path", row.scorePath);
    // A graph can contribute rows from distinct score artifacts. Snapshot
    // every row's source file, even when the command witness is shared.
    std::string sourceScoreBytes;
    if (!readExactCacheInput(row.scorePath, sourceScoreBytes, error))
      return false;
    appendExactCacheField(key, "selected-source-score-bytes", sourceScoreBytes);
    llvm::json::Object record = row.scoreRecord;
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << llvm::json::Value(std::move(record));
    stream.flush();
    appendExactCacheField(key, "selected-record", text);
  }
  for (const auto &entry : scorePaths) {
    auto it = graphBindings.find(entry.first);
    if (it == graphBindings.end() || it->second.ambiguous) {
      error = "incumbent graph is absent or has ambiguous persisted artifacts";
      return false;
    }
    const ClosureGraphBinding &graph = it->second;
    appendExactCacheField(key, "graph-binding", graph.path);
    appendExactCacheField(key, "graph-module", graph.mlirPath);
    appendExactCacheField(key, "graph-facts", graph.factsPath);
    llvm::SmallString<256> commandPath(entry.second);
    llvm::sys::path::remove_filename(commandPath);
    llvm::sys::path::append(commandPath, "score.command.json");
    PeerScoreCommand command;
    if (!readPeerScoreCommand(commandPath, command, error))
      return false;
    if (!command.startedUnix || !std::isfinite(*command.startedUnix) ||
        *command.startedUnix <= 0.0) {
      error = "incumbent cache input lacks finite score-start provenance";
      return false;
    }
    for (StringRef input : {StringRef(command.architecturePath),
         StringRef(command.modulePath), StringRef(command.candidatePath),
         StringRef(command.costPath), StringRef(graph.factsPath)})
      if (!peerInputWasStableBeforeStart(input, *command.startedUnix, error))
        return false;
    for (StringRef input : {StringRef(commandPath), StringRef(entry.second),
         StringRef(command.architecturePath), StringRef(command.modulePath),
         StringRef(command.candidatePath), StringRef(command.costPath),
         StringRef(graph.mlirPath), StringRef(graph.factsPath)}) {
      std::string bytes;
      if (!readExactCacheInput(input, bytes, error))
        return false;
      appendExactCacheField(key, "input-path", input);
      appendExactCacheField(key, "input-bytes", bytes);
    }
  }
  return true;
}

static bool validateIncumbentBindings(
    func::FuncOp currentFunction, ArrayRef<TaskMetadata> currentTasks,
    const TaskShapeCostCache &currentCosts, StringRef currentCandidatePath,
    StringRef currentCostPath, StringRef expectedDispatchPolicy,
    StringRef expectedTemporalModel, StringRef expectedScheduleSpace,
    ArrayRef<IncumbentRow> rows,
    const std::map<std::string, ClosureGraphBinding> &graphBindings,
    std::string &error) {
  (void)currentTasks;
  (void)currentCandidatePath;
  CostCatalogBinding currentBinding;
  if (!readCostCatalogBinding(currentCostPath, currentBinding, error))
    return false;
  if (currentBinding.nameSpace != currentCosts.nameSpace().str() ||
      currentBinding.architectureSchema !=
          currentCosts.architectureSchema().str() ||
      !samePath(currentBinding.architecturePath,
                currentCosts.architecturePath())) {
    error = "current cost catalogue object disagrees with its validated cache";
    return false;
  }
  std::map<std::string, std::string> scorePaths;
  for (const IncumbentRow &row : rows) {
    auto [entry, inserted] = scorePaths.emplace(row.graphVariantId, row.scorePath);
    if (!inserted && entry->second != row.scorePath) {
      error = "one incumbent graph names multiple score-command witnesses";
      return false;
    }
  }
  int64_t expectedFixedPointIterations = 0;
  if (expectedScheduleSpace == "production-scheduler" &&
      !getProductionFixedPointIterations(currentFunction,
                                          expectedFixedPointIterations, error))
    return false;

  std::string validationKey;
  if (!buildIncumbentBindingsKey(currentFunction, currentCosts, currentBinding,
      expectedDispatchPolicy, expectedTemporalModel, expectedScheduleSpace,
      rows, graphBindings, scorePaths, validationKey, error))
    return false;
  if (validationKey == validatedIncumbentBindingsKey)
    return true;

  for (const auto &entry : scorePaths) {
    const std::string &graphId = entry.first;
    const std::string &scorePath = entry.second;
    auto graphIt = graphBindings.find(graphId);
    if (graphIt == graphBindings.end()) {
      error = "incumbent graph is absent from the persisted closure";
      return false;
    }
    const ClosureGraphBinding &graph = graphIt->second;
    if (graph.ambiguous) {
      error = "incumbent graph identity has multiple persisted artifacts";
      return false;
    }
    if (!llvm::sys::fs::exists(scorePath)) {
      error = "incumbent score file is unavailable: " + scorePath;
      return false;
    }
    llvm::SmallString<256> commandPath(scorePath);
    llvm::sys::path::remove_filename(commandPath);
    llvm::sys::path::append(commandPath, "score.command.json");
    if (!llvm::sys::fs::exists(commandPath)) {
      error = "incumbent score.command.json is unavailable beside " +
              scorePath;
      return false;
    }
    PeerScoreCommand command;
    if (!readPeerScoreCommand(commandPath, command, error))
      return false;
    bool commandSeedOnly = false;
    if (auto seedOption = command.options.find("seed-only");
        seedOption != command.options.end()) {
      if (seedOption->second != "true" && seedOption->second != "false") {
        error = "incumbent score command has an invalid seed-only option";
        return false;
      }
      commandSeedOnly = seedOption->second == "true";
    }
    std::string commandScoreText;
    llvm::json::Object commandScoreHeader, commandScoreFooter;
    if (!readExactCacheInput(scorePath, commandScoreText, error) ||
        !peerScoreHeaderFooter(commandScoreText, commandScoreHeader,
                               commandScoreFooter, error))
      return false;
    const bool scoreSeedOnly =
        commandScoreHeader.getBoolean("seed_only").value_or(false) ||
        commandScoreFooter.getString("status") ==
            std::optional<StringRef>("seed-only");
    if (!requirePeerString(commandScoreHeader, "schedule_space", expectedScheduleSpace, error) ||
        !requirePeerString(commandScoreHeader, "dispatch_policy", expectedDispatchPolicy, error) ||
        !requirePeerString(commandScoreHeader, "start_policy", scoreStartPolicy(expectedTemporalModel, expectedScheduleSpace), error) ||
        !requirePeerString(commandScoreHeader, "temporal_search_scope", scoreTemporalScope(expectedTemporalModel), error))
      return false;
    if (expectedScheduleSpace == "production-scheduler") {
      for (const llvm::json::Object *record :
           {&commandScoreHeader, &commandScoreFooter}) {
        if (!requirePeerInteger(*record,
                "production_fixed_point_max_iterations",
                expectedFixedPointIterations, error))
          return false;
        // The v2 score writer initially emitted the effective 64-round
        // budget without the optional contract string. This is the only
        // accepted unmarked legacy form, with full body/schedule proof below.
        auto contract = record->getString("production_scheduler_contract");
        if ((contract && *contract != kProductionSchedulerContract) ||
            (!contract && expectedFixedPointIterations !=
                              kDefaultProductionFixedPointIterations)) {
          error = "incumbent score production scheduler contract differs";
          return false;
        }
      }
    }
    if (scoreSeedOnly != commandSeedOnly) {
      error = "incumbent score seed-only status disagrees with score.command.json";
      return false;
    }
    if (!samePath(command.scorePath, scorePath) ||
        !samePath(command.modulePath, graph.mlirPath) ||
        !samePath(command.architecturePath, currentCosts.architecturePath()) ||
        !samePeerOption(command, "graph-variant-id", graphId, error) ||
        !samePeerOption(command, "source-git-repository",
                        currentCosts.sourceRepository(), error) ||
        !samePeerOption(command, "source-git-commit",
                        currentCosts.sourceCommit(), error) ||
        !samePeerOption(command, "dispatch-policy", expectedDispatchPolicy, error) ||
        !samePeerOption(command, "schedule-space", expectedScheduleSpace, error) ||
        !samePeerOption(command, "temporal-model", expectedTemporalModel, error))
      return false;
    if (!commandSeedOnly &&
        (!samePeerOption(command, "max-expanded-nodes",
                         std::to_string(std::numeric_limits<int64_t>::max()),
                         error) ||
         !samePeerOption(command, "max-milliseconds", "0", error) ||
         !samePeerOption(command, "max-makespan", "0", error)))
      return false;
    auto topK = command.options.find("top-k");
    int64_t topKValue = 0;
    if (topK == command.options.end() ||
        StringRef(topK->second).getAsInteger(10, topKValue) || topKValue < 5) {
      error = "incumbent score command does not request at least five rows";
      return false;
    }
    if (!command.hasExitCode || command.exitCode != 0 || !command.startedUnix ||
        !std::isfinite(*command.startedUnix) || *command.startedUnix <= 0.0) {
      error = "incumbent score command lacks successful start/exit provenance";
      return false;
    }
    for (StringRef input : {StringRef(command.architecturePath),
                            StringRef(command.modulePath),
                            StringRef(command.candidatePath),
                            StringRef(command.costPath),
                            StringRef(graph.factsPath)})
      if (!peerInputWasStableBeforeStart(input, *command.startedUnix, error))
        return false;

    OwningOpRef<ModuleOp> peerModule = parseSourceFile<ModuleOp>(
        command.modulePath, ParserConfig(currentFunction.getContext()));
    if (!peerModule) {
      error = "cannot parse incumbent peer module: " + command.modulePath;
      return false;
    }
    FailureOr<func::FuncOp> peerFunction =
        selectTaskFunction(*peerModule, currentFunction.getSymName(), error);
    if (failed(peerFunction))
      return false;
    if (expectedScheduleSpace == "production-scheduler") {
      int64_t peerIterations = 0;
      if (!getProductionFixedPointIterations(*peerFunction, peerIterations,
                                              error))
        return false;
      if (peerIterations != expectedFixedPointIterations) {
        error = "incumbent peer/current production fixed-point budgets differ";
        return false;
      }
    }
    FailureOr<SmallVector<TaskMetadata>> peerMetadata =
        collectAnalyticalTaskMetadata(*peerFunction, error);
    if (failed(peerMetadata))
      return false;
    const ::mlir::neura::Architecture &architecture =
        ::mlir::neura::getArchitecture();
    FactoredSpaceDescriptor peerDescriptor;
    if (!parseFactoredSpace(command.candidatePath, *peerFunction,
                            *peerMetadata, graphId, architecture,
                            peerDescriptor, error))
      return false;
    TaskShapeCostCache peerCosts;
    if (!peerCosts.load(command.costPath, currentFunction.getSymName(),
                        *peerMetadata, currentCosts.sourceRepository(),
                        currentCosts.sourceCommit(),
                        currentCosts.architecturePath(), graphId, error))
      return false;
    CostCatalogBinding peerBinding;
    if (!readCostCatalogBinding(command.costPath, peerBinding, error) ||
        !compareCostCatalogBindings(peerBinding, currentBinding,
                                    currentCosts.architecturePath(), error))
      return false;
    if (peerCosts.candidateCountDecimal() != peerDescriptor.candidateCount) {
      error = "incumbent cost catalogue and candidate descriptor counts differ";
      return false;
    }

    std::string peerGraphError;
    FailureOr<TaskEdgeGraph> peerGraph = buildTaskEdgeGraph(
        *peerFunction, TaskEdgeGraphOptions{/*require_payload=*/true,
                                            /*read_control_predecessors=*/true},
        peerGraphError);
    if (failed(peerGraph)) {
      error = "cannot build incumbent communication graph: " + peerGraphError;
      return false;
    }
    llvm::DenseMap<Operation *, unsigned> peerOperationIndex;
    std::vector<Operation *> peerTaskOperations(peerMetadata->size(), nullptr);
    for (taskflow::TaskflowTaskOp task : peerGraph->getTasks()) {
      auto taskIt = llvm::find_if(*peerMetadata, [&](const TaskMetadata &item) {
        return item.name == task.getTaskName();
      });
      if (taskIt == peerMetadata->end()) {
        error = "incumbent communication graph contains an unknown task";
        return false;
      }
      unsigned index = static_cast<unsigned>(taskIt - peerMetadata->begin());
      if (peerTaskOperations[index]) {
        error = "incumbent communication graph repeats a task operation";
        return false;
      }
      peerTaskOperations[index] = task.getOperation();
      peerOperationIndex[task.getOperation()] = index;
    }
    if (llvm::any_of(peerTaskOperations,
                     [](Operation *operation) { return operation == nullptr; })) {
      error = "incumbent communication graph omits a task operation";
      return false;
    }
    std::vector<std::vector<unsigned>> peerPredecessors(peerMetadata->size());
    for (TaskEdge edge : peerGraph->getEdges()) {
      auto producer = peerOperationIndex.find(edge.producer.getOperation());
      auto consumer = peerOperationIndex.find(edge.consumer.getOperation());
      if (producer == peerOperationIndex.end() ||
          consumer == peerOperationIndex.end()) {
        error = "incumbent communication graph has an unknown edge endpoint";
        return false;
      }
      auto &list = peerPredecessors[consumer->second];
      if (!llvm::is_contained(list, producer->second))
        list.push_back(producer->second);
    }
    auto peerNetwork = createInterTaskNetworkCommunicationModel(
        *peerFunction, 4, 4, peerGraphError);
    if (failed(peerNetwork)) {
      error = "cannot create incumbent communication model: " + peerGraphError;
      return false;
    }
    ProductionFixedScheduleCommunication peerCommunication(**peerNetwork,
                                                           peerTaskOperations);

    // Validate every embedded score row against the actual command-bound
    // catalogue. Global ranker selections carry a copy of the score row; read
    // the source score file as well so forged selection payloads cannot supply
    // an incumbent's task costs.
    for (const IncumbentRow &row : rows) {
      if (row.graphVariantId != graphId)
        continue;
      llvm::json::Object record = row.scoreRecord;
      std::string scoreText;
      if (!readExactCacheInput(row.scorePath, scoreText, error))
        return false;
      llvm::MemoryBufferRef scoreBuffer(scoreText, "incumbent-score");
      llvm::line_iterator scoreLines(scoreBuffer, true);
      bool found = false;
      for (; !scoreLines.is_at_end(); ++scoreLines) {
        llvm::Expected<llvm::json::Value> parsed =
            llvm::json::parse(*scoreLines);
        if (!parsed) {
          error = "incumbent score file contains invalid JSON";
          return false;
        }
        const llvm::json::Object *object = parsed->getAsObject();
        if (!object || object->getString("record_type") !=
                           std::optional<StringRef>("score"))
          continue;
        if (object->getString("candidate_id") ==
                std::optional<StringRef>(row.candidateId) &&
            object->getInteger("predicted_whole_program_cycles") ==
                std::optional<int64_t>(row.cycles)) {
          record = *object;
          found = true;
          break;
        }
      }
      if (!found) {
        error = "incumbent candidate is absent from its source score file";
        return false;
      }
      if (!validateScoreRecordAgainstCatalog(
              record, "incumbent candidate " + row.candidateId,
              peerDescriptor, *peerMetadata, peerCosts, error))
        return false;
      if (!validateScoreRecordCommunication(
              record, "incumbent candidate " + row.candidateId, peerDescriptor,
              *peerMetadata, peerPredecessors, peerCosts, peerCommunication,
              *peerFunction, expectedScheduleSpace, expectedDispatchPolicy, error))
        return false;
    }
  }
  // Re-read after the proof before publishing the cache entry: concurrent
  // file changes cannot attach a successful proof to pre-validation bytes.
  std::string confirmedKey;
  if (!buildIncumbentBindingsKey(currentFunction, currentCosts, currentBinding,
      expectedDispatchPolicy, expectedTemporalModel, expectedScheduleSpace,
      rows, graphBindings, scorePaths, confirmedKey, error))
    return false;
  if (validationKey != confirmedKey) {
    error = "incumbent inputs changed during validation";
    return false;
  }
  validatedIncumbentBindingsKey = std::move(validationKey);
  return true;
}

static PeerProbeStatus importCompletePeerScore(
    func::FuncOp function, StringRef peerRoot, StringRef graphVariantId,
    StringRef currentCandidate, StringRef currentCost,
    StringRef currentArchitecture, StringRef sourceRepository,
    StringRef sourceCommit, StringRef dispatchPolicy, StringRef temporalModel,
    StringRef scheduleSpace, StringRef currentCandidateCount,
    int64_t topK, int64_t maxExpandedNodes, int64_t maxMilliseconds,
    int64_t maxMakespan, int64_t maxCgrasPerTask, int64_t stateMemoCapacity,
    StringRef currentSourceGraphId, ExactScoreCacheLease &cache,
    std::string &error) {
  const std::string peerDirectory =
      resolvePeerVariantDirectory(peerRoot, graphVariantId);
  if (peerDirectory.empty()) {
    error = "peer variant directory for the requested graph is unavailable";
    return PeerProbeStatus::Rejected;
  }
  if (!llvm::sys::fs::exists(peerDirectory + "/score.command.json")) {
    error = "peer score command is unavailable for the requested graph";
    return PeerProbeStatus::Rejected;
  }
  PeerScoreCommand command;
  if (!readPeerScoreCommand(peerDirectory + "/score.command.json", command,
                            error))
    return PeerProbeStatus::Rejected;
  if (!samePeerOption(command, "top-k", std::to_string(topK), error) ||
      !samePeerOption(command, "max-expanded-nodes",
                      std::to_string(maxExpandedNodes), error) ||
      !samePeerOption(command, "max-milliseconds",
                      std::to_string(maxMilliseconds), error) ||
      !samePeerOption(command, "max-makespan", std::to_string(maxMakespan),
                      error) ||
      !samePeerOption(command, "dispatch-policy", dispatchPolicy, error) ||
      !samePeerOption(command, "schedule-space", scheduleSpace, error) ||
      !samePeerOption(command, "temporal-model", temporalModel, error) ||
      !samePeerOption(command, "graph-variant-id", graphVariantId, error) ||
      !samePeerOption(command, "source-git-repository", sourceRepository,
                      error) ||
      !samePeerOption(command, "source-git-commit", sourceCommit, error))
    return PeerProbeStatus::Rejected;
  if (command.hasExitCode && command.exitCode != 0) {
    error = "peer score command did not finish successfully";
    return PeerProbeStatus::Rejected;
  }
  if (!command.startedUnix || !std::isfinite(*command.startedUnix) ||
      *command.startedUnix < 0.0 ||
      *command.startedUnix > std::chrono::duration<double>(
          std::chrono::system_clock::now().time_since_epoch()).count()) {
    error = "peer score command has no started_unix provenance";
    return PeerProbeStatus::Rejected;
  }
  if (auto it = command.options.find("state-memo-capacity");
      it != command.options.end() &&
      it->second != std::to_string(stateMemoCapacity)) {
    error = "peer score command differs at state-memo-capacity";
    return PeerProbeStatus::Rejected;
  }
  if (!sameFileBytes(command.candidatePath, currentCandidate, error) ||
      !sameFileBytes(command.costPath, currentCost, error) ||
      !sameFileBytes(command.architecturePath, currentArchitecture, error) ||
      !peerFunctionMatches(function, command.modulePath, error))
    return PeerProbeStatus::Rejected;
  for (StringRef input : {StringRef(command.modulePath),
                          StringRef(command.architecturePath),
                          StringRef(command.candidatePath),
                          StringRef(command.costPath)}) {
    if (!peerInputWasStableBeforeStart(input, *command.startedUnix, error))
      return PeerProbeStatus::Rejected;
  }

  std::string peerScore;
  if (!readExactCacheInput(command.scorePath, peerScore, error)) {
    if (command.hasExitCode) {
      error = "peer score command finished without a score file";
      return PeerProbeStatus::Rejected;
    }
    error = "peer score command is still running without a score file";
    return PeerProbeStatus::Pending;
  }
  llvm::json::Object header, footer;
  if (!peerScoreHeaderFooter(peerScore, header, footer, error)) {
    if (command.hasExitCode)
      return PeerProbeStatus::Rejected;
    return PeerProbeStatus::Pending;
  }
  if (footer.getBoolean("incomplete") != std::optional<bool>(false) ||
      footer.getBoolean("top_k_certified") != std::optional<bool>(true) ||
      footer.getString("status") != std::optional<StringRef>("complete")) {
    error = "peer score is present but not complete/top-k certified";
    return command.hasExitCode ? PeerProbeStatus::Rejected
                               : PeerProbeStatus::Pending;
  }
  int64_t expectedProductionFixedPointIterations = 0;
  if (scheduleSpace == "production-scheduler" &&
      !getProductionFixedPointIterations(
          function, expectedProductionFixedPointIterations, error))
    return PeerProbeStatus::Rejected;
  auto checkOptionalProductionContract = [&](const llvm::json::Object &record) {
    auto contract = record.getString("production_scheduler_contract");
    if (!contract)
      return true;
    if (*contract != kProductionSchedulerContract) {
      error = "peer score production scheduler contract differs";
      return false;
    }
    return true;
  };
  if (!requirePeerString(header, "schema", "amoeba-exact-joint-task-scores-v1",
                         error) ||
      !requirePeerString(header, "function", function.getSymName(), error) ||
      !requirePeerString(header, "graph_variant_id", graphVariantId, error) ||
      !requirePeerString(header, "source_graph_id", currentSourceGraphId,
                         error) ||
      !requirePeerCount(header, "candidate_count", currentCandidateCount,
                        error) ||
      !requirePeerString(header, "dispatch_policy", dispatchPolicy, error) ||
      !requirePeerString(header, "schedule_space", scheduleSpace, error) ||
      !requirePeerString(header, "source_repository", sourceRepository,
                         error) ||
      !requirePeerString(header, "source_commit", sourceCommit, error) ||
      !requirePeerInteger(header, "top_k", topK, error) ||
      !requirePeerInteger(header, "max_makespan", maxMakespan, error) ||
      !requirePeerInteger(header, "state_memo_capacity",
                          stateMemoCapacity, error) ||
      !requirePeerInteger(header, "max_cgras_per_task", maxCgrasPerTask,
                          error) ||
      !requirePeerString(header, "start_policy", scoreStartPolicy(temporalModel, scheduleSpace), error) ||
      !requirePeerString(header, "temporal_search_scope", scoreTemporalScope(temporalModel), error) ||
      !requirePeerString(header, "communication_mode", "explicit", error) ||
      !requirePeerBool(header, "cycle_starts_are_search_dimensions", false,
                       error) ||
      !requirePeerBool(header, "horizon_proven", true, error) ||
      !requirePeerBool(header, "production_ready", false, error))
    return PeerProbeStatus::Rejected;
  if (scheduleSpace == "production-scheduler" &&
      (!checkOptionalProductionContract(header) ||
       !requirePeerInteger(header, "production_fixed_point_max_iterations",
                           expectedProductionFixedPointIterations, error)))
    return PeerProbeStatus::Rejected;
  if (!requirePeerBool(footer, "incomplete", false, error) ||
      !requirePeerBool(footer, "top_k_certified", true, error) ||
      !requirePeerString(footer, "status", "complete", error) ||
      !requirePeerCount(footer, "candidate_count", currentCandidateCount,
                        error) ||
      !requirePeerString(footer, "dispatch_policy", dispatchPolicy, error) ||
      !requirePeerString(footer, "schedule_space", scheduleSpace, error) ||
      !requirePeerInteger(footer, "top_k_requested", topK, error) ||
      !requirePeerInteger(footer, "state_memo_capacity",
                          stateMemoCapacity, error) ||
      !requirePeerBool(footer, "horizon_proven", true, error) ||
      !requirePeerBool(footer, "production_ready", false, error))
    return PeerProbeStatus::Rejected;
  if (scheduleSpace == "production-scheduler" &&
      (!checkOptionalProductionContract(footer) ||
       !requirePeerInteger(footer, "production_fixed_point_max_iterations",
                           expectedProductionFixedPointIterations, error)))
    return PeerProbeStatus::Rejected;
  const std::string expectedCertificate = scoreRankingCertificate(temporalModel, scheduleSpace).str();
  if (!requirePeerString(footer, "ranking_certificate", expectedCertificate,
                         error))
    return PeerProbeStatus::Rejected;
  if (!cache.publishImportedPeer(peerScore, error))
    return PeerProbeStatus::Rejected;
  return PeerProbeStatus::Imported;
}

static bool writePeerProbeOutput(
    func::FuncOp function, const FactoredSpaceDescriptor &descriptor,
    const TaskShapeCostCache &costs, StringRef graphVariantId,
    StringRef outputPath, bool scoreCacheRequested,
    bool scoreCachePeerRequested, StringRef peerStatus, StringRef reason,
    StringRef dispatchPolicy, StringRef temporalModel, StringRef scheduleSpace,
    int64_t topK, int64_t stateMemoCapacity, std::string &error) {
  return writeAtomically(
      outputPath,
      [&](llvm::raw_ostream &stream) {
        llvm::json::Object header;
        header["record_type"] = "header";
        header["schema"] = "amoeba-exact-joint-task-scores-v1";
        header["function"] = function.getSymName().str();
        header["representation"] = "factored-plus-exact-schedule";
        header["candidate_count"] = descriptor.candidateCount;
        header["top_k"] = topK;
        header["grid_rows"] = 4;
        header["grid_cols"] = 4;
        header["max_cgras_per_task"] =
            static_cast<int64_t>(descriptor.maxCgrasPerTask);
        header["communication_mode"] = "explicit";
        header["start_policy"] = scoreStartPolicy(temporalModel, scheduleSpace).str();
        header["schedule_space"] = scheduleSpace.str();
        header["placement_coverage_limited"] =
            scheduleSpace != "exact";
        header["production_ready"] = false;
        header["predictor_status"] = "exploratory";
        header["state_memo_requested"] = scheduleSpace == "exact";
        header["state_memo_capacity"] = stateMemoCapacity;
        header["dispatch_policy"] = dispatchPolicy.str();
        header["max_makespan"] = 0;
        header["horizon_proven"] = false;
        header["global_shape_location_temporal_space"] = scheduleSpace != "production-scheduler";
        header["temporal_search_scope"] =
            scoreTemporalScope(temporalModel).str();
        header["cycle_starts_are_search_dimensions"] =
            temporalModel == "all-integer-diagnostic";
        header["predictor_source"] = kMlPredictorSource.str();
        header["source_repository"] = costs.sourceRepository().str();
        header["source_commit"] = costs.sourceCommit().str();
        header["architecture_path"] = costs.architecturePath().str();
        header["source_graph_id"] = costs.sourceGraphId().str();
        header["graph_variant_id"] = graphVariantId.str();
        header["score_cache_enabled"] = scoreCacheRequested;
        header["score_cache_hit"] = false;
        header["score_cache_reused"] = false;
        header["score_cache_new_expanded_nodes"] = 0;
        header["score_cache_peer_requested"] = scoreCachePeerRequested;
        header["score_cache_peer_imported"] = false;
        header["score_cache_peer_pending"] = peerStatus == "pending";
        header["score_cache_peer_status"] = peerStatus.str();
        addProductionCoverage(header, temporalModel, scheduleSpace, dispatchPolicy, function);
        writeJsonLine(stream, std::move(header));

        llvm::json::Object footer;
        footer["record_type"] = "footer";
        footer["schema"] = "amoeba-exact-joint-task-scores-v1";
        footer["scored_count"] = 0;
        footer["status"] = peerStatus == "pending" ? "peer_pending"
                                                     : "peer_rejected";
        footer["incomplete"] = true;
        footer["reason"] = reason.str();
        footer["candidate_count"] = descriptor.candidateCount;
        footer["shape_candidates_visited"] = "0";
        footer["valid_shape_candidates"] = "0";
        footer["unmeasured_shape_candidates"] = descriptor.candidateCount;
        footer["bound_pruned_shape_candidates"] = "0";
        footer["expanded_shape_nodes"] = 0;
        footer["expanded_schedule_nodes"] = 0;
        footer["schedule_paths"] = 0;
        footer["dispatch_policy"] = dispatchPolicy.str();
        footer["schedule_space"] = scheduleSpace.str();
        footer["production_ready"] = false;
        footer["horizon_proven"] = false;
        footer["top_k_requested"] = topK;
        footer["top_k_certified"] = false;
        footer["ranking_certificate"] = "best-found-uncertified";
        putDecimal(footer, "production_scheduler_calls", "0");
        footer["orbit_4x4_pruning"] = scheduleSpace != "production-scheduler";
        footer["score_cache_enabled"] = scoreCacheRequested;
        footer["score_cache_hit"] = false;
        footer["score_cache_reused"] = false;
        footer["score_cache_new_expanded_nodes"] = 0;
        footer["score_cache_peer_requested"] = scoreCachePeerRequested;
        footer["score_cache_peer_imported"] = false;
        footer["score_cache_peer_pending"] = peerStatus == "pending";
        footer["score_cache_peer_status"] = peerStatus.str();
        addProductionCoverage(footer, temporalModel, scheduleSpace, dispatchPolicy, function);
        writeJsonLine(stream, std::move(footer));
        return true;
      },
      error);
}

struct ScoreExactJointTaskCandidatesPass
    : PassWrapper<ScoreExactJointTaskCandidatesPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScoreExactJointTaskCandidatesPass)

  ScoreExactJointTaskCandidatesPass() = default;
  ScoreExactJointTaskCandidatesPass(
      const ScoreExactJointTaskCandidatesPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "score-exact-joint-task-candidates";
  }
  StringRef getDescription() const override {
    return "Rank the complete factored shape plus explicit 4x4 temporal space "
           "with fixed or all-ready dispatch";
  }

  Option<std::string> functionName{*this, "function", llvm::cl::init("")};
  Option<std::string> candidateFile{*this, "candidates", llvm::cl::init("")};
  Option<std::string> costFile{*this, "cost-file", llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output", llvm::cl::init("")};
  Option<int64_t> topK{*this, "top-k", llvm::cl::init(5)};
  Option<int64_t> maxShapeCandidates{
      *this, "max-shape-candidates", llvm::cl::init(0)};
  Option<int64_t> maxExpandedNodes{
      *this, "max-expanded-nodes", llvm::cl::init(1000000)};
  Option<int64_t> maxMilliseconds{*this, "max-milliseconds", llvm::cl::init(0)};
  Option<int64_t> maxMakespan{*this, "max-makespan", llvm::cl::init(0)};
  Option<int64_t> stateMemoCapacity{
      *this, "state-memo-capacity", llvm::cl::init(100000)};
  Option<std::string> dispatchPolicy{
      *this, "dispatch-policy", llvm::cl::init("all-ready")};
  Option<std::string> temporalModel{*this, "temporal-model", llvm::cl::init("dispatch-order")};
  Option<std::string> scheduleSpace{*this, "schedule-space", llvm::cl::init("exact")};
  // Emit a bounded set of C++-replayed feasible rows for use as a global
  // incumbent.  This mode is intentionally incomplete and never certifies a
  // local top-five or the enclosing graph space.
  Option<bool> seedOnly{*this, "seed-only", llvm::cl::init(false)};
  Option<bool> rankingPruning{*this, "ranking-pruning", llvm::cl::init(true)};
  // Exact contiguous partitions preserve original shape ordinals. They are
  // coverage partitions, never candidate limits or legality cuts.
  Option<int64_t> shapeShardCount{*this, "shape-shard-count", llvm::cl::init(1)};
  Option<int64_t> shapeShardIndex{*this, "shape-shard-index", llvm::cl::init(0)};
  // Post-enumeration graph screening reuses the exact scorer's provenance
  // checks and optimistic graph bound, without seeds or joint traversal.
  Option<bool> graphBoundOnly{*this, "graph-bound-only", llvm::cl::init(false)};
  Option<bool> horizonProven{*this, "horizon-proven", llvm::cl::init(false)};
  Option<std::string> graphVariantId{*this, "graph-variant-id", llvm::cl::init("")};
  Option<std::string> sourceRepository{
      *this, "source-git-repository", llvm::cl::init("")};
  Option<std::string> sourceCommit{*this, "source-git-commit", llvm::cl::init("")};
  // When present, this is a C++-verified incumbent from another graph of the
  // same program/stage/architecture.  An empty option preserves the existing
  // per-graph search exactly.
  Option<std::string> stage{*this, "stage", llvm::cl::init("")};
  Option<std::string> globalIncumbentFile{
      *this, "global-incumbent", llvm::cl::init("")};
  Option<std::string> globalStageManifestFile{
      *this, "global-stage-manifest", llvm::cl::init("")};
  Option<std::string> scoreCacheDirectory{
      *this, "score-cache-dir", llvm::cl::init("")};
  // Old production score entries did not record the fixed-point budget.  A
  // missing budget is reusable only when this explicit compatibility option
  // declares the known legacy upper bound; it is never guessed silently.
  Option<int64_t> legacyProductionCacheMaxIterations{
      *this, "legacy-production-cache-max-iterations", llvm::cl::init(-1)};
  Option<std::string> scoreCachePeerDirectory{
      *this, "score-cache-peer-directory", llvm::cl::init("")};
  Option<bool> scoreCachePeerProbe{
      *this, "score-cache-peer-probe", llvm::cl::init(false)};
  // A prior score/checkpoint may seed current production scheduling with
  // shape ordinals.  It is intentionally excluded from the score-cache key:
  // hints affect only search order, never the meaning of a cached result.
  Option<std::string> warmStartShapes{
      *this, "warm-start-shapes", llvm::cl::init("")};
  // A frontier snapshot is a source-owned continuation token for the exact
  // production shape DFS.  It is deliberately separate from the legacy
  // top-five checkpoint and warm-start rows: neither of those files records
  // which subtrees have already been visited.
  Option<std::string> shapeFrontierSnapshot{
      *this, "shape-frontier-snapshot", llvm::cl::init("")};
  Option<std::string> resumeShapeFrontier{
      *this, "resume-shape-frontier", llvm::cl::init("")};
  // Zero keeps the feature opt-in.  Positive values checkpoint after this
  // many shape DFS nodes; the budget is a cadence, not a search limit.
  Option<int64_t> frontierCheckpointEvery{
      *this, "frontier-checkpoint-every", llvm::cl::init(0)};

  // Optional feasible seeds improve the cycle/numeric-ordinal incumbent.
  // Every trial still invokes the production scheduler; this changes search
  // order only and never removes a shape or certifies a partial traversal.
  Option<bool> canonicalIncumbentSeed{
      *this, "canonical-incumbent-seed", llvm::cl::init(false)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selected)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp function = *selected;
    const bool productionScheduler = scheduleSpace == "production-scheduler";
    if (canonicalIncumbentSeed.getValue() && !productionScheduler) {
      module.emitError("canonical-incumbent-seed requires production-scheduler");
      return signalPassFailure();
    }
    const std::optional<DispatchPolicy> parsedDispatchPolicy =
        parseDispatchPolicy(dispatchPolicy.getValue());
    if (candidateFile.getValue().empty() || costFile.getValue().empty() ||
        outputFile.getValue().empty() || topK.getValue() <= 0 ||
        maxShapeCandidates.getValue() < 0 || maxExpandedNodes.getValue() <= 0 ||
        maxMilliseconds.getValue() < 0 || maxMakespan.getValue() < 0 ||
        stateMemoCapacity.getValue() < 0 ||
        frontierCheckpointEvery.getValue() < 0 ||
        legacyProductionCacheMaxIterations.getValue() < -1 ||
        legacyProductionCacheMaxIterations.getValue() >
            std::numeric_limits<int>::max() ||
        (!productionScheduler &&
         legacyProductionCacheMaxIterations.getValue() != -1) ||
        (legacyProductionCacheMaxIterations.getValue() != -1 &&
         legacyProductionCacheMaxIterations.getValue() != 64) ||
        shapeShardCount.getValue() <= 0 || shapeShardIndex.getValue() < 0 ||
        shapeShardIndex.getValue() >= shapeShardCount.getValue() ||
        (!rankingPruning.getValue() && (!productionScheduler || seedOnly.getValue() ||
          graphBoundOnly.getValue() || !globalIncumbentFile.getValue().empty())) ||
        (shapeShardCount.getValue() > 1 && (rankingPruning.getValue() || !productionScheduler)) ||
        (temporalModel == "all-integer-diagnostic" && maxMakespan.getValue() == 0) ||
        (temporalModel != "dispatch-order" && temporalModel != "all-integer-diagnostic" && temporalModel != "production-scheduler")
        || (productionScheduler != (temporalModel == "production-scheduler")) ||
        (productionScheduler && maxMakespan.getValue() != 0) ||
        sourceRepository.getValue().empty() || sourceCommit.getValue().empty() ||
        graphVariantId.getValue().empty() || !isFactoredSpace(candidateFile) ||
        (!globalIncumbentFile.getValue().empty() && stage.getValue().empty()) ||
        (!globalStageManifestFile.getValue().empty() &&
         globalIncumbentFile.getValue().empty()) ||
        (!warmStartShapes.getValue().empty() &&
         (!productionScheduler || graphBoundOnly.getValue())) ||
        ((!shapeFrontierSnapshot.getValue().empty() ||
          !resumeShapeFrontier.getValue().empty()) &&
         (!productionScheduler || seedOnly.getValue() ||
          graphBoundOnly.getValue())) ||
        (frontierCheckpointEvery.getValue() > 0 &&
         shapeFrontierSnapshot.getValue().empty() &&
         resumeShapeFrontier.getValue().empty()) ||
        (!resumeShapeFrontier.getValue().empty() &&
         !shapeFrontierSnapshot.getValue().empty() &&
         samePath(resumeShapeFrontier, shapeFrontierSnapshot)) ||
        (!resumeShapeFrontier.getValue().empty() &&
         !scoreCacheDirectory.getValue().empty()) ||
        ((!shapeFrontierSnapshot.getValue().empty() ||
          !resumeShapeFrontier.getValue().empty()) &&
         (!scoreCacheDirectory.getValue().empty() ||
          !scoreCachePeerDirectory.getValue().empty() ||
          scoreCachePeerProbe.getValue())) ||
        (seedOnly.getValue() && stage.getValue().empty()) ||
        (graphBoundOnly.getValue() &&
         (seedOnly.getValue() || globalIncumbentFile.getValue().empty())) ||
        !parsedDispatchPolicy || (scheduleSpace != "exact" &&
                                  scheduleSpace != "heuristic-placement" &&
                                  scheduleSpace != "production-scheduler") ||
        (parsedDispatchPolicy &&
         (productionScheduler ? *parsedDispatchPolicy == DispatchPolicy::AllReady
                              : *parsedDispatchPolicy == DispatchPolicy::CriticalPath))) {
      function.emitError() << "exact joint ranking requires a factored candidate "
                              "space, positive top-k/budgets/horizon, and "
                              "explicit provenance";
      return signalPassFailure();
    }
    const DispatchPolicy selectedDispatchPolicy = *parsedDispatchPolicy;
    if (samePath(candidateFile, outputFile) || samePath(costFile, outputFile) ||
        samePath(candidateFile, costFile) ||
        (!shapeFrontierSnapshot.getValue().empty() &&
         (samePath(shapeFrontierSnapshot, candidateFile) ||
          samePath(shapeFrontierSnapshot, costFile) ||
          samePath(shapeFrontierSnapshot, outputFile))) ||
        (!resumeShapeFrontier.getValue().empty() &&
         (samePath(resumeShapeFrontier, candidateFile) ||
          samePath(resumeShapeFrontier, costFile) ||
          samePath(resumeShapeFrontier, outputFile))) ||
        (!warmStartShapes.getValue().empty() &&
         (samePath(warmStartShapes, candidateFile) ||
          samePath(warmStartShapes, costFile) ||
          samePath(warmStartShapes, outputFile)))) {
      function.emitError() << "candidate, cost, and output paths must differ";
      return signalPassFailure();
    }

    FailureOr<SmallVector<TaskMetadata>> metadata =
        collectAnalyticalTaskMetadata(function, error);
    if (failed(metadata)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    const ::mlir::neura::Architecture &architecture =
        ::mlir::neura::getArchitecture();
    FactoredSpaceDescriptor descriptor;
    if (!parseFactoredSpace(candidateFile, function, *metadata, graphVariantId,
                            architecture, descriptor, error)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    TaskShapeCostCache costs;
    if (!costs.load(costFile, function.getSymName(), *metadata, sourceRepository,
                    sourceCommit, ::mlir::amoeba::getNeuraArchitectureSpecFile(),
                    graphVariantId, error)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    if (!hasTrustedMlPredictorSource(costFile, error)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    if (costs.candidateCountDecimal() != descriptor.candidateCount) {
      function.emitError() << "ML cost catalogue does not cover the exact "
                              "factored candidate count";
      return signalPassFailure();
    }
    const unsigned shapeOrdinalWidth = std::max<unsigned>(
        64u, 64u * static_cast<unsigned>(metadata->size() + 1));
    std::vector<WarmStartShapeHint> warmStartHints;
    if (!warmStartShapes.getValue().empty()) {
      if (!loadWarmStartShapeHints(warmStartShapes, descriptor,
                                   shapeOrdinalWidth, warmStartHints, error)) {
        function.emitError() << error;
        return signalPassFailure();
      }
      // A hint outside this invocation's exact shard would alter that shard's
      // incumbent without contributing to its coverage proof.  Reject it
      // before cache lookup so a malformed/foreign hint cannot be hidden by a
      // complete cache hit.
      const llvm::APInt warmFullShapeCount(shapeOrdinalWidth,
                                           descriptor.candidateCount, 10);
      const llvm::APInt warmShardCount(
          shapeOrdinalWidth,
          static_cast<uint64_t>(shapeShardCount.getValue()));
      const llvm::APInt warmShardIndex(
          shapeOrdinalWidth,
          static_cast<uint64_t>(shapeShardIndex.getValue()));
      const llvm::APInt warmShardBase =
          warmFullShapeCount.udiv(warmShardCount);
      const llvm::APInt warmShardRemainder =
          warmFullShapeCount.urem(warmShardCount);
      const llvm::APInt warmShardBegin =
          warmShardBase * warmShardIndex +
          (warmShardIndex.ult(warmShardRemainder) ? warmShardIndex
                                                  : warmShardRemainder);
      const llvm::APInt warmShardSize =
          warmShardBase + llvm::APInt(
                              shapeOrdinalWidth,
                              warmShardIndex.ult(warmShardRemainder) ? 1 : 0);
      const llvm::APInt warmShardEnd = warmShardBegin + warmShardSize;
      for (const WarmStartShapeHint &hint : warmStartHints) {
        if (hint.ordinal.ult(warmShardBegin) ||
            hint.ordinal.uge(warmShardEnd)) {
          error = "warm-start shape candidate " + hint.canonicalOrdinal +
                  " is outside the current shape partition";
          function.emitError() << error;
          return signalPassFailure();
        }
        for (size_t task = 0; task < hint.shapeIndices.size(); ++task) {
          const RectShape &shape =
              descriptor.shapesByTask[task][hint.shapeIndices[task]];
          TaskShapeChoice choice{(*metadata)[task].name,
                                 (*metadata)[task].tripCount, shape};
          const TaskShapeCost *cost = costs.get(choice, error);
          if (!cost) {
            function.emitError()
                << "warm-start shape " << hint.canonicalOrdinal
                << " has no current cost row: " << error;
            return signalPassFailure();
          }
          if (!cost->supported) {
            function.emitError()
                << "warm-start shape " << hint.canonicalOrdinal
                << " uses an unsupported current shape cost";
            return signalPassFailure();
          }
          if (!predictDuration(*cost, choice.tripCount, error)) {
            function.emitError()
                << "warm-start shape " << hint.canonicalOrdinal
                << " has an invalid current duration: " << error;
            return signalPassFailure();
          }
        }
      }
    }
    GlobalIncumbent globalIncumbent;
    if (!globalIncumbentFile.getValue().empty()) {
      globalIncumbent.requested = true;
      if (!loadGlobalIncumbent(
              globalIncumbentFile.getValue(), function.getSymName(),
              stage.getValue(), costs.sourceRepository(), costs.sourceCommit(),
              costs.architecturePath(), costs.architectureSchema(),
              costs.nameSpace(), globalStageManifestFile.getValue(),
              function, *metadata, costs, candidateFile, costFile,
              dispatchPolicy, temporalModel, scheduleSpace, globalIncumbent)) {
        function.emitError() << globalIncumbent.error;
        return signalPassFailure();
      }
    }

    FailureOr<TaskEdgeGraph> graph = buildTaskEdgeGraph(
        function, TaskEdgeGraphOptions{/*require_payload=*/true,
                                       /*read_control_predecessors=*/true},
        error);
    if (failed(graph)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    llvm::DenseMap<Operation *, unsigned> operationIndex;
    std::vector<Operation *> taskOperations(metadata->size(), nullptr);
    for (taskflow::TaskflowTaskOp task : graph->getTasks()) {
      auto it = llvm::find_if(*metadata, [&](const TaskMetadata &item) {
        return item.name == task.getTaskName();
      });
      if (it == metadata->end()) {
        function.emitError() << "edge graph task is absent from metadata";
        return signalPassFailure();
      }
      unsigned index = static_cast<unsigned>(it - metadata->begin());
      if (taskOperations[index]) {
        function.emitError() << "duplicate task operation in edge graph";
        return signalPassFailure();
      }
      taskOperations[index] = task.getOperation();
      operationIndex[task.getOperation()] = index;
    }
    if (llvm::any_of(taskOperations, [](Operation *op) { return op == nullptr; })) {
      function.emitError() << "edge graph does not cover all task metadata";
      return signalPassFailure();
    }

    std::vector<std::vector<unsigned>> predecessors(metadata->size());
    for (TaskEdge edge : graph->getEdges()) {
      auto producer = operationIndex.find(edge.producer.getOperation());
      auto consumer = operationIndex.find(edge.consumer.getOperation());
      if (producer == operationIndex.end() || consumer == operationIndex.end()) {
        function.emitError() << "edge endpoint is absent from task index";
        return signalPassFailure();
      }
      auto &list = predecessors[consumer->second];
      if (llvm::is_contained(list, producer->second))
        continue;
      list.push_back(producer->second);
    }
    const bool frontierRequested =
        !shapeFrontierSnapshot.getValue().empty() ||
        !resumeShapeFrontier.getValue().empty();
    const std::string frontierInputPath = resumeShapeFrontier.getValue();
    const std::string frontierOutputPath =
        !shapeFrontierSnapshot.getValue().empty()
            ? shapeFrontierSnapshot.getValue()
            : resumeShapeFrontier.getValue();
    const std::string frontierInputBindingPath =
        frontierInputPath.empty() ? std::string() : frontierInputPath + ".binding";
    const std::string frontierOutputBindingPath =
        frontierOutputPath.empty() ? std::string()
                                   : frontierOutputPath + ".binding";
    std::string frontierBinding;
    std::optional<ShapeFrontierSnapshot> loadedFrontier;
    if (frontierRequested) {
      if (!makeShapeFrontierBinding(
              function, *metadata, predecessors, descriptor, costs,
              candidateFile, costFile, dispatchPolicy, temporalModel,
              scheduleSpace, topK, maxMakespan, stateMemoCapacity,
              horizonProven, sourceRepository, sourceCommit, graphVariantId,
              stage, warmStartShapes, globalIncumbentFile,
              globalStageManifestFile, rankingPruning, graphBoundOnly, seedOnly,
              canonicalIncumbentSeed, shapeShardCount, shapeShardIndex,
              frontierBinding, error)) {
        function.emitError() << error;
        return signalPassFailure();
      }
      if (!frontierInputPath.empty()) {
        std::string persistedBinding;
        if (!readExactCacheInput(frontierInputBindingPath, persistedBinding,
                                 error) || persistedBinding != frontierBinding) {
          if (error.empty())
            error = "shape frontier binding witness differs byte-for-byte";
          function.emitError() << error;
          return signalPassFailure();
        }
        ShapeFrontierSnapshot parsedFrontier;
        if (!loadShapeFrontierSnapshot(frontierInputPath, frontierBinding,
                                       descriptor, *metadata, costs,
                                       function.getSymName(), graphVariantId,
                                       sourceRepository, sourceCommit,
                                       dispatchPolicy, temporalModel,
                                       scheduleSpace, topK.getValue(),
                                       canonicalIncumbentSeed.getValue(),
                                       parsedFrontier, error)) {
          function.emitError() << error;
          return signalPassFailure();
        }
        loadedFrontier = std::move(parsedFrontier);
      }
      if (frontierInputPath.empty() ||
          frontierOutputPath != frontierInputPath) {
        if (!writeShapeFrontierBinding(frontierOutputBindingPath,
                                       frontierBinding, error)) {
          function.emitError() << error;
          return signalPassFailure();
        }
      }
    }
    ExactScoreCacheLease scoreCache;
    // A global incumbent changes the admissible cutoff and therefore the
    // meaning of a complete local score.  Keep the cache disabled for these
    // queries until the cutoff certificate is part of the cache key; a
    // locally complete cache entry must never be published as a globally
    // pruned result.
    const bool scoreCacheRequested =
        !scoreCacheDirectory.getValue().empty() && !frontierRequested &&
        !globalIncumbent.requested && !seedOnly.getValue() &&
        rankingPruning.getValue() && shapeShardCount.getValue() == 1;
    const bool scoreCacheDisabledForGlobalCutoff =
        globalIncumbent.requested && !scoreCacheDirectory.getValue().empty();
    const bool scoreCachePeerRequested =
        !scoreCachePeerDirectory.getValue().empty();
    bool scoreCacheHit = false;
    bool scoreCacheMigrated = false;
    bool scoreCachePeerImported = false;
    bool scoreCachePeerPending = false;
    std::string scoreCachePeerStatus =
        scoreCachePeerRequested ? "not-probed" : "not-requested";
    std::string scoreCachePeerError;
    if (scoreCachePeerProbe.getValue() && !scoreCachePeerRequested) {
      function.emitError()
          << "score-cache-peer-probe requires score-cache-peer-directory";
      return signalPassFailure();
    }
    if (scoreCacheRequested) {
      std::string cacheKey;
      std::string legacyCacheKey;
      std::string cacheError;
      if (!makeExactScoreCacheKey(
              function, *metadata, predecessors, descriptor, costs, candidateFile,
              costFile, dispatchPolicy, temporalModel,
              scheduleSpace, topK, maxShapeCandidates, maxExpandedNodes,
              maxMilliseconds, maxMakespan, stateMemoCapacity, horizonProven,
              /*includeProductionSchedulerContract=*/true, cacheKey,
              cacheError)) {
        if (scoreCachePeerProbe.getValue()) {
          function.emitError()
              << "score-cache-peer-probe cannot establish an exact cache key: "
              << cacheError;
          return signalPassFailure();
        }
        function.emitRemark()
            << "exact score cache bypassed while building key: " << cacheError;
      } else if (productionScheduler &&
                 !makeExactScoreCacheKey(
                     function, *metadata, predecessors, descriptor, costs,
                     candidateFile, costFile, dispatchPolicy, temporalModel,
                     scheduleSpace, topK, maxShapeCandidates, maxExpandedNodes,
                     maxMilliseconds, maxMakespan, stateMemoCapacity,
                     horizonProven,
                     /*includeProductionSchedulerContract=*/false,
                     legacyCacheKey, cacheError)) {
        if (scoreCachePeerProbe.getValue()) {
          function.emitError()
              << "score-cache-peer-probe cannot establish the legacy cache key: "
              << cacheError;
          return signalPassFailure();
        }
        function.emitRemark()
            << "legacy exact score cache migration bypassed while building key: "
            << cacheError;
      } else {
        ExactScoreCacheMigrationOptions migration;
        migration.legacyKey = legacyCacheKey;
        migration.productionContract = kProductionSchedulerContract;
        std::string iterationError;
        int64_t effectiveIterations = 0;
        if (productionScheduler &&
            getProductionFixedPointIterations(function, effectiveIterations,
                                              iterationError)) {
          migration.effectiveProductionFixedPointIterations =
              effectiveIterations;
          migration.explicitLegacyMaxIterations =
              legacyProductionCacheMaxIterations.getValue();
          migration.proofReason =
              "complete-production-certificate; no failed production tuple; deterministic scheduler accepts first legal convergence/cycle before upper bound; all other exact-key bytes equal";
        }
        if (!scoreCache.begin(scoreCacheDirectory, cacheKey,
                             costs.architecturePath(), graphVariantId,
                             costs.sourceGraphId(), stage.getValue(), outputFile,
                             migration, cacheError)) {
        if (scoreCachePeerProbe.getValue()) {
          function.emitError()
              << "score-cache-peer-probe cannot acquire the exact cache lease: "
              << cacheError;
          return signalPassFailure();
        }
        function.emitRemark() << "exact score cache bypassed: " << cacheError;
      } else if (scoreCache.hit()) {
        scoreCacheHit = true;
        scoreCacheMigrated = scoreCache.migrated();
        return;
      } else if (scoreCachePeerRequested) {
        PeerProbeStatus peerStatus = importCompletePeerScore(
            function, scoreCachePeerDirectory, graphVariantId, candidateFile,
            costFile, costs.architecturePath(), sourceRepository, sourceCommit,
            dispatchPolicy, temporalModel, scheduleSpace,
            descriptor.candidateCount, topK, maxExpandedNodes, maxMilliseconds,
            maxMakespan, descriptor.maxCgrasPerTask, stateMemoCapacity,
            costs.sourceGraphId(), scoreCache, scoreCachePeerError);
        if (peerStatus == PeerProbeStatus::Imported) {
          scoreCacheHit = true;
          scoreCachePeerImported = true;
          scoreCachePeerStatus = "imported-complete";
          return;
        }
        scoreCachePeerPending = peerStatus == PeerProbeStatus::Pending;
        scoreCachePeerStatus =
            scoreCachePeerPending ? "pending" : "rejected";
        function.emitRemark() << "exact score cache peer not imported ("
                              << scoreCachePeerStatus << "): "
                              << scoreCachePeerError;
        if (scoreCachePeerProbe.getValue()) {
          scoreCache.discard();
          std::string probeError;
          if (!writePeerProbeOutput(
                  function, descriptor, costs, graphVariantId, outputFile,
                  scoreCacheRequested, scoreCachePeerRequested,
                  scoreCachePeerStatus, scoreCachePeerError, dispatchPolicy,
                  temporalModel, scheduleSpace, topK, stateMemoCapacity,
                  probeError)) {
            function.emitError() << probeError;
            return signalPassFailure();
          }
          return;
        }
      }
      }
    } else if (scoreCachePeerRequested) {
      scoreCachePeerStatus = "cache-disabled";
      if (scoreCachePeerProbe.getValue()) {
        function.emitError()
            << "score-cache-peer-probe requires score-cache-dir";
        return signalPassFailure();
      }
      function.emitRemark()
          << "score-cache-peer-directory requires score-cache-dir";
    }
    auto network = createInterTaskNetworkCommunicationModel(function, 4, 4, error);
    if (failed(network)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    ProductionFixedScheduleCommunication communication(**network,
                                                       taskOperations);

    const unsigned counterWidth = shapeOrdinalWidth;
    const llvm::APInt fullShapeCount(counterWidth, descriptor.candidateCount, 10);
    const llvm::APInt shardCount(counterWidth, static_cast<uint64_t>(shapeShardCount.getValue()));
    const llvm::APInt shardIndex(counterWidth, static_cast<uint64_t>(shapeShardIndex.getValue()));
    const llvm::APInt shardBase = fullShapeCount.udiv(shardCount);
    const llvm::APInt shardRemainder = fullShapeCount.urem(shardCount);
    const llvm::APInt shardBegin = shardBase * shardIndex +
        (shardIndex.ult(shardRemainder) ? shardIndex : shardRemainder);
    const llvm::APInt shardSize = shardBase + llvm::APInt(counterWidth, shardIndex.ult(shardRemainder) ? 1 : 0);
    const llvm::APInt shardEnd = shardBegin + shardSize;
    uint64_t partitionSkippedSubtrees = 0;
    llvm::APInt visitedShapes(counterWidth, 0);
    llvm::APInt validShapes(counterWidth, 0);
    uint64_t expandedNodes = 0;
    uint64_t productionSchedulerCalls = 0;
    std::set<std::string> productionSolvedShapes;
    std::set<std::string> frontierWarmupSolvedShapes;
    uint64_t overlapRejectedBranches = 0;
    uint64_t outOfBoundsRejectedBranches = 0;
    uint64_t schedulePaths = 0;
    uint64_t globalCutoffFilteredScheduleCallbacks = 0;
    uint64_t boundRejectedBranches = 0, seedSchedules = 0, expandedShapeNodes = 0;
    uint64_t geometryTimeBoundRejectedBranches = 0;
    // These counters cover only the additional fixed-prefix invocation of
    // the source-owned shape-domain necessary-condition helper.  They are
    // diagnostics; the cache key and candidate accounting remain unchanged.
    uint64_t prefixWindowBoundCalls = 0;
    uint64_t prefixWindowBoundRejected = 0;
    uint64_t canonicalSeedSchedulerCalls = 0;
    uint64_t canonicalSeedAccepted = 0;
    bool canonicalSeedContinuationComplete =
        !canonicalIncumbentSeed.getValue();
    uint64_t tieRestrictedCpBoundCalls = 0;
    uint64_t tieRestrictedCpBoundPruned = 0;
    uint64_t tieRestrictedCpNoEligible = 0;
    uint64_t tieRestrictedCpInvalid = 0;
    uint64_t stateMemoEntries = 0, stateMemoHits = 0,
             stateMemoPrunedSubtrees = 0, stateMemoCapacitySkips = 0;
    uint64_t optimisticEndsComputations = 0, optimisticEndsMemoHits = 0,
             optimisticEndsMemoEntries = 0, optimisticEndsMemoCapacitySkips = 0;
    bool optimisticEndsMemoEnabled = false;
    bool stateMemoEnabled = false, stateMemoCanonicalStateAvailable = false,
         stateMemoCutoffViolation = false;
    llvm::APInt boundPrunedShapes(counterWidth, 0);
    llvm::APInt domainPrunedShapes(counterWidth, 0);
    llvm::APInt globalBoundPrunedShapes(counterWidth, 0);
    bool globalCutoffShapePruned = false;
    bool globalCutoffProof = false;
    bool globalGraphPruned = false;
    bool globalShapeDomainContradiction = false;
    uint64_t globalShapeDomainContradictionTask =
        std::numeric_limits<uint64_t>::max();
    std::string globalCutoffProofKind = "none";
    int64_t graphLowerBound = 0;
    bool warmingUp = false;
    uint64_t warmStartHintsReplayed = 0;
    bool warmStartReplayFailed = false;
    bool replayingWarmStartHint = false;
    std::set<std::string> warmupKeys;
    uint64_t unsupportedShapes = 0;
    uint64_t scheduleSerial = 0;
    bool interrupted = false;
    bool fatalError = false;
    std::string interruptionReason;
    std::string failedProductionCandidateId;
    bool failedProductionWarmup = false;
    SmallVector<TaskShapeChoice> failedProductionChoices;
    SmallVector<TaskShapeCost> failedProductionCosts;
    std::string firstBlockingReason;
    const auto begun = std::chrono::steady_clock::now();
    std::vector<RankedSchedule> ranked;
    // These domains are populated after the verified warmup incumbent.  The
    // original descriptor remains untouched so candidate ordinals continue to
    // refer to the complete Cartesian alphabet.
    std::vector<std::vector<size_t>> shapeDomains(metadata->size());
    std::vector<std::vector<ShapeDomainOption>> shapeDomainOptions(
        metadata->size());
    int64_t shapeDomainBound = 0;
    uint64_t shapeDomainIterations = 0;
    uint64_t shapeDomainWindowRejectedOptions = 0;
    std::string shapeDomainStatus = "not-run";
    auto addCounter = [](uint64_t &total, uint64_t value) {
      if (value > std::numeric_limits<uint64_t>::max() - total)
        total = std::numeric_limits<uint64_t>::max();
      else
        total += value;
    };

    auto retain = [&](RankedSchedule candidate) {
      ranked.push_back(std::move(candidate));
      llvm::sort(ranked, [](const RankedSchedule &lhs, const RankedSchedule &rhs) {
        if (lhs.cycles != rhs.cycles)
          return lhs.cycles < rhs.cycles;
        int shapeOrder = compareDecimal(lhs.shapeOrdinal, rhs.shapeOrdinal);
        if (shapeOrder != 0)
          return shapeOrder < 0;
        return lhs.scheduleOrdinal < rhs.scheduleOrdinal;
      });
      if (ranked.size() > static_cast<size_t>(topK.getValue()))
        ranked.pop_back();
    };

    bool frontierResumeActive = loadedFrontier.has_value();
    bool frontierSnapshotWritten = false;
    uint64_t frontierRestoredSchedulerCalls = 0;
    uint64_t frontierRestoredExpandedShapeNodes = 0;
    if (loadedFrontier) {
      if (loadedFrontier->ranked.size() >
          static_cast<size_t>(topK.getValue())) {
        function.emitError()
            << "shape frontier contains more incumbent rows than top-k";
        return signalPassFailure();
      }
      if (loadedFrontier->savedMaxShapeCandidates > 0 &&
          maxShapeCandidates.getValue() > 0 &&
          maxShapeCandidates.getValue() <
              loadedFrontier->savedMaxShapeCandidates) {
        function.emitError()
            << "shape frontier resource limit max-shape-candidates was lowered";
        return signalPassFailure();
      }
      if (maxExpandedNodes.getValue() <
          loadedFrontier->savedMaxExpandedNodes ||
          (loadedFrontier->savedMaxMilliseconds > 0 &&
           maxMilliseconds.getValue() > 0 &&
           maxMilliseconds.getValue() <
               loadedFrontier->savedMaxMilliseconds)) {
        function.emitError()
            << "shape frontier resource limit was lowered";
        return signalPassFailure();
      }
      llvm::APInt restoredVisited(counterWidth, loadedFrontier->visitedShapes,
                                  10);
      llvm::APInt restoredBound(counterWidth,
                                loadedFrontier->boundPrunedShapes, 10);
      if (restoredVisited.ugt(shardSize)) {
        function.emitError()
            << "shape frontier visited count exceeds current shard";
        return signalPassFailure();
      }
      if (loadedFrontier->reason == "max-shape-candidates" &&
          maxShapeCandidates.getValue() > 0 &&
          restoredVisited.uge(llvm::APInt(
              counterWidth,
              static_cast<uint64_t>(maxShapeCandidates.getValue())))) {
        function.emitError()
            << "shape frontier max-shape-candidates budget cannot make progress";
        return signalPassFailure();
      }
      if (loadedFrontier->reason == "max-expanded-nodes" &&
          static_cast<uint64_t>(maxExpandedNodes.getValue()) <=
              loadedFrontier->expandedNodes +
                  loadedFrontier->expandedShapeNodes) {
        function.emitError()
            << "shape frontier max-expanded-nodes budget cannot make progress";
        return signalPassFailure();
      }
      visitedShapes = restoredVisited;
      boundPrunedShapes = restoredBound;
      expandedNodes = loadedFrontier->expandedNodes;
      expandedShapeNodes = loadedFrontier->expandedShapeNodes;
      productionSchedulerCalls = loadedFrontier->productionSchedulerCalls;
      frontierRestoredSchedulerCalls = productionSchedulerCalls;
      frontierRestoredExpandedShapeNodes = expandedShapeNodes;
      validShapes = llvm::APInt(counterWidth, loadedFrontier->validShapes);
      unsupportedShapes = loadedFrontier->unsupportedShapes;
      scheduleSerial = loadedFrontier->scheduleSerial;
      canonicalSeedContinuationComplete =
          loadedFrontier->canonicalSeedContinuationState == "complete";
      canonicalSeedSchedulerCalls = loadedFrontier->canonicalSeedSchedulerCalls;
      canonicalSeedAccepted = loadedFrontier->canonicalSeedAccepted;
      for (const std::string &shape : loadedFrontier->warmupSolvedShapes) {
        frontierWarmupSolvedShapes.insert(shape);
        productionSolvedShapes.insert(shape);
      }
      for (RankedSchedule &entry : loadedFrontier->ranked)
        retain(std::move(entry));
    }

    auto lastCheckpoint = begun;
    auto checkpoint = [&]() {
      std::string checkpointError;
      if (!writeAtomically(outputFile.getValue() + ".checkpoint.jsonl", [&](llvm::raw_ostream &stream) {
        llvm::json::Object header;
        header["record_type"] = "header"; header["schema"] = "amoeba-exact-joint-task-scores-v1";
        header["function"] = function.getSymName().str(); header["graph_variant_id"] = graphVariantId.getValue();
        header["source_repository"] = costs.sourceRepository().str(); header["source_commit"] = costs.sourceCommit().str();
        header["source_graph_id"] = costs.sourceGraphId().str(); header["architecture_path"] = costs.architecturePath().str();
        header["architecture_schema"] = costs.architectureSchema().str();
        header["cost_namespace"] = costs.nameSpace().str();
        header["cost_provenance_schema"] = "orbit-cost-provenance-v1";
        if (!stage.getValue().empty()) header["stage"] = stage.getValue();
        header["global_cutoff_requested"] = globalIncumbent.requested;
        if (globalIncumbent.requested) {
          header["global_cutoff_cycles"] = globalIncumbent.fifthCycles;
          header["global_cutoff_evidence_path"] = globalIncumbent.path;
        }
        header["communication_mode"] = "explicit"; header["predictor_source"] = kMlPredictorSource.str();
        header["representation"] = "factored-plus-exact-schedule"; header["candidate_count"] = descriptor.candidateCount;
        header["warm_start_shapes_requested"] = !warmStartShapes.getValue().empty();
        putDecimal(header, "warm_start_shape_hint_count",
                   std::to_string(warmStartHints.size()));
        putDecimal(header, "warm_start_shape_hint_replayed",
                   std::to_string(warmStartHintsReplayed));
        header["warm_start_shape_hint_replay_failed"] = warmStartReplayFailed;
        if (!warmStartShapes.getValue().empty())
          header["warm_start_shapes_path"] = warmStartShapes.getValue();
        header["top_k"] = topK.getValue(); header["grid_rows"] = 4; header["grid_cols"] = 4;
        header["max_cgras_per_task"] = static_cast<int64_t>(descriptor.maxCgrasPerTask);
        header["start_policy"] = scoreStartPolicy(temporalModel, scheduleSpace).str();
        header["dispatch_policy"] = dispatchPolicy.getValue(); header["max_makespan"] = maxMakespan.getValue();
        header["scheduler_backend"] = productionScheduler ? "orchestrate-tasks-on-accelerators" : "orbit-exact-enumerator";
        header["orbit_4x4_pruning"] = !productionScheduler;
        header["schedule_space"] = scheduleSpace.getValue(); header["placement_coverage_limited"] = scheduleSpace != "exact";
        header["production_ready"] = false; header["horizon_proven"] = false;
        header["state_memo_requested"] = scheduleSpace == "exact";
        header["state_memo_capacity"] = stateMemoCapacity.getValue();
        header["state_memo_external_cutoff_monotonic"] =
            scheduleSpace == "exact";
        header["global_shape_location_temporal_space"] = scheduleSpace != "production-scheduler";
        header["temporal_search_scope"] = scoreTemporalScope(temporalModel).str();
        header["cycle_starts_are_search_dimensions"] = temporalModel == "all-integer-diagnostic";
        header["shape_domain_pruning"] =
            productionScheduler ? "dependency-release-tail-cost-only"
                                : "dependency-release-tail-and-energetic-work";
        header["shape_domain_status"] = shapeDomainStatus;
        header["shape_domain_bound_cycles"] = shapeDomainBound;
        header["shape_domain_fixed_point_iterations"] =
            static_cast<int64_t>(shapeDomainIterations);
        header["shape_domain_window_rejected_options"] =
            static_cast<int64_t>(shapeDomainWindowRejectedOptions);
        llvm::json::Array domainSummary;
        for (unsigned task = 0; task < shapeDomains.size(); ++task) {
          llvm::json::Object item;
          item["task"] = (*metadata)[task].name;
          item["original_shape_count"] = static_cast<int64_t>(
              descriptor.shapesByTask[task].size());
          item["retained_shape_count"] =
              static_cast<int64_t>(shapeDomains[task].size());
          int64_t minimumArea = std::numeric_limits<int64_t>::max();
          int64_t maximumArea = 0;
          llvm::json::Array retained;
          for (size_t ordinal : shapeDomains[task]) {
            const RectShape &shape = descriptor.shapesByTask[task][ordinal];
            const int64_t area = shape.cgraCount();
            minimumArea = std::min(minimumArea, area);
            maximumArea = std::max(maximumArea, area);
            llvm::json::Object shapeRecord;
            shapeRecord["original_shape_ordinal"] =
                static_cast<int64_t>(ordinal);
            shapeRecord["rows"] = shape.rows;
            shapeRecord["cols"] = shape.cols;
            shapeRecord["mapper_tile_rows"] = shape.mapperRows;
            shapeRecord["mapper_tile_cols"] = shape.mapperCols;
            shapeRecord["cgra_count"] = area;
            retained.push_back(std::move(shapeRecord));
          }
          item["retained_shapes"] = std::move(retained);
          item["retained_area_min"] =
              minimumArea == std::numeric_limits<int64_t>::max()
                  ? 0
                  : minimumArea;
          item["retained_area_max"] = maximumArea;
          domainSummary.push_back(std::move(item));
        }
        header["shape_domain_summary"] = std::move(domainSummary);
        header["ranking_pruning"] = rankingPruning.getValue();
        header["tie_restricted_cp_scope"] =
            "local-equal-cycle-canonical-ordinal";
        putDecimal(header, "tie_restricted_cp_bound_calls",
                   std::to_string(tieRestrictedCpBoundCalls));
        putDecimal(header, "tie_restricted_cp_bound_pruned",
                   std::to_string(tieRestrictedCpBoundPruned));
        putDecimal(header, "tie_restricted_cp_no_eligible",
                   std::to_string(tieRestrictedCpNoEligible));
        putDecimal(header, "tie_restricted_cp_invalid",
                   std::to_string(tieRestrictedCpInvalid));
        header["shape_shard_count"] = shapeShardCount.getValue();
        header["shape_shard_index"] = shapeShardIndex.getValue();
        putDecimal(header, "shape_partition_begin", exactShapeIndexString(shardBegin));
        putDecimal(header, "shape_partition_end", exactShapeIndexString(shardEnd));
        putDecimal(header, "shape_partition_candidate_count", exactShapeIndexString(shardSize));
        putDecimal(header, "partition_skipped_subtrees", std::to_string(partitionSkippedSubtrees));
        header["elapsed_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
        addProductionCoverage(header, temporalModel, scheduleSpace, dispatchPolicy, function);
        writeJsonLine(stream, std::move(header));
        for (const auto &entry : ranked) writeJsonLine(stream, llvm::json::Object(entry.record));
        llvm::json::Object footer;
        footer["record_type"] = "footer"; footer["schema"] = "amoeba-exact-joint-task-scores-v1";
        footer["status"] = "bounded_incomplete"; footer["incomplete"] = true;
        footer["reason"] = "search-in-progress"; footer["top_k_certified"] = false;
        footer["scored_count"] = static_cast<int64_t>(ranked.size()); footer["top_k_requested"] = topK.getValue();
        footer["warm_start_shapes_requested"] = !warmStartShapes.getValue().empty();
        putDecimal(footer, "warm_start_shape_hint_count",
                   std::to_string(warmStartHints.size()));
        putDecimal(footer, "warm_start_shape_hint_replayed",
                   std::to_string(warmStartHintsReplayed));
        footer["warm_start_shape_hint_replay_failed"] = warmStartReplayFailed;
        putDecimal(footer, "geometry_time_bound_rejected_branches", std::to_string(geometryTimeBoundRejectedBranches));
        putDecimal(footer, "prefix_window_bound_calls",
                   std::to_string(prefixWindowBoundCalls));
        putDecimal(footer, "prefix_window_bound_rejected",
                   std::to_string(prefixWindowBoundRejected));
        footer["tie_restricted_cp_scope"] =
            "local-equal-cycle-canonical-ordinal";
        putDecimal(footer, "tie_restricted_cp_bound_calls",
                   std::to_string(tieRestrictedCpBoundCalls));
        putDecimal(footer, "tie_restricted_cp_bound_pruned",
                   std::to_string(tieRestrictedCpBoundPruned));
        putDecimal(footer, "tie_restricted_cp_no_eligible",
                   std::to_string(tieRestrictedCpNoEligible));
        putDecimal(footer, "tie_restricted_cp_invalid",
                   std::to_string(tieRestrictedCpInvalid));
        footer["ranking_certificate"] = "best-found-uncertified";
        putDecimal(footer, "production_scheduler_calls", std::to_string(productionSchedulerCalls));
        footer["canonical_incumbent_seed_enabled"] =
            canonicalIncumbentSeed.getValue();
        footer["canonical_seed_continuation_state"] =
            canonicalIncumbentSeed.getValue()
                ? (canonicalSeedContinuationComplete ? "complete"
                                                      : "incomplete")
                : "disabled";
        footer["canonical_seed_scheduler_calls"] =
            static_cast<int64_t>(canonicalSeedSchedulerCalls);
        footer["canonical_seed_accepted"] =
            static_cast<int64_t>(canonicalSeedAccepted);
        footer["orbit_4x4_pruning"] = !productionScheduler;
        footer["state_memo_requested"] = scheduleSpace == "exact";
        footer["state_memo_enabled"] = stateMemoEnabled;
        footer["state_memo_canonical_state_available"] =
            stateMemoCanonicalStateAvailable;
        footer["state_memo_external_cutoff_monotonic"] =
            scheduleSpace == "exact";
        footer["state_memo_cutoff_violation"] = stateMemoCutoffViolation;
        footer["state_memo_capacity"] = stateMemoCapacity.getValue();
        footer["state_memo_entries"] = static_cast<int64_t>(stateMemoEntries);
        footer["state_memo_hits"] = static_cast<int64_t>(stateMemoHits);
        footer["state_memo_pruned_subtrees"] =
            static_cast<int64_t>(stateMemoPrunedSubtrees);
        footer["state_memo_capacity_skips"] =
            static_cast<int64_t>(stateMemoCapacitySkips);
        footer["optimistic_ends_memo_enabled"] = optimisticEndsMemoEnabled;
        putDecimal(footer, "optimistic_ends_computations", std::to_string(optimisticEndsComputations));
        putDecimal(footer, "optimistic_ends_memo_hits", std::to_string(optimisticEndsMemoHits));
        putDecimal(footer, "optimistic_ends_memo_entries", std::to_string(optimisticEndsMemoEntries));
        putDecimal(footer, "optimistic_ends_memo_capacity_skips", std::to_string(optimisticEndsMemoCapacitySkips));
        putDecimal(footer, "candidate_count", descriptor.candidateCount);
        putDecimal(footer, "shape_candidates_visited", exactShapeIndexString(visitedShapes));
        putDecimal(footer, "domain_pruned_shape_candidates",
                   exactShapeIndexString(domainPrunedShapes));
        putDecimal(footer, "bound_pruned_shape_candidates", exactShapeIndexString(boundPrunedShapes));
        footer["shape_domain_status"] = shapeDomainStatus;
        footer["shape_domain_bound_cycles"] = shapeDomainBound;
        footer["shape_domain_fixed_point_iterations"] =
            static_cast<int64_t>(shapeDomainIterations);
        footer["shape_domain_window_rejected_options"] =
            static_cast<int64_t>(shapeDomainWindowRejectedOptions);
        putDecimal(footer, "global_cutoff_pruned_shape_candidates",
                   exactShapeIndexString(globalBoundPrunedShapes));
        footer["global_cutoff_requested"] = globalIncumbent.requested;
        footer["global_cutoff_proven"] = false;
        putDecimal(footer, "global_cutoff_filtered_schedule_callbacks",
                   std::to_string(globalCutoffFilteredScheduleCallbacks));
        putDecimal(footer, "expanded_schedule_nodes", std::to_string(expandedNodes));
        footer["ranking_pruning"] = rankingPruning.getValue();
        footer["shape_shard_count"] = shapeShardCount.getValue();
        footer["shape_shard_index"] = shapeShardIndex.getValue();
        putDecimal(footer, "shape_partition_begin", exactShapeIndexString(shardBegin));
        putDecimal(footer, "shape_partition_end", exactShapeIndexString(shardEnd));
        putDecimal(footer, "shape_partition_candidate_count", exactShapeIndexString(shardSize));
        putDecimal(footer, "partition_skipped_subtrees", std::to_string(partitionSkippedSubtrees));
        footer["elapsed_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
        addProductionCoverage(footer, temporalModel, scheduleSpace, dispatchPolicy, function);
        writeJsonLine(stream, std::move(footer));
        return true;
      }, checkpointError)) llvm::errs() << "checkpoint write failed: " << checkpointError << "\n";
      llvm::errs() << "[JointScheduling] shape tuples=" << exactShapeIndexString(visitedShapes)
                   << " proven-pruned=" << exactShapeIndexString(boundPrunedShapes)
                   << " schedule nodes=" << expandedNodes << " topK=" << ranked.size()
                   << (ranked.empty() ? 0 : ranked.front().cycles) << "\n";
      lastCheckpoint = std::chrono::steady_clock::now();
    };
    // Query every task/mapper shape once, before any bound-based pruning.
    // Unknown cost support is a coverage blocker, never a legality proof.
    std::vector<std::vector<int64_t>> edgeLowerBound(metadata->size(), std::vector<int64_t>(metadata->size(), 0));
    for (unsigned i = 0; i < metadata->size(); ++i)
      for (unsigned p : predecessors[i])
        if (!communication.getTransferLowerBound(p, i, edgeLowerBound[i][p], error)) {
          function.emitError() << error; return signalPassFailure();
        }
    std::vector<std::vector<int64_t>> durations(metadata->size());
    std::vector<std::vector<uint8_t>> supportedShapes(metadata->size());
    std::vector<int64_t> minimumDuration(metadata->size(), std::numeric_limits<int64_t>::max());
    std::vector<unsigned __int128> minimumWork(metadata->size(), ~static_cast<unsigned __int128>(0));
    std::vector<std::vector<size_t>> preferredShapes(metadata->size());
    SmallVector<size_t> fastest;
    for (unsigned i = 0; i < metadata->size(); ++i) {
      for (unsigned j = 0; j < descriptor.shapesByTask[i].size(); ++j) {
        TaskShapeChoice choice{(*metadata)[i].name, (*metadata)[i].tripCount,
                               descriptor.shapesByTask[i][j]};
        const TaskShapeCost *cost = costs.get(choice, error);
        if (!cost) { function.emitError() << error; return signalPassFailure(); }
        auto duration = cost->supported ? predictDuration(*cost, choice.tripCount, error)
                                       : std::optional<int64_t>(std::numeric_limits<int64_t>::max());
        if (!duration) { function.emitError() << error; return signalPassFailure(); }
        if (!cost->supported) firstBlockingReason = "unsupported-task-shape-cost";
        durations[i].push_back(*duration);
        supportedShapes[i].push_back(cost->supported ? 1 : 0);
        shapeDomainOptions[i].push_back(
            ShapeDomainOption{j, *duration,
                              static_cast<unsigned>(choice.shape.cgraCount()),
                              cost->supported});
        shapeDomains[i].push_back(j);
        minimumDuration[i] = std::min(minimumDuration[i], *duration);
        minimumWork[i] = std::min(minimumWork[i], static_cast<unsigned __int128>(*duration) * choice.shape.cgraCount());
        preferredShapes[i].push_back(j);
      }
      std::stable_sort(preferredShapes[i].begin(), preferredShapes[i].end(),
          [&](size_t a, size_t b) { return durations[i][a] < durations[i][b]; });
      if (llvm::any_of(supportedShapes[i],
                       [](uint8_t supported) { return supported == 0; })) {
        minimumDuration[i] = 0;
        minimumWork[i] = 0;
      }
      fastest.push_back(preferredShapes[i].front());
    }
    // Original suffix counts preserve the stable factored candidate ordinal.
    // A separate surviving-domain suffix is used only for exact proof counts
    // after shape-domain propagation.
    std::vector<llvm::APInt> originalSuffixCounts(
        metadata->size() + 1, llvm::APInt(counterWidth, 1));
    // This is an optimistic graph-wide lower bound: it uses the fastest
    // supported duration for every task and the communication model's proven
    // edge lower bounds, while ignoring placement/resource contention.  It is
    // therefore safe for a strict incumbent cutoff and never a candidate cap.
    if (globalIncumbent.requested) {
      // A missing optional end means that an intermediate checked add
      // overflowed (or the dependency graph was cyclic).  Such a value is
      // unavailable evidence; it must not be saturated to INT64_MAX and then
      // treated as a strict cutoff proof.
      std::vector<std::optional<int64_t>> optimisticEnds(metadata->size());
      std::vector<bool> visiting(metadata->size(), false);
      std::function<std::optional<int64_t>(unsigned)> earliestEnd =
          [&](unsigned task) -> std::optional<int64_t> {
        if (optimisticEnds[task])
          return optimisticEnds[task];
        if (visiting[task])
          return std::nullopt;
        visiting[task] = true;
        int64_t start = 0;
        for (unsigned predecessor : predecessors[task]) {
          std::optional<int64_t> predecessorEnd = earliestEnd(predecessor);
          int64_t delay = edgeLowerBound[task][predecessor];
          if (!predecessorEnd || delay < 0 ||
              delay > std::numeric_limits<int64_t>::max() - *predecessorEnd) {
            visiting[task] = false;
            return std::nullopt;
          }
          start = std::max(start, *predecessorEnd + delay);
        }
        int64_t duration = minimumDuration[task];
        if (duration < 0 ||
            duration > std::numeric_limits<int64_t>::max() - start) {
          visiting[task] = false;
          return std::nullopt;
        }
        optimisticEnds[task] = start + duration;
        visiting[task] = false;
        return optimisticEnds[task];
      };
      unsigned __int128 totalMinimumWork = 0;
      bool totalMinimumWorkFinite = true;
      for (unsigned task = 0; task < metadata->size(); ++task) {
        if (std::optional<int64_t> end = earliestEnd(task))
          graphLowerBound = std::max(graphLowerBound, *end);
        if (minimumWork[task] != ~static_cast<unsigned __int128>(0))
          totalMinimumWorkFinite =
              ::mlir::amoeba::neura::joint_scheduling::detail::addUnsignedWide(
                  totalMinimumWork, minimumWork[task], totalMinimumWork) &&
              totalMinimumWorkFinite;
      }
      if (!productionScheduler && totalMinimumWorkFinite) {
        unsigned __int128 roundedNumerator = 0;
        if (::mlir::amoeba::neura::joint_scheduling::detail::addUnsignedWide(
                totalMinimumWork, 15, roundedNumerator)) {
          const unsigned __int128 roundedWork = roundedNumerator / 16;
          if (roundedWork <= static_cast<unsigned __int128>(
                                 std::numeric_limits<int64_t>::max()))
            graphLowerBound = std::max(
                graphLowerBound, static_cast<int64_t>(roundedWork));
        }
      }
    }
    for (size_t i = metadata->size(); i > 0; --i)
      originalSuffixCounts[i - 1] =
          originalSuffixCounts[i] * descriptor.shapesByTask[i - 1].size();
    std::vector<ShapeFrontierFrameSnapshot> frontierStack;
    std::function<bool(StringRef)> writeFrontierSnapshot;
    uint64_t lastFrontierCheckpointNodes = expandedShapeNodes;
    auto scoreTuple = [&](const llvm::APInt &shapeOrdinal,
                          ArrayRef<size_t> shapeIndices) -> bool {
          if (!warmingUp && maxShapeCandidates.getValue() > 0 &&
              visitedShapes.uge(llvm::APInt(
                  counterWidth, static_cast<uint64_t>(maxShapeCandidates)))) {
            interrupted = true;
            interruptionReason = "max-shape-candidates";
            if (writeFrontierSnapshot)
              writeFrontierSnapshot(interruptionReason);
            return false;
          }
          if (maxMilliseconds.getValue() > 0 &&
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - begun)
                      .count() >= maxMilliseconds.getValue()) {
            interrupted = true;
            interruptionReason = "max-milliseconds";
            if (writeFrontierSnapshot)
              writeFrontierSnapshot(interruptionReason);
            return false;
          }
          if (!warmingUp) ++visitedShapes;

          SmallVector<TaskShapeChoice> choices;
          SmallVector<TaskShapeCost> shapeCosts;
          choices.reserve(metadata->size());
          shapeCosts.reserve(metadata->size());
          std::vector<ExactScheduleTask> exactTasks;
          exactTasks.reserve(metadata->size());
          bool unsupportedShape = false;
          for (auto [taskIndex, shapeIndex] : llvm::enumerate(shapeIndices)) {
            const RectShape &shape = descriptor.shapesByTask[taskIndex][shapeIndex];
            TaskShapeChoice choice{(*metadata)[taskIndex].name,
                                   (*metadata)[taskIndex].tripCount, shape};
            const TaskShapeCost *cost = costs.get(choice, error);
            if (!cost) {
              fatalError = true;
              return false;
            }
            if (!cost->supported) {
              unsupportedShape = true;
              if (firstBlockingReason.empty())
                firstBlockingReason = "unsupported-task-shape-cost";
              continue;
            }
            auto duration = predictDuration(*cost, choice.tripCount, error);
            if (!duration) {
              fatalError = true;
              return false;
            }
            choices.push_back(choice);
            shapeCosts.push_back(*cost);
            ExactScheduleTask item;
            item.name = choice.task;
            item.rows = static_cast<int>(shape.rows);
            item.cols = static_cast<int>(shape.cols);
            item.duration = *duration;
            item.predecessors = predecessors[taskIndex];
            exactTasks.push_back(std::move(item));
          }
          if (unsupportedShape) {
            if (!warmingUp) ++unsupportedShapes;
            return true;
          }
          if (choices.size() != metadata->size()) {
            fatalError = true;
            error = "exact ranker did not construct one choice per task";
            return false;
          }
          if (!warmingUp) ++validShapes;

          const std::string shapeId = exactShapeIndexString(shapeOrdinal);
          auto emitSchedule = [&](const std::vector<ExactSchedulePlacement> &schedule,
                  const std::vector<unsigned> &order,
                  const FixedScheduleScore &score) {
                if (!productionScheduler && selectedDispatchPolicy == DispatchPolicy::Fixed &&
                    !isCanonicalFixedDispatch(order, exactTasks))
                  return;
                std::string scheduleKey = shapeId;
                for (const auto &p : schedule)
                  scheduleKey += ";" + std::to_string(p.task) + ":" + std::to_string(p.row) + "," + std::to_string(p.col) + "@" + std::to_string(p.start) + "-" + std::to_string(p.end);
                for (unsigned task : order) scheduleKey += "/" + std::to_string(task);
                if (warmingUp) warmupKeys.insert(scheduleKey);
                else if (warmupKeys.count(scheduleKey)) return;
                if (globalIncumbent.requested &&
                    score.predictedMakespan > globalIncumbent.fifthCycles) {
                  ++globalCutoffFilteredScheduleCallbacks;
                  return;
                }
                ++schedulePaths;
                llvm::json::Object record;
                record["record_type"] = "score";
                record["valid"] = true;
                record["scheduler_backend"] = productionScheduler ? "orchestrate-tasks-on-accelerators" : "orbit-exact-enumerator";
                record["schema"] = "amoeba-exact-joint-task-scores-v1";
                const uint64_t scheduleOrdinal = scheduleSerial++;
                record["candidate_id"] = "shape-" + shapeId + "/schedule-" +
                                         std::to_string(scheduleOrdinal);
                record["shape_candidate_id"] = "candidate-" + shapeId;
                record["predicted_whole_program_cycles"] =
                    score.predictedMakespan;
                record["score_source"] =
                    "ml-ii-startup-plus-production-explicit-communication";
                record["task_latency_provenance"] =
                    "source-owned-ml-ii-startup-trip-count-contract";
                if (replayingWarmStartHint)
                  record["warm_start_replayed_with_current_scheduler"] = true;
                record["task_costs"] = makeTaskCosts(choices, shapeCosts);
                llvm::json::Array entries;
                for (const auto &placement : schedule) {
                  llvm::json::Object entry;
                  entry["task"] = exactTasks[placement.task].name;
                  entry["row"] = placement.row;
                  entry["col"] = placement.col;
                  entry["rows"] = exactTasks[placement.task].rows;
                  entry["cols"] = exactTasks[placement.task].cols;
                  entry["start_cycle"] = placement.start;
                  entry["end_cycle"] = placement.end;
                  entry["idle_cycles"] = placement.idleCycles;
                  entries.push_back(std::move(entry));
                }
                record["task_schedule"] = std::move(entries);
                llvm::json::Array orderRecords;
                std::string key = "shape-" + shapeId + "/schedule-" +
                                  std::to_string(scheduleOrdinal);
                for (unsigned task : order) {
                  orderRecords.push_back(exactTasks[task].name);

                }
                record["dispatch_order"] = std::move(orderRecords);
                record["replayed_communication_edges"] =
                    static_cast<int64_t>(score.replayedEdges);
                retain({std::move(key), shapeId, scheduleOrdinal,
                        score.predictedMakespan, std::move(record)});
              };
          if (productionScheduler) {
            // Warmup tuples are actual pass results and need not be scheduled twice.
            if (productionSolvedShapes.count(shapeId))
              return true;
            SmallVector<int64_t> predictedDurations;
            for (const auto &task : exactTasks)
              predictedDurations.push_back(task.duration);
            ProductionScheduledResult scheduled;
            ++productionSchedulerCalls;
            if (!scheduleWithAmoebaProductionPass(module, function, choices,
                    predictedDurations, "candidate-" + shapeId,
                    selectedDispatchPolicy == DispatchPolicy::Fixed,
                    scheduled, error)) {
              interrupted = true;
              interruptionReason = error.empty() ? "production-scheduler-failed" : error;
              failedProductionCandidateId = "candidate-" + shapeId;
              failedProductionWarmup = warmingUp;
              failedProductionChoices = choices;
              failedProductionCosts = shapeCosts;
              return false;
            }
            productionSolvedShapes.insert(shapeId);
            if (warmingUp) {
              ++seedSchedules;
              frontierWarmupSolvedShapes.insert(shapeId);
            }
            FixedScheduleScore score;
            score.predictedMakespan = scheduled.makespan;
            score.replayedEdges = scheduled.replayedEdges;
            emitSchedule(scheduled.placements, scheduled.order, score);
            return true;
          }
          uint64_t remainingNodes =
              expandedNodes >= static_cast<uint64_t>(maxExpandedNodes)
                  ? 0
                  : static_cast<uint64_t>(maxExpandedNodes) - expandedNodes - std::min(
                        expandedShapeNodes, static_cast<uint64_t>(maxExpandedNodes) - expandedNodes);
          if (remainingNodes == 0) {
            interrupted = true;
            interruptionReason = "max-expanded-nodes";
            return false;
          }
          uint64_t remainingMs = 0;
          if (maxMilliseconds.getValue() > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - begun).count();
            if (elapsed >= maxMilliseconds.getValue()) {
              interrupted = true;
              interruptionReason = "max-milliseconds";
              return false;
            }
            remainingMs = static_cast<uint64_t>(maxMilliseconds - elapsed);
          }
          ExactScheduleLimits limits{remainingNodes, remainingMs};
          limits.rankingTopK = topK.getValue();
          limits.seedIncumbents = true;
          limits.seedPreferences = scheduleSpace == "heuristic-placement" ? 1 : 16;
          // The production score pass owns the non-increasing shape/top-K
          // cutoff.  Warmup and the single-placement diagnostic do not enable
          // state reuse; the optional communication API also fails closed.
          limits.enableStateMemo =
              !warmingUp && scheduleSpace == "exact" &&
              communication.supportsCanonicalReservationState();
          limits.stateMemoCapacity =
              static_cast<uint64_t>(stateMemoCapacity.getValue());
          limits.externalUpperBoundMonotonic = limits.enableStateMemo;
          ExactScheduleResult result;
          CommunicationScheduleCertificate certificate;
          bool accepted = enumerateCommunicationScheduleSpace(
              4, 4, exactTasks, limits,
              (warmingUp || scheduleSpace == "heuristic-placement")
                  ? CommunicationStartPolicy::SeedOnly
                  : (temporalModel == "dispatch-order" ? CommunicationStartPolicy::OrderDerived : CommunicationStartPolicy::AllInteger),
              maxMakespan, communication,
              emitSchedule,
              result, certificate, error,
              selectedDispatchPolicy == DispatchPolicy::Fixed
                  ? CommunicationDispatchPolicy::CanonicalSmallestReady
                  : CommunicationDispatchPolicy::AllReady,
              [&]() {
                int64_t cutoff = maxMakespan.getValue() > 0
                                     ? maxMakespan.getValue()
                                     : std::numeric_limits<int64_t>::max();
                if (ranked.size() >= static_cast<size_t>(topK.getValue())) {
                  // The local bound retains the established deterministic
                  // shape/schedule tie policy.  A global incumbent has no
                  // graph-local candidate key, so it is inclusive: cycles
                  // equal to the fifth incumbent are always searched.
                  const int64_t localCutoff =
                      ranked.back().cycles -
                      (compareDecimal(shapeId, ranked.back().shapeOrdinal) >= 0
                           ? 1
                           : 0);
                  cutoff = std::min(cutoff, localCutoff);
                }
                if (globalIncumbent.requested)
                  cutoff = std::min(cutoff, globalIncumbent.fifthCycles);
                return cutoff;
              });
          addCounter(expandedNodes, result.expandedNodes);
          addCounter(boundRejectedBranches, result.boundRejectedBranches);
          addCounter(geometryTimeBoundRejectedBranches, result.geometryTimeBoundRejectedBranches);
          addCounter(seedSchedules, result.seedSchedules);
          addCounter(overlapRejectedBranches, result.overlapRejectedBranches);
          addCounter(outOfBoundsRejectedBranches,
                     result.outOfBoundsRejectedBranches);
          addCounter(stateMemoEntries, result.stateMemoEntries);
          addCounter(stateMemoHits, result.stateMemoHits);
          addCounter(stateMemoPrunedSubtrees, result.stateMemoPrunedSubtrees);
          addCounter(stateMemoCapacitySkips, result.stateMemoCapacitySkips);
          addCounter(optimisticEndsComputations, result.optimisticEndsMemoComputations);
          addCounter(optimisticEndsMemoHits, result.optimisticEndsMemoHits);
          addCounter(optimisticEndsMemoEntries, result.optimisticEndsMemoEntries);
          addCounter(optimisticEndsMemoCapacitySkips, result.optimisticEndsMemoCapacitySkips);
          optimisticEndsMemoEnabled =
              optimisticEndsMemoEnabled || result.optimisticEndsMemoEnabled;
          stateMemoEnabled = stateMemoEnabled || result.stateMemoEnabled;
          stateMemoCanonicalStateAvailable =
              stateMemoCanonicalStateAvailable ||
              result.stateMemoCanonicalStateAvailable;
          stateMemoCutoffViolation =
              stateMemoCutoffViolation || result.stateMemoCutoffViolation;
          if (!accepted) {
            interrupted = true;
            interruptionReason = error.empty() ? "communication-search-failed" : error;
            return false;
          }
          if (!result.complete) {
            interrupted = true;
            interruptionReason = result.reason.empty() ? "schedule-budget" : result.reason;
            return false;
          }
          return true;
        };
    // Replay only the current tuple ordinals from the immutable hint file.
    // scoreTuple invokes the production adapter and current cost catalogue,
    // so an old row can improve the incumbent but can never certify itself.
    if (!frontierResumeActive && !warmStartHints.empty()) {
      warmingUp = true;
      for (const WarmStartShapeHint &hint : warmStartHints) {
        replayingWarmStartHint = true;
        if (!scoreTuple(hint.ordinal, hint.shapeIndices)) {
          replayingWarmStartHint = false;
          warmStartReplayFailed = true;
          break;
        }
        replayingWarmStartHint = false;
        ++warmStartHintsReplayed;
      }
      warmingUp = false;
    }
    llvm::APInt fastestOrdinal(counterWidth, 0);
    for (unsigned i = 0; i < fastest.size(); ++i)
      fastestOrdinal += originalSuffixCounts[i + 1] * fastest[i];
    auto seedShapeTuples = [&]() {
      if (!scoreTuple(fastestOrdinal, fastest))
        return false;
      if (!productionScheduler)
        return true;
      for (unsigned task = 0; task < fastest.size(); ++task) {
        for (size_t alternative : preferredShapes[task]) {
          if (alternative == fastest[task])
            continue;
          SmallVector<size_t> tuple(fastest);
          tuple[task] = alternative;
          llvm::APInt ordinal(counterWidth, 0);
          for (unsigned i = 0; i < tuple.size(); ++i)
            ordinal += originalSuffixCounts[i + 1] * tuple[i];
          if (!scoreTuple(ordinal, tuple))
            return false;
          if (ranked.size() >= static_cast<size_t>(topK.getValue()))
            return true;
        }
      }
      return true;
    };
    bool traversed = !warmStartReplayFailed;
    bool seedOnlyTraversal = false;
    if (frontierResumeActive) {
      // The snapshot already contains the source-owned incumbent and the
      // exact DFS prefix.  Replaying warm rows or seed tuples would duplicate
      // scheduler calls and would advance schedule ordinals, so resume enters
      // domain reconstruction directly below.
      warmingUp = false;
    } else if (warmStartReplayFailed) {
      // scoreTuple already recorded the source-owned failure and its complete
      // current task-cost context.  Preserve the normal incomplete result;
      // never fall back to an old hint's score or continue as if it succeeded.
      checkpoint();
    } else if (globalIncumbent.requested &&
        graphLowerBound > globalIncumbent.fifthCycles) {
      // Every legal shape/schedule has the same optimistic dependency/work
      // lower bound above the incumbent.  The complete Cartesian space is
      // therefore pruned by proof, without manufacturing a local top five.
      globalGraphPruned = true;
      globalCutoffProof = true;
      globalCutoffProofKind = "numeric-graph-lower-bound";
      llvm::APInt totalShapeCount(counterWidth, descriptor.candidateCount, 10);
      boundPrunedShapes = totalShapeCount;
      globalBoundPrunedShapes = totalShapeCount;
    } else if (graphBoundOnly.getValue()) {
      // Bound-only mode has no warmup schedule to justify a local domain
      // cutoff, but a validated global fifth is sufficient for the shared
      // optimistic domain proof.  It performs no schedule or ML work.
      shapeDomainBound = globalIncumbent.fifthCycles;
      ShapeDomainPropagationResult propagation = propagateShapeDomains(
          shapeDomainOptions, predecessors, edgeLowerBound, shapeDomainBound,
          /*gridArea=*/16, /*enableResourceBounds=*/!productionScheduler);
      shapeDomainIterations = propagation.iterations;
      shapeDomainWindowRejectedOptions = propagation.windowRejectedOptions;
      if (!propagation.graphAcyclic) {
        interrupted = true;
        interruptionReason = "graph-bound-only-retained";
        shapeDomainStatus = "dependency-graph-unavailable";
      } else if (propagation.wholeGraphAboveBound &&
                 firstBlockingReason.empty() &&
                 propagation.contradictionTask !=
                     std::numeric_limits<uint64_t>::max()) {
        // Keep the last known option in the helper's domain.  The graph-level
        // certificate is a separate result and is consumed only after the
        // existing cost/provenance gates have passed.
        globalGraphPruned = true;
        globalShapeDomainContradiction = true;
        globalShapeDomainContradictionTask =
            propagation.contradictionTask;
        globalCutoffProof = true;
        globalCutoffProofKind =
            "shape-domain-necessary-condition-contradiction";
        shapeDomainStatus = "whole-graph-above-global-fifth";
        llvm::APInt totalShapeCount(counterWidth, descriptor.candidateCount,
                                    10);
        boundPrunedShapes = totalShapeCount;
        globalBoundPrunedShapes = totalShapeCount;
      } else if (propagation.applied) {
        interrupted = true;
        interruptionReason = "graph-bound-only-retained";
        shapeDomainStatus = "applied-bound-only-retained";
      } else {
        interrupted = true;
        interruptionReason = "graph-bound-only-retained";
        shapeDomainStatus = "no-reduction-bound-only";
      }
    } else if (!rankingPruning.getValue()) {
      // Full scoring starts directly at the requested partition: no upper
      // bound seeds, domain filtering, or rank-bound subtree rejection.
      checkpoint();
    } else if (seedOnly.getValue()) {
      // Seed mode visits only source-owned C++ schedule seeds for the fastest
      // legal shape tuple.  These rows are feasible upper-bound evidence; no
      // shape-space or local top-K completeness is claimed.
      warmingUp = true;
      traversed = ranked.size() >= static_cast<size_t>(topK.getValue()) ||
                  seedShapeTuples();
      warmingUp = false;
      seedOnlyTraversal = true;
      interrupted = true;
      interruptionReason = "seed-only-requested";
      checkpoint();
    } else {
      if (!warmStartHints.empty() &&
          ranked.size() >= static_cast<size_t>(topK.getValue())) {
        // The current replay already supplied a complete local incumbent;
        // avoid repeating those expensive seed tuples.  Full traversal below
        // still proves the requested candidate space.
        checkpoint();
      } else {
        warmingUp = true;
        traversed = seedShapeTuples();
        warmingUp = false;
        checkpoint();
      }
    }

    // A low numeric ordinal at the same actual cycle count tightens stable
    // tie pruning.  Greedily try smaller canonical digits in the current best
    // complete tuple, retaining a trial only through the normal ranker.  This
    // is a feasible incumbent heuristic, not a lower bound or a candidate cap.
    if (canonicalIncumbentSeed.getValue() &&
        !canonicalSeedContinuationComplete && rankingPruning.getValue() &&
        traversed && !globalGraphPruned && !graphBoundOnly.getValue() &&
        !ranked.empty()) {
      warmingUp = true;
      for (unsigned task = 0; task < metadata->size() && traversed; ++task) {
        const llvm::APInt incumbent(counterWidth, ranked.front().shapeOrdinal, 10);
        SmallVector<size_t> tuple;
        tuple.reserve(metadata->size());
        for (unsigned index = 0; index < metadata->size(); ++index) {
          llvm::APInt digit = incumbent.udiv(originalSuffixCounts[index + 1]);
          digit = digit.urem(llvm::APInt(counterWidth,
                                        descriptor.shapesByTask[index].size()));
          tuple.push_back(static_cast<size_t>(digit.getZExtValue()));
        }
        const size_t oldDigit = tuple[task];
        for (size_t smaller = 0; smaller < oldDigit; ++smaller) {
          if (!supportedShapes[task][smaller])
            continue;
          tuple[task] = smaller;
          llvm::APInt ordinal(counterWidth, 0);
          for (unsigned index = 0; index < tuple.size(); ++index)
            ordinal += originalSuffixCounts[index + 1] * tuple[index];
          const uint64_t previousCalls = productionSchedulerCalls;
          traversed = scoreTuple(ordinal, tuple);
          canonicalSeedSchedulerCalls += productionSchedulerCalls - previousCalls;
          if (!traversed)
            break;
          if (ranked.front().shapeOrdinal == exactShapeIndexString(ordinal)) {
            ++canonicalSeedAccepted;
            break;
          }
        }
      }
      warmingUp = false;
      if (traversed)
        canonicalSeedContinuationComplete = true;
      checkpoint();
    }

    // The warmup row is a real verified schedule.  Use the tighter of the
    // local fifth and an externally verified global incumbent as the domain
    // propagation bound.  This only narrows the traversal alphabet; it does
    // not change candidate ordinals or the exact scheduler's legality checks.
    if (rankingPruning.getValue() && traversed && !globalGraphPruned && !graphBoundOnly.getValue() &&
        (ranked.size() >= static_cast<size_t>(topK.getValue()) ||
         globalIncumbent.requested)) {
      shapeDomainBound =
          loadedFrontier
              ? loadedFrontier->shapeDomainBound
              : (ranked.size() >= static_cast<size_t>(topK.getValue())
                     ? ranked.back().cycles
                     : std::numeric_limits<int64_t>::max());
      if (globalIncumbent.requested)
        shapeDomainBound =
            std::min(shapeDomainBound, globalIncumbent.fifthCycles);
      ShapeDomainPropagationResult propagation = propagateShapeDomains(
          shapeDomainOptions, predecessors, edgeLowerBound, shapeDomainBound,
          /*gridArea=*/16, /*enableResourceBounds=*/!productionScheduler);
      shapeDomainIterations = propagation.iterations;
      shapeDomainWindowRejectedOptions = propagation.windowRejectedOptions;
      if (!propagation.graphAcyclic) {
        shapeDomainStatus = "dependency-graph-unavailable";
      } else if (globalIncumbent.valid &&
                 globalIncumbent.requested &&
                 firstBlockingReason.empty() &&
                 propagation.wholeGraphAboveBound &&
                 propagation.contradictionTask !=
                     std::numeric_limits<uint64_t>::max() &&
                 shapeDomainBound == globalIncumbent.fifthCycles) {
        // The normal full-search path used to record this propagation but
        // ignore its explicit contradiction result.  Consume it only when
        // the bound is the validated global fifth; a local fifth alone is
        // not a stage-wide certificate.  The helper's strict comparisons
        // preserve candidates equal to the witness.
        globalGraphPruned = true;
        globalShapeDomainContradiction = true;
        globalShapeDomainContradictionTask = propagation.contradictionTask;
        globalCutoffProof = true;
        globalCutoffProofKind =
            "shape-domain-necessary-condition-contradiction";
        shapeDomainStatus = "whole-graph-above-global-fifth";
        llvm::APInt totalShapeCount(counterWidth, descriptor.candidateCount,
                                    10);
        boundPrunedShapes = totalShapeCount;
        globalBoundPrunedShapes = totalShapeCount;
      } else if (propagation.applied) {
        shapeDomainStatus = "applied";
      } else {
        shapeDomainStatus = "no-reduction";
      }
    } else if (!traversed) {
      shapeDomainStatus = "warmup-incomplete";
    } else if (!graphBoundOnly.getValue()) {
      shapeDomainStatus = "warmup-fewer-than-top-k";
    }
    // Reconstruct stable original ordinals from the filtered options and
    // refresh all lower bounds/order preferences over the surviving domains.
    for (unsigned task = 0; task < metadata->size(); ++task) {
      shapeDomains[task].clear();
      for (const ShapeDomainOption &option : shapeDomainOptions[task])
        shapeDomains[task].push_back(option.originalOrdinal);
      minimumDuration[task] = std::numeric_limits<int64_t>::max();
      minimumWork[task] = ~static_cast<unsigned __int128>(0);
      bool hasUnknownCost = false;
      preferredShapes[task].clear();
      for (size_t ordinal : shapeDomains[task])
        preferredShapes[task].push_back(ordinal);
      // Look up by original ordinal rather than relying on domain ordering;
      // orientation order is part of the descriptor contract.
      for (const ShapeDomainOption &option : shapeDomainOptions[task]) {
        if (!option.supported || option.duration <= 0) {
          hasUnknownCost = true;
          continue;
        }
        minimumDuration[task] =
            std::min(minimumDuration[task], option.duration);
        minimumWork[task] = std::min(
            minimumWork[task],
            static_cast<unsigned __int128>(option.duration) * option.area);
      }
      // An unknown row may be faster or cheaper than every known row.  Treat
      // it as an optimistic zero lower bound so post-filter branch bounds
      // cannot be justified by ignoring an unmeasured alternative.
      if (hasUnknownCost) {
        minimumDuration[task] = 0;
        minimumWork[task] = 0;
      }
      std::stable_sort(preferredShapes[task].begin(),
                       preferredShapes[task].end(), [&](size_t lhs, size_t rhs) {
                         if (durations[task][lhs] != durations[task][rhs])
                           return durations[task][lhs] < durations[task][rhs];
                         return lhs < rhs;
                       });
    }
    // Production ranking-pruning is the one path where the order in which
    // shape variables are decided affects the amount of work before the
    // source-owned scheduler is called.  Choose the most constrained/urgent
    // task first using only already-loaded C++ facts.  The canonical task
    // index remains the coordinate used for every candidate ordinal and for
    // the shape tuple passed to the production adapter.  Keep the canonical
    // order for shards, legacy exact traversal, and unpruned production so
    // their traversal and partition contracts remain byte-for-byte stable.
    std::vector<unsigned> decisionOrder(metadata->size());
    for (unsigned task = 0; task < metadata->size(); ++task)
      decisionOrder[task] = task;
    bool nonCanonicalDecisionOrder = false;
    const bool mayReorderProductionShapes =
        productionScheduler && rankingPruning.getValue() &&
        shapeShardCount.getValue() == 1;
    if (mayReorderProductionShapes && metadata->size() > 1) {
      const unsigned taskCount = metadata->size();
      std::vector<std::vector<unsigned>> successors(taskCount);
      std::vector<unsigned> indegree(taskCount, 0);
      bool validGraph = true;
      for (unsigned task = 0; task < taskCount && validGraph; ++task) {
        for (unsigned predecessor : predecessors[task]) {
          if (predecessor >= taskCount) {
            validGraph = false;
            break;
          }
          successors[predecessor].push_back(task);
          ++indegree[task];
        }
      }
      std::vector<unsigned> topologicalOrder;
      std::vector<uint8_t> emitted(taskCount, 0);
      if (validGraph) {
        for (unsigned step = 0; step < taskCount; ++step) {
          unsigned ready = taskCount;
          for (unsigned task = 0; task < taskCount; ++task) {
            if (!emitted[task] && indegree[task] == 0) {
              ready = task;
              break;
            }
          }
          if (ready == taskCount)
            break;
          emitted[ready] = 1;
          topologicalOrder.push_back(ready);
          for (unsigned successor : successors[ready])
            --indegree[successor];
        }
      }
      if (topologicalOrder.size() == taskCount) {
        auto saturatingAdd = [](int64_t lhs, int64_t rhs) {
          if (lhs < 0 || rhs < 0 ||
              lhs > std::numeric_limits<int64_t>::max() - rhs)
            return std::numeric_limits<int64_t>::max();
          return lhs + rhs;
        };
        std::vector<int64_t> minimumTaskDuration(taskCount, 0);
        for (unsigned task = 0; task < taskCount; ++task)
          minimumTaskDuration[task] =
              minimumDuration[task] == std::numeric_limits<int64_t>::max()
                  ? 0
                  : minimumDuration[task];
        std::vector<int64_t> earliestStart(taskCount, 0);
        for (unsigned task : topologicalOrder) {
          for (unsigned predecessor : predecessors[task]) {
            const int64_t predecessorEnd = saturatingAdd(
                earliestStart[predecessor], minimumTaskDuration[predecessor]);
            earliestStart[task] = std::max(
                earliestStart[task],
                saturatingAdd(predecessorEnd, edgeLowerBound[task][predecessor]));
          }
        }
        std::vector<int64_t> criticalPathTail(taskCount, 0);
        for (auto it = topologicalOrder.rbegin();
             it != topologicalOrder.rend(); ++it) {
          const unsigned task = *it;
          int64_t successorPath = 0;
          for (unsigned successor : successors[task])
            successorPath = std::max(
                successorPath,
                saturatingAdd(edgeLowerBound[successor][task],
                              criticalPathTail[successor]));
          criticalPathTail[task] =
              saturatingAdd(minimumTaskDuration[task], successorPath);
        }
        int64_t graphCriticalPath = 0;
        for (unsigned task = 0; task < taskCount; ++task)
          graphCriticalPath = std::max(
              graphCriticalPath,
              saturatingAdd(earliestStart[task], criticalPathTail[task]));

        struct DecisionKey {
          int64_t throughTaskPath = 0;
          int64_t slack = 0;
          int64_t durationSpread = 0;
          int64_t criticalPathTail = 0;
          unsigned task = 0;
        };
        std::vector<DecisionKey> keys;
        keys.reserve(taskCount);
        for (unsigned task = 0; task < taskCount; ++task) {
          const int64_t end =
              saturatingAdd(earliestStart[task], criticalPathTail[task]);
          int64_t spread = 0;
          int64_t fastest = std::numeric_limits<int64_t>::max();
          int64_t slowest = 0;
          for (size_t ordinal : shapeDomains[task]) {
            if (!supportedShapes[task][ordinal])
              continue;
            const int64_t duration = durations[task][ordinal];
            if (duration < 0 ||
                duration == std::numeric_limits<int64_t>::max())
              continue;
            fastest = std::min(fastest, duration);
            slowest = std::max(slowest, duration);
          }
          if (fastest != std::numeric_limits<int64_t>::max() &&
              slowest >= fastest)
            spread = slowest - fastest;
          keys.push_back({end,
                          graphCriticalPath >= end ? graphCriticalPath - end
                                                    : 0,
                          spread, criticalPathTail[task], task});
        }
        std::stable_sort(keys.begin(), keys.end(), [](const DecisionKey &lhs,
                                                       const DecisionKey &rhs) {
          if (lhs.throughTaskPath != rhs.throughTaskPath)
            return lhs.throughTaskPath > rhs.throughTaskPath;
          if (lhs.slack != rhs.slack)
            return lhs.slack < rhs.slack;
          if (lhs.durationSpread != rhs.durationSpread)
            return lhs.durationSpread > rhs.durationSpread;
          if (lhs.criticalPathTail != rhs.criticalPathTail)
            return lhs.criticalPathTail > rhs.criticalPathTail;
          return lhs.task < rhs.task;
        });
        for (unsigned depth = 0; depth < taskCount; ++depth)
          decisionOrder[depth] = keys[depth].task;
        for (unsigned depth = 0; depth < taskCount; ++depth) {
          if (decisionOrder[depth] != depth) {
            nonCanonicalDecisionOrder = true;
            break;
          }
        }
      }
    }
    // The surviving suffix count is used only for proof accounting.  Shape
    // ordinals still use originalSuffixCounts below.
    std::vector<llvm::APInt> domainSuffixCounts(
        metadata->size() + 1, llvm::APInt(counterWidth, 1));
    for (size_t depth = metadata->size(); depth > 0; --depth)
      domainSuffixCounts[depth - 1] =
          domainSuffixCounts[depth] *
          shapeDomains[decisionOrder[depth - 1]].size();
    llvm::APInt totalShapeCount(counterWidth, descriptor.candidateCount, 10);
    llvm::APInt survivingShapeCount = domainSuffixCounts.front();
    domainPrunedShapes = totalShapeCount.uge(survivingShapeCount)
                              ? totalShapeCount - survivingShapeCount
                              : llvm::APInt(counterWidth, 0);
    if (!globalGraphPruned) {
      boundPrunedShapes = domainPrunedShapes;
      if (globalIncumbent.requested &&
          shapeDomainBound == globalIncumbent.fifthCycles)
        globalBoundPrunedShapes = domainPrunedShapes;
    }
    // selectedShapes is always indexed by canonical task index.  The flags
    // and decisionPrefix track the non-canonical decision prefix separately.
    SmallVector<size_t> selectedShapes(metadata->size(), 0);
    std::vector<uint8_t> selectedShapeFlags(metadata->size(), 0);
    SmallVector<unsigned> decisionPrefix;
    std::vector<ShapeFrontierFrameSnapshot> savedFrontierFrames;
    if (loadedFrontier) {
      if (loadedFrontier->decisionOrder != decisionOrder) {
        auto formatDecisionOrder = [](ArrayRef<unsigned> order) {
          std::string formatted;
          for (unsigned task : order) {
            if (!formatted.empty())
              formatted += ",";
            formatted += std::to_string(task);
          }
          return formatted;
        };
        function.emitError()
            << "shape frontier decision order differs from current scheduler facts"
            << " saved=" << formatDecisionOrder(loadedFrontier->decisionOrder)
            << " current=" << formatDecisionOrder(decisionOrder);
        return signalPassFailure();
      }
      llvm::APInt expectedOrdinal(counterWidth, 0);
      for (size_t depth = 0; depth < loadedFrontier->frames.size(); ++depth) {
        const ShapeFrontierFrameSnapshot &frame =
            loadedFrontier->frames[depth];
        if (depth > 0) {
          const unsigned selectedTask = decisionOrder[depth - 1];
          if (!llvm::is_contained(preferredShapes[selectedTask],
                                  frame.chosenIndex)) {
            function.emitError()
                << "shape frontier selected shape is outside current domain";
            return signalPassFailure();
          }
          expectedOrdinal += originalSuffixCounts[selectedTask + 1] *
                             frame.chosenIndex;
        } else if (frame.chosenIndex != 0) {
          function.emitError() << "shape frontier root has a nonzero choice";
          return signalPassFailure();
        }
        if (frame.depth != depth ||
            frame.task != (depth < metadata->size() ? decisionOrder[depth]
                                                    : metadata->size()) ||
            frame.ordinal != exactShapeIndexString(expectedOrdinal)) {
          function.emitError()
              << "shape frontier frame does not match current DFS prefix depth="
              << depth << " task=" << frame.task << " expected-task="
              << (depth < metadata->size() ? decisionOrder[depth]
                                           : metadata->size())
              << " ordinal=" << frame.ordinal << " expected-ordinal="
              << exactShapeIndexString(expectedOrdinal);
          return signalPassFailure();
        }
        if (depth < metadata->size()) {
          if (frame.nextChoice > preferredShapes[frame.task].size()) {
            function.emitError()
                << "shape frontier next choice exceeds current shape domain";
            return signalPassFailure();
          }
        } else if (frame.nextChoice != 0) {
          function.emitError() << "shape frontier leaf has a next choice";
          return signalPassFailure();
        }
      }
      savedFrontierFrames = loadedFrontier->frames;
    }
    writeFrontierSnapshot = [&](StringRef reason) {
      if (frontierOutputPath.empty())
        return true;
      if (canonicalIncumbentSeed.getValue() &&
          !canonicalSeedContinuationComplete) {
        error =
            "cannot write shape frontier before canonical incumbent seed "
            "continuation is complete";
        fatalError = true;
        return false;
      }
      llvm::json::Object root;
      root["record_type"] = "shape-frontier";
      root["schema"] = "amoeba-shape-frontier-v1";
      root["status"] = "incomplete";
      root["phase"] = "shape-search";
      root["resume_eligible"] = true;
      root["implementation_contract"] =
          "score-exact-joint-task-candidates-frontier-v1";
      root["binding_witness_path"] = frontierOutputBindingPath;
      root["reason"] = reason.str();
      root["candidate_count"] = descriptor.candidateCount;
      root["top_k"] = topK.getValue();
      root["function"] = function.getSymName().str();
      root["graph_variant_id"] = graphVariantId.getValue();
      root["source_repository"] = sourceRepository.getValue();
      root["source_commit"] = sourceCommit.getValue();
      root["dispatch_policy"] = dispatchPolicy.getValue();
      root["temporal_model"] = temporalModel.getValue();
      root["schedule_space"] = scheduleSpace.getValue();
      root["production_scheduler_contract"] = kProductionSchedulerContract.str();
      root["canonical_incumbent_seed_enabled"] =
          canonicalIncumbentSeed.getValue();
      root["canonical_seed_continuation_state"] =
          canonicalIncumbentSeed.getValue()
              ? (canonicalSeedContinuationComplete ? "complete" : "incomplete")
              : "disabled";
      putDecimal(root, "canonical_seed_scheduler_calls",
                 std::to_string(canonicalSeedSchedulerCalls));
      putDecimal(root, "canonical_seed_accepted",
                 std::to_string(canonicalSeedAccepted));
      putDecimal(root, "canonical_seed_next_task",
                 std::to_string(canonicalIncumbentSeed.getValue()
                                    ? metadata->size()
                                    : 0));
      putDecimal(root, "canonical_seed_next_smaller", "0");
      llvm::json::Array order;
      for (unsigned task : decisionOrder)
        order.push_back(static_cast<int64_t>(task));
      root["decision_order"] = std::move(order);
      llvm::json::Array warmupSolved;
      for (const std::string &shape : frontierWarmupSolvedShapes)
        warmupSolved.push_back(shape);
      root["warmup_solved_shapes"] = std::move(warmupSolved);
      llvm::json::Array frames;
      if (frontierStack.empty()) {
        ShapeFrontierFrameSnapshot rootFrame;
        rootFrame.depth = 0;
        rootFrame.task = metadata->empty() ? 0 : decisionOrder.front();
        rootFrame.chosenIndex = 0;
        rootFrame.nextChoice = 0;
        rootFrame.ordinal = "0";
        frontierStack.push_back(std::move(rootFrame));
        for (const ShapeFrontierFrameSnapshot &frame : frontierStack) {
          llvm::json::Object item;
          item["depth"] = static_cast<int64_t>(frame.depth);
          item["task"] = static_cast<int64_t>(frame.task);
          item["chosen_index"] = static_cast<int64_t>(frame.chosenIndex);
          putDecimal(item, "next_choice", std::to_string(frame.nextChoice));
          item["ordinal"] = frame.ordinal;
          frames.push_back(std::move(item));
        }
        frontierStack.pop_back();
      } else {
        for (const ShapeFrontierFrameSnapshot &frame : frontierStack) {
          llvm::json::Object item;
          item["depth"] = static_cast<int64_t>(frame.depth);
          item["task"] = static_cast<int64_t>(frame.task);
          item["chosen_index"] = static_cast<int64_t>(frame.chosenIndex);
          putDecimal(item, "next_choice", std::to_string(frame.nextChoice));
          item["ordinal"] = frame.ordinal;
          frames.push_back(std::move(item));
        }
      }
      root["frames"] = std::move(frames);
      llvm::json::Array rows;
      for (const RankedSchedule &entry : ranked) {
        llvm::json::Object row;
        row["key"] = entry.key;
        row["shape_ordinal"] = entry.shapeOrdinal;
        putDecimal(row, "schedule_ordinal", std::to_string(entry.scheduleOrdinal));
        row["candidate_id"] = "shape-" + entry.shapeOrdinal + "/schedule-" +
                               std::to_string(entry.scheduleOrdinal);
        row["cycles"] = entry.cycles;
        llvm::json::Object record = entry.record;
        row["record"] = std::move(record);
        rows.push_back(std::move(row));
      }
      root["ranked"] = std::move(rows);
      putDecimal(root, "schedule_serial", std::to_string(scheduleSerial));
      putDecimal(root, "expanded_nodes", std::to_string(expandedNodes));
      putDecimal(root, "expanded_shape_nodes",
                 std::to_string(expandedShapeNodes));
      putDecimal(root, "production_scheduler_calls",
                 std::to_string(productionSchedulerCalls));
      putDecimal(root, "valid_shapes", exactShapeIndexString(validShapes));
      putDecimal(root, "unsupported_shapes", std::to_string(unsupportedShapes));
      root["shape_domain_bound_cycles"] = shapeDomainBound;
      root["visited_shapes"] = exactShapeIndexString(visitedShapes);
      root["bound_pruned_shapes"] = exactShapeIndexString(boundPrunedShapes);
      root["saved_max_shape_candidates"] = maxShapeCandidates.getValue();
      root["saved_max_expanded_nodes"] = maxExpandedNodes.getValue();
      root["saved_max_milliseconds"] = maxMilliseconds.getValue();
      std::string writeError;
      if (!writeAtomically(
              frontierOutputPath,
              [&](llvm::raw_ostream &stream) {
                stream << llvm::json::Value(std::move(root));
                stream << "\n";
                return true;
              },
              writeError)) {
        error = "cannot write shape frontier: " + writeError;
        fatalError = true;
        return false;
      }
      frontierSnapshotWritten = true;
      return true;
    };
    auto shapeBoundExceeds = [&](const llvm::APInt &ordinal) {
      if (!rankingPruning.getValue()) return false;
      globalCutoffShapePruned = false;
      const bool haveLocalBound =
          ranked.size() >= static_cast<size_t>(topK.getValue());
      if (!haveLocalBound && !globalIncumbent.requested)
        return false;
      const int64_t localBound =
          haveLocalBound ? ranked.back().cycles
                         : std::numeric_limits<int64_t>::max();
      std::vector<int64_t> ends(metadata->size(), -1);
      std::vector<bool> endsFinite(metadata->size(), true);
      auto selectedOption = [&](unsigned i) {
        const size_t ordinal = selectedShapes[i];
        return ShapeDomainOption{
            ordinal, durations[i][ordinal],
            static_cast<unsigned>(
                descriptor.shapesByTask[i][ordinal].cgraCount()),
            supportedShapes[i][ordinal] != 0};
      };
      auto selectedDuration = [&](unsigned i) {
        return selectedShapeFlags[i]
                   ? optimisticShapeDuration(selectedOption(i))
                   : minimumDuration[i];
      };
      auto selectedWork = [&](unsigned i) {
        return selectedShapeFlags[i]
                   ? optimisticShapeWork(selectedOption(i))
                   : minimumWork[i];
      };
      std::function<int64_t(unsigned)> earliest = [&](unsigned i) {
        if (ends[i] >= 0) return ends[i];
        int64_t start = 0;
        for (unsigned p : predecessors[i]) {
          int64_t end = earliest(p), delay = edgeLowerBound[i][p];
          if (!endsFinite[p] || delay < 0 ||
              delay > std::numeric_limits<int64_t>::max() - end) {
            ends[i] = std::numeric_limits<int64_t>::max();
            endsFinite[i] = false;
            return ends[i];
          }
          start = std::max(start, end + delay);
        }
        const int64_t duration = selectedDuration(i);
        if (duration < 0 ||
            duration > std::numeric_limits<int64_t>::max() - start) {
          ends[i] = std::numeric_limits<int64_t>::max();
          endsFinite[i] = false;
          return ends[i];
        }
        endsFinite[i] = true;
        return ends[i] = start + duration;
      };
      bool canImproveTie = false;
      if (haveLocalBound) {
        if (!nonCanonicalDecisionOrder) {
          canImproveTie = compareDecimal(
                              exactShapeIndexString(ordinal),
                              ranked.back().shapeOrdinal) < 0;
        } else {
          // An arbitrary decision prefix does not make `ordinal` a canonical
          // prefix.  Complete it with the smallest surviving ordinal of each
          // undecided task before applying the stable shape tie cutoff.
          llvm::APInt minimumCompletion = ordinal;
          for (unsigned task = 0; task < metadata->size(); ++task) {
            if (selectedShapeFlags[task] || shapeDomains[task].empty())
              continue;
            const size_t minimumOrdinal = *std::min_element(
                shapeDomains[task].begin(), shapeDomains[task].end());
            minimumCompletion +=
                originalSuffixCounts[task + 1] * minimumOrdinal;
          }
          canImproveTie = compareDecimal(
                              exactShapeIndexString(minimumCompletion),
                              ranked.back().shapeOrdinal) < 0;
        }
      }
      bool localRejected = false;
      bool globalRejected = false;
      int64_t ordinaryLowerBound = 0;
      bool ordinaryBoundFinite = true;
      unsigned __int128 work = 0;
      bool workFinite = true;
      for (unsigned i = 0; i < metadata->size(); ++i) {
        int64_t lower = earliest(i);
        if (!endsFinite[i]) {
          ordinaryBoundFinite = false;
        } else {
          ordinaryLowerBound = std::max(ordinaryLowerBound, lower);
        }
        if (endsFinite[i] && haveLocalBound &&
            (lower > localBound || (lower == localBound && !canImproveTie)))
          localRejected = true;
        if (endsFinite[i] && globalIncumbent.requested &&
            lower > globalIncumbent.fifthCycles)
          globalRejected = true;
        if (workFinite &&
            !::mlir::amoeba::neura::joint_scheduling::detail::addUnsignedWide(
                work, selectedWork(i), work))
          workFinite = false;
      }
      // The local stable-key proof is only useful on a genuine equal-cycle
      // branch.  If the unrestricted CP bound is below H, a descendant with
      // a larger ordinal may still finish below H; applying the restricted
      // ordinal bound there would be unsound.  A global incumbent has no
      // graph-local ordinal key, so this proof is deliberately disabled for
      // cross-graph cutoffs.
      const bool genuineEqualCycleBranch =
          !localRejected && !globalIncumbent.requested && haveLocalBound &&
          ordinaryBoundFinite && ordinaryLowerBound == localBound &&
          canImproveTie;
      if (genuineEqualCycleBranch) {
        ++tieRestrictedCpBoundCalls;
        TieRestrictedCriticalPathResult restricted =
            computeTieRestrictedCriticalPath(
                ranked.back().shapeOrdinal, counterWidth, descriptor,
                shapeDomainOptions, selectedShapes, selectedShapeFlags,
                predecessors, edgeLowerBound);
        if (restricted.valid) {
          if (!restricted.hasEligibleCase) {
            ++tieRestrictedCpNoEligible;
            localRejected = true;
          } else if (restricted.lowerBound > localBound) {
            ++tieRestrictedCpBoundPruned;
            localRejected = true;
          }
        } else {
          ++tieRestrictedCpInvalid;
        }
      }
      // This integrates rectangle area over execution time. It does not
      // replace the per-interval cell occupancy test with an area sum.
      if (!productionScheduler && workFinite && haveLocalBound &&
          (work > static_cast<unsigned __int128>(localBound) * 16 ||
           (!canImproveTie &&
            work == static_cast<unsigned __int128>(localBound) * 16)))
        localRejected = true;
      // The incumbent cutoff is intentionally strict on lower bounds only.
      // A shape whose optimistic bound equals the fifth cycle count remains
      // in the search so equal-cycle candidates and their stable keys survive.
      if (!productionScheduler && workFinite && globalIncumbent.requested &&
          work > static_cast<unsigned __int128>(globalIncumbent.fifthCycles) *
                     16)
        globalRejected = true;

      // Run the energetic release/deadline-window proof on a private snapshot
      // of the surviving shape domains after the cheap CP/work tests pass.
      // A selected prefix is fixed to its exact original-ordinal option; all
      // unselected tasks retain every surviving option.  The helper is
      // necessary-condition-only and fail-open for unsupported/overflowing
      // rows, so no production domain or ordinal accounting is mutated here.
      const bool haveGlobalWitness =
          globalIncumbent.requested && globalIncumbent.valid;
      const bool havePrefixWitness = haveLocalBound || haveGlobalWitness;
      const int64_t globalBound =
          haveGlobalWitness ? globalIncumbent.fifthCycles
                            : std::numeric_limits<int64_t>::max();
      const int64_t prefixBound =
          havePrefixWitness
              ? std::min(haveLocalBound ? localBound : globalBound,
                         haveGlobalWitness ? globalBound
                                           : std::numeric_limits<int64_t>::max())
              : 0;
      const bool prefixBoundFinite =
          havePrefixWitness && prefixBound > 0 &&
          prefixBound != std::numeric_limits<int64_t>::max();
      if (!productionScheduler && !localRejected && !globalRejected && prefixBoundFinite &&
          decisionPrefix.size() < metadata->size() &&
          !decisionPrefix.empty()) {
        std::vector<std::vector<ShapeDomainOption>> prefixDomains =
            shapeDomainOptions;
        bool prefixSnapshotValid = true;
        for (unsigned task : decisionPrefix) {
          const size_t selectedOrdinal = selectedShapes[task];
          auto selectedIt = llvm::find_if(
              prefixDomains[task], [&](const ShapeDomainOption &option) {
                return option.originalOrdinal == selectedOrdinal;
              });
          if (selectedIt == prefixDomains[task].end()) {
            // The selected ordinal must come from the surviving domain.  If
            // malformed/stale input violates that invariant, fail open.
            prefixSnapshotValid = false;
            break;
          }
          ShapeDomainOption selected = *selectedIt;
          prefixDomains[task].clear();
          prefixDomains[task].push_back(selected);
        }
        if (prefixSnapshotValid) {
          ++prefixWindowBoundCalls;
          ShapeDomainPropagationResult prefixPropagation =
              propagateShapeDomains(prefixDomains, predecessors,
                                    edgeLowerBound, prefixBound,
                                    /*gridArea=*/16, /*enableResourceBounds=*/!productionScheduler);
          if (prefixPropagation.wholeGraphAboveBound &&
              prefixPropagation.contradictionTask !=
                  std::numeric_limits<uint64_t>::max()) {
            ++prefixWindowBoundRejected;
            // A global witness contributes to global-cutoff accounting only
            // when it is the active (tightest) bound.  A stricter local fifth
            // can still reject this subtree without claiming stage-wide
            // omission.
            globalCutoffShapePruned =
                globalRejected ||
                (haveGlobalWitness && globalBound <= prefixBound);
            return true;
          }
        }
      }
      globalCutoffShapePruned = globalRejected;
      return localRejected || globalRejected;
    };
    std::function<bool(size_t, const llvm::APInt &)> visitShapes =
        [&](size_t depth, const llvm::APInt &ordinal) {
      if (!nonCanonicalDecisionOrder && shapeShardCount.getValue() > 1 &&
          (ordinal.uge(shardEnd) ||
           (ordinal + originalSuffixCounts[depth]).ule(shardBegin))) {
        ++partitionSkippedSubtrees;
        return true;
      }
      size_t resumeChoice = 0;
      bool restoredFrame = false;
      if (frontierResumeActive) {
        // A checkpoint records the active stack, not the child currently
        // being entered.  Descending one level beyond that stack is the
        // legitimate continuation of its saved next choice.
        if (depth >= savedFrontierFrames.size()) {
          frontierResumeActive = false;
        } else {
          bool matches = true;
          const ShapeFrontierFrameSnapshot &saved = savedFrontierFrames[depth];
          const unsigned expectedTask =
              depth < metadata->size() ? decisionOrder[depth]
                                       : metadata->size();
          const size_t expectedChoice =
              depth == 0 ? 0 : selectedShapes[decisionOrder[depth - 1]];
          matches = saved.depth == depth && saved.task == expectedTask &&
                    saved.chosenIndex == expectedChoice &&
                    saved.ordinal == exactShapeIndexString(ordinal);
          if (matches) {
            resumeChoice = static_cast<size_t>(saved.nextChoice);
            restoredFrame = true;
          }
          if (!matches) {
            // The loader already checked this prefix against the current
            // descriptor.  A mismatch here means the runtime reconstruction
            // diverged from the authenticated snapshot; restarting at the
            // root would silently duplicate or skip subtrees, so fail closed.
            error = "shape frontier runtime frame diverged from authenticated "
                    "prefix at depth=" + std::to_string(depth) +
                    " ordinal=" + exactShapeIndexString(ordinal);
            fatalError = true;
            return false;
          }
        }
      }
      ShapeFrontierFrameSnapshot frame;
      frame.depth = depth;
      frame.task = depth < metadata->size() ? decisionOrder[depth]
                                            : metadata->size();
      frame.chosenIndex = depth == 0 ? 0 : selectedShapes[decisionOrder[depth - 1]];
      frame.nextChoice = resumeChoice;
      frame.ordinal = exactShapeIndexString(ordinal);
      frontierStack.push_back(frame);
      auto popFrame = [&]() { frontierStack.pop_back(); };
      if (!restoredFrame) {
        if (std::chrono::steady_clock::now() - lastCheckpoint >=
            std::chrono::seconds(60))
          checkpoint();
        ++expandedShapeNodes;
        if (frontierCheckpointEvery.getValue() > 0 &&
            (expandedShapeNodes - lastFrontierCheckpointNodes >=
             static_cast<uint64_t>(frontierCheckpointEvery.getValue()))) {
          if (!writeFrontierSnapshot("periodic")) {
            interrupted = true;
            interruptionReason = "frontier-write-failed";
            popFrame();
            return false;
          }
          lastFrontierCheckpointNodes = expandedShapeNodes;
        }
        if (expandedShapeNodes + expandedNodes >=
            static_cast<uint64_t>(maxExpandedNodes)) {
          interrupted = true;
          interruptionReason = "max-expanded-nodes";
          writeFrontierSnapshot(interruptionReason);
          popFrame();
          return false;
        }
        if (maxMilliseconds > 0 &&
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - begun)
                    .count() >= maxMilliseconds) {
          interrupted = true;
          interruptionReason = "max-milliseconds";
          writeFrontierSnapshot(interruptionReason);
          popFrame();
          return false;
        }
      }
      if (!restoredFrame && shapeBoundExceeds(ordinal)) {
        const llvm::APInt pruned = domainSuffixCounts[depth];
        boundPrunedShapes += pruned;
        if (globalCutoffShapePruned)
          globalBoundPrunedShapes += pruned;
        popFrame();
        return true;
      }
      if (depth == metadata->size()) {
        bool keepGoing = scoreTuple(ordinal, selectedShapes);
        const bool resumableStop =
            interruptionReason == "max-shape-candidates" ||
            interruptionReason == "max-milliseconds" ||
            interruptionReason == "max-expanded-nodes";
        if (!keepGoing && resumableStop && writeFrontierSnapshot)
          writeFrontierSnapshot(interruptionReason);
        popFrame();
        return keepGoing;
      }
      const unsigned task = decisionOrder[depth];
      for (size_t position = resumeChoice;
           position < preferredShapes[task].size(); ++position) {
        const size_t index = preferredShapes[task][position];
        selectedShapes[task] = index;
        selectedShapeFlags[task] = 1;
        decisionPrefix.push_back(task);
        frontierStack.back().nextChoice = position;
        if (frontierCheckpointEvery.getValue() > 0 &&
            (expandedShapeNodes - lastFrontierCheckpointNodes >=
             static_cast<uint64_t>(frontierCheckpointEvery.getValue()))) {
          if (!writeFrontierSnapshot("before-child")) {
            interrupted = true;
            interruptionReason = "frontier-write-failed";
            decisionPrefix.pop_back();
            selectedShapeFlags[task] = 0;
            popFrame();
            return false;
          }
          lastFrontierCheckpointNodes = expandedShapeNodes;
        }
        bool keepGoing = visitShapes(
            depth + 1, ordinal + originalSuffixCounts[task + 1] * index);
        decisionPrefix.pop_back();
        selectedShapeFlags[task] = 0;
        frontierStack.back().nextChoice = position + 1;
        if (!keepGoing) {
          popFrame();
          return false;
        }
        // The authenticated prefix has now been replayed successfully.  Any
        // later sibling is a fresh subtree after the checkpoint, so its
        // frame is allowed to diverge from the saved prefix and must not be
        // mistaken for a corrupted snapshot.
        if (restoredFrame)
          frontierResumeActive = false;
      }
      popFrame();
      return true;
    };
    if (traversed && !globalGraphPruned && !seedOnlyTraversal &&
        !graphBoundOnly.getValue())
      traversed = visitShapes(0, llvm::APInt(counterWidth, 0));
    if (fatalError) {
      function.emitError() << error;
      return signalPassFailure();
    }
    if (!traversed && !interrupted) {
      interrupted = true;
      interruptionReason = "shape-space-traversal-failed";
    }
    if (costs.coveredQueries() != costs.catalogQueries() && !interrupted) {
      interrupted = true;
      interruptionReason = "cost-catalogue-coverage-incomplete";
      if (firstBlockingReason.empty())
        firstBlockingReason = interruptionReason;
    }
    // Five feasible candidates at or below H prove that schedules finishing
    // after H cannot enter this top five; no caller assertion is required.
    const bool horizonCertified = productionScheduler || temporalModel == "dispatch-order" || horizonProven.getValue() ||
        (ranked.size() >= static_cast<size_t>(topK.getValue()) && ranked.back().cycles <= maxMakespan);

    if (!horizonCertified && !interrupted) {
      interruptionReason = "max-makespan-not-proven";
      if (firstBlockingReason.empty())
        firstBlockingReason = interruptionReason;
    }
    if (!firstBlockingReason.empty())
      interruptionReason = firstBlockingReason;
    if (globalIncumbent.requested)
      globalCutoffProof =
          traversed && !interrupted && firstBlockingReason.empty() &&
          unsupportedShapes == 0 && !stateMemoCutoffViolation;
    const bool partitionFinished = traversed && !interrupted && firstBlockingReason.empty() && visitedShapes == shardSize;
    const bool complete = shapeShardCount.getValue() == 1 && !interrupted && firstBlockingReason.empty() && horizonCertified &&
                          unsupportedShapes == 0 &&
                          ranked.size() >= static_cast<size_t>(topK.getValue());
    const bool globallyComplete = globalIncumbent.requested &&
                                  globalCutoffProof;
    const bool outputComplete = !seedOnly.getValue() &&
                                (complete || globallyComplete);
    const bool globallyCutoffResult = globallyComplete && !complete;

    if (!writeAtomically(
            outputFile,
            [&](llvm::raw_ostream &stream) {
              llvm::json::Object header;
              header["record_type"] = "header";
              header["schema"] = "amoeba-exact-joint-task-scores-v1";
              header["graph_bound_only"] = graphBoundOnly.getValue();
              header["graph_bound_basis"] =
                  productionScheduler ? "cached-cost-critical-path"
                                      : "cached-cost-critical-path-and-minimum-cgra-cycle-work";
              header["function"] = function.getSymName().str();
              header["representation"] = "factored-plus-exact-schedule";
              header["candidate_count"] = descriptor.candidateCount;
              header["warm_start_shapes_requested"] =
                  !warmStartShapes.getValue().empty();
              putDecimal(header, "warm_start_shape_hint_count",
                         std::to_string(warmStartHints.size()));
              putDecimal(header, "warm_start_shape_hint_replayed",
                         std::to_string(warmStartHintsReplayed));
              header["warm_start_shape_hint_replay_failed"] =
                  warmStartReplayFailed;
              if (!warmStartShapes.getValue().empty())
                header["warm_start_shapes_path"] = warmStartShapes.getValue();
              header["shape_frontier_requested"] = frontierRequested;
              header["shape_frontier_resumed"] = loadedFrontier.has_value();
              header["shape_frontier_snapshot_written"] =
                  frontierSnapshotWritten;
              header["shape_frontier_restored_scheduler_calls"] =
                  static_cast<int64_t>(frontierRestoredSchedulerCalls);
              header["shape_frontier_scheduler_calls_this_run"] =
                  static_cast<int64_t>(
                      productionSchedulerCalls >= frontierRestoredSchedulerCalls
                          ? productionSchedulerCalls - frontierRestoredSchedulerCalls
                          : 0);
              header["shape_frontier_restored_expanded_shape_nodes"] =
                  static_cast<int64_t>(frontierRestoredExpandedShapeNodes);
              header["shape_frontier_contract"] =
                  "score-exact-joint-task-candidates-frontier-v1";
              header["canonical_incumbent_seed_enabled"] =
                  canonicalIncumbentSeed.getValue();
              header["canonical_seed_continuation_state"] =
                  canonicalIncumbentSeed.getValue()
                      ? (canonicalSeedContinuationComplete ? "complete"
                                                            : "incomplete")
                      : "disabled";
              header["canonical_seed_scheduler_calls"] =
                  static_cast<int64_t>(canonicalSeedSchedulerCalls);
              header["canonical_seed_accepted"] =
                  static_cast<int64_t>(canonicalSeedAccepted);
              if (frontierRequested) {
                header["shape_frontier_snapshot_path"] = frontierOutputPath;
                header["shape_frontier_binding_witness_path"] =
                    frontierOutputBindingPath;
              }
              header["top_k"] = topK.getValue();
              header["grid_rows"] = 4;
              header["grid_cols"] = 4;
              header["max_cgras_per_task"] =
                  static_cast<int64_t>(descriptor.maxCgrasPerTask);
              header["communication_mode"] = "explicit";
              header["start_policy"] = scoreStartPolicy(temporalModel, scheduleSpace).str();
              header["scheduler_backend"] = productionScheduler ? "orchestrate-tasks-on-accelerators" : "orbit-exact-enumerator";
        header["orbit_4x4_pruning"] = !productionScheduler;
        header["schedule_space"] = scheduleSpace.getValue();
              if (productionScheduler)
                header["tie_key_policy"] =
                    kProductionNumericTieKeyPolicy.str();
              header["placement_coverage_limited"] = scheduleSpace != "exact";
              header["production_ready"] = false;
              header["predictor_status"] = "exploratory";
              header["state_memo_requested"] = scheduleSpace == "exact";
              header["state_memo_capacity"] = stateMemoCapacity.getValue();
              header["state_memo_external_cutoff_monotonic"] =
                  scheduleSpace == "exact";
              header["dispatch_policy"] = dispatchPolicy.getValue();
              header["dispatch_policy_scope"] =
                  selectedDispatchPolicy == DispatchPolicy::Fixed
                      ? "canonical-smallest-ready-task"
                      : productionScheduler ? "production-critical-path-dependency-ready"
                                            : "all-ready-task-orders";
              header["max_makespan"] = maxMakespan.getValue();
              header["horizon_proven"] = !seedOnly.getValue() && horizonCertified;
              header["global_shape_location_temporal_space"] = scheduleSpace != "production-scheduler";
        header["temporal_search_scope"] = scoreTemporalScope(temporalModel).str();
        header["cycle_starts_are_search_dimensions"] = temporalModel == "all-integer-diagnostic";
              header["shape_domain_pruning"] =
                  productionScheduler ? "dependency-release-tail-cost-only"
                                : "dependency-release-tail-and-energetic-work";
              header["shape_domain_status"] = shapeDomainStatus;
              header["shape_domain_bound_cycles"] = shapeDomainBound;
              header["shape_domain_fixed_point_iterations"] =
                  static_cast<int64_t>(shapeDomainIterations);
              header["shape_domain_window_rejected_options"] =
                  static_cast<int64_t>(shapeDomainWindowRejectedOptions);
              llvm::json::Array domainSummary;
              for (unsigned task = 0; task < shapeDomains.size(); ++task) {
                llvm::json::Object item;
                item["task"] = (*metadata)[task].name;
                item["original_shape_count"] = static_cast<int64_t>(
                    descriptor.shapesByTask[task].size());
                item["retained_shape_count"] =
                    static_cast<int64_t>(shapeDomains[task].size());
                int64_t minimumArea = std::numeric_limits<int64_t>::max();
                int64_t maximumArea = 0;
                llvm::json::Array retained;
                for (size_t ordinal : shapeDomains[task]) {
                  const RectShape &shape =
                      descriptor.shapesByTask[task][ordinal];
                  const int64_t area = shape.cgraCount();
                  minimumArea = std::min(minimumArea, area);
                  maximumArea = std::max(maximumArea, area);
                  llvm::json::Object shapeRecord;
                  shapeRecord["original_shape_ordinal"] =
                      static_cast<int64_t>(ordinal);
                  shapeRecord["rows"] = shape.rows;
                  shapeRecord["cols"] = shape.cols;
                  shapeRecord["mapper_tile_rows"] = shape.mapperRows;
                  shapeRecord["mapper_tile_cols"] = shape.mapperCols;
                  shapeRecord["cgra_count"] = area;
                  retained.push_back(std::move(shapeRecord));
                }
                item["retained_shapes"] = std::move(retained);
                item["retained_area_min"] =
                    minimumArea == std::numeric_limits<int64_t>::max()
                        ? 0
                        : minimumArea;
                item["retained_area_max"] = maximumArea;
                domainSummary.push_back(std::move(item));
              }
              header["shape_domain_summary"] = std::move(domainSummary);
              header["predictor_source"] = kMlPredictorSource.str();
              header["source_repository"] = costs.sourceRepository().str();
              header["source_commit"] = costs.sourceCommit().str();
              header["architecture_path"] = costs.architecturePath().str();
              header["architecture_schema"] = costs.architectureSchema().str();
              header["cost_namespace"] = costs.nameSpace().str();
              header["cost_provenance_schema"] = "orbit-cost-provenance-v1";
              header["source_graph_id"] = costs.sourceGraphId().str();
              header["graph_variant_id"] = graphVariantId.getValue();
              header["seed_only"] = seedOnly.getValue();
              if (!stage.getValue().empty())
                header["stage"] = stage.getValue();
              header["global_cutoff_requested"] = globalIncumbent.requested;
              if (globalIncumbent.requested) {
                header["global_cutoff_cycles"] = globalIncumbent.fifthCycles;
                header["global_cutoff_evidence_path"] = globalIncumbent.path;
                header["global_cutoff_key_policy"] = globalIncumbent.keyPolicy;
              }
              header["score_cache_enabled"] = scoreCacheRequested;
              header["score_cache_disabled_for_global_cutoff"] =
                  scoreCacheDisabledForGlobalCutoff;
              header["score_cache_hit"] = scoreCacheHit;
              header["score_cache_reused"] = false;
              header["score_cache_migrated"] = scoreCacheMigrated;
              header["score_cache_new_expanded_nodes"] =
                  static_cast<int64_t>(expandedNodes);
              header["score_cache_peer_requested"] = scoreCachePeerRequested;
              header["score_cache_peer_imported"] = scoreCachePeerImported;
              header["score_cache_peer_pending"] = scoreCachePeerPending;
              header["score_cache_peer_status"] = scoreCachePeerStatus;
              header["ranking_pruning"] = rankingPruning.getValue();
              header["tie_restricted_cp_scope"] =
                  "local-equal-cycle-canonical-ordinal";
              putDecimal(header, "tie_restricted_cp_bound_calls",
                         std::to_string(tieRestrictedCpBoundCalls));
              putDecimal(header, "tie_restricted_cp_bound_pruned",
                         std::to_string(tieRestrictedCpBoundPruned));
              putDecimal(header, "tie_restricted_cp_no_eligible",
                         std::to_string(tieRestrictedCpNoEligible));
              putDecimal(header, "tie_restricted_cp_invalid",
                         std::to_string(tieRestrictedCpInvalid));
        header["shape_shard_count"] = shapeShardCount.getValue();
        header["shape_shard_index"] = shapeShardIndex.getValue();
        putDecimal(header, "shape_partition_begin", exactShapeIndexString(shardBegin));
        putDecimal(header, "shape_partition_end", exactShapeIndexString(shardEnd));
        putDecimal(header, "shape_partition_candidate_count", exactShapeIndexString(shardSize));
        putDecimal(header, "partition_skipped_subtrees", std::to_string(partitionSkippedSubtrees));
        header["elapsed_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
        addProductionCoverage(header, temporalModel, scheduleSpace, dispatchPolicy, function);
        writeJsonLine(stream, std::move(header));
              for (RankedSchedule &entry : ranked)
                writeJsonLine(stream, std::move(entry.record));
              llvm::json::Object footer;
              footer["record_type"] = "footer";
              footer["schema"] = "amoeba-exact-joint-task-scores-v1";
              footer["scored_count"] = static_cast<int64_t>(ranked.size());
              footer["warm_start_shapes_requested"] =
                  !warmStartShapes.getValue().empty();
              putDecimal(footer, "warm_start_shape_hint_count",
                         std::to_string(warmStartHints.size()));
              putDecimal(footer, "warm_start_shape_hint_replayed",
                         std::to_string(warmStartHintsReplayed));
              footer["warm_start_shape_hint_replay_failed"] =
                  warmStartReplayFailed;
              footer["shape_frontier_requested"] = frontierRequested;
              footer["shape_frontier_resumed"] = loadedFrontier.has_value();
              footer["shape_frontier_snapshot_written"] =
                  frontierSnapshotWritten;
              footer["shape_frontier_restored_scheduler_calls"] =
                  static_cast<int64_t>(frontierRestoredSchedulerCalls);
              footer["shape_frontier_scheduler_calls_this_run"] =
                  static_cast<int64_t>(
                      productionSchedulerCalls >= frontierRestoredSchedulerCalls
                          ? productionSchedulerCalls - frontierRestoredSchedulerCalls
                          : 0);
              footer["shape_frontier_restored_expanded_shape_nodes"] =
                  static_cast<int64_t>(frontierRestoredExpandedShapeNodes);
              footer["shape_frontier_contract"] =
                  "score-exact-joint-task-candidates-frontier-v1";
              if (frontierRequested) {
                footer["shape_frontier_snapshot_path"] = frontierOutputPath;
                footer["shape_frontier_binding_witness_path"] =
                    frontierOutputBindingPath;
              }
              footer["status"] = seedOnly.getValue()
                                      ? "seed-only"
                                      : outputComplete
                                      ? (globallyComplete && !complete
                                             ? "globally-pruned"
                                             : "complete")
                                      : (shapeShardCount.getValue() > 1 && partitionFinished
                                             ? "partition-complete"
                                             : graphBoundOnly.getValue() &&
                                         firstBlockingReason.empty()
                                             ? "bound-only-retained"
                                             : "bounded_incomplete");
              footer["graph_bound_only"] = graphBoundOnly.getValue();
              footer["incomplete"] = seedOnly.getValue() || !outputComplete;
              footer["reason"] = seedOnly.getValue() ? "seed-only-requested" :
                  outputComplete ? "" :
                  (shapeShardCount.getValue() > 1 && partitionFinished ? "complete-shape-partition" : interruptionReason.empty() ? "fewer-than-top-k" : interruptionReason);
              if (!failedProductionCandidateId.empty()) {
                footer["failed_production_shape_candidate_id"] = failedProductionCandidateId;
                footer["failed_production_phase"] = failedProductionWarmup ? "warmup" : "search";
                footer["failed_production_task_costs"] =
                    makeTaskCosts(failedProductionChoices, failedProductionCosts);
              }
              footer["seed_only"] = seedOnly.getValue();
              footer["feasible_only"] = seedOnly.getValue();
              footer["seed_feasible_count"] =
                  seedOnly.getValue() ? static_cast<int64_t>(ranked.size()) : 0;
              putDecimal(footer, "candidate_count", descriptor.candidateCount);
              putDecimal(footer, "shape_candidates_visited",
                          exactShapeIndexString(visitedShapes));
              putDecimal(footer, "valid_shape_candidates",
                          exactShapeIndexString(validShapes));
              putDecimal(footer, "domain_pruned_shape_candidates",
                         exactShapeIndexString(domainPrunedShapes));
              llvm::APInt totalShapeCount(counterWidth, descriptor.candidateCount, 10);
              llvm::APInt unmeasuredShapes =
                  totalShapeCount.uge(visitedShapes + boundPrunedShapes)
                      ? totalShapeCount - visitedShapes - boundPrunedShapes
                      : llvm::APInt(counterWidth, 0);
              putDecimal(footer, "unmeasured_shape_candidates",
                         exactShapeIndexString(unmeasuredShapes));
              putDecimal(footer, "bound_pruned_shape_candidates", exactShapeIndexString(boundPrunedShapes));
              footer["shape_domain_status"] = shapeDomainStatus;
              footer["shape_domain_bound_cycles"] = shapeDomainBound;
              footer["shape_domain_fixed_point_iterations"] =
                  static_cast<int64_t>(shapeDomainIterations);
              footer["shape_domain_window_rejected_options"] =
                  static_cast<int64_t>(shapeDomainWindowRejectedOptions);
              putDecimal(footer, "expanded_shape_nodes", std::to_string(expandedShapeNodes));
              putDecimal(footer, "bound_rejected_schedule_branches", std::to_string(boundRejectedBranches));
              putDecimal(footer, "geometry_time_bound_rejected_branches", std::to_string(geometryTimeBoundRejectedBranches));
              putDecimal(footer, "prefix_window_bound_calls",
                         std::to_string(prefixWindowBoundCalls));
              putDecimal(footer, "prefix_window_bound_rejected",
                         std::to_string(prefixWindowBoundRejected));
              footer["tie_restricted_cp_scope"] =
                  "local-equal-cycle-canonical-ordinal";
              putDecimal(footer, "tie_restricted_cp_bound_calls",
                         std::to_string(tieRestrictedCpBoundCalls));
              putDecimal(footer, "tie_restricted_cp_bound_pruned",
                         std::to_string(tieRestrictedCpBoundPruned));
              putDecimal(footer, "tie_restricted_cp_no_eligible",
                         std::to_string(tieRestrictedCpNoEligible));
              putDecimal(footer, "tie_restricted_cp_invalid",
                         std::to_string(tieRestrictedCpInvalid));
              putDecimal(footer, "seed_schedules", std::to_string(seedSchedules));
              footer["canonical_incumbent_seed_enabled"] =
                  canonicalIncumbentSeed.getValue();
              footer["canonical_seed_continuation_state"] =
                  canonicalIncumbentSeed.getValue()
                      ? (canonicalSeedContinuationComplete ? "complete"
                                                            : "incomplete")
                      : "disabled";
              footer["canonical_seed_scheduler_calls"] =
                  static_cast<int64_t>(canonicalSeedSchedulerCalls);
              footer["canonical_seed_accepted"] =
                  static_cast<int64_t>(canonicalSeedAccepted);
              footer["placement_coverage_limited"] = scheduleSpace != "exact";
              footer["schedule_space"] = scheduleSpace.getValue();
              footer["production_ready"] = false;
              footer["state_memo_requested"] = scheduleSpace == "exact";
              footer["state_memo_enabled"] = stateMemoEnabled;
              footer["state_memo_canonical_state_available"] =
                  stateMemoCanonicalStateAvailable;
              footer["state_memo_external_cutoff_monotonic"] =
                  scheduleSpace == "exact";
              footer["state_memo_cutoff_violation"] =
                  stateMemoCutoffViolation;
              footer["state_memo_capacity"] = stateMemoCapacity.getValue();
              footer["state_memo_entries"] =
                  static_cast<int64_t>(stateMemoEntries);
              footer["state_memo_hits"] = static_cast<int64_t>(stateMemoHits);
              footer["state_memo_pruned_subtrees"] =
                  static_cast<int64_t>(stateMemoPrunedSubtrees);
              footer["state_memo_capacity_skips"] =
                  static_cast<int64_t>(stateMemoCapacitySkips);
              footer["optimistic_ends_memo_enabled"] = optimisticEndsMemoEnabled;
              putDecimal(footer, "optimistic_ends_computations", std::to_string(optimisticEndsComputations));
              putDecimal(footer, "optimistic_ends_memo_hits", std::to_string(optimisticEndsMemoHits));
              putDecimal(footer, "optimistic_ends_memo_entries", std::to_string(optimisticEndsMemoEntries));
              putDecimal(footer, "optimistic_ends_memo_capacity_skips", std::to_string(optimisticEndsMemoCapacitySkips));
              footer["unsupported_shape_candidates"] =
                  static_cast<int64_t>(unsupportedShapes);
              footer["global_cutoff_requested"] = globalIncumbent.requested;
              footer["global_cutoff_complete"] = globallyCutoffResult;
              footer["global_cutoff_proven"] = globalCutoffProof;
              footer["global_cutoff_graph_pruned"] = globalGraphPruned;
              footer["global_cutoff_proof_kind"] = globalCutoffProofKind;
              footer["global_cutoff_shape_domain_contradiction"] =
                  globalShapeDomainContradiction;
              if (globalShapeDomainContradiction)
                footer["global_cutoff_shape_domain_contradiction_task"] =
                    static_cast<int64_t>(globalShapeDomainContradictionTask);
              footer["global_cutoff_omitted_candidates_strictly_worse"] =
                  globallyCutoffResult && globalGraphPruned;
              footer["global_cutoff_omitted_candidates_stable_key_dominated"] =
                  globallyCutoffResult;
              footer["global_cutoff_key_policy"] = globalIncumbent.keyPolicy;
              putDecimal(footer, "global_cutoff_pruned_shape_candidates",
                         exactShapeIndexString(globalBoundPrunedShapes));
              if (globalIncumbent.requested) {
                footer["global_cutoff_cycles"] = globalIncumbent.fifthCycles;
                footer["global_cutoff_graph_lower_bound"] = graphLowerBound;
                footer["global_cutoff_evidence_path"] = globalIncumbent.path;
                footer["global_cutoff_evidence_candidate_count"] =
                    static_cast<int64_t>(globalIncumbent.candidateCount);
                footer["global_cutoff_evidence_source_repository"] =
                    globalIncumbent.sourceRepository;
                footer["global_cutoff_evidence_source_commit"] =
                    globalIncumbent.sourceCommit;
                footer["global_cutoff_evidence_architecture_path"] =
                    globalIncumbent.architecturePath;
                footer["global_cutoff_evidence_function"] =
                    globalIncumbent.function;
                footer["global_cutoff_evidence_stage"] = globalIncumbent.stage;
                footer["global_cutoff_evidence_closure_complete"] =
                    globalIncumbent.evidenceClosureComplete;
                footer["global_cutoff_evidence_seed_only"] =
                    globalIncumbent.evidenceSeedOnly;
                footer["global_cutoff_evidence_scope"] =
                    globalIncumbent.evidenceClosureComplete
                        ? "complete-stage-closure"
                        : "bounded-legal-member";
              }
              putDecimal(footer, "expanded_schedule_nodes",
                         std::to_string(expandedNodes));
              putDecimal(footer, "overlap_rejected_branches",
                         std::to_string(overlapRejectedBranches));
              putDecimal(footer, "out_of_bounds_rejected_branches",
                         std::to_string(outOfBoundsRejectedBranches));
              putDecimal(footer, "schedule_paths",
                         std::to_string(schedulePaths));
              putDecimal(footer, "global_cutoff_filtered_schedule_callbacks",
                         std::to_string(globalCutoffFilteredScheduleCallbacks));
              footer["dispatch_policy"] = dispatchPolicy.getValue();
              footer["dispatch_policy_scope"] =
                  selectedDispatchPolicy == DispatchPolicy::Fixed
                      ? "canonical-smallest-ready-task"
                      : productionScheduler ? "production-critical-path-dependency-ready"
                                            : "all-ready-task-orders";
              footer["horizon_proven"] = !seedOnly.getValue() && horizonCertified;
              footer["top_k_requested"] = topK.getValue();
              footer["top_k_certified"] = !seedOnly.getValue() && complete;
              footer["ranking_certificate"] =
                  seedOnly.getValue() ? "feasible-seed-only"
                  : complete ? scoreRankingCertificate(temporalModel, scheduleSpace).str()
                           : globallyComplete ? "proven-global-incumbent-cycle-cutoff"
                           : "best-found-uncertified";
              putDecimal(footer, "production_scheduler_calls", std::to_string(productionSchedulerCalls));
              footer["orbit_4x4_pruning"] = !productionScheduler;
              if (productionScheduler)
                footer["tie_key_policy"] =
                    kProductionNumericTieKeyPolicy.str();
              footer["score_cache_enabled"] = scoreCacheRequested;
              footer["score_cache_disabled_for_global_cutoff"] =
                  scoreCacheDisabledForGlobalCutoff;
              footer["score_cache_hit"] = scoreCacheHit;
              footer["score_cache_reused"] = false;
              footer["score_cache_migrated"] = scoreCacheMigrated;
              footer["score_cache_new_expanded_nodes"] =
                  static_cast<int64_t>(expandedNodes);
              footer["score_cache_peer_requested"] = scoreCachePeerRequested;
              footer["score_cache_peer_imported"] = scoreCachePeerImported;
              footer["score_cache_peer_pending"] = scoreCachePeerPending;
              footer["score_cache_peer_status"] = scoreCachePeerStatus;
              llvm::json::Array shortlist;
              for (auto [rank, entry] : llvm::enumerate(ranked)) {
                llvm::json::Object item;
                item["rank"] = static_cast<int64_t>(rank);
                item["candidate_id"] = entry.key;
                item["predicted_whole_program_cycles"] = entry.cycles;
                shortlist.push_back(std::move(item));
              }
              footer["shortlist"] = std::move(shortlist);
              footer["ranking_pruning"] = rankingPruning.getValue();
        footer["shape_shard_count"] = shapeShardCount.getValue();
        footer["shape_shard_index"] = shapeShardIndex.getValue();
        putDecimal(footer, "shape_partition_begin", exactShapeIndexString(shardBegin));
        putDecimal(footer, "shape_partition_end", exactShapeIndexString(shardEnd));
        putDecimal(footer, "shape_partition_candidate_count", exactShapeIndexString(shardSize));
        putDecimal(footer, "partition_skipped_subtrees", std::to_string(partitionSkippedSubtrees));
        footer["elapsed_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
        addProductionCoverage(footer, temporalModel, scheduleSpace, dispatchPolicy, function);
        writeJsonLine(stream, std::move(footer));
              return true;
            },
            error)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    if (complete && scoreCache.owner()) {
      std::string cacheError;
      if (!scoreCache.publish(outputFile, cacheError))
        function.emitRemark() << "exact score cache publication skipped: "
                              << cacheError;
    } else if (!complete && scoreCache.owner()) {
      scoreCache.discard();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::amoeba::neura::createScoreExactJointTaskCandidatesPass() {
  return std::make_unique<ScoreExactJointTaskCandidatesPass>();
}
