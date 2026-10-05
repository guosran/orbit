//===- RetimingOriginalAmoebaFixedDecisionsPass.cpp ----------------------===//
// Replays an original AMOEBA throughput-guided trace with common mapped
// durations and explicit communication while holding every original spatial
// and dispatch decision fixed.
//===----------------------------------------------------------------------===//

#include "RetimingOriginalAmoebaFixedDecisionsPass.h"

#include "AnalyticalMLPInference.h"
#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalTaskCostCatalog.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.h"
#include "Backend/Neura/Orchestration/JointScheduling/SourceIterationDomainPartitionProof.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "OriginalAmoebaFixedDecisionRetimer.h"
#include "OriginalAmoebaProfileCostAdapter.h"
#include "NeighborhoodReplaySelection.h"
#include "ProductionFixedScheduleCommunication.h"
#include "SpatialTaskCandidateSpace.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;
using mlir::taskflow::TaskflowTaskOp;

namespace {

constexpr llvm::StringLiteral kProfileFileFormat = "amoeba-task-profile-v1";
constexpr llvm::StringLiteral kOriginalBaselineGraphVariantId =
    "original-amoeba-full";

constexpr llvm::StringLiteral kOriginalScheduleInfoAttr =
    "amoeba.task_scheduler_schedule_info";
constexpr llvm::StringLiteral kOriginalScheduleTableAttr =
    "amoeba.task_scheduler_task_schedule";
constexpr llvm::StringLiteral kOriginalDispatchAttr =
    "amoeba.task_scheduler_dispatch_order";
constexpr llvm::StringLiteral kOriginalSummaryAttr =
    "task_orchestration_summary";
constexpr llvm::StringLiteral kOriginalTimeUnitAttr =
    "amoeba.task_scheduler_time_unit";
constexpr llvm::StringLiteral kOriginalTimeScaleAttr =
    "amoeba.task_scheduler_time_scale";
constexpr llvm::StringLiteral kOriginalMaterializedDecisionsAttr =
    "amoeba.original_amoeba.materialized_decisions";
constexpr llvm::StringLiteral kOriginalReplicaDecisionAttr =
    "amoeba.original_amoeba.replica_decision";
constexpr llvm::StringLiteral kOriginalReplicaPlacementsAttr =
    "amoeba.original_amoeba.replica_placements";
constexpr llvm::StringLiteral kOriginalPartitionRealizationAttr =
    "amoeba.original_amoeba.source_partition_realization";
constexpr llvm::StringLiteral kReplicaProfileEvidenceSchema =
    "amoeba-original-replica-profile-evidence-v1";
constexpr llvm::StringLiteral kReplicaMaterializationSchema =
    "amoeba-original-replica-materialization-v1";

constexpr std::array<llvm::StringLiteral, 8> kOriginalProfileShapes = {
    "1x1", "1x2", "2x1", "1x3", "3x1", "2x2", "1x4", "4x1"};

struct OriginalReplicaTrace {
  unsigned replicaId = 0;
  std::string shape;
  int64_t cgraCount = 0;
  int row = 0;
  int col = 0;
  int rows = 1;
  int cols = 1;
  std::vector<OriginalAmoebaOccupiedCell> occupiedCells;
  std::vector<int64_t> contextIds;
};

struct SourceDomainBound {
  int64_t ordinal = 0;
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 1;
};

struct OriginalTraceTask {
  OriginalAmoebaFixedPlacement placement;
  std::vector<OriginalReplicaTrace> replicas;
  std::string selectedShape;
  std::string actualShape;
  int64_t selectedCount = 0;
  int64_t profileDuration = 0;
  int64_t fullMapperDuration = 0;
  int64_t compiledII = 0;
  int64_t materializedOperationCount = 0;
  int64_t sampleTripCount = 0;
  int64_t steps = 0;
  int64_t schedulerStart = 0;
  int64_t schedulerEnd = 0;
  int64_t schedulerDuration = 0;
  std::vector<int64_t> contextIds;
  std::vector<OriginalAmoebaOccupiedCell> originalOccupiedCells;
  std::vector<int64_t> originalContextIds;
  std::optional<unsigned> sourceReplicaId;
};

struct OriginalReplicaProfileEvidenceRef {
  std::string parentTask;
  unsigned replicaId = 0;
  std::string materializedTask;
  std::string function;
  std::string materializedModule;
  std::string profileFile;
  std::string bodyExportFile;
};

struct OriginalReplicaProfileEvidence {
  OriginalAmoebaProfileBodyEvidence body;
  std::string profileFile;
  std::string bodyExportFile;
  std::string selectedShape;
  int64_t cgraCount = 0;
  int64_t compiledII = 0;
  int64_t steps = 0;
  int64_t sampleTripCount = 0;
  int64_t materializedOperationCount = 0;
  int64_t estimatedLatency = 0;
};

struct MaterializedReplicaDecision {
  unsigned replicaId = 0;
  std::string taskName;
  DictionaryAttr shape;
  ArrayAttr placements;
  ArrayAttr partitionBounds;
  int64_t sourceWorkCount = 0;
  int64_t tripCount = 0;
  int64_t sourceMultiplicity = 0;
  std::vector<int64_t> expandedInternalExtents;
  std::vector<SourceDomainBound> decodedPartitionBounds;
};

struct MaterializedParentDecision {
  std::string parentTask;
  unsigned replicaCount = 0;
  int64_t selectedCounterOrdinal = -1;
  int64_t selectedOutputAxis = -1;
  ArrayAttr originalDomain;
  int64_t sourceMultiplicity = 0;
  int64_t sourceWorkCount = 0;
  std::vector<int64_t> expandedInternalExtents;
  std::vector<SourceDomainBound> decodedOriginalDomain;
  std::map<unsigned, MaterializedReplicaDecision> replicas;
};

struct ReplayTaskRecord {
  TaskMetadata task;
  OriginalTraceTask trace;
  OriginalAmoebaProfileBodyEvidence body;
  std::string parentTask;
  std::optional<unsigned> replicaId;
  std::string profileFile;
  std::string bodyExportFile;
};

static std::optional<int64_t> getInteger(DictionaryAttr dictionary,
                                         StringRef key) {
  if (!dictionary)
    return std::nullopt;
  auto value = dyn_cast_or_null<IntegerAttr>(dictionary.get(key));
  if (!value)
    return std::nullopt;
  return value.getInt();
}

static std::optional<StringRef> getString(DictionaryAttr dictionary,
                                          StringRef key) {
  if (!dictionary)
    return std::nullopt;
  auto value = dyn_cast_or_null<StringAttr>(dictionary.get(key));
  if (!value)
    return std::nullopt;
  return value.getValue();
}

static bool parseShape(StringRef text, int64_t &rows, int64_t &cols) {
  auto parts = text.split('x');
  return !parts.first.empty() && !parts.second.empty() &&
         !parts.second.contains('x') && !parts.first.getAsInteger(10, rows) &&
         !parts.second.getAsInteger(10, cols) && rows > 0 && cols > 0 &&
         rows <= 4 && cols <= 4 && rows * cols <= 4;
}

static bool checkedProfileDuration(int64_t compiledII, int64_t tripCount,
                                   int64_t steps, int64_t &duration) {
  if (compiledII < 1 || tripCount < 1 || steps < 1 ||
      tripCount - 1 >
          (std::numeric_limits<int64_t>::max() - steps) / compiledII)
    return false;
  duration = compiledII * (tripCount - 1) + steps;
  return duration > 0;
}

// Validate the historical F45 profile_info field only; native child costs use
// each child's full mapper profile and actual trip count.
static bool checkedF45SchedulerDuration(int64_t fullMapperDuration,
                                        int64_t activeReplicas,
                                        int64_t &schedulerDuration) {
  if (fullMapperDuration < 1 || activeReplicas < 1)
    return false;
  schedulerDuration = fullMapperDuration / activeReplicas +
                      (fullMapperDuration % activeReplicas != 0 ? 1 : 0);
  return schedulerDuration > 0;
}

static bool readReplicaProfileEvidenceManifest(
    StringRef path, StringRef expectedFunction,
    const llvm::StringMap<unsigned> &expectedReplicaCounts,
    std::map<std::pair<std::string, unsigned>,
             OriginalReplicaProfileEvidenceRef> &records,
    std::string &materializedModule, std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read original replica profile evidence manifest " +
            path.str() + ": " + buffer.getError().message();
    return false;
  }
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "invalid original replica profile evidence manifest: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  auto schema = root ? root->getString("schema") : std::nullopt;
  const llvm::json::Array *rawRecords =
      root ? root->getArray("records") : nullptr;
  if (!root || root->size() != 2 || !schema ||
      *schema != kReplicaProfileEvidenceSchema || !rawRecords) {
    error = "replica profile evidence manifest has the wrong exact schema";
    return false;
  }

  std::map<std::pair<std::string, unsigned>,
           OriginalReplicaProfileEvidenceRef> parsedRecords;
  std::set<std::string> childNames;
  std::string commonModule;
  for (const llvm::json::Value &rawRecord : *rawRecords) {
    const llvm::json::Object *record = rawRecord.getAsObject();
    auto parent = record ? record->getString("parent_task") : std::nullopt;
    auto replica = record ? record->getInteger("replica_id") : std::nullopt;
    auto child = record ? record->getString("materialized_task") : std::nullopt;
    auto function = record ? record->getString("function") : std::nullopt;
    auto materialized =
        record ? record->getString("materialized_module") : std::nullopt;
    auto profile = record ? record->getString("profile_file") : std::nullopt;
    auto body = record ? record->getString("body_export_file") : std::nullopt;
    auto expected = parent ? expectedReplicaCounts.find(*parent)
                           : expectedReplicaCounts.end();
    if (!record || record->size() != 7 || !parent || parent->empty() ||
        expected == expectedReplicaCounts.end() || !replica || *replica < 0 ||
        *replica >= expected->second || !child || child->empty() ||
        !function || *function != expectedFunction || !materialized ||
        materialized->empty() || !profile || profile->empty() || !body ||
        body->empty()) {
      error = "replica profile evidence record is malformed or names an "
              "unknown split parent, replica, or function";
      return false;
    }
    OriginalReplicaProfileEvidenceRef reference;
    reference.parentTask = parent->str();
    reference.replicaId = static_cast<unsigned>(*replica);
    reference.materializedTask = child->str();
    reference.function = function->str();
    reference.materializedModule = materialized->str();
    reference.profileFile = profile->str();
    reference.bodyExportFile = body->str();
    if (commonModule.empty())
      commonModule = reference.materializedModule;
    if (commonModule != reference.materializedModule ||
        !childNames.insert(reference.materializedTask).second ||
        !parsedRecords
             .emplace(std::make_pair(reference.parentTask,
                                     reference.replicaId),
                      std::move(reference))
             .second) {
      error = "replica profile evidence repeats a child/replica or refers to "
              "multiple materialized modules";
      return false;
    }
  }
  size_t expectedRecords = 0;
  for (const auto &entry : expectedReplicaCounts)
    expectedRecords += entry.second;
  if (parsedRecords.size() != expectedRecords) {
    error = "replica profile evidence does not contain exactly one record "
            "for every original split replica";
    return false;
  }
  for (const auto &entry : expectedReplicaCounts)
    for (unsigned replicaId = 0; replicaId < entry.second; ++replicaId)
      if (!parsedRecords.count({entry.getKey().str(), replicaId})) {
        error = "replica profile evidence omits parent=" +
                entry.getKey().str() + " replica_id=" +
                std::to_string(replicaId);
        return false;
      }
  if (commonModule.empty()) {
    error = "replica profile evidence has no materialized child records";
    return false;
  }
  records = std::move(parsedRecords);
  materializedModule = std::move(commonModule);
  return true;
}

static bool readCanonicalModuleWitness(StringRef costPath,
                                       std::string &witness,
                                       std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(costPath);
  if (!buffer) {
    error = "cannot read cost catalog for canonical source witness: " +
            buffer.getError().message();
    return false;
  }
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "cannot parse cost catalog for canonical source witness: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  const llvm::json::Object *binding =
      root ? root->getObject("original_amoeba_profile_binding") : nullptr;
  auto bound = binding ? binding->getString("canonical_module_witness")
                       : std::nullopt;
  if (!root || !binding || !bound || bound->empty()) {
    error = "multi-replica retiming requires the exact canonical source module "
            "in the original profile binding";
    return false;
  }
  witness = bound->str();
  return true;
}

static bool verifyEvidenceMaterializedModule(StringRef path,
                                            StringRef expectedFunction,
                                            MLIRContext *context,
                                            StringRef currentModuleWitness,
                                            std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read replica evidence materialized module " + path.str() +
            ": " + buffer.getError().message();
    return false;
  }
  ParserConfig parserConfig(context);
  OwningOpRef<ModuleOp> evidenceModule = parseSourceString<ModuleOp>(
      (*buffer)->getBuffer(), parserConfig, path);
  if (!evidenceModule) {
    error = "replica evidence materialized module could not be parsed";
    return false;
  }
  FailureOr<func::FuncOp> evidenceFunction =
      selectTaskFunction(*evidenceModule, expectedFunction, error);
  if (failed(evidenceFunction))
    return false;
  if (neighborhoodReplaySourceText(*evidenceModule) !=
      currentModuleWitness) {
    error = "replica evidence materialized module differs from the exact "
            "current retimer input";
    return false;
  }
  return true;
}

static bool decodeSourceDomainBounds(ArrayAttr array,
                                     std::vector<SourceDomainBound> &bounds,
                                     StringRef label, std::string &error) {
  if (!array || array.empty()) {
    error = label.str() + " must be a nonempty exact source-bound array";
    return false;
  }
  bounds.clear();
  bounds.reserve(array.size());
  for (auto [ordinal, attribute] : llvm::enumerate(array)) {
    auto axis = dyn_cast<DictionaryAttr>(attribute);
    auto actualOrdinal = getInteger(axis, "ordinal");
    auto lower = getInteger(axis, "lower");
    auto upper = getInteger(axis, "upper");
    auto step = getInteger(axis, "step");
    if (!actualOrdinal || *actualOrdinal != static_cast<int64_t>(ordinal) ||
        !lower || !upper || *upper <= *lower || !step || *step != 1) {
      error = label.str() + " contains a malformed or non-unit-step interval";
      return false;
    }
    bounds.push_back({*actualOrdinal, *lower, *upper, *step});
  }
  return true;
}

static bool sourceDomainVolume(ArrayRef<SourceDomainBound> bounds,
                               int64_t &volume, StringRef label,
                               std::string &error) {
  volume = 1;
  for (const SourceDomainBound &bound : bounds) {
    const __int128 wideExtent = static_cast<__int128>(bound.upper) -
                                static_cast<__int128>(bound.lower);
    if (wideExtent < 1 ||
        wideExtent > std::numeric_limits<int64_t>::max()) {
      error = label.str() + " volume is invalid or overflows signed 64-bit";
      return false;
    }
    const int64_t extent = static_cast<int64_t>(wideExtent);
    if (volume > std::numeric_limits<int64_t>::max() / extent) {
      error = label.str() + " volume is invalid or overflows signed 64-bit";
      return false;
    }
    volume *= extent;
  }
  return true;
}

static llvm::json::Array sourceDomainBoundsJson(
    ArrayRef<SourceDomainBound> bounds) {
  llvm::json::Array result;
  for (const SourceDomainBound &bound : bounds)
    result.push_back(llvm::json::Object{{"ordinal", bound.ordinal},
                                        {"lower", bound.lower},
                                        {"upper", bound.upper},
                                        {"step", bound.step}});
  return result;
}

static bool readReplicaProfileAndBody(
    const OriginalReplicaProfileEvidenceRef &reference,
    ArrayRef<std::string> expectedProfileTasks,
    ArrayRef<std::string> expectedBodyTasks, int64_t expectedTrip,
    StringRef expectedPlacedShape, int64_t expectedCgraCount,
    int64_t runtimeIICeiling,
    Operation *currentTask,
    const OriginalAmoebaProfileBodyEvidence &originalParentBody,
    OriginalReplicaProfileEvidence &result, std::string &error) {
  OriginalAmoebaProfileBodyExport bodyExport;
  if (!readOriginalAmoebaProfileBodyExport(
          reference.bodyExportFile, reference.function, expectedBodyTasks,
          bodyExport, error))
    return false;
  auto bodyIt = bodyExport.tasks.find(reference.materializedTask);
  if (bodyIt == bodyExport.tasks.end()) {
    error = "replica body export omits task=" + reference.materializedTask;
    return false;
  }
  const OriginalAmoebaProfileBodyEvidence &childBody = bodyIt->second;
  if (childBody.staticTripCount != expectedTrip ||
      childBody.normalizedMapperBody !=
          originalParentBody.normalizedMapperBody) {
    error = "replica body export trip count or normalized body differs from "
            "the canonical parent for task=" + reference.materializedTask;
    return false;
  }
  std::string currentBody;
  if (!computeOriginalAmoebaNormalizedMapperBody(currentTask, currentBody,
                                                error) ||
      currentBody != childBody.normalizedMapperBody) {
    if (error.empty())
      error = "replica body export does not exactly match current task=" +
              reference.materializedTask;
    return false;
  }

  auto buffer = llvm::MemoryBuffer::getFile(reference.profileFile);
  if (!buffer) {
    error = "cannot read replica profile " + reference.profileFile + ": " +
            buffer.getError().message();
    return false;
  }
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "invalid replica profile JSON: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  const llvm::json::Array *tasks = root ? root->getArray("tasks") : nullptr;
  const llvm::json::Array *attempts =
      root ? root->getArray("candidate_attempts") : nullptr;
  auto format = root ? root->getString("format") : std::nullopt;
  auto function = root ? root->getString("function") : std::nullopt;
  auto taskCount = root ? root->getInteger("task_count") : std::nullopt;
  auto expectedCandidates =
      root ? root->getInteger("expected_candidate_count") : std::nullopt;
  auto completedCandidates =
      root ? root->getInteger("completed_candidate_count") : std::nullopt;
  if (!root || !tasks || !attempts || !format || *format != kProfileFileFormat ||
      !function || *function != reference.function || !taskCount ||
      *taskCount != static_cast<int64_t>(tasks->size()) || !expectedCandidates ||
      !completedCandidates ||
      *expectedCandidates != *taskCount *
                                 static_cast<int64_t>(kOriginalProfileShapes.size()) ||
      *completedCandidates != *expectedCandidates ||
      *expectedCandidates != static_cast<int64_t>(attempts->size())) {
    error = "replica profile format or completed candidate count is invalid";
    return false;
  }
  std::set<std::string> taskNames;
  const llvm::json::Object *selectedTask = nullptr;
  for (const llvm::json::Value &rawTask : *tasks) {
    const llvm::json::Object *task = rawTask.getAsObject();
    auto name = task ? task->getString("task") : std::nullopt;
    if (!task || !name || name->empty() || !taskNames.insert(name->str()).second) {
      error = "replica profile task list is malformed or duplicated";
      return false;
    }
    if (*name == reference.materializedTask)
      selectedTask = task;
  }
  std::set<std::string> expectedTaskSet(expectedProfileTasks.begin(),
                                        expectedProfileTasks.end());
  std::set<std::string> attemptTaskNames;
  for (const llvm::json::Value &rawAttempt : *attempts) {
    const llvm::json::Object *attempt = rawAttempt.getAsObject();
    auto name = attempt ? attempt->getString("task") : std::nullopt;
    if (!name || !expectedTaskSet.count(name->str())) {
      error = "replica profile candidate attempts name a malformed or "
              "unexpected task";
      return false;
    }
    attemptTaskNames.insert(name->str());
  }
  if (taskNames != expectedTaskSet || attemptTaskNames != expectedTaskSet ||
      !selectedTask) {
    error = "replica profile task inventory does not exactly match manifest "
            "records for its file";
    return false;
  }
  const llvm::json::Array *profiles = selectedTask->getArray("profiles");
  if (!profiles) {
    error = "replica profile omits task=" + reference.materializedTask;
    return false;
  }
  std::map<std::string, const llvm::json::Object *> rowsByShape;
  for (const llvm::json::Value &rawProfile : *profiles) {
    const llvm::json::Object *profile = rawProfile.getAsObject();
    auto shape = profile ? profile->getString("composed_cgra_shape")
                         : std::nullopt;
    auto count = profile ? profile->getInteger("composed_cgra_count")
                         : std::nullopt;
    auto succeeded = profile ? profile->getBoolean("mapper_succeeded")
                             : std::nullopt;
    int64_t rows = 0, cols = 0;
    if (!profile || !shape ||
        !llvm::any_of(kOriginalProfileShapes, [&](StringRef expected) {
          return expected == *shape;
        }) || !count || !parseShape(*shape, rows, cols) ||
        *count != rows * cols || !succeeded || !*succeeded ||
        !rowsByShape.emplace(shape->str(), profile).second) {
      error = "replica profile contains malformed, duplicate, or unsupported "
              "shape rows for task=" + reference.materializedTask;
      return false;
    }
  }
  std::map<std::string, const llvm::json::Object *> attemptsByShape;
  for (const llvm::json::Value &rawAttempt : *attempts) {
    const llvm::json::Object *attempt = rawAttempt.getAsObject();
    auto taskName = attempt ? attempt->getString("task") : std::nullopt;
    if (!taskName || *taskName != reference.materializedTask)
      continue;
    auto index = attempt->getInteger("candidate_index_in_task");
    auto shape = attempt->getString("shape");
    auto count = attempt->getInteger("composed_cgra_count");
    auto created = attempt->getBoolean("profile_created");
    auto succeeded = attempt->getBoolean("mapper_succeeded");
    int64_t rows = 0, cols = 0;
    if (!index || *index < 1 ||
        *index > static_cast<int64_t>(kOriginalProfileShapes.size()) ||
        !shape || *shape != kOriginalProfileShapes[*index - 1] || !count ||
        !parseShape(*shape, rows, cols) || *count != rows * cols || !created ||
        !succeeded || !attemptsByShape.emplace(shape->str(), attempt).second) {
      error = "replica profile candidate-attempt inventory is malformed or "
              "duplicated for task=" + reference.materializedTask;
      return false;
    }
  }
  if (attemptsByShape.size() != kOriginalProfileShapes.size()) {
    error = "replica profile does not contain all eight explicit candidate "
            "attempts for task=" + reference.materializedTask;
    return false;
  }
  for (StringRef shape : kOriginalProfileShapes) {
    const llvm::json::Object &attempt = *attemptsByShape.at(shape.str());
    const bool created = *attempt.getBoolean("profile_created");
    const bool succeeded = *attempt.getBoolean("mapper_succeeded");
    auto row = rowsByShape.find(shape.str());
    const bool hasRow = row != rowsByShape.end();
    if (created != hasRow ||
        succeeded !=
            (created && hasRow &&
             row->second->getBoolean("mapper_succeeded") &&
             *row->second->getBoolean("mapper_succeeded"))) {
      error = "replica candidate attempt and emitted profile row disagree for "
              "task=" + reference.materializedTask + " shape=" + shape.str();
      return false;
    }
  }
  auto selected = rowsByShape.find(expectedPlacedShape.str());
  if (selected == rowsByShape.end()) {
    error = "replica profile omits the actual placed orientation " +
            expectedPlacedShape.str() + " for task=" +
            reference.materializedTask;
    return false;
  }
  const llvm::json::Object &profile = *selected->second;
  auto succeeded = profile.getBoolean("mapper_succeeded");
  auto count = profile.getInteger("composed_cgra_count");
  auto compiledII = profile.getInteger("compiled_ii");
  auto steps = profile.getInteger("steps");
  auto sampleTrip = profile.getInteger("sample_trip_count");
  auto operations = profile.getInteger("materialized_operation_count");
  auto latency = profile.getInteger("estimated_latency");
  int64_t formulaLatency = 0;
  if (!succeeded || !*succeeded || !count || *count != expectedCgraCount ||
      !compiledII || *compiledII < 1 || *compiledII > runtimeIICeiling ||
      !steps || *steps < 1 || !sampleTrip ||
      *sampleTrip != expectedTrip || !operations || *operations < 1 ||
      !latency || *latency < 1 ||
      !checkedProfileDuration(*compiledII, *sampleTrip, *steps,
                              formulaLatency) || formulaLatency != *latency) {
    error = "actual-orientation replica profile failed or disagrees with "
            "current body trip count for task=" + reference.materializedTask;
    return false;
  }
  result.body = childBody;
  result.profileFile = reference.profileFile;
  result.bodyExportFile = reference.bodyExportFile;
  result.selectedShape = expectedPlacedShape.str();
  result.cgraCount = *count;
  result.compiledII = *compiledII;
  result.steps = *steps;
  result.sampleTripCount = *sampleTrip;
  result.materializedOperationCount = *operations;
  result.estimatedLatency = *latency;
  return true;
}

static std::optional<int64_t> computeMappedDuration(int64_t compiledII,
                                                    double startupCycles,
                                                    int64_t tripCount,
                                                    std::string &error) {
  if (compiledII < 1 || tripCount < 1 || !std::isfinite(startupCycles) ||
      startupCycles <= 0.0) {
    error = "profiled II, actual trip count, or catalog startup is invalid";
    return std::nullopt;
  }
  long double duration = static_cast<long double>(startupCycles) +
                         static_cast<long double>(compiledII) *
                             static_cast<long double>(tripCount - 1);
  if (!std::isfinite(duration) || duration <= 0.0L ||
      duration >
          static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    error =
        "profiled-II/catalog-startup duration exceeds int64 scheduler range";
    return std::nullopt;
  }
  duration = std::ceil(duration);
  return std::max<int64_t>(1, static_cast<int64_t>(duration));
}

static bool readCostProvenance(StringRef path, std::string &repository,
                               std::string &commit,
                               std::string &architecturePath,
                               bool &productionReady, std::string &modelStatus,
                               std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read parent cost catalog " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "invalid parent cost catalog JSON: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  const llvm::json::Object *metadata =
      root ? root->getObject("predictor_metadata") : nullptr;
  auto sourceRepository =
      metadata ? metadata->getString("source_repository") : std::nullopt;
  auto sourceCommit =
      metadata ? metadata->getString("source_commit") : std::nullopt;
  auto sourceArchitecture =
      metadata ? metadata->getString("architecture_path") : std::nullopt;
  auto status = metadata ? metadata->getString("model_status") : std::nullopt;
  auto ready =
      metadata ? metadata->getBoolean("production_ready") : std::nullopt;
  if (!root || !metadata || !sourceRepository || !sourceCommit ||
      !sourceArchitecture || !status || !ready || sourceRepository->empty() ||
      sourceCommit->empty() || sourceArchitecture->empty() || status->empty()) {
    error = "parent cost catalog is missing bound source/architecture/model "
            "provenance";
    return false;
  }
  repository = sourceRepository->str();
  commit = sourceCommit->str();
  architecturePath = sourceArchitecture->str();
  productionReady = *ready;
  modelStatus = status->str();
  return true;
}

static bool closeDirectValue(double lhs, double rhs) {
  return std::isfinite(lhs) && std::isfinite(rhs) &&
         std::abs(lhs - rhs) <=
             1.0e-5 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

static bool verifyDiagnosticOverrideObject(
    const llvm::json::Object *overrideMetadata,
    StringRef trainingArchitectureText, StringRef runtimeArchitectureText,
    std::string &error) {
  llvm::json::Object expected;
  if (!makePerCgra2x2DiagnosticOverrideMetadata(
          trainingArchitectureText, runtimeArchitectureText, expected, error))
    return false;
  if (!overrideMetadata || overrideMetadata->size() != expected.size()) {
    error = "II23 catalog diagnostic override is absent or has unexpected keys";
    return false;
  }
  auto schema = overrideMetadata->getString("schema");
  auto trainingCeiling = overrideMetadata->getInteger("training_ii_ceiling");
  auto runtimeCeiling = overrideMetadata->getInteger("runtime_ii_ceiling");
  auto extrapolation = overrideMetadata->getBoolean("extrapolation_enabled");
  auto formal = overrideMetadata->getBoolean("formal");
  auto outputRule = overrideMetadata->getString("output_rule");
  auto trainingArchitecture =
      overrideMetadata->getString("training_architecture_exact_yaml_text");
  auto runtimeArchitecture =
      overrideMetadata->getString("runtime_architecture_exact_yaml_text");
  if (!schema || *schema != kPerCgra2x2DiagnosticSchema ||
      !trainingCeiling || *trainingCeiling != 20 || !runtimeCeiling ||
      *runtimeCeiling != 23 || !extrapolation || !*extrapolation || !formal ||
      *formal || !outputRule || *outputRule != kPerCgra2x2DiagnosticOutputRule ||
      !trainingArchitecture ||
      *trainingArchitecture != trainingArchitectureText ||
      !runtimeArchitecture ||
      *runtimeArchitecture != runtimeArchitectureText) {
    error = "II23 diagnostic override differs from exact training/runtime contract";
    return false;
  }
  return true;
}

static bool verifyDirectOriginalProfileBinding(
    StringRef costPath, StringRef bodyExportPath, func::FuncOp function,
    StringRef graphVariant, ArrayRef<TaskMetadata> tasks,
    ArrayRef<OriginalTraceTask> traces, int64_t perCgraRows,
    int64_t perCgraCols, int64_t runtimeIICeiling,
    std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(costPath);
  if (!buffer) {
    error = "cannot read direct-model original cost catalogue " +
            costPath.str() + ": " + buffer.getError().message();
    return false;
  }
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "cannot parse direct-model original cost catalogue: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  const llvm::json::Object *metadata =
      root ? root->getObject("predictor_metadata") : nullptr;
  const llvm::json::Object *binding =
      root ? root->getObject("original_amoeba_profile_binding") : nullptr;
  const llvm::json::Array *entries = root ? root->getArray("entries") : nullptr;
  const llvm::json::Array *boundTasks =
      binding ? binding->getArray("tasks") : nullptr;
  if (!root || !metadata || !binding || !entries || !boundTasks) {
    error = "direct-model cost catalogue lacks its profile/body binding";
    return false;
  }
  if (perCgraRows != 2 || perCgraCols != 2) {
    error = "direct-model original profile binding requires exactly 2x2 PEs "
            "per CGRA";
    return false;
  }
  auto schema = root->getString("schema");
  auto functionName = root->getString("function");
  auto nameSpace = root->getString("namespace");
  auto bindingSchema = binding->getString("schema");
  auto candidateId = binding->getString("candidate_id");
  auto boundFunction = binding->getString("function");
  auto boundGraph = binding->getString("graph_variant_id");
  auto bodyFormat = binding->getString("body_export_format");
  auto profileFormat = binding->getString("profile_file_format");
  auto functionSignature = binding->getString("function_signature");
  auto resultSignature = binding->getString("function_result_signature");
  auto profilePath = binding->getString("profile_file");
  auto architecturePath = binding->getString("architecture_path");
  auto architectureContract = binding->getString("architecture_contract");
  auto architectureText = binding->getString("architecture_text");
  auto sourceRepository = binding->getString("source_repository");
  auto sourceCommit = binding->getString("source_commit");
  auto modelNamespace = binding->getString("model_namespace");
  auto modelName = binding->getString("model");
  auto modelSchema = binding->getString("model_schema");
  auto featureContract = binding->getString("feature_contract_id");
  auto featureExtractor = binding->getString("feature_extractor");
  auto shapeProtocol = binding->getString("shape_protocol_id");
  auto ensemblePath = binding->getString("ensemble_path");
  auto ensembleText = binding->getString("ensemble_text");
  auto sourceModelBinding = binding->getObject("source_model");
  auto metadataNamespace = metadata->getString("model_namespace");
  auto metadataModel = metadata->getString("model");
  auto metadataSchema = metadata->getString("model_schema");
  auto metadataFeature = metadata->getString("feature_contract_id");
  auto metadataExtractor = metadata->getString("feature_extractor");
  auto metadataFrontend = metadata->getString("feature_frontend");
  auto metadataShape = metadata->getString("shape_protocol_id");
  auto metadataArchitecturePath = metadata->getString("architecture_path");
  auto metadataArchitectureContract =
      metadata->getString("architecture_contract");
  auto metadataArchitectureSchema = metadata->getString("architecture_schema");
  auto metadataRepository = metadata->getString("source_repository");
  auto metadataCommit = metadata->getString("source_commit");
  auto metadataGraph = metadata->getString("source_graph_id");
  auto metadataCanonicalWitness =
      metadata->getString("canonical_module_witness");
  auto metadataStatus = metadata->getString("model_status");
  auto metadataQuality = metadata->getString("quality_status");
  auto productionReady = metadata->getBoolean("production_ready");
  auto candidateOnly = metadata->getBoolean("candidate_only");
  auto overlapAudit =
      metadata->getBoolean("amoeba_benchmark_overlap_audit_complete");
  auto oldLabels = metadata->getBoolean("old_4x4_labels_reused");
  auto wholeProgramClaim =
      metadata->getBoolean("supports_whole_program_latency_or_throughput_claim");
  auto coverageStatus =
      metadata->getString("iteration_domain_coverage_status");
  auto bindingCoverageStatus =
      binding->getString("source_iteration_domain_coverage_status");
  auto unsupportedPolicy =
      metadata->getString("unsupported_prediction_policy");
  auto modelIntervalMaxII = metadata->getNumber("model_interval_max_ii");
  auto diagnosticOnly = metadata->getBoolean("diagnostic_only");
  auto metadataFormal = metadata->getBoolean("formal");
  const llvm::json::Object *metadataDiagnosticOverride =
      metadata->getObject("diagnostic_override");
  const llvm::json::Object *bindingDiagnosticOverride =
      binding->getObject("diagnostic_override");
  auto coverageVerified =
      metadata->getBoolean("iteration_domain_coverage_verified");
  auto bodyEquivalenceChecked =
      metadata->getBoolean("body_equivalence_checked");
  auto bindingCoverageVerified =
      binding->getBoolean("source_iteration_domain_coverage_verified");
  auto bindingBodyVerified =
      binding->getBoolean("current_ir_body_equivalence_verified");
  const StringRef expectedCoverageStatus =
      originalAmoebaSourceDomainCoverageStatus(tasks);
  const bool bindingCoverageStatusMatches =
      runtimeIICeiling == 23
          ? bindingCoverageStatus &&
                *bindingCoverageStatus == expectedCoverageStatus
          : !bindingCoverageStatus;
  if (!schema || *schema != "amoeba-task-shape-cost" || !functionName ||
      *functionName != function.getSymName() || !nameSpace ||
      *nameSpace != kPerCgra2x2ModelNamespace || !bindingSchema ||
      *bindingSchema != kOriginalAmoebaProfileBindingSchema || !candidateId ||
      *candidateId != "candidate-0" || !boundFunction ||
      *boundFunction != function.getSymName() || !boundGraph ||
      *boundGraph != graphVariant || !bodyFormat ||
      *bodyFormat != "amoeba-pre-mapper-task-bodies-v1" || !profileFormat ||
      *profileFormat != kProfileFileFormat || !functionSignature ||
      !resultSignature || !profilePath || profilePath->empty() ||
      !architecturePath || architecturePath->empty() ||
      !architectureContract ||
      *architectureContract !=
          "neura-architecture-v1:amoeba_4x4_cgra_2x2_context6" ||
      !architectureText || architectureText->empty() || !sourceRepository ||
      *sourceRepository != "https://github.com/guosran/orbit.git" ||
      !sourceCommit ||
      *sourceCommit != "6a1b6fcf6e155651e96b3b881565ad58fdf0c03e" ||
      !modelNamespace || *modelNamespace != kPerCgra2x2ModelNamespace ||
      !modelName ||
      *modelName != "orbit-cgra-ii-per-cgra-2x2-direct-4member-v1" ||
      !modelSchema || *modelSchema != kPerCgra2x2EnsembleSchema ||
      !featureContract || *featureContract != kPerCgra2x2FeatureContractId ||
      !featureExtractor ||
      *featureExtractor !=
          "cgra_ii_predictor.mapper_model:mapper_feature_vector" ||
      !shapeProtocol || *shapeProtocol != kPerCgra2x2ShapeProtocolId ||
      !ensemblePath || ensemblePath->empty() || !ensembleText ||
      ensembleText->empty() || !sourceModelBinding || !metadataNamespace ||
      *metadataNamespace != *modelNamespace || !metadataModel ||
      *metadataModel != *modelName || !metadataSchema ||
      *metadataSchema != *modelSchema || !metadataFeature ||
      *metadataFeature != *featureContract || !metadataExtractor ||
      *metadataExtractor != *featureExtractor || !metadataShape ||
      *metadataShape != *shapeProtocol || !metadataFrontend ||
      *metadataFrontend != "c++-mlir-current-body-direct-148-v1" ||
      !metadataArchitecturePath ||
      *metadataArchitecturePath != *architecturePath ||
      !metadataArchitectureContract ||
      *metadataArchitectureContract != *architectureContract ||
      !metadataArchitectureSchema ||
      *metadataArchitectureSchema != "neura-architecture-v1" ||
      !metadataRepository || *metadataRepository != *sourceRepository ||
      !metadataCommit || *metadataCommit != *sourceCommit || !metadataGraph ||
      *metadataGraph != graphVariant || !metadataStatus ||
      *metadataStatus != "candidate_pending_amoeba_benchmark_overlap_audit" ||
      !metadataQuality || *metadataQuality != *metadataStatus ||
      !productionReady || *productionReady || !candidateOnly ||
      !*candidateOnly || !overlapAudit || *overlapAudit || !oldLabels ||
      *oldLabels || !wholeProgramClaim || *wholeProgramClaim ||
      !coverageStatus || *coverageStatus != expectedCoverageStatus ||
      !bindingCoverageStatusMatches ||
      !modelIntervalMaxII || *modelIntervalMaxII != 20.0 ||
      !unsupportedPolicy ||
      *unsupportedPolicy !=
          (runtimeIICeiling == 23
               ? StringRef("analytical-lower-bound-exceeds-diagnostic-runtime-ceiling-v1")
               : StringRef("analytical-lower-bound-exceeds-model-ceiling-v1")) ||
      !coverageVerified || !*coverageVerified || !bodyEquivalenceChecked ||
      !*bodyEquivalenceChecked || !bindingCoverageVerified ||
      !*bindingCoverageVerified || !bindingBodyVerified ||
      !*bindingBodyVerified || boundTasks->size() != tasks.size() ||
      traces.size() != tasks.size()) {
    error = "direct 2x2 cost provenance is stale, mixed with legacy model "
            "metadata, or not bound to the original graph/profile";
    return false;
  }
  if (runtimeIICeiling == 20) {
    if (metadataDiagnosticOverride || bindingDiagnosticOverride ||
        (diagnosticOnly && *diagnosticOnly) ||
        (metadataFormal && !*metadataFormal)) {
      error = "training-II20 catalogue carries an unexpected diagnostic override";
      return false;
    }
  } else if (runtimeIICeiling != 23 || !diagnosticOnly || !*diagnosticOnly ||
             !metadataFormal || *metadataFormal) {
    error = "II23 catalogue is not explicitly diagnostic and non-formal";
    return false;
  }
  if (runtimeIICeiling == 23) {
    ModuleOp parentModule = function->getParentOfType<ModuleOp>();
    if (!metadataCanonicalWitness ||
        *metadataCanonicalWitness != neighborhoodReplaySourceText(parentModule)) {
      error = "II23 direct-model catalogue witness does not match the exact "
              "current canonical module";
      return false;
    }
  } else if (metadataCanonicalWitness) {
    error = "training-II20 catalogue carries an unbound canonical module witness";
    return false;
  }

  auto actualArchitecture = llvm::MemoryBuffer::getFile(*architecturePath);
  if (!actualArchitecture ||
      (*actualArchitecture)->getBuffer() != *architectureText) {
    error = "direct model architecture text differs from its named 2x2 file";
    return false;
  }
  auto actualEnsemble = llvm::MemoryBuffer::getFile(*ensemblePath);
  if (!actualEnsemble || (*actualEnsemble)->getBuffer() != *ensembleText) {
    error = "direct model bundle text differs from its named source file";
    return false;
  }
  auto ensembleValue = llvm::json::parse(*ensembleText);
  if (!ensembleValue) {
    error = "direct model bundle text is invalid JSON: " +
            llvm::toString(ensembleValue.takeError());
    return false;
  }
  const llvm::json::Object *bundle = ensembleValue->getAsObject();
  const llvm::json::Object *bundleArchitecture =
      bundle ? bundle->getObject("architecture") : nullptr;
  const llvm::json::Object *bundleFeatures =
      bundle ? bundle->getObject("feature_contract") : nullptr;
  const llvm::json::Object *bundleSourceModel =
      bundle ? bundle->getObject("source_model") : nullptr;
  const llvm::json::Array *bundleMembers =
      bundle ? bundle->getArray("members") : nullptr;
  auto bundleSchema = bundle ? bundle->getString("schema") : std::nullopt;
  auto bundleNamespace =
      bundle ? bundle->getString("model_namespace") : std::nullopt;
  auto bundleArchitectureText =
      bundleArchitecture
          ? bundleArchitecture->getString("exact_yaml_text")
          : std::nullopt;
  auto bundleFeatureContract =
      bundleFeatures ? bundleFeatures->getString("contract_id")
                     : std::nullopt;
  auto bundleExtractor =
      bundleFeatures ? bundleFeatures->getString("extractor")
                     : std::nullopt;
  auto bundleShapeProtocol =
      bundleFeatures ? bundleFeatures->getString("shape_protocol_id")
                     : std::nullopt;
  if (!bundle || !bundleSchema || *bundleSchema != kPerCgra2x2EnsembleSchema ||
      !bundleNamespace || *bundleNamespace != kPerCgra2x2ModelNamespace ||
      !bundleArchitectureText || bundleArchitectureText->empty() ||
      !bundleFeatureContract ||
      *bundleFeatureContract != kPerCgra2x2FeatureContractId ||
      !bundleExtractor || *bundleExtractor != *featureExtractor ||
      !bundleShapeProtocol || *bundleShapeProtocol != kPerCgra2x2ShapeProtocolId ||
      !bundleSourceModel || !sourceModelBinding->getString("repository") ||
      !bundleSourceModel->getString("repository") ||
      *sourceModelBinding->getString("repository") !=
          *bundleSourceModel->getString("repository") ||
      *bundleSourceModel->getString("repository") !=
          "https://github.com/guosran/cgra-ii-predictor" ||
      !sourceModelBinding->getString("commit") ||
      !bundleSourceModel->getString("commit") ||
      *sourceModelBinding->getString("commit") !=
          *bundleSourceModel->getString("commit") ||
      *bundleSourceModel->getString("commit") !=
          "3ade31806cb4c92e31888109f7c42b8a77e4cbce" ||
      !sourceModelBinding->getString("branch") ||
      !bundleSourceModel->getString("branch") ||
      *sourceModelBinding->getString("branch") !=
          *bundleSourceModel->getString("branch") ||
      *bundleSourceModel->getString("branch") != "orbit-2x2-predictor" ||
      !bundleMembers || bundleMembers->size() != 4) {
    error = "embedded direct 2x2 package does not match the selected model "
            "namespace, architecture, or pinned source provenance";
    return false;
  }
  if (!validatePerCgra2x2RuntimeContract(runtimeIICeiling,
                                          *bundleArchitectureText,
                                          *architectureText, error))
    return false;
  if (runtimeIICeiling == 23) {
    if (!verifyDiagnosticOverrideObject(metadataDiagnosticOverride,
                                        *bundleArchitectureText,
                                        *architectureText, error) ||
        !verifyDiagnosticOverrideObject(bindingDiagnosticOverride,
                                        *bundleArchitectureText,
                                        *architectureText, error))
      return false;
  } else if (metadata->get("diagnostic_override") ||
             binding->get("diagnostic_override")) {
    error = "training-II20 catalogue carries a diagnostic_override object";
    return false;
  }
  const std::vector<int64_t> expectedSeeds = {17, 41, 113, 239};
  const llvm::json::Array *boundMembers =
      binding->getArray("direct_ensemble_members");
  const llvm::json::Object *metadataEnsemble =
      metadata->getObject("direct_ensemble");
  const llvm::json::Array *metadataSeeds =
      metadataEnsemble ? metadataEnsemble->getArray("member_seeds") : nullptr;
  if (!boundMembers || boundMembers->size() != expectedSeeds.size() ||
      !metadataEnsemble ||
      metadataEnsemble->getInteger("member_count") != 4 || !metadataSeeds ||
      metadataSeeds->size() != expectedSeeds.size() ||
      metadataEnsemble->getString("reduction") !=
          std::optional<StringRef>("arithmetic_mean") ||
      metadataEnsemble->getString("uncertainty") !=
          std::optional<StringRef>("population_standard_deviation")) {
    error = "direct ensemble member provenance is incomplete";
    return false;
  }
  for (unsigned index = 0; index < expectedSeeds.size(); ++index) {
    const llvm::json::Object *bundleMember =
        (*bundleMembers)[index].getAsObject();
    const llvm::json::Object *boundMember =
        (*boundMembers)[index].getAsObject();
    auto bundleIndex = bundleMember
                           ? bundleMember->getInteger("member_index")
                           : std::nullopt;
    auto bundleSeed = bundleMember ? bundleMember->getInteger("seed")
                                   : std::nullopt;
    auto boundIndex = boundMember ? boundMember->getInteger("member_index")
                                  : std::nullopt;
    auto boundSeed = boundMember ? boundMember->getInteger("seed")
                                 : std::nullopt;
    auto metadataSeed = (*metadataSeeds)[index].getAsInteger();
    if (!bundleMember || !boundMember || !bundleIndex ||
        *bundleIndex != index || !bundleSeed ||
        *bundleSeed != expectedSeeds[index] || !boundIndex ||
        *boundIndex != index || !boundSeed ||
        *boundSeed != expectedSeeds[index] || !metadataSeed ||
        *metadataSeed != expectedSeeds[index]) {
      error = "direct 2x2 four-member ensemble order/seed binding changed";
      return false;
    }
  }

  OriginalAmoebaProfileBodyExport bodyExport;
  std::vector<std::string> taskNames;
  for (const TaskMetadata &task : tasks)
    taskNames.push_back(task.name);
  if (!readOriginalAmoebaProfileBodyExport(
          bodyExportPath, function.getSymName(), taskNames, bodyExport, error))
    return false;

  using DirectCostKey = std::tuple<std::string, int64_t, int64_t>;
  std::map<DirectCostKey, const llvm::json::Object *> directCosts;
  for (const llvm::json::Value &rawEntry : *entries) {
    const llvm::json::Object *entry = rawEntry.getAsObject();
    auto task = entry ? entry->getString("task") : std::nullopt;
    auto rows = entry ? entry->getInteger("mapper_tile_rows") : std::nullopt;
    auto cols = entry ? entry->getInteger("mapper_tile_cols") : std::nullopt;
    if (!entry || !task || !rows || !cols || *rows <= 0 || *cols <= 0 ||
        !directCosts.emplace(DirectCostKey{task->str(), *rows, *cols}, entry)
             .second) {
      error = "direct cost catalogue has a malformed or duplicate task/shape row";
      return false;
    }
  }
  static constexpr std::array<std::pair<int64_t, int64_t>, 8> kCgraShapes =
      {{{1, 1}, {1, 2}, {2, 1}, {1, 3}, {3, 1}, {1, 4}, {2, 2}, {4, 1}}};
  if (directCosts.size() != tasks.size() * kCgraShapes.size() ||
      boundTasks->size() != tasks.size()) {
    error = "direct cost catalogue must contain all eight model shapes per "
            "original task";
    return false;
  }

  for (auto [index, rawTask] : llvm::enumerate(*boundTasks)) {
    const llvm::json::Object *record = rawTask.getAsObject();
    const TaskMetadata &task = tasks[index];
    const OriginalTraceTask &trace = traces[index];
    auto bodyIt = bodyExport.tasks.find(task.name);
    if (!record || bodyIt == bodyExport.tasks.end()) {
      error = "direct profile binding contains an invalid current task";
      return false;
    }
    const OriginalAmoebaProfileBodyEvidence &body = bodyIt->second;
    std::string currentBody;
    std::string bodyError;
    if (!computeOriginalAmoebaNormalizedMapperBody(
            task.op.operator->(), currentBody, bodyError) ||
        currentBody != body.normalizedMapperBody) {
      error = bodyError.empty()
                  ? "direct profile body export differs from current task=" +
                        task.name
                  : bodyError;
      return false;
    }
    auto recordTask = record->getString("task");
    auto taskSignature = record->getString("task_signature");
    auto counterSignature = record->getString("counter_signature");
    auto kernelSignature = record->getString("kernel_binding_signature");
    auto mapperBody = record->getString("normalized_mapper_body");
    auto staticTrip = record->getInteger("static_trip_count");
    auto selectedShape = record->getString("selected_cgra_shape");
    auto selectedCount = record->getInteger("selected_cgra_count");
    auto mapperRows = record->getInteger("mapper_tile_rows");
    auto mapperCols = record->getInteger("mapper_tile_cols");
    auto compiledII = record->getInteger("compiled_ii");
    auto steps = record->getInteger("steps");
    auto sampleTrip = record->getInteger("sample_trip_count");
    auto materialized = record->getInteger("materialized_operation_count");
    auto estimatedLatency = record->getInteger("estimated_latency");
    auto mapperSucceeded = record->getBoolean("mapper_succeeded");
    auto domainStatus =
        record->getString("source_iteration_domain_status");
    auto domainComplete =
        record->getBoolean("source_iteration_domain_complete");
    auto sourceWorkCount =
        record->getInteger("source_iteration_work_count");
    auto sourceMultiplicity =
        record->getInteger("source_iteration_multiplicity");
    const llvm::json::Array *expandedExtents =
        record->getArray("expanded_internal_extents");
    if (!recordTask || *recordTask != task.name || !taskSignature ||
        *taskSignature != body.taskSignature || !counterSignature ||
        *counterSignature != body.counterSignature || !kernelSignature ||
        *kernelSignature != body.kernelBindingSignature || !mapperBody ||
        *mapperBody != body.normalizedMapperBody || !staticTrip ||
        *staticTrip != body.staticTripCount || *staticTrip != task.tripCount ||
        !selectedShape || *selectedShape != trace.selectedShape ||
        !selectedCount || *selectedCount != trace.selectedCount ||
        !mapperRows || !mapperCols || !compiledII ||
        *compiledII != trace.compiledII || !steps || *steps != trace.steps ||
        !sampleTrip || *sampleTrip != trace.sampleTripCount ||
        *sampleTrip != task.tripCount || !materialized ||
        *materialized != trace.materializedOperationCount ||
        !estimatedLatency || *estimatedLatency != trace.fullMapperDuration ||
        !mapperSucceeded || !*mapperSucceeded || !domainStatus ||
        *domainStatus != task.sourceIterationDomainStatus ||
        !domainComplete || !*domainComplete || !sourceWorkCount ||
        *sourceWorkCount != task.sourceIterationWorkCount ||
        trace.placement.selectedRows >
            std::numeric_limits<int64_t>::max() / perCgraRows ||
        trace.placement.selectedCols >
            std::numeric_limits<int64_t>::max() / perCgraCols ||
        *mapperRows != trace.placement.selectedRows * perCgraRows ||
        *mapperCols != trace.placement.selectedCols * perCgraCols) {
      error = "direct cost selected profile/body evidence disagrees with the "
              "current original trace for task=" + task.name;
      return false;
    }
    if (runtimeIICeiling == 23) {
      if (!sourceMultiplicity ||
          *sourceMultiplicity != task.sourceIterationMultiplicity ||
          !expandedExtents ||
          expandedExtents->size() != task.expandedInternalExtents.size()) {
        error = "II23 direct profile binding omits complete static-internal source facts for task=" +
                task.name;
        return false;
      }
      for (auto [extentIndex, extentValue] : llvm::enumerate(*expandedExtents)) {
        auto extent = extentValue.getAsInteger();
        if (!extent || *extent != task.expandedInternalExtents[extentIndex]) {
          error = "direct profile binding has stale static-internal source extents for task=" +
                  task.name;
          return false;
        }
      }
    } else if (record->get("source_iteration_multiplicity") ||
               record->get("expanded_internal_extents")) {
      error = "training-II20 profile binding has unexpected internal-expansion facts";
      return false;
    }

    const llvm::json::Array *shapePredictions =
        record->getArray("direct_shape_predictions");
    if (!shapePredictions || shapePredictions->size() != kCgraShapes.size()) {
      error = "direct body binding does not contain all eight oriented shapes "
              "for task=" + task.name;
      return false;
    }
    for (auto [shapeIndex, rawShape] : llvm::enumerate(*shapePredictions)) {
      const llvm::json::Object *shape = rawShape.getAsObject();
      const auto [cgraRows, cgraCols] = kCgraShapes[shapeIndex];
      const int64_t expectedRows = cgraRows * perCgraRows;
      const int64_t expectedCols = cgraCols * perCgraCols;
      const llvm::json::Object *entry =
          directCosts[DirectCostKey{task.name, expectedRows, expectedCols}];
      auto boundRows = shape ? shape->getInteger("mapper_tile_rows")
                             : std::nullopt;
      auto boundCols = shape ? shape->getInteger("mapper_tile_cols")
                             : std::nullopt;
      auto boundCgraRows = shape ? shape->getInteger("cgra_rows")
                                 : std::nullopt;
      auto boundCgraCols = shape ? shape->getInteger("cgra_cols")
                                 : std::nullopt;
      auto supportStatus = shape ? shape->getString("support_status")
                                 : std::nullopt;
      auto lowerBound = shape ? shape->getNumber("analytical_lower_bound")
                              : std::nullopt;
      auto shapeRec = shape ? shape->getNumber("rec_mii") : std::nullopt;
      auto shapeRes = shape ? shape->getNumber("res_mii") : std::nullopt;
      auto shapeStartup =
          shape ? shape->getNumber("startup_cycles") : std::nullopt;
      auto entryStatus = entry ? entry->getString("support_status")
                               : std::nullopt;
      auto entryLowerBound =
          entry ? entry->getNumber("analytical_lower_bound") : std::nullopt;
      if (!shape || !entry || !boundRows || *boundRows != expectedRows ||
          !boundCols || *boundCols != expectedCols || !boundCgraRows ||
          *boundCgraRows != cgraRows || !boundCgraCols ||
          *boundCgraCols != cgraCols || !supportStatus || !entryStatus ||
          *supportStatus != *entryStatus || !lowerBound ||
          !std::isfinite(*lowerBound) || !entryLowerBound ||
          !closeDirectValue(*lowerBound, *entryLowerBound)) {
        error = "direct shape prediction does not match the exact model "
                "domain for task=" + task.name;
        return false;
      }
      if (*supportStatus == "unsupported") {
        auto reason = shape->getString("unsupported_reason");
        auto shapeStatus = shape->getString("status");
        auto status = entry->getString("status");
        auto entryReason = entry->getString("unsupported_reason");
        auto interval = entry->getNumber("model_interval_max_ii");
        auto shapeInterval = shape->getNumber("model_interval_max_ii");
        const StringRef expectedReason =
            runtimeIICeiling == 23
                ? StringRef("analytical-lower-bound-exceeds-diagnostic-runtime-ceiling")
                : StringRef("analytical-lower-bound-exceeds-model-ceiling");
        if (*lowerBound <= runtimeIICeiling || !reason ||
            *reason != expectedReason ||
            (runtimeIICeiling == 23 &&
             (!shapeStatus || *shapeStatus != "unsupported-model-domain")) ||
            !status || *status != "unsupported-model-domain" ||
            !entryReason || *entryReason != *reason || !interval ||
            *interval != 20.0 || !shapeInterval || *shapeInterval != 20.0 ||
            entry->get("predicted_ii") ||
            entry->get("direct_ensemble_members")) {
          error = "unsupported direct prediction is not explicitly fail-closed";
          return false;
        }
        if (runtimeIICeiling == 23) {
          auto trainingCeiling = entry->getInteger("training_ceiling_ii");
          auto runtimeCeiling = entry->getInteger("runtime_ceiling_ii");
          auto extrapolation = entry->getString("extrapolation_status");
          auto shapeTrainingCeiling = shape->getInteger("training_ceiling_ii");
          auto shapeRuntimeCeiling = shape->getInteger("runtime_ceiling_ii");
          auto shapeExtrapolation = shape->getString("extrapolation_status");
          if (!trainingCeiling || *trainingCeiling != 20 || !runtimeCeiling ||
              *runtimeCeiling != 23 || !extrapolation ||
              *extrapolation != "outside-diagnostic-runtime-domain" ||
              !shapeTrainingCeiling || *shapeTrainingCeiling != 20 ||
              !shapeRuntimeCeiling || *shapeRuntimeCeiling != 23 ||
              !shapeExtrapolation ||
              *shapeExtrapolation != "outside-diagnostic-runtime-domain") {
            error = "unsupported II23 row omits exact training/runtime-domain facts";
            return false;
          }
        } else if (entry->get("training_ceiling_ii") ||
                   entry->get("runtime_ceiling_ii") ||
                   entry->get("extrapolation_status") ||
                   shape->get("training_ceiling_ii") ||
                   shape->get("runtime_ceiling_ii") ||
                   shape->get("extrapolation_status")) {
          error = "ordinary II20 row carries unbound diagnostic metadata";
          return false;
        }
        continue;
      }
      if (*supportStatus != "supported" || *lowerBound > runtimeIICeiling) {
        error = "direct prediction has an unknown or out-of-domain status";
        return false;
      }
      auto boundMeanSource = shape->getString("ii_mean_source");
      auto entryMeanSource = entry->getString("ii_mean_source");
      auto entryModelStatus = entry->getString("model_status");
      auto entryCandidateOnly = entry->getBoolean("candidate_only");
      auto entryProductionReady = entry->getBoolean("production_ready");
      auto entryII = entry->getNumber("predicted_ii");
      auto entryStd = entry->getNumber("predicted_ii_std");
      auto entryRec = entry->getNumber("rec_mii");
      auto entryRes = entry->getNumber("res_mii");
      auto entryStartup = entry->getNumber("startup_cycles");
      const llvm::json::Array *boundMembers =
          shape->getArray("direct_ensemble_members");
      const llvm::json::Array *entryMembers =
          entry->getArray("direct_ensemble_members");
      if (!entryII || !entryStd || !entryRec || !entryRes || !entryStartup ||
          !boundMembers || !entryMembers || boundMembers->size() != 4 ||
          entryMembers->size() != 4 || !shapeRec || !shapeRes ||
          !shapeStartup || !boundMeanSource ||
          *boundMeanSource != "direct_four_member_arithmetic_mean" ||
          !entryMeanSource || *entryMeanSource != *boundMeanSource ||
          !entryModelStatus ||
          *entryModelStatus !=
              "candidate_pending_amoeba_benchmark_overlap_audit" ||
          !entryCandidateOnly || !*entryCandidateOnly ||
          !entryProductionReady || *entryProductionReady) {
        error = "supported direct prediction omits four-member numeric proof";
        return false;
      }
      double memberSum = 0.0;
      std::array<double, 4> memberValues{};
      for (unsigned memberIndex = 0; memberIndex < 4; ++memberIndex) {
        const llvm::json::Object *boundMember =
            (*boundMembers)[memberIndex].getAsObject();
        const llvm::json::Object *costMember =
            (*entryMembers)[memberIndex].getAsObject();
        auto boundMemberIndex = boundMember
                                    ? boundMember->getInteger("member_index")
                                    : std::nullopt;
        auto costMemberIndex = costMember
                                   ? costMember->getInteger("member_index")
                                   : std::nullopt;
        auto boundSeed = boundMember ? boundMember->getInteger("seed")
                                     : std::nullopt;
        auto costSeed = costMember ? costMember->getInteger("seed")
                                   : std::nullopt;
        auto boundPrediction =
            boundMember ? boundMember->getNumber("predicted_ii")
                        : std::nullopt;
        auto costPrediction =
            costMember ? costMember->getNumber("predicted_ii")
                       : std::nullopt;
        if (!boundMember || !costMember || !boundMemberIndex ||
            *boundMemberIndex != memberIndex || !costMemberIndex ||
            *costMemberIndex != memberIndex || !boundSeed ||
            *boundSeed != expectedSeeds[memberIndex] || !costSeed ||
            *costSeed != expectedSeeds[memberIndex] || !boundPrediction ||
            !costPrediction ||
            !closeDirectValue(*boundPrediction, *costPrediction) ||
            !std::isfinite(*costPrediction) ||
            *costPrediction < *lowerBound ||
            *costPrediction > runtimeIICeiling) {
          error = "direct ensemble member value or provenance is invalid for " +
                  task.name;
          return false;
        }
        memberValues[memberIndex] = *costPrediction;
        memberSum += *costPrediction;
      }
      if (runtimeIICeiling == 23) {
        auto entryTrainingCeiling = entry->getInteger("training_ceiling_ii");
        auto entryRuntimeCeiling = entry->getInteger("runtime_ceiling_ii");
        auto entryExtrapolation = entry->getString("extrapolation_status");
        auto shapeTrainingCeiling = shape->getInteger("training_ceiling_ii");
        auto shapeRuntimeCeiling = shape->getInteger("runtime_ceiling_ii");
        auto shapeExtrapolation = shape->getString("extrapolation_status");
        const bool extrapolated =
            *lowerBound > 20.0 ||
            llvm::any_of(memberValues,
                         [](double value) { return value > 20.0; });
        const StringRef expectedLabel =
            extrapolated ? "out-of-training-ceiling"
                         : "within-training-ceiling";
        if (!entryTrainingCeiling || *entryTrainingCeiling != 20 ||
            !entryRuntimeCeiling || *entryRuntimeCeiling != 23 ||
            !shapeTrainingCeiling || *shapeTrainingCeiling != 20 ||
            !shapeRuntimeCeiling || *shapeRuntimeCeiling != 23 ||
            !entryExtrapolation || *entryExtrapolation != expectedLabel ||
            !shapeExtrapolation || *shapeExtrapolation != expectedLabel) {
          error = "supported II23 row has stale training/runtime extrapolation labels";
          return false;
        }
      } else if (entry->get("training_ceiling_ii") ||
                 entry->get("runtime_ceiling_ii") ||
                 entry->get("extrapolation_status") ||
                 shape->get("training_ceiling_ii") ||
                 shape->get("runtime_ceiling_ii") ||
                 shape->get("extrapolation_status")) {
        error = "ordinary II20 row carries unbound diagnostic metadata";
        return false;
      }
      const double memberMean = memberSum / 4.0;
      double variance = 0.0;
      for (double member : memberValues) {
        const double delta = member - memberMean;
        variance += delta * delta;
      }
      if (!closeDirectValue(memberMean, *entryII) ||
          !closeDirectValue(std::sqrt(variance / 4.0), *entryStd) ||
          !closeDirectValue(*entryRec, *shapeRec) ||
          !closeDirectValue(*entryRes, *shapeRes) ||
          !closeDirectValue(*entryStartup, *shapeStartup)) {
        error = "direct ensemble mean/std or shape facts do not match member "
                "values for task=" + task.name;
        return false;
      }
    }
  }
  return true;
}

static bool verifyOriginalProfileBinding(
    StringRef costPath, StringRef bodyExportPath, func::FuncOp function,
    StringRef graphVariant, ArrayRef<TaskMetadata> tasks,
    ArrayRef<OriginalTraceTask> traces, int64_t perCgraRows,
    int64_t perCgraCols, int64_t runtimeIICeiling,
    StringRef expectedCanonicalModuleWitness,
    OriginalAmoebaProfileBodyExport &bodyExport,
    std::string &error) {
  if (bodyExportPath.empty()) {
    error = "profile-body-export-file is required to bind selected costs to "
            "the original normalized mapper bodies";
    return false;
  }
  std::vector<std::string> taskNames;
  taskNames.reserve(tasks.size());
  for (const TaskMetadata &task : tasks)
    taskNames.push_back(task.name);
  if (!readOriginalAmoebaProfileBodyExport(
          bodyExportPath, function.getSymName(), taskNames, bodyExport, error))
    return false;

  auto buffer = llvm::MemoryBuffer::getFile(costPath);
  if (!buffer) {
    error = "cannot read profile-bound cost catalogue " + costPath.str() +
            ": " + buffer.getError().message();
    return false;
  }
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "cannot parse profile-bound cost catalogue: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  const llvm::json::Object *metadata =
      root ? root->getObject("predictor_metadata") : nullptr;
  const llvm::json::Object *binding =
      root ? root->getObject("original_amoeba_profile_binding") : nullptr;
  if (!root || !metadata || !binding) {
    error = "cost catalogue has no exact original AMOEBA profile/body binding";
    return false;
  }
  const auto directNamespace = root->getString("namespace");
  const auto directModelNamespace = metadata->getString("model_namespace");
  const auto directModelSchema = metadata->getString("model_schema");
  const bool declaresDirect2x2 =
      (directNamespace && *directNamespace == kPerCgra2x2ModelNamespace) ||
      (directModelNamespace &&
       *directModelNamespace == kPerCgra2x2ModelNamespace) ||
      (directModelSchema && *directModelSchema == kPerCgra2x2EnsembleSchema);
  if (declaresDirect2x2) {
    if (!directNamespace ||
        *directNamespace != kPerCgra2x2ModelNamespace ||
        !directModelNamespace ||
        *directModelNamespace != kPerCgra2x2ModelNamespace ||
        !directModelSchema ||
        *directModelSchema != kPerCgra2x2EnsembleSchema) {
      error = "direct 2x2 original profile catalogue has a mixed or missing "
              "model namespace/schema";
      return false;
    }
    const bool hasMultiReplicaParent = llvm::any_of(
        tasks, [](const TaskMetadata &task) {
          auto active = task.op->getAttrOfType<IntegerAttr>("active_replicas");
          return active && active.getInt() > 1;
        });
    auto bindingCanonicalWitness =
        binding->getString("canonical_module_witness");
    if (hasMultiReplicaParent) {
      if (!bindingCanonicalWitness ||
          *bindingCanonicalWitness != expectedCanonicalModuleWitness ||
          expectedCanonicalModuleWitness.empty()) {
        error = "multi-replica parent profile binding lacks the exact "
                "canonical source module witness";
        return false;
      }
    } else if (bindingCanonicalWitness) {
      error = "single-replica parent profile binding carries an unrequested "
              "canonical module witness";
      return false;
    }
    return verifyDirectOriginalProfileBinding(
        costPath, bodyExportPath, function, graphVariant, tasks, traces,
        perCgraRows, perCgraCols, runtimeIICeiling, error);
  }
  if (runtimeIICeiling == 23) {
    error = "diagnostic II23 requires the direct per-CGRA 2x2 model catalog";
    return false;
  }
  auto schema = binding->getString("schema");
  auto candidateId = binding->getString("candidate_id");
  auto boundFunction = binding->getString("function");
  auto boundGraph = binding->getString("graph_variant_id");
  auto bodyFormat = binding->getString("body_export_format");
  auto profileFormat = binding->getString("profile_file_format");
  auto functionSignature = binding->getString("function_signature");
  auto resultSignature = binding->getString("function_result_signature");
  auto profilePath = binding->getString("profile_file");
  auto bindingArchitecturePath = binding->getString("architecture_path");
  auto architectureContract = binding->getString("architecture_contract");
  auto architectureText = binding->getString("architecture_text");
  auto bindingRepository = binding->getString("source_repository");
  auto bindingCommit = binding->getString("source_commit");
  auto modelName = binding->getString("model");
  auto modelSchema = binding->getString("model_schema");
  auto featureContract = binding->getString("feature_contract_id");
  auto featureExtractor = binding->getString("feature_extractor");
  auto shapeProtocol = binding->getString("shape_protocol_id");
  auto ensemblePath = binding->getString("ensemble_path");
  auto checkpointDirectory = binding->getString("checkpoint_directory");
  auto ensembleText = binding->getString("ensemble_text");
  const llvm::json::Array *checkpointTexts =
      binding->getArray("checkpoint_texts");
  auto metadataArchitectureContract =
      metadata->getString("architecture_contract");
  auto metadataArchitecturePath = metadata->getString("architecture_path");
  auto metadataArchitectureSchema = metadata->getString("architecture_schema");
  auto metadataModel = metadata->getString("model");
  auto metadataModelSchema = metadata->getString("model_schema");
  auto metadataFeatureContract = metadata->getString("feature_contract_id");
  auto metadataFeatureExtractor = metadata->getString("feature_extractor");
  auto metadataShapeProtocol = metadata->getString("shape_protocol_id");
  auto metadataRepository = metadata->getString("source_repository");
  auto metadataCommit = metadata->getString("source_commit");
  auto metadataGraph = metadata->getString("source_graph_id");
  auto metadataStatus = metadata->getString("model_status");
  auto metadataReady = metadata->getBoolean("production_ready");
  auto domainCoverageStatus =
      metadata->getString("iteration_domain_coverage_status");
  auto domainCoverageVerified =
      metadata->getBoolean("iteration_domain_coverage_verified");
  auto bindingDomainCoverageVerified =
      binding->getBoolean("source_iteration_domain_coverage_verified");
  auto bodyEquivalenceChecked =
      metadata->getBoolean("body_equivalence_checked");
  auto currentIRBodyVerified =
      binding->getBoolean("current_ir_body_equivalence_verified");
  const llvm::json::Array *boundTasks = binding->getArray("tasks");
  if (!schema || *schema != kOriginalAmoebaProfileBindingSchema ||
      !candidateId || *candidateId != "candidate-0" || !boundFunction ||
      *boundFunction != function.getSymName() || !boundGraph ||
      *boundGraph != graphVariant || !bodyFormat ||
      *bodyFormat != "amoeba-pre-mapper-task-bodies-v1" || !profileFormat ||
      *profileFormat != kProfileFileFormat || !functionSignature ||
      *functionSignature != bodyExport.functionSignature || !resultSignature ||
      *resultSignature != bodyExport.functionResultSignature || !profilePath ||
      profilePath->empty() || !bindingArchitecturePath ||
      bindingArchitecturePath->empty() || !architectureContract ||
      architectureContract->empty() || !architectureText ||
      architectureText->empty() || !bindingRepository || !metadataRepository ||
      *bindingRepository != *metadataRepository || !bindingCommit ||
      !metadataCommit || *bindingCommit != *metadataCommit || !modelName ||
      *modelName != "formal-max4-nohash-v2" || !modelSchema ||
      modelSchema->empty() || !featureContract || featureContract->empty() ||
      !featureExtractor || featureExtractor->empty() || !shapeProtocol ||
      shapeProtocol->empty() || !ensembleText || ensembleText->empty() ||
      !ensemblePath || ensemblePath->empty() || !checkpointDirectory ||
      checkpointDirectory->empty() || !checkpointTexts ||
      checkpointTexts->size() != 3 || !metadataArchitectureContract ||
      *metadataArchitectureContract != *architectureContract ||
      !metadataArchitecturePath || metadataArchitecturePath->empty() ||
      *metadataArchitecturePath != *bindingArchitecturePath ||
      !metadataArchitectureSchema ||
      *metadataArchitectureSchema != "neura-architecture-v1" ||
      !metadataModel || *metadataModel != *modelName || !metadataModelSchema ||
      *metadataModelSchema != *modelSchema || !metadataFeatureContract ||
      *metadataFeatureContract != *featureContract ||
      !metadataFeatureExtractor ||
      *metadataFeatureExtractor != *featureExtractor ||
      !metadataShapeProtocol || *metadataShapeProtocol != *shapeProtocol ||
      !metadataRepository || metadataRepository->empty() || !metadataCommit ||
      metadataCommit->empty() || !metadataGraph ||
      *metadataGraph != graphVariant || !metadataStatus ||
      *metadataStatus != "exploratory" || !metadataReady || *metadataReady ||
      !domainCoverageStatus ||
      *domainCoverageStatus !=
          kOriginalAmoebaSourceDomainCoverageStatus ||
      !domainCoverageVerified || !*domainCoverageVerified ||
      !bindingDomainCoverageVerified || !*bindingDomainCoverageVerified ||
      !bodyEquivalenceChecked || !*bodyEquivalenceChecked ||
      !currentIRBodyVerified || !*currentIRBodyVerified || !boundTasks ||
      boundTasks->size() != tasks.size() || traces.size() != tasks.size()) {
    error = "cost catalog body/profile binding does not match function, graph, "
            "or complete task coverage";
    return false;
  }
  for (const llvm::json::Value &rawCheckpoint : *checkpointTexts) {
    const llvm::json::Object *checkpoint = rawCheckpoint.getAsObject();
    auto name = checkpoint ? checkpoint->getString("name") : std::nullopt;
    auto text = checkpoint ? checkpoint->getString("text") : std::nullopt;
    if (!checkpoint || !name || name->empty() || !text || text->empty()) {
      error = "cost catalog model evidence omits an exact checkpoint source";
      return false;
    }
  }
  auto architectureBuffer =
      llvm::MemoryBuffer::getFile(*bindingArchitecturePath);
  if (!architectureBuffer ||
      (*architectureBuffer)->getBuffer() != *architectureText) {
    error = "embedded catalog architecture text differs from the file named "
            "by its provenance";
    return false;
  }
  auto ensembleValue = llvm::json::parse(*ensembleText);
  if (!ensembleValue) {
    error = "embedded model ensemble is not valid JSON: " +
            llvm::toString(ensembleValue.takeError());
    return false;
  }
  const llvm::json::Object *embeddedEnsemble = ensembleValue->getAsObject();
  auto embeddedSchema =
      embeddedEnsemble ? embeddedEnsemble->getString("schema") : std::nullopt;
  auto embeddedQuality = embeddedEnsemble
                             ? embeddedEnsemble->getString("quality_status")
                             : std::nullopt;
  auto embeddedReady = embeddedEnsemble
                           ? embeddedEnsemble->getBoolean("production_ready")
                           : std::nullopt;
  if (!embeddedSchema || *embeddedSchema != *modelSchema || !embeddedQuality ||
      *embeddedQuality != "exploratory" || !embeddedReady || *embeddedReady) {
    error = "embedded model evidence disagrees with exploratory catalogue "
            "metadata";
    return false;
  }
  auto ensembleBuffer = llvm::MemoryBuffer::getFile(*ensemblePath);
  if (!ensembleBuffer || (*ensembleBuffer)->getBuffer() != *ensembleText) {
    error = "embedded model ensemble text differs from the source file named "
            "by its provenance";
    return false;
  }
  const llvm::json::Object *embeddedCheckpoints =
      embeddedEnsemble->getObject("checkpoints");
  if (!embeddedCheckpoints || embeddedCheckpoints->size() != 3) {
    error = "embedded model ensemble does not enumerate exactly three "
            "checkpoints";
    return false;
  }
  std::set<std::string> checkedCheckpointNames;
  for (const llvm::json::Value &rawCheckpoint : *checkpointTexts) {
    const llvm::json::Object *checkpoint = rawCheckpoint.getAsObject();
    llvm::StringRef name = *checkpoint->getString("name");
    llvm::StringRef text = *checkpoint->getString("text");
    const llvm::json::Object *source = embeddedCheckpoints->getObject(name);
    auto relativePath = source ? source->getString("path") : std::nullopt;
    if (!relativePath || !checkedCheckpointNames.insert(name.str()).second) {
      error = "embedded model checkpoint names are missing or duplicated";
      return false;
    }
    llvm::SmallString<256> checkpointPath(*checkpointDirectory);
    if (llvm::sys::path::is_absolute(*relativePath))
      checkpointPath = *relativePath;
    else
      llvm::sys::path::append(checkpointPath, *relativePath);
    auto checkpointBuffer = llvm::MemoryBuffer::getFile(checkpointPath);
    if (!checkpointBuffer || (*checkpointBuffer)->getBuffer() != text) {
      error = "embedded model checkpoint text differs from the exact loaded "
              "source for " +
              name.str();
      return false;
    }
  }
  if (checkedCheckpointNames !=
      std::set<std::string>{"baseline", "large-operation", "ranking"}) {
    error = "embedded model checkpoint names differ from the formal ensemble";
    return false;
  }

  for (auto [index, rawRecord] : llvm::enumerate(*boundTasks)) {
    const llvm::json::Object *record = rawRecord.getAsObject();
    const TaskMetadata &task = tasks[index];
    const OriginalTraceTask &trace = traces[index];
    auto bodyIt = bodyExport.tasks.find(task.name);
    if (!record || bodyIt == bodyExport.tasks.end()) {
      error = "cost catalog profile binding contains an invalid task record";
      return false;
    }
    const OriginalAmoebaProfileBodyEvidence &body = bodyIt->second;
    std::string currentMapperBody;
    std::string bodyError;
    if (!computeOriginalAmoebaNormalizedMapperBody(
            task.op.operator->(), currentMapperBody, bodyError) ||
        currentMapperBody != body.normalizedMapperBody) {
      error = bodyError.empty()
                  ? "exported normalized mapper body does not exactly match "
                    "the current kernel after original TaskProfiler lowering "
                    "for task=" +
                        task.name
                  : bodyError;
      return false;
    }
    auto recordTask = record->getString("task");
    auto taskSignature = record->getString("task_signature");
    auto counterSignature = record->getString("counter_signature");
    auto kernelSignature = record->getString("kernel_binding_signature");
    auto mapperBody = record->getString("normalized_mapper_body");
    auto staticTrip = record->getInteger("static_trip_count");
    auto shape = record->getString("selected_cgra_shape");
    auto cgraCount = record->getInteger("selected_cgra_count");
    auto mapperRows = record->getInteger("mapper_tile_rows");
    auto mapperCols = record->getInteger("mapper_tile_cols");
    auto compiledII = record->getInteger("compiled_ii");
    auto steps = record->getInteger("steps");
    auto sampleTrip = record->getInteger("sample_trip_count");
    auto materialized = record->getInteger("materialized_operation_count");
    auto estimatedLatency = record->getInteger("estimated_latency");
    auto mapperSucceeded = record->getBoolean("mapper_succeeded");
    auto sourceDomainStatus =
        record->getString("source_iteration_domain_status");
    auto sourceDomainComplete =
        record->getBoolean("source_iteration_domain_complete");
    auto sourceWorkCount =
        record->getInteger("source_iteration_work_count");
    if (!recordTask || *recordTask != task.name || !taskSignature ||
        *taskSignature != body.taskSignature || !counterSignature ||
        *counterSignature != body.counterSignature || !kernelSignature ||
        *kernelSignature != body.kernelBindingSignature || !mapperBody ||
        *mapperBody != body.normalizedMapperBody || !staticTrip ||
        *staticTrip != body.staticTripCount || *staticTrip != task.tripCount ||
        !shape || *shape != trace.selectedShape || !cgraCount ||
        *cgraCount != trace.selectedCount || !mapperRows || !mapperCols ||
        !compiledII || *compiledII != trace.compiledII || !steps ||
        *steps != trace.steps || !sampleTrip ||
        *sampleTrip != trace.sampleTripCount || *sampleTrip != task.tripCount ||
        !materialized || *materialized != trace.materializedOperationCount ||
        !estimatedLatency || *estimatedLatency != trace.fullMapperDuration ||
        !mapperSucceeded || !*mapperSucceeded || !sourceDomainStatus ||
        *sourceDomainStatus != task.sourceIterationDomainStatus ||
        !sourceDomainComplete || !*sourceDomainComplete ||
        !sourceWorkCount ||
        *sourceWorkCount != task.sourceIterationWorkCount) {
      error = "cost catalog selected profile/body evidence disagrees with the "
              "body export or current Taskflow trace for task=" +
              task.name;
      return false;
    }
    if (task.tripCount < 1 || perCgraRows <= 0 || perCgraCols <= 0 ||
        trace.placement.selectedRows >
            std::numeric_limits<int64_t>::max() / perCgraRows ||
        trace.placement.selectedCols >
            std::numeric_limits<int64_t>::max() / perCgraCols ||
        *mapperRows != trace.placement.selectedRows * perCgraRows ||
        *mapperCols != trace.placement.selectedCols * perCgraCols) {
      error =
          "cost catalog body binding has invalid selected mapper shape for " +
          task.name;
      return false;
    }
  }
  return true;
}

static bool readTaskTrace(TaskflowTaskOp task, unsigned taskIndex,
                          unsigned dispatchIndex, OriginalTraceTask &trace,
                          std::string &error) {
  auto fail = [&](StringRef message) {
    error = "task " + task.getTaskName().str() + ": " + message.str();
    return false;
  };

  auto selectedShapeAttr =
      task->getAttrOfType<StringAttr>("composed_cgra_shape");
  auto selectedCountAttr =
      task->getAttrOfType<IntegerAttr>("composed_cgra_count");
  auto topActive = task->getAttrOfType<IntegerAttr>("active_replicas");
  auto info = task->getAttrOfType<DictionaryAttr>(kOriginalScheduleInfoAttr);
  auto profile = task->getAttrOfType<DictionaryAttr>("profile_info");
  if (!selectedShapeAttr || !selectedCountAttr || !topActive || !info ||
      !profile)
    return fail("original selected-shape, profile, or scheduler trace attrs "
                "are missing");

  int64_t selectedRows = 0;
  int64_t selectedCols = 0;
  if (!parseShape(selectedShapeAttr.getValue(), selectedRows, selectedCols) ||
      selectedCountAttr.getInt() != selectedRows * selectedCols)
    return fail("selected composed CGRA shape/count is invalid");
  auto infoSelectedShape = getString(info, "composed_cgra_shape");
  auto infoSelectedCount = getInteger(info, "composed_cgra_count");
  auto activeReplicas = getInteger(info, "active_replicas");
  auto infoTaskName = getString(info, "task_name");
  auto infoDispatch = getInteger(info, "dispatch_index");
  auto rawDuration = getInteger(info, "duration");
  auto durationUnit = getString(info, "duration_unit");
  auto schedulerStart = getInteger(info, "scheduler_start_time");
  auto schedulerEnd = getInteger(info, "scheduler_end_time");
  auto placements = info.getAs<ArrayAttr>("placements");
  auto replicaShapes = info.getAs<ArrayAttr>("replica_shapes");
  if (!infoSelectedShape || !infoSelectedCount || !activeReplicas ||
      !infoTaskName || !infoDispatch || !rawDuration || !durationUnit ||
      !schedulerStart || !schedulerEnd || !placements || !replicaShapes)
    return fail("original per-task scheduler trace is incomplete");
  if (*infoSelectedShape != selectedShapeAttr.getValue() ||
      *infoSelectedCount != selectedCountAttr.getInt() ||
      *infoTaskName != task.getTaskName() || *infoDispatch != dispatchIndex ||
      *activeReplicas != topActive.getInt())
    return fail("task and function-level original trace identities disagree");
  if (*activeReplicas < 1 || *activeReplicas > 4 ||
      replicaShapes.size() != static_cast<size_t>(*activeReplicas))
    return fail("active replica count and replica_shapes inventory disagree");
  if (*durationUnit != "profiled-cycles" || *rawDuration < 1 ||
      *schedulerStart < 0 || *schedulerEnd <= *schedulerStart)
    return fail(
        "profiled duration unit or original scheduler times are invalid");
  if (placements.empty())
    return fail("actual placement cell inventory is empty");

  auto compiledII = getInteger(profile, "compiled_ii");
  auto profileDuration = getInteger(profile, "duration");
  auto materializedOperations =
      getInteger(profile, "materialized_operation_count");
  auto sampleTripCount = getInteger(profile, "sample_trip_count");
  auto steps = getInteger(profile, "steps");
  if (!compiledII || !profileDuration || !materializedOperations ||
      !sampleTripCount || !steps || *compiledII < 1 || *profileDuration < 1 ||
      *materializedOperations < 1 || *sampleTripCount < 1 || *steps < 1 ||
      *profileDuration != *rawDuration)
    return fail("profile_info must contain positive compiled_ii, duration, "
                "materialized_operation_count, sample_trip_count, and steps");
  // ResourceAssignmentState's no-profile fallback is exactly the all-one
  // tuple. Reject it; original TaskProfiler installs the non-default profile
  // tuple only after a successful map. The captured IR predates an explicit
  // mapper_succeeded boolean, so the JSON reports this as source-inferred
  // evidence rather than an explicit flag.
  if (*compiledII == 1 && *profileDuration == 1 &&
      *materializedOperations == 1 && *sampleTripCount == 1 && *steps == 1)
    return fail("profile_info contains only the no-profile fallback tuple");

  int64_t formulaDuration = 0;
  int64_t expectedF45Duration = 0;
  if (!checkedProfileDuration(*compiledII, *sampleTripCount, *steps,
                              formulaDuration) ||
      !checkedF45SchedulerDuration(formulaDuration, *activeReplicas,
                                   expectedF45Duration) ||
      expectedF45Duration != *profileDuration)
    return fail("F45 profile_info duration is inconsistent with the full "
                "mapper formula and original active-replica ceil divisor");

  std::vector<OriginalReplicaTrace> replicas(*activeReplicas);
  for (Attribute attribute : replicaShapes) {
    auto shapeValue = dyn_cast<DictionaryAttr>(attribute);
    if (!shapeValue)
      return fail("replica_shapes contains a non-dictionary entry");
    auto replicaId = getInteger(shapeValue, "replica_id");
    auto replicaCount = getInteger(shapeValue, "cgra_count");
    auto replicaRow = getInteger(shapeValue, "row");
    auto replicaCol = getInteger(shapeValue, "col");
    auto replicaRows = getInteger(shapeValue, "placement_rows");
    auto replicaCols = getInteger(shapeValue, "placement_cols");
    auto actualShape = getString(shapeValue, "shape");
    int64_t actualShapeRows = 0, actualShapeCols = 0;
    if (!replicaId || *replicaId < 0 || *replicaId >= *activeReplicas ||
        !replicaCount || *replicaCount != *infoSelectedCount || !replicaRow ||
        *replicaRow < 0 || !replicaCol || *replicaCol < 0 || !replicaRows ||
        *replicaRows < 1 || *replicaRows > 4 || !replicaCols ||
        *replicaCols < 1 || *replicaCols > 4 || !actualShape ||
        !parseShape(*actualShape, actualShapeRows, actualShapeCols) ||
        actualShapeRows != *replicaRows || actualShapeCols != *replicaCols ||
        *replicaRows * *replicaCols != *replicaCount ||
        *replicaRow + *replicaRows > 4 || *replicaCol + *replicaCols > 4 ||
        ((*replicaRows != selectedRows || *replicaCols != selectedCols) &&
         (*replicaRows != selectedCols || *replicaCols != selectedRows)) ||
        replicas[*replicaId].cgraCount != 0)
      return fail("captured replica shape fields are invalid or duplicated");
    OriginalReplicaTrace &replica = replicas[*replicaId];
    replica.replicaId = static_cast<unsigned>(*replicaId);
    replica.shape = actualShape->str();
    replica.cgraCount = *replicaCount;
    replica.row = static_cast<int>(*replicaRow);
    replica.col = static_cast<int>(*replicaCol);
    replica.rows = static_cast<int>(*replicaRows);
    replica.cols = static_cast<int>(*replicaCols);
  }

  std::map<std::pair<int64_t, int64_t>, int64_t> contexts;
  int64_t commonStart = -1;
  int64_t commonEnd = -1;
  int64_t commonDuration = -1;
  for (Attribute attribute : placements) {
    auto cell = dyn_cast<DictionaryAttr>(attribute);
    if (!cell)
      return fail("placements contains a non-dictionary entry");
    auto row = getInteger(cell, "row");
    auto col = getInteger(cell, "col");
    auto context = getInteger(cell, "context_id");
    auto placedReplica = getInteger(cell, "replica_id");
    auto start = getInteger(cell, "scheduler_start_time");
    auto end = getInteger(cell, "scheduler_end_time");
    auto duration = getInteger(cell, "scheduler_duration");
    if (!row || !col || !context || !placedReplica || !start || !end ||
        !duration || *row < 0 || *row >= 4 || *col < 0 || *col >= 4 ||
        *context < 0 || *context >= 6 || *placedReplica < 0 ||
        *placedReplica >= *activeReplicas || *start < 0 || *end <= *start ||
        *duration != *end - *start ||
        !contexts.emplace(std::make_pair(*row, *col), *context).second)
      return fail("placement cell, context, replica, or original times are "
                  "invalid or duplicated");
    if (commonStart < 0) {
      commonStart = *start;
      commonEnd = *end;
      commonDuration = *duration;
    } else if (commonStart != *start || commonEnd != *end ||
               commonDuration != *duration) {
      return fail("original replicas do not share one task-level interval");
    }
    OriginalReplicaTrace &replica = replicas[*placedReplica];
    if (replica.cgraCount == 0 || *row < replica.row ||
        *row >= replica.row + replica.rows || *col < replica.col ||
        *col >= replica.col + replica.cols)
      return fail("placement cell lies outside its recorded replica rectangle");
    replica.occupiedCells.push_back(
        {static_cast<int>(*row), static_cast<int>(*col),
         static_cast<unsigned>(*placedReplica)});
    replica.contextIds.push_back(*context);
    trace.originalOccupiedCells.push_back(
        {static_cast<int>(*row), static_cast<int>(*col),
         static_cast<unsigned>(*placedReplica)});
    trace.originalContextIds.push_back(*context);
  }
  for (OriginalReplicaTrace &replica : replicas) {
    if (replica.cgraCount == 0 ||
        replica.occupiedCells.size() != static_cast<size_t>(replica.cgraCount) ||
        replica.cgraCount != replica.rows * replica.cols)
      return fail("replica placement omits exact shape cells");
    std::set<std::pair<int, int>> cells;
    for (const OriginalAmoebaOccupiedCell &cell : replica.occupiedCells)
      cells.emplace(cell.row, cell.col);
    if (cells.size() != static_cast<size_t>(replica.cgraCount))
      return fail("replica placement duplicates a cell");
    for (int row = replica.row; row < replica.row + replica.rows; ++row)
      for (int col = replica.col; col < replica.col + replica.cols; ++col)
        if (!cells.count({row, col}))
          return fail("replica placement rectangle has a cell gap");
  }
  if (commonStart != *schedulerStart || commonEnd != *schedulerEnd ||
      commonDuration != commonEnd - commonStart)
    return fail("replica placement intervals disagree with original task times");

  auto orchestration =
      task->getAttrOfType<DictionaryAttr>("task_orchestration_info");
  auto orchestrationCells =
      orchestration ? orchestration.getAs<ArrayAttr>("cgra_positions")
                    : ArrayAttr();
  if (!orchestrationCells || orchestrationCells.size() != contexts.size())
    return fail("task_orchestration_info does not cover every placed cell");
  std::set<std::pair<int64_t, int64_t>> checkedOrchestrationCells;
  for (Attribute attribute : orchestrationCells) {
    auto cell = dyn_cast<DictionaryAttr>(attribute);
    auto row = cell ? getInteger(cell, "row") : std::nullopt;
    auto col = cell ? getInteger(cell, "col") : std::nullopt;
    auto context = cell ? getInteger(cell, "context_id") : std::nullopt;
    if (!row || !col || !context || *context < 0 ||
        !checkedOrchestrationCells.emplace(*row, *col).second ||
        !contexts.count({*row, *col}) || contexts[{*row, *col}] != *context)
      return fail("task_orchestration_info cell/context data disagrees with "
                  "the original scheduler trace");
  }

  trace.replicas = std::move(replicas);
  const OriginalReplicaTrace &firstReplica = trace.replicas.front();
  trace.placement.task = taskIndex;
  trace.placement.row = firstReplica.row;
  trace.placement.col = firstReplica.col;
  trace.placement.rows = firstReplica.rows;
  trace.placement.cols = firstReplica.cols;
  trace.placement.selectedRows = static_cast<int>(selectedRows);
  trace.placement.selectedCols = static_cast<int>(selectedCols);
  trace.placement.activeReplicas = static_cast<unsigned>(*activeReplicas);
  trace.placement.occupiedCells = firstReplica.occupiedCells;
  trace.contextIds = firstReplica.contextIds;
  trace.selectedShape = selectedShapeAttr.getValue().str();
  trace.actualShape = firstReplica.shape;
  trace.selectedCount = *infoSelectedCount;
  trace.profileDuration = *profileDuration;
  trace.fullMapperDuration = formulaDuration;
  trace.compiledII = *compiledII;
  trace.materializedOperationCount = *materializedOperations;
  trace.sampleTripCount = *sampleTripCount;
  trace.steps = *steps;
  trace.schedulerStart = commonStart;
  trace.schedulerEnd = commonEnd;
  trace.schedulerDuration = commonDuration;
  return true;
}

struct OriginalScheduleInventory {
  std::vector<OriginalTraceTask> traces;
  std::vector<unsigned> dispatchOrder;
  std::vector<std::string> dispatchNames;
  int64_t pipelineInterval = 0;
  std::string internalTimeUnit;
  int64_t internalTimeScale = 0;
};

static bool readOriginalScheduleInventory(
    func::FuncOp function, ArrayRef<TaskMetadata> tasks,
    OriginalScheduleInventory &inventory, std::string &error) {
  auto fail = [&](StringRef reason) {
    error = "original throughput-guided schedule inventory: " + reason.str();
    return false;
  };
  auto strategy = dyn_cast_or_null<DictionaryAttr>(
      function->getAttr(kOriginalSummaryAttr));
  auto strategyName = getString(strategy, "strategy");
  auto pipelineInterval = getInteger(strategy, "pipeline_interval");
  auto internalTimeUnit =
      function->getAttrOfType<StringAttr>(kOriginalTimeUnitAttr);
  auto internalTimeScale =
      function->getAttrOfType<IntegerAttr>(kOriginalTimeScaleAttr);
  auto functionDispatch =
      function->getAttrOfType<ArrayAttr>(kOriginalDispatchAttr);
  auto functionSchedule =
      function->getAttrOfType<ArrayAttr>(kOriginalScheduleTableAttr);
  if (!strategyName || *strategyName != "throughput-guided" ||
      !pipelineInterval || *pipelineInterval < 1 || !internalTimeUnit ||
      internalTimeUnit.getValue() != "scaled-internal-placement-slots" ||
      !internalTimeScale || internalTimeScale.getInt() < 1 ||
      !functionDispatch || !functionSchedule)
    return fail("required dispatch, time-unit, pipeline, or schedule attrs are missing");
  if (functionDispatch.size() != tasks.size() ||
      functionSchedule.size() != tasks.size())
    return fail("parent dispatch and schedule do not cover every source task exactly once");

  llvm::StringMap<unsigned> taskIndices;
  for (unsigned index = 0; index < tasks.size(); ++index)
    if (tasks[index].name.empty() ||
        !taskIndices.try_emplace(tasks[index].name, index).second)
      return fail("parent task names are duplicated or empty");
  llvm::StringSet<> seenDispatch;
  inventory.dispatchOrder.clear();
  inventory.dispatchNames.clear();
  for (Attribute attribute : functionDispatch) {
    auto name = dyn_cast<StringAttr>(attribute);
    auto task = name ? taskIndices.find(name.getValue()) : taskIndices.end();
    if (!name || task == taskIndices.end() ||
        !seenDispatch.insert(name.getValue()).second)
      return fail("parent dispatch order is not a unique task permutation");
    inventory.dispatchOrder.push_back(task->second);
    inventory.dispatchNames.push_back(name.getValue().str());
  }

  std::map<std::string, DictionaryAttr> scheduleInfoByTask;
  for (Attribute attribute : functionSchedule) {
    auto record = dyn_cast<DictionaryAttr>(attribute);
    auto name = getString(record, "task_name");
    auto index = getInteger(record, "dispatch_index");
    auto task = name ? taskIndices.find(*name) : taskIndices.end();
    if (!record || !name || task == taskIndices.end() || !index || *index < 0 ||
        *index >= static_cast<int64_t>(tasks.size()) ||
        inventory.dispatchNames[*index] != *name ||
        !scheduleInfoByTask.emplace(name->str(), record).second)
      return fail("parent schedule records are malformed or inconsistent with dispatch");
  }
  if (scheduleInfoByTask.size() != tasks.size())
    return fail("parent function-level scheduler trace omits a source task");

  inventory.traces.assign(tasks.size(), OriginalTraceTask{});
  for (unsigned index = 0; index < tasks.size(); ++index) {
    TaskflowTaskOp task = tasks[index].op;
    auto found = scheduleInfoByTask.find(tasks[index].name);
    auto taskInfo = task->getAttrOfType<DictionaryAttr>(kOriginalScheduleInfoAttr);
    if (found == scheduleInfoByTask.end() || !taskInfo ||
        found->second != taskInfo)
      return fail("task-level and function-level parent schedule records differ");
    auto dispatchIndex = getInteger(taskInfo, "dispatch_index");
    if (!dispatchIndex || *dispatchIndex < 0 ||
        *dispatchIndex >= static_cast<int64_t>(tasks.size()) ||
        inventory.dispatchOrder[*dispatchIndex] != index)
      return fail("parent task dispatch index disagrees with original order");
    if (!readTaskTrace(task, index, static_cast<unsigned>(*dispatchIndex),
                       inventory.traces[index], error))
      return false;
    if (inventory.traces[index].sampleTripCount != tasks[index].tripCount)
      return fail("parent profile sample trip differs from compiler-inferred macro trip");
  }
  inventory.pipelineInterval = *pipelineInterval;
  inventory.internalTimeUnit = internalTimeUnit.getValue().str();
  inventory.internalTimeScale = internalTimeScale.getInt();
  return true;
}

static bool readMaterializedDecisions(
    func::FuncOp currentFunction, ArrayRef<TaskMetadata> currentTasks,
    ArrayRef<TaskMetadata> sourceTasks,
    const std::map<std::string, OriginalTraceTask> &sourceTraces,
    std::map<std::string, MaterializedParentDecision> &decisions,
    std::map<std::string, std::pair<std::string, unsigned>> &childLineage,
    std::string &error) {
  auto fail = [&](StringRef reason) {
    error = "original replica materialization: " + reason.str();
    return false;
  };
  auto rawDecisions = currentFunction->getAttrOfType<ArrayAttr>(
      kOriginalMaterializedDecisionsAttr);
  if (!rawDecisions || rawDecisions.empty())
    return fail("function has no parent decision manifest");

  std::map<std::string, const TaskMetadata *> currentByName;
  for (const TaskMetadata &task : currentTasks)
    if (!currentByName.emplace(task.name, &task).second)
      return fail("current child task names are duplicated");
  std::map<std::string, const TaskMetadata *> sourceByName;
  for (const TaskMetadata &task : sourceTasks)
    if (!sourceByName.emplace(task.name, &task).second)
      return fail("canonical parent task names are duplicated");

  for (Attribute rawDecision : rawDecisions) {
    auto decision = dyn_cast<DictionaryAttr>(rawDecision);
    auto schema = getString(decision, "schema");
    auto parentName = getString(decision, "parent_task");
    auto replicaCount = getInteger(decision, "original_replica_count");
    auto outputAxis = getInteger(decision, "selected_output_axis");
    auto counterOrdinal = getInteger(decision, "selected_counter_ordinal");
    auto originalDomain =
        decision ? decision.getAs<ArrayAttr>("original_domain") : ArrayAttr();
    auto schedulerRecord = decision
                               ? decision.getAs<DictionaryAttr>(
                                     "scheduler_record")
                               : DictionaryAttr();
    auto replicas =
        decision ? decision.getAs<ArrayAttr>("replicas") : ArrayAttr();
    if (!decision || decision.size() != 8 || !schema ||
        *schema != "amoeba-original-fixed-decision-realization-v1" ||
        !parentName || parentName->empty() || !replicaCount ||
        *replicaCount < 2 || *replicaCount > 4 || !outputAxis ||
        *outputAxis < 0 || !counterOrdinal || *counterOrdinal < 0 ||
        !originalDomain || !schedulerRecord || !replicas ||
        replicas.size() != static_cast<size_t>(*replicaCount))
      return fail("function parent decision record has an invalid exact schema");
    auto parent = sourceByName.find(parentName->str());
    auto trace = sourceTraces.find(parentName->str());
    auto parentActive = parent == sourceByName.end()
                            ? IntegerAttr()
                            : parent->second->op->getAttrOfType<IntegerAttr>(
                                  "active_replicas");
    auto parentSchedule = parent == sourceByName.end()
                              ? DictionaryAttr()
                              : parent->second->op->getAttrOfType<DictionaryAttr>(
                                    kOriginalScheduleInfoAttr);
    MaterializedParentDecision parsed;
    parsed.parentTask = parentName->str();
    parsed.replicaCount = static_cast<unsigned>(*replicaCount);
    parsed.selectedCounterOrdinal = *counterOrdinal;
    parsed.selectedOutputAxis = *outputAxis;
    parsed.originalDomain = originalDomain;
    if (parent == sourceByName.end() || trace == sourceTraces.end() ||
        !parentActive || parentActive.getInt() != *replicaCount ||
        trace->second.replicas.size() != static_cast<size_t>(*replicaCount) ||
        !parentSchedule || parentSchedule != schedulerRecord ||
        !decodeSourceDomainBounds(originalDomain, parsed.decodedOriginalDomain,
                                  "parent original_domain", error))
      return fail(error.empty()
                      ? "parent decision differs from the canonical scheduler"
                      : error);
    if (static_cast<size_t>(*counterOrdinal) >=
        parsed.decodedOriginalDomain.size())
      return fail("selected source counter ordinal is outside the original rank");
    parsed.sourceMultiplicity =
        parent->second->sourceIterationMultiplicity;
    parsed.sourceWorkCount = parent->second->sourceIterationWorkCount;
    parsed.expandedInternalExtents.assign(
        parent->second->expandedInternalExtents.begin(),
        parent->second->expandedInternalExtents.end());
    int64_t parentDomainVolume = 0;
    int64_t extentMultiplicity = 1;
    for (int64_t extent : parsed.expandedInternalExtents) {
      if (extent < 1 ||
          extentMultiplicity >
              std::numeric_limits<int64_t>::max() / extent)
        return fail("parent expanded internal extent product is invalid");
      extentMultiplicity *= extent;
    }
    if (!sourceDomainVolume(parsed.decodedOriginalDomain, parentDomainVolume,
                            "parent original_domain", error) ||
        parentDomainVolume != parent->second->tripCount ||
        parent->second->sourceIterationDomainStatus != "certified-complete" ||
        !parent->second->sourceIterationDomainComplete ||
        parsed.sourceMultiplicity < 1 ||
        extentMultiplicity != parsed.sourceMultiplicity ||
        parentDomainVolume >
            std::numeric_limits<int64_t>::max() / parsed.sourceMultiplicity ||
        parentDomainVolume * parsed.sourceMultiplicity !=
            parsed.sourceWorkCount)
      return fail(error.empty()
                      ? "parent source bounds, macro trip count, and work count disagree"
                      : error);

    for (Attribute rawReplica : replicas) {
      auto replica = dyn_cast<DictionaryAttr>(rawReplica);
      auto replicaId = getInteger(replica, "replica_id");
      auto childName = getString(replica, "task_name");
      auto shape = replica ? replica.getAs<DictionaryAttr>("replica_shape")
                           : DictionaryAttr();
      auto placements = replica ? replica.getAs<ArrayAttr>("placements")
                                : ArrayAttr();
      auto partitionBounds =
          replica ? replica.getAs<ArrayAttr>("partition_bounds") : ArrayAttr();
      auto sourceWork = getInteger(replica, "source_work_count");
      if (!replica || replica.size() != 6 || !replicaId || *replicaId < 0 ||
          *replicaId >= *replicaCount || !childName || childName->empty() ||
          !shape || !placements || placements.empty() || !partitionBounds ||
          !sourceWork || *sourceWork < 1 ||
          parsed.replicas.count(static_cast<unsigned>(*replicaId)))
        return fail("parent replica record is malformed or duplicated");
      auto current = currentByName.find(childName->str());
      auto shapeName = getString(shape, "shape");
      auto shapeCount = getInteger(shape, "cgra_count");
      auto shapeRow = getInteger(shape, "row");
      auto shapeCol = getInteger(shape, "col");
      auto shapeRows = getInteger(shape, "placement_rows");
      auto shapeCols = getInteger(shape, "placement_cols");
      int64_t parsedRows = 0, parsedCols = 0;
      std::vector<SourceDomainBound> decodedBounds;
      int64_t boundsVolume = 0;
      if (current == currentByName.end() || !shapeName || !shapeCount ||
          !shapeRow || !shapeCol || !shapeRows || !shapeCols ||
          !parseShape(*shapeName, parsedRows, parsedCols) ||
          parsedRows != *shapeRows || parsedCols != *shapeCols ||
          *shapeCount != *shapeRows * *shapeCols || *shapeRow < 0 ||
          *shapeCol < 0 || *shapeRow + *shapeRows > 4 ||
          *shapeCol + *shapeCols > 4 ||
          !decodeSourceDomainBounds(partitionBounds, decodedBounds,
                                    "replica partition_bounds", error) ||
          decodedBounds.size() != parsed.decodedOriginalDomain.size() ||
          !sourceDomainVolume(decodedBounds, boundsVolume,
                              "replica partition_bounds", error))
        return fail(error.empty()
                        ? "replica placement or source bounds are malformed"
                        : error);

      const TaskMetadata &childTask = *current->second;
      int64_t expectedChildWork = 0;
      if (childTask.tripCount != boundsVolume ||
          childTask.sourceIterationDomainStatus != "certified-complete" ||
          !childTask.sourceIterationDomainComplete ||
          childTask.sourceIterationMultiplicity != parsed.sourceMultiplicity ||
          ArrayRef<int64_t>(childTask.expandedInternalExtents) !=
              ArrayRef<int64_t>(parsed.expandedInternalExtents) ||
          childTask.sourceIterationWorkCount < 1 ||
          childTask.tripCount >
              std::numeric_limits<int64_t>::max() /
                  childTask.sourceIterationMultiplicity)
        return fail("child partition volume, actual trip, multiplicity, and source work disagree");
      expectedChildWork = childTask.tripCount *
                          childTask.sourceIterationMultiplicity;
      if (
          expectedChildWork != childTask.sourceIterationWorkCount ||
          *sourceWork != expectedChildWork)
        return fail("child partition volume, actual trip, multiplicity, and source work disagree");
      if (!llvm::all_of(decodedBounds,
                        [&](const SourceDomainBound &bound) {
                          const SourceDomainBound &whole =
                              parsed.decodedOriginalDomain[bound.ordinal];
                          return bound.lower >= whole.lower &&
                                 bound.upper <= whole.upper &&
                                 bound.step == whole.step &&
                                 (bound.lower - whole.lower) % whole.step == 0 &&
                                 (bound.upper - whole.lower) % whole.step == 0 &&
                                 (bound.ordinal == *counterOrdinal ||
                                  (bound.lower == whole.lower &&
                                   bound.upper == whole.upper));
                        }))
        return fail("child partition escapes parent bounds or changes a nonselected axis");

      auto realization = childTask.op->getAttrOfType<DictionaryAttr>(
          kOriginalPartitionRealizationAttr);
      auto childDecision = childTask.op->getAttrOfType<DictionaryAttr>(
          kOriginalReplicaDecisionAttr);
      auto childPlacements = childTask.op->getAttrOfType<ArrayAttr>(
          kOriginalReplicaPlacementsAttr);
      auto realizationSchema = getString(realization, "schema");
      auto realizationStatus = getString(realization, "status");
      auto realizationOrigin = getString(realization, "origin");
      auto noF45Bounds = realization
                             ? realization.getAs<BoolAttr>(
                                   "f45_axis_and_bounds_present")
                             : BoolAttr();
      auto realizationAxis = getInteger(realization, "selected_counter_ordinal");
      auto realizationOutput = getInteger(realization, "selected_output_axis");
      auto realizationCount = getInteger(realization, "original_replica_count");
      auto realizationId = getInteger(realization, "replica_id");
      auto realizationDomain =
          realization ? realization.getAs<ArrayAttr>("original_domain")
                      : ArrayAttr();
      auto realizationBounds =
          realization ? realization.getAs<ArrayAttr>("partition_bounds")
                      : ArrayAttr();
      auto realizationWork = getInteger(realization, "source_work_count");
      auto realizationMultiplicity =
          getInteger(realization, "internal_multiplicity");
      auto decisionScheduler = childDecision
                                   ? childDecision.getAs<DictionaryAttr>(
                                         "scheduler_record")
                                   : DictionaryAttr();
      auto decisionShape = childDecision
                               ? childDecision.getAs<DictionaryAttr>(
                                     "replica_shape")
                               : DictionaryAttr();
      auto decisionPlacements = childDecision
                                    ? childDecision.getAs<ArrayAttr>(
                                          "placements")
                                    : ArrayAttr();
      auto decisionBinding =
          getString(realization, "decision_binding");
      if (!realization || realization.size() != 13 || !realizationSchema ||
          *realizationSchema != "amoeba-source-certified-replica-realization-v1" ||
          !realizationStatus ||
          *realizationStatus != "verified-source-certified-realization-v1" ||
          !realizationOrigin ||
          *realizationOrigin !=
              "orbit-native-materializer-selected-counter-axis-after-f45" ||
          !noF45Bounds || noF45Bounds.getValue() || !realizationAxis ||
          *realizationAxis != *counterOrdinal || !realizationOutput ||
          *realizationOutput != *outputAxis || !realizationCount ||
          *realizationCount != *replicaCount || !realizationId ||
          *realizationId != *replicaId || realizationDomain != originalDomain ||
          realizationBounds != partitionBounds || !realizationWork ||
          *realizationWork != *sourceWork || !realizationMultiplicity ||
          *realizationMultiplicity != parsed.sourceMultiplicity ||
          !decisionBinding || *decisionBinding !=
                                  "amoeba.task_scheduler_schedule_info" ||
          !childDecision || childDecision.size() != 3 ||
          decisionScheduler != schedulerRecord || decisionShape != shape ||
          decisionPlacements != placements || childPlacements != placements)
        return fail("child realization attrs differ from its complete parent decision manifest");

      const OriginalReplicaTrace &sourceReplica =
          trace->second.replicas[*replicaId];
      if (sourceReplica.shape != *shapeName ||
          sourceReplica.cgraCount != *shapeCount ||
          sourceReplica.row != *shapeRow || sourceReplica.col != *shapeCol ||
          sourceReplica.rows != *shapeRows || sourceReplica.cols != *shapeCols ||
          sourceReplica.occupiedCells.size() != placements.size() ||
          sourceReplica.contextIds.size() != placements.size())
        return fail("child shape, placement, or context inventory differs from original replica id");
      for (auto [index, rawPlacement] : llvm::enumerate(placements)) {
        auto placement = dyn_cast<DictionaryAttr>(rawPlacement);
        auto row = getInteger(placement, "row");
        auto col = getInteger(placement, "col");
        auto replica = getInteger(placement, "replica_id");
        auto context = getInteger(placement, "context_id");
        if (!placement || !row || !col || !replica ||
            *replica != *replicaId || !context || *context < 0 ||
            *context >= 6 ||
            *row != sourceReplica.occupiedCells[index].row ||
            *col != sourceReplica.occupiedCells[index].col ||
            *context != sourceReplica.contextIds[index])
          return fail("child placement records do not preserve original cells, contexts, and replica order");
      }

      MaterializedReplicaDecision parsedReplica;
      parsedReplica.replicaId = static_cast<unsigned>(*replicaId);
      parsedReplica.taskName = childName->str();
      parsedReplica.shape = shape;
      parsedReplica.placements = placements;
      parsedReplica.partitionBounds = partitionBounds;
      parsedReplica.sourceWorkCount = *sourceWork;
      parsedReplica.tripCount = childTask.tripCount;
      parsedReplica.sourceMultiplicity =
          childTask.sourceIterationMultiplicity;
      parsedReplica.expandedInternalExtents.assign(
          childTask.expandedInternalExtents.begin(),
          childTask.expandedInternalExtents.end());
      parsedReplica.decodedPartitionBounds = std::move(decodedBounds);
      if (!childLineage.emplace(parsedReplica.taskName,
                                std::make_pair(parsed.parentTask,
                                               parsedReplica.replicaId))
               .second ||
          !parsed.replicas.emplace(parsedReplica.replicaId,
                                   std::move(parsedReplica))
               .second)
        return fail("child task or replica identity is duplicated");
    }

    int64_t partitionAxisLower = -1;
    int64_t partitionAxisUpper = -1;
    int64_t childWorkTotal = 0;
    for (unsigned replicaId = 0; replicaId < parsed.replicaCount; ++replicaId) {
      auto found = parsed.replicas.find(replicaId);
      if (found == parsed.replicas.end())
        return fail("parent partition omits a zero-based replica id");
      const SourceDomainBound &bound =
          found->second.decodedPartitionBounds[parsed.selectedCounterOrdinal];
      if (replicaId == 0)
        partitionAxisLower = bound.lower;
      if (bound.upper <= bound.lower ||
          (replicaId > 0 && bound.lower != partitionAxisUpper))
        return fail("ordered child intervals do not form a contiguous source partition");
      partitionAxisUpper = bound.upper;
      if (childWorkTotal >
          std::numeric_limits<int64_t>::max() - found->second.sourceWorkCount)
        return fail("child source-work total overflows signed 64-bit");
      childWorkTotal += found->second.sourceWorkCount;
    }
    const SourceDomainBound &wholeAxis =
        parsed.decodedOriginalDomain[parsed.selectedCounterOrdinal];
    const int64_t axisExtent = wholeAxis.upper - wholeAxis.lower;
    if (axisExtent % parsed.replicaCount != 0)
      return fail("selected source counter extent is not divisible by replica count");
    const int64_t expectedWidth = axisExtent / parsed.replicaCount;
    if (partitionAxisLower != wholeAxis.lower ||
        partitionAxisUpper != wholeAxis.upper ||
        llvm::any_of(parsed.replicas, [&](const auto &entry) {
          const SourceDomainBound &bound =
              entry.second.decodedPartitionBounds[parsed.selectedCounterOrdinal];
          return bound.upper - bound.lower != expectedWidth ||
                 bound.lower != wholeAxis.lower +
                                    static_cast<int64_t>(entry.first) *
                                        expectedWidth;
        }) ||
        !sourceDomainVolume(parsed.decodedOriginalDomain, parentDomainVolume,
                            "parent original_domain", error) ||
        childWorkTotal != parsed.sourceWorkCount ||
        parentDomainVolume * parsed.sourceMultiplicity !=
            parsed.sourceWorkCount)
      return fail("child intervals do not exactly cover parent macro/source work");
    if (!decisions.emplace(parsed.parentTask, std::move(parsed)).second)
      return fail("parent task decision is duplicated");
  }

  std::set<std::string> expectedCurrentNames;
  for (const TaskMetadata &sourceTask : sourceTasks) {
    auto decision = decisions.find(sourceTask.name);
    if (decision == decisions.end()) {
      auto activeReplicas = sourceTask.op->getAttrOfType<IntegerAttr>(
          "active_replicas");
      if (!activeReplicas || activeReplicas.getInt() != 1)
        return fail("scheduled multi-replica parent has no materialized child partition: " +
                    sourceTask.name);
      auto current = currentByName.find(sourceTask.name);
      if (current == currentByName.end())
        return fail("unreplicated parent task disappeared from current module");
      expectedCurrentNames.insert(sourceTask.name);
    } else {
      if (sourceTask.sourceIterationWorkCount != decision->second.sourceWorkCount ||
          sourceTask.sourceIterationMultiplicity !=
              decision->second.sourceMultiplicity)
        return fail("materialized decision source facts differ from canonical parent task");
      for (const auto &replica : decision->second.replicas)
        expectedCurrentNames.insert(replica.second.taskName);
    }
  }
  std::set<std::string> actualCurrentNames;
  for (const TaskMetadata &task : currentTasks)
    actualCurrentNames.insert(task.name);
  if (expectedCurrentNames != actualCurrentNames ||
      childLineage.size() + (sourceTasks.size() - decisions.size()) !=
          currentTasks.size())
    return fail("parent and expanded child task inventories are not an exact permutation");
  return true;
}

static bool makeExpandedReplicaTrace(
    unsigned taskIndex, const OriginalTraceTask &parentTrace,
    const MaterializedReplicaDecision &replica,
    const OriginalReplicaProfileEvidence &profile,
    OriginalTraceTask &trace, std::string &error) {
  auto shape = getString(replica.shape, "shape");
  auto count = getInteger(replica.shape, "cgra_count");
  auto row = getInteger(replica.shape, "row");
  auto col = getInteger(replica.shape, "col");
  auto rows = getInteger(replica.shape, "placement_rows");
  auto cols = getInteger(replica.shape, "placement_cols");
  int64_t parsedRows = 0, parsedCols = 0;
  if (!shape || !count || !row || !col || !rows || !cols ||
      !parseShape(*shape, parsedRows, parsedCols) ||
      parsedRows != *rows || parsedCols != *cols || *count != *rows * *cols ||
      profile.selectedShape != *shape || profile.cgraCount != *count ||
      profile.sampleTripCount != replica.tripCount ||
      parentTrace.replicas.size() <= replica.replicaId) {
    error = "expanded child shape/profile does not match its original replica";
    return false;
  }
  const OriginalReplicaTrace &sourceReplica =
      parentTrace.replicas[replica.replicaId];
  if (sourceReplica.shape != *shape || sourceReplica.cgraCount != *count ||
      sourceReplica.row != *row || sourceReplica.col != *col ||
      sourceReplica.rows != *rows || sourceReplica.cols != *cols) {
    error = "expanded child placement differs from its original replica ID";
    return false;
  }
  trace = parentTrace;
  trace.placement = OriginalAmoebaFixedPlacement{};
  trace.placement.task = taskIndex;
  trace.placement.row = static_cast<int>(*row);
  trace.placement.col = static_cast<int>(*col);
  trace.placement.rows = static_cast<int>(*rows);
  trace.placement.cols = static_cast<int>(*cols);
  trace.placement.selectedRows = static_cast<int>(*rows);
  trace.placement.selectedCols = static_cast<int>(*cols);
  trace.placement.activeReplicas = 1;
  trace.placement.occupiedCells = sourceReplica.occupiedCells;
  for (OriginalAmoebaOccupiedCell &cell : trace.placement.occupiedCells)
    cell.replicaId = 0;
  trace.contextIds = sourceReplica.contextIds;
  trace.replicas.assign(1, sourceReplica);
  trace.selectedShape = shape->str();
  trace.actualShape = shape->str();
  trace.selectedCount = *count;
  trace.profileDuration = profile.estimatedLatency;
  trace.fullMapperDuration = profile.estimatedLatency;
  trace.compiledII = profile.compiledII;
  trace.materializedOperationCount = profile.materializedOperationCount;
  trace.sampleTripCount = profile.sampleTripCount;
  trace.steps = profile.steps;
  trace.originalOccupiedCells = sourceReplica.occupiedCells;
  trace.originalContextIds = sourceReplica.contextIds;
  trace.sourceReplicaId = replica.replicaId;
  return true;
}

static llvm::json::Array integerArray(ArrayRef<int64_t> values) {
  llvm::json::Array result;
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

struct RetimingOriginalAmoebaFixedDecisionsPass
    : PassWrapper<RetimingOriginalAmoebaFixedDecisionsPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RetimingOriginalAmoebaFixedDecisionsPass)

  RetimingOriginalAmoebaFixedDecisionsPass() = default;
  RetimingOriginalAmoebaFixedDecisionsPass(
      const RetimingOriginalAmoebaFixedDecisionsPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "retime-original-amoeba-fixed-decisions";
  }
  StringRef getDescription() const override {
    return "Retiming captured original AMOEBA decisions with mapped costs and "
           "explicit communication";
  }

  Option<std::string> functionName{
      *this, "function", llvm::cl::desc("Original Taskflow function."),
      llvm::cl::init("")};
  Option<std::string> parentCostFile{
      *this, "parent-cost-file",
      llvm::cl::desc("Bound ORBIT analytical task cost catalog."),
      llvm::cl::init("")};
  Option<std::string> bodyExportFile{
      *this, "profile-body-export-file",
      llvm::cl::desc("Exact original TaskProfiler mapper-body export."),
      llvm::cl::init("")};
  Option<std::string> replicaProfileEvidenceFile{
      *this, "replica-profile-evidence-file",
      llvm::cl::desc("Exact profiles and body exports for materialized replicas."),
      llvm::cl::init("")};
  Option<bool> originalF45ReplicaScaling{
      *this, "original-f45-replica-scaling",
      llvm::cl::desc("Preserve original F45 replica duration estimation without "
                     "claiming actual split-child mapper profiles."),
      llvm::cl::init(false)};
  Option<std::string> outputFile{*this, "output",
                                 llvm::cl::desc("Atomic JSON result path."),
                                 llvm::cl::init("")};
  Option<int64_t> diagnosticIICeiling{
      *this, "diagnostic-ii-ceiling",
      llvm::cl::desc("Training ceiling 20 or explicit diagnostic runtime 23."),
      llvm::cl::init(20)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const std::string currentInputModuleWitness =
        neighborhoodReplaySourceText(module);
    if (diagnosticIICeiling != 20 && diagnosticIICeiling != 23) {
      module.emitError()
          << "diagnostic-ii-ceiling accepts only 20 (default) or 23";
      return signalPassFailure();
    }
    if (functionName.empty() || parentCostFile.empty() ||
        bodyExportFile.empty() || outputFile.empty()) {
      module.emitError()
          << "function, parent-cost-file, profile-body-export-file and output "
             "are required";
      return signalPassFailure();
    }

    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selected)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp function = *selected;
    auto graphId =
        function->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
    const bool assignedOriginalGraphId = !graphId || graphId.getValue().empty();
    if (assignedOriginalGraphId) {
      graphId = StringAttr::get(module.getContext(),
                                kOriginalBaselineGraphVariantId);
      function->setAttr("amoeba.graph_variant_id", graphId);
    }

    FailureOr<llvm::SmallVector<TaskMetadata>> collectedCurrent =
        collectAnalyticalTaskMetadata(function, error);
    if (failed(collectedCurrent) || collectedCurrent->empty()) {
      function.emitError() << (error.empty() ? "no static Taskflow tasks"
                                              : error);
      return signalPassFailure();
    }
    llvm::SmallVector<TaskMetadata> currentTasks = std::move(*collectedCurrent);
    const bool hasReplicaEvidence = !replicaProfileEvidenceFile.empty();
    const bool hasDecisionManifest =
        function->hasAttr(kOriginalMaterializedDecisionsAttr);
    if (originalF45ReplicaScaling &&
        (hasReplicaEvidence || hasDecisionManifest)) {
      function.emitError()
          << "original F45 replica scaling and actual materialized-child "
             "profile evidence are mutually exclusive";
      return signalPassFailure();
    }
    if (hasReplicaEvidence != hasDecisionManifest) {
      function.emitError()
          << "materialized replica decisions and replica profile evidence must "
             "be supplied together";
      return signalPassFailure();
    }

    func::FuncOp sourceFunction = function;
    OwningOpRef<ModuleOp> canonicalSourceModule;
    std::string canonicalModuleWitness;
    if (hasReplicaEvidence) {
      if (!readCanonicalModuleWitness(parentCostFile.getValue(),
                                      canonicalModuleWitness, error)) {
        function.emitError() << error;
        return signalPassFailure();
      }
      ParserConfig parserConfig(module.getContext());
      canonicalSourceModule = parseSourceString<ModuleOp>(
          canonicalModuleWitness, parserConfig, "canonical-original-source.mlir");
      if (!canonicalSourceModule) {
        function.emitError()
            << "bound canonical source module witness could not be parsed";
        return signalPassFailure();
      }
      FailureOr<func::FuncOp> canonicalSelected = selectTaskFunction(
          *canonicalSourceModule, functionName.getValue(), error);
      if (failed(canonicalSelected)) {
        function.emitError() << error;
        return signalPassFailure();
      }
      sourceFunction = *canonicalSelected;
    }

    FailureOr<llvm::SmallVector<TaskMetadata>> collectedSource =
        hasReplicaEvidence
            ? collectAnalyticalTaskMetadata(sourceFunction, error)
            : FailureOr<llvm::SmallVector<TaskMetadata>>(currentTasks);
    if (failed(collectedSource) || collectedSource->empty()) {
      sourceFunction.emitError()
          << (error.empty() ? "no canonical source Taskflow tasks" : error);
      return signalPassFailure();
    }
    llvm::SmallVector<TaskMetadata> sourceTasks = std::move(*collectedSource);
    auto sourceGraphId =
        sourceFunction->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
    if (sourceGraphId && !sourceGraphId.getValue().empty() &&
        sourceGraphId.getValue() != graphId.getValue()) {
      function.emitError()
          << "current and canonical source graph identities differ";
      return signalPassFailure();
    }

    const bool hasScheduledReplicaInventory = llvm::any_of(
        sourceTasks, [](const TaskMetadata &task) {
          auto active = task.op->getAttrOfType<IntegerAttr>("active_replicas");
          return active && active.getInt() > 1;
        });
    // The original estimate consumes the unchanged parent module. Reprint
    // that current module independently, including the graph label installed
    // above, rather than trusting the catalog's serialized witness.
    if (originalF45ReplicaScaling && hasScheduledReplicaInventory)
      canonicalModuleWitness = neighborhoodReplaySourceText(module);
    if (!originalF45ReplicaScaling &&
        hasScheduledReplicaInventory != hasReplicaEvidence) {
      function.emitError()
          << "replica profile evidence presence does not match the canonical "
             "parent replica inventory";
      return signalPassFailure();
    }
    if (!verifyOriginalAmoebaSourceDomainCoverage(
            sourceTasks, error, diagnosticIICeiling == 23,
            hasScheduledReplicaInventory)) {
      sourceFunction.emitError() << error;
      return signalPassFailure();
    }

    OriginalScheduleInventory sourceSchedule;
    if (!readOriginalScheduleInventory(sourceFunction, sourceTasks,
                                       sourceSchedule, error)) {
      sourceFunction.emitError() << error;
      return signalPassFailure();
    }
    for (const OriginalTraceTask &trace : sourceSchedule.traces) {
      if (trace.compiledII > diagnosticIICeiling) {
        sourceFunction.emitError()
            << "original selected mapper II exceeds the runtime control-memory "
               "ceiling";
        return signalPassFailure();
      }
    }
    std::map<std::string, OriginalTraceTask> sourceTraceByName;
    for (auto [index, task] : llvm::enumerate(sourceTasks))
      sourceTraceByName.emplace(task.name, sourceSchedule.traces[index]);

    std::map<std::string, MaterializedParentDecision> materializedDecisions;
    std::map<std::string, std::pair<std::string, unsigned>> childLineage;
    std::map<std::pair<std::string, unsigned>,
             OriginalReplicaProfileEvidenceRef> replicaEvidenceRecords;
    std::string evidenceMaterializedModule;
    std::map<std::string, std::vector<std::string>> profileTasksByFile;
    std::map<std::string, std::vector<std::string>> bodyTasksByFile;
    if (hasReplicaEvidence) {
      llvm::StringMap<unsigned> expectedReplicaCounts;
      auto rawDecisions = function->getAttrOfType<ArrayAttr>(
          kOriginalMaterializedDecisionsAttr);
      for (Attribute raw : rawDecisions) {
        auto decision = dyn_cast<DictionaryAttr>(raw);
        auto parent = getString(decision, "parent_task");
        auto count = getInteger(decision, "original_replica_count");
        if (!decision || !parent || parent->empty() || !count || *count < 2 ||
            *count > 4 ||
            !expectedReplicaCounts.try_emplace(parent->str(),
                                                static_cast<unsigned>(*count))
                 .second) {
          function.emitError()
              << "materialized parent manifest has invalid replica identities";
          return signalPassFailure();
        }
      }
      if (expectedReplicaCounts.empty() ||
          !readReplicaProfileEvidenceManifest(
              replicaProfileEvidenceFile.getValue(), functionName.getValue(),
              expectedReplicaCounts, replicaEvidenceRecords,
              evidenceMaterializedModule, error) ||
          !verifyEvidenceMaterializedModule(evidenceMaterializedModule,
                                            functionName.getValue(),
                                            module.getContext(),
                                            currentInputModuleWitness, error) ||
          failed(verifySourceIterationDomainPartition(sourceFunction, function,
                                                       error)) ||
          !readMaterializedDecisions(function, currentTasks, sourceTasks,
                                     sourceTraceByName, materializedDecisions,
                                     childLineage, error)) {
        function.emitError()
            << (error.empty()
                    ? "materialized replica source proof or inventory is incomplete"
                    : error);
        return signalPassFailure();
      }
      std::set<std::string> allEvidenceChildren;
      for (const auto &entry : replicaEvidenceRecords) {
        const OriginalReplicaProfileEvidenceRef &reference = entry.second;
        allEvidenceChildren.insert(reference.materializedTask);
        profileTasksByFile[reference.profileFile].push_back(
            reference.materializedTask);
        bodyTasksByFile[reference.bodyExportFile].push_back(
            reference.materializedTask);
      }
      if (allEvidenceChildren.size() != childLineage.size() ||
          llvm::any_of(childLineage, [&](const auto &entry) {
            return !allEvidenceChildren.count(entry.first);
          })) {
        function.emitError()
            << "replica evidence children differ from the exact source-proved "
               "materialized child inventory";
        return signalPassFailure();
      }
      for (auto &entry : profileTasksByFile)
        llvm::sort(entry.second);
      for (auto &entry : bodyTasksByFile) {
        // The body verifier examines the full companion module, while the
        // mapper filter profiles only children listed in the manifest.
        entry.second.clear();
        for (const TaskMetadata &task : currentTasks)
          entry.second.push_back(task.name);
        llvm::sort(entry.second);
      }
    }

    llvm::SmallVector<TaskMetadata> tasks = std::move(currentTasks);
    llvm::StringMap<unsigned> taskIndices;
    std::vector<Operation *> taskOperations;
    std::vector<OriginalAmoebaRetimerTask> retimerTasks(tasks.size());
    taskOperations.reserve(tasks.size());
    for (unsigned index = 0; index < tasks.size(); ++index) {
      TaskMetadata &task = tasks[index];
      if (task.name.empty() ||
          !taskIndices.try_emplace(task.name, index).second) {
        function.emitError()
            << "expanded task names must be unique and nonempty";
        return signalPassFailure();
      }
      if (task.tripCount < 1) {
        task.op.emitError()
            << "compiler-inferred actual trip count is invalid";
        return signalPassFailure();
      }
      retimerTasks[index].name = task.name;
      taskOperations.push_back(task.op.getOperation());
    }

    std::vector<unsigned> dispatchOrder;
    std::vector<std::string> dispatchNames;
    if (hasReplicaEvidence) {
      for (const std::string &parentName : sourceSchedule.dispatchNames) {
        auto decision = materializedDecisions.find(parentName);
        if (decision == materializedDecisions.end()) {
          dispatchNames.push_back(parentName);
          continue;
        }
        for (unsigned replicaId = 0;
             replicaId < decision->second.replicaCount; ++replicaId)
          dispatchNames.push_back(
              decision->second.replicas.at(replicaId).taskName);
      }
      if (dispatchNames.size() != tasks.size()) {
        function.emitError()
            << "expanded dispatch order does not cover every materialized child";
        return signalPassFailure();
      }
      for (const std::string &name : dispatchNames) {
        auto task = taskIndices.find(name);
        if (task == taskIndices.end()) {
          function.emitError()
              << "expanded dispatch order names a missing child task";
          return signalPassFailure();
        }
        dispatchOrder.push_back(task->second);
      }
    } else {
      dispatchNames = sourceSchedule.dispatchNames;
      dispatchOrder = sourceSchedule.dispatchOrder;
      if (dispatchNames.size() != tasks.size()) {
        function.emitError()
            << "original dispatch order does not cover every task";
        return signalPassFailure();
      }
    }

    std::vector<OriginalTraceTask> traces(tasks.size());
    if (!hasReplicaEvidence) {
      traces = sourceSchedule.traces;
      for (auto [index, trace] : llvm::enumerate(traces)) {
        trace.placement.task = index;
        if (originalF45ReplicaScaling) {
          trace.placement.occupiedCells = trace.originalOccupiedCells;
          trace.contextIds = trace.originalContextIds;
        }
      }
    }
    std::vector<OriginalAmoebaFixedPlacement> placements;

    const ::mlir::neura::Architecture &architecture =
        ::mlir::neura::getArchitecture();
    const int gridRows = architecture.getMultiCgraRows();
    const int gridCols = architecture.getMultiCgraColumns();
    const int64_t perCgraRows = architecture.getPerCgraRows();
    const int64_t perCgraCols = architecture.getPerCgraColumns();
    if (gridRows != 4 || gridCols != 4 || perCgraRows < 1 || perCgraCols < 1) {
      function.emitError() << "configured architecture must be a validated "
                              "4x4 multi-CGRA grid";
      return signalPassFailure();
    }

    OriginalAmoebaProfileBodyExport bodyExport;
    if (!verifyOriginalProfileBinding(
            parentCostFile.getValue(), bodyExportFile.getValue(),
            sourceFunction, graphId.getValue(), sourceTasks,
            sourceSchedule.traces, perCgraRows, perCgraCols,
            diagnosticIICeiling, canonicalModuleWitness, bodyExport, error)) {
      function.emitError() << error;
      return signalPassFailure();
    }

    std::map<std::string, OriginalReplicaProfileEvidence> childEvidence;
    std::vector<std::string> parentTaskNames(tasks.size());
    std::vector<std::optional<unsigned>> replicaIds(tasks.size());
    std::vector<std::string> expandedProfileFiles(tasks.size());
    std::vector<std::string> expandedBodyExportFiles(tasks.size());
    std::vector<OriginalAmoebaProfileBodyEvidence> expandedBodies(tasks.size());
    if (hasReplicaEvidence) {
      for (const auto &entry : replicaEvidenceRecords) {
        const OriginalReplicaProfileEvidenceRef &reference = entry.second;
        auto decision = materializedDecisions.find(reference.parentTask);
        auto currentTask = taskIndices.find(reference.materializedTask);
        auto parentBody = bodyExport.tasks.find(reference.parentTask);
        auto profileTaskNames = profileTasksByFile.find(reference.profileFile);
        auto bodyTaskNames = bodyTasksByFile.find(reference.bodyExportFile);
        if (decision == materializedDecisions.end() ||
            currentTask == taskIndices.end() ||
            parentBody == bodyExport.tasks.end() ||
            profileTaskNames == profileTasksByFile.end() ||
            bodyTaskNames == bodyTasksByFile.end()) {
          function.emitError()
              << "replica evidence cannot bind to its parent, child, or profile file";
          return signalPassFailure();
        }
        const MaterializedReplicaDecision &replica =
            decision->second.replicas.at(reference.replicaId);
        auto shape = getString(replica.shape, "shape");
        auto count = getInteger(replica.shape, "cgra_count");
        if (!shape || !count) {
          function.emitError() << "replica manifest shape/count is malformed";
          return signalPassFailure();
        }
        OriginalReplicaProfileEvidence evidence;
        if (!readReplicaProfileAndBody(
                reference, profileTaskNames->second, bodyTaskNames->second,
                replica.tripCount, *shape, *count, diagnosticIICeiling,
                tasks[currentTask->second].op.getOperation(),
                parentBody->second, evidence, error) ||
            !childEvidence.emplace(reference.materializedTask,
                                   std::move(evidence))
                 .second) {
          function.emitError()
              << (error.empty() ? "replica profile evidence is duplicated"
                                : error);
          return signalPassFailure();
        }
      }
      for (unsigned index = 0; index < tasks.size(); ++index) {
        const TaskMetadata &currentTask = tasks[index];
        auto lineage = childLineage.find(currentTask.name);
        if (lineage == childLineage.end()) {
          auto parentTrace = sourceTraceByName.find(currentTask.name);
          auto body = bodyExport.tasks.find(currentTask.name);
          if (parentTrace == sourceTraceByName.end() ||
              body == bodyExport.tasks.end()) {
            function.emitError()
                << "unreplicated current task has no canonical parent binding";
            return signalPassFailure();
          }
          std::string currentBody;
          if (!computeOriginalAmoebaNormalizedMapperBody(
                  currentTask.op.operator->(), currentBody, error) ||
              currentBody != body->second.normalizedMapperBody) {
            function.emitError()
                << (error.empty()
                        ? "unreplicated current body differs from its canonical parent"
                        : error);
            return signalPassFailure();
          }
          traces[index] = parentTrace->second;
          traces[index].placement.task = index;
          parentTaskNames[index] = currentTask.name;
          expandedBodies[index] = body->second;
        } else {
          const auto &[parentName, replicaId] = lineage->second;
          auto parentTrace = sourceTraceByName.find(parentName);
          auto decision = materializedDecisions.find(parentName);
          auto evidence = childEvidence.find(currentTask.name);
          if (parentTrace == sourceTraceByName.end() ||
              decision == materializedDecisions.end() ||
              evidence == childEvidence.end() ||
              !makeExpandedReplicaTrace(
                  index, parentTrace->second,
                  decision->second.replicas.at(replicaId), evidence->second,
                  traces[index], error)) {
            function.emitError()
                << (error.empty()
                        ? "expanded replica is missing a parent trace or mapper profile"
                        : error);
            return signalPassFailure();
          }
          parentTaskNames[index] = parentName;
          replicaIds[index] = replicaId;
          expandedBodies[index] = evidence->second.body;
          const OriginalReplicaProfileEvidenceRef &reference =
              replicaEvidenceRecords.at({parentName, replicaId});
          expandedProfileFiles[index] = reference.profileFile;
          expandedBodyExportFiles[index] = reference.bodyExportFile;
        }
      }
    } else {
      for (unsigned index = 0; index < tasks.size(); ++index) {
        parentTaskNames[index] = tasks[index].name;
        auto body = bodyExport.tasks.find(tasks[index].name);
        if (body == bodyExport.tasks.end()) {
          function.emitError() << "canonical body export omits current task";
          return signalPassFailure();
        }
        expandedBodies[index] = body->second;
      }
    }
    placements.reserve(tasks.size());
    for (unsigned index = 0; index < tasks.size(); ++index) {
      traces[index].placement.task = index;
      placements.push_back(traces[index].placement);
    }

    std::string sourceRepository;
    std::string sourceCommit;
    std::string catalogArchitecturePath;
    std::string catalogModelStatus;
    bool catalogProductionReady = false;
    if (!readCostProvenance(parentCostFile.getValue(), sourceRepository,
                            sourceCommit, catalogArchitecturePath,
                            catalogProductionReady, catalogModelStatus,
                            error) ||
        !samePath(catalogArchitecturePath,
                  ::mlir::amoeba::getNeuraArchitectureSpecFile())) {
      function.emitError() << (error.empty()
                                   ? "cost catalog architecture differs from "
                                     "the configured architecture"
                                   : error);
      return signalPassFailure();
    }
    TaskShapeCostCache costs;
    const std::string costCanonicalModuleWitness =
        diagnosticIICeiling == 23
            ? (hasReplicaEvidence ? canonicalModuleWitness
                                  : neighborhoodReplaySourceText(module))
            : std::string();
    if (!costs.load(parentCostFile.getValue(), function.getSymName(), sourceTasks,
                    sourceRepository, sourceCommit, catalogArchitecturePath,
                    graphId.getValue(), error, diagnosticIICeiling == 23,
                    costCanonicalModuleWitness)) {
      function.emitError()
          << (error.empty()
                  ? "cost catalogue loader rejected input without diagnostic"
                  : error);
      return signalPassFailure();
    }

    std::vector<int64_t> durations(tasks.size(), 0);
    std::vector<int64_t> fullParentMappedDurations(tasks.size(), 0);
    std::vector<double> startupCycles(tasks.size(), 0.0);
    llvm::json::Array taskCostsJson;
    for (unsigned index = 0; index < tasks.size(); ++index) {
      const OriginalTraceTask &trace = traces[index];
      int64_t mapperRows = 0;
      int64_t mapperCols = 0;
      if (trace.placement.selectedRows >
              std::numeric_limits<int64_t>::max() / perCgraRows ||
          trace.placement.selectedCols >
              std::numeric_limits<int64_t>::max() / perCgraCols) {
        tasks[index].op.emitError() << "selected mapper dimensions overflow";
        return signalPassFailure();
      }
      mapperRows = trace.placement.selectedRows * perCgraRows;
      mapperCols = trace.placement.selectedCols * perCgraCols;
      TaskShapeChoice choice{parentTaskNames[index], tasks[index].tripCount,
                             RectShape{trace.placement.selectedRows,
                                       trace.placement.selectedCols, mapperRows,
                                       mapperCols}};
      std::string costError;
      const TaskShapeCost *cost = costs.get(choice, costError);
      if (!cost || !cost->supported) {
        tasks[index].op.emitError()
            << (costError.empty() ? "selected original shape has no supported "
                                    "bound startup cost"
                                  : costError);
        return signalPassFailure();
      }
      std::optional<int64_t> duration =
          computeMappedDuration(trace.compiledII, cost->startupCycles,
                                tasks[index].tripCount, costError);
      if (!duration) {
        tasks[index].op.emitError() << costError;
        return signalPassFailure();
      }
      fullParentMappedDurations[index] = *duration;
      if (originalF45ReplicaScaling) {
        int64_t scaledDuration = 0;
        if (!checkedF45SchedulerDuration(
                *duration, trace.placement.activeReplicas, scaledDuration)) {
          tasks[index].op.emitError()
              << "original F45 replica duration scaling is invalid";
          return signalPassFailure();
        }
        duration = scaledDuration;
      }
      durations[index] = *duration;
      startupCycles[index] = cost->startupCycles;
      llvm::json::Object item;
      item["task"] = tasks[index].name;
      if (hasReplicaEvidence) {
        item["parent_task"] = parentTaskNames[index];
        if (replicaIds[index])
          item["replica_id"] = static_cast<int64_t>(*replicaIds[index]);
        else
          item["replica_id"] = nullptr;
        item["selected_cgra_shape"] = trace.selectedShape;
        item["selected_cgra_count"] = trace.selectedCount;
      }
      item["trip_count"] = tasks[index].tripCount;
      item["profile_compiled_ii"] = trace.compiledII;
      item["profile_materialized_operation_count"] =
          trace.materializedOperationCount;
      item["profile_sample_trip_count"] = trace.sampleTripCount;
      item["profile_steps"] = trace.steps;
      item["original_profile_duration_cycles"] = trace.profileDuration;
      item["catalog_startup_cycles"] = cost->startupCycles;
      item["catalog_predicted_ii_not_used"] = cost->predictedII;
      item["mapped_duration_cycles"] = *duration;
      if (originalF45ReplicaScaling) {
        item["full_parent_mapped_duration_cycles"] =
            fullParentMappedDurations[index];
        item["original_active_replicas"] =
            static_cast<int64_t>(trace.placement.activeReplicas);
        item["replica_duration_rule"] =
            "ceil-full-parent-mapped-duration-over-original-active-replicas-v1";
      }
      item["selected_profile_shape"] = trace.selectedShape;
      item["selected_mapper_tile_rows"] = mapperRows;
      item["selected_mapper_tile_cols"] = mapperCols;
      if (diagnosticIICeiling == 23 || hasReplicaEvidence) {
        item["source_iteration_domain_status"] =
            tasks[index].sourceIterationDomainStatus;
        item["source_iteration_domain_complete"] =
            tasks[index].sourceIterationDomainComplete;
        item["source_iteration_multiplicity"] =
            tasks[index].sourceIterationMultiplicity;
        item["source_iteration_work_count"] =
            tasks[index].sourceIterationWorkCount;
        item["expanded_internal_extents"] =
            integerArray(tasks[index].expandedInternalExtents);
      }
      taskCostsJson.push_back(std::move(item));
    }

    auto network = createInterTaskNetworkCommunicationModel(function, gridRows,
                                                            gridCols, error);
    if (failed(network)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    ProductionFixedScheduleCommunication communication(**network,
                                                       taskOperations);
    for (unsigned task = 0; task < tasks.size(); ++task) {
      if (!communication.getPredecessors(task, retimerTasks[task].predecessors,
                                         error)) {
        function.emitError() << error;
        return signalPassFailure();
      }
    }

    OriginalAmoebaFixedDecisionResult retimed;
    if (!retimeOriginalAmoebaFixedDecisions(
            gridRows, gridCols, retimerTasks,
            originalF45ReplicaScaling ? fullParentMappedDurations : durations,
            placements,
            dispatchOrder, communication, retimed, originalF45ReplicaScaling)) {
      function.emitError() << (retimed.rejection.empty()
                                   ? "original fixed-decision retiming failed"
                                   : retimed.rejection);
      return signalPassFailure();
    }
    if (retimed.tasks.size() != durations.size() ||
        llvm::any_of(llvm::enumerate(retimed.tasks), [&](const auto &entry) {
          return entry.value().duration != durations[entry.index()];
        })) {
      function.emitError()
          << "retimer duration disagrees with the serialized mapper timing "
             "policy";
      return signalPassFailure();
    }

    // Keep this independently replayable record source-owned.  It serializes
    // every typed dependency and committed data route from the same strict
    // graph/model that verified the fixed schedule above.
    llvm::json::Object communicationContractJson;
    const InterTaskNetworkSpec &networkSpec = (*network)->getNetworkSpec();
    if (networkSpec.getLocalBandwidthBitsPerCycle() >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      function.emitError() << "local-channel bandwidth exceeds signed 64-bit "
                              "JSON";
      return signalPassFailure();
    }
    communicationContractJson["schema"] = "inter-task-network-v1";
    communicationContractJson["resource_interval_semantics"] =
        "half-open-start-inclusive-end-exclusive";
    communicationContractJson["version"] =
        static_cast<int64_t>(networkSpec.getVersion());
    communicationContractJson["rows"] = networkSpec.getRows();
    communicationContractJson["columns"] = networkSpec.getColumns();
    communicationContractJson["local_bandwidth_bits_per_cycle"] =
        static_cast<int64_t>(networkSpec.getLocalBandwidthBitsPerCycle());
    llvm::json::Array networkLinksJson;
    for (auto [linkIndex, link] : llvm::enumerate(networkSpec.getLinks())) {
      if (link.latency_cycles >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          link.bandwidth_bits_per_cycle >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        function.emitError() << "network contract exceeds signed 64-bit JSON";
        return signalPassFailure();
      }
      llvm::json::Object linkJson;
      linkJson["link_index"] = static_cast<int64_t>(linkIndex);
      linkJson["source_row"] = link.source.row;
      linkJson["source_col"] = link.source.column;
      linkJson["destination_row"] = link.destination.row;
      linkJson["destination_col"] = link.destination.column;
      linkJson["latency_cycles"] = static_cast<int64_t>(link.latency_cycles);
      linkJson["bandwidth_bits_per_cycle"] =
          static_cast<int64_t>(link.bandwidth_bits_per_cycle);
      networkLinksJson.push_back(std::move(linkJson));
    }
    communicationContractJson["links"] = std::move(networkLinksJson);

    llvm::json::Array dependenciesJson;
    llvm::json::Array routesJson;
    for (auto [edgeIndex, edge] :
         llvm::enumerate((*network)->getTypedEdges())) {
      llvm::json::Object dependency;
      dependency["edge_index"] = static_cast<int64_t>(edgeIndex);
      dependency["edge_id"] = "edge-" + std::to_string(edgeIndex);
      TaskflowTaskOp producer = edge.producer;
      TaskflowTaskOp consumer = edge.consumer;
      dependency["producer"] = producer.getTaskName().str();
      dependency["consumer"] = consumer.getTaskName().str();
      dependency["kind"] = stringifyTaskEdgeKind(edge.kind).str();
      dependency["origin"] = stringifyTaskEdgeOrigin(edge.origin).str();
      dependency["scope"] = stringifyTaskEdgeScope(edge.scope).str();
      dependency["producer_segment"] =
          stringifyTaskResultSegment(edge.producer_segment).str();
      dependency["producer_index"] = static_cast<int64_t>(edge.producer_index);
      dependency["consumer_segment"] =
          stringifyTaskOperandSegment(edge.consumer_segment).str();
      dependency["consumer_index"] = static_cast<int64_t>(edge.consumer_index);
      dependency["payload_known"] = edge.payload_bits.has_value();
      if (edge.payload_bits) {
        if (*edge.payload_bits >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          function.emitError() << "edge payload exceeds signed 64-bit JSON";
          return signalPassFailure();
        }
        dependency["payload_bits"] = static_cast<int64_t>(*edge.payload_bits);
      } else {
        dependency["payload_bits"] = nullptr;
      }
      dependency["transfer_region_lower"] =
          integerArray(edge.transfer_region_lower);
      dependency["transfer_region_upper"] =
          integerArray(edge.transfer_region_upper);
      dependenciesJson.push_back(std::move(dependency));
    }
    for (const auto &transfer : (*network)->getCommittedTransfers()) {
      auto producerOperation =
          std::find(taskOperations.begin(), taskOperations.end(),
                    transfer.producer);
      auto consumerOperation =
          std::find(taskOperations.begin(), taskOperations.end(),
                    transfer.consumer);
      if (producerOperation == taskOperations.end() ||
          consumerOperation == taskOperations.end() || transfer.links.empty() ||
          transfer.payloadBits >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          transfer.pathLatencyCycles >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          transfer.bottleneckBandwidthBitsPerCycle >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
          transfer.transferCycles >
              static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        function.emitError() << "committed route has invalid endpoints, no "
                                "resource intervals, or oversized metrics";
        return signalPassFailure();
      }
      const unsigned producerIndex = static_cast<unsigned>(
          std::distance(taskOperations.begin(), producerOperation));
      const unsigned consumerIndex = static_cast<unsigned>(
          std::distance(taskOperations.begin(), consumerOperation));
      const ExactSchedulePlacement &producerPlacement =
          retimed.placements[producerIndex];
      const ExactSchedulePlacement &consumerPlacement =
          retimed.placements[consumerIndex];
      llvm::json::Object route;
      route["route_id"] = tasks[producerIndex].name + "->" +
                          tasks[consumerIndex].name;
      route["producer"] = tasks[producerIndex].name;
      route["consumer"] = tasks[consumerIndex].name;
      route["payload_bits"] = static_cast<int64_t>(transfer.payloadBits);
      route["path_latency_cycles"] =
          static_cast<int64_t>(transfer.pathLatencyCycles);
      route["bottleneck_bandwidth_bits_per_cycle"] =
          static_cast<int64_t>(transfer.bottleneckBandwidthBitsPerCycle);
      route["transfer_cycles"] = static_cast<int64_t>(transfer.transferCycles);
      route["source_row"] = transfer.source.row;
      route["source_col"] = transfer.source.column;
      route["destination_row"] = transfer.destination.row;
      route["destination_col"] = transfer.destination.column;
      route["producer_finish_cycle"] = producerPlacement.end;
      route["consumer_start_cycle"] = consumerPlacement.start;
      route["ready_cycle"] = transfer.readyCycle;
      route["start_cycle"] = transfer.links.front().startCycle;
      llvm::json::Array edgeIndicesJson;
      for (uint32_t edgeIndex : transfer.edgeIndices)
        edgeIndicesJson.push_back(static_cast<int64_t>(edgeIndex));
      route["edge_indices"] = std::move(edgeIndicesJson);
      llvm::json::Array intervalsJson;
      for (const auto &interval : transfer.links) {
        if (interval.startCycle != transfer.links.front().startCycle ||
            interval.startCycle < producerPlacement.end ||
            transfer.readyCycle > consumerPlacement.start ||
            interval.startCycle < 0 ||
            interval.endCycle <= interval.startCycle ||
            interval.endCycle != transfer.readyCycle) {
          function.emitError() << "committed route interval is malformed or "
                                  "disagrees with ready cycle";
          return signalPassFailure();
        }
        llvm::json::Object intervalJson;
        intervalJson["start_cycle"] = interval.startCycle;
        intervalJson["end_cycle"] = interval.endCycle;
        if (interval.localChannel) {
          intervalJson["resource_kind"] = "local_channel";
          intervalJson["row"] = interval.localCoordinate.row;
          intervalJson["col"] = interval.localCoordinate.column;
        } else {
          if (interval.linkIndex >= networkSpec.getLinks().size()) {
            function.emitError() << "committed route names an unknown network "
                                    "link";
            return signalPassFailure();
          }
          intervalJson["resource_kind"] = "network_link";
          intervalJson["link_index"] = static_cast<int64_t>(interval.linkIndex);
        }
        intervalsJson.push_back(std::move(intervalJson));
      }
      route["intervals"] = std::move(intervalsJson);
      routesJson.push_back(std::move(route));
    }

    llvm::json::Array taskProfileBindingsJson;
    for (auto [index, task] : llvm::enumerate(tasks)) {
      const OriginalTraceTask &trace = traces[index];
      const OriginalAmoebaProfileBodyEvidence &body = expandedBodies[index];
      llvm::json::Object profileBinding;
      profileBinding["task"] = task.name;
      if (hasReplicaEvidence) {
        profileBinding["parent_task"] = parentTaskNames[index];
        if (replicaIds[index])
          profileBinding["replica_id"] =
              static_cast<int64_t>(*replicaIds[index]);
        else
          profileBinding["replica_id"] = nullptr;
      }
      profileBinding["static_trip_count"] = task.tripCount;
      profileBinding["task_signature"] = body.taskSignature;
      profileBinding["counter_signature"] = body.counterSignature;
      profileBinding["kernel_binding_signature"] =
          body.kernelBindingSignature;
      profileBinding["normalized_mapper_body"] =
          body.normalizedMapperBody;
      profileBinding["selected_profile_shape"] = trace.selectedShape;
      profileBinding["selected_cgra_count"] = trace.selectedCount;
      if (hasReplicaEvidence) {
        profileBinding["selected_mapper_tile_rows"] =
            trace.placement.selectedRows * perCgraRows;
        profileBinding["selected_mapper_tile_cols"] =
            trace.placement.selectedCols * perCgraCols;
      }
      profileBinding["compiled_ii"] = trace.compiledII;
      profileBinding["steps"] = trace.steps;
      profileBinding["sample_trip_count"] = trace.sampleTripCount;
      profileBinding["materialized_operation_count"] =
          trace.materializedOperationCount;
      profileBinding["profile_duration_cycles"] = trace.profileDuration;
      profileBinding["catalog_startup_cycles"] = startupCycles[index];
      profileBinding["mapped_duration_cycles"] = durations[index];
      if (originalF45ReplicaScaling) {
        profileBinding["full_parent_mapped_duration_cycles"] =
            fullParentMappedDurations[index];
        profileBinding["original_active_replicas"] =
            static_cast<int64_t>(trace.placement.activeReplicas);
        profileBinding["replica_duration_rule"] =
            "ceil-full-parent-mapped-duration-over-original-active-replicas-v1";
      }
      profileBinding["mapper_succeeded"] = true;
      profileBinding["current_ir_body_equivalence_verified"] = true;
      if (replicaIds[index]) {
        profileBinding["materialized_task"] = task.name;
        profileBinding["profile_file"] = expandedProfileFiles[index];
        profileBinding["body_export_file"] = expandedBodyExportFiles[index];
        profileBinding["source_iteration_domain_status"] =
            task.sourceIterationDomainStatus;
        profileBinding["source_iteration_domain_complete"] =
            task.sourceIterationDomainComplete;
        profileBinding["source_iteration_multiplicity"] =
            task.sourceIterationMultiplicity;
        profileBinding["source_iteration_work_count"] =
            task.sourceIterationWorkCount;
        profileBinding["expanded_internal_extents"] =
            integerArray(task.expandedInternalExtents);
      }
      taskProfileBindingsJson.push_back(std::move(profileBinding));
    }

    llvm::json::Array dispatchJson;
    for (unsigned task : retimed.dispatchOrder)
      dispatchJson.push_back(tasks[task].name);
    llvm::json::Array scheduleJson;
    for (const ExactSchedulePlacement &placement : retimed.placements) {
      const OriginalTraceTask &trace = traces[placement.task];
      llvm::json::Object item;
      item["task"] = tasks[placement.task].name;
      item["row"] = placement.row;
      item["col"] = placement.col;
      item["rows"] = trace.placement.rows;
      item["cols"] = trace.placement.cols;
      item["start_cycle"] = placement.start;
      item["end_cycle"] = placement.end;
      item["idle_cycles"] = placement.idleCycles;
      item["duration_cycles"] = durations[placement.task];
      llvm::json::Array occupiedCellsJson;
      for (auto [cellIndex, cell] :
           llvm::enumerate(traces[placement.task].placement.occupiedCells)) {
        llvm::json::Object cellJson;
        cellJson["row"] = cell.row;
        cellJson["col"] = cell.col;
        cellJson["replica_id"] = static_cast<int64_t>(
            traces[placement.task].sourceReplicaId.value_or(cell.replicaId));
        cellJson["context_id"] = traces[placement.task].contextIds[cellIndex];
        occupiedCellsJson.push_back(std::move(cellJson));
      }
      item["occupied_cells"] = std::move(occupiedCellsJson);
      scheduleJson.push_back(std::move(item));
    }
    llvm::json::Array originalDecisionsJson;
    for (unsigned index = 0; index < sourceTasks.size(); ++index) {
      const TaskMetadata &sourceTask = sourceTasks[index];
      const OriginalTraceTask &trace = sourceSchedule.traces[index];
      const unsigned originalDispatchIndex = static_cast<unsigned>(
          std::find(sourceSchedule.dispatchOrder.begin(),
                    sourceSchedule.dispatchOrder.end(), index) -
          sourceSchedule.dispatchOrder.begin());
      llvm::json::Object item;
      item["task"] = sourceTask.name;
      item["selected_profile_shape"] = trace.selectedShape;
      item["active_replicas"] =
          static_cast<int64_t>(trace.placement.activeReplicas);
      item["selected_cgra_count"] = trace.selectedCount;
      item["dispatch_index"] = static_cast<int64_t>(originalDispatchIndex);
      item["original_scheduler_start_internal"] = trace.schedulerStart;
      item["original_scheduler_end_internal"] = trace.schedulerEnd;
      item["original_scheduler_duration_internal"] = trace.schedulerDuration;
      item["original_profile_duration_cycles"] = trace.profileDuration;
      if (trace.placement.activeReplicas > 1) {
        item["original_profile_duration_binding"] = llvm::json::Object{
            {"schema", "amoeba-original-f45-profile-duration-binding-v1"},
            {"full_parent_mapper_duration_cycles", trace.fullMapperDuration},
            {"f45_scheduler_duration_cycles", trace.profileDuration},
            {"active_replicas",
             static_cast<int64_t>(trace.placement.activeReplicas)},
            {"compiled_ii", trace.compiledII},
            {"sample_trip_count", trace.sampleTripCount},
            {"steps", trace.steps},
            {"materialized_operation_count", trace.materializedOperationCount},
            {"rounding_rule",
             "ceil-full-parent-duration-over-original-active-replicas-v1"},
            {"status", "verified"}};
      }

      if (hasReplicaEvidence ||
          (originalF45ReplicaScaling && trace.placement.activeReplicas > 1)) {
        if (trace.replicas.empty() ||
            trace.originalOccupiedCells.size() != trace.originalContextIds.size()) {
          function.emitError()
              << "original parent trace omits its complete replica cell inventory";
          return signalPassFailure();
        }
        const OriginalReplicaTrace &firstReplica = trace.replicas.front();
        item["actual_placed_shape"] = firstReplica.shape;
        item["actual_placed_row"] = firstReplica.row;
        item["actual_placed_col"] = firstReplica.col;
        llvm::json::Array allCells;
        for (auto [cellIndex, cell] :
             llvm::enumerate(trace.originalOccupiedCells))
          allCells.push_back(llvm::json::Object{
              {"row", cell.row},
              {"col", cell.col},
              {"replica_id", static_cast<int64_t>(cell.replicaId)},
              {"context_id", trace.originalContextIds[cellIndex]}});
        item["actual_cells"] = std::move(allCells);
        item["context_ids"] = integerArray(trace.originalContextIds);
        llvm::json::Array replicaRows;
        for (const OriginalReplicaTrace &replica : trace.replicas) {
          if (replica.occupiedCells.size() != replica.contextIds.size()) {
            function.emitError()
                << "original replica trace omits exact context IDs";
            return signalPassFailure();
          }
          llvm::json::Array cells;
          for (auto [cellIndex, cell] :
               llvm::enumerate(replica.occupiedCells))
            cells.push_back(llvm::json::Object{
                {"row", cell.row},
                {"col", cell.col},
                {"replica_id", static_cast<int64_t>(replica.replicaId)},
                {"context_id", replica.contextIds[cellIndex]}});
          llvm::json::Object replicaJson{
              {"replica_id", static_cast<int64_t>(replica.replicaId)},
              {"shape", replica.shape},
              {"cgra_count", replica.cgraCount},
              {"row", replica.row},
              {"col", replica.col},
              {"rows", replica.rows},
              {"cols", replica.cols},
              {"actual_cells", std::move(cells)},
              {"context_ids", integerArray(replica.contextIds)}};
          replicaRows.push_back(std::move(replicaJson));
        }
        item["replicas"] = std::move(replicaRows);
      } else {
        item["actual_placed_shape"] = trace.actualShape;
        item["actual_placed_row"] = trace.placement.row;
        item["actual_placed_col"] = trace.placement.col;
        llvm::json::Array cells;
        for (auto [cellIndex, cell] :
             llvm::enumerate(trace.placement.occupiedCells))
          cells.push_back(llvm::json::Object{
              {"row", cell.row},
              {"col", cell.col},
              {"replica_id", static_cast<int64_t>(cell.replicaId)},
              {"context_id", trace.contextIds[cellIndex]}});
        item["actual_cells"] = std::move(cells);
        item["context_ids"] = integerArray(trace.contextIds);
      }
      originalDecisionsJson.push_back(std::move(item));
    }

    llvm::json::Object replicaMaterializationJson;
    if (hasReplicaEvidence) {
      llvm::json::Array parentPartitions;
      for (const std::string &parentName : sourceSchedule.dispatchNames) {
        auto decision = materializedDecisions.find(parentName);
        if (decision == materializedDecisions.end())
          continue;
        auto sourceTask = llvm::find_if(sourceTasks, [&](const TaskMetadata &task) {
          return task.name == parentName;
        });
        auto sourceTrace = sourceTraceByName.find(parentName);
        if (sourceTask == sourceTasks.end() ||
            sourceTrace == sourceTraceByName.end()) {
          function.emitError()
              << "materialization proof lost its canonical parent task";
          return signalPassFailure();
        }
        const MaterializedParentDecision &parent = decision->second;
        llvm::json::Object parentJson{
            {"parent_task", parentName},
            {"original_replica_count",
             static_cast<int64_t>(parent.replicaCount)},
            {"selected_counter_ordinal", parent.selectedCounterOrdinal},
            {"selected_output_axis", parent.selectedOutputAxis},
            {"partition_axis", parent.selectedCounterOrdinal},
            {"original_domain",
             sourceDomainBoundsJson(parent.decodedOriginalDomain)},
            {"source_iteration_domain_status",
             sourceTask->sourceIterationDomainStatus},
            {"source_iteration_domain_complete",
             sourceTask->sourceIterationDomainComplete},
            {"source_iteration_multiplicity",
             sourceTask->sourceIterationMultiplicity},
            {"source_iteration_work_count",
             sourceTask->sourceIterationWorkCount},
            {"expanded_internal_extents",
             integerArray(sourceTask->expandedInternalExtents)}};
        llvm::json::Array replicaPartitions;
        for (unsigned replicaId = 0; replicaId < parent.replicaCount;
             ++replicaId) {
          const MaterializedReplicaDecision &replica =
              parent.replicas.at(replicaId);
          const OriginalReplicaTrace &originalReplica =
              sourceTrace->second.replicas[replicaId];
          auto childIndex = taskIndices.find(replica.taskName);
          if (childIndex == taskIndices.end()) {
            function.emitError()
                << "source partition proof lost a current materialized child";
            return signalPassFailure();
          }
          const TaskMetadata &childTask = tasks[childIndex->second];
          llvm::json::Array cells;
          for (auto [cellIndex, cell] :
               llvm::enumerate(originalReplica.occupiedCells))
            cells.push_back(llvm::json::Object{
                {"row", cell.row},
                {"col", cell.col},
                {"replica_id", static_cast<int64_t>(replicaId)},
                {"context_id", originalReplica.contextIds[cellIndex]}});
          llvm::json::Object replicaJson{
              {"replica_id", static_cast<int64_t>(replicaId)},
              {"materialized_task", replica.taskName},
              {"partition_axis", parent.selectedCounterOrdinal},
              {"partition_bounds",
               sourceDomainBoundsJson(replica.decodedPartitionBounds)},
              {"source_iteration_domain_status",
               childTask.sourceIterationDomainStatus},
              {"source_iteration_domain_complete",
               childTask.sourceIterationDomainComplete},
              {"source_iteration_multiplicity",
               childTask.sourceIterationMultiplicity},
              {"source_iteration_work_count",
               childTask.sourceIterationWorkCount},
              {"expanded_internal_extents",
               integerArray(childTask.expandedInternalExtents)},
              {"trip_count", childTask.tripCount},
              {"actual_placed_shape", originalReplica.shape},
              {"selected_profile_shape", originalReplica.shape},
              {"selected_cgra_count", originalReplica.cgraCount},
              {"row", originalReplica.row},
              {"col", originalReplica.col},
              {"rows", originalReplica.rows},
              {"cols", originalReplica.cols},
              {"actual_cells", std::move(cells)},
              {"context_ids", integerArray(originalReplica.contextIds)}};
          replicaPartitions.push_back(std::move(replicaJson));
        }
        parentJson["replicas"] = std::move(replicaPartitions);
        parentPartitions.push_back(std::move(parentJson));
      }
      replicaMaterializationJson["schema"] =
          kReplicaMaterializationSchema.str();
      replicaMaterializationJson["verified"] = true;
      replicaMaterializationJson["materialized_module"] =
          evidenceMaterializedModule;
      replicaMaterializationJson["parent_partitions"] =
          std::move(parentPartitions);
    }

    llvm::json::Object replicaTimingPolicy;
    if (originalF45ReplicaScaling) {
      replicaTimingPolicy = llvm::json::Object{
          {"schema", "amoeba-original-f45-replica-scaling-v1"},
          {"duration_formula",
           "ceil(ceil(catalog_startup_cycles + compiled_ii * "
           "(source_macro_firings - 1)) / original_active_replicas)"},
          {"replica_duration_rule",
           "ceil-full-parent-mapped-duration-over-original-active-replicas-v1"},
          {"child_mapper_profiles_used", false},
          {"status", "original-f45-scheduler-estimate"}};
    }
    llvm::json::Object result;
    if (originalF45ReplicaScaling)
      result["replica_timing_policy"] = llvm::json::Object(replicaTimingPolicy);
    result["record_type"] = "result";
    result["schema"] = "amoeba-original-fixed-decision-retiming-v1";
    result["function"] = function.getSymName().str();
    result["graph_variant_id"] = graphId.getValue().str();
    result["graph_variant_id_source"] =
        assignedOriginalGraphId ? "assigned-original-amoeba-full-baseline-label"
                                : "preserved-source-graph-variant-id";
    result["candidate_id"] = "candidate-0";
    result["candidate_origin"] = "original-amoeba-throughput-guided";
    result["valid"] = true;
    result["diagnostic_only"] = true;
    result["formal_go"] = false;
    result["body_equivalence_checked"] = true;
    result["body_equivalence_status"] =
        "verified-current-kernel-equals-exported-original-normalized-body";
    const StringRef sourceCoverageStatus =
        originalAmoebaSourceDomainCoverageStatus(sourceTasks);
    result["iteration_domain_coverage_status"] = sourceCoverageStatus.str();
    result["iteration_domain_coverage_verified"] = true;
    result["iteration_domain_coverage_evidence"] =
        originalF45ReplicaScaling
            ? "the complete original parent source domains, actual full-parent "
              "mapper bodies/profiles, and every original replica placement "
              "were verified; replica durations follow the original F45 "
              "scheduler estimate and are not actual child mapper measurements"
            : hasReplicaEvidence
            ? "the exact canonical parent graph, complete source partition, "
              "all actual child bodies/profiles, and original replica cells "
              "were revalidated in C++"
            : diagnosticIICeiling == 23
            ? "the source-domain resolver revalidated every current task, "
              "including the complete static-internal expansion; sample trips "
              "remain macro firings and source work counts include expanded extents"
            : "the source-domain guard revalidated a complete bound certificate "
              "for every original task and rejected internal multiplicity, changed "
              "firing extents, active replicas, and replica/tiling lineage";
    result["selected_profile_body_binding_schema"] =
        kOriginalAmoebaProfileBindingSchema.str();
    result["selected_profile_body_binding_verified"] = true;
    result["whole_program_result_scope"] =
        "retime-of-supplied-original-decision-trace; not-certified-as-"
        "original-full-flow";
    result["required_external_evidence"] = llvm::json::Array{
        "mapper-profiles-from-equivalent-correct-original-flow",
        "native-simulation-and-independent-trace"};
    result["mapper_success_evidence"] =
        "source-inferred-success-only-profile-info; all-one-fallback-rejected";
    result["explicit_mapper_succeeded_attribute_present"] = false;
    result["duration_source"] =
        originalF45ReplicaScaling
            ? "original-f45-replica-scaling-of-common-full-parent-compiled-ii-"
              "and-structural-startup-duration"
            : hasReplicaEvidence
            ? "actual-parent-or-replica-profile-compiled-ii-times-actual-trip-"
              "minus-one-plus-original-parent-orientation-catalog-startup"
            : "profile_info.compiled_ii-times-inferred-trip-count-minus-one-plus-"
              "bound-cost-catalog-startup";
    result["score_source"] =
        "preserved-original-amoeba-decisions-plus-explicit-production-network";
    result["mapped_whole_program_cycles"] = retimed.replay.predictedMakespan;
    result["predicted_whole_program_cycles"] = retimed.replay.predictedMakespan;
    result["replayed_communication_edges"] =
        static_cast<int64_t>(retimed.replay.replayedEdges);
    result["original_pipeline_interval"] = sourceSchedule.pipelineInterval;
    result["original_pipeline_interval_source"] =
        "task_orchestration_summary.pipeline_interval; separate metadata";
    result["original_internal_time_unit"] = sourceSchedule.internalTimeUnit;
    result["original_internal_time_scale"] = sourceSchedule.internalTimeScale;
    result["grid_rows"] = gridRows;
    result["grid_cols"] = gridCols;
    result["max_cgras_per_task"] = 4;
    result["cost_catalog_namespace"] = costs.nameSpace().str();
    result["cost_catalog_source_repository"] = costs.sourceRepository().str();
    result["cost_catalog_model_status"] = catalogModelStatus;
    result["cost_catalog_production_ready"] = catalogProductionReady;
    result["cost_catalog_architecture_path"] = costs.architecturePath().str();
    if (hasReplicaEvidence) {
      llvm::json::Array originalParentDispatchJson;
      for (const std::string &name : sourceSchedule.dispatchNames)
        originalParentDispatchJson.push_back(name);
      result["original_parent_dispatch_order"] =
          std::move(originalParentDispatchJson);
      result["replica_materialization"] =
          llvm::json::Object(replicaMaterializationJson);
    }
    if (diagnosticIICeiling == 23) {
      auto catalogBuffer =
          llvm::MemoryBuffer::getFile(parentCostFile.getValue());
      if (!catalogBuffer) {
        function.emitError() << "cannot reread validated II23 cost catalog";
        return signalPassFailure();
      }
      auto catalogValue = llvm::json::parse((*catalogBuffer)->getBuffer());
      if (!catalogValue) {
        function.emitError() << "cannot parse validated II23 cost catalog: "
                             << llvm::toString(catalogValue.takeError());
        return signalPassFailure();
      }
      const llvm::json::Object *catalogRoot =
          catalogValue->getAsObject();
      const llvm::json::Object *catalogMetadata =
          catalogRoot ? catalogRoot->getObject("predictor_metadata") : nullptr;
      const llvm::json::Object *diagnosticOverride =
          catalogMetadata ? catalogMetadata->getObject("diagnostic_override")
                          : nullptr;
      if (!diagnosticOverride) {
        function.emitError() << "validated II23 cost catalog lost its diagnostic proof";
        return signalPassFailure();
      }
      result["diagnostic_override"] = llvm::json::Object(*diagnosticOverride);
      result["training_ii_ceiling"] = 20;
      result["runtime_ii_ceiling"] = 23;
      result["model_interval_max_ii"] = 20;
      result["unsupported_prediction_policy"] =
          "analytical-lower-bound-exceeds-diagnostic-runtime-ceiling-v1";
    }
    result["original_decisions"] = std::move(originalDecisionsJson);
    result["task_costs"] = std::move(taskCostsJson);
    result["dispatch_order"] = std::move(dispatchJson);
    result["task_schedule"] = std::move(scheduleJson);
    llvm::json::Object fixedDecisionTrace;
    if (originalF45ReplicaScaling)
      fixedDecisionTrace["replica_timing_policy"] =
          llvm::json::Object(replicaTimingPolicy);
    fixedDecisionTrace["schema"] = "amoeba-fixed-decision-trace-v1";
    fixedDecisionTrace["candidate_id"] = "candidate-0";
    fixedDecisionTrace["graph_variant_id"] = graphId.getValue().str();
    fixedDecisionTrace["formal_go"] = false;
    fixedDecisionTrace["iteration_domain_coverage_status"] =
        sourceCoverageStatus.str();
    fixedDecisionTrace["iteration_domain_coverage_verified"] = true;
    fixedDecisionTrace["body_equivalence_checked"] = true;
    fixedDecisionTrace["body_equivalence_status"] =
        "current-ir-exactly-matches-profile-body-export";
    fixedDecisionTrace["communication_contract"] =
        std::move(communicationContractJson);
    fixedDecisionTrace["task_profile_bindings"] =
        std::move(taskProfileBindingsJson);
    if (diagnosticIICeiling == 23) {
      fixedDecisionTrace["iteration_domain_coverage_evidence"] =
          "the C++ source resolver revalidated static-internal extents and "
          "kept macro trip count separate from expanded source work";
    }
    fixedDecisionTrace["dependencies"] = std::move(dependenciesJson);
    fixedDecisionTrace["routes"] = std::move(routesJson);
    llvm::json::Array traceDispatchJson;
    for (const std::string &name : dispatchNames)
      traceDispatchJson.push_back(name);
    fixedDecisionTrace["dispatch_order"] = std::move(traceDispatchJson);
    llvm::json::Array traceScheduleJson;
    // Rebuild a compact schedule with exact fixed cells and retimed intervals.
    for (const ExactSchedulePlacement &placement : retimed.placements) {
      const OriginalTraceTask &trace = traces[placement.task];
      llvm::json::Object schedule;
      schedule["task"] = tasks[placement.task].name;
      schedule["row"] = placement.row;
      schedule["col"] = placement.col;
      schedule["rows"] = trace.placement.rows;
      schedule["cols"] = trace.placement.cols;
      schedule["start_cycle"] = placement.start;
      schedule["end_cycle"] = placement.end;
      schedule["duration_cycles"] = durations[placement.task];
      schedule["idle_cycles"] = placement.idleCycles;
      llvm::json::Array cells;
      for (auto [cellIndex, cell] :
           llvm::enumerate(trace.placement.occupiedCells)) {
        llvm::json::Object cellJson;
        cellJson["row"] = cell.row;
        cellJson["col"] = cell.col;
        cellJson["replica_id"] = static_cast<int64_t>(
            trace.sourceReplicaId.value_or(cell.replicaId));
        cellJson["context_id"] = trace.contextIds[cellIndex];
        cells.push_back(std::move(cellJson));
      }
      schedule["occupied_cells"] = std::move(cells);
      traceScheduleJson.push_back(std::move(schedule));
    }
    fixedDecisionTrace["task_schedule"] = std::move(traceScheduleJson);
    if (hasReplicaEvidence) {
      llvm::json::Array originalParentDispatchJson;
      for (const std::string &name : sourceSchedule.dispatchNames)
        originalParentDispatchJson.push_back(name);
      fixedDecisionTrace["original_parent_dispatch_order"] =
          std::move(originalParentDispatchJson);
    }
    result["fixed_decision_trace"] = std::move(fixedDecisionTrace);

    if (!writeAtomically(
            outputFile.getValue(),
            [&](llvm::raw_ostream &out) {
              writeJsonLine(out, std::move(result));
              return true;
            },
            error)) {
      function.emitError() << error;
      return signalPassFailure();
    }

    // The common host scheduler consumes these established exact-replay
    // attributes. They retain the original cell rectangle and dispatch order;
    // the pass never launches the scheduler or chooses a replacement.
    OpBuilder builder(module.getContext());
    SmallVector<Attribute> exactDispatch;
    for (const std::string &name : dispatchNames)
      exactDispatch.push_back(builder.getStringAttr(name));
    function->setAttr("joint_scheduling_exact_dispatch_order",
                      builder.getArrayAttr(exactDispatch));
    function->setAttr(
        "joint_scheduling_predicted_makespan",
        builder.getI64IntegerAttr(retimed.replay.predictedMakespan));
    function->setAttr("joint_scheduling_exact_replay_timing",
                      builder.getStringAttr(
                          "mapped-duration-preserve-location-dispatch-idle"));
    function->setAttr("amoeba.original_amoeba_fixed_decision_retime",
                      builder.getStringAttr("diagnostic-only"));
    function->setAttr("joint_scheduling_candidate_id",
                      builder.getStringAttr("candidate-0"));
    function->setAttr("joint_scheduling_graph_variant_id",
                      builder.getStringAttr(graphId.getValue()));
    for (unsigned index = 0; index < tasks.size(); ++index) {
      taskflow::TaskflowTaskOp task = tasks[index].op;
      const ExactSchedulePlacement &placement = retimed.placements[index];
      const OriginalTraceTask &trace = traces[index];
      SmallVector<NamedAttribute> fields{
          builder.getNamedAttr("row", builder.getI64IntegerAttr(placement.row)),
          builder.getNamedAttr("col", builder.getI64IntegerAttr(placement.col)),
          builder.getNamedAttr("rows",
                               builder.getI64IntegerAttr(trace.placement.rows)),
          builder.getNamedAttr("cols",
                               builder.getI64IntegerAttr(trace.placement.cols)),
          builder.getNamedAttr("start_cycle",
                               builder.getI64IntegerAttr(placement.start)),
          builder.getNamedAttr("end_cycle",
                               builder.getI64IntegerAttr(placement.end)),
          builder.getNamedAttr(
              "idle_cycles", builder.getI64IntegerAttr(placement.idleCycles))};
      task->setAttr("amoeba.exact_schedule", builder.getDictionaryAttr(fields));
      task->setAttr("amoeba.exact_replay_mapped_timing", builder.getUnitAttr());
      task->setAttr("amoeba.joint_shape_orientation_fixed",
                    builder.getUnitAttr());
      task->setAttr("cgra_shape", builder.getStringAttr(trace.actualShape));
      task->setAttr("cgra_count",
                    builder.getI32IntegerAttr(trace.placement.rows *
                                              trace.placement.cols));
      task->setAttr("est_latency", builder.getI64IntegerAttr(durations[index]));
      task->setAttr("amoeba.original_amoeba_selected_cgra_shape",
                    builder.getStringAttr(trace.selectedShape));
    }
  }
};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createRetimingOriginalAmoebaFixedDecisionsPass() {
  return std::make_unique<RetimingOriginalAmoebaFixedDecisionsPass>();
}
} // namespace mlir::amoeba::neura
