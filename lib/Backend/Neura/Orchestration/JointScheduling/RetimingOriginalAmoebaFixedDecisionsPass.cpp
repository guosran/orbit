//===- RetimingOriginalAmoebaFixedDecisionsPass.cpp ----------------------===//
// Replays an original AMOEBA throughput-guided trace with common mapped
// durations and explicit communication while holding every original spatial
// and dispatch decision fixed.
//===----------------------------------------------------------------------===//

#include "RetimingOriginalAmoebaFixedDecisionsPass.h"

#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalTaskCostCatalog.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "OriginalAmoebaFixedDecisionRetimer.h"
#include "OriginalAmoebaProfileCostAdapter.h"
#include "ProductionFixedScheduleCommunication.h"
#include "SpatialTaskCandidateSpace.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

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

struct OriginalTraceTask {
  OriginalAmoebaFixedPlacement placement;
  std::string selectedShape;
  std::string actualShape;
  int64_t selectedCount = 0;
  int64_t profileDuration = 0;
  int64_t compiledII = 0;
  int64_t materializedOperationCount = 0;
  int64_t sampleTripCount = 0;
  int64_t steps = 0;
  int64_t schedulerStart = 0;
  int64_t schedulerEnd = 0;
  int64_t schedulerDuration = 0;
  std::vector<int64_t> contextIds;
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

static bool verifyOriginalProfileBinding(
    StringRef costPath, StringRef bodyExportPath, func::FuncOp function,
    StringRef graphVariant, ArrayRef<TaskMetadata> tasks,
    ArrayRef<OriginalTraceTask> traces, int64_t perCgraRows,
    int64_t perCgraCols, OriginalAmoebaProfileBodyExport &bodyExport,
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
      !domainCoverageStatus || *domainCoverageStatus != "pending" ||
      !domainCoverageVerified || *domainCoverageVerified ||
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
        !estimatedLatency || *estimatedLatency != trace.profileDuration ||
        !mapperSucceeded || !*mapperSucceeded) {
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
  if (*activeReplicas != 1)
    return fail(
        "active_replicas is not 1; shard materialization is unsupported");
  if (*durationUnit != "profiled-cycles" || *rawDuration < 1 ||
      *schedulerStart < 0 || *schedulerEnd <= *schedulerStart)
    return fail(
        "profiled duration unit or original scheduler times are invalid");
  if (placements.empty() || replicaShapes.size() != 1)
    return fail("one actual replica shape and a nonempty cell inventory are "
                "required");

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
  if (!checkedProfileDuration(*compiledII, *sampleTripCount, *steps,
                              formulaDuration) ||
      formulaDuration != *profileDuration)
    return fail("profile duration is inconsistent with compiled_ii, sample "
                "trip count, and steps");

  auto shapeValue = dyn_cast<DictionaryAttr>(replicaShapes[0]);
  if (!shapeValue)
    return fail("replica_shapes contains a non-dictionary entry");
  auto replicaId = getInteger(shapeValue, "replica_id");
  auto replicaCount = getInteger(shapeValue, "cgra_count");
  auto replicaRow = getInteger(shapeValue, "row");
  auto replicaCol = getInteger(shapeValue, "col");
  auto replicaRows = getInteger(shapeValue, "placement_rows");
  auto replicaCols = getInteger(shapeValue, "placement_cols");
  auto actualShapeAttr = getString(shapeValue, "shape");
  if (!replicaId || !replicaCount || !replicaRow || !replicaCol ||
      !replicaRows || !replicaCols || !actualShapeAttr || *replicaId != 0 ||
      *replicaCount < 1 || *replicaRows < 1 || *replicaCols < 1 ||
      *replicaRows > 4 || *replicaCols > 4 ||
      *replicaRows * *replicaCols != *replicaCount)
    return fail("captured replica shape fields are invalid");
  int64_t actualShapeRows = 0;
  int64_t actualShapeCols = 0;
  if (!parseShape(*actualShapeAttr, actualShapeRows, actualShapeCols) ||
      actualShapeRows != *replicaRows || actualShapeCols != *replicaCols ||
      *replicaRows * *replicaCols > 4 ||
      ((*replicaRows != selectedRows || *replicaCols != selectedCols) &&
       (*replicaRows != selectedCols || *replicaCols != selectedRows)))
    return fail("actual placed shape disagrees with the captured replica "
                "shape or selected orientation/area");

  std::set<std::pair<int64_t, int64_t>> cells;
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
        *context < 0 || *placedReplica != 0 || *start < 0 || *end <= *start ||
        *duration != *end - *start || !cells.emplace(*row, *col).second)
      return fail("placement cell, context, replica, or original times are "
                  "invalid or duplicated");
    if (commonStart < 0) {
      commonStart = *start;
      commonEnd = *end;
      commonDuration = *duration;
    } else if (commonStart != *start || commonEnd != *end ||
               commonDuration != *duration) {
      return fail("one original replica has inconsistent per-cell times");
    }
    contexts.emplace(std::make_pair(*row, *col), *context);
  }
  const int64_t minRow = cells.begin()->first;
  const int64_t maxRow = cells.rbegin()->first;
  int64_t minCol = std::numeric_limits<int64_t>::max();
  int64_t maxCol = -1;
  for (const auto &cell : cells) {
    minCol = std::min(minCol, cell.second);
    maxCol = std::max(maxCol, cell.second);
  }
  const int64_t boundingRows = maxRow - minRow + 1;
  const int64_t boundingCols = maxCol - minCol + 1;
  if (cells.size() != static_cast<size_t>(*replicaCount) ||
      cells.size() != static_cast<size_t>(boundingRows * boundingCols) ||
      minRow != *replicaRow || minCol != *replicaCol ||
      boundingRows != *replicaRows || boundingCols != *replicaCols ||
      commonStart != *schedulerStart || commonEnd != *schedulerEnd ||
      commonDuration != commonEnd - commonStart)
    return fail("actual cells do not exactly fill the recorded rectangle or "
                "agree with task-level original times");

  auto orchestration =
      task->getAttrOfType<DictionaryAttr>("task_orchestration_info");
  auto orchestrationCells =
      orchestration ? orchestration.getAs<ArrayAttr>("cgra_positions")
                    : ArrayAttr();
  if (!orchestrationCells || orchestrationCells.size() != cells.size())
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

  trace.placement.task = taskIndex;
  trace.placement.row = static_cast<int>(minRow);
  trace.placement.col = static_cast<int>(minCol);
  trace.placement.rows = static_cast<int>(boundingRows);
  trace.placement.cols = static_cast<int>(boundingCols);
  trace.placement.selectedRows = static_cast<int>(selectedRows);
  trace.placement.selectedCols = static_cast<int>(selectedCols);
  trace.placement.activeReplicas = static_cast<unsigned>(*activeReplicas);
  for (const auto &cell : cells) {
    trace.placement.occupiedCells.push_back(
        {static_cast<int>(cell.first), static_cast<int>(cell.second), 0});
    trace.contextIds.push_back(contexts[cell]);
  }
  trace.selectedShape = selectedShapeAttr.getValue().str();
  trace.actualShape = actualShapeAttr->str();
  trace.selectedCount = *infoSelectedCount;
  trace.profileDuration = *profileDuration;
  trace.compiledII = *compiledII;
  trace.materializedOperationCount = *materializedOperations;
  trace.sampleTripCount = *sampleTripCount;
  trace.steps = *steps;
  trace.schedulerStart = commonStart;
  trace.schedulerEnd = commonEnd;
  trace.schedulerDuration = commonDuration;
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
  Option<std::string> outputFile{*this, "output",
                                 llvm::cl::desc("Atomic JSON result path."),
                                 llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
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
      graphId =
          StringAttr::get(module.getContext(), kOriginalBaselineGraphVariantId);
      function->setAttr("amoeba.graph_variant_id", graphId);
    }
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
    if (!graphId || graphId.getValue().empty() || !strategyName ||
        *strategyName != "throughput-guided" || !pipelineInterval ||
        *pipelineInterval < 1 || !internalTimeUnit ||
        internalTimeUnit.getValue() != "scaled-internal-placement-slots" ||
        !internalTimeScale || internalTimeScale.getInt() < 1 ||
        !functionDispatch || !functionSchedule) {
      function.emitError() << "original throughput-guided function trace is "
                              "missing required dispatch, time-unit, "
                              "pipeline, or schedule attrs";
      return signalPassFailure();
    }

    FailureOr<llvm::SmallVector<TaskMetadata>> collected =
        collectAnalyticalTaskMetadata(function, error);
    if (failed(collected) || collected->empty()) {
      function.emitError() << (error.empty() ? "no static Taskflow tasks"
                                             : error);
      return signalPassFailure();
    }
    llvm::SmallVector<TaskMetadata> tasks = std::move(*collected);
    if (functionDispatch.size() != tasks.size() ||
        functionSchedule.size() != tasks.size()) {
      function.emitError() << "original dispatch and schedule trace must "
                              "cover every Taskflow task exactly once";
      return signalPassFailure();
    }

    llvm::StringMap<unsigned> taskIndices;
    std::vector<Operation *> taskOperations;
    std::vector<OriginalAmoebaRetimerTask> retimerTasks(tasks.size());
    taskOperations.reserve(tasks.size());
    for (unsigned index = 0; index < tasks.size(); ++index) {
      const TaskMetadata &task = tasks[index];
      if (task.name.empty() ||
          !taskIndices.try_emplace(task.name, index).second) {
        function.emitError()
            << "original task names must be unique and nonempty";
        return signalPassFailure();
      }
      if (task.tripCount < 1) {
        TaskflowTaskOp taskOperation = task.op;
        taskOperation.emitError()
            << "compiler-inferred actual trip count is invalid";
        return signalPassFailure();
      }
      retimerTasks[index].name = task.name;
      TaskflowTaskOp taskOperation = task.op;
      taskOperations.push_back(taskOperation.getOperation());
    }

    std::vector<unsigned> dispatchOrder;
    std::vector<std::string> dispatchNames;
    llvm::StringSet<> seenDispatch;
    dispatchOrder.reserve(tasks.size());
    dispatchNames.reserve(tasks.size());
    for (Attribute attribute : functionDispatch) {
      auto name = dyn_cast<StringAttr>(attribute);
      auto task = name ? taskIndices.find(name.getValue()) : taskIndices.end();
      if (!name || task == taskIndices.end() ||
          !seenDispatch.insert(name.getValue()).second) {
        function.emitError() << "original dispatch order is not a unique "
                                "permutation of current task names";
        return signalPassFailure();
      }
      dispatchOrder.push_back(task->second);
      dispatchNames.push_back(name.getValue().str());
    }

    std::map<std::string, DictionaryAttr> scheduleInfoByTask;
    for (Attribute attribute : functionSchedule) {
      auto record = dyn_cast<DictionaryAttr>(attribute);
      auto name = record ? getString(record, "task_name") : std::nullopt;
      auto index = record ? getInteger(record, "dispatch_index") : std::nullopt;
      auto task = name ? taskIndices.find(*name) : taskIndices.end();
      if (!record || !name || task == taskIndices.end() || !index ||
          *index < 0 || *index >= static_cast<int64_t>(tasks.size()) ||
          dispatchNames[*index] != *name ||
          !scheduleInfoByTask.emplace(name->str(), record).second) {
        function.emitError() << "function-level task schedule records are "
                                "malformed or inconsistent with dispatch";
        return signalPassFailure();
      }
    }
    if (scheduleInfoByTask.size() != tasks.size()) {
      function.emitError() << "function-level scheduler trace omits a task";
      return signalPassFailure();
    }

    std::vector<OriginalTraceTask> traces(tasks.size());
    std::vector<OriginalAmoebaFixedPlacement> placements;
    placements.reserve(tasks.size());
    for (unsigned index = 0; index < tasks.size(); ++index) {
      auto found = scheduleInfoByTask.find(tasks[index].name);
      auto taskInfo = tasks[index].op->getAttrOfType<DictionaryAttr>(
          kOriginalScheduleInfoAttr);
      if (found == scheduleInfoByTask.end() || !taskInfo ||
          found->second != taskInfo) {
        tasks[index].op.emitError()
            << "task-level and function-level original schedule records differ";
        return signalPassFailure();
      }
      auto dispatchIndex = getInteger(taskInfo, "dispatch_index");
      if (!dispatchIndex || *dispatchIndex < 0 ||
          *dispatchIndex >= static_cast<int64_t>(tasks.size()) ||
          dispatchOrder[*dispatchIndex] != index) {
        tasks[index].op.emitError()
            << "task schedule dispatch index disagrees with function order";
        return signalPassFailure();
      }
      if (!readTaskTrace(tasks[index].op, index,
                         static_cast<unsigned>(*dispatchIndex), traces[index],
                         error)) {
        tasks[index].op.emitError() << error;
        return signalPassFailure();
      }
      if (traces[index].sampleTripCount != tasks[index].tripCount) {
        tasks[index].op.emitError()
            << "profile sample_trip_count does not match compiler-inferred "
               "static actual trip count";
        return signalPassFailure();
      }
      placements.push_back(traces[index].placement);
    }

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
            parentCostFile.getValue(), bodyExportFile.getValue(), function,
            graphId.getValue(), tasks, traces, perCgraRows, perCgraCols,
            bodyExport, error)) {
      function.emitError() << error;
      return signalPassFailure();
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
    if (!costs.load(parentCostFile.getValue(), function.getSymName(), tasks,
                    sourceRepository, sourceCommit, catalogArchitecturePath,
                    graphId.getValue(), error)) {
      function.emitError() << error;
      return signalPassFailure();
    }

    std::vector<int64_t> durations(tasks.size(), 0);
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
      TaskShapeChoice choice{tasks[index].name, tasks[index].tripCount,
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
      durations[index] = *duration;
      startupCycles[index] = cost->startupCycles;
      llvm::json::Object item;
      item["task"] = tasks[index].name;
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
      item["selected_profile_shape"] = trace.selectedShape;
      item["selected_mapper_tile_rows"] = mapperRows;
      item["selected_mapper_tile_cols"] = mapperCols;
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
            gridRows, gridCols, retimerTasks, durations, placements,
            dispatchOrder, communication, retimed)) {
      function.emitError() << (retimed.rejection.empty()
                                   ? "original fixed-decision retiming failed"
                                   : retimed.rejection);
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
      auto body = bodyExport.tasks.find(task.name);
      if (body == bodyExport.tasks.end()) {
        task.op.emitError() << "profile body export omits current task";
        return signalPassFailure();
      }
      llvm::json::Object profileBinding;
      profileBinding["task"] = task.name;
      profileBinding["static_trip_count"] = task.tripCount;
      profileBinding["task_signature"] = body->second.taskSignature;
      profileBinding["counter_signature"] = body->second.counterSignature;
      profileBinding["kernel_binding_signature"] =
          body->second.kernelBindingSignature;
      profileBinding["normalized_mapper_body"] =
          body->second.normalizedMapperBody;
      profileBinding["selected_profile_shape"] = trace.selectedShape;
      profileBinding["selected_cgra_count"] = trace.selectedCount;
      profileBinding["compiled_ii"] = trace.compiledII;
      profileBinding["steps"] = trace.steps;
      profileBinding["sample_trip_count"] = trace.sampleTripCount;
      profileBinding["materialized_operation_count"] =
          trace.materializedOperationCount;
      profileBinding["profile_duration_cycles"] = trace.profileDuration;
      profileBinding["catalog_startup_cycles"] = startupCycles[index];
      profileBinding["mapped_duration_cycles"] = durations[index];
      profileBinding["mapper_succeeded"] = true;
      profileBinding["current_ir_body_equivalence_verified"] = true;
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
        cellJson["replica_id"] = static_cast<int64_t>(cell.replicaId);
        cellJson["context_id"] = traces[placement.task].contextIds[cellIndex];
        occupiedCellsJson.push_back(std::move(cellJson));
      }
      item["occupied_cells"] = std::move(occupiedCellsJson);
      scheduleJson.push_back(std::move(item));
    }
    llvm::json::Array originalDecisionsJson;
    for (unsigned index = 0; index < tasks.size(); ++index) {
      const OriginalTraceTask &trace = traces[index];
      llvm::json::Object item;
      item["task"] = tasks[index].name;
      item["selected_profile_shape"] = trace.selectedShape;
      item["actual_placed_shape"] = trace.actualShape;
      item["actual_placed_row"] = trace.placement.row;
      item["actual_placed_col"] = trace.placement.col;
      item["active_replicas"] =
          static_cast<int64_t>(trace.placement.activeReplicas);
      item["selected_cgra_count"] = trace.selectedCount;
      item["dispatch_index"] = static_cast<int64_t>(
          std::find(dispatchOrder.begin(), dispatchOrder.end(), index) -
          dispatchOrder.begin());
      llvm::json::Array cells;
      for (auto [cellIndex, cell] :
           llvm::enumerate(trace.placement.occupiedCells)) {
        llvm::json::Object entry;
        entry["row"] = cell.row;
        entry["col"] = cell.col;
        entry["replica_id"] = static_cast<int64_t>(cell.replicaId);
        entry["context_id"] = trace.contextIds[cellIndex];
        cells.push_back(std::move(entry));
      }
      item["actual_cells"] = std::move(cells);
      item["context_ids"] = integerArray(trace.contextIds);
      item["original_scheduler_start_internal"] = trace.schedulerStart;
      item["original_scheduler_end_internal"] = trace.schedulerEnd;
      item["original_scheduler_duration_internal"] = trace.schedulerDuration;
      item["original_profile_duration_cycles"] = trace.profileDuration;
      originalDecisionsJson.push_back(std::move(item));
    }

    llvm::json::Object result;
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
    result["iteration_domain_coverage_status"] = "pending";
    result["iteration_domain_coverage_verified"] = false;
    result["iteration_domain_coverage_evidence"] =
        "body/profile trip-count agreement does not certify internal mapper "
        "loop coverage; the explicit source iteration-domain guard has not "
        "run";
    result["selected_profile_body_binding_schema"] =
        kOriginalAmoebaProfileBindingSchema.str();
    result["selected_profile_body_binding_verified"] = true;
    result["whole_program_result_scope"] =
        "retime-of-supplied-original-decision-trace; not-certified-as-"
        "original-full-flow";
    result["required_external_evidence"] = llvm::json::Array{
        "mapper-profiles-from-equivalent-correct-original-flow",
        "source-iteration-domain-coverage"};
    result["mapper_success_evidence"] =
        "source-inferred-success-only-profile-info; all-one-fallback-rejected";
    result["explicit_mapper_succeeded_attribute_present"] = false;
    result["duration_source"] =
        "profile_info.compiled_ii-times-inferred-trip-count-minus-one-plus-"
        "bound-cost-catalog-startup";
    result["score_source"] =
        "preserved-original-amoeba-decisions-plus-explicit-production-network";
    result["mapped_whole_program_cycles"] = retimed.replay.predictedMakespan;
    result["predicted_whole_program_cycles"] = retimed.replay.predictedMakespan;
    result["replayed_communication_edges"] =
        static_cast<int64_t>(retimed.replay.replayedEdges);
    result["original_pipeline_interval"] = *pipelineInterval;
    result["original_pipeline_interval_source"] =
        "task_orchestration_summary.pipeline_interval; separate metadata";
    result["original_internal_time_unit"] = internalTimeUnit.getValue().str();
    result["original_internal_time_scale"] = internalTimeScale.getInt();
    result["grid_rows"] = gridRows;
    result["grid_cols"] = gridCols;
    result["max_cgras_per_task"] = 4;
    result["cost_catalog_namespace"] = costs.nameSpace().str();
    result["cost_catalog_source_repository"] = costs.sourceRepository().str();
    result["cost_catalog_model_status"] = catalogModelStatus;
    result["cost_catalog_production_ready"] = catalogProductionReady;
    result["cost_catalog_architecture_path"] = costs.architecturePath().str();
    result["original_decisions"] = std::move(originalDecisionsJson);
    result["task_costs"] = std::move(taskCostsJson);
    result["dispatch_order"] = std::move(dispatchJson);
    result["task_schedule"] = std::move(scheduleJson);
    llvm::json::Object fixedDecisionTrace;
    fixedDecisionTrace["schema"] = "amoeba-fixed-decision-trace-v1";
    fixedDecisionTrace["candidate_id"] = "candidate-0";
    fixedDecisionTrace["graph_variant_id"] = graphId.getValue().str();
    fixedDecisionTrace["formal_go"] = false;
    fixedDecisionTrace["iteration_domain_coverage_status"] = "pending";
    fixedDecisionTrace["iteration_domain_coverage_verified"] = false;
    fixedDecisionTrace["body_equivalence_checked"] = true;
    fixedDecisionTrace["body_equivalence_status"] =
        "current-ir-exactly-matches-profile-body-export";
    fixedDecisionTrace["communication_contract"] =
        std::move(communicationContractJson);
    fixedDecisionTrace["task_profile_bindings"] =
        std::move(taskProfileBindingsJson);
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
        cellJson["replica_id"] = static_cast<int64_t>(cell.replicaId);
        cellJson["context_id"] = trace.contextIds[cellIndex];
        cells.push_back(std::move(cellJson));
      }
      schedule["occupied_cells"] = std::move(cells);
      traceScheduleJson.push_back(std::move(schedule));
    }
    fixedDecisionTrace["task_schedule"] = std::move(traceScheduleJson);
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
