//===- AnalyticalTaskCostCatalog.cpp - Validated ML cost oracle ----------===//
//
// Implements the no-file-hash task-shape cost catalogue contract.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCostCatalog.h"
#include "AnalyticalMLPInference.h"

#include "NeuraDialect/Architecture/Architecture.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <cmath>
#include <array>
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <utility>

using namespace mlir;

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

static SmallString<256> normalizedPath(StringRef path) {
  SmallString<256> result(path);
  if (std::error_code ec = llvm::sys::fs::make_absolute(result))
    return SmallString<256>(path);
  llvm::sys::path::remove_dots(result, /*remove_dot_dot=*/true);
  return result;
}

bool samePath(StringRef lhs, StringRef rhs) {
  bool equivalent = false;
  if (!llvm::sys::fs::equivalent(lhs, rhs, equivalent) && equivalent)
    return true;
  return normalizedPath(lhs) == normalizedPath(rhs);
}

static std::optional<StringRef> requiredString(const llvm::json::Object &object,
                                               StringRef key,
                                               std::string &error) {
  std::optional<StringRef> value = object.getString(key);
  if (!value)
    error = "missing or invalid string field \"" + key.str() + "\"";
  return value;
}

static std::optional<int64_t> requiredInteger(const llvm::json::Object &object,
                                              StringRef key,
                                              std::string &error) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value)
    error = "missing or invalid integer field \"" + key.str() + "\"";
  return value;
}

// JSON integers are limited to signed 64 bits by LLVM's JSON API. Factored
// spaces can have a Cartesian cardinality much larger than that, so accept a
// canonical positive decimal string as the lossless representation while
// retaining legacy positive integer catalogues.
static std::optional<std::string>
requiredPositiveDecimal(const llvm::json::Object &object, StringRef key,
                        std::string &error) {
  if (std::optional<int64_t> numeric = object.getInteger(key)) {
    if (*numeric <= 0) {
      error = "missing or invalid positive decimal field \"" + key.str() +
              "\"";
      return std::nullopt;
    }
    return std::to_string(*numeric);
  }
  std::optional<StringRef> text = object.getString(key);
  if (!text || text->empty() ||
      llvm::any_of(*text, [](char value) {
        return value < '0' || value > '9';
      })) {
    error = "missing or invalid positive decimal field \"" + key.str() +
            "\"";
    return std::nullopt;
  }
  StringRef normalized = text->ltrim('0');
  if (normalized.empty()) {
    error = "missing or invalid positive decimal field \"" + key.str() +
            "\"";
    return std::nullopt;
  }
  return normalized.str();
}

static bool isGitCommit(StringRef value) {
  if (value.size() < 7 || value.size() > 64)
    return false;
  return llvm::all_of(value, [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f') ||
           (character >= 'A' && character <= 'F');
  });
}

static bool closeModelValue(double lhs, double rhs) {
  return std::isfinite(lhs) && std::isfinite(rhs) &&
         std::abs(lhs - rhs) <= 1.0e-6 * std::max(1.0, std::abs(lhs));
}

static constexpr std::array<int64_t, 4> kDirectModelMemberSeeds = {
    {17, 41, 113, 239}};

static bool getModelDomainShapes(
    bool directPerCgra2x2,
    std::array<std::pair<int64_t, int64_t>, 8> &shapes,
    std::string &error) {
  if (!directPerCgra2x2) {
    shapes = kFormalMax4CostShapes;
    return true;
  }

  const ::mlir::neura::Architecture &architecture =
      ::mlir::neura::getArchitecture();
  const int64_t perCgraRows = architecture.getPerCgraRows();
  const int64_t perCgraCols = architecture.getPerCgraColumns();
  if (perCgraRows != 2 || perCgraCols != 2 ||
      architecture.getMultiCgraRows() < 4 ||
      architecture.getMultiCgraColumns() < 4) {
    error = "direct 2x2-per-CGRA model requires the bound architecture to "
            "provide a 2x2 PE core and at least a 4x4 CGRA grid";
    return false;
  }

  // The direct model schema names a maximum of four physical CGRAs per task.
  // Keep the protocol's oriented rectangle sequence in CGRA coordinates and
  // derive mapper-tile dimensions from the actual loaded architecture.
  constexpr std::array<std::pair<int64_t, int64_t>, 8> kMaxFourCgraShapes =
      {{{1, 1}, {1, 2}, {2, 1}, {1, 3}, {3, 1}, {1, 4}, {2, 2}, {4, 1}}};
  for (auto [index, shape] : llvm::enumerate(kMaxFourCgraShapes))
    shapes[index] = {shape.first * perCgraRows,
                     shape.second * perCgraCols};
  return true;
}

// Loads one complete external predictor catalogue and memoizes lookups by the
// source task ID and oriented mapper shape. The catalogue is bound through
// explicit Git, architecture, graph, task, and candidate identifiers; no
// file-content hash is part of this contract.
bool TaskShapeCostCache::load(StringRef path, StringRef expectedFunction,
                              ArrayRef<TaskMetadata> expectedTasks,
                              StringRef expectedSourceRepository,
                              StringRef expectedSourceCommit,
                              StringRef expectedArchitecturePath,
                              StringRef expectedGraphId,
                              std::string &error,
                              bool allowModelDomainUnsupported,
                              StringRef expectedCanonicalModuleWitness) {
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
    error =
        "invalid cost catalogue JSON: " + llvm::toString(parsed.takeError());
    return false;
  }
  llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    error = "cost catalogue must be a JSON object";
    return false;
  }
  auto schema = requiredString(*root, "schema", error);
  auto function = requiredString(*root, "function", error);
  auto modelNamespace = requiredString(*root, "namespace", error);
  llvm::json::Object *metadata = root->getObject("predictor_metadata");
  llvm::json::Array *entries = root->getArray("entries");
  if (!schema || !function || !modelNamespace || !metadata || !entries)
    return false;
  if (*schema != kCostSchema || *function != expectedFunction ||
      modelNamespace->empty()) {
    error = "cost catalogue schema/function/namespace mismatch";
    return false;
  }

  const bool directPerCgra2x2 =
      *modelNamespace == kPerCgra2x2ModelNamespace;
  const std::optional<StringRef> catalogModelSchema =
      metadata->getString("model_schema");
  if (directPerCgra2x2 ||
      (catalogModelSchema && *catalogModelSchema == kPerCgra2x2EnsembleSchema)) {
    std::optional<StringRef> featureContractId =
        metadata->getString("feature_contract_id");
    std::optional<StringRef> shapeProtocolId =
        metadata->getString("shape_protocol_id");
    if (!directPerCgra2x2 || !catalogModelSchema ||
        *catalogModelSchema != kPerCgra2x2EnsembleSchema ||
        !featureContractId ||
        *featureContractId != kPerCgra2x2FeatureContractId ||
        !shapeProtocolId ||
        *shapeProtocolId != kPerCgra2x2ShapeProtocolId ||
        !metadata->getBoolean("candidate_only").value_or(false) ||
        metadata->getBoolean("production_ready").value_or(true) ||
        metadata->getBoolean("amoeba_benchmark_overlap_audit_complete")
            .value_or(true) ||
        metadata->getBoolean("old_4x4_labels_reused").value_or(true) ||
        metadata->getBoolean("supports_whole_program_latency_or_throughput_claim")
            .value_or(true)) {
      error = "direct per-CGRA cost catalogue does not match its explicit "
              "namespace, schema, feature contract, shape protocol, or "
              "candidate-only status";
      return false;
    }
  }

  auto provenanceSchema =
      requiredString(*metadata, "provenance_schema", error);
  auto sourceRepository =
      requiredString(*metadata, "source_repository", error);
  auto sourceCommit = requiredString(*metadata, "source_commit", error);
  auto architecturePath =
      requiredString(*metadata, "architecture_path", error);
  auto architectureSchema =
      requiredString(*metadata, "architecture_schema", error);
  auto sourceGraphId = requiredString(*metadata, "source_graph_id", error);
  auto candidateIdScheme =
      requiredString(*metadata, "candidate_id_scheme", error);
  auto candidateCount =
      requiredPositiveDecimal(*metadata, "candidate_count", error);
  const llvm::json::Array *sourceTaskIds =
      metadata->getArray("source_task_ids");
  llvm::json::Object *rankingPolicy = metadata->getObject("ranking_policy");
  if (!provenanceSchema || !sourceRepository || !sourceCommit ||
      !architecturePath || !architectureSchema || !sourceGraphId ||
      !candidateIdScheme || !candidateCount || !sourceTaskIds ||
      !rankingPolicy)
    return false;

  const llvm::json::Object *diagnostic =
      metadata->getObject("diagnostic_override");
  const bool diagnosticOverride = diagnostic != nullptr;
  double runtimeIICeiling = kFormalMax4ModelCeilingII;
  if (diagnosticOverride) {
    std::string architectureError;
    auto runtimeArchitectureFile = llvm::MemoryBuffer::getFile(*architecturePath);
    if (!directPerCgra2x2 || !runtimeArchitectureFile ||
        !validatePerCgra2x2DiagnosticMetadata(
            *metadata, (*runtimeArchitectureFile)->getBuffer(),
            architectureError)) {
      error = architectureError.empty()
                  ? "direct diagnostic override metadata is missing, forged, "
                    "or differs from the exact runtime architecture file"
                  : architectureError;
      return false;
    }
    runtimeIICeiling = kPerCgra2x2DiagnosticIICeiling;
  } else if (metadata->get("diagnostic_only") || metadata->get("formal")) {
    error = "diagnostic direct-model metadata is missing its override proof";
    return false;
  }
  if (diagnosticOverride && !allowModelDomainUnsupported) {
    error = "diagnostic direct-model catalog requires explicit model-domain "
            "row validation";
    return false;
  }

  if (allowModelDomainUnsupported) {
    std::optional<StringRef> witness =
        metadata->getString("canonical_module_witness");
    std::optional<StringRef> domainPolicy =
        metadata->getString("unsupported_prediction_policy");
    std::optional<double> intervalMax =
        metadata->getNumber("model_interval_max_ii");
    const StringRef requiredPolicy =
        diagnosticOverride
            ? StringRef("analytical-lower-bound-exceeds-diagnostic-runtime-ceiling-v1")
            : StringRef("analytical-lower-bound-exceeds-model-ceiling-v1");
    if (!witness || witness->empty() || expectedCanonicalModuleWitness.empty() ||
        *witness != expectedCanonicalModuleWitness || !domainPolicy ||
        *domainPolicy != requiredPolicy ||
        !intervalMax || !std::isfinite(*intervalMax) ||
        *intervalMax != kFormalMax4ModelCeilingII) {
      error = "cost catalogue model-domain proof is not bound to the exact "
              "current canonical module and formal interval";
      return false;
    }
  }

  if (*provenanceSchema != kCostProvenanceSchema ||
      sourceRepository->empty() || !isGitCommit(*sourceCommit) ||
      expectedSourceRepository.empty() || expectedSourceCommit.empty() ||
      !isGitCommit(expectedSourceCommit) ||
      *sourceRepository != expectedSourceRepository ||
      *sourceCommit != expectedSourceCommit || expectedArchitecturePath.empty() ||
      architecturePath->empty() ||
      !samePath(*architecturePath, expectedArchitecturePath) ||
      *architectureSchema != kArchitectureSchema ||
      *candidateIdScheme != kCandidateIdScheme ||
      *sourceGraphId != expectedGraphId) {
    error = "cost catalogue provenance does not match the requested Git, "
            "architecture, graph, or candidate contract";
    return false;
  }
  if (sourceTaskIds->size() != expectedTasks.size()) {
    error = "cost catalogue source task IDs do not exactly cover current IR "
            "tasks";
    return false;
  }
  std::set<std::string> sourceTaskSet;
  for (auto [index, value] : llvm::enumerate(*sourceTaskIds)) {
    std::optional<StringRef> taskId = value.getAsString();
    if (!taskId || taskId->empty() || *taskId != expectedTasks[index].name ||
        !sourceTaskSet.insert(taskId->str()).second) {
      error = "cost catalogue source task IDs do not match current IR tasks";
      return false;
    }
  }

  auto objective = requiredString(*rankingPolicy, "objective", error);
  auto probabilityRole =
      requiredString(*rankingPolicy, "mapper_success_probability", error);
  std::optional<bool> usesProbability =
      rankingPolicy->getBoolean("uses_mapper_success_probability");
  if (!objective || *objective != "predicted_scheduler_makespan" ||
      !probabilityRole ||
      (*probabilityRole != "diagnostic_only" &&
       *probabilityRole != "not_predicted") ||
      !usesProbability || *usesProbability) {
    error = "cost catalogue must exclude mapper success probability from "
            "support and ranking";
    return false;
  }

  namespace_ = modelNamespace->str();
  sourceRepository_ = sourceRepository->str();
  sourceCommit_ = sourceCommit->str();
  architecturePath_ = architecturePath->str();
  architectureSchema_ = architectureSchema->str();
  sourceGraphId_ = sourceGraphId->str();
  candidateIdScheme_ = candidateIdScheme->str();
  candidateCountDecimal_ = *candidateCount;
  candidateCount_ = 0;
  candidateCountFitsInt64_ = false;
  uint64_t representableCount = 0;
  if (!StringRef(candidateCountDecimal_).getAsInteger(10, representableCount) &&
      representableCount <=
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    candidateCount_ = representableCount;
    candidateCountFitsInt64_ = true;
  }
  mapperSuccessProbabilityRole_ = probabilityRole->str();
  taskIds_ = std::move(sourceTaskSet);
  catalog_.clear();
  predictionCache_.clear();
  coveredQueries_.clear();
  hits_ = 0;
  misses_ = 0;
  diagnosticOverride_ = diagnosticOverride;
  runtimeIICeiling_ = runtimeIICeiling;

  for (llvm::json::Value &value : *entries) {
    llvm::json::Object *entry = value.getAsObject();
    if (!entry) {
      error = "cost entry is not an object";
      return false;
    }
    auto task = requiredString(*entry, "task", error);
    auto mapperRows = requiredInteger(*entry, "mapper_tile_rows", error);
    auto mapperCols = requiredInteger(*entry, "mapper_tile_cols", error);
    auto status = requiredString(*entry, "support_status", error);
    if (!task || !mapperRows || !mapperCols || !status)
      return false;
    if (taskIds_.find(task->str()) == taskIds_.end()) {
      error = "cost entry names a task outside current IR";
      return false;
    }
    if (*mapperRows <= 0 || *mapperCols <= 0) {
      error = "cost entry mapper tile dimensions must be positive";
      return false;
    }
    if (const llvm::json::Value *rawProbability =
            entry->get("mapper_success_probability")) {
      std::optional<double> probability = rawProbability->getAsNumber();
      if (*probabilityRole != "diagnostic_only" || !probability ||
          !std::isfinite(*probability) || *probability < 0.0 ||
          *probability > 1.0) {
        error = "mapper success probability must be a diagnostic in [0, 1]";
        return false;
      }
    }

    TaskShapeCost cost;
    if (*status == "supported") {
      auto ii = entry->getNumber("predicted_ii");
      auto startup = entry->getNumber("startup_cycles");
      auto lowerBound = entry->getNumber("analytical_lower_bound");
      if (!ii || !startup || !lowerBound || !std::isfinite(*ii) ||
          !std::isfinite(*startup) || !std::isfinite(*lowerBound) ||
          *ii <= 0.0 || *startup <= 0.0 || *lowerBound <= 0.0 ||
          *ii < *lowerBound || *ii > runtimeIICeiling + 1.0e-6) {
        error = "supported cost requires positive finite predicted_ii and "
                "startup_cycles and analytical_lower_bound, and "
                "predicted_ii >= analytical_lower_bound";
        return false;
      }
      cost = {*ii, *startup, true};
      cost.analyticalLowerBound = *lowerBound;
      cost.runtimeIICeiling = runtimeIICeiling;
      if (diagnosticOverride) {
        auto trainingCeiling = entry->getNumber("training_ceiling_ii");
        auto rowRuntimeCeiling = entry->getNumber("runtime_ceiling_ii");
        auto extrapolationStatus = entry->getString("extrapolation_status");
        auto reportedStd = entry->getNumber("predicted_ii_std");
        auto meanSource = entry->getString("ii_mean_source");
        const llvm::json::Array *members =
            entry->getArray("direct_ensemble_members");
        if (!directPerCgra2x2 || !trainingCeiling ||
            *trainingCeiling != kPerCgra2x2TrainingIICeiling ||
            !rowRuntimeCeiling || *rowRuntimeCeiling != runtimeIICeiling ||
            !reportedStd || !std::isfinite(*reportedStd) ||
            *reportedStd < 0.0 || !meanSource ||
            *meanSource != "direct_four_member_arithmetic_mean" || !members ||
            members->size() != kDirectModelMemberSeeds.size()) {
          error = "supported diagnostic cost lacks its exact ceiling and "
                  "four-member metadata";
          return false;
        }
        std::array<double, 4> memberValues{};
        double memberMean = 0.0;
        bool extrapolated = *lowerBound > kPerCgra2x2TrainingIICeiling;
        for (size_t index = 0; index < members->size(); ++index) {
          const llvm::json::Object *member = (*members)[index].getAsObject();
          auto memberIndex = member ? member->getInteger("member_index")
                                    : std::nullopt;
          auto seed = member ? member->getInteger("seed") : std::nullopt;
          auto value = member ? member->getNumber("predicted_ii")
                              : std::nullopt;
          if (!memberIndex || *memberIndex != static_cast<int64_t>(index) ||
              !seed || *seed != kDirectModelMemberSeeds[index] || !value ||
              !std::isfinite(*value) || *value < *lowerBound ||
              *value > runtimeIICeiling + 1.0e-6) {
            error = "diagnostic direct ensemble member is outside its "
                    "validated runtime interval";
            return false;
          }
          memberValues[index] = *value;
          memberMean += *value;
          extrapolated |= *value > kPerCgra2x2TrainingIICeiling;
        }
        memberMean /= memberValues.size();
        double variance = 0.0;
        for (double value : memberValues) {
          double delta = value - memberMean;
          variance += delta * delta;
        }
        variance /= memberValues.size();
        if (!closeModelValue(memberMean, *ii) ||
            !closeModelValue(std::sqrt(variance), *reportedStd) ||
            !extrapolationStatus ||
            *extrapolationStatus !=
                (extrapolated ? "out-of-training-ceiling"
                              : "within-training-ceiling")) {
          error = "diagnostic direct ensemble aggregate or extrapolation "
                  "status disagrees with its four members";
          return false;
        }
      } else if (entry->get("training_ceiling_ii") ||
                 entry->get("runtime_ceiling_ii") ||
                 entry->get("extrapolation_status")) {
        error = "ordinary model cost carries unbound diagnostic metadata";
        return false;
      }
    } else if (*status != "unsupported") {
      error = "support_status must be supported or unsupported";
      return false;
    } else {
      if (entry->get("predicted_ii") || entry->get("startup_cycles") ||
          entry->get("predicted_ii_std") || entry->get("ii_mean_source") ||
          entry->get("model_status") || entry->get("production_ready") ||
          entry->get("mapper_success_probability")) {
        error = "unsupported cost entries must contain no numeric timing "
                "prediction";
        return false;
      }
      if (auto unsupportedStatus = entry->getString("status")) {
        if (!allowModelDomainUnsupported ||
            *unsupportedStatus != kModelDomainUnsupportedStatus) {
          error = "unsupported model-domain row is not permitted by the "
                  "active cost-catalog policy";
          return false;
        }
        auto reason = requiredString(*entry, "unsupported_reason", error);
        auto lowerBound = entry->getNumber("analytical_lower_bound");
        auto intervalMax = entry->getNumber("model_interval_max_ii");
        auto rowRuntimeCeiling = entry->getNumber("runtime_ceiling_ii");
        auto rowTrainingCeiling = entry->getNumber("training_ceiling_ii");
        auto extrapolationStatus = entry->getString("extrapolation_status");
        const StringRef expectedReason =
            diagnosticOverride
                ? StringRef("analytical-lower-bound-exceeds-diagnostic-runtime-ceiling")
                : StringRef(kModelDomainUnsupportedReason);
        if (!reason || *reason != expectedReason ||
            !lowerBound || !std::isfinite(*lowerBound) || *lowerBound <= 0.0 ||
            !intervalMax || !std::isfinite(*intervalMax) ||
            *intervalMax != kFormalMax4ModelCeilingII ||
            *lowerBound <= runtimeIICeiling ||
            (diagnosticOverride &&
             (!rowRuntimeCeiling || *rowRuntimeCeiling != runtimeIICeiling ||
              !rowTrainingCeiling ||
              *rowTrainingCeiling != kPerCgra2x2TrainingIICeiling ||
              !extrapolationStatus ||
              *extrapolationStatus != "outside-diagnostic-runtime-domain")) ||
            (!diagnosticOverride &&
             (rowRuntimeCeiling || rowTrainingCeiling ||
              extrapolationStatus))) {
          error = "unsupported model-domain row lacks a valid analytical "
                  "lower-bound proof above the model ceiling";
          return false;
        }
        cost.supported = false;
        cost.modelDomainUnsupported = true;
        cost.analyticalLowerBound = *lowerBound;
        cost.modelIntervalMaxII = *intervalMax;
        cost.runtimeIICeiling = runtimeIICeiling;
        cost.unsupportedReason = reason->str();
      } else if (allowModelDomainUnsupported) {
        error = "model-domain catalogue unsupported row lacks explicit "
                "unsupported status";
        return false;
      } else if (entry->get("analytical_lower_bound")) {
        error = "unsupported cost entries must not carry an untyped lower "
                "bound";
        return false;
      }
    }
    CostQueryKey queryKey{task->str(), *mapperRows, *mapperCols};
    if (!catalog_.emplace(queryKey, cost).second) {
      error = "duplicate task/mapper-shape cost entry";
      return false;
    }
  }
  if (catalog_.empty()) {
    error = "cost catalogue contains no task-shape entries";
    return false;
  }
  if (allowModelDomainUnsupported || directPerCgra2x2) {
    std::array<std::pair<int64_t, int64_t>, 8> expectedShapes;
    if (!getModelDomainShapes(directPerCgra2x2, expectedShapes, error))
      return false;
    if (catalog_.size() != expectedTasks.size() * expectedShapes.size()) {
      error = directPerCgra2x2
                  ? "direct model-domain catalogue must explicitly cover "
                    "every task and all eight mapper shapes"
                  : "model-domain catalogue must explicitly cover every task "
                    "and all eight formal max-four shapes";
      return false;
    }
    bool sawSupported = false;
    for (const TaskMetadata &task : expectedTasks) {
      for (auto [rows, cols] : expectedShapes) {
        auto found = catalog_.find(CostQueryKey{task.name, rows, cols});
        if (found == catalog_.end()) {
          error = directPerCgra2x2
                      ? "direct model-domain catalogue is missing task=" +
                            task.name + " shape=rect-" +
                            std::to_string(rows) + "x" +
                            std::to_string(cols)
                      : "model-domain catalogue is missing task=" +
                            task.name + " shape=rect-" +
                            std::to_string(rows) + "x" +
                            std::to_string(cols);
          return false;
        }
        sawSupported |= found->second.supported;
      }
    }
    if (!sawSupported) {
      error = "model-domain catalogue contains no supported task-shape query";
      return false;
    }
  }
  return true;
}

const TaskShapeCost *TaskShapeCostCache::get(const TaskShapeChoice &choice,
                                             std::string &error) {
  CostQueryKey queryKey{choice.task, choice.shape.mapperRows,
                        choice.shape.mapperCols};
  auto found = catalog_.find(queryKey);
  if (found == catalog_.end()) {
    error = "missing cost for task=" + choice.task + ", mapper_shape=rect-" +
            std::to_string(choice.shape.mapperRows) + "x" +
            std::to_string(choice.shape.mapperCols);
    return nullptr;
  }
  coveredQueries_.insert(queryKey);
  if (taskIds_.find(choice.task) == taskIds_.end()) {
    error = "cost lookup names a task outside current IR";
    return nullptr;
  }
  // Trip count is absent from this key on purpose. The cached value is the
  // predictor's per-iteration timing pair; the caller computes duration with
  // the current task's independently validated trip count.
  PredictionCacheKey stableKey{choice.task, choice.shape.mapperRows,
                               choice.shape.mapperCols};
  auto cached = predictionCache_.find(stableKey);
  if (cached != predictionCache_.end()) {
    ++hits_;
    return &cached->second;
  }
  ++misses_;
  auto inserted = predictionCache_.emplace(stableKey, found->second);
  return &inserted.first->second;
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir
