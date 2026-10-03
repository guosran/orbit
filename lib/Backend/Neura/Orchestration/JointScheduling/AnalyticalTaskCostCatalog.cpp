//===- AnalyticalTaskCostCatalog.cpp - Validated ML cost oracle ----------===//
//
// Implements the no-file-hash task-shape cost catalogue contract.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCostCatalog.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <cmath>
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
                              std::string &error) {
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
          *ii < *lowerBound) {
        error = "supported cost requires positive finite predicted_ii and "
                "startup_cycles and analytical_lower_bound, and "
                "predicted_ii >= analytical_lower_bound";
        return false;
      }
      cost = {*ii, *startup, true};
    } else if (*status != "unsupported") {
      error = "support_status must be supported or unsupported";
      return false;
    } else if (entry->get("predicted_ii") || entry->get("startup_cycles") ||
               entry->get("analytical_lower_bound")) {
      error = "unsupported cost entries must be censored and contain no "
              "numeric timing prediction";
      return false;
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
