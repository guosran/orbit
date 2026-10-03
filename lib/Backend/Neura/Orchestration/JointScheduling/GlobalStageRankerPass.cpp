//===- GlobalStageRankerPass.cpp -----------------------------------------===//
//
// C++/MLIR-only global top-five selection for one ORBIT ablation stage.
//
// The pass consumes a small JSON manifest that binds source-owned exact score
// JSONL files to one source-owned graph-closure JSONL file.  It deliberately
// does no prediction and never treats an incomplete graph or score artifact as
// a certified stage.  A failed validation still writes a bounded diagnostic
// shortlist when score rows were readable; the pass then fails.
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "Backend/Neura/GlobalStageRankerPass.h"
#include "Backend/Neura/Orchestration/JointScheduling/JointSchedulingCandidateOrder.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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
namespace json = llvm::json;

namespace {

constexpr llvm::StringLiteral kInputSchema =
    "orbit-global-stage-ranker-input-v1";
constexpr llvm::StringLiteral kOutputSchema = "orbit-global-stage-top5-v1";
constexpr llvm::StringLiteral kScoreSchema =
    "amoeba-exact-joint-task-scores-v1";
constexpr llvm::StringLiteral kClosureSchema = "orbit-joint-graph-closure-v1";
constexpr llvm::StringLiteral kPredictorSource = "ml-ii-startup-predictor";
constexpr llvm::StringLiteral kProductionTemporalModel =
    "production-scheduler";
constexpr llvm::StringLiteral kProductionScheduleSpace =
    "production-scheduler";
constexpr llvm::StringLiteral kProductionTemporalScope =
    "production-task-scheduler-only";
constexpr llvm::StringLiteral kProductionStartPolicy =
    "production-spatial-temporal-scheduler";
constexpr llvm::StringLiteral kProductionSchedulerBackend =
    "orchestrate-tasks-on-accelerators";
constexpr llvm::StringLiteral kProductionCertificate =
    "proven-shape-space-under-production-scheduler";

struct VariantSpec {
  std::string graphVariantId;
  std::string semanticGraphId;
  std::string scorePath;
  std::string mapperReplayPath;
  std::string stage;
  std::vector<std::string> rewritePaths;
};

struct InputManifest {
  std::string sourceRepository;
  std::string sourceCommit;
  std::string architecturePath;
  std::string architectureId;
  std::string function;
  std::string stage;
  std::string closurePath;
  std::vector<VariantSpec> variants;
};

struct ScoreCandidate {
  std::string candidateId;
  int64_t cycles = 0;
  std::string scorePath;
  int64_t sourceLine = 0;
  json::Object scoreRecord;
};

struct EvidenceRow {
  int64_t cycles = 0;
  std::string semanticGraphId;
  std::string candidateId;
  std::string graphVariantId;
  std::string scorePath;
  json::Object scoreRecord;
  bool certified = false;
  bool feasible = false;
  bool seedOnly = false;
};

struct ScoreArtifact {
  std::optional<json::Object> header;
  std::optional<json::Object> footer;
  std::vector<json::Object> rows;
};

struct ScoreInfo {
  bool valid = false;
  bool seedOnly = false;
  bool globalCutoffComplete = false;
  bool placementCoverageLimited = false;
  bool productionScheduler = false;
  bool globalShapeLocationTemporalSpace = true;
  bool locationCoverageLimited = false;
  bool temporalCoverageLimited = false;
  bool orderDerivedTemporal = false;
  bool temporalContractPresent = false;
  std::string temporalModel;
  std::string scheduleSpace;
  std::string dispatchPolicy;
  std::string dispatchPolicyScope;
  std::string schedulerBackend;
  std::string temporalSearchScope;
  std::string startPolicy;
  std::optional<bool> cycleStartsAreSearchDimensions;
  std::string candidateCount;
  std::string shapeCandidatesVisited;
  std::string boundPrunedShapeCandidates;
  std::string unmeasuredShapeCandidates;
  std::string rankingCertificate;
  std::string costNamespace;
  std::string architectureSchema;
  std::string predictorSource;
  std::string globalCutoffEvidencePath;
  std::string globalCutoffCycles;
  std::vector<ScoreCandidate> candidates;
  std::vector<std::string> errors;
};

struct ClosureGraph {
  std::string path;
  std::string structuralText;
  std::string semanticGraphId;
  std::string graphVariantId;
  bool deduplicated = false;
  std::string firstPath;
  std::string mlirPath;
  std::string factsPath;
};

struct ClosureInfo {
  bool valid = false;
  bool scopeComplete = false;
  std::string headerStage;
  std::vector<ClosureGraph> graphs;
  std::set<std::string> structuralKeys;
  std::vector<std::string> errors;
};

struct RankedCandidate {
  std::string semanticGraphId;
  std::string graphVariantId;
  std::string candidateId;
  int64_t cycles = 0;
  std::string scorePath;
  int64_t sourceLine = 0;
  json::Object scoreRecord;
  std::string mapperReplayPath;
  std::vector<std::string> graphVariantIds;
  std::vector<std::string> scorePaths;
  std::vector<std::string> rewritePaths;
  std::vector<std::string> mapperReplayPaths;
};

struct RankResult {
  bool complete = false;
  bool seedOnly = false;
  InputManifest input;
  ClosureInfo closure;
  std::vector<ScoreInfo> scores;
  std::vector<RankedCandidate> candidates;
  std::vector<std::string> errors;
  std::string unmeasuredCandidateCount = "0";
  std::string scoredCandidateCount = "0";
  std::string visitedShapeCandidateCount = "0";
  std::string boundPrunedShapeCandidateCount = "0";
  std::string unmeasuredGraphVariantCount = "0";
  unsigned diagnosticLimit = 5;
  bool placementCoverageLimited = false;
  bool productionScheduler = false;
  bool globalShapeLocationTemporalSpace = true;
  bool locationCoverageLimited = false;
  bool temporalCoverageLimited = false;
  std::string globallyPrunedGraphVariantCount = "0";
  bool orderDerivedTemporal = false;
  std::string temporalModel;
  std::string scheduleSpace;
  std::string dispatchPolicy;
  std::string dispatchPolicyScope;
  std::string schedulerBackend;
  bool temporalModelSeen = false;
  std::string temporalSearchScope;
  std::string startPolicy;
  std::optional<bool> cycleStartsAreSearchDimensions;
  std::string costNamespace;
  std::string architectureSchema;
};

static void addError(std::vector<std::string> &errors, StringRef error,
                     unsigned limit = 32) {
  if (errors.size() < limit)
    errors.push_back(error.str());
}

static std::string jsonText(const json::Value &value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << value;
  stream.flush();
  return text;
}

static std::string jsonLine(json::Object object) {
  return jsonText(json::Value(std::move(object))) + "\n";
}

static std::string resolvePath(StringRef manifestPath, StringRef rawPath) {
  llvm::SmallString<256> resolved;
  if (llvm::sys::path::is_absolute(rawPath)) {
    resolved = rawPath;
  } else {
    resolved = manifestPath;
    llvm::sys::path::remove_filename(resolved);
    llvm::sys::path::append(resolved, rawPath);
  }
  if (std::error_code ec = llvm::sys::fs::make_absolute(resolved))
    return rawPath.str();
  llvm::sys::path::remove_dots(resolved, /*remove_dot_dot=*/true);
  return resolved.str().str();
}

static std::string normalizedPath(StringRef path) {
  llvm::SmallString<256> normalized(path);
  if (std::error_code ec = llvm::sys::fs::make_absolute(normalized))
    return path.str();
  llvm::sys::path::remove_dots(normalized, /*remove_dot_dot=*/true);
  return normalized.str().str();
}

static std::optional<std::string> stringField(const json::Object &object,
                                              StringRef key) {
  if (std::optional<StringRef> value = object.getString(key))
    return value->str();
  return std::nullopt;
}

static std::optional<std::string>
stringField(const json::Object &object, StringRef flatKey, StringRef nestedKey,
            StringRef nestedObject) {
  if (std::optional<std::string> value = stringField(object, flatKey))
    return value;
  const json::Object *nested = object.getObject(nestedObject);
  return nested ? stringField(*nested, nestedKey) : std::nullopt;
}

static bool requireString(const json::Object &object, StringRef key,
                          std::string &destination,
                          std::vector<std::string> &errors,
                          StringRef label = {}) {
  std::optional<std::string> value = stringField(object, key);
  if (!value || value->empty()) {
    addError(errors, (label.empty() ? key : label).str() + " is missing");
    return false;
  }
  destination = std::move(*value);
  return true;
}

static bool requireStringAny(const json::Object &object, StringRef flatKey,
                             StringRef nestedObject, StringRef nestedKey,
                             std::string &destination,
                             std::vector<std::string> &errors,
                             StringRef label) {
  std::optional<std::string> value =
      stringField(object, flatKey, nestedKey, nestedObject);
  if (!value || value->empty()) {
    addError(errors, label.str() + " is missing");
    return false;
  }
  destination = std::move(*value);
  return true;
}

static std::optional<std::string>
decimalField(const json::Object &object, StringRef key) {
  if (std::optional<int64_t> value = object.getInteger(key)) {
    if (*value < 0)
      return std::nullopt;
    return std::to_string(*value);
  }
  std::optional<StringRef> text = object.getString(key);
  if (!text || text->empty() ||
      llvm::any_of(*text, [](char value) {
        return value < '0' || value > '9';
      }))
    return std::nullopt;
  StringRef normalized = text->ltrim('0');
  return (normalized.empty() ? StringRef("0") : normalized).str();
}

static bool isZero(StringRef decimal) {
  return decimal.empty() || llvm::all_of(decimal, [](char value) {
           return value == '0';
         });
}

static int compareDecimal(StringRef lhs, StringRef rhs) {
  if (lhs.size() != rhs.size())
    return lhs.size() < rhs.size() ? -1 : 1;
  if (lhs == rhs)
    return 0;
  return lhs < rhs ? -1 : 1;
}

static std::string addDecimal(StringRef lhs, StringRef rhs) {
  int left = static_cast<int>(lhs.size()) - 1;
  int right = static_cast<int>(rhs.size()) - 1;
  int carry = 0;
  std::string result;
  while (left >= 0 || right >= 0 || carry != 0) {
    int value = carry;
    if (left >= 0)
      value += lhs[left--] - '0';
    if (right >= 0)
      value += rhs[right--] - '0';
    result.push_back(static_cast<char>('0' + (value % 10)));
    carry = value / 10;
  }
  std::reverse(result.begin(), result.end());
  return result.empty() ? "0" : result;
}

static void putDecimal(json::Object &object, StringRef key,
                       StringRef decimal) {
  uint64_t value = 0;
  if (!decimal.getAsInteger(10, value) &&
      value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    object[key] = static_cast<int64_t>(value);
  else
    object[key] = decimal.str();
}

static void appendUnique(std::vector<std::string> &values, StringRef value) {
  if (!value.empty() &&
      std::find(values.begin(), values.end(), value) == values.end())
    values.push_back(value.str());
}

static void sortUnique(std::vector<std::string> &values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

static std::optional<std::string>
readPathField(const json::Object &object, StringRef primary, StringRef alias) {
  if (std::optional<std::string> value = stringField(object, primary))
    return value;
  return stringField(object, alias);
}

static bool parseManifest(StringRef path, InputManifest &manifest,
                          std::vector<std::string> &errors) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    addError(errors, "cannot read input manifest " + path.str() + ": " +
                         buffer.getError().message());
    return false;
  }
  llvm::Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    addError(errors, "invalid input manifest JSON: " +
                         llvm::toString(parsed.takeError()));
    return false;
  }
  const json::Object *root = parsed->getAsObject();
  if (!root) {
    addError(errors, "input manifest must be one JSON object");
    return false;
  }
  std::optional<StringRef> schema = root->getString("schema");
  bool valid = true;
  if (!schema || *schema != kInputSchema) {
    addError(errors, "input manifest schema is not " + kInputSchema.str());
    valid = false;
  }
  valid &= requireStringAny(*root, "source_repository", "source",
                            "repository", manifest.sourceRepository, errors,
                            "source repository");
  valid &= requireStringAny(*root, "source_commit", "source", "commit",
                            manifest.sourceCommit, errors, "source commit");
  valid &= requireStringAny(*root, "architecture_path", "architecture", "path",
                            manifest.architecturePath, errors,
                            "architecture path");
  // architecture_id is a semantic label supplied by the stage controller. It
  // is optional for compatibility with the source-owned score header, which
  // identifies the architecture by its exact path.
  if (std::optional<std::string> architectureId =
          stringField(*root, "architecture_id", "id", "architecture"))
    manifest.architectureId = std::move(*architectureId);
  valid &= requireString(*root, "function", manifest.function, errors);
  valid &= requireString(*root, "stage", manifest.stage, errors);

  std::optional<std::string> closure =
      readPathField(*root, "graph_closure_record_path", "graph_closure_path");
  if (!closure || closure->empty()) {
    addError(errors, "graph closure record path is missing");
    valid = false;
  } else {
    manifest.closurePath = resolvePath(path, *closure);
  }

  const json::Array *variants = root->getArray("graph_variants");
  if (!variants)
    variants = root->getArray("variants");
  if (!variants || variants->empty()) {
    addError(errors, "graph_variants must be a non-empty array");
    return false;
  }
  std::set<std::string> graphIds;
  for (const json::Value &value : *variants) {
    const json::Object *object = value.getAsObject();
    if (!object) {
      addError(errors, "graph variant entry is not an object");
      valid = false;
      continue;
    }
    VariantSpec variant;
    bool variantValid = true;
    variantValid &= requireString(*object, "graph_variant_id",
                                  variant.graphVariantId, errors);
    if (variant.graphVariantId.empty() ||
        !graphIds.insert(variant.graphVariantId).second) {
      addError(errors, "graph variant IDs must be unique and non-empty");
      variantValid = false;
    }
    std::optional<std::string> semantic =
        stringField(*object, "semantic_graph_id");
    if (!semantic)
      semantic = stringField(*object, "semantic_id");
    if (!semantic || semantic->empty()) {
      addError(errors, "graph variant semantic_graph_id is missing");
      variantValid = false;
    } else {
      variant.semanticGraphId = std::move(*semantic);
    }
    std::optional<std::string> score =
        readPathField(*object, "score_file_path", "score_path");
    if (!score || score->empty()) {
      addError(errors, "graph variant score_file_path is missing");
      variantValid = false;
    } else {
      variant.scorePath = resolvePath(path, *score);
    }
    std::optional<std::string> mapperReplay =
        readPathField(*object, "mapper_replay_path", "prepared_ir_path");
    if (!mapperReplay)
      mapperReplay = readPathField(*object, "materialized_ir_path",
                                   "graph_ir_path");
    if (!mapperReplay || mapperReplay->empty()) {
      addError(errors, "graph variant mapper_replay_path is missing");
      variantValid = false;
    } else {
      variant.mapperReplayPath = resolvePath(path, *mapperReplay);
      if (!llvm::sys::fs::exists(variant.mapperReplayPath)) {
        addError(errors, "graph variant mapper_replay_path does not exist: " +
                             variant.mapperReplayPath);
        variantValid = false;
      }
    }
    if (std::optional<std::string> stage = stringField(*object, "stage"))
      variant.stage = std::move(*stage);
    const json::Array *rewritePaths = object->getArray("rewrite_paths");
    if (std::optional<std::string> rewrite = stringField(*object, "rewrite_path"))
      appendUnique(variant.rewritePaths, *rewrite);
    else if (std::optional<std::string> rewrite = stringField(*object, "path"))
      appendUnique(variant.rewritePaths, *rewrite);
    if (rewritePaths) {
      for (const json::Value &rewriteValue : *rewritePaths) {
        std::optional<StringRef> rewrite = rewriteValue.getAsString();
        if (!rewrite || rewrite->empty()) {
          addError(errors, "graph variant rewrite_paths contains a non-string");
          variantValid = false;
        } else {
          appendUnique(variant.rewritePaths, *rewrite);
        }
      }
    }
    if (variant.rewritePaths.empty()) {
      addError(errors, "graph variant rewrite_path is missing");
      variantValid = false;
    }
    if (variant.stage.empty())
      variant.stage = manifest.stage;
    if (!variantValid)
      valid = false;
    manifest.variants.push_back(std::move(variant));
  }
  return valid && !manifest.variants.empty();
}

static void closureError(ClosureInfo &info, StringRef message) {
  addError(info.errors, message);
}

static bool parseClosure(StringRef path, const InputManifest &input,
                         ClosureInfo &info, bool allowIncompleteScope) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    closureError(info, "cannot read graph closure " + path.str() + ": " +
                           buffer.getError().message());
    return false;
  }
  llvm::line_iterator lines(**buffer, true);
  bool sawHeader = false;
  bool sawFooter = false;
  unsigned lineNumber = 0;
  std::set<std::string> paths;
  std::optional<json::Object> footer;
  for (; !lines.is_at_end(); ++lines) {
    ++lineNumber;
    llvm::Expected<json::Value> parsed = json::parse(*lines);
    if (!parsed) {
      closureError(info, "invalid graph closure JSON at line " +
                             std::to_string(lineNumber) + ": " +
                             llvm::toString(parsed.takeError()));
      continue;
    }
    const json::Object *object = parsed->getAsObject();
    if (!object) {
      closureError(info, "graph closure record is not an object");
      continue;
    }
    std::optional<StringRef> schema = object->getString("schema");
    if (!schema || *schema != kClosureSchema) {
      closureError(info, "graph closure record has the wrong schema");
      continue;
    }
    std::optional<StringRef> type = object->getString("record_type");
    if (!type) {
      closureError(info, "graph closure record has no record_type");
      continue;
    }
    if (*type == "header") {
      if (sawHeader || sawFooter) {
        closureError(info, "graph closure header is misplaced or repeated");
        continue;
      }
      sawHeader = true;
      std::optional<StringRef> stage = object->getString("stage");
      std::optional<StringRef> function = object->getString("function");
      if (!stage || *stage != input.stage)
        closureError(info, "graph closure stage does not match the manifest");
      else
        info.headerStage = stage->str();
      if (!function || *function != input.function)
        closureError(info,
                     "graph closure function does not match the manifest");
      if (std::optional<StringRef> architecture =
              object->getString("architecture_path")) {
        if (*architecture != input.architecturePath)
          closureError(info,
                       "graph closure architecture does not match the manifest");
      }
      if (std::optional<StringRef> source =
              object->getString("source_repository")) {
        if (*source != input.sourceRepository)
          closureError(info,
                       "graph closure source repository does not match");
      }
      if (std::optional<StringRef> commit = object->getString("source_commit")) {
        if (*commit != input.sourceCommit)
          closureError(info, "graph closure source commit does not match");
      }
      continue;
    }
    if (!sawHeader || sawFooter) {
      closureError(info, "graph closure record appears outside header/footer");
      continue;
    }
    if (*type == "footer") {
      sawFooter = true;
      footer = *object;
      continue;
    }
    if (*type == "action") {
      if (std::optional<StringRef> status = object->getString("status")) {
        // A seed-only merge may consume a bounded closure that records
        // rejected/unsupported action attempts alongside its legal graph
        // rows.  Those attempts are outside the admitted candidate set; the
        // individual graph rows below still require status=legal and complete
        // facts.  Treating the bounded diagnostics as a closure-wide error
        // would prevent reusing otherwise source-owned feasible rows.
        if (*status == "proven_illegal") {
          auto reason = object->getString("reason");
          if (!reason || reason->empty())
            closureError(info, "proven-illegal closure action has no proof reason");
        } else if (*status != "legal" && !allowIncompleteScope) {
          closureError(info, "graph closure contains an unknown action");
        }
      } else {
        closureError(info, "graph closure action has no status");
      }
      continue;
    }
    if (*type != "graph") {
      closureError(info, "graph closure contains an unknown record type");
      continue;
    }
    std::optional<StringRef> status = object->getString("status");
    std::optional<StringRef> graphPath = object->getString("path");
    const json::Object *structural = object->getObject("structural_key");
    if (!status || *status != "legal" || !graphPath || graphPath->empty() ||
        !structural) {
      // In incumbent-only mode, the closure may be a bounded frontier whose
      // rejected/incomplete rows are retained for diagnostics.  They are not
      // admitted members; only complete legal rows can bind score evidence.
      if (!allowIncompleteScope)
        closureError(info, "graph closure contains an incomplete graph record");
      continue;
    }
    if (!paths.insert(graphPath->str()).second) {
      closureError(info, "graph closure repeats a rewrite path");
      continue;
    }
    ClosureGraph graph;
    graph.path = graphPath->str();
    graph.structuralText =
        jsonText(json::Value(json::Object(*structural)));
    info.structuralKeys.insert(graph.structuralText);
    if (std::optional<StringRef> semantic =
            object->getString("semantic_graph_id"))
      graph.semanticGraphId = semantic->str();
    if (std::optional<StringRef> graphId =
            object->getString("graph_variant_id"))
      graph.graphVariantId = graphId->str();
    // The production closure currently calls these fields graph_id/path.
    // Bind that source-owned identity into the ranker's graph namespace rather
    // than leaving the graph unbound when the newer aliases are absent.
    if (graph.graphVariantId.empty())
      if (std::optional<StringRef> graphId = object->getString("graph_id"))
        graph.graphVariantId = graphId->str();
    if (graph.semanticGraphId.empty())
      graph.semanticGraphId = graph.graphVariantId;
    if (std::optional<StringRef> mlirPath = object->getString("mlir_path"))
      graph.mlirPath = mlirPath->str();
    if (std::optional<StringRef> factsPath = object->getString("facts_path"))
      graph.factsPath = factsPath->str();
    if (graph.mlirPath.empty() || graph.factsPath.empty())
      closureError(info, "graph closure graph has no persisted MLIR/facts paths");
    else if (!llvm::sys::fs::exists(graph.mlirPath) ||
             !llvm::sys::fs::exists(graph.factsPath))
      closureError(info, "graph closure graph MLIR/facts artifact is missing");
    const bool requiresKFacts =
        input.stage == "shape-temporal-replica-tiling" ||
        input.stage == "full-joint";
    const bool requiresFusionFacts = input.stage == "full-joint";
    // K/fusion facts describe coverage of possible future rewrites.  They
    // do not invalidate a persisted, verified graph used as a feasible seed.
    // Full-stage ranking still requires these facts and a complete closure;
    // incumbent-only output remains seed-only and cannot certify the stage.
    if (!allowIncompleteScope &&
        ((requiresKFacts &&
          !object->getBoolean("k_facts_complete").value_or(false)) ||
         (requiresFusionFacts &&
          (!object->getBoolean("sibling_facts_complete").value_or(false) ||
           !object->getBoolean("sibling_materialization_complete")
                .value_or(false)))))
      closureError(info, "graph closure graph facts are not complete");
    if (std::optional<bool> deduplicated = object->getBoolean("deduplicated"))
      graph.deduplicated = *deduplicated;
    if (std::optional<StringRef> firstPath = object->getString("first_path"))
      graph.firstPath = firstPath->str();
    info.graphs.push_back(std::move(graph));
  }
  if (!sawHeader)
    closureError(info, "graph closure is missing its header");
  if (!sawFooter || !footer)
    closureError(info, "graph closure is missing its footer");
  if (footer) {
    std::optional<StringRef> status = footer->getString("status");
    std::optional<bool> complete = footer->getBoolean("complete");
    std::optional<bool> graphComplete = footer->getBoolean("graph_scope_complete");
    const bool scopeComplete =
        status && *status == "complete" && complete && *complete &&
        graphComplete && *graphComplete;
    info.scopeComplete = scopeComplete;
    if (!scopeComplete && !allowIncompleteScope)
      closureError(info, "graph closure is incomplete");
    if (std::optional<bool> resourceBound =
            footer->getBoolean("resource_bound_reached")) {
      if (*resourceBound && !allowIncompleteScope)
        closureError(info, "graph closure reached its resource bound");
    }
    if (std::optional<int64_t> unknown =
            footer->getInteger("unknown_action_count")) {
      if (*unknown != 0 && !allowIncompleteScope)
        closureError(info, "graph closure contains unknown actions");
    } else {
      closureError(info, "graph closure footer has no unknown_action_count");
    }
    if (std::optional<int64_t> incompleteFacts =
            footer->getInteger("incomplete_facts_count")) {
      if (*incompleteFacts != 0 && !allowIncompleteScope)
        closureError(info, "graph closure contains incomplete graph facts");
    } else {
      closureError(info, "graph closure footer has no incomplete_facts_count");
    }
    if (std::optional<int64_t> unique =
            footer->getInteger("unique_structural_keys")) {
      if (!allowIncompleteScope &&
          (*unique < 0 ||
           static_cast<size_t>(*unique) != info.structuralKeys.size()))
        closureError(info, "graph closure unique structural-key count disagrees");
    } else {
      closureError(info, "graph closure footer has no unique_structural_keys");
    }
    if (std::optional<int64_t> graphCount =
            footer->getInteger("graph_count")) {
      if (!allowIncompleteScope &&
          (*graphCount < 0 ||
           static_cast<size_t>(*graphCount) != info.structuralKeys.size()))
        closureError(info, "graph closure graph_count disagrees with graph records");
    }
  }
  for (const ClosureGraph &graph : info.graphs) {
    if (graph.deduplicated && graph.firstPath.empty())
      if (!allowIncompleteScope)
        closureError(info, "deduplicated graph closure row has no first_path");
    if (!graph.firstPath.empty() && !paths.count(graph.firstPath))
      if (!allowIncompleteScope)
        closureError(info, "graph closure first_path is not a recorded path");
  }
  info.valid = info.errors.empty();
  return info.valid;
}

static void scoreError(ScoreInfo &info, StringRef message) {
  addError(info.errors, message);
}

static bool isProductionStageOne(StringRef stage) {
  // The source-owned stage names currently use shape-only for stage 1.  Keep
  // the explicit stage-1 spelling accepted for manifests produced by the
  // ablation driver while treating every later stage as critical-path.
  return stage == "shape-only" || stage == "stage1" || stage == "stage-1";
}

static bool isProductionDispatchPolicy(StringRef stage, StringRef policy) {
  return isProductionStageOne(stage) ? policy == "fixed"
                                    : policy == "critical-path";
}

static StringRef productionDispatchPolicyScope(StringRef policy) {
  return policy == "fixed" ? StringRef("canonical-smallest-ready-task")
                            : StringRef(
                                  "production-critical-path-dependency-ready");
}

// Legacy exact score files predate the namespace/schema fields in their
// header.  They remain usable only when the adjacent source-owned command
// binds the score to its actual graph module, architecture, candidate space,
// and cost catalogue.  Header strings alone are never provenance.
static bool validateLegacyScoreCommand(StringRef scorePath,
                                       StringRef graphVariantId,
                                       const InputManifest &input,
                                       const ClosureInfo &closure,
                                       bool seedOnly = false,
                                       StringRef expectedCostNamespace = {}) {
  llvm::SmallString<256> commandPath(scorePath);
  llvm::sys::path::remove_filename(commandPath);
  llvm::sys::path::append(commandPath, "score.command.json");
  if (!llvm::sys::fs::exists(commandPath))
    return false;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> commandBuffer =
      llvm::MemoryBuffer::getFile(commandPath);
  if (!commandBuffer)
    return false;
  llvm::Expected<json::Value> commandParsed =
      json::parse((*commandBuffer)->getBuffer());
  if (!commandParsed)
    return false;
  const json::Object *commandObject = commandParsed->getAsObject();
  const json::Array *argv = commandObject
                                ? commandObject->getArray("argv")
                                : nullptr;
  if (!commandObject || !argv || argv->size() < 3 ||
      commandObject->getInteger("exit_code") != std::optional<int64_t>(0))
    return false;
  // The command sidecar is authoritative for provenance, while the score
  // header selects which CLI contract is expected.  Keep the legacy exact
  // tuple strict and admit the production scheduler tuple only when the
  // source-owned score explicitly declares it.
  bool productionScheduler = false;
  {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> scoreBuffer =
        llvm::MemoryBuffer::getFile(scorePath);
    if (!scoreBuffer)
      return false;
    llvm::line_iterator scoreLines(**scoreBuffer, true);
    if (scoreLines.is_at_end())
      return false;
    llvm::Expected<json::Value> scoreParsed = json::parse(*scoreLines);
    if (!scoreParsed)
      return false;
    const json::Object *scoreHeader = scoreParsed->getAsObject();
    if (!scoreHeader ||
        scoreHeader->getString("record_type") !=
            std::optional<StringRef>("header"))
      return false;
    productionScheduler =
        scoreHeader->getString("temporal_model") ==
            std::optional<StringRef>(kProductionTemporalModel) ||
        scoreHeader->getString("schedule_space") ==
            std::optional<StringRef>(kProductionScheduleSpace);
    if (productionScheduler &&
        (scoreHeader->getString("temporal_model") !=
             std::optional<StringRef>(kProductionTemporalModel) ||
         scoreHeader->getString("schedule_space") !=
             std::optional<StringRef>(kProductionScheduleSpace)))
      return false;
  }
  auto started = commandObject->getNumber("started_unix");
  if (!started || !std::isfinite(*started) || *started <= 0.0)
    return false;
  std::string modulePath;
  std::string architecturePath;
  std::map<std::string, std::string> options;
  for (auto [index, value] : llvm::enumerate(*argv)) {
    auto text = value.getAsString();
    if (!text)
      continue;
    if (index == 1)
      modulePath = text->str();
    if (text->starts_with("--architecture-spec="))
      architecturePath =
          text->drop_front(strlen("--architecture-spec=")).str();
    if (text->starts_with("--score-exact-joint-task-candidates=")) {
      SmallVector<StringRef> fields;
      text->drop_front(strlen("--score-exact-joint-task-candidates="))
          .split(fields, ' ', -1, false);
      for (StringRef field : fields) {
        size_t equals = field.find('=');
        if (equals == StringRef::npos || equals == 0)
          return false;
        options[field.take_front(equals).str()] =
            field.drop_front(equals + 1).str();
      }
    }
  }
  auto option = [&](StringRef name) -> std::optional<StringRef> {
    auto it = options.find(name.str());
    return it == options.end() ? std::nullopt
                               : std::optional<StringRef>(it->second);
  };
  auto output = option("output");
  auto candidate = option("candidates");
  auto cost = option("cost-file");
  const StringRef expectedDispatchPolicy =
      productionScheduler
          ? (isProductionStageOne(input.stage) ? StringRef("fixed")
                                               : StringRef("critical-path"))
          : StringRef("all-ready");
  const StringRef expectedScheduleSpace =
      productionScheduler ? StringRef(kProductionScheduleSpace) : StringRef("exact");
  const StringRef expectedTemporalModel =
      productionScheduler ? StringRef(kProductionTemporalModel)
                          : StringRef("dispatch-order");
  if (modulePath.empty() || architecturePath.empty() || !output || !candidate ||
      !cost || normalizedPath(*output) != normalizedPath(scorePath) ||
      normalizedPath(architecturePath) != normalizedPath(input.architecturePath) ||
      option("graph-variant-id") != std::optional<StringRef>(graphVariantId) ||
      option("source-git-repository") !=
          std::optional<StringRef>(input.sourceRepository) ||
      option("source-git-commit") !=
          std::optional<StringRef>(input.sourceCommit) ||
      option("dispatch-policy") !=
          std::optional<StringRef>(expectedDispatchPolicy) ||
      option("schedule-space") !=
          std::optional<StringRef>(expectedScheduleSpace) ||
      option("temporal-model") !=
          std::optional<StringRef>(expectedTemporalModel) ||
      (!seedOnly && option("max-expanded-nodes") !=
          std::optional<StringRef>("9223372036854775807")) ||
      option("max-milliseconds") != std::optional<StringRef>("0") ||
      option("max-makespan") != std::optional<StringRef>("0"))
    return false;
  if (seedOnly) {
    if (option("seed-only") != std::optional<StringRef>("true"))
      return false;
    if (option("stage") != std::optional<StringRef>(input.stage))
      return false;
    int64_t maxExpanded = 0;
    if (!option("max-expanded-nodes") ||
        option("max-expanded-nodes")->getAsInteger(10, maxExpanded) ||
        maxExpanded <= 0)
      return false;
  } else if (option("seed-only") == std::optional<StringRef>("true")) {
    return false;
  }
  auto graphIt = llvm::find_if(closure.graphs, [&](const ClosureGraph &graph) {
    return graph.graphVariantId == graphVariantId;
  });
  if (graphIt == closure.graphs.end() ||
      normalizedPath(modulePath) != normalizedPath(graphIt->mlirPath) ||
      !llvm::sys::fs::exists(graphIt->factsPath))
    return false;
  for (StringRef path : {StringRef(architecturePath), StringRef(modulePath),
                         *candidate, *cost, StringRef(graphIt->factsPath)}) {
    llvm::sys::fs::file_status status;
    if (std::error_code ec = llvm::sys::fs::status(path, status))
      return false;
    using namespace std::chrono;
    const auto startedNanos =
        duration_cast<nanoseconds>(duration<double>(*started));
    if (status.getLastModificationTime() >
        llvm::sys::TimePoint<>(startedNanos))
      return false;
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> costBuffer =
      llvm::MemoryBuffer::getFile(*cost);
  if (!costBuffer)
    return false;
  llvm::Expected<json::Value> costParsed =
      json::parse((*costBuffer)->getBuffer());
  if (!costParsed)
    return false;
  const json::Object *costRoot = costParsed->getAsObject();
  const json::Object *metadata =
      costRoot ? costRoot->getObject("predictor_metadata") : nullptr;
  if (!costRoot || !metadata)
    return false;
  auto namespaceValue = costRoot->getString("namespace");
  auto sourceGraph = metadata->getString("source_graph_id");
  auto architectureSchema = metadata->getString("architecture_schema");
  auto architectureValue = metadata->getString("architecture_path");
  auto sourceRepository = metadata->getString("source_repository");
  auto sourceCommit = metadata->getString("source_commit");
  auto predictor = metadata->getString("predictor_source");
  auto provenance = metadata->getString("provenance_schema");
  auto model = metadata->getString("model");
  auto modelSchema = metadata->getString("model_schema");
  auto featureContract = metadata->getString("feature_contract_id");
  auto featureExtractor = metadata->getString("feature_extractor");
  if (!namespaceValue || namespaceValue->empty() || !sourceGraph ||
      *sourceGraph != graphVariantId || !architectureSchema ||
      *architectureSchema != "neura-architecture-v1" || !architectureValue ||
      normalizedPath(*architectureValue) != normalizedPath(input.architecturePath) ||
      !sourceRepository || *sourceRepository != input.sourceRepository ||
      !sourceCommit || *sourceCommit != input.sourceCommit || !predictor ||
      *predictor != kPredictorSource || !provenance ||
      *provenance != "orbit-cost-provenance-v1" || !model ||
      model->empty() || !modelSchema || modelSchema->empty() ||
      !featureContract || featureContract->empty() || !featureExtractor ||
      featureExtractor->empty())
    return false;
  if (!expectedCostNamespace.empty() &&
      *namespaceValue != expectedCostNamespace)
    return false;
  return true;
}

static bool hasCompleteScoreRecord(const json::Object &record) {
  return record.getString("schema") == std::optional<StringRef>(kScoreSchema) &&
         record.getString("record_type") ==
             std::optional<StringRef>("score") &&
         record.getBoolean("valid").value_or(false) &&
         record.getString("candidate_id") &&
         !record.getString("candidate_id")->empty() &&
         record.getInteger("predicted_whole_program_cycles").value_or(0) > 0 &&
         record.getArray("task_costs") &&
         !record.getArray("task_costs")->empty() &&
         record.getArray("task_schedule") &&
         !record.getArray("task_schedule")->empty() &&
         record.getString("score_source") == std::optional<StringRef>(
             "ml-ii-startup-plus-production-explicit-communication") &&
         record.getString("task_latency_provenance") ==
             std::optional<StringRef>(
                 "source-owned-ml-ii-startup-trip-count-contract");
}

// Read one source-owned exact score artifact.  Global ranker selections carry
// a copy of a score row, so the copy must be compared with the row in the
// command-bound score file instead of being trusted on its own.
static bool readScoreArtifact(StringRef path, ScoreArtifact &artifact) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return false;
  llvm::line_iterator lines(**buffer, true);
  bool sawHeader = false;
  bool sawFooter = false;
  std::set<std::string> candidateIds;
  for (; !lines.is_at_end(); ++lines) {
    llvm::Expected<json::Value> parsed = json::parse(*lines);
    if (!parsed)
      return false;
    const json::Object *object = parsed->getAsObject();
    if (!object ||
        object->getString("schema") != std::optional<StringRef>(kScoreSchema))
      return false;
    std::optional<StringRef> type = object->getString("record_type");
    if (!type)
      return false;
    if (!sawHeader) {
      if (*type != "header")
        return false;
      artifact.header = *object;
      sawHeader = true;
      continue;
    }
    if (sawFooter)
      return false;
    if (*type == "footer") {
      artifact.footer = *object;
      sawFooter = true;
      continue;
    }
    if (*type != "score" || !hasCompleteScoreRecord(*object))
      return false;
    std::optional<StringRef> candidateId = object->getString("candidate_id");
    if (!candidateId || !candidateIds.insert(candidateId->str()).second)
      return false;
    artifact.rows.push_back(*object);
  }
  return sawHeader && sawFooter && artifact.header.has_value() &&
         artifact.footer.has_value();
}

static bool validateBoundScoreArtifact(
    const EvidenceRow &row, const InputManifest &input,
    const ClosureInfo &closure, bool seedOnly,
    StringRef expectedCostNamespace) {
  ScoreArtifact artifact;
  if (!readScoreArtifact(row.scorePath, artifact))
    return false;
  const json::Object &header = *artifact.header;
  const json::Object &footer = *artifact.footer;
  const bool actualSeedOnly = header.getBoolean("seed_only").value_or(false);
  const bool productionScheduler =
      header.getString("temporal_model") ==
          std::optional<StringRef>(kProductionTemporalModel) ||
      header.getString("schedule_space") ==
          std::optional<StringRef>(kProductionScheduleSpace);
  if (productionScheduler &&
      (header.getString("temporal_model") !=
           std::optional<StringRef>(kProductionTemporalModel) ||
       header.getString("schedule_space") !=
           std::optional<StringRef>(kProductionScheduleSpace)))
    return false;
  if (!seedOnly && actualSeedOnly)
    return false;
  if (!validateLegacyScoreCommand(row.scorePath, row.graphVariantId, input,
                                  closure, actualSeedOnly,
                                  expectedCostNamespace))
    return false;
  auto headerEquals = [&](StringRef key, StringRef expected) {
    return header.getString(key) == std::optional<StringRef>(expected);
  };
  if (!headerEquals("graph_variant_id", row.graphVariantId) ||
      !headerEquals("function", input.function) ||
      !headerEquals("source_repository", input.sourceRepository) ||
      !headerEquals("source_commit", input.sourceCommit) ||
      !headerEquals("architecture_path", input.architecturePath) ||
      header.getString("predictor_source") !=
          std::optional<StringRef>(kPredictorSource))
    return false;
  if (productionScheduler) {
    const auto expectedPolicy = isProductionStageOne(input.stage)
                                    ? StringRef("fixed")
                                    : StringRef("critical-path");
    if (header.getString("temporal_model") !=
            std::optional<StringRef>(kProductionTemporalModel) ||
        header.getString("schedule_space") !=
            std::optional<StringRef>(kProductionScheduleSpace) ||
        header.getString("start_policy") !=
            std::optional<StringRef>(kProductionStartPolicy) ||
        header.getString("temporal_search_scope") !=
            std::optional<StringRef>(kProductionTemporalScope) ||
        header.getBoolean("cycle_starts_are_search_dimensions") !=
            std::optional<bool>(false) ||
        header.getBoolean("global_shape_location_temporal_space") !=
            std::optional<bool>(false) ||
        header.getBoolean("placement_coverage_limited") !=
            std::optional<bool>(true) ||
        (header.getBoolean("location_coverage_limited") &&
         !header.getBoolean("location_coverage_limited").value()) ||
        (header.getBoolean("temporal_coverage_limited") &&
         !header.getBoolean("temporal_coverage_limited").value()) ||
        header.getBoolean("orbit_4x4_pruning").value_or(true) ||
        header.getString("shape_domain_pruning") !=
            std::optional<StringRef>("dependency-release-tail-cost-only") ||
        header.getString("scheduler_backend") !=
            std::optional<StringRef>(kProductionSchedulerBackend) ||
        header.getString("dispatch_policy") !=
            std::optional<StringRef>(expectedPolicy) ||
        header.getString("dispatch_policy_scope") !=
            std::optional<StringRef>(productionDispatchPolicyScope(expectedPolicy)))
      return false;
  }
  if (auto sourceGraph = header.getString("source_graph_id"))
    if (*sourceGraph != row.graphVariantId)
      return false;
  if (auto stage = header.getString("stage")) {
    if (*stage != input.stage)
      return false;
  } else if (actualSeedOnly) {
    return false;
  }
  if (auto architectureSchema = header.getString("architecture_schema"))
    if (*architectureSchema != "neura-architecture-v1")
      return false;
  if (auto costNamespace = header.getString("cost_namespace")) {
    if (costNamespace->empty() ||
        (!expectedCostNamespace.empty() &&
         *costNamespace != expectedCostNamespace))
      return false;
    if (header.getString("cost_provenance_schema") !=
        std::optional<StringRef>("orbit-cost-provenance-v1"))
      return false;
  } else if (actualSeedOnly && !expectedCostNamespace.empty()) {
    return false;
  } else if (header.getString("cost_provenance_schema")) {
    return false;
  }

  const bool footerSeedOnly =
      footer.getString("status") == std::optional<StringRef>("seed-only") &&
      footer.getBoolean("incomplete") == std::optional<bool>(true) &&
      footer.getBoolean("top_k_certified") == std::optional<bool>(false) &&
      footer.getBoolean("seed_only").value_or(false) &&
      footer.getBoolean("feasible_only").value_or(false) &&
      !footer.getBoolean("complete").value_or(false) &&
      !footer.getBoolean("stage_certified").value_or(false) &&
      footer.getString("ranking_certificate") ==
          std::optional<StringRef>("feasible-seed-only");
  const bool headerSeedOnly = header.getBoolean("seed_only").value_or(false);
  if (header.getBoolean("global_cutoff_requested").value_or(false) ||
      footer.getBoolean("global_cutoff_complete").value_or(false) ||
      footer.getString("global_cutoff_evidence_path"))
    return false;
  if (headerSeedOnly != actualSeedOnly || footerSeedOnly != actualSeedOnly)
    return false;
  if (actualSeedOnly) {
    if (decimalField(footer, "seed_feasible_count") !=
            std::optional<std::string>(std::to_string(artifact.rows.size())) ||
        artifact.rows.size() < 5)
      return false;
  } else {
    if (footer.getString("status") !=
            std::optional<StringRef>("complete") ||
        footer.getBoolean("incomplete") != std::optional<bool>(false) ||
        footer.getBoolean("top_k_certified") != std::optional<bool>(true) ||
        footer.getBoolean("seed_only").value_or(false) ||
        footer.getBoolean("feasible_only").value_or(false))
      return false;
    const std::optional<StringRef> certificate =
        footer.getString("ranking_certificate");
    const bool orderDerivedCertificate =
        certificate == std::optional<StringRef>(
            "proven-shape-location-dispatch-order-branch-and-bound");
    const bool productionCertificate =
        certificate == std::optional<StringRef>(kProductionCertificate);
    const bool completeCertificate =
        certificate == std::optional<StringRef>(
                           "proven-shape-and-schedule-branch-and-bound") ||
        certificate == std::optional<StringRef>(
                           "exhaustive-shape-plus-bounded-exact-schedule") ||
        orderDerivedCertificate ||
        productionCertificate ||
        certificate == std::optional<StringRef>(
                           "proven-shape-ranking-at-single-heuristic-placement");
    if (!completeCertificate ||
        (!orderDerivedCertificate && !productionCertificate &&
         !footer.getBoolean("horizon_proven").value_or(false)))
      return false;
    if (productionScheduler &&
        (footer.getString("temporal_model") !=
             std::optional<StringRef>(kProductionTemporalModel) ||
         footer.getString("schedule_space") !=
             std::optional<StringRef>(kProductionScheduleSpace) ||
         footer.getString("temporal_search_scope") !=
             std::optional<StringRef>(kProductionTemporalScope) ||
         footer.getString("start_policy") !=
             std::optional<StringRef>(kProductionStartPolicy) ||
         footer.getBoolean("cycle_starts_are_search_dimensions") !=
             std::optional<bool>(false) ||
         footer.getBoolean("global_shape_location_temporal_space") !=
             std::optional<bool>(false) ||
         footer.getBoolean("placement_coverage_limited") !=
             std::optional<bool>(true) ||
         (footer.getBoolean("location_coverage_limited") &&
          !footer.getBoolean("location_coverage_limited").value()) ||
         (footer.getBoolean("temporal_coverage_limited") &&
          !footer.getBoolean("temporal_coverage_limited").value()) ||
         footer.getBoolean("orbit_4x4_pruning").value_or(true) ||
         footer.getString("shape_domain_pruning") !=
             std::optional<StringRef>("dependency-release-tail-cost-only") ||
         footer.getString("scheduler_backend") !=
             std::optional<StringRef>(kProductionSchedulerBackend) ||
         footer.getString("dispatch_policy") !=
             header.getString("dispatch_policy") ||
         footer.getString("dispatch_policy_scope") !=
             header.getString("dispatch_policy_scope")))
      return false;
  }
  if (productionScheduler && actualSeedOnly &&
      (footer.getString("temporal_model") !=
           std::optional<StringRef>(kProductionTemporalModel) ||
       footer.getString("schedule_space") !=
           std::optional<StringRef>(kProductionScheduleSpace) ||
       footer.getString("temporal_search_scope") !=
           std::optional<StringRef>(kProductionTemporalScope) ||
       footer.getString("start_policy") !=
           std::optional<StringRef>(kProductionStartPolicy) ||
       footer.getBoolean("cycle_starts_are_search_dimensions") !=
           std::optional<bool>(false) ||
       footer.getBoolean("global_shape_location_temporal_space") !=
           std::optional<bool>(false) ||
       footer.getBoolean("placement_coverage_limited") !=
           std::optional<bool>(true) ||
       (footer.getBoolean("location_coverage_limited") &&
        !footer.getBoolean("location_coverage_limited").value()) ||
       (footer.getBoolean("temporal_coverage_limited") &&
        !footer.getBoolean("temporal_coverage_limited").value()) ||
       footer.getBoolean("orbit_4x4_pruning").value_or(true) ||
       footer.getString("shape_domain_pruning") !=
           std::optional<StringRef>("dependency-release-tail-cost-only") ||
       footer.getString("scheduler_backend") !=
           std::optional<StringRef>(kProductionSchedulerBackend) ||
       footer.getString("dispatch_policy") !=
           header.getString("dispatch_policy") ||
       footer.getString("dispatch_policy_scope") !=
           header.getString("dispatch_policy_scope")))
    return false;
  std::optional<std::string> headerCount =
      decimalField(header, "candidate_count");
  std::optional<std::string> footerCount =
      decimalField(footer, "candidate_count");
  if (!headerCount || !footerCount || *headerCount != *footerCount ||
      decimalField(footer, "scored_count") !=
          std::optional<std::string>(std::to_string(artifact.rows.size())))
    return false;

  const json::Object *matching = nullptr;
  for (const json::Object &candidate : artifact.rows) {
    if (candidate.getString("candidate_id") ==
            std::optional<StringRef>(row.candidateId) &&
        candidate.getInteger("predicted_whole_program_cycles") ==
            std::optional<int64_t>(row.cycles)) {
      if (matching)
        return false;
      matching = &candidate;
    }
  }
  return matching && *matching == row.scoreRecord;
}

// Re-open the evidence named by a globally-pruned score.  The score footer is
// a certificate emitted by the C++ score pass, but GlobalStageRanker must also
// bind that certificate to the same provenance instead of trusting a numeric
// cutoff copied into a JSON footer.
static bool validateGlobalCutoffEvidence(const ScoreInfo &score,
                                         const InputManifest &input,
                                         const ClosureInfo &closure) {
  if (score.globalCutoffEvidencePath.empty() ||
      score.globalCutoffCycles.empty() || closure.graphs.empty() ||
      closure.headerStage != input.stage)
    return false;
  // Five verified feasible rows suffice for a numeric cutoff even when
  // enumeration has unresolved rewrite paths. The row/graph/provenance gates
  // below still apply; complete stage certification separately requires a
  // valid and complete closure.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(score.globalCutoffEvidencePath);
  if (!buffer)
    return false;
  llvm::line_iterator lines(**buffer, true);
  std::optional<json::Object> header;
  std::optional<json::Object> footer;
  std::vector<EvidenceRow> candidates;
  std::string schema;
  bool sawHeader = false;
  bool sawFooter = false;
  for (; !lines.is_at_end(); ++lines) {
    llvm::Expected<json::Value> parsed = json::parse(*lines);
    if (!parsed)
      return false;
    const json::Object *object = parsed->getAsObject();
    if (!object)
      return false;
    std::optional<StringRef> recordSchema = object->getString("schema");
    std::optional<StringRef> type = object->getString("record_type");
    if (!recordSchema || !type)
      return false;
    if (!sawHeader) {
      if (*type != "header")
        return false;
      schema = recordSchema->str();
      if (schema != kScoreSchema && schema != kOutputSchema)
        return false;
      header = *object;
      sawHeader = true;
      continue;
    }
    if (sawFooter)
      return false;
    if (*type == "footer") {
      if (*recordSchema != schema)
        return false;
      footer = *object;
      sawFooter = true;
      continue;
    }
    const bool exact = schema == kScoreSchema;
    if ((exact && *type != "score") || (!exact && *type != "selection"))
      return false;
    std::optional<StringRef> candidateId = object->getString("candidate_id");
    std::optional<int64_t> cycles =
        object->getInteger("predicted_whole_program_cycles");
    if (!candidateId || candidateId->empty() || !cycles || *cycles <= 0)
      return false;
    EvidenceRow row;
    row.cycles = *cycles;
    row.candidateId = candidateId->str();
    if (!exact) {
      auto semantic = object->getString("semantic_graph_id");
      auto graph = object->getString("graph_variant_id");
      auto scorePath = object->getString("score_file_path");
      if (!semantic || semantic->empty() || !graph || graph->empty() ||
          !scorePath || scorePath->empty())
        return false;
      row.semanticGraphId = semantic->str();
      row.graphVariantId = graph->str();
      row.scorePath = scorePath->str();
      row.certified = object->getBoolean("certified").value_or(false);
      row.feasible = object->getBoolean("feasible").value_or(false);
      row.seedOnly = object->getBoolean("seed_only").value_or(false);
      const json::Object *scoreRecord = object->getObject("score_record");
      if (!scoreRecord || !hasCompleteScoreRecord(*scoreRecord) ||
          scoreRecord->getString("candidate_id") !=
              std::optional<StringRef>(*candidateId) ||
          scoreRecord->getInteger("predicted_whole_program_cycles") !=
              std::optional<int64_t>(*cycles))
        return false;
      row.scoreRecord = *scoreRecord;
    } else {
      if (!hasCompleteScoreRecord(*object))
        return false;
      row.scorePath = score.globalCutoffEvidencePath;
      row.scoreRecord = *object;
    }
    candidates.push_back(std::move(row));
  }
  if (!header || !footer || candidates.size() < 5)
    return false;
  const json::Object &headerObject = *header;
  auto equals = [&](StringRef key, StringRef expected) {
    std::optional<StringRef> actual = headerObject.getString(key);
    return actual && *actual == expected;
  };
  if (!equals("function", input.function) ||
      !equals("source_repository", input.sourceRepository) ||
      !equals("source_commit", input.sourceCommit) ||
      !equals("architecture_path", input.architecturePath) ||
      !llvm::sys::fs::exists(input.architecturePath))
    return false;
  const bool exact = schema == kScoreSchema;
  const bool headerSeedOnly =
      headerObject.getBoolean("seed_only").value_or(false);
  if (auto stage = headerObject.getString("stage")) {
    if (*stage != input.stage)
      return false;
  } else if (!exact || headerSeedOnly) {
    return false;
  }
  if (auto closurePath = headerObject.getString("graph_closure_record_path")) {
    if (normalizedPath(*closurePath) != normalizedPath(input.closurePath))
      return false;
  } else if (schema != kScoreSchema) {
    return false;
  }
  if (headerObject.getString("predictor_source") !=
      std::optional<StringRef>(kPredictorSource))
    return false;
  if (auto architectureSchema = headerObject.getString("architecture_schema"))
    if (*architectureSchema != "neura-architecture-v1")
      return false;
  if (auto costNamespace = headerObject.getString("cost_namespace"))
    if (costNamespace->empty())
      return false;
  if (auto provenance = headerObject.getString("cost_provenance_schema")) {
    if (*provenance != "orbit-cost-provenance-v1")
      return false;
  } else if (headerSeedOnly) {
    return false;
  }
  const bool productionGlobal =
      headerObject.getString("temporal_model") ==
          std::optional<StringRef>(kProductionTemporalModel) ||
      headerObject.getString("schedule_space") ==
          std::optional<StringRef>(kProductionScheduleSpace);
  if (productionGlobal && !exact &&
      (headerObject.getString("schedule_space") !=
           std::optional<StringRef>(kProductionScheduleSpace) ||
       headerObject.getString("temporal_search_scope") !=
           std::optional<StringRef>(kProductionTemporalScope) ||
       headerObject.getString("start_policy") !=
           std::optional<StringRef>(kProductionStartPolicy) ||
       headerObject.getBoolean("cycle_starts_are_search_dimensions") !=
           std::optional<bool>(false) ||
       headerObject.getBoolean("global_shape_location_temporal_space") !=
           std::optional<bool>(false) ||
       headerObject.getBoolean("placement_coverage_limited") !=
           std::optional<bool>(true) ||
       headerObject.getBoolean("location_coverage_limited") !=
           std::optional<bool>(true) ||
       headerObject.getBoolean("temporal_coverage_limited") !=
           std::optional<bool>(true) ||
       headerObject.getString("scheduler_backend") !=
           std::optional<StringRef>(kProductionSchedulerBackend) ||
       headerObject.getBoolean("orbit_4x4_pruning").value_or(true) ||
       headerObject.getString("shape_domain_pruning") !=
           std::optional<StringRef>("dependency-release-tail-cost-only")))
    return false;
  if (!score.costNamespace.empty() &&
      headerObject.getString("cost_namespace") !=
          std::optional<StringRef>(score.costNamespace))
    return false;
  if (!exact && headerSeedOnly &&
      (!headerObject.getBoolean("complete").has_value() ||
       headerObject.getBoolean("complete").value() ||
       !headerObject.getBoolean("feasible_only").value_or(false) ||
       headerObject.getBoolean("stage_certified").value_or(true) ||
       headerObject.getString("status") !=
           std::optional<StringRef>("seed-only")))
    return false;
  std::set<std::string> closureGraphIds;
  std::map<std::string, std::string> closureSemanticByGraph;
  for (const ClosureGraph &graph : closure.graphs) {
    if (!graph.graphVariantId.empty()) {
      closureGraphIds.insert(graph.graphVariantId);
      if (!graph.semanticGraphId.empty())
        closureSemanticByGraph[graph.graphVariantId] = graph.semanticGraphId;
    }
  }
  if (closureGraphIds.empty())
    return false;
  if (exact) {
    auto graph = headerObject.getString("graph_variant_id");
    if (!graph || graph->empty())
      return false;
    auto semantic = headerObject.getString("semantic_graph_id");
    for (EvidenceRow &row : candidates) {
      row.graphVariantId = graph->str();
      row.semanticGraphId = semantic.value_or(*graph).str();
    }
  }
  std::set<std::tuple<int64_t, std::string, std::string, std::string>> keys;
  for (const EvidenceRow &row : candidates) {
    if (!closureGraphIds.count(row.graphVariantId) ||
        (closureSemanticByGraph.count(row.graphVariantId) &&
         closureSemanticByGraph[row.graphVariantId] != row.semanticGraphId) ||
        !keys.emplace(row.cycles, row.semanticGraphId, row.candidateId,
                      row.graphVariantId)
             .second)
      return false;
  }
  const json::Object &footerObject = *footer;
  if (productionGlobal && !exact &&
      (footerObject.getString("temporal_model") !=
           std::optional<StringRef>(kProductionTemporalModel) ||
       footerObject.getString("schedule_space") !=
           std::optional<StringRef>(kProductionScheduleSpace) ||
       footerObject.getBoolean("global_shape_location_temporal_space") !=
           std::optional<bool>(false) ||
       footerObject.getBoolean("placement_coverage_limited") !=
           std::optional<bool>(true) ||
       footerObject.getBoolean("location_coverage_limited") !=
           std::optional<bool>(true) ||
       footerObject.getBoolean("temporal_coverage_limited") !=
           std::optional<bool>(true) ||
       footerObject.getString("scheduler_backend") !=
           std::optional<StringRef>(kProductionSchedulerBackend) ||
       footerObject.getBoolean("orbit_4x4_pruning").value_or(true) ||
       footerObject.getString("shape_domain_pruning") !=
           std::optional<StringRef>("dependency-release-tail-cost-only") ||
       footerObject.getString("ranking_certificate") !=
           std::optional<StringRef>(
               headerSeedOnly
                   ? "feasible-seed-only"
                   : "global-proven-shape-space-under-production-scheduler-top-five")))
    return false;
  auto zeroField = [&](StringRef key) {
    if (auto value = footerObject.getInteger(key))
      return *value == 0;
    if (auto text = footerObject.getString(key))
      return isZero(*text);
    return false;
  };
  const std::optional<std::string> unmeasuredGraphVariantCount =
      decimalField(footerObject, "unmeasured_graph_variant_count");
  // The outer global JSONL is the only artifact that aggregates graph
  // variants.  Its count is decimal text when it exceeds JSON's signed
  // integer range, but it must always be present and parseable.  Seed-only
  // evidence may report a positive count; complete evidence must report zero.
  if (!exact && !unmeasuredGraphVariantCount)
    return false;
  const bool exactComplete =
      footerObject.getString("status") ==
          std::optional<StringRef>("complete") &&
      footerObject.getBoolean("top_k_certified").value_or(false) &&
      footerObject.getBoolean("incomplete").has_value() &&
      !footerObject.getBoolean("incomplete").value();
  const bool globalComplete =
      footerObject.getString("status") ==
          std::optional<StringRef>("complete") &&
      footerObject.getBoolean("complete").value_or(false) &&
      footerObject.getBoolean("stage_certified").value_or(false) &&
      footerObject.getBoolean("top_k_certified").value_or(false);
  const bool seedFooter =
      footerObject.getString("status") ==
          std::optional<StringRef>("seed-only") &&
      footerObject.getBoolean("incomplete") == std::optional<bool>(true) &&
      footerObject.getBoolean("top_k_certified") ==
          std::optional<bool>(false) &&
      footerObject.getBoolean("seed_only").value_or(false) &&
      footerObject.getBoolean("feasible_only").value_or(false) &&
      footerObject.getString("ranking_certificate") ==
          std::optional<StringRef>("feasible-seed-only") &&
      decimalField(footerObject, "seed_feasible_count").has_value() &&
      compareDecimal(*decimalField(footerObject, "seed_feasible_count"), "5") >=
          0 &&
      (exact ||
       (footerObject.getBoolean("complete").has_value() &&
        !footerObject.getBoolean("complete").value() &&
        footerObject.getBoolean("stage_certified").has_value() &&
        !footerObject.getBoolean("stage_certified").value()));
  if ((exact ? (!exactComplete && !seedFooter)
             : (!globalComplete && !seedFooter)) ||
      headerSeedOnly != seedFooter)
    return false;
  if (!exact && !seedFooter &&
      !isZero(*unmeasuredGraphVariantCount))
    return false;
  if (seedFooter) {
    if (footerObject.getBoolean("seed_only") != std::optional<bool>(true) ||
        decimalField(footerObject, "seed_feasible_count") !=
            std::optional<std::string>(std::to_string(candidates.size())))
      return false;
    if (!exact) {
      for (const EvidenceRow &row : candidates)
        if (row.certified || !row.feasible || !row.seedOnly)
          return false;
    }
  } else if (!exact) {
    for (const EvidenceRow &row : candidates)
      if (!row.certified || row.seedOnly)
        return false;
  }
  int64_t topK = headerObject.getInteger("top_k").value_or(0);
  if (topK == 0)
    topK = headerObject.getInteger("top_k_requested").value_or(0);
  if (topK < 5)
    return false;
  if (exact) {
    if (decimalField(footerObject, "scored_count") !=
            std::optional<std::string>(std::to_string(candidates.size())) ||
        (!seedFooter &&
         (!zeroField("unsupported_shape_candidates") ||
          !zeroField("unmeasured_shape_candidates"))))
      return false;
  } else if (decimalField(footerObject, "selected_count") !=
                 std::optional<std::string>(std::to_string(candidates.size())) ||
             !decimalField(headerObject, "scored_candidate_count") ||
             !decimalField(footerObject, "scored_candidate_count") ||
             *decimalField(headerObject, "scored_candidate_count") !=
                 *decimalField(footerObject, "scored_candidate_count") ||
             (seedFooter
                  ? (!decimalField(footerObject, "candidate_rows_considered") ||
                     compareDecimal(
                         *decimalField(footerObject, "candidate_rows_considered"),
                         std::to_string(candidates.size())) < 0)
                  : (!decimalField(footerObject, "candidate_rows_considered") ||
                     *decimalField(footerObject, "candidate_rows_considered") !=
                         std::to_string(candidates.size()) ||
                     !zeroField("unmeasured_candidate_count") ||
                     !zeroField("unmeasured_graph_variant_count"))))
    return false;
  if (seedFooter) {
    if (auto footerFunction = footerObject.getString("function"))
      if (*footerFunction != input.function)
        return false;
    for (StringRef key : {StringRef("stage"), StringRef("source_repository"),
                          StringRef("source_commit"), StringRef("architecture_path")}) {
      if (auto value = footerObject.getString(key)) {
        if (key == "stage" && *value != input.stage)
          return false;
        if (key == "source_repository" && *value != input.sourceRepository)
          return false;
        if (key == "source_commit" && *value != input.sourceCommit)
          return false;
        if (key == "architecture_path" &&
            normalizedPath(*value) != normalizedPath(input.architecturePath))
          return false;
      }
    }
    if (auto footerCost = footerObject.getString("cost_namespace"))
      if (score.costNamespace.empty() || *footerCost != score.costNamespace)
        return false;
    if (auto footerProvenance =
            footerObject.getString("cost_provenance_schema"))
      if (*footerProvenance != "orbit-cost-provenance-v1")
        return false;
  }
  for (const EvidenceRow &row : candidates)
    if (!validateBoundScoreArtifact(row, input, closure, seedFooter,
                                    score.costNamespace))
      return false;
  std::sort(candidates.begin(), candidates.end(),
            [](const EvidenceRow &lhs, const EvidenceRow &rhs) {
              if (lhs.cycles != rhs.cycles)
                return lhs.cycles < rhs.cycles;
              if (lhs.semanticGraphId != rhs.semanticGraphId)
                return lhs.semanticGraphId < rhs.semanticGraphId;
              int candidateOrder =
                  ::mlir::amoeba::neura::joint_scheduling::
                      compareProductionCandidateIds(lhs.candidateId,
                                                    rhs.candidateId);
              if (candidateOrder != 0)
                return candidateOrder < 0;
              return lhs.graphVariantId < rhs.graphVariantId;
            });
  uint64_t cutoff = 0;
  if (StringRef(score.globalCutoffCycles).getAsInteger(10, cutoff) ||
      cutoff > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  return candidates[4].cycles == static_cast<int64_t>(cutoff);
}

static bool parseScore(StringRef path, const InputManifest &input,
                       const VariantSpec &variant, const ClosureInfo &closure,
                       ScoreInfo &info, bool seedMode) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    scoreError(info, "cannot read score file " + path.str() + ": " +
                         buffer.getError().message());
    return false;
  }
  llvm::line_iterator lines(**buffer, true);
  bool sawHeader = false;
  bool sawFooter = false;
  unsigned lineNumber = 0;
  bool headerGlobalCutoffRequested = false;
  std::optional<json::Object> header;
  std::optional<json::Object> footer;
  std::set<std::string> candidateIds;
  for (; !lines.is_at_end(); ++lines) {
    ++lineNumber;
    llvm::Expected<json::Value> parsed = json::parse(*lines);
    if (!parsed) {
      scoreError(info, "invalid score JSON at line " +
                           std::to_string(lineNumber) + ": " +
                           llvm::toString(parsed.takeError()));
      continue;
    }
    const json::Object *object = parsed->getAsObject();
    if (!object) {
      scoreError(info, "score record is not an object");
      continue;
    }
    std::optional<StringRef> schema = object->getString("schema");
    if (!schema || *schema != kScoreSchema) {
      scoreError(info, "score record has the wrong schema");
      continue;
    }
    std::optional<StringRef> type = object->getString("record_type");
    if (!type) {
      scoreError(info, "score record has no record_type");
      continue;
    }
    if (*type == "header") {
      if (sawHeader || sawFooter) {
        scoreError(info, "score header is misplaced or repeated");
        continue;
      }
      sawHeader = true;
      header = *object;
      continue;
    }
    if (!sawHeader || sawFooter) {
      scoreError(info, "score record appears outside header/footer");
      continue;
    }
    if (*type == "footer") {
      sawFooter = true;
      footer = *object;
      continue;
    }
    if (*type != "score") {
      scoreError(info, "score file contains an unknown record type");
      continue;
    }
    std::optional<StringRef> candidateId = object->getString("candidate_id");
    std::optional<int64_t> cycles =
        object->getInteger("predicted_whole_program_cycles");
    if (!candidateId || candidateId->empty() || !cycles || *cycles <= 0) {
      scoreError(info, "score row has no positive candidate ID/cycle count");
      continue;
    }
    if (!candidateIds.insert(candidateId->str()).second) {
      scoreError(info, "score file repeats a candidate ID");
      continue;
    }
    if (seedMode &&
        (!object->getBoolean("valid").value_or(false) ||
         !object->getArray("task_costs") ||
         object->getArray("task_costs")->empty() ||
         !object->getArray("task_schedule") ||
         object->getArray("task_schedule")->empty() ||
         object->getString("score_source") !=
             std::optional<StringRef>(
                 "ml-ii-startup-plus-production-explicit-communication") ||
         object->getString("task_latency_provenance") !=
             std::optional<StringRef>(
                 "source-owned-ml-ii-startup-trip-count-contract"))) {
      // Seed rows are feasible evidence only after the source score pass has
      // emitted its complete cost/schedule record.  The downstream Score pass
      // replays these records against the actual graph/catalogue, while this
      // ranker rejects malformed or uncertified payloads before selection.
      scoreError(info, "seed score row lacks a complete feasible record");
      continue;
    }
    ScoreCandidate candidate{candidateId->str(), *cycles, path.str(),
                             static_cast<int64_t>(lineNumber), *object};
    info.candidates.push_back(std::move(candidate));
  }
  if (!sawHeader)
    scoreError(info, "score file is missing its header");
  if (!sawFooter || !footer)
    scoreError(info, "score file is missing its footer");
  if (header) {
    info.seedOnly = header->getBoolean("seed_only").value_or(false);
    if (info.seedOnly && !seedMode)
      scoreError(info, "score seed-only marker disagrees with ranker mode");
    std::optional<StringRef> graphId = header->getString("graph_variant_id");
    std::optional<StringRef> function = header->getString("function");
    std::optional<StringRef> repository =
        header->getString("source_repository");
    std::optional<StringRef> commit = header->getString("source_commit");
    std::optional<StringRef> architecture =
        header->getString("architecture_path");
    std::optional<StringRef> sourceGraph = header->getString("source_graph_id");
    headerGlobalCutoffRequested =
        header->getBoolean("global_cutoff_requested").value_or(false);
    if (!graphId || *graphId != variant.graphVariantId)
      scoreError(info, "score graph_variant_id does not match the manifest");
    if (!function || *function != input.function)
      scoreError(info, "score function does not match the manifest");
    if (!repository || *repository != input.sourceRepository)
      scoreError(info, "score source repository does not match the manifest");
    if (!commit || *commit != input.sourceCommit)
      scoreError(info, "score source commit does not match the manifest");
    if (!architecture || *architecture != input.architecturePath)
      scoreError(info, "score architecture path does not match the manifest");
    if (sourceGraph) {
      if (*sourceGraph != variant.graphVariantId)
        scoreError(info, "score source_graph_id does not match the graph variant");
    } else if (info.seedOnly || headerGlobalCutoffRequested) {
      scoreError(info,
                 "seed/global-cutoff score has no source_graph_id binding");
    }
    if (std::optional<StringRef> stage = header->getString("stage")) {
      if (*stage != input.stage || *stage != variant.stage)
        scoreError(info, "score stage does not match the manifest");
    } else if (headerGlobalCutoffRequested || info.seedOnly) {
      scoreError(info,
                 "incomplete score has no C++ stage identity in its header");
    }
    if (std::optional<StringRef> communication =
            header->getString("communication_mode")) {
      if (*communication != "explicit")
        scoreError(info, "score was not produced with explicit communication");
    } else {
      scoreError(info, "score header has no communication_mode");
    }
    if (std::optional<StringRef> predictor =
            header->getString("predictor_source")) {
      if (*predictor != kPredictorSource)
        scoreError(info, "score predictor source is not source-owned ML");
      else
        info.predictorSource = predictor->str();
    } else {
      scoreError(info, "score header has no predictor_source");
    }
    if (std::optional<StringRef> architectureSchema =
            header->getString("architecture_schema")) {
      if (*architectureSchema != "neura-architecture-v1")
        scoreError(info, "score architecture schema is not source-owned Neura");
      else
        info.architectureSchema = architectureSchema->str();
    }
    if (std::optional<StringRef> costNamespace =
            header->getString("cost_namespace")) {
      if (costNamespace->empty())
        scoreError(info, "score cost namespace is empty");
      else
        info.costNamespace = costNamespace->str();
      if (header->getString("cost_provenance_schema") !=
          std::optional<StringRef>("orbit-cost-provenance-v1"))
        scoreError(info, "score cost provenance schema is missing or unknown");
    }
    const std::optional<StringRef> temporalModel =
        header->getString("temporal_model");
    const std::optional<StringRef> declaredScheduleSpace =
        header->getString("schedule_space");
    info.temporalModel = temporalModel ? temporalModel->str() : "";
    info.scheduleSpace =
        declaredScheduleSpace ? declaredScheduleSpace->str() : "";
    info.productionScheduler =
        (temporalModel && *temporalModel == kProductionTemporalModel) ||
        (declaredScheduleSpace &&
         *declaredScheduleSpace == kProductionScheduleSpace);
    if (info.productionScheduler &&
        (!temporalModel || *temporalModel != kProductionTemporalModel))
      scoreError(info,
                 "production scheduler score has no production temporal_model");
    if (temporalModel && !info.productionScheduler &&
        *temporalModel != "dispatch-order" && *temporalModel != "all-integer")
      scoreError(info, "score header has an unknown temporal_model");
    if (std::optional<bool> globalSpace =
            header->getBoolean("global_shape_location_temporal_space")) {
      info.globalShapeLocationTemporalSpace = *globalSpace;
      if (info.productionScheduler ? *globalSpace : !*globalSpace)
        scoreError(info, info.productionScheduler
                              ? "production scheduler score claims exact shape/location/temporal coverage"
                              : "score does not cover shape/location/temporal space");
    } else {
      scoreError(info, "score header has no global space marker");
    }
    if (std::optional<int64_t> topK = header->getInteger("top_k")) {
      if (*topK < 5)
        scoreError(info, "score top_k is smaller than five");
    } else {
      scoreError(info, "score header has no top_k");
    }
    info.placementCoverageLimited =
        header->getBoolean("placement_coverage_limited").value_or(false);
    if (info.productionScheduler) {
      info.locationCoverageLimited =
          header->getBoolean("location_coverage_limited")
              .value_or(info.placementCoverageLimited);
      info.temporalCoverageLimited =
          header->getBoolean("temporal_coverage_limited").value_or(true);
      if (!info.placementCoverageLimited || !info.locationCoverageLimited ||
          !info.temporalCoverageLimited)
        scoreError(info,
                   "production scheduler score must mark location and temporal coverage limited");
    }
    if (std::optional<StringRef> scheduleSpace =
            header->getString("schedule_space")) {
      info.scheduleSpace = scheduleSpace->str();
      if (info.productionScheduler) {
        if (*scheduleSpace != kProductionScheduleSpace)
          scoreError(info,
                     "production scheduler score has the wrong schedule_space");
      } else if (*scheduleSpace == "heuristic-placement")
        info.placementCoverageLimited = true;
      else if (*scheduleSpace == "exact" && info.placementCoverageLimited)
        scoreError(info,
                   "exact schedule_space cannot claim limited placement coverage");
      else if (*scheduleSpace == "order-derived" ||
               *scheduleSpace == "order-derived-dispatch")
        info.orderDerivedTemporal = true;
      else if (*scheduleSpace != "exact")
        scoreError(info, "score header has an unknown schedule_space");
    }
    if (info.productionScheduler && info.scheduleSpace.empty())
      scoreError(info, "production scheduler score has no schedule_space");
    if (info.productionScheduler &&
        header->getBoolean("orbit_4x4_pruning").value_or(true))
      scoreError(info,
                 "production scheduler score must disable orbit 4x4 pruning");
    if (info.productionScheduler &&
        header->getString("shape_domain_pruning") !=
            std::optional<StringRef>("dependency-release-tail-cost-only"))
      scoreError(info,
                 "production scheduler score must use the cost-only shape bound");
    std::optional<StringRef> temporalScope =
        header->getString("temporal_search_scope");
    std::optional<StringRef> startPolicy = header->getString("start_policy");
    std::optional<bool> cycleStarts =
        header->getBoolean("cycle_starts_are_search_dimensions");
    if (temporalScope)
      info.temporalSearchScope = temporalScope->str();
    if (startPolicy)
      info.startPolicy = startPolicy->str();
    if (std::optional<StringRef> dispatchPolicy =
            header->getString("dispatch_policy"))
      info.dispatchPolicy = dispatchPolicy->str();
    if (std::optional<StringRef> dispatchPolicyScope =
            header->getString("dispatch_policy_scope"))
      info.dispatchPolicyScope = dispatchPolicyScope->str();
    if (std::optional<StringRef> schedulerBackend =
            header->getString("scheduler_backend"))
      info.schedulerBackend = schedulerBackend->str();
    if (cycleStarts)
      info.cycleStartsAreSearchDimensions = *cycleStarts;
    // Stage-1 heuristic-placement scores already use
    // start_policy=single-greedy-placement.  Treat that legacy field as
    // temporal-contract input only when the order-derived schedule marker or
    // one of the new temporal fields is present.
    const bool anyTemporalContractField =
        info.orderDerivedTemporal || temporalScope || cycleStarts ||
        (startPolicy &&
         *startPolicy == "dependency-and-resource-ready");
    if (info.productionScheduler) {
      info.temporalContractPresent = true;
      if (!temporalScope || !startPolicy || !cycleStarts ||
          *temporalScope != kProductionTemporalScope ||
          *startPolicy != kProductionStartPolicy || *cycleStarts)
        scoreError(info,
                   "production scheduler score has an invalid temporal contract");
      info.dispatchPolicy =
          header->getString("dispatch_policy").value_or(StringRef()).str();
      info.dispatchPolicyScope =
          header->getString("dispatch_policy_scope")
              .value_or(StringRef())
              .str();
      info.schedulerBackend =
          header->getString("scheduler_backend").value_or(StringRef()).str();
      if (!isProductionDispatchPolicy(input.stage, info.dispatchPolicy) ||
          info.dispatchPolicyScope !=
              productionDispatchPolicyScope(info.dispatchPolicy).str())
        scoreError(info,
                   "production scheduler score has an invalid dispatch policy contract");
      if (info.schedulerBackend != kProductionSchedulerBackend)
        scoreError(info,
                   "production scheduler score has an unknown scheduler backend");
    } else if (anyTemporalContractField) {
      if (!temporalScope || !startPolicy || !cycleStarts)
        scoreError(info,
                   "temporal score header must provide all order-derived fields");
      else {
        info.temporalContractPresent = true;
        if (*temporalScope != "task-dispatch-order-only")
          scoreError(info, "score temporal_search_scope is not dispatch-order-only");
        const bool stageOneGreedy = input.stage == "shape-only" &&
            info.placementCoverageLimited &&
            header->getString("schedule_space") ==
                std::optional<StringRef>("heuristic-placement") &&
            *startPolicy == "single-greedy-placement";
        if (*startPolicy != "dependency-and-resource-ready" && !stageOneGreedy)
          scoreError(info, "score start_policy is not dependency-and-resource-ready");
        if (*cycleStarts)
          scoreError(info,
                     "order-derived score cannot search integer cycle starts");
      }
    }
    if (info.placementCoverageLimited && !info.productionScheduler &&
        input.stage != "shape-only")
      scoreError(info,
                 "heuristic placement coverage is allowed only for shape-only");
    info.candidateCount = decimalField(*header, "candidate_count").value_or("");
    if (info.candidateCount.empty() || isZero(info.candidateCount))
      scoreError(info, "score candidate_count is missing or zero");
  }
  if (footer) {
    std::optional<StringRef> status = footer->getString("status");
    std::optional<bool> incomplete = footer->getBoolean("incomplete");
    std::optional<bool> topCertified = footer->getBoolean("top_k_certified");
    std::optional<bool> horizon = footer->getBoolean("horizon_proven");
    std::optional<StringRef> footerCertificate =
        footer->getString("ranking_certificate");
    const bool globalCutoffComplete =
        footer->getBoolean("global_cutoff_complete").value_or(false);
    const bool seedFooter =
        status && *status == "seed-only" && incomplete && *incomplete &&
        topCertified && !*topCertified &&
        footer->getBoolean("seed_only").value_or(false) &&
        footer->getBoolean("feasible_only").value_or(false) &&
        footerCertificate ==
            std::optional<StringRef>("feasible-seed-only");
    info.globalCutoffComplete = globalCutoffComplete;
    const bool orderDerivedCertificate =
        footerCertificate &&
        *footerCertificate ==
            "proven-shape-location-dispatch-order-branch-and-bound";
    const bool productionCertificate =
        footerCertificate && *footerCertificate == kProductionCertificate;
    if (globalCutoffComplete) {
      if (!headerGlobalCutoffRequested)
        scoreError(info,
                   "global-cutoff score footer lacks the header request marker");
      if (!status || (*status != "globally-pruned" && *status != "complete") ||
          !incomplete || *incomplete ||
          !footer->getBoolean("global_cutoff_proven").value_or(false) ||
          (!footer->getBoolean("global_cutoff_omitted_candidates_strictly_worse")
                .value_or(false) &&
           !footer->getBoolean(
                "global_cutoff_omitted_candidates_stable_key_dominated")
                .value_or(false)) ||
          !footerCertificate ||
          *footerCertificate != "proven-global-incumbent-cycle-cutoff")
        scoreError(info, "global-cutoff score footer is incomplete or uncertified");
      std::optional<int64_t> cutoff =
          footer->getInteger("global_cutoff_cycles");
      if (!cutoff || *cutoff <= 0)
        scoreError(info, "global-cutoff score has no positive cutoff cycles");
      else
        info.globalCutoffCycles = std::to_string(*cutoff);
      std::optional<StringRef> evidence =
          footer->getString("global_cutoff_evidence_path");
      if (!evidence || evidence->empty())
        scoreError(info, "global-cutoff score has no evidence path");
      else
        info.globalCutoffEvidencePath = evidence->str();
      if (footer->getInteger("global_cutoff_evidence_candidate_count")
                  .value_or(0) < 5)
        scoreError(info,
                   "global-cutoff evidence has fewer than five candidates");
      if (footer->getString("global_cutoff_key_policy") !=
          std::optional<StringRef>(
              "predicted_whole_program_cycles,semantic_graph_id,candidate_id,graph_variant_id"))
        scoreError(info, "global-cutoff score has an unstable key policy");
      if (footer->getString("global_cutoff_evidence_function") !=
              std::optional<StringRef>(input.function) ||
          footer->getString("global_cutoff_evidence_stage") !=
              std::optional<StringRef>(input.stage) ||
          footer->getString("global_cutoff_evidence_source_repository") !=
              std::optional<StringRef>(input.sourceRepository) ||
          footer->getString("global_cutoff_evidence_source_commit") !=
              std::optional<StringRef>(input.sourceCommit) ||
          footer->getString("global_cutoff_evidence_architecture_path") !=
              std::optional<StringRef>(input.architecturePath))
        scoreError(info, "global-cutoff evidence provenance disagrees with manifest");
      if (!llvm::sys::fs::exists(info.globalCutoffEvidencePath))
        scoreError(info, "global-cutoff evidence path does not exist");
    } else if (seedFooter) {
      if (!seedMode || footer->getInteger("seed_feasible_count").value_or(-1) < 5 ||
          footer->getInteger("seed_feasible_count").value_or(-1) !=
              static_cast<int64_t>(info.candidates.size()))
        scoreError(info, "seed-only score footer does not cover its feasible rows");
    } else if (!status || *status != "complete" || !incomplete || *incomplete ||
               !topCertified || !*topCertified ||
               (!orderDerivedCertificate && !productionCertificate &&
                (!horizon || !*horizon))) {
      scoreError(info, "score footer is incomplete or uncertified");
    }
    if (seedMode && info.seedOnly && !seedFooter)
      scoreError(info, "seed-only ranker input has no feasible-seed footer");
    std::optional<std::string> footerCount =
        decimalField(*footer, "candidate_count");
    if (!footerCount || info.candidateCount.empty() ||
        *footerCount != info.candidateCount)
      scoreError(info, "score header/footer candidate counts disagree");
    std::optional<std::string> visited =
        decimalField(*footer, "shape_candidates_visited");
    if (!visited)
      scoreError(info, "score footer has no shape_candidates_visited");
    else
      info.shapeCandidatesVisited = *visited;
    std::optional<std::string> boundPruned =
        decimalField(*footer, "bound_pruned_shape_candidates");
    info.boundPrunedShapeCandidates = boundPruned.value_or("0");
    std::optional<std::string> unmeasured =
        decimalField(*footer, "unmeasured_shape_candidates");
    if (!unmeasured)
      scoreError(info, "score footer has no unmeasured_shape_candidates");
    else {
      info.unmeasuredShapeCandidates = *unmeasured;
      if (!isZero(*unmeasured) && !seedFooter)
        scoreError(info, "score has unmeasured shape candidates");
    }
    if (visited && unmeasured && !info.candidateCount.empty() &&
        addDecimal(addDecimal(*visited, info.boundPrunedShapeCandidates),
                   *unmeasured) != info.candidateCount)
      scoreError(info,
                 "score shape coverage does not partition the full factored space");
    std::optional<std::string> validShapes =
        decimalField(*footer, "valid_shape_candidates");
    if (!validShapes ||
        (isZero(*validShapes) && !globalCutoffComplete && !seedFooter))
      scoreError(info, "score has no valid shape candidates");
    if (std::optional<std::string> unsupported =
            decimalField(*footer, "unsupported_shape_candidates")) {
      if (!isZero(*unsupported))
        scoreError(info, "score has unsupported shape candidates");
    } else {
      scoreError(info, "score footer has no unsupported_shape_candidates");
    }
    if (std::optional<int64_t> topK = footer->getInteger("top_k_requested")) {
      if (*topK < 5)
        scoreError(info, "score footer top_k_requested is smaller than five");
    } else {
      scoreError(info, "score footer has no top_k_requested");
    }
    if (std::optional<bool> footerLimited =
            footer->getBoolean("placement_coverage_limited")) {
      if (*footerLimited != info.placementCoverageLimited)
        scoreError(info,
                   "score header/footer placement coverage markers disagree");
    } else if (info.placementCoverageLimited) {
      scoreError(info, "limited placement score footer has no coverage marker");
    }
    if (info.productionScheduler) {
      if (footer->getString("temporal_model") !=
              std::optional<StringRef>(kProductionTemporalModel) ||
          footer->getString("schedule_space") !=
              std::optional<StringRef>(kProductionScheduleSpace) ||
          footer->getBoolean("global_shape_location_temporal_space") !=
              std::optional<bool>(false) ||
          footer->getString("temporal_search_scope") !=
              std::optional<StringRef>(kProductionTemporalScope) ||
          footer->getString("start_policy") !=
              std::optional<StringRef>(kProductionStartPolicy) ||
          footer->getBoolean("cycle_starts_are_search_dimensions") !=
              std::optional<bool>(false) ||
          (footer->getBoolean("location_coverage_limited") &&
           !footer->getBoolean("location_coverage_limited").value()) ||
          (footer->getBoolean("temporal_coverage_limited") &&
           !footer->getBoolean("temporal_coverage_limited").value()) ||
          footer->getBoolean("orbit_4x4_pruning").value_or(true) ||
          footer->getString("shape_domain_pruning") !=
              std::optional<StringRef>("dependency-release-tail-cost-only") ||
          footer->getString("scheduler_backend") !=
              std::optional<StringRef>(kProductionSchedulerBackend) ||
          footer->getString("dispatch_policy") !=
              std::optional<StringRef>(info.dispatchPolicy) ||
          footer->getString("dispatch_policy_scope") !=
              std::optional<StringRef>(info.dispatchPolicyScope))
        scoreError(info,
                   "production scheduler score footer contract disagrees with its header");
    }
    if (footerCertificate) {
      info.rankingCertificate = footerCertificate->str();
      const bool exactCertificate =
          *footerCertificate == "proven-shape-and-schedule-branch-and-bound" ||
          *footerCertificate == "exhaustive-shape-plus-bounded-exact-schedule";
      const bool limitedCertificate =
          *footerCertificate == "proven-shape-ranking-at-single-heuristic-placement";
      if (seedFooter) {
        if (*footerCertificate != "feasible-seed-only")
          scoreError(info, "seed-only score has the wrong certificate");
      } else if (orderDerivedCertificate) {
        info.orderDerivedTemporal = true;
        if (info.placementCoverageLimited || input.stage == "shape-only")
          scoreError(info,
                     "order-derived temporal certificate is not valid for this stage");
        if (!info.temporalContractPresent)
          scoreError(info,
                     "order-derived score footer requires the temporal header contract");
      } else if (productionCertificate) {
        if (!info.productionScheduler || !info.temporalContractPresent ||
            !info.placementCoverageLimited ||
            !isZero(info.unmeasuredShapeCandidates))
          scoreError(info,
                     "production scheduler certificate lacks complete shape coverage");
      } else if (globalCutoffComplete) {
        if (*footerCertificate != "proven-global-incumbent-cycle-cutoff")
          scoreError(info, "global-cutoff score has the wrong certificate");
      } else if ((info.placementCoverageLimited && !limitedCertificate) ||
                 (!info.placementCoverageLimited && !exactCertificate)) {
        scoreError(info,
                   "score ranking certificate does not match placement coverage");
      }
    } else {
      scoreError(info, "score footer has no ranking_certificate");
    }
    if (std::optional<StringRef> footerScope =
            footer->getString("temporal_search_scope")) {
      if (info.temporalSearchScope.empty())
        info.temporalSearchScope = footerScope->str();
      else if (info.temporalSearchScope != *footerScope)
        scoreError(info, "score header/footer temporal scopes disagree");
    }
    if (std::optional<StringRef> footerStartPolicy =
            footer->getString("start_policy")) {
      if (info.startPolicy.empty())
        info.startPolicy = footerStartPolicy->str();
      else if (info.startPolicy != *footerStartPolicy)
        scoreError(info, "score header/footer start policies disagree");
    }
    if (std::optional<bool> footerCycleStarts =
            footer->getBoolean("cycle_starts_are_search_dimensions")) {
      if (!info.cycleStartsAreSearchDimensions)
        info.cycleStartsAreSearchDimensions = *footerCycleStarts;
      else if (*info.cycleStartsAreSearchDimensions != *footerCycleStarts)
        scoreError(info,
                   "score header/footer cycle-start search markers disagree");
    }
  }
  if (seedMode && info.seedOnly) {
    if (info.costNamespace.empty())
      scoreError(info, "seed-only score has no cost namespace");
    else if (!validateLegacyScoreCommand(path, variant.graphVariantId, input,
                                         closure, true, info.costNamespace))
      scoreError(info, "seed-only score command/catalogue binding is invalid");
  }
  if (info.candidates.empty() && !info.globalCutoffComplete)
    scoreError(info, "score file contains no readable score rows");
  info.valid = info.errors.empty();
  return info.valid;
}

static const ClosureGraph *findClosureGraph(const ClosureInfo &closure,
                                            StringRef path) {
  for (const ClosureGraph &graph : closure.graphs)
    if (graph.path == path)
      return &graph;
  return nullptr;
}

static bool bindClosureToVariants(const InputManifest &input,
                                  ClosureInfo &closure,
                                  std::vector<std::string> &errors,
                                  bool allowUnscoredGraphs) {
  if (!closure.valid)
    return false;
  std::map<std::string, std::string> semanticByStructural;
  std::map<std::string, std::string> structuralBySemantic;
  std::set<std::string> semanticIds;
  for (const VariantSpec &variant : input.variants) {
    if (variant.stage != input.stage)
      addError(errors, "graph variant stage does not match the manifest");
    std::set<std::string> variantStructural;
    for (const std::string &path : variant.rewritePaths) {
      const ClosureGraph *graph = findClosureGraph(closure, path);
      if (!graph) {
        addError(errors, "rewrite path is absent from graph closure: " + path);
        continue;
      }
      variantStructural.insert(graph->structuralText);
      if (!graph->semanticGraphId.empty() &&
          graph->semanticGraphId != variant.semanticGraphId)
        addError(errors, "closure semantic graph identity disagrees with manifest");
      if (!graph->graphVariantId.empty() &&
          graph->graphVariantId != variant.graphVariantId)
        addError(errors, "closure graph_variant_id disagrees with manifest");
      auto semanticIt = semanticByStructural.find(graph->structuralText);
      if (semanticIt != semanticByStructural.end() &&
          semanticIt->second != variant.semanticGraphId)
        addError(errors, "equivalent structural graphs have different semantic IDs");
      else
        semanticByStructural[graph->structuralText] = variant.semanticGraphId;
    }
    semanticIds.insert(variant.semanticGraphId);
    if (variantStructural.empty()) {
      addError(errors, "graph variant has no closure-bound rewrite path: " +
                           variant.graphVariantId);
      continue;
    }
    if (variantStructural.size() != 1)
      addError(errors, "one graph variant names multiple structural keys: " +
                           variant.graphVariantId);
    for (const std::string &structural : variantStructural) {
      auto structuralIt = structuralBySemantic.find(variant.semanticGraphId);
      if (structuralIt != structuralBySemantic.end() &&
          structuralIt->second != structural)
        addError(errors, "one semantic graph ID names multiple structural keys");
      else
        structuralBySemantic[variant.semanticGraphId] = structural;
    }
  }
  for (const ClosureGraph &graph : closure.graphs) {
    if (!allowUnscoredGraphs && !semanticByStructural.count(graph.structuralText))
      addError(errors, "graph closure contains an unscored semantic graph");
  }
  if (semanticIds.empty() || semanticIds.size() != structuralBySemantic.size())
    addError(errors, "graph closure has no score-bound semantic graph variants");
  return errors.empty();
}

static void addRankedCandidate(
    std::map<std::tuple<std::string, std::string, std::string>, size_t> &indices,
                               std::vector<RankedCandidate> &ranked,
                               const VariantSpec &variant,
                               const ScoreCandidate &score,
                               const ClosureInfo &closure) {
  std::tuple<std::string, std::string, std::string> key{
      variant.semanticGraphId, score.candidateId, variant.graphVariantId};
  auto found = indices.find(key);
  std::vector<std::string> rewritePaths = variant.rewritePaths;
  std::set<std::string> variantStructural;
  for (const std::string &path : variant.rewritePaths) {
    if (const ClosureGraph *graph = findClosureGraph(closure, path))
      variantStructural.insert(graph->structuralText);
  }
  for (const ClosureGraph &graph : closure.graphs) {
    bool sameSemantic = graph.semanticGraphId == variant.semanticGraphId;
    bool sameStructural = variantStructural.count(graph.structuralText) != 0;
    // Source-owned closure records currently carry structural_key rather than
    // semantic_graph_id. Include every action path that names the same
    // structural graph so mapper replay can choose a concrete materialized
    // path even when the closure has deduplicated aliases.
    if (!sameSemantic && !sameStructural)
      continue;
    appendUnique(rewritePaths, graph.path);
    appendUnique(rewritePaths, graph.firstPath);
  }
  if (found == indices.end()) {
    RankedCandidate candidate;
    candidate.semanticGraphId = variant.semanticGraphId;
    candidate.graphVariantId = variant.graphVariantId;
    candidate.candidateId = score.candidateId;
    candidate.cycles = score.cycles;
    candidate.scorePath = score.scorePath;
    candidate.sourceLine = score.sourceLine;
    candidate.scoreRecord = score.scoreRecord;
    candidate.mapperReplayPath = variant.mapperReplayPath;
    candidate.graphVariantIds.push_back(variant.graphVariantId);
    candidate.scorePaths.push_back(score.scorePath);
    candidate.rewritePaths = std::move(rewritePaths);
    candidate.mapperReplayPaths.push_back(variant.mapperReplayPath);
    ranked.push_back(std::move(candidate));
    indices.emplace(std::move(key), ranked.size() - 1);
    return;
  }
  RankedCandidate &candidate = ranked[found->second];
  appendUnique(candidate.graphVariantIds, variant.graphVariantId);
  appendUnique(candidate.scorePaths, score.scorePath);
  for (const std::string &path : rewritePaths)
    appendUnique(candidate.rewritePaths, path);
  appendUnique(candidate.mapperReplayPaths, variant.mapperReplayPath);
  if (score.cycles < candidate.cycles ||
      (score.cycles == candidate.cycles &&
       variant.graphVariantId < candidate.graphVariantId)) {
    candidate.graphVariantId = variant.graphVariantId;
    candidate.cycles = score.cycles;
    candidate.scorePath = score.scorePath;
    candidate.sourceLine = score.sourceLine;
    candidate.scoreRecord = score.scoreRecord;
    candidate.mapperReplayPath = variant.mapperReplayPath;
  }
}

static bool rankedLess(const RankedCandidate &lhs, const RankedCandidate &rhs) {
  if (lhs.cycles != rhs.cycles)
    return lhs.cycles < rhs.cycles;
  if (lhs.semanticGraphId != rhs.semanticGraphId)
    return lhs.semanticGraphId < rhs.semanticGraphId;
  int candidateOrder =
      ::mlir::amoeba::neura::joint_scheduling::compareProductionCandidateIds(
          lhs.candidateId, rhs.candidateId);
  if (candidateOrder != 0)
    return candidateOrder < 0;
  return lhs.graphVariantId < rhs.graphVariantId;
}

static void buildRanking(RankResult &result) {
  std::map<std::tuple<std::string, std::string, std::string>, size_t> indices;
  for (size_t index = 0; index < result.input.variants.size(); ++index) {
    const VariantSpec &variant = result.input.variants[index];
    if (index >= result.scores.size()) {
      result.unmeasuredGraphVariantCount =
          addDecimal(result.unmeasuredGraphVariantCount, "1");
      addError(result.errors, "score result is missing for graph " +
                               variant.graphVariantId);
      continue;
    }
    const ScoreInfo &score = result.scores[index];
    if (!score.valid)
      result.unmeasuredGraphVariantCount =
          addDecimal(result.unmeasuredGraphVariantCount, "1");
    if (score.globalCutoffComplete)
      result.globallyPrunedGraphVariantCount =
          addDecimal(result.globallyPrunedGraphVariantCount, "1");
    if (!result.temporalModelSeen) {
      result.temporalModelSeen = true;
      result.productionScheduler = score.productionScheduler;
      result.temporalModel = score.temporalModel;
      result.scheduleSpace = score.scheduleSpace;
      result.dispatchPolicy = score.dispatchPolicy;
      result.dispatchPolicyScope = score.dispatchPolicyScope;
      result.schedulerBackend = score.schedulerBackend;
      result.globalShapeLocationTemporalSpace =
          score.globalShapeLocationTemporalSpace;
    } else if (result.productionScheduler != score.productionScheduler ||
               result.temporalModel != score.temporalModel ||
               result.scheduleSpace != score.scheduleSpace ||
               result.dispatchPolicy != score.dispatchPolicy ||
               result.dispatchPolicyScope != score.dispatchPolicyScope ||
               result.schedulerBackend != score.schedulerBackend ||
               result.globalShapeLocationTemporalSpace !=
                   score.globalShapeLocationTemporalSpace) {
      addError(result.errors,
               "score temporal scheduling contracts disagree across graph variants");
    }
    result.placementCoverageLimited |= score.placementCoverageLimited;
    result.locationCoverageLimited |= score.locationCoverageLimited;
    result.temporalCoverageLimited |= score.temporalCoverageLimited;
    result.orderDerivedTemporal |= score.orderDerivedTemporal;
    if (!score.temporalSearchScope.empty()) {
      if (result.temporalSearchScope.empty())
        result.temporalSearchScope = score.temporalSearchScope;
      else if (result.temporalSearchScope != score.temporalSearchScope)
        addError(result.errors, "score temporal scopes disagree across graph variants");
    }
    if (!score.startPolicy.empty()) {
      if (result.startPolicy.empty())
        result.startPolicy = score.startPolicy;
      else if (result.startPolicy != score.startPolicy)
        addError(result.errors,
                 "score start policies disagree across graph variants");
    }
    if (score.cycleStartsAreSearchDimensions) {
      if (!result.cycleStartsAreSearchDimensions)
        result.cycleStartsAreSearchDimensions =
            score.cycleStartsAreSearchDimensions;
      else if (*result.cycleStartsAreSearchDimensions !=
               *score.cycleStartsAreSearchDimensions)
        addError(result.errors,
                 "score cycle-start search markers disagree across graph variants");
    }
    if (!score.candidateCount.empty())
      result.scoredCandidateCount = addDecimal(
          result.scoredCandidateCount, score.candidateCount);
    if (!score.shapeCandidatesVisited.empty())
      result.visitedShapeCandidateCount = addDecimal(
          result.visitedShapeCandidateCount, score.shapeCandidatesVisited);
    if (!score.boundPrunedShapeCandidates.empty())
      result.boundPrunedShapeCandidateCount = addDecimal(
          result.boundPrunedShapeCandidateCount,
          score.boundPrunedShapeCandidates);
    if (!score.unmeasuredShapeCandidates.empty())
      result.unmeasuredCandidateCount = addDecimal(
          result.unmeasuredCandidateCount, score.unmeasuredShapeCandidates);
    if (score.valid || !result.seedOnly)
      for (const ScoreCandidate &candidate : score.candidates)
        addRankedCandidate(indices, result.candidates, variant, candidate,
                           result.closure);
    if (!score.costNamespace.empty()) {
      if (result.costNamespace.empty())
        result.costNamespace = score.costNamespace;
      else if (result.costNamespace != score.costNamespace)
        addError(result.errors, "score cost namespaces disagree across graph variants");
    }
    if (!score.architectureSchema.empty()) {
      if (result.architectureSchema.empty())
        result.architectureSchema = score.architectureSchema;
      else if (result.architectureSchema != score.architectureSchema)
        addError(result.errors,
                 "score architecture schemas disagree across graph variants");
    }
  }
  std::sort(result.candidates.begin(), result.candidates.end(), rankedLess);
  std::set<std::string> semanticIds;
  std::map<std::string, std::string> representativeCount;
  for (size_t index = 0; index < result.input.variants.size(); ++index) {
    const VariantSpec &variant = result.input.variants[index];
    semanticIds.insert(variant.semanticGraphId);
    if (index >= result.scores.size() ||
        result.scores[index].candidateCount.empty())
      continue;
    auto found = representativeCount.find(variant.semanticGraphId);
    if (found == representativeCount.end()) {
      representativeCount.emplace(variant.semanticGraphId,
                                  result.scores[index].candidateCount);
    } else if (found->second != result.scores[index].candidateCount) {
      addError(result.errors,
               "equivalent semantic graph variants disagree on candidate_count");
    }
  }
  std::string declared = "0";
  for (const auto &entry : representativeCount)
    declared = addDecimal(declared, entry.second);
  // Keep the number available to the output writer without making it part of
  // the ranking decision. A missing score means it is intentionally unknown.
  if (!result.seedOnly && representativeCount.size() != semanticIds.size())
    addError(result.errors,
             "at least one semantic graph has no complete score count");
  bool allScoresValid = result.scores.size() == result.input.variants.size();
  for (const ScoreInfo &score : result.scores)
    allScoresValid &= score.valid;
  if (result.productionScheduler) {
    if (result.temporalModel != kProductionTemporalModel ||
        result.scheduleSpace != kProductionScheduleSpace ||
        result.temporalSearchScope != kProductionTemporalScope ||
        result.startPolicy != kProductionStartPolicy ||
        !result.cycleStartsAreSearchDimensions ||
        *result.cycleStartsAreSearchDimensions ||
        !result.placementCoverageLimited || !result.locationCoverageLimited ||
        !result.temporalCoverageLimited ||
        result.globalShapeLocationTemporalSpace ||
        result.schedulerBackend != kProductionSchedulerBackend ||
        !isProductionDispatchPolicy(result.input.stage,
                                     result.dispatchPolicy) ||
        result.dispatchPolicyScope !=
            productionDispatchPolicyScope(result.dispatchPolicy))
      addError(result.errors,
               "production scheduler global ranking lacks its coverage contract");
    for (const ScoreInfo &score : result.scores)
      if (!score.productionScheduler || !score.temporalContractPresent)
        addError(result.errors,
                 "production scheduler global ranking mixes a legacy score");
  } else if (result.orderDerivedTemporal) {
    if (result.temporalSearchScope != "task-dispatch-order-only" ||
        result.startPolicy != "dependency-and-resource-ready" ||
        !result.cycleStartsAreSearchDimensions ||
        *result.cycleStartsAreSearchDimensions)
      addError(result.errors,
               "order-derived global ranking lacks its temporal contract");
    for (const ScoreInfo &score : result.scores)
      if (!score.temporalContractPresent)
        addError(result.errors,
                 "order-derived global ranking mixes a score without the temporal contract");
  }
  if (!result.seedOnly && !isZero(result.unmeasuredGraphVariantCount))
    addError(result.errors,
             "complete global ranking has unmeasured graph variants");
  result.complete = !result.seedOnly && result.errors.empty() && allScoresValid &&
                    result.closure.valid && result.candidates.size() >= 5 &&
                    result.input.variants.size() == result.scores.size();
  (void)declared;
}

static bool writeOutput(StringRef path, const RankResult &result,
                        std::vector<RankedCandidate> selected,
                        std::string &error) {
  std::error_code ec;
  llvm::raw_fd_ostream stream(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    error = "cannot write global stage ranking " + path.str() + ": " +
            ec.message();
    return false;
  }
  json::Object header;
  header["schema"] = kOutputSchema.str();
  header["record_type"] = "header";
  header["status"] = result.seedOnly ? "seed-only"
                                     : result.complete ? "complete" : "incomplete";
  header["complete"] = result.complete;
  header["stage_certified"] = result.complete;
  header["seed_only"] = result.seedOnly;
  header["feasible_only"] = result.seedOnly;
  header["top_k_requested"] = 5;
  header["source_repository"] = result.input.sourceRepository;
  header["source_commit"] = result.input.sourceCommit;
  header["architecture_path"] = result.input.architecturePath;
  if (!result.input.architectureId.empty())
    header["architecture_id"] = result.input.architectureId;
  header["function"] = result.input.function;
  header["stage"] = result.input.stage;
  header["graph_closure_record_path"] = result.input.closurePath;
  putDecimal(header, "graph_variant_count",
             std::to_string(result.input.variants.size()));
  std::set<std::string> semanticIds;
  for (const VariantSpec &variant : result.input.variants)
    semanticIds.insert(variant.semanticGraphId);
  putDecimal(header, "semantic_graph_count",
             std::to_string(semanticIds.size()));
  header["ranking_scope"] = "global-across-all-closure-graphs";
  header["ranking_objective"] = "predicted_whole_program_cycles";
  if (result.seedOnly && selected.size() >= 5)
    header["seed_cutoff_cycles"] = selected[4].cycles;
  header["predictor_source"] = kPredictorSource.str();
  if (!result.architectureSchema.empty())
    header["architecture_schema"] = result.architectureSchema;
  if (!result.costNamespace.empty()) {
    header["cost_namespace"] = result.costNamespace;
    header["cost_provenance_schema"] = "orbit-cost-provenance-v1";
  }
  header["model_status"] = "exploratory";
  header["production_ready"] = false;
  header["placement_coverage_limited"] = result.placementCoverageLimited;
  if (!result.temporalModel.empty())
    header["temporal_model"] = result.temporalModel;
  if (!result.scheduleSpace.empty())
    header["schedule_space"] = result.scheduleSpace;
  if (!result.dispatchPolicy.empty())
    header["dispatch_policy"] = result.dispatchPolicy;
  if (!result.dispatchPolicyScope.empty())
    header["dispatch_policy_scope"] = result.dispatchPolicyScope;
  if (!result.schedulerBackend.empty())
    header["scheduler_backend"] = result.schedulerBackend;
  if (result.productionScheduler) {
    header["temporal_model"] = kProductionTemporalModel.str();
    header["schedule_space"] = kProductionScheduleSpace.str();
    header["global_shape_location_temporal_space"] = false;
    header["location_coverage_limited"] = result.locationCoverageLimited;
    header["temporal_coverage_limited"] = result.temporalCoverageLimited;
    header["scheduler_backend"] = kProductionSchedulerBackend.str();
    header["orbit_4x4_pruning"] = false;
    header["shape_domain_pruning"] =
        "dependency-release-tail-cost-only";
    header["dispatch_policy"] = result.dispatchPolicy;
    header["dispatch_policy_scope"] = result.dispatchPolicyScope;
    header["tie_key_policy"] =
        ::mlir::amoeba::neura::joint_scheduling::
            kProductionNumericTieKeyPolicy.str();
  }
  putDecimal(header, "globally_pruned_graph_variant_count",
             result.globallyPrunedGraphVariantCount);
  header["placement_coverage_policy"] =
      result.productionScheduler
          ? "single-production-scheduler-placement-and-order"
          : result.placementCoverageLimited ? "single-heuristic-placement"
                                      : "exact-4x4-location-temporal";
  if (result.productionScheduler || result.orderDerivedTemporal ||
      !result.temporalSearchScope.empty() || !result.startPolicy.empty()) {
    header["temporal_search_scope"] =
        result.temporalSearchScope.empty()
            ? (result.productionScheduler ? kProductionTemporalScope.str()
                                          : "task-dispatch-order-only")
            : result.temporalSearchScope;
    header["start_policy"] = result.startPolicy.empty()
                                  ? (result.productionScheduler
                                         ? kProductionStartPolicy.str()
                                         : "dependency-and-resource-ready")
                                  : result.startPolicy;
    header["cycle_starts_are_search_dimensions"] =
        result.cycleStartsAreSearchDimensions.value_or(false);
  }
  putDecimal(header, "scored_candidate_count", result.scoredCandidateCount);
  putDecimal(header, "shape_candidates_visited",
              result.visitedShapeCandidateCount);
  putDecimal(header, "bound_pruned_shape_candidates",
              result.boundPrunedShapeCandidateCount);
  putDecimal(header, "unmeasured_candidate_count",
              result.unmeasuredCandidateCount);
  stream << jsonLine(std::move(header));

  for (size_t rank = 0; rank < selected.size(); ++rank) {
    RankedCandidate &candidate = selected[rank];
    sortUnique(candidate.graphVariantIds);
    sortUnique(candidate.scorePaths);
    sortUnique(candidate.rewritePaths);
    json::Object row;
    row["schema"] = kOutputSchema.str();
    row["record_type"] = "selection";
    row["rank"] = static_cast<int64_t>(rank);
    row["certified"] = result.complete;
    row["feasible"] = result.seedOnly || result.complete;
    row["seed_only"] = result.seedOnly;
    row["semantic_graph_id"] = candidate.semanticGraphId;
    row["graph_variant_id"] = candidate.graphVariantId;
    row["candidate_id"] = candidate.candidateId;
    row["predicted_whole_program_cycles"] = candidate.cycles;
    row["score_file_path"] = candidate.scorePath;
    row["score_record_line"] = candidate.sourceLine;
    row["mapper_replay_path"] = candidate.mapperReplayPath;
    row["placement_coverage_limited"] = result.placementCoverageLimited;
    if (result.productionScheduler) {
      row["temporal_model"] = kProductionTemporalModel.str();
      row["schedule_space"] = kProductionScheduleSpace.str();
      row["location_coverage_limited"] = result.locationCoverageLimited;
      row["temporal_coverage_limited"] = result.temporalCoverageLimited;
    }
    row["replay_record_schema"] = kScoreSchema.str();
    row["score_record"] = json::Value(json::Object(candidate.scoreRecord));
    json::Array graphIds;
    for (const std::string &graphId : candidate.graphVariantIds)
      graphIds.push_back(graphId);
    row["graph_variant_ids"] = std::move(graphIds);
    json::Array scorePaths;
    for (const std::string &scorePath : candidate.scorePaths)
      scorePaths.push_back(scorePath);
    row["score_file_paths"] = std::move(scorePaths);
    json::Array rewritePaths;
    for (const std::string &rewritePath : candidate.rewritePaths)
      rewritePaths.push_back(rewritePath);
    row["rewrite_paths"] = std::move(rewritePaths);
    json::Array mapperReplayPaths;
    for (const std::string &mapperReplayPath : candidate.mapperReplayPaths)
      mapperReplayPaths.push_back(mapperReplayPath);
    row["mapper_replay_paths"] = std::move(mapperReplayPaths);
    stream << jsonLine(std::move(row));
  }

  json::Object footer;
  footer["schema"] = kOutputSchema.str();
  footer["record_type"] = "footer";
  footer["status"] = result.seedOnly ? "seed-only"
                                     : result.complete ? "complete" : "incomplete";
  footer["complete"] = result.complete;
  footer["incomplete"] = !result.complete;
  footer["stage_certified"] = result.complete;
  footer["top_k_certified"] = result.complete;
  footer["seed_only"] = result.seedOnly;
  footer["feasible_only"] = result.seedOnly;
  footer["top_k_requested"] = 5;
  putDecimal(footer, "selected_count", std::to_string(selected.size()));
  putDecimal(footer, "candidate_rows_considered",
             std::to_string(result.candidates.size()));
  putDecimal(footer, "seed_feasible_count",
             result.seedOnly ? std::to_string(selected.size()) : "0");
  footer["unmeasured_candidate_count"] = result.unmeasuredCandidateCount;
  footer["scored_candidate_count"] = result.scoredCandidateCount;
  footer["shape_candidates_visited"] = result.visitedShapeCandidateCount;
  footer["bound_pruned_shape_candidates"] =
      result.boundPrunedShapeCandidateCount;
  footer["placement_coverage_limited"] = result.placementCoverageLimited;
  footer["placement_coverage_policy"] =
      result.productionScheduler
          ? "single-production-scheduler-placement-and-order"
          : result.placementCoverageLimited ? "single-heuristic-placement"
                                      : "exact-4x4-location-temporal";
  if (result.productionScheduler) {
    footer["temporal_model"] = kProductionTemporalModel.str();
    footer["schedule_space"] = kProductionScheduleSpace.str();
    footer["global_shape_location_temporal_space"] = false;
    footer["location_coverage_limited"] = result.locationCoverageLimited;
    footer["temporal_coverage_limited"] = result.temporalCoverageLimited;
    footer["scheduler_backend"] = kProductionSchedulerBackend.str();
    footer["orbit_4x4_pruning"] = false;
    footer["shape_domain_pruning"] =
        "dependency-release-tail-cost-only";
    footer["dispatch_policy"] = result.dispatchPolicy;
    footer["dispatch_policy_scope"] = result.dispatchPolicyScope;
    footer["tie_key_policy"] =
        ::mlir::amoeba::neura::joint_scheduling::
            kProductionNumericTieKeyPolicy.str();
  }
  if (result.productionScheduler || result.orderDerivedTemporal ||
      !result.temporalSearchScope.empty() || !result.startPolicy.empty()) {
    footer["temporal_search_scope"] =
        result.temporalSearchScope.empty()
            ? (result.productionScheduler ? kProductionTemporalScope.str()
                                          : "task-dispatch-order-only")
            : result.temporalSearchScope;
    footer["start_policy"] = result.startPolicy.empty()
                                  ? (result.productionScheduler
                                         ? kProductionStartPolicy.str()
                                         : "dependency-and-resource-ready")
                                  : result.startPolicy;
    footer["cycle_starts_are_search_dimensions"] =
        result.cycleStartsAreSearchDimensions.value_or(false);
  }
  footer["model_status"] = "exploratory";
  if (!result.temporalModel.empty())
    footer["temporal_model"] = result.temporalModel;
  if (!result.scheduleSpace.empty())
    footer["schedule_space"] = result.scheduleSpace;
  if (!result.dispatchPolicy.empty())
    footer["dispatch_policy"] = result.dispatchPolicy;
  if (!result.dispatchPolicyScope.empty())
    footer["dispatch_policy_scope"] = result.dispatchPolicyScope;
  if (!result.schedulerBackend.empty())
    footer["scheduler_backend"] = result.schedulerBackend;
  if (!result.architectureSchema.empty())
    footer["architecture_schema"] = result.architectureSchema;
  if (!result.costNamespace.empty()) {
    footer["cost_namespace"] = result.costNamespace;
    footer["cost_provenance_schema"] = "orbit-cost-provenance-v1";
  }
  footer["production_ready"] = false;
  putDecimal(footer, "unmeasured_graph_variant_count",
             result.unmeasuredGraphVariantCount);
  putDecimal(footer, "globally_pruned_graph_variant_count",
             result.globallyPrunedGraphVariantCount);
  putDecimal(footer, "unselected_candidate_rows",
             std::to_string(result.candidates.size() >= selected.size()
                                ? result.candidates.size() - selected.size()
                                : 0));
  footer["ranking_certificate"] =
      result.seedOnly
          ? "feasible-seed-only"
          : result.complete
          ? (result.productionScheduler
                 ? "global-proven-shape-space-under-production-scheduler-top-five"
                 : result.placementCoverageLimited
                 ? "global-proven-shape-ranking-at-single-heuristic-placement-top-five"
                 : result.orderDerivedTemporal
                       ? "global-proven-shape-location-dispatch-order-branch-and-bound-top-five"
                 : "global-proven-shape-and-schedule-branch-and-bound-top-five")
          : "best-found-diagnostic-only";
  footer["ranking_objective"] = "predicted_whole_program_cycles";
  if (result.seedOnly && selected.size() >= 5)
    footer["seed_cutoff_cycles"] = selected[4].cycles;
  json::Array errors;
  for (const std::string &message : result.errors)
    errors.push_back(message);
  for (const ScoreInfo &score : result.scores)
    for (const std::string &message : score.errors)
      if (errors.size() < 32)
        errors.push_back(message);
  for (const std::string &message : result.closure.errors)
    if (errors.size() < 32)
      errors.push_back(message);
  footer["diagnostics"] = std::move(errors);
  stream << jsonLine(std::move(footer));
  stream.flush();
  if (stream.has_error()) {
    error = "failed while writing global stage ranking " + path.str();
    return false;
  }
  return true;
}

class GlobalStageRankerPass
    : public PassWrapper<GlobalStageRankerPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GlobalStageRankerPass)

  GlobalStageRankerPass() = default;
  GlobalStageRankerPass(const GlobalStageRankerPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "select-global-stage-top-five";
  }
  StringRef getDescription() const override {
    return "Select one certified global predicted top five across graph variants";
  }

  Option<std::string> manifestFile{*this, "manifest", llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output", llvm::cl::init("")};
  Option<bool> incumbentOnly{*this, "incumbent-only", llvm::cl::init(false)};
  Option<int64_t> diagnosticLimit{*this, "max-diagnostics",
                                  llvm::cl::init(5)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (manifestFile.getValue().empty() || outputFile.getValue().empty() ||
        diagnosticLimit.getValue() <= 0) {
      module.emitError("global stage ranker requires manifest, output, and a "
                       "positive max-diagnostics");
      return signalPassFailure();
    }

    RankResult result;
    result.diagnosticLimit =
        static_cast<unsigned>(diagnosticLimit.getValue());
    // Output is interpreted by the driver, while alias checks use one
    // normalized spelling before any output is opened.
    const std::string normalizedOutputPath =
        normalizedPath(outputFile.getValue());
    bool manifestValid = parseManifest(manifestFile.getValue(), result.input,
                                       result.errors);
    if (manifestValid) {
      result.closure.valid = false;
      parseClosure(result.input.closurePath, result.input, result.closure,
                   incumbentOnly.getValue());
      bindClosureToVariants(result.input, result.closure, result.errors,
                            incumbentOnly.getValue());
      result.scores.reserve(result.input.variants.size());
      for (const VariantSpec &variant : result.input.variants) {
        ScoreInfo score;
        parseScore(variant.scorePath, result.input, variant, result.closure,
                   score, incumbentOnly.getValue());
        if (score.globalCutoffComplete &&
            !validateGlobalCutoffEvidence(score, result.input,
                                          result.closure))
          scoreError(
              score,
              "global-cutoff evidence is not a matching complete C++ top-five");
        result.scores.push_back(std::move(score));
      }
      result.seedOnly = incumbentOnly.getValue();
      buildRanking(result);
    } else {
      result.closure.errors.push_back("manifest validation failed");
    }

    // Never overwrite an input artifact. This check happens after parsing the
    // manifest so relative paths are compared in the same namespace as output.
    bool outputAliasesInput = false;
    if (normalizedPath(manifestFile.getValue()) == normalizedOutputPath ||
        (!result.input.closurePath.empty() &&
         result.input.closurePath == normalizedOutputPath)) {
      addError(result.errors, "output path aliases an input artifact");
      result.complete = false;
      outputAliasesInput = true;
    }
    for (const VariantSpec &variant : result.input.variants) {
      if (variant.scorePath == outputFile.getValue() ||
          variant.scorePath == normalizedOutputPath ||
          variant.mapperReplayPath == normalizedOutputPath) {
        addError(result.errors, "output path aliases a graph variant artifact");
        result.complete = false;
        outputAliasesInput = true;
      }
    }

    if (outputAliasesInput) {
      module.emitError() << "global stage ranker refuses to overwrite an "
                            "input artifact";
      return signalPassFailure();
    }
    std::vector<RankedCandidate> selected = result.candidates;
    if (selected.size() > result.diagnosticLimit)
      selected.resize(result.diagnosticLimit);
    if (incumbentOnly.getValue() && selected.size() > 5)
      selected.resize(5);
    if (result.complete) {
      if (selected.size() > 5)
        selected.resize(5);
      if (selected.size() != 5) {
        result.complete = false;
        addError(result.errors, "global top five has fewer than five entries");
      }
    }
    std::string outputError;
    const size_t selectedCount = selected.size();
    if (!writeOutput(outputFile.getValue(), result, selected,
                     outputError)) {
      module.emitError() << outputError;
      return signalPassFailure();
    }
    if (incumbentOnly.getValue() && result.closure.valid &&
        selectedCount >= 5 && !outputAliasesInput) {
      return;
    }
    if (!result.complete) {
      StringRef reason = result.errors.empty()
                             ? StringRef("incomplete graph or score artifact")
                             : StringRef(result.errors.front());
      module.emitError() << "global stage ranker is incomplete: " << reason;
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::amoeba::neura::createGlobalStageRankerPass() {
  return std::make_unique<GlobalStageRankerPass>();
}
