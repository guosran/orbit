//===- AnalyticalTaskCandidateManifest.cpp - Candidate JSONL reader ------===//
//
// Implements strict parsing and validation of static task-shape manifests.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateManifest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir;

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

// Reads a required JSON string and reports a field-specific error.
static std::optional<StringRef> requiredString(const llvm::json::Object &object,
                                               StringRef key,
                                               std::string &error) {
  std::optional<StringRef> value = object.getString(key);
  if (!value) {
    error = "missing or invalid string field \"" + key.str() + "\"";
  }
  return value;
}

// Reads a required JSON integer and reports a field-specific error.
static std::optional<int64_t> requiredInteger(const llvm::json::Object &object,
                                              StringRef key,
                                              std::string &error) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value) {
    error = "missing or invalid integer field \"" + key.str() + "\"";
  }
  return value;
}

// Accepts an exact nonnegative count encoded as an LLVM JSON integer when it
// fits int64_t, or as a decimal string for spaces larger than uint64_t. The
// returned value is normalized so header/footer comparisons are deterministic.
static std::optional<std::string>
requiredDecimalCount(const llvm::json::Object &object, StringRef key,
                    std::string &error) {
  if (std::optional<int64_t> numeric = object.getInteger(key)) {
    if (*numeric < 0) {
      error = "count field \"" + key.str() + "\" must be nonnegative";
      return std::nullopt;
    }
    return std::to_string(*numeric);
  }
  std::optional<StringRef> text = object.getString(key);
  if (!text || text->empty() ||
      llvm::any_of(*text, [](char character) {
        return character < '0' || character > '9';
      })) {
    error = "missing or invalid decimal count field \"" + key.str() +
            "\"";
    return std::nullopt;
  }
  StringRef normalized = text->ltrim('0');
  return (normalized.empty() ? StringRef("0") : normalized).str();
}

static int compareDecimalCounts(StringRef lhs, StringRef rhs) {
  if (lhs.size() != rhs.size())
    return lhs.size() < rhs.size() ? -1 : 1;
  if (lhs == rhs)
    return 0;
  return lhs < rhs ? -1 : 1;
}

// Reads the concrete trip count required by the static-shape manifest.
static bool parseTripCount(const llvm::json::Object &object, int64_t &tripCount,
                           std::string &error) {
  if (object.get("trip_count_kind")) {
    error = "symbol-dynamic trip counts are unsupported by static shapes";
    return false;
  }
  std::optional<int64_t> numeric = requiredInteger(object, "trip_count", error);
  if (!numeric || *numeric <= 0) {
    if (numeric)
      error = "trip_count must be positive";
    return false;
  }
  tripCount = *numeric;
  return true;
}

// Multiplies two positive counts while detecting overflow before the product.
static std::optional<int64_t> checkedPositiveProduct(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs) {
    return std::nullopt;
  }
  return lhs * rhs;
}

// Parses the manifest header, including the explicit architecture
// dimensions that describe how physical CGRAs map to tiles.
static bool parseHeader(const llvm::json::Object &object,
                        ManifestHeader &header, std::string &error) {
  std::optional<StringRef> schema = requiredString(object, "schema", error);
  std::optional<StringRef> function = requiredString(object, "function", error);
  std::optional<StringRef> scope =
      requiredString(object, "search_scope", error);
  std::optional<StringRef> policy =
      requiredString(object, "shape_policy", error);
  std::optional<StringRef> capacityPolicy =
      requiredString(object, "spatial_capacity_policy", error);
  std::optional<StringRef> pruningPolicy =
      requiredString(object, "shape_pruning_policy", error);
  auto maxChangedTasks = requiredInteger(object, "max_changed_tasks", error);
  auto unrestrictedCount =
      requiredDecimalCount(object, "unrestricted_candidate_count", error);
  auto declaredCount =
      requiredDecimalCount(object, "declared_candidate_count", error);
  auto excludedCount =
      requiredDecimalCount(object, "excluded_candidate_count", error);
  auto unmeasuredCount =
      requiredDecimalCount(object, "unmeasured_candidate_count", error);
  const llvm::json::Object *architecture = object.getObject("architecture");
  if (!schema || !function || !scope || !policy || !capacityPolicy ||
      !pruningPolicy || !maxChangedTasks || !unrestrictedCount ||
      !declaredCount || !excludedCount || !unmeasuredCount || !architecture) {
    return false;
  }
  if (*schema != kCandidateSchema || *policy != kShapePolicy ||
      *capacityPolicy != kSpatialCapacityPolicy) {
    error = "unsupported candidate manifest contract";
    return false;
  }
  StringRef expectedScope =
      *pruningPolicy == kShapePruningPolicyNone
          ? StringRef(kSearchScope)
          : *pruningPolicy == kShapePruningPolicyBoundedChangesFromUnit
                ? StringRef(kBoundedShapeSearchScope)
                : StringRef();
  if (expectedScope.empty() || *scope != expectedScope ||
      (*pruningPolicy != kShapePruningPolicyNone &&
       *pruningPolicy != kShapePruningPolicyBoundedChangesFromUnit)) {
    error = "unsupported candidate shape-pruning policy or search scope";
    return false;
  }
  if ((*pruningPolicy == kShapePruningPolicyNone && *maxChangedTasks != -1) ||
      (*pruningPolicy == kShapePruningPolicyBoundedChangesFromUnit &&
       *maxChangedTasks < 0)) {
    error = "candidate manifest max_changed_tasks does not match its policy";
    return false;
  }
  if (compareDecimalCounts(*declaredCount, *unrestrictedCount) > 0 ||
      *excludedCount != *unmeasuredCount) {
    error = "candidate manifest candidate-space counts are inconsistent";
    return false;
  }

  const llvm::json::Object *budget = object.getObject("candidate_budget");
  if (!budget) {
    error = "candidate manifest is missing its enumeration budget";
    return false;
  }
  auto budgetPolicy = requiredString(*budget, "policy", error);
  auto maxCandidates = requiredInteger(*budget, "max_candidates", error);
  std::optional<bool> complete = budget->getBoolean("complete");
  if (!budgetPolicy || !maxCandidates || !complete)
    return false;
  if (*budgetPolicy != kSpatialCandidateBudgetPolicy || !*complete ||
      *maxCandidates <= 0) {
    error = "candidate budget must be positive, complete, and "
            "fail-before-publish";
    return false;
  }

  auto gridRows = requiredInteger(*architecture, "grid_rows", error);
  auto gridCols = requiredInteger(*architecture, "grid_cols", error);
  auto perRows = requiredInteger(*architecture, "per_cgra_tile_rows", error);
  auto perCols = requiredInteger(*architecture, "per_cgra_tile_cols", error);
  // File-content fingerprints are outside the candidate identity contract.
  // Reject them instead of silently accepting a legacy manifest whose
  // provenance cannot be represented by Git and semantic IDs.
  if (architecture->get("spec_sha256") || object.get("graph_variant_sha256")) {
    error = "legacy SHA-256 candidate metadata is unsupported";
    return false;
  }
  auto maxCgras = requiredInteger(object, "max_cgras_per_task", error);
  if (!gridRows || !gridCols || !perRows || !perCols || !maxCgras) {
    return false;
  }
  std::optional<StringRef> graphVariantId =
      object.getString("graph_variant_id");
  header.function = function->str();
  header.graphVariantId = graphVariantId ? graphVariantId->str() : std::string();
  header.searchScope = scope->str();
  header.shapePruningPolicy = pruningPolicy->str();
  header.unrestrictedCandidateCount = *unrestrictedCount;
  header.declaredCandidateCount = *declaredCount;
  header.excludedCandidateCount = *excludedCount;
  header.unmeasuredCandidateCount = *unmeasuredCount;
  header.maxChangedTasks = *maxChangedTasks;
  header.gridRows = *gridRows;
  header.gridCols = *gridCols;
  header.perCgraRows = *perRows;
  header.perCgraCols = *perCols;
  header.maxCgrasPerTask = *maxCgras;
  header.maxCandidates = *maxCandidates;
  if (header.gridRows <= 0 || header.gridCols <= 0 || header.perCgraRows <= 0 ||
      header.perCgraCols <= 0 || header.maxCgrasPerTask <= 0) {
    error = "candidate manifest dimensions must be positive";
    return false;
  }
  if (!header.graphVariantId.empty() &&
      StringRef(header.graphVariantId).trim().empty()) {
    error = "candidate manifest graph variant identity is invalid";
    return false;
  }
  return true;
}

// Bind every saved shape choice to the named source task and its static
// trip count. Task names are the source task IDs; the source Git commit and
// graph ID are carried by the surrounding provenance contract.
static bool validateHeaderTasks(const llvm::json::Object &object,
                                ArrayRef<TaskMetadata> tasks,
                                std::string &error) {
  const llvm::json::Array *records = object.getArray("tasks");
  if (!records || records->size() != tasks.size()) {
    error = "candidate manifest header task list does not match current IR";
    return false;
  }
  for (auto [index, value] : llvm::enumerate(*records)) {
    const llvm::json::Object *record = value.getAsObject();
    if (!record) {
      error = "candidate manifest header task is not an object";
      return false;
    }
    auto name = requiredString(*record, "task", error);
    int64_t tripCount = 0;
    if (!name || !parseTripCount(*record, tripCount, error)) {
      return false;
    }
    if (record->get("body_sha256")) {
      error = "legacy SHA-256 task metadata is unsupported";
      return false;
    }
    if (*name != tasks[index].name || tripCount != tasks[index].tripCount) {
      error = "candidate manifest header task metadata do not match current IR";
      return false;
    }
  }
  return true;
}

// Parses one rectangular shape and checks its redundant fields against the
// dimensions. The explicit tile rows and columns are the mapper-shape truth.
static bool parseShape(const llvm::json::Object &object, RectShape &shape,
                       std::string &error) {
  // Every supported shape field is a concrete positive integer and is
  // validated redundantly below.
  auto kind = requiredString(object, "kind", error);
  auto rows = requiredInteger(object, "rows", error);
  auto cols = requiredInteger(object, "cols", error);
  auto count = requiredInteger(object, "cgra_count", error);
  auto irShape = requiredString(object, "cgra_shape", error);
  auto mapperRows = requiredInteger(object, "mapper_tile_rows", error);
  auto mapperCols = requiredInteger(object, "mapper_tile_cols", error);
  if (!kind || !rows || !cols || !count || !irShape || !mapperRows ||
      !mapperCols) {
    return false;
  }
  std::optional<int64_t> computedCount = checkedPositiveProduct(*rows, *cols);
  if (*kind != "rect" || !computedCount || *count != *computedCount ||
      *mapperRows <= 0 || *mapperCols <= 0) {
    error = "candidate contains a non-rectangular or invalid shape";
    return false;
  }
  shape = {*rows, *cols, *mapperRows, *mapperCols};
  if (*irShape != shape.toCgraShapeAttrValue()) {
    error = "candidate physical shape label does not match its dimensions";
    return false;
  }
  return true;
}

// Compares the physical and mapper dimensions of two rectangles.
static bool sameShape(const RectShape &lhs, const RectShape &rhs) {
  return std::tie(lhs.rows, lhs.cols, lhs.mapperRows, lhs.mapperCols) ==
         std::tie(rhs.rows, rhs.cols, rhs.mapperRows, rhs.mapperCols);
}

// Parses one candidate record and verifies task names and trip-count metadata.
// Candidate ordering and the sequential ID are checked by the stream reader,
// which knows the record's canonical mixed-radix index.
static bool parseCandidate(const llvm::json::Object &object,
                           ArrayRef<TaskMetadata> tasks, Candidate &candidate,
                           std::string &error) {
  auto schema = requiredString(object, "schema", error);
  auto id = requiredString(object, "candidate_id", error);
  const llvm::json::Array *records = object.getArray("task_shapes");
  if (!schema || !id || !records) {
    return false;
  }
  if (*schema != kCandidateSchema || records->size() != tasks.size()) {
    error = "candidate schema or task count does not match its manifest";
    return false;
  }

  candidate.id = id->str();
  candidate.choices.clear();
  for (auto [index, value] : llvm::enumerate(*records)) {
    const llvm::json::Object *record = value.getAsObject();
    if (!record) {
      error = "task_shapes entry is not an object";
      return false;
    }
    auto taskName = requiredString(*record, "task", error);
    int64_t tripCount = 0;
    const llvm::json::Object *shapeObject = record->getObject("shape");
    if (!taskName || !shapeObject ||
        !parseTripCount(*record, tripCount, error)) {
      return false;
    }
    const TaskMetadata &task = tasks[index];
    if (*taskName != task.name || tripCount != task.tripCount) {
      error = "candidate task metadata do not match the current IR";
      return false;
    }
    RectShape shape;
    if (!parseShape(*shapeObject, shape, error)) {
      return false;
    }
    candidate.choices.push_back({task.name, tripCount, std::move(shape)});
  }
  return true;
}

// Verifies one record from the Cartesian candidate stream. Shape tuples must
// remain in the lexicographic order produced by visitShapeCartesianProduct;
// this rejects duplicates and reordering without storing the entire manifest.
static bool validateCandidateAtIndex(
    uint64_t index, ArrayRef<TaskMetadata> tasks,
    ArrayRef<SmallVector<RectShape>> shapesByTask, StringRef shapePruningPolicy,
    int64_t maxChangedTasks, const Candidate &candidate,
    SmallVectorImpl<size_t> &previousShapeIndices, std::string &error) {
  if (candidate.id != makeSequentialCandidateId(index)) {
    error = "candidate ID does not match its canonical manifest index";
    return false;
  }
  if (candidate.choices.size() != tasks.size()) {
    error = "candidate task count does not match its declared space";
    return false;
  }
  SmallVector<size_t> shapeIndices;
  for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
    const RectShape &candidateShape = candidate.choices[taskIndex].shape;
    ArrayRef<RectShape> shapes = shapesByTask[taskIndex];
    auto found = llvm::find_if(shapes, [&](const RectShape &legalShape) {
      return sameShape(candidateShape, legalShape);
    });
    if (found == shapes.end()) {
      error = "candidate contains a shape outside its declared shape space";
      return false;
    }
    shapeIndices.push_back(static_cast<size_t>(found - shapes.begin()));
  }
  if (shapePruningPolicy == kShapePruningPolicyBoundedChangesFromUnit) {
    int64_t changedTasks = 0;
    for (const TaskShapeChoice &choice : candidate.choices)
      changedTasks += !(choice.shape.rows == 1 && choice.shape.cols == 1);
    if (changedTasks > maxChangedTasks) {
      error = "candidate is outside the bounded shape-pruning policy";
      return false;
    }
  }
  if (!previousShapeIndices.empty() &&
      !std::lexicographical_compare(previousShapeIndices.begin(),
                                    previousShapeIndices.end(),
                                    shapeIndices.begin(), shapeIndices.end())) {
    error = "candidate manifest is duplicated or out of canonical order";
    return false;
  }
  previousShapeIndices.assign(shapeIndices.begin(), shapeIndices.end());
  return true;
}

// Counts the exact Cartesian space, but stops as soon as it proves
// that the manifest's declared count is too small. Combined with strictly
// increasing legal records, equal counts prove that no valid tuple is missing.
static bool hasExactCandidateCount(
    uint64_t declaredCount, ArrayRef<SmallVector<RectShape>> shapesByTask,
    StringRef shapePruningPolicy, int64_t maxChangedTasks) {
  uint64_t computedCount = 0;
  bool exceededDeclaredCount = false;
  bool completed = visitShapeCandidateSpace(
      shapesByTask, shapePruningPolicy, maxChangedTasks,
      [&](uint64_t index, ArrayRef<size_t>) {
        if (index >= declaredCount) {
          exceededDeclaredCount = true;
          return false;
        }
        computedCount = index + 1;
        return true;
      });
  return completed && !exceededDeclaredCount && computedCount == declaredCount;
}

// Bind the manifest to the explicit physical architecture dimensions.
// The architecture path/schema and source Git commit are carried by the
// predictor provenance; no file-content fingerprint is recomputed here.
static bool architectureMatches(const ManifestHeader &header,
                                const ::mlir::neura::Architecture &architecture,
                                std::string &error) {
  if (header.gridRows != architecture.getMultiCgraRows() ||
      header.gridCols != architecture.getMultiCgraColumns() ||
      header.perCgraRows != architecture.getPerCgraRows() ||
      header.perCgraCols != architecture.getPerCgraColumns()) {
    error = "candidate manifest architecture dimensions do not match current "
            "Neura architecture";
    return false;
  }
  return true;
}

// Streams and validates a complete candidate manifest before forwarding each
// candidate to the consumer. Every record must be a legal shape tuple in
// canonical order. The footer is checked against an independent traversal of
// the exact Cartesian product.
bool readCandidateManifest(StringRef path, ArrayRef<TaskMetadata> tasks,
                           StringRef expectedFunction,
                           const ::mlir::neura::Architecture &architecture,
                           CandidateConsumer consume, ManifestHeader &header,
                           ManifestFooter &footer, std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read candidate manifest " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  bool sawHeader = false;
  bool sawFooter = false;
  uint64_t count = 0;
  SmallVector<SmallVector<RectShape>> legalShapesByTask;
  SmallVector<size_t> previousShapeIndices;
  for (llvm::line_iterator lines(**buffer, /*SkipBlanks=*/true);
       !lines.is_at_end(); ++lines) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
    if (!parsed) {
      error = "invalid candidate JSONL: " + llvm::toString(parsed.takeError());
      return false;
    }
    llvm::json::Object *object = parsed->getAsObject();
    if (!object) {
      error = "candidate JSONL record is not an object";
      return false;
    }
    auto recordType = object->getString("record_type");
    if (!recordType) {
      error = "candidate JSONL record has no record_type";
      return false;
    }
    if (*recordType == "header") {
      if (sawHeader || count != 0 || sawFooter ||
          !parseHeader(*object, header, error) ||
          !validateHeaderTasks(*object, tasks, error)) {
        if (error.empty()) {
          error = "candidate manifest header is misplaced or duplicated";
        }
        return false;
      }
      if (header.function != expectedFunction) {
        error = "candidate manifest function does not match current IR";
        return false;
      }
      if (!architectureMatches(header, architecture, error)) {
        return false;
      }
      SmallVector<RectShape> legalShapes = enumerateStaticRectShapes(
          header.gridRows, header.gridCols, header.perCgraRows,
          header.perCgraCols, header.maxCgrasPerTask);
      // Reconstruct the same full rectangular alphabet for every task. The
      // only per-task size bound is the manifest's explicit/physical maximum;
      // operation counts cannot silently remove candidate shapes here.
      legalShapesByTask.assign(tasks.size(), legalShapes);
      if (legalShapes.empty()) {
        error = "candidate manifest declares an empty shape space";
        return false;
      }
      sawHeader = true;
      continue;
    }
    if (*recordType == "candidate") {
      if (!sawHeader || sawFooter) {
        error = "candidate record is outside header/footer";
        return false;
      }
      Candidate candidate;
      if (!parseCandidate(*object, tasks, candidate, error) ||
          !validateCandidateAtIndex(
              count, tasks, legalShapesByTask, header.shapePruningPolicy,
              header.maxChangedTasks, candidate, previousShapeIndices, error) ||
          !consume(count, candidate, error)) {
        return false;
      }
      ++count;
      continue;
    }
    if (*recordType == "footer") {
      auto schema = requiredString(*object, "schema", error);
      auto expectedCount = requiredInteger(*object, "candidate_count", error);
      auto footerPolicy = requiredString(*object, "shape_pruning_policy", error);
      const llvm::json::Object *spaceSummary =
          object->getObject("space_summary");
      auto footerMaxChanged =
          spaceSummary
              ? requiredInteger(*spaceSummary, "max_changed_tasks", error)
              : std::nullopt;
      auto footerUnrestricted =
          spaceSummary
              ? requiredDecimalCount(*spaceSummary,
                                     "unrestricted_candidate_count", error)
              : std::nullopt;
      auto footerDeclared =
          spaceSummary
              ? requiredDecimalCount(*spaceSummary, "declared_candidate_count",
                                     error)
              : std::nullopt;
      auto footerExcluded =
          spaceSummary
              ? requiredDecimalCount(*spaceSummary, "excluded_candidate_count",
                                     error)
              : std::nullopt;
      auto footerUnmeasured =
          spaceSummary
              ? requiredDecimalCount(*spaceSummary,
                                     "unmeasured_candidate_count", error)
              : std::nullopt;
      if (!sawHeader || sawFooter || !schema || !expectedCount ||
          !footerPolicy || !spaceSummary || !footerMaxChanged ||
          !footerUnrestricted || !footerDeclared || !footerExcluded ||
          !footerUnmeasured || *schema != kCandidateSchema ||
          *expectedCount <= 0) {
        if (error.empty()) {
          error = "invalid candidate manifest footer";
        }
        return false;
      }
      const std::string countText = std::to_string(*expectedCount);
      ShapeCandidateSpaceCounts expected = countShapeCandidateSpace(
          legalShapesByTask, header.shapePruningPolicy,
          header.maxChangedTasks);
      if (expected.declared.empty() ||
          static_cast<uint64_t>(*expectedCount) != count ||
          countText != header.declaredCandidateCount ||
          *footerPolicy != header.shapePruningPolicy ||
          *footerMaxChanged != header.maxChangedTasks ||
          *footerUnrestricted != expected.unrestricted ||
          *footerDeclared != expected.declared ||
          *footerExcluded != expected.excluded ||
          *footerUnmeasured != expected.excluded ||
          header.unrestrictedCandidateCount != expected.unrestricted ||
          header.excludedCandidateCount != expected.excluded ||
          header.unmeasuredCandidateCount != expected.excluded ||
          !hasExactCandidateCount(count, legalShapesByTask,
                                  header.shapePruningPolicy,
                                  header.maxChangedTasks)) {
        error = "candidate manifest count does not match its declared "
                "shape-pruning family";
        return false;
      }
      if (count > static_cast<uint64_t>(header.maxCandidates)) {
        error = "candidate manifest exceeds its recorded budget";
        return false;
      }
      footer.candidateCount = count;
      footer.shapePruningPolicy = footerPolicy->str();
      footer.unrestrictedCandidateCount = *footerUnrestricted;
      footer.declaredCandidateCount = *footerDeclared;
      footer.excludedCandidateCount = *footerExcluded;
      footer.unmeasuredCandidateCount = *footerUnmeasured;
      footer.maxChangedTasks = *footerMaxChanged;
      sawFooter = true;
      continue;
    }
    error = "unknown candidate JSONL record_type " + recordType->str();
    return false;
  }
  if (!sawHeader || !sawFooter) {
    error = "candidate manifest is incomplete";
    return false;
  }
  return true;
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir
