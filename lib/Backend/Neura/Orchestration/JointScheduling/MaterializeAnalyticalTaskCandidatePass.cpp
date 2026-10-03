//===- MaterializeAnalyticalTaskCandidatePass.cpp ------------------------===//
//
// Implements replay of one validated shape candidate onto Taskflow IR.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateManifest.h"
#include "NeighborhoodReplaySelection.h"

#include "Backend/Neura/NeuraBackendPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;

namespace {

static bool isNeighborhoodSelection(StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return false;
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return false;
  }
  const auto *object = parsed->getAsObject();
  return object && object->getString("schema") ==
                       std::optional<StringRef>(kNeighborhoodShapeSelectionSchema);
}

// Validate the entire singular selection before the existing materializer
// mutates any task. Exact source text binds all bodies, bounds and graph roots;
// task metadata and physical dimensions independently bind the selected tuple.
static bool readNeighborhoodSelection(
    StringRef path, StringRef requestedId, ModuleOp module, func::FuncOp function,
    ArrayRef<TaskMetadata> tasks, const ::mlir::neura::Architecture &architecture,
    Candidate &candidate, std::string &graphVariantId, std::string &error) {
  auto reject = [&](StringRef reason) {
    error = (Twine("invalid neighborhood replay selection: ") + reason).str();
    return false;
  };
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return reject("cannot read file");
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return reject("invalid JSON");
  }
  const auto *object = parsed->getAsObject();
  if (!object)
    return reject("root is not an object");
  auto schema = object->getString("schema");
  auto recordType = object->getString("record_type");
  auto id = object->getString("candidate_id");
  auto savedFunction = object->getString("function");
  auto graph = object->getString("graph_variant_id");
  auto source = object->getString("source_ir");
  auto scope = object->getString("search_scope");
  auto bestFound = object->getBoolean("best_found");
  const auto *dimensions = object->getObject("architecture");
  const auto *shapes = object->getArray("task_shapes");
  auto irGraph = function->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
  StringRef expectedGraph = irGraph ? irGraph.getValue() : StringRef();
  if (!schema || *schema != kNeighborhoodShapeSelectionSchema || !recordType ||
      *recordType != "selection" || !id || *id != requestedId || id->empty() ||
      !savedFunction || *savedFunction != function.getSymName() || !graph ||
      *graph != expectedGraph || !source || !scope ||
      *scope != "budgeted-complete-program-neighborhood" || !bestFound ||
      !*bestFound || !dimensions || !shapes || shapes->size() != tasks.size())
    return reject("identity, scope, graph or task count mismatch");
  if (*source != neighborhoodReplaySourceText(module))
    return reject("source IR bytes differ");
  const std::pair<StringRef, int64_t> expectedDimensions[] = {
      {"grid_rows", architecture.getMultiCgraRows()},
      {"grid_cols", architecture.getMultiCgraColumns()},
      {"per_cgra_tile_rows", architecture.getPerCgraRows()},
      {"per_cgra_tile_cols", architecture.getPerCgraColumns()}};
  for (const auto &entry : expectedDimensions) {
    auto value = dimensions->getInteger(entry.first);
    if (!value || *value <= 0 || *value != entry.second)
      return reject("architecture dimensions differ");
  }
  SmallVector<RectShape> legal = enumerateStaticRectShapes(
      architecture.getMultiCgraRows(), architecture.getMultiCgraColumns(),
      architecture.getPerCgraRows(), architecture.getPerCgraColumns(), 4);
  if (legal.empty())
    return reject("architecture has no legal area-four rectangles");
  Candidate result;
  result.id = id->str();
  for (auto [index, value] : llvm::enumerate(*shapes)) {
    const auto *entry = value.getAsObject();
    const auto *shape = entry ? entry->getObject("shape") : nullptr;
    auto task = entry ? entry->getString("task") : std::nullopt;
    auto tripCount = entry ? entry->getInteger("trip_count") : std::nullopt;
    if (!task || *task != tasks[index].name || !tripCount ||
        *tripCount != tasks[index].tripCount || !shape)
      return reject("task order or static trip count mismatch");
    auto rows = shape->getInteger("rows");
    auto cols = shape->getInteger("cols");
    auto mapperRows = shape->getInteger("mapper_tile_rows");
    auto mapperCols = shape->getInteger("mapper_tile_cols");
    auto count = shape->getInteger("cgra_count");
    auto spelling = shape->getString("cgra_shape");
    auto kind = shape->getString("kind");
    if (!rows || !cols || !mapperRows || !mapperCols || !count || !spelling ||
        !kind || *kind != "rect")
      return reject("missing oriented rectangle fields");
    auto found = llvm::find_if(legal, [&](const RectShape &rectangle) {
      return rectangle.rows == *rows && rectangle.cols == *cols &&
             rectangle.mapperRows == *mapperRows &&
             rectangle.mapperCols == *mapperCols &&
             rectangle.cgraCount() == *count &&
             rectangle.toCgraShapeAttrValue() == *spelling;
    });
    if (found == legal.end())
      return reject("rectangle is outside the architecture and area-four policy");
    result.choices.push_back({tasks[index].name, tasks[index].tripCount, *found});
  }
  candidate = std::move(result);
  graphVariantId = graph->str();
  return true;
}

static std::optional<std::string>
readUnsignedDecimal(const llvm::json::Object &object, StringRef key,
                    std::string &error) {
  if (std::optional<int64_t> numeric = object.getInteger(key)) {
    if (*numeric >= 0)
      return std::to_string(*numeric);
  } else if (std::optional<StringRef> value = object.getString(key)) {
    if (!value->empty() &&
        llvm::all_of(*value, [](char c) { return c >= '0' && c <= '9'; }) &&
        (value->size() == 1 || value->front() != '0'))
      return value->str();
  }
  error = "factored candidate " + key.str() + " must be a canonical "
          "unsigned decimal";
  return std::nullopt;
}

static bool isFactoredCandidateSpace(StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return false;
  for (llvm::line_iterator lines(**buffer, /*SkipBlanks=*/true);
       !lines.is_at_end(); ++lines) {
    auto parsed = llvm::json::parse(*lines);
    if (!parsed)
      return false;
    const auto *object = parsed->getAsObject();
    if (!object)
      return false;
    auto recordType = object->getString("record_type");
    return recordType && *recordType == "space";
  }
  return false;
}

static bool readFactoredCandidate(StringRef path, StringRef requestedId,
                                  func::FuncOp func,
                                  ArrayRef<TaskMetadata> tasks,
                                  const ::mlir::neura::Architecture &architecture,
                                  Candidate &selected,
                                  std::string &graphVariantId,
                                  std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read factored candidate space " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  std::optional<llvm::json::Object> header;
  std::optional<llvm::json::Object> footer;
  unsigned records = 0;
  for (llvm::line_iterator lines(**buffer, /*SkipBlanks=*/true);
       !lines.is_at_end(); ++lines) {
    auto parsed = llvm::json::parse(*lines);
    if (!parsed || !parsed->getAsObject()) {
      error = "factored candidate space contains invalid JSON";
      return false;
    }
    const auto *object = parsed->getAsObject();
    auto recordType = object->getString("record_type");
    if (records == 0 && recordType && *recordType == "space")
      header = *object;
    else if (records == 1 && recordType && *recordType == "footer")
      footer = *object;
    else {
      error = "factored candidate space must contain exactly one header and "
              "one footer";
      return false;
    }
    ++records;
  }
  if (records != 2 || !header || !footer) {
    error = "factored candidate space is missing its header or footer";
    return false;
  }
  auto schema = header->getString("schema");
  auto representation = header->getString("representation");
  auto exact = header->getBoolean("exact");
  auto function = header->getString("function");
  auto graph = header->getString("graph_variant_id");
  auto searchScope = header->getString("search_scope");
  auto shapePolicy = header->getString("shape_policy");
  auto capacityPolicy = header->getString("spatial_capacity_policy");
  auto pruningPolicy = header->getString("shape_pruning_policy");
  auto maxChanged = header->getInteger("max_changed_tasks");
  auto maxCgras = header->getInteger("max_cgras_per_task");
  const auto *factors = header->getArray("factors");
  auto irGraph = func->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
  if (!schema || *schema != kFactoredSpaceSchema || !representation ||
      *representation != "factored" || !exact || !*exact || !function ||
      *function != func.getSymName() || !graph || graph->empty() || !irGraph ||
      *graph != irGraph.getValue() || !searchScope ||
      *searchScope != kFactoredSearchScope || !shapePolicy ||
      *shapePolicy != kShapePolicy || !capacityPolicy ||
      *capacityPolicy != kSpatialCapacityPolicy || !pruningPolicy ||
      *pruningPolicy != kShapePruningPolicyNone || !maxChanged ||
      *maxChanged != -1 || !maxCgras || *maxCgras <= 0 || !factors ||
      factors->size() != tasks.size()) {
    error = "factored candidate header does not match the selected IR and "
            "exact shape-space contract";
    return false;
  }
  const int64_t rows = architecture.getMultiCgraRows();
  const int64_t cols = architecture.getMultiCgraColumns();
  if (rows <= 0 || cols <= 0 ||
      rows > std::numeric_limits<int64_t>::max() / cols ||
      *maxCgras > rows * cols) {
    error = "factored candidate max-cgras-per-task exceeds the architecture";
    return false;
  }
  SmallVector<RectShape> canonicalShapes = enumerateStaticRectShapes(
      rows, cols, architecture.getPerCgraRows(),
      architecture.getPerCgraColumns(), *maxCgras);
  if (canonicalShapes.empty()) {
    error = "factored candidate shape alphabet is empty";
    return false;
  }
  for (auto [taskIndex, factorValue] : llvm::enumerate(*factors)) {
    const auto *factor = factorValue.getAsObject();
    if (!factor) {
      error = "factored candidate factor is not an object";
      return false;
    }
    auto taskName = factor->getString("task");
    auto tripCount = factor->getInteger("trip_count");
    const auto *shapes = factor->getArray("shapes");
    if (!taskName || *taskName != tasks[taskIndex].name || !tripCount ||
        *tripCount != tasks[taskIndex].tripCount || !shapes ||
        shapes->size() != canonicalShapes.size()) {
      error = "factored candidate factor does not match current task metadata";
      return false;
    }
    for (auto [shapeIndex, shapeValue] : llvm::enumerate(*shapes)) {
      const auto *shape = shapeValue.getAsObject();
      if (!shape) {
        error = "factored candidate shape is not an object";
        return false;
      }
      const RectShape &canonical = canonicalShapes[shapeIndex];
      auto kind = shape->getString("kind");
      auto shapeText = shape->getString("cgra_shape");
      auto shapeRows = shape->getInteger("rows");
      auto shapeCols = shape->getInteger("cols");
      auto cgraCount = shape->getInteger("cgra_count");
      auto mapperRows = shape->getInteger("mapper_tile_rows");
      auto mapperCols = shape->getInteger("mapper_tile_cols");
      if (!kind || *kind != "rect" || !shapeText ||
          *shapeText != canonical.toCgraShapeAttrValue() || !shapeRows ||
          *shapeRows != canonical.rows || !shapeCols ||
          *shapeCols != canonical.cols || !cgraCount ||
          *cgraCount != canonical.cgraCount() || !mapperRows ||
          *mapperRows != canonical.mapperRows || !mapperCols ||
          *mapperCols != canonical.mapperCols) {
        error = "factored candidate shape alphabet is not canonical for "
                "the current architecture";
        return false;
      }
    }
  }
  auto count = readUnsignedDecimal(*header, "candidate_count", error);
  if (!count)
    return false;
  auto footerCount = readUnsignedDecimal(*footer, "candidate_count", error);
  if (!footerCount)
    return false;
  auto footerSchema = footer->getString("schema");
  auto footerRepresentation = footer->getString("representation");
  auto footerExact = footer->getBoolean("exact");
  auto ranked = footer->getBoolean("ranked");
  auto status = footer->getString("status");
  if (!footerSchema || *footerSchema != kFactoredSpaceSchema ||
      !footerRepresentation || *footerRepresentation != "factored" ||
      !footerExact || !*footerExact || !ranked || *ranked || !status ||
      *status != "unranked-factored-space" || *footerCount != *count) {
    error = "factored candidate footer does not close the exact descriptor";
    return false;
  }
  if (tasks.size() > (std::numeric_limits<unsigned>::max() / 64u) - 1u) {
    error = "factored candidate task count exceeds exact index width";
    return false;
  }
  SmallVector<SmallVector<RectShape>> shapesByTask(tasks.size(),
                                                    canonicalShapes);
  ShapeCandidateSpaceCounts expected = countShapeCandidateSpace(
      shapesByTask, kShapePruningPolicyNone, -1);
  if (expected.declared.empty() || *count != expected.declared) {
    error = "factored candidate count does not match the canonical Cartesian "
            "space";
    return false;
  }
  StringRef ordinal = requestedId;
  if (!ordinal.consume_front("candidate-") || ordinal.empty() ||
      (ordinal.size() > 1 && ordinal.front() == '0') ||
      !llvm::all_of(ordinal,
                    [](char c) { return c >= '0' && c <= '9'; })) {
    error = "factored candidate ID must be candidate-N with a canonical "
            "unsigned decimal N";
    return false;
  }
  const unsigned width =
      std::max<unsigned>(64u, 64u * static_cast<unsigned>(tasks.size() + 1));
  // Check the decimal length before constructing APInt, which otherwise
  // truncates an oversized candidate ID to the requested bit width.
  if (ordinal.size() > count->size() ||
      (ordinal.size() == count->size() && ordinal >= StringRef(*count))) {
    error = "factored candidate ID is outside the complete shape space";
    return false;
  }
  llvm::APInt index(width, ordinal, 10);
  llvm::APInt declared(width, StringRef(*count), 10);
  if (index.uge(declared)) {
    error = "factored candidate ID is outside the complete shape space";
    return false;
  }
  selected.id = requestedId.str();
  selected.choices.resize(tasks.size());
  for (size_t reverseIndex = tasks.size(); reverseIndex != 0; --reverseIndex) {
    const size_t taskIndex = reverseIndex - 1;
    llvm::APInt radix(width, canonicalShapes.size());
    const size_t shapeIndex = (index.urem(radix)).getZExtValue();
    index = index.udiv(radix);
    selected.choices[taskIndex] =
        {tasks[taskIndex].name, tasks[taskIndex].tripCount,
         canonicalShapes[shapeIndex]};
  }
  if (!index.isZero()) {
    error = "factored candidate ID does not decode within the shape space";
    return false;
  }
  graphVariantId = graph->str();
  return true;
}

struct MaterializeAnalyticalTaskCandidatePass
    : public PassWrapper<MaterializeAnalyticalTaskCandidatePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializeAnalyticalTaskCandidatePass)

  MaterializeAnalyticalTaskCandidatePass() = default;
  MaterializeAnalyticalTaskCandidatePass(
      const MaterializeAnalyticalTaskCandidatePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "materialize-analytical-task-candidate";
  }
  StringRef getDescription() const override {
    return "Writes one selected shape candidate onto Taskflow IR without "
           "invoking the mapper";
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Taskflow function; inferred when exactly one exists."),
      llvm::cl::init("")};
  Option<std::string> candidateFile{
      *this, "candidates", llvm::cl::desc("Frozen candidate JSONL path."),
      llvm::cl::init("")};
  Option<std::string> candidateIdOption{
      *this, "candidate-id", llvm::cl::desc("Candidate ID to materialize."),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selectedFunction =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selectedFunction)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp func = *selectedFunction;
    if (candidateFile.getValue().empty() ||
        candidateIdOption.getValue().empty()) {
      func.emitError() << "candidates and candidate-id are required";
      return signalPassFailure();
    }

    FailureOr<SmallVector<TaskMetadata>> taskMetadata =
        collectAnalyticalTaskMetadata(func, error);
    if (failed(taskMetadata)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    std::optional<Candidate> selected;
    std::string graphVariantId;
    bool neighborhoodSelection = isNeighborhoodSelection(candidateFile.getValue());
    if (neighborhoodSelection) {
      Candidate tuple;
      if (!readNeighborhoodSelection(
              candidateFile.getValue(), candidateIdOption.getValue(), module,
              func, *taskMetadata, ::mlir::neura::getArchitecture(), tuple,
              graphVariantId, error)) {
        func.emitError() << error;
        return signalPassFailure();
      }
      selected = std::move(tuple);
    } else if (isFactoredCandidateSpace(candidateFile.getValue())) {
      Candidate factoredCandidate;
      if (!readFactoredCandidate(candidateFile.getValue(),
                                 candidateIdOption.getValue(), func,
                                 *taskMetadata, ::mlir::neura::getArchitecture(),
                                 factoredCandidate, graphVariantId, error)) {
        func.emitError() << error;
        return signalPassFailure();
      }
      selected = std::move(factoredCandidate);
    } else {
      // The row manifest reader verifies every row and the closing footer;
      // selecting an early row never bypasses completeness validation.
      ManifestHeader header;
      ManifestFooter footer;
      auto consume = [&](uint64_t, const Candidate &candidate,
                         std::string &) {
        if (candidate.id == candidateIdOption.getValue()) {
          if (selected)
            return false;
          selected = candidate;
        }
        return true;
      };
      if (!readCandidateManifest(
              candidateFile.getValue(), *taskMetadata, func.getSymName(),
              ::mlir::neura::getArchitecture(), consume, header, footer,
              error)) {
        if (error.empty())
          error = "candidate selection is ambiguous";
        func.emitError() << error;
        return signalPassFailure();
      }
      if (!selected) {
        func.emitError() << "requested candidate is absent from the complete "
                            "manifest";
        return signalPassFailure();
      }
      graphVariantId = header.graphVariantId;
      auto irGraph =
          func->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
      if ((irGraph && irGraph.getValue() != graphVariantId) ||
          static_cast<bool>(irGraph) != !graphVariantId.empty()) {
        func.emitError() << "candidate manifest graph variant identity does "
                            "not match the selected IR";
        return signalPassFailure();
      }
    }

    // Delays mutation until the manifest, IR, architecture, and record have
    // all been validated. These attributes configure the unchanged downstream
    // heuristic mapper; this pass fabricates no placement or II.  Remove any
    // legacy file-fingerprint attributes carried by an input IR so the
    // materialized output obeys the no-SHA contract.
    func->removeAttr("joint_scheduling_architecture_sha256");
    func->removeAttr("joint_scheduling_graph_variant_sha256");
    func->removeAttr("amoeba.graph_variant_sha256");
    OpBuilder builder(func.getContext());
    for (auto [task, choice] : llvm::zip(*taskMetadata, selected->choices)) {
      // These attributes are derived from a previous shape, placement, or
      // profiling run. Keeping any of them would let a newly selected
      // rectangle inherit facts for a different candidate.
      for (StringRef attribute :
           {"compiled_ii", "profile_info", "task_orchestration_info",
            "replicas", "tiling", "est_latency"})
        task.op->removeAttr(attribute);
      task.op->setAttr("cgra_count",
                       builder.getI32IntegerAttr(
                           static_cast<int32_t>(choice.shape.cgraCount())));
      task.op->setAttr("cgra_shape", builder.getStringAttr(
                                         choice.shape.toCgraShapeAttrValue()));
      task.op->setAttr("amoeba.selected_cgra_count",
                       builder.getI64IntegerAttr(choice.shape.cgraCount()));
      task.op->setAttr(
          "amoeba.selected_cgra_shape",
          builder.getStringAttr(choice.shape.toCgraShapeAttrValue()));
      task.op->setAttr("amoeba.selected_mapper_tile_rows",
                       builder.getI64IntegerAttr(choice.shape.mapperRows));
      task.op->setAttr("amoeba.selected_mapper_tile_cols",
                       builder.getI64IntegerAttr(choice.shape.mapperCols));
      task.op->setAttr("amoeba.selected_trip_count",
                       builder.getI64IntegerAttr(choice.tripCount));
      // The rectangle orientation is part of the selected candidate and must
      // remain unchanged during resource allocation.
      task.op->setAttr("amoeba.joint_shape_orientation_fixed",
                       builder.getUnitAttr());
    }
    func->setAttr("joint_scheduling_candidate_id",
                  builder.getStringAttr(selected->id));
    func->setAttr("joint_scheduling_candidate_scope",
                  builder.getStringAttr(neighborhoodSelection
                      ? StringRef("budgeted-complete-program-neighborhood")
                      : StringRef(kSearchScope)));
    if (!graphVariantId.empty())
      func->setAttr("joint_scheduling_graph_variant_id",
                    builder.getStringAttr(graphVariantId));
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createMaterializeAnalyticalTaskCandidatePass() {
  return std::make_unique<MaterializeAnalyticalTaskCandidatePass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
