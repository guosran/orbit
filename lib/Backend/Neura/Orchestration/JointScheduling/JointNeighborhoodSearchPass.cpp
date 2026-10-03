//===- JointNeighborhoodSearchPass.cpp ----------------------*- C++ -*-===//
//
// A bounded, source-owned complete-program neighborhood search.  This pass
// deliberately keeps graph materialization and production scheduling at the
// existing C++ seams: the search itself only chooses a finite action, clones
// the resulting module, obtains source-owned structural facts, and invokes
// the production scheduler once for the resulting whole program.
//
// The pass is intentionally best-found.  Its output never claims exhaustive
// coverage or global optimality.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalTaskCostCatalog.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "JointNeighborhoodActions.h"
#include "NeighborhoodReplaySelection.h"
#include "Backend/Neura/Orchestration/JointScheduling/GraphFactsIO.h"
#include "ProductionSchedulerAdapter.h"

#include "TaskflowDialect/TaskflowOps.h"
#include "NeuraDialect/NeuraOps.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <filesystem>
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
namespace json = llvm::json;

namespace {

constexpr StringLiteral kSearchSchema = "orbit-neighborhood-search-v1";
constexpr StringLiteral kCheckpointSchema =
    "orbit-neighborhood-search-checkpoint-v2";
constexpr StringLiteral kSearchScope = "budgeted-complete-program-neighborhood";
constexpr int64_t kDefaultRounds = 20;
constexpr int64_t kDefaultCandidates = 20000;
constexpr int64_t kDefaultBeam = 16;
constexpr int64_t kDefaultDiversity = 4;

static std::string jsonText(const json::Value &value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << value;
  stream.flush();
  return text;
}

static std::string quoteOption(StringRef value) {
  std::string quoted = "\"";
  for (char character : value) {
    if (character == '\\' || character == '"')
      quoted += '\\';
    quoted += character;
  }
  quoted += '"';
  return quoted;
}

static bool writeTextAtomically(StringRef path, StringRef text,
                                std::string &error) {
  return writeAtomically(
      path,
      [&](llvm::raw_ostream &stream) {
        stream << text;
        return true;
      },
      error);
}

static bool ensureDirectory(StringRef path, std::string &error) {
  if (path.empty()) {
    error = "neighborhood search output directory is empty";
    return false;
  }
  std::error_code ec = llvm::sys::fs::create_directories(path);
  if (ec) {
    error = "cannot create neighborhood search directory " + path.str() +
            ": " + ec.message();
    return false;
  }
  return true;
}

static std::string actionSignature(const NeighborhoodAction &action) {
  std::string text = action.family + ":" + action.label + ":" +
                     action.shapeTask + ":" + std::to_string(action.shapeRows) +
                     "x" + std::to_string(action.shapeCols) + ":reset=" +
                     (action.canonicalReset ? "1" : "0");
  for (const NeighborhoodPrimitive &primitive : action.primitives) {
    text += ":" + primitive.kind + ":" + primitive.firstTask + ":" +
            primitive.secondTask + ":" + primitive.mode + ":axis=" +
            std::to_string(primitive.axis) + ":factor=" +
            std::to_string(primitive.factor);
  }
  return text;
}

static std::string shapeSignature(ArrayRef<NeighborhoodShape> shapes) {
  std::string text;
  for (const NeighborhoodShape &shape : shapes) {
    // Tasks are in source order; names are path provenance rather than shape
    // semantics. Canonical graph facts bind the task/input identity.
    text += std::to_string(shape.rows) + "x" + std::to_string(shape.cols) + ";";
  }
  return text;
}

static bool parseShape(StringRef value, int64_t &rows, int64_t &cols) {
  size_t separator = value.find('x');
  if (separator == StringRef::npos)
    separator = value.find('X');
  if (separator == StringRef::npos)
    return false;
  StringRef rowText = value.take_front(separator);
  StringRef colText = value.drop_front(separator + 1);
  return !rowText.getAsInteger(10, rows) &&
         !colText.getAsInteger(10, cols) && rows > 0 && cols > 0;
}

static std::optional<std::pair<int64_t, int64_t>> taskShapeFromAttrs(
    taskflow::TaskflowTaskOp task) {
  if (auto shape = task->getAttrOfType<StringAttr>("cgra_shape")) {
    int64_t rows = 0, cols = 0;
    if (parseShape(shape.getValue(), rows, cols))
      return std::make_pair(rows, cols);
  }
  return std::nullopt;
}

static std::vector<NeighborhoodShape>
initialShapes(func::FuncOp function) {
  std::vector<NeighborhoodShape> shapes;
  function.walk([&](taskflow::TaskflowTaskOp task) {
    NeighborhoodShape shape;
    shape.task = task.getTaskName().str();
    if (auto parsed = taskShapeFromAttrs(task)) {
      shape.rows = parsed->first;
      shape.cols = parsed->second;
    }
    shapes.push_back(std::move(shape));
  });
  return shapes;
}

static const NeighborhoodShape *findShape(ArrayRef<NeighborhoodShape> shapes,
                                           StringRef task) {
  for (const NeighborhoodShape &shape : shapes)
    if (shape.task == task)
      return &shape;
  return nullptr;
}

static void synchronizeShapes(func::FuncOp function,
                              ArrayRef<NeighborhoodShape> prior,
                              std::vector<NeighborhoodShape> &result) {
  result.clear();
  function.walk([&](taskflow::TaskflowTaskOp task) {
    NeighborhoodShape shape;
    shape.task = task.getTaskName().str();
    if (const NeighborhoodShape *old = findShape(prior, shape.task)) {
      shape.rows = old->rows;
      shape.cols = old->cols;
    } else if (auto parsed = taskShapeFromAttrs(task)) {
      shape.rows = parsed->first;
      shape.cols = parsed->second;
    }
    if (shape.rows <= 0 || shape.cols <= 0) {
      shape.rows = 1;
      shape.cols = 1;
    }
    result.push_back(std::move(shape));
  });
}

static bool runPassOnModule(ModuleOp module, std::unique_ptr<Pass> pass,
                            std::string &reason, std::string &diagnostic) {
  PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(std::move(pass));
  LogicalResult runResult = failure();
  LogicalResult verifyResult = failure();
  {
    ScopedDiagnosticHandler capture(module.getContext(),
                                    [&](Diagnostic &entry) {
                                      if (diagnostic.empty()) {
                                        llvm::raw_string_ostream stream(
                                            diagnostic);
                                        entry.print(stream);
                                        stream.flush();
                                      }
                                      return success();
                                    });
    runResult = manager.run(module);
    if (succeeded(runResult))
      verifyResult = verify(module.getOperation());
  }
  if (failed(runResult)) {
    reason = reason.empty() ? "source_owned_pass_failed" : reason;
    if (diagnostic.empty())
      diagnostic = "source-owned pass failed without a diagnostic";
    return false;
  }
  if (failed(verifyResult)) {
    reason = "materialized_module_verifier_failed";
    return false;
  }
  return true;
}

static bool extractStructuralFacts(ModuleOp module, StringRef functionName,
                                   StringRef outputDir, uint64_t serial,
                                   std::string &key, std::string &reason,
                                   std::string &diagnostic) {
  std::string factsPath = outputDir.str() + "/.facts-" +
                          std::to_string(serial) + ".json";
  auto pass = mlir::amoeba::neura::createExtractJointTaskGraphFactsPass();
  std::string options = "function=" + quoteOption(functionName) +
                        " output=" + quoteOption(factsPath) +
                        " fact-only=true max-tile-factor=8 "
                        "max-fission-actions-per-task=0";
  if (failed(pass->initializeOptions(options, [&](const llvm::Twine &message) {
        reason = message.str();
        return failure();
      }))) {
    if (reason.empty())
      reason = "facts_pass_option_parse_failed";
    return false;
  }
  if (!runPassOnModule(module, std::move(pass), reason, diagnostic)) {
    llvm::sys::fs::remove(factsPath);
    return false;
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(factsPath);
  if (!buffer) {
    reason = "facts_output_missing";
    llvm::sys::fs::remove(factsPath);
    return false;
  }
  llvm::Expected<json::Value> parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    reason = "facts_output_invalid_json";
    llvm::consumeError(parsed.takeError());
    llvm::sys::fs::remove(factsPath);
    return false;
  }
  json::Object *root = parsed->getAsObject();
  const json::Value *structural = root ? root->get("structural_key") : nullptr;
  if (!structural) {
    reason = "facts_output_missing_structural_key";
    llvm::sys::fs::remove(factsPath);
    return false;
  }
  key = jsonText(*structural);
  llvm::sys::fs::remove(factsPath);
  return true;
}

static std::optional<int64_t> durationFromCost(const TaskShapeCost &cost,
                                               int64_t tripCount,
                                               std::string &error) {
  if (!cost.supported)
    return std::nullopt;
  long double duration = static_cast<long double>(cost.startupCycles) +
                         static_cast<long double>(cost.predictedII) *
                             static_cast<long double>(tripCount - 1);
  if (!std::isfinite(duration) || duration <= 0.0L ||
      duration > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    error = "ML II/startup duration is outside scheduler range";
    return std::nullopt;
  }
  duration = std::ceil(duration);
  return std::max<int64_t>(1, static_cast<int64_t>(duration));
}

static bool sameTaskSet(ArrayRef<TaskMetadata> tasks,
                        ArrayRef<NeighborhoodShape> shapes) {
  if (tasks.size() != shapes.size()) return false;
  std::set<std::string> names;
  for (unsigned index = 0; index < tasks.size(); ++index)
    if (tasks[index].name != shapes[index].task ||
        !names.insert(shapes[index].task).second) return false;
  return true;
}

struct SearchState {
  std::string id;
  std::string factKey;
  std::string key;
  std::string parentId;
  std::string rejectReason;
  std::vector<std::string> path;
  std::vector<NeighborhoodShape> shapes;
  OwningOpRef<ModuleOp> module;
  int64_t score = std::numeric_limits<int64_t>::max();
  ProductionScheduledResult schedule;
  SmallVector<TaskShapeChoice> choices;
  SmallVector<int64_t> durations;
  SmallVector<TaskShapeCost> costs;
  std::string costPath;
  std::vector<std::string> controlRoles;
  bool control = false;
  bool keyReserved = false;
  bool scored = false;
};

struct ArchiveRecord {
  std::string id;
  std::string key;
  std::string factKey;
  std::string parentId;
  std::string candidatePath;
  std::vector<std::string> path;
  std::vector<NeighborhoodShape> shapes;
  SmallVector<TaskShapeChoice> choices;
  SmallVector<int64_t> durations;
  SmallVector<TaskShapeCost> costs;
  ProductionScheduledResult schedule;
  std::string costPath;
  std::vector<std::string> controlRoles;
  std::string graphId;
  bool control = false;
  std::vector<std::vector<std::string>> alternatePaths;
  int64_t score = std::numeric_limits<int64_t>::max();
  bool valid = false;
  std::string rejectReason;
  int64_t round = 0;
};

static std::vector<int64_t> numericTieKey(ArrayRef<TaskShapeChoice> choices,
                                         const ProductionScheduledResult &schedule) {
  std::vector<int64_t> key;
  for (const auto &choice : choices) {
    key.push_back(choice.shape.rows); key.push_back(choice.shape.cols);
  }
  for (const auto &placement : schedule.placements) {
    key.push_back(placement.task); key.push_back(placement.row);
    key.push_back(placement.col); key.push_back(placement.start);
    key.push_back(placement.end); key.push_back(placement.idleCycles);
  }
  for (unsigned task : schedule.dispatch) key.push_back(task);
  return key;
}
static bool archiveLess(const ArchiveRecord &lhs, const ArchiveRecord &rhs) {
  if (lhs.valid != rhs.valid) return lhs.valid > rhs.valid;
  if (lhs.score != rhs.score) return lhs.score < rhs.score;
  auto left = numericTieKey(lhs.choices, lhs.schedule);
  auto right = numericTieKey(rhs.choices, rhs.schedule);
  if (left != right) return left < right;
  if (lhs.key != rhs.key) return lhs.key < rhs.key;
  return lhs.id < rhs.id;
}
static bool stateLess(const SearchState &lhs, const SearchState &rhs) {
  if (lhs.scored != rhs.scored) return lhs.scored > rhs.scored;
  if (lhs.score != rhs.score) return lhs.score < rhs.score;
  auto left = numericTieKey(lhs.choices, lhs.schedule);
  auto right = numericTieKey(rhs.choices, rhs.schedule);
  if (left != right) return left < right;
  if (lhs.key != rhs.key) return lhs.key < rhs.key;
  return lhs.id < rhs.id;
}

static json::Object shapeObject(const NeighborhoodShape &shape) {
  return json::Object{{"task", shape.task},
                      {"rows", shape.rows},
                      {"cols", shape.cols}};
}

static json::Array stringArray(ArrayRef<std::string> values) {
  json::Array result;
  for (StringRef value : values)
    result.push_back(value.str());
  return result;
}

static json::Array choiceRecords(ArrayRef<TaskShapeChoice> choices) {
  json::Array result;
  for (const TaskShapeChoice &choice : choices)
    result.push_back(json::Object{{"task", choice.task},
                                  {"rows", choice.shape.rows},
                                  {"cols", choice.shape.cols},
                                  {"mapper_tile_rows", choice.shape.mapperRows},
                                  {"mapper_tile_cols", choice.shape.mapperCols},
                                  {"trip_count", choice.tripCount}});
  return result;
}

static json::Array costRecords(ArrayRef<TaskShapeChoice> choices,
                               ArrayRef<TaskShapeCost> costs,
                               ArrayRef<int64_t> durations) {
  json::Array result;
  for (size_t index = 0; index < choices.size(); ++index) {
    json::Object record{{"task", choices[index].task},
                        {"mapper_tile_rows", choices[index].shape.mapperRows},
                        {"mapper_tile_cols", choices[index].shape.mapperCols},
                        {"trip_count", choices[index].tripCount},
                        {"predicted_duration", durations[index]}};
    if (index < costs.size()) {
      record["predicted_ii"] = costs[index].predictedII;
      record["startup_cycles"] = costs[index].startupCycles;
      record["support_status"] = costs[index].supported ? "supported"
                                                           : "unsupported";
    }
    result.push_back(std::move(record));
  }
  return result;
}

static json::Array scheduleRecords(ArrayRef<TaskShapeChoice> choices,
                                   ArrayRef<ExactSchedulePlacement> placements) {
  json::Array result;
  for (const ExactSchedulePlacement &placement : placements) {
    if (placement.task >= choices.size())
      continue;
    const TaskShapeChoice &choice = choices[placement.task];
    result.push_back(json::Object{{"task", choice.task},
                                  {"row", placement.row},
                                  {"col", placement.col},
                                  {"rows", choice.shape.rows},
                                  {"cols", choice.shape.cols},
                                  {"start_cycle", placement.start},
                                  {"end_cycle", placement.end},
                                  {"idle_cycles", placement.idleCycles}});
  }
  return result;
}

static json::Array indexArray(ArrayRef<unsigned> values,
                              ArrayRef<TaskShapeChoice> choices) {
  json::Array result;
  for (unsigned index : values)
    if (index < choices.size())
      result.push_back(choices[index].task);
  return result;
}

static json::Object archiveObject(const ArchiveRecord &record) {
  json::Object object{
      {"record_type", "candidate"},
      {"schema", kSearchSchema},
      {"candidate_id", record.id},
      {"candidate_key", record.key},
      {"graph_facts_key", record.factKey},
      {"parent_candidate_id", record.parentId},
      {"candidate_path", record.candidatePath},
      {"action_path", stringArray(record.path)},
      {"alternate_action_paths", json::Array{}},
      {"shapes", json::Array{}},
      {"task_choices", choiceRecords(record.choices)},
      {"task_costs", costRecords(record.choices, record.costs,
                                  record.durations)},
      {"task_schedule", scheduleRecords(record.choices,
                                         record.schedule.placements)},
      {"dispatch_order", indexArray(record.schedule.dispatch,
                                     record.choices)},
      {"replayed_communication_edges",
       static_cast<int64_t>(record.schedule.replayedEdges)},
      {"communication_trace_known", record.schedule.replayedEdgesKnown},
      {"cost_catalogue_path", record.costPath},
      {"control_candidate", record.control},
      {"control_roles", stringArray(record.controlRoles)},
      {"graph_variant_id", record.graphId},
      {"predicted_whole_program_cycles", record.valid
                                             ? json::Value(record.score)
                                             : json::Value(nullptr)},
      {"valid", record.valid},
      {"reject_reason", record.rejectReason},
      {"round", record.round}};
  json::Array &shapes = *object.getArray("shapes");
  for (const NeighborhoodShape &shape : record.shapes)
    shapes.push_back(shapeObject(shape));
  json::Array &alternatePaths = *object.getArray("alternate_action_paths");
  for (ArrayRef<std::string> path : record.alternatePaths)
    alternatePaths.push_back(stringArray(path));
  return object;
}

static bool parseShapeArray(const json::Array *array,
                            std::vector<NeighborhoodShape> &shapes) {
  if (!array)
    return false;
  shapes.clear();
  for (const json::Value &value : *array) {
    const json::Object *object = value.getAsObject();
    auto task = object ? object->getString("task") : std::optional<StringRef>();
    auto rows = object ? object->getInteger("rows") : std::optional<int64_t>();
    auto cols = object ? object->getInteger("cols") : std::optional<int64_t>();
    if (!task || !rows || !cols || task->empty() || *rows <= 0 || *cols <= 0)
      return false;
    shapes.push_back({task->str(), *rows, *cols});
  }
  return true;
}

static bool parseShapeArrayFromObject(const json::Object &object,
                                      std::vector<NeighborhoodShape> &shapes) {
  if (parseShapeArray(object.getArray("shapes"), shapes))
    return true;
  if (parseShapeArray(object.getArray("task_shapes"), shapes))
    return true;
  return false;
}

static bool parseStringArray(const json::Array *array,
                             std::vector<std::string> &values) {
  if (!array)
    return false;
  values.clear();
  for (const json::Value &value : *array) {
    auto string = value.getAsString();
    if (!string)
      return false;
    values.push_back(string->str());
  }
  return true;
}

static std::optional<std::string> readCatalogGraphId(StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return std::nullopt;
  llvm::Expected<json::Value> parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return std::nullopt;
  json::Object *root = parsed->getAsObject();
  json::Object *metadata = root ? root->getObject("predictor_metadata") : nullptr;
  auto graph = metadata ? metadata->getString("source_graph_id")
                        : std::optional<StringRef>();
  return graph ? std::optional<std::string>(graph->str()) : std::nullopt;
}

static std::string decimal128(unsigned __int128 value) {
  if (value == 0)
    return "0";
  std::string text;
  while (value) {
    text.push_back(static_cast<char>('0' + value % 10));
    value /= 10;
  }
  std::reverse(text.begin(), text.end());
  return text;
}

static bool writeFiniteShapeDomain(StringRef path, func::FuncOp function,
                                   StringRef graphId, std::string &error) {
  static constexpr int64_t kRows[] = {1, 1, 2, 1, 3, 1, 2, 4};
  static constexpr int64_t kCols[] = {1, 2, 1, 3, 1, 4, 2, 1};
  json::Array factors;
  unsigned __int128 candidateCount = 1;
  SmallVector<TaskMetadata> metadata;
  std::string metadataError;
  FailureOr<SmallVector<TaskMetadata>> collected =
      collectAnalyticalTaskMetadata(function, metadataError);
  if (failed(collected)) {
    error = metadataError;
    return false;
  }
  metadata = std::move(*collected);
  for (const TaskMetadata &task : metadata) {
    json::Array shapes;
    for (unsigned index = 0; index < 8; ++index) {
      shapes.push_back(json::Object{{"kind", "rect"},
                                    {"rows", kRows[index]},
                                    {"cols", kCols[index]},
                                    {"cgra_count", kRows[index] * kCols[index]},
                                    {"cgra_shape",
                                     std::to_string(kRows[index]) + "x" +
                                         std::to_string(kCols[index])},
                                    {"mapper_tile_rows", kRows[index] * 4},
                                    {"mapper_tile_cols", kCols[index] * 4}});
    }
    factors.push_back(json::Object{{"task", task.name},
                                   {"trip_count", task.tripCount},
                                   {"shapes", std::move(shapes)}});
    candidateCount *= 8;
  }
  json::Object header{
      {"record_type", "space"},
      {"schema", kFactoredSpaceSchema},
      {"representation", "factored"},
      {"exact", true},
      {"function", function.getSymName()},
      {"graph_variant_id", graphId},
      {"search_scope", kFactoredSearchScope},
      {"shape_policy", kShapePolicy},
      {"shape_pruning_policy", kShapePruningPolicyNone},
      {"spatial_capacity_policy", kSpatialCapacityPolicy},
      {"max_cgras_per_task", 4},
      {"max_changed_tasks", -1},
      {"candidate_count", decimal128(candidateCount)},
      {"factors", std::move(factors)}};
  json::Object footer{{"record_type", "footer"},
                      {"schema", kFactoredSpaceSchema},
                      {"representation", "factored"},
                      {"ranked", false},
                      {"status", "unranked-factored-space"},
                      {"candidate_count", decimal128(candidateCount)}};
  std::string text = jsonText(json::Value(std::move(header))) + "\n" +
                     jsonText(json::Value(std::move(footer))) + "\n";
  return writeTextAtomically(path, text, error);
}

static bool runCurrentCostPredictor(
    ModuleOp module, func::FuncOp function, StringRef functionName,
    StringRef candidateId, StringRef outputDirectory, StringRef modelFile,
    StringRef cacheFile, StringRef checkpointDirectory,
    StringRef architectureContract, StringRef architecturePath,
    StringRef sourceRepository, StringRef sourceCommit,
    StringRef architectureTransferCatalog, StringRef modelNamespace,
    std::string &costPath, std::string &error, bool &fatalFailure) {
  fatalFailure = false;
  if (modelFile.empty() || cacheFile.empty() || architectureContract.empty() ||
      architecturePath.empty() || sourceRepository.empty() ||
      sourceCommit.empty()) {
    error = "changed graph requires model/cache and complete cost provenance";
    fatalFailure = true;
    return false;
  }
  std::string spacesDirectory = outputDirectory.str() + "/spaces";
  if (!ensureDirectory(spacesDirectory, error)) {
    fatalFailure = true;
    return false;
  }
  costPath = outputDirectory.str() + "/costs-" + candidateId.str() + ".json";
  std::string spacePath = spacesDirectory + "/" + candidateId.str() + ".jsonl";
  if (!writeFiniteShapeDomain(spacePath, function, candidateId, error)) {
    fatalFailure = true;
    return false;
  }
  std::string outputOptions =
      "function=" + quoteOption(functionName) +
      " space-file=" + quoteOption(spacePath) +
      " ensemble-file=" + quoteOption(modelFile) +
      " checkpoint-dir=" + quoteOption(checkpointDirectory) +
      " architecture-contract=" + quoteOption(architectureContract) +
      " architecture-path=" + quoteOption(architecturePath) +
      " source-git-repository=" + quoteOption(sourceRepository) +
      " source-git-commit=" + quoteOption(sourceCommit) +
      " graph-variant-id=" + quoteOption(candidateId) +
      " model-namespace=" + quoteOption(modelNamespace) +
      " cache=" + quoteOption(cacheFile) + " output=" +
      quoteOption(costPath);
  if (!architectureTransferCatalog.empty())
    outputOptions += " architecture-transfer-catalog=" +
                     quoteOption(architectureTransferCatalog);
  auto pass = mlir::amoeba::neura::createPredictAnalyticalTaskCostCatalogPass();
  if (failed(pass->initializeOptions(outputOptions,
                                     [&](const llvm::Twine &message) {
                                       error = message.str();
                                       return failure();
                                     }))) {
    if (error.empty())
      error = "predictor option parsing failed";
    fatalFailure = true;
    return false;
  }
  std::string diagnostic;
  if (!runPassOnModule(module, std::move(pass), error, diagnostic)) {
    if (!diagnostic.empty())
      error += ":" + diagnostic;
    return false;
  }
  if (!llvm::sys::fs::exists(costPath)) {
    error = "predictor completed without writing its cost catalogue";
    fatalFailure = true;
    return false;
  }
  return true;
}

static bool writeModule(ModuleOp module, StringRef path, std::string &error) {
  return writeAtomically(
      path,
      [&](llvm::raw_ostream &stream) {
        module.print(stream);
        stream << "\n";
        return true;
      },
      error);
}


static bool readFileBytes(StringRef path, std::string &bytes,
                          std::string &error, bool optional = false) {
  if (path.empty() && optional) { bytes.clear(); return true; }
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read bound contract " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  bytes = (*buffer)->getBuffer().str();
  return true;
}

static bool fatalStorageError(StringRef text) {
  std::string lower = text.lower();
  return StringRef(lower).contains("no space") ||
         StringRef(lower).contains("cannot write") ||
         StringRef(lower).contains("cannot open") ||
         StringRef(lower).contains("atomic") ||
         StringRef(lower).contains("permission denied") ||
         StringRef(lower).contains("disk guard") ||
         StringRef(lower).contains("cannot create temporary output") ||
         StringRef(lower).contains("cannot publish") ||
         StringRef(lower).contains("failed while writing") ||
         StringRef(lower).contains("read-only file system") ||
         StringRef(lower).contains("input/output error") ||
         StringRef(lower).contains("write failure");
}

static json::Object actionObject(const NeighborhoodAction &action) {
  json::Array primitives;
  for (const auto &primitive : action.primitives)
    primitives.push_back(json::Object{{"kind", primitive.kind},
        {"first_task", primitive.firstTask}, {"second_task", primitive.secondTask},
        {"mode", primitive.mode}, {"axis", primitive.axis},
        {"factor", primitive.factor}});
  return json::Object{{"family", action.family}, {"label", action.label},
      {"shape_task", action.shapeTask}, {"shape_rows", action.shapeRows},
      {"shape_cols", action.shapeCols}, {"canonical_reset", action.canonicalReset},
      {"primitives", std::move(primitives)}};
}

static bool parseAction(const json::Object &object, NeighborhoodAction &action) {
  auto family = object.getString("family"), label = object.getString("label"),
       task = object.getString("shape_task");
  auto rows = object.getInteger("shape_rows"), cols = object.getInteger("shape_cols");
  auto reset = object.getBoolean("canonical_reset");
  auto primitives = object.getArray("primitives");
  if (!family || !label || !task || !rows || !cols || !reset || !primitives)
    return false;
  action.family = family->str(); action.label = label->str();
  action.shapeTask = task->str(); action.shapeRows = *rows;
  action.shapeCols = *cols; action.canonicalReset = *reset;
  for (const auto &value : *primitives) {
    const auto *primitive = value.getAsObject();
    if (!primitive) return false;
    auto kind = primitive->getString("kind"), first = primitive->getString("first_task"),
         second = primitive->getString("second_task"), mode = primitive->getString("mode");
    auto axis = primitive->getInteger("axis"), factor = primitive->getInteger("factor");
    if (!kind || !first || !second || !mode || !axis || !factor) return false;
    action.primitives.push_back({kind->str(), first->str(), second->str(),
                                 mode->str(), *axis, *factor});
  }
  return true;
}

struct PendingNeighbor {
  uint64_t parent = 0;
  NeighborhoodAction action;
};

static bool parseArchiveObject(const json::Object &object, ArchiveRecord &record,
                               std::string &error) {
  auto id = object.getString("candidate_id"), key = object.getString("candidate_key"),
       fact = object.getString("graph_facts_key");
  auto valid = object.getBoolean("valid");
  if (!id || !key || !fact || !valid) {
    error = "checkpoint archive record lacks source-owned candidate identity";
    return false;
  }
  record.id = id->str(); record.key = key->str(); record.factKey = fact->str();
  record.valid = *valid;
  record.score = object.getInteger("predicted_whole_program_cycles").value_or(
      std::numeric_limits<int64_t>::max());
  record.candidatePath = object.getString("candidate_path").value_or("").str();
  record.parentId = object.getString("parent_candidate_id").value_or("").str();
  record.graphId = object.getString("graph_variant_id").value_or("").str();
  record.costPath = object.getString("cost_catalogue_path").value_or("").str();
  record.control = object.getBoolean("control_candidate").value_or(false);
  record.rejectReason = object.getString("reject_reason").value_or("").str();
  record.round = object.getInteger("round").value_or(0);
  if (!parseStringArray(object.getArray("action_path"), record.path) ||
      !parseShapeArray(object.getArray("shapes"), record.shapes)) {
    error = "checkpoint archive action or shape state is malformed";
    return false;
  }
  parseStringArray(object.getArray("control_roles"), record.controlRoles);
  if (auto alternatives = object.getArray("alternate_action_paths"))
    for (const auto &value : *alternatives) {
      std::vector<std::string> path;
      if (!parseStringArray(value.getAsArray(), path)) return false;
      record.alternatePaths.push_back(std::move(path));
    }
  if (!record.valid) return true;
  auto choices = object.getArray("task_choices"), costs = object.getArray("task_costs"),
       schedule = object.getArray("task_schedule"), dispatch = object.getArray("dispatch_order");
  if (!choices || !costs || !schedule || !dispatch || choices->size() != costs->size()) {
    error = "checkpoint scored record lacks cost and production schedule witnesses";
    return false;
  }
  std::map<std::string, unsigned> indices;
  for (const auto &value : *choices) {
    const auto *choice = value.getAsObject();
    if (!choice) return false;
    auto task = choice->getString("task");
    auto rows = choice->getInteger("rows"), cols = choice->getInteger("cols"),
         mapperRows = choice->getInteger("mapper_tile_rows"),
         mapperCols = choice->getInteger("mapper_tile_cols"),
         trip = choice->getInteger("trip_count");
    if (!task || !rows || !cols || !mapperRows || !mapperCols || !trip ||
        !indices.emplace(task->str(), record.choices.size()).second) return false;
    record.choices.push_back({task->str(), *trip, {*rows, *cols, *mapperRows, *mapperCols}});
  }
  for (unsigned index = 0; index < costs->size(); ++index) {
    const auto *cost = (*costs)[index].getAsObject();
    if (!cost || cost->getString("task") != record.choices[index].task) return false;
    auto ii = cost->getNumber("predicted_ii"), startup = cost->getNumber("startup_cycles");
    auto duration = cost->getInteger("predicted_duration");
    if (!ii || !startup || !duration) return false;
    record.costs.push_back({*ii, *startup, true});
    record.durations.push_back(*duration);
  }
  for (const auto &value : *schedule) {
    const auto *entry = value.getAsObject();
    if (!entry) return false;
    auto task = entry->getString("task");
    auto row = entry->getInteger("row"), col = entry->getInteger("col"),
         start = entry->getInteger("start_cycle"), end = entry->getInteger("end_cycle");
    if (!task || !row || !col || !start || !end || !indices.count(task->str())) return false;
    ExactSchedulePlacement placement;
    placement.task = indices[task->str()]; placement.row = *row; placement.col = *col;
    placement.start = *start; placement.end = *end;
    placement.idleCycles = entry->getInteger("idle_cycles").value_or(0);
    record.schedule.placements.push_back(placement);
  }
  for (const auto &value : *dispatch) {
    auto task = value.getAsString();
    if (!task || !indices.count(task->str())) return false;
    record.schedule.dispatch.push_back(indices[task->str()]);
  }
  record.schedule.order = record.schedule.dispatch;
  record.schedule.makespan = record.score;
  record.schedule.replayedEdges = object.getInteger("replayed_communication_edges").value_or(0);
  record.schedule.replayedEdgesKnown = object.getBoolean("communication_trace_known").value_or(false);
  return true;
}

static ArchiveRecord stateRecord(const SearchState &state) {
  ArchiveRecord record;
  record.id = state.id; record.key = state.key; record.factKey = state.factKey;
  record.parentId = state.parentId; record.path = state.path;
  record.shapes = state.shapes; record.choices = state.choices;
  record.costs = state.costs; record.durations = state.durations;
  record.schedule = state.schedule; record.costPath = state.costPath;
  record.control = state.control; record.controlRoles = state.controlRoles;
  record.score = state.score; record.valid = state.scored;
  record.rejectReason = state.rejectReason;
  return record;
}

static json::Object stateObject(const SearchState &state) {
  json::Object object = archiveObject(stateRecord(state));
  object["module_ir"] = neighborhoodReplaySourceText(state.module.get());
  object["key_reserved"] = state.keyReserved;
  return object;
}

static bool parseState(const json::Object &object, MLIRContext *context,
                       SearchState &state, std::string &error) {
  ArchiveRecord record;
  if (!parseArchiveObject(object, record, error)) return false;
  auto ir = object.getString("module_ir");
  if (!ir) { error = "checkpoint missing exact materialized module snapshot"; return false; }
  state.module = parseSourceString<ModuleOp>(*ir, context);
  if (!state.module) { error = "checkpoint materialized module snapshot does not parse"; return false; }
  state.id = record.id; state.key = record.key; state.factKey = record.factKey;
  state.parentId = record.parentId; state.path = std::move(record.path);
  state.shapes = std::move(record.shapes); state.choices = std::move(record.choices);
  state.costs = std::move(record.costs); state.durations = std::move(record.durations);
  state.schedule = std::move(record.schedule); state.costPath = record.costPath;
  state.control = record.control; state.controlRoles = std::move(record.controlRoles);
  state.score = record.score; state.scored = record.valid;
  state.keyReserved = object.getBoolean("key_reserved").value_or(true);
  return true;
}
static bool parseStageFixed(StringRef stage) {
  return stage == "S1" || stage == "s1" || stage == "shape" ||
         stage == "shape-only" ||
         stage.contains("fixed") || stage.contains("dependency-ready");
}

static bool parseCheckpointHeader(const json::Object &object, StringRef function,
                                  StringRef stage, StringRef sourceRepository,
                                  StringRef sourceCommit, StringRef architecture,
                                  StringRef protocol, std::string &error) {
  auto schema = object.getString("schema");
  auto savedFunction = object.getString("function");
  auto savedStage = object.getString("stage");
  auto savedRepository = object.getString("source_repository");
  auto savedCommit = object.getString("source_commit");
  auto savedArchitecture = object.getString("architecture_path");
  auto savedProtocol = object.getString("protocol_path");
  if (!schema || !savedFunction || !savedStage || !savedRepository ||
      !savedCommit || !savedArchitecture || !savedProtocol ||
      *schema != kCheckpointSchema || *savedFunction != function ||
      *savedStage != stage || *savedRepository != sourceRepository ||
      *savedCommit != sourceCommit || *savedArchitecture != architecture ||
      *savedProtocol != protocol) {
    error = "neighborhood checkpoint version/provenance binding mismatch";
    return false;
  }
  return true;
}

class SearchJointNeighborhoodPass
    : public PassWrapper<SearchJointNeighborhoodPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SearchJointNeighborhoodPass)

  SearchJointNeighborhoodPass() = default;
  SearchJointNeighborhoodPass(const SearchJointNeighborhoodPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override { return "search-joint-neighborhood"; }
  StringRef getDescription() const override {
    return "Budgeted cost-guided complete-program neighborhood search";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // These passes run in nested managers while the outer manager has already
    // entered its execution context. Declare their dialects here so loading a
    // K-reduction or replica materializer never happens during execution.
    mlir::amoeba::neura::createExtractJointTaskGraphFactsPass()
        ->getDependentDialects(registry);
    mlir::amoeba::neura::createPredictAnalyticalTaskCostCatalogPass()
        ->getDependentDialects(registry);
    mlir::amoeba::neura::createMaterializeJointTaskReplicasPass()
        ->getDependentDialects(registry);
    mlir::amoeba::neura::createMaterializeNeuraJointRewritePass()
        ->getDependentDialects(registry);
    mlir::amoeba::neura::createMaterializeNeuraKReductionPass()
        ->getDependentDialects(registry);
  }

  Option<std::string> outputDir{*this, "output-dir", llvm::cl::init("")};
  Option<std::string> functionName{*this, "function", llvm::cl::init("")};
  Option<std::string> stage{*this, "stage", llvm::cl::init("")};
  Option<std::string> parentCostFile{*this, "parent-cost-file",
                                     llvm::cl::init("")};
  Option<std::string> architecturePath{*this, "architecture-path",
                                       llvm::cl::init("")};
  Option<std::string> sourceRepository{*this, "source-repository",
                                      llvm::cl::init("")};
  Option<std::string> sourceCommit{*this, "source-commit", llvm::cl::init("")};
  Option<std::string> modelPath{*this, "model", llvm::cl::init("")};
  Option<std::string> cachePath{*this, "cache", llvm::cl::init("")};
  Option<std::string> architectureContract{
      *this, "architecture-contract",
      llvm::cl::init("neura-architecture-v1:amoeba_4x4_full_mesh_context12")};
  Option<std::string> architectureTransferCatalog{
      *this, "architecture-transfer-catalog", llvm::cl::init("")};
  Option<std::string> modelNamespace{
      *this, "model-namespace",
      llvm::cl::init("formal-max4-nohash-v2-exploratory")};
  Option<std::string> seedManifest{*this, "seed-manifest", llvm::cl::init("")};
  Option<std::string> previousWinner{*this, "previous-winner",
                                     llvm::cl::init("")};
  Option<int64_t> maxRounds{*this, "max-rounds", llvm::cl::init(kDefaultRounds)};
  Option<int64_t> maxCandidates{*this, "max-candidates",
                                llvm::cl::init(kDefaultCandidates)};
  Option<int64_t> beamWidth{*this, "beam-width", llvm::cl::init(kDefaultBeam)};
  Option<int64_t> diversitySlots{*this, "diversity-slots",
                                 llvm::cl::init(kDefaultDiversity)};
  Option<std::string> checkpoint{*this, "checkpoint", llvm::cl::init("")};
  Option<bool> resume{*this, "resume", llvm::cl::init(false)};

  Option<std::string> protocolFile{*this, "protocol",
      llvm::cl::init("config/protocols/amoeba_input0_neighborhood_v3.json")};
  Option<std::string> sourceContractFile{*this, "source-contract-file", llvm::cl::init("")};
  Option<int64_t> checkpointActions{*this, "checkpoint-actions", llvm::cl::init(128)};
  Option<int64_t> pauseAfterCandidates{*this, "test-pause-after-candidates", llvm::cl::init(0)};
  Option<int64_t> minimumFreeBytes{*this, "minimum-free-bytes", llvm::cl::init(16777216)};

  void runOnOperation() override {
    ModuleOp canonical = getOperation();
    std::string error;
    auto fatal = [&](StringRef message) {
      canonical.emitError() << message;
      signalPassFailure();
    };
    if (outputDir.empty() || functionName.empty() || stage.empty() ||
        maxRounds <= 0 || maxCandidates <= 0 || beamWidth <= 0 ||
        diversitySlots <= 0 || diversitySlots > beamWidth || checkpointActions <= 0 ||
        minimumFreeBytes < 0 || sourceContractFile.empty()) {
      fatal("neighborhood search requires positive budgets, valid beam/diversity limits, and source-contract-file");
      return;
    }
    if (!stageNumber(stage)) { fatal("unknown cumulative ablation stage"); return; }
    if (!parseStageFixed(stage) && maxCandidates < 2) {
      fatal("S2-S5 candidate budget must accommodate identity and previous measured control"); return;
    }
    if (!parseStageFixed(stage) && previousWinner.empty()) {
      fatal("S2-S5 require previous-winner measured selection outside native top5"); return;
    }
    const std::string &schedulerArchitecture =
        ::mlir::amoeba::getNeuraArchitectureSpecFile();
    if (schedulerArchitecture.empty() || architecturePath.empty()) {
      fatal("neighborhood search requires the production scheduler architecture-spec and bound architecture-path");
      return;
    }
    if (schedulerArchitecture != architecturePath.getValue()) {
      std::string schedulerBytes, boundBytes;
      if (!readFileBytes(schedulerArchitecture, schedulerBytes, error) ||
          !readFileBytes(architecturePath, boundBytes, error)) {
        fatal(error); return;
      }
      if (schedulerBytes != boundBytes) {
        fatal("production scheduler architecture-spec differs from the bound task-cost architecture");
        return;
      }
    }
    if (!ensureDirectory(outputDir, error) ||
        !ensureDirectory(outputDir + "/candidates", error)) { fatal(error); return; }
    const std::string candidatesDir = outputDir + "/candidates";
    const std::string checkpointPath = checkpoint.empty() ? outputDir + "/checkpoint.json" : checkpoint;
    journalPath = outputDir + "/archive.journal.jsonl";
    FailureOr<func::FuncOp> selected = selectTaskFunction(canonical, functionName, error);
    if (failed(selected)) { fatal(error); return; }
    auto metadata = collectAnalyticalTaskMetadata(*selected, error);
    if (failed(metadata) || metadata->empty()) { fatal(error.empty() ? "function has no analytical tasks" : error); return; }
    TaskShapeCostCache costCache;
    bool costCacheLoaded = false;
    if (!parentCostFile.empty()) {
      auto graphId = readCatalogGraphId(parentCostFile);
      if (!graphId || !costCache.load(parentCostFile, functionName, *metadata,
          sourceRepository, sourceCommit, architecturePath, *graphId, error)) {
        fatal("parent cost catalogue does not match current cost protocol: " + error); return;
      }
      costCacheLoaded = true;
    }
    if (!costCacheLoaded && modelPath.empty()) { fatal("validated parent costs or C++ predictor required"); return; }
    std::string binding;
    if (!buildBinding(canonical, binding, error)) { fatal(error); return; }
    bindingPath = checkpointPath + ".binding.json";
    if (resume) {
      std::string savedBinding;
      if (!readFileBytes(bindingPath, savedBinding, error) || savedBinding != binding) {
        fatal(error.empty() ? "neighborhood exact byte/options binding mismatch" : error); return;
      }
    } else {
      if (llvm::sys::fs::exists(checkpointPath) || llvm::sys::fs::exists(journalPath)) {
        fatal("search output already contains continuation state; use resume or a new isolated directory"); return;
      }
      if (!writeTextAtomically(bindingPath, binding, error) ||
          !writeTextAtomically(journalPath, "", error)) { fatal(error); return; }
    }
    std::vector<ArchiveRecord> archive;
    rankedArchivePrefix.clear();
    std::set<std::string> seenKeys;
    std::vector<SearchState> beam, generated;
    std::vector<PendingNeighbor> pending;
    uint64_t pendingCursor = 0, nextSerial = 0, scoredCount = 0, rejectedCount = 0;
    int64_t round = 0;
    std::string canonicalFactKey, stopReason;
    if (resume) {
      if (!restoreCheckpoint(canonical, checkpointPath, binding, round, nextSerial,
          scoredCount, rejectedCount, canonicalFactKey, archive, beam, generated,
          pending, pendingCursor, seenKeys, error)) { fatal(error); return; }
      // The prefix is derived from the exact restored archive. Rebuilding it
      // once avoids storing a second authoritative ranking in the checkpoint.
      for (size_t index = 0; index < archive.size(); ++index)
        if (archive[index].valid) insertRankedArchiveIndex(index, archive);
    } else {
      SearchState identity;
      identity.id = "neighborhood-" + std::to_string(nextSerial++);
      identity.module = canonical.clone(); identity.shapes = initialShapes(*selected);
      identity.control = true; identity.controlRoles.push_back("identity");
      if (!prepareCandidateKey(identity, functionName, outputDir, nextSerial, error) ||
          identity.key.empty()) { fatal(error.empty() ? "canonical identity facts unknown" : error); return; }
      canonicalFactKey = identity.factKey;
      seenKeys.insert(identity.key); identity.keyReserved = true;
      if (!evaluateCandidate(identity, canonical, functionName, stage, outputDir,
          candidatesDir, canonicalFactKey, costCacheLoaded, costCache, 0,
          scoredCount, nextSerial, error) || !identity.scored) {
        fatal(error.empty() ? "canonical identity cost/schedule unknown: " + identity.rejectReason : error); return;
      }
      if (!registerCandidate(identity, archive, seenKeys, rejectedCount,
          candidatesDir, error)) { fatal(error); return; }
      beam.push_back(std::move(identity));
      auto ingest = [&](StringRef path, StringRef controlRole, bool required) -> bool {
        if (path.empty()) return !required;
        std::string text;
        if (!readFileBytes(path, text, error)) {
          if (!required) { error.clear(); return true; }
          return false;
        }
        auto buffer = llvm::MemoryBuffer::getMemBuffer(text);
        unsigned imported = 0;
        for (llvm::line_iterator lines(*buffer, true); !lines.is_at_end() && imported < 16; ++lines) {
          auto value = json::parse(*lines);
          if (!value) { llvm::consumeError(value.takeError()); continue; }
          const auto *object = value->getAsObject();
          if (!object) continue;
          auto type = object->getString("record_type").value_or("");
          if (type != "selection" && type != "control" && type != "score") continue;
          std::string sourcePath = object->getString("candidate_path").value_or(
              object->getString("mapper_replay_path").value_or(
                  object->getString("mlir_path").value_or(""))).str();
          if (required && sourcePath.empty()) {
            error = "previous measured winner selection lacks its complete source module";
            return false;
          }
          SearchState seed;
          seed.id = "neighborhood-seed-" + std::to_string(nextSerial++);
          seed.module = sourcePath.empty() ? canonical.clone() :
              parseSourceFile<ModuleOp>(sourcePath, canonical.getContext());
          if (!seed.module) { if (required) { error = "previous winner module missing: " + sourcePath; return false; } continue; }
          if (!validateStageSeed(seed.module.get(), stage, error)) {
            if (required) return false;
            error.clear(); continue;
          }
          auto function = selectTaskFunction(seed.module.get(), functionName, error);
          if (failed(function)) { if (required) return false; error.clear(); continue; }
          if (!parseShapeArrayFromObject(*object, seed.shapes)) {
            const auto *score = object->getObject("score_record");
            const auto *costs = score ? score->getArray("task_costs") : object->getArray("task_costs");
            if (costs) for (const auto &entry : *costs) {
              const auto *cost = entry.getAsObject();
              if (!cost) continue;
              auto task = cost->getString("task");
              auto rows = cost->getInteger("mapper_tile_rows"), cols = cost->getInteger("mapper_tile_cols");
              if (task && rows && cols && *rows > 0 && *cols > 0 && *rows % 4 == 0 && *cols % 4 == 0)
                seed.shapes.push_back({task->str(), *rows / 4, *cols / 4});
            }
          }
          if (seed.shapes.empty()) seed.shapes = initialShapes(*function);
          seed.control = !controlRole.empty();
          if (seed.control) seed.controlRoles.push_back(controlRole.str());
          if (!prepareCandidateKey(seed, functionName, outputDir, nextSerial, error) || seed.key.empty()) {
            if (required) { if (error.empty()) error = "previous winner canonical facts unknown"; return false; }
            error.clear(); continue;
          }
          if (!seenKeys.insert(seed.key).second) {
            if (!mergeDuplicate(seed, archive, candidatesDir, error)) return false;
            ++imported; continue;
          }
          seed.keyReserved = true;
          if (!required && scoredCount + (!previousWinner.empty() ? 1 : 0) >= uint64_t(maxCandidates)) {
            seenKeys.erase(seed.key); break;
          }
          if (!evaluateCandidate(seed, canonical, functionName, stage, outputDir,
              candidatesDir, canonicalFactKey, costCacheLoaded, costCache, 0,
              scoredCount, nextSerial, error)) return false;
          if (required && !seed.scored) { error = "previous measured winner could not be scored: " + seed.rejectReason; return false; }
          if (!registerCandidate(seed, archive, seenKeys, rejectedCount, candidatesDir, error)) return false;
          if (seed.scored) beam.push_back(std::move(seed));
          ++imported;
          if (required) break;
        }
        if (required && !imported) { error = "previous winner file contains no legal complete selection"; return false; }
        return true;
      };
      if (!ingest(seedManifest, "", false) ||
          !ingest(previousWinner, "previous_stage_measured_winner", !previousWinner.empty())) {
        fatal(error.empty() ? "seed import failed" : error); return;
      }
      beam = selectBeam(std::move(beam), beamWidth, diversitySlots);
      if (!retainCanonicalBeam(beam, archive, error)) { fatal(error); return; }
      if (!saveCheckpoint(checkpointPath, round, nextSerial, scoredCount, rejectedCount,
          canonicalFactKey, archive, beam, generated, pending, pendingCursor, "", error)) { fatal(error); return; }
    }
    auto began = std::chrono::steady_clock::now();
    const int64_t previousElapsed = elapsedMilliseconds;
    uint64_t actionsSinceCheckpoint = 0;
    while (round < maxRounds && scoredCount < static_cast<uint64_t>(maxCandidates)) {
      if (beam.empty()) { stopReason = "no-new-legal-candidates"; break; }
      if (pending.empty()) {
        generated.clear(); pendingCursor = 0;
        for (auto indexed : llvm::enumerate(beam)) {
          auto actions = enumerateNeighborhoodActions(indexed.value().module.get(),
              functionName, indexed.value().shapes, stage, static_cast<unsigned>(round));
          for (auto &action : actions) pending.push_back({indexed.index(), std::move(action)});
        }
        if (!saveCheckpoint(checkpointPath, round, nextSerial, scoredCount, rejectedCount,
            canonicalFactKey, archive, beam, generated, pending, pendingCursor, "", error)) { fatal(error); return; }
      }
      while (pendingCursor < pending.size() && scoredCount < static_cast<uint64_t>(maxCandidates)) {
        const auto &neighbor = pending[pendingCursor];
        if (neighbor.parent >= beam.size()) { fatal("checkpoint pending parent outside beam"); return; }
        SearchState &parent = beam[neighbor.parent];
        SearchState child;
        child.id = "neighborhood-" + std::to_string(nextSerial++);
        child.parentId = parent.id; child.path = parent.path;
        child.path.push_back(actionSignature(neighbor.action));
        child.shapes = parent.shapes; child.module = parent.module->clone();
        std::string reason, diagnostic;
        bool materialized = applyNeighborhoodAction(child.module.get(), canonical,
            functionName, neighbor.action, child.shapes, reason, diagnostic);
        if (!materialized) {
          if (fatalStorageError(diagnostic) || fatalStorageError(reason)) { fatal(reason + ":" + diagnostic); return; }
          ++rejectedCount; appendRejected(archive, child,
              (reason.empty() ? "unsupported_or_unknown_action" : reason) + ":" + diagnostic, round);
        } else {
          auto function = selectTaskFunction(child.module.get(), functionName, error);
          if (failed(function)) { fatal(error); return; }
          auto oldShapes = child.shapes; synchronizeShapes(*function, oldShapes, child.shapes);
          if (!prepareCandidateKey(child, functionName, outputDir, nextSerial, error)) { fatal(error); return; }
          if (child.key.empty()) {
            ++rejectedCount; appendRejected(archive, child, child.rejectReason, round);
          } else if (!seenKeys.insert(child.key).second) {
            ++rejectedCount;
            if (!mergeDuplicate(child, archive, candidatesDir, error)) { fatal(error); return; }
          } else {
            child.keyReserved = true;
            if (!evaluateCandidate(child, canonical, functionName, stage, outputDir,
                candidatesDir, canonicalFactKey, costCacheLoaded, costCache,
                round, scoredCount, nextSerial, error) ||
                !registerCandidate(child, archive, seenKeys, rejectedCount, candidatesDir, error)) { fatal(error); return; }
            archive.back().round = round;
            if (child.scored) {
              generated.push_back(std::move(child));
              // Keep the expansion frontier bounded rather than serializing
              // every evaluated program in a round. Four diverse slots remain
              // available independently of the twelve cost-selected slots.
              generated = selectBeam(std::move(generated), beamWidth, diversitySlots);
            }
          }
        }
        ++pendingCursor; ++actionsSinceCheckpoint;
        bool pause = pauseAfterCandidates > 0 && scoredCount >= static_cast<uint64_t>(pauseAfterCandidates);
        if (actionsSinceCheckpoint >= static_cast<uint64_t>(checkpointActions) || pause) {
          if (!saveCheckpoint(checkpointPath, round, nextSerial, scoredCount, rejectedCount,
              canonicalFactKey, archive, beam, generated, pending, pendingCursor,
              pause ? "explicit-pause" : "", error)) { fatal(error); return; }
          actionsSinceCheckpoint = 0;
        }
        if (pause) { stopReason = "explicit-pause"; break; }
      }
      if (stopReason == "explicit-pause" || scoredCount >= static_cast<uint64_t>(maxCandidates)) break;
      ++round; pending.clear(); pendingCursor = 0;
      if (generated.empty()) { stopReason = "no-new-legal-candidates"; break; }
      beam = selectBeam(std::move(generated), beamWidth, diversitySlots);
      if (!retainCanonicalBeam(beam, archive, error)) { fatal(error); return; }
      generated.clear();
      if (!saveCheckpoint(checkpointPath, round, nextSerial, scoredCount, rejectedCount,
          canonicalFactKey, archive, beam, generated, pending, pendingCursor, "", error)) { fatal(error); return; }
    }
    if (stopReason.empty()) stopReason = round >= maxRounds ? "max-rounds" : "max-unique-candidates";
    elapsedMilliseconds = previousElapsed + std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - began).count();
    if (!saveCheckpoint(checkpointPath, round, nextSerial, scoredCount, rejectedCount,
        canonicalFactKey, archive, beam, generated, pending, pendingCursor, stopReason, error)) { fatal(error); return; }
    if (!writeOutputs(outputDir, checkpointPath, functionName, stage,
        sourceRepository, sourceCommit, architecturePath, protocolFile,
        maxRounds, maxCandidates, beamWidth, diversitySlots, round,
        scoredCount, rejectedCount, costCacheLoaded, !previousWinner.empty(),
        archive, stopReason, error)) { fatal(error); return; }
  }

private:
  std::string checkpointDirectoryForModel() const {
    if (modelPath.empty())
      return {};
    StringRef parent = llvm::sys::path::parent_path(modelPath);
    return parent.empty() ? std::string(".") : parent.str();
  }

  static void appendRejected(std::vector<ArchiveRecord> &archive,
                             const SearchState &state, StringRef reason,
                             int64_t round) {
    ArchiveRecord record;
    record.id = state.id;
    record.parentId = state.parentId;
    record.key = state.key; record.factKey = state.factKey;
    record.path = state.path;
    record.shapes = state.shapes;
    record.rejectReason = reason.str();
    record.round = round;
    record.valid = false;
    archive.push_back(std::move(record));
  }

  static void appendAlternatePath(std::vector<ArchiveRecord> &archive,
                                  const SearchState &state) {
    for (ArchiveRecord &record : archive) {
      if (record.valid && record.key == state.key) {
        record.alternatePaths.push_back(state.path);
        return;
      }
    }
  }

  bool prepareCandidateKey(SearchState &state, StringRef function,
                           StringRef outputDirectory, uint64_t serial,
                           std::string &error) {
    if (!state.key.empty())
      return true;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(state.module.get(), function, error);
    if (failed(selected)) {
      state.rejectReason = "task_function_missing";
      error.clear();
      return true;
    }
    FailureOr<SmallVector<TaskMetadata>> metadata =
        collectAnalyticalTaskMetadata(*selected, error);
    if (failed(metadata)) {
      state.rejectReason = "task_metadata_unknown";
      error.clear();
      return true;
    }
    if (!sameTaskSet(*metadata, state.shapes)) {
      state.rejectReason = "shape_state_does_not_cover_task_set";
      error.clear();
      return true;
    }
    if (!extractStructuralFacts(state.module.get(), function, outputDirectory,
                                serial, state.factKey, state.rejectReason,
                                error)) {
      if (fatalStorageError(error) || fatalStorageError(state.rejectReason))
        return false;
      if (state.rejectReason.empty())
        state.rejectReason = "graph_facts_unknown";
      error.clear();
      return true;
    }
    // Intern typed bodies and graph descriptors once. Archive rows and seen
    // keys refer to exact dictionary IDs, never repeated multi-KB IR strings.
    json::Array bodies, lineages;
    (*selected).walk([&](taskflow::TaskflowTaskOp task) {
      std::string ledger; llvm::raw_string_ostream ledgerStream(ledger);
      if (auto attr = task->getAttr("amoeba.neighborhood.partition_lineage.v1")) attr.print(ledgerStream);
      ledgerStream.flush(); lineages.push_back(ledger);
    });
    bool bodyFailure = false;
    (*selected).walk([&](::mlir::neura::KernelOp kernel) {
      if (bodyFailure) return;
      std::string body; llvm::raw_string_ostream stream(body);
      Operation *bodyClone = kernel->clone();
      bodyClone->removeAttr("kernel_symname"); bodyClone->removeAttr("sym_name");
      bodyClone->print(stream, OpPrintingFlags().printGenericOpForm()); stream.flush();
      bodyClone->destroy();
      std::string id;
      if (!internWitness(body, true, id, error)) { bodyFailure = true; return; }
      bodies.push_back(id);
    });
    if (bodyFailure) return false;
    std::string graphWitness = jsonText(json::Value(json::Object{
        {"canonical_structural_facts", state.factKey}, {"typed_kernel_body_ids", std::move(bodies)},
        {"partition_lineages", std::move(lineages)}}));
    if (!internWitness(graphWitness, false, state.factKey, error)) return false;
    state.key = state.factKey + "|" + shapeSignature(state.shapes);
    (*selected)->setAttr("amoeba.graph_variant_id",
                        StringAttr::get(state.module->getContext(), state.factKey));
    return true;
  }

  bool evaluateCandidate(SearchState &state, ModuleOp canonical,
                         StringRef function, StringRef stageName,
                         StringRef outputDirectory, StringRef candidatesDirectory,
                         StringRef canonicalFactKey, bool costCacheLoaded,
                         TaskShapeCostCache &costCache,
                         int64_t round, uint64_t &scoredCount,
                         uint64_t serial, std::string &error) {
    FailureOr<func::FuncOp> selected = selectTaskFunction(
        state.module.get(), function, error);
    if (failed(selected))
      return false;
    FailureOr<SmallVector<TaskMetadata>> metadata =
        collectAnalyticalTaskMetadata(*selected, error);
    if (failed(metadata))
      return false;
    if (!sameTaskSet(*metadata, state.shapes)) {
      error.clear();
      state.rejectReason = "shape_state_does_not_cover_task_set";
      state.scored = false;
      return true;
    }
    if (state.key.empty() &&
        !prepareCandidateKey(state, function, outputDirectory, serial, error)) {
      state.scored = false;
      return false;
    }
    if (state.key.empty()) {
      state.scored = false;
      error.clear();
      return true;
    }

    state.choices.clear();
    state.durations.clear();
    state.costs.clear();
    state.schedule = {};
    TaskShapeCostCache *activeCostCache = nullptr;
    if (state.factKey == canonicalFactKey && costCacheLoaded) {
      activeCostCache = &costCache; state.costPath = parentCostFile;
    } else {
      // Catalogues bind source task names, while candidate dedup uses ordered
      // graph facts. Names can differ across equivalent rewrite paths, so bind
      // the memoization key to the current catalogue task vocabulary as well.
      std::string graphCostKey = state.factKey;
      for (const auto &task : *metadata) graphCostKey += "|" + task.name;
      auto cached = graphCostCaches.find(graphCostKey);
      if (cached == graphCostCaches.end()) {
        auto knownPath = graphCostPaths.find(graphCostKey);
        if (knownPath != graphCostPaths.end()) state.costPath = knownPath->second;
        else {
          if (!diskGuard(error)) return false;
          OwningOpRef<ModuleOp> predictorModule = state.module->clone();
          auto predictorFunction = selectTaskFunction(*predictorModule, function, error);
          if (failed(predictorFunction)) return false;
          const std::string catalogueId = "graph-catalogue-" + std::to_string(graphCostPaths.size());
          bool fatalPredictorFailure = false;
          if (!runCurrentCostPredictor(*predictorModule, *predictorFunction,
                  function, catalogueId, outputDirectory, modelPath, cachePath,
                  checkpointDirectoryForModel(), architectureContract,
                  architecturePath, sourceRepository, sourceCommit,
                  architectureTransferCatalog, modelNamespace, state.costPath,
                  error, fatalPredictorFailure)) {
            if (fatalPredictorFailure || fatalStorageError(error)) return false;
            state.rejectReason = "current_graph_cost_unknown:" + error;
            state.scored = false; error.clear(); return true;
          }
          graphCostPaths[graphCostKey] = state.costPath;
        }
        auto graphId = readCatalogGraphId(state.costPath);
        auto oracle = std::make_unique<TaskShapeCostCache>();
        if (!graphId || !oracle->load(state.costPath, function, *metadata,
            sourceRepository, sourceCommit, architecturePath, *graphId, error)) {
          if (error.empty()) error = "generated cost catalogue has no bound graph ID";
          error = "generated_cost_catalogue_binding_failed:" + error;
          state.scored = false; return false;
        }
        cached = graphCostCaches.emplace(graphCostKey, std::move(oracle)).first;
      }
      state.costPath = graphCostPaths[graphCostKey]; activeCostCache = cached->second.get();
    }

    std::map<std::string, const NeighborhoodShape *> shapeMap;
    for (const NeighborhoodShape &shape : state.shapes)
      shapeMap[shape.task] = &shape;
    for (const TaskMetadata &task : *metadata) {
      const NeighborhoodShape *shape = shapeMap[task.name];
      if (!shape || shape->rows <= 0 || shape->cols <= 0 ||
          shape->rows > std::numeric_limits<int64_t>::max() / shape->cols ||
          shape->rows * shape->cols > 4) {
        state.rejectReason = "invalid_or_oversized_shape";
        state.scored = false;
        error.clear();
        return true;
      }
      RectShape rectangle{shape->rows, shape->cols, shape->rows * 4,
                          shape->cols * 4};
      TaskShapeChoice choice{task.name, task.tripCount, rectangle};
      std::string costError;
      const uint64_t oldHits = activeCostCache ? activeCostCache->hits() : 0;
      const uint64_t oldMisses = activeCostCache ? activeCostCache->misses() : 0;
      const TaskShapeCost *cost = activeCostCache
                                      ? activeCostCache->get(choice, costError)
                                      : nullptr;
      if (activeCostCache) {
        costCacheHits += activeCostCache->hits() - oldHits;
        costCacheMisses += activeCostCache->misses() - oldMisses;
      }
      if (!cost || !cost->supported) {
        state.rejectReason = costError.empty()
                                 ? "unsupported_task_shape_cost"
                                 : "unsupported_task_shape_cost:" + costError;
        state.scored = false;
        error.clear();
        return true;
      }
      std::optional<int64_t> duration =
          durationFromCost(*cost, task.tripCount, costError);
      if (!duration || *duration <= 0) {
        state.rejectReason = "invalid_task_shape_duration";
        state.scored = false;
        error.clear();
        return true;
      }
      state.choices.push_back(choice);
      state.durations.push_back(*duration);
      state.costs.push_back(*cost);
    }
    if (state.choices.empty()) {
      state.rejectReason = "empty_candidate";
      state.scored = false;
      error.clear();
      return true;
    }
    bool fixedDispatch = parseStageFixed(stageName);
    std::string scheduleError;
    ++scoredCount; ++productionSchedulerCalls;
    if (!scheduleWithAmoebaProductionPass(
            state.module.get(), *selected, state.choices, state.durations,
            state.id, fixedDispatch, state.schedule, scheduleError)) {
      state.rejectReason = "production_scheduler_rejected";
      if (!scheduleError.empty())
        state.rejectReason += ":" + scheduleError;
      state.scored = false;
      error.clear();
      return true;
    }
    state.score = state.schedule.makespan;
    state.scored = true;
    (void)round;
    (void)candidatesDirectory;
    return true;
  }

  bool registerCandidate(SearchState &state,
                                std::vector<ArchiveRecord> &archive,
                                std::set<std::string> &seenKeys,
                                uint64_t &rejectedCount,
                                StringRef candidatesDirectory,
                                std::string &error) {
    if (!state.scored || state.key.empty()) {
      ++rejectedCount;
      appendRejected(archive, state,
                     state.rejectReason.empty() ? "candidate_unknown"
                                                : state.rejectReason,
                     0);
      return true;
    }
    if (!state.keyReserved && !seenKeys.insert(state.key).second) {
      ++rejectedCount;
      appendAlternatePath(archive, state);
      return true;
    }
    ArchiveRecord record;
    record.id = state.id;
    record.key = state.key;
    record.factKey = state.factKey;
    record.parentId = state.parentId;
    record.path = state.path;
    record.shapes = state.shapes;
    record.choices = state.choices;
    record.durations = state.durations;
    record.costs = state.costs;
    record.schedule = state.schedule;
    record.costPath = state.costPath;
    record.control = state.control;
    record.controlRoles = state.controlRoles;
    auto function = selectTaskFunction(state.module.get(), "", error);
    if (failed(function)) return false;
    auto graph = (*function)->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
    record.graphId = graph ? graph.getValue().str() : state.id;
    record.score = state.score;
    record.valid = true;
    record.round = state.path.empty() ? 0 : 1;
    // Keep materialized IR bounded to candidates that can currently enter
    // native top-5.  A candidate that is outside this cutoff can never enter
    // it later because archive scores are immutable and only more rows are
    // added.  This avoids a full 20k-module snapshot while retaining every
    // final replay input.
    bool keepModule = state.control || rankedArchivePrefix.size() < 5;
    if (!keepModule) {
      keepModule = archiveLess(record, archive[rankedArchivePrefix.back()]);
    }
    if (keepModule) {
      std::string candidatePath =
          candidatesDirectory.str() + "/" + state.id + ".mlir";
      if (!diskGuard(error) || !writeModule(state.module.get(), candidatePath, error))
        return false;
      record.candidatePath = candidatePath;
    }
    const std::vector<size_t> previousPrefix = rankedArchivePrefix;
    archive.push_back(std::move(record));
    insertRankedArchiveIndex(archive.size() - 1, archive);
    // Only a previous top-5 entry can lose a non-control replay snapshot.
    // Controls stay live regardless of their predicted rank.
    for (size_t index : previousPrefix) {
      auto &old = archive[index];
      if (old.control || old.candidatePath.empty() ||
          llvm::is_contained(rankedArchivePrefix, index)) continue;
      obsoleteCandidatePaths.push_back(old.candidatePath); old.candidatePath.clear();
      dirtyJournalIndices.insert(index);
    }
    return true;
  }

  void insertRankedArchiveIndex(size_t index,
                               const std::vector<ArchiveRecord> &archive) {
    auto position = std::lower_bound(
        rankedArchivePrefix.begin(), rankedArchivePrefix.end(), index,
        [&](size_t left, size_t right) {
          return archiveLess(archive[left], archive[right]);
        });
    rankedArchivePrefix.insert(position, index);
    if (rankedArchivePrefix.size() > 5) rankedArchivePrefix.pop_back();
  }

  bool retainCanonicalBeam(std::vector<SearchState> &beam,
                           ArrayRef<ArchiveRecord> archive,
                           std::string &error) {
    const ArchiveRecord *identity = nullptr;
    for (const auto &record : archive)
      if (record.valid && llvm::is_contained(record.controlRoles,
                                            std::string("identity"))) {
        identity = &record;
        break;
      }
    if (!identity || identity->candidatePath.empty()) {
      error = "canonical identity control lacks its immutable source snapshot";
      return false;
    }
    for (const auto &state : beam)
      if (state.key == identity->key) return true;
    SearchState state;
    state.module = parseSourceFile<ModuleOp>(identity->candidatePath, &getContext());
    if (!state.module) {
      error = "canonical identity source snapshot no longer parses";
      return false;
    }
    state.id = identity->id; state.key = identity->key;
    state.factKey = identity->factKey; state.parentId = identity->parentId;
    state.path = identity->path; state.shapes = identity->shapes;
    state.choices = identity->choices; state.durations = identity->durations;
    state.costs = identity->costs; state.schedule = identity->schedule;
    state.costPath = identity->costPath; state.score = identity->score;
    state.control = identity->control; state.controlRoles = identity->controlRoles;
    state.keyReserved = true; state.scored = true;
    // The cost-selected prefix has width-diversity entries. When the beam is
    // full its last entry belongs to the diversity suffix, so reserving this
    // slot never evicts one of the twelve cost-selected candidates. The
    // identity is cloned from its archive witness without another score or
    // budget debit, and can replace or withdraw every historical decision.
    if (beam.size() >= static_cast<size_t>(beamWidth.getValue())) beam.pop_back();
    beam.push_back(std::move(state));
    return true;
  }

  static std::vector<SearchState> selectBeam(std::vector<SearchState> candidates,
                                             int64_t width,
                                             int64_t diversity) {
    std::stable_sort(candidates.begin(), candidates.end(), stateLess);
    std::vector<SearchState> selected;
    std::vector<size_t> indices;
    std::set<std::string> keys, configurations;
    const size_t costSlots = static_cast<size_t>(std::max<int64_t>(0, width - diversity));
    auto configuration = [](const SearchState &state) {
      return (state.path.empty() ? std::string("identity") : state.path.back()) +
             "|" + shapeSignature(state.shapes);
    };
    for (size_t index = 0; index < candidates.size() && indices.size() < costSlots; ++index) {
      if (!candidates[index].scored || candidates[index].key.empty() ||
          !keys.insert(candidates[index].key).second) continue;
      indices.push_back(index); configurations.insert(configuration(candidates[index]));
    }
    // Diversity selects complete legal candidates independently of immediate
    // improvement. Choose distinct action paths or parameter configurations;
    // this affects expansion only, never the global archive/native ranking.
    for (size_t index = 0; index < candidates.size() && indices.size() < size_t(width); ++index) {
      if (!candidates[index].scored || candidates[index].key.empty() ||
          keys.count(candidates[index].key) ||
          !configurations.insert(configuration(candidates[index])).second) continue;
      keys.insert(candidates[index].key); indices.push_back(index);
    }
    for (size_t index = 0; index < candidates.size() && indices.size() < size_t(width); ++index) {
      if (!candidates[index].scored || candidates[index].key.empty() ||
          !keys.insert(candidates[index].key).second) continue;
      indices.push_back(index);
    }
    // Decide all indices before moving. A moved-from state must never enter
    // the beam through a second traversal of the same vector.
    for (size_t index : indices) selected.push_back(std::move(candidates[index]));
    return selected;
  }

  static int stageNumber(StringRef value) {
    if (value == "S1" || value == "s1" || value == "shape-only" || value == "shape") return 1;
    if (value == "S2" || value == "s2" || value == "shape-temporal") return 2;
    if (value == "S3" || value == "s3" || value == "shape-temporal-replica") return 3;
    if (value == "S4" || value == "s4" || value == "shape-temporal-replica-tiling") return 4;
    if (value == "S5" || value == "s5" || value == "full-joint") return 5;
    return 0;
  }

  static bool validateStageSeed(ModuleOp module, StringRef stageName,
                                std::string &error) {
    const int number = stageNumber(stageName);
    bool valid = number != 0;
    module.walk([&](Operation *operation) {
      for (const NamedAttribute &attribute : operation->getAttrs()) {
        StringRef name = attribute.getName().strref();
        if ((number < 3 && name.starts_with("amoeba.replica.")) ||
            (number < 4 && (name.starts_with("amoeba.neura.tiling.") ||
                           name.starts_with("amoeba.tiling.") ||
                           name.starts_with("amoeba.semantic.k_block") ||
                           name == "amoeba.semantic.K_range")) ||
            (number < 5 && (name.starts_with("amoeba.neura.fusion.") ||
                           name.starts_with("amoeba.fusion.")))) valid = false;
      }
    });
    if (!valid) error = "seed contains a rewrite dimension outside this cumulative stage";
    return valid;
  }

  bool buildBinding(ModuleOp canonical, std::string &text, std::string &error) {
    json::Object files;
    auto bind = [&](StringRef role, StringRef path, bool optional = false) {
      std::string bytes;
      if (!readFileBytes(path, bytes, error, optional)) return false;
      files[role] = json::Object{{"path", path}, {"exact_bytes", std::move(bytes)}};
      return true;
    };
    if (!bind("protocol", protocolFile) || !bind("source_and_model_contract", sourceContractFile) ||
        !bind("architecture", architecturePath) || !bind("ensemble", modelPath, true) ||
        !bind("parent_costs", parentCostFile, true) ||
        !bind("architecture_transfer", architectureTransferCatalog, true) ||
        !bind("seed_manifest", seedManifest, true) || !bind("previous_winner", previousWinner, true)) return false;
    auto protocolBytes = files.getObject("protocol")->getString("exact_bytes");
    auto value = json::parse(*protocolBytes);
    if (!value) { llvm::consumeError(value.takeError()); error = "bound protocol is not JSON"; return false; }
    auto object = value->getAsObject();
    auto search = object ? object->getObject("search") : nullptr;
    if (!search || search->getInteger("max_rounds") != maxRounds.getValue() ||
        search->getInteger("max_unique_complete_candidates_scored") != maxCandidates.getValue() ||
        search->getInteger("beam_width") != beamWidth.getValue() ||
        search->getInteger("diversity_min_slots") != diversitySlots.getValue()) {
      error = "search budgets do not match the bound common protocol"; return false;
    }
    json::Object objectBinding{
        {"schema", "orbit-neighborhood-exact-binding-v1"},
        {"canonical_ir", neighborhoodReplaySourceText(canonical)},
        {"files", std::move(files)}, {"function", functionName.getValue()},
        {"stage", stage.getValue()}, {"source_repository", sourceRepository.getValue()},
        {"source_commit", sourceCommit.getValue()}, {"model_namespace", modelNamespace.getValue()},
        {"cache_path", cachePath.getValue()}, {"architecture_contract", architectureContract.getValue()},
        {"max_rounds", maxRounds.getValue()}, {"max_candidates", maxCandidates.getValue()},
        {"beam_width", beamWidth.getValue()}, {"diversity_slots", diversitySlots.getValue()},
        {"tie_key_policy", "cost-numeric-shape-schedule-graph-key-v1"},
        {"search_contract", kSearchSchema}, {"checkpoint_contract", kCheckpointSchema}};
    text = jsonText(json::Value(std::move(objectBinding))) + "\n";
    return true;
  }

  bool internWitness(StringRef bytes, bool body, std::string &id,
                     std::string &error) {
    auto &dictionary = body ? bodyWitnessIds : graphWitnessIds;
    auto existing = dictionary.find(bytes.str());
    if (existing != dictionary.end()) { id = existing->second; return true; }
    id = (body ? "body-" : "graph-") + std::to_string(dictionary.size());
    const std::string directory = outputDir + "/witnesses";
    const std::string path = directory + "/" + id + ".txt";
    if (!ensureDirectory(directory, error) || !diskGuard(error)) return false;
    if (llvm::sys::fs::exists(path)) {
      std::string old;
      if (!readFileBytes(path, old, error)) return false;
      // A crash may leave an uncommitted immutable dictionary entry. It is
      // reusable only by direct byte equality, never by a numeric ID alone.
      if (old != bytes) {
        error = "immutable graph/body witness file has different exact bytes";
        return false;
      }
    } else if (!writeTextAtomically(path, bytes, error)) return false;
    dictionary.emplace(bytes.str(), id); witnessPaths[id] = path;
    return true;
  }

  bool diskGuard(std::string &error) const {
    std::error_code ec;
    auto space = std::filesystem::space(outputDir.getValue(), ec);
    if (ec || space.available < uintmax_t(minimumFreeBytes.getValue())) {
      error = ec ? "disk guard cannot query filesystem: " + ec.message() :
                   "disk guard: insufficient free bytes for durable search output";
      return false;
    }
    return true;
  }

  bool mergeDuplicate(const SearchState &state, std::vector<ArchiveRecord> &archive,
                      StringRef candidatesDirectory, std::string &error) {
    for (size_t index = 0; index < archive.size(); ++index) {
      ArchiveRecord &record = archive[index];
      if (record.key != state.key || record.key.empty()) continue;
      if (state.control && !record.valid) {
        error = "required stage control duplicates a candidate without a legal production score";
        return false;
      }
      if (record.path != state.path &&
          !llvm::is_contained(record.alternatePaths, state.path))
        record.alternatePaths.push_back(state.path);
      for (const auto &role : state.controlRoles)
        if (!llvm::is_contained(record.controlRoles, role)) record.controlRoles.push_back(role);
      record.control |= state.control;
      if (state.control && record.candidatePath.empty()) {
        if (!diskGuard(error)) return false;
        record.candidatePath = candidatesDirectory.str() + "/" + record.id + ".mlir";
        if (!writeModule(state.module.get(), record.candidatePath, error)) return false;
      }
      dirtyJournalIndices.insert(index);
      ++duplicateCandidates;
      return true;
    }
    // Rejected actions with no canonical key do not reach this branch.
    error = "dedup key is reserved but absent from the archive";
    return false;
  }

  bool saveCheckpoint(StringRef path, int64_t round, uint64_t nextSerial,
                      uint64_t scoredCount, uint64_t rejectedCount,
                      StringRef canonicalFactKey, const std::vector<ArchiveRecord> &archive,
                      ArrayRef<SearchState> beam, ArrayRef<SearchState> generated,
                      ArrayRef<PendingNeighbor> pending, uint64_t pendingCursor,
                      StringRef stopReason, std::string &error) {
    if (!diskGuard(error)) return false;
    // Archive is append-only with explicit indexed amendments. Checkpoints
    // reference one committed journal prefix instead of rewriting 20k rows
    // every 128 actions. A crash beyond that prefix is ignored on recovery.
    std::string additions;
    llvm::raw_string_ostream stream(additions);
    for (size_t index : dirtyJournalIndices)
      if (index < journalArchiveCount)
        stream << json::Value(json::Object{{"index", int64_t(index)},
                {"candidate", archiveObject(archive[index])}}) << "\n";
    for (size_t index = journalArchiveCount; index < archive.size(); ++index)
      stream << json::Value(json::Object{{"index", int64_t(index)},
              {"candidate", archiveObject(archive[index])}}) << "\n";
    stream.flush();
    if (!additions.empty()) {
      std::ofstream journal(journalPath, std::ios::binary | std::ios::app);
      if (!journal) { error = "cannot open archive journal for append"; return false; }
      journal.write(additions.data(), additions.size()); journal.flush();
      if (!journal) { error = "archive journal write failure"; return false; }
      journal.close();
      if (journal.fail()) { error = "archive journal close/write failure"; return false; }
    }
    const uint64_t newJournalBytes = journalBytes + additions.size();
    json::Array beams, generatedStates, neighbors, graphCosts, witnesses;
    for (const auto &state : beam) beams.push_back(stateObject(state));
    for (const auto &state : generated) generatedStates.push_back(stateObject(state));
    for (const auto &neighbor : pending)
      neighbors.push_back(json::Object{{"parent", int64_t(neighbor.parent)},
                                       {"action", actionObject(neighbor.action)}});
    // Seen failed scoring keys survive restore too. Their canonical graph may
    // lack a score, so reconstructing seen solely from valid rows is unsafe.
    for (const auto &entry : graphCostPaths)
      graphCosts.push_back(json::Object{{"graph_key", entry.first}, {"path", entry.second}});
    for (const auto &entry : witnessPaths)
      witnesses.push_back(json::Object{{"id", entry.first}, {"path", entry.second}});
    json::Object root{
        {"schema", kCheckpointSchema}, {"record_type", "checkpoint"},
        {"function", functionName.getValue()}, {"stage", stage.getValue()},
        {"source_repository", sourceRepository.getValue()}, {"source_commit", sourceCommit.getValue()},
        {"architecture_path", architecturePath.getValue()}, {"protocol_path", protocolFile.getValue()},
        {"binding_witness_path", bindingPath}, {"canonical_fact_key", canonicalFactKey},
        {"round", round}, {"next_serial", int64_t(nextSerial)},
        {"scored_candidates", int64_t(scoredCount)}, {"rejected_candidates", int64_t(rejectedCount)},
        {"duplicate_candidates", int64_t(duplicateCandidates)},
        {"production_scheduler_calls", int64_t(productionSchedulerCalls)},
        {"cost_cache_hits", int64_t(costCacheHits)}, {"cost_cache_misses", int64_t(costCacheMisses)},
        {"elapsed_milliseconds", elapsedMilliseconds},
        {"pending_action_cursor", int64_t(pendingCursor)}, {"pending_neighbors", std::move(neighbors)},
        {"beam", std::move(beams)}, {"generated", std::move(generatedStates)},
        {"witness_dictionary", std::move(witnesses)}, {"graph_cost_catalogues", std::move(graphCosts)},
        {"archive_journal_path", journalPath}, {"archive_journal_bytes", int64_t(newJournalBytes)},
        {"archive_count", int64_t(archive.size())}, {"stop_reason", stopReason},
        {"search_scope", kSearchScope}, {"best_found", true}, {"exhaustive", false}};
    if (!writeTextAtomically(path, jsonText(json::Value(std::move(root))) + "\n", error)) return false;
    journalBytes = newJournalBytes; journalArchiveCount = archive.size(); dirtyJournalIndices.clear();
    // Snapshot deletion happens only after a checkpoint binds the amended
    // archive. A failed checkpoint always leaves the previous replay inputs.
    for (const auto &obsolete : obsoleteCandidatePaths) {
      std::error_code ec = llvm::sys::fs::remove(obsolete);
      if (ec && ec != std::errc::no_such_file_or_directory) {
        error = "cannot retire superseded candidate snapshot: " + ec.message(); return false;
      }
    }
    obsoleteCandidatePaths.clear();
    return true;
  }

  bool restoreCheckpoint(ModuleOp canonical, StringRef path, StringRef binding,
      int64_t &round, uint64_t &nextSerial, uint64_t &scoredCount,
      uint64_t &rejectedCount, std::string &canonicalFactKey,
      std::vector<ArchiveRecord> &archive, std::vector<SearchState> &beam,
      std::vector<SearchState> &generated, std::vector<PendingNeighbor> &pending,
      uint64_t &pendingCursor, std::set<std::string> &seenKeys, std::string &error) {
    std::string bytes;
    if (!readFileBytes(path, bytes, error)) return false;
    auto parsed = json::parse(bytes);
    if (!parsed) { llvm::consumeError(parsed.takeError()); error = "invalid neighborhood checkpoint JSON"; return false; }
    const auto *root = parsed->getAsObject();
    if (!root || !parseCheckpointHeader(*root, functionName, stage, sourceRepository,
        sourceCommit, architecturePath, protocolFile, error)) return false;
    auto savedRound = root->getInteger("round"), serial = root->getInteger("next_serial"),
         count = root->getInteger("scored_candidates"), rejected = root->getInteger("rejected_candidates"),
         cursor = root->getInteger("pending_action_cursor"), prefix = root->getInteger("archive_journal_bytes"),
         archiveCount = root->getInteger("archive_count");
    auto beamRecords = root->getArray("beam"), generatedRecords = root->getArray("generated"),
         neighbors = root->getArray("pending_neighbors");
    auto facts = root->getString("canonical_fact_key"), witness = root->getString("binding_witness_path"),
         savedJournal = root->getString("archive_journal_path");
    if (!savedRound || !serial || !count || !rejected || !cursor || !prefix || !archiveCount ||
        !beamRecords || !generatedRecords || !neighbors || !facts || !witness || !savedJournal ||
        *witness != bindingPath || *savedJournal != journalPath || *savedRound < 0 || *serial < 0 ||
        *count < 0 || *rejected < 0 || *cursor < 0 || uint64_t(*cursor) > neighbors->size() ||
        *prefix < 0 || *archiveCount < 0) {
      error = "checkpoint lacks exact frontier/journal continuation state"; return false;
    }
    if (!readFileBytes(journalPath, bytes, error) || bytes.size() < uint64_t(*prefix)) {
      if (error.empty()) error = "checkpoint committed archive journal prefix is truncated";
      return false;
    }
    const std::string committed = bytes.substr(0, *prefix);
    auto journalBuffer = llvm::MemoryBuffer::getMemBuffer(committed);
    for (llvm::line_iterator lines(*journalBuffer, true); !lines.is_at_end(); ++lines) {
      auto entry = json::parse(*lines);
      if (!entry) { llvm::consumeError(entry.takeError()); error = "invalid committed archive journal JSON"; return false; }
      auto object = entry->getAsObject();
      auto index = object ? object->getInteger("index") : std::nullopt;
      auto candidate = object ? object->getObject("candidate") : nullptr;
      if (!index || !candidate || *index < 0 || uint64_t(*index) > archive.size()) {
        error = "archive journal index is not an append or existing amendment"; return false;
      }
      ArchiveRecord record;
      if (!parseArchiveObject(*candidate, record, error)) return false;
      if (uint64_t(*index) == archive.size()) archive.push_back(std::move(record));
      else archive[*index] = std::move(record);
    }
    if (archive.size() != uint64_t(*archiveCount)) { error = "committed archive row count mismatch"; return false; }
    // Discard only this search's uncommitted journal suffix, after exact byte
    // binding and all committed rows have validated. No candidate traversal.
    if (bytes.size() != uint64_t(*prefix) && !writeTextAtomically(journalPath, committed, error)) return false;
    auto states = [&](const json::Array &records, std::vector<SearchState> &out) {
      for (const auto &value : records) {
        const auto *object = value.getAsObject(); SearchState state;
        if (!object || !parseState(*object, canonical.getContext(), state, error)) return false;
        if (!validateStageSeed(state.module.get(), stage, error)) return false;
        out.push_back(std::move(state));
      }
      return true;
    };
    if (!states(*beamRecords, beam) || !states(*generatedRecords, generated)) return false;
    for (const auto &value : *neighbors) {
      const auto *object = value.getAsObject();
      auto parent = object ? object->getInteger("parent") : std::nullopt;
      auto action = object ? object->getObject("action") : nullptr;
      PendingNeighbor neighbor;
      if (!parent || *parent < 0 || uint64_t(*parent) >= beam.size() || !action ||
          !parseAction(*action, neighbor.action)) { error = "malformed exact pending neighbor"; return false; }
      neighbor.parent = *parent; pending.push_back(std::move(neighbor));
    }
    for (const auto &record : archive) if (!record.key.empty()) seenKeys.insert(record.key);
    const auto *witnesses = root->getArray("witness_dictionary");
    if (!witnesses) { error = "checkpoint missing exact graph/body witness dictionary"; return false; }
    for (const auto &value : *witnesses) {
      const auto *object = value.getAsObject();
      auto id = object ? object->getString("id") : std::nullopt;
      auto witnessPath = object ? object->getString("path") : std::nullopt;
      std::string witnessBytes;
      if (!id || !witnessPath || !readFileBytes(*witnessPath, witnessBytes, error) ||
          witnessPaths.count(id->str())) { if (error.empty()) error = "malformed witness dictionary"; return false; }
      bool body = id->starts_with("body-");
      auto &dictionary = body ? bodyWitnessIds : graphWitnessIds;
      if (!dictionary.emplace(witnessBytes, id->str()).second) {
        error = "duplicate exact dictionary witness with distinct ID"; return false;
      }
      witnessPaths[id->str()] = witnessPath->str();
    }
    auto validateRestoredStates = [&](std::vector<SearchState> &states) {
      for (auto &state : states) {
        const std::string key = state.key, graphKey = state.factKey;
        const size_t oldWitnessCount = witnessPaths.size();
        state.key.clear(); state.factKey.clear();
        if (!prepareCandidateKey(state, functionName, outputDir, nextSerial,
                                 error) || state.key != key || state.factKey != graphKey ||
            witnessPaths.size() != oldWitnessCount) {
          if (error.empty()) error = "restored module differs from exact canonical graph/body/lineage witnesses";
          return false;
        }
      }
      return true;
    };
    if (!validateRestoredStates(beam) || !validateRestoredStates(generated)) return false;
    if (const auto *costs = root->getArray("graph_cost_catalogues"))
      for (const auto &value : *costs) {
        const auto *object = value.getAsObject();
        auto key = object ? object->getString("graph_key") : std::nullopt;
        auto costPath = object ? object->getString("path") : std::nullopt;
        if (!key || !costPath) { error = "malformed cached graph catalogue continuation"; return false; }
        graphCostPaths[key->str()] = costPath->str();
      }
    round = *savedRound; nextSerial = *serial; scoredCount = *count; rejectedCount = *rejected;
    pendingCursor = *cursor; canonicalFactKey = facts->str();
    journalBytes = *prefix; journalArchiveCount = archive.size();
    duplicateCandidates = root->getInteger("duplicate_candidates").value_or(0);
    productionSchedulerCalls = root->getInteger("production_scheduler_calls").value_or(0);
    costCacheHits = root->getInteger("cost_cache_hits").value_or(0);
    costCacheMisses = root->getInteger("cost_cache_misses").value_or(0);
    elapsedMilliseconds = root->getInteger("elapsed_milliseconds").value_or(0);
    (void)binding;
    return true;
  }

  static json::Object scoreRecord(const ArchiveRecord &record, StringRef shapeId) {
    return json::Object{
        {"schema", "amoeba-exact-joint-task-scores-v1"}, {"record_type", "score"},
        {"candidate_id", record.id}, {"shape_candidate_id", shapeId.str()},
        {"graph_variant_id", record.graphId}, {"valid", true},
        {"predicted_whole_program_cycles", record.score},
        {"score_source", "ML-task-cost-production-scheduler-explicit-communication"},
        {"task_costs", costRecords(record.choices, record.costs, record.durations)},
        {"task_schedule", scheduleRecords(record.choices, record.schedule.placements)},
        {"dispatch_order", indexArray(record.schedule.dispatch, record.choices)},
        {"replayed_communication_edges", int64_t(record.schedule.replayedEdges)},
        {"communication_trace_known", record.schedule.replayedEdgesKnown},
        {"best_found", true}, {"certified", false}, {"exhaustive", false}};
  }

  bool replaySelection(const ArchiveRecord &record, StringRef function,
                       StringRef stageName, StringRef outputDirectory,
                       json::Object &selection, std::string &error) {
    if (record.candidatePath.empty()) { error = "retained native candidate has no materialized source module"; return false; }
    auto module = parseSourceFile<ModuleOp>(record.candidatePath, &getContext());
    if (!module) { error = "retained native candidate module does not parse"; return false; }
    auto selected = selectTaskFunction(*module, function, error);
    if (failed(selected)) return false;
    auto graph = (*selected)->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
    if (!graph || graph.getValue() != record.graphId) {
      error = "replay candidate graph label differs from saved C++ archive"; return false;
    }
    const std::string shapeId = "shape-" + record.id;
    const std::string replayDirectory = outputDirectory.str() + "/replay";
    if (!ensureDirectory(replayDirectory, error) || !diskGuard(error)) return false;
    const std::string shapePath = replayDirectory + "/" + record.id + "-shape.json";
    const std::string scorePath = replayDirectory + "/" + record.id + "-score.jsonl";
    Candidate candidate; candidate.id = shapeId; candidate.choices = record.choices;
    if (!writeNeighborhoodShapeSelection(*module, *selected, candidate, shapePath, error)) return false;
    int64_t iterations = 64;
    if (auto configured = (*selected)->getAttrOfType<IntegerAttr>("joint_scheduling_fixed_point_max_iterations"))
      iterations = configured.getInt();
    json::Object header{
        {"record_type", "header"}, {"schema", "amoeba-exact-joint-task-scores-v1"},
        {"function", function}, {"schedule_space", "production-scheduler"},
        {"temporal_model", "production-scheduler"},
        {"dispatch_policy", parseStageFixed(stageName) ? "fixed" : "critical-path"},
        {"start_policy", "production-spatial-temporal-scheduler"},
        {"temporal_search_scope", "production-task-scheduler-only"},
        {"production_fixed_point_max_iterations", iterations},
        {"search_scope", kSearchScope}, {"best_found", true}, {"exhaustive", false},
        {"source_repository", sourceRepository.getValue()}, {"source_commit", sourceCommit.getValue()},
        {"architecture_path", architecturePath.getValue()}, {"source_binding_witness", bindingPath}};
    json::Object footer{{"record_type", "footer"}, {"schema", "amoeba-exact-joint-task-scores-v1"},
                        {"scored_count", 1}, {"status", "best-found"}, {"certified", false}};
    std::string text = jsonText(json::Value(std::move(header))) + "\n" +
        jsonText(json::Value(scoreRecord(record, shapeId))) + "\n" +
        jsonText(json::Value(std::move(footer))) + "\n";
    if (!writeTextAtomically(scorePath, text, error)) return false;
    selection = archiveObject(record);
    selection["record_type"] = "selection";
    selection["shape_candidate_id"] = shapeId;
    selection["score_file"] = scorePath;
    selection["shape_manifest_file"] = shapePath;
    selection["score_record"] = scoreRecord(record, shapeId);
    selection["certified"] = false; selection["best_found"] = true;
    selection["source_binding_witness"] = bindingPath;
    selection["transformations"] = json::Object{
        {"shape", choiceRecords(record.choices)}, {"action_path", stringArray(record.path)},
        {"lineage_source", record.candidatePath}};
    return true;
  }

  bool writeOutputs(
      StringRef outputDirectory, StringRef checkpointPath, StringRef function,
      StringRef stageName, StringRef repository, StringRef commit,
      StringRef architecture, StringRef protocol, int64_t maxRoundsValue,
      int64_t maxCandidatesValue, int64_t beamWidthValue,
      int64_t diversityValue, int64_t round, uint64_t scoredCount,
      uint64_t rejectedCount, bool costCacheLoaded,
      bool previousWinnerRequested, ArrayRef<ArchiveRecord> archive,
      StringRef stopReason, std::string &error) {
    std::vector<const ArchiveRecord *> ranked;
    json::Object reasons;
    uint64_t validCount = 0;
    for (const auto &record : archive) {
      if (record.valid) { ranked.push_back(&record); ++validCount; }
      else {
        StringRef reason = record.rejectReason;
        std::string category = reason.take_front(reason.find(':')).str();
        if (category.empty()) category = "unknown";
        reasons[category] = reasons.getInteger(category).value_or(0) + 1;
      }
    }
    reasons["canonical_duplicate"] = int64_t(duplicateCandidates);
    std::stable_sort(ranked.begin(), ranked.end(), [](auto *left, auto *right) {
      return archiveLess(*left, *right);
    });
    if (ranked.size() > 5) ranked.resize(5);
    std::string archiveText;
    llvm::raw_string_ostream archiveStream(archiveText);
    for (const auto &record : archive) archiveStream << json::Value(archiveObject(record)) << "\n";
    archiveStream.flush();
    if (!diskGuard(error) || !writeTextAtomically(outputDirectory.str() + "/archive.jsonl", archiveText, error)) return false;
    json::Object metadata{
        {"schema", kSearchSchema}, {"search_scope", kSearchScope}, {"function", function},
        {"stage", stageName}, {"source_repository", repository}, {"source_commit", commit},
        {"architecture_path", architecture}, {"protocol_path", protocol},
        {"source_binding_witness", bindingPath}, {"max_rounds", maxRoundsValue},
        {"max_candidates", maxCandidatesValue}, {"beam_width", beamWidthValue},
        {"diversity_slots", diversityValue}, {"rounds", round}, {"rounds_completed", round},
        {"unique_complete_candidates_scored", int64_t(scoredCount)},
        {"unique_scored_candidates", int64_t(scoredCount)}, {"unique_valid_candidates", int64_t(validCount)},
        {"rejected_or_duplicate_candidates", int64_t(rejectedCount)},
        {"cache_hits", int64_t(costCacheHits)}, {"cache_misses", int64_t(costCacheMisses)},
        {"reject_reasons", std::move(reasons)}, {"elapsed_seconds", elapsedMilliseconds / 1000.0},
        {"production_scheduler_calls", int64_t(productionSchedulerCalls)},
        {"graph_cost_catalogues", int64_t(graphCostPaths.size())},
        {"cost_catalogue_loaded", costCacheLoaded}, {"best_found", true}, {"exhaustive", false},
        {"stop_reason", stopReason}, {"checkpoint", checkpointPath},
        {"identity_control_retained", true}, {"previous_winner_requested", previousWinnerRequested},
        {"native_top5_required", true}, {"numeric_trace_sram_separate", true},
        {"native_shortlist_count", int64_t(ranked.size())}};
    json::Object header = metadata; header["record_type"] = "header";
    std::string topText = jsonText(json::Value(std::move(header))) + "\n";
    for (size_t rank = 0; rank < ranked.size(); ++rank) {
      json::Object selection;
      if (!replaySelection(*ranked[rank], function, stageName, outputDirectory, selection, error)) return false;
      selection["rank"] = int64_t(rank);
      topText += jsonText(json::Value(std::move(selection))) + "\n";
    }
    json::Object footer = metadata; footer["record_type"] = "footer"; footer["status"] = "best-found";
    topText += jsonText(json::Value(std::move(footer))) + "\n";
    std::string controlsText;
    std::set<std::string> roles;
    for (const auto &record : archive) {
      if (!record.valid || !record.control) continue;
      for (const auto &role : record.controlRoles) {
        if (!roles.insert(role).second) { error = "duplicate stage control role"; return false; }
        json::Object selection;
        if (!replaySelection(record, function, stageName, outputDirectory, selection, error)) return false;
        selection["record_type"] = "control"; selection["control_role"] = role;
        controlsText += jsonText(json::Value(std::move(selection))) + "\n";
      }
    }
    if (!roles.count("identity") || (previousWinnerRequested && !roles.count("previous_stage_measured_winner"))) {
      error = "required stage native controls missing from C++ archive"; return false;
    }
    return writeTextAtomically(outputDirectory.str() + "/top5.jsonl", topText, error) &&
           writeTextAtomically(outputDirectory.str() + "/controls.jsonl", controlsText, error) &&
           writeTextAtomically(outputDirectory.str() + "/search-summary.json", jsonText(json::Value(std::move(metadata))) + "\n", error);
  }

  std::string journalPath, bindingPath;
  uint64_t journalBytes = 0, journalArchiveCount = 0, duplicateCandidates = 0;
  uint64_t productionSchedulerCalls = 0, costCacheHits = 0, costCacheMisses = 0;
  int64_t elapsedMilliseconds = 0;
  std::set<size_t> dirtyJournalIndices;
  std::vector<std::string> obsoleteCandidatePaths;
  std::vector<size_t> rankedArchivePrefix;
  std::map<std::string, std::string> graphCostPaths;
  std::map<std::string, std::string> bodyWitnessIds, graphWitnessIds, witnessPaths;
  std::map<std::string, std::unique_ptr<TaskShapeCostCache>> graphCostCaches;

};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createSearchJointNeighborhoodPass() {
  return std::make_unique<SearchJointNeighborhoodPass>();
}
} // namespace mlir::amoeba::neura
