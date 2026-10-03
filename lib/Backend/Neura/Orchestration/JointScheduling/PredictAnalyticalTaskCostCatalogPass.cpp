//===- PredictAnalyticalTaskCostCatalogPass.cpp ----------------*- C++ -*-===//
//
// Predicts an exploratory task-shape II catalogue from a C++ feature query
// file and the pinned no-SHA formal max-four JSON ensemble.  The current
// route-expanded Neura module is the authoritative mapper/DFG frontend. An
// optional feature query file is checked against that body and is never
// trusted as an independent task-name lookup. This pass performs no Python
// calls and never measures a native mapper result.
//
// Feature-query contract (one JSON object, deliberately independent of task
// names in the cache identity):
//   schema = "orbit-cgra-ii-feature-queries-v1"
//   function, architecture_contract, feature_contract_id,
//   feature_extractor, shape_protocol_id
//   entries = [{ task, body_structural_text, mapper_tile_rows,
//                mapper_tile_cols, rec_mii, res_mii,
//                analytical_lower_bound, startup_cycles, features[156] }]
// body_structural_text is the canonical mapper-visible body serialization
// supplied by the C++ frontend.  It is not a digest and must not be empty.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalMLPInference.h"
#include "AnalyticalTaskCandidateCommon.h"
#include "MapperCounterBounds.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/MapperFeatureExtractor.h"
#include "Conversion/NeuraConversionPasses.h"
#include "NeuraDialect/NeuraPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <fcntl.h>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <sys/file.h>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;
using llvm::ArrayRef;
using orbit::mapper_features::FeatureVector;
using orbit::mapper_features::RouteExpandedGraph;
namespace json = llvm::json;

namespace {

constexpr llvm::StringLiteral kFeatureQuerySchema =
    "orbit-cgra-ii-feature-queries-v1";
constexpr llvm::StringLiteral kCostProvenanceSchema =
    "orbit-cost-provenance-v1";
constexpr llvm::StringLiteral kArchitectureSchema = "neura-architecture-v1";
constexpr llvm::StringLiteral kModelName = "formal-max4-nohash-v2";
constexpr llvm::StringLiteral kModelStatus = "exploratory";
constexpr llvm::StringLiteral kCacheArchitecturePrefix =
    "neura-architecture-v1:";

struct FeatureQuery {
  std::string task;
  std::string bodyStructuralText;
  int64_t rows = 0;
  int64_t cols = 0;
  double recMII = 0.0;
  double resMII = 0.0;
  double lowerBound = 0.0;
  double startupCycles = 0.0;
  std::vector<double> features;
};

// The model's formal max-four contract is deliberately enumerated here in
// the same fixed orientation used by the candidate-space pass.  The model
// loader validates this list against the JSON protocol before any query is
// evaluated.
constexpr std::array<std::pair<int64_t, int64_t>, 8> kFormalMax4Shapes = {{
    {4, 4}, {4, 8}, {8, 4}, {4, 12},
    {12, 4}, {4, 16}, {8, 8}, {16, 4},
}};

struct CurrentTaskBody {
  std::string task;
  // This is the temporary function after the same kernel-body cloning and
  // yield-to-return preparation used by the production mapper.  Keep the
  // operation alive through mapperInput while feature extraction reads its
  // current mapper-visible body.
  func::FuncOp mapperFunction;
  RouteExpandedGraph graph;
  std::string bodyStructuralText;
  double startupCycles = 0.0;
};

// Serialize the persistent cache's read/lookup/inference/write transaction.
// The lock is advisory and keyed by the cache path, so independent graph
// worker processes cannot load the same old snapshot and overwrite one
// another's rows.  Atomic publication of the JSON file remains the cache's
// durability mechanism; this lock supplies the missing transaction boundary.
class MLCostCacheLock {
  int fd = -1;

public:
  bool acquire(llvm::StringRef cachePath, std::string &error) {
    if (cachePath.empty())
      return true;
    std::string lockPath = cachePath.str() + ".lock";
    fd = ::open(lockPath.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
      error = "cannot open ML cost cache lock " + lockPath;
      return false;
    }
    while (::flock(fd, LOCK_EX) != 0) {
      if (errno == EINTR)
        continue;
      ::close(fd);
      fd = -1;
      error = "cannot acquire ML cost cache lock " + lockPath;
      return false;
    }
    return true;
  }

  ~MLCostCacheLock() {
    if (fd >= 0) {
      ::flock(fd, LOCK_UN);
      ::close(fd);
    }
  }
};

static bool decimalField(const json::Object &object, llvm::StringRef key,
                         std::string &value, std::string &error) {
  if (std::optional<int64_t> numeric = object.getInteger(key)) {
    if (*numeric <= 0) {
      error = "candidate count must be positive";
      return false;
    }
    value = std::to_string(*numeric);
    return true;
  }
  std::optional<llvm::StringRef> text = object.getString(key);
  if (!text || text->empty() ||
      llvm::any_of(*text, [](char character) {
        return character < '0' || character > '9';
      })) {
    error = "candidate count is not a positive decimal";
    return false;
  }
  llvm::StringRef normalized = text->ltrim('0');
  if (normalized.empty()) {
    error = "candidate count must be positive";
    return false;
  }
  value = normalized.str();
  return true;
}

static bool readCandidateCount(llvm::StringRef path, llvm::StringRef function,
                               llvm::StringRef graphVariantId,
                               std::string &candidateCount,
                               std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read candidate space " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  bool found = false;
  for (llvm::line_iterator lines(**buffer, /*SkipBlanks=*/true);
       !lines.is_at_end(); ++lines) {
    llvm::Expected<json::Value> parsed = json::parse(*lines);
    if (!parsed) {
      error = "candidate space is not valid JSONL: " +
              llvm::toString(parsed.takeError());
      return false;
    }
    const json::Object *object = parsed->getAsObject();
    if (!object) {
      error = "candidate space record is not an object";
      return false;
    }
    std::optional<llvm::StringRef> recordType = object->getString("record_type");
    if (recordType && *recordType != "header" && *recordType != "space" &&
        *recordType != "footer")
      continue;
    if (!recordType && !object->get("candidate_count") &&
        !object->get("declared_candidate_count"))
      continue;
    if (std::optional<llvm::StringRef> actualFunction =
            object->getString("function")) {
      if (*actualFunction != function) {
        error = "candidate space function does not match the selected IR";
        return false;
      }
    }
    if (std::optional<llvm::StringRef> actualGraph =
            object->getString("graph_variant_id")) {
      if (*actualGraph != graphVariantId) {
        error = "candidate space graph variant does not match the requested "
                "graph";
        return false;
      }
    }
    llvm::StringRef countKey = object->get("candidate_count")
                                   ? llvm::StringRef("candidate_count")
                                   : llvm::StringRef("declared_candidate_count");
    if (!decimalField(*object, countKey, candidateCount, error))
      return false;
    found = true;
    break;
  }
  if (!found) {
    error = "candidate space has no header candidate_count";
    return false;
  }
  return true;
}

static bool readTextFile(llvm::StringRef path, std::string &text,
                         std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read architecture source " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  text = (*buffer)->getBuffer().str();
  if (text.empty()) {
    error = "architecture source is empty: " + path.str();
    return false;
  }
  return true;
}

static bool getString(const json::Object &object, llvm::StringRef key,
                      std::string &value, std::string &error) {
  std::optional<llvm::StringRef> raw = object.getString(key);
  if (!raw || raw->empty()) {
    error = "feature query is missing non-empty string field \"" + key.str() +
            "\"";
    return false;
  }
  value = raw->str();
  return true;
}

static bool getInteger(const json::Object &object, llvm::StringRef key,
                       int64_t &value, std::string &error) {
  std::optional<int64_t> raw = object.getInteger(key);
  if (!raw) {
    error = "feature query is missing integer field \"" + key.str() + "\"";
    return false;
  }
  value = *raw;
  return true;
}

static bool getNumber(const json::Object &object, llvm::StringRef key,
                      double &value, std::string &error) {
  std::optional<double> raw = object.getNumber(key);
  if (!raw || !std::isfinite(*raw)) {
    error = "feature query is missing finite number field \"" + key.str() +
            "\"";
    return false;
  }
  value = *raw;
  return true;
}

static bool requireString(const json::Object &object, llvm::StringRef key,
                          llvm::StringRef expected, std::string &error) {
  std::optional<llvm::StringRef> actual = object.getString(key);
  if (!actual || *actual != expected) {
    error = "feature query field \"" + key.str() + "\" does not match \"" +
            expected.str() + "\"";
    return false;
  }
  return true;
}

static bool rejectTransferDigestKeys(const json::Value &value,
                                     std::string &error) {
  if (const json::Object *object = value.getAsObject()) {
    for (const auto &item : *object) {
      std::string key = item.first.str();
      std::string lower;
      lower.reserve(key.size());
      for (char character : key)
        lower.push_back(static_cast<char>(std::tolower(
            static_cast<unsigned char>(character))));
      if (lower == "hash" || lower == "sha256" ||
          llvm::StringRef(lower).ends_with("_hash") ||
          llvm::StringRef(lower).ends_with("_sha256")) {
        error = "architecture transfer evidence contains a forbidden digest "
                "identity field " +
                key;
        return false;
      }
      if (!rejectTransferDigestKeys(item.second, error))
        return false;
    }
    return true;
  }
  if (const json::Array *array = value.getAsArray()) {
    for (const json::Value &element : *array)
      if (!rejectTransferDigestKeys(element, error))
        return false;
  }
  return true;
}

static bool readArchitectureTransferEvidence(
    llvm::StringRef path, llvm::StringRef architecturePath,
    std::optional<json::Object> &evidence, std::string &error) {
  if (path.empty())
    return true;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read architecture transfer catalogue " + path.str() +
            ": " + buffer.getError().message();
    return false;
  }
  llvm::Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "cannot parse architecture transfer catalogue " + path.str() +
            ": " + llvm::toString(parsed.takeError());
    return false;
  }
  if (!rejectTransferDigestKeys(*parsed, error))
    return false;
  json::Object *root = parsed->getAsObject();
  json::Object *metadata = root ? root->getObject("predictor_metadata") : nullptr;
  json::Object *transfer =
      metadata ? metadata->getObject("architecture_transfer") : nullptr;
  if (!transfer) {
    error = "architecture transfer catalogue has no predictor_metadata.";
    error += "architecture_transfer object";
    return false;
  }
  if (!requireString(*transfer, "schema", "orbit-architecture-transfer-v1",
                    error) ||
      !requireString(*transfer, "mode",
                     "explicit-structural-equivalence-task-ii", error) ||
      !requireString(*transfer, "actual_architecture_path", architecturePath,
                     error))
    return false;
  std::optional<bool> productionReady = transfer->getBoolean("production_ready");
  if (!productionReady || *productionReady) {
    error = "architecture transfer evidence must remain exploratory "
            "(production_ready=false)";
    return false;
  }
  std::optional<llvm::StringRef> architectureName =
      transfer->getString("actual_architecture_name");
  if (!architectureName || architectureName->empty()) {
    error = "architecture transfer evidence has no actual architecture name";
    return false;
  }
  const json::Object *checks = transfer->getObject("structural_checks");
  if (!checks || !requireString(*checks, "directed_link_graph", "matched",
                                error) ||
      !requireString(*checks, "mapper_visible_resources", "matched", error) ||
      !requireString(*checks, "per_cgra_tile", "matched-4x4", error) ||
      !requireString(*checks, "physical_grid", "matched-4x4", error))
    return false;
  evidence = std::move(*transfer);
  return true;
}

static bool parseFeatures(const json::Object &object,
                          std::vector<double> &features, std::string &error) {
  const json::Array *raw = object.getArray("features");
  if (!raw || raw->size() != kFormalMapperFeatureWidth) {
    error = "feature query must contain exactly 156 feature values";
    return false;
  }
  features.clear();
  features.reserve(kFormalMapperFeatureWidth);
  for (const json::Value &value : *raw) {
    std::optional<double> number = value.getAsNumber();
    if (!number || !std::isfinite(*number)) {
      error = "feature query contains a non-finite feature value";
      return false;
    }
    features.push_back(*number);
  }
  return true;
}

// The standalone C++ feature worker publishes its graph arrays alongside the
// 156 values.  Serialize only those mapper-visible arrays in a fixed field
// order so two tasks with identical bodies share a cache entry without using
// a digest or a task name.
static bool appendCanonicalValue(const json::Value &value,
                                 llvm::raw_ostream &os, std::string &error) {
  if (const json::Array *array = value.getAsArray()) {
    os << '[';
    bool first = true;
    for (const json::Value &element : *array) {
      if (!first)
        os << ',';
      first = false;
      if (!appendCanonicalValue(element, os, error))
        return false;
    }
    os << ']';
    return true;
  }
  if (std::optional<int64_t> integer = value.getAsInteger()) {
    os << *integer;
    return true;
  }
  if (std::optional<double> number = value.getAsNumber()) {
    if (!std::isfinite(*number)) {
      error = "feature output graph contains a non-finite number";
      return false;
    }
    os << *number;
    return true;
  }
  error = "feature output graph contains an unsupported JSON value";
  return false;
}

static bool canonicalBodyFromCppFeatureOutput(const json::Object &entry,
                                              std::string &body,
                                              std::string &error) {
  static constexpr llvm::StringLiteral graphFields[] = {
      "node_types", "node_features", "edges", "semantic_edges"};
  std::string text;
  llvm::raw_string_ostream os(text);
  for (llvm::StringRef field : graphFields) {
    const json::Value *value = entry.get(field);
    if (!value) {
      error = "C++ feature output entry is missing " + field.str();
      return false;
    }
    os << field << '=';
    if (!appendCanonicalValue(*value, os, error))
      return false;
    os << ';';
  }
  os.flush();
  if (text.empty()) {
    error = "C++ feature output graph serialization is empty";
    return false;
  }
  body = std::move(text);
  return true;
}

static json::Array graphEdges(ArrayRef<std::pair<int, int>> edges) {
  json::Array result;
  for (auto [source, target] : edges)
    result.push_back(json::Array{static_cast<int64_t>(source),
                                 static_cast<int64_t>(target)});
  return result;
}

static json::Array graphNodeTypes(ArrayRef<int> nodeTypes) {
  json::Array result;
  for (int type : nodeTypes)
    result.push_back(static_cast<int64_t>(type));
  return result;
}

static json::Array graphNodeFeatures(
    ArrayRef<orbit::mapper_features::NodeFeature> nodeFeatures) {
  json::Array result;
  for (const auto &row : nodeFeatures) {
    json::Array values;
    for (double value : row)
      values.push_back(value);
    result.push_back(std::move(values));
  }
  return result;
}

static json::Object graphJson(const RouteExpandedGraph &graph) {
  json::Object result;
  result["node_types"] = graphNodeTypes(graph.nodeTypes);
  result["node_features"] = graphNodeFeatures(graph.nodeFeatures);
  result["edges"] = graphEdges(graph.edges);
  result["semantic_edges"] = graphEdges(graph.semanticEdges);
  return result;
}

static bool deriveStartupCyclesFromCppFeatureOutput(
    const json::Object &entry, double &startup, std::string &error);

static bool canonicalBodyFromGraph(const RouteExpandedGraph &graph,
                                   std::string &body, std::string &error) {
  json::Object object = graphJson(graph);
  return canonicalBodyFromCppFeatureOutput(object, body, error);
}

static bool startupCyclesFromGraph(const RouteExpandedGraph &graph,
                                   double &startup, std::string &error) {
  json::Object object = graphJson(graph);
  return deriveStartupCyclesFromCppFeatureOutput(object, startup, error);
}

static bool collectCurrentTaskBodies(
    func::FuncOp function, OwningOpRef<ModuleOp> &mapperInput,
    std::vector<CurrentTaskBody> &bodies, std::string &error) {
  bodies.clear();
  mapperInput = ModuleOp::create(function.getLoc());

  struct PendingBody {
    std::string task;
    func::FuncOp mapperFunction;
  };
  std::vector<PendingBody> pending;
  bool collectionFailed = false;
  MLIRContext *context = function.getContext();

  function.walk([&](::mlir::taskflow::TaskflowTaskOp task) {
    if (collectionFailed)
      return WalkResult::interrupt();
    SmallVector<::mlir::neura::KernelOp> kernels;
    task.walk([&](::mlir::neura::KernelOp kernel) {
      kernels.push_back(kernel);
    });
    if (kernels.size() != 1) {
      error = "current task " + task.getTaskName().str() +
              " must contain exactly one mapper-visible neura.kernel";
      collectionFailed = true;
      return WalkResult::interrupt();
    }
    ::mlir::neura::KernelOp sourceKernel = kernels.front();
    if (sourceKernel.getBody().empty()) {
      error = "current task " + task.getTaskName().str() +
              " contains an empty mapper-visible neura.kernel";
      collectionFailed = true;
      return WalkResult::interrupt();
    }

    // The production mapper does not map the task's kernel in place. It
    // clones the kernel *body* into a temporary accelerator function, then
    // replaces each neura.yield with the same data_mov/func.return boundary
    // used by MapJointSchedulingTasksPass. In particular, kernel operands are
    // not function arguments: the kernel body already owns its entry block
    // arguments, and adding them again creates an invalid duplicate-argument
    // wrapper.
    SmallVector<Type> argumentTypes;
    for (BlockArgument argument : sourceKernel.getBody().front().getArguments())
      argumentTypes.push_back(argument.getType());
    SmallVector<Type> resultTypes;
    bool sawYield = false;
    for (Block &block : sourceKernel.getBody()) {
      auto yield = dyn_cast<::mlir::neura::YieldOp>(block.getTerminator());
      if (!yield)
        continue;
      SmallVector<Type> blockResultTypes;
      for (Value result : yield.getResults())
        blockResultTypes.push_back(result.getType());
      if (!sawYield) {
        resultTypes = std::move(blockResultTypes);
        sawYield = true;
      } else if (resultTypes != blockResultTypes) {
        error = "current task " + task.getTaskName().str() +
                " mapper kernel yields inconsistent result types";
        collectionFailed = true;
        return WalkResult::interrupt();
      }
    }
    if (!sawYield) {
      error = "current task " + task.getTaskName().str() +
              " mapper kernel has no terminal neura.yield";
      collectionFailed = true;
      return WalkResult::interrupt();
    }
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(mapperInput->getBody());
    const std::string wrapperName =
        "__orbit_predict_mapper_input_" + std::to_string(pending.size());
    auto wrapper = builder.create<func::FuncOp>(
        sourceKernel.getLoc(), wrapperName,
        builder.getFunctionType(argumentTypes, resultTypes));
    wrapper->setAttr("accelerator", builder.getStringAttr("neura"));
    IRMapping mapping;
    sourceKernel.getBody().cloneInto(&wrapper.getBody(), mapping);
    for (Block &block : wrapper.getBody()) {
      auto yield = dyn_cast<::mlir::neura::YieldOp>(block.getTerminator());
      if (!yield) {
        error = "current task " + task.getTaskName().str() +
                " mapper wrapper clone has a non-yield block terminator";
        collectionFailed = true;
        return WalkResult::interrupt();
      }
      builder.setInsertionPoint(yield);
      SmallVector<Value> returnValues;
      for (Value result : yield.getResults()) {
        auto move = builder.create<::mlir::neura::DataMovOp>(
            sourceKernel.getLoc(), result.getType(), result);
        returnValues.push_back(move.getResult());
      }
      builder.create<func::ReturnOp>(sourceKernel.getLoc(), returnValues);
      yield.erase();
    }

    if (failed(prepareMapperCounterBounds(wrapper))) {
      error = "current task " + task.getTaskName().str() +
              " has invalid counter metadata or an unprepared mapper body";
      collectionFailed = true;
      return WalkResult::interrupt();
    }

    pending.push_back({task.getTaskName().str(), wrapper});
    return WalkResult::advance();
  });
  if (collectionFailed)
    return false;
  if (pending.empty()) {
    error = "selected function contains no current mapper-visible task body";
    return false;
  }
  PassManager manager(context);
  manager.addPass(::mlir::neura::createAssignAcceleratorPass());
  manager.addPass(::mlir::neura::createInsertDataMovPass());
  if (failed(manager.run(*mapperInput))) {
    error = "C++ mapper frontend failed while preparing temporary task "
            "bodies with the compiler lowering/InsertDataMov pipeline";
    return false;
  }

  // Keep the verifier enabled in the pass manager above.  Verify the
  // prepared temporary module once more before printing any feature graph so
  // a malformed wrapper cannot be accepted merely because the feature parser
  // can consume its text.
  if (failed(verify(*mapperInput))) {
    error = "C++ mapper frontend produced an invalid temporary mapper module "
            "after preparation";
    return false;
  }

  for (PendingBody &item : pending) {
    std::string residualDialect;
    item.mapperFunction.walk([&](Operation *operation) {
      llvm::StringRef dialect = operation->getDialect()->getNamespace();
      if (residualDialect.empty() &&
          (dialect == "arith" || dialect == "memref" || dialect == "scf"))
        residualDialect = operation->getName().getStringRef().str();
    });
    if (!residualDialect.empty()) {
      error = "current task " + item.task +
              " raw kernel could not be lowered to mapper-visible Neura IR; "
              "unsupported residual operation " + residualDialect;
      return false;
    }

    std::string mapperText;
    llvm::raw_string_ostream mapperStream(mapperText);
    // Keep one SSA namespace across the kernel result and its nested region;
    // useLocalScope would restart numbering in the region and make the text
    // parser see duplicate values.
    item.mapperFunction.print(mapperStream);
    mapperStream.flush();

    CurrentTaskBody current;
    current.task = item.task;
    current.mapperFunction = item.mapperFunction;
    if (!orbit::mapper_features::parseRouteExpandedDFG(
            mapperText, current.graph, error, /*requireExpanded=*/true)) {
      error = "current task " + current.task +
              " mapper-visible body is not a valid route-expanded DFG after "
              "the C++ frontend: " + error;
      return false;
    }
    if (!canonicalBodyFromGraph(current.graph, current.bodyStructuralText,
                                error) ||
        !startupCyclesFromGraph(current.graph, current.startupCycles, error)) {
      error = "current task " + current.task +
              " mapper-visible body has invalid structural facts: " + error;
      return false;
    }
    bodies.push_back(std::move(current));
  }
  return true;
}

static bool computeCurrentAnalyticalFacts(
    Region &region, int64_t rows, int64_t columns,
    double &recMII, double &resMII, double &lowerBound, std::string &error) {
  if (rows <= 0 || columns <= 0 ||
      rows > std::numeric_limits<int64_t>::max() / columns ||
      rows > std::numeric_limits<int>::max() ||
      columns > std::numeric_limits<int>::max()) {
    error = "current mapper shape has invalid dimensions";
    return false;
  }
  const int64_t tileCount = rows * columns;
  auto architecture = ::mlir::neura::getArchitecture().cloneWithNewDimensions(
      static_cast<int>(rows), static_cast<int>(columns));
  if (!architecture || architecture->getNumTiles() <= 0) {
    error = "cannot construct the current mapper architecture shape";
    return false;
  }
  const auto cycles = ::mlir::neura::collectRecurrenceCycles(region);
  int rec = 1;
  for (const auto &cycle : cycles)
    rec = std::max(rec, cycle.length);
  const int res = ::mlir::neura::calculateResMii(region, *architecture);
  if (rec <= 0 || res <= 0 || tileCount != architecture->getNumTiles()) {
    error = "current mapper analytical bounds are invalid";
    return false;
  }
  recMII = static_cast<double>(rec);
  resMII = static_cast<double>(res);
  lowerBound = std::max(recMII, resMII);
  return std::isfinite(recMII) && std::isfinite(resMII) &&
         std::isfinite(lowerBound);
}

static bool deriveStartupCyclesFromCppFeatureOutput(const json::Object &entry,
                                                    double &startup,
                                                    std::string &error) {
  const json::Array *rawTypes = entry.getArray("node_types");
  const json::Array *rawEdges = entry.getArray("edges");
  if (!rawTypes || !rawEdges || rawTypes->empty()) {
    error = "C++ feature output cannot derive startup without node_types and "
            "edges";
    return false;
  }
  std::vector<int64_t> types;
  types.reserve(rawTypes->size());
  for (const json::Value &value : *rawTypes) {
    std::optional<int64_t> type = value.getAsInteger();
    if (!type || *type < 0) {
      error = "C++ feature output node_types are invalid";
      return false;
    }
    types.push_back(*type);
  }
  std::vector<int64_t> depth(types.size(), 1);
  for (const json::Value &rawEdge : *rawEdges) {
    const json::Array *edge = rawEdge.getAsArray();
    if (!edge || edge->size() != 2 || !(*edge)[0].getAsInteger() ||
        !(*edge)[1].getAsInteger()) {
      error = "C++ feature output edges are invalid";
      return false;
    }
    int64_t source = *(*edge)[0].getAsInteger();
    int64_t target = *(*edge)[1].getAsInteger();
    if (source < 0 || target < 0 || source >= static_cast<int64_t>(types.size()) ||
        target >= static_cast<int64_t>(types.size())) {
      error = "C++ feature output edge endpoint is out of range";
      return false;
    }
    // Route-expanded DFGs are emitted in source order.  Feedback edges are
    // deliberately excluded from the startup critical path, matching the
    // pinned frontend's unit-latency semantic depth calculation.
    if (source >= target)
      continue;
    int64_t increment = (types[target] == 40 || types[target] == 41 ||
                         types[target] == 42 || types[target] == 43)
                            ? 0
                            : 1;
    depth[target] = std::max(depth[target], depth[source] + increment);
  }
  startup = static_cast<double>(*std::max_element(depth.begin(), depth.end()));
  if (!std::isfinite(startup) || startup <= 0.0) {
    error = "C++ feature output derived a non-positive startup critical path";
    return false;
  }
  return true;
}

static bool readFeatureQueries(llvm::StringRef path, llvm::StringRef function,
                               llvm::StringRef architectureContract,
                               const FormalMax4MLPEnsemble &model,
                               std::vector<FeatureQuery> &queries,
                               std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read feature query file " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  llvm::Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "cannot parse feature query file " + path.str() + ": " +
            llvm::toString(parsed.takeError());
    return false;
  }
  const json::Object *root = parsed->getAsObject();
  if (!root) {
    error = "feature query root is not an object";
    return false;
  }
  std::optional<llvm::StringRef> schema = root->getString("schema");
  bool cppFeatureOutput =
      schema && *schema == "cgra-ii-cpp-feature-output-v1";
  if (!schema || (!cppFeatureOutput && *schema != kFeatureQuerySchema)) {
    error = "feature query schema is not a supported C++ contract";
    return false;
  }
  if (!cppFeatureOutput &&
      (!requireString(*root, "feature_contract_id",
                      model.getFeatureContractId(), error) ||
       !requireString(*root, "feature_extractor", model.getFeatureExtractor(),
                      error) ||
       !requireString(*root, "shape_protocol_id", model.getShapeProtocolId(),
                      error) ||
       !requireString(*root, "architecture_contract", architectureContract,
                      error) ||
       !requireString(*root, "function", function, error)))
    return false;
  if (cppFeatureOutput) {
    std::optional<int64_t> width = root->getInteger("feature_width");
    const json::Array *names = root->getArray("feature_names");
    if (!width || *width != kFormalMapperFeatureWidth || !names ||
        names->size() != kFormalMapperFeatureWidth) {
      error = "C++ feature output does not enumerate 156 mapper features";
      return false;
    }
    std::set<std::string> uniqueNames;
    std::vector<std::string> actualNames;
    actualNames.reserve(names->size());
    for (const json::Value &rawName : *names) {
      std::optional<llvm::StringRef> name = rawName.getAsString();
      if (!name || name->empty() || !uniqueNames.insert(name->str()).second) {
        error = "C++ feature output contains a missing or duplicate feature "
                "name";
        return false;
      }
      actualNames.push_back(name->str());
    }
    if (!model.featureNamesMatch(actualNames)) {
      error = "C++ feature output feature_names differ from the model contract";
      return false;
    }
    if (std::optional<llvm::StringRef> actualFunction =
            root->getString("function")) {
      if (*actualFunction != function) {
        error = "C++ feature output function does not match the selected IR";
        return false;
      }
    }
  }
  const json::Array *rawQueries =
      root->getArray(cppFeatureOutput ? "queries" : "entries");
  if (!rawQueries || rawQueries->empty()) {
    error = "feature query file contains no entries";
    return false;
  }
  queries.clear();
  const json::Object *startupByTask = root->getObject("startup_cycles_by_task");
  std::set<std::tuple<std::string, int64_t, int64_t>> seenTaskShapes;
  for (const json::Value &rawQuery : *rawQueries) {
    const json::Object *object = rawQuery.getAsObject();
    if (!object) {
      error = "feature query entry is not an object";
      return false;
    }
    FeatureQuery query;
    if (!getString(*object, "task", query.task, error) ||
        (cppFeatureOutput
             ? (!getInteger(*object, "rows", query.rows, error) ||
                !getInteger(*object, "columns", query.cols, error))
             : (!getString(*object, "body_structural_text",
                           query.bodyStructuralText, error) ||
                !getInteger(*object, "mapper_tile_rows", query.rows, error) ||
                !getInteger(*object, "mapper_tile_cols", query.cols, error))) ||
        !getNumber(*object, "rec_mii", query.recMII, error) ||
        !getNumber(*object, "res_mii", query.resMII, error) ||
        !(cppFeatureOutput
              ? getNumber(*object, "lower_bound", query.lowerBound, error)
              : getNumber(*object, "analytical_lower_bound", query.lowerBound,
                          error)) ||
        !parseFeatures(*object, query.features, error))
      return false;
    if (cppFeatureOutput) {
      if (const json::Value *startup = object->get("startup_cycles")) {
        std::optional<double> value = startup->getAsNumber();
        if (!value || !std::isfinite(*value)) {
          error = "C++ feature output startup_cycles is invalid";
          return false;
        }
        query.startupCycles = *value;
      } else if (startupByTask) {
        if (!getNumber(*startupByTask, query.task, query.startupCycles, error))
          return false;
      } else {
        if (!deriveStartupCyclesFromCppFeatureOutput(*object,
                                                      query.startupCycles,
                                                      error))
          return false;
      }
      if (!canonicalBodyFromCppFeatureOutput(*object,
                                              query.bodyStructuralText, error))
        return false;
    } else if (!getNumber(*object, "startup_cycles", query.startupCycles,
                          error)) {
      return false;
    }
    if (!isFormalMax4MapperShape(query.rows, query.cols)) {
      error = "feature query contains an unsupported formal mapper shape";
      return false;
    }
    if (query.recMII <= 0.0 || query.resMII <= 0.0 ||
        query.lowerBound <= 0.0 || query.startupCycles <= 0.0 ||
        query.lowerBound != std::max(query.recMII, query.resMII)) {
      error = "feature query has invalid RecMII/ResMII/lower-bound/startup "
              "facts";
      return false;
    }
    auto taskShape = std::make_tuple(query.task, query.rows, query.cols);
    if (!seenTaskShapes.insert(taskShape).second) {
      error = "feature query contains duplicate task/mapper-shape entries";
      return false;
    }
    queries.push_back(std::move(query));
  }
  return true;
}

static bool generateCurrentFeatureQueries(
    std::vector<CurrentTaskBody> &bodies,
    const FormalMax4MLPEnsemble &model,
    std::vector<FeatureQuery> &queries, std::string &error) {
  const auto &names = orbit::mapper_features::mapperFeatureNames();
  if (!model.featureNamesMatch(ArrayRef<std::string>(names.data(),
                                                     names.size()))) {
    error = "current C++ feature frontend feature_names differ from the model "
            "contract";
    return false;
  }
  queries.clear();
  queries.reserve(bodies.size() * kFormalMax4Shapes.size());
  for (CurrentTaskBody &body : bodies) {
    for (auto [rows, columns] : kFormalMax4Shapes) {
      double recMII = 0.0;
      double resMII = 0.0;
      double lowerBound = 0.0;
      if (!computeCurrentAnalyticalFacts(body.mapperFunction.getBody(), rows,
                                         columns, recMII, resMII, lowerBound,
                                         error)) {
        error = "current task " + body.task + " shape rect-" +
                std::to_string(rows) + "x" + std::to_string(columns) +
                " has invalid analytical facts: " + error;
        return false;
      }
      FeatureVector featureVector;
      if (!orbit::mapper_features::computeMapperFeatures(
              body.graph, static_cast<int>(rows), static_cast<int>(columns),
              recMII, resMII, lowerBound, featureVector, error)) {
        error = "current task " + body.task + " shape rect-" +
                std::to_string(rows) + "x" + std::to_string(columns) +
                " feature extraction failed: " + error;
        return false;
      }
      FeatureQuery query;
      query.task = body.task;
      query.bodyStructuralText = body.bodyStructuralText;
      query.rows = rows;
      query.cols = columns;
      query.recMII = recMII;
      query.resMII = resMII;
      query.lowerBound = lowerBound;
      query.startupCycles = body.startupCycles;
      query.features.assign(featureVector.values.begin(),
                            featureVector.values.end());
      queries.push_back(std::move(query));
    }
  }
  return true;
}

static bool sameFeatureNumber(double lhs, double rhs) {
  const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
  return std::abs(lhs - rhs) <= 1.0e-12 * scale;
}

static bool verifyFeatureQueriesBoundToCurrentBodies(
    ArrayRef<FeatureQuery> generated, ArrayRef<FeatureQuery> supplied,
    std::string &error) {
  using QueryKey = std::tuple<std::string, int64_t, int64_t>;
  std::map<QueryKey, const FeatureQuery *> expected;
  for (const FeatureQuery &query : generated)
    expected.emplace(QueryKey{query.task, query.rows, query.cols}, &query);
  if (expected.size() != generated.size() || supplied.size() != generated.size()) {
    error = "supplied feature queries do not cover exactly the current task "
            "and formal mapper-shape space";
    return false;
  }
  for (const FeatureQuery &actual : supplied) {
    auto found = expected.find(QueryKey{actual.task, actual.rows, actual.cols});
    if (found == expected.end()) {
      error = "supplied feature query names a task/shape absent from the "
              "current module";
      return false;
    }
    const FeatureQuery &current = *found->second;
    if (actual.bodyStructuralText != current.bodyStructuralText) {
      error = "supplied feature query body does not match the current "
              "mapper-visible task body for task=" + actual.task +
              " shape=rect-" + std::to_string(actual.rows) + "x" +
              std::to_string(actual.cols);
      return false;
    }
    if (!sameFeatureNumber(actual.recMII, current.recMII) ||
        !sameFeatureNumber(actual.resMII, current.resMII) ||
        !sameFeatureNumber(actual.lowerBound, current.lowerBound) ||
        !sameFeatureNumber(actual.startupCycles, current.startupCycles) ||
        actual.features.size() != current.features.size()) {
      error = "supplied feature query facts do not match the current "
              "mapper-visible task body for task=" + actual.task +
              " shape=rect-" + std::to_string(actual.rows) + "x" +
              std::to_string(actual.cols);
      return false;
    }
    for (size_t index = 0; index < actual.features.size(); ++index) {
      if (!sameFeatureNumber(actual.features[index],
                             current.features[index])) {
        error = "supplied feature query vector does not match the current "
                "mapper-visible task body for task=" + actual.task +
                " shape=rect-" + std::to_string(actual.rows) + "x" +
                std::to_string(actual.cols);
        return false;
      }
    }
  }
  return true;
}

static bool writeCurrentFeatureOutput(
    llvm::StringRef outputPath, llvm::StringRef function,
    llvm::StringRef architectureContract, const FormalMax4MLPEnsemble &model,
    ArrayRef<CurrentTaskBody> bodies, ArrayRef<FeatureQuery> queries,
    std::string &error) {
  std::map<std::string, const CurrentTaskBody *> bodyByTask;
  for (const CurrentTaskBody &body : bodies)
    bodyByTask.emplace(body.task, &body);
  return writeAtomically(
      outputPath,
      [&](llvm::raw_ostream &os) {
        json::Object root;
        root["schema"] = "cgra-ii-cpp-feature-output-v1";
        root["function"] = function.str();
        root["architecture_contract"] = architectureContract.str();
        root["feature_contract_id"] = model.getFeatureContractId().str();
        root["feature_extractor"] = model.getFeatureExtractor().str();
        root["shape_protocol_id"] = model.getShapeProtocolId().str();
        root["feature_width"] = static_cast<int64_t>(kFormalMapperFeatureWidth);
        json::Array names;
        for (const std::string &name : orbit::mapper_features::mapperFeatureNames())
          names.push_back(name);
        root["feature_names"] = std::move(names);
        json::Array entries;
        for (const FeatureQuery &query : queries) {
          auto found = bodyByTask.find(query.task);
          if (found == bodyByTask.end())
            return false;
          json::Object entry = graphJson(found->second->graph);
          entry["task"] = query.task;
          entry["rows"] = query.rows;
          entry["columns"] = query.cols;
          entry["rec_mii"] = query.recMII;
          entry["res_mii"] = query.resMII;
          entry["lower_bound"] = query.lowerBound;
          entry["startup_cycles"] = query.startupCycles;
          json::Array featureValues;
          for (double value : query.features)
            featureValues.push_back(value);
          entry["features"] = std::move(featureValues);
          entries.push_back(std::move(entry));
        }
        root["queries"] = std::move(entries);
        os << json::Value(std::move(root)) << "\n";
        return true;
      },
      error);
}

static bool writeCatalogue(llvm::raw_ostream &os, llvm::StringRef function,
                           llvm::StringRef modelNamespace,
                           llvm::StringRef sourceRepository,
                           llvm::StringRef sourceCommit,
                           llvm::StringRef architecturePath,
                           llvm::StringRef architectureContract,
                           llvm::StringRef graphVariantId,
                           const FormalMax4MLPEnsemble &model,
                           llvm::ArrayRef<TaskMetadata> taskMetadata,
                           llvm::StringRef candidateCount,
                           const std::vector<FeatureQuery> &queries,
                           const std::vector<MLPEnsemblePrediction> &predictions,
                           const PersistentMLCostCache &cache,
                           std::optional<json::Object> architectureTransfer) {
  json::Object root;
  root["schema"] = "amoeba-task-shape-cost";
  root["function"] = function.str();
  root["namespace"] = modelNamespace.str();

  json::Object metadata;
  metadata["provenance_schema"] = kCostProvenanceSchema.str();
  metadata["predictor_source"] = "ml-ii-startup-predictor";
  metadata["source_repository"] = sourceRepository.str();
  metadata["source_commit"] = sourceCommit.str();
  metadata["architecture_path"] = architecturePath.str();
  metadata["architecture_schema"] = kArchitectureSchema.str();
  metadata["architecture_contract"] = architectureContract.str();
  metadata["source_graph_id"] = graphVariantId.str();
  metadata["candidate_id_scheme"] = "candidate-<sequential-index>";
  uint64_t count = 0;
  if (!candidateCount.getAsInteger(10, count) &&
      count <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    metadata["candidate_count"] = static_cast<int64_t>(count);
  else
    metadata["candidate_count"] = candidateCount.str();
  json::Array sourceTaskIds;
  for (const TaskMetadata &task : taskMetadata)
    sourceTaskIds.push_back(task.name);
  metadata["source_task_ids"] = std::move(sourceTaskIds);
  metadata["model"] = kModelName.str();
  metadata["model_schema"] = model.getModelSchema().str();
  metadata["model_status"] = kModelStatus.str();
  metadata["quality_status"] = kModelStatus.str();
  metadata["production_ready"] = false;
  metadata["feature_contract_id"] = model.getFeatureContractId().str();
  metadata["feature_extractor"] = model.getFeatureExtractor().str();
  metadata["feature_frontend"] = "c++-mlir-current-body-v2";
  metadata["shape_protocol_id"] = model.getShapeProtocolId().str();
  metadata["communication_latency"] = "scored_by_scheduler";
  metadata["ranking_policy"] = json::Object{
      {"objective", "predicted_scheduler_makespan"},
      {"mapper_success_probability", "not_predicted"},
      {"uses_mapper_success_probability", false}};
  metadata["ml_cache"] = json::Object{
      {"identity", "canonical-body-text-and-mapper-shape"},
      {"entries", static_cast<int64_t>(cache.size())},
      {"hits", static_cast<int64_t>(cache.getHitCount())},
      {"misses", static_cast<int64_t>(cache.getMissCount())},
      {"trip_count_in_identity", false}};
  if (architectureTransfer)
    metadata["architecture_transfer"] = std::move(*architectureTransfer);
  root["predictor_metadata"] = std::move(metadata);

  json::Array entries;
  for (size_t index = 0; index < queries.size(); ++index) {
    const FeatureQuery &query = queries[index];
    const MLPEnsemblePrediction &prediction = predictions[index];
    entries.push_back(json::Object{
        {"task", query.task},
        {"mapper_tile_rows", query.rows},
        {"mapper_tile_cols", query.cols},
        {"support_status", "supported"},
        {"predicted_ii", prediction.predictedII},
        {"startup_cycles", query.startupCycles},
        {"predicted_ii_std", prediction.predictedIIStd},
        {"analytical_lower_bound", query.lowerBound},
        {"ii_mean_source", kModelName.str()},
        {"model_status", kModelStatus.str()},
        {"production_ready", false}});
  }
  root["entries"] = std::move(entries);
  os << json::Value(std::move(root));
  return true;
}

struct PredictAnalyticalTaskCostCatalogPass
    : public PassWrapper<PredictAnalyticalTaskCostCatalogPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      PredictAnalyticalTaskCostCatalogPass)

  PredictAnalyticalTaskCostCatalogPass() = default;
  PredictAnalyticalTaskCostCatalogPass(
      const PredictAnalyticalTaskCostCatalogPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "predict-analytical-task-cost-catalog";
  }
  StringRef getDescription() const override {
    return "Predict an exploratory C++ formal max-four task-shape catalogue";
  }

  Option<std::string> functionName{
      *this, "function", llvm::cl::desc("Taskflow function name"),
      llvm::cl::init("")};
  Option<std::string> featureFile{
      *this, "feature-file", llvm::cl::desc("C++ mapper feature query JSON"),
      llvm::cl::init("")};
  Option<std::string> featureOutputFile{
      *this, "feature-output",
      llvm::cl::desc("Optional generated current-module feature JSON"),
      llvm::cl::init("")};
  Option<std::string> spaceFile{
      *this, "space-file", llvm::cl::desc("Exhaustive candidate-space header"),
      llvm::cl::init("")};
  Option<std::string> ensembleFile{
      *this, "ensemble-file", llvm::cl::desc("Formal JSON ensemble"),
      llvm::cl::init("")};
  Option<std::string> checkpointDirectory{
      *this, "checkpoint-dir", llvm::cl::desc("Formal JSON checkpoint directory"),
      llvm::cl::init("")};
  Option<std::string> architectureContract{
      *this, "architecture-contract",
      llvm::cl::desc("Neura architecture and shape contract"),
      llvm::cl::init("")};
  Option<std::string> architecturePath{
      *this, "architecture-path", llvm::cl::desc("Target architecture path"),
      llvm::cl::init("")};
  Option<std::string> sourceRepository{
      *this, "source-git-repository", llvm::cl::desc("Source repository"),
      llvm::cl::init("")};
  Option<std::string> sourceCommit{
      *this, "source-git-commit", llvm::cl::desc("Source Git commit"),
      llvm::cl::init("")};
  Option<std::string> graphVariantId{
      *this, "graph-variant-id", llvm::cl::desc("Graph variant identity"),
      llvm::cl::init("")};
  Option<std::string> architectureTransferCatalog{
      *this, "architecture-transfer-catalog",
      llvm::cl::desc("Optional validated rebound catalogue with architecture transfer evidence"),
      llvm::cl::init("")};
  Option<std::string> modelNamespace{
      *this, "model-namespace", llvm::cl::desc("Output catalogue namespace"),
      llvm::cl::init("formal-max4-nohash-v2-exploratory")};
  Option<std::string> cacheFile{
      *this, "cache", llvm::cl::desc("Persistent structural ML cache"),
      llvm::cl::init("")};
  Option<std::string> outputFile{
      *this, "output", llvm::cl::desc("Output task-shape catalogue"),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](llvm::StringRef message) {
      module.emitError() << message;
      signalPassFailure();
    };
    if (spaceFile.empty() || ensembleFile.empty() ||
        checkpointDirectory.empty() || architectureContract.empty() ||
        architecturePath.empty() || sourceRepository.empty() ||
        sourceCommit.empty() || graphVariantId.empty() || outputFile.empty()) {
      fail("predict-analytical-task-cost-catalog requires space-file, "
           "ensemble-file, checkpoint-dir, architecture-contract, "
           "architecture-path, source-git-repository, source-git-commit, "
           "graph-variant-id, and output");
      return;
    }
    if (!llvm::StringRef(architectureContract).starts_with(
            kCacheArchitecturePrefix)) {
      fail("architecture-contract must use the neura-architecture-v1: prefix");
      return;
    }
    if (!llvm::sys::fs::exists(architecturePath)) {
      fail("architecture-path does not name an existing target architecture");
      return;
    }

    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName, error);
    if (failed(selected)) {
      fail(error);
      return;
    }
    FailureOr<llvm::SmallVector<TaskMetadata>> taskMetadata =
        collectAnalyticalTaskMetadata(*selected, error);
    if (failed(taskMetadata)) {
      fail(error);
      return;
    }
    std::map<std::string, size_t> taskOrder;
    for (size_t index = 0; index < taskMetadata->size(); ++index)
      taskOrder.emplace((*taskMetadata)[index].name, index);

    FormalMax4MLPEnsemble model;
    if (!model.load(ensembleFile, checkpointDirectory, architectureContract,
                   error)) {
      fail(error);
      return;
    }
    std::string architectureText;
    if (!readTextFile(architecturePath, architectureText, error)) {
      fail(error);
      return;
    }
    MLCostCacheResources cacheResources =
        model.makeCacheResources(architectureText);
    std::optional<json::Object> architectureTransfer;
    if (!readArchitectureTransferEvidence(architectureTransferCatalog,
                                           architecturePath,
                                           architectureTransfer, error)) {
      fail(error);
      return;
    }

    // Always derive the mapper-visible body and 156 features from this
    // module. An optional feature-file is an oracle/debug input only: it must
    // exactly agree with the current module before it can be accepted. This
    // prevents stale or task-name-substituted records from driving costs for
    // a rewritten graph.
    OwningOpRef<ModuleOp> mapperInput;
    std::vector<CurrentTaskBody> currentBodies;
    if (!collectCurrentTaskBodies(*selected, mapperInput, currentBodies,
                                  error)) {
      fail(error);
      return;
    }
    std::vector<FeatureQuery> generatedQueries;
    if (!generateCurrentFeatureQueries(currentBodies, model, generatedQueries,
                                       error)) {
      fail(error);
      return;
    }
    if (!featureFile.empty()) {
      std::vector<FeatureQuery> suppliedQueries;
      if (!readFeatureQueries(featureFile, selected->getName(),
                              architectureContract, model, suppliedQueries,
                              error) ||
          !verifyFeatureQueriesBoundToCurrentBodies(generatedQueries,
                                                     suppliedQueries, error)) {
        fail(error);
        return;
      }
    }
    if (!featureOutputFile.empty() &&
        !writeCurrentFeatureOutput(featureOutputFile, selected->getName(),
                                   architectureContract, model, currentBodies,
                                   generatedQueries, error)) {
      fail(error);
      return;
    }
    std::vector<FeatureQuery> queries = std::move(generatedQueries);
    std::string candidateCount;
    if (!readCandidateCount(spaceFile, selected->getName(), graphVariantId,
                            candidateCount, error)) {
      fail(error);
      return;
    }
    for (const FeatureQuery &query : queries) {
      if (!taskOrder.count(query.task)) {
        fail("feature query names a task outside the selected function");
        return;
      }
    }
    for (const auto &task : *taskMetadata) {
      if (!llvm::any_of(queries, [&](const FeatureQuery &query) {
            return query.task == task.name;
          })) {
        fail("feature query file does not cover every Taskflow task");
        return;
      }
    }
    llvm::sort(queries, [&](const FeatureQuery &left, const FeatureQuery &right) {
      return std::tie(taskOrder[left.task], left.rows, left.cols) <
             std::tie(taskOrder[right.task], right.rows, right.cols);
    });

    // Keep this lock alive from cache load through all lookups, predictions,
    // and the atomic cache write.  A shared cache is used by independent
    // graph workers, so locking only the final rename would lose rows when
    // two workers start from the same snapshot.
    MLCostCacheLock cacheLock;
    if (!cacheLock.acquire(cacheFile, error)) {
      fail(error);
      return;
    }
    PersistentMLCostCache cache;
    if (!cache.load(cacheFile, model.getModelSchema(),
                    model.getFeatureContractId(), architectureContract,
                    cacheResources, error)) {
      fail(error);
      return;
    }
    std::vector<MLPEnsemblePrediction> predictions;
    predictions.reserve(queries.size());
    for (const FeatureQuery &query : queries) {
      MLCostCacheKey key{query.bodyStructuralText, query.rows, query.cols};
      MLCostCacheFacts facts{query.recMII, query.resMII, query.lowerBound};
      MLPEnsemblePrediction prediction;
      if (!cache.lookup(key, facts, prediction)) {
        if (!model.predict(query.features, query.lowerBound, prediction, error)) {
          fail(error);
          return;
        }
        cache.insert(key, facts, prediction);
      }
      if (!std::isfinite(prediction.predictedII) ||
          prediction.predictedII < query.lowerBound ||
          prediction.predictedII > 20.0 + 1.0e-6 ||
          !std::isfinite(prediction.predictedIIStd) ||
          prediction.predictedIIStd < 0.0) {
        fail("model prediction violates finite/lower-bound/ceiling contract");
        return;
      }
      predictions.push_back(prediction);
    }
    if (!cache.write(cacheFile, model.getModelSchema(),
                     model.getFeatureContractId(), architectureContract,
                     cacheResources, error)) {
      fail(error);
      return;
    }
    if (!writeAtomically(
            outputFile,
            [&](llvm::raw_ostream &os) {
              return writeCatalogue(
                  os, selected->getName(), modelNamespace, sourceRepository,
                  sourceCommit, architecturePath, architectureContract,
                  graphVariantId, model, *taskMetadata, candidateCount, queries,
                  predictions, cache, std::move(architectureTransfer));
            },
            error)) {
      fail(error);
      return;
    }
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createPredictAnalyticalTaskCostCatalogPass() {
  return std::make_unique<PredictAnalyticalTaskCostCatalogPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
