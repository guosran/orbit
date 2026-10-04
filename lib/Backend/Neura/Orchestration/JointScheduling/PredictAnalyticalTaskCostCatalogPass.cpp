//===- PredictAnalyticalTaskCostCatalogPass.cpp ----------------*- C++ -*-===//
//
// Predicts an exploratory task-shape II catalogue from a C++ feature query
// file and either the pinned no-SHA formal max-four or direct per-CGRA 2x2
// JSON ensemble.  The current
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
//                analytical_lower_bound, startup_cycles, features[156|148] }]
// body_structural_text is the canonical mapper-visible body serialization
// supplied by the C++ frontend.  It is not a digest and must not be empty.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalMLPInference.h"
#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalTaskCostCatalog.h"
#include "MapperCounterBounds.h"
#include "MapperCostAnalysis.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/MapperFeatureExtractor.h"
#include "NeighborhoodReplaySelection.h"
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
constexpr std::array<std::pair<int64_t, int64_t>, 8> kPerCgra2x2Shapes = {{
    {2, 2},
    {2, 4},
    {4, 2},
    {2, 6},
    {6, 2},
    {2, 8},
    {8, 2},
    {4, 4},
}};
constexpr llvm::StringLiteral kDirectModelName =
    "orbit-cgra-ii-per-cgra-2x2-direct-4member-v1";
constexpr llvm::StringLiteral kDirectCandidateStatus =
    "candidate_pending_amoeba_benchmark_overlap_audit";
constexpr llvm::StringLiteral kDirectSourceRepository =
    "https://github.com/guosran/cgra-ii-predictor";
constexpr llvm::StringLiteral kDirectSourceBranch = "orbit-2x2-predictor";
constexpr llvm::StringLiteral kDirectSourceCommit =
    "3ade31806cb4c92e31888109f7c42b8a77e4cbce";
constexpr std::array<int64_t, 4> kDirectMemberSeeds = {{17, 41, 113, 239}};

struct FeatureQuery {
  std::string task;
  std::string bodyStructuralText;
  int64_t rows = 0;
  int64_t cols = 0;
  double recMII = 0.0;
  double resMII = 0.0;
  double lowerBound = 0.0;
  double startupCycles = 0.0;
  bool modelDomainUnsupported = false;
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
    error = "cannot read source text " + path.str() + ": " +
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

static bool readDirectSourceModel(llvm::StringRef path,
                                  json::Object &sourceModel,
                                  std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read direct-model provenance " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  llvm::Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "cannot parse direct-model provenance: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  const json::Object *root = parsed->getAsObject();
  const json::Object *source = root ? root->getObject("source_model") : nullptr;
  const json::Object *candidate =
      source ? source->getObject("candidate_metadata") : nullptr;
  auto matches = [&](llvm::StringRef key, llvm::StringRef expected) {
    std::optional<llvm::StringRef> actual =
        source ? source->getString(key) : std::nullopt;
    return actual && *actual == expected;
  };
  if (!source || !candidate ||
      !matches("repository", kDirectSourceRepository) ||
      !matches("branch", kDirectSourceBranch) ||
      !matches("commit", kDirectSourceCommit)) {
    error = "direct model bundle lacks validated candidate provenance";
    return false;
  }
  sourceModel = json::Object(*source);
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

static bool parseFeatures(const json::Object &object, unsigned featureWidth,
                          std::vector<double> &features, std::string &error) {
  const json::Array *raw = object.getArray("features");
  if (!raw || raw->size() != featureWidth) {
    error = "feature query must contain exactly " +
            std::to_string(featureWidth) + " feature values";
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

static bool readFeatureQueries(
    llvm::StringRef path, llvm::StringRef function,
    llvm::StringRef architectureContract, llvm::StringRef featureContractId,
    llvm::StringRef featureExtractor, llvm::StringRef shapeProtocolId,
    ArrayRef<std::string> expectedNames, bool directModel,
    std::vector<FeatureQuery> &queries, std::string &error) {
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
      (!requireString(*root, "feature_contract_id", featureContractId, error) ||
       !requireString(*root, "feature_extractor", featureExtractor, error) ||
       !requireString(*root, "shape_protocol_id", shapeProtocolId, error) ||
       !requireString(*root, "architecture_contract", architectureContract,
                      error) ||
       !requireString(*root, "function", function, error)))
    return false;
  if (cppFeatureOutput) {
    std::optional<int64_t> width = root->getInteger("feature_width");
    const json::Array *names = root->getArray("feature_names");
    if (!width || *width != static_cast<int64_t>(expectedNames.size()) ||
        !names || names->size() != expectedNames.size()) {
      error = "C++ feature output width does not match the selected model";
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
    if (!std::equal(actualNames.begin(), actualNames.end(),
                    expectedNames.begin(), expectedNames.end())) {
      error = "C++ feature output feature_names differ from the model contract";
      return false;
    }
    if (directModel &&
        (!requireString(*root, "feature_contract_id", featureContractId,
                        error) ||
         !requireString(*root, "feature_extractor", featureExtractor, error) ||
         !requireString(*root, "shape_protocol_id", shapeProtocolId, error) ||
         !requireString(*root, "architecture_contract", architectureContract,
                        error)))
      return false;
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
        !parseFeatures(*object, expectedNames.size(), query.features, error))
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
    if (!(directModel ? isPerCgra2x2MapperShape(query.rows, query.cols)
                      : isFormalMax4MapperShape(query.rows, query.cols))) {
      error = directModel
                  ? "feature query contains a shape outside the direct "
                    "per-CGRA 2x2 protocol"
                  : "feature query contains an unsupported formal mapper shape";
      return false;
    }
    if (query.recMII <= 0.0 || query.resMII <= 0.0 || query.lowerBound <= 0.0 ||
        query.startupCycles <= 0.0 ||
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
    std::vector<CurrentTaskBody> &bodies, ArrayRef<std::string> names,
    bool directModel, std::vector<FeatureQuery> &queries, std::string &error,
    bool allowUnsupportedAboveModelCeiling) {
  queries.clear();
  ArrayRef<std::pair<int64_t, int64_t>> shapes =
      directModel ? ArrayRef<std::pair<int64_t, int64_t>>(kPerCgra2x2Shapes)
                  : ArrayRef<std::pair<int64_t, int64_t>>(kFormalMax4Shapes);
  queries.reserve(bodies.size() * shapes.size());
  for (CurrentTaskBody &body : bodies) {
    for (auto [rows, columns] : shapes) {
      double recMII = 0.0;
      double resMII = 0.0;
      double lowerBound = 0.0;
      if (!computeMapperAnalyticalFacts(body.mapperFunction.getBody(), rows,
                                         columns, recMII, resMII, lowerBound,
                                         error)) {
        error = "current task " + body.task + " shape rect-" +
                std::to_string(rows) + "x" + std::to_string(columns) +
                " has invalid analytical facts: " + error;
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
      if (allowUnsupportedAboveModelCeiling &&
          lowerBound > kFormalMax4ModelCeilingII) {
        query.modelDomainUnsupported = true;
        queries.push_back(std::move(query));
        continue;
      }
      if (directModel && lowerBound > kFormalMax4ModelCeilingII) {
        error = "current task " + body.task + " shape rect-" +
                std::to_string(rows) + "x" + std::to_string(columns) +
                " has an analytical lower bound above the direct-model II "
                "ceiling of 20; enable explicit unsupported-domain records";
        return false;
      }
      if (directModel) {
        orbit::mapper_features::PerCgra2x2FeatureVector featureVector;
        if (!orbit::mapper_features::computePerCgra2x2MapperFeatures(
                body.graph, static_cast<int>(rows), static_cast<int>(columns),
                recMII, resMII, lowerBound, featureVector, error)) {
          error = "current task " + body.task + " shape rect-" +
                  std::to_string(rows) + "x" + std::to_string(columns) +
                  " direct feature extraction failed: " + error;
          return false;
        }
        query.features.assign(featureVector.values.begin(),
                              featureVector.values.end());
      } else {
        FeatureVector featureVector;
        if (!orbit::mapper_features::computeMapperFeatures(
                body.graph, static_cast<int>(rows), static_cast<int>(columns),
                recMII, resMII, lowerBound, featureVector, error)) {
          error = "current task " + body.task + " shape rect-" +
                  std::to_string(rows) + "x" + std::to_string(columns) +
                  " feature extraction failed: " + error;
          return false;
        }
        query.features.assign(featureVector.values.begin(),
                              featureVector.values.end());
      }
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
    llvm::StringRef architectureContract, llvm::StringRef featureContractId,
    llvm::StringRef featureExtractor, llvm::StringRef shapeProtocolId,
    ArrayRef<std::string> featureNames, ArrayRef<CurrentTaskBody> bodies,
    ArrayRef<FeatureQuery> queries, std::string &error) {
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
        root["feature_contract_id"] = featureContractId.str();
        root["feature_extractor"] = featureExtractor.str();
        root["shape_protocol_id"] = shapeProtocolId.str();
        root["feature_width"] = static_cast<int64_t>(featureNames.size());
        json::Array names;
        for (const std::string &name : featureNames)
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

static bool writeCatalogue(
    llvm::raw_ostream &os, llvm::StringRef function,
    llvm::StringRef modelNamespace, llvm::StringRef sourceRepository,
    llvm::StringRef sourceCommit, llvm::StringRef architecturePath,
    llvm::StringRef architectureContract, llvm::StringRef graphVariantId,
    llvm::StringRef modelSchema, llvm::StringRef featureContractId,
    llvm::StringRef featureExtractor, llvm::StringRef shapeProtocolId,
    bool directModel,
    const std::vector<DirectMapperIIPrediction> &directPredictions,
    const json::Object *directSourceModel,
    llvm::ArrayRef<TaskMetadata> taskMetadata, llvm::StringRef candidateCount,
    const std::vector<FeatureQuery> &queries,
    const std::vector<MLPEnsemblePrediction> &predictions,
    const PersistentMLCostCache &cache,
    std::optional<json::Object> architectureTransfer,
    bool allowUnsupportedAboveModelCeiling, StringRef canonicalModuleWitness) {
  json::Object root;
  root["schema"] = "amoeba-task-shape-cost";
  root["function"] = function.str();
  root["namespace"] = modelNamespace.str();

  json::Object metadata;
  metadata["provenance_schema"] = kCostProvenanceSchema.str();
  metadata["predictor_source"] =
      directModel ? "per-cgra-2x2-direct-four-member-ii-predictor"
                  : "ml-ii-startup-predictor";
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
  metadata["model"] = directModel ? kDirectModelName.str() : kModelName.str();
  metadata["model_schema"] = modelSchema.str();
  metadata["model_status"] =
      directModel ? kDirectCandidateStatus.str() : kModelStatus.str();
  metadata["quality_status"] =
      directModel ? kDirectCandidateStatus.str() : kModelStatus.str();
  metadata["production_ready"] = false;
  metadata["feature_contract_id"] = featureContractId.str();
  metadata["feature_extractor"] = featureExtractor.str();
  metadata["feature_frontend"] = directModel
                                     ? "c++-mlir-current-body-direct-148-v1"
                                     : "c++-mlir-current-body-v2";
  metadata["shape_protocol_id"] = shapeProtocolId.str();
  if (directModel) {
    metadata["candidate_only"] = true;
    metadata["amoeba_benchmark_overlap_audit_complete"] = false;
    metadata["old_4x4_labels_reused"] = false;
    metadata["supports_whole_program_latency_or_throughput_claim"] = false;
    if (directSourceModel)
      metadata["source_model"] = json::Value(json::Object(*directSourceModel));
    json::Array memberSeeds;
    for (int64_t seed : kDirectMemberSeeds)
      memberSeeds.push_back(seed);
    metadata["direct_ensemble"] =
        json::Object{{"member_count", 4},
                     {"member_seeds", std::move(memberSeeds)},
                     {"reduction", "arithmetic_mean"},
                     {"uncertainty", "population_standard_deviation"}};
  }
  metadata["communication_latency"] = "scored_by_scheduler";
  metadata["ranking_policy"] =
      json::Object{{"objective", "predicted_scheduler_makespan"},
                   {"mapper_success_probability", "not_predicted"},
                   {"uses_mapper_success_probability", false}};
  metadata["ml_cache"] = json::Object{
      {"identity", "canonical-body-text-and-mapper-shape"},
      {"entries", static_cast<int64_t>(cache.size())},
      {"hits", static_cast<int64_t>(cache.getHitCount())},
      {"misses", static_cast<int64_t>(cache.getMissCount())},
      {"trip_count_in_identity", false}};
  if (allowUnsupportedAboveModelCeiling) {
    metadata["unsupported_prediction_policy"] =
        "analytical-lower-bound-exceeds-model-ceiling-v1";
    metadata["model_interval_max_ii"] = kFormalMax4ModelCeilingII;
    metadata["canonical_module_witness"] = canonicalModuleWitness.str();
  }
  if (architectureTransfer)
    metadata["architecture_transfer"] = std::move(*architectureTransfer);
  root["predictor_metadata"] = std::move(metadata);

  json::Array entries;
  for (size_t index = 0; index < queries.size(); ++index) {
    const FeatureQuery &query = queries[index];
    const MLPEnsemblePrediction &prediction = predictions[index];
    if (query.modelDomainUnsupported) {
      entries.push_back(json::Object{
          {"task", query.task},
          {"mapper_tile_rows", query.rows},
          {"mapper_tile_cols", query.cols},
          {"support_status", "unsupported"},
          {"status", kModelDomainUnsupportedStatus.str()},
          {"unsupported_reason", kModelDomainUnsupportedReason.str()},
          {"analytical_lower_bound", query.lowerBound},
          {"model_interval_max_ii", kFormalMax4ModelCeilingII}});
      continue;
    }
    json::Object entry{
        {"task", query.task},
        {"mapper_tile_rows", query.rows},
        {"mapper_tile_cols", query.cols},
        {"support_status", "supported"},
        {"predicted_ii", prediction.predictedII},
        {"startup_cycles", query.startupCycles},
        {"predicted_ii_std", prediction.predictedIIStd},
        {"analytical_lower_bound", query.lowerBound},
        {"ii_mean_source",
         directModel ? "direct_four_member_arithmetic_mean" : kModelName.str()},
        {"model_status",
         directModel ? kDirectCandidateStatus.str() : kModelStatus.str()},
        {"production_ready", false}};
    if (directModel) {
      if (directPredictions.size() != queries.size())
        return false;
      json::Array members;
      for (size_t member = 0;
           member < directPredictions[index].memberPredictions.size();
           ++member) {
        members.push_back(
            json::Object{{"member_index", static_cast<int64_t>(member)},
                         {"seed", kDirectMemberSeeds[member]},
                         {"predicted_ii",
                          directPredictions[index].memberPredictions[member]}});
      }
      entry["direct_ensemble_members"] = std::move(members);
      entry["candidate_only"] = true;
    }
    entries.push_back(std::move(entry));
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
    return "Predict an exploratory C++ mapper task-shape cost catalogue";
  }

  Option<std::string> functionName{*this, "function",
                                   llvm::cl::desc("Taskflow function name"),
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
  Option<bool> allowUnsupportedAboveModelCeiling{
      *this, "allow-unsupported-above-model-ceiling",
      llvm::cl::desc("Record C++-proved mapper lower bounds above the model "
                     "interval as explicit unsupported shapes"),
      llvm::cl::init(false)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](llvm::StringRef message) {
      module.emitError() << message;
      signalPassFailure();
    };
    const bool directModelMode =
        llvm::StringRef(modelNamespace) == kPerCgra2x2ModelNamespace;
    if (spaceFile.empty() || ensembleFile.empty() ||
        (!directModelMode && checkpointDirectory.empty()) ||
        architectureContract.empty() || architecturePath.empty() ||
        sourceRepository.empty() || sourceCommit.empty() ||
        graphVariantId.empty() || outputFile.empty()) {
      fail("predict-analytical-task-cost-catalog requires space-file, "
           "ensemble-file, (checkpoint-dir for formal-max4), "
           "architecture-contract, "
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
    if (directModelMode && !architectureTransferCatalog.empty()) {
      fail("direct per-CGRA 2x2 predictions cannot consume the legacy "
           "architecture-transfer catalogue");
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

    std::string architectureText;
    if (!readTextFile(architecturePath, architectureText, error)) {
      fail(error);
      return;
    }
    FormalMax4MLPEnsemble model;
    PerCgra2x2DirectEnsemble directModel;
    std::string featureContractId;
    std::string featureExtractor;
    std::string shapeProtocolId;
    std::vector<std::string> featureNames;
    std::string modelSchema;
    MLCostCacheResources cacheResources;
    json::Object directSourceModel;
    if (directModelMode) {
      // The direct bundle validates its embedded exact architecture YAML
      // against the caller's byte-for-byte source before any cache access.
      if (!directModel.load(ensembleFile, architectureText, error) ||
          !readDirectSourceModel(ensembleFile, directSourceModel, error)) {
        fail(error);
        return;
      }
      const auto &names =
          orbit::mapper_features::perCgra2x2MapperFeatureNames();
      featureNames.assign(names.begin(), names.end());
      if (!directModel.featureNamesMatch(featureNames)) {
        fail("direct model feature names differ from the C++ 148-feature "
             "frontend");
        return;
      }
      featureContractId = kPerCgra2x2FeatureContractId.str();
      featureExtractor = "cgra_ii_predictor.mapper_model:mapper_feature_vector";
      shapeProtocolId = kPerCgra2x2ShapeProtocolId.str();
      modelSchema = kPerCgra2x2EnsembleSchema.str();
      cacheResources.architectureText = architectureText;
      if (!readTextFile(ensembleFile, cacheResources.ensembleText, error)) {
        fail(error);
        return;
      }
    } else {
      if (!model.load(ensembleFile, checkpointDirectory, architectureContract,
                      error)) {
        fail(error);
        return;
      }
      const auto &names = orbit::mapper_features::mapperFeatureNames();
      featureNames.assign(names.begin(), names.end());
      if (!model.featureNamesMatch(featureNames)) {
        fail("current C++ feature frontend feature_names differ from the model "
             "contract");
        return;
      }
      featureContractId = model.getFeatureContractId().str();
      featureExtractor = model.getFeatureExtractor().str();
      shapeProtocolId = model.getShapeProtocolId().str();
      modelSchema = model.getModelSchema().str();
      cacheResources = model.makeCacheResources(architectureText);
    }
    std::optional<json::Object> architectureTransfer;
    if (!readArchitectureTransferEvidence(architectureTransferCatalog,
                                          architecturePath,
                                          architectureTransfer, error)) {
      fail(error);
      return;
    }

    // Always derive the mapper-visible body and model-contract features from
    // this module. An optional feature-file is an oracle/debug input only: it
    // must exactly agree with the current module before it can be accepted.
    // This prevents stale or task-name-substituted records from driving costs
    // for a rewritten graph.
    OwningOpRef<ModuleOp> mapperInput;
    std::vector<CurrentTaskBody> currentBodies;
    if (!collectCurrentTaskBodies(*selected, mapperInput, currentBodies,
                                  error)) {
      fail(error);
      return;
    }
    std::vector<FeatureQuery> generatedQueries;
    if (!generateCurrentFeatureQueries(currentBodies, featureNames,
                                       directModelMode, generatedQueries, error,
                                       allowUnsupportedAboveModelCeiling)) {
      fail(error);
      return;
    }
    if (llvm::any_of(generatedQueries, [](const FeatureQuery &query) {
          return query.modelDomainUnsupported;
        }) && (!featureFile.empty() || !featureOutputFile.empty())) {
      fail("feature-file and feature-output do not support explicit "
           "unsupported model-domain queries");
      return;
    }
    if (!llvm::any_of(generatedQueries, [](const FeatureQuery &query) {
          return !query.modelDomainUnsupported;
        })) {
      fail("all task-shape queries are outside the supported model interval");
      return;
    }
    if (!featureFile.empty()) {
      std::vector<FeatureQuery> suppliedQueries;
      if (!readFeatureQueries(featureFile, selected->getName(),
                              architectureContract, featureContractId,
                              featureExtractor, shapeProtocolId, featureNames,
                              directModelMode, suppliedQueries, error) ||
          !verifyFeatureQueriesBoundToCurrentBodies(generatedQueries,
                                                    suppliedQueries, error)) {
        fail(error);
        return;
      }
    }
    if (!featureOutputFile.empty() &&
        !writeCurrentFeatureOutput(
            featureOutputFile, selected->getName(), architectureContract,
            featureContractId, featureExtractor, shapeProtocolId, featureNames,
            currentBodies, generatedQueries, error)) {
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
    if (!cache.load(cacheFile, modelSchema, featureContractId,
                    architectureContract, cacheResources, error)) {
      fail(error);
      return;
    }
    std::vector<MLPEnsemblePrediction> predictions;
    predictions.reserve(queries.size());
    std::vector<DirectMapperIIPrediction> directPredictions;
    if (directModelMode)
      directPredictions.resize(queries.size());
    for (size_t queryIndex = 0; queryIndex < queries.size(); ++queryIndex) {
      const FeatureQuery &query = queries[queryIndex];
      if (query.modelDomainUnsupported) {
        predictions.emplace_back();
        continue;
      }
      MLCostCacheKey key{query.bodyStructuralText, query.rows, query.cols};
      MLCostCacheFacts facts{query.recMII, query.resMII, query.lowerBound};
      MLPEnsemblePrediction prediction;
      if (directModelMode) {
        DirectMapperIIPrediction directPrediction;
        if (!directModel.predict(query.features, query.recMII, query.resMII,
                                 query.lowerBound, directPrediction, error) ||
            directPrediction.memberPredictions.size() !=
                kDirectMemberSeeds.size()) {
          if (error.empty())
            error =
                "direct model did not return exactly four member predictions";
          fail(error);
          return;
        }
        double mean = 0.0;
        for (double value : directPrediction.memberPredictions) {
          if (!std::isfinite(value) || value < query.lowerBound ||
              value > kFormalMax4ModelCeilingII + 1.0e-6) {
            fail("direct member prediction violates finite/lower-bound/ceiling "
                 "contract");
            return;
          }
          mean += value;
        }
        mean /= static_cast<double>(directPrediction.memberPredictions.size());
        double variance = 0.0;
        for (double value : directPrediction.memberPredictions)
          variance += (value - mean) * (value - mean);
        variance /=
            static_cast<double>(directPrediction.memberPredictions.size());
        MLPEnsemblePrediction scalarPrediction;
        scalarPrediction.predictedII = directPrediction.predictedII;
        scalarPrediction.predictedIIStd = std::sqrt(variance);
        if (std::abs(mean - scalarPrediction.predictedII) >
            1.0e-6 * std::max(1.0, std::abs(mean))) {
          fail("direct ensemble arithmetic mean disagrees with its declared "
               "reduction");
          return;
        }
        MLPEnsemblePrediction cachedPrediction;
        if (cache.lookup(key, facts, cachedPrediction)) {
          if (!sameFeatureNumber(cachedPrediction.predictedII,
                                 scalarPrediction.predictedII) ||
              !sameFeatureNumber(cachedPrediction.predictedIIStd,
                                 scalarPrediction.predictedIIStd)) {
            fail("persistent direct-model scalar cache disagrees with the "
                 "loaded bundle");
            return;
          }
          prediction = cachedPrediction;
        } else {
          prediction = scalarPrediction;
          cache.insert(key, facts, prediction);
        }
        directPredictions[queryIndex] = std::move(directPrediction);
      } else {
        if (!cache.lookup(key, facts, prediction)) {
          if (!model.predict(query.features, query.lowerBound, prediction,
                             error)) {
            fail(error);
            return;
          }
          cache.insert(key, facts, prediction);
        }
        if (!std::isfinite(prediction.predictedII) ||
            prediction.predictedII < query.lowerBound ||
            prediction.predictedII > kFormalMax4ModelCeilingII + 1.0e-6 ||
            !std::isfinite(prediction.predictedIIStd) ||
            prediction.predictedIIStd < 0.0) {
          fail("model prediction violates finite/lower-bound/ceiling contract");
          return;
        }
      }
      predictions.push_back(prediction);
    }
    if (!cache.write(cacheFile, modelSchema, featureContractId,
                     architectureContract, cacheResources, error)) {
      fail(error);
      return;
    }
    if (!writeAtomically(
            outputFile,
            [&](llvm::raw_ostream &os) {
              return writeCatalogue(
                  os, selected->getName(), modelNamespace, sourceRepository,
                  sourceCommit, architecturePath, architectureContract,
                  graphVariantId, modelSchema, featureContractId,
                  featureExtractor, shapeProtocolId, directModelMode,
                  directPredictions,
                  directModelMode ? &directSourceModel : nullptr, *taskMetadata,
                  candidateCount, queries, predictions, cache,
                  std::move(architectureTransfer),
                  allowUnsupportedAboveModelCeiling,
                  neighborhoodReplaySourceText(module));
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

bool verifyCurrentModelDomainCostCatalog(ModuleOp module,
                                         StringRef functionName,
                                         StringRef catalogPath,
                                         std::string &error) {
  auto fail = [&](StringRef message) {
    error = message.str();
    return false;
  };
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(catalogPath);
  if (!buffer)
    return fail("cannot read model-domain catalogue for source verification");
  llvm::Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "invalid model-domain catalogue JSON: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  json::Object *root = parsed->getAsObject();
  json::Object *metadata = root ? root->getObject("predictor_metadata") : nullptr;
  json::Array *entries = root ? root->getArray("entries") : nullptr;
  if (!root || !metadata || !entries)
    return fail("model-domain catalogue lacks root metadata or entries");
  const auto namespaceValue = root->getString("namespace");
  const bool directModel =
      namespaceValue && *namespaceValue == kPerCgra2x2ModelNamespace;
  ArrayRef<std::pair<int64_t, int64_t>> shapes =
      directModel ? ArrayRef<std::pair<int64_t, int64_t>>(kPerCgra2x2Shapes)
                  : ArrayRef<std::pair<int64_t, int64_t>>(kFormalMax4Shapes);
  auto witness = metadata->getString("canonical_module_witness");
  auto policy = metadata->getString("unsupported_prediction_policy");
  auto ceiling = metadata->getNumber("model_interval_max_ii");
  if (!witness || *witness != neighborhoodReplaySourceText(module) || !policy ||
      *policy != "analytical-lower-bound-exceeds-model-ceiling-v1" ||
      !ceiling || !std::isfinite(*ceiling) ||
      *ceiling != kFormalMax4ModelCeilingII)
    return fail("model-domain catalogue source binding or interval policy "
                "does not match the exact current module");
  if (directModel &&
      (!requireString(*metadata, "model_schema", kPerCgra2x2EnsembleSchema,
                      error) ||
       !requireString(*metadata, "model_status", kDirectCandidateStatus,
                      error) ||
       !requireString(*metadata, "shape_protocol_id",
                      kPerCgra2x2ShapeProtocolId, error) ||
       !metadata->getBoolean("candidate_only").value_or(false) ||
       metadata->getBoolean("amoeba_benchmark_overlap_audit_complete")
           .value_or(true) ||
       metadata->getBoolean("old_4x4_labels_reused").value_or(true) ||
       metadata
           ->getBoolean("supports_whole_program_latency_or_throughput_claim")
           .value_or(true)))
    return fail("direct model-domain catalogue lost its candidate-only "
                "source contract");
  if (directModel) {
    const json::Object *sourceModel = metadata->getObject("source_model");
    const json::Object *candidate =
        sourceModel ? sourceModel->getObject("candidate_metadata") : nullptr;
    if (!sourceModel || !candidate ||
        !requireString(*sourceModel, "repository", kDirectSourceRepository,
                       error) ||
        !requireString(*sourceModel, "branch", kDirectSourceBranch, error) ||
        !requireString(*sourceModel, "commit", kDirectSourceCommit, error) ||
        !requireString(*candidate, "promotion_status", kDirectCandidateStatus,
                       error) ||
        !requireString(*candidate, "shape_protocol_id",
                       kPerCgra2x2ShapeProtocolId, error) ||
        !candidate->getBoolean("candidate_only").value_or(false) ||
        candidate->getBoolean("amoeba_benchmark_overlap_audit_complete")
            .value_or(true) ||
        candidate->getBoolean("old_4x4_labels_reused").value_or(true) ||
        candidate
            ->getBoolean("supports_whole_program_latency_or_throughput_claim")
            .value_or(true))
      return fail("direct model-domain catalogue has invalid predictor source "
                  "or candidate provenance");
  }

  FailureOr<func::FuncOp> selected =
      selectTaskFunction(module, functionName, error);
  if (failed(selected))
    return false;
  OwningOpRef<ModuleOp> mapperInput;
  std::vector<CurrentTaskBody> bodies;
  if (!collectCurrentTaskBodies(*selected, mapperInput, bodies, error))
    return false;
  using QueryKey = std::tuple<std::string, int64_t, int64_t>;
  std::map<QueryKey, double> expectedBounds;
  for (CurrentTaskBody &body : bodies) {
    for (auto [rows, cols] : shapes) {
      double recMII = 0.0, resMII = 0.0, lowerBound = 0.0;
      if (!computeMapperAnalyticalFacts(body.mapperFunction.getBody(), rows,
                                        cols, recMII, resMII, lowerBound,
                                        error)) {
        error = "source lower-bound recomputation failed for task=" +
                body.task + " shape=rect-" + std::to_string(rows) + "x" +
                std::to_string(cols) + ": " + error;
        return false;
      }
      expectedBounds.emplace(QueryKey{body.task, rows, cols}, lowerBound);
    }
  }
  if (entries->size() != expectedBounds.size())
    return fail("model-domain catalogue does not cover every current task and "
                "model-supported mapper shape exactly once");
  std::set<QueryKey> seen;
  for (llvm::json::Value &value : *entries) {
    json::Object *entry = value.getAsObject();
    if (!entry)
      return fail("model-domain catalogue entry is not an object");
    auto task = entry->getString("task");
    auto rows = entry->getInteger("mapper_tile_rows");
    auto cols = entry->getInteger("mapper_tile_cols");
    auto supportStatus = entry->getString("support_status");
    auto reportedBound = entry->getNumber("analytical_lower_bound");
    if (!task || !rows || !cols || !supportStatus || !reportedBound ||
        !std::isfinite(*reportedBound))
      return fail("model-domain catalogue row lacks task, shape, status, or "
                  "finite analytical lower bound");
    QueryKey key{task->str(), *rows, *cols};
    auto expected = expectedBounds.find(key);
    if (expected == expectedBounds.end() || !seen.insert(key).second ||
        !sameFeatureNumber(*reportedBound, expected->second))
      return fail("catalogue analytical lower bound is stale or does not "
                  "match the current C++ mapper body");
    if (*supportStatus == "supported") {
      if (entry->get("status") || entry->get("unsupported_reason"))
        return fail("supported model-domain row carries unsupported status "
                    "metadata");
      auto predictedII = entry->getNumber("predicted_ii");
      auto startup = entry->getNumber("startup_cycles");
      const json::Array *members =
          directModel ? entry->getArray("direct_ensemble_members") : nullptr;
      if (!predictedII || !startup || !std::isfinite(*predictedII) ||
          !std::isfinite(*startup) || *predictedII < expected->second ||
          *predictedII > kFormalMax4ModelCeilingII + 1.0e-6 ||
          *startup <= 0.0 ||
          (directModel &&
           (!members || members->size() != 4 || entry->get("baseline_ii") ||
            entry->get("large_operation_ii") || entry->get("ranking_ii") ||
            !entry->getBoolean("candidate_only").value_or(false) ||
            entry->getBoolean("production_ready").value_or(true) ||
            entry->getString("model_status") != kDirectCandidateStatus)))
        return fail("supported model-domain row violates current lower-bound "
                    "or model-interval facts");
      if (directModel) {
        std::array<double, 4> memberValues{};
        double mean = 0.0;
        for (size_t index = 0; index < members->size(); ++index) {
          const json::Object *member = (*members)[index].getAsObject();
          auto memberIndex =
              member ? member->getInteger("member_index") : std::nullopt;
          auto seed = member ? member->getInteger("seed") : std::nullopt;
          auto value =
              member ? member->getNumber("predicted_ii") : std::nullopt;
          if (!memberIndex || *memberIndex != static_cast<int64_t>(index) ||
              !seed || *seed != kDirectMemberSeeds[index] || !value ||
              !std::isfinite(*value) || *value < expected->second ||
              *value > kFormalMax4ModelCeilingII + 1.0e-6)
            return fail(
                "direct model-domain row has an invalid member prediction");
          memberValues[index] = *value;
          mean += *value;
        }
        mean /= memberValues.size();
        double variance = 0.0;
        for (double value : memberValues)
          variance += (value - mean) * (value - mean);
        variance /= memberValues.size();
        auto reportedStd = entry->getNumber("predicted_ii_std");
        auto meanSource = entry->getString("ii_mean_source");
        if (std::abs(mean - *predictedII) >
                1.0e-6 * std::max(1.0, std::abs(mean)) ||
            !reportedStd ||
            !sameFeatureNumber(std::sqrt(variance), *reportedStd) ||
            !meanSource || *meanSource != "direct_four_member_arithmetic_mean")
          return fail(
              "direct model-domain aggregate disagrees with its four members");
      }
      continue;
    }
    auto status = entry->getString("status");
    auto reason = entry->getString("unsupported_reason");
    auto rowCeiling = entry->getNumber("model_interval_max_ii");
    if (*supportStatus != "unsupported" || !status ||
        *status != kModelDomainUnsupportedStatus || !reason ||
        *reason != kModelDomainUnsupportedReason || !rowCeiling ||
        !std::isfinite(*rowCeiling) ||
        *rowCeiling != kFormalMax4ModelCeilingII ||
        expected->second <= kFormalMax4ModelCeilingII ||
        entry->get("predicted_ii") || entry->get("startup_cycles") ||
        entry->get("predicted_ii_std") || entry->get("ii_mean_source") ||
        entry->get("direct_ensemble_members") || entry->get("model_status") ||
        entry->get("production_ready") ||
        entry->get("mapper_success_probability"))
      return fail("unsupported model-domain row is not justified by the "
                  "current source lower bound");
  }
  if (seen.size() != expectedBounds.size())
    return fail("model-domain catalogue omits a current task/shape query");
  if (!llvm::any_of(*entries, [](const json::Value &value) {
        const json::Object *entry = value.getAsObject();
        return entry && entry->getString("support_status") == "supported";
      }))
    return fail("model-domain catalogue contains no supported query");
  return true;
}

std::unique_ptr<Pass> createPredictAnalyticalTaskCostCatalogPass() {
  return std::make_unique<PredictAnalyticalTaskCostCatalogPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir
