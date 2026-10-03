//===- RebindAnalyticalTaskCostCatalogPass.cpp ---------------------------===//
//
// Rebinds a task-shape II catalogue to the exact factored spatial descriptor.
// The pass is deliberately fail-closed: it copies predictor records only
// after checking function, graph, Git, architecture, model, and complete
// task/mapper-shape coverage provenance.  It never creates a missing II.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalTaskCostCatalog.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <utility>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;
using namespace mlir::taskflow;
namespace json = llvm::json;

namespace {

constexpr StringLiteral kFactoredSpaceSchema = "amoeba-analytical-task-space";
constexpr StringLiteral kRebindSchema = "orbit-cost-catalog-rebind-v1";
constexpr StringLiteral kMlPredictorSource = "ml-ii-startup-predictor";

struct ShapeKey {
  std::string task;
  int64_t rows = 0;
  int64_t cols = 0;

  bool operator<(const ShapeKey &other) const {
    return std::tie(task, rows, cols) <
           std::tie(other.task, other.rows, other.cols);
  }
  bool operator==(const ShapeKey &other) const {
    return task == other.task && rows == other.rows && cols == other.cols;
  }
};

struct FactoredDescriptor {
  std::string function;
  std::string graphVariantId;
  std::string candidateCount;
  std::vector<std::string> tasks;
  std::set<ShapeKey> shapes;
};

static std::optional<StringRef> requiredString(const json::Object &object,
                                               StringRef key,
                                               std::string &error) {
  std::optional<StringRef> value = object.getString(key);
  if (!value || value->empty())
    error = "missing or empty string field \"" + key.str() + "\"";
  return value;
}

static std::optional<int64_t> requiredInteger(const json::Object &object,
                                              StringRef key,
                                              std::string &error) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value)
    error = "missing or invalid integer field \"" + key.str() + "\"";
  return value;
}

static std::optional<bool> requiredBoolean(const json::Object &object,
                                           StringRef key,
                                           std::string &error) {
  std::optional<bool> value = object.getBoolean(key);
  if (!value)
    error = "missing or invalid boolean field \"" + key.str() + "\"";
  return value;
}

// JSON integers are signed 64-bit in LLVM's JSON API.  Keep the exact
// factored-space cardinality as canonical decimal text when it is larger.
static std::optional<std::string>
requiredDecimal(const json::Object &object, StringRef key, std::string &error,
                bool requirePositive = true) {
  if (std::optional<int64_t> numeric = object.getInteger(key)) {
    if (*numeric < 0 || (requirePositive && *numeric == 0)) {
      error = "invalid decimal field \"" + key.str() + "\"";
      return std::nullopt;
    }
    return std::to_string(*numeric);
  }
  std::optional<StringRef> text = object.getString(key);
  if (!text || text->empty() ||
      llvm::any_of(*text, [](char value) {
        return value < '0' || value > '9';
      })) {
    error = "missing or invalid decimal field \"" + key.str() + "\"";
    return std::nullopt;
  }
  StringRef normalized = text->ltrim('0');
  if (normalized.empty() && requirePositive) {
    error = "invalid decimal field \"" + key.str() + "\"";
    return std::nullopt;
  }
  return (normalized.empty() ? StringRef("0") : normalized).str();
}

static std::string multiplyDecimal(StringRef value, uint64_t factor) {
  if (factor == 0)
    return "0";
  std::string result;
  uint64_t carry = 0;
  for (auto it = value.rbegin(); it != value.rend(); ++it) {
    uint64_t digit = static_cast<uint64_t>(*it - '0');
    unsigned __int128 product = static_cast<unsigned __int128>(digit) * factor + carry;
    result.push_back(static_cast<char>('0' + static_cast<unsigned>(product % 10)));
    carry = static_cast<uint64_t>(product / 10);
  }
  while (carry) {
    result.push_back(static_cast<char>('0' + carry % 10));
    carry /= 10;
  }
  std::reverse(result.begin(), result.end());
  return result;
}

static void putDecimal(json::Object &object, StringRef key, StringRef decimal) {
  uint64_t value = 0;
  if (!decimal.getAsInteger(10, value) &&
      value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    object[key] = static_cast<int64_t>(value);
  else
    object[key] = decimal.str();
}

static bool parseFactoredDescriptor(StringRef path, StringRef expectedFunction,
                                    StringRef expectedGraphVariant,
                                    FactoredDescriptor &descriptor,
                                    std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read factored descriptor " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }

  std::optional<json::Object> header;
  std::optional<json::Object> footer;
  size_t recordCount = 0;
  for (llvm::line_iterator lines(**buffer, /*SkipBlanks=*/true);
       !lines.is_at_end(); ++lines) {
    llvm::Expected<json::Value> parsed = json::parse(*lines);
    if (!parsed) {
      error = "invalid factored descriptor JSONL: " +
              llvm::toString(parsed.takeError());
      return false;
    }
    json::Object *object = parsed->getAsObject();
    if (!object) {
      error = "factored descriptor record is not an object";
      return false;
    }
    auto recordType = object->getString("record_type");
    if (!recordType) {
      error = "factored descriptor record has no record_type";
      return false;
    }
    if (*recordType == "space" && !header && recordCount == 0)
      header = std::move(*object);
    else if (*recordType == "footer" && header && !footer && recordCount == 1)
      footer = std::move(*object);
    else {
      error = "factored descriptor must contain one space header followed by "
              "one footer";
      return false;
    }
    ++recordCount;
  }
  if (!header || !footer || recordCount != 2) {
    error = "factored descriptor is missing its header or footer";
    return false;
  }

  auto schema = requiredString(*header, "schema", error);
  auto representation = requiredString(*header, "representation", error);
  auto exact = header->getBoolean("exact");
  auto function = requiredString(*header, "function", error);
  auto graphVariant = requiredString(*header, "graph_variant_id", error);
  auto candidateCount = requiredDecimal(*header, "candidate_count", error);
  auto factors = header->getArray("factors");
  if (!schema || !representation || !exact || !function || !graphVariant ||
      !candidateCount || !factors)
    return false;
  if (*schema != kFactoredSpaceSchema || *representation != "factored" ||
      !*exact || *function != expectedFunction ||
      *graphVariant != expectedGraphVariant || factors->empty()) {
    error = "factored descriptor function, graph, schema, or exactness does "
            "not match the requested candidate space";
    return false;
  }

  auto footerSchema = requiredString(*footer, "schema", error);
  auto footerRepresentation = requiredString(*footer, "representation", error);
  auto footerCount = requiredDecimal(*footer, "candidate_count", error);
  auto ranked = footer->getBoolean("ranked");
  if (!footerSchema || !footerRepresentation || !footerCount || !ranked)
    return false;
  if (*footerSchema != *schema || *footerRepresentation != *representation ||
      *footerCount != *candidateCount || *ranked) {
    error = "factored descriptor footer does not exactly repeat the space "
            "cardinality and unranked status";
    return false;
  }

  for (json::Value &factorValue : *factors) {
    json::Object *factor = factorValue.getAsObject();
    if (!factor) {
      error = "factored descriptor factor is not an object";
      return false;
    }
    auto task = requiredString(*factor, "task", error);
    auto shapes = factor->getArray("shapes");
    if (!task || !shapes || shapes->empty())
      return false;
    if (llvm::is_contained(descriptor.tasks, task->str())) {
      error = "factored descriptor repeats task " + task->str();
      return false;
    }
    descriptor.tasks.push_back(task->str());
    for (json::Value &shapeValue : *shapes) {
      json::Object *shape = shapeValue.getAsObject();
      if (!shape) {
        error = "factored descriptor shape is not an object";
        return false;
      }
      auto rows = requiredInteger(*shape, "mapper_tile_rows", error);
      auto cols = requiredInteger(*shape, "mapper_tile_cols", error);
      if (!rows || !cols || *rows <= 0 || *cols <= 0) {
        error = "factored descriptor mapper dimensions must be positive";
        return false;
      }
      if (!descriptor.shapes.insert({task->str(), *rows, *cols}).second) {
        error = "factored descriptor repeats task/mapper shape " +
                task->str();
        return false;
      }
    }
  }
  std::string computedCount = "1";
  for (json::Value &factorValue : *factors) {
    json::Object *factor = factorValue.getAsObject();
    auto shapes = factor->getArray("shapes");
    computedCount = multiplyDecimal(computedCount, shapes->size());
  }
  if (computedCount != *candidateCount) {
    error = "factored descriptor candidate_count does not equal the exact "
            "Cartesian product of its factors";
    return false;
  }
  descriptor.function = function->str();
  descriptor.graphVariantId = graphVariant->str();
  descriptor.candidateCount = *candidateCount;
  return true;
}

static bool isGitCommit(StringRef value) {
  return value.size() >= 7 && value.size() <= 64 &&
         llvm::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f') ||
                  (character >= 'A' && character <= 'F');
         });
}

static bool rejectMeasuredMapperFields(const json::Object &object,
                                      StringRef where,
                                      std::string &error) {
  for (const auto &item : object) {
    StringRef key = item.first;
    if (key.contains_insensitive("measured") ||
        key.contains_insensitive("mapper_cost") ||
        key == "native_cycles" || key == "mapper_cycles") {
      error = "measured mapper cost cache field is forbidden in " +
              where.str() + ": " + key.str();
      return true;
    }
  }
  return false;
}

static bool validateModelProvenance(const json::Object &metadata,
                                    StringRef actualNamespace,
                                    StringRef expectedNamespace,
                                    StringRef expectedModel,
                                    StringRef expectedModelSchema,
                                    StringRef expectedFeatureContract,
                                    StringRef expectedExtractor,
                                    std::string &error) {
  auto model = requiredString(metadata, "model", error);
  auto modelSchema = requiredString(metadata, "model_schema", error);
  auto featureContract =
      requiredString(metadata, "feature_contract_id", error);
  auto extractor = requiredString(metadata, "feature_extractor", error);
  auto modelStatus = requiredString(metadata, "model_status", error);
  auto qualityStatus = requiredString(metadata, "quality_status", error);
  auto productionReady = requiredBoolean(metadata, "production_ready", error);
  if (!model || !modelSchema || !featureContract ||
      !extractor || !modelStatus || !qualityStatus || !productionReady)
    return false;
  if (actualNamespace.empty() || expectedNamespace.empty() ||
      expectedModel.empty() || expectedModelSchema.empty() ||
      expectedFeatureContract.empty() || expectedExtractor.empty() ||
      actualNamespace != expectedNamespace ||
      *model != expectedModel || *modelSchema != expectedModelSchema ||
      *featureContract != expectedFeatureContract ||
      *extractor != expectedExtractor) {
    error = "ML model provenance is absent or does not match the requested "
            "model contract";
    return false;
  }
  return true;
}

static bool validateCatalogCoverage(
    StringRef path, StringRef expectedFunction, ArrayRef<TaskMetadata> tasks,
    const FactoredDescriptor &descriptor, StringRef sourceRepository,
    StringRef sourceCommit, StringRef architecturePath, StringRef graphVariant,
    StringRef expectedNamespace, StringRef expectedModel,
    StringRef expectedModelSchema, StringRef expectedFeatureContract,
    StringRef expectedExtractor, json::Object &root, std::string &sourceCount,
    std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read cost catalogue " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  llvm::Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "invalid cost catalogue JSON: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  json::Object *parsedRoot = parsed->getAsObject();
  if (!parsedRoot) {
    error = "cost catalogue must be a JSON object";
    return false;
  }
  auto schema = requiredString(*parsedRoot, "schema", error);
  auto function = requiredString(*parsedRoot, "function", error);
  auto namespaceValue = requiredString(*parsedRoot, "namespace", error);
  json::Object *metadata = parsedRoot->getObject("predictor_metadata");
  json::Array *entries = parsedRoot->getArray("entries");
  if (!schema || !function || !namespaceValue || !metadata || !entries)
    return false;
  if (rejectMeasuredMapperFields(*parsedRoot, "catalogue", error) ||
      rejectMeasuredMapperFields(*metadata, "predictor metadata", error))
    return false;
  if (const json::Value *rawSource = metadata->get("predictor_source")) {
    std::optional<StringRef> source = rawSource->getAsString();
    if (!source || *source != kMlPredictorSource) {
      error = "cost catalogue predictor_source conflicts with the ML II/startup "
              "predictor contract";
      return false;
    }
  }
  if (*schema != kCostSchema || *function != expectedFunction ||
      *namespaceValue != expectedNamespace) {
    error = "cost catalogue schema, function, or predictor namespace does "
            "not match";
    return false;
  }

  if (!validateModelProvenance(*metadata, *namespaceValue, expectedNamespace,
                               expectedModel,
                               expectedModelSchema, expectedFeatureContract,
                               expectedExtractor, error))
    return false;

  // Reuse the production parser for Git, architecture, task IDs, ranking
  // policy, numeric II/startup validation, and unsupported-entry censorship.
  TaskShapeCostCache cache;
  if (!cache.load(path, expectedFunction, tasks, sourceRepository, sourceCommit,
                  architecturePath, graphVariant, error))
    return false;
  auto metadataCount =
      requiredDecimal(*metadata, "candidate_count", error, /*positive=*/true);
  if (!metadataCount)
    return false;
  sourceCount = *metadataCount;

  if (descriptor.tasks.size() != tasks.size()) {
    error = "factored descriptor task count does not match current IR";
    return false;
  }
  for (auto [index, task] : llvm::enumerate(tasks)) {
    if (descriptor.tasks[index] != task.name) {
      error = "factored descriptor task order does not match current IR";
      return false;
    }
  }

  std::set<ShapeKey> catalogShapes;
  for (json::Value &entryValue : *entries) {
    json::Object *entry = entryValue.getAsObject();
    if (!entry) {
      error = "cost catalogue entry is not an object";
      return false;
    }
    if (rejectMeasuredMapperFields(*entry, "cost entry", error))
      return false;
    auto task = requiredString(*entry, "task", error);
    auto rows = requiredInteger(*entry, "mapper_tile_rows", error);
    auto cols = requiredInteger(*entry, "mapper_tile_cols", error);
    if (!task || !rows || !cols)
      return false;
    if (!catalogShapes.insert({task->str(), *rows, *cols}).second) {
      error = "cost catalogue contains duplicate task/mapper shape";
      return false;
    }
  }
  if (catalogShapes != descriptor.shapes) {
    error = "cost catalogue does not exactly cover every task/oriented mapper "
            "shape in the factored descriptor";
    return false;
  }

  root = std::move(*parsedRoot);
  return true;
}

struct RebindAnalyticalTaskCostCatalogPass
    : PassWrapper<RebindAnalyticalTaskCostCatalogPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RebindAnalyticalTaskCostCatalogPass)

  RebindAnalyticalTaskCostCatalogPass() = default;
  RebindAnalyticalTaskCostCatalogPass(
      const RebindAnalyticalTaskCostCatalogPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "rebind-analytical-task-cost-catalog";
  }
  StringRef getDescription() const override {
    return "Rebind an ML II catalogue to an exact factored task-shape space";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, TaskflowDialect>();
  }

  Option<std::string> functionName{
      *this, "function", llvm::cl::desc("Taskflow function to validate"),
      llvm::cl::init("")};
  Option<std::string> costFile{
      *this, "cost-file", llvm::cl::desc("Input predictor catalogue"),
      llvm::cl::init("")};
  Option<std::string> spaceFile{
      *this, "space-file", llvm::cl::desc("Exact factored space JSONL"),
      llvm::cl::init("")};
  Option<std::string> outputFile{
      *this, "output", llvm::cl::desc("Rebound catalogue output"),
      llvm::cl::init("")};
  Option<std::string> sourceRepository{
      *this, "source-git-repository", llvm::cl::desc("Expected Git repository"),
      llvm::cl::init("")};
  Option<std::string> sourceCommit{
      *this, "source-git-commit", llvm::cl::desc("Expected Git commit"),
      llvm::cl::init("")};
  Option<std::string> architecturePath{
      *this, "architecture-path", llvm::cl::desc("Expected architecture path"),
      llvm::cl::init("")};
  Option<std::string> graphVariantId{
      *this, "graph-variant-id", llvm::cl::desc("Expected graph variant"),
      llvm::cl::init("")};
  Option<std::string> modelNamespace{
      *this, "model-namespace", llvm::cl::desc("Expected predictor namespace"),
      llvm::cl::init("")};
  Option<std::string> model{
      *this, "model", llvm::cl::desc("Expected predictor model identity"),
      llvm::cl::init("")};
  Option<std::string> modelSchema{
      *this, "model-schema", llvm::cl::desc("Expected predictor model schema"),
      llvm::cl::init("")};
  Option<std::string> featureContract{
      *this, "feature-contract-id",
      llvm::cl::desc("Expected predictor feature contract"), llvm::cl::init("")};
  Option<std::string> featureExtractor{
      *this, "feature-extractor",
      llvm::cl::desc("Expected predictor feature extractor"),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selected)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp function = *selected;
    if (costFile.getValue().empty() || spaceFile.getValue().empty() ||
        outputFile.getValue().empty() || sourceRepository.getValue().empty() ||
        sourceCommit.getValue().empty() || architecturePath.getValue().empty() ||
        graphVariantId.getValue().empty() || modelNamespace.getValue().empty() ||
        model.getValue().empty() || modelSchema.getValue().empty() ||
        featureContract.getValue().empty() || featureExtractor.getValue().empty() ||
        samePath(costFile.getValue(), outputFile.getValue()) ||
        samePath(spaceFile.getValue(), outputFile.getValue()) ||
        samePath(costFile.getValue(), spaceFile.getValue()) ||
        !isGitCommit(sourceCommit.getValue())) {
      function.emitError()
          << "cost rebinding requires distinct paths and complete source, "
             "architecture, graph, and model provenance options";
      return signalPassFailure();
    }

    FailureOr<SmallVector<TaskMetadata>> tasks =
        collectAnalyticalTaskMetadata(function, error);
    if (failed(tasks)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    FactoredDescriptor descriptor;
    if (!parseFactoredDescriptor(spaceFile.getValue(), function.getSymName(),
                                 graphVariantId, descriptor, error)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    json::Object root;
    std::string sourceCount;
    if (!validateCatalogCoverage(
            costFile.getValue(), function.getSymName(), *tasks, descriptor,
            sourceRepository.getValue(), sourceCommit.getValue(),
            architecturePath.getValue(), graphVariantId.getValue(),
            modelNamespace.getValue(), model.getValue(), modelSchema.getValue(),
            featureContract.getValue(), featureExtractor.getValue(), root, sourceCount, error)) {
      function.emitError() << error;
      return signalPassFailure();
    }

    json::Object *metadata = root.getObject("predictor_metadata");
    if (!metadata) {
      function.emitError() << "validated catalogue lost predictor metadata";
      return signalPassFailure();
    }
    putDecimal(*metadata, "candidate_count", descriptor.candidateCount);
    (*metadata)["predictor_source"] = kMlPredictorSource.str();
    (*metadata)["rebind_provenance"] = json::Object{
        {"schema", kRebindSchema},
        {"status", "complete"},
        {"coverage", "exact-task-oriented-mapper-shapes"},
        {"input_candidate_count", sourceCount},
        {"rebound_candidate_count", descriptor.candidateCount},
        {"space_schema", kFactoredSpaceSchema},
        {"space_path", spaceFile.getValue()},
        {"source_catalog_path", costFile.getValue()},
        {"model_provenance_checked", true},
        {"source_provenance_checked", true}};

    if (!writeAtomically(
            outputFile.getValue(),
            [&](llvm::raw_ostream &os) {
              os << json::Value(std::move(root)) << "\n";
              return true;
            },
            error)) {
      function.emitError() << error;
      return signalPassFailure();
    }
    llvm::errs() << "[JointScheduling] rebound exact cost catalogue with "
                 << descriptor.shapes.size() << " task/mapper-shape entries and "
                 << descriptor.candidateCount << " factored candidates\n";
  }
};

} // namespace

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createRebindAnalyticalTaskCostCatalogPass() {
  return std::make_unique<RebindAnalyticalTaskCostCatalogPass>();
}
} // namespace mlir::amoeba::neura
