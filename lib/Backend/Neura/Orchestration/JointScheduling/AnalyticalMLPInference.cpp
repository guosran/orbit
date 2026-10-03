//===- AnalyticalMLPInference.cpp ------------------------------*- C++ -*-===//

#include "AnalyticalMLPInference.h"

#include "AnalyticalTaskCandidateCommon.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

using namespace mlir::amoeba::neura::joint_scheduling;
namespace json = llvm::json;

namespace {

constexpr float kMapperIICeiling = 20.0F;
constexpr float kFactsTolerance = 1.0e-6F;
constexpr llvm::StringLiteral kMLCostCacheSchema =
    "orbit-cgra-ii-ml-cache-v2";

static bool finiteNumber(double value) { return std::isfinite(value); }

static bool getRequiredString(const json::Object &object, llvm::StringRef key,
                              std::string &result, std::string &error) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value || value->empty()) {
    error = "missing or empty string field \"" + key.str() + "\"";
    return false;
  }
  result = value->str();
  return true;
}

static bool getRequiredNumber(const json::Object &object, llvm::StringRef key,
                              double &result, std::string &error) {
  std::optional<double> value = object.getNumber(key);
  if (!value || !finiteNumber(*value)) {
    error = "missing or non-finite number field \"" + key.str() + "\"";
    return false;
  }
  result = *value;
  return true;
}

static bool getRequiredInteger(const json::Object &object, llvm::StringRef key,
                               int64_t &result, std::string &error) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value) {
    error = "missing or invalid integer field \"" + key.str() + "\"";
    return false;
  }
  result = *value;
  return true;
}

static bool rejectDigestIdentityKeys(const json::Value &value,
                                     std::string &error,
                                     llvm::StringRef context) {
  const json::Object *object = value.getAsObject();
  if (object) {
    for (const auto &item : *object) {
      std::string key = item.first.str();
      std::string lower;
      lower.reserve(key.size());
      for (char character : key)
        lower.push_back(static_cast<char>(std::tolower(
            static_cast<unsigned char>(character))));
      // "shape" and "shape_protocol" are legitimate model fields.  Reject
      // only actual digest identity fields, never ordinary shape metadata.
      if (lower == "hash" || lower == "sha256" ||
          llvm::StringRef(lower).ends_with("_hash") ||
          llvm::StringRef(lower).ends_with("_sha256")) {
        error = context.str() + " contains a forbidden digest identity field " +
                key;
        return false;
      }
      if (!rejectDigestIdentityKeys(item.second, error, context))
        return false;
    }
    return true;
  }
  const json::Array *array = value.getAsArray();
  if (!array)
    return true;
  for (const json::Value &element : *array)
    if (!rejectDigestIdentityKeys(element, error, context))
      return false;
  return true;
}

static bool readJsonObject(llvm::StringRef path, json::Object &object,
                           std::string &error,
                           std::string *sourceText = nullptr) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read JSON file " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  llvm::Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "cannot parse JSON file " + path.str() + ": " +
            llvm::toString(parsed.takeError());
    return false;
  }
  if (sourceText)
    *sourceText = (*buffer)->getBuffer().str();
  json::Object *parsedObject = parsed->getAsObject();
  if (!parsedObject) {
    error = "JSON file " + path.str() + " does not contain an object";
    return false;
  }
  if (!rejectDigestIdentityKeys(*parsed, error, path))
    return false;
  object = std::move(*parsedObject);
  return true;
}

static bool requireExactString(const json::Object &object, llvm::StringRef key,
                               llvm::StringRef expected, std::string &error) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value || *value != expected) {
    error = "field \"" + key.str() + "\" does not match \"" +
            expected.str() + "\"";
    return false;
  }
  return true;
}

static bool requireExactBoolean(const json::Object &object,
                                llvm::StringRef key, bool expected,
                                std::string &error) {
  std::optional<bool> value = object.getBoolean(key);
  if (!value || *value != expected) {
    error = "field \"" + key.str() + "\" has the wrong boolean value";
    return false;
  }
  return true;
}

static bool parseShapeArray(const json::Value *raw, unsigned expectedRank,
                            unsigned expectedRows, unsigned expectedCols,
                            std::string &error) {
  const json::Array *array = raw ? raw->getAsArray() : nullptr;
  if (!array || array->size() != expectedRank) {
    error = "tensor shape has the wrong rank";
    return false;
  }
  if (expectedRank == 1) {
    std::optional<int64_t> value = (*array)[0].getAsInteger();
    if (!value || *value != static_cast<int64_t>(expectedRows)) {
      error = "tensor shape has the wrong width";
      return false;
    }
    return true;
  }
  std::optional<int64_t> rows = (*array)[0].getAsInteger();
  std::optional<int64_t> cols = (*array)[1].getAsInteger();
  if (!rows || !cols || *rows != static_cast<int64_t>(expectedRows) ||
      *cols != static_cast<int64_t>(expectedCols)) {
    error = "tensor shape has the wrong dimensions";
    return false;
  }
  return true;
}

static bool appendTensorValues(const json::Value &value,
                               std::vector<float> &values,
                               std::string &error) {
  if (const json::Array *array = value.getAsArray()) {
    for (const json::Value &element : *array)
      if (!appendTensorValues(element, values, error))
        return false;
    return true;
  }
  std::optional<double> number = value.getAsNumber();
  if (!number || !finiteNumber(*number) ||
      *number > std::numeric_limits<float>::max() ||
      *number < -std::numeric_limits<float>::max()) {
    error = "tensor contains a non-finite or out-of-range value";
    return false;
  }
  values.push_back(static_cast<float>(*number));
  return true;
}

static bool parseTensor(const json::Object &tensors, llvm::StringRef name,
                        unsigned rows, unsigned cols, bool vector,
                        std::vector<float> &values, std::string &error) {
  const json::Value *raw = tensors.get(name);
  const json::Object *tensor = raw ? raw->getAsObject() : nullptr;
  if (!tensor) {
    error = "missing tensor \"" + name.str() + "\"";
    return false;
  }
  const json::Value *shape = tensor->get("shape");
  if (!parseShapeArray(shape, vector ? 1 : 2, rows, cols, error)) {
    error = "tensor \"" + name.str() + "\": " + error;
    return false;
  }
  const json::Value *rawValues = tensor->get("values");
  if (!rawValues) {
    error = "tensor \"" + name.str() + "\" has no values";
    return false;
  }
  values.clear();
  if (!appendTensorValues(*rawValues, values, error)) {
    error = "tensor \"" + name.str() + "\": " + error;
    return false;
  }
  size_t expected = vector ? rows : static_cast<size_t>(rows) * cols;
  if (values.size() != expected) {
    error = "tensor \"" + name.str() + "\" value count does not match " +
            std::to_string(expected);
    return false;
  }
  return true;
}

static bool validateFeatureContract(const json::Object &contract,
                                    llvm::StringRef expectedId,
                                    std::string &error,
                                    std::vector<std::string> *namesOut = nullptr) {
  if (!requireExactString(contract, "contract_id", expectedId, error) ||
      !requireExactString(contract, "extractor", kFormalFeatureExtractor,
                          error))
    return false;
  std::optional<int64_t> width = contract.getInteger("width");
  if (!width || *width != static_cast<int64_t>(kFormalMapperFeatureWidth)) {
    error = "feature contract width is not 156";
    return false;
  }
  const json::Array *names = contract.getArray("feature_names");
  if (!names || names->size() != kFormalMapperFeatureWidth) {
    error = "feature contract must enumerate all 156 feature names";
    return false;
  }
  std::set<std::string> uniqueNames;
  if (namesOut)
    namesOut->clear();
  for (const json::Value &value : *names) {
    std::optional<llvm::StringRef> name = value.getAsString();
    if (!name || name->empty() || !uniqueNames.insert(name->str()).second) {
      error = "feature contract contains a missing or duplicate feature name";
      return false;
    }
    if (namesOut)
      namesOut->push_back(name->str());
  }
  const json::Array *inputs = contract.getArray("inputs");
  if (!inputs || inputs->size() != 6) {
    error = "feature contract inputs are incomplete";
    return false;
  }
  static constexpr llvm::StringLiteral expectedInputs[] = {
      "route_expanded_task_dfg", "mapper_rows", "mapper_columns", "rec_mii",
      "res_mii", "analytical_lower_bound"};
  for (unsigned index = 0; index < 6; ++index) {
    std::optional<llvm::StringRef> value = (*inputs)[index].getAsString();
    if (!value || *value != expectedInputs[index]) {
      error = "feature contract input ordering differs from the pinned model";
      return false;
    }
  }
  return requireExactString(contract, "shape_protocol_id", kFormalShapeProtocolId,
                            error);
}

static bool validateSupportedShapeArray(const json::Object &shapeProtocol,
                                        std::string &error) {
  if (!requireExactString(shapeProtocol, "protocol_id", kFormalShapeProtocolId,
                          error))
    return false;
  const json::Array *shapes = shapeProtocol.getArray("supported_mapper_tile_shapes");
  if (!shapes || shapes->size() != 8) {
    error = "shape protocol does not enumerate the eight formal mapper shapes";
    return false;
  }
  static constexpr std::pair<int64_t, int64_t> expected[] = {
      {4, 4}, {4, 8}, {8, 4}, {4, 12}, {12, 4}, {4, 16}, {8, 8}, {16, 4}};
  std::set<std::pair<int64_t, int64_t>> actual;
  for (const json::Value &value : *shapes) {
    const json::Object *shape = value.getAsObject();
    if (!shape) {
      error = "shape protocol contains a non-object mapper shape";
      return false;
    }
    int64_t rows = 0;
    int64_t cols = 0;
    if (!getRequiredInteger(*shape, "rows", rows, error) ||
        !getRequiredInteger(*shape, "cols", cols, error))
      return false;
    actual.insert({rows, cols});
  }
  std::set<std::pair<int64_t, int64_t>> expectedSet(expected, expected + 8);
  if (actual != expectedSet) {
    error = "shape protocol supported mapper shapes differ from the pinned model";
    return false;
  }
  return true;
}

} // namespace

bool FormalMax4MLPEnsemble::load(llvm::StringRef ensemblePath,
                                  llvm::StringRef checkpointDirectory,
                                  llvm::StringRef requestedArchitectureContract,
                                  std::string &error) {
  loaded = false;
  featureNames.clear();
  checkpoints.clear();
  ensembleWeights.clear();
  ensembleSourceText.clear();
  checkpointSourceTexts.clear();
  if (ensemblePath.empty() || checkpointDirectory.empty() ||
      requestedArchitectureContract.empty()) {
    error = "ensemble, checkpoint directory, and architecture contract are "
            "required";
    return false;
  }

  json::Object ensemble;
  if (!readJsonObject(ensemblePath, ensemble, error, &ensembleSourceText))
    return false;
  if (!requireExactString(ensemble, "schema", kFormalEnsembleSchema, error) ||
      !requireExactString(ensemble, "quality_status", "exploratory", error) ||
      !requireExactBoolean(ensemble, "production_ready", false, error))
    return false;

  const json::Object *featureContract = ensemble.getObject("feature_contract");
  if (!featureContract ||
      !validateFeatureContract(*featureContract, kFormalFeatureContractId,
                               error, &featureNames))
    return false;
  const json::Object *shapeProtocol = ensemble.getObject("shape_protocol");
  if (!shapeProtocol || !validateSupportedShapeArray(*shapeProtocol, error))
    return false;

  const json::Object *architecture = ensemble.getObject("architecture");
  if (!architecture ||
      !requireExactString(*architecture, "protocol_id", kFormalShapeProtocolId,
                          error))
    return false;
  // The C++ feature frontend supplies the actual architecture contract.  The
  // model package is tied to the same 4x4 protocol; a non-empty caller
  // contract is retained in cache metadata and compared on reload.
  architectureContract = requestedArchitectureContract.str();
  modelSchema = kFormalEnsembleSchema.str();
  featureContractId = kFormalFeatureContractId.str();
  featureExtractor = kFormalFeatureExtractor.str();
  shapeProtocolId = kFormalShapeProtocolId.str();

  const json::Object *weights = ensemble.getObject("weights");
  if (!weights || weights->size() != 4) {
    error = "ensemble weights must contain exactly four entries";
    return false;
  }
  static constexpr llvm::StringLiteral weightNames[] = {
      "analytical_lower_bound", "baseline", "large-operation", "ranking"};
  double weightSum = 0.0;
  std::vector<float> parsedWeights;
  parsedWeights.reserve(4);
  for (llvm::StringRef name : weightNames) {
    double value = 0.0;
    if (!getRequiredNumber(*weights, name, value, error) || value < 0.0) {
      if (error.empty())
        error = "ensemble weight must be non-negative";
      return false;
    }
    parsedWeights.push_back(static_cast<float>(value));
    weightSum += value;
  }
  if (std::fabs(weightSum - 1.0) > 1.0e-6) {
    error = "ensemble weights do not form a probability simplex";
    return false;
  }

  const json::Object *checkpointRecords = ensemble.getObject("checkpoints");
  if (!checkpointRecords || checkpointRecords->size() != 3) {
    error = "ensemble checkpoints must contain baseline, large-operation, and "
            "ranking";
    return false;
  }
  static constexpr llvm::StringLiteral checkpointNames[] = {
      "baseline", "large-operation", "ranking"};
  for (llvm::StringRef name : checkpointNames) {
    const json::Object *record = checkpointRecords->getObject(name);
    if (!record) {
      error = "ensemble checkpoint record is missing " + name.str();
      return false;
    }
    std::string relativePath;
    if (!getRequiredString(*record, "path", relativePath, error))
      return false;
    llvm::SmallString<256> checkpointPath(checkpointDirectory);
    if (llvm::sys::path::is_absolute(relativePath))
      checkpointPath = relativePath;
    else
      llvm::sys::path::append(checkpointPath, relativePath);
    json::Object checkpointJson;
    std::string checkpointSourceText;
    if (!readJsonObject(checkpointPath, checkpointJson, error,
                        &checkpointSourceText))
      return false;
    Checkpoint checkpoint;
    checkpoint.name = name.str();
    if (!requireExactString(checkpointJson, "schema", kFormalCheckpointSchema,
                            error) ||
        !requireExactString(checkpointJson, "quality_status", "exploratory",
                            error) ||
        !requireExactString(checkpointJson, "output_mode", "continuous",
                            error))
      return false;
    std::optional<int64_t> version = checkpointJson.getInteger("version");
    if (!version || *version != 2) {
      error = "checkpoint version is not 2";
      return false;
    }
    const json::Object *checkpointFeatureContract =
        checkpointJson.getObject("feature_contract");
    std::vector<std::string> checkpointFeatureNames;
    if (!checkpointFeatureContract ||
        !validateFeatureContract(*checkpointFeatureContract,
                                 kFormalFeatureContractId, error,
                                 &checkpointFeatureNames))
      return false;
    if (checkpointFeatureNames != featureNames) {
      error = "checkpoint feature names differ from the ensemble contract";
      return false;
    }
    const json::Object *checkpointShapeProtocol =
        checkpointJson.getObject("shape_protocol");
    if (!checkpointShapeProtocol ||
        !validateSupportedShapeArray(*checkpointShapeProtocol, error))
      return false;
    const json::Object *config = checkpointJson.getObject("config");
    if (!config || !requireExactString(*config, "shape_protocol",
                                       kFormalShapeProtocolId, error))
      return false;
    const json::Array *hidden = config->getArray("hidden_dimensions");
    if (!hidden || hidden->size() != 2 ||
        !hidden->front().getAsInteger() ||
        *hidden->front().getAsInteger() != 64 ||
        !hidden->back().getAsInteger() || *hidden->back().getAsInteger() != 32) {
      error = "checkpoint hidden dimensions are not [64, 32]";
      return false;
    }
    double ceiling = 0.0;
    if (!getRequiredNumber(*config, "mapper_ii_ceiling", ceiling, error) ||
        ceiling != 20.0) {
      error = "checkpoint mapper II ceiling is not 20";
      return false;
    }
    const json::Object *evidence = checkpointJson.getObject("training_evidence");
    if (!evidence || !requireExactBoolean(*evidence, "production_ready", false,
                                          error))
      return false;
    const json::Object *tensors = checkpointJson.getObject("tensors");
    if (!tensors || tensors->size() != 8) {
      error = "checkpoint tensor set is incomplete";
      return false;
    }
    if (!parseTensor(*tensors, "feature_mean", kFormalMapperFeatureWidth, 0,
                     true, checkpoint.featureMean, error) ||
        !parseTensor(*tensors, "feature_scale", kFormalMapperFeatureWidth, 0,
                     true, checkpoint.featureScale, error) ||
        !parseTensor(*tensors, "regressor.0.weight", 64,
                     kFormalMapperFeatureWidth, false, checkpoint.layer0Weight,
                     error) ||
        !parseTensor(*tensors, "regressor.0.bias", 64, 0, true,
                     checkpoint.layer0Bias, error) ||
        !parseTensor(*tensors, "regressor.2.weight", 32, 64, false,
                     checkpoint.layer2Weight, error) ||
        !parseTensor(*tensors, "regressor.2.bias", 32, 0, true,
                     checkpoint.layer2Bias, error) ||
        !parseTensor(*tensors, "regressor.4.weight", 1, 32, false,
                     checkpoint.layer4Weight, error) ||
        !parseTensor(*tensors, "regressor.4.bias", 1, 0, true,
                     checkpoint.layer4Bias, error))
      return false;
    for (float scale : checkpoint.featureScale) {
      if (!std::isfinite(scale) || scale <= 0.0F) {
        error = "checkpoint feature scale must be finite and positive";
        return false;
      }
    }
    checkpoints.push_back(std::move(checkpoint));
    checkpointSourceTexts.push_back({name.str(), std::move(checkpointSourceText)});
  }
  ensembleWeights = std::move(parsedWeights);
  loaded = true;
  return true;
}

bool FormalMax4MLPEnsemble::featureNamesMatch(
    llvm::ArrayRef<std::string> names) const {
  return names.size() == featureNames.size() &&
         std::equal(names.begin(), names.end(), featureNames.begin());
}

MLCostCacheResources FormalMax4MLPEnsemble::makeCacheResources(
    llvm::StringRef architectureText) const {
  MLCostCacheResources resources;
  resources.architectureText = architectureText.str();
  resources.ensembleText = ensembleSourceText;
  resources.checkpointTexts = checkpointSourceTexts;
  return resources;
}

static float gelu(float value) {
  constexpr float inverseSqrtTwo = 0.7071067811865475244F;
  return 0.5F * value *
         (1.0F + static_cast<float>(std::erf(
                       static_cast<double>(value * inverseSqrtTwo))));
}

static float softplus(float value) {
  if (value > 20.0F)
    return value;
  return std::log1p(std::exp(value));
}

static float linear(const std::vector<float> &weights,
                    const std::vector<float> &bias,
                    llvm::ArrayRef<float> input, unsigned output,
                    unsigned inputWidth) {
  float sum = bias[output];
  const size_t row = static_cast<size_t>(output) * inputWidth;
  for (unsigned index = 0; index < inputWidth; ++index)
    sum += weights[row + index] * input[index];
  return sum;
}

bool FormalMax4MLPEnsemble::predict(llvm::ArrayRef<double> features,
                                    double lowerBound,
                                    MLPEnsemblePrediction &prediction,
                                    std::string &error) const {
  if (!loaded) {
    error = "formal max-four ensemble is not loaded";
    return false;
  }
  if (features.size() != kFormalMapperFeatureWidth) {
    error = "feature vector width is not 156";
    return false;
  }
  if (!finiteNumber(lowerBound) || lowerBound <= 0.0 ||
      lowerBound > kMapperIICeiling) {
    error = "analytical lower bound must be in (0, 20]";
    return false;
  }

  SmallVector<float, kFormalMapperFeatureWidth> input;
  input.reserve(kFormalMapperFeatureWidth);
  for (double feature : features) {
    if (!finiteNumber(feature) || feature > std::numeric_limits<float>::max() ||
        feature < -std::numeric_limits<float>::max()) {
      error = "feature vector contains a non-finite or out-of-range value";
      return false;
    }
    input.push_back(static_cast<float>(feature));
  }

  float lower = static_cast<float>(lowerBound);
  SmallVector<float, 3> modelValues;
  for (const Checkpoint &checkpoint : checkpoints) {
    SmallVector<float, kFormalMapperFeatureWidth> normalized;
    normalized.reserve(kFormalMapperFeatureWidth);
    for (unsigned index = 0; index < kFormalMapperFeatureWidth; ++index)
      normalized.push_back((input[index] - checkpoint.featureMean[index]) /
                           checkpoint.featureScale[index]);
    SmallVector<float, 64> hidden0;
    hidden0.reserve(64);
    for (unsigned index = 0; index < 64; ++index)
      hidden0.push_back(gelu(linear(checkpoint.layer0Weight,
                                    checkpoint.layer0Bias, normalized, index,
                                    kFormalMapperFeatureWidth)));
    SmallVector<float, 32> hidden2;
    hidden2.reserve(32);
    for (unsigned index = 0; index < 32; ++index)
      hidden2.push_back(gelu(linear(checkpoint.layer2Weight,
                                    checkpoint.layer2Bias, hidden0, index, 64)));
    float raw = linear(checkpoint.layer4Weight, checkpoint.layer4Bias, hidden2,
                       0, 32);
    float value = lower + softplus(raw);
    value = std::min(value, kMapperIICeiling);
    if (!std::isfinite(value)) {
      error = "model produced a non-finite prediction";
      return false;
    }
    modelValues.push_back(value);
  }
  if (modelValues.size() != 3) {
    error = "formal ensemble does not contain exactly three regressors";
    return false;
  }
  double weightedMean =
      static_cast<double>(ensembleWeights[0]) * lower +
      static_cast<double>(ensembleWeights[1]) * modelValues[0] +
      static_cast<double>(ensembleWeights[2]) * modelValues[1] +
      static_cast<double>(ensembleWeights[3]) * modelValues[2];
  double mean = std::max(lowerBound, weightedMean);
  double center = (static_cast<double>(modelValues[0]) +
                   static_cast<double>(modelValues[1]) +
                   static_cast<double>(modelValues[2])) /
                  3.0;
  double variance = 0.0;
  for (float value : modelValues) {
    double delta = static_cast<double>(value) - center;
    variance += delta * delta;
  }
  variance /= 3.0;
  prediction.predictedII = mean;
  prediction.predictedIIStd = std::sqrt(std::max(0.0, variance));
  prediction.baselineII = modelValues[0];
  prediction.largeOperationII = modelValues[1];
  prediction.rankingII = modelValues[2];
  return true;
}

bool mlir::amoeba::neura::joint_scheduling::isFormalMax4MapperShape(
    int64_t rows, int64_t cols) {
  static constexpr std::pair<int64_t, int64_t> shapes[] = {
      {4, 4}, {4, 8}, {8, 4}, {4, 12}, {12, 4}, {4, 16}, {8, 8}, {16, 4}};
  return std::find(std::begin(shapes), std::end(shapes),
                   std::pair<int64_t, int64_t>{rows, cols}) !=
         std::end(shapes);
}

static bool sameFacts(const MLCostCacheFacts &left,
                      const MLCostCacheFacts &right) {
  return std::fabs(left.recMII - right.recMII) <= kFactsTolerance &&
         std::fabs(left.resMII - right.resMII) <= kFactsTolerance &&
         std::fabs(left.lowerBound - right.lowerBound) <= kFactsTolerance;
}

bool PersistentMLCostCache::load(llvm::StringRef path,
                                 llvm::StringRef modelSchema,
                                 llvm::StringRef featureContractId,
                                 llvm::StringRef architectureContract,
                                 const MLCostCacheResources &resources,
                                 std::string &error) {
  entries.clear();
  hitCount = 0;
  missCount = 0;
  if (path.empty())
    return true;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    if (buffer.getError() == std::errc::no_such_file_or_directory)
      return true;
    error = "cannot read ML cache " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  llvm::Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "cannot parse ML cache " + path.str() + ": " +
            llvm::toString(parsed.takeError());
    return false;
  }
  if (!rejectDigestIdentityKeys(*parsed, error, path))
    return false;
  const json::Object *root = parsed->getAsObject();
  if (!root || !requireExactString(*root, "schema", kMLCostCacheSchema,
                                   error) ||
      !requireExactString(*root, "model_schema", modelSchema, error) ||
      !requireExactString(*root, "feature_contract_id", featureContractId,
                          error) ||
      !requireExactString(*root, "architecture_contract", architectureContract,
                          error))
    return false;
  const json::Object *resourceContract = root->getObject("resource_contract");
  if (!resourceContract ||
      !requireExactString(*resourceContract, "architecture_text",
                          resources.architectureText, error) ||
      !requireExactString(*resourceContract, "ensemble_text",
                          resources.ensembleText, error))
    return false;
  const json::Array *rawCheckpointTexts =
      resourceContract->getArray("checkpoint_texts");
  if (!rawCheckpointTexts ||
      rawCheckpointTexts->size() != resources.checkpointTexts.size()) {
    error = "ML cache checkpoint resource contract does not match the loaded "
            "model";
    return false;
  }
  for (size_t index = 0; index < resources.checkpointTexts.size(); ++index) {
    const json::Object *checkpoint = (*rawCheckpointTexts)[index].getAsObject();
    if (!checkpoint ||
        !requireExactString(*checkpoint, "name",
                            resources.checkpointTexts[index].first, error) ||
        !requireExactString(*checkpoint, "text",
                            resources.checkpointTexts[index].second, error))
      return false;
  }
  const json::Array *rawEntries = root->getArray("entries");
  if (!rawEntries) {
    error = "ML cache has no entries array";
    return false;
  }
  for (const json::Value &rawEntry : *rawEntries) {
    const json::Object *object = rawEntry.getAsObject();
    if (!object) {
      error = "ML cache entry is not an object";
      return false;
    }
    MLCostCacheEntry entry;
    if (!getRequiredString(*object, "body_structural_text",
                           entry.key.bodyStructuralText, error) ||
        !getRequiredInteger(*object, "mapper_tile_rows",
                            entry.key.mapperTileRows, error) ||
        !getRequiredInteger(*object, "mapper_tile_cols",
                            entry.key.mapperTileCols, error) ||
        !getRequiredNumber(*object, "rec_mii", entry.facts.recMII, error) ||
        !getRequiredNumber(*object, "res_mii", entry.facts.resMII, error) ||
        !getRequiredNumber(*object, "lower_bound", entry.facts.lowerBound,
                           error) ||
        !getRequiredNumber(*object, "predicted_ii",
                           entry.prediction.predictedII, error) ||
        !getRequiredNumber(*object, "predicted_ii_std",
                           entry.prediction.predictedIIStd, error) ||
        !getRequiredNumber(*object, "baseline_ii", entry.prediction.baselineII,
                           error) ||
        !getRequiredNumber(*object, "large_operation_ii",
                           entry.prediction.largeOperationII, error) ||
        !getRequiredNumber(*object, "ranking_ii", entry.prediction.rankingII,
                           error))
      return false;
    if (entry.key.bodyStructuralText.empty() ||
        !isFormalMax4MapperShape(entry.key.mapperTileRows,
                                 entry.key.mapperTileCols) ||
        entry.facts.recMII <= 0.0 || entry.facts.resMII <= 0.0 ||
        entry.facts.lowerBound <= 0.0 || entry.prediction.predictedII <
            entry.facts.lowerBound ||
        entry.prediction.predictedII > kMapperIICeiling + kFactsTolerance ||
        entry.prediction.predictedIIStd < 0.0) {
      error = "ML cache contains an invalid entry";
      return false;
    }
    for (const MLCostCacheEntry &existing : entries) {
      if (existing.key.bodyStructuralText == entry.key.bodyStructuralText &&
          existing.key.mapperTileRows == entry.key.mapperTileRows &&
          existing.key.mapperTileCols == entry.key.mapperTileCols) {
        error = "ML cache contains duplicate structural keys";
        return false;
      }
    }
    entries.push_back(std::move(entry));
  }
  return true;
}

bool PersistentMLCostCache::lookup(const MLCostCacheKey &key,
                                   const MLCostCacheFacts &facts,
                                   MLPEnsemblePrediction &prediction) const {
  for (const MLCostCacheEntry &entry : entries) {
    if (entry.key.bodyStructuralText == key.bodyStructuralText &&
        entry.key.mapperTileRows == key.mapperTileRows &&
        entry.key.mapperTileCols == key.mapperTileCols &&
        sameFacts(entry.facts, facts)) {
      prediction = entry.prediction;
      ++hitCount;
      return true;
    }
  }
  ++missCount;
  return false;
}

void PersistentMLCostCache::insert(const MLCostCacheKey &key,
                                   const MLCostCacheFacts &facts,
                                   const MLPEnsemblePrediction &prediction) {
  for (MLCostCacheEntry &entry : entries) {
    if (entry.key.bodyStructuralText == key.bodyStructuralText &&
        entry.key.mapperTileRows == key.mapperTileRows &&
        entry.key.mapperTileCols == key.mapperTileCols) {
      // A body/shape key may only be rebound when its feature facts agree.
      // The pass rejects conflicting duplicate queries before reaching this
      // method; retaining the first result is therefore deterministic.
      if (sameFacts(entry.facts, facts))
        return;
      return;
    }
  }
  entries.push_back({key, facts, prediction});
}

bool PersistentMLCostCache::write(llvm::StringRef path,
                                  llvm::StringRef modelSchema,
                                  llvm::StringRef featureContractId,
                                  llvm::StringRef architectureContract,
                                  const MLCostCacheResources &resources,
                                  std::string &error) const {
  if (path.empty())
    return true;
  return writeAtomically(
      path,
      [&](llvm::raw_ostream &os) {
        json::Object root;
        root["schema"] = kMLCostCacheSchema.str();
        root["model_schema"] = modelSchema.str();
        root["feature_contract_id"] = featureContractId.str();
        root["architecture_contract"] = architectureContract.str();
        json::Array rawCheckpointTexts;
        for (const auto &checkpoint : resources.checkpointTexts) {
          rawCheckpointTexts.push_back(json::Object{
              {"name", checkpoint.first}, {"text", checkpoint.second}});
        }
        root["resource_contract"] = json::Object{
            {"architecture_text", resources.architectureText},
            {"ensemble_text", resources.ensembleText},
            {"checkpoint_texts", std::move(rawCheckpointTexts)}};
        json::Array rawEntries;
        for (const MLCostCacheEntry &entry : entries) {
          rawEntries.push_back(json::Object{
              {"body_structural_text", entry.key.bodyStructuralText},
              {"mapper_tile_rows", entry.key.mapperTileRows},
              {"mapper_tile_cols", entry.key.mapperTileCols},
              {"rec_mii", entry.facts.recMII},
              {"res_mii", entry.facts.resMII},
              {"lower_bound", entry.facts.lowerBound},
              {"predicted_ii", entry.prediction.predictedII},
              {"predicted_ii_std", entry.prediction.predictedIIStd},
              {"baseline_ii", entry.prediction.baselineII},
              {"large_operation_ii", entry.prediction.largeOperationII},
              {"ranking_ii", entry.prediction.rankingII}});
        }
        root["entries"] = std::move(rawEntries);
        os << json::Value(std::move(root));
        return true;
      },
      error);
}
