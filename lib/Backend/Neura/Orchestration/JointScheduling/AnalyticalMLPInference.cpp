//===- AnalyticalMLPInference.cpp ------------------------------*- C++ -*-===//

#include "AnalyticalMLPInference.h"

#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/Orchestration/JointScheduling/MapperFeatureExtractor.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
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

static bool parseDirectFloatVector(const json::Value *raw, unsigned expected,
                                   std::vector<float> &values,
                                   std::string &error) {
  const json::Array *array = raw ? raw->getAsArray() : nullptr;
  if (!array || array->size() != expected) {
    error = "direct-model vector has the wrong width";
    return false;
  }
  values.clear();
  if (!appendTensorValues(*raw, values, error))
    return false;
  return values.size() == expected;
}

static bool parseDirectFloatMatrix(const json::Value *raw, unsigned rows,
                                   unsigned columns,
                                   std::vector<float> &values,
                                   std::string &error) {
  const json::Array *matrix = raw ? raw->getAsArray() : nullptr;
  if (!matrix || matrix->size() != rows) {
    error = "direct-model matrix has the wrong row count";
    return false;
  }
  for (const json::Value &rowValue : *matrix) {
    const json::Array *row = rowValue.getAsArray();
    if (!row || row->size() != columns) {
      error = "direct-model matrix has the wrong column count";
      return false;
    }
  }
  values.clear();
  if (!appendTensorValues(*raw, values, error))
    return false;
  return values.size() == static_cast<std::size_t>(rows) * columns;
}

static bool parseDirectStringArray(const json::Value *raw,
                                   std::vector<std::string> &values,
                                   std::string &error) {
  const json::Array *array = raw ? raw->getAsArray() : nullptr;
  if (!array) {
    error = "direct-model feature names are missing";
    return false;
  }
  values.clear();
  std::set<std::string> unique;
  for (const json::Value &entry : *array) {
    std::optional<llvm::StringRef> value = entry.getAsString();
    if (!value || value->empty() || !unique.insert(value->str()).second) {
      error = "direct-model feature names are missing or duplicated";
      return false;
    }
    values.push_back(value->str());
  }
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

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

bool validatePerCgra2x2RuntimeContract(
    double runtimeIICeiling, llvm::StringRef trainingArchitectureText,
    llvm::StringRef runtimeArchitectureText, std::string &error) {
  if (trainingArchitectureText.empty() || runtimeArchitectureText.empty()) {
    error = "training and runtime architecture text are required";
    return false;
  }
  if (runtimeIICeiling == kPerCgra2x2TrainingIICeiling) {
    if (trainingArchitectureText != runtimeArchitectureText) {
      error = "default direct-model inference requires exact training/runtime "
              "architecture text equality";
      return false;
    }
    return true;
  }
  if (runtimeIICeiling != kPerCgra2x2DiagnosticIICeiling) {
    error = "direct-model runtime II ceiling must be exactly 20 or diagnostic "
            "23";
    return false;
  }

  constexpr llvm::StringLiteral trainingLine = "ctrl_mem_items: 20";
  constexpr llvm::StringLiteral runtimeLine = "ctrl_mem_items: 23";
  const size_t position = trainingArchitectureText.find(trainingLine);
  if (position == llvm::StringRef::npos ||
      trainingArchitectureText.find(trainingLine, position + trainingLine.size()) !=
          llvm::StringRef::npos) {
    error = "diagnostic architecture must contain exactly one training "
            "ctrl_mem_items: 20 field";
    return false;
  }
  const size_t previousNewline =
      trainingArchitectureText.substr(0, position).rfind('\n');
  const size_t lineStart = previousNewline == llvm::StringRef::npos
                               ? 0
                               : previousNewline + 1;
  llvm::StringRef indentation = trainingArchitectureText.substr(
      lineStart, position - lineStart);
  const size_t lineEnd = position + trainingLine.size();
  if (llvm::any_of(indentation, [](char character) {
        return character != ' ' && character != '\t';
      }) ||
      (lineEnd < trainingArchitectureText.size() &&
       trainingArchitectureText[lineEnd] != '\r' &&
       trainingArchitectureText[lineEnd] != '\n')) {
    error = "diagnostic ctrl_mem_items change must target one complete YAML "
            "field line";
    return false;
  }
  std::string expectedRuntime = trainingArchitectureText.str();
  expectedRuntime.replace(position, trainingLine.size(), runtimeLine.data(),
                          runtimeLine.size());
  if (runtimeArchitectureText != expectedRuntime) {
    error = "diagnostic runtime architecture must differ only by the exact "
            "ctrl_mem_items: 20 to 23 field change";
    return false;
  }
  return true;
}

bool makePerCgra2x2DiagnosticOverrideMetadata(
    llvm::StringRef trainingArchitectureText,
    llvm::StringRef runtimeArchitectureText, json::Object &overrideMetadata,
    std::string &error) {
  error.clear();
  overrideMetadata.clear();
  if (!validatePerCgra2x2RuntimeContract(
          kPerCgra2x2DiagnosticIICeiling, trainingArchitectureText,
          runtimeArchitectureText, error))
    return false;
  overrideMetadata = json::Object{
      {"schema", kPerCgra2x2DiagnosticSchema.str()},
      {"training_ii_ceiling", kPerCgra2x2TrainingIICeiling},
      {"runtime_ii_ceiling", kPerCgra2x2DiagnosticIICeiling},
      {"extrapolation_enabled", true},
      {"formal", false},
      {"output_rule", kPerCgra2x2DiagnosticOutputRule.str()},
      {"training_architecture_exact_yaml_text",
       trainingArchitectureText.str()},
      {"runtime_architecture_exact_yaml_text", runtimeArchitectureText.str()}};
  return true;
}

bool validatePerCgra2x2DiagnosticMetadata(
    const json::Object &predictorMetadata,
    llvm::StringRef currentRuntimeArchitectureText, std::string &error) {
  error.clear();
  const json::Object *overrideMetadata =
      predictorMetadata.getObject("diagnostic_override");
  std::optional<double> trainingCeiling = overrideMetadata
                                              ? overrideMetadata->getNumber(
                                                    "training_ii_ceiling")
                                              : std::nullopt;
  std::optional<double> runtimeCeiling = overrideMetadata
                                             ? overrideMetadata->getNumber(
                                                   "runtime_ii_ceiling")
                                             : std::nullopt;
  std::optional<llvm::StringRef> trainingArchitecture =
      overrideMetadata
          ? overrideMetadata->getString(
                "training_architecture_exact_yaml_text")
          : std::nullopt;
  std::optional<llvm::StringRef> runtimeArchitecture =
      overrideMetadata
          ? overrideMetadata->getString("runtime_architecture_exact_yaml_text")
          : std::nullopt;
  std::string architectureError;
  if (!overrideMetadata || overrideMetadata->size() != 8 ||
      overrideMetadata->getString("schema") != kPerCgra2x2DiagnosticSchema ||
      !trainingCeiling ||
      *trainingCeiling != kPerCgra2x2TrainingIICeiling || !runtimeCeiling ||
      *runtimeCeiling != kPerCgra2x2DiagnosticIICeiling ||
      !overrideMetadata->getBoolean("extrapolation_enabled").value_or(false) ||
      overrideMetadata->getBoolean("formal").value_or(true) ||
      overrideMetadata->getString("output_rule") !=
          kPerCgra2x2DiagnosticOutputRule ||
      !predictorMetadata.getBoolean("diagnostic_only").value_or(false) ||
      predictorMetadata.getBoolean("formal").value_or(true) ||
      !trainingArchitecture || trainingArchitecture->empty() ||
      !runtimeArchitecture || runtimeArchitecture->empty() ||
      *runtimeArchitecture != currentRuntimeArchitectureText ||
      !validatePerCgra2x2RuntimeContract(
          kPerCgra2x2DiagnosticIICeiling,
          trainingArchitecture ? *trainingArchitecture : llvm::StringRef(),
          runtimeArchitecture ? *runtimeArchitecture : llvm::StringRef(),
          architectureError)) {
    error = architectureError.empty()
                ? "direct diagnostic override metadata is missing, forged, or "
                  "differs from the exact runtime architecture"
                : architectureError;
    return false;
  }
  return true;
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

bool PerCgra2x2DirectEnsemble::load(
    llvm::StringRef ensemblePath, llvm::StringRef expectedArchitectureText,
    std::string &error) {
  return load(ensemblePath, expectedArchitectureText,
              kPerCgra2x2TrainingIICeiling, error);
}

bool PerCgra2x2DirectEnsemble::load(
    llvm::StringRef ensemblePath, llvm::StringRef runtimeArchitectureText,
    double runtimeIICeiling, std::string &error) {
  loaded = false;
  modelNamespace.clear();
  featureContractId.clear();
  shapeProtocolId.clear();
  architectureText.clear();
  this->runtimeArchitectureText.clear();
  featureNames.clear();
  selectedFeatureNames.clear();
  selectedFeatureIndices.clear();
  members.clear();
  mapperIICeiling = 0.0F;
  this->runtimeIICeiling = 0.0F;
  if (ensemblePath.empty() || runtimeArchitectureText.empty()) {
    error = "direct ensemble path and expected architecture text are required";
    return false;
  }

  json::Object root;
  if (!readJsonObject(ensemblePath, root, error))
    return false;
  if (!requireExactString(root, "schema", kPerCgra2x2EnsembleSchema, error) ||
      !requireExactString(root, "model_namespace",
                          kPerCgra2x2ModelNamespace, error))
    return false;

  const json::Object *source = root.getObject("source_model");
  if (!source ||
      !requireExactString(*source, "repository",
                          "https://github.com/guosran/cgra-ii-predictor",
                          error) ||
      !requireExactString(*source, "branch", "orbit-2x2-predictor", error) ||
      !requireExactString(*source, "commit",
                          "3ade31806cb4c92e31888109f7c42b8a77e4cbce", error))
    return false;
  const json::Object *candidate = source->getObject("candidate_metadata");
  if (!candidate ||
      !requireExactString(
          *candidate, "promotion_status",
          "candidate_pending_amoeba_benchmark_overlap_audit", error) ||
      !requireExactString(*candidate, "shape_protocol_id",
                          kPerCgra2x2ShapeProtocolId, error) ||
      !requireExactBoolean(*candidate, "candidate_only", true, error) ||
      !requireExactBoolean(*candidate,
                           "amoeba_benchmark_overlap_audit_complete", false,
                           error) ||
      !requireExactBoolean(*candidate, "old_4x4_labels_reused", false, error) ||
      !requireExactBoolean(
          *candidate, "supports_whole_program_latency_or_throughput_claim",
          false, error))
    return false;
  int64_t rawFeatureCount = 0;
  int64_t selectedFeatureCount = 0;
  int64_t memberCount = 0;
  if (!getRequiredInteger(*candidate, "feature_count", rawFeatureCount,
                          error) ||
      !getRequiredInteger(*candidate, "effective_feature_count",
                          selectedFeatureCount, error) ||
      !getRequiredInteger(*candidate, "ensemble_member_count", memberCount,
                          error) ||
      rawFeatureCount != kPerCgra2x2MapperFeatureWidth ||
      selectedFeatureCount != 61 || memberCount != 4) {
    error = "candidate metadata does not match the 148/61/four-member model";
    return false;
  }

  const json::Object *featureContract = root.getObject("feature_contract");
  if (!featureContract ||
      !requireExactString(*featureContract, "contract_id",
                          kPerCgra2x2FeatureContractId, error) ||
      !requireExactString(
          *featureContract, "extractor",
          "cgra_ii_predictor.mapper_model:mapper_feature_vector", error) ||
      !requireExactString(*featureContract, "shape_protocol_id",
                          kPerCgra2x2ShapeProtocolId, error))
    return false;
  int64_t featureWidth = 0;
  if (!getRequiredInteger(*featureContract, "feature_width", featureWidth,
                          error) ||
      featureWidth != kPerCgra2x2MapperFeatureWidth ||
      !parseDirectStringArray(featureContract->get("feature_names"),
                              featureNames, error) ||
      featureNames.size() != kPerCgra2x2MapperFeatureWidth)
    return false;
  const auto &expectedNames =
      orbit::mapper_features::perCgra2x2MapperFeatureNames();
  if (!std::equal(featureNames.begin(), featureNames.end(),
                  expectedNames.begin(), expectedNames.end())) {
    error = "direct ensemble feature names differ from the C++ frontend";
    return false;
  }

  const json::Array *rawIndices =
      featureContract->getArray("selected_feature_indices");
  if (!rawIndices || rawIndices->size() != 61 ||
      !parseDirectStringArray(featureContract->get("selected_feature_names"),
                              selectedFeatureNames, error) ||
      selectedFeatureNames.size() != 61) {
    error = "direct ensemble selected feature contract must contain 61 entries";
    return false;
  }
  unsigned previousIndex = 0;
  for (unsigned position = 0; position < rawIndices->size(); ++position) {
    std::optional<int64_t> parsedIndex = (*rawIndices)[position].getAsInteger();
    if (!parsedIndex || *parsedIndex < 0 ||
        *parsedIndex >= static_cast<int64_t>(featureNames.size()) ||
        (position != 0 && static_cast<unsigned>(*parsedIndex) <= previousIndex)) {
      error = "direct ensemble feature indices must be increasing and in range";
      return false;
    }
    unsigned index = static_cast<unsigned>(*parsedIndex);
    if (selectedFeatureNames[position] != featureNames[index]) {
      error = "selected feature names do not match their ordered indices";
      return false;
    }
    selectedFeatureIndices.push_back(index);
    previousIndex = index;
  }

  const json::Object *shapeProtocol = root.getObject("shape_protocol");
  if (!shapeProtocol ||
      !requireExactString(*shapeProtocol, "protocol_id",
                          kPerCgra2x2ShapeProtocolId, error) ||
      !requireExactBoolean(*shapeProtocol, "orientation_equivalent", false,
                           error))
    return false;
  int64_t maxPhysicalCgras = 0;
  if (!getRequiredInteger(*shapeProtocol,
                          "maximum_physical_cgras_per_task",
                          maxPhysicalCgras, error) ||
      maxPhysicalCgras != 4)
    return false;
  static constexpr std::pair<int64_t, int64_t> expectedShapes[] = {
      {2, 2}, {2, 4}, {4, 2}, {2, 6},
      {6, 2}, {2, 8}, {8, 2}, {4, 4},
  };
  const json::Array *shapeArray =
      shapeProtocol->getArray("supported_mapper_tile_shapes");
  if (!shapeArray || shapeArray->size() != std::size(expectedShapes)) {
    error = "direct ensemble shape protocol must enumerate eight shapes";
    return false;
  }
  for (unsigned index = 0; index < std::size(expectedShapes); ++index) {
    const json::Object *shape = (*shapeArray)[index].getAsObject();
    int64_t rows = 0;
    int64_t columns = 0;
    if (!shape || !getRequiredInteger(*shape, "rows", rows, error) ||
        !getRequiredInteger(*shape, "cols", columns, error) ||
        std::pair<int64_t, int64_t>{rows, columns} != expectedShapes[index]) {
      error = "direct ensemble shape order differs from the pinned protocol";
      return false;
    }
  }

  const json::Object *architecture = root.getObject("architecture");
  std::string trainingArchitectureText;
  if (!architecture ||
      !requireExactString(*architecture, "name",
                          "AMOEBA_4x4_CGRA_2x2_Tiles", error) ||
      !getRequiredString(*architecture, "exact_yaml_text",
                         trainingArchitectureText, error) ||
      !validatePerCgra2x2RuntimeContract(
          runtimeIICeiling, trainingArchitectureText, runtimeArchitectureText,
          error)) {
    if (error.empty())
      error = "direct ensemble training/runtime architecture contract is invalid";
    return false;
  }

  const json::Object *network = root.getObject("network");
  if (!network ||
      !requireExactString(*network, "activation", "gelu_erf", error) ||
      !requireExactString(*network, "residual_activation", "softplus", error) ||
      !requireExactString(*network, "lower_bound_rule",
                          "max(rec_mii,res_mii)", error) ||
      !requireExactString(
          *network, "output_rule",
          "min(lower_bound + softplus(logit), mapper_ii_ceiling)", error) ||
      !requireExactString(*network, "ensemble_reduction", "arithmetic_mean",
                          error))
    return false;
  int64_t networkInputWidth = 0;
  int64_t selectedInputWidth = 0;
  double ceiling = 0.0;
  if (!getRequiredInteger(*network, "input_width", networkInputWidth, error) ||
      !getRequiredInteger(*network, "selected_input_width", selectedInputWidth,
                          error) ||
      !getRequiredNumber(*network, "mapper_ii_ceiling", ceiling, error) ||
      networkInputWidth != 148 || selectedInputWidth != 61 || ceiling != 20.0) {
    error = "direct ensemble network dimensions or II ceiling are invalid";
    return false;
  }
  const json::Array *hidden = network->getArray("hidden_dimensions");
  if (!hidden || hidden->size() != 2 || !(*hidden)[0].getAsInteger() ||
      *(*hidden)[0].getAsInteger() != 64 ||
      !(*hidden)[1].getAsInteger() || *(*hidden)[1].getAsInteger() != 32) {
    error = "direct ensemble hidden dimensions are not [64, 32]";
    return false;
  }

  const json::Array *rawMembers = root.getArray("members");
  if (!rawMembers || rawMembers->size() != 4) {
    error = "direct ensemble must contain exactly four members";
    return false;
  }
  static constexpr int64_t expectedSeeds[] = {17, 41, 113, 239};
  std::vector<Member> parsedMembers;
  parsedMembers.reserve(4);
  for (unsigned memberIndex = 0; memberIndex < 4; ++memberIndex) {
    const json::Object *rawMember = (*rawMembers)[memberIndex].getAsObject();
    if (!rawMember) {
      error = "direct ensemble member is not an object";
      return false;
    }
    int64_t actualIndex = -1;
    int64_t seed = -1;
    if (!getRequiredInteger(*rawMember, "member_index", actualIndex, error) ||
        !getRequiredInteger(*rawMember, "seed", seed, error) ||
        actualIndex != static_cast<int64_t>(memberIndex) ||
        seed != expectedSeeds[memberIndex]) {
      error = "direct ensemble member order or seed changed";
      return false;
    }
    Member member;
    member.seed = seed;
    if (!parseDirectFloatVector(rawMember->get("feature_mean"), 148,
                                member.featureMean, error) ||
        !parseDirectFloatVector(rawMember->get("feature_scale"), 148,
                                member.featureScale, error))
      return false;
    for (float scale : member.featureScale) {
      if (!std::isfinite(scale) || scale <= 0.0F) {
        error = "direct ensemble feature scales must be positive";
        return false;
      }
    }
    const json::Array *layers = rawMember->getArray("layers");
    if (!layers || layers->size() != 3) {
      error = "direct ensemble member must contain three linear layers";
      return false;
    }
    std::vector<float> *weights[] = {
        &member.layer0Weight, &member.layer1Weight, &member.layer2Weight};
    std::vector<float> *biases[] = {
        &member.layer0Bias, &member.layer1Bias, &member.layer2Bias};
    static constexpr unsigned inputWidths[] = {61, 64, 32};
    static constexpr unsigned outputWidths[] = {64, 32, 1};
    for (unsigned layerIndex = 0; layerIndex < 3; ++layerIndex) {
      const json::Object *layer = (*layers)[layerIndex].getAsObject();
      int64_t inputWidth = 0;
      int64_t outputWidth = 0;
      if (!layer ||
          !getRequiredInteger(*layer, "input_width", inputWidth, error) ||
          !getRequiredInteger(*layer, "output_width", outputWidth, error) ||
          inputWidth != inputWidths[layerIndex] ||
          outputWidth != outputWidths[layerIndex] ||
          !parseDirectFloatMatrix(layer->get("weights"), outputWidths[layerIndex],
                                  inputWidths[layerIndex], *weights[layerIndex],
                                  error) ||
          !parseDirectFloatVector(layer->get("bias"), outputWidths[layerIndex],
                                  *biases[layerIndex], error)) {
        error = "direct ensemble member layer has invalid dimensions or values";
        return false;
      }
    }
    parsedMembers.push_back(std::move(member));
  }

  modelNamespace = kPerCgra2x2ModelNamespace.str();
  featureContractId = kPerCgra2x2FeatureContractId.str();
  shapeProtocolId = kPerCgra2x2ShapeProtocolId.str();
  architectureText = std::move(trainingArchitectureText);
  this->runtimeArchitectureText = runtimeArchitectureText.str();
  members = std::move(parsedMembers);
  mapperIICeiling = static_cast<float>(ceiling);
  this->runtimeIICeiling = static_cast<float>(runtimeIICeiling);
  loaded = true;
  return true;
}

bool PerCgra2x2DirectEnsemble::featureNamesMatch(
    llvm::ArrayRef<std::string> names) const {
  return names.size() == featureNames.size() &&
         std::equal(names.begin(), names.end(), featureNames.begin());
}

bool PerCgra2x2DirectEnsemble::predict(
    llvm::ArrayRef<double> features, double recMii, double resMii,
    double lowerBound, DirectMapperIIPrediction &prediction,
    std::string &error) const {
  if (!loaded) {
    error = "per-CGRA 2x2 direct ensemble is not loaded";
    return false;
  }
  if (features.size() != kPerCgra2x2MapperFeatureWidth) {
    error = "per-CGRA 2x2 feature vector width is not 148";
    return false;
  }
  if (!finiteNumber(recMii) || !finiteNumber(resMii) ||
      !finiteNumber(lowerBound) || recMii < 0.0 || resMii < 0.0 ||
      lowerBound != std::max(recMii, resMii) ||
      lowerBound > runtimeIICeiling) {
    error = "direct-model lower bound must equal max(RecMII, ResMII) and fit "
            "the explicitly loaded runtime interval";
    return false;
  }
  SmallVector<float, kPerCgra2x2MapperFeatureWidth> input;
  input.reserve(kPerCgra2x2MapperFeatureWidth);
  for (double feature : features) {
    if (!finiteNumber(feature) || feature > std::numeric_limits<float>::max() ||
        feature < -std::numeric_limits<float>::max()) {
      error = "direct-model feature vector contains a non-finite or out-of-range value";
      return false;
    }
    input.push_back(static_cast<float>(feature));
  }

  std::vector<double> outputs;
  outputs.reserve(members.size());
  float lower = static_cast<float>(lowerBound);
  float sum = 0.0F;
  for (const Member &member : members) {
    SmallVector<float, 61> normalized;
    normalized.reserve(61);
    for (unsigned index : selectedFeatureIndices)
      normalized.push_back((input[index] - member.featureMean[index]) /
                           member.featureScale[index]);
    SmallVector<float, 64> hidden0;
    hidden0.reserve(64);
    for (unsigned index = 0; index < 64; ++index)
      hidden0.push_back(gelu(linear(member.layer0Weight, member.layer0Bias,
                                    normalized, index, 61)));
    SmallVector<float, 32> hidden1;
    hidden1.reserve(32);
    for (unsigned index = 0; index < 32; ++index)
      hidden1.push_back(gelu(linear(member.layer1Weight, member.layer1Bias,
                                    hidden0, index, 64)));
    float raw = linear(member.layer2Weight, member.layer2Bias, hidden1, 0, 32);
    float value = std::min(lower + softplus(raw), runtimeIICeiling);
    if (!std::isfinite(value) || value < lower) {
      error = "direct ensemble produced an invalid prediction";
      return false;
    }
    outputs.push_back(static_cast<double>(value));
    sum += value;
  }
  prediction.predictedII = static_cast<double>(sum / 4.0F);
  prediction.memberPredictions = std::move(outputs);
  return std::isfinite(prediction.predictedII);
}

bool mlir::amoeba::neura::joint_scheduling::isFormalMax4MapperShape(
    int64_t rows, int64_t cols) {
  static constexpr std::pair<int64_t, int64_t> shapes[] = {
      {4, 4}, {4, 8}, {8, 4}, {4, 12}, {12, 4}, {4, 16}, {8, 8}, {16, 4}};
  return std::find(std::begin(shapes), std::end(shapes),
                   std::pair<int64_t, int64_t>{rows, cols}) !=
         std::end(shapes);
}

bool mlir::amoeba::neura::joint_scheduling::isPerCgra2x2MapperShape(
    int64_t rows, int64_t cols) {
  static constexpr std::pair<int64_t, int64_t> shapes[] = {
      {2, 2}, {2, 4}, {4, 2}, {2, 6},
      {6, 2}, {2, 8}, {8, 2}, {4, 4}};
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
                                 double runtimeIICeiling,
                                 llvm::StringRef
                                     validatedTrainingArchitectureText,
                                 std::string &error) {
  entries.clear();
  hitCount = 0;
  missCount = 0;
  const bool directModel = modelSchema == kPerCgra2x2EnsembleSchema;
  if (directModel) {
    if (!validatePerCgra2x2RuntimeContract(
            runtimeIICeiling, validatedTrainingArchitectureText,
            resources.architectureText, error)) {
      error = "ML cache runtime architecture contract is invalid: " + error;
      return false;
    }
  } else if (runtimeIICeiling != kPerCgra2x2TrainingIICeiling ||
             !validatedTrainingArchitectureText.empty()) {
    error = "non-direct ML cache requires the training II ceiling and no "
            "direct-model architecture override";
    return false;
  }
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
                           entry.prediction.predictedIIStd, error))
      return false;
    if (!directModel &&
        (!getRequiredNumber(*object, "baseline_ii", entry.prediction.baselineII,
                            error) ||
         !getRequiredNumber(*object, "large_operation_ii",
                            entry.prediction.largeOperationII, error) ||
         !getRequiredNumber(*object, "ranking_ii", entry.prediction.rankingII,
                            error)))
      return false;
    if (directModel &&
        (object->get("baseline_ii") || object->get("large_operation_ii") ||
         object->get("ranking_ii"))) {
      error = "direct-model cache entry must not carry legacy member labels";
      return false;
    }
    const bool shapeAllowed =
        directModel ? isPerCgra2x2MapperShape(entry.key.mapperTileRows,
                                              entry.key.mapperTileCols)
                    : isFormalMax4MapperShape(entry.key.mapperTileRows,
                                              entry.key.mapperTileCols);
    if (entry.key.bodyStructuralText.empty() || !shapeAllowed ||
        entry.facts.recMII <= 0.0 || entry.facts.resMII <= 0.0 ||
        entry.facts.lowerBound <= 0.0 ||
        entry.prediction.predictedII < entry.facts.lowerBound ||
        entry.prediction.predictedII > runtimeIICeiling + kFactsTolerance ||
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
        root["resource_contract"] =
            json::Object{{"architecture_text", resources.architectureText},
                         {"ensemble_text", resources.ensembleText},
                         {"checkpoint_texts", std::move(rawCheckpointTexts)}};
        json::Array rawEntries;
        for (const MLCostCacheEntry &entry : entries) {
          json::Object rawEntry{
              {"body_structural_text", entry.key.bodyStructuralText},
              {"mapper_tile_rows", entry.key.mapperTileRows},
              {"mapper_tile_cols", entry.key.mapperTileCols},
              {"rec_mii", entry.facts.recMII},
              {"res_mii", entry.facts.resMII},
              {"lower_bound", entry.facts.lowerBound},
              {"predicted_ii", entry.prediction.predictedII},
              {"predicted_ii_std", entry.prediction.predictedIIStd}};
          if (modelSchema != kPerCgra2x2EnsembleSchema) {
            rawEntry["baseline_ii"] = entry.prediction.baselineII;
            rawEntry["large_operation_ii"] = entry.prediction.largeOperationII;
            rawEntry["ranking_ii"] = entry.prediction.rankingII;
          }
          rawEntries.push_back(std::move(rawEntry));
        }
        root["entries"] = std::move(rawEntries);
        os << json::Value(std::move(root));
        return true;
      },
      error);
}
