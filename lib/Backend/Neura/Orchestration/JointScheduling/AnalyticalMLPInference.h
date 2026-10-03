//===- AnalyticalMLPInference.h --------------------------------*- C++ -*-===//
//
// A small, fail-closed reader/evaluator for the formal max-four exploratory
// mapper model.  The feature frontend is intentionally outside this class:
// it supplies the 156 values produced by the pinned mapper feature contract.
// This keeps Python out of the production inference path while allowing the
// C++ MLIR pass to consume the same feature contract as the reference adapter.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_ANALYTICAL_MLP_INFERENCE_H
#define AMOEBA_ANALYTICAL_MLP_INFERENCE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

inline constexpr unsigned kFormalMapperFeatureWidth = 156;
inline constexpr llvm::StringLiteral kFormalFeatureContractId =
    "cgra-ii-pre-mapper-features-156-formal-max4-v2";
inline constexpr llvm::StringLiteral kFormalFeatureExtractor =
    "cgra_ii_predictor.mapper_model:mapper_feature_vector";
inline constexpr llvm::StringLiteral kFormalEnsembleSchema =
    "cgra-ii-formal-max4-ensemble-v2-nohash";
inline constexpr llvm::StringLiteral kFormalCheckpointSchema =
    "cgra-ii-formal-max4-checkpoint-v2-nohash";
inline constexpr llvm::StringLiteral kFormalShapeProtocolId =
    "amoeba-static-rectangles-up-to-four-cgras-formal";

struct MLPEnsemblePrediction {
  double predictedII = 0.0;
  double predictedIIStd = 0.0;
  double baselineII = 0.0;
  double largeOperationII = 0.0;
  double rankingII = 0.0;
};

// Cache identity is deliberately structural and human-readable.  It never
// contains a task name, trip count, digest, or SHA identity.  RecMII/ResMII
// and the lower bound are stored with an entry as a consistency guard because
// they are inputs to the feature vector; they do not form the identity.
struct MLCostCacheKey {
  std::string bodyStructuralText;
  int64_t mapperTileRows = 0;
  int64_t mapperTileCols = 0;
};

struct MLCostCacheFacts {
  double recMII = 0.0;
  double resMII = 0.0;
  double lowerBound = 0.0;
};

struct MLCostCacheEntry {
  MLCostCacheKey key;
  MLCostCacheFacts facts;
  MLPEnsemblePrediction prediction;
};

// Cache resources are the exact source texts that define a prediction
// contract.  They are stored and compared verbatim so a changed architecture
// YAML or model/checkpoint file cannot reuse an old structural prediction.
// No path, task name, digest, or SHA identity participates in this contract.
struct MLCostCacheResources {
  std::string architectureText;
  std::string ensembleText;
  std::vector<std::pair<std::string, std::string>> checkpointTexts;
};

class FormalMax4MLPEnsemble {
public:
  FormalMax4MLPEnsemble() = default;

  // Loads and validates the exact exploratory no-SHA ensemble and its three
  // JSON checkpoints.  The requested architecture contract is a caller-owned
  // name (normally the Neura architecture schema plus protocol ID); it is
  // retained in the loaded contract and must match cache metadata.
  bool load(llvm::StringRef ensemblePath, llvm::StringRef checkpointDirectory,
            llvm::StringRef architectureContract, std::string &error);

  bool predict(llvm::ArrayRef<double> features, double lowerBound,
               MLPEnsemblePrediction &prediction, std::string &error) const;

  llvm::StringRef getModelSchema() const { return modelSchema; }
  llvm::StringRef getFeatureContractId() const { return featureContractId; }
  llvm::StringRef getFeatureExtractor() const { return featureExtractor; }
  llvm::StringRef getShapeProtocolId() const { return shapeProtocolId; }
  llvm::StringRef getArchitectureContract() const {
    return architectureContract;
  }
  llvm::StringRef getModelName() const { return modelName; }
  bool isLoaded() const { return loaded; }
  bool featureNamesMatch(llvm::ArrayRef<std::string> names) const;

  MLCostCacheResources makeCacheResources(
      llvm::StringRef architectureText) const;

private:
  struct Checkpoint {
    std::string name;
    std::vector<float> featureMean;
    std::vector<float> featureScale;
    std::vector<float> layer0Weight;
    std::vector<float> layer0Bias;
    std::vector<float> layer2Weight;
    std::vector<float> layer2Bias;
    std::vector<float> layer4Weight;
    std::vector<float> layer4Bias;
  };

  std::string modelSchema;
  std::string featureContractId;
  std::string featureExtractor;
  std::string shapeProtocolId;
  std::string architectureContract;
  std::string ensembleSourceText;
  std::vector<std::pair<std::string, std::string>> checkpointSourceTexts;
  std::string modelName = "formal-max4-nohash-v2";
  std::vector<std::string> featureNames;
  std::vector<Checkpoint> checkpoints;
  std::vector<float> ensembleWeights;
  bool loaded = false;
};

// A persistent cache is optional at the command-line boundary, but all
// callers get an in-memory cache even when no file was requested.  Loading an
// absent file is an empty-cache operation; malformed or mismatched files fail
// closed rather than silently mixing model contracts.
class PersistentMLCostCache {
public:
  bool load(llvm::StringRef path, llvm::StringRef modelSchema,
            llvm::StringRef featureContractId,
            llvm::StringRef architectureContract,
            const MLCostCacheResources &resources, std::string &error);

  bool lookup(const MLCostCacheKey &key, const MLCostCacheFacts &facts,
              MLPEnsemblePrediction &prediction) const;

  void insert(const MLCostCacheKey &key, const MLCostCacheFacts &facts,
              const MLPEnsemblePrediction &prediction);

  bool write(llvm::StringRef path, llvm::StringRef modelSchema,
             llvm::StringRef featureContractId,
             llvm::StringRef architectureContract,
             const MLCostCacheResources &resources,
             std::string &error) const;

  uint64_t getHitCount() const { return hitCount; }
  uint64_t getMissCount() const { return missCount; }
  size_t size() const { return entries.size(); }

private:
  std::vector<MLCostCacheEntry> entries;
  mutable uint64_t hitCount = 0;
  mutable uint64_t missCount = 0;
};

bool isFormalMax4MapperShape(int64_t rows, int64_t cols);

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_MLP_INFERENCE_H
