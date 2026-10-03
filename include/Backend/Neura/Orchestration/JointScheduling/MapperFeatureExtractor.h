//===- MapperFeatureExtractor.h -----------------------------------------===//
// C++ route-expanded DFG and formal-max4-nohash-v2 feature contract.
//===----------------------------------------------------------------------===//

#ifndef ORBIT_CPP_MAPPER_FEATURE_EXTRACTOR_H
#define ORBIT_CPP_MAPPER_FEATURE_EXTRACTOR_H

#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace orbit {
namespace mapper_features {

constexpr std::size_t kFeatureWidth = 156;
constexpr std::size_t kNodeFeatureWidth = 14;

using Edge = std::pair<int, int>;
using NodeFeature = std::array<double, kNodeFeatureWidth>;

struct RouteExpandedGraph {
  std::vector<int> nodeTypes;
  std::vector<NodeFeature> nodeFeatures;
  std::vector<Edge> edges;
  std::vector<Edge> semanticEdges;
};

struct FeatureVector {
  std::array<double, kFeatureWidth> values{};
};

// The returned names are in the exact order consumed by the 156-input model.
const std::array<std::string, kFeatureWidth> &mapperFeatureNames();

// Parse compiler-emitted Neura route-expanded text.  When requireExpanded is
// true, unsupported operation kinds, missing movement edges, and semantic
// edges that bypass data_mov are rejected fail-closed.
bool parseRouteExpandedDFG(llvm::StringRef text, RouteExpandedGraph &graph,
                           std::string &error,
                           bool requireExpanded = true);

// Compute the formal-max4-nohash-v2 feature vector.  The lower bound must be
// exactly max(recMii, resMii), and rows/columns must be one of the 16 complete
// fixed-orientation shapes on the 4x4 physical fabric.
bool computeMapperFeatures(const RouteExpandedGraph &graph, int rows,
                           int columns, double recMii, double resMii,
                           double lowerBound, FeatureVector &features,
                           std::string &error);

} // namespace mapper_features
} // namespace orbit

#endif // ORBIT_CPP_MAPPER_FEATURE_EXTRACTOR_H
