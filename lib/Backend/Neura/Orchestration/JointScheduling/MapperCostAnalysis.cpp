//===- MapperCostAnalysis.cpp ----------------------------------*- C++ -*-===//

#include "MapperCostAnalysis.h"

#include "MapperCounterBounds.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/mapping_util.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {

bool computeMapperAnalyticalFacts(mlir::Region &region, int64_t rows,
                                  int64_t columns, double &recMII,
                                  double &resMII, double &lowerBound,
                                  std::string &error) {
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

bool deriveStartupCyclesFromCppFeatureOutput(const llvm::json::Object &entry,
                                             double &startup,
                                             std::string &error) {
  const llvm::json::Array *rawTypes = entry.getArray("node_types");
  const llvm::json::Array *rawEdges = entry.getArray("edges");
  if (!rawTypes || !rawEdges || rawTypes->empty()) {
    error = "C++ feature output cannot derive startup without node_types and "
            "edges";
    return false;
  }
  std::vector<int64_t> types;
  types.reserve(rawTypes->size());
  for (const llvm::json::Value &value : *rawTypes) {
    std::optional<int64_t> type = value.getAsInteger();
    if (!type || *type < 0) {
      error = "C++ feature output node_types are invalid";
      return false;
    }
    types.push_back(*type);
  }
  std::vector<int64_t> depth(types.size(), 1);
  for (const llvm::json::Value &rawEdge : *rawEdges) {
    const llvm::json::Array *edge = rawEdge.getAsArray();
    if (!edge || edge->size() != 2 || !(*edge)[0].getAsInteger() ||
        !(*edge)[1].getAsInteger()) {
      error = "C++ feature output edges are invalid";
      return false;
    }
    int64_t source = *(*edge)[0].getAsInteger();
    int64_t target = *(*edge)[1].getAsInteger();
    if (source < 0 || target < 0 ||
        source >= static_cast<int64_t>(types.size()) ||
        target >= static_cast<int64_t>(types.size())) {
      error = "C++ feature output edge endpoint is out of range";
      return false;
    }
    // Route-expanded DFGs are emitted in source order. Feedback edges are
    // deliberately excluded from startup, matching the pinned frontend's
    // unit-latency semantic depth calculation.
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

} // namespace mlir::amoeba::neura::joint_scheduling
