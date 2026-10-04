//===- MapperFeatureExtractor.cpp ---------------------------------------===//
// C++ route-expanded DFG and formal-max4-nohash-v2 feature contract.
//===----------------------------------------------------------------------===//

#include "Backend/Neura/Orchestration/JointScheduling/MapperFeatureExtractor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace orbit {
namespace mapper_features {
namespace {

constexpr const char *kOperationTypes[] = {
    "<pad>",          "<unknown>",       "<pe>",
    "constant",       "grant_once",      "grant_always",
    "grant_predicate", "loop_control",   "phi",
    "phi_start",      "add",             "sub",
    "mul",             "div",             "rem",
    "fadd",            "fsub",            "fmul",
    "fdiv",            "fmul_fadd",      "fadd_fadd",
    "vfmul",           "or",              "and",
    "xor",             "not",             "shl",
    "icmp",            "fcmp",            "sel",
    "cast",            "sext",            "zext",
    "alloca",          "gep",             "load",
    "store",           "memset",          "load_indexed",
    "store_indexed",   "data_mov",        "ctrl_mov",
    "reserve",         "yield",           "return",
    "counter",         "fneg",            "fmax",
    "fmin",            "gather",          "br",
    "return_void",     "return_value",    "vmul",
    "vadd",            "vfadd",           "vector_reduce_add",
    "mul_add",         "extract_predicate", "true_steer",
    "false_steer",     "carry",           "merge",
    "invariant",       "fused_op",
};

constexpr std::size_t kOperationTypeCount =
    sizeof(kOperationTypes) / sizeof(kOperationTypes[0]);

constexpr const char *kTransparentOperations[] = {
    "data_mov", "ctrl_mov", "reserve", "yield",
};

constexpr const char *kMemoryOperations[] = {
    "load", "store", "memset", "load_indexed", "store_indexed", "gather",
};

constexpr const char *kPointerOperations[] = {"alloca", "gep"};

constexpr const char *kControlOperations[] = {
    "grant_predicate", "loop_control", "phi", "phi_start", "not", "icmp",
    "fcmp", "sel", "counter", "extract_predicate", "true_steer",
    "false_steer", "carry", "merge", "invariant",
};

constexpr std::array<std::pair<int, int>, 16> kMapperShapes = {{
    {4, 4},   {4, 8},   {8, 4},   {4, 12},  {12, 4},  {4, 16},
    {8, 8},   {16, 4},  {8, 12},  {12, 8},  {8, 16},  {16, 8},
    {12, 12}, {12, 16}, {16, 12}, {16, 16},
}};

constexpr std::array<std::pair<int, int>, 8> kPerCgra2x2MapperShapes = {{
    {2, 2}, {2, 4}, {4, 2}, {2, 6},
    {6, 2}, {2, 8}, {8, 2}, {4, 4},
}};

bool inSet(llvm::StringRef value, const char *const *values,
           std::size_t count) {
  for (std::size_t index = 0; index < count; ++index)
    if (value == values[index])
      return true;
  return false;
}

bool isTransparent(llvm::StringRef value) {
  return inSet(value, kTransparentOperations,
               sizeof(kTransparentOperations) /
                   sizeof(kTransparentOperations[0]));
}

bool isMemory(llvm::StringRef value) {
  return inSet(value, kMemoryOperations,
               sizeof(kMemoryOperations) / sizeof(kMemoryOperations[0]));
}

bool isPointer(llvm::StringRef value) {
  return inSet(value, kPointerOperations,
               sizeof(kPointerOperations) / sizeof(kPointerOperations[0])) ||
         value == "load";
}

bool isControl(llvm::StringRef value) {
  return inSet(value, kControlOperations,
               sizeof(kControlOperations) / sizeof(kControlOperations[0]));
}

int operationId(llvm::StringRef value) {
  for (std::size_t index = 0; index < kOperationTypeCount; ++index)
    if (value == kOperationTypes[index])
      return static_cast<int>(index);
  return 1; // route-expanded <unknown>
}

bool isSupportedOperation(llvm::StringRef value) {
  if (value == "kernel")
    return true;
  for (std::size_t index = 0; index < kOperationTypeCount; ++index)
    if (value == kOperationTypes[index])
      return true;
  return false;
}

std::string replaceDots(std::string value) {
  std::replace(value.begin(), value.end(), '.', '_');
  return value;
}

bool parseResultLine(const std::string &line, std::string &value,
                     std::string &expression) {
  static const std::regex pattern(R"(^\s*(%[A-Za-z0-9_]+)\s*=\s*(.*)$)");
  std::smatch match;
  if (!std::regex_match(line, match, pattern))
    return false;
  value = match[1].str();
  expression = match[2].str();
  return true;
}

bool extractKind(const std::string &expression, std::string &kind) {
  static const std::regex pattern(R"(neura\.([a-z_][a-z0-9_.]*))");
  std::smatch match;
  if (!std::regex_search(expression, match, pattern))
    return false;
  kind = replaceDots(match[1].str());
  return true;
}

std::vector<std::string> operandsOf(const std::string &text) {
  static const std::regex pattern(R"(%[A-Za-z0-9_]+)");
  std::vector<std::string> operands;
  for (std::sregex_iterator it(text.begin(), text.end(), pattern), end;
       it != end; ++it)
    operands.push_back(it->str());
  return operands;
}

bool parseEffectLine(const std::string &line, std::string &kind) {
  static const std::regex pattern(
      R"re(^\s*"?neura\.([a-z_][a-z0-9_.]*)"?(?:\s|\()(.*)$)re");
  std::smatch match;
  if (!std::regex_match(line, match, pattern))
    return false;
  kind = replaceDots(match[1].str());
  return true;
}

bool parseControlMove(const std::string &line, std::string &source,
                      std::string &target) {
  static const std::regex pattern(
      R"(neura\.ctrl_mov"?\s+(%[A-Za-z0-9_]+)\s*->\s*(%[A-Za-z0-9_]+))");
  std::smatch match;
  if (!std::regex_search(line, match, pattern))
    return false;
  source = match[1].str();
  target = match[2].str();
  return true;
}

bool isReturnLine(const std::string &line, std::string &rest) {
  static const std::regex pattern(R"(^\s*(?:func\.)?return\b(.*)$)");
  std::smatch match;
  if (!std::regex_match(line, match, pattern))
    return false;
  rest = match[1].str();
  return true;
}

std::vector<std::string>
meaningfulRoots(const std::string &value,
               const std::map<std::string, std::string> &kinds,
               const std::map<std::string, std::vector<std::string>> &operands,
               std::set<std::string> seen = {}) {
  auto kind = kinds.find(value);
  if (kind == kinds.end())
    return {};
  if (!isTransparent(kind->second))
    return {value};
  if (!seen.insert(value).second)
    return {};
  std::vector<std::string> roots;
  auto foundOperands = operands.find(value);
  if (foundOperands == operands.end())
    return roots;
  for (const std::string &parent : foundOperands->second) {
    std::vector<std::string> parentRoots =
        meaningfulRoots(parent, kinds, operands, seen);
    roots.insert(roots.end(), parentRoots.begin(), parentRoots.end());
  }
  return roots;
}

struct TopologySummary {
  int depth = 0;
  int width = 0;
  int sources = 0;
  int sinks = 0;
  int maxIndegree = 0;
};

TopologySummary forwardTopology(
    int nodeCount, const std::vector<Edge> &edges) {
  std::vector<std::vector<int>> parents(static_cast<std::size_t>(nodeCount));
  std::vector<std::vector<int>> children(static_cast<std::size_t>(nodeCount));
  for (Edge edge : edges) {
    if (edge.first < edge.second) {
      parents[edge.second].push_back(edge.first);
      children[edge.first].push_back(edge.second);
    }
  }
  std::vector<int> levels(static_cast<std::size_t>(nodeCount), 1);
  for (int node = 0; node < nodeCount; ++node) {
    for (int parent : parents[node])
      levels[node] = std::max(levels[node], levels[parent] + 1);
  }
  const int depth = *std::max_element(levels.begin(), levels.end());
  std::vector<int> widths(static_cast<std::size_t>(depth), 0);
  for (int level : levels)
    ++widths[level - 1];
  TopologySummary summary;
  summary.depth = depth;
  summary.width = *std::max_element(widths.begin(), widths.end());
  for (const auto &values : parents)
    summary.sources += values.empty();
  for (const auto &values : children)
    summary.sinks += values.empty();
  for (const auto &values : parents)
    summary.maxIndegree = std::max(summary.maxIndegree,
                                   static_cast<int>(values.size()));
  return summary;
}

bool containsEdge(const std::vector<Edge> &edges, Edge wanted) {
  return std::find(edges.begin(), edges.end(), wanted) != edges.end();
}

} // namespace

template <std::size_t Width, std::size_t ShapeCount>
std::array<std::string, Width> buildMapperFeatureNames(
    const std::array<std::pair<int, int>, ShapeCount> &shapes) {
    std::array<std::string, Width> result{};
    std::size_t index = 0;
    for (const char *operation : kOperationTypes)
      result[index++] = std::string("log_count_") + operation;
    constexpr const char *kNodeNames[] = {
        "normalized_indegree", "normalized_outdegree", "normalized_asap",
        "normalized_reverse_depth", "normalized_slack",
        "normalized_position", "is_source", "is_sink", "is_memory",
        "is_pointer", "is_control", "is_materialized", "is_data_movement",
        "is_recurrence_cycle",
    };
    for (const char *name : kNodeNames)
      result[index++] = std::string("mean_") + name;
    for (const char *name : kNodeNames)
      result[index++] = std::string("max_") + name;
    constexpr const char *kGraphNames[] = {
        "log_node_count", "log_raw_edge_count", "log_semantic_edge_count",
    };
    for (const char *name : kGraphNames)
      result[index++] = name;
    constexpr const char *kTopologyNames[] = {
        "log_materialized_node_count", "log_movement_node_count",
        "log_forward_depth", "log_semantic_forward_depth",
        "log_max_forward_width", "normalized_max_forward_width",
        "log_source_count", "log_sink_count", "log_max_indegree",
        "log_max_outdegree", "raw_edge_density", "semantic_edge_density",
        "log_recurrence_node_count",
    };
    for (const char *name : kTopologyNames)
      result[index++] = name;
    constexpr const char *kContextNames[] = {
        "normalized_rows", "normalized_columns", "normalized_tiles",
        "log_aspect_ratio", "normalized_links", "normalized_bisection_links",
        "normalized_rec_mii", "normalized_res_mii", "normalized_lower_bound",
    };
    for (const char *name : kContextNames)
      result[index++] = name;
    for (auto shape : shapes)
      result[index++] = "shape_" + std::to_string(shape.first) + "x" +
                        std::to_string(shape.second);
    constexpr const char *kContextTail[] = {
        "log_nodes_per_tile", "log_materialized_nodes_per_tile",
        "log_raw_edges_per_link", "log_semantic_edges_per_link",
        "forward_depth_per_max_dimension", "forward_width_per_tile",
    };
    for (const char *name : kContextTail)
      result[index++] = name;
    constexpr const char *kPressureNames[] = {
        "mapper_bucket_width_per_tile", "mapper_bucket_width_per_tile_ii",
        "recurrence_nodes_per_tile", "fanin_collision_pressure",
        "fanout_collision_pressure", "route_edge_pressure",
    };
    for (const char *name : kPressureNames)
      result[index++] = name;
    constexpr const char *kTraversalNames[] = {
        "forward_depth_per_rows", "forward_depth_per_columns",
        "semantic_depth_per_rows", "semantic_depth_per_columns",
        "mapper_bucket_width_per_rows", "mapper_bucket_width_per_columns",
        "mapper_bucket_degree_pressure",
        "mapper_bucket_fanin_collision_pressure",
        "mapper_bucket_fanout_collision_pressure",
        "mapper_bucket_quadratic_occupancy",
    };
    for (const char *name : kTraversalNames)
      result[index++] = name;
    if (index != Width)
      std::abort();
    return result;
}

const std::array<std::string, kFeatureWidth> &mapperFeatureNames() {
  static const std::array<std::string, kFeatureWidth> names =
      buildMapperFeatureNames<kFeatureWidth>(kMapperShapes);
  return names;
}

const std::array<std::string, kPerCgra2x2FeatureWidth> &
perCgra2x2MapperFeatureNames() {
  static const std::array<std::string, kPerCgra2x2FeatureWidth> names =
      buildMapperFeatureNames<kPerCgra2x2FeatureWidth>(
          kPerCgra2x2MapperShapes);
  return names;
}

bool parseRouteExpandedDFG(llvm::StringRef text, RouteExpandedGraph &graph,
                           std::string &error, bool requireExpanded) {
  std::vector<std::string> order;
  std::map<std::string, std::string> kinds;
  std::map<std::string, std::vector<std::string>> operands;
  std::vector<std::pair<std::string, std::string>> controlMoves;
  std::set<std::string> observedKinds;
  int effectIndex = 0;
  const std::string source = text.str();
  static const std::regex kernelPattern(R"(\bneura\.kernel\b)");
  const bool hasKernelRegion = std::regex_search(source, kernelPattern);

  std::size_t lineStart = 0;
  while (lineStart <= source.size()) {
    const std::size_t lineEnd = source.find('\n', lineStart);
    std::string line = source.substr(
        lineStart, lineEnd == std::string::npos ? std::string::npos
                                                : lineEnd - lineStart);
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    std::string value;
    std::string expression;
    if (parseResultLine(line, value, expression)) {
      std::string kind;
      if (extractKind(expression, kind)) {
        if (!isSupportedOperation(kind) && kind != "kernel") {
          error = "unsupported route-expanded Neura operation: " + kind;
          return false;
        }
        if (!kinds.emplace(value, kind).second) {
          error = "Neura DFG repeats an SSA value: " + value;
          return false;
        }
        operands.emplace(value, operandsOf(expression));
        order.push_back(value);
        observedKinds.insert(kind);
      }
    } else {
      std::string moveSource;
      std::string moveTarget;
      if (parseControlMove(line, moveSource, moveTarget))
        controlMoves.emplace_back(moveSource, moveTarget);

      std::string effectKind;
      if (parseEffectLine(line, effectKind) && effectKind != "kernel") {
        if (!isSupportedOperation(effectKind)) {
          error = "unsupported route-expanded Neura operation: " + effectKind;
          return false;
        }
        value = "%__graph_effect_" + std::to_string(effectIndex++);
        kinds.emplace(value, effectKind);
        operands.emplace(value, operandsOf(line));
        order.push_back(value);
        observedKinds.insert(effectKind);
      }

      std::string returnOperands;
      if (isReturnLine(line, returnOperands) && !hasKernelRegion) {
        value = "%__graph_effect_" + std::to_string(effectIndex++);
        kinds.emplace(value, "return");
        operands.emplace(value, operandsOf(returnOperands));
        order.push_back(value);
        observedKinds.insert("return");
      }
    }
    if (lineEnd == std::string::npos)
      break;
    lineStart = lineEnd + 1;
  }

  if (order.empty()) {
    error = "Neura DFG contains no operations";
    return false;
  }
  for (const std::string &kind : observedKinds) {
    if (!isSupportedOperation(kind)) {
      error = "Neura DFG contains operations outside the 156-feature contract: " +
              kind;
      return false;
    }
  }

  std::map<std::string, int> index;
  for (std::size_t position = 0; position < order.size(); ++position)
    index.emplace(order[position], static_cast<int>(position));

  std::set<Edge> rawEdgeSet;
  for (const std::string &target : order) {
    auto targetIndex = index.find(target);
    auto foundOperands = operands.find(target);
    if (targetIndex == index.end() || foundOperands == operands.end())
      continue;
    for (const std::string &operand : foundOperands->second) {
      auto sourceIndex = index.find(operand);
      if (sourceIndex != index.end() && sourceIndex->second != targetIndex->second)
        rawEdgeSet.emplace(sourceIndex->second, targetIndex->second);
    }
  }

  std::set<Edge> controlEdgeSet;
  for (const auto &move : controlMoves) {
    auto sourceIndex = index.find(move.first);
    auto targetIndex = index.find(move.second);
    if (sourceIndex != index.end() && targetIndex != index.end() &&
        sourceIndex->second != targetIndex->second)
      controlEdgeSet.emplace(sourceIndex->second, targetIndex->second);
  }
  rawEdgeSet.insert(controlEdgeSet.begin(), controlEdgeSet.end());

  std::set<std::string> materialized;
  for (const std::string &value : order)
    if (!isTransparent(kinds[value]))
      materialized.insert(value);

  std::set<Edge> semanticEdgeSet;
  for (const std::string &target : materialized) {
    auto foundOperands = operands.find(target);
    if (foundOperands == operands.end())
      continue;
    auto targetIndex = index.find(target);
    for (const std::string &operand : foundOperands->second) {
      for (const std::string &root : meaningfulRoots(operand, kinds, operands)) {
        auto sourceIndex = index.find(root);
        if (sourceIndex != index.end() && targetIndex != index.end() &&
            sourceIndex->second != targetIndex->second)
          semanticEdgeSet.emplace(sourceIndex->second, targetIndex->second);
      }
    }
  }
  for (const auto &move : controlMoves) {
    std::vector<std::string> roots =
        meaningfulRoots(move.first, kinds, operands);
    for (const std::string &reserve : {move.second}) {
      for (const std::string &target : materialized) {
        auto foundOperands = operands.find(target);
        if (foundOperands == operands.end() ||
            std::find(foundOperands->second.begin(), foundOperands->second.end(),
                      reserve) == foundOperands->second.end())
          continue;
        for (const std::string &root : roots) {
          auto sourceIndex = index.find(root);
          auto targetIndex = index.find(target);
          if (sourceIndex != index.end() && targetIndex != index.end() &&
              sourceIndex->second != targetIndex->second)
            semanticEdgeSet.emplace(sourceIndex->second, targetIndex->second);
        }
      }
    }
  }

  graph.nodeTypes.clear();
  graph.nodeFeatures.clear();
  graph.edges.assign(rawEdgeSet.begin(), rawEdgeSet.end());
  graph.semanticEdges.assign(semanticEdgeSet.begin(), semanticEdgeSet.end());
  for (const std::string &value : order)
    graph.nodeTypes.push_back(operationId(kinds[value]));

  const int nodeCount = static_cast<int>(order.size());
  std::vector<Edge> forwardEdges;
  for (Edge edge : graph.edges)
    if (!controlEdgeSet.count(edge) && edge.first < edge.second)
      forwardEdges.push_back(edge);
  std::vector<std::vector<int>> parents(static_cast<std::size_t>(nodeCount));
  std::vector<std::vector<int>> children(static_cast<std::size_t>(nodeCount));
  for (Edge edge : forwardEdges) {
    parents[edge.second].push_back(edge.first);
    children[edge.first].push_back(edge.second);
  }
  std::vector<int> asap(static_cast<std::size_t>(nodeCount), 1);
  for (int node = 0; node < nodeCount; ++node)
    for (int parent : parents[node])
      asap[node] = std::max(asap[node], asap[parent] + 1);
  std::vector<int> reverseDepth(static_cast<std::size_t>(nodeCount), 1);
  for (int node = nodeCount - 1; node >= 0; --node)
    for (int child : children[node])
      reverseDepth[node] = std::max(reverseDepth[node], reverseDepth[child] + 1);
  const int makespan = *std::max_element(asap.begin(), asap.end());
  std::vector<int> alap(static_cast<std::size_t>(nodeCount), 0);
  for (int node = 0; node < nodeCount; ++node)
    alap[node] = makespan - reverseDepth[node] + 1;

  std::set<int> cycleNodes;
  for (Edge feedback : controlEdgeSet) {
    std::set<int> descendants{feedback.second};
    std::vector<int> work{feedback.second};
    while (!work.empty()) {
      int node = work.back();
      work.pop_back();
      for (int child : children[node])
        if (descendants.insert(child).second)
          work.push_back(child);
    }
    std::set<int> ancestors{feedback.first};
    work = {feedback.first};
    while (!work.empty()) {
      int node = work.back();
      work.pop_back();
      for (int parent : parents[node])
        if (ancestors.insert(parent).second)
          work.push_back(parent);
    }
    for (int node : descendants)
      if (ancestors.count(node))
        cycleNodes.insert(node);
    cycleNodes.insert(feedback.first);
    cycleNodes.insert(feedback.second);
  }

  const double denominator = static_cast<double>(std::max(1, nodeCount - 1));
  graph.nodeFeatures.reserve(order.size());
  for (int node = 0; node < nodeCount; ++node) {
    const std::string &kind = kinds[order[node]];
    NodeFeature row{};
    row[0] = static_cast<double>(parents[node].size()) /
             static_cast<double>(std::max(1, nodeCount));
    row[1] = static_cast<double>(children[node].size()) /
             static_cast<double>(std::max(1, nodeCount));
    row[2] = static_cast<double>(asap[node]) /
             static_cast<double>(std::max(1, makespan));
    row[3] = static_cast<double>(reverseDepth[node]) /
             static_cast<double>(std::max(1, makespan));
    row[4] = static_cast<double>(std::max(0, alap[node] - asap[node])) /
             static_cast<double>(std::max(1, makespan));
    row[5] = static_cast<double>(node) / denominator;
    row[6] = parents[node].empty();
    row[7] = children[node].empty();
    row[8] = isMemory(kind);
    row[9] = isPointer(kind);
    row[10] = isControl(kind);
    row[11] = !isTransparent(kind);
    row[12] = kind == "data_mov";
    row[13] = cycleNodes.count(node) != 0;
    graph.nodeFeatures.push_back(row);
  }

  if (requireExpanded && !graph.semanticEdges.empty()) {
    bool hasMovement = false;
    for (const NodeFeature &row : graph.nodeFeatures)
      hasMovement |= row[12] > 0.5;
    for (Edge edge : graph.semanticEdges) {
      if (containsEdge(graph.edges, edge)) {
        error = "Neura DFG is not route-expanded; semantic edge bypasses data_mov";
        return false;
      }
    }
    if (!hasMovement) {
      error = "Neura DFG is not route-expanded; no data_mov nodes were found";
      return false;
    }
  }
  return true;
}

template <std::size_t Width>
static bool computeMapperFeaturesForProtocol(
    const RouteExpandedGraph &graph, int rows, int columns, double recMii,
    double resMii, double lowerBound,
    llvm::ArrayRef<std::pair<int, int>> supportedShapes, double maxRows,
    double maxColumns, double maxTiles, double maxDirectedLinks,
    double maxBisectionLinks, double mapperIICeiling,
    std::array<double, Width> &features, std::string &error) {
  auto validShape = [&] {
    return std::find(supportedShapes.begin(), supportedShapes.end(),
                     std::pair<int, int>{rows, columns}) !=
           supportedShapes.end();
  };
  if (!validShape()) {
    error = "mapper shape is outside the selected static shape protocol";
    return false;
  }
  if (!std::isfinite(recMii) || !std::isfinite(resMii) ||
      !std::isfinite(lowerBound)) {
    error = "analytical mapper bounds must be finite";
    return false;
  }
  if (std::max(recMii, resMii) != lowerBound) {
    error = "lower_bound must equal max(rec_mii, res_mii)";
    return false;
  }
  if (lowerBound < 0.0 || lowerBound > mapperIICeiling) {
    error = "lower_bound is outside the mapper search interval";
    return false;
  }
  const std::size_t nodeCount = graph.nodeTypes.size();
  if (nodeCount == 0 || graph.nodeFeatures.size() != nodeCount) {
    error = "DFG node types and node features must be non-empty and aligned";
    return false;
  }
  if (nodeCount > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    error = "DFG is too large for the indexed feature contract";
    return false;
  }
  for (int type : graph.nodeTypes) {
    if (type < 0 || type >= static_cast<int>(kOperationTypeCount)) {
      error = "DFG node type is outside the route-expanded vocabulary";
      return false;
    }
  }
  for (const NodeFeature &row : graph.nodeFeatures) {
    for (double value : row) {
      if (!std::isfinite(value)) {
        error = "DFG node features must be finite";
        return false;
      }
    }
  }
  const int nodeCountInt = static_cast<int>(nodeCount);
  for (Edge edge : graph.edges) {
    if (edge.first < 0 || edge.second < 0 ||
        edge.first >= nodeCountInt || edge.second >= nodeCountInt) {
      error = "DFG edge endpoint is outside the node range";
      return false;
    }
  }
  for (Edge edge : graph.semanticEdges) {
    if (edge.first < 0 || edge.second < 0 ||
        edge.first >= nodeCountInt || edge.second >= nodeCountInt) {
      error = "semantic DFG edge endpoint is outside the node range";
      return false;
    }
  }
  const TopologySummary rawTopology =
      forwardTopology(nodeCountInt, graph.edges);
  const TopologySummary semanticTopology =
      forwardTopology(nodeCountInt, graph.semanticEdges);

  std::array<int, kOperationTypeCount> typeCounts{};
  for (int type : graph.nodeTypes)
    ++typeCounts[static_cast<std::size_t>(type)];
  std::array<double, kNodeFeatureWidth> means{};
  std::array<double, kNodeFeatureWidth> maxima{};
  maxima.fill(-std::numeric_limits<double>::infinity());
  for (const NodeFeature &row : graph.nodeFeatures) {
    for (std::size_t feature = 0; feature < kNodeFeatureWidth; ++feature) {
      means[feature] += row[feature];
      maxima[feature] = std::max(maxima[feature], row[feature]);
    }
  }
  for (double &value : means)
    value /= static_cast<double>(nodeCount);

  std::vector<int> rawOutdegree(nodeCount, 0);
  std::vector<std::vector<int>> forwardChildren(nodeCount);
  std::vector<std::vector<int>> forwardParents(nodeCount);
  for (Edge edge : graph.edges) {
    if (edge.first < edge.second) {
      ++rawOutdegree[static_cast<std::size_t>(edge.first)];
      forwardChildren[static_cast<std::size_t>(edge.first)].push_back(
          edge.second);
      forwardParents[static_cast<std::size_t>(edge.second)].push_back(
          edge.first);
    }
  }
  const int maxOutdegree =
      *std::max_element(rawOutdegree.begin(), rawOutdegree.end());
  int materializedCount = 0;
  int movementCount = 0;
  int recurrenceCount = 0;
  std::vector<bool> materialized(nodeCount, false);
  for (std::size_t node = 0; node < nodeCount; ++node) {
    materialized[node] = graph.nodeFeatures[node][11] > 0.5;
    materializedCount += materialized[node];
    movementCount += graph.nodeFeatures[node][12] > 0.5;
    recurrenceCount += graph.nodeFeatures[node][13] > 0.5;
  }

  std::vector<int> reverseMaterializedDepth(nodeCount, 0);
  for (int node = nodeCountInt - 1; node >= 0; --node) {
    int reverseDepth = 0;
    for (int user : forwardChildren[static_cast<std::size_t>(node)])
      reverseDepth = std::max(
          reverseDepth,
          reverseMaterializedDepth[static_cast<std::size_t>(user)] +
              static_cast<int>(materialized[static_cast<std::size_t>(user)]));
    reverseMaterializedDepth[static_cast<std::size_t>(node)] = reverseDepth;
  }
  const int maximumReverseDepth =
      *std::max_element(reverseMaterializedDepth.begin(),
                        reverseMaterializedDepth.end());
  std::vector<int> bucketWidths(
      static_cast<std::size_t>(maximumReverseDepth + 1), 0);
  std::vector<std::vector<int>> bucketNodes(
      static_cast<std::size_t>(maximumReverseDepth + 1));
  for (std::size_t node = 0; node < nodeCount; ++node) {
    if (!materialized[node])
      continue;
    const int bucket = maximumReverseDepth - reverseMaterializedDepth[node];
    ++bucketWidths[static_cast<std::size_t>(bucket)];
    bucketNodes[static_cast<std::size_t>(bucket)].push_back(
        static_cast<int>(node));
  }
  const int maximumBucketWidth =
      *std::max_element(bucketWidths.begin(), bucketWidths.end());
  double faninCollisions = 0.0;
  double fanoutCollisions = 0.0;
  for (const auto &parents : forwardParents)
    faninCollisions += static_cast<double>(parents.size()) *
                       static_cast<double>(parents.size() - 1) / 2.0;
  for (const auto &children : forwardChildren)
    fanoutCollisions += static_cast<double>(children.size()) *
                        static_cast<double>(children.size() - 1) / 2.0;
  int bucketDegree = 0;
  double bucketFaninCollisions = 0.0;
  double bucketFanoutCollisions = 0.0;
  for (const auto &nodes : bucketNodes) {
    int degree = 0;
    double fanin = 0.0;
    double fanout = 0.0;
    for (int node : nodes) {
      const auto &parents = forwardParents[static_cast<std::size_t>(node)];
      const auto &children = forwardChildren[static_cast<std::size_t>(node)];
      degree += static_cast<int>(parents.size() + children.size());
      fanin += static_cast<double>(parents.size()) *
               static_cast<double>(parents.size() - 1) / 2.0;
      fanout += static_cast<double>(children.size()) *
                static_cast<double>(children.size() - 1) / 2.0;
    }
    bucketDegree = std::max(bucketDegree, degree);
    bucketFaninCollisions = std::max(bucketFaninCollisions, fanin);
    bucketFanoutCollisions = std::max(bucketFanoutCollisions, fanout);
  }

  const double tiles = static_cast<double>(rows * columns);
  const double links = static_cast<double>(
      2 * (rows * std::max(0, columns - 1) +
           columns * std::max(0, rows - 1)));
  const double bisectionLinks =
      tiles == 1.0 ? 0.0 : 2.0 * static_cast<double>(std::min(rows, columns));
  const double routedCapacity = std::max(1.0, links * lowerBound);
  const double possibleEdges = static_cast<double>(std::max<std::size_t>(
      1, nodeCount * (nodeCount - 1)));
  std::size_t index = 0;
  auto push = [&](double value) {
    if (index < Width)
      features[index++] = value;
  };
  for (int count : typeCounts)
    push(std::log1p(static_cast<double>(count)));
  for (double value : means)
    push(value);
  for (double value : maxima)
    push(value);
  push(std::log1p(static_cast<double>(nodeCount)));
  push(std::log1p(static_cast<double>(graph.edges.size())));
  push(std::log1p(static_cast<double>(graph.semanticEdges.size())));
  push(std::log1p(static_cast<double>(materializedCount)));
  push(std::log1p(static_cast<double>(movementCount)));
  push(std::log1p(static_cast<double>(rawTopology.depth)));
  push(std::log1p(static_cast<double>(semanticTopology.depth)));
  push(std::log1p(static_cast<double>(rawTopology.width)));
  push(static_cast<double>(rawTopology.width) / static_cast<double>(nodeCount));
  push(std::log1p(static_cast<double>(rawTopology.sources)));
  push(std::log1p(static_cast<double>(rawTopology.sinks)));
  push(std::log1p(static_cast<double>(rawTopology.maxIndegree)));
  push(std::log1p(static_cast<double>(maxOutdegree)));
  push(static_cast<double>(graph.edges.size()) / possibleEdges);
  push(static_cast<double>(graph.semanticEdges.size()) / possibleEdges);
  push(std::log1p(static_cast<double>(recurrenceCount)));
  push(static_cast<double>(rows) / maxRows);
  push(static_cast<double>(columns) / maxColumns);
  push(tiles / maxTiles);
  push(std::log(static_cast<double>(columns) / static_cast<double>(rows)));
  push(links / maxDirectedLinks);
  push(bisectionLinks / maxBisectionLinks);
  push(recMii / mapperIICeiling);
  push(resMii / mapperIICeiling);
  push(lowerBound / mapperIICeiling);
  for (auto shape : supportedShapes)
    push(static_cast<double>(rows == shape.first && columns == shape.second));
  push(std::log1p(static_cast<double>(nodeCount) / tiles));
  push(std::log1p(static_cast<double>(materializedCount) / tiles));
  push(std::log1p(static_cast<double>(graph.edges.size()) /
                  std::max(1.0, links)));
  push(std::log1p(static_cast<double>(graph.semanticEdges.size()) /
                  std::max(1.0, links)));
  push(static_cast<double>(rawTopology.depth) /
       static_cast<double>(std::max(rows, columns)));
  push(static_cast<double>(rawTopology.width) / tiles);
  push(static_cast<double>(maximumBucketWidth) / tiles);
  push(static_cast<double>(maximumBucketWidth) /
       std::max(1.0, tiles * lowerBound));
  push(static_cast<double>(recurrenceCount) / tiles);
  push(faninCollisions / routedCapacity);
  push(fanoutCollisions / routedCapacity);
  push(static_cast<double>(graph.edges.size()) / routedCapacity);
  push(static_cast<double>(rawTopology.depth) / static_cast<double>(rows));
  push(static_cast<double>(rawTopology.depth) / static_cast<double>(columns));
  push(static_cast<double>(semanticTopology.depth) /
       static_cast<double>(rows));
  push(static_cast<double>(semanticTopology.depth) /
       static_cast<double>(columns));
  push(static_cast<double>(maximumBucketWidth) / static_cast<double>(rows));
  push(static_cast<double>(maximumBucketWidth) /
       static_cast<double>(columns));
  push(static_cast<double>(bucketDegree) / routedCapacity);
  push(bucketFaninCollisions / routedCapacity);
  push(bucketFanoutCollisions / routedCapacity);
  double bucketQuadraticOccupancy = 0.0;
  for (int width : bucketWidths)
    bucketQuadraticOccupancy += static_cast<double>(width) * width;
  push(bucketQuadraticOccupancy /
       std::max(1.0, tiles * tiles * lowerBound));

  if (index != Width) {
    error = "internal mapper feature contract width mismatch";
    return false;
  }
  for (double value : features) {
    if (!std::isfinite(value)) {
      error = "mapper features must be finite";
      return false;
    }
  }
  return true;
}

bool computeMapperFeatures(const RouteExpandedGraph &graph, int rows,
                           int columns, double recMii, double resMii,
                           double lowerBound, FeatureVector &features,
                           std::string &error) {
  return computeMapperFeaturesForProtocol(
      graph, rows, columns, recMii, resMii, lowerBound, kMapperShapes, 16.0,
      16.0, 256.0, 224.0, 16.0, 20.0, features.values, error);
}

bool computePerCgra2x2MapperFeatures(
    const RouteExpandedGraph &graph, int rows, int columns, double recMii,
    double resMii, double lowerBound, PerCgra2x2FeatureVector &features,
    std::string &error) {
  return computeMapperFeaturesForProtocol(
      graph, rows, columns, recMii, resMii, lowerBound,
      kPerCgra2x2MapperShapes, 8.0, 8.0, 16.0, 48.0, 8.0, 20.0,
      features.values, error);
}

} // namespace mapper_features
} // namespace orbit
