//===- JointRewritePreflight.h -----------------------------------*- C++ -*-===//
// Exact necessary conditions for the source-owned graph rewrites. A successful
// preflight still requires the real materializer and final facts validation.
//===----------------------------------------------------------------------===//
#ifndef AMOEBA_JOINT_REWRITE_PREFLIGHT_H
#define AMOEBA_JOINT_REWRITE_PREFLIGHT_H

#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskGraphRewriteLegality.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::amoeba::neura::joint_scheduling {

enum class JointRewritePreflightKind { Other, ProducerConsumer, Sibling };
enum class JointRewritePreflightDisposition { Proceed, ProvenIllegal, Unknown };

struct JointRewritePreflightResult {
  JointRewritePreflightDisposition disposition =
      JointRewritePreflightDisposition::Proceed;
  std::string reason;
};

inline JointRewritePreflightResult
preflightJointRewrite(ModuleOp module, JointRewritePreflightKind kind,
                      StringRef firstName, StringRef secondName) {
  if (kind == JointRewritePreflightKind::Other)
    return {};
  SmallVector<taskflow::TaskflowTaskOp> firstMatches, secondMatches;
  module.walk([&](taskflow::TaskflowTaskOp task) {
    if (task.getTaskName() == firstName)
      firstMatches.push_back(task);
    if (task.getTaskName() == secondName)
      secondMatches.push_back(task);
  });
  if (firstMatches.size() != 1 || secondMatches.size() != 1)
    return {JointRewritePreflightDisposition::Unknown,
            "rewrite_endpoint_not_uniquely_identified"};
  auto first = firstMatches.front(), second = secondMatches.front();
  if (first == second)
    return {JointRewritePreflightDisposition::ProvenIllegal,
            "fusion_requires_two_distinct_tasks"};
  auto function = first->getParentOfType<func::FuncOp>();
  if (!function || function != second->getParentOfType<func::FuncOp>())
    return {JointRewritePreflightDisposition::Unknown,
            "rewrite_function_not_identified"};
  std::string error;
  TaskEdgeGraphOptions options;
  FailureOr<TaskEdgeGraph> graph = buildTaskEdgeGraph(function, options, error);
  if (failed(graph))
    return {JointRewritePreflightDisposition::Unknown,
            "rewrite_typed_graph_unavailable"};
  size_t rawEdges = 0;
  bool forwardValueEdge = false;
  for (const TaskEdge &edge : graph->getEdges()) {
    if (edge.origin != TaskEdgeOrigin::Taskflow)
      continue;
    bool forward = edge.producer == first && edge.consumer == second;
    bool reverse = edge.producer == second && edge.consumer == first;
    forwardValueEdge |= forward && edge.kind == TaskEdgeKind::Value;
    if (kind == JointRewritePreflightKind::Sibling && (forward || reverse) &&
        (edge.consumer_segment == TaskOperandSegment::WillReads ||
         edge.consumer_segment == TaskOperandSegment::ValueInputs))
      return {JointRewritePreflightDisposition::ProvenIllegal,
              "sibling_has_ssa_data_dependency"};
    if (forward && edge.kind == TaskEdgeKind::Raw &&
        edge.producer_segment == TaskResultSegment::DoneWrites &&
        edge.consumer_segment == TaskOperandSegment::WillReads)
      ++rawEdges;
  }
  // A missing SSA RAW alone is not an illegality proof: a memory-only or
  // scalar relationship may be legal but unsupported by this materializer.
  // Typed memory precedence can coalesce edge kinds, so prove every original
  // write/read storage pair distinct directly instead of assuming an absent
  // graph RAW means absent aliasing.
  if (kind == JointRewritePreflightKind::ProducerConsumer && rawEdges == 0) {
    if (forwardValueEdge ||
        first.getOriginalWriteMemrefs().size() != first.getWillWrites().size() ||
        second.getOriginalReadMemrefs().size() != second.getWillReads().size())
      return {JointRewritePreflightDisposition::Unknown,
              "producer_consumer_relationship_not_classified"};
    for (Value write : first.getOriginalWriteMemrefs())
      for (Value read : second.getOriginalReadMemrefs())
        if (!provesDistinctTaskStorage(write, read))
          return {JointRewritePreflightDisposition::Unknown,
                  "producer_consumer_memory_relationship_not_classified"};
    return {JointRewritePreflightDisposition::ProvenIllegal,
            "producer_consumer_no_data_relationship"};
  }
  // Multiple intermediates and cross-block/reversed placement are current
  // materializer coverage limits. They are not proofs of semantic illegality.
  if (kind == JointRewritePreflightKind::ProducerConsumer && rawEdges > 1)
    return {JointRewritePreflightDisposition::Unknown,
            "producer_consumer_multiple_raw_not_supported"};
  if (first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second))
    return {JointRewritePreflightDisposition::Unknown,
            "fusion_order_scope_not_proven"};
  return {};
}

// A preflight context is scoped to one immutable source module. Its task-name
// index is built once, and each function's typed edge graph is built at most
// once. Failed graph construction is cached as well, so repeated fusion
// probes cannot repeat the same expensive failure. The context must not be
// used after the source module is destroyed or mutated.
class JointRewritePreflightContext {
public:
  explicit JointRewritePreflightContext(ModuleOp source,
                                        uint64_t *externalBuildCount = nullptr)
      : source_(source), sourceOperation_(source.getOperation()),
        externalBuildCount_(externalBuildCount) {
    if (source_)
      source_.walk([&](taskflow::TaskflowTaskOp task) {
        tasksByName_[task.getTaskName()].push_back(task);
      });
  }

  JointRewritePreflightContext(const JointRewritePreflightContext &) = delete;
  JointRewritePreflightContext &
  operator=(const JointRewritePreflightContext &) = delete;

  bool isFor(ModuleOp source) const {
    return sourceOperation_ == source.getOperation();
  }

  uint64_t getGraphBuildCount() const { return graphBuildCount_; }

  JointRewritePreflightResult
  evaluate(JointRewritePreflightKind kind, StringRef firstName,
           StringRef secondName) {
    if (kind == JointRewritePreflightKind::Other)
      return {};

    auto firstIt = tasksByName_.find(firstName);
    auto secondIt = tasksByName_.find(secondName);
    if (firstIt == tasksByName_.end() || secondIt == tasksByName_.end() ||
        firstIt->second.size() != 1 || secondIt->second.size() != 1)
      return {JointRewritePreflightDisposition::Unknown,
              "rewrite_endpoint_not_uniquely_identified"};

    auto first = firstIt->second.front();
    auto second = secondIt->second.front();
    if (first == second)
      return {JointRewritePreflightDisposition::ProvenIllegal,
              "fusion_requires_two_distinct_tasks"};
    auto function = first->getParentOfType<func::FuncOp>();
    if (!function || function != second->getParentOfType<func::FuncOp>())
      return {JointRewritePreflightDisposition::Unknown,
              "rewrite_function_not_identified"};

    const TaskEdgeGraph *graph = getGraph(function);
    if (!graph)
      return {JointRewritePreflightDisposition::Unknown,
              "rewrite_typed_graph_unavailable"};
    size_t rawEdges = 0;
    bool forwardValueEdge = false;
    for (const TaskEdge &edge : graph->getEdges()) {
      if (edge.origin != TaskEdgeOrigin::Taskflow)
        continue;
      bool forward = edge.producer == first && edge.consumer == second;
      bool reverse = edge.producer == second && edge.consumer == first;
      forwardValueEdge |= forward && edge.kind == TaskEdgeKind::Value;
      if (kind == JointRewritePreflightKind::Sibling &&
          (forward || reverse) &&
          (edge.consumer_segment == TaskOperandSegment::WillReads ||
           edge.consumer_segment == TaskOperandSegment::ValueInputs))
        return {JointRewritePreflightDisposition::ProvenIllegal,
                "sibling_has_ssa_data_dependency"};
      if (forward && edge.kind == TaskEdgeKind::Raw &&
          edge.producer_segment == TaskResultSegment::DoneWrites &&
          edge.consumer_segment == TaskOperandSegment::WillReads)
        ++rawEdges;
    }
    // A missing SSA RAW alone is not an illegality proof: a memory-only or
    // scalar relationship may be legal but unsupported by this materializer.
    // Typed memory precedence can coalesce edge kinds, so prove every original
    // write/read storage pair distinct directly instead of assuming an absent
    // graph RAW means absent aliasing.
    if (kind == JointRewritePreflightKind::ProducerConsumer && rawEdges == 0) {
      if (forwardValueEdge ||
          first.getOriginalWriteMemrefs().size() !=
              first.getWillWrites().size() ||
          second.getOriginalReadMemrefs().size() !=
              second.getWillReads().size())
        return {JointRewritePreflightDisposition::Unknown,
                "producer_consumer_relationship_not_classified"};
      for (Value write : first.getOriginalWriteMemrefs())
        for (Value read : second.getOriginalReadMemrefs())
          if (!provesDistinctTaskStorage(write, read))
            return {JointRewritePreflightDisposition::Unknown,
                    "producer_consumer_memory_relationship_not_classified"};
      return {JointRewritePreflightDisposition::ProvenIllegal,
              "producer_consumer_no_data_relationship"};
    }
    // Multiple intermediates and cross-block/reversed placement are current
    // materializer coverage limits. They are not proofs of semantic
    // illegality.
    if (kind == JointRewritePreflightKind::ProducerConsumer && rawEdges > 1)
      return {JointRewritePreflightDisposition::Unknown,
              "producer_consumer_multiple_raw_not_supported"};
    if (first->getBlock() != second->getBlock() ||
        !first->isBeforeInBlock(second))
      return {JointRewritePreflightDisposition::Unknown,
              "fusion_order_scope_not_proven"};
    return {};
  }

private:
  struct CachedGraph {
    bool attempted = false;
    std::optional<TaskEdgeGraph> graph;
    std::string error;
  };

  const TaskEdgeGraph *getGraph(func::FuncOp function) {
    CachedGraph &cached = graphsByFunction_[function.getOperation()];
    if (!cached.attempted) {
      cached.attempted = true;
      ++graphBuildCount_;
      if (externalBuildCount_)
        ++*externalBuildCount_;
      TaskEdgeGraphOptions options;
      FailureOr<TaskEdgeGraph> graph =
          buildTaskEdgeGraph(function, options, cached.error);
      if (succeeded(graph))
        cached.graph.emplace(std::move(*graph));
    }
    return cached.graph ? &*cached.graph : nullptr;
  }

  ModuleOp source_;
  Operation *sourceOperation_ = nullptr;
  llvm::StringMap<SmallVector<taskflow::TaskflowTaskOp>> tasksByName_;
  llvm::DenseMap<Operation *, CachedGraph> graphsByFunction_;
  uint64_t graphBuildCount_ = 0;
  uint64_t *externalBuildCount_ = nullptr;
};

} // namespace mlir::amoeba::neura::joint_scheduling
#endif
