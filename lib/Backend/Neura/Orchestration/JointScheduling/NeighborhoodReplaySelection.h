// A single selected complete-program shape tuple for budgeted native replay.
// This contract makes no statement about coverage of other shape tuples.
#ifndef AMOEBA_NEIGHBORHOOD_REPLAY_SELECTION_H
#define AMOEBA_NEIGHBORHOOD_REPLAY_SELECTION_H

#include "AnalyticalTaskCandidateCommon.h"
#include "SpatialTaskCandidateSpace.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::amoeba::neura::joint_scheduling {
inline constexpr llvm::StringLiteral kNeighborhoodShapeSelectionSchema =
    "orbit-neighborhood-shape-selection-v1";

inline std::string neighborhoodReplaySourceText(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

inline bool writeNeighborhoodShapeSelection(ModuleOp module,
                                            func::FuncOp function,
                                            const Candidate &candidate,
                                            llvm::StringRef path,
                                            std::string &error) {
  const auto &architecture = ::mlir::neura::getArchitecture();
  llvm::json::Object selection = candidateJson(candidate);
  selection["schema"] = kNeighborhoodShapeSelectionSchema;
  selection["record_type"] = "selection";
  selection["function"] = function.getSymName();
  auto graph = function->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
  selection["graph_variant_id"] = graph ? graph.getValue() : llvm::StringRef();
  selection["search_scope"] = "budgeted-complete-program-neighborhood";
  selection["best_found"] = true;
  selection["source_ir"] = neighborhoodReplaySourceText(module);
  selection["architecture"] = llvm::json::Object{
      {"grid_rows", int64_t(architecture.getMultiCgraRows())},
      {"grid_cols", int64_t(architecture.getMultiCgraColumns())},
      {"per_cgra_tile_rows", int64_t(architecture.getPerCgraRows())},
      {"per_cgra_tile_cols", int64_t(architecture.getPerCgraColumns())}};
  return writeAtomically(path, [&](llvm::raw_ostream &stream) {
    stream << llvm::json::Value(std::move(selection)) << "\n";
    return true;
  }, error);
}
} // namespace mlir::amoeba::neura::joint_scheduling
#endif
