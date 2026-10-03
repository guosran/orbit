#ifndef ORBIT_JOINT_NEIGHBORHOOD_ACTIONS_H
#define ORBIT_JOINT_NEIGHBORHOOD_ACTIONS_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {
struct NeighborhoodShape {
  std::string task;
  int64_t rows = 1;
  int64_t cols = 1;
};
struct NeighborhoodPrimitive {
  std::string kind;
  std::string firstTask;
  std::string secondTask;
  std::string mode;
  int64_t axis = 0;
  int64_t factor = 1;
};
struct NeighborhoodAction {
  std::string family;
  std::string label;
  std::vector<NeighborhoodPrimitive> primitives;
  std::string shapeTask;
  int64_t shapeRows = 0;
  int64_t shapeCols = 0;
  bool canonicalReset = false;
};
std::vector<NeighborhoodAction> enumerateNeighborhoodActions(
    ModuleOp module, llvm::StringRef function,
    llvm::ArrayRef<NeighborhoodShape> shapes, llvm::StringRef stage,
    unsigned round);
bool applyNeighborhoodAction(
    ModuleOp cloned, ModuleOp canonical, llvm::StringRef function,
    const NeighborhoodAction &action, std::vector<NeighborhoodShape> &shapes,
    std::string &reason, std::string &diagnostic);
} // namespace mlir::amoeba::neura::joint_scheduling

#endif
