#ifndef ORBIT_TASKFLOW_GENERIC_ANALYSIS_H
#define ORBIT_TASKFLOW_GENERIC_ANALYSIS_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <vector>
namespace orbit::taskflow {
struct GenericInterval {
  int64_t lower = 0, upper = 0;
}; // half open bounding region
struct GenericStorage {
  std::string id, role, dtype;
  mlir::Value root;
  std::vector<int64_t> shape;
};
struct GenericAccess {
  size_t storage = 0;
  std::vector<GenericInterval> region;
  bool read = false, write = false;
  bool exactRegion = false; // bounding rectangle is an exact access set
  std::string writeVersion;
  std::vector<std::string>
      readVersions; // conservative overlapping producer versions
  mlir::Operation *operation = nullptr;
};
struct GenericLoop {
  mlir::Operation *operation = nullptr;
  mlir::Value induction;
  int64_t lower = 0, upper = 0, step = 1;
};
struct GenericTask {
  std::string id;
  mlir::Operation *operation = nullptr;
  std::vector<GenericAccess> accesses;
  // Storage indices for leading memref body arguments; scalar arguments follow.
  std::vector<size_t> argumentStorages;
  std::vector<GenericLoop> loops;
  std::vector<size_t> dependencies;
};
struct GenericAnalysis {
  mlir::func::FuncOp function;
  std::vector<GenericStorage> storages;
  std::vector<GenericTask> tasks;
  std::string bodyFingerprint;
};
bool analyzeGenericProgram(mlir::ModuleOp module, GenericAnalysis &analysis,
                           std::string &error);
llvm::json::Value serializeGenericAnalysis(const GenericAnalysis &analysis);
} // namespace orbit::taskflow
#endif
