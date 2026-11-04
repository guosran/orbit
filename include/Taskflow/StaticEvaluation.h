#ifndef ORBIT_TASKFLOW_STATIC_EVALUATION_H
#define ORBIT_TASKFLOW_STATIC_EVALUATION_H
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <string>
namespace orbit::taskflow {
// Native boundary: parses actual MLIR and runs typed
// analysis/planning/checking. No process launch and no Python evaluator is
// involved.
llvm::json::Value evaluateStaticMLIR(llvm::StringRef text,
                                     const llvm::json::Value &config,
                                     std::string &error);
} // namespace orbit::taskflow
#endif
