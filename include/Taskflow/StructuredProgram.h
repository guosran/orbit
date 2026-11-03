#ifndef ORBIT_TASKFLOW_STRUCTURED_PROGRAM_H
#define ORBIT_TASKFLOW_STRUCTURED_PROGRAM_H
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <string>
#include <vector>
namespace orbit::taskflow {
struct Range {
  int64_t lo = 0, hi = 0;
  bool operator==(const Range &o) const { return lo == o.lo && hi == o.hi; }
};
struct Buffer {
  std::string id, dtype, role;
  int64_t rows, cols;
};
struct Compute {
  std::string kind;
  std::vector<std::string> inputs;
  std::string output;
  Range m, n, k;
  bool accumulate = false;
};
struct Task {
  std::string id;
  std::vector<std::string> dependsOn;
  std::vector<Compute> operations;
};
struct Program {
  std::string sourceId, candidateId;
  std::vector<Buffer> buffers;
  std::vector<Task> tasks;
  std::vector<std::string> outputs;
  llvm::json::Array transforms;
};
struct TransformOptions {
  int64_t tileM = 0, tileN = 0, tileK = 0;
  bool fuse = false, fission = false;
};
// Deliberately constrained static f32 exchange. Taskflow MLIR is authenticated
// against this witness before a plan is exported or transformed.
bool parsePlan(const llvm::json::Value &, Program &, std::string &error);
llvm::json::Value printPlan(const Program &);
bool validate(const Program &, std::string &error);
bool transform(Program &, const TransformOptions &, std::string &error);
std::string emitTaskflow(const Program &);
std::string fingerprint(const Program &);
} // namespace orbit::taskflow
#endif
