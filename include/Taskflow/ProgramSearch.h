#ifndef ORBIT_TASKFLOW_PROGRAM_SEARCH_H
#define ORBIT_TASKFLOW_PROGRAM_SEARCH_H

#include "Taskflow/ConstraintSearch.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace orbit::taskflow {
struct ProgramSearchChoice {
  std::string id;
  llvm::json::Value binding = nullptr;
};
struct ProgramSearchDimension {
  std::string id, scope, status, reason;
  std::vector<ProgramSearchChoice> choices;
};
// A local compatibility table, never a catalogue of priced complete plans.
struct ProgramSearchConstraint {
  std::vector<size_t> dimensions;
  std::vector<std::vector<int64_t>> allowedTuples;
};
struct ProgramSearchSpace {
  std::vector<ProgramSearchDimension> dimensions;
  std::vector<ProgramSearchConstraint> constraints;
};
struct ProgramSearchEvaluation {
  bool feasible = false;
  // False means a checker/provider could not decide; it is not a proof of
  // infeasibility and prevents an exhaustive optimality claim.
  bool feasibilityKnown = true;
  std::optional<double> costMs;
  std::string message;
  llvm::json::Value result = nullptr;
  std::string sourceMLIR;
};
struct ProgramSearchOptions {
  double wallSeconds = 600;
  int64_t maxEvaluations = 2048, maxVariables = 100000;
  int64_t maxConstraints = 500000;
  int64_t maxTupleCells = 1000000;
  // Keep typed callers and JSON parsing on the same bounded solver contract.
  // Seeds are nonnegative signed integers; workers are in [1,16].
  int workers = 1, seed = 0;
};
struct ProgramSearchResult {
  ConstraintSearchStatus status = ConstraintSearchStatus::Unknown;
  bool hasBest = false;
  std::vector<int64_t> bestAssignment;
  ProgramSearchEvaluation bestEvaluation;
  llvm::json::Object report;
};
using ProgramSearchEvaluator =
    std::function<ProgramSearchEvaluation(const std::vector<int64_t> &)>;

// Validates and bounds the input domain, then checks the all-zero baseline
// before constructing the solver model. Oversized input domains are rejected
// before callback execution. Uses one model with integer choice
// variables and excludes each checked assignment. No additive cost surrogate
// is inferred. Wall quota is shared by construction, solver and callbacks;
// callbacks are synchronous/nonpreemptive and may exceed this soft quota.
ProgramSearchResult searchProgramSpace(const ProgramSearchSpace &,
                                       const ProgramSearchEvaluator &,
                                       const ProgramSearchOptions & = {});
} // namespace orbit::taskflow
#endif
