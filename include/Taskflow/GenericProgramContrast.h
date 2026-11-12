#ifndef ORBIT_TASKFLOW_GENERIC_PROGRAM_CONTRAST_H
#define ORBIT_TASKFLOW_GENERIC_PROGRAM_CONTRAST_H
#include "Taskflow/GenericMapping.h"
#include <map>
namespace orbit::taskflow {
// Two views of existing checked plans. No graph, decisions or durations are
// synthesized here. Quantities and deltas are native facts, in right-left
// order.
struct GenericContrastPlanFacts {
  std::string graphId, planId, bodyFingerprint, familyFingerprint,
      architectureFingerprint;
  std::map<std::string, int64_t> quantities;
  std::vector<std::string> implementations;
};
struct GenericProgramContrastFacts {
  GenericContrastPlanFacts baseline, candidate;
  std::string changeClass, reason;
  std::map<std::string, int64_t> delta;
  std::vector<std::string> coveredChange, commonConditions, unknowns;
  bool supported = false;
};
struct GenericProgramContrastCost {
  std::string status = "unknown", reason, modelIdentity, changeClass;
  std::optional<double> deltaNs, lowerDeltaNs, upperDeltaNs;
  llvm::json::Array contributions;
  // Absolute invocation times and a physical resource timeline remain unknown.
  // A contrast-only fit must never be silently promoted to a full search score.
};
bool collectGenericProgramContrastFacts(const GenericAnalysis &,
                                        const GenericMappingPlan &,
                                        const GenericAnalysis &,
                                        const GenericMappingPlan &,
                                        GenericProgramContrastFacts &,
                                        std::string &);
bool evaluateGenericProgramContrast(const GenericProgramContrastFacts &,
                                    const llvm::json::Value &model,
                                    const llvm::json::Value &target,
                                    GenericProgramContrastCost &,
                                    std::string &);
llvm::json::Value
serializeGenericProgramContrastFacts(const GenericProgramContrastFacts &);
llvm::json::Value
serializeGenericProgramContrastCost(const GenericProgramContrastCost &);
} // namespace orbit::taskflow
#endif
