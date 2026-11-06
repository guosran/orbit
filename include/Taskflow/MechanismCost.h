#ifndef ORBIT_TASKFLOW_MECHANISM_COST_H
#define ORBIT_TASKFLOW_MECHANISM_COST_H

#include "Taskflow/ProgramMapping.h"
#include <map>

namespace orbit::taskflow {
// Explicit executor binding, not inferred physical DDR behavior. The initial
// supported contract is a prepared hot call with all ABI buffers caller-bound.
struct MechanismRuntimeContract {
  std::string identity = "unbound";
  std::string state = "prepared-hot-bound-bo";
  std::string syncToDevice = "all-abi";
  std::string syncFromDevice = "outputs";
  bool boundBuffers = true, inputReset = true, finalOutputCopy = true;
  bool outputPoisoned = true;
  // Runtime cost labels bind to a declared serial host worker, never to a
  // fabricated physical DRAM port. Parameter resource labels stay unchanged.
  std::map<std::string,std::string> resourceAliases = {
      {"host.thread","CPU.compute"},{"host.memory","CPU.compute"}};
};
struct MechanismResourceBinding {
  std::string targetResource, semantics;
  int64_t targetCapacity = 0;
};
struct MechanismItem {
  size_t id = 0;
  int64_t region = -1, storage = -1;
  std::string mechanism, path, state, resource, boundary;
  int64_t bytes = 0;
  std::vector<size_t> copies, sourceTasks, publicEvents, regionEvents;
  // Composite items remain required but uncalibrated. Descriptive rows own no
  // separate duration and point at the enclosing composite's item identity.
  bool composite = false, descriptive = false, runtimeOnly = false;
  int64_t coveredBy = -1;
};
struct MechanismFacts {
  std::string graphId, bodyFingerprint, identity;
  MechanismRuntimeContract runtime;
  std::map<std::string,MechanismResourceBinding> runtimeResourceBindings;
  std::vector<MechanismItem> items;
};
struct MechanismParameter {
  std::string mechanism, path, state, resource, boundary, runtimeIdentity;
  std::string provenance, measurementBoundary, rawManifestSHA256;
  int64_t minBytes = 0, maxBytes = 0;
  double alphaMs = 0, betaMsPerByte = 0;
};
struct MechanismPricedItem {
  size_t item = 0;
  std::optional<double> durationMs, startMs, endMs;
  std::string status, reason;
};
struct MechanismCostEvaluation {
  std::string factsIdentity, parameterIdentity;
  double knownSubtotalMs = 0;
  std::optional<double> totalMs;
  std::vector<MechanismPricedItem> items;
};
bool collectMechanismFacts(const ProgramGraph &, const MechanismRuntimeContract &,
                           MechanismFacts &, std::string &error);
bool parseMechanismParameters(const llvm::json::Value &,
                              std::vector<MechanismParameter> &,
                              std::string &error);
bool evaluateMechanismCost(const MechanismFacts &,
                           const std::vector<MechanismParameter> &,
                           MechanismCostEvaluation &, std::string &error);
llvm::json::Value serializeMechanismFacts(const MechanismFacts &);
llvm::json::Value serializeMechanismCost(const MechanismCostEvaluation &);
} // namespace orbit::taskflow
#endif
