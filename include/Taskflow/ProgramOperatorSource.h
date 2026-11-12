#ifndef ORBIT_TASKFLOW_PROGRAM_OPERATOR_SOURCE_H
#define ORBIT_TASKFLOW_PROGRAM_OPERATOR_SOURCE_H

#include "Taskflow/ConstraintSearch.h"
#include "Taskflow/ProgramMapping.h"
#include <memory>

namespace orbit::taskflow {
inline constexpr size_t programOperatorMaxOperators = 200000;
inline constexpr size_t programOperatorMaxValues = 1000000;
// A first-class external source contract, not a synthetic GenericAnalysis body.
// The producer proves source/artifact hashes and tensor flow; this interface
// checks its typed data/control semantics. It does not prove external
// arithmetic.
struct ProgramOperatorBinding {
  std::string id, file, sha256, symbol;
};
struct ProgramOperatorView {
  size_t base = 0;
  std::vector<int64_t> offsets, strides;
};
struct ProgramOperatorValue {
  std::string id, dtype, role, layout;
  std::vector<int64_t> shape;
  std::optional<ProgramOperatorView> view;
  std::optional<ProgramOperatorView> allocationView;
  std::string initialization;
  std::optional<size_t> stateAliasOf;
  std::string initialMemory;
};
struct ProgramSourceOperator {
  std::string id, kind, binding;
  int64_t sourceOrdinal = 0;
  std::vector<size_t> inputs, outputs, dependencies;
  llvm::json::Value attributes = llvm::json::Object{};
};
struct ProgramOperatorSource {
  std::string sourceId, numericalContract;
  std::vector<ProgramOperatorBinding> bindings;
  std::vector<ProgramOperatorValue> values;
  std::vector<ProgramSourceOperator> operators;
  // Preserve the registered source document, including view semantics, for
  // canonical identity. Typed fields above are independently checked against
  // it.
  llvm::json::Value document = nullptr;
};
// An implementation covers actual source operators. Coverage is not permission
// to split a fused kernel: only independently executable registered covers may
// appear in a selection. Boundary values are derived and checked, never
// guessed.
struct ProgramOperatorImplementation {
  std::string id, sourceId, artifactIdentity, entrypoint;
  std::string device, resource, memory;
  std::vector<size_t> operators, inputs, outputs;
  std::optional<double> wallMs;
  bool executable = false;
  int64_t launchCount = 0;
  std::string transportOwnership = "implementation-inclusive";
  std::string visibilityPolicy = "host-materialized";
  std::map<size_t, std::string> inputLocations, outputLocations;
  // Inclusive implementation wall owns invocation and transport. This contract
  // cannot be passed off as transfer-exclusive MappingComputeEstimate data.
  std::string costBoundary = "inclusive-region-wall";
};
struct ProgramOperatorChoiceGroup {
  std::string id;
  std::vector<std::string> choiceIds;
  std::vector<std::vector<size_t>>
      choices; // local implementation catalogue indices
};
struct ProgramOperatorSelection {
  std::vector<ProgramOperatorImplementation> regions;
};
bool parseProgramOperatorSource(const llvm::json::Value &,
                                std::shared_ptr<ProgramOperatorSource> &,
                                std::string &error);
bool parseProgramOperatorImplementations(
    const ProgramOperatorSource &, const llvm::json::Value &,
    std::vector<ProgramOperatorImplementation> &, std::string &error);
// Certifies local exact covers and invariant cross-group ABI once. Candidate
// cost updates may then use the product domain without rewalking immutable IR.
bool certifyProgramOperatorChoices(
    const ProgramOperatorSource &,
    const std::vector<ProgramOperatorImplementation> &,
    const std::vector<ProgramOperatorChoiceGroup> &, llvm::json::Value &receipt,
    std::string &error);
bool checkProgramOperatorSource(const ProgramOperatorSource &, std::string &);
llvm::json::Value serializeProgramOperatorSource(const ProgramOperatorSource &);
// These graphs use the existing regions/versions/copies/events/regions as their
// only physical ledger. Native MLIR consumers must reject this source variant.
bool materializeProgramOperatorGraph(
    std::shared_ptr<const ProgramOperatorSource>,
    const ProgramOperatorSelection &, const StaticConfig &,
    llvm::StringRef hostMemory, ProgramGraph &, std::string &error);
bool checkProgramOperatorGraph(const ProgramGraph &, std::string &error);
llvm::json::Value summarizeProgramOperatorGraph(const ProgramGraph &);
ConstraintSearchResult
solveProgramOperatorCover(const ProgramOperatorSource &,
                          const std::vector<ProgramOperatorImplementation> &,
                          ProgramOperatorSelection &,
                          const ConstraintSearchOptions &);
llvm::json::Value
serializeProgramOperatorSelection(const ProgramOperatorSelection &);
} // namespace orbit::taskflow
#endif
