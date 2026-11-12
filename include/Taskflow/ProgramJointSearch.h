#ifndef ORBIT_TASKFLOW_PROGRAM_JOINT_SEARCH_H
#define ORBIT_TASKFLOW_PROGRAM_JOINT_SEARCH_H
#include "Taskflow/ConstraintSearch.h"
#include "Taskflow/ProgramMapping.h"
#include "Taskflow/ProgramMechanismCost.h"

namespace orbit::taskflow {
// Both integer witnesses and binary64 solver-bound telemetry must distinguish
// adjacent nanoseconds. Every checked addition must remain within this range.
inline constexpr int64_t programJointMaxExactNs = (int64_t{1} << 53) - 1;
// A finite domain of LOCAL decisions over the existing public source graph.
// No element below is a pre-priced complete execution plan.
struct ProgramJointVariant {
  ProgramGraph graph; // source-bound instances, exact cells and versions
  int64_t chunk = 0;
  std::vector<int64_t> computeNs, valueBytes;
  std::vector<std::vector<size_t>> readers; // version -> instance indices
  std::vector<int64_t> producers; // version -> instance, -1 initial input
};
struct ProgramJointGroup {
  size_t variant = 0, slice = 0;
  std::vector<size_t> instances, values, inputs, outputs;
};
struct ProgramJointPlace {
  MappingTaskPlacement binding; // resource and accessible memory, task unused
  std::vector<size_t> inputLinks, outputLinks; // public architecture indices
};
struct ProgramJointDomain {
  std::vector<ProgramJointVariant> variants;
  std::vector<ProgramJointGroup> groups;
  std::vector<ProgramJointPlace> places;
  StaticConfig target;
  ProgramMechanismModel model;
  std::string identity, hostMemory;
  int64_t configurationNs = 0, submitNs = 0, completeNs = 0;
  int64_t maxConfigurations = 16, maxGroupInputs = 2, maxGroupOutputs = 2;
  int64_t maxConfigurationInputs = 2, maxConfigurationOutputs = 2;
  bool complete = true;
  // Structural generation and pricing coverage are independent. A supported
  // route with no usable price must never become a proof of infeasibility.
  bool costsComplete = true;
  std::vector<std::string> unknownCosts;
  double generationMs = 0;
};
struct ProgramJointTransfer {
  size_t version = 0, link = 0;
  int64_t startNs = 0, endNs = 0;
};
struct ProgramJointSelection {
  size_t group = 0, place = 0, configuration = 0;
  std::vector<int64_t> computeStartNs;
  std::vector<ProgramJointTransfer> inputs, outputs;
};
struct ProgramJointSolution {
  ConstraintSearchStatus status = ConstraintSearchStatus::Unknown;
  std::string message, selection = "none";
  int64_t variant = -1, makespanNs = 0;
  std::vector<ProgramJointSelection> groups;
  std::vector<int64_t> configurationStartNs, configurationEndNs;
  ConstraintSearchTelemetry telemetry;
  bool hasPlan = false;
  // Solver-observed pricing coverage, independent of caller domain metadata.
  // A bound from a priced subset is not a bound on unpriced alternatives.
  bool pricingComplete = true;
};

bool buildProgramJointDomain(const GenericAnalysis &, const llvm::json::Value &,
                             ProgramJointDomain &, std::string &);
bool programJointGroupFits(const ProgramJointDomain &, size_t group,
                           size_t place, int64_t &bytes, std::string &);
bool programJointTransferSupported(const ProgramJointDomain &, size_t variant,
                                   size_t version, size_t link, std::string &);
bool programJointTransferNs(const ProgramJointDomain &, size_t variant,
                            size_t version, size_t link, int64_t &ns,
                            std::string &);
ProgramJointSolution baselineProgramJointDomain(const ProgramJointDomain &);
ProgramJointSolution
solveProgramJointDomain(const ProgramJointDomain &,
                        const ConstraintSearchOptions & = {},
                        const ProgramJointSolution *checkedBaseline = nullptr);
bool materializeProgramJointSolution(const ProgramJointDomain &,
                                     const ProgramJointSolution &,
                                     ProgramGraph &, std::string &);
// Independent from OR-Tools: source, coverage, costs, data versions, capacity,
// paths, resource calendars and configuration barriers are all rechecked.
bool checkProgramJointSolution(const ProgramJointDomain &,
                               const ProgramJointSolution &,
                               const ProgramGraph &, std::string &);
llvm::json::Value serializeProgramJointDomain(const ProgramJointDomain &);
llvm::json::Value serializeProgramJointSolution(const ProgramJointDomain &,
                                                const ProgramJointSolution &,
                                                const ProgramGraph * = nullptr);
bool parseProgramJointSolution(const llvm::json::Value &,
                               ProgramJointSolution &, std::string &);
} // namespace orbit::taskflow
#endif
