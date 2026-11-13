// First-class operator source -> local implementation-cover ORBIT beam ->
// ProgramGraph. Actual external runtime replay is a separate checked adapter
// operation.
#include "Taskflow/JSONUtils.h"
#include "Taskflow/ProgramBeamSearch.h"
#include "Taskflow/ProgramOperatorMapping.h"
#include "Taskflow/ProgramOperatorSource.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
using namespace orbit::taskflow;
namespace j = llvm::json;
static j::Value readJSON(const char *path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    throw std::runtime_error("cannot read JSON input");
  auto parsed = j::parse((*buffer)->getBuffer());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    throw std::runtime_error("invalid JSON input");
  }
  return std::move(*parsed);
}
static bool target(const j::Value &catalogue, StaticConfig &out,
                   std::string &error) {
  auto *root = catalogue.getAsObject();
  if (root && root->get("architecture"))
    return parseMemoryArchitecture(*root->get("architecture"), out, error);
  StaticResourceDomain system;
  system.id = "runtime.system";
  system.kind = "system";
  StaticResourceDomain npu;
  npu.id = "runtime.npu";
  npu.parent = system.id;
  npu.kind = "device";
  out.domains = {system, npu};
  StaticResource cpu;
  cpu.id = "cpu.host";
  cpu.kind = "compute";
  cpu.device = "cpu";
  cpu.capacity = 1;
  cpu.domain = system.id;
  StaticResource device;
  device.id = "npu.device";
  device.kind = "compute";
  device.device = "npu";
  device.capacity = 1;
  device.domain = npu.id;
  out.resources = {cpu, device};
  StaticMemory memory;
  memory.id = "runtime-abi";
  memory.kind = "system-bo";
  memory.domain = system.id;
  memory.accessibleResources = {cpu.id, device.id};
  memory.capacityKnown = false;
  memory.capacityBytes = INT64_MAX;
  out.memories = {memory};
  return validateMemoryArchitecture(out, error);
}
static void saveJSON(const std::string &path, const j::Value &value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << value;
  stream.flush();
  std::ofstream output(path);
  if (!output)
    throw std::runtime_error("cannot write output " + path);
  output << text << "\n";
}
int main(int argc, char **argv) {
  try {
    const bool auditCheckpoints =
        argc == 6 && std::string(argv[4]) == "--audit-checkpoints";
    const bool materializeChoices =
        argc == 6 && std::string(argv[4]) == "--materialize-choices";
    const char *warmStartPath =
        argc == 6 && std::string(argv[4]) == "--warm-start"   ? argv[5]
        : argc == 7 && std::string(argv[5]) == "--warm-start" ? argv[6]
                                                              : nullptr;
    if (argc != 4 && argc != 5 && !auditCheckpoints && !materializeChoices &&
        !warmStartPath)
      throw std::runtime_error(
          "usage: program-operator-search source.json "
          "implementations.json output.json [beam-width] "
          "or --audit-checkpoints checkpoint.jsonl "
          "or --materialize-choices assignment.json "
          "or [beam-width] --warm-start prior-result.json");
    std::string outputPath = argv[3],
                checkpointPath = outputPath + ".checkpoint.jsonl",
                beamTracePath = outputPath + ".beam-trace.jsonl";
    if (std::filesystem::exists(outputPath) ||
        std::filesystem::exists(checkpointPath) ||
        std::filesystem::exists(beamTracePath))
      throw std::runtime_error(
          "refusing to overwrite existing search evidence");
    auto start = std::chrono::steady_clock::now();
    auto sourceJSON = readJSON(argv[1]);
    bool sourceOnly = std::string(argv[2]) == "--check-source";
    auto catalogue = sourceOnly ? j::Value(nullptr) : readJSON(argv[2]);
    std::shared_ptr<ProgramOperatorSource> source;
    std::string error;
    if (!parseProgramOperatorSource(sourceJSON, source, error))
      throw std::runtime_error(error);
    std::map<std::string, std::string> verified;
    for (const auto &binding : source->bindings) {
      auto found = verified.find(binding.file);
      if (found == verified.end()) {
        auto buffer = llvm::MemoryBuffer::getFile(binding.file);
        if (!buffer)
          throw std::runtime_error("source binding file missing: " +
                                   binding.file);
        auto hash = llvm::SHA256::hash(
            llvm::arrayRefFromStringRef((*buffer)->getBuffer()));
        found = verified.emplace(binding.file, llvm::toHex(hash, true)).first;
      }
      if (found->second != binding.sha256)
        throw std::runtime_error("actual external source file hash differs: " +
                                 binding.file);
    }
    if (sourceOnly) {
      saveJSON(
          outputPath,
          j::Object{{"source_id", source->sourceId},
                    {"source_operators", int64_t(source->operators.size())},
                    {"values", int64_t(source->values.size())},
                    {"actual_source_files_verified", int64_t(verified.size())},
                    {"typed_source_checked", true}});
      return 0;
    }
    std::vector<ProgramOperatorImplementation> implementations;
    if (!parseProgramOperatorImplementations(*source, catalogue,
                                             implementations, error))
      throw std::runtime_error(error);
    std::optional<ProgramOperatorInvocationControlModel> invocationControlModel;
    if (!parseProgramOperatorInvocationControlModel(
            *source, catalogue, invocationControlModel, error))
      throw std::runtime_error(error);
    std::map<std::string, size_t> implementationIds;
    for (size_t i = 0; i < implementations.size(); ++i)
      implementationIds.emplace(implementations[i].id, i);
    auto *root = catalogue.getAsObject();
    auto *groupJSON = root ? root->getArray("choice_groups") : nullptr;
    if (!groupJSON)
      throw std::runtime_error(
          "catalogue requires actual local implementation choice_groups");
    std::vector<ProgramOperatorChoiceGroup> groups;
    for (const auto &value : *groupJSON) {
      auto *object = value.getAsObject();
      ProgramOperatorChoiceGroup group;
      auto id = object ? object->getString("id") : std::nullopt;
      auto *choices = object ? object->getArray("choices") : nullptr;
      if (!id || !choices)
        throw std::runtime_error("invalid local choice group");
      group.id = id->str();
      for (const auto &entry : *choices) {
        auto *choice = entry.getAsObject();
        auto cid = choice ? choice->getString("id") : std::nullopt;
        auto *bindings = choice ? choice->getArray("implementations") : nullptr;
        if (!cid || !bindings)
          throw std::runtime_error("invalid local compound choice");
        group.choiceIds.push_back(cid->str());
        std::vector<size_t> indices;
        for (const auto &binding : *bindings) {
          auto name = binding.getAsString();
          if (!name || !implementationIds.count(name->str()))
            throw std::runtime_error("unknown local implementation id");
          indices.push_back(implementationIds.at(name->str()));
        }
        group.choices.push_back(std::move(indices));
      }
      groups.push_back(std::move(group));
    }
    if (root->get("execution_policy") &&
        root->getString("execution_policy") != "source-dependency-ready" &&
        root->getString("execution_policy") != "synchronous-source-order")
      throw std::runtime_error("unknown catalogue execution policy");
    const bool readyMapping =
        root->getString("execution_policy") == "source-dependency-ready";
    j::Value certificate = nullptr;
    if (!certifyProgramOperatorChoices(*source, implementations, groups,
                                       certificate, error, readyMapping))
      throw std::runtime_error(error);
    const auto certificateIdentity = staticFingerprint(certificate);
    saveJSON(outputPath + ".certificate.json", certificate);
    StaticConfig architecture;
    if (!target(catalogue, architecture, error))
      throw std::runtime_error(error);
    const auto *application = root->getObject("fixed_application_invocation");
    auto applicationPrice = application ? application->getNumber("wall_ms")
                                        : std::optional<double>{};
    if (applicationPrice &&
        (!std::isfinite(*applicationPrice) || *applicationPrice < 0))
      throw std::runtime_error("invalid fixed application invocation price");
    struct OptionCost {
      double price = 0;
      bool known = true;
      int64_t invocations = 0, launches = 0;
    };
    std::vector<std::vector<OptionCost>> localCosts;
    ProgramSearchSpace space;
    std::vector<size_t> variableGroups;
    std::map<size_t, size_t> dimensionForGroup;
    for (size_t g = 0; g < groups.size(); ++g) {
      std::vector<OptionCost> costs;
      for (const auto &choice : groups[g].choices) {
        OptionCost cost;
        for (size_t index : choice) {
          const auto &impl = implementations[index];
          if (impl.wallMs)
            cost.price += *impl.wallMs;
          else
            cost.known = false;
          ++cost.invocations;
          cost.launches += impl.launchCount;
        }
        if (!std::isfinite(cost.price))
          throw std::runtime_error("local cost overflow");
        costs.push_back(cost);
      }
      localCosts.push_back(std::move(costs));
      if (groups[g].choices.size() > 1) {
        dimensionForGroup[g] = space.dimensions.size();
        variableGroups.push_back(g);
        ProgramSearchDimension dimension;
        dimension.id = groups[g].id;
        dimension.scope = "source-certified-local-implementation-cover";
        dimension.status = "searched";
        for (const auto &id : groups[g].choiceIds) {
          ProgramSearchChoice choice;
          choice.id = id;
          dimension.choices.push_back(std::move(choice));
        }
        space.dimensions.push_back(std::move(dimension));
      }
    }
    const bool jointMapping =
        root->get("execution_registration_identity") != nullptr;
    std::string executionRegistration;
    std::map<std::string, ProgramOperatorExecutionVariant> executionVariants;
    if (jointMapping && !parseProgramOperatorMappingRegistrations(
                            *source, catalogue, implementations,
                            executionRegistration, executionVariants, error))
      throw std::runtime_error(error);
    for (const auto &entry : executionVariants)
      if ((entry.second.executionPolicy == "source-dependency-ready") !=
          readyMapping)
        throw std::runtime_error(
            "catalogue and variant execution policy disagree");
    std::shared_ptr<const ProgramOperatorMappingDomain> mappingDomain;
    if (jointMapping &&
        !prepareProgramOperatorMappingDomain(
            *source, implementations, groups, executionVariants,
            executionRegistration, architecture, "runtime-abi", mappingDomain,
            error, invocationControlModel ? &*invocationControlModel : nullptr))
      throw std::runtime_error(error);
    std::vector<size_t> fixedPriorityKeys;
    std::string fixedPriorityIdentity = "named-ready-priority-seed";
    auto refreshPriorityIdentity = [&]() {
      j::Array keys;
      for (size_t key : fixedPriorityKeys)
        keys.push_back(int64_t(key));
      fixedPriorityIdentity = staticFingerprint(j::Value(std::move(keys)));
    };
    std::vector<std::string> priorityPolicies{"source-order"};
    size_t priorityDimension = SIZE_MAX;
    if (readyMapping) {
      if (!jointMapping)
        throw std::runtime_error(
            "ready execution requires registered backend variants");
      auto *policies = root->getArray("ready_priority_policies");
      if (!policies || policies->empty())
        throw std::runtime_error("ready catalogue must explicitly register "
                                 "finite priority policies");
      priorityPolicies.clear();
      ProgramSearchDimension dimension;
      dimension.id = "runtime.ready-priority";
      dimension.scope = "registered-async-ready-order";
      dimension.status = "searched";
      std::set<std::string> unique;
      for (const auto &policy : *policies) {
        auto name = policy.getAsString();
        if (!name || !unique.insert(name->str()).second)
          throw std::runtime_error("invalid duplicate priority policy");
        ProgramOperatorSelection empty;
        ProgramOperatorMapping probe;
        if (!setProgramOperatorReadyPriority(empty, probe, *name, error))
          throw std::runtime_error(error);
        priorityPolicies.push_back(name->str());
        ProgramSearchChoice choice;
        choice.id = name->str();
        dimension.choices.push_back(std::move(choice));
      }
      priorityDimension = space.dimensions.size();
      space.dimensions.push_back(std::move(dimension));
    }
    auto selectedPriority =
        [&](const std::vector<int64_t> &assignment) -> const std::string & {
      return priorityPolicies[priorityDimension == SIZE_MAX
                                  ? 0
                                  : size_t(assignment[priorityDimension])];
    };
    auto mappingFor = [&](const ProgramOperatorSelection &selected,
                          llvm::StringRef priority) {
      ProgramOperatorMapping mapping;
      mapping.registrationIdentity = executionRegistration;
      mapping.controlModel = invocationControlModel;
      mapping.hostMemory = "runtime-abi";
      for (const auto &impl : selected.regions) {
        mapping.readyPriority.push_back(mapping.variants.size());
        mapping.variants.push_back(executionVariants.at(impl.id));
      }
      if (!setProgramOperatorReadyPriority(selected, mapping, priority, error))
        throw std::runtime_error(error);
      if (!fixedPriorityKeys.empty())
        std::stable_sort(mapping.readyPriority.begin(),
                         mapping.readyPriority.end(), [&](size_t a, size_t b) {
                           return fixedPriorityKeys[implementationIds.at(
                                      selected.regions[a].id)] <
                                  fixedPriorityKeys[implementationIds.at(
                                      selected.regions[b].id)];
                         });
      return mapping;
    };
    ProgramSearchOptions options;
    options.wallSeconds = std::numeric_limits<double>::infinity();
    options.maxEvaluations = 0;
    // This source-certified finite catalogue determines model admission. Grow
    // the caller's structural quota without changing the public API default or
    // imposing an evaluation/time cutoff, and never drop registered choices.
    const size_t maximumCardinality =
        size_t(std::numeric_limits<int64_t>::max());
    if (space.dimensions.size() > maximumCardinality)
      throw std::runtime_error(
          "registered dimension cardinality exceeds int64");
    int64_t registeredChoices = 0;
    for (const auto &dimension : space.dimensions) {
      if (dimension.choices.size() >
          size_t(std::numeric_limits<int64_t>::max() - registeredChoices))
        throw std::runtime_error("registered choice cardinality exceeds int64");
      registeredChoices += int64_t(dimension.choices.size());
    }
    options.maxVariables =
        std::max(options.maxVariables,
                 std::max(int64_t(space.dimensions.size()), registeredChoices));
    auto admissionJSON = [&]() {
      return j::Object{
          {"registered_dimensions", int64_t(space.dimensions.size())},
          {"registered_dimension_choices", registeredChoices},
          {"caller_model_variable_quota", options.maxVariables},
          {"generic_api_default_changed", false},
          {"registered_choices_dropped", false}};
    };
    auto elapsed = [&]() {
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - start)
          .count();
    };
    if (materializeChoices) {
      // A baseline is a prescribed registered assignment, not a one-candidate
      // search: bypass beam, seed selection and all temporal neighborhoods.
      if (!mappingDomain || !applicationPrice)
        throw std::runtime_error("manual materialization requires registered "
                                 "mapping and fixed price");
      auto assignmentJSON = readJSON(argv[5]);
      const auto *manual = assignmentJSON.getAsObject();
      const auto *entries = manual ? manual->getArray("choices") : nullptr;
      const auto policy =
          manual ? manual->getString("ready_priority_policy") : std::nullopt;
      if (!manual ||
          manual->getString("schema") != "orbit.program-manual-choices.v1" ||
          manual->getString("source_id") != source->sourceId ||
          manual->getString("execution_registration_identity") !=
              executionRegistration ||
          manual->getString("purpose") != "manual-baseline" || !entries ||
          entries->size() != groups.size() || !policy ||
          std::find(priorityPolicies.begin(), priorityPolicies.end(),
                    *policy) == priorityPolicies.end())
        throw std::runtime_error("invalid or unbound manual choices contract");
      std::map<std::string, std::string> requested;
      for (const auto &entry : *entries) {
        const auto *object = entry.getAsObject();
        auto group = object ? object->getString("group_id") : std::nullopt;
        auto choice = object ? object->getString("choice_id") : std::nullopt;
        if (!group || !choice ||
            !requested.emplace(group->str(), choice->str()).second)
          throw std::runtime_error("invalid duplicate manual choice group");
      }
      std::vector<size_t> choices;
      ProgramOperatorSelection selected;
      for (const auto &group : groups) {
        auto found = requested.find(group.id);
        if (found == requested.end())
          throw std::runtime_error("manual choices omit a registered group");
        auto choice = std::find(group.choiceIds.begin(), group.choiceIds.end(),
                                found->second);
        if (choice == group.choiceIds.end())
          throw std::runtime_error("unknown manual registered choice ID");
        size_t index = size_t(choice - group.choiceIds.begin());
        choices.push_back(index);
        for (size_t implementation : group.choices[index])
          selected.regions.push_back(implementations[implementation]);
      }
      auto evaluation = evaluateProgramOperatorMappingDomain(*mappingDomain,
                                                             choices, *policy);
      if (!evaluation.feasible || !evaluation.feasibilityKnown ||
          !evaluation.costMs)
        throw std::runtime_error(
            "manual registered assignment is infeasible or unknown");
      std::sort(selected.regions.begin(), selected.regions.end(),
                [](const auto &a, const auto &b) {
                  return a.operators.front() < b.operators.front();
                });
      ProgramOperatorMappingCandidate candidate{selected,
                                                mappingFor(selected, *policy)};
      ProgramGraph graph;
      if (!canonicalizeProgramOperatorCandidate(*source, candidate, error) ||
          !materializeProgramOperatorGraph(source, candidate.selection,
                                           architecture, "runtime-abi", graph,
                                           error) ||
          !scheduleProgramOperatorGraph(graph, candidate.mapping, error))
        throw std::runtime_error(error);
      double controlMs;
      std::map<std::string, int64_t> features;
      if (!estimateProgramOperatorInvocationControl(*graph.operatorSelection,
                                                    invocationControlModel,
                                                    controlMs, features, error))
        throw std::runtime_error(error);
      if (!graph.estimatedMakespanMs ||
          std::abs(*graph.estimatedMakespanMs + controlMs -
                   *evaluation.costMs) > 1e-8)
        throw std::runtime_error(
            "manual projection and independently checked graph differ");
      j::Object featureJSON;
      for (const auto &feature : features)
        featureJSON[feature.first] = feature.second;
      saveJSON(outputPath,
               j::Object{
                   {"schema", "orbit.program-operator-search.v1"},
                   {"mode", "manual-materialization"},
                   {"purpose", "manual-baseline"},
                   {"algorithm", "none-prescribed-registered-assignment"},
                   {"search_performed", false},
                   {"source_id", source->sourceId},
                   {"execution_registration_identity", executionRegistration},
                   {"actual_source_files_verified", int64_t(verified.size())},
                   {"choice_certificate", certificateIdentity},
                   {"manual_assignment", std::move(assignmentJSON)},
                   {"selected_program_graph", serializeProgramGraph(graph)},
                   {"selected_cover_prediction",
                    summarizeProgramOperatorGraph(graph)},
                   {"selected_graph_independently_checked", true},
                   {"full_model_predicted_ms",
                    *evaluation.costMs + *applicationPrice},
                   {"cost_components",
                    j::Object{{"physical_callback_makespan_ms",
                               *graph.estimatedMakespanMs},
                              {"invocation_control_ms", controlMs},
                              {"other_fixed_application_ms", *applicationPrice},
                              {"controller_features", std::move(featureJSON)},
                              {"controller_is_approximate",
                               invocationControlModel.has_value()}}},
                   {"invocation_control_model",
                    root->get("invocation_control_model")
                        ? copyJSONWithoutFields(
                              *root->get("invocation_control_model"))
                        : j::Value(nullptr)},
                   {"global_optimality_proved", false},
                   {"total_wall_ms", elapsed()}});
      return 0;
    }
    if (auditCheckpoints) {
      if (!mappingDomain)
        throw std::runtime_error(
            "checkpoint audit requires registered execution mapping");
      std::ifstream input(argv[5]);
      if (!input)
        throw std::runtime_error("cannot read checkpoint input");
      int64_t checked = 0, capturedCostsCompared = 0;
      double fastMs = 0, referenceMs = 0, maxError = 0;
      std::string line;
      for (; checked < 100 && std::getline(input, line); ++checked) {
        auto parsed = j::parse(line);
        if (!parsed) {
          llvm::consumeError(parsed.takeError());
          throw std::runtime_error("invalid checkpoint line");
        }
        auto *row = parsed->getAsObject();
        auto *values = row ? row->getArray("assignment") : nullptr;
        if (row && row->getString("priority_key_identity") &&
            row->getString("priority_key_identity") !=
                "named-ready-priority-seed")
          throw std::runtime_error(
              "checkpoint audit requires its learned priority-state recipe");
        if (!values || values->size() != space.dimensions.size())
          throw std::runtime_error("checkpoint assignment dimension mismatch");
        std::vector<int64_t> capturedAssignment;
        for (size_t d = 0; d < values->size(); ++d) {
          auto value = (*values)[d].getAsInteger();
          if (!value || *value < 0 ||
              size_t(*value) >= space.dimensions[d].choices.size())
            throw std::runtime_error(
                "checkpoint assignment is outside registered domain");
          capturedAssignment.push_back(*value);
        }
        std::vector<size_t> choices(groups.size());
        for (size_t g = 0; g < groups.size(); ++g) {
          auto dim = dimensionForGroup.find(g);
          if (dim == dimensionForGroup.end())
            continue;
          auto n = (*values)[dim->second].getAsInteger();
          if (!n || *n < 0 || size_t(*n) >= groups[g].choices.size())
            throw std::runtime_error("checkpoint choice out of bounds");
          choices[g] = size_t(*n);
        }
        auto begin = std::chrono::steady_clock::now();
        auto fast = evaluateProgramOperatorMappingDomain(
            *mappingDomain, choices, selectedPriority(capturedAssignment));
        auto middle = std::chrono::steady_clock::now();
        auto reference = evaluateProgramOperatorMappingDomainReference(
            *mappingDomain, choices, selectedPriority(capturedAssignment));
        auto end = std::chrono::steady_clock::now();
        fastMs +=
            std::chrono::duration<double, std::milli>(middle - begin).count();
        referenceMs +=
            std::chrono::duration<double, std::milli>(end - middle).count();
        if (fast.feasible != reference.feasible ||
            fast.feasibilityKnown != reference.feasibilityKnown ||
            fast.costMs.has_value() != reference.costMs.has_value())
          throw std::runtime_error(
              "fast/reference feasibility or pricing coverage differs");
        if (fast.costMs) {
          double error = std::abs(*fast.costMs - *reference.costMs);
          maxError = std::max(maxError, error);
          if (error > 1e-8)
            throw std::runtime_error("fast/reference cost differs");
          auto captured = row->getNumber("cost_ms");
          if (captured && applicationPrice) {
            ++capturedCostsCompared;
            if (std::abs(*captured - *fast.costMs - *applicationPrice) > 1e-8)
              throw std::runtime_error(
                  "optimized cost differs from captured actual assignment");
          }
        }
      }
      if (checked < 100)
        throw std::runtime_error(
            "audit needs at least 100 captured assignments");
      saveJSON(outputPath,
               j::Object{{"schema", "orbit.program-operator-calendar-audit.v1"},
                         {"assignments_checked", checked},
                         {"registered_domain_admission", admissionJSON()},
                         {"source_id", source->sourceId},
                         {"fast_ms", fastMs},
                         {"reference_ms", referenceMs},
                         {"max_cost_error_ms", maxError},
                         {"feasibility_and_unknown_coverage_equal", true},
                         {"captured_costs_compared", capturedCostsCompared},
                         {"actual_captured_costs_checked",
                          capturedCostsCompared == checked},
                         {"total_wall_ms", elapsed()}});
      return 0;
    }
    ProgramBeamInitialStates initialStates;
    j::Value warmStartEvidence = nullptr;
    if (warmStartPath) {
      if (!readyMapping || !jointMapping)
        throw std::runtime_error(
            "cover warm start requires registered ready mapping");
      // Only source/implementation identities are carried. In particular,
      // neither prior costs, old group indices nor learned ready ranks enter
      // this run.
      auto priorBuffer = llvm::MemoryBuffer::getFile(warmStartPath);
      if (!priorBuffer)
        throw std::runtime_error("cannot read warm start result");
      auto parsedPrior = j::parse((*priorBuffer)->getBuffer());
      if (!parsedPrior) {
        llvm::consumeError(parsedPrior.takeError());
        throw std::runtime_error("invalid warm start result JSON");
      }
      auto priorJSON = std::move(*parsedPrior);
      const auto *prior = priorJSON.getAsObject();
      const auto *priorGraph =
          prior ? prior->getObject("selected_program_graph") : nullptr;
      const auto *priorSelection =
          priorGraph
              ? priorGraph->getObject("external_implementation_selection")
              : nullptr;
      const auto *regions =
          priorSelection ? priorSelection->getArray("regions") : nullptr;
      if (!prior ||
          prior->getBoolean("selected_graph_independently_checked") != true ||
          prior->getString("source_id") != source->sourceId ||
          (prior->getString("algorithm") !=
               "ORBIT-beam-plus-joint-cover-ready-neighborhood" &&
           prior->getString("algorithm") !=
               "existing-ORBIT-program-beam-search") ||
          !regions || regions->empty())
        throw std::runtime_error(
            "warm start requires a prior checked ORBIT result on this source");
      std::set<std::string> selectedIds;
      std::vector<bool> selectedSource(source->operators.size(), false);
      for (const auto &region : *regions) {
        const auto *old = region.getAsObject();
        auto id = old ? old->getString("id") : std::nullopt;
        if (!id || !selectedIds.insert(id->str()).second ||
            !implementationIds.count(id->str()))
          throw std::runtime_error(
              "prior implementation ID is missing or duplicated in new domain");
        const auto &implementation =
            implementations[implementationIds.at(id->str())];
        ProgramOperatorSelection single;
        single.regions.push_back(implementation);
        auto currentJSON = serializeProgramOperatorSelection(single);
        const auto *current = currentJSON.getAsObject()
                                  ->getArray("regions")
                                  ->front()
                                  .getAsObject();
        for (const char *field :
             {"source_id", "artifact_identity", "entrypoint", "device",
              "resource", "memory", "source_operators", "inputs", "outputs",
              "launch_count", "cost_boundary", "transport_ownership",
              "visibility_policy", "input_locations", "output_locations"}) {
          const auto *oldValue = old->get(field),
                     *newValue = current->get(field);
          if (!oldValue || !newValue || !(*oldValue == *newValue))
            throw std::runtime_error(
                "warm implementation source/artifact/ABI identity changed: " +
                id->str());
        }
        for (size_t op : implementation.operators) {
          if (selectedSource[op])
            throw std::runtime_error(
                "prior registered implementations overlap source instances");
          selectedSource[op] = true;
        }
      }
      if (std::find(selectedSource.begin(), selectedSource.end(), false) !=
          selectedSource.end())
        throw std::runtime_error(
            "prior implementations do not cover the complete current source");
      std::vector<size_t> sourceGroup(source->operators.size(), SIZE_MAX);
      for (size_t g = 0; g < groups.size(); ++g)
        for (size_t index : groups[g].choices.front())
          for (size_t op : implementations[index].operators)
            sourceGroup[op] = g;
      std::vector<std::set<std::string>> required(groups.size());
      for (const auto &id : selectedIds) {
        const auto &implementation = implementations[implementationIds.at(id)];
        size_t g = sourceGroup[implementation.operators.front()];
        if (g == SIZE_MAX ||
            std::any_of(implementation.operators.begin(),
                        implementation.operators.end(),
                        [&](size_t op) { return sourceGroup[op] != g; }))
          throw std::runtime_error(
              "prior cover crosses incompatible new component boundaries");
        required[g].insert(id);
      }
      std::vector<int64_t> assignment(space.dimensions.size(), 0);
      j::Array mappedChoices;
      for (size_t g = 0; g < groups.size(); ++g) {
        size_t match = SIZE_MAX;
        for (size_t c = 0; c < groups[g].choices.size(); ++c) {
          std::set<std::string> ids;
          for (size_t index : groups[g].choices[c])
            ids.insert(implementations[index].id);
          if (ids == required[g]) {
            if (match != SIZE_MAX)
              throw std::runtime_error(
                  "prior cover matches ambiguous new choice IDs");
            match = c;
          }
        }
        if (match == SIZE_MAX)
          throw std::runtime_error(
              "prior implementation cover has no exact new registered choice");
        auto dimension = dimensionForGroup.find(g);
        if (dimension != dimensionForGroup.end())
          assignment[dimension->second] = int64_t(match);
        mappedChoices.push_back(
            j::Object{{"group_id", groups[g].id},
                      {"choice_id", groups[g].choiceIds[match]}});
      }
      for (size_t priority = 0; priority < priorityPolicies.size();
           ++priority) {
        assignment[priorityDimension] = int64_t(priority);
        initialStates.assignments.push_back(assignment);
      }
      const auto hash = llvm::SHA256::hash(
          llvm::arrayRefFromStringRef((*priorBuffer)->getBuffer()));
      warmStartEvidence =
          j::Object{{"prior_result_file", warmStartPath},
                    {"prior_result_sha256", llvm::toHex(hash, true)},
                    {"source_id", source->sourceId},
                    {"mapped_choices", std::move(mappedChoices)},
                    {"initial_state_proposals",
                     int64_t(initialStates.assignments.size())},
                    {"historical_costs_reused", false},
                    {"old_assignment_indices_used", false},
                    {"prior_ready_priority_ignored", true},
                    {"priority_scope",
                     "same prior cover re-evaluated under every registered "
                     "named seed; general ready-priority search follows"},
                    {"domain_pruned", false}};
    }
    std::ofstream checkpoint(checkpointPath);
    if (!checkpoint)
      throw std::runtime_error("cannot create checkpoint");
    int64_t evaluations = 0;
    std::string evaluationPhase = "coordinate-initialization";
    std::optional<double> checkpointBestCost;
    auto evaluate = [&](const std::vector<int64_t> &assignment) {
      ++evaluations;
      double price = applicationPrice.value_or(0);
      bool known = applicationPrice.has_value();
      int64_t launches = 0, invocations = 0;
      for (size_t g = 0; g < groups.size(); ++g) {
        auto dimension = dimensionForGroup.find(g);
        size_t choice = dimension == dimensionForGroup.end()
                            ? 0
                            : size_t(assignment[dimension->second]);
        const auto &cost = localCosts[g][choice];
        price += cost.price;
        known &= cost.known;
        launches += cost.launches;
        invocations += cost.invocations;
      }
      ProgramSearchEvaluation checkedMapping;
      if (jointMapping) {
        std::vector<size_t> groupChoices;
        for (size_t g = 0; g < groups.size(); ++g) {
          auto dimension = dimensionForGroup.find(g);
          groupChoices.push_back(dimension == dimensionForGroup.end()
                                     ? 0
                                     : size_t(assignment[dimension->second]));
        }
        checkedMapping = evaluateProgramOperatorMappingDomain(
            *mappingDomain, groupChoices, selectedPriority(assignment),
            fixedPriorityKeys.empty() ? nullptr : &fixedPriorityKeys);
        known = checkedMapping.feasible && checkedMapping.costMs.has_value() &&
                applicationPrice.has_value();
        price =
            checkedMapping.costMs.value_or(0) + applicationPrice.value_or(0);
      }
      if (!std::isfinite(price))
        throw std::runtime_error("candidate cost overflow");
      bool improved =
          known && (!checkpointBestCost || price < *checkpointBestCost);
      if (improved)
        checkpointBestCost = price;
      j::Array indices;
      for (auto n : assignment)
        indices.push_back(n);
      j::Object row{{"evaluation", evaluations},
                    {"evaluation_phase", evaluationPhase},
                    {"elapsed_ms", elapsed()},
                    {"assignment", std::move(indices)},
                    {"cost_ms", known ? j::Value(price) : j::Value(nullptr)},
                    {"improved", improved},
                    {"source_id", source->sourceId},
                    {"choice_certificate", certificateIdentity},
                    {"priority_key_identity", fixedPriorityIdentity},
                    {"priority_seed", selectedPriority(assignment)},
                    {"feasible", jointMapping ? checkedMapping.feasible : true},
                    {"feasibility_known",
                     jointMapping ? checkedMapping.feasibilityKnown : true},
                    {"mapping_evidence",
                     jointMapping ? copyJSONWithoutFields(checkedMapping.result)
                                  : j::Value(nullptr)}};
      std::string line;
      llvm::raw_string_ostream stream(line);
      stream << j::Value(std::move(row));
      stream.flush();
      checkpoint << line << "\n";
      if (improved || evaluations % 128 == 0)
        checkpoint.flush();
      if (improved) {
        std::ofstream best(outputPath + ".best-checkpoint.json");
        best << line << "\n";
      }
      ProgramSearchEvaluation evaluation;
      if (jointMapping) {
        evaluation = std::move(checkedMapping);
        if (known)
          evaluation.costMs = price;
        else
          evaluation.costMs.reset();
        return evaluation;
      }
      evaluation.feasible = true;
      if (known)
        evaluation.costMs = price;
      evaluation.result =
          j::Object{{"source_id", source->sourceId},
                    {"choice_certificate", certificateIdentity},
                    {"actual_implementation_invocations", invocations},
                    {"actual_npu_launches", launches},
                    {"legality", "certified product of exact source covers "
                                 "with invariant cross-group ABI"}};
      return evaluation;
    };
    const int64_t width =
        (argc == 5 || (warmStartPath && argc == 7)) ? std::stoll(argv[4]) : 4;
    std::ofstream beamTrace;
    int64_t traceRows = 0;
    ProgramBeamTraceOptions traceOptions;
    if (jointMapping) {
      beamTrace.open(beamTracePath);
      if (!beamTrace)
        throw std::runtime_error("cannot create complete beam trace");
      traceOptions.retainInMemory = false;
      traceOptions.sink = [&](const j::Value &row) {
        std::string line;
        llvm::raw_string_ostream stream(line);
        stream << row;
        stream.flush();
        beamTrace << line << "\n";
        ++traceRows;
        if (traceRows % 128 == 0)
          beamTrace.flush();
        if (!beamTrace)
          throw std::runtime_error("complete beam trace write failed");
      };
    }
    ProgramCoordinateInitialization coordinateInitialization;
    auto coordinateStart = std::chrono::steady_clock::now();
    if (jointMapping) {
      coordinateInitialization = initializeProgramCoordinates(
          space, evaluate, initialStates.assignments,
          [&](const std::vector<int64_t> &assignment,
              const ProgramSearchEvaluation &evaluation) {
            j::Array indices;
            for (int64_t value : assignment)
              indices.push_back(value);
            traceOptions.sink(j::Object{
                {"kind", "coordinate-initialization"},
                {"evaluation", evaluations},
                {"assignment", std::move(indices)},
                {"cost_ms", evaluation.costMs ? j::Value(*evaluation.costMs)
                                              : j::Value(nullptr)},
                {"feasible", evaluation.feasible},
                {"feasibility_known", evaluation.feasibilityKnown}});
          });
      if (!coordinateInitialization.valid)
        throw std::runtime_error(coordinateInitialization.message);
      if (coordinateInitialization.hasPricedSeed)
        initialStates.assignments.push_back(
            coordinateInitialization.assignment);
    }
    double coordinateMs =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - coordinateStart)
            .count();
    evaluationPhase = "beam";
    auto result = searchProgramSpaceBeam(space, evaluate, options, width, -1, 0,
                                         {}, traceOptions, initialStates);
    evaluationPhase = "joint-cover-neighborhood";
    int64_t temporalEvaluations = 0, alternatingPasses = 0;
    int64_t comparablePrioritySwapsSkipped = 0, temporalImprovements = 0;
    const double improvementToleranceMs = 1e-9;
    const std::string actionTracePath = outputPath + ".joint-actions.jsonl";
    auto jointStart = std::chrono::steady_clock::now();
    const bool jointNeighborhoodExecuted =
        readyMapping && result.hasBest &&
        result.bestEvaluation.costMs.has_value();
    if (jointNeighborhoodExecuted) {
      if (std::filesystem::exists(actionTracePath))
        throw std::runtime_error("refusing to overwrite joint action evidence");
      std::ofstream actions(actionTracePath);
      if (!actions)
        throw std::runtime_error("cannot create joint action evidence");
      int64_t state = 0;
      auto emitAction = [&](j::Object row) {
        row["elapsed_ms"] = elapsed();
        std::string line;
        llvm::raw_string_ostream stream(line);
        stream << j::Value(std::move(row));
        stream.flush();
        actions << line << "\n";
        if (!actions)
          throw std::runtime_error("joint action trace write failed");
      };
      auto indicesJSON = [](const auto &values) {
        j::Array row;
        for (auto v : values)
          row.push_back(int64_t(v));
        return row;
      };
      auto groupChoicesFor = [&](const std::vector<int64_t> &assignment) {
        std::vector<size_t> choices(groups.size());
        for (size_t g = 0; g < groups.size(); ++g) {
          auto d = dimensionForGroup.find(g);
          if (d != dimensionForGroup.end())
            choices[g] = size_t(assignment[d->second]);
        }
        return choices;
      };
      while (true) {
        ++alternatingPasses;
        auto choices = groupChoicesFor(result.bestAssignment);
        std::vector<size_t> selectedIndices;
        for (size_t g = 0; g < groups.size(); ++g)
          for (size_t i : groups[g].choices[choices[g]])
            selectedIndices.push_back(i);
        std::sort(selectedIndices.begin(), selectedIndices.end(),
                  [&](size_t a, size_t b) {
                    return implementations[a].operators.front() <
                           implementations[b].operators.front();
                  });
        ProgramOperatorSelection selected;
        for (size_t i : selectedIndices)
          selected.regions.push_back(implementations[i]);
        auto mapping =
            mappingFor(selected, selectedPriority(result.bestAssignment));
        if (!configureProgramOperatorExecutionPolicy(selected, mapping, error))
          throw std::runtime_error(error);
        std::vector<std::vector<size_t>> predecessors;
        if (!programOperatorInvocationDependencies(*source, selected,
                                                   predecessors, error))
          throw std::runtime_error(error);
        auto priority = mapping.readyPriority;
        emitAction(j::Object{
            {"kind", "priority-state"},
            {"state", state},
            {"assignment", indicesJSON(result.bestAssignment)},
            {"selected_catalogue_indices", indicesJSON(selectedIndices)},
            {"ready_priority", indicesJSON(priority)}});
        auto comparable = [&](size_t a, size_t b) {
          auto reaches = [&](size_t from, size_t target) {
            std::vector<bool> visited(predecessors.size());
            std::vector<size_t> pending{from};
            while (!pending.empty()) {
              auto event = pending.back();
              pending.pop_back();
              if (event == target)
                return true;
              if (visited[event])
                continue;
              visited[event] = true;
              pending.insert(pending.end(), predecessors[event].begin(),
                             predecessors[event].end());
            }
            return false;
          };
          return reaches(a, b) || reaches(b, a);
        };
        // Source rank is only a candidate priority key. It never changes source
        // semantics. A different cover inherits the minimum rank of its source
        // members, so structural choices remain coupled to temporal search.
        auto keysFor = [&](const std::vector<size_t> &order) {
          std::vector<size_t> ranks(source->operators.size(), SIZE_MAX);
          for (size_t rank = 0; rank < order.size(); ++rank)
            for (size_t op : selected.regions[order[rank]].operators)
              ranks[op] = rank;
          std::vector<size_t> keys(implementations.size(), SIZE_MAX);
          for (size_t i = 0; i < implementations.size(); ++i)
            for (size_t op : implementations[i].operators)
              keys[i] = std::min(keys[i], ranks[op]);
          return keys;
        };
        fixedPriorityKeys = keysFor(priority);
        refreshPriorityIdentity();
        emitAction(j::Object{
            {"kind", "priority-key-state"},
            {"state", state},
            {"priority_key_identity", fixedPriorityIdentity},
            {"implementation_priority_keys", indicesJSON(fixedPriorityKeys)}});
        while (true) {
          std::optional<std::vector<size_t>> bestOrder, bestKeys;
          ProgramSearchEvaluation bestEvaluation;
          double bestCost = *result.bestEvaluation.costMs;
          for (size_t position = 1; position < priority.size(); ++position) {
            if (comparable(priority[position - 1], priority[position])) {
              ++comparablePrioritySwapsSkipped;
              continue;
            }
            auto neighbor = priority;
            std::swap(neighbor[position - 1], neighbor[position]);
            auto keys = keysFor(neighbor);
            auto evaluation = evaluateProgramOperatorMappingDomain(
                *mappingDomain, choices, "source-order", &keys);
            ++temporalEvaluations;
            if (evaluation.costMs && applicationPrice)
              *evaluation.costMs += *applicationPrice;
            else
              evaluation.costMs.reset();
            emitAction(j::Object{
                {"kind", "adjacent-priority-swap"},
                {"parent_state", state},
                {"position", int64_t(position)},
                {"feasible", evaluation.feasible},
                {"feasibility_known", evaluation.feasibilityKnown},
                {"cost_ms", evaluation.costMs ? j::Value(*evaluation.costMs)
                                              : j::Value(nullptr)}});
            if (evaluation.feasible && evaluation.feasibilityKnown &&
                evaluation.costMs &&
                *evaluation.costMs < bestCost - improvementToleranceMs) {
              bestCost = *evaluation.costMs;
              bestOrder = std::move(neighbor);
              bestKeys = std::move(keys);
              bestEvaluation = std::move(evaluation);
            }
          }
          if (!bestOrder)
            break;
          priority = std::move(*bestOrder);
          fixedPriorityKeys = std::move(*bestKeys);
          refreshPriorityIdentity();
          result.bestEvaluation = std::move(bestEvaluation);
          checkpointBestCost = result.bestEvaluation.costMs;
          ++state;
          ++temporalImprovements;
          emitAction(j::Object{
              {"kind", "accepted-priority"},
              {"state", state},
              {"ready_priority", indicesJSON(priority)},
              {"priority_key_identity", fixedPriorityIdentity},
              {"implementation_priority_keys", indicesJSON(fixedPriorityKeys)},
              {"cost_ms", *result.bestEvaluation.costMs}});
        }
        // Revisit every one-group implementation neighbor with the learned
        // priority. No iteration/time quota substitutes for convergence.
        std::optional<std::vector<int64_t>> bestAssignment;
        ProgramSearchEvaluation bestCoverEvaluation;
        double bestCoverCost = *result.bestEvaluation.costMs;
        for (const auto &dimension : dimensionForGroup) {
          size_t g = dimension.first, d = dimension.second;
          for (size_t c = 0; c < groups[g].choices.size(); ++c) {
            if (int64_t(c) == result.bestAssignment[d])
              continue;
            auto neighbor = result.bestAssignment;
            neighbor[d] = int64_t(c);
            auto evaluation = evaluate(neighbor);
            traceOptions.sink(j::Object{
                {"kind", "joint-cover-neighbor"},
                {"parent_priority_key_identity", fixedPriorityIdentity},
                {"assignment", indicesJSON(neighbor)},
                {"cost_ms", evaluation.costMs ? j::Value(*evaluation.costMs)
                                              : j::Value(nullptr)},
                {"feasible", evaluation.feasible},
                {"feasibility_known", evaluation.feasibilityKnown}});
            if (evaluation.feasible && evaluation.feasibilityKnown &&
                evaluation.costMs &&
                *evaluation.costMs < bestCoverCost - improvementToleranceMs) {
              bestCoverCost = *evaluation.costMs;
              bestAssignment = std::move(neighbor);
              bestCoverEvaluation = std::move(evaluation);
            }
          }
        }
        if (!bestAssignment)
          break;
        result.bestAssignment = std::move(*bestAssignment);
        result.bestEvaluation = std::move(bestCoverEvaluation);
        ++state;
        emitAction(j::Object{{"kind", "accepted-cover"},
                             {"state", state},
                             {"assignment", indicesJSON(result.bestAssignment)},
                             {"cost_ms", *result.bestEvaluation.costMs}});
      }
      actions.flush();
      if (!actions)
        throw std::runtime_error("joint action evidence flush failed");
    }
    checkpoint.flush();
    if (jointMapping) {
      beamTrace.flush();
      if (!beamTrace || traceRows != evaluations)
        throw std::runtime_error("beam trace lost evaluated assignments");
    }
    j::Object report{
        {"schema", "orbit.program-operator-search.v1"},
        {"source_id", source->sourceId},
        {"algorithm", readyMapping
                          ? "ORBIT-beam-plus-joint-cover-ready-neighborhood"
                          : "existing-ORBIT-program-beam-search"},
        {"evaluation_kind",
         jointMapping ? "certified-resource-calendar-plus-final-physical-check"
                      : "independent-local-cover-fees"},
        {"all_abstract_dimensions_supported", false},
        {"runtime_schedule_scope",
         readyMapping ? "source data/control/alias hazards; finite registered "
                        "async ready priorities"
                      : "registered synchronous invocation dependencies; "
                        "arbitrary overlap unsupported"},
        {"source_operators", int64_t(source->operators.size())},
        {"local_implementations", int64_t(implementations.size())},
        {"searched_dimensions", int64_t(space.dimensions.size())},
        {"actual_source_files_verified", int64_t(verified.size())},
        {"choice_certificate", certificateIdentity},
        {"report", std::move(result.report)},
        {"report_scope", readyMapping ? "initial ORBIT beam; final joint "
                                        "winner and additional work are below"
                                      : "complete synchronous ORBIT search"},
        {"source_proof_scope",
         "external source hashes plus typed tensor/control/state contracts; "
         "not native MLIR arithmetic proof"},
        {"capacity_scope",
         jointMapping
             ? "source ABI and registered managed BO/resident/scratch plus "
               "target SRAM capacities; unmanaged runtime RSS remains subject "
               "to the backend admission guard"
             : "ABI allocations checked on final graph; private/preparation "
               "admission belongs to registered runtime"}};
    j::Array coordinateAssignment;
    for (int64_t value : coordinateInitialization.assignment)
      coordinateAssignment.push_back(value);
    report["coordinate_initializer"] = j::Object{
        {"enabled", jointMapping},
        {"attempted", coordinateInitialization.attempted},
        {"completed", coordinateInitialization.completed},
        {"attempted_passes", coordinateInitialization.attempted ? 1 : 0},
        {"passes", coordinateInitialization.completed ? 1 : 0},
        {"reason", coordinateInitialization.message},
        {"evaluations", coordinateInitialization.evaluations},
        {"elapsed_ms", coordinateMs},
        {"improved_coordinates", coordinateInitialization.improvedCoordinates},
        {"has_priced_seed", coordinateInitialization.hasPricedSeed},
        {"start_cost_ms", coordinateInitialization.startCostMs
                              ? j::Value(*coordinateInitialization.startCostMs)
                              : j::Value(nullptr)},
        {"seed_cost_ms",
         coordinateInitialization.evaluation.costMs
             ? j::Value(*coordinateInitialization.evaluation.costMs)
             : j::Value(nullptr)},
        {"seed_assignment", std::move(coordinateAssignment)},
        {"search_stopping_rule", false},
        {"scope", "one joint exact-evaluator coordinate pass supplies a seed; "
                  "full uncapped beam and cover/priority closure follow"}};
    report["actual_cost_evaluations"] = evaluations + temporalEvaluations;
    report["actual_evaluation_breakdown"] = j::Object{
        {"coordinate_initialization", coordinateInitialization.evaluations},
        {"beam_and_cover_neighborhood",
         evaluations - coordinateInitialization.evaluations},
        {"ready_priority_neighborhood", temporalEvaluations}};
    report["joint_alternating_search"] = j::Object{
        {"available", readyMapping},
        {"enabled", jointNeighborhoodExecuted},
        {"passes", alternatingPasses},
        {"temporal_evaluations", temporalEvaluations},
        {"combined_evaluations", temporalEvaluations + evaluations},
        {"cover_evaluations_including_initial_beam", evaluations},
        {"temporal_improvements", temporalImprovements},
        {"comparable_adjacent_swaps_skipped", comparablePrioritySwapsSkipped},
        {"elapsed_ms", std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - jointStart)
                           .count()},
        {"improvement_tolerance_ms", improvementToleranceMs},
        {"stop_reason", !readyMapping ? "not applicable to synchronous runtime"
                        : !result.hasBest ? "no checked feasible incumbent"
                        : !result.bestEvaluation.costMs
                            ? "no priced feasible incumbent; numeric "
                              "improvement comparisons not run"
                            : "no improving adjacent ready-priority swap or "
                              "single-group cover replacement"},
        {"action_trace", jointNeighborhoodExecuted ? j::Value(actionTracePath)
                                                   : j::Value(nullptr)},
        {"implementation_priority_keys", [&]() {
           j::Array a;
           for (auto n : fixedPriorityKeys)
             a.push_back(int64_t(n));
           return a;
         }()}};
    if (result.hasBest) {
      ProgramOperatorSelection selected;
      for (size_t g = 0; g < groups.size(); ++g) {
        auto dimension = dimensionForGroup.find(g);
        size_t choice = dimension == dimensionForGroup.end()
                            ? 0
                            : size_t(result.bestAssignment[dimension->second]);
        for (size_t index : groups[g].choices[choice])
          selected.regions.push_back(implementations[index]);
      }
      std::sort(selected.regions.begin(), selected.regions.end(),
                [](const auto &a, const auto &b) {
                  return a.operators.front() < b.operators.front();
                });
      ProgramGraph graph;
      if (jointMapping) {
        ProgramOperatorMappingCandidate candidate{
            selected,
            mappingFor(selected, selectedPriority(result.bestAssignment))};
        if (!canonicalizeProgramOperatorCandidate(*source, candidate, error) ||
            !materializeProgramOperatorGraph(source, candidate.selection,
                                             architecture, "runtime-abi", graph,
                                             error) ||
            !scheduleProgramOperatorGraph(graph, candidate.mapping, error))
          throw std::runtime_error(error);
      } else if (!materializeProgramOperatorGraph(source, selected,
                                                  architecture, "runtime-abi",
                                                  graph, error))
        throw std::runtime_error(error);
      double controllerMs;
      std::map<std::string, int64_t> controllerFeatures;
      if (!estimateProgramOperatorInvocationControl(
              *graph.operatorSelection, invocationControlModel, controllerMs,
              controllerFeatures, error))
        throw std::runtime_error(error);
      j::Object featureReport;
      for (const auto &feature : controllerFeatures)
        featureReport[feature.first] = feature.second;
      report["cost_components"] = j::Object{
          {"physical_callback_makespan_ms",
           graph.estimatedMakespanMs ? j::Value(*graph.estimatedMakespanMs)
                                     : j::Value(nullptr)},
          {"invocation_control_ms", controllerMs},
          {"other_fixed_application_ms",
           applicationPrice ? j::Value(*applicationPrice) : j::Value(nullptr)},
          {"controller_features", std::move(featureReport)},
          {"controller_is_approximate", invocationControlModel.has_value()},
          {"controller_model_identity", invocationControlModel
                                            ? invocationControlModel->identity
                                            : "disabled"},
          {"controller_boundary", "outside-inclusive-invocation-wall"}};
      report["selected_cover_prediction"] =
          summarizeProgramOperatorGraph(graph);
      report["selected_program_graph"] = serializeProgramGraph(graph);
      report["selected_graph_independently_checked"] = true;
      if (jointMapping && result.bestEvaluation.costMs && applicationPrice &&
          (!graph.estimatedMakespanMs ||
           std::abs(*graph.estimatedMakespanMs + controllerMs +
                    *applicationPrice - *result.bestEvaluation.costMs) > 1e-8))
        throw std::runtime_error(
            "certified projection and final full physical graph cost differ");
    }
    if (jointMapping) {
      j::Object traceStorage;
      traceStorage["path"] = beamTracePath;
      traceStorage["rows"] = traceRows;
      traceStorage["retained_in_memory"] = false;
      traceStorage["format"] = "json-lines";
      traceStorage["complete"] = traceRows == evaluations;
      report["beam_trace"] = std::move(traceStorage);
    } else {
      report["beam_trace"] = nullptr;
    }
    report["registered_domain_admission"] = admissionJSON();
    report["cover_warm_start"] = std::move(warmStartEvidence);
    report["registered_search_scope"] =
        root->get("search_scope")
            ? copyJSONWithoutFields(*root->get("search_scope"))
            : j::Value(nullptr);
    report["fixed_application_invocation"] =
        application
            ? copyJSONWithoutFields(*root->get("fixed_application_invocation"))
            : j::Value(nullptr);
    report["invocation_control_model"] =
        root->get("invocation_control_model")
            ? copyJSONWithoutFields(*root->get("invocation_control_model"))
            : j::Value(nullptr);
    report["full_model_predicted_ms"] =
        result.bestEvaluation.costMs ? j::Value(*result.bestEvaluation.costMs)
                                     : j::Value(nullptr);
    report["structural_limits"] =
        j::Object{{"source_operators", int64_t(programOperatorMaxOperators)},
                  {"source_values", int64_t(programOperatorMaxValues)},
                  {"beam_width", width},
                  {"variables", options.maxVariables},
                  {"constraints", options.maxConstraints},
                  {"assignment_tuple_cells", options.maxTupleCells},
                  {"unknown_pending_states", 256}};
    report["global_optimality_proved"] = false;
    report["total_wall_ms"] = elapsed();
    saveJSON(outputPath, j::Value(std::move(report)));
    llvm::outs() << "wrote " << outputPath << " (" << evaluations
                 << " checked local-cover candidates)\n";
    return 0;
  } catch (const std::exception &e) {
    llvm::errs() << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
