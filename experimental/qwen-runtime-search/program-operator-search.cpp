// First-class operator source -> local implementation-cover ORBIT beam ->
// ProgramGraph. Actual external runtime replay is a separate checked adapter
// operation.
#include "Taskflow/JSONUtils.h"
#include "Taskflow/ProgramBeamSearch.h"
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
    if (argc != 4 && argc != 5)
      throw std::runtime_error("usage: program-operator-search source.json "
                               "implementations.json output.json [beam-width]");
    std::string outputPath = argv[3],
                checkpointPath = outputPath + ".checkpoint.jsonl";
    if (std::filesystem::exists(outputPath) ||
        std::filesystem::exists(checkpointPath))
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
    j::Value certificate = nullptr;
    if (!certifyProgramOperatorChoices(*source, implementations, groups,
                                       certificate, error))
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
    auto elapsed = [&]() {
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - start)
          .count();
    };
    std::ofstream checkpoint(checkpointPath);
    if (!checkpoint)
      throw std::runtime_error("cannot create checkpoint");
    int64_t evaluations = 0;
    std::optional<double> bestCost;
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
      if (!std::isfinite(price))
        throw std::runtime_error("candidate cost overflow");
      bool improved = known && (!bestCost || price < *bestCost);
      if (improved)
        bestCost = price;
      j::Array indices;
      for (auto n : assignment)
        indices.push_back(n);
      j::Object row{{"evaluation", evaluations},
                    {"elapsed_ms", elapsed()},
                    {"assignment", std::move(indices)},
                    {"cost_ms", known ? j::Value(price) : j::Value(nullptr)},
                    {"improved", improved},
                    {"source_id", source->sourceId},
                    {"choice_certificate", certificateIdentity}};
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
    ProgramSearchOptions options;
    options.wallSeconds = std::numeric_limits<double>::infinity();
    options.maxEvaluations = 0;
    const int64_t width = argc == 5 ? std::stoll(argv[4]) : 4;
    auto result = searchProgramSpaceBeam(space, evaluate, options, width, -1);
    checkpoint.flush();
    j::Object report{
        {"schema", "orbit.program-operator-search.v1"},
        {"source_id", source->sourceId},
        {"algorithm", "existing-ORBIT-program-beam-search"},
        {"source_operators", int64_t(source->operators.size())},
        {"local_implementations", int64_t(implementations.size())},
        {"searched_dimensions", int64_t(space.dimensions.size())},
        {"actual_source_files_verified", int64_t(verified.size())},
        {"choice_certificate", certificateIdentity},
        {"report", std::move(result.report)},
        {"source_proof_scope",
         "external source hashes plus typed tensor/control/state contracts; "
         "not native MLIR arithmetic proof"},
        {"capacity_scope",
         "ABI allocations checked on final graph; private/preparation "
         "admission belongs to registered runtime"}};
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
      if (!materializeProgramOperatorGraph(source, selected, architecture,
                                           "runtime-abi", graph, error))
        throw std::runtime_error(error);
      report["selected_cover_prediction"] =
          summarizeProgramOperatorGraph(graph);
      report["selected_program_graph"] = serializeProgramGraph(graph);
      report["selected_graph_independently_checked"] = true;
    }
    report["fixed_application_invocation"] =
        application
            ? copyJSONWithoutFields(*root->get("fixed_application_invocation"))
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
