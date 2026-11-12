#include "Taskflow/AmoebaMapping.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>

using namespace orbit::taskflow;
namespace {
bool fail(std::string &error, const std::string &message) {
  error = "Amoeba program graph: " + message;
  return false;
}
std::string endpoint(size_t task) {
  return "task." + std::to_string(task) + ".availability";
}
bool contract(const GenericAnalysis &source, const AmoebaTargetContract &target,
              const std::vector<AmoebaTaskShape> &shapes, StaticConfig &config,
              ProgramDataOrganizationChoice &choice, std::string &error) {
  if (target.fabricRows <= 0 || target.fabricColumns <= 0 ||
      target.fabricRows > INT64_MAX / target.fabricColumns ||
      target.peRows <= 0 || target.peColumns <= 0 ||
      target.peRows > INT64_MAX / target.peColumns || target.contextSlots < 0 ||
      target.architectureIdentity.empty() || target.networkIdentity.empty())
    return fail(error, "positive fabric/PE dimensions and explicit "
                       "architecture/network identities required");
  if (source.tasks.empty() || shapes.size() != source.tasks.size())
    return fail(error, "one shape per actual source task required");
  // This is a finite semantic endpoint graph, not an implicit fabric-size or
  // physical-capacity limit. Prevent accidental quadratic host expansion.
  if (source.tasks.size() > 1024)
    return fail(error, "abstract endpoint host-plan expansion guard exceeded");
  for (const auto &storage : source.storages) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(storage.root.getType());
    if (!type || !type.getElementType().isIntOrFloat() ||
        type.getElementType().getIntOrFloatBitWidth() % 8 != 0)
      return fail(error, "byte-exact memref element types required for network "
                         "payload binding");
  }
  for (const auto &task : source.tasks)
    for (const auto &access : task.accesses)
      if (access.storage >= source.storages.size())
        return fail(error, "invalid source storage index");

  config = {};
  choice = {};
  choice.partitionExactRegions = true;
  config.domains = {{"amoeba", "", "device"},
                    {"amoeba.fabric", "amoeba", "fabric"},
                    {"amoeba.external", "", "external"}};
  config.resources = {
      {"fabric.compute", "compute", "amoeba",
       target.fabricRows * target.fabricColumns, "amoeba.fabric"},
      {"intertask.network", "transfer", "amoeba", 1, "amoeba.fabric"},
      {"external.supply", "transfer", "external", 1, "amoeba.external"}};
  auto memory = [&](const std::string &id, const std::string &domain,
                    const std::string &kind) {
    StaticMemory entry;
    entry.id = id;
    // The integer is an overflow/accounting guard only, never a capacity fact.
    entry.capacityBytes = INT64_MAX;
    entry.capacityKnown = false;
    entry.domain = domain;
    entry.kind = kind;
    entry.accessibleResources = {"fabric.compute"};
    config.memories.push_back(std::move(entry));
  };
  memory("external.memory", "amoeba.external", "external-data-availability");
  auto link = [&](const std::string &from, const std::string &to,
                  const std::string &resource, bool external) {
    StaticTransfer entry;
    entry.id = from + "->" + to;
    entry.source = from;
    entry.target = to;
    entry.resource = resource;
    entry.domain = external ? "amoeba.external" : "amoeba.fabric";
    // Zero bandwidth is the existing explicit unknown-cost marker. These
    // semantic links are not physical SRAM/DMA or calibrated network links.
    entry.provenance = {"target_contract",
                        external
                            ? "external I/O supply is unpriced and unverified"
                            : "abstract data availability; concrete directed "
                              "network route and cycles are backend-bound"};
    config.links.push_back(std::move(entry));
  };
  for (size_t task = 0; task < shapes.size(); ++task) {
    const auto &shape = shapes[task];
    if (shape.rows <= 0 || shape.columns <= 0 ||
        shape.rows > target.fabricRows || shape.columns > target.fabricColumns)
      return fail(error, "rectangular task shape exceeds CGRA fabric");
    memory(endpoint(task), "amoeba.fabric", "abstract-data-availability");
    MappingTaskPlacement place;
    place.task = task;
    place.resource = "fabric.compute";
    place.memory = endpoint(task);
    place.implementation = "amoeba-native";
    place.resourceCount = shape.rows * shape.columns;
    place.resourceShape = {shape.rows, shape.columns};
    choice.placements.push_back(std::move(place));
    link("external.memory", endpoint(task), "external.supply", true);
    link(endpoint(task), "external.memory", "external.supply", true);
  }
  for (size_t from = 0; from < shapes.size(); ++from)
    for (size_t to = 0; to < shapes.size(); ++to)
      if (from != to)
        link(endpoint(from), endpoint(to), "intertask.network", false);
  for (const auto &storage : source.storages) {
    if (storage.role == "input")
      config.assumptions.initialLocations[storage.id] = "external.memory";
    if (storage.role == "output")
      config.assumptions.outputLocations[storage.id] = "external.memory";
  }
  return true;
}

bool build(const GenericAnalysis &source, const AmoebaTargetContract &target,
           const std::vector<AmoebaTaskShape> &shapes, ProgramGraph &graph,
           std::string &error) {
  StaticConfig config;
  ProgramDataOrganizationChoice choice;
  if (!contract(source, target, shapes, config, choice, error) ||
      !planProgramDataOrganization(source, config, choice, graph, error))
    return false;
  std::vector<size_t> computations(source.tasks.size(), SIZE_MAX);
  for (size_t event = 0; event < graph.events.size(); ++event) {
    auto &value = graph.events[event];
    value.dependencies.clear();
    if (value.kind == "compute") {
      if (value.task >= computations.size() ||
          computations[value.task] != SIZE_MAX)
        return fail(error, "adapter requires one computation per actual task");
      computations[value.task] = event;
    }
  }
  auto depend = [&](size_t event, size_t parent) {
    if (event != parent)
      graph.events[event].dependencies.push_back(parent);
  };
  for (size_t event = 0; event < graph.events.size(); ++event) {
    for (size_t id : graph.events[event].reads) {
      const auto &copy = graph.copies[id];
      if (copy.createdBy >= 0)
        depend(event, size_t(copy.createdBy));
      const auto &version = graph.versions[copy.version];
      if (version.producer >= 0)
        depend(event, size_t(version.producer));
    }
  }
  for (const auto &edge : graph.orderEdges)
    depend(edge.consumer, edge.producer);
  for (size_t task = 0; task < computations.size(); ++task) {
    if (computations[task] == SIZE_MAX)
      return fail(error, "missing source task computation");
    for (size_t parent : source.tasks[task].dependencies) {
      if (parent >= task)
        return fail(error, "source order is not topological");
      depend(computations[task], computations[parent]);
    }
  }
  // Whole-storage versions are ordered. Preserve readers-before-overwrite
  // hazards even where a token dependency alone does not encode that hazard.
  std::map<size_t, size_t> lastVersion;
  for (size_t version = 0; version < graph.versions.size(); ++version) {
    const auto &value = graph.versions[version];
    size_t storage = value.region; // independent source partition cell
    auto previous = lastVersion.find(storage);
    if (value.producer >= 0 && previous != lastVersion.end()) {
      const auto &prior = graph.versions[previous->second];
      if (prior.producer >= 0)
        depend(size_t(value.producer), size_t(prior.producer));
      for (const auto &copy : graph.copies)
        if (copy.version == previous->second)
          for (size_t user : copy.users)
            if (user < size_t(value.producer))
              depend(size_t(value.producer), user);
    }
    lastVersion[storage] = version;
  }
  // No retained imported copy is silently shared across computations. A
  // produced source version can naturally feed separately counted transfers.
  for (const auto &copy : graph.copies) {
    if (copy.createdBy < 0 ||
        graph.events[size_t(copy.createdBy)].kind != "transfer")
      continue;
    std::set<size_t> users;
    for (size_t event : copy.users)
      users.insert(graph.events[event].task);
    if (users.size() > 1)
      return fail(
          error,
          "shared imported copy is outside the phase-local Amoeba adapter");
  }
  for (auto &event : graph.events) {
    std::sort(event.dependencies.begin(), event.dependencies.end());
    event.dependencies.erase(
        std::unique(event.dependencies.begin(), event.dependencies.end()),
        event.dependencies.end());
  }
  graph.targetIdentity = staticFingerprint(serializeAmoebaTarget(target));
  graph.schedule.policy = "dependency-order-witness";
  graph.estimatedMakespanMs.reset();
  return finalizeProgramGraph(source, graph, nullptr, error);
}
} // namespace

llvm::json::Value
orbit::taskflow::serializeAmoebaTarget(const AmoebaTargetContract &target) {
  return llvm::json::Object{
      {"kind", "amoeba-native-abstract-target-v1"},
      {"fabric_rows", target.fabricRows},
      {"fabric_columns", target.fabricColumns},
      {"fabric_shape_unit", "cgra_block"},
      {"pe_rows", target.peRows},
      {"pe_columns", target.peColumns},
      {"implementation_shape_unit", "processing_element"},
      {"context_slots", target.contextSlots > 0
                            ? llvm::json::Value(target.contextSlots)
                            : llvm::json::Value(nullptr)},
      {"architecture_identity", target.architectureIdentity},
      {"network_identity", target.networkIdentity},
      {"physical_memory_capacity", nullptr},
      {"external_io_cost", "unknown"},
      {"network_cost", "backend-route-cycles"}};
}

bool orbit::taskflow::buildAmoebaProgramGraph(
    GenericAnalysis &source, const AmoebaTargetContract &target,
    const std::vector<AmoebaTaskShape> &shapes, ProgramGraph &graph,
    std::string &error) {
  if (!build(source, target, shapes, graph, error))
    return false;
  return checkProgramGraphSource(source, graph, error) &&
         checkProgramGraph(graph, error);
}

bool orbit::taskflow::checkAmoebaProgramGraph(
    const GenericAnalysis &source, const AmoebaTargetContract &target,
    const std::vector<AmoebaTaskShape> &shapes, const ProgramGraph &graph,
    std::string &error) {
  if (!checkProgramGraphSource(source, graph, error) ||
      !checkProgramGraph(graph, error))
    return false;
  ProgramGraph expected;
  if (!build(source, target, shapes, expected, error))
    return false;
  if (programGraphIdentity(expected) != programGraphIdentity(graph))
    return fail(error, "graph differs from source-derived phase-local "
                       "target/shape realization");
  return true;
}
