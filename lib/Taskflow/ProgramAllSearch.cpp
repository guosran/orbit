#include "Taskflow/ProgramAllSearch.h"
#include "Taskflow/GenericNativeCompositeTiming.h"
#include "Taskflow/GenericReuse.h"
#include "Taskflow/JSONUtils.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramBeamSearch.h"
#include "Taskflow/ProgramInvocationEvaluation.h"
#include "Taskflow/ProgramMechanismCost.h"
#include "Taskflow/ProgramPartition.h"
#include "Taskflow/ProgramSearch.h"
#include "Taskflow/ProgramTargetBinding.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>

using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
bool fail(std::string &e, const std::string &s) {
  e = "public all-dimension search: " + s;
  return false;
}
// raw_fd_ostream reports buffered write and close failures only after close.
// Clear an acknowledged error so its destructor does not abort the process.
bool closeEvidence(llvm::raw_fd_ostream &out, const std::string &path,
                   std::string &error) {
  out.close();
  if (!out.has_error())
    return true;
  const auto ec = out.error();
  out.clear_error();
  return fail(error, "cannot persist " + path + ": " + ec.message());
}
bool save(const std::string &path, const j::Value &value, std::string &e) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path + ".tmp", ec);
  if (ec)
    return fail(e, ec.message());
  out << llvm::formatv("{0:2}\n", value);
  if (!closeEvidence(out, path + ".tmp", e))
    return false;
  if (auto x = llvm::sys::fs::rename(path + ".tmp", path))
    return fail(e, x.message());
  return true;
}
std::string print(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream out(text);
  module.print(out);
  return text;
}
j::Array numbers(const std::vector<int64_t> &v) {
  j::Array a;
  for (auto n : v)
    a.push_back(n);
  return a;
}
j::Array indices(const std::vector<size_t> &v) {
  j::Array a;
  for (auto n : v)
    a.push_back(int64_t(n));
  return a;
}
bool nativeCompositeProvider(const j::Value &configuration) {
  auto root = configuration.getAsObject();
  auto air = root ? root->getObject("generic_air") : nullptr;
  auto model = air ? air->getObject("cost_model") : nullptr;
  auto kind = model ? model->getString("kind") : std::nullopt;
  return kind == "linear-composite" || kind == "implementation-composite";
}
// A neutral public binding preserves the native executable witness. Rebuilding
// even that binding with the whole-task copy adapter would replace real native
// pools/stages with a different, unlowered graph and lose its complete price.
bool preservesNativeExecution(const GenericMappingPlan &native,
                              const StaticConfig &target,
                              const ProgramDataOrganizationChoice &choice,
                              const std::vector<int64_t> &order, bool reuse) {
  if (reuse || !choice.transferChoices.empty() ||
      choice.semanticDependenciesOnly ||
      choice.policy != native.communication.dataPolicy ||
      staticFingerprint(serializeMemoryArchitecture(target)) !=
          staticFingerprint(
              serializeMemoryArchitecture(native.memoryBinding.architecture)) ||
      order.size() != native.orderedTasks.size())
    return false;
  for (size_t i = 0; i < order.size(); ++i)
    if (order[i] != int64_t(native.orderedTasks[i]))
      return false;
  for (const auto &p : choice.placements) {
    auto implementation = std::find_if(
        native.taskImplementations.begin(), native.taskImplementations.end(),
        [&](const auto &v) { return v.task == p.task; });
    if (p.resource != native.memoryBinding.computeResource ||
        p.memory != native.memoryBinding.localMemory || p.resourceCount != 1 ||
        !p.resourceShape.empty() ||
        implementation == native.taskImplementations.end() ||
        p.implementation != implementation->kind)
      return false;
  }
  return true;
}
// Build the native implementation witness before pricing the selected public
// invocation. A runtime provider's unsupported ABI is unknown cost, not proof
// that the native implementation graph is physically infeasible.
j::Value nativePlanningConfig(const j::Value &configuration) {
  auto root = configuration.getAsObject();
  auto air = root ? root->getObject("generic_air") : nullptr;
  auto cost = air ? air->getObject("cost_model") : nullptr;
  const bool structural =
      cost && (cost->getString("kind") == "region-invocation" ||
               nativeCompositeProvider(configuration));
  // The witness is structural; providers are evaluated after graph checking.
  return structural ? copyJSONWithoutNestedFields(configuration, "generic_air",
                                                  {"cost_model"})
                    : copyJSONWithoutFields(configuration);
}
j::Value mapping(const ProgramMappingChoice &c) {
  j::Array groups;
  for (auto &g : genericResidencyGroups(c))
    groups.push_back(j::Object{{"tasks", indices(g.tasks)},
                               {"storages", indices(g.retainedStorages)}});
  return j::Object{{"residency", c.residency},
                   {"grouping", c.grouping},
                   {"data_organization", c.dataOrganization},
                   {"storage", c.storage},
                   {"reuse_loop", c.reuseLoop},
                   {"panel_loop", c.panelLoop},
                   {"groups", std::move(groups)}};
}
bool mapping(const j::Value &value, ProgramMappingChoice &c, std::string &e) {
  auto o = value.getAsObject();
  if (!o)
    return fail(e, "mapping choice requires an object");
  for (auto &entry : *o) {
    auto key = entry.first;
    if (key == "residency" || key == "grouping" || key == "data_organization") {
      if (!entry.second.getAsString())
        return fail(e, "mapping string field has wrong type: " + key.str());
    } else if (key == "storage" || key == "reuse_loop" || key == "panel_loop") {
      if (!entry.second.getAsInteger())
        return fail(e, "mapping index field has wrong type: " + key.str());
    } else if (key != "groups")
      return fail(e, "unconsumed mapping field: " + key.str());
  }
  c = {};
  c.residency = o->getString("residency").value_or("").str();
  c.grouping = o->getString("grouping").value_or("").str();
  c.dataOrganization = o->getString("data_organization").value_or("").str();
  c.storage = o->getInteger("storage").value_or(-1);
  c.reuseLoop = o->getInteger("reuse_loop").value_or(-1);
  c.panelLoop = o->getInteger("panel_loop").value_or(-1);
  if (auto value = o->get("groups")) {
    auto groups = value->getAsArray();
    if (!groups)
      return fail(e, "mapping groups must be an array");
    for (auto &v : *groups) {
      auto g = v.getAsObject();
      if (!g || g->size() != 2 || !g->getArray("tasks") ||
          !g->getArray("storages"))
        return fail(e, "invalid public residency group");
      ProgramResidencyGroup x;
      for (auto &n : *g->getArray("tasks")) {
        auto i = n.getAsInteger();
        if (!i || *i < 0)
          return fail(e, "invalid group task");
        x.tasks.push_back(*i);
      }
      for (auto &n : *g->getArray("storages")) {
        auto i = n.getAsInteger();
        if (!i || *i < 0)
          return fail(e, "invalid group storage");
        x.retainedStorages.push_back(*i);
      }
      c.residencyGroups.push_back(x);
    }
  }
  return true;
}
// Validate every admitted binding before starting CP-SAT. A misspelled or
// unsupported choice is a configuration error, never a silently fixed
// dimension.
bool binding(const j::Value &v, bool whole, bool heterogeneous, size_t tasks,
             std::set<std::string> &properties, std::string &e) {
  if (v.getAsNull())
    return true;
  auto b = v.getAsObject();
  if (!b)
    return fail(e, "choice binding must be an object or null");
  auto task = b->getInteger("task");
  if (b->get("task") && (!task || *task < 0 || size_t(*task) >= tasks))
    return fail(e, "invalid binding task");
  if (task && !b->get("resource") && !b->get("memory") &&
      !b->get("resource_count") && !b->get("resource_shape") &&
      !b->get("implementation") && !b->get("device"))
    return fail(e,
                "task requires an actual placement or implementation binding");
  for (auto &entry : *b) {
    auto key = entry.first;
    bool valid = true;
    if (key == "fission" || key == "distribute" || key == "reuse_slots" ||
        key == "partition_exact_regions" ||
        key == "semantic_dependencies_only" || key == "cut")
      valid = entry.second.getAsBoolean().has_value();
    else if (key == "tile_axis" || key == "tile" || key == "lanes" ||
             key == "partition" || key == "task" || key == "resource_count" ||
             key == "before_task")
      valid = entry.second.getAsInteger().has_value();
    else if (key == "implementation" || key == "resource" || key == "memory" ||
             key == "policy" || key == "device")
      valid = entry.second.getAsString().has_value();
    else if (key == "resource_shape" || key == "priority") {
      auto a = entry.second.getAsArray();
      valid = a != nullptr;
      if (a)
        for (auto &n : *a)
          valid &= n.getAsInteger().has_value();
    } else if (key == "mapping") {
      ProgramMappingChoice c;
      if (!mapping(entry.second, c, e))
        return false;
    } else if (key == "route") {
      auto r = entry.second.getAsObject();
      if (!r)
        return fail(e, "route must be an object");
      for (auto &field : *r)
        if (field.first != "storage" && field.first != "consumer_task" &&
            field.first != "source_memory" && field.first != "target_memory" &&
            field.first != "links")
          return fail(e, "unconsumed route field: " + field.first.str());
      auto storage = r->getInteger("storage"),
           consumer = r->getInteger("consumer_task");
      auto links = r->getArray("links");
      valid = storage && *storage >= 0 && consumer && *consumer >= 0 &&
              size_t(*consumer) <= tasks && r->getString("source_memory") &&
              r->getString("target_memory") && links;
      if (links)
        for (auto &l : *links)
          valid &= l.getAsString().has_value();
    } else
      return fail(e, "unconsumed public decision field: " + key.str());
    if (!valid)
      return fail(e, "invalid type for public decision field: " + key.str());
    if (key == "device") {
      auto device = entry.second.getAsString();
      if (!heterogeneous || !task || (*device != "cpu" && *device != "npu"))
        return fail(
            e,
            "device requires enabled heterogeneous search and a source task");
    }
    if (key == "before_task" || key == "cut") {
      auto before = b->getInteger("before_task");
      if (!heterogeneous || !before || *before < 1 ||
          size_t(*before) >= tasks || !b->getBoolean("cut"))
        return fail(e, "cut requires enabled heterogeneous search and an "
                       "internal source boundary");
    }
    if ((key == "resource" || key == "memory" || key == "resource_count" ||
         key == "resource_shape") &&
        !task)
      return fail(e, key.str() + " requires task");
    if (key == "partition" && *entry.second.getAsInteger() < 0)
      return fail(e, "partition must be nonnegative");
    if (key == "tile" && *entry.second.getAsInteger() < 0)
      return fail(e, "tile must be nonnegative");
    if (key == "lanes" && whole && *entry.second.getAsInteger() != 1)
      return fail(
          e, "whole-task implementation adapter has no vector-lane binding");
    if (key == "lanes" && !b->getString("implementation"))
      return fail(e, "lanes requires implementation");
    if (key == "tile" && !b->getInteger("tile_axis"))
      return fail(e, "tile requires tile_axis");
    if (key == "tile_axis" && !b->getInteger("tile"))
      return fail(e, "tile_axis requires tile");
    if (key == "implementation" && !whole &&
        !validateNativeImplementation(b->getString("implementation")->str(),
                                      b->getInteger("lanes").value_or(1), e))
      return false;
    if (!whole && (key == "partition_exact_regions" ||
                   key == "semantic_dependencies_only"))
      return fail(e, "native phase adapter derives region and semantic "
                     "dependencies from its checked source plan");
    if (whole && (key == "tile" || key == "mapping" || key == "partition"))
      return fail(e, "whole-task adapter does not consume native "
                     "tile/mapping/partition attributes");
    if (key == "task" || key == "lanes" || key == "tile_axis" ||
        key == "before_task")
      continue;
    std::string property = key.str();
    if (key == "tile")
      property += "." + std::to_string(*b->getInteger("tile_axis"));
    if (task &&
        (key == "resource" || key == "memory" || key == "resource_count" ||
         key == "resource_shape" || key == "implementation" || key == "device"))
      property += "." + std::to_string(*task);
    if (key == "cut")
      property += "." + std::to_string(*b->getInteger("before_task"));
    if (key == "route") {
      auto r = b->getObject("route");
      property += "." + std::to_string(*r->getInteger("storage")) + "." +
                  std::to_string(*r->getInteger("consumer_task"));
    }
    properties.insert(property);
  }
  return true;
}
j::Value placement(const MappingTaskPlacement &p) {
  return j::Object{{"task", int64_t(p.task)},
                   {"resource", p.resource},
                   {"memory", p.memory},
                   {"implementation", p.implementation},
                   {"resource_count", p.resourceCount},
                   {"resource_shape", numbers(p.resourceShape)}};
}
// Existing architecture only. Local route alternatives retain edge identities,
// including parallel links, and never assign a made-up bandwidth.
void paths(const StaticConfig &target, const std::string &from,
           const std::string &to, size_t cap,
           std::vector<std::vector<std::string>> &out, bool &truncated) {
  std::vector<std::string> path;
  std::set<std::string> seen{from};
  std::function<void(const std::string &)> visit = [&](const std::string &at) {
    if (out.size() >= cap) {
      truncated = true;
      return;
    }
    if (at == to) {
      out.push_back(path);
      return;
    }
    for (auto &l : target.links)
      if (l.source == at && !seen.count(l.target)) {
        seen.insert(l.target);
        path.push_back(l.id);
        visit(l.target);
        path.pop_back();
        seen.erase(l.target);
      }
  };
  visit(from);
}
bool parameters(const j::Value &config, ProgramMechanismModel &model,
                std::string &e) {
  auto root = config.getAsObject();
  auto air = root ? root->getObject("generic_air") : nullptr;
  auto o = air ? air->getObject("cost_model") : nullptr;
  if (o && o->getString("kind") == "region-invocation") {
    ProgramInvocationModel invocation;
    if (!parseProgramInvocationModel(*air->get("cost_model"), invocation, e))
      return false;
    model = std::move(invocation.mechanisms);
    return true;
  }
  if (!o || o->getString("kind") != "event-mechanism")
    return fail(e, "joint public graph evaluation requires event-mechanism or "
                   "region-invocation provider");
  model.id = staticFingerprint(j::Value(j::Object(*o)));
  model.mode = o->getString("mode").value_or("").str();
  model.boundary = o->getString("boundary").value_or("").str();
  auto values = o->getObject("parameters");
  auto impl = o->getArray("implementations");
  if (!values || !impl)
    return fail(e, "provider coverage and parameter provenance required");
  for (auto &i : *impl) {
    auto s = i.getAsString();
    if (!s)
      return false;
    model.implementations.push_back(s->str());
  }
  for (auto &v : *values) {
    auto row = v.second.getAsObject();
    auto p = row ? row->getObject("provenance") : nullptr;
    if (!row || !p)
      return false;
    ProgramMechanismParameter x;
    x.value = row->getNumber("value");
    x.unit = row->getString("unit").value_or("").str();
    x.kind = p->getString("kind").value_or("").str();
    x.source = p->getString("source").value_or("").str();
    model.parameters[v.first.str()] = x;
  }
  return true;
}
// Reuse the public timing model, changing only its declared priority decision.
bool timing(const ProgramGraph &graph, const ProgramMechanismModel &model,
            const j::Value &config, const std::vector<int64_t> &order,
            ProgramTimingEvaluation &out, std::string &e) {
  ProgramMechanismEvaluation base;
  auto root = config.getAsObject();
  auto air = root ? root->getObject("generic_air") : nullptr;
  auto cost = air ? air->getObject("cost_model") : nullptr;
  if (cost && cost->getString("kind") == "region-invocation") {
    ProgramInvocationModel invocation;
    j::Value receipt = nullptr;
    if (!parseProgramInvocationModel(*air->get("cost_model"), invocation, e) ||
        !evaluateProgramInvocationModel(graph, invocation, base, receipt, e))
      return false;
  } else if (!evaluateProgramMechanismCost(graph, model, base, e))
    return false;
  ProgramTimingRequest q;
  q.graphId = programGraphIdentity(graph);
  q.modelIdentity = model.id;
  q.boundary = model.boundary;
  std::vector<std::pair<int64_t, std::string>> priorities;
  std::vector<int64_t> rank(order.size());
  for (size_t i = 0; i < order.size(); ++i)
    rank[order[i]] = i;
  for (auto &a : base.timing.activities) {
    ProgramActivityCost c;
    c.id = a.id;
    c.boundary = a.boundary;
    c.provenance = a.provenance;
    c.events = a.events;
    c.regionEvents = a.regionEvents;
    c.executionRegions = a.executionRegions;
    c.resourceDemands = a.resourceDemands;
    c.durationMs = a.durationMs;
    c.unknownReason = a.reason;
    c.predecessors = a.predecessors;
    c.runtimeOnly = c.events.empty() && c.regionEvents.empty() &&
                    c.executionRegions.empty();
    q.costs.push_back(c);
    int64_t priority = rank.size();
    for (auto event : a.events)
      priority = std::min(priority, rank[graph.events[event].task]);
    priorities.emplace_back(priority, a.id);
  }
  std::stable_sort(priorities.begin(), priorities.end(),
                   [](auto &a, auto &b) { return a.first < b.first; });
  for (auto &a : priorities)
    q.schedulePriority.push_back(a.second);
  return evaluateProgramTiming(graph, q, out, e);
}
bool mergeSlots(ProgramGraph &g, std::string &e) {
  // Same logical storage and memory only. The existing buffer-order checker
  // supplies required anti-dependencies; no new physical buffering model.
  std::vector<size_t> map(g.allocations.size(), SIZE_MAX);
  std::vector<MappingAllocation> allocations;
  std::vector<std::vector<size_t>> members;
  for (size_t i = 0; i < g.allocations.size(); ++i) {
    auto a = g.allocations[i];
    size_t slot = SIZE_MAX;
    if (a.addressing != "region-addressed")
      for (size_t j = 0; j < allocations.size(); ++j) {
        auto &b = allocations[j];
        if (a.storage != b.storage || a.memory != b.memory ||
            a.bytes != b.bytes || b.addressing == "region-addressed")
          continue;
        bool overlaps = false;
        for (auto &c : g.copies)
          if (c.allocation == i)
            for (auto &d : g.copies)
              if (std::find(members[j].begin(), members[j].end(),
                            d.allocation) != members[j].end())
                overlaps |= c.availableFrom < d.releasedAfter &&
                            d.availableFrom < c.releasedAfter;
        if (!overlaps) {
          slot = j;
          break;
        }
      }
    if (slot == SIZE_MAX) {
      slot = allocations.size();
      a.key = "search-slot-" + std::to_string(slot);
      allocations.push_back(a);
      members.push_back({});
    } else {
      allocations[slot].beginPhase =
          std::min(allocations[slot].beginPhase, a.beginPhase);
      allocations[slot].endPhase =
          std::max(allocations[slot].endPhase, a.endPhase);
      allocations[slot].addressing = "reusable-slot";
    }
    map[i] = slot;
    members[slot].push_back(i);
  }
  for (auto &c : g.copies)
    c.allocation = map[c.allocation];
  g.allocations = std::move(allocations);
  return checkProgramGraph(g, e);
}
} // namespace

bool orbit::taskflow::materializeProgramAllHeterogeneousCandidate(
    mlir::ModuleOp module, const j::Value &configuration,
    const HeterogeneousMappingChoice &deviceChoice, const StaticConfig &target,
    const ProgramDataOrganizationChoice &publicChoice,
    const std::vector<int64_t> &priority, bool reuseSlots,
    int64_t maxSourceInstances, ProgramAllHeterogeneousCandidate &candidate,
    std::string &error) {
  candidate = {};
  GenericAnalysis source;
  if (!analyzeGenericProgram(module, source, error))
    return false;
  if (priority.size() != source.tasks.size() || maxSourceInstances < 1)
    return fail(error,
                "combined candidate needs a complete priority and work cap");
  if (!publicChoice.computeEstimates.empty())
    return fail(error, "combined region costs cannot silently consume separate "
                       "whole-task compute estimates");
  auto sortedPriority = priority;
  std::sort(sortedPriority.begin(), sortedPriority.end());
  for (size_t i = 0; i < sortedPriority.size(); ++i)
    if (sortedPriority[i] != int64_t(i))
      return fail(error,
                  "combined priority must cover each transformed task once");
  if (auto partition = module->getAttrOfType<mlir::IntegerAttr>(
          "orbit.program_partition_extent"))
    if (partition.getInt() > 0)
      return fail(error, "combined host partition/assembly implementation is "
                         "unsupported; selected partition remains unknown");
  auto choice = deviceChoice;
  choice.implementationOwned = true;
  candidate.source = module;
  if (!buildHeterogeneousMapping(source, choice, candidate.parent, error) ||
      !checkHeterogeneousMapping(source, choice, candidate.parent, error))
    return false;
  // This executor serializes regions in source order. Checking priorities only
  // inside each child would silently ignore a requested cross-region reorder.
  std::vector<size_t> taskRegions(source.tasks.size());
  for (size_t region = 0; region < candidate.parent.slices.size(); ++region)
    for (size_t task = candidate.parent.slices[region].beginTask;
         task < candidate.parent.slices[region].endTask; ++task)
      taskRegions[task] = region;
  size_t previousRegion = 0;
  for (auto task : priority) {
    if (taskRegions[task] < previousRegion)
      return fail(error, "serial CPU/NPU executor cannot realize selected "
                         "cross-region temporal priority");
    previousRegion = taskRegions[task];
  }
  ProgramMappingChoice parentMapping;
  if (!readNativeMappingChoice(module, parentMapping, error))
    return false;
  j::Array regions;
  int64_t remainingInstances = maxSourceInstances;
  for (size_t index = 0; index < candidate.parent.slices.size(); ++index) {
    ProgramAllHeterogeneousRegion region;
    region.region = index;
    if (!materializeHeterogeneousRegion(candidate.parent, index, region.source,
                                        region.projection, error))
      return false;
    GenericAnalysis child;
    if (!analyzeGenericProgram(*region.source, child, error))
      return false;
    std::map<size_t, size_t> tasks, storages;
    for (size_t i = 0; i < region.projection.parentTasks.size(); ++i)
      tasks.emplace(region.projection.parentTasks[i], i);
    for (size_t i = 0; i < child.storages.size(); ++i) {
      auto found = std::find_if(source.storages.begin(), source.storages.end(),
                                [&](const auto &storage) {
                                  return storage.id == child.storages[i].id;
                                });
      if (found == source.storages.end())
        return fail(error, "combined child has an unbound logical storage");
      storages.emplace(found - source.storages.begin(), i);
    }
    std::vector<int64_t> childPriority;
    for (auto task : priority)
      if (tasks.count(task))
        childPriority.push_back(tasks.at(task));
    const bool device = candidate.parent.slices[index].device == "npu";
    if (device) {
      auto mappingChoice = parentMapping;
      mappingChoice.taskGroup.clear();
      mappingChoice.retainedStorages.clear();
      mappingChoice.residencyGroups.clear();
      for (const auto &group : genericResidencyGroups(parentMapping)) {
        size_t members = 0;
        for (auto task : group.tasks)
          members += tasks.count(task);
        if (!members)
          continue;
        if (members != group.tasks.size())
          return fail(error,
                      "selected native residency group crosses a CPU/NPU "
                      "region boundary; refinement is unsupported");
        ProgramResidencyGroup projected;
        for (auto task : group.tasks)
          projected.tasks.push_back(tasks.at(task));
        for (auto storage : group.retainedStorages) {
          if (!storages.count(storage))
            return fail(error,
                        "selected retained storage has no child binding");
          projected.retainedStorages.push_back(storages.at(storage));
        }
        mappingChoice.residencyGroups.push_back(std::move(projected));
      }
      if (mappingChoice.storage >= 0) {
        if (!storages.count(mappingChoice.storage))
          return fail(error, "selected native storage scope is outside region");
        mappingChoice.storage = storages.at(mappingChoice.storage);
      }
      if (!setNativeMappingChoice(*region.source, mappingChoice, error))
        return false;
      if (auto tiles = region.source->getOperation()
                           ->getAttrOfType<mlir::DenseI64ArrayAttr>(
                               "orbit.air_tile_sizes")) {
        auto extents = tiles.asArrayRef();
        if (extents.size() != 3 ||
            std::any_of(extents.begin(), extents.end(),
                        [](int64_t extent) { return extent < 1; }))
          return fail(error,
                      "combined native tile requires three positive extents");
        int64_t total = 0;
        for (const auto &task : child.tasks) {
          int64_t count = 1;
          for (size_t axis = 0; axis < std::min(size_t(3), task.loops.size());
               ++axis) {
            const auto &loop = task.loops[axis];
            auto trips = (loop.upper - loop.lower - 1) / loop.step + 1;
            auto pieces = (trips - 1) / extents[axis] + 1;
            if (count > remainingInstances / pieces)
              return fail(
                  error,
                  "combined source-instance work cap; unknown refinement");
            count *= pieces;
          }
          if (total > remainingInstances - count)
            return fail(
                error, "combined source-instance work cap; unknown refinement");
          total += count;
        }
      }
      auto planning = copyJSONWithoutNestedFields(configuration, "generic_air",
                                                  {"cost_model"});
      // A selected source tile is authoritative, including through this typed
      // API. An inherited configuration tile must not replace that decision.
      if (region.source->getOperation()->hasAttr("orbit.air_tile_sizes"))
        planning = copyJSONWithoutNestedFields(planning, "generic_air",
                                               {"tile_sizes"});
      if (!planGenericMapping(child, planning, region.native, error))
        return false;
      ProgramDataOrganizationChoice childChoice = publicChoice;
      childChoice.placements.clear();
      childChoice.transferChoices.clear();
      childChoice.computeEstimates.clear();
      for (const auto &placement : publicChoice.placements)
        if (tasks.count(placement.task)) {
          auto projected = placement;
          projected.task = tasks.at(placement.task);
          auto implementation = std::find_if(
              region.native.taskImplementations.begin(),
              region.native.taskImplementations.end(),
              [&](const auto &value) { return value.task == projected.task; });
          if (implementation == region.native.taskImplementations.end())
            return fail(error, "native child implementation coverage missing");
          projected.implementation = implementation->kind;
          childChoice.placements.push_back(std::move(projected));
        }
      if (childChoice.placements.size() != child.tasks.size())
        return fail(error, "native child placement coverage missing");
      std::sort(childChoice.placements.begin(), childChoice.placements.end(),
                [](const auto &a, const auto &b) { return a.task < b.task; });
      for (const auto &route : publicChoice.transferChoices) {
        const bool finalOutput = route.consumerTask == source.tasks.size();
        if ((!finalOutput && !tasks.count(route.consumerTask)) ||
            !storages.count(route.storage))
          continue;
        auto projected = route;
        projected.storage = storages.at(route.storage);
        projected.consumerTask =
            finalOutput ? child.tasks.size() : tasks.at(route.consumerTask);
        childChoice.transferChoices.push_back(std::move(projected));
      }
      // A scheduling priority is executable only if the selected native phase
      // order actually implements it. Do not price a reordered graph while
      // handing the backend an unchanged native plan.
      if (childPriority.size() != region.native.orderedTasks.size() ||
          !std::equal(childPriority.begin(), childPriority.end(),
                      region.native.orderedTasks.begin()))
        return fail(error,
                    "native child cannot realize selected temporal priority");
      if (preservesNativeExecution(region.native, target, childChoice,
                                   childPriority, reuseSlots))
        region.selectedGraph = region.native.communication;
      else {
        if (!rebindProgramGraph(region.native.communication, target,
                                childChoice, region.selectedGraph, error) ||
            (reuseSlots && !mergeSlots(region.selectedGraph, error)))
          return false;
        // The backend receives this exact selected graph. If the native plan
        // checker cannot certify it, retain UNKNOWN instead of silently using
        // the old default copy/resource plan during lowering.
        region.native.communication = region.selectedGraph;
        if (!checkGenericMapping(child, region.native, error))
          return false;
      }
      region.hasNativePlan = true;
      if (!bindHeterogeneousRegionNativePlan(candidate.parent, region.native,
                                             region.projection, error))
        return false;
      region.receipt = serializeGenericMapping(child, region.native);
    } else {
      std::vector<int64_t> sourceOrder(child.tasks.size());
      std::iota(sourceOrder.begin(), sourceOrder.end(), 0);
      if (childPriority != sourceOrder)
        return fail(
            error,
            "sequential CPU implementation cannot realize selected priority");
      HeterogeneousMappingChoice cpuChoice;
      cpuChoice.taskDevices.assign(child.tasks.size(), "cpu");
      cpuChoice.implementationOwned = true;
      HeterogeneousMappingPlan cpu;
      if (!buildHeterogeneousMapping(child, cpuChoice, cpu, error) ||
          !bindHeterogeneousRegionGraph(candidate.parent, cpu.graph,
                                        region.projection, error))
        return false;
      region.selectedGraph = std::move(cpu.graph);
      region.receipt = j::Object{
          {"implementation", "llvm-ciface-f32"},
          {"inactive_native_choices",
           "NPU L1/DMA/mapping bindings are conditional on NPU placement"}};
    }
    if (region.selectedGraph.instances.size() > uint64_t(remainingInstances))
      return fail(error, "combined source-instance work cap across regions; "
                         "unknown refinement");
    remainingInstances -= int64_t(region.selectedGraph.instances.size());
    if (!checkProgramGraph(region.selectedGraph, error))
      return false;
    regions.push_back(j::Object{
        {"region", int64_t(index)},
        {"device", candidate.parent.slices[index].device},
        {"source", print(*region.source)},
        {"projection",
         serializeHeterogeneousRegionProjection(region.projection)},
        {"selected_graph_id", programGraphIdentity(region.selectedGraph)},
        {"selected_graph", serializeProgramGraph(region.selectedGraph)},
        {"has_native_plan", region.hasNativePlan},
        {"implementation", j::Value(region.receipt)}});
    candidate.regions.push_back(std::move(region));
  }
  j::Array selectedPlacements, selectedRoutes;
  for (const auto &selected : publicChoice.placements)
    selectedPlacements.push_back(placement(selected));
  for (const auto &route : publicChoice.transferChoices) {
    j::Array links;
    for (const auto &link : route.links)
      links.push_back(link);
    selectedRoutes.push_back(
        j::Object{{"storage", int64_t(route.storage)},
                  {"consumer_task", int64_t(route.consumerTask)},
                  {"source_memory", route.sourceMemory},
                  {"target_memory", route.targetMemory},
                  {"links", std::move(links)}});
  }
  candidate.receipt = j::Object{
      {"schema", "orbit.combined-native-heterogeneous-candidate.v32"},
      {"source", print(module)},
      {"parent", serializeHeterogeneousMapping(candidate.parent)},
      {"regions", std::move(regions)},
      {"priority", numbers(priority)},
      {"reuse_slots", reuseSlots},
      {"target", serializeMemoryArchitecture(target)},
      {"selected_placements", std::move(selectedPlacements)},
      {"selected_routes", std::move(selectedRoutes)},
      {"selected_retention_policy",
       programDataPolicyName(publicChoice.policy).str()},
      {"runtime_cost_ownership",
       "one complete call per region, plus application reset/control/output "
       "once; child body/DMA never charged again"},
      {"backend_replanning_allowed", false}};
  candidate.assignmentId = staticFingerprint(candidate.receipt);
  (*candidate.receipt.getAsObject())["candidate_id"] = candidate.assignmentId;
  return true;
}

bool orbit::taskflow::runProgramAllSearch(llvm::StringRef text,
                                          const SearchOptions &options,
                                          const j::Value &configuration,
                                          std::string &error) {
  return runProgramAllSearch(text, options, configuration, error, {});
}

bool orbit::taskflow::runProgramAllSearch(
    llvm::StringRef text, const SearchOptions &options,
    const j::Value &configuration, std::string &error,
    const ProgramAllHeterogeneousEvaluator &heterogeneousEvaluator) {
  const auto started = std::chrono::steady_clock::now();
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto parse = [&](llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  };
  auto original = parse(text);
  GenericAnalysis facts;
  if (!original || !analyzeGenericProgram(*original, facts, error))
    return false;
  auto root = configuration.getAsObject();
  auto search = root ? root->getObject("program_search") : nullptr;
  auto graphConfig = root ? root->getObject("program_graph") : nullptr;
  bool whole = graphConfig && graphConfig->getString("adapter") == "whole-task";
  if (!root)
    return fail(error, "configuration object required");
  if (root->get("program_search") && !search)
    return fail(error, "program_search must be an object");
  if (search)
    for (auto &entry : *search) {
      auto key = entry.first;
      if (key == "heterogeneous") {
        if (!entry.second.getAsBoolean())
          return fail(error, "heterogeneous search control must be boolean");
      } else if (key == "seconds") {
        if (!entry.second.getAsNumber())
          return fail(error, "seconds must be numeric");
      } else if (key == "dimensions") {
        if (!entry.second.getAsArray())
          return fail(error, "dimensions must be an array");
      } else if (key == "tiles") {
        auto tiles = entry.second.getAsObject();
        if (!tiles)
          return fail(error, "tiles must be an object");
        for (auto &axis : *tiles) {
          if (axis.first != "m" && axis.first != "n" && axis.first != "k")
            return fail(error, "unknown tile axis");
          if (!axis.second.getAsArray())
            return fail(error, "tile domain must be an array");
        }
      } else if (key == "max_evaluations" || key == "max_variables" ||
                 key == "max_constraints" || key == "max_tuple_cells" ||
                 key == "max_values_per_dimension" || key == "max_routes" ||
                 key == "max_priorities" || key == "max_source_instances" ||
                 key == "max_full_evaluations" || key == "workers" ||
                 key == "seed") {
        if (!entry.second.getAsInteger())
          return fail(error, "search limit must be an integer: " + key.str());
      } else
        return fail(error, "unknown public search control: " + key.str());
    }
  if (options.backend != "air")
    return fail(
        error, "cp-sat-all currently uses the registered native AIR evaluator; "
               "other backend callbacks must be registered explicitly");
  const bool heterogeneous =
      search && search->getBoolean("heterogeneous").value_or(false);
  if (heterogeneous && whole)
    return fail(error, "combined CPU/NPU search requires the native adapter");
  const bool nativeComposite = nativeCompositeProvider(configuration);
  if (nativeComposite && whole)
    return fail(error,
                "complete native invocation costs require the native adapter");
  if (nativeComposite) {
    auto air = root->getObject("generic_air");
    const auto *provider = air->getObject("cost_model");
    if (provider->getString("schema") != "orbit.direct-air.cost.v1")
      return fail(error,
                  "native composite requires schema orbit.direct-air.cost.v1");
    auto calibration = provider->getString("calibration");
    if (calibration != "calibrated" && calibration != "uncalibrated")
      return fail(error,
                  "native composite requires explicit calibration status");
    if (!air->getObject("cost_target"))
      return fail(error, "native composite requires an explicit calibration "
                         "target/runtime contract");
  }
  ProgramMechanismModel checkedModel;
  if (!heterogeneous && !nativeComposite &&
      !parameters(configuration, checkedModel, error))
    return false;
  if (llvm::sys::fs::exists(options.outputDir + "/search.json"))
    return fail(error, "fresh search output required");
  if (auto ec = llvm::sys::fs::create_directories(options.outputDir))
    return fail(error, ec.message());
  ProgramSearchOptions budget;
  budget.maxEvaluations = 2048;
  budget.wallSeconds = 600;
  auto integer = [&](const char *key, int64_t fallback) {
    return search ? search->getInteger(key).value_or(fallback) : fallback;
  };
  if (search)
    if (auto seconds = search->getNumber("seconds"))
      budget.wallSeconds = *seconds;
  // Validate JSON's signed 64-bit values before narrowing to solver options.
  // Otherwise values such as 2^32 + 1 can silently become one worker.
  const int64_t workers = integer("workers", budget.workers);
  const int64_t seed = integer("seed", budget.seed);
  if (workers < 1 || workers > 16 || seed < 0 ||
      seed > std::numeric_limits<int>::max())
    return fail(error, "workers must be in [1,16] and seed in [0,INT_MAX]");
  budget.workers = static_cast<int>(workers);
  budget.seed = static_cast<int>(seed);
  budget.maxEvaluations = integer("max_evaluations", budget.maxEvaluations);
  budget.maxVariables = integer("max_variables", budget.maxVariables);
  budget.maxConstraints = integer("max_constraints", budget.maxConstraints);
  budget.maxTupleCells = integer("max_tuple_cells", budget.maxTupleCells);
  if (!std::isfinite(budget.wallSeconds) || budget.wallSeconds <= 0 ||
      budget.workers < 1 || budget.maxEvaluations < 1 ||
      budget.maxVariables < 1 || budget.maxConstraints < 1 ||
      budget.maxTupleCells < 1)
    return fail(error, "invalid public search budget/options");
  auto maxInstances = integer("max_source_instances", 512),
       maxFullEvaluations = integer("max_full_evaluations", 64);
  auto maxValues = integer("max_values_per_dimension", 16),
       maxRoutes = integer("max_routes", 16),
       maxPriorities = integer("max_priorities", 24);
  if (maxFullEvaluations < 0 || maxFullEvaluations > 100000 ||
      maxInstances < 1 || maxInstances > 100000 || maxValues < 1 ||
      maxValues > 256 || maxRoutes < 1 || maxRoutes > 1024 ||
      maxPriorities < 1 || maxPriorities > 720)
    return fail(error, "invalid finite domain limits");
  StaticConfig target;
  ProgramDataOrganizationChoice baseChoice;
  GenericMemoryBinding nativeBinding;
  GenericTargetCapabilities capabilities;
  if (whole) {
    if (!parseWholeTaskProgramGraphConfig(configuration, target, baseChoice,
                                          error))
      return false;
  } else {
    if (!getGenericTargetCapabilities(configuration, capabilities, error) ||
        !configureGenericMemory(configuration, capabilities.capacityBytes,
                                nativeBinding, error))
      return false;
    target = nativeBinding.architecture;
    auto joint = root->getObject("program_joint");
    if (joint && joint->get("architecture")) {
      if (!parseMemoryArchitecture(*joint->get("architecture"), target, error))
        return false;
    }
    for (size_t task = 0; task < facts.tasks.size(); ++task) {
      auto resource = programResource(target, nativeBinding.computeResource);
      auto memory = programMemory(target, nativeBinding.localMemory);
      if (!resource || !memory) {
        resource = nullptr;
        for (auto &r : target.resources)
          if (r.kind == "compute" && r.device == "npu") {
            resource = &r;
            break;
          }
        if (!resource)
          return fail(error, "NPU target has no available compute binding");
        memory = nullptr;
        for (auto &m : target.memories)
          if (memoryAccessible(target, m.id, resource->id) &&
              m.kind == "local-sram") {
            memory = &m;
            break;
          }
        if (!memory)
          return fail(error, "compute binding lacks accessible local memory");
      }
      baseChoice.placements.push_back(
          {task, resource->id, memory->id, "scalar"});
    }
  }
  for (auto &route : baseChoice.transferChoices)
    if (route.consumerTask > facts.tasks.size() ||
        route.storage >= facts.storages.size())
      return fail(
          error,
          "configured transfer source/consumer is outside the source graph");
  // The parser preserves placement input order; public task identities do not.
  std::sort(baseChoice.placements.begin(), baseChoice.placements.end(),
            [](auto &a, auto &b) { return a.task < b.task; });
  for (size_t t = 0; t < baseChoice.placements.size(); ++t)
    if (baseChoice.placements[t].task != t)
      return fail(error, "complete unique task placements required");
  ProgramSearchSpace space;
  bool generationComplete = true;
  std::map<std::string, bool> domainTruncated;
  j::Array domainNotes;
  auto add = [&](std::string id, std::string scope,
                 std::vector<j::Value> values, std::string reason = "",
                 bool truncated = false) {
    domainTruncated[id] = truncated;
    ProgramSearchDimension d;
    d.id = id;
    d.scope = scope;
    d.reason = reason;
    std::set<std::string> seen;
    for (auto &value : values) {
      auto key = staticFingerprint(value);
      if (!seen.insert(key).second)
        continue;
      if (d.choices.size() >= size_t(maxValues)) {
        domainTruncated[id] = true;
        break;
      }
      d.choices.push_back(
          {id + "." + std::to_string(d.choices.size()), std::move(value)});
    }
    if (d.choices.empty())
      d.choices.push_back({id + ".source", nullptr});
    d.status = d.choices.size() > 1 ? "searched" : "singleton_by_capability";
    space.dimensions.push_back(std::move(d));
  };
  add("source.fission", "source bodies",
      hasSplittableNativeLoops(*original)
          ? std::vector<j::Value>{j::Object{{"fission", false}},
                                  j::Object{{"fission", true}}}
          : std::vector<j::Value>{j::Object{{"fission", false}}},
      "Existing source loop fission legality applies.");
  auto initProbe = parse(text);
  std::string initWhy;
  bool hasInitialization =
      initProbe && distributeNativeInitialization(*initProbe, initWhy) &&
      print(*initProbe) != print(*original);
  add("source.initialization", "source initializer distribution",
      hasInitialization
          ? std::vector<j::Value>{j::Object{{"distribute", false}},
                                  j::Object{{"distribute", true}}}
          : std::vector<j::Value>{j::Object{{"distribute", false}}},
      "Only source-changing initialization actions are admitted.");
  std::vector<int64_t> sizes(3, 1);
  for (auto &task : facts.tasks)
    for (size_t i = 0; i < std::min(size_t(3), task.loops.size()); ++i)
      sizes[i] = std::max(sizes[i], (task.loops[i].upper - task.loops[i].lower -
                                     1) / task.loops[i].step +
                                        1);
  if (!whole) {
    for (size_t axis = 0; axis < 3; ++axis) {
      std::vector<j::Value> values{
          j::Object{{"tile_axis", int64_t(axis)}, {"tile", int64_t(0)}}};
      std::vector<int64_t> candidates{
          sizes[axis], std::max(int64_t(1), (sizes[axis] + 1) / 2), 1};
      if (search)
        if (auto tiles = search->getObject("tiles"))
          if (auto array = tiles->getArray(axis == 0   ? "m"
                                           : axis == 1 ? "n"
                                                       : "k")) {
            candidates.clear();
            for (auto &v : *array) {
              auto n = v.getAsInteger();
              if (!n || *n < 1)
                return fail(error, "positive tile domains required");
              candidates.push_back(*n);
            }
          }
      for (auto n : candidates)
        values.push_back(j::Object{{"tile_axis", int64_t(axis)}, {"tile", n}});
      add(axis == 0   ? "tiling.m"
          : axis == 1 ? "tiling.n"
                      : "tiling.k",
          "existing native phase domains", std::move(values));
    }
    std::vector<j::Value> impl{
        j::Object{{"implementation", "scalar"}, {"lanes", 1}}};
    for (auto &i : capabilities.implementations)
      impl.push_back(j::Object{{"implementation", i.kind}, {"lanes", i.lanes}});
    add("implementation", "existing target implementation catalogue",
        std::move(impl));
    // Local selectors are independent of the module default. Null explicitly
    // inherits it, so choice zero preserves the pre-existing baseline policy.
    for (size_t task = 0; task < facts.tasks.size(); ++task) {
      std::vector<j::Value> local{nullptr};
      local.push_back(j::Object{
          {"task", int64_t(task)}, {"implementation", "scalar"}, {"lanes", 1}});
      for (const auto &implementation : capabilities.implementations)
        local.push_back(j::Object{{"task", int64_t(task)},
                                  {"implementation", implementation.kind},
                                  {"lanes", implementation.lanes}});
      add("task." + std::to_string(task) + ".implementation",
          "strict native task implementation; inactive on CPU",
          std::move(local));
    }
    ProgramMappingChoice current;
    if (!readNativeMappingChoice(*original, current, error))
      return false;
    std::vector<j::Value> mappings{j::Object{{"mapping", mapping(current)}}};
    for (auto &m : discoverGenericMappingChoices(facts, maxValues, maxValues))
      mappings.push_back(j::Object{{"mapping", mapping(m)}});
    add("fusion.residency.traversal", "public ProgramMappingChoice",
        std::move(mappings));
    std::vector<j::Value> partitions{j::Object{{"partition", 0}}};
    for (auto n :
         {sizes[0], std::max(int64_t(1), (sizes[0] + 1) / 2), int64_t(1)}) {
      ProgramPartitionPlan proof;
      std::string why;
      if (planProgramPartition(facts, n, proof, why))
        partitions.push_back(j::Object{{"partition", n}});
    }
    add("replica.partition", "proved independent source-axis coverage",
        std::move(partitions),
        "Actual concurrent binding comes from graph placement; partition alone "
        "does not imply parallel execution.");
  } else {
    for (auto id : {"tiling.m", "tiling.n", "tiling.k",
                    "fusion.residency.traversal", "replica.partition"})
      add(id, "whole-task adapter", {nullptr},
          "This adapter has full source task instances. Use the native phase "
          "adapter for loop tiling; source fission remains active.");
  }
  add("copies.retention", "public version/copy lifetime",
      {j::Object{{"policy", programDataPolicyName(baseChoice.policy).str()}},
       j::Object{{"policy", "phase-local"}},
       j::Object{{"policy", "retain-across-users"}}});
  add("data.regions", "public exact access region partition",
      whole
          ? std::vector<j::Value>{j::Object{{"partition_exact_regions",
                                             baseChoice.partitionExactRegions}},
                                  j::Object{{"partition_exact_regions", false}},
                                  j::Object{{"partition_exact_regions", true}}}
          : std::vector<j::Value>{nullptr},
      "Native phase regions already derive from checked tile domains.");
  add("dependencies.implementation_serialization",
      "public semantic-only versus serial whole-task witness",
      whole ? std::vector<j::Value>{j::Object{
                                        {"semantic_dependencies_only",
                                         baseChoice.semanticDependenciesOnly}},
                                    j::Object{
                                        {"semantic_dependencies_only", false}},
                                    j::Object{
                                        {"semantic_dependencies_only", true}}}
            : std::vector<j::Value>{nullptr},
      "Native semantic dependencies are preserved; implementation group "
      "serialization is rescheduled by the public timing evaluator.");
  add("buffering.slots",
      "public allocation aliases and buffer anti-dependencies",
      {j::Object{{"reuse_slots", false}}, j::Object{{"reuse_slots", true}}});
  for (auto &p : baseChoice.placements) {
    std::vector<j::Value> resources{
        j::Object{{"task", int64_t(p.task)}, {"resource", p.resource}}},
        memories{j::Object{{"task", int64_t(p.task)}, {"memory", p.memory}}},
        counts{j::Object{{"task", int64_t(p.task)},
                         {"resource_count", p.resourceCount}}},
        shapes{j::Object{{"task", int64_t(p.task)},
                         {"resource_shape", numbers(p.resourceShape)}}};
    int64_t capacity = 1;
    for (auto &r : target.resources)
      if (r.kind == "compute" &&
          (whole || r.device == programResource(target, p.resource)->device)) {
        resources.push_back(
            j::Object{{"task", int64_t(p.task)}, {"resource", r.id}});
        capacity = std::max(capacity, r.capacity);
      }
    for (auto &m : target.memories)
      if (std::any_of(
              target.resources.begin(), target.resources.end(), [&](auto &r) {
                return r.kind == "compute" &&
                       (whole ||
                        r.device ==
                            programResource(target, p.resource)->device) &&
                       memoryAccessible(target, m.id, r.id);
              }))
        memories.push_back(
            j::Object{{"task", int64_t(p.task)}, {"memory", m.id}});
    for (int64_t n = 1; n <= std::min(capacity, maxValues); ++n) {
      counts.push_back(
          j::Object{{"task", int64_t(p.task)}, {"resource_count", n}});
      if (capacity > 1)
        for (int64_t a = 1; a <= n; ++a)
          if (n % a == 0)
            shapes.push_back(j::Object{{"task", int64_t(p.task)},
                                       {"resource_shape", j::Array{a, n / a}}});
    }
    bool countTruncated = capacity > maxValues;
    auto prefix = "task." + std::to_string(p.task);
    add(prefix + ".resource", "public placement", std::move(resources));
    add(prefix + ".memory", "public accessible storage", std::move(memories));
    add(prefix + ".count", "public resource demand", std::move(counts), "",
        countTruncated);
    add(prefix + ".shape", "public resource geometry", std::move(shapes));
  }
  for (size_t task = 0; task <= facts.tasks.size(); ++task) {
    std::set<size_t> reads;
    if (task == facts.tasks.size()) {
      for (size_t st = 0; st < facts.storages.size(); ++st)
        if (facts.storages[st].role == "output")
          reads.insert(st);
    } else
      for (auto &a : facts.tasks[task].accesses)
        if (a.read)
          reads.insert(a.storage);
    for (auto storage : reads) {
      std::vector<j::Value> alternatives{nullptr};
      bool truncated = false;
      for (auto &source : target.memories)
        for (auto &dest : target.memories)
          if (task == facts.tasks.size()
                  ? dest.kind == "system-bo"
                  : std::any_of(
                        target.resources.begin(), target.resources.end(),
                        [&](auto &r) {
                          return r.kind == "compute" &&
                                 (whole ||
                                  r.device ==
                                      programResource(
                                          target,
                                          baseChoice.placements[task].resource)
                                          ->device) &&
                                 memoryAccessible(target, dest.id, r.id);
                        })) {
            std::vector<std::vector<std::string>> routes;
            paths(target, source.id, dest.id, maxRoutes, routes, truncated);
            for (auto &route : routes) {
              j::Array links;
              for (auto &l : route)
                links.push_back(l);
              j::Object selected{{"storage", int64_t(storage)},
                                 {"consumer_task", int64_t(task)},
                                 {"source_memory", source.id},
                                 {"target_memory", dest.id},
                                 {"links", std::move(links)}};
              alternatives.push_back(j::Object{{"route", std::move(selected)}});
            }
          }
      add("task." + std::to_string(task) + ".data." + std::to_string(storage) +
              ".route",
          "public source copy and multi-hop path", std::move(alternatives), "",
          truncated);
    }
  }
  std::vector<int64_t> priority(facts.tasks.size());
  std::iota(priority.begin(), priority.end(), 0);
  std::vector<j::Value> priorities;
  bool priorityTruncated = false;
  do {
    priorities.push_back(j::Object{{"priority", numbers(priority)}});
    if (priorities.size() >= size_t(maxPriorities)) {
      auto next = priority;
      if (std::next_permutation(next.begin(), next.end()))
        priorityTruncated = true;
      break;
    }
  } while (std::next_permutation(priority.begin(), priority.end()));
  add("temporal.priority", "public ProgramTimingRequest.schedulePriority",
      std::move(priorities), "", priorityTruncated);
  add("layout.dtype", "source and target copy/implementation contracts",
      {nullptr},
      "The existing native/common graph checker admits row_major source dtype. "
      "Other layouts require a registered backend implementation, not a "
      "relabelled copy.");
  if (heterogeneous) {
    for (size_t task = 0; task < facts.tasks.size(); ++task)
      add("device.task." + std::to_string(task),
          "source lineage CPU/NPU placement",
          {j::Object{{"task", int64_t(task)}, {"device", "cpu"}},
           j::Object{{"task", int64_t(task)}, {"device", "npu"}}},
          "Combined with every native choice; NPU child bindings are "
          "conditional on device placement.");
    for (size_t task = 1; task < facts.tasks.size(); ++task)
      add("boundary.before." + std::to_string(task),
          "source lineage execution-call boundary",
          {j::Object{{"before_task", int64_t(task)}, {"cut", false}},
           j::Object{{"before_task", int64_t(task)}, {"cut", true}}},
          "Device changes imply boundaries; same-device cuts select additional "
          "complete calls.");
  }
  // An explicit domain replaces its automatic dimension by ID. It never
  // appends another writer that silently shadows an already searched property.
  std::set<std::string> customIds;
  if (search)
    if (auto extra = search->getArray("dimensions"))
      for (auto &v : *extra) {
        auto o = v.getAsObject();
        auto id = o ? o->getString("id") : std::nullopt;
        auto choices = o ? o->getArray("choices") : nullptr;
        if (!id || !choices || choices->empty())
          return fail(error,
                      "extra dimension requires id and nonempty choices");
        for (auto &field : *o)
          if (field.first != "id" && field.first != "choices")
            return fail(error, "unknown dimension declaration field: " +
                                   field.first.str());
        if (!customIds.insert(id->str()).second)
          return fail(error, "duplicate explicit dimension ID: " + id->str());
        std::vector<j::Value> values;
        for (auto &x : *choices)
          values.push_back(j::Value(x));
        auto it = std::find_if(space.dimensions.begin(), space.dimensions.end(),
                               [&](auto &d) { return d.id == *id; });
        if (it != space.dimensions.end())
          space.dimensions.erase(it);
        add(id->str(), "declared public choice domain", std::move(values));
      }
  std::map<std::string, std::string> writers;
  j::Array declared;
  for (auto &d : space.dimensions) {
    std::set<std::string> properties;
    j::Array choices;
    for (auto &c : d.choices) {
      if (!binding(c.binding, whole, heterogeneous, facts.tasks.size(),
                   properties, error))
        return false;
      choices.push_back(
          j::Object{{"id", c.id}, {"binding", j::Value(c.binding)}});
    }
    for (auto &property : properties) {
      auto inserted = writers.emplace(property, d.id);
      if (!inserted.second)
        return fail(error, "conflicting dimensions bind " + property + ": " +
                               inserted.first->second + " and " + d.id);
    }
    if (d.id == "layout.dtype")
      d.status = "fixed_by_source";
    else if (customIds.count(d.id) && d.choices.size() == 1)
      d.status = "fixed_by_domain";
    if (heterogeneous &&
        (d.id == "data.regions" ||
         d.id == "dependencies.implementation_serialization")) {
      d.status = "unsupported_by_adapter";
      d.reason = "Native child derives exact regions and implementation order; "
                 "an independent whole-task rebind is not an executable native "
                 "choice.";
    }
    if (whole &&
        (d.id.rfind("tiling.", 0) == 0 ||
         d.id == "fusion.residency.traversal" || d.id == "replica.partition"))
      d.status = "unsupported_by_adapter";
    declared.push_back(j::Object{{"id", d.id},
                                 {"scope", d.scope},
                                 {"status", d.status},
                                 {"reason", d.reason},
                                 {"choices", std::move(choices)}});
  }
  // Cheap public target relations are CP-SAT tables. Source/version/group
  // compatibility is checked after composing the complete assignment.
  auto dimension = [&](const std::string &id) {
    for (size_t i = 0; i < space.dimensions.size(); ++i)
      if (space.dimensions[i].id == id)
        return i;
    return SIZE_MAX;
  };
  auto relation = [&](size_t a, size_t b, size_t task,
                      const std::function<bool(const j::Value &,
                                               const j::Value &)> &accept) {
    if (a == SIZE_MAX || b == SIZE_MAX || a == b)
      return;
    // NPU bindings are inactive on CPU children. Applying their resource
    // relations unconditionally would reject an executable CPU assignment
    // before the evaluator can see it. Locate the actual property writer,
    // including custom compound dimensions, rather than relying on its ID.
    size_t device = SIZE_MAX;
    if (heterogeneous) {
      auto writer = writers.find("device." + std::to_string(task));
      if (writer != writers.end())
        device = dimension(writer->second);
    }
    ProgramSearchConstraint table;
    table.dimensions = {a, b};
    const bool separateDevice =
        device != SIZE_MAX && device != a && device != b;
    if (separateDevice)
      table.dimensions.push_back(device);
    const size_t deviceChoices =
        separateDevice ? space.dimensions[device].choices.size() : 1;
    for (size_t x = 0; x < space.dimensions[a].choices.size(); ++x)
      for (size_t y = 0; y < space.dimensions[b].choices.size(); ++y)
        for (size_t z = 0; z < deviceChoices; ++z) {
          const auto &left = space.dimensions[a].choices[x].binding;
          const auto &right = space.dimensions[b].choices[y].binding;
          const auto *selectedDevice =
              device == a   ? left.getAsObject()
              : device == b ? right.getAsObject()
              : separateDevice
                  ? space.dimensions[device].choices[z].binding.getAsObject()
                  : nullptr;
          const bool inactive =
              heterogeneous &&
              (!selectedDevice ||
               selectedDevice->getInteger("task") != int64_t(task) ||
               selectedDevice->getString("device") != "npu");
          if (inactive || accept(left, right)) {
            std::vector<int64_t> tuple{int64_t(x), int64_t(y)};
            if (separateDevice)
              tuple.push_back(int64_t(z));
            table.allowedTuples.push_back(std::move(tuple));
          }
        }
    space.constraints.push_back(std::move(table));
  };
  // Custom choices may bind different tasks. These cheap relations only
  // inspect properties of the current task; the complete checker resolves
  // inherited/default bindings and all remaining combinations.
  for (auto &p : baseChoice.placements) {
    auto prefix = "task." + std::to_string(p.task);
    auto r = dimension(prefix + ".resource"), m = dimension(prefix + ".memory"),
         n = dimension(prefix + ".count"), shape = dimension(prefix + ".shape");
    relation(r, m, p.task, [&](auto &a, auto &b) {
      auto left = a.getAsObject(), right = b.getAsObject();
      if (!left || !right || left->getInteger("task") != int64_t(p.task) ||
          right->getInteger("task") != int64_t(p.task))
        return true;
      auto resource = left->getString("resource"),
           memory = right->getString("memory");
      return !resource || !memory ||
             memoryAccessible(target, *memory, *resource);
    });
    relation(r, n, p.task, [&](auto &a, auto &b) {
      auto left = a.getAsObject(), right = b.getAsObject();
      if (!left || !right || left->getInteger("task") != int64_t(p.task) ||
          right->getInteger("task") != int64_t(p.task))
        return true;
      auto id = left->getString("resource");
      auto count = right->getInteger("resource_count");
      if (!id || !count)
        return true;
      auto resource = programResource(target, *id);
      return resource && *count >= 1 && *count <= resource->capacity;
    });
    relation(n, shape, p.task, [&](auto &a, auto &b) {
      auto left = a.getAsObject(), right = b.getAsObject();
      if (!left || !right || left->getInteger("task") != int64_t(p.task) ||
          right->getInteger("task") != int64_t(p.task))
        return true;
      auto count = left->getInteger("resource_count");
      auto extents = right->getArray("resource_shape");
      if (!count || !extents)
        return true;
      if (extents->empty())
        return true;
      int64_t product = 1;
      for (auto &v : *extents) {
        auto n = v.getAsInteger();
        if (!n || *n < 1 || product > INT64_MAX / *n)
          return false;
        product *= *n;
      }
      return product == *count;
    });
    for (size_t st = 0; st < facts.storages.size(); ++st)
      relation(m, dimension(prefix + ".data." + std::to_string(st) + ".route"),
               p.task, [&](auto &a, auto &b) {
                 if (b.getAsNull())
                   return true;
                 auto left = a.getAsObject(), right = b.getAsObject();
                 if (!left || !right)
                   return true;
                 auto memory = left->getString("memory");
                 auto route = right->getObject("route");
                 return !memory || !route ||
                        left->getInteger("task") != int64_t(p.task) ||
                        route->getInteger("consumer_task") != int64_t(p.task) ||
                        route->getString("target_memory") == memory;
               });
  }
  generationComplete = true;
  for (auto &d : space.dimensions)
    generationComplete &= !domainTruncated[d.id];
  auto evaluationConfig = copyJSONWithoutFields(
      configuration, {"program_search", "native_auto_candidates"});
  auto evaluatorID = staticFingerprint(evaluationConfig);
  j::Array tables;
  for (auto &c : space.constraints) {
    j::Array tuples;
    for (auto &tuple : c.allowedTuples)
      tuples.push_back(numbers(tuple));
    tables.push_back(j::Object{{"dimensions", indices(c.dimensions)},
                               {"allowed_tuples", std::move(tuples)}});
  }
  j::Object description{{"schema", "orbit.public-program-space.v26"},
                        {"source_fingerprint", facts.bodyFingerprint},
                        {"evaluator_id", evaluatorID},
                        {"dimensions", std::move(declared)},
                        {"constraints", std::move(tables)},
                        {"generation_complete", generationComplete}};
  description["combined_cpu_npu_native"] = heterogeneous;
  auto domainID = staticFingerprint(j::Value(j::Object(description)));
  description["id"] = domainID;
  if (!save(options.outputDir + "/space.json", j::Value(std::move(description)),
            error))
    return false;
  // Every dimension is materialized together; no reuse-first or tile-first
  // stage freezes a decision. The callback owns semantic/backend validation.
  j::Array archive;
  size_t evaluations = 0, semanticCacheHits = 0, callbackCount = 0;
  bool evidenceError = false;
  std::optional<double> incumbentCost;
  struct CachedEvaluation {
    bool feasible, known;
    std::optional<double> cost;
    std::string message, path;
  };
  std::map<std::string, CachedEvaluation> cache;
  auto evaluate = [&](const std::vector<int64_t> &assignment) {
    ++callbackCount;
    auto evaluateImpl = [&]() {
      ProgramSearchEvaluation out;
      // A failed combined source/native refinement is an implementation gap,
      // not a proof that no physical CPU/NPU realization of the choice exists.
      out.feasibilityKnown = !heterogeneous;
      std::string why;
      mlir::MLIRContext context(registry);
      context.disableMultithreading();
      auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      auto cfg = copyJSONWithoutFields(configuration);
      auto choice = baseChoice;
      std::vector<int64_t> tile(3, 0), order(facts.tasks.size());
      std::iota(order.begin(), order.end(), 0);
      bool reuse = false, explicitCopyPolicy = false;
      int64_t partition = 0;
      std::string impl;
      int64_t lanes = 1;
      std::vector<NativeTaskImplementationChoice> originalImplementations;
      if (!whole &&
          !readNativeTaskImplementations(*module, facts.tasks.size(),
                                         originalImplementations, why)) {
        out.message = why;
        return out;
      }
      HeterogeneousMappingChoice deviceChoice;
      deviceChoice.taskDevices.assign(facts.tasks.size(), "cpu");
      GenericAnalysis tagged;
      if (!module || !analyzeGenericProgram(*module, tagged, why)) {
        out.message = why;
        return out;
      }
      for (size_t t = 0; t < tagged.tasks.size(); ++t)
        tagged.tasks[t].operation->setAttr(
            "orbit.search_origin",
            mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), t));
      for (size_t i = 0; i < assignment.size(); ++i) {
        auto &value = space.dimensions[i].choices[assignment[i]].binding;
        if (value.getAsNull())
          continue;
        auto b = value.getAsObject();
        if (!b) {
          out.message = "choice binding is not an object";
          return out;
        }
        for (auto &entry : *b)
          if (entry.first != "fission" && entry.first != "distribute" &&
              entry.first != "tile_axis" && entry.first != "tile" &&
              entry.first != "implementation" && entry.first != "lanes" &&
              entry.first != "mapping" && entry.first != "partition" &&
              entry.first != "policy" && entry.first != "reuse_slots" &&
              entry.first != "task" && entry.first != "resource" &&
              entry.first != "memory" && entry.first != "resource_count" &&
              entry.first != "resource_shape" && entry.first != "route" &&
              entry.first != "priority" &&
              entry.first != "partition_exact_regions" &&
              entry.first != "semantic_dependencies_only" &&
              entry.first != "device" && entry.first != "before_task" &&
              entry.first != "cut") {
            out.message =
                "unconsumed public decision field: " + entry.first.str();
            return out;
          }
        if (b->getBoolean("fission").value_or(false) &&
            !splitNativeOuterLoops(*module, why)) {
          out.message = why;
          return out;
        }
        if (b->getBoolean("distribute").value_or(false) &&
            !distributeNativeInitialization(*module, why)) {
          out.message = why;
          return out;
        }
        if (auto axis = b->getInteger("tile_axis")) {
          if (*axis < 0 || *axis > 2 || !b->getInteger("tile")) {
            out.message = "invalid tile choice";
            return out;
          }
          tile[*axis] = *b->getInteger("tile");
        }
        if (auto m = b->get("mapping")) {
          ProgramMappingChoice x;
          if (!mapping(*m, x, why) ||
              !setNativeMappingChoice(*module, x, why)) {
            out.message = why;
            return out;
          }
        }
        if (auto x = b->getString("implementation")) {
          if (whole) {
            if (auto task = b->getInteger("task"))
              choice.placements[*task].implementation = x->str();
            else
              for (auto &p : choice.placements)
                p.implementation = x->str();
          } else if (auto task = b->getInteger("task")) {
            auto old =
                llvm::find_if(originalImplementations, [&](const auto &v) {
                  return v.task == size_t(*task);
                });
            NativeTaskImplementationChoice selected{
                size_t(*task), x->str(), b->getInteger("lanes").value_or(1)};
            if (old == originalImplementations.end())
              originalImplementations.push_back(std::move(selected));
            else
              *old = std::move(selected);
          } else {
            impl = x->str();
            lanes = b->getInteger("lanes").value_or(1);
          }
        }
        if (auto x = b->getInteger("partition"))
          partition = *x;
        if (auto device = b->getString("device")) {
          auto task = b->getInteger("task");
          if (!heterogeneous || !task || *task < 0 ||
              size_t(*task) >= deviceChoice.taskDevices.size()) {
            out.message = "invalid combined source task/device binding";
            return out;
          }
          deviceChoice.taskDevices[*task] = device->str();
        }
        if (auto before = b->getInteger("before_task"))
          if (b->getBoolean("cut").value_or(false))
            deviceChoice.cuts.push_back(*before);
        if (auto x = b->getString("policy")) {
          explicitCopyPolicy = true;
          if (!parseProgramDataPolicy(*x, choice.policy, why)) {
            out.message = why;
            return out;
          }
        }
        if (auto x = b->getBoolean("reuse_slots"))
          reuse = *x;
        if (auto x = b->getBoolean("partition_exact_regions"))
          choice.partitionExactRegions = *x;
        if (auto x = b->getBoolean("semantic_dependencies_only"))
          choice.semanticDependenciesOnly = *x;
        if (auto x = b->getInteger("task")) {
          if (*x < 0 || size_t(*x) >= choice.placements.size()) {
            out.message = "unknown placement task";
            return out;
          }
          auto &p = choice.placements[*x];
          if (auto x = b->getString("resource"))
            p.resource = x->str();
          if (auto x = b->getString("memory"))
            p.memory = x->str();
          if (auto x = b->getInteger("resource_count"))
            p.resourceCount = *x;
          if (auto a = b->getArray("resource_shape")) {
            p.resourceShape.clear();
            for (auto &v : *a) {
              auto n = v.getAsInteger();
              if (!n || *n < 1) {
                out.message = "invalid shape extent";
                return out;
              }
              p.resourceShape.push_back(*n);
            }
          }
        }
        if (auto r = b->getObject("route")) {
          ProgramTransferChoice t;
          auto st = r->getInteger("storage"),
               task = r->getInteger("consumer_task");
          auto links = r->getArray("links");
          if (!st || *st < 0 || !task || *task < 0 || !links) {
            out.message = "invalid public transfer selection";
            return out;
          }
          t.storage = *st;
          t.consumerTask = *task;
          t.sourceMemory = r->getString("source_memory").value_or("").str();
          t.targetMemory = r->getString("target_memory").value_or("").str();
          for (auto &l : *links) {
            auto name = l.getAsString();
            if (!name) {
              out.message = "invalid path link";
              return out;
            }
            t.links.push_back(name->str());
          }
          choice.transferChoices.erase(
              std::remove_if(choice.transferChoices.begin(),
                             choice.transferChoices.end(),
                             [&](auto &old) {
                               return old.storage == t.storage &&
                                      old.consumerTask == t.consumerTask;
                             }),
              choice.transferChoices.end());
          choice.transferChoices.push_back(t);
        }
        if (auto p = b->getArray("priority")) {
          order.clear();
          for (auto &v : *p) {
            auto n = v.getAsInteger();
            if (!n || *n < 0 || size_t(*n) >= facts.tasks.size()) {
              out.message = "invalid source priority";
              return out;
            }
            order.push_back(*n);
          }
          auto sorted = order;
          std::sort(sorted.begin(), sorted.end());
          for (size_t k = 0; k < sorted.size(); ++k)
            if (sorted[k] != int64_t(k)) {
              out.message = "priority must be a task permutation";
              return out;
            }
          if (order.size() != facts.tasks.size()) {
            out.message = "priority coverage missing";
            return out;
          }
        }
      }
      GenericAnalysis transformed;
      if (!analyzeGenericProgram(*module, transformed, why)) {
        out.message = why;
        return out;
      }
      std::vector<std::vector<size_t>> descendants(facts.tasks.size());
      for (size_t t = 0; t < transformed.tasks.size(); ++t) {
        auto origin =
            transformed.tasks[t].operation->getAttrOfType<mlir::IntegerAttr>(
                "orbit.search_origin");
        if (!origin || origin.getInt() < 0 ||
            size_t(origin.getInt()) >= facts.tasks.size()) {
          out.message = "source transform lost task lineage";
          return out;
        }
        descendants[origin.getInt()].push_back(t);
      }
      std::vector<size_t> initializerTasks;
      if (!readNativeInitializerTasks(transformed, initializerTasks, why)) {
        out.message = why;
        return out;
      }
      if (heterogeneous) {
        HeterogeneousMappingChoice expanded;
        expanded.implementationOwned = true;
        expanded.taskDevices.resize(transformed.tasks.size());
        for (size_t origin = 0; origin < descendants.size(); ++origin)
          for (auto child : descendants[origin])
            expanded.taskDevices[child] = deviceChoice.taskDevices[origin];
        for (auto before : deviceChoice.cuts) {
          if (before >= descendants.size() || descendants[before].empty()) {
            out.feasibilityKnown = false;
            out.message = "region boundary lost its transformed source lineage";
            return out;
          }
          expanded.cuts.push_back(descendants[before].front());
        }
        deviceChoice = std::move(expanded);
      }
      auto originalChoice = choice;
      choice.placements.clear();
      choice.transferChoices.clear();
      for (auto &p : originalChoice.placements)
        for (auto t : descendants[p.task]) {
          auto inherited = p;
          inherited.task = t;
          choice.placements.push_back(inherited);
        }
      std::sort(choice.placements.begin(), choice.placements.end(),
                [](auto &a, auto &b) { return a.task < b.task; });
      for (auto &route : originalChoice.transferChoices) {
        if (route.consumerTask == facts.tasks.size()) {
          auto inherited = route;
          inherited.consumerTask = transformed.tasks.size();
          choice.transferChoices.push_back(inherited);
          continue;
        }
        for (auto t : descendants[route.consumerTask])
          if (std::any_of(transformed.tasks[t].accesses.begin(),
                          transformed.tasks[t].accesses.end(), [&](auto &a) {
                            return a.read && a.storage == route.storage;
                          })) {
            auto inherited = route;
            inherited.consumerTask = t;
            choice.transferChoices.push_back(inherited);
          }
      }
      std::vector<int64_t> expandedOrder;
      for (auto t : order)
        for (auto child : descendants[t])
          expandedOrder.push_back(child);
      order = std::move(expandedOrder);
      ProgramMappingChoice inheritedMapping;
      if (!whole && !readNativeMappingChoice(*module, inheritedMapping, why)) {
        out.message = why;
        return out;
      }
      // A null public retention binding inherits the selected native mapping,
      // not the phase-local policy from before the mapping decision. Otherwise
      // a whole-program selection is needlessly rebuilt by the whole-task
      // adapter, replacing the native execution-group witness. An explicitly
      // selected public policy remains authoritative and must still be checked.
      if (!whole && !explicitCopyPolicy &&
          !parseProgramDataPolicy(inheritedMapping.dataOrganization,
                                  choice.policy, why)) {
        out.message = why;
        return out;
      }
      if (!whole && !inheritedMapping.residencyGroups.empty()) {
        for (auto &group : inheritedMapping.residencyGroups) {
          std::vector<size_t> expanded;
          for (auto t : group.tasks) {
            if (t >= descendants.size()) {
              out.message = "mapping task has no source lineage";
              return out;
            }
            expanded.insert(expanded.end(), descendants[t].begin(),
                            descendants[t].end());
          }
          group.tasks = std::move(expanded);
        }
        if (!setNativeMappingChoice(*module, inheritedMapping, why)) {
          out.message = why;
          return out;
        }
      }
      if (!whole) {
        std::vector<NativeTaskImplementationChoice> expanded;
        for (const auto &selected : originalImplementations)
          for (auto task : descendants[selected.task]) {
            if (heterogeneous && deviceChoice.taskDevices[task] == "cpu")
              continue;
            const bool initializer = descendants[selected.task].size() > 1 &&
                                     llvm::is_contained(initializerTasks, task);
            expanded.push_back({task, initializer ? "scalar" : selected.kind,
                                initializer ? 1 : selected.lanes});
          }
        if (!setNativeTaskImplementations(*module, transformed.tasks.size(),
                                          expanded, why)) {
          out.message = why;
          return out;
        }
      }
      if (!impl.empty() &&
          !setNativeImplementation(*module, impl, lanes, why)) {
        out.message = why;
        return out;
      }
      if (std::any_of(tile.begin(), tile.end(), [](auto n) { return n > 0; })) {
        ProgramMappingChoice selectedMapping;
        if (!readNativeMappingChoice(*module, selectedMapping, why)) {
          out.message = why;
          return out;
        }
        if (selectedMapping.residency == "whole-program") {
          out.message = "explicit tiled domains conflict with selected "
                        "whole-program residency";
          return out;
        }
        for (size_t i = 0; i < tile.size(); ++i)
          if (tile[i] == 0)
            tile[i] = sizes[i];
        if (!setNativeAIRTileSizes(*module, tile[0], tile[1], tile[2], why)) {
          out.message = why;
          return out;
        }
        cfg = copyJSONWithoutNestedFields(cfg, "generic_air", {"tile_sizes"});
      }
      if (partition > 0) {
        module->getOperation()->setAttr(
            "orbit.program_partition_extent",
            mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64),
                                   partition));
      }
      if (heterogeneous) {
        std::vector<mlir::Attribute> devices;
        for (const auto &device : deviceChoice.taskDevices)
          devices.push_back(mlir::StringAttr::get(&context, device));
        module->getOperation()->setAttr(
            "orbit.heterogeneous_devices",
            mlir::ArrayAttr::get(&context, devices));
        std::vector<int64_t> cuts(deviceChoice.cuts.begin(),
                                  deviceChoice.cuts.end());
        module->getOperation()->setAttr(
            "orbit.heterogeneous_cuts",
            mlir::DenseI64ArrayAttr::get(&context, cuts));
      }
      auto source = print(*module);
      j::Array placements;
      for (auto &p : choice.placements)
        placements.push_back(placement(p));
      j::Array routes;
      for (auto &t : choice.transferChoices) {
        j::Array links;
        for (auto &l : t.links)
          links.push_back(l);
        routes.push_back(j::Object{{"storage", int64_t(t.storage)},
                                   {"consumer_task", int64_t(t.consumerTask)},
                                   {"source_memory", t.sourceMemory},
                                   {"target_memory", t.targetMemory},
                                   {"links", std::move(links)}});
      }
      auto key = staticFingerprint(j::Object{
          {"source", source},
          {"placements", j::Array(placements)},
          {"policy", programDataPolicyName(choice.policy).str()},
          {"slots", reuse},
          {"partition_exact_regions", choice.partitionExactRegions},
          {"semantic_dependencies_only", choice.semanticDependenciesOnly},
          {"routes", std::move(routes)},
          {"priority", numbers(order)}});
      auto found = cache.find(key);
      if (found != cache.end()) {
        ++semanticCacheHits;
        auto &entry = found->second;
        out.feasible = entry.feasible;
        out.feasibilityKnown = entry.known;
        out.costMs = entry.cost;
        out.message = entry.message;
        out.sourceMLIR = source;
        // Reload only a candidate that can become incumbent. Equal/worse cached
        // candidates cannot replace the monotonically improving checked
        // incumbent.
        if (out.feasible && out.costMs &&
            (!incumbentCost || *out.costMs < *incumbentCost)) {
          auto buffer = llvm::MemoryBuffer::getFile(entry.path);
          if (!buffer) {
            out.feasibilityKnown = false;
            out.costMs.reset();
            out.message = "cached evidence unavailable";
          } else {
            auto parsed = j::parse((*buffer)->getBuffer());
            if (!parsed) {
              llvm::consumeError(parsed.takeError());
              out.feasibilityKnown = false;
              out.costMs.reset();
              out.message = "cached evidence invalid";
            } else
              out.result = std::move(*parsed);
          }
        }
        return out;
      }
      const auto folder =
          options.outputDir + "/candidate-" + std::to_string(evaluations++);
      if (auto ec = llvm::sys::fs::create_directories(folder)) {
        out.feasibilityKnown = false;
        out.message = ec.message();
        return out;
      }
      GenericAnalysis analyzed;
      ProgramGraph reference, graph;
      GenericMappingPlan native;
      ProgramMechanismModel model;
      ProgramTimingEvaluation schedule;
      if (!analyzeGenericProgram(*module, analyzed, why)) {
        out.message = why;
        return out;
      }
      auto evaluateGraph = [&](mlir::ModuleOp actual, ProgramGraph &selected,
                               j::Value &nativeReport,
                               ProgramTimingEvaluation &timed) {
        GenericAnalysis current;
        ProgramGraph reference;
        GenericMappingPlan native;
        auto localChoice = choice;
        if (!analyzeGenericProgram(actual, current, why))
          return false;
        if (!whole)
          if (auto tiles = actual->getAttrOfType<mlir::DenseI64ArrayAttr>(
                  "orbit.air_tile_sizes")) {
            auto extents = tiles.asArrayRef();
            if (extents.size() != 3 ||
                std::any_of(extents.begin(), extents.end(),
                            [](auto n) { return n < 1; })) {
              why = "source tile attribute requires three positive extents";
              return false;
            }
            int64_t total = 0;
            for (auto &task : current.tasks) {
              int64_t count = 1;
              for (size_t axis = 0;
                   axis < std::min(size_t(3), task.loops.size()); ++axis) {
                auto &loop = task.loops[axis];
                auto trips = (loop.upper - loop.lower - 1) / loop.step + 1;
                auto pieces = (trips - 1) / extents[axis] + 1;
                if (count > maxInstances / pieces) {
                  out.feasibilityKnown = false;
                  why = "source-instance materialization work cap reached "
                        "(search limit, feasibility unknown)";
                  return false;
                }
                count *= pieces;
              }
              if (total > maxInstances - count) {
                out.feasibilityKnown = false;
                why = "source-instance materialization work cap reached "
                      "(search limit, feasibility unknown)";
                return false;
              }
              total += count;
            }
          }

        if (whole) {
          StaticConfig declaredTarget;
          ProgramDataOrganizationChoice referenceChoice;
          if (!parseWholeTaskProgramGraphConfig(cfg, declaredTarget,
                                                referenceChoice, why) ||
              !planProgramDataOrganization(current, declaredTarget, localChoice,
                                           reference, why))
            return false;
        } else {
          if (!planGenericMapping(current, nativePlanningConfig(cfg), native,
                                  why))
            return false;
          reference = native.communication;
          for (auto &p : localChoice.placements) {
            auto it = std::find_if(reference.instances.begin(),
                                   reference.instances.end(),
                                   [&](auto &v) { return v.task == p.task; });
            if (it != reference.instances.end())
              p.implementation = it->implementation;
          }
        }
        if (nativeCompositeProvider(cfg) &&
            preservesNativeExecution(native, target, localChoice, order, reuse))
          selected = reference;
        else if (!rebindProgramGraph(reference, target, localChoice, selected,
                                     why) ||
                 (reuse && !mergeSlots(selected, why)))
          return false;
        if (nativeCompositeProvider(cfg)) {
          auto air = cfg.getAsObject()->getObject("generic_air");
          j::Value receipt = nullptr;
          if (!evaluateGenericNativeCompositeTiming(
                  current, native, selected, *air->get("cost_model"),
                  *air->get("cost_target"), timed, receipt, why)) {
            // A provider/schema/numerical failure is not proof that the checked
            // source graph has no physical realization. Prevent UNSAT after
            // tuple exclusion from turning this into an infeasibility proof.
            out.feasibilityKnown = false;
            return false;
          }
          nativeReport = serializeGenericMapping(current, native);
          (*nativeReport.getAsObject())["public_complete_invocation"] =
              std::move(receipt);
          return true;
        }
        if (!parameters(cfg, model, why) ||
            !timing(selected, model, cfg, order, timed, why)) {
          auto root = cfg.getAsObject();
          auto air = root ? root->getObject("generic_air") : nullptr;
          auto provider = air ? air->getObject("cost_model") : nullptr;
          if (provider && provider->getString("kind") == "region-invocation")
            out.feasibilityKnown = false;
          return false;
        }
        nativeReport = whole ? j::Value(nullptr)
                             : serializeGenericMapping(current, native);
        return true;
      };
      if (heterogeneous) {
        ProgramAllHeterogeneousCandidate candidate;
        if (!materializeProgramAllHeterogeneousCandidate(
                *module, cfg, deviceChoice, target, choice, order, reuse,
                maxInstances, candidate, why)) {
          out.feasibilityKnown = false;
          out.message = "combined refinement unknown: " + why;
          return out;
        }
        candidate.evidenceDirectory = folder;
        j::Array selectedBindings;
        for (size_t i = 0; i < assignment.size(); ++i)
          selectedBindings.push_back(j::Object{
              {"dimension", space.dimensions[i].id},
              {"binding",
               j::Value(space.dimensions[i].choices[assignment[i]].binding)}});
        candidate.assignment =
            j::Object{{"indices", numbers(assignment)},
                      {"bindings", std::move(selectedBindings)}};
        if (heterogeneousEvaluator)
          out = heterogeneousEvaluator(candidate);
        else {
          out.feasible = true;
          out.feasibilityKnown = false;
          out.message = "combined source/child plans checked; complete region "
                        "runtime provider is not registered";
        }
        if (out.costMs && (!std::isfinite(*out.costMs) || *out.costMs < 0)) {
          out.feasibilityKnown = false;
          out.costMs.reset();
          out.message =
              "combined provider returned an invalid complete latency";
        }
        if (out.feasible && out.costMs) {
          // A valid number is not enough: the provider's backend/cost receipt
          // must identify the same parent and every selected child graph.
          // Never stamp a foreign plan's price with this candidate's identity.
          const auto *receipt = out.result.getAsObject();
          const auto *children =
              receipt ? receipt->getArray("region_graph_ids") : nullptr;
          bool matches =
              receipt &&
              receipt->getString("candidate_id") == candidate.assignmentId &&
              receipt->getString("parent_graph_id") ==
                  programGraphIdentity(candidate.parent.graph) &&
              children && children->size() == candidate.regions.size();
          if (matches)
            for (size_t i = 0; i < children->size(); ++i)
              matches &=
                  (*children)[i].getAsString() ==
                  programGraphIdentity(candidate.regions[i].selectedGraph);
          if (!matches) {
            out.feasibilityKnown = false;
            out.costMs.reset();
            out.message = "combined provider receipt differs from selected "
                          "parent/child assignment";
          }
        }
        out.sourceMLIR = source;
        auto costEvaluation = std::move(out.result);
        out.result = j::Object{
            {"schema", "orbit.combined-native-heterogeneous-evaluation.v32"},
            {"candidate_id", candidate.assignmentId},
            {"assignment", j::Value(candidate.assignment)},
            {"parent_graph_id", programGraphIdentity(candidate.parent.graph)},
            {"program_graph", serializeProgramGraph(candidate.parent.graph)},
            {"combined_candidate", std::move(candidate.receipt)},
            {"cost_evaluation", std::move(costEvaluation)},
            {"estimated_ms",
             out.costMs ? j::Value(*out.costMs) : j::Value(nullptr)},
            {"backend_replanning_allowed", false}};
      } else if (partition > 0) {
        ProgramPartitionPlan proof;
        if (!planProgramPartition(analyzed, partition, proof, why)) {
          out.message = why;
          return out;
        }
        auto manifest = serializeProgramPartition(analyzed, proof);
        auto &witness = *manifest.getAsObject();
        j::Array templates;
        bool priced = true;
        double templateCost = 0;
        for (auto &entry : *witness.getArray("templates")) {
          auto &t = *entry.getAsObject();
          auto extent = *t.getInteger("extent");
          int64_t count = 0;
          for (auto &inv : *witness.getArray("invocations"))
            count += inv.getAsObject()->getString("template_id") ==
                     t.getString("id");
          mlir::OwningOpRef<mlir::ModuleOp> child;
          if (!materializeProgramPartition(*module, proof, extent, child,
                                           why)) {
            out.message = why;
            return out;
          }
          ProgramGraph selected;
          j::Value nativeReport = nullptr;
          ProgramTimingEvaluation timed;
          if (!evaluateGraph(*child, selected, nativeReport, timed)) {
            out.message = why;
            return out;
          }
          priced &= bool(timed.makespanMs);
          if (timed.makespanMs)
            templateCost += count * *timed.makespanMs;
          templates.push_back(
              j::Object{{"template_id", t.getString("id")->str()},
                        {"invocations", count},
                        {"program_graph", serializeProgramGraph(selected)},
                        {"native_reference", std::move(nativeReport)},
                        {"timing", serializeProgramTiming(timed)}});
        }
        out.feasible = true;
        out.sourceMLIR = source;
        out.message =
            "all decisions composed in every checked partition template; "
            "complete host slicing/assembly/control cost remains unknown";
        out.result = j::Object{
            {"status", "uncalibrated-complete-program-host-boundary"},
            {"partition", std::move(manifest)},
            {"templates", std::move(templates)},
            {"template_execution_estimated_ms",
             priced ? j::Value(templateCost) : j::Value(nullptr)},
            {"estimated_ms", nullptr},
            {"physical_realization_checked", false},
            {"missing_costs",
             "host slicing, assembly, and partition invocation control"}};
      } else {
        j::Value nativeReport = nullptr;
        if (!evaluateGraph(*module, graph, nativeReport, schedule)) {
          out.message = why;
          return out;
        }
        placements.clear();
        for (auto &p : graph.placements)
          placements.push_back(placement(p));
        out.feasible = true;
        out.costMs = schedule.makespanMs;
        out.sourceMLIR = source;
        out.message = out.costMs
                          ? "checked public graph and complete activity cost"
                          : "checked graph; unresolved cost remains unknown";
        out.result =
            j::Object{{"status", out.costMs ? "estimated" : "uncalibrated"},
                      {"program_graph", serializeProgramGraph(graph)},
                      {"timing", serializeProgramTiming(schedule)},
                      {"native_reference", std::move(nativeReport)},
                      {"selected_placements", std::move(placements)},
                      {"source_fingerprint", analyzed.bodyFingerprint},
                      {"graph_id", programGraphIdentity(graph)},
                      {"estimated_ms",
                       out.costMs ? j::Value(*out.costMs) : j::Value(nullptr)},
                      {"physical_realization_checked", false}};
      }
      bool improves = out.feasible && out.costMs &&
                      (!incumbentCost || *out.costMs < *incumbentCost);
      bool fullRecord = evaluations <= size_t(maxFullEvaluations) || improves;
      if ((fullRecord &&
           !save(folder + "/evaluation.json", out.result, error)) ||
          !save(folder + "/assignment.json", numbers(assignment), error)) {
        evidenceError = true;
        out.feasibilityKnown = false;
        out.message = error;
      }
      if (fullRecord) {
        std::error_code ec;
        const auto path = folder + "/taskflow.mlir";
        llvm::raw_fd_ostream stream(path, ec);
        bool persisted = false;
        if (ec)
          fail(error, "cannot open " + path + ": " + ec.message());
        else {
          stream << source;
          persisted = closeEvidence(stream, path, error);
        }
        if (!persisted) {
          evidenceError = true;
          out.feasibilityKnown = false;
          out.message = error;
        }
      }
      archive.push_back(j::Object{
          {"assignment", numbers(assignment)},
          {"feasible", out.feasible},
          {"cost_ms", out.costMs ? j::Value(*out.costMs) : j::Value(nullptr)},
          {"directory", folder},
          {"full_evaluation_saved", fullRecord},
          {"message", out.message}});
      if (archive.size() == 1 || archive.size() % 64 == 0 || improves)
        save(options.outputDir + "/archive.json", j::Value(j::Array(archive)),
             error);
      bool acceptedImprover = out.feasible && out.feasibilityKnown &&
                              out.costMs &&
                              (!incumbentCost || *out.costMs < *incumbentCost);
      if (acceptedImprover) {
        auto bestPath = options.outputDir + "/incumbent.json";
        if (save(bestPath,
                 j::Object{{"assignment", numbers(assignment)},
                           {"evaluation", j::Value(out.result)}},
                 error))
          incumbentCost = out.costMs;
        else {
          evidenceError = true;
          out.feasibilityKnown = false;
          out.message = error;
        }
      }
      cache.emplace(
          key, CachedEvaluation{out.feasible, out.feasibilityKnown, out.costMs,
                                out.message,
                                fullRecord ? folder + "/evaluation.json" : ""});
      return out;
    };
    auto out = evaluateImpl();
    if (!save(options.outputDir + "/progress.json",
              j::Object{{"evaluations", int64_t(callbackCount)},
                        {"materializations", int64_t(evaluations)},
                        {"semantic_cache_hits", int64_t(semanticCacheHits)},
                        {"elapsed_seconds",
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count()},
                        {"last_feasible", out.feasible},
                        {"last_message", out.message},
                        {"incumbent_cost_ms", incumbentCost
                                                  ? j::Value(*incumbentCost)
                                                  : j::Value(nullptr)}},
              error)) {
      evidenceError = true;
      out.feasibilityKnown = false;
      out.message = error;
    }
    if (std::all_of(assignment.begin(), assignment.end(),
                    [](auto x) { return x == 0; }))
      if (!save(options.outputDir + "/baseline.json",
                j::Object{{"assignment", numbers(assignment)},
                          {"feasible", out.feasible},
                          {"message", out.message},
                          {"evaluation", j::Value(out.result)}},
                error)) {
        evidenceError = true;
        out.feasibilityKnown = false;
        out.message = error;
      }
    return out;
  };
  // Discovery is part of the same wall quota; it never receives a fresh 600s.
  budget.wallSeconds -=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  auto automatic = root->getObject("native_auto_candidates");
  bool beam =
      automatic && automatic->getString("algorithm") == "orbit-public-beam";
  ProgramSearchResult solved;
  if (budget.wallSeconds <= 0) {
    // A valid user quota may be consumed by discovery before the solver is
    // entered. Passing the negative remainder as an option would mislabel a
    // timeout INVALID_INPUT. Do not start a callback with a fresh budget.
    solved.status = ConstraintSearchStatus::Unknown;
    const double elapsedMs = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
    solved.report = j::Object{
        {"algorithm",
         beam ? "orbit-public-beam" : "joint-cp-sat-branch-and-check"},
        {"status", "UNKNOWN"},
        {"reason", "Shared wall quota exhausted during domain discovery"},
        {"has_best", false},
        {"evaluations", 0},
        {"feasible_evaluations", 0},
        {"unknown_cost_evaluations", 0},
        {"unknown_feasibility_evaluations", 0},
        {"finite_domain_exhausted", false},
        {"finite_domain_optimality_proved", false},
        {"complete_plan_prepricing", false},
        {"budget_exhausted", true},
        {"callback_preemptible", false},
        {"total_ms", elapsedMs},
        {"callback_ms", 0},
        {"solve_ms", 0},
        {"model_build_ms", 0},
        {"evaluation_trace", j::Array{}},
        {"solve_trace", j::Array{}},
        {"returned_after_deadline", true},
        {"deadline_overrun_ms", -budget.wallSeconds * 1000.}};
  } else {
    solved = beam ? searchProgramSpaceBeam(space, evaluate, budget,
                                           options.beamWidth, options.maxRounds)
                  : searchProgramSpace(space, evaluate, budget);
  }
  if (evidenceError)
    return fail(error, "candidate evidence could not be persisted: " + error);
  if (!save(options.outputDir + "/archive.json", j::Value(j::Array(archive)),
            error))
    return false;
  auto report = j::Value(std::move(solved.report));
  auto &r = *report.getAsObject();
  r["backend"] = options.backend;
  r["schema"] = "orbit.public-program-search.v26";
  r["domain_id"] = domainID;
  r["evaluator_id"] = evaluatorID;
  r["unrestricted_optimum_proven"] = false;
  r["optimality_scope"] =
      "Only the declared finite public-choice domain and the frozen complete "
      "evaluator; not unrestricted abstract graphs or hardware optimality.";
  r["domain_generation_complete"] = generationComplete;
  r["original_source_fingerprint"] = facts.bodyFingerprint;
  r["physical_realization_checked"] = false;
  r["combined_cpu_npu_native"] = heterogeneous;
  r["combined_runtime_provider_registered"] = bool(heterogeneousEvaluator);
  r["semantic_cache_hits"] = int64_t(semanticCacheHits);
  r["unique_materializations"] = int64_t(evaluations);
  r["max_source_instances"] = maxInstances;
  r["max_full_evaluations"] = maxFullEvaluations;
  r["semantic_cache_payload"] =
      "metadata and evidence path only; no retained full evaluation JSON";
  r["best_mlir_is_complete_execution_plan"] = false;
  r["complete_plan_artifact"] =
      "best.json includes the selected graph and schedule; best.mlir alone "
      "contains only source attributes";
  // The search adapter does not define the provider's runtime boundary. Read
  // the selected timing receipt: a registered callback or a model schema alone
  // cannot establish that its complete invocation has been priced.
  r["cost_model_boundary"] = "unknown: no selected timing boundary receipt";
  r["cost_model_boundary_scope"] = "selected incumbent receipt";
  r["cost_model_boundary_priced"] = false;
  if (solved.hasBest) {
    const auto *evaluation = solved.bestEvaluation.result.getAsObject();
    const auto *cost = evaluation;
    if (heterogeneous && evaluation)
      cost = evaluation->getObject("cost_evaluation");
    const auto *timed = cost ? cost->getObject("timing") : nullptr;
    if (timed) {
      if (auto boundary = timed->getString("boundary");
          boundary && !boundary->empty()) {
        r["cost_model_boundary"] = boundary->str();
        r["cost_model_boundary_priced"] =
            solved.bestEvaluation.feasible &&
            solved.bestEvaluation.feasibilityKnown &&
            solved.bestEvaluation.costMs.has_value();
      }
    } else if (evaluation &&
               evaluation->getString("status") ==
                   "uncalibrated-complete-program-host-boundary") {
      r["cost_model_boundary"] =
          "unknown: partition host slicing, assembly, and invocation control";
    }
  }
  r["scope"] =
      "All supplied public dimensions composed by one branch-and-check master; "
      "source/target capabilities and finite-domain truncation explicit. No "
      "global hardware optimality claim.";
  if (!generationComplete) {
    r["unrestricted_optimum_proven"] = false;
    r["optimality_scope"] =
        "only the admitted finite domain; generated domains were truncated";
  }
  if (!save(options.outputDir + "/search.json", report, error))
    return false;
  if (solved.hasBest) {
    if (!save(options.outputDir + "/best.json", solved.bestEvaluation.result,
              error) ||
        !save(options.outputDir + "/best-assignment.json",
              numbers(solved.bestAssignment), error))
      return false;
    std::error_code ec;
    llvm::raw_fd_ostream stream(options.outputDir + "/best.mlir", ec);
    if (ec)
      return fail(error, ec.message());
    stream << solved.bestEvaluation.sourceMLIR;
    if (!closeEvidence(stream, options.outputDir + "/best.mlir", error))
      return false;
  }
  return solved.hasBest ||
         solved.status == ConstraintSearchStatus::Unavailable ||
         solved.status == ConstraintSearchStatus::Unknown ||
         solved.status == ConstraintSearchStatus::ModelLimit;
}

bool orbit::taskflow::replayPublicProgramGraph(
    mlir::ModuleOp source, const j::Value &configuration,
    const j::Value &spaceValue, const j::Value &assignmentValue,
    const j::Value &expectedValue, ProgramGraph &graph, j::Value &receipt,
    std::string &error) {
  auto declared = spaceValue.getAsObject(),
       expected = expectedValue.getAsObject();
  auto dimensions = declared ? declared->getArray("dimensions") : nullptr;
  auto assignment = assignmentValue.getAsArray();
  if (!dimensions || !assignment || dimensions->size() != assignment->size() ||
      !expected || !expected->getObject("program_graph") ||
      !expected->getObject("native_reference") ||
      !expected->getArray("selected_placements"))
    return fail(error, "replay requires a complete frozen native evaluation, "
                       "space and assignment");
  auto domainID = declared->getString("id");
  if (!domainID)
    return fail(error, "frozen space lacks identity");
  std::string savedID = domainID->str();
  if (staticFingerprint(copyJSONWithoutFields(spaceValue, {"id"})) != savedID)
    return fail(error, "frozen space identity mismatch");
  GenericAnalysis facts;
  GenericMappingPlan native;
  if (!analyzeGenericProgram(source, facts, error) ||
      !planGenericMapping(facts, nativePlanningConfig(configuration), native,
                          error))
    return false;
  if (staticFingerprint(serializeGenericMapping(facts, native)) !=
      staticFingerprint(*expected->get("native_reference")))
    return fail(error, "frozen native witness does not match materialized "
                       "source and target reconstruction");
  ProgramDataOrganizationChoice choice;
  // Match the search decoder's null-copy inheritance before replaying any
  // explicit policy override. The source already contains the frozen mapping.
  choice.policy = native.communication.dataPolicy;
  bool slots = false;
  std::vector<int64_t> order;
  for (auto &p : *expected->getArray("selected_placements")) {
    auto o = p.getAsObject();
    if (!o || !o->getInteger("task") || !o->getString("resource") ||
        !o->getString("memory") || !o->getString("implementation") ||
        !o->getInteger("resource_count") || !o->getArray("resource_shape"))
      return fail(error, "invalid frozen placement");
    MappingTaskPlacement x;
    x.task = *o->getInteger("task");
    x.resource = o->getString("resource")->str();
    x.memory = o->getString("memory")->str();
    x.implementation = o->getString("implementation")->str();
    x.resourceCount = *o->getInteger("resource_count");
    for (auto &v : *o->getArray("resource_shape")) {
      auto n = v.getAsInteger();
      if (!n || *n < 1)
        return fail(error, "invalid frozen shape");
      x.resourceShape.push_back(*n);
    }
    choice.placements.push_back(x);
  }
  for (size_t i = 0; i < assignment->size(); ++i) {
    auto n = (*assignment)[i].getAsInteger();
    auto d = (*dimensions)[i].getAsObject();
    auto c = d ? d->getArray("choices") : nullptr;
    if (!n || *n < 0 || !c || size_t(*n) >= c->size())
      return fail(error, "frozen assignment outside declared space");
    auto chosen = (*c)[*n].getAsObject();
    auto value = chosen ? chosen->get("binding") : nullptr;
    if (!value)
      return fail(error, "missing frozen binding");
    if (value->getAsNull())
      continue;
    auto b = value->getAsObject();
    if (!b)
      return fail(error, "invalid frozen binding");
    if (b->getBoolean("fission").value_or(false))
      return fail(error,
                  "fission replay not yet supported by the physical bridge");
    if (auto requested = b->getBoolean("distribute")) {
      bool distributed =
          std::any_of(facts.tasks.begin(), facts.tasks.end(), [](auto &t) {
            return t.operation->hasAttr("orbit.distributed_initialization");
          });
      if (distributed != *requested)
        return fail(
            error,
            "frozen initializer distribution differs from materialized source");
    }
    if (auto value = b->get("mapping")) {
      ProgramMappingChoice selectedMapping, actualMapping, normalizedMapping;
      if (!mapping(*value, selectedMapping, error) ||
          !readNativeMappingChoice(source, actualMapping, error))
        return false;
      for (auto &group : selectedMapping.residencyGroups) {
        std::vector<size_t> expanded;
        for (auto parent : group.tasks)
          for (size_t ti = 0; ti < facts.tasks.size(); ++ti) {
            auto origin =
                facts.tasks[ti].operation->getAttrOfType<mlir::IntegerAttr>(
                    "orbit.search_origin");
            if (origin && origin.getInt() == int64_t(parent))
              expanded.push_back(ti);
          }
        group.tasks = std::move(expanded);
      }
      mlir::OwningOpRef<mlir::ModuleOp> normalized(
          mlir::cast<mlir::ModuleOp>(source->clone()));
      if (!setNativeMappingChoice(*normalized, selectedMapping, error) ||
          !readNativeMappingChoice(*normalized, normalizedMapping, error))
        return false;
      if (staticFingerprint(mapping(normalizedMapping)) !=
          staticFingerprint(mapping(actualMapping)))
        return fail(
            error,
            "frozen mapping assignment differs from materialized source");
    }
    if (auto axis = b->getInteger("tile_axis")) {
      auto tile = b->getInteger("tile");
      auto sizes = source->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "orbit.air_tile_sizes");
      if (!tile || *axis < 0 || *axis > 2 ||
          (*tile > 0 && (!sizes || sizes.size() != 3 || sizes[*axis] != *tile)))
        return fail(error,
                    "frozen tile assignment differs from materialized source");
    }
    if (auto impl = b->getString("implementation")) {
      if (auto original = b->getInteger("task")) {
        bool found = false;
        for (size_t ti = 0; ti < facts.tasks.size(); ++ti) {
          auto origin =
              facts.tasks[ti].operation->getAttrOfType<mlir::IntegerAttr>(
                  "orbit.search_origin");
          if (!origin || origin.getInt() != *original)
            continue;
          found = true;
          std::vector<size_t> initializerTasks;
          if (!readNativeInitializerTasks(facts, initializerTasks, error))
            return false;
          size_t descendantCount = 0;
          for (const auto &task : facts.tasks) {
            auto parent = task.operation->getAttrOfType<mlir::IntegerAttr>(
                "orbit.search_origin");
            descendantCount += parent && parent.getInt() == *original;
          }
          const bool initializer =
              descendantCount > 1 && llvm::is_contained(initializerTasks, ti);
          const auto &actual = native.taskImplementations[ti];
          if (actual.kind != (initializer ? "scalar" : impl->str()) ||
              actual.lanes !=
                  (initializer ? 1 : b->getInteger("lanes").value_or(1)))
            return fail(
                error,
                "frozen local implementation differs from source witness");
        }
        if (!found)
          return fail(error, "frozen local implementation has no lineage");
      } else {
        auto selected =
            source->getAttrOfType<mlir::StringAttr>("orbit.implementation");
        auto laneAttr =
            source->getAttrOfType<mlir::IntegerAttr>("orbit.vector_lanes");
        if ((selected ? selected.getValue() : "scalar") != *impl ||
            (laneAttr ? laneAttr.getInt() : 1) !=
                b->getInteger("lanes").value_or(1))
          return fail(
              error, "frozen module implementation differs from source policy");
      }
    }
    if (auto task = b->getInteger("task"))
      for (size_t ti = 0; ti < facts.tasks.size(); ++ti) {
        auto origin =
            facts.tasks[ti].operation->getAttrOfType<mlir::IntegerAttr>(
                "orbit.search_origin");
        if (!origin || origin.getInt() != *task)
          continue;
        auto p =
            std::find_if(choice.placements.begin(), choice.placements.end(),
                         [&](auto &p) { return p.task == ti; });
        if (p == choice.placements.end())
          return fail(error, "frozen placement task has no lineage");
        if ((b->getString("resource") &&
             p->resource != *b->getString("resource")) ||
            (b->getString("memory") && p->memory != *b->getString("memory")) ||
            (b->getInteger("resource_count") &&
             p->resourceCount != *b->getInteger("resource_count")))
          return fail(
              error, "frozen assignment placement differs from selected graph");
        if (auto shape = b->getArray("resource_shape"))
          if (staticFingerprint(j::Value(j::Array(*shape))) !=
              staticFingerprint(j::Value(numbers(p->resourceShape))))
            return fail(error, "frozen shape differs from selected graph");
      }
    if (auto p = b->getString("policy"))
      if (!parseProgramDataPolicy(*p, choice.policy, error))
        return false;
    if (auto p = b->getBoolean("reuse_slots"))
      slots = *p;
    if (auto p = b->getInteger("partition"))
      if (*p)
        return fail(error, "partition host boundary is not a supported replay");
    if (b->get("route"))
      return fail(error, "explicit route replay is not yet supported; refusing "
                         "to drop binding");
    if (auto a = b->getArray("priority"))
      for (auto &v : *a) {
        auto n = v.getAsInteger();
        if (!n || *n < 0)
          return fail(error, "invalid frozen priority");
        order.push_back(*n);
      }
  }
  if (!order.empty()) {
    std::vector<int64_t> expanded;
    for (auto original : order)
      for (size_t task = 0; task < facts.tasks.size(); ++task) {
        auto origin =
            facts.tasks[task].operation->getAttrOfType<mlir::IntegerAttr>(
                "orbit.search_origin");
        if (!origin)
          return fail(error, "frozen source lacks origin lineage");
        if (origin.getInt() == original)
          expanded.push_back(task);
      }
    order = std::move(expanded);
  } else {
    order.resize(facts.tasks.size());
    std::iota(order.begin(), order.end(), 0);
  }
  auto sorted = order;
  std::sort(sorted.begin(), sorted.end());
  if (sorted.size() != facts.tasks.size())
    return fail(error, "frozen priority lacks transformed task coverage");
  for (size_t i = 0; i < sorted.size(); ++i)
    if (sorted[i] != int64_t(i))
      return fail(error,
                  "frozen priority is not a transformed task permutation");
  StaticConfig target = native.memoryBinding.architecture;
  if (auto root = configuration.getAsObject())
    if (auto joint = root->getObject("program_joint"))
      if (auto a = joint->get("architecture"))
        if (!parseMemoryArchitecture(*a, target, error))
          return false;
  if (!rebindProgramGraph(native.communication, target, choice, graph, error) ||
      (slots && !mergeSlots(graph, error)))
    return false;
  if (staticFingerprint(serializeProgramGraph(graph)) !=
      staticFingerprint(*expected->get("program_graph")))
    return fail(error, "replayed complete public graph differs from frozen "
                       "candidate; no remapping permitted");
  ProgramMechanismModel model;
  ProgramTimingEvaluation timed;
  if (!parameters(configuration, model, error) ||
      !timing(graph, model, configuration, order, timed, error))
    return false;
  if (!expected->get("timing") ||
      staticFingerprint(serializeProgramTiming(timed)) !=
          staticFingerprint(*expected->get("timing")))
    return fail(error, "replayed frozen analytical schedule differs");
  receipt = j::Object{{"schema", "orbit.public-program.replay.v27"},
                      {"valid", true},
                      {"graph_id", programGraphIdentity(graph)},
                      {"native_reference_id", native.planId},
                      {"complete_graph_matches", true},
                      {"analytical_timing_matches", true},
                      {"search_performed", false},
                      {"physical_realization_checked", false}};
  return true;
}
