#include "Taskflow/GenericMapping.h"
#include "Taskflow/ProgramOperatorSource.h"
#include "Taskflow/ProgramTargetBinding.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <sstream>

using namespace orbit::taskflow;
namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;
bool fail(std::string &error, const std::string &message) {
  error = "program graph: " + message;
  return false;
}
bool add(int64_t &sum, int64_t amount) {
  if (amount < 0 || sum > INT64_MAX - amount)
    return false;
  sum += amount;
  return true;
}
Array numbers(const std::vector<int64_t> &values) {
  Array out;
  for (auto value : values)
    out.push_back(value);
  return out;
}
Array indices(const std::vector<size_t> &values) {
  Array out;
  for (auto value : values)
    out.push_back(int64_t(value));
  return out;
}
int64_t regionBytes(const GenericAnalysis &source,
                    const MappingRegion &region) {
  if (region.storage >= source.storages.size())
    return -1;
  const auto &storage = source.storages[region.storage];
  if (region.offsets.size() != storage.shape.size() ||
      region.sizes.size() != storage.shape.size())
    return -1;
  auto type = mlir::dyn_cast<mlir::MemRefType>(storage.root.getType());
  if (!type)
    return -1;
  int64_t bytes = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
  for (size_t axis = 0; axis < region.sizes.size(); ++axis) {
    auto size = region.sizes[axis], offset = region.offsets[axis];
    if (size <= 0 || offset < 0 || offset > storage.shape[axis] ||
        size > storage.shape[axis] - offset || bytes > INT64_MAX / size)
      return -1;
    bytes *= size;
  }
  return bytes;
}
bool inside(const MappingRegion &cell, const MappingRegion &region) {
  if (cell.storage != region.storage ||
      cell.sizes.size() != region.sizes.size())
    return false;
  for (size_t axis = 0; axis < cell.sizes.size(); ++axis)
    if (cell.offsets[axis] < region.offsets[axis] ||
        cell.offsets[axis] + cell.sizes[axis] >
            region.offsets[axis] + region.sizes[axis])
      return false;
  return true;
}
bool sameJSON(const llvm::json::Value &left, const llvm::json::Value &right) {
  return staticFingerprint(left) == staticFingerprint(right);
}
const StaticResource *resource(const ProgramGraph &graph,
                               const std::string &id) {
  auto found = std::find_if(graph.resources.begin(), graph.resources.end(),
                            [&](const auto &value) { return value.id == id; });
  return found == graph.resources.end() ? nullptr : &*found;
}
bool precedes(const ProgramGraph &graph, size_t parent, size_t child) {
  if (parent >= child)
    return false;
  std::vector<size_t> stack{child};
  std::set<size_t> seen;
  while (!stack.empty()) {
    size_t current = stack.back();
    stack.pop_back();
    if (!seen.insert(current).second)
      continue;
    for (size_t dependency : graph.events[current].dependencies) {
      if (dependency == parent)
        return true;
      if (dependency > parent && dependency < current)
        stack.push_back(dependency);
    }
  }
  return false;
}
bool linkDuration(const ProgramGraph &graph, const MappingEvent &event,
                  double &duration) {
  auto link =
      std::find_if(graph.links.begin(), graph.links.end(),
                   [&](const auto &value) { return value.id == event.link; });
  if (link == graph.links.end() || !std::isfinite(link->bandwidthBytesPerMs) ||
      link->bandwidthBytesPerMs <= 0 || !std::isfinite(link->latencyMs) ||
      link->latencyMs < 0 || link->provenance.source.empty())
    return false;
  duration = link->latencyMs + event.transferBytes / link->bandwidthBytesPerMs;
  return std::isfinite(duration) && duration >= 0;
}
// Check execution coverage independently of copy bookkeeping. Project each
// source access onto its actual ancestor loops: sibling nests are not a single
// Cartesian iteration space. Distributed initializers have a separate adapter
// proof relating their body to the controlling output tile.
bool checkSourceInstanceCoverage(const ProgramGraph &graph,
                                 std::string &error) {
  for (size_t task = 0; task < graph.source.tasks.size(); ++task) {
    const auto &source = graph.source.tasks[task];
    std::vector<const MappingTaskInstance *> instances;
    for (const auto &instance : graph.instances)
      if (instance.task == task && instance.kind == "source-task")
        instances.push_back(&instance);
    if (instances.empty())
      continue;
    std::set<std::vector<size_t>> projections;
    for (const auto &access : source.accesses) {
      std::vector<size_t> axes;
      for (size_t loop = 0; loop < source.loops.size(); ++loop)
        if (source.loops[loop].operation->isProperAncestor(access.operation))
          axes.push_back(loop);
      projections.insert(std::move(axes));
    }
    for (const auto &axes : projections) {
      using Box = std::vector<std::pair<int64_t, int64_t>>;
      std::vector<Box> boxes;
      std::vector<std::set<int64_t>> cuts(axes.size());
      for (size_t axis = 0; axis < axes.size(); ++axis) {
        const auto &loop = source.loops[axes[axis]];
        cuts[axis] = {0, (loop.upper - loop.lower - 1) / loop.step + 1};
      }
      for (const auto *instance : instances) {
        Box box;
        for (size_t axis = 0; axis < axes.size(); ++axis) {
          const auto &loop = source.loops[axes[axis]];
          auto range = std::find_if(
              instance->domain.begin(), instance->domain.end(),
              [&](const auto &value) { return value.loop == axes[axis]; });
          if (range == instance->domain.end())
            return fail(error, "execution coverage lacks source loop");
          int64_t begin = (range->lower - loop.lower) / loop.step;
          int64_t end = (range->upper - loop.lower - 1) / loop.step + 1;
          box.push_back({begin, end});
          cuts[axis].insert(begin);
          cuts[axis].insert(end);
        }
        boxes.push_back(std::move(box));
      }
      size_t expected = 1;
      std::vector<std::vector<int64_t>> grid;
      for (const auto &axis : cuts) {
        if (expected > 1048576 / (axis.size() - 1))
          return fail(
              error,
              "source execution coverage exceeds finite checker cell limit");
        expected *= axis.size() - 1;
        grid.emplace_back(axis.begin(), axis.end());
      }
      std::set<std::vector<size_t>> covered;
      for (const auto &box : boxes) {
        std::vector<size_t> position(axes.size());
        std::function<bool(size_t)> visit = [&](size_t axis) {
          if (axis == axes.size())
            return covered.insert(position).second;
          auto begin = std::lower_bound(grid[axis].begin(), grid[axis].end(),
                                        box[axis].first);
          auto end = std::lower_bound(grid[axis].begin(), grid[axis].end(),
                                      box[axis].second);
          for (auto cell = begin; cell != end; ++cell) {
            position[axis] = cell - grid[axis].begin();
            if (!visit(axis + 1))
              return false;
          }
          return true;
        };
        if (!visit(0))
          return fail(error, "source access iteration executes more than once");
      }
      if (covered.size() != expected)
        return fail(error, "source access iteration domain is incomplete");
    }
  }
  return true;
}
// Preserve the established serial plan. The extra predecessor records resource
// and schedule serialization explicitly without changing semantic dependencies.
bool schedule(const ProgramGraph &graph, MappingGraphSchedule &out,
              std::string &error) {
  const auto policy = graph.schedule.policy;
  if (policy == "constraint-resource-schedule") {
    if (!checkProgramResourceSchedule(graph, error))
      return false;
    out = graph.schedule;
    return true;
  }
  if (policy != "serial-event-order" && policy != "dependency-order-witness")
    return fail(error, "unsupported graph schedule policy");
  out = {};
  out.policy = policy;
  bool known = policy == "serial-event-order";
  if (!known)
    out.uncalibrated.push_back(
        "backend resource schedule and cycle durations are separately bound; "
        "no millisecond schedule inferred");
  double time = 0;
  for (size_t index = 0; index < graph.events.size(); ++index) {
    const auto &event = graph.events[index];
    MappingScheduledEvent item;
    item.event = index;
    item.predecessors = event.dependencies;
    if (index && policy == "serial-event-order")
      item.predecessors.push_back(index - 1);
    std::sort(item.predecessors.begin(), item.predecessors.end());
    item.predecessors.erase(
        std::unique(item.predecessors.begin(), item.predecessors.end()),
        item.predecessors.end());
    if (std::any_of(item.predecessors.begin(), item.predecessors.end(),
                    [&](size_t parent) { return parent >= index; }))
      return fail(error, "schedule has a forward or cyclic predecessor");
    auto execution = resource(graph, event.resource);
    if (!execution || execution->capacity < 1)
      return fail(error, "schedule uses missing execution resource");
    double duration = 0;
    bool calibrated = false;
    if (event.kind == "transfer")
      calibrated = linkDuration(graph, event, duration);
    else {
      auto estimate = std::find_if(
          graph.computeEstimates.begin(), graph.computeEstimates.end(),
          [&](const auto &value) {
            return value.task == event.task && value.resource == event.resource;
          });
      if (estimate != graph.computeEstimates.end()) {
        if (estimate->resourceCount > execution->capacity)
          return fail(error, "compute demand exceeds resource capacity");
        // Preserve the estimate unit and convert with the explicitly supplied
        // graph clock; missing cycle conversion never invents a duration.
        std::string reason;
        calibrated = convertComputeEstimateToMs(*estimate, graph.clockMHz,
                                                duration, reason);
      }
    }
    if (!calibrated) {
      out.uncalibrated.push_back("event:" + std::to_string(index) + ":" +
                                 event.kind);
      known = false;
    }
    if (known) {
      item.startMs = time;
      time += duration;
      if (!std::isfinite(time))
        return fail(error, "schedule duration overflows");
      item.endMs = time;
    }
    out.events.push_back(std::move(item));
  }
  for (const auto &region : graph.executionRegions)
    if (region.submissionCount > 0) {
      // No decomposed launch parameter is inferred from a fitted whole-command
      // intercept. The native composite provider owns the complete call once.
      out.uncalibrated.push_back("region:" + region.id +
                                 ":submission-and-completion");
      known = false;
    }
  if (known)
    out.estimatedMakespanMs = time;
  return true;
}
llvm::json::Value scheduleJSON(const MappingGraphSchedule &value) {
  Array events, missing;
  for (const auto &event : value.events)
    events.push_back(
        Object{{"event", int64_t(event.event)},
               {"predecessors", indices(event.predecessors)},
               {"start_ms", event.startMs ? llvm::json::Value(*event.startMs)
                                          : llvm::json::Value(nullptr)},
               {"end_ms", event.endMs ? llvm::json::Value(*event.endMs)
                                      : llvm::json::Value(nullptr)}});
  for (const auto &reason : value.uncalibrated)
    missing.push_back(reason);
  Object result{
      {"policy", value.policy},
      {"events", std::move(events)},
      {"estimated_makespan_ms",
       value.estimatedMakespanMs ? llvm::json::Value(*value.estimatedMakespanMs)
                                 : llvm::json::Value(nullptr)},
      {"uncalibrated", std::move(missing)},
      {"timing_boundary",
       value.policy == "serial-event-order"
           ? "serial compute and transfer events; explicit region call "
             "overhead remains unknown unless separately calibrated"
           : "dependency witness only; backend cycle/resource schedule is "
             "separately bound"}};
  if (value.policy == "constraint-resource-schedule") {
    Array costs, configurations;
    for (auto cost : value.eventDurationsMs)
      costs.push_back(cost);
    for (const auto &c : value.configurations)
      configurations.push_back(Object{{"groups", indices(c.groups)},
                                      {"start_ms", c.startMs},
                                      {"end_ms", c.endMs},
                                      {"setup_ms", c.setupMs}});
    result["model_identity"] = value.modelIdentity;
    result["timing_boundary"] = value.timingBoundary;
    result["event_durations_ms"] = std::move(costs);
    result["configurations"] = std::move(configurations);
  }
  return result;
}
llvm::json::Value graphJSON(const ProgramGraph &graph) {
  Array instances, transfers, regions, logicalData, sourceDependencies;
  for (size_t storage = 0; storage < graph.source.storages.size(); ++storage) {
    const auto &value = graph.source.storages[storage];
    Array cells, versions;
    for (size_t index = 0; index < graph.regions.size(); ++index)
      if (graph.regions[index].storage == storage)
        cells.push_back(int64_t(index));
    for (size_t index = 0; index < graph.versions.size(); ++index)
      if (graph.regions[graph.versions[index].region].storage == storage)
        versions.push_back(int64_t(index));
    logicalData.push_back(Object{{"storage", int64_t(storage)},
                                 {"id", value.id},
                                 {"role", value.role},
                                 {"dtype", value.dtype},
                                 {"shape", numbers(value.shape)},
                                 {"layout", "row_major"},
                                 {"regions", std::move(cells)},
                                 {"versions", std::move(versions)}});
  }
  for (size_t task = 0; task < graph.source.tasks.size(); ++task)
    for (size_t parent : graph.source.tasks[task].dependencies) {
      std::set<std::string> hazards;
      for (const auto &before : graph.source.tasks[parent].accesses)
        for (const auto &after : graph.source.tasks[task].accesses) {
          if (before.storage != after.storage)
            continue;
          bool overlaps = true;
          for (size_t axis = 0; axis < before.region.size(); ++axis)
            overlaps &= before.region[axis].lower < after.region[axis].upper &&
                        after.region[axis].lower < before.region[axis].upper;
          if (!overlaps)
            continue;
          if (before.write && after.read)
            hazards.insert("RAW");
          if (before.read && after.write)
            hazards.insert("WAR");
          if (before.write && after.write)
            hazards.insert("WAW");
        }
      if (hazards.empty())
        hazards.insert("ordering-only");
      Array kinds;
      for (const auto &kind : hazards)
        kinds.push_back(kind);
      sourceDependencies.push_back(
          Object{{"producer_task", int64_t(parent)},
                 {"consumer_task", int64_t(task)},
                 {"kinds", std::move(kinds)},
                 {"scope", "source bounding-overlap facts; exact consumed "
                           "versions and copies are in ledger data edges"}});
    }
  for (const auto &instance : graph.instances) {
    Array domain;
    for (const auto &loop : instance.domain)
      domain.push_back(Object{{"loop", int64_t(loop.loop)},
                              {"lower", loop.lower},
                              {"upper", loop.upper},
                              {"step", loop.step}});
    Object entry{{"task", int64_t(instance.task)},
                 {"event", int64_t(instance.event)},
                 {"phase", int64_t(instance.phase)},
                 {"domain_task", int64_t(instance.domainTask)},
                 {"kind", instance.kind},
                 {"domain", std::move(domain)},
                 {"implementation", instance.implementation},
                 {"lanes", instance.lanes},
                 {"parallel_loop", instance.parallelLoop}};
    if (instance.resourceCount != 1 || !instance.resourceShape.empty()) {
      entry["resource_count"] = instance.resourceCount;
      entry["resource_shape"] = numbers(instance.resourceShape);
    }
    instances.push_back(std::move(entry));
  }
  for (size_t index = 0; index < graph.events.size(); ++index) {
    const auto &event = graph.events[index];
    if (event.kind != "transfer" || !event.transferRegion)
      continue;
    const auto &region = *event.transferRegion;
    transfers.push_back(Object{{"event", int64_t(index)},
                               {"kind", "copy"},
                               {"storage", int64_t(region.storage)},
                               {"offsets", numbers(region.offsets)},
                               {"sizes", numbers(region.sizes)},
                               {"bytes", event.transferBytes},
                               {"source_copies", indices(event.reads)},
                               {"target_copies", indices(event.writes)},
                               {"link", event.link},
                               {"resource", event.resource},
                               {"dependencies", indices(event.dependencies)}});
  }
  for (const auto &region : graph.executionRegions) {
    Array inputs, terminal;
    std::set<size_t> consumed;
    for (size_t index = region.beginEvent;
         index < region.endEvent && index < graph.events.size(); ++index)
      for (size_t parent : graph.events[index].dependencies)
        if (parent >= region.beginEvent)
          consumed.insert(parent);
    for (size_t index = region.beginEvent;
         index < region.endEvent && index < graph.events.size(); ++index)
      if (!consumed.count(index))
        terminal.push_back(int64_t(index));
    for (size_t edge = 0; edge < graph.dataEdges.size(); ++edge)
      if (graph.dataEdges[edge].producer < 0 &&
          graph.dataEdges[edge].consumer >= region.beginEvent &&
          graph.dataEdges[edge].consumer < region.endEvent)
        inputs.push_back(int64_t(edge));
    Object entry{
        {"id", region.id},
        {"placement", region.placement},
        {"begin_event", int64_t(region.beginEvent)},
        {"end_event", int64_t(region.endEvent)},
        {"input_data_edges", std::move(inputs)},
        {"submission_event", Object{{"id", region.id + ".submit"},
                                    {"resource", region.submissionResource},
                                    {"count", region.submissionCount},
                                    {"shared_by_region_inputs", true}}},
        {"completion_event", Object{{"id", region.id + ".complete"},
                                    {"after", std::move(terminal)}}}};
    if (!region.sourceTasks.empty()) {
      entry["source_tasks"] = indices(region.sourceTasks);
      entry["member_events"] = indices(region.memberEvents);
      entry["input_copies"] = indices(region.inputCopies);
      entry["output_copies"] = indices(region.outputCopies);
      entry["predecessor_regions"] = indices(region.predecessorRegions);
      entry["implementation"] = region.implementation;
      entry["endpoint_memory"] = region.endpointMemory;
    }
    if (!region.invocationLifecycle.empty())
      entry["invocation_lifecycle"] = region.invocationLifecycle;
    regions.push_back(std::move(entry));
  }
  std::string body;
  if (graph.source.function) {
    llvm::raw_string_ostream stream(body);
    graph.source.function->print(stream);
  }
  Object inputs, outputs;
  for (const auto &entry : graph.inputLocations)
    inputs[std::to_string(entry.first)] = entry.second;
  for (const auto &entry : graph.outputLocations)
    outputs[std::to_string(entry.first)] = entry.second;
  Object result{
      {"schema", "orbit.program-graph.v1"},
      {"source", graph.operatorSource
                     ? serializeProgramOperatorSource(*graph.operatorSource)
                     : serializeGenericAnalysis(graph.source)},
      {"computation_mlir", body},
      {"logical_data", std::move(logicalData)},
      {"source_dependencies", std::move(sourceDependencies)},
      {"boundaries", Object{{"inputs", std::move(inputs)},
                            {"outputs", std::move(outputs)},
                            {"output_ownership",
                             "external-after-completion; lifetime endpoints "
                             "are clipped to the graph event domain"}}},
      {"instances", std::move(instances)},
      {"communications", std::move(transfers)},
      {"execution_regions", std::move(regions)},
      {"schedule", scheduleJSON(graph.schedule)},
      {"allocation_lifetime", graph.allocationLifetime},
      {"ledger", serializeProgramCommunication(graph)},
      {"supported_communication",
       Array{"copy", "shared-read-of-existing-copy"}},
      {"unsupported_communication",
       Array{"stream", "multicast", "arbitrary-cross-stage-retention"}},
      {"source_access_versions",
       "conservative may-overlap source facts; exact finite execution versions "
       "are in the shared ledger"},
      {"source_binding", graph.operatorSource
                             ? "hash-bound external operator/tensor/state "
                               "contract; registered implementation coverage "
                               "and ABI are checked, not native MLIR arithmetic"
                             : "borrowed actual MLIR operations and values; "
                               "serialized source is a "
                               "view, not an executable graph import"}};
  if (!graph.targetIdentity.empty())
    result["target_identity"] = graph.targetIdentity;
  Array bindings;
  for (size_t a = 0; a < graph.allocations.size(); ++a) {
    const auto &v = graph.allocations[a];
    if (v.addressing.empty() && v.spatialOwner.empty())
      continue;
    Object x{{"allocation", int64_t(a)}};
    if (!v.addressing.empty())
      x["addressing"] = v.addressing;
    if (!v.spatialOwner.empty())
      x["spatial_owner"] = v.spatialOwner;
    bindings.push_back(std::move(x));
  }
  if (!bindings.empty())
    result["allocation_bindings"] = std::move(bindings);
  if (!graph.executionGroups.empty()) {
    Array groups;
    for (size_t i = 0; i < graph.executionGroups.size(); ++i) {
      const auto &g = graph.executionGroups[i];
      Object x{{"id", int64_t(i)},
               {"begin_phase", int64_t(g.beginPhase)},
               {"end_phase", int64_t(g.endPhase)},
               {"resource", g.resource},
               {"memory", g.memory},
               {"dependencies", indices(g.dependencies)},
               {"member_events", indices(g.memberEvents)}};
      if (!g.spatialOwner.empty())
        x["spatial_owner"] = g.spatialOwner;
      if (!g.implementationDependencies.empty())
        x["implementation_dependencies"] =
            indices(g.implementationDependencies);
      groups.push_back(std::move(x));
    }
    result["execution_groups"] = std::move(groups);
  }
  if (!graph.regionEvents.empty()) {
    Array lifecycle;
    for (size_t i = 0; i < graph.regionEvents.size(); ++i) {
      const auto &event = graph.regionEvents[i];
      lifecycle.push_back(Object{{"id", int64_t(i)},
                                 {"kind", event.kind},
                                 {"region", int64_t(event.region)},
                                 {"resource", event.resource},
                                 {"dependencies", indices(event.dependencies)},
                                 {"copies", indices(event.copies)}});
    }
    result["region_events"] = std::move(lifecycle);
  }
  if (graph.operatorSource) {
    result["source_kind"] = "source-bound-external-operators";
    if (graph.operatorSelection)
      result["external_implementation_selection"] =
          serializeProgramOperatorSelection(*graph.operatorSelection);
  }
  return result;
}
} // namespace

bool orbit::taskflow::parseWholeTaskProgramGraphConfig(
    const llvm::json::Value &config, StaticConfig &architecture,
    ProgramDataOrganizationChoice &choice, std::string &error) {
  architecture = {};
  choice = {};
  const auto *root = config.getAsObject();
  const auto *graph = root ? root->getObject("program_graph") : nullptr;
  const auto *target = graph ? graph->get("architecture") : nullptr;
  const auto *decisions = graph ? graph->getObject("choice") : nullptr;
  if (!graph || graph->getString("adapter") != "whole-task" || !target ||
      !decisions)
    return fail(error, "whole-task configuration requires program_graph "
                       "adapter, architecture and choice");
  if (!parseMemoryArchitecture(*target, architecture, error))
    return false;
  const auto *assumptions = target->getAsObject()->getObject("assumptions");
  if (!assumptions)
    return fail(error, "whole-task architecture requires explicit input/output "
                       "boundary locations");
  auto locations = [&](llvm::StringRef key,
                       std::map<std::string, std::string> &out) {
    auto values = assumptions->getObject(key);
    if (!values)
      return false;
    for (const auto &value : *values) {
      auto memory = value.second.getAsString();
      if (!memory || memory->empty())
        return false;
      out[value.first.str()] = memory->str();
    }
    return true;
  };
  if (!locations("initial_locations",
                 architecture.assumptions.initialLocations) ||
      !locations("output_locations", architecture.assumptions.outputLocations))
    return fail(error, "invalid graph boundary location map");
  auto policy = decisions->getString("policy");
  auto placements = decisions->getArray("placements");
  if (!policy || !parseProgramDataPolicy(*policy, choice.policy, error) ||
      !placements)
    return fail(error, "choice requires a policy and task placements");
  for (const auto &value : *placements) {
    auto object = value.getAsObject();
    if (!object)
      return fail(error, "invalid task placement");
    auto task = object->getInteger("task");
    auto resource = object->getString("resource"),
         memory = object->getString("memory"),
         implementation = object->getString("implementation");
    if (!task || *task < 0 || !resource || resource->empty() || !memory ||
        memory->empty() || !implementation || implementation->empty())
      return fail(error,
                  "placement requires task/resource/memory/implementation");
    MappingTaskPlacement placement{size_t(*task), resource->str(),
                                   memory->str(), implementation->str()};
    if (auto value = object->get("resource_count")) {
      auto count = value->getAsInteger();
      if (!count || *count < 1)
        return fail(error, "invalid placement resource count");
      placement.resourceCount = *count;
    }
    if (auto value = object->get("resource_shape")) {
      auto shape = value->getAsArray();
      if (!shape)
        return fail(error, "placement resource_shape must be an array");
      int64_t product = 1;
      for (const auto &extent : *shape) {
        auto n = extent.getAsInteger();
        if (!n || *n < 1 || product > INT64_MAX / (*n))
          return fail(error, "invalid placement resource shape");
        product *= *n;
        placement.resourceShape.push_back(*n);
      }
      if (!shape->empty() && product != placement.resourceCount)
        return fail(error, "placement resource shape/count mismatch");
    }
    choice.placements.push_back(std::move(placement));
  }
  if (auto value = graph->get("transfer_choices")) {
    auto entries = value->getAsArray();
    if (!entries)
      return fail(error, "program_graph.transfer_choices must be an array");
    std::set<std::tuple<size_t, size_t, std::string>> seen;
    for (const auto &entry : *entries) {
      auto object = entry.getAsObject();
      if (!object)
        return fail(error, "transfer choice must be an object");
      for (const auto &field : *object)
        if (field.first != "storage" && field.first != "consumer_task" &&
            field.first != "target_memory" && field.first != "source_memory" &&
            field.first != "links")
          return fail(error, "unknown transfer choice field");
      auto storage = object->getInteger("storage"),
           consumer = object->getInteger("consumer_task");
      auto source = object->getString("source_memory"),
           dest = object->getString("target_memory");
      auto links = object->getArray("links");
      if (!storage || *storage < 0 || !consumer || *consumer < 0 || !source ||
          source->empty() || !dest || dest->empty() || !links)
        return fail(error, "incomplete transfer choice");
      ProgramTransferChoice selected;
      selected.storage = *storage;
      selected.consumerTask = *consumer;
      selected.sourceMemory = source->str();
      selected.targetMemory = dest->str();
      if (!seen.emplace(selected.storage, selected.consumerTask,
                        selected.targetMemory)
               .second)
        return fail(error, "duplicate transfer choice");
      std::string current = selected.sourceMemory;
      std::set<std::string> visited{current};
      for (const auto &id : *links) {
        auto text = id.getAsString();
        if (!text || text->empty())
          return fail(error, "invalid transfer choice link");
        auto link =
            std::find_if(architecture.links.begin(), architecture.links.end(),
                         [&](const auto &l) { return l.id == *text; });
        if (link == architecture.links.end() || link->source != current ||
            !visited.insert(link->target).second)
          return fail(error, "unknown, disconnected or cyclic selected link");
        current = link->target;
        selected.links.push_back(text->str());
      }
      if (current != selected.targetMemory ||
          std::none_of(
              architecture.memories.begin(), architecture.memories.end(),
              [&](const auto &m) { return m.id == selected.sourceMemory; }))
        return fail(error, "selected transfer path endpoints are invalid");
      choice.transferChoices.push_back(std::move(selected));
    }
  }
  if (auto value = decisions->get("clock_mhz")) {
    auto number = value->getAsNumber();
    if (!number || !std::isfinite(*number) || *number <= 0)
      return fail(error, "clock_mhz must be finite and positive");
    choice.clockMHz = *number;
  }
  auto strings = [&](const Object &object, llvm::StringRef key,
                     std::vector<std::string> &out) {
    if (!object.get(key))
      return true;
    auto array = object.getArray(key);
    if (!array)
      return false;
    for (const auto &item : *array) {
      auto text = item.getAsString();
      if (!text)
        return false;
      out.push_back(text->str());
    }
    return true;
  };
  if (auto value = decisions->get("compute_estimates")) {
    auto estimates = value->getAsArray();
    if (!estimates)
      return fail(error, "compute_estimates must be an array");
    for (const auto &entry : *estimates) {
      auto object = entry.getAsObject();
      if (!object)
        return fail(error, "compute estimate must be an object");
      MappingComputeEstimate estimate;
      auto task = object->getInteger("task");
      auto duration = object->getNumber("duration");
      if (!task || *task < 0 || !duration || !std::isfinite(*duration) ||
          *duration <= 0)
        return fail(error, "compute estimate needs task and positive duration");
      estimate.task = *task;
      estimate.duration = *duration;
      for (auto field :
           {std::make_pair("resource", &estimate.resource),
            std::make_pair("implementation", &estimate.implementation),
            std::make_pair("model", &estimate.model),
            std::make_pair("model_version", &estimate.modelVersion),
            std::make_pair("provenance", &estimate.provenance),
            std::make_pair("boundary", &estimate.boundary)}) {
        auto text = object->getString(field.first);
        if (!text || text->empty())
          return fail(error,
                      "compute estimate provider/boundary field missing");
        *field.second = text->str();
      }
      if (auto value = object->get("unit")) {
        auto text = value->getAsString();
        if (!text)
          return fail(error, "invalid compute unit");
        estimate.unit = text->str();
      }
      if (auto value = object->get("resource_count")) {
        auto count = value->getAsInteger();
        if (!count || *count < 1)
          return fail(error, "invalid compute resource count");
        estimate.resourceCount = *count;
      }
      auto supply = object->getBoolean("supply_conditions_verified");
      if (!supply || !*supply ||
          object->getString("transfer_accounting") !=
              "excludes_inter_task_transfers")
        return fail(error, "compute estimate must verify supply and exclude "
                           "separately charged transfers");
      estimate.supplyConditionsVerified = true;
      estimate.transportScope =
          MappingComputeTransportScope::ExcludesInterTaskTransfers;
      if (!strings(*object, "included_costs", estimate.includedCosts) ||
          !strings(*object, "excluded_costs", estimate.excludedCosts) ||
          !strings(*object, "required_data_supply",
                   estimate.requiredDataSupply))
        return fail(error, "invalid compute accounting conditions");
      double milliseconds;
      if (!convertComputeEstimateToMs(estimate, choice.clockMHz, milliseconds,
                                      error))
        return false;
      choice.computeEstimates.push_back(std::move(estimate));
    }
  }
  return true;
}

bool orbit::taskflow::checkProgramGraphSource(const GenericAnalysis &source,
                                              const ProgramGraph &graph,
                                              std::string &error) {
  if (graph.operatorSource)
    return fail(error, "external operator source has no native MLIR body; use "
                       "its registered implementation cover");
  if (!source.function || !graph.source.function ||
      source.function != graph.source.function ||
      source.bodyFingerprint != graph.bodyFingerprint ||
      !sameJSON(serializeGenericAnalysis(source),
                serializeGenericAnalysis(graph.source)))
    return fail(error, "source body, access facts or logical identities differ "
                       "from the graph");
  for (size_t task = 0; task < source.tasks.size(); ++task)
    if (source.tasks[task].operation != graph.source.tasks[task].operation)
      return fail(error, "task body binding changed");
  for (size_t storage = 0; storage < source.storages.size(); ++storage)
    if (source.storages[storage].root != graph.source.storages[storage].root)
      return fail(error, "logical storage MLIR binding changed");
  GenericAnalysis fresh;
  auto module = source.function->getParentOfType<mlir::ModuleOp>();
  if (!module || !analyzeGenericProgram(module, fresh, error) ||
      !sameJSON(serializeGenericAnalysis(fresh),
                serializeGenericAnalysis(source)))
    return fail(error, "actual source MLIR changed after graph construction");
  return true;
}

bool orbit::taskflow::finalizeProgramGraph(const GenericAnalysis &source,
                                           ProgramGraph &graph,
                                           const GenericMappingPlan *native,
                                           std::string &error) {
  graph.source = source;
  graph.bodyFingerprint = source.bodyFingerprint;
  graph.allocationLifetime = native ? "phase-pool" : "copy-events";
  if (native) {
    graph.inputLocations.clear();
    graph.outputLocations.clear();
    for (size_t storage = 0; storage < source.storages.size(); ++storage) {
      if (source.storages[storage].role == "input")
        graph.inputLocations[storage] = native->memoryBinding.hostMemory;
      if (source.storages[storage].role == "output")
        graph.outputLocations[storage] = native->memoryBinding.hostMemory;
    }
  }
  if (native) {
    graph.executionGroups = native->executionGroups;
    for (auto &group : graph.executionGroups) {
      group.memberEvents.clear();
      for (size_t e = 0; e < graph.events.size(); ++e)
        if (graph.events[e].phase >= group.beginPhase &&
            graph.events[e].phase < group.endPhase)
          group.memberEvents.push_back(e);
    }
    if (graph.regionEvents.empty()) {
      for (size_t r = 0; r < graph.executionRegions.size(); ++r) {
        auto &call = graph.executionRegions[r];
        call.sourceTasks.clear();
        call.memberEvents.clear();
        call.implementation = "direct-air";
        call.endpointMemory = native->memoryBinding.hostMemory;
        call.invocationLifecycle = "host-device";
        for (size_t t = 0; t < source.tasks.size(); ++t)
          call.sourceTasks.push_back(t);
        for (size_t e = call.beginEvent; e < call.endEvent; ++e)
          call.memberEvents.push_back(e);
        std::vector<int64_t> latest(graph.regions.size(), -1);
        for (size_t v = 0; v < graph.versions.size(); ++v) {
          auto cell = graph.versions[v].region;
          if (latest[cell] < 0 ||
              graph.versions[v].producer >
                  graph.versions[size_t(latest[cell])].producer)
            latest[cell] = v;
        }
        for (size_t c = 0; c < graph.copies.size(); ++c) {
          const auto &copy = graph.copies[c];
          const auto &version = graph.versions[copy.version];
          auto storage = graph.regions[version.region].storage;
          if (graph.allocations[copy.allocation].memory != call.endpointMemory)
            continue;
          if (copy.createdBy < 0 && source.storages[storage].role == "input")
            call.inputCopies.push_back(c);
          if (source.storages[storage].role == "output" &&
              latest[version.region] == int64_t(copy.version) &&
              copy.releasedAfter == graph.events.size())
            call.outputCopies.push_back(c);
        }
        for (const auto &kind :
             {"visible-to-device", "submit", "complete", "visible-to-host"}) {
          MappingRegionEvent e;
          e.kind = kind;
          e.region = r;
          e.resource = call.submissionResource;
          if (!graph.regionEvents.empty())
            e.dependencies.push_back(graph.regionEvents.size() - 1);
          if (e.kind == "visible-to-device")
            e.copies = call.inputCopies;
          if (e.kind == "visible-to-host")
            e.copies = call.outputCopies;
          graph.regionEvents.push_back(std::move(e));
        }
      }
    }
  }
  graph.instances.clear();
  for (size_t index = 0; index < graph.events.size(); ++index) {
    const auto &event = graph.events[index];
    if (event.kind == "transfer") {
      if (!event.transferRegion ||
          regionBytes(source, *event.transferRegion) != event.transferBytes)
        return fail(error, "builder did not supply an exact source transfer "
                           "rectangle and byte count");
      continue;
    }
    if (event.task >= source.tasks.size())
      return fail(error, "invalid source instance task");
    MappingTaskInstance instance;
    instance.task = event.task;
    instance.event = index;
    instance.phase = event.phase;
    instance.domainTask = event.task;
    instance.kind =
        event.kind == "initialize" ? "source-initializer" : "source-task";
    if (native) {
      auto implementation = std::find_if(
          native->taskImplementations.begin(),
          native->taskImplementations.end(),
          [&](const auto &value) { return value.task == event.task; });
      if (implementation == native->taskImplementations.end())
        return fail(error, "native task has no implementation binding");
      instance.implementation = implementation->kind;
      instance.lanes = implementation->lanes;
      instance.parallelLoop = implementation->parallelLoop;
      if (native->strategy == "serial-tiled") {
        if (event.phase >= native->phases.size())
          return fail(error, "instance refers to an absent source phase");
        const auto &phase = native->phases[event.phase];
        instance.domainTask = phase.task;
        for (const auto &loop : phase.loops) {
          if (loop.loop >= source.tasks[phase.task].loops.size())
            return fail(error, "instance source loop is absent");
          instance.domain.push_back(
              {loop.loop, loop.lower, loop.upper,
               source.tasks[phase.task].loops[loop.loop].step});
        }
      }
    } else {
      auto placement = std::find_if(
          graph.placements.begin(), graph.placements.end(),
          [&](const auto &value) { return value.task == event.task; });
      if (placement == graph.placements.end())
        return fail(error, "whole-task instance has no placement");
      instance.implementation = placement->implementation;
      instance.resourceCount = placement->resourceCount;
      instance.resourceShape = placement->resourceShape;
    }
    if (!native || native->strategy != "serial-tiled")
      for (size_t loop = 0; loop < source.tasks[event.task].loops.size();
           ++loop) {
        const auto &range = source.tasks[event.task].loops[loop];
        instance.domain.push_back({loop, range.lower, range.upper, range.step});
      }
    graph.instances.push_back(std::move(instance));
  }
  if (!native && graph.regionEvents.empty()) {
    MappingExecutionRegion host;
    host.id = "host-program";
    host.placement = "host";
    host.endEvent = graph.events.size();
    host.submissionCount = 0;
    graph.executionRegions = {host};
  }
  return schedule(graph, graph.schedule, error);
}

bool orbit::taskflow::refreshProgramGraphSchedule(ProgramGraph &graph,
                                                  std::string &error) {
  if (graph.operatorSource)
    return fail(error, "external implementation timing cannot be redistributed "
                       "to native source events");
  MappingGraphSchedule refreshed;
  if (!schedule(graph, refreshed, error))
    return false;
  graph.schedule = std::move(refreshed);
  return true;
}

bool orbit::taskflow::programAllocationIsSlot(const ProgramGraph &g, size_t a) {
  if (a >= g.allocations.size())
    return false;
  const auto &v = g.allocations[a];
  if (v.addressing == "reusable-slot")
    return true;
  if (v.addressing == "region-addressed")
    return false;
  auto m = std::find_if(g.memories.begin(), g.memories.end(),
                        [&](const auto &x) { return x.id == v.memory; });
  return g.allocationLifetime == "phase-pool" && m != g.memories.end() &&
         m->kind != "system-bo";
}
bool orbit::taskflow::collectProgramBufferOrdering(
    const ProgramGraph &g, std::vector<ProgramBufferOrdering> &out,
    std::string &error) {
  out.clear();
  for (size_t a = 0; a < g.copies.size(); ++a)
    for (size_t b = a + 1; b < g.copies.size(); ++b) {
      const auto &x = g.copies[a], &y = g.copies[b];
      if (x.allocation >= g.allocations.size() ||
          y.allocation >= g.allocations.size() ||
          x.version >= g.versions.size() || y.version >= g.versions.size())
        return fail(error, "buffer ordering references invalid copy");
      if (x.allocation != y.allocation || x.createdBy == y.createdBy)
        continue;
      if (!programAllocationIsSlot(g, x.allocation) &&
          g.versions[x.version].region != g.versions[y.version].region)
        continue;
      size_t earlier = a, later = b;
      if (x.createdBy > y.createdBy)
        std::swap(earlier, later);
      const auto &old = g.copies[earlier], &next = g.copies[later];
      if (next.createdBy < 0)
        continue;
      std::set<size_t> parents(old.users.begin(), old.users.end());
      if (old.createdBy >= 0)
        parents.insert(size_t(old.createdBy));
      // releasedAfter is an exclusive ordinal validity witness, not a timed
      // release activity. An unrelated event before an overwrite need not
      // finish. Backend release activities must supply their actual dependency
      // explicitly.
      for (auto u : parents) {
        if (u >= g.events.size() || size_t(next.createdBy) >= g.events.size())
          return fail(error, "buffer release outside graph");
        if (u == size_t(next.createdBy))
          continue; // atomic read-old/write-new
        out.push_back({x.allocation, earlier, later, u, size_t(next.createdBy),
                       "physical allocation overwrite waits for previous "
                       "creation and all readers"});
      }
    }
  return true;
}

bool orbit::taskflow::checkProgramGraph(const ProgramGraph &graph,
                                        std::string &error) {
  if (graph.operatorSource)
    return checkProgramOperatorGraph(graph, error);
  const auto &source = graph.source;
  if (!checkProgramGraphSource(source, graph, error))
    return false;
  StaticConfig target;
  target.domains = graph.domains;
  target.memories = graph.memories;
  target.resources = graph.resources;
  target.links = graph.links;
  if (!validateMemoryArchitecture(target, error))
    return false;
  if (graph.clockMHz &&
      (!std::isfinite(*graph.clockMHz) || *graph.clockMHz <= 0))
    return fail(error, "invalid target compute clock");
  if (graph.events.empty() || graph.regions.empty() || graph.copies.empty())
    return fail(error, "empty program execution graph");
  if (graph.allocationLifetime != "phase-pool" &&
      graph.allocationLifetime != "copy-events")
    return fail(error, "unsupported allocation lifetime semantics");
  std::set<size_t> inputStorages, outputStorages;
  for (size_t storage = 0; storage < source.storages.size(); ++storage) {
    if (source.storages[storage].role == "input")
      inputStorages.insert(storage);
    if (source.storages[storage].role == "output")
      outputStorages.insert(storage);
  }
  auto boundariesValid = [&](const auto &locations, const auto &expected) {
    if (locations.size() != expected.size())
      return false;
    for (const auto &entry : locations)
      if (!expected.count(entry.first) ||
          std::none_of(
              graph.memories.begin(), graph.memories.end(),
              [&](const auto &memory) { return memory.id == entry.second; }))
        return false;
    return true;
  };
  if (!boundariesValid(graph.inputLocations, inputStorages) ||
      !boundariesValid(graph.outputLocations, outputStorages))
    return fail(error, "missing/fabricated external data boundary");
  std::vector<std::vector<size_t>> cells(source.storages.size());
  std::vector<std::vector<std::vector<int64_t>>> gridCuts(
      source.storages.size());
  std::vector<std::map<std::vector<int64_t>, size_t>> cellAt(
      source.storages.size());
  for (size_t index = 0; index < graph.regions.size(); ++index) {
    if (regionBytes(source, graph.regions[index]) <= 0)
      return fail(error, "logical region outside source storage");
    cells[graph.regions[index].storage].push_back(index);
    cellAt[graph.regions[index].storage][graph.regions[index].offsets] = index;
  }
  // Current region identity is an exact Cartesian partition, not arbitrary
  // overlapping bounding boxes. Reconstruct the partition independently.
  for (size_t storage = 0; storage < source.storages.size(); ++storage) {
    const auto &shape = source.storages[storage].shape;
    std::vector<std::set<int64_t>> cuts(shape.size());
    for (size_t axis = 0; axis < shape.size(); ++axis)
      cuts[axis] = {0, shape[axis]};
    std::set<std::vector<int64_t>> offsets;
    for (size_t cell : cells[storage]) {
      const auto &region = graph.regions[cell];
      if (!offsets.insert(region.offsets).second)
        return fail(error, "duplicate logical partition cell");
      for (size_t axis = 0; axis < shape.size(); ++axis) {
        cuts[axis].insert(region.offsets[axis]);
        cuts[axis].insert(region.offsets[axis] + region.sizes[axis]);
      }
    }
    size_t count = 1;
    for (const auto &axis : cuts) {
      if (count > 1048576 / (axis.size() - 1))
        return fail(error, "logical partition exceeds supported finite size");
      count *= axis.size() - 1;
    }
    if (count != cells[storage].size())
      return fail(error, "omitted or overlapping logical partition cells");
    for (size_t cell : cells[storage])
      for (size_t axis = 0; axis < shape.size(); ++axis) {
        const auto &region = graph.regions[cell];
        auto begin = cuts[axis].find(region.offsets[axis]);
        if (++begin == cuts[axis].end() ||
            *begin != region.offsets[axis] + region.sizes[axis])
          return fail(error, "logical cell overlaps another region boundary");
      }
    for (const auto &axis : cuts)
      gridCuts[storage].push_back(
          std::vector<int64_t>(axis.begin(), axis.end()));
  }
  auto collectCells = [&](const MappingRegion &region, std::set<size_t> &out) {
    if (regionBytes(source, region) <= 0)
      return false;
    std::vector<int64_t> offset(region.sizes.size());
    bool valid = true;
    std::function<void(size_t)> visit = [&](size_t axis) {
      if (axis == region.sizes.size()) {
        auto found = cellAt[region.storage].find(offset);
        if (found == cellAt[region.storage].end())
          valid = false;
        else
          out.insert(found->second);
        return;
      }
      const auto &cuts = gridCuts[region.storage][axis];
      auto begin =
          std::lower_bound(cuts.begin(), cuts.end(), region.offsets[axis]);
      auto end = std::lower_bound(cuts.begin(), cuts.end(),
                                  region.offsets[axis] + region.sizes[axis]);
      if (begin == cuts.end() || end == cuts.end() ||
          *begin != region.offsets[axis] ||
          *end != region.offsets[axis] + region.sizes[axis]) {
        valid = false;
        return;
      }
      for (auto point = begin; point != end; ++point) {
        offset[axis] = *point;
        visit(axis + 1);
      }
    };
    visit(0);
    return valid;
  };
  for (const auto &allocation : graph.allocations) {
    auto memory = std::find_if(
        graph.memories.begin(), graph.memories.end(),
        [&](const auto &value) { return value.id == allocation.memory; });
    if (allocation.storage >= source.storages.size() || allocation.bytes <= 0 ||
        allocation.beginPhase >= allocation.endPhase ||
        memory == graph.memories.end() ||
        allocation.bytes % memory->alignmentBytes)
      return fail(error, "invalid allocation storage, memory or lifetime");
    if (!allocation.spatialOwner.empty() &&
        (!programOwnerExists(target, allocation.spatialOwner) ||
         allocation.spatialOwner != memory->domain))
      return fail(error,
                  "allocation owner is not its target memory instance owner");
    if (!allocation.addressing.empty() &&
        allocation.addressing != "region-addressed" &&
        allocation.addressing != "reusable-slot")
      return fail(error, "unknown allocation addressing contract");
    if (allocation.addressing == "region-addressed") {
      MappingRegion full{
          allocation.storage,
          std::vector<int64_t>(source.storages[allocation.storage].shape.size(),
                               0),
          source.storages[allocation.storage].shape};
      if (allocation.bytes < regionBytes(source, full))
        return fail(
            error,
            "region-addressed allocation cannot contain full logical offsets");
    }
  }
  if (!graph.executionGroups.empty()) {
    std::vector<bool> members(graph.events.size());
    for (size_t i = 0; i < graph.executionGroups.size(); ++i) {
      const auto &g = graph.executionGroups[i];
      auto memory =
          std::find_if(graph.memories.begin(), graph.memories.end(),
                       [&](const auto &m) { return m.id == g.memory; });
      if (g.beginPhase >= g.endPhase || !resource(graph, g.resource) ||
          memory == graph.memories.end() || g.memberEvents.empty())
        return fail(error, "invalid public execution group binding");
      auto compute = resource(graph, g.resource);
      if (compute->kind != "compute" ||
          !memoryAccessible(target, g.memory, g.resource))
        return fail(error, "group compute cannot access execution memory");
      if (!g.spatialOwner.empty() &&
          (!programOwnerExists(target, g.spatialOwner) ||
           g.spatialOwner != compute->domain))
        return fail(error,
                    "group owner is not its target compute instance owner");
      if (!compute->domain.empty() && g.spatialOwner.empty())
        return fail(error, "group omits target compute instance owner");
      for (const auto *parents :
           {&g.dependencies, &g.implementationDependencies}) {
        if (!std::is_sorted(parents->begin(), parents->end()) ||
            std::adjacent_find(parents->begin(), parents->end()) !=
                parents->end())
          return fail(error, "duplicate group dependencies");
        for (auto parent : *parents)
          if (parent >= i)
            return fail(error, "cyclic public group dependencies");
      }
      for (auto e : g.memberEvents) {
        if (e >= graph.events.size() || members[e] ||
            graph.events[e].phase < g.beginPhase ||
            graph.events[e].phase >= g.endPhase)
          return fail(error,
                      "group event coverage overlaps or escapes phase domain");
        members[e] = true;
        const auto &event = graph.events[e];
        if (event.kind != "transfer" &&
            (event.resource != g.resource || event.memory != g.memory))
          return fail(error,
                      "group compute placement differs from public instance");
        if (!g.spatialOwner.empty())
          for (const auto *copies : {&event.reads, &event.writes})
            for (auto c : *copies)
              if (c < graph.copies.size()) {
                if (graph.copies[c].allocation >= graph.allocations.size())
                  return fail(error, "group copy allocation outside graph");
                const auto &a = graph.allocations[graph.copies[c].allocation];
                auto location = programMemory(target, a.memory);
                if (!location)
                  return fail(error, "group copy has unknown memory");
                if (!a.spatialOwner.empty() &&
                    a.spatialOwner != location->domain)
                  return fail(error,
                              "group accesses copy with invalid memory owner");
              }
      }
    }
    if (std::find(members.begin(), members.end(), false) != members.end())
      return fail(error, "public execution groups omit events");
  }
  std::vector<std::vector<size_t>> users(graph.copies.size()),
      edges(graph.copies.size());
  std::vector<int64_t> latest(graph.regions.size(), -1);
  std::vector<bool> producedVersions(graph.versions.size()),
      createdCopies(graph.copies.size());
  std::vector<std::set<size_t>> physicalCopies(graph.allocations.size());
  for (size_t version = 0; version < graph.versions.size(); ++version) {
    const auto &value = graph.versions[version];
    if (value.region >= graph.regions.size() || value.producer < -1 ||
        value.producer >= int64_t(graph.events.size()))
      return fail(error, "invalid logical version producer");
    if (value.producer < 0) {
      if (source.storages[graph.regions[value.region].storage].role !=
              "input" ||
          latest[value.region] >= 0)
        return fail(error, "fabricated initial version");
      latest[value.region] = version;
      producedVersions[version] = true;
    }
  }
  for (size_t copy = 0; copy < graph.copies.size(); ++copy) {
    const auto &value = graph.copies[copy];
    if (value.version >= graph.versions.size() ||
        value.allocation >= graph.allocations.size() || value.createdBy < -1 ||
        value.createdBy >= int64_t(graph.events.size()) ||
        value.availableFrom >= value.releasedAfter ||
        value.releasedAfter > graph.events.size())
      return fail(error, "invalid copy identity or lifetime");
    const auto &region = graph.regions[graph.versions[value.version].region];
    if (graph.allocations[value.allocation].storage != region.storage ||
        regionBytes(source, region) >
            graph.allocations[value.allocation].bytes ||
        value.layout != "row_major")
      return fail(error,
                  "copy allocation does not cover its logical data/layout");
    if (value.createdBy < 0) {
      if (graph.versions[value.version].producer != -1 ||
          value.availableFrom != 0 ||
          !graph.inputLocations.count(region.storage) ||
          graph.inputLocations.at(region.storage) !=
              graph.allocations[value.allocation].memory)
        return fail(
            error,
            "uncreated copy of a computed version or wrong initial boundary");
      createdCopies[copy] = true;
      physicalCopies[value.allocation].insert(copy);
    }
  }
  size_t instanceIndex = 0;
  int64_t traffic = 0;
  for (size_t index = 0; index < graph.events.size(); ++index) {
    const auto &event = graph.events[index];
    if (event.task >= source.tasks.size() || !resource(graph, event.resource))
      return fail(error, "event lacks source or resource binding");
    if (!std::is_sorted(event.dependencies.begin(), event.dependencies.end()) ||
        std::adjacent_find(event.dependencies.begin(),
                           event.dependencies.end()) !=
            event.dependencies.end() ||
        std::any_of(event.dependencies.begin(), event.dependencies.end(),
                    [&](size_t parent) { return parent >= index; }))
      return fail(error, "invalid event dependency order");
    auto linked = [&](int64_t creator) {
      return creator < 0 ||
             std::binary_search(event.dependencies.begin(),
                                event.dependencies.end(), size_t(creator));
    };
    std::set<size_t> readSet, writeSet;
    for (size_t copy : event.reads) {
      if (copy >= graph.copies.size() || !createdCopies[copy] ||
          !readSet.insert(copy).second)
        return fail(error, "missing, duplicate or uncreated read copy");
      const auto &value = graph.copies[copy];
      const auto &allocation = graph.allocations[value.allocation];
      if (value.availableFrom > index || value.releasedAfter <= index ||
          !linked(value.createdBy) ||
          !physicalCopies[value.allocation].count(copy) ||
          latest[graph.versions[value.version].region] !=
              int64_t(value.version))
        return fail(
            error,
            "stale/released copy or missing copy creation dependency at "
            "event " +
                std::to_string(index) + " copy " + std::to_string(copy) +
                " creator " + std::to_string(value.createdBy) + " release " +
                std::to_string(value.releasedAfter) + " version " +
                std::to_string(value.version) + " latest " +
                std::to_string(latest[graph.versions[value.version].region]));
      if (event.kind != "transfer" &&
          !memoryAccessible(target, allocation.memory, event.resource))
        return fail(error, "compute cannot access its required copy");
      users[copy].push_back(index);
    }
    std::set<size_t> overwritten;
    for (size_t copy : event.writes) {
      if (copy >= graph.copies.size() || createdCopies[copy] ||
          !writeSet.insert(copy).second)
        return fail(error, "duplicate or missing copy creation");
      const auto &value = graph.copies[copy];
      const auto &version = graph.versions[value.version];
      const auto &allocation = graph.allocations[value.allocation];
      const bool pool = programAllocationIsSlot(graph, value.allocation);
      if (pool && overwritten.insert(value.allocation).second) {
        for (size_t old : physicalCopies[value.allocation]) {
          const auto &previous = graph.copies[old];
          if (previous.releasedAfter > index + 1 ||
              (!previous.users.empty() && previous.users.back() > index))
            return fail(error,
                        "physical pool reused before all readers complete");
        }
        physicalCopies[value.allocation].clear();
      } else if (!pool) {
        for (auto old = physicalCopies[value.allocation].begin();
             old != physicalCopies[value.allocation].end();) {
          if (graph.versions[graph.copies[*old].version].region ==
              version.region)
            old = physicalCopies[value.allocation].erase(old);
          else
            ++old;
        }
      }
      if (value.createdBy != int64_t(index) || value.availableFrom != index)
        return fail(error, "copy creator/availability mismatch");
      if (event.kind == "transfer") {
        if (latest[version.region] != int64_t(value.version))
          return fail(error, "transfer changes logical version");
      } else {
        if (version.producer != int64_t(index) ||
            producedVersions[value.version])
          return fail(error, "compute does not create its declared version");
        producedVersions[value.version] = true;
        latest[version.region] = value.version;
        if (!memoryAccessible(target,
                              graph.allocations[value.allocation].memory,
                              event.resource))
          return fail(error, "compute cannot write chosen storage");
      }
      createdCopies[copy] = true;
      physicalCopies[value.allocation].insert(copy);
    }
    if (event.kind == "transfer") {
      if (!event.transferRegion ||
          regionBytes(source, *event.transferRegion) != event.transferBytes ||
          event.reads.empty() || event.reads.size() != event.writes.size() ||
          !event.dataEdges.empty() || !event.orderEdges.empty())
        return fail(error,
                    "communication rectangle or copy correspondence mismatch");
      auto link = std::find_if(
          graph.links.begin(), graph.links.end(),
          [&](const auto &value) { return value.id == event.link; });
      if (link == graph.links.end() || link->resource != event.resource ||
          event.transferBytes < link->minBytes ||
          event.transferBytes > link->maxBytes || link->temporaryBytes)
        return fail(error,
                    "communication path/resource/scratch is unsupported");
      int64_t bytes = 0;
      std::set<size_t> versions;
      for (size_t pair = 0; pair < event.reads.size(); ++pair) {
        const auto &from = graph.copies[event.reads[pair]],
                   &to = graph.copies[event.writes[pair]];
        const auto &cell = graph.regions[graph.versions[from.version].region];
        if (from.version != to.version ||
            !versions.insert(from.version).second ||
            !inside(cell, *event.transferRegion) ||
            graph.allocations[from.allocation].memory != link->source ||
            graph.allocations[to.allocation].memory != link->target ||
            !add(bytes, regionBytes(source, cell)))
          return fail(
              error,
              "communication endpoints, region or version differ from copies");
      }
      for (size_t pair = 0; pair < event.reads.size(); ++pair) {
        const auto &from = graph.copies[event.reads[pair]],
                   &to = graph.copies[event.writes[pair]];
        const auto &dtype =
            source
                .storages[graph.regions[graph.versions[from.version].region]
                              .storage]
                .dtype;
        if ((!link->sourceLayout.empty() &&
             link->sourceLayout != from.layout) ||
            (!link->targetLayout.empty() && link->targetLayout != to.layout) ||
            (!link->sourceDtype.empty() && link->sourceDtype != dtype) ||
            (!link->targetDtype.empty() && link->targetDtype != dtype))
          return fail(
              error,
              "transfer copy violates target link layout/dtype constraint");
      }
      if (bytes != event.transferBytes || !add(traffic, bytes))
        return fail(error,
                    "communication omits/duplicates part of its rectangle");
    } else {
      if (event.kind != "compute" && event.kind != "initialize")
        return fail(error, "unsupported execution event");
      if (event.transferRegion || event.transferBytes || !event.link.empty() ||
          instanceIndex >= graph.instances.size())
        return fail(error,
                    "compute has transport metadata or missing instance");
      const auto &instance = graph.instances[instanceIndex++];
      if (instance.task != event.task || instance.event != index ||
          instance.phase != event.phase ||
          instance.domainTask >= source.tasks.size() || instance.lanes <= 0 ||
          instance.implementation.empty() ||
          instance.kind != (event.kind == "initialize" ? "source-initializer"
                                                       : "source-task"))
        return fail(error, "source task instance binding mismatch");
      if (instance.kind == "source-task" &&
          instance.domainTask != instance.task)
        return fail(error, "source instance domain belongs to another task");
      auto instanceResource = resource(graph, event.resource);
      if (!instanceResource || instance.resourceCount < 1 ||
          instance.resourceCount > instanceResource->capacity)
        return fail(error,
                    "task instance resource demand exceeds target capacity");
      auto placement = std::find_if(
          graph.placements.begin(), graph.placements.end(),
          [&](const auto &value) { return value.task == instance.task; });
      if (placement != graph.placements.end() &&
          (placement->resource != event.resource ||
           placement->memory != event.memory ||
           placement->implementation != instance.implementation ||
           placement->resourceCount != instance.resourceCount ||
           placement->resourceShape != instance.resourceShape))
        return fail(error,
                    "task instance differs from declared resource placement");
      if (!instance.resourceShape.empty()) {
        int64_t count = 1;
        for (auto extent : instance.resourceShape) {
          if (extent <= 0 || count > INT64_MAX / extent)
            return fail(error, "invalid resource shape");
          count *= extent;
        }
        if (count != instance.resourceCount)
          return fail(error, "resource shape and count disagree");
      }
      std::set<size_t> loops;
      for (const auto &range : instance.domain) {
        if (range.loop >= source.tasks[instance.domainTask].loops.size() ||
            !loops.insert(range.loop).second)
          return fail(error, "invalid instance loop identity");
        const auto &loop = source.tasks[instance.domainTask].loops[range.loop];
        if (range.lower < loop.lower || range.upper > loop.upper ||
            range.upper <= range.lower || range.step != loop.step ||
            (range.lower - loop.lower) % loop.step)
          return fail(error, "instance domain escapes actual source loop");
      }
      if (loops.size() != source.tasks[instance.domainTask].loops.size())
        return fail(error, "missing source instance loop domain");
      if (event.dataEdges.size() != event.reads.size())
        return fail(error,
                    "data dependence incidence does not cover compute reads");
      // Derive access rectangles from the actual source body at the selected
      // instance domain. A graph cannot omit a source read/write while keeping
      // its internally consistent copy bookkeeping. Distributed initialization
      // writes the controlling tile's output regions using its proved body.
      GenericTilePhase accessPhase;
      accessPhase.task = instance.domainTask;
      for (const auto &range : instance.domain)
        accessPhase.loops.push_back({range.loop, range.lower, range.upper});
      int64_t payload = 0;
      if (!deriveGenericPhaseRegions(source, accessPhase, payload, error))
        return false;
      std::set<size_t> expectedReads, expectedWrites, actualReads, actualWrites;
      std::set<size_t> initializerWrites;
      if (event.kind == "initialize")
        for (const auto &access : source.tasks[event.task].accesses) {
          if (access.read)
            return fail(
                error,
                "distributed initializer with source reads is unsupported");
          if (access.write)
            initializerWrites.insert(access.storage);
        }
      for (const auto &access : accessPhase.regions) {
        MappingRegion region{access.storage, access.offsets, access.sizes};
        if (access.read && event.kind != "initialize" &&
            !collectCells(region, expectedReads))
          return fail(error,
                      "source read region has no complete logical partition");
        if (access.write &&
            (event.kind != "initialize" ||
             initializerWrites.count(access.storage)) &&
            !collectCells(region, expectedWrites))
          return fail(error,
                      "source write region has no complete logical partition");
      }
      for (size_t copy : event.reads)
        actualReads.insert(graph.versions[graph.copies[copy].version].region);
      for (size_t copy : event.writes)
        actualWrites.insert(graph.versions[graph.copies[copy].version].region);
      if (actualReads != expectedReads || actualWrites != expectedWrites)
        return fail(error, "compute copy regions omit/fabricate actual source "
                           "instance accesses");
    }
  }
  if (instanceIndex != graph.instances.size() ||
      traffic != graph.transferBytes ||
      std::find(createdCopies.begin(), createdCopies.end(), false) !=
          createdCopies.end() ||
      std::find(producedVersions.begin(), producedVersions.end(), false) !=
          producedVersions.end())
    return fail(error,
                "unrealized instance/version/copy or incorrect traffic total");
  if (!checkSourceInstanceCoverage(graph, error))
    return false;
  std::set<size_t> representedTasks;
  for (const auto &instance : graph.instances)
    representedTasks.insert(instance.task);
  if (representedTasks.size() != source.tasks.size())
    return fail(error, "source task has no execution instance");
  for (const auto &boundary : graph.outputLocations)
    for (size_t cell : cells[boundary.first]) {
      bool available = false;
      for (size_t allocation = 0; allocation < graph.allocations.size();
           ++allocation)
        if (graph.allocations[allocation].memory == boundary.second)
          for (size_t copy : physicalCopies[allocation]) {
            const auto &value = graph.copies[copy];
            available |= graph.versions[value.version].region == cell &&
                         latest[cell] == int64_t(value.version) &&
                         value.releasedAfter == graph.events.size();
          }
      if (!available)
        return fail(
            error,
            "latest final output is not available at the declared boundary");
    }
  for (size_t event = 0; event < graph.events.size(); ++event) {
    const auto &value = graph.events[event];
    for (size_t edge : value.dataEdges)
      if (edge >= graph.dataEdges.size() ||
          graph.dataEdges[edge].consumer != event)
        return fail(error, "invalid data-edge incidence");
    for (size_t edge : value.orderEdges)
      if (edge >= graph.orderEdges.size() ||
          graph.orderEdges[edge].consumer != event)
        return fail(error, "invalid order-edge incidence");
    if (value.kind != "transfer")
      for (size_t copy : value.reads) {
        size_t count = std::count_if(
            value.dataEdges.begin(), value.dataEdges.end(),
            [&](size_t edge) { return graph.dataEdges[edge].copy == copy; });
        if (count != 1)
          return fail(error, "compute read lacks a unique data requirement");
      }
  }
  for (size_t edge = 0; edge < graph.dataEdges.size(); ++edge) {
    const auto &value = graph.dataEdges[edge];
    if (value.copy >= graph.copies.size() ||
        value.consumer >= graph.events.size())
      return fail(error, "invalid data edge endpoints");
    const auto &copy = graph.copies[value.copy];
    const auto &version = graph.versions[copy.version];
    const auto &region = graph.regions[version.region];
    const auto &event = graph.events[value.consumer];
    if (value.version != copy.version || value.region != version.region ||
        value.producer != version.producer ||
        std::count(event.dataEdges.begin(), event.dataEdges.end(), edge) != 1 ||
        std::count(event.reads.begin(), event.reads.end(), value.copy) != 1 ||
        value.requirement.memory != graph.allocations[copy.allocation].memory ||
        value.requirement.layout != copy.layout ||
        value.requirement.dtype != source.storages[region.storage].dtype)
      return fail(error, "data edge reads a wrong region, version or copy");
    bool sourceRead = false;
    for (const auto &access : source.tasks[event.task].accesses)
      if (access.read && access.storage == region.storage) {
        // A loop fission can split one required rectangle across several source
        // accesses. Complete union coverage was checked at the instance above;
        // no single access is required to contain the whole partition cell.
        sourceRead = true;
      }
    if (!sourceRead)
      return fail(error,
                  "data requirement has no corresponding source read access");
    edges[value.copy].push_back(edge);
  }
  for (size_t edge = 0; edge < graph.orderEdges.size(); ++edge) {
    const auto &value = graph.orderEdges[edge];
    if (value.producer >= graph.events.size() ||
        value.consumer >= graph.events.size() ||
        !precedes(graph, value.producer, value.consumer) ||
        std::count(graph.events[value.consumer].orderEdges.begin(),
                   graph.events[value.consumer].orderEdges.end(), edge) != 1)
      return fail(error, "pure order edge lacks a synchronization path");
    const auto &dependencies =
        source.tasks[graph.events[value.consumer].task].dependencies;
    if (std::find(dependencies.begin(), dependencies.end(),
                  graph.events[value.producer].task) == dependencies.end())
      return fail(error, "ordering edge is not a source task dependence");
  }
  std::set<std::pair<size_t, size_t>> valueDependencies;
  std::map<size_t, size_t> firstTaskEvent, lastTaskEvent;
  for (size_t event = 0; event < graph.events.size(); ++event)
    if (graph.events[event].kind != "transfer") {
      firstTaskEvent.emplace(graph.events[event].task, event);
      lastTaskEvent[graph.events[event].task] = event;
    }
  for (const auto &edge : graph.dataEdges)
    if (edge.producer >= 0)
      valueDependencies.insert(
          {graph.events[edge.producer].task, graph.events[edge.consumer].task});
  for (size_t task = 0; task < source.tasks.size(); ++task)
    for (size_t parent : source.tasks[task].dependencies) {
      if (!firstTaskEvent.count(task) || !lastTaskEvent.count(parent))
        continue;
      if (!valueDependencies.count({parent, task})) {
        size_t found =
            std::count_if(graph.orderEdges.begin(), graph.orderEdges.end(),
                          [&](const auto &edge) {
                            return edge.producer == lastTaskEvent[parent] &&
                                   edge.consumer == firstTaskEvent[task];
                          });
        if (found != 1)
          return fail(error,
                      "missing or duplicate pure source ordering requirement");
      }
    }
  for (size_t copy = 0; copy < graph.copies.size(); ++copy) {
    const auto &value = graph.copies[copy];
    std::vector<size_t> last;
    if (!users[copy].empty())
      last.push_back(users[copy].back());
    if (value.users != users[copy] || value.lastUsers != last ||
        value.consumerEdges != edges[copy])
      return fail(error, "copy users, final user or data incidence mismatch");
  }
  std::map<std::string, std::map<size_t, int64_t>> deltas;
  if (graph.allocationLifetime == "phase-pool") {
    for (const auto &allocation : graph.allocations) {
      if (!add(deltas[allocation.memory][allocation.beginPhase],
               allocation.bytes))
        return fail(error, "capacity sum overflows");
      deltas[allocation.memory][allocation.endPhase] -= allocation.bytes;
    }
  } else {
    // Distinct copies can reference one allocation; count its union lifetime.
    for (size_t allocation = 0; allocation < graph.allocations.size();
         ++allocation) {
      std::vector<std::pair<size_t, size_t>> ranges;
      for (const auto &copy : graph.copies)
        if (copy.allocation == allocation)
          ranges.push_back({copy.availableFrom, copy.releasedAfter});
      std::sort(ranges.begin(), ranges.end());
      for (size_t index = 0; index < ranges.size();) {
        size_t begin = ranges[index].first, end = ranges[index++].second;
        while (index < ranges.size() && ranges[index].first <= end)
          end = std::max(end, ranges[index++].second);
        const auto &value = graph.allocations[allocation];
        if (!add(deltas[value.memory][begin], value.bytes))
          return fail(error, "capacity sum overflows");
        deltas[value.memory][end] -= value.bytes;
      }
    }
  }
  for (const auto &memory : graph.memories) {
    int64_t live = 0;
    for (const auto &change : deltas[memory.id]) {
      live += change.second;
      if (live < 0 || live > memory.capacityBytes)
        return fail(error, "shared allocation capacity exceeded");
    }
  }
  std::vector<bool> covered(graph.events.size());
  std::vector<size_t> owners(graph.events.size(), SIZE_MAX),
      taskOwners(source.tasks.size(), SIZE_MAX);
  bool refined = !graph.regionEvents.empty();
  for (size_t r = 0; r < graph.executionRegions.size(); ++r) {
    const auto &region = graph.executionRegions[r];
    if (!region.invocationLifecycle.empty() &&
        region.invocationLifecycle != "host-only" &&
        region.invocationLifecycle != "host-device")
      return fail(error, "unsupported invocation lifecycle policy");
    if (region.id.empty() || region.beginEvent >= region.endEvent ||
        region.endEvent > graph.events.size() || region.submissionCount < 0 ||
        (region.submissionCount &&
         (!resource(graph, region.submissionResource) ||
          resource(graph, region.submissionResource)->kind != "submission")))
      return fail(error,
                  "invalid execution region boundary/submission resource");
    std::vector<size_t> members = region.memberEvents;
    if (members.empty())
      for (size_t event = region.beginEvent; event < region.endEvent; ++event)
        members.push_back(event);
    for (size_t event : members) {
      if (event >= graph.events.size() || covered[event])
        return fail(error, "overlapping execution regions or invalid execution "
                           "region members");
      covered[event] = true;
      owners[event] = r;
    }
    if (refined) {
      if (region.sourceTasks.empty() || region.memberEvents.empty() ||
          region.implementation.empty() || region.endpointMemory.empty() ||
          region.submissionCount != 1)
        return fail(error, "refined region needs tasks, implementation, "
                           "members and one invocation");
      for (size_t task : region.sourceTasks) {
        if (task >= taskOwners.size() || taskOwners[task] != SIZE_MAX)
          return fail(error, "duplicate or invalid region task");
        taskOwners[task] = r;
      }
      for (size_t parent : region.predecessorRegions)
        if (parent >= r)
          return fail(error, "cyclic region predecessor");
      for (size_t copy : region.inputCopies)
        if (copy >= graph.copies.size())
          return fail(error, "invalid region input copy");
      for (size_t copy : region.outputCopies)
        if (copy >= graph.copies.size())
          return fail(error, "invalid region output copy");
    }
  }
  if (std::find(covered.begin(), covered.end(), false) != covered.end())
    return fail(error, "event missing execution region");
  if (refined) {
    if (std::find(taskOwners.begin(), taskOwners.end(), SIZE_MAX) !=
        taskOwners.end())
      return fail(error, "source task missing execution region");
    for (size_t event = 0; event < graph.events.size(); ++event) {
      const auto &value = graph.events[event];
      if (value.kind != "transfer" && taskOwners[value.task] != owners[event])
        return fail(error, "compute event executes in wrong region");
      for (size_t dependency : value.dependencies)
        if (owners[dependency] > owners[event])
          return fail(error, "region order violates event dependency");
    }
    size_t cursor = 0;
    for (size_t r = 0; r < graph.executionRegions.size(); ++r) {
      const auto &region = graph.executionRegions[r];
      if (!region.invocationLifecycle.empty() &&
          region.invocationLifecycle != "host-only" &&
          region.invocationLifecycle != "host-device")
        return fail(error, "unsupported invocation lifecycle policy");
      bool device = region.invocationLifecycle == "host-device";
      if (region.invocationLifecycle.empty()) {
        // Compatibility for preserved heterogeneous CPU/NPU adapters. New
        // planners declare lifecycle explicitly, without interpreting names.
        device = region.placement == "npu";
        if (!device && region.placement != "cpu")
          return fail(error, "legacy refined region needs cpu/npu placement or "
                             "explicit invocation lifecycle");
      }
      std::vector<std::string> kinds =
          device ? std::vector<std::string>{"visible-to-device", "submit",
                                            "complete", "visible-to-host"}
                 : std::vector<std::string>{"submit", "complete"};
      for (const auto &kind : kinds) {
        if (cursor >= graph.regionEvents.size())
          return fail(error, "missing region lifecycle event");
        const auto &event = graph.regionEvents[cursor];
        if (event.region != r || event.kind != kind ||
            !resource(graph, event.resource))
          return fail(error, "invalid region lifecycle placement or resource");
        std::vector<size_t> expectedDependencies;
        if (cursor)
          expectedDependencies.push_back(cursor - 1);
        if (event.dependencies != expectedDependencies)
          return fail(error,
                      "missing region visibility or completion dependency");
        const auto &expectedCopies = kind == "visible-to-device"
                                         ? region.inputCopies
                                         : region.outputCopies;
        if ((kind == "visible-to-device" || kind == "visible-to-host")
                ? event.copies != expectedCopies
                : !event.copies.empty())
          return fail(error, "region visibility copy/version mismatch");
        if ((kind == "submit" || kind == "complete") &&
            event.resource != region.submissionResource)
          return fail(error, "region invocation resource mismatch");
        ++cursor;
      }
    }
    if (cursor != graph.regionEvents.size())
      return fail(error, "extra or duplicate region lifecycle submission");
  }
  MappingGraphSchedule expected;
  if (!schedule(graph, expected, error) ||
      !sameJSON(scheduleJSON(expected), scheduleJSON(graph.schedule)))
    return fail(
        error,
        "schedule/cost is not derived from checked dependencies and resources");
  return true;
}

llvm::json::Value
orbit::taskflow::serializeProgramGraph(const ProgramGraph &graph) {
  auto value = graphJSON(graph);
  auto identity = staticFingerprint(value);
  (*value.getAsObject())["id"] = identity;
  (*value.getAsObject())["cost"] = summarizeProgramGraphCost(graph);
  return value;
}
std::string orbit::taskflow::programGraphIdentity(const ProgramGraph &graph) {
  return staticFingerprint(graphJSON(graph));
}
llvm::json::Value
orbit::taskflow::summarizeProgramGraphCost(const ProgramGraph &graph) {
  if (graph.operatorSource)
    return summarizeProgramOperatorGraph(graph);
  int64_t reads = 0, writes = 0, traffic = 0, transfers = 0, submissions = 0;
  bool overflow = false, trafficOverflow = false, submissionOverflow = false;
  for (const auto &task : graph.source.tasks)
    for (const auto &access : task.accesses) {
      int64_t count = 1;
      for (const auto &loop : task.loops)
        if (loop.operation->isProperAncestor(access.operation)) {
          int64_t n = (loop.upper - loop.lower - 1) / loop.step + 1;
          if (n <= 0 || count > INT64_MAX / n) {
            overflow = true;
            break;
          }
          count *= n;
        }
      auto type = mlir::cast<mlir::MemRefType>(
          graph.source.storages[access.storage].root.getType());
      int64_t width = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
      if (count > INT64_MAX / width)
        overflow = true;
      else {
        if (access.read)
          overflow |= !add(reads, count * width);
        if (access.write)
          overflow |= !add(writes, count * width);
      }
    }
  for (const auto &event : graph.events)
    if (event.kind == "transfer") {
      ++transfers;
      trafficOverflow |= !add(traffic, event.transferBytes);
    }
  for (const auto &region : graph.executionRegions)
    submissionOverflow |= !add(submissions, region.submissionCount);
  // Match checked allocation policy, not the number of referencing copies.
  // Padding is only target alignment rounding; dormant template space and
  // installed-backend staging/bank padding are not inferred as padding here.
  struct Delta {
    int64_t starts = 0, ends = 0, paddingStarts = 0, paddingEnds = 0;
    int64_t unknownStarts = 0, unknownEnds = 0;
  };
  std::map<std::string, std::map<size_t, Delta>> deltas;
  std::set<std::string> invalidMemory;
  for (size_t allocation = 0; allocation < graph.allocations.size();
       ++allocation) {
    const auto &value = graph.allocations[allocation];
    auto memory =
        std::find_if(graph.memories.begin(), graph.memories.end(),
                     [&](const auto &m) { return m.id == value.memory; });
    if (memory == graph.memories.end() || value.bytes <= 0 ||
        memory->alignmentBytes <= 0) {
      invalidMemory.insert(value.memory);
      continue;
    }
    int64_t padding = 0;
    bool paddingKnown = memory->alignmentBytes == 1;
    if (!paddingKnown && graph.allocationLifetime == "copy-events" &&
        value.storage < graph.source.storages.size()) {
      const auto &storage = graph.source.storages[value.storage];
      MappingRegion full{value.storage,
                         std::vector<int64_t>(storage.shape.size(), 0),
                         storage.shape};
      int64_t payload = regionBytes(graph.source, full);
      if (payload > 0 && value.bytes >= payload) {
        int64_t remainder = payload % memory->alignmentBytes;
        padding = remainder ? memory->alignmentBytes - remainder : 0;
        paddingKnown =
            payload <= INT64_MAX - padding && payload + padding == value.bytes;
      }
    }
    std::vector<std::pair<size_t, size_t>> ranges;
    if (graph.allocationLifetime == "phase-pool")
      ranges.push_back({value.beginPhase, value.endPhase});
    else
      for (const auto &copy : graph.copies)
        if (copy.allocation == allocation)
          ranges.push_back({copy.availableFrom, copy.releasedAfter});
    std::sort(ranges.begin(), ranges.end());
    for (size_t index = 0; index < ranges.size();) {
      size_t begin = ranges[index].first, end = ranges[index++].second;
      while (index < ranges.size() && ranges[index].first <= end)
        end = std::max(end, ranges[index++].second);
      if (begin >= end) {
        invalidMemory.insert(value.memory);
        continue;
      }
      auto &start = deltas[value.memory][begin],
           &finish = deltas[value.memory][end];
      if (!add(start.starts, value.bytes) || !add(finish.ends, value.bytes))
        invalidMemory.insert(value.memory);
      if (paddingKnown) {
        if (!add(start.paddingStarts, padding) ||
            !add(finish.paddingEnds, padding))
          invalidMemory.insert(value.memory);
      } else {
        ++start.unknownStarts;
        ++finish.unknownEnds;
      }
    }
  }
  Object peaks;
  Array allocationAccounting;
  for (const auto &memory : graph.memories) {
    int64_t live = 0, padding = 0, unknown = 0, peak = 0, paddingAtPeak = 0;
    bool knownAtPeak = true;
    size_t peakPosition = 0;
    bool valid = !invalidMemory.count(memory.id);
    for (const auto &entry : deltas[memory.id]) {
      const auto &delta = entry.second;
      if (delta.ends > live || delta.paddingEnds > padding ||
          delta.unknownEnds > unknown) {
        valid = false;
        break;
      }
      live -= delta.ends;
      padding -= delta.paddingEnds;
      unknown -= delta.unknownEnds;
      if (!add(live, delta.starts) || !add(padding, delta.paddingStarts) ||
          !add(unknown, delta.unknownStarts)) {
        valid = false;
        break;
      }
      if (live > peak) {
        peak = live;
        peakPosition = entry.first;
        paddingAtPeak = padding;
        knownAtPeak = unknown == 0;
      }
    }
    peaks[memory.id] =
        valid ? llvm::json::Value(peak) : llvm::json::Value(nullptr);
    Object accounting{
        {"memory", memory.id},
        {"capacity_bytes", memory.capacityKnown
                               ? llvm::json::Value(memory.capacityBytes)
                               : llvm::json::Value(nullptr)},
        {"alignment_bytes", memory.alignmentBytes},
        {"peak_allocated_bytes",
         valid ? llvm::json::Value(peak) : llvm::json::Value(nullptr)},
        {"alignment_padding_bytes_at_peak",
         valid && knownAtPeak ? llvm::json::Value(paddingAtPeak)
                              : llvm::json::Value(nullptr)},
        {"allocation_payload_bytes_at_peak",
         valid && knownAtPeak ? llvm::json::Value(peak - paddingAtPeak)
                              : llvm::json::Value(nullptr)},
        {"peak_position", valid ? llvm::json::Value(int64_t(peakPosition))
                                : llvm::json::Value(nullptr)},
        {"position_unit",
         graph.allocationLifetime == "phase-pool" ? "phase" : "event"},
        {"status", valid ? "derived-from-checked-allocation-intervals"
                         : "invalid-or-overflow"},
        {"padding_scope", "target alignment rounding only; null when unrounded "
                          "pool payload is not represented"}};
    if (!memory.capacityKnown) {
      accounting["capacity_known"] = false;
      accounting["capacity_verified"] = false;
      accounting["capacity_scope"] = "physical byte capacity unmodeled; "
                                     "arithmetic accounting ceiling only";
    }
    allocationAccounting.push_back(std::move(accounting));
  }
  return Object{
      {"logical_source_read_bytes",
       overflow ? llvm::json::Value(nullptr) : llvm::json::Value(reads)},
      {"logical_source_write_bytes",
       overflow ? llvm::json::Value(nullptr) : llvm::json::Value(writes)},
      {"planned_transfer_bytes", trafficOverflow ? llvm::json::Value(nullptr)
                                                 : llvm::json::Value(traffic)},
      {"creation_transfer_events", transfers},
      {"submission_count", submissionOverflow ? llvm::json::Value(nullptr)
                                              : llvm::json::Value(submissions)},
      {"peak_memory_bytes", std::move(peaks)},
      {"allocation_accounting", std::move(allocationAccounting)},
      {"allocation_peak_scope",
       graph.allocationLifetime == "phase-pool"
           ? "all reserved pool allocations over phase intervals, including "
             "dormant templates"
           : "union of copy event intervals per allocation; shared copies "
             "counted once"},
      {"payload_scope",
       "allocation capacity minus known alignment padding; not live logical "
       "data or measured physical SRAM usage"},
      {"backend_internal_padding_status", "not inferred from this graph"},
      {"compute_instances", int64_t(graph.instances.size())},
      {"complete_graph_ms",
       graph.schedule.estimatedMakespanMs
           ? llvm::json::Value(*graph.schedule.estimatedMakespanMs)
           : llvm::json::Value(nullptr)},
      {"status",
       graph.schedule.estimatedMakespanMs ? "target-estimate" : "uncalibrated"},
      {"boundary", "serial dependency/resource schedule; no communication "
                   "added to a separately selected whole-command composite"},
      {"traffic_scope", "logical planned rectangular payload, not measured "
                        "physical DDR traffic"}};
}
std::string orbit::taskflow::renderProgramGraphDOT(const ProgramGraph &graph) {
  auto quote = [](const std::string &value) {
    std::string out;
    llvm::raw_string_ostream stream(out);
    stream << llvm::json::Value(value);
    return out;
  };
  std::ostringstream out;
  out << "digraph orbit_program {\n  rankdir=LR;\n  label="
      << quote("ProgramGraph " + programGraphIdentity(graph)) << ";\n";
  for (size_t task = 0; task < graph.source.tasks.size(); ++task)
    out << "  task" << task
        << " [shape=box,label=" << quote("task " + graph.source.tasks[task].id)
        << "];\n";
  for (size_t storage = 0; storage < graph.source.storages.size(); ++storage)
    out << "  data" << storage << " [shape=folder,label="
        << quote("data " + graph.source.storages[storage].id + " (" +
                 graph.source.storages[storage].role + ")")
        << "];\n";
  for (size_t copy = 0; copy < graph.copies.size(); ++copy) {
    const auto &value = graph.copies[copy];
    const auto &region = graph.regions[graph.versions[value.version].region];
    out << "  copy" << copy << " [shape=ellipse,label="
        << quote("copy " + std::to_string(copy) + " v" +
                 std::to_string(value.version) + " @" +
                 graph.allocations[value.allocation].memory + " [" +
                 std::to_string(value.availableFrom) + "," +
                 std::to_string(value.releasedAfter) + ")")
        << "];\n";
    out << "  data" << region.storage << " -> copy" << copy
        << " [style=dotted,label=\"region/version\"];\n";
  }
  for (size_t index = 0; index < graph.events.size(); ++index) {
    const auto &event = graph.events[index];
    out << "  event" << index
        << " [shape=" << (event.kind == "transfer" ? "diamond" : "box")
        << ",label="
        << quote(event.kind + " " + std::to_string(index) + " @" +
                 event.resource)
        << "];\n";
    if (event.kind != "transfer")
      out << "  task" << event.task << " -> event" << index
          << " [style=dotted,label=\"instance\"];\n";
    for (size_t copy : event.reads)
      out << "  copy" << copy << " -> event" << index << " [label=\"read\"];\n";
    for (size_t copy : event.writes)
      out << "  event" << index << " -> copy" << copy
          << " [label=\"create\"];\n";
    for (size_t parent : event.dependencies)
      out << "  event" << parent << " -> event" << index
          << " [style=dashed,color=gray];\n";
  }
  for (const auto &edge : graph.orderEdges)
    out << "  event" << edge.producer << " -> event" << edge.consumer
        << " [color=red,label=\"order only\"];\n";
  out << "}\n";
  return out.str();
}
