#include "Taskflow/ProgramOperatorMapping.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <cmath>
#include <set>
using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
bool fail(std::string &error, const std::string &message) {
  error = "external joint mapping: " + message;
  return false;
}
bool validatePacking(const ProgramOperatorSource &source,
                     const ProgramOperatorImplementation &impl,
                     const ProgramOperatorPackedInput &packing,
                     std::string &error) {
  if (packing.sourceValue >= source.values.size() ||
      packing.packingIdentity.empty() || packing.fragments.empty() ||
      std::find(impl.inputs.begin(), impl.inputs.end(), packing.sourceValue) ==
          impl.inputs.end())
    return fail(error, "packing must bind an actual implementation input");
  const auto &value = source.values[packing.sourceValue];
  if (value.role != "input" || value.view || value.allocationView ||
      value.stateAliasOf)
    return fail(error,
                "packed replacement currently requires a readonly root input");
  for (size_t op = 0; op < source.operators.size(); ++op)
    if (std::find(source.operators[op].inputs.begin(),
                  source.operators[op].inputs.end(),
                  packing.sourceValue) != source.operators[op].inputs.end() &&
        std::find(impl.operators.begin(), impl.operators.end(), op) ==
            impl.operators.end())
      return fail(error, "packed representation has a source consumer outside "
                         "this registered cover");
  const size_t rank = value.shape.size();
  __int128 volume = 1, covered = 0;
  for (auto n : value.shape)
    volume *= n;
  std::set<std::string> ids;
  int64_t element =
      value.dtype == "f32" || value.dtype == "i32"
          ? 4
          : (value.dtype == "bf16" || value.dtype == "f16" ||
                     value.dtype == "i16"
                 ? 2
                 : (value.dtype == "f64" || value.dtype == "i64"
                        ? 8
                        : (value.dtype == "i8" || value.dtype == "ui8" ? 1
                                                                       : 0)));
  for (const auto &f : packing.fragments) {
    if (f.allocationId.empty() || !ids.insert(f.allocationId).second ||
        f.memory.empty() || !element || f.dtype != value.dtype ||
        f.bytes <= 0 || f.sourceOffsets.size() != rank ||
        f.sourceExtents.size() != rank || f.physicalShape.size() != rank ||
        f.physicalOffsets.size() != rank || f.strides.size() != rank)
      return fail(error, "invalid packed fragment identity/dtype/rank");
    __int128 part = 1, physical = element, stride = 1;
    for (size_t reverse = rank; reverse > 0; --reverse) {
      size_t i = reverse - 1;
      if (f.sourceOffsets[i] < 0 || f.sourceExtents[i] <= 0 ||
          __int128(f.sourceOffsets[i]) + f.sourceExtents[i] > value.shape[i] ||
          f.physicalShape[i] <= 0 || f.physicalOffsets[i] < 0 ||
          __int128(f.physicalOffsets[i]) + f.sourceExtents[i] >
              f.physicalShape[i] ||
          stride > INT64_MAX || f.strides[i] != stride)
        return fail(error, "packed fragment exceeds source/physical bounds or "
                           "row-major strides");
      part *= f.sourceExtents[i];
      physical *= f.physicalShape[i];
      stride *= f.physicalShape[i];
      if (physical > INT64_MAX || part > INT64_MAX || stride > INT64_MAX)
        return fail(error, "packed fragment extent/byte overflow");
    }
    if (physical != f.bytes)
      return fail(error,
                  "physical packing bytes do not match padded allocation");
    covered += part;
  }
  for (size_t a = 0; a < packing.fragments.size(); ++a)
    for (size_t b = a + 1; b < packing.fragments.size(); ++b) {
      bool overlap = true;
      for (size_t i = 0; i < rank; ++i)
        overlap &= __int128(packing.fragments[a].sourceOffsets[i]) <
                       __int128(packing.fragments[b].sourceOffsets[i]) +
                           packing.fragments[b].sourceExtents[i] &&
                   __int128(packing.fragments[b].sourceOffsets[i]) <
                       __int128(packing.fragments[a].sourceOffsets[i]) +
                           packing.fragments[a].sourceExtents[i];
      if (overlap)
        return fail(error, "overlapping source packing fragments");
    }
  return covered == volume ||
         fail(error, "packing leaves a source-input coverage hole");
}
bool applyPacking(ProgramGraph &graph, const ProgramOperatorMapping &mapping,
                  std::string &error);
bool validateWorkPartitions(const ProgramOperatorSource &source,
                            const ProgramOperatorImplementation &impl,
                            const ProgramOperatorExecutionVariant &variant,
                            std::string &error) {
  if (variant.replicas == 1 && variant.workPartitions.empty())
    return true;
  if (variant.replicas < 1 || variant.workPartitions.empty())
    return fail(error, "replicated variant needs explicit disjoint "
                       "source-output work partitions");
  std::map<size_t, std::vector<const ProgramOperatorWorkPartition *>> outputs;
  for (const auto &part : variant.workPartitions) {
    if (part.outputValue >= source.values.size() || part.replica < 0 ||
        part.replica >= variant.replicas ||
        std::find(impl.outputs.begin(), impl.outputs.end(), part.outputValue) ==
            impl.outputs.end())
      return fail(
          error,
          "replica work partition names invalid source output or replica");
    const auto &shape = source.values[part.outputValue].shape;
    if (part.offsets.size() != shape.size() ||
        part.extents.size() != shape.size())
      return fail(error, "replica output partition rank mismatch");
    for (size_t i = 0; i < shape.size(); ++i)
      if (part.offsets[i] < 0 || part.extents[i] <= 0 ||
          __int128(part.offsets[i]) + part.extents[i] > shape[i])
        return fail(error, "replica output partition exceeds source bounds");
    outputs[part.outputValue].push_back(&part);
  }
  if (outputs.size() != impl.outputs.size())
    return fail(error, "replica work does not cover every source output");
  for (const auto &entry : outputs) {
    __int128 volume = 1, covered = 0;
    const auto &shape = source.values[entry.first].shape;
    for (auto n : shape)
      volume *= n;
    std::set<int64_t> replicas;
    for (const auto *part : entry.second) {
      __int128 size = 1;
      for (auto n : part->extents)
        size *= n;
      covered += size;
      replicas.insert(part->replica);
    }
    if (covered != volume || replicas.size() != size_t(variant.replicas))
      return fail(error, "replica work leaves output holes or empty replicas");
    for (size_t a = 0; a < entry.second.size(); ++a)
      for (size_t b = a + 1; b < entry.second.size(); ++b) {
        bool overlap = true;
        for (size_t i = 0; i < shape.size(); ++i)
          overlap &= __int128(entry.second[a]->offsets[i]) <
                         __int128(entry.second[b]->offsets[i]) +
                             entry.second[b]->extents[i] &&
                     __int128(entry.second[b]->offsets[i]) <
                         __int128(entry.second[a]->offsets[i]) +
                             entry.second[a]->extents[i];
        if (overlap)
          return fail(error, "replica source-output work overlaps");
      }
  }
  return true;
}
using Calendar = std::map<std::string, std::map<double, int64_t>>;
bool add(int64_t &a, int64_t b) {
  if (b < 0 || a > INT64_MAX - b)
    return false;
  a += b;
  return true;
}
bool reserve(Calendar &calendar, const std::string &key, double begin,
             double end, int64_t amount, int64_t capacity, std::string &error) {
  if (amount < 0 || amount > capacity)
    return fail(error, "reservation exceeds capacity: " + key);
  if (begin == end)
    return true;
  for (auto change :
       {std::make_pair(begin, amount), std::make_pair(end, -amount)}) {
    __int128 delta = __int128(calendar[key][change.first]) + change.second;
    if (delta < INT64_MIN || delta > INT64_MAX)
      return fail(error, "calendar delta overflow: " + key);
    calendar[key][change.first] = int64_t(delta);
  }
  int64_t live = 0;
  for (auto change : calendar[key]) {
    // Summed deltas may overflow if malformed callers reserve INT64_MAX twice.
    __int128 next = __int128(live) + change.second;
    if (next < 0 || next > capacity)
      return fail(error, "overlapping reservations exceed capacity: " + key);
    live = int64_t(next);
  }
  return live == 0 || fail(error, "unclosed reservation: " + key);
}
struct ScheduleView {
  const ProgramOperatorSelection *operatorSelection;
  const std::vector<MappingEvent> &events;
  const std::vector<MappingAllocation> &allocations;
  const std::vector<StaticResource> &resources;
  const std::vector<StaticResourceDomain> &domains;
  const std::vector<StaticMemory> &memories;
  const std::vector<StaticTransfer> &links;
  explicit ScheduleView(const ProgramGraph &g)
      : operatorSelection(g.operatorSelection.get()), events(g.events),
        allocations(g.allocations), resources(g.resources), domains(g.domains),
        memories(g.memories), links(g.links) {}
  ScheduleView(const ProgramOperatorSelection &s,
               const std::vector<MappingEvent> &e,
               const std::vector<MappingAllocation> &a, const StaticConfig &t)
      : operatorSelection(&s), events(e), allocations(a),
        resources(t.resources), domains(t.domains), memories(t.memories),
        links(t.links) {}
};
bool build(const ScheduleView &g, const ProgramOperatorMapping &mapping,
           MappingGraphSchedule &schedule, std::string &error) {
  if (!g.operatorSelection || mapping.registrationIdentity.empty() ||
      mapping.variants.size() != g.events.size() ||
      mapping.readyPriority.size() != g.events.size())
    return fail(error,
                "missing registered variants or complete ready priority");
  std::vector<size_t> rank(g.events.size(), SIZE_MAX);
  for (size_t i = 0; i < mapping.readyPriority.size(); ++i) {
    auto event = mapping.readyPriority[i];
    if (event >= rank.size() || rank[event] != SIZE_MAX)
      return fail(error, "ready priority is not a permutation");
    rank[event] = i;
  }
  StaticConfig target;
  target.resources = g.resources;
  target.domains = g.domains;
  target.memories = g.memories;
  target.links = g.links;
  std::vector<std::map<std::string, int64_t>> demands(g.events.size());
  std::map<std::string, std::pair<std::string, int64_t>> residents;
  schedule = {};
  schedule.policy = "external-resource-calendar";
  schedule.modelIdentity = mapping.registrationIdentity;
  schedule.timingBoundary =
      "inclusive-invocation-wall; no additive launch/transport";
  schedule.events.resize(g.events.size());
  schedule.eventDurationsMs.resize(g.events.size());
  for (size_t i = 0; i < g.events.size(); ++i) {
    const auto &v = mapping.variants[i];
    const auto &impl = g.operatorSelection->regions[i];
    if (v.replicas > 1 && v.workPartitions.empty())
      return fail(
          error,
          "replicated invocation has no checked output partition contract");
    if (v.runtimeContextIdentity.empty() || v.dataflow.empty() ||
        v.residencePolicy.empty())
      return fail(
          error,
          "missing registered runtime context/dataflow/residence binding");
    if (v.implementationId != impl.id ||
        v.artifactIdentity != impl.artifactIdentity ||
        v.entrypoint != impl.entrypoint || v.replicas < 1 ||
        (v.replicas == 1 ? v.workDistribution != "single-invocation"
                         : v.workDistribution != "registered-disjoint-work") ||
        v.resourceDemands.empty() || !v.resourceDemands.count(impl.resource))
      return fail(error, "variant identity, replica distribution or resource "
                         "binding mismatch");
    for (auto axis : v.resourceShape)
      if (axis <= 0)
        return fail(error, "nonpositive registered spatial shape");
    if (!impl.wallMs || !std::isfinite(*impl.wallMs) || *impl.wallMs < 0)
      return fail(error, "unknown/nonfinite inclusive duration: " + impl.id);
    schedule.eventDurationsMs[i] = *impl.wallMs;
    if (!programResourceDemands(target, v.resourceDemands, demands[i], error))
      return false;
    for (const auto &resident : v.residentAllocations) {
      if (resident.first.empty() || resident.second.second <= 0 ||
          !programMemory(target, resident.second.first))
        return fail(error, "invalid resident allocation");
      auto inserted = residents.emplace(resident);
      if (!inserted.second && inserted.first->second != resident.second)
        return fail(error,
                    "resident allocation identity changes size/location");
    }
    for (const auto &temp : v.temporaryBytes)
      if (temp.second < 0 || !programMemory(target, temp.first))
        return fail(error, "unknown memory or negative temporary reservation");
  }
  // Existing ABI allocations stay live across this synchronous invocation.
  // This deliberately does not invent releases or BO migrations. A backend
  // must register a different executable residence plan to change that policy.
  std::map<std::string, int64_t> persistent;
  for (const auto &a : g.allocations)
    if (!add(persistent[a.memory], a.bytes))
      return fail(error, "ABI byte accounting overflow");
  std::map<std::string, std::pair<std::string, int64_t>> physicalAllocations;
  for (const auto &a : g.allocations) {
    if (!physicalAllocations.emplace(a.key, std::make_pair(a.memory, a.bytes))
             .second)
      return fail(error, "physical allocation identity is not unique");
  }
  for (const auto &r : residents) {
    auto existing = physicalAllocations.find(r.first);
    if (existing != physicalAllocations.end()) {
      if (existing->second != r.second)
        return fail(error,
                    "resident alias disagrees with actual physical allocation");
      continue; // Prepared host and shared device BO are one allocation.
    }
    if (!add(persistent[r.second.first], r.second.second))
      return fail(error, "resident byte accounting overflow");
  }
  for (const auto &m : target.memories)
    if (m.capacityKnown && persistent[m.id] > m.capacityBytes)
      return fail(error, "resident/ABI storage exceeds capacity: " + m.id);
  bool serial = true;
  for (size_t i = 0; i < g.events.size(); ++i)
    serial &= g.events[i].dependencies ==
              (i ? std::vector<size_t>{i - 1} : std::vector<size_t>{});
  if (serial) {
    // The current synchronous ABI has a real complete-order edge between
    // invocations. Calendar feasibility therefore reduces exactly to each
    // reservation plus persistent storage; no concurrency is being assumed.
    double clock = 0;
    for (size_t i = 0; i < g.events.size(); ++i) {
      for (const auto &d : demands[i]) {
        auto *r = programResource(target, d.first);
        if (!r || d.second > r->capacity)
          return fail(error,
                      "exclusive invocation exceeds resource capacity: " +
                          d.first);
      }
      for (const auto &t : mapping.variants[i].temporaryBytes) {
        auto *m = programMemory(target, t.first);
        if (m->capacityKnown && t.second > m->capacityBytes - persistent[m->id])
          return fail(
              error,
              "invocation scratch plus retained storage exceeds capacity: " +
                  m->id);
      }
      double end = clock + schedule.eventDurationsMs[i];
      if (!std::isfinite(end))
        return fail(error, "schedule time overflow");
      schedule.events[i] = {i, g.events[i].dependencies, clock, end};
      clock = end;
    }
    schedule.estimatedMakespanMs = clock;
    return true;
  }
  Calendar resourceCalendar, memoryCalendar;
  std::vector<bool> done(g.events.size());
  double makespan = 0;
  for (size_t issued = 0; issued < g.events.size(); ++issued) {
    size_t chosen = SIZE_MAX;
    for (size_t i = 0; i < g.events.size(); ++i) {
      if (done[i])
        continue;
      bool ready = true;
      for (size_t p : g.events[i].dependencies) {
        if (p >= g.events.size())
          return fail(error, "dependency out of bounds");
        ready &= done[p];
      }
      if (ready && (chosen == SIZE_MAX || rank[i] < rank[chosen]))
        chosen = i;
    }
    if (chosen == SIZE_MAX)
      return fail(error, "cyclic implementation schedule");
    double earliest = 0;
    for (size_t p : g.events[chosen].dependencies)
      earliest = std::max(earliest, *schedule.events[p].endMs);
    // Every contention transition is a calendar endpoint, so scanning these
    // points finds the earliest feasible nonpreemptive reservation exactly.
    std::set<double> starts{earliest};
    for (const auto *calendar : {&resourceCalendar, &memoryCalendar})
      for (const auto &r : *calendar)
        for (const auto &point : r.second)
          if (point.first >= earliest)
            starts.insert(point.first);
    bool placed = false;
    for (double begin : starts) {
      double end = begin + schedule.eventDurationsMs[chosen];
      if (!std::isfinite(end))
        return fail(error, "schedule time overflow");
      auto resources = resourceCalendar, memory = memoryCalendar;
      std::string rejected;
      bool fits = true;
      for (const auto &d : demands[chosen]) {
        auto *r = programResource(target, d.first);
        fits &= r && reserve(resources, d.first, begin, end, d.second,
                             r->capacity, rejected);
      }
      for (const auto &t : mapping.variants[chosen].temporaryBytes) {
        auto *m = programMemory(target, t.first);
        int64_t available = m->capacityKnown
                                ? m->capacityBytes - persistent[m->id]
                                : INT64_MAX - persistent[m->id];
        fits &=
            reserve(memory, t.first, begin, end, t.second, available, rejected);
      }
      if (!fits)
        continue;
      resourceCalendar = std::move(resources);
      memoryCalendar = std::move(memory);
      schedule.events[chosen] = {chosen, g.events[chosen].dependencies, begin,
                                 end};
      makespan = std::max(makespan, end);
      placed = done[chosen] = true;
      break;
    }
    if (!placed)
      return fail(error, "no capacity-feasible invocation placement");
  }
  schedule.estimatedMakespanMs = makespan;
  return true;
}
} // namespace
bool orbit::taskflow::scheduleProgramOperatorGraph(
    ProgramGraph &graph, const ProgramOperatorMapping &mapping,
    std::string &error) {
  if (!graph.operatorSource || !graph.operatorSelection ||
      graph.operatorMapping || !checkProgramGraph(graph, error))
    return fail(error,
                "requires an unscheduled checked physical source graph: " +
                    error);
  if (mapping.variants.size() != graph.events.size())
    return fail(error, "variant count does not match actual invocations");
  for (size_t i = 0; i < mapping.variants.size(); ++i)
    if (!validateWorkPartitions(*graph.operatorSource,
                                graph.operatorSelection->regions[i],
                                mapping.variants[i], error))
      return false;
  if (!applyPacking(graph, mapping, error))
    return false;
  MappingGraphSchedule schedule;
  if (!build(ScheduleView(graph), mapping, schedule, error))
    return false;
  graph.operatorMapping = std::make_shared<ProgramOperatorMapping>(mapping);
  graph.schedule = std::move(schedule);
  graph.estimatedMakespanMs = graph.schedule.estimatedMakespanMs;
  return checkProgramGraph(graph, error);
}
bool orbit::taskflow::checkProgramOperatorMapping(const ProgramGraph &graph,
                                                  std::string &error) {
  if (!graph.operatorSource || !graph.operatorSelection ||
      !graph.operatorMapping)
    return fail(error, "missing external source/selection/mapping contract");
  for (size_t i = 0; i < graph.operatorMapping->variants.size(); ++i)
    if (i >= graph.operatorSelection->regions.size() ||
        !validateWorkPartitions(*graph.operatorSource,
                                graph.operatorSelection->regions[i],
                                graph.operatorMapping->variants[i], error))
      return false;
  MappingGraphSchedule expected;
  if (!build(ScheduleView(graph), *graph.operatorMapping, expected, error))
    return false;
  const auto &actual = graph.schedule;
  if (actual.policy != expected.policy ||
      actual.modelIdentity != expected.modelIdentity ||
      actual.timingBoundary != expected.timingBoundary ||
      actual.eventDurationsMs != expected.eventDurationsMs ||
      actual.estimatedMakespanMs != expected.estimatedMakespanMs ||
      graph.estimatedMakespanMs != expected.estimatedMakespanMs ||
      !actual.uncalibrated.empty() || !actual.configurations.empty() ||
      actual.events.size() != expected.events.size())
    return fail(error, "timing/model witness differs from registered mapping");
  for (size_t i = 0; i < actual.events.size(); ++i) {
    const auto &a = actual.events[i], &b = expected.events[i];
    if (a.event != b.event || a.predecessors != b.predecessors ||
        a.startMs != b.startMs || a.endMs != b.endMs)
      return fail(error, "event timing violates dependency/resource calendar");
  }
  return true;
}
j::Value orbit::taskflow::serializeProgramOperatorMapping(
    const ProgramOperatorMapping &m) {
  j::Array variants, priority;
  for (auto p : m.readyPriority)
    priority.push_back(int64_t(p));
  for (const auto &v : m.variants) {
    j::Object demands, temporary, residents, parameters;
    for (auto p : v.implementationParameters)
      parameters[p.first] = p.second;
    j::Array shape;
    for (auto d : v.resourceDemands)
      demands[d.first] = d.second;
    for (auto t : v.temporaryBytes)
      temporary[t.first] = t.second;
    for (auto r : v.residentAllocations)
      residents[r.first] =
          j::Object{{"memory", r.second.first}, {"bytes", r.second.second}};
    for (auto s : v.resourceShape)
      shape.push_back(s);
    j::Array packedInputs, unproved;
    for (const auto &reason : v.unprovedConstraints)
      unproved.push_back(reason);
    auto integers = [](const std::vector<int64_t> &values) {
      j::Array array;
      for (auto n : values)
        array.push_back(n);
      return array;
    };
    j::Array work;
    for (const auto &part : v.workPartitions)
      work.push_back(j::Object{{"output_value", int64_t(part.outputValue)},
                               {"replica", part.replica},
                               {"offsets", integers(part.offsets)},
                               {"extents", integers(part.extents)}});
    for (const auto &packing : v.packedInputs) {
      j::Array fragments;
      for (const auto &f : packing.fragments)
        fragments.push_back(
            j::Object{{"allocation_id", f.allocationId},
                      {"memory", f.memory},
                      {"dtype", f.dtype},
                      {"bytes", f.bytes},
                      {"source_offsets", integers(f.sourceOffsets)},
                      {"source_extents", integers(f.sourceExtents)},
                      {"physical_shape", integers(f.physicalShape)},
                      {"physical_offsets", integers(f.physicalOffsets)},
                      {"strides", integers(f.strides)}});
      packedInputs.push_back(
          j::Object{{"source_value", int64_t(packing.sourceValue)},
                    {"packing_identity", packing.packingIdentity},
                    {"fragments", std::move(fragments)}});
    }
    variants.push_back(
        j::Object{{"implementation_id", v.implementationId},
                  {"artifact_identity", v.artifactIdentity},
                  {"entrypoint", v.entrypoint},
                  {"runtime_context_identity", v.runtimeContextIdentity},
                  {"dataflow", v.dataflow},
                  {"residence_policy", v.residencePolicy},
                  {"implementation_parameters", std::move(parameters)},
                  {"packed_inputs", std::move(packedInputs)},
                  {"work_partitions", std::move(work)},
                  {"unproved_constraints", std::move(unproved)},
                  {"resource_demands", std::move(demands)},
                  {"resource_shape", std::move(shape)},
                  {"replicas", v.replicas},
                  {"work_distribution", v.workDistribution},
                  {"temporary_bytes", std::move(temporary)},
                  {"resident_allocations", std::move(residents)}});
  }
  return j::Object{{"schema", "orbit.program-operator-mapping.v1"},
                   {"host_memory", m.hostMemory},
                   {"registration_identity", m.registrationIdentity},
                   {"variants", std::move(variants)},
                   {"ready_priority", std::move(priority)}};
}
bool orbit::taskflow::applyProgramOperatorMappingAction(
    const ProgramOperatorMappingCandidate &parent,
    const ProgramOperatorMappingAction &action,
    ProgramOperatorMappingCandidate &child, std::string &error) {
  child = parent;
  if (action.kind == ProgramOperatorMappingActionKind::ReadyPriority) {
    if (action.readyPriority.size() != parent.selection.regions.size())
      return fail(error, "priority action must cover every actual invocation");
    std::set<size_t> indices(action.readyPriority.begin(),
                             action.readyPriority.end());
    if (indices.size() != action.readyPriority.size() ||
        (!indices.empty() && *indices.rbegin() >= indices.size()))
      return fail(error, "priority action is not a permutation");
    child.mapping.readyPriority = action.readyPriority;
    return true;
  }
  if (action.kind != ProgramOperatorMappingActionKind::ReplaceExecutableCover)
    return fail(error, "independent copy/replica/shape/residence action "
                       "requires a registered executable variant and actual "
                       "runtime lowering; use ReplaceExecutableCover");
  if (!action.regionCount ||
      action.firstRegion >= parent.selection.regions.size() ||
      action.regionCount >
          parent.selection.regions.size() - action.firstRegion ||
      action.replacements.empty() ||
      action.replacements.size() != action.replacementVariants.size() ||
      parent.mapping.variants.size() != parent.selection.regions.size())
    return fail(error, "invalid executable structural replacement range");
  std::set<size_t> before, after;
  for (size_t i = action.firstRegion;
       i < action.firstRegion + action.regionCount; ++i)
    for (auto op : parent.selection.regions[i].operators)
      if (!before.insert(op).second)
        return fail(error, "parent cover overlap");
  for (const auto &impl : action.replacements) {
    if (!impl.executable)
      return fail(error, "replacement has no executable artifact");
    for (auto op : impl.operators)
      if (!after.insert(op).second)
        return fail(error, "replacement cover overlap");
  }
  if (before != after)
    return fail(error, "structural action changes source coverage");
  auto replace = [&](auto &values, const auto &replacement) {
    auto begin = values.begin() + action.firstRegion;
    begin = values.erase(begin, begin + action.regionCount);
    values.insert(begin, replacement.begin(), replacement.end());
  };
  replace(child.selection.regions, action.replacements);
  replace(child.mapping.variants, action.replacementVariants);
  // Changing invocation count invalidates ordinal priorities. No prior timing,
  // copies or schedule is reused across the structural materialization
  // boundary.
  child.mapping.readyPriority.resize(child.selection.regions.size());
  for (size_t i = 0; i < child.mapping.readyPriority.size(); ++i)
    child.mapping.readyPriority[i] = i;
  return true;
}
ProgramSearchEvaluation
orbit::taskflow::evaluateProgramOperatorMappingCandidate(
    std::shared_ptr<const ProgramOperatorSource> source,
    const ProgramOperatorMappingCandidate &candidate,
    const StaticConfig &target, llvm::StringRef hostMemory,
    ProgramGraph &materialized) {
  ProgramSearchEvaluation result;
  std::string error;
  if (!materializeProgramOperatorGraph(source, candidate.selection, target,
                                       hostMemory, materialized, error) ||
      !scheduleProgramOperatorGraph(materialized, candidate.mapping, error)) {
    result.message = error;
    // A missing implementation duration is unavailable evidence, rather than
    // mathematical infeasibility. No zero-duration replacement is introduced.
    result.feasibilityKnown =
        error.find("unknown/nonfinite inclusive duration") == std::string::npos;
    return result;
  }
  result.feasible = true;
  result.costMs = materialized.estimatedMakespanMs;
  j::Array unproved;
  std::set<std::string> usedMemories;
  for (const auto &a : materialized.allocations)
    usedMemories.insert(a.memory);
  for (const auto &v : candidate.mapping.variants) {
    for (const auto &reason : v.unprovedConstraints) {
      result.feasibilityKnown = false;
      unproved.push_back(v.implementationId + ": " + reason);
    }
    for (const auto &a : v.temporaryBytes)
      usedMemories.insert(a.first);
    for (const auto &a : v.residentAllocations)
      usedMemories.insert(a.second.first);
  }
  for (const auto &m : target.memories)
    if (!m.capacityKnown && usedMemories.count(m.id)) {
      result.feasibilityKnown = false;
      unproved.push_back("physical capacity unavailable: " + m.id);
    }
  result.message = "checked source/cover/ABI/copy/resource-calendar mapping; "
                   "backend arithmetic and registration proof remain external";
  result.result =
      j::Object{{"schedule_policy", materialized.schedule.policy},
                {"mapping", serializeProgramOperatorMapping(candidate.mapping)},
                {"unproved_target_constraints", std::move(unproved)},
                {"independent_additive_cover_model", false}};
  return result;
}
bool orbit::taskflow::parseProgramOperatorMappingRegistrations(
    const ProgramOperatorSource &source, const j::Value &catalogue,
    const std::vector<ProgramOperatorImplementation> &implementations,
    std::string &registrationIdentity,
    std::map<std::string, ProgramOperatorExecutionVariant> &registrations,
    std::string &error) {
  auto *root = catalogue.getAsObject();
  auto identity =
      root ? root->getString("execution_registration_identity") : std::nullopt;
  auto *entries = root ? root->getArray("implementations") : nullptr;
  if (!identity || identity->empty() || !entries)
    return fail(error, "missing backend execution registration identity");
  registrationIdentity = identity->str();
  registrations.clear();
  std::map<std::string, const ProgramOperatorImplementation *> registered;
  for (const auto &impl : implementations)
    registered.emplace(impl.id, &impl);
  auto integerMap = [&](const j::Object &object, llvm::StringRef field,
                        std::map<std::string, int64_t> &out, bool required) {
    auto *values = object.getObject(field);
    if (!values)
      return !required && !object.get(field);
    for (const auto &value : *values) {
      auto amount = value.second.getAsInteger();
      if (!amount || *amount < 0)
        return false;
      out.emplace(value.first.str(), *amount);
    }
    return true;
  };
  for (const auto &entry : *entries) {
    auto *object = entry.getAsObject();
    auto id = object ? object->getString("id") : std::nullopt;
    auto *v = object ? object->getObject("execution_variant") : nullptr;
    if (!id || !registered.count(id->str()) || !v)
      return fail(error,
                  "implementation lacks actual execution_variant registration");
    const auto &impl = *registered.at(id->str());
    ProgramOperatorExecutionVariant variant;
    variant.implementationId = impl.id;
    variant.artifactIdentity = impl.artifactIdentity;
    variant.entrypoint = impl.entrypoint;
    auto context = v->getString("runtime_context_identity");
    auto dataflow = v->getString("dataflow");
    auto residence = v->getString("residence_policy");
    auto replicas = v->getInteger("replicas");
    auto work = v->getString("work_distribution");
    if (!context || context->empty() || !dataflow || dataflow->empty() ||
        !residence || residence->empty() || !replicas || *replicas < 1 || !work)
      return fail(
          error,
          "missing explicit context/dataflow/residence/replica contract");
    variant.runtimeContextIdentity = context->str();
    variant.dataflow = dataflow->str();
    variant.residencePolicy = residence->str();
    variant.replicas = *replicas;
    variant.workDistribution = work->str();
    if (!integerMap(*v, "resource_demands", variant.resourceDemands, true) ||
        !integerMap(*v, "temporary_bytes", variant.temporaryBytes, false) ||
        !integerMap(*v, "implementation_parameters",
                    variant.implementationParameters, false))
      return fail(error, "invalid resource/scratch/parameter integer table");
    if (v->get("resource_shape") && !v->getArray("resource_shape"))
      return fail(error, "resource_shape must be an integer array");
    if (auto *shape = v->getArray("resource_shape"))
      for (const auto &axis : *shape) {
        auto n = axis.getAsInteger();
        if (!n || *n <= 0)
          return fail(error, "invalid spatial shape axis");
        variant.resourceShape.push_back(*n);
      }
    if (v->get("resident_allocations") && !v->getObject("resident_allocations"))
      return fail(error, "resident_allocations must be an allocation table");
    if (auto *residents = v->getObject("resident_allocations"))
      for (const auto &entry : *residents) {
        auto *a = entry.second.getAsObject();
        auto memory = a ? a->getString("memory") : std::nullopt;
        auto bytes = a ? a->getInteger("bytes") : std::nullopt;
        if (!memory || memory->empty() || !bytes || *bytes <= 0)
          return fail(error, "invalid retained allocation identity/bytes");
        variant.residentAllocations.emplace(
            entry.first.str(), std::make_pair(memory->str(), *bytes));
      }
    if (auto *work = v->getArray("work_partitions")) {
      for (const auto &value : *work) {
        auto *part = value.getAsObject();
        auto output = part ? part->getString("output_value") : std::nullopt;
        auto replica = part ? part->getInteger("replica") : std::nullopt;
        auto *offsets = part ? part->getArray("offsets") : nullptr;
        auto *extents = part ? part->getArray("extents") : nullptr;
        if (!output || !replica || !offsets || !extents)
          return fail(
              error,
              "replica work needs output_value, replica, offsets and extents");
        auto sourceValue =
            std::find_if(source.values.begin(), source.values.end(),
                         [&](const auto &v) { return v.id == *output; });
        if (sourceValue == source.values.end())
          return fail(error, "replica output source value does not exist");
        ProgramOperatorWorkPartition parsed;
        parsed.outputValue = size_t(sourceValue - source.values.begin());
        parsed.replica = *replica;
        for (const auto &n : *offsets) {
          auto number = n.getAsInteger();
          if (!number)
            return fail(error, "replica offsets must be integers");
          parsed.offsets.push_back(*number);
        }
        for (const auto &n : *extents) {
          auto number = n.getAsInteger();
          if (!number)
            return fail(error, "replica extents must be integers");
          parsed.extents.push_back(*number);
        }
        variant.workPartitions.push_back(std::move(parsed));
      }
    } else if (v->get("work_partitions"))
      return fail(error, "work_partitions must be an array");
    if (!validateWorkPartitions(source, impl, variant, error))
      return false;
    if (auto *unknown = v->getArray("unproved_constraints")) {
      for (const auto &value : *unknown) {
        auto reason = value.getAsString();
        if (!reason || reason->empty())
          return fail(error,
                      "unproved constraint must have a concrete diagnostic");
        variant.unprovedConstraints.push_back(reason->str());
      }
    } else if (v->get("unproved_constraints"))
      return fail(error, "unproved_constraints must be an array");
    if (object->get("storage") && !object->getObject("storage"))
      return fail(error, "storage must be a physical registration object");
    if (auto *storage = object->getObject("storage")) {
      if (storage->get("prepared_weight_copies") &&
          !storage->getArray("prepared_weight_copies"))
        return fail(error, "prepared_weight_copies must be an array");
      if (auto *copies = storage->getArray("prepared_weight_copies")) {
        std::map<size_t, ProgramOperatorPackedInput> packed;
        auto numbers = [](const j::Object &o, llvm::StringRef key,
                          std::vector<int64_t> &out) {
          auto *values = o.getArray(key);
          if (!values)
            return false;
          for (const auto &value : *values) {
            auto number = value.getAsInteger();
            if (!number)
              return false;
            out.push_back(*number);
          }
          return true;
        };
        for (const auto &copy : *copies) {
          auto *f = copy.getAsObject();
          auto sourceName = f ? f->getString("source_value") : std::nullopt;
          auto allocation = f ? f->getString("allocation_id") : std::nullopt;
          auto memory = f ? f->getString("memory") : std::nullopt;
          auto dtype = f ? f->getString("dtype") : std::nullopt;
          auto bytes = f ? f->getInteger("bytes") : std::nullopt;
          auto replaces =
              f ? f->getBoolean("replaces_source_physical_copy") : std::nullopt;
          auto shared =
              f ? f->getBoolean("host_device_same_allocation") : std::nullopt;
          if (!sourceName || !allocation || !memory || !dtype || !bytes ||
              !replaces || !*replaces || !shared)
            return fail(error, "packed weight replacement lacks explicit "
                               "physical identity/ownership");
          auto value =
              std::find_if(source.values.begin(), source.values.end(),
                           [&](const auto &v) { return v.id == *sourceName; });
          if (value == source.values.end())
            return fail(error, "packed source value does not exist");
          size_t index = size_t(value - source.values.begin());
          auto &input = packed[index];
          input.sourceValue = index;
          input.packingIdentity =
              staticFingerprint(*storage->get("prepared_weight_copies"));
          ProgramOperatorPackedFragment fragment;
          fragment.allocationId = allocation->str();
          fragment.memory = memory->str();
          fragment.dtype = dtype->str();
          fragment.bytes = *bytes;
          if (!numbers(*f, "source_offsets", fragment.sourceOffsets) ||
              !numbers(*f, "source_extents", fragment.sourceExtents) ||
              !numbers(*f, "physical_shape", fragment.physicalShape) ||
              !numbers(*f, "physical_offsets", fragment.physicalOffsets) ||
              !numbers(*f, "strides", fragment.strides))
            return fail(
                error, "packed fragment has invalid integer coordinate arrays");
          input.fragments.push_back(std::move(fragment));
        }
        for (auto &input : packed) {
          if (!validatePacking(source, impl, input.second, error))
            return false;
          variant.packedInputs.push_back(std::move(input.second));
        }
      }
    }
    if (!registrations.emplace(impl.id, std::move(variant)).second)
      return fail(error, "duplicate execution variant registration");
  }
  if (registrations.size() != implementations.size())
    return fail(error,
                "execution registration does not cover complete catalogue");
  return true;
}
namespace {
struct AllocationReservation {
  size_t identity = 0, memory = 0;
  int64_t bytes = 0;
  std::optional<size_t> sourceRoot;
};
struct SerialChoiceSummary {
  double wallMs = 0;
  bool valid = true, feasibilityKnown = true;
  std::string error;
  std::vector<AllocationReservation> allocations;
  std::vector<int64_t> temporaryMax;
  std::vector<std::string> unproved;
};
} // namespace
namespace orbit::taskflow {
struct ProgramOperatorMappingDomain {
  std::string identity, registrationIdentity;
  StaticConfig target;
  std::vector<ProgramOperatorImplementation> implementations;
  std::vector<ProgramOperatorExecutionVariant> variants;
  std::vector<ProgramOperatorChoiceGroup> groups;
  std::vector<std::vector<MappingAllocation>> boundaryAllocations;
  std::vector<std::vector<SerialChoiceSummary>> serialChoices;
  std::vector<std::string> allocationIdentities;
  std::map<std::string, size_t> memoryIndices;
};
} // namespace orbit::taskflow
bool orbit::taskflow::prepareProgramOperatorMappingDomain(
    const ProgramOperatorSource &source,
    const std::vector<ProgramOperatorImplementation> &implementations,
    const std::vector<ProgramOperatorChoiceGroup> &groups,
    const std::map<std::string, ProgramOperatorExecutionVariant> &variants,
    llvm::StringRef registrationIdentity, const StaticConfig &target,
    llvm::StringRef hostMemory,
    std::shared_ptr<const ProgramOperatorMappingDomain> &result,
    std::string &error) {
  j::Value certificate = nullptr;
  if (registrationIdentity.empty() || !programMemory(target, hostMemory) ||
      !validateMemoryArchitecture(target, error) ||
      !certifyProgramOperatorChoices(source, implementations, groups,
                                     certificate, error))
    return fail(error, "cannot certify mapping domain: " + error);
  auto domain = std::make_shared<ProgramOperatorMappingDomain>();
  domain->registrationIdentity = registrationIdentity.str();
  domain->target.domains = target.domains;
  domain->target.resources = target.resources;
  domain->target.memories = target.memories;
  domain->target.links = target.links;
  domain->implementations = implementations;
  domain->groups = groups;
  std::vector<size_t> roots(source.values.size());
  std::vector<int64_t> bytes(source.values.size());
  for (size_t v = 0; v < source.values.size(); ++v) {
    const auto &value = source.values[v];
    roots[v] =
        value.view
            ? roots[value.view->base]
            : (value.allocationView
                   ? roots[value.allocationView->base]
                   : (value.stateAliasOf ? roots[*value.stateAliasOf] : v));
    int64_t size =
        value.dtype == "f32" || value.dtype == "i32"
            ? 4
            : (value.dtype == "bf16" || value.dtype == "f16" ||
                       value.dtype == "i16"
                   ? 2
                   : (value.dtype == "i64" || value.dtype == "f64"
                          ? 8
                          : (value.dtype == "i8" || value.dtype == "ui8" ? 1
                                                                         : 0)));
    for (auto n : value.shape) {
      if (n <= 0 || !size || size > INT64_MAX / n)
        return fail(error, "source allocation byte overflow");
      size *= n;
    }
    bytes[v] = size;
  }
  ProgramOperatorMapping registration;
  registration.registrationIdentity = registrationIdentity.str();
  for (const auto &impl : implementations) {
    auto variant = variants.find(impl.id);
    const auto *resource = programResource(target, impl.resource);
    if (variant == variants.end() || !resource || resource->kind != "compute" ||
        resource->device != impl.device)
      return fail(error,
                  "missing registered variant or incompatible target resource");
    if (!validateWorkPartitions(source, impl, variant->second, error))
      return false;
    domain->variants.push_back(variant->second);
    registration.variants.push_back(variant->second);
    std::map<size_t, MappingAllocation> boundaries;
    auto port = [&](size_t v, bool input) {
      auto root = roots[v];
      const auto &locations =
          input ? impl.inputLocations : impl.outputLocations;
      auto location = locations.find(v);
      auto memory =
          location == locations.end() ? impl.memory : location->second;
      if (!programMemory(target, memory))
        return false;
      const auto &value = source.values[root];
      if ((value.role == "input" || value.role == "workspace") &&
          memory != (value.initialMemory.empty() ? hostMemory.str()
                                                 : value.initialMemory))
        return false;
      MappingAllocation allocation;
      allocation.storage = root;
      allocation.key = value.id;
      allocation.memory = memory;
      allocation.bytes = bytes[root];
      allocation.addressing = "external-abi";
      auto inserted = boundaries.emplace(root, std::move(allocation));
      return inserted.second || inserted.first->second.memory == memory;
    };
    for (size_t v : impl.inputs)
      if (!port(v, true))
        return fail(error,
                    "registered input aliases incompatible storage location");
    for (size_t v : impl.outputs)
      if (!port(v, false))
        return fail(error,
                    "registered output aliases incompatible storage location");
    std::map<size_t, const ProgramOperatorPackedInput *> packed;
    for (const auto &packing : variant->second.packedInputs) {
      if (!validatePacking(source, impl, packing, error) ||
          !packed.emplace(packing.sourceValue, &packing).second)
        return fail(error,
                    "invalid or duplicate packed input registration: " + error);
    }
    std::vector<MappingAllocation> allocations;
    for (auto &entry : boundaries) {
      auto replacement = packed.find(entry.first);
      if (replacement == packed.end()) {
        allocations.push_back(std::move(entry.second));
        continue;
      }
      for (const auto &f : replacement->second->fragments) {
        if (f.memory != entry.second.memory)
          return fail(
              error, "packing requires an undeclared source memory transition");
        auto allocation = entry.second;
        allocation.key = f.allocationId;
        allocation.bytes = f.bytes;
        allocation.addressing = "external-packed-input";
        allocations.push_back(std::move(allocation));
      }
    }
    domain->boundaryAllocations.push_back(std::move(allocations));
  }
  ProgramOperatorSelection implementationCosts;
  implementationCosts.regions = implementations;
  j::Array groupBindings;
  for (const auto &group : groups) {
    j::Array choices;
    for (size_t c = 0; c < group.choices.size(); ++c) {
      j::Array indices;
      for (size_t i : group.choices[c])
        indices.push_back(int64_t(i));
      choices.push_back(j::Object{{"id", group.choiceIds[c]},
                                  {"implementations", std::move(indices)}});
    }
    groupBindings.push_back(
        j::Object{{"id", group.id}, {"choices", std::move(choices)}});
  }
  domain->identity = staticFingerprint(
      j::Object{{"source_id", source.sourceId},
                {"host_memory", hostMemory.str()},
                {"source_abi_certificate", std::move(certificate)},
                {"registration", serializeProgramOperatorMapping(registration)},
                {"implementation_costs",
                 serializeProgramOperatorSelection(implementationCosts)},
                {"choice_bindings", std::move(groupBindings)},
                {"architecture", serializeMemoryArchitecture(target)}});
  // Source coverage and ABI have already been certified. The runtime contract
  // is synchronous, so resource-calendar legality is exactly the conjunction
  // of local invocation reservations; live storage remains a JOINT condition.
  // Cache those invariant proofs, not complete-plan prices or feasibility.
  for (size_t m = 0; m < domain->target.memories.size(); ++m)
    domain->memoryIndices.emplace(domain->target.memories[m].id, m);
  std::vector<std::string> invocationErrors(implementations.size());
  std::vector<bool> invocationKnown(implementations.size(), true);
  for (size_t i = 0; i < implementations.size(); ++i) {
    const auto &impl = implementations[i];
    const auto &v = domain->variants[i];
    std::string invalid;
    if (v.implementationId != impl.id ||
        v.artifactIdentity != impl.artifactIdentity ||
        v.entrypoint != impl.entrypoint || v.runtimeContextIdentity.empty() ||
        v.dataflow.empty() || v.residencePolicy.empty() || v.replicas < 1 ||
        (v.replicas == 1 ? v.workDistribution != "single-invocation"
                         : v.workDistribution != "registered-disjoint-work") ||
        v.resourceDemands.empty() || !v.resourceDemands.count(impl.resource))
      invalid = "variant identity/context/replica/resource contract mismatch";
    for (auto axis : v.resourceShape)
      if (axis <= 0)
        invalid = "nonpositive registered spatial shape";
    std::map<std::string, int64_t> expanded;
    if (invalid.empty() &&
        !programResourceDemands(domain->target, v.resourceDemands, expanded,
                                invalid)) {
    }
    if (invalid.empty())
      for (const auto &d : expanded) {
        auto *r = programResource(domain->target, d.first);
        if (!r || d.second > r->capacity)
          invalid =
              "exclusive invocation exceeds resource capacity: " + d.first;
      }
    if (!impl.wallMs || !std::isfinite(*impl.wallMs) || *impl.wallMs < 0) {
      invalid = "unknown/nonfinite inclusive duration: " + impl.id;
      invocationKnown[i] = false;
    }
    for (const auto &t : v.temporaryBytes)
      if (t.second < 0 || !domain->memoryIndices.count(t.first))
        invalid = "unknown memory or negative temporary reservation";
    for (const auto &r : v.residentAllocations)
      if (r.first.empty() || r.second.second <= 0 ||
          !domain->memoryIndices.count(r.second.first))
        invalid = "invalid resident allocation";
    invocationErrors[i] = std::move(invalid);
  }
  std::map<std::string, size_t> allocationIndices;
  for (const auto &group : groups) {
    std::vector<SerialChoiceSummary> choices;
    for (const auto &choice : group.choices) {
      SerialChoiceSummary summary;
      summary.temporaryMax.resize(domain->target.memories.size());
      std::map<std::string, AllocationReservation> reservations;
      auto reserveIdentity = [&](const std::string &id,
                                 const std::string &memory, int64_t bytes,
                                 std::optional<size_t> root) {
        auto m = domain->memoryIndices.find(memory);
        if (id.empty() || m == domain->memoryIndices.end() || bytes <= 0) {
          summary.valid = false;
          summary.error = "invalid physical allocation reservation";
          return;
        }
        auto insertedId =
            allocationIndices.emplace(id, allocationIndices.size());
        if (insertedId.second)
          domain->allocationIdentities.push_back(id);
        AllocationReservation reservation{insertedId.first->second, m->second,
                                          bytes, root};
        auto inserted = reservations.emplace(id, reservation);
        if (!inserted.second) {
          auto &prior = inserted.first->second;
          if (prior.memory != reservation.memory ||
              prior.bytes != reservation.bytes ||
              (prior.sourceRoot && root && prior.sourceRoot != root)) {
            summary.valid = false;
            summary.error =
                "physical allocation alias changes size/location/storage";
          }
          if (root)
            prior.sourceRoot = root;
        }
      };
      for (size_t index : choice) {
        if (!invocationErrors[index].empty()) {
          summary.valid = false;
          summary.error = invocationErrors[index];
          summary.feasibilityKnown &= invocationKnown[index];
        }
        if (implementations[index].wallMs)
          summary.wallMs += *implementations[index].wallMs;
        for (const auto &a : domain->boundaryAllocations[index])
          reserveIdentity(a.key, a.memory, a.bytes, a.storage);
        const auto &v = domain->variants[index];
        for (const auto &r : v.residentAllocations)
          reserveIdentity(r.first, r.second.first, r.second.second,
                          std::nullopt);
        for (const auto &t : v.temporaryBytes) {
          auto m = domain->memoryIndices.find(t.first);
          if (m != domain->memoryIndices.end())
            summary.temporaryMax[m->second] =
                std::max(summary.temporaryMax[m->second], t.second);
        }
        for (const auto &reason : v.unprovedConstraints)
          summary.unproved.push_back(v.implementationId + ": " + reason);
      }
      if (!std::isfinite(summary.wallMs)) {
        summary.valid = false;
        summary.error = "schedule time overflow";
      }
      for (const auto &reservation : reservations)
        summary.allocations.push_back(reservation.second);
      choices.push_back(std::move(summary));
    }
    domain->serialChoices.push_back(std::move(choices));
  }
  result = std::move(domain);
  return true;
}
ProgramSearchEvaluation
orbit::taskflow::evaluateProgramOperatorMappingDomainReference(
    const ProgramOperatorMappingDomain &domain,
    const std::vector<size_t> &groupChoices) {
  ProgramSearchEvaluation result;
  auto reject = [&](const std::string &message) {
    result.message = "certified mapping projection: " + message;
    return result;
  };
  if (groupChoices.size() != domain.groups.size())
    return reject("group choice count mismatch");
  std::vector<size_t> selected;
  for (size_t g = 0; g < groupChoices.size(); ++g) {
    if (groupChoices[g] >= domain.groups[g].choices.size())
      return reject("group choice out of bounds");
    const auto &choice = domain.groups[g].choices[groupChoices[g]];
    selected.insert(selected.end(), choice.begin(), choice.end());
  }
  std::sort(selected.begin(), selected.end(), [&](size_t a, size_t b) {
    return domain.implementations[a].operators.front() <
           domain.implementations[b].operators.front();
  });
  ProgramOperatorSelection selection;
  ProgramOperatorMapping mapping;
  mapping.registrationIdentity = domain.registrationIdentity;
  std::vector<MappingEvent> events;
  std::map<std::string, MappingAllocation> roots;
  for (size_t index : selected) {
    const auto &registered = domain.implementations[index];
    // This projection retains only invocation identity and timing. Source
    // coverage has already been certified and is immutable in this domain.
    // It is never serialized or admitted as an executable ProgramGraph.
    ProgramOperatorImplementation impl;
    impl.id = registered.id;
    impl.artifactIdentity = registered.artifactIdentity;
    impl.entrypoint = registered.entrypoint;
    impl.resource = registered.resource;
    impl.wallMs = registered.wallMs;
    selection.regions.push_back(std::move(impl));
    mapping.readyPriority.push_back(events.size());
    mapping.variants.push_back(domain.variants[index]);
    MappingEvent event;
    event.kind = "implementation";
    event.resource = selection.regions.back().resource;
    if (!events.empty())
      event.dependencies = {events.size() - 1};
    events.push_back(std::move(event));
    for (const auto &allocation : domain.boundaryAllocations[index]) {
      auto inserted = roots.emplace(allocation.key, allocation);
      if (!inserted.second &&
          (inserted.first->second.storage != allocation.storage ||
           inserted.first->second.bytes != allocation.bytes ||
           inserted.first->second.memory != allocation.memory))
        return reject("alias storage requires an unimplemented transfer");
    }
  }
  std::vector<MappingAllocation> allocations;
  for (auto &root : roots)
    allocations.push_back(std::move(root.second));
  MappingGraphSchedule schedule;
  std::string error;
  if (!build(ScheduleView(selection, events, allocations, domain.target),
             mapping, schedule, error)) {
    result.message = error;
    result.feasibilityKnown =
        error.find("unknown/nonfinite inclusive duration") == std::string::npos;
    return result;
  }
  result.feasible = true;
  result.costMs = schedule.estimatedMakespanMs;
  j::Array unproved;
  std::set<std::string> usedMemories;
  for (const auto &a : allocations)
    usedMemories.insert(a.memory);
  for (const auto &v : mapping.variants) {
    for (const auto &reason : v.unprovedConstraints) {
      result.feasibilityKnown = false;
      unproved.push_back(v.implementationId + ": " + reason);
    }
    for (const auto &a : v.temporaryBytes)
      usedMemories.insert(a.first);
    for (const auto &a : v.residentAllocations)
      usedMemories.insert(a.second.first);
  }
  for (const auto &memory : domain.target.memories)
    if (!memory.capacityKnown && usedMemories.count(memory.id)) {
      result.feasibilityKnown = false;
      unproved.push_back("physical capacity unavailable: " + memory.id);
    }
  result.result = j::Object{
      {"mapping_domain", domain.identity},
      {"resource_calendar_checked", true},
      {"source_abi_certified", true},
      {"full_physical_materialization_required", true},
      {"unproved_target_constraints", std::move(unproved)},
      {"runtime_dependencies", "actual synchronous invocation order"}};
  result.message =
      "certified invocation projection plus joint resource/resident/scratch "
      "calendar; final physical graph still required";
  return result;
}
namespace {
bool applyPacking(ProgramGraph &graph, const ProgramOperatorMapping &mapping,
                  std::string &error) {
  if (!graph.operatorSource || !graph.operatorSelection ||
      mapping.variants.size() != graph.events.size())
    return fail(error,
                "packing needs a complete source-bound invocation selection");
  const auto &source = *graph.operatorSource;
  std::map<size_t, const ProgramOperatorPackedInput *> replacements;
  std::set<size_t> removedAllocations;
  for (size_t event = 0; event < mapping.variants.size(); ++event) {
    for (const auto &packing : mapping.variants[event].packedInputs) {
      if (!validatePacking(source, graph.operatorSelection->regions[event],
                           packing, error))
        return false;
      size_t copy = SIZE_MAX;
      for (size_t c = 0; c < graph.copies.size(); ++c)
        if (graph.regions[graph.versions[graph.copies[c].version].region]
                .storage == packing.sourceValue) {
          if (copy != SIZE_MAX)
            return fail(error, "packing root has multiple ABI aliases");
          copy = c;
        }
      if (copy == SIZE_MAX || graph.copies[copy].createdBy != -1 ||
          graph.copies[copy].users != std::vector<size_t>{event} ||
          !replacements.emplace(copy, &packing).second)
        return fail(error, "packed input is absent, written, or used by an "
                           "unregistered consumer");
      auto allocation = graph.copies[copy].allocation;
      for (size_t c = 0; c < graph.copies.size(); ++c)
        if (c != copy && graph.copies[c].allocation == allocation)
          return fail(error,
                      "cannot replace a physical input with unhandled aliases");
      removedAllocations.insert(allocation);
    }
  }
  if (replacements.empty())
    return true;
  std::vector<MappingAllocation> allocations;
  std::vector<size_t> allocationMap(graph.allocations.size(), SIZE_MAX);
  std::set<std::string> allocationIds;
  for (size_t a = 0; a < graph.allocations.size(); ++a)
    if (!removedAllocations.count(a)) {
      allocationMap[a] = allocations.size();
      allocations.push_back(graph.allocations[a]);
      allocationIds.insert(graph.allocations[a].key);
    }
  std::vector<MappingRegion> regions;
  std::vector<MappingVersion> versions;
  std::vector<MappingCopy> copies;
  std::vector<std::vector<size_t>> copyMap(graph.copies.size());
  auto append = [&](size_t old, MappingRegion region, size_t allocation,
                    const std::string &layout) {
    size_t index = copies.size();
    copyMap[old].push_back(index);
    auto version = graph.versions[graph.copies[old].version];
    version.region = index;
    auto copy = graph.copies[old];
    copy.version = index;
    copy.allocation = allocation;
    copy.layout = layout;
    copy.users.clear();
    copy.consumerEdges.clear();
    copy.lastUsers.clear();
    regions.push_back(std::move(region));
    versions.push_back(std::move(version));
    copies.push_back(std::move(copy));
  };
  for (size_t c = 0; c < graph.copies.size(); ++c) {
    auto replacement = replacements.find(c);
    if (replacement == replacements.end()) {
      append(c, graph.regions[graph.versions[graph.copies[c].version].region],
             allocationMap[graph.copies[c].allocation], graph.copies[c].layout);
      continue;
    }
    const auto &packing = *replacement->second;
    for (const auto &f : packing.fragments) {
      if (!allocationIds.insert(f.allocationId).second)
        return fail(error, "packed physical allocation identity is not unique");
      auto allocation = graph.allocations[graph.copies[c].allocation];
      if (f.memory != allocation.memory)
        return fail(error,
                    "packed input requires an undeclared memory transfer");
      allocation.endPhase = graph.events.size();
      allocation.key = f.allocationId;
      allocation.memory = f.memory;
      allocation.bytes = f.bytes;
      allocation.addressing = "external-packed-input";
      size_t index = allocations.size();
      allocations.push_back(std::move(allocation));
      append(c, {packing.sourceValue, f.sourceOffsets, f.sourceExtents}, index,
             "registered-packing:" + packing.packingIdentity);
      copies.back().releasedAfter = graph.events.size();
    }
  }
  auto expand = [&](const std::vector<size_t> &old) {
    std::vector<size_t> expanded;
    for (size_t c : old) {
      if (c >= copyMap.size())
        return std::vector<size_t>{};
      expanded.insert(expanded.end(), copyMap[c].begin(), copyMap[c].end());
    }
    return expanded;
  };
  for (auto &event : graph.events) {
    event.reads = expand(event.reads);
    event.writes = expand(event.writes);
    event.dataEdges.clear();
  }
  std::vector<MappingDataEdge> edges;
  for (const auto &old : graph.dataEdges)
    for (size_t c : copyMap[old.copy]) {
      auto edge = old;
      edge.region = versions[copies[c].version].region;
      edge.version = copies[c].version;
      edge.copy = c;
      edge.requirement.layout = copies[c].layout;
      edge.requirement.memory = allocations[copies[c].allocation].memory;
      size_t index = edges.size();
      edges.push_back(std::move(edge));
      graph.events[old.consumer].dataEdges.push_back(index);
      copies[c].users.push_back(old.consumer);
      copies[c].consumerEdges.push_back(index);
    }
  for (auto &copy : copies)
    if (!copy.users.empty())
      copy.lastUsers = {copy.users.back()};
  for (auto &region : graph.executionRegions) {
    region.inputCopies = expand(region.inputCopies);
    region.outputCopies = expand(region.outputCopies);
  }
  for (auto &event : graph.regionEvents)
    event.copies = expand(event.copies);
  graph.allocations = std::move(allocations);
  graph.regions = std::move(regions);
  graph.versions = std::move(versions);
  graph.copies = std::move(copies);
  graph.dataEdges = std::move(edges);
  return true;
}
} // namespace
bool orbit::taskflow::hasProgramOperatorPackedInputs(
    const ProgramOperatorMapping &mapping) {
  for (const auto &variant : mapping.variants)
    if (!variant.packedInputs.empty())
      return true;
  return false;
}
bool orbit::taskflow::checkProgramOperatorPackedGraph(const ProgramGraph &graph,
                                                      std::string &error) {
  if (!graph.operatorSource || !graph.operatorSelection ||
      !graph.operatorMapping || graph.operatorMapping->hostMemory.empty())
    return fail(error,
                "packed graph is missing source/selection/host contract");
  StaticConfig target;
  target.domains = graph.domains;
  target.resources = graph.resources;
  target.memories = graph.memories;
  target.links = graph.links;
  ProgramGraph expected;
  // Rebuild semantic ABI from the actual immutable source, then apply the
  // checked fragment transformation. No nominal old weight BO survives in the
  // resulting physical graph and no aggregate/discontiguous BO is invented.
  if (!materializeProgramOperatorGraph(
          graph.operatorSource, *graph.operatorSelection, target,
          graph.operatorMapping->hostMemory, expected, error) ||
      !applyPacking(expected, *graph.operatorMapping, error))
    return false;
  for (size_t i = 0; i < graph.operatorMapping->variants.size(); ++i)
    if (!validateWorkPartitions(*graph.operatorSource,
                                graph.operatorSelection->regions[i],
                                graph.operatorMapping->variants[i], error))
      return false;
  expected.operatorMapping = graph.operatorMapping;
  if (!build(ScheduleView(expected), *graph.operatorMapping, expected.schedule,
             error))
    return false;
  expected.estimatedMakespanMs = expected.schedule.estimatedMakespanMs;
  if (programGraphIdentity(expected) != programGraphIdentity(graph))
    return fail(error, "packed physical graph differs from source-derived "
                       "fragment/copy/lifetime/schedule witness");
  return true;
}
ProgramSearchEvaluation orbit::taskflow::evaluateProgramOperatorMappingDomain(
    const ProgramOperatorMappingDomain &domain,
    const std::vector<size_t> &groupChoices) {
  ProgramSearchEvaluation result;
  if (groupChoices.size() != domain.serialChoices.size()) {
    result.message = "certified serial calendar: group choice count mismatch";
    return result;
  }
  // The only per-assignment storage is indexed by immutable physical identity.
  // Aliasing and global capacity are NOT products of local feasibility: a
  // second choice may retain a distinct BO, or disagree on an existing one.
  const auto memories = domain.target.memories.size();
  std::vector<int64_t> persistent(memories), temporary(memories);
  std::vector<size_t> identityMemory(domain.allocationIdentities.size(),
                                     SIZE_MAX);
  std::vector<int64_t> identityBytes(domain.allocationIdentities.size());
  std::vector<size_t> identityRoot(domain.allocationIdentities.size(),
                                   SIZE_MAX);
  j::Array unproved;
  double wall = 0;
  auto reject = [&](const std::string &message) {
    result.message = "certified serial calendar: " + message;
    return result;
  };
  for (size_t g = 0; g < groupChoices.size(); ++g) {
    if (groupChoices[g] >= domain.serialChoices[g].size())
      return reject("group choice out of bounds");
    const auto &choice = domain.serialChoices[g][groupChoices[g]];
    if (!choice.valid) {
      result.feasibilityKnown = choice.feasibilityKnown;
      return reject(choice.error);
    }
    wall += choice.wallMs;
    for (const auto &a : choice.allocations) {
      if (identityMemory[a.identity] == SIZE_MAX) {
        identityMemory[a.identity] = a.memory;
        identityBytes[a.identity] = a.bytes;
        if (!add(persistent[a.memory], a.bytes))
          return reject("joint allocation byte accounting overflow");
      } else if (identityMemory[a.identity] != a.memory ||
                 identityBytes[a.identity] != a.bytes ||
                 (a.sourceRoot && identityRoot[a.identity] != SIZE_MAX &&
                  identityRoot[a.identity] != *a.sourceRoot))
        return reject(
            "joint physical identity changes size/location/source root");
      if (a.sourceRoot)
        identityRoot[a.identity] = *a.sourceRoot;
    }
    for (size_t m = 0; m < memories; ++m)
      temporary[m] = std::max(temporary[m], choice.temporaryMax[m]);
    for (const auto &reason : choice.unproved) {
      result.feasibilityKnown = false;
      unproved.push_back(reason);
    }
  }
  if (!std::isfinite(wall))
    return reject("schedule time overflow");
  for (size_t m = 0; m < memories; ++m) {
    int64_t peak = persistent[m];
    if (!add(peak, temporary[m]))
      return reject("joint persistent/temporary reservation overflow");
    const auto &memory = domain.target.memories[m];
    if (memory.capacityKnown && peak > memory.capacityBytes)
      return reject("joint resident/ABI/scratch exceeds capacity: " +
                    memory.id);
    if (!memory.capacityKnown && peak) {
      result.feasibilityKnown = false;
      unproved.push_back("physical capacity unavailable: " + memory.id);
    }
  }
  result.feasible = true;
  result.costMs = wall;
  result.message = "immutable source/ABI/local-resource proofs plus exact "
                   "synchronous calendar and joint physical allocation unions";
  result.result = j::Object{
      {"mapping_domain", domain.identity},
      {"resource_calendar_checked", true},
      {"source_abi_certified", true},
      {"full_physical_materialization_required", true},
      {"unproved_target_constraints", std::move(unproved)},
      {"runtime_dependencies", "actual synchronous invocation order"},
      {"evaluation",
       "exact immutable choice-calendar summaries; joint storage recomputed"}};
  return result;
}
