#include "Taskflow/ProgramOperatorMapping.h"
#include "Taskflow/JSONUtils.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <cmath>
#include <queue>
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
bool fitsReservation(const Calendar &calendar, const std::string &key,
                     double begin, double end, int64_t amount,
                     int64_t capacity) {
  if (amount < 0 || amount > capacity)
    return false;
  if (begin == end || amount == 0)
    return true;
  auto entry = calendar.find(key);
  if (entry == calendar.end() || entry->second.empty() ||
      begin >= entry->second.rbegin()->first)
    return true; // No previously scheduled interval remains live.
  __int128 live = 0;
  for (const auto &point : entry->second) {
    if (point.first >= end)
      break;
    live += point.second;
    if (point.first <= begin) {
      auto next = entry->second.upper_bound(point.first);
      if (next != entry->second.end() && next->first <= begin)
        continue;
    }
    if (live + amount > capacity)
      return false;
  }
  return true;
}
bool reserve(Calendar &calendar, const std::string &key, double begin,
             double end, int64_t amount, int64_t capacity, std::string &error) {
  if (!fitsReservation(calendar, key, begin, end, amount, capacity))
    return fail(error, "overlapping reservations exceed capacity: " + key);
  if (begin == end || !amount)
    return true;
  for (auto change :
       {std::make_pair(begin, amount), std::make_pair(end, -amount)}) {
    __int128 delta = __int128(calendar[key][change.first]) + change.second;
    if (delta < INT64_MIN || delta > INT64_MAX)
      return fail(error, "calendar delta overflow: " + key);
    if (!delta)
      calendar[key].erase(change.first);
    else
      calendar[key][change.first] = int64_t(delta);
  }
  return true;
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
  ProgramOperatorSelection policy;
  for (const auto &impl : g.operatorSelection->regions) {
    ProgramOperatorImplementation identity;
    identity.id = impl.id;
    identity.sourceId = impl.sourceId;
    policy.regions.push_back(std::move(identity));
  }
  if (!configureProgramOperatorExecutionPolicy(policy, mapping, error) ||
      policy.executionPolicy != g.operatorSelection->executionPolicy ||
      policy.sourceOrderingIdentity !=
          g.operatorSelection->sourceOrderingIdentity ||
      policy.asyncRuntimeProofIdentity !=
          g.operatorSelection->asyncRuntimeProofIdentity)
    return fail(error, "graph and registered execution policy disagree");
  double outsideControlMs;
  std::map<std::string, int64_t> outsideFeatures;
  if (!estimateProgramOperatorInvocationControl(
          *g.operatorSelection, mapping.controlModel, outsideControlMs,
          outsideFeatures, error))
    return false;
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
  std::vector<size_t> remaining(g.events.size());
  std::vector<std::vector<size_t>> users(g.events.size());
  std::priority_queue<std::pair<size_t, size_t>,
                      std::vector<std::pair<size_t, size_t>>,
                      std::greater<std::pair<size_t, size_t>>>
      ready;
  for (size_t i = 0; i < g.events.size(); ++i) {
    remaining[i] = g.events[i].dependencies.size();
    for (size_t p : g.events[i].dependencies) {
      if (p >= g.events.size() || p == i)
        return fail(error, "invalid invocation dependency");
      users[p].push_back(i);
    }
    if (!remaining[i])
      ready.emplace(rank[i], i);
  }
  double makespan = 0;
  for (size_t issued = 0; issued < g.events.size(); ++issued) {
    if (ready.empty())
      return fail(error, "cyclic implementation schedule");
    size_t chosen = ready.top().second;
    ready.pop();
    double earliest = 0;
    for (size_t p : g.events[chosen].dependencies)
      earliest = std::max(earliest, *schedule.events[p].endMs);
    // Every contention transition is a calendar endpoint, so scanning these
    // points finds the earliest feasible nonpreemptive reservation exactly.
    std::set<double> starts{earliest};
    // Only resources used by this invocation can delay it. A strict serial
    // suffix starts after the current makespan and needs no endpoint scan.
    if (earliest < makespan) {
      auto endpoints = [&](const Calendar &calendar, const auto &requirements) {
        for (const auto &requirement : requirements) {
          auto entry = calendar.find(requirement.first);
          if (entry != calendar.end())
            for (auto p = entry->second.lower_bound(earliest);
                 p != entry->second.end(); ++p)
              starts.insert(p->first);
        }
      };
      endpoints(resourceCalendar, demands[chosen]);
      endpoints(memoryCalendar, mapping.variants[chosen].temporaryBytes);
    }
    bool placed = false;
    for (double begin : starts) {
      double end = begin + schedule.eventDurationsMs[chosen];
      if (!std::isfinite(end))
        return fail(error, "schedule time overflow");
      std::string rejected;
      bool fits = true;
      for (const auto &d : demands[chosen]) {
        auto *r = programResource(target, d.first);
        fits &= r && fitsReservation(resourceCalendar, d.first, begin, end,
                                     d.second, r->capacity);
      }
      for (const auto &t : mapping.variants[chosen].temporaryBytes) {
        auto *m = programMemory(target, t.first);
        int64_t available = m->capacityKnown
                                ? m->capacityBytes - persistent[m->id]
                                : INT64_MAX - persistent[m->id];
        fits &= fitsReservation(memoryCalendar, t.first, begin, end, t.second,
                                available);
      }
      if (!fits)
        continue;
      // Probe first, commit once: rejected placements never mutate a calendar
      // and evaluating a candidate does not copy all prior reservations.
      for (const auto &d : demands[chosen])
        if (!reserve(resourceCalendar, d.first, begin, end, d.second,
                     programResource(target, d.first)->capacity, rejected))
          return fail(error, rejected);
      for (const auto &t : mapping.variants[chosen].temporaryBytes) {
        const auto *m = programMemory(target, t.first);
        int64_t available = m->capacityKnown
                                ? m->capacityBytes - persistent[m->id]
                                : INT64_MAX - persistent[m->id];
        if (!reserve(memoryCalendar, t.first, begin, end, t.second, available,
                     rejected))
          return fail(error, rejected);
      }
      schedule.events[chosen] = {chosen, g.events[chosen].dependencies, begin,
                                 end};
      makespan = std::max(makespan, end);
      placed = true;
      for (size_t user : users[chosen])
        if (!--remaining[user])
          ready.emplace(rank[user], user);
      break;
    }
    if (!placed)
      return fail(error, "no capacity-feasible invocation placement");
  }
  schedule.estimatedMakespanMs = makespan;
  return true;
}
} // namespace
bool orbit::taskflow::estimateProgramOperatorInvocationControl(
    const ProgramOperatorSelection &selection,
    const std::optional<ProgramOperatorInvocationControlModel> &model,
    double &controlMs, std::map<std::string, int64_t> &features,
    std::string &error) {
  controlMs = 0;
  features.clear();
  if (!model)
    return true;
  if (model->identity.empty() || model->sourceId.empty() ||
      model->boundary != "outside-inclusive-invocation-wall" ||
      !std::isfinite(model->interceptMs) || model->interceptMs < 0 ||
      model->coefficients.size() != 2 ||
      !model->coefficients.count("cpu_callbacks") ||
      !model->coefficients.count("npu_callbacks"))
    return fail(error, "invalid outside-invocation controller model contract");
  for (const auto &coefficient : model->coefficients) {
    if (!std::isfinite(coefficient.second) || coefficient.second < 0)
      return fail(error, "unknown/nonfinite/negative controller coefficient");
    features[coefficient.first] = 0;
  }
  for (const auto &impl : selection.regions) {
    if (impl.sourceId != model->sourceId ||
        impl.controllerFeatures.size() != model->coefficients.size())
      return fail(error, "controller model source or explicit callback "
                         "features are unavailable");
    for (const auto &feature : impl.controllerFeatures) {
      auto count = features.find(feature.first);
      if (count == features.end() || feature.second < 0 ||
          feature.second > INT64_MAX - count->second)
        return fail(error,
                    "unknown/negative/overflowing controller callback feature");
      count->second += feature.second;
    }
  }
  controlMs = model->interceptMs;
  for (const auto &feature : features)
    controlMs += model->coefficients.at(feature.first) * double(feature.second);
  if (!std::isfinite(controlMs))
    return fail(error, "controller prediction overflows finite wall time");
  return true;
}
namespace {
j::Value serializeControlModel(
    const std::optional<ProgramOperatorInvocationControlModel> &model) {
  if (!model)
    return nullptr;
  j::Object coefficients;
  for (const auto &c : model->coefficients)
    coefficients[c.first] = c.second;
  return j::Object{{"schema", "orbit.program-invocation-control-model.v1"},
                   {"identity", model->identity},
                   {"source_id", model->sourceId},
                   {"boundary", model->boundary},
                   {"intercept_ms", model->interceptMs},
                   {"coefficients", std::move(coefficients)}};
}
} // namespace
bool orbit::taskflow::parseProgramOperatorInvocationControlModel(
    const ProgramOperatorSource &source, const j::Value &catalogue,
    std::optional<ProgramOperatorInvocationControlModel> &model,
    std::string &error) {
  model.reset();
  const auto *root = catalogue.getAsObject();
  const auto *value = root ? root->get("invocation_control_model") : nullptr;
  if (!value)
    return true;
  const auto *object = value->getAsObject();
  if (!object ||
      object->getString("schema") !=
          "orbit.program-invocation-control-model.v1" ||
      object->getString("source_id") != source.sourceId ||
      !object->getObject("fit"))
    return fail(error, "invalid source-bound calibrated controller model");
  auto identity = object->getString("identity"),
       boundary = object->getString("boundary");
  auto intercept = object->getNumber("intercept_ms");
  const auto *coefficients = object->getObject("coefficients");
  if (!identity || identity->empty() || !boundary || !intercept ||
      !coefficients)
    return fail(error,
                "controller identity/boundary/coefficients are unavailable");
  ProgramOperatorInvocationControlModel parsed;
  parsed.identity = identity->str();
  parsed.sourceId = source.sourceId;
  parsed.boundary = boundary->str();
  parsed.interceptMs = *intercept;
  for (const auto &entry : *coefficients) {
    auto coefficient = entry.second.getAsNumber();
    if (!coefficient)
      return fail(error,
                  "controller coefficients must be explicit finite numbers");
    parsed.coefficients.emplace(entry.first.str(), *coefficient);
  }
  double ignored;
  std::map<std::string, int64_t> features;
  if (!estimateProgramOperatorInvocationControl({}, parsed, ignored, features,
                                                error))
    return false;
  model = std::move(parsed);
  return true;
}
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
  double controlMs;
  std::map<std::string, int64_t> controlFeatures;
  if (!estimateProgramOperatorInvocationControl(*graph.operatorSelection,
                                                mapping.controlModel, controlMs,
                                                controlFeatures, error))
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
  double controlMs;
  std::map<std::string, int64_t> controlFeatures;
  if (!estimateProgramOperatorInvocationControl(
          *graph.operatorSelection, graph.operatorMapping->controlModel,
          controlMs, controlFeatures, error))
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
                  {"execution_policy", v.executionPolicy},
                  {"source_ordering_identity", v.sourceOrderingIdentity},
                  {"async_runtime_proof_identity", v.asyncRuntimeProofIdentity},
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
                   {"control_model", serializeControlModel(m.controlModel)},
                   {"variants", std::move(variants)},
                   {"ready_priority", std::move(priority)}};
}
bool orbit::taskflow::configureProgramOperatorExecutionPolicy(
    ProgramOperatorSelection &selection, const ProgramOperatorMapping &mapping,
    std::string &error) {
  if (selection.regions.size() != mapping.variants.size() ||
      selection.regions.empty())
    return fail(error, "execution policy needs every registered invocation");
  const auto &first = mapping.variants.front();
  for (size_t i = 0; i < mapping.variants.size(); ++i) {
    const auto &v = mapping.variants[i];
    if (v.executionPolicy != first.executionPolicy ||
        v.sourceOrderingIdentity != first.sourceOrderingIdentity ||
        v.asyncRuntimeProofIdentity != first.asyncRuntimeProofIdentity ||
        v.implementationId != selection.regions[i].id)
      return fail(error, "mixed or mismatched runtime execution policies");
    if (v.executionPolicy == "source-dependency-ready") {
      if (v.sourceOrderingIdentity != selection.regions[i].sourceId ||
          v.asyncRuntimeProofIdentity.empty())
        return fail(error, "async runtime lacks source-bound support proof");
    } else if (v.executionPolicy != "synchronous-source-order" ||
               !v.sourceOrderingIdentity.empty() ||
               !v.asyncRuntimeProofIdentity.empty())
      return fail(error, "invalid runtime execution policy");
  }
  selection.executionPolicy = first.executionPolicy;
  selection.sourceOrderingIdentity = first.sourceOrderingIdentity;
  selection.asyncRuntimeProofIdentity = first.asyncRuntimeProofIdentity;
  return true;
}
bool orbit::taskflow::canonicalizeProgramOperatorCandidate(
    const ProgramOperatorSource &source,
    ProgramOperatorMappingCandidate &candidate, std::string &error) {
  if (!configureProgramOperatorExecutionPolicy(candidate.selection,
                                               candidate.mapping, error))
    return false;
  if (candidate.selection.executionPolicy == "synchronous-source-order")
    return true;
  std::vector<std::vector<size_t>> dependencies;
  if (!programOperatorInvocationDependencies(source, candidate.selection,
                                             dependencies, error))
    return false;
  const auto n = dependencies.size();
  if (candidate.mapping.readyPriority.size() != n)
    return fail(error,
                "canonicalization needs a complete priority permutation");
  std::vector<size_t> indegree(n), order, inverse(n), rank(n, SIZE_MAX);
  std::vector<std::vector<size_t>> users(n);
  std::set<size_t> ready;
  for (size_t i = 0; i < n; ++i) {
    auto p = candidate.mapping.readyPriority[i];
    if (p >= n || rank[p] != SIZE_MAX)
      return fail(error, "canonicalization priority is not a permutation");
    rank[p] = i;
    indegree[i] = dependencies[i].size();
    for (size_t p : dependencies[i])
      users[p].push_back(i);
    if (!indegree[i])
      ready.insert(i);
  }
  while (!ready.empty()) {
    size_t i = *ready.begin();
    ready.erase(ready.begin());
    inverse[i] = order.size();
    order.push_back(i);
    for (size_t u : users[i])
      if (!--indegree[u])
        ready.insert(u);
  }
  if (order.size() != n)
    return fail(error, "cyclic source implementation quotient");
  auto selection = candidate.selection;
  auto variants = candidate.mapping.variants;
  for (size_t i = 0; i < n; ++i) {
    candidate.selection.regions[i] = selection.regions[order[i]];
    candidate.mapping.variants[i] = variants[order[i]];
    candidate.mapping.readyPriority[rank[i]] = inverse[i];
  }
  return true;
}
bool orbit::taskflow::setProgramOperatorReadyPriority(
    const ProgramOperatorSelection &selection, ProgramOperatorMapping &mapping,
    llvm::StringRef policy, std::string &error) {
  if (policy != "source-order" && policy != "reverse-source-order" &&
      policy != "cpu-first" && policy != "npu-first" &&
      policy != "longest-first")
    return fail(error, "unknown registered ready-priority policy");
  mapping.readyPriority.resize(selection.regions.size());
  for (size_t i = 0; i < selection.regions.size(); ++i)
    mapping.readyPriority[i] = i;
  if (policy == "reverse-source-order")
    std::reverse(mapping.readyPriority.begin(), mapping.readyPriority.end());
  else if (policy != "source-order")
    std::stable_sort(mapping.readyPriority.begin(), mapping.readyPriority.end(),
                     [&](size_t a, size_t b) {
                       const auto &x = selection.regions[a],
                                  &y = selection.regions[b];
                       if (policy == "longest-first")
                         return x.wallMs.value_or(0) > y.wallMs.value_or(0);
                       const auto device =
                           policy == "cpu-first" ? "cpu" : "npu";
                       return (x.device == device) > (y.device == device);
                     });
  return true;
}
std::vector<ProgramOperatorMappingAction>
orbit::taskflow::programOperatorReadyPriorityNeighbors(
    const ProgramOperatorMappingCandidate &candidate) {
  std::vector<ProgramOperatorMappingAction> result;
  if (candidate.selection.executionPolicy != "source-dependency-ready")
    return result;
  // Adjacent transpositions generate every finite priority permutation. A
  // priority is only a tie breaker among semantically ready invocations.
  for (size_t i = 1; i < candidate.mapping.readyPriority.size(); ++i) {
    ProgramOperatorMappingAction action;
    action.kind = ProgramOperatorMappingActionKind::ReadyPriority;
    action.readyPriority = candidate.mapping.readyPriority;
    std::swap(action.readyPriority[i - 1], action.readyPriority[i]);
    result.push_back(std::move(action));
  }
  return result;
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
bool orbit::taskflow::applyProgramOperatorRegisteredMappingAction(
    const ProgramOperatorMappingCandidate &parent,
    const ProgramOperatorMappingAction &action,
    const std::vector<ProgramOperatorImplementation> &implementations,
    const std::map<std::string, ProgramOperatorExecutionVariant> &variants,
    ProgramOperatorMappingCandidate &child, std::string &error) {
  if (action.kind != ProgramOperatorMappingActionKind::IndependentReplica &&
      action.kind !=
          ProgramOperatorMappingActionKind::IndependentResourceShape &&
      action.kind != ProgramOperatorMappingActionKind::IndependentResidence)
    return fail(
        error,
        "registered coordinate resolver supports replica/shape/residence only; "
        "transfer routes require an executable copy ABI");
  if (!action.regionCount ||
      action.firstRegion >= parent.selection.regions.size() ||
      action.regionCount >
          parent.selection.regions.size() - action.firstRegion ||
      action.replacements.empty() ||
      parent.mapping.variants.size() != parent.selection.regions.size())
    return fail(error, "invalid registered coordinate replacement range");
  ProgramOperatorMappingAction resolved = action;
  resolved.kind = ProgramOperatorMappingActionKind::ReplaceExecutableCover;
  resolved.replacements.clear();
  resolved.replacementVariants.clear();
  auto coordinate = [&](const ProgramOperatorExecutionVariant &v) {
    ProgramOperatorMapping single;
    single.variants = {v};
    auto value = serializeProgramOperatorMapping(single);
    const auto &serialized =
        *value.getAsObject()->getArray("variants")->front().getAsObject();
    j::Object result;
    if (action.kind == ProgramOperatorMappingActionKind::IndependentReplica) {
      result["replicas"] = v.replicas;
      result["work_distribution"] = v.workDistribution;
      result["work_partitions"] =
          copyJSONWithoutFields(*serialized.get("work_partitions"));
    } else if (action.kind ==
               ProgramOperatorMappingActionKind::IndependentResourceShape) {
      result["resource_shape"] =
          copyJSONWithoutFields(*serialized.get("resource_shape"));
      result["resource_demands"] =
          copyJSONWithoutFields(*serialized.get("resource_demands"));
    } else {
      result["residence_policy"] = v.residencePolicy;
      result["resident_allocations"] =
          copyJSONWithoutFields(*serialized.get("resident_allocations"));
      result["packed_inputs"] =
          copyJSONWithoutFields(*serialized.get("packed_inputs"));
    }
    return staticFingerprint(j::Value(std::move(result)));
  };
  std::map<size_t, std::pair<std::string, std::string>> oldCoordinates;
  auto binding = [](const ProgramOperatorExecutionVariant &v) {
    return staticFingerprint(j::Object{{"artifact", v.artifactIdentity},
                                       {"entrypoint", v.entrypoint},
                                       {"context", v.runtimeContextIdentity},
                                       {"residence", v.residencePolicy}});
  };
  for (size_t i = action.firstRegion;
       i < action.firstRegion + action.regionCount; ++i)
    for (size_t op : parent.selection.regions[i].operators)
      oldCoordinates[op] = {coordinate(parent.mapping.variants[i]),
                            binding(parent.mapping.variants[i])};
  bool changed = false;
  for (const auto &request : action.replacements) {
    auto found =
        std::find_if(implementations.begin(), implementations.end(),
                     [&](const auto &i) { return i.id == request.id; });
    auto variant = variants.find(request.id);
    if (found == implementations.end() || variant == variants.end() ||
        !found->executable || variant->second.implementationId != found->id ||
        variant->second.artifactIdentity != found->artifactIdentity ||
        variant->second.entrypoint != found->entrypoint)
      return fail(error,
                  "coordinate action has no registered executable artifact");
    // The request names an immutable registration. Its supplied field edits
    // are ignored, so neither a caller flag nor a stale partial ABI is trusted.
    resolved.replacements.push_back(*found);
    resolved.replacementVariants.push_back(variant->second);
    const auto key = coordinate(variant->second);
    for (size_t op : found->operators) {
      auto old = oldCoordinates.find(op);
      if (old == oldCoordinates.end())
        return fail(error, "coordinate action changes logical source coverage");
      changed |= old->second.first != key &&
                 old->second.second != binding(variant->second);
    }
  }
  if (!changed)
    return fail(
        error,
        "requested coordinate has no actual registered implementation change");
  if (!applyProgramOperatorMappingAction(parent, resolved, child, error) ||
      !configureProgramOperatorExecutionPolicy(child.selection, child.mapping,
                                               error))
    return false;
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
  auto checked = candidate;
  if (!source ||
      !canonicalizeProgramOperatorCandidate(*source, checked, error) ||
      !materializeProgramOperatorGraph(source, checked.selection, target,
                                       hostMemory, materialized, error) ||
      !scheduleProgramOperatorGraph(materialized, checked.mapping, error)) {
    result.message = error;
    // A missing implementation duration is unavailable evidence, rather than
    // mathematical infeasibility. No zero-duration replacement is introduced.
    result.feasibilityKnown =
        error.find("unknown/nonfinite inclusive duration") == std::string::npos;
    return result;
  }
  result.feasible = true;
  double controlMs;
  std::map<std::string, int64_t> controlFeatures;
  if (!estimateProgramOperatorInvocationControl(
          *materialized.operatorSelection, candidate.mapping.controlModel,
          controlMs, controlFeatures, error)) {
    result.feasible = false;
    result.feasibilityKnown = false;
    result.message = error;
    return result;
  }
  result.costMs = materialized.estimatedMakespanMs.value_or(0) + controlMs;
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
  result.result = j::Object{
      {"schedule_policy", materialized.schedule.policy},
      {"physical_callback_makespan_ms",
       materialized.estimatedMakespanMs.value_or(0)},
      {"invocation_control_ms", controlMs},
      {"controller_is_approximate", candidate.mapping.controlModel.has_value()},
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
    for (const auto *field : {"execution_policy", "source_ordering_identity",
                              "async_runtime_proof_identity"})
      if (v->get(field) && !v->getString(field))
        return fail(error, "execution policy/proof fields must be strings");
    if (auto policy = v->getString("execution_policy"))
      variant.executionPolicy = policy->str();
    if (auto identity = v->getString("source_ordering_identity"))
      variant.sourceOrderingIdentity = identity->str();
    if (auto proof = v->getString("async_runtime_proof_identity"))
      variant.asyncRuntimeProofIdentity = proof->str();
    if (variant.executionPolicy == "source-dependency-ready") {
      if (variant.sourceOrderingIdentity != source.sourceId ||
          variant.asyncRuntimeProofIdentity.empty())
        return fail(
            error,
            "ready policy requires source identity and actual runtime proof");
    } else if (variant.executionPolicy != "synchronous-source-order" ||
               !variant.sourceOrderingIdentity.empty() ||
               !variant.asyncRuntimeProofIdentity.empty())
      return fail(error, "unknown or inconsistent execution policy");
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
struct ChoiceDependencyFrontier {
  size_t group = 0;
  // Each choice contracts the same immutable source predecessors to its
  // actual implementation owners. No timing or whole-plan price is cached.
  std::vector<std::vector<size_t>> ownersByChoice;
};
struct DenseInvocationSummary {
  double wallMs = 0;
  std::vector<std::pair<size_t, int64_t>> demands, temporary;
};
struct SerialChoiceSummary {
  double wallMs = 0;
  bool valid = true, feasibilityKnown = true;
  std::string error;
  std::vector<AllocationReservation> allocations;
  std::map<std::string, int64_t> controllerFeatures;
  std::vector<int64_t> temporaryMax;
  std::vector<std::string> unproved;
};
} // namespace
namespace orbit::taskflow {
struct ProgramOperatorMappingDomain {
  std::string identity, registrationIdentity;
  std::optional<ProgramOperatorInvocationControlModel> controlModel;
  StaticConfig target;
  std::vector<ProgramOperatorImplementation> implementations;
  std::vector<ProgramOperatorExecutionVariant> variants;
  std::vector<ProgramOperatorChoiceGroup> groups;
  std::vector<std::vector<MappingAllocation>> boundaryAllocations;
  std::vector<std::vector<size_t>> implementationPredecessors;
  bool sourceDependencyReady = false;
  std::vector<size_t> sourceGroups;
  std::vector<std::vector<std::map<size_t, size_t>>> choiceSourceOwners;
  std::vector<std::vector<SerialChoiceSummary>> serialChoices;
  std::map<std::string, size_t> resourceIndices;
  std::vector<DenseInvocationSummary> denseInvocations;
  std::vector<std::vector<ChoiceDependencyFrontier>> dependencyFrontiers;
  std::vector<std::string> allocationIdentities;
  std::map<std::string, size_t> memoryIndices;
};
} // namespace orbit::taskflow
static ProgramSearchEvaluation evaluateSerialStorageProjection(
    const ProgramOperatorMappingDomain &, const std::vector<size_t> &,
    std::vector<int64_t> *persistentTotals = nullptr);
bool orbit::taskflow::prepareProgramOperatorMappingDomain(
    const ProgramOperatorSource &source,
    const std::vector<ProgramOperatorImplementation> &implementations,
    const std::vector<ProgramOperatorChoiceGroup> &groups,
    const std::map<std::string, ProgramOperatorExecutionVariant> &variants,
    llvm::StringRef registrationIdentity, const StaticConfig &target,
    llvm::StringRef hostMemory,
    std::shared_ptr<const ProgramOperatorMappingDomain> &result,
    std::string &error,
    const ProgramOperatorInvocationControlModel *controlModel) {
  bool readyDomain =
      std::any_of(variants.begin(), variants.end(), [](const auto &entry) {
        return entry.second.executionPolicy == "source-dependency-ready";
      });
  j::Value certificate = nullptr;
  if (registrationIdentity.empty() || !programMemory(target, hostMemory) ||
      !validateMemoryArchitecture(target, error) ||
      !certifyProgramOperatorChoices(source, implementations, groups,
                                     certificate, error, readyDomain))
    return fail(error, "cannot certify mapping domain: " + error);
  auto domain = std::make_shared<ProgramOperatorMappingDomain>();
  domain->registrationIdentity = registrationIdentity.str();
  if (controlModel)
    domain->controlModel = *controlModel;
  domain->target.domains = target.domains;
  domain->target.resources = target.resources;
  domain->target.memories = target.memories;
  domain->target.links = target.links;
  domain->implementations = implementations;
  domain->groups = groups;
  domain->sourceGroups.assign(source.operators.size(), SIZE_MAX);
  if (readyDomain) {
    for (size_t g = 0; g < groups.size(); ++g) {
      std::vector<std::map<size_t, size_t>> choices;
      for (const auto &choice : groups[g].choices) {
        std::map<size_t, size_t> owners;
        for (size_t i : choice)
          for (size_t op : implementations[i].operators) {
            owners.emplace(op, i);
            domain->sourceGroups[op] = g;
          }
        choices.push_back(std::move(owners));
      }
      domain->choiceSourceOwners.push_back(std::move(choices));
    }
  }
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
  std::vector<std::vector<size_t>> semanticDependencies;
  bool sourceDependencyReady =
      std::any_of(variants.begin(), variants.end(), [](const auto &entry) {
        return entry.second.executionPolicy == "source-dependency-ready";
      });
  if (sourceDependencyReady &&
      !programOperatorSourceDependencies(source, semanticDependencies, error))
    return false;
  domain->sourceDependencyReady = sourceDependencyReady;
  ProgramOperatorMapping registration;
  registration.registrationIdentity = registrationIdentity.str();
  registration.controlModel = domain->controlModel;
  for (const auto &impl : implementations) {
    auto variant = variants.find(impl.id);
    const auto *resource = programResource(target, impl.resource);
    if (variant == variants.end() || !resource || resource->kind != "compute" ||
        resource->device != impl.device)
      return fail(error,
                  "missing registered variant or incompatible target resource");
    if (!validateWorkPartitions(source, impl, variant->second, error))
      return false;
    if ((variant->second.executionPolicy == "source-dependency-ready") !=
        sourceDependencyReady)
      return fail(error,
                  "domain mixes synchronous and ready execution protocols");
    std::set<size_t> externalPredecessors;
    if (sourceDependencyReady) {
      std::set<size_t> members(impl.operators.begin(), impl.operators.end());
      for (size_t op : impl.operators)
        for (size_t p : semanticDependencies[op])
          if (!members.count(p))
            externalPredecessors.insert(p);
    }
    domain->implementationPredecessors.emplace_back(
        externalPredecessors.begin(), externalPredecessors.end());
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
  if (readyDomain) {
    domain->dependencyFrontiers.resize(implementations.size());
    for (size_t i = 0; i < implementations.size(); ++i) {
      std::map<size_t, std::vector<size_t>> groupedPredecessors;
      for (size_t op : domain->implementationPredecessors[i])
        groupedPredecessors[domain->sourceGroups[op]].push_back(op);
      for (const auto &entry : groupedPredecessors) {
        ChoiceDependencyFrontier frontier;
        frontier.group = entry.first;
        for (const auto &owners : domain->choiceSourceOwners[entry.first]) {
          std::set<size_t> uniqueOwners;
          for (size_t op : entry.second) {
            auto owner = owners.find(op);
            if (owner == owners.end())
              return fail(error,
                          "cached choice frontier omits source predecessor");
            uniqueOwners.insert(owner->second);
          }
          frontier.ownersByChoice.emplace_back(uniqueOwners.begin(),
                                               uniqueOwners.end());
        }
        domain->dependencyFrontiers[i].push_back(std::move(frontier));
      }
    }
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
  ProgramOperatorSelection registrationSelection;
  registrationSelection.regions = implementations;
  if (domain->controlModel) {
    ProgramOperatorSelection one;
    for (const auto &impl : implementations) {
      one.regions = {impl};
      double controlValidation;
      std::map<std::string, int64_t> features;
      if (!estimateProgramOperatorInvocationControl(
              one, domain->controlModel, controlValidation, features, error))
        return false;
    }
  }
  if (!configureProgramOperatorExecutionPolicy(registrationSelection,
                                               registration, error))
    return false;
  for (size_t r = 0; r < domain->target.resources.size(); ++r)
    domain->resourceIndices.emplace(domain->target.resources[r].id, r);
  domain->denseInvocations.resize(implementations.size());
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
    auto &dense = domain->denseInvocations[i];
    dense.wallMs = impl.wallMs.value_or(0);
    for (const auto &d : expanded)
      if (d.second && domain->resourceIndices.count(d.first))
        dense.demands.emplace_back(domain->resourceIndices.at(d.first),
                                   d.second);
    for (const auto &t : v.temporaryBytes)
      if (t.second && domain->memoryIndices.count(t.first))
        dense.temporary.emplace_back(domain->memoryIndices.at(t.first),
                                     t.second);
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
        if (domain->controlModel)
          for (const auto &feature : implementations[index].controllerFeatures)
            if (!add(summary.controllerFeatures[feature.first],
                     feature.second)) {
              summary.valid = false;
              summary.error = "controller feature sum overflow";
            }
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
static ProgramSearchEvaluation evaluateMappingDomainCalendar(
    const ProgramOperatorMappingDomain &domain,
    const std::vector<size_t> &groupChoices, llvm::StringRef priorityPolicy,
    const std::vector<size_t> *implementationPriorityKeys) {
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
  mapping.controlModel = domain.controlModel;
  std::vector<MappingEvent> events;
  std::map<std::string, MappingAllocation> roots;
  for (size_t index : selected) {
    const auto &registered = domain.implementations[index];
    // This projection retains only invocation identity and timing. Source
    // coverage has already been certified and is immutable in this domain.
    // It is never serialized or admitted as an executable ProgramGraph.
    ProgramOperatorImplementation impl;
    impl.id = registered.id;
    impl.sourceId = registered.sourceId;
    impl.artifactIdentity = registered.artifactIdentity;
    impl.entrypoint = registered.entrypoint;
    impl.resource = registered.resource;
    impl.device = registered.device;
    impl.wallMs = registered.wallMs;
    impl.controllerFeatures = registered.controllerFeatures;
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
  std::string policyError;
  if (!setProgramOperatorReadyPriority(selection, mapping, priorityPolicy,
                                       policyError) ||
      !configureProgramOperatorExecutionPolicy(selection, mapping, policyError))
    return reject(policyError);
  if (implementationPriorityKeys) {
    if (implementationPriorityKeys->size() != domain.implementations.size())
      return reject("implementation priority key count mismatch");
    std::stable_sort(mapping.readyPriority.begin(), mapping.readyPriority.end(),
                     [&](size_t a, size_t b) {
                       return (*implementationPriorityKeys)[selected[a]] <
                              (*implementationPriorityKeys)[selected[b]];
                     });
  }
  if (domain.sourceDependencyReady) {
    // Immutable source DAG is contracted once per registered cover. Per
    // assignment, resolve only the cross-cover frontier, never rewalk source.
    std::vector<size_t> eventForImplementation(domain.implementations.size(),
                                               SIZE_MAX);
    for (size_t r = 0; r < selected.size(); ++r)
      eventForImplementation[selected[r]] = r;
    for (size_t r = 0; r < selected.size(); ++r) {
      std::set<size_t> predecessors;
      for (size_t op : domain.implementationPredecessors[selected[r]]) {
        size_t group = domain.sourceGroups[op];
        auto owner =
            domain.choiceSourceOwners[group][groupChoices[group]].find(op);
        if (owner ==
            domain.choiceSourceOwners[group][groupChoices[group]].end())
          return reject("source predecessor has no selected cover");
        size_t event = eventForImplementation[owner->second];
        if (event == SIZE_MAX)
          return reject("source predecessor implementation was not selected");
        predecessors.insert(event);
      }
      events[r].dependencies.assign(predecessors.begin(), predecessors.end());
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
  double controlMs;
  std::map<std::string, int64_t> controllerFeatures;
  if (!estimateProgramOperatorInvocationControl(
          selection, domain.controlModel, controlMs, controllerFeatures, error))
    return reject(error);
  result.costMs = schedule.estimatedMakespanMs.value_or(0) + controlMs;
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
  result.result =
      j::Object{{"mapping_domain", domain.identity},
                {"physical_callback_makespan_ms",
                 schedule.estimatedMakespanMs.value_or(0)},
                {"invocation_control_ms", controlMs},
                {"controller_is_approximate", domain.controlModel.has_value()},
                {"resource_calendar_checked", true},
                {"source_abi_certified", true},
                {"full_physical_materialization_required", true},
                {"unproved_target_constraints", std::move(unproved)},
                {"runtime_dependencies",
                 domain.sourceDependencyReady
                     ? "source data/control and conservative physical hazards; "
                       "registered async backend"
                     : "actual synchronous invocation order"}};
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
      copy.lastUsers =
          graph.operatorSelection->executionPolicy == "source-dependency-ready"
              ? copy.users
              : std::vector<size_t>{copy.users.back()};
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
// The dense evaluator consumes only immutable certified invocation facts.
// It does not retain whole-plan prices. Joint physical identities are still
// rebuilt for each assignment, then an exact resource/memory calendar is run.
static ProgramSearchEvaluation
evaluateDenseReadyCalendar(const ProgramOperatorMappingDomain &domain,
                           const std::vector<size_t> &choices,
                           llvm::StringRef priorityPolicy,
                           const std::vector<size_t> *priorityKeys) {
  std::vector<int64_t> persistent;
  auto result = evaluateSerialStorageProjection(domain, choices, &persistent);
  if (!result.feasible)
    return result;
  auto reject = [&](const std::string &reason) {
    result.feasible = false;
    result.costMs.reset();
    result.message = "certified dense ready calendar: " + reason;
    return std::move(result);
  };
  if (priorityKeys && priorityKeys->size() != domain.implementations.size())
    return reject("implementation priority key count mismatch");
  if (priorityPolicy != "source-order" &&
      priorityPolicy != "reverse-source-order" &&
      priorityPolicy != "cpu-first" && priorityPolicy != "npu-first" &&
      priorityPolicy != "longest-first")
    return reject("unknown ready-priority policy");
  std::vector<size_t> selected;
  for (size_t g = 0; g < choices.size(); ++g) {
    const auto &members = domain.groups[g].choices[choices[g]];
    selected.insert(selected.end(), members.begin(), members.end());
  }
  std::sort(selected.begin(), selected.end(), [&](size_t a, size_t b) {
    return domain.implementations[a].operators.front() <
           domain.implementations[b].operators.front();
  });
  size_t n = selected.size();
  std::vector<size_t> eventForImplementation(domain.implementations.size(),
                                             SIZE_MAX);
  for (size_t i = 0; i < n; ++i)
    eventForImplementation[selected[i]] = i;
  std::vector<std::vector<size_t>> dependencies(n);
  for (size_t i = 0; i < n; ++i) {
    auto &predecessors = dependencies[i];
    for (const auto &frontier : domain.dependencyFrontiers[selected[i]])
      for (size_t implementation :
           frontier.ownersByChoice[choices[frontier.group]]) {
        size_t event = eventForImplementation[implementation];
        if (event == SIZE_MAX)
          return reject("cached predecessor has no selected implementation");
        predecessors.push_back(event);
      }
    std::sort(predecessors.begin(), predecessors.end());
    predecessors.erase(std::unique(predecessors.begin(), predecessors.end()),
                       predecessors.end());
  }
  std::vector<size_t> priority(n), rank(n);
  for (size_t i = 0; i < n; ++i)
    priority[i] = i;
  if (priorityPolicy == "reverse-source-order")
    std::reverse(priority.begin(), priority.end());
  else if (priorityPolicy != "source-order")
    std::stable_sort(priority.begin(), priority.end(), [&](size_t a, size_t b) {
      const auto &x = domain.implementations[selected[a]],
                 &y = domain.implementations[selected[b]];
      if (priorityPolicy == "longest-first")
        return *x.wallMs > *y.wallMs;
      const char *device = priorityPolicy == "cpu-first" ? "cpu" : "npu";
      return (x.device == device) > (y.device == device);
    });
  if (priorityKeys)
    std::stable_sort(priority.begin(), priority.end(), [&](size_t a, size_t b) {
      return (*priorityKeys)[selected[a]] < (*priorityKeys)[selected[b]];
    });
  for (size_t i = 0; i < n; ++i)
    rank[priority[i]] = i;

  // A strict source-ordered suffix can be summed only after every prefix
  // invocation must have completed. Its REAL invocation identities remain in
  // selection/final materialization; this is an evaluator reduction only.
  size_t prefix = n;
  while (prefix > 1 &&
         std::binary_search(dependencies[prefix - 1].begin(),
                            dependencies[prefix - 1].end(), prefix - 2))
    --prefix;
  bool tailCertified = prefix + 1 < n;
  for (size_t i = 0; i < n && tailCertified; ++i)
    for (size_t p : dependencies[i])
      if ((i < prefix && p >= prefix) || (i >= prefix && p >= i))
        tailCertified = false;
  std::vector<bool> ancestors(n);
  std::vector<size_t> pending;
  if (tailCertified)
    pending = dependencies[prefix];
  while (!pending.empty()) {
    size_t p = pending.back();
    pending.pop_back();
    if (ancestors[p])
      continue;
    ancestors[p] = true;
    pending.insert(pending.end(), dependencies[p].begin(),
                   dependencies[p].end());
  }
  for (size_t p = 0; p < prefix && tailCertified; ++p)
    tailCertified &= ancestors[p];
  if (!tailCertified)
    prefix = n;
  double tailMs = 0;
  for (size_t i = prefix; i < n; ++i)
    tailMs += domain.denseInvocations[selected[i]].wallMs;

  const size_t memories = domain.target.memories.size();
  const size_t resources = domain.target.resources.size();
  // Each invocation chooses one resource it fully occupies. At most one
  // invocation assigned to that lane can execute at once. Sum of per-lane
  // maxima is therefore an upper bound on ANY resource-legal scratch overlap.
  // Failure to prove this bound falls back to the exact memory calendar.
  std::vector<std::vector<int64_t>> laneMax(resources,
                                            std::vector<int64_t>(memories));
  bool memoryIndependent = true;
  for (size_t index : selected) {
    const auto &invocation = domain.denseInvocations[index];
    size_t lane = SIZE_MAX;
    for (const auto &d : invocation.demands)
      if (d.second == domain.target.resources[d.first].capacity) {
        lane = d.first;
        break;
      }
    if (lane == SIZE_MAX) {
      memoryIndependent = false;
      break;
    }
    for (const auto &t : invocation.temporary)
      laneMax[lane][t.first] = std::max(laneMax[lane][t.first], t.second);
  }
  if (memoryIndependent)
    for (size_t m = 0; m < memories; ++m) {
      __int128 peak = persistent[m];
      for (size_t lane = 0; lane < resources; ++lane)
        peak += laneMax[lane][m];
      const auto &memory = domain.target.memories[m];
      if (peak > INT64_MAX ||
          (peak && (!memory.capacityKnown || peak > memory.capacityBytes)))
        memoryIndependent = false;
    }

  using Timeline = std::map<double, int64_t>;
  std::vector<Timeline> resourceCalendar(resources), memoryCalendar(memories);
  auto fits = [](const Timeline &timeline, double begin, double end,
                 int64_t amount, int64_t capacity) {
    if (amount > capacity || amount < 0)
      return false;
    if (!amount || begin == end || timeline.empty() ||
        begin >= timeline.rbegin()->first)
      return true;
    __int128 live = 0;
    for (auto point = timeline.begin();
         point != timeline.end() && point->first < end; ++point) {
      live += point->second;
      auto next = std::next(point);
      if (point->first <= begin && next != timeline.end() &&
          next->first <= begin)
        continue;
      if (live + amount > capacity)
        return false;
    }
    return true;
  };
  auto commit = [](Timeline &timeline, double begin, double end,
                   int64_t amount) {
    if (!amount || begin == end)
      return;
    for (auto point :
         {std::make_pair(begin, amount), std::make_pair(end, -amount)}) {
      // Local bounds and the successful overlap probe ensure this difference
      // of two capacity-bounded live sets fits signed int64.
      auto &delta = timeline[point.first];
      delta = int64_t(__int128(delta) + point.second);
      if (!delta)
        timeline.erase(point.first);
    }
  };
  std::vector<size_t> remaining(prefix);
  std::vector<std::vector<size_t>> users(prefix);
  std::priority_queue<std::pair<size_t, size_t>,
                      std::vector<std::pair<size_t, size_t>>,
                      std::greater<std::pair<size_t, size_t>>>
      ready;
  for (size_t i = 0; i < prefix; ++i) {
    remaining[i] = dependencies[i].size();
    for (size_t p : dependencies[i]) {
      if (p >= prefix || p == i)
        return reject("invalid prefix dependency");
      users[p].push_back(i);
    }
    if (!remaining[i])
      ready.emplace(rank[i], i);
  }
  std::vector<double> ends(prefix);
  double makespan = 0;
  for (size_t issued = 0; issued < prefix; ++issued) {
    if (ready.empty())
      return reject("cyclic source-cover quotient");
    size_t i = ready.top().second;
    ready.pop();
    const auto &invocation = domain.denseInvocations[selected[i]];
    double earliest = 0;
    for (size_t p : dependencies[i])
      earliest = std::max(earliest, ends[p]);
    std::set<double> starts{earliest};
    if (earliest < makespan) {
      for (const auto &d : invocation.demands)
        for (auto p = resourceCalendar[d.first].lower_bound(earliest);
             p != resourceCalendar[d.first].end(); ++p)
          starts.insert(p->first);
      if (!memoryIndependent)
        for (const auto &t : invocation.temporary)
          for (auto p = memoryCalendar[t.first].lower_bound(earliest);
               p != memoryCalendar[t.first].end(); ++p)
            starts.insert(p->first);
    }
    bool placed = false;
    for (double begin : starts) {
      double end = begin + invocation.wallMs;
      if (!std::isfinite(end))
        return reject("schedule time overflow");
      bool legal = true;
      for (const auto &d : invocation.demands)
        legal &= fits(resourceCalendar[d.first], begin, end, d.second,
                      domain.target.resources[d.first].capacity);
      if (!memoryIndependent)
        for (const auto &t : invocation.temporary) {
          const auto &m = domain.target.memories[t.first];
          int64_t available = (m.capacityKnown ? m.capacityBytes : INT64_MAX) -
                              persistent[t.first];
          legal &=
              fits(memoryCalendar[t.first], begin, end, t.second, available);
        }
      if (!legal)
        continue;
      for (const auto &d : invocation.demands)
        commit(resourceCalendar[d.first], begin, end, d.second);
      if (!memoryIndependent)
        for (const auto &t : invocation.temporary)
          commit(memoryCalendar[t.first], begin, end, t.second);
      ends[i] = end;
      makespan = std::max(makespan, end);
      for (size_t u : users[i])
        if (!--remaining[u])
          ready.emplace(rank[u], u);
      placed = true;
      break;
    }
    if (!placed)
      return reject("no resource/memory-feasible invocation placement");
  }
  double controlMs = result.result.getAsObject()
                         ->getNumber("invocation_control_ms")
                         .value_or(0);
  result.costMs = makespan + tailMs + controlMs;
  result.message = "immutable source frontier plus exact dense resource/memory "
                   "calendar; final graph required";
  auto &receipt = *result.result.getAsObject();
  receipt["physical_callback_makespan_ms"] = makespan + tailMs;
  receipt["runtime_dependencies"] =
      "source data/control/physical hazards; registered async backend";
  receipt["evaluation"] = "exact dense ready resource calendar; joint "
                          "allocation identities recomputed";
  receipt["serial_tail_invocations"] = int64_t(n - prefix);
  receipt["serial_tail_ms"] = tailMs;
  receipt["serial_tail_prefix_barrier_proved"] = tailCertified;
  receipt["scratch_noncontention_certificate"] = memoryIndependent;
  receipt["no_invocations_fused_by_evaluator"] = true;
  return result;
}
ProgramSearchEvaluation
orbit::taskflow::evaluateProgramOperatorMappingDomainReference(
    const ProgramOperatorMappingDomain &domain,
    const std::vector<size_t> &choices, llvm::StringRef priority,
    const std::vector<size_t> *keys) {
  return evaluateMappingDomainCalendar(domain, choices, priority, keys);
}
ProgramSearchEvaluation orbit::taskflow::evaluateProgramOperatorMappingDomain(
    const ProgramOperatorMappingDomain &domain,
    const std::vector<size_t> &choices, llvm::StringRef priority,
    const std::vector<size_t> *keys) {
  if (domain.sourceDependencyReady)
    return evaluateDenseReadyCalendar(domain, choices, priority, keys);
  if (priority != "source-order" || keys) {
    ProgramSearchEvaluation rejected;
    rejected.message =
        "synchronous domain has no registered ready-priority search";
    return rejected;
  }
  return evaluateSerialStorageProjection(domain, choices);
}
static ProgramSearchEvaluation
evaluateSerialStorageProjection(const ProgramOperatorMappingDomain &domain,
                                const std::vector<size_t> &groupChoices,
                                std::vector<int64_t> *persistentTotals) {
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
  double wall = 0,
         controlMs = domain.controlModel ? domain.controlModel->interceptMs : 0;
  std::map<std::string, int64_t> controllerFeatures;
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
    for (const auto &feature : choice.controllerFeatures)
      if (!add(controllerFeatures[feature.first], feature.second))
        return reject("controller feature accounting overflow");
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
  if (domain.controlModel)
    for (const auto &feature : controllerFeatures)
      controlMs += domain.controlModel->coefficients.at(feature.first) *
                   double(feature.second);
  if (!std::isfinite(wall) || !std::isfinite(controlMs) ||
      !std::isfinite(wall + controlMs))
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
  if (persistentTotals)
    *persistentTotals = persistent;
  result.feasible = true;
  result.costMs = wall + controlMs;
  result.message = "immutable source/ABI/local-resource proofs plus exact "
                   "synchronous calendar and joint physical allocation unions";
  result.result = j::Object{
      {"mapping_domain", domain.identity},
      {"physical_callback_makespan_ms", wall},
      {"invocation_control_ms", controlMs},
      {"controller_is_approximate", domain.controlModel.has_value()},
      {"controller_model_identity",
       domain.controlModel ? domain.controlModel->identity : "disabled"},
      {"controller_features",
       [&]() {
         j::Object values;
         for (const auto &f : controllerFeatures)
           values[f.first] = f.second;
         return values;
       }()},
      {"resource_calendar_checked", true},
      {"source_abi_certified", true},
      {"full_physical_materialization_required", true},
      {"unproved_target_constraints", std::move(unproved)},
      {"runtime_dependencies", "actual synchronous invocation order"},
      {"evaluation",
       "exact immutable choice-calendar summaries; joint storage recomputed"}};
  return result;
}
