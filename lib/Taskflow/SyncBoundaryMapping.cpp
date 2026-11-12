#include "Taskflow/SyncBoundaryMapping.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <map>
#include <set>

using namespace orbit::taskflow;
namespace {
bool fail(std::string &error, const std::string &why) {
  error = "sync boundary mapping: " + why;
  return false;
}
std::string jsonText(const llvm::json::Value &v) {
  std::string s;
  llvm::raw_string_ostream out(s);
  out << v;
  return out.str();
}
llvm::json::Array indices(const std::vector<size_t> &v) {
  llvm::json::Array a;
  for (auto x : v)
    a.push_back(int64_t(x));
  return a;
}
bool tensorBytes(const GenericStorage &s, int64_t &bytes, std::string &error) {
  if (s.dtype != "f32")
    return fail(error, "only checked static f32 tensor bindings are supported");
  bytes = 4;
  for (auto n : s.shape) {
    if (n <= 0 || bytes > INT64_MAX / n)
      return fail(error, "tensor byte extent overflow");
    bytes *= n;
  }
  return true;
}
llvm::json::Value rangeJSON(const PhysicalBORange &r) {
  return llvm::json::Object{{"bo", r.bo},
                            {"generation", int64_t(r.generation)},
                            {"offset", r.offset},
                            {"bytes", r.bytes}};
}
const char *kindName(SyncObservationKind k) {
  switch (k) {
  case SyncObservationKind::HostRead:
    return "host-read";
  case SyncObservationKind::HostWrite:
    return "host-write";
  case SyncObservationKind::SyncBegin:
    return "sync-begin";
  case SyncObservationKind::SyncComplete:
    return "sync-complete";
  case SyncObservationKind::DeviceSubmit:
    return "device-submit";
  case SyncObservationKind::WaitSuccess:
    return "wait-success";
  case SyncObservationKind::WaitFailure:
    return "wait-failure";
  case SyncObservationKind::UnknownMutation:
    return "unknown-mutation";
  }
  return "invalid";
}
} // namespace

bool orbit::taskflow::collectSyncBoundaryQueries(
    const ProgramGraph &graph, const MechanismFacts &facts,
    const std::vector<SyncRuntimeBOBinding> &bindings,
    const SyncQueryContext &context, SyncBoundaryTrace &output,
    std::string &error) {
  // Recollect through the checked public graph before dereferencing supplied
  // IDs.
  MechanismFacts canonical;
  if (!collectMechanismFacts(graph, facts.runtime, canonical, error))
    return false;
  if (jsonText(serializeMechanismFacts(canonical)) !=
      jsonText(serializeMechanismFacts(facts)))
    return fail(
        error,
        "mechanism facts do not match the checked graph/runtime contract");
  if (context.callForm != SyncCallForm::ImplicitFullBuffer ||
      context.observerBoundary != SyncObserverBoundary::DetailedCompatibility ||
      context.runtimeIdentity != facts.runtime.identity ||
      context.observerIdentity.empty())
    return fail(
        error, "requires bound implicit-full-buffer detailed observer context");
  SyncBoundaryTrace result;
  result.graphIdentity = facts.graphId;
  result.factsIdentity = facts.identity;
  result.bindings = bindings;
  SyncState engine;
  std::map<size_t, const SyncRuntimeBOBinding *> byStorage;
  std::map<size_t, int64_t> sizes;
  std::set<std::string> physical;
  for (const auto &b : bindings) {
    if (b.storage >= graph.source.storages.size() || b.physicalId.empty() ||
        b.extentEvidence.empty() || b.observedExtentBytes <= 0 ||
        b.storageOffset < 0 || !physical.insert(b.physicalId).second ||
        !byStorage.emplace(b.storage, &b).second)
      return fail(error,
                  "invalid or duplicate storage/physical BO runtime binding");
    int64_t bytes = 0;
    if (!tensorBytes(graph.source.storages[b.storage], bytes, error))
      return false;
    if (b.storageOffset > b.observedExtentBytes ||
        bytes > b.observedExtentBytes - b.storageOffset)
      return fail(error, "logical storage range exceeds observed BO extent");
    sizes[b.storage] = bytes;
    std::set<size_t> seen;
    for (auto copy : b.copies) {
      if (copy >= graph.copies.size() || !seen.insert(copy).second)
        return fail(error, "invalid or duplicate logical copy alias");
      const auto &allocation = graph.allocations[graph.copies[copy].allocation];
      if (allocation.storage != b.storage ||
          allocation.memory != "system.shared")
        return fail(error,
                    "copy alias is not this storage's shared host endpoint");
    }
    if (!engine.registerPhysicalBO(b.physicalId, b.generation,
                                   b.observedExtentBytes, b.copies, error))
      return false;
  }
  for (const auto &r : graph.executionRegions)
    if (r.placement == "npu") {
      for (const auto *list : {&r.inputCopies, &r.outputCopies})
        for (auto copy : *list) {
          auto storage =
              graph.allocations[graph.copies[copy].allocation].storage;
          auto found = byStorage.find(storage);
          if (found == byStorage.end())
            return fail(error,
                        "missing observed BO binding for NPU ABI storage");
          const auto &aliases = found->second->copies;
          if (std::find(aliases.begin(), aliases.end(), copy) == aliases.end())
            return fail(error,
                        "NPU ABI host copy is absent from runtime aliases");
        }
    }
  auto range = [&](size_t storage, bool wholeBO = false) {
    const auto &b = *byStorage.at(storage);
    return PhysicalBORange{b.physicalId, b.generation,
                           wholeBO ? 0 : b.storageOffset,
                           wholeBO ? b.observedExtentBytes : sizes.at(storage)};
  };
  auto observe = [&](SyncObservation o, int64_t item,
                     const std::vector<size_t> &copies,
                     const std::vector<size_t> &events,
                     const std::vector<size_t> &regionEvents) {
    o.id = result.observations.size() + 1;
    if (!engine.observe(o, error))
      return false;
    result.observations.push_back(
        {std::move(o), item, copies, events, regionEvents});
    return true;
  };
  auto host = [&](size_t storage, bool write, int64_t item,
                  const std::vector<size_t> &copies,
                  const std::vector<size_t> &events,
                  const std::vector<size_t> &regionEvents) {
    if (!byStorage.count(storage))
      return true; // ordinary host arrays have no BO state
    SyncObservation o;
    o.kind =
        write ? SyncObservationKind::HostWrite : SyncObservationKind::HostRead;
    o.range = range(storage);
    return observe(o, item, copies, events, regionEvents);
  };
  // The installed executor poisons exposed non-input arrays outside its clock.
  if (facts.runtime.outputPoisoned)
    for (const auto &entry : byStorage)
      if (graph.source.storages[entry.first].role != "input" &&
          !host(entry.first, true, -1, entry.second->copies, {}, {}))
        return false;
  for (const auto &item : facts.items) {
    if (item.mechanism == "host.input_reset") {
      if (!host(size_t(item.storage), true, item.id, item.copies,
                item.publicEvents, item.regionEvents))
        return false;
    } else if (item.mechanism == "cpu.call") {
      for (auto task :
           graph.executionRegions[size_t(item.region)].sourceTasks) {
        const auto &t = graph.source.tasks[task];
        std::set<size_t> reads, writes;
        for (const auto &a : t.accesses)
          if (byStorage.count(a.storage)) {
            const auto &s = graph.source.storages[a.storage];
            bool whole = a.exactRegion && a.region.size() == s.shape.size();
            for (size_t i = 0; whole && i < s.shape.size(); ++i)
              whole = a.region[i].lower == 0 && a.region[i].upper == s.shape[i];
            if (!whole)
              return fail(
                  error,
                  "CPU BO access is not a checked exact whole-storage region");
            if (a.read)
              reads.insert(a.storage);
            if (a.write)
              writes.insert(a.storage);
          }
        for (auto storage : writes)
          if (reads.count(storage))
            return fail(
                error,
                "CPU in-place BO access order is outside the finite adapter");
        for (bool write : {false, true})
          for (auto storage : write ? writes : reads)
            if (!host(storage, write, item.id, byStorage.at(storage)->copies,
                      item.publicEvents, item.regionEvents))
              return false;
      }
    } else if (item.mechanism == "bo.sync_to_device" ||
               item.mechanism == "bo.sync_from_device") {
      auto storage = size_t(item.storage);
      auto full = range(storage, true);
      auto direction = item.mechanism == "bo.sync_to_device"
                           ? SyncDirection::ToDevice
                           : SyncDirection::FromDevice;
      SyncBoundaryQuery q;
      q.mechanismItem = item.id;
      q.region = item.region;
      q.storage = item.storage;
      q.direction = direction;
      q.copies = item.copies;
      q.publicEvents = item.publicEvents;
      q.regionEvents = item.regionEvents;
      q.sourceTasks = item.sourceTasks;
      if (!engine.query(full, context, q.result, error))
        return false;
      result.queries.push_back(std::move(q));
      SyncObservation o;
      o.kind = SyncObservationKind::SyncBegin;
      o.range = full;
      o.direction = direction;
      o.operation = "sync-item-" + std::to_string(item.id);
      if (!observe(o, item.id, item.copies, item.publicEvents,
                   item.regionEvents))
        return false;
      o.kind = SyncObservationKind::SyncComplete;
      if (!observe(o, item.id, item.copies, item.publicEvents,
                   item.regionEvents))
        return false;
    } else if (item.mechanism == "npu.submit" || item.mechanism == "npu.wait") {
      SyncObservation o;
      o.operation = "region-" + std::to_string(item.region);
      o.kind = item.mechanism == "npu.submit"
                   ? SyncObservationKind::DeviceSubmit
                   : SyncObservationKind::WaitSuccess;
      const auto &r = graph.executionRegions[size_t(item.region)];
      std::set<size_t> abiCopies(r.inputCopies.begin(), r.inputCopies.end());
      abiCopies.insert(r.outputCopies.begin(), r.outputCopies.end());
      if (o.kind == SyncObservationKind::DeviceSubmit) {
        std::map<size_t, SyncDeclaredAccess> accesses;
        for (bool write : {false, true})
          for (auto copy : write ? r.outputCopies : r.inputCopies) {
            auto storage =
                graph.allocations[graph.copies[copy].allocation].storage;
            auto &a = accesses[storage];
            a.range = range(storage);
            if (write)
              a.write = true;
            else
              a.read = true;
          }
        for (auto &a : accesses)
          o.accesses.push_back(a.second);
      }
      std::vector<size_t> copies(abiCopies.begin(), abiCopies.end());
      if (!observe(o, item.id, copies, item.publicEvents, item.regionEvents))
        return false;
    } else if (item.mechanism == "host.output_copy") {
      for (const auto &entry : byStorage)
        if (graph.source.storages[entry.first].role == "output")
          if (!host(entry.first, false, item.id, entry.second->copies,
                    item.publicEvents, item.regionEvents))
            return false;
    }
    // Native DMA, source-access descriptive rows and bookkeeping create no
    // additional BO state transitions outside their owning calls.
  }
  output = std::move(result);
  return true;
}

llvm::json::Value
orbit::taskflow::serializeSyncBoundaryTrace(const SyncBoundaryTrace &t) {
  llvm::json::Array bindings, observations, queries, predicates;
  for (const auto &s : t.requiredPredicates)
    predicates.push_back(s);
  for (const auto &b : t.bindings)
    bindings.push_back(
        llvm::json::Object{{"storage", int64_t(b.storage)},
                           {"physical_id", b.physicalId},
                           {"generation", int64_t(b.generation)},
                           {"observed_extent_bytes", b.observedExtentBytes},
                           {"storage_offset", b.storageOffset},
                           {"copies", indices(b.copies)},
                           {"extent_evidence", b.extentEvidence}});
  for (const auto &v : t.observations) {
    const auto &o = v.observation;
    llvm::json::Array accesses;
    for (const auto &a : o.accesses)
      accesses.push_back(llvm::json::Object{
          {"range", rangeJSON(a.range)}, {"read", a.read}, {"write", a.write}});
    observations.push_back(llvm::json::Object{
        {"id", int64_t(o.id)},
        {"kind", kindName(o.kind)},
        {"operation", o.operation},
        {"range", rangeJSON(o.range)},
        {"accesses", std::move(accesses)},
        {"direction",
         o.direction == SyncDirection::ToDevice ? "TO_DEVICE" : "FROM_DEVICE"},
        {"mechanism_item", v.mechanismItem},
        {"copies", indices(v.copies)},
        {"public_events", indices(v.publicEvents)},
        {"region_events", indices(v.regionEvents)}});
  }
  for (const auto &q : t.queries)
    queries.push_back(llvm::json::Object{
        {"mechanism_item", int64_t(q.mechanismItem)},
        {"region", q.region},
        {"storage", q.storage},
        {"direction",
         q.direction == SyncDirection::ToDevice ? "TO_DEVICE" : "FROM_DEVICE"},
        {"copies", indices(q.copies)},
        {"public_events", indices(q.publicEvents)},
        {"region_events", indices(q.regionEvents)},
        {"source_tasks", indices(q.sourceTasks)},
        {"query", serializeSyncQuery(q.result)}});
  return llvm::json::Object{{"schema", "orbit.sync-boundary-projection.v14"},
                            {"origin", t.origin},
                            {"required_predicates", std::move(predicates)},
                            {"graph_id", t.graphIdentity},
                            {"facts_id", t.factsIdentity},
                            {"bindings", std::move(bindings)},
                            {"observations", std::move(observations)},
                            {"queries", std::move(queries)},
                            {"timing_inferred", false},
                            {"cache_dirty_proof", false}};
}
