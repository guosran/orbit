#include "Taskflow/SyncState.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <utility>

using namespace orbit::taskflow;
namespace {
bool fail(std::string &error, const std::string &why) {
  error = "sync state: " + why;
  return false;
}
bool same(const PhysicalBORange &a, const PhysicalBORange &b) {
  return a.bo == b.bo && a.generation == b.generation && a.offset == b.offset &&
         a.bytes == b.bytes;
}
bool overlaps(const PhysicalBORange &a, const PhysicalBORange &b) {
  return a.bo == b.bo && a.generation == b.generation &&
         a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes;
}
std::string classification(const SyncRangeState &s) {
  if (!s.known)
    return "unknown";
  auto sync = std::max(s.toCompleteEpoch, s.fromCompleteEpoch);
  if (s.hostWriteEpoch >
      std::max({sync, s.deviceReadEpoch, s.deviceWriteEpoch}))
    return sync ? "host-write-since-sync" : "host-write-observed";
  if (s.deviceWriteEpoch > std::max(s.hostWriteEpoch, s.fromCompleteEpoch))
    return "after-device-write-complete";
  if (s.deviceReadEpoch >
      std::max({sync, s.hostWriteEpoch, s.hostReadEpoch, s.deviceWriteEpoch}))
    return "after-device-read-complete";
  if (s.hostReadEpoch > std::max(sync, s.hostWriteEpoch))
    return sync ? "host-read-since-sync" : "host-read-observed";
  if (s.fromCompleteEpoch >= s.toCompleteEpoch && s.fromCompleteEpoch)
    return "post-FROM-no-access";
  if (s.toCompleteEpoch)
    return "post-TO-no-access";
  return "unknown";
}
auto signature(const SyncRangeState &s) {
  return std::make_tuple(s.known, s.hostVisible, s.deviceVisible,
                         s.contentEpoch, s.hostWriteEpoch, s.hostReadEpoch,
                         s.deviceReadEpoch, s.deviceWriteEpoch,
                         s.toCompleteEpoch, s.fromCompleteEpoch);
}
llvm::json::Value rangeJSON(const PhysicalBORange &r) {
  return llvm::json::Object{{"bo", r.bo},
                            {"generation", int64_t(r.generation)},
                            {"offset", r.offset},
                            {"bytes", r.bytes}};
}
} // namespace
struct SyncState::Impl {
  struct Buffer {
    uint64_t generation = 0;
    int64_t bytes = 0;
    std::vector<size_t> copies;
    std::vector<SyncRangeState> cells;
  };
  struct PendingSync {
    PhysicalBORange range;
    SyncDirection direction;
    uint64_t begin;
  };
  struct PendingCommand {
    std::vector<SyncDeclaredAccess> accesses;
    uint64_t submit;
  };
  std::map<std::string, Buffer> buffers;
  std::map<std::string, PendingSync> syncs;
  std::map<std::string, PendingCommand> commands;
  std::set<std::string> usedOperations;
  uint64_t last = 0;
  bool valid(const PhysicalBORange &range, std::string &error) const {
    auto b = buffers.find(range.bo);
    if (b == buffers.end() || range.generation != b->second.generation)
      return fail(error, "unknown or retired physical BO generation");
    if (range.offset < 0 || range.bytes <= 0 ||
        range.offset > b->second.bytes ||
        range.bytes > b->second.bytes - range.offset)
      return fail(error, "byte range outside physical BO");
    return true;
  }
  bool pending(const PhysicalBORange &range,
               const std::string &except = {}) const {
    for (const auto &p : syncs)
      if (p.first != except && overlaps(range, p.second.range))
        return true;
    for (const auto &p : commands)
      if (p.first != except)
        for (const auto &access : p.second.accesses)
          if (overlaps(range, access.range))
            return true;
    return false;
  }
  template <typename F> void touch(const PhysicalBORange &range, F action) {
    auto &cells = buffers.at(range.bo).cells;
    for (int64_t cut : {range.offset, range.offset + range.bytes})
      for (size_t i = 0; i < cells.size(); ++i) {
        auto end = cells[i].range.offset + cells[i].range.bytes;
        if (cells[i].range.offset < cut && cut < end) {
          auto right = cells[i];
          right.range.offset = cut;
          right.range.bytes = end - cut;
          cells[i].range.bytes = cut - cells[i].range.offset;
          cells.insert(cells.begin() + i + 1, std::move(right));
          break;
        }
      }
    for (auto &cell : cells)
      if (cell.range.offset >= range.offset &&
          cell.range.offset + cell.range.bytes <= range.offset + range.bytes)
        action(cell);
  }
};
SyncState::SyncState() : impl(new Impl) {}
SyncState::~SyncState() = default;
SyncState::SyncState(SyncState &&) noexcept = default;
SyncState &SyncState::operator=(SyncState &&) noexcept = default;

bool SyncState::registerPhysicalBO(const std::string &id, uint64_t generation,
                                   int64_t totalBytes,
                                   const std::vector<size_t> &copies,
                                   std::string &error) {
  if (id.empty() || generation > uint64_t(INT64_MAX) || totalBytes <= 0)
    return fail(
        error,
        "BO identity, representable generation and positive extent required");
  std::set<size_t> unique;
  for (auto copy : copies)
    if (copy > size_t(INT64_MAX) || !unique.insert(copy).second)
      return fail(error, "invalid or duplicate logical copy evidence alias");
  auto existing = impl->buffers.find(id);
  if (existing != impl->buffers.end()) {
    if (generation <= existing->second.generation)
      return fail(error, "physical BO generation must increase on reuse");
    PhysicalBORange old{id, existing->second.generation, 0,
                        existing->second.bytes};
    if (impl->pending(old))
      return fail(error, "cannot retire a physical BO with pending operations");
  }
  Impl::Buffer buffer;
  buffer.generation = generation;
  buffer.bytes = totalBytes;
  buffer.copies = copies;
  SyncRangeState initial;
  initial.range = {id, generation, 0, totalBytes};
  buffer.cells = {initial};
  impl->buffers[id] = std::move(buffer);
  return true;
}

bool SyncState::observe(const SyncObservation &event, std::string &error) {
  if (event.id == 0 || event.id > uint64_t(INT64_MAX) || event.id <= impl->last)
    return fail(error, "positive strictly increasing observation IDs required");
  if ((event.kind == SyncObservationKind::SyncBegin ||
       event.kind == SyncObservationKind::SyncComplete) &&
      event.direction != SyncDirection::ToDevice &&
      event.direction != SyncDirection::FromDevice)
    return fail(error, "unsupported sync direction");
  auto evidence = [&](SyncRangeState &cell) {
    cell.evidence.push_back(event.id);
  };
  switch (event.kind) {
  case SyncObservationKind::HostRead:
  case SyncObservationKind::HostWrite:
  case SyncObservationKind::UnknownMutation:
    if (!impl->valid(event.range, error))
      return false;
    if (impl->pending(event.range))
      return fail(error, "host access/mutation overlaps an outstanding sync or "
                         "device command");
    impl->touch(event.range, [&](auto &cell) {
      if (event.kind == SyncObservationKind::HostRead)
        cell.hostReadEpoch = event.id;
      else if (event.kind == SyncObservationKind::HostWrite) {
        cell.known = true;
        cell.contentEpoch = event.id;
        cell.hostWriteEpoch = event.id;
        cell.hostVisible = true;
        cell.deviceVisible = false;
      } else {
        cell.known = false;
        cell.contentEpoch = 0;
        cell.hostVisible = false;
        cell.deviceVisible = false;
      }
      evidence(cell);
    });
    break;
  case SyncObservationKind::SyncBegin:
    if (!impl->valid(event.range, error))
      return false;
    if (event.operation.empty() || impl->usedOperations.count(event.operation))
      return fail(error, "unique sync operation identity required");
    if (impl->pending(event.range))
      return fail(error, "sync overlaps an outstanding operation");
    impl->syncs.emplace(
        event.operation,
        Impl::PendingSync{event.range, event.direction, event.id});
    impl->usedOperations.insert(event.operation);
    break;
  case SyncObservationKind::SyncComplete: {
    auto pending = impl->syncs.find(event.operation);
    if (pending == impl->syncs.end() ||
        !same(event.range, pending->second.range) ||
        event.direction != pending->second.direction)
      return fail(error,
                  "sync completion lacks matching begin/range/direction");
    auto begin = pending->second.begin;
    impl->touch(event.range, [&](auto &cell) {
      if (event.direction == SyncDirection::ToDevice) {
        cell.toCompleteEpoch = event.id;
        cell.deviceVisible = cell.known && cell.hostVisible;
        if (!cell.deviceVisible) {
          cell.known = false;
          cell.contentEpoch = 0;
        }
      } else {
        cell.fromCompleteEpoch = event.id;
        cell.hostVisible = cell.known && cell.deviceVisible;
        if (!cell.hostVisible) {
          cell.known = false;
          cell.contentEpoch = 0;
        }
      }
      cell.evidence.push_back(begin);
      evidence(cell);
    });
    impl->syncs.erase(pending);
    break;
  }
  case SyncObservationKind::DeviceSubmit: {
    if (event.operation.empty() ||
        impl->usedOperations.count(event.operation) || event.accesses.empty())
      return fail(error,
                  "unique command and nonempty declared accesses required");
    for (const auto &access : event.accesses) {
      if (!access.read && !access.write)
        return fail(error, "command access must declare read or write");
      if (!impl->valid(access.range, error))
        return false;
      if (impl->pending(access.range))
        return fail(error, "device command overlaps an outstanding operation");
    }
    impl->commands.emplace(event.operation,
                           Impl::PendingCommand{event.accesses, event.id});
    impl->usedOperations.insert(event.operation);
    break;
  }
  case SyncObservationKind::WaitSuccess:
  case SyncObservationKind::WaitFailure: {
    auto pending = impl->commands.find(event.operation);
    if (pending == impl->commands.end())
      return fail(error, "wait has no outstanding declared command");
    // Cut all declaration boundaries first so overlapping read/write operands
    // are combined into one final update per cell, rather than ordered stores.
    for (const auto &access : pending->second.accesses)
      impl->touch(access.range, [](auto &) {});
    for (auto &buffer : impl->buffers)
      for (auto &cell : buffer.second.cells) {
        bool read = false, write = false;
        for (const auto &access : pending->second.accesses)
          if (overlaps(cell.range, access.range)) {
            read |= access.read;
            write |= access.write;
          }
        if (!read && !write)
          continue;
        if (event.kind == SyncObservationKind::WaitFailure) {
          cell.known = false;
          cell.contentEpoch = 0;
          cell.hostVisible = false;
          cell.deviceVisible = false;
        } else {
          if (read)
            cell.deviceReadEpoch = event.id;
          if (write) {
            cell.known = true;
            cell.contentEpoch = event.id;
            cell.deviceWriteEpoch = event.id;
            cell.deviceVisible = true;
            cell.hostVisible = false;
          } else if (read && (!cell.known || !cell.deviceVisible)) {
            cell.known = false;
            cell.contentEpoch = 0;
          }
        }
        cell.evidence.push_back(pending->second.submit);
        evidence(cell);
      }
    impl->commands.erase(pending);
    break;
  }
  default:
    return fail(error, "unsupported observation kind");
  }
  impl->last = event.id;
  return true;
}

bool SyncState::query(const PhysicalBORange &range,
                      const SyncQueryContext &context, SyncQueryResult &result,
                      std::string &error) const {
  result = {};
  result.range = range;
  result.context = context;
  if (!context.callForm || !context.observerBoundary)
    return fail(
        error,
        "query requires explicit invocation form and observation boundary");
  if ((*context.callForm != SyncCallForm::ImplicitFullBuffer &&
       *context.callForm != SyncCallForm::ExplicitRange) ||
      (*context.observerBoundary != SyncObserverBoundary::MinimalAPI &&
       *context.observerBoundary !=
           SyncObserverBoundary::DetailedCompatibility &&
       *context.observerBoundary != SyncObserverBoundary::BatchOuter))
    return fail(error, "unsupported invocation form or observation boundary");
  if (!impl->valid(range, error))
    return false;
  const auto &buffer = impl->buffers.at(range.bo);
  if (*context.callForm == SyncCallForm::ImplicitFullBuffer &&
      (range.offset != 0 || range.bytes != buffer.bytes))
    return fail(error,
                "implicit sync invocation cannot denote a partial BO range");
  result.copies = buffer.copies;
  result.pending = impl->pending(range);
  for (const auto &cell : buffer.cells)
    if (overlaps(cell.range, range)) {
      auto clipped = cell;
      auto begin = std::max(cell.range.offset, range.offset),
           end = std::min(cell.range.offset + cell.range.bytes,
                          range.offset + range.bytes);
      clipped.range.offset = begin;
      clipped.range.bytes = end - begin;
      result.segments.push_back(std::move(clipped));
    }
  result.homogeneous = !result.segments.empty();
  for (size_t i = 1; i < result.segments.size(); ++i)
    result.homogeneous &=
        signature(result.segments[i]) == signature(result.segments.front());
  result.known =
      result.homogeneous && !result.pending && result.segments.front().known;
  result.state =
      result.known ? classification(result.segments.front()) : "unknown";
  if (result.pending)
    result.reason = "overlapping sync or command has not completed";
  else if (!result.homogeneous)
    result.reason = "requested range has mixed observable epochs/history";
  else if (!result.known)
    result.reason =
        "content origin or completed visibility is unobserved/invalidated";
  else
    result.reason = "observed API/access ordering only; not a physical "
                    "cache-dirty or DDR-traffic proof";
  return true;
}
const char *orbit::taskflow::syncCallFormName(SyncCallForm form) {
  return form == SyncCallForm::ImplicitFullBuffer ? "implicit-full-buffer"
         : form == SyncCallForm::ExplicitRange    ? "explicit-range"
                                                  : "unknown";
}
const char *
orbit::taskflow::syncObserverBoundaryName(SyncObserverBoundary boundary) {
  switch (boundary) {
  case SyncObserverBoundary::MinimalAPI:
    return "minimal-api";
  case SyncObserverBoundary::DetailedCompatibility:
    return "detailed-compatibility";
  case SyncObserverBoundary::BatchOuter:
    return "batch-outer";
  }
  return "unknown";
}
llvm::json::Value
orbit::taskflow::serializeSyncQuery(const SyncQueryResult &query) {
  llvm::json::Array segments, copies;
  for (auto copy : query.copies)
    copies.push_back(int64_t(copy));
  for (const auto &state : query.segments) {
    llvm::json::Array evidence;
    for (auto id : state.evidence)
      evidence.push_back(int64_t(id));
    segments.push_back(llvm::json::Object{
        {"range", rangeJSON(state.range)},
        {"known", state.known},
        {"host_visible", state.hostVisible},
        {"device_visible", state.deviceVisible},
        {"content_epoch", int64_t(state.contentEpoch)},
        {"host_write_epoch", int64_t(state.hostWriteEpoch)},
        {"host_read_epoch", int64_t(state.hostReadEpoch)},
        {"device_read_epoch", int64_t(state.deviceReadEpoch)},
        {"device_write_epoch", int64_t(state.deviceWriteEpoch)},
        {"to_complete_epoch", int64_t(state.toCompleteEpoch)},
        {"from_complete_epoch", int64_t(state.fromCompleteEpoch)},
        {"state", classification(state)},
        {"evidence", std::move(evidence)}});
  }
  return llvm::json::Object{
      {"schema", "orbit.sync-query.v14"},
      {"range", rangeJSON(query.range)},
      {"call_form", query.context.callForm ? llvm::json::Value(syncCallFormName(
                                                 *query.context.callForm))
                                           : llvm::json::Value(nullptr)},
      {"observation_boundary", query.context.observerBoundary
                                   ? llvm::json::Value(syncObserverBoundaryName(
                                         *query.context.observerBoundary))
                                   : llvm::json::Value(nullptr)},
      {"runtime_identity", query.context.runtimeIdentity},
      {"observer_identity", query.context.observerIdentity},
      {"known", query.known},
      {"homogeneous", query.homogeneous},
      {"pending", query.pending},
      {"state", query.state},
      {"reason", query.reason},
      {"copy_aliases", std::move(copies)},
      {"segments", std::move(segments)},
      {"scope", "observed range/access/API completion history; no timing-based "
                "state, cache or physical DDR inference"}};
}
