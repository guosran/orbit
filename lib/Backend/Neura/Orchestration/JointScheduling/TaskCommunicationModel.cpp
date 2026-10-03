//===- TaskCommunicationModel.cpp ---------------------------*- C++ -*-===//
//
// Adapts the explicit task-edge and inter-task network contracts to the
// optional communication-aware TaskScheduler mode.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.h"

#include "Backend/Neura/NeuraBackendOptions.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

namespace {

// The runtime route selector can choose any simple directed route and then
// accounts for link reservations.  For a lower bound we therefore avoid
// assuming that the BFS route in InterTaskNetworkSpec::route() is the route
// selected by routeTransfer().  For one endpoint pair, every route has at
// least the shortest possible latency and no route can have a bottleneck
// wider than the widest possible path.  Their sum is consequently an
// admissible (possibly non-tight) lower bound for every route, including
// routes selected after reservations are applied.
static std::optional<uint64_t> minimumPathLatency(
    const InterTaskNetworkSpec &network, TaskNetworkCoordinate source,
    TaskNetworkCoordinate destination) {
  const int rows = network.getRows();
  const int columns = network.getColumns();
  if (rows <= 0 || columns <= 0 || source.row < 0 || source.row >= rows ||
      source.column < 0 || source.column >= columns || destination.row < 0 ||
      destination.row >= rows || destination.column < 0 ||
      destination.column >= columns)
    return std::nullopt;
  if (source == destination)
    return 0;

  const int nodeCount = rows * columns;
  auto nodeId = [columns](TaskNetworkCoordinate coordinate) {
    return coordinate.row * columns + coordinate.column;
  };
  const int sourceId = nodeId(source);
  const int destinationId = nodeId(destination);
  std::vector<uint64_t> distance(nodeCount,
                                 std::numeric_limits<uint64_t>::max());
  using QueueEntry = std::pair<uint64_t, int>;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                      std::greater<QueueEntry>> queue;
  distance[sourceId] = 0;
  queue.push({0, sourceId});
  while (!queue.empty()) {
    const auto [currentDistance, current] = queue.top();
    queue.pop();
    if (currentDistance != distance[current])
      continue;
    if (current == destinationId)
      return currentDistance;
    for (const InterTaskNetworkLink &link : network.getLinks()) {
      if (link.source.row * columns + link.source.column != current)
        continue;
      const int next = nodeId(link.destination);
      if (currentDistance > std::numeric_limits<uint64_t>::max() -
                                 link.latency_cycles)
        continue;
      const uint64_t candidate = currentDistance + link.latency_cycles;
      if (candidate < distance[next]) {
        distance[next] = candidate;
        queue.push({candidate, next});
      }
    }
  }
  return std::nullopt;
}

static std::optional<uint64_t> maximumPathBandwidth(
    const InterTaskNetworkSpec &network, TaskNetworkCoordinate source,
    TaskNetworkCoordinate destination) {
  const int rows = network.getRows();
  const int columns = network.getColumns();
  if (rows <= 0 || columns <= 0 || source.row < 0 || source.row >= rows ||
      source.column < 0 || source.column >= columns || destination.row < 0 ||
      destination.row >= rows || destination.column < 0 ||
      destination.column >= columns)
    return std::nullopt;
  if (source == destination)
    return network.getLocalBandwidthBitsPerCycle();

  const int nodeCount = rows * columns;
  auto nodeId = [columns](TaskNetworkCoordinate coordinate) {
    return coordinate.row * columns + coordinate.column;
  };
  const int sourceId = nodeId(source);
  const int destinationId = nodeId(destination);
  std::vector<uint64_t> widest(nodeCount, 0);
  using QueueEntry = std::pair<uint64_t, int>;
  std::priority_queue<QueueEntry> queue;
  widest[sourceId] = std::numeric_limits<uint64_t>::max();
  queue.push({widest[sourceId], sourceId});
  while (!queue.empty()) {
    const auto [currentBandwidth, current] = queue.top();
    queue.pop();
    if (currentBandwidth != widest[current])
      continue;
    if (current == destinationId)
      return currentBandwidth;
    for (const InterTaskNetworkLink &link : network.getLinks()) {
      if (link.source.row * columns + link.source.column != current)
        continue;
      const int next = nodeId(link.destination);
      const uint64_t candidate =
          std::min(currentBandwidth, link.bandwidth_bits_per_cycle);
      if (candidate > widest[next]) {
        widest[next] = candidate;
        queue.push({candidate, next});
      }
    }
  }
  return std::nullopt;
}

} // namespace

bool InterTaskNetworkCommunicationModel::getTaskPredecessors(
    Operation *consumer, llvm::SmallVectorImpl<Operation *> &predecessors,
    std::string &error) const {
  error.clear();
  predecessors.clear();
  if (!consumer) {
    error = "communication predecessor query has a null consumer";
    return false;
  }

  llvm::DenseSet<Operation *> seen;
  for (const TaskEdge &edge : edge_graph_.getEdges()) {
    if (edge.consumer != consumer)
      continue;
    taskflow::TaskflowTaskOp producer_task = edge.producer;
    Operation *producer = producer_task.getOperation();
    if (seen.insert(producer).second)
      predecessors.push_back(producer);
  }
  return true;
}

const InterTaskNetworkCommunicationModel::TaskPairPayloadCache *
InterTaskNetworkCommunicationModel::getTaskPairPayloadCache(
    Operation *producer, Operation *consumer) const {
  if (!producer || !consumer)
    return nullptr;

  TaskPairCache &taskPair = taskPairCache_[producer][consumer];
  TaskPairPayloadCache &payload = taskPair.payload;
  if (payload.computed)
    return &payload;

  payload.computed = true;
  payload.found = false;
  payload.valid = true;
  payload.unknown_payload = false;
  payload.overflowed = false;
  payload.payload_bits = 0;
  payload.edge_indices.clear();
  for (auto indexed : llvm::enumerate(edge_graph_.getEdges())) {
    const TaskEdge &edge = indexed.value();
    taskflow::TaskflowTaskOp producerTask = edge.producer;
    taskflow::TaskflowTaskOp consumerTask = edge.consumer;
    if (producerTask.getOperation() != producer ||
        consumerTask.getOperation() != consumer)
      continue;
    payload.found = true;
    if (edge.kind != TaskEdgeKind::Value && edge.kind != TaskEdgeKind::Raw)
      continue;
    payload.edge_indices.push_back(static_cast<uint32_t>(indexed.index()));
    if (!edge.payload_bits) {
      payload.valid = false;
      payload.unknown_payload = true;
      return &payload;
    }
    if (*edge.payload_bits > UINT64_MAX - payload.payload_bits) {
      payload.valid = false;
      payload.overflowed = true;
      return &payload;
    }
    payload.payload_bits += *edge.payload_bits;
  }
  return &payload;
}

bool InterTaskNetworkCommunicationModel::getEndpointPathMetrics(
    int source_row, int source_col, int destination_row, int destination_col,
    EndpointPathMetrics &metrics) const {
  metrics = {};
  const int rows = network_.getRows();
  const int columns = network_.getColumns();
  if (rows <= 0 || columns <= 0 || source_row < 0 || source_row >= rows ||
      source_col < 0 || source_col >= columns || destination_row < 0 ||
      destination_row >= rows || destination_col < 0 ||
      destination_col >= columns)
    return false;

  const uint64_t nodeCount64 = static_cast<uint64_t>(rows) *
                               static_cast<uint64_t>(columns);
  if (nodeCount64 == 0 ||
      nodeCount64 > static_cast<uint64_t>(
                        std::numeric_limits<size_t>::max()))
    return false;
  const size_t nodeCount = static_cast<size_t>(nodeCount64);
  if (nodeCount > std::numeric_limits<size_t>::max() / nodeCount)
    return false;
  const size_t pairCount = nodeCount * nodeCount;
  if (endpointPathCache_.size() != pairCount)
    endpointPathCache_.assign(pairCount, EndpointPathMetrics{});

  const size_t sourceId = static_cast<size_t>(source_row * columns + source_col);
  const size_t destinationId =
      static_cast<size_t>(destination_row * columns + destination_col);
  EndpointPathMetrics &cached =
      endpointPathCache_[sourceId * nodeCount + destinationId];
  if (!cached.computed) {
    TaskNetworkCoordinate source{source_row, source_col};
    TaskNetworkCoordinate destination{destination_row, destination_col};
    std::optional<uint64_t> latency =
        minimumPathLatency(network_, source, destination);
    std::optional<uint64_t> bandwidth =
        maximumPathBandwidth(network_, source, destination);
    cached.computed = true;
    if (latency && bandwidth && *bandwidth != 0) {
      cached.reachable = true;
      cached.latency_cycles = *latency;
      cached.bandwidth_bits_per_cycle = *bandwidth;
    }
  }
  metrics = cached;
  return true;
}

bool InterTaskNetworkCommunicationModel::getCachedEndpointLowerBound(
    Operation *producer, Operation *consumer, int source_row, int source_col,
    int destination_row, int destination_col, int64_t &cycles,
    std::string &error) const {
  error.clear();
  cycles = 0;

  const int rows = network_.getRows();
  const int columns = network_.getColumns();
  const bool validEndpoint =
      rows > 0 && columns > 0 && source_row >= 0 && source_row < rows &&
      source_col >= 0 && source_col < columns && destination_row >= 0 &&
      destination_row < rows && destination_col >= 0 &&
      destination_col < columns;
  const TaskPairPayloadCache *payload =
      getTaskPairPayloadCache(producer, consumer);

  // Preserve the zero-payload contract before endpoint validation.  A
  // control-only edge has no physical transfer, even for an otherwise
  // malformed coordinate supplied by a caller.
  if (payload && payload->valid && payload->found &&
      network_.getLocalBandwidthBitsPerCycle() != 0 &&
      payload->payload_bits == 0) {
    if (!validEndpoint)
      return true;
  }

  TaskPairCache *taskPair = nullptr;
  EndpointBoundCacheEntry *cached = nullptr;
  if (producer && consumer && validEndpoint) {
    taskPair = &taskPairCache_[producer][consumer];
    const uint64_t nodeCount64 = static_cast<uint64_t>(rows) *
                                 static_cast<uint64_t>(columns);
    if (nodeCount64 == 0 ||
        nodeCount64 > static_cast<uint64_t>(
                          std::numeric_limits<size_t>::max())) {
      error = "communication grid is too large for endpoint lower-bound cache";
      return false;
    }
    const size_t nodeCount = static_cast<size_t>(nodeCount64);
    if (nodeCount > std::numeric_limits<size_t>::max() / nodeCount) {
      error = "communication grid is too large for endpoint lower-bound cache";
      return false;
    }
    const size_t pairCount = nodeCount * nodeCount;
    if (taskPair->endpoint_bounds.size() != pairCount)
      taskPair->endpoint_bounds.assign(pairCount, EndpointBoundCacheEntry{});
    const size_t sourceId =
        static_cast<size_t>(source_row * columns + source_col);
    const size_t destinationId =
        static_cast<size_t>(destination_row * columns + destination_col);
    cached = &taskPair->endpoint_bounds[sourceId * nodeCount + destinationId];
    if (cached->computed) {
      cycles = cached->cycles;
      error = cached->error;
      return cached->success;
    }
    cached->computed = true;
    cached->success = false;
    cached->cycles = 0;
    cached->error.clear();
  }

  auto fail = [&](llvm::StringRef message) {
    error = message.str();
    if (cached)
      cached->error = error;
    return false;
  };

  if (!payload || !payload->valid) {
    if (payload && payload->unknown_payload)
      return fail("unproved payload for endpoint communication lower bound");
    if (payload && payload->overflowed)
      return fail("unproved payload for endpoint communication lower bound");
    return fail("unproved payload for endpoint communication lower bound");
  }
  if (!payload->found || network_.getLocalBandwidthBitsPerCycle() == 0)
    return fail("invalid edge or bandwidth for endpoint lower bound");
  if (!validEndpoint) {
    error = "endpoint lower-bound query is outside the communication grid";
    return false;
  }

  if (payload->payload_bits == 0) {
    cached->success = true;
    cycles = 0;
    return true;
  }

  EndpointPathMetrics metrics;
  if (!getEndpointPathMetrics(source_row, source_col, destination_row,
                              destination_col, metrics))
    return fail("endpoint lower-bound query is outside the communication grid");
  // A disconnected pair is not a proof that the candidate is illegal.  The
  // exact ready-cycle replay will reject it if every endpoint pair is
  // disconnected; this lower-bound query contributes no bound for this pair.
  if (!metrics.reachable) {
    cached->success = true;
    cycles = 0;
    return true;
  }

  const uint64_t transmission =
      (payload->payload_bits - 1) / metrics.bandwidth_bits_per_cycle + 1;
  if (metrics.latency_cycles >
      std::numeric_limits<uint64_t>::max() - transmission)
    return fail("endpoint communication lower bound overflows unsigned cycles");
  const uint64_t bound = metrics.latency_cycles + transmission;
  if (bound > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return fail("endpoint communication lower bound overflows signed cycles");
  cached->success = true;
  cached->cycles = static_cast<int64_t>(bound);
  cycles = cached->cycles;
  return true;
}

bool InterTaskNetworkCommunicationModel::getTransferLowerBound(
    Operation *producer, Operation *consumer, int64_t &cycles,
    std::string &error) const {
  error.clear();
  cycles = 0;
  const TaskPairPayloadCache *payload =
      getTaskPairPayloadCache(producer, consumer);
  if (!payload || !payload->valid) {
    error = "unproved payload for communication lower bound";
    return false;
  }
  if (!payload->found || network_.getLocalBandwidthBitsPerCycle() == 0) {
    error = "invalid edge or bandwidth for lower bound";
    return false;
  }
  // A control-only edge has no physical transfer.  This also preserves the
  // runtime contract that a zero payload returns producer_finish immediately,
  // even when its endpoints are remote and have no directed network path.
  if (payload->payload_bits == 0)
    return true;

  std::optional<uint64_t> best;
  for (int sourceRow = 0; sourceRow < network_.getRows(); ++sourceRow) {
    for (int sourceColumn = 0; sourceColumn < network_.getColumns();
         ++sourceColumn) {
      for (int destinationRow = 0; destinationRow < network_.getRows();
           ++destinationRow) {
        for (int destinationColumn = 0;
             destinationColumn < network_.getColumns(); ++destinationColumn) {
          EndpointPathMetrics metrics;
          if (!getEndpointPathMetrics(sourceRow, sourceColumn, destinationRow,
                                      destinationColumn, metrics) ||
              !metrics.reachable)
            continue;
          int64_t candidate = 0;
          std::string candidateError;
          if (!getCachedEndpointLowerBound(
                  producer, consumer, sourceRow, sourceColumn, destinationRow,
                  destinationColumn, candidate, candidateError))
            continue;
          if (candidate < 0)
            continue;
          const uint64_t candidateUnsigned = static_cast<uint64_t>(candidate);
          if (!best || candidateUnsigned < *best)
            best = candidateUnsigned;
        }
      }
    }
  }
  if (!best || *best > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    error = "communication lower bound has no representable endpoint route";
    return false;
  }
  // Same-coordinate traffic is intentionally included above: its path
  // latency is exactly zero and its bandwidth is the explicit local channel.
  // Do not add a positive handoff constant that the ready-cycle model does
  // not charge.  For remote routes, shortest latency and widest bandwidth are
  // individually optimistic; reservations can only make the actual ready
  // cycle later.  Taking the minimum over all endpoint pairs is therefore
  // admissible for every legal placement.
  cycles = static_cast<int64_t>(*best);
  return true;
}

bool InterTaskNetworkCommunicationModel::getTransferLowerBoundForEndpoints(
    Operation *producer, Operation *consumer, int source_row, int source_col,
    int destination_row, int destination_col, int64_t &cycles,
    std::string &error) const {
  return getCachedEndpointLowerBound(producer, consumer, source_row, source_col,
                                     destination_row, destination_col, cycles,
                                     error);
}

void InterTaskNetworkCommunicationModel::resetReservations() {
  for (auto &intervals : committedLinkIntervals_)
    intervals.clear();
  for (auto &intervals : committedLocalIntervals_)
    intervals.clear();
  trialLinkIntervals_ = committedLinkIntervals_;
  trialLocalIntervals_ = committedLocalIntervals_;
  committedTransfers_.clear();
  trialTransfers_.clear();
}

void InterTaskNetworkCommunicationModel::beginTrial() {
  trialLinkIntervals_ = committedLinkIntervals_;
  trialLocalIntervals_ = committedLocalIntervals_;
  trialTransfers_.clear();
}

void InterTaskNetworkCommunicationModel::finishTrial(bool commit) {
  if (commit) {
    committedLinkIntervals_ = trialLinkIntervals_;
    committedLocalIntervals_ = trialLocalIntervals_;
    committedTransfers_.append(trialTransfers_.begin(), trialTransfers_.end());
  }
  trialTransfers_.clear();
}

bool InterTaskNetworkCommunicationModel::getCanonicalReservationState(
    std::string &state, std::string &error) const {
  error.clear();
  state.clear();
  // Keep this representation deliberately structural and unhashed.  The
  // search memo is scoped to one fixed graph/shape invocation, so preserving
  // every resource index and interval is sufficient and avoids a second
  // identity scheme.  Transfer records are intentionally omitted: route
  // selection reads only these interval vectors, while records are diagnostic.
  state = "inter-task-network-reservations-v1|";
  auto appendIntervals = [&](const char *label, const auto &resources) {
    state += label;
    state += ':';
    state += std::to_string(resources.size());
    state += '|';
    for (size_t resource = 0; resource < resources.size(); ++resource) {
      state += std::to_string(resource);
      state += '[';
      for (const auto &interval : resources[resource]) {
        state += std::to_string(interval.first);
        state += ',';
        state += std::to_string(interval.second);
        state += ';';
      }
      state += ']';
    }
    state += '|';
  };
  appendIntervals("committed-link", committedLinkIntervals_);
  appendIntervals("trial-link", trialLinkIntervals_);
  appendIntervals("committed-local", committedLocalIntervals_);
  appendIntervals("trial-local", trialLocalIntervals_);
  return true;
}

bool InterTaskNetworkCommunicationModel::captureReservationCheckpoint(
    ::mlir::taskflow::TaskCommunicationReservationCheckpoint &checkpoint,
    std::string &error) const {
  error.clear();
  checkpoint = {};
  auto snapshot = std::make_shared<ReservationCheckpointState>();
  // Route selection observes all four interval sets.  Transfer histories are
  // copied as well because they are exposed for diagnostics and must remain
  // identical after a speculative child returns.
  snapshot->committedLinkIntervals = committedLinkIntervals_;
  snapshot->trialLinkIntervals = trialLinkIntervals_;
  snapshot->committedLocalIntervals = committedLocalIntervals_;
  snapshot->trialLocalIntervals = trialLocalIntervals_;
  snapshot->committedTransfers = committedTransfers_;
  snapshot->trialTransfers = trialTransfers_;
  checkpoint.owner = this;
  checkpoint.typeTag = kReservationCheckpointTypeTag;
  checkpoint.state = std::move(snapshot);
  return true;
}

bool InterTaskNetworkCommunicationModel::restoreReservationCheckpoint(
    const ::mlir::taskflow::TaskCommunicationReservationCheckpoint &checkpoint,
    std::string &error) {
  error.clear();
  if (checkpoint.owner != this) {
    error = "communication reservation checkpoint belongs to another model";
    return false;
  }
  if (checkpoint.typeTag != kReservationCheckpointTypeTag) {
    error = "communication reservation checkpoint has an unknown type";
    return false;
  }
  if (!checkpoint.state) {
    error = "communication reservation checkpoint has no state";
    return false;
  }
  const auto snapshot = std::static_pointer_cast<const ReservationCheckpointState>(
      checkpoint.state);
  if (!snapshot) {
    error = "communication reservation checkpoint has invalid state";
    return false;
  }
  const size_t linkCount = network_.getLinks().size();
  const size_t localCount = static_cast<size_t>(network_.getRows()) *
                            static_cast<size_t>(network_.getColumns());
  if (snapshot->committedLinkIntervals.size() != linkCount ||
      snapshot->trialLinkIntervals.size() != linkCount ||
      snapshot->committedLocalIntervals.size() != localCount ||
      snapshot->trialLocalIntervals.size() != localCount) {
    error = "communication reservation checkpoint has incompatible dimensions";
    return false;
  }
  committedLinkIntervals_ = snapshot->committedLinkIntervals;
  trialLinkIntervals_ = snapshot->trialLinkIntervals;
  committedLocalIntervals_ = snapshot->committedLocalIntervals;
  trialLocalIntervals_ = snapshot->trialLocalIntervals;
  committedTransfers_ = snapshot->committedTransfers;
  trialTransfers_ = snapshot->trialTransfers;
  return true;
}

static bool checkedAddCycle(int64_t lhs, uint64_t rhs, int64_t &result) {
  if (lhs < 0 ||
      rhs > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      lhs > std::numeric_limits<int64_t>::max() - static_cast<int64_t>(rhs))
    return false;
  result = lhs + static_cast<int64_t>(rhs);
  return true;
}

bool InterTaskNetworkCommunicationModel::routeTransfer(
    Operation *producer, Operation *consumer, TaskNetworkCoordinate source,
    TaskNetworkCoordinate destination, uint64_t payload_bits,
    ArrayRef<uint32_t> edge_indices, int64_t producer_finish, bool reserve,
    int64_t &ready_cycle, std::string &error) {
  const int rows = network_.getRows();
  const int columns = network_.getColumns();
  auto inBounds = [&](TaskNetworkCoordinate coordinate) {
    return coordinate.row >= 0 && coordinate.row < rows &&
           coordinate.column >= 0 && coordinate.column < columns;
  };
  if (!inBounds(source) || !inBounds(destination) || producer_finish < 0) {
    error = "communication transfer endpoint or start cycle is invalid";
    return false;
  }

  TransferRecord record;
  record.producer = producer;
  record.consumer = consumer;
  record.source = source;
  record.destination = destination;
  record.payloadBits = payload_bits;
  record.edgeIndices.assign(edge_indices.begin(), edge_indices.end());
  if (source == destination) {
    const uint64_t bandwidth = network_.getLocalBandwidthBitsPerCycle();
    if (bandwidth == 0) {
      error = "local communication bandwidth must be positive";
      return false;
    }
    const uint64_t cycles =
        payload_bits == 0 ? 0 : ((payload_bits - 1) / bandwidth) + 1;
    record.pathLatencyCycles = 0;
    record.bottleneckBandwidthBitsPerCycle = bandwidth;
    record.transferCycles = cycles;
    const int local_index = source.row * columns + source.column;
    int64_t start = producer_finish;
    while (true) {
      int64_t finish = 0;
      if (!checkedAddCycle(start, cycles, finish)) {
        error = "local communication ready cycle overflows signed 64-bit";
        return false;
      }
      int64_t next_start = start;
      for (auto [occupied_start, occupied_end] :
           trialLocalIntervals_[local_index]) {
        if (start < occupied_end && finish > occupied_start)
          next_start = std::max(next_start, occupied_end);
      }
      if (next_start == start) {
        ready_cycle = finish;
        break;
      }
      start = next_start;
    }
    record.links.push_back({0, start, ready_cycle, true, source});
    record.readyCycle = ready_cycle;
    if (reserve) {
      auto &intervals = trialLocalIntervals_[local_index];
      intervals.push_back({start, ready_cycle});
      llvm::sort(intervals);
      trialTransfers_.push_back(std::move(record));
    }
    return true;
  }

  // Once a route reaches the immutable endpoint lower bound, no later route
  // can improve its finish.  The DFS visits simple paths in link-index
  // lexicographic order, so stopping at the first exact-bound route also
  // preserves the deterministic tie winner.  If the bound is unavailable or
  // not attained, the complete route search below remains unchanged.
  int64_t endpointLowerBoundFinish = 0;
  bool endpointLowerBoundFinishKnown = false;
  int64_t endpointLowerBoundCycles = 0;
  std::string endpointLowerBoundError;
  if (getTransferLowerBoundForEndpoints(
          producer, consumer, source.row, source.column, destination.row,
          destination.column, endpointLowerBoundCycles,
          endpointLowerBoundError) &&
      endpointLowerBoundCycles >= 0)
    endpointLowerBoundFinishKnown = checkedAddCycle(
        producer_finish, static_cast<uint64_t>(endpointLowerBoundCycles),
        endpointLowerBoundFinish);

  const int node_count = rows * columns;
  const int source_id = source.row * columns + source.column;
  const int destination_id = destination.row * columns + destination.column;
  ArrayRef<InterTaskNetworkLink> links = network_.getLinks();
  struct RouteChoice {
    SmallVector<uint32_t> links;
    int64_t start = 0;
    int64_t finish = std::numeric_limits<int64_t>::max();
    uint64_t latency = 0;
    uint64_t bottleneck = 0;
    uint64_t duration = 0;
  };
  RouteChoice best;
  SmallVector<uint32_t> path;
  std::vector<uint8_t> visited(node_count, 0);
  visited[source_id] = 1;
  bool arithmetic_error = false;
  bool search_budget_exceeded = false;
  bool exact_lower_bound_reached = false;
  uint64_t search_steps = 0;
  constexpr uint64_t kRouteSearchStepBudget = 100000;

  // A prefix bound is only enabled when it cannot hide an error that the
  // original exhaustive search would report.  In particular, a route with a
  // zero-bandwidth link, a wrapping latency sum, or a signed ready-cycle
  // overflow must still be visited by the original evaluator.  The
  // source-to-destination reachability filter below is deliberately
  // conservative: it may disable the bound for an edge that cannot occur in
  // one simple path, but it never enables pruning in that case.
  bool prefixPruningEnabled = true;
  std::vector<uint8_t> reachableFromSource(node_count, 0);
  std::vector<uint8_t> canReachDestination(node_count, 0);
  std::vector<int> worklist;
  worklist.push_back(source_id);
  reachableFromSource[source_id] = 1;
  while (!worklist.empty()) {
    const int current = worklist.back();
    worklist.pop_back();
    for (const InterTaskNetworkLink &link : links) {
      const int linkSource = link.source.row * columns + link.source.column;
      const int next = link.destination.row * columns + link.destination.column;
      if (linkSource != current || next < 0 || next >= node_count ||
          reachableFromSource[next])
        continue;
      reachableFromSource[next] = 1;
      worklist.push_back(next);
    }
  }
  worklist.push_back(destination_id);
  canReachDestination[destination_id] = 1;
  while (!worklist.empty()) {
    const int current = worklist.back();
    worklist.pop_back();
    for (const InterTaskNetworkLink &link : links) {
      const int linkSource = link.source.row * columns + link.source.column;
      const int next = link.destination.row * columns + link.destination.column;
      if (next != current || linkSource < 0 || linkSource >= node_count ||
          canReachDestination[linkSource])
        continue;
      canReachDestination[linkSource] = 1;
      worklist.push_back(linkSource);
    }
  }

  uint64_t relevantLatencyUpperBound = 0;
  for (const InterTaskNetworkLink &link : links) {
    const int linkSource = link.source.row * columns + link.source.column;
    const int next = link.destination.row * columns + link.destination.column;
    if (linkSource < 0 || linkSource >= node_count || next < 0 ||
        next >= node_count) {
      prefixPruningEnabled = false;
      continue;
    }
    if (!reachableFromSource[linkSource] || !canReachDestination[next])
      continue;
    if (link.bandwidth_bits_per_cycle == 0 ||
        relevantLatencyUpperBound >
            std::numeric_limits<uint64_t>::max() - link.latency_cycles) {
      prefixPruningEnabled = false;
      continue;
    }
    relevantLatencyUpperBound += link.latency_cycles;
  }
  uint64_t routeDurationUpperBound = 0;
  if (prefixPruningEnabled) {
    // For positive bandwidth, ceil(payload / bandwidth) is no larger than
    // payload.  This upper bound covers every simple route and keeps the
    // overflow check independent of the path chosen by the DFS.
    if (relevantLatencyUpperBound >
            std::numeric_limits<uint64_t>::max() - payload_bits)
      prefixPruningEnabled = false;
    else {
      routeDurationUpperBound = relevantLatencyUpperBound + payload_bits;
      if (routeDurationUpperBound >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
                static_cast<uint64_t>(producer_finish))
        prefixPruningEnabled = false;
    }
  }
  if (prefixPruningEnabled) {
    // A reservation can move a candidate's start beyond producer_finish.  If
    // any relevant occupied interval is close enough to signed INT64_MAX
    // that even the conservative duration upper bound could overflow, keep
    // the original evaluator on every route.  This is intentionally
    // fail-open; it preserves its reservation-induced overflow failure.
    for (size_t linkIndex = 0; linkIndex < links.size(); ++linkIndex) {
      const InterTaskNetworkLink &link = links[linkIndex];
      const int linkSource = link.source.row * columns + link.source.column;
      const int next = link.destination.row * columns + link.destination.column;
      if (linkSource < 0 || linkSource >= node_count || next < 0 ||
          next >= node_count || !reachableFromSource[linkSource] ||
          !canReachDestination[next])
        continue;
      for (auto [occupiedStart, occupiedEnd] : trialLinkIntervals_[linkIndex]) {
        (void)occupiedStart;
        if (occupiedEnd < 0 ||
            static_cast<uint64_t>(occupiedEnd) >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
                    routeDurationUpperBound) {
          prefixPruningEnabled = false;
          break;
        }
      }
      if (!prefixPruningEnabled)
        break;
    }
  }

  std::vector<uint8_t> remainingMetricsComputed(node_count, 0);
  std::vector<std::optional<uint64_t>> remainingLatency(node_count);
  std::vector<std::optional<uint64_t>> remainingBandwidth(node_count);

  auto prefixLowerBoundExceedsBest = [&](int current,
                                         uint64_t prefixLatency,
                                         uint64_t prefixBottleneck) {
    if (!prefixPruningEnabled || best.links.empty())
      return false;
    if (!remainingMetricsComputed[current]) {
      const TaskNetworkCoordinate coordinate{current / columns, current % columns};
      remainingLatency[current] =
          minimumPathLatency(network_, coordinate, destination);
      remainingBandwidth[current] =
          maximumPathBandwidth(network_, coordinate, destination);
      remainingMetricsComputed[current] = 1;
    }
    if (!remainingLatency[current] || !remainingBandwidth[current])
      return false;
    const uint64_t bottleneck =
        std::min(prefixBottleneck, *remainingBandwidth[current]);
    uint64_t transmissionLowerBound = 0;
    if (payload_bits != 0) {
      if (bottleneck == 0)
        return false;
      transmissionLowerBound = (payload_bits - 1) / bottleneck + 1;
    }
    if (prefixLatency >
        std::numeric_limits<uint64_t>::max() - *remainingLatency[current])
      return false;
    uint64_t durationLowerBound = prefixLatency + *remainingLatency[current];
    if (durationLowerBound >
        std::numeric_limits<uint64_t>::max() - transmissionLowerBound)
      return false;
    durationLowerBound += transmissionLowerBound;
    int64_t lowerBoundFinish = 0;
    if (!checkedAddCycle(producer_finish, durationLowerBound,
                         lowerBoundFinish))
      return false;
    // The DFS order makes this subtree lexicographically later than the
    // incumbent.  Keep equal bounds so equal-finish paths still reach the
    // existing lexicographic tie comparison.
    return lowerBoundFinish > best.finish;
  };

  auto evaluate = [&]() {
    uint64_t latency = 0;
    uint64_t bottleneck = UINT64_MAX;
    for (uint32_t link_index : path) {
      const InterTaskNetworkLink &link = links[link_index];
      if (link.bandwidth_bits_per_cycle == 0 ||
          latency > UINT64_MAX - link.latency_cycles) {
        arithmetic_error = true;
        return;
      }
      latency += link.latency_cycles;
      bottleneck = std::min(bottleneck, link.bandwidth_bits_per_cycle);
    }
    const uint64_t transmission =
        payload_bits == 0 ? 0 : ((payload_bits - 1) / bottleneck) + 1;
    if (latency > UINT64_MAX - transmission) {
      arithmetic_error = true;
      return;
    }
    const uint64_t duration = latency + transmission;
    int64_t start = producer_finish;
    int64_t finish = 0;
    // Finds the first common half-open gap across all route links. This stays
    // exact even when transfers are committed out of start-time order.
    while (true) {
      if (!checkedAddCycle(start, duration, finish)) {
        arithmetic_error = true;
        return;
      }
      int64_t next_start = start;
      for (uint32_t link_index : path) {
        for (auto [occupied_start, occupied_end] :
             trialLinkIntervals_[link_index]) {
          if (start < occupied_end && finish > occupied_start)
            next_start = std::max(next_start, occupied_end);
        }
      }
      if (next_start == start)
        break;
      start = next_start;
    }
    if (finish < best.finish ||
        (finish == best.finish &&
         std::lexicographical_compare(path.begin(), path.end(),
                                      best.links.begin(), best.links.end())))
      best = {path, start, finish, latency, bottleneck, duration};
    if (endpointLowerBoundFinishKnown && best.finish == endpointLowerBoundFinish)
      exact_lower_bound_reached = true;
  };

  std::function<void(int, uint64_t, uint64_t, bool)> enumerate =
      [&](int current, uint64_t prefixLatency, uint64_t prefixBottleneck,
          bool prefixLatencyOverflow) {
    if (search_budget_exceeded || exact_lower_bound_reached)
      return;
    if (++search_steps > kRouteSearchStepBudget) {
      search_budget_exceeded = true;
      return;
    }
    if (current == destination_id) {
      evaluate();
      return;
    }
    if (!prefixLatencyOverflow &&
        prefixLowerBoundExceedsBest(current, prefixLatency, prefixBottleneck))
      return;
    for (auto indexed : llvm::enumerate(links)) {
      if (exact_lower_bound_reached)
        return;
      const InterTaskNetworkLink &link = indexed.value();
      const int link_source = link.source.row * columns + link.source.column;
      const int next = link.destination.row * columns + link.destination.column;
      if (link_source != current || visited[next])
        continue;
      uint64_t nextPrefixLatency = prefixLatency;
      bool nextPrefixLatencyOverflow = prefixLatencyOverflow;
      if (!nextPrefixLatencyOverflow) {
        if (nextPrefixLatency >
            std::numeric_limits<uint64_t>::max() - link.latency_cycles)
          nextPrefixLatencyOverflow = true;
        else
          nextPrefixLatency += link.latency_cycles;
      }
      const uint64_t nextPrefixBottleneck =
          std::min(prefixBottleneck, link.bandwidth_bits_per_cycle);
      visited[next] = 1;
      path.push_back(static_cast<uint32_t>(indexed.index()));
      enumerate(next, nextPrefixLatency, nextPrefixBottleneck,
                nextPrefixLatencyOverflow);
      path.pop_back();
      visited[next] = 0;
    }
  };
  enumerate(source_id, 0, std::numeric_limits<uint64_t>::max(), false);
  if (search_budget_exceeded) {
    error = "communication route search exceeds its deterministic budget";
    return false;
  }
  if (arithmetic_error) {
    error = "communication route cost overflows its 64-bit contract";
    return false;
  }
  if (best.links.empty()) {
    error = "communication network has no directed path between endpoints";
    return false;
  }

  for (uint32_t link_index : best.links) {
    record.links.push_back({link_index, best.start, best.finish, false, {}});
    if (reserve) {
      auto &intervals = trialLinkIntervals_[link_index];
      intervals.push_back({best.start, best.finish});
      llvm::sort(intervals);
    }
  }
  ready_cycle = best.finish;
  record.pathLatencyCycles = best.latency;
  record.bottleneckBandwidthBitsPerCycle = best.bottleneck;
  record.transferCycles = best.duration;
  record.readyCycle = best.finish;
  if (reserve)
    trialTransfers_.push_back(std::move(record));
  return true;
}

bool InterTaskNetworkCommunicationModel::getTransferReadyCycle(
    Operation *producer, Operation *consumer, int source_row, int source_col,
    int destination_row, int destination_col, int64_t producer_finish,
    bool reserve, int64_t &ready_cycle, std::string &error) {
  error.clear();
  ready_cycle = producer_finish;
  if (!producer || !consumer) {
    error = "communication transfer query has a null task endpoint";
    return false;
  }
  if (source_row < 0 || source_col < 0 || destination_row < 0 ||
      destination_col < 0) {
    error = "communication transfer endpoint is outside the grid";
    return false;
  }

  const TaskPairPayloadCache *payload =
      getTaskPairPayloadCache(producer, consumer);
  if (!payload) {
    error = "communication transfer query does not match a typed task edge";
    return false;
  }
  if (!payload->valid) {
    if (payload->unknown_payload) {
      error = "communication transfer has an unknown Value/RAW payload";
      return false;
    }
    if (payload->overflowed) {
      error = "communication payload sum overflows unsigned 64-bit bits";
      return false;
    }
    error = "communication transfer has an unproved payload";
    return false;
  }
  if (!payload->found) {
    error = "communication transfer query does not match a typed task edge";
    return false;
  }
  if (payload->payload_bits == 0)
    return true;
  return routeTransfer(producer, consumer, {source_row, source_col},
                       {destination_row, destination_col},
                       payload->payload_bits, payload->edge_indices,
                       producer_finish, reserve, ready_cycle, error);
}

FailureOr<std::unique_ptr<InterTaskNetworkCommunicationModel>>
createInterTaskNetworkCommunicationModel(func::FuncOp function, int grid_rows,
                                         int grid_columns, std::string &error) {
  error.clear();
  TaskEdgeGraphOptions edge_options;
  edge_options.require_payload = true;
  edge_options.read_control_predecessors = true;
  FailureOr<TaskEdgeGraph> edge_graph =
      buildTaskEdgeGraph(function, edge_options, error);
  if (failed(edge_graph))
    return failure();

  FailureOr<std::optional<InterTaskNetworkSpec>> network =
      loadInterTaskNetworkSpec(::mlir::amoeba::getNeuraArchitectureSpecFile(),
                               true, error);
  if (failed(network) || !network->has_value()) {
    if (succeeded(network) && !network->has_value())
      error = "communication model requires an inter_task_network section";
    return failure();
  }
  if (!network->value().validateForGrid(grid_rows, grid_columns, error))
    return failure();

  return std::make_unique<InterTaskNetworkCommunicationModel>(
      std::move(*edge_graph), std::move(**network));
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir
