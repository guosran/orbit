//===- CommunicationExactScheduleSpace.cpp - Network-aware 4x4 search ----===//

#include "CommunicationExactScheduleSpace.h"
#include "RectanglePairTimeBounds.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {
namespace {
using WideCycle = __int128_t;

// All schedule durations are nonnegative.  Keep intermediate dependency,
// window and area*cycle calculations in 128 bits so an overflow can never
// turn an admissible lower bound into an unjustified rejection.
static int64_t clampCycle(WideCycle value) {
  if (value <= 0)
    return value < 0 ? std::numeric_limits<int64_t>::min() : 0;
  const WideCycle maximum =
      static_cast<WideCycle>(std::numeric_limits<int64_t>::max());
  return value >= maximum ? std::numeric_limits<int64_t>::max()
                          : static_cast<int64_t>(value);
}

static WideCycle addCycles(WideCycle lhs, WideCycle rhs) {
  const WideCycle maximum = (static_cast<WideCycle>(1) << 126) - 1;
  if (lhs >= maximum || rhs >= maximum || lhs > maximum - rhs)
    return maximum;
  return lhs + rhs;
}

static int64_t strictSuccessor(int64_t value) {
  return value == std::numeric_limits<int64_t>::max() ? value : value + 1;
}

// A candidate origin changes network endpoints. Reconstruct each branch's
// reservations from its dispatch prefix; beginTrial/finishTrial(false) alone
// cannot undo transfers already committed by an ancestor.
bool replayPrefixAndGetReady(
    const std::vector<ExactScheduleTask> &tasks,
    const std::vector<ExactSchedulePlacement> &placed, unsigned index,
    int row, int col, FixedScheduleCommunication &communication,
    int64_t &candidateReady, std::string &error) {
  std::vector<const ExactSchedulePlacement *> byTask(tasks.size(), nullptr);
  for (const auto &placement : placed)
    byTask[placement.task] = &placement;
  ExactSchedulePlacement candidate{index, row, col, 0, tasks[index].duration};
  byTask[index] = &candidate;
  communication.resetReservations();
  for (unsigned ordinal = 0; ordinal <= placed.size(); ++ordinal) {
    const unsigned current =
        ordinal == placed.size() ? index : placed[ordinal].task;
    std::vector<unsigned> predecessors;
    if (!communication.getPredecessors(current, predecessors, error))
      return false;
    const std::set<unsigned> declared(tasks[current].predecessors.begin(),
                                      tasks[current].predecessors.end());
    const std::set<unsigned> observed(predecessors.begin(), predecessors.end());
    if (observed != declared || observed.size() != predecessors.size()) {
      error = "fixed schedule dependencies disagree with communication graph";
      return false;
    }
    int64_t ready = 0;
    communication.beginTrial();
    for (unsigned predecessor : predecessors) {
      const auto *source = byTask[predecessor];
      const auto *destination = byTask[current];
      if (!source) {
        communication.finishTrial(false);
        error = "communication predecessor is absent from dispatch prefix";
        return false;
      }
      const auto &sourceShape = tasks[predecessor];
      const auto &destinationShape = tasks[current];
      int64_t bestReady = std::numeric_limits<int64_t>::max();
      int bestSourceRow = -1, bestSourceCol = -1;
      int bestDestinationRow = -1, bestDestinationCol = -1;
      for (int sr = source->row; sr < source->row + sourceShape.rows; ++sr)
        for (int sc = source->col; sc < source->col + sourceShape.cols; ++sc)
          for (int dr = destination->row;
               dr < destination->row + destinationShape.rows; ++dr)
            for (int dc = destination->col;
                 dc < destination->col + destinationShape.cols; ++dc) {
              int64_t endpointReady = 0;
              if (!communication.getTransferReadyCycle(
                      predecessor, current, sr, sc, dr, dc, source->end,
                      false, endpointReady, error)) {
                communication.finishTrial(false);
                return false;
              }
              if (endpointReady < source->end) {
                communication.finishTrial(false);
                error = "communication model returns a cycle before producer finish";
                return false;
              }
              if (endpointReady < bestReady) {
                bestReady = endpointReady;
                bestSourceRow = sr;
                bestSourceCol = sc;
                bestDestinationRow = dr;
                bestDestinationCol = dc;
              }
            }
      if (bestSourceRow < 0) {
        communication.finishTrial(false);
        error = "communication transfer has no rectangle endpoint";
        return false;
      }
      if (!communication.getTransferReadyCycle(
              predecessor, current, bestSourceRow, bestSourceCol,
              bestDestinationRow, bestDestinationCol, source->end, true,
              bestReady, error)) {
        communication.finishTrial(false);
        return false;
      }
      if (bestReady < source->end) {
        communication.finishTrial(false);
        error = "communication reservation precedes producer finish";
        return false;
      }
      ready = std::max(ready, bestReady);
    }
    communication.finishTrial(true);
    if (ordinal == placed.size()) {
      candidateReady = ready;
    } else if (ready > placed[ordinal].start) {
      error = "dispatch prefix starts before communication ready";
      return false;
    }
  }
  return true;
}

// The checkpoint path enters with reservations for exactly `placed`. Restore
// has already selected that prefix, so only the candidate's incoming edges
// need to be queried and committed. Keep the endpoint loops and strict `<`
// tie rule identical to replayPrefixAndGetReady().
bool reserveCandidateIncomingTransfers(
    const std::vector<ExactScheduleTask> &tasks,
    const std::vector<ExactSchedulePlacement> &placed, unsigned index,
    int row, int col, FixedScheduleCommunication &communication,
    int64_t &candidateReady, std::string &error) {
  std::vector<const ExactSchedulePlacement *> byTask(tasks.size(), nullptr);
  for (const auto &placement : placed)
    byTask[placement.task] = &placement;
  ExactSchedulePlacement candidate{index, row, col, 0, tasks[index].duration};
  byTask[index] = &candidate;

  std::vector<unsigned> predecessors;
  if (!communication.getPredecessors(index, predecessors, error))
    return false;
  const std::set<unsigned> declared(tasks[index].predecessors.begin(),
                                    tasks[index].predecessors.end());
  const std::set<unsigned> observed(predecessors.begin(), predecessors.end());
  if (observed != declared || observed.size() != predecessors.size()) {
    error = "fixed schedule dependencies disagree with communication graph";
    return false;
  }

  int64_t ready = 0;
  communication.beginTrial();
  for (unsigned predecessor : predecessors) {
    const auto *source = byTask[predecessor];
    if (!source) {
      communication.finishTrial(false);
      error = "communication predecessor is absent from dispatch prefix";
      return false;
    }
    const auto &sourceShape = tasks[predecessor];
    const auto &destinationShape = tasks[index];
    int64_t bestReady = std::numeric_limits<int64_t>::max();
    int bestSourceRow = -1, bestSourceCol = -1;
    int bestDestinationRow = -1, bestDestinationCol = -1;
    for (int sr = source->row; sr < source->row + sourceShape.rows; ++sr)
      for (int sc = source->col; sc < source->col + sourceShape.cols; ++sc)
        for (int dr = row; dr < row + destinationShape.rows; ++dr)
          for (int dc = col; dc < col + destinationShape.cols; ++dc) {
            int64_t endpointReady = 0;
            if (!communication.getTransferReadyCycle(
                    predecessor, index, sr, sc, dr, dc, source->end, false,
                    endpointReady, error)) {
              communication.finishTrial(false);
              return false;
            }
            if (endpointReady < source->end) {
              communication.finishTrial(false);
              error =
                  "communication model returns a cycle before producer finish";
              return false;
            }
            if (endpointReady < bestReady) {
              bestReady = endpointReady;
              bestSourceRow = sr;
              bestSourceCol = sc;
              bestDestinationRow = dr;
              bestDestinationCol = dc;
            }
          }
    if (bestSourceRow < 0) {
      communication.finishTrial(false);
      error = "communication transfer has no rectangle endpoint";
      return false;
    }
    if (!communication.getTransferReadyCycle(
            predecessor, index, bestSourceRow, bestSourceCol,
            bestDestinationRow, bestDestinationCol, source->end, true,
            bestReady, error)) {
      communication.finishTrial(false);
      return false;
    }
    if (bestReady < source->end) {
      communication.finishTrial(false);
      error = "communication reservation precedes producer finish";
      return false;
    }
    ready = std::max(ready, bestReady);
  }
  communication.finishTrial(true);
  candidateReady = ready;
  return true;
}

struct Search {
  int gridRows;
  int gridCols;
  const std::vector<ExactScheduleTask> &tasks;
  ExactScheduleLimits limits;
  CommunicationStartPolicy startPolicy;
  int64_t maxMakespan;
  FixedScheduleCommunication &communication;
  const std::function<void(const std::vector<ExactSchedulePlacement> &,
                           const std::vector<unsigned> &,
                           const FixedScheduleScore &)> &emit;
  ExactScheduleResult &result;
  CommunicationScheduleCertificate &certificate;
  std::string &error;
  std::set<std::string> seen;
  std::set<std::string> seenPaths;
  std::vector<std::pair<int64_t, std::string>> bestFive;
  std::vector<ExactSchedulePlacement> placed;
  std::vector<bool> scheduled;
  std::chrono::steady_clock::time_point began =
      std::chrono::steady_clock::now();
  CommunicationDispatchPolicy dispatchPolicy =
      CommunicationDispatchPolicy::AllReady;
  std::function<int64_t()> externalUpperBound;
  std::vector<int64_t> successorTail;
  std::vector<std::vector<int64_t>> edgeLowerBound;
  // Keys are exact structural strings rather than hashes.  The cache lives
  // only for this fixed graph/shape Search invocation.
  std::map<std::string, int64_t> stateMemo;
  bool stateMemoUsable = false;
  mutable bool stateMemoCutoffViolation = false;
  mutable int64_t lastObservedUpperBound =
      std::numeric_limits<int64_t>::max();

  std::vector<std::pair<unsigned, unsigned>> incompatibleRectangles = {};

  // This cache is deliberately scoped to one Search object.  The object owns
  // one immutable task/shape vector and one fixed architecture/payload
  // communication model, so the structural key below cannot cross-contaminate
  // another graph, shape assignment, or network.  It stores only the exact
  // minimum of endpoint lower bounds; trial/committed reservations never
  // enter the key or the value.  Providers without the explicit static-bound
  // capability use the original endpoint loop below.
  struct EndpointRectangleCacheEntry {
    bool computed = false;
    bool proven = false;
    int64_t cycles = 0;
  };
  using EndpointRectangleCacheKey =
      std::tuple<unsigned, unsigned, int, int, int, int, int, int, int, int>;
  std::map<EndpointRectangleCacheKey, EndpointRectangleCacheEntry>
      endpointRectangleLowerBounds;

  // The optimistic end vector depends on the fixed task/shape graph, the
  // fixed communication model, and the already placed half-open occupancy.
  // Keep an exact structural key for that occupancy so equivalent prefixes
  // reached through different dispatch orders can reuse the complete vector.
  // This memo is enabled only for providers whose endpoint lower bounds are
  // explicitly reservation-independent.  It intentionally does not include
  // the incumbent cutoff or any bound/window result: those are recomputed at
  // every visit because the cutoff can change between visits.
  struct OptimisticEndPlacementKey {
    unsigned task = 0;
    int row = 0;
    int col = 0;
    int64_t start = 0;
    int64_t end = 0;

    bool operator<(const OptimisticEndPlacementKey &other) const {
      return std::tie(task, row, col, start, end) <
             std::tie(other.task, other.row, other.col, other.start,
                      other.end);
    }
  };
  using OptimisticEndsKey = std::vector<OptimisticEndPlacementKey>;
  struct OptimisticEndsCacheEntry {
    std::vector<int64_t> releases;
    std::vector<int64_t> ends;
  };
  std::map<OptimisticEndsKey, OptimisticEndsCacheEntry> optimisticEndsMemo;
  static constexpr std::size_t kOptimisticEndsCacheCapacity = 100000;

  OptimisticEndsKey makeOptimisticEndsKey() const {
    OptimisticEndsKey key;
    key.reserve(placed.size());
    for (const auto &placement : placed)
      key.push_back({placement.task, placement.row, placement.col,
                     placement.start, placement.end});
    std::sort(key.begin(), key.end());
    return key;
  }

  bool getCachedEndpointRectangleLowerBound(
      unsigned producer, unsigned consumer, int sourceRow, int sourceCol,
      int destinationRow, int destinationCol, int64_t &cycles) {
    cycles = 0;
    if (!communication.supportsStaticEndpointLowerBoundCache())
      return false;

    // Include the fixed task dimensions in addition to the task/origin key.
    // Search lifetime already binds the architecture, edge payloads and model
    // instance; these fields make accidental reuse after shape mutation fail
    // as a distinct key rather than silently reusing a bound.
    const EndpointRectangleCacheKey key{
        producer,
        consumer,
        sourceRow,
        sourceCol,
        destinationRow,
        destinationCol,
        tasks[producer].rows,
        tasks[producer].cols,
        tasks[consumer].rows,
        tasks[consumer].cols};
    auto [it, inserted] = endpointRectangleLowerBounds.emplace(
        key, EndpointRectangleCacheEntry{});
    EndpointRectangleCacheEntry &entry = it->second;
    if (!inserted && entry.computed) {
      if (entry.proven)
        cycles = entry.cycles;
      return entry.proven;
    }

    entry.computed = true;
    entry.proven = true;
    entry.cycles = std::numeric_limits<int64_t>::max();
    for (int sourceEndpointRow = sourceRow;
         sourceEndpointRow < sourceRow + tasks[producer].rows;
         ++sourceEndpointRow)
      for (int sourceEndpointCol = sourceCol;
           sourceEndpointCol < sourceCol + tasks[producer].cols;
           ++sourceEndpointCol)
        for (int destinationEndpointRow = destinationRow;
             destinationEndpointRow < destinationRow + tasks[consumer].rows;
             ++destinationEndpointRow)
          for (int destinationEndpointCol = destinationCol;
               destinationEndpointCol < destinationCol + tasks[consumer].cols;
               ++destinationEndpointCol) {
            int64_t pairBound = 0;
            std::string endpointError;
            if (!communication.getTransferLowerBoundForEndpoints(
                    producer, consumer, sourceEndpointRow, sourceEndpointCol,
                    destinationEndpointRow, destinationEndpointCol, pairBound,
                    endpointError)) {
              // Preserve the existing fail-closed semantics: an unavailable
              // endpoint proof falls back to the universal edge bound.  Do
              // not turn a disconnected/unknown endpoint into rejection.
              entry.proven = false;
              entry.cycles = 0;
              return false;
            }
            entry.cycles = std::min(entry.cycles, pairBound);
          }
    if (entry.cycles == std::numeric_limits<int64_t>::max()) {
      entry.proven = false;
      entry.cycles = 0;
      return false;
    }
    cycles = entry.cycles;
    return true;
  }

  void configureRectanglePairs() {
    for (unsigned i = 0; i < tasks.size(); ++i)
      for (unsigned j = i + 1; j < tasks.size(); ++j)
        if (!rectanglePairCanOverlapInTime(gridRows, gridCols,
                                          tasks[i].rows, tasks[i].cols,
                                          tasks[j].rows, tasks[j].cols))
          incompatibleRectangles.emplace_back(i, j);
  }

  void configureStateMemo() {
    result.stateMemoCanonicalStateAvailable =
        communication.supportsCanonicalReservationState();
    result.stateMemoExternalCutoffMonotonic =
        limits.externalUpperBoundMonotonic;
    stateMemoUsable =
        limits.enableStateMemo && result.stateMemoCanonicalStateAvailable &&
        (!externalUpperBound || limits.externalUpperBoundMonotonic);
    result.stateMemoEnabled = stateMemoUsable;
    if (stateMemoUsable)
      lastObservedUpperBound = maxMakespan;
  }

  bool memoEnabled() const {
    return stateMemoUsable && !stateMemoCutoffViolation;
  }

  bool makeStateMemoKey(std::string &key) {
    key.clear();
    if (!memoEnabled())
      return false;
    std::string reservationState;
    std::string reservationError;
    if (!communication.getCanonicalReservationState(reservationState,
                                                    reservationError)) {
      // Canonical state is an optional optimization contract.  If a model
      // cannot provide it at runtime, fail closed by disabling only the memo.
      stateMemoUsable = false;
      result.stateMemoEnabled = false;
      return false;
    }
    std::vector<ExactSchedulePlacement> canonical = placed;
    std::sort(canonical.begin(), canonical.end(),
              [](const ExactSchedulePlacement &lhs,
                 const ExactSchedulePlacement &rhs) {
                if (lhs.task != rhs.task)
                  return lhs.task < rhs.task;
                if (lhs.row != rhs.row)
                  return lhs.row < rhs.row;
                if (lhs.col != rhs.col)
                  return lhs.col < rhs.col;
                if (lhs.start != rhs.start)
                  return lhs.start < rhs.start;
                return lhs.end < rhs.end;
              });
    std::ostringstream state;
    state << "fixed-schedule-state-v1|placed:" << canonical.size() << ':';
    for (const auto &placement : canonical)
      state << placement.task << '@' << placement.row << ',' << placement.col
            << ':' << placement.start << '-' << placement.end << ';';
    state << "|reservations:" << reservationState;
    key = state.str();
    return true;
  }

  bool tryStateMemo(const std::string &key) {
    if (!memoEnabled())
      return false;
    auto it = stateMemo.find(key);
    if (it == stateMemo.end())
      return false;
    const int64_t cutoff = upperBound();
    if (stateMemoCutoffViolation)
      return false;
    // Equality is intentionally retained: another dispatch prefix can carry
    // a distinct path key at the same makespan and still enter top-K.
    if (it->second <= cutoff)
      return false;
    ++result.stateMemoHits;
    ++result.stateMemoPrunedSubtrees;
    return true;
  }

  void insertStateMemo(const std::string &key, int64_t terminalMinimum) {
    if (!memoEnabled() || key.empty())
      return;
    const int64_t cutoff = upperBound();
    if (stateMemoCutoffViolation)
      return;
    const int64_t lowerBound =
        std::min(terminalMinimum, strictSuccessor(cutoff));
    auto existing = stateMemo.find(key);
    if (existing != stateMemo.end()) {
      // A smaller bound is the conservative merge if the same state was
      // reached under a later incumbent cutoff.
      existing->second = std::min(existing->second, lowerBound);
      result.stateMemoEntries = stateMemo.size();
      return;
    }
    if (limits.stateMemoCapacity &&
        stateMemo.size() >= limits.stateMemoCapacity) {
      ++result.stateMemoCapacitySkips;
      return;
    }
    stateMemo.emplace(key, lowerBound);
    result.stateMemoEntries = stateMemo.size();
  }

  void computeSuccessorTails() {
    successorTail.assign(tasks.size(), -1);
    std::function<int64_t(unsigned)> tail = [&](unsigned i) {
      if (successorTail[i] >= 0)
        return successorTail[i];
      int64_t value = 0;
      for (unsigned j = 0; j < tasks.size(); ++j)
        if (std::find(tasks[j].predecessors.begin(),
                      tasks[j].predecessors.end(), i) != tasks[j].predecessors.end()) {
          int64_t suffix = tail(j);
          suffix = suffix > std::numeric_limits<int64_t>::max() - edgeLowerBound[j][i]
              ? std::numeric_limits<int64_t>::max() : suffix + edgeLowerBound[j][i];
          value = std::max(value,
              suffix > std::numeric_limits<int64_t>::max() - tasks[j].duration
                  ? std::numeric_limits<int64_t>::max() : suffix + tasks[j].duration);
        }
      return successorTail[i] = value;
    };
    for (unsigned i = 0; i < tasks.size(); ++i)
      tail(i);
  }

  int64_t latestStart(unsigned index) const {
    int64_t bound = upperBound();
    int64_t tail = limits.rankingTopK ? successorTail[index] : 0;
    if (tail > bound || tasks[index].duration > bound - tail)
      return -1;
    return bound - tail - tasks[index].duration;
  }

  int64_t upperBound() const {
    int64_t bound = maxMakespan > 0 ? maxMakespan : std::numeric_limits<int64_t>::max();
    if (!limits.rankingTopK)
      return bound;
    if (bestFive.size() >= limits.rankingTopK)
      bound = std::min(bound, bestFive.back().first);
    if (externalUpperBound)
      bound = std::min(bound, externalUpperBound());
    if (stateMemoUsable && bound > lastObservedUpperBound) {
      // A completed-subtree lower bound is sound only for a monotonic
      // cutoff.  The actual score pass sets the explicit proof flag; if its
      // callback violates that contract, fail closed for the whole result
      // instead of allowing a memo hit to affect an uncertified ranking.
      stateMemoCutoffViolation = true;
      result.stateMemoCutoffViolation = true;
      result.stateMemoEnabled = false;
      if (result.reason.empty())
        result.reason = "non-monotonic-external-cutoff";
    }
    if (bound < lastObservedUpperBound)
      lastObservedUpperBound = bound;
    return bound;
  }

  // Compute a lower bound on each not-yet-dispatched task's end.  A task may
  // use any legal origin, so the minimum over all origins is admissible.  The
  // already placed half-open reservations are fixed in this branch; jumping
  // over them with firstNonconflictingStart is therefore also admissible for
  // both OrderDerived and AllInteger searches.  A placed predecessor also
  // fixes its source rectangle.  In that case use the minimum proven bound
  // over source/consumer endpoint pairs for each candidate origin; an
  // unplaced predecessor retains the universal edge bound.
  bool computeOptimisticEnds(std::vector<int64_t> &releases,
                             std::vector<int64_t> &ends) {
    // Custom/dynamic providers may make endpoint lower bounds depend on
    // committed or trial reservations.  Preserve the original computation
    // completely for them; canonical reservation state alone is insufficient
    // to prove this memo safe.
    const bool staticEndpointCache =
        communication.supportsStaticEndpointLowerBoundCache();
    result.optimisticEndsMemoEnabled = staticEndpointCache;
    OptimisticEndsKey cacheKey;
    if (staticEndpointCache) {
      cacheKey = makeOptimisticEndsKey();
      auto cached = optimisticEndsMemo.find(cacheKey);
      if (cached != optimisticEndsMemo.end()) {
        ++result.optimisticEndsMemoHits;
        releases = cached->second.releases;
        ends = cached->second.ends;
        return true;
      }
    }

    ++result.optimisticEndsMemoComputations;
    releases.assign(tasks.size(), 0);
    ends.assign(tasks.size(), -1);
    std::vector<bool> active(tasks.size(), false);
    for (const auto &p : placed) {
      releases[p.task] = p.start;
      ends[p.task] = p.end;
    }

    std::function<int64_t(unsigned)> earliestEnd = [&](unsigned i) {
      if (ends[i] >= 0)
        return ends[i];
      if (active[i])
        return std::numeric_limits<int64_t>::max();
      active[i] = true;
      int64_t bestStart = std::numeric_limits<int64_t>::max();
      int64_t bestEnd = std::numeric_limits<int64_t>::max();
      for (int row = 0; row < gridRows; ++row)
        for (int col = 0; col < gridCols; ++col) {
          if (row + tasks[i].rows > gridRows ||
              col + tasks[i].cols > gridCols)
            continue;

          WideCycle release = 0;
          for (unsigned predecessor : tasks[i].predecessors) {
            const WideCycle predecessorEnd = earliestEnd(predecessor);
            int64_t delay = edgeLowerBound[i][predecessor];
            auto source = std::find_if(
                placed.begin(), placed.end(),
                [&](const ExactSchedulePlacement &placement) {
                  return placement.task == predecessor;
                });
            if (source != placed.end()) {
              int64_t endpointDelay = std::numeric_limits<int64_t>::max();
              bool endpointProof = true;
              if (staticEndpointCache) {
                endpointProof = getCachedEndpointRectangleLowerBound(
                    predecessor, i, source->row, source->col, row, col,
                    endpointDelay);
              } else {
                for (int sourceRow = source->row;
                     sourceRow < source->row + tasks[predecessor].rows;
                     ++sourceRow)
                  for (int sourceCol = source->col;
                       sourceCol < source->col + tasks[predecessor].cols;
                       ++sourceCol)
                    for (int destinationRow = row;
                         destinationRow < row + tasks[i].rows;
                         ++destinationRow)
                      for (int destinationCol = col;
                           destinationCol < col + tasks[i].cols;
                           ++destinationCol) {
                        int64_t pairBound = 0;
                        std::string endpointError;
                        if (!communication.getTransferLowerBoundForEndpoints(
                                predecessor, i, sourceRow, sourceCol,
                                destinationRow, destinationCol, pairBound,
                                endpointError)) {
                          endpointProof = false;
                          break;
                        }
                        endpointDelay = std::min(endpointDelay, pairBound);
                      }
              }
              // A model may have no endpoint-specific proof.  Falling back to
              // the universal bound keeps this branch admissible and avoids
              // treating an unavailable/disconnected endpoint as illegal.
              if (endpointProof &&
                  endpointDelay != std::numeric_limits<int64_t>::max())
                delay = endpointDelay;
            }
            release = std::max(
                release, addCycles(predecessorEnd, static_cast<WideCycle>(delay)));
          }
          const int64_t dependencyReady = clampCycle(release);
          const int64_t start = firstNonconflictingStart(
              row, col, dependencyReady, tasks[i]);
          if (start == std::numeric_limits<int64_t>::max())
            continue;
          bestStart = std::min(bestStart, start);
          const WideCycle end = addCycles(start, tasks[i].duration);
          bestEnd = std::min(bestEnd, clampCycle(end));
        }
      active[i] = false;
      releases[i] = bestStart;
      ends[i] = bestEnd;
      return bestEnd;
    };

    for (unsigned i = 0; i < tasks.size(); ++i)
      earliestEnd(i);

    // Insert only a complete vector.  A full cache never limits candidates or
    // search; it merely disables future insertion until an existing key hits.
    // INT64_MAX is an intentional overflow/no-legal-origin sentinel and is
    // stored verbatim with the rest of the vector.
    if (staticEndpointCache) {
      if (optimisticEndsMemo.size() < kOptimisticEndsCacheCapacity) {
        optimisticEndsMemo.emplace(std::move(cacheKey),
                                   OptimisticEndsCacheEntry{releases, ends});
        result.optimisticEndsMemoEntries = optimisticEndsMemo.size();
      } else {
        ++result.optimisticEndsMemoCapacitySkips;
      }
    }
    return true;
  }

  // Apply an energetic resource lower bound to the fixed branch.  For every
  // unplaced task, [release, deadline] is an optimistic window: release is
  // obtained from dependency/edge lower bounds and the task's best legal
  // origin, while deadline is upperBound()-successorTail.  In an arbitrary
  // completion, at least
  //   max(0, duration - max(0, a-release) - max(0, deadline-b))
  // cycles of that task must execute inside [a,b).  Multiplying by its
  // rectangle area and adding the exact occupied part of placed tasks gives a
  // necessary amount of tile work.  If it exceeds 16*(b-a), no completion can
  // meet the incumbent bound.  The event set includes all window boundaries
  // and duration offsets, so the check is conservative even for integer
  // starts; it never imposes a candidate or symmetry limit.
  bool exceedsResourceWindowBound(
      const std::vector<int64_t> &releases) {
    const int64_t bound = upperBound();
    const WideCycle wideBound = bound;
    std::vector<int64_t> deadlines(tasks.size(), 0);
    std::vector<int64_t> events{0, bound};
    for (unsigned i = 0; i < tasks.size(); ++i) {
      if (scheduled[i]) {
        for (const auto &p : placed)
          if (p.task == i) {
            events.push_back(p.start);
            events.push_back(p.end);
            break;
          }
        continue;
      }
      const WideCycle deadline =
          wideBound - static_cast<WideCycle>(successorTail[i]);
      if (deadline < 0)
        return true;
      deadlines[i] = clampCycle(deadline);
      const WideCycle completion =
          addCycles(releases[i], tasks[i].duration);
      if (completion > deadline)
        return true;
      events.push_back(releases[i]);
      events.push_back(deadlines[i]);
      events.push_back(clampCycle(completion));
      if (deadline >= tasks[i].duration)
        events.push_back(clampCycle(
            deadline - static_cast<WideCycle>(tasks[i].duration)));
    }
    std::sort(events.begin(), events.end());
    events.erase(std::unique(events.begin(), events.end()), events.end());
    const WideCycle capacityPerTile =
        static_cast<WideCycle>(gridRows) * gridCols;
    for (std::size_t left = 0; left < events.size(); ++left)
      for (std::size_t right = left + 1; right < events.size(); ++right) {
        const WideCycle a = events[left];
        const WideCycle b = events[right];
        if (b <= a)
          continue;
        WideCycle demand = 0;
        for (const auto &p : placed) {
          const WideCycle overlap = std::max<WideCycle>(
              0, std::min<WideCycle>(p.end, b) - std::max<WideCycle>(p.start, a));
          demand = addCycles(
              demand, overlap * static_cast<WideCycle>(tasks[p.task].rows) *
                          tasks[p.task].cols);
        }
        for (unsigned i = 0; i < tasks.size(); ++i) {
          if (scheduled[i])
            continue;
          const WideCycle release = releases[i];
          const WideCycle deadline = deadlines[i];
          const WideCycle before = std::max<WideCycle>(0, a - release);
          const WideCycle after = std::max<WideCycle>(0, deadline - b);
          const WideCycle mandatory = std::max<WideCycle>(
              0, static_cast<WideCycle>(tasks[i].duration) - before - after);
          demand = addCycles(
              demand, mandatory * static_cast<WideCycle>(tasks[i].rows) *
                          tasks[i].cols);
        }
        const WideCycle capacity = (b - a) * capacityPerTile;
        if (demand > capacity)
          return true;
      }
    return false;
  }

  // Ignore future communication reservations and future cell conflicts:
  // both can only delay a task.  The dependency/location lower bound and the
  // energetic window test are necessary conditions, so rejecting here cannot
  // remove a legal AllInteger or OrderDerived schedule under the cutoff.
  bool exceedsBound() {
    if (!limits.rankingTopK)
      return false;
    std::vector<int64_t> releases;
    std::vector<int64_t> ends;
    computeOptimisticEnds(releases, ends);
    const int64_t bound = upperBound();
    for (unsigned i = 0; i < tasks.size(); ++i)
      if (ends[i] > bound) {
        ++result.boundRejectedBranches;
        return true;
      }
    for (auto [i, j] : incompatibleRectangles) {
      // Placed tasks already constrain every optimistic origin/start through
      // firstNonconflictingStart. This strengthens only unplaced pairs.
      if (scheduled[i] || scheduled[j])
        continue;
      const int64_t deadlineI = bound - successorTail[i];
      const int64_t deadlineJ = bound - successorTail[j];
      if (!rectanglePairHasSerialWindow(releases[i], tasks[i].duration,
                                       deadlineI, releases[j], tasks[j].duration,
                                       deadlineJ)) {
        ++result.boundRejectedBranches;
        ++result.geometryTimeBoundRejectedBranches;
        return true;
      }
    }
    if (exceedsResourceWindowBound(releases)) {
      ++result.boundRejectedBranches;
      return true;
    }
    return false;
  }

  // Every start between start and the end of an intersecting reservation
  // overlaps that reservation. Jump over exactly those forbidden integers.
  int64_t firstNonconflictingStart(int row, int col, int64_t start,
                                  const ExactScheduleTask &task) const {
    while (start <= std::numeric_limits<int64_t>::max() - task.duration) {
      int64_t next = start;
      for (const auto &p : placed) {
        const auto &shape = tasks[p.task];
        if (start < p.end && p.start < start + task.duration &&
            row < p.row + shape.rows && p.row < row + task.rows &&
            col < p.col + shape.cols && p.col < col + task.cols)
          next = std::max(next, p.end);
      }
      if (next == start)
        return start;
      start = next;
    }
    return std::numeric_limits<int64_t>::max();
  }

  // Seeds are ordinary verified candidates, never a restriction on the
  // subsequent exact search. Different origin preferences supply early ties.
  bool seed() {
    if (!limits.seedIncumbents || !limits.rankingTopK)
      return true;
    for (unsigned preference = 0; preference < std::min<unsigned>(
             limits.seedPreferences, gridRows * gridCols); ++preference) {
      placed.clear();
      std::fill(scheduled.begin(), scheduled.end(), false);
      while (placed.size() < tasks.size()) {
        if (stopped())
          return false;
        ++result.expandedNodes;
        unsigned index = 0;
        for (; index < tasks.size(); ++index)
          if (!scheduled[index] &&
              std::all_of(tasks[index].predecessors.begin(),
                          tasks[index].predecessors.end(),
                          [&](unsigned p) { return scheduled[p]; }))
            break;
        const auto &task = tasks[index];
        ExactSchedulePlacement best{index, -1, -1, 0,
                                    std::numeric_limits<int64_t>::max()};
        for (unsigned offset = 0;
             offset < static_cast<unsigned>(gridRows * gridCols); ++offset) {
          const unsigned origin = (preference + offset) % (gridRows * gridCols);
          int row = origin / gridCols, col = origin % gridCols;
          if (row + task.rows > gridRows || col + task.cols > gridCols)
            continue;
          int64_t ready = 0;
          if (!replayPrefixAndGetReady(tasks, placed, index, row, col,
                                      communication, ready, error)) {
            result.reason = "communication-replay-failed";
            return false;
          }
          const int64_t start = firstNonconflictingStart(row, col, ready, task);
          if (start > maxMakespan - task.duration)
            continue;
          if (start + task.duration < best.end)
            best = {index, row, col, start, start + task.duration};
        }
        if (best.row < 0)
          break;
        placed.push_back(best);
        scheduled[index] = true;
      }
      if (placed.size() == tasks.size()) {
        const uint64_t before = result.acceptedPaths;
        int64_t seedMinimum = std::numeric_limits<int64_t>::max();
        if (!visit(seedMinimum))
          return false;
        result.seedSchedules += result.acceptedPaths - before;
      }
      if (result.seedSchedules >= limits.rankingTopK)
        break;
    }
    placed.clear();
    std::fill(scheduled.begin(), scheduled.end(), false);
    // A seed visit verifies a complete schedule and leaves the communication
    // model committed to that schedule.  The subsequent root visit has an
    // empty placed prefix, so restore the matching empty reservation state.
    if (startPolicy == CommunicationStartPolicy::OrderDerived &&
        communication.supportsReservationCheckpoint())
      communication.resetReservations();
    return true;
  }

  bool stopped() {
    if (stateMemoCutoffViolation)
      return true;
    if (result.expandedNodes >= limits.maxExpandedNodes) {
      result.reason = "max-expanded-nodes";
      return true;
    }
    if (limits.maxMilliseconds &&
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - began).count() >=
            static_cast<int64_t>(limits.maxMilliseconds)) {
      result.reason = "max-milliseconds";
      return true;
    }
    return false;
  }

  bool canPlace(int row, int col, int64_t start, int64_t end,
                const ExactScheduleTask &task) {
    if (row < 0 || col < 0 || row + task.rows > gridRows ||
        col + task.cols > gridCols) {
      ++result.outOfBoundsRejectedBranches;
      return false;
    }
    for (const auto &other : placed) {
      if (start >= other.end || end <= other.start)
        continue;
      const auto &shape = tasks[other.task];
      if (row < other.row + shape.rows && other.row < row + task.rows &&
          col < other.col + shape.cols && other.col < col + task.cols) {
        ++result.overlapRejectedBranches;
        return false;
      }
    }
    return true;
  }

  bool visit(int64_t &subtreeMinimum) {
    subtreeMinimum = std::numeric_limits<int64_t>::max();
    if (stopped())
      return false;
    ++result.expandedNodes;
    std::string memoKey;
    const bool memoCandidate = makeStateMemoKey(memoKey);
    if (memoCandidate && tryStateMemo(memoKey))
      return true;
    if (stateMemoCutoffViolation)
      return false;
    FixedScheduleCommunication::ReservationCheckpoint parentCheckpoint;
    bool checkpointPath = false;
    auto restoreParent = [&]() -> bool {
      if (!checkpointPath)
        return true;
      std::string restoreError;
      if (communication.restoreReservationCheckpoint(parentCheckpoint,
                                                     restoreError))
        return true;
      error = restoreError.empty()
                  ? "communication reservation checkpoint restore failed"
                  : std::move(restoreError);
      result.reason = "communication-checkpoint-restore-failed";
      stateMemoUsable = false;
      result.stateMemoEnabled = false;
      return false;
    };
    auto finish = [&]() {
      if (!restoreParent())
        return false;
      if (memoCandidate && !stateMemoCutoffViolation)
        insertStateMemo(memoKey, subtreeMinimum);
      // insertStateMemo() observes the external cutoff as well.  Propagate a
      // late monotonicity violation so a terminal finish cannot report a
      // complete result after the memo contract has failed.
      return !stateMemoCutoffViolation;
    };
    if (exceedsBound()) {
      if (stateMemoCutoffViolation)
        return false;
      // No verified terminal is needed here: the final cutoff+1 term records
      // the admissible proof for every branch rejected by this bound.
      return finish();
    }
    if (stateMemoCutoffViolation)
      return false;
    // The checkpoint path is exact for one candidate start in OrderDerived.
    // AllInteger and EventNormalized may visit several starts for the same
    // origin after one incoming-transfer reservation; retain their original
    // full-prefix replay until those multi-start reservations are handled.
    checkpointPath = startPolicy == CommunicationStartPolicy::OrderDerived &&
                     communication.supportsReservationCheckpoint();
    if (checkpointPath) {
      std::string checkpointError;
      if (!communication.captureReservationCheckpoint(parentCheckpoint,
                                                       checkpointError))
        checkpointPath = false;
    }
    if (placed.size() == tasks.size()) {
      std::vector<unsigned> order;
      order.reserve(placed.size());
      for (const auto &placement : placed)
        order.push_back(placement.task);
      auto score = verifyAndScoreFixedSchedule(
          gridRows, gridCols, tasks, placed, order, communication);
      // The independent verifier is authoritative and resets/reserves the
      // provider while checking the terminal. Restore this node's exact
      // prefix before any return so the caller's branch state is unchanged.
      if (!restoreParent())
        return false;
      if (!score.valid) {
        error = score.rejection;
        result.reason = "fixed-schedule-replay-disagreement";
        return false;
      }
      // Include every verified terminal, including a path already present in
      // seenPaths.  A duplicate emission is not evidence that its score is
      // absent from this subtree's lower bound.
      subtreeMinimum = score.predictedMakespan;
      std::vector<ExactSchedulePlacement> canonical = placed;
      std::sort(canonical.begin(), canonical.end(),
                [](const auto &a, const auto &b) { return a.task < b.task; });
      std::ostringstream key;
      for (const auto &p : canonical)
        key << p.task << ':' << p.row << ',' << p.col << '@'
            << p.start << '-' << p.end << ';';
      std::ostringstream pathKey;
      pathKey << key.str() << " order:";
      for (unsigned task : order)
        pathKey << task << ',';
      if (!seenPaths.insert(pathKey.str()).second)
        return finish();
      if (result.acceptedPaths == std::numeric_limits<uint64_t>::max()) {
        result.reason = "count-overflow";
        return false;
      }
      ++result.acceptedPaths;
      if (seen.insert(key.str()).second) {
        if (result.uniqueSchedules == std::numeric_limits<uint64_t>::max()) {
          result.reason = "count-overflow";
          return false;
        }
        ++result.uniqueSchedules;
      }
      bestFive.push_back({score.predictedMakespan, pathKey.str()});
      std::sort(bestFive.begin(), bestFive.end());
      if (bestFive.size() > (limits.rankingTopK ? limits.rankingTopK : 5))
        bestFive.pop_back();
      emit(canonical, order, score);
      return finish();
    }

    for (unsigned index = 0; index < tasks.size(); ++index) {
      if (scheduled[index])
        continue;
      const auto &task = tasks[index];
      bool readyTask = true;
      for (unsigned predecessor : task.predecessors)
        if (!scheduled[predecessor]) {
          readyTask = false;
          break;
        }
      if (!readyTask)
        continue;
      if (dispatchPolicy == CommunicationDispatchPolicy::CanonicalSmallestReady) {
        bool earlierReady = false;
        for (unsigned earlier = 0; earlier < index; ++earlier) {
          if (scheduled[earlier])
            continue;
          const auto &earlierTask = tasks[earlier];
          if (std::all_of(earlierTask.predecessors.begin(),
                          earlierTask.predecessors.end(),
                          [&](unsigned predecessor) {
                            return scheduled[predecessor];
                          })) {
            earlierReady = true;
            break;
          }
        }
        if (earlierReady)
          continue;
      }
      for (int row = 0; row < gridRows; ++row)
        for (int col = 0; col < gridCols; ++col) {
          if (row + task.rows > gridRows || col + task.cols > gridCols) {
            ++result.outOfBoundsRejectedBranches;
            continue;
          }
          if (checkpointPath) {
            std::string restoreError;
            if (!communication.restoreReservationCheckpoint(parentCheckpoint,
                                                            restoreError)) {
              // A failed optional restore cannot make a candidate unsafe: use
              // the original reset-and-replay path for this and later peers.
              checkpointPath = false;
              stateMemoUsable = false;
              result.stateMemoEnabled = false;
            }
          }
          int64_t ready = 0;
          const bool prepared = checkpointPath
                                    ? reserveCandidateIncomingTransfers(
                                          tasks, placed, index, row, col,
                                          communication, ready, error)
                                    : replayPrefixAndGetReady(
                                          tasks, placed, index, row, col,
                                          communication, ready, error);
          if (!prepared) {
            result.reason = "communication-replay-failed";
            return false;
          }
          // For this fixed dispatch prefix and origin, starts before ready
          // violate the production edge and network reservation contract.
          // Release events expose temporal reuse after occupied cells free.
          auto visitStart = [&](int64_t start) -> bool {
            if (start > std::numeric_limits<int64_t>::max() - task.duration) {
              if (!restoreParent())
                return false;
              result.reason = "cycle-overflow";
              return false;
            }
            const int64_t end = start + task.duration;
            if (!canPlace(row, col, start, end, task)) {
              if (checkpointPath) {
                std::string restoreError;
                if (!communication.restoreReservationCheckpoint(
                        parentCheckpoint, restoreError)) {
                  // This candidate is already rejected by cell occupancy. A
                  // failed optional restore only disables the optimization;
                  // the next candidate's legacy replay resets state exactly.
                  checkpointPath = false;
                  stateMemoUsable = false;
                  result.stateMemoEnabled = false;
                }
              }
              return true;
            }
            placed.push_back({index, row, col, start, end,
                start - firstNonconflictingStart(row, col, ready, task)});
            scheduled[index] = true;
            int64_t childMinimum = std::numeric_limits<int64_t>::max();
            if (!visit(childMinimum)) {
              if (!restoreParent())
                return false;
              return false;
            }
            if (!restoreParent())
              return false;
            subtreeMinimum = std::min(subtreeMinimum, childMinimum);
            scheduled[index] = false;
            placed.pop_back();
            return true;
          };
          if (startPolicy == CommunicationStartPolicy::OrderDerived) {
            // Temporal choices are dispatch orders. Times are consequences
            // of dependency/network readiness and half-open cell occupancy.
            const int64_t start = firstNonconflictingStart(row, col, ready, task);
            const int64_t latest = latestStart(index);
            if (stateMemoCutoffViolation) {
              if (!restoreParent())
                return false;
              return false;
            }
            if (start > latest) {
              if (!restoreParent())
                return false;
              ++result.boundRejectedBranches;
              continue;
            }
            if (!visitStart(start)) return false;
          } else if (startPolicy == CommunicationStartPolicy::AllInteger) {
            int64_t start = ready;
            while (start <= latestStart(index)) {
              if (stateMemoCutoffViolation) {
                if (!restoreParent())
                  return false;
                return false;
              }
              if (stopped()) {
                if (!restoreParent())
                  return false;
                return false;
              }
              ++result.expandedNodes;
              const int64_t freeStart = firstNonconflictingStart(row, col, start, task);
              if (freeStart > start) {
                ++result.overlapRejectedBranches;
                start = freeStart;
                continue;
              }
              if (!visitStart(start))
                return false;
              if (start == std::numeric_limits<int64_t>::max())
                break;
              ++start;
            }
          } else {
            std::vector<int64_t> events{ready};
            for (const auto &other : placed)
              if (other.end >= ready)
                events.push_back(other.end);
            std::sort(events.begin(), events.end());
            events.erase(std::unique(events.begin(), events.end()), events.end());
            for (int64_t start : events)
              if (!visitStart(start))
                return false;
          }
        }
    }
    return finish();
  }
};

bool validateInputs(int gridRows, int gridCols,
                    const std::vector<ExactScheduleTask> &tasks,
                    ExactScheduleLimits limits,
                    CommunicationStartPolicy startPolicy,
                    int64_t maxMakespan, std::string &error) {
  if (gridRows != 4 || gridCols != 4 || tasks.empty() ||
      limits.maxExpandedNodes == 0 ||
      limits.maxMilliseconds >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      maxMakespan < 0 ||
      (startPolicy == CommunicationStartPolicy::AllInteger &&
       maxMakespan == 0)) {
    error = "communication scheduling requires nonempty 4x4 input and valid budget";
    return false;
  }
  std::set<std::string> names;
  std::vector<unsigned> indegree(tasks.size(), 0);
  std::vector<std::vector<unsigned>> users(tasks.size());
  for (unsigned i = 0; i < tasks.size(); ++i) {
    const auto &task = tasks[i];
    if (task.name.empty() || !names.insert(task.name).second ||
        task.rows < 1 || task.cols < 1 || task.rows > 4 || task.cols > 4 ||
        task.rows * task.cols > 4 || task.duration < 1) {
      error = "invalid fixed task shape, unique name or predicted duration";
      return false;
    }
    std::set<unsigned> distinct;
    for (unsigned predecessor : task.predecessors)
      if (predecessor >= tasks.size() || predecessor == i ||
          !distinct.insert(predecessor).second) {
        error = "invalid or duplicate task dependency";
        return false;
      }
    indegree[i] = distinct.size();
    for (unsigned predecessor : distinct)
      users[predecessor].push_back(i);
  }
  std::vector<unsigned> frontier;
  for (unsigned i = 0; i < tasks.size(); ++i)
    if (indegree[i] == 0)
      frontier.push_back(i);
  size_t visited = 0;
  while (!frontier.empty()) {
    unsigned next = frontier.back();
    frontier.pop_back();
    ++visited;
    for (unsigned user : users[next])
      if (--indegree[user] == 0)
        frontier.push_back(user);
  }
  if (visited != tasks.size()) {
    error = "task dependency graph has a cycle";
    return false;
  }
  return true;
}
} // namespace

bool enumerateCommunicationScheduleSpace(
    int gridRows, int gridCols, const std::vector<ExactScheduleTask> &tasks,
    ExactScheduleLimits limits, CommunicationStartPolicy startPolicy,
    int64_t maxMakespan, FixedScheduleCommunication &communication,
    const std::function<void(const std::vector<ExactSchedulePlacement> &,
                             const std::vector<unsigned> &,
                             const FixedScheduleScore &)> &emit,
    ExactScheduleResult &result,
    CommunicationScheduleCertificate &certificate, std::string &error,
    CommunicationDispatchPolicy dispatchPolicy,
    const std::function<int64_t()> &externalUpperBound) {
  result = {};
  certificate = {};
  error.clear();
  if (!validateInputs(gridRows, gridCols, tasks, limits, startPolicy,
                      maxMakespan, error))
    return false;
  Search search{gridRows, gridCols, tasks, limits, startPolicy,
                maxMakespan > 0 ? maxMakespan : std::numeric_limits<int64_t>::max(),
                communication, emit, result, certificate, error, {}, {}, {},
                {}, {}, std::chrono::steady_clock::now(),
                CommunicationDispatchPolicy::AllReady, {}, {}, {}, {}, false,
                false, std::numeric_limits<int64_t>::max(), {}, {}, {}};
  search.scheduled.assign(tasks.size(), false);
  search.dispatchPolicy = dispatchPolicy;
  search.externalUpperBound = externalUpperBound;
  search.configureStateMemo();
  search.edgeLowerBound.assign(tasks.size(), std::vector<int64_t>(tasks.size(), 0));
  for (unsigned i = 0; i < tasks.size(); ++i)
    for (unsigned p : tasks[i].predecessors)
      if (!communication.getTransferLowerBound(p, i, search.edgeLowerBound[i][p], error) || search.edgeLowerBound[i][p] < 0)
        return false;
  search.computeSuccessorTails();
  search.configureRectanglePairs();
  int64_t rootMinimum = std::numeric_limits<int64_t>::max();
  // The root has no placed prefix.  This also protects invocations that reuse
  // one production communication model after an earlier shape search whose
  // verifier left committed reservations behind.
  if (startPolicy == CommunicationStartPolicy::OrderDerived &&
      communication.supportsReservationCheckpoint())
    communication.resetReservations();
  result.complete = search.seed() &&
      (startPolicy == CommunicationStartPolicy::SeedOnly ||
       search.visit(rootMinimum));
  if (!result.complete && result.reason.empty())
    result.reason = "interrupted";
  certificate.distinctCandidates = result.acceptedPaths;
  if (result.complete && startPolicy == CommunicationStartPolicy::AllInteger &&
      search.bestFive.size() == 5 &&
      search.bestFive[4].first <= maxMakespan) {
    certificate.topFiveCertifiedWithinFixedShapeGraph = true;
    certificate.fifthPredictedCycles = search.bestFive[4].first;
  }
  return true;
}
} // namespace mlir::amoeba::neura::joint_scheduling
