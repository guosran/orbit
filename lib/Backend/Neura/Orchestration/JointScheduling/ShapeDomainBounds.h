//===- ShapeDomainBounds.h - Sound per-task shape-domain filtering -------===//
//
// This helper filters only shape alternatives which cannot participate in a
// schedule at or below an already verified makespan bound.  It deliberately
// treats unsupported cost rows as opaque coverage blockers: they are retained
// and never used to justify a pruning decision.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_SHAPE_DOMAIN_BOUNDS_H
#define AMOEBA_SHAPE_DOMAIN_BOUNDS_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {

struct ShapeDomainOption {
  // This is the original factored-space ordinal.  It is never rewritten by
  // filtering, so candidate IDs and shape ordinals remain stable.
  std::size_t originalOrdinal = 0;
  int64_t duration = 0;
  unsigned area = 0;
  bool supported = false;
};

struct ShapeDomainPropagationResult {
  bool applied = false;
  bool graphAcyclic = true;
  // Every known option of one task violated a finite necessary condition at
  // upperBound.  Callers must still gate this result with their verified
  // incumbent and provenance policy.
  bool wholeGraphAboveBound = false;
  uint64_t contradictionTask = std::numeric_limits<uint64_t>::max();
  uint64_t iterations = 0;
  uint64_t removedOptions = 0;
  uint64_t windowRejectedOptions = 0;
  std::vector<uint64_t> removedByTask;
};

namespace detail {

inline bool addWide(int64_t lhs, int64_t rhs, int64_t &result) {
  const __int128 value = static_cast<__int128>(lhs) +
                         static_cast<__int128>(rhs);
  if (value > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
    result = std::numeric_limits<int64_t>::max();
    return false;
  }
  if (value < 0) {
    result = 0;
    return false;
  }
  result = static_cast<int64_t>(value);
  return true;
}

inline bool addThree(int64_t first, int64_t second, int64_t third,
                     int64_t &result) {
  int64_t partial = 0;
  if (!addWide(first, second, partial)) {
    result = std::numeric_limits<int64_t>::max();
    return false;
  }
  return addWide(partial, third, result);
}

inline bool multiplyWide(int64_t duration, unsigned area,
                         unsigned __int128 &result) {
  if (duration < 0)
    return false;
  const unsigned __int128 wideDuration =
      static_cast<unsigned __int128>(duration);
  const unsigned __int128 wideArea = static_cast<unsigned __int128>(area);
  const unsigned __int128 maximum =
      std::numeric_limits<unsigned __int128>::max();
  if (wideArea != 0 && wideDuration > maximum / wideArea) {
    result = maximum;
    return false;
  }
  result = wideDuration * wideArea;
  return true;
}

inline bool addUnsignedWide(unsigned __int128 lhs, unsigned __int128 rhs,
                            unsigned __int128 &result) {
  const unsigned __int128 maximum =
      std::numeric_limits<unsigned __int128>::max();
  if (rhs > maximum - lhs) {
    result = maximum;
    return false;
  }
  result = lhs + rhs;
  return true;
}

// Return the mandatory area-work contribution of one option in [start,end).
// The release/deadline inputs are optimistic bounds.  signed __int128 keeps
// subtraction well-defined at the int64 boundaries; a false result means the
// caller must discard this window proof rather than infer a cut.
inline bool mandatoryWindowWork(const ShapeDomainOption &option,
                                int64_t release, int64_t deadline,
                                int64_t start, int64_t end,
                                unsigned __int128 &result) {
  result = 0;
  if (!option.supported || option.duration <= 0)
    return true;
  if (release < 0 || deadline < 0 || start < 0 || end <= start)
    return false;

  __int128 before = static_cast<__int128>(start) -
                    static_cast<__int128>(release);
  __int128 after = static_cast<__int128>(deadline) -
                   static_cast<__int128>(end);
  if (before < 0)
    before = 0;
  if (after < 0)
    after = 0;
  const __int128 remaining = static_cast<__int128>(option.duration) -
                             before - after;
  if (remaining <= 0)
    return true;

  const unsigned __int128 units = static_cast<unsigned __int128>(remaining);
  const unsigned __int128 area = static_cast<unsigned __int128>(option.area);
  const unsigned __int128 maximum =
      std::numeric_limits<unsigned __int128>::max();
  if (area != 0 && units > maximum / area)
    return false;
  result = units * area;
  return true;
}

} // namespace detail

// A selected unsupported row is still part of the exact traversal alphabet,
// but its cost cannot justify a lower-bound cut.  Keep this conversion shared
// with the DFS prefix bound so an unknown row cannot turn into the INT64_MAX
// sentinel used by the catalogue query path.
inline int64_t optimisticShapeDuration(const ShapeDomainOption &option) {
  return option.supported && option.duration > 0 ? option.duration : 0;
}

inline unsigned __int128 optimisticShapeWork(
    const ShapeDomainOption &option) {
  if (!option.supported || option.duration <= 0)
    return 0;
  return static_cast<unsigned __int128>(option.duration) * option.area;
}

// Fixed-point necessary-condition propagation for a DAG.  The lower bound is
// with respect to the scorer's predicted task durations and communication
// edge lower bounds.  It is not an ML legality classifier and it never
// assumes that adding CGRAs improves duration.
//
// For a candidate shape s of task i, reject only when one of these necessary
// conditions is violated:
//   release(i) + duration(i,s) + tail(i) > upperBound
//   sum_j min(area_j*duration_j) with i fixed to s > gridArea*upperBound
//   w_i(s) + sum_{j != i} min_s' w_j(s') > gridArea*(b-a) for an event window
//
// The window contribution is the standard energetic lower bound, where
// w_i(s) is the work that cannot fit before the window or after its optimistic
// deadline.  Event endpoints are only a useful finite set of witnesses.  The
// normal domain result does not claim whole-graph feasibility; it can emit an
// explicit contradiction only when one task's complete known domain has no
// surviving option and no unknown row.
//
// The domains are mutated in place.  Unsupported options remain in place and
// contribute an optimistic zero lower bound; callers can therefore retain
// their existing incomplete-coverage diagnostic without allowing an unknown
// row to justify a cut.  If filtering would remove every supported option of
// a task, that task's round is left unchanged as a fail-closed guard against
// malformed input or an inconsistent incumbent.  When every known option of
// a task is independently rejected by a finite necessary condition and the
// task has no unknown option, wholeGraphAboveBound is set before that guard.
// The last option remains physically present so ordinary domain consumers keep
// their fail-closed behavior; a verified global-bound caller may consume the
// separate contradiction result.
inline ShapeDomainPropagationResult propagateShapeDomains(
    std::vector<std::vector<ShapeDomainOption>> &domains,
    const std::vector<std::vector<unsigned>> &predecessors,
    const std::vector<std::vector<int64_t>> &edgeLowerBounds,
    int64_t upperBound, unsigned gridArea = 16,
    bool enableResourceBounds = true) {
  ShapeDomainPropagationResult result;
  result.removedByTask.assign(domains.size(), 0);
  if (domains.empty() || upperBound <= 0 || gridArea == 0 ||
      predecessors.size() != domains.size() ||
      edgeLowerBounds.size() != domains.size())
    return result;
  if (domains.size() > std::numeric_limits<unsigned>::max())
    return result;

  const unsigned taskCount = static_cast<unsigned>(domains.size());
  std::vector<std::vector<unsigned>> successors(taskCount);
  std::vector<unsigned> indegree(taskCount, 0);
  for (unsigned consumer = 0; consumer < taskCount; ++consumer) {
    if (edgeLowerBounds[consumer].size() != taskCount)
      return result;
    for (unsigned producer : predecessors[consumer]) {
      if (producer >= taskCount || producer == consumer ||
          edgeLowerBounds[consumer][producer] < 0)
        return result;
      successors[producer].push_back(consumer);
      ++indegree[consumer];
    }
  }

  // Use a stable Kahn order instead of relying on the order in which the
  // parser happened to expose task operations.
  std::vector<unsigned> topological;
  topological.reserve(taskCount);
  for (unsigned emitted = 0; emitted < taskCount; ++emitted) {
    unsigned next = taskCount;
    for (unsigned task = 0; task < taskCount; ++task)
      if (indegree[task] == 0 &&
          std::find(topological.begin(), topological.end(), task) ==
              topological.end()) {
        next = task;
        break;
      }
    if (next == taskCount) {
      result.graphAcyclic = false;
      return result;
    }
    topological.push_back(next);
    for (unsigned user : successors[next])
      --indegree[user];
  }

  while (true) {
    ++result.iterations;
    std::vector<int64_t> minDuration(taskCount,
                                     std::numeric_limits<int64_t>::max());
    std::vector<unsigned __int128> minWork(
        taskCount, std::numeric_limits<unsigned __int128>::max());
    for (unsigned task = 0; task < taskCount; ++task) {
      bool hasOption = false;
      bool hasUnknownCost = false;
      for (const ShapeDomainOption &option : domains[task]) {
        hasOption = true;
        if (!option.supported || option.duration <= 0) {
          hasUnknownCost = true;
          continue;
        }
        minDuration[task] = std::min(minDuration[task], option.duration);
        unsigned __int128 work = 0;
        if (detail::multiplyWide(option.duration, option.area, work))
          minWork[task] = std::min(minWork[task], work);
      }
      if (!hasOption)
        return result;
      // An unknown row may be faster or cheaper than every known row.  Zero
      // is the only safe optimistic lower bound for both latency and work.
      if (hasUnknownCost) {
        minDuration[task] = 0;
        minWork[task] = 0;
      } else if (minDuration[task] == std::numeric_limits<int64_t>::max() ||
                 minWork[task] ==
                     std::numeric_limits<unsigned __int128>::max()) {
        return result;
      }
    }

    std::vector<int64_t> release(taskCount, 0);
    bool releaseFinite = true;
    for (unsigned task : topological) {
      int64_t best = 0;
      for (unsigned producer : predecessors[task]) {
        int64_t producerEnd = 0;
        const bool producerEndFinite =
            detail::addWide(release[producer], minDuration[producer],
                            producerEnd);
        int64_t ready = 0;
        const bool readyFinite = detail::addWide(
            producerEnd, edgeLowerBounds[task][producer], ready);
        releaseFinite = releaseFinite && producerEndFinite && readyFinite;
        best = std::max(best, ready);
      }
      release[task] = best;
    }

    std::vector<int64_t> tail(taskCount, 0);
    bool tailFinite = true;
    for (auto reverse = topological.rbegin(); reverse != topological.rend();
         ++reverse) {
      const unsigned task = *reverse;
      int64_t best = 0;
      for (unsigned user : successors[task]) {
        int64_t userAfter = 0;
        int64_t userTail = tail[user];
        const bool userAfterFinite =
            detail::addWide(minDuration[user], userTail, userAfter);
        int64_t candidate = 0;
        const bool candidateFinite = detail::addWide(
            edgeLowerBounds[user][task], userAfter, candidate);
        tailFinite = tailFinite && userAfterFinite && candidateFinite;
        best = std::max(best, candidate);
      }
      tail[task] = best;
    }

    unsigned __int128 totalMinimumWork = 0;
    bool totalMinimumWorkFinite = true;
    for (unsigned task = 0; task < taskCount; ++task)
      totalMinimumWorkFinite =
          detail::addUnsignedWide(totalMinimumWork, minWork[task],
                                  totalMinimumWork) &&
          totalMinimumWorkFinite;
    const unsigned __int128 capacity =
        static_cast<unsigned __int128>(gridArea) *
        static_cast<unsigned __int128>(upperBound);

    // Build optimistic release/deadline event windows.  A negative deadline
    // already makes every positive-duration option fail the time bound; leave
    // the energetic path disabled in that malformed/impossible round rather
    // than interpreting a pre-time-zero window as evidence.
    std::vector<int64_t> deadline(taskCount, 0);
    std::vector<int64_t> events;
    bool windowInputsFinite = true;
    if (releaseFinite && tailFinite) {
      events.push_back(0);
      events.push_back(upperBound);
      for (unsigned task = 0; task < taskCount; ++task) {
        const __int128 deadlineWide =
            static_cast<__int128>(upperBound) -
            static_cast<__int128>(tail[task]);
        if (deadlineWide < 0 ||
            deadlineWide >
                static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
          windowInputsFinite = false;
          break;
        }
        deadline[task] = static_cast<int64_t>(deadlineWide);
        if (release[task] >= 0 && release[task] <= upperBound)
          events.push_back(release[task]);
        if (deadline[task] >= 0 && deadline[task] <= upperBound)
          events.push_back(deadline[task]);
      }
    } else {
      windowInputsFinite = false;
    }

    std::vector<std::vector<unsigned char>> windowRejected(taskCount);
    for (unsigned task = 0; task < taskCount; ++task)
      windowRejected[task].assign(domains[task].size(), 0);
    if (enableResourceBounds && windowInputsFinite) {
      std::sort(events.begin(), events.end());
      events.erase(std::unique(events.begin(), events.end()), events.end());
      const unsigned __int128 maximum =
          std::numeric_limits<unsigned __int128>::max();
      for (size_t startIndex = 0; startIndex < events.size(); ++startIndex) {
        for (size_t endIndex = startIndex + 1; endIndex < events.size();
             ++endIndex) {
          const int64_t start = events[startIndex];
          const int64_t end = events[endIndex];
          const unsigned __int128 width =
              static_cast<unsigned __int128>(end) -
              static_cast<unsigned __int128>(start);
          if (gridArea != 0 && width > maximum / gridArea)
            continue;
          const unsigned __int128 windowCapacity =
              width * static_cast<unsigned __int128>(gridArea);

          std::vector<unsigned __int128> minimumWindowWork(
              taskCount, maximum);
          unsigned __int128 totalWindowWork = 0;
          bool validWindow = true;
          for (unsigned task = 0; task < taskCount && validWindow; ++task) {
            bool hasOption = false;
            bool hasUnknownCost = false;
            for (const ShapeDomainOption &option : domains[task]) {
              hasOption = true;
              if (!option.supported || option.duration <= 0) {
                hasUnknownCost = true;
                continue;
              }
              unsigned __int128 work = 0;
              if (!detail::mandatoryWindowWork(
                      option, release[task], deadline[task], start, end,
                      work)) {
                validWindow = false;
                break;
              }
              minimumWindowWork[task] =
                  std::min(minimumWindowWork[task], work);
            }
            if (!hasOption || (!hasUnknownCost &&
                               minimumWindowWork[task] == maximum)) {
              validWindow = false;
              break;
            }
            if (hasUnknownCost)
              minimumWindowWork[task] = 0;
            if (!detail::addUnsignedWide(totalWindowWork,
                                         minimumWindowWork[task],
                                         totalWindowWork)) {
              validWindow = false;
              break;
            }
          }
          if (!validWindow)
            continue;

          for (unsigned task = 0; task < taskCount; ++task) {
            const unsigned __int128 otherWork =
                totalWindowWork - minimumWindowWork[task];
            for (size_t optionIndex = 0;
                 optionIndex < domains[task].size(); ++optionIndex) {
              const ShapeDomainOption &option = domains[task][optionIndex];
              if (!option.supported || option.duration <= 0)
                continue;
              unsigned __int128 optionWork = 0;
              if (!detail::mandatoryWindowWork(
                      option, release[task], deadline[task], start, end,
                      optionWork))
                continue;
              if (otherWork > windowCapacity ||
                  optionWork > windowCapacity - otherWork)
                windowRejected[task][optionIndex] = 1;
            }
          }
        }
      }
    }

    std::vector<std::vector<unsigned char>> optionRejected(taskCount);
    for (unsigned task = 0; task < taskCount; ++task)
      optionRejected[task].assign(domains[task].size(), 0);

    // Evaluate every option against the same domain snapshot before mutating
    // anything.  This permits a graph-level contradiction result without
    // leaving a partially swapped next-domain behind.
    for (unsigned task = 0; task < taskCount; ++task) {
      bool hasKnownOption = false;
      bool hasUnknownOption = false;
      bool allKnownOptionsRejected = true;
      uint64_t windowRejectedInContradiction = 0;
      for (size_t optionIndex = 0; optionIndex < domains[task].size();
           ++optionIndex) {
        const ShapeDomainOption &option = domains[task][optionIndex];
        if (!option.supported || option.duration <= 0) {
          hasUnknownOption = true;
          continue;
        }
        hasKnownOption = true;
        int64_t completion = 0;
        const bool finite = detail::addThree(
            release[task], option.duration, tail[task], completion);
        const bool impossibleByTime = finite && completion > upperBound;

        unsigned __int128 optionWork = 0;
        const bool optionWorkFinite =
            detail::multiplyWide(option.duration, option.area, optionWork);
        bool impossibleByWork = false;
        if (enableResourceBounds && totalMinimumWorkFinite && optionWorkFinite) {
          const unsigned __int128 otherWork =
              totalMinimumWork - minWork[task];
          impossibleByWork = otherWork > capacity ||
                             optionWork > capacity - otherWork;
        }
        const bool impossibleByWindow =
            windowRejected[task][optionIndex] != 0;
        if (impossibleByWindow &&
            windowRejectedInContradiction !=
                std::numeric_limits<uint64_t>::max())
          ++windowRejectedInContradiction;
        const bool rejected = impossibleByTime || impossibleByWork ||
                              impossibleByWindow;
        optionRejected[task][optionIndex] = rejected ? 1 : 0;
        allKnownOptionsRejected = allKnownOptionsRejected && rejected;
      }
      if (hasKnownOption && !hasUnknownOption && allKnownOptionsRejected) {
        result.wholeGraphAboveBound = true;
        result.contradictionTask = task;
        if (windowRejectedInContradiction >
            std::numeric_limits<uint64_t>::max() -
                result.windowRejectedOptions)
          result.windowRejectedOptions = std::numeric_limits<uint64_t>::max();
        else
          result.windowRejectedOptions += windowRejectedInContradiction;
        return result;
      }
    }

    std::vector<std::vector<ShapeDomainOption>> next = domains;
    bool changed = false;
    for (unsigned task = 0; task < taskCount; ++task) {
      unsigned supportedBefore = 0;
      unsigned supportedAfter = 0;
      for (const ShapeDomainOption &option : domains[task])
        supportedBefore += option.supported && option.duration > 0;
      for (size_t optionIndex = 0; optionIndex < domains[task].size();
           ++optionIndex) {
        const ShapeDomainOption &option = domains[task][optionIndex];
        if (!option.supported || option.duration <= 0) {
          ++supportedAfter;
          continue;
        }
        const bool rejected = optionRejected[task][optionIndex] != 0;
        const bool impossibleByWindow =
            windowRejected[task][optionIndex] != 0;
        if (!rejected) {
          ++supportedAfter;
          continue;
        }
        // Do not remove the last supported alternative.  A valid warmup
        // incumbent should make this guard unnecessary, but retaining the
        // domain is safer than turning an inconsistent bound into a false
        // empty-space proof.
        if (supportedBefore <= 1) {
          ++supportedAfter;
          continue;
        }
        next[task].erase(std::remove_if(
                             next[task].begin(), next[task].end(),
                             [&](const ShapeDomainOption &candidate) {
                               return candidate.originalOrdinal ==
                                          option.originalOrdinal &&
                                      candidate.supported;
                             }),
                         next[task].end());
        --supportedBefore;
        changed = true;
        ++result.removedOptions;
        ++result.removedByTask[task];
        if (impossibleByWindow)
          ++result.windowRejectedOptions;
      }
      (void)supportedAfter;
    }
    if (!changed)
      break;
    domains.swap(next);
    result.applied = true;
  }
  return result;
}

} // namespace mlir::amoeba::neura::joint_scheduling

#endif // AMOEBA_SHAPE_DOMAIN_BOUNDS_H
