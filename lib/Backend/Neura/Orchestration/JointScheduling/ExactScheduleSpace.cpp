//===- ExactScheduleSpace.cpp - Grid/time branch legality -----------------===//

#include "ExactScheduleSpace.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

namespace mlir::amoeba::neura::joint_scheduling {
namespace {
struct Search {
  int gridRows;
  int gridCols;
  const std::vector<ExactScheduleTask> &tasks;
  ExactScheduleLimits limits;
  const std::function<void(const std::vector<ExactSchedulePlacement> &,
                           const std::vector<unsigned> &)> &emit;
  ExactScheduleResult &result;
  std::set<std::string> seen;
  std::vector<ExactSchedulePlacement> placed;
  std::vector<bool> scheduled;
  std::chrono::steady_clock::time_point began =
      std::chrono::steady_clock::now();

  bool stopped() {
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

  bool visit() {
    if (stopped())
      return false;
    ++result.expandedNodes;
    if (placed.size() == tasks.size()) {
      if (result.acceptedPaths == std::numeric_limits<uint64_t>::max()) {
        result.reason = "count-overflow";
        return false;
      }
      ++result.acceptedPaths;
      // Preserve every ready-task order: two orders can reserve network
      // transfers differently once communication-aware replay is added.
      // Count geometric/time-equivalent schedules separately for diagnostics.
      std::vector<unsigned> dispatchOrder;
      dispatchOrder.reserve(placed.size());
      for (const auto &placement : placed)
        dispatchOrder.push_back(placement.task);
      std::vector<ExactSchedulePlacement> canonical = placed;
      std::sort(canonical.begin(), canonical.end(),
                [](const auto &a, const auto &b) { return a.task < b.task; });
      std::ostringstream key;
      for (const auto &p : canonical)
        key << p.task << ':' << p.row << ',' << p.col << '@'
            << p.start << '-' << p.end << ';';
      if (seen.insert(key.str()).second) {
        if (result.uniqueSchedules == std::numeric_limits<uint64_t>::max()) {
          result.reason = "count-overflow";
          return false;
        }
        ++result.uniqueSchedules;
      }
      emit(canonical, dispatchOrder);
      return true;
    }

    for (unsigned index = 0; index < tasks.size(); ++index) {
      if (scheduled[index])
        continue;
      const auto &task = tasks[index];
      int64_t ready = 0;
      bool allPredecessorsPlaced = true;
      for (unsigned predecessor : task.predecessors) {
        if (!scheduled[predecessor]) {
          allPredecessorsPlaced = false;
          break;
        }
        const auto found = std::find_if(
            placed.begin(), placed.end(), [&](const auto &entry) {
              return entry.task == predecessor;
            });
        ready = std::max(ready, found->end);
      }
      if (!allPredecessorsPlaced)
        continue;

      // A useful start is dependency-ready or a previously occupied cell's
      // release event. A later arbitrary idle time can be left-shifted until
      // one of these events, for this fixed order and rectangle origin.
      std::vector<int64_t> events{ready};
      for (const auto &other : placed)
        if (other.end >= ready)
          events.push_back(other.end);
      std::sort(events.begin(), events.end());
      events.erase(std::unique(events.begin(), events.end()), events.end());

      for (int64_t start : events) {
        if (start > std::numeric_limits<int64_t>::max() - task.duration)
          continue;
        int64_t end = start + task.duration;
        for (int row = 0; row < gridRows; ++row)
          for (int col = 0; col < gridCols; ++col) {
            if (!canPlace(row, col, start, end, task))
              continue;
            placed.push_back({index, row, col, start, end});
            scheduled[index] = true;
            if (!visit())
              return false;
            scheduled[index] = false;
            placed.pop_back();
          }
      }
    }
    return true;
  }
};
} // namespace

bool enumerateExactScheduleSpace(
    int gridRows, int gridCols, const std::vector<ExactScheduleTask> &tasks,
    ExactScheduleLimits limits,
    const std::function<void(const std::vector<ExactSchedulePlacement> &,
                             const std::vector<unsigned> &)> &emit,
    ExactScheduleResult &result, std::string &error) {
  result = {};
  error.clear();
  if (gridRows != 4 || gridCols != 4 || limits.maxExpandedNodes == 0 ||
      limits.maxMilliseconds >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    error = "exact scheduling requires a 4x4 grid and valid positive budget";
    return false;
  }
  for (unsigned i = 0; i < tasks.size(); ++i) {
    const auto &task = tasks[i];
    if (task.name.empty() || task.rows < 1 || task.cols < 1 ||
        task.rows > 4 || task.cols > 4 ||
        task.rows * task.cols > 4 || task.rows > gridRows ||
        task.cols > gridCols || task.duration < 1) {
      error = "invalid task name, fixed shape, duration or per-task CGRA limit";
      return false;
    }
    for (unsigned predecessor : task.predecessors)
      if (predecessor >= tasks.size() || predecessor == i) {
        error = "invalid task dependency";
        return false;
      }
  }
  // Reject cycles before DFS; an empty ready frontier is not a complete
  // enumeration of a cyclic input.
  std::vector<unsigned> indegree(tasks.size(), 0);
  std::vector<std::vector<unsigned>> users(tasks.size());
  for (unsigned i = 0; i < tasks.size(); ++i) {
    std::set<unsigned> distinct(tasks[i].predecessors.begin(),
                                tasks[i].predecessors.end());
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
  Search search{gridRows, gridCols, tasks, limits, emit, result, {}, {}, {},
                std::chrono::steady_clock::now()};
  search.scheduled.assign(tasks.size(), false);
  result.complete = search.visit();
  if (!result.complete && result.reason.empty())
    result.reason = "interrupted";
  return true;
}
} // namespace mlir::amoeba::neura::joint_scheduling
