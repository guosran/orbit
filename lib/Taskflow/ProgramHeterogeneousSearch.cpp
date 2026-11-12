#include "Taskflow/ProgramHeterogeneousSearch.h"
#include "Taskflow/StructuredProgram.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
bool fail(std::string &error, const std::string &message) {
  error = "public heterogeneous search: " + message;
  return false;
}
j::Array numbers(const std::vector<size_t> &values) {
  j::Array a;
  for (auto v : values)
    a.push_back(int64_t(v));
  return a;
}
j::Value choiceJSON(const HeterogeneousMappingChoice &choice) {
  j::Array devices;
  for (auto &v : choice.taskDevices)
    devices.push_back(v);
  return j::Object{{"task_devices", std::move(devices)},
                   {"cuts", numbers(choice.cuts)}};
}
j::Value spaceJSON(const GenericAnalysis &source,
                   const ProgramSearchSpace &space) {
  j::Array dimensions, tables;
  for (auto &d : space.dimensions) {
    j::Array choices;
    for (auto &c : d.choices)
      choices.push_back(
          j::Object{{"id", c.id}, {"binding", j::Value(c.binding)}});
    dimensions.push_back(j::Object{{"id", d.id},
                                   {"scope", d.scope},
                                   {"status", d.status},
                                   {"reason", d.reason},
                                   {"choices", std::move(choices)}});
  }
  for (auto &t : space.constraints) {
    j::Array rows;
    for (auto &r : t.allowedTuples) {
      j::Array values;
      for (auto v : r)
        values.push_back(v);
      rows.push_back(std::move(values));
    }
    tables.push_back(j::Object{{"dimensions", numbers(t.dimensions)},
                               {"allowed_tuples", std::move(rows)}});
  }
  return j::Object{{"schema", "orbit.public-heterogeneous-space.v28"},
                   {"source_fingerprint", source.bodyFingerprint},
                   {"source_task_count", int64_t(source.tasks.size())},
                   {"dimensions", std::move(dimensions)},
                   {"constraints", std::move(tables)}};
}
bool decode(const GenericAnalysis &source, const ProgramSearchSpace &space,
            const std::vector<int64_t> &assignment,
            HeterogeneousMappingChoice &choice, std::string &error) {
  const size_t n = source.tasks.size();
  if (!n || n > SIZE_MAX / 2 || space.dimensions.size() != 2 * n - 1 ||
      assignment.size() != space.dimensions.size())
    return fail(
        error,
        "domain must contain exactly N device and N-1 boundary dimensions");
  choice = {};
  choice.taskDevices.resize(n);
  std::set<std::string> ids;
  for (size_t i = 0; i < space.dimensions.size(); ++i) {
    const auto &d = space.dimensions[i];
    bool device = i < n;
    size_t task = device ? i : i - n + 1;
    std::string expected =
        (device ? "device.task." : "boundary.before.") + std::to_string(task);
    if (d.id != expected || !ids.insert(d.id).second || d.choices.empty())
      return fail(error, "missing, aliased, reordered or empty dimension: " +
                             expected);
    if (assignment[i] < 0 || uint64_t(assignment[i]) >= d.choices.size())
      return fail(error, "choice index outside declared domain");
    std::set<std::string> values, choiceIDs;
    for (size_t c = 0; c < d.choices.size(); ++c) {
      const auto &value = d.choices[c];
      auto *binding = value.binding.getAsObject();
      if (!binding || binding->size() != 2 || value.id.empty() ||
          !choiceIDs.insert(value.id).second)
        return fail(error, "choice needs an exact typed binding and unique ID");
      std::string key;
      if (device) {
        auto index = binding->getInteger("task");
        auto v = binding->getString("device");
        if (!index || *index != int64_t(task) || !v ||
            (*v != "cpu" && *v != "npu"))
          return fail(error, "invalid source task/device binding");
        key = v->str();
        if (c == size_t(assignment[i]))
          choice.taskDevices[task] = key;
      } else {
        auto index = binding->getInteger("before_task");
        auto v = binding->getBoolean("cut");
        if (!index || *index != int64_t(task) || !v)
          return fail(error, "invalid source boundary binding");
        key = *v ? "cut" : "join";
        if (c == size_t(assignment[i]) && *v)
          choice.cuts.push_back(task);
      }
      if (!values.insert(key).second)
        return fail(error, "duplicate semantic choice in dimension");
    }
  }
  for (const auto &t : space.constraints) {
    if (t.dimensions.empty())
      return fail(error, "empty compatibility table scope");
    std::set<size_t> axes;
    for (auto a : t.dimensions)
      if (a >= assignment.size() || !axes.insert(a).second)
        return fail(error, "invalid or repeated compatibility table dimension");
    bool admitted = false;
    for (auto &row : t.allowedTuples) {
      if (row.size() != t.dimensions.size())
        return fail(error, "compatibility tuple arity mismatch");
      bool same = true;
      for (size_t k = 0; k < row.size(); ++k) {
        size_t a = t.dimensions[k];
        if (row[k] < 0 ||
            uint64_t(row[k]) >= space.dimensions[a].choices.size())
          return fail(error, "compatibility tuple value outside domain");
        same &= assignment[a] == row[k];
      }
      admitted |= same;
    }
    if (!admitted)
      return fail(error, "assignment excluded by static compatibility table");
  }
  return true;
}
} // namespace
bool orbit::taskflow::buildProgramHeterogeneousSearchSpace(
    const GenericAnalysis &source,
    const ProgramHeterogeneousSearchOptions &options, ProgramSearchSpace &space,
    std::string &error) {
  space = {};
  error.clear();
  size_t n = source.tasks.size();
  if (!source.function || source.bodyFingerprint.empty() || !n ||
      n > SIZE_MAX / 2)
    return fail(error, "nonempty analyzed source required");
  if (!options.taskDeviceChoices.empty() &&
      options.taskDeviceChoices.size() != n)
    return fail(error, "device domains need one row per source task");
  if (!options.boundaryChoices.empty() &&
      options.boundaryChoices.size() != n - 1)
    return fail(error, "boundary domains need one row per source boundary");
  if (options.search.maxVariables < 0 ||
      2 * n - 1 > uint64_t(options.search.maxVariables))
    return fail(error, "dimension count exceeds variable cap");
  int64_t selectors = 0;
  for (size_t t = 0; t < n; ++t) {
    auto values = options.taskDeviceChoices.empty()
                      ? std::vector<std::string>{"cpu", "npu"}
                      : options.taskDeviceChoices[t];
    if (values.empty() || values.size() > 2)
      return fail(error, "device choice row must contain cpu and/or npu");
    std::set<std::string> seen;
    ProgramSearchDimension d;
    d.id = "device.task." + std::to_string(t);
    d.scope = "original-source-task";
    d.status = values.size() > 1 ? "searched" : "fixed_by_domain";
    d.reason = "Existing CPU/NPU region adapter; capability/physical "
               "feasibility checked on complete assignment.";
    for (auto &v : values) {
      if ((v != "cpu" && v != "npu") || !seen.insert(v).second)
        return fail(error, "invalid or duplicate device choice");
      d.choices.push_back({v, j::Object{{"task", int64_t(t)}, {"device", v}}});
    }
    if (d.choices.size() > 1)
      selectors += d.choices.size();
    space.dimensions.push_back(std::move(d));
  }
  for (size_t t = 1; t < n; ++t) {
    auto values = options.boundaryChoices.empty()
                      ? std::vector<bool>{false, true}
                      : options.boundaryChoices[t - 1];
    if (values.empty() || values.size() > 2)
      return fail(error, "boundary choice row must contain join and/or cut");
    std::set<bool> seen;
    ProgramSearchDimension d;
    d.id = "boundary.before." + std::to_string(t);
    d.scope = "original-source-task-boundary";
    d.status = values.size() > 1 ? "searched" : "fixed_by_domain";
    d.reason = "Device changes always form a region; same-device cuts form "
               "separate calls.";
    for (bool v : values) {
      if (!seen.insert(v).second)
        return fail(error, "duplicate boundary choice");
      d.choices.push_back({v ? "cut" : "join",
                           j::Object{{"before_task", int64_t(t)}, {"cut", v}}});
    }
    if (d.choices.size() > 1)
      selectors += d.choices.size();
    space.dimensions.push_back(std::move(d));
  }
  if (selectors >
      options.search.maxVariables - int64_t(space.dimensions.size())) {
    space = {};
    return fail(error, "dimension selectors exceed variable cap");
  }
  return true;
}
std::string orbit::taskflow::programHeterogeneousSearchSpaceIdentity(
    const GenericAnalysis &source, const ProgramSearchSpace &space) {
  return staticFingerprint(spaceJSON(source, space));
}
bool orbit::taskflow::materializeProgramHeterogeneousAssignment(
    const GenericAnalysis &source, const ProgramSearchSpace &space,
    const std::vector<int64_t> &assignment, HeterogeneousMappingPlan &plan,
    std::string &error) {
  plan = {};
  HeterogeneousMappingChoice choice;
  if (!decode(source, space, assignment, choice, error))
    return false;
  return buildHeterogeneousMapping(source, choice, plan, error) &&
         checkHeterogeneousMapping(source, choice, plan, error);
}
ProgramSearchEvaluation orbit::taskflow::evaluateProgramHeterogeneousAssignment(
    const GenericAnalysis &source, const ProgramSearchSpace &space,
    const std::vector<int64_t> &assignment,
    const ProgramHeterogeneousEvaluator &evaluate) {
  ProgramSearchEvaluation out;
  HeterogeneousMappingPlan plan;
  std::string error;
  if (!evaluate) {
    out.feasibilityKnown = false;
    out.message = "missing complete heterogeneous cost/checker callback";
    return out;
  }
  if (!materializeProgramHeterogeneousAssignment(source, space, assignment,
                                                 plan, error)) {
    out.feasibilityKnown = false;
    out.message = error;
    out.result = j::Object{
        {"status", "unsupported-existing-region-adapter"},
        {"source_fingerprint", source.bodyFingerprint},
        {"space_id", programHeterogeneousSearchSpaceIdentity(source, space)},
        {"reason", error},
        {"physical_infeasibility_proved", false}};
    return out;
  }
  out = evaluate(plan);
  auto payload = std::move(out.result);
  out.result = j::Object{
      {"cost_evaluation", std::move(payload)},
      {"heterogeneous",
       j::Object{
           {"schema", "orbit.public-heterogeneous-candidate.v28"},
           {"source_fingerprint", source.bodyFingerprint},
           {"space_id", programHeterogeneousSearchSpaceIdentity(source, space)},
           {"choice", choiceJSON(plan.choice)},
           {"graph_id", programGraphIdentity(plan.graph)},
           {"program_graph", serializeProgramGraph(plan.graph)},
           {"physical_realization_checked", false}}}};
  if (out.sourceMLIR.empty()) {
    llvm::raw_string_ostream stream(out.sourceMLIR);
    source.function->getParentOfType<mlir::ModuleOp>().print(stream);
  }
  return out;
}
ProgramSearchResult orbit::taskflow::searchProgramHeterogeneous(
    const GenericAnalysis &source,
    const ProgramHeterogeneousEvaluator &evaluate,
    const ProgramHeterogeneousSearchOptions &options) {
  if (!std::isfinite(options.search.wallSeconds) ||
      options.search.wallSeconds <= 0)
    return searchProgramSpace({}, ProgramSearchEvaluator{}, options.search);
  auto start = std::chrono::steady_clock::now();
  auto elapsed = [&]() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
  };
  ProgramSearchSpace space;
  std::string error;
  if (!buildProgramHeterogeneousSearchSpace(source, options, space, error)) {
    ProgramSearchResult r;
    r.status = ConstraintSearchStatus::InvalidInput;
    r.report = j::Object{{"status", "INVALID_INPUT"},
                         {"reason", error},
                         {"has_best", false},
                         {"total_ms", elapsed()},
                         {"finite_domain_optimality_proved", false}};
    return r;
  }
  double buildMs = elapsed();
  auto budget = options.search;
  if (std::isfinite(budget.wallSeconds))
    budget.wallSeconds =
        std::max(0.000001, budget.wallSeconds - buildMs / 1000.0);
  auto result = searchProgramSpace(
      space,
      [&](const std::vector<int64_t> &a) {
        return evaluateProgramHeterogeneousAssignment(source, space, a,
                                                      evaluate);
      },
      budget);
  auto &report = result.report;
  report["schema"] = "orbit.public-heterogeneous-search.v28";
  report["space_id"] = programHeterogeneousSearchSpaceIdentity(source, space);
  report["domain"] = spaceJSON(source, space);
  report["source_fingerprint"] = source.bodyFingerprint;
  report["domain_build_ms"] = buildMs;
  report["total_ms"] = elapsed();
  report["wall_seconds"] = std::isfinite(options.search.wallSeconds)
                               ? j::Value(options.search.wallSeconds)
                               : j::Value(nullptr);
  report["scope"] =
      "Joint finite per-source-task CPU/NPU placement and call boundaries "
      "using existing HeterogeneousMapping; full evaluator cost, no "
      "complete-plan prepricing or hardware optimum claim.";
  report["adapter"] = "public-heterogeneous-regions";
  report["capability_scope"] =
      "Existing static identity-layout host f32 ABI, one source writer per "
      "storage, no inout alias region ABI; complete child implementation "
      "feasibility belongs to caller evaluator.";
  report["cost_provider"] =
      "caller-supplied complete checked region graph evaluator";
  report["unsupported_mapping_is_unknown"] = true;
  report["physical_realization_checked"] = false;
  report["best_mlir_is_complete_execution_plan"] = false;
  if (std::isfinite(options.search.wallSeconds) &&
      options.search.wallSeconds >= 0) {
    double extra = std::max(0.0, elapsed() - options.search.wallSeconds * 1000);
    report["wall_budget_overshoot_ms"] = extra;
    report["wall_budget_exceeded"] = extra > 0;
  }
  return result;
}
