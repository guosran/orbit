#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/HeterogeneousSearch.h"
#include "Taskflow/ProgramRegionCompositeCost.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>
using namespace orbit::taskflow;
namespace {
using J = llvm::json::Value;
using O = llvm::json::Object;
using A = llvm::json::Array;
std::string unit(const std::string &key) {
  return key == "call"
             ? "ns"
             : (key == "abi_bytes" || key == "requested_transfer_bytes"
                    ? "ns/byte"
                    : "ns/count");
}
J parameter(const ProgramMechanismParameter &p) {
  return O{{"value", p.value ? J(*p.value) : J(nullptr)},
           {"unit", p.unit},
           {"provenance", O{{"kind", p.kind}, {"source", p.source}}}};
}
J json(const ProgramRegionCompositeModel &m) {
  A rows;
  for (auto &r : m.regions) {
    O cs, ds;
    for (auto &p : r.coefficients)
      cs[p.first] = parameter(p.second);
    for (auto &d : r.domain)
      ds[d.first] = O{{"min", d.second.minimum}, {"max", d.second.maximum}};
    rows.push_back(O{{"implementation", r.implementation},
                     {"source_family", r.sourceFamily},
                     {"control_family", r.controlFamily},
                     {"provenance", r.provenance},
                     {"coefficients", std::move(cs)},
                     {"domain", std::move(ds)}});
  }
  O parameters;
  for (auto &p : m.parameters)
    parameters[p.first] = parameter(p.second);
  return O{
      {"kind", "region-composite"},
      {"schema", "orbit.region-composite.cost.v29"},
      {"model_id", m.id},
      {"mode", m.mode},
      {"boundary", m.boundary},
      {"calibration_runtime_identity", m.calibrationRuntimeIdentity},
      {"runtime", O{{"identity", m.runtime.identity},
                    {"state", m.runtime.state},
                    {"host_worker_resource", m.runtime.hostWorkerResource},
                    {"bound_buffers", m.runtime.boundBuffers},
                    {"stage_outputs_to_device", m.runtime.stageOutputsToDevice},
                    {"reset_inputs", m.runtime.resetInputs},
                    {"copy_final_outputs", m.runtime.copyFinalOutputs}}},
      {"region_models", std::move(rows)},
      {"parameters", std::move(parameters)}};
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis source;
  std::string error;
  unsigned checks = 0;
  bool ok = true;
  auto require = [&](bool v, const char *label) {
    if (v)
      ++checks;
    else {
      ok = false;
      llvm::errs() << label << ": " << error << "\n";
    }
    error.clear();
  };
  if (!module || !analyzeGenericProgram(*module, source, error))
    return 1;
  ProgramInvocationRuntimeContract runtime;
  runtime.identity = "synthetic-composite-bound.v29";
  runtime.hostWorkerResource = "CPU.compute";
  runtime.resetInputs = true;
  auto planFor = [&](std::vector<std::string> devices,
                     HeterogeneousMappingPlan &p) {
    HeterogeneousMappingChoice c;
    c.taskDevices = std::move(devices);
    return buildHeterogeneousMapping(source, c, p, error);
  };
  auto witnesses = [&](const HeterogeneousMappingPlan &p) {
    std::vector<ProgramRegionImplementation> out;
    for (size_t i = 0; i < p.graph.executionRegions.size(); ++i) {
      mlir::OwningOpRef<mlir::ModuleOp> child;
      HeterogeneousRegionProjection projection;
      GenericAnalysis analysis;
      if (!materializeHeterogeneousRegion(p, i, child, projection, error) ||
          !analyzeGenericProgram(*child, analysis, error))
        return out;
      out.push_back(
          {i, p.graph.executionRegions[i].implementation,
           analysis.bodyFingerprint,
           "synthetic-code-identity-" + std::to_string(i),
           "synthetic-control-sequence-" + std::to_string(i),
           p.graph.executionRegions[i].placement + "-synthetic-fullcall.v29",
           "synthetic CPU-only compiler contract; no hardware calibration"});
    }
    return out;
  };
  auto modelFor = [&](const ProgramRegionCompositeFeatureSet &f) {
    ProgramRegionCompositeModel m;
    m.id = "synthetic-affine-composite.v29";
    m.runtime = runtime;
    m.calibrationRuntimeIdentity = runtime.identity;
    m.boundary = "synthetic exclusive region calls and once-only global "
                 "application copies";
    std::set<std::tuple<std::string, std::string, std::string>> seen;
    for (auto &r : f.regions) {
      if (!seen.emplace(r.implementation.implementation, r.sourceFamily,
                        r.implementation.controlFamily)
               .second)
        continue;
      ProgramRegionCompositeCalibration row;
      row.implementation = r.implementation.implementation;
      row.sourceFamily = r.sourceFamily;
      row.controlFamily = r.implementation.controlFamily;
      row.provenance = "synthetic calibration contract, not measured hardware";
      for (auto &q : r.quantities) {
        row.coefficients[q.first] = {
            q.first == "call" ? 100.0 : 0.0, unit(q.first), "architecture",
            "explicit synthetic compiler test coefficient"};
        row.domain[q.first] = {0, 1e12};
      }
      m.regions.push_back(std::move(row));
    }
    for (auto name : {"input_reset", "output_copy"})
      for (auto term : {"per_call", "per_bo", "per_byte"})
        m.parameters[std::string("invocation.") + name + "." + term] = {
            std::string(term) == "per_call" ? 10.0 : 0.0,
            std::string(term) == "per_byte" ? "ns/byte" : "ns", "architecture",
            "explicit synthetic global boundary coefficient"};
    m.parameters["invocation.control.per_call"] = {
        7, "ns", "architecture",
        "synthetic scattered orchestration coefficient"};
    m.parameters["invocation.control.per_region"] = {
        2, "ns/count", "architecture",
        "synthetic per-region orchestration coefficient"};
    return m;
  };
  HeterogeneousMappingPlan plan;
  require(planFor({"cpu", "npu", "cpu"}, plan),
          "build real CPU NPU CPU parent graph");
  auto bindings = witnesses(plan);
  ProgramRegionCompositeFeatureSet features;
  require(collectProgramRegionCompositeFeatures(plan, runtime, bindings,
                                                features, error),
          "collect source-backed region features");
  require(features.regions.size() == 3,
          "one feature row per actual invocation");
  for (const auto &r : features.regions) {
    require(r.quantities.size() == 8 && r.quantities.at("call") == 1 &&
                r.quantities.at("task_count") == 1 &&
                r.quantities.at("source_operations") > 0 &&
                r.quantities.at("source_memory_accesses") > 0,
            "source work and eight feature quantities derive from actual "
            "source instances");
    require(!r.sourceFamily.empty() &&
                r.sourceFingerprint == r.implementation.sourceFingerprint,
            "family normalization and exact source guard are separate");
  }
  auto model = modelFor(features);
  ProgramTimingRequest request;
  J receipt = nullptr;
  require(buildProgramRegionCompositeCosts(plan, features, model, request,
                                           receipt, error),
          "build exclusive region composites");
  std::vector<int> body(plan.graph.events.size()),
      lifecycle(plan.graph.regionEvents.size());
  for (auto &c : request.costs) {
    for (auto i : c.events)
      ++body[i];
    for (auto i : c.regionEvents)
      ++lifecycle[i];
  }
  require(std::all_of(body.begin(), body.end(), [](int n) { return n == 1; }) &&
              std::all_of(lifecycle.begin(), lifecycle.end(),
                          [](int n) { return n == 1; }),
          "every source body and lifecycle atom has exactly one owner");
  require(request.costs.size() == 6,
          "three regions plus global reset, control and output acquisition, no "
          "additive mechanism fees");
  ProgramTimingEvaluation timing;
  require(evaluateProgramTiming(plan.graph, request, timing, error) &&
              timing.makespanMs &&
              std::abs(*timing.makespanMs - .000333) < 1e-12,
          "whole invocation costs count once with separate global copies");
  require(receipt.getAsObject()->getBoolean(
              "additive_body_or_lifecycle_costs") == false,
          "receipt explicitly forbids double-counting measured composite plus "
          "body");
  auto duplicate = request;
  auto extra = request.costs.front();
  extra.id = "duplicate.body";
  duplicate.costs.push_back(extra);
  require(
      !evaluateProgramTiming(plan.graph, duplicate, timing, error),
      "common checker rejects whole-call plus separately priced body coverage");
  auto bad = features;
  bad.regions[1].quantities["abi_bytes"]++;
  require(!buildProgramRegionCompositeCosts(plan, bad, model, request, receipt,
                                            error),
          "tampered feature row cannot be priced");
  auto wrong = bindings;
  wrong[1].sourceFingerprint = "different-source";
  require(
      !collectProgramRegionCompositeFeatures(plan, runtime, wrong, bad, error),
      "compiled code witness must match materialized source");
  wrong = bindings;
  wrong[1].codeIdentity.clear();
  require(
      !collectProgramRegionCompositeFeatures(plan, runtime, wrong, bad, error),
      "missing actual code identity cannot silently enter calibrated model");
  wrong = bindings;
  wrong[1].controlSequenceIdentity.clear();
  require(
      !collectProgramRegionCompositeFeatures(plan, runtime, wrong, bad, error),
      "missing control sequence identity rejected");
  wrong = bindings;
  wrong[1].controlSequenceIdentity =
      "different-code-with-same-certified-control-family";
  require(evaluateProgramRegionCompositeModel(plan, model, wrong, timing,
                                              receipt, error) &&
              timing.makespanMs &&
              std::abs(*timing.makespanMs - .000333) < 1e-12,
          "code identity is evidence, never a complete-plan price lookup key");
  auto unknown = [&](ProgramRegionCompositeModel m, const char *label) {
    require(evaluateProgramRegionCompositeModel(plan, m, bindings, timing,
                                                receipt, error) &&
                !timing.makespanMs,
            label);
  };
  auto missing = model;
  missing.regions[1].coefficients.erase("call");
  unknown(missing, "missing full-call coefficient preserves unknown total");
  missing = model;
  missing.regions[1].domain["abi_bytes"] = {0, 1};
  unknown(missing, "out-of-range shape cannot silently extrapolate");
  missing = model;
  missing.regions[1].domain.erase("source_operations");
  unknown(missing, "missing fit range is unknown even with a coefficient");
  missing = model;
  missing.regions[1].sourceFamily = "different-arithmetic-family";
  unknown(missing,
          "different semantic source cannot borrow a family calibration");
  missing = model;
  missing.regions[1].controlFamily = "resident-no-reconfiguration";
  unknown(
      missing,
      "different physical control family cannot borrow full-call calibration");
  missing = model;
  missing.regions[1].coefficients["call"].kind = "assumed";
  unknown(missing, "strict mode rejects assumed call coefficient");
  missing = model;
  missing.parameters.erase("invocation.output_copy.per_call");
  unknown(missing,
          "unpriced global output copy cannot disappear inside region prices");
  missing = model;
  missing.runtime.boundBuffers = false;
  require(!buildProgramRegionCompositeCosts(plan, features, missing, request,
                                            receipt, error),
          "calibration runtime flags cannot change under same model features");
  missing = model;
  missing.parameters.erase("invocation.control.per_call");
  unknown(
      missing,
      "missing global control per-call overhead cannot silently become free");
  missing = model;
  missing.parameters.erase("invocation.control.per_region");
  unknown(missing,
          "missing per-region orchestration coefficient remains unknown");
  auto document = json(model);
  ProgramRegionCompositeModel parsed;
  require(parseProgramRegionCompositeModel(document, parsed, error) &&
              evaluateProgramRegionCompositeModel(plan, parsed, bindings,
                                                  timing, receipt, error) &&
              timing.makespanMs,
          "strict JSON model parses and prices the same checked graph");
  auto invalid = json(model);
  invalid.getAsObject()->getObject("runtime")->erase("bound_buffers");
  require(!parseProgramRegionCompositeModel(invalid, parsed, error),
          "runtime flags cannot be implicit");
  invalid = json(model);
  (*invalid.getAsObject())["completion_policy"] = "polling-quantized";
  require(!parseProgramRegionCompositeModel(invalid, parsed, error),
          "full-call composite cannot add a separate polling model");
  invalid = json(model);
  (*invalid.getAsObject()->getObject("parameters"))["compute.operation"] =
      parameter({1, "ns", "assumed", "forbidden additive body"});
  require(!parseProgramRegionCompositeModel(invalid, parsed, error),
          "parser rejects additive per-event mechanism parameters");
  invalid = json(model);
  auto *rows = invalid.getAsObject()->getArray("region_models");
  rows->push_back(
      json(model).getAsObject()->getArray("region_models")->front());
  require(!parseProgramRegionCompositeModel(invalid, parsed, error),
          "duplicate family calibration is rejected");
  HeterogeneousMappingPlan allCPU, allNPU, twice;
  for (auto entry :
       {std::make_pair(std::vector<std::string>{"cpu", "cpu", "cpu"}, &allCPU),
        std::make_pair(std::vector<std::string>{"npu", "npu", "npu"}, &allNPU),
        std::make_pair(std::vector<std::string>{"npu", "cpu", "npu"},
                       &twice)}) {
    require(planFor(entry.first, *entry.second),
            "build actual resident invocation choices");
    auto b = witnesses(*entry.second);
    require(collectProgramRegionCompositeFeatures(*entry.second, runtime, b,
                                                  features, error),
            "collect CPU/NPU region feature partition");
    auto m = modelFor(features);
    require(evaluateProgramRegionCompositeModel(*entry.second, m, b, timing,
                                                receipt, error) &&
                timing.makespanMs,
            "price resident choice using exclusive full calls");
    require(timing.activities.size() == features.regions.size() + 3,
            "one composite per call plus reset/control/output boundaries");
  }
  require(allCPU.graph.executionRegions.size() == 1 &&
              allNPU.graph.executionRegions.size() == 1 &&
              twice.graph.executionRegions.size() == 3,
          "all-CPU and resident NPU are one call, split NPU has two physical "
          "invocations");
  llvm::outs() << "program region composite cost checks: " << checks << "\n";
  return ok ? 0 : 1;
}
