#include "Taskflow/GenericMapping.h"
#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramInvocationCost.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
using namespace orbit::taskflow;
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  std::string error;
  unsigned checks = 0;
  bool ok = true;
  auto require = [&](bool value, const char *label) {
    if (value)
      ++checks;
    else {
      ok = false;
      llvm::errs() << label << ": " << error << "\n";
    }
    error.clear();
  };
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis source;
  if (!module || !analyzeGenericProgram(*module, source, error)) {
    llvm::errs() << error;
    return 1;
  }
  ProgramInvocationRuntimeContract runtime;
  runtime.identity = "synthetic-warm-bound-v28";
  runtime.hostWorkerResource = "CPU.compute";
  runtime.resetInputs = true;
  ProgramMechanismModel model;
  model.id = "synthetic-host-cost-contract";
  model.mode = "estimate";
  model.boundary =
      "synthetic exclusive CPU/device body plus prepared host API boundary";
  for (auto prefix : {"input_reset", "output_copy", "device.stage",
                      "device.to_device", "device.submit", "device.complete",
                      "device.from_device", "host.submit", "host.complete"})
    for (auto term : {"per_call", "per_bo", "per_byte"}) {
      auto &p =
          model.parameters[std::string("invocation.") + prefix + "." + term];
      p.value = std::string(term) == "per_byte" ? .5 : 1;
      p.unit = std::string(term) == "per_byte" ? "ns/byte" : "ns";
      p.kind = "assumed";
      p.source = "synthetic compiler-only contract, never NPU measurements";
    }
  // Explicit synthetic zeros preserve the old test boundary while requiring
  // physical setup/teardown to be priced rather than inferred free.
  for (auto phase : {"setup", "teardown"}) {
    auto &p = model.parameters[std::string("invocation.device.") + phase +
                               ".per_call"];
    p.value = 0;
    p.unit = "ns";
    p.kind = "assumed";
    p.source = "explicit synthetic zero, not calibrated device configuration";
  }
  auto build = [&](std::vector<std::string> devices,
                   HeterogeneousMappingPlan &p) {
    HeterogeneousMappingChoice c;
    c.taskDevices = std::move(devices);
    return buildHeterogeneousMapping(source, c, p, error);
  };
  auto request = [&](const ProgramGraph &g,
                     const std::vector<ProgramActivityCost> &boundary) {
    ProgramTimingRequest q;
    q.graphId = programGraphIdentity(g);
    q.modelIdentity = model.id;
    q.boundary = model.boundary;
    bool reset = std::any_of(boundary.begin(), boundary.end(), [](auto &c) {
      return c.id == "invocation.input_reset";
    });
    for (size_t i = 0; i < g.events.size(); ++i) {
      ProgramActivityCost c;
      c.id = "body." + std::to_string(i);
      c.events = {i};
      c.boundary = "synthetic unit work";
      c.provenance = "test fixture, no measurement";
      c.durationMs = .000001;
      c.resourceDemands[g.events[i].resource] = 1;
      if (reset)
        c.predecessors = {"invocation.input_reset"};
      for (size_t ri = 0; ri < g.executionRegions.size(); ++ri) {
        const auto &r = g.executionRegions[ri];
        bool member = r.memberEvents.empty()
                          ? (i >= r.beginEvent && i < r.endEvent)
                          : llvm::is_contained(r.memberEvents, i);
        if (member &&
            std::any_of(boundary.begin(), boundary.end(), [&](auto &b) {
              return b.id == "invocation.device_setup." + std::to_string(ri);
            }))
          c.predecessors.push_back("invocation.device_setup." +
                                   std::to_string(ri));
      }
      q.costs.push_back(c);
    }
    q.costs.insert(q.costs.end(), boundary.begin(), boundary.end());
    for (size_t ri = 0; ri < g.executionRegions.size(); ++ri) {
      auto teardown =
          std::find_if(q.costs.begin(), q.costs.end(), [&](auto &c) {
            return c.id == "invocation.device_teardown." + std::to_string(ri);
          });
      if (teardown == q.costs.end())
        continue;
      const auto &r = g.executionRegions[ri];
      for (size_t i = 0; i < g.events.size(); ++i)
        if (r.memberEvents.empty() ? (i >= r.beginEvent && i < r.endEvent)
                                   : llvm::is_contained(r.memberEvents, i))
          teardown->predecessors.push_back("body." + std::to_string(i));
      for (auto &c : q.costs)
        for (auto ei : c.regionEvents)
          if (g.regionEvents[ei].region == ri &&
              g.regionEvents[ei].kind == "complete")
            c.predecessors.push_back(teardown->id);
    }
    return q;
  };
  HeterogeneousMappingPlan multi;
  require(build({"npu", "cpu", "cpu"}, multi),
          "source-backed two-input NPU followed by CPU region");
  ProgramInvocationFeatures features;
  require(
      collectProgramInvocationFeatures(multi.graph, runtime, features, error),
      "collect explicit warm bound ABI features");
  require(features.regions.size() == 2 && features.regions[0].device &&
              features.regions[0].inputAllocations.size() == 2 &&
              features.regions[0].submissions == 1,
          "two inputs share one launch");
  require(features.regions[0].abiAllocations.size() == 3 &&
              features.regions[0].toDeviceBytes == 192 &&
              features.regions[0].fromDeviceBytes == 64 &&
              features.regions[0].stagingBytes == 0,
          "bound BO syncs inputs and output, without staging memcpy");
  require(features.initialInputBytes == 128 && features.finalOutputBytes == 64,
          "application reset and final output acquisition each counted once");
  std::vector<ProgramActivityCost> costs;
  llvm::json::Value receipt = nullptr;
  require(buildProgramInvocationCosts(multi.graph, features, model,
                                      runtime.identity, costs, receipt, error),
          "build lifecycle-only prices");
  bool coverage = true, resetOrdered = true;
  for (auto &c : costs) {
    coverage &= c.events.empty();
    if (!c.regionEvents.empty())
      resetOrdered &=
          std::find(c.predecessors.begin(), c.predecessors.end(),
                    "invocation.input_reset") != c.predecessors.end();
  }
  require(
      coverage && resetOrdered,
      "body events remain exclusively owned, every lifecycle follows reset");
  require(receipt.getAsObject()->getBoolean(
              "logical_group_configuration_charged") == false,
          "logical groups never fabricate configuration calls");
  auto setups = std::count_if(costs.begin(), costs.end(), [](auto &c) {
    return c.id.find("invocation.device_setup.") == 0;
  });
  require(setups == 1, "two NPU inputs have exactly one physical setup");
  require(receipt.getAsObject()->getInteger("configured_runtime_count") == 1,
          "receipt counts actual device invocations, not logical groups");
  auto setup = std::find_if(costs.begin(), costs.end(), [](auto &c) {
    return c.id == "invocation.device_setup.0";
  });
  require(setup != costs.end() && setup->runtimeOnly && setup->events.empty() &&
              setup->regionEvents.empty() &&
              setup->resourceDemands.count("NPU.compute") &&
              setup->resourceDemands.count("NPU.dma") &&
              setup->predecessors ==
                  std::vector<std::string>{
                      "invocation.lifecycle." +
                      std::to_string(features.regions[0].submit)},
          "setup owns exact device resources and follows submit without owning "
          "body events");
  auto q = request(multi.graph, costs);
  ProgramTimingEvaluation timing;
  require(evaluateProgramTiming(multi.graph, q, timing, error) &&
              timing.makespanMs,
          "body and host boundary compose to finite full cost");
  auto duplicate = q;
  duplicate.costs.push_back(costs.front());
  duplicate.costs.back().id = "duplicate.lifecycle";
  require(!evaluateProgramTiming(multi.graph, duplicate, timing, error),
          "duplicate lifecycle ownership independently rejected");
  auto missing = model;
  missing.parameters.erase("invocation.device.complete.per_call");
  std::vector<ProgramActivityCost> unknown;
  require(buildProgramInvocationCosts(multi.graph, features, missing,
                                      runtime.identity, unknown, receipt,
                                      error),
          "missing complete parameter preserves feasible unknown cost");
  require(evaluateProgramTiming(multi.graph, request(multi.graph, unknown),
                                timing, error) &&
              !timing.makespanMs,
          "unpriced polling/return overhead cannot become zero full latency");
  for (auto phase : {"setup", "teardown"}) {
    auto missingPhase = model;
    missingPhase.parameters.erase(std::string("invocation.device.") + phase +
                                  ".per_call");
    require(buildProgramInvocationCosts(multi.graph, features, missingPhase,
                                        runtime.identity, unknown, receipt,
                                        error),
            "missing physical phase preserves feasible graph");
    require(evaluateProgramTiming(multi.graph, request(multi.graph, unknown),
                                  timing, error) &&
                !timing.makespanMs,
            "missing setup or teardown cannot silently price the complete "
            "invocation");
  }
  require(!buildProgramInvocationCosts(multi.graph, features, model,
                                       "different-runtime", unknown, receipt,
                                       error),
          "foreign runtime calibration cannot price this invocation");
  auto corrupt = features;
  corrupt.regions[0].toDeviceBytes++;
  require(!buildProgramInvocationCosts(multi.graph, corrupt, model,
                                       runtime.identity, unknown, receipt,
                                       error),
          "tampered ABI byte feature rejected");
  auto unbound = runtime;
  unbound.boundBuffers = false;
  unbound.identity = "synthetic-warm-unbound-v28";
  ProgramInvocationFeatures staged;
  require(
      collectProgramInvocationFeatures(multi.graph, unbound, staged, error) &&
          staged.regions[0].stagingBytes == 192,
      "unbound stages ABI arrays into BOs separately from application reset");
  require(buildProgramInvocationCosts(multi.graph, staged, model,
                                      unbound.identity, unknown, receipt,
                                      error),
          "unbound staging uses explicit runtime calibration domain");
  bool stage = false;
  for (auto &row : *receipt.getAsObject()->getArray("quantities"))
    stage |= row.getAsObject()->getString("mechanism") == "device.stage";
  require(stage, "unbound staging receives distinct exclusive API charge");
  HeterogeneousMappingPlan twice;
  require(build({"npu", "cpu", "npu"}, twice),
          "two distinct NPU execution regions");
  ProgramInvocationFeatures two;
  require(collectProgramInvocationFeatures(twice.graph, runtime, two, error),
          "collect two-call features");
  int64_t calls = 0;
  for (auto &r : two.regions)
    if (r.device)
      calls += r.submissions;
  require(calls == 2, "two NPU regions produce exactly two launches");
  require(buildProgramInvocationCosts(twice.graph, two, model, runtime.identity,
                                      costs, receipt, error),
          "price both physical invocations");
  require(
      std::count_if(costs.begin(), costs.end(),
                    [](auto &c) {
                      return c.id.find("invocation.device_setup.") == 0;
                    }) == 2 &&
          std::count_if(costs.begin(), costs.end(),
                        [](auto &c) {
                          return c.id.find("invocation.device_teardown.") == 0;
                        }) == 2 &&
          receipt.getAsObject()->getInteger("configured_runtime_count") == 2,
      "two NPU regions have two setup and teardown phases");
  HeterogeneousMappingPlan mixed;
  require(build({"cpu", "npu", "cpu"}, mixed),
          "source-backed CPU to NPU to CPU pipeline");
  require(collectProgramInvocationFeatures(mixed.graph, runtime, two, error) &&
              buildProgramInvocationCosts(mixed.graph, two, model,
                                          runtime.identity, costs, receipt,
                                          error),
          "price CPU NPU CPU with explicit lifecycle costs");
  q = request(mixed.graph, costs);
  require(evaluateProgramTiming(mixed.graph, q, timing, error) &&
              timing.makespanMs,
          "cross-region pipeline timing is complete");
  auto acquisition = std::find_if(costs.begin(), costs.end(), [](auto &c) {
    return c.id == "invocation.output_copy";
  });
  require(acquisition != costs.end() &&
              acquisition->predecessors.size() == two.regions.size(),
          "output acquisition explicitly waits for every region terminal, not "
          "the last event index");
  for (const auto &region : two.regions) {
    auto terminal = region.device ? region.visibleToHost : region.complete;
    require(acquisition != costs.end() &&
                std::find(acquisition->predecessors.begin(),
                          acquisition->predecessors.end(),
                          "invocation.lifecycle." + std::to_string(terminal)) !=
                    acquisition->predecessors.end(),
            "every CPU completion or device readback is an output-copy "
            "predecessor");
  }

  for (auto &a : timing.activities)
    for (auto &pred : a.predecessors) {
      auto p = std::find_if(timing.activities.begin(), timing.activities.end(),
                            [&](auto &x) { return x.id == pred; });
      require(p != timing.activities.end() && a.startMs && p->endMs &&
                  *a.startMs >= *p->endMs,
              "CPU NPU CPU precedences survive cost composition");
    }
  HeterogeneousMappingPlan cpu;
  require(build({"cpu", "cpu", "cpu"}, cpu), "CPU-only region builder");
  ProgramInvocationFeatures host;
  require(collectProgramInvocationFeatures(cpu.graph, runtime, host, error) &&
              !host.regions[0].device && host.finalOutputBytes == 64,
          "CPU-only final output is still acquired");
  require(buildProgramInvocationCosts(cpu.graph, host, model, runtime.identity,
                                      costs, receipt, error),
          "CPU-only lifecycle is priced");
  require(std::any_of(costs.begin(), costs.end(),
                      [](auto &c) { return c.id == "invocation.output_copy"; }),
          "CPU-only output-copy cost is explicit");
  require(std::none_of(costs.begin(), costs.end(),
                       [](auto &c) {
                         return c.id.find("invocation.device_setup.") == 0 ||
                                c.id.find("invocation.device_teardown.") == 0;
                       }),
          "CPU-only invocation has no physical device setup or teardown");
  // A native tiled graph has many cell copies in the same caller BO. Count the
  // BO once, while retaining internal hand-off arguments in the launch ABI.
  auto nativeSource = module->clone();
  require(setNativeAIRTileSizes(nativeSource, 8, 1, 1, error),
          "select actual native tiles");
  GenericAnalysis nativeAnalysis;
  GenericMappingPlan native;
  require(analyzeGenericProgram(nativeSource, nativeAnalysis, error) &&
              planGenericMapping(nativeAnalysis, llvm::json::Object{}, native,
                                 error),
          "source-backed native tiled graph");
  auto nativeRuntime = runtime;
  nativeRuntime.hostWorkerResource = "host.worker";
  ProgramInvocationFeatures dedup;
  require(collectProgramInvocationFeatures(native.communication, nativeRuntime,
                                           dedup, error),
          "native graph boundary ABI is collected");
  require(native.communication.executionRegions.front().inputCopies.size() >
                  dedup.regions.front().inputAllocations.size() &&
              dedup.regions.front().inputAllocations.size() == 2,
          "multiple exact-cell copies deduplicate to two input BOs");
  require(dedup.regions.front().scratchAllocations.size() == 2 &&
              dedup.regions.front().abiAllocations.size() == 5,
          "internal host handoff BOs count as launch arguments, not fresh "
          "live-in data");
  nativeSource.erase();
  // Explicitly accessible memory does not license an invocation spanning chips.
  auto cross = multi.graph;
  cross.domains.push_back({"foreign-chip", "", "device"});
  for (auto &r : cross.resources)
    if (r.id == "NPU.compute")
      r.domain = "foreign-chip";
  require(checkProgramGraph(cross, error),
          "public graph can represent cross-chip resource ownership");
  require(!collectProgramInvocationFeatures(cross, runtime, dedup, error),
          "physical invocation cannot silently span chips");
  llvm::outs() << "program invocation cost checks: " << checks << "\n";
  return ok ? 0 : 1;
}
