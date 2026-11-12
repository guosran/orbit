#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/MechanismCost.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <tuple>
using namespace orbit::taskflow;
int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: taskflow-mechanism-cost-test heterogeneous.mlir\n";
    return 2;
  }
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
  };
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis source;
  if (!module || !analyzeGenericProgram(*module, source, error)) {
    llvm::errs() << error;
    return 1;
  }
  HeterogeneousMappingChoice choice;
  choice.taskDevices = {"cpu", "npu", "cpu"};
  HeterogeneousMappingPlan plan;
  if (!buildHeterogeneousMapping(source, choice, plan, error)) {
    llvm::errs() << error;
    return 1;
  }
  MechanismRuntimeContract runtime;
  runtime.identity = "synthetic-unit-runtime-not-measured";
  MechanismFacts facts;
  if (!collectMechanismFacts(plan.graph, runtime, facts, error)) {
    llvm::errs() << error;
    return 1;
  }
  require(!facts.identity.empty() &&
              facts.graphId == programGraphIdentity(plan.graph),
          "actual checked graph identity");
  require(
      facts.runtimeResourceBindings.at("host.thread").targetResource ==
              "CPU.compute" &&
          facts.runtimeResourceBindings.at("host.memory").targetResource ==
              "CPU.compute" &&
          facts.runtimeResourceBindings.at("host.memory").targetCapacity == 1,
      "runtime thread/copy labels alias the real serial host compute worker");
  unsigned resets = 0, finals = 0, to = 0, from = 0, submit = 0, wait = 0,
           cpu = 0, conservative = 0, overhead = 0;
  int64_t dma = 0, cpuReads = 0, cpuWrites = 0;
  std::vector<MechanismParameter> parameters;
  std::set<std::tuple<std::string, std::string, std::string, std::string,
                      std::string>>
      seen;
  for (const auto &item : facts.items) {
    resets += item.mechanism == "host.input_reset";
    finals += item.mechanism == "host.output_copy";
    to += item.mechanism == "bo.sync_to_device";
    from += item.mechanism == "bo.sync_from_device";
    submit += item.mechanism == "npu.submit";
    wait += item.mechanism == "npu.wait";
    cpu += item.mechanism == "cpu.call";
    overhead += item.mechanism == "host.call_overhead" ||
                item.mechanism == "runtime.region_overhead";
    if (item.mechanism == "native.dma") {
      dma += item.bytes;
      require(item.descriptive && item.coveredBy >= 0 &&
                  facts.items[size_t(item.coveredBy)].mechanism == "npu.wait" &&
                  item.publicEvents.size() == 1 &&
                  plan.graph.events[item.publicEvents.front()].kind ==
                      "transfer" &&
                  plan.graph.events[item.publicEvents.front()].transferBytes ==
                      item.bytes,
              "descriptive DMA binds one real graph transfer to its opaque "
              "wait owner");
    }
    if (item.mechanism == "cpu.logical_read")
      cpuReads += item.bytes;
    if (item.mechanism == "cpu.logical_write")
      cpuWrites += item.bytes;
    if (item.runtimeOnly) {
      ++conservative;
      require(item.mechanism == "bo.sync_to_device" &&
                  item.regionEvents.empty(),
              "conservative output sync is not a logical data read");
    }
    if (item.mechanism == "npu.submit" || item.mechanism == "npu.wait")
      require(item.regionEvents.size() == 1 &&
                  plan.graph.regionEvents[item.regionEvents.front()].kind ==
                      (item.mechanism == "npu.submit" ? "submit" : "complete"),
              "submit/wait binds only its actual public lifecycle API");
    if ((item.mechanism == "bo.sync_to_device" ||
         item.mechanism == "bo.sync_from_device") &&
        !item.runtimeOnly) {
      bool exact = item.regionEvents.size() == 1 && item.copies.size() == 1;
      for (auto id : item.regionEvents) {
        const auto &event = plan.graph.regionEvents[id];
        exact &= event.kind == (item.mechanism == "bo.sync_to_device"
                                    ? "visible-to-device"
                                    : "visible-to-host") &&
                 std::find(event.copies.begin(), event.copies.end(),
                           item.copies.front()) != event.copies.end();
      }
      require(exact, "BO sync binds only matching direction and host-copy "
                     "public visibility");
    }
    if (item.mechanism == "cpu.call")
      require(
          item.regionEvents.size() == 2 &&
              plan.graph.regionEvents[item.regionEvents[0]].kind == "submit" &&
              plan.graph.regionEvents[item.regionEvents[1]].kind == "complete",
          "one CPU library call refines submit plus complete");
    if (item.mechanism == "host.call_overhead" ||
        item.mechanism == "runtime.region_overhead")
      require(
          item.regionEvents.empty(),
          "unknown bookkeeping does not invent a public API correspondence");
    if (item.mechanism == "bo.sync_from_device")
      require(item.state == "device-completed",
              "FROM requires real device completion state");
    if (item.descriptive || item.composite)
      continue;
    auto key = std::make_tuple(item.mechanism, item.path, item.state,
                               item.resource, item.boundary);
    if (!seen.insert(key).second)
      continue;
    MechanismParameter p;
    p.mechanism = item.mechanism;
    p.path = item.path;
    p.state = item.state;
    p.resource = item.resource;
    p.boundary = item.boundary;
    p.runtimeIdentity = runtime.identity;
    p.alphaMs = .001;
    p.betaMsPerByte = .000001;
    p.minBytes = 0;
    p.maxBytes = 8192;
    p.provenance = "synthetic CPU contract test only";
    p.measurementBoundary = p.boundary;
    p.rawManifestSHA256 = std::string(64, 'a');
    parameters.push_back(p);
  }
  require(resets == 2 && finals == 1,
          "global input resets and final copy counted once");
  require(to == 2 && from == 1 && conservative == 1,
          "runner syncs inputs AND output TO, output FROM");
  require(submit == 1 && wait == 1 && cpu == 2 && overhead == 4,
          "actual API/composite and unmodeled overhead count");
  require(dma == plan.graph.transferBytes && dma == 128,
          "native DMA payload described once inside wait ownership");
  require(cpuReads == 256 && cpuWrites == 128,
          "CPU mapped/host access is visible rather than zero-cost");
  MechanismCostEvaluation cost;
  require(evaluateMechanismCost(facts, parameters, cost, error),
          "typed mechanism parameter evaluation");
  require(!cost.totalMs && std::abs(cost.knownSubtotalMs - .007384) < 1e-12,
          "API known subtotal does not become full program cost");
  require(parameters.front().resource == "host.memory",
          "resource aliasing preserves calibrated parameter label identity");
  for (size_t i = 0; i < facts.items.size(); ++i)
    if (facts.items[i].descriptive)
      require(cost.items[i].status == "included-in-composite" &&
                  !cost.items[i].durationMs,
              "descriptive DMA/access never adds elapsed time");
  require(!cost.items.front().startMs && !cost.items.back().startMs,
          "unknown serial prefix is never assigned zero duration");
  auto reject =
      [&](const char *label,
          std::function<void(std::vector<MechanismParameter> &)> change) {
        auto p = parameters;
        change(p);
        MechanismCostEvaluation c;
        error.clear();
        require(!evaluateMechanismCost(facts, p, c, error), label);
      };
  reject("negative coefficient", [](auto &p) { p.front().alphaMs = -1; });
  reject("overlapping duplicate API domains",
         [](auto &p) { p.push_back(p.front()); });
  reject("inverted byte domain",
         [](auto &p) { p.front().minBytes = p.front().maxBytes + 1; });
  reject("missing raw digest",
         [](auto &p) { p.front().rawManifestSHA256.clear(); });
  reject("invalid raw digest",
         [](auto &p) { p.front().rawManifestSHA256 = std::string(64, 'z'); });
  reject("mismatched measurement boundary", [](auto &p) {
    p.front().measurementBoundary = "whole-command composite";
  });
  reject("native DMA cannot be charged in addition to opaque wait",
         [](auto &p) { p.front().mechanism = "native.dma"; });
  reject("CPU mapped accesses cannot be charged in addition to CPU call",
         [](auto &p) { p.front().mechanism = "cpu.logical_read"; });
  reject("wait cannot be modeled as independently calibrated bandwidth",
         [](auto &p) { p.front().mechanism = "npu.wait"; });
  reject("all-zero coefficients do not invent free APIs", [](auto &p) {
    p.front().alphaMs = 0;
    p.front().betaMsPerByte = 0;
  });
  auto mismatch = parameters;
  for (auto &p : mismatch)
    p.runtimeIdentity = "other-runtime";
  require(evaluateMechanismCost(facts, mismatch, cost, error) &&
              cost.knownSubtotalMs == 0 && !cost.totalMs,
          "different runtime remains uncalibrated");
  mismatch = parameters;
  for (auto &p : mismatch)
    p.state = "host-read";
  require(evaluateMechanismCost(facts, mismatch, cost, error) &&
              cost.knownSubtotalMs == 0,
          "standalone host-read state is not relabeled as program "
          "device-completed");
  mismatch = parameters;
  for (auto &p : mismatch) {
    p.minBytes = 9000;
    p.maxBytes = 10000;
  }
  require(evaluateMechanismCost(facts, mismatch, cost, error) &&
              cost.knownSubtotalMs == 0,
          "outside byte domain remains unknown");
  auto altered = facts;
  altered.items.front().bytes++;
  require(!evaluateMechanismCost(altered, parameters, cost, error),
          "tampered mechanism facts rejected");
  runtime.boundBuffers = false;
  require(!collectMechanismFacts(plan.graph, runtime, altered, error),
          "unbound runner cannot claim no staging path");
  runtime.boundBuffers = true;
  runtime.outputPoisoned = false;
  require(!collectMechanismFacts(plan.graph, runtime, altered, error),
          "unproven pre-dispatch output state rejected");
  runtime.outputPoisoned = true;
  auto invalidResource = runtime;
  invalidResource.resourceAliases["host.memory"] = "nonexistent-worker";
  require(!collectMechanismFacts(plan.graph, invalidResource, altered, error),
          "runtime alias cannot refer to an absent target resource");
  invalidResource = runtime;
  invalidResource.resourceAliases.erase("host.thread");
  require(!collectMechanismFacts(plan.graph, invalidResource, altered, error),
          "missing runtime thread target binding rejected");
  invalidResource = runtime;
  invalidResource.resourceAliases["host.memory"] = "NPU.compute";
  require(!collectMechanismFacts(plan.graph, invalidResource, altered, error),
          "host copying cannot claim a device compute resource");
  HeterogeneousMappingChoice chain;
  chain.taskDevices = {"npu", "npu", "npu"};
  chain.cuts = {1, 2};
  HeterogeneousMappingPlan chainPlan;
  bool built = buildHeterogeneousMapping(source, chain, chainPlan, error);
  require(built &&
              collectMechanismFacts(chainPlan.graph, runtime, altered, error),
          "actual cross-NPU-region mechanism extraction");
  if (built) {
    unsigned deviceProduced = 0, cleanVisible = 0;
    for (const auto &item : altered.items)
      if (item.mechanism == "bo.sync_to_device") {
        deviceProduced += item.state == "device-completed-host-visible";
        cleanVisible += item.state == "host-visible-after-sync";
      }
    require(deviceProduced == 2, "previous-device output TO state is not "
                                 "fabricated as CPU-host-written");
    require(cleanVisible == 1, "retained external B TO sync is host-visible "
                               "after first sync, not host-written again");
  }
  // Parser smoke plus malformed schema and number types.
  llvm::json::Value empty =
      llvm::json::Object{{"schema", "orbit.mechanism-cost-parameters.v13"},
                         {"parameters", llvm::json::Array{}}};
  std::vector<MechanismParameter> parsed;
  require(parseMechanismParameters(empty, parsed, error) && parsed.empty(),
          "empty calibration table leaves facts unpriced");
  empty.getAsObject()->operator[]("schema") = "wrong";
  require(!parseMechanismParameters(empty, parsed, error),
          "unsupported parameter schema rejected");
  const auto &p = parameters.front();
  llvm::json::Value row =
      llvm::json::Object{{"mechanism", p.mechanism},
                         {"path", p.path},
                         {"state", p.state},
                         {"resource", p.resource},
                         {"boundary", p.boundary},
                         {"runtime_identity", p.runtimeIdentity},
                         {"alpha_ms", p.alphaMs},
                         {"beta_ms_per_byte", p.betaMsPerByte},
                         {"min_bytes", p.minBytes},
                         {"max_bytes", p.maxBytes},
                         {"provenance", p.provenance},
                         {"measurement_boundary", p.measurementBoundary},
                         {"raw_manifest_sha256", p.rawManifestSHA256}};
  llvm::json::Array validRows;
  validRows.push_back(std::move(row));
  llvm::json::Value table =
      llvm::json::Object{{"schema", "orbit.mechanism-cost-parameters.v13"},
                         {"parameters", std::move(validRows)}};
  require(parseMechanismParameters(table, parsed, error) &&
              parsed.size() == 1 && parsed.front().path == p.path,
          "explicit JSON calibration row parses with exact API binding");
  table.getAsObject()
      ->getArray("parameters")
      ->front()
      .getAsObject()
      ->operator[]("alpha_ms") = "not-a-number";
  require(!parseMechanismParameters(table, parsed, error),
          "string coefficient cannot enter the analytical model");
  require(serializeMechanismFacts(facts).getAsObject() &&
              serializeMechanismCost(cost).getAsObject(),
          "facts and evaluation JSON export");
  llvm::outs() << "{\"status\":\"" << (ok ? "passed" : "failed")
               << "\",\"checks\":" << checks
               << ",\"scope\":\"synthetic CPU mechanism contracts; no runtime "
                  "measurements or NPU submissions\"}\n";
  return ok ? 0 : 1;
}
