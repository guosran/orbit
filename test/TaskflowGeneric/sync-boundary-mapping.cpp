#include "Taskflow/HeterogeneousMapping.h"
#include "Taskflow/SyncBoundaryMapping.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <set>
using namespace orbit::taskflow;
int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs()
        << "usage: taskflow-sync-boundary-mapping-test heterogeneous.mlir\n";
    return 2;
  }
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  std::string error;
  unsigned checks = 0;
  bool passed = true;
  auto require = [&](bool value, const char *name) {
    if (value)
      ++checks;
    else {
      passed = false;
      llvm::errs() << name << ": " << error << "\n";
    }
  };
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis source;
  if (!module || !analyzeGenericProgram(*module, source, error)) {
    llvm::errs() << error;
    return 1;
  }
  MechanismRuntimeContract runtime;
  runtime.identity = "synthetic-unit-runtime-not-hardware";
  SyncQueryContext query;
  query.callForm = SyncCallForm::ImplicitFullBuffer;
  query.observerBoundary = SyncObserverBoundary::DetailedCompatibility;
  query.runtimeIdentity = runtime.identity;
  query.observerIdentity = "synthetic-observer-source-identity";
  auto bindingsFor = [&](const ProgramGraph &g) {
    std::set<size_t> used;
    for (const auto &r : g.executionRegions)
      if (r.placement == "npu")
        for (const auto *list : {&r.inputCopies, &r.outputCopies})
          for (auto copy : *list)
            used.insert(g.allocations[g.copies[copy].allocation].storage);
    std::vector<SyncRuntimeBOBinding> bindings;
    for (auto storage : used) {
      SyncRuntimeBOBinding b;
      b.storage = storage;
      b.physicalId = "unit-bo-" + std::to_string(storage);
      b.generation = 1;
      b.observedExtentBytes = 4;
      for (auto n : g.source.storages[storage].shape)
        b.observedExtentBytes *= n;
      b.extentEvidence =
          "synthetic explicit API extent, not a page allocation claim";
      for (size_t i = 0; i < g.copies.size(); ++i) {
        const auto &a = g.allocations[g.copies[i].allocation];
        if (a.storage == storage && a.memory == "system.shared")
          b.copies.push_back(i);
      }
      bindings.push_back(std::move(b));
    }
    return bindings;
  };
  for (const std::vector<std::string> devices :
       {std::vector<std::string>{"cpu", "npu", "cpu"},
        {"npu", "cpu", "npu"},
        {"npu", "npu", "npu"}}) {
    HeterogeneousMappingChoice choice;
    choice.taskDevices = devices;
    if (devices == std::vector<std::string>{"npu", "npu", "npu"})
      choice.cuts = {1, 2};
    HeterogeneousMappingPlan plan;
    if (!buildHeterogeneousMapping(source, choice, plan, error)) {
      llvm::errs() << error;
      return 1;
    }
    MechanismFacts facts;
    if (!collectMechanismFacts(plan.graph, runtime, facts, error)) {
      llvm::errs() << error;
      return 1;
    }
    auto bindings = bindingsFor(plan.graph);
    SyncBoundaryTrace trace;
    require(collectSyncBoundaryQueries(plan.graph, facts, bindings, query,
                                       trace, error),
            "actual checked source/runtime projection");
    require(trace.origin == "static-success-path-projection" &&
                trace.requiredPredicates.size() == 3,
            "static success scenario is not observed runtime evidence");
    size_t syncs = 0;
    for (const auto &item : facts.items)
      syncs += item.mechanism == "bo.sync_to_device" ||
               item.mechanism == "bo.sync_from_device";
    require(trace.queries.size() == syncs,
            "one pre-call query per actual sync mechanism");
    if (devices == std::vector<std::string>{"npu", "cpu", "npu"}) {
      bool reusedInput = false;
      for (const auto &q : trace.queries)
        if (q.region == 2 && q.direction == SyncDirection::ToDevice &&
            source.storages[size_t(q.storage)].role == "input") {
          reusedInput = true;
          require(q.result.state == "after-device-read-complete",
                  "unchanged external BO is not falsely host-written again "
                  "before a later NPU region");
        }
      require(reusedInput,
              "real source has an external input reused across NPU regions");
    }
    for (const auto &q : trace.queries) {
      const auto &item = facts.items[q.mechanismItem];
      require(q.copies == item.copies && q.publicEvents == item.publicEvents &&
                  q.regionEvents == item.regionEvents &&
                  q.sourceTasks == item.sourceTasks,
              "query preserves exact source/copy/lifecycle identity");
      require(q.result.known && !q.result.pending && q.result.homogeneous,
              "whole exact tensor BO successful sequence is known");
      if (q.direction == SyncDirection::FromDevice)
        require(q.result.state == "after-device-write-complete",
                "FROM queries follow successful device writes before sync");
      auto begin = std::find_if(
          trace.observations.begin(), trace.observations.end(),
          [&](const auto &o) {
            return o.mechanismItem == int64_t(q.mechanismItem) &&
                   o.observation.kind == SyncObservationKind::SyncBegin;
          });
      require(begin != trace.observations.end(),
              "query corresponds to ordered sync begin");
      if (begin != trace.observations.end())
        for (const auto &s : q.result.segments)
          for (auto event : s.evidence)
            require(event < begin->observation.id,
                    "query evidence strictly precedes this sync call");
    }
    size_t deviceWaits = 0, deviceSubmits = 0;
    for (const auto &o : trace.observations) {
      deviceWaits += o.observation.kind == SyncObservationKind::WaitSuccess;
      deviceSubmits += o.observation.kind == SyncObservationKind::DeviceSubmit;
      if (o.mechanismItem >= 0)
        require(
            !facts.items[size_t(o.mechanismItem)].descriptive,
            "descriptive native DMA/source counts cause no extra state access");
    }
    require(deviceWaits == deviceSubmits && deviceWaits > 0,
            "one declared command and successful wait per NPU region");
    auto reject = [&](std::vector<SyncRuntimeBOBinding> b, SyncQueryContext c,
                      MechanismFacts f, const char *label) {
      SyncBoundaryTrace ignored;
      require(!collectSyncBoundaryQueries(plan.graph, f, b, c, ignored, error),
              label);
    };
    auto changed = bindings;
    changed.front().observedExtentBytes = 0;
    reject(changed, query, facts, "missing observed BO extent rejected");
    changed = bindings;
    changed.front().observedExtentBytes--;
    reject(changed, query, facts, "runtime BO too small rejected");
    changed = bindings;
    changed.front().extentEvidence.clear();
    reject(changed, query, facts, "missing runtime extent evidence rejected");
    changed = bindings;
    changed.front().copies.clear();
    reject(changed, query, facts, "missing runtime ABI alias rejected");
    changed = bindings;
    changed.front().copies.push_back(plan.graph.copies.size());
    reject(changed, query, facts, "bad copy alias fails closed");
    changed = bindings;
    changed.push_back(changed.front());
    reject(changed, query, facts, "duplicate physical binding rejected");
    changed = bindings;
    changed.erase(changed.begin());
    reject(changed, query, facts, "missing NPU storage binding rejected");
    auto c = query;
    c.runtimeIdentity = "different";
    reject(bindings, c, facts, "runtime identity mismatch rejected");
    c = query;
    c.observerIdentity.clear();
    reject(bindings, c, facts, "observer identity absent rejected");
    c = query;
    c.callForm = SyncCallForm::ExplicitRange;
    reject(bindings, c, facts,
           "explicit-range call not substituted for installed implicit form");
    c = query;
    c.observerBoundary = SyncObserverBoundary::MinimalAPI;
    reject(bindings, c, facts,
           "minimal API not substituted for installed detailed wrapper");
    auto f = facts;
    f.identity = "tampered";
    reject(bindings, query, f, "stale facts identity rejected");
    f = facts;
    f.items.front().copies = {plan.graph.copies.size()};
    reject(bindings, query, f,
           "mutated mechanism metadata rejected before dereference");
    changed = bindings;
    for (auto &b : changed)
      b.observedExtentBytes += 64;
    SyncBoundaryTrace padded;
    require(
        collectSyncBoundaryQueries(plan.graph, facts, changed, query, padded,
                                   error),
        "explicit observed padding supported without invented initialization");
    for (const auto &q : padded.queries)
      require(!q.result.known && q.result.segments.size() > 1 &&
                  !q.result.reason.empty(),
              "implicit full BO includes unknown padding and remains unpriced");
    auto serialized = serializeSyncBoundaryTrace(trace);
    const auto *object = serialized.getAsObject();
    require(object &&
                object->getString("origin") ==
                    "static-success-path-projection" &&
                object->getBoolean("timing_inferred") == false,
            "serializer states non-timing/non-observed scope");
  }
  llvm::outs() << "{\"passed\":" << (passed ? "true" : "false")
               << ",\"checks\":" << checks << "}\n";
  return passed ? 0 : 1;
}
