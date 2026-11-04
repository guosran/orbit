#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
using namespace orbit::taskflow;
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect,mlir::affine::AffineDialect,
      mlir::arith::ArithDialect,mlir::math::MathDialect,mlir::memref::MemRefDialect,
      mlir::func::FuncDialect,mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);
  std::string error;
  GenericAnalysis analysis;
  GenericMappingPlan baseline,panel;
  llvm::json::Value config = llvm::json::Object{{"generic_air",
      llvm::json::Object{{"tile_sizes",llvm::json::Array{2,2,4}}}}};
  if (!module || !distributeNativeInitialization(*module,error) ||
      !analyzeGenericProgram(*module,analysis,error) ||
      !planGenericMapping(analysis,config,baseline,error)) {
    llvm::errs() << error << "\n"; return 1;
  }
  auto opportunities = discoverGenericMappingChoices(analysis);
  bool found = false;
  ProgramMappingChoice selected;
  for (const auto &choice : opportunities)
    if (choice.storage == 1 && choice.reuseLoop == 0 && choice.panelLoop == 1) {
      found = true; selected = choice;
    }
  if (!found || !setNativeMappingChoice(*module,selected,error) ||
      !analyzeGenericProgram(*module,analysis,error) ||
      !planGenericMapping(analysis,config,panel,error)) {
    llvm::errs() << "source-discovered panel failed: " << error << "\n"; return 1;
  }
  int64_t baselineBytes = 0,panelBytes = 0;
  for (const auto &t : baseline.transfers) baselineBytes += t.bytes;
  for (const auto &t : panel.transfers) panelBytes += t.bytes;
  if (baselineBytes != 576 || panelBytes != 448 || panel.readOnlyRetentions.size() != 2 ||
      panel.executionGroups.size() != 2 || panel.taskLoopOrders.back() != std::vector<size_t>{1,0,2} ||
      panel.communication.events.empty() || panel.communication.copies.empty()) {
    llvm::errs() << "panel did not materialize proven cross-outer-axis reuse: "
                 << baselineBytes << " -> " << panelBytes << "\n"; return 1;
  }
  unsigned checks = 2;
  bool ok = true;
  // The backend imports the enclosing panel before the phase's streaming
  // input. Both orders must satisfy the same dependency DAG; pure local
  // initialization has no dependency on either independent input copy.
  std::vector<size_t> firstInputs, firstInitializers;
  size_t firstPhase = panel.hardwareStages.front().beginPhase;
  for (size_t e = 0; e < panel.communication.events.size(); ++e) {
    const auto &event = panel.communication.events[e];
    if (event.phase != firstPhase) continue;
    if (event.kind == "transfer" && event.link == "host_to_l1") firstInputs.push_back(e);
    if (event.kind == "initialize") firstInitializers.push_back(e);
  }
  if (firstInputs.size() != 2 || firstInitializers.size() != 1 ||
      !panel.communication.events[firstInputs[0]].dependencies.empty() ||
      !panel.communication.events[firstInputs[1]].dependencies.empty() ||
      !panel.communication.events[firstInitializers[0]].dependencies.empty()) {
    llvm::errs() << "independent same-phase transfers/initializer were artificially serialized\n";
    ok = false;
  } else ++checks;
  auto reject = [&](const char *name,std::function<void(GenericMappingPlan &)> mutate) {
    auto changed = panel; changed.planId.clear(); mutate(changed);
    if (checkGenericMapping(analysis,changed,error)) {
      llvm::errs() << "accepted invalid public mapping: " << name << "\n"; ok = false;
    } else ++checks;
  };
  reject("initial output version",[](auto &p) {
    p.communication.versions.front().region = p.communication.regions.size()-1;
  });
  reject("missing copy creation",[](auto &p) { p.communication.copies.pop_back(); });
  reject("initial input copy placed in local pool",[](auto &p) {
    for (size_t i = 0; i < p.communication.allocations.size(); ++i)
      if (p.communication.allocations[i].memory == "aie_l1") {
        p.communication.copies.front().allocation = i; break;
      }
  });
  reject("copy becomes available after use",[](auto &p) {
    p.communication.copies.front().availableFrom = p.communication.events.size();
  });
  reject("early copy release",[](auto &p) { p.communication.copies.front().releasedAfter = 0; });
  reject("fabricated final user",[](auto &p) { p.communication.copies.front().lastUsers.clear(); });
  reject("missing source/destination transfer binding",[](auto &p) { p.transfers.front().sourceCopies.clear(); });
  reject("fabricated same-phase input dependence",[](auto &p) {
    auto &events = p.communication.events;
    for (size_t e = 1; e < events.size(); ++e)
      if (events[e].kind == "transfer" && events[e].phase == events[e-1].phase) {
        events[e].dependencies.push_back(e-1); return;
      }
  });
  reject("wrong transfer event",[](auto &p) { ++p.transfers.front().event; });
  reject("missing execution event",[](auto &p) { p.communication.events.pop_back(); });
  reject("missing task instance event binding",[](auto &p) {
    for (auto &phase : p.phases) if (phase.computationEvent >= 0) { phase.computationEvent = -1; break; }
  });
  reject("pool overwrite lacks completion",[](auto &p) { p.communication.events.back().dependencies.clear(); });
  reject("physical pool allocated too small",[](auto &p) { --p.communication.allocations.back().bytes; });
  reject("allocation outlives public group",[](auto &p) { ++p.communication.allocations.back().endPhase; });
  reject("wrong memory capacity",[](auto &p) { --p.communication.memories.back().capacityBytes; });
  reject("resource overbooking",[](auto &p) { p.communication.resources.front().capacity = 0; });
  reject("unavailable communication link",[](auto &p) { p.communication.links.front().target = "system_memory_bo"; });
  reject("group without schedule dependency",[](auto &p) { p.executionGroups.back().dependencies.clear(); });
  reject("unproved loop reordering",[](auto &p) { std::swap(p.taskLoopOrders.back()[1],p.taskLoopOrders.back()[2]); });
  reject("wrong panel region",[](auto &p) { --p.readOnlyRetentions.front().sizes.front(); });
  reject("panel released before reuse",[](auto &p) { --p.readOnlyRetentions.front().endPhase; });
  reject("merge two independent panel execution groups",[](auto &p) {
    p.hardwareStages.front().endPhase = p.hardwareStages.back().endPhase;
    p.hardwareStages.resize(1);
    p.executionGroups.front().endPhase = p.executionGroups.back().endPhase;
    p.executionGroups.resize(1);
  });
  reject("panel has no location copy",[](auto &p) {
    auto &t = p.transfers.front();
    auto &event = p.communication.events[t.event];
    event.writes.clear(); t.targetCopies.clear();
  });
  reject("read stale same-region copy after overwrite",[](auto &p) {
    auto &c = p.communication;
    for (size_t e = 0; e < c.events.size(); ++e) {
      auto &event = c.events[e];
      if (event.kind != "compute") continue;
      for (auto &read : event.reads) {
        const auto &copy = c.copies[read];
        for (size_t old = 0; old < read; ++old)
          if (c.copies[old].allocation == copy.allocation &&
              c.versions[c.copies[old].version].region == c.versions[copy.version].region &&
              c.copies[old].releasedAfter < e) { read = old; return; }
      }
    }
  });
  reject("fake public task placement on native adapter",[](auto &p) {
    p.communication.placements.push_back({0,"invented.compute","invented.memory","invented.impl"});
  });
  reject("unplanned native transfer layout",[](auto &p) { p.communication.links.front().sourceLayout="transposed"; });
  reject("unplanned native transfer dtype",[](auto &p) { p.communication.links.front().targetDtype="i8"; });
  reject("unplanned native transfer device",[](auto &p) { p.communication.links.front().sourceDevice="other"; });
  reject("unplanned native transfer memory",[](auto &p) { p.communication.links.front().memory="other"; });
  reject("invented native link calibration",[](auto &p) { p.communication.links.front().provenance.source="measured"; });
  auto report = serializeGenericMapping(analysis,panel);
  const auto *object = report.getAsObject();
  if (!object->getObject("program_communication") ||
      object->getArray("discovered_mapping_opportunities")->size() < 2) ok = false;
  else ++checks;
  const char *indirectSource = R"mlir(
module attributes {orbit.implementation = "independent-lane-f32", orbit.vector_lanes = 2 : i64} {
  func.func @derived(%x: memref<8xf32> {orbit.role = "input"}, %y: memref<4xf32> {orbit.role = "output"}) attributes {orbit.noalias} {
    %done = taskflow.task @copy will_reads(%x : memref<8xf32>) will_writes(%y : memref<4xf32>) [original_read_memrefs(%x : memref<8xf32>), original_write_memrefs(%y : memref<4xf32>)] : (memref<8xf32>, memref<4xf32>) -> memref<4xf32> {
    ^bb0(%a: memref<8xf32>, %b: memref<4xf32>):
      affine.for %i = 0 to 4 {
        %twice = affine.apply affine_map<(d0) -> (d0 * 2)> (%i)
        %v = affine.load %a[%twice] : memref<8xf32>
        affine.store %v, %b[%i] : memref<4xf32>
      }
      taskflow.yield done_writes(%b : memref<4xf32>)
    }
    return
  }
})mlir";
  auto indirect = mlir::parseSourceString<mlir::ModuleOp>(indirectSource,&context);
  GenericAnalysis indirectAnalysis;
  GenericMappingPlan indirectPlan;
  llvm::json::Value indirectConfig = llvm::json::Object{{"generic_air",
      llvm::json::Object{{"tile_sizes",llvm::json::Array{2}}}}};
  if (!indirect || !analyzeGenericProgram(*indirect,indirectAnalysis,error) ||
      !planGenericMapping(indirectAnalysis,indirectConfig,indirectPlan,error) ||
      indirectPlan.taskImplementations.front().kind != "scalar") {
    llvm::errs() << "derived lane index falsely proved invariant/contiguous: " << error << "\n";
    ok = false;
  } else {
    ++checks;
    indirectPlan.taskImplementations.front() = {0,"independent-lane-f32",0,2};
    if (checkGenericMapping(indirectAnalysis,indirectPlan,error)) {
      llvm::errs() << "accepted fabricated vector proof for indirect lane index\n";
      ok = false;
    } else ++checks;
  }
  llvm::outs() << "{\"passed\":" << (ok ? "true" : "false") << ",\"checks\":" << checks << "}\n";
  return ok ? 0 : 1;
}
