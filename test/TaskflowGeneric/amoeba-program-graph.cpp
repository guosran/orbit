#include "Taskflow/AmoebaMapping.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <set>

using namespace orbit::taskflow;
// This is the actual v11 executable fanout body, including an unused SSA
// memory-state input that imposes ordering without reading its tensor bytes.
static const char *fanout = R"mlir(
module {
 func.func @fanout(%a: memref<16xf32> {orbit.role="input",orbit.name="A"},
                   %b: memref<16xf32> {orbit.role="output",orbit.name="B"},
                   %c: memref<16xf32> {orbit.role="output",orbit.name="C"}) attributes {orbit.noalias} {
  %x = memref.alloc() : memref<16xf32>
  %xd = taskflow.task @produce will_reads(%a : memref<16xf32>) will_writes(%x : memref<16xf32>) [original_read_memrefs(%a : memref<16xf32>), original_write_memrefs(%x : memref<16xf32>)] : (memref<16xf32>,memref<16xf32>) -> memref<16xf32> {
  ^bb0(%in: memref<16xf32>,%out: memref<16xf32>):
   affine.for %i=0 to 16 {
    %v=affine.load %in[%i] : memref<16xf32>
    %two=arith.constant 2.0 : f32
    %r=arith.mulf %v,%two : f32
    affine.store %r,%out[%i] : memref<16xf32>
   }
   taskflow.yield done_writes(%out : memref<16xf32>)
  }
  %bd = taskflow.task @first will_reads(%xd : memref<16xf32>) will_writes(%b : memref<16xf32>) [original_read_memrefs(%xd : memref<16xf32>), original_write_memrefs(%b : memref<16xf32>)] : (memref<16xf32>,memref<16xf32>) -> memref<16xf32> {
  ^bb0(%in: memref<16xf32>,%out: memref<16xf32>):
   affine.for %i=0 to 16 {
    %v=affine.load %in[%i] : memref<16xf32>
    affine.store %v,%out[%i] : memref<16xf32>
   }
   taskflow.yield done_writes(%out : memref<16xf32>)
  }
  %cd = taskflow.task @second will_reads(%xd,%bd : memref<16xf32>,memref<16xf32>) will_writes(%c : memref<16xf32>) [original_read_memrefs(%xd,%bd : memref<16xf32>,memref<16xf32>), original_write_memrefs(%c : memref<16xf32>)] : (memref<16xf32>,memref<16xf32>,memref<16xf32>) -> memref<16xf32> {
  ^bb0(%in: memref<16xf32>,%orderOnly: memref<16xf32>,%out: memref<16xf32>):
   affine.for %i=0 to 16 {
    %v=affine.load %in[%i] : memref<16xf32>
    affine.store %v,%out[%i] : memref<16xf32>
   }
   taskflow.yield done_writes(%out : memref<16xf32>)
  }
  memref.dealloc %x : memref<16xf32>
  return
 }
})mlir";

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: taskflow-amoeba-program-graph-test add-scale.mlir\n";
    return 2;
  }
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  unsigned checks = 0;
  bool ok = true;
  std::string error;
  auto require = [&](bool condition, const char *label) {
    if (condition) ++checks;
    else { ok = false; llvm::errs() << label << ": " << error << "\n"; }
  };
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis source;
  if (!module || !analyzeGenericProgram(*module, source, error)) {
    llvm::errs() << "actual source analysis failed: " << error << "\n";
    return 1;
  }
  require(source.tasks.size() == 2 && source.storages.size() == 4,
          "actual add-scale fixture source facts");
  AmoebaTargetContract target;
  target.fabricRows = target.fabricColumns = target.peRows = target.peColumns = 4;
  target.contextSlots = 12;
  // Explicit CPU-test target identities, not claims of a hardware measurement.
  target.architectureIdentity = staticFingerprint(llvm::json::Object{
      {"fixture", "mapper-arch-4x4-contract"}, {"fabric", 4}, {"pe", 4}, {"contexts", 12}});
  target.networkIdentity = staticFingerprint(llvm::json::Object{
      {"fixture", "directed-mesh-contract"}, {"bits_per_cycle", 32}, {"latency_cycles", 1}});
  std::vector<AmoebaTaskShape> shapes{{1, 1}, {1, 2}};
  ProgramGraph graph;
  if (!buildAmoebaProgramGraph(source, target, shapes, graph, error)) {
    llvm::errs() << error << "\n";
    return 1;
  }
  require(checkAmoebaProgramGraph(source, target, shapes, graph, error),
          "independent Amoeba adapter and common checks");
  require(graph.source.function == source.function && graph.instances.size() == 2 &&
          graph.instances[0].implementation == "amoeba-native" &&
          graph.instances[1].resourceCount == 2 &&
          graph.instances[1].resourceShape == std::vector<int64_t>({1, 2}),
          "borrowed actual source and CGRA block shape/count");
  require(graph.schedule.policy == "dependency-order-witness" &&
          !graph.schedule.estimatedMakespanMs && !graph.estimatedMakespanMs &&
          !graph.clockMHz && graph.computeEstimates.empty(),
          "dependency witness does not invent ML supply or milliseconds");
  bool timesUnknown = true;
  for (const auto &event : graph.schedule.events)
    timesUnknown &= !event.startMs && !event.endMs;
  require(timesUnknown, "all unpriced event timestamps remain unknown");
  require(graph.events.size() > 1 && graph.events[0].dependencies.empty() &&
          graph.events[1].dependencies.empty(), "independent imports do not inherit serial edges");
  size_t networkEvent = SIZE_MAX;
  int64_t networkBytes = 0, externalBytes = 0;
  unsigned routes = 0;
  for (size_t i = 0; i < graph.events.size(); ++i) {
    const auto &event = graph.events[i];
    if (event.kind != "transfer") continue;
    if (event.resource == "intertask.network") {
      ++routes; networkBytes += event.transferBytes; networkEvent = i;
      require(event.reads.size() == 1 && event.writes.size() == 1 &&
              graph.copies[event.reads.front()].version == graph.copies[event.writes.front()].version &&
              graph.versions[graph.copies[event.reads.front()].version].producer >= 0,
              "internal transfer preserves source-produced version and endpoint copies");
    } else externalBytes += event.transferBytes;
  }
  require(routes == 1 && networkBytes == 64 && externalBytes == 192 &&
          graph.transferBytes == 256, "one 64B internal transfer and explicit external I/O");
  StaticConfig architecture;
  architecture.domains = graph.domains; architecture.resources = graph.resources;
  architecture.memories = graph.memories; architecture.links = graph.links;
  auto exported = serializeMemoryArchitecture(architecture);
  bool unknown = true;
  for (const auto &value : *exported.getAsObject()->getArray("memories")) {
    auto memory = value.getAsObject();
    unknown &= memory->get("capacity_bytes")->getAsNull().has_value() &&
               memory->getBoolean("capacity_known") == false;
  }
  require(unknown, "all abstract endpoint capacities export null/false");
  auto cost = summarizeProgramGraphCost(graph);
  bool unverified = true;
  for (const auto &value : *cost.getAsObject()->getArray("allocation_accounting"))
    unverified &= value.getAsObject()->getBoolean("capacity_verified") == false &&
                  value.getAsObject()->get("capacity_bytes")->getAsNull().has_value();
  require(unverified, "logical peaks do not imply verified physical SRAM capacity");
  require(graph.dataEdges.size() == 3 && graph.orderEdges.empty() &&
          graph.versions.size() == 4 && graph.copies.size() == 8,
          "actual source/version/data edge/copy accounting");
  auto reject = [&](const char *label, std::function<void(ProgramGraph &)> mutate) {
    auto changed = graph; mutate(changed); error.clear();
    require(!checkAmoebaProgramGraph(source, target, shapes, changed, error) && !error.empty(), label);
  };
  reject("wrong copy version", [&](auto &g) { g.copies[g.events[networkEvent].writes.front()].version = 0; });
  reject("missing transfer read", [&](auto &g) { g.events[networkEvent].reads.clear(); });
  reject("missing creation dependence", [&](auto &g) { g.events[networkEvent].dependencies.clear(); });
  reject("wrong target binding", [](auto &g) { g.targetIdentity = "unbound-target"; });
  reject("wrong shape demand", [](auto &g) { g.placements.back().resourceCount = 3; });
  reject("wrong instance resource shape", [](auto &g) { g.instances.back().resourceShape = {2, 2}; });
  reject("unknown copy location", [](auto &g) { g.allocations.back().memory = "unmodeled-sram"; });
  reject("unreachable semantic route", [&](auto &g) {
    auto id = g.events[networkEvent].link;
    g.links.erase(std::remove_if(g.links.begin(), g.links.end(),
                    [&](const auto &link) { return link.id == id; }), g.links.end());
  });
  auto rejectBuild = [&](const char *label, const AmoebaTargetContract &t,
                         const std::vector<AmoebaTaskShape> &s) {
    ProgramGraph result; error.clear();
    require(!buildAmoebaProgramGraph(source, t, s, result, error) && !error.empty(), label);
  };
  rejectBuild("missing shape", target, {{1, 1}});
  rejectBuild("oversized fabric shape", target, {{1, 1}, {1, 5}});
  rejectBuild("invalid/nonrectangular shape encoding", target, {{1, 1}, {0, 2}});
  auto changedTarget = target; changedTarget.networkIdentity.clear();
  rejectBuild("missing network identity", changedTarget, shapes);
  changedTarget = target; changedTarget.contextSlots = -1;
  rejectBuild("invalid context count", changedTarget, shapes);
  changedTarget = target; changedTarget.networkIdentity = "other-network";
  error.clear();
  require(!checkAmoebaProgramGraph(source, changedTarget, shapes, graph, error),
          "target identity participates in graph check");

  // Clone the actual parsed task bodies into four disjoint constant-bound
  // row tiles. No communication JSON or invented arithmetic stands in for IR.
  mlir::OwningOpRef<mlir::ModuleOp> tiledModule(
      mlir::cast<mlir::ModuleOp>((*module)->clone()));
  std::vector<mlir::Operation *> originals;
  tiledModule->walk([&](mlir::taskflow::TaskflowTaskOp task) {
    originals.push_back(task.getOperation());
  });
  std::vector<mlir::Value> produced;
  auto restrictRow = [&](mlir::Operation *task, int64_t row, const std::string &name) {
    task->setAttr("task_name", mlir::StringAttr::get(&context, name));
    mlir::affine::AffineForOp outer;
    task->walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp loop) { if (!outer) outer = loop; });
    outer.setConstantLowerBound(row); outer.setConstantUpperBound(row + 1);
  };
  for (int64_t row = 0; row < 4; ++row) {
    mlir::IRMapping mapping;
    mlir::OpBuilder builder(originals[0]);
    auto *task = builder.clone(*originals[0], mapping);
    restrictRow(task, row, "add.row" + std::to_string(row));
    produced.push_back(task->getResult(0));
  }
  for (int64_t row = 0; row < 4; ++row) {
    mlir::IRMapping mapping;
    mapping.map(originals[0]->getResult(0), produced[row]);
    mlir::OpBuilder builder(originals[1]);
    auto *task = builder.clone(*originals[1], mapping);
    restrictRow(task, row, "scale.row" + std::to_string(row));
  }
  originals[1]->erase(); originals[0]->erase();
  GenericAnalysis tiledSource; ProgramGraph tiledGraph;
  std::vector<AmoebaTaskShape> tiledShapes(8);
  bool tiledAnalyzed = analyzeGenericProgram(*tiledModule, tiledSource, error);
  require(tiledAnalyzed, "real constant-bound task tile clones reanalyze");
  if (tiledAnalyzed) {
    bool built = buildAmoebaProgramGraph(tiledSource, target, tiledShapes, tiledGraph, error);
    require(built, "shared planner partitions true task tile access boundaries");
    if (built) {
      unsigned tileTransfers = 0; int64_t tileBytes = 0; bool exact = true;
      for (const auto &event : tiledGraph.events) if (event.resource == "intertask.network") {
        ++tileTransfers; tileBytes += event.transferBytes;
        exact &= event.transferBytes == 16 && event.transferRegion &&
                 event.transferRegion->sizes == std::vector<int64_t>({1, 4});
      }
      require(tileTransfers == 4 && tileBytes == 64 && exact &&
              tiledGraph.instances.size() == 8 && tiledGraph.regions.size() == 16,
              "four 16B internal transfers replace one 64B full-storage transfer");
      auto wrongCell = tiledGraph;
      for (auto &event : wrongCell.events) if (event.resource == "intertask.network") {
        ++event.transferRegion->offsets.front(); break;
      }
      require(!checkAmoebaProgramGraph(tiledSource, target, tiledShapes, wrongCell, error),
              "tile transfer cannot name an adjacent logical cell");
    }
  }

  auto fanoutModule = mlir::parseSourceString<mlir::ModuleOp>(fanout, &context);
  GenericAnalysis fanoutSource; ProgramGraph fanoutGraph;
  std::vector<AmoebaTaskShape> fanoutShapes(3);
  if (!fanoutModule || !analyzeGenericProgram(*fanoutModule, fanoutSource, error) ||
      !buildAmoebaProgramGraph(fanoutSource, target, fanoutShapes, fanoutGraph, error)) {
    llvm::errs() << "actual fanout fixture failed: " << error << "\n"; return 1;
  }
  int64_t fanoutNetworkBytes = 0; unsigned fanoutRoutes = 0;
  std::set<size_t> destinations;
  for (const auto &event : fanoutGraph.events) if (event.resource == "intertask.network") {
    fanoutNetworkBytes += event.transferBytes; ++fanoutRoutes;
    destinations.insert(fanoutGraph.copies[event.writes.front()].allocation);
    require(event.task == 1 || event.task == 2, "fanout transfer has actual consuming task");
    auto version = fanoutGraph.copies[event.reads.front()].version;
    auto producer = fanoutGraph.versions[version].producer;
    require(producer >= 0 && fanoutGraph.events[size_t(producer)].task == 0,
            "pure-order B is never a transported payload to C");
  }
  require(fanoutRoutes == 2 && fanoutNetworkBytes == 128 && destinations.size() == 2 &&
          fanoutGraph.orderEdges.size() == 1 && fanoutGraph.dataEdges.size() == 3,
          "phase-local fanout is two counted transfers with distinct consumer copies");
  auto withoutOrder = fanoutGraph;
  withoutOrder.orderEdges.clear();
  for (auto &event : withoutOrder.events) event.orderEdges.clear();
  error.clear();
  require(!checkAmoebaProgramGraph(fanoutSource, target, fanoutShapes, withoutOrder, error),
          "pure-order edge cannot disappear");
  auto sharedPolicy = fanoutGraph; sharedPolicy.dataPolicy = ProgramDataPolicy::RetainAcrossUsers;
  require(!checkAmoebaProgramGraph(fanoutSource, target, fanoutShapes, sharedPolicy, error),
          "shared-copy policy is not silently substituted");

  // Actual body changes are reanalyzed, never forged GenericAnalysis facts.
  std::string partial(fanout);
  auto upper = partial.find("affine.for %i=0 to 16");
  partial.replace(upper, std::string("affine.for %i=0 to 16").size(), "affine.for %i=0 to 8");
  auto partialModule = mlir::parseSourceString<mlir::ModuleOp>(partial, &context);
  GenericAnalysis partialSource; ProgramGraph unsupported;
  error.clear();
  bool partialAnalyzed = partialModule && analyzeGenericProgram(*partialModule, partialSource, error);
  require(!partialAnalyzed || !buildAmoebaProgramGraph(partialSource, target, fanoutShapes, unsupported, error),
          "partial non-whole-storage source rejects rather than padding facts");
  require(!error.empty(), "partial-region rejection carries diagnostic");
  std::string selfInit(fanout);
  auto firstLoad = selfInit.find("affine.load %in[%i]");
  selfInit.replace(firstLoad, std::string("affine.load %in[%i]").size(), "affine.load %out[%i]");
  auto selfModule = mlir::parseSourceString<mlir::ModuleOp>(selfInit, &context);
  GenericAnalysis selfSource; error.clear();
  bool selfAnalyzed = selfModule && analyzeGenericProgram(*selfModule, selfSource, error);
  require(!selfAnalyzed || !buildAmoebaProgramGraph(selfSource, target, fanoutShapes, unsupported, error),
          "local read-before-initialization cannot invent a version");
  require(!error.empty(), "local initialization rejection carries diagnostic");
  static const char *subbyte = R"mlir(
module {
 func.func @bits(%a: memref<16xi1> {orbit.role="input"},
                 %b: memref<16xi1> {orbit.role="output"}) attributes {orbit.noalias} {
  %done = taskflow.task @copy will_reads(%a : memref<16xi1>) will_writes(%b : memref<16xi1>) [original_read_memrefs(%a : memref<16xi1>), original_write_memrefs(%b : memref<16xi1>)] : (memref<16xi1>, memref<16xi1>) -> memref<16xi1> {
  ^bb0(%in: memref<16xi1>, %out: memref<16xi1>):
   affine.for %i=0 to 16 {
    %v=affine.load %in[%i] : memref<16xi1>
    affine.store %v,%out[%i] : memref<16xi1>
   }
   taskflow.yield done_writes(%out : memref<16xi1>)
  }
  return
 }
})mlir";
  auto bitModule = mlir::parseSourceString<mlir::ModuleOp>(subbyte, &context);
  GenericAnalysis bits; error.clear();
  bool bitsAnalyzed = bitModule && analyzeGenericProgram(*bitModule, bits, error);
  require(bitsAnalyzed, "actual i1 source is analyzed before adapter payload rejection");
  if (bitsAnalyzed) {
    require(!buildAmoebaProgramGraph(bits, target, {{1, 1}}, unsupported, error) &&
            error.find("byte-exact") != std::string::npos,
            "subbyte payload does not silently round to network bytes");
  }
  llvm::outs() << "{\"status\":\"" << (ok ? "passed" : "failed")
               << "\",\"checks\":" << checks
               << ",\"scope\":\"CPU semantic adapter; no mapper or hardware execution claim\"}\n";
  return ok ? 0 : 1;
}
