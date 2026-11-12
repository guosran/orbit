#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramMapping.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <functional>
using namespace orbit::taskflow;
// Every node and edge below is recovered from this executable Taskflow body.
static const char *source = R"mlir(
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
int main() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  GenericAnalysis a;
  std::string error;
  unsigned checks = 0;
  if (!module || !analyzeGenericProgram(*module, a, error)) {
    llvm::errs() << error;
    return 1;
  }
  auto require = [&](bool condition, const char *name) {
    if (!condition) {
      llvm::errs() << name << ": " << error << "\n";
      std::exit(1);
    }
    ++checks;
  };
  size_t intermediate = SIZE_MAX;
  for (auto &access : a.tasks[0].accesses)
    if (access.write)
      intermediate = access.storage;
  StaticConfig target;
  target.resources = {{"cpu", "compute", "cpu", 4},
                      {"accel", "compute", "accel", 2},
                      {"dma", "transfer", "system", 1}};
  target.memories = {{"dram", 4096, {"cpu", "accel"}},
                     {"sram", 4096, {"accel"}},
                     {"relay", 4096, {"accel"}}};
  auto link = [](const char *id, const char *from, const char *to) {
    StaticTransfer l;
    l.id = id;
    l.source = from;
    l.target = to;
    l.resource = "dma";
    l.latencyMs = .1;
    l.bandwidthBytesPerMs = 64;
    l.provenance = {"synthetic", "routing regression, not hardware timing"};
    return l;
  };
  target.links = {link("parallel-first", "dram", "sram"),
                  link("parallel-second", "dram", "sram"),
                  link("to-relay", "dram", "relay"),
                  link("via-relay", "relay", "sram"),
                  link("from-sram", "sram", "relay"),
                  link("export", "sram", "dram")};
  target.assumptions.initialLocations = {{"A", "dram"}};
  target.assumptions.outputLocations = {{"B", "sram"}, {"C", "sram"}};
  ProgramDataOrganizationChoice choice;
  choice.placements = {{0, "cpu", "dram", "scalar", 4, {2, 2}},
                       {1, "accel", "sram", "scalar", 2, {2}},
                       {2, "accel", "sram", "scalar"}};
  choice.transferChoices = {
      {intermediate, 1, "sram", "dram", {"to-relay", "via-relay"}},
      {intermediate, 2, "sram", "dram", {"parallel-second"}}};
  ProgramCommunicationPlan plan;
  require(planProgramDataOrganization(a, target, choice, plan, error),
          "explicit nonshortest and second parallel link");
  require(checkProgramDataOrganization(a, target, choice, plan, error),
          "independent selected route check");
  std::vector<std::string> routes;
  for (const auto &event : plan.events)
    if (event.kind == "transfer")
      routes.push_back(event.link);
  require(routes == std::vector<std::string>{"to-relay", "via-relay",
                                             "parallel-second"},
          "exact selected route sequence");
  require(plan.instances.front().resourceCount == 4 &&
              plan.instances.front().resourceShape ==
                  std::vector<int64_t>{2, 2},
          "resource shape propagated into public instance");
  auto other = choice;
  other.transferChoices[1].links = {"parallel-first"};
  require(!checkProgramDataOrganization(a, target, other, plan, error),
          "checker rejects valid alternative unselected parallel route");
  auto malformed = choice;
  malformed.transferChoices[0].links = {"to-relay", "parallel-second"};
  require(!planProgramDataOrganization(a, target, malformed, plan, error),
          "disconnected route not repaired");
  malformed = choice;
  malformed.transferChoices[0].sourceMemory = "sram";
  malformed.transferChoices[0].links.clear();
  require(!planProgramDataOrganization(a, target, malformed, plan, error),
          "missing selected current-version source not repaired");
  malformed = choice;
  malformed.transferChoices.push_back(malformed.transferChoices.front());
  require(!planProgramDataOrganization(a, target, malformed, plan, error),
          "duplicate demand choice rejected");
  malformed = choice;
  malformed.transferChoices.front().consumerTask = 0;
  require(!planProgramDataOrganization(a, target, malformed, plan, error),
          "unconsumed extra choice rejected");
  malformed = choice;
  malformed.transferChoices.front().storage = a.storages.size();
  require(!planProgramDataOrganization(a, target, malformed, plan, error),
          "unknown storage choice rejected");
  auto incompatible = target;
  incompatible.links[1].sourceDtype = "i32";
  require(!planProgramDataOrganization(a, incompatible, choice, plan, error),
          "selected incompatible parallel link rejected");
  auto automatic = choice;
  automatic.transferChoices.clear();
  incompatible = target;
  incompatible.links.front().sourceDtype = "i32";
  require(planProgramDataOrganization(a, incompatible, automatic, plan, error),
          "default route keeps compatible second parallel link");
  require(std::all_of(plan.events.begin(), plan.events.end(),
                      [](const auto &e) {
                        return e.kind != "transfer" ||
                               e.link == "parallel-second";
                      }),
          "default parallel link compatibility");
  auto retained = choice;
  retained.policy = ProgramDataPolicy::RetainAcrossUsers;
  retained.transferChoices = {
      {intermediate, 1, "sram", "dram", {"parallel-second"}},
      {intermediate, 2, "sram", "sram", {}}};
  require(planProgramDataOrganization(a, target, retained, plan, error),
          "explicit local retained source");
  auto relayTarget = target;
  relayTarget.assumptions.outputLocations["C"] = "relay";
  retained.placements[2].memory = "relay";
  retained.transferChoices[1] = {
      intermediate, 2, "relay", "sram", {"from-sram"}};
  require(planProgramDataOrganization(a, relayTarget, retained, plan, error),
          "explicit imported source preferred over producer copy");
  require(checkProgramDataOrganization(a, relayTarget, retained, plan, error),
          "imported source independently checked");
  auto phaseLocal = retained;
  phaseLocal.policy = ProgramDataPolicy::PhaseLocal;
  require(!planProgramDataOrganization(a, relayTarget, phaseLocal, plan, error),
          "released imported source not silently reconstructed");
  auto exported = choice;
  auto exportTarget = target;
  exportTarget.assumptions.outputLocations["B"] = "dram";
  size_t outputB = SIZE_MAX;
  for (size_t i = 0; i < a.storages.size(); ++i)
    if (a.storages[i].id == "B")
      outputB = i;
  exported.transferChoices.push_back(
      {outputB, a.tasks.size(), "dram", "sram", {"export"}});
  require(planProgramDataOrganization(a, exportTarget, exported, plan, error),
          "explicit final output boundary route");
  auto wrongShape = choice;
  wrongShape.placements[0].resourceShape = {2, 3};
  require(!planProgramDataOrganization(a, target, wrongShape, plan, error),
          "shape count mismatch rejected");
  wrongShape = choice;
  wrongShape.placements[0].resourceCount = 5;
  wrongShape.placements[0].resourceShape.clear();
  require(!planProgramDataOrganization(a, target, wrongShape, plan, error),
          "resource overcapacity rejected");
  require(planProgramDataOrganization(a, target, choice, plan, error),
          "restore selected route fixture");
  plan.placements[0].resourceShape = {4};
  require(!checkProgramDataOrganization(a, target, choice, plan, error),
          "independent checker rejects changed shape with same count");
  auto architecture = serializeMemoryArchitecture(target);
  auto *arch = architecture.getAsObject();
  (*arch)["assumptions"] = llvm::json::Object{
      {"initial_locations", llvm::json::Object{{"A", "dram"}}},
      {"output_locations", llvm::json::Object{{"B", "sram"}, {"C", "sram"}}}};
  llvm::json::Array placementJSON;
  for (const auto &p : choice.placements) {
    llvm::json::Array shape;
    for (auto n : p.resourceShape)
      shape.push_back(n);
    placementJSON.push_back(
        llvm::json::Object{{"task", int64_t(p.task)},
                           {"resource", p.resource},
                           {"memory", p.memory},
                           {"implementation", p.implementation},
                           {"resource_count", p.resourceCount},
                           {"resource_shape", std::move(shape)}});
  }
  llvm::json::Value config = llvm::json::Object{
      {"program_graph",
       llvm::json::Object{
           {"adapter", "whole-task"},
           {"architecture", std::move(architecture)},
           {"choice",
            llvm::json::Object{{"policy", "phase-local"},
                               {"placements", std::move(placementJSON)}}},
           {"transfer_choices",
            llvm::json::Array{llvm::json::Object{
                {"storage", int64_t(intermediate)},
                {"consumer_task", 1},
                {"source_memory", "dram"},
                {"target_memory", "sram"},
                {"links", llvm::json::Array{"parallel-second"}}}}}}}};
  StaticConfig parsedTarget;
  ProgramDataOrganizationChoice parsed;
  require(parseWholeTaskProgramGraphConfig(config, parsedTarget, parsed, error),
          "parse explicit choices and resource shape");
  require(parsed.transferChoices.size() == 1 &&
              parsed.placements[0].resourceCount == 4 &&
              parsed.placements[0].resourceShape == std::vector<int64_t>{2, 2},
          "parser preserves complete typed choices");
  require(planProgramDataOrganization(a, parsedTarget, parsed, plan, error),
          "parsed selection goes through existing planner");
  auto &entry = *config.getAsObject()
                     ->getObject("program_graph")
                     ->getArray("transfer_choices")
                     ->front()
                     .getAsObject();
  entry["extra"] = true;
  require(
      !parseWholeTaskProgramGraphConfig(config, parsedTarget, parsed, error),
      "parser rejects extra choice fields");
  entry.erase("extra");
  entry["links"] = llvm::json::Array{"unknown"};
  require(
      !parseWholeTaskProgramGraphConfig(config, parsedTarget, parsed, error),
      "parser rejects unknown selected link");
  llvm::outs() << "program-routing-search: " << checks << " checks passed\n";
  return 0;
}
