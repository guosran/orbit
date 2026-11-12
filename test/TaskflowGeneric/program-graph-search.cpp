#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramMapping.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>
using namespace orbit::taskflow;
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

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs()
        << "usage: taskflow-program-graph-search-test gemm-scale.mlir\n";
    return 2;
  }
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  std::string error;
  unsigned checks = 0, failures = 0;
  auto require = [&](bool value, const char *name) {
    if (value)
      ++checks;
    else {
      ++failures;
      llvm::errs() << name << ": " << error << "\n";
    }
    error.clear();
    return value;
  };
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  GenericAnalysis a;
  if (!module || !analyzeGenericProgram(*module, a, error)) {
    llvm::errs() << error;
    return 1;
  }
  // Compare against the exact historical direct fn.print hash, not merely
  // new-analysis versus new-analysis (both could share a changed print path).
  std::string historicalText;
  llvm::raw_string_ostream historicalStream(historicalText);
  a.function.print(historicalStream);
  const auto historicalFingerprint =
      staticFingerprint(llvm::json::Value(historicalText));
  require(a.bodyFingerprint == historicalFingerprint,
          "plain analysis preserves the historical direct function-print "
          "fingerprint");
  // Internal search lineage must not change source/calibration identity, and
  // fingerprinting must never remove the caller's live operation attributes.
  auto *lineageTask = a.tasks.front().operation;
  auto marker =
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 41);
  lineageTask->setAttr("orbit.search_origin", marker);
  a.function->setAttr("orbit.search_origin", marker);
  GenericAnalysis tagged;
  require(analyzeGenericProgram(*module, tagged, error) &&
              tagged.bodyFingerprint == historicalFingerprint,
          "adding internal search lineage preserves source fingerprint");
  require(lineageTask->getAttr("orbit.search_origin") == marker &&
              a.function->getAttr("orbit.search_origin") == marker,
          "fingerprinting leaves original function and task lineage intact");
  auto changedMarker =
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 97);
  lineageTask->setAttr("orbit.search_origin", changedMarker);
  require(analyzeGenericProgram(*module, tagged, error) &&
              tagged.bodyFingerprint == historicalFingerprint &&
              lineageTask->getAttr("orbit.search_origin") == changedMarker,
          "changing lineage preserves fingerprint and changed live attribute");
  a.function->setAttr("orbit.source_semantic_contract",
                      mlir::StringAttr::get(&context, "different"));
  require(analyzeGenericProgram(*module, tagged, error) &&
              tagged.bodyFingerprint != a.bodyFingerprint,
          "other source semantic attributes remain fingerprinted");
  a.function->removeAttr("orbit.source_semantic_contract");
  std::string arithmeticSource = source;
  auto product = arithmeticSource.find("arith.mulf");
  arithmeticSource.replace(product, std::string("arith.mulf").size(),
                           "arith.addf");
  auto arithmeticModule =
      mlir::parseSourceString<mlir::ModuleOp>(arithmeticSource, &context);
  GenericAnalysis arithmeticAnalysis;
  require(
      arithmeticModule &&
          analyzeGenericProgram(*arithmeticModule, arithmeticAnalysis, error) &&
          arithmeticAnalysis.bodyFingerprint != a.bodyFingerprint,
      "real multiply-to-add arithmetic change still changes source "
      "fingerprint");
  lineageTask->removeAttr("orbit.search_origin");
  a.function->removeAttr("orbit.search_origin");
  StaticConfig target;
  target.resources = {{"cpu", "compute", "cpu", 4},
                      {"core0", "compute", "accel", 2},
                      {"core1", "compute", "accel", 2},
                      {"dma", "transfer", "system", 1}};
  target.memories = {{"host", 4096, {"cpu", "accel"}},
                     {"local0", 4096, {"accel"}},
                     {"local1", 4096, {"accel"}},
                     {"relay", 4096, {"accel"}}};
  target.memories[1].accessibleResources = {"core0"};
  target.memories[2].accessibleResources = {"core1"};
  auto link = [](const std::string &id, const std::string &from,
                 const std::string &to) {
    StaticTransfer l;
    l.id = id;
    l.source = from;
    l.target = to;
    l.resource = "dma";
    return l;
  };
  for (const auto &local : {"local0", "local1"}) {
    target.links.push_back(link(std::string("in-") + local, "host", local));
    target.links.push_back(link(std::string("out-") + local, local, "host"));
  }
  target.links.push_back(link("host-relay", "host", "relay"));
  target.links.push_back(link("relay-local0", "relay", "local0"));
  target.links.push_back(link("parallel-local0", "host", "local0"));
  target.links.push_back(link("core0-core1", "local0", "local1"));
  target.assumptions.initialLocations = {{"A", "host"}};
  target.assumptions.outputLocations = {{"B", "local0"}, {"C", "local0"}};
  ProgramDataOrganizationChoice original;
  original.placements = {{0, "cpu", "host", "scalar"},
                         {1, "core0", "local0", "scalar"},
                         {2, "core0", "local0", "scalar"}};
  ProgramGraph reference;
  if (!planProgramDataOrganization(a, target, original, reference, error)) {
    llvm::errs() << error;
    return 1;
  }
  auto rebound = [&](const ProgramDataOrganizationChoice &choice,
                     ProgramGraph &g, const char *name) {
    return require(rebindProgramGraph(reference, target, choice, g, error),
                   name);
  };
  ProgramGraph local, retained;
  if (rebound(original, local, "whole-task phase-local rebind")) {
    require(checkProgramGraph(local, error) &&
                checkProgramGraphSource(a, local, error),
            "rebound source and common checker");
    require(local.transferBytes == 128 &&
                local.instances.size() == reference.instances.size(),
            "phase-local copy accounting");
    require(local.orderEdges.size() == 1, "preserve source order-only edge");
    auto refreshed = local;
    auto identity = programGraphIdentity(local);
    refreshed.schedule.events.clear();
    refreshed.schedule.uncalibrated.clear();
    require(refreshProgramGraphSchedule(refreshed, error) &&
                programGraphIdentity(refreshed) == identity,
            "schedule refresh reconstructs witness without changing graph "
            "identity");
    require(refreshed.instances.size() == local.instances.size() &&
                refreshed.versions.size() == local.versions.size(),
            "schedule refresh preserves source instances and versions");
    auto bad = local;
    bad.orderEdges.clear();
    require(!checkProgramGraph(bad, error),
            "missing source order independently rejected");
    bad = local;
    bad.source.tasks.back().dependencies.clear();
    require(!checkProgramGraphSource(a, bad, error),
            "tampered source dependencies rejected");
    bad = local;
    bad.outputLocations[1] = "missing";
    require(!checkProgramGraph(bad, error), "invalid output boundary rejected");
  }
  auto choice = original;
  choice.policy = ProgramDataPolicy::RetainAcrossUsers;
  if (rebound(choice, retained, "whole-task retained rebind")) {
    require(retained.transferBytes == 64,
            "retained source copy delivered once");
    if (!local.instances.empty())
      require(retained.transferBytes < local.transferBytes,
              "same source policy choice changes real traffic");
  }
  size_t shared = SIZE_MAX;
  for (auto &access : a.tasks[0].accesses)
    if (access.write)
      shared = access.storage;
  auto explicitRoute = original;
  explicitRoute.transferChoices = {
      {shared, 1, "local0", "host", {"host-relay", "relay-local0"}},
      {shared, 2, "local0", "host", {"parallel-local0"}}};
  ProgramGraph routed;
  if (rebound(explicitRoute, routed,
              "nonshortest explicit multihop plus parallel link")) {
    std::vector<std::string> links;
    for (auto &event : routed.events)
      if (event.kind == "transfer")
        links.push_back(event.link);
    require(links == std::vector<std::string>{"host-relay", "relay-local0",
                                              "parallel-local0"},
            "exact chosen physical links");
  }
  auto unavailable = explicitRoute;
  unavailable.transferChoices[0] = {shared, 1, "local0", "local0", {}};
  ProgramGraph rejected;
  require(!rebindProgramGraph(reference, target, unavailable, rejected, error),
          "selected unavailable source never repaired");
  auto extra = original;
  extra.transferChoices = {{shared, 0, "host", "host", {}}};
  require(!rebindProgramGraph(reference, target, extra, rejected, error),
          "unconsumed explicit source choice rejected");
  auto twoCore = original;
  twoCore.placements[1].resourceCount = 2;
  twoCore.placements[1].resourceShape = {1, 2};
  twoCore.placements[2] = {2, "core1", "local1", "scalar", 2, {2, 1}};
  auto twoTarget = target;
  twoTarget.assumptions.outputLocations["C"] = "local1";
  ProgramGraph placed;
  if (require(rebindProgramGraph(reference, twoTarget, twoCore, placed, error),
              "two-core binding and explicit resource geometry")) {
    require(placed.placements[1].resourceCount == 2 &&
                placed.placements[1].resourceShape ==
                    std::vector<int64_t>{1, 2},
            "placement retains resource count and shape");
    bool found = false;
    for (auto &instance : placed.instances)
      if (instance.task == 2)
        found = instance.resourceCount == 2 &&
                instance.resourceShape == std::vector<int64_t>{2, 1};
    require(found, "instance consumes exact geometry");
  }
  auto badShape = original;
  badShape.placements[1].resourceCount = 2;
  badShape.placements[1].resourceShape = {3};
  require(!rebindProgramGraph(reference, target, badShape, rejected, error),
          "mismatched resource geometry rejected");
  // Export B from the copy imported by a later consumer, not B's original
  // compute producer. Both source tasks and execution groups are real bindings.
  std::string consumerSource = source;
  auto consumerAt = consumerSource.find("%cd = taskflow.task");
  auto loadAt = consumerSource.find("affine.load %in[", consumerAt);
  consumerSource.replace(loadAt, std::string("affine.load %in[").size(),
                         "affine.load %orderOnly[");
  auto consumerModule =
      mlir::parseSourceString<mlir::ModuleOp>(consumerSource, &context);
  GenericAnalysis consumers;
  auto groupedTarget = twoTarget;
  groupedTarget.assumptions.outputLocations = {{"B", "host"}, {"C", "host"}};
  auto groupedChoice = twoCore;
  groupedChoice.policy = ProgramDataPolicy::RetainAcrossUsers;
  ProgramGraph groupedReference, exported;
  if (require(consumerModule &&
                  analyzeGenericProgram(*consumerModule, consumers, error) &&
                  planProgramDataOrganization(consumers, groupedTarget,
                                              groupedChoice, groupedReference,
                                              error),
              "real later consumer with retained imported output")) {
    groupedReference.executionGroups.resize(consumers.tasks.size());
    for (size_t group = 0; group < groupedReference.executionGroups.size();
         ++group) {
      auto &g = groupedReference.executionGroups[group];
      g.beginPhase = group;
      g.endPhase = group + 1;
      g.resource = groupedChoice.placements[group].resource;
      g.memory = groupedChoice.placements[group].memory;
      for (size_t event = 0; event < groupedReference.events.size(); ++event)
        if (groupedReference.events[event].phase == group)
          g.memberEvents.push_back(event);
      std::set<size_t> parents;
      for (auto event : g.memberEvents)
        for (auto dep : groupedReference.events[event].dependencies)
          if (groupedReference.events[dep].phase != group)
            parents.insert(groupedReference.events[dep].phase);
      g.dependencies.assign(parents.begin(), parents.end());
    }
    if (require(checkProgramGraph(groupedReference, error),
                "source-authenticated three-group reference")) {
      size_t bStorage = SIZE_MAX;
      for (size_t st = 0; st < consumers.storages.size(); ++st)
        if (consumers.storages[st].id == "B")
          bStorage = st;
      groupedChoice.transferChoices = {
          {bStorage, consumers.tasks.size(), "host", "local1", {"out-local1"}}};
      if (require(rebindProgramGraph(groupedReference, groupedTarget,
                                     groupedChoice, exported, error),
                  "final export owned by actual later source-copy group")) {
        bool ownerFound = false;
        for (auto event : exported.executionGroups[2].memberEvents) {
          auto &e = exported.events[event];
          if (e.kind != "transfer" || e.link != "out-local1" ||
              !e.transferRegion || e.transferRegion->storage != bStorage)
            continue;
          auto &copy = exported.copies[e.reads.front()];
          ownerFound = copy.createdBy >= 0 &&
                       exported.events[copy.createdBy].kind == "transfer";
        }
        require(ownerFound,
                "later imported B source and export owner are explicit");
      }
    }
  }
  // Three-dimensional access coverage is inherited from actual source
  // instances.
  std::string rank3 = source;
  auto replaceAll = [](std::string &s, const std::string &from,
                       const std::string &to) {
    size_t at = 0;
    while ((at = s.find(from, at)) != std::string::npos) {
      s.replace(at, from.size(), to);
      at += to.size();
    }
  };
  replaceAll(rank3, "16xf32", "2x2x4xf32");
  replaceAll(
      rank3, "affine.for %i=0 to 16 {",
      "affine.for %i=0 to 2 { affine.for %j=0 to 2 { affine.for %k=0 to 4 {");
  replaceAll(rank3, "[%i]", "[%i,%j,%k]");
  replaceAll(rank3, "   }\n   taskflow.yield", "   } } }\n   taskflow.yield");
  auto tensor = mlir::parseSourceString<mlir::ModuleOp>(rank3, &context);
  GenericAnalysis t;
  ProgramGraph tensorReference, tensorResult;
  if (require(tensor && analyzeGenericProgram(*tensor, t, error) &&
                  planProgramDataOrganization(t, target, original,
                                              tensorReference, error),
              "rank-three real source reference"))
    if (require(rebindProgramGraph(tensorReference, target, choice,
                                   tensorResult, error),
                "rank-three source graph rebind"))
      require(tensorResult.instances.front().domain.size() == 3 &&
                  checkProgramGraphSource(t, tensorResult, error),
              "rank-three source iteration coverage");
  // Actual GEMM has an initialized accumulation and tiled partial-K versions.
  auto native = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis b;
  GenericMappingPlan nativePlan;
  llvm::json::Value config = llvm::json::Object{
      {"generic_air",
       llvm::json::Object{{"tile_sizes", llvm::json::Array{2, 2, 4}}}}};
  if (require(native && distributeNativeInitialization(*native, error) &&
                  analyzeGenericProgram(*native, b, error) &&
                  planGenericMapping(b, config, nativePlan, error),
              "native tiled GEMM reference")) {
    const auto &ref = nativePlan.communication;
    auto arch = nativePlan.memoryBinding.architecture;
    ProgramDataOrganizationChoice bindings;
    bindings.policy = ProgramDataPolicy::RetainAcrossUsers;
    for (size_t task = 0; task < b.tasks.size(); ++task) {
      auto it = std::find_if(ref.instances.begin(), ref.instances.end(),
                             [&](const auto &i) { return i.task == task; });
      std::string implementation =
          it == ref.instances.end() ? "scalar" : it->implementation;
      bindings.placements.push_back(
          {task, nativePlan.memoryBinding.computeResource,
           nativePlan.memoryBinding.localMemory, implementation});
    }
    std::map<size_t, std::set<int64_t>> writers;
    for (auto &v : ref.versions)
      if (v.producer >= 0)
        writers[v.region].insert(v.producer);
    bool partialVersions = false;
    for (auto &entry : writers)
      partialVersions |= entry.second.size() > 1;
    require(partialVersions && ref.instances.size() > b.tasks.size(),
            "GEMM reference really has multiple source-region writers");
    ProgramGraph gemm;
    if (require(rebindProgramGraph(ref, arch, bindings, gemm, error),
                "GEMM reduction and partial-K graph rebind")) {
      require(gemm.versions.size() == ref.versions.size() &&
                  gemm.instances.size() == ref.instances.size(),
              "all native versions and instances preserved");
      bool sameDomains = true;
      for (size_t i = 0; i < ref.instances.size(); ++i) {
        auto &left = ref.instances[i].domain;
        auto &right = gemm.instances[i].domain;
        sameDomains &= left.size() == right.size();
        for (size_t j = 0; j < std::min(left.size(), right.size()); ++j)
          sameDomains &= left[j].loop == right[j].loop &&
                         left[j].lower == right[j].lower &&
                         left[j].upper == right[j].upper &&
                         left[j].step == right[j].step;
      }
      require(sameDomains && checkProgramGraphSource(b, gemm, error),
              "GEMM source domains not reinterpreted as pointwise");
      require(gemm.outputLocations == ref.outputLocations,
              "complete native output boundary preserved");
      std::set<size_t> members;
      for (auto &group : gemm.executionGroups)
        for (auto event : group.memberEvents)
          members.insert(event);
      require(
          members.size() == gemm.events.size(),
          "native execution groups own every final export and compute event");
      require(gemm.executionRegions.size() == 1 &&
                  gemm.executionRegions.front().memberEvents.size() ==
                      gemm.events.size(),
              "native invocation retains explicit complete event membership");
      auto missing = gemm;
      for (auto &copy : missing.copies)
        if (gemm.source
                    .storages[gemm.regions[gemm.versions[copy.version].region]
                                  .storage]
                    .role == "output" &&
            copy.releasedAfter == gemm.events.size()) {
          copy.releasedAfter = copy.availableFrom + 1;
          break;
        }
      require(!checkProgramGraph(missing, error),
              "incomplete native final output rejected");
    }
  }
  llvm::outs() << llvm::json::Object{{"status", failures ? "FAIL" : "PASS"},
                                     {"checks", int64_t(checks)},
                                     {"failures", int64_t(failures)},
                                     {"hardware_executed", false}}
               << "\n";
  return failures ? 1 : 0;
}
