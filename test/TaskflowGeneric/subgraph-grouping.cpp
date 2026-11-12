#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace orbit::taskflow;
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis facts;
  std::string error;
  unsigned checks = 0;
  auto require = [&](bool value, const char *name) {
    if (value)
      ++checks;
    else
      llvm::errs() << name << ": " << error << "\n";
    return value;
  };
  if (!require(module && analyzeGenericProgram(*module, facts, error) &&
                   facts.tasks.size() == 7,
               "seven-task source"))
    return 1;
  auto discovered = discoverGenericMappingChoices(facts);
  ProgramMappingChoice first, second;
  for (const auto &choice : discovered)
    if (choice.residencyGroups.size() == 1) {
      auto tasks = choice.residencyGroups.front().tasks;
      if (tasks == std::vector<size_t>{1, 2})
        first = choice;
      if (tasks == std::vector<size_t>{4, 5})
        second = choice;
    }
  if (!require(first.residencyGroups.size() == 1 &&
                   second.residencyGroups.size() == 1,
               "access-based local scope discovery"))
    return 1;
  llvm::json::Value config = llvm::json::Object{
      {"generic_air",
       llvm::json::Object{{"tile_sizes", llvm::json::Array{4, 64, 1}}}}};
  ProgramMappingChoice plain;
  plain.residency = "tiled-phase-local";
  plain.grouping = "task-shape-stages";
  plain.dataOrganization = "phase-local";
  auto build = [&](const ProgramMappingChoice &choice,
                   GenericMappingPlan &plan) {
    return setNativeMappingChoice(*module, choice, error) &&
           analyzeGenericProgram(*module, facts, error) &&
           planGenericMapping(facts, config, plan, error);
  };
  GenericMappingPlan baseline, g1, g2, both;
  if (!require(build(plain, baseline), "materialized baseline") ||
      !require(build(first, g1), "partial group with external consumer") ||
      !require(build(second, g2), "partial group with prefix and suffix"))
    return 1;
  auto combined = first;
  combined.residencyGroups.push_back(second.residencyGroups.front());
  if (!require(build(combined, both) && both.residencyScopes.size() == 2,
               "two disjoint local scopes"))
    return 1;
  size_t x = first.residencyGroups.front().retainedStorages.front(),
         v = second.residencyGroups.front().retainedStorages.front();
  auto host = [&](const GenericMappingPlan &p, size_t storage) {
    return llvm::any_of(p.communication.allocations, [&](const auto &a) {
      return a.storage == storage && a.memory == p.memoryBinding.hostMemory;
    });
  };
  if (!require(host(both, x) && !host(both, v),
               "boundary export keeps X BO while internal V has no BO"))
    return 1;
  if (!require(both.residencyScopes[0].boundaryExports ==
                       std::vector<size_t>{x} &&
                   both.residencyScopes[1].boundaryExports.empty(),
               "scope boundary derived from outside consumers"))
    return 1;
  int64_t cells = 1;
  for (auto n : facts.storages[x].shape)
    cells *= n;
  int64_t bytes = cells * 4;
  if (!require(baseline.communication.transferBytes -
                       g1.communication.transferBytes ==
                   bytes,
               "shared X eliminates only internal reload") ||
      !require(baseline.communication.transferBytes -
                       g2.communication.transferBytes ==
                   2 * bytes,
               "internal V eliminates writeback and reload") ||
      !require(baseline.communication.transferBytes -
                       both.communication.transferBytes ==
                   3 * bytes,
               "scope traffic composes without global dropping"))
    return 1;
  bool sharedAndExported = false;
  for (const auto &copy : both.communication.copies)
    if (both.communication
                .regions[both.communication.versions[copy.version].region]
                .storage == x &&
        both.communication.allocations[copy.allocation].memory ==
            both.memoryBinding.localMemory) {
      bool reader = false, exporter = false;
      for (auto user : copy.users) {
        reader |= both.communication.events[user].kind == "compute";
        exporter |= both.communication.events[user].kind == "transfer";
      }
      sharedAndExported |= reader && exporter;
    }
  if (!require(sharedAndExported,
               "same version copy serves internal reader and boundary export"))
    return 1;
  bool prefix = false, suffix = false, distinctDependency = false;
  for (auto &phase : both.phases) {
    prefix |= phase.task == 0;
    suffix |= phase.task == 6;
  }
  for (const auto &group : both.executionGroups)
    distinctDependency |=
        group.dependencies != group.implementationDependencies;
  if (!require(prefix && suffix, "prefix and suffix coverage retained") ||
      !require(
          distinctDependency,
          "semantic dependencies distinguish implementation serialization"))
    return 1;
  if (!require(build(first, g1),
               "first-group source for compositional discovery"))
    return 1;
  auto proposals = discoverGenericMappingChoices(facts);
  bool combinedDiscovered = false;
  for (auto &choice : proposals)
    combinedDiscovered |= choice.residencyGroups == combined.residencyGroups;
  if (!require(combinedDiscovered,
               "original action discovery composes an additional group"))
    return 1;
  auto rejected = [&](ProgramMappingChoice choice, const char *name) {
    GenericMappingPlan p;
    error.clear();
    return require(!build(choice, p), name);
  };
  auto overlap = combined;
  overlap.residencyGroups.push_back(first.residencyGroups.front());
  if (!rejected(overlap, "overlapping group rejection"))
    return 1;
  auto noncontiguous = first;
  noncontiguous.residencyGroups[0].tasks = {1, 3};
  GenericMappingPlan reordered;
  if (!require(
          build(noncontiguous, reordered) &&
              reordered.residencyScopes.front().tasks ==
                  std::vector<size_t>({1, 3}),
          "dependency-convex noncontiguous group preserves task identities"))
    return 1;
  auto wrongProducer = second;
  wrongProducer.residencyGroups[0].retainedStorages = {x};
  if (!rejected(wrongProducer,
                "retained producer outside selected scope rejected"))
    return 1;
  if (!require(build(combined, both), "restore valid graph"))
    return 1;
  auto bad = both;
  bad.residencyScopes.front().boundaryExports.clear();
  bad.planId.clear();
  if (!require(!checkGenericMapping(facts, bad, error),
               "fabricated missing boundary export rejected"))
    return 1;
  bad = both;
  bad.executionGroups.back().implementationDependencies.clear();
  bad.planId.clear();
  if (!require(!checkGenericMapping(facts, bad, error),
               "missing implementation control rejected"))
    return 1;
  bad = both;
  auto retained = both.residencyScopes.front().retainedStorages.front();
  bad.transfers.erase(std::remove_if(bad.transfers.begin(), bad.transfers.end(),
                                     [&](auto &t) {
                                       return t.storage == retained &&
                                              t.direction == "l1_to_host";
                                     }),
                      bad.transfers.end());
  bad.planId.clear();
  if (!require(!checkGenericMapping(facts, bad, error),
               "outside reader cannot lose materialized version"))
    return 1;
  bad = both;
  bool semanticParent = false;
  for (auto &group : bad.executionGroups)
    if (!group.dependencies.empty()) {
      group.dependencies.clear();
      semanticParent = true;
      break;
    }
  bad.planId.clear();
  if (!require(semanticParent && !checkGenericMapping(facts, bad, error),
               "missing semantic group dependency rejected independently"))
    return 1;
  // Independent common-graph mutations exercise the source access/version
  // checker, rather than trusting native planner recomputation alone.
  const auto &graph = both.communication;
  size_t reader = SIZE_MAX, exporter = SIZE_MAX;
  for (size_t e = 0; e < graph.events.size(); ++e) {
    const auto &event = graph.events[e];
    for (auto copy : event.reads)
      if (graph.regions[graph.versions[graph.copies[copy].version].region]
              .storage == x) {
        if (event.kind == "compute" && event.task == 2)
          reader = e;
        if (event.kind == "transfer" &&
            event.link == both.memoryBinding.outputLink)
          exporter = e;
      }
  }
  if (!require(reader != SIZE_MAX && exporter != SIZE_MAX,
               "independent audit locates internal reader and boundary export"))
    return 1;
  auto rejectGraph = [&](const ProgramGraph &g, const char *name) {
    error.clear();
    return require(!checkProgramGraph(g, error), name);
  };
  auto altered = graph;
  auto readCopy = altered.events[reader].reads.front();
  auto creator = altered.copies[readCopy].createdBy;
  altered.events[reader].dependencies.erase(
      std::remove(altered.events[reader].dependencies.begin(),
                  altered.events[reader].dependencies.end(), size_t(creator)),
      altered.events[reader].dependencies.end());
  if (!rejectGraph(altered,
                   "internal consumer cannot lose retained-copy readiness"))
    return 1;
  altered = graph;
  auto exportCopy = altered.events[exporter].reads.front();
  creator = altered.copies[exportCopy].createdBy;
  altered.events[exporter].dependencies.erase(
      std::remove(altered.events[exporter].dependencies.begin(),
                  altered.events[exporter].dependencies.end(), size_t(creator)),
      altered.events[exporter].dependencies.end());
  if (!rejectGraph(altered,
                   "outside export cannot lose retained-copy readiness"))
    return 1;
  altered = graph;
  altered.copies[exportCopy].releasedAfter = exporter;
  if (!rejectGraph(
          altered,
          "retained copy cannot expire before boundary export consumes it"))
    return 1;
  altered = graph;
  auto &users = altered.copies[exportCopy].users;
  users.erase(std::remove(users.begin(), users.end(), exporter), users.end());
  if (!rejectGraph(altered, "boundary export must be recorded as a copy user"))
    return 1;
  altered = graph;
  altered.events[exporter].writes.clear();
  if (!rejectGraph(altered,
                   "boundary transfer must materialize its host replica"))
    return 1;
  altered = graph;
  bool wrongVersion = false;
  for (size_t v = 0; v < graph.versions.size(); ++v)
    if (v != graph.copies[readCopy].version) {
      altered.copies[readCopy].version = v;
      wrongVersion = true;
      break;
    }
  if (!require(wrongVersion, "fixture has distinct source versions") ||
      !rejectGraph(altered,
                   "consumer cannot substitute another logical region version"))
    return 1;
  altered = graph;
  for (auto &memory : altered.memories)
    if (memory.id == both.memoryBinding.localMemory) {
      memory.capacityKnown = true;
      memory.capacityBytes = 1;
    }
  if (!rejectGraph(altered,
                   "grouped tail program obeys actual target capacity"))
    return 1;
  altered = graph;
  altered.allocations[graph.copies[readCopy].allocation].spatialOwner =
      "undeclared-owner";
  if (!rejectGraph(altered,
                   "retained buffer requires an existing target owner"))
    return 1;
  ProgramMappingChoice outputScope;
  for (auto &choice : discoverGenericMappingChoices(facts))
    if (choice.residencyGroups.size() == 1 &&
        choice.residencyGroups.front().tasks == std::vector<size_t>{3, 4})
      outputScope = choice;
  GenericMappingPlan outputPlan;
  if (!require(!outputScope.residencyGroups.empty() &&
                   build(outputScope, outputPlan),
               "output-role retention discovered and checked"))
    return 1;
  size_t z = outputScope.residencyGroups.front().retainedStorages.front();
  if (!require(facts.storages[z].role == "output" && host(outputPlan, z) &&
                   outputPlan.residencyScopes.front().boundaryExports ==
                       std::vector<size_t>{z},
               "output-role retention keeps explicit final host copy"))
    return 1;
  if (!require(
          baseline.communication.transferBytes -
                  outputPlan.communication.transferBytes ==
              bytes,
          "output-role retention removes reload but preserves final export"))
    return 1;
  const auto &scope = outputPlan.residencyScopes.front();
  size_t firstEvent = outputPlan.phases[scope.beginPhase].computationEvent,
         lateInput = SIZE_MAX;
  for (auto &transfer : outputPlan.transfers)
    if (transfer.direction == "host_to_l1" &&
        transfer.phase == scope.beginPhase + 1) {
      lateInput = transfer.event;
      break;
    }
  if (!require(
          lateInput != SIZE_MAX && lateInput < firstEvent &&
              llvm::is_contained(
                  outputPlan.communication.events[firstEvent].dependencies,
                  lateInput),
          "backend-declared group entry waits for later-task external input"))
    return 1;
  if (!require(scope.inputReleasePolicy == "last-reader",
               "late-input realization declares per-input last-reader release"))
    return 1;
  size_t tailEvent = outputPlan.phases[scope.beginPhase + 1].computationEvent,
         nextEarly = SIZE_MAX, nextLate = SIZE_MAX;
  for (auto &transfer : outputPlan.transfers)
    if (transfer.direction == "host_to_l1") {
      if (transfer.phase == scope.beginPhase + scope.tasks.size())
        nextEarly = transfer.event;
      if (transfer.phase == scope.beginPhase + scope.tasks.size() + 1)
        nextLate = transfer.event;
    }
  if (!require(nextEarly != SIZE_MAX && nextLate != SIZE_MAX &&
                   llvm::is_contained(
                       outputPlan.communication.events[nextEarly].dependencies,
                       firstEvent) &&
                   !llvm::is_contained(
                       outputPlan.communication.events[nextEarly].dependencies,
                       tailEvent),
               "early input refills after its actual last reader rather than "
               "unrelated tail"))
    return 1;
  if (!require(llvm::is_contained(
                   outputPlan.communication.events[nextLate].dependencies,
                   tailEvent),
               "later input still waits for its own last reader"))
    return 1;
  auto fakeRelease = outputPlan;
  fakeRelease.planId.clear();
  fakeRelease.residencyScopes.front().inputReleasePolicy = "group-tail";
  if (!require(!checkGenericMapping(facts, fakeRelease, error),
               "fabricated input release policy rejected"))
    return 1;
  fakeRelease = outputPlan;
  fakeRelease.residencyScopes.front().inputReleasePolicy = "unrecognized";
  if (!require(!checkProgramCommunication(facts, fakeRelease, error),
               "standalone communication rejects unknown release policy"))
    return 1;
  fakeRelease = outputPlan;
  fakeRelease.residencyScopes.front().tasks.clear();
  if (!require(!checkProgramCommunication(facts, fakeRelease, error),
               "standalone communication rejects empty resolved block width"))
    return 1;
  auto missingEntry = outputPlan;
  missingEntry.planId.clear();
  auto &entryDeps = missingEntry.communication.events[firstEvent].dependencies;
  entryDeps.erase(std::remove(entryDeps.begin(), entryDeps.end(), lateInput),
                  entryDeps.end());
  if (!require(!checkGenericMapping(facts, missingEntry, error),
               "missing declared input-acquisition constraint rejected"))
    return 1;
  llvm::outs() << checks << " subgraph grouping checks passed\n";
  return 0;
}
