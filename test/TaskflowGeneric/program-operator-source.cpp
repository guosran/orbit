#include "Taskflow/JSONUtils.h"
#include "Taskflow/ProgramOperatorMapping.h"
#include "Taskflow/ProgramOperatorSource.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <limits>
using namespace orbit::taskflow;
namespace j = llvm::json;
int main() {
  auto parsed = j::parse(
      R"JSON({"schema":"orbit.program-operator-source.v1","source_id":"0000000000000000000000000000000000000000000000000000000000000000","numerical_contract":"structural-fixture-not-arithmetic-proof","bindings":[{"id":"fixture","file":"fixture.cpp","sha256":"0000000000000000000000000000000000000000000000000000000000000000","symbol":"fixture"}],"values":[{"id":"x","dtype":"f32","shape":[4],"role":"input","layout":"row_major"},{"id":"raw","dtype":"f32","shape":[8],"role":"workspace","layout":"row_major","initialization":"uninitialized"},{"id":"a","dtype":"f32","shape":[4],"role":"temporary","layout":"row_major","allocation_view":{"base":"raw","offsets":[0],"strides":[1]}},{"id":"b","dtype":"f32","shape":[4],"role":"temporary","layout":"row_major","allocation_view":{"base":"raw","offsets":[4],"strides":[1]}},{"id":"sum","dtype":"f32","shape":[4],"role":"output","layout":"row_major"}],"operators":[{"id":"write-a","kind":"copy","binding":"fixture","source_ordinal":0,"inputs":["x"],"outputs":["a"],"dependencies":[],"attributes":{}},{"id":"write-b","kind":"copy","binding":"fixture","source_ordinal":1,"inputs":["x"],"outputs":["b"],"dependencies":[],"attributes":{}},{"id":"sum","kind":"add","binding":"fixture","source_ordinal":2,"inputs":["a","b"],"outputs":["sum"],"dependencies":[],"attributes":{}}]})JSON");
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return 1;
  }
  std::string error;
  unsigned checks = 0;
  auto require = [&](bool value, const char *message) {
    ++checks;
    if (!value)
      llvm::errs() << message << ": " << error << "\n";
    return value;
  };
  std::shared_ptr<ProgramOperatorSource> source;
  if (!require(parseProgramOperatorSource(*parsed, source, error),
               "workspace source parses"))
    return 1;
  StaticConfig target;
  StaticResourceDomain system;
  system.id = "system";
  system.kind = "system";
  StaticResourceDomain device;
  device.id = "device";
  device.kind = "device";
  device.parent = "system";
  target.domains = {system, device};
  StaticResource cpu;
  cpu.id = "cpu";
  cpu.kind = "compute";
  cpu.device = "cpu";
  cpu.capacity = 1;
  cpu.domain = "system";
  StaticResource npu = cpu;
  npu.id = "npu";
  npu.device = "npu";
  npu.domain = "device";
  target.resources = {cpu, npu};
  StaticMemory memory;
  memory.id = "abi";
  memory.kind = "system-bo";
  memory.domain = "system";
  memory.capacityBytes = INT64_MAX;
  memory.capacityKnown = false;
  memory.accessibleResources = {"cpu", "npu"};
  target.memories = {memory};
  ProgramOperatorSelection selection;
  for (size_t o = 0; o < 3; ++o) {
    ProgramOperatorImplementation impl;
    impl.id = "impl" + std::to_string(o);
    impl.sourceId = source->sourceId;
    impl.artifactIdentity = "fixture-artifact";
    impl.entrypoint = impl.id;
    impl.executable = true;
    impl.device = o ? "cpu" : "npu";
    impl.resource = o ? "cpu" : "npu";
    impl.memory = "abi";
    impl.operators = {o};
    impl.inputs = source->operators[o].inputs;
    impl.outputs = source->operators[o].outputs;
    impl.wallMs = .001;
    impl.launchCount = o ? 0 : 1;
    impl.visibilityPolicy = o ? "host-materialized" : "device-resident";
    selection.regions.push_back(impl);
  }
  ProgramGraph graph;
  if (!require(materializeProgramOperatorGraph(source, selection, target, "abi",
                                               graph, error),
               "actual cover materializes shared workspace"))
    return 1;
  if (!require(checkProgramGraph(graph, error),
               "public checker accepts first-class external source"))
    return 1;
  if (!require(graph.allocations.size() == 3 && graph.copies.size() == 4 &&
                   std::none_of(graph.regions.begin(), graph.regions.end(),
                                [](const auto &r) { return r.storage == 1; }),
               "uninitialized workspace has allocation and no fabricated "
               "initial copy"))
    return 1;
  if (!require(graph.regionEvents[0].kind == "submit" &&
                   graph.regionEvents[1].kind == "complete",
               "resident implementation has no fake host synchronization"))
    return 1;
  auto broken = graph;
  broken.dataEdges[0].version = broken.versions.size();
  if (!require(!checkProgramGraph(broken, error), "wrong ABI version rejected"))
    return 1;
  broken = graph;
  broken.regionEvents.push_back({"visible-to-host", 0, "npu", {}, {}});
  if (!require(!checkProgramGraph(broken, error),
               "fabricated host readback rejected"))
    return 1;
  GenericAnalysis native;
  if (!require(!checkProgramGraphSource(native, graph, error),
               "native source consumer explicitly rejects external source"))
    return 1;
  auto oldShape = source->values[0].shape;
  source->values[0].shape = {5};
  if (!require(!checkProgramOperatorSource(*source, error),
               "typed mutation cannot retain registered document identity"))
    return 1;
  source->values[0].shape = oldShape;
  auto unknown = selection;
  unknown.regions[0].wallMs.reset();
  ProgramGraph unpriced;
  if (!require(materializeProgramOperatorGraph(source, unknown, target, "abi",
                                               unpriced, error) &&
                   summarizeProgramOperatorGraph(unpriced)
                       .getAsObject()
                       ->get("estimated_ms")
                       ->getAsNull(),
               "missing local price remains unknown"))
    return 1;
  auto badPrice = selection;
  badPrice.regions[0].wallMs = std::numeric_limits<double>::quiet_NaN();
  if (!require(!materializeProgramOperatorGraph(source, badPrice, target, "abi",
                                                broken, error),
               "nonfinite implementation price rejected"))
    return 1;
  badPrice = selection;
  badPrice.regions[0].wallMs = -1;
  if (!require(!materializeProgramOperatorGraph(source, badPrice, target, "abi",
                                                broken, error),
               "negative implementation price rejected"))
    return 1;
  auto overlap = copyJSONWithoutFields(*parsed);
  auto *values = overlap.getAsObject()->getArray("values");
  *(*values)[3].getAsObject()->getObject("allocation_view")->get("offsets") =
      j::Array{2};
  std::shared_ptr<ProgramOperatorSource> invalid;
  if (!require(!parseProgramOperatorSource(overlap, invalid, error),
               "overwritten workspace slice reader rejected"))
    return 1;
  auto overflow = copyJSONWithoutFields(*parsed);
  *(*overflow.getAsObject()->getArray("values"))[2]
       .getAsObject()
       ->getObject("allocation_view")
       ->get("offsets") = j::Array{INT64_MAX};
  if (!require(!parseProgramOperatorSource(overflow, invalid, error),
               "view address overflow rejected before integer addition"))
    return 1;
  ProgramOperatorMapping mapping;
  mapping.registrationIdentity = "fixture-backend-registration";
  for (size_t i = 0; i < selection.regions.size(); ++i) {
    const auto &impl = selection.regions[i];
    ProgramOperatorExecutionVariant v;
    v.runtimeContextIdentity = "fixture-context";
    v.dataflow = "fixture-synchronous";
    v.residencePolicy = "fixture-ABI-lifetime";
    v.implementationId = impl.id;
    v.artifactIdentity = impl.artifactIdentity;
    v.entrypoint = impl.entrypoint;
    v.resourceDemands = {{impl.resource, 1}};
    mapping.variants.push_back(v);
    mapping.readyPriority.push_back(i);
  }
  auto timed = graph;
  if (!require(scheduleProgramOperatorGraph(timed, mapping, error) &&
                   timed.estimatedMakespanMs &&
                   std::abs(*timed.estimatedMakespanMs - .003) < 1e-12 &&
                   checkProgramGraph(timed, error),
               "inclusive invocation schedule preserves real serial runtime "
               "dependencies"))
    return 1;
  broken = timed;
  broken.schedule.events[1].startMs = 0;
  if (!require(!checkProgramGraph(broken, error),
               "forged temporal overlap rejected"))
    return 1;
  auto invalidMapping = mapping;
  invalidMapping.variants[0].resourceDemands["npu"] = 2;
  broken = graph;
  if (!require(!scheduleProgramOperatorGraph(broken, invalidMapping, error),
               "spatial/resource demand above actual capacity rejected"))
    return 1;
  invalidMapping = mapping;
  invalidMapping.variants[0].artifactIdentity = "different-kernel";
  broken = graph;
  if (!require(
          !scheduleProgramOperatorGraph(broken, invalidMapping, error),
          "variant cannot change artifact without registered implementation"))
    return 1;
  invalidMapping = mapping;
  invalidMapping.variants[0].replicas = 2;
  broken = graph;
  if (!require(!scheduleProgramOperatorGraph(broken, invalidMapping, error),
               "replicas require registered disjoint work distribution"))
    return 1;
  invalidMapping = mapping;
  invalidMapping.readyPriority = {0, 0, 2};
  broken = graph;
  if (!require(!scheduleProgramOperatorGraph(broken, invalidMapping, error),
               "duplicate ready priority rejected"))
    return 1;
  broken = graph;
  broken.memories[0].capacityKnown = true;
  broken.memories[0].capacityBytes = 64;
  invalidMapping = mapping;
  invalidMapping.variants[0].residentAllocations["retained-weights"] = {"abi",
                                                                        64};
  if (!require(!scheduleProgramOperatorGraph(broken, invalidMapping, error),
               "ABI plus retained weights must fit physical memory together"))
    return 1;
  broken = graph;
  invalidMapping = mapping;
  invalidMapping.variants[0].temporaryBytes["nonexistent-memory"] = 1;
  if (!require(!scheduleProgramOperatorGraph(broken, invalidMapping, error),
               "unknown scratch location rejected"))
    return 1;
  ProgramOperatorMappingCandidate candidate{selection, mapping}, child;
  ProgramOperatorMappingAction action;
  action.kind = ProgramOperatorMappingActionKind::ReadyPriority;
  action.readyPriority = {2, 1, 0};
  if (!require(
          applyProgramOperatorMappingAction(candidate, action, child, error),
          "ready priority action represents a checked candidate decision"))
    return 1;
  ProgramGraph evaluated;
  auto evaluation = evaluateProgramOperatorMappingCandidate(
      source, child, target, "abi", evaluated);
  if (!require(evaluation.feasible && !evaluation.feasibilityKnown &&
                   evaluation.costMs &&
                   std::abs(*evaluation.costMs - .003) < 1e-12,
               "priority cannot bypass dependencies and unknown physical "
               "capacity stays unproved"))
    return 1;
  action.kind = ProgramOperatorMappingActionKind::ReplaceExecutableCover;
  action.firstRegion = 0;
  action.regionCount = 1;
  action.replacements = {selection.regions[1]};
  action.replacementVariants = {mapping.variants[1]};
  if (!require(
          !applyProgramOperatorMappingAction(candidate, action, child, error),
          "structural action cannot replace different source coverage"))
    return 1;
  action.kind = ProgramOperatorMappingActionKind::IndependentResidence;
  if (!require(
          !applyProgramOperatorMappingAction(candidate, action, child, error),
          "unsupported residence change requires actual backend variant"))
    return 1;
  std::vector<ProgramOperatorImplementation> implementations =
      selection.regions;
  auto alt0 = implementations[0], alt1 = implementations[1];
  alt0.id = "retained0";
  alt1.id = "retained1";
  alt0.wallMs = .0005;
  alt1.wallMs = .0005;
  implementations.push_back(alt0);
  implementations.push_back(alt1);
  std::map<std::string, ProgramOperatorExecutionVariant> registrations;
  for (size_t i = 0; i < implementations.size(); ++i) {
    auto variant = mapping.variants[i == 3 ? 0 : (i == 4 ? 1 : i)];
    variant.implementationId = implementations[i].id;
    if (i >= 3)
      variant.residentAllocations["retained-weight-" + std::to_string(i)] = {
          "abi", 16};
    registrations.emplace(variant.implementationId, variant);
  }
  std::vector<ProgramOperatorChoiceGroup> groups(3);
  for (size_t i = 0; i < 3; ++i) {
    groups[i].id = "group" + std::to_string(i);
    groups[i].choiceIds = {"base"};
    groups[i].choices = {{i}};
  }
  groups[0].choiceIds.push_back("retain");
  groups[0].choices.push_back({3});
  groups[1].choiceIds.push_back("retain");
  groups[1].choices.push_back({4});
  auto bounded = target;
  bounded.memories[0].capacityKnown = true;
  bounded.memories[0].capacityBytes = 88;
  std::shared_ptr<const ProgramOperatorMappingDomain> domain;
  if (!require(prepareProgramOperatorMappingDomain(
                   *source, implementations, groups, registrations,
                   "backend-registration", bounded, "abi", domain, error),
               "immutable domain certifies actual source coverage and boundary "
               "allocations"))
    return 1;
  auto projected = evaluateProgramOperatorMappingDomain(*domain, {0, 0, 0});
  if (!require(projected.feasible && projected.feasibilityKnown &&
                   projected.costMs &&
                   std::abs(*projected.costMs - *timed.estimatedMakespanMs) <
                       1e-12,
               "cached projection agrees with full physical graph schedule"))
    return 1;
  if (!require(
          evaluateProgramOperatorMappingDomain(*domain, {1, 0, 0}).feasible &&
              evaluateProgramOperatorMappingDomain(*domain, {0, 1, 0})
                  .feasible &&
              !evaluateProgramOperatorMappingDomain(*domain, {1, 1, 0})
                   .feasible,
          "joint retained capacity rejects combination although each local "
          "choice fits alone"))
    return 1;
  implementations[0].wallMs = 100;
  registrations["impl0"].resourceDemands["npu"] = 999;
  auto unchanged = evaluateProgramOperatorMappingDomain(*domain, {0, 0, 0});
  if (!require(unchanged.feasible && unchanged.costMs == projected.costMs,
               "prepared domain owns immutable fee and resource registration "
               "snapshots"))
    return 1;
  if (!require(
          !evaluateProgramOperatorMappingDomain(*domain, {2, 0, 0}).feasible,
          "out of domain choices rejected before schedule evaluation"))
    return 1;
  auto fused = selection.regions[0];
  fused.id = "fixture-packed-fused";
  fused.operators = {0, 1, 2};
  fused.inputs = {0};
  fused.outputs = {4};
  fused.wallMs = .003;
  ProgramOperatorSelection packedSelection;
  packedSelection.regions = {fused};
  ProgramOperatorMapping packedMapping;
  packedMapping.registrationIdentity = "fixture-bitchecked-registration";
  packedMapping.hostMemory = "abi";
  packedMapping.readyPriority = {0};
  auto packedVariant = mapping.variants[0];
  packedVariant.implementationId = fused.id;
  ProgramOperatorPackedInput packing;
  packing.sourceValue = 0;
  packing.packingIdentity = "fixture-bitwise-packing";
  for (int64_t part = 0; part < 2; ++part) {
    ProgramOperatorPackedFragment fragment;
    fragment.allocationId = "packed-weight-part" + std::to_string(part);
    fragment.memory = "abi";
    fragment.dtype = "f32";
    fragment.bytes = 12;
    fragment.sourceOffsets = {part * 2};
    fragment.sourceExtents = {2};
    fragment.physicalShape = {3};
    fragment.physicalOffsets = {0};
    fragment.strides = {1};
    packing.fragments.push_back(fragment);
    packedVariant.residentAllocations[fragment.allocationId] = {"abi", 12};
  }
  packedVariant.packedInputs = {packing};
  packedMapping.variants = {packedVariant};
  auto packedTarget = bounded;
  packedTarget.memories[0].capacityBytes = 40;
  ProgramGraph packedGraph;
  if (!require(
          materializeProgramOperatorGraph(source, packedSelection, packedTarget,
                                          "abi", packedGraph, error) &&
              scheduleProgramOperatorGraph(packedGraph, packedMapping, error) &&
              checkProgramGraph(packedGraph, error),
          "source input is physically replaced by two real padded BO "
          "fragments"))
    return 1;
  int64_t actualBytes = 0;
  for (const auto &allocation : packedGraph.allocations)
    actualBytes += allocation.bytes;
  if (!require(packedGraph.allocations.size() == 3 && actualBytes == 40 &&
                   packedGraph.events[0].reads.size() == 2 &&
                   packedGraph.dataEdges.size() == 2,
               "packed input neither retains obsolete source BO nor double "
               "counts shared resident BOs"))
    return 1;
  std::shared_ptr<const ProgramOperatorMappingDomain> packedDomain;
  std::vector<ProgramOperatorChoiceGroup> packedGroups(1);
  packedGroups[0].id = "whole";
  packedGroups[0].choiceIds = {"packed"};
  packedGroups[0].choices = {{0}};
  std::map<std::string, ProgramOperatorExecutionVariant> packedRegistration{
      {fused.id, packedVariant}};
  if (!require(
          prepareProgramOperatorMappingDomain(
              *source, {fused}, packedGroups, packedRegistration,
              "fixture-bitchecked-registration", packedTarget, "abi",
              packedDomain, error) &&
              evaluateProgramOperatorMappingDomain(*packedDomain, {0}).feasible,
          "packed projection and complete physical graph share exact "
          "allocation accounting"))
    return 1;
  auto badPacking = packedMapping;
  badPacking.variants[0].packedInputs[0].fragments[1].sourceOffsets = {1};
  ProgramGraph invalidPacked;
  if (!require(
          materializeProgramOperatorGraph(source, packedSelection, packedTarget,
                                          "abi", invalidPacked, error) &&
              !scheduleProgramOperatorGraph(invalidPacked, badPacking, error),
          "overlapping source packing fragments rejected"))
    return 1;
  badPacking = packedMapping;
  badPacking.variants[0].packedInputs[0].fragments[1].sourceExtents = {1};
  if (!require(
          materializeProgramOperatorGraph(source, packedSelection, packedTarget,
                                          "abi", invalidPacked, error) &&
              !scheduleProgramOperatorGraph(invalidPacked, badPacking, error),
          "incomplete source packing coverage rejected"))
    return 1;
  badPacking = packedMapping;
  badPacking.variants[0].packedInputs[0].fragments[0].strides = {2};
  if (!require(
          materializeProgramOperatorGraph(source, packedSelection, packedTarget,
                                          "abi", invalidPacked, error) &&
              !scheduleProgramOperatorGraph(invalidPacked, badPacking, error),
          "physical packing stride mismatch rejected"))
    return 1;
  broken = packedGraph;
  broken.allocations.push_back(graph.allocations[0]);
  if (!require(!checkProgramGraph(broken, error),
               "obsolete source allocation cannot be reintroduced into packed "
               "graph"))
    return 1;
  broken = packedGraph;
  broken.regions[0].offsets = {1};
  if (!require(!checkProgramGraph(broken, error),
               "forged packed logical source interval rejected"))
    return 1;
  badPacking = packedMapping;
  badPacking.variants[0].residentAllocations["packed-weight-part0"].second = 13;
  if (!require(
          materializeProgramOperatorGraph(source, packedSelection, packedTarget,
                                          "abi", invalidPacked, error) &&
              !scheduleProgramOperatorGraph(invalidPacked, badPacking, error),
          "shared resident alias must have identical physical bytes"))
    return 1;
  auto replicatedMapping = packedMapping;
  replicatedMapping.variants[0].replicas = 2;
  replicatedMapping.variants[0].workDistribution = "registered-disjoint-work";
  replicatedMapping.variants[0].workPartitions = {{4, 0, {0}, {2}},
                                                  {4, 1, {2}, {2}}};
  if (!require(materializeProgramOperatorGraph(source, packedSelection,
                                               packedTarget, "abi",
                                               invalidPacked, error) &&
                   scheduleProgramOperatorGraph(invalidPacked,
                                                replicatedMapping, error),
               "registered two-replica output shards form an exact disjoint "
               "source-work cover"))
    return 1;
  auto overlappingWork = replicatedMapping;
  overlappingWork.variants[0].workPartitions[1].offsets = {1};
  if (!require(materializeProgramOperatorGraph(source, packedSelection,
                                               packedTarget, "abi",
                                               invalidPacked, error) &&
                   !scheduleProgramOperatorGraph(invalidPacked, overlappingWork,
                                                 error),
               "replica output work cannot overlap"))
    return 1;
  overlappingWork = replicatedMapping;
  overlappingWork.variants[0].workPartitions[1].extents = {1};
  if (!require(materializeProgramOperatorGraph(source, packedSelection,
                                               packedTarget, "abi",
                                               invalidPacked, error) &&
                   !scheduleProgramOperatorGraph(invalidPacked, overlappingWork,
                                                 error),
               "replica output work cannot leave coverage holes"))
    return 1;
  auto unprovedMapping = packedMapping;
  unprovedMapping.variants[0].unprovedConstraints = {
      "private decoder template bytes unavailable"};
  ProgramOperatorMappingCandidate unprovedCandidate{packedSelection,
                                                    unprovedMapping};
  auto unprovedEvaluation = evaluateProgramOperatorMappingCandidate(
      source, unprovedCandidate, packedTarget, "abi", invalidPacked);
  if (!require(unprovedEvaluation.feasible &&
                   !unprovedEvaluation.feasibilityKnown,
               "known SRAM capacities cannot hide unproved private-storage "
               "reservations"))
    return 1;
  broken = graph;
  broken.instances[0].resourceCount = 8;
  if (!require(!checkProgramGraph(broken, error),
               "logical source instance cannot invent physical resource "
               "reservations outside its invocation variant"))
    return 1;
  for (size_t a = 0; a < 2; ++a)
    for (size_t b = 0; b < 2; ++b) {
      auto fast = evaluateProgramOperatorMappingDomain(*domain, {a, b, 0});
      auto reference =
          evaluateProgramOperatorMappingDomainReference(*domain, {a, b, 0});
      if (!require(fast.feasible == reference.feasible &&
                       fast.feasibilityKnown == reference.feasibilityKnown &&
                       fast.costMs.has_value() ==
                           reference.costMs.has_value() &&
                       (!fast.costMs ||
                        std::abs(*fast.costMs - *reference.costMs) < 1e-12),
                   "choice summary equals invocation calendar including joint "
                   "capacity failures"))
        return 1;
    }
  auto fastPacked = evaluateProgramOperatorMappingDomain(*packedDomain, {0});
  auto referencePacked =
      evaluateProgramOperatorMappingDomainReference(*packedDomain, {0});
  if (!require(
          fastPacked.feasible == referencePacked.feasible &&
              fastPacked.feasibilityKnown == referencePacked.feasibilityKnown &&
              fastPacked.costMs == referencePacked.costMs,
          "choice summary preserves packed source/resident alias accounting"))
    return 1;
  auto aliasImplementations = selection.regions;
  aliasImplementations.push_back(alt0);
  aliasImplementations.push_back(alt1);
  auto aliasRegistrations = registrations;
  aliasRegistrations["impl0"] = mapping.variants[0];
  aliasRegistrations["retained0"].residentAllocations = {
      {"shared-cache", {"abi", 16}}};
  aliasRegistrations["retained1"].residentAllocations = {
      {"shared-cache", {"abi", 17}}};
  std::shared_ptr<const ProgramOperatorMappingDomain> aliasDomain;
  if (!require(prepareProgramOperatorMappingDomain(
                   *source, aliasImplementations, groups, aliasRegistrations,
                   "alias-conflict-fixture", bounded, "abi", aliasDomain,
                   error),
               "immutable summaries retain mutually exclusive physical "
               "identity payloads"))
    return 1;
  auto fastAlias =
      evaluateProgramOperatorMappingDomain(*aliasDomain, {1, 1, 0});
  auto referenceAlias =
      evaluateProgramOperatorMappingDomainReference(*aliasDomain, {1, 1, 0});
  if (!require(!fastAlias.feasible && !referenceAlias.feasible &&
                   fastAlias.feasibilityKnown ==
                       referenceAlias.feasibilityKnown,
               "joint identity size conflict cannot be hidden by local summary "
               "proofs"))
    return 1;
  // These are source/ABI scheduling fixtures. They do not claim any backend
  // supports asynchronous execution until its independent registration gate.
  auto readyDocument = copyJSONWithoutFields(*parsed);
  auto &readyValues = *readyDocument.getAsObject()->getArray("values");
  readyValues[2].getAsObject()->erase("allocation_view");
  readyValues[3].getAsObject()->erase("allocation_view");
  std::shared_ptr<ProgramOperatorSource> readySource;
  if (!require(parseProgramOperatorSource(readyDocument, readySource, error),
               "independent storage fixture has explicit source semantics"))
    return 1;
  auto readySelection = selection;
  auto readyMapping = mapping;
  for (auto &v : readyMapping.variants) {
    v.executionPolicy = "source-dependency-ready";
    v.sourceOrderingIdentity = readySource->sourceId;
    v.asyncRuntimeProofIdentity = "fixture-only-async-contract";
  }
  readySelection.regions[0].wallMs = 10;
  readySelection.regions[1].wallMs = 2;
  readySelection.regions[2].wallMs = 1;
  auto readyTarget = target;
  readyTarget.memories[0].capacityKnown = true;
  readyTarget.memories[0].capacityBytes = 4096;
  ProgramOperatorMappingCandidate readyCandidate{readySelection, readyMapping};
  ProgramGraph readyGraph;
  auto readyResult = evaluateProgramOperatorMappingCandidate(
      readySource, readyCandidate, readyTarget, "abi", readyGraph);
  if (!require(readyResult.feasible && readyResult.feasibilityKnown &&
                   readyResult.costMs && *readyResult.costMs == 11 &&
                   readyGraph.events[1].dependencies.empty() &&
                   readyGraph.events[2].dependencies ==
                       std::vector<size_t>({0, 1}) &&
                   readyGraph.regionEvents[2].dependencies.empty(),
               "independent CPU/NPU calls overlap only under explicit ready "
               "contract"))
    return 1;
  broken = readyGraph;
  broken.events[2].dependencies.clear();
  if (!require(!checkProgramGraph(broken, error),
               "source data dependencies cannot be deleted to create overlap"))
    return 1;
  auto missingProof = readyCandidate;
  missingProof.mapping.variants[1].asyncRuntimeProofIdentity.clear();
  if (!require(!evaluateProgramOperatorMappingCandidate(
                    readySource, missingProof, readyTarget, "abi", broken)
                    .feasible,
               "mixed or absent async support proof is rejected"))
    return 1;
  auto staleProof = readyCandidate;
  staleProof.mapping.variants[0].sourceOrderingIdentity = "another-source";
  if (!require(!evaluateProgramOperatorMappingCandidate(
                    readySource, staleProof, readyTarget, "abi", broken)
                    .feasible,
               "async registration cannot authorize a different source"))
    return 1;
  auto hazardResult = evaluateProgramOperatorMappingCandidate(
      source, readyCandidate, readyTarget, "abi", broken);
  if (!require(
          hazardResult.feasible && hazardResult.costMs &&
              *hazardResult.costMs == 13,
          "writes to shared physical workspace remain conservatively ordered"))
    return 1;
  auto controlledDocument = copyJSONWithoutFields(readyDocument);
  (*controlledDocument.getAsObject()->getArray("operators"))[1].getAsObject()->
  operator[]("dependencies") = j::Array{"write-a"};
  std::shared_ptr<ProgramOperatorSource> controlledSource;
  if (!require(parseProgramOperatorSource(controlledDocument, controlledSource,
                                          error),
               "source control fixture parses"))
    return 1;
  auto controlled = evaluateProgramOperatorMappingCandidate(
      controlledSource, readyCandidate, readyTarget, "abi", broken);
  if (!require(controlled.feasible && controlled.costMs &&
                   *controlled.costMs == 13,
               "original source control edge survives ready lowering"))
    return 1;
  auto scratchCandidate = readyCandidate;
  scratchCandidate.mapping.variants[0].temporaryBytes = {{"abi", 16}};
  scratchCandidate.mapping.variants[1].temporaryBytes = {{"abi", 16}};
  auto scratchTarget = readyTarget;
  scratchTarget.memories[0].capacityBytes = 80; // 64 persistent + one scratch.
  auto scratchResult = evaluateProgramOperatorMappingCandidate(
      readySource, scratchCandidate, scratchTarget, "abi", broken);
  if (!require(scratchResult.feasible && scratchResult.costMs &&
                   *scratchResult.costMs == 13,
               "simultaneous temporary storage is capacity constrained"))
    return 1;
  // A(2ms CPU) is independent; B(1ms CPU) feeds C(10ms NPU).
  // Ready priority changes the critical path without changing a source edge.
  auto priorityDocument = copyJSONWithoutFields(readyDocument);
  (*priorityDocument.getAsObject()->getArray("values"))[2].getAsObject()->
  operator[]("role") = "output";
  (*priorityDocument.getAsObject()->getArray("operators"))[2].getAsObject()->
  operator[]("inputs") = j::Array{"b"};
  std::shared_ptr<ProgramOperatorSource> prioritySource;
  if (!require(
          parseProgramOperatorSource(priorityDocument, prioritySource, error),
          "branch critical-path fixture parses"))
    return 1;
  auto priorityCandidate = readyCandidate;
  for (size_t i = 0; i < 3; ++i) {
    auto &impl = priorityCandidate.selection.regions[i];
    auto &variant = priorityCandidate.mapping.variants[i];
    impl.device = i == 2 ? "npu" : "cpu";
    impl.resource = impl.device;
    impl.launchCount = i == 2 ? 1 : 0;
    impl.wallMs = i == 0 ? 2 : i == 1 ? 1 : 10;
    impl.inputs = prioritySource->operators[i].inputs;
    variant.resourceDemands = {{impl.resource, 1}};
  }
  auto firstPriority = evaluateProgramOperatorMappingCandidate(
      prioritySource, priorityCandidate, readyTarget, "abi", broken);
  priorityCandidate.selection.executionPolicy = "source-dependency-ready";
  auto priorityNeighbors =
      programOperatorReadyPriorityNeighbors(priorityCandidate);
  if (!require(priorityNeighbors.size() == 2 &&
                   applyProgramOperatorMappingAction(
                       priorityCandidate, priorityNeighbors[0], child, error),
               "ready neighborhood contains adjacent priority transpositions"))
    return 1;
  auto changedPriority = evaluateProgramOperatorMappingCandidate(
      prioritySource, child, readyTarget, "abi", readyGraph);
  if (!require(firstPriority.costMs && *firstPriority.costMs == 13 &&
                   changedPriority.feasible && changedPriority.costMs &&
                   *changedPriority.costMs == 11,
               "actual ready order changes resource calendar makespan"))
    return 1;
  std::vector<ProgramOperatorChoiceGroup> priorityGroups;
  std::map<std::string, ProgramOperatorExecutionVariant> priorityVariants;
  for (size_t i = 0; i < 3; ++i) {
    priorityGroups.push_back({"priority" + std::to_string(i), {"only"}, {{i}}});
    priorityVariants.emplace(priorityCandidate.selection.regions[i].id,
                             priorityCandidate.mapping.variants[i]);
  }
  std::shared_ptr<const ProgramOperatorMappingDomain> priorityDomain;
  if (!require(prepareProgramOperatorMappingDomain(
                   *prioritySource, priorityCandidate.selection.regions,
                   priorityGroups, priorityVariants, "fixture-ready-domain",
                   readyTarget, "abi", priorityDomain, error),
               "immutable source frontier certifies ready domain"))
    return 1;
  auto projection = evaluateProgramOperatorMappingDomain(
      *priorityDomain, {0, 0, 0}, "reverse-source-order");
  if (!require(projection.feasible && projection.costMs &&
                   *projection.costMs == 11,
               "cached dependency projection uses exact nonserial calendar"))
    return 1;
  // Noncontiguous {A,C} is legal only when no outside path re-enters it.
  auto convexCandidate = priorityCandidate;
  auto merged = convexCandidate.selection.regions[0];
  merged.id = "fixture-noncontiguous-cover";
  merged.operators = {0, 2};
  merged.inputs = {0, 3};
  merged.outputs = {2, 4};
  merged.wallMs = 12;
  auto mergedVariant = convexCandidate.mapping.variants[0];
  mergedVariant.implementationId = merged.id;
  convexCandidate.selection.regions = {merged,
                                       priorityCandidate.selection.regions[1]};
  convexCandidate.mapping.variants = {mergedVariant,
                                      priorityCandidate.mapping.variants[1]};
  convexCandidate.mapping.readyPriority = {0, 1};
  auto convexResult = evaluateProgramOperatorMappingCandidate(
      prioritySource, convexCandidate, readyTarget, "abi", readyGraph);
  if (!require(convexResult.feasible &&
                   readyGraph.operatorSelection->regions[0].operators ==
                       std::vector<size_t>{1},
               "DAG-convex noncontiguous cover is canonically reordered"))
    return 1;
  auto convexImplementations = priorityCandidate.selection.regions;
  convexImplementations.push_back(merged);
  auto convexVariants = priorityVariants;
  convexVariants.emplace(merged.id, mergedVariant);
  std::vector<ProgramOperatorChoiceGroup> convexGroups{
      {"noncontiguous-component", {"unfused", "fused"}, {{0, 2}, {3}}},
      {"outside-producer", {"only"}, {{1}}}};
  std::shared_ptr<const ProgramOperatorMappingDomain> convexDomain;
  if (!require(prepareProgramOperatorMappingDomain(
                   *prioritySource, convexImplementations, convexGroups,
                   convexVariants, "fixture-convex-domain", readyTarget, "abi",
                   convexDomain, error),
               "noncontiguous component choices certify invariant exact ABI"))
    return 1;
  auto convexProjection =
      evaluateProgramOperatorMappingDomain(*convexDomain, {1, 0});
  auto convexReference =
      evaluateProgramOperatorMappingDomainReference(*convexDomain, {1, 0});
  auto unfusedProjection = evaluateProgramOperatorMappingDomain(
      *convexDomain, {0, 0}, "reverse-source-order");
  if (!require(convexProjection.feasible &&
                   convexProjection.costMs == convexResult.costMs &&
                   convexProjection.costMs == convexReference.costMs &&
                   unfusedProjection.costMs && *unfusedProjection.costMs == 11,
               "choice-contracted source frontier preserves fused/unfused "
               "noncontiguous semantics"))
    return 1;
  auto cycleDocument = copyJSONWithoutFields(priorityDocument);
  (*cycleDocument.getAsObject()->getArray("operators"))[1].getAsObject()->
  operator[]("dependencies") = j::Array{"write-a"};
  std::shared_ptr<ProgramOperatorSource> cycleSource;
  if (!require(parseProgramOperatorSource(cycleDocument, cycleSource, error),
               "nonconvex cycle fixture parses before contraction"))
    return 1;
  auto cycleResult = evaluateProgramOperatorMappingCandidate(
      cycleSource, convexCandidate, readyTarget, "abi", broken);
  if (!require(!cycleResult.feasible &&
                   cycleResult.message.find("cyclic source-cover quotient") !=
                       std::string::npos,
               "fusion contracting an outside dependency cycle is rejected"))
    return 1;
  std::shared_ptr<const ProgramOperatorMappingDomain> nonconvexDomain;
  if (!require(prepareProgramOperatorMappingDomain(
                   *cycleSource, convexImplementations, convexGroups,
                   convexVariants, "fixture-nonconvex-domain", readyTarget,
                   "abi", nonconvexDomain, error),
               "local ABI certificate does not falsely claim global quotient "
               "legality"))
    return 1;
  auto nonconvexProjection =
      evaluateProgramOperatorMappingDomain(*nonconvexDomain, {1, 0});
  auto nonconvexReference =
      evaluateProgramOperatorMappingDomainReference(*nonconvexDomain, {1, 0});
  if (!require(!nonconvexProjection.feasible && !nonconvexReference.feasible &&
                   nonconvexProjection.feasibilityKnown &&
                   nonconvexReference.feasibilityKnown,
               "both calendars reject a quotient cycle introduced by an "
               "otherwise exact cover"))
    return 1;
  auto coordinateImpl = selection.regions[0];
  coordinateImpl.id = "registered-shape-artifact";
  coordinateImpl.artifactIdentity = "fixture-shape-binary";
  auto coordinateVariant = mapping.variants[0];
  coordinateVariant.implementationId = coordinateImpl.id;
  coordinateVariant.artifactIdentity = coordinateImpl.artifactIdentity;
  coordinateVariant.resourceShape = {1, 1};
  std::vector<ProgramOperatorImplementation> coordinateImplementations{
      coordinateImpl};
  std::map<std::string, ProgramOperatorExecutionVariant> coordinateVariants{
      {coordinateImpl.id, coordinateVariant}};
  ProgramOperatorMappingAction coordinateAction;
  coordinateAction.kind =
      ProgramOperatorMappingActionKind::IndependentResourceShape;
  coordinateAction.firstRegion = 0;
  coordinateAction.regionCount = 1;
  coordinateAction.replacements = {coordinateImpl};
  coordinateAction.replacements[0].artifactIdentity = "caller-invented-binary";
  if (!require(applyProgramOperatorRegisteredMappingAction(
                   candidate, coordinateAction, coordinateImplementations,
                   coordinateVariants, child, error) &&
                   child.selection.regions[0].artifactIdentity ==
                       "fixture-shape-binary",
               "resource action resolves the actual atomic registration, "
               "ignoring caller field edits"))
    return 1;
  coordinateAction.replacements[0].id = "unregistered-id";
  if (!require(!applyProgramOperatorRegisteredMappingAction(
                   candidate, coordinateAction, coordinateImplementations,
                   coordinateVariants, child, error),
               "coordinate action cannot invent an unregistered executable"))
    return 1;
  coordinateAction.replacements = {coordinateImpl};
  coordinateAction.kind = ProgramOperatorMappingActionKind::IndependentReplica;
  coordinateVariants[coordinateImpl.id].replicas = 2;
  coordinateVariants[coordinateImpl.id].workDistribution =
      "registered-disjoint-work";
  coordinateVariants[coordinateImpl.id].workPartitions = {{2, 0, {0}, {2}},
                                                          {2, 1, {2}, {2}}};
  if (!require(applyProgramOperatorRegisteredMappingAction(
                   candidate, coordinateAction, coordinateImplementations,
                   coordinateVariants, child, error) &&
                   child.mapping.variants[0].replicas == 2,
               "replica action binds an actual registered work-partition "
               "implementation"))
    return 1;
  coordinateAction.kind =
      ProgramOperatorMappingActionKind::IndependentResidence;
  coordinateVariants[coordinateImpl.id] = coordinateVariant;
  coordinateVariants[coordinateImpl.id].residencePolicy = "fixture-retain";
  coordinateVariants[coordinateImpl.id].residentAllocations = {
      {"fixture-weight", {"abi", 16}}};
  if (!require(
          applyProgramOperatorRegisteredMappingAction(
              candidate, coordinateAction, coordinateImplementations,
              coordinateVariants, child, error),
          "residence action uses a complete registered physical reservation"))
    return 1;
  coordinateVariants[coordinateImpl.id] = mapping.variants[0];
  coordinateVariants[coordinateImpl.id].implementationId = coordinateImpl.id;
  coordinateVariants[coordinateImpl.id].artifactIdentity =
      coordinateImpl.artifactIdentity;
  if (!require(!applyProgramOperatorRegisteredMappingAction(
                   candidate, coordinateAction, coordinateImplementations,
                   coordinateVariants, child, error),
               "renamed artifact alone does not constitute a residence change"))
    return 1;
  coordinateAction.kind = ProgramOperatorMappingActionKind::TransferRoute;
  if (!require(!applyProgramOperatorRegisteredMappingAction(
                   candidate, coordinateAction, coordinateImplementations,
                   coordinateVariants, child, error),
               "transfer route still requires an actually lowered copy ABI"))
    return 1;
  // Independent discrete-time oracle: durations and reservations are integral,
  // so exhaustive tick placement agrees with endpoint placement without
  // sharing the production calendar or its resource-probe implementation.
  uint32_t randomState = 971;
  auto nextRandom = [&]() {
    randomState = randomState * 1664525u + 1013904223u;
    return randomState;
  };
  for (size_t trial = 0; trial < 24; ++trial) {
    j::Array fixtureValues, fixtureOperators;
    fixtureValues.push_back(j::Object{{"id", "x"},
                                      {"dtype", "f32"},
                                      {"shape", j::Array{4}},
                                      {"role", "input"},
                                      {"layout", "row_major"}});
    std::vector<std::vector<size_t>> expectedDependencies(6);
    for (size_t i = 0; i < 6; ++i) {
      auto name = "v" + std::to_string(i);
      fixtureValues.push_back(j::Object{{"id", name},
                                        {"dtype", "f32"},
                                        {"shape", j::Array{4}},
                                        {"role", "output"},
                                        {"layout", "row_major"}});
      j::Array inputs{"x"}, control;
      for (size_t p = 0; p < i; ++p) {
        if ((trial == 0 && p + 1 == i) || nextRandom() % 4 == 0) {
          inputs.push_back("v" + std::to_string(p));
          expectedDependencies[i].push_back(p);
        } else if (nextRandom() % 5 == 0) {
          control.push_back("op" + std::to_string(p));
          expectedDependencies[i].push_back(p);
        }
      }
      fixtureOperators.push_back(j::Object{{"id", "op" + std::to_string(i)},
                                           {"kind", "fixture"},
                                           {"binding", "fixture"},
                                           {"source_ordinal", int64_t(i)},
                                           {"inputs", std::move(inputs)},
                                           {"outputs", j::Array{name}},
                                           {"dependencies", std::move(control)},
                                           {"attributes", j::Object{}}});
    }
    auto oracleDocument = copyJSONWithoutFields(*parsed);
    (*oracleDocument.getAsObject())["values"] = std::move(fixtureValues);
    (*oracleDocument.getAsObject())["operators"] = std::move(fixtureOperators);
    std::shared_ptr<ProgramOperatorSource> oracleSource;
    if (!require(
            parseProgramOperatorSource(oracleDocument, oracleSource, error),
            "random discrete calendar source parses"))
      return 1;
    ProgramOperatorMappingCandidate oracleCandidate;
    oracleCandidate.mapping.registrationIdentity = "fixture-discrete-oracle";
    auto oracleTarget = readyTarget;
    oracleTarget.resources[1].capacity = 2;
    const int64_t scratchCapacity = 3 + nextRandom() % 5;
    oracleTarget.memories[0].capacityBytes = 7 * 16 + scratchCapacity;
    std::vector<int64_t> duration(6), scratch(6), resource(6);
    for (size_t i = 0; i < 6; ++i) {
      auto impl = selection.regions[0];
      impl.id = "oracle" + std::to_string(i);
      impl.operators = {i};
      impl.inputs = oracleSource->operators[i].inputs;
      impl.outputs = oracleSource->operators[i].outputs;
      resource[i] = nextRandom() % 2;
      impl.device = impl.resource = resource[i] ? "npu" : "cpu";
      impl.launchCount = resource[i];
      duration[i] = 1 + nextRandom() % 5;
      impl.wallMs = double(duration[i]);
      auto variant = readyMapping.variants[0];
      variant.implementationId = impl.id;
      variant.resourceDemands = {{impl.resource, 1}};
      scratch[i] = nextRandom() % (scratchCapacity + 1);
      variant.temporaryBytes = {{"abi", scratch[i]}};
      oracleCandidate.selection.regions.push_back(impl);
      oracleCandidate.mapping.variants.push_back(variant);
      oracleCandidate.mapping.readyPriority.push_back(i);
    }
    for (size_t i = 5; i; --i)
      std::swap(oracleCandidate.mapping.readyPriority[i],
                oracleCandidate.mapping.readyPriority[nextRandom() % (i + 1)]);
    std::vector<int64_t> starts(6, -1), ends(6, -1);
    int64_t expectedMakespan = 0;
    for (size_t issued = 0; issued < 6; ++issued) {
      size_t chosen = SIZE_MAX;
      for (size_t i : oracleCandidate.mapping.readyPriority) {
        if (ends[i] >= 0)
          continue;
        bool ready = true;
        for (size_t p : expectedDependencies[i])
          ready &= ends[p] >= 0;
        if (ready) {
          chosen = i;
          break;
        }
      }
      int64_t earliest = 0;
      for (size_t p : expectedDependencies[chosen])
        earliest = std::max(earliest, ends[p]);
      for (int64_t begin = earliest;; ++begin) {
        bool fits = true;
        for (int64_t tick = begin; tick < begin + duration[chosen]; ++tick) {
          int64_t memoryLive = scratch[chosen], resourceLive = 1;
          for (size_t i = 0; i < 6; ++i)
            if (starts[i] <= tick && tick < ends[i]) {
              memoryLive += scratch[i];
              resourceLive += resource[i] == resource[chosen];
            }
          fits &= memoryLive <= scratchCapacity &&
                  resourceLive <= (resource[chosen] ? 2 : 1);
        }
        if (!fits)
          continue;
        starts[chosen] = begin;
        ends[chosen] = begin + duration[chosen];
        expectedMakespan = std::max(expectedMakespan, ends[chosen]);
        break;
      }
    }
    auto oracleResult = evaluateProgramOperatorMappingCandidate(
        oracleSource, oracleCandidate, oracleTarget, "abi", readyGraph);
    if (!require(oracleResult.feasible && oracleResult.feasibilityKnown &&
                     oracleResult.costMs &&
                     *oracleResult.costMs == expectedMakespan,
                 "endpoint resource/memory calendar agrees with independent "
                 "discrete oracle"))
      return 1;
    std::vector<ProgramOperatorChoiceGroup> oracleGroups;
    std::map<std::string, ProgramOperatorExecutionVariant> oracleVariants;
    for (size_t i = 0; i < 6; ++i) {
      oracleGroups.push_back(
          {"oracle-group" + std::to_string(i), {"only"}, {{i}}});
      oracleVariants.emplace(oracleCandidate.selection.regions[i].id,
                             oracleCandidate.mapping.variants[i]);
    }
    std::shared_ptr<const ProgramOperatorMappingDomain> oracleDomain;
    if (!require(prepareProgramOperatorMappingDomain(
                     *oracleSource, oracleCandidate.selection.regions,
                     oracleGroups, oracleVariants, "oracle-domain",
                     oracleTarget, "abi", oracleDomain, error),
                 "random dense calendar domain prepares"))
      return 1;
    std::vector<size_t> oracleKeys(6);
    for (size_t i = 0; i < 6; ++i)
      oracleKeys[oracleCandidate.mapping.readyPriority[i]] = i;
    auto denseResult = evaluateProgramOperatorMappingDomain(
        *oracleDomain, {0, 0, 0, 0, 0, 0}, "source-order", &oracleKeys);
    auto fullReference = evaluateProgramOperatorMappingDomainReference(
        *oracleDomain, {0, 0, 0, 0, 0, 0}, "source-order", &oracleKeys);
    if (!require(denseResult.feasible && denseResult.feasibilityKnown &&
                     denseResult.costMs &&
                     *denseResult.costMs == expectedMakespan &&
                     fullReference.costMs == denseResult.costMs,
                 "dense calendar/scratch certificate/serial suffix match full "
                 "calendar and discrete oracle"))
      return 1;
    if (trial == 0 && !require(denseResult.result.getAsObject()
                                       ->getInteger("serial_tail_invocations")
                                       .value_or(0) > 0,
                               "strict source-dependent suffix actually "
                               "exercises exact evaluator reduction"))
      return 1;
  }
  ProgramOperatorInvocationControlModel controllerModel;
  controllerModel.identity = "fixture-controller-model";
  controllerModel.sourceId = prioritySource->sourceId;
  controllerModel.coefficients = {{"cpu_callbacks", 0}, {"npu_callbacks", 20}};
  auto controllerImplementations = convexImplementations;
  for (auto &impl : controllerImplementations)
    impl.controllerFeatures = {{"cpu_callbacks", impl.device == "cpu" ? 1 : 0},
                               {"npu_callbacks", impl.device == "npu" ? 1 : 0}};
  std::shared_ptr<const ProgramOperatorMappingDomain> controllerDomain;
  if (!require(prepareProgramOperatorMappingDomain(
                   *prioritySource, controllerImplementations, convexGroups,
                   convexVariants, "fixture-controller-domain", readyTarget,
                   "abi", controllerDomain, error, &controllerModel),
               "candidate-dependent controller features prepare with local "
               "prices unchanged"))
    return 1;
  auto controllerFused =
      evaluateProgramOperatorMappingDomain(*controllerDomain, {1, 0});
  auto controllerUnfused = evaluateProgramOperatorMappingDomain(
      *controllerDomain, {0, 0}, "reverse-source-order");
  auto controllerReference = evaluateProgramOperatorMappingDomainReference(
      *controllerDomain, {0, 0}, "reverse-source-order");
  if (!require(controllerFused.costMs && *controllerFused.costMs == 13 &&
                   controllerUnfused.costMs &&
                   *controllerUnfused.costMs == 31 &&
                   controllerReference.costMs == controllerUnfused.costMs &&
                   controllerUnfused.result.getAsObject()->getNumber(
                       "physical_callback_makespan_ms") == 11 &&
                   controllerUnfused.result.getAsObject()->getNumber(
                       "invocation_control_ms") == 20,
               "outside callback controller cost changes fused/unfused "
               "preference and stays separately auditable"))
    return 1;
  auto missingControllerFeatures = controllerImplementations;
  missingControllerFeatures[0].controllerFeatures.erase("npu_callbacks");
  std::shared_ptr<const ProgramOperatorMappingDomain> badControllerDomain;
  if (!require(!prepareProgramOperatorMappingDomain(
                   *prioritySource, missingControllerFeatures, convexGroups,
                   convexVariants, "fixture-missing-feature", readyTarget,
                   "abi", badControllerDomain, error, &controllerModel),
               "missing explicit zero controller feature is unknown, never "
               "inferred free"))
    return 1;
  auto controllerCatalogue = j::Value(j::Object{
      {"invocation_control_model",
       j::Object{
           {"schema", "orbit.program-invocation-control-model.v1"},
           {"identity", "fixture-controller-model"},
           {"source_id", prioritySource->sourceId},
           {"boundary", "outside-inclusive-invocation-wall"},
           {"intercept_ms", 0.5},
           {"coefficients",
            j::Object{{"cpu_callbacks", 0.1}, {"npu_callbacks", 0.2}}},
           {"fit",
            j::Object{
                {"scope", "synthetic fixture; not a device calibration"}}}}}});
  std::optional<ProgramOperatorInvocationControlModel> parsedController;
  if (!require(
          parseProgramOperatorInvocationControlModel(
              *prioritySource, controllerCatalogue, parsedController, error) &&
              parsedController,
          "explicit calibrated controller model parses"))
    return 1;
  auto unknownController = copyJSONWithoutFields(controllerCatalogue);
  (*unknownController.getAsObject()
        ->getObject("invocation_control_model")
        ->getObject("coefficients"))["cpu_callbacks"] = nullptr;
  if (!require(!parseProgramOperatorInvocationControlModel(
                   *prioritySource, unknownController, parsedController, error),
               "unknown controller coefficient cannot become zero"))
    return 1;
  auto negativeController = copyJSONWithoutFields(controllerCatalogue);
  (*negativeController.getAsObject()->getObject(
      "invocation_control_model"))["intercept_ms"] = -1;
  if (!require(
          !parseProgramOperatorInvocationControlModel(
              *prioritySource, negativeController, parsedController, error),
          "negative controller intercept is rejected"))
    return 1;
  controllerModel.coefficients["cpu_callbacks"] =
      std::numeric_limits<double>::quiet_NaN();
  double controllerWall;
  std::map<std::string, int64_t> controllerCounts;
  if (!require(
          !estimateProgramOperatorInvocationControl(
              {}, controllerModel, controllerWall, controllerCounts, error),
          "nonfinite typed controller model is rejected"))
    return 1;
  llvm::outs() << "{\"status\":\"passed\",\"checks\":" << checks << "}\n";
  return 0;
}
