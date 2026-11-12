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
  llvm::outs() << "{\"status\":\"passed\",\"checks\":" << checks << "}\n";
  return 0;
}
