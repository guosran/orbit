// Actual source-backed common-graph tests. Costs/architecture are synthetic.
static bool runExecutionSynthetic(mlir::MLIRContext &context, unsigned &checks,
                                  std::string &error,
                                  llvm::json::Object *evidence = nullptr) {
  const char *text = R"mlir(module {
 func.func @branches(%a:memref<1x16xf32> {orbit.name="A",orbit.role="input"},%d:memref<1x16xf32> {orbit.name="D",orbit.role="input"},%b:memref<1x16xf32> {orbit.name="B",orbit.role="output"},%c:memref<1x16xf32> {orbit.name="C",orbit.role="output"}) attributes {orbit.noalias} {
 %b0=taskflow.task @long will_reads(%a:memref<1x16xf32>) will_writes(%b:memref<1x16xf32>) [original_read_memrefs(%a:memref<1x16xf32>),original_write_memrefs(%b:memref<1x16xf32>)] : (memref<1x16xf32>,memref<1x16xf32>)->memref<1x16xf32> {
 ^bb0(%x:memref<1x16xf32>,%y:memref<1x16xf32>):
 affine.for %i=0 to 1 { affine.for %j=0 to 16 { %v=affine.load %x[%i,%j]:memref<1x16xf32>
 affine.store %v,%y[%i,%j]:memref<1x16xf32> } }
 taskflow.yield done_writes(%y:memref<1x16xf32>) }
 %c0=taskflow.task @short will_reads(%d:memref<1x16xf32>) will_writes(%c:memref<1x16xf32>) [original_read_memrefs(%d:memref<1x16xf32>),original_write_memrefs(%c:memref<1x16xf32>)] : (memref<1x16xf32>,memref<1x16xf32>)->memref<1x16xf32> {
 ^bb0(%x:memref<1x16xf32>,%y:memref<1x16xf32>):
 affine.for %i=0 to 1 { affine.for %j=0 to 16 { %v=affine.load %x[%i,%j]:memref<1x16xf32>
 affine.store %v,%y[%i,%j]:memref<1x16xf32> } }
 taskflow.yield done_writes(%y:memref<1x16xf32>) }
 return }
 })mlir";
  bool ok = true;
  auto require = [&](bool value, const char *name) {
    if (value)
      ++checks;
    else {
      ok = false;
      llvm::errs() << name << ": " << error << "\n";
    }
    error.clear();
  };
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  GenericAnalysis a;
  if (!module || !analyzeGenericProgram(*module, a, error))
    return false;
  StaticConfig cfg;
  cfg.resources = {{"compute0", "compute", "synth", 1},
                   {"compute1", "compute", "synth", 1},
                   {"dma", "transfer", "synth", 1}};
  cfg.memories = {{"host", 4096, {"synth"}},
                  {"local0", 256, {"synth"}},
                  {"local1", 256, {"synth"}}};
  cfg.memories[1].accessibleResources = {"compute0"};
  cfg.memories[2].accessibleResources = {"compute1"};
  for (auto entry :
       {std::make_pair("in0", "local0"), std::make_pair("in1", "local1")}) {
    StaticTransfer l;
    l.id = entry.first;
    l.source = "host";
    l.target = entry.second;
    l.resource = "dma";
    cfg.links.push_back(l);
    l.id = std::string("out") + (entry.first[2]);
    std::swap(l.source, l.target);
    cfg.links.push_back(l);
  }
  cfg.assumptions.initialLocations = {{"A", "host"}, {"D", "host"}};
  cfg.assumptions.outputLocations = {{"B", "host"}, {"C", "host"}};
  ProgramDataOrganizationChoice choice;
  choice.semanticDependenciesOnly = true;
  choice.policy = ProgramDataPolicy::RetainAcrossUsers;
  choice.placements = {{0, "compute0", "local0", "scalar"},
                       {1, "compute1", "local1", "scalar"}};
  ProgramGraph g;
  if (!planProgramDataOrganization(a, cfg, choice, g, error)) {
    llvm::errs() << error;
    return false;
  }
  require(
      checkProgramGraph(g, error),
      "two independent actual source branches have checked copies/resources");
  ProgramTimingRequest q;
  q.graphId = programGraphIdentity(g);
  q.modelIdentity = "synthetic-v18-assumed-ms";
  q.boundary = "source-graph-only";
  std::string in0, in1, c0, c1, out0, out1;
  for (size_t i = 0; i < g.events.size(); ++i) {
    const auto &e = g.events[i];
    ProgramActivityCost c;
    c.id = "s" + std::to_string(i);
    c.events = {i};
    c.provenance = "synthetic target assumption, not measured NPU";
    c.boundary = "one-public-event";
    c.resourceDemands[e.resource] = 1;
    if (e.kind == "transfer") {
      c.durationMs = e.link.rfind("in", 0) == 0 ? 2 : 1;
      if (e.link == "in0")
        in0 = c.id;
      if (e.link == "in1")
        in1 = c.id;
      if (e.link == "out0")
        out0 = c.id;
      if (e.link == "out1")
        out1 = c.id;
    } else {
      c.durationMs = e.task == 0 ? 10 : 1;
      if (e.task == 0)
        c0 = c.id;
      else
        c1 = c.id;
    }
    q.costs.push_back(c);
  }
  auto semantic = programGraphIdentity(g);
  q.schedulePriority = {in0, c0, in1, c1, out0, out1};
  ProgramTimingEvaluation r;
  require(evaluateProgramTiming(g, q, r, error) && r.makespanMs &&
              *r.makespanMs == 13,
          "long-branch input priority gives 13ms");
  if (evidence) {
    (*evidence)["parallel_graph"] = serializeProgramGraph(g);
    (*evidence)["long_first_timing"] = serializeProgramTiming(r);
  }
  require(programGraphIdentity(g) == semantic,
          "priority does not rewrite semantic graph");
  bool dmaExclusive = true, computeOverlap = false;
  for (size_t i = 0; i < r.activities.size(); ++i)
    for (size_t j = i + 1; j < r.activities.size(); ++j) {
      auto &l = r.activities[i], &v = r.activities[j];
      bool overlap = *l.startMs < *v.endMs && *v.startMs < *l.endMs;
      if (l.resourceDemands.count("dma") && v.resourceDemands.count("dma") &&
          overlap)
        dmaExclusive = false;
      if (l.resourceDemands.count("compute0") &&
          v.resourceDemands.count("compute1") && overlap)
        computeOverlap = true;
    }
  require(
      dmaExclusive,
      "single DMA resource excludes independent delivery/export activities");
  require(computeOverlap, "distinct actual compute resources overlap");
  q.schedulePriority = {in1, c1, in0, c0, out1, out0};
  require(evaluateProgramTiming(g, q, r, error) && r.makespanMs &&
              *r.makespanMs == 15,
          "short-branch priority gives 15ms without new dependencies");
  if (evidence)
    (*evidence)["short_first_timing"] = serializeProgramTiming(r);
  require(programGraphIdentity(g) == semantic &&
              r.schedulePriority == q.schedulePriority,
          "explicit schedule identity exported independently");
  auto bad = q;
  bad.schedulePriority[1] = bad.schedulePriority[0];
  require(!evaluateProgramTiming(g, bad, r, error),
          "duplicate scheduling priority rejected");
  bad = q;
  bad.costs[0].resourceDemands = {{"compute0", 1}};
  require(!evaluateProgramTiming(g, bad, r, error),
          "DMA reservation cannot be replaced by a compute slot");
  auto inaccessible = g;
  for (auto &m : inaccessible.memories)
    if (m.id == "local0")
      m.accessibleResources = {"compute1"};
  require(!checkProgramGraph(inaccessible, error),
          "compute cannot read inaccessible replica");
  auto small = g;
  for (auto &m : small.memories)
    if (m.id == "local0")
      m.capacityBytes = 63;
  require(!checkProgramGraph(small, error),
          "physical memory capacity rejects undersized target");
  auto stale = g;
  bool changed = false;
  for (auto &e : stale.events)
    if (e.kind == "compute" && !e.reads.empty()) {
      e.reads[0] = g.events.back().writes.front();
      changed = true;
      break;
    }
  require(changed && !checkProgramGraph(stale, error),
          "fabricated stale/wrong storage copy rejected against actual read "
          "source");

  // An order-only source token remains a real completion dependency even when
  // the scheduling vector is no longer treated as a semantic chain.
  std::string orderedSource = text;
  size_t shortAt = orderedSource.find("%c0=taskflow.task");
  std::string shortBody = orderedSource.substr(shortAt);
  auto replaceOne = [&](const std::string &from, const std::string &to) {
    auto at = shortBody.find(from);
    if (at != std::string::npos)
      shortBody.replace(at, from.size(), to);
  };
  replaceOne("will_reads(%d:memref<1x16xf32>)",
             "will_reads(%d,%b0:memref<1x16xf32>,memref<1x16xf32>)");
  replaceOne("original_read_memrefs(%d:memref<1x16xf32>)",
             "original_read_memrefs(%d,%b0:memref<1x16xf32>,memref<1x16xf32>)");
  replaceOne("(memref<1x16xf32>,memref<1x16xf32>)->",
             "(memref<1x16xf32>,memref<1x16xf32>,memref<1x16xf32>)->");
  replaceOne("^bb0(%x:memref<1x16xf32>,%y:",
             "^bb0(%x:memref<1x16xf32>,%orderOnly:memref<1x16xf32>,%y:");
  orderedSource.replace(shortAt, std::string::npos, shortBody);
  auto orderedModule =
      mlir::parseSourceString<mlir::ModuleOp>(orderedSource, &context);
  GenericAnalysis oa;
  ProgramGraph orderedGraph;
  require(
      orderedModule && analyzeGenericProgram(*orderedModule, oa, error) &&
          planProgramDataOrganization(oa, cfg, choice, orderedGraph, error) &&
          orderedGraph.orderEdges.size() == 1,
      "semantic-only planner preserves actual source order-only dependency");
  if (!orderedGraph.orderEdges.empty()) {
    auto &edge = orderedGraph.orderEdges.front();
    require(std::find(orderedGraph.events[edge.consumer].dependencies.begin(),
                      orderedGraph.events[edge.consumer].dependencies.end(),
                      edge.producer) !=
                orderedGraph.events[edge.consumer].dependencies.end(),
            "order-only completion is explicit independently of priority");
  }

  // Multi-hop paths reuse one event/copy per directed architectural link.
  auto routedCfg = cfg;
  routedCfg.memories.push_back({"staging", 256, {"synth"}});
  for (auto &l : routedCfg.links)
    if (l.id == "in0")
      l.source = "staging";
  StaticTransfer stageLink;
  stageLink.id = "host-stage";
  stageLink.source = "host";
  stageLink.target = "staging";
  stageLink.resource = "dma";
  routedCfg.links.push_back(stageLink);
  ProgramGraph routed;
  require(planProgramDataOrganization(a, routedCfg, choice, routed, error),
          "checked source graph accepts host-to-staging-to-local path");
  size_t stagedCopies = 0, lastHop = SIZE_MAX;
  for (size_t i = 0; i < routed.events.size(); ++i)
    if (routed.events[i].link == "host-stage")
      ++stagedCopies;
    else if (routed.events[i].link == "in0")
      lastHop = i;
  require(stagedCopies == 1 && lastHop != SIZE_MAX &&
              routed.events[lastHop].transferBytes == 64 &&
              routed.transferBytes == g.transferBytes + 64,
          "each physical hop preserves version/payload and reports its own "
          "traffic");
  auto missingHop = routed;
  if (lastHop != SIZE_MAX)
    missingHop.events[lastHop].dependencies.clear();
  require(lastHop != SIZE_MAX && !checkProgramGraph(missingHop, error),
          "multi-hop destination cannot precede intermediate replica creation");
  if (evidence)
    (*evidence)["multi_hop_graph"] = serializeProgramGraph(routed);

  // Actual checked same-source repeated deliveries distinguish physical slots.
  // The source tasks are independent; only a single-slot overwrite orders them.
  std::string repeatedSource = text;
  for (const auto &pattern : {std::string("@short will_reads(%d"),
                              std::string("original_read_memrefs(%d")}) {
    auto at = repeatedSource.find(pattern);
    if (at != std::string::npos)
      repeatedSource[at + pattern.size() - 1] = 'a';
  }
  auto repeatedModule =
      mlir::parseSourceString<mlir::ModuleOp>(repeatedSource, &context);
  GenericAnalysis ra;
  if (!repeatedModule || !analyzeGenericProgram(*repeatedModule, ra, error))
    return false;
  auto repeatedCfg = cfg;
  repeatedCfg.memories[1].accessibleResources = {"compute0", "compute1"};
  auto repeatedChoice = choice;
  repeatedChoice.policy = ProgramDataPolicy::PhaseLocal;
  repeatedChoice.placements[1].memory = "local0";
  ProgramGraph twoSlots;
  if (!planProgramDataOrganization(ra, repeatedCfg, repeatedChoice, twoSlots,
                                   error))
    return false;
  auto sharedChoice = repeatedChoice;
  sharedChoice.policy = ProgramDataPolicy::RetainAcrossUsers;
  ProgramGraph shared;
  require(
      planProgramDataOrganization(ra, repeatedCfg, sharedChoice, shared, error),
      "same-source retained multi-consumer plan is valid");
  size_t sharedImports = 0, sharedReaders = 0;
  for (auto &copy : shared.copies)
    if (copy.createdBy >= 0 &&
        shared.events[copy.createdBy].kind == "transfer" &&
        shared.allocations[copy.allocation].memory == "local0") {
      ++sharedImports;
      sharedReaders = copy.users.size();
    }
  require(sharedImports == 1 && sharedReaders == 2 &&
              shared.transferBytes + 64 == twoSlots.transferBytes,
          "two consumers share one actual resident input replica and save one "
          "64-byte import");
  if (evidence)
    (*evidence)["shared_replica_graph"] = serializeProgramGraph(shared);
  std::vector<size_t> delivered;
  for (size_t i = 0; i < twoSlots.copies.size(); ++i) {
    auto &c = twoSlots.copies[i];
    if (c.createdBy >= 0 && twoSlots.events[c.createdBy].kind == "transfer" &&
        twoSlots.allocations[c.allocation].memory == "local0")
      delivered.push_back(i);
  }
  require(delivered.size() == 2,
          "phase-local same-source graph materializes two input deliveries");
  if (delivered.size() != 2)
    return false;
  ProgramTimingRequest rq;
  rq.graphId = programGraphIdentity(twoSlots);
  rq.modelIdentity = q.modelIdentity;
  rq.boundary = q.boundary;
  for (size_t i = 0; i < twoSlots.events.size(); ++i) {
    auto &e = twoSlots.events[i];
    ProgramActivityCost cost;
    cost.id = "r" + std::to_string(i);
    cost.events = {i};
    cost.provenance = "synthetic target assumption, not measured NPU";
    cost.boundary = "one-public-event";
    cost.resourceDemands[e.resource] = 1;
    cost.durationMs = e.kind == "transfer"
                          ? (e.link.rfind("in", 0) == 0 ? 2 : 1)
                          : (e.task == 0 ? 10 : 1);
    rq.costs.push_back(cost);
  }
  for (const auto &c : rq.costs)
    rq.schedulePriority.push_back(c.id);
  ProgramTimingEvaluation twoResult;
  require(evaluateProgramTiming(twoSlots, rq, twoResult, error) &&
              twoResult.makespanMs && *twoResult.makespanMs == 13,
          "two physical input slots permit delivery/other-core overlap");
  auto oneSlot = twoSlots;
  size_t oldAllocation = oneSlot.copies[delivered[0]].allocation;
  oneSlot.allocations[oldAllocation].addressing = "reusable-slot";
  size_t removedAllocation = oneSlot.copies[delivered[1]].allocation;
  oneSlot.copies[delivered[1]].allocation = oldAllocation;
  oneSlot.allocations.erase(oneSlot.allocations.begin() + removedAllocation);
  for (auto &c : oneSlot.copies)
    if (c.allocation > removedAllocation)
      --c.allocation;
  require(checkProgramGraph(oneSlot, error),
          "single-slot alias plan passes actual source and lifetime checker");
  auto oneRequest = rq;
  oneRequest.graphId = programGraphIdentity(oneSlot);
  ProgramTimingEvaluation oneResult;
  require(evaluateProgramTiming(oneSlot, oneRequest, oneResult, error) &&
              oneResult.makespanMs && *oneResult.makespanMs == 16 &&
              !oneResult.bufferOrdering.empty(),
          "single physical slot adds overwrite ordering and 16ms completion");
  if (evidence) {
    (*evidence)["one_slot_graph"] = serializeProgramGraph(oneSlot);
    (*evidence)["two_slot_graph"] = serializeProgramGraph(twoSlots);
    (*evidence)["one_slot_timing"] = serializeProgramTiming(oneResult);
    (*evidence)["two_slot_timing"] = serializeProgramTiming(twoResult);
  }

  // Put an unrelated export ordinal immediately before the next input
  // overwrite. A topological validity boundary must not become a hidden global
  // drain edge.
  auto ordinalGraph = oneSlot;
  const std::vector<size_t> permutation = {0, 1, 4, 2, 3, 5};
  std::vector<size_t> remap(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i)
    remap[permutation[i]] = i;
  auto oldEvents = ordinalGraph.events;
  for (size_t i = 0; i < permutation.size(); ++i) {
    ordinalGraph.events[i] = oldEvents[permutation[i]];
    for (auto &p : ordinalGraph.events[i].dependencies)
      p = remap[p];
    std::sort(ordinalGraph.events[i].dependencies.begin(),
              ordinalGraph.events[i].dependencies.end());
  }
  for (auto &v : ordinalGraph.versions)
    if (v.producer >= 0)
      v.producer = remap[v.producer];
  for (auto &copy : ordinalGraph.copies) {
    if (copy.createdBy >= 0) {
      copy.createdBy = remap[copy.createdBy];
      copy.availableFrom = size_t(copy.createdBy);
    }
    if (copy.releasedAfter < remap.size())
      copy.releasedAfter = remap[copy.releasedAfter];
    for (auto &u : copy.users)
      u = remap[u];
    std::sort(copy.users.begin(), copy.users.end());
    copy.lastUsers = copy.users.empty()
                         ? std::vector<size_t>{}
                         : std::vector<size_t>{copy.users.back()};
    copy.releasedAfter = std::max(copy.releasedAfter, copy.availableFrom + 1);
    if (!copy.users.empty())
      copy.releasedAfter = std::max(copy.releasedAfter, copy.users.back() + 1);
  }
  for (auto &edge : ordinalGraph.dataEdges) {
    if (edge.producer >= 0)
      edge.producer = remap[edge.producer];
    edge.consumer = remap[edge.consumer];
  }
  for (auto &edge : ordinalGraph.orderEdges) {
    edge.producer = remap[edge.producer];
    edge.consumer = remap[edge.consumer];
  }
  require(finalizeProgramGraph(ra, ordinalGraph, nullptr, error) &&
              checkProgramGraph(ordinalGraph, error),
          "unrelated export may precede input reuse in a source-backed ordinal "
          "witness");
  auto ordinalRequest = oneRequest;
  ordinalRequest.graphId = programGraphIdentity(ordinalGraph);
  for (auto &cost : ordinalRequest.costs)
    for (auto &event : cost.events)
      event = remap[event];
  ProgramTimingEvaluation ordinalResult;
  require(evaluateProgramTiming(ordinalGraph, ordinalRequest, ordinalResult,
                                error) &&
              ordinalResult.makespanMs && *ordinalResult.makespanMs == 16,
          "ordinal release boundary cannot add unrelated export drain latency");
  require(std::none_of(ordinalResult.bufferOrdering.begin(),
                       ordinalResult.bufferOrdering.end(),
                       [&](auto &order) {
                         return order.predecessor == remap[4] &&
                                order.successor == remap[2];
                       }),
          "buffer ordering never guesses a release activity from releasedAfter "
          "minus one");
  auto explicitRelease = ordinalGraph;
  explicitRelease.events[remap[2]].dependencies.push_back(remap[4]);
  std::sort(explicitRelease.events[remap[2]].dependencies.begin(),
            explicitRelease.events[remap[2]].dependencies.end());
  require(finalizeProgramGraph(ra, explicitRelease, nullptr, error) &&
              checkProgramGraph(explicitRelease, error),
          "an explicit physical release dependency is a checked graph choice");
  auto releaseRequest = ordinalRequest;
  releaseRequest.graphId = programGraphIdentity(explicitRelease);
  ProgramTimingEvaluation releaseResult;
  require(evaluateProgramTiming(explicitRelease, releaseRequest, releaseResult,
                                error) &&
              releaseResult.makespanMs && *releaseResult.makespanMs == 17,
          "explicit release activity still gates input reuse when actually "
          "declared");
  if (evidence) {
    (*evidence)["ordinal_release_graph"] = serializeProgramGraph(ordinalGraph);
    (*evidence)["ordinal_release_timing"] =
        serializeProgramTiming(ordinalResult);
    (*evidence)["explicit_release_graph"] =
        serializeProgramGraph(explicitRelease);
    (*evidence)["explicit_release_timing"] =
        serializeProgramTiming(releaseResult);
  }

  auto premature = oneSlot;
  premature.copies[delivered[0]].releasedAfter =
      oneSlot.copies[delivered[1]].createdBy + 2;
  require(!checkProgramGraph(premature, error),
          "reuse before prior replica lifetime ends is rejected");
  auto overCapacity = twoSlots;
  for (auto &m : overCapacity.memories)
    if (m.id == "local0")
      m.capacityBytes = 127;
  auto capRequest = rq;
  capRequest.graphId = programGraphIdentity(overCapacity);
  ProgramTimingEvaluation capResult;
  require(!evaluateProgramTiming(overCapacity, capRequest, capResult, error),
          "overlapping physical copies must fit actual memory capacity");

  // Slot semantics are explicit even when logical offsets differ. The common
  // ordering helper must wait for every user, not only the largest ordinal.
  auto slots = g;
  size_t allocation = slots.copies.front().allocation;
  slots.allocations[allocation].addressing = "reusable-slot";
  MappingCopy old = slots.copies.front(), next = old;
  old.allocation = next.allocation = allocation;
  old.createdBy = 0;
  old.users = {1, 3};
  old.releasedAfter = 4;
  next.createdBy = 4;
  next.users = {5};
  next.releasedAfter = 6;
  slots.events.resize(6);
  slots.copies = {old, next};
  std::vector<ProgramBufferOrdering> ordering;
  require(collectProgramBufferOrdering(slots, ordering, error) &&
              std::any_of(ordering.begin(), ordering.end(),
                          [](const auto &x) {
                            return x.predecessor == 1 && x.successor == 4;
                          }) &&
              std::any_of(ordering.begin(), ordering.end(),
                          [](const auto &x) {
                            return x.predecessor == 3 && x.successor == 4;
                          }),
          "single slot overwrite waits for all consumers");
  auto ordinalOnly = slots;
  ordinalOnly.copies[0].users = {1};
  ordinalOnly.copies[0].releasedAfter = 4;
  ordinalOnly.copies[1].createdBy = 5;
  require(
      collectProgramBufferOrdering(ordinalOnly, ordering, error) &&
          std::none_of(ordering.begin(), ordering.end(),
                       [](const auto &x) { return x.predecessor == 3; }),
      "nonconsumer ordinal lifetime endpoint never invents a release activity");
  slots.allocations.push_back(slots.allocations[allocation]);
  slots.copies[1].allocation = slots.allocations.size() - 1;
  require(collectProgramBufferOrdering(slots, ordering, error) &&
              ordering.empty(),
          "double slot separates physical overwrite constraints");
  return ok;
}
