# Data-region reuse discovery and compatible tiling (v20)

The production path remains native Taskflow → GenericAnalysis → checked
ProgramGraph → ProgramTiming/original structured search → AIR → AIE. Python
generates arithmetic inputs, invokes tools and records observations. It does not
select production groups, derive legality, construct copies or price candidates.

## Intake audit

1. `GenericAccess` binds actual MLIR operations to logical storage, exact bounding
   rectangles, `writeVersion` and `readVersions`. `GenericTask.dependencies`
   contains both data and order constraints. ProgramGraph represents individual
   versions, requirements, location copies, allocations and users separately.
2. v19 discovers producer/reader intervals in the current topological task
   numbering. `arrangePhases` requires their members to be contiguous in the
   selected execution order; accepting arbitrary members without contracting
   dependencies would allow cycles. Neither storage-name matching nor deleting
   dependencies is a valid extension.
3. `deriveGenericPhaseRegions`, the native source transforms,
   `planGenericMapping`, `checkGenericMapping`, communication checking and actual
   AIR/AIE checking provide the existing region, capacity and implementation
   proofs. Target capabilities declare memory/resource bindings, stream limits,
   supported implementations and acquisition policy.
4. Tiling is a source attribute set through `setNativeAIRTileSizes`. The original
   SearchAction already carries both tile options and a mapping choice. Its
   native transformation materializes both before source identity, evaluation,
   caching, deduplication and checkpointing. No second search driver is needed.
5. Discovery can reject unsupported identity-grid propagation, infeasible
   capacity, stream counts, dependency contraction and source access relations.
   Actual operations, DMA routes, buffer identity, lock release and runtime
   coverage still require final implementation checking.

## Interface and algorithm

`GenericReuse.h` adds analysis facts and bounded proposals to the existing Core.
An opportunity names storage, exact rectangle/version, optional producer and
consumer demands. A proposal carries the existing `ProgramMappingChoice`, tile
sizes and diagnostics. It is not a parallel execution graph or scheduler.

For each opportunity, bounded consumer subsets seed a producer/consumer group.
Dependency-path closure includes intervening required tasks, not independently
interleaved tasks. The quotient graph contracts selected groups and topologically
orders the quotient; a cycle rejects the proposal. Expansion preserves original
task indices, task body pointers and all dependencies. Existing tile-major
lowering therefore receives contiguous execution groups without rewriting or
renumbering source tasks. The source can remain in either legal arrangement.

Exact identity-grid accesses propagate the same rectangle through selected
tasks. Candidate tiles and scopes are checked together by cloning source and
using the existing planner. Required internal produced edges are retained;
outside readers and caller outputs retain boundary exports. Shared inputs select
one immutable external version and identical rectangular block reads. They keep
the first inbound transfer and share its local copy among all block consumers;
refill waits for all old readers. Unequal reader rectangles reject.

Capacity/traffic/configuration diagnostics screen candidates before expensive
evaluation. Whole-program resource scheduling and the same explicit-assumption
model choose search parents; fewer bytes alone does not establish lower latency.
Strict mode preserves unknown absolute cost. Search budgets are configuration,
not target facts. The compatibility default is v19 interval discovery; the joint
policy is explicitly selected in `native_auto_candidates.algorithm`.

## Relation to Welder

[Welder, OSDI 2023, §3.1–3.2 and Figure 7](https://www.usenix.org/system/files/osdi23-shi.pdf)
motivates trying memory reuse connections together with compatible tile shapes,
then rejecting capacity violations and screening by traffic. This implementation
applies those ideas to existing ProgramGraph facts and source transformations.
It does not add Welder's tile-graph, assume GPU memory-layer independence or
replace whole-program timing with a traffic objective. Propagation here is
restricted to proven rectangular identity grids; reductions, in-place updates
and nonidentity access patterns do not inherit a general propagation claim.

## Team and evidence

The reuse agent owns opportunity/constraint analysis and contraction; the search
agent owns existing-driver integration and four-policy auditing; the backend
agent owns shared-input realization and independent negative checks. The main
agent owns interfaces, integration, new arithmetic fixtures and all device runs.

Current results, failed attempts, immutable tool snapshots and fresh replay
commands belong in companion `results/taskflow/data-reuse-v20`. Old v19 evidence
and frozen binaries remain unchanged. Final result claims are selected by that
directory's DELIVERY.json after validation; this design description alone is
not execution evidence.
