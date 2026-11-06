# Public mechanism evaluation (v16)

`ProgramGraph` remains the single semantic program and data-realization ledger.
`GenericAnalysis` supplies real MLIR bodies, logical storage, regions and versions;
`GenericMappingPlan.communication` owns copies, allocations, transfer links,
physical execution events and execution regions. No edge is assumed to have an
independent additive latency. Backend descriptor coalescing can bind multiple
public events to one exclusive implementation cost.

`ProgramTiming.h/.cpp` adds evaluation annotations over this checked graph, not a
second graph or scheduler. `evaluateProgramTiming` rechecks source, graph and
identity, contracts exclusive event coverage into implementation intervals, and
calls the existing `scheduleStaticPlan` resource-calendar scheduler. Production
logic remains C++/MLIR; JSON is an evidence export.

The three layers are:

| Layer | Existing object / v16 use |
| --- | --- |
| Semantic computation and dependencies | `GenericAnalysis`, `MappingDataEdge`, `MappingOrderEdge`; never changed by a price |
| Physical realization | `MappingCopy`, `MappingAllocation`, `MappingEvent`, links, execution-region lifecycle |
| Cost and timing | `ProgramActivityCost` annotations, exclusive owners, duration/bounds, fixed resource demands; existing scheduler |

A cost annotation names covered public and/or lifecycle events, implementation,
measurement boundary, provenance, resource demand and duration or bounds. Missing
coverage becomes an explicit unknown activity. For native graphs without refined lifecycle events, the `executionRegions`
annotation must explicitly own each submitted region and all its member events;
otherwise the evaluator adds an unknown invocation boundary. Pricing computation
and transfers alone cannot produce a complete native-call prediction.
Extra executor work may be marked
`runtimeOnly` with no graph coverage and explicit predecessor cost IDs; it is not
silently assumed absent. The default boundary is graph events plus region
lifecycle, not all unlisted Python/host work. A caller claiming a larger boundary
must supply the corresponding runtime activities.

A composite may own DMA, scalar local loads/stores, locks, configuration and
polling together when they cannot be independently identified. It reserves every
covered event's declared resource and may reserve additional shared resources.
Independent prices for covered events fail. Nonconvex coverage that creates a
contracted dependency cycle also fails, rather than discarding intermediate
ordering. Internal ordering in a valid composite belongs to its implementation
provider and binding evidence.

The scheduler respects event/data dependencies, submit-before-member and
member-before-complete boundaries, existing lifecycle visibility dependencies,
explicit runtime predecessor IDs, and buffer reuse completion. Distinct engines
can overlap. A transfer can reserve a DMA engine and an additional shared bus;
activities compete for integer capacity on both resources. Fixed bandwidth shares
require supplied implementation durations at those shares. Adaptive fluid
bandwidth redistribution is not inferred or measured by this API.

For exact implementation durations `d_i`, resource demands `u_ir` and capacities
`C_r`, the scheduler chooses ready intervals satisfying dependencies and calendar
occupancy. Whole completion is `max_i end_i`, with separate calendar resource predecessors and a realized
critical path, not the sum of edge latencies or an
unconditional compute/communication maximum. With lower durations `l_i`, the
reported mathematical lower bound is

```
L = max(longest_precedence_path(l), max_r sum_i(l_i * u_ir) / C_r).
```

The lower-duration list-schedule timeline is separately marked diagnostic; its
heuristic makespan is not a mathematical lower bound. Known work is the sum of
known activity durations and is not elapsed time. Unknown activities prevent an
exact makespan. If every activity has an upper bound `h_i`, `sum_i h_i` is a
serialized time/resource work bound. Its capacity scope is explicit; incomplete
models do not certify temporal memory feasibility.

A complete predicted timeline rechecks simultaneous allocation lifetimes against
known memory capacities. Copies sharing an allocation are charged once across
the conservative union span. `releasedAfter` is an exclusive event-index boundary,
so its final retained event is `releasedAfter - 1`; program-retained copies last
until completion. Version reuse of an allocation adds release/last-consumer to
next-creator completion constraints. Capacity overflow rejects the evaluation.
Unknown physical capacity remains reported; `knownCapacitiesVerified` and
`allPhysicalCapacitiesKnown` distinguish checked bounds from physical knowledge.
A composite retains each owned allocation for its entire enclosing interval,
which may reject a physically feasible implementation whose internal lifetimes
are shorter. Scalar GEMM/K-tile feature models are not evidence of measured
compute/communication overlap. This evaluator detects capacity-infeasible overlap; it does not perform a new
memory placement or optimal buffer rescheduling search.

Existing `ProgramGraph.schedule`, Amoeba mapper/scheduler and CGRA ML cost
contracts remain unchanged. Backends opt into this evaluator with explicit model
identity and implementation annotations. NPU BD/lock/FIFO details remain backend
binding evidence; only their dependency and reservation effects enter public
annotations.

`test/TaskflowGeneric/program-timing.cpp` exercises real source-derived graphs
with synthetic costs. It checks unknown diagnostics/bounds, dependency completion,
exclusive composite coverage, contracted dependency cycles, malformed domains,
resource substitution/capacity, mutual exclusion, overlap and fixed resource
shares. These are compiler/model contract tests, not device timing evidence.
