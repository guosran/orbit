# Public ProgramGraph joint search (v25)

`native_auto_candidates.algorithm = "cp-sat-all"` selects a C++ joint search
using the existing source actions, `ProgramMappingChoice`, `ProgramGraph`, target
architecture, common checker, mechanism provider and resource timing evaluator.
It does not use the v23 pointwise-only `ProgramJointDomain` builder.

Every proposal assigns the admitted dimensions together. CP-SAT maintains integer
choice variables, public compatibility tables and forbidden checked assignments.
Its proposal objective covers untried or infrequently tried choices. The complete
materialized graph's evaluator cost determines the incumbent. **CP-SAT's coverage
objective is not latency, a latency bound or a hardware optimality certificate.**
No complete-plan catalogue is priced before solving. Finite-domain optimality
requires exhausting all admitted compatible assignments, with known feasibility
and known cost for every feasible assignment.

```json
{
  "native_auto_candidates": {"mode": "joint", "algorithm": "cp-sat-all"},
  "program_search": {
    "seconds": 600,
    "max_evaluations": 2048,
    "max_variables": 100000,
    "max_constraints": 500000,
    "max_tuple_cells": 1000000,
    "max_values_per_dimension": 16,
    "max_source_instances": 512,
    "max_full_evaluations": 64,
    "max_routes": 16,
    "max_priorities": 24,
    "workers": 1,
    "seed": 0
  }
}
```

Merge these keys into a normal native AIR configuration with an existing
`generic_air.cost_model.kind = "event-mechanism"` provider. Its implementation
coverage, units, parameter provenance and strict/estimate mode remain mandatory.
The refined host/device lifecycle also needs `host.visibility` coverage. Missing
parameters preserve unknown costs. Explicit assumed parameters produce analytical
estimates, not measured NPU latency.

```sh
taskflow-opt input.mlir --search --backend=air \
  --target-config=target.json --search-output=fresh-results
cmake --build BUILD --target check-taskflow-public-search
```

| Dimension | Existing object/action consumed |
|---|---|
| Source fission | `splitNativeOuterLoops` |
| Initialization distribution | `distributeNativeInitialization`; cloned tasks inherit origin bindings |
| M/N/K tiling | `setNativeAIRTileSizes`, native phase/region generation |
| Implementation and lanes | Native implementation catalogue and its source proof |
| Fusion, residency, traversal | `ProgramMappingChoice`, including scoped retained groups |
| Program partition | `ProgramPartitionPlan`; all other selections compose in each checked template |
| Task compute placement | Target compute resource and accessible memory |
| Resource count and shape | Public instance demand and geometry constraints |
| Data retention | Phase-local or retained version/copy availability |
| Exact region partition and implementation serialization | Whole-task adapter flags; native phase facts remain checked source facts |
| Buffer reuse | Same-storage/memory allocation aliasing and checked anti-dependencies |
| Transfer route | Current source copy and explicit ordered link path, including parallel links and output exports |
| Temporal priority | `ProgramTimingRequest.schedulePriority` and resource/calendar evaluation |
| Layout and dtype | Fixed by the current source/common checker, explicitly reported |

Resource count reserves capacity; it does not create replicas or imply a count-fold
speedup. Program partition searches serial template invocations and currently has
unknown host slicing/assembly/control costs. Such costs cannot win against a
known complete-program cost. General per-replica placements and arbitrary
layouts are not executable capabilities of this adapter. The whole-task adapter
can search its placement/copy/timing choices, but reports native phase tiling and
native grouping as unsupported by that adapter.

`program_search.tiles` can supply positive `m`, `n`, `k` candidate arrays. The
source/default tile is retained as the initial choice. Explicit
`program_search.dimensions` entries contain `id` and `choices`; using an automatic
dimension's ID replaces its finite domain. Distinct dimensions cannot write the
same property. Unknown, mistyped, orphaned or unconsumed decision fields are
rejected before solving. A native implementation binds the complete module;
per-task vector implementations require a separate consuming adapter.

The native initializer transform adds source tasks. A source-origin attribute
carries the selected placement, routes, priority and scoped task group to its
new initializer/compute tasks. This preserves coupled M/N/K and initialization
choices. Exact source versions, reduction ordering and access cells are rebuilt
by the original native planner. Rebinding preserves these instances while
materializing the selected copies, paths and placements; independent common
source/graph checking follows.

The source-instance work cap conservatively limits expansion for explicit tiles.
Reaching it means feasibility is unknown under the search budget; it is not a
proof that the program is illegal. Domain truncation and this work cap are
reported. Each choice exposure is recorded, which establishes proposal coverage;
it does not establish feasible coverage or enumeration of all combinations.

The time quota is shared by discovery, model construction, solver calls and
synchronous callbacks. Callbacks are not preemptible. Reports include solver,
callback and total time, peak process RSS, returned-after-deadline and callback
overshoot. A caller needing a hard process deadline should use an external
watchdog and retain `progress.json`, `baseline.json` and `incumbent.json`.

Each callback owns its MLIR context. The semantic cache retains metadata and evidence paths only. Complete candidate records are bounded by `max_full_evaluations`; improved incumbents, baseline and final best retain complete evidence. Candidate summaries and the full assignment trace remain available.

Outputs include `space.json` (actual binding domains), `progress.json`,
`search.json` (coverage, trace, limits and proof scope), candidate source and
complete graph evaluations, and the selected `best.json` / `best.mlir`.
**`best.mlir` alone is not the selected execution plan**: placements, copies,
route/slot decisions and analytical schedule are in `best.json`. This path does
not yet lower every selected graph decision to NPU runtime/AIE, so
`physical_realization_checked` stays false. A search cost improvement cannot be
reported as a device speedup without that separate realization and execution
check.

The optional solver-OFF build checks a valid baseline and returns `Unavailable`.
The public solver library itself has no MLIR dependency. The dedicated check
target covers solver ON/OFF behavior, proposal coverage, numerical/source graph
contracts, real reduction tiling, explicit routing and strict binding consumption.
