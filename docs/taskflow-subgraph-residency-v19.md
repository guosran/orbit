# Scoped subgraph residency and original search (v19)

This extends the existing public native mapping path. Actual Taskflow source is
analyzed once; source actions select a mapping; the checked mapping constructs
ProgramGraph regions, versions, allocations, copies and events. AIR consumes
that same plan and graph. Python invokes tools and checks receipts; it does not
construct plans or predict costs.

## Source decisions and resolved scopes

`ProgramMappingChoice.residencyGroups` contains `ProgramResidencyGroup` values:

```cpp
struct ProgramResidencyGroup {
  std::vector<size_t> tasks, retainedStorages;
};
```

Indices refer to analyzed source objects. Their canonical source form uses the
existing `orbit.mapping` dictionary:

```mlir
orbit.mapping = {
  data_organization = "retain-across-users",
  residency = "tiled-phase-local",
  grouping = "tile-major",
  residency_groups = [
    {tasks = [1 : i64, 2 : i64], retained_storages = [4 : i64]},
    {tasks = [4 : i64, 5 : i64], retained_storages = [6 : i64]}
  ],
  storage = -1 : i64, reuse_loop = -1 : i64, panel_loop = -1 : i64
}
```

Legacy `task_group` and `retained_storages` still describe a single group. They
cannot be mixed with the structured list. Empty new fields preserve the old
source spelling; the compatibility adapter does not rewrite historical sources.

The planner resolves each decision into `GenericResidencyScope`:

```cpp
struct GenericResidencyScope {
  size_t beginPhase, endPhase;
  std::vector<size_t> tasks, retainedStorages, boundaryExports;
  std::string inputReleasePolicy; // group-tail | last-reader
};
```

A scope contains matching-grid blocks in producer/consumer order. Independent
blocks with equal actual shapes can share an implementation stage. Prefix,
suffix and intervening tasks remain in the program; stages never cross a scope
boundary. `genericResidencyScope(plan, phase)` and
`genericLocallyRetained(plan, phase, storage)` are the shared lowering queries.
The structured source decision, resolved scopes, phase coverage and transfers
are checked again before accepting the plan.

Retention is local to a phase interval. It does not globally suppress a logical
storage's host allocation. `boundaryExports` is derived from source roles and
consumers, rather than supplied as an unchecked policy assertion:

- An intermediate consumed only inside its scope needs no host copy.
- An intermediate also consumed outside retains a host allocation and one
  export for each produced block. Internal consumers and that export read the
  same version of the local copy.
- A retained output always keeps its final host boundary, including when the
  same output version also has an internal consumer.

Allocation lifetimes cover all internal readers and required exports. The next
slot overwrite waits for all users. The graph retains the logical version and
copy relationships; it does not replace each semantic dependency with an
independently charged transfer.

## Current legality proof

A selected scope is a contiguous interval of the existing topological source
order. Multiple scopes must be ordered and task-disjoint; a retained storage
belongs to one scope. This is an explicit conservative proof boundary, not a
claim that every noncontiguous subgraph is illegal. Supporting a general
noncontiguous scope would require proving that collapsing its tasks preserves
a cycle-free source dependency order and all version/alias constraints.

The current implementation requires static independent pointwise grids with
matching source domains and exact retained producer/consumer regions. Every
retained storage has one source writer inside the scope and at least one
internal reader. Inputs cannot be replaced by a produced retained copy. Both
temporary and output roles are supported. Internal producer/consumer data must
be explicitly selected for retention. Pure whole-task order barriers inside a
scope, reductions, in-place updates, panel scopes, multiple task writers of one
storage and repeated external-input use across different grouped tasks are
rejected with reasons. The last case requires an explicit shared-input
implementation; the backend must not silently deduplicate multiple public DMA
requests.

The implementation computes actual per-stage storage pools and checks their
capacity. Whole-program residency may exceed a memory budget while each block
scope fits. This does not authorize overlap between configurations or a larger
physical memory than the target declares.

## Semantic and implementation order

`ProgramExecutionGroup.dependencies` records dependencies induced by source
tasks across groups. `implementationDependencies` records the selected
implementation's configuration serialization. A one-tile configuration sequence
can therefore serialize semantically independent groups without turning that
serialization into a source dependency. Public graph checking and event timing
honor both categories, while preserving their different meanings.

Inside a configured group, the graph has explicit core order and copy
producer/overwrite constraints. A source phase index is an instance witness,
not a DMA-completion barrier. Block exports occur after the last block compute;
the next input can refill after its old readers release the slot. For the original group-tail template, input refill waits for the prior block
core tail. The selected frontloaded template instead declares
`inputReleasePolicy = "last-reader"`: each input slot can refill after its actual
last reader, while outputs remain released at the block tail. This variant is
derived when an external input is first used by a later group member under the
all-inputs-at-entry capability, not from a task name or output role. It is stored
in the resolved plan, reconstructed by the mapping checker and independently
validated against actual buffer releases. Unknown policy strings are rejected by
the standalone communication API. The next core
entry waits for output-slot reuse. Group transitions inherit the complete
terminal frontier of the parent implementation group.

Backend-declared `acquireAllInputsAtGroupEntry` refines this order when the
implementation acquires all block input locks before its first body. All
inbound block transfers are then created before that first compute, which
explicitly waits for inputs consumed only by a later task as well. This is an
implementation constraint; it is not a fabricated source data edge.
`maxGroupInputs` and `maxGroupOutputs` are target capabilities, not universal
properties of the public graph. The installed backend still validates its own
actual stream and channel limits.

Resource and memory owners are looked up through the target architecture using
`ProgramTargetBinding.h`. Public planning does not parse a tile name or impose
an NPU coordinate, memory-kind spelling or 24 KiB ceiling. Backends can reject a
target that their lowering cannot realize.

The invocation lifecycle is also explicit. Native prepared execution uses
`MappingExecutionRegion.invocationLifecycle = "host-device"`; a host-only
implementation can declare `"host-only"`. Placement is the selected compute
resource's device identifier, not a test for an NPU spelling. Empty lifecycle
fields remain compatibility inputs for historical adapters. Source task reports
record the actual device, resource and architecture owner rather than invented
herd/core numbers.

Chosen-plan and unpriced execution identities exclude
`discovered_mapping_opportunities`: alternative actions are diagnostic metadata,
not facts of the selected implementation. Full reports retain those proposals.
`plan_identity_policy = "orbit.chosen-plan.v19.excludes-proposal-diagnostics"`
makes this revision visible. The price-attached plan ID still differs from the
unpriced execution fingerprint. Earlier frozen IDs remain unchanged in their
artifacts; re-export receipts must enumerate the revision and compare complete
chosen activities rather than claiming old literal IDs match.

## Original two-round search

`discoverGenericMappingChoices` finds scopes from actual source writer/reader
relations. It does not recognize task names, operator names or fixture indices.
It proposes explicit retained data for internal producer/consumer relations and
lets the planner reject unsupported scopes. From a source with a selected
scope, the same discovery composes an additional disjoint scope. Tiling remains
an existing source action, so either another scope or a tile action can be the
second action.

The existing `native_auto_candidates` configuration accepts
`max_residency_proposals` (default 8) and `max_residency_groups` (default 4), each
in `[1,64]`. These are search bounds. They do not relax legality or supply
hardware costs. Requested configurations, rejection reasons, action chains,
source hashes, graphs and priced plans stay in the original archive and
checkpoint/cache machinery.

`test/TaskflowGeneric/subgraph_search_v19.py` snapshots the real C++ tool before
running two rounds. It checks that scored round-one candidates determine the
second-round parents, exports all checked archived candidates without editing
them, and repeats strict interpretation with unknown scores preserved.
Assumption-scored search progress, correctness of the executable partial order,
real candidate ranking and speedup are separate claims.

New evidence belongs under companion
`results/taskflow/subgraph-execution-v19/`. Earlier v18 tools, measured packages,
models and failed observations remain unchanged. The initial v19 search is
`estimated-search-r1` (seven tasks) and `estimated-search-five-r1` (five tasks).
They are preserved before the output-role discovery extension. A final search
revision and same-graph receipts must identify the exact compiler used for that
extension; never amend an index after a compilation or protocol binds its hash.

Parameter sensitivity is orchestrated by companion
`taskflow/scripts/analyze_subgraph_sensitivity_v19.py`. It asks C++ to re-evaluate
the same archived source graphs under declared bandwidth, configuration and
compute/memory parameter changes, checks complete graph equality, and reports
winner and parent-set changes. It performs no fitting and makes no device
ranking claim.

## Final public evidence checkpoints

Paths are relative to companion `results/taskflow/subgraph-execution-v19/`.

| Evidence | Public scope |
|---|---|
| `estimated-search-r1/`, `estimated-search-five-r1/` | Preserved original search revisions used for the initial measured source packages; no retrospective index changes |
| `estimated-search-r3/audit.json` | Final post-guard original two-round search:29 archived candidates,26 assumption-priced; actual numeric-score parent selection and strict unknowns |
| `representatives-output-role-r3.json` | Final existing round-one output-role scope {3,4}, with explicit last-reader policy: local reader plus caller export; targeted mechanism case, not the ranked winner |
| `identity-revision-bridge-r2/summary.json` | All seven frozen sources retain exact chosen physical facts and predictions after explicitly enumerated identity/lifecycle/reporting revisions |
| `current-core-frozen-bridge-r1/replay-r1/result.json` | Independent strict path-by-path bridge of the same frozen sources; no unexplained differences |
| `build-core-r2/test-output/core.rrw2sC/results/` | Final40-group Core replay including target44, scoped-group45, synthetic timing83 and mechanism19 checks |
| `sensitivity-r1/summary.json`, `sensitivity-r2/summary.json` | Preserved pre-release-refinement parameter studies; their scope and earlier numerical parent ties remain explicit |
| `sensitivity-r3/summary.json` | Final29 immutable graphs, each re-evaluated under seven settings by C++;26 priced/3 unknown, same unique winner and exact first-round parent set in7/7 settings |
| `final-search-guard-bridge-r1/receipt.json` | All29 source artifacts, complete graphs, complete priced plans, scores, action chains and selections exactly match the fresh-generation archive; no normalization |

The final current Core snapshot is bound by the search and bridge receipts;
older source indices, compiler packages and observation protocols are unchanged.
The first extra identity auditor failed because it omitted the public graph's
duplicated ledger from lifecycle-policy normalization. That failure remains in
`identity-revision-bridge-r1/`; revision2 enumerates the nested ledger policy and
its digest explicitly, without weakening checks on events, copies, versions,
resources, transfers or predictions. Device timing and fresh clean-shell replay
results are reported by the coordinating main agent, separately from these CPU
facts.

The final sensitivity supplement `sensitivity-r2/parent-tie-audit.json` diagnoses
the exact second-parent changes: three sources tie at the cutoff to machine
precision, and the reference parents remain eligible within1e-12 ms in all seven
settings. This tolerance is only a numerical diagnostic; it neither changes the
original search selections nor substitutes for a device/model error budget.

Common-plan legality alone does not prove that lowering realizes every declared
entry constraint. The first output-role compilation exposed a late-consumer
input whose actual acquire occurred after the first body, despite the declared
all-inputs-at-entry capability. After frontloading was implemented, an additional
check found that the early input was actually released after its own reader,
rather than the unrelated last body. The final-AIE checker rejected both overly
strong graph contracts. The explicit last-reader template above addresses that
second mismatch without inventing arithmetic or a dummy read. Preserve these
failures and require an actual implementation receipt after the corrections;
do not infer device support from the common CPU checks or from earlier plans
whose inputs happened to be consumed by the first task.

The subsequent clean `fresh-generate-r2/` recreates source, original search,
eight selected packages and a frozen observation protocol with zero device
submissions. All eight complete native-plan/public-graph identity checks and
independent actual-AIE checks pass, including the corrected last-reader template.
The current score-selected two-group winner uses source groups {1,2,3} and
{5,6}; it is an actual archived candidate, not a runner-generated combination.
New observations belong to the coordinator's separate frozen protocol and must
retain their first-observation versus repeated-plan labels.

The final common compiler snapshot is SHA256
`8173d56eb7436f6fd13bc956eda8a38985d9e3b51a854593895dbc9839b828e6`.
Formal `estimated-search-r3` and the immutable pre-validation-guard generation
compiler produce exactly the same29 valid archived graphs and priced plans.
The selected late-input two-group winner has source identity
`4662641667b5df1f86dec3eb53dc92c7194897ae26ade80295c0925c77aa5a52`.
The new guards only reject malformed resolved spans, zero block widths and
unknown release policies; they do not change valid implementation facts.

`late-input-closure-r1/` retains the rejected overly strong graph and the
corrected actual-AIE proof. The coordinator's final frozen eight-plan protocol
has completed two NPU sessions with correct outputs, including the output-role
case and the new last-reader winner. Do not describe that result as validating
all fine candidate rankings: the measured comparison includes unresolved or
incorrect fine ordering, despite a positive algorithmic loop and a measured
benefit for the two-group example. Parameter sensitivity is an assumption study,
not a substitute for a calibrated error budget. The coordinator records exact
counts, distributions, pairwise uncertainty and cross-session limits separately.
