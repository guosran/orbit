# Explicit data organization actions (v17)

The existing `ProgramMappingChoice` now distinguishes an explicit execution
realization from the historical adapter default. No program graph, mapping
configuration, search driver or AIR path is duplicated.

| Choice `residency` | Data policy | Grouping | Realization |
| --- | --- | --- | --- |
| `phase-local` | `phase-local` | `task-shape-stages` | Historical adapter-auto: resident if it fits and no tiling/vector choice requests the tiled path |
| `tiled-phase-local` | `phase-local` | `task-shape-stages` | Explicit `serial-tiled`; intermediate results cross the existing BO boundaries |
| `readonly-panel` | `retain-across-users` | `readonly-panel-scope` | Existing source-proven immutable input panel retained across independent tile consumers |
| `whole-program` | `retain-across-users` | `whole-program` | Explicit `single-herd-resident`; shared inputs/intermediates stay in checked local copies across program consumers |

Only `readonly-panel` takes storage/reuse/panel loop indices. The whole-program
choice conservatively sums all declared storage payloads, rather than performing
fine-grained lifetime allocation. That complete working set must fit the target budget and
currently uses the scalar resident implementation. It rejects an incompatible
explicit target `tile_sizes`/`tiled` request or vector implementation and never
silently falls back to tiling after a capacity failure.

Choices are materialized in the existing `orbit.mapping` source attribute.
Selecting whole-program removes the previous source tile attribute: this is an
explicit change of realization by the mapping action. A subsequent ordinary tile
action changes whole-program to tiled-phase-local and sets actual tile extents.
The old phase-local/default spelling still normalizes to no mapping attribute,
preserving its historical identity. Contradictory raw policy/realization/grouping
or loop-scope annotations fail closed.

`discoverGenericMappingChoices` proposes both explicit whole-program and tiled
realizations, as well as the existing source-derived readonly panels.
`runStructuredSearch` applies the original automatic action generation and beam
machinery; it emits `mapping_whole_program`, `mapping_tiled_phase_local` and the
existing `mapping_retain-across-users_storage_*` actions. Automatic proposals are
checked by the same planner before consuming the finite evaluation budget.
Infeasible proposal source and diagnostics remain in `automatic-proposals.json`.
Action chains, candidate source hashes, checked plans and graph IDs are retained.
Ordinary two-round actions compose tiling with input retention or switch between
intermediate materialization and local residency. These are executable mapping
choices, not runner labels or separately constructed Python plans.

The input proof uses actual affine SSA accesses: immutable input, independence
from the reuse induction variable, and disjoint output slices under both selected
outer axes. Arithmetic operation names are not part of this proof. Consequently
`C[i,j] = A[0,j]` is supported with the existing readonly-panel mechanism, without
a GEMM or multiplication special case. On the 4-by-16 compiler fixture and tile
1-by-16, repeated loading requests 512 bytes; sharing one local input copy across
four source consumers requests 320 bytes. Both use one hardware stage.

Panel scope still coincides with a backend execution group. With N=1024 and
N-tile=256 the current implementation realizes four panel stages, whereas the
repeated-load plan may have one stage. Such a pair changes configuration and
buffer organization as well as requested bytes. A full-width N-tile=1024 or
N=512/N-tile=512 retains the input within one stage and provides the cleaner
same-stage control. Requested DMA bytes are compiler descriptor facts, not
measured DDR traffic or independent elapsed-time predictions.

The checked `ProgramGraph` carries logical regions/versions, shared copies,
consumers, lifetimes, resources and transport events. The backend continues to
consume only its existing two strategy values. Whole-program residency gives
one intermediate local copy multiple real consumers; the tiled realization
creates the corresponding system copies and reloads. Any speed or reliable
ordering claim requires new frozen, matched-condition physical evidence; action
availability alone does not establish it.

## Checked call chain and backend receipts

The production path is the existing `runStructuredSearch` over actual Taskflow
MLIR. A selected source action is reanalyzed by `GenericAnalysis`, planned and
checked by `planGenericMapping`/`checkGenericMapping`, and represented by
`GenericMappingPlan.communication`, the public `ProgramGraph`. Feature collection
and cost evaluation recheck that same plan. `lowerGenericToAIR` consumes that
checked realization. AIR carries the source body fingerprint, native plan ID and
ProgramGraph ID; its checker also inspects actual transfers, allocations,
placement and ABI forwarding instead of accepting identity tags alone.

The backend receipt and compiled-package audit connect the source action/plan
identity to actual AIR/AIE implementation: retained input/core transactions,
intermediate system BO versus local copies, runtime descriptor payload, child
configuration ownership, ELF and ABI. Requested bytes, runtime BDs and core
transactions are separate views; their counts are not additive physical DDR
traffic. Receipt validity and numerical correctness do not establish independent
DMA durations, overlap or a timing model's accuracy.

## Conditional implementation contrasts

`GenericProgramContrast.h/.cpp` reads two existing checked plans and derives
signed quantities in candidate-minus-baseline order. Its supported finite changes
are one-stage immutable-input retention with fixed ABI and output delivery, and
materialized intermediate regions changing to explicit whole-program residency.
The latter owns the changed configuration, ABI sync, buffering, locks and local
loop behavior together. It is not an isolated configuration-latency parameter.
The public graph remains the source of copies, versions, consumers and events.

A frozen `orbit.direct-air.cost.v1` model with `kind: contrast-composite` can price
one exclusive changed-implementation composite:

```
conditional_delta_ns = sum_j coefficient_ns[j] *
                       (candidate_quantity[j] - baseline_quantity[j]).
```

The provider requires actual runtime/toolchain/condition identity, source-family,
architecture, implementation and both plans' finite feature ranges. It returns
signed contributions and a declared training-derived error bound. The common
unknown absolute offset is a hypothesis confined to that domain; unseen-plan and
same-parameter cross-session validation must assess it. Same arithmetic does not
prove unchanged generated local-code time. Physical link bandwidth, core-only
latency, precise internal critical intervals and measured overlap remain unknown.

`taskflow-contrast-opt baseline.mlir --candidate=candidate.mlir` rebuilds both
plans from their actual source attributes and emits facts, both plans, both
absolute-timing views and the contrast. `--baseline-config`, `--candidate-config`,
`--cost-model` and `--target-binding` reuse the existing target/model mechanism.
Without a calibrated, matching row the difference remains unknown with a reason.
The ordinary absolute provider recognizes contrast-composite and explicitly
returns uncalibrated: it never converts a signed difference into an absolute
invocation duration or original search score. Automatic candidate generation is
available independently of performance ranking. A bounded performance-search
demonstration still requires an independently passed ranking/readiness gate.

## Clean-shell compiler reproduction

Build with the exact compiler recorded by the LLVM build (the existing build
entry reads it from the LLVM CMake cache):

```sh
cd /home/shiran/project/orbit
env -i HOME=/home/shiran USER=shiran PATH=/home/shiran/bin:/usr/bin:/bin \
  LANG=C.UTF-8 LLVM_BUILD=/home/shiran/project/llvm-project/build \
  bash scripts/build-core.sh /absolute/new-core-build
```

Replay the public semantics, automatic candidate/source identity, contrast,
resource-time and original CPU compiler regressions into a new directory:

```sh
env -i HOME=/home/shiran USER=shiran PATH=/usr/bin:/bin LANG=C.UTF-8 \
  bash /home/shiran/project/orbit/scripts/run-data-organization-v17-cpu.sh \
  /absolute/core-build /absolute/new-cpu-results
```

The optional third argument selects an explicit existing Amoeba optimizer for
original-driver regressions; the optional fourth selects an AIR-enabled Taskflow
optimizer for compiler-only backend checks. These modes do not submit NPU work.
The separate public Amoeba/model golden entry is
`../orbit-artifact/taskflow/scripts/run_taskflow_program_graph_amoeba_v12.sh cpu`;
its tool/model paths must identify the preserved backend under test.


The final compiler-only checkpoint is recorded in companion
`results/taskflow/data-organization-v17/clean-shell-core-final-r1/`: all36 Core
regression groups pass from the clean-shell entry, including18 contrast-provider
assertions,50 mapping assertions and44 public timing assertions. Original
Amoeba/public-adapter17-check and CGRA8-contract results are in
`amoeba-contract-final-r1/`, explicitly using the preserved v13 backend with the
current Core; model scores and shortlist remain unchanged. These checks perform
no hardware submissions and do not train a model. Frozen physical-study
prediction errors, same-model session stability and ranking readiness must be
reported separately from their retained raw observations.

For combined compiler-only AIR and original-native-backend regressions, source
the existing toolchain environment and retain the LLVM tool directory (including
`not`) on PATH. Select the preserved backend explicitly; this does not rebuild or
modify the SHA-bound production AIR optimizer:

```sh
env -i HOME=/home/shiran USER=shiran PATH=/usr/bin:/bin LANG=C.UTF-8 bash -c '
  source /home/shiran/project/aie-toy/scripts/env.sh
  export PATH=/home/shiran/project/llvm-project/build/bin:$PATH
  bash /home/shiran/project/orbit/scripts/run-data-organization-v17-cpu.sh \
    /absolute/core-build /absolute/new-combined-results \
    /absolute/preserved-amoeba-optimizer /absolute/air-taskflow-optimizer
'
```


The final combined compiler run in
`data-organization-v17/final-backend-validation-r1/combined/` passes53 groups
(36 Core, one original native-search group,16 AIR groups). The same evidence root
also records10 new data-organization AIR assertions,15 original graph/AIR binding
assertions, and the preserved35 original lit cases, all passing. Exact commands
are in `commands.json`; `environment-entry.json` identifies the sourced toolchain
environment and LLVM PATH addition. Production Core/AIR/Amoeba tool SHA256 values
match before and after; these checks perform no builds or device submissions.
The broad suites exercised18 contrast assertions before the subsequent focused
JSON string-lifetime regression; repaired-provider verification and frozen-model
query receipts are separate evidence, not retroactively included in those counts.

## Unpriced two-round diagnostic on formal heldout sources

`two-round-diagnostic-r1/` runs the original C++ generator on the already
registered broadcast M32/N768 and fanout E768 baseline sources, with their
materialized attributes and the unchanged `generated-pairs-r1/config.json`.
Each run allows two rounds,24 candidate evaluations and beam width4. The archives
contain16 and15 legal candidates respectively. All absolute scores remain null
and both performance rankings are empty. No conditional contrast is fed back as
an absolute search score, and no new hardware measurement is made.

Every legal materialized source is independently replanned and its full public
graph checked against the search artifact. The frozen contrast CLI is invoked
for all31 candidates: two conditional differences match the existing formal
pair predictions exactly;20 return explicit unknown coverage; nine reject the
same-implementation guard before producing a cost artifact. Rejected queries
retain command, exit status and stderr and receive no fabricated numeric cost.

Four official baseline/candidate source SHAs appear in the archives. Their full
compiled ProgramGraphs also match, establishing eight links to the existing
heldout and cross-session raw-sample evidence. Other plans remain unmeasured.
`TABLE.md`, `table.csv` and `rows.json` distinguish coverage, raw evidence and
unknowns; any displayed runner-wall medians are marginal hot medians from those
existing sessions, not paired differences or new observations. Two structural
combinations are selected for compile-only receipts: tiling followed by input
retention, and whole-program residency followed by a tile action that changes it
back to the real tiled realization. Selection uses actions and scalar legality,
not predicted or observed timing.

The artifact-side replay entry is
`taskflow/scripts/diagnose_data_candidates_v17.py --study-root STUDY --output NEW`.
The exact source executed for the retained first diagnostic is also saved in its
`runner.py`. Frozen model, parameters, target, config and compiler-tool bytes are
checked unchanged before/after. This diagnostic establishes an honest candidate
and coverage loop; it does not replace the frozen study's prediction/ranking
acceptance or constitute a performance-search demonstration.

Both structural representatives have now passed compile-only implementation and
code audits in `two-round-compiled-representatives-r2/`; the diagnostic's
`representative-compile-links.json` links their exact source/actions/full graphs.
Broadcast tiling to N384 followed by retention gives two configurations, four
runtime BDs,66 core transactions and101376 requested bytes with3072 local payload
bytes. Fanout whole-program followed by tile1x8 really returns to tiled-phase-local:
three configurations, six runtime BDs,576 core transactions,18432 requested bytes
and64 local payload bytes, with X back in the system ABI. These compiler facts
have no new duration measurements. The frozen contrast correctly leaves their
changed structural coverage unknown.

Native price-attached plan IDs and price-free physical plan IDs are recorded
separately. For the two representatives and four formally measured archive
sources, all native plan fields other than `plan_id`, `direct_air_cost` and
`cost_provenance` match the compiled package, while the search cost feature's
`execution_fingerprint` equals its compiled physical plan ID. Full public graphs
also match. The initial excessive literal-ID assertion remains as a preserved
failed attempt; it is not converted into a claim that different IDs were equal.
A final unfiltered lit discovery after excluding pure input fixtures discovers
and passes exactly35 original tests (`final-lit-discovery-r1/`).
