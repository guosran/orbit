# Public data organization and relative prediction: v17

The public task–data–communication graph now generates and realizes two real
organization changes through the original C++ search: immutable input retention
across tile consumers, and retention of one intermediate for multiple tasks.
The bounded physical study validates only the large tail-group intermediate
contrast; the overall registered search gate fails. General performance search
remains paused. There is no new total-time fit or replacement of Amoeba/CGRA.

## Public objects and calls

`ProgramMappingChoice`, `GenericAnalysis`, `GenericMappingPlan.communication`
(the existing `ProgramGraph`), memory architecture and `ProgramTiming` remain the
production objects. `runStructuredSearch` applies native actions to real MLIR;
`planGenericMapping` and `checkGenericMapping` derive/validate the copies, paths,
regions, versions, capacity and implementation events. Features, timing and AIR
consume the same checked graph, with full JSON equality checked against compiled
packages. `lowerGenericToAIR` still selects the existing resident or serial-tiled
implementation and carries checked body/plan/graph binding evidence to AIR/AIE.

The explicit `whole-program` action requires the conservative entire payload to
fit, removes source tile attributes and rejects incompatible explicit target
choices instead of falling back. Explicit `tiled-phase-local` retains materialized
phase boundaries. `readonly-panel` uses the existing affine proof to retain an
immutable region across disjoint consumers. Historical `phase-local` preserves
the old adapter-auto behavior and canonical identity. Ordinary tiling composes
with these actions. Details and clean build instructions are in
[the action/API document](taskflow-data-organization-actions-v17.md).

Dependencies describe source consumption/order. Transfer events arise from copy
placement and readiness, not one independent transfer/price per edge. In the
resident fanout one X version/copy has two consumers; no X system BO/import/export
is needed. The tiled plan supplies those consumers through distinct local copies
of a system materialization. Peak residency includes all simultaneously allocated
A/X/B/C buffers. Current lifetimes are conservative, not optimal storage reuse.
The 24 KiB target budget is a payload policy; compiler stack/other overhead is
separate and actual final linking must succeed.

The new `GenericProgramContrast` API collects two views of checked plans and
prices an exclusive implementation difference using the existing direct-AIR
configuration schema. It does not introduce another graph, planner or scheduler.
`taskflow-contrast-opt` accepts actual baseline/candidate sources, optionally
existing mapping configs, a frozen model and an actual target binding. It exports
facts, both plans, both public timing views and the contrast. Unknown internal
costs and absolute durations remain null with reasons. The ordinary absolute
provider rejects promotion of contrast-only parameters to a complete score.

## Realizations and timing boundary

Two exact-copy programs avoid software FP32 GEMM dominating the effect:

- Input: `C[i,j]=A[0,j]`. A full-width N panel makes both plans one configuration;
  readonly-panel replaces M input deliveries with one, retaining the same output,
  tile shape, local peak and ABI sync counts/ranges.
- Intermediate: `X=A; B=X; C=X`, shape8×N. Tiled stages materialize X in a system
  BO and reload twice; whole-program allocates A/X/B/C together and shares X
  locally. The mode changes configuration, loops, locks, buffers and ABI together.

Every selected package uses one shim/core path, scalar identity-layout FP32 and
one host command. Public transfers, coalesced runtime BDs and inferred cyclic
core transactions are different views of requested payload, not additive measured
DDR bytes. Separate input/output channels permit streaming but do not establish
observed overlap. No device clock conversion or pure-DMA timing is claimed.

The measured boundary is one prepared bound-BO runner call: all ABI TO sync,
output FROM sync, submit/wait/configuration, DMA/local work and runner gaps.
Context/BO preparation, caller input reset/poison, reference work and evidence IO
are excluded. Instrumented host polling is20 us. Each pair process prepares two
contexts and alternates AB/BA over three seeds, three warmups and fifteen hot
repeats. Final source, graph, plan, ELF, ABI and raw array hashes are retained.

## Frozen mechanism contrast

The existing `orbit.direct-air.cost.v1` schema with `kind: contrast-composite`
uses signed candidate-minus-baseline quantities. Its finite domain binds source
family, architecture, implementation, both plans' feature ranges and actual
runtime/toolchain/observer condition. Every priced feature requires both plan
ranges. Contributions own their JSON strings after the model leaves scope.

```text
A Δns = 0·Δinput_delivery_count + 0.04122218403522889·Δrequested_bytes
B Δns = 42713.81891252957·Δstages
      + 9442.85153664232·Δmaterialized_region_realizations
      + 0.35168024157806893·Δrequested_bytes
```

These jointly calibrated terms describe one changed implementation. They are
not physical bandwidth, independently isolated configuration, sync or core
latencies, and cannot be added to the v16 whole-call composite. A's zero delivery
coefficient is a nonnegative-fit boundary with unresolved effects, not free DMA.
A shared unknown absolute offset is a conditional hypothesis, not a consequence
of equal source arithmetic. Both full invocation times remain unknown.

Before any v17 timing, final-code inspection found that nonuniform tiled fanout
alternates full/tail group shapes and creates48 serial child configurations,
while compact shapes create3 and residency creates1. Registration r2 preserves
the earlier registration and adds two tail training sizes before device use.
Four A pairs identify two signed columns; six B pairs identify stage/mode/byte
columns. All scaled and leave-one-pair-out designs are full rank. The descriptive
training error allowances are45.604176 us and99.665523 us, not confidence bounds
or guaranteed prediction coverage. Some A heldout results exceed them.

## Independent results and search boundary

The model and four unseen predictions were frozen and independently checked
before reading heldout performance. Exact parameters were reused in a new-process
session without refitting. The two unseen fanout sizes both exercise48→1, not3→1.

| Heldout | Frozen Δ us | Primary paired median us | Same-parameter new-process us | Conclusion |
| --- | ---: | ---: | ---: | --- |
| Input M32,N768 | -3.926 | +60.039 | -4.950 | unresolved in both |
| Input M96,N1024 | -16.040 | +13.270 | +278.473 | unresolved in both |
| Intermediate E768 | -2020.233 | -2015.171 | -2000.192 | passes both |
| Intermediate E1152 | -2021.854 | -2012.262 | -2025.021 | passes both |

Resolution requires magnitude larger than both50 us and paired p90−p10, plus
same-sign seed medians. Resolved predictions require correct sign and error
<=max(25 us,30% of the absolute observed paired median). Both classes need two passed pairs
in both sessions. Input changes fail this gate; their effects are inconclusive.
Ordinary compact fanout3→1 training effects are also unresolved. Only the bounded
large tail-group relative change is validated; no general sorting claim follows.

The formal study preserves1944 real calls/3024 exact FP32 output arrays and exact
FP64-view checks, plus a separately counted108-call clean-shell NPU-entry smoke.
All28 compiled packages have full searched/lowered graph equality and final
requested-payload/ABI/code audits. Core36 and combined53 compiler groups, new
AIR10, original binding15, original lit35, Amoeba adapter17 and unchanged CGRA
8-contract checks pass. Broad suites used18 contrast assertions; the later
owning-string fix has19 focused assertions plus14 real frozen-model export checks.
The failed first freeze with invalid JSON is retained; it changed no fitted form
or parameters and was repaired before heldout execution.

Original two-round **unpriced** diagnostic enumeration retains31 legal candidates,
actual action chains and tile/retention combinations. Domain-guarded contrasts
are known only for the two measured direct actions;20 other pairs are explicitly
unknown and9 fail the same-implementation guard with preserved diagnostics.
Two compound candidates are also compiled and audited: tile384→retention produces
two broadcast configurations; whole→tile8 restores fanout materialization and X
BO. These compile-only examples have no new timing claim. Priced search plans
carry cost attachments, so their plan_id can differ from the physical compiled
plan: full graphs and all nonpricing plan fields match, and searched cost
features.execution_fingerprint equals the compiled physical plan_id. This bridge
is explicitly recorded, not treated as literal plan-ID equality.
All absolute scores are null and rankings empty. This is candidate/diagnostic
integration, not resumed performance search. Future precise input-effect or
smaller-stage studies need a new registration and fresh heldouts.

Full formulas, units, identification, raw distributions, conditional assumptions,
example copy/event/lifetime IDs, physical paths, freeze receipt and exact replay
commands are in the companion
[results report](../../orbit-artifact/results/taskflow/data-organization-v17/RESULTS.md).
The exact-package entry is `orbit-artifact/taskflow/scripts/run_data_organization_pair_v17.sh`
(`--mode cpu|npu`, `--pair`, new `--output`); both modes passed from an empty
environment. The NPU runner owns the existing device lock. No push is performed.
