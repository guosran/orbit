# PolyBench LARGE native search and NPU validation, v30

This study executes the official LARGE ATAX, rather than a scaled toy. It has
not established a hardware optimality oracle or the preregistered stable
CP-SAT acceleration of >1.05× in all three processes.
The frozen three-feature predictor collapses to a constant and cannot rank
the measured tilings. A finite model optimum must not be presented as a
hardware optimum.

Raw evidence: `../orbit-artifact/results/taskflow/polybench-large-v30-r1`
(relative to the repository root). Earlier v23–v29 evidence is unchanged.
`search-plan.json` is historical planning; the executed domains are the
versioned search configuration and emitted `space.json` files.

## Workload and numerical contract

Upstream is PolyBench/C 4.2.1-beta from
[the official release archive](https://downloads.sourceforge.net/project/polybench/polybench-c-4.2.1-beta.tar.gz).
Archive SHA256 is
`426519ee8443a5f2175de6a3e9328cda8917a5e33053d0e8f59855ab56d689a4`.
ATAX uses official `LARGE_DATASET`, M=1900, N=2100, with the upstream
`DATA_TYPE_IS_FLOAT` option. Default upstream double is not claimed supported.
The arrays occupy 15,984,400 logical FP32 bytes; this is not physical traffic.
`tmp=A*x; y=A^T*tmp` performs 7,980,000 multiplies and as many adds,
conventionally 15.96 million FLOPs.

The source adaptation separates initialization and the two matvec stages,
and schedules independent output elements while preserving each original
FP32 reduction order. Original C versus transformed source CPU checks cover
all 4,000 tmp/y values bit-for-bit. The original static upstream initializer
and kernel are called through an independently compiled wrapper; references
also include independently recomputed FP64 results. Actual NPU validation
reads the complete external y; device tmp is not read back.

## Complete call boundary and data split

Each timed call includes A/x reset, all four BO TO synchronizations, raw run
creation/binding/start/native `wait2()`, y FROM synchronization and full y
copy. Context/BO/reference preparation, poisoning, numerical validation and
evidence IO are outside. Five spans with common endpoints sum exactly to the
wall time. There is no polling fallback or CPU numerical fallback.

One exclusive composite owner prices the entire call. It includes actual
setup/body/DMA/teardown/completion effects once. Its coefficients do not
identify pure launch, device cycles or bandwidth. Separate body or wait
charges must not be added to the same observed composite interval.

Protocol `calibration-protocol-r1.json` was frozen before new measurements,
SHA `61218935e64df3dcff56df3932632d604a445166589e26db07f49b55d40bdbb7`.
Training tiles are 512×10, 400×10, 112×50, 112×10, 48×100. Each has three
fresh processes, one warmup and three hot calls: 15 process medians, 60 full
calls and 120 FP32/FP64 comparisons, independently audited passing. The old
512×10 four-call pilot is excluded from fitting and acceptance.

Heldout tiles are 448×10 and 56×100, with the same three-process protocol.
Their C++ predictions were frozen before observing their timing labels.
The preregistered absolute-error gate is <=20% for each tile's median of
process hot medians. Ranking requires a >=5% gap with the same sign in all
three processes; insufficient eligible pairs is inconclusive.

## Frozen predictor and finite model proof

Model SHA256:
`4f602f8253e57e5656db6c65100ae02d5b0cbdb30cff330eeaa1568b1e7297ca`.
The preregistered NNLS terms are command count, source phase count and DMA
bytes, collected by the existing C++ checked native feature provider.
Fifteen equally weighted process medians give coefficients in milliseconds
`[550.5231941999999, 0, 0]`. Independent guarded training-only replay reproduces
model, training-input and fit bytes exactly, without opening heldout timings.
Numerical design rank three does not imply physical mechanism identification.
The zero slopes make all one-command predictions identical. Absolute accuracy
can therefore pass while plan discrimination remains absent.

`search-calibrated-r2` uses the existing C++ `cp-sat-all` branch-and-check
master, with a shared 600-second ceiling. M tiles are 512/112/400/448, N tiles
10/50, K=1. Initializer distribution, scalar implementation, native mapping,
placement/copies/priority and no slot reuse are fixed. All eight assignments
are checked: five feasible, three explicit capacity rejections, zero unknown
costs or feasibility results. All five feasible tiles have compiled ELF
receipts. The finite cost-model result is OPTIMAL, with equal predictions
and the original all-zero 512×10 baseline assignment retained.

Total recorded time is 99.162577 seconds: 98.532644 seconds callbacks,
6.967245 milliseconds CP-SAT, 0.400540 milliseconds model build. Peak process RSS
is 13,304,040 KiB, approximately 12.69 GiB. These are whole search-process
resources, not solver-only memory. The compact independent audit is
`audit-search-calibrated-r1`.

## Baseline and actual backend

The primary reference is the explicit first executable NPU plan 512×10,
not an unchanged legacy ORBIT default or a retrospectively chosen slow plan.
The selected assignment is the same plan: its plan speedup is 1.00×.
Re-lowering the selected search source differs only in provenance IDs;
recompilation yields the identical ELF SHA256
`c55bff32e34674f8fb5c10bea1701d422ab4e604756bdd7f7bdbf576a9ab4b4a`.

Training process hot medians, milliseconds:

| Tile | Process 1 | Process 2 | Process 3 |
| --- | ---: | ---: | ---: |
| 512×10 baseline | 550.684 | 554.205 | 555.468 |
| 400×10 | 555.655 | 554.467 | 553.127 |
| 112×50 | 558.802 | 557.382 | 560.731 |
| 112×10 | 534.115 | 534.229 | 534.378 |
| 48×100 | 554.045 | 552.366 | 548.193 |

The measured training best 112×10 has per-process baseline ratios
1.0310/1.0374/1.0395×, median 1.0374×. It is not selected by the frozen
constant model, and it does not satisfy the declared >1.05× stable-acceleration
threshold. This post-measurement best is not a CP-selected speedup.
The chosen baseline has approximately 3.74% observed training selection
regret relative to that best. Formal heldout selection conclusions must be
reported separately.

`raw/backend-performance-audit-r1` contains complete actual core ELF symbols
and disassembly. Four serial stages reuse core (0,2); they are not four
concurrent cores. Main-loop FP32 products call software `__mulsf3`; additions
use `vadd.f` sequences, without `__addsf3`. Static coverage independently
reconstructs 3,990,000 products in each matvec. No device-cycle measurements
identify what fraction of the ~550 ms belongs to those helpers, DMA or launch.

An earlier official CPU kernel-only exploratory sample was ~3.72 ms. Its
timing boundary differs and it is not a formal speedup denominator.

## Scalability, retained failures and implementation checks

The original full-timeline export was killed before CP-SAT, at peak RSS
21,447,840–21,623,624 KiB. It expanded internal buffer-ordering relationships
quadratically. Direct feature export now avoids constructing a timeline.
For a validated single complete owner, internal ordering projects to self;
multi-owner checks remain. Phase-pool capacity is independently recomputed
by an allocation-phase sweep, including dormant templates. Unknown costs
remain unknown. Direct 512×10 feature export takes ~8.8 seconds and ~5.6 GiB;
large native planning/graph serialization still has substantial memory cost.

Retain these earlier lower/compiler failures: literal initializer ordering;
exact partition cell cap; local SRAM overflow; packet-ID width; repeat count
261 outside [0,255]; and the inapplicable forced channel-conversion attempt.
The successful 512×10 canonical AIR requires no channel conversion.

Search lineage originally changed source fingerprints. A first clone-function
fix also changed old plain-source printing; both failed search attempts are
retained. The final fix strips only `orbit.search_origin` on a cloned parent
module when needed, leaving live IR unchanged and retaining the original
print path otherwise. Actual LARGE and materialized search-source analysis
JSON reproduce the historical direct analysis byte-for-byte. No model
whitelist was widened. Final frozen search compiler SHA256 is
`0a82a36f4728f55772c1996fb490c759dc24edbea167f595161a61a02314b0bf`.

Final affected ON/OFF CPU regressions pass: timing 93 each, native composite
helper 13 each, native CLI 6 existing + 2 feature-export checks each,
invocation CLI 227/192, and the public-search suites including graph 42.
The latter checks metadata invariance, old direct-print compatibility and
real arithmetic changes. Original frozen tools, model and failures are kept.

## Ten-minute native domain experiment

`search-all-native-config-r2.json` enables the generated native choices, while
keeping the declared tile ranges and required initializer distribution. Its
emitted space has 29 dimensions, 15 nonsingleton dimensions, and 3,359,232
admitted tuples. The 12 compatibility tables allow all local tuples in this
particular domain; the count is independently checked without large-graph
enumeration. Source fission, scalar/mixed8 implementations, six mapping
choices, retention, reuse, routes and priorities are exposed to the same
master. Count/shape/resource/memory are singleton choices on this default
single-core adapter. CPU placement and actual multicore execution are not
part of this experiment. `generation_complete=true` means the declared
domain was generated without truncation, not that the full ORBIT abstraction
is implemented or searched.

The 600-second shared soft budget did not safely interrupt an in-flight
callback. Progress records 114 completed evaluations at 597.964510 seconds;
the next materialization created `candidate-95` and did not return before
the external 660-second guard terminated the process, exit 124. There is
no final `search.json`, full evaluation trace, solver termination status or
nonempty final time-v receipt. Only the incumbent and partial progress
survive. This is an incomplete search, not an exhausted domain or a CP-SAT
infeasibility result. Total CP time and separate feasible/unknown counts
cannot be reconstructed honestly from these partial files.

Linux VmHWM monitoring observed 13,117,668 KiB, about 12.51 GiB, before
termination. It is a sampled whole-process high-water reading, not a final
solver-exclusive resource receipt. The unfinished callback and the earlier
finite run show a materialization/checking bottleneck; the full-domain run
does not provide a complete CP-time breakdown. A read-only debugger attach
attempt was denied by host ptrace policy and is preserved as a failed
diagnostic.

Nonneutral native choices currently pass through a graph rebind that can
rebuild copies, allocation lifetimes, groups and lifecycle events. A capacity
rejection there establishes failure of that particular rebuilt graph. It
does not prove no native implementation exists for the same high-level
binding. Even an explicit route equivalent to the default can take this
path. Work-cap failures are feasibility unknown, and unmatched complete
costs remain unknown. These distinctions and the single-core scope are
recorded in `audit-all-native-scope-r1`; they prevent interpreting this run
as the requested full-abstraction optimum.

## Independent heldout board validation

`predictions-freeze-r1.json`, SHA256
`5b180bafe34a814a1be5f7e1a5587c3ebef8a86ea9ceec6622d8dab0716b332a`,
seals the model, C++ predictions, finite selection, interrupted broad-search
status and exact selected-image bridge before heldout execution. The frozen
calibration index keeps heldout `rounds=[]`; validation constructs the three
predeclared measurement paths independently, without changing that index.

All six heldout processes complete: 24 full calls and 48 complete-y
FP32/FP64 comparisons pass. Process hot medians in milliseconds:

| Heldout tile | Process 1 | Process 2 | Process 3 | Frozen C++ prediction |
| --- | ---: | ---: | ---: | ---: |
| 448×10 | 556.880 | 557.401 | 557.508 | 550.523 |
| 56×100 | 553.764 | 552.101 | 553.944 | 550.523 |

The independent [heldout audit](/home/shiran/project/orbit-artifact/results/taskflow/polybench-large-v30-r1/audit-heldout-r1/audit.json)
passes both absolute-error gates:
1.233983% for 448×10 and 0.585314% for 56×100. It checks the real C++ receipt,
model/feature/target equality and all frozen file identities, in addition to
numerics. These are separate from search optimality. The measured heldout
gap is only 0.656774%, leaving zero eligible pairs under the declared 5%
threshold: ranking is INCONCLUSIVE, not passed. A constant predictor still
cannot select a faster tiling. Neither heldout tile satisfies the declared
all-process >1.05× acceleration condition over the fixed NPU baseline.

Formal training plus heldout totals 84 complete calls and 168 FP32/FP64
comparisons. The excluded four-call pilot and the separate selected-image
replay are additional diagnostics, not added training or holdout labels.
The exact selected-image replay passes all four full calls and eight
FP32/FP64 comparisons, with a three-hot-call median of 549.696716 ms.
Its [independent audit](/home/shiran/project/orbit-artifact/results/taskflow/polybench-large-v30-r1/audit-selected-replay-r1/audit.json)
verifies the compiled identity, full outputs and timing boundary.
Including both diagnostics, all 92 device calls and 184 numerical comparisons
pass. The diagnostic median is not substituted as a new speedup denominator.
Its reused audit schema retains a historical pilot label; this is an explicit
selected-image replay excluded from both fitting and heldout acceptance.

## Remaining requirements for an optimality benchmark

Using official LARGE alone did not yield a faster CP-selected plan, despite
its full execution and substantial arithmetic. Stronger native implementations
and a predictor that discriminates plans still require validation.
Increasing solver time cannot make
a constant cost predictor distinguish plans, provide a missing executable
mapping, or interrupt an unbounded callback.

A useful ORBIT hardware reference still requires a predictor validated for
ranking different realized controls; native realization of the proposed
copy/buffer/resource/parallel choices; and bounded or cancellable graph
construction and checking. CPU/NPU region placement must then compose with
those native choices under the same complete call boundary. Until those
requirements are met, this result supplies a limited finite-model proof and
measured negative evidence, not the requested unrestricted hardware oracle.

No new commit or push was performed in this cycle. Unrelated
`docs/presentations/` edits are untouched.
