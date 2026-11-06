# Whole-program mechanism evaluation and NPU validation (v16)

ORBIT continues to optimize actual program structure, mapping, scheduling and
copies together. This stage adds a public resource-constrained time evaluator
and validates a finite scalar NPU implementation model. It does **not** establish
reliable candidate ordering: frozen heldout accuracy is 7/8 cells, MAPE 5.586689%;
the same parameters in a new matched process obtain 6/8, MAPE 9.759001%. All three
automatic-candidate pair differences are unresolved by the measurement gate.
The conditional two-round performance search is therefore not resumed.

## Production objects and call chain

```text
Actual Taskflow MLIR
  -> GenericAnalysis
  -> existing GenericMappingPlan and checked ProgramGraph
     -> collectGenericCostFeatures -> existing guarded GenericDirectAIRCost
     -> evaluateGenericProgramTiming -> evaluateProgramTiming
          -> existing scheduleStaticPlan resource calendars
     -> existing lowerGenericToAIR -> AIR -> AIE -> ELF -> bound-BO XRT runner
```

The same graph is checked by feature extraction and the native evaluator, and
consumed by lowering. The 16 study graphs are compared in full JSON against the
actual compiled plans. Three original C++ search candidates additionally retain
source-SHA candidate identities, action/parent chains and searched/compiled graph
equality. Python only prepares source fixtures/references, invokes tools, samples,
fits numerical parameters and independently audits evidence.

`ProgramTiming.h/.cpp` annotates the existing graph with exclusive implementation
costs, resource demands, exact estimates or explicit bounds/unknown reasons.
It does not introduce a second graph, mapping configuration or scheduler.
`GenericProgramTiming.h/.cpp` adapts the existing complete-invocation NPU provider;
`taskflow-opt --program-timing-output=FILE` exports its guarded timeline, native
features, formula contributions and **unidentified internal event** view.
The CLI adapter is native-NPU-specific; the public C++ API accepts other checked
ProgramGraphs with their own backend annotations.

See [the public evaluator contract](taskflow-program-timing-v16.md) for exclusive
coverage, unrefined invocation ownership, dependencies, physical buffer reuse,
capacity checks, critical paths, fixed resource shares and mathematical bounds.
Missing costs do not become zero-duration complete predictions. Unknown memory
capacities remain explicit; conservative composite lifetimes can reject feasible
shorter internal lifetimes. Dynamic bandwidth redistribution is not modeled.
Amoeba's original mapper, scheduler, network model and CGRA ML contracts are
unchanged. No AIR/AIE/XRT/Neura dynamic dependency is added to Core.

## Actual arithmetic and observation boundary

Optimized LLVM uses scalar FP32 operations. Target object inspection shows
software `__mulsf3` multiplication; additions use scalar lane insertion, `vadd.f`
and extraction/spills. No useful SIMD arithmetic throughput is assumed. GEMM
retains its output tile and initialized partial accumulation across K panels;
local C loads/stores do not become extra system-memory DMA.

The v15 all-zero trace remains unusable. The trace path sends stop then resets
PDI without a verified partial-buffer drain contract; simply awaiting a fixed
16 KiB trace BD could wait for a buffer that never fills. No speculative drain
patch or device timestamp is claimed. Bounded polling controls instead record
last-pending-query start and first-completed-query end on host perf_counter_ns.
These bracket **command-state visibility**, conditional on monotone state
reporting. They do not bracket physical DMA or provide a device clock.
Instrumentation and OS sleep scheduling change the observer; separate-session
polling contrasts do not causally identify device execution changes.

The priced boundary string remains `prepared-command-host-to-host` for
compatibility. Its precise scope here is **one prepared caller-bound BO runner
invocation**: all TO/FROM sync, submit, configuration, DMA, local work, locks,
stalls, wait/polling and remaining runner work. Context/BO preparation, caller
input reset, output poison, references, final output snapshot and evidence I/O
are excluded. Cold setup remains unpriced. Known nested API spans are reported
but not added again to the enclosing invocation.

## Frozen implementation formula

In nanoseconds, for the preregistered scalar implementation domain:

```
T = 232201.11122139467 * command_count
  + 49678.8769524652 * hardware_stage_count
  + 2.1452334001055914 * requested_DMA_bytes
  + 62.69654277482823 * scalar_f32_products
  + 1.4173927143738569 * scalar_f32_addsub
```

These are five terms of **one exclusively owned implementation composite**.
Neither the byte term nor multiply term is independently isolated physical
bandwidth or instruction latency. Multiplication correlates with GEMM local
loads/stores, reduction and loop work; the add control has different generated
loop/unroll structure. The heldout add chain fails, preserving that limitation.
The stage term is not an independently measured configuration/start/drain cost.
In the 16 audited packages, plan stage, owning child, configure and run counts
are equal; a separate wrapper sequence must not be counted as another child.

The existing `orbit.direct-air.cost.v1` linear-composite provider now allows up
to eight distinct terms, so scalar product/add implementation classes need no
parallel model configuration. The original specialized four-term
implementation-composite contract stays four terms. Parameters bind explicit
target/toolchain/runtime/observer/condition and native feature ranges.

Model SHA: `d14d6f222304dfdc30fe836a0db33c322a8f509440be68a80053c85d1a375b74`.
Eight training cells have three seeds, two warmups and nine hot calls per seed.
Five-term normalized design rank is 5, condition 6.985; training MAPE is 2.3361%.
A prior four-term common-scalar-operation fit failed on training (109.4997% MAPE)
and is retained. The revision used training and actual software-multiply evidence
before any new heldout timing. No heldout refit or rejection of raw outliers occurs.

The domain is a finite preregistered set of source fingerprints and shape/tile/
resource ranges, scalar FP32, one core per stage, eight prepared contexts,
20 us instrumented polling, fixed runner and artifact identities. Unseen-plan
validation is not arbitrary-program/source-family extrapolation. The resident
fanout control is outside those ranges and correctly returns unknown/OOD.

## Programs and search decision

The study includes copy/long-add controls, genuine tiled GEMM with cross-K
accumulation, and a complete X=2A -> B=X, C=X fanout with an ordering-only edge.
The frozen fanout16x96 plan predicts 556.621516 us; observed runner medians are
541.017 us and 761.012 us in the two matched processes. Its checked graph has
108 physical events, 72 transfers, 36 compute instances, 120 copies, 48 versions,
10 allocations and one real ordering-only edge. Intermediate X travels through
a system BO between three phase-local stages; no on-chip inter-stage reuse is
inferred from the logical fanout.

A separately measured same-source 8x64 resident control does retain X locally
for both consumers: no X BO or X DMA, 6144 versus 12288 requested bytes, 8192
versus 1024 B local payload. It changes loops, sync count and configuration too;
its observed median 352.861 versus 432.939 us is not isolated DMA savings.
This two-context mechanism control does not widen the frozen eight-context model.

The original C++ generator supplies two tilings and a B-panel-retention plan.
Their predictions are 4.698929, 4.646208 and 4.608018 ms, while primary observed
medians are 4.651916, 4.654666 and 4.640417 ms. Pairwise raw distributions overlap
at the optimization-relevant scale. Good total-time MAPE therefore does not
justify ranking or a two-round performance search.

The original v15 frozen parameters are also replayed unchanged under the
original nine-context, 200 us observer: all 297 outputs remain exact, but 0/9
time cells pass, MAPE 62.608429%. The earlier separately recalibrated 9/9 epoch
never becomes same-parameter cross-session evidence.

## Evidence and reproduction

The companion [complete results](../../orbit-artifact/results/taskflow/mechanism-program-v16/RESULTS.md)
include raw distributions, parameter source/ranges, full graph/physical binding,
time figures, independent review, failed attempts and clean-shell commands.

Build independent Core with the exact compiler recorded by LLVM:

```sh
env -i HOME=/home/shiran USER=shiran PATH=/home/shiran/bin:/usr/bin:/bin \
  LANG=C.UTF-8 LLVM_BUILD=/home/shiran/project/llvm-project/build \
  bash scripts/build-core.sh /absolute/new-core-build
```

The companion `taskflow/scripts/run_program_mechanisms_v16.sh --help` documents
explicit `cpu|hardware` replay of SHA-bound packages. Its CPU mode opens no
device; hardware mode owns the existing lock through the runner and independently
rechecks numerical outputs. A same-parameter replay is distinct from executing a
new calibration. Source and experiment evidence are intentionally not pushed.
