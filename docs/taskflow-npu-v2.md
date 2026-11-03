# NPU execution and search experiments, 2026-11-03

This iteration preserves `orbit.taskflow.plan.v1` and adds an independent
`orbit.npu.config.v1` target configuration:

```json
{"schema":"orbit.npu.config.v1","weights":"persistent","projection":"merge_siblings","accumulation":"host"}
```

Defaults are `staged`, `separate`, `host`, preserving the previous execution.
The source plan and public candidate identity never change when selecting a
target implementation. Configuration, runtime/kernel/schedule/profiler sources,
compiler versions, layouts and input/reference fingerprints enter evaluation
identity. Public transformations still determine the actual sliced operations.

## Persistent immutable-input BOs

`Executor` owns immutable copies of its input objects. Every immutable matmul
right-hand operand is padded/rounded into an actual XRT shared BO, including
transformed slices and merged weights. NPUChain binds this BO with
`bound_buffers={1:...}` and `static_indices={1}`. Its first use synchronizes it;
hot graph executions reuse it without restaging/synchronizing that weight.
This is NPU-accessible **system DRAM**, not whole weights in AIE tile SRAM.
Activations and outputs still use staging/readback; intermediate CPU placements
remain included in graph timing. Theoretical byte counts are not measured DMA
traffic or bandwidth.

The BO key includes executor identity, configuration identity, call site and
input generation. `update_inputs()` snapshots all new values, closes the old
chain contexts, invalidates bindings and increments the generation. It handles
new objects and explicit updates to the same object. Caller mutation cannot
silently change an owned snapshot: call `update_inputs()` to observe it. Changing
configuration in place is rejected; create a new executor. No process-global
weight-name-only cache is used. BOs remain owned until close/update.

## Sibling projection implementation

`merge_siblings` recognizes two matmuls with the same immutable left operand,
M/K regions and different destination buffers; it does not dispatch on a
workload name. Their right operands are concatenated along N, and one GEMM
produces two slices. The first result is applied at its original operation,
and the second at its original operation, preserving intervening initialization
and accumulation order. The target plan records both public operations and
output slice boundaries. This reduces gate/up GEMMs from two to one for a base
FFN. SwiGLU is still CPU `SiLU(G)*U`; this is not device activation fusion.

## Sequential K: demonstrated limits

`accumulation=device` currently requires `projection=separate` and consecutive
same-input/same-output/same-region K partials inside one task. Each semantic
chunk is separately padded to physical K=64 and packed in order. A device
program initializes one FP32 accumulator, traverses an explicit semantic chunk
loop and physical inner K loop, and returns the final result once. It does not
ignore the semantic K block sizes. Its continuous FP32 MAC order differs from
independently rounded partials followed by CPU addition; it is evaluated against
the unchanged full-output tolerance and independent reference.

Small nondivisible K=7 cases work (physical inner loop trip count one). The
real-size search candidate K=5504 fails the unchanged AIR transform schedule:
`loc("-":132:43): no iteration argument is packed and unpacked in the loop`.
Both M=5 and M=128 failures and unscored results are retained. No real-size
latency is claimed for that failed device plan. The alternative two-launch
shared-C accumulation MRE fails at `expected a single payload op`; its source,
IR capture and diagnostics are retained. Installed wheels/toolchains were not
modified. No device fallback silently executes a matmul on CPU.

## Timing and evidence

All hardware runs are owned by the NPU worker and serialized by `/tmp/npu.lock`
inside the adapter. Never wrap its CLI in another lock. Full-graph hot timing
includes CPU nodes, conversions of intermediates, actual XRT submissions/waits,
necessary staging/sync, accumulation and final output copy. Compilation,
initialization/weight preparation and first execution are separate. At least
three samples follow two excluded executions; final paired comparisons use
seven samples per configuration in forward and reverse order. Clock frequencies
are not locked; small differences are not reliable acceleration claims.

The final profiling revision times NPUChain `_open_runner` only after its ELF
has explicitly compiled. `chain_load_seconds` includes real device/ELF/context/
kernel initialization and any initial context eviction. `staged_load_seconds`
uses the prior generated-launcher timer; `load_seconds` is their sum. BO
allocation/initial weight synchronization remains in
`load_and_first_execution_seconds`. Hot context reopen costs, if any, remain
inside the graph wall sample. Earlier exploratory `formal`, `full-forward` and
`*-final` evidence predates this timing fix: its zero chain `load_seconds` means
uncovered by the old timer, not zero actual load. Final evidence uses the
`final-timing` label and its exact copied adapter sources.

Every run stores authenticated Taskflow, public source plan, target placement,
source fingerprints, copied source, AIR/native artifacts, all output arrays,
full-output numerical checks and all raw hot wall samples. Failures carry no
numeric score. Results from previous iterations remain untouched.

## Reproduction

From a clean shell, prepare/rebuild the core and search as described in the
main search report, then:

```bash
cd /home/shiran/project/orbit
source ../aie-toy/scripts/env.sh
# Choose a NEW evidence root; benchmarks refuse existing result.json files.
RUN_DIR=../aie-toy/results/orbit-search-reproduce-$(date +%Y%m%d-%H%M%S)
python scripts/taskflow_search_experiment.py prepare
python scripts/taskflow_search_experiment.py search --backend npu --case ffn_m5 --label reproduce --rounds 2 --budget 9 --beam 2
python scripts/taskflow_search_experiment.py search --backend npu --case ffn_m128 --label reproduce --rounds 2 --budget 9 --beam 2
python scripts/taskflow_npu_bench.py --source-dir build/search-npu-v2/data --output-dir "$RUN_DIR/forward" --cases ffn_m5,ffn_m128 --seeds 0 --configs baseline,persistent,merge,combined --iterations 7
python scripts/taskflow_npu_bench.py --source-dir build/search-npu-v2/data --output-dir "$RUN_DIR/reverse" --cases ffn_m128,ffn_m5 --seeds 0 --configs combined,merge,persistent,baseline --iterations 7
python scripts/taskflow_npu_search_confirm.py --case ffn_m5 --label reproduce --output-dir "$RUN_DIR/confirm-m5"
python scripts/taskflow_npu_search_confirm.py --case ffn_m128 --label reproduce --output-dir "$RUN_DIR/confirm-m128"
python scripts/taskflow_npu_lifetime.py --output-dir "$RUN_DIR/lifetime"
python scripts/taskflow_npu_lifetime.py --device --output-dir "$RUN_DIR/device-lifetime"
# Expected compiler failure for real-size nested device K; retain diagnostics.
python scripts/taskflow_npu_device_k.py --strategy nested-loop --chunk-k 5504 --columns 2048 --output-dir "$RUN_DIR/device-k-5504-mre"
```

`taskflow_npu_k_compare.py` selects the first K child of the baseline from the
original search's `formal` checkpoint, retaining its provenance; it compares
host/device accumulation on precisely that materialized plan. It returns
nonzero when the device candidate fails compilation. Reproduce the standalone
MRE above to inspect the shape limitation without running a whole search.

## Final full-size observations

Final same-input comparisons below use seven samples per point in each of two
opposite orders. Values are each batch's whole-graph median milliseconds;
both numbers are retained rather than treating one ordering as definitive.
All outputs pass the original `.015/.06`, RMS 3%, peak-scaled 8% thresholds.

| Target implementation | M=5 forward / reverse ms | M=128 forward / reverse ms |
|---|---:|---:|
| Staged, separate projections | 21.665 / 19.681 | 37.832 / 37.753 |
| Persistent BOs, separate | 10.778 / 10.498 | 28.834 / 28.169 |
| Staged, merged siblings | 19.098 / 20.077 | 36.236 / 37.263 |
| Persistent BOs + merged siblings | 10.177 / 10.434 | 28.113 / 27.400 |

Weight BO reuse gives a large reproducible reduction in both sizes. Merge alone
is inconsistent at M=5 (including a reversed-order regression) and small at
M=128; a strong independent merge acceleration claim is unwarranted. Keeping
CPU SwiGLU means activation readback, conversion and down-projection staging
remain. Extra M partitions duplicate physical 128-row work and submissions;
N/K partitions may add launches and packing despite smaller individual GEMMs.

Search-selected candidates were independently confirmed under precisely the
same persistent/merged target as their source search:

| Case | Baseline then best, ms | Best then baseline, ms | Interpretation |
|---|---:|---:|---|
| M=5 | 10.283 → 9.307 | 9.609 → 9.606 | Apparent initial improvement does not persist in reverse order |
| M=128 | 27.163 → 27.373 | 27.259 → 26.706 | Archive best is identity; these repeat the same baseline and show timing variation |

For M=128 the final search retained the untransformed baseline: no measured
structural candidate beat it. The confirmation labels `baseline` and `best`
therefore refer to the same candidate identity, not two implementations. Their
differences measure run variation and cannot be called a structural regression.
Thus the search demonstrates evaluation-driven selection, not a guaranteed
performance improvement. The archive keeps both regressions and unmeasured
budget entries. Primary evidence is under
`aie-toy/results/orbit-search-v2/{full-forward-final-timing,full-reverse-final-timing,winner-confirm-m5-final-timing,winner-confirm-m128-final-timing}`;
search trajectories are `orbit/build/search-npu-v2/search/npu-*-final-timing`.
The early `formal` and earlier `*-final` directories are exploratory records,
not part of this final table.

The final small suite passes 24/24 persistent/combined runs and 12/12 device-K
runs (GEMM, GEMV, GEMM→ReLU and FFN; three seeds each). The host-accumulation
lifetime suite passes 27 checks; the device-K suite passes 18. Both poison output
buffers with NaNs before execution and exercise repeated A, B, A again, caller
mutation without update, explicit in-place update, repeated updated execution,
zero reset and rejected configuration mutation. Evidence is in
`small-final-timing`, `device-small-final-timing`, `lifetime-final-timing` and
`device-lifetime-final-timing` under the same result root.

The persistent comparison necessarily uses NPUChain's binding-capable runner,
so the observed improvement includes transport/runtime overhead changes. It is
not an isolated measurement of DMA savings. `persistent_weight_initial_sync_*`
is an API-level unique BO-generation accounting estimate, not a hardware sync
counter; runtime context eviction can require additional synchronization.
