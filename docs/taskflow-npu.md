# Restricted Taskflow NPU adapter

`scripts/taskflow_npu.py` consumes **only** `orbit.taskflow.plan.v1`.
It invokes the independently built `taskflow-opt` in a subprocess before
lowering, so missing dependencies, invalid initialization/reduction, and
incomplete regions cannot be hidden by serial host execution. Set
`ORBIT_TASKFLOW_OPT` to override the verifier binary. Missing verifier means
`unsupported`, not unchecked execution.

The selected route is **Taskflow plan → a parameterized Triton recipe → AIR →
AIE ELF → XRT** on `npu2/AIE2P`. This uses installed compatible wheels and a
mature wheel-compatible transform schedule from the local Triton-XDNA working
copy. It is an explicitly identified Triton frontend fallback, not a new direct
Taskflow-to-AIR compiler. No workload name appears in adapter dispatch.

Each materialized matmul operation selects its actual A/B slices, M/N/K
extents, padding, device executions, and output accumulation. Backend physical
tiles (128×256, K step 64) differ from semantic Taskflow tiles. Nondivisible
boundaries and M=1 are zero-padded and cropped. Dependencies are completed in
topological order. Explicit sequential K partials execute on NPU and accumulate
on CPU in FP32. Fill, ReLU, and standard `SiLU(G)*U` are explicit CPU nodes.
Fusion/fission is consumed as the actual ordered operations inside the new
Taskflow tasks; **ReLU is not fused into the device kernel**. Experimental
physical epilogues failed Peano vector compare legalization, and those failures
are retained under `aie-toy/results/orbit-taskflow/smoke-relu*.log`.

The public buffers are FP32. GEMM operands round to BF16, then the installed
AIE2P lowering uses BFP16 EBS8 MAC instructions with FP32 accumulation/output.
This is an approximate target implementation. Acceptance checks all elements
against independent whole-matrix float64 references, finite values and shapes,
with fixed `atol=0.015, rtol=0.06`, **plus relative RMS ≤ 3% and maximum absolute
error ≤ 8% of the reference peak**. All-zero references require exact zero.
The scale-independent tests prevent small FFN outputs or zero results from
passing on absolute tolerance alone. Thresholds were fixed before the formal
batch and were not loosened after numerical failures.

```bash
cd /home/shiran/project/aie-toy
source scripts/env.sh
python /home/shiran/project/orbit/scripts/taskflow_npu.py \
  --plan /home/shiran/project/orbit/build/backend-decoupling/gemm/mn/plan.json \
  --inputs /home/shiran/project/orbit/build/backend-decoupling/gemm/inputs-0.npz \
  --reference /home/shiran/project/orbit/build/backend-decoupling/gemm/reference-0.npz \
  --output-dir results/orbit-taskflow/reproduce-gemm-mn --iterations 3
```

For the complete prepared suite:

```bash
cd /home/shiran/project/orbit
source ../aie-toy/scripts/env.sh
python scripts/taskflow_experiment.py npu
```

The adapter owns `/tmp/npu.lock`; never wrap these commands in another lock.
Compilation uses the driver's `compile_only` mode (Triton `warmup` alone does
not compile the deferred AIR/AIE stage). Generated launcher instrumentation
reports device open/ELF/context/kernel loading with a host steady clock. It
changes no installed files, guards exact template locations, and participates
in the launcher's compilation cache. BO allocation/staging and first execution
remain separately visible in `load_and_first_execution_seconds`; immutable
padding/conversion has `first_weight_preparation_seconds`. Hot samples measure
one complete synchronized graph, including all CPU placements, DMA staging,
XRT waits and readback. They are not sums of per-operation times. No hardware
cycle-counter timing is claimed. The desktop is shared and clocks are unpinned.

Per-run artifacts contain the source plan, authenticated Taskflow, verifier
hash/log, target placements, dependency trace, inputs, independent reference,
every hot output, raw wall samples, compiler/cache artifacts, and source,
layout/configuration/toolchain fingerprints. Statuses distinguish unsupported,
compilation failure, runtime failure, numerical failure, and passed. The first
formal batch before load instrumentation is preserved in each candidate's
`baseline-uninstrumented/` directory.

Supported: static positive two-dimensional FP32 storage, GEMM output slicing,
sequential K accumulation, ReLU, standard SwiGLU, explicit ordered task graphs.
Unsupported: dynamic shapes, parallel K reductions, arbitrary dialect bodies,
integer kernels, device SwiGLU/ReLU epilogue, asynchronous device overlap,
resident whole-FFN fusion, and guaranteed bit-exact FP32 semantics.

## Recorded acceptance batch

The final formal batch has 16 candidates, 44 candidate/seed runs, and 132 hot
whole-graph measurements, all passing. Small cases use seeds 0, 1, 2; full
D=2048/I=11008 FFNs use seed 0 at M=5 and M=128. Synthetic signed dense weights
are asymmetric and every seventeenth element is zero. Input scales and the
independent whole-matrix float64 reference are in `scripts/taskflow_common.py`.
No pretrained weights or polynomial activation are used.

| Case | Base median ms | M/N partition median ms | Sequential K median ms |
|---|---:|---:|---:|
| GEMM 5×23 @ 23×19 | 0.712 | 2.108 | 2.318 |
| GEMV 1×23 @ 23×19 | 0.518 | 0.894 | 1.991 |
| GEMM → CPU ReLU | 0.543 | 1.738 | 2.283 |
| Standard FFN M5 D16 I24 | 1.346 | 6.316 | 3.890 |

The fused/fissioned Taskflow two-op candidates measured 0.590/0.577 ms; their
ReLU placement remains CPU. Full FFN M5/M128 measured 21.609/39.805 ms, maximum
absolute errors 0.000961/0.001123 and relative RMS errors 1.73%/1.72%. These are
shared-machine host observations, not speedup claims. Every candidate and all
samples remain in `aie-toy/results/orbit-taskflow/summary.json`, with links to
the complete run directories under `orbit/build/backend-decoupling`.

`compile_seconds` is elapsed compiler **or cache lookup** time. It is not a
claim of a cold compile: ELF/JIT artifacts may already exist, and in-process
reuse is explicitly marked. `identity.json` contains the profiling source hash
as well as runtime/kernel/schedule hashes, all package versions, full semantic
plan, target configuration and layout. `semantic_validation` stores the exact
verifier command, return code and executable hash. Core executables can change
between runs during development; each run identifies the one it used.

Source used for the formal batch is preserved in
`aie-toy/results/orbit-taskflow/source-profiled/`; later entrypoint validation
hardening additionally rejects incomplete references without any device work.
The source of new runs is also copied into each result directory.

Physical ReLU attempts and diagnostics are preserved as
`aie-toy/results/orbit-taskflow/relu-attempts.json` and `smoke-relu*.log`.
BFP instruction evidence and the standalone reproducible AIR compile command
are in `aie-toy/results/orbit-taskflow/bfp-lowering-proof.json`.

## Compiler working directory and in-process reuse

AIR's driver otherwise creates `air_project/` and `params.txt` beneath its
working directory. The adapter redirects only the **child compiler** working
directory to the result directory and passes an absolute `--tmpdir`. It never
changes the Python process working directory. `compile-commands.jsonl` and
`aircc-build.log` record the child invocation. A compiler-only check from the
repository root succeeded without creating files there; no NPU was invoked.
Earlier root artifacts are preserved in
`aie-toy/build/orbit-taskflow/early-cwd-artifacts/`.

The in-process compiled cache keys include padded extents, target/runtime,
physical tiles, dtype/layout, kernel/schedule/launcher hashes and wheel versions.
A changed recipe in the same process is explicitly refused, since the
underlying Triton JIT also retains specialization state. Use a fresh process
for a changed recipe. This guard and working-directory isolation were added
after formal hardware acceptance; neither changes the executed arithmetic.
