# Direct AIR hardware harness

`scripts/taskflow_direct_air_execute.py` accepts AIR emitted by the native C++
backend, a buffer-only ABI, and independently generated NPZ inputs/references.
It invokes the installed `aircc` executable directly. There is no Triton recipe,
operator substitution, mapping algorithm, or graph reference interpreter here.
The installed `MultiLaunchRunner` is reused solely to dispatch the resulting ELF
through XRT. Its module belongs to Triton-XDNA, but its compiler is not called.

The ABI lists function arguments in order as `buffers`, each with `name`, `shape`,
`dtype` (`f32`, `bf16`, or `i32`), and an `outputs` list of names. Internal AIE
allocations do not become host arguments. ABI names identify supplied arrays;
they do not define the computation. The C++ lowerer owns operation semantics.

## Clean shell

```bash
cd /home/shiran/project/orbit
source /home/shiran/project/aie-toy/scripts/env.sh
python scripts/taskflow_direct_air_execute.py \
  --air /absolute/new-case/program.air.mlir \
  --abi /absolute/new-case/abi.json \
  --inputs /absolute/new-case/inputs-11.npz \
  --reference /absolute/new-case/reference-11.npz \
  --inputs /absolute/new-case/inputs-23.npz \
  --reference /absolute/new-case/reference-23.npz \
  --output-dir /absolute/new-experiment --iterations 5
```

The output directory must not exist. Compilation precedes acquisition of
`/tmp/npu.lock`; execution acquires that lock itself. Do not wrap this command
in another holder of the same lock. `--compile-only` records failures without
acquiring the device. No installed compiler package is edited.
The executor passes `--stack-size=4096` to installed `aircc` by default and
records this allocation in `compile.json` and the exact command. The CLI can
override it with `--stack-size`. The scalar FFN needs 2368 measured stack bytes;
the installed default of 2048 is insufficient. Stack measurement stays enabled.

## Measurement and correctness boundary

Compile time, context loading, two excluded warmups and at least three hot
samples are separate. Hot wall time includes input BO staging/synchronization,
submission/wait, output BO synchronization/readback and the copied returned
outputs. NumPy input preparation is excluded. Raw API timestamps independently
record actual submissions, waits and synchronization. All ABI buffers are
conservatively uploaded on every execution; contexts and allocated BO sets are
reused within a seed. Each seed creates a fresh context. A BO is system memory
accessible to the NPU, not AIE SRAM residency.

References are external to the executor. `test/TaskflowAIR/prepare_inputs.py`
contains explicit independent NumPy equations for the named test workloads,
with float64 reference arithmetic, negative/zero/non-symmetric inputs and
multiple seeds. It is not a production semantic checker or a graph importer.
Every output element, shape, finiteness, maximum absolute error and normalized
L2 error is recorded; pass/fail additionally uses the explicit atol/rtol.
Default tolerances are `1e-5`/`1e-5`. If `--bf16-emulation` is used, the compile
record states it and results must separately state the changed arithmetic.

No device trace is injected by this harness, and host API timing is not labeled
AIE event timing. It does not claim compute-only latency or a calibrated model.

## Executed v5 evidence

The local result root is
`/home/shiran/project/aie-toy/results/orbit-direct-air-v5/`. These are direct
native C++ AIR artifacts, not reused v4 recipe binaries:

| Directory | Workload | Actual placement | Numerical result |
|---|---|---|---|
| `gemm-native` | GEMM 4×8×4 | one AIE core | 3 seeds × 7 executions pass, max abs 5.96e-8 |
| `add-native` | 16-element addition, no witness | one AIE core | 3 × 7 pass, max abs 1.19e-7 |
| `add-scale-native` | 4×4 addition then scaling | one AIE core, intermediate retained in L1 | 3 × 7 pass, max abs 1.19e-7 |
| `search-comparison` | original search baseline / real outer-loop split | same direct pipeline, one AIE core | 3 seeds × 11 × 2 variants pass |

All successful cases use f32 input/output/arithmetic; no BF16 emulation flag.
The 2 warmups per seed are retained but excluded. Independent audit files are
`build/direct-air-v5/{gemm,add,add-scale}-hardware-audit.json` and
`build/direct-air-v5/search-hardware/audit.json` (129 checked outputs total).
Each graph has one observed host submission. The multi-task case has no host
intermediate round trip; input/output boundary DMA remains explicit in AIR.

The same-batch search comparison uses two persistent runner/context/BO sets,
forward/reverse alternating order, 9 hot samples per variant per seed. Baseline
versus tiled medians in milliseconds are 0.340722/0.341352,
0.292283/0.293513, and 0.292023/0.293223. These small differences on an unlocked
machine establish neither acceleration nor a reliable performance regression.
The cost model has not been calibrated with these times.

`bound-cases.json` binds each search candidate's actual Taskflow, native plan,
AIR and ABI hashes; each compile record binds its ELF. The structural transform
changes loop IR. It does not promise changed physical placement or a speedup.

The failure directories are retained separately: `ffn-first` records the
installed AIR dependency pass's need for affine→SCF lowering;
`gemm-lower-affine-probe` records nested `air.execute` dominance failure;
`gemm-wait-probe` records unsupported `scf.execute_region`; and `ffn-native`
records Peano's unlegalized scalar `G_FEXP`. The early `gemm-flat-probe` is only
a diagnosis experiment. Final GEMM/add/multi-task evidence uses native C++
output without textual rewriting. No installed package was modified.

The hardware-only reproducible entrypoint generates the above native source
fixtures, independent inputs and ABI serialization automatically:

```bash
bash scripts/run_taskflow_direct_air_hardware.sh \
  --tool /home/shiran/project/orbit/build-air/tools/taskflow-opt/taskflow-opt \
  --output /home/shiran/project/aie-toy/results/orbit-direct-air-v5/repro-NEW
```

`--approx-exp` additionally attempts the small standard FFN with explicitly
requested standard MLIR f32 exponential approximation. Exact scalar exponential
and approximation compilation failures remain distinct evidence; neither causes
a silent CPU/Triton substitution. The installed Peano initially rejects exact
`G_FEXP` and the first approximation's scalar `G_FMA`.

The final native approximation expands FMA to separate f32 multiply/add and
expands its bounded floor operations through integer conversion. It explicitly
records finite-input assumptions and flushes denormals; it is an approximation
mode, not exact exponential support. The intermediate `ffn-exp-unfused-native`
failure retains the unsupported `G_FFLOOR` MRE. The first clean-shell directory,
`repro-clean-shell-20261104`, retains the measured 2368-byte stack failure.

The successful clean-shell run is
`repro-clean-shell-final-20261104/summary.json`, invoked with `env -i`, only
`HOME`, `PATH`, and `LANG` supplied, and `bash --noprofile --norc`. Its newly
compiled GEMM, add, add-scale and standard four-task FFN each pass 3 independent
seeds × (2 warmups + 5 hot executions), with unchanged atol/rtol `1e-5`/`1e-5`.
All 84 saved output sets pass independent audits. FFN maximum absolute error
against the float64-equation reference is `1.4901161193847656e-8`. It uses f32
ABI/arithmetic, no BF16 emulation, one observed submission, and L1 intermediate
buffers. This run establishes correctness on these inputs, not a speedup.
The entrypoint writes a summary after every requested case passes.
