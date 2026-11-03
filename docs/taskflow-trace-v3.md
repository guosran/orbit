# AIE hardware event evidence, static mapping round 3

These are real NPU2/AIE2P captures and host API observations. They do not execute
the new static mapping plans. Static estimates, host wall time and AIE event
cycles remain separate. No installed toolchain or runtime package was modified.

## Installed interfaces and source audit

The installed versions were `triton-xdna 3.8.0.2026110104+79d7ae3`,
`mlir-air 0.0.1.2026102900+66a24f9.no.rtti`,
`mlir-aie-no-rtti 1.4.5.dev83+g1631e1b`, and
`llvm-aie 22.0.0.2026102001+4f5df13d`; device firmware was `1.1.2.64`.
Each final case stores `toolchain.json` with hashes of the installed decoder,
Triton driver and multilaunch implementation. This inventory was collected
read-only after execution; actual experiment driver snapshots are retained.

Installed `aie.iron.Program.enable_trace` matches the vendor
`programming_examples/basic/event_trace/aie_trace.py` API. The minimal example
uses that design and its existing scalar kernel event0/event1 markers. It
selects core vector, memory/stream/lock stalls and finite DMA/port event lists.
`TraceConfig` allocates a dedicated 64 KiB trace buffer, and the installed
`parse_trace` decodes the address-annotated physical MLIR. The parser's
`device_name` argument is an MLIR symbol, not a device generation: an initial
attempt passed `npu2`, failed decoding, and was recovered without re-running
hardware. Both attempts remain in `trace-minimal`; the final standalone entry
was subsequently executed successfully in `minimal-reproduce-smoke`.

Installed NPUChain has no public trace option. AIR's existing compiler exposes
`-trace-size` and `-trace-offset`; the trace driver recompiles the original ORBIT
one-op `combined-air.mlir` with those flags and uses the existing
`MultiLaunchRunner`. Its output array is extended by 1 MiB, the trace offset is
the physical FP32 output byte count, and the entire output is explicitly zeroed
and synchronized before each dispatch. The final graph's three projection
captures are copied from that execution's output tails. No XRT/ELF mechanism
was added to public Taskflow. `fused_program.py` was inspected but not used to
invent capture/replay or another execution mechanism.

## Actual graph execution and disturbance

Final evidence is in
`../aie-toy/results/orbit-static-v3-hardware/orbit-trace-instrumented/`.
The earlier `orbit-trace/` batch and successful small `trace-gemm-probe/` are
preserved diagnostics. Every variant excludes two initial executions, then
retains seven complete graph samples. Inputs, reference, original AIR source,
precision and host mathematics are identical between off/on. All outputs pass
the unchanged `.015/.06`, normalized RMS 3% and peak-scaled 8% acceptance.

| Graph | Trace off, median ms | Trace on, median ms | Host submissions / output readbacks |
|---|---:|---:|---:|
| Small GEMM | 0.398590 | 0.408100 | 1 / 1 |
| FFN M=5, D=2048, I=11008 | 10.005982 | 10.348804 | 3 / 3 |
| FFN M=128, D=2048, I=11008 | 13.101636 | 14.719696 | 3 / 3 |

These are same-batch trace perturbation comparisons, not speedups over earlier
ORBIT reports. The experiment uses the existing runner directly and timed
NumPy packing; its execution boundary differs from the earlier Torch adapter.
AIR tracing reserves a shim column and changes placement, output BO size,
staging/readback and instrumentation. Thus on/off changes cannot be attributed
only to trace emission. Frequencies were not locked; small differences are
not strong performance conclusions.

The FFN retains separate Gate and Up GEMMs, CPU `SiLU(G)*U`, and Down GEMM.
No polynomial substitute, CPU fallback or hidden device activation is used.
The off-run CPU SwiGLU median was about 0.057 ms for M5 and 1.469 ms for M128.
Full graph timing includes fresh intermediate allocation, zero fill, packing,
conversion, actual runner calls, CPU activation and final output copies.
Compilation, context load and first host weight preparation are separately
recorded; static weight BO contents are uploaded on first execution and reused.

`host-events.json` contains observed `run.start`, the original `_wait`, and
actual BO `sync` calls. Local proxy objects unwrap the real pyxrt objects before
calling the installed native wait and kernel argument binding. Both variants
carry this same observation overhead. Hot FFN graphs observe 3 starts, 3 waits,
6 host-to-device syncs and 3 device-to-host syncs. These are API calls and their
host durations, not isolated DMA bandwidth or device compute time. The larger
`runner_staging_sync_submit_wait_readback` sections contain those API intervals;
adding both levels would double count. `trace-summary.json` under
`build/static-mapping-v3/hardware/` records the aggregate evidence.

## Trace scope and interpretation

Each final projection capture has `trace.uint32.npy`, `trace.txt`,
`physical.mlir`, `trace.cycles.json`, `trace.perfetto.json`, `parse.log` and
`manifest.json`. The manifest binds task/operation/candidate identity, logical
and physical shapes, AIR and ELF hashes, decoder source hash, and capture
repeat index 8 (the seventh hot sample). This is one execution of one
projection, not concatenated traces from all samples. Physical core/port IDs
and event names come from the actual compiler register configuration.

The small GEMM has 1792 nonzero raw words and 2982 timed decoded records. Each
FFN projection has 163840 nonzero words and roughly 231000–235000 timed records.
The repeated FFN count and open terminal event tracks indicate saturation of
the active channel windows. Zero padding in the remainder does not prove that
those channels were complete. Every capture is conservatively **partial**;
there is no independent overflow counter or whole-kernel stop proof.

The AIR kernel does not emit event0/event1 region markers, even though the
compiler configures their slots. Observed families include vector instructions,
lock acquire/release requests, TRUE and port activity. Initial manifests
incorrectly required timed `INSTR_EVENT` records; the independent audit caught
this, and final manifests explicitly record that metadata correction. Raw
events were not changed. The minimal scalar example does have event0/event1
and stall events, but terminal stall tracks can still remain open.

Native decoder timestamps are **device cycles**. No device clock calibration
or host/device alignment was performed. Perfetto-compatible files use one
viewer microsecond per cycle solely as a cycle-axis display convention; their
metadata says this is not physical microseconds. Host time appears separately
in nanoseconds relative to graph start. No shifted combined timeline is used.
Overlapping stalls must not be added as disjoint execution time; port activity
does not measure DDR byte traffic. Partial FFN traces cannot calibrate complete
projection duration or prove absence of stalls. They do establish real tile
activity and the continuing device/host boundaries.

## Reproduce from a clean shell

The commands use the existing environment and previous authenticated ORBIT AIR
artifacts retained under `orbit-search-v2`; source hashes and paths are saved in
each `compile.json`. Every output directory must be new. Each script owns
`/tmp/npu.lock`; do not wrap it in an outer lock.

```bash
cd /home/shiran/project/orbit
source ../aie-toy/scripts/env.sh
python scripts/taskflow_aie_trace_minimal_v3.py \
  --output-dir ../aie-toy/results/orbit-static-v3-hardware/minimal-new
python scripts/taskflow_npu_trace_v3.py --case all --iterations 7 \
  --output-dir ../aie-toy/results/orbit-static-v3-hardware/orbit-trace-new
python test/TaskflowMapping/hardware_evidence.py \
  --trace-manifest ../aie-toy/results/orbit-static-v3-hardware/orbit-trace-new/gemm/on/trace-gemm-1/manifest.json \
  --output build/static-mapping-v3/new-gemm-trace-audit.json
```

The minimal standalone entry and the full GEMM/M5/M128 trace entry were both
executed successfully. The independent auditor re-decodes raw data with the
installed parser; nonempty metadata alone is insufficient. AIE event capture
is verified, but complete full-size trace windows and calibrated physical
device time remain unsupported by this evidence.
