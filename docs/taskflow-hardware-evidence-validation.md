# Hardware evidence audit

`test/TaskflowMapping/hardware_evidence.py` reads saved files only. It never opens
XRT or dispatches kernels. Hardware execution remains owned by the designated
worker under `/tmp/npu.lock`; the static mapping contract/checker is unchanged.

## Installed API findings

The installed `triton.backends.amd_triton_npu.multilaunch.NPUChain` captures
compatible operations into one ELF. `MultiLaunchRunner.run` issues one actual
`xrt.run.start()` and waits for it. The runner retains its device, ELF, context
and kernel; `bo_key` retains a BO set. Static operands skip subsequent staging,
and declared scratch is initialized once and not host-staged subsequently.
The LRU can close contexts and force restaging, so successful hot reuse must be
observed, not inferred merely from using the class.

An intermediate shared combined argument is a system-memory handoff. A single
submission does not establish AIE SRAM fusion, zero configuration-switch cost,
or absence of internal `load_pdi`. CPU SwiGLU cannot occur inside this chain.
The two-GEMM experiment uses a BF16 midpoint in both variants and is a separate
workload from the standard FFN's CPU SwiGLU boundary.

The installed `aie.utils.trace.parse.parse_trace` decodes device packets using
the exact physical MLIR trace configuration. Its output `ts` values are device
cycles, despite using Trace Event Format. `Event_Sync` advances the device
counter by `2**18` cycles. A visual conversion must name its clock assumption;
no host/device clock alignment follows from this parser.

## Trace audit

A trace manifest uses `schema=orbit.hardware.trace-evidence.v1` and records:

- `raw_path`, `physical_mlir_path`, `decoded_native_path`.
- `native_timestamp_unit=device_cycles`, expected event/tile regex patterns.
- Decoder API (`parse_trace` or `parse_trace_slices`), exact kwargs, optional
  source hash, and a conservative completeness statement.
- Optionally a separate visual JSON plus `microseconds_per_cycle` and its basis.

The auditor independently re-decodes saved uint32 words and requires exact
equality with the saved native events. It rejects metadata-only or fabricated
host timelines, checks expected events/tiles and per-track monotonicity, and
records unmatched intervals and trace-buffer tail occupancy. Tail zeros alone
do not prove absence of overflow, dropped events or split-channel truncation.
A complete claim requires start/stop scope evidence, an overflow check and no
unmatched event intervals. Otherwise the trace remains explicitly partial.

The first minimal capture independently passes raw provenance and event checks:
64 KiB capacity, 80 nonzero words, 83 timed records plus 18 metadata records.
Core markers, port activity, lock stall and DMA starvation events are present.
Three tracks retain an unmatched terminal begin, so the audit labels this
capture partial. Its initial parser error mistakenly used hardware generation
`npu2` as an MLIR symbol; the original error is retained and decoding was
recovered with the correct default symbol selection.

Five adversarial fixtures reject wrong units, fabricated host events,
metadata-only events, an unsupported complete claim and a changed decoder hash.
Evidence lives under `build/static-mapping-v3/validation/hardware`.

## Separate versus packed audit

The auditor consumes `orbit.chain.experiment.v1` results, rederives the whole
two-GEMM float64 reference from saved original inputs, verifies BF16 input
rounding, and checks every saved hot output with the unchanged absolute,
relative, normalized RMS and peak-scaled criteria. It compares each operation's
frontend IR between separate and packed variants, verifies source/artifact
hashes and recomputes medians and pairwise output differences.

Actual locally observed XRT start/wait events must number two per independent
graph and one per packed graph. Hot samples must retain identical recorded
chain/runner/ELF/context/kernel/BO identities. This establishes host submission
and object-reuse boundaries; declared readback indices are not hardware DMA
counters. The local observer preserves the original wait implementation.

Any compile/runtime/numerical failure remains unscored. A separately reported
conditional holdout model may estimate packed latency as measured separate
latency minus the train shape's observed fixed saving. Its constant never uses
held-out packed results; signed and relative errors are reported. This is a
limited host-system model including staging and switching. It neither extracts
pure compute costs nor validates the static mapper's compute-only cost model.

```bash
cd /home/shiran/project/orbit
source ../aie-toy/scripts/env.sh
python test/TaskflowMapping/hardware_evidence.py \
  --trace-manifest build/static-mapping-v3/validation/hardware/minimal-manifest.json \
  --output build/static-mapping-v3-reproduce/minimal-audit.json
# Add --chain-result /absolute/successful-run/train/result.json and
# --chain-result /absolute/successful-run/holdout/result.json when available.
```

Overlapping stall intervals are never added to produce a total time, port
events are not DDR byte counters, and host API observations are not device
events. Successful raw parsing establishes the provenance of the captured
events, not comprehensive coverage of all kernels, tiles or graph boundaries.

## Final audited batch

`build/static-mapping-v3/validation/hardware/final-audit.json` is the final
combined audit. Its exact read-only command is saved beside it as
`final-audit-command.json`. It covers the successfully rerun standalone minimal
example, seven instrumented ORBIT projection captures, three graph on/off
comparisons and both successful BF16 two-GEMM chain shapes. Earlier timeout
and metadata-error records remain diagnostic evidence, not successful samples.

All eight raw captures independently re-decode to their saved native JSON.
All viewer events equal those raw-derived events. The viewer deliberately
uses an unscaled cycle-axis surrogate: its displayed microsecond labels are
not physical microseconds. No CPU/device alignment is asserted.

AIR configures vector, lock request/release and port events; configured
`INSTR_EVENT` marker slots have no observed marker records in these projection
captures. The initial manifest incorrectly expected those markers; this
metadata error was corrected against the physical configuration, retaining the
absence of markers and the partial-coverage conclusion. Full-size captures all
contain 163840 nonzero words in a 1 MiB allocation with channel/window fullness
risk. Their zero tail does not establish completeness. No compute-time or
complete-stall calibration is extracted from these partial traces.

The auditor checks candidate/task/operation/shape to AIR/ELF/physical-config
identities and the captured final hot iteration. It recomputes all **54** saved
graph outputs from original source equations, including excluded first/warmup
outputs. Trace-on/off outputs are identical in this batch. For FFN, each hot
graph has observed three actual XRT starts and waits, six TO_DEVICE sync calls
and three FROM_DEVICE sync calls; the CPU SwiGLU node remains present.

| Graph | Trace off median ms | Trace on median ms | Ratio |
|---|---:|---:|---:|
| Small GEMM | 0.398590 | 0.408100 | 1.02386 |
| FFN M=5, D=2048, I=11008 | 10.005982 | 10.348804 | 1.03426 |
| FFN M=128, D=2048, I=11008 | 13.101636 | 14.719696 | 1.12350 |

Each mode retains seven complete hot samples. These ratios include changed
physical placement, trace storage/readback and local observer overhead. They
are not isolated event-tracing cost, and these driver times must not be compared
with prior runtime batches to claim acceleration.

The successful chain retry uses BF16 stores for **both** GEMMs, with final
BF16-to-FP32 host conversion timed equally in both variants. All **28** hot
outputs pass independent whole-source checks; maximum normalized RMS is 1.381%,
and paired independent/chain output difference is exactly zero. Actual starts
and waits fall from two to one; all recorded hot execution-object and BO
identities remain stable. Corresponding frontend kernels are identical.

| Two-GEMM shape (M,K,H,N) | Separate ms | Packed ms |
|---|---:|---:|
| Calibration (128,256,256,256) | 1.075213 | 0.555146 |
| Held out (128,256,512,256) | 1.078674 | 0.559896 |

The train-only fixed-saving conditional model predicts held-out packed latency
0.558607 ms versus observed 0.559896 ms: absolute relative error 0.2301%.
This narrow result cannot establish a general launch constant or predict pure
device compute. The original FP32 second-output epilogue attempt timed out and
is retained separately; it is not counted as a successful chain implementation.
