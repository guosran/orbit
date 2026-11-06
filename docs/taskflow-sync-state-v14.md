# BO sync state and API cost boundaries (v14)

ORBIT's public ProgramGraph remains the source of task/data/region decisions.
Performance search is paused. CGRA retains its existing ML II model and original
mapper/scheduler/network. This stage adds ordered BO observations and guarded
API-cost queries; it does not replace unknown compute or DMA time with zero.

## Why the v13 repetition assumption failed

The frozen v13 model prices an API as `alpha_ms + beta_ms_per_byte * B`, summing
over calls. B is the requested sync range, not measured traffic or allocated
physical pages; n counts returned API invocations, not observed transfers.
Its nonnegative least-squares calibration used raw observations. The preserved
post-TO training row includes a 714691 ns observation accounting for 71.5% of
that row's sum; its median is 450 ns. The resulting intercept is about 1983 ns.
The original 29.10% heldout error and all raw observations remain untouched.

There are also distinguishable measurement contracts: the v13 standalone probe
uses explicit three-argument sync through a Python wrapper; the installed runner
uses one-argument whole-BO sync. Its detailed observer ends the inner timestamp
after forwarding and enum-string conversion, so this span is not raw sync alone.
New controlled observations show a sequence-entry transient even after a prior
TO, and different first and later invocation costs. Operational BO content state
alone therefore does not identify the observation path or its cost.

The evidence does not identify physical cache state, cache eviction, or DDR bytes
from latency. These remain possible hidden effects, not fitted physical causes.

## Local path evidence

Installed XRT/pyxrt 2.21.75, the actual loaded XDNA shim, active kernel-module
identity, runtime/config fingerprints, disassembly, GDB stacks and strace are
preserved in the companion `results/taskflow/sync-state-v14/` directory.
The new probe executes the userspace CLFLUSH loop in 64-byte steps. No sync ioctl
occurs between its manual-sync markers. A constructor-triggered TO precedes
mapping and initialization: "first initialized" means first manually measured
sync after initialization, not the first underlying sync in a BO's lifetime.

This is evidence for the captured local branch, not retroactive proof of the v13
process branch or every possible XRT platform. Installed DKMS source differs
from the active Ubuntu module and is not presented as its matching source.
Successful API return is the observable completion boundary. Successful device
output checks prove visibility for tested sequences; they do not count physical
transfers. The absence of a fence in one shim symbol alone does not establish
global completion or ordering semantics.

## Interfaces and ownership

| Interface | Producer / consumer | Meaning |
|---|---|---|
| `SyncState.h/.cpp` | Ordered host/device/sync observations; queried by planning and replay | Physical BO identity plus generation, exact ranges, logical visibility/content epochs, pending work and typed completion evidence |
| `SyncBoundaryMapping.h/.cpp` | Checked ProgramGraph + MechanismFacts + observed runtime bindings | Successful serial projection; preserves graph, copy, region and event identities; exposes required execution predicates |
| `SyncCost.h/.cpp` | State query + exact invocation/site/observer/runtime/condition + frozen parameters | Guarded API duration or explicit unknown; never a complete-program score |
| `taskflow-sync-opt` | Real Taskflow body + existing heterogeneous choice + runtime bindings | Rebuilds/checks the public graph, projects original ABI order, emits graph/facts/queries/cost |
| `taskflow-sync-replay` | Recorded operation streams | Replays state in C++; timestamps and test labels do not select state or parameters |

Host staging copies, CPU mapped loads/stores, BO visibility calls, NPU
system/local DMA, submission and wait remain separate. A shared mapped BO removes
the observed runner staging copy; it does not establish zero data movement.
Descriptive child DMA remains included in the opaque device/wait composite.
Pricing the sync API does not add a second DMA or hardware-bandwidth charge.

Aliases reference one physical BO; generations prevent stale identity reuse.
Partial accesses split range evidence. Pending work, missing completion, unknown
padding, unsupported histories and absent evidence cannot acquire calibrated
cost by choosing a convenient test name. The initial graph adapter supports
distinct BOs per logical storage and checked exact whole-storage CPU accesses;
suballocation and arbitrary in-place host/device aliasing remain unsupported.

## Frozen formulas and validation design

For a fresh standalone BO, after a prior same-range TO and with no intervening
mapped access, the batch API-loop envelope is

`T_ns = entry_ns + n * (call_ns + cacheline_ns * L)`, `L = requested_bytes / 64`.

Offsets and ranges must be 64-byte aligned. Three invocation forms are calibrated
separately. Entry includes the combined timer/wrapper/loop entry transient;
cacheline cost describes the captured loop-size effect, not physical bandwidth.
Coefficients are nonnegative least squares on equally weighted calibration-cell
medians. All raw samples, including outliers, remain available.

For the original installed runner, one sync at an exact ABI loop site/argument
ordinal uses a fixed API-span parameter over observed 1024--4096 byte BOs. It is
the median of four training-cell medians. No byte slope is forced onto that small
range. Device computation, wait, full region wall time and preparation are not
fit labels. The detailed observer is included in this explicitly named boundary.

The operational condition distinguishes zero prepared contexts with fresh
standalone BOs from two prepared contexts with reused caller-bound BOs, and binds
the measurement source. These domains are not pooled. The public cost query must
provide a matching condition identity, derived from execution evidence.

Before calibration, the protocol fixes training BO sizes 1024/8192/32768/131072,
counts 1/4/16, and heldout sizes 2048/12288/49152/98304, counts 2/7/23. Region sync
uses quarter ranges; n=23 specifically tests count extrapolation beyond training.
The original program sequence uses calibration counts 1/4 and heldout 2/7.
Three seeds and interleaved repetitions are retained. First/modified/read/device
contrasts identify limits; unsupported histories remain unpriced.

Acceptance is specified before observing heldout labels: cell-median batch error
at most max(500 ns, 15% actual), and matched program sync-subtotal error at most
max(2000 ns, 20% actual). All groups, failures, coverage, absolute/relative error,
MAD, tails and calibration sensitivity are reported. This criterion evaluates
small API subtotals; it cannot justify whole-program performance search.

Exact parameter bytes, calibration evidence and source are frozen before new
heldout runs. The companion report contains parameter values, independent
heldout results, API/driver audit and commands. The frozen v13 empirical region
interpolation, analytical parameters, failed heldout and original executor are
preserved with separate identities.

The primary frozen parameter file has SHA256
`a82c00acff10a4eb128659f62c1b3fc1f9462017fcc5871413da7d843f79da37`.
All coefficients below are in nanoseconds (the line coefficient is ns per
requested cacheline visit). They are mechanism-envelope parameter calibration,
not a regression of complete program/region time.

| Observed path | Entry | Per call | Per requested line |
|---|---:|---:|---:|
| Batch implicit whole BO | 387.0833 | 94.3551 | 0.504001 |
| Batch explicit whole BO | 380.8750 | 99.5067 | 0.503486 |
| Batch explicit region | 397.9167 | 120.2690 | 0.398169 |
| Original input loop, ABI 0 TO | 0 | 4095.0 | 0 |
| Original input loop, ABI 1 TO | 0 | 1545.0 | 0 |
| Original input loop, ABI 2 TO | 0 | 1187.5 | 0 |
| Original readback loop, ABI 2 FROM | 0 | 3265.0 | 0 |

Independent heldout results: 35/36 batch cells meet the preregistered threshold;
MAPE is 5.175%, maximum cell-median absolute error 541.806 ns. The failed cell
is implicit 2048 bytes, n=23: predicted 2928.194 ns versus measured 3470 ns,
exceeding its 520.5 ns limit. This remains a failure; there is no heldout refit.
Program sync subtotals pass 4/4 groups, MAPE 5.992%, maximum relative error
10.271%. Individual API/seed groups reach 27.686% error: subtotal acceptance does
not imply equally accurate individual calls. All 486 heldout program outputs
pass independent float64-reference checks, maximum absolute error
2.384185791015625e-7. Calibration separately checks 270 outputs.

The v13 29.10% and these percentages have different boundaries and sample units;
they are not a same-contract percentage improvement. Old coefficients applied
mathematically to new envelopes are explicitly labeled out-of-domain diagnostic
counterfactuals. Per-seed calibration, leave-one-size-out sensitivity, observation
overhead, group distributions and unsupported-state contrasts remain available.
The `raw_manifest_sha256` row field identifies canonical compact JSON content;
the companion canonical evidence file matches it byte-for-byte, while the
original frozen pretty JSON retains its separate exact-file hash.

## Reproduction

From `../orbit-artifact`, use `taskflow/scripts/run_sync_state_v14.sh --help`.
The entry requires an existing frozen protocol, the fitter path and a new output
directory; it performs calibration, freeze, new heldout runs and independent
audit. Microbenchmarks own the NPU lock; only the unmodified program observer
needs the wrapper's outer lock. Do not lock the entire entry recursively.

Build the CPU-only public tools with `scripts/build-core.sh NEW_BUILD` and run
`taskflow-sync-opt INPUT.mlir --target-config CONFIG --runtime-bindings BINDINGS
--sync-model MODEL --output NEW_DIRECTORY`. Runtime bindings contain observed
BO extents, physical IDs/generations, source copy aliases and exact observation
identities. The exported trace is a static successful-path projection, explicitly
distinct from `taskflow-sync-replay` of recorded execution observations.
