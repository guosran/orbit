# v32 repeated-input configuration isolation

The prep8 compiler applies a conservative source-bound configuration restriction
following the prep7 64×64 hardware timeout. This change has passed CPU checks;
**new board correctness has not yet been established**. The timeout's complete
cause remains unknown. Static DMA coverage is not a hardware-correctness proof.

## Actual failure and scope of the evidence

The failed workload uses explicit BF16 matrix inputs, FP32 accumulation and a
FP32 ReLU epilogue, with matrix tile M16×N16×K64. Prep7 configuration 0 contains
two matrix calls sharing the same A rectangle and physical C buffer. The full
run timed out after the external 45-second limit without producing outputs.
The original AIR, AIE, contracts, ELFs and timeout evidence remain preserved in
`orbit-artifact/results/taskflow/full-joint-v32-r1/preparation-r7`.

Read-only inspection of that configuration's `runtime.generic.mlir` establishes
an output-lifetime defect independently of the timeout: two core MM2S
channels read buffer `%16` at offset 0, length 256 (lines 28 and 37). The core
acquires both output-empty locks before either matrix call (65–66), then
releases both output-ready locks only after the second call and unpack
(260–261). The first produced tile may therefore be overwritten before its DMA
reads it. Exact host addresses, routes and descriptor counts alone do not
prove that an output observes the correct produced value.

The initial hypothesis that the core continuously holds A across both calls is
not supported literally by this IR: it releases A-empty at 162, reacquires
A-ready at 163, then releases B-empty and reacquires B-ready at 164–165. This
report does not claim that the output-lifetime defect fully explains the hang.

The runtime receipt explicitly leaves `core_lock_protocol_checked`,
`kernel_arithmetic_checked` and `core_numeric_checked` false. These limits
continue to apply after this change.

## Implemented restriction

[ProgramGraphAIRSelection.cpp](../lib/TaskflowAIR/ProgramGraphAIRSelection.cpp)
identifies repeated system-BO input rectangles by logical storage, offsets,
sizes and local target memory/owner. Before admitting a second identical
rectangle, the selector stops range expansion and chooses an earlier, already
checked empty-local-copy frontier. A creator and any consumer of a local copy
must remain in the same configuration. If no such frontier exists, the
realization is unsupported.

The independent single-selection and complete-union checkers enforce this
restriction as well. The union still covers every selected source event and
DMA occurrence exactly once, preserves the full ABI, and adds no host
transfers. Separate configurations require serial native `wait2`; input sync
occurs once before the sequence and output sync once after it. Private
intermediates must not be overwritten by another input sync.

[ProgramMatrixRuntime.cpp](../lib/TaskflowAIR/ProgramMatrixRuntime.cpp) also
rejects an old single-configuration contract containing repeated identical
input address streams, even if its static route/address/FIFO checks pass. This
runtime guard is deliberately conservative: without a trusted per-target
lock-lifetime proof it does not exempt distinct targets. Existing exact
coverage, packet boundaries, route/storage/dtype, FIFO and completion checks
remain mandatory.

Different reduction panels K0 and K1 have different input rectangles and stay
legal. This is not a global three-DMA limit, a source rewrite, deletion of
redundant-looking transfers, or a complete lock-protocol verifier.

## Checked consequences

Native checked exports with the applied C++ implementation give:

| Workload tile | Prior configurations | New configurations |
|---|---:|---:|
| 64×64, M16 N16 K64 | 12 | 17 |
| 48×48, M16 N16 K48 | 7 | 9 |
| 64×64, M32 N32 K32 | 5 | 5 |

The larger counts must enter the new physical invocation model and its
calibration. Earlier timing/model evidence does not price this new control
sequence automatically, and none of these counts is a speedup claim.

CPU validation:

* `check-taskflow-matrix-runtime-static`: 11/11 cases, including rejection of the
  previously accepted repeated static feed; dynamic/IV-dependent/side-effecting
  and overflowing loops still reach their specific runtime guards.
* The existing 16 matrix contract: 34/34 checks.
* The existing 64 tile32 contract: 54/54 checks, including a hand-authored unsafe
  repeated-input range rejection and preservation of its legal K2 selections.
* Exact event/DMA union checks pass for the three exports in the table.

The actual extracted fixtures and sealed contracts remain byte-for-byte in
[fixtures/matrix-runtime-v32](../test/TaskflowGeneric/fixtures/matrix-runtime-v32).
Their provenance file records hashes. The constant-feed fixture is now a
known-unsafe negative witness, not a positive execution certificate. The
[Python driver](../test/TaskflowGeneric/program_matrix_runtime_static_v32.py)
extracts SSA IDs rather than hardcoding printer names, and a nonzero exit counts
only when the expected runtime diagnostic is present without parser failure.
No vendor compiler or hardware is needed to replay that target.

The reviewed patch and detailed local audit were recorded in
`/tmp/matrix-repeat-input-safety-v32.patch`,
`/tmp/matrix-r7-lock-output-audit-v32.md` and
`/tmp/matrix-repeat-input-64-export-r1/summary.json`. These are local working
artifacts; repository fixtures and this report provide the durable regression
scope. Production application followed explicit parent review after automatic
approval first required a reviewable patch. No vendor source was modified.

The resulting prep8 frozen compiler SHA256 is
`e9d04f4275897596c6db8a16b008aec9716871e05201b081d4fb0d75a6764592`.
Final AIE recompilation and fresh numerical board gates are required before
claiming this realization works on hardware.

## Subsequent prep8 limits and single-output restriction

Prep8 created 16 prepared NPU contexts but rejected context index16 with error22
before executing the 17-configuration candidate. This is an observed prepared
runtime limit, not successful numerical evidence. An independent diagnostic
that creates/runs/unloads one context at a time uses a different runtime policy
and is excluded from the calibration. Its log completed configurations0–11 and
then hung after submitting configuration12.

Configuration12's checked range is [75,86). Its selected outputs are event79
(C at host element3120), event82 (Y at0) and event85 (Y at16): it combines the
last matrix tile and two ReLU tiles. This directly demonstrates that repeated
input isolation alone is insufficient to establish the supported control
sequence. It does not, by itself, prove the full cause of the hang.

After review, the selector, independent selection/union checkers and final
runtime checker now additionally require **at most one selected host output
transfer per configuration**. All source events and DMA occurrences remain;
only checked local-copy frontiers can split them. A complete K2 matrix tile
with two A inputs, two B inputs and one C output remains legal. Old
multiple-output contracts are explicitly unsupported even when exact static
coverage succeeds. Helper names and receipts describe both input and output
isolation; the full lock/arithmetic proof remains outside scope.

Checked exports under the original **24 KiB** target give these replacement
candidates (source-export diagnostics, not board-ready evidence):

| Full workload / tile M×N×K | Configurations | Logical payload + layout scratch |
|---|---:|---:|
| GEMM64 / 16×32×64 | 16 | 12,288 + 8,192 B |
| GEMM64 / 32×16×64 | 16 | 12,288 + 8,192 B |
| GEMM48 / 16×48×48 | 6 | 15,360 + 9,216 B |
| GEMM16 / 16×16×16 | 2 | 4,096 + 2,048 B |

Every child contract has at most one output, and complete event/DMA union
checks pass. These exports reference an older executor contract and cannot be
used as executable plans for the new runtime. Fresh registration, final
compilation, calibration and numerical gates are required; older timing labels
must not be reused.

The single-output CPU target passes 11/11 runtime cases. Additional C++ checks
pass: 22 source/selection checks for GEMM16; 68 checks for GEMM64 tile32 including
an actual single-output K2 AIE realization; and 13 checks rejecting the old
multiple-output GEMM16 module as unsupported. This replaces the earlier legacy
16-module positive expectation; no check was removed to disguise a failure.
Local source-export evidence is
`/tmp/matrix-single-output-candidates-v32-r1/summary.json`, with CPU logs
`/tmp/matrix-single-output-runtime-contracts-r1.log`,
`/tmp/matrix16-single-output-selection-r1.log` and
`/tmp/matrix64-single-output-regression-r1.log`.

At this report update, a new numerical board gate is still pending. The
single-output restriction is a conservative supported-subset policy, not a
claim that arbitrary lock protocols are now verified.
