# v32 r9 NPU validation and calibration checkpoint

All twelve selected CPU/NPU GEMM+ReLU implementations passed numerical checks
in three fresh processes, with five warmups and thirty-one measured calls per
process: 1,296 calls in total. This is execution and local-cost evidence, not a
CP-SAT speedup or a proof over the complete ORBIT abstraction.

The frozen compiler SHA256 is
`b7b7d4849ed07439ad2b30114899224432acbab6e892bba04383af9c350105e8`.
Raw commands, manifests, exact selected graph identities, final runtime proofs,
ELFs, process receipts and measurements are in
`orbit-artifact/results/taskflow/full-joint-v32-r1/preparation-r9` (sibling
artifact repository). `measurement-summary.json` contains all process medians;
`calibration/archive-manifest.json` seals the copied fit/validation evidence.

## Execution boundary and resource limits

The measurement includes input reset, serial CPU/NPU regions, host submission,
native wait2, host/device synchronization, readback and output copy. Compilation
and persistent context preparation are outside this prepared-call boundary.
All configurations remain resident, and the explicit execution contract limits
their total across regions to sixteen. A direct probe created contexts0–15 and
rejected context16. This is evidence for this installed runtime, not a universal
NPU capacity. Streaming context setup would require a different contract and
explicitly measured setup/teardown costs; no silent fallback is used.

Every selected configuration now has at most one host output transfer and no
repeated host input rectangle. Splits occur only at independently checked empty
local-copy frontiers; the complete event/DMA union remains unchanged. This
conservative policy follows actual r7/r8 failures, not a complete core-lock
proof. The final runtime receipts still disclose that full lock protocols and
kernel arithmetic are not statically certified. See
[multi-output failure evidence and isolation](matrix-runtime-repeated-feed-isolation-v32.md).

CPU affinity was fixed to logical CPU0. Compilation and team CPU jobs were
stopped during measurements. The governor and unrelated processes were not
changed. Process variation remains visible: GEMM64 CPU medians were0.223794,
0.130686 and0.223244ms; no result was discarded.

## Actual medians

Each row is the median of three independently measured process medians.

| Program | CPU ms | Selected NPU tile | NPU ms |
|---|---:|---|---:|
| GEMM16 + ReLU |0.002050|16×16×16|0.357511|
| GEMM32 + ReLU |0.005030|32×32×32|0.421039|
| GEMM32 + ReLU |0.005030|16×16×16|1.571821|
| GEMM64 + ReLU |0.223244|32×32×32|1.764607|
| GEMM64 + ReLU |0.223244|16×32×64|3.272169|
| GEMM64 + ReLU |0.223244|32×16×64|3.309539|
| GEMM48 + ReLU (holdout) |0.052189|16×48×48|1.286038|

These programs favor CPU under the measured boundary. Choosing the better NPU
tile is not evidence that NPU beats CPU, nor that CP-SAT beats ORBIT search.

## Frozen fit and held-out validation

The model was fit from training rows only, after all input files had been
sealed. Heldout timing labels were first opened by validation after model
freeze. No r5/r7/r8 timing labels were reused. New case declarations were based
on source legality and the original24KiB SRAM/sixteen-context constraints;
`case-design-protocol.json` retains the changes and reasons.

Model SHA256:
`67a8c4e112daa0ba1648583ea3e46fa865a4de0bccc135fad9f90eac205bcb4b`.
The predeclared median absolute relative error threshold was20%; the two heldout
implementation groups gave16.44% median, so this aggregate gate passed. The CPU
prediction error was29.40%, while NPU was3.49%. Thus an aggregate pass must not
be described as every prediction being within20%. The measured CPU winner's
three-process interval was strictly separated from the NPU interval, and the
model selected the same winner. Submit terms remain collinear; fitted values
must not be presented as independently identified physical launch parameters.

Native C++ search/replay and comparison with an explicit baseline are separate
required evidence. Neither the validation script nor this report claims a
CP-SAT hardware acceleration.

## MLP follow-up

A same-source two-layer shared-weight MLP reduces four phase-local configurations
and ten DMA events to one whole-program configuration and three DMA events.
Its old numerical failure exposed non-RNE intermediate BF16 conversion. The
ORBIT-owned RNE refinement has CPU boundary tests, but r9's actual MLP pipeline
stops on a vendor/pinned-MLIR `arith.trunci` default-property incompatibility.
That failure is retained. A separately versioned compatibility refinement and
actual numerical/timing validation are required before reporting MLP speedup.
