# Existing NPUChain two-GEMM experiment

`scripts/taskflow_npu_chain_v3.py` compares two implementations of the same
CPU-free `Y = (X @ W1) @ W2` graph. This is a separate two-GEMM workload, not
Qwen FFN: it does not remove SwiGLU, substitute polynomial activation, or equate
Gate/Up projection merging with submission packing.

The public source is ordinary f32 Taskflow with explicit zero fills and two
matmuls. The target implementation absorbs each zero fill into its existing
GEMM kernel's accumulator initialization. Both comparison variants use BF16
inputs, BFP16 EBS8 device arithmetic, BF16 intermediate and final stores, and a timed final BF16-to-FP32
host representation conversion. Thus they have the identical approximate numerical boundary; the BF16
midpoint differs from the earlier FFN runtime's FP32 host intermediates.
All outputs are checked against independent whole-matrix float64 reference
under the existing `.015/.06`, normalized RMS 3%, peak-scaled 8% acceptance.
This approximate target permission is explicit, not strict FP32 equivalence.

## Implementations and controls

- **Independent:** two persistent one-operation `NPUChain` objects, two ELF
  contexts, persistent weight buffers. Each hot graph runs both objects. The
  first BF16 result is read to host and staged as the second call's input.
- **Chain:** one existing two-operation `NPUChain`, one context and one host
  submission. The first output and second input use the same combined buffer
  argument. The BF16 midpoint stays in NPU-accessible system DRAM and is not
  read to host. Weights and all buffers are reused.

Neither case creates contexts or weights anew on each hot execution. The
comparison changes context arrangement, host intermediate handling and host
submission count together; its timing difference cannot isolate pure launch
overhead. The chain can contain configuration switches/PDI loading. One host
submission does not mean one device kernel or SRAM-resident kernel fusion.

Both variants reuse the upstream `matmul_bf16_epilogue` kernel and its existing
AIE2P transform schedule, with `SCALE=1`, BF16-store epilogue on the first GEMM,
and the same BF16-store epilogue on the second. The Triton grid tile is 128×256;
each program has its full K extent and the unchanged schedule controls physical
K tiling. This is not a newly developed runtime or kernel. The first attempted
ORBIT loop-recipe BF16-store specialization failed compilation with an L3
allocation inside `air.segment`; the source/IR/error remain separately under
`chain-compile-check`. The first upstream epilogue attempt used FP32 output on the second operation
and compiled but timed out during the independent second call; `chain/`
retains both calibration and held-out failures. The BF16-store variant avoids
that unverified FP32 epilogue and is retained separately under `chain-bf16/`.
The supported upstream epilogue recipe is used on both
sides of the final comparison, rather than silently substituting one side.

## Measurement and evidence

Shapes are M=128, K=256, H=256, N=256 for calibration, and M=128, K=256,
H=512, N=256 held out from calibration. The held-out shape is not used to choose
model coefficients. Both cases use seed 0 and preserve source tensors, actual
BF16-rounded input values and the independent reference.

Compilation, explicit context loading, first execution and an extra warm-up
are recorded separately. Seven full-graph hot samples alternate variant order.
The graph wall includes necessary host staging, synchronization, both actual
submissions/waits (independent), and final output copy. Output verification and
file writes are outside that timer. Every sample saves the complete output.

A local observer counts real `xrt.run.start` and existing `_wait` calls. The
observer unwraps the Python proxy before invoking the original `_wait`, keeping
native `npu_wait` versus polling selection unchanged. It does not edit installed
packages. Observation overhead remains in both measured variants. Per-sample
context/kernel/runner/BO object identities show reuse. These host observations
are not AIE hardware event traces. Performance measurements here are trace-off.

Each case stores the public source, core-authenticated Taskflow, target buffer
mapping, backend IR, compiled artifacts and SHA256 identities, excluded runs,
all hot samples and full-array numerical checks. A failed build/run remains a
failure with traceback and reproduction source; no automatic splitting turns
an unsupported packed chain into a reported success.

## Reproduction

Choose a new directory. The script itself owns `/tmp/npu.lock`; do not wrap it
in another lock. In a team, send execution to the single hardware owner.

```bash
cd /home/shiran/project/orbit
source ../aie-toy/scripts/env.sh
python scripts/taskflow_npu_chain_v3.py --case all --iterations 7 \
  --output-dir /home/shiran/project/aie-toy/results/orbit-static-v3-hardware/chain-reproduce
```

`--compile-only` builds and saves artifacts without acquiring the hardware
execution lock or opening a runner. Its output must use a separate directory.
The static mapping package is unchanged by this experiment. Host chain timing
can inform a separately labeled conditional model, but cannot be inserted as
pure compute latency into the static model or claimed as SRAM/DMA calibration.

`taskflow_chain_model_v3.py` produces a backend-only capability/cost catalogue.
It recognizes the exact supported two-GEMM region and rejects whole FFN with
CPU SwiGLU. Successful exact-region wall measurements remain opaque complete
region costs, explicitly rejected for import as static `compute_only` costs.
The independent hardware auditor owns the train-only held-out prediction check;
the catalogue references its evidence instead of fitting another model.

## Recorded BF16-store result

The final `chain-bf16` run passed the independent evidence auditor: 28 hot
outputs passed, separate and packed outputs were bit-identical, and normalized
RMS errors against the full float64 source reference were about 1.38%. All
hot samples showed persistent context/BO identities and actual submissions /
waits of 2 / 2 for independent and 1 / 1 for chain.

| Shape M/K/H/N | Independent median ms | Chain median ms |
|---|---:|---:|
| Calibration 128/256/256/256 | 1.075213 | 0.555146 |
| Held out 128/256/512/256 | 1.078674 | 0.559896 |

These are seven alternating-order complete trace-off graph samples per point.
The auditor's conditional model learns only the calibration saving, then
subtracts it from the measured held-out independent latency. It predicts
0.558607 ms against 0.559896 ms observed packed latency (0.2301% absolute
relative error). This conditional host-system check does not identify pure
compute, DMA, or launch overhead and does not validate the original static
compute-only assumptions.

The auditable catalogue is
`build/static-mapping-v3/hardware/chain-capability-cost.json`; independent
checks and held-out prediction are in
`build/static-mapping-v3/validation/hardware/chain-bf16-audit.json`.
Only a matching independently passed audit enables a region's score. Without
it, raw observations remain visible but score is null.

```bash
python scripts/taskflow_chain_model_v3.py \
  --chain-root /home/shiran/project/aie-toy/results/orbit-static-v3-hardware/chain-bf16 \
  --audit build/static-mapping-v3/validation/hardware/chain-bf16-audit.json \
  --program build/search-npu-v2/data/ffn/base.json \
  --output build/static-mapping-v3/hardware/chain-capability-cost.json
```
