# V4 measured region calibration

The experiment driver reuses the installed `NPUChain` and upstream
`matmul_bf16_epilogue` recipe. It does not introduce a runtime or change installed
packages. `taskflow_npu_chain_v4.py` compares two persistent single-operation
contexts with one persistent two-operation context. Both cache immutable weights.
Each measured wall starts with ordinary FP32 activation conversion to BF16 and
ends after the BF16 result has been read and converted to FP32. Compilation,
context creation, and initial weight upload are separately excluded. This is an
opaque warm host-to-host region cost, not a compute-only latency.

The original predeclared four corners used H=256/768. Both H=768 cases failed
compilation because the second GEMM's existing `tl.arange(0,K)` requires a power
of two. The original split, successful cases, and failures remain under
`aie-toy/results/orbit-static-v4-hardware`. No held-out measurement was made.
A separate declaration under `pow2-revision` uses M=128/384, H=256/1024,
K=N=256; held-out M=256,H=512. All four training cases passed with seven
alternating-order hot samples per mode. The declaration and script hashes are
saved before execution. The source, materialized Taskflow, Triton shared IR,
AIR, compiled artifacts, inputs, outputs, independent reference, and real XRT
start/wait observations are retained for every case.

`taskflow_cost_calibration_v4.py freeze` reads only training results, rejects a
preexisting held-out result, fits a four-coefficient bilinear function of M,H,
and writes an immutable model and predictions before held-out execution. Signed
coefficients are permitted within the declared positive interpolation domain;
this is an interpolation of measured system costs, not a hardware performance
law. Outside the rectangle or unsupported kernel shape the answer is unknown.
Frequency was not locked; small changes and cross-batch differences are not
proof of speedup.

`taskflow_npu_single_calibration_v4.py` independently measures first- and
second-GEMM region families. The second uses a separate seed-19 signed FP32
activation, not an output borrowed from a prior timed operation. Each family
has its own reference and seven samples. Their costs are not obtained by
splitting pair costs, and they are different computations rather than two
implementations to rank. The saved pair Taskflow is a compilation witness;
each sample's timing scope is exactly its named single operation.

`taskflow_execute_region_plan_v4.py` requires actual native plans before hardware
execution. It invokes the independent native checker on each plan and actual
MLIR body, checks the four covered fill/matmul steps, shape, model, layout,
precision, prepared state and mode, then dispatches exactly the requested
existing implementation. Comparison requires one checked native plan per mode;
the selected-plan and model identities are stored with the executed samples.
That initial pair executor accepts the complete two-GEMM region. The later comparison executor also accepts singleton and mixed exact covers.

## Results and exact execution

Held-out M=256,H=512 was executed only after native plans for both alternatives
had been generated and checked. Frozen predictions were 1.177399 ms independent
and 0.629754 ms chain; observed medians were 1.203395 and 0.659756 ms. Relative
prediction errors were -2.16% and -4.55%; the chosen chain had zero observed
regret. No refit occurred.

The native two-round search's N=256 partition candidate was executed directly
from its selected plan, not replaced by the baseline. Its two first-projection
column regions and one second projection made three observed submissions and
waits per graph. All seven outputs passed the independent reference
(normalized RMS 1.364%). Prediction: 1.636019 ms; complete graph median:
1.806159 ms; error: -9.42%. The graph timer also includes host slice assembly
that standalone region measurements do not capture. This candidate did not
show a benefit. Baseline and structural measurements are different batches,
so their difference is not presented as a controlled speedup ratio.

The initial singleton adapter rejected the M-partition mixed cover and retained
`unsupported-execution.json`. The final comparison adapter executes that exact
selected mixed plan; `execution-supported.json` records the later success.
The original plans and rejection evidence were not replaced.

Reproduce training and freeze predictions from a clean shell using new result
directories (existing evidence is intentionally not overwritten):

```bash
cd /home/shiran/project/orbit
source ../aie-toy/scripts/env.sh
python scripts/taskflow_npu_chain_v4.py --declare --output-dir /tmp/orbit-v4-pair-repro
python scripts/taskflow_npu_chain_v4.py --split train --iterations 7 --output-dir /tmp/orbit-v4-pair-repro
python scripts/taskflow_cost_calibration_v4.py freeze --hardware-root /tmp/orbit-v4-pair-repro --output-dir /tmp/orbit-v4-model-repro
```

The new region configuration must then enter native mapping before held-out
execution. For the retained selected structural candidate, a directly runnable
replay is:

```bash
cd /home/shiran/project/orbit
source ../aie-toy/scripts/env.sh
python scripts/taskflow_execute_structural_plan_v4.py \
  --source build/static-mapping-v4/region-calibration/search/candidates/9830aa65adc90bdf6e2bb51a6322f7f09c5107143affd8e4f8770798f1d4daff/taskflow.mlir \
  --plan build/static-mapping-v4/region-calibration/search/candidates/9830aa65adc90bdf6e2bb51a6322f7f09c5107143affd8e4f8770798f1d4daff/selected-plan.json \
  --config build/static-mapping-v4/region-calibration/search-config.json \
  --output-dir /tmp/orbit-v4-structural-repro
```

Both execution scripts own `/tmp/npu.lock`; do not wrap them in another lock.
Their prepared state means persistent contexts and weights in NPU-accessible
system memory, not complete weights resident in AIE SRAM.


## Final controlled comparison

`taskflow_execute_comparison_v4.py` prepares all three native plans together,
checks identical seed-0 input bytes, and runs nine forward/reverse alternating
hot iterations. All contexts and immutable weights persist. Every hot sample
records its actual regions, real submissions/waits, runner and BO identities,
and static weight object identities. The timer includes activation conversion,
region slice assembly, temporary allocation, synchronization and final copying.

| Exact native plan | Predicted ms | Same-batch median ms | Relative prediction error | Ratio to baseline |
|---|---:|---:|---:|---:|
| Baseline chain | 0.629754 | 0.678584 | -7.20% | 1.000 |
| N256, three singleton regions | 1.636019 | 1.988844 | -17.74% | 2.931 |
| M128, pair plus two singleton regions | 1.723164 | 1.884186 | -8.55% | 2.777 |

All 27 outputs passed the independent reference. The two transformed candidates
regressed. Their extra submissions and host boundary assembly account for
additional work; the small difference between the transformed candidates is not
claimed to be robust with unlocked frequency. No held-out or structural timing
was fed back into the frozen models.

The installed runtime does **not** automatically notice changed static weights.
The adapter binds immutable prepared weight content to explicit generation BO
keys. A separate same-runner A/B/A test changes W1's sign and changes the BO key,
then reuses the original A key; all three outputs passed. This validates explicit
generation separation, not mutation detection or SRAM residency. Probe outputs
are outside performance samples.

Final evidence is `aie-toy/results/orbit-static-v4-hardware/native-plan-comparison-final`.
Replay using a new output directory:

```bash
cd /home/shiran/project/orbit
source ../aie-toy/scripts/env.sh
python scripts/taskflow_execute_comparison_v4.py \
  --cases build/static-mapping-v4/region-calibration/comparison-cases.json \
  --iterations 9 --output-dir /tmp/orbit-v4-comparison-repro
```
