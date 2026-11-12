# v29: measured validation of complete CPU/NPU region costs

The complete region model passes the preregistered held-out accuracy and ranking checks on the existing three-task source family. Across 36 held-out plan/shape groups, median absolute relative error is **5.3493%**, p90 is **19.6919%**, and all **247/247** eligible pairwise orderings are correct. CP-SAT selects the measured fastest plan at both held-out shapes, with zero selection regret.

The winning plan is one all-CPU region. Relative to the declared all-zero/all-CPU baseline, measured speedup is **1.0× in every process**. This validates the placement decision and model benchmark; it does not demonstrate acceleration over that baseline. Extra all-NPU references are much slower because these small programs do not amortize NPU invocation costs.

## Complete invocation ownership

`ProgramRegionCompositeCost` prices each execution region as one exclusive complete implementation. An NPU region includes bound-BO TO synchronization, run creation/argument binding/start, physical device setup, source body and internal communication, teardown/reset, native completion wait, and output FROM synchronization. A CPU region includes its actual compiled CPU function and prepared ctypes dispatch. These are observed callable costs, not separately identified device-cycle, firmware-setup, DMA-bandwidth, or launch-only measurements.

Application input reset, serial orchestration/control, and final output copy have separate single owners. Enclosed graph events and lifecycle events belong to the region composite; graph compute/DMA, setup/teardown, or a full submit/wait duration are not added a second time. This avoids the v28 omission of physical setup while also avoiding double counting. Multiple inputs to one NPU region still cause one launch; a same-chip cut creates another real invocation.

The runtime uses caller-owned mapped BOs shared with the compiled CPU ABI. It caches descriptors, by-reference arguments, BO mappings, operand lists, and kernels before timing. Each NPU invocation constructs/binds a raw XRT run, starts it, and calls native `pyxrt.run.wait2()` with no polling fallback. The executor is serial; direct wait may hold the GIL, and asynchronous CPU/NPU overlap is outside this contract. Its source performs no global Python API monkeypatching. Native wait2 versus old v28 polling is a runtime change, not an optimizer speedup comparison.

## Frozen domain and training separation

The source computes `T=A+B; U=2*T; C=U+B` in FP32. Training lengths are N=16, 256, and 1024; untouched validation lengths are N=64 and 512. Each length has 32 raw CPU/NPU placement/cut assignments and 18 canonical execution-region graphs. Redundant cuts at a chip change alias the same graph. Every canonical plan is compiled and measured; the validator does not measure only a few selected candidates.

The preregistered model has eight source/ABI-derived features: call count, ABI BO count/bytes, source operations, source memory accesses, task count, transfer count, and requested transfer bytes. Models are grouped by implementation, source-semantic family, and runtime control family. The NPU control family includes the number of prepared NPU contexts, a known manifest property fixed before fitting. It is not a timing label or a whole-plan identity lookup. Nonnegative affine coefficients are fitted to normalized columns using training-shape/plan/process medians; correlated coefficients are used for prediction and are not interpreted as independently identified physical mechanisms.

The frozen training manifest lists exactly 162 timing files: 54 training plans × three processes. An independent rerun places a `Path.open` guard on all N=64/512 timing paths, opens none of them, and reproduces the model bytes exactly. C++ search receives that frozen model; no held-out wait or measured candidate cost supplies a search score. Independent coefficient/feature dot products match every C++ full-plan price, and complete enumeration matches each CP minimum.

For all five lengths, CP-SAT checks all 32 raw assignments, with known feasibility and cost, and reports finite-domain `OPTIMAL`. Total search takes approximately 858–990 ms per length, including 18–21 ms in CP-SAT and 838–968 ms in evaluation callbacks. Recorded process-lifetime peak RSS is 226212 KiB (about 220.9 MiB), not memory attributable exclusively to CP-SAT. Every domain has only five binary choices (15 model variables, 52 constraints); these timings do not establish scalability to large domains. This is optimality under the calibrated model within the declared domain. Independent measurement of all 18 canonical plans supports the selected hardware winner at the two held-out lengths. Exact solver analysis and production source snapshots are retained at raw `raw/source-r1` and `raw/source-r2`; the extra `raw/` path component is intentional.

## Hardware and numerical evidence

Three fresh-process rounds use seeds 11, 23, and 47 and independently randomized orders of all 90 shape/plan groups. Each group has three warmups and 15 hot calls per process. There are **4860 calls and complete output arrays**, including **4050 hot calls** and **1620 held-out hot calls**. The recorded region counts correspond to **5670 NPU region invocations** across the full run.

Inputs are regenerated deterministically for each repeat and shared logically across candidate plans. Every call poisons non-input actual backing arrays with NaN outside its inner clock, consistently in training and validation. Outputs and timings remain buffered through a whole plan batch; numerical checking and evidence serialization happen after that batch and context unloading. Saved outer envelope timestamps include clock-excluded poisoning/bookkeeping and permit exact outer-gap checks; inner `wall_ns` measures input reset through final output copy.

The independent audit verifies all saved outputs against sequential FP32 and independent FP64 source formulas: **9720 full-array comparisons**, all passing at `atol=rtol=1e-5`. Maximum absolute errors are 2.3841858e-7 against FP32 and 2.0861626e-7 against FP64. It also verifies generated inputs and saved maximum errors, compiled source/library/ELF identities, runtime closure, BO address certificates, all raw-to-canonical aliases, randomized orders, nonoverlapping region spans, exact exclusive cost arithmetic, and saved outer gaps. The auditor executes no NPU work.

## Held-out gates and their limits

The accuracy unit is a **(N, canonical plan) group**. Its measured target is the median of the three process hot medians. The gate requires the median group relative error ≤10% and the linear-interpolated p90 ≤20%. It does not require every group, process, or individual call to be within 20%. The observed p90, 19.6919%, is close to the acceptance boundary.

Four groups exceed 20% error, retained without exclusion:

| Group | Absolute relative error |
|---|---:|
| N=64, CPU/CPU/CPU, cut after task 2 | 20.5709% |
| N=512, CPU/CPU/CPU, cut after task 2 | 27.8158% |
| N=512, CPU/CPU/CPU, cut after task 1 | 30.0002% |
| N=512, three separate CPU regions | 30.3531% |

Process-specific errors across the same 36 held-out groups are:

| Process | Median relative error | p90 relative error |
|---|---:|---:|
| Seed 11 | 7.3034% | 20.2330% |
| Seed 23 | 7.4970% | 19.6919% |
| Seed 47 | 9.3578% | 24.9714% |

These process-level p90 values are reported, not substituted for the preregistered group-level target. Per-group/per-process raw errors are retained in `independent-validation-r1/heldout-groups.json`.

For pairwise ranking, two plans at the same held-out N are eligible only if their group medians differ by at least 5%, normalized by the smaller median, and the ordering has the same nonzero sign in all three processes. All 247 eligible pairs are predicted correctly; exact predicted ties would count as wrong. For selection, measured regret must be ≤5% at each held-out N. Both selected plans have zero regret and are the only plan within 5% of the corresponding measured best.

## Baseline and reference comparisons

| Held-out N | CP prediction for selected plan (µs) | Measured selected group median (µs) | Measured selection regret | Primary baseline speedup by process |
|---|---:|---:|---:|---|
| 64 | 2.630608 | 2.420 | 0.0% | 1.0×, 1.0×, 1.0× |
| 512 | 3.357612 | 2.930 | 0.0% | 1.0×, 1.0×, 1.0× |

The primary baseline and selected plan are the same all-CPU/no-cut assignment, so the three-process ratios are exactly one. No stable acceleration over the primary baseline is claimed.

Additional reference ratios use each process's reference hot median divided by its selected-plan hot median, then take the median of those three ratios:

| Held-out N | One all-NPU region / selected | Three NPU regions / selected | CPU→NPU→CPU / selected | NPU→CPU→NPU / selected |
|---|---:|---:|---:|---:|
| 64 | 75.5159× | 241.0255× | 79.0259× | 157.1873× |
| 512 | 78.0709× | 203.8686× | 73.6936× | 139.9860× |

The 75.5×/78.1× comparison against one all-NPU region reflects avoiding costly NPU invocations for this small workload. It is an additional reference, not a replacement baseline or evidence that NPU execution itself became that much faster. The saved summary retains every process ratio and reference latency.

## Reproduction, validation, and scope

The formal record is `/home/shiran/project/orbit-artifact/results/taskflow/region-validation-v29-r1`; curated evidence is `/home/shiran/project/orbit-artifact/taskflow/evidence/region-validation-v29`. Original failed/diagnostic pilots and all v28 records remain unchanged.

- Final measurement protocol: `invocation-regions-v29-r2.json`, SHA-256 `3cb2e1b4afdb490f92baa7fe8db74222a44b592effd6b3fe4f163f3230bfb0d0`.
- Actual executor `region_executor_v29_r2.py`: `c2097b3a80eb52eb8336fe83e458608b761304691d78655529b7434db40f82b7`.
- Batch producer: `1609b742eeb5eadd85636846aec4962da4a29dbf7dbc8c1737cc8db979c6e7d5`.
- Frozen training-only model: `450b87c994ccf50aa1d53d452b0e2f363b1435621bd4fd75b196d3b5cd9fdb72`.
- Plan-export compiler, preserved at raw `tools/taskflow-opt`: `f57eac3b207dc8678b89d01ce01e6b2cd5003f2d196d543454c049ace1605cd0`.
- Feature/search compiler, independently frozen for v29: `0630e946288ffe36b3ae2c73d952750e4f27e3d0346becd56c8acf35e428c281`.

The old preparation helper's v13 execution contract remains bound for preparation/ABI compatibility. The actual timed runtime is separately bound to the new executor and batch source above. The initial CLI pilot mislabeled a nanosecond summary as milliseconds while retaining correct raw `wall_ns`; that pilot and correction are preserved, and formal batch units are independently verified. Protocol metadata amendments did not change numeric gates, training/held-out lengths, features, or fit method.

One protocol item was recorded late: the host timer-pair diagnostic was collected **after validation**, with three 10000-pair batches giving 60 ns medians and 70 ns p90. It is a post-validation diagnostic, not the timing calibration available during measurement. It was not subtracted, fitted, or used to alter gates. This deviation is retained explicitly rather than asserting complete preregistration compliance.

All final ON/OFF regression commands pass. Coverage includes 48 region-composite checks in each build, 96 invocation checks in each, 58/56 heterogeneous-search checks, 227/192 CLI checks, the existing public-search/identity/replay contracts, and five ON public-graph AIR cases. Exact commands, reported checks, return codes, and logs accompany the evidence.

The curated tree retains numerical arrays, raw timing/identity records, frozen model and training provenance, feature/search receipts, independently reproduced audits, protocol revisions, relevant executable dependencies and selected source snapshots, plus manifests with original references and SHA-256. Large compiler binaries and redundant generated build products can be referenced by verified raw-path hashes rather than copied. It is an evidence archive, not a complete standalone toolchain distribution.

This benchmark covers the same actual three-task source family, two device types, serial resident execution, and 32 placement/cut choices. It does not jointly search all native tiling, retention, buffering, fusion/fission, routing, shape, or replica dimensions, and does not establish a universal NPU or inference optimum. The measured winner agreement and passing prediction/ranking gates are restricted to this declared executable domain and runtime contract.
