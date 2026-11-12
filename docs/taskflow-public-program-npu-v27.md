# v27: frozen public ProgramGraph optima on NPU

The four fully executable baseline/optimum pairs do **not establish a stable end-to-end NPU speedup**. Their v26 analytical speedups are not hardware speedups. The `gemm-reduction-trap` first session has a 0.833526× ratio of wall-time medians and must remain in the result. All saved output arrays passed independent numerical checks.

This experiment replays the **frozen v26 initial assignment and analytical optimum**. It does not rerun search, select a faster replacement baseline, or prove an NPU runtime optimum. “Optimal” below means the minimum of the declared finite v26 domain under its old analytical evaluator. Its exhaustive CP-SAT coverage proof and independent enumeration remain in the v26 evidence. The baseline is each domain's all-zero assignment, not a uniform default ORBIT configuration or an unchanged legacy ORBIT search.

## Measured results

The primary speedup is `median(baseline wall_ms) / median(optimal wall_ms)`, separately for each fresh-process session. The paired ratio is the median of the 30 adjacent baseline/optimal pair ratios. These statistics differ when timing distributions drift; neither is substituted for the other. Positive saved time means the optimum is faster.

| Case | Session | Baseline median ms | Optimum median ms | Ratio of medians | Median paired ratio | Median paired saved time, bootstrap 95% interval (µs) |
|---|---:|---:|---:|---:|---:|---:|
| fork-shared-intermediate | 1 | 0.353626 | 0.355426 | 0.994936× | 0.993746× | [-3.1100, 0.0405] |
| fork-shared-intermediate | 2 | 0.340251 | 0.331402 | 1.026705× | 1.004485× | [0.1900, 3.3395] |
| fork-join-residual | 1 | 0.331707 | 0.331852 | 0.999562× | 0.999451× | [-1.4750, 2.4400] |
| fork-join-residual | 2 | 0.377466 | 0.375591 | 1.004991× | 1.002822× | [-0.5195, 2.8590] |
| gemm-reduction-64 | 1 | 0.332572 | 0.333257 | 0.997945× | 1.002081× | [-1.6845, 2.4195] |
| gemm-reduction-64 | 2 | 0.316793 | 0.315553 | 1.003930× | 1.002539× | [-0.0500, 2.1890] |
| gemm-reduction-trap | 1 | 0.316907 | 0.380201 | 0.833526× | 0.994270× | [-4.3300, -0.1800] |
| gemm-reduction-trap | 2 | 0.314843 | 0.315508 | 0.997891× | 0.998271× | [-1.1855, 0.5395] |

For comparison, the corresponding old analytical speedups are 2.657277× (`fork-shared-intermediate`), 1.405348× (`fork-join-residual`), 1.191860× (`gemm-reduction-64`), and 1.000969× (`gemm-reduction-trap`). These estimates use assumed device/runtime prices and are not calibrated end-to-end predictions.

The bootstrap uses 5,000 resamples of the observed within-process paired saved times. It describes these samples, not independent days or devices. `fork-shared-intermediate` session 2 has a positive interval; `gemm-reduction-trap` session 1 has a negative interval. The remaining intervals include zero. Two processes do not establish durable speedup or slowdown. The trap first-session slowdown remains visible even though the paired-ratio median is close to one.

## All eight frozen cases

| Case | Old model speedup | Hardware status |
|---|---:|---|
| fork-shared-intermediate | 2.657277× | Both compiled, independently AIE checked, two paired sessions |
| fork-join-residual | 1.405348× | Both compiled, independently AIE checked, two paired sessions |
| shared-input-three-branches | 3.493888× | Baseline AIR compilation fails: `aie.mem` has more than 16 blocks; optimum compiled/checked and passed three unpaired correctness calls |
| fork-capacity-boundary | 2.423983× | Both fail direct lowering: static allocation sum exceeds declared 384-byte SRAM; lifetime-based physical reuse is not implemented |
| gemm-two-consumers | 1.253036× | Baseline compiled/checked; optimum fails direct lowering because cross-core dependencies lack an explicit physical synchronization implementation; no paired device timing |
| gemm-reduction-64 | 1.191860× | Both compiled, independently AIE checked, two paired sessions |
| gemm-reduction-216 | 2.936047× | Baseline AIR compilation fails: `aie.mem` has more than 16 blocks; optimum compiled/checked and passed three unpaired correctness calls |
| gemm-reduction-trap | 1.000969× | Both compiled, independently AIE checked, two paired sessions |

The capacity case needs 512 bytes of distinct static allocations although the logical liveness model admits a smaller peak; the backend fails closed rather than silently changing the capacity or aliasing allocations without a proof. The two AIR block-limit failures have 24 and 48 logical DMA operations respectively. Their failure logs are preserved. Hardware failure is not turned into a speedup by replacing the baseline. The three GEMM-reduction cases use different starting assignments/domains, so their different analytical ratios do not isolate a search-size effect.

## Execution and correctness contract

The direct backend consumes the selected public ProgramGraph: selected source instances, copies and versions, rectangles, storage owners, and core coordinates. It reconstructs source-native mapping without a new search and checks frozen source, assignment, domain, graph, and timing identities. Each physical core gets one herd. Distinct logical allocations in the same storage can occupy disjoint slices of one physical pool; packing preserves their bytes and does not implement lifetime aliasing.

The independent final-AIE checker traces finite scalar operations and actual memory addresses back to the frozen source/graph. It checks host ABI forwarding, DMA rectangles, packet/direct flow routes, allocation slices, core owners, and cyclic buffer-descriptor chains. Three-invocation semaphore progress and vector-clock checks establish the modeled refill/compute/export happens-before relations. It rejects unsupported executable operations or extra physical flows. The saved mutation controls contain one accepted original and five rejected arithmetic/owner/host-offset/lock-initialization/allocation-provenance changes. These bounded checks certify the supported implementation contract; they do not certify device cycle counts, arbitrary iteration counts, or hardware optimality.

A material mismatch remains between the analytical schedule and execution: logical groups on one core share **one physical configuration**, while the old model charges configuration setup per logical group. `analytical_group_setups_realized=false` and `physical_configurations=1` explicitly record this. Graph copy/compute realization does not imply the old absolute resource reservations or timed priorities were physically enforced. GEMM read-modify-write realization also uses an explicit local old-value staging copy (128 bytes for the optimum), recorded in the lowering manifest; the old analytical evaluator did not charge this staging. The lowering manifest's `final_aie_checked=false` is a pre-check snapshot; successful `aie-check.json` is the separate final checker receipt.

Each of the four paired cases has two fresh processes. Each process runs seeds 11, 23, and 47, with three warmup pairs and ten measured pairs per seed. Order alternates within pairs and flips between sessions. There are 240 measured pairs (480 measured calls) plus 144 warmup calls: **624 paired-protocol calls, 780 full output arrays, 1,560 FP32/FP64 comparisons**. Two unpaired optima add six calls, 12 arrays, and 24 comparisons. Overall: **630 calls, 792 arrays, 1,584 full-array comparisons**, all passed with `atol=rtol=1e-5`. Unpaired diagnostics contribute no speedup observations.

Every call stages NaN output buffers before device execution. Full outputs are retained as NPZ files and checked against sequential FP32 source references and independent FP64 formulas. The independent audit also verifies saved error values, identities, ELF linkage, actual alternating call order, warmup exclusion, all timing summaries, and deterministic bootstrap intervals. Maximum errors against FP32/FP64 respectively are about 1.19e-7/1.71e-7 for the fork, 2.38e-7/3.43e-7 for the join, and 5.96e-8/9.02e-8 for the two paired GEMM cases.

## Timing boundary and model implications

The wall-time window includes host input/output-canary staging and buffer synchronization, run creation/binding/start/wait, output synchronization/readback, and output reshape/copy. Context loading, reference computation, numerical validation, and result serialization are outside the window. There are no device-cycle measurements or locked device frequencies. NPU runs were serialized under `/tmp/npu.lock` with bounded external timeouts.

Measured warm submit-to-completion medians are about 0.262–0.276 ms, while total wall-time medians are about 0.315–0.380 ms. The submit/wait bracket contains device execution **and** runtime/host completion behavior; it must not be treated as an additional serial host cost on top of a separately counted device execution duration. These measurements do not identify pure launch latency. The protocol did not record the selected native-wait versus polling implementation, so it does not establish that polling caused the bracket duration.

The results require an execution-region boundary model for heterogeneous allocation: charge launch per actual invocation, transfer per actual cross-region copy, and completion/visibility according to dependencies and runtime policy. Multiple input edges of one NPU invocation must not multiply its launch charge. Conversely, separate invocations on the same chip still require distinct launch boundaries. The v27 data exposes the gap; this report does not claim that a later calibrated model or heterogeneous joint search has already been completed. No unfinished v28 result is included here.

## Frozen evidence and reproduction

- Raw immutable run: `/home/shiran/project/orbit-artifact/results/taskflow/public-optimum-npu-v27-r1`.
- Curated evidence: `/home/shiran/project/orbit-artifact/taskflow/evidence/public-optimum-npu-v27`.
- Source finite oracle: `/home/shiran/project/orbit-artifact/results/taskflow/optimality-reference-v26-r1`.
- Frozen compiler SHA-256: `2b2f5d8cead4c5410f5290358cdc73882361f1574b3b8f1dc90f4f2a029b366c`.
- Protocol SHA-256: `ce1cde3703fb8b3cc5d475d49e03ac0c6b558880e90577fd1b08e7e212691e81`.
- Producer script SHA-256: `a8da848cd051860d55161bfc88455a53be74b2d24487e75efd152ac3831c6f51`.

The curated tree preserves every formal raw file except the approximately 204 MB compiler executable. It retains full saved arrays, timing samples, frozen evaluations/assignments/source/configuration, lowered AIR, final AIE, hardware products/logs, checker receipts, independent audit, mutation controls, and the exact producer script. The compiler remains at the raw path and its size/hash are in `raw-inventory.json`. `manifest.json` hashes every curated payload file and gives its original absolute path where applicable; `raw-inventory.json` records every raw file including omitted payloads. The manifest excludes itself to avoid recursive hashing. Earlier failed pilots and all v26 evidence remain unchanged at their original paths.
