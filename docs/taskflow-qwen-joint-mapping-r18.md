# Qwen executable joint mapping, r18

This experiment extends the r17 full Qwen operator graph with source-bound,
compiled alternatives. It uses the original ORBIT C++ beam search, without a
wall-time, evaluation-count, or round-count cutoff. Neighborhood convergence
does not prove global optimality.

The input remains the original Triton-XDNA Qwen2.5-3B-Instruct Q4_0 inference:
36 layers, hidden size 2048, FFN size 11008, five prompt tokens, and 128 generated
tokens. The complete typed source contains 138759 operators, including attention,
KV state and the 127 recurrent decode invocations. Python executes registered
backends; it does not choose the winning graph or substitute whole-plan prices.

## Measured result

The checked medians, raw sample timings and evidence hashes are also available
in [the machine-readable results](taskflow-qwen-joint-mapping-r18-results.json).

ORBIT independently selected the same execution configuration as the previous
manual plan: all 144 prefill matrix multiplications on NPU, the other 437
prefill operators on CPU, and the complete coalesced NPU decoder. No new FFN
fusion, GEMM tiling or replica implementation beat that configuration under the
registered complete-call prices. Searching a dimension does not require its
new alternative to win.

Three fresh processes each performed one warmup and three complete 128-token
requests, with the same routed runner, four CPU threads and preparation policy.
The table reports medians, excluding compilation and preparation.

| Configuration | Prefill / TTFT (ms) | Full request (ms) |
| --- | ---: | ---: |
| ORBIT selected | 392.185005 | 6682.586993 |
| All supported prefill operators on NPU | 640.789651 | 6988.365615 |
| Previous manual configuration | 418.079803 | 6741.432807 |

Against the all-NPU baseline, the selected plan is **1.633896×** faster for
TTFT and **1.045758×** for the complete request. Against the manual baseline,
the measured ratios are 1.066027× and 1.008806×, but the execution choices are
identical. Those differences are run variation, not a newly discovered
algorithmic improvement.

All nine measured requests cover 581 prefill source operators and 127 complete
decode invocations, conserve their exclusive timing boundaries, and pass the
original memory guards with zero process swap. The selected full 128-token
logits and tokens are bitwise identical to the fresh manual baseline, the old
r17 manual capture and r15 tuned capture. All seven original mutual-top5 checks
pass. The HF reference contains 32 tokens: the selected stream first differs
at token index 5 and passes the original gate; it is not bitwise HF equivalence.

The frozen prediction was 6722.891381 ms. Measured minus predicted is
−40.304388 ms, or −0.59951% of predicted time. No heldout timing was fed back
into the prices or selection.

## Search completion and cost

The completed search evaluated **1,366,914** assignments over **110** rounds and
stopped because the beam was unchanged after checking its legal neighborhood.
There was no time, evaluation or round cutoff. The beam took **64.033 s**; the
reported total including source parsing, certification and final graph checking
was **90.174 s**, before writing the final result file. Offline artifact
compilation and the 15 calibration cohorts are additional work.

The catalogue has 2,716 priced implementations and 182 variable groups: 36 FFN
groups with 76 choices each, 72 QKV/output groups with seven choices each, and
74 two-choice normalization/decoder groups. An independent audit of the actual
first 3,207 evaluation rows witnesses **every registered local choice**. This
is stronger than merely listing configurable fields, but does not enumerate
their Cartesian product.

The selected graph passes independent full physical checking and agrees with
the compact evaluator. A post-search calculation also matches the unconstrained
local-fee lower bound within 9.1e-12 ms. That certifies this selected point for
the registered serial cost model; it is not global optimality over unsupported
transformations or actual hardware behavior.

The process observer recorded a kernel RSS high-water mark of approximately
**9.73 GiB**, with zero process swap. It started after launch and may miss a
short-lived peak after its last sample; raw observations retain that limitation.

## Executable search choices

| Dimension | Registered change | Validation boundary |
| --- | --- | --- |
| CPU/NPU placement | Original projection, normalization and SwiGLU implementations | Original complete-source synchronous calls |
| Fusion/fission | Four original FFN operators versus one firmware-sequenced dispatch; combined gate/up projection versus independent projections | Same BF16 projection drains and original NPU SwiGLU; intermediate BOs are in DDR |
| Tiling and resource size | Actual M/N/K tiling variants using different physical herd shapes | Compiled AIR, actual AIE buffers, cores and ELF |
| Replica and spatial placement | Two independent output-N shards on disjoint physical cores; column groups 0–3 or 4–7 | Explicit disjoint logical output partitions and hardware numerical gates |
| Temporal order | Independent gate then up, or up then gate | Two actual FFN ELFs; original source dependencies remain elsewhere |
| Dataflow | Two distinct complete decoder instruction streams, plus registered GEMM/FFN dataflows | Complete 36-layer decoder and all 25 vocabulary waves |
| Data residence | FFN prepared shared BOs, private BOs with repeated uploads, or BO release after each call | Actual allocation identities, transfer policy and lifetimes |

These are finite executable alternatives, not every imaginable point in each
dimension. The runtime retains its synchronous source traversal. It cannot
execute arbitrary CPU/NPU overlap, arbitrary graph fusion, or independent
resource edits without a registered lowering. Those capabilities remain
unsupported rather than receiving an invented speedup.

The Qwen `PROJ_RC_CACHE` switch does not affect the unpaired projection kernel
used here. Cache0/cache1 compile to the same device program and are not separate
residence choices. Four column-rotated decoder candidates compiled and returned
completion but failed logits/KV comparisons; they are rejected. Compilation or
completion alone is not numerical admission.

## Cost and capacity

Training uses complete-model executions with one warmup and three measured
requests. Exclusive source-hook and decoder-region wall times include actual
launch, context opening/closing where applicable, staging, synchronization and
readback. A source cover owns its inclusive transport costs; the evaluator does
not charge them again on its edges. Decoder host-before, device service,
host-after, argmax and invocation control have separate exclusive boundaries.
Every training request must conserve its observed total wall time exactly.

The common evaluator checks the selected resource calendar and the combination
of ABI allocations, prepared allocations and invocation scratch. A fused FFN
replaces the original readonly weight representation with three checked padded
BF16 fragments; it does not retain and count a fictitious second source copy.
Shared physical allocation IDs are counted once. Replicas require disjoint exact
coverage of the actual logical outputs.

GEMM and decode SRAM use compiler buffer/stack ledgers. Original NPU and FFN
implementations currently reserve the whole array SRAM as an explicitly
conservative bound. The runtime-ABI budget is a 20 GiB admission quota, not a
claim about physical DRAM size. Framework/driver object overhead is outside this
managed-buffer model. The unchanged 20 GiB RSS and 0.5 GiB process-swap guards
therefore remain mandatory during actual execution.

One full-36-layer reload/up-gate calibration failed the swap guard during
warmup (974413824 bytes). Its failure and memory trace are retained. That local
implementation is instead calibrated in three complete-Qwen cohorts with 12
selected FFN layers each. The other layers execute original NPU implementations.
This samples every source layer without admitting the failed complete plan.

## Reproduction and evidence

The implementation entry points are in `experimental/qwen-runtime-search/`:

- `prepare-r18-campaign.py` freezes complete-model training plans.
- `npu-gemm-variants.py` and `decode-variants.py` emit and compile actual variants.
- `register-r18-{gemm,ffn,capacity}.py` bind source, physical resources and storage.
- `fit-r18-catalogue.py` admits conserved training hook prices and exact covers.
- `program-operator-search.cpp` invokes common checking and the ORBIT beam.
- `replay-r18-selection.py` translates the actual selected common graph.
- `routed-r18-runtime.py` executes per-layer FFN policies over the frozen runtime.
- `execute-r18-prepared.py` verifies sealed files before heldout execution.

Machine-local raw evidence is under
`/home/shiran/project/orbit-artifact/results/taskflow/` in
`qwen-whole-graph-r18`, `qwen-compute-variants-r18`, and `qwen-full-dim-r18`.
The original failed campaign is retained separately from its admitted training
campaign. Heldout results must not feed the training prices.

The first all-NPU heldout attempt failed the swap check after preparing weights,
before the runner's existing allocator-page release. The owned routed adapter
now performs collection and `malloc_trim` before returning from preparation,
so the same guard observes the prepared live state. All three final cohorts
were rerun with that same wrapper. For example, selected preparation RSS drops
from about 16.14 GB to 7.37 GB before the unchanged guard. This preparation work
is outside every request timer. The failed attempt and earlier selected samples
remain separate from the final comparison.

Final evidence filenames under `qwen-whole-graph-r18/` include:

- `search-launch-r18-v4.json`, `orbit-joint-r18-v4.json` and both complete JSONL traces.
- `search-choice-coverage-r18-v4.json` and `search-memory-r18-v4.summary.json`.
- `selected-program-graph-r18-v4.json` and `heldout-selection-seal-r18-v2.json`.
- `heldout-selected-v5-prepared/` and the two `heldout-baseline-*-v4-prepared/` directories.
- `heldout-audit-r18-v2.json` and `heldout-model-error-r18-v2.json`.
- `diagnostic-mixed-routes-v1-prepared/diagnostic-audit.json`.
- `common-mixed-lowering-probe-v1-audit.json` and its sealed prepared execution.
- `native-final-regression/summary.json`, `streaming-beam-tests.log` and `streaming-mapping-tests.log`.

The common CPU contract suite currently passes 52 checks, including packed
weight bounds/coverage, replica overlap/gaps, unknown constraints and final
projection/physical-graph agreement. Standalone hardware admission passed all
18 GEMM artifacts (54 calls); the valid nonrotated decoder variants matched both
logits and the complete physical KV readback in controlled three-step probes.
These gates do not constitute full-model performance results.
The beam suite additionally passes 35 checks, including retained/streamed trace
equivalence and sink failure; the three final native regressions pass.

A real mixed-path diagnostic additionally executes all six FFN policy/order
combinations and all 18 GEMM artifacts in one 36-layer, four-output request.
It passes exact source coverage, ownership, memory and original numerical
checks. Its four tokens match the first four HF tokens. This diagnostic is not
a steady-state timing sample and does not replace the full heldout results.

A final validation-only probe forces one existing choice per group, preserving
all prices and binding the parent catalogue hash. The actual C++ checker lowers
its six FFN routes and 20 GEMM calls covering all 18 artifacts into the common
physical graph; the unchanged replay bridge translates packed weights and
replica work partitions. One warmup and three complete 128-token requests pass
source coverage, exclusive phase conservation, BO ownership and the original
memory guards. Observed process swap reaches 231411712 bytes, below the original
0.5 GiB limit. Both original numerical gates pass. This mixed configuration first
differs from the selected token stream at index 96 and is not bitwise equivalent;
its same-context logits have maximum absolute error 0.6875. This is integration
validation of nonwinning alternatives, not another search or speedup result.

The serial-calendar evaluator is audited against its original per-invocation
reference on 100 actual search assignments. Both feasibility and unknown-status
results agree; the maximum cost difference is 5.46e-12 ms. The audit measured
9.531 ms for the compact evaluator and 1331.133 ms for the reference over those
100 assignments. Only immutable source/ABI/resource facts are cached; each
assignment still recomputes joint physical allocation and scratch feasibility.

An initial optimized run reached more than one million evaluations before its
process was killed with exit 137. Monitoring showed sharply growing retained
JSON trace memory. The beam now optionally streams every complete evaluation
row to JSONL instead of retaining a second JSON assignment archive in RAM.
Assignment deduplication, candidate ordering, unknown-state exploration and
convergence remain unchanged and are compared in regression tests. Both the
failed run and its checkpoints are retained; it is not a completed search.
