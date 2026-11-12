# Whole Qwen operator graph search (r17)

This experiment connects the original Qwen2.5-3B-Instruct Q4_0 Triton-XDNA/AIR source to ORBIT through a typed external operator source. It includes all 36 prefill layers, the KV handoff, and 127 autoregressive decode steps for a 128-token output. External source and artifact hashes plus tensor, view, mutable-state, and implementation-boundary contracts establish the source binding; this is not a native MLIR arithmetic proof.

The source contains 138,759 operator instances. The executable search domain has independently selected CPU/NPU implementations for 253 supported prefill operators (144 matrix multiplications, 73 RMSNorms, and 36 SwiGLUs), plus a choice of actual compiled decode wave partitions at each of 127 steps. Other source operations remain explicitly represented with their supported implementation. A logical graph partition does not create a fused kernel: a decode cover is admitted only when its original AIR wave-range implementation exists and its full-model ABI and initialization artifacts match.

Per the user's instruction, this full graph uses ORBIT beam search without a fixed wall-clock, evaluation-count, or round limit. Search terminates when the selected beam has no unexpanded legal neighbors. This is a feasible-search result, not a global optimality certificate. CP-SAT and ILP are not run on the full graph.

## Execution and cost boundaries

Each request uses the original Qwen prefill body and decoder dispatch, with checked implementation selection. Prefill NPU calls retain their actual launch/synchronization/readback behavior. Each compiled decode region owns its instruction patch, submission, and completion time; adjacent decode regions share the original X, W, RMS, Y, and KV buffer objects. Splitting the decode does not add a fictitious KV copy. Mutable KV states and physical views retain their shared allocation and source ordering.

Host work is measured explicitly, including context clear, embedding and other source operations, KV handoff, decoder preparation/readback, argmax, result capture, and request control. Inclusive implementation durations already include their launch/transport overhead; those costs are not charged again as extra edges. Decode host completion and dispatch control are priced per step and per partition, and selected together with their matching device regions. Remaining prefill and outer application control is a separately reported constant approximation, rather than distributed into unmeasured source operators; heldout model error tests that approximation. Missing implementation prices remain unknown.

The replay uses the same tensor provenance and source profiling instrumentation for baseline and selected plans. Compilation, model loading, weight preparation, and file writing are outside the request timer. Each process has a 20 GiB RSS and 0.5 GiB process-swap guard. CPU-selected projections keep the original BF16 tensors; NPU-selected projections use the original exact-K resident-weight preparation with bitwise weight validation.

## Calibration completed

Evidence root: `/home/shiran/project/orbit-artifact/results/taskflow/qwen-whole-graph-r17`.

Three independent fresh processes ran a five-token prompt, one warmup, and three measured complete 128-output requests each. These are training measurements, not a selected-plan performance comparison.

| Training plan | Median TTFT (ms) | Median complete request (ms) |
|---|---:|---:|
| CPU prefill, whole decode | 1332.948 | 7547.040 |
| All supported prefill operations on NPU, two decode regions | 639.164 | 6990.332 |
| Prefill matmul on NPU and small operations on CPU, three decode regions | 469.813 | 6980.450 |

All three pass the original first-32-token mutual-top-5 stream gate against the prior complete source run and the HF BF16 reference. The matmul-NPU/small-CPU capture also matches the previous tuned source run bit for bit across all 128 output logits and tokens. The other two diverge at token index 2 while passing the original numerical gate; they are not claimed to be bitwise equivalent. Post-divergence logits are not compared as if they shared the same context.

The training protocol, frozen runtime files, commands, process exit receipts, phase traces, captures, and independent numerical audit are retained in the evidence root. The final selected-plan measurements below were collected after freezing the selection and all replay commands.

## ORBIT search and selected plan

The formal v2 run uses the actual existing `searchProgramSpaceBeam` driver: beam width 4, 380 nontrivial dimensions, 294,809 evaluated candidates, and 149 rounds. It stops because the beam is unchanged after expanding its legal neighborhood. The driver took 12.482 seconds; the complete CLI took 41.933 seconds including source/certificate preparation, final materialization, checking, and JSON output. No wall, evaluation, or round budget was set. Structural table and memory controls remain explicit in the report. The earlier v1 run took 10.916/30.345 seconds; it is preserved separately and is not the formal heldout binding.

The selected plan puts all 144 prefill matrix multiplications on NPU and all 73 RMSNorms and 36 SwiGLUs on CPU; the remaining explicit prefill nodes use their supported CPU implementation. Decode selects 124 whole dispatches and three two-region dispatches. The resulting request has 274 actual NPU launches, versus 380 for the fixed all-supported-NPU baseline and 271 for the previous manual plan. The three split choices come from training estimates; they do not establish that splitting is intrinsically faster.

`orbit-whole-search-v2-root.json` contains the source, evaluation history, selection, and independently checked common physical ledger. `search-summary-v2-root.json` is a small report. The selected common graph ID is `83ff8bf2783f0d534626f1cf5bc774858c43e0cc3a9dc78fec15e8f5c8c4798e`; its source ID is `a80070e7f879e074ce803d4e67dd4fa1f2f131505cad1b3abeb6687dd0a831fe`.

## Complete inference board validation

All three heldout configurations use the same frozen original-source runtime, source profiling, per-step selector, five-token prompt, 128 generated tokens, capacity 2048, four CPU threads, static exact-K prefill weights, one warmup, and three measured requests. Each configuration runs in its own fresh process. Compilation and preparation are excluded from the following request timers. The two baselines have explicitly manual replay protocols; they are not represented as solver-selected ProgramGraphs.

| Heldout configuration | Median TTFT (ms) | Median 128-output request (ms) |
|---|---:|---:|
| ORBIT selected complete graph | 445.913 | 6818.570 |
| Fixed all-supported-NPU prefill, whole decode | 609.142 | 6899.411 |
| Previous manual matmul-NPU/small-CPU prefill, whole decode | 447.634 | 6753.848 |

Against the fixed NPU baseline, ORBIT is **1.366x on TTFT** and **1.0119x on the complete request**. Against the previous manual plan, it is 1.0039x on TTFT and **0.9905x on the complete request**, meaning the complete request is about 0.96% slower. ORBIT automatically recovers the important placement choice; this measurement does not establish a further gain over the manual plan. The small complete-request difference is within the observed request variation; three requests are descriptive evidence, not a statistical significance claim.

Every heldout process exits successfully and passes its unchanged memory guards. Actual per-step region IDs and positions match the sealed selection, and complete request, prefill, and dispatch timing partitions conserve their measured wall time. Selected output matches the fresh manual baseline and the previous r15 tuned reference bit for bit across all 128 tokens and all logits. All three heldout configurations pass the original first-32-token mutual-top-5 gate against HF BF16. That gate is not a claim of all-logit BF16 equivalence.

The selected model predicts 6634.532 ms versus 6818.570 ms measured, an underestimate of **2.70%**. Launch, wait, readback, and partition-dependent control were measured; remaining prefill/application control is still a constant approximation. Timing noise and local-cost composition remain limits of this model.

The selection and commands were frozen in `heldout-selection-seal.json` before any heldout device feedback. `heldout-audit.json` records bindings, per-sample timings, numerical gates, invocation counts, and model error. `run-heldout.py` and `audit-heldout.py` preserve the actual collection and independent audit code. `model-tokenizer-receipt.json` records the original model hash and tokenizer files. The model SHA-256 remains `c2f1dcb6fe1e819cc50547f091096f4849890b2fb170d031ff7d790ea0c5e2ce`.

## Code validation

The rebuilt common library, operator search CLI, and native consumers compile with LLVM formatting applied to the new C++ code. CPU contracts pass: 29 beam-search checks, 14 external source/ledger checks, and eight selected-decode tests. Fresh native graph, invocation-cost, and graph-search regressions also pass. External source tests are included in `check-taskflow-public-search`. Earlier adapter tests and local-control-domain tests are documented with the r16 experiment. All board runs are serial; compile/test processes are stopped during formal device measurement.

The search covers the complete represented graph within its registered executable implementation domain. It does not introduce arbitrary new kernel tilings or permit independent launch of internal fused-kernel operators. Further decode acceleration requires additional executable kernel implementations or transformations, not simply a longer run of this converged domain.
