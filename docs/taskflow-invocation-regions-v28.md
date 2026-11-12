# v28: execution regions, invocation boundaries, and heterogeneous search

The corrected common ProgramGraph model represents each invocation region's launch, device setup, body, device teardown, completion, and data visibility costs, and exposes a joint finite search over CPU/NPU task placement and region cuts. A region owns one launch; its data edges own their transfer/visibility work. **The hardware oracle is not established:** actual device setup/teardown durations remain unknown, so all eight current formal frozen-candidate costs are unknown. Earlier R1 finite-domain proofs belong to a superseded diagnostic profile that omitted these mechanisms, and do not certify the corrected complete model. CPU dispatch and body compute/DMA are also not fully calibrated.

## Region and cost ownership

The implementation reuses `ProgramGraph.executionRegions` and existing submit/complete/visible-to-device/visible-to-host lifecycle events. It does not add a competing program abstraction. A change of chip introduces a region boundary. A cut can also split two invocations on the same chip. Two inputs entering one NPU region cause two buffer operations and one region launch; splitting that region into three real calls causes three launches. A logical execution group does not by itself imply a host call or physical reconfiguration.

`ProgramInvocationCost` derives the region's ABI allocations, unique inputs/outputs, scratch allocations, staged bytes, and to/from-device bytes from the checked graph. It prices per-call, per-BO, and per-byte terms with explicit runtime identity and provenance. Scratch is initialized outside the prepared-hot boundary and is not restaged each hot call. Optional application input reset is separate from unbound BO staging. Final application output copying has a separate owner.

`ProgramInvocationEvaluation` combines these lifecycle costs with body event costs through the existing scheduler. Body/lifecycle coverage is disjoint. The prepared-hot boundary excludes context/ELF/initial BO loading and stops charging the old logical-group configuration setup. The runtime contract explicitly declares bound versus unbound BOs, output staging, application input reset, and final output copies. A profile from the instrumented unbound runner cannot be applied to a bound-buffer heterogeneous runtime by merely renaming its identity.

For the declared polling completion policy, let `B` be the modeled interval after submit and `Q` the effective sleep quantum. The completion event adds only `max(0, ceil(B/Q)*Q-B) + P`, where `P` is the declared probe overhead. Thus the body plus detection tail is counted once. This is a conditional polling-phase estimate, not a measured device clock or an OS wakeup guarantee. Missing body/quantum/probe terms preserve unknown total cost. Adding the entire measured submit/wait bracket on top of separately counted device work would double count device time and is explicitly avoided.

Prepared ELF/context objects do not prove that physical device setup is outside each invocation. The generated AIE `main` runtime sequence includes `aiex.configure` and `aiex.run`; one physical configuration in the manifest does not imply zero per-run firmware configuration/cache-check work. That internal setup/readiness cost has not been independently identified or calibrated in this profile. Removing a setup charge per logical group therefore does not justify treating actual per-invocation physical setup as free. This is another unresolved boundary within the device-side estimate, alongside assumed compute/DMA costs.

The CPU/NPU adapter jointly searches task-device choices and invocation cuts through `ProgramHeterogeneousSearch`, the existing heterogeneous mapping builder/checker, and the public finite-domain search engine. The current common checker requires a global linear lifecycle, and the heterogeneous adapter uses serial resident regions. It does not implement asynchronous CPU/NPU region overlap or general parallel heterogeneous scheduling. Its three-task contract has 32 raw assignments and 18 unique region graphs, because a chip change already forces a boundary and a redundant explicit cut aliases that graph. Unsupported mappings stay unknown rather than being declared physically infeasible.

This adapter currently builds resident whole-region implementations. **It does not yet jointly enumerate CPU/NPU placement/cuts with every tiling, retention, buffering, fusion/fission, routing, and resource-shape dimension in one domain.** The v26 native finite domains and this new heterogeneous finite domain are separate search adapters. A complete hardware oracle requires composing those choices with executable legality and adequately calibrated costs.

## Calibration data and limits

The immutable calibration run is `/home/shiran/project/orbit-artifact/results/taskflow/invocation-calibration-v28-r1`. `calibrate_invocation_v28.py` uses existing v27 ELF/ABI products without compilation or a new solve. The main agent ran NPU phases sequentially under the existing lock and external timeout. `fit_program_invocation_v28.py` performs a separate offline full-array/event audit and exports `program-fit-r2/model.json` and `target.json`.

Memory calibration uses independent XRT BOs at training sizes 32, 256, 4096, and 65536 bytes; held-out sizes are 128, 1024, and 16384 bytes. Each size has three warmups and 20 measured repeats: 161 complete saved arrays in total. It separately measures host input reset, map/frombuffer/copyto staging, TO sync, FROM sync, and output wrapping/copying. All arrays match regenerated byte patterns. XRT sync here exercises cache-maintenance APIs on host-written memory; this is **not a pure NPU DMA or bus-bandwidth benchmark**.

Calls use `fork-shared-intermediate` baseline/optimum as training and `fork-join-residual` plus `gemm-reduction-64` baseline/optimum as holdout. Each of six runners executes seeds 11 and 23, with three warmups and 20 measured calls per seed. There are **276 actual calls, 368 full output arrays, and 736 FP32/FP64 comparisons**, all passing. The 240 measured calls split into 80 training and 160 held-out observations. Outputs are NaN staged before each dispatch; all arrays, references, raw timings, and API events are retained.

All 276 calls selected the installed `_wait_polling` implementation. The requested sleep interval was 200 µs. Observed sleep counts, including warmups, were: zero sleeps in one call; one in 137; two in 77; three in 45; four in 14; five in one; and sixteen in one. A single fixed sleep is therefore not an adequate account of all observed calls. Device progress, polling phase, scheduling, and runtime behavior remain entangled; the experiment does not isolate their individual causes.

Training gives an effective elapsed-sleep median `Q=254323 ns`, completed `run.state()` probe median 270 ns, and declared two-probe term `P=540 ns`. Run creation, argument setting, and start medians are 18899 ns, 1220 ns per argument, and 7895 ns. The residual framework median is 63924 ns. This residual explicitly contains Python instrumentation overhead and is allocated once per invocation; it is not a measurement of uninstrumented host-only latency. Framework plus create/start contributes 90718 ns per call before per-argument handling.

The affine memory models use nonnegative coefficients fitted only to training-size hot medians. Extra per-call intercepts are zero by explicit ownership of the affine term per BO, with architecture provenance; these zeros do not claim an unmeasured operation was calibrated. CPU dispatch/completion coefficients remain null. Scalar compute and communication body parameters retain their old **assumed** provenance. Strict mode therefore leaves required costs unknown and cannot certify a calibrated optimum.

The independent audit verifies frozen producer and compiled identities; every saved memory/output array against regenerated patterns or independent source formulas; saved maximum errors; the complete unique call set; nonoverlapping operation spans; sleep containment within wait; exact residual arithmetic; and run/argument/start counts. It covers 7219 raw events. Completed-state probes run after the measured call and do not alter its timing window.

The producer's conditional host validation reports a median absolute relative wall error of 2.1032%. It inserts the **observed held-out wait duration** during validation. That statistic validates host-component accounting conditional on actual wait; it is not an independent end-to-end prediction and cannot establish accurate placement decisions. The separate old-whole-v26-cost sensitivity proxy is not a calibrated device estimate and is not used as the final C++ body prediction.

## R1 independent complete prediction and frozen-graph repricing

The final compiler replays each frozen v27 graph exactly and applies the new `program-fit-r2` profile, using body costs with old lifecycle/group setup removed. These full predictions do not insert the observed wait. Across 160 held-out hot calls, median absolute relative error is **42.7526%**. This fails the accuracy needed for a hardware oracle. Host-component conditional error of 2.1032% cannot replace this independent test.

| Held-out case/variant | Full model prediction ms | Actual calibration-run median ms | Median absolute relative error per call |
|---|---:|---:|---:|
| fork-join-residual / baseline | 0.356297096 | 0.422234000 | 15.61% |
| fork-join-residual / optimal | 0.356297096 | 0.828553500 | 56.99% |
| gemm-reduction-64 / baseline | 0.360391748 | 0.639633000 | 43.66% |
| gemm-reduction-64 / optimal | 0.360391748 | 0.634638000 | 43.21% |

For the four frozen baseline/old-optimum pairs, new predicted ratios are all exactly 1.0×: same region ABI plus one estimated polling quantum dominates their differently priced bodies. Predictions are about 0.362909 ms for the fork, 0.356297 ms for the join, and 0.360392 ms for the two GEMM cases. These are model predictions, not new measured speedups. The calibration calls execute each runner in sequence; their wall medians are not a new alternating paired speedup experiment. The earlier v27 paired hardware result remains separately reported.

All eight strict-mode frozen-candidate evaluations remain unknown because required body/DMA terms are assumed. No true optimal heterogeneous CPU/NPU assignment follows from this partial instrumented profile. The first-quantum estimate misses calls with additional polls or other readiness/wakeup behavior; the saved data does not isolate the physical cause.

## New search under the invocation model

A fresh campaign replaces only the v26 target's cost model and retains each original source, ordered finite choice/binding domain, compatibility table, search controls, and 600-second per-case solver budget. All evaluated assignment sets match v26. It runs the new evaluator through the existing native finite-domain search, separately from the CPU/NPU placement/cut contract above. Changing evaluator identity changes the full domain ID even when the ordered choice space is unchanged.

All eight domains exhaust: **5208 assignments in 80.060 seconds total process wall time**, at most 29.818 seconds per case and 240412 KiB peak process RSS. No solver budget or external timeout is reached on these small domains. This does not establish scalability to large inference graphs.

| Case | New status | Assignments | Unknown feasibility | Wall seconds | Predicted baseline ms | Best known ms | Predicted saving ns |
|---|---|---:|---:|---:|---:|---:|---:|
| fork-shared-intermediate | FEASIBLE | 512 | 144 | 5.404 | 0.362909139 | 0.362869139 | 40 |
| fork-join-residual | FEASIBLE | 1536 | 480 | 29.818 | 0.356297096 | 0.356297096 | ≈0 (roundoff) |
| shared-input-three-branches | OPTIMAL | 768 | 0 | 10.807 | 0.369521181 | 0.369441181 | 80 |
| fork-capacity-boundary | FEASIBLE | 512 | 144 | 5.204 | 0.362909139 | 0.362869139 | 40 |
| gemm-two-consumers | FEASIBLE | 1536 | 288 | 25.022 | 0.366988415 | 0.366988415 | ≈0 (roundoff) |
| gemm-reduction-64 | OPTIMAL | 64 | 0 | 0.601 | 0.360391748 | 0.360391748 | ≈0 (roundoff) |
| gemm-reduction-216 | OPTIMAL | 216 | 0 | 2.602 | 0.360391748 | 0.360391748 | ≈0 (roundoff) |
| gemm-reduction-trap | OPTIMAL | 64 | 0 | 0.601 | 0.360391748 | 0.360391748 | ≈0 (roundoff) |

Four cases have finite-domain analytical proofs under the frozen R1 diagnostic evaluator. The later actual-ELF audit shows that evaluator omitted required per-invocation physical setup/teardown, so these proofs do not certify a complete invocation-cost model. Four remain `FEASIBLE`: **1056 assignments require an explicit physical ABI refinement for partial/reusable host allocations**. They are unknown, not known infeasible, so exhaustion alone cannot establish those four minima. There are 1556 known-feasible points across all cases. Every comparison remains about the declared finite evaluator, with no new hardware realization check.

The only proved saving beyond numerical roundoff is 80 ns (1.00021654×) for `shared-input-three-branches`; its v27 baseline had failed hardware lowering, so this is not a paired physical speedup. The two 40 ns improvements are best-known results in unproved domains. The GEMM reductions differ by 5.55e-11 ns in binary64 arithmetic, which provides no speedup evidence. The evaluator intentionally preserves unrounded binary64 values; the report does not turn a floating-point tie into a performance result.

## Implementation validation and frozen identities

Frozen r1 ON contracts pass: invocation-cost 77 checks, heterogeneous-search 58 checks, and CLI invocation/placement/polling 196 checks. Frozen r1 OFF contracts pass: invocation-cost 77, heterogeneous-search 56, and CLI 161. The common search, beam, routing, graph, mechanism, source-identity, and frozen replay contracts also pass in both configurations. The ON search checks all 32 heterogeneous contract assignments against independent single-point evaluation; OFF reports unavailable search and a checked baseline rather than a fabricated proof.

The assumed contract changes launch price from 5 µs to 200 µs and changes the minimum from one NPU region to all CPU, without changing source. This validates sensitivity of region ownership and the search interface, not a calibrated placement recommendation. Tests also cover two-input/one-launch ownership, three same-chip calls, CPU→NPU→CPU dependencies, explicit cuts, strict/missing costs, runtime identity/type rejection, and hand-derived polling totals without whole-wait double charging.

- Frozen r1 ON compiler SHA-256: `e7120cab78eb22452701b98285e581878fa12b87792862e792d2da81bc081c80`.
- Frozen r1 OFF compiler SHA-256: `f899d8c55c97432c862b8043202c5b334ce19e4ffaf33a2b7fb1c5bc52d6b5c1`.
- Frozen runtime model `program-fit-r2/model.json` SHA-256: `61264ffc9722caa59aebd091928c43ebfc4d8f3dc602cdd50716e282ff4e01eb`.
- Formal frozen-graph repricing: `invocation-calibration-v28-r1/public-reprice-r2`.
- New native-domain search: `/home/shiran/project/orbit-artifact/results/taskflow/invocation-oracle-v28-r1`.
- Reports: `/home/shiran/project/orbit/docs/taskflow-invocation-regions-v28.md` and `/home/shiran/project/orbit-artifact/taskflow/docs/taskflow-invocation-regions-v28.md`.
- Curated evidence destination: `/home/shiran/project/orbit-artifact/taskflow/evidence/invocation-regions-v28`.

The curated record includes full numerical arrays and raw API samples, formal profile/repricing/search outputs, the final source/test snapshots and producer scripts, compiler identities and contract logs, plus SHA-256 inventories. Available large compiler binaries remain at their original locations with hashes recorded. The R1 OFF executable has no separate archived backup; only its recorded SHA and passing validation logs are retained. Earlier `program-fit-r1` and `replay-cost-r1` are preserved at raw paths but do not supply formal conclusions. Existing v26/v27 evidence remains unchanged.

## R2 diagnostic: buffered evidence and explicit device setup

R1 saved JSON/NPZ evidence between calls, outside each measured wall-time window. Such work can still alter the next call's arrival/readiness behavior. R2 buffers all evidence for a complete runner (both seeds), unloads that runner, then writes the files. Its runtime identity is `orbit.invocation.prepared-hot.buffered-evidence.v28.r2`, separate from R1. R2 retains NaN staging, API instrumentation, references, warmups, and training/holdout separation. The original memory calibration is reused byte-for-byte with a source/copy SHA receipt; no extra memory or device observations are invented.

R2 repeats 276 calls, 368 complete output arrays, and 736 comparisons, all independently audited. All 240 measured calls have exactly one polling sleep. Across warmups too, 275 calls have one sleep and one first warmup has zero. The interarrival medians by workload are 155.901 µs (fork), 64.243 µs (join), and 91.3425 µs (GEMM). Interarrival includes clock-excluded numerical checking, state probes, output poisoning, and bookkeeping; it is not a throughput measurement. The producer source and null-first/nonnegative-gap records are audited. Absolute call timestamps were not saved, so these gaps cannot be independently reconstructed from timestamps. Static source audit confirms the per-call repeat loop performs no evidence file writes.

R2 training gives elapsed polling quantum 254053 ns, a two-probe term of 910 ns, and framework residual 59023.5 ns. Its conditional host-component median relative error is **10.2842%**, again reusing actual held-out wait. R1's 2.1032% conditional statistic must not be reused for R2. R1's failed complete-prediction evidence and original profile remain unchanged; the buffered run is a separate diagnostic, not a replacement for the frozen R1 oracle.

An independent offline decode of the actual v27 optimum ELF shows dispatched `main` section `.ctrltext.3` has **83 control operations / 5220 bytes**, including an initial empty PDI load (ID 2), configuration register/block writes, the data-plane sequence, and a final empty PDI load (ID 1). The standalone data-plane sequence is 13 operations / 464 bytes. One physical configuration therefore does not mean physical setup happens only at context creation. The alternating PDI IDs also prevent the documented same-PDI skip from removing both resets. This establishes actual binary structure, not measured operation latency. Local compiler source excerpts explain lowering/cache behavior but are not asserted to be the exact source revision of the installed compiler.

The corrected model requires separate once-per-region runtime-only `device.setup` and `device.teardown` atoms, ordered `submit → setup → body → teardown → complete`. They are not charged per logical task group. The formal R2 profile at `invocation-calibration-v28-r2/program-fit-r1/model.json` leaves both durations **null** because they have not been measured. CPU dispatch remains unknown and body compute/DMA prices remain assumed. The separate `sensitivity-model.json` sets setup/teardown to **zero with assumed provenance** solely to test host/polling sensitivity; it is not the formal model or a measured claim that these operations are free.


The corrected R2 compiler evaluates all eight frozen baseline/old-optimum candidates with unknown formal cost: missing setup and teardown propagate into polling completion and the whole invocation. All eight strict sensitivity evaluations are also unknown. This is deliberate missing-cost propagation, not an optimizer failure or a zero-cost assumption.

The separate zero-setup sensitivity run predicts about 0.346119 ms for the fork, 0.339407 ms for the join, and 0.343602 ms for GEMM. Across 160 held-out calls its median absolute relative wall error is **10.0750%** without inserting the observed wait. These are independent predictions of an explicitly incomplete sensitivity model, not validated full invocation costs. The data are:

| R2 held-out case/variant | Zero-setup sensitivity prediction ms | Actual median ms | Median absolute relative error per call |
|---|---:|---:|---:|
| fork-join-residual / baseline | 0.339407096 | 0.305542500 | 11.08% |
| fork-join-residual / optimal | 0.339407096 | 0.304882000 | 11.32% |
| gemm-reduction-64 / baseline | 0.343601748 | 0.315192500 | 9.01% |
| gemm-reduction-64 / optimal | 0.343601748 | 0.344871000 | 0.71% |

Frozen R2 validation passes 96 invocation-cost checks, 58/56 heterogeneous-search checks with solver ON/OFF, and 227/192 CLI checks. Existing regression contracts and five public-graph AIR contracts also pass. The frozen R2 ON compiler SHA-256 is `f57eac3b207dc8678b89d01ce01e6b2cd5003f2d196d543454c049ace1605cd0`; OFF is `fc4b5514bde6b1b77cbbf8c0c6389427954c6335170c02095fed7365cd4db6eb`. Formal profile SHA-256 is `c15fd38a76ffcc945543414ff6b9b00e356d62c9375acd40e7585805d9aab8db`; the separate sensitivity profile is `469c2e441574c0e974c9faf59755917bbda5cee1f653e6c7f7ae0b717e2a5714`.

The current implementation explicitly represents the previously omitted costs and refuses to fabricate them. Establishing a hardware oracle still requires identifying physical setup/teardown and device-body costs, calibrating CPU execution/dispatch under the same runtime boundary, validating complete held-out predictions, and composing the resident heterogeneous choices with the remaining native optimization domains and executable constraints.
