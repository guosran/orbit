# Mixed CPU/NPU calibration and solver reference

The 42 previously unpriced mixed placements now have local phase models. All
48 assignments in the declared r12 CPU/NPU placement and native-residency domain
are checker-feasible and priced. CP-SAT exhausts that domain and reports OPTIMAL;
ORBIT's public beam finds the same plan and objective. This closes the missing
mixed-price evidence, not unrestricted optimality over every ORBIT abstraction.

## Calibration and validation

The old calibration contained only whole-program families. The mixed plans need
18 additional device/source-family/implementation keys, for 20 keys overall.
Training uses 41 mixed designs plus seven complete-program designs at batches
16 and 32. Four previously unmeasured mixed batch-24 designs form the heldout
set. Old full-program batch-24 timings and candidate-6/9 diagnostic executions
are disclosed prior evidence and excluded from these new blind holdouts.

Application control uses the explicit features `call`, `regions`, and
`configuration_count`: the executor visits each CPU/NPU region and each physical
NPU configuration. Other phases separately account for reset/output copy, CPU
body, NPU input synchronization, argument binding/submission, wait, and readback.
No complete-plan price table or arbitrary replacement for unknown costs is used.
All target phase vectors lie in the corresponding training joint convex hulls.
Control has rank 3/3. Four submit families have rank-one feature matrices;
individual coefficients are predictive fits, not separately identified physical
constants. The global default feature policy is unchanged.

All 52 designs were freshly exported and compiled under the new frozen compiler
and execution contract. Each ran in three fresh processes, five warmups and 31
hot calls per process, CPU affinity 0, round-robin order. Our compilations and
CPU-heavy checks finished before serial device measurement. All 5,616 calls
passed unchanged `atol=rtol=1e-4` checks. Independent auditing checks all 5,616
saved Y arrays in 156 NPZ files against a sequential FP32/BF16-RNE reference:
bit-identical, maximum absolute error zero. Every exclusive phase sum equals
full prepared-call wall time. Compiled/source/manifest identities also pass.

The model is trained only from training labels. Model and selected search
artifacts were sealed before opening heldout timing labels. All four heldout
designs have three verified fresh processes and known predictions. Group median
absolute relative error is **5.1485%**, p90 **7.2983%**. The predicted winner
matches the measured winner, whose process-median interval is strictly separated
from the other three. These are four heldout witnesses, not 12 independent
workloads. All 97 full C++ evaluations across the four search runs, including
each phase, match independent local-model recomputation.

## Search results

Every run has a 600-second wall budget, one worker and a fixed seed. Timing
includes materialization and checker/cost callbacks; `solve` is only solver time.

| Search | Evaluations | Unknown costs | Selected plan | Predicted ms | Total ms | Solve ms | Status |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| NPU CP-SAT | 3 | 0 | Whole-program NPU | 0.201331757 | 474.218 | 4.351 | OPTIMAL |
| NPU public beam | 3 | 0 | Same NPU plan | 0.201331757 | 367.422 | — | FEASIBLE |
| CPU/NPU CP-SAT | 48 | 0 | All CPU | 0.004803031 | 4950.677 | 65.233 | OPTIMAL |
| CPU/NPU public beam | 43 | 0 | Same CPU plan | 0.004803031 | 4252.776 | — | FEASIBLE |

The two CP runs peak at 73,284 and 72,116 KiB RSS respectively. Most time is in
callbacks, not CP solving; the ten-minute budget is not exhausted. The CP master
uses branch-and-check coverage, with optimality obtained by exhausting the
finite domain and checking every feasible cost. This is not a latency-valued
CP lower-bound model. Beam claims no exhaustive proof, but its observed model
objective gap against this CP reference is zero. The corresponding CP and beam
plan IDs, parent graph and ordered child identities match exactly.

The bounded unknown-price frontier fix is independently demonstrated on the
frozen old r12 records: old beam explores 15 points and misses CPU; the repaired
beam explores 48 and reaches the CP winner. In this fully priced r13 experiment
no unknown frontier entries are needed. The two improvements—pricing coverage
and exploration through unknown prices—must not be conflated.

## Scope and evidence

The varying choices are three native tiling/residency mappings and four original
tasks' independent CPU/NPU placements, giving `3 * 2^4 = 48`. Resource shape,
replica count, layout, general fission, physical slot reuse, arbitrary scheduling,
routing and overlap are not independently varied here. Unsupported backend
choices remain unsupported; this report does not call fixed dimensions searched
or claim a global hardware optimum. Larger programs' fixed-plan board speedups
are reported separately in `taskflow-npu-larger-r13-v32.md`.

Evidence root:
`/home/shiran/project/orbit-artifact/results/taskflow/full-joint-v32-r1/preparation-r13`.
It contains frozen producers, registration and protocols, all compile/measurement
receipts, static and independent audits, training-only input hashes, model,
selection-before-validation seal, native searches and validation.

- Compiler SHA256: `46f28808482500d9b40c33d465bdb9c2ac48690b7c8aa41e528574768d05587a`.
- Execution contract: `76197db1541ec24144e2023874f2d0ddae81ac8955de0a01a2a360228f556afa`.
- Model SHA256: `0f74da10a6f07c32c71be3aed0a01d45d718ee367d27db97d942cd72185d8eef`.

Timing covers a prepared host-to-host call, including launch, synchronization,
readback and host control. Initial compilation, context creation and allocation
are excluded. No vendor patch, polling fallback or global runtime monkeypatch
is involved. The independent auditor is archived as `independent-audit.py/json`.

## Actual selected-plan replay

The selected NPU and CPU manifests were independently compiled, alongside the
same-source phase-M24 NPU baseline. CP and beam have identical selected plan,
parent and ordered-child IDs, so each shared selected plan is executed once per
fresh process. Nine fresh processes add 324 checked calls, separate from fitting
and heldout validation. Group medians are baseline **0.778621 ms**, selected NPU
**0.202785 ms**, and selected CPU **0.004250 ms**. Thus the selected NPU plan is
**3.8396×** faster than its NPU baseline, while CPU is **47.7141×** faster than
the selected NPU plan for this small batch-24 program. The heterogeneous choice
of CPU is supported by execution; NPU is not forced into the winning plan.

Process medians (ms): baseline 0.747411/0.778621/0.830989;
NPU 0.200365/0.203575/0.202785; CPU 0.004250/0.004220/0.004280.
Independent auditing confirms all 324 saved outputs are bit-identical to the
RNE reference, every phase sum is conserved, and compiled/source/runtime hashes
match. Identity checks use the parent plus ordered children and plan ID together;
the plan ID alone does not distinguish every native realization.
The replay producer, commands, compiled identities and outputs are under
`selected-replay/`. These data are not used to refit or choose the model.

The archived r13 top-level `search.json.cost_model_boundary` still contains an
older hard-coded partition-provider message saying host assembly is unknown.
This is a reporting defect: the selected `cost_evaluation.timing.boundary` and
phase records correctly describe the prepared host-to-host boundary above.
A subsequent metadata-only fix reads the selected provider's actual boundary
receipt and reports its pricing status separately. It does not change these
frozen r13 binaries, prices, selections or measurement evidence.
