# Rectangular MLP2: calibrated native search and NPU validation (r12)

The frozen C++ search selected whole-program NPU execution for batch 24 before
heldout timing labels were opened. Against the declared NPU phase-local M24
baseline, the main hardware campaign measured **3.968181x** speedup:
0.774212ms versus 0.195105ms. Beam selected the same plan. This is a result for
three declared NPU choices; it is neither a CP-exclusive improvement nor an
optimum over the full ORBIT abstraction. All-CPU remains faster, at 0.004310ms.

Evidence root:
`/home/shiran/project/orbit-artifact/results/taskflow/full-joint-v32-r1/preparation-r12`.
The main campaign, selected-plan replay, and search timing are separate records.
`evidence-seal.json` hashes 489 source/protocol/model/search/measurement files
and compiled receipts; intermediate build trees remain in their original paths.

## Source, protocol and cost boundary

The source is the same two-layer MLP arithmetic family: BF16 X and 16x16 W,
FP32 sequential accumulation, ReLU, BF16 round-to-nearest ties-to-even between
layers, then FP32 output. Batch M varies 16/24/32; N=K=16. Original batch 16
input/reference arrays were asserted identical to r10. References for 24/32
use the same separate FP32 multiply/add order and explicit RNE conversion.
The shared deterministic seed and weight structure mean this is a shape/design
holdout, not a new input-distribution test.

The compiler is frozen at SHA256
`3616f1ae32734e74df37dfe9fb228b7f0b35a9f864a7455f2e321834839aef1c`.
The runtime is `orbit.serial-prepared-typed-native-wait2.v32`, with execution
contract `e699053c5e8e47b7a2cce1551762419425344df344de83420357e409c71c5b83`.
Source/feature/manifest/compiled receipts bind the parent graph and ordered
selected child graphs. Source initializer distribution is an explicit lowering
transformation, so selected best.mlir is not claimed byte-identical to the
original source body. The arithmetic contract and typed source-bound witnesses
are retained. Old r9/r10 timings were not relabelled as this compiler's training.

Seven training designs are CPU16/32, NPU whole16/32, phase16 at batch 16,
and phase8 at batch 16/32. Four heldout designs are CPU24 and NPU phase8,
phase24, whole at batch 24. `case-spec.json`, `registration.json`,
`protocol-files.json`, `premeasure-check.json`, and `static-design-audit.json`
record the split and static identities. All 11 exported local source families
match `a99d0a1da88eb4a5c525e11c5e8de78092f9af98c33eedf18abfb9063ff26e6d`.
Every fixed phase's distinct-design training matrix has rank2/2; all applicable
heldout feature vectors are within both coordinate ranges and the joint convex
hull. Full rank does not establish physical identification of individual costs;
CPU training has only two independent design points.

Each design used 3 fresh measurement Python processes, CPU affinity 0, 5 warmups
and 31 hot calls per process: **33 processes and 1188 calls**, all numerical
checks passed. Independent review of saved outputs found maximum absolute
error 0 against the declared RNE references, including warmups. All 1188 recorded
calls satisfy exact integer-nanosecond phase conservation.

The boundary is prepared host-to-host execution. Application reset, control
and output copy are charged once. CPU body or each NPU region's sync-to,
create/bind/start submission, native wait2, and sync-from phases own the
remaining cost exclusively. Wait includes unseparated physical setup, device
body, teardown and completion effects; it is not pure compute or pure launch.
These costs are not added a second time from a separate device estimate.
Context/BO preparation and evidence writing remain outside this boundary.

Training-only local-phase NNLS produced model SHA256
`1605c8c6200cc8df463dcf553d2fad59000c5839f1b2a86dad7cee35a13d6837`.
`model/training-inputs.json` contains no batch 24 measurement labels. Independent
review checked all recorded training-input hashes and the model/search hashes
in `selection-before-validation.json`. Selection was sealed before validation.
The fitted model is a feature-based phase model, not a complete-plan price table.

## Heldout measurements and prediction

Times are hot-call medians in milliseconds. The group target is the median of
three fresh-process medians, not every individual invocation.

| Batch 24 plan | Physical configurations | Process1 /2 /3 | Group median | Prediction | Relative error |
|---|---:|---|---:|---:|---:|
| NPU phase8 |12|2.295797 /2.347026 /2.255958|2.295797|2.363589|2.953%|
| NPU phase24 (baseline) |4|0.774212 /0.758123 /0.847101|0.774212|0.795598|2.762%|
| NPU whole |1|0.195105 /0.195096 /0.195316|0.195105|0.203584|4.346%|
| CPU |0|0.004250 /0.005810 /0.004310|0.004310|0.004773|10.750%|

`validation/validation.json` reports complete 4/4 group coverage, median absolute
relative error **3.649400%**, p90 **8.829106%**, passing the preregistered 20%
median gate and three-process requirement. Its 12 process-row error median is
4.560853%, p90 12.157649%; these are different aggregation targets.
The legacy process-row pair metric is 54/58 correct and is retained, including
comparisons between repetitions of the same witness. At the distinct-design
level, process-median min/max intervals are strictly separated in the order
CPU < NPU whole < NPU phase24 < NPU phase8; all six design pairs and NPU top 1
agree with model ranking. No uncertain interval was silently treated as a win.

Whole is 3.968181x faster than the NPU phase24 baseline; phase8 is slower than
that baseline. CPU is 45.268x faster than whole NPU. The CPU comparison measures
placement choice and must not be presented as an NPU optimization speedup.
The whole/phase improvement combines configuration count, retention and the
selected native implementation; this experiment does not isolate their causal
contributions. Fixed round-robin case order leaves time-drift confounding, and
three processes provide limited evidence of stability.

## Native CP and beam searches

`native-search/declaration.json` fixes the batch 24 source and NPU-only domain
{phase M24,N16,K16; phase M8,N16,K16; whole-program}. These coupled native
choices vary tiling/residency, while dtype, arithmetic, implementation, one core,
routes, scheduling, slots and other dimensions remain fixed. Missing support is
not interpreted as full-dimensional coverage. The 600 s budget is a limit, not
observed solve time.

| Search | Evaluations | Unknown cost | Selected plan | Status | Total time | CP solve time |
|---|---:|---:|---|---|---:|---:|
| NPU CP |3|0|whole|OPTIMAL in declared model domain|458.514ms|4.526ms|
| NPU beam |3|0|whole|FEASIBLE|334.098ms|—|
| CPU/NPU CP extension |48|42|all CPU|FEASIBLE|4888.451ms|65.011ms|
| CPU/NPU beam extension |15|12|whole NPU|FEASIBLE|1657.718ms|—|

The NPU CP callbacks took 452.273 ms, so most search time was materialization,
checking, export and cost evaluation rather than CP itself. CP peak RSS was
67,016 KiB for the NPU domain and 66,900 KiB for the heterogeneous extension.
The branch-and-check master proposes choices; exhaustive checked evaluation
establishes the finite-domain result, rather than a latency-valued CP bound.
Beam found the same
NPU result faster in this small domain. This does not establish large-domain
solver scaling.

The CPU/NPU extension jointly exposes the same native choice and four task
placement bits: 48 assignments. All 48 were structurally feasible, but 42 lack
covered local phase prices. CP exhausted the domain and found CPU among the
six priced assignments; unknown costs prevent an optimality proof. Beam stopped
after 15 checks with 12 unpriced assignments and retained whole NPU. Neither
result proves an optimum over arbitrary CPU/NPU placement or concurrent region
schedules. Unknown cost is not mathematical infeasibility.

For all four selected search artifacts, independent Python reconstruction of
all local model phases matches the C++ estimated total to relative tolerance
1e-12. Selected parent/ordered-child graph IDs match the measured compiled
witnesses. C++ cost receipts explicitly mark `physical_execution=false`: search
and real device execution are separate evidence. The selected artifacts were independently compiled and measured again,
as recorded below; this replay does not replace the main heldout campaign.

## Independent selected-export replay

`selected-replay/summary.json` records a separate post-selection campaign. The
actual NPU CP and heterogeneous CP selected manifests were independently compiled;
plan IDs, parent graph IDs and ordered child graph IDs exactly match the search
outputs. Three fresh processes again used 5 warmups and 31 hot calls per plan.

| Replay plan | Process1 /2 /3 hot median (ms) | Median (ms) |
|---|---|---:|
| NPU phase24 baseline |0.805942 /0.835281 /0.776422|0.805942|
| CP-selected whole NPU |0.194885 /0.196666 /0.195345|0.195345|
| Heterogeneous CP-selected CPU |0.004249 /0.004250 /0.004270|0.004250|

All 324 replay calls passed; independent saved-output review again found zero
maximum absolute error and exact phase conservation. Replay gives 4.125737x
against its own NPU baseline. The main 3.968181x result remains the primary
heldout result; the larger replay ratio is not substituted for it. Both rounds
show approximately 4x NPU improvement. Main plus replay totals **1512 calls**.
Replay was not used for fitting, heldout accuracy, model tuning or reselection.
The CPU result is a separate placement comparison and remains faster.

## Independent audit and remaining scope

Independent audit scripts/results are available as
`/tmp/audit-mlp-r12-independent.py` and
`/tmp/audit-mlp-r12-independent.json` (copied into the evidence root by the main
agent). Additional replay audit is
`/tmp/audit-mlp-r12-selected-replay-independent.json`.
They checked 1188 output arrays, numerical references, exclusive phase sums,
training and selection hashes, compiled graph identities, and C++/Python
selected prediction agreement without using a device or refitting the model.

This result validates the declared local model and finite native choices for
this small MLP family. It does not finish full-abstraction joint optimization,
prove hardware optimality, price all mixed CPU/NPU regions, or establish
arbitrary-depth MLP support. Matrix grouped-retention probes remain unsupported
at the scoped pointwise dependence guard and were not invented as legal
holdouts. The larger calibrated-domain coverage problem remains explicit.
