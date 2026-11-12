# Native CP-SAT / ORBIT beam pilot (preparation r9)

Both searches selected the same CPU mapping and predicted **1.0×** improvement
over the explicit CPU baseline. CP-SAT took **5.93 s**, ORBIT beam **1.89 s**.
This experiment provides neither a CP advantage over beam nor a full-domain
optimality proof: only 37 of the 128 assignments had known prices; 91 remained
unknown.

## Reproduction and scope

Evidence root:
`/home/shiran/project/orbit-artifact/results/taskflow/full-joint-v32-r1/preparation-r9/native-search-pilot/`.
Use `domain-declaration-r2.json`, `cp-r2/target.json`, and
`orbit-beam-r2/target.json` for the actual validated domain. The earlier `cp/`
and `orbit-beam/` failures are retained: their invalid `task_group` field was
rejected before search. They are not successful experiments. The corrected
public mapping uses `groups: [{tasks: [0, 1], storages: [3]}]`.

The source is one complete GEMM64 plus ReLU, with BF16 inputs and FP32
accumulation. Seven binary choices give 128 assignments: M/N tiles 16 or 32,
K tile 32 or 64, CPU/NPU for each of the two tasks, phase-local versus
tile-major retention of the intermediate, and slot reuse off/on. Device
choices are independent; unsupported mixed placements remain unknown.
Implementation, dtype, resource/count/shape, replica/partition, routes,
priority, source fission and initialization settings are fixed as recorded in
the r2 declaration. This is a finite public-domain experiment, not a search
over every ORBIT abstraction.

The compiler SHA256 is
`b7b7d4849ed07439ad2b30114899224432acbab6e892bba04383af9c350105e8`.
The train-only model SHA256 is
`67a8c4e112daa0ba1648583ea3e46fa865a4de0bccc135fad9f90eac205bcb4b`.
`producer.json` binds these inputs and the execution contract. Both reports
have the same dimensions, constraints, evaluator and domain ID
`ba3ebf106900dfe0d7439508061ebbe69d0eb556fb63760b16d397eb45f4918a`.
Prices come from the C++ calibrated mechanism evaluator, not a table of
complete-plan prices. `singleton-price-crosscheck.json` records agreement
with the independently decomposed Python prediction to floating-point
precision.

## Search result

| Measure | CP-SAT branch-and-check | ORBIT public beam |
| --- | ---: | ---: |
| Wall time | 5.931655 s | 1.894315 s |
| Evaluations | 128 | 28 |
| Priced | 37 | 18 |
| Known feasible, unknown cost | 16 | 10 |
| Unknown feasibility | 75 | 0 |
| Best predicted complete-call cost | 0.140895 ms | 0.140895 ms |
| Exhausted finite domain | yes | no |
| Proved finite-domain optimum | no | no |
| Peak RSS | 91,612 KiB | 75,608 KiB |

Both have status `FEASIBLE`. The all-zero baseline is CPU/CPU, M16/N16/K32,
phase-local, no slot reuse; both incumbents have the same predicted cost as
this baseline. The **600 s** budget per algorithm (660 s external timeout,
one worker) is an upper bound, not time consumed. CP-SAT model construction
took 1.73 ms, solver work 166.24 ms, and evaluator callbacks 5,747.91 ms.
Callbacks dominate this small experiment; its runtime does not establish
scalability for larger domains. Neither run recorded semantic-cache hits.

The 91 CP unknowns comprise 32 unsupported cross-device residency groups,
16 unproved cross-tile orderings, 16 native execution-group witness
mismatches, eight memory-capacity failures, two prepared-context limits,
one SRAM/layout-scratch limit, and 16 feasible mappings without a covered
runtime price. They were not assigned zero cost or silently converted to
known infeasibility. Exhaustion therefore establishes only the best known
priced result, not an optimum over all 128 assignments. Full raw reports and
failure reasons are retained in `summary.json` and each search directory.

## Corresponding formal measurements

These measurements were collected separately by the hardware owner under
the frozen r9 contract; this search pilot did not launch hardware. Each row
has three fresh processes, 36 checked calls per process, with all output
checks passing. Values below are medians of the three process medians.

| Formal GEMM64 mapping | Process medians (ms) | Median (ms) |
| --- | --- | ---: |
| CPU | 0.223794, 0.130686, 0.223244 | **0.223244** |
| NPU M32/N32/K32 | 1.767547, 1.696738, 1.764607 | **1.764607** |
| NPU M16/N32/K64 | 3.237040, 3.290759, 3.272169 | 3.272169 |
| NPU M32/N16/K64 | 3.206871, 3.309539, 3.361117 | 3.309539 |

The CPU result varies from 0.130686 to 0.223794 ms across processes; the
0.140895 ms model prediction must not be substituted for the measured
0.223244 ms median. `baseline-formal-graph-check.json` binds the singleton
CPU and NPU M16/N32/K64 parent and ordered child graph identities to their
formal manifests. Measurement files are
`preparation-r9/<mapping>-measure-p{1,2,3}/result.json`, with mapping names
`gemm64-cpu`, `gemm64-npu-32x32x32`, `gemm64-npu-16x32x64` and
`gemm64-npu-32x16x64`.

The 1.764607 ms NPU tile32 result is a separately measured NPU reference,
not the beam incumbent. Comparing it with CPU execution is a device-mapping
comparison; it is not evidence that CP improved on ORBIT. This pilot found
no measured search-induced speedup and made no hardware-optimality claim.
