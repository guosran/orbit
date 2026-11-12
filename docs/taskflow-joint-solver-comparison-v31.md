# Joint CP-SAT / SCIP comparison (v31)

This CPU-only experiment compares the existing direct CP-SAT model with the new
SCIP MILP model on the **same source-bound `ProgramJointDomain`**. It finds cases
that take seconds or minutes to prove, and a concrete MILP construction limit.
All costs below are synthetic assumptions. These results establish neither NPU
speedup nor support for arbitrary GEMM or heterogeneous programs.

## Frozen inputs and scope

The frozen tool is `/tmp/orbit-solver-scale-v31-r1/taskflow-opt`, SHA256
`41e3d599eba9b46d39b03f3bc4f1d69c1688ef8200aa5416fb1d3ce468509061`.
Its final joint contracts had passed before copying. Each campaign wrote its
protocol, source/configuration hashes, driver hash and tool hash before solving.

The source family consists of real scalar `taskflow.task` MLIR: an input producer,
a fork into two or four branches, and binary joins to an output. Width two has
four source tasks; width four has eight. All tasks traverse the same `rows x 1`
grid. Fixed one-row chunks yield 4–32 source-bound compute instances. The actual
source bodies contain scalar multiplication/addition and exact affine accesses.

The domain has one chunk variant, two compute places, one shared unary DMA
resource, 64 bytes of local memory per place, local groups of up to two tasks,
and at most two input/output streams per configuration. Group coverage, group
placement, copy boundaries and configuration schedules are chosen jointly. Each
place has one input/output route: these cases do **not** exercise alternative
routes within a place. There is no source transformation or chunk-size search in
this campaign, and no CPU/NPU partitioning or physical hardware execution.

Assumed prices are 4 ns of compute per instance, 8 ns per 4-byte input transfer,
1 ns per 4-byte output transfer, 2 ns per configuration, 3 ns submission and
5 ns completion. These are finite integer test prices, not measured latencies.
No complete plans were enumerated and pre-priced to construct either objective.

## Sixty-second scaling campaign

Each pair received the same domain, one worker, a 60-second total-pipeline soft
quota, 100,000 variables and 500,000 constraints. The external guard was 85
seconds. Each solver ran in a fresh process, serially CP-SAT then SCIP; there was
one timing trial per case. These are diagnostic timings, not repeated statistical
speedup estimates. Every returned plan, including retained baselines, passed a
separate CLI materialize/check replay. All six pairs have identical domain IDs.

`rows` multiplies instances, not source-task count. `V` below denotes variables.
The complete per-run build/solve/pipeline/external wall times, constraints and
process peak RSS are in the raw JSON and `/usr/bin/time -v` receipts.

| Fork width | Rows | Source tasks / instances | CP-SAT status; solve | CP model cost / bound (ns) | SCIP status; solve | CP / SCIP V |
|---|---:|---:|---|---:|---|---:|
| 2 | 1 | 4 / 4 | OPTIMAL; 0.072 s | 55 / 55 | OPTIMAL; 12.557 s, cost 55 | 908 / 32,724 |
| 2 | 2 | 4 / 8 | OPTIMAL; 23.094 s | 102 / 102 | RESOURCE_LIMIT; no solve | 3,558 / 100,000 |
| 2 | 4 | 4 / 16 | FEASIBLE; 53.971 s | 196 / 34 | RESOURCE_LIMIT; no solve | 14,090 / 100,000 |
| 4 | 1 | 8 / 8 | OPTIMAL; 9.902 s | 120 / 120 | RESOURCE_LIMIT; no solve | 4,507 / 100,000 |
| 4 | 2 | 8 / 16 | FEASIBLE; 53.964 s | 234 / 29 | RESOURCE_LIMIT; no solve | 17,876 / 100,000 |
| 4 | 4 | 8 / 32 | FEASIBLE; 53.897 s | 460 / 21 | RESOURCE_LIMIT; no solve | 71,206 / 100,000 |

The last three CP gaps were 82.653%, 87.607% and 95.435%. A gap describes the
solver's incumbent versus its bound; it is not a hardware slowdown or a measured
error in the candidate's cost. The width-four, one-row baseline costs 121 ns;
CP-SAT's 120 ns is a model improvement, not an observed device improvement.

SCIP's resource-limit cases hit the variable cap during construction and retain
checked baselines. They are neither solver timeouts nor infeasibility proofs.
No SCIP incumbent/bound was obtained on those paths. Their raw zero bound/gap
placeholders under `has_objective=false` are normalized to null in the analysis.
The capacity encoding creates start-anchor occupancy comparisons for pairs of
optional positive-length intervals per resource/configuration. Its quadratic
size is an implementation property; these results do not establish a general
limit of MILP or SCIP. Even the four-instance model has 32,724 variables and
97,853 constraints versus CP-SAT's 908 and 2,062. Its process peak RSS was
355,036 KiB versus 47,208 KiB. Peak RSS includes parsing, checking and construction,
not just the optimizer.

## Six-hundred-second followup

A separately frozen followup used exactly the width-two, four-row domain and the
same 100,000/500,000 caps, changing only the total-pipeline quota to 600 seconds.
Both returned plans passed independent replay; both domain IDs match the
60-second case. The external 650-second guard did not fire.

| Solver | Status | Actual solve | Model cost / bound (ns) | Gap | Peak process RSS |
|---|---|---:|---:|---:|---:|
| CP-SAT | FEASIBLE | 539.962 s | 196 / 57 | 70.918% | 425,132 KiB |
| SCIP | RESOURCE_LIMIT | 0 s | baseline 196 / unavailable | unavailable | 269,992 KiB |

The incumbent did not improve; CP-SAT spent the extra budget tightening its
bound from 34 to 57. It did not prove optimality. An independent analytical
certificate for this particular synthetic domain establishes a 196 ns lower
bound matching the checked feasible schedule: a configuration can hold two
groups only when each has one input/output, and the minimum such paired
configuration costs 23 ns. Assigning 11.5 ns to each of these singleton groups
and the isolated configuration duration to every other group gives a valid
configuration lower bound. A 16-state per-slice exact-cover relaxation yields
47 ns per slice; four slices plus submission/completion yield 196 ns. The
certificate script checks the domain conditions and enumerates the four-DMA
orders used for the paired bound. This independent certificate does not change
the recorded CP-SAT `FEASIBLE` status or turn its 57 ns bound into 196.

## A complete larger MILP and the UNKNOWN path

A third, explicitly different finite domain keeps the width-two/two-row source
and prices, restricts configurations from eight to four, and gives both solvers
160,000 variables/500,000 constraints and 60 seconds. It is not silently treated
as the original eight-configuration domain. Both solvers have the same new
domain ID and both returned plans pass replay.

| Solver | Status | Build / solve | V / constraints | Model cost / bound (ns) | Peak process RSS |
|---|---|---:|---:|---:|---:|
| CP-SAT | OPTIMAL | 3.986 ms / 0.706 s | 1,798 / 4,078 | 102 / 102 | 59,076 KiB |
| SCIP | UNKNOWN, checked baseline retained | 181.809 ms / 53.935 s | 130,130 / 389,753 | baseline 102 / unavailable | 1,154,436 KiB |

SCIP's exact message was `SCIP has no incumbent; checked baseline retained
(status 23)`. SCIP 10.0 defines status 23 as `SCIP_STATUS_TIMELIMIT`. This is the
`SCIPgetBestSol()==nullptr` return path, before integer decoding or candidate
materialization/checking. It is not a numerical-validation or checker rejection.
SCIP verbosity was zero, and its stdout/stderr are empty; no additional internal
incumbent history was recorded. Absence of an incumbent alone cannot exclude an
encoding problem. A separately instrumented fixed-baseline feasibility check is
recorded with the raw evidence; its result is described below.

The fixed-baseline diagnostic succeeded. A separate `/tmp` build retained all
130,130 variables and 389,753 original constraints, then fixed the source/group/
assignment/configuration/compute-start/route choices and times to the checked
baseline using variable bounds. SCIP found the 102 ns incumbent and completed
this restricted model (SCIP status 1) in 0.601 seconds. The independent integer
checker accepted it, and the original frozen CLI replay also returned zero.
The diagnostic explicitly disables a full-domain optimality claim. Its timing is
not part of the solver comparison. Thus the free run's UNKNOWN is a no-incumbent
time limit, while this exact MILP demonstrably contains the checked baseline;
it is not evidence of mathematical infeasibility or a failed materialization.
This checks one concrete feasible assignment, not every aspect of the encoding.

## Comparison and proof boundaries

Both backends reserve roughly 10% of their own remaining quota after construction
for finishing/checking. A 600-second CLI quota therefore permits roughly 540
seconds of CP solving here, not 600. Their build costs, remaining internal
quotas and search controls are not identical. CP-SAT receives baseline hints;
the current SCIP implementation shares the checked-baseline horizon and external
fallback but does not supply a SCIP primal warm start. This matters when comparing
incumbent discovery. Both SCIP numerical proof tolerances and the independently
checked integer-nanosecond schedule remain part of its proof boundary.

No result is an unrestricted ORBIT optimum or hardware timing claim. The useful
regression cases are a 4-instance case both solvers prove, an 8-instance case
requiring seconds, a 16-instance CP case still unproved after its 600-second
pipeline budget, and a complete 130k-variable MILP with a retained checked
baseline after its no-incumbent time limit. Stronger bounds, symmetry handling,
warm starts and a smaller exact MILP capacity formulation are followup work;
longer time limits alone did not resolve this example's proof gap.

## Raw evidence and reproduction

The original runs are new directories under `/tmp`:

- `/tmp/orbit-solver-scale-v31-r1`: six 60-second pairs, frozen protocols/inputs,
  source/config/tool hashes, commands, full domain/solution/checker evidence,
  timing/RSS logs, compact `analysis.json`, and the independent bound certificate.
- `/tmp/orbit-solver-scale-v31-r2`: 600-second same-domain followup.
- `/tmp/orbit-solver-scale-v31-r3`: explicit four-configuration/160k-variable pair
  and the separate fixed-baseline diagnostic.
- `/tmp/orbit-solver-scale-v31-summary`: normalized 16-run result table and loaded
  solver-library identity hashes. Library hashes were captured after timing;
  the campaign did not modify those libraries.

The archived evidence is
`/home/shiran/project/orbit-artifact/results/taskflow/joint-solver-comparison-v31-r1`,
with the three campaign directories and normalized summary preserved separately.
It retains commands, hashes, source/config inputs, stdout/stderr, time receipts
and the restricted-baseline diagnostic. Frozen executable binaries and diagnostic
objects are excluded from the archive; the original 217 MB timed executable stays
at the `/tmp` path above and is identified by SHA. Diagnostic source/commands/SHA
are preserved, and diagnostic timings are excluded from the comparison table.

The summary contains `source-final-manifest.json`, `source-final-sha256.txt`,
a final interface/implementation/CLI/CMake/test source snapshot, Git status and
staged/unstaged patches. This is a later integration checkpoint: no contemporaneous
working-tree diff/source snapshot was captured when the earlier executable was
frozen. The run protocol records Git HEAD and executable SHA; later readability/
comment changes were reviewed and regressed separately. Do not imply the final
source bytes are the exact recorded build inputs of the frozen executable.
No existing raw campaign or model was overwritten. To reproduce a run, use its
frozen command JSON with a fresh output directory; existing output is rejected.
