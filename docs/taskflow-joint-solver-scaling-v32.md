# Direct joint solver scaling: v32 followup

The new SCIP MILP backend proves the existing 16-instance finite fork/join model
optimal at **196 ns**. Model construction took 0.703 s, optimization 9.925 s,
and the process used about 3.18 GiB peak RSS. A 600-second pipeline quota was
allowed; the solver stopped after proving optimality. This is a CPU-only solver
benchmark with synthetic integer costs, not an NPU speedup result.

## Identical problem and different representations

The source/configuration bytes and serialized `ProgramJointDomain` exactly match
the v31 width-two, four-row case: four real scalar MLIR tasks, sixteen compute
instances, one chunk variant, 36 local groups, two places, K=16 configurations,
and a shared unary DMA resource. Costs remain 4 ns compute, 8 ns per input,
1 ns per output, 2 ns setup, 3 ns submit and 5 ns complete. No complete plans
were enumerated or pre-priced to construct the objective.

- Domain ID: `f43eae7775ee4dc5144337821f12026061248fa31279890c33bf5cf0b1b229dd`.
- Model ID: `46d06bf2a719825c22baf48bfb5fca9aea5eaf2d509e1425fdee298fd3688170`.
- Source fingerprint: `04cb9afff6ab07ca2e5eb7b62f9456a016bd5225e5ee73aaa335d9ce97fce067`.
- Serialized domain SHA256: `f68d8aa763545ab6890088fa7050c5fb2191c4597c6e37b5b140cc41a5ad92db`.

The typed runner links the existing production solver and domain/checker archive.
It changes only caller options: 500,000 variables, 1,200,000 constraints, one
worker and a 600-second pipeline quota. Production CLI defaults were not changed.
A separate CLI probe with 500,000 variables but its normal 500,000 constraints
returns `RESOURCE_LIMIT` during construction at exactly 500,000 constraints;
SCIP is never invoked. Thus this result requires the explicitly raised
representation cap. The typed run had a 6 GiB RSS guard, 12 GiB address-space
limit and 660-second external wall guard; none fired.

## Measured result and earlier CP-SAT reference

| Quantity | New SCIP typed runner | Earlier v31 CP-SAT |
|---|---:|---:|
| Status | OPTIMAL | FEASIBLE |
| Incumbent / bound (ns) | 196 / 196 | 196 / 57 |
| Gap | 0 | 70.918% |
| Variables | 362,762 | 14,090 |
| Constraints | 735,762 | 31,942 |
| Model build | 703.340 ms | 32.607 ms |
| Optimization | 9,924.944 ms | 539,961.561 ms |
| Peak process RSS (`time -v`) | 3,332,648 KiB | 425,132 KiB |

New-run preprocessing, including parsing, domain construction, baseline checking
and initial evidence writes, took 12.262 ms. The solver backend returned in
10,631.501 ms. An additional independently invoked materialize/check took
6.205 ms. Runner elapsed time through witness construction was 11,713.444 ms;
`time -v` reports 11.81 s including process teardown, and the one-second watchdog
observed 12.013 s through process detection. These boundaries differ and must
not be summed. No whole-program solve time is inferred from the polling monitor.

Both backends used one worker. The new SCIP source reserves 10% of remaining
pipeline time for postprocessing, just as the historical CP-SAT run did; neither
should be described as receiving 600 seconds of optimizer time. The earlier CP
run is reused because its domain/model/source identities exactly match. Its
100,000-variable/500,000-constraint caps differ from this larger MILP caller's
caps. SCIP also has the checked baseline primal start and the guarded valid
unary-port lower bound described in
[the v32 implementation note](taskflow-joint-solver-v32.md). CP-SAT did not receive
that additional inequality. One trial with these different formulations is not
a general solver ranking or a statistical speedup estimate.

The strong bound applies only after its resource/port guards are proved. The
separate solver-independent configuration-weight certificate also returns a
196 ns lower bound: it checks the domain and enumerates four DMA orders and a
16-state per-slice exact-cover relaxation. SCIP's reported optimum remains
subject to its numerical tolerances, followed by independent integer schedule
checking. The certificate provides additional support for this particular
synthetic domain, not a hardware cost-accuracy claim.

## Witness validation and evidence

The solver and typed runner both materialized and checked the integer witness.
The runner initially serialized decisions without the optional graph argument;
the frozen CLI correctly refused that incomplete evidence because it requires a
stored `program_graph`. The original record and failed replay are retained.
A separate helper then parsed the stored decisions, rebuilt and independently
checked their graph, and wrote `solution.json.with-graph.json` without changing
any solver fields. Frozen-tool CLI replay of that complete record passed.
The solver was not rerun to fix evidence packaging.

Archived evidence is in
`/home/shiran/project/orbit-artifact/results/taskflow/joint-solver-scaling-v32-r1`:
`protocol.json`, `freeze.json`,
`build-manifest.json`, complete compiler/link commands and library hashes,
`source-snapshot/`, the unchanged inputs, raw/final results, process memory/time
receipts, the decision and complete-graph witnesses, failed and successful replay
logs, and the independent certificate/script. `results-final.json` is the final
summary. `archive-manifest.json` records every archived file hash. The large
runner/repack binaries and object files are omitted from the archive; the frozen
runner remains at `/tmp/orbit-solver-scale-v32-r2/runner`, with its hash and full
build provenance preserved. This experiment does not modify production code or
previous raw results.

The remaining scaling concern is visible despite the short proof: sixteen
instances already require over 360,000 MILP variables and about 3.18 GiB. The
optional interval/resource representation still grows with groups, placements
and configuration count. This result establishes a successful larger same-domain
proof; it does not establish that substantially larger domains fit memory or
prove within ten minutes.
