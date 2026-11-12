# ORBIT review cleanup and LLVM format

This change makes the owned C++ sources consistent and fixes correctness issues
found by applying the [distilled Coredac review guidance](review-guidelines-from-guosran-prs.md).
The scope includes new direct SCIP integration as well as existing search and
joint-domain code. It does not establish a new hardware speedup.

## Comment collection

The read-only collection covers all discovered public PRs whose actual author is
`guosran`: 25 in `coredac/neura` and 6 in `coredac/amoeba`, including open, closed,
merged and unmerged PRs. Three Copilot-authored search results are supplementary.
All three comment/review endpoints have a final page shorter than 100 records.

The 105 response pages contain 1,077 events on the strict 31-PR population:
74 conversation comments, 620 inline comments/replies and 383 submitted reviews.
Role classification identifies 500 human-reviewer events, 523 author replies and
54 bot events. Of the human events, 378 have nonempty bodies. Empty approvals
are events, not substantive advice. The guidelines distill 12 rules and link 47
specific comments; every link is checked against the raw cache.

Deleted or unpublished comments cannot be recovered. REST collection does not
certify whether a GraphQL review thread is resolved. No GitHub messages were sent.

## Human review order

1. Read `.clang-format` and the format-check script to assess the mechanical
   changes. `BasedOnStyle: LLVM` is the single repository rule. The formatter used
   for this pass is clang-format 23.1.5.
2. Read `ProgramJointSearch.h`, `ProgramJointDomain.cpp` and
   `ProgramJointPlan.cpp` for domain completeness, structural support, costs and
   independent checking.
3. Read `ProgramJointSolver.cpp` and `ProgramJointMILPSolver.cpp` for encodings
   of the same declared domain. The SCIP integration is optional and disabled
   by default.
4. Read `ProgramAllSearch.cpp` for configuration, budget and evidence-write
   failure propagation; then inspect the targeted positive and negative tests.

All 290 owned `.h`, `.cpp` and handwritten `.inc` files under `include`, `lib`,
`tools` and `test` are checked. Third-party sources, generated build files,
frozen experiment copies and unrelated presentations are excluded. The broad
legacy Neura edits are mechanical formatting; the semantic fixes are listed
below. Existing uncommitted work predates this cleanup and is preserved.

Reproduce the format check with:

```sh
python3 scripts/check-cpp-format.py --clang-format /path/to/clang-format
```

The saved before-format source snapshot and a separately generated formatting
patch allow reviewers to distinguish formatting from subsequent semantic edits.
Comments explain constraints, numerical limits, interval semantics, failure
states and supported scope. They avoid drafting conversations and claims that
the implementation does not establish.

## Rule → finding → change → validation

| Guidance | Concrete finding and resulting behavior | Validation |
|---|---|---|
| R4, R5, R11: separate structure, cost and proof scope | Structural route support previously depended on successful pricing. Supported but unpriced alternatives now remain a cost-coverage gap; a solved known subset cannot claim full-domain optimality. | Domain audit, dual-solver CLI tests: known plus unknown routes, all prices missing, capped generation and explicit strict assumptions. |
| R4, R6: shared semantic helpers and explicit capabilities | Transfer support is shared rather than reimplemented through cost queries. A bandwidth-only override is honored; a missing required counterpart remains unknown. Unknown capacity rejects an otherwise selectable local memory, while unused unknown memories are allowed. | Real Taskflow MLIR and readable architecture/cost fixtures exercise partial overrides, supported routes and usable versus unused memories. |
| R1, R6, R11: check the actual witness | Public schedules use binary64 values, but durations are integer nanoseconds. Inputs above `2^53-1`, unsafe byte conversions and overflowing baseline sums are rejected before loss of precision. Supplied witnesses are independently checked. | Boundary values, overflowing configuration/completion additions and an unsafe supplied witness are rejected. |
| R5, R11: missing information cannot prove impossibility | Native composite provider/schema errors previously became mathematical `INFEASIBLE`. Configuration preflight and structural witness construction are separated from complete-call pricing. | Wrong schema, absent calibration, mismatched fingerprints and a positive priced control. Unknown price has unknown feasibility. |
| R7: budget accounting must identify the actual phase | A positive budget exhausted during domain discovery previously became invalid input after subtraction. It now produces `UNKNOWN` before invoking callbacks. Joint CLI receipts separate preprocessing, model build, search and later evidence work. | Tiny valid budget, invalid original budget, callback counts and timing receipts. External process wall time remains the full measurement boundary. |
| R3, R12: evidence failures must be visible | Buffered MLIR/JSON write failures were not consistently checked. Search closes streams, acknowledges stored errors and propagates failure; a failed JSON temporary is not renamed as successful output. | `/dev/full` tests cover candidate MLIR, final best MLIR and evaluation JSON; each reports a controlled failure with the affected path. |
| R3, R6, R9: test constrained behavior independently | The direct SCIP backend models the same exact cover, placement, routes, barriers and resource constraints as CP-SAT. Capacity greater than one uses exact start-anchor occupancy of half-open intervals. | Six typed cases compare SCIP, CP-SAT and an independent integer scheduling oracle, including alternate routes, capacity two, hierarchy and zero-duration boundaries. |
| R10: expose meaning in names and comments | Joint-solver domain, options, builder, horizon and dimension names replace opaque single-letter bindings. Comments explain why exact cover, route selection, resource occupancy and independent checking are required. | LLVM format check and the same solver/domain regressions; no new mathematical shortcut is introduced. |

The MILP backend rejects unsupported worker counts instead of silently ignoring
them. It enforces numerical and model-size bounds and retains a checked baseline
when unavailable, limited or timed out. A fallback baseline is not a solver
incumbent or an optimality certificate.

## Validation and limits

Both optional-solver ON and OFF builds pass the targeted regression suites.
With both enabled, the new joint tests pass 36 typed MILP checks, 28 domain-audit
checks and 86 CLI checks. The unavailable backend passes its 25 typed checks.
The native error suite passes 27 checks in both builds. Existing core, public
search/replay, heterogeneous search, invocation, region-cost, native timing and
joint-search regressions also pass. The timing suite passes 93 checks. The
legacy Neura backend is disabled in these builds; its format-only files have
not received a new legacy compiler or board run in this cleanup.

The optional SCIP backend and its comparison workloads are described separately
in [the solver comparison](taskflow-joint-solver-comparison-v31.md). Costs in
those encoding/scale experiments are synthetic. The current joint solver domain
is NPU-only and restricted to supported row-slice/identity-grid pointwise
programs. It is not the full CPU/NPU heterogeneous search, arbitrary matrix
program lowering, or a calibrated hardware oracle. Scalar compute costs shared
by selectable places do not establish a physical multicore speedup.

Raw pages, coverage manifest, evidence index, formatting receipts, regression
logs and reviewed source snapshots are preserved in the versioned
`orbit-artifact/results/taskflow/review-cleanup-v31-r1` directory. Earlier v30
inputs, compiler binaries, cost models, outputs and negative results remain
unchanged.
