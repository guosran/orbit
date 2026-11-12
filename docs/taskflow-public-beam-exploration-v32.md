# Public beam exploration across unknown prices

This change applies to the public finite-domain beam adapter. It changes
exploration within its declared choices and compatibility tables; it does not
extend source lowering or the set of realizable NPU mappings.

The public beam adapter previously retained only states with a checked cost.
A checker-feasible state with unknown cost could be evaluated once and then
never expanded. This disconnected two priced endpoints even when every
intermediate mapping passed the same checker.

`ProgramBeamExplorationOptions` adds a bounded FIFO frontier for these states.
By default one slot **within the configured beam width** explores this frontier;
`unknownSlots=0` restores scored-only behavior. Priced states still use the
shared `selectSearchBeam` selector. An unknown price remains JSON `null`, does
not acquire a surrogate score, and never beats a priced incumbent. Unknown
feasibility and infeasible states do not enter the queue.

The default pending limit is 256 assignments, further limited by
`maxTupleCells / max(1, dimension_count)`. Overflow is reported as dropped
frontier states; it is not a proof of irrelevance. Each selected state expands
once, assignments remain globally deduplicated, and callbacks share the existing
wall, evaluation and round budgets. The ranked beam is retained independently
when exploration temporarily occupies its slots. If exploration would consume
all slots, alternate rounds reserve a slot for pending priced work; a width-one
beam cannot permanently starve that frontier. Queue size, drops, expansion count and the
priced/exploration role are recorded. The adapter continues to report neither
finite-domain exhaustion nor global optimality.

## Frozen r12 algorithm replay

The diagnostic uses all 48 previously recorded checked callback results as an
oracle for the **C++ production beam adapter**. It performs no new compilation,
physical evaluation or price fitting. The recorded domain ID is
`863786faba54a3af3b08bb13aad948ce06b15da1b2fc8d30d6c31d8837941b22` and evaluator
ID is `95611d9e0cd86c5d098ee5b16c6dfbe4b6e10bffb75ca4aba7c298fcb0f16628`.
Both runs use width 4, 64 rounds and at most 48 callbacks.

| Adapter policy | Evaluations | Best recorded cost (ms) | Unknown states expanded |
| --- | ---: | ---: | ---: |
| Scored only | 15 | 0.20358412879419976 | 0 |
| Bounded FIFO | 48 | 0.004773345100045475 | 32 |

FIFO first reaches the winning recorded cost at callback 46; peak pending size
is 19, with no drops. This reproduces the old failure and crosses its pricing
barrier without assigning prices to the 42 mixed mappings. It does not establish
hardware speedup or improve the historical CP proof status. Runner, result and
build commands are preserved under `/tmp/orbit-beam-r12-trace-replay-r4.json` and its adjacent runner/links and
`/tmp/beam-replay-commands.json` for experiment archival.

The CPU contracts additionally exercise a generic four-dimensional pricing
barrier, explicit disabling, null-cost evidence, finite queue/evaluation limits,
invalid options and exclusion of unknown-feasibility/infeasible states.

## Remaining neighborhood boundary

This change retains legal single-dimension neighbors. Fully priced placement
landscapes can still contain launch-cost barriers. A future source-derived
neighborhood can move the tasks of a current execution region together to any
other device allowed by their domains, in either direction, and submit the
result through the same compatibility tables and full evaluator. This requires
an explicit bounded descriptor/callback from the public source adapter; the beam
must not infer operator semantics or hard-code a preferred device. No such
coordinated placement move is claimed by this checkpoint.

## Frozen r13 CLI integration

The immutable `/tmp/orbit-v32-beam-frontier-r13/taskflow-opt`, SHA-256
`46f28808482500d9b40c33d465bdb9c2ac48690b7c8aa41e528574768d05587a`, passed
these existing compiler-only CLI contracts, with no rebuild or device use:

- `program_all_search_v25.py --tool TOOL`: 30 cases.
- `program_search_identity_v26.py --tool TOOL`: 8 cases, including CP/beam
  shared-domain/evaluator identity and conservative beam proof status.
- `program_replay_v27.py --tool TOOL`: 10 cases.

The 48 cases took 2.56 seconds in aggregate. Full commands, timing, output logs
and the tool freeze receipt are recorded in
`/tmp/orbit-beam-r13-cli-regression-r1/receipt.json`. Together with the independent
ON/OFF 27-case typed contracts, these verify integration and the exploration
contract; they do not establish a hardware performance result.

## Metadata correction after the r13 freeze

The r13 frozen CLI's top-level `cost_model_boundary` retained a legacy,
hard-coded partition-host-assembly warning. The nested typed evaluator receipt
correctly recorded `prepared-host-to-host: reset, exact serial regions, output
copy`, including its application control phase. The later source correction
reads the selected timing receipt and labels that scope explicitly; absent or
empty receipts remain unknown, and actual partition-template host assembly
remains unknown. Boundary declaration and whether a checked price is available
are separate report fields. This metadata-only correction is not present in the
r13 frozen tool and does not change its original receipts, prices or measurements.
