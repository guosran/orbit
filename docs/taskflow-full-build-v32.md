# Full Neura / AIR build checkpoint (v32)

The independent ON and OFF build trees completed their default-all Neura/AIR
C++ builds and the CPU regressions below against the stable source checkpoint
used for prep9. This is a compiler/software checkpoint. These tests did not run
an NPU, measure a hardware speedup, or certify every physical lock schedule;
formal device and calibration evidence is maintained separately.

| Check | ON | OFF |
|---|---:|---:|
| Default-all C++ Neura/AIR build | PASS | PASS |
| JSON ownership | 6 | 6 |
| Per-task implementation contracts, including actual AIR | 80 | 80 |
| Explicit mixed direct-cost guard | 9 | 9 |
| Joint public heterogeneous choices | 36 | 27 |
| Heterogeneous costs, including context capacity | 56 | 56 |
| CPU-inactive choices | 9 | 6 |
| Regional implementation projection | 27 | 27 |
| Fake executor/runtime contracts, including context capacity | 19 | 19 |
| Static runtime positive/negative fixtures | 11 | 11 |
| Matrix source-to-AIR contracts | 11 | 11 |
| Scalar BF16 RNE trigger and generated-IR constant contracts | PASS | PASS |
| Joint MILP typed contracts, including dynamic pricing scope | 65 | 39 |
| Joint domain audit / solver CLI contracts | 28 / 86 | 28 / 86 |
| Earlier joint CLI finite-domain contracts | 23 | 23 |
| Search / beam / routing / graph / mechanism | 29 / 17 / 27 / 42 / 77 | 8 / 17 / 27 / 42 / 77 |
| Public CLI / identity / replay contracts | 30 / 8 / 9 | 29 / 8 / 9 |
| Standalone solver option ranges | 8 | 8 |
| Direct heterogeneous CLI tile precedence | 6 | 6 |
| Invocation cost / invocation CLI contracts | 96 / 227 | 96 / 192 |
| Region composite / native composite timing | 48 / 13 | 48 / 13 |
| Native composite CLI / feature exports / native errors | 6 / 2 / 27 | 6 / 2 / 27 |
| Offline calibration / calibration index / feature design | 12 / 5 / 1 | 12 / 5 / 1 |
| Taskflow core semantics | PASS | PASS |
| Neura/Amoeba lit | 35 / 35 | 35 / 35 |

OFF disables both OR-Tools and SCIP while retaining AIR and Neura C++ support.
Its smaller typed/search counts reflect explicit unavailable/baseline-only
paths; they do not establish a solver optimum. The solver range contracts use
one worker for valid boundary tests, including `INT_MAX` seeds. The typed joint
regression also appends a structurally supported but unpriced route with stale
caller metadata: both solvers retain a checked incumbent without declaring a
full-domain optimum, and serialized pricing and bound scopes survive replay.

Build roots are `/tmp/orbit-full-v32-build` and
`/tmp/orbit-full-v32-off-build`, Release, using the pinned LLVM/MLIR and `-j2`.
Runtime Python is `/home/shiran/project/aie-toy/.venv/bin/python`; offline SciPy
contracts use `/tmp/orbit-v29-fit-env/bin/python`. Optional Python bindings
remain disabled because the pinned upstream MLIR build has no Python binding
sources; that existing build capability restriction is unchanged.

Final ON build/regression logs are
`/tmp/orbit-full-v32-final-build-on-r6.log` and
`/tmp/orbit-full-v32-final-regressions-on-r6.log`. Final OFF logs are
`/tmp/orbit-full-v32-final-build-off-r7.log` and
`/tmp/orbit-full-v32-final-regressions-off-r7.log`. Matrix logs are
`/tmp/orbit-full-v32-matrix-final-on-r6.log` and
`/tmp/orbit-full-v32-matrix-final-off-r7.log`. The machine-readable checkpoint
with tool hashes, settings, actual counts and scope is
`/tmp/orbit-full-v32-final-checkpoint-r6.json`.

The first OFF suite rejected the unsafe repeated-feed fixture at the old input
guard instead of the new output guard. Its runtime object contained no
multiple-output diagnostic despite its timestamp following the source write,
so an ordinary incremental build had retained stale code from an earlier
source/compile overlap. Only that build object was removed, the unchanged
current source was rebuilt, and the identical full suite passed, including all
11 runtime fixtures. No diagnostic assertion was relaxed. The initial failure
is preserved in `/tmp/orbit-full-v32-final-regressions-off-r6.log`, with object
and binary hashes and diagnostic strings in
`/tmp/orbit-full-v32-off-stale-runtime-r6.sha256` and
`/tmp/orbit-full-v32-off-stale-runtime-r6.strings`.

Earlier failing checkpoint logs and frozen hardware artifacts remain separate.
In particular, the prep7 timeout is not erased by later CPU checks. The
metadata-provenance repair's normal-case equivalence receipt remains
`/tmp/orbit-v32-provenance-normalcase-r1/receipt.json`; it proves byte-identical
Taskflow source/AIR for that checked gemm16 export, not general device liveness.

The same-domain SCIP scaling followup is documented separately in
[the solver scaling note](taskflow-joint-solver-scaling-v32.md). It proves a
synthetic finite-model optimum and does not supply a hardware speedup claim.


## Runtime default-property compatibility increment

The subsequent parser-only compatibility increment completed default-all builds
in both independent trees. LLVM format 23 check-only validation passed for all
171 owned Taskflow C++ headers/sources/tests, including the new compatibility
header and test; no formatting edits were needed. The new unit contract checks
two exact `arith.trunci` default removals, five negative encodings/promises,
preservation of unrelated flags and attributes, idempotence, a standard SHA-256
vector, and acceptance by the registered pinned dialect parser.

The related ON/OFF regression targets all passed: runtime compatibility,
static runtime (11), fake runtime/lifecycle (19), heterogeneous context/cost
(56), BF16 RNE and Taskflow core semantics. This increment did not rerun long
solvers or the entire earlier suite. Commands were `ninja -C BUILD -j2`, then
`ninja -C BUILD -j2 check-taskflow-runtime-compatibility
check-taskflow-matrix-runtime-static check-taskflow-bf16-rne
check-taskflow-heterogeneous-runtime
check-taskflow-program-all-heterogeneous-cost check-taskflow-core`.

Independent CLI replay used the existing real MLP2 extracted generic IR and
sealed runtime contract, with fresh outputs for each build. Both tools removed
exactly two defaults, recorded independently verified input and normalized-byte
hashes, and emitted the same runtime bytes as the prior checked realization:
`4bd2c91b46082fa5bc7fd3e679db2879e91993aed4ef61edd9a8dcbb982a4927`.
A nondefault `nsw` mutation was rejected without writing runtime/proof outputs.
No vendor compiler or device was invoked, and old raw artifacts were unchanged.

Build and regression logs are
`/tmp/orbit-runtime-compatibility-{build,regressions}-{on,off}-r1.log`.
The check-only format manifest is
`/tmp/orbit-runtime-compatibility-format-all-r1.json`; CLI commands, outputs and
byte-hash summaries are in
`/tmp/orbit-runtime-compatibility-validation-r1/{on,off}`. The exact tool hashes
and scope are recorded in
`/tmp/orbit-runtime-compatibility-checkpoint-r1.json`. These parser/CPU results
do not extend any frozen hardware measurement's provenance to a newer tool.
