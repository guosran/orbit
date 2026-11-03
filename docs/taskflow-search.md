# Structured search through the ORBIT driver

`search-joint-neighborhood` now accepts `decision-flow=structured` and
`backend=amoeba|npu|static`. Its existing default `decision-flow=legacy` and legacy
Neura search options retain their behavior. Structured mode takes a static
`structured-plan` source explicitly; the pass's input module may be empty.
The search materializes each candidate as a real authenticated Taskflow module.
It does not claim to translate arbitrary legacy Neura bodies to the public core.

The driver calls `orbit::taskflow::runStructuredSearch` in
`lib/Taskflow/Search.cpp`. The independent `taskflow-opt --search` calls exactly
this same component. The scored-only beam selection, canonical-key deduplication
and diversity selection routine was extracted from the original driver's
`selectBeam` into `include/Taskflow/Search.h`; both legacy and structured modes
use it. Neura-specific cost, topology, native controls and legacy journal remain
in the legacy driver. Structured mode has its own versioned checkpoint schema
because its scores and materialized programs are not Neura search states.
The shared implementation links only Taskflow and LLVM support, not Neura/AIR.

The five boundaries are explicit: `generateSearchActions` proposes actions
from the current parent rectangles; `transform` clones/materializes the child;
`validate` checks common semantics; the external evaluator checks target
executability and numerical correctness; only a valid explicitly typed score enters
beam selection. Supported actions are identity, M or N task partition,
sequential K partition, and the existing restricted matmul/ReLU fusion/fission.
Target configuration may supply `search_preferences` with positive integer
`m_alignment`, `n_alignment`, and `k_alignment` recommendations. Public defaults
are all one; NPU launch configurations supply their 64/256/64 preferences.
These affect proposals only and never semantic legality.
This is a bounded heuristic, not an exhaustive search or a performance model.

Round zero evaluates the untransformed source. Each round expands only the
previous round's evaluated selected children. Ranking selects lowest cost first
(with canonical identity as the deterministic tie break); global archive keeps
the baseline and all previously scored candidates, even when expansion explores
a slower child. Thus a local regression can still yield a later combination,
and the overall reported winner cannot lose the baseline. Unsupported or
invalid actions, duplicates and budget-unmeasured candidates are retained in
the trace but never assigned fabricated scores. Search is synchronous, so an
evaluator's NPU lock must not be wrapped in another outer lock.

## Identities and evaluator contract

The existing public `candidate_id` is preserved unchanged. `search_key` is a
separate SHA256 of program semantics after clearing transform history and
candidate id and normalizing task names/dependency order. This allows equal
materialized programs reached through different histories to deduplicate.
Parent links, beam ids and ranking use `search_key`; archive rows and selected
rows also carry the actual public candidate id. The archive's program retains
the complete public transform history. Invalid and duplicate edges are in
`attempts`; successful and failed materialized candidates are in `archive`.

The evaluator must be executable and accepts two positional file paths:
`evaluator request.json response.json`. Requests contain:

```json
{"schema":"orbit.search.evaluation.v1","backend":"npu",
 "candidate_id":"public id","search_key":"canonical search id",
 "source_id":"source id","plan_path":"/absolute/plan.json",
 "mlir_path":"/absolute/taskflow.mlir","output_dir":"/absolute/evaluation",
 "target_config":{}}
```

Responses carry matching `candidate_id` and `source_id`, a `status` of
`passed`, `unsupported`, `compile_failed`, `runtime_failed`, or
`numerical_failed`, and, only for passed evaluations, a finite positive
`score.value` with `score.unit=ms` (NPU/static) or `cycles` (Amoeba). Responses may
add `score.provenance`, `evidence_path` and `cache_identity`; the complete
response is preserved. Static responses additionally require
`evaluation_kind=static_estimate`; their millisecond estimates are never labeled
as measured NPU execution. The provided static evaluator independently verifies
the complete plan and records `executed=false`. Failed invocation, malformed/missing score, wrong id,
wrong units or invalid numeric score becomes `runtime_failed`. Milliseconds
never enter a Neura II or cycle field. The evaluator owns numerical reference,
compiler/runtime provenance, target plan and backend cache validation.

## Checkpoints

`checkpoint.json` is atomically replaced after each completed evaluation and
round selection. `--resume` restores the committed archive and current beam;
a partially expanded round is reconstructed through canonical deduplication.
An interrupted uncommitted evaluator call may run again. Only committed
measurement responses are reused. The binding includes the source semantics,
backend, all target configuration, evaluator executable bytes, the running
search executable bytes, existing file paths recursively found in the config,
and search budgets. The integration config additionally carries an
`implementation_fingerprint` covering imported backend code and toolchains.
Changing those identities rejects resume. Recursive file binding uses sorted
object keys so identity is stable across processes; this was verified with the
full multi-file static configuration, not only a synthetic scalar config. Candidate bodies and successful
response ids/status/unit/value are checked again during restore.

## Invocation

```bash
cd /home/shiran/project/orbit
# config.json supplies target/evaluation settings, including inputs/reference.
build-amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt empty.mlir \
  --search-joint-neighborhood='decision-flow=structured backend=npu structured-plan=/absolute/source.json evaluator=/absolute/evaluator.py target-config=/absolute/config.json output-dir=/absolute/run max-rounds=2 max-candidates=9 beam-width=1'

build-core/tools/taskflow-opt/taskflow-opt /absolute/source.json --search \
  --backend=npu --evaluator=/absolute/evaluator.py \
  --target-config=/absolute/config.json --search-output=/absolute/run \
  --max-rounds=2 --max-candidates=9 --beam-width=1
# Add --resume to the same command to restore its checkpoint.
```

`search.json` records all scores, selections and explanations. Per-candidate
folders contain the public plan, authenticated Taskflow, evaluation request and
response, and evaluator artifacts. Synthetic evaluator tests in
`test/TaskflowValidation/search_contract.py` verify protocol behavior only;
their scores are not hardware performance evidence.
