# Search/NPU v2 validation

Evidence for this round lives in `build/search-npu-v2/validation`; the preceding
`backend-decoupling` batch was not overwritten. The validation owner never
invokes the NPU; all hardware work is serialized by the NPU owner.

## Semantic and legacy regression

- Independent core `test/TaskflowCore/check.py` passes, including nondivisible
  M/N/K, fusion/fission, authenticated body roundtrip and illegal programs.
- 35/35 existing Amoeba lit tests pass. `LLVM_BUILD/bin` must be on PATH because
  negative tests execute LLVM's `not`. The initial missing-PATH attempt and
  corrected successful run are both retained.
- 14 public candidates across GEMM, GEMV, GEMM→ReLU and standard SwiGLU FFN,
  three seeds each, pass the independent whole-matrix float64 CPU oracle.
- All 11 malformed-program/body-witness rejection checks pass.
- All 14 candidates (42 seeds) also pass compiled MLIR→LLVM CPU replay.
  The replay now poisons temporary/output buffers with NaNs before each of
  three repetitions, outside timing, to detect missing initialization and
  stale reduction state. It retains the existing 2e-5 absolute/relative
  acceptance, adds normalized RMS reporting, and evaluates the original
  source graph in float64. These timings are CPU correctness diagnostics.
- Amoeba mapper scores are invalidated if any subsequent compilation or
  numerical replay fails; mapping evidence is retained separately.

## Search protocol adversarial checks

`test/TaskflowValidation/search_contract.py` uses an explicitly synthetic
protocol fixture; its scores are **not** NPU/CGRA performance evidence.
Sixteen checks pass, including:

- Two expansion rounds; first-round evaluation determines the next parent.
- Completed resume preserves archive without additional evaluations.
- Actual SIGKILL during evaluation after two checkpointed candidates, followed
  by resume, reproduces the uninterrupted archive and ranking.
- Compile/numerical/unsupported failures, absent scores, wrong candidate IDs,
  wrong score units, and nonpositive scores cannot enter ranking.
- Budget exhaustion retains unmeasured candidates with null scores.
- Changed configuration/evaluator and forged checkpoint identity/score reject
  recovery.

The public candidate identity remains separate from canonical `search_key`,
which is used for structural deduplication and parent selection. Both must be
interpreted according to their named fields, rather than equated.

```bash
cd /home/shiran/project/orbit
export PATH=/home/shiran/project/llvm-project/build/bin:$PATH
python3 test/TaskflowCore/check.py --tool build-core/tools/taskflow-opt/taskflow-opt
llvm-lit -sv -j2 build-amoeba/test
build-amoeba/.venv/bin/python test/TaskflowValidation/search_contract.py \
  --tool build-core/tools/taskflow-opt/taskflow-opt \
  --output build/search-npu-v2/validation/search-contract-reproduction
```

Use a fresh output directory for the interruption test: it preserves a
one-time interruption marker and intentionally changes its fixture evaluator
at the final invalidation check.

## Real original-driver Amoeba search

The final coherent batch is
`build/search-npu-v2/search/amoeba-gemm-formal-cacheguard/search.json`.
The original `search-joint-neighborhood` pass ran two rounds, beam width two,
seven measured candidates. Every measured candidate passed MLIR→LLVM replay
and the existing Neura mapper plus production scheduler.

| Round | Action chain | Scheduler cycles |
|---|---|---:|
| 0 | baseline | 6653 |
| 1 | N=10 | 6656 |
| 1 | K=12 | 6655 |
| 1 | M=3 | 6656 |
| 2 | K=12 → N=10 | 6660 |
| 2 | K=12 → K=6 | 6659 |
| 2 | K=12 → M=3 | 6660 |

The lowest first-round score selected K=12; N=10 won the deterministic tie for
the second beam slot. This feedback determined second-round parents. Two
additional candidates generated from N=10 were retained as budget-unmeasured,
with null scores. The baseline remains the best archive entry. These values
are mapper/scheduler cycle estimates, **not physical CGRA measurements**.
Each candidate retains actual mapper IR, logs, provenance and three-seed CPU
replay in its `evaluation/amoeba/` directory.

The earlier `amoeba-gemm-formal` batch is retained as diagnostic evidence:
its last evaluation correctly rejected an implementation fingerprint change
while the evaluator cache guard was being hardened. It is not mixed with the
final coherent comparison above.

```bash
cd /home/shiran/project/orbit
export PATH="$PWD/build-amoeba/.venv/bin:/home/shiran/project/llvm-project/build/bin:$PATH"
python scripts/taskflow_search_experiment.py search --backend amoeba \
  --case gemm --label reproduction --rounds 2 --budget 7 --beam 2
```

## Evaluation cache integrity

`test/TaskflowValidation/evaluation_cache.py` first reproduced a stale-score
bug: a changed positive cached response scalar could override intact mapper
evidence. The evaluator now derives the accepted score from the persisted
backend evidence and checks response identities. All six adversarial checks
pass: authentic cache hit; modified cached score, ID, or unit rejected;
modified evidence or inputs invalidate reuse. The before/after test artifacts
are `validation/cache-before*` and `validation/cache-after*`; they contain only
synthetic fixture results, never device timing claims.

```bash
build-amoeba/.venv/bin/python test/TaskflowValidation/evaluation_cache.py \
  --output build/search-npu-v2/validation/cache-reproduction
```
