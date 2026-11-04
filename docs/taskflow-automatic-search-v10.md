# Source-derived native proposals

The original C++ whole-program search driver supports opt-in automatic native proposals. Python does not generate a tile grid or plans. `GenericAnalysis` supplies source loop iteration domains; the target supplies memory capacity, implementation capabilities and tile alignment. The current native tile interface addresses source loop indices0/1/2, conventionally M/N/K, rather than inferring arbitrary reduction axes or recognizing named arithmetic.

Example target/search configuration:

```json
{
  "generic_air": {
    "capacity_bytes": 24576,
    "capabilities": {
      "tile_alignment": [1, 8, 1],
      "implementations": [
        {"kind": "scalar", "lanes": 1},
        {"kind": "scalar-product-vector-add-f32", "lanes": 8}
      ]
    }
  },
  "native_auto_candidates": {
    "mode": "joint",
    "max_tile_proposals": 12,
    "exploration_slots": 2
  }
}
```

Target memory architecture/binding is configured through the public static resource, memory and link types; search reads `GenericTargetCapabilities` and does not create a second memory model. Target alignment and optional search preference alignment are combined by their least common multiple. A finite neighborhood proposes half, double, minimum-aligned and full-dimension extents, beginning with spatial index1, then2 and0. Current tiles come from the materialized module, or from its checked plan when no explicit tiles exist. Each proposal is materialized in a clone, re-analyzed and passed through the actual public planner. Unavailable prechecks are recorded, with reasons and target-capability identity, in `automatic-proposals.json`. This phase has no cost-model requirement and does not mistake unknown link cost for illegal communication.

`joint` combines those tile actions, declared implementation actions and source-discovered retention choices. `tiling-only` excludes implementation/residency actions; `implementation-residency` excludes tile actions. Existing explicit candidate lists remain available; with automatic proposals disabled their path is unchanged. Automatic and explicit lists can be combined, subject to the chosen mode. Implementation legality, tails, source dependence and capacity remain planner/backend checks; discovering an opportunity does not prove it legal.

Each child still flows through the original transform, analysis, candidate identity, evaluation, archive, beam and checkpoint implementation. A tile action changes actual module attributes and its resulting plan. Retention can then compose in a second round, or vice versa. Candidate identity is the canonical materialized MLIR; target/model/configuration and native binary bindings remain part of evaluation identity. Resume reconstructs current proposals and replays actions rather than trusting action names alone. The default scalar implementation has one canonical spelling: absent implementation/lane attributes. Explicit scalar/1 requests and transitions back from mixed execution remove those attributes. Source inputs with legacy scalar/1 attributes are normalized before identity, so aliases cannot consume separate beam slots or extra action depth; other implementation choices retain their explicit attributes.

## Unknown-cost exploration

`exploration_slots` defaults to zero and must lie between zero and beam width. When positive, it reserves at most that many parent positions for **legal but uncalibrated** native plans. These candidates retain `status=uncalibrated` and `score=null`; they do not enter the scored ranking. Selection is deterministic by candidate key, deduplicated through the original archive, and labeled `uncalibrated-exploration`. Unsupported, failed, and budget-unmeasured candidates are ineligible. If fewer unknown candidates are available, ordinary scored candidates fill remaining positions.

An uncalibrated baseline can start exploration. A model OOD candidate need not disappear solely because no prediction is available. This is an explicit exploration budget, not an invented prediction or measured label. Checkpoint replay independently rebuilds uncalibrated plans and validates their null scores and selection history. A future device measurement belongs to a separately bound feedback provider.

The effective mapping target is resolved once: when the outer search configuration uses `mapping_config`, capabilities and automatic prechecks use that same nested configuration as evaluation. Search controls remain in the outer object. This prevents proposals from silently using default capacity while evaluation uses a different target.

Failed proposals are requests, not valid plans. `automatic-proposals.json` records their canonical native MLIR, candidate SHA, requested tile/policy/implementation fields and native planner error. Evaluated candidates also retain requested fields in `request.json`. An auditor can verify source bytes and replay the exact request through the compiler; an unsupported request does not gain a fabricated `best_plan`.

Large legal unknown plans retain complete `mapping-result.json` files once per candidate. Automatic exploration feedback/checkpoints carry only required legality flags, null estimate, plan identity and a canonical digest of the complete result. Resume rebuilds the plan with current analysis, compares its complete canonical digest and plan identity, and checks the summary; it does not establish validity merely by trusting a previous disk file. Explicit and scored/observed feedback paths keep their existing representation.

## CPU regression and limits

`test/TaskflowGeneric/automatic_search_v10.py --tool PATH --output NEW_DIRECTORY` exercises two rounds of each mode under identical configured round/evaluation/beam limits, audits real source/plan identities, checks tile→retention composition and target-capacity legality, resumes searches, rejects target-binding changes, and preserves the legacy explicit-list path. A separate no-model case checks two rounds of null-score exploration and rejects forged scores/selection reasons. Its explicit theoretical test provider is an uncalibrated proposal baseline, not performance evidence.

Equal evaluation limits do not mean equal compiler CPU cost: automatic prechecks also plan candidates, and modes visit different numbers of proposals. The proposal bound is per selected parent. Static byte counts are planned transfer payloads, not DDR counters. No multicore placement, hidden precision change, arbitrary cross-stage SRAM retention, or device execution is introduced by this search extension.

## Frozen predictive comparison and novel archive selection

The companion `taskflow_implementation_experiment_v10.py predict` writes both predeclared holdout predictions and a separate `predictive-search-inputs.json`. Its `predictive-search` command invokes the original C++ search with the frozen primary model. The registered matrix-coverage `propose`/`selected-cases.json` path is not production winner selection. Separate `select`, `compile-selected` and `prepare-selected-profile` phases freeze a novel native-ranked winner, implement it and hand it to the existing hardware profiler. The clean-shell `all` entry runs this complete sequence and its independent audit.

For an equal-budget CPU comparison, derive three configurations from the same frozen predictive configuration, changing only `native_auto_candidates.mode`: `joint`, `tiling-only`, and `implementation-residency`. Use the same frozen tool, M32/K128/N64 source, root tiles8/16/32, model and target; retain six rounds,512 maximum evaluations,beam16,12 tile proposals, four unknown exploration slots, two structural diversity slots, and preference alignments4/8/16. Each mode needs its own output/checkpoint and a manifest binding its derived configuration. The existing companion predictive CLI has no mode override; the other modes can use the original native CLI directly:

```sh
"$TOOL" "$SOURCE" --search --backend=air --air-distribute-initialization \
  --tile-m=8 --tile-n=16 --tile-k=32 \
  --target-config="$MODE_CONFIG" --search-output="$NEW_OUTPUT/search" \
  --max-rounds=6 --max-candidates=512 --beam-width=16
```

These are equal limits, not guaranteed equal actual evaluations or compiler CPU work. Record early exhaustion, actual evaluations, prechecks, archive status counts and elapsed time. Model-OOD candidates may consume exploration slots but retain null scores and cannot become predictive winners.

Before reading new timings, freeze exclusions for the twelve training executions and the four previously measured v9 executions. Traverse `search.json`'s native `ranking` in order; require a supported calibrated plan, positive finite score and native `model_prediction` feedback. Skip prior executions using their actual source/semantic execution identity, not candidate/plan hashes alone: changing compiler, target or model metadata does not make an old execution novel. Conversely, identical tiles on different source programs do not alone establish equivalence. Select the first eligible novel row without restricting selection to the six holdout coordinates or re-ranking in Python. If none remains, report that outcome. Freeze the selected rank, skipped exclusions, prediction and complete identity chain before fresh compilation or profiling.

The authoritative archived source is `search/candidates/KEY/taskflow.mlir`; the complete native mapping is `mapping-result.json` under `best_plan`, including `plan_id` and `direct_air_cost.features.execution_fingerprint`. The adjacent `plan.json` is only a native interface placeholder. Preserve `request.json`, `response.json`, `search.json`'s parent/action/action_chain/evaluation_identity fields, `binding-components.json` and `checkpoint.json`. Source/candidate identity, mapping identity and evaluation identity serve different purposes. Archive legality still requires backend compilation, final transfer/port/physical audits and numerical validation; it is not evidence of execution.

The completed experiment selects the first novel native-ranked `[4,32,64]` mixed8 phase-local candidate and executes it with both restricted-mode winners. A separate budget-one hardware exploration executes the first eligible legal unpriced archive entry (`[8,8,32]` scalar); it retains its null prediction after measurement. The four candidates pass 84 independent output checks. Joint and implementation/residency predictions tie, and paired timing does not establish a joint-search gain over that control. The search mechanism and new-candidate execution are verified independently of a speedup claim. No model prediction or static byte count is a hardware measurement; see the companion v10 report and the [validation record](taskflow-implementation-search-v10-validation.md).
