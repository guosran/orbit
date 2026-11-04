# Optional direct AIR cost provider

`generic_air.cost_model` selects an optional native feature/composite provider.
Without that key, existing mapping output and the legacy theoretical formula are
unchanged. `theoretical` and `cost_model` are mutually exclusive. This provider
does not change the CGRA ML model, computation, transfers, or target legality.

The `features-only` provider with schema `orbit.direct-air.cost.v1` records native
facts and leaves cost unknown. Features come from actual source operations and a
checked mapping: source loop multiplicities, scalar f32 operation classes,
loads/stores, implementation identities, actual planned DMA bytes, phase/group/
stage counts, payload capacity, and local initialization/staging. Source product
operations count `arith.mulf` and `arith.divf`; source exp counts `math.exp` with
f32 operands/results. These counts are not generated instruction/helper counts.
Stage count is not command count. Single-program native plans currently represent
one submitted command; partition controller and host slice/assembly costs need
separate calibration.

A `linear-composite` model supplies frozen nonnegative coefficients in milliseconds
per feature unit. Up to four terms permit command count, source product count,
source exp count, and hardware stage count without treating exp as a cheap add or
confounding command overhead with every stage. Other known native feature names
are accepted for explicit experiments. Terms describe one full observed host call;
they are not independently measured physical components. DMA effects already
included in that label must not be charged again. Legacy theoretical coefficients
are null in the composite report.

A calibrated model must identify its model/version, SHA256 training manifest,
target/toolchain/runtime, timing boundary, supported source fingerprints,
implementation identities, operation names/dtypes, and feature ranges. Evaluation
requires the matching `generic_air.cost_target` identity. Unsupported source,
implementation, storage dtype, feature range, target, or timing boundary produces
`out_of_domain` with null cost. Uncalibrated models and zero-only estimates remain
unknown. Malformed coefficients, identities, ranges, overflow, or mixing charge
owners fail closed. The current calibrated boundary is
`prepared-command-host-to-host`: prepared input host arrays through returned output,
including runner/transfer effects, excluding context load, input/reference loading,
reference checks, and output serialization. A partitioned whole program is not a
calibrated sum of child estimates when controller/slice/assembly costs are unknown.

The model, target context, native features and digests participate in plan identity.
The checker regenerates facts and reevaluates the provider, rejecting mutated
features, model identity, estimates, or legacy coefficient charges. The compiler
reports calibration provenance; it does not turn a synthetic contract test into a
hardware calibration claim.

Experiment fitting, timing audits, training/holdout manifests and frozen parameter
files live in `orbit-artifact/taskflow`. Heldout labels must be measured only after
parameter freeze. Correlated FFN shapes are a shape-generalization experiment,
not independent workload families. Host polling quantization limits interpretation
of tiny control-graph stage differences. Unknown or out-of-domain costs remain
unknown to search; coefficients for scalar execution do not imply support for a
new vector implementation.

`domain.source_feature_ranges` optionally narrows configuration coverage per source:

```json
{"source_feature_ranges": {
  "<source fingerprint>": {
    "max_tile_loop_0_iterations": [8, 8],
    "serial_tiled": [1, 1],
    "dma_bytes": [123456, 123456],
    "core_count_per_stage": [1, 1]
  }
}}
```

When present, every evaluated source needs a declared nonempty entry. Its native
ranges intersect the global ranges. Invalid/unknown range keys fail; a source
without an entry or a value outside its range returns unknown. A model tightened
this way receives a new model/version and digest; frozen coefficients and the
original model identity remain preserved.

`kind: "observed-table"` supplies exact complete-command feedback without fitting
or generalization. It has `model_id`, `model_version`, `target`, `boundary`, and
`rows`. Every row contains `execution_fingerprint`, positive finite `median_ms`,
positive integer `sample_count`, lowercase SHA256 `raw_manifest_sha256`, plus
`target` and `boundary` exactly matching the table. Duplicate execution keys and
malformed rows fail even if they would not match the current candidate. Matching
runtime/target/boundary and execution key yield `status: "measured"`; a missing
key returns null cost. The raw digest records provenance, not an in-compiler
reexecution or verification of the external raw measurement files.

The execution fingerprint hashes the serialized full plan with price metadata
removed: provider facts/model, score/hasCost, theoretical coefficients,
provenance and plan ID. Source, transfers, regions/versions/copies, grouping,
schedule, resources and actual implementation remain bound. Thus a different
mapping with coincident operation counts cannot borrow another observation.
Provider/model digest still participates separately in the priced plan ID.
This feedback can compare measured scalar/mixed/tiling/residency candidates;
unmeasured candidates remain unknown. It is not a compute-only estimate and must
not be added to separate DMA or runtime charges.

The native search feedback envelope preserves the provider distinction:
`evaluation_kind=measured_feedback` and score provenance
`exact-observation-table` identify observed-table scores. Its
`physical_hardware_measurement=true` describes the historical score's origin;
`executed=false` records that this evaluation reused the observation rather than
submitting hardware. A calibrated composite instead uses `model_prediction` and
`calibrated-direct-air-model`, with the hardware measurement flag false. Native
checkpoint recovery reevaluates the plan and checks these provenance fields;
changing a measurement into a prediction/static estimate or claiming a fresh
execution is rejected.
