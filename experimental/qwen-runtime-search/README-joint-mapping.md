# External operator mappings and executable variants

`ProgramOperatorMapping` extends the existing external `ProgramGraph`; it does
not synthesize MLIR arithmetic. A candidate has executable source covers,
backend resource/shape/replica bindings, a ready priority, and actual ABI copies.
`ReplaceExecutableCover` replaces the same source-operator set with registered
implementations; it can represent executable fusion, fission, tiling, resource,
replica or dataflow variants. It cannot grant a transformation merely by naming
it. Backend registration and numerical validation are separate obligations.

The mapping checker binds implementation, artifact, entry point, device context,
dataflow, residence policy and implementation parameters. Timing reserves the
registered resources, including transitive shared resources. Prepared allocations
share an identity; different sizes or locations for the same identity are errors.
ABI storage and retained storage occupy memory together. Temporary reservations
compete with that live storage and other simultaneous reservations. A ready
priority never overrides an implementation or source dependency.

Current Qwen runtime calls are synchronous. Their actual control edges remain in
the graph. The exact serial-calendar fast path does not manufacture overlap.
Inclusive invocation wall time already includes launch, synchronization and
transport; those costs are not charged again. Different executable partitions or
residence policies need their own measured implementation timings.

The optional `execution_registration_identity` catalogue field activates the new
calendar evaluator in `taskflow-program-operator-search`. Every implementation
must then provide `execution_variant`:

```json
{
  "runtime_context_identity": "backend-registered-context-hash",
  "dataflow": "actual-compiled-dataflow",
  "residence_policy": "actual-runtime-policy",
  "replicas": 1,
  "work_distribution": "single-invocation",
  "resource_demands": {"npu.device": 1, "npu.command": 1},
  "resource_shape": [2, 4],
  "implementation_parameters": {"block_m": 128},
  "temporary_bytes": {"core-0-2-L1": 49152},
  "resident_allocations": {
    "prepared-weight-identity": {"memory": "runtime-abi", "bytes": 4096}
  }
}
```

Resource and memory names must exist in `catalogue.architecture`, using the
existing memory architecture schema. Replica counts above one require
`registered-disjoint-work`, an actual independently partitioned implementation,
and backend replay validation. Core partition counts are not task replicas.
Unknown used-memory capacity makes feasibility unproved. Unknown implementation
wall time remains unknown; it never becomes a free implementation.

The immutable mapping domain certifies source coverage and boundary ABI once.
For the actual serial invocation order it derives compact per-choice resource,
allocation and scratch summaries. Every assignment still recomputes the joint
physical allocation union, shared identities and memory bounds; this is not a
complete-plan price cache. This matters when two retention choices fit
individually but exceed capacity together. The selected candidate is fully
materialized into the public
regions/versions/copies/events graph and independently checked; the CLI also
requires its cost to equal the certified projection. No executable graph is
replaced by the compact scheduling view.

The public ORBIT beam retains unlimited wall/evaluation/round behavior and stops
at its declared neighborhood convergence. This is not a global-optimality proof.
The old catalogue path remains explicitly labeled `independent-local-cover-fees`.

## Current limits

Arbitrary transfer routes, copy elimination, asynchronous overlap, and independent
replica/shape/residence edits require actual registered runtime lowering; these
actions currently return an explicit unsupported diagnostic. Kernel-internal
copies and launch costs belong to the registered inclusive invocation, rather
than an invented decomposition. New XRT/CDO contexts must not be mixed across
decode steps unless runtime supports it: use one global compound variant choice
containing the 127 distinct step invocations. A compound choice does not fuse
those recurrent invocations.

## CPU contract evidence

`/tmp/orbit-r18-streaming-mapping-tests.log` records 52 external-source/mapping
checks; `/tmp/orbit-r18-streaming-beam-tests.log` records 35 beam checks, including
streamed evidence and unknown-state exploration. The final three native checks
are recorded under `qwen-whole-graph-r18/native-final-regression/summary.json`.
`/tmp/orbit-r18-joint-cli-fixture/` is a synthetic CPU-only CLI contract test, not
NPU performance evidence. It checks both unknown-capacity rejection and final
projection/full-physical-graph agreement. Actual Qwen numerical and performance
gates are produced separately by the backend experiments.

The CLI streams every complete evaluation row to `.beam-trace.jsonl`, retaining
exact assignment deduplication and normal beam convergence. This avoids a second
in-memory JSON archive at million-candidate scale. Stream failures abort the
search rather than producing a complete-result claim. The completed full-Qwen
experiment, actual choice coverage and heldout NPU comparisons are documented
in [the r18 report](../../docs/taskflow-qwen-joint-mapping-r18.md).

## Checked input packing

A fused implementation may replace a readonly source-weight representation with
several actual padded buffers. `implementation.storage.prepared_weight_copies`
registers `source_value` (source ID string), `source_offsets`, `source_extents`,
`allocation_id`, `memory`, `dtype`, `bytes`, `physical_shape`, `physical_offsets`,
`strides`, `replaces_source_physical_copy: true`, and the explicit
`host_device_same_allocation` boolean. The registered bitwise preparation and
numerical gate remain backend obligations.

Common validation requires exact source coverage by disjoint in-bounds
rectangles, matching dtype, in-bounds padded physical regions, row-major physical
strides and exact physical allocation bytes. The source must be a readonly root
input whose consumers all belong to the selected cover. More general state/view
replacement requires another checked transformation and is currently rejected.

Materialization uses the existing region/version/copy/data-edge tables: each
logical source rectangle has its own real physical allocation and consumption
edge. The obsolete full-source allocation is removed. No discontiguous aggregate
BO is fabricated. Prepared copies remain valid through the invocation scope;
shared resident identities must match the resulting allocation memory and bytes
exactly and are counted once. Private device copies require distinct identities
and additional reservations.

The packed graph checker independently rebuilds the unmodified source ABI,
applies the checked fragment transformation, recomputes resource timing, and
compares canonical graph identity. Injected old allocations, altered source
intervals, or changed physical strides cannot reuse the original proof.
`external_mapping.variants[].packed_inputs` preserves the full contract for the
backend replay bridge. Every declared `unproved_constraints` diagnostic, including
unknown private storage, makes candidate feasibility unproved.

Task replicas also require `execution_variant.work_partitions`: each item names
`output_value` (source value ID), `replica` (zero-based), and logical `offsets` /
`extents`. For every source output all replicas must have nonempty, disjoint,
in-bounds work, with exact complete coverage. A two-way N partition of `[P,N]`
uses offsets `[0,0]` / `[0,N/2]` and extents `[P,N/2]`. Padding is not source work.
The backend still proves that these partitions are what the executable actually
runs; merely setting a replica count cannot duplicate the original output.

## Exact summaries for the synchronous runtime

The invocation-by-invocation evaluator remains available as
`evaluateProgramOperatorMappingDomainReference`. For this runtime's complete
synchronous invocation order, local resource feasibility can be certified once:
there are no simultaneous calls whose reservations need a new overlap proof.
An immutable choice summary stores the sum of its actual local invocation
latencies, local resource legality, physical allocation identities and maximum
temporary reservation per memory. It is not a table of complete-plan prices.

Every assignment still constructs the joint physical-identity union, checks
size/location/source-root agreement, counts shared allocations once, and checks
all persistent ABI/resident bytes plus the largest actually simultaneous
scratch reservation. Distinct retention choices can fit separately and fail
together. Thus caching local proofs does not turn global storage feasibility
into an independent product. The selected full physical graph and its timing
must still agree with the summarized evaluation.

`program-operator-search SOURCE CATALOGUE RECEIPT --audit-checkpoints TRACE`
checks the first 100 previously captured assignments against both evaluators
and the captured costs, reporting timing and the maximum numeric difference.
This CPU audit is separate from an unbounded search and does not cap that search.
