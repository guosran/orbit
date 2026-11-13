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

The r18 Qwen registration uses synchronous calls. Its actual control edges remain
in the graph, and the exact serial-calendar fast path does not manufacture
overlap. The r19 source-dependency-ready registration instead permits only the
CPU/NPU overlap supported by its source-bound runtime proof, as described below.
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

Arbitrary unregistered transfer routes, copy elimination, asynchronous overlap,
and independent replica/shape/residence edits require actual runtime lowering;
unsupported requests return an explicit diagnostic. Registered r19 ready
execution and atomic executable replacements support the cases described below.
Kernel-internal
copies and launch costs belong to the registered inclusive invocation, rather
than an invented decomposition. New XRT/CDO contexts must not be mixed across
decode steps unless runtime supports it: use one global compound variant choice
containing the 127 distinct step invocations. A compound choice does not fuse
those recurrent invocations.

## CPU contract evidence

Historical r18 evidence in `/tmp/orbit-r18-streaming-mapping-tests.log` records
52 external-source/mapping checks; `/tmp/orbit-r18-streaming-beam-tests.log`
records 35 beam checks, including
streamed evidence and unknown-state exploration. The final three native checks
are recorded under `qwen-whole-graph-r18/native-final-regression/summary.json`.
`/tmp/orbit-r18-joint-cli-fixture/` is a synthetic CPU-only CLI contract test, not
NPU performance evidence. It checks both unknown-capacity rejection and final
projection/full-physical-graph agreement. Actual Qwen numerical and performance
gates are produced separately by the backend experiments.

The r19 public-beam regression records 61 passing checks in
`qwen-ready-r19/common-final-api-review-v1/orbit-r19-final-api-beam-tests.json`.
The independent `check-taskflow-program-operator-ready` target additionally
checks the actual CLI's priority, manual, warm-start and unknown-cost paths.

The CLI streams every complete evaluation row to `.beam-trace.jsonl`, retaining
exact assignment deduplication and normal beam convergence. This avoids a second
in-memory JSON archive at million-candidate scale. Stream failures abort the
search rather than producing a complete-result claim. The current full-Qwen
experiment, actual choice coverage and heldout NPU comparisons are documented
in [the r19 report](../../docs/taskflow-qwen-joint-mapping-r19.md). The earlier
synchronous experiment remains in
[the historical r18 report](../../docs/taskflow-qwen-joint-mapping-r18.md).

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

## Registered ready execution

The default remains `synchronous-source-order`. That policy adds the real
synchronous invocation order; changing a priority cannot remove it. An actual
asynchronous backend opts in with catalogue `execution_policy` equal to
`source-dependency-ready`. Every selected `execution_variant` must register:

```json
{
  "execution_policy": "source-dependency-ready",
  "source_ordering_identity": "<exact whole source_id>",
  "async_runtime_proof_identity": "<actual backend replay/gate identity>"
}
```

All invocations agree on this protocol and proof identity. The backend proof
must bind its real executor, source effect/ordering contract and validated
implementations. A nonempty string is only a registration identity to common
C++; it is not independent evidence that a Python/XRT executor overlaps calls.
The replay bridge must verify the actual registered evidence and refuse a
synchronous executor. Unrepresented side effects or unproved executor support
remain `unproved_constraints`, rather than permission to erase a source edge.

`programOperatorSourceDependencies` exposes source data/control dependencies
and conservative backing-allocation read/write/anti-dependencies. Readonly views
do not write their base. Conflicting slices of one backing allocation remain
ordered unless a separate byte-range independence proof is introduced. Ready
materialization adds no adjacency-based invocation ordering. It preserves all
source control edges and checks the contracted implementation DAG for cycles.
This permits a registered noncontiguous DAG-convex cover; contracting a path
through an outside operator is rejected. The final physical graph is reordered
topologically, with variants and priorities remapped together. Copy lifetime
indices are therefore topological phases, not guessed asynchronous wall times;
managed ABI and retained allocations remain conservatively live for the full
execution.

`external_mapping.ready_priority` is a permutation of normalized event indices.
Each index refers to the corresponding selection region and registered variant.
It chooses among dependency-ready calls only. A runtime must obey the graph's
real dependencies and capacities; replaying the permutation as an unconditional
linear instruction list is incorrect.

The catalogue explicitly lists finite `ready_priority_policies` seeds from
`source-order`, `reverse-source-order`, `cpu-first`, `npu-first`, and
`longest-first`. These seeds do not exhaust temporal search. After the existing
ORBIT beam, the CLI alternates general adjacent priority transpositions with all
single-group implementation replacements, accepting strict improvements until
both neighborhoods have no improvement. Comparable adjacent events are skipped
because their relative priority cannot affect a dependency-ready schedule. A
learned source priority is inherited by alternative covers; source semantics
stay unchanged. The joint action log records parent states, checked swaps and
accepted states. The report separates the initial beam from this subsequent
joint search, records numeric improvement tolerance and actual work, and makes
no global-optimality claim. There is no wall/evaluation/round cutoff.

Ready evaluation caches immutable source dependency frontiers, local resource
closure and executable ABI proofs. Each assignment recomputes joint physical
allocation identities, resident storage and capacity, then evaluates an exact
nonpreemptive resource/memory calendar. A scratch noncontention certificate may
skip the memory calendar: assign every invocation to a resource it completely
occupies and prove persistent storage plus the sum of per-lane scratch maxima
fits each memory. Failure falls back to the full memory calendar. A serial suffix
is summed only when its real per-invocation order and a dependency barrier after
all prefix events are proved. This evaluator reduction neither fuses kernels
nor removes recurrent decode invocations. The full selected physical graph is
still materialized and independently checked against the projected cost.

Coordinate actions for replica, resource shape and residence use
`applyProgramOperatorRegisteredMappingAction`: the request names an immutable
backend registration and an actual changed coordinate. Caller field edits are
ignored; source coverage is preserved and the complete coupled artifact/ABI/
resource contract replaces the old one atomically. Transfer-route edits remain
unsupported without an actually executable copy ABI.

CPU checks include an independent discrete-time calendar oracle, dense/full
calendar differential tests, source-control and shared-write negatives,
nonconvex-cover rejection, and a CLI fixture where a general priority swap
improves a 13 ms critical path to 11 ms from a single template seed. These are
structural contract checks, not device measurements. The ready-overlay audit of
old r18 assignments deliberately does not compare old synchronous measured
prices against an unvalidated asynchronous cost model.

Run the CPU-only ready CLI fixture independently of the public API aggregate:

```sh
cmake --build /path/to/build --target check-taskflow-program-operator-ready
```

The target builds the experimental CLI and checks priority search, checked
manual baselines, warm-start identity handling, and unknown-cost reporting.

## Candidate-dependent host controller model

A calibrated catalogue may register `invocation_control_model`:

```json
{
  "schema": "orbit.program-invocation-control-model.v1",
  "identity": "<source/fit/producer-bound model identity>",
  "source_id": "<exact whole source_id>",
  "intercept_ms": 0.0,
  "coefficients": {"cpu_callbacks": 0.0, "npu_callbacks": 0.0},
  "boundary": "outside-inclusive-invocation-wall",
  "fit": {"scope": "calibrated proxy; held-out errors required"}
}
```

Every implementation supplies `controller_features` with explicit nonnegative
integer `cpu_callbacks` and `npu_callbacks`, including zero. Missing features,
null/nonfinite coefficients, negative values, unknown features, wrong source or
cost boundary, and count/prediction overflow are rejected. Absence of the whole
optional model preserves the previous contract; absence of a field inside an
active model never means free execution.

The objective is callback resource-calendar makespan plus the model intercept
and the selected implementations' weighted callback counts, plus any genuinely
separate fixed application invocation. The physical `ProgramGraph` callback
schedule remains unchanged. Its `external_mapping.control_model` binds the
numeric model; final reporting exposes callback time, predicted controller time,
feature totals and remaining fixed time separately, and labels the controller
prediction approximate. A source operator count is not automatically a callback
count: fused, split and compound implementations register actual executor
features. These can act as dispatch-count proxies; they are not isolated
measurements of per-launch cost.

Calibration must use time outside the registered implementations' **inclusive
spans**. For a compound implementation, the complement of the union of its
primitive child callback intervals also contains internal gaps already included
in the compound local price. Charging those gaps again as outside controller
time double counts them. The fitter owns this interval audit, records the
subtraction, and evaluates held-out residuals. The common evaluator keeps this
boundary explicit rather than creating fictional controller device events.

## r19 experiment entry points

The common search lives in the C++ library and `program-operator-search.cpp`.
Python prepares executable candidates, records actual calls and replays checked
graphs; it does not select the winning mapping. The complete experiment and its
validation boundaries are described in
[the r19 report](../../docs/taskflow-qwen-joint-mapping-r19.md).

| Stage | Entry points |
| --- | --- |
| Source-ready execution | `run-qwen-r19-ready.py`, `qwen-r19-source-ready.py`, `qwen-r19-ready.py` |
| CPU lowering and candidate registration | `qwen-r19-regions.py`, `register-r19-regions.py` |
| Private device ownership and decode overlap | `qwen-r19-xrt.py`, `qwen-r19-decode-ready.py` |
| Initial calibration | `prepare-r19-campaign.py`, `run-independent-r19-campaign-v2.py` |
| Admission and current cost fitting | `audit-qwen-r19-ready.py`, `fit-r19-catalogue-v2.py` |
| Source-cover domain and frontier | `build-r19-cover-domain.py`, `expand-r19-cover-neighbors.py` |
| Frontier experiments and receipt inventory | `prepare-r19-cover-round.py`, `summarize-r19-cover-round.py` |
| Strict fit inspection and stable CPU identities | `receipt-r19-frontier-fit-v2.py`, `stabilize-r19-cpu-identities-v2.py` |
| Observed, uncapped search | `launch-observed-search.py` |
| Explicit baseline assignment | `prepare-r19-baseline-choices.py`, C++ `--materialize-choices` |
| Checked graph replay | `replay-r19-selection.py`, `run-r19-selected.py`, `execute-r19-prepared.py` |
| Uniform file verification | `qwen-r19-verification.py` |
| Actual local-choice coverage | `audit-r19-choice-coverage.py` |

The versioned preparation, continuation, splitting and diagnostic helpers are
immutable producers in the experiment's evidence chain. Old versions remain
because actual manifests seal their exact paths and bytes. Their continued
presence does not make an old diagnostic a priced implementation. Use the
admitted campaign and its explicit preparation/verifier protocol; do not choose
a historical helper by its version suffix or silently overwrite its outputs.
The original fitter remains sealed as `fit-r19-catalogue.py`; the initial
catalogue's two explicit schema upgrades retain its original fit receipt and
prices. New frontier fits use the corrected v2 producer.

The registered C++ search first evaluates a complete joint coordinate pass to
obtain an initial proposal, then runs the uncapped beam and general cover/ready
neighborhoods to convergence. This pass neither sums independent local minima
nor stops the search. The optional `--warm-start` takes a previous actual checked
ORBIT result, matches stable implementations into the new domain and re-evaluates
their costs under every named priority. It preserves every newly registered
choice. Initialization, beam and final neighborhood costs remain separately
visible in the actual checkpoints and report.

Formal calibration and heldout device requests must run without concurrent CPU
builds, fitting or search. The device runner owns the NPU lock itself. An outer
lock on the same file would deadlock the runner. Compilation, model loading,
readonly page preparation and file verification remain outside request timing;
their separate wall costs are retained in execution receipts.
