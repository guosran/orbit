# Static heterogeneous planning contract

`python/orbit_mapping` consumes the existing `orbit.taskflow.plan.v1`; it does
not replace Taskflow, change its numerical semantics, or import any backend or
runtime. The core-authenticated materialized program remains the authority.
`map_program(program, config)` returns `orbit.static.mapping-result.v1` with
`status`, `best_plan`, `alternatives`, `reasons`, and bounded-search accounting.
Only independently checked, fully scored plans can become `best_plan`.

## Configuration and identity

`orbit.static.config.v1` has these required components:

- `hardware.resources`: IDs, integral concurrent `capacity`, `kind`, `device`.
- `hardware.memories`: IDs, `capacity_bytes`, `accessible_devices`.
- `hardware.links`: directed source/target memories, shared resource ID,
  `latency_ms`, `bandwidth_bytes_per_ms`, cost provenance and optional byte
  validity bounds. First reachable shortest-hop route is used conservatively;
  this first version does not optimize alternative network routes.
- `hardware.conversions`: explicit same-memory layout conversions, source and
  target layout/dtype, resource, latency and bandwidth. Dtype conversion is
  deliberately rejected in this version.
- `hardware.synchronizations`: source/target devices, memory, resource, latency
  and provenance for direct shared-memory cross-device visibility. Missing
  required synchronization cost is unsupported, never silently zero.
- `implementations`: versioned IDs, supported operation kinds, device, memory,
  layout, f32 interface dtype, numerical contract, resource demands, temporary
  bytes and affine compute-only latency model. The model requires fixed,
  per-output-element and per-FLOP coefficients, m/n/k validity ranges and
  `synthetic`, `theoretical`, `calibrated` or `measured` provenance with source.
- `assumptions`: explicit input/output storage locations, allowed numerical
  contracts, `overlap` (`forbid` or `explicit`), single execution, mapping
  assignment budget and beam width. `program_source_id`, when supplied, binds
  numerical acceptance to that original source. `output_devices` declares the
  final consumer; it is required when output memory is shared by multiple devices.

An affine compute model is `fixed_ms + per_element_ms*M*N +
per_flop_ms*(2*M*N*K)` for matmul; other operations have zero FLOP term and
must supply their relevant element coefficient. This is not a claim that an
activation costs no arithmetic. All costs must be explicitly present, finite,
nonnegative and within validity bounds. Cycles and opaque historical full-graph
measurements are not accepted as decomposed compute costs.

Implementation legality (for example axis multiples) is separate from search
recommendations. Device-preferred action granularity belongs in the target
configuration consumed by the outer search, not in common mathematical
semantics. Approximate NPU arithmetic is selectable only when the explicit
allowed numerical contracts permit it. Existing NPU matmul primitives are
restricted with `full_k_only` and `overwrite_zero_accumulator`: the immediately
previous operation must be the exact zero fill of that output region. The
planner retains that fill and conservatively retains the zero-accumulator
read/transport. Partial K matmuls therefore select explicitly available CPU
implementations; they are not silently executed by another runtime.

Program identity hashes the complete public program including access regions.
Configuration identity hashes all hardware, implementation versions, costs,
precision/layout requirements and initial-state assumptions. A separate plan
identity hashes the actual events, allocations, assignment and timing. These
are distinct from the public semantic candidate ID. The evaluator additionally
binds implementation source hashes and evidence to its cache/checkpoint.

## Expansion, communication and storage

Each operation becomes a target compute step `task_id:operation_index`.
Operations inside a fused task remain ordered. A task completion dependency
joins to the last operation of its predecessor; it never creates a numerical
sum. Fill and every sequential K partial remain present, including accumulator
reads. A merged task does not imply an implementation-fused kernel or removal
of intermediate storage.

Communication partitions logical access rectangles at actual region boundaries
and tracks each cell's current writer/version and valid physical copies. A data
read at the same location/layout reuses a valid copy. Pure task ordering edges,
WAR and WAW ordering do not manufacture copy bytes. Copies preserve version;
writes invalidate stale replicas. Shared consumers reuse already-valid copies.
Cross-device shared-memory access has an explicit synchronization event when
needed, even when no bytes move. Conversion has its own event and allocation.

Physical allocation is intentionally conservative: each logical buffer has a
whole-buffer allocation for each used memory/layout, live for the entire static
graph. Thus initial weights, output buffers, intermediates, accumulators and
layout replicas count simultaneously. Per-implementation temporary allocations
live during their compute event. This may reject plans a sophisticated allocator
could fit, but never assumes free alias reuse. Dynamic shapes, aliasing,
multi-request state and runtime BO/cache mechanisms are outside this contract.

Compute costs include local memory access. Copy/conversion costs include their
completion visibility. Direct cross-device shared-memory handoff has explicit
synchronization. No staging cost is counted both in compute and transfer.
Predicted bytes are logical transferred byte estimates, not measured traffic.
Cold initial inputs are present only at explicitly declared initial locations;
other locations require paid transport. Only one graph execution is supported.

## Mapping and scheduling

For a small implementation Cartesian product, all assignments are evaluated.
Otherwise the bounded search seeds fastest-compute and homogeneous-device
assignments, then expands implementation-choice neighbors of the best complete
plans. If all seeds are infeasible, their neighbors are still explored without
assigning artificial scores. Feasibility barriers beyond the bounded frontier
can hide better mappings; the algorithm does not claim completeness when the
assignment budget truncates it. Every ranking score includes the rebuilt communication DAG, resource
schedule and storage feasibility; isolated task latency is never the final
selection criterion. The result saves every evaluated alternative and failure,
its assignment, and successful full plans. Budget truncation is explicit.

Scheduling uses deterministic topological list order with earliest available
resource-capacity intervals. Shared communication resources serialize when
capacity is one. `overlap=forbid` serializes every event; `explicit` permits
concurrency only under the separately declared resource capacities. This is a
bounded heuristic and does not claim global placement/schedule optimality.
The independent checker recomputes semantic coverage/accesses, dependencies,
versions, costs, resources, capacities and completion time before acceptance.

`orbit.static.plan.v1` records events (`compute`, `copy`, `convert`, `sync`),
reads/writes with buffer, region, allocation and version, explicit dependencies,
resource demands, start/end times, allocations and lifetimes. `totals` separates
compute/copy/convert/sync service time, complete makespan, transfer bytes and
peak memory. Service-time sums can exceed makespan under legal overlap.
Every plan says `prediction_kind=static_estimate` and
`execution_status=not_executed`. Existing runtimes have not executed these new
static plans; no historical hardware latency is attached as their measured cost.
