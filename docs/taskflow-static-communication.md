# Static communication derivation

> 历史 v3 说明：Python 规划器/检查器现已退出活动代码，旧差分仅为历史证据。默认生产与检查均为 C++/MLIR；复现入口为 `bash ../orbit-artifact/taskflow/scripts/run_taskflow_static_cpp.sh NEW_OUTPUT_DIRECTORY`。v5 审计保留了退出前代码快照和全部旧结果。


`python/orbit_mapping/communication.py` consumes the existing
`orbit.taskflow.plan.v1` operations and the static configuration owned by
`orbit_mapping.contract`. It does not introduce another computation language
or import a runtime/backend. `build_graph(program, config, steps, assignment)`
returns unscheduled compute, copy, conversion and visibility events, physical
allocations and final output requirements; the mapper schedules these events.

## Accesses, versions and copies

All access rectangles are derived from the actual operation's half-open M/N/K
ranges. Matrix multiplication reads its current output when `accumulate=true`;
the initialization and every sequential K update remain real compute events.
Task completion dependencies add ordering only. WAR/WAW constraints add
ordering without transferring data.

Rectangular boundaries partition each logical buffer into exact cells. A cell's
version is either its initial input or the semantic operation that last wrote
it. Physical allocation identity separately records logical buffer, memory,
layout and dtype. A copy is valid only for the current version of that region.
Consumers reuse valid copies, including copies made for another consumer;
disjoint accesses copy only their own bytes. Several cells with the same source
allocation are grouped into one transfer with one startup cost. Their byte
count is a logical transfer estimate, not hardware traffic measurement.

Direct use at the same location/layout emits no copy. A value produced by a
different device in shared memory requires an explicit synchronization with a
configured source, destination, resource and cost. Pure synchronization has
zero transfer bytes. Configured copies/conversions include completion
visibility in their cost boundary, so a second visibility charge is not added
after them. Repeated identical synchronization requirements are reused.

Inputs begin valid and visible to every device permitted to access their
declared initial memory. This is an explicit ready-input assumption for one
cold graph, not free preprocessing or preloading at another location. Outputs
are made valid in `output_locations`; `output_devices` specifies the final
consumer. It can be inferred only when that memory has one accessible device.
Shared-memory NPU outputs requested by a CPU therefore require visibility too.

## Costs, capacity and limitations

Task costs include local loads/stores. Explicit transfer and conversion costs
are startup milliseconds plus logical bytes divided by configured effective
bandwidth; visibility costs are separately configured. Every descriptor carries
provenance. Missing paths, conversions or required synchronization costs are
unsupported, not zero. Shared-resource exclusion and overlap are enforced by
the mapper's resource schedule, not inferred from device names.

Every physical data copy reserves the entire logical buffer for the whole
graph, while validity and communication operate on regions. This deliberately
conservative capacity model counts source and layout replicas simultaneously;
it does not reclaim dead buffers or reuse address space. Implementation scratch
and optional conversion/transfer `temporary_bytes` have event lifetimes. A
conversion with no extra scratch field is an out-of-place conversion using its
already counted destination buffer. No in-place aliasing or dtype conversion
is supported. Layout conversion currently occurs at the chosen source before
routing, using an explicitly declared converter.

The planner chooses a deterministic feasible source and a shortest-hop memory
path; it does not optimize communication routes globally. Exact partitioning
is bounded by `assumptions.max_region_cells` (default 100000), and fails closed
above that limit. No dynamic buffers, runtime cache state, XRT BOs or repeated
request assumptions are represented. Plans remain static predictions until a
backend explicitly implements and verifies the same plan.
