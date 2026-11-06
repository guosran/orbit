# Public program graph and native AIR realization

`ProgramGraph` is the common task/data/communication object. The compatibility
name `ProgramCommunicationPlan` aliases the same type; it is not a second ledger.
Its source is the existing `GenericAnalysis` of real Taskflow MLIR, with borrowed
operation/value handles. JSON and DOT are checked views, not a replacement
program language or a user-authored communication graph.

The graph contains logical storage and source dependencies, task instances with
actual loop domains and implementation choices, region versions, location
copies and allocations, data requirements and order-only edges, transfer events,
execution/submission boundaries, and a checked conservative schedule. Copy events
carry their exact rectangular region and byte count. Unknown durations remain
unknown; a schedule or static byte count is not device timing or DDR traffic.

| Entry | Input and result | Responsibility |
|---|---|---|
| `GenericAnalysis.cpp`: `analyzeGenericProgram` | Real module → source tasks, accesses, storage and dependencies | Source facts and supported affine interface |
| `GenericMapping.cpp`: `planGenericMapping` | Source and target/choices → native realization with graph | Existing tile, implementation, grouping and retention choices |
| `GenericCommunication.cpp`: `buildProgramCommunication` | Native realization → the common ledger | Version/copy events and adapter correspondence |
| `ProgramGraph.cpp`: `finalizeProgramGraph`, `checkProgramGraphSource`, `checkProgramGraph` | Graph bound to actual source → checked graph/schedule | Source binding, coverage, versions, placement, capacity and dependencies |
| `AIRBackend.cpp`: `lowerGenericToAIR` | Checked graph and native realization → actual source body in AIR | Common allocation/transfer/instance consumption and explicit serial realization |
| `GenericMapping.cpp`: `collectGenericCostFeatures` | Checked graph and backend realization → native features | Common source/instances, transfers and submission counts; explicit backend stage/core facts |
| `Search.cpp`: `runStructuredSearch` | Canonical source candidates → replanned graph, evaluation, archive and checkpoint | Original native actions, ranking, unknown exploration, identity and resume |

## Graph-to-AIR object correspondence

AIR allocates local pools from `graph.allocations`: key, bytes, storage, memory and
lifetime are authoritative. A computation's copy IDs resolve its pool through
copy → allocation, rather than reconstructing a storage-name allocation policy.
The current tiled adapter requires an allocation's lifetime to equal its checked
hardware stage. Resident buffers keep their source shape and must have exactly
the common allocation's byte size.

DMA selection and geometry come from common transfer events, regions, links and
source/destination copy IDs. The adapter checks concrete host/local operands,
memory spaces, pool keys, byte sizes, cell coverage and lifetime. Native transfer
records remain an independently checked correspondence witness; they do not
choose a different transfer. Source bodies come from `graph.source`; task, domain,
implementation and lane choices come from `graph.instances`. A distributed source
initializer executes the proved initializer loop prefix of its controlling tile,
without repeating initialization along the reduction axis.

Native phases still describe source demand views and exact translation-equivalent
loop compression. Hardware stages, readonly scope descriptors and checked channel
assignments remain target realization constraints. This is not an arbitrary
public-graph placement or event scheduler. The explicit
`orbit.graph_schedule_realization` is `serial-by-phase`: it preserves the public
partial order through a stronger serial realization. Compressed operations carry
representative public IDs; the checked finite schedule establishes their expanded
coverage. `orbit.allocation`, `orbit.source_instance`, `orbit.computation_event`,
`orbit.transfer_event`, copy IDs and public dependency metadata make the concrete
correspondence inspectable. Both resident and tiled canonical AIR checkers
regenerate expected AIR; channel checking also requires the graph identity.

Current support remains the existing static rectangular host-BO/local-pool path:
scalar resident execution, scalar or strict FP32 mixed8 tiled execution, retained
accumulation and proved readonly panels within one execution group. Stage
boundaries use host BOs; there is no arbitrary cross-stage SRAM retention. Existing
stream-count, lane, tail/channel, alias, capacity and source-body restrictions
remain enforced. The 24,576-byte policy is logical local tensor payload, not a
single physical SRAM-bank capacity; stack and installed allocation verification
are separate. No new precision, arithmetic matcher, runtime or NPU execution is
introduced by exposing this graph.

## Search, cost and identities

Native evaluation returns the actual `program_graph`, `program_graph_id` and DOT;
the mapping embeds the same graph. Each evaluated search candidate saves
`program-graph.json` and `program-graph.dot` beside its source, request, feedback
and complete `mapping-result.json`. Resume independently reconstructs the graph
from current source/target and compares graph artifacts and identities.

The existing `candidate_id` remains the canonical materialized-MLIR SHA. The
separate `program_candidate_id` combines that source ID with `program_graph_id`.
The graph ID covers the checked graph view, including its schedule and supplied
compute estimates. The derived top-level `cost` summary is excluded. This is
not a universal price-independent machine-code equivalence key.
Target/model/runtime evaluation identity remains separate. Frozen v10 artifacts,
models and hardware observations keep their original identities; CPU graph
replanning does not re-label those measurements as v11 executions.

Traffic features sum common transfer-event byte counts once, and submission
features use common execution boundaries. Source/implementation facts use the
common source and instances. Phase/stage counts and the one-core realization
remain explicitly backend features. Whole-command composite labels already
include transfer/runtime costs: graph exposure does not add another DMA or
per-input submission charge. Unsupported or uncalibrated providers retain their
existing failure/null boundaries.

## Supported common planning and legacy boundaries

The finite whole-task planning adapter also builds/checks `ProgramGraph`, including
shared-copy users and common memory/resource/link contracts. Its CPU schedule is
not direct-AIR code generation for arbitrary targets.

The older `StaticCommunication.cpp` path still builds `StaticPlan` from its own
2D static access/cell representation. Original Amoeba `JointNeighborhoodSearchPass`
still obtains task-shape costs from the existing provider and invokes the actual
production scheduler through `ProductionSchedulerAdapter`. Explicit Amoeba
communication uses `TaskEdgeGraph`, `InterTaskNetworkSpec` and route reservations.
Those paths do not yet consume public location copies. Shared target types do not
constitute a completed migration. Their existing scheduling capability is retained;
no physical CGRA memory capacities, transfer costs or cross-task residency are
invented. A future bridge must prove payload, placement, version/copy and scheduler
correspondence before claiming common-copy execution support.

The common checker also proves exact execution coverage by projecting task
instances onto each actual memory access's ancestor loops, without multiplying
sibling loop nests. Its finite Cartesian partition limit is 1,048,576 cells.
Repeated source accesses outside a partitioned loop are conservatively rejected;
this does not implement arbitrary loop-invariant code motion. Distributed
initializers retain the native adapter's separate prefix/projection proof.
The common checker validates copy/execution semantics; the adapter additionally
checks that a requested policy, initializer, tile domain and stage realization
produce those facts. These are complementary checks of the same graph.

Allocation peaks use phase intervals for reserved native pools and the union
of copy-validity intervals for whole-task allocations. Shared copies do not
multiply an allocation's capacity charge. Alignment padding is reported only
when its unrounded payload is known. System BO capacity, the local tensor-pool
budget, and final compiler allocation/storage verification are separate checks.

`test/TaskflowAIR/program_graph_backend_v11.py` checks resident, scalar tiled and
mixed8 readonly-panel lowering, then rejects changed allocation, source-instance,
transfer-event and graph-ID metadata. The current v11 AIR binary passed all15
compiler-only checks in `backend-cpu-r2`. These are CPU lowering/checker results,
not numerical, physical-traffic, device-progress or new performance evidence.

## Native entry and acceptance

The main C++ entry is `tools/taskflow-opt/taskflow-opt.cpp`. For the prepared
GEMM-scale source, one invocation exports and lowers the very same checked graph:

```bash
AIR_TOOL=/path/to/air-build/tools/taskflow-opt/taskflow-opt
"$AIR_TOOL" NEW/prepared/gemm-scale/source.mlir \
  --air-distribute-initialization --tile-m=8 --tile-n=16 --tile-k=32 \
  --data-organization=retain-across-users --resident-storage=1 \
  --reuse-loop=0 --panel-loop=1 --lower-to-air \
  --mapping-output=NEW/plan.json --program-graph-output=NEW/graph.json \
  --program-graph-dot=NEW/graph.dot --air-output=NEW/program.air.mlir
```

The independent Core tool supports the same graph/check/search exports without
`--lower-to-air`. The finite whole-task adapter uses
`--program-graph --target-config=CONFIG`; the configuration supplies architecture,
placement and policy while real computation remains the input MLIR. An exported
view can be checked against fresh source planning with `--check-program-graph`.

The fresh-directory companion entry is
`../orbit-artifact/taskflow/scripts/run_taskflow_program_graph_v11.sh` with
`prepare`, `cpu`, `compile-only` and `all` modes. The actual frozen-tool `all`
run is `../orbit-artifact/results/taskflow/program-graph-v11/acceptance-r1`.
Its plain M32/K128/N64 GEMM-scale program reserves 65,536 B of logical system
BO data. Phase-local, retained B-panel, and a native two-round search-generated
8/16/16 + retention candidate use 3,584 / 9,728 / 9,216 B of local tensor pools;
planned transfers are 221,184 / 122,880 / 122,880 B. Their original graph JSON
equals the graph embedded in each compiled plan. All three are directly AIR
compiled and NPU executed: three seeds, two warmups and five measured calls per
seed give 63 independently checked outputs, maximum absolute error 7.15e-7,
unchanged finite FP32 `atol=rtol=1e-5`. Host medians are
18.562311 / 18.725276 / 18.440053 ms; their variation does not establish a gain.

The CPU fanout example has three real tasks, one pure ordering edge and shared
or separate copies of X. Shared SRAM needs one 64 B creation transfer versus
two transfers totaling 128 B. The same program on a shared-plus-two-local
topology uses 192 B with retained transit versus 256 B with task-local transit.
Those targets and 4.1 / 6.3 / 7.4 ms estimates are explicit synthetic parameters,
separate from NPU evidence. The artifact validation report contains original
JSON/DOT, topology configurations, candidate lineages, ABI/ELF identities,
runtime/storage correspondence, raw API samples and numerical audits.

The original-search regression also materializes `split_outer_loops`: actual
source loops change from one to two and access facts from three to six. Its
source and graph IDs change, and fresh source/copy checks pass. The whole-resident
realization still needs the same six copies and 192 B of transfers. This is loop
fission, not a new task-partition or device fission measurement; bound records
are in companion `results/taskflow/program-graph-v11/loop-fission-evidence-r1`.


## Subsequent Amoeba migration

The v11 legacy-Amoeba boundary above is historical. Stage v12 adds an opt-in
checked ProgramGraph bridge to the original CGRA ML provider, mapper and
production scheduler. It shares public source/data/copy/dependency semantics
with direct AIR, while keeping target-specific routing and cycle estimates.
See [the v12 migration and validation](taskflow-program-graph-amoeba-v12.md) for
actual supported source interfaces, legacy differentials and remaining paths.
