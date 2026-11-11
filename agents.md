# ORBIT: Joint Optimization of Whole Programs

## Project direction

ORBIT builds on AMOEBA/Neura and optimizes **an entire program's task graph and execution plan**. Its core is a joint search over graph structure, resource allocation, temporal scheduling, and data organization. Starting from the same program, it materializes alternative task partitions, parallelism, resource shapes, and execution arrangements, evaluates whole-program cost, and expands promising candidates.

The common Taskflow/ProgramGraph path now carries executable source-bound plans through the original C++ search. Amoeba and AMD XDNA NPU are backends that consume search results and provide target constraints, implementation capabilities, and cost feedback. Communication modeling and heterogeneous execution extend the original search space: ORBIT jointly decides how computation is partitioned, where it executes, where data resides, and when data moves.

Three questions guide the project: which whole-program optimization opportunities exist, how search discovers them, and how they affect execution cost. NPU tiling, residency, and communication experiments validate the common abstraction, search decisions, and executable lowering.

Workloads such as GEMM, GEMV, attention, FFN, GCN, and Llama/Qwen provide different computation, dependence, and communication patterns for evaluating this framework.

## Search dimensions

The original ORBIT dimensions include shape, temporal scheduling, replica, tiling, fusion, and fission. Their optimization intent carries into common Taskflow transformations, with changes materialized in actual tasks, loops, access regions, and dependencies.

| Dimension | Decisions | Whole-program effects |
|---|---|---|
| Shape | Resource shape and size per task, including CGRA row/column configurations | Task throughput, resource occupancy, concurrent task capacity |
| Temporal scheduling | Ready-task order, priorities, and timing on resources | Critical path, waiting, utilization |
| Replica | Number of task or execution-resource instances and work distribution | Parallelism, load balance, input reuse, communication pressure |
| Tiling | Iteration dimensions and tile sizes, including M/N | Task granularity, working set, parallelism, reuse |
| K-tiling | Reduction partitioning and partial-result accumulation | Accumulation dependencies, working set, partial-result storage and traffic |
| Fusion | Tasks to combine and fusion boundaries | Intermediate reuse, execution overhead, parallelism, resource demand |
| Fission | Task or computation-stage split points | Mapping feasibility, resource fit, scheduling freedom, intermediate traffic |

These dimensions interact. Smaller tiles reduce local storage demand but can increase transfers and execution count. Fusion improves locality but can reduce parallelism. Replication changes both compute and bandwidth demand. The evaluation unit is the transformed program and its execution plan.

Implementation coverage varies by path. The original Neura path contains established shape, structural, and scheduling capabilities. The common structured path supports M/N/K tiling and restricted fusion/fission. Native Taskflow/AIR supports checked loop tiling, readonly-panel reuse, whole-program residency and restricted pointwise shape-/tile-major grouping. Action generators and target capabilities determine the available choices.

## Search workflow

The original `search-joint-neighborhood` and common search driver support iterative exploration:

```text
Taskflow input + target resources, implementations, and cost configuration
    ↓
Baseline candidate
    ↓
Parent selection → neighborhood actions → materialized Taskflow
    ↓
Semantic and target legality checks → deduplication
    ↓
Mapping, communication planning, scheduling → whole-program evaluation
    ↓
Archive and beam update → next round
    ↓
Best candidates, transformation histories, execution plans, and evaluations
```

Search state records candidate identity, parent, action history, materialized IR, evaluation, and selection rationale. Checkpoints support resuming experiments; caching and deduplication reduce repeated compilation and evaluation.

There are two categories of parameters:

- **Program and mapping decisions:** resource shapes, replica counts, M/N/K tile sizes, fusion/fission boundaries, task placement, storage, and communication strategies.
- **Search controls:** round and candidate budgets, beam width, proposal rules, alignment preferences, and diversity policies supported by each path.

Target preferences guide proposals, legality analysis establishes feasibility, and cost models or measurements rank candidates. Common interfaces are in `include/Taskflow/Search.h`, with actions and the driver in `lib/Taskflow/Search.cpp`. The original joint search is in `lib/Backend/Neura/Orchestration/JointScheduling/JointNeighborhoodSearchPass.cpp`.

## Communication and heterogeneous execution

Communication modeling connects structural choices to whole-program cost. Analysis derives access regions, sharing, data versions, and dependencies from actual computation. Search chooses partitioning, execution placement, and residency. Planning derives transfers and synchronization, then uses lifetimes to determine storage occupancy.

Semantic facts and optimization choices are separate. Multiple tasks reading the same region is an analysis fact. Shared residency, replication, broadcast, and repeated loading are alternative implementations. The compiler generates an explicit communication plan for evaluation and lowering.

| Extension | Decisions | Constraints and costs |
|---|---|---|
| Task placement | CPU, NPU, or other execution target | Implementation support, compute time, inter-device traffic |
| Implementation and region coverage | Individual, fused, or whole-region implementations | Numerical contracts, interfaces, resources, implementation cost |
| Storage placement and residency | Storage level, retention scope, release point | Capacity, reuse, repeated loading |
| Transfer partitioning | Regions, granularity, relation to compute tiles | Transfer count, bytes, boundary handling |
| Buffer organization | Reuse, single or double buffering | Lifetimes, capacity, access conflicts |
| Compute/communication scheduling | Serialization, prefetch, overlap | Dependencies, compute/DMA resources, bandwidth |

The supported subset of these choices is materialized by the existing C++ mapping/actions/search path and lowered through existing backend execution facilities. Unsupported placement, grouping or buffering remains explicit rather than being inferred by a Python runner.

Data reuse is a whole-program concern. Weights, input tiles, intermediate activations, and reduction partials share the same concepts: data regions and versions, storage copies, consumers, and validity intervals. Structural transformations and scheduling create reuse opportunities; storage planning selects retention; communication modeling evaluates the consequences. Retaining a weight tile across several M tiles may save weight traffic while increasing partial-output storage or transfers.

For NPU execution, the common storage plan determines on-chip tile residency. Reusing XRT BOs or execution objects across invocations belongs to execution preparation. The backend realizes storage placements in system memory, NPU-accessible buffers, and AIE local memory.

## Compute nodes, dependency edges, and data copies

The implemented common ProgramGraph extends source computation with candidate-owned regions, versions, allocations, location copies and execution activities:

| Object | Meaning |
|---|---|
| Compute node | Actual body, task instance domain, implementation choice, resource count and shape |
| Dependency edge | Data or ordering dependence; data edges identify regions, versions, and consumption requirements |
| Logical data | Source storage identity, region, and version |
| Data copy | A realization of logical data in a storage location, with layout, validity interval, and consumers |
| Architecture resource | Compute, storage, and communication resources, hierarchy, connectivity, and sharing constraints |

Multiple data edges can reference the same copy. If A produces X and B/C consume it, a candidate can share one SRAM copy or create separate local copies. Allocation and copy-creation traffic are accounted for once per shared copy, with a lifetime covering all consumers. Producer and consumer completion events define validity constraints; scheduling realizes them on a timeline.

SRAM and DRAM describe storage technologies or locations; a buffer is an allocated object in a location. Logical dependencies, physical copies, and transfers have separate identities. An existing valid copy can satisfy a read directly; creating a copy elsewhere requires a suitable transfer. Ordering-only edges establish execution order.

Resource domains describe system/device/cluster/tile ownership and sharing scopes. A separate connectivity graph describes reachable storage locations and communication paths. DMA engines, links, and buses have capacity and contention constraints. Backends bind these common resources to hardware.

A search candidate consists of:

```text
program graph + resource configuration + data organization choices + schedule
```

Candidate and cache identities bind these choices, target configuration, and model versions. Different residency decisions for the same computation remain distinct candidates. After structural transformations, analysis refreshes access and version relationships before rebuilding copies and transfers.

Public neighborhood actions now choose repeated loading versus readonly reuse, whole-program versus phase-local materialization, and restricted producer/consumer block retention. They compose with tiling through the original search driver. General placement, arbitrary replication, multi-instance buffering and broader streaming remain future work; do not describe synthetic scheduling examples as executable native support. See the v18 section and docs/taskflow-execution-graph-v18.md for current guards and evidence.

## Common representation and backends

```text
Taskflow MLIR: computation bodies, iteration domains, accesses, dependencies
    ↓
Common structural search, materialization, and semantic checking
    ↓
Static mapping, communication planning, resource scheduling, and evaluation
    ├── Amoeba → existing mapper / scheduler
    └── NPU → MLIR-AIR → AIE → XRT
```

Taskflow represents the entire program's computation and dependencies. A complete candidate adds resource, scheduling, storage, and communication decisions. Search can adjust these together. Backends supply resource contracts, implementation capabilities, costs, and lowering. Common structural transformations operate on tasks and loops; backends interpret target-specific choices. The independent Core and backend components build separately.

The original Taskflow dialect served Neura/Amoeba. Later additions, `GenericAnalysis` and `GenericMappingPlan`, provide analysis and execution planning above the computation IR. The common layer represents task instances, resources, data copies, and lifetimes. The AIR backend implements hierarchy, DMA, FIFOs, and tokens.

The NPU path lowers Taskflow directly to AIR. AIR launch, segment, and herd operations realize grouping and resource decisions from the plan. Historical Triton recipe and chain experiments provide implementation and cost comparisons.

## Frontend and implementation organization

Actual MLIR bodies define computation semantics. Generic analysis reads function interfaces, allocations, loops, accesses, and dependencies to establish storage identities, regions, and producer/consumer relationships. Interfaces and necessary attributes identify inputs, temporaries, and outputs. Reporting names connect configuration and evidence.

Structured operation semantics can support specialized optimization and cost estimation. Generic tasks lower through their loops, arithmetic, and accesses. Existing JSON import and witness verification serve historical compatibility paths.

Production analysis, transformations, search, planning, checking, evaluation, and lowering use C++/MLIR. Python supports experiment orchestration, toolchain invocation, hardware execution, measurement, and numerical references.

## Evaluation and cost models

The primary objective is whole-program execution time. Evaluation combines computation, communication, synchronization, resource contention, and execution overhead. Capacity, dependencies, target support, and numerical contracts determine feasibility.

Cost models are associated with implementations, shape ranges, precision, initial data state, and timing boundaries. The Amoeba path uses mapper/scheduler cycles and existing predictive models. The NPU path combines theoretical models, calibrated models, and execution feedback. Calibration uses training and held-out candidates to assess prediction error, ordering, and best-candidate selection.

The existing CGRA compute predictor uses ML and considers CGRA count and shape. It remains the backend compute model. Its interface records task features, resource configuration, applicable data-supply conditions, and the timing boundary of its labels. Communication estimates come from copies, paths, and target parameters. The scheduler combines dependencies and resource occupancy into whole-program time. Explicit accounting boundaries identify which memory and communication costs are already included in compute predictions.

As supported, model interfaces provide execution latency, resource demand, workspace, and initiation interval or throughput. A change in data placement triggers a check of model applicability, followed by reuse of the prediction, a calibrated correction, or additional training features as appropriate. Analytical and ML models serve the same evaluation interface.

Experiments retain candidate identities, action histories, generated IR, predictions, measurements, and raw samples. Core, Amoeba, search protocol, static checking, and NPU numerical regressions validate changes.

## Recorded progress and near-term work

The historical v6-v17 checkpoints below retain their original acceptance scope,
failures, compiler identities and measurements. The former v6 M128 timeout was
subsequently diagnosed and full-program validation completed; it is not the
current work queue. Large M128 FFN is not a default v18 experiment.

Current implementation includes executable data versions/copies, paths,
lifetimes, capacity checks, shared-consumer retention actions and explicit cost
coverage. Native pointwise block retention supports a working set that fits
locally even when the full graph exceeds the payload budget. ProgramGraph
execution groups and invocation lifecycle are authoritative for evaluation and
lowering. The explicit-assumption event model can drive the original two-round
search; strict mode keeps required unknown mechanisms explicit.

Current validation priorities are actual backend execution-topology agreement,
complete graph/source/plan identity, observation boundaries, unseen-plan error
and the resolution of candidate differences. Score-driven search, valid measured
ordering and demonstrated speedup are separate conclusions. General placement,
validated physical overlap/bandwidth sharing and reliable arbitrary-program
ranking are not implied by the implemented closed-loop mechanism. New results
and current reproduction entries are summarized in the v18 section.

## Related-work perspective

Comparisons should identify program inputs, optimization objects, adjustable decisions, and evaluation objectives. ORBIT focuses on joint optimization of whole-program graph structure and execution plans. Its overlap with dataflow programming and mapping systems such as Dato includes task merging, resource mapping, and communication organization. Useful ideas include separating logical instances from physical resources and making communication contracts and resource constraints explicit.

User-specified versus compiler-inferred information is one comparison axis. The central research question is which whole-program opportunities joint search discovers, and how it improves on fixed-structure or staged decisions. Related systems' own automatic mapping and search belong in that comparison.

The task-graph representation supports the implementation; the research opportunity lies in the coupling between structural changes and data realization. Strong examples show that a structural change alters reuse opportunities, that storage/communication choices change the preferred structure, and that joint search finds better executable plans than optimizing each independently.

## Entry points

- [Repository overview](README.md): original ORBIT background, search, and build.
- [Current public execution graph and search](docs/taskflow-execution-graph-v18.md): authoritative groups/copies, explicit-assumption scoring and evidence boundaries.
- [Clean compiler replay](scripts/run-execution-graph-v18-cpu.sh): fresh outputs, explicit tool paths, Core/AIR/original Amoeba/CGRA/lit regression entry; no device submissions.
- [Backend integration](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-backends.md): shared candidates and execution paths.
- [Search integration](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-search-npu-v2.md): common actions, original driver, and NPU evaluation.
- [C++ static planning](docs/taskflow-static-cpp.md): mapping, communication, and search integration.
- [Region implementation contracts](docs/taskflow-static-regions-v4.md): coverage and implementation costs.
- [Direct AIR lowering](docs/taskflow-direct-air-lowering.md): native interfaces and code generation.
- [Direct AIR hardware validation](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-direct-air-hardware.md): execution and measurement.
- [Direct AIR reproduction](https://github.com/guosran/orbit-artifact/blob/main/taskflow/scripts/run_taskflow_direct_air.sh): CPU and hardware experiments.
- [Tiled AIR validation](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-direct-air-v6-validation.md): oversized workloads, full FFN, and regressions.
- [Tiled AIR reproduction](https://github.com/guosran/orbit-artifact/blob/main/taskflow/scripts/run_taskflow_tiled_air.sh): static tiling and hardware experiments.
- `build/direct-air-v6/RESULTS.md`: local baseline report and evidence.


## Original v7 frozen acceptance checkpoint — 2026-11-04

The original hardware queue is complete: M128 full D2048/I11008 FP32 FFN passed three independent full outputs through16 compiler-issued complete-M8 commands per seed (about581.6 seconds each). Four joint-search candidates passed60 complete outputs; partition-tail, M5, all-dimension GEMM-tail and oversized generic add→scale regressions each passed15 complete outputs. exp/SiLU/SwiGLU finite probes passed15 each, with overflow/underflow edges separately classified under the unchanged finite acceptance rule. Raw failures and all timing boundaries remain preserved; no clear residency latency speedup is inferred from the four-candidate samples.

Execution identities remain per artifact: M128 native package3e041…, later frozen native executable2444…, and literal search/partition packages bind their own plan/AIR/ABI/ELF hashes. The frozen snapshot under `aie-toy/results/orbit-direct-air-v7/frozen-original-r1` passed328-file integrity checking. `build/direct-air-v7/RESULTS.md` and the result root's `validation-summary.json` record the completed evidence. New development is isolated in `/home/shiran/project/orbit-data-v8`; this original tree is frozen except for reports. The existing CGRA model is preserved without retraining; its backend compute/label/data-supply boundary audit is in the new tree. NPU scheduling is idle pending that tree's CPU checks. The architecture and research positioning above remain unchanged.


## 阶段进展：公共数据组织与联合搜索 v8

补充任务已完成实现和验收：实际 Taskflow 分析驱动公共数据/排序边与共享副本，复用 Static memory/resource/link；phase-local 与 retain-across-users 接回原搜索，与 tiling 组合得到两轮容量耦合候选。公共副本39项、native mapping36项、Core/Amoeba/旧搜索与直接AIR回归、真实CGRA模型8个合约均通过。CGRA ML保留原II模型和生产scheduler duration，公共费用合约明确数据供给条件及通信归属，未重新训练。

新版本两个驻留候选在NPU上3seed共30完整输出通过；计划DMA由221184B降至122880B，热中位数18.658558/18.650719ms，差异不足以宣称加速。新M5完整FFN15输出通过；原M128完整FFN3seed长验收已全部通过，约581.6秒/完整program，原包/工具/ELF分别绑定。最后checker-only保护经完整CPU复跑，5候选与M5生成的语义产物逐字节等于已上机包，未将等价检查冒称额外NPU执行。

设计和范围见 `docs/taskflow-data-organization-v8.md`；验收、计时/模型边界和复现见 `docs/taskflow-data-organization-v8-validation.md`。入口 `../orbit-artifact/taskflow/scripts/run_taskflow_data_organization_v8.sh cpu|all NEW_DIRECTORY`。隔离构建和全部原始证据位于 `/home/shiran/project/orbit-data-v8/build/data-organization-v8`，旧v6/v7证据保持原状。有限whole-task共享方案先在CPU合成目标验证，AIR保留仍限单execution group内已证明的只读panel；任意跨stage SRAM共享、自动异构放置、Exact/profile公共provider仍是后续扩展。

用户已授权验收完成后提交并push；提交范围为相关源码、测试和文档，生成产物继续本地保留。


## Repository separation and queued FFN profiling

Core implementation, source semantics, mapping/search/checkers, backend lowering and compiler-only regression fixtures remain in this repository. Experiment orchestration, numerical input/reference generation, hardware execution/audits, model regression and validation reports now live in the sibling `guosran/orbit-artifact` repository under `taskflow/`. New build/output defaults are `orbit-artifact/builds/taskflow/{core,air,amoeba}` and `orbit-artifact/results/taskflow`; shared helpers accept ORBIT_SRC, ORBIT_BUILD_ROOT and ORBIT_RESULTS_ROOT. Core-only tests use standard-library test/Support and do not import artifact runtime code. See `docs/repository-layout.md` and `scripts/build-core.sh`.

Historic raw results, frozen tools and modified v8 source snapshots were physically moved under the companion's ignored local storage. Compatibility aliases preserve old absolute references; original manifests and measurements retain their bytes. Reproduce into fresh directories rather than rebuilding historical CMake caches. New evidence and code-generation tests verify the separation.

Queued next: evidence-first full FP32 FFN profiling. Decompose the original M128/16×M8 timeline, explain the N16 residency comparison, inspect current executed ELF/helper paths, perform controlled stage/full-graph measurements, then choose one or two supported optimizations and validate a frozen direct-AIR model on held-out candidates. Keep CGRA ML unchanged and preserve artifact/compiler versions. The main agent coordinates NPU use; current queued work starts with read-only evidence analysis while repository separation is validated.

## Latest completed stage: whole-program profiling v9

The profiling queue and evidence-driven implementation experiment are complete.
Original M128 is16 serial full-M8 commands (about581.6s total); recorded wait is
about99.986% of its program wall. Independent full-size stage mains and executed
ELF inspection identify three expensive GEMMs on one core. Wait still combines
runtime, device compute, DMA and stalls. N16 residency reduces planned transfers
44.4% without demonstrated latency improvement; two/five device stages both use
one host submission, and static bytes are not physical DDR counters.

Native candidates can select scalar-product/vector-add FP32 lanes8, combined
with existing tiling and read-only retention. Full M1/M5/M8 pass three seeds with
observed7.37%/7.89%/6.83% hot-wall reductions; a separate complete M128 single-seed
profiling run passes in541.8698s. Original three-seed M128 acceptance is unchanged.
Numerical tolerance, K order, products, payload budget and generated DMA are
preserved. Failed lane4/r2/vector and four-stream experiments remain explicit;
target guards reject unsupported cases rather than changing execution silently.

Core provides opt-in native features, checked finite-domain composite prediction
and exact execution-fingerprint measurement feedback. CGRA ML is unchanged.
Frozen scalar holdout gives0.474% M3-full error but29–44% tail underprediction;
configuration guards and unknown/null costs prevent extrapolating to arbitrary
tiles or mixed implementations. Two native rounds compare structure-enabled
residency plus implementation, selecting a measured mixed8/B-panel candidate.
Four cells (144 checked outputs total), parents, model/target identities and checkpoint
provenance are retained; only observed executions receive measured scores.

Core/Amoeba/search/AIR regressions and real CGRA boundary checks pass. See the
[v9 report](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-ffn-profiling-v9.md)
and sibling artifact reproduction entries. Public optimization remains centered
on entire program candidates and their interacting structure/resource/schedule/
data decisions; backend-specific code experiments validate those choices and
their cost feedback. Full FFN device trace, physical DDR counts, arbitrary
cross-stage sharing and calibrated multicore/mixed predictors remain open.

## Latest completed stage: target memory architecture and predictive search v10

ORBIT remains a whole-program joint search framework. A candidate combines its
Taskflow computation/dependencies with structure, implementation, resources,
schedule and data organization. Amoeba/NPU consume these decisions and provide
capabilities, constraints, cost feedback and code generation. Common memory
architecture now reuses static resource/memory/link types, with ownership
domains, precise accessors and directed paths. Copy lifetime/capacity checks
work without calibrated link costs. Synthetic shared/local SRAM tests distinguish
phase-local transit imports from retained read-only copies. No physical Amoeba
capacities or NPU memtile budget are invented.

The direct AIR binding names host BO/local tensor pools, compute/DMA/submission
resources and links. Its 24,576 B tensor-payload policy and separate 4,096 B stack
are unchanged; the pool does not mean a single physical SRAM bank. Public
execution regions share one actual submission across their input dependencies.
AIR realizes these choices, including explicitly checked whole-module async
legalization for the installed tail DMA-to-channel conversion.

Original C++ search generates finite tile options from analyzed dimensions,
alignment and capacity, implementation choices from target capabilities, and
retention from access reuse. It rebuilds plans after transformations and retains
identities, actions, legal/failed requests, null-cost exploration and checkpoints.
Scalar spelling is canonical. The implementation-aware four-term prepared-call
NPU provider leaves CGRA ML and earlier providers intact. Composite labels
already include communication/runtime; no separate DMA or per-input launch is
added. Calibration is limited to its source family, target and feature envelope.

Before timing, 12 train / 6 holdout / 1 capacity-negative cells were registered.
Train design rank is four, scaled condition10.0643. Three-seed calibration and
post-freeze holdouts pass 252 and126 output checks. Frozen native predictions
have1.6304% MAPE /5.4624% maximum error, versus2.6846%/8.6393% for the three-term
ablation and96.4006%/96.6354% for the old uncalibrated theoretical coefficients.
Tail15-stage behavior is a narrow tested extrapolation, not broad model coverage.

The frozen model drives original search (224 evaluations,30 calibrated ranks).
An automatically generated new4/32/64 mixed8 candidate is selected by native
rank outside training and the four old measured executions, then compiled and
run with two restricted-search winners and one unpriced exploration. All84
outputs pass. Equal search ceilings are512 evaluations/six rounds/beam16; actual
control evaluations are56 and6. Joint/implementation-control predictions tie,
and paired timing does not establish a joint-search gain. Unknown exploration
keeps a null prediction after measurement.

A full M5/D2048/I11008 FFN comparison passes12 outputs (max abs2.42144e-8).
Lifting the new tile choice reduces local payload but doubles planned DMA and
stages:21.418708s versus21.214368s baseline, no observed speedup. FFN remains
outside this small arithmetic model. Historical M128 results retain v7/v9
versions; no new M128 timing is claimed. All474 new device outputs preserve the
FP32/K-order/1e-5 contract. No DDR counter or new full-device trace is claimed.

Clean-shell CPU reproduction,35 Core/Amoeba/search/AIR groups, existing CGRA
contracts and new protocol/selection/parser tests pass. Final rebuilt AIR binary
equals the executed frozen binary; all24 packages regenerate byte-identical
Taskflow/facts/plans/AIR. Independent Core remains free of AIR/AIE/XRT linkage.
Original failures, amendments, raw samples and compiler snapshots are preserved.

See `docs/taskflow-implementation-search-v10-validation.md`, the memory/cost/
automatic-search v10 contracts, and companion `taskflow/docs/implementation-search-v10.md`.
The fresh-directory entry is `../orbit-artifact/taskflow/scripts/run_taskflow_implementation_search_v10.sh cpu|all NEW_DIRECTORY`.
The all entry includes selected-candidate hardware and full-M5 validation; this
delivery ran phases separately and validates the clean-shell CPU entry. Future
work includes broader calibration, meaningful multicore placement, other
residency locations, and full-FFN cost coverage driven by measured error.

## Latest completed stage: checked public program graph v11

`ProgramGraph` in `include/Taskflow/ProgramMapping.h` is now the candidate-owned
public task/data/communication representation; `ProgramCommunicationPlan` is an
alias of this same type. Real `GenericAnalysis`/MLIR handles, source-task instance
domains and implementations, logical region versions, location copies, allocation
lifetimes, data requirements, pure order edges, transfer events, target resources,
execution boundaries and a conservative serial schedule share common identities.
The source module must outlive the graph. JSON/DOT are exported views rather than
an imported production program language.

Both the native mapping builder and finite whole-task planning adapter finalize
and independently check this representation. Checks cover real source binding,
instance access/iteration coverage, latest versions, rectangular transfers,
creation and synchronization dependencies, live accessible copies, final output
boundaries, allocation-union capacity and resource/schedule consistency. The
per-access coverage proof uses actual loop ancestry, with a finite 1M-cell limit;
repeated accesses outside a tiled loop are conservatively unsupported. Native
initializer projections and selected policy/phase/stage correspondence retain
their adapter-specific proofs. Alignment affects allocation capacity separately
from transfer payload. The 24,576 B local tensor-payload budget is unchanged.

Original native C++ search rebuilds the graph after structural/data actions and
saves the actual JSON/DOT, graph ID and source-plus-graph program-candidate ID.
Source candidate SHA and target/model evaluation identity remain separate. Two
rounds produced32 graph artifacts and7 legal tile+retention combinations; exact
resume and modified graph/DOT rejection pass. Direct-AIR cost features read common
source/instances, transfer events and region submissions, with stage/core features
explicitly backend-specific. Whole-command composite costs own communication and
runtime once; unknown component durations remain null. v9/v10 models and CGRA ML
are preserved without retraining or re-labeling old observations.

AIR now consumes common source/instance implementation choices, allocations and
transfer rectangles/copy endpoints. Native descriptors witness loop compression,
stages and legal binding. Generated AIR records graph, source-instance, allocation,
computation and communication identities. Its stronger serial-by-phase realization
and canonical/channel checks preserve the public dependencies. This supports the
existing resident scalar and tiled scalar/mixed8 paths, accumulation, and proven
stage-local readonly panels; arbitrary cross-stage SRAM sharing is not implemented.

The plain fanout example includes a genuine order-only edge. Separate versus
shared SRAM copies transfer128 versus64 B; a shared-plus-two-local target yields
192 B with transit retention versus256 B phase-local. These topology tests and
their costs are explicitly synthetic CPU planning evidence. Real M32/K128/N64
GEMM-scale uses65,536 B of logical system BO data, while phase-local/retained/
searched8/16/16 retained plans use3,584/9,728/9,216 B local pools and
221,184/122,880/122,880 B planned transfers. The actual searched candidate source
is compiled unchanged. All three plans' original graphs equal their compiled
graphs. Three seeds with2 warmups and5 hot calls per seed give63 real NPU outputs,
all finite FP32 atol=rtol=1e-5 passes, maximum absolute error7.15e-7. Detailed host
medians18.562311/18.725276/18.440053 ms do not establish a speedup. Wait includes
the installed polling/runtime/device interval; static payload is not measured DDR.

The new clean-directory `all` entry passes, along with38 Core/Amoeba/search/AIR
regression groups,36 common-graph assertions,71 cost contracts and15 graph-to-AIR
binding checks. Independent Core remains free of AIR/AIE/XRT/Neura linkage.
Frozen tools, source snapshots, original failures, raw API events, full outputs,
runtime-transfer and final storage audits are retained in companion
`results/taskflow/program-graph-v11/`. No new full-FFN timing is claimed.

The old `StaticCommunication`/`StaticPlan` and production Amoeba task-edge/network
scheduler still use their legacy communication representations. Sharing target
types does not migrate their data copies. Next steps are an explicit checked
Amoeba bridge, broader scheduling/placement/communication actions and performance
work justified by error/profiling evidence. Streams, multicast, arbitrary coherence
and physical bank/DDR proofs remain outside this finite implementation.

See `docs/taskflow-program-graph-v11.md` and the companion
`taskflow/docs/taskflow-program-graph-v11.md`. Reproduce with
`../orbit-artifact/taskflow/scripts/run_taskflow_program_graph_v11.sh cpu|all NEW_DIRECTORY`
and explicit built/frozen tool paths. This stage is intentionally not pushed.


## Stage v12: Amoeba consumes the public ProgramGraph (2026-11-05)

ORBIT remains a whole-program joint search over actual task structure, resources,
scheduling and data organization. Direct AIR and the new opt-in Amoeba bridge now
consume the same public ProgramGraph semantic interface, with different target
realizations. The v11 legacy-boundary paragraph above describes the historical
stage; this finite migration supersedes it for the new entry only.

`AmoebaMapping.h/.cpp` builds the source-checked graph from GenericAnalysis and a
real architecture contract. The shared data planner can partition exact access
rectangles into independently versioned cells. CGRA count/oriented shape are
public resource requirements. Amoeba availability locations have explicitly
unknown byte capacity (`capacity_bytes:null`, `capacity_known:false`); they are
not invented SRAM banks or DMA engines. The acceptance input is a4x4 CGRA mesh,
4x4PEs/CGRA,12 context slots,48 directed links at32 bits/cycle and1-cycle latency.

`decision-flow=program-graph backend=amoeba` registers native callbacks with the
original C++ `runStructuredSearch`. It reanalyzes real source after rectangular
task tiling/shape actions, checks source/common/target contracts and binds graph,
source, target, model checkpoint bytes and evaluator identity. Rectangular tiling
requires a strict perfect nest and direct injective axes for all accesses to
written storage; unsafe cross-slice accesses and executable prologues are rejected.
Different source/copy coverage failures remain unsupported, not silently lowered.

The same source is converted through the original Taskflow-to-Neura pipeline,
with bijective task lineage. The unchanged route-expanded DFG CGRA ML provider
predicts II; unchanged startup and firing-count conversion produce durations.
`buildTaskEdgeGraphFromProgramGraph` derives the authoritative original-algorithm
view from checked internal creation transfers and ordering requirements. The
original production scheduler/network oracle and actual joint mapper consume it.
No alternative Python planner, all-unit mapper substitute or retrained model is
used. The receipt associates a second production schedule using compiled II;
search score continues to use predicted makespan. Both are cycle estimates,
not Amoeba hardware measurements. External I/O remains unpriced.

Clean-environment acceptance retained6 legal candidates across2 rounds: baseline,
real2x2 task tiles,1x2/2x1 shapes and combinations. Predicted cycles are
67/72/74/66/68/76; compiled-II schedule cycles are69/68/69/69/68/68. Untiled
legacy comparisons are identical under the corresponding durations. Tiled public
transfers use four128-bit matched producer/consumer regions, versus legacy's
four512-bit whole-value transfers from the final producer tile. This is an
explicit opt-in semantic migration correction; original callers remain unchanged.

Independent checks cover42 common adapter assertions,17 native integration
groups,1281 finite graph/receipt assertions,38 Core/AIR regression groups,
17 original native-search checks,35 original Amoeba lit tests and8 unchanged
CGRA model golden groups. The same unmodified add-to-scale Taskflow input was
also compiled by frozen direct-AIR tools and executed on the real NPU:3 seeds,
2 warmups plus5 hot calls per seed,21 passing FP32 outputs at the existing
atol=rtol=1e-5. Raw API samples, references and generated AIR/AIE/ELF are retained;
no new M128 long timing or Amoeba hardware performance claim is made.

Legacy Neura-only/value/channel entry points, general replica/fusion/fission
materializers and StaticCommunication/StaticPlan callers remain available outside
this finite bridge. Shared remote residency/multicast and physical SRAM/cycle
buffer certification are not implemented for Amoeba. Future migration should
extend target capabilities and authoritative graph projections, preserving the
original algorithms and explicit model boundaries.

See `docs/taskflow-program-graph-amoeba-v12.md` and companion
`taskflow/docs/taskflow-program-graph-amoeba-v12.md`. Reproduce using
`../orbit-artifact/taskflow/scripts/run_taskflow_program_graph_amoeba_v12.sh cpu|all NEW_DIRECTORY`
with explicit v12 built/frozen tool paths. Acceptance and preserved failed runs
are under `results/taskflow/program-graph-amoeba-v12/`. No push was performed.

## Stage v13: static heterogeneous execution; analytical NPU cost priority (in progress)

ORBIT's public ProgramGraph now supports source-bound CPU/NPU execution regions,
shared input submission events, explicit completion/visibility, checked child
projections and static serial execution. CPU functions compile the actual MLIR
body through the existing LLVM toolchain; NPU children retain direct AIR/AIE.
Caller-owned mapped system BOs are shared across CPU functions and NPU contexts.
No runner host staging copy was observed, but BO sync, NPU system/local DMA and
CPU mapped loads/stores remain distinct real operations. Region-local24KiB
payload scopes share one serial allocation budget, not multiple physical banks.

The original three-task heterogeneous smoke and210 independently checked outputs
remain preserved. The first frozen v13 model is explicitly historical empirical
program-region interpolation, **not** an analytical communication model. It must
not supply fictitious physical coefficients. Additional candidate generation and
hardware search expansion are paused at the user's direction; already generated
plans and compiled holdout artifacts are retained without claiming execution.

CGRA continues to use its existing ML II model, feature/units contract and
original mapper/scheduler/network algorithms. NPU cost work now prioritizes
mechanism-driven analytical memory, communication and launch costs. Measurements
calibrate fixed API latency and size-dependent mechanism parameters, with frozen
independent validation. Actual copy paths, sync directions/counts/state, launch
and completion scopes come from the checked graph and installed execution
mechanism. Wait/compute/internal DMA that cannot be independently separated stays
an explicitly uncalibrated composite. Known subtotals do not become full-program
predictions; no charge is added twice. SIMD/VLIW detail and expanded performance
search wait for evidence that justifies them.

Implementation/limits: `docs/taskflow-program-graph-heterogeneous-v13.md` and the
companion v13 report. New output directories preserve all v11/v12 and prior model
identities. This stage remains unpushed.

### v13 focused mechanism-model acceptance and remaining limit

The memory/communication/launch mechanism implementation is now exercised end to
end. `MechanismCost.h/.cpp` derives API quantities and state from the checked
ProgramGraph plus the installed execution binding. Parameters mean fixed API
milliseconds plus milliseconds/requested-byte, not program operation counts or
physical DDR bandwidth. Region submit/completion, conservative sync, actual
copies, unknown CPU/wait/runtime composites and descriptive native DMA have
explicit accounting ownership. Resource aliases bind existing host capacity;
opaque and included work cannot receive duplicate independent charges.

Primary calibration/heldout runs validate1554 actual program outputs across3
seeds, max absolute error2.384185791015625e-7. All18 coefficients independently
reproduce. Ten C++ metadata reexports retain original graph/plan identity and
byte-identical implementations;1110 hot calls match actual API/copy/event spans.
A clean-shell full calibration/freeze/heldout/audit repeat completed separately.
Thirty constructor observations submit no kernels and distinguish host XRT
constructor API time from expensive file-hash/import/other preparation work.

Accuracy is deliberately not declared sufficient to resume performance search:
96 held API cells have29.10% MAPE, with a badly contaminated post-TO fit (max
436.74% error). A retained714691ns sample dominates71.5% of that training row's
sum. No sample deletion or heldout-driven refit occurs. Context population and
state changes further limit transfer. Complete cost remains null; when mechanism
parameters are requested, a historical empirical score cannot replace that null.
CGRA ML remains unchanged. Next work is a preregistered, noise-aware calibration
and explicit supported runtime-state coverage, followed separately by NPU
compute/wait understanding; broadened performance search remains paused.

Final regressions:43 Core/search/AIR groups,35 Amoeba lit tests, unchanged real
CGRA model contracts and v12 adapter differential,58 mechanism C++ assertions,
56 native export checks and225 heterogeneous mapping assertions. Core link
isolation remains intact. Sources, failures, primary freeze and repeated-epoch
measurements remain separately bound. No push was performed.

## Stage v14: BO sync state, observation boundaries and frozen API calibration

Performance/placement search and full FFN expansion remain paused. CGRA continues
to use its unchanged ML II model, original features, conversion and scheduler.
NPU work remains mechanism-driven analytical modeling, with measurements used
for parameter calibration and independent validation, not whole-region fitting.

The preserved v13 29.10% failure is explained in part by its retained 714691 ns
post-TO sample (71.5% of that training row's sum), unseparated API overload and
observer boundaries, and sequence-entry effects despite an unchanged BO. A new
binary/GDB/strace probe confirms userspace CLFLUSH with a 64-byte stride and no
manual-sync ioctl in the captured local branch. BO construction itself performs
an earlier implicit TO. This is not retrospective proof of v13's branch and
does not measure physical traffic or full cache writeback completion. Installed
DKMS source differs from the loaded Ubuntu module and stays separately labeled.

New C++ `SyncState`, `SyncBoundaryMapping` and `SyncCost` interfaces track exact
BO/generation/range observations, checked graph/ABI/copy/event correspondence and
guarded cost queries. `taskflow-sync-opt` rebuilds the actual Taskflow ProgramGraph
and derives successful-path queries from its MechanismFacts plus observed BO
bindings. `taskflow-sync-replay` independently consumes ordered recorded accesses,
sync return and command/wait events. Neither elapsed time nor test name selects
state. Missing/pending/mixed content, unknown padding, absent producer evidence
and mismatched runtime/observer/site/condition return unknown or a diagnostic.

The primary post-TO batch formula is entry+n*(call+line*requested_cachelines),
with separate invocation forms, aligned ranges and an explicit fresh-BO/zero
prepared-context domain. The original runner uses four fixed observed API-site
parameters over 1024--4096 byte BOs, separately bound to its detailed observer and
two reused Prepared contexts. These are API-envelope coefficients, not physical
bandwidth, internal DMA or computation. Compute/wait and complete-program time
remain null; known subtotals never become search scores or duplicate charges.

Exploration preserves 34992 sample sequences and 2916 independently checked device
outputs. New calibration/heldout runs preserve 4284 micro samples per split and
270 / 486 original program calls including warmups. Frozen primary model SHA is
`a82c00acff10a4eb128659f62c1b3fc1f9462017fcc5871413da7d843f79da37`.
Independent heldout batch acceptance is 35/36 cells (MAPE 5.175%); implicit 2048 B,
n=23 remains a failed extrapolation, 541.806 ns absolute error. Program API subtotals
pass 4/4 groups (MAPE 5.992%), but individual API/seed errors reach 27.686%.
All 486 heldout outputs pass unchanged FP32 tolerances and independent float64
references, max absolute error 2.384185791015625e-7. No heldout refit occurs.

A clean-shell complete calibration/freeze/heldout/audit repeat separately passes
structural/numerical checks, again 35/36 batch and 4/4 program-subtotal thresholds.
Its independently frozen parameters and observations remain a different epoch,
not a replacement for primary results. These metrics are not a same-boundary
percentage improvement over v13's 29.10% error.

Independent integration checks 486 calls / 1944 actual sync records, preserves original
Taskflow/AIR/ABI/ELF identities, and C++ replays 392 queries from two held program
calls and 36 held micro sequences. Batch charging occurs once per sequence;
348 later queries are state-only. New units/export/replay cover 520 assertions;
48 Core/search/AIR regression groups, 35 Amoeba lit tests and unchanged eight-candidate
CGRA contracts pass. The Core tools have no Neura/AIR/AIE/XRT dynamic dependency.
Unchanged AIR/Amoeba regressions explicitly reuse frozen v13 backend binaries;
the new Core tools are built in a separate v14 directory.

See `docs/taskflow-sync-state-v14.md`, companion API audit, and
`../orbit-artifact/results/taskflow/sync-state-v14/RESULTS.md`. The clean-shell
entry is `../orbit-artifact/taskflow/scripts/run_sync_state_v14.sh`; output must
be new. Historical artifacts, frozen models and failed attempts remain intact.
Next work is to explain the persistent small-range n=23 residual and independently
identify remaining copy/submit/wait/internal-DMA/compute mechanisms. Do not resume
expanded performance search or infer hardware traffic from requested bytes.
This stage remains unpushed.

## Stage v15: device path evidence and guarded copy completion costs

Performance/placement search and FFN expansion remain paused. CGRA keeps its
existing ML II model and original mapper/scheduler/network contract. V14's frozen
sync model, 35/36 heldout result and failed cell are preserved unchanged.

The public ProgramGraph now supplies checked communication facts to a narrow
device-cost provider. `DeviceTransferCost` derives copies, versions, endpoints,
resources, regions and payload from the existing GenericMappingPlan;
`DeviceTransferAudit` checks actual generic-form final AIE IR for source/graph/plan
identity, ABI, flows, core/runtime BDs and successful output-token completion.
`taskflow-transfer-opt` is the C++/MLIR entry, not a Python planner. Explicit
`--cost-model` or target-config `device_transfer_model` binds evaluation without
changing program identity. Missing parameters or an unmatched domain stay unknown;
duplicate DMA/compute/submit/wait prices are rejected. Core remains backend-link
independent.

The verified path is noalias identity-layout FP32 copy through system BOs,
shim(0,0) and core(0,2), with no memtile. R uniform tiles generate2R public transfer
requests and2R core transactions, but only2 coalesced whole-tensor runtime BDs.
These are different counts of the same payload, not additive DDR bytes. The output
token await and later FROM sync distinguish logical device completion from CPU
visibility. Shared BOs eliminate extra runner staging here, not device transfers.

The primary frozen completion-envelope formula is219391.9444ns +
2.636321779ns/requested-byte for the exact guarded duplex-copy path. It includes
local copy, DMA, locks, configuration/runtime and200us polling together; it is not
pure DMA, isolated launch, bandwidth, or full-program cost. The separate segment
coefficient is omitted/unidentified. Two-parameter nonnegative calibration uses
nine preregistered cell medians. Model SHA:
`cc02bfdf882f621fc0bc78f3d9b69a2d5c7c63978237e840cb78d000f124d9ab`.

All297 calibration and297 heldout outputs pass exact independent copy checks.
Primary heldout accuracy is8/9 cells, MAPE19.44%; E49152,C128 remains failed
(501536ns error,439393ns limit). All samples, per-seed instability and tails are
retained; no heldout refit or sub300us ranking claim. A separate nonuniform-tail
copy passes3 real seeds while correctly remaining outside this cost model.
Standard-toolchain trace on/off executes10 correct copies but captures only zero
words; no DMA clock/duration or physical byte claim is made. Trace drain/reset is
an unresolved hypothesis. V14 BO-sync conditions differ and cannot be reused for
this new observer/context/ABI by analogy. Complete cost remains unknown.

See `docs/taskflow-device-transfer-v15.md` and companion artifact results/path audit.
The clean-shell entry consumes explicit SHA-bound compiled packages and new output
directories. Next work is verified device trace drainage/event coverage, runtime
polling/completion variation and matched-condition boundary API calibration before
splitting or broadening the model. This stage remains unpushed.

V15 final validation:50 Core/search/AIR groups,35 original Amoeba lit tests,
unchanged CGRA model contracts and v12 adapter differential pass. New provider
units cover41 assertions and export/config-binding tests113 checks. Fresh Core
and AIR builds are separate from historical binaries; Core has no backend dynamic
dependency. A clean-shell full replay independently freezes another model before
its heldout and obtains9/9 cells (14.26% MAPE), with297+297 correct outputs. This
separate epoch does not replace primary8/9 or relax its threshold. The existing
small copy program's actual calls partition into completion, sync and remaining
runner work; only the guarded completion subtotal is priced, full cost stays null.

## Stage v16: public event timing and finite NPU whole-invocation validation

The new public `ProgramTiming.h/.cpp` evaluates the existing checked ProgramGraph
and reuses `scheduleStaticPlan`; no second graph, planner, scheduler or model
configuration is introduced. Exclusive implementation coverage, explicit native
invocation ownership, resource demands, shared-capacity reservations, buffer
version reuse, temporal allocation checks, resource predecessors and critical
paths are implemented. Unknown costs retain reasons, known work and rigorous
precedence/resource lower bounds; they do not become complete zero-cost scores.
Fixed shares are not dynamic fluid bandwidth. Composite allocation lifetimes
are conservative and physical capacities can remain unknown.

`GenericProgramTiming.h/.cpp` adapts the existing guarded direct-AIR provider.
`taskflow-opt --program-timing-output` exports native features, the priced whole
invocation, algebraic contributions and a separately unknown internal-event
view. Generic linear composites support up to eight distinct feature terms;
the specialized existing implementation-composite remains exactly four terms.
Core retains independent linkage. Original Amoeba/CGRA ML, mapper, scheduler and
network contracts remain unchanged.

Actual scalar FP32 multiplication uses `__mulsf3`; scalar additions use lane
insertion/vector-add/extraction. A shared scalar-operation fit fails on training
(109.4997% MAPE). A training-only revision separates product and add/subtract work
within one whole-invocation composite, before any new heldout performance.
The five frozen coefficients in ns are call232201.11122139467,
plan-stage49678.8769524652, requested-DMA-byte2.1452334001055914,
scalar-product62.69654277482823 and scalar-addsub1.4173927143738569.
They include correlated local memory/loop/runtime effects, not independently
isolated physical link bandwidth or instruction/configuration latency.
Model SHA is d14d6f222304dfdc30fe836a0db33c322a8f509440be68a80053c85d1a375b74.

The boundary is one prepared bound-BO runner call, including actual TO/FROM sync,
submit/wait/configuration/DMA/local work and runner gaps. Context/BO preparation,
caller input reset/poison and evidence/reference work are excluded. The finite
native scalar source/shape/tile/resource domain binds eight prepared contexts,
20us instrumented polling, runtime, observer and toolchain. Missing or mismatched
coverage stays unknown. Pure DMA/core intervals and measured overlap remain
unidentified: v15 trace has no verified partial-buffer drain before PDI reset.
Host polling observations only bracket command-state visibility, not physical
device completion or cross-tile clocks. Polling probes have changed observer
conditions; they are not same-condition v15 revalidation.

Eight training and eight heldout plans each preserve264 calls (3seeds,2warmups,
9hot repeats); another matched new process preserves264 calls under the exact
same parameters. Independent source-order FP32/f64 checks pass all792 calls.
Primary heldout is7/8, MAPE5.586689%; same-model new-session is6/8,
MAPE9.759001%. The add-chain heldout fails; fanout also fails the new-session
cell gate. Sixteen native feature/evaluation graphs equal actual compiled graphs
in full JSON. Three heldout candidates retain original C++ generator actions,
source-SHA identities and searched/lowered graph equality. Their predicted
38--91us gaps are not resolved by measured paired distributions. The registered
search-readiness gate fails; the conditional two-round performance search is not
resumed and no speedup or reliable ranking is claimed.

GEMM retains C partials across K panels locally. The phase-local complete fanout
materializes X in a system BO for two consumer reloads. A separate same-source
resident8x64 control proves one local X for both consumers, no X BO/DMA,6144 vs
12288 requested bytes and8192 vs1024 B local payload. Its66-call two-context
measurement is outside the frozen domain; the provider correctly returns OOD.
A clean-shell hardware entry repeats66 numerical calls separately. Static BD
payload accounting matches all16 main packages. Configuration audit r1 prose
mistakenly counted multiple BDs as multiple children; r2 explicitly corrects it:
this batch has plan-stages==owning-child==configure/run counts, plus one wrapper.
The erroneous report and all failed attempts remain preserved.

The original v15 frozen cc02bf parameters are replayed with original runner,
observer,9contexts and200us polling:297 exact outputs,0/9 time cells,
MAPE62.608429%. Runtime snapshots match, but unobserved thermal/load/history
factors remain possible; the cause of drift is unidentified. This is actual
same-parameter cross-session failure, distinct from old recalibrated9/9.

Final validation includes44 public timing assertions,77 direct-AIR provider/
timing-bridge assertions,34 clean-shell Core groups and51 clean-shell combined
Core/AIR/original-native-search groups,17 original public-Amoeba adapter/differential
groups,8 unchanged CGRA golden candidate/contracts and35 original lit cases.
Frozen v13 Amoeba tools remain explicit
where original-backend tests use them; the new Core/AIR binaries are separate.
The new `run_program_mechanisms_v16.sh` CPU and small hardware modes both pass.
Full results, exact commands, raw distributions, independent review and preserved
failures are under companion results/taskflow/mechanism-program-v16/. See
`docs/taskflow-mechanism-program-v16.md` and `docs/taskflow-program-timing-v16.md`.
No push is performed. Next model work needs optimization-relevant resolution,
new preregistered add/local-memory/loop controls, and independently supported
internal overlap observations; do not widen frozen domains or fit heldout labels.

## Stage v17: explicit data organization and conditional plan contrasts

`ProgramMappingChoice` now exposes explicit `whole-program` and
`tiled-phase-local` realizations alongside the existing `readonly-panel` and
legacy adapter-auto `phase-local` spelling. The old default canonical identity
is retained. Whole-program requires the conservative sum of all storage payloads
to fit and rejects incompatible target tiling/vector requests; there is no
implicit fallback. A whole-program source action removes the old source tile
attribute; an ordinary subsequent tile action selects tiled-phase-local. The
checker binds canonical policy/realization/grouping/scope to actual source
attributes. These choices retain the existing `single-herd-resident` and
`serial-tiled` AIR paths, not a parallel graph or lowering pipeline.

The original C++ automatic search discovers both realizations and immutable
input panels from actual affine accesses. It checks mapping proposals using the
same planner before consuming evaluation budget, preserving failed proposal
source/diagnostics. Broadcast `C[i,j]=A[0,j]` uses this ordinary invariant-input
proof without arithmetic-name matching. The checked ProgramGraph records one
retained local copy and its multiple source consumers. Whole-program fanout
keeps intermediate copies local; the tiled realization materializes system BO
copies and reloads. Selected source SHA, action/parent chain, native plan and
graph identities continue through feature collection, AIR checking and backend
receipts. Runtime BDs, public delivery requests and core transactions are
separate views of payload, never additive physical DDR byte measurements.

`GenericProgramContrast.h/.cpp` and `taskflow-contrast-opt` compare two existing
checked plans. The existing direct-AIR model schema now accepts
`contrast-composite` for a guarded candidate-minus-baseline implementation
contrast with signed contributions and an explicit training-derived error bound.
Supported structural change classes are one-stage readonly input retention with
fixed ABI/output delivery, and materialized intermediate regions changing to
whole-program residency. The latter keeps configuration, sync, buffers, locks
and generated local loops in one exclusive coupled cost. Models bind source
family, architecture, arithmetic implementation, both feature domains and actual
runtime/toolchain/condition identity. Unsupported coverage stays unknown.
Absolute invocation times, independently identified DMA/core intervals, physical
bandwidth and observed overlap remain unknown. The absolute provider explicitly
returns uncalibrated for contrast-only parameters, so no signed delta becomes a
complete search score. Candidate availability is not a ranking-validity claim.

The clean-shell final Core checkpoint passes all36 regression groups, including
50 mapping assertions,18 paired-provider assertions,44 public timing assertions,
new broadcast and fanout two-round automatic candidate/source identity/graph/tile
composition and resume regressions, the old three-mode automatic-search suite,
and old data-policy alias/default/capacity/body regressions. A separate run using
the explicit preserved v13 Amoeba optimizer and current Core passes17 public
adapter/differential checks and8 unchanged CGRA golden candidate/contracts;
no model training or device submissions occur in those compiler checks.
A subsequent combined compiler run passes53 groups (36 Core, one original-native
search and16 AIR groups). Separately,10 v17 data-organization AIR assertions,
15 original ProgramGraph-to-AIR binding assertions and the preserved35 original
lit cases all pass. These runs reuse explicit existing tools and confirm their
SHA256 values unchanged before/after; no SHA-bound production optimizer is
rebuilt. The broad suites recorded18 contrast assertions before the later
focused JSON string-lifetime regression was added; that repaired-provider check
is reported separately from the broad-suite count. Broadcast M64/N512 with tile1x512 requests
262144 versus133120 bytes under repeated versus retained input delivery, both
one stage and4096 local payload bytes. These are checked compiler facts, not
latency measurements. N1024/tileN256 retention instead creates four stages;
that comparison must not be described as a same-stage input-only control.

See `docs/taskflow-data-organization-actions-v17.md`. The clean-shell CPU replay
entry is `scripts/run-data-organization-v17-cpu.sh`; the existing
`scripts/build-core.sh` uses the LLVM-cache compiler and builds the contrast
and public test targets. New evidence belongs under companion
`results/taskflow/data-organization-v17/`, with fresh directories for every
attempt. Old v13-v16 models, failed attempts and raw distributions remain intact.
Amoeba/CGRA retains its original ML, mapper, scheduler and network contracts.
Hardware calibration, unseen-plan/session accuracy and search-readiness status
must be reported from the separately frozen study; this compiler checkpoint
makes no new physical-performance claim. No push is performed.

The v17 formal heldout-source diagnostic now runs two original C++ unpriced
rounds on broadcast M32/N768 and fanout E768 (24-candidate budget,beam4), producing
16+15 legal candidates with null absolute scores and empty rankings. All31
sources are independently replanned with full graph equality. Frozen contrast
queries return two conditional predictions matching the original formal pairs,
20 explicit unknowns and nine same-implementation guard rejections with original
stderr; no rejected query receives a fabricated cost. Four formal source SHAs
and their full compiled graphs establish eight joins to existing heldout and
cross-session raw evidence. No new NPU measurements occur. Structural two-action
representatives are requested for compile-only receipts, independently of timing
acceptance. See companion `two-round-diagnostic-r1/{TABLE.md,rows.json,
formal-identity-audit.json,representatives.json}` and the artifact-side
`diagnose_data_candidates_v17.py` fresh-output entry. This remains an UNPRICED
diagnostic, not the conditional performance-search demonstration or a readiness
claim. Frozen model, target, parameters, config and tools remain byte-identical.

Final v17 physical-study checkpoint:28 compiled packages/14 original automatic
pairs have complete searched/lowered graph equality and actual AIE/ABI/code
ledgers. Formal training/heldout/same-parameter-new-process evaluation preserves
1944 calls (1620 hot,324 warm) and3024 independently exact FP32 output arrays,
with corresponding exact FP64-view checks. A separate clean-shell hardware-entry
smoke passes108 calls/arrays and is excluded from fitting and the gate.
Registration r2 adds two tail training sizes before any v17 NPU execution after
static inspection found48 child configurations for nonuniform fanout versus3 for
compact shapes and1 for residency. The frozen exclusive contrast uses A delivery/
byte terms and B stage/mode/byte terms. Full-rank numerical identification does
not isolate physical mechanisms. A's zero delivery fit is a constrained boundary,
not evidence of free DMA. Error allowances are descriptive training/LOO values,
not confidence bounds or guarantees.

Frozen model SHA is a436342b55f5c71a8e79c2f8410c295cb4b39215864ee9b24c026e4cecf12116;
independent pre-heldout receipt SHA is
f2f85a27612b08838151a25323d565097f3689679861dad4fe1d73d4cf168229.
Both unseen fanout tail shapes pass signed-difference accuracy and resolution in
both sessions: predicted−2020.233/−2021.854 us, primary−2015.171/−2012.262 us,
new-process−2000.192/−2025.021 us. This validates only the coupled48→1 group.
Input-copy effects remain unresolved in both sessions; ordinary3→1 fanout
training effects are also unresolved. The two-class registered gate fails;
performance search stays paused. Original two-round unpriced diagnostics retain
31 legal candidates,2 known contrasts,20 explicit unknowns and9 implementation-
guard rejections, with null absolute scores and empty rankings. This is not a
performance-search demonstration.

The first freeze's bad JSON is preserved: contribution StringRef fields outlived
the model. Owned strings and a19th focused provider assertion repair it before
the successful freeze; fourteen real-model exports and independent recomputation
pass. No fit, split or threshold changed. New model terms never become complete
absolute scores; pure DMA/core time, physical DDR bytes, device clocks, internal
overlap and arbitrary-plan sorting remain unknown. The24 KiB capacity is payload
policy; E768 resident also reserves4 KiB stack and other nonpayload overhead is
not fully modeled. No further trace retries or new runtime were required.
See `docs/taskflow-data-organization-v17.md` and companion
`results/taskflow/data-organization-v17/RESULTS.md` plus independent review. All
hardware execution was main-agent serialized under `/tmp/npu.lock`. This stage
remains unpushed, preserving all earlier models, histories and failed attempts.

Two additional original-driver compound candidates pass compile-only physical
audits: tile384→retention produces two broadcast configurations; whole→tile8
restores fanout materialization and the X BO. No extra timing claim follows.
Priced search and physical plan IDs are explicitly bridged by cost features
execution_fingerprint, full ProgramGraph equality and all nonpricing plan fields;
price-only attachments are not mistaken for changed execution. Unfiltered final
lit discovery passes35/35 after pure fixture inputs are excluded. The preservation
audit finds1114 original ORBIT dirty/untracked files unchanged,10 edited within
v17 scope, no missing files; all72 original artifact dirty/untracked files and
the exact v15/v16 frozen-model hashes remain unchanged.

The two diagnostic combinations are now compile-only audited under
`two-round-compiled-representatives-r2/`, with diagnostic links in
`representative-compile-links.json`; they remain unmeasured and have unknown
contrast coverage. Search price-attached plan IDs differ from compiled physical
IDs only through price metadata: all other native fields and full graphs match,
and searched execution_fingerprint equals compiled physical plan_id. Keep the
preserved initial literal-ID assertion failure and r2's explicit successful
broadcast artifact reuse. Final unfiltered lit discovery excludes the three
non-RUN input fixtures and passes exactly35 original tests, recorded separately
in `final-lit-discovery-r1/`. No tool/model rebuild or NPU submissions occurred.

## v18: authoritative execution groups and explicit estimation loop

The v18 public path retains actual Taskflow source as the executable authority:
canonical native actions -> GenericAnalysis/GenericMapping -> checked
ProgramGraph -> event mechanism features/timing -> the original C++ search beam,
with the same checked plan consumed by AIR/AIE lowering. See
`docs/taskflow-execution-graph-v18.md`. New evidence belongs under companion
`results/taskflow/execution-graph-v18/`; preserve all v15-v17 freezes, binaries,
raw distributions and failed attempts. Do not rebuild an old frozen optimizer.

ProgramMappingChoice adds optional taskGroup/retainedStorages source indices,
serialized as task_group/retained_storages only when nonempty. shape-major
batches proven independent tile shapes inside a source task. tile-major executes
the corresponding producer and consumers before reusing their spatial block,
batching independent equal-shape blocks across tails. These are public grouping
and data-lifetime decisions, realized through the existing serial-tiled mapper,
not a new NPU-specific residency policy or Python planner. Initial tile-major
support is a complete closed pointwise group with matching grids, exact retained
regions, one producer and actual consumers. In-place/reduction/panel work,
non-tile work, partial groups and pure-order whole-task barriers are rejected.
The original source implementation choice is preserved.

ProgramExecutionGroup lives in ProgramMapping.h. ProgramGraph.executionGroups
is authoritative, explicitly serialized into v18 graph identity, and checked
against the native witness. Each group carries member events, phase interval,
resource, memory, dependencies and concrete spatial owner. Native resident
programs have one explicit group as well. MappingAllocation declares
region-addressed versus reusable-slot addressing and physical owner. The current
validated owner is npu.tile(0,2); a resource-pool name is not a physical owner.
Retained local X is one shared version/copy with two consumers and checked
release/overwrite ordering. It has neither a host allocation nor a host transfer.
The serialized host_buffers and AIR ABI follow graph allocations; source argument
positions remain intact. Group/lifecycle/owner mutations must be rejected.

Native invocation lifecycle is explicit: visibility to device, one shared
submission, completion and visibility to host. Arbitrarily splitting ordinal
intervals does not create valid refined invocations. Public buffer-ordering and
capacity checks account for reusable-slot aliasing, final users and release
boundaries. Semantic source dependencies remain separate from group execution
barriers and time annotations. No transformation may delete the repository
fanout fixture's pure-order dependency to make a grouping feasible.

The historical measured v17 E768 plain fixture is a separate RAW-only diamond
with no pure-order edge. Its shape-major materialized control is 48 -> 6 groups
at unchanged18432 requested bytes. The repository fanout-shared-v17.mlir fixture
has one additional pure-order edge; its [1,48,1] shape-major control is 48 -> 6
with unchanged12288 requested bytes and that edge preserved. Tile-major is
rejected for the latter. Do not describe the measured E768 source as carrying
that repository-only barrier or insert one into historical source evidence.

ProgramMechanismCost annotates the existing graph and delegates resource time
evaluation to ProgramTiming. GenericMechanismTiming uses the existing
orbit.direct-air.cost.v1 schema, kind=event-mechanism, mode=strict|estimate.
Parameters retain units, source and calibrated/architecture/assumed provenance;
unknown or assumed backend link defaults cannot become architectural facts.
Strict mode preserves unknown required mechanisms and known subtotals/bounds;
estimate mode may use explicitly declared assumptions and lists them. Charges
are activity-exclusive through event_timing, never a duplicated whole-call
composite. Fixed-duration effective bandwidth is an assumption, not validated
fluid sharing or observed hardware overlap. Logical scalar operations/accesses
are not a complete SIMD/VLIW simulator or isolated DMA/core measurements.

The add diamond X=A+1;B=X+2;C=X+X has131072 logical bytes, above the24576-byte
payload policy. The checked512-element materialized plan uses4096 local bytes,
196608 requested transfer bytes and3 groups; retained blocks use8192 local bytes,
98304 requested bytes and1 group. Retained1024-element blocks use16384 local bytes.
These are compiler facts; stack and other nonpayload allocations remain separate.
Tail M65/N136 has4 grouped shapes and106080 requested bytes. The public tests
cover shared-copy users, source/action identity, dependence, owner, channel,
capacity, lifecycle and illegal grouping/coverage cases.

Keep three claims distinct: (1) an estimation model actually controls the
original search's next-round parents; (2) it correctly orders real candidates;
(3) a measured candidate improves the baseline. Formal estimated-search-r6
establishes (1), archiving20 candidates,17 assumption-priced candidates, exact
source/graph/plan identities and score-selected second-round parents. The same
assumed table in strict mode has unknown absolute scores and an empty ranking.
No workload-specific total-time fitting is used. Four compiler representatives
cover baseline, retained512, retained1024 two-action composition and materialized
1024 control. CPU/compiler evidence alone does not establish (2) or (3).
Hardware errors, real ordering, measurement resolution and speedup belong in the
separate device report; never substitute assumed ranking for those results.

The current compiler checkpoint records38 complete Core regression groups,
28 grouping assertions,15 initial event-provider assertions and77 public timing
assertions; later focused provider additions are reported separately rather than
silently changing the broad-suite count. Four physical packages have exact
source/full-graph/native-plan correspondence and r2/r3 equivalence receipts.
Amoeba/CGRA retains its ML, mapper, scheduler and network contracts. New compile
receipts and clean-shell replay commands must select tools explicitly and retain
tool hashes. NPU experiments remain main-agent serialized under /tmp/npu.lock.
No push is performed. Device results and final regression additions are appended
separately after their evidence is available.

The subsequent v18 topology audit found that matching source/graph/plan IDs alone
was insufficient: a synthetic phase-frontier chain made the first consumer's
output DMA complete before the second consumer computed, whereas the real
single-buffer core released both outputs only after all block computations.
Preserve those earlier graph/search/compile artifacts as superseded. The repair
must expose the actual group-release partial order: all block computations before
both exports, next input refill after the previous core release, and next core
entry after both prior output slots are free. Source phase is instance binding,
not event-order time. Slot overwrite checks use all readers; lastUsers retains
its historical final-ordinal witness meaning and is not a general DAG frontier.
New search predictions and physical topology receipts must bind the repaired
graph before a same-plan execution claim is made. Do not hide this correction
behind metadata equality or describe an unimplemented all-terminal barrier as a
backend constraint.

Initial v18 block-release repair: public event construction now defers both output
exports until the last block compute, permits input refill after prior core
release, and waits for both prior output slots at next core entry. All-reader
slot reuse is explicit. Timing no longer invents releasedAfter-1 dependencies
from a validity ordinal. The initial repaired estimated-search-r5 retains20 candidates,
17 assumption-priced candidates and the same score-selected two-round action
composition. The same parameters were retained; old observations are already
known, so this repair is not an untouched-holdout experiment. Fresh compiled-r3
packages match all four full final graphs and full native priced plans, with
independent actual-AIE topology and buffer-ordering audits. A final empty-group
standalone-checker guard rejects malformed input without changing any valid
plan: all four re-exported complete plan JSON values and priced IDs are exactly
equal to r5, documented in standalone-group-guard-r1/valid-output-bridge.json.
That intermediate guard checkpoint reported37 grouping,19 event-provider and83 public timing
checks. Full clean-shell replay and device outcome summaries are recorded from
the final evidence rather than inferred from these assertion counts.

The final generalization removes fabricated phase frontiers inside every native
serial-tiled group, including materialized single-task groups. The core chain
(including initialization) remains explicit; input refill waits for actual old
readers, output-slot overwrite waits for actual exports, and only configured
group transitions inherit the complete parent terminal frontier. Reduction
versions, inline initialization, readonly reuse and all-reader alias protection
remain checked. The final directed grouping unit has41 checks; legacy panel/K
mapping50 and tiled-domain48 assertions pass alongside public timing83 and
mechanism19. The original final search is estimated-search-r6, fresh physical
packages are compiled-representatives-r4, and the final observation freeze is
observation-freeze-r5. All four plans have actual-AIE/public bidirectional causal
order checks, including materialized groups and real configuration transitions.
Do not extrapolate that certification to the historical nonserial resident path.

The fresh replay entry is companion
`taskflow/scripts/run_taskflow_execution_graph_v18.sh`: cpu runs the clean
compiler suites, generate materializes a numerical source fixture then calls the
original C++ search and compile-four/freeze wrappers with zero device submissions,
and npu consumes a read-only SHA-bound tolerance protocol in a new output with a
120-second process timeout. NPU locking remains inside the observer. The default
is the finite arithmetic tolerance adapter, not the preserved failed exact-bitwise
trial. Final main-agent device replay paths are npu-diamond-r6/r7. No model or
source-index manifest may be mutated after another artifact binds its hash; tool
snapshot metadata belongs in a separate file. Earlier source-index metadata
amendment, exact reconstruction, failure and revision receipts remain preserved.

Final v18 clean-shell evidence is final-clean-regressions-r4: Core38 and
combined55 groups, AIR13/10/15, preserved original compiler17/CGRA8 and
unfiltered lit35 all pass. Included public units report41 grouping,83 synthetic
timing and19 mechanism checks. The original optimizer remains the preserved
binary; the current Core adapter is checked against it, not rebuilt here.
Tool hashes are unchanged throughout the replay. The prior r3 metadata-assertion
failure is preserved; only its obsolete cross-stage wording was changed to the
current cross-group wording, retaining all data/lifetime checks. The fresh
generation entry is independently exercised in fresh-generate-r2 with zero
hardware submissions and all four full graphs/priced plans exactly equal to
formal search r6. Final device observations remain exclusively the main-agent
npu-diamond-r6/r7 runs; CPU reproducibility is not another timing observation.

## v19: scoped subgraphs, target bindings and original search composition

The native path now represents partial and multiple residency scopes through
`ProgramMappingChoice.residencyGroups`: each group names source task indices and
retained logical storages. Canonical source uses `orbit.mapping.residency_groups`
with `tasks` and `retained_storages` fields. The old task_group/retained_storages
pair remains a single-group compatibility input. Do not mix the two spellings.
The source action, checked mapping and AIR lowering consume the same source
analysis and public graph; no experiment wrapper may manufacture a plan.

`GenericMappingPlan.residencyScopes` resolves each choice to source phase bounds,
task members, retained storages and derived boundaryExports. Queries are
`genericResidencyScope(plan, phase)` and
`genericLocallyRetained(plan, phase, storage)`. Retention is phase-local, never a
global instruction to drop a logical storage's host allocation. An intermediate
with outside consumers retains its host boundary and exports each produced block;
internal readers and that export share one local version. A purely internal
intermediate needs no host copy. Output-role data can be retained for internal
consumers while preserving its mandatory caller export and original ABI slot.
Every overwrite waits for all actual readers and exports.

The current proof admits multiple ordered, nonoverlapping contiguous intervals
of the existing topological source order, with complete prefix/suffix coverage.
It requires independent static pointwise grids, matching exact retained regions,
one source writer and actual internal readers. Whole-task pure-order barriers
inside a scope, reductions/panels, in-place updates, multiple writer tasks of one
storage and repeated external-input use across different grouped tasks remain
explicitly unsupported. Noncontiguous groups require a dependency-convex
contraction and alias/version proof; deleting the interval guard is unsafe.
Broader streaming and concurrent native configurations are not implied.

ProgramExecutionGroup.dependencies records source-induced intergroup order;
implementationDependencies separately records the chosen configuration sequence.
The actual event graph honors both. Source phase ordinals remain instance
witnesses rather than DMA-drain barriers. Backend-declared
acquireAllInputsAtGroupEntry adds explicit readiness from every block input to
its first compute, including inputs consumed only by a later grouped task.
maxGroupInputs/maxGroupOutputs are target capabilities, not universal public
hardware constants. Group/resource and allocation/memory owners resolve through
ProgramTargetBinding lookups. Public planning does not parse NPU coordinate names
or impose a 24 KiB ceiling; the actual backend still rejects unsupported targets.

Invocation lifecycle is declared independently of the device identifier:
host-device and host-only are explicit policies, with an empty-field adapter for
preserved historical plans. Native placement and task reports use the selected
compute resource's device/resource/owner, rather than invented herd/core labels.
The serialized identity policy is
orbit.chosen-plan.v19.excludes-proposal-diagnostics: alternative discovered
mapping opportunities remain in reports but do not affect either chosen-plan
identity or the unpriced execution fingerprint. Cost metadata still distinguishes
price-attached plan IDs from unpriced fingerprints. Old frozen IDs stay immutable;
bridge receipts enumerate identity/lifecycle/reporting changes and compare all
selected physical facts, not merely source hashes or numerical predictions.

Original C++ discovery uses source writer/reader relations, not task names,
operator names or fixture indices. It proposes local scopes and, from a selected
source, another disjoint group. Original tiling actions remain composable. Search
bounds live in existing native_auto_candidates fields max_residency_proposals
(default8) and max_residency_groups (default4), each1..64. The original beam,
checkpoint/cache and legality machinery make all decisions. Python replay only
invokes the compiler and audits artifacts. See
`test/TaskflowGeneric/subgraph_search_v19.py` and
`docs/taskflow-subgraph-residency-v19.md`.

New evidence belongs under companion results/taskflow/subgraph-execution-v19.
The earlier post-identity search revision estimated-search-r2 snapshots its real compiler, preserves
29 candidates/26 assumption-priced plans and retains strict unknowns. Its score
selected the actual second-round parents before scope/tiling composition. The
initial measured examples remain bound to immutable r1 sources/tools/protocols;
new typed-lifecycle and identity policy are documented through explicit seven-
source bridge receipts. The output-role {3,4} example is an existing r2 archived
candidate, exported separately for targeted validation. Algorithmic closed-loop
search, executable causal-order correctness, real-time ranking and acceleration
must continue to be reported as separate claims. Preserve all v18 freezes and
failed v19 fixture/build/audit evidence. Production changes are frozen pending
final reproduction; no automatic push.

The final v19 late-input audit adds an explicit resolved inputReleasePolicy:
group-tail for the original template, last-reader for the selected frontloaded
late-input template. All-input readiness at first compute remains mandatory.
The latter permits each input refill after its own last reader; output release
stays at the block tail. This is a generic implementation/lifetime fact derived
from source access and selected entry behavior, not an output/workload special
case. It is serialized, reconstructed, and rejects unknown values in standalone
communication checks. Preserve the initial failed output-role acquire and
release receipts; require actual-AIE proof before claiming the new template is
executable. No fitting or new model coefficients accompany this correction.

Final v19 public delivery uses estimated-search-r3, with29 archived candidates,
26 assumption-priced plans and strict unknown scores preserved. The final Core
snapshot8173d56e...b828e6 has40 regression groups passing, including target44,
scoped-group45, public timing83 and mechanism19 checks. The validation-only
resolved-scope guards reject empty widths, invalid spans and unknown policies
before modulo/index operations. final-search-guard-bridge-r1 proves all29 final
source artifacts, complete graphs, priced plans, scores, action chains and
score-driven selections exactly equal the fresh-generation archive, without
normalization. Old identity revisions remain explicitly bridged and immutable.

The final source winner is4662641667...5a52: groups{1,2,3} and{5,6}, with the
latter using the declared last-reader input lifetime. sensitivity-r3 re-evaluates
all29 unchanged graphs under seven declared parameter settings through C++; the
same unique winner and exact first-round parent set survive7/7 settings, with
26 priced and3 unknown candidates each time. Earlier sensitivity-r1/r2 results
remain preserved for their pre-release-refinement graphs.

Both late-input mismatches are closed through real implementation behavior:
all inbound block inputs are acquired before the first computation, and each
frontloaded input is reusable after its actual last reader while outputs remain
at the block tail. Preserve the failed acquire/release receipts and the final
actual-AIE proof in late-input-closure-r1. Main-agent final frozen execution has
correct outputs in two eight-plan NPU sessions, including output-role reuse and
the new winner. This does not validate all fine performance ordering; the
coordinator reports incorrect/unresolved comparisons and session instability.
Algorithmic feedback, executable topology, observed benefit and predictive
ranking reliability remain separate claims. Preserve the production snapshots
bound by these evidence checkpoints; reproduce through new output directories.

Final v19 delivery is selected by companion
`results/taskflow/subgraph-execution-v19/DELIVERY.json` and its RESULTS,
CONCRETE-PLAN and REPRODUCE files. The final immutable Core SHA is
8173d56eb7436f6fd13bc956eda8a38985d9e3b51a854593895dbc9839b828e6;
AIR delivery SHA is
2aca1d38a50a35c2f0e389239d856fce4da111e17563dfba5fd2347c0262ba20.
Independent final eight-case receipts compare complete graphs/plans without
normalization; actual AIR bytes match and all eight actual final-AIE checks pass.
The output-role failure is closed by real entry DMA/wait plus explicit per-input
release. Preserve both old failing artifacts and the successful causal proof.

`fresh-generate-r2` is a successful clean-shell source/numerical-reference →
original C++ two-round search → eight actual compilations/checks → freeze,
with zero NPU calls during generation. Its archive exactly matches formal
`estimated-search-r3`:29 candidates,26 assumption-priced and3 unknown; final
29×7 sensitivity keeps the unique winner and exact first-round parent set.
The selected composition retains X in tasks1/2/3 and W in tasks5/6. Input
last-reader release permits predicted prefetch; do not call that a measured
hardware-overlap result.

Final clean-shell `final-clean-regressions-r2` passes Core40, combined57,
new/old AIR and preserved original Neura17/CGRA8 plus unfiltered lit35. Current
Core/AIR builds and the preserved original backend binary are explicitly
identified; do not describe the latter as freshly rebuilt. Its tools remain
unchanged during replay. The earlier r1 replay was invalidated when a live AIR
binary was relinked; the successful replay uses an immutable copy.

Two distinct frozen NPU studies remain separate. The initial seven-plan freeze
precedes all v19 performance,462 calls/924 arrays. The final eight-plan freeze
follows those observations with the same nine assumed numbers and no refit;
six source plans recur and two organizations are newly measured. Two final
sessions add528 calls/1056 arrays; cumulative990 calls/1980 full arrays pass
independent FP32/FP64 tolerance checks. All submissions are main-agent-only and
serialized through the existing /tmp/npu.lock. Host completion visibility,
physical DMA/core duration, actual DDR byte counts and real overlap are not
interchangeable observations.

The final composition improves paired baseline medians by280.853 and244.364us,
but expanded median pairwise ordering supports only10/16 then15/16. One first-
session resolved ordering is wrong and several small differences are unresolved;
means and cross-session medians show material variability. Keep estimate-driven
algorithm validation enabled, strict unknowns intact, and fine performance search
uncertified. Do not fit away this noisy session or claim the globally measured
best subgraph. The existing serialized native `placement` diagnostic tag is a
legacy NPU adapter label; actual resource/device/owner binding is target-derived
and this tag must never be interpreted as an architecture or mapping decision.
Broader noncontiguous residency, multi-core hardware and complex layouts remain
explicit backend boundaries. No automatic push or commit.

## v20: data-region reuse and compatible subgraph tiling

`GenericReuse.h/.cpp` discovers opportunities from actual exact rectangular
accesses, singleton versions, unique producers and consumer demands, including
immutable external inputs. It emits the existing ProgramMappingChoice plus tile
sizes; ProgramGraph remains the sole public execution graph. No workload name,
fixed task/storage index, Python planner or new scheduler participates.

Selected groups are contracted in the complete source dependency DAG. Convex
path closure supplies required intermediate tasks; a cyclic quotient rejects.
Topological expansion changes orderedTasks while preserving source indices,
body pointers, versions and data/order constraints. Tile-major lowering still
consumes contiguous execution groups, which may now be noncontiguous in source.
Internal whole-task pure-order barriers remain explicitly unsupported. External
pure-order dependencies must survive contraction and actual implementation
checking; never delete one to make a group legal.

Bounded consumer scopes and current/row-shrunk/column-shrunk compatible tiles
are jointly checked through native source transforms and the existing mapping
planner. Capacity rejection feeds tile selection. Scope round-robin preserves
consumer coverage before additional tile variants. Diagnostics retain the seed,
exact region/version, consumers, closure, memory, payload peak, boundary bytes,
delivery/configuration counts and rejection reasons. This finite enumeration is
not an exhaustive Pareto or globally optimal solver. Timing and parent selection
still use the complete checked ProgramGraph/ProgramTiming.

The existing retainedStorages selection also accepts globally immutable inputs
with at least two group readers. The first equal-rectangle import creates one
local copy for all readers; later duplicate deliveries disappear. Input ABI
allocations remain, inputs are never exported, and refill waits for every old
reader. Different rectangles reject. Explicit last-reader policy and actual
AIR/AIE lock/buffer checking substantiate this behavior; runner BO reuse is not
the implementation. One logical storage still belongs to at most one disjoint
selected residency scope. Broader layouts, reductions, in-place updates and
multi-core native realization do not inherit this certification.

`native_auto_candidates.algorithm` selects `v19-contiguous` (compatibility
default), `fixed-tiling`, `reuse-then-tiling` or `data-reuse-joint`. The last three
require mode=joint. max_joint_proposals/max_joint_tiles each range1..64 with
defaults8/4; tests use32/4,96 evaluations,beam2,two rounds. Reuse-first chooses
scope at the current tile or the first feasible capacity repair, then tunes
tiles in later rounds; it is not deliberately denied repaired scopes. Original
source hashing, action chains, cache, checkpoint replay and target binding are
retained. Strict mode keeps absolute costs unknown. All nine v19/v18 numeric
assumptions are unchanged; no device fitting occurs.

New evidence belongs in companion results/taskflow/data-reuse-v20 and its final
DELIVERY/RESULTS/REPRODUCE files. Two different17x136 arithmetic programs test
interleaved fanout/join plus an independent reuse chain, and shared-input/join.
Both source arrangements have equal semantic reuse opportunities. The original
two-round search selects and expands scored first-round parents; case1 honestly
selects the same best plan for joint and reuse-first. CPU search-plan signatures
are scoped to the same source and are not hardware measurements or compiled
binary-equivalence proofs. Actual compiled representatives have exact complete
graph/plan correspondence and independent actual AIR/AIE checks.

Current independent Core regression coverage is42 groups, including16 reuse and
25 shared-input checks. Preserve the first Core failure, which expected a now
legal dependency-convex noncontiguous group to reject; the subsequent change is
an acceptance-contract update, not a production bypass. Preserve all failed
drafts, interrupted CPU snapshots and previous evidence. Current Core/AIR builds
are new; the original Neura/CGRA optimizer remains a preserved historical binary.
All device work is main-agent-only, serial under /tmp/npu.lock with an outer
120-second timeout. No commit or push is authorized automatically.

## v21: controlled reuse and tiling decisions

The new native policies are `strong-reuse-first`, `strong-tile-first`,
`alternating`, `joint`, and the offline `catalogue` reference. Historical v20
policies remain available with their original decisions. Strong reuse-first
admits multiple disjoint scopes and deterministic feasible tile repairs before
freezing organization; strong tile-first freezes its tile before searching all
admitted organizations. Alternating uses reuse/tile/reuse/tile stages. Joint can
change one scope and tile together. All use existing source actions, complete
ProgramGraph, planner, scheduler, checker and structured search. Do not infer a
general affine or multilevel memory solver from this bounded implementation.

`GenericReuseOptions` builds a shared scope/tile catalogue from source-backed
reuse opportunities. Source singleton/full-consumer scopes, immutable-input
first-pair/full scopes and bounded disjoint combinations are explicit limits.
Tile origins include capacity boundaries, divisor/tail sizes, target alignment
preferences, current neighbors and halving. A proven distinct-buffer payload
lower bound can reject excessive occupancy; a small lower bound cannot certify
capacity. Every admitted configuration still passes the complete planner.
Pareto comparisons require identical groups and boundary exports and include
peak storage, bytes, deliveries, configurations and compute instances; the
reference retains dominated configurations.

All new policies/generators share the explicit optimizer-domain compute-instance
bound256. Exact static task-tile counts screen expensive plans and are checked
against generated phases. These exclusions are search complexity limits, never
backend illegality claims. Hard alignment/divisibility restrictions require
explicit tile_constraints; target alignment remains a preference. Protocolr3
records the pre-hardware CPU deadline pilot and this uniform amendment while
preserving earlier protocols and failures.

Raw telemetry counts generation attempts, internal planner/checker entries,
actual complete cost-provider calls, score queries, cache hits and elapsed time.
Independent checker repricing is charged. Cost budgets16/32/64 and wall budgets
5/15seconds include catalogue construction, with equal remaining round quotas.
Nonpreemptive work can overshoot a wall deadline and reports that amount.
Historical controls use unmatched score-query budgets and are context only.
Checkpoint revalidation has separate actual-work counters; the restored policy
ledger and current-process raw counters have different meanings.

Checked-outcome caches hold no MLIR pointers or reusable scored plans, bind
source semantics/target/model/choice/domain, and avoid repeated discovery probes.
Selected plans and resumed artifacts still undergo complete independent checks.
Atomic completed-round prefixes retain interrupted evidence and can repeat an
unfinished round. Do not erase failed builds, JSON reproductions, timeout pilots,
old numerical data, or immutable tool snapshots when reproducing this work.

Evidence belongs in companion `results/taskflow/data-reuse-v21`; its final
DELIVERY/RESULTS/REPRODUCE identify the accepted build and measurements. Questions
A (space), B (strategy within a common pool), C (budget) and D (tile generation)
must be answered separately. A catalogue optimum is only for its admitted finite
set and assumed model. Deduplicate final actual implementations across strategy
labels before hardware measurement. Main alone runs locked NPU sessions with
complete outputs and unchanged tolerance/model, including two fresh processes.
Current Core/AIR binaries and preserved historical Amoeba/CGRA binaries remain
distinct. No automatic commit or push.

The completed v21 delivery passes Core44, combined61, preserved Neura17/CGRA8
and lit35. All120 primary policy runs plus six finite references complete; an
extra four-round legacy series is resource-stopped after one completed run and
one incomplete checkpoint, with the original two-round compatibility retained.
The32 distinct compiled AIE implementations pass full searched/compiled plan and
graph equality; five batches in ten main-agent processes produce1224 calls and
2448 complete independently checked output arrays. FP32/FP64 maxima are
2.980232238769531e-7 and2.384185791015625e-7. New shapes improve paired timing
against halving-reference plans in interleaved and coupling; shared-input has no
stable improvement. Strong tile-first/alternating match or beat joint, and three
model-untied pairs have resolved wrong order in both sessions. Candidate geometry,
work budget and decision freezing are distinct contributions; neither pure joint
superiority nor fine hardware ranking is established. Final evidence lives in
the v21 DELIVERY/RESULTS/REPRODUCE files. Preserve every failure and frozen tool.


## v22: bounded native CP-SAT and physical independent NPU shards

Optional C++ OR-Tools v9.15.6755 composes single local reuse modes with common
tiling through the original native search. Exact cover, source/quotient DAG
legality and exclusive retained storage are encoded; full planning/checking
remains authoritative for capacity and complete cost. Signed local deltas are a
surrogate. A rechecked composition never replaces a cheaper already checked
local incumbent. Finite-domain optimality is not hardware-latency optimality.
The default ORBIT_ENABLE_ORTOOLS=OFF Core has no OR-Tools runtime dependency.

A separate physical entry partitions a proven independent outer axis, places
static local groups on NPU2 cores and models explicit compute/DMA intervals.
Actual x_loc/y_loc pins, source arithmetic/addresses, static allocations,
resource ownership, full output coverage and runtime lifecycle are checked on
final AIE. Child template graph identities stay in per-shard records. Current
certification covers direct shim/compute streams and one/two-core examples;
packet/memtile/inter-core dependencies fail closed. Assumed interval times do
not guarantee timed runtime dispatch or measured overlap.

Core44, combined61, preserved original Neura/CGRA and lit35 pass. Supplemental
checks pass42 finite oracle/resource cases, six native integration/budget cases
and15 actual multi-core AIE cases. Four hardware programs/seven variants pass
441 final complete calls and arrays against unchanged FP32/FP64 references in
three fresh processes per variant. One earlier paired speedup does not reproduce
in the final campaign; stable measured acceleration is not established.

Solver scaling is tracked with raw JSONL, model/solve/process/verification time,
bounds/gaps/status, branches/conflicts and RSS. Larger synthetic cases frequently
exhaust useful budgets. A supervisor saves independently checked baselines before
workers and preserves external-timeout status. Native discovery/pricing are
budgeted separately so32–128 task examples reach the solver: 8–128-task process
times are1.22–4.78 seconds, with5.7–17.9ms solving. Domains at16+ are truncated;
these are bounded samples, not an asymptotic guarantee. Planner calls remain
nonpreemptive and overshoot is explicit. CP-SAT process quotas do not restore the
v21 staged-policy budget ledger across resume.

See docs/taskflow-cpsat-v22.md and companion taskflow/docs/taskflow-cpsat-v22-validation.md.
Production code and finite compiler contracts remain here; workload generation,
performance campaigns, plots, hardware execution and reviewable/raw evidence
remain in orbit-artifact. Preserve all failed stack/routing/property/timeout
pilots and all frozen original binaries. NPU sessions remain main-only under
/tmp/npu.lock with process deadlines. This checkpoint alone does not authorize future commits
or pushes.

## v23: public ProgramGraph joint constraint realization

ProgramJointDomain/Solver/Plan now build one source-bound public graph model
for finite partition variants, connected local grouping, copy boundaries,
compute/SRAM placement, transfer links, configurations and resource intervals.
No complete-plan price table or separate placement solve is used by this entry.
The original v22 local `cp-sat` mode remains historical. Native AIR search selects
this entry with native_auto_candidates={mode:joint,algorithm:cp-sat-program};
legacy proposal fields and staged resume are rejected for this fresh finite solve.

The checked public baseline is saved before CP-SAT; AIR mode also checks its
source/copy/static-buffer realization first. Model limits and deadlines retain
that baseline with truthful UNKNOWN/RESOURCE_LIMIT. Final AIE checking consumes
frozen decisions without solving, proves complete caller ABI forwarding and
cross-owner/symbol isolation before per-core projection, and checks actual
selected computations, copies, addresses, locks, placement and configurations.
Each configuration has a separate synchronous AIR launch so installed channel
conversion preserves the explicit host BO round trip.

Current executable scope is static scalar pointwise identity-grid programs,
independent axis-zero partitions, equal storage grids, unique writers and direct
host/core DMA. Connected group copies and explicit host-mediated inter-group
communication are supported; GEMM/reductions, layout search, packet/memtile and
arbitrary cross-core SRAM residency are outside this joint path. An assumed
resource timeline guides choices; timed runtime dispatch and hardware-latency
optimality are not claimed. Final toolchain/AIE checking remains mandatory.

Validation: Core44 and combined61 pass; optional solver-on/off public contracts
pass23 each; final AIE contracts pass39 cases across three implementations.
Two source graphs (shared-input FP32 arithmetic and a17x64 chain) yield five
actual NPU implementations, including changed grouping, two cores, a9-row/tail
partition and two configurations with a host temporary. Two fresh processes per
implementation pass80 calls/128 complete arrays against FP32 and FP64 references
at unchanged1e-5 tolerances; maximum absolute error is1.1920928955078125e-7.

Sixteen serialized scaling runs vary2–128 tasks,1/2/4 compute resources and one
or three partition variants. The two-core single-variant model reaches96125
variables/220307 constraints at64 tasks;32/64 exhaust useful1-second solves,
while128 stops at the100000-variable guard. All retain checked baselines. Whole
process time can exceed the soft budget (about1.30seconds at64); finite-domain
completeness does not mean unrestricted program-space completeness. Raw RSS is
process high-water telemetry, not isolated solver allocations.

See docs/taskflow-program-joint-v23.md and companion
 taskflow/docs/taskflow-program-joint-v23-validation.md. Artifact scripts own
experiments, full output arrays and measurements. Frozen prior tools/failures
remain preserved. NPU work stays main-only under/tmp/npu.lock with deadlines.
This checkpoint does not authorize future commits or pushes.
