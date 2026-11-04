# ORBIT: Joint Optimization of Whole Programs

## Project direction

ORBIT builds on AMOEBA/Neura and optimizes **an entire program's task graph and execution plan**. Its core is a joint search over graph structure, resource allocation, temporal scheduling, and data organization. Starting from the same program, it materializes alternative task partitions, parallelism, resource shapes, and execution arrangements, evaluates whole-program cost, and expands promising candidates.

The project is moving this search onto a common Taskflow abstraction. Amoeba and AMD XDNA NPU are backends that consume search results and provide target constraints, implementation capabilities, and cost feedback. Communication modeling and heterogeneous execution extend the original search space: ORBIT jointly decides how computation is partitioned, where it executes, where data resides, and when data moves.

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

Implementation coverage varies by path. The original Neura path contains established shape, structural, and scheduling capabilities. The common structured path supports M/N/K tiling and restricted fusion/fission. Native Taskflow/AIR supports an evolving set of loop transformations and tile configurations. Action generators and target capabilities determine the available choices.

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

These choices enter static mapping and search incrementally. Current development focuses on static plans and code generation, using existing backend execution facilities.

Data reuse is a whole-program concern. Weights, input tiles, intermediate activations, and reduction partials share the same concepts: data regions and versions, storage copies, consumers, and validity intervals. Structural transformations and scheduling create reuse opportunities; storage planning selects retention; communication modeling evaluates the consequences. Retaining a weight tile across several M tiles may save weight traffic while increasing partial-output storage or transfers.

For NPU execution, the common storage plan determines on-chip tile residency. Reusing XRT BOs or execution objects across invocations belongs to execution preparation. The backend realizes storage placements in system memory, NPU-accessible buffers, and AIE local memory.

## Compute nodes, dependency edges, and data copies

The proposed common representation extends the task graph with a candidate-owned table of data copies:

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

The immediate extension is a neighborhood action choosing between release-after-use and retention across multiple consumers, combined with existing tiling. Later actions cover storage location, sharing/replication, streaming/materialization, and buffer organization. Compound actions or beam diversity help discover sequences such as reducing tile size to make longer residency feasible.

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

The `direct-air-v6` baseline extended full-graph residency to static tiled execution. With the 24 KiB payload policy budget unchanged, oversized GEMM, non-divisible M/N/K tiling, add→scale, and full M5/D2048/I11008 SwiGLU passed numerical validation on the NPU. In that baseline, M128 compiled and was submitted but timed out; diagnosis and numerical validation are follow-up work.

Current work consolidates tiling, residency, lifetimes, and transfers into the common mapping representation and connects them to the original search. Experiments focus on coupled choices, such as tile size and retention scope jointly changing traffic, peak capacity, and whole-program latency. FFN testing and M128 diagnosis provide concrete workloads for this development.

The next common-layer milestone is an executable data-copy representation, shared versus separate-copy examples, residency actions combined with tiling, and explicit compute/communication cost boundaries. CPU planning tests can progress while hardware experiments run. Separate build/output artifacts preserve the identity of running experiments; hardware results remain associated with the exact compiler, plan, and configuration used for execution.

Subsequent experiments compare structural search, mapping search, staged optimization, and joint search. Multiple backends test the common representation and target adaptation; NPU execution validates plan realization and actual costs.

## Related-work perspective

Comparisons should identify program inputs, optimization objects, adjustable decisions, and evaluation objectives. ORBIT focuses on joint optimization of whole-program graph structure and execution plans. Its overlap with dataflow programming and mapping systems such as Dato includes task merging, resource mapping, and communication organization. Useful ideas include separating logical instances from physical resources and making communication contracts and resource constraints explicit.

User-specified versus compiler-inferred information is one comparison axis. The central research question is which whole-program opportunities joint search discovers, and how it improves on fixed-structure or staged decisions. Related systems' own automatic mapping and search belong in that comparison.

The task-graph representation supports the implementation; the research opportunity lies in the coupling between structural changes and data realization. Strong examples show that a structural change alters reuse opportunities, that storage/communication choices change the preferred structure, and that joint search finds better executable plans than optimizing each independently.

## Entry points

- [Repository overview](README.md): original ORBIT background, search, and build.
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
