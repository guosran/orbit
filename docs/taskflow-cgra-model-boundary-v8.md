# Existing CGRA model boundary audit

This is a read-only audit of the existing CGRA cost path. No model, training data, checkpoint, feature contract, or inference code was changed, and no model was trained. New development is isolated in `/home/shiran/project/orbit-data-v8`; original v7 NPU execution uses the frozen snapshot under `/home/shiran/project/aie-toy/results/orbit-direct-air-v7/frozen-original-r1`.

The existing model predicts **compiled mapper initiation interval (II), in cycles**, for a route-expanded task DFG and concrete mapper rectangle. It does not predict end-to-end program wall time, memory-ready latency, or successful mapping probability. Whole-program cost must compose that conditional compute estimate with independently checked data movement, availability, and resource supply. The current direct-model metadata explicitly sets `supports_whole_program_latency_or_throughput_claim=false` and marks the bundle as a candidate pending benchmark-overlap audit.

## Call chain and identities

`PredictAnalyticalTaskCostCatalogPass.cpp` constructs the current mapper-visible task bodies, lowers them to route-expanded Neura DFGs, derives canonical structural text, computes RecMII/ResMII for the concrete rectangle, and extracts features in C++. An optional feature query file is an oracle/debug input that must agree with the current body; it is not a task-name lookup. The pass calls `FormalMax4MLPEnsemble` or `PerCgra2x2DirectEnsemble`, then emits an `amoeba-task-shape-cost` catalogue. `AnalyticalTaskCostCatalog` checks the catalogue and supplies its II/startup pair to the scheduler.

The original selected-shape adapter is a separate entry point: `OriginalAmoebaProfileCostAdapter.cpp` checks successful profile records, current/source iteration-domain coverage, mapper-body witnesses, selected shape, and source trips before invoking the same feature/inference contracts. It does not search shapes by itself.

The persistent ML cache is keyed by canonical mapper body text and concrete rows/columns. Task name and trip count are deliberately absent. Architecture YAML, ensemble text and checkpoint texts are bound to the cache resources verbatim, preventing a changed architecture/model bundle from silently reusing an old prediction. The catalogue-level task/shape cache also leaves trip count outside its key; callers must supply the independently validated current trip count.

Sources: [predictor body reconstruction and invocation](../lib/Backend/Neura/Orchestration/JointScheduling/PredictAnalyticalTaskCostCatalogPass.cpp#L1532), [model/cache interfaces](../lib/Backend/Neura/Orchestration/JointScheduling/AnalyticalMLPInference.h#L90), [catalogue trip-count boundary](../lib/Backend/Neura/Orchestration/JointScheduling/AnalyticalTaskCostCatalog.cpp#L609), [original profile adapter](../lib/Backend/Neura/Orchestration/JointScheduling/OriginalAmoebaProfileCostAdapter.cpp#L931).

## Existing labels and feature domains

The inspected local bundles are `/home/shiran/project/aie-toy/vendor/orbit-artifact/reference/input0-neighborhood/models/ensemble.json` and its `per-cgra-2x2/ensemble.json`. Their recorded labels and counts below are bundle provenance, not new training or independently reproduced training metrics.

| Contract | Existing target and domain | Representation |
|---|---|---|
| `formal-max4-nohash-v2` | `compiled_ii_for_static_route_expanded_task_and_mapper_shape`; `success_only_censored_never_imputed`; exploratory, not production-ready | 156 features; three equal-weight baseline/large-operation/ranking checkpoints; original per-CGRA geometry is 4×4 PEs |
| `orbit-cgra-ii-per-cgra-2x2-direct-4member-v1` | `fresh_native_compiled_ii_for_given_dfg_and_2x2_cgra_shape`; metadata reports 2569 successful and 15 censored native queries; old 4×4 labels were not reused | 148 declared features, 61 selected inputs; four direct members with seeds17/41/113/239; 61→64→32→1 networks, GELU and softplus residual; arithmetic-mean II |

The direct bundle records source commit `3ade31806cb4c92e31888109f7c42b8a77e4cbce` on `orbit-2x2-predictor`. Its ordinary runtime contract requires exact training/runtime architecture text, including `ctrl_mem_items: 20`. The existing diagnostic 23 ceiling permits only the explicitly checked 20→23 control-memory substitution, keeps training feature normalization at20, and emits diagnostic metadata. It is not a new trained model.

For the 2×2-per-CGRA protocol, allowed mapper rectangles are 2×2, 2×4, 4×2, 2×6, 6×2, 2×8, 8×2 and 4×4: up to four CGRAs per task, with orientation preserved. The older formal max-four predictor checks its own 4×4-based rectangle protocol. Neither model permits arbitrary geometry to be treated as an equivalent count.

Both feature families include operation counts (including loads/stores/indexed memory operations), node flags, graph depth/width, recurrence information, movement/route statistics, normalized rows/columns/tile count/aspect ratio/link counts, shape indicators, RecMII/ResMII/lower bound and occupancy/collision pressures. Thus CGRA count and arrangement affect the query through a concrete PE rectangle, analytical facts and shape-dependent graph pressure. There is no feature for full tensor bytes, a working-set/L1 capacity budget, data version residency, DRAM bandwidth, cache hit rate, bank assignment, or a caller's local-memory supply schedule. Trip count is not a learned feature.

Sources: [contract constants](../lib/Backend/Neura/Orchestration/JointScheduling/AnalyticalMLPInference.h#L30), [feature names](../lib/Backend/Neura/Orchestration/JointScheduling/MapperFeatureExtractor.cpp#L270), [shape/MII context values](../lib/Backend/Neura/Orchestration/JointScheduling/MapperFeatureExtractor.cpp#L810), [direct network inference](../lib/Backend/Neura/Orchestration/JointScheduling/AnalyticalMLPInference.cpp#L1157).

## Timing and local-memory supply boundary

`startup_cycles` is currently derived from route-expanded graph structure, not learned from an end-to-end runtime measurement. The implementation takes a unit-latency forward semantic depth, excludes feedback edges from that depth, and gives movement/reserve/yield node types zero additional depth. Calling the catalogue an II/startup predictor must not be read as evidence that startup is a separately trained hardware latency.

The scheduler converts the pair and the checked mapper-firing count into

`duration_cycles = ceil(startup_cycles + predicted_ii * (trip_count - 1))`.

The II prediction is constrained above `max(RecMII,ResMII)` and within the model's accepted ceiling. Importantly, the current `calculateResMii` is `ceil(materialized_operation_count / architecture_tile_count)`: it does not derive a bank/port-specific memory service bound. Memory operations can affect the DFG, mapper labels, graph features and materialized operation count, but this does not establish a general local-memory bandwidth guarantee.

The existing `TaskCommunicationModel` independently accounts for inter-task payload bits, route latency/bottleneck bandwidth, local-channel bandwidth and reserved communication intervals. For same-coordinate communication it charges `ceil(payload_bits / local_bandwidth_bits_per_cycle)` and reserves the local interval. It models task-boundary availability; it does not prove every internal load/store can be supplied on each mapper cycle, and it is not a measured DDR model.

The valid composition condition is therefore explicit: required data versions must be available where the task starts, its local accesses must satisfy the fixed mapper architecture's service assumptions, and additional movement/supply constraints must be checked outside this unchanged II model. This condition is inferred from the actual feature/label/code boundary; the current model does not certify it automatically.

Sources: [startup derivation](../lib/Backend/Neura/Orchestration/JointScheduling/MapperCostAnalysis.cpp#L54), [duration conversion](../lib/Backend/Neura/Orchestration/JointScheduling/ScoreExactJointTaskCandidatesPass.cpp#L1583), [current ResMII](../thirdparty/neura/lib/NeuraDialect/Mapping/mapping_util.cpp#L269), [local communication supply](../lib/Backend/Neura/Orchestration/JointScheduling/TaskCommunicationModel.cpp#L597).

## Suggested integration boundary for the new public plan

Keep the original model and its inference path intact. A C++ cost adapter should return the II/startup pair together with units, exact model/architecture/feature identities, concrete mapper rectangle, checked firing count, mapper-body witness, support status and the conditional compute boundary. The adapter should consume a current, backend-produced mapper-visible DFG rather than a graph-name or tensor-shape shortcut.

The public plan should separately supply data-version placement/residency, byte ranges, capacity/lifetime checks, movement availability and local-memory supply assumptions. A missing supply contract should yield unknown/unsupported end-to-end cost instead of zero communication or a claim that compiled II includes all memory stalls. If a local service bound is derived, expose it separately; changing II to `max(predicted_ii, supply_bound)` would be a new analytical composition policy, not an unchanged ML prediction, and must retain that provenance.

Whole-program scoring should compose conditional compute events with movement/resource dependencies under one timing contract. It must avoid adding external movement twice when the chosen catalogue/mapper contract already includes it, and must distinguish predicted scheduler cycles from measured NPU host wall time. No C++ interface or cost code was changed by this audit; these are proposals for root's Search/evaluation integration.

## Minimal adapter insertion in the real C++ path

The existing chain is concrete, not a proposed replacement formula:

| Boundary | Current C++ owner and value |
|---|---|
| Existing inference result | `FormalMax4MLPEnsemble::predict` / `PerCgra2x2DirectEnsemble::predict` in `AnalyticalMLPInference.cpp`; the predictor pass serializes it into a checked catalogue |
| Validated task/shape lookup | `TaskShapeCostCache::get(const TaskShapeChoice&, std::string&)` in `AnalyticalTaskCostCatalog.cpp`; returns `TaskShapeCost` from `AnalyticalTaskCostCatalog.h` (II, startup, support/domain fields) |
| Task execution duration | `ScoreAnalyticalTaskCandidatesPass.cpp:600–650` already computes `schedulerDurations`; the exact scorer's existing `predictDuration` at `ScoreExactJointTaskCandidatesPass.cpp:1583` performs its checked integer conversion |
| Existing graph scheduling | The analytical scorer writes `cgra_count`, `cgra_shape`, fixed orientation and `est_latency`, calls `TaskScheduler::schedule` with the existing strategy/communication model, then reads `getScheduleMakespan` |
| Scheduler consumption | `TaskNode::getDuration` in `orchestration_utils.cpp:319` reads fixed-orientation `est_latency`, otherwise profile duration, otherwise defaults to1; this default must not be used as an implicit successful ML estimate in a new public cost adapter |

The smallest insertion is **at the already validated `scheduledDuration` boundary in the Neura scorer/provider**, before graph scheduling. A Neura-owned adapter can populate a Core-owned plain `ComputeEstimate` value with that existing duration, units/cycle-domain, source/model/architecture/body/shape/trip identities, support status, and conditional charging scope. That reuses the actual original inference→lookup→duration path rather than copying the II expression into a new evaluator. If a reusable query API is needed, factor the existing checked duration helper into a Neura header/implementation and have both the old scorer and the provider call it, preserving that scorer's numerical conversion policy.

Core declares the value and provider interface without including `TaskShapeCost`, Neura dialects, mapper types or model headers. The implementation under `Backend/Neura` includes Core's contract and translates the existing checked Neura result: **Neura→Core dependency only**. The new Core scheduler consumes the supplied estimate and explicit movement events. A synthetic test planner supplies the same contract with a visibly different source such as `synthetic-test`; it cannot label an arbitrary coefficient as the existing CGRA ML result. Unsupported lookup, unknown service assumptions, missing identities or incompatible cycle/time domains remain explicit failures/unknowns instead of falling through to scheduler duration1.

No adapter implementation was changed by this audit. These concrete insertion points and dependency constraints were sent to root for the requested minimal real integration.


## Implemented value adapter and regression boundary

The new tree now implements the minimal adapter inside `ScoreAnalyticalTaskCandidatesPass`: it projects the existing checked `scheduledCycles` result into the Core-owned `MappingComputeEstimate` value and publishes `compute_contract` in score task records. The original model, inference/catalogue lookup, II/startup policy and graph-scheduler duration remain the source of that value. Core has no Neura/model dependency. The contract declares cycles, task/mapper shape/count, catalogue SHA/provenance, included and excluded charging scopes, and explicitly leaves `supply_conditions_verified=false`. It therefore does not establish an unconditional end-to-end time estimate.

`test/TaskflowMapping/compute_contract_v8.py --tool /absolute/new/mlir-amoeba-opt --output /absolute/new/output` runs the existing all-unit fixture through real candidate enumeration, original-model prediction and production scoring. It requires the original model files and saved pre-adapter baseline; missing artifacts fail. It checks exact production scheduled duration, II/startup/trips, shape/count, catalogue SHA, units and excluded communication, then removes only `compute_contract` and requires unchanged task costs, schedules, scores and shortlist. This is CPU-only and does not train or generate a model. Execution status is recorded separately after the stable integrated tool is available.

The resource contract distinguishes CGRA blocks from mapper PEs: `resource_count` and `resource_shape` use `cgra_block`, while `implementation_shape` uses `processing_element`. The catalogue SHA is computed from the exact buffer parsed by `TaskShapeCostCache`, avoiding a second-file-read identity race. The regression asserts both shape domains and their units.


The checked-in `test/TaskflowMapping/fixtures/compute-contract-baseline-v8.jsonl` is the actual pre-adapter output (eight complete score rows plus original header/footer). Its companion provenance JSON saves the original executable SHA, exact three pass commands, catalogue/candidate/log hashes and existing model hashes. The regression defaults to this delivered golden and checks its provenance SHA; it does not depend on an unshipped build directory or recompute expected scores. Original model checkpoint files remain an external dependency at the documented `aie-toy/vendor/orbit-artifact/reference/input0-neighborhood/models` location, overridable with `--models`; missing files fail explicitly.

The implemented provider is currently consumed only by the Analytical scorer to publish the public value from its existing scheduled duration. Exact/profile scoring paths have not been adapted. The audit does not claim that all schedulers or Neura callers now consume the new public contract.


Inter-task movement exclusion is now machine-checkable: the provider sets `MappingComputeTransportScope::ExcludesInterTaskTransfers`, serialized as `transfer_accounting="excludes_inter_task_transfers"`. The public combiner rejects the default unspecified scope rather than trusting free-text cost descriptions. This prevents adding explicit movement events to an estimate whose transfer charging is unknown. The original Neura graph scheduler still charges graph communication separately; the analytical compute value excludes that inter-task charge. The CPU regression asserts this serialized scope in every actual task contract.

## 本轮实际回归

`build/data-organization-v8/acceptance-final-cpu-r1/cgra-compute-contract/summary.json` 记录真实原模型→catalogue→Analytical scorer 的 8 个合约全部通过，移除新增字段后与已交付 pre-adapter golden 的分数、schedule 和 shortlist 完全一致。该既有 fixture 的 trip_count=1，测试验证适配与行为保留，不是多 trip 的模型精度评估。模型来源/权重/架构和 golden 的哈希一同保存；Core 的独立链接回归也通过。
