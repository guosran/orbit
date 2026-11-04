# Target-driven whole-program search and frozen NPU prediction

This stage extends ORBIT's program candidate: computation and task dependencies,
resource placement, schedule, implementation and data-copy organization remain
joint choices. Amoeba and direct AIR supply target constraints and implementation
capabilities. The public task/dependency/data graph does not encode AIR channel
or FIFO instructions.

The common memory architecture reuses `StaticResource`, `StaticMemory` and
`StaticTransfer`, adds ownership domains and precise compute access, and supports
directed reachability with unknown costs. Public copies have a logical region,
version, layout, memory location, creator, users and release events. Multiple
dependencies can share a copy and its single creation transfer. A synthetic
shared-SRAM/two-local-SRAM target validates multi-hop transfer and capacity
accounting. Phase-local transit imports expire at task boundaries; retained
read-only staging transfers 192 B versus 256 B for the same local placements.
These CPU examples do not claim arbitrary NPU cross-stage sharing.

The NPU target binds BO memory, a 24,576 B local tensor payload pool, compute,
DMA and host-submission resources, and directed transfer links. The stack remains
separate. Ownership is distinct from accessibility; a host worker can access BOs
directly, whereas the NPU compute resource accesses the local pool. A logical
pool is not a claim about the capacity of one physical SRAM bank. AIR realizes
the selected plan using memory spaces, buffers and DMA/channel tokens. See the
[architecture contract](taskflow-memory-architecture-v10.md).

One native program call is an explicit execution region with one host submission
shared by its input edges. Compute/transfer events and device stages remain
distinct. The fitted label is the complete prepared host call, including staging,
submit, combined XRT wait and output return. Context creation, file loading,
reference checking and evidence serialization are measured outside that label.
The composite prediction owns those included costs once; it is not added to a
second DMA or task estimate. This finite single-call provider does not price a
fissioned multi-call program or FFN outside its family. Existing CGRA ML and its
II/scheduler contract are unchanged.

Original C++ search now generates tiles from analyzed extents, target alignment
and capacity, implementation options from target capabilities, and retention
actions from proved access reuse. Structural changes rerun semantic analysis,
copy/communication construction and legality checks. Candidate identity includes
the actual transformed IR; evaluation separately binds target/model conditions.
Default scalar spelling is canonicalized so it cannot create false novelty.
Unknown costs remain null and use explicit exploration slots. The automatic
search report preserves parents, action chains, failures and checkpoints.

Before timing, the experiment registered 12 training cells, 6 holdouts and a
capacity rejection. Training design has rank 4 and scaled condition 10.0643.
The 12 cells vary shape, tile, scalar/mixed8 and phase-local/retained B panels.
Three seeds with two warmups and five hot calls give 252 independently checked
training outputs and 180 labels. Parameters and native predictions were frozen
before any holdout execution. Holdouts include an unseen M/K tile combination
and simultaneous M/N/K tails, with both implementations. Their 126 outputs and
90 hot calls pass unchanged finite FP32 `atol=rtol=1e-5` checks.

| Frozen provider | Holdout MAPE | Maximum error |
| --- | ---: | ---: |
| Implementation + stage, four terms | 1.6304% | 5.4624% |
| Implementation only, three terms | 2.6846% | 8.6393% |
| Previous uncalibrated theoretical coefficients | 96.4006% | 96.6354% |

The four-term provider selects the measured-median best implementation in both
holdout program groups, with zero descriptive regret within those finite groups.
It reverses one scalar phase/panel pair. Phase/panel differences are small and
overlap paired variation; they do not establish a residency speedup. Three-term
and theoretical ties retain regret ranges instead of an arbitrary favorable tie
break. A coefficient is a composite statistical association, not a causal
measurement of stage setup or scalar instructions. Tail conversion has 15 stages
versus 2/3/5 in training and remains a narrowly tested extrapolation. Full details
and model ownership are in the [cost contract](taskflow-direct-air-cost-v10.md).

With the frozen model, joint search evaluates 224 candidates and ranks 30
calibrated candidates. The tiling-only and implementation/residency controls use
the same ceiling (512 evaluations, six rounds, beam 16) but exhaust after 56 and
6 evaluations respectively. They are equal limits, not equal actual work. Model
ties preserve the native stable order. The first joint result is a new
4/32/64 mixed8 phase-local plan; its action chain includes implementation and
three tiling changes. Its complete tensors occupy 65,536 B including the system
intermediate BO, its maximum local tensor payload is 9,728 B, and its planned
DMA is 319,488 B. These bytes describe generated logical transfers, not measured
DDR traffic.

That new candidate, both restricted-mode winners and one explicit unpriced
exploration (`[8,8,32]` scalar) are compiled and executed in one interleaved
three-seed comparison. All 84 outputs pass; their hot medians are 17.068420,
18.627799, 17.269644 and 18.462234 ms. The three calibrated predictions have
0.5404% MAPE. Joint and implementation/residency predictions tie; paired
control-minus-joint timing ranges from -0.803499 to +0.605125 ms, so the observed
median difference does not establish a joint-search gain. Unknown exploration
keeps its null predicted cost and a separately bound measured result.

A representative full M5/D2048/I11008 FFN lifts the chosen tile choice and
compares it against `[8,64,64]`, both mixed8 phase-local. All 12 outputs pass
with maximum absolute error 2.42144e-8. Three hot medians are 21.214368 s for
the baseline and 21.418708 s for the lifted plan: no observed speedup. Local
payload decreases 18,944 to 9,728 B while planned DMA increases 292,809,728 to
584,477,696 B and device stages increase four to eight. This is a concrete
capacity/communication/organization tradeoff, not a small-GEMM prediction for
FFN. Historical M128 execution remains attributed to v7/v9; this stage does
not repeat that long run or claim an improved M128 result.

Real compilation exposed a tail DMA-to-channel token-scope failure. The checked
plan now explicitly records module-wide asynchronous herd legalization when
the installed conversion acts on the entire module. It retains the selected
tiles, data movement and FP32 K order. Both scalar and mixed8 tail candidates
compile and execute correctly. A separate artifact checker incorrectly included
trailing host IR in the last device's buffer scope; balanced device extraction
fixes that parser while preserving conflicting-buffer, port, payload and
provenance negatives. Original failing artifacts and scripts remain preserved.

Evidence, numerical references, profiling, model fitting and hardware
orchestration live in the companion repository. The clean-shell entry is:

```sh
LLVM_BUILD=/home/shiran/project/llvm-project/build \
  ../orbit-artifact/taskflow/scripts/run_taskflow_implementation_search_v10.sh \
  all /absolute/path/to/NEW_OUTPUT_DIRECTORY
```

The entry requires the installed AIR/AIE/XRT environment and uses existing
execution tools, not a new runtime. Use `cpu` for compiler/protocol validation.
All output paths must be new. Detailed raw results and final execution receipts
are documented in the companion v10 report; legacy measurements retain their
own frozen versions and are not new execution claims.
