# Independent static mapping validation

> 历史 v3 说明：Python 规划器/检查器现已退出活动代码，旧差分仅为历史证据。默认生产与检查均为 C++/MLIR；复现入口为 `bash scripts/run_taskflow_static_cpp.sh NEW_OUTPUT_DIRECTORY`。v5 审计保留了退出前代码快照和全部旧结果。

The independent checker is `python/orbit_mapping/checker.py`. It imports only
Python standard-library modules, never the mapper, scheduler or communication
builder. The public Taskflow program and versioned target configuration are its
authorities; emitted plan metadata is checked against independently derived
facts. This is a static legality check, not numerical execution or hardware
validation. The existing Core remains responsible for authenticating the
materialized Taskflow body against its restricted public program.

## Checked invariants

The checker independently expands ordered task bodies and dependencies, derives
matrix/activation/accumulator access rectangles, partitions storage into atomic
rectangles, and propagates values through actual compute/copy/conversion events.
Value identities come from input storage or semantic operation identity, not
from trusting the planner's producer/version claims. It checks:

- Every operation appears exactly once, including fills and every sequential K
  partial. Accumulating matmul consumes the preceding accumulator version.
- Producer, transfer, conversion and consumer dependencies are ordered. Physical
  WAR/WAW hazards require ordering too, without inventing data transfers.
- Required regions exist in accessible allocations with the implementation's
  layout/dtype. Shared-memory cross-device visibility needs explicit sync for
  the current value version, including the final output consumer.
- Each implementation supports the operation, shape and numerical acceptance
  contract. Full-K/preceding-zero restrictions for the existing NPU overwrite
  implementation are independently checked; partial-K NPU placement is excluded.
- Compute, communication and sync costs are recomputed from configured formulas,
  byte counts and provenance. Missing, negative or unknown costs are rejected.
- Resource demand matches the implementation/path. Integral concurrency demand
  must fit capacities; an implementation needs an execution resource on its
  device. Additional declared shared-resource reservations are permitted.
- Physical buffer sizes, graph-wide data lifetimes and event-local temporary
  lifetimes are recomputed. Layout copies and copy/conversion scratch count
  toward capacity. Zero-duration operations do not make storage free.
- The final output contains the latest semantic version in the required memory
  and has the required visibility. Both makespan fields and all cost/memory
  aggregates match the actual schedule.
- Program/configuration/plan identities match their content; assignment metadata
  matches actual compute implementation selection.

The supported storage policy conservatively reserves each whole logical buffer
at each physical memory/layout for the entire single graph. This may reject a
plan a future lifetime-aware allocator could fit; it never assumes unimplemented
reuse or omits simultaneously live copies. Aliasing, dynamic storage, arbitrary
control flow and runtime state are outside this contract.

## Synthetic fixtures and adversarial plans

`test/TaskflowMapping/check.py` passes **22 positive assertions and 28 illegal-plan
checks**. Evidence is under `build/static-mapping-v3/validation/synthetic`.
Every fixture saves its public program, configuration and complete mapping
result, including rejected alternatives. These costs are explicitly synthetic,
not CPU/NPU hardware performance evidence.

The required eight scenarios cover communication reversing the individually
fastest device; same-location reuse; multiple consumers sharing one upload and
requiring separate outputs; pure ordering edges; fast assignments rejected by
memory capacity; compute/link contention; layout conversion cost and scratch;
and missing implementation/cost rejection. Additional fixtures cover shared
memory visibility, missing paths/synchronization costs, zero-duration capacity,
and sequential K/full-K implementation eligibility.

For the two-op chain, all four placement assignments are exhaustively compared
with an independent closed-form oracle: sum of selected compute durations plus
each required boundary crossing's startup and bytes/bandwidth cost. This proves
the reported assignment optimum for that serial fixture only; it does not claim
the general list scheduler is globally optimal.

Adversarial plans are **re-signed after mutation** so identity checks alone
cannot satisfy the negative tests. They omit fills/partials/accumulator reads,
forge regions, versions, resources, allocation bytes/lifetimes, assignment,
provenance, transfer bytes, costs and totals, drop scratch and sync events,
introduce overlap, and violate precision/implementation restrictions.

```bash
cd /home/shiran/project/orbit
python3 test/TaskflowMapping/check.py \
  --output build/static-mapping-v3-reproduce/validation/synthetic
```

## Existing path regression

After the original ORBIT structured-search changes were rebuilt, the following
passed using the existing binaries, without accessing NPU hardware:

| Check | Result |
|---|---:|
| Independent Core materialization/tiling/K/fusion/witness/negative tests | PASS |
| Existing Amoeba lit, including mapper/scheduler and verifier cases | 35/35 |
| Existing structured-search protocol, interruption and recovery | 16/16 |
| Existing evaluator cache corruption/invalidation | 6/6 |

Logs live in `build/static-mapping-v3/validation/regressions`. The legacy
Amoeba checks retain cycle estimates and CPU replay as their original evidence
types; this round does not execute a physical CGRA or alter the NPU runtime.
After fixing deterministic traversal in the original search checkpoint binding,
all 16 protocol checks were rerun in `search-contract-after-binding-fix` and
passed. `core-isolation.json` records the rebuilt Core's link command, dynamic
dependencies and symbol audit, with no Neura/AIR/XRT backend linkage.

```bash
cd /home/shiran/project/orbit
python3 test/TaskflowCore/check.py --tool build-core/tools/taskflow-opt/taskflow-opt
PATH=/home/shiran/project/llvm-project/build/bin:$PATH \
  /home/shiran/project/llvm-project/build/bin/llvm-lit -sv -j2 build-amoeba/test
build-amoeba/.venv/bin/python test/TaskflowValidation/search_contract.py \
  --tool build-core/tools/taskflow-opt/taskflow-opt \
  --output build/static-mapping-v3-reproduce/validation/search-protocol
build-amoeba/.venv/bin/python test/TaskflowValidation/evaluation_cache.py \
  --output build/static-mapping-v3-reproduce/validation/cache
```

Use a new output directory for each protocol run: its interruption marker and
intentionally modified evaluator/configuration are part of the test evidence.
