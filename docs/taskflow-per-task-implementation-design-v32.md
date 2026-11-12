# Native per-task implementation binding: v32 implementation

The implementation now accepts strict per-task native kind/lanes bindings in
public joint search, using the existing plan, graph, and AIR representations.
Module-wide requests retain their legacy fallback behavior. No executor, AIR
runtime, frozen model, or historical measurement was changed by this extension.
The detailed design below records the adapter and its proof boundary.

## Existing contract and gap

`GenericTaskImplementation` in `include/Taskflow/GenericMapping.h` already holds
`task`, `kind`, `parallelLoop`, and `lanes`. `planGenericMapping` in
`lib/Taskflow/GenericMapping.cpp` reads `orbit.implementation` and
`orbit.vector_lanes` inline, then fills `taskImplementations`. There is no
existing shared `readNativeImplementation` function. The module request is a
catalogue policy: matrix or vector modes apply to eligible tasks, with scalar
fallback for other tasks. Matrix and mixed-vector module requests must match at
least one task. Preserve that legacy behavior when no local overrides exist.

The independent checker already validates capability membership, per-task
coverage, vector loop identity, lane restrictions, matrix source contracts,
and per-phase matrix/vector eligibility. It now independently re-reads the
local source request and compares it against the chosen plan; planner-only
support would have left a forged-plan gap.

## Minimal representation and API

Keep selection metadata on the module, outside the function body:

```mlir
orbit.task_implementations = [
  {task = 0 : i64, kind = "scalar", lanes = 1 : i64},
  {task = 2 : i64, kind = "aie2p-bf16-mmul-f32acc", lanes = 1 : i64}
]
```

The array is sparse, sorted, and unique by current analyzed task index. Absence
means inherit the module policy; an explicit scalar entry overrides a matrix
module policy. Do not reuse the derived `parallelLoop` field as user input.
Do not put these selectors directly on Taskflow task operations: those attrs
would enter the current function-body fingerprint and change existing source
calibration identity. The selected plan/graph and complete candidate identity
must still include the table and its realized per-task choices.

Typed API in `NativeTransforms.h/.cpp`:

```cpp
struct NativeTaskImplementationChoice {
  size_t task;
  std::string kind;
  int64_t lanes;
};
bool readNativeTaskImplementations(
    mlir::ModuleOp, size_t taskCount,
    std::vector<NativeTaskImplementationChoice> &, std::string &error);
bool setNativeTaskImplementations(
    mlir::ModuleOp, size_t taskCount,
    const std::vector<NativeTaskImplementationChoice> &, std::string &error);
```

Use a shared syntax validator for the global and local requests. Reject unknown
fields, duplicate task indices, out-of-range/negative indices, wrong attribute
types, unknown kinds, scalar/matrix lanes other than one, vector lanes outside
power-of-two 2..16, and mixed lanes outside 4/8. Source eligibility and target
capabilities belong to planning/checking after transformation. A CPU-inactive
request must remain syntactically valid, but must not require a native source
proof or consume an NPU resource.

Local requests are strict selections. Resolve local override before the module
policy; prove the exact requested implementation or report unsupported. Never
silently replace an explicit local matrix/vector choice with scalar. Have the
checker independently read the table and compare the selected kind/lanes and
derived loop against that request, in addition to its existing proofs.

## Transformation lineage and CPU inactivity

`ProgramAllSearch.cpp` already marks original tasks with `orbit.search_origin`,
applies source actions, analyzes the result, and builds original-to-descendant
task lists. Collect local bindings by original task index, then expand and
write the module table after this analysis, alongside the existing global
`setNativeImplementation` call. Task-specific selectors require unique property
writers; explicitly define global-default/local-override precedence rather than
treating two local writers as an overwrite opportunity. Replay must consume and
verify the same table, not only the existing module-wide selector.

The current `splitNativeOuterLoops` splits affine loop ranges inside the same
Taskflow task; it does not itself create separate Taskflow tasks. Eligibility
must nevertheless be re-proved on the changed body and every selected phase.
Do not filter modes using only the untransformed source: initialization and
other source actions can change eligibility.

`distributeNativeInitialization` does create a cloned prologue task. That clone
inherits `orbit.search_origin`, so copying a strict matrix choice to every
descendant would incorrectly select matrix for a fill-only initializer. Record
generated task indices in the module-level
`orbit.generated_initializer_tasks` table and select scalar for those adapter
components. The table is recomputed from actual cloned operations, checked
against constant-store source and exact write-only accesses, and reindexed
when projecting children. It does not alter the function-body fingerprint.
Do not infer provenance from the `.initialize` name. Explicit non-scalar local
requests for an initializer are rejected, even with valid provenance. During
search, only newly split descendants of the selected original compute task
receive the generated-fill scalar policy; an original task request cannot be
silently erased by a forged table. Source-family normalization strips this
module metadata while retaining every actual source operation.

When a task is CPU placed, its NPU local selector is inactive just like existing
native resource/shape/route choices. Keep that fact in the coverage report. In a
mixed placement, eligibility checks run on the NPU child tasks only. A source
mode that fails a registered implementation proof excludes that mode, not all
possible physical implementations of the source. An unimplemented lowering or
missing price remains unknown under the existing evaluator contract.

## Region projection and pricing

`extractHeterogeneousRegionSource` in `HeterogeneousMapping.cpp` copies all module
attrs with `setAttrs(parent->getAttrs())`. It then clones a task slice and records
`projection.parentTasks`. Filter and reindex the sparse implementation table
using that map; never copy parent task numbers unchanged into a child. CPU
children should remove native selectors from their executable input while
retaining inactive selection provenance in the parent receipt. Existing global
matrix fallback for a child with no eligible matrix task must not erase explicit
local requests for NPU children.

`ProgramAllHeterogeneousCost.cpp::heterogeneousCostSourceFamily` currently strips
global implementation attrs but not a future local table. Extend normalization
to strip that table and the module initializer-provenance table. This keeps
semantic source family separate from implementation selection; it does not make
old implementation calibrations valid for new mixed assignments.

There is a concrete pricing limitation. Heterogeneous row matching uses the set
of implementation kind strings, without lanes or task association. Its quantities
include total source operations/accesses, matrix instance count, phases, DMA,
layout/scratch/ABI bytes, but no per-task implementation assignment or lane work
distribution. Two equally shaped matrix-eligible tasks with swapped scalar/matrix
choices can have the same row key and quantities while having different work or
critical-path behavior. `collectGenericCostFeatures` has more detailed mixed8
arithmetic quantities, but also uses an implementation set for its whitelist;
those quantities do not establish calibration for arbitrary new combinations.

Before pricing mixed assignments, collect work from the actual checked instances
separately for scalar, matrix, and vector implementations. At minimum include
source arithmetic operations and accesses per implementation, matrix contracted
work, lanes, and selected instance counts; count tiled domains rather than
duplicating the complete source operation count for each phase. Retain separate
control-family guards for physically different runtime protocols. Freeze and
calibrate these features over a declared domain covering the combinations.
Until that exists, unmatched mixed combinations are feasible with unknown cost.
Do not reuse an all-matrix or module-policy row merely because the kind set and
numeric ranges match. Keep source family dependent on numeric source semantics,
not implementation assignment. Neither complete graph/plan ID nor an individual
task-assignment fingerprint should become a separate fitted price-table row:
identity establishes evidence provenance, not a generalizable cost feature.
No timing oracle may be a pre-priced table of complete plans disguised as joint
optimization.

## File ownership for an implementation pass

| File | Required change |
|---|---|
| `include/Taskflow/NativeTransforms.h`, `lib/Taskflow/NativeTransforms.cpp` | Typed sparse input reader/writer, shared syntax validation, initializer provenance |
| `lib/Taskflow/GenericMapping.cpp` | Resolve strict local overrides, preserve legacy global policy, independent checker reconciliation |
| `lib/Taskflow/ProgramAllSearch.cpp` | Admit local native bindings; per-original-task domains; lineage expansion; CPU guard; replay and coverage |
| `lib/Taskflow/HeterogeneousMapping.cpp` | Filter/reindex selector table on projection; preserve strict NPU overrides |
| `lib/Taskflow/ProgramAllHeterogeneousCost.cpp` | Normalize new metadata, collect actual per-implementation work, and require calibrated feature/control domain |
| Native implementation/projection/search/cost tests and tools CMake | Register meaningful contracts and ON/OFF finite search tests |

No new ProgramGraph IR, architecture model, solver, or AIR emission policy is
needed for the binding itself. Audit emitter behavior with actual selected
instances; do not assume plan serialization alone proves realization.

## Acceptance tests and finite proof boundary

1. Legacy module-only requests reproduce existing per-task choices and source
   fingerprints. Explicit scalar under global matrix remains scalar; malformed
   local schemas and duplicate property writers fail before evaluation.
2. Two matrix-eligible tasks plus a scalar bridge in one NPU region realize all
   four scalar/matrix assignments. Assert each task's plan and graph instance
   kind, actual generated operation family, and distinct selection identity.
   Forge a checked plan by swapping a requested kind and require rejection.
3. Place the same source CPU/NPU/CPU: inactive local requests on CPU tasks do not
   reject the candidate; the NPU table has correctly rebased child indices.
   Test nonzero parent task indices and several projected children.
4. Enable initialization distribution and range fission with local matrix and
   mixed-vector choices. Prologue is explicitly scalar, update choice follows
   lineage, and actual per-phase tile tails/lane restrictions remain checked.
5. Swapped mixed assignments with the same kind set do not acquire a calibrated
   cost from an unrelated row. Source-family equality is permitted; complete
   graph/control identity must differ. Missing calibration is cost unknown,
   never physical infeasibility or a zero-cost substitute.
6. Exhaust a small declared two-task implementation × placement × source-action
   domain through the existing C++ evaluator and independently replay each
   assignment. CP-SAT OFF remains truthful baseline-only; ON claims finite-domain
   optimality only with exhaustive coverage and no unknown feasibility/cost.

Adding task-local selectors increases the Cartesian domain exponentially. A
600-second branch-and-check run can improve exploration but cannot promise an
optimum or full abstraction coverage. Report generated domains, inactive and
fixed dimensions, unknown lowering/pricing counts, and the exact proof scope.

## Implementation checkpoint and validation

`check-taskflow-task-implementations` currently passes 80 host-only checks in
an AIR-enabled ON build. The real BF16 source contains two contractions with
K=16 and K=8 followed by a scalar epilogue. All four independent scalar/matrix
combinations produce distinct checked graphs and valid AIR; actual matrix-call
counts follow the selections. Source/initializer fingerprints, strict malformed
and forged selections, initialization/fission, CPU inactivity, nonzero regional
index projection, and actual C++ finite search are covered. The assumed
nonadditive search oracle is explicitly a CPU contract, not hardware pricing.

Independent arithmetic expectations are matrix work 16384 versus 8192 source
operations when swapping the two choices, and scalar work 8448 versus 16640.
An otherwise valid aggregate-only calibration row leaves these mixed choices
unknown; explicit implementation-resolved terms enable the synthetic ownership
contract. Runtime layout/scratch quantities also have matrix-specific feature
names when the physical receipt supplies them. Hardware calibration for new
mixed combinations remains separate required work.

ON logs: `/tmp/orbit-per-task-v32-final-on.log` and
`/tmp/orbit-per-task-v32-final-air-contract.log`. Existing inactive bindings (9),
regional projection (27), heterogeneous costs (53), and direct CLI tile
precedence (6) passed. The OFF checkpoint also passes 80 checks (AIR enabled; solver disabled) in
`/tmp/orbit-per-task-v32-final-off.log`, plus inactive bindings (6), regional
projection (27), heterogeneous costs (53), and CLI tile precedence (6).
Solver-off evaluates only baseline and cannot claim exhaustive optimum.

## Post-r4 direct-provider guard

After the r4 measurement tool was frozen, the final source also added a narrow
legacy `linear-composite` guard. Native feature collection records explicit
local selection presence. For a mixed local scalar/mixed8 assignment, positive
implementation-resolved scalar product/add-sub and mixed product/add-sub-pack
work requires corresponding fitted terms and calibrated ranges; an aggregate
command/source-count model returns `out_of_domain` with no price. Other mixed
implementation kinds without this provider's work contract remain unknown.
The established `implementation-composite` keeps its strict four-term contract.
This guard is not attributed to the older frozen r4 binary.

`check-taskflow-task-implementation-cost` uses the real three-task f32 fixture:
two scalar tasks perform 32 adds, and the selected mixed8 task performs 16
products. Nine CPU-only checks cover collection, aggregate-only negative,
work-resolved positive, missing-range negative, and unchanged strict
implementation-composite acceptance. The synthetic linear expectation is
0.436 ms; the strict four-term expectation is 0.78 ms for one command and three
checked native stages. These are API arithmetic contracts, not measurements.
The old direct-AIR cost contract also passes all 77 checks.

Logs: `/tmp/orbit-per-task-v32-postfreeze-guard-final-on.log`,
`/tmp/orbit-per-task-v32-legacy-direct-cost-on.log`, and the corresponding
`/tmp/orbit-per-task-v32-postfreeze-guard-final-off.log`.

## Post-prep7 stale-provenance compatibility fix

A full regression found that a legitimate later source edit can invalidate an
initializer's derived constant-store role. The metadata reader now keeps strict
table type/index/uniqueness validation, but only returns structurally proved
roles. An unproved stale entry does not reject the actual source or suppress an
explicit local request. The existing arithmetic-valued initializer regression
again produces a legal native plan with unknown mechanism cost. New negatives
cover a forged epilogue claim, wrong table type, out-of-range index and duplicate
indices. Final ON/OFF typed counts are 80 each, mechanism counts 77 each.

This fix follows the prep7 measurement-tool freeze. A fresh gemm16 source was
re-exported on the current checkpoint: child/physical graph identities and both
Taskflow source and AIR bytes exactly match the prep7 artifacts. The receipt is
`/tmp/orbit-v32-provenance-normalcase-r1/receipt.json`. This source-equivalence
check is not hardware execution or a lock-progress certificate.
