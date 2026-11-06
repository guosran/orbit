# Target-bound scoped execution plans, v19

The existing production chain remains Taskflow source actions → GenericAnalysis /
GenericMapping → checked ProgramGraph → ProgramMechanismCost / ProgramTiming and
original C++ beam → direct AIR → AIE. Python records experiments and numerical
references. It does not decide placement, retain boundaries, cost or parent choice.

Target domains are concrete instances. Resources and memories reference their
owning domains, and access is authorized by explicit compute access lists and
DMA links. Domain coordinates are opaque common metadata interpreted only by the
backend. The default NPU target declares column 0, row 2; the AIR backend explicitly
rejects unsupported placements rather than remapping them. Synthetic public
planning is independent of NPU identifiers. Resource implication edges express
shared reservations; a per-activity idempotent maximum closure avoids duplicate
charges, while separate activities contend on the common shared resource.

A source mapping choice can contain several nonoverlapping `residency_groups`,
each naming source tasks and retained logical storages. Derived residency scopes
record their phase range, selected tasks, retained storages and required boundary
exports. A retained value is not globally exempt from materialization. A group
with an outside reader still exports its correct version, while internal readers
share the same local copy. A pure internal value has no host allocation or DMA.
The physical copies remain region/version scoped; all readers and exports must
finish before the allocation is overwritten. Validity ordinals are not completion
events.

The initial certified subset is scalar pointwise, static rectangular matching
grids, unique writers, contiguous source-topological task intervals and disjoint
groups. Pure order, cross-block dependencies, in-place updates, reductions,
unsupported region/layout relationships and incompatible stream counts reject
with diagnostics. Unsupported patterns do not inherit the new certification.
Repeated boundary-input deduplication within a multi-task group remains an
explicit guard until a shared-delivery lowering is supported.

Group semantic dependencies are derived from actual source data/control use.
`implementationDependencies` separately express the single-tile configuration
lifecycle. Backend capability also declares block-entry acquisition of all input
streams. These refinements are present before timing: the public evaluator does
not price a different synchronization topology from lowering. ProgramTiming keeps
semantic predecessors, resource waits and allocation reuse distinct. Multiple
independent resources and buffer overlaps remain covered by synthetic tests;
this revision does not claim real multi-tile execution or double buffering.

AIR checking validates actual operations, values, buffers, regions and token
structure against source-bound lowering. Final AIE checking re-extracts concrete
core position, buffer sizes, view and scalar address coverage, arithmetic value
DAGs, DMA buffer/channel/flow/BD coverage, cyclic lock dependencies and runtime
start/await/configuration ordering. Declaration attributes propose joins; actual
operation/SSA facts must substantiate them. Both required public constraints and
actual implementation constraints must have causal coverage. Legal backend
serialization belongs to the plan rather than being silently ignored by cost.

The original search discovers producer–consumer intervals from accesses, combines
disjoint scopes with tiling, checks capacity/regions before acceptance and archives
rejected reasons. Proposal/group/candidate budgets bound exploration. Cache keys,
source identities, parent actions, checked graphs and checkpoint behavior stay in
the original driver. Estimate mode reuses the nine explicit v18 assumptions;
strict mode leaves those absolute costs unknown. Search feedback controlling
round two, real ranking and measured acceleration are separate claims.

The companion result root is
`orbit-artifact/results/taskflow/subgraph-execution-v19/`. Its final RESULTS,
DELIVERY and REPRODUCE files identify the tested source/builds and any remaining
support boundaries. Historical v18 experiments and failed v19 attempts are kept.

## Final actual evidence

The final scoped contract adds `inputReleasePolicy`: original groups use
`group-tail`; entry-frontloaded late-input groups use `last-reader`. The latter
keeps each actual copy live until its own final reader while output reuse still
waits for the block-tail export. Upfront real DMA/wait implements the public
all-input readiness. The initial output-role AIR failed both entry and overly
broad release checks; these failures are preserved and the old actual artifact
continues to be rejected by the corrected checker.

The final eight generated/compiled plans exactly match current Core and AIR
plans/graphs; actual AIR bytes and generic final-AIE checks pass. Two final
process sessions execute528 calls/1056 complete output arrays correctly, after
the separate initial seven-plan study. Model parameters are unchanged assumptions.
Score-driven two-round search succeeds; fine real ranking does not satisfy a
general reliability claim (10/16 then15/16 median pair comparisons). The final
composition improves its paired baseline in both sessions. See the companion
RESULTS and exact proof manifests for the complete raw distributions, including
the noisy first session.

The native serializer retains its old `npu_single_core_serial` placement tag as
a compatibility diagnostic. It is not the new region placement or target binding,
and no planner/lowerer may derive resources from it. The selected resource,
memory owner, coordinates, access lists and explicit invocation lifecycle are
authoritative. Remaining legacy labels are not evidence of physical placement.
