# Controlled reuse/tiling search, v21

The production path stays native Taskflow → GenericAnalysis → ProgramGraph →
original structured search/ProgramTiming → AIR → AIE. V20 implements restricted
joint enumeration and legality screening. V21 tests its algorithmic contribution
with stronger staged policies and target/access/capacity-derived tile proposals.

## Questions and controlled objects

The experiment protocol separates four questions. A compares admitted legal
configuration spaces and actual implementations. B compares strategies within
the same bounded catalogue and explains phase exclusions. C controls both raw
complete-cost-provider calls and total search time, including construction and
internal planning. D compares tile origins, feasibility and quality per reuse
organization. A lower assumed cost, a new executable plan, observed benefit and
reliable measured ranking remain distinct claims.

The catalogue is derived in C++ from exact source regions/versions and consumer
demands, including bounded disjoint multi-group combinations and an ungrouped
baseline. It uses a shared tile universe and checks every admitted scope/tile
with the existing planner/checker. Explicit bounds, failed configurations and
whether the complete product was visited are reported. A reference optimum is
only for that admitted catalogue and the unchanged assumed model.

## Policy contracts

All new policies share the same catalogue and backend capabilities. Staged reuse
first admits multiple groups and their combinations; its feasibility witness
prefers the current tile, otherwise a deterministic feasible capacity repair.
It does not score-select a best tile before the tile stage. After stage-rounds2,
organization freezes per score-selected parent and tiling is searched. Tile
first tunes ungrouped tiles, then freezes tile and searches the full organization
set. Alternating changes reuse at fixed tile, then tile at fixed reuse. Joint
permits a scope insertion/removal/replacement and tile change in the same action.
The original source action materializes choices before hashing and evaluation.

Four rounds and beam2 are the controlled defaults. Every policy receives the
same remaining-work share across remaining rounds, so stage1 cannot consume the
entire comparison budget. Baseline work is charged. Carrying an existing state
into a later stage does not create a fake new physical identity: checkpoint
validation requires exact preceding selection and phase-specific action replay.

## Tile proposals and accounting

The new generator proposes capacity boundaries, divisors/tail alternatives,
target alignment preferences and current-tile neighbors, retaining historical
halving as fallback/control. Every tile has provenance. For static identity-grid
stages with one writer per storage, identity layouts and matching access ranges,
the sum of distinct storage element widths times the covered tile volume is a
necessary payload lower bound. A bound above capacity safely rejects; a bound
below capacity never certifies legality. Extra copies and implementation buffers
can increase occupancy. Unknown or nonmonotone cases use bounded planner probes;
no binary search assumes monotone full occupancy. Final legality remains the full
source-bound proof.
Optimizer-domain hard alignment/divisibility restrictions are explicit and do
not reinterpret an existing target alignment preference as a universal limit.

Generation probes are unpriced. Raw telemetry counts planner/checker entries
and actual cost-provider calls, including the independent checker's recomputation.
An archive score query is therefore reported separately from cost invocations.
Checked outcome caches retain no MLIR pointers and bind source semantics,
target, selections and model versions. Per-parent phase action caches avoid
rebuilding proposal pools repeatedly during checkpoint validation. Revalidation
work remains visible and independently checks source-bound artifacts. The checked
outcome cache eliminates repeated discovery probes; selected candidates are still
fully planned and priced, and checkpoint validation deliberately repeats the
independent proof. It does not promise one planner call per physical plan.

Primary raw cost budgets are16/32/64 calls; wall budgets5/15seconds include
catalogue construction. Generation/planner and catalogue bounds are also shared.
These are optimizer limits, not architecture facts. Local Pareto screening is
only justified within equivalent consumer/organization/boundary states; memory,
traffic, configuration/delivery and computation-instance tradeoffs remain visible.
The reference catalogue retains dominated configurations. Local bounded proposal
selection prioritizes nondominated rows while keeping scope diversity; it never
uses transfer bytes alone as the whole-program objective.

All new policies and both generators share a maximum of256 source compute
instances. For the supported static grids the exact count is the sum, over tasks,
of the product of ceiling(loop iterations / tile dimension). It is checked against
the complete plan's phase count. This bounds planner construction before an
expensive call; excluded configurations remain explicit domain exclusions, not
hardware-illegal plans. Legacy controls retain their historical policy and score
query limit and are explicitly excluded from matched raw-budget conclusions.

## Evidence and supported subset

New evidence belongs in companion results/taskflow/data-reuse-v21. Protocolr3
and its source/budget bindings precede new NPU observation. Both v20 programs are
retained; the third frozen23×152 program is X=A+1,V=2D,Y=2X,Z=Y+A,R=X−1,
O=Z+R,H=V+3. Its organization-dependent buffer/boundary requirements test coupling
without unused computation. It is retained even if no joint advantage appears.
Protocols r1/r2 and the CPU-only deadline pilot are preserved: a tiny halving tile
caused a single nonpreemptive planner operation to overrun the5-second deadline.
The r3 amendment applies the common instance bound before any new hardware data;
source programs, model parameters and evaluation/time budgets are unchanged.

Only static scalar rectangular identity-grid propagation is in scope. Existing
shared-copy, all-reader lifetime, group/caller export, order and tail checks still
apply. No general affine/reduction/multilevel lowering, new runtime or model
refitting is introduced. Final RESULTS/DELIVERY/REPRODUCE select verified current
builds separately from preserved historical Neura/CGRA binaries. Main alone owns
serialized locked NPU experiments. No automatic commit or push.

The completed delivery has120 primary runs, six complete bounded references,
32 checked actual implementations and1224 calls in ten fresh NPU processes.
Independent audits cover2448 complete arrays; Core44, combined61 and preserved
Neura/CGRA/lit contracts pass. An optional extended legacy experiment remains
explicitly resource-stopped. New candidates improve two programs against optimized
halving references, while shared-input does not improve consistently. Strong
tile-first and alternating match or beat joint; extra budget helps joint catch
up rather than proving superior efficiency. Three modeled pair orders are wrong
in both process sessions. Consult the final companion report for separate A–D
answers, actual paired distributions and the limits of the inferred ranking.
