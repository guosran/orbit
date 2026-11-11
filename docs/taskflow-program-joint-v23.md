# Public ProgramGraph joint CP-SAT search

This entry solves a finite realization domain over the existing source-bound
`ProgramGraph`. One CP-SAT model selects the source partition variant, an exact
cover by local task groups, compute/SRAM placement, boundary version copies,
transfer links, configurations and compute/DMA intervals. It does not select
among pre-priced complete plans. The earlier `cp-sat` local-delta search and
independent fixed-shard scheduler remain separate historical interfaces.

The common implementation is `ProgramJoint{Domain,Solver,Plan}.cpp`.
`ProgramJointAIR.cpp` implements the chosen decisions; `ProgramJointAIE.cpp`
checks final installed-toolchain output. OR-Tools is optional and remains off in
the default Core build. Use the v22 documented OR-Tools C++ SDK and build flags.

## Invocation

```sh
taskflow-opt SOURCE.mlir --target-config=MODEL.json \
  --joint-program-search=NEW_DIR --joint-seconds=5 --npu-shards=2 \
  --joint-lower-air
```

The existing native command also dispatches to this model when the target has
`"native_auto_candidates":{"mode":"joint","algorithm":"cp-sat-program"}`:

```sh
taskflow-opt SOURCE.mlir --search --backend=air --search-output=NEW_DIR \
  --target-config=MODEL.json --joint-lower-air
```

This entry uses a fresh finite solve. Legacy staged proposal fields and resume
are rejected. Domain limits belong in `program_joint`; solver budgets use
`--joint-seconds` and `--joint-max-variables`. `--joint-baseline-only` publishes a
baseline without invoking CP-SAT. Without AIR, an explicit common architecture
can be supplied and solved independently of the physical adapter.

`domain.json` contains public source graphs, versions, local modes and target
facts. `baseline.json` is independently checked and written **before** solving.
With AIR lowering enabled, `baseline.air.mlir` is also checked before solving.
`solution.json` contains frozen raw choices, status, telemetry and the resulting
public graph. `program.air.mlir`, `abi.json`, `target.json` and `manifest.json`
record the concrete realization. A host ABI may include temporary BOs when a
computed version crosses configurations.

Compile with the installed `aircc`, export generic final AIE, then check the
frozen decisions without re-solving:

```sh
taskflow-opt SOURCE.mlir --target-config=NEW_DIR/target.json \
  --joint-program-plan=NEW_DIR/solution.json \
  --check-joint-aie=FINAL.generic.mlir --implementation-output=receipt.json
```

Reconstruction binds source/target/domain identity and rejects a changed stored
graph. The joint checker reprices compute and transfers, checks exact integer
resource calendars and validates the public graph. The common resource-schedule
validator checks static occupancy, hierarchical resource reservations and
configuration barriers. Legacy graph schedules keep their existing JSON form.
Final AIE validation proves caller ABI forwarding, physical/core ownership and
symbol/SSA isolation **before** projection, then verifies each selected source
group's arithmetic, addresses, buffers, locks, DMA and output completion.

## Current executable domain

The tested realization supports static one/two-dimensional affine pointwise
programs with a common independent axis zero, equal storage/grid shapes, exact
accesses, one writer per storage, and scalar task implementations. Local groups
are connected by actual producer/consumer or shared-input values. Tiling is a
finite choice of independent row chunk sizes, including tail instances. The
source partition proof and common graph checker remain authoritative.

Each configuration has at most one static group per compute resource. Group
members execute in source order; all group inputs arrive before computation and
outputs follow the group computation. A selected group shares its internal or
read-only copies. Different dependent groups communicate through explicit host
exports/imports in later configurations. Static SRAM lasts for the complete
configuration; no intra-configuration allocation reuse is invented. Each
configuration uses a separate synchronous AIR launch to preserve host-buffer
round trips in the installed channel conversion.

The physical adapter requires core-owned local SRAM, caller BOs, explicit NPU2
coordinates and direct shim/core streams. Default expansion provides independent
input/output DMA services for the requested 1–4 cores; public target facts may
express additional shared contention. The certified examples use one and two
cores, at most two input/output streams per configuration. Packet/memtile and
cross-core SRAM transfer/retention are rejected by final checking. Layout and
dtype are source/target constraints in this version, rather than additional
search variables. GEMM, attention and arbitrary reductions are outside this
joint executable subset. Existing other compiler paths remain available.

## Costs, optimality and bounds

An explicit `orbit.direct-air.cost.v1` event-mechanism model supplies scalar
compute, configuration, host boundary and communication quantities, with units
and provenance. Missing prices are errors; strict mode rejects assumed prices.
Durations are quantized upward to integer nanoseconds. Public solution makespan
includes host submit/complete constants and configuration setup. Public graph
schedule completion omits the final host-complete constant.

`OPTIMAL` means optimal for the admitted finite model and supplied costs. It is
not a measured NPU latency guarantee. Actual execution implements readiness,
resource arbitration and ordered configurations, without a timestamp dispatcher.
A backend can reject a selected graph but cannot silently change its partition,
groups, copies, placement or transfer set to repair it.

The checked baseline supplies a safe horizon and hints. An exhausted solve or
model limit returns that baseline with `selection=baseline` and the original
`UNKNOWN`/`RESOURCE_LIMIT` status. An infeasibility contradicting the checked
baseline is an encoding error. Model construction has checkpoints for time and
actual variable/constraint counts; discovery and validation remain nonpreemptive,
so process watchdogs and reported whole-process time remain necessary. A baseline
AIR witness is not a substitute for final toolchain/AIE validation.

The encoding expands group × placement × configuration assignments and their
optional compute/transfer intervals. Increasing all three axes can become
expensive before search begins. Explicit group/instance/variant bounds describe
the admitted domain; truncation is reported, never treated as physical
illegality or a full-space optimality proof.

## Validation

```sh
cmake --build BUILD --target check-taskflow-program-joint
```

The finite compiler contracts independently enumerate structures, placements and
all unary-resource orders for a source-derived two-task shared-input program.
The optima are 21 (8-byte SRAM), 18 (12-byte SRAM and shared DMA), and 13
(independent DMA). They verify status/fallback, frozen-plan corruption and a RAW
edge that forces two configurations, hierarchical DMA contention and shared static memory. The default-off suite checks the same
public plans while requiring `UNAVAILABLE` and a checked baseline.

The companion `run_program_joint_v23.py` compiles and checks five implementations of two source programs and,
with `--execute`, runs serialized main-agent NPU trials against FP32/FP64 source
references. `program_joint_scaling_v23.py` varies source task count, physical
resources and partition variants, records full model/solve/process times,
variables, constraints, statuses, bounds and RSS, and preserves saved baselines
and external timeouts. See the companion v23 validation report for frozen inputs,
measurements, failures and reproduction commands.
