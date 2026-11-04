# Native Taskflow to AIR lowering

`MLIRTaskflowAIR` is an optional C++ backend (`ORBIT_ENABLE_AIR=ON`).
`MLIRTaskflowStructured` and the independent Core do not link AIR, AIE, XRT,
or Neura. The backend builds AIR operations using MLIR `OperationState`;
the installed `air-opt` subsequently parses and verifies the actual AIR dialect.
This is direct lowering of Taskflow bodies, not a Triton recipe generator.

The shared `GenericAnalysis` comes from actual function arguments, allocations,
task block arguments, SSA dependence states and affine/SCF memory accesses.
A native input declares `orbit.noalias` on its function and `orbit.role` on
memref arguments. `orbit.name` is optional reporting identity. No `orbit.plan`
or complete JSON computation graph is read by this route.

## Current resource strategy

The v6 extension adds `serial-tiled` mapping alongside the resident strategy
described below. Explicit `--tile-m/--tile-n/--tile-k` choices are materialized
as the module's `orbit.air_tile_sizes` attribute before generic analysis.
The names refer to loop depths, not recognized arithmetic operators. The
mapping derives rectangular DMA regions from actual accesses. Explicit
overcapacity choices are rejected, rather than silently resized. When no
explicit choice is supplied, overcapacity input uses conservative automatic
selection; unsupported dependence/layout cases fail closed.

`--air-distribute-initialization` separates a constant-store prologue only
after proving that accesses to the written storage project onto all enclosing
loop induction values. This is an access/dependence transformation, independent
of the update arithmetic. Distinct body arguments aliasing the output storage
prevent this transformation. The update consumes the initializer's explicit
Taskflow write state. `--emit-mlir` preserves the transformed semantic source.

Tiled lowering uses checked firmware-sequenced hardware stages. Stage boundaries
follow tasks and output-shape groups without crossing retained-output scopes.
Read-only input pools distinguish actual tile shapes and loop templates; their
combined allocations are charged to each stage. The stage `buffer_pools`
ledger is collected by the same capacity routine and lists each pool shape,
bytes and stage lifetime. `host_buffers` and `system_memory_peak_bytes` describe
whole-graph BO tensor payload separately; runtime bookkeeping, reference arrays
and compiler host RAM are outside that system-memory total. The legacy per-task
`herd: 0`/`core: 0` fields denote the logical single-core slot; actual generated
herds/stages come from `hardware_stages`, including multiple tail stages for
one task and local-only initializer clones. Temporary tensors are appended to the function ABI as system-memory scratch
BOs. It compresses consecutive translation-equivalent checked phases into
static loops, preserving full/tail order. Compression checks every phase
against the plan; it does not select a different tile size. Loop-carried AIR
tokens enforce the serial schedule. The compiler invocation explicitly omits
automatic ping-pong buffering for this strategy. The finite phase list is
still enumerated internally in C++; large reports use compact domains and a
digest covering the complete schedule. This is not yet a fully symbolic
planner, and its memory cost is reported with the large-shape evidence.

A generic pure initializer can be fused into the first phase of a retained
output scope after independent source-access and consumer-coverage proofs.
The backend clones its actual body once, retains the accumulator through all
K phases in source order, and exports only after the last update. This does
not match GEMM arithmetic. The plan records initializer identity, scope,
data version and the removed host transfer. Tiled read/write regions without
this proved initializer currently fail closed: the installed bidirectional
host-feedback route has not passed hardware validation.

For stages with multiple read-only transfer templates, the plan can explicitly
request channel legalization and assign storage/direction to an available DMA
port. Such conflict stages contain exactly one retained-output scope. The
conservative hardware-stage expansion limit is 128, a host/code-generation
complexity bound rather than a claim about physical device capacity. The plan
assigns one stable logical FIFO per stage/storage/direction, with checked
message and payload-byte counts. C++ coalesces full/tail channel symbols only
after verifying matching producer/consumer stream order against that plan;
regions and arithmetic remain intact. This gives the shim allocator one
producer identity for the full/tail stream, in addition to compute-port pins. The reproduction wrapper preserves canonical DMA AIR, runs the installed
standard DMA-to-channel/canonicalize passes (aircc performs dependency
analysis once downstream), then invokes C++
`--legalize-air-channels=FILE` and the separate `--check-air-channels=FILE`.
The converted input must use generic operation syntax. The legalizer owns
resource attributes; Python only invokes tools. The standard installed
conversion is a trusted compiler boundary, not an independent formal proof of
arithmetic equivalence. Each actual hardware outcome remains recorded
separately, and channel resource checks alone do not certify device execution.

Channel-legalized L1 pools have explicit static hardware-stage scope lifetime.
They omit dynamic `memref.dealloc` operations because the installed dependency
pass otherwise creates invalid dependencies from a stage-end deallocation to
an inner reduction loop token. Capacity counts every pool for the entire stage;
there is no mid-stage release or dynamic allocation reuse. Other stages retain
the existing deallocation form, and host BO lifetime is unchanged.

The payload budget remains **24576 bytes**. The original constant was an ORBIT
policy applied to the sum of all full input, output, intermediate and scratch
allocations for one herd's lifetime, without padding or stack accounting. It
was neither an XRT BO limit nor a measured silicon capacity. The installed
`npu2` target model reports 65536 bytes local memory per compute tile and
524288 bytes per memory tile; these are compiler-model facts, not a direct
measurement of the installed chip. New linker audits separately inspect
physical buffer addresses/sizes and the 4096-byte stack reservation. Code,
compiler routing resources and any later physical traffic remain separate
verification boundaries.

See [v6 evidence](taskflow-direct-air-v6-validation.md) for actual hardware
statuses and retained failures. A successful plan or AIR verifier result alone
does not establish device correctness. Reproduce in a fresh directory with
`bash scripts/run_taskflow_tiled_air.sh cpu NEW_DIR` or `all NEW_DIR`.

### Original resident strategy

The checked `GenericMappingPlan` selects one launch, one segment and one 1x1
herd for the whole graph. Several tasks execute in the same herd; there is no
one-task/one-herd rule. Every buffer is allocated in memory space 2 (AIE local
memory) for the herd lifetime. Full input buffers transfer once into local
memory; full output buffers transfer once out. Intermediate buffers remain
local and never acquire host DMA just because their producer task ended.

The initial conservative payload limit is 24 KiB. This is a bounded strategy,
not a claim that arbitrary FFN matrices fit on chip. The AIR compiler may add
shim descriptors and internal transport resources; final generated AIE IR is
part of the evidence. Declared payload capacity alone does not certify all
compiler-generated resource use. No old Triton/chain latency is reused.

The system-memory XRT BO and AIE local allocations are different storage
levels. The execution harness uploads buffers and invokes the installed
executor; it is not a new runtime. BO reuse is not weight residence in SRAM.

## Body and dependency lowering

The backend remaps actual task block arguments to analyzed storage allocations
and clones the actual arithmetic/memory/control-flow operations. It does not
recognize GEMM, ReLU or SwiGLU before cloning. MLIR's standard
`createLowerAffinePass()` lowers affine loops/accesses to SCF/memref operations
in C++. Original source and generated IR are both retained.

Tasks are conservatively ordered by the native checked schedule. Synchronous
AIR waits delimit source tasks; input/output DMA have asynchronous tokens.
The installed AIR dependency pass converts real loop/memory effects into loop
completion tokens and DMA dependencies. For example, in add→scale its generated
second loop depends on the first loop's completion token, and final DMA depends
on the second loop. Empty source boundary tokens are reporting/order markers,
not measurements of computation. The dependency-pass output is retained to
inspect the actual asynchronous dependence graph.

The fixed installed AIR dependency pass rejects affine loops and produces an
invalid dominance edge when an existing `air.execute` encloses a whole SCF loop
nest. Both initial failures are retained. The supported route uses standard
C++ affine lowering and leaves loops directly in the herd so the existing AIR
pass constructs their asynchronous form. No installed package is modified.

`checkGenericAIRMapping` independently checks emitted resident allocations,
shape/dtype/memory spaces, exact full-buffer transfer directions/extents/counts,
external ABI arguments, task coverage/order/source dependencies and plan/source
identities. It also checks real 1x1 hierarchy sizes and nesting, ABI forwarding
through every hierarchy level, resident placement inside the herd, task start
tokens, and final output DMA dependencies. Installed `air-opt` also verifies AIR's own dialect invariants.
Arithmetic mutation and malformed-plan tests are separate from identity checks.

## Precision and unsupported cases

Default lowering preserves f32 arithmetic. It neither silently converts to bf16
nor falls back to CPU or Triton. Input aliases/in-place updates, dynamic shapes,
unsupported layouts, and used scalar task arguments are explicitly rejected by
the relevant analysis/backend stage. Analysis capability, target code generation
and availability of calibrated costs are separate statuses.

The Peano backend rejects scalar `math.exp` as `G_FEXP` for the exact small FFN.
An explicit `--air-approximate-exp` option selects the fixed LLVM installation's
f32 Cephes exponential approximation patterns only. It records
`orbit.math_approximation` in AIR. This is an approximation of the real SiLU
exponential, not replacement of SwiGLU by a toy function. The upstream expansion
flushes denormals and has limited extreme-range behavior; it is not a promise
of correctly rounded exp for all floats. No global fast-math permission is
introduced. Validation must keep the same independent FFN reference and report
any remaining compiler failure or numerical error. The option is never implied
by an FFN name.

The fixed Peano backend also rejects `math.fma` and `math.floor` introduced by
the upstream polynomial. The backend applies MLIR's standard FMA expansion to
separate multiplication and addition. It expands only the newly introduced
bounded floor using f32-to-i32 truncation, conversion back, and subtraction of
one for a negative fractional part. The polynomial clamps its argument before
this floor; existing source `math.floor` operations are untouched. This option
has an explicit finite-input contract and records `unfused-mul-add`,
`bounded-floor-i32`, and `finite-input-contract` in its approximation metadata.
It does not promise NaN/infinity input behavior or fused rounding equivalence.

The supported validation domain for standalone normal finite exp is [-80,80];
the recorded sweep is empirical evidence, not a proof for every f32 value.
The explicit approximation strategy clamps the polynomial argument to
[-87.8,88.8], flushes subnormal results, and does not preserve the exact
overflow boundary. SiLU and SwiGLU probes cover [-103,103] separately and
report both absolute and relative errors. The original finite-output exp edge
test fails and remains a failure. See `taskflow-direct-air-v6-validation.md`
and the v6 RESULTS artifact for full ranges, metrics and classification.

## Reproduction

From the repository root after configuring `build-air` with the same fixed
LLVM/MLIR installation as Core and `-DORBIT_ENABLE_AIR=ON`:

```bash
cmake --build build-air --target taskflow-opt -j2
mkdir -p build/direct-air-local
build-air/tools/taskflow-opt/taskflow-opt \
  test/TaskflowAIR/fixtures/add-scale.mlir --lower-to-air \
  --analysis-output=build/direct-air-local/facts.json \
  --mapping-output=build/direct-air-local/plan.json \
  --air-output=build/direct-air-local/program.air.mlir
source /home/shiran/project/aie-toy/scripts/env.sh
air-opt build/direct-air-local/program.air.mlir --air-dependency \
  -o build/direct-air-local/dependencies.air.mlir
```

The backend also provides an actual `taskflow-to-air` MLIR `OperationPass`;
`--air-pass-pipeline` exercises it. Callers must allow unregistered AIR
operations on their context before entering a pass manager. The pass declares
its dependent dialects before execution and invokes standard affine lowering
through MLIR's dynamic pipeline API, without changing context configuration
inside a running pass. Regression tests execute this real pass path and compare
its AIR byte-for-byte with direct lowering. Experiment execution and toolchain evidence
are documented in the v5 results. Hardware runs are serialized by their existing
`/tmp/npu.lock` owner; do not add a second outer lock.
