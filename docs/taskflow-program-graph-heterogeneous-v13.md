# ProgramGraph CPU/NPU execution and mechanism costs (v13)

ORBIT searches whole-program structure, resources, data organization and schedule.
CPU, direct AIR/NPU and Amoeba supply implementations and target contracts.
This stage first demonstrates static CPU/NPU execution, then prioritizes an
analytical memory/communication/launch model. Further heterogeneous search and
candidate hardware expansion is paused. Historical v9/v10 and the initial v13
empirical model remain unchanged.

## Public interfaces and actual consumers

| Existing abstraction | v13 extension | Actual consumer |
|---|---|---|
| ProgramGraph source/body, task instances, data/version/copies | Per-region source tasks, explicit member events, input/output copies, predecessor regions, implementation and endpoint memory | Heterogeneous checker, child materializer, implementation binder |
| Execution regions and resource schedule | Explicit submit/complete and visibility events; one invocation shared by all inputs | Static manifest exporter and execution API correspondence audit |
| MemoryArchitecture | CPU system access and per-region NPU local allocation scopes, system/local links | Shared copy planner, capacity/accessibility checker |
| Original C++ structured search | Placement and region-boundary actions materialized as MLIR candidate attributes | Native evaluation callback reconstructs the checked graph and projections |
| CPU lowering | Inline actual Taskflow body into ordinary MLIR, standard LLVM translation and strict FP32 shared library | Existing compiled-function ABI via ctypes |
| Direct AIR | Checked resident child native plan, allocation/transfer equality binding | Existing AIR/AIE compiler and XRT runner with bound BOs |

The main C++ entry is `exportHeterogeneousProgram` in
`lib/Taskflow/HeterogeneousSearch.cpp`. `HeterogeneousMapping.h/.cpp` creates and
checks the parent graph, projects real child bodies and binds checked child
implementation graphs. `HeterogeneousImplementation.cpp` lowers the actual CPU
body. The common graph remains in `ProgramMapping.h`; no JSON program planner
or alternate Python decision engine is introduced. JSON/DOT are exports.

The executor in the companion repository implements the emitted ordered region
ABI. It does not select placement, grouping, copies or arithmetic. It binds
source, CPU library or NPU ELF, graph, execution contract, ABI and observed APIs.
It verifies pointer aliasing, one invocation per region and event coverage.
Static manifest checking reconstructs decisions; executable/source byte checking
is separately performed by compilation/execution and the independent audit.

## Finite execution contract

The initial real program is `T=A+B; U=2*T; C=U+B`. B is reused by its first and
last task. Static identity-layout FP32 memrefs, single-writer storage, explicit
noalias and sequential task execution are supported. CPU kernels use the actual
body, no contraction and no approximate math. NPU regions use the existing
single-herd-resident direct AIR implementation. No scalar/NumPy fallback,
Triton recipe, new runtime or precision/tolerance change is used.

The CPU can access host arrays and mapped XRT BOs. Multiple NPU contexts reuse
caller-owned system BOs. CPU-private internal temporaries are compiled host
allocations; exposed NPU ABI storage is mapped BO storage. These are distinct
allocation realizations, despite one host-accessible logical memory location.

Each NPU region-local memory name denotes an allocation scope of one serially
reused 24576-byte tensor payload policy budget. These names do not create
independent physical banks. Compiler stack, code and physical bank allocation
are not certified by that payload number; final AIE allocation/link checks
remain authoritative. Binding verifies the conservative native allocation union
and exact directional native DMA geometry against the parent, in addition to
common event-lifetime checks.

## Four different memory mechanisms

1. Host input reset and final output acquisition are real host copies, timed
   once per whole call. Host-to-host and host-to-mapped-BO are different paths.
2. The bound runner has no additional observed host staging `numpy.copyto`.
   It still synchronizes **all** ABI BOs TO_DEVICE, including overwritten
   outputs, and synchronizes outputs FROM_DEVICE. Conservative output TO sync
   is a backend API action, not a newly created logical data copy.
3. AIR contains explicit system/local DMA. Logical DMA bytes/counts are static
   descriptor facts, not hardware DDR measurements and not the sync API's byte
   count. Reuse inside one resident NPU region changes these descriptors.
4. CPU kernels directly access mapped addresses. Their loads/stores are real
   memory accesses, not zero-cost copies and not separately measured DDR bytes.

BO sharing proves backing/address reuse. It does not remove visibility APIs,
NPU DMA, CPU memory access or initial/final host copies.

## Historical empirical model, explicitly not analytical

Frozen model SHA256 is
`235872a546eda3f8fd981c6970e846bf1e77d3a820aeabf61c555a49e9bf7a6f`.
For each `(device, region task count)` it interpolates measured median regional
wall time between two expanded float-operation-count knots:

`R(w) = t0 + (w-w0)/(w1-w0)*(t1-t0)`.

Whole-call estimate is `sum(R(region)) + B(external-copy-bytes)`. B interpolates
an observed whole-wall-minus-regions residual. CPU labels include descriptor and
ctypes work; NPU labels include sync/start/wait and observation. This is bounded
program-family empirical interpolation, not ML training and not analytical
parameter calibration. Its independent variables and composite labels cannot
identify physical bandwidth, DMA latency or launch overhead. Its exact parameter
table, sampling provenance and uncertainty are retained in the companion v13
report. No mechanism coefficients are inferred by dividing this total time.

## Analytical model boundary

The mechanism extension derives actual copy bytes, sync direction/range and
count, region start/completion and descriptive internal DMA from the same checked
graph plus the declared installed-runner binding. For a supported mechanism/path
and measured state, the initial serial law is:

`T = number_of_calls * (alpha_ms + beta_ms_per_byte * bytes_per_call)`.

`alpha` is measured fixed API cost; `beta` is a nonnegative size-dependent API
cost. For BO sync it is **not** asserted to be inverse physical DDR bandwidth.
Parameters require explicit units, state, runtime identity, measured byte domain,
raw provenance and timing boundary. Unresolved slopes can remain zero/unidentified;
out-of-domain or unmeasured work remains unknown. API-sum predictions and outer
loop wall residuals are distinct. Host scheduling/observer gaps are not silently
assigned to DMA.

NPU wait remains a composite of compute, internal DMA, stalls, device/runtime
progress and the actual wait backend. Internal DMA is descriptive and covered by
that composite, so it cannot be added again. CPU kernel calls likewise include
CPU memory accesses. Without separately supported observations these items remain
uncalibrated, full-program cost is null, and the calibrated subtotal is reported
as a partial cost. CGRA continues to use the unchanged ML II model and original
conversion/scheduler/network boundary. No SIMD/VLIW simulator is added.

## Evidence and current limits

The initial heterogeneous smoke and 210 independent training outputs passed at
unchanged FP32 atol=rtol=1e-5, max absolute error 2.384185791015625e-7. Formal
training comprises three seeds, two warmups and five hot calls per candidate.
Two-round search and 18 new holdout compilations were saved before the priority
change; those new candidates were not sent to hardware and are not accepted
performance-search results. The initial model is historical, not the new default
communication model.

Static support excludes arbitrary aliases/inout ABIs, asynchronous overlap,
tiled child-to-parent heterogeneous refinement and arbitrary cross-device
coherence. Original standalone AIR tiling and Amoeba paths remain available.
Amoeba's physical SRAM remains unknown where the original model does not specify
it; it is not inferred from the NPU payload policy.

Raw results live in the companion ignored output directory
`results/taskflow/program-graph-heterogeneous-v13/`. All reproduction uses new
output directories. See the companion mechanism report for frozen experimental
protocol, API samples, parameters, holdout errors and reproduction commands.

## Mechanism implementation and focused acceptance

`MechanismCost.h/.cpp` now extracts a separate, source/graph-bound mechanism view
from the checked ProgramGraph and explicit installed bound-BO contract. It records
per-API storage/copy/event correspondence, conservative output sync, data-state
progress, required opaque items and native-DMA/CPU-access ownership. Host copy
worker and dispatch labels explicitly bind existing `CPU.compute` capacity; they
do not invent DRAM ports. The current schedule is strictly serial.

`exportHeterogeneousProgram` always exports `mechanism-facts.json` and
`mechanism-cost.json`. Target configuration can name a `mechanism_parameters`
file. Exact runtime/path/state/resource/boundary and byte-domain matching is
required. An unknown full mechanism cost takes precedence over any optional
historical regional prediction: `cost.estimated_ms` stays null, while
`historical_empirical_cost` preserves the old baseline separately. Search cannot
silently rank the partial subtotal as a full program score.

Focused acceptance completed: 58 C++ mechanism assertions, 56 export checks,
225 heterogeneous mapping assertions, 43 complete Core/search/AIR regression
groups, 35 Amoeba lit tests and the unchanged v12 Amoeba/model differential suite.
The independent Core binary has no Neura/AIR/AIE/XRT link dependency. Historical
failed builds, the initially mismatched diagnostic and the fixture-only lit
failure remain recorded beside their corrected passing runs.

The primary mechanism epoch checks 420 training and 1134 heldout full program
outputs (1554 total, three seeds), all within original FP32 tolerances. The
independent auditor reproduces all 18 nonnegative least-squares parameters and
bridges 1110 hot calls to the reexported C++ mechanism plans. Ten reexports retain
identical parent graph/plan IDs and byte-identical actual Taskflow, CPU MLIR and
AIR relative to their frozen executable packages. Metadata reexport is not an
additional hardware run. A second, clean-shell full reproduction also completed;
it is a separately calibrated repeat, not a replacement for the primary freeze.

**Predictive accuracy is not accepted as uniformly reliable.** The primary
96-cell heldout API MAPE is 29.10%; the repeated-TO row is particularly bad (up to
436.74%). A single 714691ns training API span contributes 71.5% of that row's
recorded time; its NNLS alpha is 1983.24ns versus a 450ns training median. All
samples and parameters remain unchanged. No measured cause is asserted and no
heldout-driven refit is performed. First raw sync states have approximately
3–6% MAPE; actual program start, copies and output sync transfer less accurately.
Matched public API subtotals differ by 1.73–25.64%; these are not full program
prediction errors. CPU call, wait, unmeasured visibility states and additional
runtime overhead remain unknown, and expanded performance search stays paused.

Preparation was separately observed in 30 constructor trials with zero kernel
submissions. Across these trials median `pyxrt.hw_context` host API time is
0.8125ms; ELF construction is 0.06984ms and `ext.kernel` 0.01820ms. Full Prepared
construction additionally verifies file/tool hashes and creates views/libraries;
it is much larger and cannot be charged as device configuration. Timing the
constructor does not isolate firmware/PDI work. The installed polling wait also
prevents treating host wait as pure device execution.

The primary program benchmark holds two prepared NPU contexts in training and
eight in validation; their aggregate distinct system BO payloads are 15360B and
56320B. Those numbers are system allocations across simultaneously prepared test
cases, not per-program SRAM. Raw API timing stratified by same/different previous
context shows no transferable universal switch constant. This resource/state
migration is an applicability limit, not a fitted physical configuration cost.

Reproduce mechanism calibration and heldout verification from the companion:

```sh
env -i HOME="$HOME" PATH=/usr/bin:/bin AIE_TOY_ROOT=/home/shiran/project/aie-toy \
  bash ../orbit-artifact/taskflow/scripts/run_mechanism_model_v13.sh all NEW_OUTPUT \
  --protocol ../orbit-artifact/results/taskflow/program-graph-heterogeneous-v13/mechanism-protocol-r2/protocol.json \
  --bindings ../orbit-artifact/results/taskflow/program-graph-heterogeneous-v13/mechanism-bindings-r1/bindings.json
```

This reuses the exact frozen packages/toolchain/runtime in the protocol and
requires those dependencies. It compiles no new kernels and expands no search.
Use `validate-only` for import-blocked CPU checks. Public model export remains
C++/MLIR:

```sh
TASKFLOW_OPT SOURCE.mlir --target-config=CONFIG.json --heterogeneous-output=NEW_PLAN
```

`CONFIG.json` contains the chosen existing `heterogeneous.task_devices/cuts`,
`execution_contract` and `mechanism_parameters` file paths. Main evidence lives
under companion `results/taskflow/program-graph-heterogeneous-v13/`: primary
`mechanism-model-freeze-r1`, `mechanism-independent-holdout-audit-r5`,
`mechanism-public-plans-r2`, `mechanism-preparation-r1`,
`mechanism-state-diagnostics-r1.json` and `mechanism-clean-reproduction-r1`.
