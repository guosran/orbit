# Native CP-SAT search and independent NPU shards (v22)

The optional OR-Tools C++ backend composes bounded local residency modes with a
common global tile, then checks the resulting full program using the existing
native planner and independent checker. It also places independent outer-axis
shards on concrete NPU compute tiles and models input/compute/output intervals
on explicit unary DMA resources. Python is limited to fixtures, independent
oracles, orchestration and measurement.

## Build

The default remains `ORBIT_ENABLE_ORTOOLS=OFF`; it has no OR-Tools link or runtime
dependency. `taskflow-cpsat-opt` returns `UNAVAILABLE` in that build. To enable:

```sh
cmake -S . -B NEW_BUILD -G Ninja \
  -DMLIR_DIR=LLVM_BUILD/lib/cmake/mlir -DLLVM_DIR=LLVM_BUILD/lib/cmake/llvm \
  -DORBIT_ENABLE_NEURA=OFF -DORBIT_ENABLE_AIR=ON \
  -DORBIT_ENABLE_ORTOOLS=ON -DCMAKE_PREFIX_PATH=ORTOOLS_CPP_SDK
cmake --build NEW_BUILD --target taskflow-opt taskflow-cpsat-opt -j2
```

The validated SDK is Google's official Linux C++ OR-Tools v9.15.6755 package,
`or-tools_amd64_ubuntu-22.04_cpp_v9.15.6755.tar.gz`, SHA256
`0b30114d7c05f0596286bf3ef8d02adcf5f45be3b39273490e6bb74a2a9bd1ea`.
Set the prefix to the extracted SDK root so its bundled dependencies are found.
No toolchain files, old builds or original Amoeba/Neura drivers are replaced.

## Reuse and tiling

Set `native_auto_candidates.algorithm` to `cp-sat` in the existing target
configuration. Existing limits bound tile trials, scope sets, generation
attempts, local catalogue size, planner work, raw cost calls, compute instances
and wall time. Discovery considers single local groups, rather than enumerating
all group combinations. CP-SAT selects an exact cover, a common tile, exclusive
retained storage, and ranks that prohibit cycles in the contracted dependency
graph. The original source graph must itself be acyclic.

Costs are signed deltas against a fully priced baseline for each tile. The
objective is an additive surrogate: composition can change nonadditive backend
costs. Every composed candidate receives complete planning and checking; if it
fails or is worse than an already fully priced baseline/local mode, the checked
incumbent is retained. `objective_exact_for_backend` stays false, including when
CP-SAT proves the supplied surrogate optimal. Native capacities remain subject
to the full planner/checker; zero resource coefficients in a local pattern are
not a proof of composed capacity feasibility.

`automatic-proposals.json` separates local discovery, pricing, model building,
solve and final recheck time. `search-stats.json` reports process-wide raw work,
wall-budget overshoot and the stop reason. CP-SAT process quotas include resume
revalidation; this mode does not restore the v21 staged-policy budget ledger.
A planner call is nonpreemptive. The benchmark therefore adds an external process
watchdog, rather than treating CP-SAT's solve time limit as a whole-search limit.

## Physical NPU realization

```sh
taskflow-opt SOURCE --lower-multicore-to-air=NEW_DIR --npu-shards=2 \
  --target-config=CONFIG
# Compile NEW_DIR/program.air.mlir with the installed aircc, then print the
# final npu.input.air.mlir using air-opt --mlir-print-op-generic.
taskflow-opt SOURCE --check-multicore-aie=FINAL_GENERIC_AIE --npu-shards=2 \
  --target-config=CONFIG --implementation-output=RECEIPT
```

The current subset requires a proven independent first-axis partition of every
storage, one complete local execution group per shard, static pools and direct
shim/compute streams. AIR herds carry physical `x_loc/y_loc` pins. Final AIE is
checked against those selected coordinates, complete host output coverage,
static SRAM/BD resources, core arithmetic and addresses, lock/DMA causality,
stream ownership and runtime task lifecycle. Cross-owner SSA and symbolic
references are checked before isolating each core for the existing checker.
Unknown operations and packet/memtile/cross-core topology fail closed.

The scheduling model uses explicit assumed durations and static whole-program
SRAM/BD occupancy. Its timestamps are a planning result; the runtime implements
readiness and resource arbitration, not timed dispatch. No measured overlap or
calibrated device makespan is claimed. The compiler accepts up to four requested
shards; the validated direct-stream hardware cases currently use one and two
cores. AIR may introduce packet routing for larger stream counts; that topology
is deliberately not certified by this checker.

## Reproduction and timing growth

The companion `taskflow/scripts/run_taskflow_cpsat_v22_cpu.sh CORE_OFF_BUILD CPSAT_AIR_BUILD NEW_OUTPUT` runs
independent finite oracles, native integration contracts, default-off dependency
checks and three serialized scaling campaigns. The exact-cover campaign varies
8–256 tasks and 1/4/8 tiles. The offline schedule campaign varies 4–64 shards and
2/4/8 tiles with shared DMA; its time-multiplexed placement is broader than the
current independent-shard lowering. Both use 0.1 s and 1 s outer budgets and
record model/solve/process/verification time, bounds, gap, branches, conflicts,
peak process RSS, timeout and independently validated incumbent availability.
Identical problems share a hash across time budgets.

The native scaling campaign additionally measures bounded local discovery,
full-program pricing and checking for 8/16/32/64/128-task chains. Its 5 s soft
whole-search limit has an 8 s external watchdog. Scaling results are workload
samples, not a polynomial-time guarantee: exact covering and resource scheduling
can have combinatorial growth. Model size limits and a checked incumbent make
that growth operationally bounded.

The artifact repository's `taskflow/scripts/run_taskflow_cpsat_v22.py` compiles
one/two-core chains and a fork/join graph, validates final AIE before submission,
then optionally executes three fresh-process trials in alternating order with
three input seeds each, two warmups and five hot samples. Full outputs are
compared with source-rounded FP32 and independent FP64 references at the existing
`atol=rtol=1e-5`. Host timings include the existing runner's wait-checkpoint
serialization, so they must not be interpreted as isolated device latency.

Accepted measurements and preserved failures are in the companion [v22 validation report](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-cpsat-v22-validation.md). Native 8–128 task samples take 1.22–4.78 seconds under the bounded domain; their actual solve portion is 5.7–17.9 ms. Synthetic larger domains often return incomplete results. Correct one/two-core execution is established; stable measured speedup is not.
