# Taskflow backend boundary

Implementation owners: Core owns Taskflow dialect, common C++ structures,
materializer, verifier and CMake; Amoeba owns Neura adapter and its optimizer;
NPU owns the device adapter and all device execution; integration owns Python
fixtures, independent numerical reference, experiment driver and reporting.
The sole hardware owner serializes execution with `/tmp/npu.lock`.

The restricted exchange schema is `orbit.taskflow.plan.v1`. It records buffers
(storage id, rank-two static shape, f32 semantic dtype, role), tasks (id,
completion dependencies, ordered operations), and output storage ids. Supported
operations are zero fill, accumulating matrix multiply, ReLU, and standard
SiLU(gate)*up. Matrix multiplication uses global half-open m/n/k intervals.
Completion dependencies perform no numerical addition. Sequential K chunks
accumulate into the same initialized output. Resource allocations and physical
replication do not belong to this semantic exchange.

`taskflow-opt` materializes real Taskflow tasks with structured computation
bodies. Its MLIR witness validation binds the exchange to the executable body;
an edited body or stale witness must fail. JSON is a restricted interchange and
fixture format, not a second general computation language. Dynamic shapes,
aliasing, arbitrary control flow, parallel K reductions and arbitrary fusion
are outside this initial subset and must fail closed.

Backends consume the same materialized candidate. Target plans state actual
implementation, precision, host/device placement, layout and compilation
configuration. Hardware cost measurements use milliseconds and never overwrite
Neura II/cycle estimates. CPU diagnostic interpretation is labeled CPU; only
successful XRT execution plus numerical verification is device success.

The independent oracle in `scripts/taskflow_common.py` computes whole-matrix
float64 equations from seeded inputs without consulting candidate slices or
task boundaries. The separate CPU diagnostic executes candidate slices and
checks against that oracle. NPU precision tolerances must additionally account
for its recorded BF16/BFP implementation.
