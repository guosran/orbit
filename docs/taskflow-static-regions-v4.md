# Native region implementations and measured cost boundaries

The production compiler reads actual Taskflow MLIR bodies. `StaticAnalysis.cpp`
constructs typed steps, affine access regions, storage identities and dependency
facts. `StaticContract.cpp` reads target descriptions; `StaticRegion.cpp` matches
supported implementations and searches bounded exact covers. The independent
`StaticRegionChecker.cpp` reconstructs semantic versions, access dependencies,
coverage, resources and costs without invoking the region matcher or mapper.
The shared types are in `include/Taskflow/StaticMapping.h`.

## Implementations and semantics

A region implementation records its version, pattern, execution mode, covered
steps, external reads and writes, boundary memory, layout, dtype, numerical
contract, prepared state, model identity, calibration provenance and domain,
resource demands, and temporary workspace. Serialization uses stable step and
storage identities. Compiler analysis retains actual MLIR operation associations.
No XRT, BO, ELF or reconfiguration operation enters public Taskflow semantics.

The current measured subset supports `matmul_single` (zero fill plus a full-K
matmul) and `matmul_pair` (two such matmuls), with immutable input weights.
Pair implementations are either independent submissions or the existing
NPUChain. Singleton mode is independent. Hidden pair intermediates must have no
external reader, writer or public output. Unsupported aliasing, partial-K,
layout, numerical contract and shape requirements are rejected explicitly.

The cover search assigns every source operation exactly once. Mixed covers are
allowed. Contracting the dependency graph must not introduce a cycle. External
RAW/WAR/WAW dependencies survive contraction. Thus an M-partition candidate can
select a singleton, a pair and a singleton when two pair regions would form an
illegal cyclic cover. N-partition uses three singletons in the measured case.
A bounded search reports incomplete enumeration honestly; unmeasured or
unsupported alternatives receive no fabricated score.

## Cost and storage boundary

Measured costs use `opaque_warm_host_to_host`: FP32 host activation conversion,
submission, necessary synchronization, BF16 output readback and FP32 conversion
are included. Compilation, context creation and initial immutable-weight upload
are excluded and recorded separately. This is a complete region wall cost,
not compute-only time. `region_wall_ms` remains distinct from primitive compute,
copy and synchronization totals. Covered operations and internal communication
are not generated or charged again.

External host data dependencies and host buffer lifetimes remain explicit.
Each opaque event reserves its declared CPU/NPU/DMA resources for its entire
interval because no measured internal timeline is available. Reserving a host
worker is not a claim that the worker computes during a device wait. Workspace
amounts are declared conservative planning assumptions, not measured physical
scratch peaks or evidence of AIE SRAM residency. Layout assembly between
multiple regions adds real host work not fully explained by the current
standalone-region interpolation; the structural comparison reports that error.

The model is a finite-domain bilinear interpolation in M and H, fitted only to
four predeclared training corners. K/N and kernel legality restrictions are
part of the contract. Singles have their own measured models; pair wall costs
are never divided into fictitious compute/launch constants. Queries outside the
domain, without a cost, or with the wrong warm/cold state are unsupported.
The old theoretical primitive catalogue remains available and labeled theoretical.

## Search, validation and execution

Both `taskflow-opt --search --backend=static` and the original
`search-joint-neighborhood` structured path call the native static evaluator.
A region catalogue participates in candidate evaluation after actual structural
materialization. Model/configuration/file/tool identities bind search recovery;
changes reject a stale checkpoint. The executor first invokes the native checker,
then dispatches exactly the selected existing implementation. Unsupported plans
are not silently changed into another execution mode.

`test/TaskflowMapping/regions.py` checks independent small-cover enumeration,
resigned illegal coverage, external consumers/dependencies, duplicate charges,
location/layout/precision/state, resource/workspace capacity, missing/domain
costs and model identities. Existing native primitive tests remain regression layers. Historical Python
differential evidence is retained; the Python mapper/checker is no longer active. Python calibration and experiment
scripts never choose a production mapping.

## Reproduction and evidence

From a clean shell, using new output directories:

```bash
cd /home/shiran/project/orbit
bash scripts/run_taskflow_static_cpp.sh "$PWD/build/repro-static-v4"
/home/shiran/project/aie-toy/.venv/bin/python test/TaskflowMapping/regions.py \
  --output build/repro-regions-v4
python3 test/TaskflowMapping/calibrated_original_search_v4.py \
  --output build/repro-calibrated-search-v4
```

The measured search command uses retained frozen calibration assets. Their
reconstruction and hardware replay commands are in
[calibration and execution](taskflow-region-calibration-v4.md).
[Trace documentation](taskflow-trace-v4.md) describes the separately checked
hardware-event evidence. The complete local result index is
`build/static-mapping-v4/COMPLETED-RESULTS.md`.

Real FFN M=5/128 static planning is supported under the theoretical primitive
catalogue; current measured-region calibration does not cover its projections
or CPU SwiGLU. Such queries remain explicitly unscored. Real FFN trace execution
is numerical and event evidence, not a calibrated whole-FFN prediction claim.
