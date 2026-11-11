# ORBIT

ORBIT is a C++/MLIR framework for joint optimization of an entire program's task graph and execution plan. Search combines structural transformations, resource configuration, temporal scheduling and data organization. Amoeba/Neura and direct AIR/NPU backends provide target constraints, implementation choices and cost feedback.

This repository contains the compiler and its correctness tests. The companion [orbit-artifact repository](https://github.com/guosran/orbit-artifact) contains experimental workloads, models, execution/measurement tools, results and reproduction instructions, including the [Taskflow/AIR experiments](https://github.com/guosran/orbit-artifact/tree/main/taskflow).

## Repository boundary

| ORBIT compiler | Companion artifact |
|---|---|
| `include/`, `lib/`, `tools/`: dialects, analysis, transformations, mapping, checkers, search and backends | `taskflow/scripts/`, `taskflow/python/`: fixture generation, execution, measurement and reporting |
| `test/`: compiler semantics, plan legality, search recovery and small code-generation fixtures | `taskflow/test/`: hardware audits and experiment/model checks |
| `docs/`: compiler interfaces, supported semantics and design | `taskflow/docs/`: validation reports, timing boundaries and reproduction |
| Vendored Neura compiler source and upstream notices | `reference/`, `config/`, `diagnostics/`: model bundles and historical experiment material |

Builds, generated IR, raw arrays, logs and full hardware archives live outside the compiler checkout. See [the layout and migration contract](docs/repository-layout.md). Historical result files retain their exact bytes and original version identities.

## Independent Core build and tests

Use LLVM/MLIR revision `6146a88f60492b520a36f8f8f3231e15f3cc6082` or the compatible pinned installation described in the artifact's [build instructions](https://github.com/guosran/orbit-artifact/blob/main/docs/INPUT0_NEIGHBORHOOD_REPRODUCIBILITY.md). The build helper reuses the compiler recorded in that LLVM CMake cache.

```sh
export LLVM_BUILD=/path/to/llvm-project/build
bash scripts/build-core.sh /path/outside/orbit/core-build
```

This builds with `ORBIT_ENABLE_NEURA=OFF` and `ORBIT_ENABLE_AIR=OFF`, then runs the compiler-only suite with Python's standard library. It does not import the artifact runtime, NumPy, AIR, AIE or XRT. Without a build-directory argument, the default is `../orbit-artifact/builds/taskflow/core`; `ORBIT_BUILD_ROOT` overrides the parent containing `core/`, `air/` and `amoeba/`.

To test an already built compiler:

```sh
python3 test/run_core_regressions.py \
  --tool /path/to/core-build/tools/taskflow-opt/taskflow-opt \
  --output /path/outside/orbit/NEW_TEST_OUTPUT
```

`--amoeba-tool` and `--air-tool` opt into their compiler/code-generation tests. Runtime experiments and real-model regression belong to the companion artifact.

## Backends and experiments

```sh
# Keep the two source repositories beside each other.
git clone https://github.com/guosran/orbit.git
git clone https://github.com/guosran/orbit-artifact.git
export ORBIT_SRC="$PWD/orbit"
export ORBIT_BUILD_ROOT="$PWD/orbit-artifact/builds/taskflow"
bash orbit-artifact/taskflow/scripts/run_taskflow_data_organization_v8.sh \
  cpu "$PWD/orbit-artifact/results/taskflow/NEW_CPU_RUN"
```

The `all` mode also runs the documented NPU comparison and full M5 FFN regression. Output directories must be new. LLVM/AIR/AIE/XRT and model prerequisites, original M128 acceptance, numerical contracts and timing scopes are recorded in the [artifact validation report](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-data-organization-v8-validation.md).

The original `search-joint-neighborhood` pass retains source-owned graph actions, legality, canonical deduplication, archive/beam state, production scheduling and checkpoint recovery. Common Taskflow analysis and mapping add regions, data versions, storage copies, lifetimes and transfers; direct AIR consumes the checked decisions. See [public data organization](docs/taskflow-data-organization-v8.md), [mapping](docs/taskflow-program-mapping.md), [AIR lowering](docs/taskflow-direct-air-lowering.md), and [the CGRA model boundary](docs/taskflow-cgra-model-boundary-v8.md).

Model weights remain in the artifact and are evaluated by the existing C++ implementation. The legacy 4×4-PE and per-CGRA-2×2 models retain distinct contracts and namespaces. Upstream revisions and notices are recorded in [UPSTREAM_PROVENANCE.json](UPSTREAM_PROVENANCE.json).

## Optional native constraint optimization

The [CP-SAT v22 interface](docs/taskflow-cpsat-v22.md) adds bounded reuse/tiling composition and independently checked NPU2 shard placement using optional C++ OR-Tools. The default Core build keeps this dependency off. [Validation and solver-time scaling](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-cpsat-v22-validation.md) distinguish finite surrogate optimality, full planner costs and actual hardware behavior.

The [public ProgramGraph joint solver (v23)](docs/taskflow-program-joint-v23.md)
puts partitioning, local grouping/copies, placement, transfer routes and resource
scheduling in one source-bound model. Its current executable subset is static
pointwise programs with explicit host communication between configurations;
GEMM/attention and arbitrary cross-core residency are outside that joint path.
A checked baseline and model/time limits bound incomplete solves. The companion
[v23 validation](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-program-joint-v23-validation.md)
records independent oracle cases, actual AIE/NPU checks and scaling limits.
