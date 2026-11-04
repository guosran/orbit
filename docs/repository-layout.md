# Compiler and artifact repositories

The compiler is `guosran/orbit`. Its companion is **`guosran/orbit-artifact` (singular)**. Compiler implementation and small correctness fixtures belong to the former; reproducible experiments and their evidence belong to the latter.

## Source ownership

ORBIT keeps Taskflow bodies and semantics, C++/MLIR analysis, structural transforms, mapping and communication plans, independent checkers, search, and Amoeba/AIR backends. `test/Support` contains only standard-library fixtures and path helpers. The Core suite can run with the artifact repository unavailable and with Neura/AIR disabled.

The artifact's `taskflow/` namespace keeps the migrated execution, measurement, fixture/reference generation, hardware audits, calibration, model-regression and report tools. It uses the compiler through explicit executable paths. It does not implement a replacement planner. Existing input-0 material in the artifact repository remains in its original `reference/`, `diagnostics/`, `config/`, `scripts/` and `tests/` namespaces.

## Current paths

With sibling checkouts:

```text
project/
  orbit/                         compiler sources and correctness tests
  orbit-artifact/
    taskflow/{scripts,python,test,docs}/
    reference/                   existing models and workloads
    builds/taskflow/{core,air,amoeba}/
    results/taskflow/            full local runs (Git-ignored)
    .work/legacy/                frozen historical build caches
    .work/source-snapshots/      version-bound development snapshots
```

`ORBIT_SRC` selects the compiler checkout. `ORBIT_BUILD_ROOT` selects the directory containing `core/`, `air/` and `amoeba/`. `ORBIT_RESULTS_ROOT` selects the experiment result root. Artifact Python and shell entry points share the same defaults. Results are never implicitly written into the compiler source checkout.

Historical CMake caches are archives with absolute paths; a fresh reproduction configures a new build rather than editing those caches. On the original experiment host, old `orbit/build*`, `orbit-data-v8` and `aie-toy/results/orbit*` paths are compatibility symlinks to the companion's local storage. They are ignored local aliases, not published compiler source or the defaults for new runs. They keep absolute references in immutable logs, manifests and frozen runners usable.

The migration does not rewrite original plans, commands, golden model output, source snapshots, numerical arrays or timing logs. An independent migration receipt records old/new locations and hash checks. Eighty pre-move identity anchors and offline audits of the existing candidate and M128 outputs were verified after the move. Full local archives and build products are excluded from Git; source, small golden fixtures and portable reports are versioned in the appropriate repository.

## Reproduction

Compiler-only: `scripts/build-core.sh BUILD_DIR`, or `python3 test/run_core_regressions.py --tool TOOL --output NEW_DIR`.

Experiments: use [the companion Taskflow entry points](https://github.com/guosran/orbit-artifact/tree/main/taskflow/scripts) and [validation instructions](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-data-organization-v8-validation.md). The existing [input-0 reproduction](https://github.com/guosran/orbit-artifact/blob/main/docs/INPUT0_NEIGHBORHOOD_REPRODUCIBILITY.md) is preserved.
