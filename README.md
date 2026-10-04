# ORBIT

ORBIT is a source-owned C++ MLIR compiler workflow for joint task graph rewriting and CGRA orchestration. It builds on AMOEBA and Neura. The companion [orbit-artifact repository](https://github.com/guosran/orbit-artifact) provides the six-program fixed input-0 cumulative ablation, exact experiment inputs and costs, numerical reference code, and reproduction instructions.

This source snapshot includes complete source-iteration-domain capture and partition verification for the corrected input-0 experiments. The required Neura compiler is vendored, including the local fixes used by the experiments. The model weights are included in the companion artifact and evaluated by ORBIT C++. Their upstream origins and revisions are listed in `UPSTREAM_PROVENANCE.json`; the vendored source retains its upstream notices.

The `search-joint-neighborhood` MLIR pass owns complete-program local actions, materialization, legality, canonical deduplication, beam/archive state, production scheduling with explicit communication, checkpoint recovery, and global predicted top five selection. The artifact Python tools launch and validate those C++ results.

The paper-aligned experiment uses the AMOEBA-Test 4×4 CGRA fabric with 2×2 PEs per CGRA and the `orbit-2x2-predictor` four-member direct model. Physical mapper shapes are derived from the architecture's per-CGRA dimensions. The legacy 4×4-PE model remains a separate schema and namespace. The new model's candidate-only status and explicit out-of-domain queries are preserved.

For original Ray, every permitted Task_13 shape has an II lower bound above the new model ceiling. The artifact reports that program as out of model domain in the main table and runs source-owned fission as a separate supplement. Communication is supplied through an exact-bound `--joint-inter-task-network-spec` file so the original physical architecture YAML remains unchanged.

## Build

Use LLVM/MLIR revision `6146a88f60492b520a36f8f8f3231e15f3cc6082`, built with MLIR, RTTI and MLIR Python bindings enabled. The CMake configuration also needs Python, NumPy, pybind11 and nanobind. The companion artifact's [build instructions](https://github.com/guosran/orbit-artifact/blob/main/docs/INPUT0_NEIGHBORHOOD_REPRODUCIBILITY.md) include the LLVM configuration and runner tools.

```sh
export ORBIT_SRC="$PWD"
export LLVM_BUILD="${LLVM_BUILD:?set LLVM_BUILD to the pinned LLVM/MLIR build}"
export ORBIT_BUILD="${ORBIT_BUILD:-$ORBIT_SRC/build}"
python3 -m pip install numpy pybind11==3.0.1 nanobind==2.4.0
cmake -S "$ORBIT_SRC" -B "$ORBIT_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm"
ninja -C "$ORBIT_BUILD" tools/mlir-amoeba-opt/mlir-amoeba-opt -j1
export ORBIT_OPTIMIZER="$ORBIT_BUILD/tools/mlir-amoeba-opt/mlir-amoeba-opt"
```

Use the companion artifact's input-0 reproduction instructions to configure the model, costs, static input programs and native reference libraries. The official cumulative stages are shape with fixed dispatch, shape with critical-path dispatch, then replica, tiling and fusion. Results are budgeted best-found scheduler cycles with real mapper II, independent trace and numeric evidence; they do not claim exhaustive optimality or RTL measurements.
