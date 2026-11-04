#!/usr/bin/env bash
# Build and test the compiler without the experiment repository's Python code.
set -euo pipefail
core_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# -le 1 ]] || { echo 'usage: build-core.sh [BUILD_DIRECTORY]' >&2; exit 2; }
build_root=${ORBIT_BUILD_ROOT:-"$core_root/../orbit-artifact/builds/taskflow"}
build_dir=$(realpath -m "${1:-$build_root/core}")
llvm_build=${LLVM_BUILD:?Set LLVM_BUILD to the LLVM/MLIR build directory}
llvm_cc=$(sed -n -E 's/^CMAKE_C_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)
llvm_cxx=$(sed -n -E 's/^CMAKE_CXX_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)
cmake -S "$core_root" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$llvm_build/lib/cmake/mlir" -DLLVM_DIR="$llvm_build/lib/cmake/llvm" \
  -DCMAKE_C_COMPILER="$llvm_cc" -DCMAKE_CXX_COMPILER="$llvm_cxx" \
  -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so -DZLIB_INCLUDE_DIR=/usr/include \
  -DORBIT_ENABLE_NEURA=OFF -DORBIT_ENABLE_AIR=OFF
cmake --build "$build_dir" --target taskflow-opt taskflow-generic-mapping-test \
  taskflow-tiled-mapping-test taskflow-program-mapping-test \
  taskflow-program-partition-test taskflow-data-organization-test \
  -j "${ORBIT_BUILD_JOBS:-4}"
mkdir -p "$build_dir/test-output"
run_dir=$(mktemp -d "$build_dir/test-output/core.XXXXXX")
python3 "$core_root/test/run_core_regressions.py" \
  --tool "$build_dir/tools/taskflow-opt/taskflow-opt" --output "$run_dir/results"
echo "Compiler regression evidence: $run_dir/results"
