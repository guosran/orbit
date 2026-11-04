#!/usr/bin/env bash
# Native compiler acceptance; only the optional all mode opens the NPU.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 2 && ( "$1" == cpu || "$1" == all ) ]] || { echo 'usage: run_taskflow_direct_air.sh cpu|all NEW_OUTPUT_DIRECTORY' >&2; exit 2; }
mode=$1
out=$(realpath -m "$2")
[[ ! -e "$out" && "$out" != *[[:space:]]* ]] || { echo 'Use a new output path without whitespace.' >&2; exit 2; }
mkdir -p "$out"
exec > >(tee "$out/reproduce.log") 2>&1
cd "$root"
python_bin=${ORBIT_TEST_PYTHON:-/home/shiran/project/aie-toy/.venv/bin/python}
llvm_build=${LLVM_BUILD:-/home/shiran/project/llvm-project/build}
bash scripts/run_taskflow_static_cpp.sh "$out/static-regression"
compiler_c=$(sed -n -E 's/^CMAKE_C_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)
compiler_cxx=$(sed -n -E 's/^CMAKE_CXX_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)
cmake --fresh -S . -B build-air -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$llvm_build/lib/cmake/mlir" -DLLVM_DIR="$llvm_build/lib/cmake/llvm" \
  -DCMAKE_C_COMPILER="$compiler_c" -DCMAKE_CXX_COMPILER="$compiler_cxx" \
  -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so -DZLIB_INCLUDE_DIR=/usr/include \
  -DORBIT_ENABLE_NEURA=OFF -DORBIT_ENABLE_AIR=ON
cmake --build build-air --target taskflow-opt -j2
cmake --build build-core --target taskflow-generic-mapping-test -j2
"$python_bin" test/TaskflowGeneric/analysis.py --output "$out/analysis"
build-core/tools/taskflow-opt/taskflow-generic-mapping-test \
  test/TaskflowAIR/fixtures/add-scale.mlir > "$out/mapping-checks.json"
"$python_bin" test/TaskflowGeneric/search.py --output "$out/native-search"
"$python_bin" test/TaskflowAIR/lowering.py \
  --tool "$root/build-air/tools/taskflow-opt/taskflow-opt" --output "$out/air-validation"
build-air/tools/taskflow-opt/taskflow-opt test/TaskflowGeneric/add.mlir \
  --air-pass-pipeline --air-output="$out/pass-example.air.mlir"
if [[ "$mode" == all ]]; then
  # The hardware wrapper owns locking inside execution; never wrap in flock.
  bash scripts/run_taskflow_direct_air_hardware.sh \
    --tool "$root/build-air/tools/taskflow-opt/taskflow-opt" \
    --output "$out/hardware" --approx-exp
fi
printf 'Acceptance completed: %s\n' "$out"
