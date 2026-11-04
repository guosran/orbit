#!/usr/bin/env bash
# Native C++ compiler acceptance. Python is only an experiment/test harness.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# == 1 ]] || { echo 'usage: run_taskflow_static_cpp.sh NEW_OUTPUT_DIRECTORY' >&2; exit 2; }
output="$(realpath -m "$1")"
[[ ! -e "$output" && "$output" != *[[:space:]]* ]] || { echo 'Output must be new and have no whitespace.' >&2; exit 2; }
llvm_build="${LLVM_BUILD:-/home/shiran/project/llvm-project/build}"
python_bin="${ORBIT_TEST_PYTHON:-/home/shiran/project/aie-toy/.venv/bin/python}"
llvm_cc="$(sed -n -E 's/^CMAKE_C_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)"
llvm_cxx="$(sed -n -E 's/^CMAKE_CXX_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)"
mkdir -p "$output/validation"
exec > >(tee "$output/reproduce.log") 2>&1
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release "-DMLIR_DIR=$llvm_build/lib/cmake/mlir" "-DLLVM_DIR=$llvm_build/lib/cmake/llvm"
  "-DCMAKE_C_COMPILER=$llvm_cc" "-DCMAKE_CXX_COMPILER=$llvm_cxx"
  -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so -DZLIB_INCLUDE_DIR=/usr/include)
cmake -S "$root" -B "$root/build-core" "${common[@]}" -DORBIT_ENABLE_NEURA=OFF -DORBIT_ENABLE_AIR=OFF
cmake --build "$root/build-core" --target taskflow-opt check-taskflow-core -j2
cmake -S "$root" -B "$root/build-amoeba" "${common[@]}" -DORBIT_ENABLE_NEURA=ON -DORBIT_ENABLE_AIR=OFF -DNEURA_ENABLE_PYTHON_BINDINGS=OFF
cmake --build "$root/build-amoeba" --target mlir-amoeba-opt -j2
export PATH="$llvm_build/bin:$PATH"
cd "$root"
"$python_bin" scripts/taskflow_static_experiment.py prepare --output "$output"
"$python_bin" scripts/taskflow_static_experiment.py plan --output "$output"
"$python_bin" test/TaskflowMapping/cpp_differential.py --tool "$root/build-core/tools/taskflow-opt/taskflow-opt" --output "$output/validation/differential"
"$python_bin" test/TaskflowMapping/native_search.py --output "$output/validation/native" --workloads "$output/workloads"
"$python_bin" test/TaskflowValidation/search_contract.py --tool "$root/build-core/tools/taskflow-opt/taskflow-opt" --output "$output/validation/legacy-search"
"$python_bin" test/TaskflowValidation/evaluation_cache.py --output "$output/validation/legacy-cache"
"$python_bin" test/TaskflowMapping/search_preferences.py --output "$output/validation/preferences" --source "$output/workloads/ffn_m5/source.json"
"$llvm_build/bin/llvm-lit" -sv -j2 "$root/build-amoeba/test"
"$python_bin" scripts/taskflow_static_cpp_report.py --output "$output"
