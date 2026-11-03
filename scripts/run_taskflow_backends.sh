#!/usr/bin/env bash
# Clean-shell entry point for the installed, pinned LLVM and NPU toolchains.
set -euo pipefail
orbit_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llvm_build="${LLVM_BUILD:-/home/shiran/project/llvm-project/build}"
toy_root="${AIE_TOY_ROOT:-/home/shiran/project/aie-toy}"
stage="${1:-cpu}"
if (($#)); then shift; fi
case "$stage" in cpu|all|npu|report|amoeba) ;; *) echo 'usage: run_taskflow_backends.sh [cpu|all|amoeba|npu|report] [experiment options]' >&2; exit 2;; esac
# Compiler ABI must match the LLVM build (the local LLVM uses GCC 15.2).
llvm_cc="$(sed -n -E 's/^CMAKE_C_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)"
llvm_cxx="$(sed -n -E 's/^CMAKE_CXX_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)"
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release
  "-DMLIR_DIR=$llvm_build/lib/cmake/mlir" "-DLLVM_DIR=$llvm_build/lib/cmake/llvm"
  "-DCMAKE_C_COMPILER=$llvm_cc" "-DCMAKE_CXX_COMPILER=$llvm_cxx"
  -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so -DZLIB_INCLUDE_DIR=/usr/include)
if [[ "$stage" == cpu || "$stage" == all || "$stage" == amoeba ]]; then
  cmake --fresh -S "$orbit_root" -B "$orbit_root/build-core" "${common[@]}" -DORBIT_ENABLE_NEURA=OFF
  cmake --build "$orbit_root/build-core" --target taskflow-opt check-taskflow-core -j2
fi
if [[ "$stage" == all || "$stage" == amoeba ]]; then
  cmake --fresh -S "$orbit_root" -B "$orbit_root/build-amoeba" "${common[@]}" -DORBIT_ENABLE_NEURA=ON -DNEURA_ENABLE_PYTHON_BINDINGS=OFF
  cmake --build "$orbit_root/build-amoeba" --target mlir-amoeba-opt -j2
  if [[ ! -x "$llvm_build/bin/not" ]]; then
    cmake --build "$llvm_build" --target not -j2
  fi
  PATH="$llvm_build/bin:$PATH" "$llvm_build/bin/llvm-lit" -sv -j2 "$orbit_root/build-amoeba/test"
fi
source "$toy_root/scripts/env.sh"
cd "$orbit_root"
if [[ "$stage" == cpu ]]; then
  python scripts/taskflow_experiment.py prepare --full "$@"
  python scripts/taskflow_experiment.py cpu "$@"
  python scripts/taskflow_experiment.py negative "$@"
elif [[ "$stage" == all ]]; then
  python scripts/taskflow_experiment.py all --full "$@"
else
  python scripts/taskflow_experiment.py "$stage" "$@"
fi
