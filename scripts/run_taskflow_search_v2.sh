#!/usr/bin/env bash
# Rebuild and reproduce this round in a NEW evidence directory.
set -euo pipefail
orbit_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="${1:-cpu}"
case "$stage" in cpu|all) ;; *) echo 'usage: run_taskflow_search_v2.sh [cpu|all] NEW_OUTPUT_DIR' >&2; exit 2;; esac
[[ $# == 2 ]] || { echo 'A new output directory is required.' >&2; exit 2; }
output="$(realpath -m "$2")"
[[ ! -e "$output" && "$output" != *[[:space:]]* ]] || { echo 'Output must be new and contain no whitespace.' >&2; exit 2; }
llvm_build="${LLVM_BUILD:-/home/shiran/project/llvm-project/build}"
toy_root="${AIE_TOY_ROOT:-/home/shiran/project/aie-toy}"
llvm_cc="$(sed -n -E 's/^CMAKE_C_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)"
llvm_cxx="$(sed -n -E 's/^CMAKE_CXX_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)"
mkdir -p "$output/validation"
exec > >(tee "$output/reproduce.log") 2>&1
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release
  "-DMLIR_DIR=$llvm_build/lib/cmake/mlir" "-DLLVM_DIR=$llvm_build/lib/cmake/llvm"
  "-DCMAKE_C_COMPILER=$llvm_cc" "-DCMAKE_CXX_COMPILER=$llvm_cxx"
  -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so -DZLIB_INCLUDE_DIR=/usr/include)
cmake --fresh -S "$orbit_root" -B "$orbit_root/build-core" "${common[@]}" -DORBIT_ENABLE_NEURA=OFF
cmake --build "$orbit_root/build-core" --target taskflow-opt check-taskflow-core -j2
cmake --fresh -S "$orbit_root" -B "$orbit_root/build-amoeba" "${common[@]}" -DORBIT_ENABLE_NEURA=ON -DNEURA_ENABLE_PYTHON_BINDINGS=OFF
cmake --build "$orbit_root/build-amoeba" --target mlir-amoeba-opt -j2
export PATH="$llvm_build/bin:$PATH"
"$llvm_build/bin/llvm-lit" -sv -j2 "$orbit_root/build-amoeba/test"
source "$toy_root/scripts/env.sh"
cd "$orbit_root"
python test/TaskflowValidation/search_contract.py --tool "$orbit_root/build-core/tools/taskflow-opt/taskflow-opt" --output "$output/validation/search-contract"
python test/TaskflowValidation/evaluation_cache.py --output "$output/validation/cache"
python scripts/taskflow_search_experiment.py prepare --output "$output"
python scripts/taskflow_search_experiment.py search --output "$output" --backend amoeba --case gemm --label formal-cacheguard --rounds 2 --budget 7 --beam 2
[[ "$stage" == all ]] || exit 0
# Runtime acquires /tmp/npu.lock itself. Do not wrap these commands in flock.
npu_results="$output/npu"
for case_name in ffn_m5 ffn_m128; do
  python scripts/taskflow_search_experiment.py search --output "$output" --backend npu --case "$case_name" --label final-timing --rounds 2 --budget 9 --beam 2
done
python scripts/taskflow_npu_bench.py --source-dir "$output/data" --output-dir "$npu_results/full-forward-final-timing" --cases ffn_m5,ffn_m128 --seeds 0 --configs baseline,persistent,merge,combined --iterations 7
python scripts/taskflow_npu_bench.py --source-dir "$output/data" --output-dir "$npu_results/full-reverse-final-timing" --cases ffn_m128,ffn_m5 --seeds 0 --configs combined,merge,persistent,baseline --iterations 7
for suffix in m5 m128; do
  python scripts/taskflow_npu_search_confirm.py --search-root "$output" --case "ffn_$suffix" --label final-timing --iterations 7 --output-dir "$npu_results/winner-confirm-$suffix-final-timing"
done
python scripts/taskflow_npu_bench.py --source-dir "$output/data" --output-dir "$npu_results/small-final-timing"
for case_name in gemm gemv gemm_relu ffn; do
  mkdir -p "$output/data/$case_name/k"
  build-core/tools/taskflow-opt/taskflow-opt "$output/data/$case_name/source.json" --tile-k=7 --emit-plan="$output/data/$case_name/k/plan.json" --emit-mlir="$output/data/$case_name/k/taskflow.mlir"
done
python scripts/taskflow_npu_bench.py --source-dir "$output/data" --output-dir "$npu_results/device-small-final-timing" --variant k --configs device
python scripts/taskflow_npu_lifetime.py --output-dir "$npu_results/lifetime-final-timing"
python scripts/taskflow_npu_lifetime.py --device --output-dir "$npu_results/device-lifetime-final-timing"
# A diagnostic failure is expected; its exit code and compiler evidence are saved.
set +e
python scripts/taskflow_npu_device_k.py --strategy nested-loop --chunk-k 5504 --columns 2048 --output-dir "$npu_results/device-k-5504-mre"
mre_status=$?
set -e
echo "$mre_status" > "$npu_results/device-k-5504-mre.exit-code"
python scripts/taskflow_v2_report.py --output "$output" --npu-results "$npu_results"
