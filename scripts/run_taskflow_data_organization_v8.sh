#!/usr/bin/env bash
# C++/MLIR owns semantics, candidate generation, planning, checking and lowering.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 2 && ( "$1" == cpu || "$1" == all ) ]] || { echo 'usage: run_taskflow_data_organization_v8.sh cpu|all NEW_OUTPUT_DIRECTORY' >&2; exit 2; }
mode=$1
out=$(realpath -m "$2")
[[ ! -e "$out" && "$out" != *[[:space:]]* ]] || { echo 'Use a new output directory without whitespace.' >&2; exit 2; }
mkdir -p "$out"
exec > >(tee "$out/reproduce.log") 2>&1
cd "$root"
python_bin=${ORBIT_TEST_PYTHON:-/home/shiran/project/aie-toy/.venv/bin/python}
# Includes independent Core, original Amoeba, static/search, native analysis,
# capacity/dependency checks, tiled/readonly AIR and whole-program regressions.
bash scripts/run_taskflow_program_mapping_v7.sh cpu "$out/regression"
cmake --build build-core --target taskflow-data-organization-test -j2
build-core/tools/taskflow-opt/taskflow-data-organization-test > "$out/shared-copy-checks.json"
tool=$root/build-air/tools/taskflow-opt/taskflow-opt
"$python_bin" test/TaskflowAIR/data_organization_v8.py --tool "$tool" --output "$out/policies" --air
"$python_bin" test/TaskflowAIR/joint_data_search_v8.py --tool "$tool" --output "$out/joint-search" --air
"$python_bin" test/TaskflowMapping/compute_contract_v8.py --tool "$root/build-amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt" --output "$out/cgra-compute-contract"
sha256sum "$tool" build-core/tools/taskflow-opt/taskflow-opt build-amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt > "$out/compiler-identities.sha256"
ldd build-core/tools/taskflow-opt/taskflow-opt > "$out/core-ldd.txt"
if rg -i 'lib[^ ]*(air|aie|xrt|neura)' "$out/core-ldd.txt"; then
  echo 'Independent Core unexpectedly links a target runtime.' >&2; exit 1
fi
if [[ "$mode" == cpu ]]; then exit 0; fi
source /home/shiran/project/aie-toy/scripts/env.sh
"$python_bin" test/TaskflowAIR/prepare_program_mapping_v7.py --output "$out/input"
# Same calculation and N16 tiles, different compiler-selected copy lifetime.
args=()
for seed in 11 23 47; do args+=(--inputs "$out/input/inputs-$seed.npz" --reference "$out/input/reference-$seed.npz"); done
"$python_bin" scripts/taskflow_direct_air_compare.py --cases "$out/joint-search/cases.json" "${args[@]}" --output-dir "$out/hardware-comparison" --iterations 3
"$python_bin" test/TaskflowAIR/audit_compare_v7.py "$out/hardware-comparison" --output "$out/hardware-comparison/independent-audit.json"
# Frozen v7 M128 acceptance remains bound to its original runtime package.
# New compiler gets a complete M5 FP32 FFN regression with all three seeds.
"$python_bin" test/TaskflowAIR/prepare_v6.py --case ffn --m 5 --k 2048 --n 11008 --output "$out/m5-input"
printf '%s\n' '{"generic_air":{"tile_sizes":[8,64,64]}}' > "$out/ffn-config.json"
bash scripts/run_taskflow_direct_air_v6_case.sh --tool "$tool" --input "$out/m5-input" --output "$out/m5-hardware" --target-config "$out/ffn-config.json" --distribute-initialization --approx-exp --runtime-transfer-audit
