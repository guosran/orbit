#!/usr/bin/env bash
# Static-only clean-shell reproduction. No hardware or runtime modifications.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# == 1 ]] || { echo 'usage: run_taskflow_static_v3.sh NEW_OUTPUT_DIRECTORY' >&2; exit 2; }
output="$(realpath -m "$1")"
[[ ! -e "$output" && "$output" != *[[:space:]]* ]] || { echo 'Output must be new and have no whitespace.' >&2; exit 2; }
llvm_build="${LLVM_BUILD:-/home/shiran/project/llvm-project/build}"
toy_root="${AIE_TOY_ROOT:-/home/shiran/project/aie-toy}"
llvm_cc="$(sed -n -E 's/^CMAKE_C_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)"
llvm_cxx="$(sed -n -E 's/^CMAKE_CXX_COMPILER:(FILEPATH|STRING)=//p' "$llvm_build/CMakeCache.txt" | head -n1)"
mkdir -p "$output/validation"
exec > >(tee "$output/reproduce.log") 2>&1
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release "-DMLIR_DIR=$llvm_build/lib/cmake/mlir" "-DLLVM_DIR=$llvm_build/lib/cmake/llvm"
  "-DCMAKE_C_COMPILER=$llvm_cc" "-DCMAKE_CXX_COMPILER=$llvm_cxx"
  -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so -DZLIB_INCLUDE_DIR=/usr/include)
cmake --fresh -S "$root" -B "$root/build-core" "${common[@]}" -DORBIT_ENABLE_NEURA=OFF
cmake --build "$root/build-core" --target taskflow-opt check-taskflow-core -j2
cmake --fresh -S "$root" -B "$root/build-amoeba" "${common[@]}" -DORBIT_ENABLE_NEURA=ON -DNEURA_ENABLE_PYTHON_BINDINGS=OFF
cmake --build "$root/build-amoeba" --target mlir-amoeba-opt -j2
export PATH="$llvm_build/bin:$PATH"
"$llvm_build/bin/llvm-lit" -sv -j2 "$root/build-amoeba/test"
source "$toy_root/scripts/env.sh"
cd "$root"
python test/TaskflowMapping/check.py --output "$output/validation/synthetic"
python test/TaskflowValidation/search_contract.py --tool "$root/build-core/tools/taskflow-opt/taskflow-opt" --output "$output/validation/legacy-search"
python test/TaskflowValidation/evaluation_cache.py --output "$output/validation/legacy-cache"
python scripts/taskflow_static_experiment.py all --output "$output"
python test/TaskflowMapping/search_integration.py --run "$output/search/ffn_m128" --output "$output/validation/static-search"
python test/TaskflowMapping/search_preferences.py --output "$output/validation/preferences" --source "$output/workloads/ffn_m5/source.json"
python - "$output" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);v=root/'validation'
summary={'passed':True,'core':'passed','amoeba_lit':'passed; see reproduce.log','hardware_executed':False}
for name in ('synthetic','legacy-search','legacy-cache','static-search','preferences'):
    summary[name]=json.loads((v/name/'checks.json').read_text())
(v/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
python scripts/taskflow_static_report.py --output "$output"
