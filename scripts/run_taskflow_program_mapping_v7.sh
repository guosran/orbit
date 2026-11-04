#!/usr/bin/env bash
# Whole-program native search + public mapping + direct AIR acceptance.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 2 && ( "$1" == cpu || "$1" == all ) ]] || { echo 'usage: run_taskflow_program_mapping_v7.sh cpu|all NEW_OUTPUT_DIRECTORY' >&2; exit 2; }
mode=$1
out=$(realpath -m "$2")
[[ ! -e "$out" && "$out" != *[[:space:]]* ]] || { echo 'Use a new output directory without whitespace.' >&2; exit 2; }
mkdir -p "$out"
exec > >(tee "$out/reproduce.log") 2>&1
cd "$root"
bash scripts/run_taskflow_tiled_air.sh cpu "$out/regression"
source /home/shiran/project/aie-toy/scripts/env.sh
tool=$root/build-air/tools/taskflow-opt/taskflow-opt
sha256sum "$tool" > "$out/native-compiler.sha256"
git rev-parse HEAD > "$out/base-revision.txt"
git status --short > "$out/worktree-status.txt"
cmake --build build-air --target taskflow-program-mapping-test taskflow-program-partition-test -j2
build-air/tools/taskflow-opt/taskflow-program-mapping-test test/TaskflowAIR/fixtures/gemm.mlir > "$out/public-mapping-checks.json"
# Partition checker uses the actual body and independently mutates witnesses.
build-air/tools/taskflow-opt/taskflow-program-partition-test > "$out/program-partition-checks.json"
python test/TaskflowAIR/mapping_v7.py --tool "$tool" --air-opt "$(command -v air-opt)" --output "$out/backend-mapping"
python test/TaskflowAIR/prepare_program_mapping_v7.py --output "$out/input"
python test/TaskflowGeneric/program_search_v7.py --tool "$tool" --source "$out/input/source.mlir" --output "$out/search"
python test/TaskflowGeneric/program_partition_search_v7.py --tool "$tool" --source "$out/input/source.mlir" --output "$out/partition-search"
python test/TaskflowAIR/search_cases_v7.py --summary "$out/search/summary.json" --output "$out/selected"
python - "$out" <<'PYMANIFEST'
import json, subprocess, sys
from pathlib import Path
out=Path(sys.argv[1]); row=json.loads((out/'partition-search/summary.json').read_text())['accepted'][0]
subprocess.run([sys.executable,'test/TaskflowAIR/test_program_manifest_v7.py',str(Path(row['path'])/'lowered/manifest.json'),'--output',str(out/'manifest-negative-checks.json')],check=True)
PYMANIFEST
if [[ "$mode" == cpu ]]; then exit 0; fi
args=()
for seed in 11 23 47; do args+=(--inputs "$out/input/inputs-$seed.npz" --reference "$out/input/reference-$seed.npz"); done
python scripts/taskflow_direct_air_compare.py --cases "$out/selected/cases.json" "${args[@]}" --output-dir "$out/search-hardware" --iterations 3
python test/TaskflowAIR/audit_compare_v7.py "$out/search-hardware" --output "$out/search-hardware/independent-audit.json"
# Exercise two compiler-issued template extents and full output assembly.
python test/TaskflowAIR/prepare_program_mapping_v7.py --m 17 --output "$out/partition-tail-input"
"$tool" "$out/partition-tail-input/source.mlir" --air-distribute-initialization --tile-m=8 --tile-n=16 --tile-k=32 --program-partition=8 --lower-program-to-air="$out/partition-tail-plan"
args=()
for seed in 11 23 47; do args+=(--inputs "$out/partition-tail-input/inputs-$seed.npz" --reference "$out/partition-tail-input/reference-$seed.npz"); done
python scripts/taskflow_direct_air_program.py --manifest "$out/partition-tail-plan/manifest.json" "${args[@]}" --output-dir "$out/partition-tail-hardware" --iterations 3
python test/TaskflowAIR/audit_program_v7.py "$out/partition-tail-hardware" --output "$out/partition-tail-hardware/program-audit.json"
# Preserve the existing M5 and nondivisible M/N/K contracts.
printf '%s\n' '{"generic_air":{"tile_sizes":[8,64,64]}}' > "$out/ffn-config.json"
printf '%s\n' '{"generic_air":{"tile_sizes":[8,32,64]}}' > "$out/tails-config.json"
python test/TaskflowAIR/prepare_v6.py --case gemm --m 17 --k 129 --n 65 --output "$out/tails-input"
bash scripts/run_taskflow_direct_air_v6_case.sh --tool "$tool" --input "$out/tails-input" --output "$out/tails-hardware" --target-config "$out/tails-config.json" --distribute-initialization --runtime-transfer-audit
printf '%s\n' '{"generic_air":{"tile_sizes":[32,64,64]}}' > "$out/add-scale-config.json"
python test/TaskflowAIR/prepare_v6.py --case add-scale --m 64 --n 64 --output "$out/add-scale-input"
bash scripts/run_taskflow_direct_air_v6_case.sh --tool "$tool" --input "$out/add-scale-input" --output "$out/add-scale-hardware" --target-config "$out/add-scale-config.json" --runtime-transfer-audit
python test/TaskflowAIR/prepare_v6.py --case ffn --m 5 --k 2048 --n 11008 --output "$out/m5-input"
bash scripts/run_taskflow_direct_air_v6_case.sh --tool "$tool" --input "$out/m5-input" --output "$out/m5-hardware" --target-config "$out/ffn-config.json" --distribute-initialization --approx-exp --runtime-transfer-audit
python test/TaskflowAIR/prepare_v6.py --case ffn --m 128 --k 2048 --n 11008 --output "$out/m128-input"
"$tool" "$out/m128-input/source.mlir" --air-distribute-initialization --tile-m=8 --tile-n=64 --tile-k=64 --program-partition=8 --air-approximate-exp --lower-program-to-air="$out/m128-plan"
args=()
for seed in 11 23 47; do args+=(--inputs "$out/m128-input/inputs-$seed.npz" --reference "$out/m128-input/reference-$seed.npz"); done
python scripts/taskflow_direct_air_program.py --manifest "$out/m128-plan/manifest.json" "${args[@]}" --output-dir "$out/m128-hardware" --long-acceptance
python test/TaskflowAIR/audit_program_v7.py "$out/m128-hardware" --output "$out/m128-hardware/program-audit.json"
python test/TaskflowAIR/audit_metrics_v6.py "$out/m128-hardware" --output "$out/m128-hardware/error-metrics.json"
bash scripts/run_taskflow_exp_validation_v7.sh --output "$out/exp-validation" --tool "$tool"
echo "Completed whole-program acceptance: $out"
