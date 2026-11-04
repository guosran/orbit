#!/usr/bin/env bash
# Native compiler decisions; Python is fixture/reference/execution orchestration.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# == 2 && ( "$1" == cpu || "$1" == all ) ]] || { echo 'usage: run_taskflow_tiled_air.sh cpu|all NEW_OUTPUT_DIRECTORY' >&2; exit 2; }
mode=$1
out=$(realpath -m "$2")
[[ ! -e "$out" && "$out" != *[[:space:]]* ]] || { echo 'Use a new output directory without whitespace.' >&2; exit 2; }
mkdir -p "$out"
exec > >(tee "$out/reproduce.log") 2>&1
cd "$root"
bash scripts/run_taskflow_direct_air.sh cpu "$out/regression"
source /home/shiran/project/aie-toy/scripts/env.sh
tool=$root/build-air/tools/taskflow-opt/taskflow-opt
cmake --build build-air --target taskflow-tiled-mapping-test -j2
python test/TaskflowGeneric/distribution_v6.py --tool "$tool" --output "$out/distribution"
python test/TaskflowGeneric/unsupported_v6.py --tool "$tool" --output "$out/unsupported"
python test/TaskflowGeneric/pool_names_v6.py --tool "$tool" --output "$out/pool-names"
build-air/tools/taskflow-opt/taskflow-tiled-mapping-test "$out/distribution/gemm.transformed.mlir" > "$out/tiled-checker.json"
python test/TaskflowAIR/tiling.py --tool "$tool" --air-opt "$(command -v air-opt)" --output "$out/tiled-air-checks"
python test/TaskflowAIR/channels_v6.py --tool "$tool" --air-opt "$(command -v air-opt)" --output "$out/channel-air-checks"
python test/TaskflowAIR/prepare_v6.py --case gemm --m 16 --k 128 --n 64 --output "$out/gemm-input"
python test/TaskflowGeneric/tiled_search_v6.py --tool "$tool" --source "$out/gemm-input/source.mlir" --output "$out/search"
if [[ "$mode" == cpu ]]; then exit 0; fi
python - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
for name,sizes in [('gemm',[16,32,64]),('gemm-candidate',[8,32,64]),('tails',[8,32,64]),('add-scale',[32,64,64]),('ffn',[8,64,64])]:
    (p/(name+'.json')).write_text(json.dumps({'generic_air':{'tile_sizes':sizes}}))
PY
python test/TaskflowAIR/prepare_v6.py --case gemm --m 17 --k 129 --n 65 --output "$out/tails-input"
python test/TaskflowAIR/prepare_v6.py --case add-scale --m 64 --n 64 --output "$out/add-scale-input"
failures=0
run_case() {
  local name=$1 input=$2 config=$3
  shift 3
  if bash scripts/run_taskflow_direct_air_v6_case.sh --tool "$tool" --input "$input" --output "$out/$name" --target-config "$config" "$@"; then
    printf '%s passed\n' "$name" >> "$out/case-status.txt"
  else
    printf '%s failed\n' "$name" >> "$out/case-status.txt"
    failures=$((failures+1))
  fi
}
run_case gemm "$out/gemm-input" "$out/gemm.json" --distribute-initialization
run_case gemm-candidate "$out/gemm-input" "$out/gemm-candidate.json" --distribute-initialization
run_case tails "$out/tails-input" "$out/tails.json" --distribute-initialization
run_case add-scale "$out/add-scale-input" "$out/add-scale.json"
# Compare real archived candidates under alternating execution order.
python - "$out" <<'PYSEARCH'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); summary=json.loads((out/'search/summary.json').read_text()); cases=[]
for i,row in enumerate(summary['accepted'][:3]):
    directory=Path(row['path']); facts=json.loads((directory/'facts.json').read_text())
    storages=facts['storages']; storages=[s for s in storages if s['role']!='temporary']+[s for s in storages if s['role']=='temporary']
    abi=out/f'search-abi-{i}.json'
    abi.write_text(json.dumps(dict(buffers=[dict(name=s['id'],shape=s['shape'],dtype=s['dtype'],role=s['role']) for s in storages],outputs=[s['id'] for s in storages if s['role']=='output']),indent=2))
    cases.append(dict(name=f'archive-{i}',candidate_id=row['candidate_id'],plan_id=row['plan_id'],taskflow=str(directory/'taskflow.mlir'),plan=str(directory/'mapping.json'),air=str(directory/'program.air.mlir'),abi=str(abi)))
(out/'search-cases.json').write_text(json.dumps(cases,indent=2))
PYSEARCH
comparison_args=()
for seed in 11 23 47; do comparison_args+=(--inputs "$out/gemm-input/inputs-$seed.npz" --reference "$out/gemm-input/reference-$seed.npz"); done
if python scripts/taskflow_direct_air_compare.py --cases "$out/search-cases.json" "${comparison_args[@]}" --output-dir "$out/search-hardware" --iterations 3 > "$out/search-hardware.log" 2>&1; then
  printf 'search-hardware passed\n' >> "$out/case-status.txt"
else
  printf 'search-hardware failed\n' >> "$out/case-status.txt"
  failures=$((failures+1))
fi
for m in 5 128; do
  python test/TaskflowAIR/prepare_v6.py --case ffn --m "$m" --k 2048 --n 11008 --output "$out/ffn-m$m-input"
  run_case "ffn-m$m" "$out/ffn-m$m-input" "$out/ffn.json" --distribute-initialization --approx-exp
done
printf 'Failed cases: %s; inspect %s/case-status.txt and raw artifacts.\n' "$failures" "$out"
[[ "$failures" == 0 ]]
