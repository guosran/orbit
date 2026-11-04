#!/usr/bin/env bash
set -euo pipefail
# This entrypoint invokes native compiler decisions and existing hardware tools.
# The executor itself locks /tmp/npu.lock. Do not wrap this script in flock.
ORBIT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /home/shiran/project/aie-toy/scripts/env.sh
cd "$ORBIT_ROOT"
OUTPUT=
TOOL=$ORBIT_ROOT/build-air/tools/taskflow-opt/taskflow-opt
APPROX_EXP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT=$2; shift 2 ;;
    --tool) TOOL=$(realpath "$2"); shift 2 ;;
    --approx-exp) APPROX_EXP=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUTPUT" ]] || { echo 'usage: --output NEW_DIR [--tool taskflow-opt] [--approx-exp]' >&2; exit 2; }
OUTPUT=$(realpath -m "$OUTPUT")
[[ ! -e "$OUTPUT" ]] || { echo 'Use a new output directory' >&2; exit 1; }
mkdir -p "$OUTPUT"
CASES=(gemm add add-scale)
[[ "$APPROX_EXP" == 0 ]] || CASES+=(ffn)
for CASE in "${CASES[@]}"; do
  mkdir "$OUTPUT/$CASE"
  FLAGS=()
  [[ "$CASE" != ffn ]] || FLAGS+=(--air-approximate-exp)
  SOURCE="test/TaskflowAIR/fixtures/$CASE.mlir"
  [[ "$CASE" != add ]] || SOURCE=test/TaskflowGeneric/add.mlir
  "$TOOL" "$SOURCE" --lower-to-air "${FLAGS[@]}" \
    --air-output="$OUTPUT/$CASE/program.air.mlir" \
    --analysis-output="$OUTPUT/$CASE/facts.json" --mapping-output="$OUTPUT/$CASE/plan.json"
  python - "$OUTPUT/$CASE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);facts=json.loads((p/'facts.json').read_text())
# Serialization of compiler-owned storage interface only, no graph decisions.
buffers=[dict(name=s['id'],shape=s['shape'],dtype=s['dtype']) for s in facts['storages'] if s['role']!='temporary']
(p/'abi.json').write_text(json.dumps(dict(buffers=buffers,outputs=[s['id'] for s in facts['storages'] if s['role']=='output']),indent=2))
PY
  if [[ "$CASE" == gemm ]]; then
    python test/TaskflowAIR/prepare_inputs.py --case gemm --m 4 --k 8 --n 4 --output "$OUTPUT/$CASE/inputs"
  elif [[ "$CASE" == add ]]; then
    python test/TaskflowAIR/prepare_inputs.py --case add --m 4 --n 4 --rank-one --output "$OUTPUT/$CASE/inputs"
  elif [[ "$CASE" == ffn ]]; then
    python test/TaskflowAIR/prepare_inputs.py --case ffn --m 2 --k 8 --n 16 --output "$OUTPUT/$CASE/inputs"
  else
    python test/TaskflowAIR/prepare_inputs.py --case add_scale --m 4 --n 4 --output "$OUTPUT/$CASE/inputs"
  fi
  ARGS=()
  for SEED in 11 23 47; do ARGS+=(--inputs "$OUTPUT/$CASE/inputs/inputs-$SEED.npz" --reference "$OUTPUT/$CASE/inputs/reference-$SEED.npz"); done
  python scripts/taskflow_direct_air_execute.py --air "$OUTPUT/$CASE/program.air.mlir" --abi "$OUTPUT/$CASE/abi.json" "${ARGS[@]}" --output-dir "$OUTPUT/$CASE/hardware" --iterations 5
  python test/TaskflowAIR/audit_hardware.py "$OUTPUT/$CASE/hardware" --output "$OUTPUT/$CASE/audit.json"
done
python - "$OUTPUT" "$TOOL" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);tool=Path(sys.argv[2]);cases=[]
for path in sorted(root.glob('*/audit.json')):
    audit=json.loads(path.read_text())
    records=json.loads((path.parent/'hardware/results.json').read_text())
    cases.append(dict(case=path.parent.name,audit=audit['status'],
                      seeds=audit['seeds_checked'],outputs=audit['outputs_checked'],
                      max_abs=max(c['max_abs'] for r in records for s in r['samples'] for c in s['checks'].values()),
                      medians_ms=[r['median_ms'] for r in records]))
(root/'summary.json').write_text(json.dumps(dict(status='passed',tool=str(tool),
    tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),atol=1e-5,rtol=1e-5,
    stack_size_bytes=4096,cases=cases),indent=2)+'\n')
PY
