#!/usr/bin/env bash
# Fresh real-NPU regression of explicitly selected f32 exp/SiLU/SwiGLU lowering.
set -euo pipefail
ORBIT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /home/shiran/project/aie-toy/scripts/env.sh
cd "$ORBIT_ROOT"
OUTPUT=
TOOL=$ORBIT_ROOT/build-air/tools/taskflow-opt/taskflow-opt
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT=$(realpath -m "$2"); shift 2;;
    --tool) TOOL=$(realpath "$2"); shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$OUTPUT" && ! -e "$OUTPUT" ]] || { echo 'required: --output NEW_DIR' >&2; exit 2; }
mkdir -p "$OUTPUT"
python test/TaskflowAIR/prepare_exp_probe_v6.py --output "$OUTPUT/fixtures"
for KIND in exp silu swiglu; do
  scripts/run_taskflow_direct_air_v6_case.sh --input "$OUTPUT/fixtures/$KIND" \
    --output "$OUTPUT/$KIND" --tool "$TOOL" --approx-exp > "$OUTPUT/$KIND-wrapper.log" 2>&1
done
# The finite-only acceptance check must reject the expected overflow references.
# Preserve that raw failure, then classify all values without changing acceptance.
EDGE_STATUS=0
if scripts/run_taskflow_direct_air_v6_case.sh --input "$OUTPUT/fixtures/exp-edge" \
  --output "$OUTPUT/exp-edge" --tool "$TOOL" --approx-exp --diagnostic \
  > "$OUTPUT/exp-edge-wrapper.log" 2>&1; then
  EDGE_STATUS=0
else
  EDGE_STATUS=$?
fi
python - "$OUTPUT" "$EDGE_STATUS" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);status=int(sys.argv[2]);hardware=out/'exp-edge/hardware'
assert status==1, 'expected unchanged numeric rejection, not success or a different failure'
assert json.loads((hardware/'compile.json').read_text())['status']=='compiled'
record=json.loads((hardware/'seed-0/result.json').read_text())
assert record['status']=='numeric_failed' and len(record['samples'])==1
assert (hardware/'seed-0/outputs-0.npz').exists()
(out/'edge-expected-rejection.json').write_text(json.dumps(dict(wrapper_returncode=status,
    status='expected_numeric_rejection_preserved',scope='overflow/underflow classification only; no finite-check or tolerance relaxation'),indent=2)+'\n')
PY
python test/TaskflowAIR/classify_exp_edges_v6.py "$OUTPUT/exp-edge/hardware" --output "$OUTPUT/exp-edge/classification.json"
python test/TaskflowAIR/audit_storage_v6.py "$OUTPUT/exp-edge/hardware" --output "$OUTPUT/exp-edge/storage-audit.json"
python test/TaskflowAIR/audit_metrics_v6.py "$OUTPUT/exp-edge/hardware" --output "$OUTPUT/exp-edge/error-metrics.json"
python - "$OUTPUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);records={}
for kind in ['exp','silu','swiglu']:
    audit=json.loads((out/kind/'audit.json').read_text());assert audit['status']=='passed'
    records[kind]=audit
(out/'summary.json').write_text(json.dumps(dict(status='passed',finite_contract_regressions=records,
    edge_scope='diagnostic_only; expected numeric rejection retained, classification.json describes behavior'),indent=2)+'\n')
(out/'evidence-manifest.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='evidence-manifest.json'},indent=2)+'\n')
PY
