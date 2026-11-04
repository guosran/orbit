#!/usr/bin/env bash
# Reproduce one prepared v6 case using native C++ mapping/lowering and real XRT.
# The executor owns /tmp/npu.lock. Input directory contains source and NPZ pairs.
set -euo pipefail
ORBIT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /home/shiran/project/aie-toy/scripts/env.sh
cd "$ORBIT_ROOT"
INPUT= OUTPUT= CONFIG=
TOOL=$ORBIT_ROOT/build-air/tools/taskflow-opt/taskflow-opt
APPROX=0
DISTRIBUTE=0
DIAGNOSTIC=0
IMPLEMENTATION=
VECTOR_LANES=
RUNTIME_AUDIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT=$(realpath "$2"); shift 2;;
    --output) OUTPUT=$(realpath -m "$2"); shift 2;;
    --tool) TOOL=$(realpath "$2"); shift 2;;
    --target-config) CONFIG=$(realpath "$2"); shift 2;;
    --approx-exp) APPROX=1; shift;;
    --diagnostic) DIAGNOSTIC=1; shift;;
    --runtime-transfer-audit) RUNTIME_AUDIT=1; shift;;
    --implementation) IMPLEMENTATION=$2; shift 2;;
    --vector-lanes) VECTOR_LANES=$2; shift 2;;
    --distribute-initialization) DISTRIBUTE=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$INPUT" && -n "$OUTPUT" ]] || { echo 'required: --input PREPARED_DIR --output NEW_DIR' >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo 'use a new output directory' >&2; exit 2; }
mkdir -p "$OUTPUT"
cp "$INPUT/source.mlir" "$OUTPUT/source.mlir"
FLAGS=()
[[ -z "$IMPLEMENTATION" ]] || FLAGS+=(--implementation="$IMPLEMENTATION")
[[ -z "$VECTOR_LANES" ]] || FLAGS+=(--vector-lanes="$VECTOR_LANES")
[[ "$APPROX" == 0 ]] || FLAGS+=(--air-approximate-exp)
[[ "$DISTRIBUTE" == 0 ]] || FLAGS+=(--air-distribute-initialization)
if [[ -n "$CONFIG" ]]; then cp "$CONFIG" "$OUTPUT/target.json"; FLAGS+=(--target-config="$OUTPUT/target.json"); fi
python - "$OUTPUT" "$TOOL" "$INPUT" "${FLAGS[@]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);tool=Path(sys.argv[2])
(out/'invocation.json').write_text(json.dumps(dict(tool=str(tool),tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),input=sys.argv[3],flags=sys.argv[4:]),indent=2)+'\n')
PY
"$TOOL" "$OUTPUT/source.mlir" --lower-to-air "${FLAGS[@]}" \
  --emit-mlir="$OUTPUT/transformed-taskflow.mlir" \
  --air-output="$OUTPUT/program.air.mlir" --analysis-output="$OUTPUT/facts.json" \
  --mapping-output="$OUTPUT/plan.json" > "$OUTPUT/lower.log" 2>&1
AIR_INPUT="$OUTPUT/program.air.mlir"
if python - "$OUTPUT/plan.json" <<'PY'
import json,sys
plan=json.load(open(sys.argv[1]))
sys.exit(0 if any(s.get('legalize_channels',False) for s in plan.get('hardware_stages',[])) else 1)
PY
then
  # Installed standard passes convert DMA to channels. C++ owns and checks
  # every physical channel resource; Python only invokes recorded commands.
  python - "$OUTPUT" "$TOOL" "${FLAGS[@]}" <<'PY'
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);tool=sys.argv[2];flags=sys.argv[3:]
def run(name,command):
    (out/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
    executable=Path(command[0]).resolve()
    (out/(name+'.tool.json')).write_text(json.dumps(dict(path=str(executable),
        sha256=hashlib.sha256(executable.read_bytes()).hexdigest()),indent=2)+'\n')
    with (out/(name+'.log')).open('w') as log:
        p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    (out/(name+'.result.json')).write_text(json.dumps(dict(returncode=p.returncode))+'\n')
    (out/(name+'.artifacts.json')).write_text(json.dumps(
        {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.mlir'))},indent=2)+'\n')
    if p.returncode:raise SystemExit(p.returncode)
canonical=out/'canonical.air.mlir';shutil.copy2(out/'program.air.mlir',canonical)
converted=out/'channelized.generic.air.mlir';pinned=out/'pinned.generic.air.mlir'
run('standard-channel-conversion',[shutil.which('air-opt'),str(canonical),
    '--air-dma-to-channel','--canonicalize','--mlir-print-op-generic','-o',str(converted)])
run('native-channel-legalization',[tool,str(out/'source.mlir'),*flags,
    '--legalize-air-channels='+str(converted),'--air-output='+str(pinned)])
run('native-channel-independent-check',[tool,str(out/'source.mlir'),*flags,
    '--check-air-channels='+str(pinned)])
(out/'channel-pipeline-hashes.json').write_text(json.dumps(
    {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [canonical,converted,pinned]},indent=2)+'\n')
PY
  AIR_INPUT="$OUTPUT/pinned.generic.air.mlir"
fi
python - "$OUTPUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);facts=json.loads((out/'facts.json').read_text());plan=json.loads((out/'plan.json').read_text())
s=facts['storages'];s=[x for x in s if x['role']!='temporary']+([x for x in s if x['role']=='temporary'] if plan['strategy']=='serial-tiled' else [])
(out/'abi.json').write_text(json.dumps(dict(buffers=[dict(name=x['id'],shape=x['shape'],dtype=x['dtype'],role=x['role']) for x in s],outputs=[x['id'] for x in s if x['role']=='output']),indent=2)+'\n')
PY
ARGS=()
[[ "$RUNTIME_AUDIT" == 0 ]] || ARGS+=(--preflight-runtime-plan "$OUTPUT/plan.json")
if [[ -f "$OUTPUT/channel-pipeline-hashes.json" ]]; then
  ARGS+=(--preflight-channel-plan "$OUTPUT/plan.json")
fi
if python - "$OUTPUT/plan.json" <<'PY'
import json,sys
sys.exit(0 if json.load(open(sys.argv[1]))['strategy']=='serial-tiled' else 1)
PY
then ARGS+=(--omit-ping-pong-transform); fi
SEEDS=(11 23 47)
MEASURE=(--iterations 3)
if [[ "$DIAGNOSTIC" == 1 ]]; then SEEDS=(11); MEASURE=(--diagnostic --iterations 1 --warmups 0); fi
for SEED in "${SEEDS[@]}"; do ARGS+=(--inputs "$INPUT/inputs-$SEED.npz" --reference "$INPUT/reference-$SEED.npz"); done
python scripts/taskflow_direct_air_execute.py --air "$AIR_INPUT" \
  --abi "$OUTPUT/abi.json" "${ARGS[@]}" --output-dir "$OUTPUT/hardware" "${MEASURE[@]}" \
  > "$OUTPUT/execute.log" 2>&1
if [[ "$DIAGNOSTIC" == 0 ]]; then
  python test/TaskflowAIR/audit_hardware.py "$OUTPUT/hardware" --output "$OUTPUT/audit.json"
else
  echo '{"scope":"single-sample diagnostic only; full acceptance audit not applicable"}' > "$OUTPUT/diagnostic-scope.json"
fi
python test/TaskflowAIR/audit_storage_v6.py "$OUTPUT/hardware" --output "$OUTPUT/storage-audit.json"
python test/TaskflowAIR/audit_metrics_v6.py "$OUTPUT/hardware" --output "$OUTPUT/error-metrics.json"
if [[ -f "$OUTPUT/channel-pipeline-hashes.json" ]]; then
  python test/TaskflowAIR/audit_channel_ports_v6.py "$OUTPUT/hardware" \
    --plan "$OUTPUT/plan.json" --output "$OUTPUT/channel-port-audit.json"
fi
