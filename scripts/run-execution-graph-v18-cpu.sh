#!/usr/bin/env bash
# Compiler-only replay. All outputs are new; no builds or NPU submissions.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo 'usage: run-execution-graph-v18-cpu.sh CORE_BUILD AIR_BUILD NEW_OUTPUT [PRESERVED_AMOEBA_TOOL]' >&2
  exit 2
fi
if [[ ${ORBIT_V18_CLEAN_ENTRY:-} != 1 ]]; then
  exec env -i HOME="$HOME" PATH=/usr/bin:/bin ORBIT_V18_CLEAN_ENTRY=1 \
    /bin/bash --noprofile --norc "$0" "$@"
fi
orbit_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
core_build=$(realpath "$1")
air_build=$(realpath "$2")
output=$(realpath -m "$3")
amoeba_tool=$(realpath "${4:-$orbit_root/../orbit-artifact/builds/taskflow-v13/amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt}")
artifact_root=$(realpath "$orbit_root/../orbit-artifact")
aie_root=$(realpath "$orbit_root/../aie-toy")
llvm_root=$(realpath "$orbit_root/../llvm-project/build")
[[ ! -e "$output" ]] || { echo 'Output must be new.' >&2; exit 2; }
mkdir -p "$output"
cd "$orbit_root"
cp "$orbit_root/scripts/run-execution-graph-v18-cpu.sh" "$output/reproduction-source.sh"
core_tool="$core_build/tools/taskflow-opt/taskflow-opt"
air_tool="$air_build/tools/taskflow-opt/taskflow-opt"
for executable in "$core_tool" "$air_tool" "$amoeba_tool"; do
  [[ -x "$executable" ]] || { echo "Missing executable: $executable" >&2; exit 2; }
done
record() {
  /usr/bin/python3 - "$output" "$@" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
out=Path(sys.argv[1]); name=sys.argv[2]; command=sys.argv[3:]
(out/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
start=time.monotonic()
with (out/(name+'.stdout')).open('w') as stdout, (out/(name+'.stderr')).open('w') as stderr:
    result=subprocess.run(command,stdout=stdout,stderr=stderr)
(out/(name+'.result.json')).write_text(json.dumps({'exit_code':result.returncode,
    'elapsed_seconds':time.monotonic()-start,'hardware_executed':False},indent=2)+'\n')
raise SystemExit(result.returncode)
PY
}
/usr/bin/python3 - "$output" "$core_tool" "$air_tool" "$amoeba_tool" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
out=Path(sys.argv[1]);paths=[Path(p) for p in sys.argv[2:]]
(out/'tool-sha-before.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
(out/'clean-entry-environment.json').write_text(json.dumps(dict(os.environ),indent=2)+'\n')
PY
# This full Core run precedes loading any AIR/runtime environment.
record core38 /usr/bin/python3 "$orbit_root/test/run_core_regressions.py" \
  --tool "$core_tool" --output "$output/core"
# Compiler legalization and the preserved model test require this explicit SDK.
source "$aie_root/scripts/env.sh"
export PATH="$llvm_root/bin:$PATH"
/usr/bin/python3 - "$output" "$aie_root/scripts/env.sh" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[2]);Path(sys.argv[1],'toolchain-environment.json').write_text(json.dumps({
    'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
    'scope':'explicit existing AIR toolchain loaded after clean Core run'},indent=2)+'\n')
PY
record air-v18 python3 "$orbit_root/test/TaskflowAIR/execution_groups_backend_v18.py" \
  --tool "$air_tool" --output "$output/air-v18"
record air-v17 python3 "$orbit_root/test/TaskflowAIR/data_organization_backend_v17.py" \
  --tool "$air_tool" --output "$output/air-v17"
record air-original-binding python3 "$orbit_root/test/TaskflowAIR/program_graph_backend_v11.py" \
  --tool "$air_tool" --output "$output/air-original-binding"
record combined python3 "$orbit_root/test/run_core_regressions.py" \
  --tool "$core_tool" --air-tool "$air_tool" --amoeba-tool "$amoeba_tool" \
  --air-opt "$aie_root/.venv/bin/air-opt" --output "$output/combined"
export ORBIT_GRAPH_CORE_TOOL="$core_tool" ORBIT_GRAPH_AMOEBA_TOOL="$amoeba_tool"
export ORBIT_GRAPH_AIR_TOOL="$air_tool"
record original-amoeba-cgra bash "$artifact_root/taskflow/scripts/run_taskflow_program_graph_amoeba_v12.sh" \
  cpu "$output/original-amoeba-cgra"
/usr/bin/python3 - "$orbit_root" "$output/lit" "$amoeba_tool" "$llvm_root" <<'PY'
import sys
from pathlib import Path
source,out,amoeba,llvm=map(Path,sys.argv[1:]);out.mkdir()
text=(source/'test/lit.cfg.in').read_text()
values={'CMAKE_SOURCE_DIR':str(source),'CMAKE_BINARY_DIR':str(out.parent),
    'MLIR_AMOEBA_OPT':str(amoeba),'FILECHECK':str(llvm/'bin/FileCheck'),
    'MLIR_OPT':str(llvm/'bin/mlir-opt'),'MLIR_TRANSLATE':str(llvm/'bin/mlir-translate'),
    'LLC':str(llvm/'bin/llc'),'CLANG':str(llvm/'bin/clang')}
for key,value in values.items():text=text.replace('@'+key+'@',value)
text=text.replace("config.test_exec_root = r'"+str(out.parent)+"/test'", "config.test_exec_root = r'"+str(out)+"'")
(out/'lit.cfg').write_text(text)
PY
record original-lit "$llvm_root/bin/llvm-lit" -v -j 4 "$output/lit"
/usr/bin/python3 - "$output" "$core_tool" "$air_tool" "$amoeba_tool" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[2:]}
(out/'tool-sha-after.json').write_text(json.dumps(after,indent=2)+'\n')
assert after==json.loads((out/'tool-sha-before.json').read_text())
(out/'result.json').write_text(json.dumps({'status':'PASS','hardware_executed':False,
    'clean_core_before_toolchain':True,'tools_unchanged':True,
    'scope':'Core, combined old AIR/original driver, new/old AIR binding, preserved Amoeba/CGRA contracts and unfiltered lit'},indent=2)+'\n')
PY
