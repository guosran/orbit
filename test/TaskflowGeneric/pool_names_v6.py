#!/usr/bin/env python3
"""User reporting names cannot alias independent native local storage pools."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool

p = argparse.ArgumentParser()
p.add_argument('--tool', type=Path, default=taskflow_tool('air'))
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
source = (root / 'test/TaskflowGeneric/add.mlir').read_text().replace('orbit.name = "C"', 'orbit.name = "A.input.2.loops.2"')
src = a.output / 'source.mlir'
src.write_text(source)
air = a.output / 'program.air.mlir'
plan = a.output / 'mapping.json'
command = [str(a.tool.resolve()), str(src.resolve()), '--lower-to-air', '--tile-m=2', '--tile-n=1', '--tile-k=1', '--air-output=' + str(air.resolve()), '--mapping-output=' + str(plan.resolve()), '--analysis-output=' + str((a.output / 'analysis.json').resolve())]
result = subprocess.run(command, text=True, capture_output=True)
(a.output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
(a.output / 'stdout.log').write_text(result.stdout)
(a.output / 'stderr.log').write_text(result.stderr)
assert result.returncode == 0, result.stderr
text = air.read_text()
pools = re.findall(r'"memref.alloc"[^\n]+orbit.tile_pool = "([^"]+)"[^\n]+memref<(\d+)xf32, 2', text)
assert len(pools) == 3 and len({name for name, count in pools}) == 3, pools
payload = sum(int(count) * 4 for name, count in pools)
facts = json.loads(plan.read_text())
assert payload == facts['resident_bytes'] == 24, payload
assert sum(pool['allocated_bytes'] for pool in facts['hardware_stages'][0]['buffer_pools']) == payload
assert facts['system_memory_peak_bytes'] == 3 * 16 * 4
assert len(facts['host_buffers']) == 3 and facts['backend_tensor_padding_bytes'] == 0
(a.output / 'summary.json').write_text(json.dumps({'passed': True, 'pool_count': len(pools), 'payload_bytes': payload}, indent=2) + '\n')
print(json.dumps({'passed': True, 'pool_count': len(pools), 'payload_bytes': payload}))
