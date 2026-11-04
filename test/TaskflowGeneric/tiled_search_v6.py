#!/usr/bin/env python3
"""Exercise original C++ search and lower its actual MLIR candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool


p = argparse.ArgumentParser()
p.add_argument('--tool', type=Path, default=taskflow_tool('air'))
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
tool = a.tool.resolve()
config = a.output / 'config.json'
config.write_text(json.dumps({
    'generic_air': {'theoretical': {
        'launch_ms': .1, 'per_scalar_operation_ms': .000001,
        'per_transfer_byte_ms': .0000001}},
    'native_air_tile_candidates': [[4,8,32], [8,8,32], [4,16,64], [64,64,128]]
}, indent=2))

def run(name, command):
    (a.output / (name + '.command.json')).write_text(json.dumps(command, indent=2))
    result = subprocess.run(command, text=True, capture_output=True)
    (a.output / (name + '.log')).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (name, result.stderr)

search = a.output / 'search'
run('search', [str(tool), str(a.source.resolve()), '--search', '--backend=air',
    '--air-distribute-initialization', '--tile-m=4', '--tile-n=8', '--tile-k=64',
    '--target-config=' + str(config.resolve()), '--search-output=' + str(search.resolve()),
    '--max-rounds=1', '--max-candidates=5'])
archive = json.loads((search / 'search.json').read_text())['archive']
accepted = []
rejected = []
for row in archive:
    ident = row['candidate_id']
    directory = search / 'candidates' / ident
    if row['status'] != 'passed':
        rejected.append({'candidate_id': ident, 'status': row['status']})
        continue
    run('lower-' + ident, [str(tool), str(directory / 'taskflow.mlir'),
        '--lower-to-air', '--target-config=' + str(config),
        '--analysis-output=' + str(directory / 'facts.json'),
        '--mapping-output=' + str(directory / 'mapping.json'),
        '--air-output=' + str(directory / 'program.air.mlir')])
    plan = json.loads((directory / 'mapping.json').read_text())
    evaluation = json.loads((directory / 'mapping-result.json').read_text())
    assert plan == evaluation['best_plan']
    assert plan['strategy'] == 'serial-tiled'
    accepted.append({'candidate_id': ident, 'plan_id': plan['plan_id'],
        'air_sha256': hashlib.sha256((directory / 'program.air.mlir').read_bytes()).hexdigest(),
        'path': str(directory.resolve())})
assert len(accepted) >= 3, accepted
assert len({r['air_sha256'] for r in accepted}) == len(accepted)
assert rejected, 'expected a rejected capacity candidate'
(a.output / 'summary.json').write_text(json.dumps({
    'accepted': accepted, 'rejected': rejected,
    'compiler': str(tool),
    'compiler_sha256': hashlib.sha256(tool.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
    'hardware_executed': False,
    'cost': 'explicit sensitivity coefficients only; uncalibrated; serial transfers'
}, indent=2))
print(json.dumps({'accepted': len(accepted), 'rejected': len(rejected)}))
