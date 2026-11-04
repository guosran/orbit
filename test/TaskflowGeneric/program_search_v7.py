#!/usr/bin/env python3
"""Original search composes tiling with access-derived program residency."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
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
tool, source, out = a.tool.resolve(), a.source.resolve(), a.output.resolve()
config = out / 'config.json'
config.write_text(json.dumps({
    'generic_air': {'theoretical': {'launch_ms': .1,
        'per_scalar_operation_ms': .000001, 'per_transfer_byte_ms': .0000001}},
    'native_air_tile_candidates': [[8, 16, 32], [8, 64, 32]],
    'native_mapping_candidates': 'discover'
}, indent=2))

def run(name, cmd, success=True):
    (out / f'{name}.command.json').write_text(json.dumps(cmd, indent=2))
    r = subprocess.run(cmd, text=True, capture_output=True)
    (out / f'{name}.log').write_text(r.stdout + r.stderr)
    assert (r.returncode == 0) == success, (name, r.returncode, r.stderr)

search = out / 'search'
run('search', [str(tool), str(source), '--search', '--backend=air',
    '--air-distribute-initialization', '--tile-m=8', '--tile-n=8', '--tile-k=32',
    f'--target-config={config}', f'--search-output={search}',
    '--max-rounds=2', '--max-candidates=20', '--beam-width=4'])
report = json.loads((search / 'search.json').read_text())
by_id = {r['candidate_id']: r for r in report['archive']}
accepted, rejected = [], []
for row in report['archive']:
    ident = row['candidate_id']
    directory = search / 'candidates' / ident
    if row['parent']:
        parent = by_id[row['parent']]
        assert row['action_chain'] == parent['action_chain'] + [row['action']]
    if row['status'] != 'passed':
        rejected.append({'candidate_id': ident, 'status': row['status'],
                         'action_chain': row['action_chain']})
        continue
    run('lower-' + ident, [str(tool), str(directory / 'taskflow.mlir'),
        '--lower-to-air', f'--target-config={config}',
        f'--analysis-output={directory}/facts.json',
        f'--mapping-output={directory}/mapping.json',
        f'--air-output={directory}/program.air.mlir'])
    run('check-' + ident, [str(tool), str(directory / 'taskflow.mlir'),
        f'--target-config={config}', f'--check-air={directory}/program.air.mlir'])
    plan = json.loads((directory / 'mapping.json').read_text())
    assert plan == json.loads((directory / 'mapping-result.json').read_text())['best_plan']
    mlir = (directory / 'taskflow.mlir').read_text()
    panel = 'readonly-panel' in mlir
    b_bytes = sum(t['bytes'] for t in plan['transfer_summary']
                  if t['storage_index'] == 1 and t['direction'] == 'host_to_l1')
    accepted.append(dict(candidate_id=ident, parent=row['parent'],
        action_chain=row['action_chain'], plan_id=plan['plan_id'], panel=panel,
        tile_sizes=plan['tile_sizes'], b_transfer_bytes=b_bytes,
        total_transfer_bytes=plan['actual_dma_bytes'], local_payload=plan['resident_bytes'],
        predicted_ms=plan['estimated_ms'], path=str(directory),
        air_sha256=hashlib.sha256((directory / 'program.air.mlir').read_bytes()).hexdigest()))
base = next(r for r in accepted if not r['parent'])
coupled = [r for r in accepted if r['panel'] and r['tile_sizes'] == base['tile_sizes']
           and r['b_transfer_bytes'] < base['b_transfer_bytes']]
assert coupled, 'no access-derived weight reuse candidate'
assert any(r['local_payload'] > base['local_payload'] for r in coupled), 'expected capacity/traffic tradeoff'
assert any(r['panel'] and len(r['action_chain']) >= 2 for r in accepted), 'no joint action chain'
assert rejected, 'expected overflowing panel candidate rejected'
assert len({r['air_sha256'] for r in accepted}) == len(accepted), 'candidate AIR identity collision'
(out / 'summary.json').write_text(json.dumps({
    'accepted': accepted, 'rejected': rejected, 'ranking': report['ranking'],
    'compiler_sha256': hashlib.sha256(tool.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'source_mapping_annotations': False, 'hardware_executed': False,
    'cost': 'Explicit uncalibrated serial sensitivity model; equal scalar work plus planned transfer bytes, no measured score.'
}, indent=2) + '\n')
print(json.dumps({'accepted': len(accepted), 'rejected': len(rejected),
                  'baseline_b_bytes': base['b_transfer_bytes'],
                  'retained_b_bytes': coupled[0]['b_transfer_bytes']}))
