#!/usr/bin/env python3
"""Check rectangular tile lowering and reject same-identity AIR mutations."""
import argparse
import json
import re
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--tool', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--air-opt', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
fixture = root / 'test/TaskflowAIR/fixtures/add-scale.mlir'
config = a.output / 'target.json'
config.write_text(json.dumps({'generic_air': {'tile_sizes': [2, 2], 'capacity_bytes': 192}}))
air = a.output / 'tiled.air.mlir'
plan = a.output / 'plan.json'
base = [str(a.tool), str(fixture), '--generic-map', f'--target-config={config}']

def run(args, success=True):
    r = subprocess.run(list(map(str, args)), capture_output=True, text=True)
    assert (r.returncode == 0) == success, (args, r.stdout, r.stderr)
    return r

run([*base, '--lower-to-air', f'--air-output={air}', f'--mapping-output={plan}'])
run([a.air_opt, air, '--air-dependency', '-o', a.output / 'dependency.air.mlir'])
run([*base, f'--check-air={air}'])
facts = json.loads(plan.read_text())
assert facts['strategy'] == 'serial-tiled'
text = air.read_text()
assert 'orbit.compressed_phases' in text
assert re.search(r'compressed_phases[^\n]+!air.async.token\) -> !air.async.token', text)
assert 'orbit.phase' in text and text.count('"air.herd"') == len(facts['hardware_stages'])
assert '"arith.addf"' in text and '"arith.mulf"' in text
assert re.search(r'static_src_offsets = array<i64: (?:[1-9]|-9223372036854775808)', text)
variants = {
    'arithmetic': text.replace('"arith.addf"', '"arith.subf"', 1),
    'dma_offset': re.sub(r'(static_src_offsets = array<i64: )0', r'\g<1>1', text, count=1),
    'tile_identity': text.replace('orbit.phase = 0 : i64', 'orbit.phase = 1 : i64', 1),
}
for name, changed in variants.items():
    assert changed != text
    path = a.output / f'{name}.air.mlir'
    path.write_text(changed)
    run([*base, f'--check-air={path}'], success=False)
# A reduction tile keeps one invariant output rectangle locally across all K
# pieces, including a shorter final K piece, and clones the actual initializer.
gemm_config = a.output / 'gemm-target.json'
gemm_config.write_text(json.dumps({'generic_air': {'tile_sizes': [2, 3, 3]}}))
gemm_air = a.output / 'gemm.air.mlir'
gemm_plan = a.output / 'gemm-plan.json'
gemm_base = [str(a.tool), str(root / 'test/TaskflowAIR/fixtures/gemm.mlir'), '--generic-map', '--air-distribute-initialization', f'--target-config={gemm_config}']
run([*gemm_base, '--lower-to-air', f'--air-output={gemm_air}', f'--mapping-output={gemm_plan}'])
run([a.air_opt, gemm_air, '--air-dependency', '-o', a.output / 'gemm-dependency.air.mlir'])
run([*gemm_base, f'--check-air={gemm_air}'])
gemm_text = gemm_air.read_text()
assert 'orbit.retained_begin' in gemm_text and 'orbit.retained_end' in gemm_text
assert 'orbit.initializer_task' in gemm_text
assert '.incoming' not in gemm_text
assert json.loads(gemm_plan.read_text())['retained_regions']
(a.output / 'summary.json').write_text(json.dumps({'passed': True, 'same_identity_mutations_rejected': list(variants)}, indent=2) + '\n')
print(json.dumps({'passed': True, 'mutations': len(variants)}))
