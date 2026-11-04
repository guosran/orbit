#!/usr/bin/env python3
"""Public policy normalization and capacity-coupled native retention (CPU only)."""
import argparse
import json
import re
import subprocess
from pathlib import Path
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool


p = argparse.ArgumentParser()
p.add_argument('--tool', type=Path, default=taskflow_tool('air'))
p.add_argument('--output', type=Path, required=True)
p.add_argument('--air', action='store_true', help='also lower/check actual AIR with an AIR-enabled tool')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
source = (root / 'test/TaskflowAIR/fixtures/gemm-scale.mlir').read_text()
shapes = {'4x8': '32x128', '8x4': '128x64', '4x4': '32x64'}
source = re.sub(r'(?<=memref<)\d+x\d+(?=xf32)', lambda m: shapes[m[0]], source)
source = re.sub(r'affine.for %(i|j|k) = 0 to \d+',
                lambda m: f'affine.for %{m[1]} = 0 to {dict(i=32,j=64,k=128)[m[1]]}', source)
path = a.output / 'source.mlir'
path.write_text(source)

def run(name, extra=(), n=16, text=None, ok=True):
    src = path
    if text is not None:
        src = a.output / (name + '.source.mlir')
        src.write_text(text)
    ir = a.output / (name + '.mlir')
    report = a.output / (name + '.json')
    cmd = [str(a.tool), str(src), '--generic-map', '--air-distribute-initialization',
           '--tile-m=8', f'--tile-n={n}', '--tile-k=32',
           f'--mapping-output={report}', f'--emit-mlir={ir}', *extra]
    if a.air:
        cmd += ['--lower-to-air', f'--air-output={a.output / (name + ".air.mlir")}']
    r = subprocess.run(cmd, text=True, capture_output=True)
    (a.output / (name + '.command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
    (a.output / (name + '.stdout')).write_text(r.stdout)
    (a.output / (name + '.stderr')).write_text(r.stderr)
    assert (r.returncode == 0) == ok, (name, r.returncode, r.stderr)
    if ok and a.air:
        checked = subprocess.run([*cmd[:2], '--air-distribute-initialization',
            '--tile-m=8', f'--tile-n={n}', '--tile-k=32', *extra,
            f'--check-air={a.output / (name + ".air.mlir")}'],
            text=True, capture_output=True)
        (a.output / (name + '.check.stderr')).write_text(checked.stderr)
        assert checked.returncode == 0, (name, checked.stderr)
    return json.loads(report.read_text()) if ok else r.stderr

scope = ['--resident-storage=1', '--reuse-loop=0', '--panel-loop=1']
legacy = run('legacy', ['--residency=readonly-panel', *scope])
public = run('public', ['--data-organization=retain-across-users', *scope])
assert legacy == public
assert (a.output/'legacy.mlir').read_bytes() == (a.output/'public.mlir').read_bytes()
if a.air:
    assert (a.output/'legacy.air.mlir').read_bytes() == (a.output/'public.air.mlir').read_bytes()
base = run('baseline')
default = run('explicit-default', ['--data-organization=phase-local'])
assert base == default
assert (a.output/'baseline.mlir').read_bytes() == (a.output/'explicit-default.mlir').read_bytes()
assert base['body_fingerprint'] == public['body_fingerprint']
assert base['plan_id'] != public['plan_id']
assert public['program_mapping_choices']['data_organization'] == 'retain-across-users'
assert public['resident_bytes'] == 9728
assert public['actual_dma_bytes'] < base['actual_dma_bytes']
assert public['readonly_retentions'] and not base['readonly_retentions']
assert 'AIR stage-local' in public['data_organization_scope']
# The opportunity is discovered from source accesses, regardless of whether its
# chosen panel fits. The wide panel cannot be retained under the same 24KiB cap.
wide = run('wide-baseline', n=64)
why = run('wide-retain', ['--data-organization=retain-across-users', *scope], n=64, ok=False)
assert any(word in why.lower() for word in ['capacity', 'exceed', 'resident'])
assert wide['capacity_bytes'] == public['capacity_bytes'] == 24576
assert any(x['data_organization'] == 'retain-across-users' and x['storage_index'] == 1
           for x in wide['discovered_mapping_opportunities'])
for retention in public['readonly_retentions']:
    assert any(g['begin_phase'] == retention['begin_phase'] and g['end_phase_exclusive'] == retention['end_phase_exclusive']
               for g in public['execution_groups'])
# Raw MLIR requests must fail closed rather than silently overriding ambiguity.
for name, attrs in {
    'conflict': 'data_organization = "phase-local", residency = "readonly-panel", storage = 1 : i64, reuse_loop = 0 : i64, panel_loop = 1 : i64',
    'grouping': 'data_organization = "retain-across-users", grouping = "task-shape-stages", storage = 1 : i64, reuse_loop = 0 : i64, panel_loop = 1 : i64',
    'unknown': 'data_organization = "phase-local", typo = "phase-local"',
    'type': 'data_organization = 1 : i64',
    'empty': 'data_organization = ""',
}.items():
    run('reject-' + name, text=source.replace('module {', 'module attributes {orbit.mapping = {' + attrs + '}} {', 1), ok=False)
# Actual source arithmetic is retained, including a non-GEMM expression.
changed = run('changed-body', ['--data-organization=retain-across-users', *scope], text=source.replace('arith.mulf', 'arith.subf'))
assert changed['body_fingerprint'] != public['body_fingerprint']
if a.air:
    text = (a.output/'changed-body.air.mlir').read_text()
    assert '"arith.subf"' in text and '"arith.mulf"' not in text
    assert 'orbit.readonly_retained_begin' in text and '.readonly-panel' in text
    assert 'retain-across-users' in text and 'no cross-stage SRAM retention' in text
(a.output/'summary.json').write_text(json.dumps({'passed': True, 'hardware_execution': False,
    'air_checked': a.air, 'wide_baseline_local_bytes': wide['resident_bytes'],
    'retained_local_bytes': public['resident_bytes'],
    'baseline_dma_bytes': base['actual_dma_bytes'], 'retained_dma_bytes': public['actual_dma_bytes']}, indent=2)+'\n')
print('public data organization: alias/default identity, capacity coupling, conflicts and body preservation PASS')
