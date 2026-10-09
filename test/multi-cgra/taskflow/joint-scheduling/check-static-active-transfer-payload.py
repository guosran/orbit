#!/usr/bin/env python3
"""Regression: capacity spelling cannot disable active-transfer verification."""
import json
import subprocess
import sys
from pathlib import Path

optimizer, fixture, output = sys.argv[1:]
root = Path(output)
root.mkdir(parents=True, exist_ok=True)
source = Path(fixture).read_text()

def run(path, label, prove=False, reject=False):
    facts, result = root / (label + '.json'), root / (label + '.mlir')
    command = [optimizer, str(path), '--verify-each']
    if prove:
        command += ['--prove-static-active-transfer-shapes=function=active_payload arguments=0']
    command += [f'--extract-joint-task-graph-facts=function=active_payload output={facts} fact-only=true', '-o', str(result)]
    proc = subprocess.run(command, text=True, capture_output=True)
    (root / (label + '.log')).write_text(proc.stdout + proc.stderr)
    if reject:
        assert proc.returncode != 0, 'tampered proof was silently accepted'
        assert 'active' in proc.stderr.lower(), proc.stderr
        return
    assert proc.returncode == 0, proc.stderr
    edges = json.loads(facts.read_text())['dependencies']
    assert len(edges) == 1
    return edges[0]['payload_bits'], result

for spelling in ['static', 'dynamic']:
    path = root / (spelling + '-input.mlir')
    path.write_text(source if spelling == 'static' else source.replace('memref<4xi32>', 'memref<?xi32>'))
    assert run(path, spelling + '-full')[0] == 128
    bits, proved = run(path, spelling + '-active', prove=True)
    assert bits == 32, (spelling, bits)
    bad = root / (spelling + '-tampered.mlir')
    text = proved.read_text()
    assert 'amoeba.active_transfer_shape = array<i64: 1>' in text
    bad.write_text(text.replace('amoeba.active_transfer_shape = array<i64: 1>', 'amoeba.active_transfer_shape = array<i64: 2>'))
    run(bad, spelling + '-reject', reject=True)
print('static/dynamic: full capacity, verified active payload, tampered proof rejection PASS')
