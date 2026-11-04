#!/usr/bin/env python3
"""Compute/dependency-only whole-program fixture and independent references."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--m', type=int, default=32)
p.add_argument('--k', type=int, default=128)
p.add_argument('--n', type=int, default=64)
a = p.parse_args()
assert min(a.m, a.k, a.n) > 0
a.output.mkdir(parents=True, exist_ok=False)
source = (Path(__file__).parent / 'fixtures/gemm-scale.mlir').read_text()
shapes = {'4x8': f'{a.m}x{a.k}', '8x4': f'{a.k}x{a.n}', '4x4': f'{a.m}x{a.n}'}
source = re.sub(r'(?<=memref<)\d+x\d+(?=xf32)', lambda m: shapes[m[0]], source)
bounds = {'i': a.m, 'j': a.n, 'k': a.k}
source = re.sub(r'affine.for %(i|j|k) = 0 to \d+',
                lambda m: f'affine.for %{m[1]} = 0 to {bounds[m[1]]}', source)
assert not any(x in source for x in ['orbit.mapping', 'orbit.air_tile_sizes', 'channel', 'FIFO'])
(a.output / 'source.mlir').write_text(source)
for seed in [11, 23, 47]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-.5, .5, (a.m, a.k)).astype(np.float32)
    w = rng.uniform(-.5, .5, (a.k, a.n)).astype(np.float32)
    x.flat[0] = w.flat[0] = 0
    ref = (.5 * (x.astype(np.float64) @ w.astype(np.float64))).astype(np.float32)
    np.savez(a.output / f'inputs-{seed}.npz', A=x, B=w)
    np.savez(a.output / f'reference-{seed}.npz', Y=ref)
(a.output / 'fixture.json').write_text(json.dumps({
    'case': 'gemm-scale', 'm': a.m, 'k': a.k, 'n': a.n,
    'seeds': [11, 23, 47], 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'reference': 'independent float64 GEMM followed by exact scale .5, cast final output to f32',
    'source_decisions': 'calculation, storage ABI and task dependency only; no mapping directives'
}, indent=2) + '\n')
