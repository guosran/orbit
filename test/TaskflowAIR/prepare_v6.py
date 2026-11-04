#!/usr/bin/env python3
"""Test fixtures and independent equations; no production planner or lowerer.

Dimensions are substituted in existing test source, never in generated AIR.
NumPy float64 equations supply references, not executable fallback kernels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def fixture(case, m, k, n):
    base = Path(__file__).parent / 'fixtures'
    source = (base / ('ffn.mlir' if case == 'ffn' else case + '.mlir')).read_text()
    if case == 'gemm':
        shapes = {'4x8': f'{m}x{k}', '8x4': f'{k}x{n}', '4x4': f'{m}x{n}'}
        bounds = {'i': m, 'j': n, 'k': k}
    elif case == 'add-scale':
        shapes = {'4x4': f'{m}x{n}'}
        bounds = {'i': m, 'j': n}
    else:
        shapes = {'2x8': f'{m}x{k}', '8x16': f'{k}x{n}',
                  '16x8': f'{n}x{k}', '2x16': f'{m}x{n}'}
        # Bounds depend on which test task owns the loop.
        parts = source.split('taskflow.task ')
        for i in range(1, len(parts)):
            is_down = parts[i].startswith('@down ')
            bounds = {'i': m, 'j': k if is_down else n, 'k': n if is_down else k}
            parts[i] = re.sub(r'affine.for %(i|j|k) = 0 to \d+',
                              lambda match: f'affine.for %{match[1]} = 0 to {bounds[match[1]]}', parts[i])
        source = 'taskflow.task '.join(parts)
        bounds = {}
    source = re.sub(r'(?<=memref<)\d+x\d+(?=xf32)', lambda match: shapes[match[0]], source)
    source = re.sub(r'affine.for %(i|j|k) = 0 to \d+',
                    lambda match: f'affine.for %{match[1]} = 0 to {bounds[match[1]]}', source) if bounds else source
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['gemm', 'add-scale', 'ffn'], required=True)
    p.add_argument('--m', type=int, required=True)
    p.add_argument('--k', type=int, default=128)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 47])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-only', action='store_true')
    a = p.parse_args()
    if min(a.m, a.k, a.n) <= 0:
        p.error('dimensions must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    text = fixture(a.case, a.m, a.k, a.n)
    (a.output / 'source.mlir').write_text(text)
    records = []
    if not a.source_only:
        for seed in a.seeds:
            rng = np.random.default_rng(seed)
            def draw(shape, scale=1):
                v = (rng.uniform(-0.5, 0.5, shape) * scale).astype(np.float32)
                v.flat[0] = 0
                return v
            if a.case == 'gemm':
                x, w = draw((a.m, a.k)), draw((a.k, a.n))
                inputs = {'A': x, 'B': w}
                reference = {'C': (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)}
            elif a.case == 'add-scale':
                x, y = draw((a.m, a.n)), draw((a.m, a.n))
                inputs = {'A': x, 'B': y}
                reference = {'C': (2 * (x.astype(np.float64) + y.astype(np.float64))).astype(np.float32)}
            else:
                x = draw((a.m, a.k))
                gate, up = draw((a.k, a.n), 1 / np.sqrt(a.k)), draw((a.k, a.n), 1 / np.sqrt(a.k))
                down = draw((a.n, a.k), 1 / np.sqrt(a.n))
                inputs = {'X': x, 'W_gate': gate, 'W_up': up, 'W_down': down}
                g, u = x.astype(np.float64) @ gate.astype(np.float64), x.astype(np.float64) @ up.astype(np.float64)
                h = g / (1 + np.exp(-g)) * u
                reference = {'Y': (h @ down.astype(np.float64)).astype(np.float32)}
                records.append({'seed': seed, 'gate_min': float(g.min()), 'gate_max': float(g.max()),
                                'hidden_min': float(h.min()), 'hidden_max': float(h.max())})
            np.savez(a.output / f'inputs-{seed}.npz', **inputs)
            np.savez(a.output / f'reference-{seed}.npz', **reference)
    metadata = dict(case=a.case, m=a.m, k=a.k, n=a.n, seeds=a.seeds,
                    source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    reference='independent explicit float64 equations, cast final output to f32',
                    distribution='uniform[-.5,.5], first element zero; FFN weights scaled by inverse sqrt fan-in',
                    reference_ranges=records)
    (a.output / 'fixture.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
