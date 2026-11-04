#!/usr/bin/env python3
"""Compiler-only checks for explicit scalar-product/vector-add FP32 lowering.

No AIR/AIE binary is executed here. Installed compilation and device numerical
acceptance are separate gates owned by the experiment driver.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--tool', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--air-opt', type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
source = (root/'test/TaskflowAIR/fixtures/gemm.mlir').read_text()
source = source.replace('memref<8x4xf32>', 'memref<8x16xf32>').replace('memref<4x4xf32>', 'memref<4x16xf32>')
source = source.replace('affine.for %j = 0 to 4', 'affine.for %j = 0 to 16')
checks = []


def audit_gemm_indices(text, lanes, tile_shape=(2, 8, 4)):
    """Check actual local address SSA, including cancellation of global shifts.

    The fixture intentionally gives i/j/k different tile extents. This is a
    fixture oracle, not an arithmetic pattern used by the compiler.
    """
    constants, definitions, regions, accesses = {}, {}, [], []
    tile_m, tile_n, tile_k = tile_shape
    def linear(value, seen=()):
        assert value not in seen, ('cyclic index expression', value)
        if value in constants:
            return {} if constants[value] == 0 else {'#constant': constants[value]}
        if value not in definitions:
            return {value: 1}
        op, operands = definitions[value]
        left, right = (linear(x, (*seen, value)) for x in operands)
        if op == 'muli':
            if not left: return {}
            if not right: return {}
            if set(left) == {'#constant'}:
                return {k: v*left['#constant'] for k, v in right.items() if v*left['#constant']}
            if set(right) == {'#constant'}:
                return {k: v*right['#constant'] for k, v in left.items() if v*right['#constant']}
            raise AssertionError(('nonlinear fixture index', value))
        result = dict(left)
        for key, coefficient in right.items():
            result[key] = result.get(key, 0) + (coefficient if op == 'addi' else -coefficient)
        return {k: v for k, v in result.items() if v}
    for line in text.splitlines():
        match = re.search(r'(%[\w.$]+) = "arith.constant"\(\).*value = (-?\d+) : index', line)
        if match: constants[match[1]] = int(match[2])
        match = re.search(r'(%[\w.$]+) = "arith.(addi|subi|muli)"\(([^)]+)\)', line)
        if match: definitions[match[1]] = (match[2], [x.strip() for x in match[3].split(',')])
        if line.rstrip().endswith('({'):
            loop = re.search(r'"scf.for"\(([^)]+)\)', line)
            regions.append(dict(bounds=tuple(constants.get(x.strip()) for x in loop[1].split(',')[:3]), iv=None) if loop else None)
        block = re.search(r'\^bb\d+\((%[\w.$]+): index', line)
        if block and regions and regions[-1] is not None: regions[-1]['iv'] = block[1]
        access = re.search(r'"(memref.load|vector.transfer_read|vector.transfer_write)"\(([^)]+)\)', line)
        if access:
            active = [r for r in regions if r is not None][-3:]
            # The three source loops sit below the compact schedule loops.
            # Infer each axis by its distinct fixture domain, never position.
            # Only source update accesses have all three identifiable axes.
            expected_bounds = [(0, tile_m, 1), (0, tile_n, lanes), (0, tile_k, 1)]
            axes = [[r for r in active if r['bounds'] == bound] for bound in expected_bounds]
            if all(len(axis) == 1 for axis in axes):
                i, j, k = [axis[0]['iv'] for axis in axes]
                shape = re.search(r'memref<(\d+)x(\d+)xf32', line)
                assert shape, line
                shape = tuple(map(int, shape.groups()))
                if access[1] == 'memref.load':
                    assert shape == (tile_m, tile_k), line
                    expected = (i, k)
                elif shape == (tile_k, tile_n):
                    expected = (k, j)
                else:
                    assert shape == (tile_m, tile_n), line
                    expected = (i, j)
                args = [x.strip() for x in access[2].split(',')]
                start = 2 if access[1] == 'vector.transfer_write' else 1
                actual = [linear(x) for x in args[start:start+2]]
                wanted = [{v: 1} for v in expected]
                assert actual == wanted, ('source IV/region shift mismatch', line, actual, wanted)
                accesses.append(dict(operation=access[1], indices=args[start:start+2], simplified=actual, expected=wanted))
        if line.lstrip().startswith('})'):
            regions.pop()
    assert accesses and any(a['operation'] == 'vector.transfer_write' for a in accesses)
    return accesses


def run(name, text=source, lanes=8, expect=True, reason=None, extra=(), reduction=True):
    directory = a.output/name
    directory.mkdir()
    src, air, plan = directory/'source.mlir', directory/'program.air.mlir', directory/'plan.json'
    text_source = text
    src.write_text(text)
    flags = ['--air-distribute-initialization', '--tile-m=2', '--tile-n=8', '--tile-k=4',
        '--implementation=scalar-product-vector-add-f32', f'--vector-lanes={lanes}', *extra]
    cmd = list(map(str, [a.tool, src, *flags, '--lower-to-air', f'--air-output={air}', f'--mapping-output={plan}']))
    (directory/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    r = subprocess.run(cmd, capture_output=True, text=True)
    (directory/'stdout').write_text(r.stdout)
    (directory/'stderr').write_text(r.stderr)
    assert (r.returncode == 0) == expect, (name, r.stderr)
    if reason:
        assert reason in r.stderr, (name, r.stderr)
    if not expect:
        checks.append(dict(name=name, rejected=True))
        return None
    text = air.read_text()
    assert 'scalar-product-vector-add-f32' in text
    assert 'orbit.scalar_f32_builtin' in text and 'orbit.scalarized_lane' in text
    assert not re.search(r'"arith\.(mulf|divf)"\(', text), 'vector arithmetic must not survive as unsupported target operations'
    assert any('"arith.addf"' in line and 'vector<' in line for line in text.splitlines())
    assert not any(x in text for x in ('bf16', 'llvm.fma', 'arith.fma', 'fastmath<contract>', '"vector.reduction"', '"vector.contract"'))
    witness = json.loads(plan.read_text())
    impl = witness['task_implementations']
    assert any(i['kind'] == 'scalar-product-vector-add-f32' and i['lanes'] == lanes for i in impl)
    if reduction:
        assert any(i['kind'] == 'scalar' for i in impl), 'source initializer remains explicitly scalar'
    updates = {i['task'] for i in impl if i['kind'] == 'scalar-product-vector-add-f32'}
    k_bounds = {(bound['lower'], bound['upper']) for phase in witness['phases'] if phase['task'] in updates
        for bound in phase['loops'] if bound['loop'] == 2}
    if reduction:
        assert k_bounds == {(0, 4), (4, 8)}, 'two source-ordered K blocks must remain in the checked plan'
        assert any(r['end_phase_exclusive'] - r['begin_phase'] == 2 for r in witness['retained_regions'])
    # Inspect actual emitted SCF, rather than trusting loop-order metadata.
    # Generic MLIR prints each region opener/closer on separate lines here.
    constants, regions, call_loops = {}, [], []
    for line in text.splitlines():
        constant = re.search(r'(%[\w.$]+) = "arith.constant"\(\).*value = (-?\d+) : index', line)
        if constant:
            constants[constant[1]] = int(constant[2])
        loop = re.search(r'"scf.for"\(([^)]+)\) \(\{', line)
        if line.rstrip().endswith('({'):
            bounds = None
            if loop:
                operands = [x.strip() for x in loop[1].split(',')[:3]]
                bounds = tuple(constants.get(x) for x in operands)
            regions.append(bounds)
        if '"func.call"' in line and 'orbit.scalarized_operation' in line:
            active = [r for r in regions if r is not None][-3:]
            assert active
            if reduction:
                assert sum(bound == (0, 4, 1) for bound in active) == 1, 'scalar product must have exactly one source unit-step K ancestor'
            call_loops.append(active[-1])
        if line.lstrip().startswith('})'):
            regions.pop()
    assert call_loops and not regions
    if reduction:
        (directory/'source-index-audit.json').write_text(json.dumps(audit_gemm_indices(text, lanes), indent=2)+'\n')

    checked = subprocess.run(list(map(str, [a.tool, src, *flags, f'--check-air={air}'])), capture_output=True, text=True)
    (directory/'check.stderr').write_text(checked.stderr)
    assert checked.returncode == 0, checked.stderr
    # A fresh checker process must resolve module-level helper declarations
    # through the generic AIR hierarchy, not depend on emitter process state.
    assert src.read_text() == text_source, 'lowering must not rewrite the source file'
    if a.air_opt:
        verified = subprocess.run(list(map(str, [a.air_opt, air, '--air-dependency', '-o', directory/'dependency.air.mlir'])), capture_output=True, text=True)
        (directory/'installed-verifier.stderr').write_text(verified.stderr)
        assert verified.returncode == 0, verified.stderr
    checks.append(dict(name=name, plan_id=witness['plan_id'], air_sha256=hashlib.sha256(air.read_bytes()).hexdigest()))
    return directory

for lanes in (8,):
    directory = run('mul-lanes'+str(lanes), lanes=lanes)
    assert '__mulsf3' in (directory/'program.air.mlir').read_text()
changed = run('division-body', text=source.replace('arith.mulf', 'arith.divf'))
assert '__divsf3' in (changed/'program.air.mlir').read_text()
assert '__mulsf3' not in (changed/'program.air.mlir').read_text()
pointwise = (root/'test/TaskflowAIR/fixtures/add-scale.mlir').read_text()
pointwise = pointwise.replace('memref<4x4xf32>', 'memref<4x16xf32>').replace('affine.for %j = 0 to 4', 'affine.for %j = 0 to 16')
run('pointwise-add-scale', text=pointwise, reduction=False)
run('reject-lanes4-target', lanes=4, expect=False, reason='unavailable')
run('reject-lanes2', lanes=2, expect=False, reason='four-or-eight')
tail = source.replace('memref<8x16xf32>', 'memref<8x17xf32>').replace('memref<4x16xf32>', 'memref<4x17xf32>').replace('affine.for %j = 0 to 16', 'affine.for %j = 0 to 17')
run('reject-incomplete-lanes', text=tail, expect=False)
run('reject-fastmath', text=source.replace('arith.mulf %x, %w : f32', 'arith.mulf %x, %w fastmath<contract> : f32'), expect=False)
# Existing unsupported vector multiplication choice remains rejected.
d = a.output/'old-kind'
d.mkdir()
src = d/'source.mlir'
src.write_text(source)
r = subprocess.run(list(map(str, [a.tool, src, '--air-distribute-initialization', '--tile-m=2', '--tile-n=8', '--tile-k=4', '--implementation=independent-lane-f32', '--vector-lanes=8', '--lower-to-air', f'--air-output={d}/program.air.mlir'])), capture_output=True, text=True)
(d/'stderr').write_text(r.stderr)
assert r.returncode and 'unavailable' in r.stderr
checks.append(dict(name='old-vector-kind-still-unavailable', rejected=True))
# Canonical check must reject an arithmetic helper substitution under same IDs.
original = a.output/'mul-lanes8'
mutant = original/'wrong-helper.air.mlir'
mutant.write_text((original/'program.air.mlir').read_text().replace('__mulsf3', '__divsf3'))
r = subprocess.run(list(map(str, [a.tool, original/'source.mlir', '--air-distribute-initialization', '--tile-m=2', '--tile-n=8', '--tile-k=4', '--implementation=scalar-product-vector-add-f32', '--vector-lanes=8', f'--check-air={mutant}'])), capture_output=True, text=True)
(original/'mutant-check.stderr').write_text(r.stderr)
assert r.returncode
checks.append(dict(name='helper-mutation-rejected', rejected=True))
(a.output/'summary.json').write_text(json.dumps(dict(passed=True, checks=checks, hardware_executed=False,
    capability='compiler-only; installed target ELF/numerics/performance remain separate gates'), indent=2)+'\n')
print(json.dumps(dict(passed=True, checks=len(checks), hardware_executed=False)))
