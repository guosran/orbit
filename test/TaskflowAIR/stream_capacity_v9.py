#!/usr/bin/env python3
"""Compiler-only regression for repeated independent storage stream limits.

The frozen numerical failure remains separate evidence. This test checks the
current target capability boundary, not packet hardware behavior or arithmetic.
"""
import argparse
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--tool', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
T = 'memref<4xf32>'

def source(inputs, expressions):
    outputs = [expr[0] for expr in expressions]
    names = inputs + outputs
    signature = ', '.join(f'%{n}: {T} {{orbit.role = "{("input" if n in inputs else "output")}", orbit.name = "{n}"}}' for n in names)
    def clause(ns):
        return ', '.join('%'+n for n in ns) + ' : ' + ', '.join(T for _ in ns)
    lines = [f'module {{\n  func.func @streams({signature}) attributes {{orbit.noalias}} {{',
        f'    %done:{len(outputs)} = taskflow.task @grouped will_reads({clause(inputs)}) will_writes({clause(outputs)})',
        f'      [original_read_memrefs({clause(inputs)}), original_write_memrefs({clause(outputs)})]',
        '      : (' + ', '.join(T for _ in names) + ') -> (' + ', '.join(T for _ in outputs) + ') {',
        '    ^bb0(' + ', '.join(f'%{n}_b: {T}' for n in names) + '):',
        '      affine.for %i = 0 to 4 {']
    for i, (output, lhs, rhs, op) in enumerate(expressions):
        lines += [f'        %x{i} = affine.load %{lhs}_b[%i] : {T}',
            f'        %y{i} = affine.load %{rhs}_b[%i] : {T}',
            f'        %v{i} = {op} %x{i}, %y{i} : f32',
            f'        affine.store %v{i}, %{output}_b[%i] : {T}']
    lines += ['      }', '      taskflow.yield done_writes(' + ', '.join('%'+n+'_b' for n in outputs) + ' : ' + ', '.join(T for _ in outputs) + ')',
        '    }', '    return', '  }', '}']
    return '\n'.join(lines)+'\n'

four_in = source(['A','B','D','E'], [('Y','A','B','arith.addf'),('Z','D','E','arith.mulf')])
three_out = source(['A','B'], [('Y','A','B','arith.addf'),('Z','A','B','arith.subf'),('W','A','B','arith.mulf')])
two_each = source(['A','B'], [('Y','A','B','arith.addf'),('Z','A','B','arith.mulf')])
checks = []

def run(name, text, tile, accepted):
    d = a.output/name
    d.mkdir()
    src, air, report = d/'source.mlir', d/'program.air.mlir', d/'plan.json'
    src.write_text(text)
    flags = []
    if tile is not None:
        config = d/'target.json'
        config.write_text(json.dumps({'generic_air': {'tile_sizes': [tile]}})+'\n')
        flags.append(f'--target-config={config}')
    cmd = list(map(str, [a.tool, src, *flags, '--lower-to-air', f'--air-output={air}', f'--mapping-output={report}']))
    (d/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    r = subprocess.run(cmd,capture_output=True,text=True)
    (d/'stdout').write_text(r.stdout)
    (d/'stderr').write_text(r.stderr)
    assert (r.returncode == 0) == accepted, (name,r.stderr)
    if not accepted:
        assert 'stream' in r.stderr.lower() or 'port' in r.stderr.lower(), r.stderr
        checks.append(dict(name=name,rejected=True))
        return
    plan = json.loads(report.read_text())
    if tile is None:
        assert plan['strategy'] == 'single-herd-resident'
    else:
        assert plan['strategy'] == 'serial-tiled'
        assert len(plan['phases']) == (4+tile-1)//tile
    verified = subprocess.run(list(map(str, [a.tool,src,*flags,f'--check-air={air}'])),capture_output=True,text=True)
    (d/'check.stderr').write_text(verified.stderr)
    assert verified.returncode == 0, verified.stderr
    checks.append(dict(name=name,plan_id=plan['plan_id'],accepted=True))

run('four-inputs-repeated-rejected',four_in,2,False)
run('three-outputs-repeated-rejected',three_out,2,False)
run('two-inputs-two-outputs-repeated',two_each,2,True)
run('four-inputs-single-phase-preserved',four_in,4,True)
run('four-inputs-resident-preserved',four_in,None,True)
(a.output/'summary.json').write_text(json.dumps(dict(passed=True,checks=checks,
    hardware_executed=False,compiled_to_target_elf=False,
    scope='current serial-tiled target capability; no change to accepted single-phase/resident source behavior'),indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),hardware_executed=False)))
