#!/usr/bin/env python3
"""Exercise explicit unsupported boundaries through the native C++ pipeline."""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--tool', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
base = (root / 'test/TaskflowGeneric/add.mlir').read_text()
cases = {
    'nonidentity_layout': (base.replace('memref<16xf32>', 'memref<16xf32, strided<[2]>>'), ['--generic-analyze'], 'static identity-layout host memref required'),
    'missing_noalias_contract': (base.replace('orbit.noalias, ', ''), ['--generic-analyze'], 'requires orbit.noalias'),
    'inout_interface': (base.replace('orbit.role = "input"', 'orbit.role = "inout"', 1), ['--generic-analyze'], 'inout/alias rejected'),
    'temporary_argument': (base.replace('orbit.role = "input"', 'orbit.role = "temporary"', 1), ['--generic-analyze'], 'temporary reserved for allocations'),
}
feedback = '''module {
  func.func @feedback(%A: memref<4xf32> {orbit.role = "input"}, %C: memref<4xf32> {orbit.role = "output"}) attributes {orbit.noalias} {
    %ready = taskflow.task @copy will_reads(%A : memref<4xf32>) will_writes(%C : memref<4xf32>) [original_read_memrefs(%A : memref<4xf32>), original_write_memrefs(%C : memref<4xf32>)] : (memref<4xf32>, memref<4xf32>) -> memref<4xf32> {
    ^bb0(%a: memref<4xf32>, %c: memref<4xf32>):
      affine.for %i = 0 to 4 {
        %x = affine.load %a[%i] : memref<4xf32>
        affine.store %x, %c[%i] : memref<4xf32>
      }
      taskflow.yield done_writes(%c : memref<4xf32>)
    }
    %done = taskflow.task @update will_reads(%ready : memref<4xf32>) will_writes(%ready : memref<4xf32>) [original_read_memrefs(%C : memref<4xf32>), original_write_memrefs(%C : memref<4xf32>)] : (memref<4xf32>, memref<4xf32>) -> memref<4xf32> {
    ^bb0(%read: memref<4xf32>, %write: memref<4xf32>):
      %two = arith.constant 2.0 : f32
      affine.for %i = 0 to 4 {
        %x = affine.load %read[%i] : memref<4xf32>
        %y = arith.mulf %x, %two : f32
        affine.store %y, %write[%i] : memref<4xf32>
      }
      taskflow.yield done_writes(%write : memref<4xf32>)
    }
    return
  }
}
'''
cases['unvalidated_host_feedback'] = (feedback, ['--lower-to-air', '--tile-m=2', '--tile-n=1', '--tile-k=1'], 'read/write tile requires a proved local source initializer')
checks = []
for name, (source, flags, diagnostic) in cases.items():
    path = a.output / (name + '.mlir')
    path.write_text(source)
    command = [str(a.tool.resolve()), str(path.resolve()), *flags, '--air-output=' + str((a.output / (name + '.air.mlir')).resolve()), '--analysis-output=' + str((a.output / (name + '.analysis.json')).resolve()), '--mapping-output=' + str((a.output / (name + '.mapping.json')).resolve())]
    result = subprocess.run(command, text=True, capture_output=True)
    (a.output / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
    (a.output / (name + '.stdout.log')).write_text(result.stdout)
    (a.output / (name + '.stderr.log')).write_text(result.stderr)
    assert result.returncode != 0 and diagnostic in result.stderr, (name, result.stderr)
    assert not (a.output / (name + '.air.mlir')).exists(), name
    checks.append({'name': name, 'exit_code': result.returncode, 'diagnostic': diagnostic})
(a.output / 'summary.json').write_text(json.dumps({'passed': True, 'checks': checks}, indent=2) + '\n')
print(json.dumps({'passed': True, 'checks': len(checks)}))
