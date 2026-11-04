#!/usr/bin/env python3
"""Standalone core regression; no Neura, AIR, numpy, or hardware dependency."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tool', type=Path, default=taskflow_tool('core'))
    args = parser.parse_args()
    tool = str(args.tool.resolve())
    with tempfile.TemporaryDirectory(prefix='orbit-core-') as directory:
        root = Path(directory)
        def run(path, flags=(), success=True, diagnostic=None):
            process = subprocess.run([tool, str(path), *flags], capture_output=True, text=True)
            assert (process.returncode == 0) == success, process.stderr
            if diagnostic:
                assert diagnostic in process.stderr, process.stderr
            return process
        source = dict(schema='orbit.taskflow.plan.v1', source_id='test', candidate_id='test', transforms=[],
            buffers=[dict(id=n, shape=s, dtype='f32', role=r) for n,s,r in [
                ('A',[5,7],'input'),('B',[7,11],'input'),('C',[5,11],'temporary'),('R',[5,11],'output')]],
            tasks=[dict(id='gemm', depends_on=[], operations=[
                dict(kind='fill', inputs=[], output='C', m=[0,5], n=[0,11]),
                dict(kind='matmul', inputs=['A','B'], output='C', m=[0,5], n=[0,11], k=[0,7], accumulate=True)]),
                dict(id='relu', depends_on=['gemm'], operations=[dict(kind='relu', inputs=['C'], output='R', m=[0,5], n=[0,11])])], outputs=['R'])
        src = root/'source.json'
        src.write_text(json.dumps(source))
        for name, flags in [('base',[]),('tile',['--tile-m=3','--tile-n=6','--tile-k=4']),('fused',['--fuse'])]:
            mlir, plan = root/f'{name}.mlir', root/f'{name}.json'
            run(src, [*flags, f'--emit-mlir={mlir}', f'--emit-plan={plan}'])
            run(mlir, [f'--emit-plan={root}/roundtrip.json'])
            assert json.loads(plan.read_text()) == json.loads((root/'roundtrip.json').read_text())
        tiled=json.loads((root/'tile.json').read_text())
        assert len(tiled['tasks']) == 8
        assert len(tiled['tasks'][0]['operations']) == 3
        fused=json.loads((root/'fused.json').read_text())
        assert len(fused['tasks']) == 1
        assert len(fused['tasks'][0]['operations']) == 3
        run(root/'fused.mlir', ['--fission', f'--emit-plan={root}/fission.json'])
        assert len(json.loads((root/'fission.json').read_text())['tasks']) == 2
        forged=root/'forged.mlir'
        forged.write_text((root/'base.mlir').read_text().replace('arith.mulf','arith.addf',1))
        run(forged, success=False, diagnostic='disagree with the exchange witness')
        run(src, ['--tile-m=-1'], success=False, diagnostic='negative tile sizes')
        invalid = json.loads(json.dumps(source))
        invalid['tasks'][0]['operations'][0]['value'] = 1
        bad_field=root/'nonzero-fill.json'; bad_field.write_text(json.dumps(invalid))
        run(bad_field, success=False, diagnostic='unsupported operation field')
        invalid['tasks'][0]['operations'][0].pop('value')
        invalid['tasks'][1]['operations'][0].update(kind='unknown', inputs=['C'] * 99)
        unknown=root/'unknown99.json'; unknown.write_text(json.dumps(invalid))
        run(unknown, success=False, diagnostic='unsupported operation: unknown')
        run(forged, ['--verify-only'], success=False, diagnostic='disagree with the exchange witness')
        # Extremely large tile steps must terminate at the bounded region.
        run(src, ['--tile-m=9223372036854775807', '--tile-n=9223372036854775807',
                  '--tile-k=9223372036854775807', f'--emit-plan={root}/large-step.json'])
        # Authenticated storage identities still require WAR and WAW edges.
        hazards = dict(schema='orbit.taskflow.plan.v1', outputs=['T'], transforms=[],
            buffers=[dict(id=n, shape=[2,3], dtype='f32', role=r) for n,r in
                     [('X','input'), ('T','output'), ('Y','temporary')]],
            tasks=[dict(id='first',depends_on=[],operations=[dict(kind='relu',inputs=['X'],output='T',m=[0,2],n=[0,3])]),
                   dict(id='reader',depends_on=['first'],operations=[dict(kind='relu',inputs=['T'],output='Y',m=[0,2],n=[0,3])]),
                   dict(id='writer',depends_on=['first'],operations=[dict(kind='relu',inputs=['X'],output='T',m=[0,2],n=[0,3])])])
        hazard_path=root/'hazard.json'
        hazard_path.write_text(json.dumps(hazards))
        run(hazard_path, success=False, diagnostic='missing RAW/WAR/WAW dependency: reader -> writer on T')
        hazards['tasks'].pop(1)
        hazards['tasks'][-1]['depends_on']=[]
        hazard_path.write_text(json.dumps(hazards))
        run(hazard_path, success=False, diagnostic='missing RAW/WAR/WAW dependency: first -> writer on T')
        hazards['tasks'][-1]['depends_on']=['first']
        hazard_path.write_text(json.dumps(hazards))
        run(hazard_path, [f'--emit-plan={root}/ordered-waw.json'])
        # A forged legacy replica witness must not bypass absent backend proof.
        body='''module { func.func @legacy(%base: memref<4xf32>) {
          %a = taskflow.task @a will_writes(%base : memref<4xf32>)
            [original_write_memrefs(%base : memref<4xf32>)]
            {amoeba.replica.output_counter_axes = array<i64: 0>}
            : (memref<4xf32>) -> (memref<4xf32>) {
            ^bb0(%out: memref<4xf32>):
              taskflow.yield done_writes(%out : memref<4xf32>)
          }
          %b = taskflow.task @b will_writes(%base : memref<4xf32>)
            [original_write_memrefs(%base : memref<4xf32>)]
            : (memref<4xf32>) -> (memref<4xf32>) {
            ^bb0(%out: memref<4xf32>):
              taskflow.yield done_writes(%out : memref<4xf32>)
          }
          %j = taskflow.join states(%a, %b) base(%base) axis(0)
            region([0], [4]) : memref<4xf32>
          return
        }}'''
        legacy=root/'legacy.mlir'; legacy.write_text(body)
        run(legacy, ['--verify-only'], success=False, diagnostic='no output-coordinate proof provider registered')
    print('Taskflow core: materialization, boundary tiling, K sequence, fusion/fission, witness authentication, legacy fail-closed: PASS')


if __name__ == '__main__':
    main()
