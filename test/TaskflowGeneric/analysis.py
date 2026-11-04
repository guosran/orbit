#!/usr/bin/env python3
"""Native C++ analysis diagnostics; no Python planning or semantic oracle."""
import argparse
import json
import subprocess
from pathlib import Path
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool


ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tool', type=Path, default=taskflow_tool('core'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = (ROOT/'test/TaskflowGeneric/add.mlir').read_text()
    checks = []
    def check(name, text, diagnostic=None):
        path = args.output/(name+'.mlir'); path.write_text(text)
        output = args.output/(name+'.json')
        cmd = [str(args.tool.resolve()), str(path.resolve()), '--generic-analyze', '--analysis-output='+str(output.resolve())]
        result = subprocess.run(cmd, capture_output=True, text=True)
        (args.output/(name+'.log')).write_text(result.stdout+result.stderr)
        (args.output/(name+'.command.json')).write_text(json.dumps(cmd, indent=2)+'\n')
        if diagnostic:
            assert result.returncode != 0 and diagnostic in result.stderr, (name, result.stderr)
            checks.append({'name':name, 'rejected':diagnostic}); return
        assert result.returncode == 0, (name,result.stderr)
        value = json.loads(output.read_text())
        assert value['analysis_status']=='analyzed' and value['cost_status']=='uncalibrated'
        checks.append({'name':name, 'passed':True}); return value
    original=check('native_add',base)
    extra=base.replace('affine.for %i', '%zero = arith.constant 0.0 : f32\n      affine.for %i').replace('affine.store %s,','%different = arith.addf %s, %zero : f32\n        affine.store %different,')
    equivalent=check('equivalent_extra_arithmetic',extra)
    assert original['tasks']==equivalent['tasks']
    assert original['body_fingerprint']!=equivalent['body_fingerprint']
    changed=check('changed_actual_body',base.replace('arith.addf','arith.mulf'))
    assert changed['body_fingerprint']!=original['body_fingerprint']
    scf=base.replace('affine.for %i = 0 to 16 {','%lb = arith.constant 0 : index\n      %ub = arith.constant 16 : index\n      %step = arith.constant 1 : index\n      scf.for %i = %lb to %ub step %step {').replace('affine.load','memref.load').replace('affine.store','memref.store')
    check('static_scf',scf)
    task_block = base[base.index('    %out ='):base.index('    return')]
    first = task_block.replace('0 to 16', '0 to 8')
    second = task_block.replace('%out =', '%out2 =').replace('@add ', '@second ').replace('0 to 16', '8 to 16').replace('will_writes(%c :', 'will_writes(%out :')
    halves = base[:base.index('    %out =')] + first + second + base[base.index('    return'):]
    check('two_tasks_exact_output_union', halves)
    conflict = second.replace('8 to 16','0 to 16').replace('will_writes(%out :','will_writes(%c :')
    check('missing_waw',base[:base.index('    %out =')]+task_block+conflict+base[base.index('    return'):],'missing RAW/WAR/WAW')

    check('out_of_bounds',base.replace('0 to 16','0 to 17'),'out-of-bounds')
    check('partial_output',base.replace('0 to 16','0 to 15'),'output not completely initialized')
    check('missing_interface',base.replace('orbit.role = "input", ', '',1),'missing orbit.role')
    check('alias_contract',base.replace('orbit.noalias, ',''),'requires orbit.noalias')
    check('in_place_interface',base.replace('orbit.role = "input"','orbit.role = "inout"',1),'inout/alias rejected')
    check('read_uninitialized',base.replace('affine.load %x[%i]','affine.load %z[%i]'),'read of uninitialized')
    # A store's full iteration-domain rectangle is not available in the
    # current iteration. Only matching indices in shared loops may prove a
    # read initialized. Independent completed earlier loops remain valid.
    store='affine.store %s, %z[%i] : memref<16xf32>'
    for index in ('3','0'):
        check('cross_iteration_read_'+index,base.replace(store,store+'\n        %probe = affine.load %z['+index+'] : memref<16xf32>'),'read of uninitialized')
    check('same_iteration_read',base.replace(store,store+'\n        %probe = affine.load %z[%i] : memref<16xf32>'))
    scf_store='memref.store %s, %z[%i] : memref<16xf32>'
    scf_bad=scf.replace(scf_store,scf_store+'\n        %last = arith.constant 3 : index\n        %probe = memref.load %z[%last] : memref<16xf32>')
    check('scf_cross_iteration_read',scf_bad,'read of uninitialized')
    check('noninjective_constant_output',base.replace(store,'affine.store %s, %z[0] : memref<16xf32>'),'output not completely initialized')

    check('write_input',base.replace('affine.store %s, %z','affine.store %s, %x'),'write absent')
    chain=(ROOT/'test/TaskflowAIR/fixtures/add-scale.mlir').read_text()
    c=check('multi_task_storage_dependency',chain)
    check('nonrectangular_initialization',chain.replace('affine.store %sum, %T_b[%i, %j]', 'affine.store %sum, %T_b[%i, %i]'),'read of uninitialized')
    check('oversized_loop_domain',base.replace('0 to 16','0 to 9223372036854775807'),'invalid loop domain')
    check('invalid_noalias_attribute',base.replace('orbit.noalias,','orbit.noalias = \"yes\",'),'orbit.noalias must be unit or true')
    assert c['tasks'][1]['dependencies']==['add']
    check('missing_dependency',chain.replace('will_reads(%T_done :','will_reads(%T :').replace('original_read_memrefs(%T_done :','original_read_memrefs(%T :'),'missing RAW/WAR/WAW')
    marker='    %C_done = taskflow.task'
    check('use_after_free',chain.replace(marker,'    memref.dealloc %T : memref<4x4xf32>\n'+marker),'access after deallocation')
    check('double_free',chain.replace('    return','    memref.dealloc %T : memref<4x4xf32>\n    return'),'invalid/double free')
    for name in ('gemm','ffn'):
        check(name,(ROOT/f'test/TaskflowAIR/fixtures/{name}.mlir').read_text())
    # A corrupt historical witness cannot alter native analysis decisions.
    witness=check('ignored_legacy_witness',base.replace('module {','module attributes {orbit.plan = "not a program"} {',1))
    assert witness==original
    summary={'passed':True,'checks':checks,'count':len(checks),'decision_engine':'native C++ actual-body analysis','python_role':'test launch and assertions only'}
    (args.output/'checks.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'passed':True,'count':len(checks)}))

if __name__=='__main__': main()
