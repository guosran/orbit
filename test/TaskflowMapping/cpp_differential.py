#!/usr/bin/env python3
"""Native regression harness; fixtures and explicit expected outcomes only."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool


ROOT=Path(__file__).resolve().parents[2]


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def native_fingerprint(obj):
    # Independent implementation of LLVM JSON OStream canonical bytes:
    # lexicographically sorted keys, UTF8 strings, double max_digits10 %.17g.
    def encode(value):
        if value is None:return 'null'
        if isinstance(value,bool):return 'true' if value else 'false'
        if isinstance(value,int):return str(value)
        if isinstance(value,float):return format(value,'.17g')
        if isinstance(value,str):return json.dumps(value,ensure_ascii=False)
        if isinstance(value,list):return '['+','.join(encode(x) for x in value)+']'
        return '{'+','.join(encode(k)+':'+encode(value[k]) for k in sorted(value))+'}'
    return hashlib.sha256(encode(obj).encode()).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def run(command,log):
    completed=subprocess.run(list(map(str,command)),text=True,capture_output=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    log.write_text('$ '+ ' '.join(map(str,command))+'\n'+completed.stdout+completed.stderr)
    return completed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tool',type=Path,default=taskflow_tool('core'));parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    tool=args.tool.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('use a new evidence directory')
    out.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('static_fixture_suite',ROOT/'test/TaskflowMapping/check.py');suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
    counter=0;negative_counter=0;records=[];negative_records=[];materialized={};resign_probed=False

    def materialize(program):
        key=fingerprint(program)
        if key in materialized:return materialized[key]
        d=out/'materialized'/key;d.mkdir(parents=True)
        save(d/'source.json',program)
        result=run([tool,d/'source.json','--emit-plan='+str(d/'program.json'),'--emit-mlir='+str(d/'taskflow.mlir')],d/'materialize.log')
        assert result.returncode==0,result.stderr
        p=json.loads((d/'program.json').read_text());materialized[key]=(d,p)
        return d,p

    def checked_native(program,config,plan,folder):
        d,normalized=materialize(program);save(folder/'config.json',config);save(folder/'plan.json',plan)
        command=[tool,d/'taskflow.mlir','--target-config='+str(folder/'config.json'),'--check-mapping='+str(folder/'plan.json')]
        checked=run(command,folder/'check.log')
        try:verdict=json.loads(checked.stdout)
        except ValueError:verdict=dict(passed=False,errors=[checked.stderr.strip() or checked.stdout.strip()])
        assert bool(verdict.get('passed'))==(checked.returncode==0),'native checker status/exit mismatch'
        return verdict

    def native_map(program,config):
        nonlocal counter,resign_probed
        index=counter;counter+=1;d=out/'native'/f'{index:03d}';d.mkdir(parents=True)
        source,normalized=materialize(program);save(d/'config.json',config)
        cmd=[tool,source/'taskflow.mlir','--static-map','--target-config='+str(d/'config.json'),'--mapping-output='+str(d/'mapping.json'),'--analysis-output='+str(d/'analysis.json')]
        completed=run(cmd,d/'map.log')
        if (d/'mapping.json').exists():actual=json.loads((d/'mapping.json').read_text())
        else:actual=dict(status='unsupported',best_plan=None,alternatives=[],reasons=[completed.stderr])
        rows=[]
        for number,row in enumerate(actual.get('alternatives',[])):
            if row['status']!='passed':continue
            plan=row['plan'];assignment=tuple(sorted(row['assignment'].items()))
            native=checked_native(program,config,plan,d/f'plan-{number}')
            assert native['passed'],native
            if not resign_probed:
                resigned=copy.deepcopy(plan)
                resigned['program_id']=native_fingerprint(normalized)
                resigned['configuration_id']=native_fingerprint(config)
                resigned['plan_id']=native_fingerprint({k:v for k,v in resigned.items() if k not in ('plan_id','validation')})
                proof=checked_native(program,config,resigned,out/'unchanged-resign-proof')
                assert proof['passed'],('independent re-sign algorithm not validated',proof)
                resign_probed=True
            rows.append(dict(assignment=dict(assignment),native_checked=True,score=row['score']))
        records.append(dict(index=index,passed=True,feasible_assignments=len(rows),plans=rows,status=actual['status']))
        return actual

    def negative_check(program,config,plan):
        nonlocal negative_counter
        index=negative_counter;negative_counter+=1;d=out/'negative'/f'{index:03d}'
        source,normalized=materialize(program)
        save(d/'program.json',program);save(d/'config.json',config);save(d/'mutated-plan.json',plan)
        # The shared native canonical hashing algorithm is tested by an
        # unchanged-plan roundtrip before this suite accepts any negative.
        assert resign_probed,'must validate independent native re-sign before negatives'
        plan=copy.deepcopy(plan);plan['program_id']=native_fingerprint(normalized);plan['configuration_id']=native_fingerprint(config)
        plan['identity_scheme']='llvm-json-sorted-17g-sha256'
        plan['candidate_id']=normalized['candidate_id']
        plan['plan_id']=native_fingerprint({k:v for k,v in plan.items() if k not in ('plan_id','validation')})
        verdict=checked_native(program,config,plan,d)
        save(d/'verdict.json',verdict);negative_records.append(dict(index=index,passed=not verdict['passed'],errors=verdict['errors']))
        return verdict

    suite.map_program=native_map;suite.check_plan=negative_check
    old_argv=sys.argv
    try:
        sys.argv=[str(ROOT/'test/TaskflowMapping/check.py'),'--output',str(out/'fixture-suite')]
        suite.main()
    finally:sys.argv=old_argv
    # The corpus's independent negative labels are preserved with each payload.
    fixture_checks=json.loads((out/'fixture-suite/checks.json').read_text())
    for record,label in zip(negative_records,fixture_checks['negative_checks']):
        record['name']=label['name'];save(out/'negative'/f'{record["index"]:03d}'/'label.json',label)
    body_checks=[]
    program,config=suite.fixture();materialized_dir,_=materialize(program)
    text=(materialized_dir/'taskflow.mlir').read_text()
    def body_case(label,content,should_pass=None):
        d=out/'body-checks'/label;d.mkdir(parents=True);(d/'input.mlir').write_text(content);save(d/'config.json',config)
        proc=run([tool,d/'input.mlir','--static-map','--target-config='+str(d/'config.json'),'--mapping-output='+str(d/'mapping.json'),'--analysis-output='+str(d/'analysis.json')],d/'run.log')
        passed=proc.returncode==0 and (d/'mapping.json').exists() and json.loads((d/'mapping.json').read_text())['status']=='passed'
        if should_pass is not None:assert passed==should_pass,(label,proc.stdout,proc.stderr)
        if passed:
            facts=json.loads((d/'analysis.json').read_text());assert facts['semantic_source']=='actual_taskflow_mlir_body'
        body_checks.append(dict(name=label,accepted=passed,expected='accept or explicit reject' if should_pass is None else should_pass,diagnostic=proc.stderr.strip()))
    body_case('authentic-witness',text,True)
    body_case('body-changed-stale-witness',text.replace('arith.maximumf','arith.minimumf',1),False)
    body_case('zero-clamp-changed-stale-witness',text.replace('arith.maximumf %0, %cst :','arith.maximumf %0, %cst_2 :',1),False)
    no_witness='module {\n'+text.split('\n',1)[1]
    body_case('no-witness',no_witness,None)
    body_case('unsupported-body-without-witness',no_witness.replace('arith.maximumf','arith.minimumf',1),False)
    # A valid changed body + matching witness must change analyzed regions
    # and complete plan cost, not reuse a shape-blind cached result.
    config['implementations']=config['implementations'][:1]
    config['implementations'][0]['cost'].update(fixed_ms=0,per_element_ms=1)
    old=native_map(program,config)
    changed=copy.deepcopy(program)
    for b in changed['buffers']:b['shape'][0]=3
    for t in changed['tasks']:
        for op in t['operations']:op['m']=[0,3]
    new=native_map(changed,config)
    assert old['best_plan']['makespan_ms']==8 and new['best_plan']['makespan_ms']==12
    body_checks.append(dict(name='changed-valid-body-and-witness',accepted=True,old_makespan_ms=8,new_makespan_ms=12))
    def extra_negative(label,case,mutation):
        d=out/'fixture-suite'/case
        p=json.loads((d/'program.json').read_text());c=json.loads((d/'config.json').read_text());plan=copy.deepcopy(json.loads((d/'mapping.json').read_text())['best_plan'])
        mutation(plan)
        verdict=negative_check(p,c,plan)
        assert not verdict['passed'],label
        assert any(not any(word in e.lower() for word in ('identity mismatch','fingerprint mismatch','hash mismatch')) for e in verdict['errors']),('negative rejected only by checksum',label,verdict)
        negative_records[-1]['name']=label
        save(out/'negative'/f'{negative_records[-1]["index"]:03d}'/'label.json',dict(name=label,errors=verdict['errors']))
    extra_negative('forged region cell count','communication-reversal',lambda p:p.update(region_cell_count=p['region_cell_count']+1))
    extra_negative('forged source step expression','communication-reversal',lambda p:p['steps'][0]['operation'].update(kind='fill'))
    extra_negative('forged task operation binding','communication-reversal',lambda p:p['events'][0].update(operation_index=99))
    extra_negative('conflicting implementation aliases','communication-reversal',lambda p:p['events'][0].update(implementation='npu'))
    extra_negative('conflicting provenance aliases','communication-reversal',lambda p:p['events'][0].update(cost_provenance={'kind':'measured','source':'forged'}))
    extra_negative('forged external output consumer','communication-reversal',lambda p:p['output_consumers'].update(Y='npu'))
    extra_negative('unsupported warm initial state','communication-reversal',lambda p:p.update(initial_state='free_preloaded_weights'))
    extra_negative('forged event cost boundary','communication-reversal',lambda p:p['events'][0].update(cost_boundary='opaque_graph_already_includes_communication'))
    extra_negative('forged final output metadata','communication-reversal',lambda p:p['outputs'][0].update(version='initial:Y'))
    extra_negative('overflowing allocation byte declaration','communication-reversal',lambda p:p['allocations'][0].update(bytes=2**63))
    def remove_output_sync(plan):
        selected={e['id'] for e in plan['events'] if e['kind']=='sync' and e['id'] in plan['output_dependencies']}
        assert selected
        plan['events']=[e for e in plan['events'] if e['id'] not in selected]
        plan['output_dependencies']=[e for e in plan['output_dependencies'] if e not in selected]
    extra_negative('only final visibility sync removed','shared-memory-sync',remove_output_sync)
    result=dict(passed=True,mode='native_regression',identity_policy='native checker authenticates IDs; independently re-signed mutations exercise semantic checks',positive_assertions=len(fixture_checks['checks']),negative_assertions=len(negative_records),cases=records,negative_checks=negative_records,body_checks=body_checks,native_resigning_proved=resign_probed,hardware_executed=False)
    save(out/'summary.json',result);print(json.dumps(dict(passed=True,cases=len(records),negative=len(negative_records),mode=result['mode'])))


if __name__=='__main__':main()
