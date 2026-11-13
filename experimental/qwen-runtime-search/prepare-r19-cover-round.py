#!/usr/bin/env python3
"""Prepare real whole-Qwen experiments for an incumbent CPU-cover frontier.

No board process starts here. Every pending choice requires an actual full128
warm1/repeat3 run and unchanged numerical/memory gates. Failed experiments stay
explicit rejected/unsupported evidence; they cannot silently close a frontier.
The combined admission is deliberately unusable by the strict fitter until all
its new cohorts have actual accepted receipts and prices.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def replace_flag(command,flag,value):
    result=list(command)
    if result.count(flag)>1:raise ValueError('Repeated command flag: '+flag)
    if flag in result:result[result.index(flag)+1]=str(value)
    else:result.extend((flag,str(value)))
    return result


def ready_command(template,runtime,ready,plan,output,decode_directory):
    """Keep original ABI/guards while choosing the real sealed prep2 entrypoint."""
    command=list(template)
    if '--runtime-runner' in command:
        index=command.index('--runtime-runner');del command[index:index+2]
    command=[command[0],runtime['preparation_entrypoint']['file'],'--runtime-runner',
             runtime['runnerfile']['file'],*command[2:]]
    for flag,value in [('--ready-spec',ready),('--plan',plan),('--output',output),
        ('--decode-variant-dir',decode_directory),('--source',runtime['prefill_source_manifest'].get('requested_file',runtime['prefill_source_manifest']['file'])),
        ('--frozen-runtime',runtime['frozen_r18_runtime']['directory']),('--outputs',128),('--warmups',1),('--repeats',3)]:
        command=replace_flag(command,flag,value)
    for flag,value in [('--rss-limit-gib',20),('--swap-limit-gib',.5)]:
        if flag in command and float(command[command.index(flag)+1])!=value:raise ValueError('Original guard changed')
    return command


def merge_cpu_registration(parent,neighbors,digest):
    result=copy.deepcopy(parent);keys=set();rows=[]
    for row in [*parent['implementations'],*neighbors['implementations']]:
        p=row['runtime_config'];key=digest(dict(operators=sorted(row['operators']),
            **{name:p[name] for name in ('backend','input_policy','output_policy')}))
        if key not in keys:keys.add(key);rows.append(row)
    result['implementations']=rows
    result['admission']='pending choices become executable only after unchanged full-Qwen gate and fresh callback prices'
    return result


def retain_receipt_backings(parent,receipts,roots,fit):
    """Retain only strictly audited successes with their actual manifest/summary.

    This does not promote failed or merely planned cohorts. The caller must first
    run audit_execution_receipts; every training cohort needs that exact receipt.
    """
    training=[row for row in parent['cohorts'] if row['purpose']=='training']
    fit.require(set(receipts)=={row['name'] for row in training},
                'Every retained parent cohort needs an actual successful receipt')
    backings=copy.deepcopy(parent.get('retained_receipt_backing',[]));backed={r['cohort']['name'] for r in backings}
    for cohort in training:
        if cohort['name'] in backed:continue
        actual=receipts[cohort['name']]['actual_receipt'];found=[]
        for root in roots:
            manifest_path=root/'manifest.json'
            if actual['manifest_sha256']==fit.record(manifest_path)['sha256']:found.append(root)
        fit.require(len(found)==1,'Ambiguous actual parent execution receipt backing')
        root=found[0];summary=fit.load(root/'summary.json')
        fit.require(actual in summary,'Actual parent receipt is absent from final summary')
        backings.append(dict(cohort=cohort,execution_receipt=actual,
            execution_manifest=fit.record(root/'manifest.json'),execution_receipts=fit.record(root/'summary.json')))
    return backings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('catalogue','neighbors','incumbent-ready','incumbent-plan','decode-variant-dir',
                 'parent-admission','cpu-registration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-campaign-receipts',type=Path,action='append',default=[])
    p.add_argument('--round-name',required=True)
    p.add_argument('--template-cohort',default='parent-npu-original')
    a=p.parse_args()
    fit=module(Path(__file__).with_name('fit-r19-catalogue-v2.py'),'r19_frontier_actual_receipt_api')
    helper=module(Path(__file__).with_name('expand-r19-cover-neighbors.py'),'r19_frontier_cover_api')
    require=fit.require
    require(re.fullmatch('[A-Za-z0-9_-]+',a.round_name),'Unsafe round name')
    catalogue=fit.load(a.catalogue);neighbors=fit.load(a.neighbors);parent=fit.load(a.parent_admission)
    require(catalogue['training_campaign']['sha256']==fit.record(a.parent_admission)['sha256'],
            'Incumbent catalogue does not bind the supplied actual parent training campaign')
    require(catalogue['execution_registration_identity']==fit.digest({k:v for k,v in catalogue.items() if k!='execution_registration_identity'}),
            'Catalogue registration identity differs')
    for key in ('source','catalogue','selection','backend','producer'):
        if key in neighbors:fit.verified(neighbors[key])
    require(neighbors.get('catalogue',{}).get('sha256')==fit.record(a.catalogue)['sha256'],
            'Neighbor frontier was generated from a different actual catalogue')
    require(neighbors['implementations'] and neighbors['cohorts'],'Frontier has no pending experiments')
    pending_ids={row['id'] for row in neighbors['implementations']}
    require(pending_ids<={row['id'] for c in neighbors['cohorts'] for row in c['cpu_regions']},
            'A pending frontier choice has no actual whole-program experiment')
    ready=fit.load(a.incumbent_ready);plan=fit.load(a.incumbent_plan);runtime=catalogue['ready_runtime']
    require(ready['source_id']==neighbors['source_id']==runtime['prefill_source_manifest']['source_id'],
            'Actual incumbent and frontier prefill sources differ')
    require(runtime['preparation_protocol']=='owned-prefill-decode-release-readonly-prefault-v2',
            'New experiments require the actual two-stage preparation protocol')
    for row in [runtime['preparation_entrypoint'],runtime['runnerfile'],*runtime['code_files']]:fit.verified(row)
    receipts,verification=fit.audit_execution_receipts(parent,a.parent_campaign_receipts)
    require(set(receipts)=={row['name'] for row in parent['cohorts'] if row['purpose']=='training'},
            'Every retained parent cohort needs an actual successful receipt')
    backings=retain_receipt_backings(parent,receipts,a.parent_campaign_receipts,fit)
    template=next(row for row in parent['cohorts'] if row['name']==a.template_cohort)
    a.output.mkdir(parents=True,exist_ok=False)
    def write(path,value):
        with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    combined_cpu=merge_cpu_registration(fit.load(a.cpu_registration),neighbors,fit.digest)
    cpu_path=a.output/'cpu-registration-combined.json';write(cpu_path,combined_cpu)
    new=[];new_dependencies=[]
    for index,cohort in enumerate(neighbors['cohorts']):
        name=f'{a.round_name}-{index:03d}';spec_path=a.output/(name+'.spec.json');plan_path=a.output/(name+'.plan.json')
        execution=a.output/('train-'+name)
        spec=helper.apply_cohort(ready,plan,cohort,execution/'cpu-preparation')
        write(spec_path,spec);write(plan_path,plan)
        command=ready_command(template['command'],runtime,spec_path,plan_path,execution,a.decode_variant_dir)
        admission=replace_flag(template['admission_command'],'--execution',execution)
        admission=replace_flag(admission,'--output',execution/'gate.json')
        new.append(dict(name=name,purpose='training',command=command,admission_command=admission,
            plan_id=fit.digest({k:v for k,v in plan.items() if k!='plan_id'}),measured_requests=3,
            numeric_gate_file=str(execution/'gate.json'),preparation_protocol=runtime['preparation_protocol'],
            neighbor_cohort_id=cohort['id']))
        new_dependencies.extend((fit.record(spec_path),fit.record(plan_path)))
    require(not {row['name'] for row in new}.intersection(row['name'] for row in parent['cohorts']),
            'New frontier cohort names collide with retained actual training')
    driver=fit.record(Path(__file__).with_name('run-independent-r19-campaign-v2.py'))
    dependencies={row['file']:row for row in parent['dependencies']}
    for row in [fit.record(__file__),fit.record(helper.__file__),fit.record(fit.__file__),fit.record(a.catalogue),
        fit.record(a.neighbors),fit.record(a.incumbent_ready),fit.record(a.incumbent_plan),fit.record(a.parent_admission),
        fit.record(a.cpu_registration),fit.record(cpu_path),driver,*new_dependencies,
        *[r for backing in backings for r in (backing['execution_manifest'],backing['execution_receipts'])]]:
        if row['file'] in dependencies:require(dependencies[row['file']]['sha256']==row['sha256'],'Parent dependency changed')
        dependencies[row['file']]=row
    execution=copy.deepcopy(parent);execution['cohorts']=new;execution['dependencies']=list(dependencies.values())
    execution.pop('retained_receipt_backing',None)
    execution['retained_r19_measurements']=False
    execution['parent_timings_reused']=False
    execution['old_r18_timings_reused']=False
    execution['independent_execution_policy']=dict(failure_policy='continue-hardware-or-gate-failure-stop-on-seal-drift',
        driver=driver,execution_order=[row['name'] for row in new],timed_requests_changed=False)
    execution['frontier_round']=dict(schema='orbit.qwen-r19-incumbent-frontier-training.v1',
        producer=fit.record(__file__),neighbors=fit.record(a.neighbors),parent_catalogue=fit.record(a.catalogue),
        pending_choices=len(neighbors['implementations']),cohorts=len(new),cpu_registration=fit.record(cpu_path),
        closure='not established; all pending choices require passed full-Qwen admission or an explicit source-bound unsupported/rejected witness')
    execution_path=a.output/'campaign-execution.json';write(execution_path,execution)
    combined=copy.deepcopy(execution);combined['cohorts']=[*parent['cohorts'],*new]
    combined['retained_receipt_backing']=backings;combined['retained_r19_measurements']=True
    combined['old_r18_timings_reused']=False;combined['parent_timings_reused']=False
    admission_path=a.output/'campaign-admission.json';write(admission_path,combined)
    result=dict(schema='orbit.qwen-r19-prepared-frontier-round.v1',execution=fit.record(execution_path),
        admission=fit.record(admission_path),cpu_registration=fit.record(cpu_path),producer=fit.record(__file__),
        driver_command=[template['command'][0],driver['file'],'--manifest',str(execution_path),
            '--manifest-sha256',fit.record(execution_path)['sha256'],'--output',str(a.output/'campaign-receipts')],
        next_fit=dict(producer=fit.record(fit.__file__),base_catalogue=catalogue['parent_static_catalogue'],
            campaign=fit.record(admission_path),cpu_registration=fit.record(cpu_path),
            campaign_receipts=str(a.output/'campaign-receipts')),
        unmeasured_neighbors=len(neighbors['implementations']),closure_established=False)
    write(a.output/'prepared-round.json',result);print(json.dumps(result))


if __name__=='__main__':main()
