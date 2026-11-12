#!/usr/bin/env python3
"""Translate a checked common ProgramGraph selection to exact Qwen replay.

Indices are resolved through the graph's own source. Implementations must match
registered local covers and actual source/artifact hashes. No raw search choice
index is interpreted. The frozen original runner remains unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def load(path):return json.loads(Path(path).read_text())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def identity(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def translate(graph,catalogue,prefill,*,verify_files=True):
    require(graph.get('schema')=='orbit.program-graph.v1' and graph.get('source_kind')=='source-bound-external-operators','requires common operator ProgramGraph')
    source=graph['source'];sid=source['source_id']
    require(sid==identity({k:v for k,v in source.items() if k!='source_id'}),'source identity mismatch')
    require(sid==catalogue['source_id'],'catalogue/source mismatch')
    require(source['operators'][:len(prefill['operators'])]==prefill['operators'] and source['values'][:len(prefill['values'])]==prefill['values'] and source['bindings'][:len(prefill['bindings'])]==prefill['bindings'],'prefill prefix changed')
    require(prefill['source_id']==identity({k:v for k,v in prefill.items() if k!='source_id'}),'prefill identity mismatch')
    registration=catalogue['runtime_registration'];require(registration['source_id']==sid and registration['prefill_source_id']==prefill['source_id'],'runtime registration/source mismatch')
    original={i['id']:i for i in catalogue['implementations']};registered=registration['implementations']
    require(len(original)==len(catalogue['implementations']),'duplicate registered implementation')
    selection=graph['external_implementation_selection'];require(selection['schema']=='orbit.program-operator-selection.v1','wrong selection schema')
    selected_ids={i['id'] for i in selection['regions']}
    require(len(selected_ids)==len(selection['regions']),'duplicate selected implementation')
    grouped=set()
    for group in catalogue['choice_groups']:
        matching=[choice for choice in group['choices'] if set(choice['implementations'])<=selected_ids]
        require(len(matching)==1,'selection violates local choice coupling: '+group['id'])
        grouped.update(matching[0]['implementations'])
    require(grouped==selected_ids,'implementation is outside local registered choice groups')
    source_ops=source['operators'];seen=set();placements={};decode={};proofs={};host_seen=set();after_partitions={}
    waveops={(int(x['step']),int(x['wave'])):x['operators'] for x in registration['wave_operators']}
    hostgroups={(g['kind'],g.get('step')):g['operators'] for g in registration['host_groups']}
    if verify_files:
        for b in source['bindings']:
            require(sha(b['file'])==b['sha256'],'actual bound source changed: '+b['file'])
    fields=['id','source_id','artifact_identity','entrypoint','device','resource','memory','source_operators','inputs','outputs','wall_ms','executable','launch_count','cost_boundary','transport_ownership','visibility_policy']
    for chosen in selection['regions']:
        name=chosen['id'];require(name in original and name in registered,'unregistered implementation '+name)
        impl=original[name];reg=registered[name]
        require(all(chosen.get(k)==impl.get(k) for k in fields),'selected implementation differs from registered contract: '+name)
        require(reg['artifact_identity']==identity({k:v for k,v in reg.items() if k!='artifact_identity'})==chosen['artifact_identity'],'runtime artifact identity mismatch')
        indices=chosen['source_operators'];require(indices and all(isinstance(i,int) and not isinstance(i,bool) and 0<=i<len(source_ops) for i in indices),'invalid source operator indices')
        require(not seen.intersection(indices),'overlapping source cover');seen.update(indices)
        names=[source_ops[i]['id'] for i in indices];require(names==reg['operators'],'runtime/source cover differs')
        for record in reg.get('code_files',[])+reg.get('artifact_files',[]):
            key=record['file']
            require(key not in proofs or proofs[key]==record['sha256'],'conflicting implementation source proof')
            if verify_files and key not in proofs:require(sha(key)==record['sha256'],'actual registered implementation changed: '+key)
            proofs[key]=record['sha256']
        kind=reg['kind']
        if kind=='prefill-node':
            require(names==[reg['operator']] and reg['operator'].startswith('prefill.'),'invalid per-node prefill implementation')
            require(reg['operator'] not in placements,'repeated prefill implementation');placements[reg['operator']]=chosen['device']
        elif kind=='decode-host-group':
            key=reg['host_kind'],reg['step'];require(key in hostgroups and hostgroups[key]==names and chosen['device']=='cpu','invalid host source group');require(key not in host_seen,'duplicate host group');host_seen.add(key)
            if reg['host_kind']=='decode-host-after':after_partitions[reg['step']]=reg['partition']
        elif kind=='decode-wave-region':
            step=reg['step'];lo,hi=reg['lo'],reg['hi']
            require(isinstance(step,int) and not isinstance(step,bool) and chosen['device']=='npu' and 0<=lo<hi<=61,'invalid decode region')
            expected=[o for wave in range(lo,hi) for o in waveops[step,wave]]
            require(names==expected,'decode cover does not match actual wave source operators')
            directory=Path(reg['template_directory']);physical=load(directory/'region.json') if verify_files else dict(region=dict(lo=lo,hi=hi))
            require(physical['region']==dict(lo=lo,hi=hi),'compiled wave range differs')
            decode.setdefault(step,[]).append(dict(lo=lo,hi=hi,implementation=name,template_directory=str(directory)))
        else:raise ValueError('unknown executable runtime cover kind '+kind)
    require(seen==set(range(len(source_ops))),'source graph not completely covered')
    require(host_seen==set(hostgroups),'source host group omitted')
    require(set(placements)=={o['id'] for o in prefill['operators']},'prefill source node omitted')
    node=module(Path(__file__).resolve().parents[2]/'scripts/qwen-node-runtime.py','orbit_qwen_node_translate')
    plan=node.make_plan(prefill,placements)
    length=next(v['shape'][0] for v in prefill['values'] if v['id']=='input.tokens')
    require(sorted(decode)==list(range(127)),'complete 127-step decode selection required')
    steps=[]
    for step in range(127):
        regions=sorted(decode[step],key=lambda r:r['lo']);partition=[0]+[r['hi'] for r in regions]
        require([r['lo'] for r in regions]==partition[:-1] and partition in registration['legal_runtime_partitions'],'illegal selected decode partition')
        require(after_partitions.get(step)==partition,'decode control price/replay partition mismatch')
        steps.append(dict(step=step,position=length+step,partition=partition,regions=regions))
    replay=dict(schema='orbit.qwen-whole-selected-replay.v1',source_id=sid,program_graph_id=graph['id'],prefill_source_id=prefill['source_id'],prefill_plan=plan,decode_steps=steps,source_files=[dict(file=p,sha256=h) for p,h in sorted(proofs.items())],outputs=128,prompt_tokens=length)
    replay['replay_id']=identity(replay)
    return replay


def manual_baseline(source,catalogue,prefill,plan,label):
    require(label in ('all-supported-npu','manual-matmul-npu'),'manual baseline must have an explicit label')
    require(source['source_id']==catalogue['source_id']==identity({k:v for k,v in source.items() if k!='source_id'}),'manual source identity mismatch')
    require(source['operators'][:len(prefill['operators'])]==prefill['operators'] and source['values'][:len(prefill['values'])]==prefill['values'] and source['bindings'][:len(prefill['bindings'])]==prefill['bindings'],'manual prefill source prefix mismatch')
    node=module(Path(__file__).resolve().parents[2]/'scripts/qwen-node-runtime.py','orbit_manual_qwen_plan')
    choices=node.validate_plan(prefill,plan);require(plan['plan_id']==identity({k:v for k,v in plan.items() if k!='plan_id'}),'manual plan identity mismatch')
    for op in prefill['operators']:
        variable=op['attributes'].get('supported_devices')==['cpu','npu']
        expected='npu' if variable and (label=='all-supported-npu' or op['kind']=='matmul') else 'cpu'
        require(choices[op['id']]==expected,'manual baseline label/actual operator choice mismatch')
    registration=catalogue['runtime_registration'];impls={i['id']:i for i in catalogue['implementations']};ids=[]
    for group in catalogue['choice_groups']:
        if group['id'] in choices:
            selected=next(c for c in group['choices'] if c['id']==choices[group['id']])
        elif len(group['choices'])==1:selected=group['choices'][0]
        else:selected=next(c for c in group['choices'] if c['id']=='whole')
        ids.extend(selected['implementations'])
    require(len(set(ids))==len(ids),'manual executable covers overlap')
    proofs={};coverage=set();steps={}
    for name in ids:
        impl=impls[name];reg=registration['implementations'][name]
        require(reg['artifact_identity']==identity({k:v for k,v in reg.items() if k!='artifact_identity'})==impl['artifact_identity'],'manual implementation identity mismatch')
        require(not coverage.intersection(impl['source_operators']),'manual source coverage overlap');coverage.update(impl['source_operators'])
        for record in reg.get('code_files',[])+reg.get('artifact_files',[]):
            if record['file'] not in proofs:require(sha(record['file'])==record['sha256'],'manual actual implementation changed')
            proofs[record['file']]=record['sha256']
        if reg['kind']=='decode-wave-region':
            require(reg['lo']==0 and reg['hi']==61,'manual baseline requires whole decode')
            steps[reg['step']]=dict(step=reg['step'],partition=[0,61],regions=[dict(lo=0,hi=61,implementation=name,template_directory=reg['template_directory'])])
    require(coverage==set(range(len(source['operators']))) and sorted(steps)==list(range(127)),'manual baseline omits real source operators')
    for binding in source['bindings']:require(sha(binding['file'])==binding['sha256'],'manual actual bound source changed')
    length=next(v['shape'][0] for v in prefill['values'] if v['id']=='input.tokens')
    for step,row in steps.items():row['position']=length+step
    replay=dict(schema='orbit.qwen-whole-selected-replay.v1',mode='manual-baseline',baseline_label=label,source_id=source['source_id'],program_graph_id=None,
                graph_materialization_claim=False,prefill_source_id=prefill['source_id'],prefill_plan=plan,decode_steps=[steps[i] for i in range(127)],source_files=[dict(file=p,sha256=h) for p,h in sorted(proofs.items())],outputs=128,prompt_tokens=length)
    replay['replay_id']=identity(replay);return replay


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ['catalogue','prefill-source','output']:p.add_argument('--'+key,type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--graph',type=Path);mode.add_argument('--baseline-plan',type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--baseline-label',choices=['all-supported-npu','manual-matmul-npu'])
    p.add_argument('--frozen-runtime',type=Path,required=True);p.add_argument('--aie-root',type=Path,required=True);p.add_argument('--region-helper',type=Path,required=True);p.add_argument('--region-root',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--warmups',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();prefill=load(a.prefill_source);catalogue=load(a.catalogue)
    if a.graph:replay=translate(load(a.graph),catalogue,prefill)
    else:
        require(a.source is not None and a.baseline_label is not None,'manual baseline needs source and explicit baseline label')
        replay=manual_baseline(load(a.source),catalogue,prefill,load(a.baseline_plan),a.baseline_label)
    a.output.mkdir(parents=True,exist_ok=False)
    for name,value in [('selected-replay.json',replay),('prefill-plan.json',replay['prefill_plan'])]:
        (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    helper=Path(__file__).with_name('selected-decode-helper.py').resolve()
    command=[sys.executable,str(a.frozen_runtime/'run-qwen-node-runtime.py'),'--aie-root',str(a.aie_root),'--source',str(a.prefill_source),'--plan',str(a.output/'prefill-plan.json'),'--output',str(a.output/'execution'),'--outputs','128','--warmups',str(a.warmups),'--repeats',str(a.repeats),'--region-root',str(a.region_root),'--decode-regions','0:61','--region-helper',str(helper)]
    environment=dict(ORBIT_QWEN_SELECTED_REPLAY=str((a.output/'selected-replay.json').resolve()),ORBIT_QWEN_BASE_REGION_HELPER=str(a.region_helper.resolve()))
    receipt=dict(schema='orbit.qwen-whole-selected-command.v1',replay_id=replay['replay_id'],program_graph_id=replay['program_graph_id'],graph_sha256=sha(a.graph) if a.graph else None,manual_baseline=a.baseline_label if not a.graph else None,catalogue_sha256=sha(a.catalogue),translator_sha256=sha(__file__),selector_sha256=sha(helper),frozen_runner_sha256=sha(a.frozen_runtime/'run-qwen-node-runtime.py'),frozen_node_runtime_sha256=sha(a.frozen_runtime/'qwen-node-runtime.py'),base_helper_sha256=sha(a.region_helper),command=command,environment=environment,executed=a.execute)
    (a.output/'command-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    if a.execute:
        result=subprocess.run(command,env=dict(os.environ,**environment),check=False)
        (a.output/'process-result.json').write_text(json.dumps(dict(returncode=result.returncode))+'\n');raise SystemExit(result.returncode)

if __name__=='__main__':main()
