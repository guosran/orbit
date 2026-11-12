#!/usr/bin/env python3
"""Build local implementation covers from exclusive complete-Qwen training traces.

Never consumes validation captures or whole-plan price catalogues. Missing local
prices remain null. Invocation residuals stay explicit fixed costs, never spread
across source operators. The common C++ checker owns cover legality and solving.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def load(path):return json.loads(Path(path).read_text())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def identity(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def median_ms(values):return statistics.median(values)/1e6 if values else None


def build(source,prefill,covers,training):
    require(source['source_id']==identity({k:v for k,v in source.items() if k!='source_id'}),'whole source identity mismatch')
    require(prefill['source_id']==identity({k:v for k,v in prefill.items() if k!='source_id'}),'prefill identity mismatch')
    require(source['operators'][:len(prefill['operators'])]==prefill['operators'] and source['values'][:len(prefill['values'])]==prefill['values'] and source['bindings'][:len(prefill['bindings'])]==prefill['bindings'],'whole source does not preserve exact prefill prefix')
    require(covers['source_id']==source['source_id'],'wave cover/source mismatch')
    operators=source['operators'];byid={o['id']:i for i,o in enumerate(operators)};values={v['id']:i for i,v in enumerate(source['values'])}
    producer={v:o['id'] for o in operators for v in o['outputs']};readers=defaultdict(set)
    for o in operators:
        for v in o['inputs']:readers[v].add(o['id'])
    prices=defaultdict(list);launches=defaultdict(set);hostprices=defaultdict(list);waveprices=defaultdict(list);fixed=[];training_files=[];contracts=[];conservation=[]
    for directory in map(Path,training['directories']):
        contract=load(directory/'contract.json');require(contract['source']==prefill,'training source prefix mismatch');require(contract['outputs']==128 and contract['threads']==4,'training request contract mismatch')
        rows=[json.loads(line) for line in (directory/'measurements.jsonl').read_text().splitlines()];rows=[r for r in rows if not r['warmup']]
        require(len(rows)>=3 and len({r['sample'] for r in rows})==len(rows),'three unique measured training requests required')
        contracts.append(contract);training_files.append(dict(directory=str(directory),contract_sha256=sha(directory/'contract.json'),measurements_sha256=sha(directory/'measurements.jsonl'),samples=[r['sample'] for r in rows]))
        for row in rows:
            profile=row['prefill_source_profile'];request=row['request_phases'];detail=row['decode_trace']
            require(profile['source_id']==prefill['source_id'] and profile['plan_id']==contract['plan_id'],'training profile binding mismatch')
            require(profile['exclusive_sum_equals_wall'] and sum(profile['node_cost_ns'].values())+profile['host_control_ns']==profile['wall_ns'],'prefill phase overlap')
            require(request['exclusive_sum_equals_wall'] and row['outputs']==128 and len(detail)==127,'incomplete conserved complete-model training')
            require(set(profile['node_cost_ns'])=={o['id'] for o in prefill['operators']},'incomplete prefill prices')
            fixed_ns=profile['host_control_ns']+request['clear_context_ns']+request['capture_copy_ns']+request['request_control_ns']
            hostprices['first-token-argmax',None].append(request['first_argmax_ns']);hostprices['kv-handoff',None].append(request['handoff_ns'])
            checked=profile['wall_ns']+request['clear_context_ns']+request['capture_copy_ns']+request['request_control_ns']+request['first_argmax_ns']+request['handoff_ns']
            for name,ns in profile['node_cost_ns'].items():
                require(isinstance(ns,int) and ns>=0,'invalid source-node duration')
                device=contract['plan']['placements'][name];prices[name,device].append(ns);launches[name,device].add(profile['node_launch_counts'][name])
            for step,d in enumerate(detail):
                phase=d['phases'];require(d['step']==step and phase['position']==row['P']+step and phase['exclusive_sum_equals_wall'],'decode step/phase mismatch')
                regions=phase['region_events'];partition=[0]+[r['region']['hi'] for r in regions]
                require(partition in covers['legal_runtime_partitions'],'unregistered runtime partition')
                require([r['region']['lo'] for r in regions]==partition[:-1],'decode partition gaps')
                region_ns=0
                for region in regions:
                    ns=region['instruction_patch_ns']+region['submit_ns']+region['wait_ns'];require(region['service_ns']==region['submit_ns']+region['wait_ns'],'service double accounting')
                    region_ns+=ns;waveprices[step,region['region']['lo'],region['region']['hi']].append(ns)
                require(phase['host_before_ns']+phase['host_after_ns']+phase['host_control_ns']+region_ns==phase['wall_ns'],'decode helper phase overlap')
                require(d['outer_dispatch_ns']==phase['wall_ns']+d['outside_dispatch_wrapper_ns'],'outer decoder phase overlap')
                hostprices['decode-host-before',step].append(phase['host_before_ns']);hostprices['decode-host-after',step,tuple(partition)].append(phase['host_after_ns']+phase['host_control_ns']+d['outside_dispatch_wrapper_ns']);hostprices['decode-argmax',step].append(d['argmax_ns'])
                checked+=d['outer_dispatch_ns']+d['argmax_ns']
            require(checked==request['wall_ns'],'whole request phase conservation failure');require(fixed_ns>=0,'negative fixed invocation cost');fixed.append(fixed_ns)
            conservation.append(dict(directory=str(directory),sample=row['sample'],wall_ns=request['wall_ns'],local_source_and_cover_ns=request['wall_ns']-fixed_ns,fixed_application_ns=fixed_ns,exact_sum_equals_wall=True))
    require(contracts and len({identity(c['producer']) for c in contracts})==1,'training used different producer revisions')
    require(len({identity(c['implementation_files']) for c in contracts})==1,'training implementation files differ')
    code_files=list(contracts[0]['implementation_files'])+training.get('runtime_files',[])
    root=Path(__file__).resolve().parents[2]
    code_files+= [dict(file=str(root/'scripts'/name),sha256=sha(root/'scripts'/name)) for name in ['qwen-node-runtime.py','run-qwen-node-runtime.py']]
    for record in code_files:require(sha(record['file'])==record['sha256'],'actual implementation source changed')
    implementations=[];runtime={};covered=set();choice_groups=[]
    def add(impl_id,names,device,cost,registration,launch_count=0,visibility='host-materialized'):
        require(names and len(set(names))==len(names),'empty/duplicate cover')
        indices=[byid[name] for name in names];require(indices==list(range(indices[0],indices[-1]+1)),'source cover is not contiguous')
        member=set(names);ins=set();outs=set()
        for name in names:
            o=operators[byid[name]]
            ins.update(v for v in o['inputs'] if producer.get(v) not in member)
            outs.update(v for v in o['outputs'] if source['values'][values[v]]['role']=='output' or readers[v]-member)
        registration=dict(registration,source_id=source['source_id'],operators=names,device=device)
        registration['artifact_identity']=identity(registration)
        implementations.append(dict(id=impl_id,source_id=source['source_id'],artifact_identity=registration['artifact_identity'],entrypoint=registration['kind'],device=device,resource='cpu.host' if device=='cpu' else 'npu.device',memory='runtime-abi',source_operators=indices,inputs=sorted(values[v] for v in ins),outputs=sorted(values[v] for v in outs),wall_ms=cost,executable=True,launch_count=launch_count,cost_boundary='inclusive-region-wall',transport_ownership='implementation-inclusive',visibility_policy=visibility))
        runtime[impl_id]=registration;covered.update(names)
    for o in prefill['operators']:
        choices=[]
        devices=o['attributes'].get('supported_devices',[o['attributes'].get('fixed_device','cpu')])
        for device in devices:
            counts=launches[o['id'],device]
            require(len(counts)<=1,'implementation invocation count changed between training requests')
            count=next(iter(counts)) if counts else (1 if device=='npu' else 0)
            impl_id=o['id']+'.'+device
            add(impl_id,[o['id']],device,median_ms(prices[o['id'],device]),dict(kind='prefill-node',operator=o['id'],code_files=code_files),count)
            choices.append(dict(id=device,implementations=[impl_id]))
        choice_groups.append(dict(id=o['id'],choices=choices))
    after_groups={}
    partitions=[('whole',[0,61]),('split2',[0,36,61]),('split3',[0,18,36,61])]
    for group in covers['host_groups']:
        key=group['kind'],group.get('step')
        if group['kind']=='decode-host-after':
            after_groups[group['step']]=group
            for label,partition in partitions:
                impl_id=group['id']+'.cpu.'+label
                add(impl_id,group['operators'],'cpu',median_ms(hostprices[key+(tuple(partition),)]),dict(kind='decode-host-group',host_kind=group['kind'],step=group['step'],partition=partition,code_files=code_files))
        else:
            impl_id=group['id']+'.cpu';add(impl_id,group['operators'],'cpu',median_ms(hostprices[key]),dict(kind='decode-host-group',host_kind=group['kind'],step=group.get('step'),code_files=code_files))
            choice_groups.append(dict(id=group['id'],choices=[dict(id='cpu',implementations=[impl_id])]))
    waves={(x['step'],x['wave']):x['operators'] for x in covers['wave_operators']};steps=sorted({t for t,w in waves})
    for step in steps:
        for lo,hi in [(0,61),(0,36),(36,61),(0,18),(18,36)]:
            directory=Path(covers['proof_root'])/f'region-{lo}-{hi}';receipt=directory/'region.json';proof=load(receipt)
            require(proof['region']==dict(lo=lo,hi=hi),'physical wave region mismatch')
            names=[name for wave in range(lo,hi) for name in waves[step,wave]]
            add(f'decode.T{step}.waves.{lo}-{hi}',names,'npu',median_ms(waveprices[step,lo,hi]),dict(kind='decode-wave-region',step=step,lo=lo,hi=hi,template_directory=str(directory.resolve()),artifact_files=[dict(file=str(receipt.resolve()),sha256=sha(receipt))]),1,'device-resident')
        choices=[]
        for label,partition in partitions:
            members=[f'decode.T{step}.waves.{lo}-{hi}' for lo,hi in zip(partition,partition[1:])]
            members.append(after_groups[step]['id']+'.cpu.'+label)
            choices.append(dict(id=label,implementations=members))
        choice_groups.append(dict(id=f'decode.T{step}.partition',choices=choices))
    require(covered==set(byid),'source operator is absent from executable catalogue')
    return dict(schema='orbit.program-operator-implementations.v1',source_id=source['source_id'],implementations=implementations,choice_groups=choice_groups,producer_sha256=sha(__file__),
                runtime_registration=dict(schema='orbit.qwen-whole-runtime-registration.v1',source_id=source['source_id'],prefill_source_id=prefill['source_id'],implementations=runtime,legal_runtime_partitions=covers['legal_runtime_partitions'],wave_operators=covers['wave_operators'],host_groups=covers['host_groups']),
                fixed_application_invocation=dict(boundary='mutually-exclusive prefill host control + request clear/capture/control only; partition-dependent decoder control belongs to matching host-after variant',wall_ms=median_ms(fixed),samples_ns=fixed,selection_independent_approximation=True,limitation='Prefill/application control is pooled as a constant approximation; candidate independence is unproved and full-model error requires heldout validation'),
                training=training_files,trace_conservation_receipts=conservation,missing_local_prices=sum(i['wall_ms'] is None for i in implementations),complete_plan_prepricing=False)


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ['source','prefill-source','covers','training','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();result=build(load(a.source),load(a.prefill_source),load(a.covers),load(a.training))
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),implementations=len(result['implementations']),missing_local_prices=result['missing_local_prices'])))
if __name__=='__main__':main()
