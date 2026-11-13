#!/usr/bin/env python3
"""Audit complete source-ready Qwen requests; no fitting or device execution.

Local callback walls can overlap. Recompute their interval union instead of
summing them. Numerical checks retain the original mutual-top5 acceptance and
report full byte equality separately, never compare errors after divergence.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def union(events):
    intervals=sorted((v['start_ns'],v['end_ns']) for v in events)
    total=0;end=None
    for lo,hi in intervals:
        require(hi>=lo,'Negative callback interval')
        total+=hi-max(lo,end if end is not None else lo) if end is None or hi>end else 0
        end=max(end if end is not None else hi,hi)
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',action='append',default=[],help='NAME=actual capture.npz')
    parser.add_argument('--allow-diagnostic',action='store_true')
    parser.add_argument('--allow-code-drift',action='store_true',help='Report changed live source paths without claiming current-file identity')
    a=parser.parse_args();directory=a.execution;contract=load(directory/'contract.json');complete=load(directory/'completion.json')
    require(complete['completed'],'Incomplete Qwen generation')
    if not a.allow_diagnostic:require(complete['measured_requests']==3 and complete['warmups']==1 and contract['outputs']==128,'Formal full128 warm1/repeat3 required')
    rows=[json.loads(v) for v in (directory/'measurements.jsonl').read_text().splitlines()]
    measured=[row for row in rows if not row['warmup']]
    require(len(measured)==complete['measured_requests'] and {r['sample'] for r in measured}==set(range(len(measured))),'Repeated/missing measured sample')
    proof=load(directory/'runtime-proof.json')
    identity=hashlib.sha256(json.dumps({k:v for k,v in proof.items() if k!='async_runtime_proof_identity'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(identity==proof['async_runtime_proof_identity'],'Runtime proof identity changed')
    require(sha(proof['ready_spec']['file'])==proof['ready_spec']['sha256'],'Actual ready spec changed')
    if proof.get('whole_source'):
        require(sha(proof['whole_source']['file'])==proof['whole_source']['sha256'] and proof['source_ordering_identity']==proof['whole_source']['source_id'],'Whole source binding changed')
    drift=[v['file'] for v in proof['code_files'] if sha(v['file'])!=v['sha256']]
    require(a.allow_code_drift or not drift,'Owned runtime code changed since execution')
    report=dict(schema='orbit.qwen-r19-independent-ready-audit.v1',auditor_sha256=sha(__file__),execution=str(directory),
        validation_only=True,price_fitting=False,code_files_current_match=not drift,code_drift=drift,
        runtime_proof_identity=proof['async_runtime_proof_identity'],requests=[],numerical_pairs=[])
    expected={v['id'] for v in contract['source']['operators']}
    implementation_keys={}
    process_owners=set()
    for row in measured:
        profile=row['prefill_source_profile'];records=profile['implementation_records'];trace=profile['execution_trace'];events=trace['events']
        names=[n for r in records for n in r['source_operators']]
        require(len(names)==len(set(names))==len(expected) and set(names)==expected,'Source implementations do not form an exact cover')
        require(profile['source_id']==contract['source_id'] and profile['plan_id']==contract['plan_id'],'Source/plan identity mismatch')
        require(profile['source_ordering_identity']==proof['source_ordering_identity'] and profile.get('async_runtime_proof_identity')==proof['async_runtime_proof_identity'],'Profile async source proof mismatch')
        process_owners.update(e['owner_thread'] for e in events if e['device']=='npu')
        for record in records:
            binding=(record['local_price_key'],record['source_operators'],record['runtime_parameters'])
            require(record['implementation_id'] not in implementation_keys or implementation_keys[record['implementation_id']]==binding,'Implementation price binding changed within process')
            implementation_keys[record['implementation_id']]=binding
        actual=union(events);require(actual==profile['callback_interval_union_ns'] and actual+profile['host_control_ns']==profile['wall_ns'],'Concurrent event union does not conserve wall')
        require(len({r['local_price_key'] for r in records})==len(records),'Per-cover local price keys collide')
        require(len({e['owner_thread'] for e in events if e['device']=='npu'})<=1,'Multiple simultaneous prefill NPU owners')
        halves=[];cpu_count=0
        for record in records:
            if record['implementation_kind']=='cpu-region':
                cpu_count+=1
                for event in record['events']:
                    for phase in event['phases']:
                        require(phase['wall_ns']==phase['validation_and_staging_ns']+phase['kernel_inclusive_ns']+phase['output_copy_and_validation_ns'],'CPU region phases overlap')
                        require(phase['source_operators']==record['source_operators'] and phase['npu_launches']==0,'CPU region source/device mismatch')
            if record['implementation_kind']=='heterogeneous-gate-up':
                cpu=next(e for e in record['events'] if e['device']=='cpu' and not e['operator'].endswith('.assemble'))
                device=next(e for e in record['events'] if e['device']=='npu');phase=device['phases'][0]
                require(sum(v['wall_ns'] for v in phase['phases'])+phase['exclusive_control_ns']==phase['wall_ns'],'Private XRT phase overlap')
                require(phase['bo_lifetime']['released_ns']>=phase['host_visible_ns']>=phase['completed_ns']>=phase['launched_ns'],'BO lifetime ended before host-visible completion')
                actual_overlap=max(0,min(cpu['end_ns'],phase['completed_ns'])-max(cpu['start_ns'],phase['launched_ns']))
                artifact=load(Path(record['runtime_parameters']['directory'])/'implementation.json');replicas=artifact['resource']['task_instance_replicas']
                require(len([p for p in phase['phases'] if p['kind']=='sync-from-device'])==replicas,'Replica output BOs missing')
                halves.append(dict(source_operators=record['source_operators'],replicas=replicas,actual_cpu_device_overlap_ns=actual_overlap))
        request=row['request_phases'];checked=profile['wall_ns']+sum(request[k] for k in ('clear_context_ns','capture_copy_ns','first_argmax_ns','handoff_ns','request_control_ns'))
        overlap=[]
        require(len(row['tokens'])==row['outputs']==contract['outputs'] and len(row['decode_trace'])==row['outputs']-1,'Incomplete autoregressive inference')
        for i,detail in enumerate(row['decode_trace']):
            phase=detail['phases']
            process_owners.update(e['owner_thread'] for e in phase.get('ready_execution',{}).get('events',[]) if e['device']=='npu')
            region=sum(e['instruction_patch_ns']+e['submit_ns']+e['wait_ns'] for e in phase['region_events'])
            require(detail['step']==i and detail['position']==row['P']+i,'Autoregressive step mismatch')
            require(region+phase['host_before_ns']+phase['host_after_ns']+phase['host_control_ns']==phase['wall_ns'],'Decode phases overlap')
            require(phase['wall_ns']+detail['outside_dispatch_wrapper_ns']==detail['outer_dispatch_ns'],'Outer decode wall mismatch')
            checked+=detail['outer_dispatch_ns']+detail['argmax_ns']
            if 'pure_host_rope_device_overlap_ns' in phase:
                require(phase['prefetch_bo_writes']==0,'Speculative RoPE wrote an active BO')
                cpu=next(e for e in phase['ready_execution']['events'] if e['device']=='cpu')['phases'][0]
                require(cpu['position']==detail['position']+1 and cpu['bo_writes']==0,'Prefetch consumed wrong source position')
                actual_overlap=max(0,min(cpu['end_ns'],phase['device_complete_ns'])-max(cpu['start_ns'],phase['launch_boundary_ns'][1]))
                require(actual_overlap==phase['pure_host_rope_device_overlap_ns'],'Claimed decode overlap differs from timestamps')
                overlap.append(actual_overlap)
        require(checked==request['wall_ns'] and abs(checked-row['total_ms']*1e6)<=1,'Full request timing does not conserve')
        require(row['memory']['VmRSS_bytes']<=20*2**30 and row['memory']['VmSwap_bytes']<=.5*2**30,'Original memory guard exceeded')
        report['requests'].append(dict(sample=row['sample'],source_operators=len(names),source_exactcover=True,phase_union_conserves=True,cpu_regions=cpu_count,
            half_invocations=halves,decode_prefetch_calls=len(overlap),pure_host_rope_device_overlap_ns=sum(overlap),memory=row['memory']))
    require(len(process_owners)<=1,'Complete prefill/decode changed persistent NPU owner')
    report['persistent_npu_owner_verified']=True
    preparations=load(directory/'weight-preparation.json');compile_proofs=[proof for row in preparations if row.get('action')=='r19-actual-input-cpu-region-compilation' for proof in row['compile_proofs'].values()]
    for item in compile_proofs:
        require(item['producer']['sha256']==next(v['sha256'] for v in proof['code_files'] if v['file'].endswith('qwen-r19-regions.py')),'Compiled region producer drift')
        for record in item['generated_code']:require(sha(record['file'])==record['sha256'],'Actual generated CPU code changed')
    compiled_by_cover={tuple(item['region']['operators']):item for item in compile_proofs}
    for row in measured:
        for record in row['prefill_source_profile']['implementation_records']:
            if record['implementation_kind']=='cpu-region' and record['runtime_parameters']['backend']=='inductor-cpu':
                compiled=compiled_by_cover.get(tuple(record['source_operators']))
                require(compiled is not None,'Timed CPU native region has no actual-input compilation proof')
                for event in record['events']:
                    for phase in event['phases']:
                        require(phase['generated_kernel_audit']==compiled['generated_kernel_audit'],'Timed CPU native implementation differs from generated-code proof')
    report['cpu_compile_proofs']=dict(count=len(compile_proofs),native_multi_operator_fusion_proven=sum(v['generated_kernel_audit']['native_fusion_established'] for v in compile_proofs),actual_generated_code_verified=True)
    import numpy as np
    with np.load(directory/'capture.npz',allow_pickle=False) as z:captured={k:z[k].copy() for k in ('logits','tokens','prompt')}
    require(captured['logits'].shape==(contract['outputs'],151936) and np.isfinite(captured['logits']).all(),'Incomplete or nonfinite saved logits')
    require(np.array_equal(captured['tokens'],captured['logits'].argmax(axis=1)),'Saved token/logit mismatch')
    require(captured['tokens'].tolist()==measured[0]['tokens'],'Capture is not measured sample0')
    report['all_measured_token_streams_equal']=all(row['tokens']==measured[0]['tokens'] for row in measured)
    path=Path('/home/shiran/project/aie-toy/build/qwen25-3b/Triton-XDNA/third_party/mlir-air-src/programming_examples/llms/verify/comparators.py')
    sp=importlib.util.spec_from_file_location('original_r19_comparator',path);vendor=importlib.util.module_from_spec(sp);sys.modules[sp.name]=vendor;sp.loader.exec_module(vendor)
    for reference in a.reference:
        name,filename=reference.split('=',1)
        with np.load(filename,allow_pickle=False) as z:ref={k:z[k].copy() for k in ('logits','tokens','prompt')}
        require(np.array_equal(captured['prompt'],ref['prompt']),'Reference prompt mismatch')
        count=min(len(captured['tokens']),len(ref['tokens']));n=min(32,count)
        gate=vendor.compute_topk_set_check(captured['tokens'][:n].tolist(),np.argsort(-captured['logits'][:n],axis=1,kind='stable')[:,:5].tolist(),ref['tokens'][:n].tolist(),np.argsort(-ref['logits'][:n],axis=1,kind='stable')[:,:5].tolist(),k=5).to_dict()
        div=np.flatnonzero(captured['tokens'][:count]!=ref['tokens'][:count]);first=int(div[0]) if len(div) else None;same=count if first is None else first+1
        maximum=max(float(np.abs(captured['logits'][i].astype(np.float64)-ref['logits'][i].astype(np.float64)).max()) for i in range(same))
        report['numerical_pairs'].append(dict(reference=name,file=filename,sha256=sha(filename),original_mutual_top5=gate,first_token_divergence_zero_based=first,
            same_context_logit_rows=same,same_context_max_abs_error=maximum,full_logits_bitexact=bool(np.array_equal(captured['logits'].view(np.uint32),ref['logits'].view(np.uint32))) if captured['logits'].shape==ref['logits'].shape else None,post_divergence_error_metrics_computed=False))
    report['all_original_gates_pass']=all(p['original_mutual_top5']['status']=='OK' for p in report['numerical_pairs'])
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(a.output),gate=report['all_original_gates_pass'],cpu_compile_proofs=report['cpu_compile_proofs'])))
    require(report['all_original_gates_pass'],'Original numerical acceptance gate failed')


if __name__=='__main__':main()
