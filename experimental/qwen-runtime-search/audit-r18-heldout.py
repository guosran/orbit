#!/usr/bin/env python3
"""Independent heldout timing audit; opt-in NPZ reads after all board work.

Uses the original vendor first32-token mutual-top5 gate without changes.
Full128-token equality and same-context logit equality are separate results.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load(path):return json.loads(Path(path).read_text())


def require(value,message):
    if not value:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--captures',action='store_true')
    a=p.parse_args();spec=load(a.spec)
    report=dict(schema='orbit.qwen-r18-independent-heldout-audit.v1',spec_sha256=sha(a.spec),
        auditor_sha256=sha(__file__),captures_audited=a.captures,cohorts=[],pairs=[],comparisons=[],
        limitations=['Only first measured request has saved logits; all measured token streams checked separately.',
            'Original mutual-top5 gate stops at first divergence and is not full-logit equivalence.',
            'Post-divergence logits have different contexts and are never used for numerical error metrics.',
            'Warmup and preparation are excluded from steady-request latency; every compared cohort uses the same routed runner.'])
    captures={};timings={};allrows={};routed_hashes=set()
    for cohort in spec['cohorts']:
        name=cohort['name'];directory=Path(cohort['directory']);contract=load(directory/'contract.json')
        complete=load(directory/'completion.json')
        require(complete['completed'] and complete['measured_requests']==3 and complete['warmups']==1,'incomplete heldout cohort')
        require(contract['outputs']==128 and contract['threads']==4,'different request/CPU thread contract')
        rows=[json.loads(line) for line in (directory/'measurements.jsonl').read_text().splitlines()]
        rows=[row for row in rows if not row['warmup']]
        require(len(rows)==3 and {row['sample'] for row in rows}=={0,1,2},'three unique heldout samples required')
        for row in rows:
            profile=row['prefill_source_profile'];request=row['request_phases'];decode=row['decode_trace']
            require(row['outputs']==128 and len(row['tokens'])==128 and len(decode)==127,'incomplete Qwen inference')
            require(profile['source_id']==contract['source_id'] and profile['plan_id']==contract['plan_id'],'source/plan mismatch')
            names=list(profile['node_cost_ns'])+[op for cover in profile['fused_covers'] for op in cover['operators']]
            require(len(names)==len(set(names))==581 and set(names)=={op['id'] for op in contract['source']['operators']},'source costs are not an exact581-node cover')
            require(profile['exclusive_sum_equals_wall'] and sum(profile['node_cost_ns'].values())+
                sum(cover['exclusive_wall_ns'] for cover in profile['fused_covers'])+profile['host_control_ns']==profile['wall_ns'],'prefill timing overlap')
            checked=profile['wall_ns']+sum(request[k] for k in ('clear_context_ns','capture_copy_ns','first_argmax_ns','handoff_ns','request_control_ns'))
            for i,detail in enumerate(decode):
                phase=detail['phases'];events=phase['region_events']
                require(detail['step']==i and phase['position']==row['P']+i and phase['exclusive_sum_equals_wall'],'decode step/phase mismatch')
                region=sum(event['instruction_patch_ns']+event['submit_ns']+event['wait_ns'] for event in events)
                require(all(event['service_ns']==event['submit_ns']+event['wait_ns'] for event in events),'double-counted service')
                require(region+phase['host_before_ns']+phase['host_after_ns']+phase['host_control_ns']==phase['wall_ns'],'decode phase overlap')
                require(detail['outer_dispatch_ns']==phase['wall_ns']+detail['outside_dispatch_wrapper_ns'],'outer decode overlap')
                checked+=detail['outer_dispatch_ns']+detail['argmax_ns']
            require(checked==request['wall_ns'] and request['exclusive_sum_equals_wall'],'request timings do not conserve actual wall')
            require(math.isclose(row['total_ms']*1e6,checked,abs_tol=1.,rel_tol=0),'reported request time changed')
            require(row['memory']['VmRSS_bytes']<=20*2**30 and row['memory']['VmSwap_bytes']<=.5*2**30,'original memory guard exceeded')
        if 'receipt' in cohort:
            receipt=load(cohort['receipt']);routed_hashes.add(receipt['routing_sha256'])
            require(sha(Path(cohort['receipt']).parent/'plan.json')==receipt['plan_sha256'],'heldout plan changed')
        summary=dict(name=name,source_id=contract['source_id'],plan_id=contract['plan_id'],
            measurements_sha256=sha(directory/'measurements.jsonl'),capture_sha256=sha(directory/'capture.npz'),
            all_sample_tokens_equal=all(row['tokens']==rows[0]['tokens'] for row in rows),
            source_operators_per_request=581,decode_steps_per_request=127,
            total_ms=dict(median=statistics.median(row['total_ms'] for row in rows),samples=[row['total_ms'] for row in rows]),
            ttft_ms=dict(median=statistics.median(row['ttft_ms'] for row in rows),samples=[row['ttft_ms'] for row in rows]),
            maximum_rss_bytes=max(row['memory']['VmRSS_bytes'] for row in rows),
            maximum_swap_bytes=max(row['memory']['VmSwap_bytes'] for row in rows))
        report['cohorts'].append(summary);timings[name]=summary;allrows[name]=rows
        captures[name]=directory/'capture.npz'
    require(len(routed_hashes)<=1,'baseline and selected used different route shims')
    selected=spec.get('selected')
    if selected:
        for baseline in spec.get('baselines',[]):
            report['comparisons'].append(dict(baseline=baseline,selected=selected,
                baseline_over_selected_request=timings[baseline]['total_ms']['median']/timings[selected]['total_ms']['median'],
                baseline_over_selected_ttft=timings[baseline]['ttft_ms']['median']/timings[selected]['ttft_ms']['median']))
    if a.captures:
        import numpy as np
        comparator=Path(spec['comparator']);s=importlib.util.spec_from_file_location('original_r18_vendor_comparator',comparator)
        vendor=importlib.util.module_from_spec(s);sys.modules[s.name]=vendor;s.loader.exec_module(vendor)
        report['comparator_sha256']=sha(comparator)
        paths=dict(captures);paths.update({ref['name']:Path(ref['capture']) for ref in spec.get('references',[])})
        arrays={};report['captures']=[]
        for name,path in paths.items():
            with np.load(path,allow_pickle=False) as z:row={k:z[k].copy() for k in ('logits','tokens','prompt')}
            require(row['logits'].dtype==np.float32 and row['logits'].shape[1]==151936 and np.isfinite(row['logits']).all(),'invalid finite f32 logits')
            require(row['tokens'].shape==(len(row['logits']),) and np.array_equal(row['tokens'],row['logits'].argmax(axis=1)),'captured token/logit disagreement')
            if name in allrows:
                require(row['logits'].shape==(128,151936),'incomplete full128 logits')
                require(row['tokens'].tolist()==next(r['tokens'] for r in allrows[name] if r['sample']==0),'capture is not heldout sample0')
            arrays[name]=row;report['captures'].append(dict(name=name,file=str(path),sha256=sha(path),logit_rows=len(row['logits'])))
        for left,right in spec['pairs']:
            x,y=arrays[left],arrays[right];require(np.array_equal(x['prompt'],y['prompt']),'numerical pair has different prompts')
            count=min(len(x['tokens']),len(y['tokens']));div=np.flatnonzero(x['tokens'][:count]!=y['tokens'][:count]);first=int(div[0]) if len(div) else None
            comparable=count if first is None else first+1;gate=min(32,count)
            xt=x['tokens'][:gate].tolist();yt=y['tokens'][:gate].tolist()
            xk=np.argsort(-x['logits'][:gate],axis=1,kind='stable')[:,:5].tolist()
            yk=np.argsort(-y['logits'][:gate],axis=1,kind='stable')[:,:5].tolist()
            check=vendor.compute_topk_set_check(xt,xk,yt,yk,k=5).to_dict()
            maximum=0.;squares=0.
            for i in range(comparable):
                delta=x['logits'][i].astype(np.float64)-y['logits'][i].astype(np.float64)
                maximum=max(maximum,float(np.abs(delta).max()));squares+=float(np.dot(delta,delta))
            exact=lambda n:bool(np.array_equal(x['logits'][:n].view(np.uint32),y['logits'][:n].view(np.uint32)))
            report['pairs'].append(dict(left=left,right=right,original_32_token_mutual_top5=check,
                compared_token_rows=count,tokens_equal=not len(div),first_token_divergence_zero_based=first,
                same_context_logit_rows=comparable,same_context_logits_bitexact=exact(comparable),
                same_context_max_abs_error=maximum,same_context_rmse=math.sqrt(squares/(comparable*151936)),
                full128_logits_bitexact=exact(128) if len(x['logits'])==len(y['logits'])==128 else None,
                post_divergence_error_metrics_computed=False))
        report['all_original_vendor_gates_pass']=all(row['original_32_token_mutual_top5']['status']=='OK' for row in report['pairs'])
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(a.output),comparisons=report['comparisons'],captures_audited=a.captures,
        all_original_vendor_gates_pass=report.get('all_original_vendor_gates_pass'))))
    if a.captures:require(report['all_original_vendor_gates_pass'],'original numerical gate failed')


if __name__=='__main__':main()
