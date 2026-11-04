#!/usr/bin/env python3
"""Read-only independent arithmetic/plan/evidence audit of singleton region execution."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def legacy_main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=a.evidence
    result=json.loads((d/'result.json').read_text());program=json.loads((d/'program.json').read_text());plan=json.loads((d/'selected-plan.json').read_text());errors=[]
    def need(ok,msg):
        if not ok:errors.append(msg)
    need(result['status']=='passed','execution did not pass')
    need(result['plan_id']==plan['plan_id'],'reported plan identity mismatch')
    for field,file in [('plan_sha256','selected-plan.json'),('source_sha256','taskflow.mlir'),('config_sha256','search-config.json')]:
        need(result[field]==hashlib.sha256((d/file).read_bytes()).hexdigest(),'bound source/config/plan file changed: '+file)
    inputs={k:v.astype('f8') for k,v in np.load(d/'inputs-f32.npz').items()}
    state={**inputs,**{b['id']:np.zeros(b['shape'],dtype='f8') for b in program['buffers'] if b['id'] not in inputs}}
    covered=set();regions={e['id']:e for e in plan['events']};previous=set()
    for region in result['regions']:
        event=region['event'];need(event==regions[event['id']],'runtime region differs from native plan')
        need(set(event['depends_on'])<=previous,'runtime region order violates dependency');previous.add(event['id'])
        need(not covered.intersection(event['covered_steps']),'duplicate source-step execution');covered.update(event['covered_steps'])
    expected_steps={f"{t['id']}:{i}" for t in program['tasks'] for i in range(len(t['operations']))}
    need(covered==expected_steps,'source operation coverage missing/extra')
    for t in program['tasks']:
        for op in t['operations']:
            m,n=slice(*op['m']),slice(*op['n']);dst=state[op['output']][m,n]
            if op['kind']=='fill':dst.fill(op.get('value',0))
            elif op['kind']=='matmul':
                k=slice(*op['k']);value=state[op['inputs'][0]][m,k]@state[op['inputs'][1]][k,n]
                if op.get('accumulate',False):dst+=value
                else:dst[:]=value
            else:raise ValueError('unsupported arithmetic in independent audit')
    reference=state['Y'];need(np.array_equal(reference,np.load(d/'reference-f64.npy')),'stored reference differs from independently replayed source')
    checks=[]
    for i,s in enumerate(result['samples']):
        y=np.load(d/f'output-{i}.npy');err=y.astype('f8')-reference
        rms=float(np.sqrt(np.mean(err**2))/max(np.sqrt(np.mean(reference**2)),1e-30));peak=float(np.max(abs(err))/max(np.max(abs(reference)),1e-30))
        passed=y.shape==reference.shape and np.isfinite(y).all() and np.all(abs(err)<=.015+.06*abs(reference)) and rms<=.03 and peak<=.08
        need(passed,'independent numerical mismatch sample '+str(i))
        starts=sum(e['kind']=='actual_xrt_start' for e in s['events']);waits=sum(e['kind']=='actual_xrt_wait' for e in s['events'])
        need(starts==waits==len(regions)==s['actual_submissions']==s['actual_waits'],'actual submission/wait count mismatch')
        checks.append(dict(sample=i,passed=bool(passed),max_abs=float(abs(err).max()),normalized_rms=rms,peak_scaled=peak,actual_starts=starts,actual_waits=waits))
    median=statistics.median(s['wall_ms'] for s in result['samples']);need(median==result['median_ms'],'median mismatch');need(result['prediction_ms']==plan['totals']['makespan_ms'],'prediction not bound to native plan')
    report=dict(passed=not errors,errors=errors,plan_id=plan['plan_id'],candidate_id=result['candidate_id'],samples=checks,predicted_ms=result['prediction_ms'],observed_median_ms=median,relative_prediction_error=(result['prediction_ms']-median)/median,hardware_executed_by_audit=False,limitations=['Comparison is against full independent float64 semantic source, with unchanged BF16/BFP16 tolerances.','Opaque warm host-to-host timing includes activation conversion, assembly and sync; not compute-only.','Final runtime object snapshot is not a per-sample object-lifetime proof.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=report['passed'],outputs=len(checks),errors=errors)));return 0 if report['passed'] else 1



def overlap(a,b):
    return all(max(x[0],y[0])<min(x[1],y[1]) for x,y in zip(a,b))


def audit_batch(d):
    result=json.loads((d/'result.json').read_text());errors=[];case_reports={};all_outputs={};snapshots={}
    def need(ok,msg):
        if not ok:errors.append(msg)
    need(result['status']=='passed','batch execution failed')
    need(result['prepared_state']=='warm_contexts_weights','wrong preparation boundary')
    inputs={k:v.astype('f8') for k,v in np.load(d/'inputs-f32.npz').items()}
    ref=inputs['X']@inputs['W1']@inputs['W2'];need(np.array_equal(ref,np.load(d/'reference-f64.npy')),'independent reference mismatch')
    prepared={}
    if 'prepared_weights' in result:
        from ml_dtypes import bfloat16
        for item in result['prepared_weights']:
            key=item['kernel_key'];arrays=[]
            for descriptor,meta in zip(key[1],item['weights']):
                m,k,n,weightname,n0=descriptor
                value=np.ascontiguousarray(inputs[weightname][:k,n0:n0+n],dtype=bfloat16);arrays.append(value)
                need(meta['sha256']==hashlib.sha256(value.tobytes()).hexdigest() and meta['shape']==list(value.shape) and meta['dtype']=='bfloat16','prepared weight slice/content/dtype mismatch')
            need(item['bo_key']=='weights-'+hashlib.sha256(b''.join(x.tobytes() for x in arrays)).hexdigest(),'prepared BO generation mismatch')
            prepared[json.dumps(key)]=item
    for name,case in result['cases'].items():
        cd=d/name;program=json.loads((cd/'program.json').read_text());plan=json.loads((cd/'selected-plan.json').read_text());config=json.loads((cd/'config.json').read_text())
        for field,file in [('plan_sha256','selected-plan.json'),('source_sha256','taskflow.mlir'),('config_sha256','config.json')]:
            need(case[field]==hashlib.sha256((cd/file).read_bytes()).hexdigest(),name+': changed bound '+file)
        need(case['plan_id']==plan['plan_id'] and case['candidate_id']==program['candidate_id']==plan['candidate_id'],name+': identity mismatch')
        steps={f"{t['id']}:{i}":op for t in program['tasks'] for i,op in enumerate(t['operations'])}
        impls={i['id']:i for i in config['region_implementations']};regions={e['id']:e for e in plan['events']};seen=set();done=set();expected_ids=[];keys=[]
        state={**inputs,**{b['id']:np.zeros(b['shape'],dtype='f8') for b in program['buffers'] if b['id'] not in inputs}}
        for task in program['tasks']:
            for op in task['operations']:
                ms,ns=slice(*op['m']),slice(*op['n']);dst=state[op['output']][ms,ns]
                if op['kind']=='fill':dst.fill(op.get('value',0))
                elif op['kind']=='matmul':dst+=state[op['inputs'][0]][ms,slice(*op['k'])]@state[op['inputs'][1]][slice(*op['k']),ns]
                else:raise ValueError('unsupported source arithmetic')
        need(np.allclose(state['Y'],ref,rtol=1e-13,atol=1e-13),name+': transformed source changes reference')
        for region in case['regions']:
            e=region['event'];need(e==regions[e['id']],name+': executed region differs from native plan')
            need(set(e['depends_on'])<=done,name+': invalid execution order');done.add(e['id']);expected_ids.append(e['id'])
            covered=e['covered_steps'];need(not seen.intersection(covered),name+': duplicate source coverage');seen.update(covered)
            impl=impls[e['implementation_id']]
            for field,value in dict(device='npu',dtype='f32',layout='row_major',numerical_contract='bf16-bfp16-bf16-mid-final',prepared_state='warm_contexts_weights',boundary_memory='host').items():need(impl.get(field)==value,name+': incorrect '+field)
            operations=[steps[k] for k in covered];muls=[op for op in operations if op['kind']=='matmul'];muls.sort(key=lambda op:0 if op['inputs'][0]=='X' else 1)
            pair=e['region_pattern']=='matmul_pair';need((e['region_pattern'],e['region_mode']) in [('matmul_pair','chain'),('matmul_single','independent')],name+': unsupported mode')
            need(len(muls)==(2 if pair else 1) and len(operations)==2*len(muls),name+': wrong source region body')
            need(region['operations']==muls,name+': runtime operations differ from source')
            for op in muls:
                matching=[f for f in operations if f['kind']=='fill' and f['output']==op['output'] and f['m']==op['m'] and f['n']==op['n'] and f.get('value',0)==0]
                need(len(matching)==1 and op['k'][0]==0 and op['accumulate'],name+': missing zero/full-K semantics')
                need(op['k'][1]==state[op['inputs'][0]].shape[1],name+': partial K silently overwritten')
            if pair:
                first,last=muls;need(first['m']==last['m'] and first['n']==last['k'],name+': incompatible internal H region')
                internal=[first['m'],first['n']]
                for sid,op in steps.items():
                    if sid in covered:continue
                    for index,buffer in enumerate(op.get('inputs',[])):
                        if buffer!=first['output']:continue
                        read=[op['m'],op['k']] if index==0 else [op['k'],op['n']]
                        need(not overlap(internal,read),name+': pair hides H from an external consumer')
            key=[e['region_mode'],[[op['m'][1]-op['m'][0],op['k'][1],op['n'][1]-op['n'][0],op['inputs'][1],op['n'][0]] for op in muls]]
            need(region['key']==key,name+': runtime key misses weight slice/mode');keys.append(key)
        need(seen==set(steps) and done==set(regions),name+': incomplete exact cover')
        outputs=[];checks=[]
        for i,sample in enumerate(case['samples']):
            y=np.load(cd/f'output-{i}.npy');error=y.astype('f8')-ref
            rms=float(np.sqrt(np.mean(error**2))/max(np.sqrt(np.mean(ref**2)),1e-30));peak=float(abs(error).max()/max(abs(ref).max(),1e-30))
            passed=y.shape==ref.shape and np.isfinite(y).all() and np.all(abs(error)<=.015+.06*abs(ref)) and rms<=.03 and peak<=.08
            need(passed,name+': numerical failure '+str(i));outputs.append(y)
            starts=sum(e['kind']=='actual_xrt_start' for e in sample['events']);waits=sum(e['kind']=='actual_xrt_wait' for e in sample['events'])
            need(starts==waits==len(regions)==sample['actual_submissions']==sample['actual_waits'],name+': submit/wait mismatch')
            if prepared:
                executed=[e for e in sample['events'] if e['kind']=='region_execution']
                need([e['region_id'] for e in executed]==expected_ids,name+': actual region execution IDs differ')
                need([e['kind'] for e in sample['events']]==['actual_xrt_start','actual_xrt_wait','region_execution']*len(regions),name+': unexpected actual dispatch ordering')
                byrunner={r['runner_id']:r for r in sample['runtime_objects']}
                for key,event in zip(keys,executed):
                    item=prepared[json.dumps(key)];runner=byrunner.get(event['runner_id'])
                    need(event['bo_key']==item['bo_key'] and event['weight_object_ids']==[w['object_id'] for w in item['weights']],name+': wrong weight object/generation execution')
                    need(event['readback_indices']==([4] if key[0]=='chain' else [2]),name+': incorrect pair internal-H readback contract')
                    need(runner is not None and event['bo_key'] in runner['bo_sets'],name+': missing actual BO set')
                    identity=json.dumps(key);old=snapshots.setdefault(identity,runner)
                    need(runner==old,name+': context/kernel/BO changed during hot samples')
            checks.append(dict(index=i,passed=bool(passed),normalized_rms=rms,peak_scaled=peak,actual_submissions=starts,actual_waits=waits))
        all_outputs[name]=outputs;times=[s['wall_ms'] for s in case['samples']];median=statistics.median(times)
        need(median==case['median_ms'],name+': forged median');need(case['prediction_ms']==plan['totals']['makespan_ms'],name+': prediction identity mismatch')
        case_reports[name]=dict(plan_id=case['plan_id'],candidate_id=case['candidate_id'],outputs=checks,median_ms=median,q1_ms=float(np.percentile(times,25)),q3_ms=float(np.percentile(times,75)),min_ms=min(times),max_ms=max(times),predicted_ms=case['prediction_ms'],relative_prediction_error=(case['prediction_ms']-median)/median,execution_ids=expected_ids,region_keys=keys)
    probe_checks=[]
    probes=result.get('weight_generation_probe',[])
    if probes:
        need([p['label'] for p in probes]==['A','B','A'],'weight generation probe is not A/B/A')
        need(len({p['runner_id'] for p in probes})==1,'weight probe does not reuse same runner')
        need(probes[0]['bo_key']==probes[2]['bo_key']!=probes[1]['bo_key'],'weight generations collide')
        from ml_dtypes import bfloat16
        for i,probe in enumerate(probes):
            expected=ref if probe['label']=='A' else -ref
            y=np.load(d/f"weight-probe-{i}-{probe['label']}.npy");err=y.astype('f8')-expected
            rms=float(np.sqrt(np.mean(err**2))/max(np.sqrt(np.mean(expected**2)),1e-30));peak=float(abs(err).max()/max(abs(expected).max(),1e-30))
            passed=np.isfinite(y).all() and np.all(abs(err)<=.015+.06*abs(expected)) and rms<=.03 and peak<=.08
            need(passed,'A/B/A independent output mismatch')
            w1=np.ascontiguousarray(inputs['W1'],dtype=bfloat16);w2=np.ascontiguousarray(inputs['W2'],dtype=bfloat16)
            if probe['label']=='B':w1=-w1
            need(probe['weight_sha256']==[hashlib.sha256(w.tobytes()).hexdigest() for w in [w1,w2]],'probe weight content differs')
            need(probe['bo_key']=='weights-'+hashlib.sha256(w1.tobytes()+w2.tobytes()).hexdigest(),'probe explicit generation differs')
            need(sum(e['kind']=='actual_xrt_start' for e in probe['events'])==sum(e['kind']=='actual_xrt_wait' for e in probe['events'])==1,'probe dispatch count mismatch')
            probe_checks.append(dict(label=probe['label'],passed=bool(passed),normalized_rms=rms,peak_scaled=peak))
        need(probes[0]['runtime']['bo_sets'][probes[0]['bo_key']]==probes[2]['runtime']['bo_sets'][probes[2]['bo_key']],'A BO identity not reused after B')
        need(not set(probes[0]['runtime']['bo_sets'][probes[0]['bo_key']])&set(probes[1]['runtime']['bo_sets'][probes[1]['bo_key']]),'A/B BO allocations alias')
    names=list(case_reports);base=result['baseline'];mutual=[]
    for i,name in enumerate(names):
        for other in names[i+1:]:mutual.append(dict(a=name,b=other,max_abs=max(float(abs(x.astype('f8')-y).max()) for x,y in zip(all_outputs[name],all_outputs[other]))))
    observed=sorted(names,key=lambda n:case_reports[n]['median_ms']);predicted=sorted(names,key=lambda n:case_reports[n]['predicted_ms'])
    local=[n for n in names if n!=base];local_pred=min(local,key=lambda n:case_reports[n]['predicted_ms']);local_best=min(local,key=lambda n:case_reports[n]['median_ms'])
    return dict(passed=not errors,errors=errors,weight_generation_probe=probe_checks,prepared_weight_sets_checked=len(prepared),hot_identity_checked=bool(prepared),source=str(d),outputs_checked=sum(len(v) for v in all_outputs.values()),cases=case_reports,output_pairwise_differences=mutual,predicted_order=predicted,observed_order=observed,global_regret_ms=case_reports[predicted[0]]['median_ms']-case_reports[observed[0]]['median_ms'],structural_local_regret_ms=case_reports[local_pred]['median_ms']-case_reports[local_best]['median_ms'],hardware_executed_by_audit=False,limitations=['Same-batch interleaved comparison only; earlier single N256 result is prediction evidence, not this performance ranking.','Frequency unlocked; overlapping sample ranges/IQRs may prevent reliable local N/M ranking.','Prepared immutable weights are keyed by mode, shape, weight name and N slice in separate runners; raw runtime static_indices does not automatically invalidate changed weights.','Pair internal H need not be read back when no outside source consumer overlaps that H region.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if json.loads((a.evidence/'result.json').read_text())['schema']!='orbit.plan-comparison.v1':return legacy_main()
    report=audit_batch(a.evidence);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k in ('passed','errors','outputs_checked','global_regret_ms','structural_local_regret_ms','predicted_order','observed_order')}));return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
