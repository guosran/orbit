#!/usr/bin/env python3
"""Execute compiler-issued partition manifests with the existing XRT runner.

No graph analysis, partition choice, AIR rewrite, or numerical fallback occurs
here. The native compiler supplies every template, slice, and invocation.
"""
import argparse, fcntl, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from taskflow_direct_air_execute import compile_air, digest, save
from taskflow_npu_trace_v3 import ObservedXRT, ObservedRun
from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner
from triton.backends.amd_triton_npu import multilaunch


def validate(manifest, directory):
    if manifest.get('schema') != 'orbit.program.partition.v1':
        raise ValueError('unsupported native program manifest schema')
    if any(b['role']=='output' and b['partition_axis']<0 for b in manifest['buffers']):raise ValueError('partition manifest must explicitly slice every output')
    if not manifest['templates'] or not manifest['invocations']:raise ValueError('empty native program schedule')
    templates={t['id']:t for t in manifest['templates']}
    if len(templates)!=len(manifest['templates']):raise ValueError('duplicate template')
    abis={key:json.loads((directory/t['abi']).read_text()) for key,t in templates.items()}
    parent_entries=manifest.get('template_plans',[])
    parent_plans={p['template_id']:p for p in parent_entries}
    if len(parent_plans)!=len(parent_entries) or set(parent_plans)!=set(templates):raise ValueError('parent/template plan identity coverage mismatch')
    for key,template in templates.items():
        native=json.loads((directory/template['plan']).read_text())
        if template.get('mapping_plan_id')!=native['plan_id']:raise ValueError('template mapping plan id differs from native plan')
        if parent_plans[key]['plan']!=native:raise ValueError('parent embedded plan differs from native template plan')
        if parent_plans[key]['count']!=sum(i['template_id']==key for i in manifest['invocations']):raise ValueError('parent template invocation count differs from schedule')
    done=set();coverage={i:np.zeros(b['shape'][b['partition_axis']],dtype=np.uint8)
        for i,b in enumerate(manifest['buffers']) if b['role']=='output' and b['partition_axis']>=0}
    for invocation in manifest['invocations']:
        if invocation['index'] in done or not set(invocation['depends_on'])<=done:
            raise ValueError('invocation order/dependency violation')
        if invocation['template_id'] not in templates:raise ValueError('unknown invocation template')
        if not 0<=invocation['lower']<invocation['upper']<=manifest['partition']['extent']:raise ValueError('invalid invocation range')
        if invocation['upper']-invocation['lower']!=templates[invocation['template_id']]['extent']:raise ValueError('invocation extent differs from template')
        abi=abis[invocation['template_id']]
        bindings=invocation['bindings']
        if sorted(b['buffer_index'] for b in bindings)!=list(range(len(abi['buffers']))):
            raise ValueError('incomplete/duplicate ABI bindings')
        for binding in bindings:
            local=abi['buffers'][binding['buffer_index']]
            if not 0<=binding['full_buffer_index']<len(manifest['buffers']):raise ValueError('invalid full buffer index')
            full=manifest['buffers'][binding['full_buffer_index']]
            if local['name']!=full['name'] or local['role']!=full['role'] or local['dtype']!='f32' or full['dtype']!='f32':
                raise ValueError('binding identity/FP32 ABI mismatch')
            if binding['sizes']!=local['shape']:raise ValueError('binding sizes differ from compiled ABI')
            if binding['mode']=='temporary':
                if full['role']!='temporary' or local['role']!='temporary':raise ValueError('invalid scratch binding')
                continue
            if len(binding['offsets'])!=len(full['shape']) or len(binding['sizes'])!=len(full['shape']):raise ValueError('binding rank mismatch')
            if any(o<0 or n<=0 or o+n>limit for o,n,limit in zip(binding['offsets'],binding['sizes'],full['shape'])):raise ValueError('slice outside full buffer')
            if binding['mode']=='replicated-input':
                if full['role']!='input' or binding['offsets']!=[0]*len(full['shape']) or binding['sizes']!=full['shape']:raise ValueError('invalid replicated input')
            elif binding['mode'] in ('slice-input','slice-output'):
                expected_role='input' if binding['mode']=='slice-input' else 'output'
                if full['role']!=expected_role or full['partition_axis']<0:raise ValueError('invalid sliced binding')
                axis=full['partition_axis']
                for d,(offset,size) in enumerate(zip(binding['offsets'],binding['sizes'])):
                    if d!=axis and (offset!=0 or size!=full['shape'][d]):raise ValueError('slice changes independent nonpartitioned axis')
                if binding['offsets'][axis]!=invocation['lower'] or binding['sizes'][axis]!=invocation['upper']-invocation['lower']:raise ValueError('slice differs from compiler invocation')
                if binding['mode']=='slice-output':
                    view=coverage[binding['full_buffer_index']][invocation['lower']:invocation['upper']]
                    if np.any(view):raise ValueError('overlapping full output partitions')
                    view[:]=1
            else:raise ValueError('unknown binding mode')
        done.add(invocation['index'])
    if any(not np.all(c==1) for c in coverage.values()):raise ValueError('incomplete full output coverage')
    return templates,abis


def run_seed(manifest,templates,abis,compiled,inputs,reference,out,iterations,warmups):
    buffers=manifest['buffers'];runners={};events=[];active=None
    start=time.perf_counter_ns()
    for key,artifact in compiled.items():
        runner=MultiLaunchRunner(artifact['elf_path'],artifact['kernel_name'])
        runner._xrt=ObservedXRT(runner._xrt,events);runners[key]=runner
    load_ms=(time.perf_counter_ns()-start)/1e6
    original_wait=multilaunch._wait
    def wait(run):
        begin=time.perf_counter_ns()
        save(out/'wait-in-progress.json',dict(invocation=active,start_ns=begin,observed_events=events))
        try:value=original_wait(run.raw if isinstance(run,ObservedRun) else run)
        except Exception as error:
            events.append(dict(kind='actual_xrt_wait_failed',start_ns=begin,end_ns=time.perf_counter_ns()))
            save(out/'wait-failure.json',dict(invocation=active,message=str(error),observed_events=events));raise
        events.append(dict(kind='actual_xrt_wait',start_ns=begin,end_ns=time.perf_counter_ns()))
        (out/'wait-in-progress.json').unlink();return value
    multilaunch._wait=wait;samples=[]
    try:
        for repeat in range(iterations+warmups):
            outputs={b['name']:np.empty(b['shape'],dtype=np.float32) for b in buffers if b['role']=='output'}
            records=[];begin=time.perf_counter_ns()
            for invocation in manifest['invocations']:
                active=dict(repeat=repeat,index=invocation['index'],template_id=invocation['template_id'],lower=invocation['lower'],upper=invocation['upper'])
                key=invocation['template_id'];abi=abis[key];arrays=[None]*len(abi['buffers']);static=set();scratch=set();readback=set();bindings=invocation['bindings']
                part_begin=time.perf_counter_ns();events.clear()
                for binding in bindings:
                    i=binding['buffer_index'];full=buffers[binding['full_buffer_index']];mode=binding['mode']
                    if mode in ('replicated-input','slice-input'):
                        value=inputs[full['name']]
                        if list(value.shape)!=full['shape'] or value.dtype!=np.float32:raise ValueError('full input ABI mismatch')
                        slices=tuple(slice(o,o+n) for o,n in zip(binding['offsets'],binding['sizes']))
                        arrays[i]=np.ascontiguousarray(value[slices])
                        if mode=='replicated-input':static.add(i)
                    else:
                        arrays[i]=np.zeros(abi['buffers'][i]['shape'],dtype=np.float32)
                        if mode=='temporary':scratch.add(i)
                        else:readback.add(i)
                values=runners[key].run(arrays,bo_key='native-template',static_indices=static,intermediate_indices=scratch,output_indices=readback)
                pieces={}
                for binding in bindings:
                    if binding['mode']!='slice-output':continue
                    i=binding['buffer_index'];name=buffers[binding['full_buffer_index']]['name']
                    value=values[i].reshape(arrays[i].shape)
                    slices=tuple(slice(o,o+n) for o,n in zip(binding['offsets'],binding['sizes']))
                    np.copyto(outputs[name][slices],value);pieces[name]=value.copy()
                part_end=time.perf_counter_ns()
                records.append(dict(**active,wall_ms=(part_end-part_begin)/1e6,api_events=[{**e,'start_ns':e['start_ns']-begin,'end_ns':e['end_ns']-begin} for e in events],actual_submissions=sum(e['kind']=='actual_xrt_start' for e in events)))
                np.savez(out/f'parts-{repeat}-{invocation["index"]}.npz',**pieces)
                save(out/'progress.json',dict(repeat=repeat,completed_invocations=len(records),total_invocations=len(manifest['invocations']),elapsed_ms=(time.perf_counter_ns()-begin)/1e6,invocations=records))
                print(json.dumps(dict(seed=out.name,repeat=repeat,completed_invocations=len(records),total_invocations=len(manifest['invocations']),invocation_ms=records[-1]['wall_ms'])),flush=True)
            wall_ms=(time.perf_counter_ns()-begin)/1e6
            checks={}
            for name,actual in outputs.items():
                expected=reference[name];finite=bool(np.isfinite(actual).all() and np.isfinite(expected).all());delta=actual.astype(np.float64)-expected.astype(np.float64)
                checks[name]=dict(shape=list(actual.shape),finite=finite,max_abs=float(np.max(np.abs(delta))) if finite else None,normalized_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(expected.astype(np.float64)),1e-30)) if finite else None,passed=finite and actual.shape==expected.shape and bool(np.allclose(actual,expected,atol=1e-5,rtol=1e-5)))
            np.savez(out/f'outputs-{repeat}.npz',**outputs)
            samples.append(dict(repeat=repeat,warmup=repeat<warmups,wall_ms=wall_ms,checks=checks,invocations=records,actual_submissions=sum(r['actual_submissions'] for r in records)))
            save(out/'samples-in-progress.json',samples)
    finally:
        multilaunch._wait=original_wait
        for runner in runners.values():runner.unload()
    result=dict(status='passed' if all(c['passed'] for s in samples for c in s['checks'].values()) else 'numeric_failed',load_ms=load_ms,samples=samples,atol=1e-5,rtol=1e-5,median_ms=float(np.median([s['wall_ms'] for s in samples if not s['warmup']])),boundary='complete compiler-issued program: slice copies, all runner staging/sync/submit/waits/readbacks, output assembly, per-invocation evidence/checkpoint IO; excludes context creation, full input/reference loading, final numerical checks and full-output serialization',storage='static weights and temporary scratch reuse existing runner system-memory BOs, not AIE SRAM residency')
    save(out/'result.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--inputs',type=Path,action='append',required=True);p.add_argument('--reference',type=Path,action='append',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--diagnostic',action='store_true');p.add_argument('--long-acceptance',action='store_true');p.add_argument('--iterations',type=int,default=3);a=p.parse_args()
    if len(a.inputs)!=len(a.reference) or not a.inputs:p.error('paired full input/reference files required')
    if a.diagnostic and a.long_acceptance:p.error('choose diagnostic or long acceptance')
    if a.long_acceptance and len(a.inputs)<3:p.error('long acceptance requires at least three independent seeds')
    if not(a.diagnostic or a.long_acceptance) and (len(a.inputs)<3 or a.iterations<3):p.error('ordinary acceptance requires >=3 seeds and >=3 hot iterations')
    iterations=1 if a.diagnostic or a.long_acceptance else a.iterations;warmups=0 if a.diagnostic or a.long_acceptance else 2
    manifest=json.loads(a.manifest.read_text());directory=a.manifest.resolve().parent;templates,abis=validate(manifest,directory)
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);shutil.copy2(a.manifest,out/'manifest.json');shutil.copy2(__file__,out/'executor-source.py')
    save(out/'execution-invocation.json',dict(argv=sys.argv,cwd=str(Path.cwd()),manifest_path=str(a.manifest.resolve())))
    save(out/'measurement-mode.json',dict(diagnostic=a.diagnostic,long_acceptance=a.long_acceptance,hot_samples=iterations,warmups=warmups,manifest_sha256=digest(a.manifest)))
    compiled={}
    for key,t in templates.items():
        folder=out/key
        if a.long_acceptance and not all('transfer_summary' in stage for stage in json.loads((directory/t['plan']).read_text())['hardware_stages']):raise ValueError('long acceptance requires current native per-stage transfer summaries')
        compiled[key]=compile_air(directory/t['air'],folder,omit_ping_pong=True)
        for name in ['taskflow','plan','abi']:shutil.copy2(directory/t[name],folder/(name+Path(t[name]).suffix))
        save(folder/'source-identity.json',{name:dict(path=str(directory/t[name]),sha256=digest(directory/t[name])) for name in ['taskflow','air','plan','abi']})
        if len(abis[key]['buffers'])>compiled[key]['argument_count']:raise ValueError('compiled ABI argument mismatch')
        subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'test/TaskflowAIR/audit_storage_v6.py'),str(folder),'--plan',str(directory/t['plan']),'--output',str(folder/'storage-audit.json')],check=True)
        subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'test/TaskflowAIR/audit_runtime_transfers_v7.py'),str(folder),'--plan',str(directory/t['plan']),'--abi',str(directory/t['abi']),'--output',str(folder/'runtime-transfer-audit.json')],check=True)
        if a.long_acceptance and any(stage['expected_scope']!='per-stage-native-transfers' for stage in json.loads((folder/'runtime-transfer-audit.json').read_text())['stages']):raise ValueError('long acceptance runtime audit must compare every native stage')
    results=[]
    with open('/tmp/npu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for i,(ip,rp) in enumerate(zip(a.inputs,a.reference)):
            folder=out/f'seed-{i}';folder.mkdir();shutil.copy2(ip,folder/'inputs.npz');shutil.copy2(rp,folder/'reference.npz');save(folder/'input-provenance.json',dict(inputs_sha256=digest(ip),reference_sha256=digest(rp)))
            try:results.append(run_seed(manifest,templates,abis,compiled,dict(np.load(ip)),dict(np.load(rp)),folder,iterations,warmups))
            except Exception as error:
                save(folder/'failure.json',dict(type=type(error).__name__,message=str(error)));save(out/'results.json',dict(status='execution_failed',results=results));raise
            save(out/'results.json',dict(status='running',results=results))
    success=all(r['status']=='passed' for r in results);save(out/'results.json',dict(status='passed' if success else 'numeric_failed',results=results));return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
