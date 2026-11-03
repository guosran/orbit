#!/usr/bin/env python3
"""Trace the existing ORBIT AIR kernels with the installed XRT runner.

This is an evidence driver, not a new runtime. AIR's trace-size/offset wiring
writes into a deliberately extended output BO. No installed files are edited.
"""
import argparse,contextlib,fcntl,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
from taskflow_common import compare,write_json
ROOT=Path(__file__).resolve().parents[1]
AIE=ROOT.parent/'aie-toy'
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def toolchain():
    import aie.utils.trace.parse as parser
    from triton.backends.amd_triton_npu import driver,multilaunch
    names=['mlir-air','llvm-aie','mlir-aie-no-rtti','triton-xdna']
    return dict(packages={n:importlib.metadata.version(n) for n in names},python=sys.version,source_hashes={str(p):digest(p) for p in [Path(parser.__file__),Path(driver.__file__),Path(multilaunch.__file__)]},firmware=Path('/sys/class/accel/accel0/device/fw_version').read_text().strip(),installed_packages_modified=False)
class ObservedBO:
    def __init__(self,bo,events,index):self.raw=bo;self.events=events;self.index=index
    def __getattr__(self,name):return getattr(self.raw,name)
    def sync(self,direction,*args,**kwargs):
        start=time.perf_counter_ns();value=self.raw.sync(direction,*args,**kwargs)
        self.events.append(dict(kind='actual_bo_sync',direction=str(direction),buffer_index=self.index,start_ns=start,end_ns=time.perf_counter_ns()));return value
class ObservedRun:
    def __init__(self,run,events):self.raw=run;self.events=events
    def __getattr__(self,name):return getattr(self.raw,name)
    def set_arg(self,index,value):return self.raw.set_arg(index,value.raw if isinstance(value,ObservedBO) else value)
    def start(self):
        start=time.perf_counter_ns();value=self.raw.start();self.events.append(dict(kind='actual_xrt_start',start_ns=start,end_ns=time.perf_counter_ns()));return value
class ObservedXRT:
    def __init__(self,xrt,events):self.raw=xrt;self.events=events;self.ext=self;self.next_bo=0
    def __getattr__(self,name):return getattr(self.raw,name)
    def bo(self,*args,**kwargs):
        bo=ObservedBO(self.raw.ext.bo(*args,**kwargs),self.events,self.next_bo);self.next_bo+=1;return bo
    def run(self,*args,**kwargs):return ObservedRun(self.raw.run(*args,**kwargs),self.events)
def numeric(actual,expected):
    c=compare(actual,expected,.015,.06)
    for name,ref in expected.items():
        err=actual[name].astype('f8')-ref
        rms=float(np.sqrt(np.mean(err**2))/max(np.sqrt(np.mean(ref**2)),1e-30));peak=float(np.max(abs(err))/max(np.max(abs(ref)),1e-30))
        c['outputs'][name].update(normalized_rms=rms,peak_scaled=peak)
        c['passed'] &= rms<=.03 and peak<=.08
    return c

def decode(raw,physical,out,device_name=None):
    from aie.utils.trace import TraceConfig,parse_trace
    import aie.utils.trace.parse as parser
    out.mkdir(parents=True,exist_ok=True);np.save(out/'trace.uint32.npy',raw)
    TraceConfig(raw.nbytes,str(out/'trace.txt')).write_trace(raw)
    shutil.copy2(physical,out/'physical.mlir')
    kwargs={} if device_name is None else {'device_name':device_name}
    with (out/'parse.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        events=parse_trace(raw,Path(physical).read_text(),**kwargs)
    write_json(out/'trace.cycles.json',events)
    # A viewer can display this without inventing a physical frequency: one
    # cycle is one viewer microsecond. The axis is intentionally CYCLES.
    visual={'traceEvents':events,'displayTimeUnit':'us','metadata':{'axis':'device cycles; 1 viewer us = 1 cycle, not physical time','host_alignment':'none'}}
    write_json(out/'trace.perfetto.json',visual)
    manifest=dict(schema='orbit.hardware.trace-evidence.v1',raw_path='trace.uint32.npy',physical_mlir_path='physical.mlir',decoded_native_path='trace.cycles.json',native_timestamp_unit='device_cycles',expected_event_patterns=['INSTR_VECTOR','INSTR_LOCK_ACQUIRE_REQ','PORT_RUNNING'],decoder=dict(api='parse_trace',kwargs=kwargs,sha256=digest(parser.__file__)),completeness=dict(status='partial',reason='finite trace windows; no hardware overflow counter; terminal active events may lack ends; no global complete-graph assertion',scope_markers='ORBIT GEMM kernel does not emit INSTR_EVENT_0/1; configured event slots are not proof markers occurred'),capacity_bytes=raw.nbytes,nonzero_words=int(np.count_nonzero(raw)),viewer=dict(path='trace.perfetto.json',axis='cycles, unit-display surrogate only; not calibrated time'),event_count=len(events))
    write_json(out/'manifest.json',manifest);return manifest

def artifact(case,shape):
    m,k,n=shape
    if case=='gemm': base=AIE/'results/orbit-search-v2/small-final-timing/gemm/seed-0/persistent/artifacts'
    else: base=AIE/f'results/orbit-search-v2/full-forward-final-timing/{case}/seed-0/persistent/artifacts'
    files=list(base.glob(f'chain-{m}-{k}-{n}-*/combined-air.mlir'))
    if len(files)!=1: raise ValueError('expected exactly one authenticated prior ORBIT AIR artifact '+str((base,shape)))
    return files[0]

def compile_shape(case,shape,out,trace_bytes):
    out.mkdir(parents=True,exist_ok=True);source=artifact(case,shape);shutil.copy2(source,out/'source.mlir')
    args=['-trace-size',str(trace_bytes),'-trace-offset',str(shape[0]*shape[2]*4)] if trace_bytes else []
    code='from triton.backends.amd_triton_npu.driver import _aircc_compile;import json;from pathlib import Path;a=_aircc_compile('+repr(str(out/'source.mlir'))+',"elf","npu2",'+repr(str(out/'build'))+',extra_args='+repr(args)+');Path("artifacts.json").write_text(json.dumps(a))'
    t=time.perf_counter()
    with (out/'compile.log').open('w') as log: subprocess.run([sys.executable,'-c',code],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    meta=json.loads((out/'artifacts.json').read_text());meta.update(compile_seconds=time.perf_counter()-t,source_path=str(source),source_sha256=digest(source),elf_sha256=digest(meta['elf_path']),extra_args=args,shape=list(shape))
    write_json(out/'compile.json',meta);return meta

def run_case(case,out,iterations,trace_bytes):
    from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner
    from triton.backends.amd_triton_npu import multilaunch
    source_dir=ROOT/f'build/search-npu-v2/data/{case}'
    plan=json.loads((source_dir/'base.json').read_text());inputs=dict(np.load(source_dir/'inputs-0.npz'));reference=dict(np.load(source_dir/'reference-0.npz'))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): raise ValueError('refusing existing result')
    result=dict(schema='orbit.trace.graph-experiment.v1',case=case,status='compile_failed',candidate_id=plan['candidate_id'],source_id=plan['source_id'],input_sha256=digest(source_dir/'inputs-0.npz'),reference_sha256=digest(source_dir/'reference-0.npz'),precision='BF16 inputs, BFP16 EBS8 MAC, FP32 accumulation/output; CPU f32 SwiGLU',variants={},execution='existing MultiLaunchRunner one-op programs, same ORBIT AIR source both variants; not executing static mapping plan')
    shutil.copy2(source_dir/'base.json',out/'plan.json');shutil.copy2(source_dir/'base.mlir',out/'taskflow.mlir')
    write_json(out/'toolchain.json',toolchain())
    buffer_shapes={b['id']:b['shape'] for b in plan['buffers']};calls=[]
    for task in plan['tasks']:
        for oi,op in enumerate(task['operations']):
            if op['kind']=='matmul':
                m=op['m'][1]-op['m'][0];n=op['n'][1]-op['n'][0];k=op['k'][1]-op['k'][0]
                calls.append((task['id'],oi,op,(m,n,k),(((m+127)//128)*128,((k+63)//64)*64,((n+255)//256)*256)))
    shape_list=sorted({c[4] for c in calls});runners=[];observations=[];original_wait=multilaunch._wait
    def observed_wait(run):
        start=time.perf_counter_ns();value=original_wait(run.raw if isinstance(run,ObservedRun) else run)
        observations.append(dict(kind='actual_xrt_wait',start_ns=start,end_ns=time.perf_counter_ns()));return value
    multilaunch._wait=observed_wait
    shutil.copy2(__file__,out/'driver-source.py')
    try:
        compiled={}
        for mode,size in [('off',0),('on',trace_bytes)]:
            for shape in shape_list: compiled[mode,shape]=compile_shape(case,shape,out/mode/('shape-'+'-'.join(map(str,shape))),size)
        result['status']='runtime_failed'
        with open('/tmp/npu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for mode,size in [('off',0),('on',trace_bytes)]:
                objects={};prepared={};loads={};trace_outputs={};preparation_ms={}
                for ti,oi,op,(m,n,k),shape in calls:
                    meta=compiled[mode,shape];t=time.perf_counter();runner=MultiLaunchRunner(meta['elf_path'],meta['elf_kernel_name']);loads[f'{ti}:{oi}']=(time.perf_counter()-t)*1000;runners.append(runner);objects[ti,oi]=runner;runner._xrt=ObservedXRT(runner._xrt,observations)
                    t=time.perf_counter()
                    mp,kp,np_=shape; b=np.zeros((kp,np_),dtype=bfloat16);b[:k,:n]=inputs[op['inputs'][1]][slice(*op['k']),slice(*op['n'])]
                    prepared[ti,oi]=(b,np.zeros(mp*np_+size//4,dtype=np.float32))
                    preparation_ms[f'{ti}:{oi}']=(time.perf_counter()-t)*1000
                samples=[];exclusions=[];checks=[];host=[]
                for repeat in range(iterations+2):
                    observations.clear();sections=[];tgraph=time.perf_counter_ns();memory={**inputs,**{b:np.empty(s,dtype=np.float32) for b,s in buffer_shapes.items() if b not in inputs}}
                    for task in plan['tasks']:
                        for oi,op in enumerate(task['operations']):
                            node_start=time.perf_counter_ns();ms,ns=slice(*op['m']),slice(*op['n']);dest=memory[op['output']][ms,ns]
                            if op['kind']=='fill':dest.fill(0)
                            elif op['kind']=='matmul':
                                _,_,_,(m,n,k),shape=next(c for c in calls if c[0]==task['id'] and c[1]==oi)
                                mp,kp,np_=shape;a=np.zeros((mp,kp),dtype=bfloat16);a[:m,:k]=memory[op['inputs'][0]][ms,slice(*op['k'])]
                                b,c=prepared[task['id'],oi];c.fill(0);ready=time.perf_counter_ns()
                                values=objects[task['id'],oi].run([a,b,c],bo_key=f'{task["id"]}:{oi}',static_indices={1},output_indices={2})[2]
                                finished=time.perf_counter_ns();dest+=values[:mp*np_].reshape(mp,np_)[:m,:n]
                                if size:trace_outputs[task['id'],oi]=values[mp*np_:].view(np.uint32).copy()
                                sections.append(dict(task=task['id'],operation=oi,kind='host_prepare_convert',start_ns=node_start-tgraph,end_ns=ready-tgraph))
                                sections.append(dict(task=task['id'],operation=oi,kind='runner_staging_sync_submit_wait_readback',start_ns=ready-tgraph,end_ns=finished-tgraph,actual_submissions=1))
                            elif op['kind']=='relu':np.maximum(memory[op['inputs'][0]][ms,ns],0,out=dest)
                            elif op['kind']=='swiglu':
                                gate=memory[op['inputs'][0]][ms,ns];up=memory[op['inputs'][1]][ms,ns];dest[:]=gate/(1+np.exp(-gate))*up
                            else:raise ValueError('unsupported graph operation')
                            sections.append(dict(task=task['id'],operation=oi,kind=op['kind'],placement='npu_with_host_boundary' if op['kind']=='matmul' else 'cpu',start_ns=node_start-tgraph,end_ns=time.perf_counter_ns()-tgraph))
                    output={b:memory[b].copy() for b in plan['outputs']};wall=(time.perf_counter_ns()-tgraph)/1e6
                    checks.append(numeric(output,reference));(exclusions if repeat<2 else samples).append(wall)
                    api=[{**e,'start_ns':e['start_ns']-tgraph,'end_ns':e['end_ns']-tgraph} for e in observations]
                    actual_starts=sum(e['kind']=='actual_xrt_start' for e in api)
                    if actual_starts!=len(calls):raise ValueError('observed submission count mismatch')
                    host.append(dict(repeat=repeat,excluded=repeat<2,sections=sections,api_events=api,actual_submissions=actual_starts))
                    np.savez(out/mode/f'outputs-{repeat}.npz',**output)
                result['variants'][mode]=dict(full_graph_ms=samples,excluded_ms=exclusions,median_ms=float(np.median(samples)),numeric_checks=checks,load_ms=loads,first_host_preparation_ms=preparation_ms,trace_capacity_bytes=size,submissions_per_graph=len(calls),submissions_observed=True,trace_readback_included=bool(size))
                write_json(out/mode/'host-events.json',host)
                for key,raw in trace_outputs.items():
                    shape=next(c[4] for c in calls if c[:2]==key);physical=out/mode/('shape-'+'-'.join(map(str,shape)))/'input_with_addresses.mlir'
                    capture=out/mode/f'trace-{key[0]}-{key[1]}'
                    meta=decode(raw,physical,capture,device_name='op0_matmul_0');meta.update(candidate_id=plan['candidate_id'],task_id=key[0],operation_index=key[1],air_sha256=compiled[mode,shape]['source_sha256'],elf_sha256=compiled[mode,shape]['elf_sha256'],logical_shape=next(list(c[3]) for c in calls if c[:2]==key),physical_shape=list(shape),capture=dict(repeat_index=iterations+1,hot_sample_index=iterations-1,scope='single final hot execution of this projection; trace buffer cleared before every dispatch',graph_host_events='../host-events.json'),toolchain_path='../../toolchain.json');write_json(capture/'manifest.json',meta)
                for runner in objects.values():runner.unload()
                runners=[]
        result['status']='passed' if all(c['passed'] for v in result['variants'].values() for c in v['numeric_checks']) else 'numerical_failed'
        result['trace_perturbation_ratio']=result['variants']['on']['median_ms']/result['variants']['off']['median_ms']
        result['limitations']=['No host/device clock alignment; hardware traces in cycles, host intervals in nanoseconds relative graph start','Port activity is not measured DDR bytes or bandwidth','Trace reserves shim column and changes placement, output BO size, staging/readback and instrumentation; on/off difference is total perturbation, not isolated tracing cost','Finite traces are partial, stalls overlap and must not be summed as disjoint time','Counter/timer clocks not locked; small latency differences are not speedup evidence']
    except Exception as exc:result.update(error=str(exc),traceback=traceback.format_exc())
    finally:
        multilaunch._wait=original_wait
        for r in runners:
            try:r.unload()
            except Exception:pass
        write_json(out/'result.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['gemm','ffn_m5','ffn_m128','all'],default='gemm');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--iterations',type=int,default=7);p.add_argument('--trace-bytes',type=int,default=1048576);a=p.parse_args()
    if a.iterations<3:raise ValueError('at least3 hot samples required')
    results=[run_case(case,a.output_dir.resolve()/case,a.iterations,a.trace_bytes) for case in (['gemm','ffn_m5','ffn_m128'] if a.case=='all' else [a.case])]
    print(json.dumps([dict(case=r['case'],status=r['status'],error=r.get('error')) for r in results],indent=2));return 0 if all(r['status']=='passed' for r in results) else 1
if __name__=='__main__':raise SystemExit(main())
