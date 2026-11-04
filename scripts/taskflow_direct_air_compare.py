#!/usr/bin/env python3
"""Interleave already lowered native AIR variants using existing XRT runners."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from taskflow_direct_air_execute import compile_air,digest,save
from taskflow_npu_trace_v3 import ObservedXRT,ObservedRun
from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner
from triton.backends.amd_triton_npu import multilaunch
p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--inputs',type=Path,action='append',required=True);p.add_argument('--reference',type=Path,action='append',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--iterations',type=int,default=9);a=p.parse_args()
if a.iterations<3 or len(a.inputs)!=len(a.reference):p.error('>=3 samples and paired references required')
out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);cases=json.loads(a.cases.read_text());shutil.copy2(a.cases,out/'cases.json');shutil.copy2(__file__,out/'comparison-source.py')
compiled={}
for case in cases:
 folder=out/case['name'];plan=json.loads(Path(case['plan']).read_text())
 compiled[case['name']]=compile_air(Path(case['air']),folder,omit_ping_pong=plan.get('strategy')=='serial-tiled')
 for key in ['taskflow','plan','abi']:
  shutil.copy2(case[key],folder/(key+Path(case[key]).suffix))
 case['identity']={k:digest(case[k]) for k in ['air','taskflow','plan','abi']}
 if 'mapping' in case:
  native_mapping=Path(case['mapping']).resolve();case['identity']['mapping']=digest(native_mapping)
  shutil.copy2(native_mapping,folder/'native-mapping.json')
  audits=Path(__file__).resolve().parents[1]/'test/TaskflowAIR'
  subprocess.run([sys.executable,str(audits/'audit_runtime_transfers_v7.py'),str(folder),'--plan',str(native_mapping),'--abi',str(Path(case['abi']).resolve()),'--output',str(folder/'runtime-transfer-audit.json')],check=True)
  subprocess.run([sys.executable,str(audits/'audit_storage_v6.py'),str(folder),'--plan',str(native_mapping),'--output',str(folder/'storage-audit.json')],check=True)
save(out/'bound-cases.json',cases)
all_results=[];original_wait=multilaunch._wait;events=[];active_measurement=None
def wait(run):
 start=time.perf_counter_ns()
 pending=active_measurement.with_name(active_measurement.name+'-wait-in-progress.json') if active_measurement else None
 if pending:save(pending,dict(status='waiting_for_actual_xrt',start_ns=start,observed_events=events))
 try:
  value=original_wait(run.raw if isinstance(run,ObservedRun) else run)
 except Exception as error:
  events.append(dict(kind='actual_xrt_wait_failed',start_ns=start,end_ns=time.perf_counter_ns()))
  if active_measurement:save(active_measurement.with_name(active_measurement.name+'-wait-failure.json'),dict(status='actual_xrt_wait_failed',message=str(error),observed_events=events))
  raise
 events.append(dict(kind='actual_xrt_wait',start_ns=start,end_ns=time.perf_counter_ns()))
 if pending:pending.unlink(missing_ok=True)
 return value
multilaunch._wait=wait
try:
 with open('/tmp/npu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for seed,(ip,rp) in enumerate(zip(a.inputs,a.reference)):
   inputs=dict(np.load(ip));reference=dict(np.load(rp));seedout=out/f'seed-{seed}';seedout.mkdir();shutil.copy2(ip,seedout/'inputs.npz');shutil.copy2(rp,seedout/'reference.npz')
   runners={};buffers={};abis={};load={}
   for c in cases:
    name=c['name'];abi=json.loads(Path(c['abi']).read_text());abis[name]=abi
    if any(b['dtype']!='f32' for b in abi['buffers']):raise ValueError('comparison currently f32 ABI only')
    buffers[name]=[np.ascontiguousarray(inputs.get(b['name'],np.zeros(b['shape'],dtype=np.float32)),dtype=np.float32) for b in abi['buffers']]
    start=time.perf_counter();runner=MultiLaunchRunner(compiled[name]['elf_path'],compiled[name]['kernel_name']);load[name]=(time.perf_counter()-start)*1000;runner._xrt=ObservedXRT(runner._xrt,events);runners[name]=runner
   samples=[]
   for repeat in range(a.iterations+2):
    for c in (cases if repeat%2==0 else list(reversed(cases))):
     name=c['name'];active_measurement=seedout/f'{name}-{repeat}';abi=abis[name];indices={i for i,b in enumerate(abi['buffers']) if b['name'] in abi['outputs']};events.clear();start=time.perf_counter_ns()
     values=runners[name].run(buffers[name],bo_key='stable',output_indices=indices)
     outputs={abi['buffers'][i]['name']:values[i].reshape(buffers[name][i].shape).copy() for i in indices};wall=(time.perf_counter_ns()-start)/1e6
     checks={}
     for n,v in outputs.items():
      r=reference[n];delta=v.astype(float)-r.astype(float);checks[n]=dict(passed=v.shape==r.shape and bool(np.isfinite(v).all()) and bool(np.allclose(v,r,atol=1e-5,rtol=1e-5)),max_abs=float(np.max(np.abs(delta))),normalized_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(r.astype(np.float64)),1e-30)))
     np.savez(seedout/f'{name}-{repeat}.npz',**outputs)
     samples.append(dict(name=name,repeat=repeat,warmup=repeat<2,wall_ms=wall,checks=checks,api_events=[{**e,'start_ns':e['start_ns']-start,'end_ns':e['end_ns']-start} for e in events],actual_submissions=sum(e['kind']=='actual_xrt_start' for e in events)))
     save(seedout/'samples-in-progress.json',dict(seed=seed,samples=samples))
   record=dict(seed=seed,input_sha256=digest(ip),reference_sha256=digest(rp),load_ms=load,samples=samples,medians_ms={c['name']:float(np.median([s['wall_ms'] for s in samples if s['name']==c['name'] and not s['warmup']])) for c in cases})
   save(seedout/'result.json',record);(seedout/'samples-in-progress.json').unlink(missing_ok=True);all_results.append(record);runners.clear();del runner
finally:multilaunch._wait=original_wait
save(out/'results.json',dict(status='passed' if all(v['passed'] for r in all_results for s in r['samples'] for v in s['checks'].values()) else 'numeric_failed',results=all_results,boundary='prepared host f32 arrays through staging/sync/submit/wait/readback/output-copy; persistent context and BOs for every variant; alternating forward/reverse order; two warmups then hot samples; unlocked frequency'))
raise SystemExit(0 if all(v['passed'] for r in all_results for s in r['samples'] for v in s['checks'].values()) else 1)
