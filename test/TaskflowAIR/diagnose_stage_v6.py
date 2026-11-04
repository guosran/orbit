#!/usr/bin/env python3
"""Partial-stage diagnostic of an existing ELF, never whole-graph acceptance."""
import argparse,fcntl,json,shutil,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from taskflow_direct_air_execute import digest,save
from taskflow_npu_trace_v3 import ObservedXRT,ObservedRun
from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner
from triton.backends.amd_triton_npu import multilaunch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--hardware',type=Path,required=True);p.add_argument('--kernel',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--rows',type=int,nargs=2,required=True);p.add_argument('--cols',type=int,nargs=2,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--buffer');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,a.output/'diagnostic-source.py');artifact=json.loads((a.hardware/'compile.json').read_text());assert digest(artifact['elf_path'])==artifact['elf_sha256'];abi=json.loads((a.hardware/'abi.json').read_text());inputs=dict(np.load(a.inputs));ref=dict(np.load(a.reference));arrays=[np.ascontiguousarray(inputs.get(b['name'],np.zeros(b['shape'],dtype=np.float32)),dtype=np.float32) for b in abi['buffers']];indices={i for i,b in enumerate(abi['buffers']) if b['name'] in ([a.buffer] if a.buffer else abi['outputs'])};assert indices;events=[]
save(a.output/'invocation.json',dict(scope='PARTIAL DIAGNOSTIC ONLY; not whole graph correctness',hardware=str(a.hardware),elf_sha256=artifact['elf_sha256'],kernel=a.kernel,rows=a.rows,cols=a.cols,input_sha256=digest(a.inputs),reference_sha256=digest(a.reference)))
original_wait=multilaunch._wait
try:
 with open('/tmp/npu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);runner=MultiLaunchRunner(artifact['elf_path'],a.kernel);runner._xrt=ObservedXRT(runner._xrt,events)
  def wait(run):
   start=time.perf_counter_ns();save(a.output/'wait-in-progress.json',dict(start_ns=start,events=list(events)))
   try:return original_wait(run.raw if isinstance(run,ObservedRun) else run)
   finally:events.append(dict(kind='actual_xrt_wait',start_ns=start,end_ns=time.perf_counter_ns()));save(a.output/'events.json',events)
  multilaunch._wait=wait;start=time.perf_counter_ns();values=runner.run(arrays,bo_key='diagnostic',output_indices=indices);wall=(time.perf_counter_ns()-start)/1e6
  outputs={abi['buffers'][i]['name']:values[i].copy() for i in indices};np.savez(a.output/'outputs.npz',**outputs);checks={}
  for name,x in outputs.items():
   actual=x[a.rows[0]:a.rows[1],a.cols[0]:a.cols[1]].astype(float);expected=ref[name][a.rows[0]:a.rows[1],a.cols[0]:a.cols[1]].astype(float);checks[name]=dict(max_abs=float(np.max(np.abs(actual-expected))),passed=bool(np.isfinite(actual).all() and np.allclose(actual,expected,atol=1e-5,rtol=1e-5)),shape=list(actual.shape))
  save(a.output/'result.json',dict(scope='PARTIAL DIAGNOSTIC ONLY',wall_ms=wall,checks=checks));runner.unload()
except Exception as error:
 save(a.output/'failure.json',dict(scope='PARTIAL DIAGNOSTIC ONLY',type=type(error).__name__,message=str(error)));raise
finally:multilaunch._wait=original_wait
