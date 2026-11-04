#!/usr/bin/env python3
"""Execute native exact-cover single-GEMM regions, including M partitions.

All graph allocation, slicing, activation conversion, host assembly, dispatch,
synchronization and final output copying are inside the hot graph wall.
"""
import argparse,importlib.util,json,shutil,statistics,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from filelock import FileLock
from taskflow_npu_chain_v4 import ROOT,RunProxy,XrtProxy,check,digest,snapshot
from taskflow_common import make_inputs,write_json

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ['source','plan','config','output-dir']:p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--iterations',type=int,default=7);a=p.parse_args();out=a.output_dir.resolve()
 if out.exists() or a.iterations<7:raise ValueError('new output directory and >=7 iterations required')
 out.mkdir(parents=True);core=ROOT/'build-core/tools/taskflow-opt/taskflow-opt'
 cmd=[str(core),str(a.source),'--target-config='+str(a.config),'--check-mapping='+str(a.plan)]
 r=subprocess.run(cmd,capture_output=True,text=True);(out/'native-check.log').write_text(r.stdout+r.stderr)
 if r.returncode:raise ValueError('native checker rejected plan')
 r=subprocess.run([str(core),str(a.source),'--emit-plan='+str(out/'program.json')],capture_output=True,text=True)
 if r.returncode:raise ValueError('actual source export failed')
 program=json.loads((out/'program.json').read_text());plan=json.loads(a.plan.read_text());config=json.loads(a.config.read_text())
 for path in [a.source,a.plan,a.config]:shutil.copyfile(path,out/path.name)
 implementations={i['id']:i for i in config['region_implementations']};steps={f"{t['id']}:{j}":op for t in program['tasks'] for j,op in enumerate(t['operations'])}
 if program['outputs']!=['Y'] or {b['id'] for b in program['buffers']}!={'X','W1','H','W2','Y'}:raise ValueError('unsupported source')
 events=sorted(plan['events'],key=lambda e:(e['start_ms'],e['id']));done=set();regions=[]
 for e in events:
  if e['kind']!='region' or e['region_pattern']!='matmul_single' or e['region_mode']!='independent':raise ValueError('only singleton exact cover supported')
  if not set(e['depends_on'])<=done:raise ValueError('non-topological execution order')
  done.add(e['id']);impl=implementations[e['implementation_id']]
  for key,value in dict(layout='row_major',dtype='f32',numerical_contract='bf16-bfp16-bf16-mid-final',prepared_state='warm_contexts_weights',boundary_memory='host',device='npu').items():
   if impl.get(key)!=value:raise ValueError('unsupported implementation '+key)
  ops=[steps[k] for k in e['covered_steps']];muls=[op for op in ops if op['kind']=='matmul']
  if len(ops)!=2 or len(muls)!=1 or sum(op['kind']=='fill' for op in ops)!=1:raise ValueError('fill/full-K matmul required')
  op=muls[0]
  if (op['inputs'],op['output']) not in [(['X','W1'],'H'),(['H','W2'],'Y')]:raise ValueError('unsupported equations')
  regions.append((e,op))
 inputs=make_inputs(program,0);np.savez(out/'inputs-f32.npz',**inputs)
 ref=(inputs['X'].astype('f8')@inputs['W1'].astype('f8'))@inputs['W2'].astype('f8');np.save(out/'reference-f64.npy',ref)
 import torch
 from ml_dtypes import bfloat16
 from triton.backends.amd_triton_npu.multilaunch import NPUChain
 import triton.backends.amd_triton_npu.multilaunch as multilaunch
 from triton.backends.amd_triton_npu.config import npu_config
 from triton.backends.amd_triton_npu.driver_scope import driver_scope
 from orbit_npu.runtime import RECIPES
 kernel_source=RECIPES.parent/'matmul_bf16_epilogue/matmul_bf16_epilogue.py';sys.path.insert(0,str(RECIPES.parent))
 spec=importlib.util.spec_from_file_location('orbit_structural_epilogue',kernel_source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 schedule=kernel_source.parent/'transform_aie2p.mlir';chains={};observed=[];compiled={};oldwait=multilaunch._wait
 result=dict(candidate_id=plan['candidate_id'],implementation_sha256={str(f):digest(f) for f in (Path(__file__),kernel_source,schedule,Path(multilaunch.__file__))},precision='BF16 inputs, BFP16 EBS8 MAC, FP32 accumulator, BF16 output per region then f32 boundary',schema='orbit.structural.execution.v1',status='compile_failed',plan_id=plan['plan_id'],plan_sha256=digest(a.plan),source_sha256=digest(a.source),config_sha256=digest(a.config),regions=[dict(event=e,operation=op) for e,op in regions],samples=[],excluded=[],prepared_state='warm_contexts_weights',boundary='f32 host inputs to f32 host outputs; all temporary allocation, activation slicing/conversion, region assembly, synchronization and output copy timed',prediction_ms=plan['totals']['makespan_ms'])
 for f in (Path(__file__),kernel_source,schedule):shutil.copyfile(f,out/f.name)
 npu_config.target='npu2';npu_config.runtime='xrt';npu_config.bf16_emulation=False
 try:
  # Compilation is CPU-only; hardware lock below covers context creation and all calls.
  with driver_scope('npu'),torch.inference_mode():
   for e,op in regions:
    m=op['m'][1]-op['m'][0];n=op['n'][1]-op['n'][0];k=op['k'][1]-op['k'][0]
    if op['k'][0]!=0 or m%128 or n%256 or k&(k-1):raise ValueError('unsupported kernel regions')
    key=(m,k,n,op['inputs'][1],op['n'][0]);
    if key in chains:continue
    d=out/('kernel-'+'-'.join(map(str,key)));d.mkdir();c=NPUChain('orbit_structural_'+'_'.join(map(str,key)),air_project_path=str(d/'air-build'));chains[key]=c
    tensors=[torch.zeros((m,k),dtype=torch.bfloat16),torch.zeros((k,n),dtype=torch.bfloat16),torch.zeros((m,n),dtype=torch.bfloat16)]
    c.add(module.matmul_epilogue,grid=(m//128,n//256),arg_map={0:0,1:1,2:2},args=(*tensors,m,n,k,k,1,n,1,n,1),constexprs=dict(SCALE=1.0,EPILOGUE='scale_bf16',BLOCK_SIZE_M=128,BLOCK_SIZE_N=256),transform_script=str(schedule))
    t=time.perf_counter();c._build();compiled[str(key)]=dict(seconds=time.perf_counter()-t,elf=c._elf_path,elf_sha256=digest(c._elf_path));(d/'combined-air.mlir').write_text(c._builder.build_module_text());shutil.copytree(Path(c._elf_path).parent,d/'compiled-cache',dirs_exist_ok=True)
   weights={key:np.ascontiguousarray(inputs[key[3]][:key[1],key[4]:key[4]+key[2]],dtype=bfloat16) for key in chains}
   def wait(run):
    t=time.perf_counter_ns();r=oldwait(run.run if isinstance(run,RunProxy) else run);observed.append(dict(kind='actual_xrt_wait',start_ns=t,end_ns=time.perf_counter_ns()));return r
   with FileLock('/tmp/npu.lock',timeout=1800):
    multilaunch._wait=wait;result['status']='runtime_failed'
    for key,c in chains.items():c._open_runner();c._runner._xrt=XrtProxy(c._runner._xrt,observed)
    def execute():
     state=dict(inputs)
     for b in program['buffers']:
      if b['role']!='input':state[b['id']]=np.empty(b['shape'],dtype='f4')
     for e,op in regions:
      m0,m1=op['m'];n0,n1=op['n'];k0,k1=op['k'];key=(m1-m0,k1-k0,n1-n0,op['inputs'][1],n0);c=chains[key]
      activation=np.ascontiguousarray(state[op['inputs'][0]][m0:m1,k0:k1],dtype=bfloat16)
      weight=weights[key];output=np.empty((m1-m0,n1-n0),dtype=bfloat16)
      y=c.run([activation,weight,output],bo_key='region',static_indices={1},intermediate_indices={2},output_indices={2})[2]
      state[op['output']][m0:m1,n0:n1]=y.astype('f4')
     return state['Y'].copy()
    for i in range(a.iterations+2):
     observed.clear();t=time.perf_counter();y=execute();wall=(time.perf_counter()-t)*1000;entry=dict(iteration=i-2,wall_ms=wall,check=check(y,ref),events=list(observed),actual_submissions=sum(x['kind']=='actual_xrt_start' for x in observed),actual_waits=sum(x['kind']=='actual_xrt_wait' for x in observed))
     if i<2:result['excluded'].append(entry)
     else:np.save(out/f'output-{i-2}.npy',y);result['samples'].append(entry)
    result['runtime_objects']=[snapshot(c) for c in chains.values()]
   result['status']='passed' if all(s['check']['passed'] and s['actual_submissions']==len(regions) for s in result['samples']+result['excluded']) else 'numerical_failed'
   result['median_ms']=statistics.median(s['wall_ms'] for s in result['samples']);result['relative_prediction_error']=(result['prediction_ms']-result['median_ms'])/result['median_ms']
 except Exception as exc:result.update(error=str(exc),traceback=traceback.format_exc())
 finally:
  multilaunch._wait=oldwait
  for c in chains.values():c.close()
  result['compilation']=compiled;write_json(out/'result.json',result)
 print(json.dumps({k:result.get(k) for k in ['status','error','plan_id','prediction_ms','median_ms','relative_prediction_error']},indent=2));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
