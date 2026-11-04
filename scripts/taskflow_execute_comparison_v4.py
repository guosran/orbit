#!/usr/bin/env python3
"""Interleave checked native single/pair exact-cover plans using existing NPUChain."""
import argparse,hashlib,importlib.util,json,shutil,statistics,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from filelock import FileLock
from taskflow_npu_chain_v4 import ROOT,RunProxy,XrtProxy,check,digest,snapshot
from taskflow_common import make_inputs,write_json

def load_case(name,source,planpath,configpath,out):
 core=ROOT/'build-core/tools/taskflow-opt/taskflow-opt';d=out/name;d.mkdir()
 checked=subprocess.run([str(core),str(source),'--target-config='+str(configpath),'--check-mapping='+str(planpath)],capture_output=True,text=True);(d/'native-check.log').write_text(checked.stdout+checked.stderr)
 if checked.returncode:raise ValueError('native checker rejected '+name)
 exported=subprocess.run([str(core),str(source),'--emit-plan='+str(d/'program.json')],capture_output=True,text=True)
 if exported.returncode:raise ValueError('actual MLIR export failed '+name)
 program=json.loads((d/'program.json').read_text());plan=json.loads(planpath.read_text());config=json.loads(configpath.read_text())
 for file,label in [(source,'taskflow.mlir'),(planpath,'selected-plan.json'),(configpath,'config.json')]:shutil.copyfile(file,d/label)
 if program['outputs']!=['Y'] or {b['id'] for b in program['buffers']}!={'X','W1','H','W2','Y'}:raise ValueError('unsupported equations')
 implementations={i['id']:i for i in config['region_implementations']};steps={f"{t['id']}:{j}":op for t in program['tasks'] for j,op in enumerate(t['operations'])};regions=[];done=set()
 for e in sorted(plan['events'],key=lambda e:(e['start_ms'],e['id'])):
  if e['kind']!='region' or (e['region_pattern'],e['region_mode']) not in [('matmul_single','independent'),('matmul_pair','chain')]:raise ValueError('unsupported region implementation')
  if not set(e['depends_on'])<=done:raise ValueError('schedule order violates dependencies')
  done.add(e['id']);impl=implementations[e['implementation_id']]
  for key,value in dict(layout='row_major',dtype='f32',numerical_contract='bf16-bfp16-bf16-mid-final',prepared_state='warm_contexts_weights',boundary_memory='host',device='npu').items():
   if impl.get(key)!=value or e['prepared_state']!='warm_contexts_weights':raise ValueError('unsupported target contract')
  ops=[steps[s] for s in e['covered_steps']];muls=[op for op in ops if op['kind']=='matmul'];count=2 if e['region_pattern']=='matmul_pair' else 1
  if len(muls)!=count or len(ops)!=count*2 or sum(op['kind']=='fill' for op in ops)!=count:raise ValueError('expected full-K GEMMs with zero fills')
  muls.sort(key=lambda op:0 if op['inputs'][0]=='X' else 1)
  for op in muls:
   if (op['inputs'],op['output']) not in [(['X','W1'],'H'),(['H','W2'],'Y')]:raise ValueError('unsupported equations')
   m=op['m'][1]-op['m'][0];n=op['n'][1]-op['n'][0];k=op['k'][1]-op['k'][0]
   if op['k'][0]!=0 or m%128 or n%256 or k&(k-1):raise ValueError('unsupported physical shape')
  if count==2:
   first,second=muls
   if first['m']!=second['m'] or first['n']!=second['k'] or first['n'][0]!=0 or second['n'][0]!=0:raise ValueError('unsupported pair slicing')
  # Shape+weight slice form immutable prepared object identity; row offset is dynamic input.
  key=(e['region_mode'],tuple((op['m'][1]-op['m'][0],op['k'][1],op['n'][1]-op['n'][0],op['inputs'][1],op['n'][0]) for op in muls))
  regions.append(dict(event=e,operations=muls,key=key))
 return dict(name=name,program=program,plan=plan,regions=regions,samples=[],excluded=[],source_sha256=digest(source),config_sha256=digest(configpath),plan_sha256=digest(planpath))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--iterations',type=int,default=9);a=p.parse_args();out=a.output_dir.resolve()
 if out.exists() or a.iterations<7:raise ValueError('new directory and at least seven hot samples required')
 out.mkdir(parents=True);declaration=json.loads(a.cases.read_text());shutil.copyfile(a.cases,out/'cases.json')
 cases=[load_case(c['name'],Path(c['source']),Path(c['plan']),Path(c['config']),out) for c in declaration['cases']]
 inputs=make_inputs(cases[0]['program'],0)
 for case in cases:
  other=make_inputs(case['program'],0)
  if set(other)!=set(inputs) or any(not np.array_equal(other[k],inputs[k]) for k in inputs):raise ValueError('comparison inputs differ')
  if case['program']['buffers']!=cases[0]['program']['buffers']:raise ValueError('comparison logical shapes differ')
 np.savez(out/'inputs-f32.npz',**inputs);ref=(inputs['X'].astype('f8')@inputs['W1'].astype('f8'))@inputs['W2'].astype('f8');np.save(out/'reference-f64.npy',ref)
 import torch
 from ml_dtypes import bfloat16
 from triton.backends.amd_triton_npu.multilaunch import NPUChain
 import triton.backends.amd_triton_npu.multilaunch as multilaunch
 from triton.backends.amd_triton_npu.config import npu_config
 from triton.backends.amd_triton_npu.driver_scope import driver_scope
 from orbit_npu.runtime import RECIPES
 kernel=RECIPES.parent/'matmul_bf16_epilogue/matmul_bf16_epilogue.py';sys.path.insert(0,str(RECIPES.parent));spec=importlib.util.spec_from_file_location('orbit_comparison_epilogue',kernel);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 schedule=kernel.parent/'transform_aie2p.mlir';chains={};weights={};events=[];compiled={};oldwait=multilaunch._wait
 result=dict(schema='orbit.plan-comparison.v1',status='compile_failed',prepared_state='warm_contexts_weights',boundary='ordinary f32 host activation to f32 host result including all temporary allocation, slices, conversions, region assembly, submissions, waits and final copy',precision='BF16 inputs/BFP16 EBS8 MAC/FP32 accumulator/BF16 result then f32 host boundary; same existing kernel for every region',frequency_locked=False,implementation_sha256={str(f):digest(f) for f in [Path(__file__),kernel,schedule,Path(multilaunch.__file__)]},cases={})
 for f in [Path(__file__),kernel,schedule]:shutil.copyfile(f,out/f.name)
 npu_config.target='npu2';npu_config.runtime='xrt';npu_config.bf16_emulation=False
 try:
  with driver_scope('npu'),torch.inference_mode():
   for case in cases:
    for region in case['regions']:
     key=region['key']
     if key in chains:continue
     index=len(chains);d=out/f'kernel-{index}';d.mkdir();c=NPUChain('orbit_interleaved_v4_'+str(index),air_project_path=str(d/'air-build'));chains[key]=c;weights[key]=[]
     for j,(m,k,n,weightname,n0) in enumerate(key[1]):
      tensors=[torch.zeros((m,k),dtype=torch.bfloat16),torch.zeros((k,n),dtype=torch.bfloat16),torch.zeros((m,n),dtype=torch.bfloat16)]
      mapping={0:0,1:1,2:2} if j==0 else {0:2,1:3,2:4}
      c.add(module.matmul_epilogue,grid=(m//128,n//256),arg_map=mapping,args=(*tensors,m,n,k,k,1,n,1,n,1),constexprs=dict(SCALE=1.0,EPILOGUE='scale_bf16',BLOCK_SIZE_M=128,BLOCK_SIZE_N=256),transform_script=str(schedule))
      weights[key].append(np.ascontiguousarray(inputs[weightname][:k,n0:n0+n],dtype=bfloat16))
     t=time.perf_counter();c._build();compiled[str(key)]=dict(seconds=time.perf_counter()-t,elf=c._elf_path,elf_sha256=digest(c._elf_path));(d/'combined-air.mlir').write_text(c._builder.build_module_text());shutil.copytree(Path(c._elf_path).parent,d/'compiled-cache',dirs_exist_ok=True)
   buffer_keys={key:'weights-'+hashlib.sha256(b''.join(w.tobytes() for w in ws)).hexdigest() for key,ws in weights.items()}
   result['prepared_weights']=[dict(kernel_key=key,bo_key=buffer_keys[key],weights=[dict(object_id=id(w),shape=list(w.shape),dtype=str(w.dtype),sha256=hashlib.sha256(w.tobytes()).hexdigest()) for w in ws]) for key,ws in weights.items()]
   result['weight_contract']='Prepared weights immutable during all timed executions; explicit content-generation BO key separates changed prepared weight sets. Runtime itself does not detect mutation.'
   def wait(run):
    t=time.perf_counter_ns();value=oldwait(run.run if isinstance(run,RunProxy) else run);events.append(dict(kind='actual_xrt_wait',start_ns=t,end_ns=time.perf_counter_ns()));return value
   with FileLock('/tmp/npu.lock',timeout=1800):
    multilaunch._wait=wait;result['status']='runtime_failed'
    for key,c in chains.items():
     t=time.perf_counter();c._open_runner();compiled[str(key)]['context_load_seconds']=time.perf_counter()-t;c._runner._xrt=XrtProxy(c._runner._xrt,events)
    def execute(case):
     state=dict(inputs)
     for b in case['program']['buffers']:
      if b['role']!='input':state[b['id']]=np.empty(b['shape'],dtype='f4')
     for region in case['regions']:
      key=region['key'];ops=region['operations'];first=ops[0];m0,m1=first['m'];k=first['k'][1];n=first['n'][1]-first['n'][0]
      activation=np.ascontiguousarray(state[first['inputs'][0]][m0:m1,:k],dtype=bfloat16);mid=np.empty((m1-m0,n),dtype=bfloat16)
      if len(ops)==1:
       y=chains[key].run([activation,weights[key][0],mid],bo_key=buffer_keys[key],static_indices={1},intermediate_indices={2},output_indices={2})[2]
      else:
       last=ops[1];final=np.empty((m1-m0,last['n'][1]-last['n'][0]),dtype=bfloat16)
       y=chains[key].run([activation,weights[key][0],mid,weights[key][1],final],bo_key=buffer_keys[key],static_indices={1,3},intermediate_indices={2,4},output_indices={4})[4]
      events.append(dict(kind='region_execution',region_id=region['event']['id'],runner_id=id(chains[key]._runner),bo_key=buffer_keys[key],weight_object_ids=[id(w) for w in weights[key]],readback_indices=[2] if len(ops)==1 else [4]))
      last=ops[-1];state[last['output']][last['m'][0]:last['m'][1],last['n'][0]:last['n'][1]]=y.astype('f4')
     return state['Y'].copy()
    for iteration in range(a.iterations+2):
     order=cases if iteration%2==0 else list(reversed(cases))
     for position,case in enumerate(order):
      events.clear();t=time.perf_counter();y=execute(case);wall=(time.perf_counter()-t)*1000
      sample=dict(iteration=iteration-2,order_position=position,wall_ms=wall,check=check(y,ref),events=list(events),actual_submissions=sum(e['kind']=='actual_xrt_start' for e in events),actual_waits=sum(e['kind']=='actual_xrt_wait' for e in events),runtime_objects=[snapshot(chains[r['key']]) for r in case['regions']])
      if iteration<2:case['excluded'].append(sample)
      else:np.save(out/case['name']/f'output-{iteration-2}.npy',y);case['samples'].append(sample)
    result['runtime_objects']=[snapshot(c) for c in chains.values()]
    # Explicit generation keys test changed prepared weights on the SAME runner.
    base=cases[0];region=base['regions'][0];key=region['key']
    if len(base['regions'])==1 and len(region['operations'])==2:
     wa=weights[key];wb=[-wa[0],wa[1]];result['weight_generation_probe']=[]
     for label,ws,reference in [('A',wa,ref),('B',wb,-ref),('A',wa,ref)]:
      generation='weights-'+hashlib.sha256(b''.join(w.tobytes() for w in ws)).hexdigest()
      activation=inputs['X'].astype(bfloat16);mid=np.empty((activation.shape[0],ws[0].shape[1]),dtype=bfloat16);final=np.empty((activation.shape[0],ws[1].shape[1]),dtype=bfloat16)
      events.clear();y=chains[key].run([activation,ws[0],mid,ws[1],final],bo_key=generation,static_indices={1,3},intermediate_indices={2,4},output_indices={4})[4].astype('f4')
      pi=len(result['weight_generation_probe']);np.save(out/f'weight-probe-{pi}-{label}.npy',y)
      result['weight_generation_probe'].append(dict(label=label,bo_key=generation,runner_id=id(chains[key]._runner),weight_object_ids=[id(w) for w in ws],weight_sha256=[hashlib.sha256(w.tobytes()).hexdigest() for w in ws],check=check(y,reference),events=list(events),runtime=snapshot(chains[key])))
   for case in cases:
    name=case['name'];median=statistics.median(x['wall_ms'] for x in case['samples']);prediction=case['plan']['totals']['makespan_ms']
    result['cases'][name]=dict(plan_id=case['plan']['plan_id'],candidate_id=case['plan']['candidate_id'],plan_sha256=case['plan_sha256'],source_sha256=case['source_sha256'],config_sha256=case['config_sha256'],regions=case['regions'],samples=case['samples'],excluded=case['excluded'],median_ms=median,prediction_ms=prediction,relative_prediction_error=(prediction-median)/median)
   result['status']='passed' if all(s['check']['passed'] and s['actual_submissions']==len(c['regions']) and s['actual_waits']==len(c['regions']) for c in cases for s in c['samples']+c['excluded']) else 'numerical_failed'
   if any(not x['check']['passed'] for x in result.get('weight_generation_probe',[])):result['status']='weight_generation_failed'
   base=cases[0]['name'];result['baseline']=base
   for name,c in result['cases'].items():c['ratio_to_baseline']=c['median_ms']/result['cases'][base]['median_ms']
 except Exception as exc:result.update(error=str(exc),traceback=traceback.format_exc())
 finally:
  multilaunch._wait=oldwait
  for c in chains.values():c.close()
  result['compilation']=compiled;write_json(out/'result.json',result)
 print(json.dumps(dict(status=result['status'],error=result.get('error'),cases={n:{k:v[k] for k in ['median_ms','prediction_ms','relative_prediction_error','ratio_to_baseline']} for n,v in result['cases'].items()}),indent=2));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
