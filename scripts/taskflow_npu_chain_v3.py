#!/usr/bin/env python3
"""Existing NPUChain two-GEMM comparison; owns /tmp/npu.lock for execution.

No new runtime or kernel. Both targets reuse the upstream epilogue matmul recipe,
with BF16 intermediate output and identical physical tiling and numerical path.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.util, json, shutil, statistics, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from taskflow_common import fingerprint, make_inputs, write_json, compare

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source(m,k,h,n):
    shapes={'X':[m,k],'W1':[k,h],'H':[m,h],'W2':[h,n],'Y':[m,n]}
    p=dict(schema='orbit.taskflow.plan.v1',buffers=[dict(id=b,shape=s,dtype='f32',role='input' if b in ('X','W1','W2') else 'output' if b=='Y' else 'temporary') for b,s in shapes.items()],tasks=[],outputs=['Y'],transforms=[])
    for name,a,b,c,kk,nn,deps in [('first','X','W1','H',k,h,[]),('second','H','W2','Y',h,n,['first'])]:
        p['tasks'].append(dict(id=name,depends_on=deps,operations=[dict(kind='fill',inputs=[],output=c,m=[0,m],n=[0,nn]),dict(kind='matmul',inputs=[a,b],output=c,m=[0,m],n=[0,nn],k=[0,kk],accumulate=True)]))
    p['source_id']=fingerprint(p);p['candidate_id']=p['source_id'];return p

class RunProxy:
    def __init__(self,run,events): self.run=run;self.events=events
    def __getattr__(self,key): return getattr(self.run,key)
    def start(self,*a,**kw):
        t=time.perf_counter_ns(); result=self.run.start(*a,**kw)
        self.events.append(dict(kind='actual_xrt_start',start_ns=t,end_ns=time.perf_counter_ns()));return result
    def wait(self,*a,**kw):
        t=time.perf_counter_ns();result=self.run.wait(*a,**kw)
        self.events.append(dict(kind='actual_xrt_wait',start_ns=t,end_ns=time.perf_counter_ns()));return result
class XrtProxy:
    def __init__(self,xrt,events): self.xrt=xrt;self.events=events
    def __getattr__(self,key): return getattr(self.xrt,key)
    def run(self,*a,**kw): return RunProxy(self.xrt.run(*a,**kw),self.events)

def check(y,ref):
    c=compare({'Y':y},{'Y':ref},.015,.06)
    error=y.astype('f8')-ref
    rms=float(np.sqrt(np.mean(error**2))/max(np.sqrt(np.mean(ref**2)),1e-30))
    peak=float(np.max(abs(error))/max(np.max(abs(ref)),1e-30))
    c.update(normalized_rms=rms,peak_scaled=peak)
    c['passed']=c['passed'] and rms<=.03 and peak<=.08
    return c

def snapshot(chain):
    r=chain._runner
    return dict(chain_id=id(chain),runner_id=id(r),kernel_id=id(r.kernel),context_id=id(r.context),elf_object_id=id(r.elf),elf=chain._elf_path,
        bo_sets={str(k):[id(bo) for bo in values[1]] for k,values in r._bos.items()})

def experiment(case,out,iterations,compile_only):
    import torch
    from ml_dtypes import bfloat16
    from triton.backends.amd_triton_npu.multilaunch import NPUChain
    from triton.backends.amd_triton_npu.config import npu_config
    from triton.backends.amd_triton_npu.driver_scope import driver_scope
    from orbit_npu.runtime import RECIPES
    m,k,h,n=(128,256,256,256) if case=='train' else (128,256,512,256)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): raise ValueError('refusing existing evidence '+str(out))
    result=dict(schema='orbit.chain.experiment.v1',case=case,split='calibration' if case=='train' else 'held_out',shape=dict(m=m,k=k,h=h,n=n),status='compile_failed',variants={},trace_enabled=False)
    chains=[]; original_wait=None
    try:
        p=source(m,k,h,n);write_json(out/'source.json',p)
        cmd=[str(ROOT/'build-core/tools/taskflow-opt/taskflow-opt'),str(out/'source.json'),'--emit-plan='+str(out/'plan.json'),'--emit-mlir='+str(out/'taskflow.mlir')]
        completed=subprocess.run(cmd,capture_output=True,text=True);(out/'materialize.log').write_text(completed.stdout+completed.stderr)
        if completed.returncode: raise ValueError('materialization failed')
        inputs=make_inputs(p,0);np.savez(out/'inputs-f32.npz',**inputs)
        ref=(inputs['X'].astype('f8')@inputs['W1'].astype('f8'))@inputs['W2'].astype('f8')
        np.save(out/'reference-f64.npy',ref)
        preparation_start=time.perf_counter()
        arrays=[inputs['X'].astype(bfloat16),inputs['W1'].astype(bfloat16),np.zeros((m,h),dtype=bfloat16),inputs['W2'].astype(bfloat16),np.zeros((m,n),dtype=bfloat16)]
        np.savez(out/'inputs-bf16.npz',X=arrays[0].astype('f4'),W1=arrays[1].astype('f4'),W2=arrays[3].astype('f4'))
        tensors=[torch.from_numpy(inputs['X']).bfloat16(),torch.from_numpy(inputs['W1']).bfloat16(),torch.zeros((m,h),dtype=torch.bfloat16),torch.from_numpy(inputs['W2']).bfloat16(),torch.zeros((m,n),dtype=torch.bfloat16)]
        examples=RECIPES.parent
        sys.path.insert(0,str(examples))
        kernel_source=examples/'matmul_bf16_epilogue/matmul_bf16_epilogue.py'
        spec=importlib.util.spec_from_file_location('orbit_existing_epilogue',kernel_source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        matmul=module.matmul_epilogue
        schedule=kernel_source.parent/'transform_aie2p.mlir'
        result['host_input_preparation_seconds']=time.perf_counter()-preparation_start
        import triton.backends.amd_triton_npu.multilaunch as multilaunch
        import triton.backends.amd_triton_npu.fused_program as fused_program
        files=[Path(__file__),kernel_source,Path(multilaunch.__file__),Path(fused_program.__file__),schedule]
        result['implementation_sha256']={str(f):digest(f) for f in files}
        for f in files: shutil.copyfile(f,out/f.name)
        result['input_identity']={str(f.name):digest(f) for f in (out/'inputs-f32.npz',out/'inputs-bf16.npz',out/'reference-f64.npy')}
        result['precision']='BF16 inputs, BFP16 EBS8 MAC, FP32 accumulators; both GEMMs stored BF16; final BF16-to-FP32 representation conversion included in graph wall; identical both variants'
        result['comparison_boundary']='Two reused single-op NPUChain contexts and persistent weights plus BF16 intermediate readback/staging versus one existing two-op NPUChain; improvement cannot be attributed solely to launch count'
        npu_config.target='npu2';npu_config.runtime='xrt';npu_config.bf16_emulation=False
        def add(chain,index,mapping):
            aa,bb,cc,kk,nn=(0,1,2,k,h) if index==0 else (2,3,4,h,n)
            chain.add(matmul,grid=(m//128,nn//256),arg_map=mapping,args=(tensors[aa],tensors[bb],tensors[cc],m,nn,kk,kk,1,nn,1,nn,1),constexprs=dict(SCALE=1.0,EPILOGUE='scale_bf16',BLOCK_SIZE_M=128,BLOCK_SIZE_N=256),transform_script=str(schedule))
        with driver_scope('npu'),torch.inference_mode():
            for name,indexes in [('first',[0]),('second',[1]),('chain',[0,1])]:
                d=out/name;d.mkdir(exist_ok=True)
                c=NPUChain('orbit_static_v3_'+case+'_'+name,air_project_path=str(d/'air-build'))
                chains.append(c)
                for i in indexes: add(c,i,{0:0,1:1,2:2} if name!='chain' else ({0:0,1:1,2:2} if i==0 else {0:2,1:3,2:4}))
                for i,spec in enumerate(c._specs):
                    kernel,grid,mapping,ts,args,constants,actual=spec
                    ir=c._capture_ttshared(kernel,grid,args,constants)
                    (d/f'op-{i}-ttshared.mlir').write_text(ir.decode() if isinstance(ir,bytes) else ir)
                t=time.perf_counter();c._build();seconds=time.perf_counter()-t
                (d/'combined-air.mlir').write_text(c._builder.build_module_text())
                shutil.copytree(Path(c._elf_path).parent,d/'compiled-cache',dirs_exist_ok=True)
                result.setdefault('compilation',{})[name]=dict(seconds=seconds,elf=str(c._elf_path),elf_sha256=digest(c._elf_path))
            target=dict(schema='orbit.chain.target.v1',public_plan=str(out/'plan.json'),physical_grid_tile=[128,256],kernel_recipe='existing matmul_bf16_epilogue SCALE=1; full K per Triton program, schedule physical K tiling unchanged',placement='both matmuls NPU; semantic zero initialization absorbed into each existing GEMM kernel',final_device_dtype='bf16',final_host_dtype='f32',final_host_conversion_timed=True,intermediate=dict(dtype='bf16',shape=[m,h],memory='NPU-accessible system DRAM, not SRAM'),variants=dict(independent=dict(objects=['first','second'],arg_maps=[{0:0,1:1,2:2},{0:0,1:1,2:2}],host_intermediate=True),chain=dict(objects=['chain'],arg_maps=[{0:0,1:1,2:2},{0:2,1:3,2:4}],host_intermediate=False)),combined_buffers=[dict(index=i,name=name,shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes) for i,(name,a) in enumerate(zip(['X','W1','H','W2','Y'],arrays))])
            write_json(out/'target-plan.json',target)
            if compile_only:
                result['status']='compiled';return result
            result['status']='runtime_failed'
            events=[]
            original_wait=multilaunch._wait
            def observed_wait(run):
                raw=run.run if isinstance(run,RunProxy) else run
                t=time.perf_counter_ns(); value=original_wait(raw)
                events.append(dict(kind='actual_xrt_wait',start_ns=t,end_ns=time.perf_counter_ns()))
                return value
            multilaunch._wait=observed_wait
            result['instrumentation']='Local Python observation of real run.start and existing _wait; unwraps proxy before original _wait, retaining native npu_wait selection. Host observation overhead included equally. No installed package edits.'
            for name,c in zip(['first','second','chain'],chains):
                t=time.perf_counter();c._open_runner();result.setdefault('load_seconds',{})[name]=time.perf_counter()-t
                c._runner._xrt=XrtProxy(c._runner._xrt,events)
            first,second,chain=chains
            def execute(variant):
                if variant=='independent':
                    mid=first.run(arrays[:3],bo_key='first',static_indices={1},intermediate_indices={2},output_indices={2})[2]
                    y=second.run([mid,arrays[3],arrays[4]],bo_key='second',static_indices={1},intermediate_indices={2},output_indices={2})[2]
                else:
                    y=chain.run(arrays,bo_key='chain',static_indices={1,3},intermediate_indices={2,4},output_indices={4})[4]
                return y.astype(np.float32,copy=True)
            for variant in ('independent','chain'):
                result['variants'][variant]=dict(samples=[],excluded=[],submissions_per_graph=2 if variant=='independent' else 1,readbacks_per_graph=2 if variant=='independent' else 1)
                for stage in ('first','warmup'):
                    events.clear();t=time.perf_counter();y=execute(variant);elapsed=(time.perf_counter()-t)*1000
                    result['variants'][variant]['excluded'].append(dict(stage=stage,wall_ms=elapsed,check=check(y,ref),events=list(events)))
            # Alternating order avoids treating one measurement order as proof.
            for i in range(iterations):
                for variant in (('independent','chain') if i%2==0 else ('chain','independent')):
                    events.clear();t=time.perf_counter();y=execute(variant);elapsed=(time.perf_counter()-t)*1000
                    used=chains[:2] if variant=='independent' else chains[2:]
                    checks=check(y,ref)
                    np.save(out/f'{variant}-output-{i}.npy',y)
                    result['variants'][variant]['samples'].append(dict(iteration=i,wall_ms=elapsed,check=checks,events=list(events),actual_submissions=sum(e['kind']=='actual_xrt_start' for e in events),actual_waits=sum(e['kind']=='actual_xrt_wait' for e in events),readback_indices=[[2],[2]] if variant=='independent' else [[4]],output_path=str(out/f'{variant}-output-{i}.npy'),runtime_objects=[snapshot(c) for c in used]))
            for v in result['variants'].values(): v['median_ms']=statistics.median(s['wall_ms'] for s in v['samples'])
            result['pairwise_output_max_abs']=[float(np.max(abs(np.load(out/f'independent-output-{i}.npy')-np.load(out/f'chain-output-{i}.npy')))) for i in range(iterations)]
            result['status']='passed' if all(s['check']['passed'] for v in result['variants'].values() for s in v['samples']+v['excluded']) else 'numerical_failed'
    except Exception as exc:
        result.update(error=str(exc),traceback=traceback.format_exc())
        if 'events' in locals(): result['failure_events']=list(events)
        (out/'failure.txt').write_text(result['traceback'])
    finally:
        if original_wait is not None: multilaunch._wait=original_wait
        for c in chains: c.close()
        write_json(out/'result.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--case',choices=['train','holdout','all'],default='all');p.add_argument('--iterations',type=int,default=7);p.add_argument('--compile-only',action='store_true');a=p.parse_args()
    if a.iterations<3: p.error('at least three hot samples required')
    from filelock import FileLock
    with (contextlib.nullcontext() if a.compile_only else FileLock('/tmp/npu.lock',timeout=1800)):
        results=[experiment(case,a.output_dir.resolve()/case,a.iterations,a.compile_only) for case in (['train','holdout'] if a.case=='all' else [a.case])]
    print(json.dumps([dict(case=r['case'],status=r['status'],error=r.get('error')) for r in results],indent=2))
    return 0 if all(r['status'] in ('passed','compiled') for r in results) else 1
if __name__=='__main__':raise SystemExit(main())
