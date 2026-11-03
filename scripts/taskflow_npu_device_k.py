#!/usr/bin/env python3
"""Minimal two-operation sequential K accumulator experiment; owns NPU lock."""
import argparse,json,sys,time,traceback,hashlib,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from orbit_npu.runtime import RECIPES, Executor
from orbit_npu.kernels import matmul_accumulate,matmul_chunked

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--strategy',choices=['two-launch','nested-loop'],default='two-launch');p.add_argument('--chunk-k',type=int,default=64);p.add_argument('--columns',type=int,default=256);a=p.parse_args()
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__),out/'reproduce.py')
    shutil.copyfile(ROOT_KERNEL:=Path(__file__).resolve().parents[1]/'python/orbit_npu/kernels.py',out/'kernels.py')
    import torch
    from filelock import FileLock
    from triton.backends.amd_triton_npu.driver_scope import driver_scope
    from triton.backends.amd_triton_npu.multilaunch import NPUChain
    from triton.backends.amd_triton_npu.config import npu_config
    from orbit_npu.profiling import install
    install();npu_config.target='npu2';npu_config.runtime='xrt';npu_config.bf16_emulation=False
    npu_config.air_project_path=str(out/'air-build')
    result=dict(status='compile_failed',semantic_k_chunks=[[0,a.chunk_k],[a.chunk_k,2*a.chunk_k]],
                submissions_per_graph=1,device_launches=2,readbacks_per_graph=1,
                intermediate='shared device-accessible C buffer in system memory',
                initialization='first launch zero accumulator, second launch loads same C',
                numerical_order='second launch begins FP32 MAC from first partial; differs from separate partial then host add')
    with FileLock('/tmp/npu.lock',timeout=600),driver_scope('npu'),torch.inference_mode():
        try:
            rng=np.random.default_rng(0)
            aa=rng.normal(0,.2,(128,2*a.chunk_k)).astype('f4');bb=rng.normal(0,.2,(2*a.chunk_k,a.columns)).astype('f4')
            tensors=[]
            for lo in (0,a.chunk_k):
                tensors.extend([torch.from_numpy(aa[:,lo:lo+a.chunk_k].copy()).bfloat16(),torch.from_numpy(bb[lo:lo+a.chunk_k].copy()).bfloat16()])
            tensors.append(torch.full((128,a.columns),float('nan')))
            chain=NPUChain('orbit_device_k_mre',air_project_path=npu_config.air_project_path)
            for i in range(2):
                chain.add(matmul_accumulate,grid=(1,a.columns//256),arg_map={0:2*i,1:2*i+1,2:4},
                    args=(tensors[2*i],tensors[2*i+1],tensors[4],a.chunk_k,a.columns,a.columns),
                    constexprs=dict(K=a.chunk_k,BLOCK_M=128,BLOCK_N=256,BLOCK_K=64,INITIALIZE=i==0),
                    transform_script=str(RECIPES/'transform_matmul_aie2p.mlir'))
            if a.strategy=='nested-loop':
                tensors=[torch.from_numpy(aa).bfloat16(),torch.from_numpy(bb).bfloat16(),torch.full((128,a.columns),float('nan'))]
                chain=NPUChain('orbit_device_k_nested_mre',air_project_path=npu_config.air_project_path)
                chain.add(matmul_chunked,grid=(1,a.columns//256),arg_map={0:0,1:1,2:2},
                    args=(tensors[0],tensors[1],tensors[2],2*a.chunk_k,a.columns,a.columns),
                    constexprs=dict(CHUNKS=2,CHUNK_K=a.chunk_k,BLOCK_M=128,BLOCK_N=256,BLOCK_K=64),
                    transform_script=str(RECIPES/'transform_matmul_aie2p.mlir'))
                result.update(device_launches=1,initialization='once before explicit outer semantic-K loop',intermediate='on-device accumulator',numerical_order='continuous FP32 accumulator across explicit semantic chunks')
            for i,spec in enumerate(chain._specs):
                kernel,grid,argmap,transform,args,constants,actual_sizes=spec
                ir=chain._capture_ttshared(kernel,grid,args,constants)
                (out/f'op-{i}-ttshared.mlir').write_text(ir.decode() if isinstance(ir,bytes) else ir)
            result['strategy']=a.strategy
            start=time.perf_counter();chain._build();result['compile_seconds']=time.perf_counter()-start
            result['status']='runtime_failed';samples=[];checks=[]
            for i in range(5):
                start=time.perf_counter()
                output_index=len(tensors)-1
                got=chain.run([Executor._numpy(t) for t in tensors],static_indices={1,3} if output_index==4 else {1},intermediate_indices={output_index},output_indices={output_index})
                elapsed=(time.perf_counter()-start)*1000
                y=np.asarray(got[output_index]).reshape(128,a.columns).copy();ref=aa.astype('f8')@bb.astype('f8')
                checks.append(dict(finite=bool(np.isfinite(y).all()),max_abs=float(np.max(abs(y-ref))),rms_relative=float(np.sqrt(np.mean((y-ref)**2))/np.sqrt(np.mean(ref**2)))))
                np.savez(out/f'output-{i}.npz',actual=y,reference=ref)
                if i>=2:samples.append(elapsed)
            result.update(status='passed' if all(c['finite'] and c['rms_relative']<=.03 for c in checks) else 'numerical_failed',checks=checks,host_wall_ms=samples)
            chain.close()
        except Exception as e:
            result.update(error=str(e),traceback=traceback.format_exc());(out/'failure.txt').write_text(result['traceback'])
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
