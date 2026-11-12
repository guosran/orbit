#!/usr/bin/env python3
"""Executable Qwen FFN dispatch fusion with explicit DDR ownership.

This adapter does not claim SRAM residency: NPUChain passes intermediate BOs
between firmware-sequenced array configurations in DDR. BF16 projection drains
preserve the standalone NPU SwiGLU input rounding; hardware numerical validation
is still required because the projection lowering changes.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import time
import textwrap


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(root):
    root = Path(root).resolve(strict=True)
    sys.path.insert(0, str(root / 'examples/qwen25_3b_q4nx'))
    sys.path.insert(0, str(root / 'examples/llm_q4nx'))
    return root


class QwenFFN:
    """One true multi-launch ELF; original weights stay BF16, never gamma-folded.

    Retain and reload control actual MultiLaunchRunner static_indices. All arrays
    have stable ownership per layer. close() releases the runner and all its BOs.
    """
    def __init__(self, root, rows=128, block_m=128, block_n=256,
                 retention='retain', split_order='gate-up'):
        self.root = configure(root)
        if min(rows,block_m,block_n)<=0 or rows % block_m or retention not in ('retain', 'reload','release-after-call'):
            raise ValueError('Rows must be a block_m multiple; invalid retention')
        if split_order not in ('gate-up', 'up-gate'):
            raise ValueError('Only independent gate/up order can be swapped')
        import kernels
        self.rows, self.bm, self.bn = rows, block_m, block_n
        self.retention, self.split_order = retention, split_order
        self.block = kernels.SWIGLU_BLOCK
        # Padded channels are zero weight columns, never logical model values.
        # A flat original SwiGLU block can therefore process contiguous planes.
        alignment = math.lcm(block_n, self.block // math.gcd(rows, self.block), 64)
        self.hidden = math.ceil(11008 / alignment) * alignment
        self.chain = None
        self.weights, self.arguments, self.events = {}, {}, []
        self.call_events=[]
        self._seen_keys=set()
        self.sequence = 0

    def describe(self):
        r, h = self.rows, self.hidden
        buffers = [
            ('input', [r, 2048], 2, 'host-input', 0, 3),
            ('gate-weight', [2048, h], 2, 'weight', 0, 0),
            ('up-weight', [2048, h], 2, 'weight', 1, 1),
            ('down-weight', [h, 2048], 2, 'weight', 3, 3),
            ('gate', [r, h], 2, 'intermediate', 0, 2),
            ('up', [r, h], 2, 'intermediate', 1, 2),
            ('activation', [r, h], 2, 'intermediate', 2, 3),
            ('output', [r, 2048], 4, 'host-output', 3, 3),
        ]
        if self.split_order == 'up-gate':
            buffers[1] = (*buffers[1][:4], 1, 1)
            buffers[2] = (*buffers[2][:4], 0, 0)
            buffers[4] = (*buffers[4][:4], 1, 2)
            buffers[5] = (*buffers[5][:4], 0, 2)
        return dict(schema='orbit.qwen-ffn-runtime.v1', logical_shape=[2048,11008,2048],
            row_tile=r, block_m=self.bm, block_n=self.bn,
            padded_hidden=h, retention=self.retention, split_order=self.split_order,
            fusion='single-xrt-dispatch-multiple-firmware-array-configurations',
            numerical_contract='BF16 GEMM inputs/f32 accumulation; BF16 gate/up drains; original NPU SwiGLU BF16 rounding; f32 down output; numerical gate required',
            sram_residency_claim=False, memory='device-ddr-bo',
            scratch_initialization='original runner zeroes and syncs intermediate BOs on first call',
            static_indices=[1,2,3] if self.retention=='retain' else [],
            intermediate_indices=[4,5,6,7], output_indices=[7],
            buffers=[dict(index=i,id=name,shape=shape,bytes=math.prod(shape)*width,
                         role=role,memory='device-ddr-bo',first_kernel=first,last_kernel=last,
                         lifetime='call completion' if self.retention=='release-after-call' else 'layer-bo-key until close',
                         upload='first-call-cache-sync' if role=='weight' and self.retention=='retain' else
                                'every-call' if role in ('host-input','weight') else 'first-call-zero',
                         download='every-call' if role=='host-output' else 'never')
                     for i,(name,shape,width,role,first,last) in enumerate(buffers)],
            kernel_order=[*self.split_order.split('-'),'swiglu','down'],
            executable_status='requires compiled artifact and board numerical gate')

    def build(self):
        if self.chain is not None:
            return self.chain
        import torch
        import kernels
        import fused_mlp
        from triton.backends.amd_triton_npu.multilaunch import NPUChain
        r,h = self.rows,self.hidden
        chain = NPUChain(f'orbit_qwen_ffn_r18_m{r}_{self.bm}x{self.bn}_{self.split_order.replace("-","_")}',
                         air_project_path='/tmp/orbit-qwen-r18-ffn-build')
        order = [(1,4),(2,5)] if self.split_order=='gate-up' else [(2,5),(1,4)]
        # This existing kernel accumulates in f32 and drains to BF16 before
        # SwiGLU. No gamma folding or f32 activation shortcut is permitted.
        for source,dest in order:
            chain.add(fused_mlp._mm_bf16_kernel, grid=(r//self.bm,h//self.bn),
                arg_map={0:0,1:source,2:dest},
                **fused_mlp._mm_op(r,2048,h,self.bm,self.bn,2048,
                    l2_k=fused_mlp.EPI_L2_K,out_dtype=torch.bfloat16,
                    constexprs=dict(GELU=False,C2=0.,KG=0.)),
                transform_script=kernels.script(fused_mlp.EPI_SCRIPT))
        plane=torch.empty(r*h,dtype=torch.bfloat16)
        chain.add(kernels._swiglu_kernel,grid=(r*h//self.block,),
            arg_map={0:4,1:5,2:6},args=(plane,plane,plane),
            constexprs={'BLOCK':self.block},
            transform_script=kernels.script('swiglu/transform_aie2p.mlir'))
        chain.add(fused_mlp._mm_kernel,grid=(r//self.bm,2048//self.bn),
            arg_map={0:6,1:3,2:7},
            **fused_mlp._mm_op(r,h,2048,self.bm,self.bn,h),
            transform_script=kernels.matmul_script(self.bm,self.bn))
        self.chain=chain
        return chain

    def emit(self, output, compile_artifact=False):
        """Compile-only path: never construct runner/hw_context or dispatch."""
        from triton.backends.amd_triton_npu.multilaunch import MultiLaunchBuilder
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        c=self.build();b=MultiLaunchBuilder(c.name,air_project_path=str(output))
        for i,(kernel,grid,mapping,script,args,constants,sizes) in enumerate(c._specs):
            text=c._capture_ttshared(kernel,grid,args,constants)
            if isinstance(text,bytes):text=text.decode()
            (output/f'op-{i}.ttshared.mlir').write_text(text)
            b.add_op(text,grid,mapping,transform_script=script,actual_sizes=sizes)
        (output/'combined.mlir').write_text(b.build_module_text())
        manifest=self.describe()
        manifest['compiled']=False
        if compile_artifact:
            artifact,kernel=b.compile(output_format='elf',link_profile='xrt')
            manifest.update(compiled=True,artifact=str(Path(artifact).resolve()),
                            artifact_sha256=sha(artifact),entrypoint=kernel)
        sources=[Path(__file__),self.root/'examples/llm_q4nx/kernels.py',
                 self.root/'examples/llm_q4nx/fused_mlp.py']
        manifest['source_files']=[dict(file=str(p.resolve()),sha256=sha(p)) for p in sources]
        manifest['emitted_files']=[dict(file=str(p.resolve()),sha256=sha(p)) for p in sorted(output.glob('*.mlir'))]
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return manifest

    def add_layer(self, layer, gate_up, down):
        import torch
        if tuple(gate_up.shape)!=(2048,22016) or tuple(down.shape)!=(11008,2048):
            raise ValueError('Require original unpadded Qwen BF16 weights')
        if gate_up.dtype!=torch.bfloat16 or down.dtype!=torch.bfloat16:
            raise ValueError('Weight precision must remain original BF16')
        if layer in self.weights:
            raise ValueError('A fixed fresh-process plan cannot replace a prepared layer')
        bg=torch.zeros(2048,self.hidden,dtype=torch.bfloat16)
        bu=torch.zeros_like(bg);bd=torch.zeros(self.hidden,2048,dtype=torch.bfloat16)
        bg[:,:11008]=gate_up[:,:11008];bu[:,:11008]=gate_up[:,11008:]
        bd[:11008]=down
        # Verify reinterpretation, rather than numerical equality hiding NaNs.
        if not (torch.equal(bg[:,:11008].view(torch.int16),gate_up[:,:11008].view(torch.int16)) and
                torch.equal(bu[:,:11008].view(torch.int16),gate_up[:,11008:].view(torch.int16)) and
                torch.equal(bd[:11008].view(torch.int16),down.view(torch.int16))):
            raise ValueError('Projection fission changed original weight bits')
        if self.retention=='retain':
            import kernels
            bound=[]
            for original in (bg,bu,bd):
                shared=kernels.shared_empty(original.shape,original.dtype)
                if kernels.shared_bo(shared) is None:
                    raise ValueError('Retain requires a real shared BO, no silent private duplicate')
                shared.copy_(original)
                if not torch.equal(shared.view(torch.int16),original.view(torch.int16)):
                    raise ValueError('Shared BO preparation changed BF16 bits')
                bound.append(shared)
            bg,bu,bd=bound
        self.weights[layer]=(bg,bu,bd)

    def run(self, layer, h):
        start=time.perf_counter_ns();begin=len(self.events)
        result=self._run(layer,h)
        self.call_events.append(dict(layer=layer,exclusive_wall_ns=time.perf_counter_ns()-start,
            row_events=self.events[begin:],boundary='complete FFN adapter call including input staging, dispatch, sync and output clone'))
        return result

    def _run(self, layer, h):
        import torch
        import numpy as np
        from ml_dtypes import bfloat16
        from triton.backends.amd_triton_npu.driver_scope import driver_scope
        if h.ndim!=2 or h.shape[1]!=2048 or layer not in self.weights:
            raise ValueError('Unregistered source layer or activation shape')
        bg,bu,bd=self.weights[layer]
        if layer not in self.arguments:
            def array(t): return t.numpy(force=True) if t.dtype!=torch.bfloat16 else t.view(torch.uint16).numpy().view(bfloat16)
            self.arguments[layer]=[np.zeros((self.rows,2048),dtype=bfloat16),array(bg),array(bu),array(bd),
                *[np.empty((self.rows,self.hidden),dtype=bfloat16) for _ in range(3)],
                np.empty((self.rows,2048),dtype=np.float32)]
        args=self.arguments[layer];out=torch.empty(h.shape,dtype=torch.float32)
        with driver_scope('npu'):
            c=self.build()
            for start in range(0,len(h),self.rows):
                count=min(self.rows,len(h)-start);args[0].fill(0)
                args[0][:count]=h[start:start+count].to(torch.bfloat16).view(torch.uint16).numpy().view(bfloat16)
                stamp=time.perf_counter_ns();key=f'orbit-r18-L{layer}-{id(self):x}'
                first=key not in self._seen_keys
                bound=None
                if self.retention=='retain':
                    import kernels
                    bound={i:kernels.shared_bo(w) for i,w in zip((1,2,3),(bg,bu,bd))}
                    if any(bo is None for bo in bound.values()):raise ValueError('Retain lost actual shared BO ownership')
                result=c.run(args,bo_key=key,
                    static_indices={1,2,3} if self.retention=='retain' else set(),
                    intermediate_indices={4,5,6,7},output_indices={7},bound_buffers=bound)
                out[start:start+count]=torch.from_numpy(np.asarray(result[7])[:count].copy())
                desc=self.describe();uploads=[b['index'] for b in desc['buffers'] if
                    b['upload']=='every-call' or first and b['upload'] in ('first-call','first-call-zero')]
                self.events.append(dict(sequence=self.sequence,layer=layer,row_start=start,rows=count,
                    bo_key=key,first_call=first,upload_buffers=uploads,download_buffers=[7],
                    bound_weight_cache_sync_buffers=[1,2,3] if first and bound else [],
                    weight_binding='shared BO aliases prepared BF16 storage' if bound else 'private staged BO',
                    exclusive_wall_ns=time.perf_counter_ns()-stamp,
                    wall_boundary='NPUChain.run plus output clone; includes compile if cold',
                    copy_granularity='complete padded BO',kernel_dispatches=1,array_configurations=4))
                self.sequence+=1
                self._seen_keys.add(key)
                if self.retention=='release-after-call':
                    # The output has been cloned and the synchronous run has
                    # completed. Removing this actual owner releases its BOs;
                    # the next invocation really allocates and initializes again.
                    if c._runtime!='xrt' or key not in c._runner._bos:
                        raise ValueError('Expected synchronous XRT BO owner')
                    del c._runner._bos[key]
                    self._seen_keys.remove(key)
                    self.events[-1]['released_bo_key']=key
        return out

    def close(self):
        if self.chain is not None:self.chain.close()
        self.weights.clear();self.arguments.clear();self._seen_keys.clear()


def make_prefiller_class(base, ffn, layers):
    """Rewrite only the exact recognized tail, leaving the source prefix AST.

    This changes executable implementation coverage. It is not the r17 original
    hook runner, and must have its own instrumentation and heldout baselines.
    """
    tree=ast.parse(textwrap.dedent(inspect.getsource(base._layer)))
    fn=tree.body[0]
    expected=ast.parse('g, u = self._matmul(h, w["gate_up"], **self._wide).split([INTER, INTER], dim=1)\nreturn x + self._matmul(self._swiglu(g, u), w["down"])').body
    if [ast.dump(x) for x in fn.body[-2:]]!=[ast.dump(x) for x in expected]:
        raise ValueError('Original Qwen FFN AST no longer matches executable coverage')
    # Keep the original unfused statements, including their line boundaries.
    # The fused branch has one complete-call boundary at the original return.
    replacement=ast.parse('if L in self._orbit_fused_layers:\n    return x + self._orbit_ffn.run(L, h)').body[0]
    ast.increment_lineno(replacement,fn.body[-1].lineno-2)
    replacement.orelse=fn.body[-2:]
    fn.body[-2:]=[replacement]
    namespace=dict(base._layer.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<orbit-r18-source-preserving-tail>','exec'),namespace)
    class FusedPrefill(base):
        _orbit_ffn=ffn
        _orbit_fused_layers=frozenset(layers)
    FusedPrefill._layer=namespace['_layer']
    return FusedPrefill


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--triton-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rows',type=int,default=128)
    p.add_argument('--block-m',type=int,default=128)
    p.add_argument('--block-n',type=int,default=256)
    p.add_argument('--retention',choices=['retain','reload','release-after-call'],default='retain')
    p.add_argument('--split-order',choices=['gate-up','up-gate'],default='gate-up')
    p.add_argument('--compile',action='store_true')
    a=p.parse_args()
    m=QwenFFN(a.triton_root,a.rows,a.block_m,a.block_n,a.retention,a.split_order)
    print(json.dumps(m.emit(a.output,a.compile),indent=2))


if __name__=='__main__':main()
