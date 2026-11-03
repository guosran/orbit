"""Restricted taskflow.plan.v1 executor for the npu2/AIE2P target.

Public program semantics stay FP32. This approximate target implementation
quantizes GEMM operands to BF16 and uses the wheel's BFP16-EBS8 AIE2P MAC.
All scheduling, input slices, dependency checks and K partial sums follow the
materialized plan. Host placements are explicit and are never device results.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOY = Path(os.environ.get('AIE_TOY_ROOT', '/home/shiran/project/aie-toy'))
_COMPILED = {}
_ACTIVE_RECIPE = None
RECIPES = TOY / 'build/qwen25-3b/Triton-XDNA/examples/llm_q4nx'
sys.path.insert(0, str(ROOT / 'scripts'))
from taskflow_common import compare, fingerprint, write_json

class Unsupported(ValueError):
    pass


def normalize_config(config=None):
    result = dict(schema='orbit.npu.config.v1', weights='staged',
                  projection='separate', accumulation='host')
    result.update(config or {})
    allowed = dict(schema={'orbit.npu.config.v1'}, weights={'staged','persistent'},
                   projection={'separate','merge_siblings'}, accumulation={'host','device'})
    if set(result) != set(allowed) or any(result[k] not in v for k,v in allowed.items()):
        raise Unsupported('invalid orbit.npu.config.v1 target configuration')
    if result['accumulation'] == 'device' and result['projection'] != 'separate':
        raise Unsupported('device sequential K currently requires separate projections')
    return result


def validate(plan, inputs):
    if plan.get('schema') != 'orbit.taskflow.plan.v1':
        raise Unsupported('expected orbit.taskflow.plan.v1')
    buffers = {b['id']: b for b in plan['buffers']}
    if len(buffers) != len(plan['buffers']):
        raise Unsupported('duplicate storage identity')
    for b in buffers.values():
        if b['dtype'] != 'f32' or len(b['shape']) != 2 or min(b['shape']) < 1:
            raise Unsupported('only static positive 2D f32 buffers supported')
        if b['role'] == 'input':
            if b['id'] not in inputs or list(inputs[b['id']].shape) != b['shape']:
                raise Unsupported('missing or incorrectly shaped input ' + b['id'])
            if not np.isfinite(inputs[b['id']]).all():
                raise Unsupported('nonfinite inputs unsupported')
    completed = set()
    for task in plan['tasks']:
        if task['id'] in completed or not set(task['depends_on']) <= completed:
            raise Unsupported('cyclic, duplicate or non-topological task order')
        for op in task['operations']:
            kind = op['kind']
            if kind not in ('fill', 'matmul', 'relu', 'swiglu'):
                raise Unsupported('unsupported operation: ' + kind)
            if len(op['inputs']) != dict(fill=0, matmul=2, relu=1, swiglu=2)[kind]:
                raise Unsupported('operation input arity mismatch')
            dst = buffers[op['output']]
            if dst['role'] == 'input':
                raise Unsupported('input mutation unsupported')
            for axis, extent in zip(('m', 'n'), dst['shape']):
                lo, hi = op[axis]
                if not 0 <= lo < hi <= extent:
                    raise Unsupported('out-of-bounds output slice')
            if kind == 'matmul':
                a, b = [buffers[x] for x in op['inputs']]
                if not op.get('accumulate') or a['shape'][1] != b['shape'][0]:
                    raise Unsupported('matmul requires explicit accumulation and equal K')
                lo, hi = op['k']
                if not 0 <= lo < hi <= a['shape'][1]:
                    raise Unsupported('out-of-bounds K slice')
                if op['m'][1] > a['shape'][0] or op['n'][1] > b['shape'][1]:
                    raise Unsupported('out-of-bounds matmul input slice')
            else:
                for name in op['inputs']:
                    shape = buffers[name]['shape']
                    if op['m'][1] > shape[0] or op['n'][1] > shape[1]:
                        raise Unsupported('out-of-bounds elementwise input')
        completed.add(task['id'])
    if any(name not in buffers for name in plan['outputs']):
        raise Unsupported('unknown output storage')


class Executor:
    def __init__(self, plan, inputs, directory, target_config=None):
        import torch
        from .kernels import matmul, matmul_chunked
        from triton.backends.amd_triton_npu.config import npu_config
        self.torch, self.kernel, self.config = torch, matmul, npu_config
        self.chunk_kernel = matmul_chunked
        self.plan, self.directory = plan, directory
        self.target_config = normalize_config(target_config)
        self.config_identity = fingerprint(self.target_config)
        self.generation = 0
        self.chains, self.shared_weights, self.merges, self.merged_followers = {}, {}, {}, {}
        self.pending_merged = {}
        self.k_groups, self.k_followers = {}, {}
        self.staged_generations = set()
        self.weight_uploads = 0
        self.weight_sync_bytes_theoretical = 0
        self.compiled, self.calls, self.static_weights = {}, {}, {}
        self.compile_seconds = 0.0
        self.first_weight_preparation_seconds = 0.0
        self.memory = {b['id']: np.array(inputs[b['id']], dtype=np.float32, copy=True)
                       if b['role'] == 'input' else np.full(b['shape'], np.nan, np.float32)
                       for b in plan['buffers']}
        self.input_ids = {b['id'] for b in plan['buffers'] if b['role'] == 'input'}
        self.trace = []
        self.placements = []
        self.device_dispatches = 0
        npu_config.target = 'npu2'
        npu_config.runtime = 'xrt'
        npu_config.air_project_path = str(directory / 'air-build')
        npu_config.transform_tiling_script = str(RECIPES / 'transform_matmul_aie2p.mlir')
        npu_config.bf16_emulation = False
        # Fixed implementation configuration, independent from semantic tiles.
        self.bm, self.bn, self.bk = 128, 256, 64
        self.recipe_identity = fingerprint(dict(target='npu2',runtime='xrt',format='elf',
            block=[self.bm,self.bn,self.bk],dtype='bf16-bfp16-f32',layout='row-major-padded',
            kernel=hashlib.sha256((Path(__file__).parent/'kernels.py').read_bytes()).hexdigest(),
            schedule=hashlib.sha256((RECIPES/'transform_matmul_aie2p.mlir').read_bytes()).hexdigest(),
            launcher=hashlib.sha256((Path(__file__).parent/'profiling.py').read_bytes()).hexdigest(),
            versions={p:importlib.metadata.version(p) for p in ('triton-xdna','mlir-air','mlir-aie-no-rtti','llvm-aie')}))
        global _ACTIVE_RECIPE
        if _ACTIVE_RECIPE is not None and _ACTIVE_RECIPE != self.recipe_identity:
            raise Unsupported('recipe/toolchain changed in this process; start a fresh process to avoid stale driver JIT state')
        _ACTIVE_RECIPE = self.recipe_identity
        self._find_k_groups()
        self._find_merges()
        self._prepare()

    def _find_k_groups(self):
        if self.target_config['accumulation'] != 'device':
            return
        for ti,task in enumerate(self.plan['tasks']):
            ops = task['operations']
            oi = 0
            while oi < len(ops):
                first = ops[oi]
                if first['kind'] != 'matmul':
                    oi += 1
                    continue
                group=[first];end=oi+1
                while end<len(ops):
                    op=ops[end]
                    if (op['kind']!='matmul' or any(op[x]!=first[x] for x in ('inputs','output','m','n'))
                        or op['k'][0]!=group[-1]['k'][1]):
                        break
                    group.append(op);end+=1
                if len(group)>1:
                    # No intermediate consumer/fill lies in this consecutive
                    # sequence; carries the same FP32 accumulator across chunks.
                    self.k_groups[(ti,oi)]=group
                    for other in range(oi+1,end):self.k_followers[(ti,other)]=(ti,oi)
                oi=end
        if not self.k_groups:
            raise Unsupported('device K needs consecutive materialized K partials within a task')

    def _find_merges(self):
        # Only immutable common inputs: moving the second projection earlier
        # cannot observe a changed producer. Distinct destinations avoid writes.
        if self.target_config['projection'] != 'merge_siblings':
            return
        ops = [((ti,oi),op) for ti,t in enumerate(self.plan['tasks'])
               for oi,op in enumerate(t['operations'])]
        used = set()
        for index,(key,op) in enumerate(ops):
            if key in used or op['kind'] != 'matmul' or not set(op['inputs']) <= self.input_ids:
                continue
            for key2,other in ops[index+1:]:
                if key2 in used or other['kind'] != 'matmul' or not set(other['inputs']) <= self.input_ids:
                    continue
                if (op['inputs'][0] == other['inputs'][0] and op['m']==other['m']
                    and op['k']==other['k'] and op['output']!=other['output']):
                    self.merges[key] = (key2,other)
                    self.merged_followers[key2] = key
                    used.update((key,key2))
                    break

    def update_inputs(self, inputs):
        """Snapshot new external objects, including in-place updates, explicitly.

        execute() observes owned immutable snapshots, never caller aliases.
        Updating invalidates all prepared weights/BO keys (including merged
        layouts), even when the same Python object or address is reused.
        """
        validate(self.plan,inputs)
        for chain in self.chains.values():
            chain.close()
        self.chains.clear()
        self.shared_weights.clear()
        self.generation += 1
        for name in self.input_ids:
            self.memory[name] = np.array(inputs[name],dtype=np.float32,copy=True)
        self.calls.clear()
        self.placements.clear()
        self._prepare()
        self.compile()

    @staticmethod
    def _numpy(tensor):
        import torch
        if tensor.dtype == torch.bfloat16:
            from ml_dtypes import bfloat16
            return tensor.view(torch.uint16).numpy().view(bfloat16)
        return tensor.numpy()

    def close(self):
        for chain in self.chains.values():
            chain.close()
        self.chains.clear()
        self.shared_weights.clear()

    def _prepare(self):
        """Allocate and quantize immutable input slices exactly once per seed."""
        torch = self.torch
        for ti, task in enumerate(self.plan['tasks']):
            for oi, op in enumerate(task['operations']):
                place = {'task': task['id'], 'operation': oi, 'kind': op['kind'],
                         'm': op['m'], 'n': op['n']}
                if op['kind'] == 'matmul':
                    key = (ti,oi)
                    if key in self.k_followers:
                        place.update(device='npu',implementation='inside-semantic-k-loop',
                                     producer=list(self.k_followers[key]),dispatches=0,readbacks=0)
                        self.placements.append(place)
                        continue
                    if key in self.merged_followers:
                        place.update(device='cpu', implementation='slice-merged-projection',
                                     producer=list(self.merged_followers[key]),dispatches=0)
                        self.placements.append(place)
                        continue
                    m, n, k = [op[x][1] - op[x][0] for x in ('m', 'n', 'k')]
                    original_n = n
                    merged = self.merges.get(key)
                    if merged:
                        n += merged[1]['n'][1]-merged[1]['n'][0]
                    mp, np_, kp = [(x+t-1)//t*t for x,t in ((m,self.bm),(n,self.bn),(k,self.bk))]
                    group = self.k_groups.get(key)
                    if group:
                        chunk_k = max((x['k'][1]-x['k'][0]+self.bk-1)//self.bk*self.bk for x in group)
                        kp = len(group)*chunk_k
                    t = time.perf_counter()
                    a = torch.zeros((mp, kp), dtype=torch.bfloat16)
                    if self.target_config['weights'] == 'persistent' and op['inputs'][1] in self.input_ids:
                        from triton.backends.amd_triton_npu import shared
                        buffer = shared.empty((kp,np_),dtype=torch.bfloat16,device='xrt:0')
                        if buffer.bo is None:
                            raise Unsupported('XRT shared weight BO unavailable')
                        b = buffer.torch()
                        b.zero_()
                        self.shared_weights[key] = buffer
                    else:
                        b = torch.zeros((kp, np_), dtype=torch.bfloat16)
                    c = torch.empty((mp, np_), dtype=torch.float32)
                    ms, ns, ks = slice(*op['m']), slice(*op['n']), slice(*op['k'])
                    for index, tensor, view in ((0,a,(ms,ks)), (1,b,(ks,ns))):
                        if op['inputs'][index] in self.input_ids:
                            src = self.memory[op['inputs'][index]][view]
                            tensor[:src.shape[0], :src.shape[1]].copy_(torch.from_numpy(src))
                    if group:
                        a.zero_();b.zero_()
                        for index,gop in enumerate(group):
                            ks2=slice(*gop['k']);width=ks2.stop-ks2.start;offset=index*chunk_k
                            if gop['inputs'][0] in self.input_ids:
                                a[:m,offset:offset+width].copy_(torch.from_numpy(self.memory[gop['inputs'][0]][ms,ks2]))
                            if gop['inputs'][1] in self.input_ids:
                                b[offset:offset+width,:n].copy_(torch.from_numpy(self.memory[gop['inputs'][1]][ks2,ns]))
                    if merged:
                        other = merged[1]
                        src = self.memory[other['inputs'][1]][slice(*other['k']),slice(*other['n'])]
                        b[:k,original_n:original_n+src.shape[1]].copy_(torch.from_numpy(src))
                    self.first_weight_preparation_seconds += time.perf_counter()-t
                    self.calls[(ti,oi)] = (a,b,c,(m,n,k),(mp,np_,kp))
                    place.update(device='npu', kernel='orbit_npu.kernels.matmul',
                        k=op['k'], padded_shape=[mp,np_,kp],
                        dispatches=1 if key in self.shared_weights else mp//self.bm, partial_sum_placement='cpu-f32',
                        conversion='f32→bf16→BFP16-EBS8; FP32 accumulation/output',
                        immutable_inputs=[x for x in op['inputs'] if x in self.input_ids],
                        weight_lifetime='executor owned immutable snapshot; explicit update invalidates' if key in self.shared_weights else 'staged each execution',
                        weight_buffer='XRT BO in system DRAM, not AIE SRAM' if key in self.shared_weights else 'driver staging',
                        weight_bytes_theoretical=kp*np_*2,
                        readbacks=1 if key in self.shared_weights else mp//self.bm)
                    if group:
                        place.update(kernel='orbit_npu.kernels.matmul_chunked',dispatches=1,readbacks=1,
                                     semantic_k_chunks=[x['k'] for x in group],packed_chunk_k=chunk_k,
                                     partial_sum_placement='device FP32 accumulator',
                                     numerical_order='continuous FP32 accumulation; differs from sum of individually rounded partials')
                    if merged:
                        place.update(merged_operations=[list(key),list(merged[0])],
                                     output_slices=[[0,original_n],[original_n,n]])
                else:
                    place.update(device='cpu', implementation='numpy-f32',
                                 reason='explicit host placement in restricted adapter')
                self.placements.append(place)

    def compile(self):
        from triton.backends.amd_triton_npu.driver import get_npu_cache_dir
        for key, (a,b,c,_,(mp,np_,kp)) in self.calls.items():
            if key in self.shared_weights or key in self.k_groups:
                from triton.backends.amd_triton_npu.multilaunch import NPUChain
                chain = NPUChain('orbit_'+self.recipe_identity[:12]+f'_{mp}_{np_}_{kp}',
                                 air_project_path=str(self.directory/'air-build'))
                constants=dict(BLOCK_M=self.bm,BLOCK_N=self.bn,BLOCK_K=self.bk)
                if key in self.k_groups:
                    constants.update(CHUNKS=len(self.k_groups[key]),CHUNK_K=kp//len(self.k_groups[key]))
                    kernel=self.chunk_kernel
                else:
                    constants.update(K=kp,RELU=False);kernel=self.kernel
                chain.add(kernel,grid=(mp//self.bm,np_//self.bn),arg_map={0:0,1:1,2:2},
                          args=(a,b,c,kp,np_,np_),constexprs=constants,
                          transform_script=str(RECIPES/'transform_matmul_aie2p.mlir'))
                start = time.perf_counter()
                chain._build()
                elapsed = time.perf_counter()-start
                self.compile_seconds += elapsed
                self.chains[key] = chain
                shape = (mp,kp,np_,'chain',fingerprint(constants)[:12])
                artifact = self.directory/'artifacts'/f'chain-{mp}-{kp}-{np_}-{fingerprint(constants)[:12]}'
                artifact.mkdir(parents=True,exist_ok=True)
                (artifact/'combined-air.mlir').write_text(chain._builder.build_module_text())
                shutil.copytree(Path(chain._elf_path).parent,artifact/'cache',dirs_exist_ok=True)
                self.compiled[shape] = dict(seconds=elapsed,shape=list(shape),elf=str(chain._elf_path),
                    artifact_copy=str(artifact),elf_sha256=hashlib.sha256(Path(chain._elf_path).read_bytes()).hexdigest())
                continue
            shape = (kp,np_)
            if shape in self.compiled:
                continue
            cache_key = (self.recipe_identity,*shape)
            if cache_key in _COMPILED:
                self.compiled[shape] = dict(_COMPILED[cache_key],seconds=0.0,in_process_cache_hit=True)
                continue
            kw = dict(K=kp, BLOCK_M=self.bm, BLOCK_N=self.bn, BLOCK_K=self.bk, RELU=False)
            args = (a[:self.bm],b,c[:self.bm],kp,np_,np_)
            self.config.compile_only = True
            start = time.perf_counter()
            try:
                ck = self.kernel[(1,np_//self.bn)](*args, **kw)
            finally:
                self.config.compile_only = False
            elapsed = time.perf_counter()-start
            self.compile_seconds += elapsed
            cache = get_npu_cache_dir(ck)
            self.compiled[shape] = {'seconds': elapsed, 'kernel_hash': ck.hash,
                                    'npu_cache': cache, 'shape':list(shape)}
            # Backend IR and native ELF are persisted by the installed driver.
            if cache:
                dest = self.directory / 'artifacts' / ck.hash
                shutil.copytree(cache, dest, dirs_exist_ok=True)
                self.compiled[shape]['artifact_copy'] = str(dest)
            _COMPILED[cache_key] = self.compiled[shape]
            print(json.dumps({'event':'compiled',**self.compiled[shape]}),flush=True)

    def execute(self, record=False):
        torch = self.torch
        if fingerprint(self.target_config) != self.config_identity:
            raise Unsupported('executor configuration changed; create a new Executor')
        complete = set()
        for ti, task in enumerate(self.plan['tasks']):
            if not set(task['depends_on']) <= complete:
                raise RuntimeError('dependency was not completed')
            for oi, op in enumerate(task['operations']):
                t = time.perf_counter()
                ms, ns = slice(*op['m']), slice(*op['n'])
                dst = self.memory[op['output']]
                args = [self.memory[x] for x in op['inputs']]
                if op['kind'] == 'fill':
                    dst[ms,ns] = 0
                elif op['kind'] == 'matmul':
                    key = (ti,oi)
                    if key in self.k_followers:
                        continue
                    if key in self.merged_followers:
                        dst[ms,ns] += self.pending_merged.pop(key)
                        continue
                    a,b,c,(m,n,k),(mp,np_,kp) = self.calls[(ti,oi)]
                    ks = slice(*op['k'])
                    if op['inputs'][0] not in self.input_ids:
                        a[:m,:k].copy_(torch.from_numpy(args[0][ms,ks]))
                    if op['inputs'][1] not in self.input_ids:
                        b[:k,:n].copy_(torch.from_numpy(args[1][ks,ns]))
                    if key in self.k_groups:
                        group=self.k_groups[key];chunk_k=kp//len(group)
                        if op['inputs'][0] not in self.input_ids:a.zero_()
                        if op['inputs'][1] not in self.input_ids:b.zero_()
                        for index,gop in enumerate(group):
                            ks2=slice(*gop['k']);width=ks2.stop-ks2.start;offset=index*chunk_k
                            if op['inputs'][0] not in self.input_ids:
                                a[:m,offset:offset+width].copy_(torch.from_numpy(args[0][ms,ks2]))
                            if op['inputs'][1] not in self.input_ids:
                                b[offset:offset+width,:n].copy_(torch.from_numpy(args[1][ks2,ns]))
                    if key in self.chains:
                        got = self.chains[key].run([self._numpy(a),self._numpy(b),self._numpy(c)],
                            bo_key=f'{id(self)}:{self.generation}:{key}:{self.config_identity}',
                            static_indices={1} if key in self.shared_weights else set(),intermediate_indices={2},output_indices={2},
                            bound_buffers={1:self.shared_weights[key].bo} if key in self.shared_weights else None)
                        c.copy_(torch.from_numpy(np.asarray(got[2]).reshape(mp,np_)))
                        self.device_dispatches += 1
                        generation_key = (self.generation,key)
                        if key in self.shared_weights and generation_key not in self.staged_generations:
                            self.staged_generations.add(generation_key)
                            self.weight_uploads += 1
                            self.weight_sync_bytes_theoretical += b.numel()*b.element_size()
                    for row in range(0,mp if key not in self.chains else 0,self.bm):
                        self.kernel[(1,np_//self.bn)](a[row:row+self.bm],b,c[row:row+self.bm],
                            kp,np_,np_,K=kp,BLOCK_M=self.bm,BLOCK_N=self.bn,BLOCK_K=self.bk,RELU=False)
                        self.device_dispatches += 1
                    # The operation is an explicit sequential K partial. The
                    # host accumulation is FP32 and remains in program order.
                    if key in self.merges:
                        follower,other = self.merges[key]
                        original_n = op['n'][1]-op['n'][0]
                        self.pending_merged[follower] = c[:m,original_n:n].numpy().copy()
                        dst[ms,ns] += c[:m,:original_n].numpy()
                    else:
                        dst[ms,ns] += c[:m,:n].numpy()
                elif op['kind'] == 'relu':
                    dst[ms,ns] = np.maximum(args[0][ms,ns],0)
                elif op['kind'] == 'swiglu':
                    gate = args[0][ms,ns]
                    dst[ms,ns] = gate * np.exp(-np.logaddexp(0,-gate)) * args[1][ms,ns]
                if record:
                    self.trace.append(dict(task=task['id'],operation=oi,kind=op['kind'],
                        dependencies=list(task['depends_on']),m=op['m'],n=op['n'],
                        k=op.get('k'),host_wall_ms=(time.perf_counter()-t)*1000))
            complete.add(task['id'])
        return {name:self.memory[name].copy() for name in self.plan['outputs']}


def run_plan(plan, inputs, reference, output_dir, iterations=3, atol=0.015, rtol=0.06, target_config=None):
    """Run one candidate/seed, serializing all hardware work on /tmp/npu.lock."""
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/'program.json', plan)
    np.savez(directory/'inputs.npz', **inputs)
    np.savez(directory/'reference.npz', **reference)
    result = dict(schema='orbit.npu.result.v1',candidate_id=plan.get('candidate_id') if isinstance(plan,dict) else None,
        program_sha256=fingerprint(plan),target='npu2/AIE2P',
        path='Taskflow plan → Triton recipe → AIR → AIE ELF → XRT',
        direct_air_frontend=False, status='unsupported',
        timing_boundary='Complete graph host wall: input conversion of intermediates, BO staging, synchronous XRT wait, readback, CPU nodes and K accumulation; excludes compile and immutable weight preparation',
        device_time=None, performance_environment='shared machine; clocks not locked',
        numerical_contract='FP32 public semantics; BF16 operands lowered to BFP16 EBS8 MAC, FP32 accumulation; atol/rtol fixed before acceptance, not exact FP32',
        atol=atol,rtol=rtol,iterations=iterations)
    try:
        target_config = normalize_config(target_config)
        result['target_config'] = target_config
        tool = Path(os.environ.get('ORBIT_TASKFLOW_OPT',str(ROOT/'build-core/tools/taskflow-opt/taskflow-opt')))
        if not tool.is_file():
            raise Unsupported('independent semantic verifier is required: '+str(tool))
        command = [str(tool),str(directory/'program.json'),
            '--emit-plan='+str(directory/'verified-plan.json'),
            '--emit-mlir='+str(directory/'verified-taskflow.mlir')]
        verified = subprocess.run(command,capture_output=True,text=True,timeout=120)
        (directory/'semantic-validation.log').write_text(verified.stdout+verified.stderr)
        result['semantic_validation'] = dict(command=command,returncode=verified.returncode,
            verifier_sha256=hashlib.sha256(tool.read_bytes()).hexdigest())
        if verified.returncode:
            raise Unsupported('Taskflow semantic verifier rejected input; see semantic-validation.log')
        authenticated = json.loads((directory/'verified-plan.json').read_text())
        if authenticated != plan:
            raise Unsupported('core-normalized plan differs: stale or forged candidate identity')
        result['semantic_validation'].update(candidate_id=authenticated['candidate_id'],
            verified_taskflow_sha256=hashlib.sha256((directory/'verified-taskflow.mlir').read_bytes()).hexdigest())
        validate(plan, inputs)
        if not any(op['kind']=='matmul' for task in plan['tasks'] for op in task['operations']):
            raise Unsupported('no_npu_work: this adapter requires at least one device matmul')
        buffers = {b['id']:b for b in plan['buffers']}
        if set(reference) != set(plan['outputs']):
            raise Unsupported('reference must contain every declared output and no other storage')
        for name, value in reference.items():
            if list(value.shape) != buffers[name]['shape'] or not np.isfinite(value).all():
                raise Unsupported('reference has wrong shape or nonfinite values: '+name)
        result['target_validation'] = 'passed'
        if iterations < 3:
            raise Unsupported('at least three hot samples required')
        import torch
        from filelock import FileLock
        from triton.backends.amd_triton_npu.driver_scope import driver_scope
        from .profiling import install, load_seconds, load_breakdown
        install()
        torch.set_num_threads(4)
        os.environ.setdefault('TRITON_CACHE_DIR', str(TOY/'build/orbit-taskflow/triton-cache'))
        versions = {p:importlib.metadata.version(p) for p in ('triton-xdna','mlir-air','mlir-aie-no-rtti','llvm-aie','torch','numpy')}
        impl = dict(plan=plan,semantic_validation={key:result['semantic_validation'][key] for key in
            ('candidate_id','verifier_sha256','verified_taskflow_sha256')},
            common_hash=hashlib.sha256((ROOT/'scripts/taskflow_common.py').read_bytes()).hexdigest(),
            target='npu2',dtype='bf16-bfp16-f32',layout='contiguous-padded-row-major',
            recipe_hash=hashlib.sha256((Path(__file__).parent/'kernels.py').read_bytes()).hexdigest(),
            runtime_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            profiling_hash=hashlib.sha256((Path(__file__).parent/'profiling.py').read_bytes()).hexdigest(),
            schedule_hash=hashlib.sha256((RECIPES/'transform_matmul_aie2p.mlir').read_bytes()).hexdigest(),
            target_config=target_config,config=dict(block_m=128,block_n=256,block_k=64,runtime='xrt',format='elf'),versions=versions)
        result.update(implementation_sha256=fingerprint(impl),toolchain=versions)
        write_json(directory/'identity.json',impl)
        source_dir = directory/'adapter-source'
        source_dir.mkdir(exist_ok=True)
        for name in ('runtime.py','kernels.py','profiling.py'):
            shutil.copyfile(Path(__file__).parent/name,source_dir/name)
        shutil.copyfile(ROOT/'scripts/taskflow_common.py',source_dir/'taskflow_common.py')
        result['status'] = 'compile_failed'
        # This is the ONLY NPU lock. The Triton driver used here does not lock.
        with FileLock('/tmp/npu.lock',timeout=600), driver_scope('npu'), torch.inference_mode():
            initialization_start=time.perf_counter()
            executor = Executor(plan,inputs,directory,target_config)
            result['initialization_seconds']=time.perf_counter()-initialization_start
            write_json(directory/'target-plan.json',dict(schema='orbit.npu.target.v1',
                target_config=target_config,placement=executor.placements,execution_order=[t['id'] for t in plan['tasks']],
                lock='/tmp/npu.lock',implementation_sha256=result['implementation_sha256']))
            executor.compile()
            result.update(compile_seconds=executor.compile_seconds,
                first_weight_preparation_seconds=executor.first_weight_preparation_seconds,
                compiled_kernels=list(executor.compiled.values()))
            result['status'] = 'runtime_failed'
            before_load=load_seconds()
            before_breakdown=load_breakdown()
            start=time.perf_counter(); executor.execute(record=True)
            result['load_and_first_execution_seconds']=time.perf_counter()-start
            result['load_seconds']=load_seconds()-before_load
            result.update({k:v-before_breakdown[k] for k,v in load_breakdown().items()})
            result['load_timing_note']='Staged launcher steady-clock plus guarded NPUChain _open_runner host timer, both cover device/ELF/context/kernel loading. Chain compilation is complete before the timer. BO allocation/staging remains in first execution; host weight preparation is separate. Chain LRU reopen time, if any, is included in subsequent graph wall samples.'
            executor.execute() # excluded warmup after first session initialization
            samples=[]; checks=[]
            for index in range(iterations):
                start=time.perf_counter(); actual=executor.execute(); samples.append((time.perf_counter()-start)*1000)
                check = compare(actual,reference,atol,rtol)
                for name, metric in check['outputs'].items():
                    ref_rms = float(np.sqrt(np.mean(np.square(reference[name].astype(np.float64)))))
                    metric['reference_rms'] = ref_rms
                    metric['normalized_rms_error'] = metric.get('rms_abs',float('inf')) / max(ref_rms,1e-12)
                    metric['normalized_rms_limit'] = 0.03
                    metric['peak_scaled_abs_limit'] = 0.08
                    ref_peak = float(np.max(np.abs(reference[name])))
                    metric['peak_scaled_abs_error'] = metric.get('max_abs',float('inf')) / max(ref_peak,1e-12)
                    metric['passed'] = metric['passed'] and metric['normalized_rms_error'] <= 0.03 and metric['peak_scaled_abs_error'] <= 0.08 and (ref_peak != 0 or metric.get('max_abs') == 0)
                check['passed'] = all(x['passed'] for x in check['outputs'].values())
                checks.append(check)
                np.savez(directory/f'outputs-{index}.npz',**actual)
            result.update(host_wall_ms=samples,median_host_wall_ms=float(np.median(samples)),
                numerical_checks=checks,status='passed' if all(x['passed'] for x in checks) else 'numerical_failed',
                device_dispatches_per_graph=sum(x.get('dispatches',0) for x in executor.placements),
                cpu_placements=[x for x in executor.placements if x['device']=='cpu'],
                sequential_partial_sums='device nested semantic-K loop FP32 accumulator' if executor.k_groups else 'CPU FP32 in explicit plan K order',
                output_readbacks_per_graph=sum(x.get('readbacks',0) for x in executor.placements),
                persistent_weight_initial_sync_requests=executor.weight_uploads,
                persistent_weight_initial_sync_bytes_theoretical=executor.weight_sync_bytes_theoretical,
                hot_weight_staging_bytes_theoretical=sum(x.get('weight_bytes_theoretical',0)*x.get('dispatches',0) for x in executor.placements if x.get('weight_buffer')=='driver staging'),
                buffer_accounting_note='Counts follow NPUChain static_indices API: one initial sync per BO key; not hardware traffic counters. XRT BOs are system DRAM, not AIE SRAM.',
                input_snapshot_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in inputs.items()})
            write_json(directory/'first-execution-trace.json',executor.trace)
            executor.close()
    except Unsupported as exc:
        result.update(status='unsupported',error=str(exc))
    except Exception as exc:
        result.update(error=str(exc),traceback=traceback.format_exc())
        (directory/'failure.txt').write_text(result['traceback'])
    write_json(directory/'result.json',result)
    return result
