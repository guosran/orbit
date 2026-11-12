#!/usr/bin/env python3
"""Real device GEMM tiling producer; construction never opens the NPU.

Output cores partition one GEMM. They are not independent whole-GEMM replicas.
The device grid is submitted once, without a host tile loop.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path


def identity(p):
    p = Path(p).resolve(strict=True)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def load_kernels(root):
    p = root / 'examples/llm_q4nx/kernels.py'
    spec = importlib.util.spec_from_file_location('orbit_gemm_kernels', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_chain(root, output, m, n, k, bm=128, bn=256, l1m=64,
               l1n=64, l2k=64, l1k=64):
    """Build an unopened original-kernel chain with an independently generated schedule.

    Runtime callers pass padded BF16 A/B and FP32 C, retain B with
    static_indices={1}, and keep the original chain.run BO identity contract.
    """
    if min(m, n, k, bm, bn, l1m, l1n, l2k, l1k) <= 0:
        raise ValueError('nonpositive extent')
    if bm % l1m or bn % l1n or k % l2k or l2k % l1k:
        raise ValueError('tiles must divide device blocks and exact reduction')
    if bm // l1m > 8 or bn // l1n > 4 or l1m * l1n > 8192:
        raise ValueError('herd or per-core DMA extent exceeds NPU2 contract')
    if l2k * n > 2**20:
        raise ValueError('weight DMA stride exceeds existing 2^20-word bound')
    import torch
    from triton.backends.amd_triton_npu.multilaunch import NPUChain
    from triton.backends.amd_triton_npu.matmul_transform import generate_matmul_transform
    kernel = load_kernels(root)
    mp, np_ = math.ceil(m / bm) * bm, math.ceil(n / bn) * bn
    if l2k * np_ > 2**20:
        raise ValueError('padded weight DMA stride exceeds 2^20 words')
    stride = kernel.unaliased_stride(k)
    script = output / 'schedule.mlir'
    script.write_text(generate_matmul_transform(l1_m=l1m, l1_n=l1n,
        l2_k=l2k, l1_k=l1k, contract_input_type='bf16', k_loop=True))
    chain = NPUChain('orbit_qwen_gemm', air_project_path=str(output / 'air_project'))
    chain.add(kernel._matmul_kernel, grid=(mp // bm, np_ // bn),
        arg_map={0: 0, 1: 1, 2: 2},
        args=(torch.empty((mp, stride), dtype=torch.bfloat16),
              torch.empty((k, np_), dtype=torch.bfloat16),
              torch.empty((mp, np_), dtype=torch.float32), stride, np_, np_),
        constexprs=dict(K=k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=l2k),
        transform_script=str(script))
    contract = dict(schema='orbit.qwen-npu-gemm-implementation.v1',
        source=identity(root / 'examples/llm_q4nx/kernels.py'), source_symbol='_matmul_kernel',
        producer=identity(__file__), shape=dict(M=m, N=n, K=k),
        padded_shape=dict(M=mp, N=np_, K=k, A_stride=stride),
        tiling=dict(L2_M=bm, L2_N=bn, L2_K=l2k, L1_M=l1m, L1_N=l1n, L1_K=l1k),
        resource=dict(device='npu2', compute_herd_shape=[bm // l1m, bn // l1n],
            parallel_output_core_partitions=(bm // l1m)*(bn // l1n),
            task_instance_replicas=1, exclusive_device_invocation=True,
            placement='aircc places the actual emitted herd; compiled artifact is authoritative'),
        abi=[dict(index=0, dtype='bf16', shape=[mp,stride], access='read'),
             dict(index=1, dtype='bf16', shape=[k,np_], access='read'),
             dict(index=2, dtype='f32', shape=[mp,np_], access='write')],
        numerical_contract='original BF16-input Triton dot, FP32 accumulation; ordered L1 K tiles; device validation required for equivalence',
        residence=dict(weight_static_indices=[1], requires_same_weight_identity=True),
        unsupported=dict(independent_gemm_replication='no independent concurrent kernel queues/BOs implemented',
                         arbitrary_core_placement='current aircc owns herd placement'),
        schedule=identity(script), device_loaded=False)
    return chain, contract


def lower(chain):
    from triton.backends.amd_triton_npu.multilaunch import MultiLaunchBuilder
    builder = MultiLaunchBuilder(chain.name, air_project_path=chain.air_project_path)
    for kernel, grid, argmap, script, args, constexprs, sizes in chain._specs:
        builder.add_op(chain._capture_ttshared(kernel, grid, args, constexprs),
            grid, argmap, transform_script=script, actual_sizes=sizes)
    return builder


def replicated_air(builder, replicas, bm, bn, l1m, l1n, column_offset=0):
    """Two disjoint output-N instances inside one launch and one device segment.

    The source has no dependency between instance bodies. AIR dependency lowering
    establishes each instance's internal DMA/compute ordering. Explicit physical
    x_loc/y_loc annotations reserve disjoint core rectangles, not host threads.
    """
    if replicas != 2 or len(builder.ops) != 1:
        raise ValueError('this lowering requires exactly two equal N shards')
    from triton.backends.amd_triton_npu import stitching
    text = builder.ops[0].air_text
    start = text.index('air.segment ')
    opening = text.index('{', start)
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    body = text[opening + 1:end - 1]
    signature = text[start:opening]
    assignments = re.findall(r'(%[A-Za-z0-9_]+)=(%[A-Za-z0-9_]+)', signature)
    if len(assignments) != 5:
        raise ValueError('original kernel segment ABI changed')
    originals = [a for a, _ in assignments]
    bodies, maps = [], []
    for r in range(replicas):
        prefix = f'replica{r}'
        branch = stitching._rename_all(body, prefix, set())
        target = ['%pm', '%pn', f'%b{r}', '%a', f'%c{r}']
        for old, new in zip(originals, target):
            renamed = '%' + prefix + '_' + old[1:]
            branch = re.sub(re.escape(renamed) + r'(?![A-Za-z0-9_])', new, branch)
        # The three prologue/compute/epilogue herds of one instance share cores.
        branch = re.sub(r'attributes \{(prologue_herd|compute_herd|epilogue_herd)\}',
            lambda m: 'attributes {' + m.group(1) +
                f', x_loc = {column_offset+r*(bm//l1m)} : i64, y_loc = 2 : i64' + '}', branch)
        bodies.append(branch)
        maps += [stitching._rename_all(x, prefix, set()) for x in stitching._extract_affine_maps(text)]
    # Launch bounds are read from the original producer, not inferred from names.
    launch = re.search(r'air.launch.*?in \((.*?)\) args', text).group(1)
    extent_names = re.findall(r'=%([A-Za-z0-9_]+)', launch)
    constants = dict(re.findall(r'%([A-Za-z0-9_]+) = arith.constant (\d+) : index', text))
    gx, gy, gz = [int(constants[n]) for n in extent_names]
    return '\n'.join(maps) + f"""
module {{
  func.func @orbit_qwen_gemm_replicated(%arg0: memref<*xbf16>,
      %arg1: memref<*xbf16>, %arg2: memref<*xf32>,
      %arg3: memref<*xbf16>, %arg4: memref<*xf32>) {{
    %gx = arith.constant {gx} : index
    %gy = arith.constant {gy} : index
    %gz = arith.constant {gz} : index
    air.launch (%pm0, %pn0, %pz0) in (%sx=%gx, %sy=%gy, %sz=%gz)
      args(%la=%arg0, %lb0=%arg1, %lc0=%arg2, %lb1=%arg3, %lc1=%arg4)
      : memref<*xbf16>, memref<*xbf16>, memref<*xf32>, memref<*xbf16>, memref<*xf32> {{
      air.segment @replicated args(%pm=%pm0, %pn=%pn0, %a=%la,
          %b0=%lb0, %c0=%lc0, %b1=%lb1, %c1=%lc1)
          : index, index, memref<*xbf16>, memref<*xbf16>, memref<*xf32>,
            memref<*xbf16>, memref<*xf32> {{
{''.join(bodies)}
      }}
    }}
    return
  }}
}}
"""


def compiler_evidence(output):
    paths = sorted((output / 'air_project').glob('aie.*.mlir'))
    cores = set()
    for path in paths:
        cores.update(tuple(map(int, pair)) for pair in
            re.findall(r'%core_(\d+)_(\d+) = aie.core', path.read_text()))
    placed = sorted((output / 'air_project').glob('placed.*.mlir'))
    return dict(aie_sources=[identity(p) for p in paths],
                placed_sources=[identity(p) for p in placed],
                actual_core_coordinates=[list(c) for c in sorted(cores)])


class CompiledGemm:
    """Execute an emitted ELF; only run() opens the device.

    Immutable weights are padded/partitioned once for a stage. The source weight
    object is retained to prevent accidental address reuse in the BO cache key.
    Each call measures input preparation, native inclusive service, and output
    assembly separately; their sum equals the reported complete call wall time.
    """
    def __init__(self, artifact):
        self.directory = Path(artifact).resolve()
        self.record = json.loads((self.directory/'implementation.json').read_text())
        for field in ('source_air', 'compiled_binary'):
            expected = self.record[field]
            if identity(expected['path']) != expected:
                raise ValueError('artifact hash mismatch: ' + field)
        self.runner = None
        self.weights = {}
        self.events = []

    def run(self, x, w, stage_key):
        import numpy as np
        from ml_dtypes import bfloat16
        import time
        start = time.perf_counter_ns()
        if self.runner is None:
            from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner
            self.runner = MultiLaunchRunner(self.record['compiled_binary']['path'],
                                            self.record['kernel_name'])
        opened = time.perf_counter_ns()
        shape, pad = self.record['shape'], self.record['padded_shape']
        x, w = np.asarray(x), np.asarray(w)
        m,n,k = shape['M'],shape['N'],shape['K']
        if x.shape != (m,k) or w.shape != (k,n):
            raise ValueError('runtime shape differs from compiled operator shape')
        replicas = self.record['resource']['task_instance_replicas']
        key = (str(stage_key), id(w))
        if key not in self.weights:
            shards = []
            for r in range(replicas):
                b = np.zeros((k,pad['N']), dtype=bfloat16)
                b[:,:n//replicas] = w[:,r*(n//replicas):(r+1)*(n//replicas)]
                shards.append(b)
            self.weights[key] = (w, shards)
        original, shards = self.weights[key]
        if original is not w:
            raise ValueError('weight cache identity conflict')
        a = np.zeros((pad['M'],pad['A_stride']), dtype=bfloat16)
        a[:m,:k] = x
        inputs = [a]
        outputs = set()
        for b in shards:
            inputs += [b, np.empty((pad['M'],pad['N']),dtype=np.float32)]
            outputs.add(len(inputs)-1)
        prepared = time.perf_counter_ns()
        got = self.runner.run(inputs, bo_key=str(key),
            static_indices=set(range(1,len(inputs),2)),
            intermediate_indices=outputs, output_indices=outputs)
        completed = time.perf_counter_ns()
        result = np.concatenate([got[i].reshape(pad['M'],pad['N'])[:m,:n//replicas]
                                 for i in sorted(outputs)], axis=1).copy()
        end = time.perf_counter_ns()
        event = dict(wall_ns=end-start, device_open_ns=opened-start,
                     host_prepare_ns=prepared-opened,
                     native_inclusive_service_ns=completed-prepared,
                     host_assemble_ns=end-completed, replicas=replicas)
        assert event['wall_ns'] == sum(event[q] for q in
            ('device_open_ns','host_prepare_ns','native_inclusive_service_ns','host_assemble_ns'))
        self.events.append(event)
        return result

    def close(self):
        if self.runner is not None:
            self.runner.unload()
            self.runner = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--triton-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    for key, val in [('m',5),('n',2560),('k',2048),('bm',128),('bn',256),
                     ('l1m',64),('l1n',64),('l2k',64),('l1k',64)]:
        ap.add_argument('--'+key, type=int, default=val)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--replicas', type=int, choices=[1, 2], default=1)
    ap.add_argument('--column-offset', type=int, default=0)
    a = ap.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.n % a.replicas:
        raise ValueError('replicated N shards must be equal')
    if a.column_offset < 0 or a.column_offset + a.replicas * (a.bm // a.l1m) > 8:
        raise ValueError('replicated herds exceed physical column capacity')
    chain, record = make_chain(a.triton_root.resolve(), a.output,
        a.m,a.n//a.replicas,a.k,a.bm,a.bn,a.l1m,a.l1n,a.l2k,a.l1k)
    builder = lower(chain)
    air = a.output / 'source.air.mlir'
    air.write_text(builder.build_module_text() if a.replicas == 1 else
        replicated_air(builder, a.replicas, a.bm, a.bn, a.l1m, a.l1n, a.column_offset))
    if a.replicas == 1:
        air.write_text(re.sub(r'attributes \{(prologue_herd|compute_herd|epilogue_herd)\}',
            lambda m: 'attributes {'+m.group(1)+f', x_loc = {a.column_offset} : i64, y_loc = 2 : i64'+'}', air.read_text()))
    record['resource']['column_offset'] = a.column_offset
    if a.replicas == 2:
        record['shape']['N'] = a.n
        record['resource']['task_instance_replicas'] = 2
        record['resource']['parallel_output_core_partitions'] *= 2
        record['resource']['instance_rectangles'] = [dict(x=a.column_offset+r*(a.bm//a.l1m), y=2, width=a.bm//a.l1m, height=a.bn//a.l1n) for r in range(2)]
        record['resource']['replica_semantics'] = 'two independent N-shard GEMM task instances; disjoint weights/results, shared read-only A'
        record['abi'] += [dict(record['abi'][1], index=3), dict(record['abi'][2], index=4)]
        record['residence']['weight_static_indices'] = [1,3]
        record['host_boundary'] = dict(weight_partition='B[:,0:N/2] and B[:,N/2:N], padded separately', output_assembly='concatenate C0[:M,:N/2],C1[:M,:N/2] along N; measurable host copy', activation='one read-only shared A BO')
        record['unsupported'].pop('independent_gemm_replication')
    for index, op in enumerate(builder.ops):
        (a.output / f'op{index}.air.mlir').write_text(op.air_text)
    record['source_air'] = identity(air)
    record['actual_air_herds'] = [s.strip() for s in air.read_text().splitlines() if 'air.herd' in s]
    record['actual_abi'] = builder.combined_arg_types() if a.replicas == 1 else ['memref<*xbf16>','memref<*xbf16>','memref<*xf32>','memref<*xbf16>','memref<*xf32>']
    if a.compile:
        os.chdir(a.output)
        from triton.backends.amd_triton_npu.driver import _aircc_compile
        result = _aircc_compile(str(air), 'elf', 'npu2', str(a.output/'air_project'))
        binary, name = result['elf_path'], result['elf_kernel_name']
        record['compiled_binary'], record['kernel_name'] = identity(binary), name
        record['physical_evidence'] = compiler_evidence(a.output)
        expected = {(a.column_offset+r*(a.bm//a.l1m)+x, 2+y) for r in range(a.replicas) for x in range(a.bm//a.l1m) for y in range(a.bn//a.l1n)}
        observed = {tuple(c) for c in record['physical_evidence']['actual_core_coordinates']}
        if observed != expected:
            raise ValueError(f'compiled physical placement differs: {observed} vs {expected}')
    (a.output / 'implementation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),compiled=a.compile,herds=record['actual_air_herds'])))


if __name__ == '__main__':
    main()
