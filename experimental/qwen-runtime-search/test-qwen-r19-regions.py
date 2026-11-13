#!/usr/bin/env python3
"""Check dynamic region legality and compile real Qwen CPU covers.

This is a compilation/local arithmetic receipt, not complete-model numerical
admission. Floating-point differences are reported verbatim; no tolerance is
invented and no original whole-model gate is replaced by this fixture.
"""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--vendor',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compile',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    runtime=load(Path(__file__).with_name('qwen-r19-regions.py'),'r19_regions')
    source=json.loads(args.source.read_text());program=runtime.Program(source)
    import torch
    import numpy as np
    torch.set_num_threads(4)
    sys.path.insert(0,str(args.vendor/'examples/qwen25_3b_q4nx'))
    import config
    sys.path.insert(0,config._SHARED)
    from llama_prefill import LlamaPrefill
    class Timer:
        def track(self,*args):return contextlib.nullcontext()
    class Original:
        timer=Timer()
        ATTN_CHUNK=128
        def _device(self,*args):return 'cpu'
    original=Original()
    opmap={op['id']:op for op in source['operators']}
    def original_region(proof,inputs):
        values=dict(inputs)
        for name in proof['operators']:
            op=opmap[name];a=op['attributes'];xs=[values[v] for v in op['inputs']]
            if op['kind']=='rms-norm':ys=(LlamaPrefill._rms_norm(original,*xs,a['eps']),)
            elif op['kind']=='swiglu':ys=(LlamaPrefill._swiglu(original,*xs),)
            elif op['kind']=='neox-rope':ys=(LlamaPrefill._rope(original,xs[0],xs[1][:xs[0].shape[0]],a['heads']),)
            elif op['kind']=='causal-gqa':ys=(LlamaPrefill._attention(original,*xs,a['query_heads'],a['kv_heads'],a['head_dim'],scale=a['scale']),)
            elif op['kind']=='matmul':ys=(LlamaPrefill._matmul(original,*xs),)
            elif op['kind'] in ('add','broadcast-add'):ys=(xs[0]+xs[1],)
            elif op['kind']=='split-view':ys=xs[0].split(a['sizes'],dim=a['axis'])
            else:raise ValueError(op['kind'])
            values.update(zip(op['outputs'],ys))
        return {name:values[name] for name in proof['output_values']}
    covers=[
        ['prefill.L0.residual1','prefill.L0.ffn_norm'],
        ['prefill.L0.qkv_bias','prefill.L0.qkv_split','prefill.L0.q_rope','prefill.L0.k_rope'],
        ['prefill.L0.gate_split','prefill.L0.swiglu'],
        ['prefill.L0.residual2','prefill.L1.attn_norm'],
    ]
    rows=[]
    compiled_instances=[];registrations=[]
    for index,cover in enumerate(covers):
        eager=runtime.ExecutableRegion(program,cover)
        generator=torch.Generator().manual_seed(1900+index)
        inputs={name:torch.randn(spec['shape'],generator=generator,dtype=torch.float32)
                for name,spec in eager.proof['input_specs'].items()}
        expected=original_region(eager.proof,inputs)
        actual=eager.run(inputs)
        assert all(torch.equal(actual[name],expected[name]) for name in actual)
        # An actual cut must produce the same outputs using separate callables.
        split=runtime.ExecutableFission(program,cover,[[name] for name in cover])
        cut=split.run(inputs)
        assert all(torch.equal(cut[name],expected[name]) for name in cut)
        output_policy='retain-host' if eager.proof['exported_views'] else 'copy-host'
        if eager.proof['exported_views']:
            try:runtime.ExecutableRegion(program,cover,output_policy='copy-host')
            except ValueError as error:assert 'view alias' in str(error)
            else:raise AssertionError('Independent clone of exported aliases was accepted')
        copied=runtime.ExecutableRegion(program,cover,input_policy='stage-host-copy',output_policy=output_policy)
        copy_output=copied.run(inputs)
        assert all(torch.equal(copy_output[name],expected[name]) for name in copy_output)
        assert copied.events[-1]['copied_input_bytes']>0
        assert copied.events[-1]['copied_output_bytes']>0 or eager.proof['exported_views']
        row=dict(region=eager.proof,eager_exact_original=True,fission_exact_original=True,
                 staged_exact_original=True,eager_event=eager.events[-1],fission_event=split.events[-1])
        if args.compile:
            compiled=runtime.ExecutableRegion(program,cover,backend='inductor-cpu')
            result,proof=compiled.prepare(inputs,args.output/f'compile-{index}')
            checked=compiled.run(inputs)
            comparisons={}
            arrays={}
            for j,name in enumerate(checked):
                reference=expected[name];value=checked[name]
                diff=(value-reference).abs()
                comparisons[name]=dict(bitexact=bool(torch.equal(value,reference)),
                    finite=bool(torch.isfinite(value).all()),max_abs_error=float(diff.max()),
                    rms_error=float(torch.sqrt((diff*diff).mean())))
                arrays[f'reference_{j}']=reference.numpy();arrays[f'compiled_{j}']=value.numpy()
            np.savez(args.output/f'compiled-{index}-comparison.npz',**arrays)
            row.update(compilation=proof,compiled_comparisons=comparisons,
                       compiled_event=compiled.events[-1],
                       numerical_admission='pending unchanged complete-Qwen original comparator')
            registrations.append(runtime.registration(compiled,proof))
            compiled_instances.append((compiled,inputs))
        rows.append(row)
        (args.output/'summary.json').write_text(json.dumps(dict(covers=rows),indent=2)+'\n')
        print(json.dumps(dict(cover=index,eager_exact=True,fission_exact=True,
                              compiled_comparisons=row.get('compiled_comparisons'))),flush=True)
    # qkv_split -> attention has an omitted intermediate q_rope on another
    # path. It is connected yet non-convex and must never become one task.
    try:runtime.region(program,['prefill.L0.qkv_split','prefill.L0.k_rope','prefill.L0.attention'])
    except ValueError as error:assert 'Non-convex' in str(error)
    else:raise AssertionError('Non-convex cover accepted')
    try:runtime.region(program,['prefill.L0.k_rope','prefill.L0.kv_store'])
    except ValueError as error:assert 'effectful' in str(error)
    else:raise AssertionError('KV mutation admitted without effect lowering')
    # Covers are discovered from graph edges; no list of known FFN patterns.
    enumerated=list(runtime.enumerate_regions(program,max_operators=2))
    pairs={tuple(row['operators']) for row in enumerated}
    assert tuple(covers[0]) in pairs and tuple(covers[2]) in pairs and tuple(covers[3]) in pairs
    (args.output/'enumeration.json').write_text(json.dumps(dict(max_operators=2,
        min_operators=2,count=len(enumerated),regions=enumerated),indent=2)+'\n')
    if args.compile:
        # Preparing later covers must not invalidate earlier prepared kernels
        # or move compiler work into a measured request.
        from unittest import mock
        from torch._inductor.graph import GraphLowering
        with mock.patch.object(GraphLowering,'save_output_code',side_effect=AssertionError('Request recompiled a prepared region')):
            for compiled,inputs in compiled_instances:compiled.run(inputs)
        # Cross-layer residual+norm covers have identical executable structure.
        # One native template must serve their actual separate source covers.
        first=compiled_instances[0][0]
        for layer in range(1,36):
            other=runtime.ExecutableRegion(program,[f'prefill.L{layer}.residual1',f'prefill.L{layer}.ffn_norm'],backend='inductor-cpu')
            assert other.kernel is first.kernel
        (args.output/'pending-registration.json').write_text(json.dumps(dict(
            schema='orbit.qwen-r19-cpu-region-registration.v1',implementations=registrations,
            admission='pending unchanged complete-Qwen numeric gate and fresh region-call prices'),indent=2)+'\n')
    print(json.dumps(dict(legality_checks=True,enumerated_pairs=len(enumerated))),flush=True)


if __name__=='__main__':main()
