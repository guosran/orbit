#!/usr/bin/env python3
"""Run complete original Qwen with registered gate/up heterogeneous fission.

The original 36-layer prefill, residuals, KV handoff, and autoregressive decode
remain in the frozen runner. Selected gate_up source operations are evaluated
as independent CPU/NPU halves by the owned ready executor, followed by concat.
The preparation plan must retain the original BF16 gate_up weight (CPU storage
choice); the separate source-bound r19 spec defines the actual mixed execution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=False,allow_abbrev=False)
    p.add_argument('--frozen-runtime',type=Path,required=True)
    p.add_argument('--ready-spec',type=Path,required=True)
    args,rest=p.parse_known_args()
    config=json.loads(args.ready_spec.read_text())
    if config['schema']!='orbit.qwen-r19-ready-spec.v1':raise ValueError('Unknown ready spec')
    owned=Path(__file__).parent
    ready=module(owned/'qwen-r19-ready.py','orbit_r19_ready')
    executor_class=ready.ReadyExecutor
    device_owner=ThreadPoolExecutor(max_workers=1,thread_name_prefix='orbit-r19-whole-qwen-npu-owner')
    ready.ReadyExecutor=lambda:executor_class(worker=device_owner)
    xrt=module(owned/'qwen-r19-xrt.py','orbit_r19_private_xrt')
    routing=module(owned/'routed-r18-runtime.py','orbit_r19_preparation_routing')
    runner=module(args.frozen_runtime/'scripts/run-qwen-r18-runtime.py','orbit_r19_original_runner')
    loader=runner.module
    decode_ready=module(owned/'qwen-r19-decode-ready.py','orbit_r19_decode_ready')
    source_ready=module(owned/'qwen-r19-source-ready.py','orbit_r19_source_ready')
    cpu_regions=module(owned/'qwen-r19-regions.py','orbit_r19_cpu_regions')
    option=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    option.add_argument('--source',type=Path,required=True);option.add_argument('--outputs',type=int,default=128)
    request,_=option.parse_known_args(rest)
    prefix=json.loads(request.source.read_text())
    whole_binding=config.get('whole_source')
    if config.get('source_ready',False) and not whole_binding:raise ValueError('Whole source-ready execution requires an actual full-source manifest binding')
    ordering_identity=config['source_id']
    if whole_binding:
        if set(whole_binding)!={'file','sha256','source_id'} or sha(whole_binding['file'])!=whole_binding['sha256']:raise ValueError('Whole source manifest bytes changed')
        whole=json.loads(Path(whole_binding['file']).read_text())
        canonical=hashlib.sha256(json.dumps({k:v for k,v in whole.items() if k!='source_id'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if whole['source_id']!=whole_binding['source_id'] or canonical!=whole['source_id']:raise ValueError('Whole source canonical identity mismatch')
        if whole['operators'][:len(prefix['operators'])]!=prefix['operators'] or whole['values'][:len(prefix['values'])]!=prefix['values']:raise ValueError('Whole source prefix differs from actual executable prefill source')
        bindings={row['id']:row for row in whole['bindings']}
        if any(bindings.get(row['id'])!=row for row in prefix['bindings']):raise ValueError('Whole source prefix bindings differ')
        checked=set()
        for row in whole['bindings']:
            if row['file'] not in checked:
                if sha(row['file'])!=row['sha256']:raise ValueError('Actual full-source binding changed: '+row['file'])
                checked.add(row['file'])
        ordering_identity=whole['source_id']
        del whole
    tokens=next(v['shape'][0] for v in prefix['values'] if v['id']=='input.tokens')
    last_position=tokens+request.outputs-2
    def load(path,name):
        adapter=loader(path,name)
        if Path(path).name=='qwen-r18-decode.py' and config.get('decode_rope_prefetch',False):
            old_profile=adapter.ProfiledDecoder
            adapter.ProfiledDecoder=lambda decoder:decode_ready.make_decoder(decoder,old_profile,ready,last_position)
            return adapter
        if Path(path).name!='qwen-r18-runtime.py':return adapter
        routing.adapt(adapter)
        original=adapter.make_prefiller_class
        def make(base,source,plan):
            if source['source_id']!=config['source_id']:raise ValueError('Ready spec source mismatch')
            byid={v['id']:v for v in source['operators']}
            implementations={}
            for operator,row in config['operators'].items():
                if operator not in byid or byid[operator]['kind']!='matmul' or not operator.endswith('.gate_up'):
                    raise ValueError('Only source-verified gate_up fission is registered')
                if plan['placements'][operator]!='cpu' or operator in plan['r18']['gemm_variants']:
                    raise ValueError('Mixed fission must retain original BF16 source weights')
                if row['cpu_half'] not in ('gate','up'):raise ValueError('Invalid half placement')
                if sha(Path(row['directory'])/'implementation.json')!=row['manifest_sha256']:raise ValueError('Half artifact manifest drift')
                impl=xrt.PrivateGemm(row['directory'])
                if impl.record['shape']!={'M':5,'N':11008,'K':2048}:raise ValueError('Only exact registered P5 half projection supported')
                implementations[operator]=impl
            prefiller=original(base,source,plan)
            if config.get('source_ready',False):
                return source_ready.make_class(prefiller,base,source,plan,config,ready,xrt,cpu_regions)
            class ReadyPrefiller(prefiller):
                def _matmul(self,h,w,backend=None,**kwargs):
                    layer=self._orbit_active_layer
                    operator=f'prefill.L{layer}.gate_up'
                    if operator not in implementations or self._w[layer]['gate_up'] is not w:
                        return super()._matmul(h,w,backend=backend,**kwargs)
                    import torch
                    import numpy as np
                    from ml_dtypes import bfloat16
                    row=config['operators'][operator];cpu_half=row['cpu_half'];gpu_half='up' if cpu_half=='gate' else 'gate'
                    if tuple(w.shape)!=(2048,22016):raise ValueError('Source gate/up shape drift')
                    halves=dict(gate=w[:,:11008],up=w[:,11008:])
                    trace=[]
                    def evaluate():
                        def cpu(values):
                            return {cpu_half:base._matmul(self,values['h'],halves[cpu_half],backend='cpu')}
                        def device(values):
                            a=values['h'].to(torch.bfloat16).contiguous().view(torch.uint16).numpy().view(bfloat16)
                            b=halves[gpu_half].contiguous().view(torch.uint16).numpy().view(bfloat16)
                            native=implementations[operator]
                            value=native.run(a,b,operator+'.'+gpu_half)
                            return ready.Result({gpu_half:torch.from_numpy(value)},[native.events[-1]])
                        def concat(values):return {'gate_up':torch.cat((values['gate'],values['up']),dim=1)}
                        result=ready.ReadyExecutor().run([
                            ready.Operation(operator+'.'+cpu_half,['h'],[cpu_half],'cpu',cpu),
                            ready.Operation(operator+'.'+gpu_half,['h'],[gpu_half],'npu',device),
                            ready.Operation(operator+'.assemble',['gate','up'],['gate_up'],'cpu',concat)],{'h':h},['gate_up'])
                        trace.append(result['trace']);return result['values']['gate_up']
                    output=self._record(operator,evaluate)
                    event=self.orbit_trace[-1]
                    event.update(ready_execution=trace[0],source_transformation='gate_up output-column fission; original BF16 inputs/F32 accumulation; concatenate before original split/SwiGLU',
                        source_ordering_identity=source['source_id'],implementation=config['operators'][operator],
                        execution_policy='source-dependency-ready')
                    return output
            return ReadyPrefiller
        adapter.make_prefiller_class=make
        return adapter
    runner.module=load
    proof_files=[Path(__file__),owned/'qwen-r19-ready.py',owned/'qwen-r19-xrt.py',owned/'qwen-r19-source-ready.py',owned/'qwen-r19-regions.py',owned/'qwen-r19-decode-ready.py',owned/'routed-r18-runtime.py']
    proof=dict(schema='orbit.qwen-r19-owned-runtime-proof.v1',execution_policy='source-dependency-ready',source_ordering_identity=ordering_identity,prefill_source_id=config['source_id'],whole_source=whole_binding,ready_spec=dict(file=str(args.ready_spec.resolve()),sha256=sha(args.ready_spec)),code_files=[dict(file=str(path.resolve()),sha256=sha(path)) for path in proof_files])
    proof['async_runtime_proof_identity']=hashlib.sha256(json.dumps(proof,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    config['async_runtime_proof_identity']=proof['async_runtime_proof_identity']
    target=argparse.ArgumentParser(add_help=False,allow_abbrev=False);target.add_argument('--output',type=Path,required=True)
    destination,_=target.parse_known_args(rest)
    pending=destination.output.with_name(destination.output.name+'.runtime-proof.json')
    with pending.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
    sys.argv=[str(args.frozen_runtime/'scripts/run-qwen-r18-runtime.py'),*rest]
    try:runner.main()
    finally:device_owner.shutdown(wait=True)
    contract_path=destination.output/'contract.json'
    contract=json.loads(contract_path.read_text());contract['owned_runtime_proof']=proof
    contract_path.write_text(json.dumps(contract,indent=2)+'\n')
    (destination.output/'runtime-proof.json').write_text(json.dumps(proof,indent=2)+'\n')


if __name__=='__main__':main()
