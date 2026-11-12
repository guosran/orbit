#!/usr/bin/env python3
"""Bind compiled Qwen FFN fusion to source covers and real packing/lifetimes.

The output is a pending runtime registration, never an executable search
catalogue: numerical validation and measured prices are mandatory admission
steps. Packing describes actual add_layer copies, not a fictitious SRAM cache.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_identity(path):
    p=Path(path).resolve(strict=True)
    return dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def register(source,manifest,layers,retention):
    if source['source_id']!=digest({k:v for k,v in source.items() if k!='source_id'}):
        raise ValueError('Invalid original source identity')
    if not manifest.get('compiled') or manifest.get('sram_residency_claim'):
        raise ValueError('Require actual compiled DDR implementation')
    if retention not in ('retain','reload','release-after-call'):
        raise ValueError('Unknown actual BO lifecycle')
    if file_identity(manifest['artifact'])['sha256']!=manifest['artifact_sha256']:
        raise ValueError('Compiled artifact changed')
    for row in manifest['source_files']:
        if file_identity(row['file'])!=row:
            raise ValueError('Backend producer/source changed; re-emit compile proof')
    operators={op['id']:op for op in source['operators']}
    values={v['id']:v for v in source['values']}
    r,h=manifest['row_tile'],manifest['padded_hidden']
    result=[]
    for layer in layers:
        if type(layer)!=int or not 0<=layer<36:raise ValueError('Invalid source layer')
        prefix=f'prefill.L{layer}.'
        covered=[prefix+role for role in ('gate_up','gate_split','swiglu','down')]
        if any(name not in operators for name in covered):raise ValueError('Missing original arithmetic source')
        produced={v for name in covered for v in operators[name]['outputs']}
        inputs=list(dict.fromkeys(v for name in covered for v in operators[name]['inputs'] if v not in produced))
        consumed_outside={v for op in source['operators'] if op['id'] not in covered for v in op['inputs']}
        outputs=[v for name in covered for v in operators[name]['outputs'] if v in consumed_outside or values[v]['role']=='output']
        weight_prefix=f'weights.L{layer}.'
        packing=[]
        for index,source_value,offset,extent,physical in [
            (1,weight_prefix+'gate_up',[0,0],[2048,11008],[2048,h]),
            (2,weight_prefix+'gate_up',[0,11008],[2048,11008],[2048,h]),
            (3,weight_prefix+'down',[0,0],[11008,2048],[h,2048])]:
            if source_value not in inputs:raise ValueError('Packing must bind a true external source input')
            packing.append(dict(buffer_index=index,logical_value=source_value,
                offsets=offset,extents=extent,physical_shape=physical,
                physical_strides=[physical[1],1],dtype='bf16',padding='zero',
                allocation_id=f'ffn.L{layer}.weight{index}.{retention}.prepared',memory='runtime-abi',
                bytes=math.prod(physical)*2,source_value=source_value,source_offsets=offset,
                source_extents=extent,physical_offsets=[0,0],strides=[physical[1],1],
                action='bitchecked source BF16 slice copy followed by release of original gate_up/down tensors',
                logical_copy_bytes=math.prod(extent)*2,allocated_bytes=math.prod(physical)*2,
                timing_boundary='preparation outside request; subsequent host-to-BO staging inside request',
                lifetime='prepared host tensors until close',
                replaces_source_physical_copy=True,
                host_device_same_allocation=retention=='retain'))
        weights_bytes=sum(p['allocated_bytes'] for p in packing)
        scratch_bytes=sum(b['bytes'] for b in manifest['buffers'] if b['index'] in (4,5,6,7))
        all_bo_bytes=sum(b['bytes'] for b in manifest['buffers'])
        runtime_config=dict(layer=layer,rows=r,block_m=manifest['block_m'],block_n=manifest['block_n'],
            split_order=manifest['split_order'],retention=retention)
        variant_id=digest(dict(artifact=manifest['artifact_sha256'],source_id=source['source_id'],runtime_config=runtime_config))
        retained_bytes=all_bo_bytes+(all_bo_bytes-weights_bytes if retention=='retain' else all_bo_bytes if retention=='reload' else 0)
        allocations={}
        for fragment in packing:
            allocations[fragment['allocation_id']]=dict(memory='runtime-abi',bytes=fragment['bytes'])
        for buf in manifest['buffers']:
            index=buf['index']
            if index not in (1,2,3):
                allocations[f'ffn.L{layer}.{retention}.host-arg{index}']=dict(memory='runtime-abi',bytes=buf['bytes'])
            if retention!='release-after-call' and not (retention=='retain' and index in (1,2,3)):
                allocations[f'ffn.L{layer}.{retention}.device-arg{index}']=dict(memory='runtime-abi',bytes=buf['bytes'])
        if sum(row['bytes'] for row in allocations.values())!=retained_bytes:
            raise ValueError('Actual allocation list does not conserve long-lived storage')
        result.append(dict(id='r18-ffn-'+variant_id,source_id=source['source_id'],operators=covered,
            input_values=inputs,output_values=outputs,artifact_identity=manifest['artifact_sha256'],
            entrypoint=manifest['entrypoint'],device='npu',resource='npu.device',memory='runtime-abi',
            wall_ms=None,executable=False,admission_status='compiled-pending-numerical-gate-and-local-price',
            launch_count=math.ceil(values[operators[covered[0]]['inputs'][0]]['shape'][0]/r),
            visibility_policy='host-materialized',cost_boundary='inclusive-region-wall',
            transport_ownership='implementation-inclusive',runtime_config=runtime_config,
            execution_variant=dict(resource_demands={'npu.device':1,'npu.command':1,'cpu.host':1},resource_shape=[],replicas=1,
                runtime_context_identity=digest(dict(artifact=manifest['artifact_sha256'],row_tile=r,split_order=manifest['split_order'])),
                dataflow='single-dispatch-firmware-sequenced-DDR-FFN',residence_policy=retention,
                implementation_parameters=dict(row_tile=r,block_m=manifest['block_m'],block_n=manifest['block_n'],padded_hidden=h),
                work_distribution='single-invocation',
                temporary_bytes={'runtime-abi':all_bo_bytes if retention=='release-after-call' else 0},
                resident_allocations=allocations,
                dma_overlap_supported=False),
            storage=dict(memory='device-ddr-bo',sram_residency_claim=False,
                prepared_weight_copies=packing,
                source_weight_allocation_aliases={fragment['allocation_id']:fragment for fragment in packing},
                staging='retain binds actual shared BO and cache-syncs once; reload uploads every call; release-after-call allocates/uploads/zeroes every call',
                activation_copy_bytes_per_row_tile=r*2048*2,
                output_copy_bytes_per_row_tile=r*2048*4,
                first_call_scratch_zero_upload_bytes=scratch_bytes,
                bo_lifetime='synchronous call completion and clone' if retention=='release-after-call' else 'layer key until close',
                host_prepared_weights_bytes=weights_bytes,
                device_bo_weights_bytes=weights_bytes,
                host_prepared_and_argument_array_bytes=all_bo_bytes,
                device_bo_total_bytes=all_bo_bytes,
                actual_long_lived_bytes=retained_bytes,
                host_device_weights_same_allocation=retention=='retain',
                capacity_accounting_requires_source_weight_physical_replacement=True,
                capacity_fit_claim=False)))
    if len(set(layers))!=len(layers):raise ValueError('Repeated source layer')
    runtime_root=next(Path(row['file']).parents[2] for row in manifest['source_files'] if Path(row['file']).name=='qwen-r18-ffn.py')
    runtime_files=[file_identity(runtime_root/'scripts'/name) for name in ('qwen-r18-runtime.py','run-qwen-r18-runtime.py','qwen-node-runtime.py')]
    runtime_files.append(file_identity(__file__))
    return dict(schema='orbit.qwen-r18-pending-runtime-registration.v1',source_id=source['source_id'],
                runtime_files=runtime_files,
                compiled_manifest=manifest,implementations=result,
                admission='not a search catalogue; never replace unknown price/validation with zero')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',default='all')
    p.add_argument('--retention',choices=['retain','reload','release-after-call'],default='retain')
    a=p.parse_args();layers=list(range(36)) if a.layers=='all' else [int(x) for x in a.layers.split(',')]
    result=register(json.loads(a.source.read_text()),json.loads(a.manifest.read_text()),layers,a.retention)
    result['execution_registration_identity']=digest(result)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(registrations=len(result['implementations']),identity=result['execution_registration_identity'])))


if __name__=='__main__':main()
