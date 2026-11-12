#!/usr/bin/env python3
"""Bind real compiled GEMMs to exact Qwen source nodes and execution resources.

A dyadic hardware correctness gate admits the implementation arithmetic only.
Prices must come from complete-Qwen close-after-call hooks; standalone warm
persistent-context times are never used as source-node prices.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_identity(path):
    path=Path(path).resolve(strict=True)
    return dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify_record(record):
    if file_identity(record['path'])['sha256']!=record['sha256']:
        raise ValueError('Changed compile artifact: '+record['path'])


def register(source,ledger,root,layers):
    if source['source_id']!=digest({k:v for k,v in source.items() if k!='source_id'}):
        raise ValueError('Source identity mismatch')
    operators={op['id']:op for op in source['operators']}
    values={v['id']:v for v in source['values']}
    if len(set(layers))!=len(layers) or any(type(n)!=int or not 0<=n<36 for n in layers):
        raise ValueError('Repeated or invalid source layer')
    implementations=[]
    gates={}
    for physical in ledger['implementations']:
        name=physical['name'];directory=root/name
        manifest_path=directory/'implementation.json'
        if file_identity(manifest_path)['sha256']!=physical['implementation']['sha256']:
            raise ValueError('Resource ledger references a changed implementation')
        manifest=json.loads(manifest_path.read_text())
        for key in ('source','producer','schedule','source_air','compiled_binary'):
            verify_record(manifest[key])
        for item in manifest['physical_evidence']['aie_sources']+manifest['physical_evidence']['placed_sources']:
            verify_record(item)
        gate_path=root/'numeric-gates'/name/'result.json'
        gate=json.loads(gate_path.read_text())
        if (not gate.get('passed') or len(gate['samples'])!=3 or
            not all(row['bitexact'] and row['max_abs_error']==0 for row in gate['samples']) or
            gate['implementation_sha256']!=file_identity(manifest_path)['sha256'] or
            gate['source_shape']!=manifest['shape'] or
            gate['producer_sha256']!=manifest['producer']['sha256']):
            raise ValueError('Missing or mismatched actual hardware numerical gate')
        for row in gate['samples']:
            event=row['event']
            if event['wall_ns']!=sum(event[k] for k in ('device_open_ns','host_prepare_ns','native_inclusive_service_ns','host_assemble_ns')):
                raise ValueError('Hardware gate timing is not conserved')
        gates[name]=dict(result=file_identity(gate_path),capture=file_identity(gate_path.with_name('capture.npz')),
                         input_contract=gate['input_contract'],scope=gate['scope'],calls=3,bitexact=True,
                         prices_admitted=False)
        shape=manifest['shape'];pad=manifest['padded_shape'];resource=manifest['resource']
        # Actual dimensions, never filename prefixes, bind source roles.
        roles=[]
        for role in ('qkv','o','gate_up','down'):
            op=operators.get('prefill.L0.'+role)
            if op and op['kind']=='matmul':
                a,b=[values[v]['shape'] for v in op['inputs']]
                if a==[shape['M'],shape['K']] and b==[shape['K'],shape['N']]:roles.append(role)
        if len(roles)!=1:raise ValueError('Compiled shape does not bind exactly one real source projection')
        role=roles[0];replicas=resource['task_instance_replicas']
        abi=manifest['abi'];bo_total=sum(math.prod(b['shape'])*(2 if b['dtype']=='bf16' else 4) for b in abi)
        prepared_weights=sum(math.prod(b['shape'])*2 for b in abi if b['index'] in range(1,len(abi),2))
        host_arguments=sum(math.prod(b['shape'])*(2 if b['dtype']=='bf16' else 4) for b in abi if b['index']==0 or b['access']=='write')
        host_concat_temporary=shape['M']*shape['N']*4
        sram={}
        for tile,size in physical['compiler_sram_buffer_bytes_by_tile'].items():
            memory='npu.sram.'+tile
            sram[memory]=size+(physical['stack_bytes_per_core'] if tile.startswith('tile_') else 0)
        demands={'npu.device':1,'npu.command':1,'cpu.host':1}
        for x,y in physical['physical_evidence']['actual_core_coordinates']:
            demands[f'npu.core.{x}.{y}']=1
        for layer in layers:
            opid=f'prefill.L{layer}.{role}';op=operators[opid]
            if op['kind']!='matmul' or [values[v]['shape'] for v in op['inputs']]!=[[shape['M'],shape['K']],[shape['K'],shape['N']]]:
                raise ValueError('Source layer differs from compiled GEMM shape')
            runtime=dict(operator=opid,variant=name,directory=str(directory),
                manifest_sha256=file_identity(manifest_path)['sha256'],context_policy='close-after-call',host_pack_cache_policy='clear-after-call',
                weight_identity='stable numpy BF16 view of retained original per-layer torch tensor')
            variant_id='r18-gemm-'+digest(dict(source_id=source['source_id'],runtime=runtime,artifact=manifest['compiled_binary']['sha256']))
            context_id=digest(dict(elf=manifest['compiled_binary']['sha256'],kernel=manifest['kernel_name'],policy='close-after-call'))
            packing=[]
            for index in range(replicas):
                packing.append(dict(logical_value=op['inputs'][1],buffer_index=1+2*index,
                    offsets=[0,index*(shape['N']//replicas)],extents=[shape['K'],shape['N']//replicas],
                    physical_shape=[shape['K'],pad['N']],dtype='bf16',padding='zero',
                    action='source BF16 slice copy into cached immutable host B shard',
                    lifetime='packed host weights cleared after each synchronous source call'))
            implementations.append(dict(id=variant_id,source_id=source['source_id'],operators=[opid],
                input_values=op['inputs'],output_values=op['outputs'],
                artifact_identity=manifest['compiled_binary']['sha256'],entrypoint=manifest['kernel_name'],
                device='npu',resource='npu.device',memory='runtime-abi',launch_count=1,
                wall_ms=None,executable=False,numeric_admitted=True,
                admission_status='standalone-dyadic-validated-pending-complete-qwen-replay-and-hook-price',
                cost_boundary='inclusive-complete-source-GEMM-hook-with-context-open-stage-readback-close',
                transport_ownership='implementation-inclusive',visibility_policy='host-materialized',
                runtime_config=runtime,numerical_gate=name,
                execution_variant=dict(runtime_context_identity=context_id,
                    dataflow=f"BF16-input/FP32-accumulator; L2K{manifest['tiling']['L2_K']}/L1K{manifest['tiling']['L1_K']}; independent-output-N-shards={replicas}",
                    residence_policy='original-source-weight-view-retained; padded-host-weight-packs/device-BOs/context-released-after-call',
                    implementation_parameters={**manifest['tiling'],'column_offset':resource['column_offset']},
                    replicas=replicas,work_distribution='single-invocation' if replicas==1 else 'registered-disjoint-work',
                    resource_demands=demands,resource_shape=[resource['compute_herd_shape'][0]*replicas,resource['compute_herd_shape'][1]],
                    temporary_bytes={'runtime-abi':bo_total+host_arguments+host_concat_temporary+prepared_weights,**sram},
                    resident_allocations={}),
                storage=dict(source_weight_view_aliases_original=True,source_weight_copy_bytes=0,
                    prepared_weight_copies=packing,host_prepared_weights_temporary_bytes_per_call=prepared_weights,host_pack_cache_policy='clear-after-call',
                    device_bo_total_bytes_per_call=bo_total,host_argument_array_bytes_per_call=host_arguments,
                    host_concat_temporary_bytes_per_call=host_concat_temporary,
                    device_weight_lifetime='new BO allocation/upload on every source call; released after synchronous completion',
                    context_open_and_close_in_hook=True,
                    first_call_scratch_zero_upload_bytes=sum(math.prod(b['shape'])*4 for b in abi if b['access']=='write'),
                    BO_upload_bytes_per_call=bo_total,
                    BO_readback_bytes_per_call=sum(math.prod(b['shape'])*4 for b in abi if b['access']=='write'),
                    logical_output_bytes=shape['M']*shape['N']*4,
                    sram_claim='actual compiler buffers plus 2048-byte stack per physical core; temporary for exclusive invocation',
                    capacity_fit_claim=False),
                artifact_binding=dict(manifest=file_identity(manifest_path),backend=file_identity(manifest['producer']['path']),
                    ELF=manifest['compiled_binary'],source_air=manifest['source_air'],
                    actual_core_coordinates=physical['physical_evidence']['actual_core_coordinates'])))
    capacities=json.loads((root/'target-capacities.json').read_text())
    if capacities['target']!='npu2' or capacities['columns']!=8 or capacities['rows']!=6:
        raise ValueError('Mismatched physical target capacity evidence')
    known_memories={m['id']:m['capacity_bytes'] for m in capacities['architecture_fragment']['memories']}
    for implementation in implementations:
        for memory,size in implementation['execution_variant']['temporary_bytes'].items():
            if memory.startswith('npu.sram.') and (memory not in known_memories or size>known_memories[memory]):
                raise ValueError('Compiled SRAM usage exceeds authoritative target capacity')
    architecture=capacities['architecture_fragment']
    architecture['domains']=[dict(id='system',kind='system')]
    architecture['links']=[]
    architecture['resources'] += [dict(id='npu.device',kind='compute',device='npu',capacity=1,domain='system'),dict(id='cpu.host',kind='compute',device='cpu',capacity=1,domain='system')]
    architecture['memories'].append(dict(id='runtime-abi',kind='system-bo',domain='system',capacity_bytes=20*1024**3,capacity_known=True,accessible_resources=['cpu.host','npu.device']))
    return dict(schema='orbit.qwen-r18-pending-runtime-registration.v1',source_id=source['source_id'],
        target_capacities=file_identity(root/'target-capacities.json'),architecture_fragment=architecture,
        runtime_abi_capacity_binding=dict(bytes=20*1024**3,kind='explicit-admission-quota',physical_capacity_claim=False,scope='shared source ABI, prepared host copies and per-call BO/argument scratch; not hardware SRAM or total physical DRAM'),
        producer=file_identity(__file__),resource_ledger=file_identity(root/'resource-ledger.json'),
        runtime_files=[file_identity(Path(__file__).resolve().parents[2]/'scripts'/name) for name in ('qwen-r18-runtime.py','run-qwen-r18-runtime.py')],
        context_policy='close-after-call',numeric_gates=gates,implementations=implementations,
        admission='numeric admitted on recorded dyadic hardware gates only; price null until complete-Qwen source hook training; not a search catalogue')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',default='all')
    a=p.parse_args();layers=list(range(36)) if a.layers=='all' else [int(x) for x in a.layers.split(',')]
    result=register(json.loads(a.source.read_text()),json.loads((a.artifact_root/'resource-ledger.json').read_text()),a.artifact_root.resolve(),layers)
    result['execution_registration_identity']=digest(result)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(registrations=len(result['implementations']),numerical_gates=len(result['numeric_gates']),identity=result['execution_registration_identity'])))


if __name__=='__main__':main()
