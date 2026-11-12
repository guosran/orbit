#!/usr/bin/env python3
"""Translate checked common ProgramGraph implementations into actual r18 calls.

This never synthesizes a physical graph. Manual baselines are explicitly
labelled plans; selected plans consume original source indices and registered
implementation/packing identities from the common graph.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import sys


def load(path):return json.loads(Path(path).read_text())
def canonical(row):return json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(row):return hashlib.sha256(canonical(row)).hexdigest()
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def verify_packing(source,impl,mapping):
    expected=impl.get('storage',{}).get('prepared_weight_copies',[])
    if not expected:
        require(not mapping.get('packed_inputs'),'unexpected physical source packing')
        return
    keys=('allocation_id','memory','dtype','bytes','source_offsets','source_extents','physical_shape','physical_offsets','strides')
    want={row['allocation_id']:(row['source_value'],{key:row[key] for key in keys}) for row in expected}
    actual={}
    for packed in mapping.get('packed_inputs',[]):
        index=packed['source_value'];require(type(index)==int and 0<=index<len(source['values']),'invalid packing source value index')
        value=source['values'][index]['id']
        for fragment in packed['fragments']:
            key=fragment['allocation_id'];require(key not in actual,'duplicate packed physical allocation')
            actual[key]=(value,{field:fragment[field] for field in keys})
    require(actual==want,'selected physical packing differs from actual source-bit preparation')


def verify_work_partitions(source,impl,mapping,valueids=None):
    if valueids is None:valueids={value['id']:i for i,value in enumerate(source['values'])}
    expected=[]
    for row in impl['execution_variant'].get('work_partitions',[]):
        require(row['output_value'] in valueids,'registered replica output is not a source value')
        expected.append(dict(row,output_value=valueids[row['output_value']]))
    actual=mapping.get('work_partitions',[])
    require(all(type(row.get('output_value'))==int and type(row.get('replica'))==int for row in actual),'invalid typed replica work binding')
    key=lambda row:(row['output_value'],row['replica'])
    require(sorted(actual,key=key)==sorted(expected,key=key),'selected independent replica work differs from actual source output shards')


def translate(graph,catalogue,prefill,frozen):
    require(graph.get('schema')=='orbit.program-graph.v1' and graph.get('source_kind')=='source-bound-external-operators','requires actual common source-bound ProgramGraph')
    source=graph['source'];sid=source['source_id']
    require(sid==digest({k:v for k,v in source.items() if k!='source_id'})==catalogue['source_id'],'source identity mismatch')
    require(prefill['source_id']==digest({k:v for k,v in prefill.items() if k!='source_id'}),'prefill identity mismatch')
    for key in ('operators','values','bindings'):
        require(source[key][:len(prefill[key])]==prefill[key],'whole source changed original prefill '+key)
    registration=catalogue['runtime_registration']
    require(registration['source_id']==sid,'runtime registration source mismatch')
    if 'prefill_source_id' in registration:require(registration['prefill_source_id']==prefill['source_id'],'runtime prefill binding changed')
    selected=graph['external_implementation_selection']
    require(selected['schema']=='orbit.program-operator-selection.v1','invalid common implementation selection')
    mapping=graph['external_mapping'];require(mapping['schema']=='orbit.program-operator-mapping.v1','missing checked mapping witness')
    require(mapping['registration_identity']==catalogue['execution_registration_identity'],'mapping registration identity mismatch')
    variants={row['implementation_id']:row for row in mapping['variants']}
    original={row['id']:row for row in catalogue['implementations']}
    require(len(original)==len(catalogue['implementations']),'duplicate catalogue implementation')
    chosen_ids={row['id'] for row in selected['regions']}
    require(len(chosen_ids)==len(selected['regions']) and chosen_ids==set(variants),'selected mapping/cover mismatch')
    grouped=set()
    for group in catalogue['choice_groups']:
        matches=[row for row in group['choices'] if set(row['implementations'])<=chosen_ids]
        require(len(matches)==1,'violates registered local cover coupling: '+group['id'])
        grouped.update(matches[0]['implementations'])
    require(grouped==chosen_ids,'unregistered selected implementation')
    proofs={}
    def check_file(path,expected):
        path=str(Path(path).resolve(strict=True))
        require(path not in proofs or proofs[path]==expected,'conflicting source/artifact proof')
        if path not in proofs:require(sha(path)==expected,'actual source/artifact changed: '+path)
        proofs[path]=expected
    for row in registration.get('code_files',[]):check_file(row['file'],row['sha256'])
    for row in source['bindings']:check_file(row['file'],row['sha256'])
    placements={};gemms={};routes={};decode_dirs=set();host_contexts=set();decode_steps=set();seen=set();prefill_ids={op['id'] for op in prefill['operators']}
    source_ops=source['operators'];byid={op['id']:op for op in source_ops};source_values={v['id']:v for v in source['values']};valueids={v['id']:i for i,v in enumerate(source['values'])}
    fields=('id','source_id','artifact_identity','entrypoint','device','resource','memory','source_operators','inputs','outputs','wall_ms','executable','launch_count','cost_boundary','transport_ownership','visibility_policy')
    for row in selected['regions']:
        name=row['id'];require(name in original and name in registration['implementations'],'missing actual runtime registration')
        impl=original[name];reg=registration['implementations'][name];variant=variants[name]
        require(all(row.get(key)==impl.get(key) for key in fields),'selected implementation differs from catalogue: '+name)
        require(row['executable'] and row['wall_ms'] is not None,'cannot replay unvalidated/unpriced selected implementation')
        require(variant['artifact_identity']==row['artifact_identity'] and variant['entrypoint']==row['entrypoint'],'execution artifact binding mismatch')
        require(not variant.get('unproved_constraints'),'selected implementation has unproved physical constraints')
        for key in ('runtime_context_identity','dataflow','residence_policy','implementation_parameters','resource_demands','resource_shape','replicas','work_distribution','temporary_bytes','resident_allocations'):
            require(variant.get(key)==impl['execution_variant'].get(key,[] if key=='resource_shape' else {} if key in ('implementation_parameters','temporary_bytes','resident_allocations') else None),'mapping variant changed '+key)
        verify_packing(source,impl,variant)
        verify_work_partitions(source,impl,variant,valueids)
        indices=row['source_operators'];require(indices and all(type(i)==int and 0<=i<len(source_ops) for i in indices),'invalid original source indices')
        require(not seen.intersection(indices),'overlapping source cover');seen.update(indices)
        names=[source_ops[i]['id'] for i in indices]
        if 'operators' in reg:require(names==reg['operators'],'runtime cover differs from actual source indices')
        if 'artifact_identity' in reg:require(reg['artifact_identity']==row['artifact_identity'],'runtime artifact differs')
        for proof in reg.get('code_files',[])+reg.get('artifact_files',[]):
            check_file(proof['file'],proof['sha256'])
        kind=reg['kind']
        if kind in ('prefill-original','prefill-node','prefill-gemm'):
            operator=reg['operator'];require(names==[operator] and operator in prefill_ids,'invalid source operator runtime binding')
            require(operator not in placements,'repeated prefill operator');placements[operator]=row['device']
            if kind=='prefill-gemm':
                require(row['device']=='npu' and byid[operator]['kind']=='matmul','GEMM artifact bound to wrong source operation')
                directory=Path(reg['directory']).resolve(strict=True);manifest=directory/'implementation.json'
                check_file(manifest,reg['manifest_sha256']);record=load(manifest)
                for field in ('source','source_air','compiled_binary'):
                    proof=record[field];check_file(proof['path'],proof['sha256'])
                shape=record['shape'];inputs=byid[operator]['inputs']
                require([shape['M'],shape['K']]==source_values[inputs[0]]['shape'] and [shape['K'],shape['N']]==source_values[inputs[1]]['shape'],'compiled GEMM actual source ABI mismatch')
                replicas=record['resource']['task_instance_replicas']
                require(variant['replicas']==replicas,'selected replica count differs from executable GEMM')
                if replicas>1:
                    actual_parts=variant['work_partitions']
                    wanted=[dict(output_value=valueids[out],replica=r,offsets=[0,r*(shape['N']//replicas)],extents=[shape['M'],shape['N']//replicas]) for out in byid[operator]['outputs'] for r in range(replicas)]
                    require(actual_parts==wanted,'runtime concatenation does not implement selected output-N partitions')
                gemms[operator]=dict(directory=str(directory),manifest_sha256=reg['manifest_sha256'])
        elif kind=='prefill-fused-ffn':
            layer=reg['layer'];expected=[f'prefill.L{layer}.'+role for role in ('gate_up','gate_split','swiglu','down')]
            require(names==expected and row['device']=='npu' and str(layer) not in routes,'invalid actual FFN source cover')
            for operator in names:
                require(operator not in placements,'FFN overlaps a selected per-node implementation')
                placements[operator]='cpu' if operator.endswith('.gate_split') else 'npu'
            routes[str(layer)]={key:copy.deepcopy(reg[key]) for key in ('retention','split_order','ffn_artifact')}
            check_file(reg['ffn_artifact']['manifest'],reg['ffn_artifact']['sha256'])
        elif kind=='decode-context':
            require(all(name.startswith('decode.') for name in names) and row['device']=='npu','decode context covers wrong source')
            decode_dirs.add(str(Path(reg['decode_variant_dir']).resolve(strict=True)))
            for operator in names:
                matched=re.match(r'^decode\.T([0-9]+)\.',operator)
                require(matched is not None,'decode kernel cover has no actual generation-step identity')
                decode_steps.add(int(matched.group(1)))
        elif kind=='host':
            require(row['device']=='cpu','host implementation must execute CPU')
            if 'decode_variant_dir' in reg:host_contexts.add(str(Path(reg['decode_variant_dir']).resolve(strict=True)))
            for operator in names:
                if operator in prefill_ids:
                    require(operator not in placements and byid[operator]['attributes'].get('supported_devices',['cpu'])==['cpu'],'host cover cannot silently replace a selectable arithmetic node')
                    placements[operator]='cpu'
        else:raise ValueError('unknown registered runtime implementation kind '+kind)
    require(seen==set(range(len(source_ops))),'selected plan omits source operators')
    require(set(placements)==prefill_ids,'selected prefill source cover is incomplete')
    require(len(decode_dirs)==1,'one source-bound decode context must serve all127steps')
    require(not host_contexts or host_contexts==decode_dirs,'decode host-phase prices differ from actual selected context')
    require(decode_steps==set(range(127)),'selected context must cover every one of127 decode steps')
    adapter=module(frozen/'scripts/qwen-r18-runtime.py','orbit_r18_translator_adapter')
    plan=adapter.node.make_plan(prefill,placements)
    first=routes[sorted(routes,key=int)[0]] if routes else dict(retention='retain',split_order='gate-up',ffn_artifact={})
    backend=registration['gemm_backend'] if gemms else {}
    if backend:check_file(backend['file'],backend['sha256'])
    plan['r18']=dict(fused_layers=sorted(map(int,routes)),rows=128,block_m=128,block_n=256,**copy.deepcopy(first),
        gemm_variants=gemms,gemm_backend=backend,gemm_context_policy='close-after-call')
    plan['r18_layer_ffn']=routes
    plan['launch_policy']='actual common-selected per-source GEMM/FFN implementation and per-layer residence/order; original synchronous source order elsewhere'
    plan['plan_id']=digest({k:v for k,v in plan.items() if k!='plan_id'})
    routing=module(Path(__file__).with_name('routed-r18-runtime.py'),'orbit_r18_translator_routing')
    routing.adapt(adapter).validate_plan(prefill,plan)
    context=module(frozen/'scripts/qwen-r18-decode.py','orbit_r18_translator_context').validate(next(iter(decode_dirs)))
    replay=dict(schema='orbit.qwen-r18-selected-replay.v1',mode='actual-orbit-selected',source_id=sid,
        program_graph_id=graph['id'],prefill_source_id=prefill['source_id'],plan=plan,decode_context=context,
        source_files=[dict(file=p,sha256=h) for p,h in sorted(proofs.items())],outputs=128,
        execution_policy='registered FFN kernel order; original synchronous source traversal elsewhere; no invented CPU/NPU asynchronous overlap')
    replay['replay_id']=digest(replay)
    return replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path);p.add_argument('--catalogue',type=Path)
    p.add_argument('--baseline-plan',type=Path);p.add_argument('--baseline-label')
    p.add_argument('--decode-variant-dir',type=Path)
    p.add_argument('--prefill-source',type=Path,required=True)
    p.add_argument('--frozen-runtime',type=Path,required=True)
    p.add_argument('--aie-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(bool(a.graph)!=bool(a.baseline_plan),'choose actual graph or explicit manual baseline')
    prefill=load(a.prefill_source)
    if a.graph:
        require(a.catalogue is not None,'selected graph requires registered catalogue')
        replay=translate(load(a.graph),load(a.catalogue),prefill,a.frozen_runtime)
    else:
        require(a.baseline_label and a.decode_variant_dir,'manual baseline requires label and actual decode context')
        plan=load(a.baseline_plan);adapter=module(a.frozen_runtime/'scripts/qwen-r18-runtime.py','orbit_r18_manual_adapter')
        routing=module(Path(__file__).with_name('routed-r18-runtime.py'),'orbit_r18_manual_routing')
        routing.adapt(adapter).validate_plan(prefill,plan)
        context=module(a.frozen_runtime/'scripts/qwen-r18-decode.py','orbit_r18_manual_context').validate(a.decode_variant_dir)
        replay=dict(schema='orbit.qwen-r18-selected-replay.v1',mode='manual-baseline',baseline_label=a.baseline_label,
            program_graph_id=None,graph_materialization_claim=False,prefill_source_id=prefill['source_id'],plan=plan,
            decode_context=context,outputs=128)
        replay['replay_id']=digest(replay)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'replay.json').write_text(json.dumps(replay,indent=2)+'\n')
    (a.output/'plan.json').write_text(json.dumps(replay['plan'],indent=2)+'\n')
    shim=Path(__file__).with_name('routed-r18-runtime.py').resolve()
    command=[sys.executable,str(shim),'--frozen-runtime',str(a.frozen_runtime),'--aie-root',str(a.aie_root),
        '--source',str(a.prefill_source),'--plan',str(a.output/'plan.json'),'--output',str(a.output/'execution'),
        '--decode-variant-dir',replay['decode_context']['directory'],'--outputs','128','--warmups','1','--repeats','3']
    receipt=dict(schema='orbit.qwen-r18-command.v1',replay_id=replay['replay_id'],command=command,
        graph_file=str(a.graph.resolve()) if a.graph else None,graph_sha256=sha(a.graph) if a.graph else None,
        catalogue_file=str(a.catalogue.resolve()) if a.catalogue else None,catalogue_sha256=sha(a.catalogue) if a.catalogue else None,
        replay_sha256=sha(a.output/'replay.json'),plan_sha256=sha(a.output/'plan.json'),
        translator_sha256=sha(__file__),routing_sha256=sha(shim),frozen_seal_sha256=sha(a.frozen_runtime/'freeze.json'),
        baseline_label=a.baseline_label if not a.graph else None,frozen_runtime=str(a.frozen_runtime),
        prefill_source=str(a.prefill_source),prefill_source_sha256=sha(a.prefill_source),executed=False)
    (a.output/'command-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
