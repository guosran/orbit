#!/usr/bin/env python3
"""Lower the actual checked ready ProgramGraph to immutable Qwen r19 replay.

Source indices, normalized event priorities, implementation ABI, physical BO
packing and independent replica partitions are read from the selected graph.
This does not materialize a replacement physical graph or reinterpret a manual
configuration as an ORBIT result. The r18 serial translator remains unchanged.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


def load(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
_verifier_spec=importlib.util.spec_from_file_location('orbit_r19_cache_verification',Path(__file__).with_name('qwen-r19-verification.py'))
verification=importlib.util.module_from_spec(_verifier_spec);_verifier_spec.loader.exec_module(verification)
sha=verification.checksum

def require(condition,message):
    if not condition:raise ValueError(message)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


class Files:
    """Verify actual files once, checking every repeated expected identity."""
    def __init__(self):self.actual={};self.bindings={}
    def check(self,path,expected):
        requested=str(Path(path).absolute());canonical=str(Path(path).resolve(strict=True))
        return self.record(dict(file=canonical,requested_file=requested,sha256=expected))
    def record(self,row):
        bound=verification.binding(row);path=bound['file']
        if path not in self.actual:self.actual[path]=sha(bound['requested_file'])
        require(self.actual[path]==bound['sha256'],'Actual registered file changed: '+path)
        key=(bound['requested_file'],path)
        if key in self.bindings:require(self.bindings[key]==bound,'Conflicting registered alias identity')
        self.bindings[key]=bound
        return path
    def rows(self):return [self.bindings[key] for key in sorted(self.bindings)]


def verify_support(ready,source_id,files):
    record=ready['async_runtime_support_proof'];payload=load(files.record(record))
    require(record['identity']==payload['identity']==digest({k:v for k,v in payload.items() if k!='identity'}),'Async capability proof is not bound to actual payload bytes')
    gates=[]
    def visit(value,trail=()):
        if isinstance(value,dict):
            if ('file' in value or 'path' in value) and 'sha256' in value:
                path=files.record(value)
                if any('gate' in part.lower() or 'audit' in part.lower() for part in trail):
                    if Path(path).suffix=='.json':
                        candidate=load(path)
                        if candidate.get('schema')=='orbit.qwen-r19-independent-ready-audit.v1':
                            require(candidate.get('all_original_gates_pass') and candidate.get('code_files_current_match') and candidate.get('persistent_npu_owner_verified'),'Backend support contains a failed numerical/owner/code audit')
                            gates.append(path)
            for key,item in value.items():visit(item,trail+(str(key),))
        elif isinstance(value,list):
            for item in value:visit(item,trail)
    visit(payload)
    require(gates,'Async capability proof contains no actual complete-Qwen backend audit')
    for row in ready['code_files']:files.record(row)
    files.record(ready['runnerfile'])
    require(ready.get('preparation_protocol')=='owned-prefill-decode-release-readonly-prefault-v2','Final selected replay must use the declared uniform preparation protocol')
    files.record(ready['preparation_entrypoint'])
    for key in ('source_manifest','prefill_source_manifest'):files.record(ready[key])
    require(ready['source_manifest']['source_id']==source_id,'Ready backend proof refers to a different whole source')
    frozen=ready['frozen_r18_runtime'];directory=Path(frozen['directory']).resolve(strict=True)
    require(Path(files.record(frozen['seal']))==directory/'freeze.json','Frozen adapter seal path mismatch')
    for row in load(directory/'freeze.json')['files']:files.record(row)
    return record['identity'],directory,gates


def normalized_schedule(graph,selected,variants):
    regions=graph['execution_regions'];events=graph['ledger']['events'];count=len(selected)
    require(len(regions)==len(variants)==len(events)==count,'Actual invocation/event normalization differs')
    ids=[]
    for index,(row,variant,region) in enumerate(zip(selected,variants,regions)):
        require(row['id']==variant['implementation_id']==region['implementation'],'Selected cover/variant/region index binding changed')
        # Lifecycle activities may be separate, but each actual source cover is
        # represented by one true compute event, never an assignment index.
        require(region.get('source_tasks')==row['source_operators'],'Region source-task coverage changed')
        require(region.get('member_events')==[index],'Region does not bind exactly its actual invocation event')
        ids.append(row['id'])
    priority=graph['external_mapping']['ready_priority']
    require(all(type(i)==int for i in priority) and len(priority)==count and set(priority)==set(range(count)),'ready_priority is not a complete normalized event permutation')
    ranks={ids[index]:rank for rank,index in enumerate(priority)}
    dependencies={}
    for index,region in enumerate(regions):
        predecessors=region.get('predecessor_regions',[])
        require(all(type(i)==int and 0<=i<count and i!=index for i in predecessors),'Invalid physical predecessor region index')
        event_predecessors=events[index].get('depends_on',[])
        require(all(type(i)==int and 0<=i<count for i in event_predecessors),'Invalid invocation predecessor event')
        require(set(event_predecessors)==set(predecessors),'Actual invocation and physical region predecessor edges differ')
        dependencies[ids[index]]=[ids[i] for i in predecessors]
    return ranks,dependencies


def translate(graph,catalogue,prefill,files):
    require(graph.get('schema')=='orbit.program-graph.v1' and graph.get('source_kind')=='source-bound-external-operators','An actual common source-bound ProgramGraph is required')
    source=graph['source'];sid=source['source_id'];require(sid==digest({k:v for k,v in source.items() if k!='source_id'})==catalogue['source_id'],'Whole source canonical identity mismatch')
    require(prefill['source_id']==digest({k:v for k,v in prefill.items() if k!='source_id'}),'Prefill canonical identity mismatch')
    for key in ('operators','values','bindings'):require(source[key][:len(prefill[key])]==prefill[key],'Selected graph changed executable prefill '+key)
    registration=catalogue['runtime_registration'];ready=registration['ready_runtime']
    require(registration['source_id']==sid and ready['prefill_source_manifest']['source_id']==prefill['source_id'],'Runtime source ABI mismatch')
    support,frozen,gates=verify_support(ready,sid,files)
    for row in source['bindings']:files.record(row)
    for row in registration.get('code_files',[]):files.record(row)
    selection=graph['external_implementation_selection'];mapping=graph['external_mapping']
    require(selection['schema']=='orbit.program-operator-selection.v1' and mapping['schema']=='orbit.program-operator-mapping.v1','Missing actual common selection/mapping witness')
    require(mapping['registration_identity']==catalogue['execution_registration_identity'],'Selected graph uses a different implementation catalogue')
    for key,expected in [('execution_policy','source-dependency-ready'),('source_ordering_identity',sid),('async_runtime_proof_identity',support)]:require(selection.get(key)==expected,'Selection ready execution proof changed '+key)
    control=catalogue.get('invocation_control_model')
    if control is not None:
        fields=('schema','identity','source_id','intercept_ms','coefficients','boundary')
        require(all(mapping.get('control_model',{}).get(key)==control.get(key) for key in fields),'Selected callback-controller model differs from actual fitted model')
        require(control['source_id']==sid and control['boundary']=='outside-inclusive-invocation-wall','Controller model source/cost boundary changed')
    selected=selection['regions'];variants=mapping['variants'];ranks,dependencies=normalized_schedule(graph,selected,variants)
    implementations={row['id']:row for row in catalogue['implementations']};chosen={row['id'] for row in selected}
    require(len(implementations)==len(catalogue['implementations']) and len(chosen)==len(selected),'Repeated implementation identity')
    grouped=set()
    for group in catalogue['choice_groups']:
        matches=[choice for choice in group['choices'] if set(choice['implementations'])<=chosen]
        require(len(matches)==1,'Selected plan violates actual cover coupling: '+group['id']);grouped.update(matches[0]['implementations'])
    require(grouped==chosen,'Selected implementation is not a registered choice')
    old=module(Path(__file__).with_name('replay-r18-selection.py'),'r19_physical_abi_checks')
    files.check(old.__file__,sha(old.__file__))
    source_ops=source['operators'];byop={v['id']:v for v in source_ops};byvalue={v['id']:v for v in source['values']};valueids={v['id']:i for i,v in enumerate(source['values'])}
    prefix={v['id'] for v in prefill['operators']};seen=set();placements={};gemms={};routes={};halves={};cpu_regions=[];implementation_ids={};prefix_implementations=set();contexts=set();host_contexts=set();steps=set()
    for row,variant in zip(selected,variants):
        identity=row['id'];require(identity in implementations and identity in registration['implementations'],'Selected invocation has no actual runtime registration')
        impl=implementations[identity];reg=registration['implementations'][identity]
        if control is not None:
            features=row.get('controller_features')
            require(features==impl.get('controller_features') and set(features)=={'cpu_callbacks','npu_callbacks'} and all(type(v)==int and v>=0 for v in features.values()),'Controller features differ from actual implementation callbacks')
        fields=('id','source_id','artifact_identity','entrypoint','device','resource','memory','source_operators','inputs','outputs','wall_ms','executable','launch_count','cost_boundary','transport_ownership','visibility_policy')
        require(all(row.get(k)==impl.get(k) for k in fields),'Selected implementation differs from admitted catalogue: '+identity)
        require(row['executable'] and row['wall_ms'] is not None,'Unvalidated/unpriced implementation cannot be replayed')
        require(not variant.get('unproved_constraints'),'Selected physical variant has unknown feasibility')
        require(variant['artifact_identity']==impl['artifact_identity'] and variant['entrypoint']==impl['entrypoint'],'Physical artifact entrypoint changed')
        for key in ('runtime_context_identity','dataflow','residence_policy','implementation_parameters','resource_demands','resource_shape','replicas','work_distribution','temporary_bytes','resident_allocations','execution_policy','source_ordering_identity','async_runtime_proof_identity'):
            default=[] if key=='resource_shape' else {} if key in ('implementation_parameters','temporary_bytes','resident_allocations') else None
            require(variant.get(key)==impl['execution_variant'].get(key,default),'Physical execution variant differs: '+key)
        for key,expected in [('execution_policy','source-dependency-ready'),('source_ordering_identity',sid),('async_runtime_proof_identity',support)]:require(variant.get(key)==expected,'Variant has no matching actual asynchronous capability proof')
        old.verify_packing(source,impl,variant);old.verify_work_partitions(source,impl,variant,valueids)
        indices=row['source_operators'];require(indices and all(type(i)==int and 0<=i<len(source_ops) for i in indices),'Invalid original source index')
        require(not seen.intersection(indices),'Overlapping selected source covers');seen.update(indices);names=[source_ops[i]['id'] for i in indices]
        require(reg.get('operators',names)==names,'Registered runtime body covers different source operators')
        require(reg.get('source_id',sid)==sid and reg.get('artifact_identity',row['artifact_identity'])==row['artifact_identity'],'Registered runtime source/artifact identity changed')
        for proof in reg.get('code_files',[])+reg.get('artifact_files',[]):files.record(proof)
        kind=reg['kind']
        if set(names)<=prefix:
            prefix_implementations.add(identity)
            if kind in ('prefill-original','prefill-node','prefill-gemm','heterogeneous-gate-up'):
                operator=reg['operator'];require(names==[operator],'Per-node invocation covers different source arithmetic')
                placements[operator]='cpu' if kind=='heterogeneous-gate-up' else row['device'];implementation_ids[operator]=identity
                if kind=='prefill-gemm':require(reg.get('context_policy')=='close-after-call' and reg.get('host_pack_cache_policy')=='clear-after-call','GEMM context/cache policy differs from frozen runtime')
                if kind in ('prefill-gemm','heterogeneous-gate-up'):
                    directory=Path(reg['directory']).resolve(strict=True);files.check(directory/'implementation.json',reg['manifest_sha256']);artifact=load(directory/'implementation.json')
                    for key in ('source','source_air','compiled_binary'):files.record(artifact[key])
                    shape=artifact['shape'];inputs=byop[operator]['inputs'];weight=byvalue[inputs[1]]['shape'];activation=byvalue[inputs[0]]['shape']
                    if kind=='heterogeneous-gate-up':
                        require(operator.endswith('.gate_up') and reg['cpu_half'] in ('gate','up'),'Mixed half bound to unsupported source arithmetic')
                        require([shape['M'],shape['K']]==activation and [shape['K'],2*shape['N']]==weight,'Mixed half artifact does not implement source output-column fission')
                        halves[operator]={k:reg[k] for k in ('cpu_half','directory','manifest_sha256')}
                    else:
                        require([shape['M'],shape['K']]==activation and [shape['K'],shape['N']]==weight,'Compiled GEMM shape differs from original source ABI')
                        gemms[operator]={k:reg[k] for k in ('directory','manifest_sha256')}
                    replicas=artifact['resource']['task_instance_replicas']
                    require(variant['replicas']==replicas+(1 if kind=='heterogeneous-gate-up' else 0),'Selected physical replicas differ from actual CPU/NPU invocations')
                    if kind=='heterogeneous-gate-up':
                        cpu_offset=0 if reg['cpu_half']=='gate' else 11008;npu_offset=11008-cpu_offset
                        wanted=[dict(output_value=valueids[out],replica=0,offsets=[0,cpu_offset],extents=[5,11008]) for out in byop[operator]['outputs']]
                        wanted.extend(dict(output_value=valueids[out],replica=r+1,offsets=[0,npu_offset+r*(11008//replicas)],extents=[5,11008//replicas]) for out in byop[operator]['outputs'] for r in range(replicas))
                        require(variant.get('work_partitions')==wanted,'Actual CPU half/NPU output concatenation differs from checked replica partitions')
            elif kind=='prefill-fused-ffn':
                layer=reg['layer'];require(names==[f'prefill.L{layer}.{part}' for part in ('gate_up','gate_split','swiglu','down')],'FFN executable does not cover original four-node source subgraph')
                for operator in names:placements[operator]='cpu' if operator.endswith('.gate_split') else 'npu'
                implementation_ids[names[0]]=identity;routes[str(layer)]={k:copy.deepcopy(reg[k]) for k in ('retention','split_order','ffn_artifact')};files.check(reg['ffn_artifact']['manifest'],reg['ffn_artifact']['sha256'])
            elif kind=='prefill-cpu-region':
                require(row['device']=='cpu','CPU native region mapped to non-CPU device')
                cpu=module(Path(ready['runnerfile']['file']).parent/'qwen-r19-regions.py','r19_selected_cpu_regions')
                proof=cpu.region(prefill,names);require(proof['region_id']==reg['region_id'],'CPU cover source/alias proof changed')
                instance=cpu.ExecutableRegion(prefill,names,reg['backend'],reg['input_policy'],reg['output_policy']);require(instance.template_id==reg['template_id'],'CPU implementation template differs')
                if reg['backend']=='inductor-cpu':
                    compiled=load(files.record(reg['compile_proof']));require(compiled['region']==proof and compiled['template_id']==reg['template_id'],'CPU actual-input compilation does not implement selected source cover')
                    for record in compiled['generated_code']:files.record(record)
                    for record in reg.get('actual_input_specializations',[]):files.record(record)
                else:require(reg.get('compile_proof') is None,'Eager host grouping must not claim a compiled kernel')
                cpu_regions.append(dict(id=identity,operators=names,**{k:reg[k] for k in ('backend','input_policy','output_policy')},expected_compile_proof=reg.get('compile_proof'),expected_input_specializations=reg.get('actual_input_specializations',[])))
                for operator in names:placements[operator]='cpu'
            elif kind=='host':
                require(len(names)==1 and row['device']=='cpu','A multi-node host cover needs a real CPU region implementation')
                placements[names[0]]='cpu';implementation_ids[names[0]]=identity
            else:raise ValueError('Unknown prefill runtime implementation kind: '+kind)
        else:
            require(not set(names).intersection(prefix),'Implementation crosses unsupported prefill/decode runtime boundary')
            context=(str(Path(reg['decode_variant_dir']).resolve(strict=True)),bool(reg['decode_rope_prefetch'])) if 'decode_variant_dir' in reg else None
            if kind=='decode-context':
                contexts.add(context);require(context is not None,'Decode invocation lacks actual context binding')
                template=load(files.record(reg['template_identity']))
                require(template['variant_id']==reg['variant_id'] and Path(reg['template_identity'].get('file',reg['template_identity'].get('path'))).resolve()==(Path(context[0])/'variant.json').resolve(),'Actual decode template identity differs from selected artifact')
                covered=set()
                for name in names:
                    match=re.match(r'^decode\.T([0-9]+)\.',name);require(match is not None,'Decode kernel body has no token identity');covered.add(int(match.group(1)))
                require(covered=={reg['step']},'Kernel implementation crosses autoregressive token boundary');steps.update(covered)
            elif kind=='host':
                require(row['device']=='cpu','Decode host callback mapped to non-CPU resource')
                if context is not None:host_contexts.add(context)
            else:raise ValueError('Unknown decode runtime implementation kind: '+kind)
    require(seen==set(range(len(source_ops))) and set(placements)==prefix,'Actual selected graph omits source operations')
    require(len(contexts)==1 and steps==set(range(127)),'One actual decode context must cover every full36-layer generation step')
    require(not host_contexts or host_contexts==contexts,'Decode host callback registration differs from selected device context')
    context_directory,prefetch=next(iter(contexts))
    prefix_dependencies={name:[parent for parent in dependencies[name]] for name in prefix_implementations}
    require(all(parent in prefix_implementations for values in prefix_dependencies.values() for parent in values),'Prefill physical dependency requires an unavailable decode event')
    adapter=module(frozen/'scripts/qwen-r18-runtime.py','r19_selected_storage_adapter');plan=adapter.node.make_plan(prefill,placements)
    default=routes[sorted(routes,key=int)[0]] if routes else dict(retention='retain',split_order='gate-up',ffn_artifact={})
    backend=registration['gemm_backend'] if gemms else {}
    if backend:files.record(backend)
    plan['r18']=dict(fused_layers=sorted(map(int,routes)),rows=128,block_m=128,block_n=256,**copy.deepcopy(default),gemm_variants=gemms,gemm_backend=backend,gemm_context_policy='close-after-call')
    plan['r18_layer_ffn']=routes;plan['plan_id']=digest({k:v for k,v in plan.items() if k!='plan_id'})
    routing=module(Path(ready['runnerfile']['file']).parent/'routed-r18-runtime.py','r19_selected_storage_routing');routing.adapt(adapter).validate_plan(prefill,plan)
    context=module(frozen/'scripts/qwen-r18-decode.py','r19_selected_context').validate(context_directory)
    files.record(context['template_identity'])
    for build_record in context['variant']['builds']:
        build_path=Path(files.record(build_record));build=load(build_path)
        for key in ('request','source','source_air','binary','instructions'):files.record(build[key])
        request=load(build['request']['path'])
        for key in ('source','owned_source','producer'):files.record(request[key])
        for name,expected in build['device_files'].items():files.check(build_path.parent/'air_project'/name,expected)
    spec=dict(schema='orbit.qwen-r19-ready-spec.v1',source_id=prefill['source_id'],whole_source=ready['source_manifest'],source_ready=True,decode_rope_prefetch=prefetch,
        operators=halves,cpu_regions=cpu_regions,implementation_ids=implementation_ids,ready_priority={name:ranks[name] for name in prefix_implementations},prepare_tokens=ready['prepare_tokens'])
    return dict(schema='orbit.qwen-r19-selected-replay.v1',mode='actual-orbit-selected',program_graph_id=graph['id'],source_id=sid,plan=plan,spec=spec,decode_context=context,
        prefix_implementation_dependencies=prefix_dependencies,full_invocation_dependencies=dependencies,full_ready_priority=ranks,actual_support_identity=support,source_files=files.rows(),ready_runtime=ready,
        validation_gates=gates,invocation_control_model=control,execution_policy='source-dependency-ready',outputs=128)


def verify_manual_assignment(assignment,result,graph,catalogue):
    """Check an explicit baseline result without calling it an optimization."""
    require(assignment.get('schema')=='orbit.program-manual-choices.v1' and assignment.get('purpose')=='manual-baseline','Invalid prescribed baseline assignment')
    require(result.get('mode')=='manual-materialization' and result.get('search_performed') is False,'Baseline graph must come from no-search common materialization')
    require(result.get('manual_assignment')==assignment,'Common result does not bind the exact baseline assignment')
    require(assignment['source_id']==catalogue['source_id'] and assignment['execution_registration_identity']==catalogue['execution_registration_identity'],'Baseline assignment source/catalogue changed')
    require(assignment.get('ready_priority_policy')=='source-order','Manual baseline must use source-order ready priority')
    groups={group['id']:group for group in catalogue['choice_groups']};choices=assignment['choices']
    require(len(choices)==len(groups) and {row['group_id'] for row in choices}==set(groups),'Baseline assignment must specify every group exactly once')
    expected=set()
    for row in choices:
        matches=[choice for choice in groups[row['group_id']]['choices'] if choice['id']==row['choice_id']]
        require(len(matches)==1,'Unknown or ambiguous prescribed baseline choice')
        expected.update(matches[0]['implementations'])
    actual={row['id'] for row in graph['external_implementation_selection']['regions']}
    require(actual==expected,'Materialized baseline changed prescribed implementation choices')
    require(graph['external_mapping']['ready_priority']==list(range(len(actual))),'Materialized baseline does not retain canonical source-order priority')


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('graph','catalogue','prefill-source','search-result','output','aie-root'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--manual-assignment',type=Path,help='Explicit no-search baseline assignment materialized and checked by common CLI')
    a=parser.parse_args();graph=load(a.graph);search=load(a.search_result);catalogue=load(a.catalogue)
    require(search.get('selected_graph_independently_checked') is True and search.get('selected_program_graph')==graph,'Input is not the actual independently checked selected graph from this search result')
    files=Files()
    for record in verification.files():files.record(record)
    if a.manual_assignment:
        assignment=load(a.manual_assignment);verify_manual_assignment(assignment,search,graph,catalogue)
        for record in assignment.get('provenance',{}).values():
            if isinstance(record,dict) and 'file' in record and 'sha256' in record:files.record(record)
    else:require(search.get('mode')!='manual-materialization','A manual baseline cannot be labelled an ORBIT-selected result')
    replay=translate(graph,catalogue,load(a.prefill_source),files)
    if a.manual_assignment:
        replay['mode']='manual-baseline';replay['manual_assignment']=dict(file=str(a.manual_assignment.resolve()),sha256=sha(a.manual_assignment),assignment=assignment)
    a.output.mkdir(parents=True,exist_ok=False);replay['spec']['cpu_prepare_directory']=str(a.output/'cpu-preparation')
    replay['verification_cache']=verification.receipt()
    replay['replay_id']=digest(replay)
    for name,value in [('plan.json',replay['plan']),('spec.json',replay['spec']),('replay.json',replay)]:
        (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    shim=Path(__file__).with_name('run-r19-selected.py').resolve();ready=replay['ready_runtime']
    command=[sys.executable,str(shim),'--replay',str(a.output/'replay.json'),'--runtime-runner',ready['runnerfile']['file'],'--memory-preparation-shim',ready['preparation_entrypoint']['file'],'--frozen-runtime',ready['frozen_r18_runtime']['directory'],
        '--ready-spec',str(a.output/'spec.json'),'--aie-root',str(a.aie_root),'--source',str(a.prefill_source),'--plan',str(a.output/'plan.json'),'--output',str(a.output/'execution'),
        '--decode-variant-dir',replay['decode_context']['directory'],'--outputs','128','--warmups','1','--repeats','3']
    receipt=dict(schema='orbit.qwen-r19-command.v1',mode=replay['mode'],replay_id=replay['replay_id'],command=command,program_graph_id=graph['id'],files=[verification.file_record(path) for path in (a.graph,a.catalogue,a.prefill_source,a.search_result,Path(__file__),shim,a.output/'plan.json',a.output/'spec.json',a.output/'replay.json')],registered_files=replay['source_files'],hardware_executed=False)
    receipt['verification_cache']=verification.receipt()
    if a.manual_assignment:receipt['files'].append(verification.file_record(a.manual_assignment))
    (a.output/'command-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
