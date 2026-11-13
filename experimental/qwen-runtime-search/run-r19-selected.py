#!/usr/bin/env python3
"""Apply a checked selected physical dependency plan to frozen r19 execution.

The frozen executor/math stays unchanged. This owned extension only adds the
selected graph's conservative physical predecessor edges and verifies that
actual CPU compiler specializations match admitted implementations. Its exact
bytes and the selected graph are sealed by the prepared command receipt.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
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


def bind_prepared(ready,dependencies):
    original=ready.PreparedProgram
    class SelectedProgram(original):
        def __init__(self,operations,input_names,outputs):
            operations=list(operations);byid={op.id:op for op in operations}
            endpoint={};entries={}
            for identity in dependencies:
                if identity in byid:endpoint[identity]=identity;entries[identity]=[identity]
                elif all(identity+suffix in byid for suffix in ('.gate','.up','.assemble')):
                    endpoint[identity]=identity+'.assemble';entries[identity]=[identity+'.gate',identity+'.up']
            if endpoint or 'prefill.logits' in outputs:
                require(set(endpoint)==set(dependencies),'Selected graph implementation ABI differs from actual prepared prefill nodes')
                for identity,parents in dependencies.items():
                    for entry in entries[identity]:byid[entry].dependencies.update(endpoint[parent] for parent in parents)
            # Decoder graphs are separate complete synchronous dispatches. Their
            # recurrent source dependency order remains in the original loop.
            super().__init__(operations,input_names,outputs)
    ready.PreparedProgram=SelectedProgram


def bind_cpu_specializations(cpu,config):
    expected={}
    for row in config.get('cpu_regions',[]):
        if row['backend']!='inductor-cpu':continue
        records=row.get('expected_input_specializations',[]) or [row['expected_compile_proof']]
        proofs=[]
        for record in records:
            require(record and sha(record['file'])==record['sha256'],'Admitted CPU specialization proof changed')
            proofs.append(load(record['file']))
        key=(tuple(row['operators']),row['backend'],row['input_policy'],row['output_policy']);expected[key]=proofs
    original=cpu.ExecutableRegion
    class SelectedCPURegion(original):
        def prepare(self,inputs,output_directory):
            result,proof=super().prepare(inputs,output_directory)
            key=(tuple(self.proof['operators']),self.backend,self.input_policy,self.output_policy)
            require(key in expected,'Selected native CPU cover has no admitted compilation proof')
            actual={row['sha256'] for row in proof['generated_code']}
            matches=[row for row in expected[key] if row['template_id']==proof['template_id'] and row['specialization_id']==proof['specialization_id'] and row['region']==proof['region'] and {v['sha256'] for v in row['generated_code']}==actual]
            require(matches,'Actual selected CPU input layout/code differs from admitted compiled specialization')
            return result,proof
    cpu.ExecutableRegion=SelectedCPURegion


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False,allow_abbrev=False)
    parser.add_argument('--replay',type=Path,required=True);parser.add_argument('--runtime-runner',type=Path,required=True)
    parser.add_argument('--memory-preparation-shim',type=Path,required=True)
    args,rest=parser.parse_known_args();replay=load(args.replay)
    require(replay['replay_id']==digest({k:v for k,v in replay.items() if k!='replay_id'}),'Selected replay identity changed')
    for row in replay['source_files']:verification.verify(row)
    option=argparse.ArgumentParser(add_help=False,allow_abbrev=False);option.add_argument('--output',type=Path,required=True);option.add_argument('--ready-spec',type=Path,required=True)
    request,_=option.parse_known_args(rest);config=load(request.ready_spec)
    require(config==replay['spec'],'Selected runtime spec changed')
    require(str(args.memory_preparation_shim.resolve())==str(Path(replay['ready_runtime']['preparation_entrypoint']['file']).resolve()) and sha(args.memory_preparation_shim)==replay['ready_runtime']['preparation_entrypoint']['sha256'],'Selected preparation protocol changed')
    preparation=module(args.memory_preparation_shim,'orbit_r19_selected_memory_preparation')
    runner=module(args.runtime_runner,'orbit_r19_selected_frozen_runner');loader=runner.module
    def selected_loader(path,name):
        result=loader(path,name)
        if Path(path).name=='run-qwen-r18-runtime.py':return preparation.instrument(result)
        if Path(path).name=='qwen-r19-ready.py':bind_prepared(result,replay['prefix_implementation_dependencies'])
        elif Path(path).name=='qwen-r19-regions.py':bind_cpu_specializations(result,config)
        return result
    runner.module=selected_loader;sys.argv=[str(args.runtime_runner),*rest]
    # This sidecar sits next to the fresh execution directory, so recording
    # preflight advice does not violate the frozen runner's mkdir contract.
    verification_path=Path(str(request.output)+'.verification-cache.json')
    require(not verification_path.exists(),'Preserve previous verifier evidence')
    try:runner.main()
    finally:verification_path.write_text(json.dumps(verification.receipt(),indent=2)+'\n')
    checked=[]
    for line in (request.output/'measurements.jsonl').read_text().splitlines():
        row=json.loads(line);records={v['implementation_id']:v for v in row['prefill_source_profile']['implementation_records']};count=0
        require(set(records)==set(replay['prefix_implementation_dependencies']),'Actual execution source implementation set changed')
        for identity,parents in replay['prefix_implementation_dependencies'].items():
            for parent in parents:
                require(records[parent]['end_ns']<=records[identity]['start_ns'],'Actual execution violated selected physical predecessor edge')
                count+=1
        checked.append(dict(sample=row['sample'],warmup=row['warmup'],physical_predecessor_edges_checked=count))
    proof=dict(schema='orbit.qwen-r19-selected-ready-replay-proof.v1',program_graph_id=replay['program_graph_id'],replay_id=replay['replay_id'],
        shim=dict(file=str(Path(__file__).resolve()),sha256=sha(__file__)),actual_physical_predecessor_checks=checked,
        verification_cache=verification.receipt(),source_files_verified=True,selected_ready_priority=replay['full_ready_priority'],
        decode_order='Original autoregressive dispatch/argmax loop; only pure position RoPE may prepare ahead, without BO writes')
    (request.output/'selected-runtime-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    contract_path=request.output/'contract.json';contract=load(contract_path);contract['selected_ready_replay_proof']=proof
    contract_path.write_text(json.dumps(contract,indent=2)+'\n')


if __name__=='__main__':main()
