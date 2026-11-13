#!/usr/bin/env python3
"""Expand measured CPU compiler templates over actual legal Qwen source covers.

Enumeration is graph-derived. Only templates with captured native compiler
code can become pending implementations; the output never invents numerical
admission or prices for the other layers. Whole-program execution supplies
those receipts before a common ORBIT catalogue may mark a choice executable.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--pilot-registration',type=Path,required=True)
    parser.add_argument('--backend',type=Path,default=Path(__file__).with_name('qwen-r19-regions.py'))
    parser.add_argument('--max-operators',type=int,required=True)
    parser.add_argument('--all-policies',action='store_true',
                        help='Emit eager/Inductor × real input-copy/borrow × legal output-copy/retain choices')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('r19_cpu_region_registration',args.backend)
    backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    source=json.loads(args.source.read_text());program=backend.Program(source)
    pilot=json.loads(args.pilot_registration.read_text())
    templates={}
    for impl in pilot['implementations']:
        proof=impl['compile_proof']
        backend.require(proof['producer']['sha256']==backend.file_identity(args.backend)['sha256'],
                        'Compiler-template producer changed')
        backend.require(proof['region']['source_id']==source['source_id'],'Pilot source mismatch')
        for artifact in proof['generated_code']:
            backend.require(backend.file_identity(artifact['file'])['sha256']==artifact['sha256'],
                            'Captured native wrapper changed')
        key=proof['template_id']
        if key in templates:
            backend.require([r['sha256'] for r in templates[key]['generated_code']]==
                            [r['sha256'] for r in proof['generated_code']],
                            'Equal structural template emitted different code; retain separate specializations')
        else:templates[key]=proof
    candidates=[];enumerated=0;matched=0
    for proof in backend.enumerate_regions(program,max_operators=args.max_operators):
        enumerated+=1
        # Construction contains no model weights and does not compile or run
        # tensors. Its canonical template is derived from exact source DAG,
        # operation attributes and boundary shapes/dtypes.
        executable=backend.ExecutableRegion(program,proof['operators'],backend='inductor-cpu')
        if executable.template_id not in templates:continue
        matched+=1
        original=templates[executable.template_id]
        compiled=copy.deepcopy(original)
        compiled['template_reused_from_region_id']=original['region']['region_id']
        compiled['region']=executable.proof
        policies=([dict(backend=mode,input_policy=staging,output_policy=ownership)
                   for mode in ('eager-host-group','inductor-cpu')
                   for staging in ('borrow-host','stage-host-copy')
                   for ownership in ('retain-host','copy-host')
                   if not(ownership=='copy-host' and proof['exported_views'])]
                  if args.all_policies else [dict(backend='inductor-cpu',input_policy='borrow-host',output_policy='retain-host')])
        for policy in policies:
            variant=backend.ExecutableRegion(program,proof['operators'],**policy)
            candidate=backend.registration(variant,compiled if policy['backend']=='inductor-cpu' else None)
            candidate['template_admission']='same checked arithmetic DAG/attributes/shape ABI; actual input strides and preparation specialization must be witnessed by the whole-program runtime before admission'
            candidate['requires_actual_input_specialization_receipt']=policy['backend']=='inductor-cpu'
            candidate['price_trace_key']={key:candidate['runtime_config'][key]
                                          for key in ('region_id','backend','input_policy','output_policy')}
            candidates.append(candidate)
    result=dict(schema='orbit.qwen-r19-cpu-region-registration.v1',
        source_id=source['source_id'],source=backend.file_identity(args.source),
        producer=backend.file_identity(__file__),backend=backend.file_identity(args.backend),
        pilot=backend.file_identity(args.pilot_registration),
        enumeration=dict(min_operators=2,max_operators=args.max_operators,
                         legal_covers_enumerated=enumerated,
                         actual_compiled_templates=len(templates),
                         covers_matching_compiled_templates=matched,
                         implementation_policy_choices=len(candidates),
                         scope='explicit finite cover-size neighborhood; not all possible fusion sizes'),
        implementations=candidates,
        admission='pending unchanged complete-Qwen numerical gate and fresh complete-region wall prices')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['enumeration']),flush=True)


if __name__=='__main__':main()
