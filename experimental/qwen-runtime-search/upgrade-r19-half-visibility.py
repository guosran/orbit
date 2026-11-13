#!/usr/bin/env python3
"""Repair one missing source-bound ABI classification without repricing.

PrivateGemm has already completed, synchronized and cloned its output before
ready CPU concat assembles the logical gate_up value. That real host output
requires host-materialized visibility. This producer preserves all calibrated
prices, proofs, runtime registrations and code identities, retaining the original
fitter producer rather than claiming to have rerun its numerical/price audit.
"""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--input-sha256',required=True)
    parser.add_argument('--fit-receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    frozen=Path(__file__).with_name('fit-r19-catalogue.py')
    spec=importlib.util.spec_from_file_location('r19_original_frozen_fitter',frozen)
    fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)
    parent_record=fit.record(args.input)
    fit.require(parent_record['sha256']==args.input_sha256,'Input catalogue bytes differ')
    catalogue=fit.load(args.input);old_identity=catalogue.pop('execution_registration_identity')
    fit.require(old_identity==fit.digest(catalogue),'Input catalogue identity differs')
    fit.require(catalogue['producer']==fit.record(frozen),'Original frozen fitter producer changed')
    receipt=fit.load(args.fit_receipt)
    fit.require(receipt['catalogue']['sha256']==parent_record['sha256'] and
                receipt['execution_registration_identity']==old_identity and
                receipt['missing_local_prices']==0 and receipt['old_prices_reused'] is False,
                'Original successful strict fit receipt does not bind input')
    support_record=catalogue['ready_runtime']['async_runtime_support_proof']
    support=fit.load(fit.verified(support_record))
    fit.require(support_record['identity']==support['identity']==fit.digest({k:v for k,v in support.items() if k!='identity'}),
                'Actual runtime support proof differs')
    evidence=[]
    for name in ('qwen-r19-source-ready.py','qwen-r19-xrt.py'):
        row=next(r for r in catalogue['ready_runtime']['code_files'] if Path(r['file']).name==name)
        fit.verified(row);evidence.append(row)
    source=ast.parse(Path(evidence[0]['file']).read_text())
    assemble=[n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='assemble']
    fit.require(len(assemble)==1 and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr=='cat'
                for n in ast.walk(assemble[0])),'Actual ready host concat implementation changed')
    original=copy.deepcopy(catalogue['implementations']);patched=[]
    for row in catalogue['implementations']:
        registration=catalogue['runtime_registration']['implementations'][row['id']]
        if registration['kind']!='heterogeneous-gate-up':continue
        fit.require('visibility_policy' not in row and row['device']=='npu' and
                    row['cost_boundary']=='inclusive-region-wall' and
                    row['transport_ownership']=='implementation-inclusive' and row['launch_count']==1,
                    'Unexpected half ABI; this repair is not applicable')
        fit.require('common does not independently schedule' in row['physical_evidence']['scheduling_scope'],
                    'Half implementation is not the already audited opaque compound')
        row['visibility_policy']='host-materialized';patched.append(row['id'])
    fit.require(patched,'No missing heterogeneous visibility bindings')
    for before,after in zip(original,catalogue['implementations']):
        expected=copy.deepcopy(before)
        if before['id'] in patched:expected['visibility_policy']='host-materialized'
        fit.require(expected==after,'Upgrade changed calibrated implementation fields')
    old_producer=catalogue['producer']
    catalogue.update(producer=fit.record(__file__),original_fit_producer=old_producer,
        implementation_schema_upgrade=dict(schema='orbit.qwen-r19-host-visibility-binding-upgrade.v1',
            parent_catalogue=parent_record,parent_execution_registration_identity=old_identity,
            original_fit_receipt=fit.record(args.fit_receipt),runtime_evidence=evidence,
            patched_implementations=patched,field='visibility_policy',value='host-materialized',
            reason='actual private XRT completion/sync/readback/host clone precedes CPU torch.cat output assembly',
            hardware_remeasurement=False,prices_changed=False,runtime_registration_changed=False,
            support_proof_changed=False,common_guards_relaxed=False))
    catalogue['execution_registration_identity']=fit.digest(catalogue)
    with args.output.open('x') as stream:json.dump(catalogue,stream,separators=(',',':'),allow_nan=False);stream.write('\n')
    result=dict(schema='orbit.qwen-r19-schema-binding-upgrade-receipt.v1',input=parent_record,
        output=fit.record(args.output),execution_registration_identity=catalogue['execution_registration_identity'],
        producer=fit.record(__file__),patched_implementations=len(patched),
        prices_changed=False,runtime_registration_changed=False,common_guards_relaxed=False)
    output=args.output.with_suffix('.binding-upgrade-receipt.json')
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
