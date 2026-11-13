#!/usr/bin/env python3
"""Bind top-level ready scheduling to the already admitted actual executor.

Every calibrated variant already declares dependency-ready execution. The CLI
also requires an explicit catalogue-level policy and finite named priority seeds.
The frozen runtime accepts integer priorities, uses dependency-gated CPU/NPU
heaps, and enforces the same source DAG. Registering these seeds changes the
search domain, not measured callback prices or the runtime implementation.
"""
import argparse
import importlib.util
import json
from pathlib import Path


POLICIES=['source-order','reverse-source-order','cpu-first','npu-first','longest-first']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--input-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('r19_frozen_policy_evidence',Path(__file__).with_name('fit-r19-catalogue.py'))
    fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)
    parent=fit.record(a.input);fit.require(parent['sha256']==a.input_sha256,'Parent catalogue changed')
    c=fit.load(a.input);identity=c.pop('execution_registration_identity')
    fit.require(identity==fit.digest(c),'Parent registration identity differs')
    variants=[row['execution_variant'] for row in c['implementations']]
    proof=c['ready_runtime']['async_runtime_support_proof'];support=fit.load(fit.verified(proof))
    fit.require(support['identity']==proof['identity']==fit.digest({k:v for k,v in support.items() if k!='identity'}),
                'Actual runtime support proof changed')
    fit.require(all(v['execution_policy']=='source-dependency-ready' and
                v['source_ordering_identity']==c['source_id'] and
                v['async_runtime_proof_identity']==proof['identity'] for v in variants),
                'Variants do not share the same actual ready source/proof ABI')
    evidence=[]
    for name in ('qwen-r19-ready.py','qwen-r19-source-ready.py'):
        row=next(r for r in c['ready_runtime']['code_files'] if Path(r['file']).name==name)
        fit.verified(row);evidence.append(row)
    ready=Path(evidence[0]['file']).read_text();source_ready=Path(evidence[1]['file']).read_text()
    fit.require("type(op.priority)!=int" in ready and 'heapq.heappush' in ready and
                'node.priority,node.id' in ready and 'ready_priority' in source_ready,
                'Actual frozen runtime does not support registered integer-priority ready execution')
    previous=c['producer']
    c.update(producer=fit.record(__file__),execution_policy='source-dependency-ready',
        ready_priority_policies=POLICIES,source_ordering_identity=c['source_id'],
        async_runtime_proof_identity=proof['identity'],
        ready_policy_binding_upgrade=dict(schema='orbit.qwen-r19-ready-policy-binding-upgrade.v1',
            parent_catalogue=parent,parent_execution_registration_identity=identity,parent_producer=previous,
            actual_runtime_evidence=evidence,variants_checked=len(variants),
            registered_named_seeds=POLICIES,
            priority_scope='named seeds plus common explicit priority-key moves; actual runtime accepts integer keys after dependency readiness',
            hardware_remeasurement=False,prices_changed=False,runtime_registration_changed=False,
            support_proof_changed=False,common_guards_relaxed=False))
    c['execution_registration_identity']=fit.digest(c)
    with a.output.open('x') as stream:json.dump(c,stream,separators=(',',':'),allow_nan=False);stream.write('\n')
    receipt=dict(schema='orbit.qwen-r19-ready-policy-upgrade-receipt.v1',input=parent,output=fit.record(a.output),
        execution_registration_identity=c['execution_registration_identity'],producer=fit.record(__file__),
        variants_checked=len(variants),execution_policy=c['execution_policy'],ready_priority_policies=POLICIES,
        prices_changed=False,runtime_registration_changed=False,common_guards_relaxed=False)
    with a.output.with_suffix('.policy-upgrade-receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
