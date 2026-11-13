#!/usr/bin/env python3
"""Check fitter rejection boundaries using saved device trace intervals.

The small pilot supplies real timings only. Metadata adapters below deliberately
form structural unit fixtures; they are never training or numerical admission.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--whole-source-id',required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('fit_r19_test_backend',Path(__file__).with_name('fit-r19-catalogue.py'))
    fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)
    contract=fit.load(args.execution/'contract.json')
    row=json.loads((args.execution/'measurements.jsonl').read_text().splitlines()[0])
    profile=copy.deepcopy(row['prefill_source_profile'])
    profile.update(source_ordering_identity=args.whole_source_id,async_runtime_proof_identity='structural-fixture-only')
    for record in profile['implementation_records']:
        record['artifact_key']='structural-fixture-only'
        record['local_price_key']=fit.digest(dict(source_id=contract['source_id'],source_operators=record['source_operators'],
            kind=record['implementation_kind'],parameters=record['runtime_parameters'],artifact_key=record['artifact_key']))
    def check(value):return fit.audit_prefill(value,contract['source'],contract['plan_id'],args.whole_source_id,'structural-fixture-only')
    check(profile)
    rejected=0
    for change in ('union','span','key','cover','owner'):
        bad=copy.deepcopy(profile)
        if change=='union':bad['callback_interval_union_ns']+=1
        elif change=='span':bad['implementation_records'][0]['wall_ns']+=1
        elif change=='key':bad['implementation_records'][0]['local_price_key']='wrong'
        elif change=='cover':bad['implementation_records'][0]['source_operators']*=2
        else:
            first=next(e for e in bad['execution_trace']['events'] if e['device']=='npu')
            first['owner_thread']=-1
        try:check(bad)
        except ValueError:rejected+=1
        else:raise AssertionError('Corrupt '+change+' accepted')
    halves=0
    for record in profile['implementation_records']:
        if record['implementation_kind']!='heterogeneous-gate-up':continue
        device=next(e for e in record['events'] if e['device']=='npu')
        fit.audit_private_xrt(device['phases'][0])
        template=fit.half_template(record,contract['source'],{})
        parts=template['execution_variant']['work_partitions']
        assert sum(p['extents'][0]*p['extents'][1] for p in parts)==5*22016
        assert len(parts)==template['execution_variant']['replicas']
        assert template['execution_variant']['temporary_bytes']['runtime-abi']>=2048*11008*6
        halves+=1
    try:fit.price_summary([], 'missing-price')
    except ValueError:rejected+=1
    else:raise AssertionError('Absent r19 prices became zero')
    print(json.dumps(dict(structural_fixture_only=True,original_trace_union_checked=True,
                          corrupt_cases_rejected=rejected,real_half_ledgers_checked=halves)))


if __name__=='__main__':main()
