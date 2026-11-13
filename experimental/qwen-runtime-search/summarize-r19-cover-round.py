#!/usr/bin/env python3
"""Inventory actual frontier receipts without admitting prices or dropping failures.

A failed whole-program cohort is an unresolved experiment for every choice it
contains. It is never evidence that each individual compiler choice is illegal.
Successful rows still require the strict fitter's numerical/provenance audit.
"""
import argparse
import hashlib
import json
from pathlib import Path


def require(condition,message):
    if not condition:raise ValueError(message)


def record(path):
    requested=Path(path).absolute();canonical=requested.resolve(strict=True)
    with requested.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    value=dict(file=str(canonical),sha256=checksum)
    if requested!=canonical:value['requested_file']=str(requested)
    return value


def inventory(manifest,neighbors,receipts,manifest_sha256):
    planned={row['name']:row for row in manifest['cohorts']}
    require(len(planned)==len(manifest['cohorts']),'Duplicate planned cohort names')
    actual={row['name']:row for row in receipts}
    require(len(actual)==len(receipts) and set(actual)<=set(planned),'Duplicate or unexpected actual receipt')
    statuses={};by_neighbor={row['neighbor_cohort_id']:row['name'] for row in planned.values()}
    require(len(by_neighbor)==len(planned),'Duplicate frontier experiment mapping')
    for name,row in planned.items():
        result=actual.get(name)
        if result is None:status='not-attempted'
        else:
            for key in ('command','admission_command','purpose'):
                require(result[key]==row[key],'Actual receipt differs from planned '+key)
            require(result['manifest_sha256']==manifest_sha256,'Actual receipt binds another manifest')
            require(result['driver_sha256']==manifest['independent_execution_policy']['driver']['sha256'],
                    'Actual receipt binds another driver')
            passed=(result.get('returncode')==0 and result.get('admission_returncode')==0
                    and result.get('sealed_dependencies_unchanged') is True)
            require(result.get('included_in_prices') is passed,'Inconsistent driver price-admission flag')
            status='passed-awaiting-strict-fit' if passed else 'failed-unresolved'
        statuses[name]=dict(status=status,actual_receipt=result)
    choices={row['id']:[] for row in neighbors['implementations']}
    for cohort in neighbors['cohorts']:
        require(cohort['id'] in by_neighbor,'Neighbor cohort has no actual experiment')
        name=by_neighbor[cohort['id']]
        for region in cohort['cpu_regions']:
            if region['id'] in choices:choices[region['id']].append(name)
    result=[]
    for choice,names in choices.items():
        require(names,'Pending choice has no experiment witness')
        passed=any(statuses[name]['status']=='passed-awaiting-strict-fit' for name in names)
        result.append(dict(implementation_id=choice,cohorts=names,
            status='passed-awaiting-strict-fit' if passed else 'unresolved',
            independently_unsupported=False))
    return dict(cohorts=statuses,choices=result,closure_established=False,
        prices_admitted=False,unsupported_choices=[],
        unresolved_choices=sum(row['status']=='unresolved' for row in result),
        scope='Receipt inventory only; strict fit and repeated incumbent neighborhood search remain required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('execution','neighbors','receipts','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();load=lambda path:json.loads(path.read_text())
    # The driver copied these exact manifest bytes before any board execution.
    require(a.execution.read_bytes()==(a.receipts/'manifest.json').read_bytes(),
            'Receipt directory contains a different execution manifest')
    execution=record(a.execution)
    value=inventory(load(a.execution),load(a.neighbors),load(a.receipts/'summary.json'),execution['sha256'])
    value.update(schema='orbit.qwen-r19-frontier-receipt-inventory.v1',execution=execution,
        neighbors=record(a.neighbors),receipts=record(a.receipts/'summary.json'),producer=record(__file__))
    with a.output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(a.output),unresolved_choices=value['unresolved_choices'],closure_established=False)))


if __name__=='__main__':main()
