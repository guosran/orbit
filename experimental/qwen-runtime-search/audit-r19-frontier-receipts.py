#!/usr/bin/env python3
"""Inventory actual frontier admission metadata without loading model/captures.

This verifies execution/gate/spec bindings and pending-choice coverage. Numeric
arrays, local timing prices and full dependency bytes remain the responsibility
of the sealed independent auditor and strict fitter; neither is rerun here.
"""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):return json.loads(Path(path).read_text())
def record(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(file=str(path),sha256=digest)
def require(condition,message):
    if not condition:raise ValueError(message)
def option(command,key):
    require(command.count(key)==1,'Expected one command option '+key)
    return command[command.index(key)+1]


def inventory(prepared):
    execution_path=prepared/'campaign-execution.json';manifest=load(execution_path)
    summary_path=prepared/'campaign-receipts/summary.json';summary=load(summary_path)
    status_path=prepared/'campaign-receipts/campaign-status.json';status=load(status_path)
    frontier_record=manifest['frontier_round']['neighbors'];frontier=load(frontier_record['file'])
    require(record(frontier_record['file'])['sha256']==frontier_record['sha256'],'Frontier metadata changed')
    pending={row['id']:row for row in frontier['implementations']}
    require(len(pending)==manifest['frontier_round']['pending_choices'],'Pending count mismatch')
    receipts={row['name']:row for row in summary}
    require(len(receipts)==len(summary)==len(manifest['cohorts']),'Incomplete or duplicated actual cohort receipts')
    require(status==dict(completed_cohorts=len(summary),planned_cohorts=len(summary),stopped_on_seal_drift=False,any_failure=False,all_planned_cohorts_attempted=True),'Campaign status is not complete/pass')
    execution_record=record(execution_path);coverage={identity:[] for identity in pending};cohorts=[]
    for cohort in manifest['cohorts']:
        name=cohort['name'];receipt=receipts[name];command=cohort['command']
        require(receipt['manifest_sha256']==execution_record['sha256'] and receipt['command']==command and receipt['admission_command']==cohort['admission_command'],'Actual receipt command/manifest mismatch')
        require(receipt['purpose']=='training' and receipt['returncode']==0 and receipt['admission_returncode']==0 and receipt['sealed_dependencies_unchanged'] and receipt['included_in_prices'],'Cohort lacks successful execution/admission/seal')
        require(all(event['success'] for event in receipt['file_cache_advice']),'Actual verifier advice failed')
        require([option(command,k) for k in ('--outputs','--warmups','--repeats')]==['128','1','3'],'Original full128/warm1/repeat3 protocol changed')
        directory=Path(option(command,'--output'));gate_path=Path(cohort['numeric_gate_file']);gate=load(gate_path);completion_path=directory/'completion.json';completion=load(completion_path)
        require(gate['all_original_gates_pass'] and gate['code_files_current_match'] and gate['persistent_npu_owner_verified'],'Actual source/numeric/owner gate failed')
        require(completion['completed'] and completion['measured_requests']==3 and completion['warmups']==1,'Incomplete actual full-model process')
        require(len(gate['requests'])==3 and {r['sample'] for r in gate['requests']}=={0,1,2},'Measured sample set changed')
        for request in gate['requests']:
            require(request['source_operators']==581 and request['source_exactcover'] and request['phase_union_conserves'],'Actual source/phase gate failed')
            require(request['memory']['VmSwap_bytes']<=512*1024*1024 and request['memory']['VmRSS_bytes']<=20*1024**3,'Original memory guard exceeded')
        spec_path=Path(option(command,'--ready-spec'));spec=load(spec_path)
        actual=[]
        for row in spec['cpu_regions']:
            if row['id'] not in pending:continue
            expected=pending[row['id']]['runtime_config']
            require(all(row[k]==expected[k] for k in ('operators','backend','input_policy','output_policy')),'Pending choice runtime/source differs from actual spec')
            coverage[row['id']].append(dict(cohort=name,gate=record(gate_path),execution=record(directory/'completion.json')));actual.append(row['id'])
        cohorts.append(dict(name=name,receipt=record(prepared/'campaign-receipts'/(name+'.result.json')),gate=record(gate_path),completion=record(completion_path),spec=record(spec_path),actual_pending_choices=actual,measured_requests=3,warmups=1,outputs=128))
    require(all(coverage.values()),'Some pending choices have no passed actual full-model cohort')
    admission_path=prepared/'campaign-admission.json';admission=load(admission_path)
    return dict(schema='orbit.qwen-r19-frontier-actual-receipt-inventory.v1',producer=record(__file__),execution_campaign=execution_record,admission_campaign=record(admission_path),frontier=frontier_record,
        actual_summary=record(summary_path),campaign_status=record(status_path),cohorts=cohorts,pending_choice_coverage=coverage,pending_choices=len(pending),successful_new_cohorts=len(cohorts),strictfit_support_cohorts=len(admission['cohorts']),
        failed_new_cohorts=0,protocol=dict(outputs=128,warmups=1,measured_requests=3,original_guards_unchanged=True),
        all_pending_choices_actual_admitted=True,large_dependency_hashes_repeated=False,numeric_arrays_loaded=False,local_prices_fitted=False,
        scope='Metadata inventory of actual independently gated execution; does not substitute for numerical/capture checks or strict per-invocation price fitting')


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--prepared',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=inventory(args.prepared)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('pending_choices','successful_new_cohorts','strictfit_support_cohorts','failed_new_cohorts','all_pending_choices_actual_admitted')}));print(json.dumps(record(args.output)))


if __name__=='__main__':main()
