#!/usr/bin/env python3
"""CPU-only checked-graph sync API export contracts with synthetic test prices."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, default=taskflow_tool('core'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tool = args.tool.resolve()
    sync = tool.with_name('taskflow-sync-opt')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / 'TaskflowAIR/fixtures/heterogeneous.mlir'
    checks = []

    def save(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n')

    def read(path):
        return json.loads(path.read_text())

    def require(value, name):
        assert value, name
        checks.append(name)

    def run(name, command, success=True, diagnostic=None):
        save(out/(name+'.command.json'), command)
        result = subprocess.run(command, capture_output=True, text=True)
        (out/(name+'.stdout')).write_text(result.stdout)
        (out/(name+'.stderr')).write_text(result.stderr)
        save(out/(name+'.exit.json'), dict(returncode=result.returncode,device_submissions=0))
        require((result.returncode == 0) == success, name+': '+result.stderr[-1200:])
        if diagnostic:
            require(diagnostic in result.stderr, name+': exact diagnostic '+diagnostic)

    contract = out/'synthetic-contract.json'
    save(contract, dict(schema='orbit.cpu-test.execution-contract.v13',scope='synthetic compiler-only sync export; no measured calibration'))
    config = out/'config.json'
    save(config, dict(execution_contract=str(contract),heterogeneous=dict(task_devices=['npu']*3)))
    native = out/'native'
    run('rebuild-checked-graph', [str(tool),str(source),'--target-config='+str(config),'--heterogeneous-output='+str(native)])
    graph = read(native/'program-graph.json')
    ledger = graph['ledger']
    bindings = []
    for storage in (0,1,2):
        copies = [i for i,c in enumerate(ledger['copies'])
                  if ledger['allocations'][c['allocation']]['storage'] == storage and
                  ledger['allocations'][c['allocation']]['memory'] == 'system.shared']
        require(bool(copies), 'actual checked host copy exists for storage '+str(storage))
        bindings.append(dict(storage=storage,generation=0,observed_extent_bytes=64,storage_offset=0,
                             physical_id='synthetic-observed-bo-'+str(storage),extent_evidence='CPU-test explicit BO.size witness',copies=copies))
    runtime = dict(runtime_identity='synthetic-runtime-v14',observer_identity='synthetic-observer-v14',
                   condition_identity='synthetic-condition-v14',bindings=bindings)
    model = dict(schema='orbit.sync-api.model.v14',units='ns',model_id='synthetic-fixed-api-only',rows=[])
    for ordinal,direction in [(0,'TO_DEVICE'),(1,'TO_DEVICE'),(2,'TO_DEVICE'),(2,'FROM_DEVICE')]:
        model['rows'].append(dict(domain=dict(site='installed-bound-bo-input-loop' if direction=='TO_DEVICE' else 'installed-bound-bo-readback-loop',
            argument_ordinal=ordinal,call_form='whole-implicit',observer_boundary='detailed-compatibility',direction=direction,
            runtime_identity=runtime['runtime_identity'],observer_identity=runtime['observer_identity'],condition_identity=runtime['condition_identity'],
            allowed_states=['host-write-observed','host-write-since-sync','after-device-write-complete'],
            min_bytes=64,max_bytes=4096,min_count=1,max_count=1,alignment_bytes=64),
            formula='fixed-api-site',entry_ns=0,call_ns=100+ordinal,cacheline_ns=0,
            provenance='synthetic CPU regression, not hardware coefficients',raw_manifest_sha256='a'*64))

    def export(name, r=None, m=None, success=True, diagnostic=None, destination=None):
        rp=out/(name+'.bindings.json'); mp=out/(name+'.model.json')
        save(rp,runtime if r is None else r);save(mp,model if m is None else m)
        target=out/name if destination is None else destination
        run(name,[str(sync),str(source),'--target-config='+str(config),'--runtime-bindings='+str(rp),
                  '--sync-model='+str(mp),'--output='+str(target)],success,diagnostic)
        return read(target/'sync-cost.json') if success else None

    result = export('priced')
    require(read(out/'priced/program-graph.json') == graph, 'exported graph equals freshly rebuilt checked graph')
    trace = read(out/'priced/sync-boundary-trace.json')
    require(trace['graph_id'] == result['graph_id'] == graph['id'], 'trace and prices bind exact checked graph identity')
    queries=result['queries']
    require(len(queries)==4 and [q['request']['direction'] for q in queries] == ['TO_DEVICE']*3+['FROM_DEVICE'], 'exactly three TO and one FROM queries')
    require([q['request']['argument_ordinal'] for q in queries] == [0,1,2,2], 'actual materialized ABI argument order including output TO')
    require(all(q['request']['count']==1 for q in queries), 'each installed API request has count one')
    observations=trace['observations']
    kinds=[o.get('kind',o.get('observation',{}).get('kind')) for o in observations]
    require(kinds.count('device-submit')==kinds.count('wait-success')==1, 'one modeled successful submit/wait observation')
    require(result['unknown_sync_queries']==0 and all(q['evaluation']['status']=='predicted' and q['evaluation']['duration_ns']>0 for q in queries), 'all four exact-domain API spans priced positively')
    require(result['known_sync_api_subtotal_ns']==sum(q['evaluation']['duration_ns'] for q in queries), 'API subtotal equals only individual API estimates')
    require(result['whole_program_ns'] is None and result['physical_transfer_bytes'] is None and all(not q['evaluation']['complete_program_cost'] for q in queries), 'no whole-program or physical-traffic estimate invented')

    bad=copy.deepcopy(runtime);bad['bindings'].pop()
    export('missing-binding',bad,success=False,diagnostic='missing observed BO binding')
    bad=copy.deepcopy(runtime);bad['bindings'][0]['copies']=[]
    export('missing-alias',bad,success=False,diagnostic='absent from runtime aliases')
    bad=copy.deepcopy(runtime);bad['bindings'][1]['physical_id']=bad['bindings'][0]['physical_id']
    export('duplicate-physical-alias',bad,success=False,diagnostic='duplicate storage/physical BO')
    bad=copy.deepcopy(runtime);bad['bindings'][0]['observed_extent_bytes']=128
    padded=export('unknown-padding',bad)
    q=padded['queries'][0]
    require(padded['unknown_sync_queries']>=1 and q['evaluation']['duration_ns'] is None and q['evaluation']['status']!='predicted', 'unobserved padded extent remains unknown, never zero')
    for field in ('runtime_identity','observer_identity','condition_identity'):
        bad=copy.deepcopy(runtime);bad[field]='other-'+field
        mismatch=export('wrong-'+field,bad)
        require(mismatch['unknown_sync_queries']==4 and all(q['evaluation']['duration_ns'] is None for q in mismatch['queries']), field+' mismatch is OOD, not free')
    bad=copy.deepcopy(runtime);bad.pop('condition_identity')
    export('missing-condition',bad,success=False,diagnostic='runtime/observer/condition identities')
    bad=copy.deepcopy(runtime);bad['condition_identity']=''
    export('empty-condition',bad,success=False,diagnostic='runtime/observer/condition identities')
    bad_model=copy.deepcopy(model);bad_model['rows'][0]['domain']['runtime_identity']='not-requested-runtime'
    mismatch=export('model-request-runtime-domain',m=bad_model)
    require(mismatch['unknown_sync_queries']==1 and mismatch['queries'][0]['evaluation']['duration_ns'] is None, 'model domain runtime compared to actual requested runtime independently')
    export('existing-output',success=False,diagnostic='output directory must be new',destination=out/'priced')
    save(out/'summary.json',dict(status='passed',checks=checks,count=len(checks),device_submissions=0,synthetic_prices_only=True))
    print(json.dumps(dict(status='passed',checks=len(checks),device_submissions=0)))


if __name__ == '__main__':
    main()
