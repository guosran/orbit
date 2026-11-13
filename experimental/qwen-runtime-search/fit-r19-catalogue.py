#!/usr/bin/env python3
"""Fit fresh r19 ready-execution prices and source-bound implementation choices.

The r18 catalogue contributes source ABI, target limits and physical backend
templates only. Every exported duration comes from an admitted r19 training
callback/compound span or conserved decoder phase. Missing prices are errors.
CPU compiler proofs must come from actual whole-program input preparation.
"""
import argparse
import ast
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


_CHECKSUM_MODULE=None


def verifier_runtime():
    """Reuse the frozen campaign's exact read-only hash/cache-advice protocol."""
    global _CHECKSUM_MODULE
    if _CHECKSUM_MODULE is None:
        _CHECKSUM_MODULE=module(Path(__file__).with_name('run-independent-r19-campaign-v2.py'),
                               'r19_fitter_shared_verifier')
    return _CHECKSUM_MODULE


def record(path):
    requested=Path(path).absolute();canonical=requested.resolve(strict=True)
    result=dict(file=str(canonical),sha256=verifier_runtime().checksum(requested))
    # The loader may use the original symlink spelling. Preserve it whenever
    # different from the declared canonical file, even for identical bytes.
    if str(requested)!=str(canonical):result['requested_file']=str(requested)
    return result


def require(condition,message):
    if not condition:raise ValueError(message)


def verified(row):
    path=row.get('file',row.get('path'))
    require(path,'Evidence has no actual file')
    normalized=dict(row,file=path)
    # Reuse the same fail-closed requested-alias and cache-neutral SHA checker
    # as actual campaign execution. Retargeting to an equal-content blob must
    # fail before a loader follows a different physical file.
    verifier_runtime().verify([normalized])
    return Path(normalized.get('requested_file',path))


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def nanoseconds(value):
    require(type(value) is int and value>=0,'Invalid duration or timestamp')
    return value


def union_ns(events):
    intervals=[]
    for event in events:
        lo=nanoseconds(event['start_ns']);hi=nanoseconds(event['end_ns'])
        require(hi>=lo and event['wall_ns']==hi-lo,'Callback clock mismatch')
        intervals.append((lo,hi))
    total=0;left=right=None
    for lo,hi in sorted(intervals):
        if right is None:left,right=lo,hi
        elif lo<=right:right=max(right,hi)
        else:total+=right-left;left,right=lo,hi
    return total+(right-left if right is not None else 0)


PREPARATION_MODES=('owned-decode-release-readonly-prefault-v1',
                   'owned-prefill-decode-release-readonly-prefault-v2')


def audit_preparation(directory,command,code_files):
    """Bind untimed preparation separately from the unchanged timed runtime.

    Earlier admitted cohorts remain explicit original-only observations. Later
    protocols retain their actual stages and proof bytes; a first-stage failure
    is never enough to admit a supposedly complete two-stage preparation.
    """
    entry=record(command[1]);sidecar=directory/'preparation-protocol.json'
    original=next(row for row in code_files if Path(row['file']).name=='run-qwen-r19-ready.py')
    if not sidecar.exists():
        require(entry==original,'Unknown preparation entrypoint without its actual sidecar')
        return dict(mode='original-only',entrypoint=entry,timed_runtime_unchanged=True)
    protocol=load(sidecar);mode=protocol['mode']
    require(protocol['schema']=='orbit.qwen-r19-preparation-protocol.v1' and
            mode in PREPARATION_MODES and protocol['outside_request_timing'] is True and
            protocol['changes_guard'] is False and protocol['changes_timed_runtime'] is False,
            'Preparation changed a guard or inference timing boundary')
    verified(protocol['producer'])
    require('--runtime-runner' in command and record(command[command.index('--runtime-runner')+1])==original,
            'Preparation shim does not load the sealed ready runner')
    require(protocol['producer']==entry,'Preparation sidecar does not bind the actual command entrypoint')
    two_stage=mode==PREPARATION_MODES[1]
    receipts=protocol['proofs'] if two_stage else [protocol['proof']]
    if two_stage:
        require([row['stage'] for row in receipts]==['prefill-loaded-original-cpu-weights','decode-loaded'] and
                {k:receipts[-1][k] for k in ('file','sha256')}==protocol['proof'],
                'Two-stage preparation did not finish both original guard boundaries')
    ast_change=('two owned preparation calls: after original pf.load_weights before unchanged prefill-loaded-original-cpu-weights guard; after original FusedDecoder before unchanged decode-loaded guard'
        if two_stage else 'one owned preparation call after original FusedDecoder constructor before unchanged decode-loaded guard')
    for receipt in receipts:
        proof=load(verified(receipt))
        require(proof['schema']=='orbit.qwen-r19-preparation-memory-release.v1' and
                proof['inside_request_timing'] is False and proof['reads_only'] is True and
                proof['bo_writes']==0 and proof['owned_shim']==entry and
                (not two_stage or proof['stage']==receipt['stage']),
                'Preparation was not an untimed readonly operation')
        underlying=proof['underlying_runner'];verified(underlying)
        require(underlying['ast_change']==ast_change and underlying['original_main_ast_sha256'] and
                {k:underlying[k] for k in ('file','sha256')}==record(Path(command[command.index('--frozen-runtime')+1])/'scripts/run-qwen-r18-runtime.py'),
                'Preparation AST does not bind the unchanged sealed runner')
        for phase in ('before','after_dead_page_release','after_live_weight_prefault','after'):
            state=proof[phase]
            require(all(type(state[k]) is int and state[k]>=0 for k in
                        ('VmRSS_bytes','VmSwap_bytes','available_bytes','minor_faults','major_faults')),
                    'Missing actual preparation memory observations')
        require(proof['live_storage_union_bytes']==sum(row['bytes'] for row in proof['storage_roots']) and
                proof['page_bytes']>0 and proof['pages_touched']>=0 and
                all(row['owners'] and row['bytes']>0 for row in proof['storage_roots']),
                'Preparation storage union is not backed by actual owners')
    return dict(mode=mode,entrypoint=entry,protocol=record(sidecar),
                proof=protocol['proof'],proofs=receipts,timed_runtime_unchanged=True)


def audit_execution_receipts(manifest,roots):
    """Join actual driver receipts, retaining historical verifier differences."""
    expected={row['name']:row for row in manifest['cohorts'] if row['purpose']=='training'}
    bound={};aggregate=[]
    def attach(cohort,receipt,evidence):
        name=cohort['name']
        require(name in expected and cohort['command']==expected[name]['command'] and
                receipt['command']==expected[name]['command'] and
                receipt['purpose']==expected[name]['purpose'] and receipt['name']==name,
                'Execution receipt does not bind the admitted actual command/purpose')
        require('admission_command' not in receipt or
                receipt['admission_command']==expected[name]['admission_command'],
                'Execution receipt used a different admission command')
        require(receipt['returncode']==0 and receipt['admission_returncode']==0 and
                receipt['sealed_dependencies_unchanged'] is True and
                receipt.get('included_in_prices',True) is True,
                'Unadmitted driver execution cannot contribute any price')
        require(name not in bound,'Duplicate execution receipt for admitted cohort: '+name)
        if 'verification_cache_policy' in receipt:
            require(receipt['verification_cache_policy']==verifier_runtime().CACHE_POLICY and
                    all(e['success'] is True and e['operation']=='POSIX_FADV_DONTNEED' and
                        e['threshold_bytes']==64*2**20 and e['bytes']>=e['threshold_bytes']
                        for e in receipt['file_cache_advice']),
                    'Unproved or failed actual verifier cache advice')
        bound[name]=dict(actual_receipt=receipt,evidence=evidence,
            verifier_policy_recorded='verification_cache_policy' in receipt)
    for backing in manifest.get('retained_receipt_backing',[]):
        cohort=backing['cohort'];receipt=backing['execution_receipt']
        execution_manifest=load(verified(backing['execution_manifest']))
        receipts=load(verified(backing['execution_receipts']))
        require(receipt in receipts and receipt['manifest_sha256']==backing['execution_manifest']['sha256'] and
                any(c['name']==cohort['name'] and c['command']==cohort['command'] for c in execution_manifest['cohorts']),
                'Retained receipt is not present in its exact original execution manifest/summary')
        attach(cohort,receipt,[backing['execution_manifest'],backing['execution_receipts']])
    for root in roots:
        root=Path(root);manifest_record=record(root/'manifest.json');execution_manifest=load(root/'manifest.json')
        require(execution_manifest['schema']=='orbit.qwen-r19-sealed-campaign.v1','Unknown execution manifest copy')
        driver=execution_manifest['independent_execution_policy']['driver'];verified(driver)
        require(driver['sha256']==record(verifier_runtime().__file__)['sha256'],
                'Execution receipts use a different verifier protocol producer')
        cache_record=record(root/'verification-cache-advice.json');cache=load(root/'verification-cache-advice.json')
        require(cache['policy']==verifier_runtime().CACHE_POLICY and all(e['success'] is True for e in cache['events']),
                'Campaign verifier cache advice did not complete successfully')
        aggregate.append(dict(manifest=manifest_record,cache_advice=cache_record,driver=driver))
        for cohort in execution_manifest['cohorts']:
            require(cohort['name'] in expected,'Receipts include a cohort outside the admitted campaign')
            path=root/(cohort['name']+'.result.json');receipt=load(path)
            require(receipt['manifest_sha256']==manifest_record['sha256'] and
                    receipt['driver_sha256']==driver['sha256'],
                    'Current execution receipt manifest/driver identity mismatch')
            attach(cohort,receipt,[manifest_record,record(path),cache_record])
    if roots:
        require(set(bound)==set(expected),'Missing actual execution receipt for an admitted cohort')
    return bound,aggregate


def signature(names,kind,p):
    """Match a static backend template to its actual ready-run parameters."""
    if kind=='prefill-original':parts=[p['device']]
    elif kind=='prefill-gemm':parts=[p['manifest_sha256']]
    elif kind=='prefill-fused-ffn':
        parts=[p['layer'],p['retention'],p['split_order'],p['ffn_artifact']['sha256']]
    elif kind=='cpu-region':parts=[p['backend'],p['input_policy'],p['output_policy']]
    elif kind=='heterogeneous-gate-up':parts=[p['cpu_half'],p['manifest_sha256']]
    else:raise ValueError('Unknown prefill implementation family: '+kind)
    return digest(dict(source_operators=names,kind=kind,parameters=parts))


def audit_prefill(profile,prefill,plan_id,whole_source_id,runtime_proof_id):
    require(profile['schema']=='orbit.qwen-r19-ready-source-profile.v1' and
            profile['source_id']==prefill['source_id'] and profile['plan_id']==plan_id,
            'Profile source/plan/protocol mismatch')
    require(profile['execution_policy']=='source-dependency-ready' and
            profile['source_ordering_identity']==whole_source_id and
            profile['async_runtime_proof_identity']==runtime_proof_id,'Wrong execution protocol/proof')
    expected={op['id'] for op in prefill['operators']}
    records=profile['implementation_records'];covered=[n for row in records for n in row['source_operators']]
    require(len(covered)==len(set(covered)) and set(covered)==expected and
            profile['source_operators']==len(expected),'Incomplete or overlapping source covers')
    trace=profile['execution_trace'];events=trace['events']
    require(trace['schema']=='orbit.qwen-r19-ready-execution.v1' and
            trace['execution_policy']=='source-dependency-ready','Unknown ready trace')
    require(len({event['operator'] for event in events})==len(events),'Duplicate actual callback')
    byid={event['operator']:event for event in events}
    seen=[]
    for row in records:
        parts=row['events'];require(parts,'Implementation has no actual callback')
        kind=row['implementation_kind']
        if kind=='heterogeneous-gate-up':expected_devices=['cpu','cpu','npu']
        elif kind=='cpu-region':expected_devices=['cpu']
        elif kind=='prefill-original':expected_devices=[row['runtime_parameters']['device']]
        elif kind in ('prefill-gemm','prefill-fused-ffn'):expected_devices=['npu']
        else:raise ValueError('Unknown measured implementation kind: '+kind)
        require(sorted(part['device'] for part in parts)==sorted(expected_devices),
                'Actual callback counts do not match the registered controller-feature proxy')
        require(all(part==byid.get(part['operator']) for part in parts),'Record/callback mismatch')
        seen.extend(part['operator'] for part in parts)
        lo=min(part['start_ns'] for part in parts);hi=max(part['end_ns'] for part in parts)
        require((row['start_ns'],row['end_ns'],row['wall_ns'])==(lo,hi,hi-lo),
                'Compound price is not the actual callback span')
        require(row['local_price_key']==digest(dict(source_id=prefill['source_id'],source_operators=row['source_operators'],
            kind=row['implementation_kind'],parameters=row['runtime_parameters'],artifact_key=row['artifact_key'])),
            'Runtime source/cover/artifact price-key mismatch')
    require(len(seen)==len(set(seen)) and set(seen)==set(byid),'Callbacks not partitioned by source implementations')
    total=union_ns(events)
    require(total==profile['callback_interval_union_ns'] and
            total+nanoseconds(profile['host_control_ns'])==profile['wall_ns'] and
            profile['exclusive_sum_equals_wall'],'Prefill interval-union conservation failed')
    require(trace['end_ns']-trace['start_ns']==trace['wall_ns'] and
            trace['wall_ns']<=profile['wall_ns'] and
            all(trace['start_ns']<=e['start_ns']<=e['end_ns']<=trace['end_ns'] for e in events),
            'Callback outside enclosing ready invocation')
    for device in ('cpu','npu'):
        local=sorted((e for e in events if e['device']==device),key=lambda e:e['start_ns'])
        require(all(a['end_ns']<=b['start_ns'] for a,b in zip(local,local[1:])),
                'Exclusive execution owner ran overlapping callbacks')
        require(len({e['owner_thread'] for e in local})<=1,'Multiple owners for one ready execution resource')
    owners={device:{e['owner_thread'] for e in events if e['device']==device} for device in ('cpu','npu')}
    require(not owners['cpu'].intersection(owners['npu']),'CPU main and NPU owner are not independent')
    overlap=sum(max(0,min(a['end_ns'],b['end_ns'])-max(a['start_ns'],b['start_ns']))
                for a in events if a['device']=='cpu' for b in events if b['device']=='npu')
    require(overlap==trace['measured_cpu_npu_callback_overlap_ns'],'Callback overlap arithmetic mismatch')
    require(sum(e['wall_ns'] for e in events)==profile['concurrent_callback_wall_sum_ns'],
            'Callback wall sum mismatch')
    implementation_union=union_ns(records)
    require(total<=implementation_union<=profile['wall_ns'],'Implementation spans do not cover their actual callbacks')
    return records,dict(callbacks=len(events),covers=len(records),union_ns=total,
        cpu_callbacks=sum(e['device']=='cpu' for e in events),
        npu_callbacks=sum(e['device']=='npu' for e in events),
        callback_sum_ns=profile['concurrent_callback_wall_sum_ns'],overlap_ns=overlap,
        host_control_ns=profile['host_control_ns'],implementation_span_union_ns=implementation_union,
        opaque_internal_idle_ns=implementation_union-total,
        outside_implementation_control_ns=profile['wall_ns']-implementation_union)


def audit_private_xrt(event):
    require(event['schema']=='orbit.qwen-r19-private-xrt-invocation.v1','Missing actual private XRT proof')
    phases=event['phases'];total=union_ns(phases)
    require(total==sum(p['wall_ns'] for p in phases),'Private XRT exclusive phases overlap')
    require(total+nanoseconds(event['exclusive_control_ns'])==event['wall_ns'] and
            event['end_ns']-event['start_ns']==event['wall_ns'] and event['exclusive_sum_equals_wall'],
            'Private XRT phase conservation failed')
    require(event['start_ns']<=event['bo_lifetime']['allocated_ns']<=event['launched_ns']<=
            event['completed_ns']<=event['host_visible_ns']<=event['bo_lifetime']['released_ns']<=event['end_ns'],
            'BO/context lifetime does not cover actual device completion/readback')
    require({'launch','complete','sync-to-device','sync-from-device','host-output-clone',
             'bo-and-context-release'}<={p['kind'] for p in phases},'Missing XRT lifecycle phase')


def audit_decode(phase,position,prefetch,last):
    require(phase['position']==position and phase['exclusive_sum_equals_wall'],'Decode position/conservation mismatch')
    events=phase['region_events']
    require(len(events)==1 and events[0]['region']==dict(lo=0,hi=61) and
            events[0]['instruction_patch_ns']==0,'Unsupported decoder partition or duplicate instruction patch')
    event=events[0];service=nanoseconds(event['submit_ns'])+nanoseconds(event['wait_ns'])
    require(event['service_ns']==service and service+nanoseconds(phase['host_before_ns'])+
            nanoseconds(phase['host_after_ns'])+nanoseconds(phase['host_control_ns'])==phase['wall_ns'],
            'Decode phase conservation failed')
    if prefetch and not last:
        ready=phase['ready_execution'];union=union_ns(ready['events'])
        require(union<=ready['wall_ns'] and ready['wall_ns']==ready['end_ns']-ready['start_ns'],
                'Decode ready interval bounds mismatch')
        cpu=[e for e in ready['events'] if e['device']=='cpu'];require(len(cpu)==1,'Missing pure-host prefetch task')
        pure=cpu[0]['phases'][0]
        require(pure['kind']=='next-position-pure-host-rope' and pure['position']==position+1 and
                pure['bo_writes']==0 and phase['prefetch_bo_writes']==0,'Prefetch wrote active device state')
        launch=phase['launch_boundary_ns'];complete=phase['device_complete_ns']
        require(launch[0]<=launch[1]<=complete and pure['end_ns']-pure['start_ns']==pure['wall_ns'],
                'Invalid actual launch/completion/prefetch interval')
        overlap=max(0,min(pure['end_ns'],complete)-max(pure['start_ns'],launch[1]))
        require(phase['pure_host_rope_device_overlap_ns']==overlap,'Device-active RoPE overlap arithmetic mismatch')
    elif prefetch:
        # The final token still uses the ready NPU owner, but deliberately has
        # no speculative next-position CPU task. This is not prefetch activity.
        ready=phase['ready_execution'];events=ready['events']
        require(ready['schema']=='orbit.qwen-r19-ready-execution.v1' and len(events)==1 and
                events[0]['device']=='npu' and events[0]['operator']==f'decode.{position}.dispatch' and
                union_ns(events)<=ready['wall_ns'] and ready['wall_ns']==ready['end_ns']-ready['start_ns'] and
                phase.get('pure_host_rope_device_overlap_ns',0)==0 and phase.get('prefetch_bo_writes',0)==0 and
                not events[0].get('phases',[]),
                'Final decode must contain only its actual ready device invocation')
    else:require('ready_execution' not in phase,'Unexpected unregistered decoder prefetch')
    return service


def merge_residents(variant,extra):
    for name,value in extra.items():
        require(name not in variant.setdefault('resident_allocations',{}) or
                variant['resident_allocations'][name]==value,'Conflicting actual physical allocation identity')
        variant['resident_allocations'][name]=copy.deepcopy(value)


def price_summary(values,key):
    require(len(values)>=3,'Missing three fresh r19 samples: '+str(key))
    return dict(samples=len(values),median_ns=statistics.median(values),minimum_ns=min(values),maximum_ns=max(values),key=key)


def fit_controller(observations,source_id,campaign_record):
    """Fit measured non-overlapped controller time, never whole-request price.

    There are only three coefficients. Enumerating every active subset gives
    an explicit nonnegative least-squares solution without a solver dependency.
    This is a candidate-dependent approximation whose error needs heldout runs.
    """
    import numpy as np
    require(observations,'No controller observations')
    x=np.asarray([[1,row['cpu_callbacks'],row['npu_callbacks']] for row in observations],dtype=np.float64)
    y=np.asarray([row['fixed_ns']/1e6 for row in observations],dtype=np.float64)
    best=None
    for mask in range(8):
        columns=[index for index in range(3) if mask&(1<<index)]
        weights=np.zeros(3,dtype=np.float64)
        if columns:
            solved=np.linalg.lstsq(x[:,columns],y,rcond=None)[0]
            if np.any(solved < -1e-10):continue
            weights[columns]=np.maximum(solved,0)
        residual=x@weights-y;error=float(residual@residual)
        candidate=(error,len(columns),mask)
        if best is None or candidate<best[0]:best=(candidate,weights)
    require(best is not None,'Controller NNLS did not find a nonnegative model')
    weights=best[1];predictions=x@weights;residuals=predictions-y
    samples=[dict(cohort=row['cohort'],sample=row['sample'],cpu_callbacks=row['cpu_callbacks'],
        npu_callbacks=row['npu_callbacks'],preparation_mode=row['preparation_mode'],measured_ms=float(y[index]),
        predicted_ms=float(predictions[index]),residual_ms=float(residuals[index]))
        for index,row in enumerate(observations)]
    model=dict(schema='orbit.program-invocation-control-model.v1',
        source_id=source_id,producer=record(__file__),training_campaign=campaign_record,
        intercept_ms=float(weights[0]),coefficients=dict(cpu_callbacks=float(weights[1]),npu_callbacks=float(weights[2])),
        boundary='outside-inclusive-invocation-wall',
        fit=dict(method='nonnegative least squares; exhaustive active subsets of three coefficients',
            observations=len(samples),active_mask=best[0][2],
            rmse_ms=float(np.sqrt(np.mean(residuals**2))),
            mean_absolute_error_ms=float(np.mean(np.abs(residuals))),
            maximum_absolute_error_ms=float(np.max(np.abs(residuals))),samples=samples,
            preparation_strata=[dict(mode=mode,samples=sum(r['preparation_mode']==mode for r in observations),
                rmse_ms=float(np.sqrt(np.mean([float(residuals[i])**2 for i,r in enumerate(observations) if r['preparation_mode']==mode]))))
                for mode in sorted({r['preparation_mode'] for r in observations})],
            measured_boundary='prefill inclusive implementation-span union complement + request clear/capture/control',
            exclusive_adjustment='opaque compound internal idle gaps are already in their inclusive local span and removed from the controller target',
            excludes='first-token argmax, KV handoff and all decoder host/service/wrapper phases already priced separately',
            scope='candidate-dependent controller approximation; counts are causal dispatch proxies, not isolated per-launch charges; whole-request heldout must quantify error'))
    model['identity']=digest(model)
    return model


class Training:
    def __init__(self,args,prefill,whole_source_id):
        self.args=args;self.prefill=prefill;self.prices=defaultdict(list);self.semantic=defaultdict(set)
        self.records={};self.cpu_proofs=defaultdict(dict);self.decode_prices=defaultdict(list)
        self.contexts={};self.receipts=[];self.conservation=[];self.fixed=[];self.fixed_observations=[]
        self.runtime_proofs=[];self.vendor_proofs=[];self.contract_producers=[]
        self.whole_source_id=whole_source_id
        self.frozen_runtimes=set();self.cpu_proof_receipts={}
        self.preparations=[];self.price_preparations=defaultdict(lambda:defaultdict(int))

    def audit(self):
        manifest=load(self.args.campaign)
        self.manifest_dependencies=manifest['dependencies']
        self.execution_receipts,self.verification_campaigns=audit_execution_receipts(manifest,getattr(self.args,'campaign_receipts',[]))
        require(manifest['schema']=='orbit.qwen-r19-sealed-campaign.v1','Unsealed or unknown campaign')
        require(manifest['whole_source']['source_id']==self.whole_source_id,'Campaign whole-source mismatch')
        verified(manifest['whole_source'])
        for dependency in manifest['dependencies']:verified(dependency)
        cohorts=[c for c in manifest['cohorts'] if c['purpose']=='training']
        require(cohorts and len({c['name'] for c in cohorts})==len(cohorts),'Missing/duplicate training cohorts')
        comparator=module(self.args.comparator,'r19_original_numeric_comparator')
        import numpy as np
        with np.load(self.args.reference_capture,allow_pickle=False) as data:
            reference={name:data[name].copy() for name in ('prompt','tokens','logits')}
        require(reference['logits'].shape[0]>=32,'Original numerical reference must cover 32 tokens')
        reference_top=np.argsort(-reference['logits'][:32],axis=1,kind='stable')[:,:5].tolist()
        self.prepare_tokens=reference['prompt'].tolist()
        for cohort in cohorts:
            command=cohort['command'];directory=Path(command[command.index('--output')+1])
            self.frozen_runtimes.add(str(Path(command[command.index('--frozen-runtime')+1]).resolve(strict=True)))
            require(int(command[command.index('--outputs')+1])==128,'Training requires all128 outputs')
            for flag,limit in [('--rss-limit-gib',20),('--swap-limit-gib',.5)]:
                require(flag not in command or float(command[command.index(flag)+1])==limit,'Original memory guard changed')
            contract=load(directory/'contract.json');proof=load(directory/'runtime-proof.json')
            require(contract['source']==self.prefill and contract['source_id']==self.prefill['source_id'] and
                    contract['outputs']==128 and contract['threads']==4,'Training execution ABI mismatch')
            require(contract['plan_id']==digest({k:v for k,v in contract['plan'].items() if k!='plan_id'}),
                    'Training plan identity mismatch')
            require(cohort['plan_id']==contract['plan_id'] and cohort['measured_requests']==3,'Campaign plan/request mismatch')
            require(contract['owned_runtime_proof']==proof and proof['execution_policy']=='source-dependency-ready' and
                    proof['source_ordering_identity']==self.whole_source_id and
                    proof['prefill_source_id']==self.prefill['source_id'],'Unbound owned runtime proof')
            require(proof['whole_source']['source_id']==self.whole_source_id,'Runtime whole-source binding mismatch')
            verified(proof['whole_source'])
            require(proof['async_runtime_proof_identity']==digest({k:v for k,v in proof.items() if k!='async_runtime_proof_identity'}),
                    'Runtime proof identity mismatch')
            for dependency in proof['code_files']+contract['implementation_files']:verified(dependency)
            preparation=audit_preparation(directory,command,proof['code_files'])
            self.preparations.append(dict(cohort=cohort['name'],**preparation))
            spec=load(verified(proof['ready_spec']))
            require(spec['source_id']==self.prefill['source_id'] and spec.get('source_ready'),
                    'Training did not execute the complete ready source DAG')
            self.runtime_proofs.append(proof);self.vendor_proofs.append(contract['implementation_files'])
            self.contract_producers.append(contract['producer'])
            complete=load(directory/'completion.json')
            require(complete['completed'] and complete['measured_requests']==3 and complete['warmups']==1,
                    'Training must complete one warmup and three measured requests')
            gate_path=Path(cohort.get('numeric_gate_file',directory/'gate.json'))
            require(gate_path.exists(),'Missing independently saved complete-model gate: '+str(gate_path))
            gate=load(gate_path)
            admission=cohort['admission_command']
            require('--allow-diagnostic' not in admission and '--allow-code-drift' not in admission,
                    'Training admission used a diagnostic/drift exception')
            require(Path(admission[admission.index('--output')+1])==gate_path and
                    Path(admission[admission.index('--execution')+1])==directory and
                    gate['auditor_sha256']==record(admission[1])['sha256'],
                    'Independent auditor identity or target mismatch')
            require(gate['schema']=='orbit.qwen-r19-independent-ready-audit.v1' and
                    gate['all_original_gates_pass'] is True and gate['code_files_current_match'] is True and
                    gate['runtime_proof_identity']==proof['async_runtime_proof_identity'] and
                    len(gate['requests'])==3 and gate['numerical_pairs'] and
                    gate['all_measured_token_streams_equal'] is True,
                    'Numerical/identity auditor rejected cohort '+cohort['name'])
            require(any(pair['sha256']==record(self.args.reference_capture)['sha256'] and
                        pair['original_mutual_top5']['status']=='OK' for pair in gate['numerical_pairs']),
                    'Auditor did not use the unchanged original numerical reference')
            with np.load(directory/'capture.npz',allow_pickle=False) as data:
                require(data['logits'].shape==(128,151936) and np.isfinite(data['logits']).all() and
                        data['tokens'].shape==(128,) and np.array_equal(data['prompt'],reference['prompt']),
                        'Incomplete, nonfinite or wrong-prompt full128 capture')
                require(np.array_equal(np.argmax(data['logits'],axis=1),data['tokens']),
                        'Captured tokens do not match captured logits')
                numeric=comparator.compute_topk_set_check(data['tokens'][:32].tolist(),
                    np.argsort(-data['logits'][:32],axis=1,kind='stable')[:,:5].tolist(),
                    reference['tokens'][:32].tolist(),reference_top,k=5).to_dict()
                require(numeric['status']=='OK','Unchanged original32-token comparator failed')
                captured_tokens=data['tokens'].tolist()
            rows=[json.loads(line) for line in (directory/'measurements.jsonl').read_text().splitlines()]
            measured=[r for r in rows if not r['warmup']]
            require(len(measured)==3 and {r['sample'] for r in measured}=={0,1,2},'Missing/duplicate measured samples')
            require(next(r for r in measured if r['sample']==0)['tokens']==captured_tokens,'Capture/sample0 token mismatch')
            require(all(r['tokens']==captured_tokens for r in measured),'Measured full128 token streams differ')
            context=load(directory/'decode-context.json');verified(context['template_identity'])
            require(context['variant']['column_rotation']==0 and context['variant']['parameters']['PROJ_RC_CACHE']==1,
                    'Rejected or inactive decoder choice')
            context_key=(context['template_identity']['sha256'],bool(spec.get('decode_rope_prefetch',False)))
            self.contexts[context_key]=context
            for row in rows:
                require(row['outputs']==128 and len(row['tokens'])==128 and len(row['decode_trace'])==127,
                        'Incomplete warmup or measured request')
                require(row['memory']['VmRSS_bytes']<=20*2**30 and row['memory']['VmSwap_bytes']<=.5*2**30,
                        'Original memory guard violated')
                records,stats=audit_prefill(row['prefill_source_profile'],self.prefill,contract['plan_id'],
                    self.whole_source_id,proof['async_runtime_proof_identity'])
                request=row['request_phases'];require(request['exclusive_sum_equals_wall'],'Unconserved request')
                checked=row['prefill_source_profile']['wall_ns']+sum(nanoseconds(request[k]) for k in
                    ('clear_context_ns','capture_copy_ns','request_control_ns','first_argmax_ns','handoff_ns'))
                for step,decode in enumerate(row['decode_trace']):
                    require(decode['step']==step,'Decode recurrence step mismatch')
                    phase=decode['phases'];service=audit_decode(phase,row['P']+step,context_key[1],step==126)
                    require(decode['outer_dispatch_ns']==phase['wall_ns']+decode['outside_dispatch_wrapper_ns'],
                            'Decode wrapper accounting mismatch')
                    checked+=decode['outer_dispatch_ns']+decode['argmax_ns']
                    if not row['warmup']:
                        for kind,value in [('decode',service),('decode-host-before',phase['host_before_ns']),
                            ('decode-host-after',phase['host_after_ns']+phase['host_control_ns']+decode['outside_dispatch_wrapper_ns']),
                            ('decode-argmax',decode['argmax_ns'])]:
                            self.decode_prices[kind,step,context_key].append(nanoseconds(value))
                            self.price_preparations[str((kind,step,context_key))][preparation['mode']]+=1
                require(checked==request['wall_ns'],'Complete request interval-union/decode conservation failed')
                if row['warmup']:continue
                for actual in records:
                    self.collect(actual,spec,proof)
                    self.price_preparations[actual['local_price_key']][preparation['mode']]+=1
                self.decode_prices['first-token-argmax',None,None].append(request['first_argmax_ns'])
                self.decode_prices['kv-handoff',None,None].append(request['handoff_ns'])
                for fixed_kind in ('first-token-argmax','kv-handoff'):
                    self.price_preparations[str((fixed_kind,None,None))][preparation['mode']]+=1
                fixed=stats['outside_implementation_control_ns']+sum(request[k] for k in
                    ('clear_context_ns','capture_copy_ns','request_control_ns'))
                self.fixed.append(fixed)
                self.fixed_observations.append(dict(cohort=cohort['name'],sample=row['sample'],fixed_ns=fixed,preparation_mode=preparation['mode'],**stats))
                self.conservation.append(dict(cohort=cohort['name'],sample=row['sample'],wall_ns=checked,
                    exact_union_plus_control_and_decode_equals_wall=True,**stats))
            self.receipts.append(dict(name=cohort['name'],contract=record(directory/'contract.json'),
                measurements=record(directory/'measurements.jsonl'),capture=record(directory/'capture.npz'),
                gate=record(gate_path),original32_gate=numeric,full128_capture=True,
                runtime_proof=record(directory/'runtime-proof.json'),preparation=preparation,
                execution_receipt_provenance=self.execution_receipts.get(cohort['name'])))
        require(len({digest(p['code_files']) for p in self.runtime_proofs})==1,'Training r19 runtime revisions differ')
        require(len({digest(p) for p in self.vendor_proofs})==1 and
                len({digest(p) for p in self.contract_producers})==1,'Frozen underlying runner/vendor revisions differ')
        require(len(self.contexts)==4 and {key[1] for key in self.contexts}=={False,True},
                'Need both coalescing variants with real prefetch false/true training')
        require({(context['variant']['parameters']['COALESCE'],key[1]) for key,context in self.contexts.items()}==
                {(0,False),(0,True),(1,False),(1,True)},'Four distinct coalescing/prefetch combinations required')
        require(len(self.frozen_runtimes)==1,'Training underlying frozen runtime directories differ')
        for dependency in manifest['dependencies']:verified(dependency)

    def collect(self,actual,spec,runtime):
        names=actual['source_operators'];kind=actual['implementation_kind'];params=actual['runtime_parameters']
        key=actual['local_price_key'];semantic=signature(names,kind,params)
        bindings={row['id']:row for row in self.prefill['bindings']}
        operators={row['id']:row for row in self.prefill['operators']}
        artifact=dict(source_bindings=[bindings[name] for name in sorted({operators[n]['binding'] for n in names})],
                      runtime_parameters=params)
        cpu=(kind in ('cpu-region','heterogeneous-gate-up') or
             (kind=='prefill-original' and params['device']=='cpu'))
        if cpu:artifact['cpu_lowerer_sha256']=next(row['sha256'] for row in runtime['code_files']
                                                 if Path(row['file']).name=='qwen-r19-regions.py')
        if kind=='cpu-region':artifact['region_id']=actual['events'][0]['phases'][0]['region_id']
        require(actual['artifact_key']==digest(artifact),'Runtime artifact key does not match its source/lowering')
        self.semantic[semantic].add(key);self.prices[key].append(actual['wall_ns']);self.records[key]=actual
        if kind=='cpu-region':
            phase=actual['events'][0]['phases'][0]
            require(phase['source_operators']==names and phase['backend']==params['backend'] and
                    phase['input_policy']==params['input_policy'] and phase['output_policy']==params['output_policy'] and
                    phase['npu_launches']==0 and phase['chip_transfer_bytes']==0,'CPU region runtime evidence mismatch')
            require(sum(phase[k] for k in ('validation_and_staging_ns','kernel_inclusive_ns','output_copy_and_validation_ns'))==
                    phase['wall_ns'] and phase['wall_ns']<=actual['wall_ns'],'CPU region phase overlap')
            require(all(w['same_physical_storage'] for w in phase['exported_view_witnesses']),
                    'Exported source view lost its real base')
            if params['backend']=='inductor-cpu':
                path=Path(spec['cpu_prepare_directory'])/digest(actual['implementation_id'])/'compile.json'
                compiled=load(path)
                require(compiled['region']['operators']==names and compiled['region']['source_id']==self.prefill['source_id'],
                        'Actual input specialization bound to a different cover')
                producer=next(r for r in runtime['code_files'] if Path(r['file']).name=='qwen-r19-regions.py')
                require(compiled['producer']['sha256']==producer['sha256'],'CPU compiler producer mismatch')
                verified(compiled['producer'])
                for code in compiled['generated_code']:verified(code)
                require(compiled['generated_kernel_audit']==phase['generated_kernel_audit'],
                        'Runtime CPU compiler audit differs from actual preparation')
                previous=self.cpu_proofs[key].get(compiled['specialization_id'])
                if previous is not None:
                    require([row['sha256'] for row in previous['proof']['generated_code']]==
                            [row['sha256'] for row in compiled['generated_code']],
                            'One CPU specialization emitted different actual code; separate artifact choices are required')
                self.cpu_proofs[key][compiled['specialization_id']]=dict(proof=compiled,record=record(path))
                self.cpu_proof_receipts[str(path)]=record(path)
        if kind=='heterogeneous-gate-up':
            parts=actual['events'];require(len(parts)==3,'Half compound must execute CPU, NPU and assembly callbacks')
            require(sum(p['device']=='npu' for p in parts)==1 and sum(p['device']=='cpu' for p in parts)==2,
                    'Half compound actual placement mismatch')
            device=next(p for p in parts if p['device']=='npu')
            require(len(device['phases'])==1,'Missing private NPU invocation evidence')
            audit_private_xrt(device['phases'][0])

    def match(self,names,kind,parameters):
        sig=signature(names,kind,parameters);keys=self.semantic.get(sig,set())
        require(len(keys)==1,'Missing or ambiguous fresh r19 backend price: '+sig)
        key=next(iter(keys));return key,self.records[key],self.prices[key]


def decode_prefetch_capacity(source,prompt_tokens):
    """Derive pure-host slab reservations from the actual typed RMS patch ABI.

    rope_patch remains one opaque source operation. Its written BF16 range is
    the concatenated per-layer _rope_slab; this proof does not introduce a new
    independently schedulable source task for that internal pure computation.
    """
    bindings=[row for row in source['bindings'] if row.get('symbol')=='FusedDecoder.dispatch']
    require(len(bindings)==1,'Ambiguous original decoder runtime binding')
    binding=bindings[0];native=record(binding['file'])
    require(native['sha256']==binding['sha256'],'Original native decoder changed')
    tree=ast.parse(Path(native['file']).read_text())
    methods=[node for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='FusedDecoder'
             for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='_rope_slab']
    require(len(methods)==1,'Missing original pure-host slab implementation')
    returns=[node for node in ast.walk(methods[0]) if isinstance(node,ast.Return)]
    expected=ast.parse('np.concatenate([self.gw.rope_w_layer(p,L,*self.bias[L]) for L in range(self.UNI_DEC)]).astype(self.bf16)',mode='eval').body
    require(len(returns)==1 and ast.dump(returns[0].value,include_attributes=False)==
            ast.dump(expected,include_attributes=False),'Native _rope_slab arithmetic/dtype changed')
    values={row['id']:row for row in source['values']};result={}
    for op in source['operators']:
        if op['kind']!='rope-rms-buffer-patch':continue
        require(len(op['outputs'])==1,'Unknown typed RMS patch output')
        output=values[op['outputs'][0]];attributes=op['attributes']
        words=attributes['written_words'];offset=attributes['written_offset_words']
        require(output['dtype']=='bf16' and type(words) is int and words>0 and
                type(offset) is int and 0<=offset and offset+words<=math.prod(output['shape']),
                'Typed RoPE patch range escapes actual BF16 RMS storage')
        step=attributes['position']-prompt_tokens
        require(step>=0 and step not in result,'Duplicate/unbound typed RoPE step')
        result[step]=dict(source_operator=op['id'],output_value=output['id'],dtype=output['dtype'],
            written_offset_words=offset,written_words=words,slab_bytes=words*2,
            original_runtime_binding=binding,native_runtime=native,
            native_method_ast_sha256=hashlib.sha256(ast.dump(methods[0],include_attributes=False).encode()).hexdigest())
    require(result,'Missing typed native RoPE patch geometry')
    for step,row in result.items():
        row['conservative_two_slab_bound_bytes']=2*max(row['slab_bytes'],result.get(step+1,row)['slab_bytes'])
        row['scope']='two possible host slabs conservatively reserved in every opaque decoder phase; final step retains the same bound although no unused prefetch runs; no BO/SRAM addition'
    return result


def ffn_capacity(template,registration,prefill,driver):
    """Rebuild actual warm host/BO ownership, independent of old totals.

    reload changes upload frequency, not allocation lifetime: _runner._bos
    retains a private BO for every argument until close. retain binds the three
    prepared shared weight BOs, so their host/device storage is counted once.
    release-after-call deletes the completed BO owner, but its host argument
    arrays and padded weights remain live. Torch BF16 input conversion and the
    returned logical-output clone are additional per-call managed temporaries.
    """
    artifact=registration['ffn_artifact'];path=Path(artifact['manifest'])
    require(record(path)['sha256']==artifact['sha256'],'FFN manifest changed')
    manifest=load(path);verified(driver)
    for row in manifest['source_files']:verified(row)
    require(record(manifest['artifact'])['sha256']==manifest['artifact_sha256'],
            'FFN compiled binary changed')
    buffers={row['index']:row for row in manifest['buffers']}
    require(set(buffers)==set(range(8)) and
            {i for i,row in buffers.items() if row['role']=='weight'}=={1,2,3} and
            all(row['bytes']==math.prod(row['shape'])*(4 if i==7 else 2) for i,row in buffers.items()),
            'Unknown FFN managed argument ABI')
    retention=registration['retention'];layer=registration['layer'];prefix=f'ffn.L{layer}.'
    require(retention in ('retain','reload','release-after-call'),'Unknown FFN BO ownership policy')
    resident={}
    for index,row in buffers.items():
        if index in (1,2,3):
            resident[f'{prefix}weight{index}.{retention}.prepared']=dict(memory='runtime-abi',bytes=row['bytes'])
        else:
            resident[f'{prefix}{retention}.host-arg{index}']=dict(memory='runtime-abi',bytes=row['bytes'])
        if retention!='release-after-call' and not(retention=='retain' and index in (1,2,3)):
            resident[f'{prefix}{retention}.device-arg{index}']=dict(memory='runtime-abi',bytes=row['bytes'])
    template=copy.deepcopy(template);variant=template['execution_variant']
    previous={key:value for key,value in variant['resident_allocations'].items() if key.startswith(prefix)}
    variant['resident_allocations']={key:value for key,value in variant['resident_allocations'].items() if not key.startswith(prefix)}
    variant['resident_allocations'].update(resident)
    all_bo_bytes=sum(row['bytes'] for row in buffers.values())
    op=next(op for op in prefill['operators'] if op['id']==registration['operators'][0])
    activation=next(value for value in prefill['values'] if value['id']==op['inputs'][0])
    active_rows=min(activation['shape'][0],manifest['row_tile'])
    transient=dict(input_bf16_rounding=active_rows*2048*2,output_numpy_clone=active_rows*2048*4,
        completed_private_bos=all_bo_bytes if retention=='release-after-call' else 0)
    variant['temporary_bytes']['runtime-abi']=sum(transient.values())
    template['ffn_managed_memory_audit']=dict(manifest=record(path),driver=driver,
        runtime_sources=manifest['source_files'],policy=retention,
        host_weight_bytes=sum(buffers[i]['bytes'] for i in (1,2,3)),
        cached_private_weight_bo_bytes=sum(buffers[i]['bytes'] for i in (1,2,3)) if retention=='reload' else 0,
        retained_host_and_bo_bytes=sum(row['bytes'] for row in resident.values()),
        actual_per_call_temporaries=transient,previous_resident_ledger_agreed=previous==resident,
        source_output_ownership='logical out tensor remains the existing source output; only its intermediate NumPy clone is charged above',
        shared_weight_alias='retain only: prepared host tensors and bound device BO are one physical allocation',
        unmodelled_scope='Python/XRT handle object and allocator overhead is checked by unchanged process RSS/swap guards')
    return template


def half_template(actual,source,defaults):
    """Bind actual CPU/NPU column fission to a conservatively reserved compound."""
    params=actual['runtime_parameters'];directory=Path(params['directory'])
    require(params['cpu_half'] in ('gate','up'),'Unknown actual CPU half placement')
    manifest_path=directory/'implementation.json';require(record(manifest_path)['sha256']==params['manifest_sha256'],
        'Half implementation manifest changed')
    manifest=load(manifest_path)
    require(manifest['shape']==dict(M=5,N=11008,K=2048),'Half artifact logical geometry mismatch')
    for name in ('source','producer','source_air','compiled_binary'):verified(manifest[name])
    replicas=manifest['resource']['task_instance_replicas'];require(replicas in (1,2),'Unsupported actual NPU replicas')
    for event in actual['events']:
        if event['device']=='npu':
            require(event['owner_thread']==event['phases'][0]['owner_thread'] and
                    event['phases'][0]['artifact_sha256']==manifest['compiled_binary']['sha256'],
                    'Half NPU callback executed a different artifact/owner')
    tiles=defaultdict(int);allocation_proofs=[];stacks=[]
    for proof in manifest['physical_evidence']['aie_sources']:
        text=verified(proof).read_text();allocation_proofs.append(proof)
        stacks.extend(map(int,re.findall(r'stack_size\s*=\s*(\d+)\s*:',text)))
        for line in text.splitlines():
            if 'aie.buffer(' not in line:continue
            match=re.search(r'%\w+ = aie.buffer\(%(\w+)\).*memref<([\dx]+)(bf16|f32|i32|i64|i16|i8)',line)
            require(match is not None,'Unparsed actual half SRAM allocation')
            tile,shape,dtype=match.groups()
            tiles[tile]+=math.prod(map(int,shape.rstrip('x').split('x')))*dict(bf16=2,f32=4,i32=4,i64=8,i16=2,i8=1)[dtype]
    cores=manifest['physical_evidence']['actual_core_coordinates']
    require(len(stacks)==len(cores) and len(set(stacks))==1,'Unknown/nonuniform actual core stacks')
    scratch={'npu.sram.'+tile:size+(stacks[0] if tile.startswith('tile_') else 0) for tile,size in tiles.items()}
    abi_bytes=sum(math.prod(buf['shape'])*(2 if buf['dtype']=='bf16' else 4) for buf in manifest['abi'])
    # CPU w.float(), real NPU contiguous weight copy, host padded packs and
    # separate private BOs coexist while the two branches run. Results remain
    # owned through concat. Source BF16 weights remain in their original root.
    allocation_bytes=dict(cpu_weight_f32=2048*11008*4,
        npu_contiguous_weight_bf16=2048*11008*2,host_packs=abi_bytes,private_device_bos=abi_bytes,
        cpu_input_rounding=5*2048*6,npu_input_rounding=5*2048*2,
        cpu_half_output=5*11008*4,npu_output_concat_and_clone=2*5*11008*4,
        assembly_output=5*22016*4)
    scratch['runtime-abi']=sum(allocation_bytes.values())
    opid=actual['source_operators'][0]
    op=next(op for op in source['operators'] if op['id']==opid)
    require(opid.endswith('.gate_up') and op['kind']=='matmul' and op['outputs']==[opid+'.value'],
            'Half source output ABI mismatch')
    inputs=[next(value for value in source['values'] if value['id']==name) for name in op['inputs']]
    require([(value['shape'],value['dtype']) for value in inputs]==[([5,2048],'f32'),([2048,22016],'bf16')],
            'Half logical source input ABI differs from the compiled partition contract')
    cpu_gate=params['cpu_half']=='gate';cpu_offset=0 if cpu_gate else 11008;npu_offset=11008 if cpu_gate else 0
    partitions=[dict(output_value=op['outputs'][0],replica=0,offsets=[0,cpu_offset],extents=[5,11008])]
    for index in range(replicas):
        partitions.append(dict(output_value=op['outputs'][0],replica=index+1,
            offsets=[0,npu_offset+index*(11008//replicas)],extents=[5,11008//replicas]))
    artifact=digest(dict(binary=manifest['compiled_binary'],source_operator=opid,cpu_half=params['cpu_half'],
        implementation=record(manifest_path)))
    return dict(id='r19-half-'+digest(dict(operator=opid,parameters=params)),
        artifact_identity=artifact,entrypoint='ReadyExecutor CPU/NPU halves plus CPU concat',device='npu',resource='npu.device',
        memory='runtime-abi',launch_count=1,
        execution_variant=dict(runtime_context_identity=digest(dict(artifact=artifact,parameters=params)),
            dataflow='actual CPU/NPU independent output-column fission; concatenate after both complete',
            residence_policy='original BF16 weight retained; private NPU pack/BO/context per-call through completion/clone',
            implementation_parameters={**manifest['tiling'],'column_offset':manifest['resource']['column_offset'],
                                       'cpu_half_is_gate':int(cpu_gate),'npu_replicas':replicas},
            replicas=replicas+1,work_distribution='registered-disjoint-work',work_partitions=partitions,
            resource_shape=manifest['resource']['compute_herd_shape'],
            resource_demands={'cpu.host':1,'cpu.npu-owner':1,'npu.device':1,'npu.command':1,
                             **{f'npu.core.{x}.{y}':1 for x,y in cores}},
            temporary_bytes=scratch,resident_allocations=copy.deepcopy(defaults)),
        physical_evidence=dict(manifest=record(manifest_path),aie=allocation_proofs,
            actual_core_coordinates=cores,stack_bytes_per_core=stacks[0],
            managed_temporary_breakdown=allocation_bytes,output_partitions=partitions,
            partition_placements=[dict(replica=part['replica'],device='cpu' if part['replica']==0 else 'npu',
                owner_resource='cpu.host' if part['replica']==0 else 'cpu.npu-owner',
                source_weight_columns=dict(offset=part['offsets'][1],extent=part['extents'][1]))
                for part in partitions],
            scheduling_scope='one checked compound span; common does not independently schedule its internal subtasks'))


def fit(args):
    source,prefill,base=load(args.source),load(args.prefill_source),load(args.base_catalogue)
    for document in (source,prefill):
        require(document['source_id']==digest({k:v for k,v in document.items() if k!='source_id'}),'Typed source identity mismatch')
    require(source['source_id']==base['source_id'] and source['operators'][:len(prefill['operators'])]==prefill['operators']
            and source['values'][:len(prefill['values'])]==prefill['values'],'Whole source lost its exact prefill prefix')
    training=Training(args,prefill,source['source_id']);training.audit()
    capacity_path=args.capacity_registration or Path(base['capacity_registration']['file'])
    capacity=load(capacity_path)
    for proof in capacity['evidence']:verified(proof)
    defaults=capacity['defaults']['execution_variant']['resident_allocations']
    ffn_driver=next(row for row in capacity['evidence'] if Path(row['file']).name=='multilaunch.py')
    aggregate_proof=dict(schema='orbit.qwen-r19-whole-ready-runtime-support.v1',producer=record(__file__),
        whole_source_id=source['source_id'],prefill_source_id=prefill['source_id'],
        source_manifest=dict(record(args.source),source_id=source['source_id']),
        prefill_source_manifest=dict(record(args.prefill_source),source_id=prefill['source_id']),
        code_files=training.runtime_proofs[0]['code_files'],
        vendor_files=training.vendor_proofs[0],underlying_runner_producer=training.contract_producers[0],
        execution_abi='exact source covers; CPU-main + one NPU-owner; SSA consumer-completion lifetime; original recurrent decoder context',
        training_runtime_proof_identities=sorted({p['async_runtime_proof_identity'] for p in training.runtime_proofs}),
        training_receipts=training.receipts,preparation_protocols=training.preparations,
        actual_verification_campaigns=training.verification_campaigns,
        preparation_scope='mixed outside-request preparation explicitly retained; heldout uniformly uses readonly-prefault preparation',
        sealed_campaign=record(args.campaign),sealed_dependencies=training.manifest_dependencies,
        actual_cpu_preparation_proofs=sorted(training.cpu_proof_receipts.values(),key=lambda row:row['file']),
        source_prefix_verified=True)
    verifier=verifier_runtime()
    aggregate_proof['fitter_verifier']=dict(producer=record(verifier.__file__),
        policy=copy.deepcopy(verifier.CACHE_POLICY),
        events_before_support_freeze=copy.deepcopy(verifier.CACHE_ADVICE_EVENTS))
    async_identity=digest(aggregate_proof)
    architecture=copy.deepcopy(base['architecture'])
    require(not any(r['id']=='cpu.npu-owner' for r in architecture['resources']),'Unexpected duplicate NPU owner resource')
    architecture['resources'].append(dict(id='cpu.npu-owner',kind='submission',device='cpu',capacity=1,domain='system'))
    result=copy.deepcopy(base)
    for field in ('execution_registration_identity','training','training_campaign','trace_conservation_receipts',
                  'producer','excluded_unpriced_implementations','fixed_application_invocation','choice_groups'):
        result.pop(field,None)
    result.update(architecture=architecture,implementations=[],choice_groups=[])
    runtime={};oldruntime=base['runtime_registration']['implementations']
    byop={op['id']:i for i,op in enumerate(source['operators'])}
    byvalue={v['id']:i for i,v in enumerate(source['values'])}
    producer={v:op['id'] for op in source['operators'] for v in op['outputs']}
    readers=defaultdict(set)
    for op in source['operators']:
        for value in op['inputs']:readers[value].add(op['id'])

    def admit(template,names,registration,samples,key,identifier=None):
        impl=copy.deepcopy(template);identifier=identifier or impl['id'];members=set(names)
        require(identifier not in runtime,'Duplicate admitted implementation ID')
        ins={v for name in names for v in source['operators'][byop[name]]['inputs'] if producer.get(v) not in members}
        outs={v for name in names for v in source['operators'][byop[name]]['outputs']
              if source['values'][byvalue[v]]['role']=='output' or readers[v]-members}
        summary=price_summary(samples,key)
        require(training.price_preparations[key],'Missing actual local-price preparation provenance')
        summary['preparation_modes']=sorted(training.price_preparations[key])
        summary['preparation_sample_counts']=dict(training.price_preparations[key])
        require(sum(summary['preparation_sample_counts'].values())==summary['samples'],
                'Local price/preparation sample counts disagree')
        impl.update(id=identifier,source_id=source['source_id'],source_operators=sorted(byop[n] for n in names),
            inputs=sorted(byvalue[v] for v in ins),outputs=sorted(byvalue[v] for v in outs),
            wall_ms=summary['median_ns']/1e6,executable=True,numeric_admitted=True,
            admission_status='r19-complete128-training-original32-comparator-admitted',training_price=summary,
            cost_boundary='inclusive-region-wall',transport_ownership='implementation-inclusive')
        kind=registration['kind']
        if kind=='heterogeneous-gate-up':features=dict(cpu_callbacks=2,npu_callbacks=1)
        elif kind.startswith('prefill-'):
            features=dict(cpu_callbacks=int(impl['device']=='cpu'),npu_callbacks=int(impl['device']=='npu'))
        else:features=dict(cpu_callbacks=0,npu_callbacks=0)
        impl['controller_features']=features
        for field in ('operators','input_values','output_values'):impl.pop(field,None)
        variant=impl['execution_variant']
        variant.update(execution_policy='source-dependency-ready',source_ordering_identity=source['source_id'],
                       async_runtime_proof_identity=async_identity)
        if impl['device']=='npu' and registration['kind']!='heterogeneous-gate-up':
            variant['resource_demands'].pop('cpu.host',None)
            variant['resource_demands']['cpu.npu-owner']=1
        merge_residents(variant,defaults)
        result['implementations'].append(impl)
        runtime[identifier]=dict(registration,source_id=source['source_id'],operators=names,
            device=impl['device'],artifact_identity=impl['artifact_identity'],
            execution_policy='source-dependency-ready',async_runtime_proof_identity=async_identity)
        return identifier

    # Reprice every original prefill template. No old median or old numerical
    # admission can make an unobserved r19 implementation executable.
    for template in base['implementations']:
        reg=oldruntime[template['id']];kind=reg['kind'];names=reg['operators']
        if kind not in ('prefill-original','prefill-gemm','prefill-fused-ffn'):continue
        params=dict(reg)
        if kind=='prefill-original':params['device']=template['device']
        key,actual,values=training.match(names,kind,params)
        if kind=='prefill-fused-ffn':template=ffn_capacity(template,reg,prefill,ffn_driver)
        admit(template,names,dict(reg,r19_runtime_parameters=actual['runtime_parameters']),values,key)

    pending=load(args.cpu_registration)
    region_backend_path=next(Path(row['file']) for row in aggregate_proof['code_files']
                             if Path(row['file']).name=='qwen-r19-regions.py')
    region_backend=module(region_backend_path,'r19_fit_actual_cpu_backend')
    program=region_backend.Program(prefill)
    for candidate in pending['implementations']:
        names=candidate['operators'];p=candidate['runtime_config']
        params={key:p[key] for key in ('backend','input_policy','output_policy')}
        key,actual,values=training.match(names,'cpu-region',params)
        executable=region_backend.ExecutableRegion(program,names,**params)
        require(params['input_policy']!='stage-host-copy' or
                all(alias['base'] not in executable.proof['input_values'] for alias in executable.proof['exported_views']),
                'Staged external base changes an exported source view alias; a typed physical bridge is required')
        compiled=None;specialization_records=[]
        if params['backend']=='inductor-cpu':
            proofs=training.cpu_proofs[key]
            require(len(proofs)==1,'Multiple actual input specializations need separate executable choices')
            entry=next(iter(proofs.values()));compiled=entry['proof'];specialization_records=[entry['record']]
            require(compiled['template_id']==executable.template_id,'Actual CPU compiler structure mismatch')
        template=region_backend.registration(executable,compiled)
        source_values={row['id']:row for row in prefill['values']}
        output_roots=set()
        for name in executable.proof['output_values']:
            seen=set()
            while name in producer:
                require(name not in seen,'Cyclic source view alias');seen.add(name)
                origin=source['operators'][byop[producer[name]]]
                if origin['kind']!='split-view':break
                name=origin['inputs'][0]
            output_roots.add(name)
        output_storage_bound=sum(math.prod(source_values[name]['shape'])*
            dict(f32=4,bf16=2,i64=8,i32=4)[source_values[name]['dtype']] for name in output_roots)
        template['memory_accounting']=dict(
            scheme='conservative actual compiler heap/eager workspace plus independently reserved source ABI physical roots',
            temporary_reservation_bytes=template['execution_variant']['temporary_bytes']['runtime-abi'],
            actual_generated_heap_bytes=compiled['generated_kernel_audit']['managed_heap_allocation_bytes'] if compiled else None,
            common_source_output_root_reservation_bytes=output_storage_bound,
            duplicate_output_reservation_upper_bound_bytes=min(output_storage_bound,
                template['execution_variant']['temporary_bytes']['runtime-abi']),
            duplicate_scope='possible double charge bounded by common output-root reservations and region temporary bytes; not a measured physical alias amount; exported split views trace to their full source base, not just view element count',
            subtraction_applied_bytes=0,
            reason='no typed generated-return allocation to source-physical-root witness; matching shape/dtype alone is insufficient',
            eager_scope='workspace bound includes all produced source tensors and helper intermediates; additional internal lifetime overlap is conservative',
            ultimate_validation='unchanged complete-program RSS/swap guards on actual selected replay')
        # The source proof/actual preparation stays prefill-bound as backend
        # evidence; the canonical implementation ABI below uses whole indices.
        template['actual_input_specializations']=specialization_records
        identifier='r19-cpu-region-'+digest(dict(source=source['source_id'],operators=names,
            params=params,artifact=template['artifact_identity']))
        admit(template,names,dict(kind='prefill-cpu-region',**params,region_id=executable.proof['region_id'],
            template_id=executable.template_id,compile_proof=specialization_records[0] if specialization_records else None,
            actual_input_specializations=specialization_records),values,key,identifier)

    half_domain=defaultdict(set)
    for key,actual in sorted(training.records.items()):
        if actual['implementation_kind']!='heterogeneous-gate-up':continue
        template=half_template(actual,source,defaults)
        half_domain[actual['source_operators'][0]].add((actual['runtime_parameters']['cpu_half'],
            template['execution_variant']['replicas']-1))
        admit(template,actual['source_operators'],dict(kind='heterogeneous-gate-up',
            **actual['runtime_parameters'],output_partitions=template['execution_variant']['work_partitions'],
            scheduling_scope='actual compound span; internal CPU/NPU tasks not independently scheduled by common'),
            training.prices[key],key)

    expected_half_ops={op['id'] for op in prefill['operators'] if op['id'].endswith('.gate_up')}
    require(set(half_domain)==expected_half_ops and
            all(choices=={('gate',1),('up',1),('gate',2),('up',2)} for choices in half_domain.values()),
            'Incomplete actual full-layer CPU/NPU half-fission domain')

    decode_prefetch_memory=decode_prefetch_capacity(source,len(training.prepare_tokens))
    decode_choices=[]
    fixed_kinds={'first-token-argmax','kv-handoff'}
    for template in base['implementations']:
        reg=oldruntime[template['id']]
        if reg['kind']=='host' and reg['host_kind'] in fixed_kinds:
            key=(reg['host_kind'],None,None)
            admit(template,reg['operators'],reg,training.decode_prices[key],str(key))
    for context_key,context in sorted(training.contexts.items()):
        directory=context['directory'];prefetch=context_key[1];chosen=[]
        for template in base['implementations']:
            reg=oldruntime[template['id']]
            if reg['kind'] not in ('decode-context','host') or reg.get('decode_variant_dir')!=directory:continue
            if reg['kind']=='host' and reg['host_kind'] in fixed_kinds:continue
            kind='decode' if reg['kind']=='decode-context' else reg['host_kind']
            key=(kind,reg['step'],context_key)
            variant=copy.deepcopy(template)
            variant['execution_variant']['runtime_context_identity']=digest(dict(template=context_key[0],
                prefetch=prefetch,original=template['execution_variant']['runtime_context_identity']))
            variant['execution_variant']['dataflow']+='; pure-host-RoPE-prefetch='+str(prefetch).lower()
            if prefetch:
                # Two next-position slabs can briefly coexist with native host
                # helper intermediates. No new device BO or SRAM is claimed.
                require(reg['step'] in decode_prefetch_memory,'Missing source-bound decoder prefetch geometry')
                bound=decode_prefetch_memory[reg['step']]
                variant['execution_variant']['temporary_bytes']['runtime-abi']=variant['execution_variant']['temporary_bytes'].get('runtime-abi',0)+bound['conservative_two_slab_bound_bytes']
                variant['decode_prefetch_memory_accounting']=bound
            identifier='r19.'+template['id']+'.prefetch-'+str(int(prefetch))
            chosen.append(admit(variant,reg['operators'],dict(reg,decode_rope_prefetch=prefetch,
                context_identity=digest(context_key),prefetch_source_scope='existing rope_patch compound; no synthetic pure-RoPE source task'),
                training.decode_prices[key],str(key),identifier))
        require(len(chosen)==508,'Incomplete atomic full-generation decoder context')
        decode_choices.append(dict(id='r19-context-'+digest(context_key),implementations=chosen))
    result['choice_groups']=[dict(id='decode.global-compiled-dataflow-context',choices=decode_choices)]
    result['runtime_registration']=dict(base['runtime_registration'],schema='orbit.qwen-r19-whole-runtime-registration.v1',
        implementations=runtime,async_runtime_proof_identity=async_identity,
        async_runtime_support_proof=aggregate_proof,r19_code_files=aggregate_proof['code_files'])
    domain=module(Path(__file__).with_name('build-r19-cover-domain.py'),'r19_fit_cover_domain')
    result=domain.build(source,result,{},['decode.global-compiled-dataflow-context'])
    result.update(producer=record(__file__),training=training.receipts,training_campaign=record(args.campaign),
        trace_conservation_receipts=training.conservation,missing_local_prices=0,
        parent_static_catalogue=record(args.base_catalogue),capacity_registration=record(capacity_path),
        cpu_pending_registration=record(args.cpu_registration),old_prices_reused=False,
        numerical_reference=record(args.reference_capture),original_comparator=record(args.comparator),
        fixed_application_invocation=dict(wall_ms=0,
            pooled_median_diagnostic_ms=statistics.median(training.fixed)/1e6,
            samples_ns=training.fixed,boundary='outside inclusive implementation spans plus request clear/capture/control',
            charged_by='invocation_control_model; fixed wall remains zero to prevent double counting'),
        invocation_control_model=fit_controller(training.fixed_observations,source['source_id'],record(args.campaign)),
        fixed_control_feature_observations=training.fixed_observations,
        search_scope=dict(execution='source-dependency-ready; independent CPU-main and one NPU-owner slots, not physical core counts',
            cpu_fusion_fission='dynamic checked convex covers with actual eager/Inductor and explicit host staging/copy',
            heterogeneous_fission='actual CPU/NPU output partitions priced as opaque checked compound spans',
            decode='four real compiled-coalescing/pure-host-prefetch contexts; existing typed rope_patch remains compound',
            inherited_dimensions='only actual registered tiling/core placement/replica/dataflow/residence implementations'))
    support_path=args.support_proof_output or args.output.with_suffix('.runtime-support.json')
    support=dict(aggregate_proof,identity=async_identity)
    frozen=Path(next(iter(training.frozen_runtimes)));seal=frozen/'freeze.json'
    for proof in load(seal)['files']:verified(proof)
    with support_path.open('x') as stream:json.dump(support,stream,indent=2,allow_nan=False);stream.write('\n')
    runnerfile=next(row for row in aggregate_proof['code_files'] if Path(row['file']).name=='run-qwen-r19-ready.py')
    support_record=dict(record(support_path),identity=async_identity)
    final_preparation_mode=next(mode for mode in reversed(PREPARATION_MODES)
        if any(row['mode']==mode for row in training.preparations))
    prepared_entries={digest(row['entrypoint']):row['entrypoint'] for row in training.preparations if row['mode']==final_preparation_mode}
    require(len(prepared_entries)==1,'Exactly one sealed preparation entrypoint required for uniform heldout replay')
    preparation_entrypoint=next(iter(prepared_entries.values()))
    ready_runtime=dict(code_files=aggregate_proof['code_files'],runnerfile=runnerfile,
        preparation_entrypoint=preparation_entrypoint,preparation_protocol=final_preparation_mode,
        frozen_r18_runtime=dict(directory=str(frozen),seal=record(seal)),
        prepare_tokens=training.prepare_tokens,
        source_manifest=dict(record(args.source),source_id=source['source_id']),
        prefill_source_manifest=dict(record(args.prefill_source),source_id=prefill['source_id']),
        async_runtime_support_proof=support_record,
        owner_resources=dict(cpu_main='cpu.host',npu_worker='cpu.npu-owner',
            interpretation='execution owner slots, not physical CPU core counts'))
    result['ready_runtime']=ready_runtime
    result['runtime_registration']['ready_runtime']=ready_runtime
    result['runtime_registration']['async_runtime_support_proof']=support_record
    result['fitter_verification']=dict(producer=record(verifier.__file__),
        verification_cache_policy=copy.deepcopy(verifier.CACHE_POLICY),
        file_cache_advice=copy.deepcopy(verifier.CACHE_ADVICE_EVENTS))
    result['execution_registration_identity']=digest(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','prefill-source','base-catalogue','cpu-registration','campaign',
                 'reference-capture','comparator','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--campaign-receipts',type=Path,action='append',default=[],
                        help='Actual independent driver output directory; repeat for additive campaigns')
    parser.add_argument('--capacity-registration',type=Path)
    parser.add_argument('--support-proof-output',type=Path)
    args=parser.parse_args();require(not args.output.exists(),'Output catalogue already exists');result=fit(args)
    with args.output.open('x') as stream:
        json.dump(result,stream,separators=(',',':'),allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(args.output),implementations=len(result['implementations']),
        groups=len(result['choice_groups']),registration=result['execution_registration_identity'],old_prices_reused=False)))


if __name__=='__main__':main()
