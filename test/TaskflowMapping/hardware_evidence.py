#!/usr/bin/env python3
"""Read-only hardware evidence audit. Never opens an XRT device or dispatches.

Trace timestamps are native device cycles unless an explicit visual conversion
is recorded. This checker re-decodes saved raw words with the installed parser;
a host API timeline can therefore never satisfy the hardware-trace check.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import statistics
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event_list(obj):
    return obj['traceEvents'] if isinstance(obj,dict) else obj


def trace_audit(manifest_path):
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    errors=[];limitations=[]
    def need(value,message):
        if not value:errors.append(message)
    def path(field):
        value=Path(manifest[field]);return value if value.is_absolute() else manifest_path.parent/value
    need(manifest.get('schema')=='orbit.hardware.trace-evidence.v1','unsupported evidence schema')
    raw_path,mlir_path,native_path=path('raw_path'),path('physical_mlir_path'),path('decoded_native_path')
    raw=np.load(raw_path,allow_pickle=False)
    need(raw.dtype==np.uint32 and raw.ndim==1,'raw trace must be flat uint32 device words')
    need(raw.size>0 and np.count_nonzero(raw)>0,'empty/all-zero raw trace')
    native=event_list(json.loads(native_path.read_text()))
    need(manifest.get('native_timestamp_unit')=='device_cycles','native timestamp unit must explicitly be device cycles')
    timed=[e for e in native if e.get('ph') in ('B','E','X') and 'ts' in e]
    need(bool(timed),'metadata-only JSON is not hardware event evidence')
    need(all(math.isfinite(e['ts']) and e['ts']>=0 for e in timed),'invalid device timestamps')
    counts=collections.Counter(e.get('name','') for e in timed)
    patterns=manifest.get('expected_event_patterns',[])
    need(bool(patterns),'expected event families not specified')
    for pattern in patterns:need(any(re.search(pattern,n) for n in counts),'missing expected event pattern: '+pattern)
    metadata=[e for e in native if e.get('ph')=='M']
    configured_names=sorted({e.get('args',{}).get('name','') for e in metadata if e.get('name')=='thread_name'})
    absent_configured=[name for name in configured_names if name not in counts]
    if any('INSTR_EVENT' in name for name in absent_configured):
        limitations.append('Configured scope-marker event slots have no observed INSTR_EVENT records; no marked compute interval is established.')
    for tile in manifest.get('expected_tile_patterns',[]):
        need(any(re.search(tile,json.dumps(e)) for e in metadata),'missing expected tile: '+tile)
    # Re-decoding is CPU-only: parsing installed AIE IR does not open XRT.
    from aie.utils.trace.parse import parse_trace,parse_trace_slices
    import aie.utils.trace.parse as parser_module
    decoder=manifest.get('decoder',{})
    kwargs=decoder.get('kwargs',{})
    if decoder.get('api','parse_trace')=='parse_trace_slices':
        decoded=[]
        for _,events in parse_trace_slices(raw,mlir_path.read_text(),**kwargs):decoded.extend(events)
    else:
        decoded=parse_trace(raw,mlir_path.read_text(),**kwargs)
    need(decoded==native,'saved native JSON differs from independent raw-word re-decode')
    actual_decoder_hash=digest(parser_module.__file__)
    if decoder.get('sha256'):need(decoder['sha256']==actual_decoder_hash,'decoder implementation hash changed')
    # Native records retain independent tile clocks. No host/device alignment
    # is inferred, and overlapping stall counters are not added into latency.
    bytrack=collections.defaultdict(list)
    for e in timed:bytrack[e.get('pid'),e.get('tid')].append(e)
    unmatched=0;track_stats=[]
    for (pid,tid),rows in bytrack.items():
        active=[];unmatched_end=0
        need(all(a['ts']<=b['ts'] for a,b in zip(rows,rows[1:])),'nonmonotonic per-track cycle timestamp')
        for row in rows:
            if row['ph']=='B':active.append(row)
            elif row['ph']=='E':
                if active:active.pop()
                else:unmatched_end+=1
        unmatched+=len(active)+unmatched_end
        track_stats.append(dict(pid=pid,tid=tid,event_records=len(rows),first_cycle=min(e['ts'] for e in rows),last_cycle=max(e['ts'] for e in rows),unmatched_begin=len(active),unmatched_end=unmatched_end))
    nonzero=np.flatnonzero(raw)
    last_nonzero=int(nonzero[-1]) if len(nonzero) else -1
    tail_zero_bytes=(len(raw)-last_nonzero-1)*4
    if 'capacity_bytes' in manifest:need(manifest['capacity_bytes']==raw.nbytes,'declared trace capacity differs from raw buffer')
    if 'nonzero_words' in manifest:need(manifest['nonzero_words']==len(nonzero),'declared nonzero count differs from raw buffer')
    if 'event_count' in manifest:need(manifest['event_count']==len(native),'declared decoded event count mismatch')
    if unmatched:limitations.append('unmatched begin/end events; complete event intervals are not established')
    if tail_zero_bytes==0:limitations.append('trace occupies end of allocation; saturation/truncation possible')
    limitations.append('tail zeros alone cannot exclude overflow, split-channel truncation or missing coverage')
    completeness=manifest.get('completeness',{})
    status=completeness.get('status','partial')
    need(status in ('partial','complete'),'unknown completeness status')
    if status=='complete':
        need(completeness.get('start_marker_observed') is True and completeness.get('end_marker_observed') is True,'complete claim requires observed scope markers')
        need(completeness.get('overflow_checked') is True and completeness.get('overflow_detected') is False,'complete claim requires explicit overflow check')
        need(bool(completeness.get('scope')) and bool(completeness.get('evidence')),'complete claim requires coverage scope and concrete evidence')
        need(unmatched==0,'complete claim with unmatched intervals')
    else:limitations.append('capture is partial; no full-graph duration or complete stall accounting inferred')
    visual_path=manifest.get('visual_path')
    if visual_path:
        visual=event_list(json.loads(path('visual_path').read_text()))
        conversion=manifest.get('visual_conversion',{})
        factor=conversion.get('microseconds_per_cycle')
        need(isinstance(factor,(int,float)) and factor>0 and bool(conversion.get('basis')),'visual conversion lacks explicit frequency basis')
        if isinstance(factor,(int,float)):
            expected=[]
            for e in native:
                e=dict(e)
                for key in ('ts','dur'):
                    if key in e:e[key]*=factor
                expected.append(e)
            need(visual==expected,'visual trace is not the declared cycle scaling of raw-derived events')
        limitations.append('visual frequency conversion is not host/device clock calibration')
    elif manifest.get('viewer'):
        viewer=manifest['viewer'];vp=Path(viewer['path']);vp=vp if vp.is_absolute() else manifest_path.parent/vp
        visual=json.loads(vp.read_text())
        need(event_list(visual)==native,'cycle-axis viewer events differ from raw-derived native events')
        need('cycle' in viewer.get('axis','').lower() and visual.get('metadata',{}).get('host_alignment')=='none','unscaled viewer must explicitly label cycle surrogate and no host alignment')
        limitations.append('Viewer uses an unscaled cycle axis: displayed microsecond units are a visual surrogate, not physical time.')
    return dict(passed=not errors,errors=errors,limitations=limitations,manifest=str(manifest_path),raw_bytes=int(raw.nbytes),nonzero_words=int(np.count_nonzero(raw)),trailing_zero_bytes=tail_zero_bytes,timed_event_records=len(timed),event_counts=dict(counts),configured_event_names=configured_names,configured_but_unobserved=absent_configured,tracks=track_stats,completeness=status,decoder_sha256=actual_decoder_hash,artifacts={str(p):digest(p) for p in (raw_path,mlir_path,native_path)})


def chain_audit(result_path):
    """Verify saved arrays and actual start observations, not declared counts."""
    result_path=Path(result_path).resolve();root=result_path.parent
    result=json.loads(result_path.read_text());errors=[];limitations=[]
    def need(value,message):
        if not value:errors.append(message)
    need(result.get('schema')=='orbit.chain.experiment.v1','unsupported chain evidence schema')
    need(result.get('status')=='passed','chain did not execute and validate successfully')
    if result.get('status')!='passed':
        return dict(passed=False,errors=errors,source=str(result_path),execution_status=result.get('status'),failure=result.get('error'))
    source=np.load(root/'inputs-f32.npz',allow_pickle=False)
    # Whole-matrix float64 oracle, independent from public slices and runner.
    reference=(source['X'].astype('f8')@source['W1'].astype('f8'))@source['W2'].astype('f8')
    need(np.array_equal(reference,np.load(root/'reference-f64.npy',allow_pickle=False)),'saved reference is not independent source equation')
    actual=np.load(root/'inputs-bf16.npz',allow_pickle=False)
    from ml_dtypes import bfloat16
    for name in ('X','W1','W2'):
        need(np.array_equal(source[name].astype(bfloat16).astype('f4'),actual[name]),'BF16 input rounding differs from documented contract')
    for filename,expected in result['input_identity'].items():need(digest(root/filename)==expected,'input artifact hash mismatch: '+filename)
    for original,expected in result['implementation_sha256'].items():
        copied=root/Path(original).name
        need(copied.exists() and digest(copied)==expected,'saved implementation hash mismatch: '+str(copied))
    same_frontend=[]
    for name,index in (('first',0),('second',1)):
        same=(root/name/'op-0-ttshared.mlir').read_bytes()==(root/'chain'/f'op-{index}-ttshared.mlir').read_bytes()
        same_frontend.append(same)
        need(same,'separate/chain frontend kernel IR differs for '+name)
    medians={};output_checks={};observations={}
    for name,expected_submits in (('independent',2),('chain',1)):
        variant=result['variants'][name];samples=variant['samples'];need(len(samples)>=3,'less than three complete hot samples')
        times=[s['wall_ms'] for s in samples]
        need(all(math.isfinite(t) and t>0 for t in times),'invalid complete graph samples')
        medians[name]=statistics.median(times)
        need(math.isclose(medians[name],variant['median_ms'],rel_tol=1e-12),'forged median')
        first_objects=None;checks=[];counts=[]
        for sample in samples:
            output=np.load(sample['output_path'],allow_pickle=False)
            good_shape=output.shape==reference.shape
            finite=bool(np.isfinite(output).all())
            need(good_shape and finite,'wrong/nonfinite full output')
            if good_shape and finite:
                error=output.astype('f8')-reference
                rms=float(np.sqrt(np.mean(error*error))/max(np.sqrt(np.mean(reference*reference)),1e-30))
                peak=float(np.max(np.abs(error))/max(np.max(np.abs(reference)),1e-30))
                good=bool(np.allclose(output,reference,atol=.015,rtol=.06) and rms<=.03 and peak<=.08)
                need(good,'independent complete numerical check failed')
                checks.append(dict(iteration=sample['iteration'],passed=good,max_abs=float(np.max(abs(error))),normalized_rms=rms,peak_scaled=peak))
            events=sample['events'];starts=[e for e in events if e['kind']=='actual_xrt_start'];waits=[e for e in events if e['kind']=='actual_xrt_wait']
            need(len(starts)==len(waits)==expected_submits,'actual submit/wait observations do not match packed boundary')
            need(sample['actual_submissions']==len(starts) and sample['actual_waits']==len(waits),'declared submit counts disagree with observations')
            need([e['kind'] for e in events]==['actual_xrt_start','actual_xrt_wait']*expected_submits,'unexpected XRT start/wait ordering')
            need(all(e['start_ns']<=e['end_ns'] for e in events),'invalid host API observation intervals')
            need(all(a['end_ns']<=b['start_ns'] for a,b in zip(events,events[1:])),'overlapping supposedly synchronous host API calls')
            need(sum((e['end_ns']-e['start_ns'])/1e6 for e in events)<=sample['wall_ms']+1e-6,'API observations exceed complete graph wall sample')
            expected_readback=[[2],[2]] if name=='independent' else [[4]]
            need(sample['readback_indices']==expected_readback,'unexpected output readback declaration')
            objects=sample['runtime_objects']
            need(len(objects)==expected_submits,'incorrect reused execution object count')
            if first_objects is None:first_objects=objects
            need(objects==first_objects,'ELF/context/kernel/BO set changed between hot samples')
            need(all(o['bo_sets'] for o in objects),'missing BO identity evidence')
            counts.append(dict(iteration=sample['iteration'],actual_starts=len(starts),actual_waits=len(waits)))
        output_checks[name]=checks;observations[name]=counts
    pairwise=[]
    for left,right in zip(result['variants']['independent']['samples'],result['variants']['chain']['samples']):
        a=np.load(left['output_path'],allow_pickle=False);b=np.load(right['output_path'],allow_pickle=False)
        pairwise.append(float(np.max(abs(a-b))))
    need(pairwise==result['pairwise_output_max_abs'],'forged pairwise output difference')
    target=json.loads((root/'target-plan.json').read_text())
    need(target['intermediate']['dtype']=='bf16','hidden intermediate precision change')
    need(target['variants']['independent']['host_intermediate'] is True and target['variants']['chain']['host_intermediate'] is False,'wrong intermediate boundary')
    limitations.extend(['Start/wait observations are host API instrumentation, not AIE event timing.',
        'Readback indices and static/scratch flags are code-observed configuration, not DMA hardware counters.',
        'Packing changes context selection, intermediate host readback/staging and load_pdi sequencing; savings are not isolated launch latency.',
        'BF16 intermediate is shared by both variants; this is a two-GEMM graph, not standard FFN with CPU SwiGLU.',
        'Frontend kernels match; ELF physical scheduling/switch overhead can differ, and intermediate storage is system DRAM.'])
    return dict(passed=not errors,errors=errors,limitations=limitations,source=str(result_path),case=result['case'],split=result['split'],median_ms=medians,pairwise_max_abs=pairwise,observed_submissions=observations,independent_numerical_checks=output_checks,identical_frontend_ir=same_frontend)


def conditional_holdout(results):
    """A clearly bounded, train-only fixed-saving model, not compute-only cost."""
    bycase={r['case']:r for r in results if r.get('passed')}
    if not {'train','holdout'}<=bycase.keys():return dict(status='unscored',reason='requires successful independent train and holdout evidence')
    train,holdout=bycase['train'],bycase['holdout']
    saving=train['median_ms']['independent']-train['median_ms']['chain']
    predicted=holdout['median_ms']['independent']-saving
    actual=holdout['median_ms']['chain']
    return dict(status='scored',model='train fixed saving conditional on measured held-out separate graph latency',calibration_case='train',validation_case='holdout',calibrated_saving_ms=saving,predicted_packed_ms=predicted,observed_packed_ms=actual,signed_error_ms=predicted-actual,absolute_relative_error=abs(predicted-actual)/actual,limitations='This is a conditional host-system model. Held-out packed samples do not fit its constant. It does not identify compute, DMA, or pure launch costs and does not validate the prior static compute-only model.')


def graph_trace_audit(result_path):
    result_path=Path(result_path).resolve();root=result_path.parent;r=json.loads(result_path.read_text());errors=[]
    def need(value,message):
        if not value:errors.append(message)
    need(r.get('schema')=='orbit.trace.graph-experiment.v1' and r.get('status')=='passed','trace graph failed execution/numerical checks')
    if r.get('status')!='passed':return dict(passed=False,errors=errors,source=str(result_path),failure=r.get('error'))
    data=Path(__file__).resolve().parents[2]/'build/search-npu-v2/data'/r['case']
    need(digest(data/'inputs-0.npz')==r['input_sha256'] and digest(data/'reference-0.npz')==r['reference_sha256'],'graph input/reference identity changed')
    inputs={k:v.astype('f8') for k,v in np.load(data/'inputs-0.npz',allow_pickle=False).items()}
    if set(inputs)=={'A','B'}:
        ref={'C':inputs['A']@inputs['B']}
    elif set(inputs)=={'X','W_gate','W_up','W_down'}:
        gate=inputs['X']@inputs['W_gate'];up=inputs['X']@inputs['W_up'];hidden=gate*np.exp(-np.logaddexp(0,-gate))*up
        ref={'Y':hidden@inputs['W_down']}
    else:raise ValueError('independent graph oracle does not support these source inputs')
    saved_ref=dict(np.load(data/'reference-0.npz',allow_pickle=False))
    need(set(saved_ref)==set(ref) and all(np.allclose(saved_ref[k],v,atol=1e-12,rtol=1e-12) for k,v in ref.items()),'saved graph reference differs from independent whole-matrix equation')
    plan=json.loads((root/'plan.json').read_text());need(plan['candidate_id']==r['candidate_id'],'graph candidate identity mismatch')
    matmuls=[(t['id'],i) for t in plan['tasks'] for i,o in enumerate(t['operations']) if o['kind']=='matmul']
    swiglus=[(t['id'],i) for t in plan['tasks'] for i,o in enumerate(t['operations']) if o['kind']=='swiglu']
    checks={};medians={};max_pair=0.0;api_counts={};instrumented=True
    for mode in ('off','on'):
        v=r['variants'][mode];samples=v['full_graph_ms'];need(len(samples)>=3 and all(math.isfinite(x) and x>0 for x in samples),'invalid graph hot samples')
        medians[mode]=statistics.median(samples);need(math.isclose(medians[mode],v['median_ms'],rel_tol=1e-12),'graph median mismatch')
        files=sorted((root/mode).glob('outputs-*.npz'));need(len(files)==len(samples)+2,'missing complete graph output samples')
        checks[mode]=[]
        for file in files:
            out=dict(np.load(file,allow_pickle=False));need(set(out)==set(ref),'missing graph output')
            for name,expected in ref.items():
                actual=out[name];need(actual.shape==expected.shape and np.isfinite(actual).all(),'wrong/nonfinite graph output')
                error=actual.astype('f8')-expected
                rms=float(np.sqrt(np.mean(error*error))/max(np.sqrt(np.mean(expected*expected)),1e-30));peak=float(np.max(abs(error))/max(np.max(abs(expected)),1e-30))
                passed=bool(np.allclose(actual,expected,atol=.015,rtol=.06) and rms<=.03 and peak<=.08)
                need(passed,'independent traced/untraced whole-output check failed')
                checks[mode].append(dict(file=file.name,output=name,passed=passed,max_abs=float(np.max(abs(error))),normalized_rms=rms,peak_scaled=peak))
                if mode=='on':max_pair=max(max_pair,float(np.max(abs(actual-np.load(root/'off'/file.name,allow_pickle=False)[name]))))
        host=json.loads((root/mode/'host-events.json').read_text());need(len(host)==len(samples)+2,'missing host graph observations')
        api_counts[mode]=[]
        for i,h in enumerate(host):
            sections=h['sections'];cpu=[(s['task'],s['operation']) for s in sections if s['kind']=='swiglu' and s['placement']=='cpu']
            need(cpu==swiglus,'SwiGLU missing or moved from required CPU boundary')
            nodes=[(s['task'],s['operation']) for s in sections if s['kind']=='matmul']
            need(nodes==matmuls,'matmul execution coverage mismatch')
            wall=(v['excluded_ms']+samples)[i]
            need(all(0<=s['start_ns']<=s['end_ns']<=wall*1e6+1 for s in sections),'host interval outside complete graph timing')
            if 'api_events' in h:
                api=h['api_events'];starts=[e for e in api if e['kind']=='actual_xrt_start'];waits=[e for e in api if e['kind']=='actual_xrt_wait'];syncs=[e for e in api if e['kind']=='actual_bo_sync']
                need(len(starts)==len(waits)==len(matmuls)==h['actual_submissions'],'observed graph starts/waits do not match semantic matmuls')
                need(all(0<=e['start_ns']<=e['end_ns']<=wall*1e6+1 for e in api),'actual API interval outside graph wall')
                need([e['kind'] for e in api if e['kind']!='actual_bo_sync']==['actual_xrt_start','actual_xrt_wait']*len(matmuls),'graph start/wait ordering invalid')
                to_device=sum('TO_DEVICE' in e['direction'] for e in syncs);from_device=sum('FROM_DEVICE' in e['direction'] for e in syncs)
                need(from_device==len(matmuls),'missing actual output readback sync')
                if i>=2:need(to_device==2*len(matmuls),'hot activation/output staging or static-weight reuse differs from declared path')
                api_counts[mode].append(dict(repeat=i,starts=len(starts),waits=len(waits),sync_to_device=to_device,sync_from_device=from_device))
            else:instrumented=False
    need(math.isclose(r['trace_perturbation_ratio'],medians['on']/medians['off'],rel_tol=1e-12),'trace perturbation ratio mismatch')
    # Both builds must consume exactly the same AIR source. Trace reservation
    # and resulting physical layout/ELF may differ and are explicitly recorded.
    for directory in (root/'off').glob('shape-*'):
        twin=root/'on'/directory.name
        need((directory/'source.mlir').read_bytes()==(twin/'source.mlir').read_bytes(),'on/off AIR computation differs')
    manifests=list((root/'on').glob('trace-*/manifest.json'));seen=[]
    taskmap={t['id']:t for t in plan['tasks']}
    for file in manifests:
        meta=json.loads(file.read_text());key=(meta['task_id'],meta['operation_index']);seen.append(key)
        need(meta['candidate_id']==plan['candidate_id'] and key in matmuls,'trace mapped to wrong Taskflow operation')
        op=taskmap[key[0]]['operations'][key[1]]
        logical=[op['m'][1]-op['m'][0],op['n'][1]-op['n'][0],op['k'][1]-op['k'][0]]
        need(meta['logical_shape']==logical,'trace logical shape disagrees with Taskflow operation')
        build=root/'on'/('shape-'+'-'.join(map(str,meta['physical_shape'])))
        comp=json.loads((build/'compile.json').read_text())
        need(meta['air_sha256']==digest(build/'source.mlir')==comp['source_sha256'],'trace AIR provenance differs')
        need(meta['elf_sha256']==comp['elf_sha256'],'trace ELF provenance differs')
        need((file.parent/'physical.mlir').read_bytes()==(build/'input_with_addresses.mlir').read_bytes(),'trace uses wrong physical decoder configuration')
        capture=meta.get('capture',{})
        need(capture.get('repeat_index')==len(r['variants']['on']['full_graph_ms'])+1 and capture.get('hot_sample_index')==len(r['variants']['on']['full_graph_ms'])-1,'trace capture iteration mapping missing/wrong')
    need(sorted(seen)==sorted(matmuls),'per-projection trace coverage missing/duplicated')
    limitations=['Same-case on/off comparison only; do not compare these graph-driver timings with older runtime batches.','Tracing changes output BO size, staging/readback and physical resource placement; ratio is total perturbation, not isolated trace instruction overhead.','Runner sections are combined host staging/sync/submit/wait/readback intervals, not device-only time; parent matmul intervals overlap them and must not be summed.']
    if not instrumented:limitations.append('Submission metadata follows one-op runner code; actual XRT start not independently observed in this batch.')
    else:limitations.append('BO sync calls are observed host cache-maintenance API calls, not measured DMA byte traffic.')
    return dict(passed=not errors,errors=errors,case=r['case'],source=str(result_path),candidate_id=r['candidate_id'],median_ms=medians,perturbation_ratio=medians['on']/medians['off'],trace_on_off_max_abs=max_pair,output_checks=checks,cpu_swiglu_nodes=swiglus,api_instrumented=instrumented,api_counts=api_counts,limitations=limitations)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trace-manifest',type=Path,action='append',default=[]);parser.add_argument('--chain-result',type=Path,action='append',default=[]);parser.add_argument('--graph-result',type=Path,action='append',default=[]);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();results=[];chains=[];graphs=[]
    for manifest in args.trace_manifest:
        try:results.append(trace_audit(manifest))
        except Exception as exc:results.append(dict(passed=False,manifest=str(manifest),errors=[type(exc).__name__+': '+str(exc)]))
    for result in args.chain_result:
        try:chains.append(chain_audit(result))
        except Exception as exc:chains.append(dict(passed=False,source=str(result),errors=[type(exc).__name__+': '+str(exc)]))
    for result in args.graph_result:
        try:graphs.append(graph_trace_audit(result))
        except Exception as exc:graphs.append(dict(passed=False,source=str(result),errors=[type(exc).__name__+': '+str(exc)]))
    report=dict(passed=bool(results+chains+graphs) and all(r['passed'] for r in results+chains+graphs),audit_kind='read_only_independent_hardware_evidence',traces=results,chains=chains,graphs=graphs,conditional_holdout=conditional_holdout(chains))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],traces=len(results),output=str(args.output))))
    return 0 if report['passed'] else 1


if __name__=='__main__':sys.exit(main())
