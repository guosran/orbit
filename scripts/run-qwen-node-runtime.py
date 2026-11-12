#!/usr/bin/env python3
"""Run the original complete Qwen source with an explicit per-node prefill plan.

The decoder may use checked source-preserving wave regions. This runner does
not claim that separate prefill operators fuse into one device launch. Every
measurement carries the same source-line and tensor-provenance instrumentation.
"""
import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def memory():
    status={}
    for line in Path('/proc/self/status').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('VmRSS','VmHWM','VmSwap'):status[key+'_bytes']=int(value.split()[0])*1024
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):status['available_bytes']=int(line.split()[1])*1024
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aie-root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--region-root',type=Path);p.add_argument('--decode-regions',default='0:36,36:61');p.add_argument('--region-helper',type=Path,default=Path(__file__).resolve().parents[1]/'experimental/qwen-runtime-search/decode-regions.py')
    p.add_argument('--outputs',type=int,default=128);p.add_argument('--warmups',type=int,default=1);p.add_argument('--repeats',type=int,default=3);p.add_argument('--threads',type=int,default=4);p.add_argument('--rss-limit-gib',type=float,default=20);p.add_argument('--swap-limit-gib',type=float,default=.5)
    a=p.parse_args()
    if not 1<=a.outputs<=128 or a.warmups<0 or a.repeats<1 or a.threads<1:raise ValueError('Invalid request budget')
    source=json.loads(a.source.read_text());plan=json.loads(a.plan.read_text());adapter=module(Path(__file__).with_name('qwen-node-runtime.py'),'orbit_qwen_node_runtime')
    adapter.validate_plan(source,plan)
    triton=a.aie_root/'build/qwen25-3b/Triton-XDNA'
    prompt_tokens=next(v['shape'][0] for v in source['values'] if v['id']=='input.tokens')
    if adapter.extract_prefill_source(triton,prompt_tokens)!=source:raise ValueError('Actual source no longer matches graph')
    if prompt_tokens+a.outputs-1>2048:raise ValueError('Request exceeds actual KV capacity')
    a.output.mkdir(parents=True,exist_ok=False)
    def emit(name,row):
        with (a.output/name).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    def check_memory(stage):
        m=memory();emit('memory.jsonl',dict(stage=stage,**m))
        if m['VmRSS_bytes']>a.rss_limit_gib*2**30 or m['VmSwap_bytes']>a.swap_limit_gib*2**30:raise RuntimeError('Measured process memory exceeds this run contract')
        return m
    paths=json.loads((a.aie_root/'results/qwen25-3b/model-paths.json').read_text())
    os.environ.update(OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS='1',XILINX_XRT='/usr',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',Q4NX_MODEL_SOURCE=paths['model'],Q4NX_TOKENIZER_DIR=paths['tokenizer'],DECODE_ELF='0')
    import numpy as np
    import torch
    from filelock import FileLock
    from transformers import AutoTokenizer
    torch.set_num_threads(a.threads)
    tokenizer=AutoTokenizer.from_pretrained(paths['tokenizer'],local_files_only=True)
    base=tokenizer.encode('The capital of France is',add_special_tokens=False);filler=tokenizer.encode('A computer processes information using arithmetic and memory. ',add_special_tokens=False)
    ids=base if prompt_tokens==5 else (filler*(prompt_tokens//len(filler)+1))[:prompt_tokens-5]+base
    if len(ids)!=prompt_tokens:raise ValueError('Prompt does not match source input shape')
    sys.path.insert(0,str(triton/'examples/qwen25_3b_q4nx'));import config
    sys.path.insert(0,config._SHARED)
    import harness
    import registry
    import kernels
    from qwen25_prefill import Qwen25Prefill
    cls=adapter.make_prefiller_class(Qwen25Prefill,source,plan)
    implementation_files=[dict(file=str(triton/'examples/llm_q4nx'/name),sha256=hashlib.sha256((triton/'examples/llm_q4nx'/name).read_bytes()).hexdigest()) for name in ('kernels.py','llama_prefill.py','qwen25_prefill.py','harness.py')]
    contract=dict(implementation_files=implementation_files,producer=dict(runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),node_adapter_sha256=hashlib.sha256(Path(__file__).with_name('qwen-node-runtime.py').read_bytes()).hexdigest()),schema='orbit.qwen-node-runtime-execution.v1',source_id=source['source_id'],plan_id=plan['plan_id'],prompt=ids,outputs=a.outputs,max_context=2048,threads=a.threads,
                  source=source,plan=plan,model=str(Path(paths['model']).resolve(strict=True)),
                  implementation='original Qwen25Prefill body and backend methods; original FusedDecoder.dispatch; optional checked wave-region wrapper',
                  instrumentation='source-node call/line and tensor-provenance instrumentation on every baseline/selected request',
                  cpu_weights='CPU-selected projections retain original BF16 tensors; NPU-selected projections use bitchecked original resident_weight exact-K preparation and release originals; no capacity-fit claim before memory receipts',
                  limits='device launch/sync costs belong to original synchronous calls and are not independently isolated')
    (a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    # The entire device lifetime, including preparations, uses the same lock.
    with FileLock('/tmp/npu.lock',timeout=10),torch.inference_mode():
        check_memory('before-load')
        pf=cls(backend='npu',ops='all',n_layers=36,max_seq=2048,model=paths['model'],expect_model='qwen2.5-3b');pf.SEQ_BUCKET=1;pf.load_weights(paths['model'])
        check_memory('prefill-loaded-original-cpu-weights')
        pf.make_npu_resident();check_memory('plan-specific-weight-preparation')
        (a.output/'weight-preparation.json').write_text(json.dumps(pf.orbit_preparation,indent=2)+'\n')
        # Same preparation action as the original experiment: release dead
        # loader/padding pages before allocating the decode's weight BOs. This
        # is outside every request timer and does not relax memory guards.
        emit('memory.jsonl',dict(stage='before-loader-page-release',**memory()))
        gc.collect()
        trimmed=ctypes.CDLL('libc.so.6').malloc_trim(0)
        emit('preparation.jsonl',dict(action='malloc_trim',result=int(trimmed),inside_request_timing=False))
        check_memory('after-loader-page-release')
        air=harness.air_inference_module(registry.spec('qwen2.5-3b'));decoder=air.FusedDecoder(model=paths['model'],max_L=2048)
        check_memory('decode-loaded')
        dispatch=decoder.dispatch;region_decoder=None
        if a.region_root:
            helper=module(a.region_helper,'orbit_qwen_decode_regions');directories=[a.region_root/f'region-{lo}-{hi}' for lo,hi in [part.split(':') for part in a.decode_regions.split(',')]]
            region_decoder=helper.RegionDecoder(decoder,directories);dispatch=region_decoder.dispatch
        def once(sample,warmup):
            start=time.perf_counter_ns();begin=time.perf_counter_ns();pf.clear_context();clear_ns=time.perf_counter_ns()-begin
            lg=np.asarray(pf.prefill(ids),np.float32);ttft=(time.perf_counter_ns()-start)/1e6
            begin=time.perf_counter_ns();logits=[lg.copy()];capture_copy_ns=time.perf_counter_ns()-begin
            begin=time.perf_counter_ns();tokens=[int(lg.argmax())];first_argmax_ns=time.perf_counter_ns()-begin
            steps=[];decode_trace=[]
            begin=time.perf_counter_ns();fk,fv=pf.kv_stack();decoder.seed_kv(fk,fv,len(ids));handoff_ns=time.perf_counter_ns()-begin;handoff=handoff_ns/1e6
            for i in range(a.outputs-1):
                begin=time.perf_counter_ns();lg=np.asarray(dispatch(tokens[-1],len(ids)+i),np.float32);outer_ns=time.perf_counter_ns()-begin;steps.append(outer_ns/1e6)
                begin=time.perf_counter_ns();logits.append(lg.copy());capture_copy_ns+=time.perf_counter_ns()-begin
                begin=time.perf_counter_ns();tokens.append(int(lg.argmax()));argmax_ns=time.perf_counter_ns()-begin
                detail=dict(step=i,position=len(ids)+i,outer_dispatch_ns=outer_ns,argmax_ns=argmax_ns)
                if region_decoder is not None:
                    phase=dict(region_decoder.dispatch_events[-1])
                    if phase.get('schema')!='orbit.qwen-decode-dispatch-phases.v1' or not phase.get('exclusive_sum_equals_wall'):raise ValueError('Decode helper lacks checked exclusive phase trace')
                    difference=outer_ns-phase['wall_ns']
                    if difference<0:raise ValueError('Outer decode clock smaller than nested dispatch')
                    detail.update(phases=phase,outside_dispatch_wrapper_ns=difference)
                decode_trace.append(detail)
            elapsed_ns=time.perf_counter_ns()-start;elapsed=elapsed_ns/1e6
            full=np.stack(logits)
            if full.shape!=(a.outputs,151936) or not np.isfinite(full).all():raise ValueError('Incomplete/nonfinite complete-model result')
            profile=pf.source_profile()
            exclusive=clear_ns+profile['wall_ns']+capture_copy_ns+first_argmax_ns+handoff_ns+sum(x['outer_dispatch_ns']+x['argmax_ns'] for x in decode_trace)
            request_control_ns=elapsed_ns-exclusive
            if request_control_ns<0:raise ValueError('Request phase accounting overlaps')
            m=check_memory(('warmup-' if warmup else 'sample-')+str(sample))
            row=dict(sample=sample,warmup=warmup,P=len(ids),outputs=a.outputs,total_ms=elapsed,ttft_ms=ttft,handoff_ms=handoff,decode_ms=steps,tokens=tokens,prefill_source_profile=profile,memory=m,
                     request_phases=dict(wall_ns=elapsed_ns,clear_context_ns=clear_ns,capture_copy_ns=capture_copy_ns,first_argmax_ns=first_argmax_ns,handoff_ns=handoff_ns,request_control_ns=request_control_ns,exclusive_sum_equals_wall=exclusive+request_control_ns==elapsed_ns),decode_trace=decode_trace)
            emit('measurements.jsonl',row)
            if not warmup and sample==0:np.savez(a.output/'capture.npz',logits=full,tokens=np.asarray(tokens),prompt=np.asarray(ids))
            print(json.dumps(dict(sample=sample,warmup=warmup,total_ms=elapsed,ttft_ms=ttft,peak_rss_bytes=m['VmHWM_bytes'])),flush=True)
        for i in range(a.warmups):once(i,True)
        for i in range(a.repeats):once(i,False)
        (a.output/'completion.json').write_text(json.dumps(dict(completed=True,measured_requests=a.repeats,warmups=a.warmups,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')

if __name__=='__main__':main()
