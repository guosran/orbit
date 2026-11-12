#!/usr/bin/env python3
"""Strict compile-only identity checks before selecting a decode context."""
import hashlib
import json
import time
from pathlib import Path

ABI='func.func @q4nx_decode(%arg0: memref<2048xbf16>, %arg1: memref<983040000xbf16>, %arg2: memref<246272xbf16>, %arg3: memref<158208xbf16>, %arg4: memref<37748736xbf16>) {'


def identity(path):
    p=Path(path).resolve(strict=True)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)


def checked_record(row):
    if identity(row['path'])!=row:raise ValueError('Decode artifact identity changed: '+row['path'])


def validate(directory):
    directory=Path(directory).resolve(strict=True)
    template=json.loads((directory/'variant.json').read_text())
    if template.get('schema')!='orbit.qwen-decode-variant-template.v1' or (template['layers'],template['vocab_waves'],template['capacity'])!=(36,25,2048) or template['abi']!=ABI:
        raise ValueError('Decode variant is not complete36layer25vocabwave capacity2048 ABI')
    if template['column_rotation'] not in (0,4) or template['parameters'].get('ATTN_PCOL')!=4 or any(template['parameters'].get(k) not in (0,1) for k in ('PROJ_RC_CACHE','COALESCE')):
        raise ValueError('Unregistered decode execution configuration')
    builds=[]
    for row in template['builds']:
        checked_record(row);build=json.loads(Path(row['path']).read_text())
        if build.get('schema')!='orbit.qwen-decode-variant-compiled.v1':raise ValueError('Invalid compiled decode proof')
        for key in ('variant_id','parameters','column_rotation','abi','device_files'):
            if build[key]!=template[key]:raise ValueError('Decode pair changes program/CDO/ABI: '+key)
        for key in ('request','source','source_air','binary','instructions'):checked_record(build[key])
        request=json.loads(Path(build['request']['path']).read_text())
        for key in ('source','owned_source','producer'):checked_record(request[key])
        if (request['layers'],request['vocab_waves'])!=(36,25):raise ValueError('Build request omitted source layers')
        for key in ('variant_id','parameters','column_rotation'):
            if request[key]!=template[key]:raise ValueError('Decode request does not bind compiled configuration')
        for name,expected in build['device_files'].items():
            p=Path(row['path']).parent/'air_project'/name
            if identity(p)['sha256']!=expected:raise ValueError('Actual core/CDO artifact changed')
        length=build['length']
        if (length+15)//16*16!=2048:raise ValueError('Template capacity mismatch')
        for suffix,key in (('xclbin','binary'),('insts.bin','instructions')):
            if identity(directory/f'decode_L{length}.{suffix}')['sha256']!=build[key]['sha256']:
                raise ValueError('Runtime symlink does not resolve to certified artifact')
        builds.append(build)
    if sorted(b['length'] for b in builds)!=[2047,2048]:raise ValueError('Require exact L2047/L2048 template pair')
    return dict(schema='orbit.qwen-r18-checked-decode-context.v1',directory=str(directory),
                template_identity=identity(directory/'variant.json'),variant=template,
                exclusive_context=True,context_lifetime='whole generation request series',
                device_loaded=False)


class ProfiledDecoder:
    """Execute original dispatch unchanged, exposing mutually exclusive phases."""
    def __init__(self,decoder):
        if decoder.elf_mode:raise ValueError('Phase proxy requires certified xclbin path')
        self.decoder=decoder;self.kernel=decoder.kern;self.dispatch_events=[]

    def dispatch(self,token,position):
        owner=self
        if self.decoder.kern is not self.kernel:raise ValueError('Decode context/window changed')
        events=[]
        class Run:
            def __init__(self,native):self.native=native
            def wait(self,*args):
                before=time.perf_counter_ns();result=self.native.wait(*args);after=time.perf_counter_ns()
                events[0].update(wait_start_ns=before,wait_end_ns=after)
                return result
        class Kernel:
            def __call__(self,*args):
                if events:raise ValueError('Full decode source requires exactly one kernel invocation')
                before=time.perf_counter_ns();native=owner.kernel(*args);after=time.perf_counter_ns()
                events.append(dict(submit_start_ns=before,submit_end_ns=after))
                return Run(native)
        proxy=Kernel();start=time.perf_counter_ns();self.decoder.kern=proxy
        try:result=self.decoder.dispatch(token,position)
        finally:self.decoder.kern=self.kernel
        end=time.perf_counter_ns()
        if len(events)!=1 or 'wait_end_ns' not in events[0]:raise ValueError('Incomplete synchronous decode call')
        event=events[0];before=event['submit_start_ns']-start;after=end-event['wait_end_ns']
        submit=event['submit_end_ns']-event['submit_start_ns'];wait=event['wait_end_ns']-event['wait_start_ns']
        control=end-start-before-after-submit-wait
        if min(before,after,submit,wait,control)<0:raise ValueError('Decode phases overlap')
        self.dispatch_events.append(dict(schema='orbit.qwen-decode-dispatch-phases.v1',position=position,
            length=position+1,wall_ns=end-start,host_before_ns=before,host_after_ns=after,
            host_control_ns=control,region_events=[dict(region=dict(lo=0,hi=61),instruction_patch_ns=0,
                submit_ns=submit,wait_ns=wait,service_ns=submit+wait,shared_bo_transfer_bytes=0)],
            exclusive_sum_equals_wall=before+after+control+submit+wait==end-start,
            boundary='original full-context dispatch; original instruction patch belongs to host-before'))
        return result
