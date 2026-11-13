#!/usr/bin/env python3
"""Private-BO GEMM invocation with real transfer/launch/complete boundaries.

No shared-buffer registry, NPUChain eviction cache, or vendor monkeypatch is
used. A single ReadyExecutor worker owns context creation through destruction.
The original multilaunch GIL-releasing wait is reused. BOs remain strongly
referenced until completion, host visibility, and cloned output assembly.
"""
import hashlib
import json
from pathlib import Path
import threading
import time


# Unknown device states retain every actual owner until process teardown.
# An exception is never treated as evidence of command completion.
_QUARANTINED=[]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class PrivateGemm:
    def __init__(self,directory):
        self.directory=Path(directory).resolve(strict=True)
        self.record=json.loads((self.directory/'implementation.json').read_text())
        for key in ('source_air','compiled_binary'):
            row=self.record[key]
            if sha(row['path'])!=row['sha256']:raise ValueError('Compiled artifact drift: '+key)
        self.events=[]

    def run(self,x,w,stage_key):
        import numpy as np
        from ml_dtypes import bfloat16
        from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner,_wait,_PENDING_STATES
        owner=threading.get_ident();start=time.perf_counter_ns();phases=[];bos=[];runner=None;active_run=None;submitted=False;terminal=False
        def phase(kind,begin,**detail):
            end=time.perf_counter_ns();phases.append(dict(kind=kind,start_ns=begin,end_ns=end,wall_ns=end-begin,**detail));return end
        try:
            now=time.perf_counter_ns()
            runner=MultiLaunchRunner(self.record['compiled_binary']['path'],self.record['kernel_name'])
            phase('context-open',now)
            x,w=np.asarray(x),np.asarray(w);s=self.record['shape'];pad=self.record['padded_shape']
            m,n,k=s['M'],s['N'],s['K'];replicas=self.record['resource']['task_instance_replicas']
            if x.shape!=(m,k) or w.shape!=(k,n):raise ValueError('Actual source shape differs from artifact ABI')
            now=time.perf_counter_ns();a=np.zeros((pad['M'],pad['A_stride']),dtype=bfloat16);a[:m,:k]=x
            arrays=[a];outputs=[]
            for replica in range(replicas):
                b=np.zeros((k,pad['N']),dtype=bfloat16)
                b[:,:n//replicas]=w[:,replica*(n//replicas):(replica+1)*(n//replicas)]
                arrays.extend((b,np.zeros((pad['M'],pad['N']),dtype=np.float32)));outputs.append(len(arrays)-1)
            phase('host-pack',now,bytes=sum(v.nbytes for v in arrays))
            xrt=runner._xrt;now=time.perf_counter_ns()
            bos=[xrt.ext.bo(runner.device,int(v.nbytes)) for v in arrays]
            allocation_end=phase('bo-allocate',now,allocations=[dict(id=f'{stage_key}.arg{i}',bytes=int(v.nbytes),memory='runtime-abi') for i,v in enumerate(arrays)])
            for i,(bo,array) in enumerate(zip(bos,arrays)):
                now=time.perf_counter_ns()
                np.copyto(np.frombuffer(bo.map(),dtype=np.uint8,count=array.nbytes),array.view(np.uint8).reshape(-1),casting='no')
                phase('host-to-bo-copy',now,argument=i,bytes=array.nbytes)
                now=time.perf_counter_ns();bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                phase('sync-to-device',now,argument=i,requested_sync_bytes=array.nbytes)
            run=xrt.run(runner.kernel);active_run=run
            for i,bo in enumerate(bos):run.set_arg(i,bo)
            now=time.perf_counter_ns();submitted=True;run.start();launched=phase('launch',now)
            now=time.perf_counter_ns();_wait(run);terminal=int(run.state())==4
            if not terminal:raise RuntimeError('Device wait returned without proven COMPLETED state')
            completed=phase('complete',now,wait_policy='original multilaunch._wait: npu_wait or GIL-releasing polling')
            for i in outputs:
                now=time.perf_counter_ns();bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                phase('sync-from-device',now,argument=i,requested_sync_bytes=arrays[i].nbytes)
            now=time.perf_counter_ns()
            result=np.concatenate([np.frombuffer(bos[i].map(),dtype=np.float32,count=arrays[i].size).reshape(arrays[i].shape)[:m,:n//replicas] for i in outputs],axis=1).copy()
            visible=phase('host-output-clone',now)
            # Release run first, then BOs, then context; no exported view aliases
            # a freed BO and no context can be closed while a command is active.
            run=None;active_run=None;now=time.perf_counter_ns();bos.clear();runner.unload();runner=None
            released=phase('bo-and-context-release',now)
            end=time.perf_counter_ns();accounted=sum(v['wall_ns'] for v in phases)
            if accounted>end-start:raise RuntimeError('Exclusive phase timings overlap')
            event=dict(schema='orbit.qwen-r19-private-xrt-invocation.v1',stage_key=stage_key,owner_thread=owner,
                start_ns=start,end_ns=end,wall_ns=end-start,phases=phases,
                exclusive_control_ns=end-start-accounted,exclusive_sum_equals_wall=True,
                launched_ns=launched,completed_ns=completed,host_visible_ns=visible,
                bo_lifetime=dict(allocated_ns=allocation_end,released_ns=released),
                artifact_sha256=self.record['compiled_binary']['sha256'],
                resident_policy='private-per-call; pack/upload/complete/readback/clone/release',
                transfer_semantics='host memcpy and actual XRT sync measured separately; sync bytes are requests, not claimed bus traffic')
            self.events.append(event)
            return result
        finally:
            if runner is not None:
                if submitted and not terminal:
                    # Drain after *any* post-submit failure, not just a wait
                    # error. _wait releases the GIL. A terminal error state is
                    # safe to release; an unreadable/unknown state is not.
                    try:
                        _wait(active_run)
                        terminal=int(active_run.state()) in (4,5,6)
                    except BaseException:
                        try:terminal=int(active_run.state()) in (4,5,6)  # COMPLETED, ERROR, ABORT only
                        except BaseException:terminal=False
                    if not terminal:
                        _QUARANTINED.append((active_run,bos,runner))
                        raise RuntimeError('Unknown active XRT command state: BO/context owners quarantined, never freed')
                active_run=None;bos.clear();runner.unload()
