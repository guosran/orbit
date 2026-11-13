#!/usr/bin/env python3
"""Device-free fault injection for post-submit BO/context ownership."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from ml_dtypes import bfloat16

spec=importlib.util.spec_from_file_location('owned_xrt',Path(__file__).with_name('qwen-r19-xrt.py'))
owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)


class OwnershipTests(unittest.TestCase):
    def invoke(self,unknown):
        calls=[];runs=[];unloaded=[]
        class BO:
            def __init__(self,device,size):self.data=bytearray(size)
            def map(self):return memoryview(self.data)
            def sync(self,direction):pass
        class Run:
            def __init__(self,kernel):self.pending=True;runs.append(self)
            def set_arg(self,*args):pass
            def start(self):calls.append('submit')
            def state(self):
                if unknown:raise RuntimeError('state transport unavailable')
                return 1 if self.pending else 4
        class Runner:
            def __init__(self,*args):
                self.device=object();self.kernel=object()
                self._xrt=types.SimpleNamespace(ext=types.SimpleNamespace(bo=BO),run=Run,
                    xclBOSyncDirection=types.SimpleNamespace(XCL_BO_SYNC_BO_TO_DEVICE=1,XCL_BO_SYNC_BO_FROM_DEVICE=2))
            def unload(self):unloaded.append(self)
        def wait(run):
            calls.append('wait')
            if unknown or calls.count('wait')==1:raise RuntimeError('injected post-launch failure')
            run.pending=False
        backend=types.ModuleType('triton.backends.amd_triton_npu.multilaunch')
        backend.MultiLaunchRunner=Runner;backend._wait=wait;backend._PENDING_STATES={1,2,3}
        modules={}
        for name in ('triton','triton.backends','triton.backends.amd_triton_npu'):
            package=types.ModuleType(name);package.__path__=[];modules[name]=package
        modules[backend.__name__]=backend
        impl=owned.PrivateGemm.__new__(owned.PrivateGemm)
        impl.record=dict(compiled_binary=dict(path='mock-only',sha256='mock-only'),kernel_name='mock',
            shape=dict(M=1,N=2,K=1),padded_shape=dict(M=1,N=2,A_stride=1),resource=dict(task_instance_replicas=1))
        impl.events=[]
        with patch.dict(sys.modules,modules):
            with self.assertRaises(RuntimeError):impl.run(np.ones((1,1),dtype=bfloat16),np.ones((1,2),dtype=bfloat16),'mock')
        return calls,runs,unloaded
    def test_failure_drains_before_unload(self):
        calls,runs,unloaded=self.invoke(False)
        self.assertEqual(calls,['submit','wait','wait'])
        self.assertFalse(runs[0].pending);self.assertEqual(len(unloaded),1)
    def test_unknown_state_never_unloads_actual_owners(self):
        before=len(owned._QUARANTINED);calls,runs,unloaded=self.invoke(True)
        self.assertEqual(unloaded,[]);self.assertTrue(runs[0].pending)
        self.assertEqual(len(owned._QUARANTINED),before+1)
        self.assertIs(owned._QUARANTINED[-1][0],runs[0])


if __name__=='__main__':unittest.main()
