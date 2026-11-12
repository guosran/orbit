#!/usr/bin/env python3
"""CPU-only source-rewrite and real runner-API ownership contract tests."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch


def load(path):
    spec=importlib.util.spec_from_file_location('r18_ffn_test_module',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


MODULE=load(Path(__file__).with_name('qwen-r18-ffn.py'))
ROOT=Path('/home/shiran/project/aie-toy/build/qwen25-3b/Triton-XDNA')


class OwnershipTests(unittest.TestCase):
    def test_static_reload_release_are_actual_runner_arguments(self):
        import numpy as np
        import torch
        from ml_dtypes import bfloat16
        for policy in ('retain','reload','release-after-call'):
            with self.subTest(policy=policy):
                ffn=MODULE.QwenFFN(ROOT,retention=policy)
                ffn.rows=1;ffn.hidden=1
                # Small ABI placeholders exercise ownership, without claiming
                # numerical or physical device validation of fake dispatches.
                ffn.weights[0]=(None,None,None)
                ffn.arguments[0]=[np.zeros((1,2048),dtype=bfloat16),
                    *[np.zeros(1,dtype=bfloat16) for _ in range(6)],
                    np.zeros((1,2048),dtype=np.float32)]
                calls=[]
                fake=types.SimpleNamespace(_runtime='xrt',_runner=types.SimpleNamespace(_bos={}))
                def run(args,**kwargs):
                    calls.append(kwargs)
                    fake._runner._bos[kwargs['bo_key']]=object()
                    args[7].fill(len(calls))
                    return {7:args[7]}
                fake.run=run;ffn.build=lambda:fake
                with patch('triton.backends.amd_triton_npu.driver_scope.driver_scope',lambda _:nullcontext()), patch('kernels.shared_bo',lambda _:object()):
                    first=ffn.run(0,torch.zeros(1,2048))
                    ffn.events=[];ffn.call_events=[]
                    second=ffn.run(0,torch.zeros(1,2048))
                self.assertTrue(torch.equal(first,torch.ones(1,2048)))
                self.assertTrue(torch.equal(second,torch.full((1,2048),2.)))
                self.assertEqual(calls[1]['static_indices'],{1,2,3} if policy=='retain' else set())
                self.assertEqual(ffn.events[0]['first_call'],policy=='release-after-call')
                self.assertEqual(bool(fake._runner._bos),policy!='release-after-call')
                self.assertEqual(calls[0]['intermediate_indices'],{4,5,6,7})
                self.assertEqual(calls[0]['output_indices'],{7})

    def test_registration_does_not_claim_sram(self):
        ffn=MODULE.QwenFFN(ROOT)
        desc=ffn.describe()
        self.assertFalse(desc['sram_residency_claim'])
        self.assertEqual(desc['memory'],'device-ddr-bo')
        self.assertEqual(len(desc['buffers']),8)
        self.assertEqual(desc['kernel_order'],['gate','up','swiglu','down'])
        self.assertEqual(desc['buffers'][4]['download'],'never')


class RewriteTests(unittest.TestCase):
    def test_original_prefix_and_source_layer_identity(self):
        MODULE.configure(ROOT)
        from qwen25_prefill import Qwen25Prefill
        events=[]
        class Tensor:
            def __init__(self,name):self.name=name
            def __add__(self,other):return Tensor('add')
            def split(self,sizes,dim):return [Tensor('split') for _ in sizes]
        class Chain:
            def run(self,layer,h):events.append(('fused',layer));return Tensor('down')
        fused=MODULE.make_prefiller_class(Qwen25Prefill,Chain(),[7])
        class Harness(fused):
            def _rms_norm(self,x,w,eps):events.append(('norm',w));return Tensor('norm')
            def _matmul(self,x,w,**kwargs):events.append(('matmul',w));return Tensor('mm')
            def _rope(self,x,lut,heads):events.append(('rope',heads));return x
            def _kv_store(self,L,k,v,keep):events.append(('kv',L,keep))
            def _attention(self,*args):events.append(('attention',));return Tensor('attention')
            def _swiglu(self,*args):events.append(('swiglu',));return Tensor('swiglu')
        obj=object.__new__(Harness)
        obj._w=[dict(attn_norm='an',ffn_norm='fn',qkv='qkv',qkv_bias=Tensor('bias'),o='o',gate_up='gu',down='down') for _ in range(36)]
        obj._lut=[]
        obj._layer(Tensor('input'),7,5,keep=3)
        self.assertEqual(events,[('norm','an'),('matmul','qkv'),('rope',16),('rope',2),('kv',7,3),('attention',),('matmul','o'),('norm','fn'),('fused',7)])
        events.clear();obj._layer(Tensor('input'),8,5)
        self.assertNotIn(('fused',8),events)
        self.assertEqual(events[-3:],[('matmul','gu'),('swiglu',),('matmul','down')])


if __name__=='__main__':unittest.main()
