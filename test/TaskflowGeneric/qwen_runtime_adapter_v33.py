#!/usr/bin/env python3
"""CPU-only replay binding tests; no imports of model execution or device access."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('qwen_adapter', ROOT / 'scripts/qwen-runtime-adapter.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.m = dict(schema=q.SCHEMA,native_program_graph=False,aie_root=self.tmp.name,
                      fixed=dict(q.FIXED),input=dict(prefill_tokens=5))
        self.m['source_id'] = q.identity(self.m)

    def command(self, m, p, **kw):
        with patch.object(q,'check_manifest') as check:
            result=q.command(m,p,'fresh',kw.get('repeats',3),kw.get('warmup',5),kw.get('measure',10))
            check.assert_called_once_with(m)
            return result

    def test_all_four_assignments_map_to_supported_original_cli(self):
        for rms in ('cpu','npu'):
            for swiglu in ('cpu','npu'):
                plan=q.make_plan(self.m,rms,swiglu)
                cmd=self.command(self.m,plan)
                ops=cmd[cmd.index('--ops')+1].split(',')
                self.assertEqual(ops,['matmul']+(['rms_norm'] if rms=='npu' else [])+(['swiglu'] if swiglu=='npu' else []))
                self.assertIn('--static-weights',cmd)
                self.assertIn('--exact-k',cmd)
                self.assertEqual(cmd[cmd.index('--outputs')+1],'128')
                self.assertEqual(cmd[cmd.index('--max-seq')+1],'2048')

    def test_plan_tampering_rejected(self):
        for key,value in [('source_id','wrong'),('fixed',dict(q.FIXED,bucket=128)),('plan_id','wrong')]:
            plan=q.make_plan(self.m,'cpu','npu');plan[key]=value
            with self.assertRaises(ValueError): self.command(self.m,plan)

    def test_manifest_identity_and_real_content_checked(self):
        bad=copy.deepcopy(self.m);bad['fixed']['bucket']=128
        with self.assertRaises(ValueError):q.check_manifest(bad)
        with patch.object(q,'register',return_value=dict(self.m,files=['changed'])):
            with self.assertRaisesRegex(ValueError,'changed'):q.check_manifest(self.m)

    def test_prefill_specialization_preserved(self):
        for n in (5,128,512):
            m=copy.deepcopy(self.m);m['fixed']['lengths']=str(n);m['input']['prefill_tokens']=n
            m['source_id']=q.identity({k:v for k,v in m.items() if k!='source_id'})
            cmd=self.command(m,q.make_plan(m,'cpu','cpu'))
            self.assertEqual(cmd[cmd.index('--lengths')+1],str(n))

    def test_invalid_budget_rejected(self):
        for kw in [dict(repeats=0),dict(warmup=float('nan')),dict(measure=float('inf')),dict(measure=-1)]:
            with self.assertRaises(ValueError):self.command(self.m,q.make_plan(self.m,'cpu','cpu'),**kw)

if __name__=='__main__': unittest.main()
