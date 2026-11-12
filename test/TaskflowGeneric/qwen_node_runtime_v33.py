#!/usr/bin/env python3
"""Lightweight source/partition contracts; no torch/model/device imports."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('qwen_node',ROOT/'scripts/qwen-node-runtime.py');node=importlib.util.module_from_spec(spec);spec.loader.exec_module(node)
VENDOR=Path('/home/shiran/project/aie-toy/build/qwen25-3b/Triton-XDNA')

@unittest.skipUnless(VENDOR.exists(),'requires local pinned source checkout; never downloads')
class SourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.source=node.extract_prefill_source(VENDOR,5)
    def test_actual_complete_source_and_node_domain(self):
        operators=self.source['operators'];self.assertEqual(len(operators),581)
        variable=[o for o in operators if o['attributes'].get('supported_devices')==['cpu','npu']]
        self.assertEqual(len(variable),253)
        self.assertEqual(sum(o['kind']=='matmul' for o in variable),144)
        self.assertEqual(sum(o['kind']=='rms-norm' for o in variable),73)
        self.assertEqual(sum(o['kind']=='swiglu' for o in variable),36)
        self.assertEqual(sum(o['kind']=='add' for o in operators),72)
        self.assertEqual(sum(o['kind']=='kv-prefix-store' for o in operators),36)
    def test_source_values_are_ssa_with_prior_dependencies(self):
        values={v['id']:v for v in self.source['values']};produced={};seen=set()
        for operator in self.source['operators']:
            self.assertTrue(set(operator['dependencies'])<=seen)
            for value in operator['inputs']:self.assertTrue(values[value]['role']=='input' or value in produced)
            for value in operator['outputs']:self.assertNotIn(value,produced);produced[value]=operator['id']
            seen.add(operator['id'])
        for value in values.values():
            if 'view' in value:
                view=value['view'];self.assertIn(view['base'],values)
                self.assertEqual(len(view['offsets']),len(value['shape']))
                self.assertEqual(len(view['strides']),len(value['shape']))
    def test_arbitrary_per_node_placement_retains_whole_source(self):
        overrides={'prefill.L0.qkv':'npu','prefill.L17.ffn_norm':'npu','prefill.L35.down':'npu'}
        plan=node.make_plan(self.source,overrides)
        node.validate_plan(self.source,plan)
        self.assertEqual(len(plan['placements']),581)
        self.assertTrue(all(plan['placements'][name]==device for name,device in overrides.items()))
        self.assertEqual(plan['placements']['prefill.L1.qkv'],'cpu')
    def test_backend_hard_limits_are_not_silently_repaired(self):
        for name in ['prefill.L0.attention','prefill.L2.q_rope','prefill.L3.residual1','prefill.lm_head']:
            with self.assertRaises(ValueError):node.make_plan(self.source,{name:'npu'})
    def test_missing_or_duplicate_partition_nodes_rejected(self):
        good=node.make_plan(self.source)
        for change in ('missing','duplicate','reverse','device'):
            plan=copy.deepcopy(good)
            if change=='missing':plan['regions'][0]['operators'].pop()
            elif change=='duplicate':plan['regions'][0]['operators'].append(plan['regions'][0]['operators'][0])
            elif change=='reverse':plan['regions'][0]['operators'].reverse()
            else:plan['regions'][0]['device']='npu'
            with self.assertRaises(ValueError):node.validate_plan(self.source,plan)
    def test_source_arithmetic_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for rel in ['examples/llm_q4nx/qwen25_prefill.py','examples/llm_q4nx/llama_prefill.py','examples/qwen25_3b_q4nx/config.py']:
                path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text((VENDOR/rel).read_text())
            qwen=root/'examples/llm_q4nx/qwen25_prefill.py'
            qwen.write_text(qwen.read_text().replace('qkv_bias"]','qkv_bias"] * 2'))
            with self.assertRaises(ValueError):node.extract_prefill_source(root,5)

if __name__=='__main__':unittest.main()
