#!/usr/bin/env python3
"""Small device-free tests of explicit baseline exact-cover binding."""
import copy
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).with_name('prepare-r19-baseline-choices.py')
spec=importlib.util.spec_from_file_location('baseline_choices',path)
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)


def fixture():
    source=dict(operators=[dict(id='prefill.a'),dict(id='prefill.b')])
    source['source_id']=baseline.digest(source)
    plan=dict(source_id=source['source_id'],placements={'prefill.a':'npu','prefill.b':'cpu'})
    plan['plan_id']=baseline.digest(plan)
    catalogue=dict(source_id='whole',implementations=[],choice_groups=[],runtime_registration=dict(implementations={}))
    def add(identity,indices,registration):
        catalogue['implementations'].append(dict(id=identity,source_id='whole',source_operators=indices,executable=True,wall_ms=1))
        catalogue['runtime_registration']['implementations'][identity]=registration
    for index,(op,device) in enumerate(plan['placements'].items()):
        add(op,[index],dict(kind='prefill-original',operators=[op],operator=op,device=device))
    add('fused',[0,1],dict(kind='prefill-cpu-region',operators=['prefill.a','prefill.b']))
    catalogue['choice_groups'].append(dict(id='prefix',choices=[dict(id='original',implementations=['prefill.a','prefill.b']),dict(id='fused',implementations=['fused'])]))
    ids=[]
    for step in range(127):
        identity='decode'+str(step);ids.append(identity)
        add(identity,[step+2],dict(kind='decode-context',operators=[identity],step=step,decode_variant_dir='/tmp/decoder',decode_rope_prefetch=False))
    catalogue['choice_groups'].append(dict(id='decode',choices=[dict(id='whole',implementations=ids)]))
    return catalogue,source,plan


class BaselineTests(unittest.TestCase):
    def test_original_cover_uses_actual_ids(self):
        catalogue,source,plan=fixture()
        self.assertEqual(baseline.choose(catalogue,source,plan,'/tmp/decoder',False),[dict(group_id='prefix',choice_id='original'),dict(group_id='decode',choice_id='whole')])

    def test_missing_requested_context_rejects(self):
        catalogue,source,plan=fixture()
        with self.assertRaisesRegex(ValueError,'exactly one'):baseline.choose(catalogue,source,plan,'/tmp/decoder',True)

    def test_ambiguous_original_registration_rejects(self):
        catalogue,source,plan=fixture()
        catalogue['choice_groups'][0]['choices'].append(copy.deepcopy(catalogue['choice_groups'][0]['choices'][0]))
        with self.assertRaisesRegex(ValueError,'exactly one'):baseline.choose(catalogue,source,plan,'/tmp/decoder',False)

    def test_missing_generation_step_rejects(self):
        catalogue,source,plan=fixture()
        catalogue['choice_groups'][1]['choices'][0]['implementations'].pop()
        with self.assertRaisesRegex(ValueError,'omits'):baseline.choose(catalogue,source,plan,'/tmp/decoder',False)


if __name__=='__main__':unittest.main()
