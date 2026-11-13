#!/usr/bin/env python3
"""Lightweight validation of normalized selection and physical-edge lowering."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).parent

def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


class SelectedBridgeTests(unittest.TestCase):
    def test_manual_baseline_requires_actual_no_search_materialization(self):
        bridge=module('replay-r19-selection')
        assignment=dict(schema='orbit.program-manual-choices.v1',purpose='manual-baseline',source_id='source',execution_registration_identity='registration',choices=[dict(group_id='group',choice_id='choice')],ready_priority_policy='source-order')
        catalogue=dict(source_id='source',execution_registration_identity='registration',choice_groups=[dict(id='group',choices=[dict(id='choice',implementations=['actual'])])])
        graph=dict(external_implementation_selection=dict(regions=[dict(id='actual')]),external_mapping=dict(ready_priority=[0]))
        result=dict(mode='manual-materialization',search_performed=False,manual_assignment=assignment)
        bridge.verify_manual_assignment(assignment,result,graph,catalogue)
        result['search_performed']=True
        with self.assertRaisesRegex(ValueError,'no-search'):bridge.verify_manual_assignment(assignment,result,graph,catalogue)
        result['search_performed']=False;graph['external_implementation_selection']['regions'][0]['id']='different'
        with self.assertRaisesRegex(ValueError,'changed prescribed'):bridge.verify_manual_assignment(assignment,result,graph,catalogue)

    def test_priority_indices_are_normalized_invocations(self):
        bridge=module('replay-r19-selection')
        selected=[dict(id='parent',source_operators=[0]),dict(id='child',source_operators=[1])]
        variants=[dict(implementation_id='parent'),dict(implementation_id='child')]
        graph=dict(external_mapping=dict(ready_priority=[1,0]),ledger=dict(events=[dict(depends_on=[]),dict(depends_on=[0])]),execution_regions=[
            dict(implementation='parent',source_tasks=[0],member_events=[0],predecessor_regions=[]),
            dict(implementation='child',source_tasks=[1],member_events=[1],predecessor_regions=[0])])
        ranks,deps=bridge.normalized_schedule(graph,selected,variants)
        self.assertEqual(ranks,{'child':0,'parent':1});self.assertEqual(deps['child'],['parent'])
        graph['external_mapping']['ready_priority']=[0,0]
        with self.assertRaises(ValueError):bridge.normalized_schedule(graph,selected,variants)
    def test_conservative_physical_dependency_is_executed(self):
        ready=module('qwen-r19-ready');shim=module('run-r19-selected');order=[]
        shim.bind_prepared(ready,{'a':[],'b':['a']})
        def a(values):order.append('a');return {'a-value':1}
        def b(values):order.append('b');return {'prefill.logits':2}
        operations=[ready.Operation('a',[],['a-value'],'cpu',a,priority=10),ready.Operation('b',[],['prefill.logits'],'cpu',b,priority=-10)]
        prepared=ready.PreparedProgram(operations,[],['prefill.logits'])
        result=ready.ReadyExecutor().run_prepared(prepared,{})
        self.assertEqual(order,['a','b']);self.assertEqual(result['values']['prefill.logits'],2)
    def test_half_entry_edges_bind_both_halves(self):
        ready=module('qwen-r19-ready');shim=module('run-r19-selected')
        shim.bind_prepared(ready,{'before':[],'half':['before']})
        operations=[ready.Operation('before',[],['before-value'],'cpu',lambda v:{'before-value':0}),
            ready.Operation('half.gate',[],['gate'],'cpu',lambda v:{'gate':1}),
            ready.Operation('half.up',[],['up'],'cpu',lambda v:{'up':2}),
            ready.Operation('half.assemble',['gate','up'],['prefill.logits'],'cpu',lambda v:{'prefill.logits':v['gate']+v['up']})]
        prepared=ready.PreparedProgram(operations,[],['prefill.logits'])
        self.assertIn('before',prepared.operations['half.gate'].dependencies)
        self.assertIn('before',prepared.operations['half.up'].dependencies)
        self.assertEqual(ready.ReadyExecutor().run_prepared(prepared,{})['values']['prefill.logits'],3)
    def test_unknown_physical_plan_rejected_before_execution(self):
        ready=module('qwen-r19-ready');shim=module('run-r19-selected')
        shim.bind_prepared(ready,{'missing':[]})
        with self.assertRaises(ValueError):
            ready.PreparedProgram([ready.Operation('other',[],['prefill.logits'],'cpu',lambda v:{'prefill.logits':0})],[],['prefill.logits'])
    def test_aliases_are_retained_and_same_bytes_retarget_rejected(self):
        bridge=module('replay-r19-selection')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'blob-a';b=root/'blob-b'
            a.write_text('same bytes');b.write_text('same bytes')
            first=root/'snapshot-one';second=root/'snapshot-two'
            first.symlink_to(a);second.symlink_to(a)
            files=bridge.Files()
            for alias in (first,second):files.record(dict(file=str(a),requested_file=str(alias),sha256=bridge.sha(a)))
            rows=files.rows();self.assertEqual(len(rows),2)
            self.assertEqual({r['requested_file'] for r in rows},{str(first),str(second)})
            first.unlink();first.symlink_to(b)
            bad=next(r for r in rows if r['requested_file']==str(first))
            with self.assertRaisesRegex(ValueError,'binding changed'):files.record(bad)
            # This exact verifier is called by both runselected and executor.
            with self.assertRaisesRegex(ValueError,'binding changed'):bridge.verification.verify(bad)
            bridge.verification.verify(next(r for r in rows if r['requested_file']==str(second)))

    def test_conflicting_actual_file_identity_rejected(self):
        bridge=module('replay-r19-selection')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'proof';path.write_text('actual')
            files=bridge.Files();files.check(path,bridge.sha(path))
            with self.assertRaises(ValueError):files.check(path,'0'*64)


if __name__=='__main__':unittest.main()
