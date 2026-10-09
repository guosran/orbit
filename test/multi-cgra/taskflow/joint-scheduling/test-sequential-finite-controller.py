#!/usr/bin/env python3
"""Checker corruption tests only; no fabricated performance results."""
import copy,importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('checks',Path(__file__).with_name('check-sequential-finite-controller.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Mutations(unittest.TestCase):
 def setUp(self):
  self.s={'decision_flow':'sequential','production_scheduler_calls':2,'max_candidates':4,'phase_a_budget':2,'phase_b_budget':2,'phase_a_anchor':'a'}
  self.t=[{'candidate_id':'a','key':'one','phase':'A','cumulative_evaluations':1,'action_history':{'actions':[],'fissionActions':[]},'shapes':[{'task':'Task_0','rows':1,'cols':1}]},{'candidate_id':'b','key':'two','phase':'B','cumulative_evaluations':2,'action_history':{'actions':[],'fissionActions':[]},'shapes':[{'task':'Task_0','rows':1,'cols':2}]}]
  self.d={'graph_actions':[],'variable_tasks':['Task_0'],'fixed_shapes':{},'shapes':[[1,1],[1,2],[2,2]]}
 def test_valid(self):m.validate(self.s,self.t,self.d)
 def test_repeated_debit(self):
  self.t[1]['key']='one'
  with self.assertRaises(AssertionError):m.validate(self.s,self.t,self.d)
 def test_semantic_repeat_distinct_keys(self):
  self.t[1]['shapes']=copy.deepcopy(self.t[0]['shapes'])
  with self.assertRaises(AssertionError):m.validate(self.s,self.t,self.d)
 def test_frozen_graph_change(self):
  self.t[1]['action_history']['actions']=[{'family':'fusion'}]
  with self.assertRaises(AssertionError):m.validate(self.s,self.t,self.d)
 def test_phase_a_resource_change(self):
  self.t[0]['shapes'][0]['cols']=2
  with self.assertRaises(AssertionError):m.validate(self.s,self.t,self.d)
 def test_alphabet_escape(self):
  self.t[1]['shapes'][0]['cols']=4
  with self.assertRaises(AssertionError):m.validate(self.s,self.t,self.d)
 def test_fixed_task_mutation(self):
  self.d['fixed_shapes']={'Task_0':[1,1]}
  with self.assertRaises(AssertionError):m.validate(self.s,self.t,self.d)
if __name__=='__main__':unittest.main()
