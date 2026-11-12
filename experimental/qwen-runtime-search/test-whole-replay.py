#!/usr/bin/env python3
"""CPU-only selector safety contracts. No model arrays or NPU execution."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

translate=module(HERE/'whole-replay.py','whole_translate_test')
selector=module(HERE/'selected-decode-helper.py','whole_selector_test')

class SelectorContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        helper=self.root/'fake_helper.py'
        # Only the delegation interface is simulated. These are not numerical
        # or physical-region proofs, and none of these invented ns are prices.
        helper.write_text('''class RegionDecoder:
 def __init__(self,decoder,dirs): self.decoder=decoder; self.dirs=tuple(map(str,dirs)); self.dispatch_events=[]
 def dispatch(self,token,position):
  self.decoder.calls.append((token,position,self.dirs))
  self.dispatch_events.append(dict(position=position,wall_ns=1))
  return (token,position,self.dirs)
''')
        source_files=[dict(file=str(helper),sha256=translate.sha(helper))]
        steps=[]
        for i in range(127):
            bounds=[[0,61],[0,36,61],[0,18,36,61]][i%3]
            steps.append(dict(step=i,position=5+i,partition=bounds,regions=[dict(lo=a,hi=b,implementation=f'T{i}.{a}-{b}',template_directory=str(self.root/f'region-{a}-{b}')) for a,b in zip(bounds,bounds[1:])]))
        self.replay=dict(schema='orbit.qwen-whole-selected-replay.v1',prompt_tokens=5,source_files=source_files,decode_steps=steps)
        self.helper=helper
    def construct(self,replay=None,helper=None):
        value=copy.deepcopy(self.replay if replay is None else replay);value['replay_id']=translate.identity(value)
        file=self.root/'replay.json';file.write_text(json.dumps(value))
        class Base:
            def __init__(self):self.calls=[]
        decoder=Base()
        with patch.dict(os.environ,ORBIT_QWEN_SELECTED_REPLAY=str(file),ORBIT_QWEN_BASE_REGION_HELPER=str(helper or self.helper)):
            result=selector.RegionDecoder(decoder,[])
        return decoder,result
    def test_per_step_nonconsecutive_positions_and_repeated_request(self):
        decoder,result=self.construct()
        for position in [5,7,6,131,5]:
            actual=result.dispatch(12095,position)
            choice=self.replay['decode_steps'][position-5]
            self.assertEqual(actual[2],tuple(r['template_directory'] for r in choice['regions']))
            self.assertEqual(result.dispatch_events[-1]['selected_partition'],choice['partition'])
        self.assertEqual(len(result.variants),3)
        self.assertTrue(all(x.decoder is decoder for x in result.variants.values()))
    def test_position_outside_registered_request_rejected(self):
        _,result=self.construct()
        for position in [4,132]:
            with self.assertRaises(ValueError):result.dispatch(12095,position)
    def test_duplicate_position_rejected(self):
        replay=copy.deepcopy(self.replay);replay['decode_steps'][1]['position']=5
        with self.assertRaises(ValueError):self.construct(replay)
    def test_incomplete_generation_rejected(self):
        replay=copy.deepcopy(self.replay);replay['decode_steps'].pop()
        with self.assertRaises(ValueError):self.construct(replay)
    def test_illegal_partition_rejected(self):
        replay=copy.deepcopy(self.replay);replay['decode_steps'][0]['partition']=[0,18,61]
        with self.assertRaises(ValueError):self.construct(replay)
    def test_nonsequential_step_rejected(self):
        replay=copy.deepcopy(self.replay);replay['decode_steps'][1]['step']=0
        with self.assertRaises(ValueError):self.construct(replay)
    def test_unregistered_or_changed_helper_rejected(self):
        other=self.root/'other.py';other.write_text(self.helper.read_text()+'\n# changed\n')
        with self.assertRaises(ValueError):self.construct(helper=other)
        self.helper.write_text(self.helper.read_text()+'\n# changed\n')
        with self.assertRaises(ValueError):self.construct()
    def test_noncommon_graph_cannot_be_presented_as_selected(self):
        for graph in [{},dict(schema='orbit.program-graph.v1',source_kind='native')]:
            with self.assertRaises(ValueError):translate.translate(graph,{}, {},verify_files=False)

if __name__=='__main__':unittest.main()
