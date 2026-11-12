#!/usr/bin/env python3
"""Per-step dispatch of certified wave partitions over one unchanged decoder.

This is an own adapter loaded through the frozen runner's helper interface. It
never changes NumPy, XRT, vendor source files or original decoder arithmetic.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RegionDecoder:
    def __init__(self,decoder,unused_template_dirs):
        replay=json.loads(Path(os.environ['ORBIT_QWEN_SELECTED_REPLAY']).read_text())
        expected=hashlib.sha256(canonical({k:v for k,v in replay.items() if k!='replay_id'})).hexdigest()
        if replay['schema']!='orbit.qwen-whole-selected-replay.v1' or replay['replay_id']!=expected:raise ValueError('Invalid complete replay identity')
        for record in replay['source_files']:
            if sha(record['file'])!=record['sha256']:raise ValueError('Actual registered source/artifact changed')
        helper=Path(os.environ['ORBIT_QWEN_BASE_REGION_HELPER']).resolve(strict=True)
        if sha(helper) not in {r['sha256'] for r in replay['source_files']}:raise ValueError('Base decoder helper is outside registered implementation source proofs')
        spec=importlib.util.spec_from_file_location('orbit_frozen_region_helper',helper);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
        self.decoder=decoder;self.dispatch_events=[];self.variants={};self.steps={}
        if len(replay['decode_steps'])!=127:raise ValueError('Complete 127-step request required')
        for step,choice in enumerate(replay['decode_steps']):
            if choice['step']!=step or choice['position']!=replay['prompt_tokens']+step:raise ValueError('Decode step/position differs from source request')
            bounds=[0]+[r['hi'] for r in choice['regions']]
            if bounds not in ([0,61],[0,36,61],[0,18,36,61]) or choice['partition']!=bounds or [r['lo'] for r in choice['regions']]!=bounds[:-1]:raise ValueError('Unregistered or noncontiguous wave partition')
            key=tuple((r['lo'],r['hi'],r['template_directory']) for r in choice['regions'])
            if key not in self.variants:self.variants[key]=base.RegionDecoder(decoder,[Path(r['template_directory']) for r in choice['regions']])
            position=choice['position']
            if position in self.steps:raise ValueError('Duplicate selected decode position')
            self.steps[position]=(self.variants[key],choice)
        if len(self.steps)!=127:raise ValueError('Selected source requires all 127 decode steps')
    def __getattr__(self,name):return getattr(self.decoder,name)
    def dispatch(self,token,position):
        if position not in self.steps:raise ValueError('Decode position outside the selected source graph')
        variant,choice=self.steps[position]
        result=variant.dispatch(token,position)
        phase=dict(variant.dispatch_events[-1]);phase.update(selected_step=choice['step'],selected_partition=choice['partition'],selected_implementations=[r['implementation'] for r in choice['regions']])
        self.dispatch_events.append(phase)
        return result
