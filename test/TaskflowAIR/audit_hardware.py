#!/usr/bin/env python3
"""Audit saved direct AIR execution evidence independently of executor results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
root=a.directory;compile=json.loads((root/'compile.json').read_text());abi=json.loads((root/'abi.json').read_text())
assert compile['status']=='compiled'
assert digest(root/'input.air.mlir')==compile['source_sha256']
assert digest(Path(compile['elf_path']))==compile['elf_sha256']
command=json.loads((root/'command.json').read_text())
assert Path(command[0]).name=='aircc' and '--output-format' in command
checks=[]
directories=sorted(root.glob('seed-*'))
assert len(directories)>=3, 'hardware evidence requires at least three independent seeds'
for directory in directories:
 record=json.loads((directory/'result.json').read_text());reference=dict(np.load(directory/'reference.npz'))
 provenance=directory/'input-provenance.json'
 if provenance.exists():
  bound=json.loads(provenance.read_text())
  assert digest(directory/'inputs.npz')==bound['inputs_sha256']
  assert digest(directory/'reference.npz')==bound['reference_sha256']
 assert 0<=record['atol']<=1e-5 and 0<=record['rtol']<=1e-5
 samples=record['samples']
 assert [s['repeat'] for s in samples]==list(range(len(samples)))
 assert len(samples)>=5 and [s['warmup'] for s in samples]==[i<2 for i in range(len(samples))]
 for sample in record['samples']:
  assert sample['wall_ms']>0
  events=sample['api_events'];starts=[e for e in events if e['kind']=='actual_xrt_start'];waits=[e for e in events if e['kind']=='actual_xrt_wait']
  assert len(starts)==len(waits)==1
  assert all(e['end_ns']>=e['start_ns']>=0 for e in events)
  actual=dict(np.load(directory/f'outputs-{sample["repeat"]}.npz'))
  assert set(actual)==set(abi['outputs'])==set(reference)
  for name in actual:
   x,y=actual[name],reference[name]
   assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
   assert np.allclose(x,y,atol=record['atol'],rtol=record['rtol'])
   checks.append(dict(seed=directory.name,repeat=sample['repeat'],output=name,max_abs=float(np.max(np.abs(x.astype(float)-y.astype(float))))))
a.output.write_text(json.dumps(dict(status='passed',seeds_checked=len(directories),outputs_checked=len(checks),checks=checks,source_sha256=compile['source_sha256'],elf_sha256=compile['elf_sha256']),indent=2)+'\n')
