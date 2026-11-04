#!/usr/bin/env python3
"""Independently check all saved outputs and identities in an interleaved run."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.directory
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cases=json.loads((root/'bound-cases.json').read_text());names=[c['name'] for c in cases];assert len(set(names))==len(names)
for case in cases:
 d=root/case['name'];compile=json.loads((d/'compile.json').read_text());assert compile['status']=='compiled'
 assert digest(d/'input.air.mlir')==compile['source_sha256']==case['identity']['air']
 assert digest(d/'program.elf')==compile['elf_sha256']
 for key in ['taskflow','plan','abi']:assert digest(d/(key+Path(case[key]).suffix))==case['identity'][key]
 assert json.loads((d/'plan.json').read_text())['plan_id']==case['plan_id']
 runtime=json.loads((d/'runtime-transfer-audit.json').read_text());storage=json.loads((d/'storage-audit.json').read_text())
 assert runtime['status']==storage['status']=='passed'
 assert digest(d/'native-mapping.json')==case['identity']['mapping']==runtime['plan_sha256']
 assert all(s['expected_scope']=='per-stage-native-transfers' for s in runtime['stages'])
checks=[];metrics=[];hashes=[];timings={name:[] for name in names};seed_timings=[]
seeds=sorted(root.glob('seed-*'));assert len(seeds)>=3
for seed in seeds:
 r=json.loads((seed/'result.json').read_text());assert digest(seed/'inputs.npz')==r['input_sha256'] and digest(seed/'reference.npz')==r['reference_sha256'];hashes.append(r['input_sha256'])
 expected=dict(np.load(seed/'reference.npz'));samples=r['samples'];assert len(samples)%len(cases)==0
 repeats=len(samples)//len(cases);assert repeats>=5
 expected_order=[(c['name'],repeat) for repeat in range(repeats) for c in (cases if repeat%2==0 else list(reversed(cases)))]
 assert [(s['name'],s['repeat']) for s in samples]==expected_order
 for sample in samples:
  assert sample['warmup']==(sample['repeat']<2) and sample['wall_ms']>0
  events=sample['api_events'];assert all(e['end_ns']>=e['start_ns']>=0 for e in events)
  assert sum(e['kind']=='actual_xrt_start' for e in events)==sum(e['kind']=='actual_xrt_wait' for e in events)==sample['actual_submissions']==1
  path=seed/f'{sample["name"]}-{sample["repeat"]}.npz';got=dict(np.load(path));assert set(got)==set(expected)
  for name,x in got.items():
   y=expected[name];assert x.dtype==np.float32 and x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
   assert np.allclose(x,y,atol=1e-5,rtol=1e-5)
   delta=np.abs(x.astype(np.float64)-y.astype(np.float64));normal=np.abs(y)>=1e-6;near=~normal
   checks.append(dict(seed=seed.name,candidate=sample['name'],repeat=sample['repeat'],output=name,max_abs=float(delta.max()),output_sha256=digest(path)))
   metrics.append(dict(seed=seed.name,candidate=sample['name'],repeat=sample['repeat'],output=name,normalized_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(y.astype(np.float64)),1e-30)),max_relative_reference_magnitude_at_least_1e_minus6=float(np.max(delta[normal]/np.abs(y[normal]))) if normal.any() else None,near_zero_max_abs=float(delta[near].max()) if near.any() else None))
  if not sample['warmup']:timings[sample['name']].append(sample['wall_ms'])
 seed_timings.append(dict(seed=seed.name,hot_medians_ms={name:float(np.median([s['wall_ms'] for s in samples if s['name']==name and not s['warmup']])) for name in names}))
assert len(set(hashes))==len(hashes)
a.output.write_text(json.dumps(dict(status='passed',seeds_checked=len(seeds),full_outputs_checked=len(checks),candidate_count=len(cases),acceptance='unchanged FP32 finite outputs and atol=rtol=1e-5',timing_scope='same complete-program host staging/sync/submit/wait/readback/output-copy boundary for each variant; context load, numerical checks and output serialization excluded; two warmups, alternating forward/reverse order; unlocked frequency; no significance claim',pooled_hot_medians_ms={name:float(np.median(values)) for name,values in timings.items()},seed_timings=seed_timings,checks=checks,diagnostic_metrics=metrics),indent=2)+'\n')
