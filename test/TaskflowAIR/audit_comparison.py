#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.directory
cases=json.loads((root/'bound-cases.json').read_text());count=0
assert len(cases)>=2 and len({c['name'] for c in cases})==len(cases)
for case in cases:
 for key,value in case['identity'].items():assert hashlib.sha256(Path(case[key]).read_bytes()).hexdigest()==value
 assert 'orbit.plan' not in Path(case['taskflow']).read_text()
 compile=json.loads((root/case['name']/'compile.json').read_text());assert compile['status']=='compiled' and not compile['bf16_emulation']
 assert hashlib.sha256((root/case['name']/'input.air.mlir').read_bytes()).hexdigest()==compile['source_sha256']==case['identity']['air']
 assert hashlib.sha256(Path(compile['elf_path']).read_bytes()).hexdigest()==compile['elf_sha256']
folders=sorted(root.glob('seed-*'));assert len(folders)>=3
for folder in folders:
 result=json.loads((folder/'result.json').read_text());reference=dict(np.load(folder/'reference.npz'));samples=result['samples']
 for s in samples:
  actual=dict(np.load(folder/f'{s["name"]}-{s["repeat"]}.npz'))
  assert s['actual_submissions']==1 and sum(e['kind']=='actual_xrt_start' for e in s['api_events'])==1
  assert sum(e['kind']=='actual_xrt_wait' for e in s['api_events'])==1
  assert set(actual)==set(reference) and s['wall_ms']>0 and s['warmup']==(s['repeat']<2)
  assert all(e['start_ns']>=0 and e['end_ns']>=e['start_ns'] for e in s['api_events'])
  for name,v in actual.items():
   r=reference[name];assert v.shape==r.shape and np.isfinite(v).all() and np.allclose(v,r,atol=1e-5,rtol=1e-5);count+=1
 assert samples
 for repeat in range(max(s['repeat'] for s in samples)+1):
  order=[s['name'] for s in samples if s['repeat']==repeat];assert order==[c['name'] for c in (cases if repeat%2==0 else list(reversed(cases)))]
 for c in cases:
  hot=[s['wall_ms'] for s in samples if s['name']==c['name'] and not s['warmup']];assert len(hot)>=3;assert float(np.median(hot))==result['medians_ms'][c['name']]
a.output.write_text(json.dumps(dict(status='passed',outputs_checked=count,variants=len(cases),scope='Independent numeric outputs, artifact hashes, observed submissions, alternating order and medians'),indent=2)+'\n')
