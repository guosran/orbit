#!/usr/bin/env python3
"""Classify saved exp edge outputs independently, including expected infinities."""
import argparse
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('hardware',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def category(v):
 if np.isnan(v):return 'nan'
 if np.isinf(v):return 'infinity'
 if v==0:return 'zero'
 return 'subnormal' if abs(v)<np.finfo(np.float32).tiny else 'finite_normal'
rows=[];maximum=0.;sets=0
for seed in sorted(a.hardware.glob('seed-*')):
 x=np.load(seed/'inputs.npz')['A'].reshape(-1);reference=np.load(seed/'reference.npz')['C'].reshape(-1)
 for path in sorted(seed.glob('outputs-*.npz')):
  actual=np.load(path)['C'].reshape(-1);sets+=1
  for v,got,expected in zip(x,actual,reference):
   normal=np.isfinite(expected) and abs(expected)>=np.finfo(np.float32).tiny
   error=float(abs(float(got)-float(expected))/abs(float(expected))) if normal else None
   if error is not None:maximum=max(maximum,error)
   rows.append(dict(seed=seed.name,file=path.name,input=float(v),actual=float(got) if np.isfinite(got) else str(got),reference=float(expected) if np.isfinite(expected) else str(expected),actual_class=category(got),reference_class=category(expected),relative_error_normal=error))
a.output.write_text(json.dumps(dict(status='diagnostic_only',outputs_checked=sets,maximum_relative_error_finite_normal=maximum,contract='Records overflow and underflow behavior outside ordinary [-80,80] exp accuracy contract; does not relax finite checks or tolerance',rows=rows),indent=2)+'\n')
print(json.dumps(dict(outputs=sets,max_relative_normal=maximum)))
