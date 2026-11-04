#!/usr/bin/env python3
"""Supplement the unchanged acceptance check with independent error metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('hardware',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--relative-floor',type=float,default=1e-6)
a=p.parse_args()
assert a.relative_floor > 0
records=[]
for seed in sorted(a.hardware.glob('seed-*')):
    reference=seed/'reference.npz'
    if not reference.exists(): continue
    expected=dict(np.load(reference))
    for path in sorted(seed.glob('outputs-*.npz')):
        actual=dict(np.load(path))
        for name,target in expected.items():
            x=actual[name].astype(np.float64)
            y=target.astype(np.float64)
            assert x.shape==y.shape
            finite=np.isfinite(x)&np.isfinite(y)
            delta=np.abs(x[finite]-y[finite])
            relative=finite&(np.abs(y)>=a.relative_floor)
            near_zero=finite&(np.abs(y)<a.relative_floor)
            records.append(dict(seed=seed.name,file=path.name,output=name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                all_finite=bool(finite.all()),
                max_abs=float(delta.max()) if delta.size else None,
                relative_elements=int(relative.sum()),
                max_relative=float(np.max(np.abs(x[relative]-y[relative])/np.abs(y[relative]))) if relative.any() else None,
                near_zero_elements=int(near_zero.sum()),
                near_zero_max_abs=float(np.max(np.abs(x[near_zero]-y[near_zero]))) if near_zero.any() else None,
                normalized_l2=float(np.linalg.norm(x[finite]-y[finite])/max(np.linalg.norm(y[finite]),1e-30)),
                original_1e5_pass=bool(finite.all() and np.allclose(x,y,atol=1e-5,rtol=1e-5))))
assert records
a.output.write_text(json.dumps(dict(relative_metric_domain=f'abs(reference) >= {a.relative_floor}',
    acceptance='Unchanged atol=rtol=1e-5, finite outputs required; relative metric is diagnostic only.',
    records=records),indent=2)+'\n')
