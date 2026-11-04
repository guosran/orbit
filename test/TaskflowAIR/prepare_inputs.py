#!/usr/bin/env python3
"""Independent known-workload fixtures, never used for compiler decisions."""
import argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--case',choices=['gemm','add','scale','add_scale','ffn'],required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--m',type=int,default=4)
p.add_argument('--k',type=int,default=8)
p.add_argument('--n',type=int,default=4)
p.add_argument('--rank-one',action='store_true')
p.add_argument('--seeds',type=int,nargs='+',default=[11,23,47])
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
for seed in a.seeds:
 rng=np.random.default_rng(seed)
 def draw(shape):
  v=rng.uniform(-0.5,0.5,shape).astype(np.float32);v.flat[0]=0;return v
 if a.case=='gemm':
  x=draw((a.m,a.k));w=draw((a.k,a.n));inputs={'A':x,'B':w};ref={'C':(x.astype(np.float64)@w.astype(np.float64)).astype(np.float32)}
 elif a.case in ('add','scale','add_scale'):
  x=draw((a.m,a.n));y=draw((a.m,a.n));inputs={'A':x,'B':y}
  value=x.astype(np.float64)+y.astype(np.float64) if a.case!='scale' else x.astype(np.float64)
  if a.case!='add':value*=2
  ref={'C':value.astype(np.float32)}
 else:
  x=draw((a.m,a.k));gate=draw((a.k,a.n));up=draw((a.k,a.n));down=draw((a.n,a.k))
  inputs={'X':x,'W_gate':gate,'W_up':up,'W_down':down}
  g=x.astype(np.float64)@gate.astype(np.float64);u=x.astype(np.float64)@up.astype(np.float64)
  h=g/(1+np.exp(-g))*u
  ref={'Y':(h@down.astype(np.float64)).astype(np.float32)}
 if a.rank_one:
  if a.case!='add':raise ValueError('rank-one fixture supports add only')
  inputs={k:v.reshape(-1) for k,v in inputs.items()};ref={k:v.reshape(-1) for k,v in ref.items()}
 np.savez(a.output/f'inputs-{seed}.npz',**inputs)
 np.savez(a.output/f'reference-{seed}.npz',**ref)
