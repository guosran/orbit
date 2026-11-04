#!/usr/bin/env python3
"""Independent exp/SiLU boundary probes for explicitly opted-in approximation."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
# Ordinary, near-zero, range-reduction and finite saturation boundary inputs.
values=np.unique(np.concatenate([np.linspace(-80,80,769),np.arange(-120,121)*np.log(2),
    [-103.,-100.,-90.,-88.8,-88.,-87.8,-1e-6,-0.,1e-6,87.8,88.,88.8,90.,100.,103.]]).astype(np.float32))
for kind in ('exp','silu','exp-edge','swiglu'):
    x=values[np.abs(values)<=80] if kind=='exp' else values
    if kind=='exp-edge':
        x=np.array([-104,-103,-100,-88,-87.8,-87.3365,0,80,88,88.7228394,88.8,90,103],dtype=np.float32)
    up=None
    if kind=='swiglu':
        x=np.repeat(np.array([-103,-90,-10,-1,-0.,1,10,90,103],dtype=np.float32),5)
        up=np.tile(np.array([-1e6,-1,0,1,1e6],dtype=np.float32),9)
    count=len(x);typ=f'memref<1x{count}xf32>'
    body = '%v = math.exp %x : f32' if kind in ('exp','exp-edge') else '''%one = arith.constant 1.0 : f32
          %neg = arith.negf %x : f32
          %e = math.exp %neg : f32
          %den = arith.addf %one, %e : f32
          %v = arith.divf %x, %den : f32'''
    extra_arg=f', %B: {typ} {{orbit.role = "input", orbit.name = "B"}}' if up is not None else ''
    reads=f'%A, %B : {typ}, {typ}' if up is not None else f'%A : {typ}'
    types=f'{typ}, {typ}, {typ}' if up is not None else f'{typ}, {typ}'
    block=f'%Ab: {typ}, %Bb: {typ}, %Cb: {typ}' if up is not None else f'%Ab: {typ}, %Cb: {typ}'
    stored='%result' if up is not None else '%v'
    if up is not None:body+=f'\n          %up = affine.load %Bb[%i, %j] : {typ}\n          %result = arith.mulf %v, %up : f32'
    symbol=kind.replace('-','_')
    text=f'''module {{
  func.func @{symbol}_probe(%A: {typ} {{orbit.role = "input", orbit.name = "A"}}{extra_arg}, %C: {typ} {{orbit.role = "output", orbit.name = "C"}}) attributes {{orbit.noalias, llvm.emit_c_interface}} {{
    %done = taskflow.task @probe will_reads({reads}) will_writes(%C : {typ}) [original_read_memrefs({reads}), original_write_memrefs(%C : {typ})] : ({types}) -> {typ} {{
    ^bb0({block}):
      affine.for %i = 0 to 1 {{
        affine.for %j = 0 to {count} {{
          %x = affine.load %Ab[%i, %j] : {typ}
          {body}
          affine.store {stored}, %Cb[%i, %j] : {typ}
        }}
      }}
      taskflow.yield done_writes(%Cb : {typ})
    }}
    return
  }}
}}
'''
    out=a.output/kind;out.mkdir();(out/'source.mlir').write_text(text)
    ref=np.exp(x.astype(np.float64)) if kind in ('exp','exp-edge') else x.astype(np.float64)/(1+np.exp(-x.astype(np.float64)))
    if up is not None:ref*=up.astype(np.float64)
    for seed in [11,23,47]:
        order=np.random.default_rng(seed).permutation(len(x))
        inputs=dict(A=x[order].reshape(1,-1))
        if up is not None:inputs['B']=up[order].reshape(1,-1)
        np.savez(out/f'inputs-{seed}.npz',**inputs)
        with np.errstate(over='ignore'):
            np.savez(out/f'reference-{seed}.npz',C=ref[order].astype(np.float32).reshape(1,-1))
    (out/'probe.json').write_text(json.dumps(dict(kind=kind,minimum=float(x.min()),maximum=float(x.max()),count=len(x),
        exp_contract='finite inputs; standalone accuracy range [-80,80]; SiLU finite sweep [-103,103]; no guarantee for NaN or infinity',
        references='NumPy float64 exp and SiLU, cast final result f32'),indent=2)+'\n')
