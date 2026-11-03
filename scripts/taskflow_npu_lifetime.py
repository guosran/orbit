#!/usr/bin/env python3
"""Real NPU BO lifetime, alternating inputs, updates, zero and K reset checks."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from orbit_npu.runtime import Executor,Unsupported
from taskflow_common import compare,reference,make_inputs,write_json

def check(actual,ref):
    c=compare(actual,ref,.015,.06)
    for name,v in c['outputs'].items():
        r=ref[name];rms=float(np.sqrt(np.mean(r*r)));peak=float(np.max(abs(r)))
        v['normalized_rms_error']=v.get('rms_abs',float('inf'))/max(rms,1e-12)
        v['peak_scaled_abs_error']=v.get('max_abs',float('inf'))/max(peak,1e-12)
        v['passed'] &= v['normalized_rms_error']<=.03 and v['peak_scaled_abs_error']<=.08
    c['passed']=all(v['passed'] for v in c['outputs'].values());return c

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--device',action='store_true');a=p.parse_args()
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    import torch
    from filelock import FileLock
    from triton.backends.amd_triton_npu.driver_scope import driver_scope
    from orbit_npu.profiling import install
    install();torch.set_num_threads(4);rows=[]
    with FileLock('/tmp/npu.lock',timeout=600),driver_scope('npu'),torch.inference_mode():
      for case,variant in ([('gemm','k'),('ffn','k')] if a.device else [('gemm','base'),('gemm','k'),('ffn','base')]):
        plan=json.loads((ROOT/'build/backend-decoupling'/case/variant/'plan.json').read_text())
        aa=make_inputs(plan,0);bb=make_inputs(plan,1);orig={k:v.copy() for k,v in aa.items()}
        dest=out/(case+'-'+variant);dest.mkdir(exist_ok=True)
        ex=Executor(plan,aa,dest,dict(weights='persistent',projection='separate' if a.device else 'merge_siblings',accumulation='device' if a.device else 'host'))
        ex.compile()
        def run(label,expected):
            for call in ex.calls.values():call[2].fill_(float('nan'))
            actual=ex.execute();c=check(actual,reference(case,expected));row=dict(case=case,variant=variant,label=label,generation=ex.generation,check=c)
            rows.append(row);np.savez(dest/(label+'.npz'),**actual);write_json(out/'result.json',dict(passed=all(r['check']['passed'] for r in rows),checks=rows));assert c['passed'],row
        run('A-first',orig);run('A-repeat',orig)
        ex.update_inputs(bb);run('B-object',bb)
        ex.update_inputs(aa);run('A-return',orig)
        for name in aa:
            if name not in ('A','X'): aa[name] *= -0.75
        run('external-mutated-snapshot-unchanged',orig)
        ex.update_inputs(aa);run('in-place-update',aa);run('in-place-repeat',aa)
        zeros={k:np.zeros_like(v) for k,v in aa.items()};ex.update_inputs(zeros);run('zero-reset',zeros)
        ex.target_config['projection']='merge_siblings' if a.device else 'separate'
        try: ex.execute()
        except Unsupported: rejected=True
        else: rejected=False
        assert rejected
        rows.append(dict(case=case,variant=variant,label='configuration-mutation-rejected',check=dict(passed=rejected)))
        ex.close()
    write_json(out/'result.json',dict(passed=all(r['check']['passed'] for r in rows),checks=rows));print(json.dumps(dict(passed=True,checks=len(rows))))
if __name__=='__main__':main()
