#!/usr/bin/env python3
"""Same-data full-graph target comparisons, and owned-weight lifetime checks."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from orbit_npu.runtime import run_plan,Executor
from taskflow_common import write_json
CONFIGS={
 'baseline':dict(weights='staged',projection='separate'),
 'persistent':dict(weights='persistent',projection='separate'),
 'merge':dict(weights='staged',projection='merge_siblings'),
 'device':dict(weights='persistent',projection='separate',accumulation='device'),
 'combined':dict(weights='persistent',projection='merge_siblings')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--cases',default='gemm,gemv,gemm_relu,ffn');p.add_argument('--seeds',default='0,1,2');p.add_argument('--configs',default='persistent,combined')
    p.add_argument('--variant',default='base')
    p.add_argument('--iterations',type=int,default=3);a=p.parse_args()
    source=Path(a.source_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in a.cases.split(','):
      for seed in map(int,a.seeds.split(',')):
        src=source/case
        plan_file=src/f'{a.variant}.json' if (src/f'{a.variant}.json').exists() else src/a.variant/'plan.json'
        plan=json.loads(plan_file.read_text())
        inputs=dict(np.load(src/f'inputs-{seed}.npz'));ref=dict(np.load(src/f'reference-{seed}.npz'))
        for name in a.configs.split(','):
            dest=out/case/f'seed-{seed}'/name
            if (dest/'result.json').exists():
                raise FileExistsError(f'refusing old evidence: {dest}; choose a new --output-dir')
            r=run_plan(plan,inputs,ref,dest,iterations=a.iterations,target_config=CONFIGS[name])
            row=dict(case=case,seed=seed,config=name,status=r['status'],median_ms=r.get('median_host_wall_ms'),samples=r.get('host_wall_ms'),result=str(dest/'result.json'))
            rows.append(row);write_json(out/'summary.json',rows);print(json.dumps(row),flush=True)
    return 0 if all(x['status']=='passed' for x in rows) else 1
if __name__=='__main__':raise SystemExit(main())
