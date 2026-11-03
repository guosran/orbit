#!/usr/bin/env python3
"""Compare host and device accumulation on a K candidate proposed by ORBIT."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from orbit_npu.runtime import run_plan
from taskflow_common import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--search-root',default=str(ROOT/'build/search-npu-v2'));p.add_argument('--output-dir',required=True)
    p.add_argument('--case',required=True);p.add_argument('--iterations',type=int,default=7);a=p.parse_args()
    root=Path(a.search_root);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    checkpoint=json.loads((root/'search'/f'npu-{a.case}-formal/checkpoint.json').read_text())
    selected=next(x for x in checkpoint['archive'] if x['action'].startswith('tile_k_') and x['parent']==checkpoint['archive'][0]['search_key'])
    plan=selected['program'];write_json(out/'selected-search-candidate.json',selected)
    inputs=dict(np.load(root/'data'/a.case/'inputs-0.npz'));ref=dict(np.load(root/'data'/a.case/'reference-0.npz'))
    rows=[]
    for repeat,order in enumerate([['host','device'],['device','host']]):
      for accumulation in order:
        target=dict(weights='persistent',projection='separate',accumulation=accumulation)
        dest=out/f'round-{repeat}'/accumulation
        if (dest/'result.json').exists():raise FileExistsError(f'refusing old evidence: {dest}')
        result=run_plan(plan,inputs,ref,dest,iterations=a.iterations,target_config=target)
        row=dict(case=a.case,round=repeat,accumulation=accumulation,candidate_id=plan['candidate_id'],action=selected['action'],status=result['status'],median_ms=result.get('median_host_wall_ms'),samples=result.get('host_wall_ms'),dispatches=result.get('device_dispatches_per_graph'),readbacks=result.get('output_readbacks_per_graph'),result=str(dest/'result.json'))
        rows.append(row);write_json(out/'summary.json',rows);print(json.dumps(row),flush=True)
    return 0 if all(x['status']=='passed' for x in rows) else 1
if __name__=='__main__':raise SystemExit(main())
