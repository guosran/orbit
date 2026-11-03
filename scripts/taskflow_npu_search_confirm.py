#!/usr/bin/env python3
"""Opposite-order same-data confirmation of the measured ORBIT archive best."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from orbit_npu.runtime import run_plan
from taskflow_common import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--search-root',default=str(ROOT/'build/search-npu-v2'));p.add_argument('--output-dir',required=True)
    p.add_argument('--case',required=True);p.add_argument('--label',default='final-timing');p.add_argument('--iterations',type=int,default=7);a=p.parse_args()
    root=Path(a.search_root);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    checkpoint=json.loads((root/'search'/f'npu-{a.case}-{a.label}/checkpoint.json').read_text())
    candidates=[x for x in checkpoint['archive'] if (x.get('evaluation') or {}).get('status')=='passed' and (x.get('evaluation') or {}).get('score')]
    selected=min(candidates,key=lambda x:x['evaluation']['score']['value'])
    plans=dict(baseline=checkpoint['archive'][0]['program'],best=selected['program'])
    write_json(out/'selected-search-candidate.json',selected)
    inputs=dict(np.load(root/'data'/a.case/'inputs-0.npz'));ref=dict(np.load(root/'data'/a.case/'reference-0.npz'))
    target=json.loads((root/'configs'/f'npu-{a.case}-{a.label}.json').read_text())['npu']
    write_json(out/'target-config.json',target)
    rows=[]
    for repeat,order in enumerate([['baseline','best'],['best','baseline']]):
      for name in order:
        dest=out/f'round-{repeat}'/name
        if (dest/'result.json').exists():raise FileExistsError(f'refusing old evidence: {dest}')
        result=run_plan(plans[name],inputs,ref,dest,iterations=a.iterations,target_config=target)
        row=dict(case=a.case,round=repeat,name=name,candidate_id=plans[name]['candidate_id'],action='identity' if name=='baseline' else selected['action'],status=result['status'],median_ms=result.get('median_host_wall_ms'),samples=result.get('host_wall_ms'),result=str(dest/'result.json'))
        rows.append(row);write_json(out/'summary.json',rows);print(json.dumps(row),flush=True)
    return 0 if all(x['status']=='passed' for x in rows) else 1
if __name__=='__main__':raise SystemExit(main())
