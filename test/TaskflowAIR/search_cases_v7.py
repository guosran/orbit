#!/usr/bin/env python3
"""Select a 2x2 experiment from already materialized C++ search candidates."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
rows=json.loads(a.summary.read_text())['accepted'];cases=[]
for n in (8,16):
 for panel in (False,True):
  row=next(r for r in rows if r['tile_sizes']==[8,n,32] and r['panel']==panel and (r['b_transfer_bytes']==32768 if panel else True))
  d=Path(row['path']);plan=json.loads((d/'mapping.json').read_text());facts=json.loads((d/'facts.json').read_text());ss=facts['storages'];ss=[s for s in ss if s['role']!='temporary']+[s for s in ss if s['role']=='temporary']
  name=f'n{n}-'+('panel' if panel else 'local');abi=out/(name+'-abi.json');abi.write_text(json.dumps(dict(buffers=[dict(name=s['id'],shape=s['shape'],dtype=s['dtype'],role=s['role']) for s in ss],outputs=[s['id'] for s in ss if s['role']=='output']),indent=2))
  cases.append(dict(name=name,candidate_id=row['candidate_id'],plan_id=row['plan_id'],taskflow=str(d/'taskflow.mlir'),plan=str(d/'mapping.json'),mapping=str(d/'mapping.json'),air=str(d/'program.air.mlir'),abi=str(abi)))
(out/'cases.json').write_text(json.dumps(cases,indent=2));print(out/'cases.json')
