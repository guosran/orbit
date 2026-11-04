#!/usr/bin/env python3
"""Inventory existing validation evidence; does not execute or plan workloads."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();records=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for path in sorted(root.rglob('compile.json')):
 folder=path.parent;c=json.loads(path.read_text());r=dict(directory=str(folder.relative_to(root)),compile=c,compile_record_sha256=sha(path),evidence={})
 for name in ['compile.log','command.json','abi.json','executor-source.py','results.json']:
  f=folder/name
  if f.exists():r['evidence'][name]=dict(sha256=sha(f),bytes=f.stat().st_size)
 for name in ['audit.json','storage-audit.json','invocation.json']:
  f=folder.parent/name
  if f.exists():r[name[:-5]]=json.loads(f.read_text()) if name!='audit.json' else {k:v for k,v in json.loads(f.read_text()).items() if k!='checks'}
 r['seed_failures']=[dict(file=str(f.relative_to(root)),detail=json.loads(f.read_text())) for f in sorted(folder.glob('seed-*/failure.json'))]
 if (folder/'results.json').exists():
  results=json.loads((folder/'results.json').read_text());r['numeric_statuses']=[v.get('status') for v in results] if isinstance(results,list) else [results.get('status')]
 records.append(r)
for path in sorted(root.rglob('invocation.json')):
 folder=path.parent
 if (folder/'hardware/compile.json').exists():continue
 r=dict(directory=str(folder.relative_to(root)),invocation=json.loads(path.read_text()),status='no_compile_record',evidence={})
 for name in ['lower.log','source.mlir','program.air.mlir']:
  f=folder/name
  if f.exists():r['evidence'][name]=dict(sha256=sha(f),bytes=f.stat().st_size)
 records.append(r)
a.output.write_text(json.dumps(dict(scope='Inventory of preserved attempts; numeric success requires independent output audit, compilation alone is not execution success.',records=records),indent=2)+'\n');print(json.dumps(dict(attempts=len(records))))
