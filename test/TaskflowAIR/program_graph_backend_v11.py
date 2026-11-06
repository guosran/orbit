#!/usr/bin/env python3
"""Compiler-only common graph→AIR binding and canonical-check negatives."""
import argparse,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tool',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
source=(Path(__file__).parent/'fixtures/gemm-scale.mlir').read_text();checks=[]
def run(d,name,command,ok=True):
 command=list(map(str,command));(d/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n');r=subprocess.run(command,capture_output=True,text=True)
 (d/(name+'.stdout')).write_text(r.stdout);(d/(name+'.stderr')).write_text(r.stderr)
 assert (r.returncode==0)==ok,(name,r.returncode,r.stderr[-3000:])
for name,flags in [('resident',[]),('scalar',['--air-distribute-initialization','--tile-m=2','--tile-n=8','--tile-k=4']),('mixed-panel',['--air-distribute-initialization','--tile-m=2','--tile-n=8','--tile-k=4','--implementation=scalar-product-vector-add-f32','--vector-lanes=8','--data-organization=retain-across-users','--resident-storage=1','--reuse-loop=0','--panel-loop=1'])]:
 d=a.output/name;d.mkdir();text=source
 if name!='resident':
  text=text.replace('8x4xf32','8x16xf32').replace('4x4xf32','4x16xf32').replace('affine.for %j = 0 to 4','affine.for %j = 0 to 16')
 src=d/'source.mlir';src.write_text(text);air=d/'program.air.mlir';plan=d/'plan.json';base=[a.tool,src,*flags]
 run(d,'lower',[*base,'--lower-to-air',f'--air-output={air}',f'--mapping-output={plan}'])
 native=json.loads(plan.read_text());ledger=native['program_communication'];body=air.read_text()
 assert native['program_graph_id'] and native['program_graph']
 assert f'orbit.program_graph = "{native["program_graph_id"]}"' in body
 assert 'orbit.graph_schedule_realization = "common execution groups and explicit copy/release dependencies;' in body
 allocations=[int(x) for x in re.findall(r'orbit.allocation = (\d+) : i64',body)]
 assert allocations and len(allocations)==len(set(allocations))
 for i in allocations:
  allocation=ledger['allocations'][i]
  assert allocation['bytes']>0 and not allocation['key'].startswith('host-')
  assert f'orbit.tile_pool = "{allocation["key"]}"' in body
 assert 'orbit.source_instance' in body and 'orbit.public_dependencies' in body
 assert 'orbit.transfer_event' in body and 'orbit.source_copies' in body and 'orbit.target_copies' in body
 run(d,'canonical-check',[*base,f'--check-air={air}']);checks.append(name+' common source/alloc/event/copy binding')
 for label,pattern,replacement in [('allocation',r'orbit.allocation = \d+ : i64','orbit.allocation = 999999 : i64'),('instance',r'orbit.source_instance = \d+ : i64','orbit.source_instance = 999999 : i64'),('transfer',r'orbit.transfer_event = \d+ : i64','orbit.transfer_event = 999999 : i64'),('graph-id',r'orbit.program_graph = "[0-9a-f]+"','orbit.program_graph = "wrong"')]:
  mutant,count=re.subn(pattern,replacement,body,count=1);assert count
  path=d/(label+'-mutant.air.mlir');path.write_text(mutant);run(d,label+'-rejected',[*base,f'--check-air={path}'],False);checks.append(name+' '+label+' mutation rejected')
(a.output/'summary.json').write_text(json.dumps(dict(passed=True,checks=checks,hardware_executed=False),indent=2)+'\n');print(json.dumps(dict(passed=True,checks=len(checks))))
