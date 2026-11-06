#!/usr/bin/env python3
"""Exercise common multi-task groups, local-copy ABI and shape grouping in AIR."""
import argparse,json,re,subprocess
from pathlib import Path


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tool',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 plain=(Path(__file__).parent/'fixtures/diamond-v18.mlir').read_text();checks=[]
 for name,tm,grouping in [('block512',4,'tile-major'),('block1024',8,'tile-major'),('materialized',4,'shape-major')]:
  d=a.output/name;d.mkdir();attrs=f'orbit.air_tile_sizes=array<i64:{tm},128,1>,orbit.mapping={{residency="tiled-phase-local",data_organization="{'retain-across-users' if grouping=='tile-major' else 'phase-local'}",grouping="{grouping}",storage=-1:i64,reuse_loop=-1:i64,panel_loop=-1:i64'
  if grouping=='tile-major':attrs+=',task_group=[0:i64,1:i64,2:i64],retained_storages=[3:i64]'
  attrs+='}'
  source=d/'source.mlir';source.write_text(plain.replace('module {','module attributes {'+attrs+'} {',1));air=d/'program.air.mlir';plan=d/'plan.json';graph=d/'graph.json'
  def run(label,flags,ok=True):
   cmd=[str(a.tool),str(source),*map(str,flags)];(d/(label+'.command.json')).write_text(json.dumps(cmd,indent=2));r=subprocess.run(cmd,capture_output=True,text=True);(d/(label+'.stdout')).write_text(r.stdout);(d/(label+'.stderr')).write_text(r.stderr);assert (r.returncode==0)==ok,(label,r.returncode,r.stderr[-3000:])
  run('lower',['--lower-to-air',f'--air-output={air}',f'--mapping-output={plan}',f'--program-graph-output={graph}']);data=json.loads(plan.read_text());body=air.read_text()
  run('check',[f'--check-air={air}']);checks.append(name+' canonical group/copy binding')
  assert body.count('"arith.addf"')==3,'all producer/consumer bodies must be emitted'
  if grouping=='tile-major':
   assert len(data['hardware_stages'])==1 and data['actual_dma_bytes']==98304
   assert [x['storage'] for x in data['host_buffers']]==['A','B','C']
   assert all(t['storage']!='X' for t in data['transfer_summary'])
   assert max(s['allocated_bytes'] for s in data['hardware_stages'])==tm*128*16
   assert not any('air.dma_memcpy_nd' in line and 'orbit.storage = "X"' in line for line in body.splitlines())
   assert 'orbit.compressed_phases' in body
   output_positions=[m.start() for m in re.finditer(r'"air.dma_memcpy_nd"[^\n]+orbit.transfer = "l1_to_host"',body)]
   assert len(output_positions)==2 and min(output_positions)>max(m.start() for m in re.finditer('"arith.addf"',body))
   checks.append(name+' output DMA released only after all grouped computations')
   assert 'conservative timing reservations are separate from runtime arbitration' in body
   assert 'serial-by-phase; public dependency edges preserved' not in body
   if name=='block512':
    mutant=body.replace('common execution groups and explicit copy/release dependencies; source compute order preserved; conservative timing reservations are separate from runtime arbitration','serial-by-phase; public dependency edges preserved; no general event scheduler')
    bad=d/'wrong-schedule-declaration.air.mlir';bad.write_text(mutant)
    run('wrong-schedule-declaration-rejected',[f'--check-air={bad}'],False);checks.append('obsolete phase-serial schedule declaration rejected')
   checks.append(name+' shared X without BO/DMA; repeated three-task block; capacity')
   mutant=re.sub(r'orbit.group_members = array<i64:[^>]*>', 'orbit.group_members = array<i64: 999999>',body,count=1);assert mutant!=body
   bad=d/'wrong-group.air.mlir';bad.write_text(mutant);run('wrong-group-rejected',[f'--check-air={bad}'],False);checks.append(name+' group ownership mutation rejected')
  else:
   assert data['actual_dma_bytes']==196608 and len(data['hardware_stages'])==3
   assert [x['storage'] for x in data['host_buffers']]==['A','B','C','X'];checks.append('materialized intermediate host copy retained')
   events=data['program_communication']['events'];anc=[]
   for event in events:
    parents=set(event['depends_on'])
    for parent in list(parents):parents.update(anc[parent])
    anc.append(parents)
   assert 2 not in anc[3] and 1 in anc[3] and 2 in anc[4]
   checks.append('materialized input refills after compute while output slot reuse waits for export')
 # Shape-grouped tails must keep all three bodies per group and exact regions.
 d=a.output/'tails';d.mkdir();source=d/'source.mlir'
 tail=plain.replace('64x128xf32','65x136xf32').replace('to 64 {','to 65 {').replace('to 128 {','to 136 {')
 attrs='orbit.air_tile_sizes=array<i64:4,128,1>,orbit.mapping={residency="tiled-phase-local",data_organization="retain-across-users",grouping="tile-major",storage=-1:i64,reuse_loop=-1:i64,panel_loop=-1:i64,task_group=[0:i64,1:i64,2:i64],retained_storages=[3:i64]}'
 source.write_text(tail.replace('module {','module attributes {'+attrs+'} {',1));air=d/'program.air.mlir';plan=d/'plan.json'
 run('lower',['--lower-to-air',f'--air-output={air}',f'--mapping-output={plan}']);data=json.loads(plan.read_text())
 assert len(data['hardware_stages'])==4 and data['actual_dma_bytes']==65*136*12
 assert air.read_text().count('"arith.addf"')==12
 assert all(t['storage']!='X' for t in data['transfer_summary'])
 run('check',[f'--check-air={air}']);checks.append('four legal tail-shape groups preserve all computed consumers and exact payload')
 (a.output/'summary.json').write_text(json.dumps(dict(status='passed',checks=checks,device_submissions=0),indent=2)+'\n');print(json.dumps(dict(status='passed',checks=len(checks),device_submissions=0)))

if __name__=='__main__':main()
