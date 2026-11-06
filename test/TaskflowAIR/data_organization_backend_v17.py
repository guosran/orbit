#!/usr/bin/env python3
"""Native public data-organization choices reach checked AIR copy realizations."""
import argparse,json,re,subprocess
from pathlib import Path


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tool',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);checks=[]
 source=(Path(__file__).parent/'fixtures/fanout-shared-v17.mlir').read_text()
 def run(directory,name,command,ok=True):
  command=list(map(str,command));(directory/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
  r=subprocess.run(command,capture_output=True,text=True)
  (directory/(name+'.stdout')).write_text(r.stdout);(directory/(name+'.stderr')).write_text(r.stderr)
  assert (r.returncode==0)==ok,(name,r.returncode,r.stderr[-3000:])
  return r
 plans={}
 for choice,flags in [('whole-program',['--residency=whole-program']),('tiled-phase-local',['--residency=tiled-phase-local','--tile-m=4','--tile-n=32','--tile-k=1'])]:
  d=a.output/choice;d.mkdir();src=d/'source.mlir';src.write_text(source);air=d/'program.air.mlir';plan=d/'plan.json'
  base=[a.tool,src,*flags]
  run(d,'lower',[*base,'--lower-to-air',f'--air-output={air}',f'--mapping-output={plan}'])
  data=json.loads(plan.read_text());plans[choice]=data;body=air.read_text()
  assert f'orbit.data_organization_realization = "{choice}"' in body
  assert f'orbit.program_graph = "{data["program_graph_id"]}"' in body
  run(d,'canonical-check',[*base,f'--check-air={air}']);checks.append(choice+' canonical same graph lowering')
  if choice=='whole-program':
   assert data['strategy']=='single-herd-resident' and data['actual_dma_bytes']==6144
   assert all(t['storage']!='X' for t in data['transfer_summary'])
   allocs=dict((storage,value) for value,storage in re.findall(r'(%[\w]+) = "memref.alloc"[^\n]+orbit.storage = "([^"]+)"',body))
   assert set(allocs)=={'A','B','C','X'}
   assert len(re.findall(r'"memref.load"\('+re.escape(allocs['X'])+r',',body))==2
   assert all(not (f'orbit.storage = "X"' in line and 'air.dma_memcpy_nd' in line) for line in body.splitlines())
   checks.append('one X allocation supplies two real consumers without X DMA')
   # Redirect one real consumer to an input with the same type: a semantic
   # mismatch must fail canonical source/copy binding, despite valid MLIR types.
   mutant,n=re.subn(r'("memref.load"\()'+re.escape(allocs['X'])+r',',r'\g<1>'+allocs['A']+',',body,count=1);assert n==1
   path=d/'wrong-consumer-copy.air.mlir';path.write_text(mutant)
   run(d,'wrong-consumer-copy-rejected',[*base,f'--check-air={path}'],False);checks.append('wrong consumer copy rejected')
  else:
   assert data['strategy']=='serial-tiled' and data['actual_dma_bytes']==12288
   assert sum(t['bytes'] for t in data['transfer_summary'] if t['storage']=='X')==6144
   checks.append('phase-local X materialization/reload retained in physical plan')
  mutant=body.replace(f'orbit.data_organization_realization = "{choice}"','orbit.data_organization_realization = "unrealized"',1)
  assert mutant!=body;path=d/'wrong-realization.air.mlir';path.write_text(mutant)
  run(d,'wrong-realization-rejected',[*base,f'--check-air={path}'],False);checks.append(choice+' unrealized mapping receipt rejected')
 assert plans['whole-program']['body_fingerprint']==plans['tiled-phase-local']['body_fingerprint']
 assert plans['whole-program']['program_graph_id']!=plans['tiled-phase-local']['program_graph_id'];checks.append('same source different checked copy-plan identity')
 # Explicit whole-program placement must reject insufficient local capacity.
 d=a.output/'capacity';d.mkdir();src=d/'source.mlir';src.write_text(source)
 cfg=d/'config.json';cfg.write_text(json.dumps({'generic_air':{'capacity_bytes':4096}}))
 run(d,'insufficient-capacity-rejected',[a.tool,src,'--residency=whole-program',f'--target-config={cfg}','--lower-to-air',f'--air-output={d/"program.air.mlir"}'],False)
 checks.append('whole-program cannot silently tile after capacity failure')
 # Native tiling/cross-K and tails stay on the existing lowering path.
 d=a.output/'cross-k-tail';d.mkdir();src=d/'source.mlir'
 gemm=(Path(__file__).parent/'fixtures/gemm.mlir').read_text().replace('4x8xf32','5x12xf32').replace('8x4xf32','12x20xf32').replace('4x4xf32','5x20xf32').replace('affine.for %i = 0 to 4','affine.for %i = 0 to 5').replace('affine.for %j = 0 to 4','affine.for %j = 0 to 20').replace('affine.for %k = 0 to 8','affine.for %k = 0 to 12');src.write_text(gemm)
 flags=['--residency=tiled-phase-local','--air-distribute-initialization','--tile-m=2','--tile-n=16','--tile-k=8'];air=d/'program.air.mlir'
 run(d,'lower',[a.tool,src,*flags,'--lower-to-air',f'--air-output={air}',f'--mapping-output={d/"plan.json"}'])
 run(d,'canonical-check',[a.tool,src,*flags,f'--check-air={air}'])
 assert 'orbit.initializer_task' in air.read_text();checks.append('explicit tiled path preserves cross-K init and M/N/K tails')
 (a.output/'summary.json').write_text(json.dumps(dict(status='passed',checks=checks,device_submissions=0),indent=2)+'\n')
 print(json.dumps(dict(status='passed',checks=len(checks),device_submissions=0)))

if __name__=='__main__':main()
