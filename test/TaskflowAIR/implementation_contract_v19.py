#!/usr/bin/env python3
"""Generic actual-AIR mutation controls; no workload/task/storage-name matching.

The C++ backend verifies the entire operation/SSA/token/buffer realization, not
only its orbit metadata. This wrapper generates real IR mutations and records
checker invocations. It never decides production legality or placement.
"""
import argparse,json,re,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--tool',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
 p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 flags=[] if a.config is None else ['--target-config='+str(a.config.resolve())]
 def run(name,args,success):
  command=[str(a.tool.resolve()),str(a.source.resolve()),*flags,*args]
  r=subprocess.run(command,text=True,capture_output=True)
  (a.output/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
  (a.output/(name+'.stdout')).write_text(r.stdout);(a.output/(name+'.stderr')).write_text(r.stderr)
  assert (r.returncode==0)==success,(name,r.returncode,r.stderr[-2000:])
 air=a.output/'canonical.air.mlir'
 run('lower',['--lower-to-air','--air-output='+str(air),'--mapping-output='+str(a.output/'plan.json')],True)
 body=air.read_text();run('canonical',['--check-air='+str(air)],True)
 controls={}
 # Arithmetic changes preserve all declarations and SSA types.
 match=re.search(r'"arith\.(addf|subf|mulf)"',body)
 if match:
  replacement='subf' if match[1]!='subf' else 'addf'
  controls['arithmetic-with-unchanged-metadata']=body[:match.start(1)]+replacement+body[match.end(1):]
 # Remove an actual required transfer, preserving its result and all users.
 # Replace it by a token-forwarding wait rather than causing an undefined SSA.
 dma=re.search(r'(?m)^(\s*)(%[\w]+) = "air.dma_memcpy_nd"\(([^\n]*)\)[^\n]*$',body)
 assert dma,'test requires an actual rectangular DMA'
 operands=[x.strip() for x in dma[3].split(',')]
 token=operands[0]
 replacement=f'{dma[1]}{dma[2]} = "air.wait_all"({token}) : (!air.async.token) -> !air.async.token'
 controls['missing-actual-transfer']=body[:dma.start()]+replacement+body[dma.end():]
 # If a retained logical value crosses the chosen subgraph boundary, verify
 # that removing its real export fails even though the retained marker remains.
 plan=json.loads((a.output/'plan.json').read_text())
 selected=set()
 for group in plan['program_mapping_choices'].get('residency_groups',[]):selected.update(group['retained_storages'])
 exported={entry['storage'] for entry in plan['transfer_summary'] if entry['storage_index'] in selected and entry['direction']=='l1_to_host'}
 for name in sorted(exported):
  for match in re.finditer(r'(?m)^([ \t]*)(%[\w]+) = "air.dma_memcpy_nd"\(([^\n]*)\)[^\n]*$',body):
   if f'orbit.storage = "{name}"' in match[0] and 'orbit.transfer = "l1_to_host"' in match[0]:
    token=match[3].split(',')[0].strip();forward=f'{match[1]}{match[2]} = "air.wait_all"({token}) : (!air.async.token) -> !air.async.token'
    controls['missing-required-subgraph-boundary-export']=body[:match.start()]+forward+body[match.end():];break
  if 'missing-required-subgraph-boundary-export' in controls:break

 duplicate=re.sub(r'^\s*%[\w]+ =',dma[1]+'%v19duplicate =',dma[0])
 controls['duplicate-transfer']=body[:dma.end()]+'\n'+duplicate+body[dma.end():]
 altered=re.sub(r'(static_(?:src|dst)_sizes = array<i64: )([0-9]+)',lambda m:m[1]+str(int(m[2])+1),dma[0],count=1)
 assert altered!=dma[0]
 controls['wrong-transfer-region']=body[:dma.start()]+altered+body[dma.end():]
 # A metadata-only claim is insufficient: unchanged actual operations with a
 # false source-copy join is rejected too, independently of graph identity.
 controls['declaration-only-copy-rebind']=re.sub(r'orbit.source_copies = array<i64:[^>]+>','orbit.source_copies = array<i64: 999999>',body,count=1)
 controls['wrong-local-memory']=body.replace('2 : i32>','1 : i32>')
 wait=next((line for line in body.splitlines() if '"air.wait_all"' in line and 'orbit.computation_event' in line),None)
 if wait:
  altered=re.sub(r'("air.wait_all"\()[^)]*(\))',r'\1\2',wait,count=1)
  altered=re.sub(r' : \([^)]*\) -> \(\)$',' : () -> ()',altered)
  controls['missing-compute-readiness']=body.replace(wait,altered,1)

 # Buffer-root mutation changes actual read operand while keeping all orbit
 # storage/phase markers. Pick equal-typed local views to preserve typed MLIR.
 views=re.findall(r'(%\w+) = "memref.reinterpret_cast"[^\n]* -> ([^\n]+)',body)
 for original,typ in views:
  other=next((v for v,t in views if v!=original and t==typ),None)
  if other and re.search(r'"memref.load"\('+re.escape(original)+r',',body):
   controls['wrong-load-buffer']=re.sub(r'("memref.load"\()'+re.escape(original)+r',',r'\g<1>'+other+',',body,count=1);break
 checks=[]
 for name,mutant in controls.items():
  assert mutant!=body,name
  path=a.output/(name+'.air.mlir');path.write_text(mutant)
  run(name,['--check-air='+str(path)],False);checks.append(name)
 result=dict(status='passed',checks=checks,source=str(a.source.resolve()),device_submissions=0,
  scope='Canonical direct AIR only. Final-AIE coverage requires its separate fact extraction and physical contract; this test does not certify historical noncanonical lowerings.')
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
