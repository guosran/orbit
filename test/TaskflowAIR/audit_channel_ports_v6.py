#!/usr/bin/env python3
"""Correlate planned ports with emitted AIE DMA starts, BDs and storage views."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('hardware',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());source=a.hardware/'air_project/aie.input.air.mlir';text=source.read_text();parts=re.split(r'aie\.device\(npu2\) @(\w+) \{',text);devices={parts[i]:parts[i+1] for i in range(1,len(parts),2)};rows=[];checks=[];payloads=[]
for index,stage in enumerate(plan.get('hardware_stages',[])):
 if not stage.get('legalize_channels',False):continue
 name='orbit_segment' if index==0 else 'orbit_segment_'+str(stage['begin_phase']);body=devices[name];storage={}
 tiles={m[1]:(int(m[2]),int(m[3])) for m in re.finditer(r'(%\w+) = aie.tile\((\d+), (\d+)\)',body)};buffers=[]
 for m in re.finditer(r'(%\w+) = aie.buffer\((%\w+)\).*?memref<(\d+(?:x\d+)*)x(f32|bf16|i32),',body):
  if tiles[m[2]][1]<2:continue
  size={'f32':4,'bf16':2,'i32':4}[m[4]]
  for d in m[3].split('x'):size*=int(d)
  buffers.append(dict(buffer=m[1],tile=tiles[m[2]],bytes=size))
 payload=sum(b['bytes'] for b in buffers);assert payload<=plan['capacity_bytes'],'emitted compute payload exceeds policy'
 assert payload<=stage['allocated_bytes'],'installed compiler adds unplanned compute payload'
 payloads.append(dict(stage=index,device=name,buffers=buffers,actual_compute_buffer_bytes=payload,planned_bytes=stage['allocated_bytes'],additional_local_payload_bytes=payload-stage['allocated_bytes']))
 for line in body.splitlines():
  m=re.search(r'memref\.reinterpret_cast (%\w+).*orbit.storage = "([^"]+)"',line)
  if m:
   assert m[1] not in storage or storage[m[1]]==m[2],'ambiguous buffer storage provenance'
   storage[m[1]]=m[2]
 memory=body.split(' = aie.mem(',1)[1].split(' = aie.core(',1)[0];positions=list(re.finditer(r'^    \^(bb\d+):',memory,re.M));blocks={m[1]:memory[m.end():positions[i+1].start() if i+1<len(positions) else len(memory)] for i,m in enumerate(positions)}
 for start in re.finditer(r'aie.dma_start\((S2MM|MM2S), (\d+), \^(bb\d+)',memory):
  block=start[3];visited=set()
  while block not in visited:
   visited.add(block);b=blocks[block];bd=re.search(r'aie.dma_bd\((%\w+).*?len = (\d+)',b)
   if bd:rows.append(dict(stage=index,device=name,direction=start[1],tile_channel=int(start[2]),block=block,buffer=bd[1],storage=storage.get(bd[1]),elements=int(bd[2])))
   nxt=re.search(r'aie.next_bd \^(bb\d+)',b)
   if not nxt:break
   block=nxt[1]
 for allocation in plan['channel_allocations']:
  if allocation['stage']!=index:continue
  actual=[r for r in rows if r['stage']==index and r['storage']==allocation['storage'] and r['direction']==allocation['tile_direction']];assert actual,'cannot correlate allocation to emitted BD'
  assert all(r['tile_channel']==allocation['tile_channel'] for r in actual),'emitted AIE port disagrees with native plan'
  checks.append(dict(stage=index,storage=allocation['storage'],direction=allocation['tile_direction'],planned_port=allocation['tile_channel'],actual_ports=sorted({r['tile_channel'] for r in actual}),templates=len({r['buffer'] for r in actual}),elements=sorted({r['elements'] for r in actual})))
assert checks,'no planned legalizations to audit'
a.output.write_text(json.dumps(dict(status='passed',audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),checks=checks,bd_rows=rows,compute_payloads=payloads,artifact_compile_status=json.loads((a.hardware/'compile.json').read_text())['status'],scope='Emitted AIE IR static DMA channel/BD mapping and compute-buffer declarations (linker placement may remain uncompleted; see compile status) correlated through compiler storage view provenance; not device event tracing or measured traffic. Buffer reuse, runtime scheduling and actual on-wire byte counts are not inferred.'),indent=2)+'\n');print(json.dumps(dict(allocations_checked=len(checks))))
