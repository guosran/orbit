#!/usr/bin/env python3
"""Read final AIR/AIE/linker artifacts; never chooses mapping or placement."""
import argparse
import hashlib
import json
from pathlib import Path
import re
p=argparse.ArgumentParser(description=__doc__);p.add_argument('hardware',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path);a=p.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=a.hardware;compile=json.loads((r/'compile.json').read_text());assert compile['status']=='compiled'
project=r/'air_project';stages=[]
for script in sorted(project.glob('ldScripts_*.ld.script')):
 text=script.read_text();buffers=[]
 for match in re.finditer(r'\. = (0x[0-9a-fA-F]+);\n(\w+) = \.;\n\. \+= (0x[0-9a-fA-F]+);',text):
  if match[2]=='_sp_start_value_DM_stack':continue
  buffers.append(dict(name=match[2],address=int(match[1],16),bytes=int(match[3],16)))
 stack=re.search(r'\. \+= (0x[0-9a-fA-F]+); /\* stack \*/',text);assert stack
 stack_address=re.search(r'\. = (0x[0-9a-fA-F]+);\n_sp_start_value_DM_stack = \.;',text);assert stack_address
 stack_start=int(stack_address[1],16);stack_bytes=int(stack[1],16)
 for i,b in enumerate(buffers):
  assert b['address']+b['bytes']<=stack_start or stack_start+stack_bytes<=b['address'],'buffer overlaps reserved stack'
  for c in buffers[:i]:assert b['address']+b['bytes']<=c['address'] or c['address']+c['bytes']<=b['address'],'overlap in same hardware stage'
 stages.append(dict(linker_script=script.name,runtime_reset='_reset_' in script.name,sha256=sha(script),stack_address=stack_start,stack_reserved_bytes=stack_bytes,buffers=buffers,buffer_payload_bytes=sum(b['bytes'] for b in buffers)))
assert stages
abi=json.loads((r/'abi.json').read_text());host=[]
for b in abi['buffers']:
 count=1
 for n in b['shape']:count*=n
 host.append(dict(name=b['name'],role=b.get('role','unspecified'),bytes=count*{'f32':4,'bf16':2,'i32':4}[b['dtype']]))
result=dict(status='passed',elf_sha256=compile['elf_sha256'],stack_reserved_bytes=compile['stack_size_bytes'],stages=stages,max_stage_payload_bytes=max(s['buffer_payload_bytes'] for s in stages),host_abi_buffers=host,scope='Final installed aiecc linker script buffer and reserved stack addresses; stage payloads need not coexist. BO bytes are system memory, not persistent AIE SRAM. Runtime/data/alignment consume additional memory; successful compiler allocation/link verification remains authoritative.')
result['compute_stage_count']=sum(not s['runtime_reset'] for s in stages)
result['runtime_reset_stage_count']=sum(s['runtime_reset'] for s in stages)
plan_path=a.plan or r.parent/'plan.json';plan=json.loads(plan_path.read_text()) if plan_path.exists() else {}
budget=plan.get('capacity_bytes',24576);errors=[]
for stage in stages:
 if not stage['runtime_reset'] and stage['buffer_payload_bytes']>budget:errors.append('compute payload exceeds native capacity policy')
for index,planned in enumerate(plan.get('hardware_stages',[])):
 name='orbit_segment' if index==0 else 'orbit_segment_'+str(planned['begin_phase'])
 actual=[s for s in stages if not s['runtime_reset'] and s['linker_script'].startswith('ldScripts_'+name+'_core_')]
 if len(actual)!=1:
  errors.append('cannot correlate one native stage with exactly one emitted compute core: '+name);continue
 stage=actual[0];stage['planned_payload_bytes']=planned['allocated_bytes'];stage['extra_local_payload_bytes']=stage['buffer_payload_bytes']-planned['allocated_bytes']
 if stage['extra_local_payload_bytes']>0:errors.append('installed compiler adds unplanned local payload: '+name)
result['capacity_bytes']=budget;result['errors']=errors;result['status']='failed' if errors else 'passed'
result['audit_source_sha256']=sha(Path(__file__))
if plan_path.exists():result['plan_sha256']=sha(plan_path)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(stages=len(stages),max_stage_payload_bytes=result['max_stage_payload_bytes'])))
if errors:raise SystemExit('; '.join(errors))
