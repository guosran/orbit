#!/usr/bin/env python3
"""Independent generic final-AIE mutations consumed by the production C++ checker.

Uses actual generic MLIR syntax, never workload names or fixed task counts.
"""
import argparse,json,re,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for arg in ['tool','source','aie','output']:p.add_argument('--'+arg,type=Path,required=True)
 p.add_argument('--config',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 body=a.aie.read_text();flags=[] if not a.config else ['--target-config='+str(a.config.resolve())]
 def run(name,text,ok):
  path=a.output/(name+'.mlir');path.write_text(text);receipt=a.output/(name+'.json')
  command=[str(a.tool.resolve()),str(a.source.resolve()),*flags,'--check-aie='+str(path),'--implementation-output='+str(receipt)]
  r=subprocess.run(command,text=True,capture_output=True)
  (a.output/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
  (a.output/(name+'.stdout')).write_text(r.stdout);(a.output/(name+'.stderr')).write_text(r.stderr)
  assert (r.returncode==0)==ok,(name,r.returncode,r.stderr[-2000:])
 run('canonical',body,True);mutations={}
 run_op=re.search(r'(?m)^([ \t]*"aiex.run"\()(%arg\d+), (%arg\d+)([^\n]*\n)',body)
 if run_op:
  mutations['swapped-wrapper-run-arguments']=body[:run_op.start()]+run_op[1]+run_op[3]+', '+run_op[2]+run_op[4]+body[run_op.end():]
  mutations['duplicated-wrapper-run']=body[:run_op.end()]+run_op[0]+body[run_op.end():]

 m=re.search(r'(value = )([-+0-9.eE]+)( : f32)',body)
 if m:mutations['changed-scalar-constant']=body[:m.start(2)]+('9.25' if float(m[2])!=9.25 else '8.25')+body[m.end(2):]
 m=re.search(r'("arith\.(?:addf|mulf|subf)"\()(%\w+), (%\w+)(\))',body)
 if m and m[2]!=m[3]:mutations['changed-operand-same-op-count']=body[:m.start(3)]+m[2]+body[m.end(3):]
 bds=list(re.finditer(r'("aie.dma_bd"\()(%arg\d+)',body))
 if len(bds)>=2 and bds[0][2]!=bds[1][2]:
  first,second=bds[:2];mutations['swapped-runtime-abi-arguments']=body[:first.start(2)]+second[2]+body[first.end(2):second.start(2)]+first[2]+body[second.end(2):]
 m=re.search(r'("aie.tile"\(\) <\{col = \d+ : i32, row = )([2-9]\d*)( : i32)',body)
 if m:mutations['wrong-core-coordinate']=body[:m.start(2)]+str(int(m[2])+1)+body[m.end(2):]
 tiles=re.findall(r'(%\w+) = "aie.tile"',body)
 buffer=re.search(r'("aie.buffer"\()(%\w+)(\))',body)
 if buffer:
  other=next((tile for tile in tiles if tile!=buffer[2]),None)
  if other:mutations['wrong-buffer-owner']=body[:buffer.start(2)]+other+body[buffer.end(2):]
 if '2 : i32>' in body:mutations['wrong-buffer-memory-space']=body.replace('2 : i32>','1 : i32>')
 wrong_layout=re.sub(r'(static_strides = array<i64: )(\d+), 1>',lambda m:m[1]+'1, '+m[2]+'>',body)
 wrong_layout=re.sub(r'strided<\[(\d+), 1\]>',lambda m:'strided<[1, '+m[1]+']>',wrong_layout)
 if wrong_layout!=body:mutations['wrong-actual-view-strides']=wrong_layout

 store=re.search(r'(?m)^[ \t]*"memref.store"[^\n]*\n',body)
 if store:mutations['duplicate-actual-write']=body[:store.end()]+store[0]+body[store.end():]
 runtime_bd=next((m for m in re.finditer(r'(?m)^[ \t]*"aie.dma_bd"\(%arg\d+\)[^\n]*\n',body)),None)
 if runtime_bd:
  mutations['overlapping-runtime-descriptor']=body[:runtime_bd.end()]+runtime_bd[0]+body[runtime_bd.end():]
  changed=re.sub(r'(static_offset = )(\d+)',lambda m:m[1]+str(int(m[2])+1),runtime_bd[0],count=1)
  mutations['wrong-runtime-descriptor-region']=body[:runtime_bd.start()]+changed+body[runtime_bd.end():]

 m=re.search(r'(?m)^\s*"aiex.dma_await_task"[^\n]*\n',body)
 if m:mutations['missing-required-output-await']=body[:m.start()]+body[m.end():]
 lock=re.search(r'("aie.lock"[^\n]+init = )(1)( : i32)',body)
 if lock:mutations['wrong-initial-lock-state']=body[:lock.start(2)]+'0'+body[lock.end(2):]
 amount=re.search(r'("arith.constant"\(\) <\{value = )(1)( : i32)',body)
 if amount:mutations['wrong-lock-unit-count']=body[:amount.start(2)]+'2'+body[amount.end(2):]
 locks=list(re.finditer(r'"aie.lock"[^\n]+lockID = (\d+) : i32',body))
 if len(locks)>1 and locks[0][1]!=locks[1][1]:mutations['duplicate-physical-lock-id']=body[:locks[0].start(1)]+locks[1][1]+body[locks[0].end(1):]

 # Swap actual acquire/release operation lines around a real cyclic BD.
 m=re.search(r'(?m)^(\s*"aie.use_lock"[^\n]+action = 2[^\n]*\n)(\s*"aie.dma_bd"[^\n]*\n)(\s*"aie.use_lock"[^\n]+action = 1[^\n]*\n)',body)
 if m:mutations['dma-lock-order']=body[:m.start()]+m[3]+m[2]+m[1]+body[m.end():]
 m=re.search(r'("aie.next_bd"\(\)\[\^)(\w+)(\])',body)
 if m:
  labels=re.findall(r'^\s*\^(\w+):',body,re.M);other=next(x for x in labels if x!=m[2]);mutations['wrong-cyclic-bd-successor']=body[:m.start(2)]+other+body[m.end(2):]
 # All metadata stays intact while the actual address operand is changed.
 m=re.search(r'("memref.load"\(%\w+, )(%\w+), (%\w+)(\))',body)
 if m:mutations['wrong-actual-load-index']=body[:m.start(2)]+m[3]+body[m.end(2):]
 m=re.search(r'(orbit.phase = )(\d+)( : i64)',body)
 if m:mutations['metadata-only-phase-claim']=body[:m.start(2)]+'999999'+body[m.end(2):]
 # Move the complete first two actual body sections, including their local
 # views, so SSA buffer roots remain defined but computation order changes.
 for core in re.finditer(r'"aie.core"[^\n]*\(\{',body):
  opening=core.end()-1;depth=1;end=opening+1;quoted=False;escape=False
  while depth and end<len(body):
   ch=body[end]
   if quoted:
    if escape:escape=False
    elif ch=='\\':escape=True
    elif ch=='"':quoted=False
   elif ch=='"':quoted=True
   elif ch=='{':depth+=1
   elif ch=='}':depth-=1
   end+=1
  segment=body[opening+1:end-1]
  views=list(re.finditer(r'(?m)^[ \t]*%\w+ = "memref.reinterpret_cast"[^\n]+orbit.phase = (\d+)[^\n]*\n',segment))
  transitions=[]
  for view in views:
   if not transitions or transitions[-1][1]!=view[1]:transitions.append((view.start(),view[1]))
  if len(transitions)<2:continue
  x,y=transitions[0][0],transitions[1][0]
  if len(transitions)>=3:z=transitions[2][0]
  else:
   release=re.search(r'(?m)^[ \t]*"aie.use_lock"[^\n]+action = 1[^\n]*\n',segment[y:])
   if not release:continue
   z=y+release.start()
  mutated=segment[:x]+segment[y:z]+segment[x:y]+segment[z:]
  mutations['actual-body-order']=body[:opening+1]+mutated+body[end-1:];break

 checks=[]
 for name,text in mutations.items():assert text!=body;run(name,text,False);checks.append(name)
 result=dict(status='passed',checks=checks,scope='Actual generic final AIE in declared single-core scalar identity-grid contract; mutations preserve unrelated declaration metadata.',device_submissions=0)
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
