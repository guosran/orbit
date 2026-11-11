#!/usr/bin/env python3
"""Final-AIE corruption contracts for a supplied checked joint implementation."""
import argparse,json,re,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--compiler',required=True);p.add_argument('--source',required=True);p.add_argument('--target',required=True);p.add_argument('--solution',required=True);p.add_argument('--aie',required=True);a=p.parse_args()
text=Path(a.aie).read_text();mutations={}
def change(label,old,new):
 if old in text:mutations[label]=text.replace(old,new,1)
change('wrong-target','device = 8 : i32','device = 7 : i32')
change('wrong-core-column','col = 0 : i32','col = 3 : i32')
change('wrong-BD-offset','static_offset = 0 : i32','static_offset = 1 : i32')
change('wrong-BD-offset64','static_offset = 0 : i64','static_offset = 1 : i64')
change('wrong-arithmetic','value = 1.000000e+00 : f32','value = 3.000000e+00 : f32')
change('wrong-arithmetic2','value = 2.000000e+00 : f32','value = 3.000000e+00 : f32')
for label,pattern in [('missing-await',r'^.*"aiex.dma_await_task".*\n'),('missing-free',r'^.*"aiex.dma_free_task".*\n'),('missing-start',r'^.*"aiex.dma_start_task".*\n')]:
 m=re.search(pattern,text,re.M)
 if m:mutations[label]=text[:m.start()]+text[m.end():]
m=re.search(r'"aiex.run"\((%\w+), (%\w+)',text)
if m:mutations['swapped-caller-ABI']=text[:m.start()]+text[m.start():].replace(m.group(),f'"aiex.run"({m[2]}, {m[1]}',1)
configs=list(re.finditer(r'^[ \t]*"aiex.configure"\(\)[^\n]*\n[^\n]*\n[ \t]*\}\) : \(\) -> \(\)\n',text,re.M))
if len(configs)>1:
 first,second=configs[:2];mutations['swapped-configurations']=text[:first.start()]+second.group()+text[first.end():second.start()]+first.group()+text[second.end():]
if configs:
 first=configs[0];mutations['duplicate-configuration']=text[:first.start()]+first.group()+text[first.start():]
tiles={m[1]:(int(m[2]),int(m[3])) for m in re.finditer(r'(%\w+) = "aie.tile"\(\) <\{col = (\d+) : i32, row = (\d+) : i32',text)}
for flow in re.finditer(r'"aie.flow"\((%\w+), (%\w+)\)',text):
 for index in [1,2]:
  old=flow[index]
  if old not in tiles:continue
  alternatives=[value for value,coordinate in tiles.items() if coordinate[1]==tiles[old][1] and coordinate[0]!=tiles[old][0]]
  if alternatives:
   changed=flow.group().replace(old,alternatives[0],1);mutations['cross-owner-flow']=text[:flow.start()]+changed+text[flow.end():];break
 if 'cross-owner-flow' in mutations:break
locks={m[1]:m[2] for m in re.finditer(r'(%\w+) = "aie.lock"\((%\w+)\)',text)}
for use in re.finditer(r'"aie.use_lock"\((%\w+)\)',text):
 old=use[1];tile=locks.get(old);alternatives=[value for value,owner in locks.items() if owner in tiles and tile in tiles and tiles[owner]!=tiles[tile]]
 if alternatives:
  mutations['cross-owner-lock']=text[:use.start()]+use.group().replace(old,alternatives[0],1)+text[use.end():];break
# Extra side effects must be rejected before projection can remove anything.
mutations['extra-entry']=text.replace('"builtin.module"() ({','"builtin.module"() ({\n "orbit.unmodeled_side_effect"() : () -> ()',1)
with tempfile.TemporaryDirectory(prefix='joint-aie-contract-') as d:
 root=Path(d);checks=[]
 for name,candidate in [('valid',text),*mutations.items()]:
  file=root/(name+'.mlir');file.write_text(candidate);receipt=root/(name+'.json');cmd=[a.compiler,a.source,'--target-config='+a.target,'--joint-program-plan='+a.solution,'--check-joint-aie='+str(file),'--implementation-output='+str(receipt)]
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=20);passed=(r.returncode==0)==(name=='valid');checks.append(dict(case=name,passed=passed,returncode=r.returncode,diagnostic=r.stderr.strip()));assert passed,(name,r.stdout,r.stderr)
 print(json.dumps(dict(passed=True,checks=checks),indent=2))
