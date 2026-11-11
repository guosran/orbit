#!/usr/bin/env python3
"""Actual final-AIE shared-input controls selected from source roles and SSA use.

The plan locates source input identities; actual view/load/buffer/BD/lock facts
select mutations. No workload, task index, storage name or buffer name is fixed.
"""
import argparse, json, re, subprocess
from pathlib import Path


def region_end(text, opening):
 depth, pos, quoted, escaped = 1, opening + 1, False, False
 while depth and pos < len(text):
  ch = text[pos]
  if quoted:
   if escaped: escaped = False
   elif ch == '\\': escaped = True
   elif ch == '"': quoted = False
  elif ch == '"': quoted = True
  elif ch == '{': depth += 1
  elif ch == '}': depth -= 1
  pos += 1
 assert depth == 0
 return pos


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['tool','source','aie','plan','config','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 plan=json.loads(a.plan.read_text());inputs={x['storage'] for x in plan['host_buffers'] if x['role']=='input'}
 body=a.aie.read_text();controls={};witness=None
 for core in re.finditer(r'"aie.core"[^\n]*\(\{',body):
  opening=core.end()-1;end=region_end(body,opening);segment=body[opening+1:end-1]
  views=list(re.finditer(r'(?m)^[ \t]*(%\w+) = "memref.reinterpret_cast"\((%\w+)[^\n]+orbit.phase = (\d+)[^\n]+orbit.storage = "([^"]+)"[^\n]*\n',segment))
  for first in views:
   if first[4] not in inputs or not re.search(r'"memref.load"\('+re.escape(first[1])+r'[,)]',segment):continue
   readers=[v for v in views if v[4]==first[4] and v[2]==first[2] and re.search(r'"memref.load"\('+re.escape(v[1])+r'[,)]',segment)]
   if len({v[3] for v in readers})<2:continue
   last=readers[-1]
   other=next((v for v in views if v[2]!=first[2] and v[0].split(' -> ')[-1]==last[0].split(' -> ')[-1]),None)
   if not other:continue
   # The real incoming DMA acquires a free lock before writing this buffer;
   # the same lock is released by the core only after all actual readers.
   bd=next((m for m in re.finditer(r'(?m)^[ \t]*"aie.use_lock"\((%\w+),[^\n]+action = 2[^\n]*\n[ \t]*"aie.dma_bd"\('+re.escape(first[2])+r'\)[^\n]*\n',body) if re.search(re.escape(m[1])+r' = "aie.lock"[^\n]+init = 1',body)),None)
   if not bd:continue
   release=re.search(r'(?m)^[ \t]*"aie.use_lock"\('+re.escape(bd[1])+r',[^\n]+action = 1[^\n]*\n',segment)
   if not release or release.start()<last.end():continue
   mutated=segment[:last.start(2)]+other[2]+segment[last.end(2):]
   controls['shared-external-input-divergent-buffer']=body[:opening+1]+mutated+body[end-1:]
   mutated=segment[:last.start()]+release[0]+segment[last.start():release.start()]+segment[release.end():]
   controls['shared-external-input-free-before-final-reader']=body[:opening+1]+mutated+body[end-1:]
   witness={'storage':first[4],'actual_shared_buffer':first[2],'actual_reader_pattern_phases':[int(v[3]) for v in readers],'actual_free_lock':bd[1],'first_reader_view':first[1],'last_reader_view':last[1],'divergent_equal_typed_buffer':other[2]}
   break
  if controls:break
 assert controls,'no actual multi-phase shared external input with cyclic free-lock proof found'
 results=[]
 for name,text,expected in [('canonical',body,True)]+[(k,v,False) for k,v in controls.items()]:
  path=a.output/(name+'.mlir');path.write_text(text);receipt=a.output/(name+'.json')
  cmd=[str(a.tool.resolve()),str(a.source.resolve()),'--target-config='+str(a.config.resolve()),'--check-aie='+str(path.resolve()),'--implementation-output='+str(receipt.resolve())]
  r=subprocess.run(cmd,text=True,capture_output=True)
  (a.output/(name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n');(a.output/(name+'.stdout')).write_text(r.stdout);(a.output/(name+'.stderr')).write_text(r.stderr)
  assert (r.returncode==0)==expected,(name,r.returncode,r.stderr)
  results.append({'name':name,'returncode':r.returncode,'expected_accept':expected})
 result={'status':'passed','actual_shared_input_witness':witness,'checks':results,'device_submissions':0}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
