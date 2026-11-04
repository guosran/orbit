#!/usr/bin/env python3
"""Check native physical-port legalization after trusted installed conversion."""
import argparse,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tool',type=Path,required=True);p.add_argument('--air-opt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2]
source=a.output/'source.mlir';source.write_text((root/'test/TaskflowAIR/fixtures/gemm.mlir').read_text().replace('4x8xf32','4x4xf32').replace('8x4xf32','4x4xf32').replace('0 to 8','0 to 4'))
config=a.output/'target.json';config.write_text(json.dumps({'generic_air':{'tile_sizes':[3,3,3]}}))
base=[str(a.tool.resolve()),str(source.resolve()),'--air-distribute-initialization',f'--target-config={config.resolve()}']
def run(command,ok=True,name='command'):
 r=subprocess.run(list(map(str,command)),capture_output=True,text=True)
 (a.output/(name+'.stdout')).write_text(r.stdout);(a.output/(name+'.stderr')).write_text(r.stderr)
 assert (r.returncode==0)==ok,(command,r.returncode,r.stderr)
 return r
canonical=a.output/'canonical.air.mlir';converted=a.output/'converted.air.mlir';legal=a.output/'legal.air.mlir'
run([*base,'--lower-to-air',f'--air-output={canonical}',f'--mapping-output={a.output}/plan.json'],name='lower')
canonical_text=canonical.read_text()
assert 'static-hardware-stage-region' in canonical_text and '\"memref.dealloc\"' not in canonical_text
wrong_lifetime=a.output/'lifetime-mutant.air.mlir';wrong_lifetime.write_text(canonical_text.replace('static-hardware-stage-region','dynamic-shared',1))
run([*base,f'--check-air={wrong_lifetime}'],ok=False,name='reject-canonical-lifetime')
run([a.air_opt,canonical,'--air-dma-to-channel','--canonicalize','--mlir-print-op-generic','-o',converted],name='standard-convert')
assert '"scf.parallel"' not in converted.read_text(), 'standard canonicalization must eliminate generated unit parallel reductions'
run([*base,f'--legalize-air-channels={converted}',f'--air-output={legal}'],name='legalize')
run([*base,f'--check-air-channels={legal}'],name='check')
run([a.air_opt,legal,'-o','/dev/null'],name='installed-verify')
run([a.air_opt,legal,'--air-dependency','--air-hoist-dma-in-accum-pattern','--air-dma-to-channel','-o',a.output/'reentered.air.mlir'],name='installed-reentry')
text=legal.read_text();assert 'air.tile_dma_channel' in text
plan=json.loads((a.output/'plan.json').read_text())
assert '__orbit_fifo_stage' in text and 'orbit.channel_pre_merge_fingerprint' in text
assert len(re.findall(r'"air.channel"\(',text)) == len(plan['channel_allocations'])
# Endpoints are real FIFO messages. Their absence, duplication, reordering or
# cross-storage symbol exchange must fail, including after serialization.
endpoint_lines=[(i,line) for i,line in enumerate(text.splitlines()) if '"air.channel.get"' in line or '"air.channel.put"' in line]
assert endpoint_lines

variants={'port':text.replace('air.tile_dma_channel = 0 : i32','air.tile_dma_channel = 1 : i32',1),'body':text.replace('"arith.mulf"','"arith.subf"',1)}
lines=text.splitlines()
i,line=endpoint_lines[0]
variants['missing_endpoint']='\n'.join(lines[:i]+lines[i+1:])+'\n'
# Duplicate a message with a fresh SSA result name, preserving valid SSA.
match=re.search(r'(%[a-zA-Z0-9_]+) = ',line)
assert match
copy=line.replace(match[1]+' = ','%orbit_duplicate = ',1)
variants['duplicate_endpoint']='\n'.join(lines[:i+1]+[copy]+lines[i+1:])+'\n'
symbols=re.findall(r'chan_name = (@__orbit_fifo_[a-zA-Z0-9_]+)',text)
assert len(set(symbols))>1
other=next(x for x in symbols if x!=symbols[0])
variants['cross_fifo']=text.replace('chan_name = '+symbols[0],'chan_name = '+other,1)
# Swap two adjacent full/tail messages of one storage when both are present.
for x,(i,left) in enumerate(endpoint_lines):
 symbol=re.search(r'chan_name = (@[^, >]+)',left)[1]
 for j,right in endpoint_lines[x+1:]:
  if re.search(r'chan_name = (@[^, >]+)',right)[1]!=symbol:continue
  if ('static_src_sizes = array<i64: 3, 3>' in left and 'static_src_sizes = array<i64: 3, 1>' in right) or ('static_src_sizes = array<i64: 3, 3>' in left and 'static_src_sizes = array<i64: 1, 3>' in right):
   mutated=lines.copy()
   # Exchange lengths while keeping both messages in place and preserving SSA.
   lhs=re.search(r'static_src_sizes = array<i64: [^>]+>',left)[0]
   rhs=re.search(r'static_src_sizes = array<i64: [^>]+>',right)[0]
   mutated[i]=left.replace(lhs,rhs);mutated[j]=right.replace(rhs,lhs)
   variants['stream_order']='\n'.join(mutated)+'\n'
   break
 if 'stream_order' in variants:break
assert 'stream_order' in variants
for i,line in enumerate(lines):
 if '"air.channel.put"' not in line or 'memref<4x4xf32>' not in line:continue
 match=re.search(r'"air.channel.put"\(([^)]*)\).*operandSegmentSizes = array<i32: ([^>]+)>',line)
 if not match:continue
 sizes=[int(x.strip()) for x in match[2].split(',')];operands=[x.strip() for x in match[1].split(',')];memory=operands[sizes[0]+sizes[1]]
 arg=re.fullmatch(r'%arg(\d+)',memory)
 if not arg:continue
 lines[i]=line.replace(memory,'%arg'+str(int(arg[1])+1),1);variants['external_storage']='\n'.join(lines)+'\n';break
assert 'external_storage' in variants,'test must mutate a real host argument operand'
for name,value in variants.items():
 assert value!=text,name
 path=a.output/(name+'.air.mlir');path.write_text(value)
 result=run([*base,f'--check-air-channels={path}'],ok=False,name='reject-'+name)
 if name=='stream_order':assert 'FIFO stream order' in result.stderr,result.stderr
(a.output/'summary.json').write_text(json.dumps({'passed':True,'mutations_rejected':list(variants),'conversion_trust_boundary':'checked canonical DMA AIR -> recorded installed standard passes -> checked native port legalization'},indent=2)+'\n');print(json.dumps({'passed':True,'mutations':len(variants)}))
