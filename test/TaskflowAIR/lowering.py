#!/usr/bin/env python3
"""Independent structural/negative AIR checks; no mapping decisions or hardware."""
import argparse,json,re,subprocess
from pathlib import Path
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool, air_opt

p=argparse.ArgumentParser();p.add_argument('--tool',type=Path,default=taskflow_tool('air'));p.add_argument('--output',type=Path,required=True);p.add_argument('--air-opt',type=Path,default=air_opt());a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2];fixture=root/'test/TaskflowAIR/fixtures/add-scale.mlir';checks=[]
def run(args,ok=True):
 r=subprocess.run(list(map(str,args)),capture_output=True,text=True)
 if (r.returncode==0)!=ok:raise AssertionError((args,r.returncode,r.stdout,r.stderr))
 return r
base=[a.tool,fixture,'--generic-map',f'--analysis-output={a.output}/facts.json',f'--mapping-output={a.output}/plan.json']
air=a.output/'program.air.mlir';run([*base,'--lower-to-air',f'--air-output={air}']);text=air.read_text();plan=json.loads((a.output/'plan.json').read_text())
run([a.air_opt,air,'--air-dependency','-o',a.output/'dependency.air.mlir']);checks.append('installed AIR dialect and dependency verification')
run([*base,f'--check-air={air}']);checks.append('independent C++ AIR mapping validation')
pass_air=a.output/'pass.air.mlir'
run([a.tool,fixture,'--air-pass-pipeline',f'--air-output={pass_air}'])
assert pass_air.read_text()==text
run([a.air_opt,pass_air,'--air-dependency','-o',a.output/'pass.dependency.air.mlir'])
run([*base,f'--check-air={pass_air}'])
checks.append('real multithreaded MLIR pass pipeline matches direct lowering and installed AIR validation')
assert 'orbit.plan' not in fixture.read_text();assert text.count('"air.herd"')==1;assert text.count('"memref.alloc"')==4;assert text.count('"air.dma_memcpy_nd"')==3
assert text.count('"arith.addf"')==1 and text.count('"arith.mulf"')==1
assert '"math.exp"' not in text
assert sorted(x['storage'] for x in plan['transfers'])==['A','B','C']
checks.append('two non-matcher tasks, one herd, local intermediate without host transfer')
variants={
 'memory_space':text.replace('2 : i32>', '1 : i32>'),
 'source_dependency':text.replace('orbit.source_dependencies = array<i64: 0>', 'orbit.source_dependencies = array<i64>'),
 'transfer_direction':text.replace('orbit.transfer = "host_to_l1"','orbit.transfer = "l1_to_host"',1),
 'transfer_extent':text.replace('static_dst_sizes = array<i64: 4, 4>','static_dst_sizes = array<i64: 3, 4>',1),
 'transfer_stride':text.replace('static_src_strides = array<i64: 4, 1>','static_src_strides = array<i64: 3, 1>',1),
 'task_missing':text.replace(', orbit.task = "scale"','',1),
 'task_completion':text.replace('orbit.task_completion = "scale"','orbit.task_completion = "add"'),
 'external_buffer':None,
 'herd_extent':text.replace('value = 1 : index', 'value = 2 : index', 1),
 'hierarchy_abi':None,
 'task_token':None,
 'output_dma_token':None,
}
# All external buffers have the same type: swapping an ABI argument is valid
# MLIR and leaves the source/plan identities unchanged, yet is semantically wrong.
lines=text.splitlines();i=next(i for i,s in enumerate(lines) if '"air.dma_memcpy_nd"' in s and 'orbit.storage = "A"' in s)
lines[i]=re.sub(r'%arg(\d+)',lambda m:'%arg'+str(int(m.group(1))+1),lines[i]);variants['external_buffer']='\n'.join(lines)+'\n'
# Keep source/plan identity attributes and AIR validity while breaking the
# actual launch ABI and real synchronization tokens.
lines=text.splitlines();i=next(i for i,s in enumerate(lines) if '"air.launch"' in s)
args=re.findall(r'%arg\d+',lines[i]);assert len(args)>=2
lines[i]=lines[i].replace(args[0], '__swap__',1).replace(args[1],args[0],1).replace('__swap__',args[1]);variants['hierarchy_abi']='\n'.join(lines)+'\n'
lines=text.splitlines();i=next(i for i,s in enumerate(lines) if '"air.wait_all"' in s and 'orbit.task = "scale"' in s)
lines[i]=re.sub(r'"air.wait_all"\(%\d+\)', '"air.wait_all"()', lines[i]).replace('(!air.async.token) -> ()','() -> ()');variants['task_token']='\n'.join(lines)+'\n'
lines=text.splitlines();i=next(i for i,s in enumerate(lines) if '"air.dma_memcpy_nd"' in s and 'orbit.transfer = "l1_to_host"' in s)
lines[i]=re.sub(r'"air.dma_memcpy_nd"\(%\d+, ', '"air.dma_memcpy_nd"(', lines[i]).replace('array<i32: 1, 1, 0, 0, 0, 1, 0, 0, 0>','array<i32: 0, 1, 0, 0, 0, 1, 0, 0, 0>').replace('(!air.async.token, memref','(memref');variants['output_dma_token']='\n'.join(lines)+'\n'
for name,wrong in variants.items():
 assert wrong!=text,name
 path=a.output/(name+'.air.mlir');path.write_text(wrong)
 # AIR independently rejects an L2 allocation placed inside the herd. Other
 # mutations preserve dialect validity and need the native semantic checker.
 parse=run([a.air_opt,path,'-o','/dev/null'],ok=name!='memory_space')
 (a.output/(name+'.air-opt.log')).write_text(parse.stdout+parse.stderr)
 result=run([*base,f'--check-air={path}'],ok=False)
 (a.output/(name+'.log')).write_text(result.stdout+result.stderr)
 checks.append('same-identity semantic AIR rejection: '+name)
# Modify actual arithmetic while keeping all interface names and structure.
changed=fixture.read_text().replace('arith.addf %x, %y','arith.subf %x, %y')
source=a.output/'changed.mlir';source.write_text(changed);out=a.output/'changed.air.mlir'
run([a.tool,source,'--lower-to-air',f'--air-output={out}',f'--analysis-output={a.output}/changed.facts.json',f'--mapping-output={a.output}/changed.plan.json'])
assert '"arith.subf"' in out.read_text() and '"arith.addf"' not in out.read_text();assert json.loads((a.output/'changed.facts.json').read_text())['body_fingerprint']!=json.loads((a.output/'facts.json').read_text())['body_fingerprint'];checks.append('actual body mutation changes analysis identity and AIR arithmetic')
ffn=root/'test/TaskflowAIR/fixtures/ffn.mlir'
exact=a.output/'ffn.exact.air.mlir';approx=a.output/'ffn.approx.air.mlir'
run([a.tool,ffn,'--lower-to-air',f'--air-output={exact}'])
run([a.tool,ffn,'--lower-to-air','--air-approximate-exp',f'--air-output={approx}'])
assert '"math.exp"' in exact.read_text() and 'orbit.math_approximation' not in exact.read_text()
approx_text=approx.read_text()
assert not re.search(r'"math\.(exp|fma|floor)"',approx_text)
assert all(x in approx_text for x in ['orbit.math_approximation','unfused-mul-add','bounded-floor-i32','finite-input-contract'])
run([a.air_opt,approx,'--air-dependency','-o',a.output/'ffn.approx.dependency.air.mlir'])
checks.append('exact exp default preserved; explicit bounded approximation metadata and installed AIR verification')
(a.output/'summary.json').write_text(json.dumps({'passed':True,'checks':checks,'count':len(checks),'air_opt_version':run([a.air_opt,'--version']).stdout},indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(checks)}))
