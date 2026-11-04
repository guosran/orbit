#!/usr/bin/env python3
"""Verify copy-bound readonly panels, actual body cloning, and target limits."""
import argparse,json,re,subprocess
from pathlib import Path
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool, air_opt

p=argparse.ArgumentParser();p.add_argument('--tool',default=taskflow_tool('air'),type=Path);p.add_argument('--air-opt',default=air_opt(),type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2];source=root/'test/TaskflowAIR/fixtures/gemm.mlir'
def run(args,name,ok=True):
 r=subprocess.run(list(map(str,args)),capture_output=True,text=True)
 (a.output/(name+'.command.json')).write_text(json.dumps(list(map(str,args)),indent=2)+'\n')
 (a.output/(name+'.stdout')).write_text(r.stdout);(a.output/(name+'.stderr')).write_text(r.stderr)
 assert (r.returncode==0)==ok,(args,r.returncode,r.stderr)
 return r
base=[a.tool,source,'--air-distribute-initialization','--tile-m=2','--tile-n=2','--tile-k=2']
plans={};texts={}
for name,extra in [('scalar',[]),('panel',['--residency=readonly-panel','--resident-storage=1','--reuse-loop=0','--panel-loop=1'])]:
 ir=a.output/(name+'.air.mlir');plan=a.output/(name+'.json')
 run([*base,*extra,'--lower-to-air',f'--air-output={ir}',f'--mapping-output={plan}'],name+'-lower')
 run([*base,*extra,f'--check-air={ir}'],name+'-check')
 run([a.air_opt,ir,'--air-dependency','-o',a.output/(name+'.dependency.mlir')],name+'-installed')
 plans[name]=json.loads(plan.read_text());texts[name]=ir.read_text()
 assert 'vector.transfer' not in texts[name]
 assert all(x['kind']=='scalar' for x in plans[name]['task_implementations'])
 assert 'orbit.source_copies' in texts[name] and 'orbit.target_copies' in texts[name]
 assert 'orbit.computation_event' in texts[name]
# Actual installed Peano rejected strict-f32 SIMD multiplication. Keep that
# unavailable choice explicit rather than emitting an executable claim.
unsupported=run([*base,'--implementation=independent-lane-f32','--vector-lanes=8','--lower-to-air',f'--air-output={a.output}/unsupported.air.mlir'],'unsupported-vector',False)
assert 'unavailable' in unsupported.stderr and 'strict f32 vector multiplication' in unsupported.stderr
assert plans['panel']['actual_dma_bytes']<plans['scalar']['actual_dma_bytes']
assert plans['panel']['resident_bytes']>plans['scalar']['resident_bytes']
assert plans['panel']['task_loop_orders'][-1]['loop_order']==[1,0,2]
assert len(plans['panel']['readonly_retentions'])==2
assert 'orbit.readonly_retained_begin' in texts['panel'] and '.readonly-panel' in texts['panel']
# A different actual arithmetic expression follows the same generic mechanism.
changed=a.output/'changed-source.mlir';changed.write_text(source.read_text().replace('arith.mulf','arith.subf'))
changed_ir=a.output/'changed.air.mlir'
run([a.tool,changed,*base[2:],'--lower-to-air',f'--air-output={changed_ir}'],'changed-body')
assert '"arith.subf"' in changed_ir.read_text() and '"arith.mulf"' not in changed_ir.read_text()
# Canonical checker rejects arithmetic and residency changes under the same IDs.
for name,text in {'body':texts['panel'].replace('"arith.mulf"','"arith.subf"',1),'lifetime':texts['panel'].replace('orbit.readonly_retained_end = 12','orbit.readonly_retained_end = 11',1),'copies':re.sub(r'orbit.source_copies = array<i64: [^>]+>', 'orbit.source_copies = array<i64>',texts['panel'],count=1)}.items():
 assert text!=texts['panel'],name
 path=a.output/(name+'-mutant.air.mlir');path.write_text(text)
 run([*base,'--residency=readonly-panel','--resident-storage=1','--reuse-loop=0','--panel-loop=1',f'--check-air={path}'],'reject-'+name,False)
# The original resident adapter binds and independently checks public copy IDs.
resident=a.output/'resident.air.mlir'
resident_source=root/'test/TaskflowAIR/fixtures/add-scale.mlir'
run([a.tool,resident_source,'--lower-to-air',f'--air-output={resident}'],'resident-copy-lower')
resident_text=resident.read_text()
resident_bad=re.sub(r'orbit.source_copies = array<i64: [^>]+>', 'orbit.source_copies = array<i64>',resident_text,count=1)
assert resident_bad!=resident_text
resident_mutant=a.output/'resident-copy-mutant.air.mlir';resident_mutant.write_text(resident_bad)
run([a.tool,resident_source,f'--check-air={resident_mutant}'],'reject-resident-copy',False)
(a.output/'summary.json').write_text(json.dumps({'passed':True,'scalar_dma_bytes':plans['scalar']['actual_dma_bytes'],'panel_dma_bytes':plans['panel']['actual_dma_bytes'],'scalar_local_bytes':plans['scalar']['resident_bytes'],'panel_local_bytes':plans['panel']['resident_bytes'],'hardware_execution':False},indent=2)+'\n');print(json.dumps({'passed':True,'cases':3,'mutants':4}))
