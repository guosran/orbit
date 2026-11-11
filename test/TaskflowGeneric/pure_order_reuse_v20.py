#!/usr/bin/env python3
"""Source-bound noncontiguous contraction and actual AIR pure-order regression."""
from pathlib import Path
import argparse,json,subprocess,hashlib
args=argparse.ArgumentParser(description=__doc__)
for key in ('tool','air-tool','output'):args.add_argument('--'+key,type=Path,required=True)
args.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'TaskflowAIR/fixtures/reuse-pure-order-v20.mlir')
args.add_argument('--config',type=Path)
a=args.parse_args();r=a.output.resolve();r.mkdir(parents=True,exist_ok=False)
core=a.tool.resolve();air=a.air_tool.resolve()
c=json.loads(a.config.read_text()) if a.config else {}
c.setdefault('native_auto_candidates',{}).update({'mode':'joint','algorithm':'data-reuse-joint','max_joint_proposals':32,'max_joint_tiles':4,'max_residency_groups':4})
# Without an estimate, explore explicitly unpriced parents instead of inventing
# a score. This keeps the standalone default and strict replay meaningful.
if c.get('generic_air',{}).get('cost_model',{}).get('mode')!='estimate':
 c['native_auto_candidates']['exploration_slots']=2
(r/'config.json').write_text(json.dumps(c,indent=2))
source=a.source.read_text();(r/'source.mlir').write_text(source)
attrs='module attributes {orbit.air_tile_sizes = array<i64: 6, 136, 1>, orbit.mapping = {data_organization = "retain-across-users", grouping = "tile-major", panel_loop = -1 : i64, residency = "tiled-phase-local", residency_groups = [{retained_storages = [0, 4], tasks = [0, 2]}], reuse_loop = -1 : i64, storage = -1 : i64}} {'
(r/'grouped.mlir').write_text(source.replace('module {',attrs,1))
(r/'invalid-internal-order.mlir').write_text((r/'grouped.mlir').read_text().replace('tasks = [0, 2]','tasks = [0, 1, 2]'))
records=[]
def run(name,args,expect=0):
 command=list(map(str,args));(r/(name+'.command.json')).write_text(json.dumps(command,indent=2));p=subprocess.run(command,capture_output=True,text=True);(r/(name+'.stdout')).write_text(p.stdout);(r/(name+'.stderr')).write_text(p.stderr);records.append({'name':name,'exit_code':p.returncode});assert (p.returncode==0)==(expect==0),(name,p.stderr);return p
run('analysis-plan',[core,r/'grouped.mlir','--generic-map','--target-config='+str(r/'config.json'),'--analysis-output='+str(r/'analysis.json'),'--mapping-output='+str(r/'plan.json'),'--program-graph-output='+str(r/'graph.json')])
run('lower-air',[air,r/'grouped.mlir','--lower-to-air','--air-output='+str(r/'program.air.mlir'),'--target-config='+str(r/'config.json'),'--mapping-output='+str(r/'air-plan.json')])
run('actual-air',[air,r/'grouped.mlir','--generic-map','--target-config='+str(r/'config.json'),'--check-air='+str(r/'program.air.mlir'),'--implementation-output='+str(r/'actual-air-receipt.json'),'--implementation-facts-output='+str(r/'actual-air-facts.json')])
negative=run('internal-pure-order-reject',[core,r/'invalid-internal-order.mlir','--generic-map','--target-config='+str(r/'config.json')],1)
assert 'whole-task pure-order barrier' in negative.stderr
run('discovery',[core,r/'source.mlir','--search','--backend=air','--tile-m=6','--tile-n=136','--tile-k=1','--max-rounds=1','--max-candidates=64','--beam-width=2','--target-config='+str(r/'config.json'),'--search-output='+str(r/'search')])
a=json.load(open(r/'analysis.json'));p=json.load(open(r/'plan.json'));ap=json.load(open(r/'air-plan.json'))
assert ap==p
assert a['tasks'][2]['dependencies']==['write_X','write_V']
assert all(x['storage']!='V' for x in a['tasks'][2]['accesses'])
assert [t['index'] for t in p['tasks']]==[1,0,2,3,4,5]
assert p['residency_scopes'][0]['tasks']==[0,2]
assert json.load(open(r/'actual-air-receipt.json'))['valid']
found=[]
for candidate in (r/'search/candidates').iterdir():
 f=candidate/'mapping-result.json'
 if not f.exists():continue
 doc=json.load(open(f));plan=doc.get('best_plan',doc)
 if any(s['tasks']==[0,2] for s in plan.get('residency_scopes',[])):
  assert [t['index'] for t in plan['tasks']][:3]==[1,0,2]
  found.append(candidate.name)
assert found
out={'status':'PASS','hardware_executed':False,'source_order':[0,1,2,3,4,5],'contracted_execution_order':[t['index'] for t in p['tasks']],'pure_order_edge':{'producer':'write_V','consumer':'write_Y','actual_read_of_V':False},'selected_group':[0,2],'discovered_candidate_ids':found,'negative_reason':negative.stderr.strip(),'complete_core_air_plans_equal':True,'actual_air_checker_valid':True,'commands':records,'tools':{str(t):hashlib.sha256(t.read_bytes()).hexdigest() for t in (core,air)}}
(r/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps({'status':'PASS','discovered':len(found)}))
