#!/usr/bin/env python3
"""CPU-only real existing-model→catalogue→production-score contract regression.

Uses the existing Neura fixture and frozen model files. It neither creates nor
trains a model. Missing model or pre-adapter baseline artifacts fail explicitly.
"""
import argparse, hashlib, json, math, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--tool',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--models',type=Path,default=Path('/home/shiran/project/aie-toy/vendor/orbit-artifact/reference/input0-neighborhood/models'))
p.add_argument('--baseline-score',type=Path,default=ROOT/'test/TaskflowMapping/fixtures/compute-contract-baseline-v8.jsonl')
a=p.parse_args()
fixture=ROOT/'test/multi-cgra/taskflow/joint-scheduling/map-joint-scheduling-all-unit-cache.mlir'
arch=fixture.with_name('mapper-arch-4x4.yaml')
for path in [a.tool,a.baseline_score,fixture,arch]+[a.models/name for name in ['ensemble.json','baseline.json','large-operation.json','ranking.json']]:
 if not path.is_file():p.error('required existing artifact missing (cannot claim pass): '+str(path))
baseline_provenance=ROOT/'test/TaskflowMapping/fixtures/compute-contract-baseline-v8.provenance.json'
if a.baseline_score.resolve()==(ROOT/'test/TaskflowMapping/fixtures/compute-contract-baseline-v8.jsonl').resolve():
 saved=json.loads(baseline_provenance.read_text());expected=saved['golden_sha256']
 if hashlib.sha256(a.baseline_score.read_bytes()).hexdigest()!=expected:p.error('delivered baseline differs from provenance SHA256')
 for name,expected_hash in saved['model_sha256'].items():
  if hashlib.sha256((a.models/name).read_bytes()).hexdigest()!=expected_hash:p.error('model differs from original baseline provenance: '+name)
 for relative,expected_hash in saved['source_inputs'].items():
  if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected_hash:p.error('source fixture differs from original baseline provenance: '+relative)
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def lines(path):return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]
source='git@github.com:guosran/amoeba.git';commit='a57376e7043b1681e64e7169c5a8cb02eb192331'
common=[str(a.tool.resolve()),'--neura-architecture-spec='+str(arch),str(fixture)]
steps=[('enumerate',f'enumerate-analytical-task-candidates{{function=all_unit_mapper output={out}/candidates.jsonl max-cgras-per-task=4 graph-variant-id=identity}}'),('predict',f'predict-analytical-task-cost-catalog{{function=all_unit_mapper space-file={out}/candidates.jsonl ensemble-file={a.models}/ensemble.json checkpoint-dir={a.models} architecture-contract=neura-architecture-v1:amoeba-static-rectangles-up-to-four-cgras-formal architecture-path={arch} source-git-repository={source} source-git-commit={commit} graph-variant-id=identity output={out}/costs.json}}'),('score',f'score-analytical-task-candidates{{function=all_unit_mapper candidates={out}/candidates.jsonl cost-file={out}/costs.json source-git-repository={source} source-git-commit={commit} graph-variant-id=identity output={out}/scores.jsonl top-k=0}}')]
for name,options in steps:
 command=common+['--pass-pipeline=builtin.module(bind-source-iteration-domain,'+options+')','-o',str(out/(name+'.mlir'))]
 (out/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
 with (out/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 (out/(name+'.result.json')).write_text(json.dumps(dict(returncode=result.returncode))+'\n')
 if result.returncode:raise SystemExit(name+' failed; see '+str(out/(name+'.log')))
current=lines(out/'scores.jsonl');baseline=lines(a.baseline_score)
old_rows={r['candidate_id']:r for r in baseline if r['record_type']=='score'}
rows=[r for r in current if r['record_type']=='score'];assert rows and len(rows)==len(old_rows)==8
choices={r['candidate_id']:r for r in lines(out/'candidates.jsonl') if r['record_type']=='candidate'}
catalog=json.loads((out/'costs.json').read_text());catalog_sha=digest(out/'costs.json');checks=[]
for row in rows:
 old=old_rows[row['candidate_id']];assert all('compute_contract' not in t for t in old['task_costs']), 'baseline must be pre-adapter output'
 stripped=json.loads(json.dumps(row))
 for task in stripped['task_costs']:task.pop('compute_contract',None)
 assert stripped==old, 'adapter changed existing task costs/schedule/score'
 assert row['valid']
 for index,task in enumerate(row['task_costs']):
  c=task['compute_contract'];shape=choices[row['candidate_id']]['task_shapes'][index]['shape']
  scheduled=next(s for s in row['schedule_trace']['task_schedule'] if s['task']==task['task'])
  assert c['duration']==scheduled['end_cycle']-scheduled['start_cycle']==max(1,math.ceil(task['predicted_duration']))
  assert c['predicted_ii']==task['predicted_ii'] and c['startup_cycles']==task['startup_cycles'] and c['iterations']==task['trip_count']
  assert c['resource_shape']==[shape['rows'],shape['cols']] and c['resource_count']==shape['cgra_count']
  assert c['resource_count_unit']==c['resource_shape_unit']=='cgra_block'
  assert math.prod(c['resource_shape'])==c['resource_count']
  assert c['implementation_shape']==[task['mapper_tile_rows'],task['mapper_tile_cols']]
  assert c['implementation_shape_unit']=='processing_element'
  assert c['task']==index and c['resource']=='cgra' and c['unit']=='cycles'
  assert c['model']==catalog['namespace'] and c['model_version']==catalog_sha
  assert c['provenance']=='TaskShapeCostCache:'+str(out/'costs.json')
  assert c['implementation']=='amoeba-analytical-mapped-task'
  assert 'production scheduler task occupancy' in c['boundary']
  assert c['supply_conditions_verified'] is False and c['required_data_supply']
  assert c['transfer_accounting']=='excludes_inter_task_transfers'
  assert 'explicit inter-task data movement' in c['excluded_costs']
  assert 'host preparation and submission' in c['excluded_costs']
  assert 'scheduler queue waits' in c['excluded_costs']
  assert not any('inter-task' in term for term in c['included_costs'])
  checks.append(dict(candidate=row['candidate_id'],task=task['task'],duration_cycles=c['duration'],model_catalog_sha256=c['model_version'],resource_shape=c['resource_shape'],resource_count=c['resource_count'],implementation_shape=c['implementation_shape']))
new_footer=next(r for r in current if r['record_type']=='footer');old_footer=next(r for r in baseline if r['record_type']=='footer')
assert new_footer['shortlist']==old_footer['shortlist'] and new_footer['valid_count']==old_footer['valid_count']
(out/'summary.json').write_text(json.dumps(dict(status='passed',cpu_only=True,device_submissions=0,model_training=False,candidates_checked=len(rows),contracts_checked=len(checks),scores_and_shortlist_unchanged=True,supply_verified=False,checks=checks,identities=dict(tool=str(a.tool.resolve()),tool_sha256=digest(a.tool),fixture_sha256=digest(fixture),architecture_sha256=digest(arch),baseline_score=str(a.baseline_score.resolve()),baseline_score_sha256=digest(a.baseline_score),model_files={name:digest(a.models/name) for name in ['ensemble.json','baseline.json','large-operation.json','ranking.json']})),indent=2)+'\n')
print(json.dumps(dict(status='passed',contracts=len(checks),device_submissions=0)))
