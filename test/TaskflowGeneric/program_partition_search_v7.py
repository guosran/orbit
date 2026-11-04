#!/usr/bin/env python3
"""Exercise whole-program fission in the original native search, not a new tuner."""
import argparse,json,subprocess,hashlib
from pathlib import Path
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool

p=argparse.ArgumentParser();p.add_argument('--tool',type=Path,default=taskflow_tool('air'));p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);tool=a.tool.resolve()
config=out/'config.json';config.write_text(json.dumps({'generic_air':{'theoretical':{'launch_ms':.1,'per_scalar_operation_ms':1e-6,'per_transfer_byte_ms':1e-7}},'native_program_partition_candidates':[8,16],'native_implementation_candidates':[{'kind':'independent-lane-f32','lanes':8}]}))
def run(name,args):
 (out/(name+'.command.json')).write_text(json.dumps(args,indent=2));r=subprocess.run(args,capture_output=True,text=True);(out/(name+'.log')).write_text(r.stdout+r.stderr);assert r.returncode==0,(name,r.stderr)
search=out/'search'
run('search',[str(tool),str(a.source.resolve()),'--search','--backend=air','--air-distribute-initialization','--tile-m=4','--tile-n=8','--tile-k=32','--program-partition=4',f'--target-config={config}',f'--search-output={search}','--max-rounds=1','--max-candidates=8','--beam-width=4'])
r=json.loads((search/'search.json').read_text());accepted=[];rejected=[]
for row in r['archive']:
 d=search/'candidates'/row['candidate_id'];evaluation=json.loads((d/'mapping-result.json').read_text())
 if row['status']!='passed':
  assert evaluation['status']=='unsupported';rejected.append(row['candidate_id']);continue
 plan=evaluation['best_plan'];assert plan['schema']=='orbit.program.partition.v1'
 run('lower-'+row['candidate_id'],[str(tool),str(d/'taskflow.mlir'),f'--target-config={config}',f'--lower-program-to-air={d}/lowered'])
 emitted=json.loads((d/'lowered/manifest.json').read_text());assert emitted['plan_id']==plan['plan_id'];assert emitted['invocations']==plan['invocations'];assert emitted['template_plans']==plan['template_plans']
 accepted.append(dict(candidate_id=row['candidate_id'],parent=row['parent'],action_chain=row['action_chain'],plan_id=plan['plan_id'],partition=plan['partition'],commands=len(plan['invocations']),estimated_ms=plan['estimated_ms'],actual_dma_bytes=plan['actual_dma_bytes'],path=str(d)))
assert len(accepted)==3 and len(rejected)>=1,(accepted,rejected)
assert sorted(x['partition']['chunk'] for x in accepted)==[4,8,16]
assert len(set(x['plan_id'] for x in accepted))==3
(out/'summary.json').write_text(json.dumps(dict(accepted=accepted,rejected=rejected,ranking=r['ranking'],compiler_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),executed=False,cost_scope='Explicit uncalibrated serial sensitivity; host copy/context costs excluded. Hardware watchdog is not modeled by this estimate.'),indent=2))
print('PASS: original search materializes three complete-program command schedules; unavailable target implementation rejected')
