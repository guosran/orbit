#!/usr/bin/env python3
"""Search protocol adversarial tests. Synthetic evaluator scores are NOT performance evidence."""
import argparse
import json
import pathlib
import subprocess
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'Support'))
from structured_fixtures import source_plan

EVALUATOR = '''#!/usr/bin/env python3
import json,sys,pathlib
request=json.loads(pathlib.Path(sys.argv[1]).read_text())
p=json.loads(pathlib.Path(request['plan_path']).read_text())
c=request['target_config']; mode=c.get('mode','normal')
trans=p['transforms']; s=str(trans)
r=dict(status='passed',candidate_id=request['candidate_id'],source_id=request['source_id'],score=dict(value=100.0,unit='ms'),evidence_type='synthetic protocol fixture, never hardware')
if trans:
 r['score']['value']=len(p['tasks'])+sum(len(t['operations']) for t in p['tasks'])/100
 if mode in ('compile_failed','numerical_failed','unsupported'): r['status']=mode
 if mode=='unscored': r.pop('score')
 if mode=='wrong_id': r['candidate_id']='other'
 if mode=='wrong_unit': r['score']['unit']='cycles'
 if mode=='zero': r['score']['value']=0
 if mode=='negative': r['score']['value']=-1
pathlib.Path(sys.argv[2]).write_text(json.dumps(r))
if mode=='interrupt':
 marker=pathlib.Path(sys.argv[1]).parents[2]/'interrupt.once'
 cp=pathlib.Path(sys.argv[1]).parents[2]/'checkpoint.json'
 if not marker.exists() and cp.exists() and json.loads(cp.read_text())['evaluations']==2:
  import os,signal
  marker.write_text('interrupted after two checkpointed evaluations')
  os.kill(os.getppid(),signal.SIGKILL)
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--tool',type=pathlib.Path,default=taskflow_tool('core'));p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
 evaluator=root/'fixture_evaluator.py';evaluator.write_text(EVALUATOR);evaluator.chmod(0o755)
 source=root/'source.json';source.write_text(json.dumps(source_plan('gemm_relu')))
 checks=[]
 def run(label, mode='normal', budget=12, resume=False, good=True):
  d=root/label;d.mkdir(exist_ok=True);config=d/'config.json'
  if not resume: config.write_text(json.dumps(dict(mode=mode)))
  cmd=[str(a.tool.resolve()),str(source),'--search','--backend=npu',f'--evaluator={evaluator}',f'--target-config={config}',f'--search-output={d}','--max-rounds=2',f'--max-candidates={budget}','--beam-width=1']
  if resume:cmd.append('--resume')
  r=subprocess.run(cmd,text=True,capture_output=True);(d/('resume.log' if resume else 'run.log')).write_text(r.stdout+r.stderr)
  assert (r.returncode==0)==good,(label,r.stderr)
  return json.loads((d/'search.json').read_text()) if good else None
 normal=run('normal')
 assert len(normal['selections'])==2
 selected=normal['selections'][0]['selected'][0]['search_key']
 assert all(r['parent']==selected for r in normal['archive'] if r['round']==2)
 assert selected!=normal['archive'][0]['search_key']
 checks.append('two rounds: lowest valid first-round score determines second-round parent')
 resumed=run('normal',resume=True)
 assert resumed['archive']==normal['archive'] and resumed['evaluations']==normal['evaluations']
 checks.append('completed resume reuses exact evaluations')
 run('interrupted','interrupt',good=False)
 recovered=run('interrupted',resume=True)
 assert [(r['search_key'],r['score']) for r in recovered['archive']]==[(r['search_key'],r['score']) for r in normal['archive']]
 assert recovered['ranking']==normal['ranking']
 checks.append('actual process interruption mid-round resumes identical archive and ranking')
 for mode in ['compile_failed','numerical_failed','unsupported','unscored','wrong_id','wrong_unit','zero','negative']:
  result=run(mode,mode)
  assert len(result['ranking'])==1,mode
  assert all(r['score'] is None for r in result['archive'][1:]),mode
  checks.append(mode+' never ranks')
 result=run('budget',budget=2)
 assert result['evaluations']==2 and any(r['status']=='budget_unmeasured' for r in result['archive'])
 assert all(r['search_key'] not in result['ranking'] for r in result['archive'] if r['status']=='budget_unmeasured')
 checks.append('budget-unmeasured candidates retained and excluded')
 config=root/'normal/config.json';config.write_text(json.dumps(dict(mode='changed')))
 run('normal',resume=True,good=False);checks.append('changed target configuration invalidates resume')
 config.write_text(json.dumps(dict(mode='normal')))
 cp=root/'normal/checkpoint.json';saved=cp.read_text();bad=json.loads(saved);bad['archive'][0]['search_key']='forged';cp.write_text(json.dumps(bad))
 run('normal',resume=True,good=False);checks.append('forged candidate identity invalidates resume');cp.write_text(saved)
 bad=json.loads(saved);bad['archive'][0]['score']=0.001;cp.write_text(json.dumps(bad))
 run('normal',resume=True,good=False);checks.append('forged checkpoint score invalidates resume');cp.write_text(saved)
 evaluator.write_text(EVALUATOR+'\n# source changed\n')
 run('normal',resume=True,good=False);checks.append('changed evaluator invalidates resume')
 (root/'checks.json').write_text(json.dumps(dict(passed=True,checks=checks,evidence_type='synthetic search protocol tests only'),indent=2)+'\n')
 print(json.dumps(dict(passed=True,count=len(checks))))
if __name__=='__main__':main()
