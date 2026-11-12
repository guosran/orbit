#!/usr/bin/env python3
"""Frozen public replay identity contracts. Compiler only; no device actions."""
import argparse,copy,importlib.util,json,subprocess,tempfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tool',type=Path,required=True);p.add_argument('--solver-off',action='store_true');a=p.parse_args()
 root=Path(__file__).resolve().parents[2];d=Path(tempfile.mkdtemp(prefix='orbit-public-replay-v27-'))
 spec=importlib.util.spec_from_file_location('public_test',Path(__file__).with_name('program_all_search_v25.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 cfg=m.configuration();cfg['program_search'].update(seconds=10,max_evaluations=1)
 config=d/'target.json';config.write_text(json.dumps(cfg));search=d/'search'
 source=root/'test/TaskflowAIR/fixtures/gemm.mlir'
 def run(cmd):return subprocess.run(list(map(str,cmd)),capture_output=True,text=True,timeout=30)
 q=run([a.tool,source,'--search','--backend=air','--target-config='+str(config),'--search-output='+str(search)]);assert q.returncode==0,q.stderr
 baseline=json.loads((search/'baseline.json').read_text());evaluation=d/'evaluation.json';evaluation.write_text(json.dumps(baseline['evaluation']));assignment=d/'assignment.json';assignment.write_text(json.dumps(baseline['assignment']));space=search/'space.json';materialized=search/'candidate-0/taskflow.mlir'
 def replay(name,source=materialized,cfg=config,ev=evaluation,assn=assignment,sp=space,passed=True):
  receipt=d/(name+'.json');q=run([a.tool,source,'--target-config='+str(cfg),'--public-program-evaluation='+str(ev),'--public-program-space='+str(sp),'--public-program-assignment='+str(assn),'--implementation-output='+str(receipt)])
  assert (q.returncode==0)==passed,(name,q.stderr)
  if passed:
   r=json.loads(receipt.read_text());assert r['valid'] and r['complete_graph_matches'] and r['analytical_timing_matches'] and r['search_performed'] is False and r['physical_realization_checked'] is False
 def edited(name,value):
  path=d/name;path.write_text(json.dumps(value));return path
 replay('exact-replay')
 bad=copy.deepcopy(baseline['evaluation']);bad['program_graph']['id']='forged';replay('reject-forged-graph',ev=edited('bad-evaluation.json',bad),passed=False)
 bad=json.loads(space.read_text());bad['id']='forged';replay('reject-forged-domain',sp=edited('bad-space.json',bad),passed=False)
 bad=copy.deepcopy(baseline['assignment']);bad[0]=-1;replay('reject-assignment',assn=edited('bad-assignment.json',bad),passed=False)
 bad=copy.deepcopy(cfg);bad['generic_air']['cost_model']['parameters']['compute.operation']['value']=2;replay('reject-price-drift',cfg=edited('bad-config.json',bad),passed=False)
 bad=d/'bad-source.mlir';text=materialized.read_text();assert 'arith.mulf' in text;bad.write_text(text.replace('arith.mulf','arith.addf'));replay('reject-source-drift',source=bad,passed=False)
 for dimension,name in [('fusion.residency.traversal','mapping'),('source.initialization','initialization'),('tiling.m','tile')]:
  domains=json.loads(space.read_text())['dimensions'];index=next(i for i,d in enumerate(domains) if d['id']==dimension)
  assert len(domains[index]['choices'])>1
  bad=copy.deepcopy(baseline['assignment']);bad[index]=1
  replay('reject-wrong-'+name,assn=edited('bad-'+name+'-assignment.json',bad),passed=False)
 # Freeze a whole-program native mapping while leaving public copy retention
 # null. Replay must inherit the chosen native policy, not the old phase default.
 whole_cfg=copy.deepcopy(cfg)
 whole_cfg['program_search']['dimensions']=[dict(id=v['id'],choices=[
  dict(mapping=dict(residency='whole-program',data_organization='retain-across-users'))
  if v['id']=='fusion.residency.traversal' else None]) for v in domains]
 whole_config=edited('whole-config.json',whole_cfg);whole_search=d/'whole-search'
 q=run([a.tool,source,'--search','--backend=air','--target-config='+str(whole_config),'--search-output='+str(whole_search)]);assert q.returncode==0,q.stderr
 whole=json.loads((whole_search/'baseline.json').read_text())
 assert whole['evaluation']['program_graph']['ledger']['data_organization']=='retain-across-users',whole['evaluation']['program_graph']
 replay('whole-null-retention',source=whole_search/'candidate-0/taskflow.mlir',cfg=whole_config,ev=edited('whole-evaluation.json',whole['evaluation']),assn=edited('whole-assignment.json',whole['assignment']),sp=whole_search/'space.json')
 assert json.loads((d/'whole-null-retention.json').read_text())['graph_id']==whole['evaluation']['graph_id']
 print(json.dumps(dict(status='passed',cases=10,hardware_executed=False)))
if __name__=='__main__':main()
