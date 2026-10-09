#!/usr/bin/env python3
"""Real saved controller path replay regression, with authenticated negatives.

Usage: script --optimizer AUX_BINARY --native-root SAVED_RUN/native --output NEW
Reads recorded commands/actions; never searches, maps, predicts or changes input
receipts. Positive ranks retain expected-candidate-file exact body verification.
"""
import argparse,copy,hashlib,json,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--optimizer',type=Path,required=True);p.add_argument('--native-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise ValueError('Use a fresh output namespace')
 records=[]
 def replay(rank,mutation=None):
  origin=a.native_root/f'rank-{rank}';label=f'rank-{rank}' if mutation is None else 'negative-'+mutation;dest=out/label;dest.mkdir();actionpath=origin/'source-fission-replay-actions.json';doc=json.loads(actionpath.read_text());projected=next(x for x in doc['actions'] if x['label'].startswith('projected:'))
  if mutation=='target':projected['primitives'][0]['firstTask']='__nonexistent_task__'
  elif mutation=='mode':projected['primitives'][0]['mode']='producer-consumer-forwarded'
  elif mutation=='shape':projected['shapeRows']=1
  elif mutation=='label':projected['label']='projected:forged_label'
  fresh=dest/'actions.json';write(fresh,doc);argv=json.loads((origin/'fission-replay.command.json').read_text())['argv'];argv[0]=str(a.optimizer.resolve())
  for i,arg in enumerate(argv):
   if arg.startswith('--replay-joint-neighborhood-actions='):
    prefix,options=arg.split('=',1);tokens=options.split();assert any(t.startswith('expected-candidate-file=') for t in tokens)
    argv[i]=prefix+'='+' '.join('action-file='+str(fresh) if t.startswith('action-file=') else 'output-dir='+str(dest/'replay') if t.startswith('output-dir=') else t for t in tokens)
  start=time.monotonic();proc=subprocess.run(argv,capture_output=True,text=True,timeout=120);(dest/'stdout.log').write_text(proc.stdout);(dest/'stderr.log').write_text(proc.stderr)
  facts={}
  if proc.returncode==0:
   for fp in (dest/'replay').glob('*.json'):
    candidate=json.loads(fp.read_text())
    if isinstance(candidate,dict) and 'expected_candidate_exact_replay_match' in candidate:facts=candidate;break
  positive=mutation is None
  passed=(proc.returncode==0 and facts.get('expected_candidate_exact_replay_match') is True and facts.get('source_iteration_domain_verified') is True) if positive else (proc.returncode!=0 and 'action_not_enumerated_for_current_source_graph:' in proc.stderr)
  record={'label':label,'positive':positive,'passed':passed,'returncode':proc.returncode,'wall_seconds':time.monotonic()-start,'argv':argv,'optimizer_sha256':sha(a.optimizer),'original_actions_sha256':sha(actionpath),'mutation':mutation,'exact_candidate_match':facts.get('expected_candidate_exact_replay_match'),'source_iteration_domain_verified':facts.get('source_iteration_domain_verified'),'prediction_run':facts.get('prediction_run'),'native_mapping_run':facts.get('native_mapping_run'),'outputs':{str(f):sha(f) for f in dest.rglob('*') if f.is_file()}};write(dest/'receipt.json',record);records.append(record)
 for rank in range(5):replay(rank)
 for mutation in ['target','mode','shape','label']:replay(0,mutation)
 write(out/'result.json',{'status':'pass' if all(r['passed'] for r in records) else 'fail','positive_ranks':5,'negative_mutations':4,'records':records});print(json.dumps({'passed':sum(r['passed'] for r in records),'total':len(records)}));raise SystemExit(0 if all(r['passed'] for r in records) else 1)
if __name__=='__main__':main()
