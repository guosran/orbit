#!/usr/bin/env python3
"""Corrupt-cache rejection tests with synthetic adapter evidence; no hardware."""
import argparse,json,pathlib,sys,tempfile
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'scripts'))
import taskflow_search_evaluate as ev
from taskflow_common import source_plan,fingerprint

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
 root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
 p=source_plan('gemm');source=root/'source.json';source.write_text(json.dumps(p));mlir=root/'candidate.mlir';mlir.write_text('synthetic test fixture, not executable')
 for name in ('inputs','reference'): (root/name).write_bytes(b'test-fixture')
 impl={'fixture':'synthetic'}
 target={'evaluation':dict(source_plan=str(source),inputs=str(root/'inputs'),reference=str(root/'reference'),iterations=3,implementation_fingerprint=fingerprint(impl),cache_dir=str(root/'cache')),'amoeba':{}}
 req=dict(schema='orbit.search.evaluation.v1',backend='amoeba',candidate_id=p['candidate_id'],source_id=p['source_id'],target_config=target,plan_path=str(source),mlir_path=str(mlir),output_dir=str(root/'evaluation'))
 calls=[]
 def fake_run(argv,**kw):
  dest=pathlib.Path(argv[argv.index('--output')+1]);dest.mkdir(parents=True,exist_ok=True)
  evidence=dict(status='passed',candidate_id=p['candidate_id'],source_id=p['source_id'],score=dict(status='scored',value=100,unit='cycles',provenance='synthetic test'))
  (dest/'feedback.json').write_text(json.dumps(evidence));calls.append(argv)
  return type('Result',(),dict(returncode=0,stdout='',stderr=''))()
 checks=[]
 with patch.object(ev,'implementation_identity',return_value=impl),patch.object(ev.subprocess,'run',side_effect=fake_run):
  first=ev.evaluate(req);assert not first['cache_hit'] and first['score']['value']==100
  second=ev.evaluate(req);assert second['cache_hit'] and len(calls)==1;checks.append('authentic cache hit')
  cache=next((root/'cache').glob('*.json'));auth=cache.read_text()
  for label,modify in [('score',lambda r:r['score'].update(value=.001)),('candidate',lambda r:r.update(candidate_id='other')),('unit',lambda r:r['score'].update(unit='ms'))]:
   bad=json.loads(auth);modify(bad['response']);cache.write_text(json.dumps(bad))
   got=ev.evaluate(req)
   assert not got.get('cache_hit',False),label
   checks.append('corrupt '+label+' invalidates cache')
  cache.write_text(auth)
  evidence=pathlib.Path(first['evidence_path']);evidence.write_text(evidence.read_text()+'\n')
  assert not ev.evaluate(req)['cache_hit'];checks.append('changed evidence invalidates cache')
  before=len(calls);(root/'inputs').write_bytes(b'changed input')
  assert not ev.evaluate(req)['cache_hit'] and len(calls)==before+1;checks.append('changed inputs invalidate cache')
 (root/'checks.json').write_text(json.dumps(dict(passed=True,checks=checks,evidence_type='synthetic cache test only'),indent=2)+'\n')
 print(json.dumps(dict(passed=True,count=len(checks))))
if __name__=='__main__':main()
