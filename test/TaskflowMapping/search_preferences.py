#!/usr/bin/env python3
"""Synthetic protocol checks for target recommendations and estimate labels."""
import argparse,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,default=ROOT/'build/static-mapping-v3/workloads/ffn_m5/source.json');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    evaluator=out/'fixture.py'
    evaluator.write_text('''#!/usr/bin/env python3
import json,sys
r=json.load(open(sys.argv[1]));out=dict(schema='orbit.search.feedback.v1',status='passed',candidate_id=r['candidate_id'],source_id=r['source_id'],score=dict(value=1,unit='ms'))
if r['target_config'].get('label',True):out['evaluation_kind']='static_estimate'
json.dump(out,open(sys.argv[2],'w'))
''');evaluator.chmod(0o755)
    checks=[]
    source=a.source.resolve()
    for name,config,expected in [('neutral',{},5504),('target',{'search_preferences':{'m_alignment':64,'n_alignment':256,'k_alignment':64}},5632),('unlabeled',{'label':False},None),('invalid',{'search_preferences':{'n_alignment':0}},None)]:
        target=out/(name+'.json');target.write_text(json.dumps(config));dest=out/name
        cmd=[str(ROOT/'build-core/tools/taskflow-opt/taskflow-opt'),str(source),'--search','--backend=static','--evaluator='+str(evaluator),'--target-config='+str(target),'--search-output='+str(dest),'--max-rounds=1','--max-candidates=4']
        run=subprocess.run(cmd,capture_output=True,text=True);(out/(name+'.log')).write_text(run.stdout+run.stderr)
        if name=='invalid':assert run.returncode!=0;checks.append('invalid target preference rejected');continue
        assert run.returncode==0
        result=json.loads((dest/'search.json').read_text())
        if name=='unlabeled':assert not result['ranking'];checks.append('static score without estimate provenance cannot rank')
        else:
            assert any(r['action']=='tile_n_'+str(expected) for r in result['archive']);checks.append(name+' proposal alignment verified')
    (out/'checks.json').write_text(json.dumps(dict(passed=True,checks=checks,evidence_kind='synthetic protocol only'),indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks))))

if __name__=='__main__':main()
