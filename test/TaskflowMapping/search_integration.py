#!/usr/bin/env python3
"""Exercise original ORBIT static evaluation, cache and checkpoint invalidation."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from taskflow_static_evaluate import load, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run=a.run.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    trace=load(run/'search.json');checks=[]
    assert trace['backend']=='static' and trace['score_unit']=='ms'
    assert len(trace['selections'])==2 and trace['evaluations']>=4
    passed=[r for r in trace['archive'] if r['status']=='passed']
    assert all(r['evaluation']['evaluation_kind']=='static_estimate' and r['evaluation']['executed'] is False for r in passed)
    checks.append('two rounds in original ORBIT use explicitly static millisecond feedback')
    selected={r['search_key'] for r in trace['selections'][0]['selected']}
    assert all(r['parent'] in selected for r in trace['archive'] if r['round']==2)
    checks.append('round one evaluated selections control round two parents')
    assert len({r['evaluation']['plan_id'] for r in passed})==len(passed)
    checks.append('structural candidates remapped to distinct independently checked plan identities')
    first=passed[0];request=load(run/'candidates'/first['search_key']/'request.json')
    request['output_dir']=str(out/'cache-evaluation')
    req=out/'request.json';write(req,request)
    def evaluate(name):
        dest=out/(name+'.json')
        subprocess.run([sys.executable,str(ROOT/'scripts/taskflow_static_evaluate.py'),str(req),str(dest)],check=True)
        return load(dest)
    hit=evaluate('cache-hit');assert hit['status']=='passed' and hit['cache_hit']
    checks.append('authentic static plan cache hit independently rechecked')
    cache=Path(request['target_config']['cache_dir'])/(hit['cache_identity']+'.json')
    original=cache.read_bytes()
    try:
        content=load(cache);content['result']['best_plan']['makespan_ms']=.00001;write(cache,content)
        repaired=evaluate('cache-tamper');assert repaired['status']=='passed' and not repaired['cache_hit'] and repaired['score']==hit['score']
        checks.append('tampered cached makespan rejected and remapped')
    finally:cache.write_bytes(original)
    command=load(run/'launch.json')
    option=next(x for x in command if x.startswith('--search-joint-neighborhood='))
    command=[x+(' resume=true' if x==option else '') for x in command]
    before=(run/'checkpoint.json').read_bytes()
    r=subprocess.run(command,capture_output=True,text=True)
    (out/'resume.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0 and (run/'checkpoint.json').read_bytes()==before
    checks.append('completed unchanged checkpoint resumes without reevaluation')
    config_path=Path(request['target_config']['mapping_config']);config_bytes=config_path.read_bytes()
    try:
        for field in ('cost','hardware'):
            config=json.loads(config_bytes)
            if field=='cost':config['implementations'][0]['cost']['fixed_ms']+=.001
            else:config['hardware']['memories'][0]['capacity_bytes']-=1
            write(config_path,config)
            r=subprocess.run(command,capture_output=True,text=True)
            (out/(field+'-resume.log')).write_text(r.stdout+r.stderr)
            assert r.returncode!=0 and 'checkpoint binding differs' in r.stderr+r.stdout
            checks.append(field+' contents changed at same path invalidate checkpoint')
            changed=evaluate(field+'-cache')
            assert changed['cache_identity']!=hit['cache_identity'] and not changed['cache_hit']
            checks.append(field+' contents changed at same path invalidate evaluation cache')
    finally:config_path.write_bytes(config_bytes)
    # Same candidate evaluated with a stricter precision contract must exclude
    # approximate implementations and acquire a new cache identity.
    config=json.loads(config_bytes);config['assumptions']['allowed_numerical_contracts']=['fp32']
    private=out/'exact-only.json';write(private,config)
    request['target_config']['mapping_config']=str(private);write(req,request)
    exact=evaluate('precision-cache');assert exact['status']=='passed' and exact['cache_identity']!=hit['cache_identity']
    evidence=load(exact['evidence_path'])
    assert all('npu' not in x for x in evidence['best_plan']['assignment'].values())
    checks.append('numerical permission changes cache identity and excludes approximate NPU implementation')
    write(out/'checks.json',dict(passed=True,checks=checks,evidence_kind='protocol tests, no hardware execution'))
    print(json.dumps(dict(passed=True,checks=len(checks))))


if __name__=='__main__':main()
