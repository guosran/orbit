#!/usr/bin/env python3
"""Native MLIR search protocol/identity tests; no planning implemented in Python."""
import argparse, copy, json, re, subprocess
from pathlib import Path
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tool',type=Path,default=taskflow_tool('core'));a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);tool=a.tool.resolve(); checks=[]
    cfg={'generic_air':{'capacity_bytes':24576,'theoretical':{'launch_ms':.1,'per_scalar_operation_ms':.000001,'per_transfer_byte_ms':.0000001}}}
    config=out/'config.json';config.write_text(json.dumps(cfg));source=out/'input.mlir';source.write_text((ROOT/'test/TaskflowGeneric/add.mlir').read_text());d=out/'search'
    cmd=[str(tool),str(source),'--search','--backend=air','--target-config='+str(config),'--search-output='+str(d),'--max-rounds=2','--max-candidates=4','--beam-width=1']
    def run(name, command, diagnostic=None, trace=False):
        if trace:command=['strace','-f','-e','trace=process','-o',str(out/(name+'.process.log')),*command]
        (out/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
        r=subprocess.run(command,capture_output=True,text=True);(out/(name+'.log')).write_text(r.stdout+r.stderr)
        if diagnostic:assert r.returncode and diagnostic in r.stderr,(name,r.stderr)
        else:assert r.returncode==0,(name,r.stderr)
        if trace:
            execs=[l for l in (out/(name+'.process.log')).read_text().splitlines() if 'execve(' in l];assert len(execs)==1 and 'python' not in execs[0].lower(),execs
        checks.append(name)
    run('native_search',cmd,trace=True)
    j=json.loads((d/'search.json').read_text());archive=j['archive'];assert len(archive)==3 and len(j['selections'])==2
    ids=set();root_accesses=None
    for r in archive:
        assert r['status']=='passed' and r['native_mlir'] and 'orbit.plan' not in r['native_mlir']
        assert r['program'] is None, r['program']
        assert r['candidate_id']==r['search_key'] and r['candidate_id'] not in ids;ids.add(r['candidate_id'])
        i=r['round'];mlir=out/f'round{i}.mlir';mlir.write_text(r['native_mlir']);facts=out/f'round{i}.analysis.json'
        run(f'analyze_round{i}',[str(tool),str(mlir),'--generic-analyze','--analysis-output='+str(facts)])
        analysis=json.loads(facts.read_text());loops=analysis['tasks'][0]['loops'];assert len(loops)==2**i
        assert [(x['lower'],x['upper']) for x in loops]==[(k*16//(2**i),(k+1)*16//(2**i)) for k in range(2**i)]
        assert analysis['storages']==json.loads((out/'round0.analysis.json').read_text())['storages']
        if i:assert r['parent']==archive[i-1]['search_key']
    checks.append('real_body_loop_partitions_and_parent_identities')
    run('resume',cmd+['--resume'])
    checkpoint=d/'checkpoint.json';saved=checkpoint.read_text();obj=json.loads(saved)
    r=obj['archive'][0];r['score']+=1;r['evaluation']['score']['value']+=1;checkpoint.write_text(json.dumps(obj))
    run('resigned_score_rejected',cmd+['--resume'],'independent re-evaluation');checkpoint.write_text(saved)
    for name,edit in [('cost',lambda c:c['generic_air']['theoretical'].__setitem__('launch_ms',.2)),('capacity',lambda c:c['generic_air'].__setitem__('capacity_bytes',32768))]:
        changed=copy.deepcopy(cfg);edit(changed);config.write_text(json.dumps(changed));run(name+'_invalidates',cmd+['--resume'],'binding differs');config.write_text(json.dumps(cfg))
    original=source.read_text();source.write_text(original.replace('arith.addf','arith.mulf'));run('source_invalidates',cmd+['--resume'],'binding differs');source.write_text(original)
    (out/'checks.json').write_text(json.dumps({'passed':True,'count':len(checks),'checks':checks,'hardware_executed':False,'model':'explicit theoretical, uncalibrated for device'},indent=2)+'\n');print(json.dumps({'passed':True,'count':len(checks)}))
if __name__=='__main__':main()
