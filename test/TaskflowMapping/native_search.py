#!/usr/bin/env python3
"""Native compiler/search acceptance; Python only launches and audits evidence."""
import argparse, copy, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workloads',type=Path,default=ROOT/'build/static-mapping-v3/workloads')
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    tool=ROOT/'build-core/tools/taskflow-opt/taskflow-opt'
    original=ROOT/'build-amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt'
    checks=[]
    def run(name,cmd,expected=0,trace=False):
        full=['strace','-f','-e','trace=process','-o',str(out/(name+'.process.log')),*cmd] if trace else cmd
        write(out/(name+'.command.json'),full)
        r=subprocess.run(full,capture_output=True,text=True)
        (out/(name+'.log')).write_text(r.stdout+r.stderr)
        assert (r.returncode==0)==(expected==0),(name,r.returncode,r.stderr)
        if trace:
            log=(out/(name+'.process.log')).read_text()
            execs=[l for l in log.splitlines() if 'execve(' in l]
            assert len(execs)==1 and 'python' not in execs[0].lower(),execs
            checks.append(name+': exactly one execve; no decision subprocess')
        return r
    for case in ('gemm_relu','ffn','ffn_m5','ffn_m128'):
        w=a.workloads/case;d=out/case;d.mkdir()
        cfg=json.loads((w/'config.json').read_text());write(d/'config.json',cfg)
        cmd=[str(tool),str(w/'base/taskflow.mlir'),'--static-map','--target-config='+str(d/'config.json'),'--mapping-output='+str(d/'mapping.json'),'--analysis-output='+str(d/'analysis.json')]
        run(case+'-direct',cmd,trace=True)
        result=json.loads((d/'mapping.json').read_text());assert result['status']=='passed'
        write(d/'best.json',result['best_plan'])
        run(case+'-check',[str(tool),str(w/'base/taskflow.mlir'),'--target-config='+str(d/'config.json'),'--check-mapping='+str(d/'best.json')])
        checks.append(case+': native planning and native validation')
    w=a.workloads/'gemm_relu';d=out/'search';d.mkdir()
    target={'mapping_config':str(out/'gemm_relu/config.json')}
    write(d/'target.json',target)
    base=[str(tool),str(w/'base/taskflow.mlir'),'--search','--backend=static','--target-config='+str(d/'target.json'),'--search-output='+str(d),'--max-rounds=2','--max-candidates=9','--beam-width=2']
    run('native-search',base,trace=True)
    search=json.loads((d/'search.json').read_text())
    assert len(search['selections'])==2 and all(x['selected'] for x in search['selections'])
    chosen={x['search_key'] for x in search['selections'][0]['selected']}
    assert all(x['parent'] in chosen for x in search['archive'] if x['round']==2)
    assert all(r['evaluation'].get('evaluator_kind')=='native_cpp' for r in search['archive'] if r['status']=='passed')
    checks.append('two native rounds with scored selections')
    run('native-resume',base+['--resume'],trace=True)
    checkpoint=d/'checkpoint.json'; checkpoint_bytes=checkpoint.read_bytes()
    changed=json.loads(checkpoint_bytes)
    measured=next(x for x in changed['archive'] if x['status']=='passed')
    measured['score']+=1;measured['evaluation']['score']['value']=measured['score']
    write(checkpoint,changed)
    rejected=run('tampered-native-score',base+['--resume'],expected=1)
    assert 'independent re-evaluation' in rejected.stderr
    checkpoint.write_bytes(checkpoint_bytes)
    checks.append('mutually consistent forged checkpoint scores rejected by native re-evaluation')
    cfgpath=out/'gemm_relu/config.json';saved=cfgpath.read_text();cfg=json.loads(saved)
    for label,mutate in [('cost',lambda c:c['implementations'][0]['cost'].__setitem__('fixed_ms',.123)),('hardware',lambda c:c['hardware']['resources'][0].__setitem__('capacity',2)),('implementation',lambda c:c['implementations'][0].__setitem__('version','changed'))]:
        changed=copy.deepcopy(cfg);mutate(changed);write(cfgpath,changed)
        r=run('invalidate-'+label,base+['--resume'],expected=1)
        assert 'binding differs' in r.stderr
        cfgpath.write_text(saved);checks.append(label+' change invalidates checkpoint')
    od=out/'original-search';od.mkdir()
    opts={'decision-flow':'structured','backend':'static','structured-plan':str(w/'source.json'),'target-config':str(d/'target.json'),'output-dir':str(od),'max-rounds':2,'max-candidates':9,'beam-width':2}
    command=[str(original),str(w/'base/taskflow.mlir'),'--search-joint-neighborhood='+' '.join(k+'='+str(v) for k,v in opts.items()),'-o',str(od/'output.mlir')]
    run('original-native-search',command,trace=True)
    original_result=json.loads((od/'search.json').read_text());assert len(original_result['selections'])==2 and original_result['ranking']
    checks.append('original ORBIT driver uses native evaluator by default')
    write(out/'checks.json',{'passed':True,'checks':checks,'hardware_executed':False,'cost_kind':'theoretical'})
    print(json.dumps({'passed':True,'checks':len(checks)}))
if __name__=='__main__':main()
