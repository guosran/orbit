#!/usr/bin/env python3
"""Execute checked native pair-region plans using the existing NPUChain recipe.

This deliberately rejects every plan outside the measured two-GEMM region contract.
Each compared mode must have its own native checked plan before any hardware call.
"""
import argparse,json,subprocess,sys
from pathlib import Path
from filelock import FileLock
from taskflow_npu_chain_v4 import ROOT,experiment,source,digest
from taskflow_common import fingerprint,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--plan',type=Path,action='append',required=True);p.add_argument('--config',type=Path,action='append',required=True)
    p.add_argument('--hardware-root',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--iterations',type=int,default=7)
    a=p.parse_args()
    if a.iterations<7 or len(a.plan)!=len(a.config):raise ValueError('seven samples and one config per plan required')
    root=a.hardware_root.resolve();split=json.loads((root/'split.json').read_text());pred=json.loads(a.predictions.read_text());shape=split['cases'][a.case]
    for obj,key in ((split,'declaration_id'),(pred,'prediction_id')):
        unsigned={k:v for k,v in obj.items() if k!=key}
        if fingerprint(unsigned)!=obj[key]:raise ValueError('frozen evidence identity mismatch '+key)
    if a.case not in split['held_out_ids'] or pred['declaration_id']!=split['declaration_id'] or pred['uses_holdout_measurements'] is not False:raise ValueError('frozen holdout binding invalid')
    if pred['predictions'][a.case]['shape']!=shape:raise ValueError('prediction shape mismatch')
    out=root/a.case
    if out.exists():raise ValueError('refusing existing holdout evidence')
    out.mkdir(parents=True)
    core=ROOT/'build-core/tools/taskflow-opt/taskflow-opt';bindings=[];modes=[]
    for index,(path,cfgpath) in enumerate(zip(a.plan,a.config)):
        plan=json.loads(path.read_text());cfg=json.loads(cfgpath.read_text())
        checked=subprocess.run([str(core),str(a.source),'--target-config='+str(cfgpath),'--check-mapping='+str(path)],capture_output=True,text=True)
        (out/f'check-plan-{index}.log').write_text(checked.stdout+checked.stderr)
        if checked.returncode:raise ValueError('native independent checker rejected plan')
        events=plan['events']
        if len(events)!=1 or events[0]['kind']!='region':raise ValueError('executor only supports one opaque pair region')
        e=events[0];mode=e['region_mode'];impl=next(i for i in cfg['region_implementations'] if i['id']==e['implementation_id'])
        expected={'region_pattern':'matmul_pair','prepared_state':'warm_contexts_weights','region_shape':shape,'model_id':pred['model_id']}
        if any(e.get(k)!=v for k,v in expected.items()):raise ValueError('event mode/shape/state/model incompatible')
        if mode not in ('chain','independent') or mode in modes or impl['mode']!=mode:raise ValueError('unsupported or duplicate mode')
        for key,value in dict(layout='row_major',dtype='f32',numerical_contract='bf16-bfp16-bf16-mid-final',prepared_state='warm_contexts_weights',boundary_memory='host',device='npu').items():
            if impl.get(key)!=value:raise ValueError('implementation contract mismatch '+key)
        if set(e['covered_steps'])!={'first:0','first:1','second:0','second:1'}:raise ValueError('exact semantic coverage mismatch')
        modes.append(mode);bindings.append(dict(mode=mode,plan_id=plan['plan_id'],plan_sha256=digest(path),config_sha256=digest(cfgpath),source_sha256=digest(a.source),model_id=e['model_id'],implementation_id=e['implementation_id']))
        (out/f'selected-{mode}.json').write_bytes(path.read_bytes());(out/f'config-{mode}.json').write_bytes(cfgpath.read_bytes())
    exported=out/'checked-source.json'
    q=subprocess.run([str(core),str(a.source),'--emit-plan='+str(exported)],capture_output=True,text=True)
    if q.returncode:raise ValueError('source export failed '+q.stderr)
    actual=json.loads(exported.read_text());expected=source(**shape)
    for key in ('buffers','tasks','outputs'):
        if actual[key]!=expected[key]:raise ValueError('source differs from supported executor equations '+key)
    if set(modes)!=set(pred['predictions'][a.case]['values_ms']):raise ValueError('all predicted alternatives require prechecked plans')
    write_json(out/'execution-binding.json',dict(bindings=bindings,prediction_id=pred['prediction_id'],selected=pred['predictions'][a.case]['selected'],boundary='f32 host activation to f32 host result, warm contexts and weights; no extra modeled communication'))
    with FileLock('/tmp/npu.lock',timeout=1800):
        result=experiment(a.case,out,a.iterations,False,shape,tuple(modes),split['declaration_id'])
    result['native_plan_bindings']=bindings;result['prediction_id']=pred['prediction_id'];write_json(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],bindings=bindings),indent=2));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
