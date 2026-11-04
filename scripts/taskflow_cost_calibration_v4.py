#!/usr/bin/env python3
"""Fit bounded train-only region walls, freeze predictions, evaluate holdout.

The freeze stage never opens a held-out result. Timing is a target-region cost,
not pure compute, and may not be added to explicit staging/sync costs again.
"""
import argparse,json,statistics,sys
from pathlib import Path
import numpy as np
from taskflow_common import fingerprint,write_json

def fit(root,out):
    split=json.loads((root/'split.json').read_text());train_ids=split['training_ids'];held=split['held_out_ids']
    if any((root/h/'result.json').exists() for h in held):raise ValueError('holdout was executed before prediction freeze')
    if (out/'model.json').exists():raise ValueError('refusing overwrite frozen model')
    rows=[]
    for cid in train_ids:
        p=root/cid/'result.json';d=json.loads(p.read_text())
        if d['status']!='passed' or d['declaration_id']!=split['declaration_id']:raise ValueError('training evidence invalid '+cid)
        rows.append((cid,d))
    A=np.array([[1,d['shape']['m'],d['shape']['h'],d['shape']['m']*d['shape']['h']] for cid,d in rows],dtype='f8')
    models={}
    for mode in split.get('modes',['independent','chain']):
        ys=[statistics.median(s['wall_ms'] for s in d['variants'][mode]['samples']) for cid,d in rows]
        if any(len(d['variants'][mode]['samples'])<7 for cid,d in rows):raise ValueError('insufficient training samples')
        coefficients=np.linalg.solve(A,np.array(ys,dtype='f8')).tolist()
        models[mode]=dict(coefficients=coefficients,training_medians_ms=dict(zip(train_ids,ys)),training_predictions_ms=(A@coefficients).tolist())
    model=dict(schema='orbit.region.cost-model.v1',declaration_id=split['declaration_id'],features=['m','h','k','n'],formula='c0+cM*M+cH*H+cMH*M*H',training_ids=train_ids,excluded_ids=held,shape_range=dict(m=[128,384],h=[256,1024],k=[256,256],n=[256,256]),legality=dict(m_multiple=128,h_multiple=256),prepared_state='warm_contexts_weights',boundary='host f32 activation to host f32 output including conversion/staging/sync/submission/intermediate handling; excludes compilation/context/immutable weight preparation',models=models,uses_holdout_measurements=False,training_evidence=[dict(path=str(root/cid/'result.json'),identity=fingerprint(d)) for cid,d in rows])
    model['model_id']=fingerprint(model);write_json(out/'model.json',model)
    pred=dict(schema='orbit.region.predictions.v1',model_id=model['model_id'],declaration_id=split['declaration_id'],features=['m','h','k','n'],uses_holdout_measurements=False,predictions={})
    for cid in held:
        shape=split['cases'][cid];values={mode:predict(model,shape,mode) for mode in models};pred['predictions'][cid]=dict(shape=shape,values_ms=values,selected=min(values,key=values.get) if set(values)=={'independent','chain'} else None)
    pred['prediction_id']=fingerprint(pred);write_json(out/'predictions.json',pred)
    entries=[]
    for mode,coeff in models.items():
        c=coeff['coefficients'];entry_mode=mode;entries.append(dict(id='npu-pair-'+mode,version='v4-train-only-'+model['model_id'][:12],pattern='matmul_pair',feature_axis='h',legality=dict(multiples=dict(m=128,h=256),power_of_two=['h']),mode=mode,device='npu',boundary_memory='host',layout='row_major',dtype='f32',numerical_contract='bf16-bfp16-bf16-mid-final',resources=dict(cpu=1,npu=1,dma=1),workspace_bytes=dict(host=256*2**20,shared=512*2**20),workspace_provenance='conservative configured reservation, not measured compiler/runtime scratch upper bound',prepared_state='warm_contexts_weights',cost=dict(kind='bilinear_m_h',scope='opaque_warm_host_to_host',fixed_ms=c[0],per_m_ms=c[1],per_h_ms=c[2],per_mh_ms=c[3],shape_range=model['shape_range'],provenance=dict(kind='calibrated',source=str(out/'model.json')),model_id=model['model_id'],training_ids=train_ids,excluded_ids=held)))
    for entry,mode in zip(entries,models):
        if mode.startswith('single_'):
            axis='n' if mode=='single_first' else 'k'
            entry.update(id='npu-'+mode.replace('_','-'),pattern='matmul_single',feature_axis=axis,mode='independent')
            entry['cost']['shape_range']=dict(model['shape_range'])
            entry['cost']['shape_range'][axis]=list(model['shape_range']['h'])
            entry['legality']['power_of_two']=['k'] if axis=='k' else []
    write_json(out/'region-implementations.json',entries)
    return pred

def predict(model,shape,mode):
    for axis,bounds in model['shape_range'].items():
        if axis not in shape or not bounds[0]<=shape[axis]<=bounds[1]:return None
    if shape['m']%128 or shape['h']%256 or shape['h']&(shape['h']-1):return None
    c=model['models'][mode]['coefficients'];x=[1,shape['m'],shape['h'],shape['m']*shape['h']];value=sum(a*b for a,b in zip(c,x))
    return value if np.isfinite(value) and value>0 else None

def evaluate(root,out):
    model=json.loads((out/'model.json').read_text());pred=json.loads((out/'predictions.json').read_text());rows={}
    for cid,p in pred['predictions'].items():
        d=json.loads((root/cid/'result.json').read_text())
        if d['status']!='passed':raise ValueError('heldout execution failed')
        observed={m:statistics.median(s['wall_ms'] for s in v['samples']) for m,v in d['variants'].items()};chosen=p['selected'];best=min(observed,key=observed.get) if chosen is not None else None
        rows[cid]=dict(predicted_ms=p['values_ms'],observed_ms=observed,relative_error={m:(p['values_ms'][m]-observed[m])/observed[m] for m in observed},selected=chosen,observed_best=best,regret_ms=observed[chosen]-observed[best] if chosen else None,regret_ratio=observed[chosen]/observed[best]-1 if chosen else None)
    result=dict(schema='orbit.region.holdout-evaluation.v1',model_id=model['model_id'],prediction_id=pred['prediction_id'],heldout=rows,out_of_domain_probe=dict(shape=dict(m=512,h=1024,k=256,n=256),predictions={m:predict(model,dict(m=512,h=1024,k=256,n=256),m) for m in model['models']},status='unknown'),no_refit=True)
    write_json(out/'holdout-evaluation.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','evaluate']);p.add_argument('--hardware-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    value=(fit if a.stage=='freeze' else evaluate)(a.hardware_root.resolve(),out);print(json.dumps(value,indent=2))
if __name__=='__main__':main()
