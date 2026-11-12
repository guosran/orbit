#!/usr/bin/env python3
"""Restrict registered choices to an existing diagnostic plan; no price fitting.

This produces validation-only input for the real C++ physical lowering path.
It is not an optimization result and must not replace the benchmark catalogue.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('catalogue','plan','output','decode-variant-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    catalogue=json.loads(a.catalogue.read_text());plan=json.loads(a.plan.read_text())
    registration=catalogue['runtime_registration']['implementations']
    config=plan['r18'];routes=plan['r18_layer_ffn'];gemms=config['gemm_variants']
    fused={f'prefill.L{layer}.{role}' for layer in routes for role in ('gate_up','gate_split','swiglu','down')}
    def matches(row):
        kind=row['kind']
        if kind=='prefill-original':
            op=row['operator']
            return op not in fused and op not in gemms and row['device']==plan['placements'][op]
        if kind=='prefill-gemm':
            selected=gemms.get(row['operator'])
            return selected is not None and all(row[k]==selected[k] for k in ('directory','manifest_sha256'))
        if kind=='prefill-fused-ffn':
            selected=routes.get(str(row['layer']))
            return selected is not None and all(row[k]==selected[k] for k in ('retention','split_order','ffn_artifact'))
        if kind in ('decode-context','host'):
            return 'decode_variant_dir' not in row or Path(row['decode_variant_dir']).resolve()==a.decode_variant_dir.resolve()
        raise ValueError('Unrecognized registered implementation kind: '+kind)
    choices=[]
    for group in catalogue['choice_groups']:
        accepted=[choice for choice in group['choices'] if all(matches(registration[i]) for i in choice['implementations'])]
        if len(accepted)!=1:
            raise ValueError(f"Expected exactly one source-bound choice for {group['id']}: {len(accepted)}")
        group['choices']=accepted
        choices.append(dict(group=group['id'],choice=accepted[0]['id']))
    parent=catalogue.pop('execution_registration_identity')
    catalogue['validation_only']=dict(purpose='forced real physical lowering of already hardware-checked mixed diagnostic routes; not optimization or benchmark evidence',
        parent_registration_identity=parent,parent_catalogue=dict(file=str(a.catalogue.resolve()),sha256=hashlib.sha256(a.catalogue.read_bytes()).hexdigest()),
        diagnostic_plan=dict(file=str(a.plan.resolve()),sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest()),
        producer=dict(file=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        all_prices_unchanged=True,forced_choices=choices)
    catalogue['execution_registration_identity']=digest(catalogue)
    with a.output.open('x') as f:json.dump(catalogue,f,separators=(',',':'));f.write('\n')
    print(json.dumps(dict(output=str(a.output),groups=len(choices),registration=catalogue['execution_registration_identity'])))


if __name__=='__main__':main()
