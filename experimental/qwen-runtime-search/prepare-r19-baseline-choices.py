#!/usr/bin/env python3
"""Bind an explicit original-runtime baseline to admitted common graph choices.

This does not search or change implementation prices. The common CLI must
materialize the resulting assignment with source-order ready priority and
independently check its real ProgramGraph before any hardware replay.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def record(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(file=str(path),sha256=checksum)


def choose(catalogue,prefill,plan,decode_directory,prefetch):
    """Resolve actual source IDs and exact-cover components, never guessed IDs."""
    if prefill['source_id']!=digest({k:v for k,v in prefill.items() if k!='source_id'}):raise ValueError('Prefill source identity changed')
    if plan.get('source_id')!=prefill['source_id'] or plan.get('plan_id')!=digest({k:v for k,v in plan.items() if k!='plan_id'}):raise ValueError('Original baseline plan identity/source mismatch')
    prefix={op['id'] for op in prefill['operators']}
    if set(plan['placements'])!=prefix:raise ValueError('Baseline placement must bind every original prefill operator')
    if any(device not in ('cpu','npu') for device in plan['placements'].values()):raise ValueError('Unsupported baseline placement')
    if plan.get('r18',{}).get('fused_layers') or plan.get('r18',{}).get('gemm_variants') or plan.get('r18_layer_ffn'):raise ValueError('Original baseline cannot silently include compiled alternatives')
    registrations=catalogue['runtime_registration']['implementations']
    implementations={row['id']:row for row in catalogue['implementations']}
    if len(implementations)!=len(catalogue['implementations']):raise ValueError('Repeated implementation identity')
    directory=Path(decode_directory).resolve()
    def matches(identifier):
        reg=registrations[identifier];kind=reg['kind']
        names=reg['operators']
        if set(names)<=prefix:
            if kind!='prefill-original' or len(names)!=1:return False
            return reg['operator']==names[0] and reg['device']==plan['placements'][names[0]]
        if set(names)&prefix:raise ValueError('Unsupported prefill/decode crossing implementation')
        if kind not in ('decode-context','host'):return False
        if 'decode_variant_dir' not in reg:
            return kind=='host'  # Fixed argmax/KV handoff have no context variant.
        return (Path(reg['decode_variant_dir']).resolve()==directory and
                reg.get('decode_rope_prefetch') is prefetch)
    selected=set();choices=[]
    for group in catalogue['choice_groups']:
        accepted=[choice for choice in group['choices'] if all(matches(i) for i in choice['implementations'])]
        if len(accepted)!=1:raise ValueError(f"Expected exactly one original baseline choice for {group['id']}: {len(accepted)}")
        chosen=accepted[0];members=set(chosen['implementations'])
        if selected&members:raise ValueError('Implementation duplicated across groups')
        selected.update(members);choices.append(dict(group_id=group['id'],choice_id=chosen['id']))
    seen=set();prefill_seen=set();decode_steps=set()
    for identifier in selected:
        row=implementations[identifier];reg=registrations[identifier]
        if row.get('source_id')!=catalogue['source_id'] or row.get('executable') is not True or row.get('wall_ms') is None:raise ValueError('Baseline implementation is not admitted/priced')
        members=set(row['source_operators'])
        if seen&members:raise ValueError('Baseline source cover overlaps')
        seen.update(members)
        if reg['kind']=='prefill-original':prefill_seen.add(reg['operator'])
        if reg['kind']=='decode-context':decode_steps.add(reg['step'])
    # Catalogue choice groups must partition precisely the admitted whole source.
    all_source={index for row in implementations.values() for index in row['source_operators']}
    if seen!=all_source or prefill_seen!=prefix or decode_steps!=set(range(127)):raise ValueError('Baseline omits full-program source or autoregressive steps')
    return choices


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('catalogue','prefill-source','plan','decode-variant-dir','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--decode-rope-prefetch',action='store_true')
    args=parser.parse_args()
    catalogue=json.loads(args.catalogue.read_text());prefill=json.loads(args.prefill_source.read_text());plan=json.loads(args.plan.read_text())
    assignment=dict(schema='orbit.program-manual-choices.v1',purpose='manual-baseline',name=args.name,
        source_id=catalogue['source_id'],execution_registration_identity=catalogue['execution_registration_identity'],
        choices=choose(catalogue,prefill,plan,args.decode_variant_dir,args.decode_rope_prefetch),
        ready_priority_policy='source-order',provenance=dict(catalogue=record(args.catalogue),prefill_source=record(args.prefill_source),plan=record(args.plan),
            producer=record(__file__),decode_variant_dir=str(args.decode_variant_dir.resolve()),decode_rope_prefetch=args.decode_rope_prefetch,
            optimization_performed=False,prices_modified=False))
    with args.output.open('x') as stream:json.dump(assignment,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=record(args.output),groups=len(assignment['choices']),purpose=assignment['purpose'])))


if __name__=='__main__':main()
