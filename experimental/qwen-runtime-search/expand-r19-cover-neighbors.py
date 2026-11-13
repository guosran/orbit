#!/usr/bin/env python3
"""Generate unpriced CPU merge/fission neighbors of an actual incumbent.

This is a pending experiment producer, never a compiler or price estimator.
There is deliberately no size, time or evaluation cap: fission enumerates every
unordered bipartition of each currently selected legal CPU cover. Compilation,
original numerical gates and complete-program measurements must admit every new
choice before it enters ORBIT. The neighborhood is local to the incumbent;
closure is established only by repeating after measurement and joint search.
"""
import argparse
import copy
from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load(path):return json.loads(Path(path).read_text())


def record(path):
    requested=Path(path).absolute();canonical=requested.resolve(strict=True)
    # Keep the caller's path binding: later admission verifies both the bytes
    # and the symlink target, including same-content retargets.
    with requested.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    result=dict(file=str(canonical),sha256=checksum)
    if requested!=canonical:result['requested_file']=str(requested)
    return result


def require(ok,message):
    if not ok:raise ValueError(message)


def template_signature(proof,operators):
    """Structural grouping only; actual strides/code still require preparation.

    This mirrors the lowerer's SSA-index arithmetic signature without importing
    Torch or calling torch.compile. It grants no compiler-code reuse admission.
    """
    names=proof['input_values'];outputs=proof['output_values']
    kernel_outputs=outputs+list(dict.fromkeys(a['base'] for a in proof['exported_views'] if a['base'] not in outputs))
    positions={name:i for i,name in enumerate(names)};plan=[]
    keys={'eps','heads','head_dim','query_heads','kv_heads','scale','axis','sizes'}
    for name in proof['operators']:
        op=operators[name];ins=[positions[n] for n in op['inputs']];outs=[]
        for output in op['outputs']:positions[output]=len(positions);outs.append(positions[output])
        plan.append((dict(kind=op['kind'],attributes={k:v for k,v in op.get('attributes',{}).items() if k in keys}),ins,outs))
    return digest(dict(plan=plan,outputs=[positions[n] for n in kernel_outputs],
        inputs=[dict(shape=v['shape'],dtype=v['dtype']) for v in proof['input_specs'].values()]))


def choice_key(names,policy):return digest(dict(operators=sorted(names),**policy))


def policies():
    return [dict(backend=b,input_policy=i,output_policy=o)
            for b,i,o in itertools.product(('eager-host-group','inductor-cpu'),
                ('borrow-host','stage-host-copy'),('retain-host','copy-host'))]


def legal_policy(proof,requested):
    """Respect source physical aliases across the actual region boundary.

    An exported view may retain an internal computed base under input staging.
    When its base is an external input, staging would instead export a view of
    a private clone. The common source ABI still aliases the original input;
    without a typed copy bridge only borrowing that input is executable.
    """
    policy=dict(requested)
    if proof['exported_views']:policy['output_policy']='retain-host'
    if any(alias['base'] in proof['input_values'] for alias in proof['exported_views']):
        policy['input_policy']='borrow-host'
    return policy


def expand(source,catalogue,selected,backend):
    """Return pending choices plus disjoint replacement cohorts, deterministically."""
    program=backend.Program(source);operators={op['id']:op for op in source['operators']}
    impls={row['id']:row for row in catalogue['implementations']}
    runtime=catalogue['runtime_registration']['implementations']
    require(len(selected)==len(set(selected)) and all(name in impls for name in selected),'Unknown/repeated selected implementation')
    selected_rows=[runtime[name] for name in selected]
    selected_owner={}
    for name,row in zip(selected,selected_rows):
        for op in row['operators']:
            require(op not in selected_owner,'Incumbent overlaps source operators');selected_owner[op]=name
    require(set(operators)<=selected_owner.keys(),'Incumbent does not cover the complete supplied source')
    proofs={};cpu_covers={};rejected=defaultdict(int)
    def legal(names):
        key=tuple(sorted(names))
        if key not in proofs:
            try:proofs[key]=backend.region(program,list(key))
            except ValueError:proofs[key]=None
        return proofs[key]
    for name,row in zip(selected,selected_rows):
        if impls[name]['device']!='cpu' or not set(row['operators'])<=operators.keys():continue
        proof=legal(row['operators'])
        if proof is not None:cpu_covers[name]=proof
    priced={}
    for identifier,row in runtime.items():
        impl=impls[identifier]
        if (row['kind']!='prefill-cpu-region' or not impl.get('executable') or
                not impl.get('numeric_admitted') or not impl.get('training_price')):continue
        policy={key:row[key] for key in ('backend','input_policy','output_policy')}
        priced[choice_key(row['operators'],policy)]=identifier
    producer={v:op['id'] for op in operators.values() for v in op['outputs']}
    edges={(producer[v],op['id']) for op in operators.values() for v in op['inputs'] if v in producer}
    edges.update((dep,op['id']) for op in operators.values() for dep in op.get('dependencies',[]))
    pairs={tuple(sorted((selected_owner[a],selected_owner[b]))) for a,b in edges
           if selected_owner[a]!=selected_owner[b] and selected_owner[a] in cpu_covers and selected_owner[b] in cpu_covers}
    alternatives=[]
    for pair in sorted(pairs):
        proof=legal([n for name in pair for n in cpu_covers[name]['operators']])
        if proof is None:rejected['merge_nonconvex_or_effectful']+=1;continue
        alternatives.append(dict(move='merge',replace=list(pair),regions=[proof]))
    for identifier,parent in sorted(cpu_covers.items()):
        names=parent['operators'];first,*rest=names
        # Fix the first source op on the left to eliminate symmetric duplicates.
        for size in range(len(rest)+1):
            for subset in itertools.combinations(rest,size):
                left={first,*subset};right=set(names)-left
                if not right:continue
                a=legal(left);b=legal(right)
                if a is None or b is None:rejected['fission_illegal_child']+=1;continue
                ab=any(x in left and y in right for x,y in edges)
                ba=any(x in right and y in left for x,y in edges)
                if ab and ba:rejected['fission_cyclic_quotient']+=1;continue
                alternatives.append(dict(move='fission',replace=[identifier],regions=[b,a] if ba else [a,b]))
    pending={};trials={};choice_metadata={}
    for alternative in alternatives:
        for requested in policies():
            choices=[];unseen=[]
            for proof in alternative['regions']:
                policy=legal_policy(proof,requested)
                key=choice_key(proof['operators'],policy)
                if key in priced:identifier=priced[key]
                else:
                    identifier='r19-neighbor-'+key;unseen.append(identifier)
                    pending[identifier]=dict(id=identifier,operators=proof['operators'],
                        source_id=source['source_id'],runtime_config=dict(id=identifier,operators=proof['operators'],**policy),
                        structural_template_id=template_signature(proof,operators),region_proof=proof,
                        requires_actual_input_specialization_receipt=policy['backend']=='inductor-cpu',
                        executable=False,numeric_admitted=False,
                        admission_status='pending actual whole-program preparation, original numerical gate and fresh inclusive price')
                choice_metadata[identifier]=dict(template=template_signature(proof,operators),policy=policy,operators=proof['operators'])
                choices.append(identifier)
            if not unseen:continue
            trial=dict(move=alternative['move'],replace=alternative['replace'],implementations=choices,
                unpriced_implementations=unseen,
                source_operators=sorted(n for proof in alternative['regions'] for n in proof['operators']))
            trial['id']='neighbor-trial-'+digest(trial);trials[trial['id']]=trial
    # Greedy conflict coloring has no cap. Group equal structural/policy trials
    # first, so same template across layers normally enters one whole run.
    # Then pack any disjoint trial into an existing cohort, including other
    # templates/policies. Every cover still prepares its own actual inputs.
    def group_key(trial):
        return digest([{k:v for k,v in choice_metadata[n].items() if k!='operators'} for n in trial['implementations']])
    cohorts=[]
    for trial in sorted(trials.values(),key=lambda row:(group_key(row),row['source_operators'],row['id'])):
        key=group_key(trial);members=set(trial['source_operators'])
        cohort=next((c for c in cohorts if key in c['_groups'] and not members.intersection(c['_members'])),None)
        if cohort is None:
            cohort=next((c for c in cohorts if not members.intersection(c['_members'])),None)
        if cohort is None:
            cohort=dict(_groups=set(),_members=set(),trials=[],replace=[],implementations=[]);cohorts.append(cohort)
        cohort['_groups'].add(key);cohort['_members'].update(members);cohort['trials'].append(trial['id'])
        cohort['replace'].extend(trial['replace']);cohort['implementations'].extend(trial['implementations'])
    for cohort in cohorts:
        cohort['structural_groups']=sorted(cohort.pop('_groups'));cohort.pop('_members')
        require(len(cohort['replace'])==len(set(cohort['replace'])),'Cohort replaced the same incumbent twice')
        cohort['background_selected_implementations']=[n for n in selected if n not in set(cohort['replace'])]
        cohort['selected_implementations']=cohort['background_selected_implementations']+cohort['implementations']
        cohort['replace_source_covers']={n:runtime[n]['operators'] for n in cohort['replace']}
        cohort['cpu_regions']=[dict(id=n,operators=choice_metadata[n]['operators'],**choice_metadata[n]['policy'])
                               for n in cohort['implementations']]
        cohort['ready_priority']={n:min(operators[op]['source_ordinal'] for op in runtime[n]['operators'])
            for n in cohort['background_selected_implementations']
            if set(runtime[n]['operators'])<=operators.keys()}
        cohort['ready_priority'].update({n:min(operators[op]['source_ordinal'] for op in choice_metadata[n]['operators'])
                                        for n in cohort['implementations']})
        cohort['id']='neighbor-cohort-'+digest(cohort)
    return dict(schema='orbit.qwen-r19-cover-neighbors.v1',source_id=source['source_id'],
        incumbent_selected_implementations=selected,implementations=sorted(pending.values(),key=lambda row:row['id']),
        trials=sorted(trials.values(),key=lambda row:row['id']),cohorts=cohorts,
        enumeration=dict(cpu_incumbent_covers=len(cpu_covers),adjacent_pairs=len(pairs),
            legal_merge_fission_moves=len(alternatives),unpriced_choices=len(pending),
            whole_program_disjoint_cohorts=len(cohorts),rejected=dict(rejected),
            size_cap=None,time_cap=None,evaluation_cap=None,
            scope='all legal adjacent pairwise CPU-cover merges and all legal two-way fissions of this incumbent; repeat after measurement and joint search'),
        admission='pending; structural template grouping does not establish actual strides or generated-code equivalence')


def apply_cohort(ready_spec,plan,cohort,cpu_prepare_directory):
    """Produce an untimed-preparation-ready spec; never run or admit a trial.

    All replaced source ops were CPU in the incumbent. The original preparation
    plan therefore remains valid, while old host groups and priorities are
    replaced together. A fresh preparation directory is required so that new
    input-stride/compiler receipts cannot inherit an earlier cohort's proof.
    """
    result=copy.deepcopy(ready_spec)
    replaced={op for cover in cohort['replace_source_covers'].values() for op in cover}
    require(replaced and all(plan['placements'].get(op)=='cpu' for op in replaced),
            'Neighbor must preserve actual incumbent CPU preparation placements')
    regions=[]
    for row in result.get('cpu_regions',[]):
        members=set(row['operators'])
        if members&replaced:
            require(members<=replaced,'Replacement would tear an existing compiled cover')
        else:regions.append(row)
    regions.extend(copy.deepcopy(cohort['cpu_regions']))
    names=[op for row in regions for op in row['operators']]
    require(len(names)==len(set(names)),'Replacement CPU regions overlap')
    require(set(op for row in cohort['cpu_regions'] for op in row['operators'])==replaced,
            'Neighbor does not exactly preserve replaced source coverage')
    require(str(cpu_prepare_directory)!=str(result.get('cpu_prepare_directory')),
            'Neighbor requires a fresh actual-input preparation directory')
    result['cpu_regions']=regions
    result['ready_priority']=copy.deepcopy(cohort['ready_priority'])
    result['cpu_prepare_directory']=str(cpu_prepare_directory)
    result['neighbor_trial']=dict(cohort_id=cohort['id'],trials=cohort['trials'],
        scope='pending exact-cover CPU replacement; preparation/numerical/price admission still required')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','catalogue','selection','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--backend',type=Path,default=Path(__file__).with_name('qwen-r19-regions.py'))
    a=p.parse_args();selection=load(a.selection)
    selected=selection if isinstance(selection,list) else selection['selected_implementations']
    spec=importlib.util.spec_from_file_location('r19_neighbor_region_backend',a.backend)
    backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    result=expand(load(a.source),load(a.catalogue),selected,backend)
    result.update(producer=record(__file__),source=record(a.source),catalogue=record(a.catalogue),
                  selection=record(a.selection),backend=record(a.backend))
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result['enumeration']))


if __name__=='__main__':main()
