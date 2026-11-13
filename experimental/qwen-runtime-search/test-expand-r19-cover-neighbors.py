#!/usr/bin/env python3
"""Small graph checks for uncapped incumbent merge/fission enumeration."""
import copy
import importlib.util
from pathlib import Path


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    root=Path(__file__).parent
    helper=module(root/'expand-r19-cover-neighbors.py','neighbor_fixture')
    backend=module(root/'qwen-r19-regions.py','neighbor_fixture_backend')
    path=root/'test-expand-r19-cover-neighbors.py'
    binding=dict(id='fixture',**helper.record(path))
    source=dict(numerical_contract='original-triton-xdna-qwen25-prefill-bf16-input-f32-accumulation-v1',
        bindings=[binding],operators=[],values=[dict(id='input',shape=[2],dtype='f32',role='input')])
    for i in range(4):
        name='op'+str(i);value='v'+str(i);before='input' if i==0 else 'v'+str(i-1)
        source['operators'].append(dict(id=name,kind='add',inputs=[before,'input'],outputs=[value],
            source_ordinal=i,binding='fixture',dependencies=[]))
        source['values'].append(dict(id=value,shape=[2],dtype='f32',role='output' if i==3 else 'temporary'))
    source['source_id']=helper.digest(source)
    catalogue=dict(implementations=[],runtime_registration=dict(implementations={}))
    for identifier,names in [('left',['op0','op1']),('right',['op2','op3'])]:
        catalogue['implementations'].append(dict(id=identifier,device='cpu'))
        catalogue['runtime_registration']['implementations'][identifier]=dict(kind='prefill-original',operators=names)
    selected=['left','right'];result=helper.expand(source,catalogue,selected,backend)
    assert result['enumeration']['legal_merge_fission_moves']==3
    assert len(result['implementations'])==40
    assert any(len(row['operators'])==4 for row in result['implementations'])
    assert any(len(row['operators'])==1 for row in result['implementations'])
    assert any(len(c['structural_groups'])>1 and len(c['trials'])>1 for c in result['cohorts'])
    pending={row['id']:row for row in result['implementations']}
    for cohort in result['cohorts']:
        covered=[]
        for identifier in cohort['selected_implementations']:
            row=pending.get(identifier,catalogue['runtime_registration']['implementations'].get(identifier))
            covered.extend(row['operators'])
        assert sorted(covered)==['op0','op1','op2','op3']
    ready=dict(cpu_regions=[dict(id=n,operators=catalogue['runtime_registration']['implementations'][n]['operators']) for n in selected],cpu_prepare_directory='/tmp/old-neighbor-preparation')
    plan=dict(placements={n:'cpu' for n in ('op0','op1','op2','op3')})
    converted=helper.apply_cohort(ready,plan,result['cohorts'][0],'/tmp/new-neighbor-preparation')
    assert converted['ready_priority']==result['cohorts'][0]['ready_priority']
    assert sorted(n for row in converted['cpu_regions'] for n in row['operators'])==['op0','op1','op2','op3']
    # Only actually admitted/priced variants close the neighborhood. Merely
    # proposed entries must still be reported as unmeasured on the next pass.
    extended=copy.deepcopy(catalogue)
    for row in result['implementations']:
        identifier=row['id'];policy={k:row['runtime_config'][k] for k in ('backend','input_policy','output_policy')}
        extended['implementations'].append(dict(id=identifier,device='cpu',executable=False,numeric_admitted=False))
        extended['runtime_registration']['implementations'][identifier]=dict(kind='prefill-cpu-region',operators=row['operators'],**policy)
    assert len(helper.expand(source,extended,selected,backend)['implementations'])==40
    for row in extended['implementations'][2:]:row.update(executable=True,numeric_admitted=True,training_price=dict(samples=3))
    closed=helper.expand(source,extended,selected,backend)
    assert not closed['implementations'] and not closed['cohorts']
    barrier=copy.deepcopy(source);barrier['operators'][1]['kind']='kv-cache-write'
    barrier.pop('source_id');barrier['source_id']=helper.digest(barrier)
    singleton=dict(implementations=[dict(id=n,device='cpu') for n in ('op0','op1','op2','op3')],
        runtime_registration=dict(implementations={n:dict(kind='prefill-original',operators=[n]) for n in ('op0','op1','op2','op3')}))
    safe=helper.expand(barrier,singleton,['op0','op1','op2','op3'],backend)
    assert safe['enumeration']['legal_merge_fission_moves']==1
    assert all('op1' not in row['operators'] for row in safe['implementations'])
    # A valid quotient can order the child whose earliest source ordinal is
    # larger first. Enumerating by source-contiguous cuts would miss this.
    branched=copy.deepcopy(source)
    branched['operators'][1]['inputs']=['input','input']
    branched['operators'][2]['inputs']=['v0','v1']
    branched['operators'][3]['inputs']=['v1','input']
    branched['values'][3]['role']='output'
    branched.pop('source_id');branched['source_id']=helper.digest(branched)
    whole=dict(implementations=[dict(id='whole',device='cpu')],runtime_registration=dict(
        implementations=dict(whole=dict(kind='prefill-cpu-region',operators=['op0','op1','op2','op3']))))
    branch=helper.expand(branched,whole,['whole'],backend)
    covers={frozenset(row['operators']) for row in branch['implementations']}
    assert frozenset(('op0','op2')) in covers and frozenset(('op1','op3')) in covers
    alias_source=dict(numerical_contract=source['numerical_contract'],bindings=[binding],operators=[
        dict(id='base',kind='add',inputs=['input','input'],outputs=['base.value'],source_ordinal=0,binding='fixture'),
        dict(id='split',kind='split-view',inputs=['base.value'],outputs=['left.value','right.value'],source_ordinal=1,
             binding='fixture',attributes=dict(aliases_input=True,axis=1,sizes=[2,2],base_strides=[4,1],offsets=[0,2]))],
        values=[dict(id='input',shape=[2,4],dtype='f32',role='input'),
                dict(id='base.value',shape=[2,4],dtype='f32',role='temporary'),
                dict(id='left.value',shape=[2,2],dtype='f32',role='output'),
                dict(id='right.value',shape=[2,2],dtype='f32',role='output')])
    alias_source['source_id']=helper.digest(alias_source);program=backend.Program(alias_source)
    internal=backend.region(program,['base','split']);external=backend.region(program,['split'])
    normalize=lambda proof:{helper.digest(helper.legal_policy(proof,p)) for p in helper.policies()}
    assert len(normalize(internal))==4 and len(normalize(external))==2
    assert all(helper.legal_policy(external,p)['input_policy']=='borrow-host' for p in helper.policies())
    assert any(helper.legal_policy(internal,p)['input_policy']=='stage-host-copy' for p in helper.policies())
    # Initial catalogue has 107 non-view covers and 36 internal-base QKV4
    # covers. Alias legality preserves exactly its original 1000 policy choices.
    assert 107*len(normalize(dict(exported_views=[],input_values=[])))+36*len(normalize(internal))==1000
    print('PASS: arbitrary merge, DAG fission, KV barrier, disjoint exact covers, priced closure, external/internal view-base alias policies (initial1000 unchanged)')


if __name__=='__main__':main()
