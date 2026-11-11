#!/usr/bin/env python3
"""Exercise the C++ staged policies, raw-work budgets and checkpoint guards."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser()
for key in ['tool','config','source','output']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--tile-m',type=int,default=12)
p.add_argument('--tile-n',type=int,default=152)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def read(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
binary=a.output/'taskflow-opt-used';shutil.copy2(a.tool,binary);binary=binary.resolve()
base=read(a.config)
base['native_auto_candidates'].update(mode='joint',tile_generator='constraint',stage_rounds=2,
    max_joint_tiles=16,max_scope_sets=32,max_catalogue_configurations=512,
    max_generation_attempts=2048,max_generator_planner_calls=2048,
    max_internal_planner_calls=2048,max_cost_provider_calls=24,max_compute_instances=256,
    per_round_evaluation_quota=True,exploration_slots=0)
base['search_diversity_slots']=0
base['native_auto_candidates'].pop('max_wall_time_ms',None)
checks=[];reports={};catalogues={};plans={}
def run(directory,name,command,success=True):
    save(directory/(name+'-command.json'),command)
    r=subprocess.run(command,capture_output=True,text=True)
    (directory/(name+'-stdout.txt')).write_text(r.stdout)
    (directory/(name+'-stderr.txt')).write_text(r.stderr)
    assert (r.returncode==0)==success,(name,r.returncode,r.stderr)
    checks.append(name)
for policy in ['strong-reuse-first','strong-tile-first','alternating','joint']:
    directory=a.output/policy;directory.mkdir()
    config=copy.deepcopy(base);config['native_auto_candidates']['algorithm']=policy
    config_path=(directory/'config.json').resolve();save(config_path,config)
    output=(directory/'search').resolve()
    cmd=[str(binary),str(a.source.resolve()),'--search','--backend=air',
         '--max-rounds=4','--max-candidates=4096','--beam-width=2',
         f'--tile-m={a.tile_m}',f'--tile-n={a.tile_n}','--tile-k=1',
         f'--target-config={config_path}',f'--search-output={output}']
    run(directory,policy,cmd)
    report=read(output/'search.json');reports[policy]=report
    stats=report['stats'];assert stats['budgeted_cost_provider_calls']<=24
    assert stats['raw_full_cost_provider_calls']==stats['budgeted_cost_provider_calls']
    assert stats['internal_planner_calls']<=2048
    assert stats['score_queries']==len(report['trace'])==report['evaluations']
    assert len(report['selections'])==4
    assert stats['raw_full_cost_provider_calls']==sum(x['cost_provider_calls_delta'] for x in report['trace'])
    assert stats['internal_planner_calls']==stats['generation_stats']['planner_calls']+sum(x['planner_calls_delta'] for x in report['trace'])
    catalogue=read(output/'catalogue.json');catalogues[policy]=catalogue
    proposals=read(output/'automatic-proposals.json')['parents']
    rows={x['search_key']:x for x in report['archive']}
    best=None;costs=0
    for event in report['trace']:
        row=rows[event['candidate_id']];costs+=event['cost_provider_calls_delta']
        if row['status']=='passed':best=min(best,row['score']) if best is not None else row['score']
        assert event['best_score']==best and event['raw_cost_provider_calls']==costs
    for row in rows.values():
        assert sha(output/'candidates'/row['search_key']/'taskflow.mlir')==row['search_key']
        if row['parent']:
            parent=rows[row['parent']]
            assert row['action_chain']==parent['action_chain']+[row['action']]
            admitted=proposals[row['parent']+'@round_'+str(row['round'])]
            assert any(p['admitted'] and p['candidate_id']==row['search_key'] for p in admitted)
            if row['round']>1:
                assert row['parent'] in {x['search_key'] for x in report['selections'][row['round']-2]['selected']}
        path=output/'candidates'/row['search_key']/'mapping-result.json'
        if path.exists():
            plan=read(path)
            if row['search_key'] in plans:assert plans[row['search_key']]==plan,'same configuration must retain complete exact plan'
            plans[row['search_key']]=plan
    if policy=='strong-reuse-first':
        assert any(row['round']<=2 and len(plans[row['search_key']]['best_plan']['program_mapping_choices'].get('residency_groups',[]))>=2 for row in rows.values() if row['status']=='passed')
        assert any(row['round']>2 for row in rows.values()),'tile phase must receive evaluation budget'
    before=copy.deepcopy(report);responses={str(p):(sha(p),p.stat().st_mtime_ns) for p in output.glob('candidates/*/response.json')}
    run(directory,'resume',cmd+['--resume'])
    resumed=read(output/'search.json')
    for key in ['archive','ranking','selections','attempts','trace']:assert resumed[key]==before[key],key
    assert responses=={str(p):(sha(p),p.stat().st_mtime_ns) for p in output.glob('candidates/*/response.json')}
    assert resumed['stats']['action_cache_hits']>0
    assert resumed['stats']['resumed_from_checkpoint'] is True
    assert resumed['stats']['checkpoint_revalidation_cost_provider_calls']==sum(r['cost_provider_calls_delta'] for r in before['trace'])
    assert resumed['stats']['raw_full_cost_provider_calls']==resumed['stats']['checkpoint_revalidation_cost_provider_calls']
    checkpoint=output/'checkpoint.json';original=checkpoint.read_bytes()
    for name,edit in [('forged-prefix',lambda x:x['selections'][0]['proposal_prefix'][0].__setitem__('action','forged-action')),
                      ('forged-chain',lambda x:next(r for r in x['archive'] if r['parent'])['action_chain'].__setitem__(-1,'forged-action'))]:
        mutant=json.loads(original);edit(mutant);save(checkpoint,mutant)
        try:run(directory,name,cmd+['--resume'],False)
        finally:checkpoint.write_bytes(original)
    original_config=config_path.read_bytes();changed=copy.deepcopy(config);changed['target_version']=str(changed.get('target_version',''))+'-changed';save(config_path,changed)
    try:run(directory,'forged-target',cmd+['--resume'],False)
    finally:config_path.write_bytes(original_config)
for name,updates in [('odd-provider-budget',{'max_cost_provider_calls':3}),
                     ('internal-planner-budget',{'max_internal_planner_calls':1}),
                     ('initial-wall-budget',{'max_wall_time_ms':.01}),
                     ('strict-unknown',{'exploration_slots':2,'max_cost_provider_calls':16})]:
    directory=a.output/name;directory.mkdir();config=copy.deepcopy(base)
    config['native_auto_candidates'].update(algorithm='joint',**updates)
    if name=='strict-unknown':config['generic_air']['cost_model']['mode']='strict'
    path=(directory/'config.json').resolve();save(path,config)
    output=(directory/'search').resolve()
    cmd=[str(binary),str(a.source.resolve()),'--search','--backend=air','--max-rounds=4',
        '--max-candidates=4096','--beam-width=2',f'--tile-m={a.tile_m}',f'--tile-n={a.tile_n}',
        '--tile-k=1',f'--target-config={path}',f'--search-output={output}']
    run(directory,name,cmd);report=read(output/'search.json')
    assert report['stats']['budgeted_cost_provider_calls']<=config['native_auto_candidates']['max_cost_provider_calls']
    assert report['stats']['internal_planner_calls']<=config['native_auto_candidates']['max_internal_planner_calls']
    if name=='odd-provider-budget':assert report['evaluations']==1 and report['stats']['budgeted_cost_provider_calls']==2
    if name=='initial-wall-budget':assert report['evaluations']==0 and not report['ranking']
    if name=='strict-unknown':
        assert not report['ranking'] and all(not (r.get('evaluation') or {}).get('score') for r in report['archive'])
        assert all(r['best_score'] is None for r in report['trace'])
    run(directory,name+'-resume',cmd+['--resume'])
    resumed=read(output/'search.json')
    for key in ['archive','ranking','selections','attempts','trace']:assert resumed[key]==report[key]
    assert resumed['stats']['checkpoint_revalidation_cost_provider_calls']==sum(r['cost_provider_calls_delta'] for r in report['trace'])
assert len({c['catalogue_id'] for c in catalogues.values()})==1
assert len({tuple(p['configuration_id'] for p in c['proposals']) for c in catalogues.values()})==1
save(a.output/'result.json',dict(status='PASS',checks=checks,device_submissions=0,
    tool_sha256=sha(binary),catalogue_id=next(iter(catalogues.values()))['catalogue_id'],
    policies={k:dict(evaluations=v['evaluations'],cost_provider_calls=v['stats']['budgeted_cost_provider_calls'],round_queries=[s['score_queries'] for s in v['selections']]) for k,v in reports.items()}))
print('PASS',len(checks),'staged search/ledger/checkpoint checks')
