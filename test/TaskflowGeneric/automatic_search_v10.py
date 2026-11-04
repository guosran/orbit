#!/usr/bin/env python3
"""Compiler-only automatic C++ proposal and original beam/checkpoint regression.

Python launches the driver and audits its artifacts; it supplies no tile or
implementation candidate list and never constructs plans or scores candidates.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--tool', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
text = (root/'test/TaskflowAIR/fixtures/gemm-scale.mlir').read_text()
shapes = {'4x8': '32x128', '8x4': '128x64', '4x4': '32x64'}
text = re.sub(r'(?<=memref<)\d+x\d+(?=xf32)', lambda m: shapes[m[0]], text)
text = re.sub(r'affine.for %(i|j|k) = 0 to \d+',
    lambda m: f'affine.for %{m[1]} = 0 to {dict(i=32,j=64,k=128)[m[1]]}', text)
source = a.output/'source.mlir'
source.write_text(text)

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def run(name, cmd, success=True):
    cmd = list(map(str, cmd))
    save(a.output/(name+'.command.json'), cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    (a.output/(name+'.stdout')).write_text(result.stdout)
    (a.output/(name+'.stderr')).write_text(result.stderr)
    assert (result.returncode == 0) == success, (name, result.stderr)
    return result

base_config = {'target_version':'automatic-native-v10-test',
    'generic_air':{'capacity_bytes':24576, 'capabilities':{
        'tile_alignment':[1,8,1], 'implementations':[
            {'kind':'scalar','lanes':1},
            {'kind':'scalar-product-vector-add-f32','lanes':8}]},
        'theoretical':{'launch_ms':.1,'per_scalar_operation_ms':.000001,
                       'per_transfer_byte_ms':.0000001}},
    'search_diversity_slots':2}
summary = []
commands = {}
for mode in ['joint','tiling-only','implementation-residency']:
    config = copy.deepcopy(base_config)
    config['native_auto_candidates']={'mode':mode,'max_tile_proposals':12}
    path = a.output/(mode+'.json')
    save(path,config)
    directory = a.output/mode
    command = [a.tool,source,'--search','--backend=air','--air-distribute-initialization',
        '--tile-m=8','--tile-n=64','--tile-k=32',f'--target-config={path}',
        f'--search-output={directory}','--max-rounds=2','--max-candidates=64','--beam-width=3']
    commands[mode] = command
    run(mode,command)
    report = json.loads((directory/'search.json').read_text())
    assert len(report['selections']) == 2 and all(s['selected'] for s in report['selections'])
    proposals = json.loads((directory/'automatic-proposals.json').read_text())
    assert proposals['mode'] == mode and proposals['target_capability_identity']
    rows = {r['search_key']:r for r in report['archive']}
    assert len(rows) == len(report['archive'])
    facts = []
    for row in rows.values():
        location = directory/'candidates'/row['search_key']
        assert hashlib.sha256((location/'taskflow.mlir').read_bytes()).hexdigest() == row['search_key']
        if row['parent']:
            assert row['action_chain'] == rows[row['parent']]['action_chain']+[row['action']]
        if row['status'] != 'passed': continue
        result = json.loads((location/'mapping-result.json').read_text())
        plan = result['best_plan']
        assert plan['plan_id'] == row['evaluation']['plan_id']
        assert plan['resident_bytes'] <= plan['capacity_bytes']
        assert row['evaluation_identity'] == row['evaluation']['evaluation_identity']
        facts.append({'candidate_id':row['search_key'],'parent':row['parent'],
            'round':row['round'],'action_chain':row['action_chain'],
            'plan_id':plan['plan_id'],'tile_sizes':plan['tile_sizes'],
            'policy':plan['program_mapping_choices'].get('data_organization'),
            'implementations':plan['task_implementations'],
            'resident_bytes':plan['resident_bytes'],'dma_bytes':plan['actual_dma_bytes']})
    actions = [r['action'] for r in rows.values() if r['parent']]
    if mode == 'tiling-only':
        assert actions and all(x.startswith('auto_tiles_') for x in actions)
    elif mode == 'implementation-residency':
        assert actions and not any(x.startswith('auto_tiles_') for x in actions)
        assert any(x.startswith('auto_implementation_') for x in actions)
    else:
        assert any(x.startswith('auto_tiles_') for x in actions)
        assert any(x.startswith('auto_implementation_') for x in actions)
        assert any(any(x.startswith('mapping_retain-across-users') for x in f['action_chain']) and
                   any(x.startswith('auto_tiles_') for x in f['action_chain']) for f in facts)
        assert any(f['round']==2 and len(f['action_chain'])==2 for f in facts)
    summary.append({'mode':mode,'budget':{'rounds':2,'candidates':64,'beam':3},
        'evaluations':report['evaluations'],'ranking':report['ranking'],
        'accepted':facts,'selections':report['selections']})
    before = (directory/'search.json').read_bytes()
    run(mode+'-resume',[*command,'--resume'])
    assert json.loads(before)['ranking'] == json.loads((directory/'search.json').read_text())['ranking']

# Default scalar/1 and both explicit spellings are the same canonical
# source candidate. No alternate header can consume an exploration slot.
scalar_source = a.output/'explicit-scalar-source.mlir'
scalar_source.write_text(text.replace('module {',
    'module attributes {orbit.implementation = "scalar", orbit.vector_lanes = 1 : i64} {',1))
def identities(report):
    return {(row['candidate_id'],row['parent'],tuple(row['action_chain']),row['evaluation_identity'])
            for row in report['archive']}
original = json.loads((a.output/'implementation-residency/search.json').read_text())
for spelling in ['source-attributes','cli-scalar']:
    directory = a.output/('scalar-canonical-'+spelling)
    command = [scalar_source if value==source and spelling=='source-attributes' else
        f'--search-output={directory}' if str(value).startswith('--search-output=') else value
        for value in commands['implementation-residency']]
    if spelling=='cli-scalar':command += ['--implementation=scalar','--vector-lanes=1']
    run('scalar-canonical-'+spelling,command)
    report = json.loads((directory/'search.json').read_text())
    assert identities(report)==identities(original)
    assert report['ranking']==original['ranking']
    for row in report['archive']:
        if row['parent']: continue
        body = (directory/'candidates'/row['candidate_id']/'taskflow.mlir').read_text()
        assert 'orbit.implementation = "scalar"' not in body and 'orbit.vector_lanes = 1 ' not in body
        baseline = next(r for r in original['archive'] if not r['parent'])
        assert row['evaluation']['plan_id']==baseline['evaluation']['plan_id']

# Target/config binding is part of cache/checkpoint identity.
changed = copy.deepcopy(base_config)
changed['native_auto_candidates']={'mode':'joint','max_tile_proposals':12}
changed['target_version']='changed-target'
changed_path = a.output/'changed-target.json'
save(changed_path,changed)
cmd = [f'--target-config={changed_path}' if str(v).startswith('--target-config=') else v for v in commands['joint']]
run('resume-target-rejected',[*cmd,'--resume'],False)

# A legal uncalibrated parent can use a small explicit exploration beam.
# It never gains a fabricated cost or enters the scored ranking.
unknown = copy.deepcopy(base_config)
unknown['generic_air'].pop('theoretical')
unknown['native_auto_candidates']={'mode':'joint','max_tile_proposals':6,'exploration_slots':2}
unknown_path = a.output/'uncalibrated.json';save(unknown_path,unknown)
unknown_dir = a.output/'uncalibrated'
unknown_cmd = [f'--target-config={unknown_path}' if str(v).startswith('--target-config=') else
    f'--search-output={unknown_dir}' if str(v).startswith('--search-output=') else v for v in commands['joint']]
run('uncalibrated-exploration',unknown_cmd)
unknown_report = json.loads((unknown_dir/'search.json').read_text())
assert len(unknown_report['selections']) == 2 and not unknown_report['ranking']
assert any(r['round']==2 for r in unknown_report['archive'])
assert all(r['score'] is None for r in unknown_report['archive'])
for selection in unknown_report['selections']:
    assert 0 < len(selection['selected']) <= 2
    assert all(s['reason']=='uncalibrated-exploration' and s['score'] is None for s in selection['selected'])
run('uncalibrated-resume',[*unknown_cmd,'--resume'])
checkpoint = json.loads((unknown_dir/'checkpoint.json').read_text())
mutant = copy.deepcopy(checkpoint)
next(r for r in mutant['archive'] if r['status']=='uncalibrated')['score'] = .123
mutant_path = a.output/'uncalibrated-forged-score.json';save(mutant_path,mutant)
run('uncalibrated-forged-score-rejected',[*unknown_cmd,'--resume',f'--checkpoint={mutant_path}'],False)
mutant = copy.deepcopy(checkpoint)
mutant['selections'][0]['selected'][0]['reason']='predicted-best'
mutant_path = a.output/'uncalibrated-forged-selection.json';save(mutant_path,mutant)
run('uncalibrated-forged-selection-rejected',[*unknown_cmd,'--resume',f'--checkpoint={mutant_path}'],False)

# Compact feedback retains the full disk result and its semantic digest;
# resume recomputes the plan rather than trusting either artifact alone.
for row in unknown_report['archive']:
    if row['status'] != 'uncalibrated': continue
    feedback = row['evaluation']
    assert feedback['feedback_storage']=='native-auto-uncalibrated-summary-v1'
    assert len(feedback['mapping_result_canonical_sha256'])==64
    assert feedback['physical_hardware_measurement'] is False and feedback['executed'] is False
    assert (unknown_dir/'candidates'/row['candidate_id']/'mapping-result.json').is_file()
for name, mutate in [
    ('digest',lambda f:f.update(mapping_result_canonical_sha256='0'*64)),
    ('summary',lambda f:f['diagnostics']['best_plan'].update(analysis_status='forged'))]:
    mutant = copy.deepcopy(checkpoint)
    feedback = next(r for r in mutant['archive'] if r['status']=='uncalibrated')['evaluation']
    mutate(feedback)
    path = a.output/('uncalibrated-forged-'+name+'.json');save(path,mutant)
    run('uncalibrated-forged-'+name+'-rejected',[*unknown_cmd,'--resume',f'--checkpoint={path}'],False)

# An indirect mapping_config is the effective target for both prechecks and
# evaluation, while the outer object owns only search controls.
nested_target = copy.deepcopy(base_config['generic_air'])
nested_target['capacity_bytes'] = 16384
nested_target['capabilities'] = {'tile_alignment':[1,16,1],
    'implementations':[{'kind':'scalar','lanes':1}]}
nested_path = a.output/'nested-mapping.json';save(nested_path,{'generic_air':nested_target})
nested_config = {'mapping_config':str(nested_path.resolve()),'search_diversity_slots':0,
    'native_auto_candidates':{'mode':'joint','max_tile_proposals':6}}
nested_config_path = a.output/'nested-search.json';save(nested_config_path,nested_config)
nested_dir = a.output/'nested-target'
nested_cmd = [f'--target-config={nested_config_path}' if str(v).startswith('--target-config=') else
    f'--search-output={nested_dir}' if str(v).startswith('--search-output=') else
    '--max-rounds=1' if str(v).startswith('--max-rounds=') else v for v in commands['joint']]
run('nested-target',nested_cmd)
nested_report = json.loads((nested_dir/'search.json').read_text())
assert not any('scalar-product-vector-add-f32' in r['action'] for r in nested_report['archive'])
nested_proposals = json.loads((nested_dir/'automatic-proposals.json').read_text())
assert nested_proposals['parents']
assert all(p['capacity_bytes']==16384 for group in nested_proposals['parents'].values() for p in group)
assert any(not p['planner_supported'] for group in nested_proposals['parents'].values() for p in group)
for row in nested_report['archive']:
    if row['status'] == 'passed':
        plan = json.loads((nested_dir/'candidates'/row['search_key']/'mapping-result.json').read_text())['best_plan']
        assert plan['capacity_bytes']==16384
        assert all(i['kind']=='scalar' for i in plan['task_implementations'])
run('nested-target-resume',[*nested_cmd,'--resume'])

failed_request = next(p for group in nested_proposals['parents'].values() for p in group if not p['planner_supported'])
assert hashlib.sha256(failed_request['native_mlir'].encode()).hexdigest() == failed_request['candidate_id']
assert failed_request['requested_configuration']['scope'].startswith('requested')
assert failed_request['requested_configuration']['data_organization']=='phase-local'
failed_source = a.output/'failed-native-proposal.mlir'
failed_source.write_text(failed_request['native_mlir'])
run('failed-native-request-replay',[a.tool,failed_source,'--generic-map',
    f'--target-config={nested_path}',f'--mapping-output={a.output}/failed-native-proposal.json'],False)


# The old explicit-list path remains available without automatic proposals.
explicit = copy.deepcopy(base_config)
explicit['native_air_tile_candidates']=[[8,32,32]]
explicit_path = a.output/'explicit.json';save(explicit_path,explicit)
cmd = [f'--target-config={explicit_path}' if str(v).startswith('--target-config=') else
       f'--search-output={a.output}/explicit' if str(v).startswith('--search-output=') else v for v in commands['joint']]
run('explicit-compatibility',cmd)
report = json.loads((a.output/'explicit/search.json').read_text())
assert any(r['action']=='air_tiles_8_32_32' for r in report['archive'])
assert not (a.output/'explicit/automatic-proposals.json').exists()
save(a.output/'summary.json',{'passed':True,'hardware_executed':False,
    'score_scope':'explicit uncalibrated theoretical test provider; not performance evidence',
    'modes':summary,'uncalibrated_exploration':{'rounds':2,'score':None,'ranking':unknown_report['ranking'],'evaluations':unknown_report['evaluations']},'scalar_spellings_canonical':True,'nested_target_consistent':True,'resume_target_rejected':True,'explicit_lists_compatible':True})
print(json.dumps({'passed':True,'modes':3,'hardware_executed':False}))
