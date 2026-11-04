#!/usr/bin/env python3
"""Audit two native search rounds and checked capacity-coupled copy lifetimes.

Python launches the existing C++ driver and preserves evidence. It neither
constructs mapping plans nor computes the search decisions or model scores.
"""
import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
import sys
from pathlib import Path as _TestPath
sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool


p = argparse.ArgumentParser()
p.add_argument('--tool', type=Path, default=taskflow_tool('air'))
p.add_argument('--output', type=Path, required=True)
p.add_argument('--air', action='store_true', help='lower and independently check accepted AIR artifacts')
a = p.parse_args()
tool, out = a.tool.resolve(), a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(name, cmd, ok=True, reason=None):
    cmd = list(map(str, cmd))
    write(out / (name + '.command.json'), cmd)
    r = subprocess.run(cmd, text=True, capture_output=True)
    (out / (name + '.stdout')).write_text(r.stdout)
    (out / (name + '.stderr')).write_text(r.stderr)
    write(out / (name + '.exit.json'), {'returncode': r.returncode})
    assert (r.returncode == 0) == ok, (name, r.returncode, r.stderr)
    if reason:
        assert reason in r.stderr, (name, r.stderr)
    return r

# Same compute/dependency-only fixture used by existing backend acceptance.
source_text = (root / 'test/TaskflowAIR/fixtures/gemm-scale.mlir').read_text()
shapes = {'4x8': '32x128', '8x4': '128x64', '4x4': '32x64'}
source_text = re.sub(r'(?<=memref<)\d+x\d+(?=xf32)', lambda m: shapes[m[0]], source_text)
source_text = re.sub(r'affine.for %(i|j|k) = 0 to \d+',
    lambda m: f'affine.for %{m[1]} = 0 to {dict(i=32, j=64, k=128)[m[1]]}', source_text)
assert 'orbit.mapping' not in source_text and 'orbit.air_tile_sizes' not in source_text
source = out / 'source.mlir'
source.write_text(source_text)
config_path = out / 'config.json'
config = {
    'model_version': 'joint-data-v8-explicit-sensitivity-r1',
    'target_version': 'serial-24KiB-contract-r1',
    'generic_air': {'capacity_bytes': 24576, 'theoretical': {
        'launch_ms': .1, 'per_scalar_operation_ms': .000001,
        'per_transfer_byte_ms': .0000001}},
    'native_air_tile_candidates': [[8, 16, 32]],
    'native_mapping_candidates': 'discover',
    'search_diversity_slots': 1,
}
write(config_path, config)
search = out / 'search'
base = [tool, source, '--search', '--backend=air',
    '--air-distribute-initialization', '--tile-m=8', '--tile-n=64', '--tile-k=32',
    f'--target-config={config_path}', f'--search-output={search}',
    '--max-rounds=2', '--max-candidates=64', '--beam-width=2']
run('joint-search', base)
report = json.loads((search / 'search.json').read_text())
assert len(report['selections']) == 2 and all(s['selected'] for s in report['selections'])
rows = {r['search_key']: r for r in report['archive']}
assert len(rows) == len(report['archive'])
plans = {}
accepted, rejected = [], []
for row in report['archive']:
    ident = row['search_key']
    directory = search / 'candidates' / ident
    assert row['candidate_id'] == ident
    assert digest(directory / 'taskflow.mlir') == ident
    assert row['evaluation_identity'] == row['evaluation']['evaluation_identity']
    assert json.loads((directory/'request.json').read_text())['evaluation_identity'] == row['evaluation_identity']
    if row['parent']:
        parent = rows[row['parent']]
        assert row['action_chain'] == parent['action_chain'] + [row['action']]
        assert parent['round'] < row['round']
    else:
        assert row['action_chain'] == [] and row['round'] == 0
    evaluated = json.loads((directory / 'mapping-result.json').read_text())
    record = {'candidate_id': ident, 'parent': row['parent'], 'round': row['round'],
        'action': row['action'], 'action_chain': row['action_chain'],
        'evaluation_identity': row['evaluation_identity'], 'status': row['status'],
        'taskflow_sha256': digest(directory/'taskflow.mlir'),
        'mapping_result_sha256': digest(directory/'mapping-result.json')}
    if row['status'] != 'passed':
        record['reason'] = evaluated.get('reason', '')
        rejected.append(record)
        continue
    plan = evaluated['best_plan']
    plans[ident] = plan
    assert row['evaluation']['plan_id'] == plan['plan_id']
    assert row['score'] == plan['estimated_ms']
    assert plan['capacity_bytes'] == 24576
    assert plan['program_communication']['data_organization'] == plan['program_mapping_choices']['data_organization']
    b_bytes = sum(t['bytes'] for t in plan['transfer_summary']
        if t['storage_index'] == 1 and t['direction'] == 'host_to_l1')
    record.update(path=str(directory), plan_id=plan['plan_id'], tile_sizes=plan['tile_sizes'],
        body_fingerprint=plan['body_fingerprint'], b_transfer_bytes=b_bytes,
        policy=plan['program_mapping_choices']['data_organization'],
        panel=plan['program_mapping_choices']['data_organization'] == 'retain-across-users',
        data_organization=plan['program_mapping_choices']['data_organization'],
        storage=plan['program_mapping_choices']['storage'],
        dma_bytes=plan['actual_dma_bytes'], local_payload=plan['resident_bytes'],
        predicted_ms=plan['estimated_ms'])
    # The same supplied native candidate must reproduce the checked plan.
    cmd = [tool, directory/'taskflow.mlir', '--generic-map',
        f'--target-config={config_path}', f'--mapping-output={directory}/mapping.json']
    if a.air:
        cmd += ['--lower-to-air', f'--air-output={directory}/program.air.mlir']
    run('materialize-' + ident, cmd)
    assert json.loads((directory/'mapping.json').read_text()) == plan
    record['plan_sha256'] = digest(directory/'mapping.json')
    if a.air:
        run('check-air-' + ident, [tool, directory/'taskflow.mlir',
            f'--target-config={config_path}', f'--check-air={directory}/program.air.mlir'])
        record['air_sha256'] = digest(directory/'program.air.mlir')
    accepted.append(record)

baseline = next(r for r in accepted if not r['parent'])
assert baseline['tile_sizes'] == [8, 64, 32]
assert baseline['data_organization'] == 'phase-local'
small = next(r for r in accepted if r['round'] == 1 and r['tile_sizes'] == [8, 16, 32]
    and r['data_organization'] == 'phase-local')
assert small['predicted_ms'] > baseline['predicted_ms'], 'intermediate is deliberately worse by this explicit model'
selected1 = {r['search_key'] for r in report['selections'][0]['selected']}
assert small['candidate_id'] in selected1, 'structural diversity must preserve capacity-enabling tile'
joint = next(r for r in accepted if r['round'] == 2 and r['parent'] == small['candidate_id']
    and r['storage'] == 1 and r['data_organization'] == 'retain-across-users')
assert joint['tile_sizes'] == [8, 16, 32] and joint['local_payload'] == 9728
assert joint['dma_bytes'] < small['dma_bytes']
assert joint['predicted_ms'] < small['predicted_ms']
assert len(joint['action_chain']) == 2 and 'retain-across-users' in joint['action']
wide_rejected = next(r for r in rejected if r['parent'] == baseline['candidate_id']
    and 'retain-across-users_storage_1_reuse_0_panel_1' in r['action'])
assert any(s in wide_rejected['reason'].lower() for s in ('capacity', 'exceed', 'resident'))
assert len({r['evaluation_identity'] for r in accepted}) == len(accepted)
if a.air:
    assert len({r['air_sha256'] for r in accepted}) == len(accepted)

# Alias normalization is checked through the production CLI and planner, then
# through search itself: both initial candidates must have identical identities.
scope = ['--resident-storage=1', '--reuse-loop=0', '--panel-loop=1']
alias_results = []
for spelling, flag in [('legacy', '--residency=readonly-panel'),
                       ('public', '--data-organization=retain-across-users')]:
    d = out / ('alias-' + spelling)
    run('alias-' + spelling, [tool, source, '--search', '--backend=air',
        '--air-distribute-initialization', '--tile-m=8', '--tile-n=16', '--tile-k=32',
        flag, *scope, f'--target-config={config_path}', f'--search-output={d}',
        '--max-rounds=1', '--max-candidates=1', '--beam-width=2'])
    alias_results.append(json.loads((d/'search.json').read_text())['archive'][0])
# Evidence paths name each independent invocation directory, not semantics.
alias_semantics = copy.deepcopy(alias_results)
for row in alias_semantics:
    row['evaluation'].pop('evidence_path')
assert alias_semantics[0] == alias_semantics[1]
assert alias_results[0]['candidate_id'] == joint['candidate_id']
assert alias_results[0]['evaluation_identity'] == joint['evaluation_identity']

# Resume checks actual source/config/model, plan ID and model score,
# rather than accepting mutually consistent edits to feedback fields.
run('resume', [*base, '--resume'])
assert json.loads((search/'search.json').read_text())['archive'] == report['archive']
checkpoint = search / 'checkpoint.json'
saved = checkpoint.read_bytes()
def reject_checkpoint(name, mutate, reason):
    changed = json.loads(saved)
    mutate(changed)
    write(out / ('reject-' + name + '.checkpoint.json'), changed)
    write(checkpoint, changed)
    try:
        run('reject-' + name, [*base, '--resume'], False, reason)
    finally:
        checkpoint.write_bytes(saved)

def passed(cp):
    return next(r for r in cp['archive'] if r['status'] == 'passed')

def forge_score(cp):
    row = passed(cp)
    row['score'] += 1
    row['evaluation']['score']['value'] = row['score']

def forge_identity(cp):
    row = passed(cp)
    row['evaluation_identity'] = '0' * 64
    row['evaluation']['evaluation_identity'] = '0' * 64

reject_checkpoint('score', forge_score, 'independent re-evaluation')
reject_checkpoint('plan', lambda cp: passed(cp)['evaluation'].__setitem__('plan_id', '0'*64), 'independent re-evaluation')
reject_checkpoint('evaluation-identity', forge_identity, 'evaluation identity mismatch')
reject_checkpoint('action-chain', lambda cp: next(r for r in cp['archive'] if r['parent']).__setitem__('action_chain', ['forged']), 'action chain disagrees')

reject_checkpoint('next-round-range', lambda cp: cp.__setitem__('next_round', 4), 'round/evaluation/history consistency')
reject_checkpoint('next-round-history', lambda cp: cp.__setitem__('next_round', 2), 'round/evaluation/history consistency')
reject_checkpoint('negative-evaluations', lambda cp: cp.__setitem__('evaluations', -1), 'round/evaluation/history consistency')
reject_checkpoint('evaluation-count', lambda cp: cp.__setitem__('evaluations', cp['evaluations']-1), 'evaluation count consistency')
reject_checkpoint('duplicate-archive', lambda cp: cp['archive'].append(copy.deepcopy(cp['archive'][0])), 'duplicate checkpoint archive')
reject_checkpoint('duplicate-beam', lambda cp: cp['beam'].__setitem__(1, cp['beam'][0]), 'beam identity/order consistency')
reject_checkpoint('beam-width', lambda cp: cp['beam'].append(cp['beam'][0]), 'beam width/history consistency')
reject_checkpoint('parent-round', lambda cp: next(r for r in cp['archive'] if r['candidate_id']==joint['candidate_id']).__setitem__('round', 1), 'parent/child round consistency')

def forge_baseline(cp):
    row = next(r for r in cp['archive'] if r['candidate_id'] == small['candidate_id'])
    row.update(parent='', action='identity', action_chain=[], round=0)
reject_checkpoint('source-baseline', forge_baseline, 'baseline differs from canonical source')
reject_checkpoint('selection-score', lambda cp: cp['selections'][0]['selected'][0].__setitem__('score', 1.0), 'selected beam differs from cost/diversity selection')


def forge_consistent_action(cp):
    row = next(r for r in cp['archive'] if r['candidate_id'] == joint['candidate_id'])
    parent = next(r for r in cp['archive'] if r['candidate_id'] == row['parent'])
    row['action'] = 'mapping_phase_local'  # real proposal, but not this child
    row['action_chain'] = parent['action_chain'] + [row['action']]
reject_checkpoint('consistent-action-chain', forge_consistent_action, 'action replay differs from child identity')

def forge_completed_empty_round(cp):
    cp['archive'] = [r for r in cp['archive'] if r['round'] < 2]
    cp['evaluations'] = sum(r['status'] != 'budget_unmeasured' for r in cp['archive'])
    cp['selections'][1]['selected'] = []
    cp['beam'] = []
reject_checkpoint('omitted-round-proposals', forge_completed_empty_round, 'completed round omits a current proposal')


# Reconstruct a save() boundary after the first child in round two: archive
# already contains a current-round prefix, but selection and beam still refer
# to the completed first round. This is a consistency fixture, not a claim of
# having interrupted a live compiler process at that instruction.
in_round = json.loads(saved)
first_current = next(i for i, r in enumerate(in_round['archive']) if r['round'] == 2)
in_round['archive'] = in_round['archive'][:first_current+1]
in_round['selections'] = in_round['selections'][:1]
in_round['beam'] = [r['search_key'] for r in in_round['selections'][0]['selected']]
in_round['next_round'] = 2
in_round['evaluations'] = sum(r['status'] != 'budget_unmeasured' for r in in_round['archive'])
in_round['attempts'] = [r for r in in_round['attempts'] if r['round'] < 2]
write(out/'valid-in-round.checkpoint.json', in_round)
write(checkpoint, in_round)
try:
    run('resume-valid-in-round', [*base, '--resume'])
    resumed = json.loads((search/'search.json').read_text())
    assert resumed['archive'] == report['archive']
    assert resumed['selections'] == report['selections']
    write(out/'valid-in-round-resumed.json', resumed)
finally:
    checkpoint.write_bytes(saved)


bindings = {}
for name in ('model', 'target'):
    changed = copy.deepcopy(config)
    if name == 'model':
        changed['model_version'] += '-changed'
        changed['generic_air']['theoretical']['per_transfer_byte_ms'] *= 2
    else:
        changed['target_version'] += '-changed'
        changed['generic_air']['capacity_bytes'] = 20000
    write(out / ('changed-' + name + '.config.json'), changed)
    write(config_path, changed)
    try:
        run('reject-changed-' + name + '-resume', [*base, '--resume'], False, 'binding differs')
        d = out / ('changed-' + name)
        cmd = [f'--search-output={d}' if str(v).startswith('--search-output=') else v for v in base]
        run('fresh-changed-' + name, cmd)
        changed_row = json.loads((d/'search.json').read_text())['archive'][0]
        assert changed_row['candidate_id'] == baseline['candidate_id']
        assert changed_row['evaluation_identity'] != baseline['evaluation_identity']
        bindings[name] = changed_row['evaluation_identity']
    finally:
        write(config_path, config)
run('resume-after-negative-tests', [*base, '--resume'])

# Same budget/model with no reserved diversity slots. Record observed behavior;
# this two-legal-child fixture does not assume the branch must disappear.
plain_config = copy.deepcopy(config)
plain_config['search_diversity_slots'] = 0
write(out/'diversity0.config.json', plain_config)
write(config_path, plain_config)
plain_dir = out/'diversity0-search'
try:
    cmd = [f'--search-output={plain_dir}' if str(v).startswith('--search-output=') else v for v in base]
    run('diversity0-search', cmd)
    plain = json.loads((plain_dir/'search.json').read_text())
    plain_joint = [r for r in plain['archive'] if r['status']=='passed' and r['candidate_id']==joint['candidate_id']]
    diversity_comparison = {'same_budget_and_cost_model': True,
        'reserved_slots': 1, 'control_reserved_slots': 0,
        'with_diversity_selections': report['selections'], 'without_diversity_selections': plain['selections'],
        'with_diversity_ranking': report['ranking'], 'without_diversity_ranking': plain['ranking'],
        'control_retains_joint_branch': bool(plain_joint),
        'interpretation': 'Observed candidate retention only; no assertion that structural diversity is necessary for this fixture.'}
    write(out/'diversity-comparison.json', diversity_comparison)
finally:
    write(config_path, config)

# Ready-to-run paired complete programs; parent owns any later device run.
# The ABI uses actual native analysis and the existing external-then-scratch
# argument convention, without introducing a new executor or rewriting AIR.
cases = []
for name, record in [('n16-phase-local', small), ('n16-retain-across-users', joint)]:
    directory = Path(record['path'])
    result = json.loads((directory/'mapping-result.json').read_text())
    storages = result['analysis']['storages']
    ordered = [s for s in storages if s['role'] != 'temporary'] + [s for s in storages if s['role'] == 'temporary']
    abi = directory/'abi.json'
    write(abi, {'buffers': [dict(name=s['id'], shape=s['shape'], dtype=s['dtype'], role=s['role']) for s in ordered],
        'outputs': [s['id'] for s in ordered if s['role'] == 'output']})
    cases.append(dict(name=name, candidate_id=record['candidate_id'], plan_id=record['plan_id'],
        taskflow=str(directory/'taskflow.mlir'), plan=str(directory/'mapping.json'),
        mapping=str(directory/'mapping.json'), air=str(directory/'program.air.mlir'), abi=str(abi)))
assert small['body_fingerprint'] == joint['body_fingerprint']
write(out/'cases.json', cases)
write(out/'cases-status.json', {'air_materialized_and_checked': a.air,
    'hardware_executed': False, 'note': 'Pass --air with an AIR-enabled tool before using cases.json for execution.'})
write(out/'summary.json', {'passed': True, 'hardware_execution': False,
    'cost_scope': 'explicit uncalibrated full-program scalar-work plus DMA-byte sensitivity; no hardware timing claim',
    'compiler_sha256': digest(tool), 'source_sha256': digest(source),
    'target_config_sha256': digest(config_path), 'air_checked': a.air,
    'baseline': baseline, 'capacity_enabling_parent': small, 'joint_candidate': joint,
    'wide_retention_rejected': wide_rejected, 'accepted': accepted, 'rejected': rejected,
    'ranking': report['ranking'], 'changed_evaluation_identities': bindings,
    'legacy_public_candidate_identity': alias_results[0]['candidate_id'],
    'diversity_comparison': diversity_comparison, 'valid_in_round_resume_checked': True})
print(json.dumps({'passed': True, 'accepted': len(accepted), 'rejected': len(rejected),
    'rounds': 2, 'joint_plan_id': joint['plan_id'], 'hardware_execution': False}))
