#!/usr/bin/env python3
"""Compiler-only original C++ automatic search of real data organization.

Synthetic timing exists only to exercise the old beam/resume machinery. This
script does not propose tile/mapping actions, create plans, or claim NPU ranking.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--tool', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
config = {
    'target_version': 'compiler-data-organization-v17-synthetic-test',
    'generic_air': {'capacity_bytes': 24576,
                    'theoretical': {'launch_ms': .1,
                                    'per_scalar_operation_ms': .000001,
                                    'per_transfer_byte_ms': .0000001}},
    'native_auto_candidates': {'mode': 'joint', 'max_tile_proposals': 4},
    'search_diversity_slots': 2,
}
config_path = args.output / 'config.json'
config_path.write_text(json.dumps(config, indent=2) + '\n')

def run(name, command):
    command = list(map(str, command))
    (args.output / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
    result = subprocess.run(command, capture_output=True, text=True)
    (args.output / (name + '.stdout')).write_text(result.stdout)
    (args.output / (name + '.stderr')).write_text(result.stderr)
    assert result.returncode == 0, (name, result.returncode, result.stderr)

summaries = []
for name, fixture, tiles in [
    ('broadcast', 'broadcast.mlir', (1, 16, 1)),
    ('fanout', 'fanout-shared-v17.mlir', (1, 64, 1)),
]:
    source = args.output / (name + '.mlir')
    source.write_text((root / 'test/TaskflowAIR/fixtures' / fixture).read_text())
    directory = args.output / name
    command = [args.tool, source, '--search', '--backend=air',
               f'--tile-m={tiles[0]}', f'--tile-n={tiles[1]}', f'--tile-k={tiles[2]}',
               f'--target-config={config_path}', f'--search-output={directory}',
               '--max-rounds=2', '--max-candidates=64', '--beam-width=4']
    run(name, command)
    report = json.loads((directory / 'search.json').read_text())
    rows = {row['search_key']: row for row in report['archive']}
    proposals = json.loads((directory / 'automatic-proposals.json').read_text())
    assert proposals['mode'] == 'joint'
    candidates = []
    for key, row in rows.items():
        folder = directory / 'candidates' / key
        assert hashlib.sha256((folder / 'taskflow.mlir').read_bytes()).hexdigest() == key
        if row['parent']:
            assert row['action_chain'] == rows[row['parent']]['action_chain'] + [row['action']]
        if row['status'] != 'passed':
            continue
        plan = json.loads((folder / 'mapping-result.json').read_text())['best_plan']
        assert plan['plan_id'] == row['evaluation']['plan_id']
        choice = plan['program_mapping_choices']
        assert plan['resident_bytes'] <= plan['capacity_bytes']
        assert plan['program_graph_id'] == plan['program_graph']['id']
        if choice['residency'] == 'whole-program':
            assert plan['strategy'] == 'single-herd-resident' and plan['tile_sizes'] == []
            assert choice['data_organization'] == 'retain-across-users'
        elif choice['residency'] == 'tiled-phase-local':
            assert plan['strategy'] == 'serial-tiled'
            if choice['grouping'] == 'tile-major':
                assert choice['data_organization'] == 'retain-across-users'
                assert plan['residency_scopes']
                for scope in plan['residency_scopes']:
                    assert len(scope['tasks']) >= 2 and scope['retained_storages']
                    assert set(scope['boundary_exports']) <= set(scope['retained_storages'])
            else:
                assert choice['data_organization'] == 'phase-local'
        candidates.append(dict(candidate_id=key, parent=row['parent'],
                               action_chain=row['action_chain'],
                               graph_id=plan['program_graph_id'], plan_id=plan['plan_id'],
                               choice=choice, strategy=plan['strategy'],
                               dma_bytes=plan['actual_dma_bytes'],
                               local_bytes=plan['resident_bytes']))
    assert any(c['choice']['residency'] == 'whole-program' for c in candidates)
    assert any(c['choice']['residency'] == 'tiled-phase-local' for c in candidates)
    assert any(len(c['action_chain']) == 2 for c in candidates)
    if name == 'broadcast':
        assert any(c['choice']['residency'] == 'readonly-panel' for c in candidates)
        assert any(any(a.startswith('auto_tiles_') for a in c['action_chain']) and
                   any(a.startswith('mapping_retain-across-users') for a in c['action_chain'])
                   for c in candidates)
    else:
        resident = [c['dma_bytes'] for c in candidates if c['strategy'] == 'single-herd-resident']
        tiled = [c['dma_bytes'] for c in candidates if c['choice']['residency'] == 'tiled-phase-local']
        assert min(resident) < min(tiled)
    before = json.loads((directory / 'search.json').read_text())
    run(name + '-resume', [*command, '--resume'])
    after = json.loads((directory / 'search.json').read_text())
    assert before['ranking'] == after['ranking']
    assert {(x['search_key'], tuple(x['action_chain'])) for x in before['archive']} == {
        (x['search_key'], tuple(x['action_chain'])) for x in after['archive']}
    summaries.append(dict(name=name, candidates=candidates))
(args.output / 'summary.json').write_text(json.dumps(dict(
    passed=True, hardware_executed=False,
    scope='Original C++ automatic candidate generation and checked mapping; synthetic test prices only',
    programs=summaries), indent=2) + '\n')
print('Automatic public data organization, tiling composition, source identity and resume: PASS')
