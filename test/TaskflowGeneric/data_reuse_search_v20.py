#!/usr/bin/env python3
"""Audit four C++ search policies; Python never constructs a candidate or score."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

p = argparse.ArgumentParser()
for name in ('tool', 'config', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--source', type=Path, action='append', required=True)
p.add_argument('--tile-m', type=int, default=16)
p.add_argument('--tile-n', type=int, default=64)
p.add_argument('--max-candidates', type=int, default=96)
p.add_argument('--beam-width', type=int, default=2)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def physical_signature(plan):
    # Same arithmetic source is fixed by each case. Compare chosen templates,
    # regions, copies, consumers and execution dependencies; omit requested
    # mapping choices, source SHA, proposal diagnostics and attached prices.
    keys = ('tile_sizes', 'task_implementations', 'task_loop_orders', 'phases',
            'phase_domains', 'execution_groups', 'hardware_stages', 'transfers',
            'host_buffers', 'channel_allocations', 'retained_regions',
            'readonly_retentions', 'residency_scopes', 'target_memory',
            'local_allocation_lifetime', 'fused_initializer_tasks')
    chosen = {k: plan[k] for k in keys if k in plan}
    communication = plan.get('program_communication', {})
    chosen['communication'] = {k: communication[k] for k in (
        'allocations', 'copies', 'logical_regions', 'versions', 'events',
        'data_edges', 'order_edges', 'execution_regions', 'resources',
        'task_placements', 'memories', 'links') if k in communication}
    return hashlib.sha256(json.dumps(chosen, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def invoke(directory, command, name='run', success=True):
    save(directory / (name + '-command.json'), command)
    start = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    (directory / (name + '-stdout.txt')).write_text(result.stdout)
    (directory / (name + '-stderr.txt')).write_text(result.stderr)
    assert (result.returncode == 0) == success, result.stderr
    return elapsed

binary = a.output / 'taskflow-opt-used'
shutil.copy2(a.tool, binary)
binary = binary.resolve()
tool_digest = sha(binary)
save(a.output / 'tool-identity.json', {'source': str(a.tool.resolve()), 'sha256': tool_digest})
base = json.loads(a.config.read_text())
assert base['generic_air']['cost_model']['mode'] == 'estimate'
assert base.get('search_diversity_slots', 0) == 0
summary = []
for i, source in enumerate(a.source):
    case = a.output / f'case-{i}'
    case.mkdir()
    frozen = (case / 'source.mlir').resolve()
    shutil.copy2(source, frozen)
    for algorithm in ('v19-contiguous', 'fixed-tiling', 'reuse-then-tiling', 'data-reuse-joint'):
        directory = case / algorithm
        directory.mkdir()
        config = copy.deepcopy(base)
        automatic = config.setdefault('native_auto_candidates', {})
        automatic.update(mode='joint', algorithm=algorithm, exploration_slots=0)
        path = (directory / 'config.json').resolve()
        save(path, config)
        output = (directory / 'search').resolve()
        command = [str(binary), str(frozen), '--search', '--backend=air',
                   f'--tile-m={a.tile_m}', f'--tile-n={a.tile_n}', '--tile-k=1',
                   '--max-rounds=2', f'--max-candidates={a.max_candidates}',
                   f'--beam-width={a.beam_width}', f'--target-config={path}',
                   f'--search-output={output}']
        elapsed = invoke(directory, command)
        report = json.loads((output / 'search.json').read_text())
        proposals = json.loads((output / 'automatic-proposals.json').read_text())
        assert proposals['algorithm'] == algorithm
        rows = {r['search_key']: r for r in report['archive']}
        assert len(rows) == len(report['archive'])
        signatures = set()
        candidates = []
        for key, row in rows.items():
            folder = output / 'candidates' / key
            assert sha(folder / 'taskflow.mlir') == key
            if row['parent']:
                assert row['action_chain'] == rows[row['parent']]['action_chain'] + [row['action']]
                assert row['round'] == rows[row['parent']]['round'] + 1
            if row['status'] != 'passed':
                continue
            feedback = row['evaluation']
            assert feedback['executed'] is False
            assert feedback['evaluation_kind'] == 'mechanism_estimate'
            assert feedback['score']['physical_hardware_measurement'] is False
            result = json.loads((folder / 'mapping-result.json').read_text())
            plan = result['best_plan']
            assert plan['program_graph_id'] == row['program_graph_id']
            assert plan['estimated_ms'] == row['score']
            signature = plan['direct_air_cost']['features']['execution_fingerprint']
            physical = physical_signature(plan)
            signatures.add(physical)
            candidates.append({'source': str(folder / 'taskflow.mlir'), 'source_sha256': key,
                'config': str(path), 'mapping_result': str(folder / 'mapping-result.json'),
                'graph_file': str(folder / 'program-graph.json'),
                'execution_fingerprint': signature, 'physical_signature': physical, 'estimated_ms': row['score'],
                'round': row['round'], 'parent': row['parent'], 'action_chain': row['action_chain'],
                'choices': plan['program_mapping_choices'], 'tile_sizes': plan['tile_sizes'],
                'dma_bytes': plan['actual_dma_bytes'], 'resident_bytes': plan['resident_bytes']})
        for selection in report['selections']:
            children = sorted((r for r in rows.values() if r['round'] == selection['round'] and
                               r['status'] == 'passed'), key=lambda r: (r['score'], r['search_key']))
            assert [r['search_key'] for r in selection['selected']] == [r['search_key'] for r in children[:a.beam_width]]
        if algorithm == 'data-reuse-joint':
            assert len(report['selections']) == 2 and all(s['selected'] for s in report['selections'])
            selected = {r['search_key'] for r in report['selections'][0]['selected']}
            parents = {r['parent'] for r in rows.values() if r['round'] == 2}
            assert parents and parents <= selected
            assert any(r['action'].startswith('reuse_joint_') for r in rows.values())
        before = copy.deepcopy(report)
        artifacts_before = {str(p): (sha(p), p.stat().st_mtime_ns)
                            for p in output.glob('candidates/*/response.json')}
        invoke(directory, command + ['--resume'], 'resume')
        assert json.loads((output / 'search.json').read_text()) == before
        assert {str(p): (sha(p), p.stat().st_mtime_ns)
                for p in output.glob('candidates/*/response.json')} == artifacts_before
        config_bytes = path.read_bytes()
        changed = copy.deepcopy(config)
        changed['target_version'] = str(changed.get('target_version', '')) + '-changed'
        save(path, changed)
        try:
            invoke(directory, command + ['--resume'], 'reject-target-binding', False)
        finally:
            path.write_bytes(config_bytes)
        # Corrupt a genuine chain and require native replay rejection. Restore
        # the exact checkpoint afterwards so all deliverable evidence is valid.
        checkpoint = output / 'checkpoint.json'
        original = checkpoint.read_bytes()
        mutated = json.loads(original)
        child = next((r for r in mutated['archive'] if r['parent']), None)
        if child:
            child['action_chain'][-1] = 'forged-action'
            save(checkpoint, mutated)
            try:
                invoke(directory, command + ['--resume'], 'reject-forged-chain', False)
            finally:
                checkpoint.write_bytes(original)
        all_proposals = [r for rs in proposals['parents'].values() for r in rs]
        summary.append({'case': i, 'source': str(frozen), 'algorithm': algorithm,
            'elapsed_seconds': elapsed, 'evaluations': report['evaluations'],
            'priced_legal_candidates': sum(r['status'] == 'passed' for r in rows.values()),
            'valid_unknown_candidates': sum(r['status'] == 'uncalibrated' for r in rows.values()),
            'legal_candidates': sum(r['status'] in ('passed', 'uncalibrated') for r in rows.values()),
            'best_estimate_ms': min((r['score'] for r in rows.values() if r['status'] == 'passed'), default=None),
            'physical_signatures': sorted(signatures),
            'illegal_proposals': sum(r.get('planner_supported') is False for r in all_proposals),
            'semantic_invalid': sum(r['status'] == 'semantic_invalid' for r in report['attempts']),
            'unique_physical_implementations': len(signatures),
            'compile_count': 0, 'compile_count_scope': 'C++ planning search performs no AIR/AIE compilation',
            'ranking': report['ranking'], 'selections': report['selections'], 'candidates': candidates})
        if algorithm == 'data-reuse-joint':
            strict = copy.deepcopy(config)
            strict['generic_air']['cost_model']['mode'] = 'strict'
            strict['native_auto_candidates']['exploration_slots'] = a.beam_width
            strict_path = (directory / 'strict-config.json').resolve()
            save(strict_path, strict)
            strict_command = [f'--target-config={strict_path}' if c.startswith('--target-config=') else
                              f'--search-output={directory.resolve() / "strict-search"}' if c.startswith('--search-output=') else c
                              for c in command]
            invoke(directory, strict_command, 'strict')
            strict_report = json.loads((directory / 'strict-search/search.json').read_text())
            assert not strict_report['ranking']
            assert any(r['status'] == 'uncalibrated' for r in strict_report['archive'])
            assert all(not (r.get('evaluation') or {}).get('score') for r in strict_report['archive'])
assert sha(binary) == tool_digest
for row in summary:
    peers = {r['algorithm']: {c['physical_signature'] for c in r['candidates']}
             for r in summary if r['case'] == row['case']}
    row['physical_implementations_absent_from_v19'] = sorted(
        {c['physical_signature'] for c in row['candidates']} - peers['v19-contiguous'])
save(a.output / 'comparison.json', {'status': 'PASS', 'device_submissions': 0,
    'scope': 'Same source/target/precision/budget, assumption-scored C++ search; hardware comparison separate',
    'physical_signature_policy': 'Same fixed arithmetic source: chosen templates, regions, copies/consumers and dependencies; exclude requested mapping, source identity, proposals and prices',
    'sequential_policy': 'Select scope at current tile; if capacity requires repair, first largest feasible propagated tile; later rounds tune tiles preserving selected scope',
    'results': summary})
print(f'{len(summary)} policy/case searches passed with resume, chain rejection and strict checks')
