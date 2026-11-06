#!/usr/bin/env python3
"""Replay the original C++ beam; audit artifacts without constructing plans."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
for key in ('tool', 'source', 'config', 'output'):
    p.add_argument('--' + key, type=Path, required=True)
p.add_argument('--tile-m', type=int, default=4)
p.add_argument('--tile-n', type=int, default=64)
p.add_argument('--max-candidates', type=int, default=48)
p.add_argument('--beam-width', type=int, default=2)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)

def save(name, value):
    (a.output / name).write_text(json.dumps(value, indent=2) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

config = json.loads(a.config.read_text())
assert config['native_auto_candidates']['mode'] == 'joint'
assert config['native_auto_candidates'].get('exploration_slots', 0) == 0
assert config.get('search_diversity_slots', 0) == 0
assert config['generic_air']['cost_model']['kind'] == 'event-mechanism'
assert config['generic_air']['cost_model']['mode'] == 'estimate'
tool_sha = digest(a.tool)
# Snapshot the actual binary before another artifact binds the index. Tool
# provenance is a separate file; no post-compilation manifest amendments.
shutil.copy2(a.tool, a.output / 'taskflow-opt-used')
save('tool-identity.json', {'path': str(a.tool.resolve()), 'sha256': tool_sha,
                           'snapshot': str((a.output / 'taskflow-opt-used').resolve())})
save('config.json', config)
source = a.output / 'source.mlir'
source.write_bytes(a.source.read_bytes())
common = [str((a.output / 'taskflow-opt-used').resolve()), str(source.resolve()),
          '--search', '--backend=air', '--tile-m=' + str(a.tile_m),
          '--tile-n=' + str(a.tile_n), '--tile-k=1']
command = common + ['--max-rounds=2', '--max-candidates=' + str(a.max_candidates),
    '--beam-width=' + str(a.beam_width),
    '--target-config=' + str((a.output / 'config.json').resolve()),
    '--search-output=' + str((a.output / 'search').resolve())]
save('command.json', command)
result = subprocess.run(command, text=True, capture_output=True)
(a.output / 'stdout.txt').write_text(result.stdout)
(a.output / 'stderr.txt').write_text(result.stderr)
assert result.returncode == 0, result.stderr
report = json.loads((a.output / 'search/search.json').read_text())
rows = {row['search_key']: row for row in report['archive']}
entries = []
for key, row in rows.items():
    directory = (a.output / 'search/candidates' / key).resolve()
    assert digest(directory / 'taskflow.mlir') == key
    if row['status'] != 'passed':
        continue
    feedback = row['evaluation']
    assert feedback['executed'] is False
    assert feedback['evaluation_kind'] == 'mechanism_estimate'
    assert feedback['score']['physical_hardware_measurement'] is False
    mapped = json.loads((directory / 'mapping-result.json').read_text())
    plan = mapped['best_plan']
    assert plan['program_graph_id'] == row['program_graph_id'] == mapped['program_graph_id']
    assert plan['estimated_ms'] == feedback['score']['value']
    entries.append({'name': 'candidate_' + key[:12], 'candidate_directory': str(directory),
        'source': str(directory / 'taskflow.mlir'), 'source_sha256': key,
        'config': str((a.output / 'config.json').resolve()),
        'mapping_result': str(directory / 'mapping-result.json'),
        'graph_file': str(directory / 'program-graph.json'),
        'graph_id': plan['program_graph_id'], 'price_attached_plan_id': plan['plan_id'],
        'execution_fingerprint': plan['direct_air_cost']['features']['execution_fingerprint'],
        'round': row['round'], 'parent': row['parent'], 'action_chain': row['action_chain'],
        'estimated_ms': plan['estimated_ms'], 'dma_bytes': plan['actual_dma_bytes'],
        'resident_bytes': plan['resident_bytes'], 'tile_sizes': plan['tile_sizes'],
        'choices': plan['program_mapping_choices'], 'residency_scopes': plan.get('residency_scopes', []),
        'groups': len(plan['execution_groups'])})
assert len(report['selections']) == 2
selected = report['selections'][0]['selected']
children = sorted((row for row in rows.values() if row['round'] == 1 and row['status'] == 'passed'),
                  key=lambda row: (row['score'], row['search_key']))
assert [r['search_key'] for r in selected] == [r['search_key'] for r in children[:a.beam_width]]
parents = {row['parent'] for row in rows.values() if row['round'] == 2}
assert parents and parents <= {row['search_key'] for row in selected}
assert any(e['residency_scopes'] for e in entries)
save('audit.json', {'status': 'PASS', 'original_cpp_search': True,
    'model_selected_second_round_parents': True, 'device_submissions': 0,
    'scope': 'Declared-assumption beam feedback and checked artifacts; no hardware ranking claim',
    'archive_count': len(rows), 'priced_count': len(entries),
    'round_one_selection': selected, 'round_two_parents': sorted(parents),
    'ranking': report['ranking'], 'candidates': entries})
# All checked archived choices are preserved. Downstream compilation selects
# existing C++ artifacts rather than manufacturing residency/source variants.
save('checked-candidates.json', {'scope': 'Original C++ legal assumption-priced archive; no NPU calls',
                                  'entries': entries})
strict = json.loads(json.dumps(config))
strict['generic_air']['cost_model']['mode'] = 'strict'
strict['native_auto_candidates']['exploration_slots'] = a.beam_width
save('strict-config.json', strict)
strict_command = common + ['--max-rounds=1', '--max-candidates=8',
    '--beam-width=' + str(a.beam_width),
    '--target-config=' + str((a.output / 'strict-config.json').resolve()),
    '--search-output=' + str((a.output / 'strict-search').resolve())]
save('strict-command.json', strict_command)
result = subprocess.run(strict_command, text=True, capture_output=True)
(a.output / 'strict-stdout.txt').write_text(result.stdout)
(a.output / 'strict-stderr.txt').write_text(result.stderr)
assert result.returncode == 0, result.stderr
strict_report = json.loads((a.output / 'strict-search/search.json').read_text())
assert not strict_report['ranking']
assert any(row['status'] == 'uncalibrated' for row in strict_report['archive'])
assert all(not (row.get('evaluation') or {}).get('score') for row in strict_report['archive'])
assert digest(a.tool) == tool_sha == digest(a.output / 'taskflow-opt-used')
save('result.json', {'status': 'PASS', 'device_submissions': 0,
                    'scope': 'Original two-round assumption-scored search; strict unknowns retained'})
print(f'{len(rows)} archived; {len(entries)} assumption-priced; strict remains unknown')
