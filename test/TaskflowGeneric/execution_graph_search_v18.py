#!/usr/bin/env python3
"""Run the original C++ automatic search and audit estimated parent selection.

Python only supplies invocation/configuration and checks compiler artifacts. All
candidate construction, legality, cost evaluation and selection stay in C++.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--tool', type=Path, required=True)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

config = json.loads(a.config.read_text())
assert config['generic_air']['cost_model']['kind'] == 'event-mechanism'
assert config['generic_air']['cost_model']['mode'] == 'estimate'
assert config['native_auto_candidates']['mode'] == 'joint'
assert config['native_auto_candidates'].get('exploration_slots', 0) == 0
assert config.get('search_diversity_slots', 0) == 0
tool_sha = hashlib.sha256(a.tool.read_bytes()).hexdigest()
save(a.output/'tool-identity.json', {'path':str(a.tool.resolve()),'sha256':tool_sha})
save(a.output/'config.json', config)
source = a.output/'source.mlir'
source.write_bytes(a.source.read_bytes())
command = [str(a.tool.resolve()), str(source.resolve()), '--search', '--backend=air',
    '--tile-m=4', '--tile-n=128', '--tile-k=1', '--max-rounds=2',
    '--max-candidates=32', '--beam-width=2',
    '--target-config='+str((a.output/'config.json').resolve()),
    '--search-output='+str((a.output/'search').resolve())]
save(a.output/'command.json', command)
result = subprocess.run(command, text=True, capture_output=True)
(a.output/'stdout.txt').write_text(result.stdout)
(a.output/'stderr.txt').write_text(result.stderr)
assert result.returncode == 0, result.stderr
report = json.loads((a.output/'search/search.json').read_text())
rows = {row['search_key']:row for row in report['archive']}
facts = []
for key, row in rows.items():
    directory = a.output/'search/candidates'/key
    assert hashlib.sha256((directory/'taskflow.mlir').read_bytes()).hexdigest() == key
    if row['status'] != 'passed':
        continue
    feedback = row['evaluation']
    assert feedback['evaluation_kind'] == 'mechanism_estimate'
    assert feedback['score']['provenance'] == 'declared-event-mechanism-estimate'
    assert feedback['score']['physical_hardware_measurement'] is False
    assert feedback['executed'] is False
    mapped = json.loads((directory/'mapping-result.json').read_text())
    plan = mapped['best_plan']
    assert plan['program_graph_id'] == row['program_graph_id']
    assert mapped['program_graph_id'] == row['program_graph_id']
    assert plan['estimated_ms'] == feedback['score']['value']
    facts.append({'key':key, 'round':row['round'], 'parent':row['parent'],
        'actions':row['action_chain'], 'score_ms':feedback['score']['value'],
        'graph_id':plan['program_graph_id'], 'choices':plan['program_mapping_choices'],
        'dma_bytes':plan['actual_dma_bytes'], 'resident_bytes':plan['resident_bytes']})
assert len(report['selections']) == 2
first = report['selections'][0]['selected']
children = sorted((row for row in rows.values() if row['round'] == 1 and row['status'] == 'passed'),
    key=lambda row:(row['score'], row['search_key']))
assert [r['search_key'] for r in first] == [r['search_key'] for r in children[:2]]
assert first and all(r['score'] is not None for r in first)
parents = {row['parent'] for row in rows.values() if row['round'] == 2}
assert parents and parents <= {row['search_key'] for row in first}
assert any(row['choices']['grouping'] == 'tile-major' for row in facts)
save(a.output/'audit.json', {'scope':'CPU algorithm validation using declared assumptions; no performance claim',
    'original_cpp_search':True, 'model_selected_second_round_parents':True,
    'archive_count':len(rows), 'scored_count':len(facts), 'candidates':facts,
    'round_one_selection':first, 'round_two_parents':sorted(parents)})
# Export actual original-driver representatives for the existing compile-only
# orchestration. Selection reads the driver's ranking and checked plan facts;
# it does not recompute a cost or construct a mapping.
def checked_plan(row):
    return json.loads((a.output/'search/candidates'/row['search_key']/'mapping-result.json').read_text())['best_plan']
base = next(row for row in rows.values() if row['round'] == 0)
retained_base = next(row for row in rows.values() if row['action_chain'] == ['mapping_tile_major'])
winner = rows[report['ranking'][0]]
winner_tiles = checked_plan(winner)['tile_sizes']
materialized_control = next(row for row in rows.values() if row['round'] == 1 and
    row['status'] == 'passed' and checked_plan(row)['tile_sizes'] == winner_tiles and
    checked_plan(row)['program_mapping_choices']['grouping'] == 'task-shape-stages')
representatives = []
for name,row in [('baseline',base),('tilemajor512',retained_base),
                 ('winner1024',winner),('tilingonly1024',materialized_control)]:
    directory = (a.output/'search/candidates'/row['search_key']).resolve()
    plan = checked_plan(row)
    representatives.append({'name':name,'candidate_directory':str(directory),
        'source':str(directory/'taskflow.mlir'),'source_sha256':row['search_key'],
        'config':str((a.output/'config.json').resolve()),
        'mapping_result':str(directory/'mapping-result.json'),
        'graph_file':str(directory/'program-graph.json'),'graph_id':plan['program_graph_id'],
        'price_attached_plan_id':plan['plan_id'],
        'execution_fingerprint':plan['direct_air_cost']['features']['execution_fingerprint'],
        'action_chain':row['action_chain'],'estimated_ms':plan['estimated_ms'],
        'dma_bytes':plan['actual_dma_bytes'],'resident_bytes':plan['resident_bytes'],
        'groups':len(plan['execution_groups'])})
save(a.output/'compile-representatives.json', {'scope':
    'Original C++ ranked winner and checked controls; assumptions only; no device submissions',
    'entries':representatives})
print(f"{len(rows)} archived candidates; {len(facts)} assumption-priced; original C++ beam selected round-two parents")

# Reuse the same declared assumptions under strict interpretation. The original
# driver may explore legal unknowns but must never manufacture an absolute score.
strict = json.loads(json.dumps(config))
strict['generic_air']['cost_model']['mode'] = 'strict'
strict['native_auto_candidates']['exploration_slots'] = 2
save(a.output/'strict-config.json', strict)
strict_command = [str(a.tool.resolve()), str(source.resolve()), '--search', '--backend=air',
    '--tile-m=4', '--tile-n=128', '--tile-k=1', '--max-rounds=1',
    '--max-candidates=8', '--beam-width=2',
    '--target-config='+str((a.output/'strict-config.json').resolve()),
    '--search-output='+str((a.output/'strict-search').resolve())]
save(a.output/'strict-command.json', strict_command)
strict_result = subprocess.run(strict_command, text=True, capture_output=True)
(a.output/'strict-stdout.txt').write_text(strict_result.stdout)
(a.output/'strict-stderr.txt').write_text(strict_result.stderr)
assert strict_result.returncode == 0, strict_result.stderr
strict_report = json.loads((a.output/'strict-search/search.json').read_text())
assert not strict_report['ranking']
assert any(row['status'] == 'uncalibrated' for row in strict_report['archive'])
assert all(row['status'] != 'passed' and not (row.get('evaluation') or {}).get('score')
    for row in strict_report['archive'])
print('same assumptions in strict mode preserve explicit unknown scores')

assert hashlib.sha256(a.tool.read_bytes()).hexdigest() == tool_sha, 'compiler changed during search replay'
