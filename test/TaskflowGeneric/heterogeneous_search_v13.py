#!/usr/bin/env python3
"""CPU-only native heterogeneous search contracts using synthetic test prices.

No runtime, NPU, formal calibrated model, Python planner, or Python ranking is
used. All graphs, candidate selection and prediction are produced by Core.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tool', type=Path, default=taskflow_tool('core'))
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
tool, out = args.tool.resolve(), args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
source = Path(__file__).resolve().parents[1] / 'TaskflowAIR/fixtures/heterogeneous.mlir'
checks = []


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load(path):
    return json.loads(path.read_text())


def require(condition, name):
    assert condition, name
    checks.append(name)


def run(name, command, success=True):
    save(out / (name + '.command.json'), command)
    result = subprocess.run(command, capture_output=True, text=True)
    (out / (name + '.stdout')).write_text(result.stdout)
    (out / (name + '.stderr')).write_text(result.stderr)
    save(out / (name + '.result.json'), {'returncode': result.returncode,
                                      'device_submissions': 0})
    require((result.returncode == 0) == success,
            f'{name}: expected success={success}; {result.stderr[-1200:]}')


contract_path = out / 'synthetic-contract.json'
contract = {'schema': 'orbit.cpu-test.execution-contract.v13',
            'scope': 'synthetic compiler-only unit contract, not hardware calibration',
            'observer': 'no runtime execution', 'implementation': 'scalar-f32',
            'payload_budget_bytes': 24576, 'test_target': 'native-core-planning'}
save(contract_path, contract)
config_path = out / 'config.json'
config = {'execution_contract': str(contract_path), 'search_diversity_slots': 2,
          'heterogeneous': {'task_devices': ['cpu'] * 3}}
save(config_path, config)


def export(name, input_source=source, configuration=config_path, expected=None):
    directory = out / name
    command = [str(tool), str(input_source), f'--target-config={configuration}',
               f'--heterogeneous-output={directory}']
    if expected:
        command.append(f'--check-heterogeneous={expected}')
    run(name, command)
    return load(directory / 'manifest.json')


initial = export('initial-unpriced')
require(initial['cost']['status'] == 'uncalibrated' and initial['cost']['estimated_ms'] is None,
        'absence of model is unknown, not free execution')
model_path = out / 'synthetic-model.json'
model = {'schema': 'orbit.heterogeneous.region-cost.v13',
         'program_family_key': initial['program_family_key'],
         'execution_contract_identity': initial['execution_contract_identity'],
         'provenance': 'synthetic CPU test parameters; no measured labels',
         'region_models': [],
         'boundary_model': {'knots': [{'bytes': 192, 'ms': .001},
                                      {'bytes': 384, 'ms': .002}]}}
for device in ['cpu', 'npu']:
    for count in [1, 2, 3]:
        model['region_models'].append({
            'status': 'calibrated', 'device': device, 'task_count': count,
            'provenance': 'synthetic test-only interpolation',
            'knots': [{'work_elements': points * count,
                       'ms': (.01 + points * count * .001) if device == 'cpu'
                       else (.008 + points * count * .0001)}
                      for points in [16, 32]]})
save(model_path, model)
config['heterogeneous_model'] = str(model_path)
save(config_path, config)
priced = export('priced')
require(priced['cost']['status'] == 'calibrated' and priced['cost']['estimated_ms'] > 0,
        'native provider consumes six explicitly synthetic regional models')
export('reconstruct', expected=out / 'priced/manifest.json')

# A complete three-input native region still owns exactly one submission.
all_npu_config = copy.deepcopy(config)
all_npu_config['heterogeneous']['task_devices'] = ['npu'] * 3
all_npu_path = out / 'all-npu.json'
save(all_npu_path, all_npu_config)
all_npu = export('all-npu', configuration=all_npu_path)
require(all_npu['runtime_counts']['npu_submissions'] == 1 and
        len(all_npu['regions']) == 1 and
        len(all_npu['regions'][0]['abi']) == 3,
        'one NPU call for the full input/output ABI, not one call per buffer')
projection = load(out / 'all-npu/region-0/projection.json')
require(projection['parent_graph_id'] == all_npu['graph_id'] and
        projection['child_graph_id'] == all_npu['regions'][0]['child_graph_id'] and
        projection['child_tasks_to_parent'] == [0, 1, 2],
        'checked NPU child graph and actual task projection bind parent')

# A real third input changes computation, not only metadata. Its fresh family
# is intentionally outside this synthetic model; the call-count contract is
# independent of whether a calibrated price is available.
three_input = out / 'three-input.mlir'
three_text = source.read_text()
three_text = three_text.replace(
    '%b:memref<16xf32> {orbit.role="input",orbit.name="B"},',
    '%b:memref<16xf32> {orbit.role="input",orbit.name="B"},\n'
    '                     %d:memref<16xf32> {orbit.role="input",orbit.name="D"},')
head, finish = three_text.split('%cd=taskflow.task @finish', 1)
finish = finish.replace('will_reads(%ud,%b:memref<16xf32>,memref<16xf32>)',
                        'will_reads(%ud,%b,%d:memref<16xf32>,memref<16xf32>,memref<16xf32>)', 1)
finish = finish.replace('original_read_memrefs(%ud,%b:memref<16xf32>,memref<16xf32>)',
                        'original_read_memrefs(%ud,%b,%d:memref<16xf32>,memref<16xf32>,memref<16xf32>)', 1)
finish = finish.replace(':(memref<16xf32>,memref<16xf32>,memref<16xf32>)->',
                        ':(memref<16xf32>,memref<16xf32>,memref<16xf32>,memref<16xf32>)->', 1)
finish = finish.replace('%y:memref<16xf32>,%z:memref<16xf32>',
                        '%y:memref<16xf32>,%extra:memref<16xf32>,%z:memref<16xf32>', 1)
finish = finish.replace('affine.store %r,%z[%i]:memref<16xf32>',
                        '%e=affine.load %extra[%i]:memref<16xf32>\n'
                        '    %sum=arith.addf %r,%e:f32\n'
                        '    affine.store %sum,%z[%i]:memref<16xf32>', 1)
three_input.write_text(head + '%cd=taskflow.task @finish' + finish)
three_manifest = export('three-input-region', three_input, all_npu_path)
require(sum(arg['role'] == 'input' for arg in three_manifest['regions'][0]['abi']) == 3 and
        three_manifest['runtime_counts']['npu_submissions'] == 1,
        'actual three-input NPU region still owns exactly one submission')
require(three_manifest['cost']['status'] == 'uncalibrated',
        'new three-input arithmetic family does not inherit synthetic prices')

search = out / 'search'
search_cmd = [str(tool), str(source), '--search', '--backend=cpu-npu',
              f'--target-config={config_path}', f'--search-output={search}',
              '--max-rounds=2', '--max-candidates=32', '--beam-width=6']
run('native-two-round-search', search_cmd)
report = load(search / 'search.json')
archive = report['archive']
by_id = {row['candidate_id']: row for row in archive}
passed = [row for row in archive if row['status'] == 'passed']
require(len(passed) >= 6, 'multiple legal placement and region-boundary candidates')
require(any('place-' in row['action'] for row in passed) and
        any('toggle-region-boundary-' in row['action'] for row in passed),
        'original native actions include placement and boundary changes')
require(any(len(row['action_chain']) == 2 for row in passed), 'two-round parent action chain retained')
graph_ids, plan_ids, candidate_properties = set(), set(), []
for row in passed:
    candidate = search / 'candidates' / row['candidate_id']
    graph = load(candidate / 'program-graph.json')
    manifest = load(candidate / 'evaluation/manifest.json')
    require(hashlib.sha256((candidate / 'taskflow.mlir').read_bytes()).hexdigest() == row['candidate_id'],
            'candidate ID is actual materialized source SHA')
    require(graph['id'] == row['program_graph_id'] == manifest['graph_id'],
            'candidate graph identity matches exported checked implementation')
    require(manifest['id'] == row['evaluation']['plan_id'], 'plan identity bound in native feedback')
    feedback = row['evaluation']
    require(feedback['evaluation_kind'] == 'model_prediction' and
            feedback['score']['provenance'] == 'frozen-heterogeneous-region-model' and
            feedback['score']['physical_hardware_measurement'] is False and
            feedback['executed'] is False,
            'synthetic predictions never masquerade as physical measurements')
    if row['parent']:
        require(row['action_chain'] == by_id[row['parent']]['action_chain'] + [row['action']],
                'parent chain is preserved by original search')
    for region in manifest['regions']:
        projected = load(candidate / 'evaluation' / f"region-{region['index']}/projection.json")
        require(projected['parent_graph_id'] == graph['id'] and
                projected['child_tasks_to_parent'] == region['tasks'], 'per-region parent/task projection')
    graph_ids.add(graph['id']); plan_ids.add(manifest['id'])
    candidate_properties.append({'candidate_id': row['candidate_id'], 'graph_id': graph['id'],
                                 'plan_id': manifest['id'], 'actions': row['action_chain'],
                                 'regions': len(manifest['regions']),
                                 'npu_submissions': manifest['runtime_counts']['npu_submissions']})
require(len(graph_ids) == len(passed) and len(plan_ids) == len(passed),
        'different realized choices have distinct public graphs and plans')
run('resume-exact', search_cmd + ['--resume'])
checkpoint = search / 'checkpoint.json'
checkpoint_bytes = checkpoint.read_bytes()
first = passed[0]['candidate_id']
graph_file = search / 'candidates' / first / 'program-graph.json'
graph_bytes = graph_file.read_bytes()
changed = json.loads(graph_bytes)
changed['target_identity'] = 'tampered-target'
save(graph_file, changed)
run('reject-graph-artifact', search_cmd + ['--resume'], False)
graph_file.write_bytes(graph_bytes)
changed = json.loads(checkpoint_bytes)
row = next(row for row in changed['archive'] if row['status'] == 'passed')
row['score'] += 1
row['evaluation']['score']['value'] = row['score']
save(checkpoint, changed)
run('reject-coherent-score', search_cmd + ['--resume'], False)
checkpoint.write_bytes(checkpoint_bytes)
original_model = model_path.read_bytes()
model['test_revision'] = 'changed-model-bytes'
save(model_path, model)
run('reject-model-binding', search_cmd + ['--resume'], False)
model_path.write_bytes(original_model)
original_contract = contract_path.read_bytes()
contract['test_target'] = 'different-test-target'
save(contract_path, contract)
run('reject-target-binding', search_cmd + ['--resume'], False)
contract_path.write_bytes(original_contract)
run('resume-restored', search_cmd + ['--resume'])

# Export is allowed while pricing remains unknown; Python does not assign a
# fallback score. Shape OOD and semantic-family changes are separate gates.
wide = out / 'wide.mlir'
wide.write_text(source.read_text().replace('16', '64'))
wide_manifest = export('out-of-domain', wide)
require(wide_manifest['program_family_key'] == priced['program_family_key'] and
        wide_manifest['cost']['status'] == 'uncalibrated' and
        wide_manifest['cost']['estimated_ms'] is None and wide_manifest['cost'].get('reason'),
        'same source family outside interpolation domain stays unpriced')
mutant = out / 'semantic-mutant.mlir'
mutant.write_text(source.read_text().replace('2.0', '3.0'))
mutant_manifest = export('semantic-mutant', mutant)
require(mutant_manifest['program_family_key'] != priced['program_family_key'] and
        mutant_manifest['cost']['status'] == 'uncalibrated' and
        mutant_manifest['cost']['estimated_ms'] is None,
        'arithmetic constant mutation cannot reuse same-count regional price')
# Verify source reconstruction rejects an otherwise valid stale static manifest.
run('reject-stale-static-manifest', [str(tool), str(mutant), f'--target-config={config_path}',
    f'--heterogeneous-output={out}/reject-static',
    f'--check-heterogeneous={out}/priced/manifest.json'], False)
summary = {'status': 'passed', 'checks': len(checks), 'check_names': checks,
           'accepted_candidates': len(passed), 'candidates': candidate_properties,
           'tool_sha256': hashlib.sha256(tool.read_bytes()).hexdigest(),
           'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
           'synthetic_model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
           'hardware_executed': False, 'device_submissions': 0,
           'cost_scope': 'synthetic unit-only interpolation; not the frozen formal v13 model'}
save(out / 'summary.json', summary)
print(json.dumps({'status': 'passed', 'checks': len(checks), 'accepted': len(passed)}))
