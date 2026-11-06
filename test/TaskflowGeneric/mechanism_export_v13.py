#!/usr/bin/env python3
"""CPU-only mechanism export/precedence contracts with synthetic parameters.

Uses actual Taskflow and native graph/region implementations. No search, device
execution, formal experimental model, or empirical whole-program ranking occurs.
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

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tool', type=Path, default=taskflow_tool('core'))
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
tool, out = a.tool.resolve(), a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
source = Path(__file__).resolve().parents[1] / 'TaskflowAIR/fixtures/heterogeneous.mlir'
checks = []


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def load(path):
    return json.loads(path.read_text())


def require(value, label):
    assert value, label
    checks.append(label)


def run(name, command, success=True):
    save(out / f'{name}.command.json', command)
    result = subprocess.run(command, text=True, capture_output=True)
    (out / f'{name}.stdout').write_text(result.stdout)
    (out / f'{name}.stderr').write_text(result.stderr)
    save(out / f'{name}.result.json', {'returncode': result.returncode,
                                     'device_submissions': 0})
    require((result.returncode == 0) == success,
            f'{name}: expected success={success}: {result.stderr[-1000:]}')


contract = out / 'synthetic-contract.json'
save(contract, {'schema': 'orbit.cpu-test.execution-contract.v13',
                'scope': 'synthetic CPU unit contract; not executable/calibrated hardware',
                'implementation': 'strict-f32', 'prepared': True})
config = {'execution_contract': str(contract),
          'heterogeneous': {'task_devices': ['npu', 'npu', 'cpu']}}
config_path = out / 'config.json'
save(config_path, config)


def export(name, configuration=config_path, expected=None, success=True):
    command = [str(tool), str(source), f'--target-config={configuration}',
               f'--heterogeneous-output={out / name}']
    if expected is not None:
        command.append(f'--check-heterogeneous={expected}')
    run(name, command, success)
    return load(out / name / 'manifest.json') if success else None


initial = export('initial')
facts = initial['mechanism_facts']
require(facts == load(out / 'initial/mechanism-facts.json'), 'facts sidecar is exact manifest view')
require(initial['mechanism_cost'] == load(out / 'initial/mechanism-cost.json'), 'cost sidecar is exact manifest view')
require(facts['program_graph_id'] == initial['graph_id'], 'mechanism facts bind checked parent graph')
require(initial['runtime_counts']['npu_submissions'] == 1 and
        initial['runtime_counts']['cpu_calls'] == 1, 'NNC has one actual NPU region call and one CPU call')
items = facts['items']
require(sum(i['mechanism'] == 'npu.submit' for i in items) == 1,
        'input count does not multiply submit mechanism items')
resources = {r['id']: r for r in initial['graph']['ledger']['memory_architecture']['resources']}
for label, binding in facts['runtime_resource_bindings'].items():
    require(binding['target_resource'] in resources and
            binding['target_capacity'] == resources[binding['target_resource']]['capacity'],
            'runtime owner maps to actual positive-capacity graph resource')
require(facts['runtime_resource_bindings']['host.memory']['target_resource'] == 'CPU.compute' and
        facts['runtime_resource_bindings']['host.thread']['target_resource'] == 'CPU.compute',
        'host aliases serialize on host worker, not a fictitious DRAM port')
public = initial['graph']['region_events']
for item in items:
    if item['mechanism'] in ('npu.submit', 'npu.wait'):
        expected_kind = 'submit' if item['mechanism'] == 'npu.submit' else 'complete'
        require(len(item['region_events']) == 1 and
                public[item['region_events'][0]]['kind'] == expected_kind,
                'submit/wait references only matching public lifecycle kind')
    if item['mechanism'] in ('bo.sync_to_device', 'bo.sync_from_device'):
        if item['runtime_only']:
            require(item['mechanism'] == 'bo.sync_to_device' and not item['region_events'],
                    'conservative output TO sync is not a logical data read')
        else:
            kind = 'visible-to-device' if item['mechanism'] == 'bo.sync_to_device' else 'visible-to-host'
            require(len(item['region_events']) == 1 and
                    public[item['region_events'][0]]['kind'] == kind and
                    set(item['copies']).issubset(public[item['region_events'][0]]['copies']),
                    'sync direction and host-copy public binding are exact')
    if item['descriptive_only']:
        owner = items[item['covered_by']]
        require(owner['composite'] and owner['id'] < item['id'],
                'descriptive movement has one preceding opaque owner')

# Produce parameter rows from compiler facts, never reproduce a plan or score.
# Values below are deliberately synthetic unit inputs, not measured fit data.
parameter_path = out / 'synthetic-parameters.json'
rows, seen = [], set()
for item in items:
    if item['composite'] or item['descriptive_only']:
        continue
    key = tuple(item[k] for k in ('mechanism', 'path', 'state', 'resource', 'boundary'))
    if key in seen:
        continue
    seen.add(key)
    rows.append(dict(zip(('mechanism', 'path', 'state', 'resource', 'boundary'), key),
                     runtime_identity=facts['runtime']['identity'], alpha_ms=.001,
                     beta_ms_per_byte=0, min_bytes=0, max_bytes=4096,
                     provenance='synthetic CPU test-only numbers; not microbench labels',
                     measurement_boundary=item['boundary'], raw_manifest_sha256='a' * 64))
parameters = {'schema': 'orbit.mechanism-cost-parameters.v13', 'parameters': rows}
save(parameter_path, parameters)
model_path = out / 'synthetic-historical-regions.json'
model = {'schema': 'orbit.heterogeneous.region-cost.v13',
         'program_family_key': initial['program_family_key'],
         'execution_contract_identity': initial['execution_contract_identity'],
         'provenance': 'synthetic historical model for precedence tests only',
         'region_models': [],
         'boundary_model': {'knots': [{'bytes': 192, 'ms': .001},
                                      {'bytes': 384, 'ms': .002}]}}
for device in ('cpu', 'npu'):
    for count in (1, 2, 3):
        model['region_models'].append({'status': 'calibrated', 'device': device,
            'task_count': count, 'knots': [{'work_elements': 16 * count, 'ms': .01 * count},
                                        {'work_elements': 32 * count, 'ms': .02 * count}]})
save(model_path, model)
config.update(mechanism_parameters=str(parameter_path), heterogeneous_model=str(model_path))
save(config_path, config)
priced = export('priced')
require(priced['historical_empirical_cost']['status'] == 'calibrated' and
        priced['historical_empirical_cost']['estimated_ms'] > 0,
        'optional synthetic historical regional estimate remains separately inspectable')
require(priced['cost']['analytical'] is True and priced['cost']['estimated_ms'] is None and
        priced['cost']['status'] == 'uncalibrated',
        'primary mechanism cost never falls back to an available empirical whole-program number')
require(priced['mechanism_cost']['complete_ms'] is None and
        priced['mechanism_cost']['known_subtotal_ms'] > 0,
        'known API subtotal does not price unknown wait/CPU/overhead composites')
require(priced['graph_id'] == initial['graph_id'] and priced['id'] == initial['id'],
        'cost parameter metadata does not alter source/physical plan identities')
require(priced['mechanism_facts']['id'] == initial['mechanism_facts']['id'],
        'parameter prices do not alter extracted mechanism facts')
for item, cost in zip(priced['mechanism_facts']['items'], priced['mechanism_cost']['items']):
    if item['composite']:
        require(cost['duration_ms'] is None and cost['status'] == 'unknown',
                'opaque component remains unknown rather than zero')
    if item['descriptive_only']:
        require(cost['duration_ms'] is None and cost['status'] == 'included-in-composite',
                'native DMA and CPU access receive no duplicate elapsed-time charge')
export('reconstruct', expected=out / 'priced/manifest.json')


def invalid(name, mutate):
    changed = copy.deepcopy(parameters)
    mutate(changed['parameters'])
    path = out / f'{name}.parameters.json'
    save(path, changed)
    changed_config = dict(config, mechanism_parameters=str(path))
    path_config = out / f'{name}.config.json'
    save(path_config, changed_config)
    export(name, path_config, success=False)


invalid('reject-negative', lambda r: r[0].update(alpha_ms=-1))
invalid('reject-inverted-byte-domain', lambda r: r[0].update(min_bytes=10, max_bytes=1))
invalid('reject-duplicate', lambda r: r.append(copy.deepcopy(r[0])))
invalid('reject-invalid-digest', lambda r: r[0].update(raw_manifest_sha256='not-a-sha'))
invalid('reject-boundary-overlap', lambda r: r[0].update(measurement_boundary='whole-command'))
invalid('reject-opaque-bandwidth', lambda r: r[0].update(mechanism='npu.wait'))
invalid('reject-descriptive-double-charge', lambda r: r[0].update(mechanism='native.dma'))

for name, field, value in [('wrong-runtime', 'runtime_identity', 'unmatched-runtime'),
                           ('wrong-byte-domain', 'min_bytes', 8192)]:
    changed = copy.deepcopy(parameters)
    for row in changed['parameters']:
        row[field] = value
        if field == 'min_bytes':
            row['max_bytes'] = 16384
    path = out / f'{name}.parameters.json'
    save(path, changed)
    cfg = out / f'{name}.config.json'
    save(cfg, dict(config, mechanism_parameters=str(path)))
    unpriced = export(name, cfg)
    require(unpriced['mechanism_cost']['known_subtotal_ms'] == 0 and
            unpriced['mechanism_cost']['complete_ms'] is None and
            unpriced['cost']['estimated_ms'] is None,
            'unmatched runtime/domain has no known contribution, not a free full program')
    require(all(c['duration_ms'] is None for c in unpriced['mechanism_cost']['items']),
            'unmatched calibration never writes zero-duration API items')

for name, mutate in [
        ('reject-facts-recomputation', lambda m: m['mechanism_facts']['items'][0].update(bytes=1)),
        ('reject-parameter-identity', lambda m: m['mechanism_cost'].update(parameter_id='tampered'))]:
    changed = copy.deepcopy(priced)
    mutate(changed)
    path = out / f'{name}.expected.json'
    save(path, changed)
    export(name, expected=path, success=False)

summary = {'status': 'passed', 'checks': len(checks), 'check_names': checks,
           'tool_sha256': hashlib.sha256(tool.read_bytes()).hexdigest(),
           'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
           'graph_id': priced['graph_id'], 'facts_id': priced['mechanism_facts']['id'],
           'parameter_id': priced['mechanism_cost']['parameter_id'],
           'known_subtotal_ms': priced['mechanism_cost']['known_subtotal_ms'],
           'full_estimated_ms': None, 'hardware_executed': False,
           'device_submissions': 0,
           'scope': 'CPU native mechanism export contracts with synthetic unit parameters; no search or formal model'}
save(out / 'summary.json', summary)
print(json.dumps({'status': 'passed', 'checks': len(checks),
                  'known_subtotal_ms': summary['known_subtotal_ms']}))
