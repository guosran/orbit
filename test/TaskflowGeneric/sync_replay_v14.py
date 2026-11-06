#!/usr/bin/env python3
"""CPU-only observed-event replay and API-cost boundary regression.

Synthetic event histories/parameters test the C++ consumer; no device invocation
or claim of hardware calibration occurs.
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
p.add_argument('--tool', type=Path,
               default=taskflow_tool('core').with_name('taskflow-sync-replay'))
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
checks = []


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def require(value, message):
    assert value, message
    checks.append(message)


full = {'bo': 'synthetic-bo', 'generation': 0, 'offset': 0, 'bytes': 1024}
context = {'call_form': 'implicit-full-buffer',
           'observation_boundary': 'detailed-compatibility',
           'runtime_identity': 'synthetic-runtime',
           'observer_identity': 'synthetic-observer'}
record = {
    'schema': 'orbit.sync-observation.stream.v14', 'sample_key': 'cpu-unit:0',
    'registration': {'bo': full['bo'], 'generation': 0,
                     'total_bytes': 1024, 'copies': [2, 7]},
    'scope': 'synthetic source history; not device execution',
    'raw_sample': {'sha256': 'a' * 64, 'path': 'synthetic-only'},
    'events': [
        {'id': 1, 'kind': 'HostWrite', 'range': full},
        {'id': 2, 'kind': 'SyncBegin', 'range': full,
         'operation': 'primer', 'direction': 'TO_DEVICE'},
        {'id': 3, 'kind': 'SyncComplete', 'range': full,
         'operation': 'primer', 'direction': 'TO_DEVICE'},
        {'id': 4, 'kind': 'SyncBegin', 'range': full,
         'operation': 'measured', 'direction': 'TO_DEVICE'},
        {'id': 5, 'kind': 'SyncComplete', 'range': full,
         'operation': 'measured', 'direction': 'TO_DEVICE'}],
    'queries': [{'before_event': 4, 'range': full, 'context': context,
                 'api_index': 0, 'descriptive_prefix': 'not-used-for-state'}]}


def run(name, rows, success=True, extra=(), raw=None):
    source = out / f'{name}.input.jsonl'
    source.write_text(raw if raw is not None else ''.join(
        json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
    output = out / f'{name}.output.jsonl'
    command = [str(a.tool.resolve()), '--input', str(source),
               '--output', str(output), *extra]
    save(out / f'{name}.command.json', command)
    result = subprocess.run(command, text=True, capture_output=True)
    (out / f'{name}.stdout').write_text(result.stdout)
    (out / f'{name}.stderr').write_text(result.stderr)
    save(out / f'{name}.result.json', {'returncode': result.returncode,
                                     'device_submissions': 0})
    require((result.returncode == 0) == success, f'{name}: expected success={success}: {result.stderr}')
    require(result.returncode >= 0, f'{name}: malformed data never aborts by signal')
    return [json.loads(line) for line in output.read_text().splitlines()] if success else []


result = run('valid', [record])[0]
answer = result['queries'][0]
require(answer['query']['state'] == 'post-TO-no-access' and answer['query']['known'],
        'successful primer gives authoritative pre-call post-TO state')
require(answer['source_query'] == record['queries'][0] and
        result['source_scope'] == record['scope'] and result['raw_sample'] == record['raw_sample'],
        'JSON cloning preserves original source/provenance without unchecked Expected')
require(result['sample_key'] == record['sample_key'] and not result['timing_used_as_state'],
        'sample identity retained and timing never used as state')
require(answer['cost'] is None and answer['cost_status'] == 'state-only',
        'no model means state-only with no zero price')
source_line = (out / 'valid.input.jsonl').read_text().splitlines()[0]
require(result['source_record_sha256'] == hashlib.sha256(source_line.encode()).hexdigest(),
        'exact source record bytes bound')
changed = copy.deepcopy(record)
changed['events'][0].update(start_ns=-999, end_ns=999999999999)
changed['queries'][0]['descriptive_prefix'] = 'a-different-test-name'
retimed = run('retimed', [changed])[0]
require(retimed['queries'][0]['query'] == answer['query'],
        'timestamps and descriptive test names do not classify state')
limited = run('limited', [record], extra=('--max-records', '1'),
              raw=json.dumps(record) + '\nnot valid JSON\n')
require(len(limited) == 1, 'max-records stops streaming before later invalid record')

bad = copy.deepcopy(record)
bad['events'][2]['id'] = 2
run('bad-order', [bad], False)
bad = copy.deepcopy(record)
bad['queries'][0]['range']['bytes'] = 1025
run('bad-query-extent', [bad], False)
bad = copy.deepcopy(record)
bad['registration']['total_bytes'] = 1023
run('bad-bo-extent', [bad], False)
bad = copy.deepcopy(record)
bad['queries'][0]['before_event'] = 99
run('missing-event', [bad], False)
bad = copy.deepcopy(record)
bad['events'][2]['kind'] = 'TimeMeansClean'
run('unknown-event', [bad], False)
run('bad-json', [], False, raw='not JSON\n')
pending = copy.deepcopy(record)
pending['queries'][0]['before_event'] = 5
pending_result = run('pending', [pending])[0]['queries'][0]['query']
require(pending_result['pending'] and not pending_result['known'],
        'query before completion remains pending/unknown')

domain = {'site': 'installed-bound-bo-input-loop', 'argument_ordinal': 0,
          'call_form': 'whole-implicit', 'observer_boundary': 'detailed-compatibility',
          'direction': 'TO_DEVICE', 'runtime_identity': context['runtime_identity'],
          'observer_identity': context['observer_identity'],
          'condition_identity': 'contexts=2;allocation=reused-prepared-bo',
          'allowed_states': ['post-TO-no-access'], 'min_bytes': 64, 'max_bytes': 4096,
          'min_count': 1, 'max_count': 1, 'alignment_bytes': 64}
model = {'schema': 'orbit.sync-api.model.v14', 'model_id': 'synthetic-unit-not-calibration',
         'units': 'ns', 'rows': [{'domain': domain, 'formula': 'fixed-api-site',
                                  'entry_ns': 0, 'call_ns': 500, 'cacheline_ns': 0,
                                  'provenance': 'synthetic CPU regression only',
                                  'raw_manifest_sha256': 'b' * 64}]}
model_path = out / 'synthetic-model.json'
save(model_path, model)
flags = ('--sync-model', str(model_path))
state_only = run('missing-cost-context', [record], extra=flags)[0]['queries'][0]
require(state_only['cost'] is None and state_only['cost_status'] == 'state-only',
        'model cannot fabricate missing site/ordinal/condition context')
priced = copy.deepcopy(record)
priced['queries'][0].update(site=domain['site'], argument_ordinal=0, count=1,
                            observed_extent_bytes=1024,
                            condition_identity=domain['condition_identity'])
actual = run('valid-cost', [priced], extra=flags)[0]['queries'][0]['cost']
require(actual['duration_ns'] == 500 and actual['status'] == 'predicted' and
        actual['parameter_file_sha256'] == hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'valid exact API request binds model file bytes and returns boundary-only prediction')
bad = copy.deepcopy(priced)
bad['queries'][0]['observed_extent_bytes'] = 2048
run('cost-extent-mismatch', [bad], False, extra=flags)
bad = copy.deepcopy(priced)
bad['queries'][0]['count'] = 'not-integer'
run('malformed-cost-context', [bad], False, extra=flags)
ood = copy.deepcopy(priced)
ood['queries'][0]['condition_identity'] = 'different-allocation-condition'
unpriced = run('ood-condition', [ood], extra=flags)[0]['queries'][0]['cost']
require(unpriced['duration_ns'] is None and unpriced['status'] == 'unknown',
        'valid mismatched conditions stay unknown, never zero')
save(out / 'summary.json', {'passed': True, 'checks': len(checks), 'assertions': checks,
                            'scope': 'synthetic CPU replay regression', 'device_submissions': 0})
print(json.dumps({'passed': True, 'checks': len(checks)}))
