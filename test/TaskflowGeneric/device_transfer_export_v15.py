#!/usr/bin/env python3
"""CPU-only final-descriptor/public-graph cost export regression.

The final-IR fixture came from an actual compiler output. Artifact bytes, runtime
identities and model coefficients used here are synthetic, not executable device
evidence or calibration. No AIR compilation or hardware invocation occurs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Support'))
from orbit_test_paths import taskflow_tool

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tool', type=Path, default=taskflow_tool('core'))
p.add_argument('--transfer-tool', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
tool = a.tool.resolve()
transfer = (a.transfer_tool or tool.with_name('taskflow-transfer-opt')).resolve()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
checks = []


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


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
            f'{name}: expected success={success}: {result.stderr[-1500:]}')
    require(result.returncode >= 0, f'{name}: malformed input fails without process signal')


source = '''module {
 func.func @copy_path(%A:memref<1x2048xf32> {orbit.role="input",orbit.name="A"},
                     %C:memref<1x2048xf32> {orbit.role="output",orbit.name="C"}) attributes {orbit.noalias,llvm.emit_c_interface} {
  %done=taskflow.task @copy will_reads(%A:memref<1x2048xf32>) will_writes(%C:memref<1x2048xf32>) [original_read_memrefs(%A:memref<1x2048xf32>),original_write_memrefs(%C:memref<1x2048xf32>)] :(memref<1x2048xf32>,memref<1x2048xf32>)->memref<1x2048xf32> {
  ^bb0(%a:memref<1x2048xf32>,%c:memref<1x2048xf32>):
   affine.for %i=0 to 1 { affine.for %j=0 to 2048 {
    %v=affine.load %a[%i,%j]:memref<1x2048xf32>
    affine.store %v,%c[%i,%j]:memref<1x2048xf32>
   }}
   taskflow.yield done_writes(%c:memref<1x2048xf32>)
  }
  return
 }
}
'''
source_path = out / 'source.mlir'
source_path.write_text(source)
config = out / 'target.json'
save(config, {'generic_air': {'tile_sizes': [1, 256]}})
artifact = out / 'synthetic-artifact.bin'
artifact.write_bytes(b'CPU structural-audit unit fixture; not an ELF or executable evidence\n')
metadata = out / 'runtime-binding.json'
save(metadata, {key: 'synthetic-unit-' + key for key in (
    'runtime_identity', 'toolchain_identity', 'path_identity', 'observer_identity',
    'condition_identity')})
fixture = (Path(__file__).resolve().parents[1] /
           'TaskflowAIR/fixtures/serial-copy-final.mlir').read_text()


def rebind(final, plan):
    identities = {'orbit.mapping_plan': plan['plan_id'],
                  'orbit.program_graph': plan['program_graph_id'],
                  'orbit.source_fingerprint': plan['body_fingerprint']}
    for key, value in identities.items():
        final, count = re.subn(re.escape(key) + r' = "[^"]*"',
                               key + ' = "' + value + '"', final)
        require(count == 1, f'one {key} attribute rebound from fresh native plan')
    return final


def native_plan(name, path, target_config=config):
    plan_path = out / f'{name}.plan.json'
    run(name, [str(tool), str(path), '--generic-map', '--target-config=' + str(target_config),
               '--mapping-output=' + str(plan_path)])
    return load(plan_path)


plan = native_plan('native', source_path)
final = rebind(fixture, plan)
final_path = out / 'final.generic.mlir'
final_path.write_text(final)


def export(name, final_text=None, source_file=source_path, model=None, extra=(), success=True,
           target_config=config):
    path = final_path
    if final_text is not None:
        path = out / f'{name}.generic.mlir'
        path.write_text(final_text)
    command = [str(transfer), str(source_file), '--target-config', str(target_config),
               '--final-aie', str(path), '--artifact', str(artifact),
               '--runtime-binding', str(metadata), '--output', str(out / name), *extra]
    if model is not None:
        command += ['--cost-model', str(model)]
    run(name, command, success)
    return load(out / name / 'facts.json') if success else None


facts = export('initial')
binding = load(out / 'initial/binding.json')
witness = load(out / 'initial/witness.json')
receipt = load(out / 'initial/receipt.json')
cost = load(out / 'initial/cost.json')
require((facts['public_transfer_count'], facts['runtime_descriptor_count'],
         facts['core_transactions'], facts['requested_bytes'], facts['descriptor_bytes']) ==
        (16, 2, 16, 16384, 16384), 'coalesced descriptors are distinct from logical requests/core transactions')
require(binding['graph_id'] == plan['program_graph_id'] == receipt['graph_id'] and
        binding['native_plan_id'] == plan['plan_id'] == witness['native_plan_id'],
        'fresh source/native/public graph and final witness bind the same identity')
require(receipt['source_body_fingerprint'] == plan['body_fingerprint'] and
        receipt['artifact_sha256'] == hashlib.sha256(artifact.read_bytes()).hexdigest() and
        receipt['final_aie_file_sha256'] == hashlib.sha256(final_path.read_bytes()).hexdigest(),
        'receipt binds exact actual source and fixture/artifact bytes')
require(facts['physical_ddr_bytes'] is None and cost['duration_ns'] is None and
        not cost['complete_program_cost'], 'missing parameters remain unknown, not zero or physical bandwidth')
require(witness['segments'] == 8 and witness['tile_elements'] == 256 and
        len(witness['streams']) == 2 and
        {s['argument'] for s in witness['streams']} == {0, 1},
        'two real ABI paths and core geometry inspected')
require(set(facts['reservation_resources']) == {'npu.dma', 'npu.compute', 'host-submission'},
        'interval reserves declared DMA, compute and host submission resources')
require(len(facts['covered_compute_events']) == 8 and
        len({t['event'] for t in facts['transfers']}) == 16 and
        all(t['source_copies'] and t['target_copies'] and t['versions'] for t in facts['transfers']),
        'actual versioned source/copy events remain the interval authority')


def replace_once(old, new):
    require(old in final, 'mutation pattern exists: ' + old[:55])
    return final.replace(old, new, 1)


mutations = {
    'runtime-bd-length': ('static_len = 2048 : i32', 'static_len = 1024 : i32'),
    'core-bd-length': ('static_len = 256 : i32', 'static_len = 128 : i32'),
    'bd-offset': ('static_offset = 0 : i32', 'static_offset = 4 : i32'),
    'runtime-stride': ('static_strides = array<i64: 1>', 'static_strides = array<i64: 2>'),
    'core-view-stride': ('static_strides = array<i64: 256, 1>', 'static_strides = array<i64: 256, 2>'),
    'core-repeat': ('repeat_count = 0 : i32', 'repeat_count = 1 : i32'),
    'runtime-repeat': ('alloc = @air_channel_1, issue_token = true',
                       'alloc = @air_channel_1, issue_token = true, repeat_count = 1 : i32'),
    'missing-input-start': ('      "aiex.dma_start_task"(%13) : (index) -> ()\n', ''),
    'missing-output-await': ('      "aiex.dma_await_task"(%12) : (index) -> ()\n', ''),
    'wrong-shim-direction': ('channel_dir = 0 : i32, channel_index = 0 : i64', 'channel_dir = 1 : i32, channel_index = 0 : i64'),
    'wrong-flow-channel': ('source_channel = 0 : i32', 'source_channel = 1 : i32'),
    'wrong-allocation-symbol': ('alloc = @air_channel_1', 'alloc = @not_a_channel'),
    'wrong-wrapper-ABI': ('"aiex.run"(%arg0, %arg1)', '"aiex.run"(%arg1, %arg0)'),
    'output-token-missing': ('issue_token = true', 'issue_token = false'),
    'copy-index-not-covered': ('"memref.store"(%21, %20, %18, %arg5)', '"memref.store"(%21, %20, %18, %18)'),
    'core-copy-alias': ('"memref.store"(%21, %20, %18, %arg5)', '"memref.store"(%21, %19, %18, %arg5)'),
}
for name, (old, new) in mutations.items():
    export(name, replace_once(old, new), success=False)
export('stale-native-identity', final.replace(plan['plan_id'], '0' * 64), success=False)
export('stale-graph-identity', final.replace(plan['program_graph_id'], '0' * 64), success=False)
bad_source = out / 'arithmetic.mlir'
bad_source.write_text(source.replace('affine.store %v,',
                                    '%sum=arith.addf %v,%v:f32\n    affine.store %sum,'))
bad_plan = native_plan('arithmetic-native', bad_source)
export('arithmetic-implementation', rebind(fixture, bad_plan), bad_source, success=False)
alias_source = out / 'alias.mlir'
alias_source.write_text(source.replace('orbit.noalias,', ''))
export('missing-source-noalias', source_file=alias_source, success=False)
bad_metadata = out / 'bad-runtime-binding.json'
incomplete = load(metadata)
incomplete.pop('condition_identity')
save(bad_metadata, incomplete)
run('missing-runtime-condition', [str(transfer), str(source_path), '--target-config', str(config),
    '--final-aie', str(final_path), '--artifact', str(artifact), '--runtime-binding', str(bad_metadata),
    '--output', str(out / 'missing-runtime-condition')], False)

domain = {key: binding[key] for key in ('runtime_identity', 'toolchain_identity', 'path_identity',
                                      'observer_identity', 'condition_identity', 'completion_boundary')}
domain.update(architecture_identity=facts['architecture_identity'], implementation='identity-copy-f32',
              min_bytes=1024, max_bytes=32768, min_segments=1, max_segments=16,
              min_tile_elements=16, max_tile_elements=4096)
row = {'domain': domain, 'entry_ns': 1, 'segment_ns': 0, 'byte_ns': .5,
       'provenance': {'kind': 'synthetic-unit-test', 'source': 'not hardware calibration'},
       'raw_manifest_sha256': 'b' * 64}
model = {'schema': 'orbit.device-transfer.model.v15', 'model_id': 'synthetic-unit-only',
         'units': 'ns', 'rows': [row]}
model_path = out / 'synthetic-model.json'
save(model_path, model)
export('priced', model=model_path)
priced = load(out / 'priced/cost.json')
require(priced['duration_ns'] == 8193 and priced['status'] == 'predicted' and
        priced['parameter_file_sha256'] == hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'synthetic entry+bytes composite binds exact parameter file')
require(not priced['separately_charge_native_dma'] and len(priced['covered_events']) == 24,
        'native DMA and actual copy computations are owned once')
hook_config = out / 'target-with-device-transfer-model.json'
hook_target = load(config)
hook_target['device_transfer_model'] = str(model_path)
save(hook_config, hook_target)
hook_plan = native_plan('config-hook-native', source_path, hook_config)
require(hook_plan['plan_id'] == plan['plan_id'] and
        hook_plan['program_graph_id'] == plan['program_graph_id'] and
        hook_plan['body_fingerprint'] == plan['body_fingerprint'],
        'cost-only target-config path leaves source/native/public graph identity unchanged')
export('config-hook-priced', target_config=hook_config)
require(load(out / 'config-hook-priced/cost.json') == priced,
        'top-level device_transfer_model exactly equals explicit --cost-model known estimate')
for sidecar in ('facts.json', 'binding.json', 'witness.json', 'receipt.json', 'program-graph.json'):
    require(load(out / 'config-hook-priced' / sidecar) == load(out / 'priced' / sidecar),
            'config model hook preserves exact physical/public evidence: ' + sidecar)
missing_model_config = out / 'target-with-missing-model.json'
missing_target = load(config)
missing_target['device_transfer_model'] = str(out / 'model-does-not-exist.json')
save(missing_model_config, missing_target)
export('config-hook-missing-model', target_config=missing_model_config, success=False)
export('explicit-model-precedence', model=model_path, target_config=missing_model_config)
require(load(out / 'explicit-model-precedence/cost.json') == priced,
        'explicit cost-model keeps documented precedence over target-config model path')
export('duplicate-submit-price', model=model_path, extra=('--submit-already-priced',), success=False)
export('duplicate-wait-price', model=model_path, extra=('--wait-already-priced',), success=False)
unknown = copy.deepcopy(model)
unknown['rows'][0]['domain']['runtime_identity'] = 'different-runtime'
unknown_path = out / 'unknown-runtime-model.json'
save(unknown_path, unknown)
export('unknown-runtime', model=unknown_path)
unknown_cost = load(out / 'unknown-runtime/cost.json')
require(unknown_cost['duration_ns'] is None and unknown_cost['status'] == 'unknown',
        'exact runtime mismatch returns unknown rather than zero')
unknown = copy.deepcopy(model)
for key in ('entry_ns', 'segment_ns', 'byte_ns'):
    unknown['rows'][0][key] = None
unknown_path = out / 'unknown-coefficients.json'
save(unknown_path, unknown)
export('unknown-coefficients', model=unknown_path)
require(load(out / 'unknown-coefficients/cost.json')['duration_ns'] is None,
        'matching domain without fitted coefficients stays unknown')
for name, mutate in (
        ('negative-coefficient', lambda m: m['rows'][0].update(byte_ns=-1)),
        ('bad-evidence-digest', lambda m: m['rows'][0].update(raw_manifest_sha256='wrong')),
        ('wrong-units', lambda m: m.update(units='ms'))):
    invalid = copy.deepcopy(model)
    mutate(invalid)
    path = out / f'{name}.model.json'
    save(path, invalid)
    export(name, model=path, success=False)
save(out / 'summary.json', {'passed': True, 'checks': len(checks), 'assertions': checks,
    'tool_sha256': hashlib.sha256(tool.read_bytes()).hexdigest(),
    'transfer_tool_sha256': hashlib.sha256(transfer.read_bytes()).hexdigest(),
    'scope': 'synthetic CPU structural/provider regression; not executable calibration',
    'device_submissions': 0})
print(json.dumps({'passed': True, 'checks': len(checks)}))
