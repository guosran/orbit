#!/usr/bin/env python3
"""Seal explicit case receipts; never infer v32 native quantities from v13 facts.

preregister freezes a case specification and feature policy before labels exist.
build hashes sample bytes without parsing their timing values. Holdout sample
files must exist before build with the current frozen-index calibrator; collect
or seal them independently, and do not inspect their labels before model freeze.
"""
import argparse
import hashlib
import json
import pathlib
import re
RUNTIME = 'orbit.serial-prepared-typed-native-wait2.v32'
POLICY = {'application': {'reset': ['call', 'input_bytes'],
                 'control': ['call', 'configuration_count'],
                 'copy': ['call', 'output_bytes']},
 'cpu': {'body': ['call', 'source_operations']},
 'npu': {'sync_to': ['abi_bo_count', 'to_device_bytes'],
         'submit': ['configuration_count', 'configuration_argument_bindings'],
         'wait': ['configuration_count', 'source_operations'],
         'sync_from': ['call', 'from_device_bytes']}}

def read(p):
    return json.loads(pathlib.Path(p).read_text())

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def require(c, m):
    if not c:
        raise ValueError(m)

def write(p, d):
    pathlib.Path(p).write_text(json.dumps(d, indent=2, sort_keys=True, allow_nan=False) + '\n')

def absolute(p, base):
    p = pathlib.Path(p)
    return p.resolve() if p.is_absolute() else (base / p).resolve()

def seal(p):
    return {'path': str(p), 'sha256': sha(p)}

def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('phase', choices=['preregister', 'build'])
    a.add_argument('--spec', required=True, type=pathlib.Path)
    a.add_argument('--output', required=True, type=pathlib.Path)
    a.add_argument('--registration', type=pathlib.Path)
    x = a.parse_args()
    specpath = x.spec.resolve()
    s = read(specpath)
    cases = s['cases']
    policy = s.get('feature_sets', POLICY)
    require(cases and len({c['id'] for c in cases}) == len(cases), 'unique explicit case IDs required')
    require({c['split'] for c in cases} == {'train', 'holdout'}, 'explicit train and holdout required')
    for c in cases:
        require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', c['id']) is not None, 'case IDs must be safe file names')
        require(c['shape'] in ([16, 32, 64, 96] if c['split'] == 'train' else [48]), 'shape outside preregistered train/holdout split')
        require(c.get('ranking_group'), 'explicit comparable workload/input ranking_group required')
    # Registration binds the method before labels exist. Process repetition
    # improves uncertainty estimates, but cannot add independent design rows.
    registration = {
        'schema': 'orbit.heterogeneous.index-registration.v32',
        'spec': seal(specpath),
        'builder_sha256': sha(__file__),
        'feature_sets': policy,
        'acceptance': {
            'minimum_fresh_processes': 3,
            'median_absolute_relative_error_max': 0.20,
            'ranking_uncertainty': 'minmax of fresh-process hot-call medians',
        },
        'limitations': [
            'distinct source/tiling designs determine independent feature '
            'rows; process repeats do not increase design rank',
            'NNLS collinear terms are predictive only; '
            'no physical parameter identification',
            'missing local families/features and out-of-training ranges '
            'stay unknown',
            'v13 mechanism bytes omit some actual physical scratch sync; '
            'not a native feature source',
            'r3 wait quantities cover fixed whole-module implementation mix '
            'only; explicit per-task mixed implementations require a new '
            'runtime/tool/model contract',
        ],
    }
    if x.phase == 'preregister':
        require(not x.output.exists(), 'new output only')
        write(x.output, registration)
        return
    require(x.registration is not None, 'build requires frozen --registration')
    r = read(x.registration)
    require(r == registration, 'registration/spec/builder changed')
    out = x.output.resolve()
    require(not out.exists(), 'new output directory only')
    out.mkdir(parents=True)
    entries = []
    audit = []
    contracts = set()
    seen = {}
    for c in cases:
        m = absolute(c['measurement'], specpath.parent)
        f = absolute(c['features'], specpath.parent)
        facts = absolute(c['mechanism_facts'], specpath.parent)
        identity = read(m / 'identity.json')
        compiledpath = pathlib.Path(identity['compiled']).resolve()
        require(sha(compiledpath) == identity['compiled_sha256'], 'compiled SHA mismatch')
        compiled = read(compiledpath)
        manifestpath = pathlib.Path(compiled['manifest']).resolve()
        require(sha(manifestpath) == compiled['manifest_sha256'], 'manifest SHA mismatch')
        manifest = read(manifestpath)
        require(identity['runtime'] == compiled['runtime_identity'] == RUNTIME, 'runtime mismatch')
        require(identity['plan_id'] == compiled['plan_id'] == manifest['id'], 'plan mismatch')
        contract = manifest['execution_contract_identity']
        require(contract not in ('', 'unbound'), 'unbound contract')
        contracts.add(contract)
        d = read(f)
        if d.get('schema') == 'orbit.heterogeneous.edge-evaluation.v32':
            d = d['features']
        require(d.get('schema') == 'orbit.heterogeneous.edge-features.v32', 'native C++ edge-features required; v13 facts cannot be relabeled')
        require(d['parent_graph_id'] == manifest['graph_id'], 'exact parent graph mismatch')
        require(len(d['regions']) == len(manifest['regions']) == len(compiled['regions']), 'region count mismatch')
        total = 0
        for j, (fr, mr, cr) in enumerate(zip(d['regions'], manifest['regions'], compiled['regions'])):
            # Compare the ordered witness, not just source family or plan name.
            require(
                fr['region'] == mr['index'] == cr['index'] == j
                and fr['child_graph_id'] == mr['child_graph_id'] == cr['child_graph_id']
                and fr['device'] == mr['device'] == cr['device'],
                'exact ordered child witness mismatch',
            )
            if cr['device'] == 'npu':
                n = len(cr.get('configurations', [cr]))
                require(fr['features']['configuration_count'] == mr.get('physical_invocation_count', 1) == n, 'configuration count mismatch')
                total += n
                require(fr['features']['configuration_argument_bindings'] == n * fr['features']['abi_bo_count'], 'full ABI set_arg count mismatch')
            for names in policy[fr['device']].values():
                require(all((name in fr['features'] for name in names)), 'missing actual feature; do not fill unknown with zero')
        require(d['application']['configuration_count'] == total, 'application configuration count mismatch')
        mf = read(facts)
        require(mf['schema'] == 'orbit.mechanism-facts.v13' and mf['program_graph_id'] == d['parent_graph_id'], 'foreign mechanism graph')
        # v13 is sufficient for application copy ownership, not physical BO
        # sync or native implementation quantities. Never infer those here.
        sums = {}
        for name in ['host.input_reset', 'host.output_copy']:
            sums[name] = sum(
                item['bytes'] for item in mf['items']
                if item['mechanism'] == name
                and not item.get('descriptive_only', False)
                and item.get('covered_by', -1) == -1
            )
        require(
            sums['host.input_reset'] == d['application']['input_bytes']
            and sums['host.output_copy'] == d['application']['output_bytes'],
            'application copy byte disagreement',
        )
        featurepath = out / (c['id'] + '-edge-features.json')
        write(featurepath, d)
        entry = {k: c[k] for k in ['id', 'split', 'ranking_group']}
        entry['features'] = seal(featurepath)
        for role in ['identity', 'samples', 'result', 'process']:
            p = (m / (role + '.json')).resolve()
            require(p.exists(), 'missing receipt ' + str(p))
            require(p not in seen, 'shared measurement receipt: ' + str(p))
            seen[p] = c['id']
            entry[role] = seal(p)
        process = read(m / 'process.json')
        require(process.get('pid_scope') == 'measurement-python', 'formal process scope requires measurement-python')
        require(process.get('identity_sha256') == entry['identity']['sha256'], 'process identity mismatch')
        entries.append(entry)
        audit.append({
            'id': c['id'], 'shape': c['shape'],
            'native_feature_source': seal(f), 'mechanism_facts': seal(facts),
            'compiled': seal(compiledpath), 'manifest': seal(manifestpath),
            'v13_bytes_checked': sums, 'timing_labels_parsed': False,
        })
    require(len(contracts) == 1, 'multiple execution contracts cannot share model')
    index = {
        'schema': 'orbit.heterogeneous.calibration-index.v32',
        'runtime_identity': RUNTIME,
        'execution_contract_identity': next(iter(contracts)),
        'ranking_min_relative_gap': 0.05,
        'feature_sets': policy,
        'cases': entries,
    }
    write(out / 'index.json', index)
    write(out / 'intake.json', {
        'registration': seal(x.registration.resolve()),
        'cases': audit,
        'sample_bytes_hashed_only': True,
        'native_Cpp_prediction_and_hardware_speedup_not_verified': True,
    })
    print(out / 'index.json')
if __name__ == '__main__':
    main()
