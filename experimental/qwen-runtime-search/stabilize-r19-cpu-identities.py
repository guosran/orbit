#!/usr/bin/env python3
"""Restore parent CPU identities only for independently proved path-only copies.

Fresh pooled prices, actual receipts and async support remain those of the new
strict fit. This explicit versioned upgrade never admits an unmeasured choice.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def semantic_key(row):
    return (tuple(sorted(row['operators'])), *(row[k] for k in
            ('backend', 'input_policy', 'output_policy')))


def equivalent_proofs(old, new):
    """Only generated-file location may vary; all hashes and ABI facts agree."""
    def normalized(proof):
        result = copy.deepcopy(proof)
        for row in result['generated_code']:
            row.pop('file', None)
            row.pop('requested_file', None)
        return result
    require(normalized(old) == normalized(new),
            'Compiler proof differs beyond generated-file paths')


def contract(row, runtime=False):
    result = copy.deepcopy(row)
    for name in ('id', 'artifact_identity', 'compile_proof',
                 'actual_input_specializations', 'async_runtime_proof_identity'):
        result.pop(name, None)
    if not runtime:
        for name in ('wall_ms', 'training_price'):
            result.pop(name, None)
        result['execution_variant'].pop('async_runtime_proof_identity', None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'fresh', 'output', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.receipt.exists(), 'Output exists')
    path = Path(__file__).with_name('fit-r19-catalogue-v2.py')
    spec = importlib.util.spec_from_file_location('r19_identity_verifier', path)
    fit = importlib.util.module_from_spec(spec); spec.loader.exec_module(fit)
    old = fit.load(args.parent); fresh = fit.load(args.fresh)
    for cat in (old, fresh):
        require(cat['execution_registration_identity'] == fit.digest({k:v for k,v in cat.items()
                if k != 'execution_registration_identity'}), 'Catalogue identity mismatch')
    require(old['source_id'] == fresh['source_id'], 'Source differs')
    require(fresh['missing_local_prices'] == 0 and fresh['old_prices_reused'] is False,
            'Fresh catalogue lacks strict actual prices')
    support = fit.load(fit.verified(fresh['ready_runtime']['async_runtime_support_proof']))
    parent_support = fit.load(fit.verified(old['ready_runtime']['async_runtime_support_proof']))
    new_receipts = {row['name']:row for row in support['training_receipts']}
    checked_evidence = set()
    for row in parent_support['training_receipts']:
        require(row['name'] in new_receipts, 'Parent actual training cohort was lost')
        a, b = copy.deepcopy(row), copy.deepcopy(new_receipts[row['name']])
        # Retained backing uses manifest+summary rather than the earlier
        # manifest+individual-result receipt wrapper; the actual receipt bytes
        # and every numerical/contract/measurement record must still agree.
        for receipt in (a, b):
            for evidence in receipt['execution_receipt_provenance'].pop('evidence'):
                identifier = (evidence['file'], evidence['sha256'])
                if identifier not in checked_evidence:
                    fit.verified(evidence); checked_evidence.add(identifier)
        require(a == b, 'Parent actual training receipt content changed')
    old_runtime = old['runtime_registration']['implementations']
    result = copy.deepcopy(fresh); runtime = result['runtime_registration']['implementations']
    old_impl = {row['id']:row for row in old['implementations']}
    impl = {row['id']:row for row in result['implementations']}
    def indexed(rows):
        result = {}
        for identifier, row in rows.items():
            if row['kind'] != 'prefill-cpu-region':
                continue
            key = semantic_key(row)
            require(key not in result, 'Ambiguous CPU semantic identity')
            result[key] = identifier
        return result
    old_keys = indexed(old_runtime); new_keys = indexed(runtime)
    require(old_keys.keys() <= new_keys.keys(), 'Parent CPU choice was lost')
    require(len(old_keys) == 1000 and len(new_keys) == 3854,
            'Unexpected actual round-one CPU domain')
    remap = {}; witnesses = []
    for key, previous in old_keys.items():
        current = new_keys[key]
        if current == previous:
            continue
        require(previous not in runtime, 'Parent ID already has a different meaning')
        a, b = old_runtime[previous], runtime[current]
        require(a['backend'] == b['backend'] == 'inductor-cpu', 'Only compiled path copies may migrate')
        require(contract(a, True) == contract(b, True), 'Runtime source/ABI contract changed')
        require(contract(old_impl[previous]) == contract(impl[current]),
                'Implementation ABI, memory, or execution contract changed')
        require(a['actual_input_specializations'] == [a['compile_proof']] and
                b['actual_input_specializations'] == [b['compile_proof']], 'Ambiguous specialization')
        for record in (a['compile_proof'], b['compile_proof']):
            require(record in support['actual_cpu_preparation_proofs'],
                    'Proof absent from strict fresh actual preparation backing')
        require(a['compile_proof'] in parent_support['actual_cpu_preparation_proofs'],
                'Old proof absent from strict parent actual preparation backing')
        ap = fit.load(fit.verified(a['compile_proof'])); bp = fit.load(fit.verified(b['compile_proof']))
        equivalent_proofs(ap, bp)
        require(old_impl[previous]['compile_proof'] == ap and impl[current]['compile_proof'] == bp,
                'Embedded compile proof differs from actual file')
        for proof, reg in ((ap, a), (bp, b)):
            fit.verified(proof['producer'])
            for generated in proof['generated_code']:
                fit.verified(generated)
            artifact = fit.digest(dict(producer=proof['producer'], artifacts=proof['generated_code'],
                backend=reg['backend'], torch_version=proof['torch_version']))
            require(artifact == reg['artifact_identity'], 'Actual artifact identity differs')
        row = impl[current]; before_price = copy.deepcopy(row['training_price'])
        row.update(id=previous, artifact_identity=a['artifact_identity'], compile_proof=ap,
                   actual_input_specializations=copy.deepcopy(a['actual_input_specializations']))
        b.update(artifact_identity=a['artifact_identity'], compile_proof=copy.deepcopy(a['compile_proof']),
                 actual_input_specializations=copy.deepcopy(a['actual_input_specializations']))
        runtime[previous] = runtime.pop(current)
        require(row['training_price'] == before_price, 'Fresh price changed')
        remap[current] = previous
        witnesses.append(dict(fresh_id=current, stable_parent_id=previous,
            parent_proof=a['compile_proof'], fresh_proof=fresh['runtime_registration']['implementations'][current]['compile_proof'],
            specialization_id=ap['specialization_id'], generated_sha256=[x['sha256'] for x in ap['generated_code']],
            fresh_training_price=before_price, fresh_wall_ms=row['wall_ms']))
    require(len(remap) == 3, 'Unexpected number of path-only identity changes')
    result_by_id = {row['id']:row for row in result['implementations']}
    for group in result['choice_groups']:
        for choice in group['choices']:
            choice['implementations'] = sorted(
                [remap.get(name, name) for name in choice['implementations']],
                key=lambda name: (result_by_id[name]['source_operators'][0], name))
            choice['id'] = 'cover-' + fit.digest(choice['implementations'])[:24]
    require(len(result['implementations']) == len(fresh['implementations']) == 7730,
            'Candidate count changed')
    require(set(old_runtime) <= set(runtime), 'A parent implementation identity remains missing')
    require(indexed(runtime).keys() == new_keys.keys(), 'CPU semantic candidates changed')
    for row in fresh['implementations']:
        retained = result_by_id[remap.get(row['id'], row['id'])]
        require(retained['training_price'] == row['training_price'] and
                retained['wall_ms'] == row['wall_ms'], 'A fresh actual price changed')
    require(result['ready_runtime'] == fresh['ready_runtime'], 'Actual runtime support changed')
    migration = dict(schema='orbit.qwen-r19-cpu-path-identity-migration.v1',
        producer=fit.record(__file__), parent=fit.record(args.parent), fresh=fit.record(args.fresh),
        parent_cpu_choices=len(old_keys), fresh_cpu_choices=len(new_keys), witnesses=witnesses,
        prices_changed=False, candidates_removed=0, runtime_support_changed=False,
        rule='canonical parent artifact retained only after actual source/ABI/memory/proof equivalence; generated file paths alone may differ')
    result['producer'] = fit.record(__file__)
    result['strict_fit_producer'] = fresh['producer']
    result['identity_migration'] = migration
    result.pop('execution_registration_identity')
    result['execution_registration_identity'] = fit.digest(result)
    with args.output.open('x') as stream:
        json.dump(result, stream, separators=(',', ':'), allow_nan=False); stream.write('\n')
    receipt = dict(migration, output=fit.record(args.output),
        execution_registration_identity=result['execution_registration_identity'],
        implementations=len(result['implementations']), total_choices=sum(len(g['choices']) for g in result['choice_groups']))
    with args.receipt.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'witnesses'}))


if __name__ == '__main__':
    main()
