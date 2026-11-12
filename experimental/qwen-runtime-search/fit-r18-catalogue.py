#!/usr/bin/env python3
"""Fit source-bound executable choices from complete, conserved Qwen traces.

Only training hook/region durations enter the objective. Whole-request latency
and heldout captures are never prices. Common C++ owns cover checking, joint
allocation feasibility, scheduling, and the actual ORBIT search.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    return dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fit(args):
    source, prefill, covers = map(load, (args.source, args.prefill_source, args.covers))
    for document in (source, prefill):
        require(document['source_id'] == digest({k: v for k, v in document.items()
                                                if k != 'source_id'}), 'Source hash mismatch')
    require(source['operators'][:len(prefill['operators'])] == prefill['operators']
            and source['values'][:len(prefill['values'])] == prefill['values'],
            'Whole source does not preserve exact prefill prefix')
    require(covers['source_id'] == source['source_id'], 'Cover/source mismatch')
    campaign = load(args.training / 'campaign-admitted.json')
    summary = load(args.training / 'training-admitted-summary.json')
    require(len(summary) == len(campaign['cohorts']) >= 13
            and all(row.get('passed') and row['returncode'] == 0 for row in summary),
            'All admitted complete-model numerical cohorts must pass before fitting')
    require({row['name'] for row in summary} == {row['name'] for row in campaign['cohorts']},
            'Training receipts do not match campaign')
    prices, counts = defaultdict(list), defaultdict(int)
    launches, receipts, conservation, fixed = defaultdict(set), [], [], []
    contexts, ffn_manifests, contracts = {}, {}, []

    def collect(key, ns):
        require(type(ns) is int and ns >= 0, 'Negative/noninteger exclusive duration')
        prices[key].append(ns)
        counts[key] += 1

    for cohort in campaign['cohorts']:
        command = cohort['command']
        directory = Path(command[command.index('--output') + 1])
        contract = load(directory / 'contract.json')
        require(contract['source_id'] == prefill['source_id'] and contract['outputs'] == 128
                and contract['threads'] == 4, 'Training execution contract mismatch')
        require(contract['source'] == prefill, 'Training source document differs from its admitted prefix')
        require(contract['plan_id'] == cohort['plan_id'], 'Training plan identity mismatch')
        for row in contract['implementation_files']:
            require(record(row['file'])['sha256'] == row['sha256'], 'Runtime implementation changed')
        plan = contract['plan']
        context = load(directory / 'decode-context.json')
        context_key = context['template_identity']['sha256']
        require(context['variant']['column_rotation'] == 0
                and context['variant']['parameters']['PROJ_RC_CACHE'] == 1,
                'Rejected or inactive decode choice must not enter catalogue')
        require(record(context['template_identity']['path'])['sha256'] == context_key,
                'Decode template changed')
        contexts[context_key] = context
        contracts.append(contract)
        rows = [json.loads(line) for line in (directory / 'measurements.jsonl').read_text().splitlines()]
        rows = [row for row in rows if not row['warmup']]
        require(len(rows) == 3, 'Exactly three measured training requests required')
        require(len({row['sample'] for row in rows}) == len(rows), 'Duplicate measured sample identity')
        receipts.append(dict(name=cohort['name'], directory=str(directory),
                             contract=record(directory / 'contract.json'),
                             measurements=record(directory / 'measurements.jsonl'),
                             capture=record(directory / 'capture.npz'), samples=[r['sample'] for r in rows]))
        for row in rows:
            profile, request = row['prefill_source_profile'], row['request_phases']
            require(profile['source_id'] == prefill['source_id'] and profile['plan_id'] == contract['plan_id'],
                    'Profile/source/plan binding mismatch')
            require(profile['exclusive_sum_equals_wall'] and request['exclusive_sum_equals_wall']
                    and row['outputs'] == 128 and len(row['tokens']) == 128
                    and len(row['decode_trace']) == 127, 'Incomplete/unconserved request')
            fused = profile['fused_covers']
            fused_names = [name for region in fused for name in region['operators']]
            nodes = profile['node_cost_ns']
            require(len(fused_names) == len(set(fused_names)) and not set(fused_names).intersection(nodes)
                    and set(fused_names).union(nodes) == {op['id'] for op in prefill['operators']},
                    'Missing, duplicate, or overlapping prefill source prices')
            require(sum(nodes.values()) + sum(region['exclusive_wall_ns'] for region in fused)
                    + profile['host_control_ns'] == profile['wall_ns'], 'Prefill phase conservation failed')
            for name, ns in nodes.items():
                variant = plan['r18']['gemm_variants'].get(name)
                key = ('gemm', name, variant['manifest_sha256']) if variant else (
                    'original', name, plan['placements'][name])
                collect(key, ns)
            for region in fused:
                layer = int(region['operators'][0].split('.')[1][1:])
                require(region['operators'] == [f'prefill.L{layer}.{role}' for role in
                        ('gate_up', 'gate_split', 'swiglu', 'down')], 'Unknown FFN source cover')
                config = plan['r18']
                proof = config['ffn_artifact']
                require(record(proof['manifest'])['sha256'] == proof['sha256'], 'FFN manifest changed')
                ffn_manifests[config['split_order']] = proof
                collect(('ffn', layer, config['retention'], config['split_order']), region['exclusive_wall_ns'])
            fixed_ns = profile['host_control_ns'] + request['clear_context_ns'] + request['capture_copy_ns'] + request['request_control_ns']
            fixed.append(fixed_ns)
            collect(('host', 'first-token-argmax', None, None), request['first_argmax_ns'])
            collect(('host', 'kv-handoff', None, None), request['handoff_ns'])
            checked = profile['wall_ns'] + sum(request[k] for k in
                      ('clear_context_ns', 'capture_copy_ns', 'request_control_ns', 'first_argmax_ns', 'handoff_ns'))
            for step, decode in enumerate(row['decode_trace']):
                phase = decode['phases']
                require(decode['step'] == step and phase['position'] == row['P'] + step
                        and phase['exclusive_sum_equals_wall'], 'Decode recurrence/phase mismatch')
                events = phase['region_events']
                require(len(events) == 1 and events[0]['region'] == dict(lo=0, hi=61)
                        and events[0]['instruction_patch_ns'] == 0, 'Unregistered decoder partition or double patch cost')
                event = events[0]
                service = event['submit_ns'] + event['wait_ns']
                require(event['service_ns'] == service and phase['host_before_ns'] + phase['host_after_ns']
                        + phase['host_control_ns'] + service == phase['wall_ns'], 'Decode phase overlap')
                require(decode['outer_dispatch_ns'] == phase['wall_ns'] + decode['outside_dispatch_wrapper_ns'],
                        'Dispatch wrapper overlap')
                collect(('decode', step, context_key), service)
                collect(('host', 'decode-host-before', step, context_key), phase['host_before_ns'])
                collect(('host', 'decode-host-after', step, context_key), phase['host_after_ns']
                        + phase['host_control_ns'] + decode['outside_dispatch_wrapper_ns'])
                collect(('host', 'decode-argmax', step, context_key), decode['argmax_ns'])
                checked += decode['outer_dispatch_ns'] + decode['argmax_ns']
            require(checked == request['wall_ns'], 'Complete request exclusive accounting failed')
            conservation.append(dict(cohort=cohort['name'], sample=row['sample'], wall_ns=checked,
                                     source_and_region_ns=checked-fixed_ns, fixed_control_ns=fixed_ns,
                                     exact_sum_equals_wall=True))
    require(len(contexts) == 2 and len(ffn_manifests) == 2, 'Missing real dataflow/order implementations')
    require(len({digest(c['producer']) for c in contracts}) == 1,
            'Training runtime producer revisions differ')
    require(len({digest(c['implementation_files']) for c in contracts}) == 1,
            'Training vendor implementation revisions differ')
    gemm_backends = {digest(c['plan']['r18']['gemm_backend']):c['plan']['r18']['gemm_backend']
                    for c in contracts if c['plan']['r18']['gemm_backend']}
    require(len(gemm_backends) == 1, 'GEMM hook backend identity is not unique')
    operators = source['operators']
    byid = {op['id']:i for i, op in enumerate(operators)}
    valueids = {value['id']: i for i, value in enumerate(source['values'])}
    producer = {value:op['id'] for op in operators for value in op['outputs']}
    readers = defaultdict(set)
    for op in operators:
        for value in op['inputs']:
            readers[value].add(op['id'])
    gemm = load(args.gemm_registration)
    architecture = copy.deepcopy(gemm['architecture_fragment'])
    for resource in architecture['resources']:
        if resource['kind'] == 'command':
            resource['kind'] = 'submission'
    capacity = load(args.capacity_registration) if args.capacity_registration else {}
    for witness in capacity.get('evidence', []):
        require(record(witness['file'])['sha256'] == witness['sha256'], 'Capacity implementation evidence changed')
    require(capacity, 'A source-backed capacity registration is mandatory')
    implementations, runtime, groups = [], {}, []
    covered, missing = set(), []

    def add(identifier, names, device, key, registration, template=None):
        indices = [byid[name] for name in names]
        member = set(names)
        ins = {v for name in names for v in operators[byid[name]]['inputs'] if producer.get(v) not in member}
        outs = {v for name in names for v in operators[byid[name]]['outputs']
                if source['values'][valueids[v]]['role'] == 'output' or readers[v] - member}
        require(indices == list(range(indices[0], indices[-1]+1)), 'Implementation source cover is not contiguous')
        samples = prices[key]
        require(len(samples) >= 3, 'Missing measured source-bound price: '+str(key))
        impl = copy.deepcopy(template or {})
        artifact = impl.get('artifact_identity') or digest(dict(registration=registration, code=contracts[0]['implementation_files']))
        impl.update(id=identifier, source_id=source['source_id'], source_operators=indices,
                    artifact_identity=artifact, entrypoint=impl.get('entrypoint', registration['kind']),
                    device=device, resource='cpu.host' if device == 'cpu' else 'npu.device', memory='runtime-abi',
                    inputs=sorted(valueids[v] for v in ins), outputs=sorted(valueids[v] for v in outs),
                    wall_ms=statistics.median(samples)/1e6, executable=True, launch_count=impl.get('launch_count', int(device=='npu')),
                    numeric_admitted=True, admission_status='admitted-complete-Qwen-training',
                    cost_boundary='inclusive-region-wall', transport_ownership='implementation-inclusive',
                    visibility_policy=impl.get('visibility_policy', 'host-materialized'),
                    training_price=dict(samples=len(samples), median_ns=statistics.median(samples),
                                        minimum_ns=min(samples), maximum_ns=max(samples), key=list(key)))
        # Use only the canonical integer interface; pending registrations carry
        # original string IDs as additional backend evidence, not another ABI.
        for field in ('operators', 'input_values', 'output_values'):
            impl.pop(field, None)
        variant = impl.setdefault('execution_variant', dict(
            runtime_context_identity=digest(dict(artifact=artifact, kind=registration['kind'])),
            dataflow='original-synchronous-source-call', residence_policy='original-runtime-ABI',
            replicas=1, work_distribution='single-invocation', resource_shape=[],
            resource_demands={'cpu.host':1, **({'npu.device':1, 'npu.command':1} if device=='npu' else {})},
            temporary_bytes={}, resident_allocations={}))
        if device == 'npu':
            variant['resource_demands'].update({'cpu.host':1, 'npu.device':1, 'npu.command':1})
        # Supplemental reservations must bind the actual runtime family. They
        # add private storage only; source ABI roots are already in the graph.
        supplements = [capacity.get('defaults', {})]
        if registration['kind'] == 'prefill-original':
            role = names[0].split('.')[-1]
            if device == 'npu':
                supplements.append(capacity.get('original_npu_default', {}))
            supplements.append(capacity.get('original_operators', {}).get(role, {}).get(device, {}))
        if registration['kind'] == 'decode-context':
            supplements.append(capacity.get('decode_contexts', {}).get(registration['decode_variant_dir'], {}))
        if registration['kind'] == 'prefill-fused-ffn':
            supplements.append(capacity.get('ffn_contexts', {}).get(impl['artifact_identity'], {}))
        for supplement in supplements:
            supplement = supplement.get('execution_variant', supplement)
            for field in ('resident_allocations', 'temporary_bytes', 'resource_demands'):
                for name, value in supplement.get(field, {}).items():
                    require(name not in variant.setdefault(field, {}) or variant[field][name] == value,
                            'Conflicting physical reservation '+name)
                    variant[field][name] = value
            if supplement.get('storage'):
                require('storage' not in impl, 'Duplicate source storage transformation')
                impl['storage'] = copy.deepcopy(supplement['storage'])
            if supplement.get('unproved_constraints'):
                variant.setdefault('unproved_constraints', []).extend(supplement['unproved_constraints'])
        if registration['kind'] == 'prefill-gemm':
            variant['resource_demands']['cpu.host'] = 1
        implementations.append(impl)
        runtime[identifier] = dict(registration, source_id=source['source_id'], operators=names,
                                   device=device, artifact_identity=artifact)
        covered.update(names)
        return identifier

    originals = {}
    for op in prefill['operators']:
        choices = []
        for device in op['attributes'].get('supported_devices', [op['attributes'].get('fixed_device', 'cpu')]):
            name = op['id']
            identifier = name+'.original.'+device
            add(identifier, [name], device, ('original', name, device),
                dict(kind='prefill-original', operator=name))
            originals[name,device] = identifier
            choices.append(identifier)
        originals[op['id']] = choices
    variants = defaultdict(list)
    for template in gemm['implementations']:
        config = template['runtime_config']
        name = config['operator']
        key = ('gemm', name, config['manifest_sha256'])
        if not prices[key]:
            missing.append(dict(implementation=template['id'], reason='no complete-Qwen hook price'))
            continue
        require(template.get('numeric_admitted'), 'Unvalidated compiled GEMM')
        # These packs are per-call temporaries and retain the original source
        # tensor. They are not the readonly-root replacement used by FFN.
        template['storage']['transient_weight_packing'] = template['storage'].pop('prepared_weight_copies')
        if template['execution_variant']['replicas'] > 1:
            partitions = []
            replicas = template['execution_variant']['replicas']
            for output in operators[byid[name]]['outputs']:
                shape = source['values'][valueids[output]]['shape']
                require(len(shape) == 2 and shape[1] % replicas == 0,
                        'Compiled output-N replica does not partition logical output')
                for replica in range(replicas):
                    partitions.append(dict(output_value=output, replica=replica,
                        offsets=[0, replica*(shape[1]//replicas)],
                        extents=[shape[0], shape[1]//replicas]))
            template['execution_variant']['work_partitions'] = partitions
        variants[name].append(add(template['id'], [name], 'npu', key,
            dict(kind='prefill-gemm', **config), template))
    ffn_register = module(Path(__file__).with_name('register-r18-ffn.py'), 'r18_register_ffn')
    fused = defaultdict(list)
    for retention, order in itertools.product(('retain','reload','release-after-call'), ('gate-up','up-gate')):
        proof = ffn_manifests[order]
        pending = ffn_register.register(prefill, load(proof['manifest']), list(range(36)), retention)
        for template in pending['implementations']:
            config = template['runtime_config']
            layer = config['layer']
            fused[layer].append(add(template['id'], template['operators'], 'npu',
                ('ffn', layer, retention, order), dict(kind='prefill-fused-ffn', **config, ffn_artifact=proof), template))
    ffn_names = set()
    for layer in range(36):
        names = [f'prefill.L{layer}.{role}' for role in ('gate_up','gate_split','swiglu','down')]
        ffn_names.update(names)
        choices = []
        options = [originals[name] + variants[name] for name in names]
        for combination in itertools.product(*options):
            choices.append(dict(id='unfused-'+digest(combination)[:20], implementations=list(combination)))
        choices += [dict(id=identifier, implementations=[identifier]) for identifier in fused[layer]]
        groups.append(dict(id=f'prefill.L{layer}.FFN-structure-and-mapping', choices=choices))
    for op in prefill['operators']:
        name = op['id']
        if name not in ffn_names:
            groups.append(dict(id=name, choices=[dict(id=identifier, implementations=[identifier])
                                                for identifier in originals[name]+variants[name]]))
    waves = {(x['step'],x['wave']):x['operators'] for x in covers['wave_operators']}
    contexts_group = []
    fixed_hosts = [group for group in covers['host_groups'] if group['kind'] in ('first-token-argmax','kv-handoff')]
    decode_hosts = [group for group in covers['host_groups'] if group not in fixed_hosts]
    for group in fixed_hosts:
        identifier = group['id']+'.cpu'
        add(identifier, group['operators'], 'cpu', ('host', group['kind'], group.get('step'), None),
            dict(kind='host', host_kind=group['kind'], step=group.get('step')))
        groups.append(dict(id=group['id'], choices=[dict(id='cpu',implementations=[identifier])]))
    for context_key, context in sorted(contexts.items()):
        chosen = []
        for step in range(127):
            names = [name for wave in range(61) for name in waves[step,wave]]
            identifier = f'decode.T{step}.context.'+context_key[:16]
            chosen.append(add(identifier, names, 'npu', ('decode', step, context_key),
                dict(kind='decode-context', step=step, lo=0, hi=61,
                     decode_variant_dir=context['directory'], template_identity=context['template_identity'],
                     variant_id=context['variant']['variant_id'])))
        for group in decode_hosts:
            identifier = group['id']+'.cpu.'+context_key[:16]
            chosen.append(add(identifier, group['operators'], 'cpu',
                ('host', group['kind'], group.get('step'), context_key),
                dict(kind='host', host_kind=group['kind'], step=group.get('step'),
                     decode_variant_dir=context['directory'])))
        contexts_group.append(dict(id='context-'+context_key, implementations=chosen))
    groups.append(dict(id='decode.global-compiled-dataflow-context', choices=contexts_group))
    require(covered == set(byid), 'Executable catalogue does not cover the whole operator graph')
    result = dict(schema='orbit.program-operator-implementations.v1', source_id=source['source_id'],
        architecture=architecture, implementations=implementations, choice_groups=groups,
        runtime_registration=dict(schema='orbit.qwen-r18-whole-runtime-registration.v1',
            source_id=source['source_id'], prefill_source_id=prefill['source_id'], implementations=runtime,
            wave_operators=covers['wave_operators'], host_groups=covers['host_groups'],
            gemm_backend=next(iter(gemm_backends.values())), code_files=contracts[0]['implementation_files']),
        fixed_application_invocation=dict(wall_ms=statistics.median(fixed)/1e6, samples_ns=fixed,
            boundary='exclusive prefill host control plus request clear/capture/control',
            selection_independent_approximation=True,
            limitation='Pooled fixed control is an approximation; heldout validates whole-request error'),
        training=receipts, training_campaign=record(args.training / 'campaign-admitted.json'),
        rejected_training=campaign.get('rejected_cohorts', []),
        trace_conservation_receipts=conservation, complete_plan_prepricing=False,
        missing_local_prices=0, excluded_unpriced_implementations=missing,
        producer=record(__file__), capacity_registration=record(args.capacity_registration) if args.capacity_registration else None,
        backend_registrations=dict(gemm=record(args.gemm_registration),
            ffn_producer=record(Path(__file__).with_name('register-r18-ffn.py')),
            ffn_manifests=ffn_manifests),
        search_scope=dict(temporal='registered gate/up order; original synchronous dependency order elsewhere',
            fusion_fission='four original FFN operators versus firmware-sequenced exact cover',
            tiling_shape_replica_placement='actual compiled GEMM variants only',
            dataflow='two distinct full decoder instruction streams plus actual GEMM/FFN dataflows',
            residence='actual FFN shared prepared BO, reload, or release after call; actual tile SRAM'))
    result['execution_registration_identity'] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','prefill-source','covers','training','gemm-registration','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--capacity-registration', type=Path)
    args = parser.parse_args()
    result = fit(args)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), implementations=len(result['implementations']),
                          groups=len(result['choice_groups']), registration=result['execution_registration_identity'])))


if __name__ == '__main__':
    main()
