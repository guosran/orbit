#!/usr/bin/env python3
"""Independently enumerate a PUBLIC space and replay its evaluator without CP-SAT.

The ON trace is used only for final pointwise comparison, never to generate
assignments, prune them, pick the optimum or provide any OFF cost.
"""
import argparse, concurrent.futures, copy, hashlib, itertools, json, math
import pathlib, shutil, subprocess, time


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def save(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def architecture(config):
    if config.get('program_graph', {}).get('adapter') == 'whole-task':
        return config['program_graph']['architecture']
    if 'architecture' in config.get('program_joint', {}):
        return config['program_joint']['architecture']
    air=config.get('generic_air',{})
    if 'memory_architecture' in air:
        return air['memory_architecture']
    # Immutable PUBLIC default target contract, not a planner or cost model.
    # Validate these expected facts against independently emitted ON/OFF target
    # exports below. Never infer the contract from a CP-selected assignment.
    return dict(domains=[dict(id='system',parent='',kind='system'),
        dict(id='npu',parent='system',kind='device'),
        dict(id='npu.tile(0,2)',parent='npu',kind='tile',coordinates=dict(column=0,row=2))],
        resources=[dict(id='npu.compute',kind='compute',device='npu',capacity=1,domain='npu.tile(0,2)'),
            dict(id='npu.dma',kind='transfer',device='npu',capacity=1,domain='npu'),
            dict(id='host-submission',kind='submission',device='host',capacity=1,domain='system'),
            dict(id='host.worker',kind='compute',device='host',capacity=1,domain='system')],
        memories=[dict(id='system_memory_bo',capacity_bytes=2**63-1,accessible_devices=['host'],
            domain='system',kind='system-bo',accessible_resources=['host.worker']),
            dict(id='aie_l1',capacity_bytes=air.get('capacity_bytes',24576),accessible_devices=['npu'],
            domain='npu.tile(0,2)',kind='local-sram',accessible_resources=['npu.compute'])],
        links=[dict(id='host_to_l1',source='system_memory_bo',target='aie_l1',resource='npu.dma',domain='npu'),
            dict(id='l1_to_host',source='aie_l1',target='system_memory_bo',resource='npu.dma',domain='npu')])


def check_architecture_export(expected, actual):
    defaults=dict(parent='',domain='',kind='',device='',coordinates={},capacity_bytes=None,
        accessible_devices=[],accessible_resources=[],alignment_bytes=1,capacity_known=True)
    fields=dict(domains=['id','parent','kind','coordinates'],
        resources=['id','kind','device','capacity','domain','shared_resources'],
        memories=['id','capacity_bytes','domain','kind','accessible_devices','accessible_resources','alignment_bytes','capacity_known'],
        links=['id','source','target','resource','domain'])
    for category,names in fields.items():
        left={x['id']:x for x in expected.get(category,[])}
        right={x['id']:x for x in actual.get(category,[])}
        assert set(left)==set(right), ('target identity set differs',category)
        for id,value in left.items():
            for field in names:
                fallback={} if field=='shared_resources' else defaults.get(field)
                assert value.get(field,fallback)==right[id].get(field,fallback), ('target fact differs',category,id,field)
    for link in actual.get('links',[]):
        assert link.get('min_bytes',0)>=0 and link.get('max_bytes',2**63-1)>=link.get('min_bytes',0)
        assert link.get('temporary_bytes',0)==0


def checked_tables(space, config, allow_legacy=False):
    """Reconstruct and compare every exported cheap table from public facts."""
    dimensions = space['dimensions']
    by_id = {d['id']: i for i, d in enumerate(dimensions)}
    assert len(by_id) == len(dimensions)
    target = architecture(config)
    resources = {r['id']: r for r in target['resources']}
    memories = {m['id']: m for m in target['memories']}
    def access(resource, memory):
        r, m = resources.get(resource), memories.get(memory)
        return bool(r and m and (resource in m['accessible_resources']
            if m.get('accessible_resources') else r.get('device') in m.get('accessible_devices', [])))
    def accept(kind, left, right):
        if left is None or right is None:
            return True
        if kind == 'resource-memory':
            r, m = left.get('resource'), right.get('memory')
            return r is None or m is None or access(r, m)
        if kind == 'resource-count':
            r, n = left.get('resource'), right.get('resource_count')
            return r is None or n is None or r in resources and 1 <= n <= resources[r]['capacity']
        if kind == 'count-shape':
            n, shape = left.get('resource_count'), right.get('resource_shape')
            return n is None or shape is None or not shape or all(x > 0 for x in shape) and math.prod(shape) == n
        if kind == 'memory-route':
            m, route = left.get('memory'), right.get('route')
            return m is None or route is None or route.get('target_memory') == m
        raise AssertionError(kind)
    expected = {}
    def relation(first, second, kind):
        if first not in by_id or second not in by_id:
            return
        a, b = by_id[first], by_id[second]
        expected[(a,b)] = {tuple([x,y]) for x, xc in enumerate(dimensions[a]['choices'])
            for y, yc in enumerate(dimensions[b]['choices']) if accept(kind, xc['binding'], yc['binding'])}
    import re
    tasks = sorted({int(m[1]) for name in by_id
        if (m := re.fullmatch(r'task\.(\d+)\.resource', name))})
    for task in tasks:
        prefix = f'task.{task}'
        relation(prefix+'.resource', prefix+'.memory', 'resource-memory')
        relation(prefix+'.resource', prefix+'.count', 'resource-count')
        relation(prefix+'.count', prefix+'.shape', 'count-shape')
        for name in by_id:
            if re.fullmatch(re.escape(prefix)+r'\.data\.\d+\.route', name):
                relation(prefix+'.memory', name, 'memory-route')
    if 'constraints' not in space:
        assert allow_legacy, 'space must export explicit cheap compatibility tables'
        return expected, 'independently-reconstructed-legacy-tables'
    actual = {}
    for table in space['constraints']:
        assert set(table) == {'dimensions', 'allowed_tuples'}
        axes = tuple(table['dimensions'])
        assert len(axes) == 2 and len(set(axes)) == 2 and all(0 <= a < len(dimensions) for a in axes)
        assert axes not in actual
        rows = [tuple(row) for row in table['allowed_tuples']]
        assert len(set(rows)) == len(rows)
        for row in rows:
            assert len(row) == len(axes)
            assert all(isinstance(n, int) and 0 <= n < len(dimensions[a]['choices']) for a, n in zip(axes,row))
        actual[axes] = set(rows)
    assert actual == expected, 'exported compatibility tables differ from independent public architecture derivation'
    return actual, 'exported-tables-fully-independently-validated'


def invoke(tool, source, config, folder, timeout):
    folder.mkdir()
    cfg = folder/'target.json'
    save(cfg, config)
    search = folder/'search'
    cmd = [str(tool), str(source), '--search', '--backend=air', '--max-rounds=1',
           '--max-candidates=32', '--target-config='+str(cfg), '--search-output='+str(search)]
    save(folder/'command.json', cmd)
    process = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    (folder/'stdout').write_text(process.stdout)
    (folder/'stderr').write_text(process.stderr)
    if process.returncode != 0:
        raise AssertionError(('OFF evaluator process failure', process.returncode, cmd, process.stderr))
    report = read(search/'search.json')
    assert report['status'] == 'UNAVAILABLE', ('requires solver-OFF binary', report['status'])
    assert report['solve_trace'] == [] and report['model_variables'] == 0
    assert report['evaluations'] == 1 and len(report['evaluation_trace']) == 1
    return report, read(search/'space.json'), search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', '--off-tool', dest='off_tool', type=pathlib.Path, required=True)
    parser.add_argument('--source', type=pathlib.Path, required=True)
    parser.add_argument('--config', type=pathlib.Path, required=True)
    parser.add_argument('--reference-search', '--on-search', dest='on_search', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--case-seconds', type=float, default=60)
    parser.add_argument('--point-timeout', type=float, default=10)
    parser.add_argument('--max-raw-points', type=int, default=1000000)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--allow-legacy-space', action='store_true')
    parser.add_argument('--keep-point-artifacts', action='store_true')
    args = parser.parse_args()
    assert args.workers >= 1 and args.case_seconds > 0
    args.off_tool, args.source = args.off_tool.resolve(), args.source.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config, space = read(args.config), read(args.on_search/'space.json')
    dimensions = space['dimensions']
    domain_hash = digest(space)
    semantic_config=copy.deepcopy(config)
    semantic_config.pop('program_search',None)
    semantic_config.pop('native_auto_candidates',None)
    shared_hash=digest(dict(source_sha256=file_digest(args.source),evaluation_config=semantic_config,space=space))
    tables, table_check=checked_tables(space,config,args.allow_legacy_space)
    on_report = read(args.on_search/'search.json')
    if space.get('id') is not None:
        assert on_report.get('domain_id')==space['id'], 'ON report names a different domain_id'
    assert on_report['finite_domain_exhausted'] is True
    assert on_report['finite_domain_optimality_proved'] is True
    assert on_report['all_feasibility_known'] and on_report['all_feasible_costs_known']
    protocol = dict(schema='orbit.public-oracle.off-audit.v26',
        source_sha256=file_digest(args.source), config_sha256=file_digest(args.config),
        shared_space_sha256=shared_hash, declared_space_artifact_sha256=domain_hash,
        compatibility_tables_checked=table_check, compatibility_table_count=len(tables), off_binary_sha256=file_digest(args.off_tool),
        on_report_sha256=file_digest(args.on_search/'search.json'),
        script_sha256=file_digest(__file__), assignment_generation='itertools Cartesian product of shared public domains; only independently validated exported cheap tables prune assignments',
        cost_source='one independent solver-OFF evaluator process per compatible assignment',
        trace_role='ON trace compared only after OFF enumeration; never assignment or cost ground truth',
        physical_realization_checked=False, workers=args.workers,
        case_seconds=args.case_seconds)
    save(args.output/'protocol.json', protocol)
    probe = copy.deepcopy(config)
    probe['program_search'].update(seconds=5, max_evaluations=1)
    _, off_space, probe_search = invoke(args.off_tool, args.source, probe, args.output/'space-probe', 20)
    assert digest(off_space) == domain_hash, 'ON and OFF regenerated different full spaces'
    on_baseline=read(args.on_search/'baseline.json').get('evaluation',{})
    off_baseline=read(probe_search/'baseline.json').get('evaluation',{})
    if 'program_graph' in on_baseline and 'program_graph' in off_baseline:
        on_arch=on_baseline['program_graph']['ledger']['memory_architecture']
        off_arch=off_baseline['program_graph']['ledger']['memory_architecture']
        assert digest(on_arch)==digest(off_arch), 'ON/OFF baseline public architecture exports differ'
        check_architecture_export(architecture(config),off_arch)
        protocol['independent_architecture_export_checked']=True
        protocol['architecture_export_sha256']=digest(off_arch)
    else:
        assert config.get('program_graph',{}).get('architecture') or config.get('program_joint',{}).get('architecture') or config.get('generic_air',{}).get('memory_architecture'), 'default architecture requires a valid independently emitted baseline'
        protocol['independent_architecture_export_checked']='explicit contract checked; baseline infeasible'
    save(args.output/'protocol.json',protocol)
    if args.probe_only:
        save(args.output/'result.json', dict(status='probe-passed', **protocol))
        print(json.dumps(dict(status='probe-passed', shared_space_sha256=shared_hash)))
        return
    count = math.prod(len(d['choices']) for d in dimensions)
    assert count <= args.max_raw_points, ('raw domain guard', count)
    # The entire assignment set exists before reading any ON evaluation rows.
    assignments = []
    for assignment in itertools.product(*(range(len(d['choices'])) for d in dimensions)):
        if all(tuple(assignment[a] for a in axes) in rows for axes,rows in tables.items()):
            assignments.append(assignment)
    save(args.output/'enumerated-domain.json', dict(raw_points=count,
        compatible_points=len(assignments), assignments_sha256=digest(assignments),
        shared_space_sha256=shared_hash))
    scratch = args.output/'points'
    scratch.mkdir()
    started = time.monotonic()

    def evaluate(item):
        index, assignment = item
        assert time.monotonic()-started <= args.case_seconds, 'case deadline reached before complete enumeration'
        fixed = copy.deepcopy(config)
        fixed['program_search'].update(seconds=5, max_evaluations=1,
            dimensions=[dict(id=d['id'], choices=[d['choices'][a]['binding']])
                        for d, a in zip(dimensions, assignment)])
        folder = scratch/f'{index:06d}'
        report, _, directory = invoke(args.off_tool, args.source, fixed, folder, args.point_timeout)
        result = report['evaluation_trace'][0]
        assert result['feasibility_known'], ('unknown feasibility', assignment, result)
        assert not result['feasible'] or result['cost_ms'] is not None, ('unknown feasible cost', assignment)
        cost = result['cost_ms']
        assert cost is None or math.isfinite(cost)
        row = dict(assignment=list(assignment), feasible=result['feasible'],
                   feasibility_known=result['feasibility_known'], cost_ms=cost,
                   cost_float_hex=None if cost is None else float(cost).hex(),
                   message=result['message'], assignment_config_sha256=digest(fixed))
        if result['feasible']:
            best = read(directory/'best.json')
            assert best['physical_realization_checked'] is False
            row['checked_graph_id'] = best['graph_id']
            row['timing_sha256'] = digest(best['timing'])
            assert best['estimated_ms'] == cost
        if not args.keep_point_artifacts:
            shutil.rmtree(folder)
        return row

    observed = {}
    with (args.output/'off-evaluations.jsonl').open('w') as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(evaluate, enumerate(assignments)):
                key = tuple(row['assignment'])
                assert key not in observed
                observed[key] = row
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
    elapsed = time.monotonic()-started
    assert len(observed) == len(assignments)
    # Only now inspect the solver-dependent trace for exact pointwise agreement.
    on_rows = on_report['evaluation_trace']
    on = {tuple(r['assignment']): r for r in on_rows}
    assert len(on) == len(on_rows), 'ON trace contains duplicate assignments'
    assert set(on) == set(observed), ('ON/OFF assignment coverage mismatch',
        len(set(on)-set(observed)), len(set(observed)-set(on)))
    for assignment, row in observed.items():
        other = on[assignment]
        assert row['feasibility_known'] == other['feasibility_known']
        assert row['feasible'] == other['feasible'], ('feasibility mismatch', assignment)
        assert row['cost_ms'] == other['cost_ms'], ('exact IEEE cost mismatch', assignment, row, other)
    feasible = [r for r in observed.values() if r['feasible']]
    assert feasible
    optimum = min(r['cost_ms'] for r in feasible)
    assert optimum == on_report['best_cost_ms']
    save(args.output/'result.json', dict(status='passed', **protocol,
        raw_points=count, compatible_points=len(assignments), feasible_points=len(feasible),
        pointwise_costs_exactly_equal=True, independently_enumerated_optimum_ms=optimum,
        observed_seconds=elapsed, within_case_budget=elapsed <= args.case_seconds,
        optimum_ties=sum(r['cost_ms'] == optimum for r in feasible),
        off_records_sha256=file_digest(args.output/'off-evaluations.jsonl')))
    print(json.dumps(read(args.output/'result.json'), indent=2))


if __name__ == '__main__':
    main()
