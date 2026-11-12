#!/usr/bin/env python3
"""Compiler-only contracts for public all-dimension branch-and-check search.

No planner, search, target architecture or cost predictor is implemented here.
The tiny GEMM fixture exercises the pre-existing native reduction path. Prices
are explicit synthetic mechanism parameters, never measured NPU quantities.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def configuration():
    parameters = {}
    for key, value, unit in [
        ('compute.startup', 4, 'ns'),
        ('compute.operation', 1, 'ns'),
        ('compute.memory_access', .5, 'ns'),
        ('configuration.startup', 1000, 'ns'),
        ('host.submit', 100, 'ns'),
        ('host.complete', 100, 'ns'),
        ('host.visibility', 10, 'ns'),
        ('communication.startup', 100, 'ns'),
        ('communication.bandwidth', 8, 'bytes/ns'),
    ]:
        parameters[key] = dict(value=value, unit=unit,
            provenance=dict(kind='assumed', source='v25 CLI synthetic contract'))
    return dict(
        generic_air=dict(capacity_bytes=24576, cost_model=dict(
            kind='event-mechanism', schema='orbit.direct-air.cost.v1',
            mode='estimate', model_id='v25-cli-contract',
            implementations=['scalar'], boundary='one-invocation',
            parameters=parameters)),
        native_auto_candidates=dict(mode='joint', algorithm='cp-sat-all'),
        program_search=dict(seconds=5, max_evaluations=1,
            max_variables=100000, max_constraints=500000,
            max_values_per_dimension=2, max_routes=2, max_priorities=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--solver-off', action='store_true')
    args = parser.parse_args()
    if args.output is None:
        args.output=Path(tempfile.mkdtemp(prefix='orbit-public-search-'))/'results'
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / 'gemm.mlir'
    source.write_text((ROOT/'test/TaskflowAIR/fixtures/gemm.mlir').read_text())
    results = []

    def run(name, config):
        folder = args.output/name
        folder.mkdir()
        cfg = folder/'config.json'
        cfg.write_text(json.dumps(config, indent=2)+'\n')
        output = folder/'search'
        command = [str(args.tool.resolve()), str(source.resolve()),
            '--search', '--backend=air', '--max-rounds=1', '--max-candidates=32',
            '--target-config='+str(cfg.resolve()),
            '--search-output='+str(output.resolve())]
        (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        process = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (folder/'stdout').write_text(process.stdout)
        (folder/'stderr').write_text(process.stderr)
        assert process.returncode >= 0, (name, 'compiler crash', process.stderr)
        report = json.loads((output/'search.json').read_text()) if (output/'search.json').exists() else None
        return process, report, output

    base = configuration()
    process, report, output = run('gemm-baseline', base)
    assert process.returncode == 0, process.stderr
    assert report['algorithm'] == 'joint-cp-sat-branch-and-check'
    assert report['complete_plan_prepricing'] is False
    assert report['has_best'] is True
    assert report['evaluations'] == 1
    assert report['physical_realization_checked'] is False
    dimensions = {d['id']: d for d in report['dimensions']}
    required = {'source.fission', 'source.initialization', 'tiling.m', 'tiling.n',
        'tiling.k', 'implementation', 'fusion.residency.traversal',
        'replica.partition', 'copies.retention', 'buffering.slots',
        'temporal.priority', 'layout.dtype'}
    assert required <= dimensions.keys(), dimensions.keys()
    assert all(d['status'] in {'searched', 'singleton_by_capability',
        'fixed_by_source', 'unsupported_realization'} for d in dimensions.values())
    assert dimensions['layout.dtype']['status'] in {
        'fixed_by_source', 'unsupported_realization'}, dimensions['layout.dtype']
    best = json.loads((output/'best.json').read_text())
    assert best['program_graph']['schema'] == 'orbit.program-graph.v1'
    assert 'arith.mulf' in best['program_graph']['computation_mlir']
    assert 'affine.for' in best['program_graph']['computation_mlir']
    assert best['source_fingerprint']
    assert best['native_reference'] is not None
    assert report['finite_domain_optimality_proved'] is False
    if args.solver_off:
        assert report['status'].lower() == 'unavailable'
    results.append(dict(case='gemm-baseline', passed=True))

    cfg = copy.deepcopy(base)
    cfg['program_search']['dimensions'] = [
        dict(id=('tiling.m', 'tiling.n', 'tiling.k')[axis],
             choices=[dict(tile_axis=axis, tile=extent)])
        for axis, extent in enumerate([2, 2, 4])]
    cfg['program_search']['dimensions'].append(
        dict(id='source.initialization', choices=[dict(distribute=True)]))
    process, report, output = run('simultaneous-mnk-reduction', cfg)
    assert process.returncode == 0 and report['has_best'], process.stderr
    best = json.loads((output/'best.json').read_text())
    assert best['native_reference']['tile_sizes'] == [2, 2, 4], best['native_reference']
    assert 'arith.mulf' in best['program_graph']['computation_mlir']
    results.append(dict(case='simultaneous-mnk-reduction', passed=True))

    # Every automatically truncated non-singleton domain is binary here. A
    # coverage proposal must change all of them, rather than enumerate only
    # the last route variable. Invalid combinations are still checked attempts.
    if not args.solver_off:
        cfg = copy.deepcopy(base)
        cfg['program_search']['max_evaluations'] = 2
        process, report, _ = run('coverage-before-last-axis-exhaustion', cfg)
        assert process.returncode == 0, process.stderr
        assert report['evaluations'] == 2, report
        assert 'coverage' in report['solver_objective']
        coverage = {r['dimension']: r['choice_exposures'] for r in report['choice_coverage']}
        for d in report['dimensions']:
            if d['choices'] == 2:
                assert all(n > 0 for n in coverage[d['id']]), (d, coverage[d['id']])
        results.append(dict(case='coverage-before-last-axis-exhaustion', passed=True))

    # These are intentionally malformed, unused or contradictory PUBLIC
    # decisions. Rejection can be a configuration error or an infeasible
    # baseline evaluation, but never a feasible silently unchanged plan.
    negatives = {
        'unknown-top-level': [{'unknown_public_choice': True}],
        'orphan-resource': [{'resource': 'npu.compute'}],
        'orphan-memory': [{'memory': 'aie_l1'}],
        'orphan-resource-count': [{'resource_count': 2}],
        'orphan-resource-shape': [{'resource_shape': [1, 2]}],
        'orphan-tile': [{'tile': 2}],
        'orphan-lanes': [{'lanes': 8}],
        'orphan-task': [{'task': 0}],
        'negative-partition': [{'partition': -1}],
        'wrong-boolean-type': [{'fission': 'true'}],
        'wrong-task-type': [{'task': 'zero', 'resource': 'npu.compute'}],
        'unconsumed-estimate': [{'estimate': {'not_a_compute_estimate': True}}],
        'unknown-mapping-field': [{'mapping': dict(residency='phase-local',
            grouping='task-shape-stages', data_organization='phase-local',
            silently_ignored=True)}],
        'wrong-mapping-type': [{'mapping': {'storage': 'not-an-integer'}}],
        'unsupported-layout': [{'layout': 'column_major'}],
        'unsupported-implementation': [{'implementation': 'independent-lane-f32', 'lanes': 8}],
        'unknown-route-field': [{'route': dict(storage=0, consumer_task=0,
            source_memory='system_memory_bo', target_memory='aie_l1',
            links=['host_to_l1'], silently_ignored=True)}],
        'contradictory-tile-bindings': [
            {'tile_axis': 0, 'tile': 1}, {'tile_axis': 0, 'tile': 2}],
    }
    for name, bindings in negatives.items():
        cfg = copy.deepcopy(base)
        cfg['program_search']['dimensions'] = [
            dict(id=('replica.partition' if name == 'negative-partition'
                     else f'contract.{name}.{i}'), choices=[binding])
            for i, binding in enumerate(bindings)]
        process, report, output = run(name, cfg)
        if report is not None:
            assert not report['has_best'], (name, 'silently accepted binding', report)
            assert report['feasible_evaluations'] == 0, (name, report)
            assert not (output/'best.json').exists(), name
            messages = [r['message'] for r in report['evaluation_trace']]
            assert messages or report['reason'], (name, 'no rejection diagnostic')
        else:
            assert process.returncode != 0 and process.stderr.strip(), (name, process.stderr)
        results.append(dict(case=name, passed=True, returncode=process.returncode))

    # JSON integers retain 64-bit precision until bounds are checked. These
    # values used to wrap into valid 32-bit worker/seed settings.
    for key, value in [('workers', -1), ('workers', 0), ('workers', 17),
                       ('workers', 4294967297), ('seed', -1),
                       ('seed', 2147483648), ('seed', 4294967296)]:
        cfg = copy.deepcopy(base)
        cfg['program_search'][key] = value
        name = f'control-{key}-{value}'
        process, report, output = run(name, cfg)
        assert process.returncode != 0 and report is None, (name, report)
        assert 'workers must be in [1,16]' in process.stderr, process.stderr
        assert not (output/'candidate-0').exists(), name
        results.append(dict(case=name, passed=True))
    cfg = copy.deepcopy(base)
    cfg['program_search'].update(workers=1, seed=2147483647)
    process, report, output = run('seed-int-max', cfg)
    assert process.returncode == 0 and report['has_best'], process.stderr
    assert report['evaluations'] == 1, report
    results.append(dict(case='seed-int-max', passed=True))

    cfg = copy.deepcopy(base)
    cfg['program_search']['dimensions'] = [
        dict(id='contract.duplicate', choices=[None]),
        dict(id='contract.duplicate', choices=[None])]
    process, report, output = run('duplicate-custom-id', cfg)
    assert process.returncode != 0, 'duplicate explicit domains must not silently replace each other'
    assert process.stderr.strip()
    assert not (output/'best.json').exists()
    results.append(dict(case='duplicate-custom-id', passed=True))

    (args.output/'summary.json').write_text(json.dumps(dict(
        status='passed', hardware_executed=False, cases=results,
        scope='Strict public binding consumption, coverage, native reduction source, bounded fallback'), indent=2)+'\n')
    print(json.dumps(dict(status='passed', cases=len(results), hardware_executed=False)))


if __name__ == '__main__':
    main()
