#!/usr/bin/env python3
"""Compiler-only same-domain/evaluator contracts for CP and public beam.

No plan/cost oracle is implemented here. Both algorithms invoke the compiler's
same complete public graph checker and mechanism evaluator.
"""
import argparse
import copy
import json
from pathlib import Path
import runpy
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    parser.add_argument('--solver-off', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(tempfile.mkdtemp(prefix='orbit-public-identity-v26-'))/'results'
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output/'gemm.mlir'
    source.write_text((ROOT/'test/TaskflowAIR/fixtures/gemm.mlir').read_text())
    config = runpy.run_path(str(ROOT/'test/TaskflowGeneric/program_all_search_v25.py'))['configuration']()
    config['program_search'].update(seconds=10, max_evaluations=1,
        max_values_per_dimension=2, max_routes=1, max_priorities=1,
        max_full_evaluations=1)
    records = []

    def run(name, cfg, algorithm='cp-sat-all'):
        folder = args.output/name
        folder.mkdir()
        cfg = copy.deepcopy(cfg)
        cfg['native_auto_candidates']['algorithm'] = algorithm
        path = folder/'target.json'
        path.write_text(json.dumps(cfg, indent=2)+'\n')
        command = [str(args.tool.resolve()), str(source), '--search', '--backend=air',
            '--max-rounds=4', '--beam-width=2', '--max-candidates=16',
            '--target-config='+str(path), '--search-output='+str(folder/'search')]
        (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        p = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (folder/'stdout').write_text(p.stdout)
        (folder/'stderr').write_text(p.stderr)
        assert p.returncode == 0, (name, p.returncode, p.stderr)
        report = json.loads((folder/'search/search.json').read_text())
        space = json.loads((folder/'search/space.json').read_text())
        assert report['domain_id'] == space['id']
        assert report['evaluator_id'] == space['evaluator_id']
        assert report['has_best'] is True, (name, report)
        assert report['physical_realization_checked'] is False
        assert report['unrestricted_optimum_proven'] is False
        assert space['schema'] == 'orbit.public-program-space.v26'
        records.append(dict(name=name, domain_id=report['domain_id'],
            evaluator_id=report['evaluator_id'], status=report['status'],
            evaluations=report['evaluations']))
        return report, space

    discovered, generated = run('bounded-discovery', config)
    assert discovered['domain_generation_complete'] is False
    # Freeze every dimension explicitly, replacing all old auto-truncated
    # catalogues. Only M/N remain binary; initialization is legally distributed.
    explicit = copy.deepcopy(config)
    domains = [dict(id=d['id'], choices=[d['choices'][0]['binding']])
               for d in generated['dimensions']]
    replacements = {
        'source.initialization': [dict(distribute=True)],
        'tiling.m': [dict(tile_axis=0, tile=2), dict(tile_axis=0, tile=4)],
        'tiling.n': [dict(tile_axis=1, tile=2), dict(tile_axis=1, tile=4)],
        'tiling.k': [dict(tile_axis=2, tile=4)],
    }
    for d in domains:
        if d['id'] in replacements:
            d['choices'] = replacements[d['id']]
    explicit['program_search'].update(dimensions=domains, max_evaluations=16)
    cp, cp_space = run('explicit-cp', explicit)
    beam, beam_space = run('explicit-beam', explicit, 'orbit-public-beam')
    assert cp['domain_generation_complete'] is True
    assert beam['domain_generation_complete'] is True
    assert cp_space == beam_space, 'strategy labels/budgets must not change domain identity'
    assert cp['evaluator_id'] == beam['evaluator_id']
    assert beam['algorithm'] == 'orbit-public-beam'
    assert beam['legacy_default_search'] is False
    assert beam['finite_domain_optimality_proved'] is False
    assert beam['status'] not in ['OPTIMAL', 'INFEASIBLE']
    assert beam['beam_selection_history']
    if args.solver_off:
        assert cp['status'] == 'UNAVAILABLE'
        assert cp['evaluations'] == 1
        assert cp['finite_domain_optimality_proved'] is False
    else:
        assert cp['status'] == 'OPTIMAL', cp
        assert cp['evaluations'] == 4
        assert cp['finite_domain_exhausted'] is True
        assert cp['finite_domain_optimality_proved'] is True
        assert cp['all_feasibility_known'] and cp['all_feasible_costs_known']
        assert beam['best_cost_ms'] + 1e-12 >= cp['best_cost_ms']

    budget_only = copy.deepcopy(explicit)
    budget_only['program_search'].update(max_evaluations=1, seconds=2)
    limited, _ = run('budget-is-not-objective', budget_only)
    assert limited['domain_id'] == cp['domain_id']
    assert limited['evaluator_id'] == cp['evaluator_id']
    assert limited['finite_domain_optimality_proved'] is False

    changed_domain = copy.deepcopy(explicit)
    for d in changed_domain['program_search']['dimensions']:
        if d['id'] == 'tiling.m':
            d['choices'][1]['tile'] = 3
    changed_domain['program_search']['max_evaluations'] = 1
    altered, _ = run('binding-changes-domain', changed_domain)
    assert altered['domain_id'] != cp['domain_id']
    assert altered['evaluator_id'] == cp['evaluator_id']

    changed_cost = copy.deepcopy(explicit)
    changed_cost['generic_air']['cost_model']['parameters']['compute.operation']['value'] = 2
    changed_cost['program_search']['max_evaluations'] = 1
    repriced, _ = run('price-changes-evaluator', changed_cost)
    assert repriced['evaluator_id'] != cp['evaluator_id']
    assert repriced['domain_id'] != cp['domain_id']

    # Original resource/memory/count/shape dimensions become null, while one
    # compound dimension binds those fields together. Old compatibility tables
    # must tolerate null operands and leave final validation to the checker.
    compound = copy.deepcopy(explicit)
    original = {d['id']: d['choices'][0] for d in domains}
    binding = {'task': 0}
    for suffix, key in [('resource', 'resource'), ('memory', 'memory'),
                        ('count', 'resource_count'), ('shape', 'resource_shape')]:
        dimension = 'task.0.'+suffix
        binding[key] = original[dimension][key]
        next(d for d in compound['program_search']['dimensions']
             if d['id'] == dimension)['choices'] = [None]
    compound['program_search']['dimensions'].append(
        dict(id='placement.compound', choices=[binding, None]))
    compound_cp, compound_space = run('compound-cp', compound)
    compound_beam, compound_beam_space = run('compound-beam', compound, 'orbit-public-beam')
    assert compound_space == compound_beam_space
    assert compound_cp['domain_generation_complete'] is True
    assert compound_beam['finite_domain_optimality_proved'] is False
    if not args.solver_off:
        assert compound_cp['status'] == 'OPTIMAL'
        assert compound_cp['evaluations'] == 8
        assert abs(compound_cp['best_cost_ms']-cp['best_cost_ms']) < 1e-12

    (args.output/'summary.json').write_text(json.dumps(dict(
        status='passed', hardware_executed=False, solver_off=args.solver_off,
        cases=records, scope='Same frozen finite domain and evaluator identity; explicit truncation reset; compound null; conservative beam proof status'), indent=2)+'\n')
    print(json.dumps(dict(status='passed', cases=len(records), hardware_executed=False)))


if __name__ == '__main__':
    main()
