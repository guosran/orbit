#!/usr/bin/env python3
"""Reproducible compiler-only finite-domain oracle/ORBIT beam comparison.

Prices are assumed analytical quantities. CP-SAT proposes assignments, while the
complete public evaluator checks and prices them; exhaustive all-known coverage
proves the minimum in the explicitly frozen domain. No hardware execution.
"""
import argparse
import copy
import hashlib
import json
import itertools
from pathlib import Path
import shutil
import subprocess
import time

FIXTURES = Path(__file__).resolve().parent/'fixtures/oracle-v26'
CASES = [
    ('fork-shared-intermediate', 'fork-shared-intermediate'),
    ('fork-join-residual', 'fork-join-residual'),
    ('shared-input-three-branches', 'shared-input-three-branches'),
    ('fork-capacity-boundary', 'fork-capacity-boundary'),
    ('gemm-two-consumers', 'gemm-two-consumers'),
    ('gemm-reduction-64', 'gemm-reduction'),
    ('gemm-reduction-216', 'gemm-reduction'),
    ('gemm-reduction-trap', 'gemm-reduction'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tool', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, default=FIXTURES)
    p.add_argument('--cases', nargs='*')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tool = args.tool.resolve()
    summary = dict(schema='orbit.optimality-reference.v26',
        tool_sha256=digest(tool), hardware_executed=False,
        cost_kind='assumed complete public event-mechanism evaluator',
        oracle='CP-SAT branch-and-check exhaustive finite-domain certificate',
        comparator='public-domain adapter reusing ORBIT selectSearchBeam; not unmodified legacy search',
        rows=[])
    for name, source_name in CASES:
        if args.cases and name not in args.cases:
            continue
        folder = args.output/name
        folder.mkdir()
        source = folder/'source.mlir'
        shutil.copyfile(args.fixtures/(source_name+'.mlir'), source)
        base = json.loads((args.fixtures/(name+'.json')).read_text())
        base['program_search'].update(seconds=600, max_evaluations=20000,
            max_full_evaluations=8)
        row = dict(case=name, source_sha256=digest(source), algorithms=[])
        reference = None
        for label, algorithm, width, calls in [
            ('cp-sat-optimum', 'cp-sat-all', 1, 20000),
            ('beam1-budget64', 'orbit-public-beam', 1, 64),
            ('beam4-budget64', 'orbit-public-beam', 4, 64),
            ('beam16-budget64', 'orbit-public-beam', 16, 64),
            ('beam4-budget256', 'orbit-public-beam', 4, 256),
            ('beam16-budget256', 'orbit-public-beam', 16, 256),
        ]:
            run = folder/label
            run.mkdir()
            config = copy.deepcopy(base)
            config['native_auto_candidates']['algorithm'] = algorithm
            config['program_search']['max_evaluations'] = calls
            cfg = run/'target.json'
            cfg.write_text(json.dumps(config, indent=2)+'\n')
            command = [str(tool), str(source.resolve()), '--search',
                '--backend=air', '--max-rounds=32', '--beam-width='+str(width),
                '--target-config='+str(cfg.resolve()),
                '--search-output='+str((run/'search').resolve())]
            (run/'command.json').write_text(json.dumps(command, indent=2)+'\n')
            assert digest(tool) == summary['tool_sha256'], 'compiler changed before execution'
            started = time.monotonic()
            proc = subprocess.run(command, capture_output=True, text=True, timeout=630)
            assert digest(tool) == summary['tool_sha256'], 'compiler changed during execution'
            (run/'stdout').write_text(proc.stdout)
            (run/'stderr').write_text(proc.stderr)
            if proc.returncode:
                raise RuntimeError((name, label, proc.returncode, proc.stderr))
            r = json.loads((run/'search/search.json').read_text())
            space = json.loads((run/'search/space.json').read_text())
            assert r['has_best'] and r['best_cost_ms'] is not None, (name, label)
            assert r['all_feasibility_known'] and r['all_feasible_costs_known']
            assert r['domain_generation_complete'], (name, label)
            if reference is None:
                assert r['status'] == 'OPTIMAL' and r['finite_domain_optimality_proved']
                assert r['finite_domain_exhausted']
                master = {tuple(t['assignment']):t for t in r['evaluation_trace']}
                assert len(master) == len(r['evaluation_trace']), 'duplicate oracle assignment'
                compatible = set()
                tables = [(t['dimensions'], {tuple(a) for a in t['allowed_tuples']}) for t in space['constraints']]
                for a in itertools.product(*(range(len(d['choices'])) for d in space['dimensions'])):
                    if all(tuple(a[d] for d in axes) in rows for axes, rows in tables):
                        compatible.add(a)
                assert set(master) == compatible, 'oracle did not cover the entire compatible domain'
                assert min(t['cost_ms'] for t in master.values() if t['feasible']) == r['best_cost_ms']
                reference = r
                row.update(domain_id=r['domain_id'], evaluator_id=r['evaluator_id'],
                    optimum_ms=r['best_cost_ms'],
                    active_domains={d['id']:len(d['choices']) for d in space['dimensions'] if len(d['choices'])>1},
                    fixed_domains={d['id']:d['choices'][0]['binding'] for d in space['dimensions'] if len(d['choices'])==1})
            else:
                assert r['domain_id'] == reference['domain_id'], (name, label, 'domain')
                assert r['evaluator_id'] == reference['evaluator_id'], (name, label, 'evaluator')
                assert r['finite_domain_optimality_proved'] is False
                sampled = set()
                for t in r['evaluation_trace']:
                    key = tuple(t['assignment'])
                    assert key not in sampled, 'duplicate beam evaluation'
                    sampled.add(key)
                    expected = master[key]
                    for field in ['feasible', 'feasibility_known', 'cost_ms']:
                        assert t[field] == expected[field], (name, label, key, field)
            gap = r['best_cost_ms']/reference['best_cost_ms']-1
            assert gap >= -1e-10, (name, label, gap)
            metrics = dict(label=label, gap_percent=100*gap,
                hits_optimum=abs(gap)<1e-10, process_seconds=time.monotonic()-started)
            for key in ['status','best_cost_ms','evaluations','feasible_evaluations',
                'callback_ms','solve_ms','total_ms','model_variables',
                'model_constraints','process_peak_rss_kib','finite_domain_exhausted',
                'finite_domain_optimality_proved','reason']:
                metrics[key] = r.get(key)
            row['algorithms'].append(metrics)
            print(json.dumps(dict(case=name, **metrics)), flush=True)
        summary['rows'].append(row)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(status='passed',cases=len(summary['rows']))),flush=True)


if __name__ == '__main__':
    main()
