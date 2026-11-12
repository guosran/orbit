#!/usr/bin/env python3
"""CPU-only error-path contracts: price failure is not graph infeasibility."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    parser.add_argument('--solver-off', action='store_true')
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='orbit-native-errors-v31-'))
    source = Path(__file__).resolve().parents[1] / 'TaskflowAIR/fixtures/heterogeneous.mlir'
    target = dict(id='synthetic-errors-only', toolchain='compiler-only',
                  runtime='not-hardware-calibrated', boundary='prepared-command-host-to-host')
    base = dict(generic_air=dict(cost_target=target, cost_model=dict(
        schema='orbit.direct-air.cost.v1', kind='linear-composite', calibration='uncalibrated')),
        native_auto_candidates=dict(mode='joint', algorithm='cp-sat-all'),
        program_search=dict(seconds=5, max_evaluations=1,
                            max_values_per_dimension=1, max_routes=1, max_priorities=1))
    checks = 0

    def run(name, config, valid=True, failed_evidence=None):
        nonlocal checks
        folder = root / name
        folder.mkdir()
        if failed_evidence:
            destination = folder / 'search' / failed_evidence
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to('/dev/full')
        path = folder / 'config.json'
        path.write_text(json.dumps(config))
        command = [str(args.tool.resolve()), str(source), '--search', '--backend=air',
                   '--target-config=' + str(path), '--search-output=' + str(folder / 'search')]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (folder / 'log').write_text(result.stdout + result.stderr)
        assert (result.returncode == 0) == valid, (name, result.returncode, result.stderr)
        if not valid:
            assert result.returncode > 0, (name, 'must report the error, not abort', result.returncode)
        checks += 1
        report = folder / 'search/search.json'
        return json.loads(report.read_text()) if report.exists() else None, result.stderr

    # Invalid model structure must be rejected before it can exclude the only
    # assignment and become a false proof of physical graph infeasibility.
    for name, field, value in [('schema', 'schema', 'wrong-schema'),
                               ('calibration', 'calibration', 'invalid')]:
        bad = copy.deepcopy(base)
        bad['generic_air']['cost_model'][field] = value
        report, stderr = run('bad-' + name, bad, False)
        assert report is None and ('schema' if name == 'schema' else 'calibration') in stderr
        checks += 1

    # Later source-dependent provider validation also must remain unknown. A
    # malformed coefficient cannot be evaluated, irrespective of source price.
    bad = copy.deepcopy(base)
    bad['generic_air']['cost_model'].update(
        calibration='calibrated', model_id='invalid-coefficient', model_version='1',
        training_manifest_digest='a' * 64, boundary=target['boundary'], target=target,
        domain=dict(source_fingerprints=['0' * 64], implementations=['scalar'],
                    operations=[], feature_ranges=dict(command_count=[1, 1])),
        terms=[dict(feature='command_count', coefficient_ms='invalid')])
    report, _ = run('provider-validation', bad)
    assert report['status'] == ('UNAVAILABLE' if args.solver_off else 'UNKNOWN'), report
    assert report['unknown_feasibility_evaluations'] == 1
    assert not report['finite_domain_optimality_proved'] and not report['has_best']
    assert report['evaluation_trace'][0]['feasibility_known'] is False
    checks += 4

    # The same legal positive quota becomes exhausted during discovery. The
    # distinction from a caller-supplied invalid zero/negative quota is essential.
    tiny = copy.deepcopy(base)
    tiny['program_search']['seconds'] = 1e-12
    report, _ = run('discovery-timeout', tiny)
    assert report['status'] == 'UNKNOWN' and report['budget_exhausted']
    assert report['evaluations'] == 0 and not report['finite_domain_optimality_proved']
    assert not report['has_best'] and 'discovery' in report['reason']
    checks += 3
    invalid = copy.deepcopy(base)
    invalid['program_search']['seconds'] = 0
    report, stderr = run('invalid-budget', invalid, False)
    assert report is None and 'budget' in stderr
    checks += 1

    # Uncalibrated is a supported, well-formed model with an unknown cost. It
    # still provides a checked feasible graph and must not become infeasible.
    report, _ = run('well-formed-unknown-cost', base)
    assert report['status'] == ('UNAVAILABLE' if args.solver_off else 'FEASIBLE')
    assert report['unknown_cost_evaluations'] == 1 and report['has_best']
    assert not report['finite_domain_optimality_proved']
    assert report['evaluation_trace'][0]['feasibility_known'] is True
    checks += 3
    # Freeze a tiny synthetic price from the checked native feature receipt.
    # Both failures must be controlled nonzero exits, including errors deferred
    # until stream flush/close. No device or frozen experimental file is used.
    features = json.loads((root / 'well-formed-unknown-cost/search/baseline.json').read_text())[
        'evaluation']['native_reference']['public_complete_invocation']['features']
    priced = copy.deepcopy(base)
    priced['generic_air']['cost_model'] = dict(
        schema='orbit.direct-air.cost.v1', kind='linear-composite', calibration='calibrated',
        model_id='synthetic-io-only', model_version='1', training_manifest_digest='b' * 64,
        boundary=target['boundary'], target=target,
        domain=dict(source_fingerprints=features['source_fingerprints'],
                    implementations=features['implementations'],
                    operations=list(features['source_operations']),
                    feature_ranges={key: [value, value] for key, value in features['values'].items()}),
        terms=[dict(feature='command_count', coefficient_ms=0.125)])
    report, _ = run('priced-control', priced)
    assert report['has_best'] and abs(report['best_cost_ms'] - 0.125) < 1e-12
    checks += 1
    assert Path('/dev/full').exists(), 'this contract requires the Linux write-error device'
    for name, destination in [('candidate-io', 'candidate-0/taskflow.mlir'),
                              ('best-io', 'best.mlir'),
                              ('json-io', 'candidate-0/evaluation.json.tmp')]:
        _, stderr = run(name, priced, False, destination)
        assert 'No space left on device' in stderr and 'persist' in stderr, (name, stderr)
        checks += 1
    print(json.dumps(dict(status='passed', checks=checks, root=str(root),
                          hardware_executed=False, solver_off=args.solver_off)))


if __name__ == '__main__':
    main()
