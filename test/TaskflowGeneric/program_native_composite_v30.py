#!/usr/bin/env python3
"""CPU-only full-call provider integration; all prices here are synthetic."""
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
    root = Path(tempfile.mkdtemp(prefix='orbit-native-composite-cli-'))
    fixture = Path(__file__).resolve().parents[1] / 'TaskflowAIR/fixtures/heterogeneous.mlir'
    target = dict(id='synthetic-v30-cli', toolchain='compiler-only',
                  runtime='synthetic-runtime', boundary='prepared-command-host-to-host')
    base = dict(generic_air=dict(cost_target=target, cost_model=dict(
        schema='orbit.direct-air.cost.v1', kind='linear-composite', calibration='uncalibrated')),
        native_auto_candidates=dict(mode='joint', algorithm='cp-sat-all'),
        program_search=dict(seconds=5, max_evaluations=1,
                            max_values_per_dimension=2, max_routes=2, max_priorities=2))

    def run(name, config, valid=True):
        folder = root / name
        folder.mkdir()
        path = folder / 'config.json'
        path.write_text(json.dumps(config))
        command = [str(args.tool.resolve()), str(fixture), '--search', '--backend=air',
                   '--max-rounds=1', '--max-candidates=32', '--target-config=' + str(path),
                   '--search-output=' + str(folder / 'search')]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (folder / 'log').write_text(result.stdout + result.stderr)
        assert (result.returncode == 0) == valid, (name, result.stderr)
        if not valid:
            return None, None
        report = json.loads((folder / 'search/search.json').read_text())
        evaluation = json.loads((folder / 'search/baseline.json').read_text())['evaluation']
        assert not report['finite_domain_optimality_proved']
        return report, evaluation

    # The direct collector must return exactly the quantities serialized by the
    # native mapping path, without constructing a ProgramTiming timeline.
    feature_config = root / 'feature-config.json'
    feature_config.write_text(json.dumps(base))
    direct_features = []
    for name, with_mapping in [('features-with-mapping', True),
                               ('features-only', False)]:
        folder = root / name
        folder.mkdir()
        feature_path = folder / 'features.json'
        command = [str(args.tool.resolve()), str(fixture), '--generic-map',
                   '--backend=air', '--analysis-output=/dev/null',
                   '--target-config=' + str(feature_config),
                   '--native-cost-features-output=' + str(feature_path)]
        if with_mapping:
            command.append('--mapping-output=' + str(folder / 'mapping.json'))
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (folder / 'log').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, (name, result.stderr)
        exported = json.loads(feature_path.read_text())
        direct_features.append(exported)
        assert exported['values'] and exported['source_fingerprints']
        assert 'activities' not in exported
        if with_mapping:
            mapping = json.loads((folder / 'mapping.json').read_text())
            assert exported == mapping['direct_air_cost']['features']
        else:
            assert not result.stdout.strip(), 'features-only must not serialize a mapping'
            assert not (folder / 'mapping.json').exists()
            assert not (folder / 'timing.json').exists()
    assert direct_features[0] == direct_features[1]

    report, evaluation = run('uncalibrated', base)
    assert report['unknown_cost_evaluations'] == 1 and report['best_cost_ms'] is None
    features = evaluation['native_reference']['public_complete_invocation']['features']
    assert features is not None
    model = dict(schema='orbit.direct-air.cost.v1', kind='linear-composite',
        calibration='calibrated', model_id='synthetic-contract-only', model_version='1',
        training_manifest_digest='a' * 64, boundary=target['boundary'], target=target,
        domain=dict(source_fingerprints=features['source_fingerprints'],
                    implementations=features['implementations'],
                    operations=list(features['source_operations']),
                    feature_ranges={k: [v, v] for k, v in features['values'].items()}),
        terms=[dict(feature='command_count', coefficient_ms=0.123)])
    calibrated = copy.deepcopy(base)
    calibrated['generic_air']['cost_model'] = model
    report, evaluation = run('neutral-native-witness', calibrated)
    assert abs(report['best_cost_ms'] - 0.123) < 1e-12
    assert report['unknown_cost_evaluations'] == 0
    receipt = evaluation['native_reference']['public_complete_invocation']
    assert receipt['native_graph_matches_selected'] and not receipt['additive_body_or_lifecycle_costs']
    assert len(evaluation['timing']['activities']) == 1

    foreign = copy.deepcopy(calibrated)
    foreign['generic_air']['cost_target']['runtime'] = 'different-runtime'
    report, _ = run('different-runtime', foreign)
    assert report['best_cost_ms'] is None and report['unknown_cost_evaluations'] == 1

    changed = copy.deepcopy(calibrated)
    changed['program_search']['dimensions'] = [dict(id='buffering.slots', choices=[dict(reuse_slots=True)])]
    report, evaluation = run('post-native-rebind', changed)
    receipt = evaluation['native_reference']['public_complete_invocation']
    assert report['best_cost_ms'] is None and not receipt['native_graph_matches_selected']
    assert receipt['features'] is None and not receipt['native_counts_applied_to_selected']

    missing = copy.deepcopy(calibrated)
    del missing['generic_air']['cost_target']
    run('missing-runtime-contract', missing, False)
    table = copy.deepcopy(calibrated)
    table['generic_air']['cost_model']['kind'] = 'observed-table'
    run('complete-plan-price-table', table, False)
    print(json.dumps(dict(status='passed', cases=6, feature_export_cases=2, hardware_executed=False,
                          scope='synthetic complete-call search wiring and strict unknown guards')))


if __name__ == '__main__':
    main()
