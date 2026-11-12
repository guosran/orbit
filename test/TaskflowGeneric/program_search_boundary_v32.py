#!/usr/bin/env python3
"""CPU-only search metadata contracts; synthetic prices, no device use."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix='orbit-search-boundary-')) / 'results'
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    source = repo / 'test/TaskflowAIR/fixtures/heterogeneous.mlir'
    spec = importlib.util.spec_from_file_location('public_contract',
        Path(__file__).with_name('program_all_search_v25.py'))
    public = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(public)
    base = public.configuration()
    base['program_search'].update(max_values_per_dimension=1,
        max_routes=1, max_priorities=1, max_evaluations=1)
    records = []

    def run(name, config):
        folder = output / name
        folder.mkdir()
        path = folder / 'config.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        command = [str(args.tool.resolve()), str(source), '--search', '--backend=air',
            '--target-config=' + str(path), '--search-output=' + str(folder / 'search')]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (folder / 'stdout').write_text(proc.stdout)
        (folder / 'stderr').write_text(proc.stderr)
        assert proc.returncode == 0, (name, proc.returncode, proc.stderr)
        report = json.loads((folder / 'search/search.json').read_text())
        best = folder / 'search/best.json'
        evidence = json.loads(best.read_text()) if best.exists() else None
        records.append(dict(name=name, command=command, returncode=proc.returncode))
        assert report['cost_model_boundary_scope'] == 'selected incumbent receipt'
        return report, evidence

    # Ordinary public timing declares its own boundary, not the old hard-coded
    # partition warning. A checked cost is separate from its boundary string.
    report, evidence = run('ordinary', base)
    assert report['has_best'] and evidence['timing']['boundary']
    assert report['cost_model_boundary'] == evidence['timing']['boundary']
    assert report['cost_model_boundary_priced'] is True

    exhausted = copy.deepcopy(base)
    exhausted['program_search']['seconds'] = 1e-12
    report, evidence = run('no-selected-receipt', exhausted)
    assert not report['has_best'] and evidence is None
    assert report['cost_model_boundary'] == 'unknown: no selected timing boundary receipt'
    assert report['cost_model_boundary_priced'] is False

    partition = copy.deepcopy(base)
    partition['program_search']['dimensions'] = [dict(
        id='replica.partition', choices=[dict(partition=8)])]
    report, evidence = run('partition-host-assembly', partition)
    assert report['has_best'], report
    assert evidence['status'] == 'uncalibrated-complete-program-host-boundary', evidence
    assert report['best_cost_ms'] is None
    assert report['cost_model_boundary'] == (
        'unknown: partition host slicing, assembly, and invocation control')
    assert report['cost_model_boundary_priced'] is False

    # Real typed-edge export/evaluator, using all-CPU regions so this contract
    # needs neither AIR compilation nor hardware. First discover checked local
    # source-family features, then supply explicit synthetic constant prices.
    contract = output / 'execution-contract.json'
    contract.write_text(json.dumps(dict(
        runtime_identity='orbit.serial-prepared-typed-native-wait2.v32',
        provenance='synthetic compiler-only metadata contract; no hardware calibration')))
    typed = copy.deepcopy(base)
    typed.pop('generic_air')
    typed['execution_contract'] = str(contract)
    typed['program_search'].update(heterogeneous=True, dimensions=[
        dict(id='device.task.' + str(t), choices=[dict(task=t, device='cpu')])
        for t in range(3)])
    typed['heterogeneous_cost'] = dict(schema='orbit.heterogeneous.edge-cost.v32',
        runtime_identity='orbit.serial-prepared-typed-native-wait2.v32', regions=[])
    report, evidence = run('typed-edge-unknown-price', typed)
    assert report['has_best'], report
    receipt = evidence['cost_evaluation']
    assert receipt['schema'] == 'orbit.heterogeneous.edge-evaluation.v32'
    assert receipt['timing']['boundary'] == (
        'prepared-host-to-host: reset, exact serial regions, output copy')
    assert report['cost_model_boundary'] == receipt['timing']['boundary']
    assert report['cost_model_boundary_priced'] is False and report['best_cost_ms'] is None

    def phase(value):
        return dict(coefficients_ns=dict(call=dict(value=value,
            kind='architecture' if value == 0 else 'calibrated',
            source='synthetic metadata test; no hardware calibration')),
            ranges=dict(call=[0, 1000000000]))

    manifest = json.loads(Path(receipt['implementation_manifest']).read_text())
    model = typed['heterogeneous_cost']
    model['execution_contract_identity'] = manifest['execution_contract_identity']
    model['application'] = {name: phase(0) for name in ('reset', 'control', 'copy')}
    for region in receipt['features']['regions']:
        model['regions'].append(dict(device=region['device'],
            source_family=region['source_family'], implementations=region['implementations'],
            phases=dict(body=phase(1))))
    report, evidence = run('typed-edge-priced', typed)
    assert report['has_best'] and report['best_cost_ms'] == 1e-6, report
    assert report['cost_model_boundary'] == evidence['cost_evaluation']['timing']['boundary']
    assert report['cost_model_boundary_priced'] is True
    (output / 'summary.json').write_text(json.dumps(dict(status='passed',
        cases=records, hardware_executed=False,
        scope='Selected timing receipt boundary only; no pricing or execution semantic change'), indent=2) + '\n')
    print(json.dumps(dict(status='passed', cases=len(records), hardware_executed=False)))


if __name__ == '__main__':
    main()
