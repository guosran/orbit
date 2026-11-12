#!/usr/bin/env python3
"""CPU-only CLI contracts for region launch ownership and declared polling.

All prices are explicit assumptions. Hand totals below derive from the actual
three-task fixture (three scalar bodies, 64-byte buffers), independently of
the compiler's timing implementation. No hardware execution or calibration.
"""
import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import runpy
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def parameter(value, unit='ns', kind='assumed'):
    return dict(value=value, unit=unit, provenance=dict(
        kind=kind, source='v28 CPU-only explicit mathematical contract'))


def configuration(launch=5000):
    parameters = {
        'compute.llvm-ciface-f32.startup': parameter(40000),
        'compute.direct-air-hostcall.startup': parameter(2000),
        'compute.scalar.startup': parameter(2000),
        'compute.operation': parameter(0),
        'compute.memory_access': parameter(0),
        'communication.startup': parameter(1000),
        'communication.bandwidth': parameter(64, 'bytes/ns'),
    }
    for prefix in ['device.stage', 'device.to_device', 'device.from_device',
                   'device.submit', 'device.complete', 'host.submit',
                   'host.complete', 'input_reset', 'output_copy']:
        for suffix in ['per_call', 'per_bo', 'per_byte']:
            parameters['invocation.'+prefix+'.'+suffix] = parameter(
                0, 'ns/byte' if suffix == 'per_byte' else 'ns')
    parameters['invocation.host.submit.per_call'] = parameter(1000)
    parameters['invocation.device.submit.per_call'] = parameter(launch)
    parameters['invocation.device.setup.per_call'] = parameter(0)
    parameters['invocation.device.teardown.per_call'] = parameter(0)
    return dict(program_invocation_model=dict(
        kind='region-invocation', schema='orbit.region-invocation.cost.v28',
        model_id='v28-assumed-cli-contract', mode='estimate',
        boundary='prepared-graph-body-and-invocation',
        implementations=['llvm-ciface-f32', 'direct-air-hostcall', 'scalar'],
        runtime=dict(identity='v28-cpu-contract-runtime', state='prepared-hot',
            host_worker_resource='CPU.compute', bound_buffers=True,
            stage_outputs_to_device=True, reset_inputs=False,
            copy_final_outputs=True),
        calibration_runtime_identity='v28-cpu-contract-runtime',
        completion_policy='post-device-overhead', parameters=parameters),
        heterogeneous_search=dict(seconds=20, max_evaluations=64,
            max_variables=10000, max_constraints=50000, workers=1, seed=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tool', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--solver-off', action='store_true')
    args = p.parse_args()
    out = args.output or Path(tempfile.mkdtemp(prefix='orbit-invocation-v28-'))/'results'
    out.mkdir(parents=True, exist_ok=False)
    tool = str(args.tool.resolve())
    source = ROOT/'test/TaskflowAIR/fixtures/heterogeneous.mlir'
    records = []

    def require(ok, label):
        assert ok, label
        records.append(label)

    def write(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        return path

    def invoke(name, cfg, assignment=None, success=True):
        d = out/name
        d.mkdir()
        path = write(d/'config.json', cfg)
        cmd = [tool, str(source), '--target-config='+str(path)]
        if assignment is None:
            cmd += ['--heterogeneous-program-search='+str(d/'search')]
        else:
            cmd += ['--heterogeneous-program-assignment='+str(write(d/'assignment.json', assignment)),
                    '--implementation-output='+str(d/'evaluation.json')]
        write(d/'command.json', cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        (d/'stdout').write_text(result.stdout)
        (d/'stderr').write_text(result.stderr)
        require((result.returncode == 0) == success,
                name+': expected CLI success='+str(success)+'; '+result.stderr[-1000:])
        if not success:
            return None
        if assignment is None:
            report = json.loads((d/'search/search.json').read_text())
            best = json.loads((d/'search/best.json').read_text()) if report['has_best'] else None
            return report, best
        return json.loads((d/'evaluation.json').read_text())

    def evidence(e):
        require(e['feasible'] and e['feasibility_known'], 'single assignment checked feasible')
        return e['evaluation']['cost_evaluation'], e['evaluation']['heterogeneous']

    def close(actual, expected, label):
        require(actual is not None and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12),
                label+f': expected {expected}, got {actual}')

    cfg = configuration()
    catalogue = {}
    for a in itertools.product(range(2), repeat=5):
        e = invoke('point-'+''.join(map(str, a)), cfg, list(a))
        evidence(e)
        require(e['estimated_ms'] is not None, 'all finite oracle costs known')
        catalogue[a] = e
    # Independent counts from the source: 3 bodies; each tensor has 16*f32=64B.
    # A full NPU region imports A/B and exports C, no intermediate BO transfers.
    close(catalogue[(0, 0, 0, 0, 0)]['estimated_ms'], .121, 'all CPU hand total')
    close(catalogue[(1, 1, 1, 0, 0)]['estimated_ms'], .014003, 'one NPU region hand total')
    close(catalogue[(0, 1, 0, 0, 0)]['estimated_ms'], .091002, 'CPU/NPU/CPU hand total')
    close(catalogue[(1, 1, 1, 1, 1)]['estimated_ms'], .029008, 'three NPU calls hand total')
    mixed, mapping = evidence(catalogue[(0, 1, 0, 0, 0)])
    regions = mixed['invocation']['features']['regions']
    require([r['device'] for r in regions] == [False, True, False], 'placement creates CPU/NPU/CPU regions')
    require([r['submissions'] for r in regions] == [1, 1, 1] and
            regions[1]['to_device_bytes'] == 128 and regions[1]['from_device_bytes'] == 64,
            'one NPU launch owns its 64B input and 64B output BO visibility')
    whole, whole_graph = evidence(catalogue[(1, 1, 1, 0, 0)])
    r = whole['invocation']['features']['regions']
    require(len(r) == 1 and len(r[0]['input_allocations']) == 2 and r[0]['submissions'] == 1,
            'multiple input edges share exactly one region launch')
    require(whole['invocation']['body_events_owned'] is False and
            whole['invocation']['logical_group_configuration_charged'] is False,
            'body and launch costs have disjoint ownership')
    report, best = invoke('search-low-launch', cfg)
    require(report['space_id'] == whole_graph['space_id'], 'search and independent points have identical finite domain identity')
    if args.solver_off:
        require(report['status'] == 'UNAVAILABLE' and report['evaluations'] == 1 and
                not report['finite_domain_optimality_proved'], 'OFF search remains checked-baseline-only')
        close(report['best_cost_ms'], catalogue[(0, 0, 0, 0, 0)]['estimated_ms'], 'OFF baseline equals independent point')
    else:
        require(report['status'] == 'OPTIMAL' and report['evaluations'] == 32 and
                report['finite_domain_exhausted'] and report['finite_domain_optimality_proved'],
                'CP proves finite 32-assignment full-cost optimum')
        close(report['best_cost_ms'], min(e['estimated_ms'] for e in catalogue.values()), 'CP equals independent complete enumeration')
        require(best['heterogeneous']['choice']['task_devices'] == ['npu']*3,
                'low launch optimum selects one complete NPU region')
        trace = json.loads((out/'search-low-launch/search/evaluations.json').read_text())
        actual = {tuple(row['assignment']): row['cost_ms'] for row in report['evaluation_trace']}
        require(set(actual) == set(catalogue), 'proof trace covers exactly declared assignments')
        for a, e in catalogue.items():
            close(actual[a], e['estimated_ms'], 'CP and single-point full prices agree')
        require(len(trace) == 32, 'each declared assignment checked once')
    high = configuration(200000)
    high_report, high_best = invoke('search-high-launch', high)
    close(high_report['best_cost_ms'], .121, 'high launch hand optimum')
    require(high_best['heterogeneous']['choice']['task_devices'] == ['cpu']*3,
            'high launch changes argmin to CPU without changing source')
    # Frozen domains are controls on choices, not source rewrites.
    fixed = copy.deepcopy(cfg)
    fixed['heterogeneous_search'].update(task_device_choices=[['npu']]*3,
        boundary_choices=[[False], [True]])
    frozen, fixed_best = invoke('fixed-domain', fixed)
    require(len(fixed_best['heterogeneous']['program_graph']['execution_regions']) == 2,
            'explicit fixed cut/join domains materially change regions')
    # Strict mode cannot launder assumptions or missing body mechanisms into zero.
    unknown = copy.deepcopy(cfg)
    unknown['program_invocation_model']['mode'] = 'strict'
    unknown['heterogeneous_search']['max_evaluations'] = 64
    unknown_report, _ = invoke('strict-unknown', unknown)
    require(not unknown_report['finite_domain_optimality_proved'] and
            unknown_report['best_cost_ms'] is None and unknown_report['unknown_cost_evaluations'] > 0,
            'strict assumed costs remain unknown and cannot prove optimum')
    missing = copy.deepcopy(cfg)
    del missing['program_invocation_model']['parameters']['invocation.device.submit.per_call']
    e = invoke('missing-launch', missing, [1, 1, 1, 0, 0])
    require(e['feasible'] and e['estimated_ms'] is None, 'missing launch is unknown rather than free')
    # Runtime device setup is a separate once-per-NPU-region operation after
    # submission, before all device body work. It scales with actual call cuts.
    setup_cfg = copy.deepcopy(cfg)
    setup_cfg['program_invocation_model']['parameters']['invocation.device.setup.per_call'] = parameter(50000)
    setup_whole = invoke('setup-whole', setup_cfg, [1, 1, 1, 0, 0])
    setup_split = invoke('setup-split', setup_cfg, [1, 1, 1, 1, 1])
    close(setup_whole['estimated_ms'], .064003, 'one 50us setup added to one NPU call')
    close(setup_split['estimated_ms'], .179008, 'three 50us setups added to three NPU calls')
    setup_detail, _ = evidence(setup_whole)
    activities = setup_detail['mechanism']['timing']['activities']
    setup_activity = next(x for x in activities if x['id'] == 'invocation.device_setup.0')
    submit_activity = next(x for x in activities if x['id'] == 'invocation.lifecycle.1')
    close(setup_activity['duration_ms'], .05, 'setup owns its exclusive runtime-only duration')
    require(not setup_activity['events'] and not setup_activity['region_events'] and
            setup_activity['start_ms'] >= submit_activity['end_ms'] and
            all(x['start_ms'] >= setup_activity['end_ms'] for x in activities if x['events']),
            'setup occurs after submit and before body without duplicate event coverage')
    missing_setup = copy.deepcopy(cfg)
    del missing_setup['program_invocation_model']['parameters']['invocation.device.setup.per_call']
    e = invoke('missing-device-setup', missing_setup, [1, 1, 1, 0, 0])
    require(e['feasible'] and e['estimated_ms'] is None,
            'unmeasured device setup remains required unknown for device invocation')
    e = invoke('missing-device-setup-cpu', missing_setup, [0, 0, 0, 0, 0])
    close(e['estimated_ms'], .121, 'CPU regions do not invent unused NPU setup')
    teardown_cfg = copy.deepcopy(cfg)
    teardown_cfg['program_invocation_model']['parameters']['invocation.device.teardown.per_call'] = parameter(50000)
    e = invoke('teardown-whole', teardown_cfg, [1, 1, 1, 0, 0])
    close(e['estimated_ms'], .064003, 'post-body teardown is separately charged once')
    detail, _ = evidence(e)
    activities = detail['mechanism']['timing']['activities']
    teardown = next(x for x in activities if x['id'] == 'invocation.device_teardown.0')
    completion = next(x for x in activities if x['id'] == 'invocation.lifecycle.2')
    require(not teardown['events'] and not teardown['region_events'] and
            teardown['start_ms'] >= max(x['end_ms'] for x in activities if x['events']) and
            completion['start_ms'] >= teardown['end_ms'],
            'post-body reset occurs after body before complete, never moved to pre-body setup')
    missing_teardown = copy.deepcopy(cfg)
    del missing_teardown['program_invocation_model']['parameters']['invocation.device.teardown.per_call']
    e = invoke('missing-device-teardown', missing_teardown, [1, 1, 1, 0, 0])
    require(e['feasible'] and e['estimated_ms'] is None, 'unmeasured teardown keeps device invocation unknown')
    # Declared policy: first probe at submit return, Q between probes, P probe
    # overhead; body B=9003ns => complete adds ceil(B/Q)*Q-B+P, not a whole wait.
    for name, quantum, polls, tail, total in [
            ('poll-small', 4000, 3, 3022, 17025),
            ('poll-large', 10000, 1, 1022, 15025),
            ('poll-exact', 3001, 3, 7, 14010),
            ('poll-with-setup', 4000, 15, 1022, 65025),
            ('poll-with-teardown', 4000, 15, 1022, 65025)]:
        polling = copy.deepcopy(cfg)
        model = polling['program_invocation_model']
        model['completion_policy'] = 'polling-quantized'
        if name == 'poll-with-setup':
            model['parameters']['invocation.device.setup.per_call'] = parameter(50000)
        if name == 'poll-with-teardown':
            model['parameters']['invocation.device.teardown.per_call'] = parameter(50000)
        model['parameters']['invocation.polling.quantum'] = parameter(quantum)
        model['parameters']['invocation.polling.probe'] = parameter(25 if name != 'poll-exact' else 7)
        e = invoke(name, polling, [1, 1, 1, 0, 0])
        detail, _ = evidence(e)
        row = detail['invocation']['conditional_polling'][0]
        close(row['body_ns'], 59003 if name in ['poll-with-setup', 'poll-with-teardown'] else 9003, name+' hand device interval including declared reset exactly once')
        close(row['polls'], polls, name+' hand poll count')
        close(row['post_body_detection_ns'], tail, name+' hand post-body tail')
        close(e['estimated_ms']*1e6, total, name+' hand complete time without whole-wait double charge')
        complete = next(a for a in detail['mechanism']['timing']['activities']
            if a['id'] == 'invocation.lifecycle.2')
        close(complete['duration_ms']*1e6, tail, name+' complete owns detection tail only')
    unknown_body = copy.deepcopy(polling)
    del unknown_body['program_invocation_model']['parameters']['compute.direct-air-hostcall.startup']
    unknown_body['program_invocation_model']['parameters']['compute.startup'] = parameter(None)
    e = invoke('poll-unknown-body', unknown_body, [1, 1, 1, 0, 0])
    detail, _ = evidence(e)
    require(e['estimated_ms'] is None and detail['invocation']['conditional_polling'][0]['post_body_detection_ns'] is None,
            'unknown device body keeps conditional completion and whole program unknown')
    for name, edit in [
            ('reject-runtime-identity', lambda c: c['program_invocation_model'].__setitem__('calibration_runtime_identity', 'forged')),
            ('reject-runtime-type', lambda c: c['program_invocation_model']['runtime'].__setitem__('bound_buffers', 'true')),
            ('reject-policy-type', lambda c: c['program_invocation_model'].__setitem__('completion_policy', 7)),
            ('reject-schema-type', lambda c: c['program_invocation_model'].__setitem__('schema', True)),
            ('reject-model-field', lambda c: c['program_invocation_model'].__setitem__('typo', 1)),
            ('reject-runtime-field', lambda c: c['program_invocation_model']['runtime'].__setitem__('typo', 1)),
            ('reject-search-field', lambda c: c['heterogeneous_search'].__setitem__('typo', 1)),
            ('reject-config-field', lambda c: c.__setitem__('unconsumed', 1)),
            ('reject-boundary-type', lambda c: c['heterogeneous_search'].__setitem__('boundary_choices', [[0], [False]])),
            ('reject-device-value', lambda c: c['heterogeneous_search'].__setitem__('task_device_choices', [['cpu'], ['tpu'], ['cpu']]))]:
        invalid = copy.deepcopy(cfg)
        edit(invalid)
        invoke(name, invalid, [0]*5, success=False)
    # The runtime-model flag re-evaluates an exactly frozen public graph. Its
    # source, assignment, old checker/cost evidence and graph identity stay fixed.
    replay_dir = out/'runtime-replay'
    replay_dir.mkdir()
    native = runpy.run_path(str(ROOT/'test/TaskflowGeneric/program_all_search_v25.py'))['configuration']()
    native['program_search'].update(seconds=10, max_evaluations=1)
    native_config = write(replay_dir/'native-config.json', native)
    search = replay_dir/'search'
    command = [tool, str(ROOT/'test/TaskflowAIR/fixtures/gemm.mlir'), '--search',
        '--backend=air', '--target-config='+str(native_config), '--search-output='+str(search)]
    q = subprocess.run(command, capture_output=True, text=True, timeout=40)
    (replay_dir/'search.log').write_text(q.stdout+q.stderr)
    require(q.returncode == 0, 'fresh checked native reference for runtime re-evaluation: '+q.stderr[-500:])
    baseline = json.loads((search/'baseline.json').read_text())
    expected = write(replay_dir/'evaluation.json', baseline['evaluation'])
    assignment = write(replay_dir/'assignment.json', baseline['assignment'])
    re_evaluated = []
    for label, launch in [('low', 5000), ('high', 200000)]:
        model = configuration(launch)['program_invocation_model']
        model['runtime'].update(identity='v28-native-replay-contract', host_worker_resource='host.worker')
        model['calibration_runtime_identity'] = 'v28-native-replay-contract'
        model_path = write(replay_dir/(label+'-runtime-model.json'), model)
        receipt_path = replay_dir/(label+'-runtime-receipt.json')
        command = [tool, str(search/'candidate-0/taskflow.mlir'), '--target-config='+str(native_config),
            '--public-program-evaluation='+str(expected), '--public-program-space='+str(search/'space.json'),
            '--public-program-assignment='+str(assignment), '--public-program-runtime-model='+str(model_path),
            '--implementation-output='+str(receipt_path)]
        write(replay_dir/(label+'-command.json'), command)
        q = subprocess.run(command, capture_output=True, text=True, timeout=40)
        (replay_dir/(label+'.log')).write_text(q.stdout+q.stderr)
        require(q.returncode == 0, 'public runtime-model CLI re-evaluation: '+q.stderr[-500:])
        r = json.loads(receipt_path.read_text())
        require(r['estimated_ms'] is not None and r['replay']['complete_graph_matches'] and
                r['replay']['search_performed'] is False and
                r['invocation']['graph_id'] == baseline['evaluation']['graph_id'] and
                r['physical_time_optimality_proved'] is False,
                'runtime re-evaluation keeps frozen graph and excludes physical optimum claims')
        require(len(r['invocation']['features']['regions']) == 1 and
                r['invocation']['features']['regions'][0]['submissions'] == 1,
                'native reference has one invocation shared by its original graph')
        re_evaluated.append(r)
    close((re_evaluated[1]['estimated_ms']-re_evaluated[0]['estimated_ms'])*1e6,
          195000, 'only one changed launch coefficient charged on exact graph replay')
    summary = dict(status='passed', checks=len(records), raw_points=32,
        solver_off=args.solver_off, hardware_executed=False,
        scope='Assumed finite-region price and declared polling math contracts; no physical-time accuracy claim')
    write(out/'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
