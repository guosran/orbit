#!/usr/bin/env python3
"""CPU-only independent audit of saved v4 calibration, bindings and domain queries."""
import argparse, hashlib, json, math, statistics, subprocess
from pathlib import Path
import numpy as np
from hardware_evidence import chain_audit as legacy_chain_audit

def chain_audit(path):
    result = json.loads(Path(path).read_text())
    audit = legacy_chain_audit(path)
    if result.get('schema') == 'orbit.chain.experiment.v2':
        audit['errors'] = [e for e in audit['errors'] if e != 'unsupported chain evidence schema']
        audit['passed'] = not audit['errors']
    return audit
ROOT = Path(__file__).resolve().parents[2]

def read(p):
    return json.loads(Path(p).read_text())

def identity(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'build/static-mapping-v4/region-calibration/audit')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = ROOT / 'build/static-mapping-v4/region-calibration'
    errors = []
    families = {}
    checks = []

    def need(ok, msg):
        if not ok:
            errors.append(msg)
    for family in ['pow2-revision', 'singles']:
        model = read(base / family / 'model.json')
        pred = read(base / family / 'predictions.json')
        m = dict(model)
        mid = m.pop('model_id')
        need(identity(m) == mid, family + ' model identity')
        p = dict(pred)
        pid = p.pop('prediction_id')
        need(identity(p) == pid, family + ' prediction identity')
        need(pred['model_id'] == mid and model['uses_holdout_measurements'] is False and (pred['uses_holdout_measurements'] is False), 'train-only declaration')
        need(not set(model['training_ids']) & set(model['excluded_ids']), 'split leakage')
        rows = []
        times = {mode: [] for mode in model['models']}
        for evidence in model['training_evidence']:
            path = Path(evidence['path'])
            d = read(path)
            need(identity(d) == evidence['identity'], 'training evidence identity')
            need(d['case'] in model['training_ids'] and d['case'] not in model['excluded_ids'], 'heldout used for fitting')
            need(d['status'] == 'passed' and d['declaration_id'] == model['declaration_id'], 'training execution identity')
            shape = d['shape']
            rows.append([1, shape['m'], shape['h'], shape['m'] * shape['h']])
            if family == 'pow2-revision':
                audit = chain_audit(path)
                checks.append(audit)
                need(audit['passed'], 'pair full-output audit ' + d['case'])
            else:
                inputs = np.load(path.parent / 'inputs-f32.npz')
                second = np.load(path.parent / 'single-second-input-f32.npy')
                refs = {'single_first': inputs['X'].astype('f8') @ inputs['W1'].astype('f8'), 'single_second': second.astype('f8') @ inputs['W2'].astype('f8')}
            for mode, variant in d['variants'].items():
                samples = variant['samples']
                wall = [s['wall_ms'] for s in samples]
                need(len(wall) >= 7 and all((math.isfinite(t) and t > 0 for t in wall)), 'hot sample validity')
                median = statistics.median(wall)
                times[mode].append(median)
                need(math.isclose(median, model['models'][mode]['training_medians_ms'][d['case']], rel_tol=1e-12), 'model median provenance')
                if family == 'singles':
                    ref = refs[mode]
                    need(np.array_equal(ref, np.load(path.parent / (mode + '-reference-f64.npy'))), 'single reference equation')
                    numerical = []
                    for sample in samples:
                        out = np.load(sample['output_path'])
                        err = out.astype('f8') - ref
                        rms = float(np.sqrt(np.mean(err * err)) / max(np.sqrt(np.mean(ref * ref)), 1e-30))
                        peak = float(abs(err).max() / max(abs(ref).max(), 1e-30))
                        good = out.shape == ref.shape and np.isfinite(out).all() and np.allclose(out, ref, atol=0.015, rtol=0.06) and (rms <= 0.03) and (peak <= 0.08)
                        need(good, 'single whole-output reference')
                        need([e['kind'] for e in sample['events']] == ['actual_xrt_start', 'actual_xrt_wait'] and sample['actual_submissions'] == sample['actual_waits'] == 1, 'single must have own observed one dispatch')
                        numerical.append(dict(max_abs=float(abs(err).max()), normalized_rms=rms, passed=bool(good)))
                    checks.append(dict(source=str(path), mode=mode, numerical=numerical, passed=all((n['passed'] for n in numerical))))
        coefficients = {}
        for mode, y in times.items():
            fit = np.linalg.solve(np.array(rows, dtype=float), y)
            need(np.allclose(fit, model['models'][mode]['coefficients'], rtol=1e-10, atol=1e-12), 'independent train-only coefficient refit')
            coefficients[mode] = fit.tolist()
        families[family] = dict(model_id=mid, prediction_id=pid, training_ids=model['training_ids'], excluded_ids=model['excluded_ids'], independently_refit_coefficients=coefficients)
    model = read(base / 'pow2-revision/model.json')
    pred = read(base / 'pow2-revision/predictions.json')
    evaluation = read(base / 'pow2-revision/holdout-evaluation.json')
    hardware = Path('/home/shiran/project/aie-toy/results/orbit-static-v4-hardware/pow2-revision')
    heldout = {}
    for case, frozen in pred['predictions'].items():
        path = hardware / case / 'result.json'
        d = read(path)
        audit = chain_audit(path)
        checks.append(audit)
        need(audit['passed'], 'holdout full output')
        need(d['prediction_id'] == pred['prediction_id'], 'execution bound frozen prediction')
        shape = frozen['shape']
        feature = np.array([1, shape['m'], shape['h'], shape['m'] * shape['h']])
        predicted = {mode: float(feature @ np.array(v['coefficients'])) for mode, v in model['models'].items()}
        observed = {mode: statistics.median((s['wall_ms'] for s in v['samples'])) for mode, v in d['variants'].items()}
        selected = min(predicted, key=predicted.get)
        regret = observed[selected] - min(observed.values())
        need(all((math.isclose(predicted[k], frozen['values_ms'][k], rel_tol=1e-12) for k in predicted)), 'frozen prediction equation')
        need(evaluation['heldout'][case]['selected'] == selected and math.isclose(regret, evaluation['heldout'][case]['regret_ms'], abs_tol=1e-12), 'heldout ranking/regret')
        for binding in d['native_plan_bindings']:
            mode = binding['mode']
            planpath = base / 'native' / (mode + '-plan.json')
            configpath = base / 'native' / (mode + '-config.json')
            sourcepath = base / 'native/source.mlir'
            plan = read(planpath)
            need(binding['plan_id'] == plan['plan_id'] and binding['plan_sha256'] == digest(planpath) and (binding['config_sha256'] == digest(configpath)) and (binding['source_sha256'] == digest(sourcepath)), 'native execution binding')
            need(plan['events'][0]['model_id'] == model['model_id'] and plan['events'][0]['region_mode'] == mode, 'selected region mode/model')
            need(math.isclose(plan['makespan_ms'], predicted[mode], rel_tol=1e-12), 'native score equals frozen model')
        heldout[case] = dict(predicted_ms=predicted, observed_ms=observed, selected=selected, observed_best=min(observed, key=observed.get), regret_ms=regret, relative_error={k: (predicted[k] - observed[k]) / observed[k] for k in predicted})
    queries = {}
    tool = ROOT / 'build-core/tools/taskflow-opt/taskflow-opt'
    workloads = ROOT / 'build/static-mapping-v4/cpp-migration/reproduction/workloads'
    for name in ['gemm_relu', 'ffn_m5', 'ffn_m128']:
        source = workloads / name / 'base/taskflow.mlir'
        program = read(workloads / name / 'base/plan.json')
        config = read(base / 'native/combined-config.json')
        config['assumptions'].update(program_source_id=program['source_id'], initial_locations={b['id']: 'host' for b in program['buffers'] if b['role'] == 'input'}, output_locations={b: 'host' for b in program['outputs']}, output_devices={b: 'cpu' for b in program['outputs']})
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        cp = directory / 'config.json'
        cp.write_text(json.dumps(config, indent=2))
        rp = directory / 'mapping.json'
        cmd = [str(tool), str(source), '--static-map', '--target-config=' + str(cp), '--mapping-output=' + str(rp)]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        (directory / 'command.json').write_text(json.dumps(cmd))
        (directory / 'stderr.txt').write_text(proc.stderr)
        need(proc.returncode in (0, 1) and rp.exists(), 'domain query tool execution')
        r = read(rp)
        need(r['best_plan'] is None and r['status'] != 'passed', 'uncalibrated workload must remain unscored')
        need(not any(('another source' in x for x in r['reasons'])), 'domain test must not fail source identity')
        ops = sorted({o['kind'] for t in program['tasks'] for o in t['operations']})
        queries[name] = dict(status=r['status'], reasons=r['reasons'], operations=ops, cpu_boundary=[x for x in ops if x in ['relu', 'swiglu']], interpretation='No complete calibrated exact cover; CPU activation and/or dimensions outside measured two-GEMM domain. No estimated wall is ranked.')
    for label, m, h, cpu_relu in [('power2_gap', 256, 768, False), ('outside_m', 512, 512, False), ('calibrated_gemm_cpu_relu', 256, 512, True)]:
        program = read(base / 'native/source.json')
        program.pop('source_id')
        program.pop('candidate_id')
        shapes = {'X': [m, 256], 'W1': [256, h], 'H': [m, h], 'W2': [h, 256], 'Y': [m, 256]}
        for b in program['buffers']:
            b['shape'] = shapes[b['id']]
        for t in program['tasks']:
            for op in t['operations']:
                op['m'] = [0, m]
                op['n'] = [0, h if t['id'] == 'first' else 256]
        program['tasks'][1]['operations'][1]['k'] = [0, h]
        if cpu_relu:
            program['buffers'] = [b for b in program['buffers'] if b['id'] != 'W2']
            next((b for b in program['buffers'] if b['id'] == 'Y'))['shape'] = [m, h]
            program['tasks'][1] = {'id': 'cpu_relu', 'depends_on': ['first'], 'operations': [{'kind': 'relu', 'inputs': ['H'], 'output': 'Y', 'm': [0, m], 'n': [0, h]}]}
        program['source_id'] = identity(program)
        program['candidate_id'] = program['source_id']
        directory = args.output / label
        directory.mkdir(exist_ok=True)
        source = directory / 'source.json'
        source.write_text(json.dumps(program, indent=2))
        mlir = directory / 'source.mlir'
        subprocess.run([str(tool), str(source), '--emit-mlir=' + str(mlir)], check=True, capture_output=True)
        source = mlir
        config = read(base / 'native/combined-config.json')
        config['assumptions'].update(program_source_id=program['source_id'], initial_locations={b['id']: 'host' for b in program['buffers'] if b['role'] == 'input'})
        cp = directory / 'config.json'
        cp.write_text(json.dumps(config, indent=2))
        rp = directory / 'mapping.json'
        cmd = [str(tool), str(source), '--static-map', '--target-config=' + str(cp), '--mapping-output=' + str(rp)]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        (directory / 'command.json').write_text(json.dumps(cmd))
        (directory / 'stderr.txt').write_text(proc.stderr)
        r = read(rp)
        need(r['status'] == 'unscored' and r['best_plan'] is None, 'domain or host-boundary probe must not score')
        queries[label] = {'status': r['status'], 'reasons': r['reasons']}
    report = dict(passed=not errors, errors=errors, families=families, heldout=heldout, numerical_audits=checks, domain_queries=queries, limitations=['Saved content hashes and execution prediction bindings establish artifact consistency; filesystem chronology alone is not an external timestamp attestation.', 'Model is opaque warm host-to-host wall, not decomposed compute or transfer calibration.', 'Workspace values are conservative configured reservations, not verified compiler scratch upper bounds.', 'Only heldout two-GEMM shape validates error; GEMM-ReLU and full FFN remain uncalibrated.'])
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(passed=report['passed'], errors=errors, output=str(args.output / 'audit.json'))))
    return 0 if report['passed'] else 1
if __name__ == '__main__':
    raise SystemExit(main())
