#!/usr/bin/env python3
"""Authenticated static-plan evaluator for the existing ORBIT search protocol.

No device execution, runtime imports or hardware locks. Scores are estimates.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        temporary = stream.name
    os.replace(temporary, path)


def implementation_files():
    return [str(p.resolve()) for p in [Path(__file__), ROOT / 'build-core/tools/taskflow-opt/taskflow-opt',
            *sorted((ROOT / 'python/orbit_mapping').glob('*.py'))]]


def evaluate(request):
    from orbit_mapping.mapper import map_program
    from orbit_mapping.checker import check_plan
    if request.get('schema') != 'orbit.search.evaluation.v1' or request.get('backend') != 'static':
        raise ValueError('expected static search request')
    program = load(request['plan_path'])
    if any(program[k] != request[k] for k in ('candidate_id', 'source_id')):
        raise ValueError('public candidate/source identity mismatch')
    target = request['target_config']
    config = load(target['mapping_config'])
    out = Path(request['output_dir']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Authentication checks the executable MLIR body against its witness, then
    # exports that witness and compares it with the requested public program.
    tool = ROOT / 'build-core/tools/taskflow-opt/taskflow-opt'
    proc = subprocess.run([str(tool), request['mlir_path'], '--emit-plan=' + str(out / 'authenticated.json')], capture_output=True, text=True)
    (out / 'authentication.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode or load(out / 'authenticated.json') != program:
        raise ValueError('Taskflow body/program authentication failed')
    code = {p: digest(p) for p in implementation_files()}
    if target.get('implementation_sha256') != identity(code):
        raise ValueError('static implementation changed: regenerate config in a new run')
    binding = dict(program=program, taskflow_sha256=digest(request['mlir_path']),
                   config=config, implementation=code, execution='static_only_single_cold_graph')
    key = identity(binding)
    write(out / 'evaluation-identity.json', binding)
    response = dict(schema='orbit.search.feedback.v1', backend='static',
        candidate_id=program['candidate_id'], source_id=program['source_id'],
        evaluation_kind='static_estimate', executed=False, status='unsupported',
        score=None, cache_identity=key, cache_hit=False)
    cache = Path(target['cache_dir']) / (key + '.json')
    result = None
    if cache.exists():
        try:
            saved = load(cache)
            candidate = saved['result']
            best = candidate['best_plan']
            if saved['identity'] != binding or saved['result_sha256'] != identity(candidate):
                raise ValueError('cache identity/content mismatch')
            checked = check_plan(program, config, best)
            if not checked['passed']:
                raise ValueError('cached plan failed independent check')
            result = candidate
            response['cache_hit'] = True
        except (KeyError, ValueError, TypeError, OSError) as error:
            write(out / 'cache-rejection.json', dict(reason=str(error)))
    if result is None:
        result = map_program(program, config)
    write(out / 'mapping-result.json', result)
    response['evidence_path'] = str(out / 'mapping-result.json')
    best = result.get('best_plan')
    if result['status'] == 'passed' and best:
        checked = check_plan(program, config, best)
        write(out / 'independent-check.json', checked)
        if not checked['passed']:
            response.update(status='runtime_failed', reason='independent static plan check failed')
        else:
            response.update(status='passed', plan_id=best['plan_id'],
                score=dict(value=best['makespan_ms'], unit='ms', provenance='static-plan-estimate', physical_hardware_measurement=False))
            write(cache, dict(identity=binding, result=result, result_sha256=identity(result)))
    else:
        response.update(reason=result.get('reasons', result.get('reason', 'no scored feasible plan')),
                        planning_status=result['status'])
    return response


def main():
    request = load(sys.argv[1])
    try:
        response = evaluate(request)
    except Exception as error:
        response = dict(schema='orbit.search.feedback.v1', backend='static',
            candidate_id=request.get('candidate_id'), source_id=request.get('source_id'),
            evaluation_kind='static_estimate', executed=False, status='unsupported', score=None,
            reason=type(error).__name__ + ': ' + str(error))
    write(sys.argv[2], response)


if __name__ == '__main__':
    main()
