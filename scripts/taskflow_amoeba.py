#!/usr/bin/env python3
"""Consume a materialized public Taskflow candidate with the existing Amoeba tool.

The host replay is CPU correctness evidence, never a CGRA measurement. Neura
lowering is attempted separately and its unsupported/failed state is retained.
No legacy model score is synthesized from host milliseconds.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, directory, name, timeout=180):
    started = time.perf_counter()
    try:
        p = subprocess.run([str(x) for x in argv], capture_output=True, text=True,
                           timeout=timeout)
        status, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        status, stdout, stderr = 124, e.stdout or '', e.stderr or ''
        if isinstance(stdout, bytes): stdout = stdout.decode(errors='replace')
        if isinstance(stderr, bytes): stderr = stderr.decode(errors='replace')
    (directory / (name + '.stdout')).write_text(stdout)
    (directory / (name + '.stderr')).write_text(stderr)
    return {'argv': [str(x) for x in argv], 'exit_code': status,
            'wall_ms': (time.perf_counter() - started) * 1000,
            'stdout': str(directory / (name + '.stdout')),
            'stderr': str(directory / (name + '.stderr'))}


def reference(plan, inputs):
    """NumPy kernels on the ORIGINAL untransformed source graph."""
    import numpy as np
    arrays = {b['id']: (inputs[b['id']].astype(np.float64) if b['role'] == 'input'
                      else np.zeros(b['shape'], np.float64)) for b in plan['buffers']}
    for task in plan['tasks']:
        for op in task['operations']:
            r, c = slice(*op['m']), slice(*op['n'])
            out = arrays[op['output']]
            if op['kind'] == 'fill': out[r, c] = 0
            elif op['kind'] == 'matmul':
                k = slice(*op['k'])
                value = arrays[op['inputs'][0]][r, k] @ arrays[op['inputs'][1]][k, c]
                if op['accumulate']: out[r, c] += value
                else: out[r, c] = value
            elif op['kind'] == 'relu':
                out[r, c] = np.maximum(arrays[op['inputs'][0]][r, c], 0)
            elif op['kind'] == 'swiglu':
                gate, up = (arrays[x][r, c] for x in op['inputs'])
                out[r, c] = gate / (1 + np.exp(-gate)) * up
            else: raise ValueError('unsupported source operation: ' + op['kind'])
    return arrays


def replay(library, plan, source, directory, seeds):
    import numpy as np
    class MemRef(ctypes.Structure):
        _fields_ = [('allocated', ctypes.POINTER(ctypes.c_float)),
                    ('aligned', ctypes.POINTER(ctypes.c_float)),
                    ('offset', ctypes.c_int64), ('sizes', ctypes.c_int64 * 2),
                    ('strides', ctypes.c_int64 * 2)]
    dll = ctypes.CDLL(str(library.resolve()))
    fn = dll._mlir_ciface_main
    fn.argtypes = [ctypes.POINTER(MemRef)] * len(plan['buffers'])
    fn.restype = None
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        arrays = {}
        for b in plan['buffers']:
            a = rng.uniform(-.5, .5, b['shape']).astype(np.float32) if b['role'] == 'input' else np.zeros(b['shape'], np.float32)
            if b['role'] == 'input': a.flat[::11] = 0
            arrays[b['id']] = a
        expected = reference(source, arrays)
        descs = [MemRef(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 0,
                        (ctypes.c_int64 * 2)(*a.shape),
                        (ctypes.c_int64 * 2)(a.shape[1], 1))
                 for a in arrays.values()]
        fn(*(ctypes.byref(d) for d in descs))
        samples = []
        repeated_correct = True
        for _ in range(3):
            # Poison recycled output/temporary storage outside timing. The
            # executable must initialize its reduction, not rely on old zeros.
            for b in plan['buffers']:
                if b['role'] != 'input': arrays[b['id']].fill(np.nan)
            start = time.perf_counter()
            fn(*(ctypes.byref(d) for d in descs))
            samples.append((time.perf_counter() - start) * 1000)
            repeated_correct &= all(np.isfinite(arrays[n]).all() and
                np.allclose(arrays[n], expected[n], atol=2e-5, rtol=2e-5)
                for n in plan['outputs'])
        outputs = []
        for name in plan['outputs']:
            a, e = arrays[name], expected[name]
            finite = bool(np.isfinite(a).all() and np.isfinite(e).all())
            outputs.append({'buffer': name, 'shape': list(a.shape),
                            'finite': finite, 'max_abs_error': float(np.max(np.abs(a-e))),
                            'normalized_rms': float(np.linalg.norm(a-e) / max(np.linalg.norm(e), 1e-300)),
                            'passed': finite and bool(np.allclose(a, e, atol=2e-5, rtol=2e-5)) and bool(repeated_correct)})
        np.savez(directory / f'seed-{seed}.npz', **arrays,
                 **{'reference_' + name: expected[name] for name in plan['outputs']})
        results.append({'seed': seed, 'outputs': outputs, 'host_wall_ms': samples,
                        'poisoned_buffer_repeats_passed': bool(repeated_correct)})
    return {'status': 'passed' if all(o['passed'] for r in results for o in r['outputs']) else 'numerical_failure',
            'execution_target': 'CPU', 'reference': 'NumPy on original source plan',
            'atol': 2e-5, 'rtol': 2e-5, 'seeds': results,
            'timing_scope': 'complete synchronous compiled task graph; one warmup, three samples'}


def map_candidate(args, plan, directory, stages, commands):
    """Use the unchanged source-owned Neura resource enumerator and mapper.

    The bounded resource search admits one CGRA per target stage. Its legal
    physical PE rectangle and trip count come from the architecture and source
    certificates, never from invented model costs.
    """
    flags = ['construct-hyperblock-from-task', 'classify-task-and-counter',
             'convert-taskflow-to-neura', 'cse', 'lower-affine',
             'convert-scf-to-cf', 'convert-cf-to-llvm', 'assign-accelerator',
             'lower-memref-to-neura', 'lower-arith-to-neura',
             'lower-builtin-to-neura', 'lower-llvm-to-neura',
             'promote-input-arg-to-const', 'fold-constant',
             'canonicalize-return', 'canonicalize-live-in',
             'leverage-predicated-value', 'transform-ctrl-to-data-flow',
             'fold-constant', 'insert-data-mov', 'bind-source-iteration-domain']
    d = directory.resolve()
    common = [args.optimizer, '--verify-each',
              '--neura-architecture-spec=' + str(args.architecture)]
    mapped = d / 'mapped-scheduled.mlir'
    attempts = [
        ('mapper-prepare', common + [stages, *['--' + f for f in flags],
                                     '-o', d / 'mapper-prepared.mlir']),
        ('mapper-enumerate', common + [d / 'mapper-prepared.mlir',
             '--enumerate-analytical-task-candidates=function=main output=' +
             str(d / 'mapper-candidates.jsonl') +
             ' max-candidates=10000 max-cgras-per-task=1',
             '-o', d / 'mapper-enumerated.mlir']),
        ('mapper-materialize', common + [d / 'mapper-enumerated.mlir',
             '--materialize-analytical-task-candidate=function=main candidates=' +
             str(d / 'mapper-candidates.jsonl') + ' candidate-id=candidate-0',
             '-o', d / 'mapper-resource-plan.mlir']),
        ('mapper-schedule', common + [d / 'mapper-resource-plan.mlir',
             '--pass-pipeline=builtin.module(map-joint-scheduling-tasks{function=main '
             'candidate-id=candidate-0 all-unit-baseline=true mapping-cache-dir=' +
             str(d / 'mapper-cache') + '},func.func(orchestrate-tasks-on-accelerators{'
             'orchestration-strategy=analytical-based-task-orchestration '
             'communication-mode=explicit dispatch-policy=fixed}))', '-o', mapped])]
    record = {'public_candidate_id': plan['candidate_id'],
              'resource_candidate_id': 'candidate-0', 'commands': [],
              'physical_cgra_executed': False,
              'resource_policy': 'source-owned enumeration with max-cgras-per-task=1'}
    for label, argv in attempts:
        result = command(argv, d, label, timeout=60)
        commands.append(result)
        record['commands'].append(result)
        if result['exit_code'] != 0:
            record.update(status='compile_failure', failed_stage=label,
                          reason=Path(result['stderr']).read_text()[-4000:])
            break
    else:
        text = mapped.read_text()
        makespan = re.search(r'joint_scheduling_actual_makespan = (\d+) : i64', text)
        iis = re.findall(r'actual_mapper_ii = (\d+) : i64', text)
        if not makespan or not iis:
            record.update(status='compile_failure', reason='missing certified mapper/scheduler results')
        else:
            record.update(status='passed', ir=str(mapped),
                          actual_mapper_ii_cycles=[int(x) for x in iis],
                          scheduler_makespan_cycles=int(makespan.group(1)),
                          duration_provenance='source-owned-all-unit-mapped-duration',
                          evidence_type='native mapper and production scheduler trace; no physical CGRA')
    (d / 'mapping.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--mlir', type=Path, required=True)
    p.add_argument('--source-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    root = Path(__file__).resolve().parents[1]
    p.add_argument('--optimizer', type=Path, default=root / 'build-amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt')
    p.add_argument('--core-tool', type=Path, default=root / 'build-core/tools/taskflow-opt/taskflow-opt')
    p.add_argument('--llvm-build', type=Path, default=Path('/home/shiran/project/llvm-project/build'))
    p.add_argument('--architecture', type=Path, default=root / 'test/multi-cgra/taskflow/joint-scheduling/mapper-arch-4x4.yaml')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--map', action='store_true', help='Run bounded legacy Neura resource enumeration, mapper and scheduler')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan, source = json.loads(args.plan.read_text()), json.loads(args.source_plan.read_text())
    if plan['schema'] != 'orbit.taskflow.plan.v1' or source['schema'] != plan['schema']:
        p.error('unsupported plan schema')
    if plan['source_id'] != source['source_id']:
        p.error('candidate and reference source identities differ')
    if any(b['dtype'] != 'f32' or len(b['shape']) != 2 for b in plan['buffers']):
        p.error('Amoeba host replay currently supports only rank-2 f32 buffers')
    out = args.output
    report = {'schema': 'orbit.amoeba.feedback.v1', 'candidate_id': plan['candidate_id'],
              'source_id': plan['source_id'], 'plan_sha256': digest(args.plan),
              'materialized_mlir_sha256': digest(args.mlir), 'evidence_type': 'compiler-and-CPU-replay',
              'physical_cgra_executed': False, 'score': {'status': 'unscored',
              'reason': 'CPU replay time is not Neura II or scheduler cycles'},
              'commands': []}
    def run(name, cmd):
        result = command(cmd, out, name)
        report['commands'].append(result)
        return result['exit_code'] == 0
    (out / 'input.taskflow.mlir').write_bytes(args.mlir.read_bytes())
    (out / 'input.plan.json').write_bytes(args.plan.read_bytes())
    base = [args.optimizer, args.mlir, '--verify-each']
    try:
        authenticated = out / 'authenticated.plan.json'
        if not run('authenticate', [args.core_tool, args.mlir, '--emit-plan=' + str(authenticated)]):
            report['status'] = 'unsupported'
            report['reason'] = 'public Taskflow authentication failed'
            return 1
        if json.loads(authenticated.read_text()) != plan:
            report['status'] = 'unsupported'
            report['reason'] = 'supplied plan differs from authenticated Taskflow program'
            return 1
        if not run('verify', base + ['-o', out / 'verified.mlir']):
            report['status'] = 'compile_failure'
            return 1
        neura = out / 'neura.mlir'
        stages = out / 'target-stages.mlir'
        stages_ok = run('prepare-neura-stages', base + ['--prepare-structured-taskflow-for-neura', '-o', stages])
        report['target_stages'] = {'status': 'passed' if stages_ok else 'unsupported',
                                  'ir': str(stages) if stages_ok else None}
        pipeline = ('builtin.module(func.func(construct-hyperblock-from-task),'
                    'classify-task-and-counter,convert-taskflow-to-neura,'
                    'lower-affine-to-neura,bind-source-iteration-domain)')
        lower_ok = stages_ok and run('neura-lowering', [args.optimizer, stages, '--verify-each'] + ['--neura-architecture-spec=' + str(args.architecture),
                        '--pass-pipeline=' + pipeline, '-o', neura])
        lowered_text = neura.read_text() if lower_ok else ''
        kernels = len(re.findall(r'^\s*neura\.kernel\b', lowered_text, re.MULTILINE))
        complete = lower_ok and kernels > 0 and 'affine.for' not in lowered_text
        lowering_status = 'passed' if complete else ('unsupported' if not stages_ok or lower_ok else 'compile_failure')
        report['neura_lowering'] = {'status': lowering_status,
                                   'ir': str(neura) if lower_ok else None, 'kernel_count': kernels,
                                   'scope': 'Taskflow counters and Neura kernel construction, before hardware mapping'}
        report['mapping'] = {'status': 'not_requested' if not args.map else 'unsupported',
                             'reason': 'optional bounded mapping was not requested' if not args.map else 'target staging is unsupported'}
        if args.map and stages_ok:
            report['mapping'] = map_candidate(args, plan, out, stages, report['commands'])
            if report['mapping']['status'] == 'passed':
                report['score'] = {'status': 'scored', 'target': 'Neura',
                                   'value': report['mapping']['scheduler_makespan_cycles'],
                                   'unit': 'cycles', 'provenance': 'source-owned-all-unit-mapped-duration',
                                   'evidence_type': 'mapper plus scheduler estimate',
                                   'physical_hardware_measurement': False}
        host = out / 'host.mlir'
        host_source = stages if stages_ok else args.mlir
        report['replay_source'] = str(host_source)
        if not run('host-lowering', [args.optimizer, host_source, '--verify-each'] + ['--lower-affine', '--lower-joint-taskflow-to-host-scf', '-o', host]):
            report['status'] = 'host_lowering_failure'
            return 1
        llvm_ir = out / 'llvm.mlir'
        if not run('llvm-lowering', [args.llvm_build / 'bin/mlir-opt', host, '--convert-scf-to-cf',
                '--convert-math-to-llvm', '--convert-arith-to-llvm', '--finalize-memref-to-llvm',
                '--convert-func-to-llvm', '--convert-cf-to-llvm', '--reconcile-unrealized-casts', '-o', llvm_ir]):
            report['status'] = 'compile_failure'
            return 1
        ll = out / 'host.ll'
        if not run('llvm-translate', [args.llvm_build / 'bin/mlir-translate', '--mlir-to-llvmir', llvm_ir, '-o', ll]):
            report['status'] = 'compile_failure'
            return 1
        so = out / 'host.so'
        if not run('clang', [args.llvm_build / 'bin/clang', '-O2', '-shared', '-fPIC', ll, '-lm', '-o', so]):
            report['status'] = 'compile_failure'
            return 1
        report['replay'] = replay(so, plan, source, out, [int(x) for x in args.seeds.split(',')])
        report['status'] = report['replay']['status']
    except Exception as e:
        report['status'] = 'adapter_failure'
        report['reason'] = str(e)
    finally:
        # Mapping is not an accepted candidate until its executable semantic
        # replay passes. Keep the mapping evidence, but never rank failed work.
        if report.get('status') != 'passed':
            report['score'] = {'status': 'unscored',
                               'reason': 'candidate validation did not pass: ' +
                                         str(report.get('status'))}
        (out / 'feedback.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report.get('status'), 'feedback': str(out / 'feedback.json')}))
    return 0 if report.get('status') == 'passed' else 1

if __name__ == '__main__':
    raise SystemExit(main())
