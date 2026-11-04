#!/usr/bin/env python3
"""Compile existing C++-generated AIR and execute using the installed XRT runner.

This tool contains no graph analysis, mapping, lowering, or numerical reference.
The ABI describes buffers only; references must be independently supplied.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def compile_air(source, out, bf16_emulation=False, stack_size=4096, omit_ping_pong=False):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, out / 'input.air.mlir')
    elf = out / 'program.elf'
    command = [shutil.which('aircc') or 'aircc', '--device', 'npu2',
               '--no-xchesscc', '--no-xbridge', '--output-format', 'elf',
               '--elf-name', str(elf), '--peano=' + os.environ['PEANO_INSTALL_DIR'],
               '--stack-size=' + str(stack_size)]
    if bf16_emulation:
        command.append('--bf16-emulation')
    if omit_ping_pong:
        command.append('--omit-ping-pong-transform=all')
    command.append(str(out / 'input.air.mlir'))
    save(out / 'command.json', command)
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                if any(k in d.metadata['Name'].lower() for k in ('mlir-air', 'mlir-aie', 'llvm-aie', 'triton-xdna'))}
    save(out / 'toolchain.json', versions)
    start = time.perf_counter()
    with (out / 'compile.log').open('w') as log:
        result = subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT)
    record = dict(status='compiled' if result.returncode == 0 else 'compile_failed',
                  returncode=result.returncode, seconds=time.perf_counter()-start,
                  source_sha256=digest(source), bf16_emulation=bf16_emulation,
                  stack_size_bytes=stack_size, omit_ping_pong=omit_ping_pong)
    save(out / 'compile.json', record)
    if result.returncode:
        raise RuntimeError('AIR compilation failed; see ' + str(out / 'compile.log'))
    config = json.loads((out / 'air_project/full_elf_config.json').read_text())
    main = next(k for k in config['xrt-kernels'] if k['name'] == 'main')
    record.update(elf_path=str(elf), elf_sha256=digest(elf),
                  kernel_name='main:' + str(main['instance'][0]['id']),
                  argument_count=len(main['arguments']))
    save(out / 'compile.json', record)
    return record


def run_artifact(artifact, abi, inputs, reference, out, iterations, atol, rtol, warmups=2):
    import numpy as np
    from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner
    from triton.backends.amd_triton_npu import multilaunch
    from taskflow_npu_trace_v3 import ObservedXRT, ObservedRun
    from ml_dtypes import bfloat16
    arrays = []
    types = {'f32': np.float32, 'bf16': bfloat16, 'i32': np.int32}
    for b in abi['buffers']:
        dtype = types[b['dtype']]
        value = inputs.get(b['name'], np.zeros(b['shape'], dtype=dtype))
        value = np.ascontiguousarray(value, dtype=dtype)
        if list(value.shape) != b['shape']:
            raise ValueError('input shape differs from compiled ABI: ' + b['name'])
        arrays.append(value)
    output_indices = {i for i, b in enumerate(abi['buffers']) if b['name'] in abi['outputs']}
    observations = []
    start = time.perf_counter()
    runner = MultiLaunchRunner(artifact['elf_path'], artifact['kernel_name'])
    load_ms = (time.perf_counter()-start)*1000
    runner._xrt = ObservedXRT(runner._xrt, observations)
    original_wait = multilaunch._wait
    def observed_wait(run):
        begin = time.perf_counter_ns()
        save(out / 'wait-in-progress.json', dict(status='waiting_for_actual_xrt',
             start_ns=begin, observed_events=list(observations)))
        try:
            value = original_wait(run.raw if isinstance(run, ObservedRun) else run)
        except Exception as error:
            observations.append(dict(kind='actual_xrt_wait_failed', start_ns=begin,
                                     end_ns=time.perf_counter_ns()))
            save(out / 'wait-failure.json', dict(status='actual_xrt_wait_failed',
                 message=str(error), observed_events=list(observations)))
            raise
        observations.append(dict(kind='actual_xrt_wait', start_ns=begin, end_ns=time.perf_counter_ns()))
        (out / 'wait-in-progress.json').unlink()
        return value
    multilaunch._wait = observed_wait
    samples = []
    try:
        for repeat in range(iterations + warmups):
            observations.clear()
            begin = time.perf_counter_ns()
            values = runner.run(arrays, bo_key='buffers', output_indices=output_indices)
            outputs = {abi['buffers'][i]['name']: values[i].reshape(arrays[i].shape).copy() for i in output_indices}
            wall = (time.perf_counter_ns()-begin)/1e6
            checks = {}
            for name, actual in outputs.items():
                expected = reference[name]
                delta = actual.astype(np.float64)-expected.astype(np.float64)
                finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
                checks[name] = dict(shape=list(actual.shape), finite=finite,
                                    max_abs=float(np.max(np.abs(delta))) if finite else None,
                                    normalized_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(expected.astype(np.float64)), 1e-30)) if finite else None,
                                    passed=finite and actual.shape == expected.shape and bool(np.allclose(actual, expected, atol=atol, rtol=rtol)))
            np.savez(out / f'outputs-{repeat}.npz', **outputs)
            samples.append(dict(repeat=repeat, warmup=repeat<warmups, wall_ms=wall, checks=checks,
                                api_events=[{**e, 'start_ns': e['start_ns']-begin, 'end_ns': e['end_ns']-begin} for e in observations]))
    finally:
        multilaunch._wait = original_wait
        del runner
    record = dict(status='passed' if all(c['passed'] for s in samples for c in s['checks'].values()) else 'numeric_failed',
                  load_ms=load_ms, samples=samples, median_ms=float(np.median([s['wall_ms'] for s in samples if not s['warmup']])),
                  atol=atol, rtol=rtol, boundary='existing runner host input staging and synchronization, submit/wait including wait-event checkpoint serialization, output sync/readback and output copy; context creation, numpy input preparation, numerical validation and NPZ result serialization excluded',
                  storage='system-memory XRT BOs, not persistent AIE SRAM',
                  precision='ABI element type; bf16 emulation only if compile record says enabled',
                  actual_submissions=[sum(e['kind']=='actual_xrt_start' for e in s['api_events']) for s in samples])
    save(out / 'result.json', record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--air', type=Path, required=True)
    p.add_argument('--abi', type=Path, required=True)
    p.add_argument('--inputs', type=Path, action='append', default=[])
    p.add_argument('--reference', type=Path, action='append', default=[])
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--compile-only', action='store_true')
    p.add_argument('--bf16-emulation', action='store_true')
    p.add_argument('--omit-ping-pong-transform', action='store_true')
    p.add_argument('--preflight-channel-plan', type=Path)
    p.add_argument('--preflight-runtime-plan', type=Path)
    p.add_argument('--stack-size', type=int, default=4096)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--diagnostic', action='store_true', help='explicit bounded diagnostic; permits fewer samples')
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--rtol', type=float, default=1e-5)
    a=p.parse_args()
    if a.stack_size<=0:p.error('stack size must be positive')
    if a.warmups<0 or a.iterations<(1 if a.diagnostic else 3) or (not a.diagnostic and a.warmups<2) or len(a.inputs)!=len(a.reference):p.error('need >=3 hot samples, >=2 warmups, and matching inputs; explicit --diagnostic permits fewer')
    if not a.compile_only and not a.inputs:p.error('execution needs input and independently computed reference')
    out=a.output_dir.resolve()
    artifact=compile_air(a.air.resolve(), out, a.bf16_emulation, a.stack_size, a.omit_ping_pong_transform)
    save(out/'measurement-mode.json',dict(diagnostic=a.diagnostic,warmups=a.warmups,hot_samples=a.iterations))
    shutil.copy2(a.abi, out/'abi.json');shutil.copy2(__file__,out/'executor-source.py')
    if a.preflight_channel_plan:
        # Validation-only static artifact audit, before any XRT context/device
        # submission. The native compiler already owns every port decision.
        audit=Path(__file__).resolve().parents[1]/'test/TaskflowAIR/audit_channel_ports_v6.py'
        subprocess.run([sys.executable,str(audit),str(out),'--plan',str(a.preflight_channel_plan.resolve()),
                        '--output',str(out.parent/'channel-port-audit.json')],check=True)
        storage_audit=audit.with_name('audit_storage_v6.py')
        subprocess.run([sys.executable,str(storage_audit),str(out),'--plan',str(a.preflight_channel_plan.resolve()),
                        '--output',str(out.parent/'storage-audit.json')],check=True)
    if a.preflight_runtime_plan:
        audit=Path(__file__).resolve().parents[1]/'test/TaskflowAIR/audit_runtime_transfers_v7.py'
        subprocess.run([sys.executable,str(audit),str(out),'--plan',str(a.preflight_runtime_plan.resolve()),'--abi',str(a.abi.resolve()),'--output',str(out.parent/'runtime-transfer-audit.json')],check=True)
    if a.compile_only:return 0
    if not a.inputs:p.error('execution needs input and independently computed reference')
    import numpy as np
    abi=json.loads(a.abi.read_text())
    # Installed aiecc pads ELF metadata to at least three pointer slots even
    # when the actual AIR function has fewer buffers. Those slots are unused.
    if len(abi['buffers']) > artifact['argument_count']:
        raise ValueError('buffer ABI exceeds compiled main kernel argument slots')
    results=[]
    with open('/tmp/npu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for i,(inp,ref) in enumerate(zip(a.inputs,a.reference)):
            folder=out/f'seed-{i}';folder.mkdir()
            shutil.copy2(inp,folder/'inputs.npz');shutil.copy2(ref,folder/'reference.npz')
            save(folder/'input-provenance.json',dict(inputs_sha256=digest(inp), reference_sha256=digest(ref)))
            try:
                results.append(run_artifact(artifact,abi,dict(np.load(inp)),dict(np.load(ref)),folder,a.iterations,a.atol,a.rtol,a.warmups))
            except Exception as error:
                save(folder/'failure.json',dict(status='execution_failed',type=type(error).__name__,message=str(error)))
                save(out/'results.json',results)
                raise
    save(out/'results.json',results)
    return 0 if all(r['status']=='passed' for r in results) else 1

if __name__=='__main__':raise SystemExit(main())
