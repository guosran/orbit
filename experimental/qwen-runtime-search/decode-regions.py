#!/usr/bin/env python3
"""Materialize source-bound fused-decode wave regions without editing the vendor.

A region covers complete existing decode waves. Its implementation still includes
all operators within each covered layer; this is not arbitrary operator fission.
The kernel chain preserves the original dispatch's embedding/RMS/KV preparation
and logits readback. Intermediate regions share device-valid BOs and add real
invocation events, not invented host transfers.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_file(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def public_abi(text):
    lines = re.findall(r"^\s*func\.func @[^\n]+", text, re.MULTILINE)
    if len(lines) != 1:
        raise ValueError("expected one public AIR entrypoint")
    return lines[0].strip()


def device_files(directory):
    """CDO configuration and compiled cores must agree before sharing one kernel.

    Instruction streams differ intentionally. Source/ABI identity is checked
    separately; a matching filename or xclbin UUID alone proves neither.
    """
    directory = Path(directory)
    records = {}
    for path in directory.rglob('*'):
        if path.is_file() and (path.name.endswith('.elf') or
                              (path.name.endswith('.bin') and 'cdo_' in str(path))):
            records[str(path.relative_to(directory))] = sha(path)
    if not records or not any('cdo_' in p for p in records):
        raise ValueError('missing CDO/core evidence: ' + str(directory))
    return records


def compile_region(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve(strict=True)
    if source.name != 'fused_decode.py':
        raise ValueError('expected the existing fused_decode.py builder')
    before = source_file(source)
    # These are the existing shipping model/ABI settings, not a reduced model.
    environment = dict(DECODE_MODEL='qwen2.5-3b', UNIFIED='1', LM_HEAD='0',
                       NLAYERS='1', DECODE_GOLDEN='1',
                       DECODE_GOLDEN_L=str(args.length), DECODE_DYNSEQ='0',
                       DECODE_RT_ROUNDS='0', AIR_TARGET_DEVICE='npu2')
    os.environ.update(environment)
    sys.path.insert(0, str(source.parent))
    fd = load_module(source, 'orbit_region_fused_decode')
    if (fd.UNI_DEC, fd.UNI_LM) != (36, 25):
        raise ValueError('full Qwen2.5-3B ABI changed')
    if not 0 <= args.lo < args.hi <= fd.UNI_WAVES:
        raise ValueError('region is outside the full source wave domain')
    # These pre-existing override hooks alter only the launch loop's bounds.
    fd.UNI_WAVE_LO, fd.UNI_WAVE_HI = args.lo, args.hi
    links = {}
    for name in ('proj_qmm.o', 'rms_residual.o', 'glu.o', 'rope.o',
                 'attn_qk.ll', 'attn_kv.ll'):
        path = (args.kernel_dir / name).resolve(strict=True)
        links[name] = source_file(path)
        (output / name).symlink_to(path)
    write_new(output / 'request.json', dict(
        schema='orbit.qwen-decode-region-build.v1', source=before,
        wrapper=source_file(__file__), environment=environment,
        region=dict(lo=args.lo, hi=args.hi), decoder_layers=36, vocab_waves=25,
        length=args.length, kernel_inputs=links,
        timing_boundary='device-valid shared BO region invocation; submit+wait inclusive'))
    os.chdir(output)
    module = fd.build_module()
    (output / 'source.air.mlir').write_text(str(module))
    if args.emit_only:
        return
    from air.backend.xrt import XRTBackend
    backend = XRTBackend(
        omit_while_true_loop=False, output_format='xclbin',
        kernel_name='MLIR_AIE', instance_name='', stack_size=fd.STACK_SIZE,
        use_lock_race_condition_fix_v2=True,
        coalesce_shim_dma=bool(fd.COALESCE))
    artifact = backend.compile(module, output_binary_name='decode',
                               insts='decode.insts.bin')
    if source_file(source) != before:
        raise ValueError('vendor source changed during compilation')
    write_new(output / 'compiled.json', dict(
        schema='orbit.qwen-decode-region-compiled.v1',
        request=source_file(output / 'request.json'), source=before,
        region=dict(lo=args.lo, hi=args.hi), decoder_layers=36, vocab_waves=25,
        length=args.length, source_air=source_file(output / 'source.air.mlir'),
        abi=public_abi(str(module)),
        binary=source_file(artifact.output_binary),
        instructions=source_file(artifact.insts),
        device_files=device_files(output / 'air_project')))


def seal_pair(args):
    """Package two native builds for the unchanged vendor instruction generator."""
    builds = [json.loads((p / 'compiled.json').read_text()) for p in args.build]
    ref = device_files(args.reference_device_dir)
    reference_abi = public_abi((args.reference_device_dir.parent / 'air.mlir').read_text())
    first = builds[0]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if len(builds) != 2 or len({b['length'] for b in builds}) != 2:
        raise ValueError('two distinct context builds required')
    for parent, b in zip(args.build, builds):
        if (b['source'] != first['source'] or b['region'] != first['region'] or
                b['decoder_layers'] != 36 or b['vocab_waves'] != 25):
            raise ValueError('foreign region/model/source in template pair')
        if (b['length'] + 15) // 16 != (first['length'] + 15) // 16:
            raise ValueError('pair changes physical KV capacity')
        if b['abi'] != reference_abi:
            raise ValueError('region argument ABI differs from original')
        if b['device_files'] != ref:
            raise ValueError('region CDO/core identity differs from reference')
        for role in ('binary', 'instructions'):
            p = Path(b[role]['path'])
            if sha(p) != b[role]['sha256']:
                raise ValueError('compiled artifact changed')
        stem = f"decode_L{b['length']}"
        (out / (stem + '.xclbin')).symlink_to(b['binary']['path'])
        (out / (stem + '.insts.bin')).symlink_to(b['instructions']['path'])
    write_new(out / 'region.json', dict(
        schema='orbit.qwen-decode-region-template.v1', source=first['source'],
        region=first['region'], decoder_layers=36, vocab_waves=25,
        capacity=16 * ((first['length'] + 15) // 16),
        reference_device=str(args.reference_device_dir.resolve()),
        device_files=ref, abi=reference_abi,
        reference_binaries=[source_file(p) for p in args.reference_binary], builds=[source_file(p / 'compiled.json') for p in args.build]))


class _CompletedChain:
    def __init__(self, chain):
        self.chain = chain

    def wait(self, *unused):
        # Work is already synchronized by the ordered chain. The return value
        # retains the original dispatch's completed-state validation.
        return self.chain.last_state


class RegionKernelChain:
    """Run original kernel with region streams and the original shared BOs.

    Each region's service time includes device computation and invocation. The
    reported submit and wait pieces partition that service; callers must never
    add service again as a separate graph-edge launch cost.
    """
    def __init__(self, decoder, template_dirs, context_length=None):
        import numpy as np
        self.decoder, self.kernel, self.np = decoder, decoder.kern, np
        self.xrt = decoder.xrt
        self.context_length = context_length
        self.events, self.states, self.receipts = [], [], []
        self.last_state = None
        self.last_call_start_ns = self.last_call_end_ns = None
        gen_type = type(decoder.gen)
        if decoder.elf_mode:
            raise ValueError('region stream chain requires the existing xclbin path')
        expected = 0
        for directory in map(Path, template_dirs):
            receipt_path = directory / 'region.json'
            receipt = json.loads(receipt_path.read_text())
            if sha(receipt['source']['path']) != receipt['source']['sha256']:
                raise ValueError('vendor decode source changed after compilation')
            lo, hi = (receipt['region'][k] for k in ('lo', 'hi'))
            if lo != expected or not lo < hi <= 61:
                raise ValueError('region cover must be ordered, gap-free, nonoverlapping')
            expected = hi
            original_binary = decoder.gen.xclbin_for_maxl(decoder.ATTN_MAXL)
            if sha(original_binary) not in {x['sha256'] for x in receipt['reference_binaries']}:
                raise ValueError('original decoder kernel is not the certified reference binary')
            if receipt['capacity'] != decoder.ATTN_MAXL:
                raise ValueError('region KV capacity differs from allocated decoder BOs')
            for record in receipt['builds']:
                if sha(record['path']) != record['sha256']:
                    raise ValueError('region compiled receipt changed')
                build = json.loads(Path(record['path']).read_text())
                if build['device_files'] != receipt['device_files']:
                    raise ValueError('region device receipt changed')
                for role in ('binary', 'instructions'):
                    if sha(build[role]['path']) != build[role]['sha256']:
                        raise ValueError('region binary or stream changed')
            # Reuse the vendor's L-word identification/generation unchanged.
            gen = gen_type(str(directory), max_L=decoder.ATTN_MAXL)
            i1, i2 = gen.insts_for(decoder.ATTN_MAXL, 1), gen.insts_for(decoder.ATTN_MAXL, 2)
            ld = np.flatnonzero(i1 != i2)
            state = dict(gen=gen, ld=ld, lo=int(ld.min()) if ld.size else 0,
                         hi=int(ld.max()) + 1 if ld.size else 0,
                         buf=i1.copy(), ib=self.xrt.bo(decoder.dev, i1.nbytes,
                         self.xrt.bo.cacheable, self.kernel.group_id(1)), primed=False)
            self.states.append(state)
            self.receipts.append(receipt)
        if expected != 61:
            raise ValueError('region chain omits decoder or LM-head waves')
        # The physical device configured by the saved original kernel must be
        # the reference used to certify every region, not just pairwise equal.
        if len({json.dumps(r['device_files'], sort_keys=True) for r in self.receipts}) != 1:
            raise ValueError('region templates configure different devices')

    def group_id(self, index):
        return self.kernel.group_id(index)

    def __call__(self, opcode, original_ib, original_size, *bos):
        self.last_call_start_ns = time.perf_counter_ns()
        if opcode != 3 or self.context_length is None:
            raise ValueError('unsupported dispatch ABI/context')
        expected = (self.decoder.x_bo, self.decoder.w_bo, self.decoder.r_bo,
                    self.decoder.y_bo, self.decoder.kvc) + tuple(self.decoder.kvb.extra)
        if len(bos) != len(expected) or any(a is not b for a, b in zip(bos, expected)):
            raise ValueError('region chain must retain original shared BO identities')
        TO = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        for state, receipt in zip(self.states, self.receipts):
            begin = time.perf_counter_ns()
            if not state['primed']:
                state['buf'][:] = state['gen'].insts_for(self.decoder.ATTN_MAXL, self.context_length)
                state['ib'].write(state['buf'], 0)
                state['ib'].sync(TO)
                state['primed'] = True
            elif state['ld'].size:
                lo, hi = state['lo'], state['hi']
                state['buf'][state['ld']] = state['gen'].words_for(
                    self.decoder.ATTN_MAXL, self.context_length, state['ld'])
                state['ib'].write(state['buf'][lo:hi], lo * 4)
                state['ib'].sync(TO, (hi - lo) * 4, lo * 4)
            patched = time.perf_counter_ns()
            run = self.kernel(3, state['ib'], int(state['buf'].size), *bos)
            submitted = time.perf_counter_ns()
            status = run.wait(60000)
            finished = time.perf_counter_ns()
            if not str(status).endswith('COMPLETED'):
                raise RuntimeError('region kernel failed: ' + str(status))
            self.last_state = status
            self.events.append(dict(region=receipt['region'], length=self.context_length,
                instruction_patch_ns=patched-begin, submit_ns=submitted-patched,
                wait_ns=finished-submitted, service_ns=finished-patched,
                shared_bo_transfer_bytes=0, boundary='device-valid shared BO service, submit+wait inclusive'))
        self.last_call_end_ns = time.perf_counter_ns()
        return _CompletedChain(self)


class RegionDecoder:
    """Composition adapter: prepare once through original decoder, then chain."""
    def __init__(self, decoder, template_dirs):
        self.decoder = decoder
        self.chain = RegionKernelChain(decoder, template_dirs)
        self.dispatch_events = []

    def __getattr__(self, name):
        return getattr(self.decoder, name)

    def dispatch(self, token, position):
        if self.decoder.kern is not self.chain.kernel:
            raise ValueError('decoder kernel/window changed after region binding')
        self.chain.context_length = position + 1
        event_start = len(self.chain.events)
        start = time.perf_counter_ns()
        previous = self.decoder.kern
        self.decoder.kern = self.chain
        try:
            logits = self.decoder.dispatch(token, position)
        finally:
            self.decoder.kern = previous
        end = time.perf_counter_ns()
        events = self.chain.events[event_start:]
        region_total = sum(e['instruction_patch_ns'] + e['submit_ns'] + e['wait_ns']
                           for e in events)
        call_start, call_end = self.chain.last_call_start_ns, self.chain.last_call_end_ns
        before, after = call_start-start, end-call_end
        control = call_end-call_start-region_total
        if min(before, after, control) < 0 or before+region_total+control+after != end-start:
            raise ValueError('exclusive dispatch trace does not conserve wall time')
        self.dispatch_events.append(dict(
            schema='orbit.qwen-decode-dispatch-phases.v1', position=position,
            length=position+1, wall_ns=end-start, host_before_ns=before,
            host_after_ns=after, host_control_ns=control, region_events=events,
            exclusive_sum_equals_wall=True,
            boundary='original complete dispatch including preparation, all regions, readback',
            cost_ownership='region service=submit+wait; patch separate; host groups inclusive, never spread among internal operators'))
        return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    build = sub.add_parser('build')
    build.add_argument('--source', type=Path, required=True)
    build.add_argument('--kernel-dir', type=Path, required=True)
    build.add_argument('--lo', type=int, required=True)
    build.add_argument('--hi', type=int, required=True)
    build.add_argument('--length', type=int, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--emit-only', action='store_true')
    pair = sub.add_parser('seal-pair')
    pair.add_argument('--build', type=Path, nargs=2, required=True)
    pair.add_argument('--reference-device-dir', type=Path, required=True)
    pair.add_argument('--reference-binary', type=Path, nargs='+', required=True)
    pair.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compile_region(args) if args.phase == 'build' else seal_pair(args)


if __name__ == '__main__':
    main()
