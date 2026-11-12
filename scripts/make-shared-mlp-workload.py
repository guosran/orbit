#!/usr/bin/env python3
"""Generate the shared-runtime-weight MLP used by the v32 r12 experiment.

Requires NumPy, not a compiler or device. X and W contain raw uint16 BF16 bits;
Y is a sequential FP32 reference with explicit BF16 RNE between layers. Source
and numerical generation do not establish backend feasibility, model coverage,
or performance for new shapes/depths. The r12 family used width=16, layers=2
and batch=16/24/32. Output must be a new directory.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEED = 32012
ATOL = RTOL = 1e-4


def validate_shape(batch, width, layers):
    if batch < 1 or batch % 8:
        raise ValueError('batch must be a positive multiple of 8')
    # W is square: a positive multiple of 16 satisfies both N16 and K8.
    if width < 1 or width % 16:
        raise ValueError('width must be a positive multiple of 16 (N16/K8)')
    if layers < 1:
        raise ValueError('layers must be positive')


def task(name, reads, writes, arguments, types, body):
    """Emit one source task with its actual memory-state inputs and result."""
    def typed(values):
        return ', '.join(values) + ' : ' + ', '.join(types[v] for v in values)

    ordered = reads + writes
    arguments_list = ', '.join(f'{a}: {types[v]}'
                               for a, v in zip(arguments, ordered))
    return (f'    %{name}_done = taskflow.task @{name} '
            f'will_reads({typed(reads)}) will_writes({typed(writes)}) '
            f'[original_read_memrefs({typed(reads)}), '
            f'original_write_memrefs({typed(writes)})] : '
            f'({", ".join(types[v] for v in ordered)}) -> {types[writes[0]]} {{\n'
            f'    ^bb0({arguments_list}):\n{body}'
            f'      taskflow.yield done_writes({arguments[-1]} : {types[writes[0]]})\n'
            '    }\n')


def make_source(batch=16, width=16, layers=2):
    validate_shape(batch, width, layers)
    bf16 = f'memref<{batch}x{width}xbf16>'
    f32 = f'memref<{batch}x{width}xf32>'
    weight = f'memref<{width}x{width}xbf16>'
    types = {'%X': bf16, '%W': weight, '%Y': f32}
    source = f'''module {{
  func.func @shared_weight_mlp(%X: {bf16} {{orbit.role = "input", orbit.name = "X"}}, %W: {weight} {{orbit.role = "input", orbit.name = "W"}}, %Y: {f32} {{orbit.role = "output", orbit.name = "Y"}}) attributes {{orbit.noalias, llvm.emit_c_interface, orbit.matrix_contract = "bf16-f32-accumulate-v1"}} {{
'''
    allocations = []
    previous = '%X'
    for layer in range(layers):
        accumulator = f'%C{layer}'
        types[accumulator] = types[f'%layer{layer}_done'] = f32
        source += (f'    {accumulator} = memref.alloc() '
                   f'{{orbit.name = "C{layer}"}} : {f32}\n')
        allocations.append((accumulator, f32))
        body = f'''      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to {batch} {{
        affine.for %j = 0 to {width} {{
          affine.store %zero, %c[%i, %j] : {f32}
          affine.for %k = 0 to {width} {{
            %av = affine.load %a[%i, %k] : {bf16}
            %bv = affine.load %b[%k, %j] : {weight}
            %af = arith.extf %av : bf16 to f32
            %bf = arith.extf %bv : bf16 to f32
            %old = affine.load %c[%i, %j] : {f32}
            %product = arith.mulf %af, %bf : f32
            %sum = arith.addf %old, %product : f32
            affine.store %sum, %c[%i, %j] : {f32}
          }}
        }}
      }}
'''
        source += task(f'layer{layer}', [previous, '%W'], [accumulator],
                       ['%a', '%b', '%c'], types, body)
        previous = f'%layer{layer}_done'
        if layer + 1 < layers:
            hidden = f'%H{layer}'
            types[hidden] = types[f'%round{layer}_done'] = bf16
            source += (f'    {hidden} = memref.alloc() '
                       f'{{orbit.name = "H{layer}"}} : {bf16}\n')
            allocations.append((hidden, bf16))
            body = f'''      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to {batch} {{
        affine.for %j = 0 to {width} {{
          %v = affine.load %c[%i, %j] : {f32}
          %r = arith.maximumf %v, %zero : f32
          %rounded = arith.truncf %r : f32 to bf16
          affine.store %rounded, %h[%i, %j] : {bf16}
        }}
      }}
'''
            source += task(f'round{layer}', [previous], [hidden],
                           ['%c', '%h'], types, body)
            previous = f'%round{layer}_done'
    body = f'''      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to {batch} {{
        affine.for %j = 0 to {width} {{
          %v = affine.load %c[%i, %j] : {f32}
          %r = arith.maximumf %v, %zero : f32
          affine.store %r, %y[%i, %j] : {f32}
        }}
      }}
'''
    source += task('relu', [previous], ['%Y'], ['%c', '%y'], types, body)
    for value, element_type in reversed(allocations):
        source += f'    memref.dealloc {value} : {element_type}\n'
    return source + '    return\n  }\n}\n'


def bf16_rne_bits(values):
    """Finite FP32 -> BF16 bits, round to nearest with ties to even."""
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('reference requires finite FP32 values')
    bits = values.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def decode_bf16(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def make_inputs(batch=16, width=16):
    validate_shape(batch, width, 1)
    # Explicit PCG64 preserves the generator used by the original r12 inputs.
    rng = np.random.Generator(np.random.PCG64(SEED))
    x = bf16_rne_bits(rng.uniform(-.05, .05, (batch, width)).astype(np.float32))
    w = np.zeros((width, width), np.float32)
    for i in range(width):
        w[i, i] = .5
        w[i, (i + 1) % width] = .25
        w[i, (i + 3) % width] = .25
    return x, bf16_rne_bits(w)


def sequential_reference(x_bits, w_bits, layers=2):
    """Separate FP32 multiply/add per k; never a BLAS dot or fused multiply-add."""
    if layers < 1:
        raise ValueError('layers must be positive')
    a, w = decode_bf16(x_bits), decode_bf16(w_bits)
    if a.ndim != 2 or w.shape != (a.shape[1], a.shape[1]):
        raise ValueError('expected X[batch,width] and square W[width,width]')
    for layer in range(layers):
        c = np.zeros(a.shape, np.float32)
        for k in range(w.shape[0]):
            product = np.multiply(a[:, k, None], w[k, None, :], dtype=np.float32)
            c = np.add(c, product, dtype=np.float32)
        c = np.maximum(c, np.float32(0))
        if layer + 1 < layers:
            a = decode_bf16(bf16_rne_bits(c))
    return c


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def produce(output, batch=16, width=16, layers=2):
    validate_shape(batch, width, layers)
    source = make_source(batch, width, layers)
    x, w = make_inputs(batch, width)
    y = sequential_reference(x, w, layers)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source.mlir').write_text(source)
    np.savez(output / 'inputs.npz', X=x, W=w)
    np.savez(output / 'reference.npz', Y=y)
    arrays = {'X': x, 'W': w, 'Y': y}
    receipt = dict(
        schema='orbit.shared-weight-mlp.workload.v32', seed=SEED,
        rng='PCG64', numpy_version=np.__version__,
        batch=batch, width=width, layers=layers,
        source_contract='bf16-f32-accumulate-v1',
        inputs='signed uniform [-0.05,0.05] rounded to BF16 RNE',
        weights='runtime W = .5 I + .25 cyclic shift1 + .25 cyclic shift3',
        reference='sequential FP32 multiply then FP32 add for each k; ReLU; '
                  'integer BF16 RNE between layers; final FP32 Y',
        atol=ATOL, rtol=RTOL,
        hardware_executed=False,
        scope='source/numerical producer only; no backend feasibility, '
              'calibration coverage or speedup claim for generated parameters',
        producer=dict(path=Path(__file__).name, sha256=sha256(Path(__file__))),
        files={name: sha256(output / name)
               for name in ('source.mlir', 'inputs.npz', 'reference.npz')},
        arrays={name: dict(shape=list(a.shape), dtype=a.dtype.str,
                           sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
                for name, a in arrays.items()})
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=16,
                        help='rows, a positive multiple of 8 (default: 16)')
    parser.add_argument('--width', type=int, default=16,
                        help='square weight width, a positive multiple of 16')
    parser.add_argument('--layers', type=int, default=2,
                        help='positive layer count (default: 2); no hardware feasibility promise')
    parser.add_argument('--output', type=Path, required=True,
                        help='new output directory; existing data is never replaced')
    args = parser.parse_args()
    try:
        validate_shape(args.batch, args.width, args.layers)
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error('output directory must be new')
    receipt = produce(args.output, args.batch, args.width, args.layers)
    print(json.dumps(dict(output=str(args.output), files=receipt['files'])))


if __name__ == '__main__':
    main()
