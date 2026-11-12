#!/usr/bin/env python3
"""Produce explicit BF16-input/FP32-accumulation source, not a private planner.

The optional chain contains an explicit FP32->BF16 rounding task between GEMMs.
No hardware is accessed and no candidate costs are generated.
"""
import argparse
from pathlib import Path


def matrix_type(size, dtype):
    return f'memref<{size}x{size}x{dtype}>'


def task(name, reads, writes, arguments, types, body):
    def typed(values):
        return ', '.join(values) + ' : ' + ', '.join(types[v] for v in values)
    ordered = reads + writes
    arguments_list = ', '.join(f'{a}: {types[v]}' for a, v in zip(arguments, ordered))
    return (f'    %{name}_done = taskflow.task @{name} '
            f'will_reads({typed(reads)}) will_writes({typed(writes)}) '
            f'[original_read_memrefs({typed(reads)}), '
            f'original_write_memrefs({typed(writes)})] : '
            f'({", ".join(types[v] for v in ordered)}) -> {types[writes[0]]} {{\n'
            f'    ^bb0({arguments_list}):\n{body}'
            f'      taskflow.yield done_writes({arguments[-1]} : {types[writes[0]]})\n'
            '    }\n')


def produce(size=64, chain=False):
    bf16, f32 = matrix_type(size, 'bf16'), matrix_type(size, 'f32')
    types = {'%A': bf16, '%B': bf16, '%W': bf16, '%C': f32,
             '%C_done': f32, '%H': bf16, '%round_done': bf16,
             '%D': f32, '%second_done': f32, '%Y': f32}
    args = [f'%A: {bf16} {{orbit.role = "input", orbit.name = "A"}}',
            f'%B: {bf16} {{orbit.role = "input", orbit.name = "B"}}']
    if chain:
        args.append(f'%W: {bf16} {{orbit.role = "input", orbit.name = "W"}}')
    args.append(f'%Y: {f32} {{orbit.role = "output", orbit.name = "Y"}}')
    result = ('module {\n  func.func @matrix_program(' + ', '.join(args) +
              ') attributes {orbit.noalias, llvm.emit_c_interface, '
              'orbit.matrix_contract = "bf16-f32-accumulate-v1"} {\n'
              f'    %C = memref.alloc() {{orbit.name = "C"}} : {f32}\n')

    def gemm_body():
        return f'''      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to {size} {{
        affine.for %j = 0 to {size} {{
          affine.store %zero, %c[%i, %j] : {f32}
          affine.for %k = 0 to {size} {{
            %av = affine.load %a[%i, %k] : {bf16}
            %bv = affine.load %b[%k, %j] : {bf16}
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
    result += task('C', ['%A', '%B'], ['%C'], ['%a', '%b', '%c'], types, gemm_body())
    epilogue_input = '%C_done'
    if chain:
        result += (f'    %H = memref.alloc() {{orbit.name = "H"}} : {bf16}\n'
                   f'    %D = memref.alloc() {{orbit.name = "D"}} : {f32}\n')
        body = f'''      affine.for %i = 0 to {size} {{
        affine.for %j = 0 to {size} {{
          %v = affine.load %c[%i, %j] : {f32}
          %rounded = arith.truncf %v : f32 to bf16
          affine.store %rounded, %h[%i, %j] : {bf16}
        }}
      }}
'''
        result += task('round', ['%C_done'], ['%H'], ['%c', '%h'], types, body)
        result += task('second', ['%round_done', '%W'], ['%D'],
                       ['%a', '%b', '%c'], types, gemm_body())
        epilogue_input = '%second_done'
    body = f'''      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to {size} {{
        affine.for %j = 0 to {size} {{
          %v = affine.load %c[%i, %j] : {f32}
          %r = arith.maximumf %v, %zero : f32
          affine.store %r, %y[%i, %j] : {f32}
        }}
      }}
'''
    result += task('relu', [epilogue_input], ['%Y'], ['%c', '%y'], types, body)
    if chain:
        result += f'    memref.dealloc %D : {f32}\n    memref.dealloc %H : {bf16}\n'
    return result + f'    memref.dealloc %C : {f32}\n    return\n  }}\n}}\n'


def reference(a, b, weight=None):
    """FP64 mathematical reference over exact BF16 values, explicit BF16 bridge."""
    import numpy as np
    from ml_dtypes import bfloat16
    for value in (a, b, weight):
        if value is not None and (value.dtype != bfloat16 or
                                  not np.isfinite(value.astype(np.float32)).all() or
                                  np.max(np.abs(value.astype(np.float32))) > 1):
            raise ValueError('contract requires finite BF16 matrix inputs with abs<=1')
    c = a.astype(np.float64) @ b.astype(np.float64)
    if weight is not None:
        rounded = c.astype(np.float32).astype(bfloat16)
        if np.max(np.abs(rounded.astype(np.float32))) > 1:
            raise ValueError('intermediate BF16 input is outside the matrix contract')
        c = rounded.astype(np.float64) @ weight.astype(np.float64)
    return np.maximum(c, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=[16, 32, 64, 128], default=64)
    parser.add_argument('--chain', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(produce(args.size, args.chain))
