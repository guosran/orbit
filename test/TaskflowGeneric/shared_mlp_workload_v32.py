#!/usr/bin/env python3
"""CPU contracts and optional exact r12 replay for the standalone MLP producer."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'scripts/make-shared-mlp-workload.py'
SPEC = importlib.util.spec_from_file_location('shared_mlp', SCRIPT)
PRODUCER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRODUCER)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def decode(word):
    return struct.unpack('<f', struct.pack('<I', int(word) << 16))[0]


def scalar_rne(value):
    word = struct.unpack('<I', struct.pack('<f', value))[0]
    upper, remainder = word >> 16, word & 0xffff
    # Independent compare/tie implementation, rather than the producer bias.
    if remainder > 0x8000 or (remainder == 0x8000 and upper & 1):
        upper += 1
    return upper


def scalar_reference(x, w, layers):
    a = [[decode(v) for v in row] for row in x]
    weight = [[decode(v) for v in row] for row in w]
    for layer in range(layers):
        c = []
        for row in a:
            result = []
            for j in range(len(weight)):
                total = 0.0
                for k in range(len(weight)):
                    total = f32(total + f32(row[k] * weight[k][j]))
                result.append(max(total, 0.0))
            c.append(result)
        if layer + 1 < layers:
            a = [[decode(scalar_rne(v)) for v in row] for row in c]
    return np.array(c, dtype=np.float32)


class ProducerContracts(unittest.TestCase):
    def test_fp32_and_rne_against_scalar_oracle(self):
        for width, layers in [(16, 1), (16, 2), (16, 3), (32, 2)]:
            x, w = PRODUCER.make_inputs(8, width)
            actual = PRODUCER.sequential_reference(x, w, layers)
            self.assertTrue(np.array_equal(actual, scalar_reference(x, w, layers)))
        words = np.array([0x3f808000, 0x3f818000, 0xbf808000,
                          0xbf818000, 0x00008000, 0x00008001], np.uint32)
        self.assertEqual(PRODUCER.bf16_rne_bits(words.view(np.float32)).tolist(),
                         [0x3f80, 0x3f82, 0xbf80, 0xbf82, 0, 1])

    def test_source_and_runtime_weight_contract(self):
        for layers in (1, 2, 3):
            source = PRODUCER.make_source(24, 16, layers)
            self.assertEqual(source.count('taskflow.task @'), 2 * layers)
            self.assertEqual(source.count('arith.truncf'), layers - 1)
            self.assertEqual(source.count('affine.store %zero, %c'), layers)
            self.assertIn('%W: memref<16x16xbf16>', source)
            self.assertNotIn('orbit.mapping', source)
            self.assertNotIn('orbit.air_tile_sizes', source)
        x, w = PRODUCER.make_inputs(24, 16)
        self.assertEqual(x.dtype, np.uint16)
        decoded = PRODUCER.decode_bf16(w)
        self.assertTrue(np.array_equal(decoded.sum(axis=0), np.ones(16)))
        self.assertTrue(np.array_equal(decoded.sum(axis=1), np.ones(16)))
        self.assertTrue(np.array_equal(x[:16], PRODUCER.make_inputs(16, 16)[0]))

    def test_receipt_hashes_and_fresh_output(self):
        output = Path(tempfile.mkdtemp(prefix='orbit-shared-mlp-')) / 'workload'
        receipt = PRODUCER.produce(output, 8, 16, 2)
        self.assertEqual((receipt['atol'], receipt['rtol']), (1e-4, 1e-4))
        for name, expected in receipt['files'].items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
        with np.load(output / 'inputs.npz', allow_pickle=False) as inputs:
            self.assertEqual(inputs['X'].shape, (8, 16))
        with np.load(output / 'reference.npz', allow_pickle=False) as reference:
            self.assertEqual(reference['Y'].dtype, np.float32)
        with self.assertRaises(FileExistsError):
            PRODUCER.produce(output)

    def test_cli_rejects_invalid_shapes(self):
        output = Path(tempfile.mkdtemp(prefix='orbit-shared-mlp-invalid-')) / 'new'
        for options in [('--batch', '7'), ('--batch', '0'),
                        ('--width', '8'), ('--width', '-16'), ('--layers', '0')]:
            result = subprocess.run([sys.executable, str(SCRIPT), *options,
                                     '--output', str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


def verify_r12(root):
    results = []
    output = Path(tempfile.mkdtemp(prefix='orbit-shared-mlp-r12-replay-'))
    for batch in (16, 24, 32):
        generated = output / f'b{batch}'
        PRODUCER.produce(generated, batch, 16, 2)
        source = root / f'source{batch}.mlir'
        assert (generated / 'source.mlir').read_bytes() == source.read_bytes(), batch
        numerical = root / ('numerical' if batch == 16 else f'rect-numerical/b{batch}')
        for new_name, old_name, keys in [('inputs.npz', 'inputs.npz', ['X', 'W']),
                                        ('reference.npz', 'reference-2.npz', ['Y'])]:
            with np.load(generated / new_name, allow_pickle=False) as actual, \
                    np.load(numerical / old_name, allow_pickle=False) as frozen:
                for key in keys:
                    assert actual[key].dtype == frozen[key].dtype
                    assert np.array_equal(actual[key], frozen[key]), (batch, key)
        results.append(dict(batch=batch, source_bytes_exact=True, X_W_Y_exact=True))
    receipt = dict(status='PASS', hardware_executed=False, r12_root=str(root),
                   generated_root=str(output), cases=results)
    (output / 'audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r12-root', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ProducerContracts))
    if not result.wasSuccessful():
        return 1
    if args.r12_root:
        verify_r12(args.r12_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
