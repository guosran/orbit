#!/usr/bin/env python3
"""Path-only proof copies pass; actual code hashes and ABI changes fail."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('identity_migration',
    Path(__file__).with_name('stabilize-r19-cpu-identities.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ProofTests(unittest.TestCase):
    def setUp(self):
        self.proof = dict(specialization_id='actual', generated_code=[dict(file='/old/kernel.py', sha256='abc')],
                          region=dict(input_specs={'x': {'shape': [5, 12], 'dtype': 'f32'}}),
                          generated_kernel_audit=dict(managed_heap_allocation_bytes=240))

    def test_path_only_copy(self):
        other = copy.deepcopy(self.proof); other['generated_code'][0]['file'] = '/new/kernel.py'
        m.equivalent_proofs(self.proof, other)

    def test_changed_generated_code_rejected(self):
        other = copy.deepcopy(self.proof); other['generated_code'][0]['sha256'] = 'def'
        with self.assertRaises(ValueError): m.equivalent_proofs(self.proof, other)

    def test_changed_input_abi_rejected(self):
        other = copy.deepcopy(self.proof); other['region']['input_specs']['x']['shape'] = [6, 12]
        with self.assertRaises(ValueError): m.equivalent_proofs(self.proof, other)

    def test_changed_heap_bound_rejected(self):
        other = copy.deepcopy(self.proof); other['generated_kernel_audit']['managed_heap_allocation_bytes'] += 1
        with self.assertRaises(ValueError): m.equivalent_proofs(self.proof, other)


if __name__ == '__main__': unittest.main()
