#!/usr/bin/env python3
"""Path-only proof copies pass; actual code hashes and ABI changes fail."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('identity_migration',
    Path(__file__).with_name('stabilize-r19-cpu-identities-v2.py'))
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


class DomainTests(unittest.TestCase):
    @staticmethod
    def row(operators):
        return dict(kind='prefill-cpu-region', operators=operators, backend='inductor-cpu',
                    input_policy='host', output_policy='host')

    def test_arbitrary_growth_and_one_to_one_path_change(self):
        old = {'old-a': self.row(['a']), 'stable-b': self.row(['b'])}
        fresh = {'fresh-a': self.row(['a']), 'stable-b': self.row(['b']),
                 'new-c': self.row(['c'])}
        old_keys, new_keys, changes = m.plan_cpu_identity_changes(old, fresh)
        self.assertEqual((len(old_keys), len(new_keys)), (2, 3))
        self.assertEqual(changes, [('old-a', 'fresh-a')])

    def test_zero_changes_and_growth_are_valid(self):
        old = {'a': self.row(['a'])}
        fresh = {**old, 'b': self.row(['b'])}
        self.assertEqual(m.plan_cpu_identity_changes(old, fresh)[2], [])
        self.assertEqual(m.plan_cpu_identity_changes(old, old)[2], [])

    def test_lost_parent_semantics_rejected(self):
        with self.assertRaises(ValueError):
            m.plan_cpu_identity_changes({'a': self.row(['a'])}, {'b': self.row(['b'])})

    def test_ambiguous_semantics_rejected(self):
        with self.assertRaises(ValueError):
            m.plan_cpu_identity_changes({'a': self.row(['a'])},
                {'a': self.row(['a']), 'duplicate': self.row(['a'])})

    def test_duplicate_implementation_id_rejected(self):
        with self.assertRaises(ValueError):
            m.unique_implementations([dict(id='same'), dict(id='same')])


if __name__ == '__main__': unittest.main()
