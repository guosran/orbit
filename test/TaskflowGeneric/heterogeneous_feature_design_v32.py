#!/usr/bin/env python3
"""Static rank/range regression from captured native quantities; no device."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'feature_design_audit', ROOT / 'scripts/audit-heterogeneous-feature-design.py')
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


class DesignTest(unittest.TestCase):
    def test_individual_ranges_do_not_prove_joint_support(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/heterogeneous-feature-design-v32.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cases = []
            for case in fixture['cases']:
                path = base / (case['id'] + '.json')
                path.write_text(json.dumps(case['data']))
                cases.append(dict(id=case['id'], split=case['split'],
                    features=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
            spec = base / 'spec.json'
            spec.write_text(json.dumps(dict(feature_sets=fixture['feature_sets'], cases=cases)))
            result = AUDITOR.audit(spec)
            submit = next(row for row in result['designs']
                          if row['local_family'][0] == 'npu' and row['phase'] == 'submit')
            wait = next(row for row in result['designs']
                        if row['local_family'][0] == 'npu' and row['phase'] == 'wait')
            self.assertEqual(submit['rank'], 1)
            self.assertFalse(submit['coefficient_identifiable'])
            self.assertEqual(wait['rank'], 2)
            self.assertTrue(wait['heldout'][0]['within_individual_ranges'])
            self.assertFalse(wait['heldout'][0]['within_joint_convex_hull'])
            self.assertFalse(result['timing_labels_opened'])
            data = json.loads(Path(cases[0]['features']['path']).read_text())
            data['application']['call'] = 2
            Path(cases[0]['features']['path']).write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'SHA changed'):
                AUDITOR.audit(spec)


if __name__ == '__main__':
    unittest.main()
