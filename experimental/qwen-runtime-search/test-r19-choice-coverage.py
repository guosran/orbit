#!/usr/bin/env python3
"""Small trace witnesses distinguish local exposure from product exhaustion."""
import importlib.util
import io
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "choice_coverage", Path(__file__).with_name("audit-r19-choice-coverage.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = {"source_id": "source", "execution_registration_identity": "reg",
                          "execution_policy": "source-dependency-ready",
                          "ready_priority_policies": ["source-order", "npu-first"],
                          "implementations": [{"id": str(i), "device": "cpu" if i < 2 else "npu",
                                               "source_operators": [0 if i < 2 else 1],
                                               "execution_variant": {"replicas": 1}}
                                              for i in range(3)],
                          "choice_groups": [{"id": "variable", "choices": [
                              {"id": "a", "implementations": ["0"]},
                              {"id": "b", "implementations": ["1"]}]},
                              {"id": "fixed", "choices": [{"id": "only", "implementations": ["2"]}]}]}

    def trace(self, assignments):
        return io.StringIO("".join(json.dumps({"evaluation": i + 1, "source_id": "source",
                      "assignment": values, "priority_key_identity": "named-ready-priority-seed",
                      "feasible": True, "cost_ms": 1}) + "\n" for i, values in enumerate(assignments)))

    def test_all_local_choices_need_not_exhaust_cartesian_product(self):
        result = module.audit(self.catalogue, self.trace([[0, 0], [1, 1], [0, 1]]))
        self.assertTrue(result["all_registered_local_choices_evaluated"])
        self.assertEqual(result["trace_rows_scanned"], 2)
        self.assertFalse(result["global_cartesian_product_exhausted"])
        self.assertEqual(result["axis_counts"]["placement"]["evaluated_distinct_values"], 2)
        self.assertEqual(result["groups"][1]["evaluated_choices"], 1)

    def test_incomplete_trace_has_explicit_missing_choices(self):
        result = module.audit(self.catalogue, self.trace([[0, 0]]))
        self.assertFalse(result["all_registered_local_choices_evaluated"])
        self.assertIsNone(result["groups"][0]["choice_witnesses"]["b"])

    def test_bad_source_or_index_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            module.audit(self.catalogue, self.trace([[2, 0]]))
        self.catalogue["source_id"] = "different"
        with self.assertRaisesRegex(ValueError, "source differs"):
            module.audit(self.catalogue, self.trace([[0, 0]]))


if __name__ == "__main__":
    unittest.main()
