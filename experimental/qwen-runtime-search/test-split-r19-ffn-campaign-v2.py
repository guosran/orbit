#!/usr/bin/env python3
"""Generic bounded chunking preserves source coverage and admitted identities."""
import copy
import importlib.util
from pathlib import Path
import unittest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = load("generic_split", "split-r19-ffn-campaign-v2.py")
fixtures = load("split_fixture", "test-split-r19-ffn-campaign.py")


class GenericSplitTests(unittest.TestCase):
    def test_arbitrary_chunk_counts_and_exact_order(self):
        for layers, maximum, expected in ((list(range(6)), 3, [3, 3]),
                                           (list(range(3)), 2, [2, 1]),
                                           ([8, 3], 1, [1, 1]),
                                           (list(range(12)), 5, [5, 5, 2])):
            with self.subTest(layers=layers, maximum=maximum):
                pieces = module.split_layers({"r18": {"fused_layers": layers}}, maximum)
                self.assertEqual(list(map(len, pieces)), expected)
                self.assertEqual(sum(pieces, []), layers)
                self.assertEqual(len(set(sum(pieces, []))), len(layers))

    def test_singleton_noop_invalid_sizes_and_duplicates_rejected(self):
        for layers, maximum in (([0], 1), ([0], 3), ([0, 1], 2), ([0, 1], 0),
                                ([0, 1], -1), ([0, 0], 1), ([0, 36], 1)):
            with self.assertRaises(ValueError):
                module.split_layers({"r18": {"fused_layers": layers}}, maximum)

    def test_every_chunk_resolves_real_source_coverage_and_original_fallback(self):
        fixture = fixtures.SplitTests()
        fixture.setUp()
        fixture.plan["r18"]["fused_layers"] = list(range(18, 24))
        selected = []
        for layers in module.split_layers(fixture.plan, 3):
            plan = copy.deepcopy(fixture.plan)
            plan["r18"]["fused_layers"] = layers
            ready = module.derive_ready(fixture.source, plan, fixture.ready, fixture.catalogue)
            self.assertEqual(len(ready["ready_priority"]), 144 - 3 * 3)
            for layer in range(36):
                operator = f"prefill.L{layer}.gate_up"
                expected = f"fused-{layer}" if layer in layers else "original-" + operator
                self.assertEqual(ready["implementation_ids"][operator], expected)
            selected.extend(layers)
        self.assertEqual(selected, list(range(18, 24)))

    def test_unaffected_execution_first_without_any_drop_or_duplicate(self):
        rows = [{"name": name} for name in ("chunk0", "cpu", "chunk1", "decode", "chunk2")]
        actual = module.order_pending(rows, {"chunk0", "chunk1", "chunk2"})
        self.assertEqual([row["name"] for row in actual], ["cpu", "decode", "chunk0", "chunk1", "chunk2"])
        self.assertEqual(sorted(row["name"] for row in actual), sorted(row["name"] for row in rows))


if __name__ == "__main__":
    unittest.main()
