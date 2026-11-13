#!/usr/bin/env python3
"""Light source-cover checks; no device, captures, or full graph import."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "split_campaign", Path(__file__).with_name("split-r19-ffn-campaign.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SplitTests(unittest.TestCase):
    def setUp(self):
        operators = [dict(id=f"prefill.L{layer}.{role}", source_ordinal=layer * 4 + i,
                          attributes={"supported_devices": ["npu"]})
                     for layer in range(36)
                     for i, role in enumerate(("gate_up", "gate_split", "swiglu", "down"))]
        self.source = {"schema": "orbit.program-operator-source.v1", "operators": operators}
        self.source["source_id"] = module.node.digest(self.source)
        self.plan = module.node.make_plan(self.source, {op["id"]: "npu" for op in operators})
        self.plan["r18"] = dict(fused_layers=list(range(12, 24)), gemm_variants={},
                                retention="reload", split_order="up-gate", ffn_artifact={"sha": "artifact"})
        registry = {"original-" + op["id"]: dict(kind="prefill-original", operator=op["id"], device="npu")
                    for op in operators}
        registry.update({f"fused-{layer}": dict(kind="prefill-fused-ffn", layer=layer,
                                              retention="reload", split_order="up-gate",
                                              ffn_artifact={"sha": "artifact"}) for layer in range(36)})
        self.catalogue = {"runtime_registration": {"implementations": registry}}
        self.ready = {"operators": {}, "cpu_regions": [], "decode_rope_prefetch": True,
                      "whole_source": {"source_id": "whole"}}

    def test_two_six_layer_runs_cover_parent_exactly(self):
        pieces = module.split_layers(self.plan)
        self.assertEqual([len(piece) for piece in pieces], [6, 6])
        self.assertEqual(sum(pieces, []), self.plan["r18"]["fused_layers"])
        for piece in pieces:
            plan = copy.deepcopy(self.plan)
            plan["r18"]["fused_layers"] = piece
            ready = module.derive_ready(self.source, plan, self.ready, self.catalogue)
            self.assertEqual(len(ready["ready_priority"]), 144 - 3 * 6)
            self.assertEqual(len([key for key in ready["ready_priority"] if key.startswith("fused-")]), 6)
            for layer in range(36):
                name = f"prefill.L{layer}.gate_up"
                expected = f"fused-{layer}" if layer in piece else "original-" + name
                self.assertEqual(ready["implementation_ids"][name], expected)
            self.assertEqual(ready["whole_source"], self.ready["whole_source"])
            self.assertTrue(ready["decode_rope_prefetch"])

    def test_wrong_layer_count_and_duplicates_rejected(self):
        for layers in ([0] * 12, list(range(11)), list(range(25, 37))):
            with self.assertRaises(ValueError):
                module.split_layers({"r18": {"fused_layers": layers}})

    def test_missing_or_ambiguous_actual_registration_rejected(self):
        for duplicate in (False, True):
            catalogue = copy.deepcopy(self.catalogue)
            registry = catalogue["runtime_registration"]["implementations"]
            row = registry.pop("fused-12")
            if duplicate:
                registry.update({"dup1": row, "dup2": row})
            with self.assertRaisesRegex(ValueError, "resolve uniquely"):
                module.derive_ready(self.source, self.plan, self.ready, catalogue)

    def test_hidden_cpu_or_half_override_rejected(self):
        for key in ("operators", "cpu_regions"):
            ready = dict(self.ready, **{key: ["unexpected"]})
            with self.assertRaisesRegex(ValueError, "overrides"):
                module.derive_ready(self.source, self.plan, ready, self.catalogue)


if __name__ == "__main__":
    unittest.main()
