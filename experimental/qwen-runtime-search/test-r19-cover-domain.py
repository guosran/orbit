#!/usr/bin/env python3
"""Exact-cover regression cases for fusion, fission and coupled contexts."""

import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "cover_domain", Path(__file__).with_name("build-r19-cover-domain.py"))
domain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(domain)


def implementation(name, members):
    return {"id": name, "source_operators": members}


class CoverDomainTests(unittest.TestCase):
    def test_all_legal_fusion_partitions_and_mapping_combinations(self):
        rows = [implementation("a-cpu", [0]), implementation("a-npu", [0]),
                implementation("b", [1]), implementation("c", [2]),
                implementation("ab", [0, 1]), implementation("bc", [1, 2]),
                implementation("abc", [0, 1, 2])]
        groups = domain.exact_cover_groups(3, rows)
        actual = {tuple(choice["implementations"])
                  for choice in groups[0]["choices"]}
        self.assertEqual(actual, {("a-cpu", "b", "c"), ("a-npu", "b", "c"),
                                  ("ab", "c"), ("a-cpu", "bc"),
                                  ("a-npu", "bc"), ("abc",)})

    def test_noncontiguous_region_preserves_exact_coverage(self):
        rows = [implementation("a", [0]), implementation("b", [1]),
                implementation("c", [2]), implementation("ac", [0, 2])]
        groups = domain.exact_cover_groups(3, rows)
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(len(group["choices"]) for group in groups), [1, 2])
        for combination in groups[0]["choices"]:
            indices = [i for name in combination["implementations"]
                       for row in rows if row["id"] == name for i in row["source_operators"]]
            self.assertEqual(sorted(indices), [0, 2])

    def test_decoder_context_cannot_mix_between_steps(self):
        rows = [implementation(name + str(step), [step])
                for name in ("contextA", "contextB") for step in range(3)]
        groups = domain.exact_cover_groups(3, rows, [
            ["contextA0", "contextA1", "contextA2"],
            ["contextB0", "contextB1", "contextB2"]])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["choices"]), 2)
        for choice in groups[0]["choices"]:
            self.assertEqual(len({name[:-1] for name in choice["implementations"]}), 1)

    def test_new_partial_decoder_registration_cannot_escape_old_bundle(self):
        rows = [implementation("a0", [0]), implementation("a1", [1]),
                implementation("new0", [0])]
        with self.assertRaisesRegex(ValueError, "protected context"):
            domain.exact_cover_groups(2, rows, [["a0", "a1"]], [{0, 1}])
        with self.assertRaisesRegex(ValueError, "protected context"):
            domain.exact_cover_groups(2, rows,
                                     [["a0", "a1"], ["new0"]], [{0, 1}])

    def test_explicit_complete_new_context_is_legal(self):
        rows = [implementation(name + str(i), [i])
                for name in ("a", "b") for i in range(2)]
        groups = domain.exact_cover_groups(
            2, rows, [["a0", "a1"], ["b0", "b1"]], [{0, 1}])
        self.assertEqual(len(groups[0]["choices"]), 2)

    def test_invalid_compound_overlap_and_uncovered_source_rejected(self):
        rows = [implementation("ab", [0, 1]), implementation("b", [1])]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            domain.exact_cover_groups(2, rows, [["ab", "b"]])
        with self.assertRaisesRegex(ValueError, "coverage holes"):
            domain.exact_cover_groups(3, rows)

    def test_overlap_component_without_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no executable exact cover"):
            domain.exact_cover_groups(3, [implementation("ab", [0, 1]),
                                          implementation("bc", [1, 2])])

    def test_duplicate_source_instances_and_implementation_ids_rejected(self):
        for rows in ([implementation("a", [0, 0])],
                     [implementation("a", [0]), implementation("a", [0])]):
            with self.assertRaises(ValueError):
                domain.exact_cover_groups(1, rows)

    def test_large_fission_does_not_depend_on_python_recursion_limit(self):
        count = 1100
        rows = [implementation("part-" + str(i), [i]) for i in range(count)]
        rows.append(implementation("whole", list(range(count))))
        groups = domain.exact_cover_groups(count, rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(len(choice["implementations"])
                                for choice in groups[0]["choices"]), [1, count])


if __name__ == "__main__":
    unittest.main()
