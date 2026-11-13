#!/usr/bin/env python3
"""Small file-only regressions for nested sealed campaign continuation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "resume", Path(__file__).with_name("resume-r19-campaign.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prep = self.root / "new-prep.py"
        self.prep.write_text("# preparation only\n")
        self.rows = [{"name": "case" + str(i), "purpose": "training",
                      "command": ["python", "/frozen/run-qwen-r19-ready.py",
                                  "--output", str(self.root / ("old" + str(i)))],
                      "admission_command": ["gate", str(self.root / ("old" + str(i)))],
                      "numeric_gate_file": str(self.root / ("old" + str(i)) / "gate.json")}
                     for i in range(4)]
        self.base = {"schema": module.SCHEMA, "source_id": "source",
                     "whole_source": "/source.json", "dependencies": [],
                     "cohorts": self.rows}
        self.original = self.save("original.json", self.base)
        self.old_receipts = self.save("old-receipts.json", [
            self.receipt(self.rows[0], self.original),
            self.receipt(self.rows[1], self.original, False)])
        self.admission = copy.deepcopy(self.base)
        self.admission["preparation_continuation"] = {
            "parent": module.record(self.original),
            "parent_receipts": module.record(self.old_receipts)}
        self.admission["rejected_cohorts"] = [{"cohort": "case1", "included_in_prices": False}]
        for i, row in enumerate(self.admission["cohorts"]):
            row["preparation_protocol"] = "original" if i == 0 else "prepared-v1"
            if i:
                row["command"] = ["python", "/old/prep.py", "--runtime-runner",
                                  *row["command"][1:]]
        self.admission_path = self.save("admission.json", self.admission)
        execution = dict(self.admission, cohorts=self.admission["cohorts"][1:])
        self.execution_path = self.save("execution.json", execution)
        self.receipts_path = self.save("receipts.json", [
            self.receipt(execution["cohorts"][0], self.execution_path),
            self.receipt(execution["cohorts"][1], self.execution_path, False)])

    def save(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def receipt(self, row, manifest, success=True):
        return {"name": row["name"], "purpose": row["purpose"],
                "command": row["command"], "manifest_sha256": module.record(manifest)["sha256"],
                "returncode": 0 if success else 1,
                "admission_returncode": 0 if success else None,
                "sealed_dependencies_unchanged": success}

    def run_resume(self, output="new"):
        return module.resume(self.admission_path, self.execution_path,
                             self.receipts_path, self.prep, "prepared-v2", self.root / output)

    def test_nested_continuation_preserves_protocols_history_and_all_rows(self):
        result = self.run_resume()
        self.assertEqual((result["retained_measured_cohorts"], result["pending_cohorts"]), (2, 2))
        combined = module.read(self.root / "new/campaign-admission.json")
        pending = module.read(self.root / "new/campaign-execution.json")
        self.assertEqual([r["name"] for r in combined["cohorts"]], [r["name"] for r in self.rows])
        self.assertEqual([r["preparation_protocol"] for r in combined["cohorts"]],
                         ["original", "prepared-v1", "prepared-v2", "prepared-v2"])
        self.assertEqual(len(combined["rejected_cohorts"]), 2)
        self.assertEqual(len(combined["retained_receipt_backing"]), 2)
        receipts = self.save("third-receipts.json", [
            self.receipt(pending["cohorts"][0], self.root / "new/campaign-execution.json")])
        module.resume(self.root / "new/campaign-admission.json",
                      self.root / "new/campaign-execution.json", receipts,
                      self.prep, "prepared-v3", self.root / "next")
        final = module.read(self.root / "next/campaign-admission.json")
        self.assertEqual(final["resume_provenance"]["retained_measured_cohorts"], 3)
        self.assertEqual(len(final["rejected_cohorts"]), 2)
        self.assertEqual(final["cohorts"][0], combined["cohorts"][0])

    def test_tampered_receipt_binding_rejected(self):
        for field, value in (("manifest_sha256", "bad"), ("command", ["changed"]),
                             ("purpose", "heldout"), ("name", "unknown")):
            with self.subTest(field=field):
                receipts = module.read(self.receipts_path)
                receipts[0][field] = value
                self.receipts_path.write_text(json.dumps(receipts))
                with self.assertRaises(ValueError):
                    self.run_resume()
                self.receipts_path.write_text(json.dumps([
                    self.receipt(self.admission["cohorts"][1], self.execution_path),
                    self.receipt(self.admission["cohorts"][2], self.execution_path, False)]))

    def test_omitted_unmeasured_cohort_not_admitted(self):
        admission = module.read(self.admission_path)
        admission["cohorts"].append(dict(self.rows[0], name="unmeasured"))
        self.admission_path.write_text(json.dumps(admission))
        with self.assertRaisesRegex(ValueError, "receipt backing"):
            self.run_resume()

    def test_changed_ancestor_symlink_rejected(self):
        other = self.save("other.json", {})
        link = self.root / "alias.json"
        link.symlink_to(self.original)
        admission = module.read(self.admission_path)
        admission["preparation_continuation"]["parent"] = module.record(link)
        self.admission_path.write_text(json.dumps(admission))
        link.unlink()
        link.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "Changed ancestor"):
            self.run_resume()

    def test_existing_output_never_overwritten(self):
        self.run_resume()
        with self.assertRaises(FileExistsError):
            self.run_resume()


if __name__ == "__main__":
    unittest.main()
