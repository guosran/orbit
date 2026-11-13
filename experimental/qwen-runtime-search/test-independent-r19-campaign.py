#!/usr/bin/env python3
"""Tiny actual subprocess tests of independent admission and immutable seals."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "run-independent-r19-campaign.py"
spec = importlib.util.spec_from_file_location("independent_resume", HERE / "resume-independent-r19-campaign.py")
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class IndependentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dependency = self.root / "sealed.txt"
        self.dependency.write_text("original")

    def cohort(self, name, code=0, gate=0, script=None):
        return {"name": name, "purpose": "training",
                "command": [sys.executable, "-c", script or f"raise SystemExit({code})"],
                "admission_command": [sys.executable, "-c", f"raise SystemExit({gate})"]}

    def execute(self, rows):
        manifest = {"schema": resume.SCHEMA, "cohorts": rows,
                    "dependencies": [resume.record(self.dependency), resume.record(DRIVER)],
                    "independent_execution_policy": {
                        "failure_policy": "continue-hardware-or-gate-failure-stop-on-seal-drift",
                        "driver": resume.record(DRIVER), "execution_order": [row["name"] for row in rows]}}
        path = self.root / "campaign.json"
        path.write_text(json.dumps(manifest))
        output = self.root / "receipts"
        result = subprocess.run([sys.executable, str(DRIVER), "--manifest", str(path),
                                 "--manifest-sha256", resume.record(path)["sha256"],
                                 "--output", str(output)], capture_output=True, text=True)
        return result, json.loads((output / "summary.json").read_text()), manifest, path

    def test_multiple_failures_then_success_all_retained_and_nonzero(self):
        result, receipts, manifest, path = self.execute([
            self.cohort("hardware-fail", code=2), self.cohort("gate-fail", gate=3),
            self.cohort("success")])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual([row["name"] for row in receipts], [row["name"] for row in manifest["cohorts"]])
        self.assertEqual([row["included_in_prices"] for row in receipts], [False, False, True])
        self.assertTrue(all(row["sealed_dependencies_unchanged"] for row in receipts))
        resume.verify_receipts(manifest, resume.record(path), receipts)

    def test_seal_drift_stops_before_gate_or_next_independent_cohort(self):
        script = "from pathlib import Path; Path(" + repr(str(self.dependency)) + ").write_text('changed')"
        result, receipts, _, _ = self.execute([
            self.cohort("drift", script=script), self.cohort("never-start")])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(receipts), 1)
        self.assertFalse(receipts[0]["sealed_dependencies_unchanged"])
        self.assertNotIn("admission_returncode", receipts[0])
        self.assertFalse((self.root / "receipts/never-start.started.json").exists())
        status = json.loads((self.root / "receipts/campaign-status.json").read_text())
        self.assertTrue(status["stopped_on_seal_drift"])

    def test_all_success_returns_zero(self):
        result, receipts, _, _ = self.execute([self.cohort("ok1"), self.cohort("ok2")])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(row["included_in_prices"] for row in receipts))

    def test_resume_rejects_receipts_after_drift_but_accepts_multiple_rejections(self):
        rows = [self.cohort("fail"), self.cohort("later")]
        manifest = {"schema": resume.SCHEMA, "cohorts": rows}
        receipts = [{"name": row["name"], "purpose": row["purpose"], "command": row["command"],
                     "manifest_sha256": "hash", "returncode": 1,
                     "sealed_dependencies_unchanged": True} for row in rows]
        resume.verify_receipts(manifest, {"sha256": "hash"}, receipts)
        receipts[0]["sealed_dependencies_unchanged"] = False
        with self.assertRaisesRegex(ValueError, "seal drift"):
            resume.verify_receipts(manifest, {"sha256": "hash"}, receipts)


if __name__ == "__main__":
    unittest.main()
