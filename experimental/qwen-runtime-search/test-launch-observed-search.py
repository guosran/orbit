#!/usr/bin/env python3
"""Small executable fixtures for uncapped observer evidence and error handling."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LAUNCHER = Path(__file__).with_name("launch-observed-search.py")


class ObserverTests(unittest.TestCase):
    def execute(self, returncode=0, complete=True, warm=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        binary = root / "fake-search.py"
        binary.write_text("#!/usr/bin/env python3\nimport pathlib,sys\np=pathlib.Path(sys.argv[3])\n"
                          "p.write_text('{}\\n')\n" +
                          ("for suffix in ['.checkpoint.jsonl','.beam-trace.jsonl','.joint-actions.jsonl',"
                           "'.best-checkpoint.json','.certificate.json']:\n"
                           " pathlib.Path(str(p)+suffix).write_text('{}\\n')\n" if complete else "") +
                          f"raise SystemExit({returncode})\n")
        binary.chmod(0o755)
        for name in ("source.json", "catalogue.json"):
            (root / name).write_text("{}\n")
        prior = root / "prior.json"
        if warm:
            prior.write_text('{"fixture": "actual-prior-input-bytes"}\n')
        result = subprocess.run([sys.executable, str(LAUNCHER), "--binary", str(binary),
                                 "--source", str(root / "source.json"),
                                 "--catalogue", str(root / "catalogue.json"),
                                 "--output", str(root / "result.json"),
                                 "--evidence-dir", str(root / "evidence"),
                                 *(["--warm-start", str(prior)] if warm else [])],
                                capture_output=True, text=True)
        return result, root

    def test_success_seals_complete_outputs_and_exact_child_peak(self):
        result, root = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        seal = json.loads((root / "evidence/seal.json").read_text())
        memory = json.loads((root / "evidence/memory-summary.json").read_text())
        launch = json.loads((root / "evidence/launch.json").read_text())
        self.assertTrue(seal["inputs_unchanged"])
        self.assertEqual(seal["missing_expected_outputs"], [])
        self.assertGreater(memory["wait4_peak_rss_bytes"], 0)
        self.assertIsNone(launch["wall_timeout_seconds"])
        self.assertIsNone(launch["evaluation_limit"])
        self.assertIsNone(launch["round_limit"])
        self.assertEqual(launch["argv"][-1], "4")

    def test_warm_prior_is_forwarded_and_bound_by_actual_hash(self):
        result, root = self.execute(warm=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        launch = json.loads((root / "evidence/launch.json").read_text())
        self.assertEqual(launch["argv"][-2:], ["--warm-start", str(root / "prior.json")])
        self.assertEqual(launch["warm_start_prior_result"], str(root / "prior.json"))
        self.assertTrue(any(row["requested_file"] == str(root / "prior.json")
                            for row in launch["inputs"]))

    def test_failure_and_missing_trace_are_not_reported_as_success(self):
        result, root = self.execute(returncode=3, complete=False)
        self.assertEqual(result.returncode, 3, result.stderr)
        seal = json.loads((root / "evidence/seal.json").read_text())
        self.assertEqual(seal["returncode"], 3)
        self.assertTrue(seal["missing_expected_outputs"])
        result, _ = self.execute(returncode=0, complete=False)
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
