#!/usr/bin/env python3
"""Regression checks that stale/mixed evidence cannot count as shared success.

Uses a small temporary copy of already measured artifacts; never runs hardware
or alters the original evidence.
"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from taskflow_experiment import ROOT, report


def main():
    source = ROOT / "build/backend-decoupling"
    manifest = json.loads((source / "manifest.json").read_text())
    original = next(c for c in manifest["cases"] if c["case"] == "gemm" and c["candidate"] == "base")
    with tempfile.TemporaryDirectory(prefix="orbit-feedback-test-") as tmp:
        root = Path(tmp)
        dest = root / "gemm/base"
        dest.mkdir(parents=True)
        record = dict(original, directory=str(dest))
        (root / "manifest.json").write_text(json.dumps(dict(cases=[record])))
        original_dir = Path(original["directory"])
        for name in ("plan.json", "candidate.mlir"):
            shutil.copy2(original_dir / name, dest / name)
        for sub in ["amoeba"] + [f"npu-seed-{s}" for s in original["seeds"]]:
            (dest / sub).mkdir()
            for name in (("feedback.json",) if sub == "amoeba" else
                         ("result.json", "verified-plan.json", "verified-taskflow.mlir")):
                shutil.copy2(original_dir / sub / name, dest / sub / name)
        args = SimpleNamespace(output=root)

        def check(expected, label):
            with contextlib.redirect_stdout(io.StringIO()):
                actual = report(args)
            assert actual is expected, label
            print(label + ": passed")

        check(True, "authentic shared evidence")
        mlir = dest / "candidate.mlir"
        original_mlir = mlir.read_bytes()
        mlir.write_bytes(original_mlir + b"\n// changed after measurement\n")
        check(False, "stale MLIR cannot count as shared")
        mlir.write_bytes(original_mlir)
        result_file = dest / "npu-seed-1/result.json"
        data = json.loads(result_file.read_text())
        data["implementation_sha256"] = "different-implementation"
        result_file.write_text(json.dumps(data))
        check(False, "different implementations cannot be pooled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
