#!/usr/bin/env python3
"""Execute a sealed sequence of full-model training or heldout commands.

Commands are argument arrays, never shell snippets. Every dependency is checked
before and after each process, and failed executions retain their logs and
receipts. The actual model runner owns the NPU lock and unchanged memory guards.
No search or execution deadline is introduced by this driver.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(records):
    for row in records:
        requested = Path(row.get("requested_file", row["file"]))
        if (requested.resolve(strict=True) != Path(row["file"]).resolve(strict=True)
                or checksum(requested) != row["sha256"]):
            raise ValueError("Sealed dependency changed: " + row["file"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if checksum(args.manifest) != args.manifest_sha256:
        raise ValueError("Campaign manifest identity mismatch")
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "orbit.qwen-r19-sealed-campaign.v1":
        raise ValueError("Unknown campaign schema")
    dependencies = manifest["dependencies"]
    cohorts = manifest["cohorts"]
    names = [row["name"] for row in cohorts]
    if (not dependencies or not cohorts or len(names) != len(set(names))
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names)):
        raise ValueError("Unique safe cohort names and sealed dependencies required")
    for row in cohorts:
        if (row["purpose"] not in ("training", "heldout", "validation-only")
                or not row["command"] or any(type(value) is not str
                                            for value in row["command"])):
            raise ValueError("Explicit cohort purpose and command array required")
    verify(dependencies)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_bytes(args.manifest.read_bytes())
    receipts = []
    for row in cohorts:
        verify(dependencies)
        if checksum(args.manifest) != args.manifest_sha256:
            raise ValueError("Campaign manifest changed during execution")
        receipt = dict(name=row["name"], purpose=row["purpose"],
                       command=row["command"],
                       started_utc=datetime.now(timezone.utc).isoformat(),
                       manifest_sha256=args.manifest_sha256,
                       driver_sha256=checksum(__file__))
        (args.output / (row["name"] + ".started.json")).write_text(
            json.dumps(receipt, indent=2) + "\n")
        start = time.monotonic()
        with (args.output / (row["name"] + ".log")).open("x") as log:
            process = subprocess.run(row["command"], stdout=log,
                                     stderr=subprocess.STDOUT, check=False)
        receipt.update(returncode=process.returncode,
                       elapsed_seconds=time.monotonic() - start,
                       finished_utc=datetime.now(timezone.utc).isoformat())
        # Audit only after the device process exits. Running capture comparisons
        # alongside the next request would contaminate CPU-sensitive timings.
        if process.returncode == 0 and row.get("admission_command"):
            begin = time.monotonic()
            with (args.output / (row["name"] + ".audit.log")).open("x") as log:
                admission = subprocess.run(row["admission_command"], stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            receipt.update(admission_returncode=admission.returncode,
                           admission_elapsed_seconds=time.monotonic() - begin)
        try:
            verify(dependencies)
            receipt["sealed_dependencies_unchanged"] = True
        except (OSError, ValueError) as error:
            receipt.update(sealed_dependencies_unchanged=False,
                           dependency_error=str(error))
        receipts.append(receipt)
        (args.output / (row["name"] + ".result.json")).write_text(
            json.dumps(receipt, indent=2) + "\n")
        (args.output / "summary.json").write_text(
            json.dumps(receipts, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        if (process.returncode or receipt.get("admission_returncode", 0)
                or not receipt["sealed_dependencies_unchanged"]):
            raise SystemExit(process.returncode or receipt.get("admission_returncode") or 1)


if __name__ == "__main__":
    main()
