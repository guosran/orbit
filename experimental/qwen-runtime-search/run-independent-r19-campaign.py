#!/usr/bin/env python3
"""Collect independent sealed cohorts despite hardware/numeric rejection.

A rejected run never contributes prices. Dependency or manifest drift stops the
campaign immediately; ordinary execution/admission failures remain in receipts
and do not prevent unrelated cohorts from being measured. No timeout is set.
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
        if (str(requested.resolve(strict=True)) != row["file"]
                or checksum(requested) != row["sha256"]):
            raise ValueError("Sealed dependency changed: " + row["file"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != args.manifest_sha256:
        raise ValueError("Campaign manifest identity mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest["schema"] != "orbit.qwen-r19-sealed-campaign.v1":
        raise ValueError("Unknown campaign schema")
    policy = manifest.get("independent_execution_policy", {})
    if policy.get("failure_policy") != "continue-hardware-or-gate-failure-stop-on-seal-drift":
        raise ValueError("Explicit independent execution failure policy required")
    if policy.get("driver", {}).get("sha256") != checksum(__file__):
        raise ValueError("Independent driver identity mismatch")
    dependencies, cohorts = manifest["dependencies"], manifest["cohorts"]
    names = [row["name"] for row in cohorts]
    if (not dependencies or not cohorts or len(names) != len(set(names))
            or names != policy.get("execution_order")
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names)):
        raise ValueError("Unique safe ordered cohort names and dependencies required")
    for row in cohorts:
        for field in ("command", "admission_command"):
            if (not isinstance(row.get(field), list) or not row[field]
                    or any(type(value) is not str for value in row[field])):
                raise ValueError("Explicit command and numeric gate argument arrays required")
        if row.get("purpose") not in ("training", "heldout", "validation-only"):
            raise ValueError("Explicit cohort purpose required")

    def check_seals():
        if checksum(args.manifest) != args.manifest_sha256:
            raise ValueError("Campaign manifest changed during execution")
        verify(dependencies)

    check_seals()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_bytes(manifest_bytes)
    receipts, failed, seal_drift = [], False, False

    def write_summary():
        (args.output / "summary.json").write_text(json.dumps(receipts, indent=2) + "\n")

    for row in cohorts:
        try:
            check_seals()
        except (OSError, ValueError) as error:
            seal_drift = True
            (args.output / "seal-drift.json").write_text(json.dumps(
                {"before_cohort": row["name"], "error": str(error)}, indent=2) + "\n")
            break
        receipt = {"name": row["name"], "purpose": row["purpose"],
                   "command": row["command"], "admission_command": row["admission_command"],
                   "started_utc": datetime.now(timezone.utc).isoformat(),
                   "manifest_sha256": args.manifest_sha256, "driver_sha256": checksum(__file__)}
        (args.output / (row["name"] + ".started.json")).write_text(json.dumps(receipt, indent=2) + "\n")
        begin = time.monotonic()
        try:
            with (args.output / (row["name"] + ".log")).open("x") as log:
                result = subprocess.run(row["command"], stdout=log, stderr=subprocess.STDOUT, check=False)
            receipt["returncode"] = result.returncode
        except OSError as error:
            receipt.update(returncode=127, execution_error=str(error))
        receipt["elapsed_seconds"] = time.monotonic() - begin
        # Check immediately after the hardware process, before invoking its gate.
        try:
            check_seals()
            if receipt["returncode"] == 0:
                begin = time.monotonic()
                try:
                    with (args.output / (row["name"] + ".audit.log")).open("x") as log:
                        result = subprocess.run(row["admission_command"], stdout=log,
                                                stderr=subprocess.STDOUT, check=False)
                    receipt["admission_returncode"] = result.returncode
                except OSError as error:
                    receipt.update(admission_returncode=127, admission_error=str(error))
                receipt["admission_elapsed_seconds"] = time.monotonic() - begin
                check_seals()
            receipt["sealed_dependencies_unchanged"] = True
        except (OSError, ValueError) as error:
            receipt.update(sealed_dependencies_unchanged=False, dependency_error=str(error))
            seal_drift = True
        accepted = (receipt["returncode"] == 0 and receipt.get("admission_returncode") == 0
                    and receipt["sealed_dependencies_unchanged"])
        receipt.update(included_in_prices=accepted,
                       finished_utc=datetime.now(timezone.utc).isoformat())
        failed |= not accepted
        receipts.append(receipt)
        (args.output / (row["name"] + ".result.json")).write_text(json.dumps(receipt, indent=2) + "\n")
        write_summary()
        print(json.dumps(receipt), flush=True)
        if seal_drift:
            break
    write_summary()
    (args.output / "campaign-status.json").write_text(json.dumps(
        {"completed_cohorts": len(receipts), "planned_cohorts": len(cohorts),
         "stopped_on_seal_drift": seal_drift, "any_failure": failed,
         "all_planned_cohorts_attempted": len(receipts) == len(cohorts)}, indent=2) + "\n")
    return 1 if failed or seal_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
