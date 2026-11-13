#!/usr/bin/env python3
"""Resume independent cohort executions, preserving every success and failure.

This creates fresh manifests. It never edits a sealed ancestor or rewrites a
successful cohort's preparation protocol. Every retained row needs a matching
successful sealed execution receipt, including rows omitted by a continuation.
"""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

SCHEMA = "orbit.qwen-r19-sealed-campaign.v1"


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    requested = Path(path).absolute()
    resolved = requested.resolve(strict=True)
    with resolved.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": str(resolved), "requested_file": str(requested), "sha256": checksum}


def checked_reference(reference):
    actual = record(reference.get("requested_file", reference["file"]))
    if (actual["file"] != str(Path(reference["file"]).resolve(strict=True))
            or actual["sha256"] != reference["sha256"]):
        raise ValueError("Changed ancestor requested path or checksum")
    return actual


def cohorts(manifest):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported campaign schema")
    rows = manifest["cohorts"]
    indexed = {row["name"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate cohort names")
    return indexed


def successful(row):
    return (row.get("returncode") == 0 and row.get("admission_returncode") == 0
            and row.get("sealed_dependencies_unchanged") is True)


def verify_receipts(manifest, manifest_record, receipts):
    indexed = cohorts(manifest)
    names = [row["name"] for row in receipts]
    if names != list(indexed)[:len(names)] or len(names) != len(set(names)):
        raise ValueError("Receipts must be a unique ordered execution prefix")
    for i, receipt in enumerate(receipts):
        row = indexed[receipt["name"]]
        if (receipt.get("manifest_sha256") != manifest_record["sha256"]
                or receipt.get("command") != row["command"]
                or receipt.get("purpose") != row["purpose"]):
            raise ValueError("Receipt manifest, command or purpose mismatch")
        if receipt.get("sealed_dependencies_unchanged") is not True and i + 1 < len(receipts):
            raise ValueError("Execution continued after dependency seal drift")
    return indexed


def same_measurement(left, right):
    # The first continuation explicitly labelled the previously unlabelled
    # preparation protocol. All execution/admission fields must remain exact.
    return ({k: v for k, v in left.items() if k != "preparation_protocol"}
            == {k: v for k, v in right.items() if k != "preparation_protocol"})


def ancestor_backing(admission, seen=None):
    """Verify small manifest/receipt links, without rehashing multi-GB models."""
    seen = set() if seen is None else seen
    backing, dependencies = [], []
    lineage = admission.get("resume_provenance")
    if lineage:
        admission_ref = checked_reference(lineage["admission_manifest"])
        if admission_ref["sha256"] in seen:
            raise ValueError("Cyclic admission provenance")
        seen.add(admission_ref["sha256"])
        older = read(admission_ref["file"])
        backing, dependencies = ancestor_backing(older, seen)
        execution_ref = checked_reference(lineage["execution_manifest"])
        receipts_ref = checked_reference(lineage["execution_receipts"])
        execution = read(execution_ref["file"])
        receipts = read(receipts_ref["file"])
        dependencies.append(admission_ref)
    else:
        lineage = admission.get("preparation_continuation")
        if not lineage:
            return backing, dependencies
        execution_ref = checked_reference(lineage["parent"])
        receipts_ref = checked_reference(lineage["parent_receipts"])
        execution = read(execution_ref["file"])
        receipts = read(receipts_ref["file"])
    if (execution["source_id"] != admission["source_id"]
            or execution["whole_source"] != admission["whole_source"]):
        raise ValueError("Ancestor source binding changed")
    indexed = verify_receipts(execution, execution_ref, receipts)
    dependencies.extend([execution_ref, receipts_ref])
    for receipt in receipts:
        if successful(receipt):
            backing.append({"cohort": indexed[receipt["name"]],
                            "execution_receipt": receipt,
                            "execution_manifest": execution_ref,
                            "execution_receipts": receipts_ref})
    return backing, dependencies


def resume(admission_path, execution_path, receipts_path, preparation, protocol, output, driver):
    if not protocol.strip():
        raise ValueError("Explicit preparation protocol is required")
    admission_ref, execution_ref, receipts_ref = map(
        record, (admission_path, execution_path, receipts_path))
    admission, execution, receipts = map(read, (admission_path, execution_path, receipts_path))
    all_rows = cohorts(admission)
    executed = verify_receipts(execution, execution_ref, receipts)
    if (admission["source_id"] != execution["source_id"]
            or admission["whole_source"] != execution["whole_source"]):
        raise ValueError("Admission/execution source mismatch")
    if any(name not in all_rows or all_rows[name] != row for name, row in executed.items()):
        raise ValueError("Execution must be an exact subset of combined admission")
    backing, ancestor_dependencies = ancestor_backing(admission)
    outcomes = {row["name"]: row for row in receipts}
    prep_ref = record(preparation)
    pending, combined, retained_backing = [], [], []
    rejected = copy.deepcopy(admission.get("rejected_cohorts", []))
    for item in execution.get("rejected_cohorts", []):
        if item not in rejected:
            rejected.append(copy.deepcopy(item))
    for name, row in all_rows.items():
        outcome = outcomes.get(name)
        proof = None
        if name not in executed:
            proof = next((item for item in reversed(backing)
                          if same_measurement(row, item["cohort"])), None)
            if proof is None:
                raise ValueError("Omitted cohort lacks successful receipt backing: " + name)
        elif outcome and successful(outcome):
            proof = {"cohort": row, "execution_receipt": outcome,
                     "execution_manifest": execution_ref, "execution_receipts": receipts_ref}
        if proof:
            combined.append(copy.deepcopy(row))
            retained_backing.append(copy.deepcopy(proof))
            continue
        if outcome:
            rejected.append({"cohort": name, "execution_receipt": copy.deepcopy(outcome),
                             "execution_manifest": execution_ref,
                             "execution_receipts": receipts_ref,
                             "failure_retained": True, "included_in_prices": False})
        updated = copy.deepcopy(row)
        old = row["command"]
        if not isinstance(old, list) or len(old) < 3 or not all(isinstance(x, str) for x in old):
            raise ValueError("Invalid runtime command")
        entry, arguments = old[1], old[2:]
        if arguments[:1] == ["--runtime-runner"]:
            entry, arguments = arguments[1], arguments[2:]
        if Path(entry).name != "run-qwen-r19-ready.py":
            raise ValueError("Unexpected underlying timed runtime")
        if arguments.count("--output") != 1:
            raise ValueError("Expected one output argument")
        position = arguments.index("--output") + 1
        old_directory = arguments[position]
        new_directory = str((Path(output) / ("train-" + name)).absolute())
        arguments = list(arguments)
        arguments[position] = new_directory
        updated["command"] = [old[0], prep_ref["requested_file"], "--runtime-runner", entry, *arguments]
        updated["numeric_gate_file"] = str(Path(new_directory) / "gate.json")
        updated["admission_command"] = [value.replace(old_directory, new_directory)
                                         for value in row["admission_command"]]
        updated["preparation_protocol"] = protocol
        combined.append(updated)
        pending.append(updated)
    # Preserve every recorded dependency, including distinct requested symlinks.
    dependencies = copy.deepcopy(admission["dependencies"])
    for item in execution["dependencies"] + ancestor_dependencies + [
            admission_ref, execution_ref, receipts_ref, prep_ref, record(__file__)]:
        if item not in dependencies:
            dependencies.append(copy.deepcopy(item))
    common = copy.deepcopy(admission)
    common.update(dependencies=dependencies, rejected_cohorts=rejected,
                  retained_receipt_backing=retained_backing,
                  resume_provenance={"admission_manifest": admission_ref,
                                     "execution_manifest": execution_ref,
                                     "execution_receipts": receipts_ref,
                                     "entrypoint": prep_ref,
                                     "retained_measured_cohorts": len(retained_backing),
                                     "pending_cohorts": len(pending),
                                     "preparation_strata": dict(Counter(
                                         row.get("preparation_protocol", "unspecified")
                                         for row in combined)),
                                     "changed_timed_runtime_files": False,
                                     "changed_memory_guards": False},
                  parent_timings_reused=bool(retained_backing),
                  retained_r19_measurements=bool(retained_backing),
                  old_r18_timings_reused=False, complete_plan_prepricing=False)
    common["dependencies"].append(record(driver))
    def execution_class(row):
        name = row["name"]
        if name.startswith(("cpu-", "half-", "decode-")):
            return 0
        if "ffn-reload" in name:
            return 1
        if "ffn-release-after-call" in name:
            return 2
        raise ValueError("Unclassified independent pending cohort: " + name)
    pending.sort(key=execution_class)
    common["independent_execution_policy"] = {
        "failure_policy": "continue-hardware-or-gate-failure-stop-on-seal-drift",
        "driver": record(driver),
        "execution_order_policy": "non-ffn-first-then-reload-then-release",
        "execution_order": [row["name"] for row in pending],
        "timed_requests_changed": False}
    Path(output).mkdir(parents=True, exist_ok=False)
    for filename, rows in (("campaign-execution.json", pending),
                           ("campaign-admission.json", combined)):
        with (Path(output) / filename).open("x") as stream:
            json.dump(dict(common, cohorts=rows), stream, indent=2, allow_nan=False)
            stream.write("\n")
    return common["resume_provenance"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("admission-manifest", "execution-manifest", "execution-receipts",
                 "preparation-runtime", "output", "driver"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preparation-protocol", required=True)
    args = parser.parse_args()
    print(json.dumps(resume(args.admission_manifest, args.execution_manifest,
                            args.execution_receipts, args.preparation_runtime,
                            args.preparation_protocol, args.output, args.driver)))


if __name__ == "__main__":
    main()
