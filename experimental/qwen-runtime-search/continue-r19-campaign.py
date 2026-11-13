#!/usr/bin/env python3
"""Continue a failed campaign without overwriting its successful measurements.

Only a separate, proved preparation entry point changes. Existing runtime,
source, model, configuration and numerical gates remain sealed. The execution
manifest contains pending cohorts; the admission manifest also references every
retained completed cohort and records rejected attempts separately.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    requested = Path(path).absolute()
    path = requested.resolve(strict=True)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(file=str(path), requested_file=str(requested), sha256=checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("parent-manifest", "parent-receipts", "preparation-runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    parent, receipts = read(args.parent_manifest), read(args.parent_receipts)
    outcomes = {row["name"]: row for row in receipts}
    if len(outcomes) != len(receipts):
        raise ValueError("Duplicated parent execution receipts")
    if any(row["manifest_sha256"] != record(args.parent_manifest)["sha256"] for row in receipts):
        raise ValueError("Execution receipts belong to another parent manifest")
    kept, pending, rejected = [], [], []
    args.output.mkdir(parents=True, exist_ok=False)
    for row in parent["cohorts"]:
        outcome = outcomes.get(row["name"])
        if (outcome and outcome["returncode"] == 0
                and outcome.get("admission_returncode") == 0
                and outcome["sealed_dependencies_unchanged"]):
            retained = copy.deepcopy(row)
            retained["preparation_protocol"] = "original-frozen-preparation"
            kept.append(retained)
            continue
        if outcome:
            rejected.append(dict(cohort=row["name"], execution_receipt=outcome,
                                 failure_retained=True, included_in_prices=False))
        updated = copy.deepcopy(row)
        old = row["command"]
        old_directory = Path(old[old.index("--output") + 1])
        new_directory = (args.output / ("train-" + row["name"])).resolve()
        if Path(old[1]).name != "run-qwen-r19-ready.py":
            raise ValueError("Unexpected original runtime entry point")
        command = [old[0], str(args.preparation_runtime.resolve(strict=True)),
                   "--runtime-runner", old[1], *old[2:]]
        command[command.index("--output") + 1] = str(new_directory)
        updated["command"] = command
        updated["numeric_gate_file"] = str(new_directory / "gate.json")
        updated["admission_command"] = [
            value.replace(str(old_directory), str(new_directory))
            for value in updated["admission_command"]]
        updated["preparation_protocol"] = "owned-post-decode-readonly-residency-preparation"
        pending.append(updated)
    extras = [record(args.parent_manifest), record(args.parent_receipts),
              record(args.preparation_runtime), record(__file__)]
    dependencies = list(parent["dependencies"]) + extras
    common = dict(schema=parent["schema"], source_id=parent["source_id"],
                  whole_source=parent["whole_source"], dependencies=dependencies,
                  preparation_continuation=dict(parent=record(args.parent_manifest),
                      parent_receipts=record(args.parent_receipts),
                      entrypoint=record(args.preparation_runtime),
                      retained_measured_cohorts=len(kept), pending_cohorts=len(pending),
                      mixed_preparation_protocols_explicit=True,
                      changed_timed_runtime_files=False, changed_memory_guards=False),
                  rejected_cohorts=rejected, parent_timings_reused=bool(kept),
                  retained_r19_measurements=bool(kept), old_r18_timings_reused=False,
                  complete_plan_prepricing=False)
    for filename, cohorts in (("campaign-execution.json", pending),
                              ("campaign-admission.json", kept + pending)):
        result = dict(common, cohorts=cohorts)
        path = args.output / filename
        with path.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(record(path)))


if __name__ == "__main__":
    main()
