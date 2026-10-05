#!/usr/bin/env python3
"""Exercise the real original-AMOEBA replica retimer with isolated bad evidence."""

import argparse
import copy
import json
from pathlib import Path
import subprocess


PASS_PREFIX = "--retime-original-amoeba-fixed-decisions="
MANIFEST_SCHEMA = "amoeba-original-replica-profile-evidence-v1"
RECORD_KEYS = {
    "parent_task", "replica_id", "materialized_task", "function",
    "materialized_module", "profile_file", "body_export_file",
}


def read(path):
    return json.loads(Path(path).read_text())


def pass_options(argv):
    matches = [index for index, arg in enumerate(argv)
               if arg.startswith(PASS_PREFIX)]
    if len(matches) != 1:
        raise ValueError("retimer command must contain exactly one " + PASS_PREFIX)
    index = matches[0]
    values = {}
    for item in argv[index][len(PASS_PREFIX):].split():
        if "=" not in item:
            raise ValueError("retimer pass option is not key=value: " + item)
        key, value = item.split("=", 1)
        if not key or key in values:
            raise ValueError("retimer pass options contain an empty or duplicate key")
        values[key] = value
    return index, values


def command(argv, updates, output_json, output_ir, cpu):
    result = list(argv)
    index, values = pass_options(result)
    values.update(updates)
    values["output"] = str(output_json)
    result[index] = PASS_PREFIX + " ".join(
        key + "=" + str(value) for key, value in values.items())
    output_indices = [i for i, arg in enumerate(result[:-1]) if arg == "-o"]
    if len(output_indices) != 1:
        raise ValueError("retimer command must contain exactly one '-o <path>'")
    result[output_indices[0] + 1] = str(output_ir)
    if result[:2] == ["taskset", "--cpu-list"]:
        result[2] = str(cpu)
    else:
        result = ["taskset", "--cpu-list", str(cpu)] + result
    return result


def exact_manifest(manifest):
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("replica evidence manifest has an unsupported schema")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("replica evidence manifest has no records")
    for record in records:
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise ValueError("replica evidence records must have exactly the seven schema fields")
    return records


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retimer-command", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu", type=int, choices=range(12), default=5)
    args = parser.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    original_argv = read(args.retimer_command)["argv"]
    pass_index, original_options = pass_options(original_argv)
    del pass_index
    if original_options.get("diagnostic-ii-ceiling", "20") not in ("20", "23"):
        raise ValueError("retimer command must use the supported runtime ceiling 20 or 23")
    runtime_ceiling = int(original_options.get("diagnostic-ii-ceiling", "20"))
    evidence_path = Path(original_options["replica-profile-evidence-file"]).resolve()
    catalog_path = Path(original_options["parent-cost-file"]).resolve()
    source_manifest = read(evidence_path)
    source_records = exact_manifest(source_manifest)
    source_catalog = read(catalog_path)
    reports = []
    skipped = []

    def save_report():
        write_json(root / "acceptance.json", {
            "status": "pass" if all(row["pass"] for row in reports) else "fail",
            "case_count": len(reports),
            "passed_count": sum(bool(row["pass"]) for row in reports),
            "skipped_cases": skipped,
            "cases": reports,
        })

    def run(label, argv, result_path, expect_success):
        directory = root / label
        directory.mkdir(exist_ok=True)
        write_json(directory / "command.json", {"argv": argv})
        with (directory / "stdout.log").open("wb") as stdout, \
                (directory / "stderr.log").open("wb") as stderr:
            completed = subprocess.run(argv, stdout=stdout, stderr=stderr)
        diagnostic = (directory / "stderr.log").read_text(errors="replace")
        passed = (completed.returncode == 0 and result_path.is_file()) if expect_success else (
            completed.returncode != 0 and not result_path.exists()
            and "error:" in diagnostic.lower())
        row = {"case": label, "expected_success": expect_success,
               "exit_code": completed.returncode,
               "result_exists": result_path.exists(), "pass": passed}
        reports.append(row)
        save_report()
        if not passed:
            raise RuntimeError(label + " did not meet its expected outcome; see "
                               + str(directory / "stderr.log"))

    positive_result_path = root / "retimer-positive.json"
    run("retimer-positive",
        command(original_argv, {}, positive_result_path,
                root / "retimer-positive.mlir", args.cpu),
        positive_result_path, True)
    positive = read(positive_result_path)
    materialization = positive.get("replica_materialization")
    if (positive.get("schema") != "amoeba-original-fixed-decision-retiming-v1"
            or positive.get("valid") is not True
            or not isinstance(materialization, dict)
            or materialization.get("schema") != "amoeba-original-replica-materialization-v1"
            or materialization.get("verified") is not True):
        raise ValueError("positive retimer result lacks a verified replica materialization report")

    child_name = source_records[0]["materialized_task"]
    child_parent = source_records[0]["parent_task"]
    child_replica = source_records[0]["replica_id"]
    cost_rows = {row.get("task"): row for row in positive.get("task_costs", [])
                 if isinstance(row, dict)}
    if child_name not in cost_rows:
        raise ValueError("positive retimer output omits the first evidenced child cost row")
    selected_shape = cost_rows[child_name].get("selected_profile_shape")
    if not isinstance(selected_shape, str):
        raise ValueError("positive child cost row omits its selected profile orientation")

    def run_tamper(label, mutate):
        directory = root / label
        directory.mkdir()
        manifest = copy.deepcopy(source_manifest)
        records = manifest["records"]
        profile_paths = {}
        body_paths = {}
        profile_docs = {}
        body_docs = {}
        for index, record in enumerate(records):
            old_profile = record["profile_file"]
            if old_profile not in profile_paths:
                new_profile = directory / ("child-profiles-" + str(len(profile_paths)) + ".json")
                profile_paths[old_profile] = new_profile
                profile_docs[old_profile] = read(old_profile)
            old_body = record["body_export_file"]
            if old_body not in body_paths:
                new_body = directory / ("child-bodies-" + str(len(body_paths)) + ".json")
                body_paths[old_body] = new_body
                body_docs[old_body] = read(old_body)
            record["profile_file"] = str(profile_paths[old_profile])
            record["body_export_file"] = str(body_paths[old_body])
        catalog = copy.deepcopy(source_catalog)
        mutate(manifest, records, profile_docs, body_docs, catalog)
        for old_path, document in profile_docs.items():
            write_json(profile_paths[old_path], document)
        for old_path, document in body_docs.items():
            write_json(body_paths[old_path], document)
        manifest_path = directory / "replica-profile-evidence.json"
        catalog_copy = directory / "parent-cost-catalog.json"
        write_json(manifest_path, manifest)
        write_json(catalog_copy, catalog)
        result_path = directory / "retimer-result.json"
        argv = command(original_argv, {
            "replica-profile-evidence-file": manifest_path,
            "parent-cost-file": catalog_copy,
        }, result_path, directory / "retimer-output.mlir", args.cpu)
        run(label, argv, result_path, False)

    def mutate_missing_replica(manifest, records, *_):
        records.pop()

    def mutate_duplicate_replica(manifest, records, *_):
        records.append(copy.deepcopy(records[0]))

    def mutate_wrong_parent(manifest, records, *_):
        records[0]["parent_task"] = "__unknown_parent__"

    def mutate_wrong_replica(manifest, records, *_):
        records[0]["replica_id"] = 999

    def mutate_wrong_function(manifest, records, *_):
        records[0]["function"] += ".forged"

    def mutate_wrong_child(manifest, records, *_):
        records[0]["materialized_task"] = "__unknown_materialized_child__"

    def mutate_extra_record_field(manifest, records, *_):
        records[0]["unexpected"] = "not-in-v1-schema"

    def mutate_missing_attempt(manifest, records, profile_docs, *_):
        record = records[0]
        document = profile_docs[source_records[0]["profile_file"]]
        attempts = document["candidate_attempts"]
        index = next(i for i, row in enumerate(attempts)
                     if row.get("task") == child_name)
        attempts.pop(index)

    def mutate_wrong_candidate_index(manifest, records, profile_docs, *_):
        document = profile_docs[source_records[0]["profile_file"]]
        attempt = next(row for row in document["candidate_attempts"]
                       if row.get("task") == child_name and row.get("shape") == "1x1")
        attempt["candidate_index_in_task"] = 2

    def selected_profile(profile_docs):
        document = profile_docs[source_records[0]["profile_file"]]
        task = next(row for row in document["tasks"] if row["task"] == child_name)
        return next(row for row in task["profiles"]
                    if row.get("composed_cgra_shape") == selected_shape)

    def mutate_wrong_sample_trip(manifest, records, profile_docs, *_):
        profile = selected_profile(profile_docs)
        profile["sample_trip_count"] += 1
        profile["estimated_latency"] = (
            profile["compiled_ii"] * (profile["sample_trip_count"] - 1)
            + profile["steps"])

    def mutate_profile_ii_over_ceiling(manifest, records, profile_docs, *_):
        profile = selected_profile(profile_docs)
        profile["compiled_ii"] = runtime_ceiling + 1
        profile["estimated_latency"] = (
            profile["compiled_ii"] * (profile["sample_trip_count"] - 1)
            + profile["steps"])

    def mutate_forged_child_body(manifest, records, _, body_docs, __):
        body = body_docs[source_records[0]["body_export_file"]]
        task = next(row for row in body["tasks"] if row["task"] == child_name)
        task["normalized_mapper_body"] = "forged-normalized-mapper-body"

    def mutate_missing_singleton_body(manifest, records, _, body_docs, __):
        singleton_names = {
            row.get("task") for row in positive.get("task_costs", [])
            if isinstance(row, dict) and row.get("replica_id") is None
        }
        if not singleton_names:
            raise ValueError("positive module has no singleton task for singleton-body negative")
        body = body_docs[source_records[0]["body_export_file"]]
        removed = singleton_names & {row.get("task") for row in body["tasks"]}
        if not removed:
            raise ValueError("companion body export does not include a singleton identity task")
        remove_name = sorted(removed)[0]
        body["tasks"] = [row for row in body["tasks"] if row.get("task") != remove_name]
        body["task_count"] = len(body["tasks"])

    def mutate_missing_canonical_witness(manifest, records, _, __, catalog):
        catalog["original_amoeba_profile_binding"].pop("canonical_module_witness", None)

    def mutate_forged_startup(manifest, records, _, __, catalog):
        rows, cols = map(int, selected_shape.split("x"))
        task = next(row for row in catalog["original_amoeba_profile_binding"]["tasks"]
                    if row.get("task") == child_parent)
        prediction = next(row for row in task["direct_shape_predictions"]
                          if row.get("cgra_rows") == rows and row.get("cgra_cols") == cols)
        prediction["startup_cycles"] += 1

    mutations = [
        ("missing-replica-record", mutate_missing_replica),
        ("duplicate-replica-record", mutate_duplicate_replica),
        ("wrong-parent", mutate_wrong_parent),
        ("wrong-replica-id", mutate_wrong_replica),
        ("wrong-function", mutate_wrong_function),
        ("wrong-materialized-child", mutate_wrong_child),
        ("extra-record-field", mutate_extra_record_field),
        ("missing-candidate-attempt", mutate_missing_attempt),
        ("wrong-candidate-index", mutate_wrong_candidate_index),
        ("wrong-sample-trip", mutate_wrong_sample_trip),
        ("profile-ii-over-runtime-ceiling", mutate_profile_ii_over_ceiling),
        ("forged-child-normalized-body", mutate_forged_child_body),
        ("missing-canonical-module-witness", mutate_missing_canonical_witness),
        ("forged-selected-startup", mutate_forged_startup),
    ]
    singleton_names = {
        row.get("task") for row in positive.get("task_costs", [])
        if isinstance(row, dict) and row.get("replica_id") is None
    }
    if singleton_names:
        mutations.append(("missing-singleton-body-export", mutate_missing_singleton_body))
    else:
        skipped.append({"case": "missing-singleton-body-export",
                        "reason": "positive module has no singleton task"})

    for label, mutate in mutations:
        run_tamper(label, mutate)

    save_report()
    print(json.dumps({"status": "pass", "case_count": len(reports),
                      "output_root": str(root)}))


if __name__ == "__main__":
    main()
