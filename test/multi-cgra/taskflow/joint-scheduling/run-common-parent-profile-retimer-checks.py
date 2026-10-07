#!/usr/bin/env python3
"""Exercise native common-parent-profile retiming and fail-closed bindings.

The fixture is intentionally one task. The runner invokes the source-owned
mapper for all eight canonical rectangles, installs a one-task original-F45
trace whose duration is taken from the native mapper row, and runs the common
native retimer. No Python cost values are invented: Python only copies the
mapper's JSON witness into the trace and tampers with copies for negative
controls.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "common-parent-profile-retimer.mlir"
FUNCTION = "common_profile_mapper"
PASS_PREFIX = "--retime-original-amoeba-fixed-decisions="
PROFILE_PREFIX = "--map-joint-scheduling-tasks="
PROFILE_BINDING_SCHEMA = "orbit-original-f45-common-parent-profile-binding-v1"
PROFILE_PROVENANCE = "source-owned-common-parent-profile-only-mapped-duration-v1"
REPLICA_RULE = "ceil-full-parent-mapped-duration-over-original-active-replicas-v1"


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def matching_brace(text: str, start: int) -> int:
    if start >= len(text) or text[start] != "{":
        fail("expected an opening MLIR dictionary brace")
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
    fail("unterminated MLIR attribute dictionary")
    raise AssertionError("unreachable")


def append_dictionary_attrs(text: str, marker: str, additions: str) -> str:
    start = text.find(marker)
    if start < 0 or text.find(marker, start + len(marker)) >= 0:
        fail(f"expected one MLIR attribute anchor: {marker}")
    opening = text.find("{", start + len(marker))
    if opening < 0:
        fail(f"attribute anchor {marker} has no dictionary")
    closing = matching_brace(text, opening)
    if not text[opening + 1:closing].strip():
        fail(f"attribute anchor {marker} has an empty dictionary")
    return text[:closing] + ", " + additions + text[closing:]


def checked_run(args: argparse.Namespace, label: str, command: list[str],
                *, expect_success: bool) -> subprocess.CompletedProcess[str]:
    log = args.output_root / "logs" / f"{label}.log"
    record = args.output_root / "logs" / f"{label}.command.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    actual = ["taskset", "--cpu-list", args.cpu_list, *command]
    result = subprocess.run(actual, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    write_json(record, {"argv": actual, "exit_code": result.returncode})
    if expect_success and result.returncode != 0:
        fail(f"{label} failed with exit code {result.returncode}; see {log}")
    if not expect_success and result.returncode == 0:
        fail(f"{label} unexpectedly succeeded; see {log}")
    return result


def run_profile_export(args: argparse.Namespace, canonical: Path,
                       profiles: Path, output_mlir: Path) -> None:
    options = {
        "function": FUNCTION,
        "candidate-id": "candidate-0",
        "parent-profile-output": str(profiles),
    }
    command = [
        str(args.optimizer), f"--neura-architecture-spec={args.architecture}",
        str(canonical), PROFILE_PREFIX + " ".join(
            f"{key}={value}" for key, value in options.items()),
        "--verify-each", "-o", str(output_mlir),
    ]
    checked_run(args, "common-profile-export", command, expect_success=True)


def make_f45_schedule(canonical: Path, profiles: Path,
                      scheduled: Path) -> None:
    profile = read_json(profiles)["tasks"][0]["profiles"]
    selected = next((row for row in profile
                     if row["composed_cgra_shape"] == "1x1"), None)
    if selected is None:
        fail("native source profile export omitted the selected 1x1 mapper row")
    # These are actual output values from the C++ mapper profile. The F45
    # trace is fixture metadata and uses the supplied common duration formula.
    ii = selected["compiled_ii"]
    duration = selected["estimated_latency"]
    steps = selected["steps"]
    operations = selected["materialized_operation_count"]
    trips = selected["sample_trip_count"]
    schedule_info = (
        '{active_replicas = 1 : i32, '
        'composed_cgra_count = 1 : i64, composed_cgra_shape = "1x1", '
        'dispatch_index = 0 : i64, duration = ' + str(duration) + ' : i64, '
        'duration_unit = "profiled-cycles", '
        'placements = [{col = 0 : i32, context_id = 0 : i32, '
        'replica_id = 0 : i32, row = 0 : i32, scheduler_duration = ' +
        str(duration) + ' : i64, scheduler_end_time = ' + str(duration) +
        ' : i64, scheduler_start_time = 0 : i64}], '
        'replica_shapes = [{cgra_count = 1 : i32, col = 0 : i32, '
        'placement_cols = 1 : i32, placement_rows = 1 : i32, '
        'replica_id = 0 : i32, row = 0 : i32, shape = "1x1"}], '
        'scheduler_end_time = ' + str(duration) + ' : i64, '
        'scheduler_start_time = 0 : i64, task_name = "Task"}')
    task_additions = (
        'active_replicas = 1 : i32, '
        'amoeba.task_scheduler_schedule_info = ' + schedule_info + ', '
        'composed_cgra_count = 1 : i32, composed_cgra_shape = "1x1", '
        'profile_info = {compiled_ii = ' + str(ii) + ' : i32, duration = ' +
        str(duration) + ' : i32, materialized_operation_count = ' +
        str(operations) + ' : i32, sample_trip_count = ' + str(trips) +
        ' : i32, steps = ' + str(steps) + ' : i32}, '
        'task_orchestration_info = {cgra_positions = [{col = 0 : i32, '
        'context_id = 0 : i32, row = 0 : i32}], '
        'read_sram_locations = [{col = 0 : i32, row = 0 : i32}], '
        'write_sram_locations = [{col = 0 : i32, row = 0 : i32}]}')
    function_additions = (
        'task_orchestration_summary = {pipeline_interval = 1 : i64, '
        'strategy = "throughput-guided"}, '
        'amoeba.task_scheduler_dispatch_order = ["Task"], '
        'amoeba.task_scheduler_task_schedule = [' + schedule_info + '], '
        'amoeba.task_scheduler_time_scale = 1 : i64, '
        'amoeba.task_scheduler_time_unit = "scaled-internal-placement-slots"')
    text = canonical.read_text()
    text = append_dictionary_attrs(text, "taskflow.task @Task",
                                   task_additions)
    text = append_dictionary_attrs(text, "func.func @" + FUNCTION,
                                   function_additions)
    scheduled.write_text(text)


def run_retimer(args: argparse.Namespace, scheduled: Path,
                canonical: Path, profiles: Path, label: str,
                *, expect_success: bool) -> Path:
    result = args.output_root / f"{label}.json"
    output_mlir = args.output_root / f"{label}.mlir"
    options = {
        "function": FUNCTION,
        "common-parent-profile-file": str(profiles),
        "common-canonical-module-file": str(canonical),
        "reschedule-with-production-scheduler": "true",
        "original-f45-replica-scaling": "true",
        "output": str(result),
    }
    values = list(options.items())
    if any(any(character.isspace() for character in value)
           for _, value in values):
        fail("test paths must not contain whitespace because MLIR pass options are space-delimited")
    command = [
        str(args.optimizer), f"--neura-architecture-spec={args.architecture}",
        f"--joint-inter-task-network-spec={args.network}",
        str(scheduled), PASS_PREFIX + " ".join(
            f"{key}={value}" for key, value in values),
        "--verify-each", "-o", str(output_mlir),
    ]
    checked_run(args, label, command, expect_success=expect_success)
    if expect_success and (not result.is_file() or not output_mlir.is_file()):
        fail(f"{label} did not publish both result JSON and refreshed MLIR")
    if not expect_success and result.exists():
        fail(f"{label} published a partial result despite rejecting its witness")
    return result


def validate_positive(path: Path) -> None:
    result = read_json(path)
    binding = result.get("common_mapper_profile_binding")
    if not isinstance(binding, dict) or \
       binding.get("schema") != PROFILE_BINDING_SCHEMA or \
       binding.get("schema_version") != 1 or binding.get("verified") is not True:
        fail("positive result lacks the truthful common profile binding")
    if binding.get("profile_provenance") != PROFILE_PROVENANCE or \
       binding.get("child_mapper_profiles_used") is not False or \
       result.get("common_mapper_profile_binding_verified") is not True:
        fail("positive result claims unsupported profile or child evidence")
    if "body_equivalence_checked" in result or \
       "selected_profile_body_binding_schema" in result or \
       any(key.startswith("cost_catalog_") for key in result):
        fail("common mode leaked legacy body/catalog evidence claims")
    policy = result.get("replica_timing_policy", {})
    if policy.get("replica_duration_rule") != REPLICA_RULE or \
       policy.get("child_mapper_profiles_used") is not False or \
       policy.get("status") != "original-f45-scheduler-estimate":
        fail("positive result does not state the original-F45 parent estimate policy")
    trace = result.get("fixed_decision_trace", {})
    if trace.get("common_mapper_profile_binding_verified") is not True:
        fail("fixed decision trace lacks common profile binding")
    rows = trace.get("task_profile_bindings")
    if not isinstance(rows, list) or len(rows) != 1:
        fail("one-task fixture did not emit exactly one profile binding row")
    row = rows[0]
    expected_fields = {
        "schema": PROFILE_BINDING_SCHEMA,
        "task": "Task",
        "selected_profile_shape": "1x1",
        "selected_cgra_count": 1,
        "original_active_replicas": 1,
        "replica_duration_rule": REPLICA_RULE,
        "current_ir_wrapper_exact_match_verified": True,
        "mapper_succeeded": True,
    }
    for key, expected in expected_fields.items():
        if row.get(key) != expected:
            fail(f"task binding {key} is {row.get(key)!r}, expected {expected!r}")
    full = math.ceil(row["structural_startup_cycles"] +
                     row["compiled_ii"] * (row["sample_trip_count"] - 1))
    scaled = math.ceil(full / row["original_active_replicas"])
    if row.get("common_parent_full_mapped_duration_cycles") != full or \
       row.get("mapped_duration_cycles") != scaled:
        fail("positive result duration does not follow the declared parent/N formula")
    costs = result.get("task_costs")
    if not isinstance(costs, list) or len(costs) != 1 or \
       costs[0].get("mapped_duration_cycles") != scaled:
        fail("scheduler cost row differs from the checked common parent duration")


def with_profile_copy(args: argparse.Namespace, source: Path, label: str,
                      mutate) -> Path:
    profile = copy.deepcopy(read_json(source))
    mutate(profile)
    target = args.output_root / "negative-controls" / f"{label}.json"
    write_json(target, profile)
    return target


def exercise_profile_tampers(args: argparse.Namespace, scheduled: Path,
                             canonical: Path, profiles: Path) -> None:
    def wrapper(value) -> None:
        for row in value["tasks"][0]["profiles"]:
            row["pre_mapper_wrapper_bytes"] += "\n// untrusted wrapper edit"
            row["pre_mapper_wrapper_byte_count"] = len(
                row["pre_mapper_wrapper_bytes"].encode())

    def formula(value) -> None:
        row = value["tasks"][0]["profiles"][0]
        row["estimated_latency"] += 1

    def trip(value) -> None:
        value["tasks"][0]["sample_trip_count"] += 1

    def architecture(value) -> None:
        value["hardware_coordinates"]["total_mapper_rows"] += 1

    def attempts(value) -> None:
        rows = value["candidate_attempts"]
        rows[0], rows[1] = rows[1], rows[0]

    controls = [
        ("tampered-wrapper", wrapper),
        ("tampered-formula", formula),
        ("tampered-trip-count", trip),
        ("tampered-architecture", architecture),
        ("reordered-attempts", attempts),
    ]
    for label, mutate in controls:
        changed = with_profile_copy(args, profiles, label, mutate)
        run_retimer(args, scheduled, canonical, changed, label,
                    expect_success=False)


def exercise_semantic_dag_tamper(args: argparse.Namespace,
                                 scheduled: Path, canonical: Path,
                                 profiles: Path) -> None:
    text = scheduled.read_text()
    # The imported source-binding strings themselves contain escaped generic
    # assembly for the kernel, so replace the final textual occurrence: the
    # actual live kernel constant after those proof attributes.
    offset = text.rfind('value = 1 : i32')
    if offset < 0:
        fail("semantic tamper fixture no longer contains its expected kernel constant")
    mutated = text[:offset] + 'value = 2 : i32' + text[offset + len('value = 1 : i32'):]
    path = args.output_root / "negative-controls" / "tampered-taskflow-kernel.mlir"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mutated)
    run_retimer(args, path, canonical, profiles, "tampered-semantic-dag",
                expect_success=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu-list", default="0",
                        help="single CPU affinity for every compiler invocation")
    args = parser.parse_args()
    args.optimizer = args.optimizer.resolve()
    args.architecture = args.architecture.resolve()
    args.network = args.network.resolve()
    args.output_root = args.output_root.resolve()
    if not args.optimizer.is_file() or not args.architecture.is_file():
        fail("optimizer or architecture input is missing")
    if args.output_root.exists():
        fail(f"output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    scratch = args.output_root / "scratch"
    scratch.mkdir()

    canonical = scratch / "canonical.mlir"
    bind_command = [
        str(args.optimizer), f"--neura-architecture-spec={args.architecture}",
        str(FIXTURE), "--pass-pipeline=builtin.module(bind-source-iteration-domain)",
        "--verify-each", "-o", str(canonical),
    ]
    checked_run(args, "bind-canonical-source", bind_command,
                expect_success=True)
    profiles = scratch / "task-profiles.json"
    profile_output = scratch / "profile-output.mlir"
    run_profile_export(args, canonical, profiles, profile_output)
    scheduled = scratch / "scheduled-f45.mlir"
    make_f45_schedule(canonical, profiles, scheduled)

    positive = run_retimer(args, scheduled, canonical, profiles, "positive",
                           expect_success=True)
    validate_positive(positive)
    exercise_profile_tampers(args, scheduled, canonical, profiles)
    exercise_semantic_dag_tamper(args, scheduled, canonical, profiles)
    print("PASS common parent profile native retimer positive and six negative controls")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # Keep the fixture runner's failure one-line and actionable.
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
