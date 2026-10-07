#!/usr/bin/env python3
"""Check authenticated F45 diagnostic profile initialization on captured input.

The positive case consumes the exact canonical module, native profile cache,
and F45-scheduled MLIR produced by a real source-owned Ray profile run. The
negative cases alter only the diagnostic marker (or add it to a task with a
successful 1-CGRA profile); they never invent mapper timings or profile rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


MARKER = "amoeba.diagnostic_minimum_legal_profile_initialization"
PASS_NAME = "--retime-original-amoeba-fixed-decisions="


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def matching_delimiter(text: str, start: int, opening: str,
                       closing: str) -> int:
    if start >= len(text) or text[start] != opening:
        fail(f"expected opening {opening!r} delimiter")
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
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
    fail(f"unterminated {opening!r} delimiter")
    raise AssertionError("unreachable")


def task_attribute_dictionary(text: str, task_name: str) -> tuple[int, int]:
    # Generic operation properties identify the exact Taskflow operation; the
    # balanced region-list scan then locates its trailing attribute dictionary.
    pattern = re.compile(
        r'<\{[^{}]*\btask_name = "' + re.escape(task_name) +
        r'"[^{}]*\}>\s*\('
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        fail(f"expected one generic Taskflow operation for {task_name}, found {len(matches)}")
    region_open = matches[0].end() - 1
    region_close = matching_delimiter(text, region_open, "(", ")")
    cursor = region_close + 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        fail(f"Taskflow operation {task_name} has no generic attribute dictionary")
    return cursor, matching_delimiter(text, cursor, "{", "}")


def replace_marker(text: str, *, count: int, shape: str,
                   extra_field: str = "") -> str:
    pattern = re.compile(re.escape(MARKER) + r" = \{[^{}]*\}")
    replacement = (
        MARKER + " = {composed_cgra_count = " + str(count) +
        " : i32, composed_cgra_shape = \"" + shape + "\"" +
        (", " + extra_field if extra_field else "") + "}"
    )
    result, substitutions = pattern.subn(replacement, text)
    if substitutions != 1:
        fail(f"expected one captured {MARKER} marker, found {substitutions}")
    return result


def add_marker_to_task(text: str, task_name: str, count: int,
                       shape: str) -> str:
    start, end = task_attribute_dictionary(text, task_name)
    dictionary = text[start:end + 1]
    if MARKER in dictionary:
        fail(f"Taskflow operation {task_name} already has the diagnostic marker")
    addition = (
        MARKER + " = {composed_cgra_count = " + str(count) +
        " : i32, composed_cgra_shape = \"" + shape + "\"}"
    )
    separator = ", " if dictionary[1:-1].strip() else ""
    return text[:end] + separator + addition + text[end:]


def expected_initial_profile(task_record: dict) -> dict | None:
    candidates = []
    for index, row in enumerate(task_record.get("profiles", [])):
        if row.get("mapper_succeeded") is not True:
            continue
        candidates.append((row.get("composed_cgra_count"),
                           row.get("estimated_latency"), index, row))
    if any(row.get("composed_cgra_shape") == "1x1"
           for _, _, _, row in candidates):
        return None
    candidates = [entry for entry in candidates
                  if isinstance(entry[0], int) and entry[0] > 0 and
                  isinstance(entry[1], int) and entry[1] > 0]
    if not candidates:
        fail(f"task {task_record.get('task')} has no successful mapper profile")
    return min(candidates, key=lambda entry: entry[:3])[3]


def marker_task(profile: dict, scheduled_text: str) -> tuple[str, dict]:
    task_records = {task["task"]: task for task in profile["tasks"]}
    found = []
    for task_name in task_records:
        start, end = task_attribute_dictionary(scheduled_text, task_name)
        attrs = scheduled_text[start:end + 1]
        if MARKER in attrs:
            found.append((task_name, attrs))
    if len(found) != 1:
        fail(f"expected exactly one captured diagnostic marker, found {len(found)}")
    task_name, attrs = found[0]
    marker = re.search(
        re.escape(MARKER) +
        r' = \{composed_cgra_count = ([0-9]+) : i32, '
        r'composed_cgra_shape = "([^"]+)"\}', attrs)
    if not marker:
        fail("captured diagnostic marker does not have the expected exact field spelling")
    record = task_records[task_name]
    expected = expected_initial_profile(record)
    if expected is None:
        fail(f"captured marker task {task_name} has a successful 1x1 profile")
    if int(marker.group(1)) != expected["composed_cgra_count"] or \
       marker.group(2) != expected["composed_cgra_shape"]:
        fail(f"captured marker for {task_name} differs from the authenticated fallback")
    return task_name, expected


def checked_run(args: argparse.Namespace, label: str, scheduled: Path,
                *, expect_success: bool, expected_error: str | None = None) -> None:
    result_path = args.output_root / f"{label}.json"
    mlir_path = args.output_root / f"{label}.mlir"
    options = [
        ("function", args.function),
        ("common-parent-profile-file", args.profiles),
        ("diagnostic-ii-ceiling", "23"),
        ("common-canonical-module-file", args.canonical),
        ("original-f45-replica-scaling", "true"),
        ("reschedule-with-production-scheduler", "true"),
        ("output", result_path),
    ]
    if any(any(character.isspace() for character in str(value))
           for _, value in options):
        fail("pass input paths must not contain whitespace")
    pass_options = " ".join(f"{key}={value}" for key, value in options)
    command = [
        str(args.optimizer), str(scheduled), "--verify-each",
        f"--architecture-spec={args.architecture}",
        f"--joint-inter-task-network-spec={args.network}",
        PASS_NAME + pass_options, "--mlir-print-op-generic", "-o",
        str(mlir_path),
    ]
    actual = ["taskset", "--cpu-list", args.cpu_list, *command]
    completed = subprocess.run(actual, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    log_path = args.output_root / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout)
    write_json(args.output_root / "logs" / f"{label}.command.json",
               {"argv": actual, "exit_code": completed.returncode})
    if expect_success and completed.returncode != 0:
        fail(f"{label} failed with exit {completed.returncode}; see {log_path}")
    if not expect_success and completed.returncode == 0:
        fail(f"{label} unexpectedly succeeded; see {log_path}")
    if expect_success:
        if not result_path.is_file() or not mlir_path.is_file():
            fail(f"{label} did not publish both expected outputs")
        if expected_error:
            fail("internal checker misuse: success case has an expected error")
        if MARKER not in mlir_path.read_text():
            fail("positive output did not preserve the validated diagnostic marker")
    else:
        if result_path.exists():
            fail(f"{label} published a result despite rejecting its marker")
        if expected_error and expected_error not in completed.stdout:
            fail(f"{label} failed for the wrong reason; expected {expected_error!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--scheduled", type=Path, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu-list", default="12")
    args = parser.parse_args()
    for name in ("optimizer", "architecture", "network", "canonical",
                 "profiles", "scheduled"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_root = args.output_root.resolve()
    if not args.optimizer.is_file() or any(
            not getattr(args, name).is_file()
            for name in ("architecture", "network", "canonical", "profiles",
                         "scheduled")):
        fail("one or more required native inputs are missing")
    if args.output_root.exists():
        fail(f"output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)

    profile = read_json(args.profiles)
    scheduled_text = args.scheduled.read_text()
    positive_task, fallback = marker_task(profile, scheduled_text)
    if positive_task == "Task_13":
        rows = {row["composed_cgra_shape"]: row
                for row in profile["tasks"]
                if row["task"] == positive_task
                for row in row["profiles"]}
        if not (fallback["composed_cgra_shape"] == "1x4" and
                fallback["composed_cgra_count"] == 4 and
                rows["1x4"]["estimated_latency"] <
                rows["2x2"]["estimated_latency"]):
            fail("captured Ray Task_13 fallback fingerprint changed")

    checked_run(args, "positive-captured-ray", args.scheduled,
                expect_success=True)

    marker_match = re.search(
        re.escape(MARKER) +
        r' = \{composed_cgra_count = ([0-9]+) : i32, '
        r'composed_cgra_shape = "([^"]+)"\}', scheduled_text)
    if not marker_match:
        fail("could not parse the captured diagnostic marker")
    original_count, original_shape = int(marker_match.group(1)), marker_match.group(2)
    variants = [
        ("forged-shape", 4, "1x3", "deterministic minimum legal mapper profile"),
        ("forged-count", 3, original_shape,
         "deterministic minimum legal mapper profile"),
        ("forged-fallback", 4, "2x2",
         "deterministic minimum legal mapper profile"),
    ]
    for label, count, shape, expected_error in variants:
        path = args.output_root / "negative-controls" / f"{label}.mlir"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(replace_marker(scheduled_text, count=count, shape=shape))
        checked_run(args, label, path, expect_success=False,
                    expected_error=expected_error)

    extra_path = args.output_root / "negative-controls" / "extra-field.mlir"
    extra_path.write_text(replace_marker(
        scheduled_text, count=original_count, shape=original_shape,
        extra_field="unexpected = true"))
    checked_run(args, "extra-field", extra_path, expect_success=False,
                expected_error="exactly composed_cgra_count and composed_cgra_shape")

    unit_task = next((task for task in profile["tasks"]
                      if any(row.get("mapper_succeeded") is True and
                             row.get("composed_cgra_shape") == "1x1"
                             for row in task.get("profiles", []))), None)
    if unit_task is None:
        fail("captured native inventory has no successful unit-profile task for the negative control")
    unit_path = args.output_root / "negative-controls" / "unit-profile-supported.mlir"
    unit_path.write_text(add_marker_to_task(
        scheduled_text, unit_task["task"], 1, "1x1"))
    checked_run(args, "unit-profile-supported", unit_path,
                expect_success=False,
                expected_error="successful 1-CGRA profile")

    print("PASS captured Ray diagnostic-profile marker and five fail-closed controls")
    print(f"fallback task={positive_task} count={fallback['composed_cgra_count']} "
          f"shape={fallback['composed_cgra_shape']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
