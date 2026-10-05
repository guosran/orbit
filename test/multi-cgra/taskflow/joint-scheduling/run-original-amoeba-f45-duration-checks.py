#!/usr/bin/env python3
"""Check actual multi-replica F45 profile-duration binding with isolated tampering."""

import argparse
import copy
import filecmp
import json
from pathlib import Path
import re
import shutil
import subprocess

ADAPTER_OPTION = "--adapt-original-amoeba-profile-costs="
PROFILE_INFO_RE = re.compile(r"profile_info\s*=\s*\{([^{}]*)\}")
INTEGER_FIELD_RE = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(-?[0-9]+)\s*:\s*i[0-9]+\b")
TASK_NAME_RE = re.compile(r'\btask_name\s*=\s*"([^"]+)"')
ACTIVE_RE = re.compile(r"\bactive_replicas\s*=\s*([0-9]+)\s*:\s*i32\b")
SHAPE_RE = re.compile(r'\bcomposed_cgra_shape\s*=\s*"([0-9]+x[0-9]+)"')


def read_json(path):
    return json.loads(Path(path).read_text())


def adapter_options(argv):
    matches = [arg for arg in argv if arg.startswith(ADAPTER_OPTION)]
    if len(matches) != 1:
        raise ValueError("command must contain exactly one " + ADAPTER_OPTION)
    result = {}
    for item in matches[0][len(ADAPTER_OPTION):].split():
        if "=" not in item:
            raise ValueError("malformed adapter option: " + item)
        key, value = item.split("=", 1)
        if key in result:
            raise ValueError("duplicate adapter option: " + key)
        result[key] = value
    return result


def replace_adapter_options(argv, updates):
    result = list(argv)
    index = next(i for i, arg in enumerate(result) if arg.startswith(ADAPTER_OPTION))
    values = adapter_options(result)
    values.update({key: str(value) for key, value in updates.items()})
    encoded = " ".join(key + "=" + value for key, value in values.items())
    if any(any(character.isspace() for character in value) for value in values.values()):
        raise ValueError("adapter option values may not contain whitespace")
    result[index] = ADAPTER_OPTION + encoded
    return result


def prepare_argv(argv, updates, input_ir, output_ir, cpu):
    result = replace_adapter_options(argv, updates)
    wrapped = result[:2] == ["taskset", "--cpu-list"]
    input_index = 4 if wrapped else 1
    if len(result) <= input_index:
        raise ValueError("adapter command must name an input module")
    if "-o" not in result or result.index("-o") + 1 >= len(result):
        raise ValueError("adapter command must contain '-o <output.mlir>'")
    result[input_index] = str(input_ir)
    result[result.index("-o") + 1] = str(output_ir)
    if wrapped:
        result[2] = str(cpu)
    else:
        result = ["taskset", "--cpu-list", str(cpu)] + result
    return result


def parse_ir_task_profiles(ir_text):
    records = []
    for match in PROFILE_INFO_RE.finditer(ir_text):
        profile_start = match.start()
        previous_names = list(TASK_NAME_RE.finditer(ir_text, 0, profile_start))
        if not previous_names:
            continue
        task_name_match = previous_names[-1]
        task_name = task_name_match.group(1)
        fields = {key: int(value) for key, value in INTEGER_FIELD_RE.findall(match.group(1))}
        required = {
            "compiled_ii", "duration", "materialized_operation_count",
            "sample_trip_count", "steps",
        }
        if not required.issubset(fields):
            continue

        prior = ir_text[:profile_start]
        active_matches = list(ACTIVE_RE.finditer(prior))
        shape_matches = list(SHAPE_RE.finditer(prior))
        if not active_matches or not shape_matches:
            continue
        active = int(active_matches[-1].group(1))
        selected_shape = shape_matches[-1].group(1)

        schedule_start = prior.rfind("amoeba.task_scheduler_schedule_info = {")
        if schedule_start < 0:
            continue
        schedule_text = ir_text[schedule_start:profile_start]
        schedule_active_matches = list(ACTIVE_RE.finditer(schedule_text))
        schedule_duration_match = re.search(r"\bduration\s*=\s*([0-9]+)\s*:\s*i64\b", schedule_text)
        if not schedule_active_matches or not schedule_duration_match:
            continue

        records.append({
            "task": task_name,
            "profile_info_start": match.start(),
            "profile_info_end": match.end(),
            "profile_fields": fields,
            "active_replicas": active,
            "schedule_active_replicas": int(schedule_active_matches[0].group(1)),
            "shape": selected_shape,
            "schedule_duration": int(schedule_duration_match.group(1)),
        })
    return records


def integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def selected_profile(profiles, task_name, shape):
    task_rows = [row for row in profiles.get("tasks", []) if row.get("task") == task_name]
    if len(task_rows) != 1:
        raise ValueError("profile file must contain exactly one task row for " + task_name)
    selected = [row for row in task_rows[0].get("profiles", [])
                if row.get("composed_cgra_shape") == shape]
    if len(selected) != 1:
        raise ValueError("profile file must contain exactly one selected shape for " + task_name)
    if selected[0].get("mapper_succeeded") is not True:
        raise ValueError("selected profile did not explicitly report mapper success for " + task_name)
    return selected[0]


def inspect_parent_rows(ir_path, profile_path):
    ir_text = Path(ir_path).read_text()
    profiles = read_json(profile_path)
    records = parse_ir_task_profiles(ir_text)
    split = [record for record in records if record["active_replicas"] > 1]
    if not split:
        raise ValueError("caller-bound input contains no active multi-replica parent")

    validated = []
    for record in split:
        if record["active_replicas"] != record["schedule_active_replicas"]:
            raise ValueError(record["task"] + ": task and schedule active-replica counts disagree")
        profile = selected_profile(profiles, record["task"], record["shape"])
        rows, cols = (int(part) for part in record["shape"].split("x"))
        count = integer(profile.get("composed_cgra_count"), record["task"] + ".composed_cgra_count")
        if rows * cols != count:
            raise ValueError(record["task"] + ": selected shape and count disagree")
        ii = integer(profile.get("compiled_ii"), record["task"] + ".compiled_ii")
        trip = integer(profile.get("sample_trip_count"), record["task"] + ".sample_trip_count")
        steps = integer(profile.get("steps"), record["task"] + ".steps")
        operations = integer(profile.get("materialized_operation_count"), record["task"] + ".materialized_operation_count")
        latency = integer(profile.get("estimated_latency"), record["task"] + ".estimated_latency")
        fields = record["profile_fields"]
        expected_full = ii * (trip - 1) + steps
        expected_f45 = (expected_full + record["active_replicas"] - 1) // record["active_replicas"]
        if latency != expected_full:
            raise ValueError(record["task"] + ": selected profile full latency is not II*(trip-1)+steps")
        if (fields["compiled_ii"] != ii or fields["sample_trip_count"] != trip
                or fields["steps"] != steps
                or fields["materialized_operation_count"] != operations
                or fields["duration"] != expected_f45
                or record["schedule_duration"] != expected_f45):
            raise ValueError(record["task"] + ": source profile_info/scheduler/full-profile duration bindings disagree")
        validated.append({
            "task": record["task"], "shape": record["shape"], "active_replicas": record["active_replicas"],
            "selected_profile": profile, "profile_fields": fields,
            "full_parent_mapper_duration_cycles": expected_full,
            "f45_scheduler_duration_cycles": expected_f45,
        })
    return ir_text, profiles, validated


def copy_exact(source, target):
    shutil.copy2(source, target)
    if not filecmp.cmp(source, target, shallow=False):
        raise RuntimeError("copied input is not byte-identical: " + str(source))


def mutate_profile_info(ir_text, task_name, field, delta):
    records = parse_ir_task_profiles(ir_text)
    matches = [row for row in records if row["task"] == task_name]
    if len(matches) != 1:
        raise ValueError("caller-bound IR must contain exactly one profile_info for " + task_name)
    match = matches[0]
    block = ir_text[match["profile_info_start"]:match["profile_info_end"]]
    field_pattern = re.compile(r"(\b" + re.escape(field) + r"\s*=\s*)([0-9]+)(\s*:\s*i[0-9]+\b)")
    changed, count = field_pattern.subn(lambda m: m.group(1) + str(int(m.group(2)) + delta) + m.group(3), block, count=1)
    if count != 1:
        raise ValueError("could not uniquely mutate profile_info." + field)
    return ir_text[:match["profile_info_start"]] + changed + ir_text[match["profile_info_end"]:]


def mutate_outer_active_replicas(ir_text, task_name):
    records = parse_ir_task_profiles(ir_text)
    matches = [row for row in records if row["task"] == task_name]
    if len(matches) != 1:
        raise ValueError("caller-bound IR must contain exactly one profile_info for " + task_name)
    profile_start = matches[0]["profile_info_start"]
    prior = ir_text[:profile_start]
    names = list(TASK_NAME_RE.finditer(prior))
    target_name = [match for match in names if match.group(1) == task_name]
    if not target_name:
        raise ValueError("could not locate task schedule active-replica field")
    # The last active_replicas field before profile_info is the outer task field;
    # the schedule dictionary's duplicate appears earlier.
    active_matches = list(ACTIVE_RE.finditer(prior))
    if not active_matches:
        raise ValueError("could not locate active-replica field")
    match = active_matches[-1]
    old = int(match.group(1))
    if old < 1 or old >= 4:
        raise ValueError("test mutation needs an original active-replica count in [1,3]")
    replacement = str(old + 1)
    return ir_text[:match.start(1)] + replacement + ir_text[match.end(1):]


def mutate_profile_json(profiles, task_name, shape, field, delta):
    result = copy.deepcopy(profiles)
    task_rows = [row for row in result.get("tasks", []) if row.get("task") == task_name]
    if len(task_rows) != 1:
        raise ValueError("profile file must contain exactly one task row for " + task_name)
    profile_rows = [row for row in task_rows[0].get("profiles", [])
                    if row.get("composed_cgra_shape") == shape]
    if len(profile_rows) != 1:
        raise ValueError("profile file must contain exactly one selected shape for " + task_name)
    value = integer(profile_rows[0].get(field), task_name + "." + field)
    profile_rows[0][field] = value + delta
    return result


def validate_catalog(catalog_path, parent_rows):
    catalog = read_json(catalog_path)
    binding = catalog.get("original_amoeba_profile_binding")
    if not isinstance(binding, dict) or not isinstance(binding.get("tasks"), list):
        raise ValueError("positive adapter catalog omits original profile task binding")
    bound_rows = binding["tasks"]
    for parent in parent_rows:
        rows = [row for row in bound_rows if row.get("task") == parent["task"]]
        if len(rows) != 1:
            raise ValueError("positive catalog must bind exactly one full profile for " + parent["task"])
        row = rows[0]
        profile = parent["selected_profile"]
        expected = {
            "selected_cgra_shape": parent["shape"],
            "selected_cgra_count": profile["composed_cgra_count"],
            "compiled_ii": profile["compiled_ii"],
            "steps": profile["steps"],
            "sample_trip_count": profile["sample_trip_count"],
            "materialized_operation_count": profile["materialized_operation_count"],
            "estimated_latency": profile["estimated_latency"],
            "static_trip_count": profile["sample_trip_count"],
            "mapper_succeeded": True,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(parent["task"] + ": catalog full-profile binding differs at " + key)
        replicas = row.get("replicas")
        if not isinstance(replicas, list) or len(replicas) != parent["active_replicas"]:
            raise ValueError(parent["task"] + ": catalog replica inventory differs from source active count")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-command", type=Path, required=True,
                        help="JSON object containing the reproducible adapter argv array")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu", type=int, choices=range(12), default=5)
    parser.add_argument("--parent-task",
                        help="select which active multi-replica parent to tamper (default: first in IR order)")
    args = parser.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    command_document = read_json(args.adapter_command)
    original_argv = command_document.get("argv")
    if not isinstance(original_argv, list) or not original_argv or not all(isinstance(x, str) for x in original_argv):
        raise ValueError("adapter command JSON must contain a string argv array")
    if command_document.get("exit_code", 0) not in (0, None):
        raise ValueError("provided adapter command JSON records a nonzero exit code")
    wrapped = original_argv[:2] == ["taskset", "--cpu-list"]
    input_index = 4 if wrapped else 1
    if len(original_argv) <= input_index:
        raise ValueError("adapter command must name an input module")
    source_ir = Path(original_argv[input_index]).resolve()
    options = adapter_options(original_argv)
    if "profile-file" not in options or "output" not in options:
        raise ValueError("adapter command must name profile-file and output options")
    profile_path = Path(options["profile-file"]).resolve()
    ir_text, profiles, parent_rows = inspect_parent_rows(source_ir, profile_path)
    selected_task = args.parent_task or parent_rows[0]["task"]
    target_rows = [row for row in parent_rows if row["task"] == selected_task]
    if len(target_rows) != 1:
        raise ValueError("requested parent-task is not exactly one multi-replica source task")
    target_parent = target_rows[0]
    duration_facts = [
        {
            "task": parent["task"],
            "selected_shape": parent["shape"],
            "active_replicas": parent["active_replicas"],
            "compiled_ii": parent["selected_profile"]["compiled_ii"],
            "sample_trip_count": parent["selected_profile"]["sample_trip_count"],
            "steps": parent["selected_profile"]["steps"],
            "materialized_operation_count": parent["selected_profile"]["materialized_operation_count"],
            "full_parent_mapper_duration_cycles": parent["full_parent_mapper_duration_cycles"],
            "f45_scheduler_duration_cycles": parent["f45_scheduler_duration_cycles"],
        }
        for parent in parent_rows
    ]

    command_copy = root / "adapter-command.input.json"
    shutil.copy2(args.adapter_command, command_copy)
    if not filecmp.cmp(args.adapter_command, command_copy, shallow=False):
        raise RuntimeError("saved adapter command JSON is not byte-identical")
    cases = []

    def run_case(label, case_ir_text=None, case_profiles=None, positive=False, expected_error=None):
        directory = root / label
        directory.mkdir()
        ir_copy = directory / "caller-bound.mlir"
        profile_copy = directory / "task-profiles.json"
        copy_exact(source_ir, ir_copy)
        copy_exact(profile_path, profile_copy)
        if case_ir_text is not None:
            ir_copy.write_text(case_ir_text)
        if case_profiles is not None:
            profile_copy.write_text(json.dumps(case_profiles, indent=2) + "\n")
        catalog = directory / "cost-catalog.json"
        output_ir = directory / "adapter-output.mlir"
        argv = prepare_argv(original_argv, {
            "profile-file": profile_copy,
            "output": catalog,
        }, ir_copy, output_ir, args.cpu)
        (directory / "command.json").write_text(json.dumps({"argv": argv}, indent=2) + "\n")
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            completed = subprocess.run(argv, stdout=stdout, stderr=stderr)
        stderr_text = (directory / "stderr.log").read_text(errors="replace")
        if positive:
            passed = completed.returncode == 0 and catalog.is_file() and "error:" not in stderr_text
            if passed:
                validate_catalog(catalog, parent_rows)
        else:
            diagnostic_match = expected_error is None or expected_error in stderr_text
            passed = (completed.returncode != 0 and not catalog.exists()
                      and "error:" in stderr_text and diagnostic_match)
        case = {
            "case": label,
            "expected_success": positive,
            "exit_code": completed.returncode,
            "catalog_present": catalog.is_file(),
            "expected_error": expected_error,
            "diagnostic_match": positive or expected_error is None or expected_error in stderr_text,
            "pass": passed,
        }
        cases.append(case)
        (root / "acceptance.json").write_text(json.dumps({
            "status": "in_progress" if not all(item["pass"] for item in cases) else "pass_so_far",
            "cpu": args.cpu,
            "parent_task_tampered": selected_task,
            "validated_multi_parent_count": len(parent_rows),
            "duration_facts": duration_facts,
            "cases": cases,
        }, indent=2) + "\n")
        if not passed:
            raise RuntimeError(label + " did not meet the expected outcome; see " + str(directory / "stderr.log"))

    run_case("positive", positive=True)

    mutations = [
        ("reject-forged-f45-scheduler-duration",
         lambda: mutate_profile_info(ir_text, selected_task, "duration", 1), None,
         "exact replica-duration contract"),
        ("reject-forged-profile-ii",
         lambda: mutate_profile_info(ir_text, selected_task, "compiled_ii", 1), None,
         "exact replica-duration contract"),
        ("reject-forged-profile-trip-count",
         lambda: mutate_profile_info(ir_text, selected_task, "sample_trip_count", 1), None,
         "exact replica-duration contract"),
        ("reject-forged-profile-steps",
         lambda: mutate_profile_info(ir_text, selected_task, "steps", 1), None,
         "exact replica-duration contract"),
        ("reject-forged-profile-operation-count",
         lambda: mutate_profile_info(ir_text, selected_task, "materialized_operation_count", 1), None,
         "exact replica-duration contract"),
        ("reject-forged-full-profile-latency", None,
         lambda: mutate_profile_json(profiles, selected_task, target_parent["shape"], "estimated_latency", 1),
         "profiled latency are inconsistent"),
        ("reject-forged-active-replica-count",
         lambda: mutate_outer_active_replicas(ir_text, selected_task), None,
         "source-domain control binding is stale or forged"),
    ]
    for label, mutate_ir, mutate_profiles, expected_error in mutations:
        changed_ir = mutate_ir() if mutate_ir is not None else None
        changed_profiles = mutate_profiles() if mutate_profiles is not None else None
        run_case(label, changed_ir, changed_profiles, expected_error=expected_error)

    (root / "acceptance.json").write_text(json.dumps({
        "status": "pass",
        "cpu": args.cpu,
        "parent_task_tampered": selected_task,
        "validated_multi_parent_count": len(parent_rows),
        "duration_facts": duration_facts,
        "cases": cases,
    }, indent=2) + "\n")
    print(json.dumps({"status": "pass", "case_count": len(cases), "output_root": str(root)}))


if __name__ == "__main__":
    main()
