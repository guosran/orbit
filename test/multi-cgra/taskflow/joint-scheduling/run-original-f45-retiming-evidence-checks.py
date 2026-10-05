#!/usr/bin/env python3
"""Check the original F45 replica-scaling retimer and reject forged evidence."""

import argparse
import copy
import filecmp
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

PASS_PREFIX = "--retime-original-amoeba-fixed-decisions="
SCHEDULE_PREFIX = "amoeba.task_scheduler_schedule_info"
PROFILE_INFO_RE = re.compile(r"profile_info\s*=\s*\{([^{}]*)\}")
TASK_NAME_RE = re.compile(r'\btask_name\s*=\s*"([^\"]+)"')
EXPECTED_TIMING_POLICY = {
    "schema": "amoeba-original-f45-replica-scaling-v1",
    "duration_formula": (
        "ceil(ceil(catalog_startup_cycles + compiled_ii * "
        "(source_macro_firings - 1)) / original_active_replicas)"
    ),
    "replica_duration_rule": (
        "ceil-full-parent-mapped-duration-over-original-active-replicas-v1"
    ),
    "child_mapper_profiles_used": False,
    "status": "original-f45-scheduler-estimate",
}
EXPECTED_DURATION_BINDING_KEYS = {
    "schema", "full_parent_mapper_duration_cycles",
    "f45_scheduler_duration_cycles", "active_replicas", "compiled_ii",
    "sample_trip_count", "steps", "materialized_operation_count",
    "rounding_rule", "status",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def pass_options(argv):
    matches = [(index, arg) for index, arg in enumerate(argv)
               if arg.startswith(PASS_PREFIX)]
    if len(matches) != 1:
        raise ValueError("retimer command must contain exactly one " + PASS_PREFIX)
    index, argument = matches[0]
    values = {}
    for item in argument[len(PASS_PREFIX):].split():
        if "=" not in item:
            raise ValueError("retimer pass option is not key=value: " + item)
        key, value = item.split("=", 1)
        if not key or key in values:
            raise ValueError("retimer pass options contain an empty or duplicate key")
        values[key] = value
    return index, values


def build_command(argv, updates, input_module, result_path, output_module, cpu):
    result = list(argv)
    pass_index, values = pass_options(result)
    for key, value in updates.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = str(value)
    values["output"] = str(result_path)
    if any(any(character.isspace() for character in value)
           for value in values.values()):
        raise ValueError("retimer pass option values may not contain whitespace")
    result[pass_index] = PASS_PREFIX + " ".join(
        key + "=" + value for key, value in values.items())

    wrapped = result[:2] == ["taskset", "--cpu-list"]
    input_index = 4 if wrapped else 1
    if len(result) <= input_index:
        raise ValueError("retimer command must name an input module")
    result[input_index] = str(input_module)
    output_indices = [index for index, value in enumerate(result[:-1])
                      if value == "-o"]
    if len(output_indices) != 1:
        raise ValueError("retimer command must contain exactly one '-o <path>'")
    result[output_indices[0] + 1] = str(output_module)
    if wrapped:
        if len(result) < 4:
            raise ValueError("malformed taskset retimer command")
        result[2] = str(cpu)
    else:
        result = ["taskset", "--cpu-list", str(cpu)] + result
    return result


def matching_delimiter(text, start):
    pairs = {"{": "}", "[": "]", "(": ")"}
    if start >= len(text) or text[start] not in pairs:
        raise ValueError("expected an opening delimiter")
    stack = []
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
            continue
        if character == '"':
            quoted = True
        elif character in pairs:
            stack.append(pairs[character])
        elif character in "}])":
            if not stack or character != stack.pop():
                raise ValueError("unbalanced delimiter in MLIR attribute")
            if not stack:
                return index + 1
    raise ValueError("unterminated MLIR attribute")


def split_top_level(text, base_offset=0):
    results = []
    start = 0
    stack = []
    pairs = {"{": "}", "[": "]", "(": ")"}
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in pairs:
            stack.append(pairs[character])
        elif character in "}])":
            if not stack or character != stack.pop():
                raise ValueError("unbalanced nested value in MLIR dictionary")
        elif character == "," and not stack:
            results.append((text[start:index], base_offset + start,
                            base_offset + index))
            start = index + 1
    if stack or quoted:
        raise ValueError("unterminated nested value in MLIR dictionary")
    if text[start:].strip():
        results.append((text[start:], base_offset + start,
                        base_offset + len(text)))
    return results


def parse_dictionary(raw):
    if not raw.startswith("{") or not raw.endswith("}"):
        raise ValueError("expected MLIR dictionary")
    fields = {}
    for item, _, _ in split_top_level(raw[1:-1]):
        if not item.strip():
            continue
        match = re.match(r"\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*?)\s*$", item,
                         flags=re.DOTALL)
        if not match:
            raise ValueError("malformed MLIR dictionary field: " + item[:80])
        key, value = match.groups()
        if key in fields:
            raise ValueError("duplicate MLIR dictionary field: " + key)
        fields[key] = value
    return fields


def parse_integer(value, label):
    match = re.fullmatch(r"\s*(-?[0-9]+)\s*:\s*i[0-9]+\s*", value)
    if not match:
        raise ValueError(label + " is not a typed integer")
    return int(match.group(1))


def parse_string(value, label):
    match = re.fullmatch(r'\s*"((?:\\.|[^"\\])*)"\s*', value)
    if not match:
        raise ValueError(label + " is not a quoted string")
    return bytes(match.group(1), "utf-8").decode("unicode_escape")


def named_container(text, key, opening):
    match = re.search(r"\b" + re.escape(key) + r"\s*=\s*" +
                      re.escape(opening), text)
    if not match:
        raise ValueError("missing MLIR " + key + " inventory")
    start = match.end() - 1
    return start, matching_delimiter(text, start)


def parse_array_dictionaries(text, key):
    start, end = named_container(text, key, "[")
    content_start, content_end = start + 1, end - 1
    entries = []
    for raw, local_start, local_end in split_top_level(
            text[content_start:content_end], content_start):
        if not raw.strip():
            continue
        stripped = raw.strip()
        leading = len(raw) - len(raw.lstrip())
        absolute_start = local_start + leading
        absolute_end = absolute_start + len(stripped)
        entries.append({
            "fields": parse_dictionary(stripped),
            "start": absolute_start,
            "end": absolute_end,
            "raw": stripped,
        })
    return entries, start, end


def field_int(fields, key, label):
    if key not in fields:
        raise ValueError(label + " is missing " + key)
    return parse_integer(fields[key], label + "." + key)


def field_string(fields, key, label):
    if key not in fields:
        raise ValueError(label + " is missing " + key)
    return parse_string(fields[key], label + "." + key)


def extract_source_tasks(module_text):
    records = []
    for profile_match in PROFILE_INFO_RE.finditer(module_text):
        profile_fields = parse_dictionary("{" + profile_match.group(1) + "}")
        required_profile_fields = {
            "compiled_ii", "duration", "materialized_operation_count",
            "sample_trip_count", "steps",
        }
        if not required_profile_fields.issubset(profile_fields):
            continue
        preceding = module_text[:profile_match.start()]
        schedule_assignments = list(re.finditer(
            re.escape(SCHEDULE_PREFIX) + r"\s*=\s*\{", preceding))
        if not schedule_assignments:
            raise ValueError("profile_info has no preceding original task schedule")
        assignment = schedule_assignments[-1]
        dictionary_start = assignment.end() - 1
        dictionary_end = matching_delimiter(module_text, dictionary_start)
        if dictionary_end > profile_match.start():
            raise ValueError("original schedule dictionary overlaps profile_info")
        schedule_text = module_text[dictionary_start:dictionary_end]
        schedule_fields = parse_dictionary(schedule_text)
        task_name = field_string(schedule_fields, "task_name", "schedule")
        active = field_int(schedule_fields, "active_replicas", task_name)
        selected_shape = field_string(schedule_fields, "composed_cgra_shape", task_name)
        selected_count = field_int(schedule_fields, "composed_cgra_count", task_name)
        dispatch_index = field_int(schedule_fields, "dispatch_index", task_name)
        scheduler_start = field_int(schedule_fields, "scheduler_start_time", task_name)
        scheduler_end = field_int(schedule_fields, "scheduler_end_time", task_name)
        scheduler_duration = field_int(schedule_fields, "duration", task_name)
        duration_unit = field_string(schedule_fields, "duration_unit", task_name)
        placements, placement_array_start, placement_array_end = parse_array_dictionaries(
            schedule_text, "placements")
        replica_shapes, shape_array_start, shape_array_end = parse_array_dictionaries(
            schedule_text, "replica_shapes")

        outer_text = module_text[dictionary_end:profile_match.start()]
        outer_shapes = list(re.finditer(
            r'\bcomposed_cgra_shape\s*=\s*"([0-9]+x[0-9]+)"', outer_text))
        if not outer_shapes:
            raise ValueError(task_name + ": outer task selected-shape attr is missing")
        outer_shape = outer_shapes[0].group(1)
        if outer_shape != selected_shape:
            raise ValueError(task_name + ": task and schedule selected shapes disagree")

        profile = {key: field_int(profile_fields, key, task_name + ".profile_info")
                   for key in required_profile_fields}
        parsed_placements = []
        for index, entry in enumerate(placements):
            fields = entry["fields"]
            parsed_placements.append({
                "row": field_int(fields, "row", task_name + ".placements[" + str(index) + "]"),
                "col": field_int(fields, "col", task_name + ".placements[" + str(index) + "]"),
                "context_id": field_int(fields, "context_id", task_name + ".placements[" + str(index) + "]"),
                "replica_id": field_int(fields, "replica_id", task_name + ".placements[" + str(index) + "]"),
                "scheduler_start_time": field_int(fields, "scheduler_start_time", task_name + ".placements[" + str(index) + "]"),
                "scheduler_end_time": field_int(fields, "scheduler_end_time", task_name + ".placements[" + str(index) + "]"),
                "scheduler_duration": field_int(fields, "scheduler_duration", task_name + ".placements[" + str(index) + "]"),
            })
        parsed_shapes = []
        for index, entry in enumerate(replica_shapes):
            fields = entry["fields"]
            label = task_name + ".replica_shapes[" + str(index) + "]"
            parsed_shapes.append({
                "replica_id": field_int(fields, "replica_id", label),
                "shape": field_string(fields, "shape", label),
                "cgra_count": field_int(fields, "cgra_count", label),
                "row": field_int(fields, "row", label),
                "col": field_int(fields, "col", label),
                "rows": field_int(fields, "placement_rows", label),
                "cols": field_int(fields, "placement_cols", label),
            })
        records.append({
            "task": task_name,
            "selected_shape": selected_shape,
            "selected_count": selected_count,
            "active_replicas": active,
            "dispatch_index": dispatch_index,
            "scheduler_start": scheduler_start,
            "scheduler_end": scheduler_end,
            "scheduler_duration": scheduler_duration,
            "duration_unit": duration_unit,
            "profile": profile,
            "placements": parsed_placements,
            "replica_shapes": parsed_shapes,
            "schedule_dict_start": dictionary_start,
            "schedule_dict_end": dictionary_end,
            "placement_array_start": placement_array_start + dictionary_start,
            "placement_array_end": placement_array_end + dictionary_start,
            "shape_array_start": shape_array_start + dictionary_start,
            "shape_array_end": shape_array_end + dictionary_start,
            "placement_entries": [
                {"start": item["start"] + dictionary_start,
                 "end": item["end"] + dictionary_start}
                for item in placements
            ],
            "shape_entries": [
                {"start": item["start"] + dictionary_start,
                 "end": item["end"] + dictionary_start}
                for item in replica_shapes
            ],
        })
    names = [record["task"] for record in records]
    if not records or len(set(names)) != len(names):
        raise ValueError("caller-bound module does not have a unique full task profile inventory")
    return records


def parse_dispatch_order(module_text, records):
    pattern = "amoeba.task_scheduler_dispatch_order"
    start = module_text.rfind(pattern)
    if start < 0:
        raise ValueError("original module omits task_scheduler_dispatch_order")
    array_start = module_text.find("[", start + len(pattern))
    if array_start < 0:
        raise ValueError("original task dispatch order is malformed")
    array_end = matching_delimiter(module_text, array_start)
    names = re.findall(r'"([^"\\]+)"', module_text[array_start:array_end])
    if len(names) != len(records) or set(names) != {record["task"] for record in records}:
        raise ValueError("original dispatch order does not cover exactly the caller-bound tasks")
    return names


def require_int(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(label + " must be an integer >= " + str(minimum))
    return value


def source_replica_geometry(record):
    active = record["active_replicas"]
    shapes = record["replica_shapes"]
    placements = record["placements"]
    if active < 1 or len(shapes) != active or not placements:
        raise ValueError(record["task"] + ": original replica inventory is incomplete")
    shapes_by_id = {shape["replica_id"]: shape for shape in shapes}
    if len(shapes_by_id) != active or set(shapes_by_id) != set(range(active)):
        raise ValueError(record["task"] + ": original replica IDs are not contiguous")
    replicas = []
    all_cells = []
    for replica_id in range(active):
        shape = shapes_by_id[replica_id]
        cells = []
        for placed in placements:
            if placed["replica_id"] == replica_id:
                cell = {key: placed[key] for key in ("row", "col", "replica_id", "context_id")}
                cells.append(cell)
                all_cells.append(cell)
        if len(cells) != shape["cgra_count"] or shape["cgra_count"] != shape["rows"] * shape["cols"]:
            raise ValueError(record["task"] + ": original replica shape omits exact placement cells")
        replicas.append({
            "replica_id": replica_id,
            "shape": shape["shape"],
            "cgra_count": shape["cgra_count"],
            "row": shape["row"],
            "col": shape["col"],
            "rows": shape["rows"],
            "cols": shape["cols"],
            "actual_cells": cells,
            "context_ids": [cell["context_id"] for cell in cells],
        })
    if len(all_cells) != len(placements):
        raise ValueError(record["task"] + ": placement references an unknown replica ID")
    return replicas, all_cells


def copy_exact(source, target):
    shutil.copy2(source, target)
    if not filecmp.cmp(source, target, shallow=False):
        raise RuntimeError("isolated input copy differs byte-for-byte: " + str(source))


def named_task_rows(document, task_name):
    rows = document.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("catalog original profile binding omits task rows")
    matches = [row for row in rows if isinstance(row, dict) and row.get("task") == task_name]
    if len(matches) != 1:
        raise ValueError("catalog must contain exactly one bound task row for " + task_name)
    return matches[0]


def validate_positive(result_path, source_records, dispatch_order, catalog, body_export):
    result = read_json(result_path)
    if (result.get("schema") != "amoeba-original-fixed-decision-retiming-v1"
            or result.get("record_type") != "result" or result.get("valid") is not True
            or result.get("diagnostic_only") is not True or result.get("formal_go") is not False):
        raise ValueError("positive result is not a valid non-formal diagnostic retiming")
    if result.get("body_equivalence_checked") is not True or result.get("selected_profile_body_binding_verified") is not True:
        raise ValueError("positive result does not certify original body/profile binding")
    trace = result.get("fixed_decision_trace")
    if not isinstance(trace, dict) or trace.get("schema") != "amoeba-fixed-decision-trace-v1":
        raise ValueError("positive result omits the source-owned fixed decision trace")
    policy = result.get("replica_timing_policy")
    if policy != EXPECTED_TIMING_POLICY or trace.get("replica_timing_policy") != EXPECTED_TIMING_POLICY:
        raise ValueError("positive result omits the exact original-F45 timing policy metadata")
    if trace.get("formal_go") is not False or trace.get("body_equivalence_checked") is not True:
        raise ValueError("fixed decision trace overstates formal or body proof status")
    if result.get("dispatch_order") != dispatch_order or trace.get("dispatch_order") != dispatch_order:
        raise ValueError("retimer changed original task dispatch order")
    if "original_parent_dispatch_order" in result and result["original_parent_dispatch_order"] != dispatch_order:
        raise ValueError("retimer changed separately emitted original parent dispatch order")

    catalog_binding = catalog.get("original_amoeba_profile_binding")
    if not isinstance(catalog_binding, dict) or not isinstance(catalog_binding.get("tasks"), list):
        raise ValueError("positive parent catalog omits original profile task bindings")
    body_rows = body_export.get("tasks")
    if not isinstance(body_rows, list):
        raise ValueError("positive body export omits task rows")
    body_by_task = {row.get("task"): row for row in body_rows if isinstance(row, dict)}
    catalog_by_task = {row.get("task"): row for row in catalog_binding["tasks"]
                       if isinstance(row, dict)}
    original_rows = result.get("original_decisions")
    cost_rows = result.get("task_costs")
    trace_bindings = trace.get("task_profile_bindings")
    root_schedules = result.get("task_schedule")
    trace_schedules = trace.get("task_schedule")
    for rows, label in ((original_rows, "original decisions"), (cost_rows, "task costs"),
                        (trace_bindings, "task profile bindings"),
                        (root_schedules, "retimed schedules"),
                        (trace_schedules, "fixed trace schedules")):
        if not isinstance(rows, list):
            raise ValueError("positive result omits " + label)
    source_by_task = {record["task"]: record for record in source_records}
    if (len(source_by_task) != len(source_records)
            or {row.get("task") for row in original_rows if isinstance(row, dict)} != set(source_by_task)
            or {row.get("task") for row in cost_rows if isinstance(row, dict)} != set(source_by_task)
            or {row.get("task") for row in trace_bindings if isinstance(row, dict)} != set(source_by_task)):
        raise ValueError("positive result does not cover exactly the original task inventory")

    originals = {row["task"]: row for row in original_rows if isinstance(row, dict)}
    costs = {row["task"]: row for row in cost_rows if isinstance(row, dict)}
    bindings = {row["task"]: row for row in trace_bindings if isinstance(row, dict)}
    root_schedule_map = {row["task"]: row for row in root_schedules if isinstance(row, dict)}
    trace_schedule_map = {row["task"]: row for row in trace_schedules if isinstance(row, dict)}
    duration_facts = []
    for task_name, source in source_by_task.items():
        original = originals[task_name]
        cost = costs[task_name]
        binding = bindings[task_name]
        root_schedule = root_schedule_map.get(task_name)
        trace_schedule = trace_schedule_map.get(task_name)
        if root_schedule is None or trace_schedule is None:
            raise ValueError(task_name + ": positive schedule output omits an original task")
        source_replicas, source_cells = source_replica_geometry(source)
        profile = source["profile"]
        full_profile_duration = (profile["compiled_ii"] *
                                 (profile["sample_trip_count"] - 1) + profile["steps"])
        expected_profile_duration = (
            (full_profile_duration + source["active_replicas"] - 1) // source["active_replicas"]
        )
        if (profile["duration"] != expected_profile_duration
                or source["scheduler_duration"] != profile["duration"]
                or source["duration_unit"] != "profiled-cycles"):
            raise ValueError(task_name + ": input F45 profile/scheduler duration is inconsistent")

        catalog_task = catalog_by_task.get(task_name)
        body = body_by_task.get(task_name)
        if catalog_task is None or body is None:
            raise ValueError(task_name + ": positive catalog/body exports omit an original task")
        profile_pairs = {
            "selected_cgra_shape": source["selected_shape"],
            "selected_cgra_count": source["selected_count"],
            "compiled_ii": profile["compiled_ii"],
            "steps": profile["steps"],
            "sample_trip_count": profile["sample_trip_count"],
            "materialized_operation_count": profile["materialized_operation_count"],
            "estimated_latency": full_profile_duration,
            "static_trip_count": profile["sample_trip_count"],
            "mapper_succeeded": True,
        }
        for key, expected in profile_pairs.items():
            if catalog_task.get(key) != expected:
                raise ValueError(task_name + ": original catalog profile binding differs at " + key)
        for key in ("task_signature", "counter_signature", "kernel_binding_signature",
                    "normalized_mapper_body"):
            if not isinstance(body.get(key), str) or catalog_task.get(key) != body.get(key):
                raise ValueError(task_name + ": cost catalog/body export differs at " + key)
            if binding.get(key) != body.get(key):
                raise ValueError(task_name + ": retimer output/body export differs at " + key)

        expected_actual_cells = source_cells
        expected_context_ids = [cell["context_id"] for cell in source_cells]
        expected_primary = source_replicas[0]
        expected_decision = {
            "selected_profile_shape": source["selected_shape"],
            "actual_placed_shape": expected_primary["shape"],
            "actual_placed_row": expected_primary["row"],
            "actual_placed_col": expected_primary["col"],
            "active_replicas": source["active_replicas"],
            "selected_cgra_count": source["selected_count"],
            "dispatch_index": source["dispatch_index"],
            "actual_cells": expected_actual_cells,
            "context_ids": expected_context_ids,
            "original_scheduler_start_internal": source["scheduler_start"],
            "original_scheduler_end_internal": source["scheduler_end"],
            "original_scheduler_duration_internal": source["scheduler_duration"],
            "original_profile_duration_cycles": profile["duration"],
        }
        if source["active_replicas"] > 1:
            expected_decision["replicas"] = source_replicas
            expected_decision["original_profile_duration_binding"] = {
                "schema": "amoeba-original-f45-profile-duration-binding-v1",
                "full_parent_mapper_duration_cycles": full_profile_duration,
                "f45_scheduler_duration_cycles": profile["duration"],
                "active_replicas": source["active_replicas"],
                "compiled_ii": profile["compiled_ii"],
                "sample_trip_count": profile["sample_trip_count"],
                "steps": profile["steps"],
                "materialized_operation_count": profile["materialized_operation_count"],
                "rounding_rule": "ceil-full-parent-duration-over-original-active-replicas-v1",
                "status": "verified",
            }
            certificate = original.get("original_profile_duration_binding")
            if not isinstance(certificate, dict) or set(certificate) != EXPECTED_DURATION_BINDING_KEYS:
                raise ValueError(task_name + ": split original duration certificate has the wrong exact 10-key schema")
        if original != {**expected_decision, "task": task_name}:
            raise ValueError(task_name + ": retimer changed or omitted original decision/replica evidence")

        for index, (actual, expected) in enumerate(zip(original.get("replicas", []), source_replicas)):
            if actual != expected:
                raise ValueError(task_name + ": original replica geometry/order differs at replica " + str(index))

        expected_cost_values = {
            "profile_compiled_ii": profile["compiled_ii"],
            "profile_steps": profile["steps"],
            "profile_sample_trip_count": profile["sample_trip_count"],
            "profile_materialized_operation_count": profile["materialized_operation_count"],
            "original_profile_duration_cycles": profile["duration"],
            "replica_duration_rule": EXPECTED_TIMING_POLICY["replica_duration_rule"],
            "original_active_replicas": source["active_replicas"],
        }
        if cost.get("trip_count") != profile["sample_trip_count"]:
            raise ValueError(task_name + ": task cost does not preserve the full profile trip count")
        for key, expected in expected_cost_values.items():
            if cost.get(key) != expected:
                raise ValueError(task_name + ": task cost profile binding differs at " + key)
        expected_trace_values = {
            "compiled_ii": profile["compiled_ii"],
            "steps": profile["steps"],
            "sample_trip_count": profile["sample_trip_count"],
            "materialized_operation_count": profile["materialized_operation_count"],
            "replica_duration_rule": EXPECTED_TIMING_POLICY["replica_duration_rule"],
            "original_active_replicas": source["active_replicas"],
        }
        for key, expected in expected_trace_values.items():
            if binding.get(key) != expected:
                raise ValueError(task_name + ": fixed trace profile binding differs at " + key)
        startup = cost.get("catalog_startup_cycles")
        if type(startup) not in (int, float) or not math.isfinite(startup) or startup <= 0:
            raise ValueError(task_name + ": selected direct-model startup is invalid")
        full_mapped_duration = math.ceil(startup + profile["compiled_ii"] *
                                         (profile["sample_trip_count"] - 1))
        expected_duration = ((full_mapped_duration + source["active_replicas"] - 1)
                             // source["active_replicas"])
        for row, label in ((cost, "cost"), (binding, "trace binding")):
            if (row.get("full_parent_mapped_duration_cycles") != full_mapped_duration
                    or row.get("mapped_duration_cycles") != expected_duration):
                raise ValueError(task_name + ": " + label + " does not implement the explicit F45 duration formula")
        if (binding.get("profile_duration_cycles") != profile["duration"]
                or binding.get("compiled_ii") != profile["compiled_ii"]
                or binding.get("steps") != profile["steps"]
                or binding.get("sample_trip_count") != profile["sample_trip_count"]
                or binding.get("materialized_operation_count") != profile["materialized_operation_count"]
                or binding.get("catalog_startup_cycles") != startup):
            raise ValueError(task_name + ": fixed trace profile binding differs from selected full profile")

        for schedule, label in ((root_schedule, "result schedule"),
                                (trace_schedule, "fixed trace schedule")):
            if (schedule.get("duration_cycles") != expected_duration
                    or schedule.get("occupied_cells") != expected_actual_cells):
                raise ValueError(task_name + ": " + label + " changed original occupancy or mapped duration")
            if (schedule.get("start_cycle") < 0
                    or schedule.get("end_cycle") - schedule.get("start_cycle") != expected_duration):
                raise ValueError(task_name + ": " + label + " has an inconsistent retimed interval")

        duration_facts.append({
            "task": task_name,
            "active_replicas": source["active_replicas"],
            "compiled_ii": profile["compiled_ii"],
            "trip_count": profile["sample_trip_count"],
            "steps": profile["steps"],
            "materialized_operation_count": profile["materialized_operation_count"],
            "original_f45_profile_duration_cycles": profile["duration"],
            "full_parent_mapped_duration_cycles": full_mapped_duration,
            "mapped_duration_cycles": expected_duration,
        })
    return result, duration_facts


def mutate_schedule_field(module_text, record, inventory, entry_index, key, new_value,
                          quoted=False):
    entries = record[inventory + "_entries"]
    if entry_index < 0 or entry_index >= len(entries):
        raise ValueError("requested source scheduler inventory entry is missing")
    entry = entries[entry_index]
    raw = module_text[entry["start"]:entry["end"]]
    pattern = re.compile(r"(\b" + re.escape(key) + r"\s*=\s*)"
                         + (r'"([^"\\]*)"' if quoted else r"(-?[0-9]+\s*:\s*i[0-9]+)"))
    match = pattern.search(raw)
    if not match:
        raise ValueError("could not locate scheduler inventory field " + key)
    if quoted:
        start, end = match.span(2)
        replacement = str(new_value)
    else:
        start, end = match.span(2)
        replacement = str(new_value) + " : i32"
    changed_entry = raw[:start] + replacement + raw[end:]
    return module_text[:entry["start"]] + changed_entry + module_text[entry["end"]:]


def mutate_catalog(catalog, task_name, field, mode):
    changed = copy.deepcopy(catalog)
    binding = changed.get("original_amoeba_profile_binding")
    if not isinstance(binding, dict):
        raise ValueError("catalog has no original AMOEBA profile binding")
    row = named_task_rows(binding, task_name)
    if field == "canonical_module_witness":
        if mode == "missing":
            binding.pop(field, None)
        else:
            binding[field] = "forged-canonical-module-witness"
    elif field == "normalized_mapper_body":
        row[field] = "forged-selected-normalized-mapper-body"
    else:
        value = require_int(row.get(field), task_name + "." + field)
        row[field] = value + 1
    return changed


def exact_input_manifest(command_document, argv, source_module, cost_path, body_path):
    if not isinstance(command_document, dict) or not isinstance(argv, list):
        raise ValueError("retimer command JSON must contain an argv array")
    if not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("retimer argv must contain only strings")
    pass_index, options = pass_options(argv)
    if options.get("original-f45-replica-scaling") != "true":
        raise ValueError("positive retimer command must explicitly set original-f45-replica-scaling=true")
    if options.get("replica-profile-evidence-file"):
        raise ValueError("F45 estimate mode must not consume materialized-child profile evidence")
    for key in ("function", "parent-cost-file", "profile-body-export-file", "output"):
        if not options.get(key):
            raise ValueError("retimer command omits required pass option " + key)
    if not Path(argv[3] if argv[:2] == ["taskset", "--cpu-list"] else argv[0]).is_file():
        raise ValueError("retimer binary from command JSON does not exist")
    if not source_module.is_file() or not cost_path.is_file() or not body_path.is_file():
        raise ValueError("retimer command input module, catalog, or body export is missing")
    output_tokens = [index for index, value in enumerate(argv[:-1]) if value == "-o"]
    if len(output_tokens) != 1:
        raise ValueError("retimer command must have exactly one '-o <path>'")
    return pass_index, options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retimer-command", type=Path, required=True,
                        help="JSON object containing the actual retimer argv array")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu", type=int, choices=range(12), default=5)
    parser.add_argument("--parent-task",
                        help="choose the split parent for catalog/input tampering; default is first in source order")
    args = parser.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    command_document = read_json(args.retimer_command)
    argv = command_document.get("argv")
    wrapped = isinstance(argv, list) and argv[:2] == ["taskset", "--cpu-list"]
    input_index = 4 if wrapped else 1
    if not isinstance(argv, list) or len(argv) <= input_index:
        raise ValueError("retimer command JSON does not name an input module")
    source_module = Path(argv[input_index]).resolve()
    _, options = pass_options(argv)
    cost_path = Path(options.get("parent-cost-file", "")).resolve()
    body_path = Path(options.get("profile-body-export-file", "")).resolve()
    _, options = exact_input_manifest(command_document, argv, source_module,
                                      cost_path, body_path)
    function = options["function"]
    source_text = source_module.read_text()
    source_records = extract_source_tasks(source_text)
    dispatch_order = parse_dispatch_order(source_text, source_records)
    cost_catalog = read_json(cost_path)
    body_export = read_json(body_path)
    split_records = [record for record in source_records if record["active_replicas"] > 1]
    if not split_records:
        raise ValueError("retimer input has no original multi-replica parent")
    if args.parent_task:
        target = [record for record in split_records if record["task"] == args.parent_task]
        if len(target) != 1:
            raise ValueError("parent-task is not exactly one split task in the input module")
        target_parent = target[0]
    else:
        target_parent = split_records[0]

    binding = cost_catalog.get("original_amoeba_profile_binding")
    if not isinstance(binding, dict) or not isinstance(binding.get("canonical_module_witness"), str):
        raise ValueError("positive catalog must carry its canonical source module witness")
    catalog_witness = binding["canonical_module_witness"]
    if not catalog_witness:
        raise ValueError("positive catalog canonical source module witness is empty")
    parent_binding = named_task_rows(binding, target_parent["task"])
    for key in ("compiled_ii", "steps", "sample_trip_count", "materialized_operation_count",
                "estimated_latency", "normalized_mapper_body"):
        if key not in parent_binding:
            raise ValueError("split parent catalog binding omits " + key)

    saved_command = root / "retimer-command.input.json"
    copy_exact(args.retimer_command, saved_command)
    reports = []
    duration_facts = []

    def save_acceptance(status):
        write_json(root / "acceptance.json", {
            "status": status,
            "cpu": args.cpu,
            "function": function,
            "split_parent_count": len(split_records),
            "tampered_parent": target_parent["task"],
            "duration_facts": duration_facts,
            "case_count": len(reports),
            "passed_count": sum(1 for row in reports if row["pass"]),
            "cases": reports,
        })

    def run_case(label, expected_success, catalog_mutator=None,
                 source_mutator=None, policy_update="true"):
        directory = root / label
        directory.mkdir()
        module_copy = directory / "original-input.mlir"
        cost_copy = directory / "parent-cost-catalog.json"
        body_copy = directory / "profile-body-export.json"
        copy_exact(source_module, module_copy)
        copy_exact(cost_path, cost_copy)
        copy_exact(body_path, body_copy)
        if source_mutator is not None:
            module_text = module_copy.read_text()
            module_copy.write_text(source_mutator(module_text))
        if catalog_mutator is not None:
            write_json(cost_copy, catalog_mutator(read_json(cost_copy)))

        result_path = directory / "retimer-result.json"
        output_module = directory / "retimed-output.mlir"
        changes = {
            "parent-cost-file": cost_copy,
            "profile-body-export-file": body_copy,
            "original-f45-replica-scaling": policy_update,
        }
        command_argv = build_command(argv, changes, module_copy, result_path,
                                     output_module, args.cpu)
        write_json(directory / "command.json", {
            "argv": command_argv,
            "cwd": command_document.get("cwd", "."),
        })
        with (directory / "stdout.log").open("wb") as stdout, \
                (directory / "stderr.log").open("wb") as stderr:
            completed = subprocess.run(command_argv, stdout=stdout, stderr=stderr)
        diagnostic = (directory / "stderr.log").read_text(errors="replace")
        validation_error = None
        if expected_success:
            passed = completed.returncode == 0 and result_path.is_file()
            facts = []
            if passed:
                try:
                    result, facts = validate_positive(result_path, source_records,
                                                      dispatch_order, cost_catalog, body_export)
                    duration_facts[:] = facts
                except (KeyError, TypeError, ValueError) as error:
                    passed = False
                    validation_error = str(error)
        else:
            passed = (completed.returncode != 0 and not result_path.exists()
                      and "error:" in diagnostic.lower())
        report = {
            "case": label,
            "expected_success": expected_success,
            "exit_code": completed.returncode,
            "result_exists": result_path.is_file(),
            "expected_policy_argument": policy_update,
            "pass": passed,
        }
        if validation_error is not None:
            report["validation_error"] = validation_error
        reports.append(report)
        save_acceptance("pass_so_far" if all(row["pass"] for row in reports) else "in_progress")
        if not passed:
            detail = ("; validation: " + validation_error
                      if validation_error is not None else "")
            raise RuntimeError(label + " did not meet its expected outcome" + detail
                               + "; see " + str(directory / "stderr.log"))

    run_case("positive", True)

    def mutate_cost(field, mode="increment"):
        return lambda catalog: mutate_catalog(catalog, target_parent["task"], field, mode)

    def shape_mutation(text):
        return mutate_schedule_field(text, target_parent, "shape", 0,
                                     "shape", "forged-shape", quoted=True)

    def context_mutation(text):
        record = next(row for row in extract_source_tasks(text)
                      if row["task"] == target_parent["task"])
        first = record["placements"][0]["context_id"]
        new_context = (first + 1) % 6
        return mutate_schedule_field(text, record, "placement", 0,
                                     "context_id", new_context)

    negative_cases = [
        ("reject-missing-f45-mode", None, None, None),
        ("reject-false-f45-mode", None, None, "false"),
        ("reject-missing-canonical-witness",
         mutate_cost("canonical_module_witness", "missing"), None, "true"),
        ("reject-forged-canonical-witness",
         mutate_cost("canonical_module_witness", "forged"), None, "true"),
        ("reject-forged-selected-normalized-body",
         mutate_cost("normalized_mapper_body"), None, "true"),
        ("reject-forged-replica-shape-inventory", None, shape_mutation, "true"),
        ("reject-forged-placement-context-inventory", None, context_mutation, "true"),
        ("reject-forged-full-profile-duration",
         mutate_cost("estimated_latency"), None, "true"),
        ("reject-forged-full-profile-ii",
         mutate_cost("compiled_ii"), None, "true"),
    ]
    for label, catalog_mutator, source_mutator, policy_update in negative_cases:
        run_case(label, False, catalog_mutator, source_mutator, policy_update)

    save_acceptance("pass")
    print(json.dumps({
        "status": "pass",
        "case_count": len(reports),
        "positive_result": str(root / "positive" / "retimer-result.json"),
        "output_root": str(root),
    }))


if __name__ == "__main__":
    main()
