#!/usr/bin/env python3
"""Create fail-closed negative inputs for the original-AMOEBA replica proof.

The source-domain partition verifier checks the metadata cases. The active
proof case reruns the real materializer on a source module with one forged
argument proof record. This script never edits its inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable


def unquoted_positions(text: str, needle: str, start: int = 0,
                       end: int | None = None) -> list[int]:
    limit = len(text) if end is None else min(end, len(text))
    found: list[int] = []
    quoted = False
    escaped = False
    index = 0
    while index < limit:
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif index >= start and text.startswith(needle, index):
            found.append(index)
            index += len(needle)
            continue
        index += 1
    return found


def matching_delimiter(text: str, opening: int, left: str, right: str) -> int:
    if opening >= len(text) or text[opening] != left:
        raise ValueError(f"expected {left!r} at offset {opening}")
    depth = 0
    quoted = False
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError(f"unclosed {left!r} value at offset {opening}")


def attribute_value_span(text: str, name: str, start: int = 0,
                         end: int | None = None,
                         delimiter: tuple[str, str] = ("{", "}")) -> tuple[int, int]:
    stop = len(text) if end is None else end
    matches = unquoted_positions(text, name, start, stop)
    if not matches:
        raise ValueError(f"attribute {name!r} was not found")
    position = matches[0]
    equals = text.find("=", position + len(name), stop)
    opening = text.find(delimiter[0], equals + 1, stop)
    if equals < 0 or opening < 0:
        raise ValueError(f"attribute {name!r} has no {delimiter[0]} value")
    return opening, matching_delimiter(text, opening, *delimiter)


def task_name_properties(text: str) -> list[tuple[int, str]]:
    properties: list[tuple[int, str]] = []
    for position in unquoted_positions(text, "task_name"):
        equals = text.find("=", position + len("task_name"))
        quote = text.find('"', equals + 1)
        if equals < 0 or quote < 0:
            continue
        close = quote + 1
        escaped = False
        while close < len(text):
            if escaped:
                escaped = False
            elif text[close] == "\\":
                escaped = True
            elif text[close] == '"':
                break
            close += 1
        if close < len(text):
            properties.append((position, text[quote + 1:close]))
    return properties


def task_attribute_span(text: str, parent_task: str, replica_id: int,
                        attribute: str) -> tuple[int, int]:
    child_name = f"{parent_task}.replica.{replica_id}"
    _, (left, right) = task_body_and_attributes(text, child_name)
    return attribute_value_span(text, attribute, left, right)


def manifest_child_dictionary(text: str, child_name: str) -> tuple[int, int]:
    opening, end = attribute_value_span(
        text, "amoeba.original_amoeba.materialized_decisions",
        delimiter=("[", "]"))
    properties = [
        position for position, value in task_name_properties(text)
        if value == child_name and opening < position < end
    ]
    if not properties:
        raise ValueError(f"function decision manifest omits child {child_name!r}")
    target = properties[0]
    stack: list[int] = []
    spans: list[tuple[int, int]] = []
    quoted = False
    escaped = False
    for index in range(opening, end):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            stack.append(index)
        elif char == "}" and stack:
            spans.append((stack.pop(), index + 1))
    containing = [(left, right) for left, right in spans
                  if left < target < right]
    if not containing:
        raise ValueError(f"cannot locate manifest record for {child_name!r}")
    return min(containing, key=lambda span: span[1] - span[0])


def field_value_span(text: str, dictionary: tuple[int, int], field: str,
                     delimiter: tuple[str, str] | None = None) -> tuple[int, int]:
    left, right = dictionary
    positions = unquoted_positions(text, field, left, right)
    if not positions:
        raise ValueError(f"field {field!r} is missing from selected record")
    position = positions[0]
    if delimiter is None:
        return left, right
    return attribute_value_span(text, field, position, right, delimiter)


def direct_field_span(text: str, dictionary: tuple[int, int],
                      field: str) -> tuple[int, int]:
    left, right = dictionary
    start = left + 1
    depth = 0
    quoted = False
    escaped = False
    fields: list[tuple[int, int]] = []
    for index in range(start, right - 1):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[<(":
            depth += 1
        elif char in "}])>" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            fields.append((start, index))
            start = index + 1
    fields.append((start, right - 1))
    for begin, end in fields:
        key = text[begin:end].split("=", 1)[0].strip()
        if key == field:
            return begin, end
    raise ValueError(f"direct field {field!r} is missing from selected record")


def change_integer(text: str, value_span: tuple[int, int], field: str,
                   replacement: Callable[[int], int]) -> str:
    left, right = value_span
    chunk = text[left:right]
    pattern = re.compile(rf"(\b{re.escape(field)}\s*=\s*)(-?\d+)(\s*:\s*i\d+)")
    match = pattern.search(chunk)
    if not match:
        raise ValueError(f"integer field {field!r} was not found")
    old = int(match.group(2))
    new = replacement(old)
    return text[:left] + chunk[:match.start(2)] + str(new) + chunk[match.end(2):] + text[right:]


def change_string(text: str, value_span: tuple[int, int], field: str,
                  replacement: str) -> str:
    left, right = value_span
    chunk = text[left:right]
    pattern = re.compile(rf'(\b{re.escape(field)}\s*=\s*)"([^"]*)"')
    match = pattern.search(chunk)
    if not match:
        raise ValueError(f"string field {field!r} was not found")
    return text[:left] + chunk[:match.start(2)] + replacement + chunk[match.end(2):] + text[right:]


def write_variant(source: Path, output: Path,
                  mutate: Callable[[str], str]) -> None:
    original = source.read_text()
    output.write_text(mutate(original))


def mutate_child_work(text: str, child: str) -> str:
    parent, replica_text = child.rsplit(".replica.", 1)
    span = task_attribute_span(
        text, parent, int(replica_text),
        "amoeba.original_amoeba.source_partition_realization")
    return change_integer(text, span, "source_work_count", lambda value: value + 1)


def mutate_manifest_bounds(text: str, child: str) -> str:
    summary = manifest_child_dictionary(text, child)
    bounds = field_value_span(text, summary, "partition_bounds", ("[", "]"))
    return change_integer(text, bounds, "upper", lambda value: value + 1)


def mutate_shape(text: str, child: str) -> str:
    summary = manifest_child_dictionary(text, child)
    shape = field_value_span(text, summary, "replica_shape", ("{", "}"))
    return change_string(text, shape, "shape", "9x1")


def mutate_context(text: str, child: str) -> str:
    summary = manifest_child_dictionary(text, child)
    placements = field_value_span(text, summary, "placements", ("[", "]"))
    return change_integer(text, placements, "context_id", lambda _value: 6)


def mutate_replica_id(text: str, child: str) -> str:
    summary = manifest_child_dictionary(text, child)
    replica_id = direct_field_span(text, summary, "replica_id")
    return change_integer(text, replica_id, "replica_id",
                          lambda value: value + 17)


def mutate_active_proof(text: str) -> str:
    proof = attribute_value_span(text, "amoeba.active_transfer_proof")
    left, right = proof
    chunk = text[left:right]
    match = re.search(r"(\bproven\s*=\s*)(true|false)\b", chunk)
    if not match:
        raise ValueError("active-transfer proof has no proven field")
    value = "false" if match.group(2) == "true" else "true"
    return text[:left] + chunk[:match.start(2)] + value + chunk[match.end(2):] + text[right:]


def task_body_and_attributes(text: str, task_name: str) -> tuple[tuple[int, int], tuple[int, int]]:
    pattern = re.compile(
        rf'task_name\s*=\s*"{re.escape(task_name)}"\s*}}\s*>\s*\(\{{')
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"expected one generic Taskflow operation named {task_name!r}, "
            f"found {len(matches)}")
    body_open = text.rfind("{", matches[0].start(), matches[0].end())
    body_end = matching_delimiter(text, body_open, "{", "}")
    attributes_open = text.find("{", body_end)
    if attributes_open < 0:
        raise ValueError(f"task {task_name!r} has no attribute dictionary")
    attributes_end = matching_delimiter(text, attributes_open, "{", "}")
    return (body_open, body_end), (attributes_open, attributes_end)


def top_level_dictionary_fields(text: str) -> list[str]:
    fields: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {"{": "}", "[": "]", "<": ">", "(": ")"}
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == "," and not stack:
            fields.append(text[start:index].strip())
            start = index + 1
    fields.append(text[start:].strip())
    return [field for field in fields if field]


def mark_source_body_pending(text: str, task_name: str) -> str:
    _, (left, right) = task_body_and_attributes(text, task_name)
    fields = top_level_dictionary_fields(text[left + 1:right - 1])
    binding_names = {
        "amoeba.source_iteration_control_binding",
        "amoeba.source_iteration_source_control_binding",
        "amoeba.source_iteration_capture_pending",
    }
    fields = [field for field in fields
              if field.split("=", 1)[0].strip() not in binding_names]
    fields.append("amoeba.source_iteration_capture_pending")
    replacement = "{" + ", ".join(fields) + "}"
    return text[:left] + replacement + text[right:]


def mutate_nonzero_guard_predicate(text: str, task_name: str) -> str:
    (body_left, body_right), _ = task_body_and_attributes(text, task_name)
    if_position = text.find('"scf.if"', body_left, body_right)
    if if_position < 0:
        raise ValueError(f"task {task_name!r} has no scf.if guard")
    compare = text.rfind('"arith.cmpi"', body_left, if_position)
    attribute_end = text.find("}>", compare, if_position)
    if compare < 0 or attribute_end < 0:
        raise ValueError("cannot locate the initialization guard comparison")
    attribute_end += 2
    chunk = text[compare:attribute_end]
    match = re.search(r"(\bpredicate\s*=\s*)(-?\d+)(\s*:\s*i64)", chunk)
    if not match or int(match.group(2)) != 0:
        raise ValueError("initialization guard is not an equality predicate")
    return (text[:compare] + chunk[:match.start(2)] + "1" +
            chunk[match.end(2):] + text[attribute_end:])


def mutate_unknown_guard_effect(text: str, task_name: str) -> str:
    (body_left, body_right), _ = task_body_and_attributes(text, task_name)
    if_position = text.find('"scf.if"', body_left, body_right)
    then_open_token = text.find("({", if_position, body_right)
    if if_position < 0 or then_open_token < 0:
        raise ValueError(f"task {task_name!r} has no generic scf.if then-region")
    then_open = then_open_token + 1
    then_right = matching_delimiter(text, then_open, "{", "}")
    then_body = text[then_open:then_right]
    store = re.search(r'"memref\.store"\([^,]+,\s*(%[A-Za-z0-9_.$-]+)',
                      then_body)
    if not store:
        raise ValueError("initialization then-region has no output store")
    output = store.group(1)
    argument_type = re.search(
        rf"{re.escape(output)}\s*:\s*(memref<[^>]+>)",
        text[body_left:body_right])
    if not argument_type:
        raise ValueError("cannot resolve the initialization output memref type")
    yield_position = text.find('"scf.yield"', then_open, then_right)
    if yield_position < 0:
        raise ValueError("initialization then-region has no scf.yield")
    memref_type = argument_type.group(1)
    copy = (f'"memref.copy"({output}, {output}) : '
            f"({memref_type}, {memref_type}) -> ()\n")
    return text[:yield_position] + copy + text[yield_position:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--canonical-module", type=Path, required=True)
    parser.add_argument("--materialized-module", type=Path, required=True)
    parser.add_argument("--active-source-module", type=Path, required=True)
    parser.add_argument("--guard-source-module", type=Path)
    parser.add_argument("--guard-function")
    parser.add_argument("--guard-task", default="Task_6")
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--seed-task", default="Task_16")
    parser.add_argument("--replica-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true",
                        help="run each C++ negative and require its expected diagnostic")
    args = parser.parse_args()
    for path in (args.optimizer, args.canonical_module, args.materialized_module,
                 args.active_source_module, args.architecture, args.network):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.guard_source_module and not args.guard_source_module.is_file():
        parser.error(f"guard source module does not exist: {args.guard_source_module}")
    if args.guard_source_module and not args.guard_function:
        parser.error("--guard-function is required with --guard-source-module")
    if not args.optimizer.is_file():
        parser.error(f"optimizer does not exist: {args.optimizer}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    child = f"{args.seed_task}.replica.{args.replica_id}"
    specs = [
        ("child-work", mutate_child_work,
         "native source partition metadata differs from exact child counter bounds"),
        ("manifest-bounds", mutate_manifest_bounds,
         "native source partition metadata differs from exact child counter bounds"),
        ("shape", mutate_shape,
         "replica decision metadata differs from its scheduler record"),
        ("context", mutate_context,
         "replica decision metadata differs from its scheduler record"),
        ("replica-id", mutate_replica_id,
         "replica decision metadata differs from its scheduler record"),
    ]
    records: list[dict[str, object]] = []
    for name, mutate, expected in specs:
        variant = args.output_dir / f"{name}.mlir"
        write_variant(args.materialized_module, variant,
                      lambda content, fn=mutate: fn(content, child))
        argv = [
            str(args.optimizer), str(variant), "--verify-each",
            "--verify-source-iteration-domain-partitions=" +
            f"parent-module={args.canonical_module} function={args.function}",
            "-o", "/dev/null",
        ]
        records.append({"case": name, "input": str(variant), "argv": argv,
                        "expect_exit_nonzero": True,
                        "expect_stderr_contains": expected})

    active_variant = args.output_dir / "active-proof-corruption.mlir"
    write_variant(args.active_source_module, active_variant, mutate_active_proof)
    active_argv = [
        str(args.optimizer), str(active_variant), "--verify-each",
        f"--architecture-spec={args.architecture}",
        f"--joint-inter-task-network-spec={args.network}",
        "--materialize-joint-task-replicas=" +
        f"function={args.function} task={args.seed_task} "
        "original-amoeba-fixed-decision=true",
        "--mlir-print-op-generic", "-o", "/dev/null",
    ]
    records.append({
        "case": "active-proof-corruption", "input": str(active_variant),
        "argv": active_argv, "expect_exit_nonzero": True,
        "expect_stderr_contains":
            "invalid active-transfer proof on argument",
    })

    if args.guard_source_module:
        guard_diagnostic = (
            "no output axis has a source-proved exact partition and a legal "
            "replica materialization")
        guard_cases = [
            ("guard-nonzero-predicate", mutate_nonzero_guard_predicate),
            ("guard-unknown-memory-effect", mutate_unknown_guard_effect),
        ]
        pipeline = (
            "builtin.module(bind-source-iteration-domain,"
            "materialize-joint-task-replicas{"
            f"function={args.guard_function} task={args.guard_task} "
            "original-amoeba-fixed-decision=true})")
        for name, mutate in guard_cases:
            variant = args.output_dir / f"{name}.mlir"
            changed = mutate(args.guard_source_module.read_text(), args.guard_task)
            changed = mark_source_body_pending(changed, args.guard_task)
            variant.write_text(changed)
            argv = [
                str(args.optimizer), str(variant), "--verify-each",
                f"--architecture-spec={args.architecture}",
                f"--joint-inter-task-network-spec={args.network}",
                f"--pass-pipeline={pipeline}", "--mlir-print-op-generic",
                "-o", "/dev/null",
            ]
            records.append({
                "case": name, "input": str(variant), "argv": argv,
                "expect_exit_nonzero": True,
                "expect_stderr_contains": guard_diagnostic,
            })

    report = {"schema": "orbit-replica-phase1-tamper-cases-v1",
              "function": args.function, "child_task": child,
              "cases": records}
    report_path = args.output_dir / "tamper-cases.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not args.run:
        print(report_path)
        return 0

    failures: list[str] = []
    for record in records:
        completed = subprocess.run(record["argv"], text=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, check=False)
        stdout_path = args.output_dir / f"{record['case']}.stdout.log"
        stderr_path = args.output_dir / f"{record['case']}.stderr.log"
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        record["exit_code"] = completed.returncode
        record["stdout_log"] = str(stdout_path)
        record["stderr_log"] = str(stderr_path)
        expected = str(record["expect_stderr_contains"])
        if completed.returncode == 0 or expected not in completed.stderr:
            failures.append(
                f"{record['case']}: exit={completed.returncode}, "
                f"expected diagnostic substring {expected!r}")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
