#!/usr/bin/env python3
"""Check nested source-owned tiling lineage and fail-closed replay admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


TASK_OPS = ("taskflow.task", "neura.task")
TASK_NAME = re.compile(r'\btask_name\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
PRETTY_TASK = re.compile(r"\s+@([^\s(\[]+)")
LEDGER = "amoeba.neighborhood.partition_lineage.v1 ="
CURRENT_BINDING = "amoeba.source_iteration_control_binding"
SOURCE_BINDING = "amoeba.source_iteration_source_control_binding"
CAPTURE_PENDING = "amoeba.source_iteration_capture_pending"


def fail(message: str) -> None:
    raise SystemExit(f"nested tiling lineage check: {message}")


def run(command: list[str], log: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    return result


def verify_input(optimizer: Path, source: Path, out_dir: Path,
                 label: str) -> None:
    result = run([str(optimizer), str(source), "--verify-each", "-o",
                  str(out_dir / f"{label}.verified.mlir")],
                 out_dir / f"{label}.verify.log")
    if result.returncode != 0:
        fail(f"{label} mutation is not valid MLIR; inspect {label}.verify.log")


def task_starts(ir: str) -> list[tuple[int, str | None]]:
    """Find live task ops while skipping operation names inside MLIR strings."""
    starts: list[tuple[int, str | None]] = []
    position = 0
    while position < len(ir):
        if ir[position] == '"':
            begin = position
            position += 1
            while position < len(ir):
                if ir[position] == "\\":
                    position += 2
                    continue
                if ir[position] == '"':
                    break
                position += 1
            token = ir[begin + 1:position]
            position += 1
            if token in TASK_OPS:
                operand = position
                while operand < len(ir) and ir[operand].isspace():
                    operand += 1
                if operand < len(ir) and ir[operand] == "(":
                    starts.append((begin, None))
            continue
        if position == 0 or ir[position - 1].isspace():
            for operation in TASK_OPS:
                if not ir.startswith(operation, position):
                    continue
                name = PRETTY_TASK.match(ir, position + len(operation))
                if name:
                    starts.append((position, name.group(1)))
                break
        position += 1
    return starts


def task_spans(ir: str) -> list[tuple[str, int, int, str]]:
    starts = task_starts(ir)
    spans: list[tuple[str, int, int, str]] = []
    for index, (begin, pretty_name) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(ir)
        segment = ir[begin:end]
        name = pretty_name
        if name is None:
            match = TASK_NAME.search(segment)
            name = match.group(1) if match else None
        if name is not None:
            spans.append((name, begin, end, segment))
    return spans


def task_span(ir: str, name: str) -> tuple[int, int, str]:
    for found, begin, end, segment in task_spans(ir):
        if found == name:
            return begin, end, segment
    fail(f"replay output has no task {name}")


def find_unquoted(text: str, needle: str) -> int:
    position = 0
    while position < len(text):
        if text[position] == '"':
            position += 1
            while position < len(text):
                if text[position] == "\\":
                    position += 2
                    continue
                if text[position] == '"':
                    position += 1
                    break
                position += 1
            continue
        if text.startswith(needle, position):
            return position
        position += 1
    return -1


def string_attr_value_span(segment: str, name: str) -> tuple[int, int]:
    key = find_unquoted(segment, name)
    if key < 0:
        fail(f"task has no live {name} attribute")
    equals = segment.find("=", key + len(name))
    if equals < 0:
        fail(f"task {name} attribute has no value")
    value_start = equals + 1
    while value_start < len(segment) and segment[value_start].isspace():
        value_start += 1
    if value_start >= len(segment) or segment[value_start] != '"':
        fail(f"task {name} attribute is not a string")
    position = value_start + 1
    while position < len(segment):
        if segment[position] == "\\":
            position += 2
            continue
        if segment[position] == '"':
            return value_start, position + 1
        position += 1
    fail(f"task {name} string attribute is unterminated")


def task_attr_dict_bounds(segment: str, pivot: int) -> tuple[int, int]:
    stack: list[int] = []
    position = 0
    while position < pivot:
        if segment[position] == '"':
            position += 1
            while position < pivot:
                if segment[position] == "\\":
                    position += 2
                    continue
                if segment[position] == '"':
                    position += 1
                    break
                position += 1
            continue
        if segment[position] == "{":
            stack.append(position)
        elif segment[position] == "}" and stack:
            stack.pop()
        position += 1
    if not stack:
        fail("generic task attribute dictionary was not found")
    begin = stack[-1]
    depth = 0
    position = begin
    while position < len(segment):
        if segment[position] == '"':
            position += 1
            while position < len(segment):
                if segment[position] == "\\":
                    position += 2
                    continue
                if segment[position] == '"':
                    position += 1
                    break
                position += 1
            continue
        if segment[position] == "{":
            depth += 1
        elif segment[position] == "}":
            depth -= 1
            if depth == 0:
                return begin, position
        position += 1
    fail("generic task attribute dictionary is not balanced")


def lineage_families(segment: str) -> list[str]:
    key = find_unquoted(segment, LEDGER)
    if key < 0:
        fail("derived task has no partition-lineage ledger")
    begin = key + len(LEDGER)
    while begin < len(segment) and segment[begin].isspace():
        begin += 1
    if begin >= len(segment) or segment[begin] != "[":
        fail("partition-lineage attribute is not an array")
    depth = 0
    position = begin
    in_string = False
    escaped = False
    while position < len(segment):
        char = segment[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                value = segment[begin:position + 1]
                return re.findall(r'\bfamily\s*=\s*"([^"\\]+)"', value)
        position += 1
    fail("partition-lineage array is unterminated")


def assert_lineage_sequences(candidate_text: str, expected: dict[str, list[str]],
                             label: str) -> None:
    actual_names = {name for name, _, _, _ in task_spans(candidate_text)}
    if set(expected) - actual_names:
        fail(f"{label} is missing expected derived tasks: "
             f"{sorted(set(expected) - actual_names)!r}")
    for name, families in expected.items():
        _, _, segment = task_span(candidate_text, name)
        actual = lineage_families(segment)
        if actual != families:
            fail(f"{label} task {name} has lineage {actual!r}, expected "
                 f"{families!r}")


def remove_current_binding_and_mark_pending(segment: str) -> str:
    key = find_unquoted(segment, CURRENT_BINDING)
    if key < 0:
        fail("mutated nested task has no current source binding to refresh")
    _, value_end = string_attr_value_span(segment, CURRENT_BINDING)
    end = value_end
    while end < len(segment) and segment[end].isspace():
        end += 1
    if end < len(segment) and segment[end] == ",":
        end += 1
        while end < len(segment) and segment[end].isspace():
            end += 1
    else:
        start = key
        while start > 0 and segment[start - 1].isspace():
            start -= 1
        if start > 0 and segment[start - 1] == ",":
            start -= 1
            segment = segment[:start] + segment[key:]
            key = start
            _, value_end = string_attr_value_span(segment, CURRENT_BINDING)
            end = value_end
    updated = segment[:key] + segment[end:]
    _, close = task_attr_dict_bounds(updated, key)
    dictionary_begin, _ = task_attr_dict_bounds(updated, key)
    content = updated[dictionary_begin + 1:close].strip()
    separator = ", " if content else ""
    return updated[:close] + separator + CAPTURE_PENDING + updated[close:]


def refresh_mutated_current_binding(optimizer: Path, original: str,
                                    mutated: str, task: str,
                                    out_dir: Path, label: str) -> Path:
    _, _, original_segment = task_span(original, task)
    origin_start, origin_end = string_attr_value_span(original_segment, SOURCE_BINDING)
    original_origin_token = original_segment[origin_start:origin_end]

    begin, end, segment = task_span(mutated, task)
    pending_segment = remove_current_binding_and_mark_pending(segment)
    marked = mutated[:begin] + pending_segment + mutated[end:]
    marked_input = out_dir / f"{label}.pending-binding.mlir"
    marked_input.write_text(marked)
    verify_input(optimizer, marked_input, out_dir, f"{label}-pending")
    rebound = out_dir / f"{label}.rebound.mlir"
    result = run([
        str(optimizer), str(marked_input),
        "--pass-pipeline=builtin.module(bind-source-iteration-domain)",
        "-o", str(rebound),
    ], out_dir / f"{label}.rebind.log")
    if result.returncode != 0:
        fail(f"{label}: compiler could not refresh only the current body witness")
    rebound_text = rebound.read_text()
    begin, end, segment = task_span(rebound_text, task)
    source_start, source_end = string_attr_value_span(segment, SOURCE_BINDING)
    segment = (segment[:source_start] + original_origin_token +
               segment[source_end:])
    final_text = rebound_text[:begin] + segment + rebound_text[end:]
    final_input = out_dir / f"{label}.candidate.mlir"
    final_input.write_text(final_text)
    verify_input(optimizer, final_input, out_dir, f"{label}-rebound")
    return final_input


def change_in_task(ir: str, task: str, old: str, new: str,
                   label: str) -> str:
    begin, end, segment = task_span(ir, task)
    if old not in segment:
        fail(f"{label}: mutation anchor not found in {task}: {old}")
    changed = segment.replace(old, new, 1)
    if changed == segment:
        fail(f"{label}: mutation did not change {task}")
    return ir[:begin] + changed + ir[end:]


def change_integer_task_attribute(ir: str, task: str, name: str,
                                  delta: int, label: str) -> str:
    begin, end, segment = task_span(ir, task)
    key = find_unquoted(segment, name)
    if key < 0:
        fail(f"{label}: task {task} has no {name} attribute")
    equals = segment.find("=", key + len(name))
    if equals < 0:
        fail(f"{label}: {name} attribute has no value")
    value_start = equals + 1
    while value_start < len(segment) and segment[value_start].isspace():
        value_start += 1
    match = re.match(r"[+-]?\d+", segment[value_start:])
    if not match:
        fail(f"{label}: {name} is not an integer attribute")
    old_value = int(match.group(0))
    new_value = old_value + delta
    if new_value == old_value:
        fail(f"{label}: {name} mutation did not change its value")
    value_end = value_start + len(match.group(0))
    changed = segment[:value_start] + str(new_value) + segment[value_end:]
    return ir[:begin] + changed + ir[end:]


def mutate_ledger(ir: str, task: str, mutation: str) -> str:
    begin, end, segment = task_span(ir, task)
    ledger_at = find_unquoted(segment, LEDGER)
    if ledger_at < 0:
        fail(f"{task} has no emitted source partition ledger")
    prefix, ledger_and_suffix = segment[:ledger_at], segment[ledger_at:]
    if mutation == "root":
        updated, count = re.subn(r'(root\s*=\s*)"Task"',
                                 r'\1"ForgedTask"', ledger_and_suffix,
                                 count=1)
        if count != 1:
            fail("root-forgery anchor not found in partition ledger")
    elif mutation == "range":
        match = re.search(
            r'(after\s*=\s*array<i64:\s*)(-?\d+)(,\s*)(-?\d+)',
            ledger_and_suffix)
        if not match:
            fail("derived-range anchor not found in partition ledger")
        upper = int(match.group(4))
        updated = (ledger_and_suffix[:match.start(4)] + str(upper + 1) +
                   ledger_and_suffix[match.end(4):])
    elif mutation == "step":
        updated, count = re.subn(r'(family\s*=\s*)"tiling"',
                                 r'\1"forged-family"', ledger_and_suffix,
                                 count=1)
        if count != 1:
            fail("step-family anchor not found in partition ledger")
    elif mutation == "missing":
        array_start = ledger_and_suffix.find("[", len(LEDGER))
        if array_start < 0:
            fail("partition-ledger array start was not found")
        depth = 0
        in_string = False
        escaped = False
        array_end = -1
        for index in range(array_start, len(ledger_and_suffix)):
            char = ledger_and_suffix[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    array_end = index + 1
                    break
        if array_end < 0:
            fail("partition-ledger array did not have balanced brackets")
        attribute_start = ledger_and_suffix.rfind(LEDGER, 0, array_start)
        remove_start = attribute_start
        remove_end = array_end
        following = remove_end
        while following < len(ledger_and_suffix) and ledger_and_suffix[following].isspace():
            following += 1
        if following < len(ledger_and_suffix) and ledger_and_suffix[following] == ",":
            remove_end = following + 1
        else:
            preceding = remove_start - 1
            while preceding >= 0 and ledger_and_suffix[preceding].isspace():
                preceding -= 1
            if preceding >= 0 and ledger_and_suffix[preceding] == ",":
                remove_start = preceding
        updated = ledger_and_suffix[:remove_start] + ledger_and_suffix[remove_end:]
    else:
        fail(f"unknown ledger mutation {mutation}")
    new_segment = prefix + updated
    return ir[:begin] + new_segment + ir[end:]


def identity_action() -> dict[str, object]:
    return {
        "family": "identity", "label": "identity", "primitives": [],
        "shapeTask": "", "shapeRows": 0, "shapeCols": 0,
        "canonicalReset": False,
    }


def write_actions(path: Path, canonical: Path, candidate: Path,
                  actions: list[dict[str, object]], cap: int) -> None:
    path.write_text(json.dumps({
        "schema": "orbit-joint-neighborhood-typed-actions-v1",
        "canonicalInput": str(canonical.resolve()),
        "candidateInput": str(candidate.resolve()),
        "function": "source_partition_lineage",
        "stage": "full-joint",
        "maxPartitionFactor": cap,
        "actions": actions,
    }, indent=2) + "\n")


def replay(optimizer: Path, canonical: Path, candidate: Path,
           actions: list[dict[str, object]], out_dir: Path, label: str,
           cap: int = 4) -> tuple[subprocess.CompletedProcess[str], Path]:
    action_file = out_dir / f"{label}.actions.json"
    replay_dir = out_dir / f"{label}.complete"
    write_actions(action_file, canonical, candidate, actions, cap)

    def quoted(path: Path) -> str:
        return json.dumps(str(path.resolve()))

    options = " ".join((
        f"action-file={quoted(action_file)}",
        f"canonical-input={quoted(canonical)}",
        f"candidate-input={quoted(candidate)}",
        "function=source_partition_lineage", "stage=full-joint",
        f"max-partition-factor={cap}", f"output-dir={quoted(replay_dir)}",
    ))
    result = run([
        str(optimizer), str(candidate), "--verify-each",
        f"--replay-joint-neighborhood-actions={options}", "-o", "/dev/null",
    ], out_dir / f"{label}.log")
    return result, replay_dir


def require_rejection(optimizer: Path, canonical: Path, candidate_text: str,
                      out_dir: Path, label: str, expected: str,
                      task: str | None = None, *,
                      ledger_mutation: str | None = None,
                      body_mutation: bool = False,
                      task_attr_mutation: str | None = None) -> str:
    if task and body_mutation:
        mutated = change_in_task(candidate_text, task,
                                 'lhs_value = "%input0"',
                                 'lhs_value = "%input1"', label)
    elif task and task_attr_mutation:
        mutated = change_integer_task_attribute(candidate_text, task,
                                                task_attr_mutation, 1, label)
    elif task:
        if ledger_mutation is None:
            fail(f"{label}: no ledger mutation kind was specified")
        mutated = mutate_ledger(candidate_text, task, ledger_mutation)
    else:
        fail("negative case must name its mutation task")
    raw_candidate = out_dir / f"{label}.mutated.mlir"
    raw_candidate.write_text(mutated)
    verify_input(optimizer, raw_candidate, out_dir, f"{label}-mutated")
    candidate = raw_candidate if task_attr_mutation else \
        refresh_mutated_current_binding(
            optimizer, candidate_text, mutated, task, out_dir, label)
    result, replay_dir = replay(optimizer, canonical, candidate,
                                [identity_action()], out_dir, label)
    if result.returncode == 0:
        fail(f"{label} unexpectedly passed the source-owned replay guard")
    diagnostics = [line.strip() for line in result.stdout.splitlines()
                   if "error:" in line]
    if not diagnostics:
        fail(f"{label} failed without a compiler diagnostic; inspect {label}.log")
    diagnostic = diagnostics[-1]
    if expected not in diagnostic:
        fail(f"{label} failed at an unexpected guard: {diagnostic!r}")
    if replay_dir.exists():
        fail(f"{label} published partial replay output after rejection")
    return diagnostic


def tile(task: str) -> dict[str, object]:
    return {
        "family": "tiling", "label": f"tile:{task}:axis=0:factor=2",
        "primitives": [{
            "kind": "tile", "firstTask": task, "secondTask": "",
            "axis": 0, "factor": 2, "mode": "none",
        }],
        "shapeTask": "", "shapeRows": 0, "shapeCols": 0,
        "canonicalReset": False,
    }


def replica(task: str) -> dict[str, object]:
    return {
        "family": "replica",
        "label": f"replica:{task}:axis=0:factor=2",
        "primitives": [{
            "kind": "replica", "firstTask": task, "secondTask": "",
            "axis": 0, "factor": 2, "mode": "",
        }],
        "shapeTask": "", "shapeRows": 0, "shapeCols": 0,
        "canonicalReset": False,
    }


def shape() -> dict[str, object]:
    return {
        "family": "shape", "label": "shape:Task:1x2", "primitives": [],
        "shapeTask": "Task", "shapeRows": 1, "shapeCols": 2,
        "canonicalReset": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    optimizer = args.optimizer.resolve()
    fixture = args.fixture.resolve()
    if not optimizer.is_file() or not fixture.is_file():
        fail("optimizer or action replay fixture is missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out_dir = args.output_dir.resolve()
    canonical = out_dir / "canonical.mlir"
    canonical_result = run([
        str(optimizer), str(fixture),
        "--pass-pipeline=builtin.module(bind-source-iteration-domain)",
        "-o", str(canonical),
    ], out_dir / "canonical-bind.log")
    if canonical_result.returncode != 0:
        fail("fixture could not bind its canonical source iteration domain")

    path = [shape(), tile("Task"), tile("Task.tile.0.0"),
            tile("Task.tile.0.1")]
    positive, positive_dir = replay(optimizer, canonical, canonical, path,
                                    out_dir, "positive-nested-factor-four")
    if positive.returncode != 0:
        fail("legal two-level tiling failed; inspect positive-nested-factor-four.log")
    candidate_path = positive_dir / "candidate.mlir"
    if not candidate_path.is_file():
        fail("legal nested replay omitted candidate.mlir")
    nested_candidate_text = candidate_path.read_text()
    nested_tasks = sorted(name for name, _, _, _ in task_spans(nested_candidate_text)
                          if name.startswith(("Task.tile.0.0.tile.0.",
                                              "Task.tile.0.1.tile.0.")))
    expected_nested = [
        "Task.tile.0.0.tile.0.0", "Task.tile.0.0.tile.0.1",
        "Task.tile.0.1.tile.0.0", "Task.tile.0.1.tile.0.1",
    ]
    if nested_tasks != expected_nested:
        fail(f"expected both nested child tasks, found {nested_tasks!r}")
    for task in nested_tasks:
        _, _, segment = task_span(nested_candidate_text, task)
        if find_unquoted(segment, LEDGER) < 0:
            fail(f"positive nested task {task} has no partition ledger")

    # Exercise both mixed-family orders. The final child's current counters,
    # tile/replica metadata, and full root coverage are independently checked
    # by the same source-owned guard used for all replay candidates.
    cross_family_paths = {
        "tile_then_replica": [shape(), tile("Task"),
                              replica("Task.tile.0.0")],
        "replica_then_tile": [shape(), replica("Task"),
                              tile("Task.replica.0"),
                              tile("Task.replica.1")],
    }
    cross_family_results: dict[str, str] = {}
    cross_family_reimports: dict[str, str] = {}
    for label, actions in cross_family_paths.items():
        result, result_dir = replay(optimizer, canonical, canonical, actions,
                                    out_dir, f"positive-{label}")
        if result.returncode != 0:
            fail(f"positive {label} lineage chain failed; inspect positive-{label}.log")
        facts_path = result_dir / "source-facts.json"
        if not facts_path.is_file() or not (result_dir / "candidate.mlir").is_file():
            fail(f"positive {label} did not publish complete source-guarded output")
        facts = json.loads(facts_path.read_text())
        if facts.get("status") != "complete" or facts.get("source_iteration_domain_verified") is not True:
            fail(f"positive {label} lacks a complete verified source-domain report")
        mixed_candidate_text = (result_dir / "candidate.mlir").read_text()
        if label == "tile_then_replica":
            expected_lineage = {
                "Task.tile.0.0.replica.0": ["tiling", "replica"],
                "Task.tile.0.0.replica.1": ["tiling", "replica"],
                "Task.tile.0.1": ["tiling"],
            }
        else:
            expected_lineage = {
                "Task.replica.0.tile.0.0": ["replica", "tiling"],
                "Task.replica.0.tile.0.1": ["replica", "tiling"],
                "Task.replica.1.tile.0.0": ["replica", "tiling"],
                "Task.replica.1.tile.0.1": ["replica", "tiling"],
            }
        assert_lineage_sequences(mixed_candidate_text, expected_lineage, label)
        cross_family_results[label] = str(facts_path)

        reimported, reimport_dir = replay(
            optimizer, canonical, result_dir / "candidate.mlir",
            [identity_action()], out_dir, f"reimport-{label}")
        if reimported.returncode != 0:
            fail(f"intact {label} candidate could not be reimported; "
                 f"inspect reimport-{label}.log")
        reimport_facts_path = reimport_dir / "source-facts.json"
        reimport_candidate = reimport_dir / "candidate.mlir"
        if not reimport_facts_path.is_file() or not reimport_candidate.is_file():
            fail(f"intact {label} reimport omitted complete source-guarded output")
        reimport_facts = json.loads(reimport_facts_path.read_text())
        if reimport_facts.get("status") != "complete" or reimport_facts.get("source_iteration_domain_verified") is not True:
            fail(f"intact {label} reimport lacks a complete verified source-domain report")
        assert_lineage_sequences(reimport_candidate.read_text(), expected_lineage,
                                 f"reimport-{label}")
        cross_family_reimports[label] = str(reimport_facts_path)

    negatives: dict[str, str] = {}
    negatives["forged_root"] = require_rejection(
        optimizer, canonical, nested_candidate_text, out_dir, "forged-root",
        "names an unknown parent task",
        nested_tasks[0], ledger_mutation="root")
    negatives["forged_range"] = require_rejection(
        optimizer, canonical, nested_candidate_text, out_dir, "forged-range",
        "source-owned tiling partition lineage has a broken interval chain",
        nested_tasks[0], ledger_mutation="range")
    negatives["forged_step"] = require_rejection(
        optimizer, canonical, nested_candidate_text, out_dir, "forged-step",
        "source-owned tiling partition lineage has a broken interval chain",
        nested_tasks[0], ledger_mutation="step")
    negatives["missing_ledger"] = require_rejection(
        optimizer, canonical, nested_candidate_text, out_dir, "missing-ledger",
        "names an unknown parent task",
        nested_tasks[0], ledger_mutation="missing")
    negatives["changed_body"] = require_rejection(
        optimizer, canonical, nested_candidate_text, out_dir, "changed-body",
        "source-domain child body proof failed", nested_tasks[0],
        body_mutation=True)
    replica_tiling_candidate = (
        out_dir / "positive-replica_then_tile.complete/candidate.mlir").read_text()
    negatives["forged_replica_shard_trip_count"] = require_rejection(
        optimizer, canonical, replica_tiling_candidate, out_dir,
        "forged-replica-shard-trip-count",
        "source-owned replica shard trip count disagrees with authenticated replica bounds",
        "Task.replica.0.tile.0.0",
        task_attr_mutation="amoeba.replica.shard_trip_count")

    capped, capped_dir = replay(
        optimizer, canonical, canonical, path, out_dir,
        "negative-cumulative-cap", cap=2)
    if capped.returncode == 0:
        fail("two factor-two tilings exceeded cumulative cap two without rejection")
    cap_diagnostics = [line.strip() for line in capped.stdout.splitlines()
                       if "error:" in line]
    if not cap_diagnostics or "cumulative_original_tiling_factor_exceeds_cap" not in cap_diagnostics[-1]:
        fail("cumulative cap failed without its exact source-lineage diagnostic")
    if capped_dir.exists():
        fail("cumulative-cap rejection published partial replay output")

    report = {
        "schema": "orbit-nested-tiling-lineage-checks-v1",
        "optimizer": str(optimizer),
        "fixture": str(fixture),
        "positive_candidate": str(candidate_path),
        "positive_nested_tasks": nested_tasks,
        "positive_cross_family_facts": cross_family_results,
        "positive_cross_family_reimport_facts": cross_family_reimports,
        "negative_diagnostics": negatives,
        "cumulative_cap_diagnostic": cap_diagnostics[-1],
        "result": "pass",
    }
    report_path = out_dir / "result.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
