#!/usr/bin/env python3
"""Run opt-in active-transfer neighborhood checks without invoking a mapper.

The caller supplies a built mlir-amoeba-opt and the normal source-owned search
inputs. The runner writes only beneath --output-root and uses production
scheduling to compare native search outputs; it never launches mapping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

DEFAULT_FUNCTION = "_Z8gcn_funciPA256_KiS1_S1_S1_PA16_S_S3_S3_S3_PA256_iPA256_S4_PA256_A16_iSA_SA_Pi"
PROOF_SCHEMA = "orbit-static-active-transfer-proof-v1"
EXPECTED_GCN_ACTIVE_SHAPES = {
    1: (5, 5), 2: (5, 5), 3: (5, 5), 4: (5, 5),
    5: (5, 16), 6: (16, 16), 7: (16, 16), 8: (16, 16),
    9: (4, 5), 10: (4, 5, 5), 11: (3, 5, 16), 12: (12, 5, 16),
}
EXPECTED_GCN_CAPACITIES = {
    1: (256, 256), 2: (256, 256), 3: (256, 256), 4: (256, 256),
    5: (256, 16), 6: (16, 16), 7: (16, 16), 8: (16, 16),
    9: (4, 256), 10: (4, 256, 256), 11: (3, 256, 16),
    12: (12, 256, 16), 13: (3, 256, 16),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bind_runtime_test_protocol(args: argparse.Namespace, output: Path, *,
                               stage: str, workers: int,
                               max_candidates: int) -> Path:
    """Create the exact local protocol required by this single invocation.

    The supplied protocol contributes shared model, cost, architecture, and
    scheduler contracts. Search budgets are copied from the invocation so the
    compiler's exact-byte checkpoint binding describes the tested run. These
    files are explicitly local runtime-test protocols, never publication
    protocols.
    """
    base = json.loads(args.protocol.read_text())
    search = base.get("search")
    if not isinstance(search, dict):
        fail("base protocol has no search object")
    base["source_protocol_schema"] = base.get("schema")
    base["schema"] = "orbit-active-transfer-runtime-test-v1"
    base["scope"] = "isolated active-transfer neighborhood runtime acceptance; not publication"
    base["publication_authorization"] = {
        "authorized": False,
        "scope": "local runtime acceptance only",
    }
    base["runtime_test"] = {
        "protocol_source_path": str(args.protocol.resolve()),
        "stage": stage,
        "scoring_workers": workers,
        "max_rounds": args.max_rounds,
        "max_unique_complete_candidates_scored": max_candidates,
        "beam_width": args.beam_width,
        "diversity_min_slots": args.diversity_slots,
        "max_partition_factor": args.max_partition_factor,
    }
    search["max_rounds"] = args.max_rounds
    search["max_unique_complete_candidates_scored"] = max_candidates
    search["beam_width"] = args.beam_width
    search["diversity_min_slots"] = args.diversity_slots
    search["max_partition_factor"] = args.max_partition_factor
    search["scoring_workers"] = workers
    execution = base.get("execution_policy")
    if isinstance(execution, dict):
        execution["scoring_workers"] = workers

    protocol_path = output / "protocol-runtime-test.json"
    protocol_bytes = json.dumps(base, indent=2, sort_keys=True) + "\n"
    if protocol_path.exists():
        if protocol_path.read_text() != protocol_bytes:
            fail(f"resume protocol differs from the checkpoint's local protocol: {protocol_path}")
    else:
        if not output.is_dir():
            fail(f"protocol output directory does not exist: {output}")
        protocol_path.write_text(protocol_bytes)
    return protocol_path


def run_search(args: argparse.Namespace, label: str, *, stage: str,
               output: Path, active_arguments: str, require_proven: bool = True,
               workers: int = 1, max_candidates: int = 6,
               previous_winner: Path | None = None, pause_after: int = 0,
               resume: bool = False) -> subprocess.CompletedProcess[str]:
    if resume:
        if not output.is_dir():
            fail(f"resume output directory does not exist: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    protocol_path = bind_runtime_test_protocol(
        args, output, stage=stage, workers=workers,
        max_candidates=max_candidates)
    options = [
        f"output-dir={output}",
        f"function={args.function}",
        f"stage={stage}",
        f"architecture-path={args.architecture}",
        f"source-repository={args.source_repository}",
        f"source-commit={args.source_commit}",
        f"protocol={protocol_path}",
        f"source-contract-file={args.source_contract}",
        f"max-rounds={args.max_rounds}",
        f"max-candidates={max_candidates}",
        f"beam-width={args.beam_width}",
        f"diversity-slots={args.diversity_slots}",
        f"scoring-workers={workers}",
        f"max-partition-factor={args.max_partition_factor}",
        f"active-transfer-arguments={active_arguments}",
        f"active-transfer-require-proven={'true' if require_proven else 'false'}",
        f"minimum-free-bytes={args.minimum_free_bytes}",
        f"test-pause-after-candidates={pause_after}",
        f"checkpoint={output / 'checkpoint.json'}",
    ]
    if args.model:
        options.append(f"model={args.model}")
    if args.cache:
        options.append(f"cache={args.cache}")
    if args.parent_cost_file:
        options.append(f"parent-cost-file={args.parent_cost_file}")
    if previous_winner:
        options.append(f"previous-winner={previous_winner}")
    if resume:
        options.append("resume=true")
    command = [
        str(args.optimizer), str(args.canonical), "--verify-each",
        f"--architecture-spec={args.architecture}",
        "--search-joint-neighborhood=" + " ".join(options),
        "-o", str(output / "search-output.mlir"),
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (output / f"{label}.log").write_text(result.stdout)
    return result


def verify_task8_output_shard_replay(
        args: argparse.Namespace, output: Path,
        task8_row: dict[str, Any]) -> dict[str, Any]:
    """Replay the archived Task_8 replica and authenticate its output axis.

    The action's axis is an output coordinate. Task_8 output dimension 2 maps
    to Taskflow counter 1 (the relay); the leaf counter remains the full K
    reduction domain in both generated replicas.
    """
    replay_dir = output / "task8-output-axis-proof"
    replay_dir.mkdir(parents=True, exist_ok=False)
    canonical = args.canonical.resolve()
    function = args.function
    history = task8_row.get("action_history", {})
    actions = history.get("actions")
    if not history.get("known") or not isinstance(actions, list) or len(actions) != 1:
        fail("Task_8 positive row lacks its exact typed single-action history")
    action = actions[0]
    if (action.get("family") != "replica" or
            action.get("label") != "replica:Task_8:axis=2:factor=2"):
        fail("Task_8 positive row is not the expected output-axis replica")

    action_file = replay_dir / "actions.json"
    action_file.write_text(json.dumps({
        "schema": "orbit-joint-neighborhood-typed-actions-v1",
        "canonicalInput": str(canonical),
        "candidateInput": str(canonical),
        "function": function,
        "stage": "shape-temporal-replica",
        "maxPartitionFactor": 8,
        "initialShapes": history.get("initialShapes", []),
        "actions": actions,
    }, indent=2) + "\n")

    def quoted(path: Path) -> str:
        return json.dumps(str(path.resolve()))

    replay_output = replay_dir / "replayed"
    options = " ".join((
        f"action-file={quoted(action_file)}",
        f"canonical-input={quoted(canonical)}",
        f"candidate-input={quoted(canonical)}",
        f"function={function}",
        "stage=shape-temporal-replica",
        "max-partition-factor=8",
        f"output-dir={quoted(replay_output)}",
    ))
    command = [
        str(args.optimizer), str(canonical), "--verify-each",
        f"--replay-joint-neighborhood-actions={options}", "-o", "/dev/null",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (replay_dir / "replay.log").write_text(result.stdout)
    if result.returncode:
        fail("exact Task_8 action replay failed: " + result.stdout[-6000:])

    candidate = replay_output / "candidate.mlir"
    facts_path = replay_output / "source-facts.json"
    if not candidate.is_file() or not facts_path.is_file():
        fail("Task_8 action replay omitted its candidate or source facts")
    module = candidate.read_text()
    task_starts = list(re.finditer(
        r'(?m)^    %[^\n]* = "taskflow\.task"[^\n]*'
        r'task_name = "(Task_8\.replica\.[01])"', module))
    if len(task_starts) != 2 or {match.group(1) for match in task_starts} != {
            "Task_8.replica.0", "Task_8.replica.1"}:
        fail("Task_8 output-axis replay did not emit exactly two replicas")

    parts: dict[str, tuple[int, int]] = {}
    expected_ranges = {"Task_8.replica.0": (0, 8), "Task_8.replica.1": (8, 16)}
    for index, match in enumerate(task_starts):
        name = match.group(1)
        end = task_starts[index + 1].start() if index + 1 < len(task_starts) else module.find(
            'task_name = "Task_9"', match.end())
        if end < 0:
            fail("Task_8 output-axis replay has no next task boundary")
        block = module[match.start():end]
        expected_lower, expected_upper = expected_ranges[name]
        attributes = {
            "output_shard_axis": r"amoeba\.replica\.output_shard_axis = (\d+) : i64",
            "shard_axis": r"amoeba\.replica\.shard_axis = (\d+) : i64",
            "shard_lower": r"amoeba\.replica\.shard_lower = (\d+) : i64",
            "shard_upper": r"amoeba\.replica\.shard_upper = (\d+) : i64",
        }
        values: dict[str, int] = {}
        for attribute, pattern in attributes.items():
            found = re.search(pattern, block)
            if not found:
                fail(f"{name} lacks authenticated {attribute} metadata")
            values[attribute] = int(found.group(1))
        counter_map = re.search(
            r"amoeba\.replica\.output_counter_axes = array<i64: ([^>]+)>",
            block)
        constant_map = re.search(
            r"amoeba\.replica\.output_constant_axes = array<i64: ([^>]+)>",
            block)
        if (values["output_shard_axis"] != 2 or values["shard_axis"] != 1 or
                not counter_map or
                [int(value.strip()) for value in counter_map.group(1).split(",")] !=
                [4294967295, 0, 1] or
                not constant_map or
                [int(value.strip()) for value in constant_map.group(1).split(",")] !=
                [0, -9223372036854775808, -9223372036854775808] or
                (values["shard_lower"], values["shard_upper"]) !=
                (expected_lower, expected_upper)):
            fail(f"{name} output shard is not mapped to disjoint relay regions")

        constants = {
            value_name: int(value)
            for value_name, value in re.findall(
                r'(%[\w.$-]+) = "arith\.constant"\(\) <\{value = (-?\d+) : index\}>',
                block)
        }
        taskflow_leaf = next((line for line in block.splitlines()
                              if '"taskflow.counter"' in line and
                              'counter_id = 2 : i32' in line), "")
        operands = re.search(r'"taskflow\.counter"\(([^)]*)\)', taskflow_leaf)
        taskflow_bounds = ([constants.get(value.strip())
                            for value in operands.group(1).split(",")[-3:]]
                           if operands else [])
        neura_leaf = next((line for line in block.splitlines()
                           if '"neura.counter"' in line and
                           'counter_id = 2 : i32' in line), "")
        if (taskflow_bounds != [0, 16, 1] or
                "lower_bound_value = 0 : index" not in neura_leaf or
                "upper_bound_value = 16 : index" not in neura_leaf or
                "step_value = 1 : index" not in neura_leaf):
            fail(f"{name} changed the leaf K counter instead of the relay")
        parts[name] = (values["shard_lower"], values["shard_upper"])

    facts = json.loads(facts_path.read_text())
    if (facts.get("status") != "complete" or
            facts.get("source_iteration_domain_verified") is not True or
            facts.get("cost_catalog_read") is not False or
            facts.get("prediction_run") is not False or
            facts.get("native_mapping_run") is not False or
            facts.get("ranking_performed") is not False):
        fail("Task_8 action replay did not finish source proof only")
    return {
        "output_axis": 2,
        "output_counter_axes": [4294967295, 0, 1],
        "selected_counter_axis": 1,
        "replica_regions": {
            name: [lower, upper] for name, (lower, upper) in parts.items()
        },
        "leaf_counter_domain": [0, 16],
        "source_iteration_domain_verified": True,
        "prediction_or_mapping_run": False,
    }


def fail(message: str) -> None:
    raise RuntimeError(message)


def assert_active_signature(row: dict[str, Any], arguments: list[int],
                            search_output: Path) -> None:
    key = row.get("candidate_key", "")
    marker = "|active-transfer-proof:"
    if len(key.encode("utf-8")) > 4096:
        fail(f"{row.get('candidate_id')} candidate key embeds bulky proof text")
    if marker not in key:
        fail(f"{row.get('candidate_id')} lacks the active proof witness suffix")
    witness_id = key.split(marker, 1)[1]
    if not re.fullmatch(r"active-transfer-\d+", witness_id):
        fail(f"{row.get('candidate_id')} has a malformed active proof witness ID")
    witness_path = search_output / "witnesses" / f"{witness_id}.txt"
    witness_text = witness_path.read_text()
    prefix = PROOF_SCHEMA + "\n"
    if not witness_text.startswith(prefix):
        fail(f"{row.get('candidate_id')} active proof witness has the wrong schema")
    pieces = re.split(r"argument=(\d+);bytes=\d+:", witness_text[len(prefix):])
    records = {int(pieces[index]): pieces[index + 1]
               for index in range(1, len(pieces) - 1, 2)}
    for argument in arguments:
        proof = records.get(argument)
        if proof is None or 'status = "proven"' not in proof:
            fail(f"candidate {row.get('candidate_id')} lacks a proven refreshed arg{argument} record")
        if shape_attribute(proof, "active_shape") != EXPECTED_GCN_ACTIVE_SHAPES.get(argument):
            fail(f"candidate {row.get('candidate_id')} has the wrong proof box for arg{argument}")
        if shape_attribute(proof, "capacity_shape") != EXPECTED_GCN_CAPACITIES.get(argument):
            fail(f"candidate {row.get('candidate_id')} proof changed capacity for arg{argument}")


def assert_interned_witnesses(rows: list[dict[str, Any]],
                              search_output: Path) -> None:
    marker = "|active-transfer-proof:"
    referenced: set[str] = set()
    for row in rows:
        key = row.get("candidate_key", "")
        if marker not in key:
            fail(f"{row.get('candidate_id')} has no interned active-proof key")
        referenced.add(key.split(marker, 1)[1])

    witness_dir = search_output / "witnesses"
    paths = sorted(witness_dir.glob("active-transfer-*.txt"))
    files = {path.stem: path for path in paths}
    if not referenced.issubset(files):
        fail("candidate keys reference missing active-transfer witness files")
    payloads = [path.read_bytes() for path in paths]
    if len(payloads) != len(set(payloads)):
        fail("identical active-transfer proof witnesses were interned more than once")
    for path in list(witness_dir.glob("graph-*.txt")) + \
                list(witness_dir.glob("body-*.txt")):
        if "amoeba.active_transfer_proof" in path.read_text(errors="replace"):
            fail(f"{path.name} duplicates proof metadata inside a graph/body witness")


def shape_attribute(segment: str, name: str) -> tuple[int, ...] | None:
    match = re.search(re.escape(name) + r"\s*=\s*array<i64:\s*([^>]*)>",
                      segment)
    if not match:
        return None
    values = [piece.strip() for piece in match.group(1).split(",")]
    try:
        return tuple(int(value) for value in values if value)
    except ValueError:
        fail(f"malformed {name} shape attribute")


def assert_gcn_active_metadata(row: dict[str, Any], arguments: list[int],
                               function: str) -> None:
    candidate_path = row.get("candidate_path")
    if not candidate_path:
        fail(f"candidate {row.get('candidate_id')} has no replay module")
    module_text = Path(candidate_path).read_text()
    for argument in arguments:
        expected_active = EXPECTED_GCN_ACTIVE_SHAPES.get(argument)
        expected_capacity = EXPECTED_GCN_CAPACITIES.get(argument)
        if expected_active is None or expected_capacity is None:
            fail(f"no exact GCN expectation for requested argument {argument}")
        segment = argument_segment(module_text, function, argument)
        if shape_attribute(segment, "amoeba.active_transfer_shape") != expected_active:
            fail(f"candidate {row.get('candidate_id')} has the wrong active box for arg{argument}")
        if shape_attribute(segment, "amoeba.logical_transfer_shape") != expected_capacity or \
           shape_attribute(segment, "amoeba.active_transfer_capacity_shape") != expected_capacity:
            fail(f"candidate {row.get('candidate_id')} changed the original capacity for arg{argument}")


def candidate_rows(path: Path) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(path)
            if row.get("record_type") == "candidate" and row.get("valid")]


def compare_scored_rows(first: Path, second: Path) -> None:
    def comparable(path: Path) -> list[tuple[Any, ...]]:
        return [
            (row.get("candidate_key"), row.get("predicted_whole_program_cycles"),
             row.get("task_choices"), row.get("task_schedule"),
             row.get("dispatch_order"), row.get("replayed_communication_edges"),
             row.get("communication_trace_known"), row.get("action_path"),
             row.get("action_history"))
            for row in candidate_rows(path)
        ]
    if comparable(first) != comparable(second):
        fail(f"searches produced different scored rows: {first.parent.name} vs {second.parent.name}")


def assert_typed_action_history(row: dict[str, Any], *, known: bool = True) -> None:
    history = row.get("action_history")
    if not isinstance(history, dict) or \
            history.get("schema") != "orbit-joint-neighborhood-typed-actions-v1":
        fail(f"{row.get('candidate_id')} omitted the typed action-history schema")
    if history.get("known") is not known:
        fail(f"{row.get('candidate_id')} has unexpected typed history known={history.get('known')}")
    if not known:
        if not history.get("unknownReason"):
            fail(f"{row.get('candidate_id')} did not explain unknown action history")
        return
    if not history.get("canonicalFactKey") or \
            not isinstance(history.get("initialShapes"), list) or \
            not history["initialShapes"]:
        fail(f"{row.get('candidate_id')} history lacks its canonical and shape base")
    for shape in history["initialShapes"]:
        if set(shape) != {"task", "rows", "cols"}:
            fail(f"{row.get('candidate_id')} has malformed initial shape history")
    for action in history.get("actions", []):
        if set(action) != {"family", "label", "primitives", "shapeTask",
                           "shapeRows", "shapeCols", "canonicalReset"}:
            fail(f"{row.get('candidate_id')} has a malformed typed action")
        for primitive in action["primitives"]:
            if set(primitive) != {"kind", "firstTask", "secondTask", "axis",
                                 "factor", "mode"}:
                fail(f"{row.get('candidate_id')} has a malformed typed primitive")


def compare_checkpoint_frontiers(first: Path, second: Path) -> None:
    left = json.loads(first.read_text())
    right = json.loads(second.read_text())
    for field in ("scored_candidates", "rejected_candidates",
                  "round_scored_candidates", "round_score_limit",
                  "pending_action_cursor", "archive_count", "round"):
        if left.get(field) != right.get(field):
            fail(f"checkpoint resume changed {field}: {left.get(field)} vs {right.get(field)}")
    for field in ("beam", "generated"):
        def states(root: dict[str, Any]) -> list[tuple[Any, ...]]:
            return [(state.get("candidate_key"), state.get("graph_facts_key"),
                     state.get("shapes"), state.get("action_path"),
                     state.get("action_history")) for state in root.get(field, [])]
        if states(left) != states(right):
            fail(f"checkpoint resume changed the {field} frontier")
    def pending(root: dict[str, Any]) -> list[tuple[Any, ...]]:
        return [(entry.get("parent"), entry.get("action"))
                for entry in root.get("pending_neighbors", [])]
    if pending(left) != pending(right):
        fail("checkpoint resume changed deterministic pending action order")
    def archive_rows(checkpoint: Path) -> list[tuple[Any, ...]]:
        return [(row.get("candidate_id"), row.get("candidate_key"),
                 row.get("graph_facts_key"), row.get("valid"),
                 row.get("predicted_whole_program_cycles"), row.get("shapes"),
                 row.get("action_path"), row.get("action_history"),
                 row.get("reject_reason"), row.get("round"))
                for row in read_jsonl(checkpoint.parent / "archive.jsonl")]
    if archive_rows(first) != archive_rows(second):
        fail("checkpoint resume changed its exact archive history or ranking inputs")
    def top5_rows(checkpoint: Path) -> list[tuple[Any, ...]]:
        rows = [row for row in read_jsonl(checkpoint.parent / "top5.jsonl")
                if row.get("record_type") == "selection"]
        return [(row.get("candidate_key"),
                 row.get("predicted_whole_program_cycles"),
                 row.get("shapes"), row.get("task_choices"),
                 row.get("task_schedule"), row.get("dispatch_order"),
                 row.get("replayed_communication_edges"),
                 row.get("communication_trace_known"),
                 row.get("action_path"), row.get("action_history"))
                for row in rows]
    if top5_rows(first) != top5_rows(second):
        fail("checkpoint resume changed the globally ranked top-five archive")
    if left.get("round_scored_candidates", 0) > left.get("round_score_limit", 0):
        fail("checkpoint exceeded its common per-round scheduler-score quota")
    if left.get("production_scheduler_calls") != left.get("scored_candidates"):
        fail("checkpoint budget differs from actual production scheduler calls")


def assert_balanced_pending_order(checkpoint_path: Path) -> None:
    checkpoint = json.loads(checkpoint_path.read_text())
    pairs = [(entry["parent"], entry["action"]["family"])
             for entry in checkpoint.get("pending_neighbors", [])]
    first_positions: dict[tuple[int, str], int] = {}
    for index, pair in enumerate(pairs):
        first_positions.setdefault(pair, index)
    if not first_positions:
        fail("checkpoint has no pending action proposals to check")
    complete_cycle = max(first_positions.values())
    if len(set(pairs[:complete_cycle + 1])) != complete_cycle + 1:
        fail("a parent/family stream repeated before every stream had a proposal")


def stale_proof_module(source: Path, output: Path) -> None:
    text = source.read_text()
    proof_start = text.find("amoeba.active_transfer_proof = {")
    if proof_start < 0:
        fail("baseline candidate has no active-transfer proof record")
    match = re.search(r"(active_shape\s*=\s*array<i64:\s*)[^>]*(>)",
                      text[proof_start:])
    if not match:
        fail("baseline candidate does not have a proven active shape to corrupt")
    start = proof_start + match.start()
    text = (text[:start] + match.group(1) + "4, 5" + match.group(2) +
            text[proof_start + match.end():])
    output.write_text(text)


def nonpositive_trip_count_module(source: Path, output: Path) -> None:
    text = source.read_text()
    marker = "taskflow.task @Task_0"
    start = text.find(marker)
    if start < 0:
        marker = 'task_name = "Task_0"'
        start = text.find(marker)
        if start < 0:
            fail("metadata-unknown seed module has no Task_0 operation")
        end = text.find('"taskflow.task"', start + len(marker))
        if end < 0:
            end = len(text)
        segment = text[start:end]
        footers = list(re.finditer(r'^\s*\}\) \{', segment, re.MULTILINE))
        if not footers:
            fail("generic Task_0 lacks a task attribute dictionary")
        insertion = start + footers[-1].end()
        text = text[:insertion] + "trip_count = 0 : i64, " + text[insertion:]
        output.write_text(text)
        return
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    type_separator = line.find(" : (", start - line_start)
    if type_separator < 0 or "trip_count" in line:
        fail("cannot inject the targeted Task_0 metadata failure")
    line = (line[:type_separator] + " {trip_count = 0 : i64}" +
            line[type_separator:])
    output.write_text(text[:line_start] + line + text[line_end:])


def mutate_typed_action_history(source: dict[str, Any], output: Path) -> dict[str, Any]:
    mutated = json.loads(json.dumps(source))
    history = mutated.get("action_history")
    if not isinstance(history, dict) or not history.get("known") or \
            not history.get("actions"):
        fail("no authenticated action history available for mutation fixture")
    action = history["actions"][0]
    if action.get("shapeTask"):
        action["shapeRows"] = 99
        action["shapeCols"] = 99
    elif action.get("primitives"):
        action["primitives"][0]["factor"] = 999
    else:
        fail("first typed action has no mutable semantic parameter")
    output.write_text(json.dumps(mutated) + "\n")
    return mutated


def generic_argument_spans(module_text: str, function: str) -> list[tuple[int, int]]:
    matches = list(re.finditer(
        r'^.*"func\.func"\(\).*sym_name = "' + re.escape(function) +
        r'".*$', module_text, re.MULTILINE))
    if len(matches) != 1:
        fail(f"expected one generic function signature for {function}")
    header = matches[0].group(0)
    marker = "arg_attrs = ["
    start = header.find(marker)
    if start < 0:
        fail("generic function lacks argument attributes")
    start += len(marker)
    entries = []
    entry_start = start
    depth = 0
    quoted = False
    escaped = False
    for position in range(start, len(header)):
        character = header[position]
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
        elif character == "]" and depth == 0:
            entries.append((matches[0].start() + entry_start,
                            matches[0].start() + position))
            return entries
        elif character in "[{<":
            depth += 1
        elif character in "]}>":
            depth -= 1
        elif character == "," and depth == 0:
            entries.append((matches[0].start() + entry_start,
                            matches[0].start() + position))
            entry_start = position + 1
    fail("generic function has unterminated argument attributes")


def argument_segment(module_text: str, function: str, argument: int) -> str:
    lines = [line for line in module_text.splitlines()
             if f"func.func @{function}(" in line]
    if not lines:
        entries = generic_argument_spans(module_text, function)
        if argument < 0 or argument >= len(entries):
            fail(f"missing generic function argument {argument}")
        start, end = entries[argument]
        return module_text[start:end]
    if len(lines) != 1:
        fail(f"expected one function signature for {function}")
    line = lines[0]
    start = line.find(f"%arg{argument}:")
    if start < 0:
        fail(f"missing function argument {argument}")
    next_marker = line.find(f"%arg{argument + 1}:", start + 1)
    end = next_marker if next_marker >= 0 else line.find(") attributes {", start)
    if end < 0:
        fail(f"cannot find the end of argument {argument} attributes")
    return line[start:end]


def unknown_transfer_alias_module(source: Path, output: Path,
                                     function: str, argument: int) -> None:
    text = source.read_text()
    if f"func.func @{function}(" in text:
        segment = argument_segment(text, function, argument)
        start = text.index(segment)
        end = start + len(segment)
    else:
        spans = generic_argument_spans(text, function)
        start, end = spans[argument]
        segment = text[start:end]
    updated, count = re.subn(
        r',\s*amoeba\.noalias\b',
        '', segment)
    if count != 1:
        fail("unknown-alias fixture must remove exactly one real no-alias attribute")
    output.write_text(text[:start] + updated + text[end:])



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--parent-cost-file", type=Path)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--function", default=DEFAULT_FUNCTION)
    parser.add_argument("--active-arguments", default=",".join(str(i) for i in range(1, 13)))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--diversity-slots", type=int, default=1)
    parser.add_argument("--max-partition-factor", type=int, default=4)
    parser.add_argument("--minimum-free-bytes", type=int, default=0)
    parser.add_argument("--local-action-candidates", type=int, default=80)
    options = parser.parse_args()

    args = options
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.optimizer.is_file() or not args.canonical.is_file():
        fail("optimizer or canonical input is missing")
    if not args.model and not args.parent_cost_file:
        fail("supply --model or --parent-cost-file for production scoring")
    proof_args = [int(piece) for piece in args.active_arguments.split(",")]
    results: dict[str, Any] = {}

    serial_dir = args.output_root / "serial"
    serial = run_search(args, "serial", stage="shape-only", output=serial_dir,
                        active_arguments=args.active_arguments, workers=1,
                        max_candidates=6)
    if serial.returncode:
        fail(f"serial search failed ({serial.returncode}); see {serial_dir / 'serial.log'}")
    serial_rows = candidate_rows(serial_dir / "archive.jsonl")
    if not serial_rows:
        fail("serial search produced no valid candidates")
    for row in serial_rows:
        assert_typed_action_history(row)
    assert_balanced_pending_order(serial_dir / "checkpoint.json")
    if args.max_rounds >= 2:
        scored_rounds = {row.get("round") for row in serial_rows}
        if len(scored_rounds) < 2:
            fail("tight common score budget did not reach two search depths")
        checkpoint = json.loads((serial_dir / "checkpoint.json").read_text())
        if checkpoint.get("round_scored_candidates", 0) > \
                checkpoint.get("round_score_limit", 0) or \
           checkpoint.get("scored_candidates", 0) > 6:
            fail("multi-round search exceeded its common scheduler-score budget")
    binding = json.loads((serial_dir / "checkpoint.json.binding.json").read_text())
    bound_proof = binding.get("active_transfer_proof")
    if not bound_proof or \
       bound_proof.get("schema") != PROOF_SCHEMA or \
       bound_proof.get("arguments_option") != args.active_arguments or \
       bound_proof.get("arguments") != proof_args or \
       bound_proof.get("require_proven") is not True or \
       bound_proof.get("witness_encoding") != "exact-byte-interned-active-transfer-v1":
        fail("checkpoint binding omitted exact active-proof options/schema")
    for row in serial_rows:
        assert_active_signature(row, proof_args, serial_dir)
        if row.get("candidate_path"):
            assert_gcn_active_metadata(row, proof_args, args.function)
    assert_interned_witnesses(serial_rows, serial_dir)
    serial_identity = next((row for row in read_jsonl(serial_dir / "controls.jsonl")
                            if row.get("control_role") == "identity"), None)
    if not serial_identity or not serial_identity.get("candidate_path"):
        fail("serial search omitted its identity module proof witness")
    assert_typed_action_history(serial_identity)
    if serial_identity["action_history"].get("actions"):
        fail("fresh identity history unexpectedly contains neighborhood actions")
    assert_gcn_active_metadata(serial_identity, proof_args, args.function)

    default_off_dir = args.output_root / "default-off"
    default_off = run_search(args, "default-off", stage="shape-only",
                             output=default_off_dir, active_arguments="",
                             workers=1, max_candidates=1)
    if default_off.returncode:
        fail(f"default-off compatibility search failed ({default_off.returncode})")
    default_off_binding = json.loads(
        (default_off_dir / "checkpoint.json.binding.json").read_text())
    if "active_transfer_proof" in default_off_binding:
        fail("default-off search unexpectedly bound active-transfer proof options")
    default_off_identity = next((row for row in read_jsonl(default_off_dir / "controls.jsonl")
                                 if row.get("control_role") == "identity"), None)
    if not default_off_identity or not default_off_identity.get("candidate_path"):
        fail("default-off search omitted its identity candidate snapshot")
    if "active_transfer_proof" in default_off_identity.get("candidate_key", ""):
        fail("default-off candidate key unexpectedly includes an active-proof witness")
    default_off_module = Path(default_off_identity["candidate_path"]).read_text()
    if "amoeba.active_transfer_shape" in default_off_module or \
       "amoeba.active_transfer_proof" in default_off_module:
        fail("default-off identity unexpectedly materialized active-transfer proof metadata")
    results["default_off_compatibility"] = "no active-proof binding, key suffix, or metadata"

    parallel_dir = args.output_root / "parallel"
    parallel = run_search(args, "parallel", stage="shape-only", output=parallel_dir,
                          active_arguments=args.active_arguments, workers=2,
                          max_candidates=6)
    if parallel.returncode:
        fail(f"parallel search failed ({parallel.returncode}); see {parallel_dir / 'parallel.log'}")
    compare_scored_rows(serial_dir / "archive.jsonl", parallel_dir / "archive.jsonl")
    parallel_rows = candidate_rows(parallel_dir / "archive.jsonl")
    for row in parallel_rows:
        assert_typed_action_history(row)
        assert_active_signature(row, proof_args, parallel_dir)
    assert_interned_witnesses(parallel_rows, parallel_dir)
    results["serial_parallel_equal"] = len(serial_rows)

    checkpoint_dir = args.output_root / "checkpoint-resume"
    paused = run_search(args, "paused", stage="shape-only", output=checkpoint_dir,
                        active_arguments=args.active_arguments, workers=1,
                        max_candidates=6, pause_after=2)
    if paused.returncode:
        fail(f"checkpoint seed run failed ({paused.returncode})")
    resumed = run_search(args, "resumed", stage="shape-only", output=checkpoint_dir,
                         active_arguments=args.active_arguments, workers=1,
                         max_candidates=6, resume=True)
    if resumed.returncode:
        fail(f"exact active-transfer checkpoint restore failed ({resumed.returncode})")
    compare_scored_rows(serial_dir / "archive.jsonl", checkpoint_dir / "archive.jsonl")
    compare_checkpoint_frontiers(serial_dir / "checkpoint.json",
                                 checkpoint_dir / "checkpoint.json")
    resumed_rows = candidate_rows(checkpoint_dir / "archive.jsonl")
    for row in resumed_rows:
        assert_typed_action_history(row)
        assert_active_signature(row, proof_args, checkpoint_dir)
    assert_interned_witnesses(resumed_rows, checkpoint_dir)
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    exact_checkpoint = checkpoint_path.read_bytes()
    mutated_checkpoint = json.loads(exact_checkpoint)
    mutated_checkpoint["round_score_limit"] += 1
    checkpoint_path.write_text(json.dumps(mutated_checkpoint) + "\n")
    try:
        mutated_quota = run_search(
            args, "mutated-checkpoint-quota", stage="shape-only",
            output=checkpoint_dir, active_arguments=args.active_arguments,
            workers=1, max_candidates=6, resume=True)
    finally:
        checkpoint_path.write_bytes(exact_checkpoint)
    if mutated_quota.returncode == 0 or \
            "checkpoint per-round score quota differs from protocol-derived budget" \
            not in mutated_quota.stdout:
        fail("resume accepted a genuine checkpoint with a mutated score quota")
    changed = run_search(args, "changed-binding", stage="shape-only",
                         output=checkpoint_dir, active_arguments=",".join(
                             str(value) for value in proof_args[:-1]), workers=1,
                         max_candidates=6, resume=True)
    if changed.returncode == 0 or "binding mismatch" not in changed.stdout:
        fail("checkpoint accepted a changed active-transfer argument binding")
    binding_path = checkpoint_dir / "checkpoint.json.binding.json"
    exact_binding = binding_path.read_bytes()
    binding_path.write_bytes(exact_binding + b"\n")
    forged = run_search(args, "forged-checkpoint-binding", stage="shape-only",
                        output=checkpoint_dir,
                        active_arguments=args.active_arguments, workers=1,
                        max_candidates=6, resume=True)
    binding_path.write_bytes(exact_binding)
    if forged.returncode == 0 or "exact byte/options binding mismatch" not in forged.stdout:
        fail("resume accepted a forged exact-byte checkpoint binding")
    results["checkpoint_restore"] = (
        "passed; changed arguments and forged exact-byte binding rejected")

    controls = read_jsonl(serial_dir / "controls.jsonl")
    identity = next((row for row in controls
                     if row.get("control_role") == "identity"), None)
    if not identity or not identity.get("candidate_path"):
        fail("serial search omitted its identity candidate snapshot")
    stale_path = args.output_root / "stale-seed.mlir"
    stale_proof_module(Path(identity["candidate_path"]), stale_path)
    stale_selection = dict(identity)
    stale_selection["candidate_path"] = str(stale_path)
    stale_selection["record_type"] = "control"
    stale_path_list = args.output_root / "stale-previous-winner.jsonl"
    stale_path_list.write_text(json.dumps(stale_selection) + "\n")
    stale_dir = args.output_root / "stale-seed"
    stale = run_search(args, "stale-seed", stage="shape-temporal",
                       output=stale_dir, active_arguments=args.active_arguments,
                       workers=1, max_candidates=3,
                       previous_winner=stale_path_list)
    if stale.returncode == 0 or "stale or malformed facts" not in stale.stdout:
        fail("search did not reject an imported seed with stale proof metadata")
    results["stale_seed"] = "rejected before scoring the seed"

    unknown_dir = args.output_root / "unknown-required"
    unknown_canonical = args.output_root / "unknown-alias-canonical.mlir"
    # Canonical GCN arg13 now has a genuine positive coordinate proof. Remove
    # only its no-alias evidence to test the required-unknown branch without
    # changing task bodies, counters, source certificates, storage roots or capacity.
    unknown_transfer_alias_module(args.canonical, unknown_canonical,
                                     args.function, 13)
    unknown_args = argparse.Namespace(**vars(args))
    unknown_args.canonical = unknown_canonical
    unknown = run_search(unknown_args, "unknown-required", stage="shape-only",
                         output=unknown_dir, active_arguments="13", workers=1,
                         max_candidates=1)
    if unknown.returncode == 0 or "active-transfer proof is unknown for required argument 13" not in unknown.stdout:
        fail("required unknown argument 13 was not rejected as unsupported")
    fallback_dir = args.output_root / "unknown-capacity-fallback"
    fallback = run_search(unknown_args, "unknown-capacity-fallback", stage="shape-only",
                          output=fallback_dir, active_arguments="13",
                          require_proven=False, workers=1, max_candidates=1)
    if fallback.returncode:
        fail(f"explicit unknown-proof capacity fallback failed ({fallback.returncode})")
    fallback_control = next((row for row in read_jsonl(fallback_dir / "controls.jsonl")
                             if row.get("control_role") == "identity"), None)
    if not fallback_control or not fallback_control.get("candidate_path"):
        fail("unknown-proof fallback omitted its identity module")
    fallback_module = Path(fallback_control["candidate_path"]).read_text()
    argument13 = argument_segment(fallback_module, args.function, 13)
    if "amoeba.logical_transfer_shape" not in argument13 or \
       "amoeba.active_transfer_capacity_shape" not in argument13 or \
       "amoeba.active_transfer_proof" not in argument13 or \
       'status = "unknown"' not in argument13 or \
       "amoeba.active_transfer_shape" in argument13:
        fail("unknown proof did not retain capacity without publishing active shape")
    if shape_attribute(argument13, "amoeba.logical_transfer_shape") != EXPECTED_GCN_CAPACITIES[13] or \
       shape_attribute(argument13, "amoeba.active_transfer_capacity_shape") != EXPECTED_GCN_CAPACITIES[13]:
        fail("unknown arg13 proof changed its original capacity")
    results["unknown_policy"] = "required rejects; explicit fallback retains capacity"

    metadata_selection = next((row for row in read_jsonl(serial_dir / "top5.jsonl")
                               if row.get("record_type") == "selection" and
                               row.get("candidate_path")), None)
    if not metadata_selection:
        fail("serial native shortlist has no module for metadata rejection fixture")
    metadata_module = args.output_root / "metadata-unknown-seed.mlir"
    nonpositive_trip_count_module(Path(metadata_selection["candidate_path"]),
                                  metadata_module)
    metadata_selection = dict(metadata_selection)
    metadata_selection["candidate_path"] = str(metadata_module)
    metadata_seed = args.output_root / "metadata-unknown-previous-winner.jsonl"
    metadata_seed.write_text(json.dumps(metadata_selection) + "\n")
    metadata_dir = args.output_root / "metadata-unknown-seed"
    metadata_unknown = run_search(
        args, "metadata-unknown-seed", stage="shape-only",
        output=metadata_dir, active_arguments="", workers=1,
        max_candidates=1, previous_winner=metadata_seed)
    metadata_reasons = [(
        "previous winner canonical facts unknown: task_metadata_unknown:"
        "task Task_0 has non-positive trip_count"), (
        "previous winner canonical facts unknown: source_iteration_domain_unproven:"
        "task Task_0 has trip_count=0 but its current effective mapper-firing count is 25")]
    expected_metadata_reason = next((reason for reason in metadata_reasons
                                     if reason in metadata_unknown.stdout), None)
    if metadata_unknown.returncode == 0 or expected_metadata_reason is None:
        fail("unknown task metadata rejection omitted its concrete reason")
    results["metadata_unknown_reason"] = expected_metadata_reason

    local_dir = args.output_root / "replica-tiling-refresh"
    local = run_search(args, "local-actions", stage="shape-temporal-replica-tiling",
                       output=local_dir, active_arguments=args.active_arguments,
                       workers=1, max_candidates=args.local_action_candidates,
                       previous_winner=serial_dir / "top5.jsonl")
    if local.returncode:
        fail(f"replica/tiling neighborhood run failed ({local.returncode})")
    local_archive_rows = read_jsonl(local_dir / "archive.jsonl")
    local_rows = [row for row in local_archive_rows
                  if row.get("record_type") == "candidate" and row.get("valid")]
    for row in local_rows:
        assert_typed_action_history(row)
    imported_rows = read_jsonl(local_dir / "controls.jsonl")
    previous_rows = read_jsonl(serial_dir / "top5.jsonl")
    previous_winner = next((row for row in previous_rows
                            if row.get("record_type") == "selection" and
                            row.get("action_history", {}).get("known") and
                            row.get("action_history", {}).get("actions")), None)
    imported_winner = next((row for row in imported_rows
                            if row.get("control_role") ==
                            "previous_stage_measured_winner" and
                            row.get("candidate_key") ==
                            (previous_winner or {}).get("candidate_key")), None)
    if not previous_winner or not imported_winner:
        fail("local search did not retain its imported previous-winner control")
    assert_typed_action_history(imported_winner)
    if imported_winner.get("action_history") != previous_winner.get("action_history"):
        fail("imported previous winner did not preserve its authenticated typed history")
    mutated_history_seed = args.output_root / "mutated-action-history.jsonl"
    mutated_history = mutate_typed_action_history(
        previous_winner, mutated_history_seed)
    mutated_history_dir = args.output_root / "mutated-action-history"
    mutated_history_run = run_search(
        args, "mutated-action-history", stage="shape-temporal-replica-tiling",
        output=mutated_history_dir, active_arguments="", workers=1,
        max_candidates=8, previous_winner=mutated_history_seed)
    if mutated_history_run.returncode:
        fail("mutated typed history prevented retaining its legal source seed")

    def candidate_state_identity(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("graph_facts_key"), row.get("shapes"),
                row.get("task_choices"))

    mutated_state_identity = candidate_state_identity(mutated_history)
    if not all(mutated_state_identity):
        fail("mutated-history seed omitted its source graph/shape state identity")
    mutated_control = next(
        (row for row in read_jsonl(mutated_history_dir / "controls.jsonl")
         if candidate_state_identity(row) == mutated_state_identity and
         "previous_stage_measured_winner" in row.get("control_roles", [])),
        None)
    if not mutated_control:
        fail("mutated-history source candidate was not retained as a control")

    # Active-transfer proof is part of candidate_key. The seed has a verified
    # proof suffix, while this run intentionally disables that proof protocol;
    # compare the source graph/shape state above and keep the raw identities
    # explicitly distinct instead of treating the protocols as equivalent.
    proof_marker = "|active-transfer-proof:"
    mutation_binding = json.loads(
        (mutated_history_dir / "checkpoint.json.binding.json").read_text())
    source_key = mutated_history.get("candidate_key", "")
    control_key = mutated_control.get("candidate_key", "")
    if proof_marker not in source_key or proof_marker in control_key or \
       "active_transfer_proof" in mutation_binding or source_key == control_key:
        fail("mutated-history control did not retain its distinct active-proof protocol identity")
    results["mutated_history_control"] = (
        "same graph/shapes/tasks retained; source proof key and proof-free import binding remain distinct")

    assert_typed_action_history(mutated_control, known=False)
    if not mutated_control["action_history"].get("unknownReason", "").startswith(
            "typed_action_history_replay_failed:"):
        fail("mutated typed history did not fail closed with its replay reason")
    mutated_archive_rows = read_jsonl(mutated_history_dir / "archive.jsonl")
    replacement_attempts = [
        row for row in mutated_archive_rows
        if any("lineage-replacement:replace-lineage:" in action
               for action in row.get("action_path", []))]
    if not replacement_attempts:
        fail("mutated-history fixture produced no lineage replacement attempt")
    replacement_rejection_kinds = {
        row.get("reject_reason", "").partition(":")[0]
        for row in replacement_attempts
    }
    expected_rejection_kinds = {
        "lineage_replacement_requires_authenticated_path_replay",
        "lineage_replacement_target_has_no_authenticated_producer",
    }
    if any(row.get("valid") for row in replacement_attempts) or not all(
            row.get("reject_reason", "").startswith(
                ("lineage_replacement_requires_authenticated_path_replay:",
                 "lineage_replacement_target_has_no_authenticated_producer:"))
            for row in replacement_attempts) or \
       replacement_rejection_kinds != expected_rejection_kinds:
        fail("unknown mutated history authorized a lineage replacement")
    results["mutated_history_rejections"] = {
        "count": len(replacement_attempts),
        "invalid_only": True,
        "concrete_reasons": sorted(replacement_rejection_kinds),
    }
    if not any(any(action.get("family") in {"replica", "tiling", "replica-tiling"}
                   for action in row["action_history"].get("actions", []))
               for row in local_rows if row["action_history"].get("known")):
        fail("local search did not export a typed replica/tiling decision")
    replica_rows = [row for row in local_rows
                    if any("replica" in path.lower() for path in row.get("action_path", []))]
    tiling_rows = [row for row in local_rows
                   if any("til" in path.lower() for path in row.get("action_path", []))]
    if not replica_rows or not tiling_rows:
        task9_axis2_rejection = next((
            row.get("reject_reason", "") for row in read_jsonl(local_dir / "archive.jsonl")
            if any("replica:Task_9:axis=2:factor=2" in action
                   for action in row.get("action_path", []))), "")
        if "active-transfer proof record does not match re-derived IR facts" in task9_axis2_rejection:
            fail("known-supported Task_9 axis=2 replica was rejected because its active-transfer proof became stale")
        fail("local-action budget did not produce both scored replica and tiling candidates")
    task9_replica_rows = [
        row for row in replica_rows
        if any("replica:Task_9:axis=2:factor=2" in action
               for action in row.get("action_path", []))]
    if not task9_replica_rows:
        fail("local-action budget did not score the known-supported Task_9 axis=2 replica")
    task8_action = "replica:Task_8:axis=2:factor=2"
    task9_action = "replica:Task_9:axis=2:factor=2"
    task8_pair = next(((index, row) for index, row in enumerate(local_archive_rows)
                       if row.get("valid") and
                       row.get("action_history", {}).get("known") and
                       [action.get("label") for action in
                        row.get("action_history", {}).get("actions", [])] ==
                       [task8_action]), None)
    task9_pair = next(((index, row) for index, row in enumerate(local_archive_rows)
                       if row.get("valid") and
                       any(task9_action in action
                           for action in row.get("action_path", []))), None)
    if not task8_pair:
        fail("local-action fixture did not score the authenticated Task_8 output-axis replica")
    task8_output_proof = verify_task8_output_shard_replay(
        args, local_dir, task8_pair[1])
    results["task8_output_coordinate_proof"] = task8_output_proof
    if not task9_pair or task8_pair[0] >= task9_pair[0]:
        fail("search did not score the supported Task_9 replica after Task_8")

    unsupported_tile_action = "tiling:tile:Task_0:axis=0:factor=2"
    tile_rejection = next(((index, row) for index, row in
                           enumerate(local_archive_rows)
                           if not row.get("valid") and
                           any(action.startswith(unsupported_tile_action + "::")
                               for action in row.get("action_path", []))), None)
    if (not tile_rejection or
            tile_rejection[1].get("reject_reason") !=
            "source_owned_materializer_rejected:output counter domain is outside the proven shape"):
        fail("known Task_0 tiling proposal did not retain its exact out-of-shape rejection")
    if tile_rejection[0] >= task9_pair[0]:
        fail("search did not continue to Task_9 after the retained Task_0 tiling rejection")
    results["unsupported_tiling_rejection"] = {
        "candidate_id": tile_rejection[1].get("candidate_id"),
        "action": unsupported_tile_action,
        "reason": tile_rejection[1].get("reject_reason"),
        "invalid_only": True,
        "continued_to_task9": True,
    }
    for row in replica_rows + tiling_rows:
        assert_active_signature(row, proof_args, local_dir)
        if row.get("candidate_path"):
            assert_gcn_active_metadata(row, proof_args, args.function)
    if args.max_rounds >= 2:
        def families(row: dict[str, Any]) -> set[str]:
            return {action.get("family", "")
                    for action in row.get("action_history", {}).get("actions", [])}
        def tile_actions(row: dict[str, Any]) -> list[dict[str, Any]]:
            return [action for action in row.get("action_history", {}).get(
                "actions", [])
                    if "tiling" in action.get("family", "")]
        replacement_parent = next((row for row in local_rows
                                   if row.get("candidate_path") and
                                   row.get("action_history", {}).get("known") and
                                   "shape" in families(row) and
                                   "replica" in families(row) and
                                   tile_actions(row)), None)
        if not replacement_parent:
            fail("multi-round fixture did not retain a shape-plus-replica-plus-tiling parent")
        parent_history = replacement_parent["action_history"]
        parent_record = dict(replacement_parent)
        parent_record["record_type"] = "selection"
        parent_record["best_found"] = True
        replacement_seed = args.output_root / "shape-replica-history.jsonl"
        replacement_seed.write_text(json.dumps(parent_record) + "\n")
        replacement_dir = args.output_root / "lineage-replacement"
        replacement_run = run_search(
            args, "lineage-replacement", stage="shape-temporal-replica-tiling",
            output=replacement_dir, active_arguments=args.active_arguments,
            workers=1, max_candidates=args.local_action_candidates,
            previous_winner=replacement_seed)
        if replacement_run.returncode:
            fail("authenticated lineage replacement run failed")
        replacement_rows = [row for row in read_jsonl(
            replacement_dir / "archive.jsonl")
            if row.get("valid") and row.get("action_history", {}).get("known") and
            any("lineage-replacement:replace-lineage:" in path and
                "replica" in path.lower() for path in row.get("action_path", []))]
        shape_actions = [action for action in parent_history.get("actions", [])
                         if action.get("family") == "shape"]
        parent_replica_count = sum(
            action.get("family") == "replica"
            for action in parent_history.get("actions", []))
        replacement = next((row for row in replacement_rows
                            if any(action in row.get("action_history", {}).get(
                                "actions", []) for action in shape_actions) and
                            any(action in row.get("action_history", {}).get(
                                "actions", []) for action in
                                tile_actions(replacement_parent)) and
                            sum(action.get("family") == "replica"
                                for action in row["action_history"]["actions"])
                            < parent_replica_count), None)
        if not replacement:
            fail("lineage replacement did not remove one replica while preserving independent tile and shape choices")
        assert_typed_action_history(replacement)
        if replacement["action_history"].get("initialShapes") != \
                parent_history.get("initialShapes"):
            fail("lineage replacement changed the authenticated seed shape base")
        results["selective_lineage_replacement"] = {
            "parent": replacement_parent.get("candidate_id"),
            "candidate": replacement.get("candidate_id"),
            "preserved_shape_actions": sum(
                action.get("family") == "shape"
                for action in replacement["action_history"]["actions"]),
            "preserved_tiling_actions": sum(
                "tiling" in action.get("family", "")
                for action in replacement["action_history"]["actions"]),
            "remaining_replica_actions": sum(
                action.get("family") == "replica"
                for action in replacement["action_history"]["actions"]),
        }
    assert_interned_witnesses(local_rows, local_dir)
    rejected_replica_reasons: dict[str, int] = {}
    for row in local_archive_rows:
        if row.get("valid") or not any(
                "replica" in action.lower() for action in row.get("action_path", [])):
            continue
        reason = row.get("reject_reason", "unknown rejection")
        rejected_replica_reasons[reason] = rejected_replica_reasons.get(reason, 0) + 1
    results["local_refresh"] = {
        "tiling_candidates": len(tiling_rows),
        "replica_candidates": len(replica_rows),
        "task8_output_coordinate_proof": task8_output_proof,
        "unsupported_tiling_diagnostic": tile_rejection[1].get("reject_reason"),
        "rejected_replica_proposals": sum(rejected_replica_reasons.values()),
        "replica_rejection_reasons": rejected_replica_reasons,
    }

    summary = {"status": "pass", "checks": results,
               "output_root": str(args.output_root.resolve())}
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ACTIVE TRANSFER NEIGHBORHOOD CHECKS FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
