#!/usr/bin/env python3
"""Exercise source-owned replica-coordinate and fusion proofs on Harris MLIR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


TASK_OPS = ("taskflow.task", "neura.task")
GENERIC_TASK_NAME = re.compile(
    r'\btask_name\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
PRETTY_TASK = re.compile(r"\s+@([^\s(\[]+)")


def fail(message: str) -> None:
    raise SystemExit(f"replica-coordinate/fusion fixture: {message}")


def task_starts(ir: str) -> list[tuple[int, str | None]]:
    """Find task operations without mistaking serialized source for live IR."""
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
            match = GENERIC_TASK_NAME.search(segment)
            name = match.group(1) if match else None
        if name is not None:
            spans.append((name, begin, end, segment))
    return spans


def task_segment(ir: str, name: str) -> tuple[int, int, str]:
    for task_name, begin, end, segment in task_spans(ir):
        if task_name == name:
            return begin, end, segment
    fail(f"canonical input has no task {name}")


def replace_in_task(ir: str, name: str, old: str, new: str) -> str:
    begin, end, segment = task_segment(ir, name)
    if old not in segment:
        fail(f"{name} fixture anchor not found: {old}")
    return ir[:begin] + segment.replace(old, new, 1) + ir[end:]


def get_task_shard_ranges(ir: str, parent: str) -> dict[int, tuple[int, int]]:
    found: dict[int, tuple[int, int]] = {}
    marker = re.compile(rf"{re.escape(parent)}\.replica\.(\d+)$")
    for name, _, _, segment in task_spans(ir):
        match = marker.fullmatch(name)
        if not match:
            continue
        lower = re.search(r"amoeba\.replica\.shard_lower\s*=\s*(-?\d+)\s*:\s*i64",
                          segment)
        upper = re.search(r"amoeba\.replica\.shard_upper\s*=\s*(-?\d+)\s*:\s*i64",
                          segment)
        if not lower or not upper:
            fail(f"{name} lacks exact replica shard bounds")
        found[int(match.group(1))] = (int(lower.group(1)), int(upper.group(1)))
    return found


def run_command(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    log_path.write_text(result.stdout)
    return result


def verify_input(optimizer: Path, source: Path, output_dir: Path,
                 label: str) -> None:
    verified = output_dir / f"{label}.verified.mlir"
    result = run_command(
        [str(optimizer), str(source), "--verify-each", "-o", str(verified)],
        output_dir / f"{label}.verify.log")
    if result.returncode != 0:
        fail(f"negative fixture {label} is not valid MLIR; see {verified}.verify.log")


def run_pass(optimizer: Path, source: Path, output: Path, pass_name: str,
             pass_options: str) -> subprocess.CompletedProcess[str]:
    return run_command(
        [str(optimizer), str(source), "--verify-each",
         f"--{pass_name}={pass_options}", "-o", str(output)],
        output.with_suffix(output.suffix + ".log"))


def require_proof_rejection(result: subprocess.CompletedProcess[str],
                            output: Path, case: str) -> str:
    if result.returncode == 0:
        fail(f"{case} unexpectedly passed its source proof")
    diagnostics = [line.strip() for line in result.stdout.splitlines()
                   if "error:" in line]
    if not diagnostics:
        fail(f"{case} failed without a concrete compiler diagnostic; see {output}.log")
    diagnostic = diagnostics[-1]
    parse_failures = ("failed to parse", "expected operation", "expected ')'",
                      "custom op verification", "error: expected")
    if any(marker in diagnostic.lower() for marker in parse_failures):
        fail(f"{case} reached a parser/verifier error instead of its proof gate: {diagnostic}")
    return diagnostic


def make_replay_action(label: str, family: str, primitive: dict[str, object]) -> dict[str, object]:
    return {
        "family": family,
        "label": label,
        "shapeTask": "",
        "shapeRows": 0,
        "shapeCols": 0,
        "primitives": [primitive],
    }


def run_replay(optimizer: Path, canonical: Path, original: str,
               output_dir: Path, function: str, action: dict[str, object],
               label: str) -> tuple[Path, dict[str, object]]:
    inputs = output_dir / f"{label}-inputs"
    inputs.mkdir(parents=True)
    candidate = inputs / "candidate-input.mlir"
    candidate.write_text(original)
    actions = inputs / "actions.json"
    action_data = {
        "schema": "orbit-joint-neighborhood-typed-actions-v1",
        "canonicalInput": str(canonical.resolve()),
        "candidateInput": str(candidate.resolve()),
        "function": function,
        "stage": "full-joint",
        "maxPartitionFactor": 4,
        "actions": [action],
    }
    actions.write_text(json.dumps(action_data, indent=2) + "\n")
    replay_dir = output_dir / f"{label}-replay"
    mlir_output = output_dir / f"{label}-replay-output.mlir"

    def quoted(value: Path | str) -> str:
        return json.dumps(str(value.resolve() if isinstance(value, Path) else value))

    options = " ".join((
        f"action-file={quoted(actions)}",
        f"canonical-input={quoted(canonical)}",
        f"candidate-input={quoted(candidate)}",
        f"function={quoted(function)}",
        "stage=full-joint",
        "max-partition-factor=4",
        f"output-dir={quoted(replay_dir)}",
    ))
    result = run_pass(optimizer, candidate, mlir_output,
                      "replay-joint-neighborhood-actions", options)
    if result.returncode != 0:
        fail(f"positive {label} guarded replay failed; see {mlir_output}.log")
    facts_path = replay_dir / "source-facts.json"
    if not facts_path.is_file():
        fail(f"{label} replay did not publish source-facts.json")
    facts = json.loads(facts_path.read_text())
    if (facts.get("schema") != "orbit-joint-neighborhood-action-replay-facts-v1"
            or facts.get("status") != "complete"
            or facts.get("source_iteration_domain_verified") is not True):
        fail(f"{label} replay did not publish complete source-domain facts")
    steps = facts.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        fail(f"{label} replay did not apply exactly one typed action")
    step = steps[0]
    if (step.get("label") != action["label"]
            or step.get("status") != "applied"
            or step.get("source_partition_proof") != "verified-and-refreshed"):
        fail(f"{label} action lacks refreshed source-partition proof")
    return replay_dir, facts


def run_replay_expect_rejection(optimizer: Path, canonical: Path,
                                original: str, output_dir: Path,
                                function: str, action: dict[str, object],
                                label: str, expected_diagnostic: str) -> str:
    inputs = output_dir / f"{label}-inputs"
    inputs.mkdir(parents=True)
    candidate = inputs / "candidate-input.mlir"
    candidate.write_text(original)
    verify_input(optimizer, candidate, output_dir, f"{label}-input")
    actions = inputs / "actions.json"
    actions.write_text(json.dumps({
        "schema": "orbit-joint-neighborhood-typed-actions-v1",
        "canonicalInput": str(canonical.resolve()),
        "candidateInput": str(candidate.resolve()),
        "function": function,
        "stage": "full-joint",
        "maxPartitionFactor": 4,
        "actions": [action],
    }, indent=2) + "\n")
    replay_dir = output_dir / f"{label}-replay"
    mlir_output = output_dir / f"{label}-replay-output.mlir"

    def quoted(value: Path | str) -> str:
        return json.dumps(str(value.resolve() if isinstance(value, Path) else value))

    options = " ".join((
        f"action-file={quoted(actions)}",
        f"canonical-input={quoted(canonical)}",
        f"candidate-input={quoted(candidate)}",
        f"function={quoted(function)}",
        "stage=full-joint",
        "max-partition-factor=4",
        f"output-dir={quoted(replay_dir)}",
    ))
    result = run_pass(optimizer, candidate, mlir_output,
                      "replay-joint-neighborhood-actions", options)
    diagnostic = require_proof_rejection(result, mlir_output, label)
    if expected_diagnostic not in diagnostic:
        fail(f"{label} failed at an unexpected source guard: {diagnostic}")
    if replay_dir.exists():
        fail(f"{label} published partial replay output despite source-proof rejection")
    return diagnostic


def require_task_fact(facts: dict[str, object], name: str) -> dict[str, object]:
    task_facts = facts.get("tasks")
    if not isinstance(task_facts, list):
        fail("replay facts do not contain task source-domain rows")
    matches = [task for task in task_facts
               if isinstance(task, dict) and task.get("task") == name]
    if len(matches) != 1:
        fail(f"expected exactly one refreshed source fact for {name}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True,
                        help="corrected source-domain Harris canonical MLIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.optimizer.is_file() or not args.canonical.is_file():
        fail("optimizer or canonical input is missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    canonical = args.canonical.resolve()
    original = canonical.read_text()
    function_match = re.search(r'\bsym_name\s*=\s*"([^"\\]+)"', original)
    if not function_match:
        fail("canonical module has no generic function symbol")
    function = function_match.group(1)
    results: dict[str, object] = {}

    replica_label = "replica:Task_21:axis=1:factor=4"
    replica_action = make_replay_action(
        replica_label, "replica", {
            "kind": "replica", "firstTask": "Task_21", "secondTask": "",
            "mode": "", "axis": 1, "factor": 4,
        })
    replica_dir, replica_facts = run_replay(
        args.optimizer, canonical, original, args.output_dir, function,
        replica_action, "harris-task21-factor4")
    replica_module = (replica_dir / "candidate.mlir").read_text()
    expected_ranges = {0: (1, 33), 1: (33, 65), 2: (65, 96), 3: (96, 127)}
    module_ranges = get_task_shard_ranges(replica_module, "Task_21")
    if module_ranges != expected_ranges:
        fail(f"guarded Task_21 factor-four intervals differ: {module_ranges!r}")
    fact_ranges: dict[int, tuple[int, int]] = {}
    source_work = 0
    taskflow_firings = 0
    counter_zero_ranges: set[tuple[int, int]] = set()
    for shard, expected in expected_ranges.items():
        fact = require_task_fact(replica_facts, f"Task_21.replica.{shard}")
        if fact.get("complete") is not True:
            fail(f"Task_21 replica {shard} lacks a complete source-domain fact")
        if fact.get("source_multiplicity") != 7686:
            fail(f"Task_21 replica {shard} changed the original 7686-work source certificate")
        if fact.get("current_output_coordinate_status") != "proven":
            fail(f"Task_21 replica {shard} has unknown current output-coordinate proof")
        current_axes = fact.get("current_counter_axes")
        if not isinstance(current_axes, list):
            fail(f"Task_21 replica {shard} has unknown current counter bounds")
        selected = [axis for axis in current_axes
                    if isinstance(axis, dict) and axis.get("ordinal") == 1]
        if len(selected) != 1:
            fail(f"Task_21 replica {shard} lacks exactly one current counter axis 1")
        axis = selected[0]
        interval = (axis.get("lower"), axis.get("upper"))
        if (interval != expected or axis.get("step") != 1
                or axis.get("bound_status") != "proven"):
            fail(f"Task_21 replica {shard} current counter axis 1 differs: {axis!r}")
        unchanged_axis = [candidate for candidate in current_axes
                          if isinstance(candidate, dict)
                          and candidate.get("ordinal") == 0]
        if (len(unchanged_axis) != 1
                or unchanged_axis[0].get("bound_status") != "proven"
                or not isinstance(unchanged_axis[0].get("lower"), int)
                or not isinstance(unchanged_axis[0].get("upper"), int)
                or unchanged_axis[0]["upper"] <= unchanged_axis[0]["lower"]):
            fail(f"Task_21 replica {shard} lacks its current unsharded counter bound")
        counter_zero_ranges.add((unchanged_axis[0].get("lower"),
                                 unchanged_axis[0].get("upper")))
        if not fact.get("partition_proof"):
            fail(f"Task_21 replica {shard} has no refreshed partition proof")
        current_work = fact.get("current_source_work_count")
        current_firing_count = fact.get("current_taskflow_firing_count")
        if not isinstance(current_work, int) or current_work <= 0:
            fail(f"Task_21 replica {shard} has unknown current source work")
        if not isinstance(current_firing_count, int) or current_firing_count <= 0:
            fail(f"Task_21 replica {shard} has unknown current Taskflow firing count")
        source_work += current_work
        taskflow_firings += current_firing_count
        fact_ranges[shard] = interval
    if source_work != 7686:
        fail(f"Task_21 current-domain facts cover {source_work} source iterations, expected 7686")
    if taskflow_firings <= 0 or len(counter_zero_ranges) != 1:
        fail("Task_21 current firing count or common unsharded-axis bounds are inconsistent")
    results["harris_task21_factor4"] = {
        "status": "pass", "ranges_from_module": module_ranges,
        "ranges_from_current_counter_facts": fact_ranges,
        "original_source_certificate_multiplicity_per_shard": 7686,
        "current_source_work": source_work,
        "current_taskflow_firings": taskflow_firings,
        "replay_facts": str(replica_dir / "source-facts.json"),
    }

    fusion_label = "fuse:Task_22:Task_23:producer-consumer-retained"
    fusion_action = make_replay_action(
        fusion_label, "fusion", {
            "kind": "fusion", "firstTask": "Task_22", "secondTask": "Task_23",
            "mode": "producer-consumer-retained", "axis": 0, "factor": 1,
        })
    fused_name = "Task_22.fuse.Task_23"
    fusion_materialized = args.output_dir / "harris-task22-task23-materialized.mlir"
    materializer_result = run_pass(
        args.optimizer, canonical, fusion_materialized,
        "materialize-neura-joint-rewrite",
        "first-task-name=Task_22 second-task-name=Task_23 "
        "fusion-mode=producer-consumer-retained")
    if materializer_result.returncode != 0:
        fail("Task_22 to Task_23 retained fusion materializer failed; see log")
    verify_input(args.optimizer, fusion_materialized, args.output_dir,
                 "retained-fusion-materializer-output")
    fused_module = fusion_materialized.read_text()
    if fused_name not in {name for name, _, _, _ in task_spans(fused_module)}:
        fail("retained fusion did not materialize the derived Task_22/Task_23 task")
    fusion_diagnostic = run_replay_expect_rejection(
        args.optimizer, canonical, original, args.output_dir, function,
        fusion_action, "harris-task22-task23-fusion-source-guard",
        f"child task {fused_name} is neither an unchanged parent task nor a proved replica")
    results["task22_task23_retained_fusion"] = {
        "status": "materialized-source-proof-unsupported",
        "materializer_status": "pass",
        "source_guard_status": "rejected-as-unsupported",
        "task": fused_name,
        "diagnostic": fusion_diagnostic,
        "materialized_module": str(fusion_materialized),
    }

    # These are valid input0-derived MLIR mutations. Verify each one in a
    # no-transform invocation before asking a source-owned proof to reject it.
    _, _, task21 = task_segment(original, "Task_21")
    constant = re.search(
        r'(?m)^(?P<indent>[ \t]*)(?P<const>%[\w.$-]+) = "arith\.constant"\(\) '
        r'<\{value = -1 : index\}> : \(\) -> index\s*$', task21)
    if not constant:
        fail("Task_21 arithmetic fixture constant anchor not found")
    add = re.search(
        rf'(?m)^(?P<indent>[ \t]*)(?P<result>%[\w.$-]+) = "arith\.addi"\('
        rf'(?P<lhs>%[\w.$-]+), (?P<rhs>{re.escape(constant.group("const"))})\)', task21)
    if not add:
        fail("Task_21 arithmetic fixture add anchor not found")

    # The proof walker must treat repeated operands as two independently
    # resolvable paths, not as a cycle or globally visited SSA value.
    repeated = replace_in_task(
        original, "Task_21", "value = -1 : index", "value = 31 : index")
    repeated = replace_in_task(
        repeated, "Task_21", '"arith.addi"(%arg49, %77)',
        '"arith.addi"(%77, %77)')
    repeated_input = args.output_dir / "positive-repeated-static-operand.mlir"
    repeated_input.write_text(repeated)
    verify_input(args.optimizer, repeated_input, args.output_dir,
                 "repeated-static-operand")
    repeated_output = args.output_dir / "positive-repeated-static-operand-out.mlir"
    repeated_result = run_pass(
        args.optimizer, repeated_input, repeated_output,
        "materialize-joint-task-replicas", "task=Task_21 replicas=4 axis=1")
    if repeated_result.returncode != 0:
        fail("Task_21 rejected a repeated independently static SSA operand")
    repeated_ranges = get_task_shard_ranges(repeated_output.read_text(), "Task_21")
    if repeated_ranges != expected_ranges:
        fail(f"repeated-operand proof changed Task_21 shard intervals: {repeated_ranges!r}")
    results["repeated_static_operand"] = {"status": "pass", "ranges": repeated_ranges}

    # A folded counter attribute cannot override a changed static bound.
    spoof = replace_in_task(
        original, "Task_21", 'upper_bound_value = "%input4"',
        "upper_bound_value = 61 : index")
    spoof_input = args.output_dir / "negative-static-attribute-spoof.mlir"
    spoof_input.write_text(spoof)
    verify_input(args.optimizer, spoof_input, args.output_dir, "static-attribute-spoof")
    spoof_output = args.output_dir / "negative-static-attribute-spoof-out.mlir"
    spoof_result = run_pass(
        args.optimizer, spoof_input, spoof_output,
        "materialize-joint-task-replicas", "task=Task_21 replicas=4 axis=1")
    spoof_diagnostic = require_proof_rejection(spoof_result, spoof_output,
                                                "static attribute spoof")
    results["static_attribute_spoof"] = {"status": "rejected", "diagnostic": spoof_diagnostic}

    overflow = replace_in_task(
        original, "Task_21", "value = -1 : index",
        "value = 9223372036854775807 : index")
    overflow_input = args.output_dir / "negative-index-overflow.mlir"
    overflow_input.write_text(overflow)
    verify_input(args.optimizer, overflow_input, args.output_dir, "index-overflow")
    overflow_output = args.output_dir / "negative-index-overflow-out.mlir"
    overflow_result = run_pass(
        args.optimizer, overflow_input, overflow_output,
        "materialize-joint-task-replicas", "task=Task_21 replicas=4 axis=1")
    overflow_diagnostic = require_proof_rejection(overflow_result, overflow_output,
                                                   "index overflow")
    results["index_overflow"] = {"status": "rejected", "diagnostic": overflow_diagnostic}

    block_args = re.search(r"\^bb0\((.*?)\):", task21, re.S)
    if not block_args:
        fail("Task_21 task block arguments are not present")
    arguments = re.findall(r"(%[\w.$-]+)\s*:\s*([^,]+)", block_args.group(1))
    dynamic_i32 = next((name for name, type_name in arguments
                        if type_name.strip() == "i32"), None)
    index_argument = next((name for name, type_name in arguments
                           if type_name.strip() == "index"), None)
    if not dynamic_i32 or not index_argument:
        fail("Task_21 does not expose the typed runtime i32/index operands needed by negatives")

    def rewrite_add_operand(rhs: str, replacement: str, label: str) -> Path:
        segment = task21
        segment = segment.replace(
            f'{constant.group("indent")}{constant.group("const")} = "arith.constant"() '
            '<{value = -1 : index}> : () -> index',
            f'{constant.group("indent")}{constant.group("const")} = {replacement}', 1)
        segment = re.sub(
            rf'(?m)^(?P<indent>[ \t]*){re.escape(add.group("result"))} = "arith\.addi"\('
            rf'{re.escape(add.group("lhs"))}, {re.escape(rhs)}\)',
            rf'\g<indent>{add.group("result")} = "arith.addi"('
            rf'{add.group("lhs")}, {constant.group("const")})', segment, count=1)
        if segment == task21:
            fail(f"{label} did not update Task_21 arithmetic")
        begin, end, _ = task_segment(original, "Task_21")
        module = original[:begin] + segment + original[end:]
        path = args.output_dir / f"{label}.mlir"
        path.write_text(module)
        verify_input(args.optimizer, path, args.output_dir, label)
        return path

    runtime_input = rewrite_add_operand(
        add.group("rhs"),
        f'"arith.index_cast"({dynamic_i32}) : (i32) -> index',
        "negative-runtime-index-value")
    runtime_output = args.output_dir / "negative-runtime-index-value-out.mlir"
    runtime_result = run_pass(
        args.optimizer, runtime_input, runtime_output,
        "materialize-joint-task-replicas", "task=Task_21 replicas=4 axis=1")
    runtime_diagnostic = require_proof_rejection(runtime_result, runtime_output,
                                                  "runtime index value")
    results["runtime_index_value"] = {"status": "rejected", "diagnostic": runtime_diagnostic}

    narrow_name = "%harris_fixture_narrow_bound"
    if narrow_name in original:
        fail("narrow-cast fixture SSA name unexpectedly collides with the input")
    segment = task21
    constant_text = (
        f'{constant.group("indent")}{constant.group("const")} = "arith.constant"() '
        '<{value = -1 : index}> : () -> index')
    narrow_text = (
        f'{constant.group("indent")}{narrow_name} = "arith.index_cast"({index_argument}) '
        ': (index) -> i32\n'
        f'{constant.group("indent")}{constant.group("const")} = "arith.index_cast"({narrow_name}) '
        ': (i32) -> index')
    if constant_text not in segment:
        fail("narrow-cast fixture constant anchor not found")
    segment = segment.replace(constant_text, narrow_text, 1)
    segment = re.sub(
        rf'(?m)^(?P<indent>[ \t]*){re.escape(add.group("result"))} = "arith\.addi"\('
        rf'{re.escape(add.group("lhs"))}, {re.escape(add.group("rhs"))}\)',
        rf'\g<indent>{add.group("result")} = "arith.addi"('
        rf'{add.group("lhs")}, {constant.group("const")})', segment, count=1)
    begin, end, _ = task_segment(original, "Task_21")
    narrow_module = original[:begin] + segment + original[end:]
    narrow_input = args.output_dir / "negative-narrow-index-cast.mlir"
    narrow_input.write_text(narrow_module)
    verify_input(args.optimizer, narrow_input, args.output_dir, "narrow-index-cast")
    narrow_output = args.output_dir / "negative-narrow-index-cast-out.mlir"
    narrow_result = run_pass(
        args.optimizer, narrow_input, narrow_output,
        "materialize-joint-task-replicas", "task=Task_21 replicas=4 axis=1")
    narrow_diagnostic = require_proof_rejection(narrow_result, narrow_output,
                                                 "narrow index cast")
    results["narrow_index_cast"] = {"status": "rejected", "diagnostic": narrow_diagnostic}

    # Counter/access mutations are also verifier-clean before the fusion proof
    # is invoked, so they exercise the actual source guard rather than syntax.
    _, _, task23 = task_segment(original, "Task_23")
    load = re.search(r'("neura\.load_indexed"\()(%[\w.$-]+)(, )(%[\w.$-]+)(\))', task23)
    if not load:
        fail("Task_23 indexed-load fixture anchor not found")
    bad_task23 = (task23[:load.start()] + load.group(1) + load.group(4)
                  + load.group(3) + load.group(4) + load.group(5)
                  + task23[load.end():])
    begin, end, _ = task_segment(original, "Task_23")
    access_module = original[:begin] + bad_task23 + original[end:]
    access_input = args.output_dir / "negative-fusion-access-index.mlir"
    access_input.write_text(access_module)
    verify_input(args.optimizer, access_input, args.output_dir, "fusion-access-index")
    access_output = args.output_dir / "negative-fusion-access-index-out.mlir"
    access_result = run_pass(
        args.optimizer, access_input, access_output,
        "materialize-neura-joint-rewrite",
        "first-task-name=Task_22 second-task-name=Task_23 "
        "fusion-mode=producer-consumer-retained")
    access_diagnostic = require_proof_rejection(access_result, access_output,
                                                 "mismatched fusion access index")
    results["mismatched_access_counter"] = {"status": "rejected", "diagnostic": access_diagnostic}

    counter_old = re.compile(
        r'(?m)("neura\.counter"\(\)[^\n]*counter_id\s*=\s*)1(\s*:\s*i32)')
    counter_begin, counter_end, counter_segment = task_segment(original, "Task_23")
    counter_match = counter_old.search(counter_segment)
    if not counter_match:
        fail("Task_23 Neura counter-ID fixture anchor not found")
    bad_counter_task = counter_old.sub(
        r"\g<1>3\g<2>", counter_segment, count=1)
    counter_module = original[:counter_begin] + bad_counter_task + original[counter_end:]
    counter_input = args.output_dir / "negative-fusion-counter-id.mlir"
    counter_input.write_text(counter_module)
    verify_input(args.optimizer, counter_input, args.output_dir, "fusion-counter-id")
    counter_output = args.output_dir / "negative-fusion-counter-id-out.mlir"
    counter_result = run_pass(
        args.optimizer, counter_input, counter_output,
        "materialize-neura-joint-rewrite",
        "first-task-name=Task_22 second-task-name=Task_23 "
        "fusion-mode=producer-consumer-retained")
    counter_diagnostic = require_proof_rejection(counter_result, counter_output,
                                                  "mismatched Neura counter IDs")
    results["mismatched_counter_id"] = {"status": "rejected", "diagnostic": counter_diagnostic}

    report = {
        "schema": "orbit-replica-coordinate-fusion-proof-runtime-v2",
        "scope": "source-owned replay/materializer proof fixtures; no mapper invocation",
        "optimizer": str(args.optimizer.resolve()),
        "canonical": str(canonical),
        "results": results,
    }
    report_path = args.output_dir / "result.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
