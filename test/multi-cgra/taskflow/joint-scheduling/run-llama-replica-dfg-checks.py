#!/usr/bin/env python3
"""Check the specialized LLaMA replica DFG proof on canonical source IR."""
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
PARSE_FAILURE_MARKERS = (
    "failed to parse", "expected operation", "expected ')'",
    "custom op verification", "error: expected", "invalid type",
)


def fail(message: str) -> None:
    raise SystemExit(f"LLaMA replica DFG fixture: {message}")


def task_starts(ir: str) -> list[tuple[int, str | None]]:
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
    matches = [span for span in task_spans(ir) if span[0] == name]
    if len(matches) != 1:
        fail(f"expected exactly one canonical operation for {name}, found {len(matches)}")
    _, begin, end, segment = matches[0]
    return begin, end, segment


def transform_task(ir: str, name: str, transform) -> str:
    begin, end, segment = task_segment(ir, name)
    updated = transform(segment)
    return ir[:begin] + updated + ir[end:]


def inject_into_kernel(ir: str, task: str, operations: str) -> str:
    def insert(segment: str) -> str:
        position = segment.find('"neura.yield"')
        if position < 0:
            fail(f"{task} has no direct Neura kernel yield insertion point")
        line_start = segment.rfind("\n", 0, position) + 1
        indent = segment[line_start:position]
        text = "\n".join(indent + line for line in operations.splitlines()) + "\n"
        return segment[:line_start] + text + segment[line_start:]
    return transform_task(ir, task, insert)


def inject_before_first_store(ir: str, task: str, operation: str) -> str:
    def insert(segment: str) -> str:
        position = segment.find('"neura.store_indexed"')
        if position < 0:
            fail(f"{task} has no indexed store insertion point")
        line_start = segment.rfind("\n", 0, position) + 1
        indent = segment[line_start:position]
        return (segment[:line_start] + indent + operation + "\n" +
                segment[line_start:])
    return transform_task(ir, task, insert)


def kernel_block_arguments(segment: str, task: str) -> list[str]:
    kernel = segment.find('"neura.kernel"')
    if kernel < 0:
        fail(f"{task} has no Neura kernel")
    block = segment.find("^bb0(", kernel)
    close = segment.find("):", block)
    if block < 0 or close < 0:
        fail(f"{task} kernel has no parseable entry block")
    declaration = segment[block + len("^bb0("):close]
    arguments = re.findall(r"(%[A-Za-z_.$][A-Za-z0-9_.$-]*)\s*:", declaration)
    if len(arguments) < 3:
        fail(f"{task} kernel does not expose output and two read roles")
    return arguments


def redirect_first_store_to_read_input(ir: str, task: str) -> str:
    def redirect(segment: str) -> str:
        kernel = segment.find('"neura.kernel"')
        if kernel < 0:
            fail(f"{task} has no Neura kernel")
        # Task_1's output is input1; input2 is its first read-only matrix.
        # Redirect the actual folded destination, preserving the valid i32
        # stored-value type. Replacing a forwarded input1 value only removed
        # predication from the same output and was not a cross-storage test.
        store = re.search(
            r'"neura\.store_indexed"[^\n]*rhs_value = "(%input1)"',
            segment[kernel:])
        if not store:
            fail(f"{task} has no folded output store to mutate")
        start = kernel + store.start(1)
        end = kernel + store.end(1)
        return segment[:start] + "%input2" + segment[end:]
    return transform_task(ir, task, redirect)


def replace_first_output_store_indices(ir: str, task: str,
                                       index0: str, index1: str) -> str:
    def replace(segment: str) -> str:
        match = re.search(
            r'("neura\.store_indexed"\(\s*%[A-Za-z0-9_.$][A-Za-z0-9_.$-]*\s*,\s*)'
            r'(%[A-Za-z0-9_.$][A-Za-z0-9_.$-]*)(\s*,\s*)'
            r'(%[A-Za-z0-9_.$][A-Za-z0-9_.$-]*)', segment)
        if not match:
            fail(f"{task} has no indexed output store to mutate")
        return (segment[:match.start()] + match.group(1) + index0 + match.group(3) +
                index1 + segment[match.end(4):])
    return transform_task(ir, task, replace)


def run(command: list[str], log: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    return result


def verify_fixture(optimizer: Path, source: Path, output_dir: Path,
                   label: str) -> None:
    output = output_dir / f"{label}.verified.mlir"
    result = run([str(optimizer), str(source), "--verify-each", "-o", str(output)],
                 output_dir / f"{label}.verify.log")
    if result.returncode != 0:
        fail(f"{label} fixture is not valid verified MLIR; see {output}.verify.log")


def materialize(optimizer: Path, source: Path, output_dir: Path, label: str,
                task: str, replicas: int = 2, axis: int = 0
                ) -> tuple[subprocess.CompletedProcess[str], Path]:
    output = output_dir / f"materialized-{label}.mlir"
    pipeline = ("builtin.module(materialize-joint-task-replicas{"
                f"task={task} replicas={replicas} axis={axis}" "})")
    result = run([str(optimizer), str(source), "--verify-each",
                  f"--pass-pipeline={pipeline}", "-o", str(output)],
                 output.with_suffix(".log"))
    return result, output


def replica_count(ir: str, task: str) -> int:
    escaped = re.escape(task)
    return len(re.findall(
        rf'(?:@{escaped}\.replica\.\d+|task_name\s*=\s*"{escaped}\.replica\.\d+")',
        ir))


def expect_success(optimizer: Path, canonical: Path, output_dir: Path,
                   task: str, replicas: int, axis: int) -> None:
    label = f"positive-{task}-factor{replicas}-axis{axis}"
    result, output = materialize(optimizer, canonical, output_dir, label,
                                 task, replicas, axis)
    if result.returncode != 0 or not output.is_file():
        fail(f"supported {label} failed; see {output}.log")
    count = replica_count(output.read_text(), task)
    if count != replicas:
        fail(f"{label} emitted {count} named replicas, expected {replicas}")


def expect_rejection(optimizer: Path, fixture: Path, output_dir: Path,
                     label: str, task: str, reason: str,
                     require_specialized_task_name: bool = True) -> str:
    verify_fixture(optimizer, fixture, output_dir, label)
    result, output = materialize(optimizer, fixture, output_dir, label,
                                 task, 2, 0)
    if result.returncode == 0:
        fail(f"negative {label} unexpectedly materialized")
    diagnostics = [line.strip() for line in result.stdout.splitlines()
                   if "error:" in line]
    if not diagnostics:
        fail(f"negative {label} failed without a compiler diagnostic; see {output}.log")
    diagnostic = diagnostics[-1]
    lowered = diagnostic.lower()
    if any(marker in lowered for marker in PARSE_FAILURE_MARKERS):
        fail(f"negative {label} hit a parse/verifier error instead of its proof gate: "
             f"{diagnostic}")
    if require_specialized_task_name and f"LLaMA {task} " not in diagnostic:
        fail(f"negative {label} omitted actual task name {task}: {diagnostic}")
    if reason not in diagnostic:
        fail(f"negative {label} did not reach expected gate {reason!r}: {diagnostic}")
    return diagnostic


def extract_and_check_coordinate_facts(optimizer: Path, canonical: Path,
                                      function: str, output_dir: Path,
                                      *, corrupted: bool = False) -> None:
    label = "corrupted-coordinate" if corrupted else "canonical"
    facts = output_dir / f"{label}-facts.json"
    output = output_dir / f"{label}-facts-pass-output.mlir"
    pipeline = ("builtin.module(extract-joint-task-graph-facts{"
                f"function={function} output={facts} fact-only=true" "})")
    result = run([str(optimizer), str(canonical), "--verify-each",
                  f"--pass-pipeline={pipeline}", "-o", str(output)],
                 output_dir / f"{label}-facts-pass.log")
    if result.returncode != 0 or not facts.is_file():
        fail("could not extract source facts for unsupported-shape/domain check; "
             "see facts-pass.log")
    data = json.loads(facts.read_text())
    rows = {row.get("task_name"): row for row in data.get("tasks", [])}
    if corrupted:
        row = rows.get("Task_0", {})
        if (row.get("neura_output_coordinate_status") != "unknown" or
                not row.get("neura_output_coordinate_reason")):
            fail("corrupted output-coordinate proof lost its concrete unknown status")
        return
    for task, produced, capacity in (
            ("Task_0", [320, 256], [512, 256]),
            ("Task_1", [320, 256], [512, 256]),
            ("Task_3", [320, 320], [512, 512])):
        row = rows.get(task, {})
        if (row.get("neura_output_coordinate_status") != "proven" or
                row.get("neura_output_counter_map") != [0, 1] or
                row.get("neura_output_produced_shape") != produced or
                row.get("neura_output_caller_shape") != capacity):
            fail(f"output-coordinate proof for {task} lost its exact M/N "
                 "domain or allocation capacity")
    unproven_k = {row.get("task_name"): row.get("reason")
                  for row in data.get("k_reduction_unproven_tasks", [])}
    if not unproven_k.get("Task_8"):
        fail("Task_8's unsupported K-reduction domain lost its concrete reason")


def write_fixture(path: Path, ir: str) -> Path:
    path.write_text(ir)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", required=True, type=Path)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--function", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    optimizer = args.optimizer.resolve()
    canonical = args.canonical.resolve()
    if not optimizer.is_file() or not canonical.is_file():
        fail("optimizer and canonical module must both exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = canonical.read_text()

    # Local static allocations now supply independently proved output
    # capacity. This must preserve the actual M/N domain and must not prove
    # independent ownership for the corrupted-coordinate fixture below.
    # Task_8 retains its independent unproven K-reduction fact.
    extract_and_check_coordinate_facts(optimizer, canonical, args.function,
                                       args.output_dir)

    checks: list[str] = []
    for task in ("Task_0", "Task_1", "Task_3", "Task_8"):
        for replicas in (2, 4, 8):
            for axis in (0, 1):
                expect_success(optimizer, canonical, args.output_dir,
                              task, replicas, axis)
                checks.append(f"{task}:factor{replicas}:axis{axis}")

    negative_fixtures: list[tuple[str, str, str]] = []
    alloc = '%llama_dfg_test_alloc = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xi32>'
    negative_fixtures.append(("unknown-memory-effect", "Task_0",
                              "Neura DFG must store only the private output"))
    write_fixture(args.output_dir / "unknown-memory-effect.mlir",
                  inject_into_kernel(original, "Task_0", alloc))

    negative_fixtures.append(("cross-storage-write", "Task_1",
                              "Neura DFG must store only the private output"))
    write_fixture(args.output_dir / "cross-storage-write.mlir",
                  redirect_first_store_to_read_input(original, "Task_1"))

    bad_type_move = (
        '%llama_dfg_test_bad_move = "neura.data_mov"(%arg117) : '
        '(!neura.data<i32, i1>) -> !neura.data<i64, i1>')
    negative_fixtures.append(("non-forwarding-type-mismatch", "Task_3",
                              "Neura DFG must store only the private output"))
    task3_args = kernel_block_arguments(task_segment(original, "Task_3")[2],
                                        "Task_3")
    write_fixture(args.output_dir / "non-forwarding-type-mismatch.mlir",
                  inject_into_kernel(
                      original, "Task_3",
                      bad_type_move.replace("%arg117", task3_args[0])))

    attributed_move = (
        '%llama_dfg_test_attributed_move = "neura.data_mov"(%arg117) '
        '{test_marker = 1 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>')
    negative_fixtures.append(("attributed-forwarding", "Task_0",
                              "Neura DFG must store only the private output"))
    task0_args = kernel_block_arguments(task_segment(original, "Task_0")[2],
                                        "Task_0")
    write_fixture(args.output_dir / "attributed-forwarding.mlir",
                  inject_into_kernel(
                      original, "Task_0",
                      attributed_move.replace("%arg117", task0_args[0])))

    nested_move = "\n".join((
        '%llama_dfg_test_condition = "arith.constant"() <{value = true}> : () -> i1',
        '"scf.if"(%llama_dfg_test_condition) ({',
        '^bb0:',
        '  %llama_dfg_test_nested_move = "neura.data_mov"(%arg117) : '
        '(!neura.data<i32, i1>) -> !neura.data<i32, i1>',
        '  "scf.yield"() : () -> ()',
        '}, {',
        '^bb0:',
        '  "scf.yield"() : () -> ()',
        '}) : (i1) -> ()',
    ))
    negative_fixtures.append(("cross-block-forwarding", "Task_0",
                              "Neura DFG must store only the private output"))
    write_fixture(args.output_dir / "cross-block-forwarding.mlir",
                  inject_into_kernel(
                      original, "Task_0",
                      nested_move.replace("%arg117", task0_args[0])))

    bad_index = "\n".join((
        '%llama_dfg_test_constant_index = "neura.constant"() <{value = "0"}> : '
        '() -> !neura.data<index, i1>',
    ))
    index_fixture = inject_before_first_store(original, "Task_0", bad_index)
    index_fixture = replace_first_output_store_indices(
        index_fixture, "Task_0", "%llama_dfg_test_constant_index",
        "%llama_dfg_test_constant_index")
    write_fixture(args.output_dir / "non-independent-output-coordinate.mlir",
                  index_fixture)
    extract_and_check_coordinate_facts(
        optimizer, args.output_dir / "non-independent-output-coordinate.mlir",
        args.function, args.output_dir, corrupted=True)

    for label, task, expected in negative_fixtures:
        diagnostic = expect_rejection(
            optimizer, args.output_dir / f"{label}.mlir", args.output_dir,
            label, task, expected)
        checks.append(f"{label}: {diagnostic}")
    coordinate_diagnostic = expect_rejection(
        optimizer, args.output_dir / "non-independent-output-coordinate.mlir",
        args.output_dir, "non-independent-output-coordinate", "Task_0",
        "independent M/N cell ownership", require_specialized_task_name=False)
    checks.append(f"non-independent-output-coordinate: {coordinate_diagnostic}")

    summary = {"schema": "llama-replica-dfg-checks-v1", "checks": checks}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"LLaMA replica DFG checks passed: {len(checks)} checks")


if __name__ == "__main__":
    main()
