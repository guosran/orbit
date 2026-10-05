#!/usr/bin/env python3
"""Focused verifier and dependency checks for GCN replica read fan-in.

The checker never rewrites candidate bodies to make them legal. Its negative
cases mutate one witness in a materialized module or one source access and
require the C++ verifier/materializer to reject the result.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


FACTS_PASS = "extract-joint-task-graph-facts"
MATERIALIZE_PASS = "materialize-joint-task-replicas"


def fail(message: str) -> None:
    raise RuntimeError(message)


def diagnostic_excerpt(output: str, limit: int = 1800) -> str:
    """Keep the first compiler diagnostic instead of a huge witness tail."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "<no diagnostic output>"
    diagnostic = next(
        (line for line in lines if re.search(r"\b(?:error|fatal error):", line,
                                             flags=re.IGNORECASE)),
        lines[0],
    )
    if len(diagnostic) <= limit:
        return diagnostic
    return diagnostic[:limit] + " … <diagnostic line truncated>"


def run_checked(command: list[str], *, should_succeed: bool, label: str,
                diagnostic_tokens: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    combined = result.stdout + result.stderr
    if should_succeed and result.returncode != 0:
        fail(f"{label} failed ({result.returncode}):\n"
             f"{diagnostic_excerpt(combined)}")
    if not should_succeed and result.returncode == 0:
        fail(f"{label} unexpectedly succeeded")
    if not should_succeed and diagnostic_tokens and not any(
            token in combined for token in diagnostic_tokens):
        fail(f"{label} failed without a retained proof diagnostic:\n"
             f"{diagnostic_excerpt(combined)}")
    return result


def pass_pipeline(function: str, task: str, replicas: int, output: Path,
                  facts: Path | None) -> str:
    steps = [
        f"{MATERIALIZE_PASS}{{function={function} task={task} "
        f"replicas={replicas} axis=2}}"
    ]
    if facts is not None:
        steps.append(
            f"{FACTS_PASS}{{function={function} output={facts} fact-only=true}}"
        )
    return "builtin.module(" + ",".join(steps) + ")"


def find_function(canonical: str, task: str) -> str:
    marker = f'task_name = "{task}"'
    task_offset = canonical.find(marker)
    if task_offset < 0:
        pretty_task = re.search(r'\btaskflow\.task\s+@' + re.escape(task) +
                                r'(?=\s)', canonical)
        if pretty_task is None:
            fail(f"canonical fixture does not contain {task}")
        task_offset = pretty_task.start()
    function_start = canonical.rfind('sym_name = "', 0, task_offset)
    if function_start < 0:
        pretty_functions = list(re.finditer(
            r'\bfunc\.func\s+(?:public\s+|private\s+)?@([^\s(]+)',
            canonical[:task_offset]))
        if not pretty_functions:
            fail(f"could not find the enclosing function symbol for {task}")
        return pretty_functions[-1].group(1)
    value_start = function_start + len('sym_name = "')
    value_end = canonical.find('"', value_start)
    if value_end < 0:
        fail(f"could not parse the enclosing function symbol for {task}")
    return canonical[value_start:value_end]


def verify_positive_facts(facts_path: Path, task: str, replicas: int) -> None:
    facts = json.loads(facts_path.read_text())
    read_joins = facts.get("read_completion_joins")
    output_joins = facts.get("completion_joins")
    dependencies = facts.get("dependencies")
    if not isinstance(read_joins, list) or not isinstance(output_joins, list):
        fail("facts omit typed output/read completion join records")
    if not isinstance(dependencies, list):
        fail("facts omit typed dependencies")

    candidates = [entry for entry in read_joins
                  if entry.get("replica_read_states") == replicas]
    if len(candidates) != 1:
        fail(f"{task} factor {replicas} produced {len(candidates)} read joins")
    linked_index = candidates[0].get("linked_output_completion_join")
    if not isinstance(linked_index, int) or not 0 <= linked_index < len(output_joins):
        fail(f"{task} read join is not linked to a recorded output join")
    if output_joins[linked_index].get("producer_tiles") != replicas:
        fail(f"{task} read and output joins cover different replica counts")

    expected_names = {f"{task}.replica.{replica}" for replica in range(replicas)}
    war_sources = {
        edge.get("source_task")
        for edge in dependencies
        if edge.get("kind") == "war"
        and edge.get("producer_segment") == "done_reads"
        and edge.get("consumer_segment") == "will_writes"
    }
    missing = expected_names - war_sources
    if missing:
        fail(f"{task} missing replica-to-writer WAR dependencies: {sorted(missing)}")


def materialize(optimizer: Path, canonical: Path, function: str, task: str,
                replicas: int, output: Path, facts: Path | None,
                *, should_succeed: bool,
                diagnostic_tokens: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    pipeline = pass_pipeline(function, task, replicas, output, facts)
    return run_checked(
        [str(optimizer), f"--pass-pipeline={pipeline}", "--verify-each",
         str(canonical), "-o", str(output)],
        should_succeed=should_succeed,
        label=f"{task} axis2 factor{replicas}",
        diagnostic_tokens=diagnostic_tokens)


def mutate_join_operand(module_text: str, operand_name: str,
                        replacement: str) -> str:
    lines = module_text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if '"taskflow.read_completion_join"' not in line and \
                "taskflow.read_completion_join" not in line:
            continue
        pattern = rf"({operand_name}\s*\(\s*)%[A-Za-z0-9_.$#-]+"
        updated, count = re.subn(pattern, rf"\g<1>{replacement}", line, count=1)
        if count != 1:
            fail(f"could not locate the {operand_name} operand in read join")
        lines[index] = updated
        return "".join(lines)
    fail("materialized module has no read-completion join")


def mutate_first_replica_shard_upper(module_text: str, task: str) -> str:
    marker = f'task_name = "{task}.replica.0"'
    start = module_text.find(marker)
    if start < 0:
        pretty = re.search(r'\btaskflow\.task\s+@' + re.escape(task) +
                           r'\.replica\.0(?=\s)', module_text)
        if pretty is None:
            fail(f"materialized module has no first replica for {task}")
        start = pretty.start()
    match = re.search(
        r"amoeba\.replica\.shard_upper\s*=\s*(-?\d+)",
        module_text[start:],
    )
    if not match:
        fail(f"replica {task}.replica.0 has no shard upper witness")
    old = int(match.group(1))
    begin = start + match.start(1)
    end = start + match.end(1)
    return module_text[:begin] + str(old + 1) + module_text[end:]


def corrupt_source_store_root(canonical: str, task: str) -> str:
    start = canonical.find(f'task_name = "{task}"')
    if start < 0:
        fail(f"canonical fixture does not contain {task}")
    next_task = canonical.find('task_name = "', start + 1)
    end = next_task if next_task >= 0 else len(canonical)
    section = canonical[start:end]
    updated, count = re.subn(
        r'("neura\.store_indexed"[^\n]*\{rhs_value = ")%input1("\})',
        r"\g<1>%input2\g<2>", section, count=1,
    )
    if count != 1:
        fail(f"could not find {task}'s output store for cross-root mutation")
    return canonical[:start] + updated + canonical[end:]


def verify_mutated_module(optimizer: Path, mutant: Path, task: str,
                          function: str, work: Path, label: str,
                          diagnostic_tokens: tuple[str, ...]) -> None:
    output = work / f"negative-{label}.mlir"
    facts = work / f"negative-{label}.json"
    pipeline = (
        f"builtin.module({FACTS_PASS}{{function={function} output={facts} "
        "fact-only=true})"
    )
    run_checked(
        [str(optimizer), f"--pass-pipeline={pipeline}", "--verify-each",
         str(mutant), "-o", str(output)],
        should_succeed=False,
        label=f"negative {label}",
        diagnostic_tokens=diagnostic_tokens,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--skip-negative", action="store_true")
    args = parser.parse_args()
    optimizer = args.optimizer.resolve()
    canonical_path = args.canonical.resolve()
    if not optimizer.is_file() or not canonical_path.is_file():
        fail("optimizer and canonical input must be regular files")
    canonical = canonical_path.read_text()

    temporary = None
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="gcn-read-join-")
        work = Path(temporary.name)

    positive_outputs: dict[tuple[str, int], Path] = {}
    for task in ("Task_14", "Task_20"):
        function = find_function(canonical, task)
        for replicas in (2, 4, 8):
            output = work / f"{task}-axis2-factor{replicas}.mlir"
            facts = work / f"{task}-axis2-factor{replicas}.json"
            materialize(optimizer, canonical_path, function, task, replicas,
                        output, facts, should_succeed=True)
            if "taskflow.read_completion_join" not in output.read_text():
                fail(f"{task} factor {replicas} omitted the typed read join")
            verify_positive_facts(facts, task, replicas)
            positive_outputs[(task, replicas)] = output

    if not args.skip_negative:
        seed = positive_outputs[("Task_14", 2)].read_text()
        mutants = {
            "read-root-mismatch": mutate_join_operand(seed, "root", "%arg11"),
            "base-read-version-mismatch": mutate_join_operand(seed, "base", "%arg13"),
            "child-done-read-state-mismatch": mutate_join_operand(seed, "states", "%arg13"),
            "escaped-shard-index": mutate_first_replica_shard_upper(seed,
                                                                       "Task_14"),
        }
        for label, contents in mutants.items():
            if contents == seed:
                fail(f"negative {label} did not change its proof witness")
            path = work / f"negative-{label}-input.mlir"
            path.write_text(contents)
            tokens = ("read-completion", "read state", "replica", "join")
            verify_mutated_module(optimizer, path, "Task_14",
                                  find_function(canonical, "Task_14"), work, label,
                                  tokens)

        cross_root = work / "negative-cross-root-source.mlir"
        cross_root.write_text(corrupt_source_store_root(canonical, "Task_14"))
        function = find_function(canonical, "Task_14")
        materialize(
            optimizer, cross_root, function, "Task_14", 2,
            work / "negative-cross-root-output.mlir", None,
            should_succeed=False,
            diagnostic_tokens=("output-coordinate", "replica", "memory", "store"),
        )

    print(f"PASS: Task14/20 axis2 factors2/4/8 and WAR fan-in; outputs in {work}")
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
