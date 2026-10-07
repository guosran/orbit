#!/usr/bin/env python3
"""Check source-owned sibling-fusion proof and fail-closed tamper cases."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile


FUNCTION = "sibling_source_domain"
SCHEMA = "orbit-joint-neighborhood-typed-actions-v1"


def fail(message: str) -> None:
    raise SystemExit(message)


def run(command: list[str], expect_success: bool) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if (result.returncode == 0) != expect_success:
        fail(
            f"unexpected exit status {result.returncode}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def task_span(text: str, task_name: str) -> tuple[int, int]:
    # Ignore quoted source witnesses when locating the actual region braces.
    masked = list(text)
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            masked[index] = " " if char != "\n" else "\n"
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            masked[index] = " "
            quoted = True
    plain = "".join(masked)
    start = plain.find(f"taskflow.task @{task_name}")
    if start < 0:
        fail(f"missing task {task_name}")
    signature = re.search(r":\s*\(", plain[start:])
    if not signature:
        fail(f"missing type signature for {task_name}")
    body = plain.find("{", start + signature.start())
    if body < 0:
        fail(f"missing body for {task_name}")
    depth = 0
    for position in range(body, len(plain)):
        if plain[position] == "{":
            depth += 1
        elif plain[position] == "}":
            depth -= 1
            if depth == 0:
                return text.rfind("\n", 0, start) + 1, position + 1
    fail(f"unterminated task body for {task_name}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        fail(f"{label}: expected exactly one occurrence of {old!r}")
    return text.replace(old, new, 1)


def extract_graph_facts(
    optimizer: str, module: pathlib.Path, function: str, output: pathlib.Path
) -> dict[str, object]:
    pipeline = (
        "builtin.module(extract-joint-task-graph-facts{"
        f"function={function} output={output} fact-only=true"
        "})"
    )
    run(
        [optimizer, str(module), f"--pass-pipeline={pipeline}", "-o", "/dev/null"],
        expect_success=True,
    )
    return json.loads(output.read_text())


def mutate_task(text: str, task_name: str, transform, label: str) -> str:
    begin, end = task_span(text, task_name)
    body = transform(text[begin:end])
    if body == text[begin:end]:
        fail(f"{label}: mutation made no change")
    return text[:begin] + body + text[end:]


def typed_fusion_action() -> dict[str, object]:
    return {
        "family": "sibling-fusion",
        "label": "sibling-fuse:Task_A:Task_B",
        "primitives": [
            {
                "kind": "sibling-fusion",
                "firstTask": "Task_A",
                "secondTask": "Task_B",
                "axis": 0,
                "factor": 1,
                "mode": "sibling",
            }
        ],
        "shapeTask": "",
        "shapeRows": 0,
        "shapeCols": 0,
        "canonicalReset": False,
    }


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: run-sibling-fusion-source-domain-checks.py FIXTURE OPTIMIZER")
    fixture = pathlib.Path(sys.argv[1]).resolve()
    optimizer = sys.argv[2]

    with tempfile.TemporaryDirectory(prefix="sibling-fusion-source-domain-") as temp:
        directory = pathlib.Path(temp)
        canonical = directory / "canonical.mlir"
        source_pipeline = (
            "builtin.module(func.func(construct-hyperblock-from-task),"
            "assign-accelerator,classify-task-and-counter,"
            "convert-taskflow-to-neura,lower-affine-to-neura,"
            "bind-source-iteration-domain)"
        )
        run(
            [
                optimizer,
                str(fixture),
                f"--pass-pipeline={source_pipeline}",
                "-o",
                str(canonical),
            ],
            expect_success=True,
        )

        actions = directory / "sibling-fusion.actions.json"
        actions.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "canonicalInput": str(canonical),
                    "candidateInput": str(canonical),
                    "function": FUNCTION,
                    "stage": "full-joint",
                    "maxPartitionFactor": 4,
                    "actions": [typed_fusion_action()],
                },
                indent=2,
            )
            + "\n"
        )
        replay_dir = directory / "sibling-fusion-replay"
        pipeline = (
            "builtin.module(replay-joint-neighborhood-actions{"
            f"action-file={actions} canonical-input={canonical} "
            f"candidate-input={canonical} function={FUNCTION} "
            f"stage=full-joint max-partition-factor=4 output-dir={replay_dir}"
            "})"
        )
        run(
            [
                optimizer,
                str(canonical),
                f"--pass-pipeline={pipeline}",
                "-o",
                "/dev/null",
            ],
            expect_success=True,
        )
        candidate = directory / "candidate-readable.mlir"
        run([optimizer, str(replay_dir / "candidate.mlir"), "--verify-each",
             "-o", str(candidate)], expect_success=True)
        facts = json.loads((replay_dir / "source-facts.json").read_text())
        if facts.get("status") != "complete" or facts.get("actions_applied") != 1:
            fail(f"unexpected complete replay facts: {facts}")
        if facts.get("source_iteration_domain_verified") is not True:
            fail(f"sibling fusion did not pass the source-domain proof: {facts}")
        positive = candidate.read_text()
        if "Task_A.fuse.Task_B" not in positive or "taskflow.task @Task_C" not in positive:
            fail("positive replay did not preserve the fused pair and downstream task")
        run(
            [
                optimizer,
                str(candidate),
                "--verify-source-iteration-domain-partitions="
                f"parent-module={canonical} function={FUNCTION}",
                "-o",
                "/dev/null",
            ],
            expect_success=True,
        )
        graph_facts = extract_graph_facts(
            optimizer,
            candidate,
            FUNCTION,
            directory / "sibling-graph-facts.json",
        )
        fused_record = next(
            (
                task
                for task in graph_facts.get("tasks", [])
                if task.get("task_name") == "Task_A.fuse.Task_B"
            ),
            None,
        )
        if fused_record is None or fused_record.get("neura_fusion") != {
            "mode": "sibling",
            "eliminated_loads": 1,
            "eliminated_stores": 0,
            "sibling_first": "Task_A",
            "sibling_second": "Task_B",
        }:
            fail(
                "fact extraction did not preserve valid sibling shared-read "
                f"metadata: {fused_record}"
            )

        def reject(label: str, mutant: str) -> None:
            path = directory / f"{label}.mlir"
            path.write_text(mutant)
            result = run(
                [
                    optimizer,
                    str(path),
                    "--verify-source-iteration-domain-partitions="
                    f"parent-module={canonical} function={FUNCTION}",
                    "-o",
                    "/dev/null",
                ],
                expect_success=False,
            )
            diagnostics = result.stdout + result.stderr
            if "source iteration-domain partition is unproven" not in diagnostics:
                fail(f"{label}: failed outside the source-proof diagnostic\n{diagnostics}")

        reject(
            "forged-eliminated-load-count",
            replace_once(
                positive,
                "amoeba.neura.fusion.eliminated_loads = 1 : i64",
                "amoeba.neura.fusion.eliminated_loads = 0 : i64",
                "sibling eliminated load count",
            ),
        )

        reject(
            "forged-parent-name",
            replace_once(
                positive,
                'amoeba.neura.fusion.sibling_first = "Task_A"',
                'amoeba.neura.fusion.sibling_first = "Task_X"',
                "forged parent name",
            ),
        )

        def cut_fused_counter(operation: str) -> str:
            changed, count = re.subn(
                r"(?m)^(\s*%[^\n=]+=\s*taskflow.counter[^\n]*?counter_id\s*=\s*)0(\s*: i32)",
                r"\g<1>7\g<2>",
                operation,
                count=1,
            )
            if count != 1:
                fail("cut-body: could not change one fused counter ID")
            return changed

        reject(
            "cut-fused-body-counter",
            mutate_task(positive, "Task_A.fuse.Task_B", cut_fused_counter, "cut body"),
        )

        reject(
            "mismatched-fused-counter",
            mutate_task(
                positive,
                "Task_A.fuse.Task_B",
                lambda operation: re.sub(
                    r"(?m)^(\s*%[^\n=]+=\s*neura.counter[^\n]*?counter_id\s*=\s*)0(\s*: i32)",
                    r"\g<1>7\g<2>", operation, count=1),
                "counter",
            ),
        )

        def reverse_fused_outputs(operation: str) -> str:
            match = re.search(r"will_writes\(([^)]*)\)", operation, re.S)
            if not match:
                fail("output mutation: missing fused will_writes")
            names = list(re.finditer(r"%[A-Za-z0-9_.$-]+", match.group(1)))
            if len(names) != 2:
                fail("output mutation: expected two fused write roots")
            begin_a, end_a = names[0].span()
            begin_b, end_b = names[1].span()
            segment = match.group(1)
            swapped = (
                segment[:begin_a]
                + names[1].group(0)
                + segment[end_a:begin_b]
                + names[0].group(0)
                + segment[end_b:]
            )
            return operation[: match.start(1)] + swapped + operation[match.end(1) :]

        reject(
            "changed-fused-output-order",
            mutate_task(positive, "Task_A.fuse.Task_B", reverse_fused_outputs, "output"),
        )

        def remove_output_noalias(text: str) -> str:
            changed, count = re.subn(
                r"(%[A-Za-z0-9_.$-]+\s*:\s*memref<4xi32>)\s*\{\s*amoeba\.noalias\s*\}",
                r"\1",
                text,
                count=1,
            )
            if count != 1:
                fail("alias-contract mutation: could not remove one output noalias")
            return changed

        reject("changed-alias-contract", remove_output_noalias(positive))

        def redirect_downstream_edge(operation: str) -> str:
            changed, count = re.subn(
                r"(will_reads\()(%[A-Za-z0-9_.$-]+(?:#\d+)?)(?=\s*[, :])",
                r"\1%arg0",
                operation,
                count=1,
            )
            if count != 1:
                fail("downstream completion mutation: could not redirect one state")
            return changed

        reject(
            "changed-downstream-completion-edge",
            mutate_task(positive, "Task_C", redirect_downstream_edge, "completion edge"),
        )

        begin, end = task_span(positive, "Task_C")
        removed_downstream = positive[:begin] + positive[end:]
        reject("dropped-unrelated-task", removed_downstream)

    print("sibling fusion source-domain positive and tamper checks passed")


if __name__ == "__main__":
    main()
