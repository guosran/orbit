#!/usr/bin/env python3
"""Exercise source-guarded replay through the C++ neighborhood API."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


FUNCTION = "source_partition_lineage"
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


def typed_action(
    family: str,
    label: str,
    *,
    primitive: dict[str, object] | None = None,
    shape_task: str = "",
    rows: int = 0,
    cols: int = 0,
) -> dict[str, object]:
    return {
        "family": family,
        "label": label,
        "primitives": [] if primitive is None else [primitive],
        "shapeTask": shape_task,
        "shapeRows": rows,
        "shapeCols": cols,
        "canonicalReset": False,
    }


def tile(task: str, factor: int = 2, axis: int = 0) -> dict[str, object]:
    return typed_action(
        "tiling",
        f"tile:{task}:axis={axis}:factor={factor}",
        primitive={
            "kind": "tile",
            "firstTask": task,
            "secondTask": "",
            "axis": axis,
            "factor": factor,
            "mode": "none",
        },
    )


def shape(task: str, rows: int, cols: int) -> dict[str, object]:
    return typed_action(
        "shape",
        f"shape:{task}:{rows}x{cols}",
        shape_task=task,
        rows=rows,
        cols=cols,
    )


def identity() -> dict[str, object]:
    return typed_action("identity", "identity")


def write_action_file(
    path: pathlib.Path,
    canonical: pathlib.Path,
    candidate: pathlib.Path,
    actions: list[dict[str, object]],
    cap: int,
    initial_shapes: list[dict[str, object]] | None = None,
) -> None:
    document: dict[str, object] = {
        "schema": SCHEMA,
        "canonicalInput": str(canonical.resolve()),
        "candidateInput": str(candidate.resolve()),
        "function": FUNCTION,
        "stage": "full-joint",
        "maxPartitionFactor": cap,
        "actions": actions,
    }
    if initial_shapes is not None:
        document["initialShapes"] = initial_shapes
    path.write_text(json.dumps(document, indent=2) + "\n")


def replay(
    optimizer: str,
    candidate: pathlib.Path,
    canonical: pathlib.Path,
    actions: list[dict[str, object]],
    directory: pathlib.Path,
    label: str,
    *,
    cap: int = 4,
    initial_shapes: list[dict[str, object]] | None = None,
    success: bool = True,
) -> tuple[subprocess.CompletedProcess[str], pathlib.Path]:
    output = directory / f"{label}.complete"
    action_file = directory / f"{label}.actions.json"
    write_action_file(
        action_file, canonical, candidate, actions, cap, initial_shapes
    )
    pass_options = " ".join(
        [
            f"action-file={action_file}",
            f"canonical-input={canonical.resolve()}",
            f"candidate-input={candidate.resolve()}",
            f"function={FUNCTION}",
            "stage=full-joint",
            f"max-partition-factor={cap}",
            f"output-dir={output}",
        ]
    )
    pipeline = (
        "builtin.module(replay-joint-neighborhood-actions{"
        + pass_options
        + "})"
    )
    result = run(
        [
            optimizer,
            str(candidate),
            f"--pass-pipeline={pipeline}",
            "-o",
            "/dev/null",
        ],
        expect_success=success,
    )
    if success and not (output / "candidate.mlir").is_file():
        fail(f"{label}: complete replay candidate was not published")
    if success:
        for witness, original in (
            ("canonical-input.mlir", canonical),
            ("candidate-input.mlir", candidate),
            ("actions.json", action_file),
        ):
            if (output / witness).read_bytes() != original.read_bytes():
                fail(f"{label}: retained {witness} differs from the exact input bytes")
    if not success and output.exists():
        fail(f"{label}: failed replay published a partial candidate directory")
    return result, output


def assert_complete(output: pathlib.Path, expected_steps: int) -> dict[str, object]:
    facts = json.loads((output / "source-facts.json").read_text())
    if facts.get("status") != "complete" or facts.get("actions_applied") != expected_steps:
        fail(f"unexpected complete replay facts in {output}: {facts}")
    if facts.get("source_iteration_domain_verified") is not True:
        fail(f"replay did not report the source-domain guard: {facts}")
    if (
        facts.get("cost_catalog_read") is not False
        or facts.get("prediction_run") is not False
        or facts.get("native_mapping_run") is not False
        or facts.get("ranking_performed") is not False
    ):
        fail(f"replay facts imply prohibited scoring behavior: {facts}")
    for witness in (
        "canonical-input.mlir",
        "candidate-input.mlir",
        "actions.json",
    ):
        if not (output / witness).is_file():
            fail(f"replay omitted exact input witness {witness}")
    return facts


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: run-joint-neighborhood-action-replay-checks.py FIXTURE OPTIMIZER")
    fixture = pathlib.Path(sys.argv[1]).resolve()
    optimizer = sys.argv[2]
    with tempfile.TemporaryDirectory(prefix="joint-action-replay-") as temp:
        directory = pathlib.Path(temp)
        canonical = directory / "canonical.mlir"
        run(
            [
                optimizer,
                str(fixture),
                "--pass-pipeline=builtin.module(bind-source-iteration-domain)",
                "-o",
                str(canonical),
            ],
            expect_success=True,
        )

        _, noop_output = replay(
            optimizer, canonical, canonical, [identity()], directory, "noop"
        )
        assert_complete(noop_output, 1)

        # Apply a shape choice and two real C++ tiling actions to the same
        # source task, splitting the selected original domain by four.
        first_path = [
            shape("Task", 1, 2),
            tile("Task"),
            tile("Task.tile.0.0"),
        ]
        _, first_output = replay(
            optimizer,
            canonical,
            canonical,
            first_path,
            directory,
            "shape-then-two-tilings",
            cap=4,
        )
        first_facts = assert_complete(first_output, 3)

        # A second legal ordering makes a graph action before selecting the
        # child shape, then proves the next graph action from that predecessor.
        alternate_path = [
            tile("Task"),
            shape("Task.tile.0.0", 1, 2),
            tile("Task.tile.0.0"),
        ]
        _, alternate_output = replay(
            optimizer,
            canonical,
            canonical,
            alternate_path,
            directory,
            "tiling-then-child-shape",
            cap=4,
        )
        assert_complete(alternate_output, 3)

        # The prior candidate is an exact new input, checked again against the
        # immutable canonical source before another graph action is applied.
        _, chained_output = replay(
            optimizer,
            first_output / "candidate.mlir",
            canonical,
            [tile("Task.tile.0.1")],
            directory,
            "legal-previous-module",
            cap=4,
            initial_shapes=first_facts["selected_shapes"],
        )
        assert_complete(chained_output, 1)

        unknown_action = typed_action("unknown-family", "unknown-family:test")
        unknown_result, _ = replay(
            optimizer,
            canonical,
            canonical,
            [unknown_action],
            directory,
            "unknown-action",
            success=False,
        )
        if "unknown_neighborhood_action_family:unknown-family" not in unknown_result.stderr:
            fail("unknown action rejection omitted its concrete reason")

        cap_result, _ = replay(
            optimizer,
            canonical,
            canonical,
            [tile("Task")],
            directory,
            "factor-cap",
            cap=1,
            success=False,
        )
        if "action_factor_exceeds_cumulative_cap" not in cap_result.stderr:
            fail("factor cap rejection omitted its concrete reason")

        materializer_result, _ = replay(
            optimizer,
            canonical,
            canonical,
            [tile("Task", factor=8)],
            directory,
            "unsupported-source-factor",
            cap=8,
            success=False,
        )
        if "action_application_failed:tile:Task:axis=0:factor=8" not in materializer_result.stderr:
            fail("source materializer failure did not retain the rejected action")

    print("source-guarded typed neighborhood action replay checks passed")


if __name__ == "__main__":
    main()
