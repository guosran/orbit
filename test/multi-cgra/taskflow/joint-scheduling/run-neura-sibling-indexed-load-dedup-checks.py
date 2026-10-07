#!/usr/bin/env python3
"""Check exact folded-root/index/predicate identity for Neura indexed reads."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile


def fail(message: str) -> None:
    raise SystemExit(message)


def run(optimizer: str, source: pathlib.Path, output: pathlib.Path) -> str:
    command = [
        optimizer,
        str(source),
        "--verify-each",
        "--materialize-neura-joint-rewrite=fusion-mode=sibling first-task-name=Task_A second-task-name=Task_B",
        "-o",
        str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        fail(
            f"sibling fusion failed: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return output.read_text()


def check(text: str, loads: int, label: str) -> None:
    fused = re.search(
        r"taskflow\.task @Task_A\.fuse\.Task_B(?P<body>.*?)(?:\n\s*func\.return|\n\s*return|\Z)",
        text,
        re.S,
    )
    if not fused:
        fail(f"{label}: expected fused sibling task")
    body = fused.group("body")
    metric = re.search(
        r"amoeba\.neura\.fusion\.eliminated_loads\s*=\s*(\d+)", body
    )
    if not metric or int(metric.group(1)) != 2 - loads:
        fail(f"{label}: eliminated_loads does not match body\n{body}")
    if body.count("neura.load_indexed") != loads:
        fail(f"{label}: expected {loads} indexed loads\n{body}")
    if body.count("neura.store_indexed") != 2:
        fail(f"{label}: both distinct output stores must remain\n{body}")
    stores = re.search(
        r"amoeba\.neura\.fusion\.eliminated_stores\s*=\s*(\d+)", body
    )
    if not stores or int(stores.group(1)) != 0:
        fail(f"{label}: distinct output stores must not be claimed eliminated")


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: run-neura-sibling-indexed-load-dedup-checks.py FIXTURE OPTIMIZER")
    fixture = pathlib.Path(sys.argv[1]).resolve()
    optimizer = sys.argv[2]
    original = fixture.read_text()
    task_b = original.index("taskflow.task @Task_B")
    before, second = original[:task_b], original[task_b:]
    old_load_indices = "%predicated_row, %column_index : !neura.data<index, i1>, !neura.data<index, i1>"
    if second.count(old_load_indices) != 1:
        fail("expected one indexed load with shared counters and validity")
    different_index = second.replace(
        old_load_indices,
        "%predicated_row, %row_index : !neura.data<index, i1>, !neura.data<index, i1>",
        1,
    )
    old_validity = '%load_validity = "neura.constant"() <{value = 1 : i1}>'
    if second.count(old_validity) != 1:
        fail("expected Task_B to have one explicit load-validity constant")
    different_validity = second.replace(
        old_validity,
        '%load_validity = "neura.constant"() <{value = 0 : i1}>',
        1,
    )
    old_read = "will_reads(%input : memref<4x4xi32>)"
    old_original = "original_read_memrefs(%input : memref<4x4xi32>)"
    if second.count(old_read) != 1 or second.count(old_original) != 1:
        fail("could not isolate Task_B's declared input root")
    different_root = second.replace(old_read, "will_reads(%input_alt : memref<4x4xi32>)", 1)
    different_root = different_root.replace(
        old_original,
        "original_read_memrefs(%input_alt : memref<4x4xi32>)",
        1,
    )
    effect_marker = "            neura.store_indexed %value to [%row_index, %column_index"
    if second.count(effect_marker) != 1:
        fail("could not locate Task_B's output store for effectful control")
    effectful = second.replace(
        effect_marker,
        '            "neura.memset"(%input1, %value, %column_index) '
        '<{is_volatile = false}> : '
        '(!neura.data<memref<4x4xi32>, i1>, !neura.data<i32, i1>, '
        '!neura.data<index, i1>) -> ()\n'
        + effect_marker,
        1,
    )

    with tempfile.TemporaryDirectory(prefix="neura-sibling-indexed-dedup-") as temp:
        directory = pathlib.Path(temp)
        positive_path = directory / "positive.mlir"
        positive = run(optimizer, fixture, positive_path)
        check(positive, loads=1, label="same folded root, indices, predicate")

        mutants = [
            (different_index, "different index"),
            (different_validity, "different grant_predicate validity"),
            (different_root, "different folded memory root"),
            (effectful, "unclassified effectful operation"),
        ]
        for index, (body, label) in enumerate(mutants):
            mutant = directory / f"negative-{index}.mlir"
            mutant.write_text(before + body)
            negative_path = directory / f"negative-{index}-output.mlir"
            negative = run(optimizer, mutant, negative_path)
            check(negative, loads=2, label=label)


if __name__ == "__main__":
    main()
