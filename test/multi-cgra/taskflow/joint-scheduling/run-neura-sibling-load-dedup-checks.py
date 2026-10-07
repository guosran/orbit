#!/usr/bin/env python3
"""Check exact sibling-load deduplication and a different-address reject."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile


def fail(message: str) -> None:
    raise SystemExit(message)


def transform(optimizer: str, source: pathlib.Path, output: pathlib.Path) -> str:
    command = [
        optimizer,
        str(source),
        "--verify-each",
        "--classify-task-and-counter",
        "--convert-taskflow-to-neura",
        "--materialize-neura-joint-rewrite=fusion-mode=sibling first-task-name=sibling_a second-task-name=sibling_b",
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
        r"taskflow\.task @sibling_a\.fuse\.sibling_b(?P<body>.*?)(?:\n\s*taskflow\.task |\n\s*func\.return|\n\s*return|\Z)",
        text,
        re.S,
    )
    if not fused:
        fail(f"{label}: expected fused sibling task")
    body = fused.group("body")
    match = re.search(r"amoeba\.neura\.fusion\.eliminated_loads\s*=\s*(\d+)", body)
    if not match or int(match.group(1)) != 2 - loads:
        fail(f"{label}: eliminated_loads does not match read count\n{body}")
    actual_loads = body.count("memref.load")
    stores = body.count("memref.store")
    if actual_loads != loads or stores != 2:
        fail(
            f"{label}: expected {loads} memref.load and two retained stores, "
            f"got {actual_loads} loads and {stores} stores\n{body}"
        )
    store_count = re.search(
        r"amoeba\.neura\.fusion\.eliminated_stores\s*=\s*(\d+)", body
    )
    if not store_count or int(store_count.group(1)) != 0:
        fail(f"{label}: sibling fusion must not claim an eliminated output store")


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: run-neura-sibling-load-dedup-checks.py FIXTURE OPTIMIZER")
    fixture = pathlib.Path(sys.argv[1]).resolve()
    optimizer = sys.argv[2]
    source_text = fixture.read_text()
    old_load = "memref.load %read[%ii, %jj] : memref<4x5xi32>"
    if source_text.count(old_load) != 2:
        fail("fixture must contain the same indexed input read in both siblings")

    with tempfile.TemporaryDirectory(prefix="neura-sibling-read-dedup-") as temp:
        directory = pathlib.Path(temp)
        positive_output = directory / "positive.mlir"
        positive = transform(optimizer, fixture, positive_output)
        check(positive, loads=1, label="identical shared read")

        task_b = source_text.index("taskflow.task @sibling_b")
        before, second = source_text[:task_b], source_text[task_b:]
        if second.count(old_load) != 1:
            fail("could not isolate Task_B read for the negative case")
        different_address = directory / "different-address.mlir"
        different_address.write_text(
            before + second.replace(old_load, "memref.load %read[%ii, %c0] : memref<4x5xi32>", 1)
        )
        negative_output = directory / "different-address-output.mlir"
        negative = transform(optimizer, different_address, negative_output)
        check(negative, loads=2, label="different address")


if __name__ == "__main__":
    main()
