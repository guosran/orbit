#!/usr/bin/env python3
"""Exercise source-partition witness binding and semantic shell rejection."""

from __future__ import annotations

import pathlib
import os
import re
import subprocess
import sys
import tempfile


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


def quoted_attr(text: str, name: str) -> str:
    pattern = re.compile(rf"({re.escape(name)}\s*=\s*)(\"(?:\\.|[^\"\\])*\")")
    match = pattern.search(text)
    if not match:
        fail(f"missing string attribute {name}")
    return match.group(2)


def replace_quoted_attr(text: str, name: str, quoted_value: str) -> str:
    pattern = re.compile(rf"({re.escape(name)}\s*=\s*)(\"(?:\\.|[^\"\\])*\")")
    updated, count = pattern.subn(lambda match: match.group(1) + quoted_value, text, count=1)
    if count != 1:
        fail(f"could not replace exactly one string attribute {name}")
    return updated


def add_candidate_attrs(source: str, semantic_change: bool = False) -> str:
    marker = "{amoeba.source_iteration_capture_pending,"
    attrs = "{amoeba.source_iteration_capture_pending,\n         "
    attrs += (
        'amoeba.neighborhood.partition_lineage.v1 = '
        '[{original = array<i64: 0, 4, 1>, root = "Task", steps = []}],\n         '
        'amoeba.source_iteration_partition_verified = '
        '"amoeba-source-iteration-domain-v1\\ncomplete=1\\n'
        'represented_multiplicity=4\\ninternal_multiplicity=1\\n'
        'source_multiplicity=4\\naxis_count=1\\n'
        'axis=T,0,0,4,1,4,0,-1,0,0\\nreason_bytes=0:\\n",\n         '
    )
    if semantic_change:
        attrs += "runtime_managable = false,\n         "
    updated = source.replace(marker, attrs.rstrip(), 1)
    if updated == source:
        fail("fixture could not add candidate attributes")
    return updated


def bind(optimizer: str, source: str, directory: pathlib.Path, label: str) -> pathlib.Path:
    input_path = directory / f"{label}.mlir"
    output_path = directory / f"{label}.bound.mlir"
    input_path.write_text(source)
    run(
        [
            optimizer,
            str(input_path),
            "--pass-pipeline=builtin.module(bind-source-iteration-domain)",
            "-o",
            str(output_path),
        ],
        expect_success=True,
    )
    return output_path


def prepare_imported_candidate(
    candidate: pathlib.Path, source_origin: str
) -> str:
    text = candidate.read_text()
    text = replace_quoted_attr(
        text, "amoeba.source_iteration_source_control_binding", source_origin
    )
    return text


def verify(
    optimizer: str,
    parent: pathlib.Path,
    candidate_text: str,
    directory: pathlib.Path,
    label: str,
    expect_success: bool,
) -> subprocess.CompletedProcess[str]:
    candidate = directory / f"{label}.imported.mlir"
    candidate.write_text(candidate_text)
    return run(
        [
            optimizer,
            str(candidate),
            "--verify-source-iteration-domain-partitions="
            f"parent-module={parent} function=source_partition_lineage",
            "-o",
            "/dev/null",
        ],
        expect_success=expect_success,
    )


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: run-source-iteration-domain-partition-lineage-checks.py FIXTURE OPTIMIZER")
    fixture = pathlib.Path(sys.argv[1])
    optimizer = sys.argv[2]
    expect_pre_fix = os.environ.get("ORBIT_SOURCE_LINEAGE_EXPECT_PRE_FIX") == "1"
    template = fixture.read_text()
    with tempfile.TemporaryDirectory(prefix="source-domain-lineage-") as temp:
        directory = pathlib.Path(temp)
        parent = bind(optimizer, template, directory, "parent")
        parent_text = parent.read_text()
        canonical_origin = quoted_attr(
            parent_text, "amoeba.source_iteration_source_control_binding"
        )

        candidate_source = add_candidate_attrs(template)
        candidate_bound = bind(optimizer, candidate_source, directory, "candidate")
        positive = prepare_imported_candidate(candidate_bound, canonical_origin)
        positive_result = verify(
            optimizer,
            parent,
            positive,
            directory,
            "positive",
            expect_success=not expect_pre_fix,
        )
        if expect_pre_fix and "task shell inputs, outputs, or semantic attributes changed" not in positive_result.stderr:
            fail("pre-fix positive case failed for a reason other than the lineage shell mismatch")
        if "amoeba.neighborhood.partition_lineage.v1" not in positive:
            fail("read-only verification removed the partition ledger")

        semantic_source = add_candidate_attrs(template, semantic_change=True)
        semantic_bound = bind(optimizer, semantic_source, directory, "semantic")
        semantic = prepare_imported_candidate(semantic_bound, canonical_origin)
        semantic_result = verify(
            optimizer,
            parent,
            semantic,
            directory,
            "semantic-negative",
            expect_success=False,
        )
        if "task shell inputs, outputs, or semantic attributes changed" not in semantic_result.stderr:
            fail("semantic shell mutation did not fail with the shell proof diagnostic")

        stale = positive.replace(
            "amoeba.neighborhood.partition_lineage.v1 = "
            "[{original = array<i64: 0, 4, 1>, root = \"Task\", steps = []}]",
            "amoeba.neighborhood.partition_lineage.v1 = "
            "[{original = array<i64: 1, 4, 1>, root = \"Task\", steps = []}]",
            1,
        )
        if stale == positive:
            fail("fixture could not alter the post-binding ledger")
        stale_result = verify(
            optimizer,
            parent,
            stale,
            directory,
            "stale-negative",
            expect_success=False,
        )
        expected_stale_diagnostic = (
            "task shell inputs, outputs, or semantic attributes changed"
            if expect_pre_fix
            else "stale or forged current body binding"
        )
        if expected_stale_diagnostic not in stale_result.stderr:
            fail(
                "post-binding ledger mutation did not fail at the expected proof stage:\n"
                + stale_result.stderr
            )

    label = "pre-fix mismatch reproduced; " if expect_pre_fix else ""
    print(label + "source-domain partition lineage positive/negative checks passed")


if __name__ == "__main__":
    main()
