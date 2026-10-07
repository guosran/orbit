#!/usr/bin/env python3
"""Validate a native search run for source-safe fission history rebasing.

The caller runs the bounded search with
``fission-rebase-disjoint-history.mlir`` as its source, then passes that
search directory here. This checker does not launch compilation, prediction,
or mapping itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FUNNEL_SCHEMA = "orbit-joint-neighborhood-family-funnel-v1"
WITNESS_SCHEMA = "orbit-joint-neighborhood-family-best-witness-v1"
FISSION_REPLAY_BINDING = (
    "orbit-taskflow-fission-source-replay-v2-ordinary-suffix-rebase"
)


def fail(message: str) -> None:
    raise SystemExit(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"{path}:{line_number}: malformed JSON: {error}")
        if not isinstance(value, dict):
            fail(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def fixture_source_text(text: str) -> str:
    """Return MLIR source with line comments removed for structural guards."""
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def has_authentic_affine_fixture_source(text: str) -> bool:
    source = fixture_source_text(text)
    return (
        "@fission_rebase_disjoint_history" in source
        and source.count("affine.for %i = 0 to 4") == 2
        and source.count("memref.alloc()") == 4
        and "taskflow.task @FissionTask" not in source
        and "amoeba.source_iteration_domain" not in source
    )


def action_footprint(action: dict[str, Any]) -> set[str] | None:
    family = action.get("family", "")
    if action.get("canonicalReset") is True or family in (
        "canonical-reset", "lineage-replacement", "fission"
    ):
        return None
    if family not in {
        "shape", "replica", "tiling", "k-tiling",
        "producer-consumer-co-tiling", "producer-consumer-co-k-tiling",
        "fusion", "fusion-plus-tiling", "tiling-plus-fusion",
        "fusion-plus-shape", "sibling-fusion",
    }:
        return None
    footprint: set[str] = set()
    shape_task = action.get("shapeTask", "")
    shape_family = family == "shape"
    fusion_shape_family = family == "fusion-plus-shape"
    if (shape_family or fusion_shape_family) != bool(shape_task):
        return None
    if shape_task:
        footprint.add(shape_task)
    primitives = action.get("primitives", [])
    if not isinstance(primitives, list) or (shape_family and primitives):
        return None

    def family_allows(kind: str) -> bool:
        if family == "replica":
            return kind == "replica"
        if family in {
            "tiling", "producer-consumer-co-tiling",
            "fusion-plus-tiling", "tiling-plus-fusion",
        }:
            return kind == "tile" or (
                family in {"fusion-plus-tiling", "tiling-plus-fusion"}
                and kind == "fusion"
            )
        if family in {
            "k-tiling", "producer-consumer-co-k-tiling",
        }:
            return kind == "k-tile"
        if family in {"fusion", "fusion-plus-shape"}:
            return kind == "fusion"
        if family == "sibling-fusion":
            return kind == "sibling-fusion"
        return False

    saw_tile = False
    saw_fusion = False
    for primitive in primitives:
        if not isinstance(primitive, dict):
            return None
        kind = primitive.get("kind", "")
        first = primitive.get("firstTask", "")
        second = primitive.get("secondTask", "")
        is_fusion = kind in {"fusion", "sibling-fusion"}
        if kind not in {"tile", "k-tile", "replica", "fusion", "sibling-fusion"}:
            return None
        if not family_allows(kind) or not first:
            return None
        if is_fusion and (not second or second == first):
            return None
        if not is_fusion and second:
            return None
        footprint.add(first)
        if second:
            footprint.add(second)
        saw_tile |= kind == "tile"
        saw_fusion |= is_fusion

    if family == "replica" and len(primitives) != 1:
        return None
    if family == "tiling" and (len(primitives) != 1 or not saw_tile):
        return None
    if family == "k-tiling" and len(primitives) != 1:
        return None
    if family in {
        "producer-consumer-co-tiling", "producer-consumer-co-k-tiling",
    } and len(primitives) != 2:
        return None
    if family in {"fusion", "sibling-fusion"} and len(primitives) != 1:
        return None
    if family == "fusion-plus-tiling" and (
        len(primitives) != 2 or not saw_tile or not saw_fusion
    ):
        return None
    if family == "tiling-plus-fusion" and (
        len(primitives) != 3 or not saw_tile or not saw_fusion
    ):
        return None
    if fusion_shape_family and (len(primitives) != 1 or not saw_fusion):
        return None
    return footprint or None


def has_disjoint_fission_rebase(event: dict[str, Any]) -> bool:
    action = event.get("action")
    history = event.get("typed_action_history")
    result = event.get("result")
    if not isinstance(action, dict) or not isinstance(history, dict) or not isinstance(result, dict):
        return False
    if event.get("action_family") != "fission" or history.get("known") is not True:
        return False
    prior_actions = history.get("actions")
    fission_actions = history.get("fissionActions")
    if not isinstance(prior_actions, list) or not prior_actions:
        return False
    if not isinstance(fission_actions, list) or not fission_actions:
        return False
    fission_primitives = action.get("primitives", [])
    if action.get("canonicalReset") is True or action.get("shapeTask") or \
            not isinstance(fission_primitives, list) or len(fission_primitives) != 1:
        return False
    fission_primitive = fission_primitives[0]
    if not isinstance(fission_primitive, dict) or \
            fission_primitive.get("kind") != "fission" or \
            not fission_primitive.get("firstTask") or \
            fission_primitive.get("secondTask"):
        return False
    target = fission_primitive["firstTask"]
    if target != "Task_0":
        return False
    if not any(
        isinstance(fission, dict)
        and any(
            isinstance(primitive, dict)
            and primitive.get("kind") == "fission"
            and primitive.get("firstTask") == target
            for primitive in fission.get("primitives", [])
        )
        for fission in fission_actions
    ):
        return False
    if any(
        not isinstance(prior, dict)
        or (footprint := action_footprint(prior)) is None
        or target in footprint
        for prior in prior_actions
    ):
        return False
    if not any(
        isinstance(prior, dict)
        and prior.get("family") == "shape"
        and prior.get("shapeTask") == "Task_1"
        and isinstance(prior.get("shapeRows"), int)
        and prior["shapeRows"] > 0
        and isinstance(prior.get("shapeCols"), int)
        and prior["shapeCols"] > 0
        for prior in prior_actions
    ):
        return False
    return (
        result.get("materialized") is True
        and result.get("scheduler_calls", 0) > 0
        and result.get("scheduler_pass") is True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("search_output", type=Path)
    parser.add_argument(
        "--require-menu-reject",
        action="append",
        default=[],
        choices=(
            "task_footprint_overlap",
            "shape_target_overlap",
            "history_unknown",
            "malformed_fission_target",
            "target_already_fissioned",
            "fission_in_ordinary_history",
            "unknown_footprint",
            "unknown_action_family",
            "unknown_primitive_kind",
            "unknown_primitive_family",
            "global_or_unknown_footprint",
            "canonical_reset",
            "lineage_replacement",
        ),
        help="also require this fail-closed fission-menu rejection reason",
    )
    args = parser.parse_args()

    if not has_authentic_affine_fixture_source(args.fixture.read_text()):
        fail("fixture must use two authentic affine source loops and four distinct allocations")

    output = args.search_output.resolve()
    summary_path = output / "diagnostics" / "family-funnel-summary.json"
    funnel_path = output / "diagnostics" / "family-funnel.jsonl"
    archive_path = output / "archive.jsonl"
    for path in (summary_path, funnel_path, archive_path):
        if not path.is_file():
            fail(f"search output is missing required diagnostic {path}")

    summary = json.loads(summary_path.read_text())
    if summary.get("schema") != FUNNEL_SCHEMA:
        fail("family funnel summary schema mismatch")
    if summary.get("fission_source_replay") != FISSION_REPLAY_BINDING:
        fail("search output is not bound to fission ordinary-suffix replay v2")
    edge = summary.get("current_edge_by_family", {}).get("fission")
    presence = summary.get("typed_path_presence_by_family", {}).get("fission")
    if not isinstance(edge, dict) or not isinstance(presence, dict):
        fail("summary has no exact fission current-edge and typed-path records")

    rows = read_jsonl(funnel_path)
    events = [row for row in rows if row.get("record_type") == "candidate_attempt"]
    positive = [event for event in events if has_disjoint_fission_rebase(event)]
    if not positive:
        fail("no disjoint ordinary-history -> fission edge completed production scoring and scheduler pass")
    archive_rows = read_jsonl(archive_path)
    archived_by_id = {
        row.get("candidate_id"): row
        for row in archive_rows
        if row.get("record_type") == "candidate" and row.get("valid") is True
    }
    matched = next(
        (
            event
            for event in positive
            if event.get("candidate_id") in archived_by_id
            and archived_by_id[event["candidate_id"]]
            .get("action_history", {})
            .get("known") is True
            and archived_by_id[event["candidate_id"]]
            .get("action_history", {})
            .get("actions")
        ),
        None,
    )
    if matched is None:
        fail("disjoint fission candidate is absent from the authenticated valid archive")

    history = archived_by_id[matched["candidate_id"]]["action_history"]
    if not history.get("canonicalFactKey") or not history.get("initialShapes"):
        fail("rebased archive history lacks canonical fact key or complete initial shapes")
    initial_shape_records = history["initialShapes"]
    if not isinstance(initial_shape_records, list):
        fail("rebased initial shapes are not an array")
    initial_shape_by_task = {
        shape.get("task"): shape
        for shape in initial_shape_records
        if isinstance(shape, dict) and isinstance(shape.get("task"), str)
    }
    if not all(
        name in initial_shape_by_task
        for name in ("Task_0.split.0", "Task_0.split.1", "Task_1")
    ):
        fail("rebased initial shapes do not preserve both fission children and Task_1")
    for child in ("Task_0.split.0", "Task_0.split.1"):
        child_shape = initial_shape_by_task[child]
        if child_shape.get("rows") != 1 or child_shape.get("cols") != 1:
            fail("rebased fission child did not retain its default 1x1 shape")
    event_history = matched["typed_action_history"]
    for field in ("canonicalFactKey", "initialShapes", "fissionActions", "actions"):
        if event_history.get(field) != history.get(field):
            fail(f"candidate event and valid archive disagree on typed history field {field}")
    if int(edge.get("materialized", 0)) <= 0 or \
            int(edge.get("scheduler_calls", 0)) <= 0 or \
            int(edge.get("scheduler_pass", 0)) <= 0:
        fail("fission edge aggregate does not show materialization and production scoring")
    if int(presence.get("scored", 0)) <= 0:
        fail("fission full-path presence has no scored candidates")

    best = presence.get("best")
    if not isinstance(best, dict) or best.get("candidate_id") is None:
        fail("fission path family has no ranked best candidate")
    if not isinstance(best.get("score"), int) or \
            not isinstance(best.get("global_rank"), int):
        fail("fission best family candidate lacks score or global rank")
    witness_path = best.get("family_witness_path")
    if not witness_path:
        fail("fission best candidate does not reference its retained family witness")
    witness_file = Path(witness_path)
    if not witness_file.is_file() and not witness_file.is_absolute():
        witness_file = output / witness_file
    if not witness_file.is_file():
        fail(f"fission family witness file is missing: {witness_file}")
    witness = json.loads(witness_file.read_text())
    if witness.get("schema") != WITNESS_SCHEMA or \
            witness.get("candidate_id") != best.get("candidate_id"):
        fail("fission family witness does not match the ranked best candidate")
    if "fission" not in witness.get("typed_path_families", []):
        fail("fission best witness is not attributed to the fission family")
    body = witness.get("candidate_module_ir")
    if not isinstance(body, str) or \
            len(body.encode("utf-8")) != witness.get("candidate_module_bytes"):
        fail("fission family witness omitted or changed the exact module body")
    witness_history = witness.get("action_history", {})
    witness_fission_targets = {
        primitive.get("firstTask")
        for fission in witness_history.get("fissionActions", [])
        if isinstance(fission, dict) and fission.get("family") == "fission"
        for primitive in fission.get("primitives", [])
        if isinstance(primitive, dict)
        and primitive.get("kind") == "fission"
        and isinstance(primitive.get("firstTask"), str)
        and primitive.get("firstTask")
    }
    if witness_history.get("known") is not True or \
            not witness_fission_targets or \
            any(f"{target}.split." not in body
                for target in witness_fission_targets):
        fail("fission best witness lacks its authenticated fission history or split body")
    if witness.get("cost_catalogue_snapshot_available"):
        cost_json = witness.get("cost_catalogue_exact_json")
        if not isinstance(cost_json, str):
            fail("fission family witness omits its exact cost catalogue snapshot")
        json.loads(cost_json)
    task_costs = witness.get("task_costs")
    if not isinstance(task_costs, list) or any(
        not all(key in cost for key in
                ("task", "predicted_ii", "trip_count", "predicted_duration"))
        for cost in task_costs
    ):
        fail("fission family witness lacks selected shape, II, trip, or duration evidence")

    reject_reasons = edge.get("reject_reasons", {})
    for reason in args.require_menu_reject:
        if int(reject_reasons.get(f"menu:{reason}", 0)) <= 0:
            fail(f"fission menu did not record the required fail-closed reason {reason}")

    print(json.dumps({
        "status": "passed",
        "positive_candidate_id": matched.get("candidate_id"),
        "fission_family_best": best,
        "fission_current_edge": edge,
        "required_menu_rejects": args.require_menu_reject,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
