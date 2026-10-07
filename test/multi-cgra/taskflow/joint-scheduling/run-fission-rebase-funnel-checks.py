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
import shlex
from pathlib import Path
from typing import Any


FUNNEL_SCHEMA = "orbit-joint-neighborhood-family-funnel-v1"
WITNESS_SCHEMA = "orbit-joint-neighborhood-family-best-witness-v1"
TYPED_HISTORY_SCHEMA = "orbit-joint-neighborhood-typed-actions-v1"
REPLAY_FACTS_SCHEMA = "orbit-joint-neighborhood-action-replay-facts-v1"
ARCHIVE_HISTORY_RESTORE_REASON = (
    "checkpoint_archive_history_has_no_retained_candidate_module"
)
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


def is_typed_action(value: Any, fission: bool) -> bool:
    if not isinstance(value, dict):
        return False
    family = value.get("family")
    if not isinstance(family, str) or family not in {
        "identity", "canonical-reset", "lineage-replacement", "shape",
        "replica", "tiling", "k-tiling", "producer-consumer-co-tiling",
        "producer-consumer-co-k-tiling", "fusion", "fusion-plus-tiling",
        "tiling-plus-fusion", "fusion-plus-shape", "sibling-fusion", "fission",
    }:
        return False
    if not family or \
            not isinstance(value.get("label"), str) or \
            not isinstance(value.get("shapeTask"), str) or \
            type(value.get("shapeRows")) is not int or \
            type(value.get("shapeCols")) is not int or \
            not isinstance(value.get("canonicalReset"), bool):
        return False
    if (family == "fission") != fission:
        return False
    primitives = value.get("primitives")
    if not isinstance(primitives, list):
        return False
    for primitive in primitives:
        if not isinstance(primitive, dict) or \
                not isinstance(primitive.get("kind"), str) or \
                not isinstance(primitive.get("firstTask"), str) or \
                not isinstance(primitive.get("secondTask"), str) or \
                not isinstance(primitive.get("mode"), str) or \
                type(primitive.get("axis")) is not int or \
                type(primitive.get("factor")) is not int or \
                not isinstance(primitive.get("leftNodes"), list) or \
                any(type(node) is not int or node < 0
                    for node in primitive["leftNodes"]):
            return False
        if fission:
            if primitive.get("kind") != "fission" or \
                    not primitive.get("firstTask") or \
                    primitive.get("secondTask") or \
                    not primitive.get("leftNodes"):
                return False
        elif primitive.get("kind") == "fission" or primitive.get("leftNodes"):
            return False
    if fission and (value.get("canonicalReset") is True or
                    value.get("shapeTask") or len(primitives) != 1):
        return False
    return True


def is_typed_action_history(value: Any) -> bool:
    if not isinstance(value, dict) or \
            value.get("schema") != TYPED_HISTORY_SCHEMA or \
            value.get("known") is not True or \
            not isinstance(value.get("canonicalFactKey"), str) or \
            not value.get("canonicalFactKey"):
        return False
    initial_shapes = value.get("initialShapes")
    if not isinstance(initial_shapes, list) or not initial_shapes:
        return False
    seen_tasks: set[str] = set()
    for shape in initial_shapes:
        if not isinstance(shape, dict) or \
                not isinstance(shape.get("task"), str) or \
                not shape.get("task") or shape["task"] in seen_tasks or \
                type(shape.get("rows")) is not int or shape["rows"] <= 0 or \
                type(shape.get("cols")) is not int or shape["cols"] <= 0:
            return False
        seen_tasks.add(shape["task"])
    actions = value.get("actions")
    fission_actions = value.get("fissionActions")
    if not isinstance(actions, list) or not isinstance(fission_actions, list):
        return False
    return all(is_typed_action(action, False) for action in actions) and \
        all(is_typed_action(action, True) for action in fission_actions)


def same_typed_history(lhs: Any, rhs: Any, *, compare_fact_key: bool = True) -> bool:
    if not isinstance(lhs, dict) or not isinstance(rhs, dict):
        return False
    fields = ["schema", "initialShapes", "fissionActions", "actions"]
    if compare_fact_key:
        fields.append("canonicalFactKey")
    return lhs.get("known") is True and rhs.get("known") is True and \
        all(lhs.get(field) == rhs.get(field) for field in fields)


def is_supported_history_downgrade(record: dict[str, Any]) -> bool:
    history = record.get("action_history")
    return (
        record.get("valid") is True
        and record.get("candidate_path") == ""
        and isinstance(record.get("candidate_key"), str)
        and bool(record.get("candidate_key"))
        and isinstance(record.get("graph_facts_key"), str)
        and bool(record.get("graph_facts_key"))
        and isinstance(history, dict)
        and history.get("schema") == TYPED_HISTORY_SCHEMA
        and history.get("known") is False
        and history.get("unknownReason") == ARCHIVE_HISTORY_RESTORE_REASON
        and history.get("canonicalFactKey") == ""
        and history.get("initialShapes") == []
        and history.get("fissionActions") == []
        and history.get("actions") == []
    )


def validate_exact_replay_record(
    record_path: Path,
    candidate_id: str,
    event_history: dict[str, Any],
    archive_record: dict[str, Any],
    journal_path: Path,
    summary: dict[str, Any],
) -> None:
    record = json.loads(record_path.read_text())
    if record.get("status") != "pass" or record.get("exit_code") != 0 or \
            record.get("selected_candidate_id") != candidate_id or \
            record.get("source_fission_history_retained") is not True or \
            record.get("disjoint_ordinary_history_rebased") is not True:
        fail("exact replay receipt does not identify a successful disjoint fission candidate")
    origin = record.get("expected_candidate_origin")
    if not isinstance(origin, dict) or \
            origin.get("archive_journal") != str(journal_path.resolve()) or \
            origin.get("candidate_id") != candidate_id or \
            origin.get("cost_catalogue") != archive_record.get("cost_catalogue_path") or \
            origin.get("field") != "predictor_metadata.canonical_module_witness" or \
            origin.get("source_commit") != summary.get("source_commit"):
        fail("exact replay receipt is not bound to the archive journal, cost witness, and source commit")

    expected_witness = record.get("expected_candidate_witness")
    if not isinstance(expected_witness, str) or not expected_witness:
        fail("exact replay receipt omits its expected candidate witness")
    expected_witness_path = Path(expected_witness).resolve()
    if not expected_witness_path.is_file():
        fail(f"exact replay expected candidate witness is missing: {expected_witness_path}")
    cost_catalogue_path = Path(origin["cost_catalogue"]).resolve()
    if not cost_catalogue_path.is_file():
        fail(f"exact replay cost catalogue is missing: {cost_catalogue_path}")
    cost_catalogue = json.loads(cost_catalogue_path.read_text())
    metadata = cost_catalogue.get("predictor_metadata")
    archive_shapes = archive_record.get("shapes")
    if not isinstance(archive_shapes, list) or not archive_shapes or \
            any(not isinstance(shape, dict) or
                not isinstance(shape.get("task"), str) or not shape["task"]
                for shape in archive_shapes):
        fail("authenticated archive candidate has malformed task shapes")
    expected_source_tasks = sorted(
        shape["task"] for shape in archive_shapes
    )
    source_task_ids = metadata.get("source_task_ids") if isinstance(metadata, dict) else None
    if not isinstance(metadata, dict) or \
            cost_catalogue.get("function") != summary.get("function") or \
            metadata.get("source_commit") != summary.get("source_commit") or \
            metadata.get("source_repository") != summary.get("source_repository") or \
            not isinstance(source_task_ids, list) or \
            any(not isinstance(task, str) for task in source_task_ids) or \
            sorted(source_task_ids) != expected_source_tasks or \
            not isinstance(metadata.get("canonical_module_witness"), str) or \
            metadata["canonical_module_witness"] != expected_witness_path.read_text():
        fail("exact replay witness differs from the cited fresh cost catalogue body")

    argv = record.get("argv")
    if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
        fail("exact replay receipt has no command argument vector")
    replay_arguments = [
        value.split("=", 1)[1]
        for value in argv
        if value.startswith("--replay-joint-neighborhood-actions=")
    ]
    if len(replay_arguments) != 1:
        fail("exact replay receipt does not identify exactly one C++ replay invocation")
    try:
        replay_options = dict(
            token.split("=", 1)
            for token in shlex.split(replay_arguments[0])
            if "=" in token
        )
    except ValueError as error:
        fail(f"exact replay command options are malformed: {error}")
    required_options = (
        "action-file", "canonical-input", "candidate-input", "function",
        "stage", "max-partition-factor", "prepared-source-file",
        "max-fission-actions-per-task", "expected-candidate-file", "output-dir",
    )
    if any(not replay_options.get(option) for option in required_options) or \
            replay_options["function"] != summary.get("function") or \
            replay_options["stage"] != "full-joint-fission" or \
            replay_options["max-partition-factor"] != "8" or \
            replay_options["max-fission-actions-per-task"] != "64" or \
            Path(replay_options["expected-candidate-file"]).resolve() != expected_witness_path:
        fail("exact replay command is missing the fission, partition, or witness bindings")

    replay_dir = Path(replay_options["output-dir"]).resolve()
    facts_path = replay_dir / "source-facts.json"
    if not facts_path.is_file():
        fail(f"exact replay C++ source facts are missing: {facts_path}")
    facts = json.loads(facts_path.read_text())
    expected_path = Path(replay_options["expected-candidate-file"]).resolve()
    if facts.get("schema") != REPLAY_FACTS_SCHEMA or \
            facts.get("status") != "complete" or \
            facts.get("record_type") != "selection" or \
            facts.get("function") != summary.get("function") or \
            facts.get("stage") != "full-joint-fission" or \
            facts.get("source_iteration_domain_verified") is not True or \
            facts.get("fission_source_replay_verified") is not True or \
            facts.get("expected_candidate_exact_replay_match") is not True or \
            facts.get("expected_candidate_comparison") != \
            "exact-generic-module-after-removing-only-function-graph-variant-id" or \
            Path(facts.get("expected_candidate_path", "")).resolve() != expected_path or \
            facts.get("cost_catalog_read") is not False or \
            facts.get("prediction_run") is not False or \
            facts.get("native_mapping_run") is not False or \
            facts.get("ranking_performed") is not False:
        fail("C++ exact replay facts do not prove the expected source-safe replay")

    action_file_path = Path(replay_options["action-file"]).resolve()
    prepared_source_path = Path(replay_options["prepared-source-file"]).resolve()
    if not action_file_path.is_file() or not prepared_source_path.is_file() or \
            facts.get("action_file_path") != str(action_file_path) or \
            facts.get("prepared_source_input_path") != str(prepared_source_path):
        fail("exact replay action or prepared-source paths differ from the C++ receipt")
    action_document = json.loads(action_file_path.read_text())
    expected_actions = event_history["fissionActions"] + event_history["actions"]
    if action_document.get("schema") != TYPED_HISTORY_SCHEMA or \
            action_document.get("function") != summary.get("function") or \
            action_document.get("stage") != "full-joint-fission" or \
            action_document.get("maxPartitionFactor") != 8 or \
            action_document.get("maxFissionActionsPerTask") != 64 or \
            action_document.get("actions") != expected_actions or \
            action_document.get("initialShapes") != event_history.get("initialShapes") or \
            action_document.get("preparedSourceInput") != str(prepared_source_path) or \
            action_document.get("preparedSourceExactBytes") != prepared_source_path.read_text():
        fail("exact replay action file does not reproduce the authenticated journal history")
    replay_history = facts.get("action_history")
    if not is_typed_action_history(replay_history) or \
            not same_typed_history(replay_history, event_history, compare_fact_key=False):
        fail("C++ exact replay typed history differs from the positive search event")
    if facts.get("action_path") != [action["label"] for action in expected_actions] or \
            facts.get("actions_applied") != len(expected_actions) or \
            facts.get("shapes") != archive_record.get("shapes") or \
            facts.get("selected_shapes") != archive_record.get("shapes"):
        fail("C++ exact replay action count or selected task shapes differ from the archive")
    if facts.get("initial_shapes") != event_history.get("initialShapes") or \
            not isinstance(facts.get("tasks"), list) or \
            not facts["tasks"] or \
            any(not isinstance(task, dict) or task.get("complete") is not True
                for task in facts["tasks"]):
        fail("C++ exact replay omitted the complete source-domain task proof")
    candidate_path = Path(facts.get("candidate_path", "")).resolve()
    if candidate_path != replay_dir / "candidate.mlir" or not candidate_path.is_file():
        fail("C++ exact replay candidate module witness is missing")
    for witness_name, input_path in (
        ("actions.json", action_file_path),
        ("prepared-source.mlir", prepared_source_path),
        ("expected-candidate.mlir", expected_path),
        ("canonical-input.mlir", Path(replay_options["canonical-input"]).resolve()),
        ("candidate-input.mlir", Path(replay_options["candidate-input"]).resolve()),
    ):
        witness_path = replay_dir / witness_name
        if not witness_path.is_file() or witness_path.read_bytes() != input_path.read_bytes():
            fail(f"C++ exact replay retained witness differs: {witness_name}")


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
        "--exact-replay-record",
        type=Path,
        help="C++ exact-replay receipt for a positive candidate whose archive history was fail-closed on resume",
    )
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
    journal_path = output / "archive.journal.jsonl"
    for path in (summary_path, funnel_path, archive_path, journal_path):
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
    archive_rows = read_jsonl(archive_path)
    archive_by_all_id = {
        row.get("candidate_id"): row
        for row in archive_rows
        if row.get("record_type") == "candidate"
    }
    archived_by_id = {
        row.get("candidate_id"): row
        for row in archive_rows
        if row.get("record_type") == "candidate" and row.get("valid") is True
    }
    if len(archive_by_all_id) != len(archive_rows):
        fail("final archive contains duplicate IDs or non-candidate records")
    journal_versions: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(read_jsonl(journal_path)):
        candidate = row.get("candidate")
        if not isinstance(candidate, dict) or \
                not isinstance(candidate.get("candidate_id"), str) or \
                type(row.get("index")) is not int or row["index"] < 0:
            fail(f"archive journal row {index} has an invalid candidate or checkpoint-local index")
        journal_versions.setdefault(candidate["candidate_id"], []).append(candidate)
    if set(journal_versions) != set(archive_by_all_id):
        fail("archive journal and final archive have different candidate ID sets")
    for candidate_id, archived in archive_by_all_id.items():
        if journal_versions[candidate_id][-1] != archived:
            fail(f"final archive candidate {candidate_id} differs from its latest journal snapshot")

    downgraded_scored_ids: set[str] = set()
    downgraded_scored_event_count = 0
    for event in events:
        result = event.get("result")
        if not isinstance(result, dict) or result.get("scheduler_pass") is not True:
            continue
        candidate_id = event.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            fail("successful candidate attempt has no candidate ID")
        if result.get("materialized") is not True or \
                result.get("scheduler_calls", 0) <= 0:
            fail(
                f"successful candidate attempt {candidate_id} lacks "
                "materialization or scoring evidence"
            )
        archived = archived_by_id.get(candidate_id)
        if archived is None:
            fail(
                f"successful candidate attempt {candidate_id} is absent from "
                "the authenticated valid archive"
            )
        event_history = event.get("typed_action_history")
        archive_history = archived.get("action_history")
        if not is_typed_action_history(event_history):
            fail(f"successful candidate attempt {candidate_id} has malformed or unknown typed history")
        if is_typed_action_history(archive_history):
            if not same_typed_history(event_history, archive_history):
                fail(f"successful candidate attempt {candidate_id} disagrees with final archive typed history")
        elif is_supported_history_downgrade(archived):
            matching_prior_snapshot = any(
                prior.get("valid") is True
                and prior.get("candidate_key") == archived.get("candidate_key")
                and prior.get("graph_facts_key") == archived.get("graph_facts_key")
                and prior.get("cost_catalogue_path") == archived.get("cost_catalogue_path")
                and prior.get("shapes") == archived.get("shapes")
                and is_typed_action_history(prior.get("action_history"))
                and same_typed_history(event_history, prior.get("action_history"))
                for prior in journal_versions[candidate_id][:-1]
            )
            if not matching_prior_snapshot:
                fail(
                    f"successful candidate attempt {candidate_id} has no matching "
                    "authenticated pre-downgrade journal history"
                )
            downgraded_scored_ids.add(candidate_id)
            downgraded_scored_event_count += 1
        else:
            fail(f"successful candidate attempt {candidate_id} has unsupported final archive history")
    if not positive:
        fail("no disjoint ordinary-history -> fission edge completed production scoring and scheduler pass")
    if args.exact_replay_record is not None:
        try:
            exact_replay_record = json.loads(args.exact_replay_record.read_text())
        except (OSError, json.JSONDecodeError) as error:
            fail(f"cannot read exact replay receipt: {error}")
        exact_candidate_id = exact_replay_record.get("selected_candidate_id")
        matched = next(
            (event for event in positive
             if event.get("candidate_id") == exact_candidate_id),
            None,
        )
        if matched is None or exact_candidate_id not in downgraded_scored_ids:
            fail("exact replay receipt does not identify a scored disjoint fission candidate with an authenticated archive downgrade")
    else:
        matched = next(
            (
                event
                for event in positive
                if event.get("candidate_id") in archived_by_id
                and is_typed_action_history(
                    archived_by_id[event["candidate_id"]].get("action_history")
                )
                and archived_by_id[event["candidate_id"]]
                .get("action_history", {})
                .get("actions")
            ),
            None,
        )
    if matched is None:
        if args.exact_replay_record is None and any(
            event.get("candidate_id") in downgraded_scored_ids
            for event in positive
        ):
            fail("all disjoint fission positives have a fail-closed downgraded archive history; provide --exact-replay-record for an authenticated C++ exact replay")
        fail("disjoint fission candidate is absent from the authenticated valid archive")

    archive_record = archived_by_id[matched["candidate_id"]]
    history = matched["typed_action_history"]
    final_history = archive_record.get("action_history")
    if is_typed_action_history(final_history) and \
            not same_typed_history(history, final_history):
        fail("selected fission candidate does not match its known final archive history")
    if not is_typed_action_history(final_history):
        if args.exact_replay_record is None or \
                not is_supported_history_downgrade(archive_record):
            fail("selected positive has only a downgraded archive history; a valid exact replay receipt is required")
        validate_exact_replay_record(
            args.exact_replay_record,
            matched["candidate_id"],
            history,
            archive_record,
            journal_path,
            summary,
        )
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
        "positive_archive_history_source": (
            "final_archive" if is_typed_action_history(final_history)
            else "authenticated_pre_downgrade_journal_plus_exact_replay"
        ),
        "archive_history_downgrade_scored_candidate_count": len(downgraded_scored_ids),
        "archive_history_downgrade_scored_event_count": downgraded_scored_event_count,
        "archive_history_downgrade_reason": ARCHIVE_HISTORY_RESTORE_REASON,
        "exact_replay_receipt": (
            str(args.exact_replay_record.resolve())
            if args.exact_replay_record is not None else None
        ),
        "fission_family_best": best,
        "fission_current_edge": edge,
        "required_menu_rejects": args.require_menu_reject,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
