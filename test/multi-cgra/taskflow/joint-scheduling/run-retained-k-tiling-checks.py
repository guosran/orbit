#!/usr/bin/env python3
"""Check full-rank source-owned M/N tiling with an unchanged K reduction."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess


DEFAULT_FUNCTION = "_Z10llama_funciPA256_KiS1_S1_S1_S1_S1_S1_PA256_i"
EXPECTED_K = {"Task_0": 256, "Task_1": 256, "Task_3": 256,
              "Task_8": 256, "Task_6": 320}


def load_helpers():
    path = Path(__file__).with_name("run-nested-tiling-lineage-checks.py")
    spec = importlib.util.spec_from_file_location("nested_lineage_helpers", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load helper script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load_helpers()


def fail(message: str) -> None:
    raise SystemExit(f"retained-K tiling check: {message}")


def run(command: list[str], log: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    return result


def tile(task: str, axis: int, factor: int = 2) -> dict[str, object]:
    return {
        "family": "tiling",
        "label": f"tile:{task}:axis={axis}:factor={factor}",
        "primitives": [{"kind": "tile", "firstTask": task, "secondTask": "",
                        "axis": axis, "factor": factor, "mode": "none"}],
        "shapeTask": "", "shapeRows": 0, "shapeCols": 0,
        "canonicalReset": False,
    }


def identity() -> dict[str, object]:
    return {"family": "identity", "label": "identity", "primitives": [],
            "shapeTask": "", "shapeRows": 0, "shapeCols": 0,
            "canonicalReset": False}


def replay(optimizer: Path, canonical: Path, candidate: Path, function: str,
           actions: list[dict[str, object]], out: Path, label: str):
    action_file = out / f"{label}.actions.json"
    replay_dir = out / f"{label}.complete"
    action_file.write_text(json.dumps({
        "schema": "orbit-joint-neighborhood-typed-actions-v1",
        "canonicalInput": str(canonical.resolve()),
        "candidateInput": str(candidate.resolve()),
        "function": function, "stage": "full-joint",
        "maxPartitionFactor": 8, "actions": actions,
    }, indent=2) + "\n")

    def q(path: Path) -> str:
        return json.dumps(str(path.resolve()))

    options = " ".join((
        f"action-file={q(action_file)}", f"canonical-input={q(canonical)}",
        f"candidate-input={q(candidate)}", f"function={function}",
        "stage=full-joint", "max-partition-factor=8",
        f"output-dir={q(replay_dir)}"))
    result = run([str(optimizer), str(candidate), "--verify-each",
                  f"--replay-joint-neighborhood-actions={options}",
                  "-o", "/dev/null"], out / f"{label}.log")
    return result, replay_dir


def neura_bounds(task_text: str) -> dict[int, tuple[str, str, str]]:
    op = re.compile(
        r'(?m)^\s*%[^=\n]+\s*=\s*"neura\.counter"\([^\n)]*\)\s*'
        r'<\{([^}]*)}>\s*\{([^}]*)\}')
    found: dict[int, tuple[str, str, str]] = {}

    def value(name: str, attributes: str) -> str | None:
        match = re.search(
            rf"\b{name}\s*=\s*(\"(?:[^\"\\]|\\.)*\"|-?\d+)",
            attributes)
        return match.group(1).strip('"') if match else None

    for attrs, values in op.findall(task_text):
        ident = re.search(r"\bcounter_id\s*=\s*(\d+)\s*:\s*i32", attrs)
        if not ident:
            continue
        lower = value("lower_bound_value", values)
        upper = value("upper_bound_value", values)
        step = value("step_value", values)
        found[int(ident.group(1))] = (
            lower if lower is not None else "<missing>",
            upper if upper is not None else "<missing>",
            step if step is not None else "<missing>")
    return found


def source_k_extent(task_text: str) -> int:
    domain = re.search(
        r"amoeba\.source_iteration_domain\s*=\s*\{axes\s*=\s*\[(.*?)\]\s*,\s*canonical_witness",
        task_text, re.S)
    if not domain:
        fail("task has no structured source iteration domain")
    for axis in re.finditer(r"\{([^{}]*)\}", domain.group(1)):
        attrs = axis.group(1)
        ordinal = re.search(r"\bordinal\s*=\s*2\s*:\s*i64", attrs)
        extent = re.search(r"\bextent\s*=\s*(\d+)\s*:\s*i64", attrs)
        if ordinal and extent:
            return int(extent.group(1))
    fail("task has no authenticated source K extent")


def canonical_k(text: str, task: str) -> tuple[tuple[str, str, str], int]:
    _, _, body = H.task_span(text, task)
    bounds = neura_bounds(body)
    signature = bounds.get(2)
    extent = source_k_extent(body)
    if not signature or signature[0] != "0" or signature[2] != "1":
        fail(f"canonical {task} has invalid Neura K signature {signature!r}")
    if extent != EXPECTED_K[task]:
        fail(f"canonical {task} source K extent is {extent}, expected {EXPECTED_K[task]}")
    return signature, extent


def check_leaf_k(candidate: str, task: str,
                 expected_signature: tuple[str, str, str], extent: int,
                 families, label: str) -> list[str]:
    names = []
    for name, _, _, body in H.task_spans(candidate):
        if not name.startswith(task + "."):
            continue
        bounds = neura_bounds(body)
        if len(bounds) != 3 or bounds.get(2) != expected_signature:
            fail(f"{label}: {name} has changed Neura K/counter bounds {bounds!r}")
        key = H.find_unquoted(body, H.LEDGER)
        if key < 0:
            fail(f"{label}: {name} has no typed full-rank ledger")
        encoded = body[key:]
        for match in re.finditer(
                r"(?:before|after)\s*=\s*array<i64:\s*([^>]*)>", encoded):
            values = [int(value) for value in re.findall(r"-?\d+", match.group(1))]
            if len(values) != 9 or values[6:9] != [0, extent, 1]:
                fail(f"{label}: {name} ledger does not retain K=[0,{extent})")
        actual_families = H.lineage_families(body)
        expected_families = (families.get(name) if isinstance(families, dict)
                             else families)
        if expected_families is None or actual_families != expected_families:
            fail(f"{label}: {name} lineage {actual_families!r}, expected "
                 f"{expected_families!r}")
        names.append(name)
    if not names:
        fail(f"{label}: no derived tasks found under {task}")
    return sorted(names)


def positive(optimizer: Path, canonical: Path, function: str, out: Path,
             label: str, task: str, actions: list[dict[str, object]],
             expected_k: tuple[str, str, str], expected_extent: int,
             families):
    result, replay_dir = replay(optimizer, canonical, canonical, function,
                                actions, out, label)
    if result.returncode:
        fail(f"positive {label} failed; inspect {label}.log")
    facts_path = replay_dir / "source-facts.json"
    candidate_path = replay_dir / "candidate.mlir"
    if not facts_path.is_file() or not candidate_path.is_file():
        fail(f"positive {label} omitted candidate output or source facts")
    facts = json.loads(facts_path.read_text())
    if facts.get("status") != "complete" or \
            facts.get("source_iteration_domain_verified") is not True:
        fail(f"positive {label} lacks a complete source-domain proof")
    leaves = check_leaf_k(candidate_path.read_text(), task, expected_k,
                          expected_extent, families, label)
    return {"facts": str(facts_path), "candidate": str(candidate_path),
            "derived_tasks": leaves}


def mutate_k_ledger(ir: str, task: str) -> str:
    begin, end, body = H.task_span(ir, task)
    key = H.find_unquoted(body, H.LEDGER)
    if key < 0:
        fail(f"{task} has no partition ledger to mutate")
    match = re.search(
        r"(after\s*=\s*array<i64:\s*(?:-?\d+\s*,\s*){7})(-?\d+)",
        body[key:])
    if not match:
        fail(f"{task} has no full-rank after bounds in its ledger")
    old = int(match.group(2))
    if old <= 1:
        fail(f"{task} K upper bound cannot be safely decremented")
    pos = key + match.start(2)
    changed = body[:pos] + str(old - 1) + body[key + match.end(2):]
    return ir[:begin] + changed + ir[end:]


def mutate_source_root(ir: str, task: str) -> str:
    begin, end, body = H.task_span(ir, task)
    key = H.find_unquoted(body, H.LEDGER)
    if key < 0:
        fail(f"{task} has no partition ledger to mutate")
    match = re.search(r'(root\s*=\s*)"[^"\\]+"', body[key:])
    if not match:
        fail(f"{task} has no typed lineage root to mutate")
    start = key + match.start()
    value = key + match.end()
    changed = body[:start] + match.group(1) + '"ForgedSourceRoot"' + body[value:]
    return ir[:begin] + changed + ir[end:]


def mutate_m_ledger_volume(ir: str, task: str) -> str:
    begin, end, body = H.task_span(ir, task)
    key = H.find_unquoted(body, H.LEDGER)
    if key < 0:
        fail(f"{task} has no partition ledger to mutate")
    match = re.search(
        r'(after\s*=\s*array<i64:\s*-?\d+\s*,\s*)(-?\d+)', body[key:])
    if not match:
        fail(f"{task} has no rank-three M interval to mutate")
    upper = int(match.group(2))
    pos = key + match.start(2)
    body = body[:pos] + str(upper + 1) + body[key + match.end(2):]
    return ir[:begin] + body + ir[end:]


def mutate_output_region(ir: str, task: str) -> str:
    begin, end, body = H.task_span(ir, task)
    key = H.find_unquoted(body, "amoeba.tiling.output_region_uppers")
    if key < 0:
        fail(f"{task} has no output-region upper bound to mutate")
    match = re.search(
        r'(\[\s*array<i64:\s*)(-?\d+)(\s*,\s*)(-?\d+)', body[key:])
    if not match:
        fail(f"{task} has no two-dimensional output region to mutate")
    old = int(match.group(2))
    pos = key + match.start(2)
    changed = body[:pos] + str(old + 1) + body[key + match.end(2):]
    return ir[:begin] + changed + ir[end:]


def mutate_first_join_region(ir: str) -> str:
    match = re.search(
        r'(?m)^(\s*%[^\n]*=\s*"taskflow\.join"[^\n]*?'
        r'region_upper\s*=\s*array<i64:\s*)(-?\d+)(\s*,\s*)(-?\d+)', ir)
    if not match:
        fail("candidate has no typed completion-join region to mutate")
    old = int(match.group(2))
    return ir[:match.start(2)] + str(old + 1) + ir[match.end(2):]


def require_invalid_join_rejection(optimizer: Path, out: Path, label: str,
                                   mutated: str, expected: str) -> str:
    raw = out / f"{label}.mutated.mlir"
    raw.write_text(mutated)
    result = run([str(optimizer), str(raw), "--verify-each", "-o", "/dev/null"],
                 out / f"{label}.verify.log")
    diagnostics = [line.strip() for line in result.stdout.splitlines()
                   if "error:" in line]
    if result.returncode == 0:
        fail(f"invalid join mutant {label} passed MLIR verification")
    if not diagnostics or expected not in diagnostics[-1]:
        fail(f"invalid join mutant {label} failed unexpectedly: "
             f"{diagnostics[-1] if diagnostics else '<no diagnostic>'!r}")
    return diagnostics[-1]


def reject_mutant(optimizer: Path, canonical: Path, function: str,
                  out: Path, label: str, task: str, mutated: str,
                  original: str, expected: str, *, rebind: bool = True) -> str:
    raw = out / f"{label}.mutated.mlir"
    raw.write_text(mutated)
    H.verify_input(optimizer, raw, out, f"{label}-mutated")
    candidate = (H.refresh_mutated_current_binding(
        optimizer, original, mutated, task, out, label) if rebind else raw)
    result, replay_dir = replay(optimizer, canonical, candidate, function,
                                [identity()], out, label)
    diagnostics = [line.strip() for line in result.stdout.splitlines()
                   if "error:" in line]
    if result.returncode == 0:
        fail(f"negative {label} was accepted")
    if not diagnostics or expected not in diagnostics[-1]:
        fail(f"negative {label} failed at an unexpected guard: "
             f"{diagnostics[-1] if diagnostics else '<no diagnostic>'!r}")
    if replay_dir.exists():
        fail(f"negative {label} published partial replay output")
    return diagnostics[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--function", default=DEFAULT_FUNCTION)
    parser.add_argument("--tasks", default="Task_0,Task_1,Task_3")
    args = parser.parse_args()
    optimizer, canonical = args.optimizer.resolve(), args.canonical.resolve()
    if not optimizer.is_file() or not canonical.is_file():
        fail("optimizer or canonical input is missing")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    canonical_text = canonical.read_text()
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    if not tasks or any(task not in EXPECTED_K for task in tasks):
        fail(f"--tasks must be a nonempty subset of {sorted(EXPECTED_K)!r}")
    k_info = {task: canonical_k(canonical_text, task) for task in tasks}
    k_bounds = {task: info[0] for task, info in k_info.items()}
    k_extents = {task: info[1] for task, info in k_info.items()}

    positives = {}
    for task in tasks:
        for axis in (0, 1):
            for factor in (2, 4, 8):
                label = f"pure-{task}-axis{axis}-factor{factor}"
                positives[label] = positive(
                    optimizer, canonical, args.function, out, label, task,
                    [tile(task, axis, factor)], k_bounds[task],
                    k_extents[task], ["tiling"])

        label = f"repeated-{task}-axis0"
        positives[label] = positive(
            optimizer, canonical, args.function, out, label, task,
            [tile(task, 0), tile(f"{task}.tile.0.0", 0),
             tile(f"{task}.tile.0.1", 0)],
            k_bounds[task], k_extents[task], ["tiling", "tiling"])

        label = f"mixed-MN-{task}"
        positives[label] = positive(
            optimizer, canonical, args.function, out, label, task,
            [tile(task, 0), tile(f"{task}.tile.0.0", 1),
             tile(f"{task}.tile.0.1", 1)],
            k_bounds[task], k_extents[task], ["tiling", "tiling"])

    first_task = tasks[0]
    first_child = f"{first_task}.tile.0.0"
    candidate_text = Path(positives[
        f"pure-{first_task}-axis0-factor2"]["candidate"]).read_text()
    forged_k = reject_mutant(
        optimizer, canonical, args.function, out,
        f"forged-K-bound-{first_task}", first_child,
        mutate_k_ledger(candidate_text, first_child), candidate_text,
        "source-owned tiling partition lineage has a broken interval chain")
    missing_ledger = reject_mutant(
        optimizer, canonical, args.function, out,
        f"missing-rank3-ledger-{first_task}", first_child,
        H.mutate_ledger(candidate_text, first_child, "missing"),
        candidate_text,
        "source-owned rank-three partition requires an authenticated full-rank lineage ledger")
    forged_source = reject_mutant(
        optimizer, canonical, args.function, out,
        f"forged-source-root-{first_task}", first_child,
        mutate_source_root(candidate_text, first_child), candidate_text,
        "source-owned tiling partition lineage does not match its canonical root")
    forged_volume = reject_mutant(
        optimizer, canonical, args.function, out,
        f"forged-m-partition-volume-{first_task}", first_child,
        mutate_m_ledger_volume(candidate_text, first_child),
        candidate_text,
        "source-owned tiling partition lineage has a broken interval chain")

    mixed_label = f"mixed-MN-{first_task}"
    mixed_candidate = Path(positives[mixed_label]["candidate"]).read_text()
    mixed_leaves = [name for name in positives[mixed_label]["derived_tasks"]
                    if ".tile.1." in name]
    if not mixed_leaves:
        fail(f"{mixed_label}: no second-axis leaves for output-region mutant")
    forged_output_region = reject_mutant(
        optimizer, canonical, args.function, out,
        f"forged-output-region-{first_task}", mixed_leaves[0],
        mutate_output_region(mixed_candidate, mixed_leaves[0]),
        mixed_candidate,
        "output region disagrees with its actual indexed output coordinates",
        rebind=False)
    forged_join_region = require_invalid_join_rejection(
        optimizer, out, f"forged-join-region-{first_task}",
        mutate_first_join_region(mixed_candidate),
        "tile regions differ outside the selected axis")

    report = {
        "schema": "orbit-retained-k-tiling-checks-v1",
        "optimizer": str(optimizer), "canonical": str(canonical),
        "function": args.function, "canonical_k_bounds": k_bounds,
        "canonical_k_extents": k_extents,
        "positive_checks": positives,
        "negative_diagnostics": {
            "forged_k_bound": forged_k,
            "missing_rank3_ledger": missing_ledger,
            "forged_source_root": forged_source,
            "forged_current_volume": forged_volume,
            "forged_output_region": forged_output_region,
            "forged_join_region": forged_join_region,
        },
        "result": "pass",
    }
    result = out / "result.json"
    result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(result)


if __name__ == "__main__":
    main()
