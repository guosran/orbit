#!/usr/bin/env python3
"""Witness registered local choices in actual named-seed ORBIT evaluations.

Full local choice exposure does not imply a Cartesian-product/global search.
By default scanning stops only after every registered local choice has an
actual evaluation witness; the complete search trace remains sealed separately.
"""
import argparse
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def axis_values(implementation, runtime):
    variant = implementation.get("execution_variant", {})
    values = {"placement": implementation["device"],
              "dataflow": variant.get("dataflow"),
              "residence_policy": variant.get("residence_policy"),
              "replicas": variant.get("replicas"),
              "resource_shape": variant.get("resource_shape"),
              "implementation_parameters": variant.get("implementation_parameters"),
              "work_distribution": variant.get("work_distribution"),
              "transport_ownership": implementation.get("transport_ownership"),
              "visibility_policy": implementation.get("visibility_policy"),
              "runtime_kind": runtime.get("kind")}
    config = runtime.get("runtime_config", runtime)
    for field in ("backend", "input_policy", "output_policy"):
        values[field] = config.get(field)
    return {key: canonical(value) for key, value in values.items() if value is not None}


def audit(catalogue, stream, scan_all=False):
    groups = catalogue["choice_groups"]
    implementations = {row["id"]: row for row in catalogue["implementations"]}
    if len(implementations) != len(catalogue["implementations"]):
        raise ValueError("Duplicate implementation IDs")
    runtime = catalogue.get("runtime_registration", {}).get("implementations", {})
    dimensions = [group for group in groups if len(group["choices"]) > 1]
    if catalogue.get("execution_policy") == "source-dependency-ready":
        dimensions = dimensions + [{"id": "runtime.ready-priority", "choices": [
            {"id": policy, "implementations": []}
            for policy in catalogue["ready_priority_policies"]]}]
    if len({group["id"] for group in groups}) != len(groups):
        raise ValueError("Duplicate group IDs")
    all_groups = list(groups)
    if dimensions and dimensions[-1]["id"] == "runtime.ready-priority":
        all_groups.append(dimensions[-1])
    witnesses = {group["id"]: {choice["id"]: None for choice in group["choices"]}
                 for group in all_groups}
    priced = {key: set() for key in witnesses}
    for group in all_groups:
        if not group["choices"] or len(witnesses[group["id"]]) != len(group["choices"]):
            raise ValueError("Empty or duplicate registered choice IDs")
    outstanding = {d for d, group in enumerate(dimensions)}
    fixed = [group for group in groups if len(group["choices"]) == 1]
    scanned = eligible = 0
    stopped_full = False
    for line in stream:
        scanned += 1
        row = json.loads(line)
        if row.get("source_id") != catalogue["source_id"]:
            raise ValueError("Trace source differs from admitted catalogue")
        if row.get("priority_key_identity") != "named-ready-priority-seed":
            continue
        values = row.get("assignment")
        if (not isinstance(values, list) or len(values) != len(dimensions)
                or any(type(value) is not int or not 0 <= value < len(group["choices"])
                       for value, group in zip(values, dimensions))):
            raise ValueError("Actual assignment is outside registered dimensions")
        eligible += 1
        witness = {"evaluation": row["evaluation"], "trace_line": scanned,
                   "feasible": row.get("feasible"),
                   "cost_known": row.get("cost_ms") is not None}
        for group in fixed:
            choice = group["choices"][0]["id"]
            if witnesses[group["id"]][choice] is None:
                witnesses[group["id"]][choice] = dict(witness)
            if witness["feasible"] is True and witness["cost_known"]:
                priced[group["id"]].add(choice)
        for d, group in enumerate(dimensions):
            choice = group["choices"][values[d]]["id"]
            if witnesses[group["id"]][choice] is None:
                witnesses[group["id"]][choice] = dict(witness)
            if witness["feasible"] is True and witness["cost_known"]:
                priced[group["id"]].add(choice)
            if d in outstanding and all(value is not None for value in witnesses[group["id"]].values()):
                outstanding.remove(d)
        if not outstanding and not scan_all:
            stopped_full = True
            break
    if not eligible:
        raise ValueError("No actual named-seed evaluation rows found")
    axes = {}
    details = []
    for group in all_groups:
        witness_map = witnesses[group["id"]]
        for choice in group["choices"]:
            exposed = witness_map[choice["id"]] is not None
            feasible = choice["id"] in priced[group["id"]]
            values = {}
            covers = []
            for identifier in choice["implementations"]:
                implementation = implementations[identifier]
                covers.append(implementation["source_operators"])
                for axis, value in axis_values(implementation, runtime.get(identifier, {})).items():
                    values.setdefault(axis, set()).add(value)
            values["source_partition"] = {canonical(sorted(covers))}
            if group["id"] == "runtime.ready-priority":
                values["named_ready_priority"] = {canonical(choice["id"])}
            for axis, signatures in values.items():
                sets = axes.setdefault(axis, {"registered": set(), "evaluated": set(),
                                              "priced_feasible": set()})
                sets["registered"].update(signatures)
                if exposed:
                    sets["evaluated"].update(signatures)
                if feasible:
                    sets["priced_feasible"].update(signatures)
        details.append({"group_id": group["id"], "registered_choices": len(witness_map),
                        "evaluated_choices": sum(value is not None for value in witness_map.values()),
                        "priced_feasible_choices_in_scanned_rows": len(priced[group["id"]]),
                        "choice_witnesses": witness_map})
    return {"schema": "orbit.actual-local-choice-coverage.v1", "source_id": catalogue["source_id"],
            "execution_registration_identity": catalogue["execution_registration_identity"],
            "trace_rows_scanned": scanned, "named_seed_evaluations_scanned": eligible,
            "scan_stopped_after_full_local_choice_coverage": stopped_full,
            "all_registered_local_choices_evaluated": all(
                item["registered_choices"] == item["evaluated_choices"] for item in details),
            "global_cartesian_product_exhausted": False,
            "general_temporal_neighborhood_scope": "separate joint-action trace and final convergence report",
            "axis_counts": {axis: {key + "_distinct_values": len(values)
                                    for key, values in sets.items()} for axis, sets in axes.items()},
            "groups": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-all", action="store_true")
    args = parser.parse_args()
    catalogue_bytes = args.catalogue.read_bytes()
    catalogue = json.loads(catalogue_bytes)
    with args.checkpoint.open() as stream:
        result = audit(catalogue, stream, args.scan_all)
    result["catalogue_sha256"] = hashlib.sha256(catalogue_bytes).hexdigest()
    result["checkpoint_path"] = str(args.checkpoint.resolve(strict=True))
    result["checkpoint_hash_binding"] = "complete trace hash belongs to final search seal"
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "all_registered_local_choices_evaluated", "trace_rows_scanned", "axis_counts")}))


if __name__ == "__main__":
    main()
