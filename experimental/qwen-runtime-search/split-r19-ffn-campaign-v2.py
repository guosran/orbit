#!/usr/bin/env python3
"""Resume sealed evidence, splitting pending FFN cohorts into smaller complete-model runs.

Only the number of fused layers per request changes. All replacement cohorts
still execute the complete model and unchanged numeric admission command. The
union of their selected fused layers is exactly the failed parent's selection.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
resume = load_module("sealed_resume", HERE / "resume-r19-campaign.py")
node = load_module("node_contract", HERE.parents[1] / "scripts/qwen-node-runtime.py")


def argument(command, name):
    if command.count(name) != 1:
        raise ValueError("Expected one command argument " + name)
    return command[command.index(name) + 1]


def split_layers(plan, maximum):
    layers = plan["r18"]["fused_layers"]
    if (type(maximum) is not int or maximum < 1 or len(layers) < 2
            or len(set(layers)) != len(layers)
            or any(type(i) is not int or not 0 <= i < 36 for i in layers)):
        raise ValueError("Expected at least two distinct fused layers and a positive chunk size")
    if len(layers) <= maximum:
        raise ValueError("Requested chunk size would not split the pending cohort")
    pieces = [layers[index:index + maximum] for index in range(0, len(layers), maximum)]
    if sum(pieces, []) != layers or any(not piece or len(piece) > maximum for piece in pieces):
        raise ValueError("Chunking changed exact fused source coverage")
    return pieces



def derive_ready(source, plan, old_ready, catalogue):
    node.validate_plan(source, plan)
    if old_ready.get("operators") or old_ready.get("cpu_regions"):
        raise ValueError("FFN split target must not contain half/CPU-region overrides")
    registry = catalogue["runtime_registration"]["implementations"]
    ids = {}
    for name, device in plan["placements"].items():
        matches = [identifier for identifier, row in registry.items()
                   if row["kind"] == "prefill-original"
                   and row["operator"] == name and row["device"] == device]
        if len(matches) != 1:
            raise ValueError("Original registration does not resolve uniquely: " + name)
        ids[name] = matches[0]
    for name, selected in plan["r18"]["gemm_variants"].items():
        matches = [identifier for identifier, row in registry.items()
                   if row["kind"] == "prefill-gemm" and row["operator"] == name
                   and row["manifest_sha256"] == selected["manifest_sha256"]]
        if len(matches) != 1:
            raise ValueError("GEMM registration does not resolve uniquely")
        ids[name] = matches[0]
    fused_members, priorities = set(), {}
    ordinals = {op["id"]: op["source_ordinal"] for op in source["operators"]}
    for layer in plan["r18"]["fused_layers"]:
        route = plan.get("r18_layer_ffn", {}).get(str(layer), plan["r18"])
        matches = [identifier for identifier, row in registry.items()
                   if row["kind"] == "prefill-fused-ffn" and row["layer"] == layer
                   and all(row[key] == route[key]
                           for key in ("retention", "split_order", "ffn_artifact"))]
        if len(matches) != 1:
            raise ValueError("Fused FFN registration does not resolve uniquely")
        names = [f"prefill.L{layer}.{role}" for role in
                 ("gate_up", "gate_split", "swiglu", "down")]
        if any(name not in ordinals for name in names):
            raise ValueError("Fused FFN source cover is incomplete")
        ids[names[0]] = matches[0]
        priorities[matches[0]] = ordinals[names[0]]
        fused_members.update(names)
    for name, ordinal in ordinals.items():
        if name not in fused_members:
            priorities[ids[name]] = ordinal
    if len(priorities) != len(source["operators"]) - 3 * len(plan["r18"]["fused_layers"]):
        raise ValueError("Invocation priorities are not an exact source cover")
    ready = copy.deepcopy(old_ready)
    ready.update(implementation_ids=ids, ready_priority=priorities)
    return ready


def order_pending(rows, replacement_names):
    """Measure unaffected cohorts first; preserve each partition's input order."""
    return ([row for row in rows if row["name"] not in replacement_names]
            + [row for row in rows if row["name"] in replacement_names])


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("admission-manifest", "execution-manifest", "execution-receipts",
                 "source", "catalogue", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cohort", action="append", required=True)
    parser.add_argument("--max-fused-layers", type=int, required=True)
    args = parser.parse_args()
    admission = resume.read(args.admission_manifest)
    if len(args.cohort) != len(set(args.cohort)):
        raise ValueError("Duplicate split cohort")
    targets = [next(row for row in admission["cohorts"] if row["name"] == name)
               for name in args.cohort]
    target = targets[0]
    prep = target["command"][1]
    protocol = target["preparation_protocol"]
    if any(row["command"][1] != prep or row["preparation_protocol"] != protocol
           for row in targets):
        raise ValueError("Split cohorts must use the same sealed preparation")
    # The intermediate recovery manifests remain available and sealed as inputs
    # to the final split manifests; no prior campaign file is rewritten.
    intermediate = args.output / "resumed-before-split"
    resume.resume(args.admission_manifest, args.execution_manifest,
                  args.execution_receipts, prep, protocol, intermediate)
    combined = resume.read(intermediate / "campaign-admission.json")
    execution = resume.read(intermediate / "campaign-execution.json")
    source, catalogue = map(resume.read, (args.source, args.catalogue))
    replacements, split_provenance = {}, []
    extra_records = [resume.record(path) for path in
                     [args.source, args.catalogue, __file__,
                      intermediate / "campaign-admission.json",
                      intermediate / "campaign-execution.json"]]
    for target in targets:
        if sum(row["name"] == target["name"] for row in execution["cohorts"]) != 1:
            raise ValueError("Split target must still be pending")
        old_plan_path = argument(target["command"], "--plan")
        old_ready_path = argument(target["command"], "--ready-spec")
        old_plan, old_ready = map(resume.read, (old_plan_path, old_ready_path))
        if source["source_id"] != old_plan["source_id"] or old_ready["source_id"] != source["source_id"]:
            raise ValueError("Prefill source/plan/spec mismatch")
        if catalogue["source_id"] != admission["whole_source"]["source_id"]:
            raise ValueError("Catalogue whole-source binding differs")
        base = next(row for row in execution["cohorts"] if row["name"] == target["name"])
        new_rows, new_files = [], []
        for index, layers in enumerate(split_layers(old_plan, args.max_fused_layers)):
            name = target["name"] + "-part" + str(index)
            plan = copy.deepcopy(old_plan)
            plan["r18"]["fused_layers"] = layers
            if "r18_layer_ffn" in plan:
                plan["r18_layer_ffn"] = {key: value for key, value in plan["r18_layer_ffn"].items()
                                         if key in {str(layer) for layer in layers}}
            plan["plan_id"] = node.digest({k: v for k, v in plan.items() if k != "plan_id"})
            ready = derive_ready(source, plan, old_ready, catalogue)
            ready["cpu_prepare_directory"] = str((args.output / (name + "-cpu-preparation")).absolute())
            paths = [args.output / (name + ".plan.json"), args.output / (name + ".spec.json")]
            for path, value in zip(paths, (plan, ready)):
                with path.open("x") as stream:
                    json.dump(value, stream, indent=2, allow_nan=False)
                    stream.write("\n")
            new_files.extend(paths)
            row = copy.deepcopy(base)
            old_directory = argument(row["command"], "--output")
            directory = str((args.output / ("train-" + name)).absolute())
            row.update(name=name, plan_id=plan["plan_id"], numeric_gate_file=str(Path(directory) / "gate.json"))
            for flag, value in (("--plan", str(paths[0].absolute())),
                                ("--ready-spec", str(paths[1].absolute())), ("--output", directory)):
                row["command"][row["command"].index(flag) + 1] = value
            row["admission_command"] = [value.replace(old_directory, directory)
                                         for value in row["admission_command"]]
            new_rows.append(row)
        extra_records.extend(resume.record(path) for path in
                             [old_plan_path, old_ready_path, *new_files])
        replacements[target["name"]] = new_rows
        split_provenance.append({"target": target["name"], "layers": split_layers(old_plan, args.max_fused_layers),
                                 "replacement_cohorts": [row["name"] for row in new_rows]})
    final_admission_count = len(combined["cohorts"]) + sum(len(rows) - 1 for rows in replacements.values())
    final_execution_count = len(execution["cohorts"]) + sum(len(rows) - 1 for rows in replacements.values())
    for manifest, filename in ((combined, "campaign-admission.json"),
                               (execution, "campaign-execution.json")):
        manifest["cohorts"] = [new for row in manifest["cohorts"]
                               for new in replacements.get(row["name"], [row])]
        if filename == "campaign-execution.json":
            new_names = {row["name"] for rows in replacements.values() for row in rows}
            manifest["cohorts"] = order_pending(manifest["cohorts"], new_names)
        manifest["dependencies"].extend(extra_records)
        # Recovery counts describe the input; publish final split counts
        # separately so the campaign is not mistaken for 18 pending requests.
        manifest["split_campaign_counts"] = {
            "admission_cohorts": final_admission_count,
            "pending_cohorts": final_execution_count,
            "retained_measured_cohorts": len(manifest["retained_receipt_backing"])}
        manifest["split_cohort_provenance"] = {
            "replacements": split_provenance,
            "execution_order_policy": "unsplit-pending-first-then-new-chunks",
            "execution_order": [row["name"] for row in manifest["cohorts"]]
                if filename == "campaign-execution.json" else None,
            "max_fused_layers": args.max_fused_layers,
            "complete_model_outputs": 128, "old_local_prices_reused": False,
            "memory_guards_changed": False, "producer": resume.record(__file__)}
        with (args.output / filename).open("x") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"admission_cohorts": len(combined["cohorts"]),
                      "execution_cohorts": len(execution["cohorts"]),
                      "retained_cohorts": len(combined["retained_receipt_backing"]),
                      "rejected_attempts": len(combined["rejected_cohorts"]),
                      "admission": resume.record(args.output / "campaign-admission.json"),
                      "execution": resume.record(args.output / "campaign-execution.json")}))


if __name__ == "__main__":
    main()
