#!/usr/bin/env python3
"""Prepare measured full-Qwen coverage of registered r19 execution choices.

Old campaigns contribute configurations, not prices. Every cohort runs the new
source-ready executor with complete generation. CPU regions retain their actual
pending registration IDs; a policy that would clone an exported alias is never
substituted for a legal copy operation.
"""

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": str(path), "sha256": checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("parent-campaign", "parent-catalogue", "cpu-registration",
                 "base-plan", "source", "whole-source", "frozen-runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--half-artifact", type=Path, action="append", required=True)
    parser.add_argument("--runtime-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--aie-root", type=Path, default=Path("/home/shiran/project/aie-toy"))
    args = parser.parse_args()
    source, parent, catalogue = map(read, (args.source, args.parent_campaign, args.parent_catalogue))
    if source["source_id"] != parent["source_id"]:
        raise ValueError("Parent campaign source differs")
    whole = record(args.whole_source)
    whole["source_id"] = catalogue["source_id"]
    baseline = read(args.base_plan)
    cpu = read(args.cpu_registration)["implementations"]
    grouped = {}
    for row in cpu:
        config = row["runtime_config"]
        key = (config["backend"], config["input_policy"], config["output_policy"])
        choices = grouped.setdefault(tuple(row["operators"]), {})
        if key in choices:
            raise ValueError("Repeated CPU policy registration")
        choices[key] = row
    covered = [name for names in grouped for name in names]
    if len(covered) != len(set(covered)):
        raise ValueError("Training region set needs disjoint cover coloring")
    spec = importlib.util.spec_from_file_location(
        "r19_node_contract", Path(__file__).resolve().parents[2] / "scripts/qwen-node-runtime.py")
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    args.output.mkdir(parents=True, exist_ok=False)
    cohorts, dependencies = [], {}

    def depend(path):
        row = record(path)
        # Preserve the spelling passed to loaders as well as the blob identity.
        # Hugging Face snapshot files are symlinks; checking only the old blob
        # would miss a changed snapshot link that the actual loader follows.
        row["requested_file"] = str(Path(path).absolute())
        dependencies[row["file"]] = row

    def write(path, value):
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        depend(path)

    def implementation_ids(plan):
        result = {}
        routes = plan.get("r18_layer_ffn", {})
        for identifier, registration in catalogue["runtime_registration"]["implementations"].items():
            kind = registration["kind"]
            if kind == "prefill-original":
                name = registration["operator"]
                if plan["placements"].get(name) == registration["device"]:
                    result[name] = identifier
            elif kind == "prefill-gemm":
                selected = plan["r18"]["gemm_variants"].get(registration["operator"])
                if selected and selected["manifest_sha256"] == registration["manifest_sha256"]:
                    result[registration["operator"]] = identifier
            elif kind == "prefill-fused-ffn" and registration["layer"] in plan["r18"]["fused_layers"]:
                layer = registration["layer"]
                route = routes.get(str(layer), plan["r18"])
                if all(route[key] == registration[key] for key in ("retention", "split_order", "ffn_artifact")):
                    result[f"prefill.L{layer}.gate_up"] = identifier
        # Registry order must not let the original overwrite its GEMM variant.
        for name, selected in plan["r18"]["gemm_variants"].items():
            matches = [identifier for identifier, row in catalogue["runtime_registration"]["implementations"].items()
                       if row["kind"] == "prefill-gemm" and row["operator"] == name
                       and row["manifest_sha256"] == selected["manifest_sha256"]]
            if len(matches) != 1:
                raise ValueError("GEMM registration does not resolve uniquely")
            result[name] = matches[0]
        return result

    prompt = [785, 6722, 315, 9625, 374]

    def add(name, plan, decode, regions=(), halves=None, prefetch=True):
        plan = copy.deepcopy(plan)
        plan["plan_id"] = digest({k: v for k, v in plan.items() if k != "plan_id"})
        node.validate_plan(source, plan)
        plan_path, ready_path = args.output / (name + ".plan.json"), args.output / (name + ".spec.json")
        ready = dict(schema="orbit.qwen-r19-ready-spec.v1", source_id=source["source_id"],
                     whole_source=whole, source_ready=True, decode_rope_prefetch=prefetch,
                     operators=halves or {}, cpu_regions=list(regions),
                     implementation_ids=implementation_ids(plan), ready_priority={},
                     prepare_tokens=prompt,
                     cpu_prepare_directory=str((args.output / (name + "-cpu-preparation")).resolve()))
        for operator, parameters in ready["operators"].items():
            ready["implementation_ids"][operator] = "r19-heterogeneous-" + digest(
                dict(operator=operator, parameters=parameters))
        ordinals = {op["id"]: op["source_ordinal"] for op in source["operators"]}
        region_members = set()
        for region in regions:
            ready["ready_priority"][region["id"]] = min(ordinals[n] for n in region["operators"])
            region_members.update(region["operators"])
        for layer in plan["r18"]["fused_layers"]:
            names = [f"prefill.L{layer}.{role}" for role in
                     ("gate_up", "gate_split", "swiglu", "down")]
            identifier = ready["implementation_ids"][names[0]]
            ready["ready_priority"][identifier] = ordinals[names[0]]
            region_members.update(names)
        for op in source["operators"]:
            if op["id"] not in region_members:
                ready["ready_priority"][ready["implementation_ids"][op["id"]]] = op["source_ordinal"]
        write(plan_path, plan)
        write(ready_path, ready)
        directory = (args.output / ("train-" + name)).resolve()
        command = [str(args.aie_root / ".venv/bin/python"),
                   str(args.runtime_dir.resolve() / "run-qwen-r19-ready.py"),
                   "--frozen-runtime", str(args.frozen_runtime.resolve()),
                   "--ready-spec", str(ready_path.resolve()),
                   "--aie-root", str(args.aie_root), "--source", str(args.source.resolve()),
                   "--plan", str(plan_path.resolve()), "--output", str(directory),
                   "--decode-variant-dir", str(decode), "--outputs", "128",
                   "--warmups", "1", "--repeats", "3"]
        reference = args.aie_root / "results/qwen25-3b/hf-bf16-P5-capture.npz"
        audit = [str(args.aie_root / ".venv/bin/python"),
                 str(args.runtime_dir.resolve() / "audit-qwen-r19-ready.py"),
                 "--execution", str(directory), "--output", str(directory / "gate.json"),
                 "--reference", "hf-bf16=" + str(reference)]
        cohorts.append(dict(name=name, purpose="training", command=command,
                            admission_command=audit,
                            plan_id=plan["plan_id"], measured_requests=3,
                            numeric_gate_file=str(directory / "gate.json")))

    for cohort in parent["cohorts"]:
        command = cohort["command"]
        decode = command[command.index("--decode-variant-dir") + 1]
        add("parent-" + cohort["name"], read(cohort["plan"]), decode)
    decode_root = Path(parent["cohorts"][0]["command"][
        parent["cohorts"][0]["command"].index("--decode-variant-dir") + 1]).parent
    selected_decode = decode_root / "decode-r0-cache1-coalesce1"
    for backend, input_policy, output_policy in itertools.product(
            ("eager-host-group", "inductor-cpu"), ("borrow-host", "stage-host-copy"),
            ("retain-host", "copy-host")):
        regions = []
        for names, policies in grouped.items():
            key = (backend, input_policy, output_policy)
            if key not in policies:
                # Exported views cannot be cloned while claiming the same ABI.
                key = (backend, input_policy, "retain-host")
            row = policies[key]
            regions.append(dict(id=row["id"], operators=list(names),
                                **{k: row["runtime_config"][k] for k in
                                   ("backend", "input_policy", "output_policy")}))
        name = "cpu-" + "-".join((backend, input_policy, output_policy))
        add(name, baseline, selected_decode, regions)
    for artifact, side in itertools.product(args.half_artifact, ("gate", "up")):
        plan = copy.deepcopy(baseline)
        selected = {}
        for op in source["operators"]:
            if op["id"].endswith(".gate_up"):
                plan["placements"][op["id"]] = "cpu"
                selected[op["id"]] = dict(cpu_half=side, directory=str(artifact.resolve()),
                                          manifest_sha256=record(artifact / "implementation.json")["sha256"])
        partition = node.make_plan(source, plan["placements"])
        plan["regions"] = partition["regions"]
        add("half-" + artifact.name + "-cpu-" + side, plan, selected_decode, halves=selected)
    for coalesce in (0, 1):
        add("decode-no-prefetch-coalesce" + str(coalesce), baseline,
            decode_root / ("decode-r0-cache1-coalesce" + str(coalesce)), prefetch=False)
    for path in (args.parent_campaign, args.parent_catalogue, args.cpu_registration,
                 args.base_plan, args.source, args.whole_source, Path(__file__),
                 args.runtime_dir / "run-r19-campaign.py"):
        depend(path)
    for name in ("run-qwen-r19-ready.py", "qwen-r19-ready.py", "qwen-r19-xrt.py",
                 "qwen-r19-regions.py", "qwen-r19-source-ready.py",
                 "qwen-r19-decode-ready.py"):
        depend(args.runtime_dir / name)
    depend(args.runtime_dir / "routed-r18-runtime.py")
    depend(args.runtime_dir / "audit-qwen-r19-ready.py")
    for path in args.frozen_runtime.rglob("*.py"):
        depend(path)
    for binding in source["bindings"]:
        depend(binding["file"])
    paths_file = args.aie_root / "results/qwen25-3b/model-paths.json"
    model_paths = read(paths_file)
    depend(paths_file)
    depend(model_paths["model"])
    for path in Path(model_paths["tokenizer"]).iterdir():
        if path.is_file():
            depend(path)
    depend(args.aie_root / "results/qwen25-3b/hf-bf16-P5-capture.npz")
    depend(args.aie_root / "build/qwen25-3b/Triton-XDNA/third_party/mlir-air-src/"
           "programming_examples/llms/verify/comparators.py")
    result = dict(schema="orbit.qwen-r19-sealed-campaign.v1", source_id=source["source_id"],
                  whole_source=whole, dependencies=list(dependencies.values()), cohorts=cohorts,
                  configuration_parent=record(args.parent_campaign),
                  parent_timings_reused=False, complete_plan_prepricing=False,
                  numerical_admission="Each completed cohort requires the unchanged original numerical gate")
    write(args.output / "campaign.json", result)
    print(json.dumps(dict(cohorts=len(cohorts), cpu_covers=len(grouped),
                          manifest=record(args.output / "campaign.json"))))


if __name__ == "__main__":
    main()
