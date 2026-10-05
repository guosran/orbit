#!/usr/bin/env python3
"""Check actual II23 adapter/retimer evidence with isolated tampered copies."""

import argparse
import copy
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def options(argv, prefix):
    matches = [arg for arg in argv if arg.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("command must contain exactly one " + prefix)
    return dict(item.split("=", 1) for item in matches[0][len(prefix):].split())


def command(argv, prefix, updates, output_ir, cpu):
    result = list(argv)
    values = options(result, prefix)
    values.update(updates)
    index = next(i for i, arg in enumerate(result) if arg.startswith(prefix))
    result[index] = prefix + " ".join(key + "=" + str(value) for key, value in values.items())
    result[result.index("-o") + 1] = str(output_ir)
    if result[:2] == ["taskset", "--cpu-list"]:
        result[2] = str(cpu)
    else:
        result = ["taskset", "--cpu-list", str(cpu)] + result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-command", type=Path, required=True)
    parser.add_argument("--retimer-command", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu", type=int, choices=range(12), default=5)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    adapter = read(args.adapter_command)["argv"]
    ap = "--adapt-original-amoeba-profile-costs="
    ao = options(adapter, ap)
    if ao.get("diagnostic-ii-ceiling") != "23":
        raise ValueError("adapter command must explicitly select II23")
    profiles = read(ao["profile-file"])
    reports = []

    def run(label, argv, target, succeeds):
        directory = root / label
        directory.mkdir()
        (directory / "command.json").write_text(json.dumps({"argv": argv}, indent=2) + "\n")
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            completed = subprocess.run(argv, stdout=stdout, stderr=stderr)
        text = (directory / "stderr.log").read_text(errors="replace")
        passed = (completed.returncode == 0 and target.is_file()) if succeeds else (
            completed.returncode != 0 and not target.exists() and "error:" in text)
        reports.append({"case": label, "expected_success": succeeds,
                        "exit_code": completed.returncode, "pass": passed})
        (root / "acceptance.json").write_text(json.dumps({"cases": reports}, indent=2) + "\n")
        if not passed:
            raise RuntimeError(label + " did not meet the expected outcome: " + text[-1000:])

    catalog = root / "positive-cost-catalog.json"
    run("adapter-positive", command(adapter, ap, {"output": catalog}, root / "positive-adapter.mlir", args.cpu), catalog, True)
    mutations = (
        ("missing-attempt", lambda value: value["candidate_attempts"].pop()),
        ("wrong-shape-index", lambda value: value["candidate_attempts"][0].__setitem__("candidate_index_in_task", 2)),
        ("wrong-cgra-count", lambda value: value["candidate_attempts"][0].__setitem__("composed_cgra_count", 2)),
        ("forged-profile-created", lambda value: value["candidate_attempts"][0].__setitem__("profile_created", False)),
        ("forged-success", lambda value: next(row for row in value["candidate_attempts"] if not row["mapper_succeeded"]).__setitem__("mapper_succeeded", True)),
    )
    for label, mutate in mutations:
        value = copy.deepcopy(profiles)
        mutate(value)
        path = root / (label + "-profiles.json")
        path.write_text(json.dumps(value) + "\n")
        target = root / (label + "-costs.json")
        run(label, command(adapter, ap, {"profile-file": path, "output": target}, root / (label + ".mlir"), args.cpu), target, False)
    target = root / "implicit20-costs.json"
    run("reject-runtime23-as-training20", command(adapter, ap, {"diagnostic-ii-ceiling": "20", "output": target}, root / "implicit20.mlir", args.cpu), target, False)

    if args.retimer_command:
        retimer = read(args.retimer_command)["argv"]
        rp = "--retime-original-amoeba-fixed-decisions="
        ro = options(retimer, rp)
        if ro.get("diagnostic-ii-ceiling") != "23":
            raise ValueError("retimer command must explicitly select II23")
        original_catalog = read(catalog)
        target = root / "positive-retiming.json"
        run("retimer-positive", command(retimer, rp, {
            "parent-cost-file": catalog, "output": target,
        }, root / "positive-retiming.mlir", args.cpu), target, True)

        def drop_override(value):
            value["predictor_metadata"].pop("diagnostic_override")
            value["original_amoeba_profile_binding"].pop("diagnostic_override")

        def forge_unsupported(value):
            next(row for row in value["entries"] if row["support_status"] == "unsupported")["predicted_ii"] = 23

        catalog_mutations = (
            ("missing-override", drop_override),
            ("mismatched-override", lambda value: value["predictor_metadata"]["diagnostic_override"].__setitem__("runtime_ii_ceiling", 22)),
            ("forged-formal-go", lambda value: value["predictor_metadata"]["diagnostic_override"].__setitem__("formal_go", True)),
            ("synthetic-unsupported-ii", forge_unsupported),
            ("forged-body-binding", lambda value: value["original_amoeba_profile_binding"]["tasks"][0].__setitem__("normalized_mapper_body", "forged")),
        )
        for label, mutate in catalog_mutations:
            value = copy.deepcopy(original_catalog)
            mutate(value)
            path = root / (label + "-catalog.json")
            path.write_text(json.dumps(value) + "\n")
            target = root / (label + "-retiming.json")
            run(label, command(retimer, rp, {"parent-cost-file": path, "output": target}, root / (label + ".mlir"), args.cpu), target, False)
    print(json.dumps({"status": "pass", "case_count": len(reports), "output_root": str(root)}))


if __name__ == "__main__":
    main()
