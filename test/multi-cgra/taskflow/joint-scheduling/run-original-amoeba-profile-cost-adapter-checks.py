#!/usr/bin/env python3
"""Exercise the native original-AMOEBA profile adapter on captured evidence.

This opt-in fixture runner consumes the completed input-0 profile and
orchestration artifacts. It does not run TaskProfiler or a mapper. The
optimizer must be built with both original-AMOEBA passes registered.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

APPLICATIONS = ("llama", "lu", "harris", "radar", "gcn", "raytracing")
SOURCE_CERTIFIED_APPLICATIONS = ("llama", "harris", "radar")
DEFAULT_SOURCE_REPOSITORY = "git@github.com:guosran/amoeba.git"
DEFAULT_SOURCE_COMMIT = "a57376e7043b1681e64e7169c5a8cb02eb192331"
DEFAULT_ARCHITECTURE_CONTRACT = (
    "neura-architecture-v1:amoeba_4x4_full_mesh_context12"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        fail(f"expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fail(message: str) -> None:
    raise RuntimeError(message)


def run_pass(args: argparse.Namespace, app: str, label: str,
             pass_name: str, input_ir: Path, options: dict[str, str],
             *, expect_success: bool,
             output_ir: Path | None = None) -> subprocess.CompletedProcess[str]:
    log_path = args.output_root / "logs" / f"{app}-{label}.log"
    command_path = args.output_root / "logs" / f"{app}-{label}.command.json"
    command = [
        str(args.optimizer), str(input_ir), "--verify-each",
        f"--architecture-spec={args.architecture}",
        f"--{pass_name}=" + " ".join(f"{key}={value}" for key, value in options.items()),
        "-o", str(output_ir) if output_ir is not None else "/dev/null",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout)
    write_json(command_path, {"argv": command, "exit_code": result.returncode})
    if expect_success and result.returncode != 0:
        fail(f"{label} failed for {app} ({result.returncode}); see {log_path}")
    if not expect_success and result.returncode == 0:
        fail(f"{label} unexpectedly succeeded for {app}; see {log_path}")
    return result


def adapter_options(args: argparse.Namespace, proof: Path, profiles: Path,
                    output: Path, function: str) -> dict[str, str]:
    return {
        "function": function,
        "body-export-file": str(proof),
        "profile-file": str(profiles),
        "ensemble-file": str(args.model_dir / "ensemble.json"),
        "checkpoint-dir": str(args.model_dir),
        "architecture-contract": args.architecture_contract,
        "architecture-path": str(args.architecture),
        "source-git-repository": args.source_repository,
        "source-git-commit": args.source_commit,
        "output": str(output),
    }


def retimer_options(costs: Path, proof: Path, output: Path,
                    function: str) -> dict[str, str]:
    return {
        "function": function,
        "parent-cost-file": str(costs),
        "profile-body-export-file": str(proof),
        "output": str(output),
    }


def expect_no_output(path: Path, label: str) -> None:
    if path.exists():
        fail(f"{label} published a partial output despite failure: {path}")


def exercise_negative_controls(args: argparse.Namespace, app: str,
                               input_ir: Path, proof: Path, profiles: Path,
                               function: str, costs: Path) -> None:
    negative_root = args.output_root / "negative-controls" / app
    original_proof = read_json(proof)
    selected_shape = read_json(costs)["original_amoeba_profile_binding"][
        "tasks"][0]["selected_cgra_shape"]

    failed_unselected_profiles = read_json(profiles)
    first_task_profiles = failed_unselected_profiles["tasks"][0]["profiles"]
    unselected = next((profile for profile in first_task_profiles
                       if profile["composed_cgra_shape"] != selected_shape), None)
    if unselected is None:
        fail(f"no unselected mapper profile is available for {app}")
    unselected["mapper_succeeded"] = False
    failed_profile_path = negative_root / "unselected-failure-profile.json"
    write_json(failed_profile_path, failed_unselected_profiles)
    failed_profile_costs = negative_root / "unselected-failure-costs.json"
    run_pass(args, app, "unselected-profile-failed",
             "adapt-original-amoeba-profile-costs", input_ir,
             adapter_options(args, proof, failed_profile_path,
                             failed_profile_costs, function), expect_success=True)
    failed_catalog = read_json(failed_profile_costs)
    if failed_catalog["original_amoeba_profile_binding"]["tasks"][0][
        "mapper_succeeded"] is not True:
        fail("selected mapper success was not preserved after an unselected failure")

    failed_selected_profiles = read_json(profiles)
    selected = next((profile for profile in failed_selected_profiles["tasks"][0][
                     "profiles"]
                     if profile["composed_cgra_shape"] == selected_shape), None)
    if selected is None:
        fail(f"selected profile is absent for {app}")
    selected["mapper_succeeded"] = False
    failed_selected_path = negative_root / "selected-failure-profile.json"
    write_json(failed_selected_path, failed_selected_profiles)
    failed_selected_costs = negative_root / "selected-failure-costs.json"
    run_pass(args, app, "selected-profile-failed",
             "adapt-original-amoeba-profile-costs", input_ir,
             adapter_options(args, proof, failed_selected_path,
                             failed_selected_costs, function), expect_success=False)
    expect_no_output(failed_selected_costs, "selected-failure adapter")

    stale_trip = copy.deepcopy(original_proof)
    stale_trip["tasks"][0]["static_trip_count"] += 1
    stale_trip_path = negative_root / "stale-trip-proof.json"
    write_json(stale_trip_path, stale_trip)
    stale_trip_cost = negative_root / "stale-trip-costs.json"
    run_pass(args, app, "stale-trip", "adapt-original-amoeba-profile-costs",
             input_ir,
             adapter_options(args, stale_trip_path, profiles,
                             stale_trip_cost, function), expect_success=False)
    expect_no_output(stale_trip_cost, "stale-trip adapter")

    tampered_proof = copy.deepcopy(original_proof)
    tampered_proof["tasks"][0]["normalized_mapper_body"] += " // altered export"
    tampered_proof_path = negative_root / "tampered-body-proof.json"
    write_json(tampered_proof_path, tampered_proof)
    tampered_adapter_costs = negative_root / "tampered-body-adapter-costs.json"
    run_pass(args, app, "tampered-export-body",
             "adapt-original-amoeba-profile-costs", input_ir,
             adapter_options(args, tampered_proof_path, profiles,
                             tampered_adapter_costs, function),
             expect_success=False)
    expect_no_output(tampered_adapter_costs, "tampered-body adapter")

    incomplete_profiles = read_json(profiles)
    incomplete_profiles["tasks"][0]["profiles"].pop()
    incomplete_profiles["expected_candidate_count"] -= 1
    incomplete_profiles["completed_candidate_count"] -= 1
    incomplete_profile_path = negative_root / "incomplete-profile-domain.json"
    write_json(incomplete_profile_path, incomplete_profiles)
    incomplete_cost = negative_root / "incomplete-profile-costs.json"
    run_pass(args, app, "incomplete-profile-domain",
             "adapt-original-amoeba-profile-costs", input_ir,
             adapter_options(args, proof, incomplete_profile_path,
                             incomplete_cost, function), expect_success=False)
    expect_no_output(incomplete_cost, "incomplete-profile adapter")

    catalog = read_json(costs)
    binding = catalog["original_amoeba_profile_binding"]
    binding["tasks"][0]["normalized_mapper_body"] += " // stale body"
    stale_body_catalog = negative_root / "stale-body-costs.json"
    write_json(stale_body_catalog, catalog)
    stale_body_result = negative_root / "stale-body-retiming.json"
    run_pass(args, app, "stale-body-binding",
             "retime-original-amoeba-fixed-decisions", input_ir,
             retimer_options(stale_body_catalog, proof, stale_body_result,
                             function), expect_success=False)
    expect_no_output(stale_body_result, "stale-body retimer")

    forged_matching_proof = copy.deepcopy(original_proof)
    forged_matching_catalog = read_json(costs)
    forged_body = forged_matching_proof["tasks"][0]["normalized_mapper_body"]
    forged_body += " // same altered export and catalog"
    forged_matching_proof["tasks"][0]["normalized_mapper_body"] = forged_body
    forged_matching_catalog["original_amoeba_profile_binding"]["tasks"][0][
        "normalized_mapper_body"] = forged_body
    forged_proof_path = negative_root / "forged-matching-body-proof.json"
    forged_catalog_path = negative_root / "forged-matching-body-costs.json"
    write_json(forged_proof_path, forged_matching_proof)
    write_json(forged_catalog_path, forged_matching_catalog)
    forged_result = negative_root / "forged-matching-body-retiming.json"
    run_pass(args, app, "forged-matching-export-and-catalog-body",
             "retime-original-amoeba-fixed-decisions", input_ir,
             retimer_options(forged_catalog_path, forged_proof_path,
                             forged_result, function), expect_success=False)
    expect_no_output(forged_result, "forged matching body retimer")

    source_ir = input_ir.read_text()
    mutated_ir_text = source_ir.replace(
        "value = 256 : index", "value = 257 : index", 1)
    if mutated_ir_text == source_ir:
        fail(f"no kernel constant was available for current-IR mutation: {input_ir}")
    mutated_ir = negative_root / "mutated-kernel.mlir"
    mutated_ir.parent.mkdir(parents=True, exist_ok=True)
    mutated_ir.write_text(mutated_ir_text)
    mutated_ir_result = negative_root / "mutated-kernel-retiming.json"
    run_pass(args, app, "mutated-current-kernel-body",
             "retime-original-amoeba-fixed-decisions", mutated_ir,
             retimer_options(costs, proof, mutated_ir_result, function),
             expect_success=False)
    expect_no_output(mutated_ir_result, "mutated-current-kernel retimer")

    source_ir = input_ir.read_text()
    if "nontemporal = false" not in source_ir:
        fail(f"no default-false memref.load attribute was available in {input_ir}")
    true_nontemporal_ir = negative_root / "true-nontemporal.mlir"
    true_nontemporal_ir.write_text(
        source_ir.replace("nontemporal = false", "nontemporal = true", 1)
    )
    true_nontemporal_costs = negative_root / "true-nontemporal-costs.json"
    run_pass(args, app, "true-nontemporal-is-semantic",
             "adapt-original-amoeba-profile-costs", true_nontemporal_ir,
             adapter_options(args, proof, profiles, true_nontemporal_costs,
                             function), expect_success=False)
    expect_no_output(true_nontemporal_costs, "true-nontemporal adapter")

    unknown_nontemporal_ir = negative_root / "unknown-nontemporal.mlir"
    unknown_nontemporal_ir.write_text(
        source_ir.replace('nontemporal = false',
                          'nontemporal = "unknown"', 1)
    )
    unknown_nontemporal_costs = negative_root / "unknown-nontemporal-costs.json"
    run_pass(args, app, "unknown-nontemporal-is-rejected",
             "adapt-original-amoeba-profile-costs", unknown_nontemporal_ir,
             adapter_options(args, proof, profiles, unknown_nontemporal_costs,
                             function), expect_success=False)
    expect_no_output(unknown_nontemporal_costs, "unknown-nontemporal adapter")

    stale_domain_ir_text = re.sub(
        r'(amoeba\.source_iteration_control_binding = ")([^"\n]+)',
        lambda match: match.group(1) + "X" + match.group(2)[1:],
        source_ir,
        count=1,
    )
    if stale_domain_ir_text == source_ir:
        fail(f"no source-domain control binding was available in {input_ir}")
    stale_domain_ir = negative_root / "stale-source-domain-binding.mlir"
    stale_domain_ir.write_text(stale_domain_ir_text)
    stale_domain_costs = negative_root / "stale-source-domain-costs.json"
    run_pass(args, app, "stale-source-domain-binding",
             "adapt-original-amoeba-profile-costs", stale_domain_ir,
             adapter_options(args, proof, profiles, stale_domain_costs,
                             function), expect_success=False)
    expect_no_output(stale_domain_costs, "stale-source-domain adapter")

    catalog = read_json(costs)
    catalog["original_amoeba_profile_binding"]["tasks"][0][
        "selected_cgra_shape"] = "4x1"
    stale_shape_catalog = negative_root / "stale-shape-costs.json"
    write_json(stale_shape_catalog, catalog)
    stale_shape_result = negative_root / "stale-shape-retiming.json"
    run_pass(args, app, "stale-shape-binding",
             "retime-original-amoeba-fixed-decisions", input_ir,
             retimer_options(stale_shape_catalog, proof, stale_shape_result,
                             function), expect_success=False)
    expect_no_output(stale_shape_result, "stale-shape retimer")

    write_failure = negative_root / "missing-parent" / "costs.json"
    run_pass(args, app, "write-failure",
             "adapt-original-amoeba-profile-costs", input_ir,
             adapter_options(args, proof, profiles, write_failure, function),
             expect_success=False)
    expect_no_output(write_failure, "write-failure adapter")


def check_outputs(costs: Path, retimed: Path, task_count: int) -> None:
    catalog = read_json(costs)
    binding = catalog.get("original_amoeba_profile_binding")
    entries = catalog.get("entries")
    metadata = catalog.get("predictor_metadata")
    if not isinstance(binding, dict) or len(binding.get("tasks", [])) != task_count:
        fail(f"catalog did not bind all {task_count} body/profile tasks: {costs}")
    if not isinstance(metadata, dict) or \
       metadata.get("iteration_domain_coverage_status") != \
           "verified-unsharded-source-domain-v1" or \
       metadata.get("iteration_domain_coverage_verified") is not True:
        fail(f"catalog omitted verified source-domain coverage: {costs}")
    if metadata.get("body_equivalence_checked") is not True or \
       binding.get("current_ir_body_equivalence_verified") is not True:
        fail(f"catalog omitted current-IR normalized-body verification: {costs}")
    if binding.get("candidate_id") != "candidate-0" or \
       not binding.get("graph_variant_id"):
        fail(f"catalog omitted stable replay candidate/graph identity: {costs}")
    if binding.get("source_iteration_domain_coverage_verified") is not True:
        fail(f"catalog omitted bound source-domain evidence: {costs}")
    if not isinstance(entries, list) or len(entries) != task_count:
        fail(f"catalog does not contain one selected-shape cost per task: {costs}")

    result = read_json(retimed)
    if result.get("valid") is not True or result.get("formal_go") is not False:
        fail(f"retimed output has invalid diagnostic/formal status: {retimed}")
    if result.get("selected_profile_body_binding_verified") is not True:
        fail(f"retimed output did not verify its exact body binding: {retimed}")
    if result.get("body_equivalence_checked") is not True or \
       result.get("body_equivalence_status") != \
       "verified-current-kernel-equals-exported-original-normalized-body":
        fail(f"retimed output did not verify current-kernel body equality: {retimed}")
    if result.get("iteration_domain_coverage_status") != \
           "verified-unsharded-source-domain-v1" or \
       result.get("iteration_domain_coverage_verified") is not True:
        fail(f"retimed output omitted verified source-domain coverage: {retimed}")
    if "mapper_replay_verified" in result:
        fail(f"retimed output must not claim mapper replay verification: {retimed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-root", type=Path,
        help="directory containing source-certified <workload>/{orchestrated.mlir,static-body-proof.json,task-profiles.json}",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--architecture", type=Path)
    parser.add_argument("--architecture-contract", default=DEFAULT_ARCHITECTURE_CONTRACT)
    parser.add_argument("--source-repository", default=DEFAULT_SOURCE_REPOSITORY)
    parser.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--apps", nargs="+", choices=APPLICATIONS,
                        default=list(SOURCE_CERTIFIED_APPLICATIONS))
    parser.add_argument("--skip-negative-controls", action="store_true")
    args = parser.parse_args()
    args.optimizer = args.optimizer.resolve()
    args.artifact_root = args.artifact_root.resolve()
    args.baseline_root = (args.baseline_root or
                          args.artifact_root /
                          "results/input0-amoeba-full-correct-flow").resolve()
    args.output_root = args.output_root.resolve()
    args.model_dir = (args.model_dir or
                      args.artifact_root / ".work/formal-model-nohash-v2-trained-rerun").resolve()
    args.architecture = (args.architecture or
                         args.artifact_root /
                         "config/architectures/amoeba_4x4_full_mesh_context12.yaml").resolve()
    if not args.optimizer.is_file():
        fail(f"optimizer is missing: {args.optimizer}")
    if not (args.model_dir / "ensemble.json").is_file() or \
       not args.architecture.is_file():
        fail("formal model files or architecture spec are missing")
    args.output_root.mkdir(parents=True, exist_ok=False)

    for app in args.apps:
        app_root = args.baseline_root / app
        input_ir = app_root / "orchestrated.mlir"
        proof = app_root / "static-body-proof.json"
        profiles = app_root / "task-profiles.json"
        if not all(path.is_file() for path in (input_ir, proof, profiles)):
            fail(f"completed body/profile/orchestration fixture is incomplete: {app_root}")
        proof_json = read_json(proof)
        function = proof_json.get("function")
        task_count = proof_json.get("task_count")
        if not isinstance(function, str) or not isinstance(task_count, int):
            fail(f"body proof is missing function/task count: {proof}")

        costs = args.output_root / "costs" / f"{app}.json"
        retimed = args.output_root / "retimed" / f"{app}.json"
        roundtrip = args.output_root / "roundtrip" / f"{app}.mlir"
        costs.parent.mkdir(parents=True, exist_ok=True)
        retimed.parent.mkdir(parents=True, exist_ok=True)
        roundtrip.parent.mkdir(parents=True, exist_ok=True)
        run_pass(args, app, "adapter", "adapt-original-amoeba-profile-costs",
                 input_ir,
                 adapter_options(args, proof, profiles, costs, function),
                 expect_success=True, output_ir=roundtrip)
        run_pass(args, app, "retimer", "retime-original-amoeba-fixed-decisions",
                 roundtrip,
                 retimer_options(costs, proof, retimed, function),
                 expect_success=True)
        check_outputs(costs, retimed, task_count)
        if not args.skip_negative_controls and app == args.apps[0]:
            exercise_negative_controls(args, app, input_ir, proof, profiles,
                                       function, costs)
        print(f"PASS {app}: {task_count} exact profile/body bindings", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
