#!/usr/bin/env python3
"""Focused positive/negative checks for the direct 2x2 II=23 diagnostic.

The predictor command JSON names the canonical ORBIT Ray source module. The
script creates runtime YAML from the model bundle's exact training YAML, runs
prediction on that source, and verifies all 216 task-shape rows: 211 numeric
predictions and five shapes that exceed the runtime II=23 domain. It then
checks three fail-closed CLI cases. A retimer command is optional; when
supplied, it also checks rejection of a catalog with its diagnostic proof
removed.
"""

import argparse
import json
import math
import shlex
import subprocess
import tempfile
from pathlib import Path


TRAINING_CEILING = 20.0
RUNTIME_CEILING = 23.0
DIAGNOSTIC_SCHEMA = "per-cgra-2x2-ii-extrapolation-v1"
OUTPUT_RULE = (
    "min(lower_bound + softplus(logit), diagnostic_runtime_ii_ceiling)"
)
UNSUPPORTED_POLICY = (
    "analytical-lower-bound-exceeds-diagnostic-runtime-ceiling-v1"
)
UNSUPPORTED_REASON = (
    "analytical-lower-bound-exceeds-diagnostic-runtime-ceiling"
)
MEMBER_SEEDS = [17, 41, 113, 239]
EXPECTED_RAY_TASKS = 27
EXPECTED_SHAPES_PER_TASK = 8
EXPECTED_RUNTIME_23_SUPPORTED_PREDICTIONS = 211
EXPECTED_RUNTIME_23_UNSUPPORTED_ROWS = 5


def fail(message):
    raise SystemExit(message)


def load_argv(path):
    record = json.loads(Path(path).read_text())
    argv = record.get("argv")
    if not isinstance(argv, list) or not argv or not all(
        isinstance(arg, str) for arg in argv
    ):
        fail(f"{path}: expected a non-empty argv array")
    return argv


def update_pass_options(argv, pass_name, updates):
    prefix = f"--{pass_name}="
    result = list(argv)
    matches = [i for i, arg in enumerate(result) if arg.startswith(prefix)]
    if len(matches) != 1:
        fail(f"expected exactly one {prefix} option")
    index = matches[0]
    tokens = shlex.split(result[index][len(prefix) :])
    options = {}
    for token in tokens:
        key, separator, value = token.partition("=")
        if not separator or not key or key in options:
            fail(f"malformed sub-option in {pass_name}: {token}")
        options[key] = value
    options.update({key: str(value) for key, value in updates.items()})
    result[index] = prefix + " ".join(f"{key}={value}" for key, value in options.items())
    return result


def replace_architecture(argv, runtime_path):
    result = list(argv)
    prefix = "--architecture-spec="
    indexes = [i for i, arg in enumerate(result) if arg.startswith(prefix)]
    if len(indexes) > 1:
        fail("expected at most one --architecture-spec argument")
    if indexes:
        result[indexes[0]] = prefix + str(runtime_path)
    return result


def replace_mlir_output(argv, output_path):
    result = list(argv)
    for index, argument in enumerate(result[:-1]):
        if argument == "-o":
            result[index + 1] = str(output_path)
            return result
    fail("predictor command has no -o destination")


def replace_retimer_cost_and_output(argv, cost_path, output_path):
    result = update_pass_options(
        argv,
        "retime-original-amoeba-fixed-decisions",
        {"parent-cost-file": cost_path, "output": output_path},
    )
    for index, arg in enumerate(result[:-1]):
        if arg == "-o":
            result[index + 1] = str(Path(output_path).with_suffix(".mlir"))
            return result
    fail("retimer command has no -o destination")


def run_expect_success(argv, label):
    completed = subprocess.run(
        argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if completed.returncode:
        fail(f"{label} unexpectedly failed:\n{completed.stdout[-4000:]}")
    return completed


def run_expect_failure(argv, label):
    completed = subprocess.run(
        argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if completed.returncode == 0:
        fail(f"{label} unexpectedly succeeded")
    return completed


def close(lhs, rhs):
    return math.isfinite(lhs) and math.isfinite(rhs) and abs(lhs - rhs) <= 1e-6 * max(
        1.0, abs(lhs)
    )


def check_catalog(path, training_yaml, runtime_yaml):
    catalog = json.loads(Path(path).read_text())
    metadata = catalog.get("predictor_metadata", {})
    override = metadata.get("diagnostic_override")
    expected_override = {
        "schema": DIAGNOSTIC_SCHEMA,
        "training_ii_ceiling": TRAINING_CEILING,
        "runtime_ii_ceiling": RUNTIME_CEILING,
        "extrapolation_enabled": True,
        "formal": False,
        "output_rule": OUTPUT_RULE,
        "training_architecture_exact_yaml_text": training_yaml,
        "runtime_architecture_exact_yaml_text": runtime_yaml,
    }
    if override != expected_override:
        fail("positive catalog does not carry the exact diagnostic override proof")
    if metadata.get("model_interval_max_ii") != TRAINING_CEILING:
        fail("training ceiling was not preserved as exactly 20")
    if metadata.get("unsupported_prediction_policy") != UNSUPPORTED_POLICY:
        fail("diagnostic unsupported policy is missing or wrong")
    if metadata.get("diagnostic_only") is not True or metadata.get("formal") is not False:
        fail("diagnostic catalog is not explicitly non-formal")
    if metadata.get("production_ready") is not False:
        fail("diagnostic catalog unexpectedly claims production readiness")

    saw_extrapolated = False
    saw_unsupported = False
    supported_prediction_count = 0
    unsupported_row_count = 0
    entries = catalog.get("entries")
    if not isinstance(entries, list) or len(entries) != (
        EXPECTED_RAY_TASKS * EXPECTED_SHAPES_PER_TASK
    ):
        fail("catalog does not contain all 27 Ray tasks across 8 mapper shapes")
    task_shapes = {}
    for entry in entries:
        task = entry.get("task")
        shape = (entry.get("mapper_tile_rows"), entry.get("mapper_tile_cols"))
        if not isinstance(task, str) or not task:
            fail("catalog row is missing its source task identity")
        task_shapes.setdefault(task, set()).add(shape)
        if entry.get("support_status") == "unsupported":
            saw_unsupported = True
            unsupported_row_count += 1
            if (
                entry.get("unsupported_reason") != UNSUPPORTED_REASON
                or entry.get("model_interval_max_ii") != TRAINING_CEILING
                or entry.get("training_ceiling_ii") != TRAINING_CEILING
                or entry.get("runtime_ceiling_ii") != RUNTIME_CEILING
                or entry.get("extrapolation_status")
                != "outside-diagnostic-runtime-domain"
                or entry.get("analytical_lower_bound", 0) <= RUNTIME_CEILING
            ):
                fail("unsupported row does not prove it exceeds runtime ceiling 23")
            continue

        lower = entry.get("analytical_lower_bound")
        predicted = entry.get("predicted_ii")
        members = entry.get("direct_ensemble_members")
        if (
            entry.get("support_status") != "supported"
            or not isinstance(lower, (int, float))
            or not isinstance(predicted, (int, float))
            or not lower <= predicted <= RUNTIME_CEILING + 1e-6
            or entry.get("training_ceiling_ii") != TRAINING_CEILING
            or entry.get("runtime_ceiling_ii") != RUNTIME_CEILING
            or not isinstance(members, list)
            or len(members) != 4
        ):
            fail("supported diagnostic row violates its runtime interval proof")
        values = []
        for index, member in enumerate(members):
            value = member.get("predicted_ii")
            if (
                member.get("member_index") != index
                or member.get("seed") != MEMBER_SEEDS[index]
                or not isinstance(value, (int, float))
                or not lower <= value <= RUNTIME_CEILING + 1e-6
            ):
                fail("diagnostic member is invalid or outside [lower_bound, 23]")
            values.append(float(value))
        mean = sum(values) / 4.0
        variance = sum((value - mean) ** 2 for value in values) / 4.0
        if not close(mean, predicted) or not close(
            math.sqrt(variance), entry.get("predicted_ii_std", math.nan)
        ):
            fail("direct four-member mean or population standard deviation is stale")
        extrapolated = lower > TRAINING_CEILING or any(
            value > TRAINING_CEILING for value in values
        )
        expected_status = (
            "out-of-training-ceiling"
            if extrapolated
            else "within-training-ceiling"
        )
        if entry.get("extrapolation_status") != expected_status:
            fail("supported row extrapolation status does not match its values")
        saw_extrapolated |= extrapolated
        supported_prediction_count += 1

    if not saw_extrapolated:
        fail("positive fixture never exercised an out-of-training-ceiling row")
    if not saw_unsupported:
        fail("positive fixture never exercised an unsupported row above runtime 23")
    if (
        supported_prediction_count != EXPECTED_RUNTIME_23_SUPPORTED_PREDICTIONS
        or unsupported_row_count != EXPECTED_RUNTIME_23_UNSUPPORTED_ROWS
    ):
        fail(
            "canonical Ray catalog must contain 211 numeric predictions and "
            "five runtime-unsupported rows; got "
            f"{supported_prediction_count} and {unsupported_row_count}"
        )
    if len(task_shapes) != EXPECTED_RAY_TASKS or any(
        len(shapes) != EXPECTED_SHAPES_PER_TASK for shapes in task_shapes.values()
    ):
        fail("catalog does not cover 8 unique shapes for each of 27 Ray tasks")
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-command-json", required=True)
    parser.add_argument("--expected-input", required=True)
    parser.add_argument("--retimer-command-json")
    parser.add_argument("--ensemble", required=True)
    args = parser.parse_args()

    predictor_template = load_argv(args.predictor_command_json)
    expected_input = Path(args.expected_input).resolve()
    if not expected_input.is_file() or str(expected_input) not in predictor_template:
        fail("predictor command does not use the exact expected canonical Ray input")
    ensemble = json.loads(Path(args.ensemble).read_text())
    predictor_pass_name = "predict-analytical-task-cost-catalog"
    predictor_pass_prefix = f"--{predictor_pass_name}="
    predictor_pass_args = next(
        (
            arg.split("=", 1)[1]
            for arg in predictor_template
            if arg.startswith(predictor_pass_prefix)
        ),
        None,
    )
    if predictor_pass_args is None:
        fail("command does not contain the direct analytical cost predictor")
    predictor_options = dict(
        token.split("=", 1)
        for token in shlex.split(predictor_pass_args)
        if "=" in token
    )
    if Path(predictor_options.get("ensemble-file", "")).resolve() != Path(
        args.ensemble
    ).resolve():
        fail("--ensemble differs from the bundle named by the predictor command")
    if predictor_options.get("feature-file"):
        fail("predictor command must derive current features from the source module")
    if predictor_options.get("feature-output"):
        fail(
            "predictor command must omit feature-output because this canonical "
            "catalog includes runtime-unsupported shapes"
        )
    training_yaml = ensemble["architecture"]["exact_yaml_text"]
    marker = "ctrl_mem_items: 20"
    if training_yaml.count(marker) != 1:
        fail("training bundle must contain exactly one ctrl_mem_items: 20 line")
    runtime_yaml = training_yaml.replace(marker, "ctrl_mem_items: 23", 1)

    with tempfile.TemporaryDirectory(prefix="orbit-direct-ii23-check-") as tmp:
        temp = Path(tmp)
        runtime_arch = temp / "runtime-ctrl23.yaml"
        runtime_arch.write_text(runtime_yaml)
        cost_path = temp / "diagnostic-cost-catalog.json"
        positive_argv = replace_architecture(predictor_template, runtime_arch)
        positive_argv = update_pass_options(
            positive_argv,
            predictor_pass_name,
            {
                "diagnostic-ii-ceiling": "23",
                "allow-unsupported-above-model-ceiling": "true",
                "architecture-path": runtime_arch,
                "cache": temp / "direct-model-cache.json",
                "output": cost_path,
            },
        )
        positive_argv = replace_mlir_output(
            positive_argv, temp / "predicted.mlir"
        )
        run_expect_success(positive_argv, "II=23 source-body predictor positive case")
        catalog = check_catalog(cost_path, training_yaml, runtime_yaml)

        invalid_ceiling_argv = update_pass_options(
            positive_argv,
            predictor_pass_name,
            {
                "diagnostic-ii-ceiling": "22",
                "output": temp / "invalid-ceiling.json",
            },
        )
        invalid_ceiling_argv = replace_mlir_output(
            invalid_ceiling_argv, temp / "invalid-ceiling.mlir"
        )
        run_expect_failure(invalid_ceiling_argv, "invalid II ceiling 22")

        invalid_architecture = runtime_yaml.replace(
            "context_mem_items: 6", "context_mem_items: 7", 1
        )
        if invalid_architecture == runtime_yaml:
            fail("model training YAML lacks the expected context capacity field")
        invalid_architecture_path = temp / "runtime-context7.yaml"
        invalid_architecture_path.write_text(invalid_architecture)
        invalid_architecture_argv = replace_architecture(
            positive_argv, invalid_architecture_path
        )
        invalid_architecture_argv = update_pass_options(
            invalid_architecture_argv,
            predictor_pass_name,
            {
                "architecture-path": invalid_architecture_path,
                "output": temp / "invalid-architecture.json",
            },
        )
        invalid_architecture_argv = replace_mlir_output(
            invalid_architecture_argv, temp / "invalid-architecture.mlir"
        )
        run_expect_failure(invalid_architecture_argv, "changed context capacity")

        missing_unsupported_flag_argv = update_pass_options(
            positive_argv,
            predictor_pass_name,
            {
                "allow-unsupported-above-model-ceiling": "false",
                "output": temp / "missing-unsupported-flag.json",
            },
        )
        missing_unsupported_flag_argv = replace_mlir_output(
            missing_unsupported_flag_argv,
            temp / "missing-unsupported-flag.mlir",
        )
        run_expect_failure(
            missing_unsupported_flag_argv,
            "II=23 without explicit unsupported-domain records",
        )

        if args.retimer_command_json:
            retimer_template = load_argv(args.retimer_command_json)
            mutated = json.loads(json.dumps(catalog))
            metadata = mutated["predictor_metadata"]
            metadata.pop("diagnostic_override", None)
            metadata.pop("diagnostic_only", None)
            metadata.pop("formal", None)
            missing_proof = temp / "missing-diagnostic-proof.json"
            missing_proof.write_text(json.dumps(mutated))
            retimer_argv = replace_architecture(retimer_template, runtime_arch)
            retimer_argv = replace_retimer_cost_and_output(
                retimer_argv, missing_proof, temp / "missing-proof-result.json"
            )
            run_expect_failure(
                retimer_argv, "catalog with missing diagnostic proof"
            )

    print(
        "II=23 diagnostic checks passed: exact ctrl_mem_items override, "
        f"canonical input {expected_input}, {len(catalog['entries'])} rows "
        "(211 numeric predictions, 5 unsupported), member means, "
        "unsupported boundary, three fail-closed CLI cases, and optional "
        "missing-proof rejection"
    )


if __name__ == "__main__":
    main()
