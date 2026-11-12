#!/usr/bin/env python3
"""Fit local typed-region phase costs, then validate a frozen model separately.

No device calls occur here. The input index explicitly declares train/holdout
and fixed feature lists. Fit never opens holdout evidence. Coefficients predict
exclusive runtime phases; collinear terms are not physical parameter estimates.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

RUNTIME = "orbit.serial-prepared-typed-native-wait2.v32"
APP_PHASES = ("reset", "control", "copy")
NPU_PHASES = ("sync_to", "submit", "wait", "sync_from")
ACCEPTANCE = dict(minimum_fresh_processes=3, median_absolute_relative_error_max=0.20,
                  ranking_uncertainty="minmax of fresh-process hot-call medians")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "expected finite nonnegative quantity")
    return float(value)


def sealed(reference, base, used):
    require(type(reference) is dict and set(reference) == {"path", "sha256"},
            "evidence references require exact path/sha256")
    path = Path(reference["path"])
    path = path.resolve() if path.is_absolute() else (base / path).resolve()
    require(sha(path) == reference["sha256"], "evidence hash changed: " + str(path))
    used[str(path)] = dict(sha256=reference["sha256"], bytes=path.stat().st_size)
    return path


def linked(path, digest, used):
    path = Path(path)
    require(path.is_absolute(), "linked artifact path must be absolute")
    path = path.resolve()
    require(sha(path) == digest, "linked artifact changed: " + str(path))
    used[str(path)] = dict(sha256=digest, bytes=path.stat().st_size)
    return path


def key(region):
    return (region["device"], region["source_family"],
            tuple(sorted(region["implementations"])))


def load_index(path):
    path = Path(path).resolve()
    index = read(path)
    require(index.get("schema") == "orbit.heterogeneous.calibration-index.v32", "wrong index schema")
    require(index.get("runtime_identity") == RUNTIME, "foreign runtime")
    require(type(index.get("execution_contract_identity")) is str and
            index["execution_contract_identity"] not in ("", "unbound"), "unbound contract")
    entries = index["cases"]
    require(entries and len({entry["id"] for entry in entries}) == len(entries), "duplicate/empty case IDs")
    require({entry["split"] for entry in entries} == {"train", "holdout"},
            "explicit nonempty train and holdout splits required")
    def evidence_paths(split):
        result = set()
        for entry in entries:
            if entry["split"] == split:
                for role in ("features", "identity", "samples", "result", "process"):
                    if role not in entry:
                        continue
                    value = Path(entry[role]["path"])
                    result.add(value.resolve() if value.is_absolute() else (path.parent / value).resolve())
        return result
    require(evidence_paths("train").isdisjoint(evidence_paths("holdout")),
            "train/holdout evidence must be disjoint, including path aliases")
    require(all(type(entry.get("ranking_group")) is str and entry["ranking_group"] for entry in entries),
            "explicit comparable workload ranking groups required")
    policy = index["feature_sets"]
    require(set(policy) == {"application", "cpu", "npu"}, "fixed phase feature sets required")
    for group, phases in (("application", APP_PHASES), ("cpu", ("body",)), ("npu", NPU_PHASES)):
        require(set(policy[group]) == set(phases), "missing or foreign phase feature list")
        for names in policy[group].values():
            require(names and len(names) == len(set(names)) and all(type(n) is str for n in names),
                    "features must be explicit nonempty unique names")
    return path, index


def load_case(entry, index_path, index, used):
    base = index_path.parent
    features = read(sealed(entry["features"], base, used))
    identity = read(sealed(entry["identity"], base, used))
    samples = read(sealed(entry["samples"], base, used))
    result = read(sealed(entry["result"], base, used))
    require(features.get("schema") == "orbit.heterogeneous.edge-features.v32", "foreign feature schema")
    require(identity.get("runtime") == RUNTIME and identity.get("native_wait2") is True, "foreign completion contract")
    compiled_path = linked(identity["compiled"], identity["compiled_sha256"], used)
    compiled = read(compiled_path)
    require(compiled.get("schema") == "orbit.heterogeneous.compiled.v32" and
            compiled.get("runtime_identity") == RUNTIME, "foreign compiled runtime")
    manifest_path = linked(compiled["manifest"], compiled["manifest_sha256"], used)
    manifest = read(manifest_path)
    contract = manifest.get("execution_contract", {})
    require(manifest.get("schema") == "orbit.heterogeneous.static.v32" and
            manifest.get("execution_contract_identity") == index["execution_contract_identity"] and
            contract.get("runtime_identity") == RUNTIME, "foreign manifest runtime/contract")
    require(identity["executor_sha256"] == compiled["executor_sha256"] == contract["executor_sha256"],
            "executor source identities differ")
    require(identity["plan_id"] == compiled["plan_id"] == manifest["id"], "foreign plan receipt")
    require(features["parent_graph_id"] == manifest["graph_id"], "foreign parent graph")
    regions = features["regions"]
    require(len(regions) == len(manifest["regions"]) == len(compiled["regions"]), "incomplete regional receipt")
    for ordinal, (feature, runtime, artifact) in enumerate(zip(regions, manifest["regions"], compiled["regions"])):
        require(feature["device"] in ("cpu", "npu") and type(feature["source_family"]) is str and
                feature["source_family"] and type(feature["implementations"]) is list and
                feature["implementations"] and
                all(type(name) is str and name for name in feature["implementations"]) and
                len(set(feature["implementations"])) == len(feature["implementations"]),
                "local source family needs an exact device/implementation set")
        require(feature["region"] == runtime["index"] == artifact["index"] == ordinal and
                feature["device"] == runtime["device"] == artifact["device"] and
                feature["child_graph_id"] == runtime["child_graph_id"] == artifact["child_graph_id"],
                "foreign regional source/graph")
        source = Path(runtime["source"])
        source = source if source.is_absolute() else manifest_path.parent / source
        linked(source, artifact["source_sha256"], used)
        if artifact["device"] == "cpu":
            linked(artifact["library"], artifact["library_sha256"], used)
        else:
            configurations = artifact.get("configurations", [artifact])
            require(type(configurations) is list and configurations and
                    runtime.get("physical_invocation_count", 1) == len(configurations) and
                    feature["features"].get("configuration_count", 1) == len(configurations),
                    "compiled configuration count differs from regional features")
            air = Path(runtime["air"])
            air = air if air.is_absolute() else manifest_path.parent / air
            physical_path = linked(str(air) + ".physical.json", artifact["physical_receipt_sha256"], used)
            physical = read(physical_path)
            expected = physical.get("configurations", [dict(air=str(air))])
            require(physical.get("graph_id") == feature["child_graph_id"] and
                    physical.get("physical_invocation_count", 1) == len(configurations) and
                    len(expected) == len(configurations), "foreign physical configuration sequence")
            for expected_item, configuration in zip(expected, configurations):
                selected_air = Path(expected_item["air"])
                selected_air = selected_air if selected_air.is_absolute() else air.parent / selected_air
                linked(selected_air, configuration["air_sha256"], used)
                selected_physical = read(linked(str(selected_air) + ".physical.json",
                                                configuration["physical_receipt_sha256"], used))
                proof = read(linked(configuration["runtime_proof"], configuration["runtime_proof_sha256"], used))
                require(selected_physical.get("graph_id") == feature["child_graph_id"] and
                        proof.get("program_graph_id") == feature["child_graph_id"] and
                        proof.get("runtime_contract_sha256") == configuration["runtime_contract_sha256"] ==
                            selected_physical.get("runtime_contract_sha256") and
                        proof.get("generic_module_sha256") == configuration["realized_runtime_sha256"],
                        "foreign configuration runtime proof")
                linked(configuration["elf"], configuration["elf_sha256"], used)
    configuration_count = sum(len(a.get("configurations", [a])) for a in compiled["regions"] if a["device"] == "npu")
    require(features["application"].get("configuration_count", configuration_count) == configuration_count,
            "application configuration count differs from compiled NPU calls")
    require(result.get("status") == "passed" and result["calls"] == len(samples) and
            len(result["checks"]) == len(samples) and result["checks"] and
            all(row and all(value is True for value in row.values()) for row in result["checks"]),
            "producer numeric checks incomplete or failed")
    hot = []
    for sample in samples:
        require(type(sample.get("warmup")) is bool, "warmup status must be explicit")
        rows = sample["regions"]
        require(len(rows) == len(regions), "missing runtime region phase")
        owned = sum(number(sample[name + "_ns"]) for name in APP_PHASES)
        for ordinal, (row, region) in enumerate(zip(rows, regions)):
            phases = ("body",) if region["device"] == "cpu" else NPU_PHASES
            expected_keys = {"region", *(phase + "_ns" for phase in phases)}
            if region["device"] == "npu":
                expected_keys.add("configuration_count")
                expected_count = len(compiled["regions"][ordinal].get("configurations", [compiled["regions"][ordinal]]))
                require(type(row.get("configuration_count")) is int and row["configuration_count"] == expected_count,
                        "runtime invocation count differs from compiled configuration sequence")
            require(row["region"] == ordinal and set(row) == expected_keys,
                    "missing, overlapping or foreign runtime phase")
            owned += sum(number(row[phase + "_ns"]) for phase in phases)
        require(owned == number(sample["wall_ns"]), "exclusive phase sum differs from wall")
        if not sample["warmup"]:
            hot.append(sample)
    require(hot, "no hot calls")
    process_token, process_status = None, "missing process receipt"
    if "process" in entry:
        process = read(sealed(entry["process"], base, used))
        require(process.get("schema") == "orbit.heterogeneous.process.v32" and
                type(process.get("pid")) is int and process["pid"] > 0 and
                type(process.get("started_ns")) is int and type(process.get("ended_ns")) is int and
                process["started_ns"] < process["ended_ns"] and process.get("returncode") == 0 and
                process.get("identity_sha256") == entry["identity"]["sha256"], "foreign process receipt")
        process_status = "diagnostic: PID scope=" + str(process.get("pid_scope", "unspecified"))
        if process.get("pid_scope") == "measurement-python":
            process_token = (process["pid"], process["started_ns"])
            process_status = "verified measurement-python invocation"
    witness = dict(runtime_contract=index["execution_contract_identity"], parent_graph_id=features["parent_graph_id"],
                   region_graph_ids=[r["child_graph_id"] for r in regions])
    return dict(id=entry["id"], ranking_group=entry["ranking_group"], features=features,
                witness=witness, process_token=process_token, process_status=process_status,
                application={phase: statistics.median(s[phase + "_ns"] for s in hot) for phase in APP_PHASES},
                regions=[{phase: statistics.median(s["regions"][i][phase + "_ns"] for s in hot)
                          for phase in (("body",) if r["device"] == "cpu" else NPU_PHASES)}
                         for i, r in enumerate(regions)],
                wall_ns=statistics.median(s["wall_ns"] for s in hot), hot_calls=len(hot))


def fit_phase(rows, names, source, diagnostics):
    import numpy as np
    from scipy.optimize import nnls
    x = np.asarray([[number(features[name]) for name in names] for features, _ in rows], dtype=float)
    y = np.asarray([number(duration) for _, duration in rows], dtype=float)
    # Scaling avoids treating bytes/operations columns as interchangeable with
    # an intercept numerically. NNLS is unchanged by positive column scaling.
    scale = np.maximum(np.max(x, axis=0), 1.0)
    coefficients, residual = nnls(x / scale, y, maxiter=max(100, len(names) * 100))
    coefficients /= scale
    diagnostics.append(dict(features=names, rows=len(rows), rank=int(np.linalg.matrix_rank(x / scale)),
                            columns=len(names), residual_l2_ns=float(residual),
                            interpretation="predictive only; collinear coefficients are not separately physical"))
    return dict(coefficients_ns={name: dict(value=float(value), kind="calibrated", source=source)
                                 for name, value in zip(names, coefficients)},
                ranges={name: [float(np.min(x[:, i])), float(np.max(x[:, i]))]
                        for i, name in enumerate(names)})


def fit(index_path, output):
    index_path, index = load_index(index_path)
    used = {str(index_path): dict(sha256=sha(index_path), bytes=index_path.stat().st_size)}
    # This is the sole data-opening loop. No holdout reference is resolved here.
    cases = [load_case(entry, index_path, index, used) for entry in index["cases"] if entry["split"] == "train"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    diagnostics = []
    provenance = "training-only local phase NNLS; index sha256=" + sha(index_path)
    model = dict(schema="orbit.heterogeneous.edge-cost.v32", runtime_identity=RUNTIME,
                 execution_contract_identity=index["execution_contract_identity"], application={}, regions=[])
    for phase in APP_PHASES:
        rows = [(case["features"]["application"], case["application"][phase]) for case in cases]
        model["application"][phase] = fit_phase(rows, index["feature_sets"]["application"][phase], provenance, diagnostics)
    buckets = {}
    for case in cases:
        for region, durations in zip(case["features"]["regions"], case["regions"]):
            buckets.setdefault(key(region), []).append((region["features"], durations))
    for (device, family, implementations), observations in sorted(buckets.items()):
        phases = ("body",) if device == "cpu" else NPU_PHASES
        row = dict(device=device, source_family=family, implementations=list(implementations), phases={})
        for phase in phases:
            names = index["feature_sets"][device][phase]
            require(device != "npu" or phase not in ("submit", "wait") or
                    not any(f.get("configuration_count", 1) > 1 for f, _ in observations) or
                    "configuration_count" in names,
                    "multiple configurations require preregistered submit/wait count terms")
            row["phases"][phase] = fit_phase([(f, d[phase]) for f, d in observations],
                                            names, provenance, diagnostics)
        model["regions"].append(row)
    save(output / "model.json", model)
    save(output / "training-inputs.json", used)
    save(output / "training-fit.json", dict(cases=[c["id"] for c in cases], diagnostics=diagnostics,
         numeric_scope="producer checks verified; no independent FP64 recomputation",
         heldout_timings_opened=False))
    import numpy
    import scipy
    freeze = dict(acceptance=ACCEPTANCE,
                  versions=dict(python=sys.version, numpy=numpy.__version__, scipy=scipy.__version__),
                  schema="orbit.heterogeneous.model-freeze.v32", model_sha256=sha(output / "model.json"),
                  index_sha256=sha(index_path), producer_sha256=sha(__file__),
                  training_inputs_sha256=sha(output / "training-inputs.json"))
    save(output / "model-manifest.json", freeze)
    return freeze


def predict_phase(phase, features):
    if not phase or not phase.get("coefficients_ns"):
        return None
    total = 0.0
    for name, coefficient in phase["coefficients_ns"].items():
        if name not in features or name not in phase.get("ranges", {}):
            return None
        value = number(features[name])
        low, high = phase["ranges"][name]
        if not low <= value <= high:
            return None
        total += value * number(coefficient["value"])
    return total if math.isfinite(total) else None


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def aggregate_validation(rows):
    """Group complete typed witnesses; unknowns and weak process proof persist."""
    buckets = {}
    for row in rows:
        witness_json = json.dumps(row["witness"], sort_keys=True, separators=(",", ":"))
        buckets.setdefault((row["ranking_group"], witness_json), []).append(row)
    groups = []
    for (ranking_group, witness_json), values in buckets.items():
        processes = {}
        for row in values:
            if row["process_token"] is not None:
                processes.setdefault(tuple(row["process_token"]), []).append(row["measured_ns"])
        process_medians = [statistics.median(v) for v in processes.values()]
        verified = len(processes) >= ACCEPTANCE["minimum_fresh_processes"] and all(
            row["process_token"] is not None for row in values)
        measured = statistics.median(process_medians if verified else [r["measured_ns"] for r in values])
        predictions = [r["prediction_ns"] for r in values]
        prediction = statistics.median(predictions) if all(v is not None for v in predictions) else None
        consistent = prediction is None or all(abs(v - prediction) <= 1e-12 * max(1, prediction) for v in predictions)
        if not consistent:
            prediction = None
        error = abs(prediction - measured) / measured if prediction is not None and measured > 0 else None
        intervals = process_medians if verified else [r["measured_ns"] for r in values]
        groups.append(dict(ranking_group=ranking_group, witness=json.loads(witness_json),
                           witness_id=hashlib.sha256(witness_json.encode()).hexdigest(),
                           measurement_rows=len(values), verified_fresh_processes=len(processes),
                           fresh_process_gate=verified, prediction_consistent=consistent,
                           prediction_ns=prediction, measured_ns=measured,
                           process_median_interval_ns=[min(intervals), max(intervals)],
                           absolute_relative_error=error, status="known" if error is not None else "unknown"))
    errors = [g["absolute_relative_error"] for g in groups if g["absolute_relative_error"] is not None]
    coverage = len(errors) == len(groups) and bool(groups)
    fresh = bool(groups) and all(g["fresh_process_gate"] for g in groups)
    median_error = statistics.median(errors) if errors else None
    accuracy = ("diagnostic_unknown" if not coverage or not fresh else
                "passed" if median_error <= ACCEPTANCE["median_absolute_relative_error_max"] else "failed")
    rankings = []
    for ranking_group in sorted({g["ranking_group"] for g in groups}):
        choices = [g for g in groups if g["ranking_group"] == ranking_group]
        status, reason = "unknown", "need at least two fully covered choices with three verified fresh processes"
        selected = winner = None
        if len(choices) >= 2 and all(g["fresh_process_gate"] and g["status"] == "known" for g in choices):
            best = min(choices, key=lambda g: g["measured_ns"])
            others = [g for g in choices if g is not best]
            predicted = min(choices, key=lambda g: g["prediction_ns"])
            tied = any(abs(g["prediction_ns"] - predicted["prediction_ns"]) <=
                       1e-12 * max(1, predicted["prediction_ns"]) for g in choices if g is not predicted)
            if best["process_median_interval_ns"][1] >= min(g["process_median_interval_ns"][0] for g in others):
                reason = "measured top-one gap does not exceed process-median minmax uncertainty"
            elif tied:
                reason = "model top-one is tied"
            else:
                selected, winner = predicted["witness_id"], best["witness_id"]
                status = "passed" if selected == winner else "failed"
                reason = "measured winner interval strictly separated from every alternative"
        rankings.append(dict(ranking_group=ranking_group, status=status, reason=reason,
                             predicted_witness=selected, measured_witness=winner))
    return dict(groups=groups,
                group_accuracy=dict(total=len(groups), known=len(errors), coverage_complete=coverage,
                                    verified_fresh_process_gate=fresh,
                                    median_absolute_relative_error=median_error,
                                    p90_absolute_relative_error=percentile(errors, .9), status=accuracy),
                top_one=rankings, preregistered_acceptance=ACCEPTANCE,
                cp_speedup_claim=False)


def validate(index_path, model_dir, output):
    index_path, index = load_index(index_path)
    model_dir = Path(model_dir)
    freeze = read(model_dir / "model-manifest.json")
    require(freeze.get("acceptance") == ACCEPTANCE, "preregistered acceptance changed")
    require(sha(model_dir / "model.json") == freeze["model_sha256"] and
            sha(index_path) == freeze["index_sha256"] and sha(__file__) == freeze["producer_sha256"] and
            sha(model_dir / "training-inputs.json") == freeze["training_inputs_sha256"], "frozen model/input changed")
    for path, receipt in read(model_dir / "training-inputs.json").items():
        require(Path(path).stat().st_size == receipt["bytes"] and sha(path) == receipt["sha256"],
                "frozen training evidence changed: " + path)
    model = read(model_dir / "model.json")
    require(model["runtime_identity"] == RUNTIME and model["execution_contract_identity"] == index["execution_contract_identity"],
            "foreign frozen runtime/model contract")
    used = {}
    cases = [load_case(entry, index_path, index, used) for entry in index["cases"] if entry["split"] == "holdout"]
    rows = []
    for case in cases:
        components = [predict_phase(model["application"].get(phase), case["features"]["application"]) for phase in APP_PHASES]
        for region in case["features"]["regions"]:
            matches = [row for row in model["regions"] if key(row) == key(region)]
            require(len(matches) <= 1, "duplicate local family model")
            phases = ("body",) if region["device"] == "cpu" else NPU_PHASES
            for phase in phases:
                phase_model = matches[0]["phases"].get(phase) if matches else None
                if (region["device"] == "npu" and region["features"].get("configuration_count", 1) > 1 and
                        phase in ("submit", "wait") and
                        (not phase_model or "configuration_count" not in phase_model.get("coefficients_ns", {}))):
                    components.append(None)
                else:
                    components.append(predict_phase(phase_model, region["features"]))
        prediction = sum(components) if all(c is not None for c in components) else None
        measured = case["wall_ns"]
        error = abs(prediction - measured) / measured if prediction is not None and measured > 0 else None
        rows.append(dict(case=case["id"], ranking_group=case["ranking_group"], prediction_ns=prediction, measured_ns=measured,
                         witness=case["witness"], process_token=case["process_token"], process_status=case["process_status"],
                         absolute_relative_error=error, status="known" if prediction is not None else "unknown"))
    errors = [row["absolute_relative_error"] for row in rows if row["absolute_relative_error"] is not None]
    threshold = number(index["ranking_min_relative_gap"])
    correct = eligible = 0
    for left, right in itertools.combinations(rows, 2):
        if left["ranking_group"] != right["ranking_group"] or left["measured_ns"] == right["measured_ns"]:
            continue
        if abs(left["measured_ns"] - right["measured_ns"]) < threshold * min(left["measured_ns"], right["measured_ns"]):
            continue
        eligible += 1
        if left["prediction_ns"] is not None and right["prediction_ns"] is not None:
            correct += (left["prediction_ns"] - right["prediction_ns"]) * (left["measured_ns"] - right["measured_ns"]) > 0
    grouped = aggregate_validation(rows)
    report = dict(schema="orbit.heterogeneous.local-validation.v32", rows=rows, **grouped,
                  known=len(errors), total=len(rows), coverage_complete=len(errors) == len(rows),
                  median_absolute_relative_error=statistics.median(errors) if errors else None,
                  p90_absolute_relative_error=percentile(errors, .9),
                  ranking=dict(eligible=eligible, correct=correct, minimum_relative_gap=threshold),
                  scope="Python local-phase prediction diagnostic; hardware speedup and native C++ selection require separate evidence",
                  model_sha256=freeze["model_sha256"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "validation.json", report)
    save(output / "validation-inputs.json", used)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "validate"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    require(args.phase == "fit" or args.model_dir is not None, "validate requires --model-dir")
    result = fit(args.index, args.output) if args.phase == "fit" else validate(args.index, args.model_dir, args.output)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
