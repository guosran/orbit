#!/usr/bin/env python3
"""Process boundary from ORBIT's search to typed, evidence-backed evaluation.

Usage: taskflow_search_evaluate.py REQUEST.json RESPONSE.json
No NPU lock is taken here: the NPU runtime is the sole lock owner.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from taskflow_common import fingerprint


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def implementation_identity(backend):
    files = [Path(__file__), ROOT / "scripts/taskflow_common.py",
             Path(os.environ.get("ORBIT_TASKFLOW_OPT", ROOT / "build-core/tools/taskflow-opt/taskflow-opt"))]
    packages = ["numpy"]
    if backend == "npu":
        files += sorted((ROOT / "python/orbit_npu").glob("*.py"))
        toy = Path(os.environ.get("AIE_TOY_ROOT", "/home/shiran/project/aie-toy"))
        files += [toy / "build/qwen25-3b/Triton-XDNA/examples/llm_q4nx/transform_matmul_aie2p.mlir"]
        packages += ["triton-xdna", "mlir-air", "mlir-aie-no-rtti", "llvm-aie", "torch"]
        # Version labels alone do not identify locally patched launch/runtime
        # code. Hash the installed backend Python sources without importing or
        # initializing the device driver.
        distribution = importlib.metadata.distribution("triton-xdna")
        files += sorted(Path(distribution.locate_file(p)) for p in (distribution.files or [])
                        if str(p).startswith("triton/backends/amd_triton_npu/") and str(p).endswith(".py"))
    elif backend == "amoeba":
        files += [ROOT / "scripts/taskflow_amoeba.py", ROOT / "build-amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt"]
    else:
        raise ValueError("unsupported backend: " + str(backend))
    versions = {}
    for package in packages:
        versions[package] = importlib.metadata.version(package)
    return dict(files={str(p): digest(p) for p in files}, packages=versions,
                python=sys.version, backend=backend)


def evidence_score(backend, evidence, candidate_id, source_id):
    """Derive rankable cost from backend evidence, not a cached scalar claim."""
    if evidence.get("status") != "passed" or evidence.get("candidate_id") != candidate_id:
        return None
    if evidence.get("source_id", source_id) != source_id:
        return None
    if backend == "npu":
        samples = evidence.get("host_wall_ms", [])
        checks = evidence.get("numerical_checks", [])
        if (len(samples) < 3 or len(checks) != len(samples) or
                not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in samples) or
                not all(c.get("passed") for c in checks) or evidence.get("device_dispatches_per_graph", 0) < 1):
            return None
        value = statistics.median(samples)
        if value != evidence.get("median_host_wall_ms"):
            return None
        return dict(value=value, unit="ms", provenance="measured-complete-host-wall-npu-xrt", samples=samples)
    score = evidence.get("score", {})
    value = score.get("value")
    if (score.get("status") != "scored" or score.get("unit") != "cycles" or
            not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0):
        return None
    return dict(value=value, unit="cycles", provenance=score["provenance"], physical_hardware_measurement=False)


def evaluate(request):
    if request.get("schema") != "orbit.search.evaluation.v1":
        raise ValueError("unsupported evaluation protocol")
    backend = request["backend"]
    target = request["target_config"]
    config = target["evaluation"]
    plan_path, mlir_path = Path(request["plan_path"]), Path(request["mlir_path"])
    directory = Path(request["output_dir"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_path.read_text())
    source = json.loads(Path(config["source_plan"]).read_text())
    if plan["candidate_id"] != request["candidate_id"] or plan["source_id"] != request["source_id"] or plan["source_id"] != source["source_id"]:
        raise ValueError("source or candidate identity mismatch")
    implementation = implementation_identity(backend)
    implementation_hash = fingerprint(implementation)
    if config.get("implementation_fingerprint") != implementation_hash:
        raise ValueError("implementation fingerprint changed; regenerate target configuration before search/resume")
    iterations = config.get("iterations", 3)
    if not isinstance(iterations, int) or iterations < 3:
        raise ValueError("at least three full graph samples required")
    identity = dict(schema="orbit.search.cache.v1", backend=backend,
        plan=plan, mlir_sha256=digest(mlir_path), source_sha256=digest(config["source_plan"]),
        inputs_sha256=digest(config["inputs"]), reference_sha256=digest(config["reference"]),
        implementation=implementation, target=target.get(backend, {}), iterations=iterations,
        measurement_batch=config.get("measurement_batch", str(directory.parent)))
    def referenced_files(value):
        if isinstance(value, dict):
            return {k: referenced_files(v) for k, v in value.items()}
        if isinstance(value, list):
            return [referenced_files(v) for v in value]
        if isinstance(value, str) and Path(value).is_file():
            return {"path": value, "sha256": digest(value)}
        return value
    # In particular, architecture YAML contents must invalidate Amoeba cache
    # even if a caller edits the file in place while keeping its pathname.
    identity["target_file_contents"] = referenced_files(target.get(backend, {}))
    key = fingerprint(identity)
    cache = Path(config["cache_dir"]) / (key + ".json")
    atomic_json(directory / "evaluation-identity.json", identity)
    response = dict(schema="orbit.search.feedback.v1", backend=backend,
        candidate_id=plan["candidate_id"], source_id=plan["source_id"], cache_identity=key,
        implementation_fingerprint=implementation_hash, cache_hit=False,
        status="unsupported", score=None)
    if config.get("use_cache", True) and cache.exists():
        try:
            saved = json.loads(cache.read_text())
            prior = saved["response"]
            evidence = Path(prior["evidence_path"])
            measured = json.loads(evidence.read_text())
            cost = evidence_score(backend, measured, plan["candidate_id"], plan["source_id"])
            if (saved["identity"] == identity and saved.get("evidence_sha256") == digest(evidence) and
                    prior.get("status") == "passed" and prior.get("backend") == backend and
                    prior.get("candidate_id") == plan["candidate_id"] and prior.get("source_id") == plan["source_id"] and
                    prior.get("cache_identity") == key and cost and prior.get("score") == cost):
                return dict(prior, cache_hit=True, cache_entry=str(cache))
            response["cache_rejected"] = "cached identity or score differs from authenticated execution evidence"
        except (ValueError, KeyError, OSError, TypeError):
            response["cache_rejected"] = "malformed or missing cached evidence"
        atomic_json(directory / "cache-rejection.json", dict(cache=str(cache), reason=response["cache_rejected"]))
    started = time.perf_counter()
    if backend == "npu":
        import numpy as np
        from orbit_npu.runtime import run_plan
        with np.load(config["inputs"]) as arrays:
            inputs = dict(arrays)
        with np.load(config["reference"]) as arrays:
            reference = dict(arrays)
        result = run_plan(plan, inputs, reference, directory / "npu", iterations=iterations,
                          target_config=target.get("npu"))
        response.update(status=result["status"], evidence_path=str(directory / "npu/result.json"))
        if result["status"] == "passed":
            samples = result["host_wall_ms"]
            if len(samples) < iterations or not all(math.isfinite(v) and v > 0 for v in samples):
                response.update(status="runtime_failed", reason="invalid full graph timing samples")
            else:
                response["score"] = dict(value=result["median_host_wall_ms"], unit="ms",
                    provenance="measured-complete-host-wall-npu-xrt", samples=samples)
        else:
            response["reason"] = result.get("error", "backend numerical or execution validation failed")
    else:
        argv = [sys.executable, str(ROOT / "scripts/taskflow_amoeba.py"), "--plan", str(plan_path),
            "--mlir", str(mlir_path), "--source-plan", config["source_plan"],
            "--output", str(directory / "amoeba"), "--map"]
        if "architecture" in target.get("amoeba", {}):
            argv += ["--architecture", target["amoeba"]["architecture"]]
        result = subprocess.run(argv, capture_output=True, text=True)
        (directory / "amoeba-evaluation.log").write_text(result.stdout + result.stderr)
        evidence = directory / "amoeba/feedback.json"
        response["evidence_path"] = str(evidence)
        if not evidence.exists():
            response.update(status="compile_failed", reason="Amoeba adapter produced no feedback")
        else:
            feedback = json.loads(evidence.read_text())
            status = feedback["status"]
            score = feedback.get("score", {})
            if status == "passed" and score.get("status") == "scored":
                response.update(status="passed", score=dict(value=score["value"], unit=score["unit"],
                    provenance=score["provenance"], physical_hardware_measurement=False))
            elif status == "passed":
                mapping = feedback.get("mapping", {})
                response.update(status="unsupported", reason="Replay passed but target mapping unscored: " + str(mapping.get("reason", mapping.get("status"))))
            else:
                response.update(status={"compile_failure":"compile_failed", "host_lowering_failure":"compile_failed",
                    "numerical_failure":"numerical_failed", "unsupported":"unsupported"}.get(status, "runtime_failed"),
                    reason=feedback.get("reason", status))
    response["evaluation_wall_seconds"] = time.perf_counter() - started
    if response["status"] == "passed":
        measured = json.loads(Path(response["evidence_path"]).read_text())
        accepted = evidence_score(backend, measured, plan["candidate_id"], plan["source_id"])
        if accepted != response.get("score"):
            response.update(status="runtime_failed", score=None, reason="score does not match complete backend evidence")
    if response["status"] == "passed":
        score = response.get("score")
        unit = "ms" if backend == "npu" else "cycles"
        if not score or score.get("unit") != unit or not math.isfinite(score["value"]) or score["value"] <= 0:
            response.update(status="runtime_failed", score=None, reason="invalid score or unit")
    else:
        response["score"] = None
    if response["status"] == "passed":
        atomic_json(cache, dict(identity=identity, response=response, evidence_sha256=digest(response["evidence_path"])))
    return response


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    request_path, response_path = map(Path, sys.argv[1:])
    request = {}
    try:
        request = json.loads(request_path.read_text())
        response = evaluate(request)
    except (ValueError, KeyError, FileNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
        response = dict(schema="orbit.search.feedback.v1", status="unsupported", score=None, reason=str(exc))
    except Exception as exc:
        import traceback
        response = dict(schema="orbit.search.feedback.v1", status="runtime_failed", score=None,
                        reason=str(exc), traceback=traceback.format_exc())
    if isinstance(request, dict):
        for field in ("candidate_id", "source_id", "backend"):
            response.setdefault(field, request.get(field))
    atomic_json(response_path, response)
    print(json.dumps(response, allow_nan=False), flush=True)
    # Classified failures are valid protocol responses; the search decides what
    # is feasible/ranked, rather than losing the distinction to exit status.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
