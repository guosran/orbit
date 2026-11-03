#!/usr/bin/env python3
"""Reproducible shared-candidate experiments, with separate CPU/device evidence."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from taskflow_common import compare, execute_cpu, fingerprint, make_inputs, reference, source_plan, write_json

ROOT = Path(__file__).resolve().parents[1]


def command(argv, log):
    start = time.perf_counter()
    result = subprocess.run([str(x) for x in argv], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout)
    return dict(command=[str(x) for x in argv], returncode=result.returncode,
                wall_ms=(time.perf_counter() - start) * 1000, log=str(log))


def cases(full):
    result = [(name, source_plan(name)) for name in ("gemm", "gemv", "gemm_relu", "ffn")]
    if full:
        result += [(f"ffn_qwen_m{m}", source_plan("ffn", m=m, d=2048, i=11008)) for m in (5, 128)]
    return result


def prepare(args):
    manifest = dict(schema="orbit.taskflow.experiment.v1", cases=[], environment={
        "python": sys.version, "numpy": np.__version__,
        "orbit_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "tool_sha256": hashlib.sha256(args.tool.read_bytes()).hexdigest(),
        "timing_context": "Shared desktop machine; frequency not pinned; host wall with synchronous completion."})
    for name, source in cases(args.full):
        path = args.output / name
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "source.json", source)
        workload = "ffn" if name.startswith("ffn") else name
        seeds = [0] if "qwen" in name else [0, 1, 2]
        for seed in seeds:
            inputs = make_inputs(source, seed)
            np.savez(path / f"inputs-{seed}.npz", **inputs)
            np.savez(path / f"reference-{seed}.npz", **reference(workload, inputs))
        transforms = [("base", [])]
        if "qwen" not in name:
            transforms += [("mn", ["--tile-m=3", "--tile-n=11"]), ("k", ["--tile-k=7"])]
            if name == "gemm_relu":
                transforms += [("fused", ["--fuse"]), ("fission", ["--fission"])]
        for candidate, flags in transforms:
            dest = path / candidate
            dest.mkdir(parents=True, exist_ok=True)
            inp = path / "fused" / "candidate.mlir" if candidate == "fission" else path / "source.json"
            run = command([args.tool, inp, *flags, f"--emit-plan={dest / 'plan.json'}",
                           f"--emit-mlir={dest / 'candidate.mlir'}"], dest / "materialize.log")
            record = dict(workload=workload, case=name, candidate=candidate, directory=str(dest),
                          seeds=seeds, materialize=run,
                          status="materialized" if run["returncode"] == 0 else "materialization_failed")
            if run["returncode"] == 0:
                # Cross-version exchange starts from this exact, verified MLIR artifact.
                rt = command([args.tool, dest / "candidate.mlir", f"--emit-plan={dest / 'roundtrip.json'}"], dest / "roundtrip.log")
                record["roundtrip"] = rt
                if rt["returncode"]:
                    record["status"] = "roundtrip_failed"
                plan = json.loads((dest / "plan.json").read_text())
                if rt["returncode"] == 0 and json.loads((dest / "roundtrip.json").read_text()) != plan:
                    record["status"] = "roundtrip_identity_mismatch"
                record.update(candidate_id=plan["candidate_id"], source_id=plan["source_id"],
                              task_count=len(plan["tasks"]), operation_count=sum(len(t["operations"]) for t in plan["tasks"]),
                              mlir_sha256=hashlib.sha256((dest / "candidate.mlir").read_bytes()).hexdigest(),
                              plan_sha256=hashlib.sha256((dest / "plan.json").read_bytes()).hexdigest())
            manifest["cases"].append(record)
            write_json(args.output / "manifest.json", manifest)
            print(f"{name}/{candidate}: {record['status']}", flush=True)
    return all(c["status"] == "materialized" for c in manifest["cases"])


def cpu(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    passed = True
    for record in manifest["cases"]:
        if record["status"] != "materialized" or "qwen" in record["case"]:
            continue
        dest = Path(record["directory"])
        plan = json.loads((dest / "plan.json").read_text())
        results = []
        for seed in record["seeds"]:
            with np.load(dest.parent / f"inputs-{seed}.npz") as data:
                inputs = dict(data)
            with np.load(dest.parent / f"reference-{seed}.npz") as data:
                ref = dict(data)
            output = execute_cpu(plan, inputs)
            check = compare(output, ref, 2e-6, 2e-5)
            check["seed"] = seed
            results.append(check)
            passed &= check["passed"]
            np.savez(dest / f"cpu-output-{seed}.npz", **output)
        write_json(dest / "cpu.json", dict(backend="cpu_diagnostic", results=results,
                   numerical_contract="Sequential float32 candidate versus independent whole-matrix float64 oracle; atol=2e-6, rtol=2e-5."))
        print(f"CPU {record['case']}/{record['candidate']}: {all(r['passed'] for r in results)}", flush=True)
    return passed


def negative(args):
    """Malformed programs must fail before backend compilation."""
    base = source_plan("gemm_relu")
    tests = []

    def add(name, edit):
        value = copy.deepcopy(base)
        edit(value)
        tests.append((name, value))

    add("missing_raw", lambda p: p["tasks"][1].update(depends_on=[]))
    add("missing_init", lambda p: p["tasks"][0]["operations"].pop(0))
    add("out_of_bounds", lambda p: p["tasks"][0]["operations"][1].update(n=[0, 20]))
    add("missing_reduction", lambda p: p["tasks"][0]["operations"][1].update(k=[0, 22]))
    add("double_reduction", lambda p: p["tasks"][0]["operations"].append(copy.deepcopy(p["tasks"][0]["operations"][1])))
    add("unknown_operation", lambda p: p["tasks"][1]["operations"][0].update(kind="attention"))
    add("unsupported_dtype", lambda p: p["buffers"][0].update(dtype="bf16"))
    add("dynamic_shape", lambda p: p["buffers"][0].update(shape=[-1, 23]))
    add("cycle", lambda p: p["tasks"][0].update(depends_on=["relu"]))
    add("write_input", lambda p: p["tasks"][0]["operations"][0].update(output="A"))
    results = []
    for name, value in tests:
        path = args.output / "negative" / name
        write_json(path.with_suffix(".json"), value)
        run = command([args.tool, path.with_suffix(".json"), f"--emit-plan={path.with_suffix('.out.json')}"], path.with_suffix(".log"))
        results.append(dict(name=name, rejected=run["returncode"] != 0, **run))
    # A valid witness cannot authorize arbitrary changes to its executable body.
    candidate = args.output / "gemm" / "base" / "candidate.mlir"
    if candidate.exists():
        text = candidate.read_text()
        mutated = text.replace("arith.mulf", "arith.addf", 1)
        if mutated != text:
            path = args.output / "negative" / "forged-body.mlir"
            path.write_text(mutated)
            run = command([args.tool, path, f"--emit-plan={path.with_suffix('.json')}"], path.with_suffix(".log"))
            results.append(dict(name="forged_body", rejected=run["returncode"] != 0, **run))
    write_json(args.output / "negative.json", results)
    for result in results:
        print(f"negative {result['name']}: {'rejected' if result['rejected'] else 'ACCEPTED'}", flush=True)
    return all(r["rejected"] for r in results)


def npu(args):
    """The adapter itself owns /tmp/npu.lock; do not wrap this in flock."""
    manifest = json.loads((args.output / "manifest.json").read_text())
    results = []
    for record in manifest["cases"]:
        if record["status"] != "materialized":
            continue
        dest = Path(record["directory"])
        for seed in record["seeds"]:
            out = dest / f"npu-seed-{seed}"
            run = command([sys.executable, ROOT / "scripts/taskflow_npu.py", "--plan", dest / "plan.json",
                "--inputs", dest.parent / f"inputs-{seed}.npz", "--reference", dest.parent / f"reference-{seed}.npz",
                "--output-dir", out, "--iterations", str(args.iterations)], dest / f"npu-seed-{seed}.log")
            results.append(dict(case=record["case"], candidate=record["candidate"], seed=seed, **run))
            write_json(args.output / "npu-runs.json", results)
            print(f"NPU {record['case']}/{record['candidate']}/{seed}: exit {run['returncode']}", flush=True)
    return all(r["returncode"] == 0 for r in results)


def amoeba(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    results = []
    for record in manifest["cases"]:
        if record["status"] != "materialized" or "qwen" in record["case"]:
            continue
        dest = Path(record["directory"])
        argv = [sys.executable, ROOT / "scripts/taskflow_amoeba.py", "--plan", dest / "plan.json",
            "--mlir", dest / "candidate.mlir", "--source-plan", dest.parent / "source.json",
            "--output", dest / "amoeba", "--core-tool", args.tool]
        if record["case"] == "gemm" and record["candidate"] in ("base", "mn"):
            argv.append("--map")
        run = command(argv, dest / "amoeba.log")
        results.append(dict(case=record["case"], candidate=record["candidate"], **run))
        write_json(args.output / "amoeba-runs.json", results)
        print(f"Amoeba {record['case']}/{record['candidate']}: exit {run['returncode']}", flush=True)
    return all(r["returncode"] == 0 for r in results)


def report(args):
    """Retain every candidate and failure; rank only complete verified runs."""
    manifest = json.loads((args.output / "manifest.json").read_text())
    feedback, rankings, shared = [], {}, []
    for record in manifest["cases"]:
        dest = Path(record["directory"])
        entry = {key: record.get(key) for key in ("case", "candidate", "source_id", "candidate_id", "task_count", "operation_count", "mlir_sha256", "plan_sha256")}
        entry.update(unit="ms", measurement="full graph synchronous host wall", seeds=[], status=record["status"])
        samples, metrics = [], []
        plan = json.loads((dest / "plan.json").read_text()) if (dest / "plan.json").exists() else None
        expected_hash = fingerprint(plan) if plan else None
        disk_identity_ok = bool(plan) and all((dest / file).exists() and
            hashlib.sha256((dest / file).read_bytes()).hexdigest() == record.get(field)
            for file, field in (("plan.json", "plan_sha256"), ("candidate.mlir", "mlir_sha256")))
        entry["disk_identity_matches_manifest"] = disk_identity_ok
        for seed in record["seeds"]:
            file = dest / f"npu-seed-{seed}" / "result.json"
            if not file.exists():
                entry["seeds"].append(dict(seed=seed, status="missing", file=str(file)))
                continue
            result = json.loads(file.read_text())
            # Read the actual compiler-authenticated artifact, including original
            # measurements predating the added result hash field. Do not rewrite
            # old measurements to claim they ran the latest implementation.
            verified_mlir = file.parent / "verified-taskflow.mlir"
            verified_plan = file.parent / "verified-plan.json"
            verified_hash = hashlib.sha256(verified_mlir.read_bytes()).hexdigest() if verified_mlir.exists() else None
            identity_ok = disk_identity_ok and result.get("candidate_id") == record.get("candidate_id") and result.get("program_sha256") == expected_hash and verified_hash == record.get("mlir_sha256") and verified_plan.exists() and json.loads(verified_plan.read_text()) == plan
            raw = result.get("host_wall_ms", [])
            checks = result.get("numerical_checks", [])
            evidence_ok = (len(raw) >= 3 and len(checks) == len(raw) and
                all(np.isfinite(v) and v > 0 for v in raw) and result.get("device_dispatches_per_graph", 0) > 0 and
                all(c.get("passed") and set(c.get("outputs", {})) == set(plan["outputs"]) and
                    all(m.get("passed") and m.get("finite") for m in c["outputs"].values()) for c in checks))
            status = result["status"]
            if not identity_ok:
                status = "identity_mismatch"
            elif status == "passed" and not evidence_ok:
                status = "incomplete_evidence"
            item = dict(seed=seed, status=status, file=str(file), verified_taskflow_sha256=verified_hash,
                        identity_matches=identity_ok, implementation_sha256=result.get("implementation_sha256"),
                        host_wall_ms=result.get("host_wall_ms", []), compile_seconds=result.get("compile_seconds"),
                        load_seconds=result.get("load_seconds"), first_weight_preparation_seconds=result.get("first_weight_preparation_seconds"),
                        device_dispatches_per_graph=result.get("device_dispatches_per_graph"))
            entry["seeds"].append(item)
            samples.extend(item["host_wall_ms"])
            for check in result.get("numerical_checks", []):
                metrics.extend(check["outputs"].values())
        implementations = {s.get("implementation_sha256") for s in entry["seeds"]}
        entry["consistent_implementation"] = len(implementations) == 1 and None not in implementations
        if entry["seeds"] and entry["consistent_implementation"] and all(s["status"] == "passed" for s in entry["seeds"]):
            entry.update(status="passed", score_status="measured", median_host_wall_ms=statistics.median(samples),
                         raw_host_wall_ms=samples, min_host_wall_ms=min(samples), max_host_wall_ms=max(samples),
                         max_abs_error=max(m["max_abs"] for m in metrics),
                         max_normalized_rms_error=max(m["normalized_rms_error"] for m in metrics))
            rankings.setdefault(record["case"], []).append(entry)
        else:
            entry.update(status="incomplete_or_failed", score_status="unscored", raw_host_wall_ms=samples)
        amoeba_path = dest / "amoeba" / "feedback.json"
        if amoeba_path.exists():
            amoeba = json.loads(amoeba_path.read_text())
            matches = disk_identity_ok and amoeba.get("candidate_id") == record.get("candidate_id") and amoeba.get("source_id") == record.get("source_id") and amoeba.get("plan_sha256") == record.get("plan_sha256") and amoeba.get("materialized_mlir_sha256") == record.get("mlir_sha256")
            entry["amoeba"] = dict(status=amoeba.get("status"), evidence=amoeba.get("evidence_type"),
                identity_matches=matches, neura_lowering=amoeba.get("neura_lowering"),
                mapping=amoeba.get("mapping"), score=amoeba.get("score"), file=str(amoeba_path))
            if matches and amoeba.get("status") == "passed" and entry["status"] == "passed":
                shared.append({key: entry[key] for key in ("case", "candidate", "source_id", "candidate_id", "mlir_sha256", "plan_sha256")})
        feedback.append(entry)
    ranking = {name: [dict(candidate=e["candidate"], candidate_id=e["candidate_id"], median_host_wall_ms=e["median_host_wall_ms"])
                      for e in sorted(entries, key=lambda e: e["median_host_wall_ms"])] for name, entries in rankings.items()}
    result = dict(schema="orbit.taskflow.search-feedback.v1", method="deterministic structural enumeration and measured ranking",
                  score_unit="ms", timing_scope="complete graph host wall including synchronous device completion and all CPU placements",
                  limitations="Shared desktop, clocks unpinned, sequential batches; small differences do not establish speedup.",
                  candidates=feedback, rankings=ranking, shared_candidates=shared)
    write_json(args.output / "search-feedback.json", result)
    lines = ["# Taskflow experiment results", "", "All listed timings are complete graph host wall milliseconds; CPU placements are included.",
             "No Neura II or mapper estimate is mixed into these scores.", "", "| Case | Candidate | Tasks | NPU status | Median ms | Max abs | RMS relative |", "|---|---|---:|---|---:|---:|---:|"]
    for e in feedback:
        lines.append(f"| {e['case']} | {e['candidate']} | {e['task_count']} | {e['status']} | " +
                     (f"{e['median_host_wall_ms']:.4f} | {e['max_abs_error']:.6g} | {100*e['max_normalized_rms_error']:.3f}% |" if e['status'] == 'passed' else "— | — | — |"))
    lines += ["", f"Shared verified candidate identities consumed by both adapters: {len(shared)}.",
              "See `search-feedback.json` for every raw sample, seed, status, identity and measured ranking.", ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(dict(candidates=len(feedback), npu_passed=sum(e['status']=='passed' for e in feedback), shared_candidates=len(shared), rankings=ranking), indent=2))
    return bool(shared) and all(e["status"] == "passed" for e in feedback)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "cpu", "negative", "amoeba", "npu", "report", "all"])
    p.add_argument("--tool", type=Path, default=ROOT / "build-core/tools/taskflow-opt/taskflow-opt")
    p.add_argument("--output", type=Path, default=ROOT / "build/backend-decoupling")
    p.add_argument("--full", action="store_true", help="Include complete D=2048 I=11008 FFN at M=5 and 128")
    p.add_argument("--iterations", type=int, default=3)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.tool = args.tool.resolve()
    stages = [prepare, cpu, negative, amoeba, npu, report] if args.stage == "all" else [globals()[args.stage]]
    ok = True
    for stage in stages:
        ok = stage(args) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
