#!/usr/bin/env python3
"""Prepare shared inputs and invoke the original ORBIT structured search pass.

This launcher does not enumerate or select candidates. C++ ORBIT search owns
actions, materialization, budgets, feedback selection, archives and recovery.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from taskflow_common import source_plan, make_inputs, reference, write_json, fingerprint
from taskflow_search_evaluate import implementation_identity

ROOT = Path(__file__).resolve().parents[1]


def prepare(args):
    cases = [(name, source_plan(name)) for name in ("gemm", "gemv", "gemm_relu", "ffn")]
    cases += [(f"ffn_m{m}", source_plan("ffn", m=m, d=2048, i=11008)) for m in (5, 128)]
    tool = ROOT / "build-core/tools/taskflow-opt/taskflow-opt"
    manifest = []
    for name, plan in cases:
        directory = args.output / "data" / name
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / "source.json"
        if source.exists():
            if json.loads(source.read_text()) != plan:
                raise ValueError(f"refusing to replace different existing source {source}")
        else:
            write_json(source, plan)
        seeds = [0] if name.startswith("ffn_m") else [0, 1, 2]
        for seed in seeds:
            input_path, ref_path = directory / f"inputs-{seed}.npz", directory / f"reference-{seed}.npz"
            if not input_path.exists() or not ref_path.exists():
                inputs = make_inputs(plan, seed)
                ref = reference("ffn" if name.startswith("ffn") else name, inputs)
                if not input_path.exists():
                    np.savez(input_path, **inputs)
                if not ref_path.exists():
                    np.savez(ref_path, **ref)
        if not (directory / "base.mlir").exists():
            result = subprocess.run([str(tool), str(source), f"--emit-plan={directory/'base.json'}",
                                     f"--emit-mlir={directory/'base.mlir'}"], text=True, capture_output=True)
            (directory / "materialize.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(result.stderr)
        manifest.append(dict(case=name, directory=str(directory), source_id=plan["source_id"], seeds=seeds))
        print(f"prepared {name}", flush=True)
    write_json(args.output / "data-manifest.json", manifest)


def configure(args):
    identity = implementation_identity(args.backend)
    path = args.output / "data" / args.case
    target = dict(evaluation=dict(inputs=str(path / "inputs-0.npz"),
        reference=str(path / "reference-0.npz"), source_plan=str(path / "source.json"),
        iterations=args.iterations, seed=0, case=args.case,
        cache_dir=str(args.output / "evaluation-cache"),
        measurement_batch=str(args.output / args.label), use_cache=not args.no_cache,
        implementation_fingerprint=fingerprint(identity)))
    if args.backend == "npu":
        target["npu"] = dict(schema="orbit.npu.config.v1", weights=args.weights,
                              projection=args.projection, accumulation=args.accumulation)
    else:
        target["amoeba"] = dict(architecture=str(ROOT / "test/multi-cgra/taskflow/joint-scheduling/mapper-arch-4x4.yaml"))
    # Backend proposal preferences, not public semantic restrictions.
    target["search_preferences"] = dict(m_alignment=64, n_alignment=256, k_alignment=64) if args.backend == "npu" else dict(m_alignment=1, n_alignment=1, k_alignment=1)
    file = args.output / "configs" / f"{args.backend}-{args.case}-{args.label}.json"
    if file.exists() and json.loads(file.read_text()) != target:
        raise ValueError(f"refusing to overwrite different target configuration {file}; use a new label/output")
    write_json(file, target)
    write_json(file.with_suffix(".implementation.json"), identity)
    print(file, flush=True)
    return file


def search(args):
    directory = args.output / "search" / f"{args.backend}-{args.case}-{args.label}"
    if (directory / "checkpoint.json").exists() and not args.resume:
        raise ValueError(f"search already exists at {directory}; use --resume or a new label/output")
    target = configure(args)
    directory.mkdir(parents=True, exist_ok=True)
    source = args.output / "data" / args.case / "source.json"
    mlir = args.output / "data" / args.case / "base.mlir"
    optimizer = ROOT / "build-amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt"
    options = dict(**{"decision-flow":"structured", "backend":args.backend,
        "structured-plan":str(source), "evaluator":str(ROOT / "scripts/taskflow_search_evaluate.py"),
        "target-config":str(target), "output-dir":str(directory), "max-rounds":args.rounds,
        "max-candidates":args.budget, "beam-width":args.beam})
    if args.resume:
        options["resume"] = "true"
    # MLIR pass options require quoting paths with spaces; reject these here
    # rather than silently constructing a different option stream.
    if any(any(c.isspace() for c in str(v)) for v in options.values()):
        raise ValueError("this launcher requires paths without whitespace")
    option_string = " ".join(f"{key}={value}" for key, value in options.items())
    argv = [str(optimizer), str(mlir), "--search-joint-neighborhood=" + option_string,
            "-o", str(directory / "search-output.mlir")]
    write_json(directory / "launch.json", dict(command=argv, backend=args.backend,
        source=str(source), target_config=str(target), note="Candidates are proposed/selected inside original ORBIT C++ pass."))
    with (directory / "search.log").open("w") as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"ORBIT search failed ({result.returncode}); see {directory/'search.log'}")
    print(f"completed original ORBIT search: {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "configure", "search"])
    parser.add_argument("--output", type=Path, default=ROOT / "build/search-npu-v2")
    parser.add_argument("--backend", choices=["npu", "amoeba"], default="npu")
    parser.add_argument("--case", default="ffn_m5")
    parser.add_argument("--label", default="formal")
    parser.add_argument("--weights", choices=["staged", "persistent"], default="persistent")
    parser.add_argument("--projection", choices=["separate", "merge_siblings"], default="merge_siblings")
    parser.add_argument("--accumulation", choices=["host", "device"], default="host")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--budget", type=int, default=9)
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    {"prepare":prepare, "configure":configure, "search":search}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
