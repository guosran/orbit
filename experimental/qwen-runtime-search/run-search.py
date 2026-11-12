#!/usr/bin/env python3
"""Verify frozen sources, fit local training costs, run C++ searches, seal results.

This command does not access a device. Run it after training completes, before
opening independent heldout labels. Existing models/results are never replaced.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def save(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--npu", type=Path, required=True)
    args = parser.parse_args()
    own_dir = Path(__file__).resolve().parent
    adapter = own_dir.parents[1] / "scripts/qwen-runtime-adapter.py"
    spec = importlib.util.spec_from_file_location("qwen_source_adapter", adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sealed = [args.binary, adapter, Path(__file__), own_dir / "qwen-local-search.cpp",
              own_dir / "make-local-cost-model.py", own_dir / "build-prototype.py",
              args.cpu, args.npu]
    build_receipt = Path(str(args.binary) + ".build.json")
    if build_receipt.exists():
        sealed.append(build_receipt)
    summaries = []
    for length in [5, 128, 512]:
        graph_path = args.root / f"source-p{length}.json"
        graph = json.loads(graph_path.read_text())
        assert graph["input"]["prefill_tokens"] == length
        started = time.perf_counter()
        # This rehashes actual source, model, tokenizer, and decoder artifacts.
        # A source ID string comparison alone cannot replace this validation.
        module.check_manifest(graph)
        check_path = args.root / f"source-check-p{length}.json"
        save(check_path, {
            "schema": "orbit.external-qwen-source-check.v1",
            "source_id": graph["source_id"], "manifest": identity(graph_path),
            "adapter": identity(adapter), "actual_files_rehashed": True,
            "verified_groups": ["source", "model", "tokenizer", "decoder_artifacts"],
            "passed": True, "verification_ms": (time.perf_counter() - started) * 1000,
        })
        model_path = args.root / f"model-p{length}.json"
        producer_command = ["python3", str(own_dir / "make-local-cost-model.py"),
                            "--graph", str(graph_path), "--cpu", str(args.cpu),
                            "--npu", str(args.npu), "--output", str(model_path)]
        subprocess.run(producer_command, check=True)
        search_dir = args.root / f"search-p{length}"
        search_dir.mkdir(exist_ok=True)
        sealed.extend([graph_path, check_path, model_path])
        for algorithm in ["beam", "cp"]:
            command = [str(args.binary), str(graph_path), str(model_path), algorithm]
            started = time.perf_counter()
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            external_ms = (time.perf_counter() - started) * 1000
            payload = json.loads(result.stdout)
            assert payload["source_id"] == graph["source_id"]
            if algorithm == "cp":
                assert payload["report"]["finite_domain_optimality_proved"] is True
            assert payload["evaluator_calls"] == 4
            output = search_dir / f"{algorithm}.json"
            save(output, payload)
            receipt_path = search_dir / f"{algorithm}.command.json"
            save(receipt_path, {"command": command, "external_wall_ms": external_ms,
                                "returncode": result.returncode, "stderr": result.stderr})
            sealed.extend([output, receipt_path])
            summaries.append({
                "P": length, "algorithm": algorithm,
                "status": payload["report"]["status"],
                "evaluations": payload["evaluator_calls"],
                "total_ms": payload["report"]["total_ms"],
                "external_wall_ms": external_ms,
                "model_cost_ms": payload["best_cost_ms"],
                "rms_norm_device": payload["rms_norm_device"],
                "swiglu_device": payload["swiglu_device"],
            })
    save(args.root / "seal.json", {
        "schema": "orbit.external-qwen-search-seal.v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": "selection sealed before independent four-placement heldout measurements",
        "validation_used_for_fit": False,
        "scope": "two global runtime placements; opaque complete decode; external adapter",
        "files": [identity(path) for path in sealed], "summary": summaries,
    })
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
