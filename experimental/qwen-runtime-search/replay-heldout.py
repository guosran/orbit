#!/usr/bin/env python3
"""Replay the sealed search selection and the other three runtime placements.

Run in the existing AIE Python environment. Each process uses the original
runtime's device lock; an outer flock would deadlock that lock acquisition.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    adapter = Path(__file__).resolve().parents[2] / "scripts/qwen-runtime-adapter.py"
    seal = json.loads((root / "seal.json").read_text())
    for item in seal["files"]:
        digest = hashlib.sha256()
        with Path(item["path"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError("Changed sealed search input: " + item["path"])

    for length in (5, 128, 512):
        results = [json.loads((root / f"search-p{length}/{algorithm}.json").read_text())
                   for algorithm in ("beam", "cp")]
        # Never interpret assignment indices using the manifest's choice order:
        # the C++ search enumerates choices in a different order.
        choices = [(r["rms_norm_device"], r["swiglu_device"]) for r in results]
        if choices[0] != choices[1]:
            raise ValueError("Beam and CP differ; register separate selected replays")
        selected = choices[0]
        pairs = [selected] + [pair for pair in
                             (("cpu", "cpu"), ("cpu", "npu"),
                              ("npu", "cpu"), ("npu", "npu")) if pair != selected]
        for rms, swiglu in pairs:
            name = f"{args.prefix}-p{length}-{rms}-{swiglu}"
            manifest = root / f"source-p{length}.json"
            plan = root / (name + ".plan.json")
            subprocess.run([sys.executable, str(adapter), "plan", "--manifest",
                            str(manifest), "--rms", rms, "--swiglu", swiglu,
                            "--output", str(plan)], check=True)
            command = ["timeout", "-k", "10s", "300s", sys.executable, str(adapter),
                       "replay", "--manifest", str(manifest), "--plan", str(plan),
                       "--name", name, "--receipt", str(root / (name + ".receipt.json")),
                       "--execute"]
            save(root / (name + ".command.json"), command)
            start = time.monotonic()
            with (root / (name + ".log")).open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            save(root / (name + ".result.json"), {
                "returncode": result.returncode, "seconds": time.monotonic() - start,
                "selected_by_beam_and_cp": (rms, swiglu) == selected,
            })
            print(name, result.returncode, flush=True)
            result.check_returncode()


if __name__ == "__main__":
    main()
