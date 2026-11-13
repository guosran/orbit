#!/usr/bin/env python3
"""Run an uncapped ORBIT search, sample memory, and seal the complete evidence.

The executable writes its own full JSONL traces directly to disk. This launcher
sets neither a timeout nor an evaluation limit. Linux wait4 supplies the child's
actual peak RSS, including final graph serialization between /proc samples.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def record(path):
    requested = Path(path).absolute()
    resolved = requested.resolve(strict=True)
    with resolved.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"requested_file": str(requested), "file": str(resolved),
            "sha256": checksum, "bytes": resolved.stat().st_size}


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def memory_sample(pid, elapsed):
    result = {"elapsed_seconds": elapsed, "utc": utc(), "pid": pid}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, text = line.partition(":")
        if key in {"VmRSS", "VmHWM", "VmSwap", "RssAnon", "RssFile", "RssShmem"}:
            value, unit = text.split()
            if unit != "kB":
                raise ValueError("Unexpected Linux /proc memory unit")
            result[key + "_bytes"] = int(value) * 1024
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            result["MemAvailable_bytes"] = int(line.split()[1]) * 1024
            break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("binary", "source", "catalogue", "output", "evidence-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--seal-file", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.beam_width <= 0:
        raise ValueError("Beam width must be positive")
    output = args.output.absolute()
    suffixes = ["", ".checkpoint.jsonl", ".beam-trace.jsonl", ".joint-actions.jsonl",
                ".best-checkpoint.json", ".certificate.json"]
    if any(Path(str(output) + suffix).exists() for suffix in suffixes):
        raise FileExistsError("Refusing to overwrite previous search evidence")
    inputs = [record(path) for path in [args.binary, args.source, args.catalogue, __file__,
              *([args.warm_start] if args.warm_start else []), *args.seal_file]]
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(args.binary.absolute()), str(args.source.absolute()),
            str(args.catalogue.absolute()), str(output), str(args.beam_width)]
    if args.warm_start:
        argv.extend(["--warm-start", str(args.warm_start.absolute())])
    launch = {"schema": "orbit.observed-search-launch.v1", "utc": utc(),
              "argv": argv, "inputs": inputs, "output": str(output),
              "warm_start_prior_result": str(args.warm_start.absolute()) if args.warm_start else None,
              "wall_timeout_seconds": None, "evaluation_limit": None,
              "round_limit": None, "sample_interval_seconds": 1,
              "stopping_policy": "ORBIT convergence; no launcher cutoff"}
    save(args.evidence_dir / "launch.json", launch)
    started = time.monotonic()
    stop = threading.Event()
    summary = {"samples": 0, "sampled_max_bytes": {}, "sampled_min_mem_available_bytes": None,
               "monitor_errors": []}
    samples_path = args.evidence_dir / "memory.jsonl"
    with (args.evidence_dir / "process.log").open("xb") as log, samples_path.open("x") as samples:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        save(args.evidence_dir / "process.json", {"pid": process.pid, "utc": utc(), "argv": argv})

        def observe():
            while True:
                try:
                    row = memory_sample(process.pid, time.monotonic() - started)
                    samples.write(json.dumps(row, allow_nan=False) + "\n")
                    samples.flush()
                    summary["samples"] += 1
                    for key, value in row.items():
                        if key.endswith("_bytes") and key != "MemAvailable_bytes":
                            summary["sampled_max_bytes"][key] = max(
                                summary["sampled_max_bytes"].get(key, 0), value)
                    available = row.get("MemAvailable_bytes")
                    if available is not None:
                        old = summary["sampled_min_mem_available_bytes"]
                        summary["sampled_min_mem_available_bytes"] = available if old is None else min(old, available)
                except (FileNotFoundError, ProcessLookupError):
                    break
                except Exception as error:
                    summary["monitor_errors"].append({"utc": utc(), "error": repr(error)})
                if stop.wait(1):
                    break

        monitor = threading.Thread(target=observe, name="search-memory-observer", daemon=True)
        monitor.start()
        # Do not call poll/wait first: wait4 must retain this particular child's
        # kernel RSS high-water mark instead of a previous child's aggregate.
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
        stop.set()
        monitor.join()
    completed = time.monotonic()
    summary.update(schema="orbit.observed-search-memory.v1", pid=process.pid,
                   elapsed_seconds=completed - started, returncode=process.returncode,
                   wait4_peak_rss_bytes=int(usage.ru_maxrss) * 1024,
                   wait4_user_seconds=usage.ru_utime, wait4_system_seconds=usage.ru_stime,
                   wait4_peak_scope="exact child high-water mark including final serialization",
                   swap_peak_scope="sampled /proc VmSwap; no kernel lifetime swap high-water counter")
    save(args.evidence_dir / "memory-summary.json", summary)
    final_inputs = [record(row["requested_file"]) for row in inputs]
    missing_outputs = [str(output) + suffix for suffix in suffixes
                       if not Path(str(output) + suffix).is_file()]
    artifacts = [record(Path(str(output) + suffix)) for suffix in suffixes
                 if Path(str(output) + suffix).is_file()]
    artifacts.extend(record(args.evidence_dir / name) for name in
                     ("launch.json", "process.json", "process.log", "memory.jsonl", "memory-summary.json"))
    seal = {"schema": "orbit.observed-search-final-seal.v1", "utc": utc(),
            "returncode": process.returncode, "inputs_unchanged": final_inputs == inputs,
            "inputs_before": inputs, "inputs_after": final_inputs, "artifacts": artifacts,
            "search_elapsed_seconds": completed - started,
            "observer_and_seal_elapsed_seconds": time.monotonic() - started,
            "complete_search_result_present": output.is_file(),
            "missing_expected_outputs": missing_outputs}
    save(args.evidence_dir / "seal.json", seal)
    print(json.dumps({"returncode": process.returncode, "inputs_unchanged": seal["inputs_unchanged"],
                      "search_elapsed_seconds": summary["elapsed_seconds"],
                      "wait4_peak_rss_bytes": summary["wait4_peak_rss_bytes"],
                      "seal": str((args.evidence_dir / "seal.json").absolute())}))
    if process.returncode == 0 and (not seal["inputs_unchanged"] or missing_outputs):
        return 1
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
