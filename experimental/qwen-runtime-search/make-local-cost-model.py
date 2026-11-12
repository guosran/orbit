#!/usr/bin/env python3
"""Fit local Qwen runtime costs from separately collected CPU/NPU training traces.

This model is specific to the source manifest, prompt, and prepared runtime.
It allocates synchronous call wall times to source nodes. Launch and transfer
costs already belong to those times, so consumers must not add them again.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

RESIDUAL_BOUNDARY = (
    "total_request_ms - sum(prefill_op_records_ms) - sum(inclusive_decode_steps_ms); "
    "includes kv handoff, embedding, inline/control, capture copies outside dispatch"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--npu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text())
    payload = {key: value for key, value in graph.items() if key != "source_id"}
    assert graph["source_id"] == hashlib.sha256(canonical(payload)).hexdigest()
    length = graph["input"]["prefill_tokens"]
    assert length in [5, 128, 512]

    groups, provenance = {}, []
    for device, path in [("cpu", args.cpu), ("npu", args.npu)]:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        loaded = next(row for row in rows if row.get("event") == "loaded")
        expected_ops = "matmul" if device == "cpu" else "all"
        assert loaded["static_weights"] is True and loaded["exact_k"] is True
        assert loaded["bucket"] == 1 and loaded["outputs"] == 128
        assert loaded["ops"] == expected_ops
        assert loaded["threads"] == 4 and loaded["block_m"] == 0
        assert loaded["mode"] == "generate" and loaded["runtime"] == "xrt"
        assert loaded["decode_format"] == "xclbin"
        decoder = next(row for row in rows if row.get("event") == "decode_loaded")
        assert decoder["maxL"] == 2048 and decoder["attn_maxl"] == 2048
        assert Path(loaded["model"]).resolve(strict=True) == Path(
            graph["model"]["path"]
        ).resolve(strict=True), "model binding mismatch"
        samples = [row for row in rows if "sample" in row and row["P"] == length]
        assert len(samples) >= 3
        assert len({row["sample"] for row in samples}) == len(samples)
        groups[device] = samples
        provenance.append(
            {
                "device": device,
                "path": str(path),
                "sha256": digest(path),
                "P": length,
                "samples": [row["sample"] for row in samples],
            }
        )

    # The source adapter preserves execution order. Match every timed call,
    # including both RoPE calls per layer, before assigning any local price.
    untimed = {"opaque_decode", "kv_store", "kv_handoff"}
    timed = [node for node in graph["nodes"] if node["family"] not in untimed]
    family_sequence = [node["family"] for node in timed]
    residuals = []
    for device, samples in groups.items():
        for row in samples:
            assert [key for key, _ in row["op_records"]] == family_sequence
            assert len(row["decode_ms"]) == 127 and len(row["tokens"]) == 128
            assert row["capacity"] == 2048 and row["outputs"] == 128
            assert row["static_weights"] is True and row["exact_k"] is True
            assert row["bucket"] == 1 and row["threads"] == 4 and row["block_m"] == 0
            assert row["ops"] == ("matmul" if device == "cpu" else "all")
            values = [value for _, value in row["op_records"]] + row["decode_ms"]
            assert all(math.isfinite(value) and value >= 0 for value in values)
            # Inclusive dispatch steps contain their internal host/device work.
            # The exclusive remainder owns KV handoff and uninstrumented work.
            residual = (
                row["total_ms"]
                - sum(value for _, value in row["op_records"])
                - sum(row["decode_ms"])
            )
            assert math.isfinite(residual) and residual >= 0
            residuals.append(residual)

    timed_index = {node["id"]: index for index, node in enumerate(timed)}
    prices = []
    for node in graph["nodes"]:
        family = node["family"]
        price = {"id": node["id"], "family": family}
        if family in ["rms_norm", "swiglu"]:
            index = timed_index[node["id"]]
            price["cost_by_choice_ms"] = {
                device: statistics.median(row["op_records"][index][1] for row in rows)
                for device, rows in groups.items()
            }
        elif family == "opaque_decode":
            index = int(node["id"].split(".")[-1])
            price["cost_ms"] = statistics.median(
                row["decode_ms"][index] for rows in groups.values() for row in rows
            )
        elif family in ["kv_store", "kv_handoff"]:
            price["cost_ms"] = 0
            price["accounting"] = (
                "absorbed by host_residual_ms; zero allocates accounting, "
                "it does not mean measured free work"
            )
        else:
            index = timed_index[node["id"]]
            price["cost_ms"] = statistics.median(
                row["op_records"][index][1] for rows in groups.values() for row in rows
            )
        prices.append(price)

    output = {
        "schema": "orbit.external-qwen-local-cost.v1",
        "source_id": graph["source_id"],
        "cost_boundary": "synchronous-op-wall-includes-launch-and-transfer",
        "nodes": prices,
        "host_residual_ms": statistics.median(residuals),
        "host_residual_samples_ms": residuals,
        "training": provenance,
        "training_policy": (
            "per-node median for variable CPU/NPU; pooled two-cohort median "
            "for fixed nodes and mutually exclusive host residual"
        ),
        "host_residual_boundary": RESIDUAL_BOUNDARY,
        "edge_cost_policy": (
            "no extra transfer/launch edge charge; synchronous local calls include them"
        ),
        "validation_trace_used_for_fit": False,
        "producer_sha256": digest(__file__),
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "P": length, "nodes": len(prices)}))


if __name__ == "__main__":
    main()
