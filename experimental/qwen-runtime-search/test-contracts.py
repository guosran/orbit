#!/usr/bin/env python3
"""CPU-only contracts using synthetic timings, never performance evidence.

Requires an already built prototype and a frozen source manifest. It creates
only temporary copies/fixtures, never touches the NPU or existing evidence.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    args = parser.parse_args()
    producer = Path(__file__).with_name("make-local-cost-model.py")
    original = json.loads(args.graph.read_text())
    checks = []
    with tempfile.TemporaryDirectory(prefix="qwen-cpu-contract-") as directory:
        root = Path(directory)
        model = root / "fixture-model.bin"
        model.write_bytes(b"synthetic test model, not executable")
        alias = root / "fixture-model-alias.bin"
        alias.symlink_to(model)
        graph = json.loads(json.dumps(original))
        graph["model"]["path"] = str(model)
        graph["source_id"] = hashlib.sha256(canonical({
            key: value for key, value in graph.items() if key != "source_id"
        })).hexdigest()
        graph_path = root / "graph.json"
        graph_path.write_text(json.dumps(graph))
        timed = [node for node in graph["nodes"] if node["family"] not in {
            "opaque_decode", "kv_store", "kv_handoff"
        }]
        paths, traces = {}, {}
        for device in ["cpu", "npu"]:
            header = dict(event="loaded", model=str(alias), static_weights=True,
                          exact_k=True, bucket=1, outputs=128, threads=4, block_m=0,
                          mode="generate", runtime="xrt", decode_format="xclbin",
                          ops="matmul" if device == "cpu" else "all")
            rows = [header, dict(event="decode_loaded", maxL=2048, attn_maxl=2048)]
            for sample in range(3):
                ops = [[node["family"], 2.0 if device == "npu" and node["family"]
                        in {"rms_norm", "swiglu"} else 1.0] for node in timed]
                rows.append(dict(header, event=None, sample=sample,
                                 P=graph["input"]["prefill_tokens"], capacity=2048,
                                 tokens=[0] * 128, decode_ms=[1.0] * 127,
                                 op_records=ops, total_ms=sum(v for _, v in ops) + 128))
            traces[device] = rows
            paths[device] = root / (device + ".jsonl")

        def write_traces():
            for device in traces:
                paths[device].write_text("\n".join(json.dumps(row) for row in traces[device]) + "\n")

        def fit(output):
            return subprocess.run(["python3", str(producer), "--graph", str(graph_path),
                                   "--cpu", str(paths["cpu"]), "--npu", str(paths["npu"]),
                                   "--output", str(output)], capture_output=True, text=True)

        write_traces()
        price_path = root / "prices.json"
        result = fit(price_path)
        assert result.returncode == 0, result.stderr
        prices = json.loads(price_path.read_text())
        assert prices["host_residual_ms"] == 1
        assert all(node["cost_by_choice_ms"] == {"cpu": 1, "npu": 2}
                   for node in prices["nodes"] if "cost_by_choice_ms" in node)
        checks.append("source-node median costs and exclusive residual; model symlink identity")
        before = price_path.read_bytes()
        assert fit(price_path).returncode != 0 and price_path.read_bytes() == before
        checks.append("existing model is never overwritten")

        for case in ["source-id", "timer-order", "negative", "nonfinite", "decode-count", "threads"]:
            saved = json.loads(json.dumps(traces))
            if case == "source-id":
                graph["source_id"] = "bad"
                graph_path.write_text(json.dumps(graph))
            elif case == "timer-order":
                traces["cpu"][2]["op_records"][0][0] = "wrong-family"
            elif case == "negative":
                traces["cpu"][2]["op_records"][0][1] = -1
            elif case == "nonfinite":
                traces["cpu"][2]["op_records"][0][1] = float("nan")
            elif case == "decode-count":
                traces["cpu"][2]["decode_ms"].pop()
            elif case == "threads":
                traces["cpu"][0]["threads"] = 8
            write_traces()
            assert fit(root / (case + ".json")).returncode != 0, case
            traces = saved
            # Restore the valid synthetic graph with its fixture model identity.
            graph["model"]["path"] = str(model)
            graph["source_id"] = hashlib.sha256(canonical({
                key: value for key, value in graph.items() if key != "source_id"
            })).hexdigest()
            graph_path.write_text(json.dumps(graph))
            checks.append("producer rejects " + case)
        write_traces()

        for algorithm in ["beam", "cp"]:
            result = subprocess.run([str(args.binary), str(graph_path), str(price_path), algorithm],
                                    capture_output=True, text=True, check=True)
            payload = json.loads(result.stdout)
            assert payload["rms_norm_device"] == payload["swiglu_device"] == "cpu"
            assert payload["evaluator_calls"] == 4
            if algorithm == "cp":
                assert payload["report"]["finite_domain_optimality_proved"] is True
            checks.append("C++ " + algorithm + " selects synthetic local-cost minimum")

        for case in ["missing-node", "missing-choice", "negative", "source-id", "family", "duplicate", "residual"]:
            bad = json.loads(json.dumps(prices))
            if case == "missing-node":
                bad["nodes"].pop()
            elif case == "missing-choice":
                next(node for node in bad["nodes"] if "cost_by_choice_ms" in node)["cost_by_choice_ms"].pop("cpu")
            elif case == "negative":
                bad["nodes"][-1]["cost_ms"] = -1
            elif case == "source-id":
                bad["source_id"] = "wrong"
            elif case == "family":
                bad["nodes"][0]["family"] = "wrong"
            elif case == "duplicate":
                bad["nodes"][1]["id"] = bad["nodes"][0]["id"]
            elif case == "residual":
                bad["host_residual_boundary"] = "double-count launch edges"
            bad_path = root / ("bad-" + case + ".json")
            bad_path.write_text(json.dumps(bad))
            result = subprocess.run([str(args.binary), str(graph_path), str(bad_path), "cp"],
                                    capture_output=True, text=True)
            assert result.returncode == 1, case
            checks.append("C++ rejects " + case)
    print(json.dumps({"synthetic_contracts_only": True, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
