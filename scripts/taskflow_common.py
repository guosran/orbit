#!/usr/bin/env python3
"""Fixtures and independent numerical oracle for the restricted Taskflow exchange.

This is not a compiler: taskflow-opt owns materialization and semantic validation.
The plan interpreter below is a CPU diagnostic, never a device result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "orbit.taskflow.plan.v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_plan(workload, m=5, n=19, k=23, d=16, i=24):
    buffers, tasks = [], []

    def buffer(name, rows, cols, role):
        buffers.append(dict(id=name, shape=[rows, cols], dtype="f32", role=role))

    def matmul(name, a, b, c, rows, cols, reduction, deps=()):
        tasks.append(dict(id=name, depends_on=list(deps), operations=[
            dict(kind="fill", inputs=[], output=c, m=[0, rows], n=[0, cols]),
            dict(kind="matmul", inputs=[a, b], output=c, m=[0, rows], n=[0, cols],
                 k=[0, reduction], accumulate=True)]))

    if workload in ("gemm", "gemv", "gemm_relu"):
        if workload == "gemv":
            m = 1
        buffer("A", m, k, "input")
        buffer("B", k, n, "input")
        buffer("C", m, n, "temporary" if workload == "gemm_relu" else "output")
        matmul("gemm", "A", "B", "C", m, n, k)
        outputs = ["C"]
        if workload == "gemm_relu":
            buffer("R", m, n, "output")
            tasks.append(dict(id="relu", depends_on=["gemm"], operations=[
                dict(kind="relu", inputs=["C"], output="R", m=[0, m], n=[0, n])]))
            outputs = ["R"]
    elif workload == "ffn":
        for name, rows, cols in (("X", m, d), ("W_gate", d, i), ("W_up", d, i), ("W_down", i, d)):
            buffer(name, rows, cols, "input")
        for name in ("G", "U", "H"):
            buffer(name, m, i, "temporary")
        buffer("Y", m, d, "output")
        matmul("gate", "X", "W_gate", "G", m, i, d)
        matmul("up", "X", "W_up", "U", m, i, d)
        tasks.append(dict(id="activation", depends_on=["gate", "up"], operations=[
            dict(kind="swiglu", inputs=["G", "U"], output="H", m=[0, m], n=[0, i])]))
        matmul("down", "H", "W_down", "Y", m, d, i, ["activation"])
        outputs = ["Y"]
    else:
        raise ValueError(f"unsupported fixture: {workload}")
    plan = dict(schema=SCHEMA, buffers=buffers, tasks=tasks, outputs=outputs, transforms=[])
    plan["source_id"] = fingerprint(plan)
    plan["candidate_id"] = plan["source_id"]
    return plan


def make_inputs(plan, seed):
    """Signed, asymmetric dense data, including explicit zeros; no overflow."""
    rng = np.random.default_rng(seed)
    result = {}
    for buf in plan["buffers"]:
        if buf["role"] != "input":
            continue
        scale = 0.5 if buf["id"] in ("A", "X") else 0.5 / np.sqrt(buf["shape"][0])
        value = (rng.normal(size=buf["shape"]) * scale).astype(np.float32)
        value.flat[::17] = 0
        result[buf["id"]] = value
    return result


def reference(workload, inputs):
    """Independent whole-matrix float64 oracle; never uses candidate slices."""
    x = {key: np.asarray(value, dtype=np.float64) for key, value in inputs.items()}
    if workload in ("gemm", "gemv", "gemm_relu"):
        out = x["A"] @ x["B"]
        return {"R": np.maximum(out, 0)} if workload == "gemm_relu" else {"C": out}
    if workload == "ffn":
        gate, up = x["X"] @ x["W_gate"], x["X"] @ x["W_up"]
        sigmoid = np.exp(-np.logaddexp(0, -gate))
        return {"Y": (gate * sigmoid * up) @ x["W_down"]}
    raise ValueError(workload)


def execute_cpu(plan, inputs):
    """Diagnostic execution of materialized operations, not an independent oracle."""
    if plan["schema"] != SCHEMA:
        raise ValueError("unsupported plan schema")
    memory = {}
    for buf in plan["buffers"]:
        if buf["dtype"] != "f32":
            raise ValueError("unsupported dtype")
        if buf["role"] == "input":
            a = np.array(inputs[buf["id"]], dtype=np.float32, copy=True)
            if list(a.shape) != buf["shape"]:
                raise ValueError("input shape mismatch")
            memory[buf["id"]] = a
        else:
            memory[buf["id"]] = np.full(buf["shape"], np.nan, dtype=np.float32)
    completed = set()
    for task in plan["tasks"]:
        if not set(task["depends_on"]) <= completed:
            raise ValueError("task order violates dependencies")
        for op in task["operations"]:
            ms, ns = slice(*op["m"]), slice(*op["n"])
            dst = memory[op["output"]]
            args = [memory[name] for name in op["inputs"]]
            if op["kind"] == "fill":
                dst[ms, ns] = 0
            elif op["kind"] == "matmul":
                if op.get("accumulate") is not True:
                    raise ValueError("matmul must explicitly accumulate")
                ks = slice(*op["k"])
                # Sequential FP32 reference preserves each explicit K chunk.
                for kk in range(ks.start, ks.stop):
                    dst[ms, ns] += args[0][ms, kk:kk + 1] * args[1][kk:kk + 1, ns]
            elif op["kind"] == "relu":
                dst[ms, ns] = np.maximum(args[0][ms, ns], 0)
            elif op["kind"] == "swiglu":
                gate = args[0][ms, ns]
                dst[ms, ns] = gate / (1 + np.exp(-gate)) * args[1][ms, ns]
            else:
                raise ValueError(f"unsupported operation {op['kind']}")
        completed.add(task["id"])
    return {name: memory[name] for name in plan["outputs"]}


def compare(actual, expected, atol, rtol):
    checks = {}
    for name, ref in expected.items():
        value = np.asarray(actual[name])
        shape_ok = value.shape == ref.shape
        finite = bool(np.isfinite(value).all())
        if shape_ok and finite:
            error = np.abs(value.astype(np.float64) - ref)
            checks[name] = dict(shape=list(value.shape), finite=True,
                max_abs=float(error.max(initial=0)), mean_abs=float(error.mean()),
                rms_abs=float(np.sqrt(np.mean(error * error))),
                max_rel_nonzero=float(np.max(error / np.maximum(np.abs(ref), 1e-12), initial=0)),
                failed_elements=int(np.count_nonzero(error > atol + rtol * np.abs(ref))),
                passed=bool(np.all(error <= atol + rtol * np.abs(ref))))
        else:
            checks[name] = dict(shape=list(value.shape), finite=finite, passed=False,
                                error="shape mismatch" if not shape_ok else "NaN/Inf")
    return dict(passed=all(c["passed"] for c in checks.values()), atol=atol, rtol=rtol, outputs=checks)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
