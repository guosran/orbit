#!/usr/bin/env python3
"""Audit and summarize this round's recorded measurements without hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from taskflow_common import write_json

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured(path):
    r = load(path)
    if r["status"] != "passed":
        return dict(status=r["status"], reason=r.get("error"), evidence=str(path))
    assert len(r["host_wall_ms"]) >= 3
    assert all(c["passed"] for c in r["numerical_checks"])
    assert r["device_dispatches_per_graph"] > 0
    metrics = [m for c in r["numerical_checks"] for m in c["outputs"].values()]
    return dict(status="passed", candidate_id=r["candidate_id"],
        implementation_sha256=r["implementation_sha256"], input_sha256=r["input_snapshot_sha256"],
        samples_ms=r["host_wall_ms"], median_ms=statistics.median(r["host_wall_ms"]),
        max_abs=max(m["max_abs"] for m in metrics), normalized_rms=max(m["normalized_rms_error"] for m in metrics),
        dispatches=r["device_dispatches_per_graph"], readbacks=r["output_readbacks_per_graph"],
        compile_seconds=r["compile_seconds"], first_weight_preparation_seconds=r["first_weight_preparation_seconds"],
        chain_load_seconds=r.get("chain_load_seconds"), staged_load_seconds=r.get("staged_load_seconds"),
        hot_weight_staging_bytes_theoretical=r["hot_weight_staging_bytes_theoretical"],
        evidence=str(path), evidence_sha256=sha(path))


def summarize(args):
    root, npu = args.output, args.npu_results
    comparisons = []
    for case in ("ffn_m5", "ffn_m128"):
        input_identity = None
        for config in ("baseline", "persistent", "merge", "combined"):
            runs = [measured(npu / order / case / "seed-0" / config / "result.json")
                    for order in ("full-forward-final-timing", "full-reverse-final-timing")]
            assert all(r["status"] == "passed" for r in runs)
            assert runs[0]["implementation_sha256"] == runs[1]["implementation_sha256"], "cannot pool different implementations"
            assert runs[0]["input_sha256"] == runs[1]["input_sha256"]
            if input_identity is None:
                input_identity = runs[0]["input_sha256"]
            assert input_identity == runs[0]["input_sha256"], "cannot compare different weights/input"
            samples = [x for r in runs for x in r["samples_ms"]]
            comparisons.append(dict(case=case, config=config, status="passed", runs=runs,
                median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples),
                samples_ms=samples, max_abs=max(r["max_abs"] for r in runs),
                normalized_rms=max(r["normalized_rms"] for r in runs),
                dispatches=runs[0]["dispatches"], readbacks=runs[0]["readbacks"],
                hot_weight_staging_bytes_theoretical=runs[0]["hot_weight_staging_bytes_theoretical"]))
    searches = []
    names = ["amoeba-gemm-formal-cacheguard", "npu-ffn_m5-final-timing", "npu-ffn_m128-final-timing"]
    for name in names:
        file = root / "search" / name / "search.json"
        s = load(file)
        records = []
        for r in s["archive"]:
            evidence = r.get("evaluation") or {}
            if r["status"] == "passed":
                assert evidence["status"] == "passed" and evidence["score"]["value"] == r["score"]
                assert evidence["candidate_id"] == r["candidate_id"]
                assert evidence["score"]["unit"] == s["score_unit"]
            records.append({k:r[k] for k in ("search_key", "candidate_id", "parent", "action", "round", "status", "score")})
        searches.append(dict(name=name, backend=s["backend"], unit=s["score_unit"], evaluations=s["evaluations"],
            records=records, selections=s["selections"], ranking=s["ranking"], attempts=s["attempts"], evidence=str(file)))
    confirms = []
    for suffix in ("m5", "m128"):
        directory = npu / f"winner-confirm-{suffix}-final-timing"
        rows = load(directory / "summary.json")
        confirms.append(dict(case="ffn_"+suffix, selected=load(directory / "selected-search-candidate.json"),
            runs=[dict(name=r["name"], round=r["round"], action=r["action"], **measured(Path(r["result"]))) for r in rows], evidence=str(directory / "summary.json")))
    failures = []
    for suffix in ("m5", "m128"):
        summary = npu / ("k-compare-"+suffix) / "summary.json"
        if summary.exists():
            for r in load(summary):
                if r["status"] != "passed":
                    failures.append(dict(case=r["case"], action=r["action"], accumulation=r["accumulation"],
                        **measured(Path(r["result"]))))
    suites = {}
    for name in ("small-final-timing", "device-small-final-timing"):
        path = npu / name / "summary.json"
        if path.exists():
            rows = load(path)
            assert all(r["status"] == "passed" for r in rows)
            suites[name] = dict(passed=len(rows), evidence=str(path))
    for name in ("lifetime-final-timing", "device-lifetime-final-timing"):
        path = npu / name / "result.json"
        if path.exists():
            r = load(path)
            assert r["passed"]
            suites[name] = dict(passed=len(r["checks"]), evidence=str(path))
    result = dict(npu_validation=suites, schema="orbit.search-npu-v2.report.v1", comparisons=comparisons, searches=searches,
        winner_confirmation=confirms, device_k_failures=failures,
        validation=load(root / "validation/summary.json") if (root / "validation/summary.json").exists() else {"status":"see individual validation logs; aggregate not collected"},
        timing_scope="Complete synchronous graph host wall including CPU activation, conversions, dispatch and final readback.",
        pooling="Two opposite-order final batches with identical input/weight and implementation fingerprints; 7 samples per batch.",
        limitations="Shared machine, unlocked clocks; byte counts are theoretical API accounting, not measured hardware traffic. No physical Amoeba CGRA execution.")
    write_json(root / "final-report.json", result)
    lines = ["# ORBIT 搜索与 NPU FFN 本轮结果", "", "仅汇总 final-timing 批次；不使用上一轮历史延迟计算加速。", "",
        "每项 2 个相反执行顺序批次，每批 7 次完整热图。输入/权重与实现身份已核验。", "",
        "| 配置 | M=5 ms | M=128 ms | 提交/读回 |", "|---|---:|---:|---|"]
    for config in ("baseline", "persistent", "merge", "combined"):
        rows = [next(r for r in comparisons if r["case"]==case and r["config"]==config) for case in ("ffn_m5", "ffn_m128")]
        lines.append(f"| {config} | {rows[0]['median_ms']:.3f} | {rows[1]['median_ms']:.3f} | {rows[0]['dispatches']}/{rows[0]['readbacks']} |")
    lines += ["", "## 原 ORBIT 的实际搜索轨迹", ""]
    for s in searches:
        lines += [f"### {s['name']}", "", f"评价 {s['evaluations']} 个候选，单位 {s['unit']}。", "",
            "| 轮次 | 动作 | 状态 | 分数 | 公共候选 ID 前缀 |", "|---:|---|---|---:|---|"]
        for r in s["records"]:
            score = f"{r['score']:.4f}" if r["score"] is not None else "未评分"
            lines.append(f"| {r['round']} | {r['action']} | {r['status']} | {score} | {r['candidate_id'][:12]} |")
        lines += ["", "父候选、动作链、选择理由、预算未测量及重复/非法动作见 final-report.json 和原 search.json。", ""]
    lines += ["## 独立复验", "", "| 尺寸 | 选中动作 | 正序 baseline / best ms | 逆序 baseline / best ms |", "|---|---|---:|---:|"]
    for c in confirms:
        values = {(r["round"], r["name"]):r["median_ms"] for r in c["runs"]}
        lines.append(f"| {c['case']} | {c['selected']['action']} | {values[0,'baseline']:.3f} / {values[0,'best']:.3f} | {values[1,'baseline']:.3f} / {values[1,'best']:.3f} |")
    lines += ["", "M5 的 N5632 优势在逆序复验消失；M128 全局最优仍是 identity，复验两组是同一公共候选，差异反映测量波动。不能据此宣称结构加速或结构退化。", "", "权重复用的改善包含绑定式 runner 的开销变化，不是隔离的 DMA 测量。合并投影收益较小且 M5 逆序有退化；频率未锁定。", ""]
    lines += ["## 边界", "", "设备 K 累加在小尺寸已验证；真实 FFN K5504 的嵌套循环方案编译失败，保留错误与完整最小复现，不计为设备成功。",
        "权重 BO 位于 NPU 可访问系统内存，不是 AIE SRAM。SiLU/SwiGLU 仍在 CPU。", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(dict(comparisons=len(comparisons), searches=len(searches), device_k_failures=len(failures)), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "build/search-npu-v2")
    p.add_argument("--npu-results", type=Path, default=Path("/home/shiran/project/aie-toy/results/orbit-search-v2"))
    summarize(p.parse_args())


if __name__ == "__main__":
    main()
