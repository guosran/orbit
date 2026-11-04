#!/usr/bin/env python3
"""Summarize independently audited trace/chain evidence without device access."""
import argparse
from pathlib import Path
import statistics
from taskflow_static_io import load,write,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hardware-root',type=Path,default=Path('/home/shiran/project/aie-toy/results/orbit-static-v3-hardware'))
    a=p.parse_args();root=a.output.resolve();hw=a.hardware_root.resolve()
    graphs=[]
    for name in ('gemm','ffn_m5','ffn_m128'):
        path=hw/'orbit-trace-instrumented'/name/'result.json';r=load(path)
        assert r['status']=='passed'
        row=dict(case=name,candidate_id=r['candidate_id'],variants={},trace_manifests=[],evidence=str(path),sha256=digest(path))
        for mode,v in r['variants'].items():
            assert len(v['full_graph_ms'])>=3 and all(c['passed'] for c in v['numeric_checks'])
            events=[e for e in load(path.parent/mode/'host-events.json') if not e['excluded']]
            assert all(e['actual_submissions']==v['submissions_per_graph'] for e in events)
            # These are host API durations, never device-only execution times.
            measures={kind:statistics.median(sum((x['end_ns']-x['start_ns'])/1e6 for x in e['api_events'] if x['kind']==kind) for e in events)
                      for kind in ('actual_bo_sync','actual_xrt_start','actual_xrt_wait')}
            measures['cpu_swiglu_ms']=statistics.median(sum((x['end_ns']-x['start_ns'])/1e6 for x in e['sections'] if x['kind']=='swiglu') for e in events)
            row['variants'][mode]=dict(median_ms=statistics.median(v['full_graph_ms']),samples_ms=v['full_graph_ms'],submissions=v['submissions_per_graph'],host_api_medians_ms=measures,
                excluded_ms=v['excluded_ms'],initial_load_ms=v['load_ms'],first_host_preparation_ms=v['first_host_preparation_ms'])
        row['perturbation_ratio']=row['variants']['on']['median_ms']/row['variants']['off']['median_ms']
        for m in sorted((path.parent/'on').glob('trace-*/manifest.json')):
            manifest=load(m);assert manifest['nonzero_words']>0
            row['trace_manifests'].append(dict(path=str(m),sha256=digest(m),task=manifest['task_id'],operation=manifest['operation_index'],raw_bytes=manifest['capacity_bytes'],nonzero_words=manifest['nonzero_words'],events=manifest['event_count'],completeness=manifest['completeness'],perfetto=str(m.parent/'trace.perfetto.json')))
        graphs.append(row)
    chain_audit=load(root/'validation/hardware/chain-bf16-audit.json')
    assert chain_audit['passed']
    final_path=root/'validation/hardware/final-audit.json'
    final=load(final_path);assert final['passed']
    chain=[]
    for case in ('train','holdout'):
        path=hw/'chain-bf16'/case/'result.json';r=load(path);assert r['status']=='passed'
        chain.append(dict(case=case,shape=r['shape'],variants={k:dict(median_ms=v['median_ms'],samples_ms=[s['wall_ms'] for s in v['samples']],actual_submissions=[s['actual_submissions'] for s in v['samples']]) for k,v in r['variants'].items()},evidence=str(path),sha256=digest(path)))
    result=dict(schema='orbit.hardware.summary.v1',graphs=graphs,chain=chain,conditional_holdout=chain_audit['conditional_holdout'],
        audit=dict(path=str(final_path),sha256=digest(final_path)),
        boundaries=['AIE trace native clock is uncalibrated device cycles; host nanoseconds are a separate axis',
        'Trace captures are partial and cannot calibrate complete compute duration or DDR bandwidth',
        'Trace-on alters placement/output-BO size and observer overhead; primary performance comparisons use off',
        'Chain timing is complete host region wall and cannot be inserted into static compute-only costs',
        'Static FFN sensitivity profiles are not claimed as accurate predictions of these warm runtime measurements'])
    write(root/'hardware-report.json',result)
    lines=['# AIE硬件trace与现有NPUChain验证','',
        '全部NPU运行由唯一owner串行执行，使用现有/tmp/npu.lock；没有新runtime或安装工具链修改。','',
        '## trace-off/on同条件完整图对照','',
        '最终instrumented批次，每项7次热样本。只说明整体tracing扰动；单批顺序、未锁频，不把微小差异当加速。','',
        '| 图 | off ms | on ms | on/off | 提交/图 | off CPU SwiGLU ms |','|---|---:|---:|---:|---:|---:|']
    for r in graphs:
        off,on=r['variants']['off'],r['variants']['on'];lines.append(f"| {r['case']} | {off['median_ms']:.6f} | {on['median_ms']:.6f} | {r['perturbation_ratio']:.3f} | {off['submissions']} | {off['host_api_medians_ms']['cpu_swiglu_ms']:.6f} |")
    lines+=['','数据准备/转换、BO sync、真实start/wait、CPU节点与完整图wall分开保存。wait是主机等待区间，不是独立设备compute时间。编译、加载、首次权重准备和两次排除执行另列。','',
        '原始trace来自AIE硬件；Perfetto文件只映射cycle到可视轴，没有校准CPU/device时钟。最小IRON示例有EVENT0/1，原ORBIT AIR配置了EVENT0/1槽位，但kernel没有产生可观察的scope marker；实际观察到vector/lock请求/port事件，不能建立完整kernel起止区间。真实FFN多个通道达到有限窗口，截断可能性明确，不能用尾部零证明完整或将stall区间求和。','',
        '| 图/投影 | raw容量bytes | 非零words | 解析条目 | 完整性 |','|---|---:|---:|---:|---|']
    for r in graphs:
        for t in r['trace_manifests']:lines.append(f"| {r['case']}/{t['task']} | {t['raw_bytes']} | {t['nonzero_words']} | {t['events']} | partial |")
    lines+=['','## NPUChain同条件对照','','| 形状M/K/H/N | 独立ms | chain ms | 真实提交 |','|---|---:|---:|---:|']
    for r in chain:
        shape='/'.join(str(r['shape'][k]) for k in ('m','k','h','n'));v=r['variants']
        lines.append(f"| {shape} | {v['independent']['median_ms']:.6f} | {v['chain']['median_ms']:.6f} | 2→1 |")
    model=chain_audit['conditional_holdout']
    lines+=['','独立路径复用两个one-op NPUChain、context及权重；打包路径复用一个two-op链。内核前端IR一致，28个热输出全量检查通过，两路径输出逐位相同。BF16中间及最终存储、最后转f32表示显式记录。打包同时减少中间主机读回/staging及context切换，不能全部归因纯launch；中间仍在系统DRAM，一次提交不等于片上融合。', '',
        f"仅用训练形状节省量作条件模型，留出形状预测 {model['predicted_packed_ms']:.6f} ms，实际 {model['observed_packed_ms']:.6f} ms，相对误差 {100*model['absolute_relative_error']:.4f}%。模型还使用留出独立路径的实测wall作为条件输入，不是纯尺寸预测或静态compute-only模型的验证。",'',
        '## 静态模型接入与限制','',
        '经审计的chain-capability-cost.json记录精确区域的实现、布局、身份和完整wall成本，拒绝全FFN的CPU SwiGLU边界，也拒绝向compute_only模型重复灌入包含staging/sync的wall值。当前静态mapper只处理operation级实现，尚不能内生选择opaque packed region。硬件证据用于约束目标能力与单独验证条件系统模型；主静态表仍保留理论敏感性标签。','',
        '早期loop-BF16-store编译失败与第二算子FP32 epilogue timeout均单独保存，最终双方一致改用已有BF16 epilogue，没有静默fallback。raw/解析/Perfetto/host记录及审计路径见hardware-report.json。','']
    (root/'HARDWARE-RESULTS.md').write_text('\n'.join(lines))
    print('hardware report: audited 3 graphs, 7 projection captures and 2 chain shapes')


if __name__=='__main__':main()
