#!/usr/bin/env python3
"""Recheck recorded static plans and summarize them; no hardware execution."""
import argparse
import csv
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from taskflow_static_evaluate import load,write
from orbit_mapping.checker import check_plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve()
    rows=[];comparisons=[];searches=[];checked=0
    for d in sorted((root/'workloads').iterdir()):
        config=load(d/'config.json')
        for variant in ('base','n','k','fused'):
            path=d/variant/'mapping.json'
            if not path.exists():continue
            result=load(path);program=load(d/variant/'plan.json');best=result['best_plan']
            for alt in result['alternatives']:
                if alt['status']=='passed':
                    check=check_plan(program,config,alt['plan'])
                    assert check['passed'],check
                    checked+=1
            row=dict(case=d.name,variant=variant,assignment=best['assignment'],plan_id=best['plan_id'],
                totals=best['totals'],search=result['search'],evidence=str(path))
            rows.append(row)
            if variant=='base':
                for alt in result['alternatives']:
                    if alt['status']!='passed':continue
                    mapping=alt['assignment'];npu=sum('npu' in s for s in mapping.values())
                    if npu in (0,3) or d.name=='gemm_relu':
                        comparisons.append(dict(case=d.name,npu_steps=npu,assignment=mapping,totals=alt['plan']['totals'],plan_id=alt['plan']['plan_id'],evidence=str(path)))
        slow=load(d/'slow-link-mapping.json');slowconfig=load(d/'slow-link-config.json')
        assert check_plan(load(d/'base/plan.json'),slowconfig,slow['best_plan'])['passed']
        comparisons.append(dict(case=d.name,profile='slow-link',assignment=slow['best_plan']['assignment'],totals=slow['best_plan']['totals'],evidence=str(d/'slow-link-mapping.json')))
        search=load(root/'search'/d.name/'search.json')
        archive=[]
        for r in search['archive']:
            if r['status']=='passed':
                feedback=r['evaluation'];assert feedback['evaluation_kind']=='static_estimate' and feedback['executed'] is False
                evidence=load(feedback['evidence_path']);best=evidence['best_plan']
                assert check_plan(r['program'],config,best)['passed']
                assert r['score']==best['makespan_ms'] and feedback['plan_id']==best['plan_id'];checked+=1
            archive.append({k:r[k] for k in ('round','parent','action','candidate_id','search_key','status','score')})
        searches.append(dict(case=d.name,evaluations=search['evaluations'],selections=search['selections'],ranking=search['ranking'],archive=archive,evidence=str(root/'search'/d.name/'search.json')))
    result=dict(schema='orbit.static.report.v1',executed=False,cost_provenance='theoretical_sensitivity_not_hardware_measurement',
        independently_rechecked_plans=checked,workloads=rows,comparisons=comparisons,searches=searches,
        validation=load(root/'validation/summary.json'))
    write(root/'static-report.json',result)
    lines=['# 静态异构 mapping 与通信建模','',
        '**以下毫秒全部是明确假设参数下的静态预测，不是硬件测量。** CPU 100 GFLOP/s、NPU 1000 GFLOP/s、拷贝12 GB/s仅是敏感性场景。旧完整图实测没有被拆作纯计算成本。','',
        '单次冷图；初始数据仅在 host。任务成本包含内部精度转换/padding/本地访存；跨位置 copy、布局转换和可见性 sync 单独计费。NPU workspace512 MiB 是本轮源程序范围内的假设预留，非已验证实际峰值。','',
        '| 图/结构 | 最优放置 | 计算 ms | copy ms | convert/sync ms | 总时间 ms | host/shared峰值 MiB |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        t=r['totals'];count=sum('npu' in s for s in r['assignment'].values())
        lines.append(f"| {r['case']}/{r['variant']} | NPU {count}步，其余CPU | {t['compute_ms']:.6f} | {t['copy_ms']:.6f} | {t['convert_ms']:.6f}/{t['sync_ms']:.6f} | {t['makespan_ms']:.6f} | {t['peak_memory_bytes']['host']/2**20:.2f}/{t['peak_memory_bytes']['shared']/2**20:.2f} |")
    lines+=['','## 同图完整方案比较','','| 图 | NPU步骤或场景 | 完整计划 ms | 计算 ms | 通信 ms |','|---|---|---:|---:|---:|']
    for c in comparisons:
        t=c['totals'];label=c.get('profile',str(c.get('npu_steps')))
        lines.append(f"| {c['case']} | {label} | {t['makespan_ms']:.6f} | {t['compute_ms']:.6f} | {t['copy_ms']+t['sync_ms']+t['convert_ms']:.6f} |")
    lines+=['','在M5场景中，孤立的NPU计算更快，但权重/中间数据搬运使整图选择CPU。M128在12 GB/s假设下选择三个NPU GEMM；把链路改为1 GB/s后选择CPU。K分块排除不支持部分和累加的NPU overwrite实现，因此M128的最佳映射及成本改变。','',
        '受限算法枚举完整放置（预算不足时有界邻域搜索），每个方案重建通信并运行独立检查器。调度为确定性list scheduling，不宣称全局最优。整个图生存期保留数据副本，空间估计保守。','',
        '## 原ORBIT两轮搜索','']
    for s in searches:
        lines += [f"### {s['case']}：{s['evaluations']}次评价",'', '| 轮次 | 动作 | 状态 | 预计ms |','|---:|---|---|---:|']
        for r in s['archive']:lines.append(f"| {r['round']} | {r['action']} | {r['status']} | {r['score'] if r['score'] is not None else '未评分'} |")
    # A tabular predicted schedule, deliberately not called hardware trace.
    plan=load(root/'workloads/ffn_m128/base/mapping.json')['best_plan']
    lines+=['','## M128基线方案的预计时间表','','下列为编译期预测时序，不是AIE硬件trace。','','| 步骤 | 类别 | 资源 | 起始ms | 结束ms | 理论字节 |','|---|---|---|---:|---:|---:|']
    with (root/'predicted-schedule.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['event','kind','resources','start_ms','end_ms','theoretical_bytes'])
        for e in sorted(plan['events'],key=lambda e:(e['start_ms'],e['id'])):
            fields=[e['id'],e['kind'],','.join(e['resource_demands']),e['start_ms'],e['end_ms'],e.get('bytes',0)]
            writer.writerow(fields);lines.append(f"| {fields[0]} | {fields[1]} | {fields[2]} | {fields[3]:.6f} | {fields[4]:.6f} | {fields[5]} |")
    lines+=['','所有父候选、候选身份、计划身份、成本配置、未选方案及失败原因见 static-report.json 与各 mapping.json/search.json。共享内存sync、layout转换等非零场景单独见 validation/synthetic。实体Amoeba cycles仍隔离评价，没有跨单位混比。','']
    (root/'STATIC-RESULTS.md').write_text('\n'.join(lines))
    print(f'Independently rechecked {checked} recorded plans')


if __name__=='__main__':main()
