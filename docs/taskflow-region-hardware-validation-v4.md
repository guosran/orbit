# v4 同批原生区域计划硬件验收

最终性能比较只采用
`aie-toy/results/orbit-static-v4-hardware/native-plan-comparison-final`。
基线 chain、N=256 分块、M=128 混合区域使用同一输入、权重、精度、热准备状态，
同一批九次热执行交错次序。每个条件另外有两次排除执行。未锁定设备频率。
旧 `structural-n256` 单独运行只保留为独立预测证据，不参与这张性能排名。

| 方案 | 预计 ms | 实测中位 ms | 实测 IQR ms | 实测范围 ms | 实际 start/wait |
|---|---:|---:|---|---|---:|
| baseline chain | 0.629754 | 0.678584 | 0.624196–0.708423 | 0.608985–0.787012 | 1/1 |
| N=256 | 1.636019 | 1.988844 | 1.828647–2.022473 | 1.775318–2.069321 | 3/3 |
| M=128 混合 | 1.723164 | 1.884186 | 1.726800–1.908305 | 1.609462–2.036863 | 3/3 |

三个方案的 27 份已保存热执行输出逐元素相同，全部通过独立 float64 源程序
回放参考及既有容差。精度为 BF16 输入、BFP16 EBS8 MAC、FP32 累加、各 GEMM
BF16 输出；host 边界转换为 f32。完整边界包含临时分配、激活切片/转换、区域
组装、提交、等待与最终复制。pair 的内部 H 留在同次 NPU 程序中，外部消费者
不与该 H 区域重叠时，无需逐操作读回。审计检查这一条件，不强迫每个源操作
拥有独立提交事件。

模型选对全局最佳基线，本批全局 regret 为 0。模型认为 N 优于 M，实测中位数
相反；只在两个结构候选中选择时，局部 regret 为 0.104658 ms。两者 IQR 与范围
重叠，且未锁频，不能将局部排序当作稳定的硬件结论。两个结构候选都显著慢于
本批基线，这一退化完整保留。

## 独立审计范围

`test/TaskflowMapping/structural_hardware_v4.py` 核对：

- native source/config/plan 文件身份与 candidate/plan IDs；
- 每个 fill 和 GEMM 的 exact cover、重复/遗漏、依赖与实际区域执行顺序；
- single/pair mode、f32 row-major 外部布局及明确精度合同；
- pair 内部 H 的访问区域和外部消费者；
- 七套 prepared 权重的原始片段、BF16 内容哈希、对象身份和 generation BO key；
- 每个热样本的真实 start/wait、实际 region IDs、允许的 readback 索引；
- runner/context/kernel/BO 在九次样本中的身份稳定性；
- 27 份完整热输出、三份独立 A/B/A 权重切换输出，以及方案间输出差异。

不同 N 权重片段的 runner key 包含 mode、M/K/N、权重名称及 N 起点。M 切片的
行偏移只改变激活，同权重复用合法。运行时 `static_indices` 本身只依赖
`runner._bos[bo_key]` 的首次调用，不自动检测权重对象或内容更新。测试层明确
以 immutable prepared 权重内容生成 BO key；改变权重必须建立新 generation。
同一 runner 的 A/B/A 测试验证 A 的 BO 可复用，B 使用不同 BO，三次输出正确。
这不证明原地修改 immutable 权重能够自动失效；不提供这一支持声明。

热样本前的两次排除执行只保留运行时数值诊断，未保存输出文件供此审计重算。
独立输出统计严格为 27 份热输出加三份权重切换输出。

## 复现审计

```bash
cd /home/shiran/project/orbit
source /home/shiran/project/aie-toy/scripts/env.sh
python test/TaskflowMapping/structural_hardware_v4.py \
  --evidence /home/shiran/project/aie-toy/results/orbit-static-v4-hardware/native-plan-comparison-final \
  --output build/static-mapping-v4/validation/native-plan-comparison-final-audit-replay.json
```

审计不打开硬件。最终报告为
`build/static-mapping-v4/validation/native-plan-comparison-final-audit.json`。
四种篡改（执行 region ID、BO generation、pair readback、权重内容哈希）均被拒绝，
记录位于 `validation/structural-audit-negative-checks.json`。它们只修改临时目录中的
结果副本，原始硬件证据未改动。
