# ORBIT 通用搜索与真实 FFN：第二轮交付

本轮主报告是 `build/search-npu-v2/RESULTS.md`，机器可读证据索引是同目录
`final-report.json`。NPU 原始证据单独保存在
`../aie-toy/results/orbit-search-v2`。第一轮 `build/backend-decoupling`
未被覆盖。目录中的 exploratory/formal/早期 final 批次保留作诊断，最终表
只汇总 final-timing 批次，不能混用历史延迟推导加速。

## 接口与实施

原 `search-joint-neighborhood` 新增 structured 模式，调用独立 Core 中的
共同搜索实现；旧 legacy 路径保留。共享候选物化、合法性与候选身份，
目标可执行性和测量通过 evaluator 分离。原驱动与新模式共享抽取后的
beam 选择逻辑，Python 仅准备数据和启动，不固定枚举候选。
详见 [搜索接口](taskflow-search.md)。

持久 BO、通用 sibling projection 合并、受限设备 K 循环已落实到 runtime。
权重更新会失效绑定，热执行复用真正 XRT BO。BO 位于系统内存；CPU
SwiGLU 和必要转换仍计入图时间。详见 [NPU 实现及限制](taskflow-npu-v2.md)。

## 验收结果

- Core 独立编译与符号/链接检查通过，无 Neura/AIR 后端链接依赖。
- 原有 lit 35/35；14 个公共候选、42 个 seed replay；11 个负面语义检查。
- 搜索协议 16 项（含实际 SIGKILL 恢复），缓存完整性 6 项。
- 新搜索产生 7 个候选进入真实 Neura mapper/scheduler；基线 6653 cycles
  最优。此为估计，不是实体 CGRA 时间。
- 两个真实 FFN 尺寸各执行 9 个搜索候选、两轮展开，全部数值通过。
- NPU 小回归 24/24，设备 K 小回归 12/12；BO 生命周期 27 项，设备 K
  初始化/生命周期 18 项，包含 NaN poison、交替权重和显式更新。

详细回归与证据见 [验证记录](taskflow-validation-v2.md)。

## 性能解读

输入/权重相同，D=2048、I=11008，每配置两批相反顺序，各 7 次完整热图。
下表为合并 14 个样本的中位数，原始样本与两批分别的中位数均有保留。

| 配置 | M=5 ms | M=128 ms | 提交/读回 |
|---|---:|---:|---:|
| 原 staged/separate | 20.611 | 37.805 | 3/3 |
| persistent/separate | 10.729 | 28.705 | 3/3 |
| staged/merge | 19.725 | 36.712 | 2/2 |
| persistent/merge | 10.292 | 27.590 | 2/2 |

权重复用有明显收益，但也改变绑定式 runner 开销，不能把全部差异归为
DMA 节省。投影合并收益较小，M5 逆序批次甚至退化；频率未锁定。
搜索第一轮两尺寸均选 N5632、K5504 为后续父候选，淘汰较慢 M 分块，
再展开 N2816、K2752 和组合。M5 全局最优 N5632 的优势在逆序复验消失；
M128 全局最优仍是无变换基线。故不宣称稳定结构加速。

设备 K 使用真实语义 chunk 循环并分别 padding，小尺寸成功，一组 K
partials 只提交/读取一次。大 K5504 因 AIR schedule 的嵌套循环变换失败，
另一共享 C 双提交方案也编译失败。两种 MRE、IR、构建日志保留在 NPU
结果目录的 `device-k-5504-mre-final`、`device-k-two-launch-final`。
它们没有成功延迟，不使用 CPU fallback 冒充设备执行。

## 干净 shell 复现

使用此机器已有的固定 LLVM/AIE 环境；脚本会重建 Core 和原 ORBIT 工具。
输出目录必须不存在，旧证据不会被覆盖。所有设备访问由 runtime 获取
`/tmp/npu.lock`，不要额外套 `flock`。

```bash
cd /home/shiran/project/orbit
# CPU、协议与真实 mapper/scheduler；不访问 NPU。
bash scripts/run_taskflow_search_v2.sh cpu /home/shiran/project/orbit/build/reproduce-v2-cpu
# 完整搜索、同条件对照、复验、小回归、生命周期、失败 MRE 和报告。
bash scripts/run_taskflow_search_v2.sh all /home/shiran/project/orbit/build/reproduce-v2-all
# 只重新审计已有证据、生成汇总，不访问设备。
source ../aie-toy/scripts/env.sh
python scripts/taskflow_v2_report.py
```

包装脚本已做 shell 语法检查；本轮实际实验通过其调用的各独立 CLI
完成，没有为验证包装再次重复全部硬件实验。新的 all 运行中的 MRE 失败
会保存退出码后继续汇总。已有完整对照分别保存编译、加载、权重准备、
首次执行和热样本；理论传输字节不是硬件计数器。

后续事项：修复/替换可支持大 K 嵌套循环的 schedule，再扩大设备累加范围；
在锁频或更稳定环境下复验小差异；设备激活融合及直接 Taskflow→AIR
仍未实现。本轮不要求复杂 NPU performance model 或完整 Qwen 模型。
