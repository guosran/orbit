# r10 MLP2 上板与兼容性验收

同一 16×16、两层 shared-weight MLP 的三个固定方案共完成 **324 calls，原 BF16 RNE reference 数值检查全部通过**。whole-program NPU 的完整调用中位耗时为 0.200225 ms，相对 phase-local NPU 的 0.798921 ms 为 **3.990116×**。CPU 为 0.003410 ms，仍约快 58.72 倍。这是指定执行方案的实测，**不是 CP-SAT 选出的加速比或最优解证明**。

原始证据：[measurement-summary.json](/home/shiran/project/orbit-artifact/results/taskflow/full-joint-v32-r1/preparation-r10/measurement-summary.json)，其余文件位于同一 `preparation-r10` 目录。此次只读核对了九个 process 的 `result.json`：各 36 calls、所有 checks 为 true，与汇总 324 一致；没有改写旧证据或重新运行设备。

| 固定方案 | 物理 NPU configs | 三进程 hot median（ms） | 进程中位数的中位数（ms） |
|---|---:|---|---:|
| NPU phase-local | 4 | 0.763631 / 0.826860 / 0.798921 | 0.798921 |
| NPU whole-program | 1 | 0.200225 / 0.213235 / 0.188245 | 0.200225 |
| CPU | — | 0.003420 / 0.003370 / 0.003410 | 0.003410 |

NPU 两方案各是一个 typed region，physical config 数量为 4 与 1，不能误写成四个异构区域。源码函数 body 逐字相同；whole-program 仅在 module header 增加 residency/data-organization mapping 元数据，CPU 使用 phase-local 原 source。源文件完整 SHA 因 mapping header 不同而不同，`cases.json` 如实保留。

## 边界与身份

[measurement-policy.json](/home/shiran/project/orbit-artifact/results/taskflow/full-joint-v32-r1/preparation-r10/measurement-policy.json) 固定 CPU0 affinity，每方案三个 fresh Python processes；每 process 5 warmups、31 hot calls。每方案 108 calls，其中 93 个 hot calls；三方案合计 324。三个方案按 process round 轮换，编译在测量前完成。

测量完整 prepared host-to-host 调用：input reset + exact regions（sync/submit/wait/readback）+ control + output copy。context/BO 准备不在边界内。同一区域的 configs 串行 native wait2；不能把该完整时间或减少的时间全部称为 pure launch。4→1 配置和中间数据保留是方案差异，现有测量未独立分解每种机制的因果收益。

冻结工具 SHA：`0812d28cc6ca271a64e881a76d4f0edc9d0f35fec3b3d5312e308d3484baca03`；三方案 manifest 的 execution contract identity 均为 `d7147f9b2ec7f1c18f10638af62f6f0951c8427f9089d724bad3eb5343506dd7`。原输入 SHA 为 `6a32e737f802ea1539a8e65ce877e606f5bb201a5ac8eabb4321341dd7e7938e`，原 RNE reference SHA 为 `6f7e5466b881f050465aa9012bb47328dd0c29f8e6b0c7aacd133838fc08c0ba`。r10 numerical 副本已独立计算匹配；保留原容差。

## 修复与结论范围

此前 MLP2 三次调用的旧输出均偏离 RNE 参考；对本次 ReLU-positive 数据，它们恰好等于 RTZ/floor 诊断参考。ORBIT-owned integer BF16 RNE lowering 修复了转换语义，runtime generic compatibility 仅规范化 pinned MLIR 不支持的 `trunci overflow<none>` 默认 property。原 source 算术、参考和 vendor 未改。详见 [BF16 修复报告](taskflow-bf16-rne-v32.md)。

兼容 receipt 记录原输入和 normalized SHA-256、默认属性移除数，最终 runtime proof 另绑定 realized IR。phase-local 的四配置移除数为 0/2/0/0；whole-program 单配置为 2。独立 CPU 正负例、ON/OFF target 与真实 MLP generic/AIE 验证已通过。

本结果证明此固定 MLP2 的正确执行和上述 NPU 方案对比收益。它没有运行全维联合 CP 选择，没有证明全抽象域最优，也没有证明任意层数、形状或数值输入都支持。CPU 是本组三方案中最快的方案；相对 CPU，whole-program NPU 的比值为 0.017031×，不能替换基线来宣称异构分配获益。r9 校准属于旧合同，其价格不被自动挪为 r10 已校准模型。后续 CP/ORBIT 优化效果须另外验证相同 workload、固定 baseline 和同一 assignment 的模型/执行身份。
