# 公共数据组织与联合搜索：实现和验收

最终源码在独立工作树 `/home/shiran/project/orbit-data-v8` 完成构建及验证，再按文件集成回 `/home/shiran/project/orbit`。原有未提交实现、被删除的 Python 规划器、v6/v7 结果和失败记录全部保留。生产语义、分析、候选变换、计划、检查与 AIR lowering 使用 C++/MLIR；Python 用于测试、独立参考、报告和现有工具链/XRT 执行编排。

## 完整程序的新联合优化机会

一个候选包含真实 Taskflow 程序、结构变换、资源/实现配置、数据组织和调度。原 `runStructuredSearch` 及与 Neura 共用的 `selectSearchBeam` 继续负责 archive/beam；新增公共驻留策略 `phase-local` / `retain-across-users` 与 M/N/K tiling 组合，没有另建搜索器。

分析从实际访存发现共享区域、版本与循环不变性。搜索选择分块和副本的保留范围；规划推导创建传输、所有使用者及释放条件。公共层复用 Static memory/resource/link，AIR 后端将受检选择落实为 readonly panel、execution group、pool、DMA 与 FIFO。当前全任务共享副本在合成目标上验证；AIR 支持同一执行组内可证明的只读 panel，不支持任意跨 stage SRAM 共享。

具体类型、有限支持范围和检查器见 [公共设计](taskflow-data-organization-v8.md)。原 Neura 的 shape、调度、replica、fusion/fission 能力保留；本轮没有将所有旧路径替换为统一自动异构放置。

## 两个实际计算输入

`taskflow-data-organization-test` 解析 A 产生 X、B/C 读取 X 的真实 MLIR，并从实际 body 分辨 C 对 B 的纯排序依赖。合成 DRAM/SRAM 目标下，两份独立副本为 128 B 搬运，共享副本为 64 B；共享创建只计一次，两个消费者共同延长寿命。两方案 SRAM 事件峰值均为 192 B，DRAM 为 128 B；共享不必然降低峰值。显式 synthetic 参数下整图成本分别 5.2/4.1 ms，不能当硬件测量。最终 **39 项检查通过**，包括版本/区域/创建依赖、释放、容量、越界索引、整存储字节溢出、费用归属和结构变化后的重新分析。

第二个输入是纯计算和任务依赖的 M32/K128/N64 GEMM→scale，含输入、临时 C 和完整输出的总张量为 65536 B。它不包含 channel、FIFO 或驻留指令。通用访问分析发现 A/B 可复用关系，原搜索生成两轮真实候选：

| 候选 | M/N/K | 数据组织 | 实际池峰值 B | B 搬运 B | 整图计划 DMA B | 理论成本 ms |
|---|---|---|---:|---:|---:|---:|
| 初始程序 | 8/64/32 | phase-local | 11264 | 131072 | 172032 | 0.6435392 |
| 第 1 轮中间候选 | 8/16/32 | phase-local | 3584 | 131072 | 221184 | 0.6484544 |
| 第 2 轮联合候选 | 8/16/32 | retain B across users | 9728 | 32768 | 122880 | 0.6386240 |

N64 保留完整 K×N 的 B panel 单独就需 32768 B，被原 24576 B payload 预算拒绝。先减小 N，才能延长 B 驻留；它增加本地占用、减少 B 重读，并改变 A 重读和完整图成本。中间候选按当前模型更差，beam 保留它后第二轮得到更优方案。实验得到 5 个合法候选、1 个容量拒绝，记录父候选/动作链/选择原因，并逐一生成和独立检查 AIR。

配置使用 beam=2 和一个结构多样性保留槽。相同预算、相同成本的 diversity=0 对照也找到该分支，因此本例证明机制与两步机会，**不证明多样性选项带来额外搜索收益**。

成本是显式未标定敏感性模型：每完整图 0.1 ms 固定项、每标量操作 1e-6 ms、每计划搬运字节 1e-7 ms，串行相加。未覆盖每个执行组的配置、实际等待、host 准备或硬件带宽竞争，不借用旧 Triton/chain 标定。

## 身份、缓存与恢复

候选 ID 绑定规范化的真实 MLIR，旧 readonly-panel 拼写与公共 retain 请求得到相同 IR/计划身份。评价 ID 另绑定目标配置、模型版本、编译器和文件依赖；相同计算图换目标/模型会保留候选 ID、改变评价 ID。

恢复重新规划并核对分数和计划 ID，检查 archive 唯一性、源 baseline、预算与轮次、父候选、动作重放、已完成轮次的 proposal 覆盖、重新计算的 beam，以及当前轮中途保存的合法前缀。测试包含合法轮内恢复和多种不一致检查点拒绝。这是可复现性与一致性检查，不是对可任意修改全部文件的恶意环境提供认证。

## 新版本 NPU 对照

`build/data-organization-v8/acceptance-r1` 是实际完整 `all` 入口，退出码 0。同 N16 计算的两个候选各 3 seeds ×（2 warmups + 3 hot），**30 个完整输出通过**，FP32 和 `atol=rtol=1e-5` 均未改变，最大绝对误差 7.152557e-7。

| 策略 | 热中位数 ms | runtime stage 数 | 计划 DMA B | 实际本地 payload 峰值 B |
|---|---:|---:|---:|---:|
| phase-local | 18.658558 | 2 | 221184 | 3584 |
| retain-across-users | 18.650719 | 5 | 122880 | 9728 |

测量按 forward/reverse 交错顺序执行。差异很小、频率未锁定，不能宣称加速。每阶段最终 runtime BD 长度×重复次数和本地池审计均通过；字节是计划及生成代码的一致性结果，不是实测 DDR 流量。审计未完全证明下游编译器所有 DMA offset/stride、任意端口行为或真实物理流量，边界见 [AIR 审计](taskflow-air-audit-boundary-v7.md)。

计时包含 runner 输入 staging、BO sync、提交/等待、输出回读/复制；不含 context 创建、NumPy 准备、数值检查及最终 NPZ 序列化。等待中进度记录的开销仍在该边界。编译、context load 与逐 API 原始事件另存；aircc 缓存命中未独立测量，不能把其子进程耗时解释为冷编译性能。

## FFN 验收与版本边界

原 v7 M128 超时的诊断是四个串行单核心阶段的标量工作量与约 60 秒单命令 watchdog 冲突：M32 Gate 48.359 s，M128 Gate 实际 63.786 s 后 `ERT_CMD_STATE_TIMEOUT`；设备快照无法证明每条指令的进度。修复采用 ORBIT 受检的整程序 independent-M fission，16 次完整 M8 FFN 调用，所有算术仍在 NPU。每个调用保存原 K 累加顺序，输出区域不重叠且覆盖完整 Y。

原 M128 三个 seed 的完整 Y[128,2048] 均通过：581.584170433 / 581.617078537 / 581.599105110 s，最大绝对误差 2.142041922e-8。每 seed 是零 warmup、一次完整长验收，不是热分布。原包 compiler `3e041bf5…`、后续冻结工具 `2444a3b5…` 与各 ELF/计划的身份分别保留，不能混称同一版本。每调用 payload 20480 B、stack 4096 B；权重 BO 跨调用保留属于系统内存复用。

本轮新工具另做完整 M5/D2048/I11008 FFN：3 seeds、15 个输出全部通过，最大绝对误差 2.421438694e-8；各 seed 热中位数 23035.843445 / 23035.459701 / 23033.900633 ms。本地实际峰值 18944 B，stack 4096 B。最大相对误差（仅参考绝对值≥1e-6）为 0.0041073，归一化 L2 最大 2.26269e-6；相对量是诊断指标，原有限值及绝对/相对联合容差不变。

M128 原结果见 `/home/shiran/project/orbit/build/direct-air-v7/RESULTS.md` 和原硬件目录；新版本另重新生成并检查 M128 16 调用计划，**未冒充另一次新版本 M128 上机**。原 M5、非整除 GEMM、64 KiB add→scale 和 exp/SiLU/SwiGLU 冻结回归均通过。近似 exp 的支持区间与溢出/下溢诊断保持原记录，见 [v7 数值验证](taskflow-direct-air-v7-validation.md)；不能从小随机输入或普通区间测试推导普遍 IEEE exp 正确性。

## CGRA ML 与回归

实际原 ML 推理→catalogue→生产 Analytical scorer 路径完成 8 个形状候选的公共合约回归；去掉新增 contract 后，原分数、schedule 和 shortlist 与已保存 pre-adapter golden 完全一致。模型未训练或替换。预测标签为 II，公共接口发布原调度器已有的 scheduled task duration（cycles）；CGRA block 与 PE 形状分别标单位。catalog SHA 来自实际解析同一 buffer，通信费用显式排除，未验证的新存储供数条件保持 false。Exact/profile provider 尚未适配。

该既有小 fixture 的 trip_count=1，此测试验证字段投影、版本绑定和原调度行为保留，不证明多 trip 的 ML 精度。公共合约另有 cycles→ms 正例及未知/重复通信计费拒绝。详细标签、特征和计时边界见 [模型审计](taskflow-cgra-model-boundary-v8.md)。

最终 `acceptance-final-cpu-r1` 入口退出码 0：独立 Core、35/35 Amoeba lit、旧 static/通信/搜索/cache、Generic 分析、旧直接 AIR/分块/channel、public mapping 36、共享副本 39、partition 56、两轮搜索和真实模型合约全部通过。Core 独立链接检查未发现 AIR/AIE/XRT/Neura 依赖。

最后增加的检查器保护经完整 CPU 重跑；5 个候选的 Taskflow/计划/AIR 和 M5 的 Taskflow/facts/计划/AIR 与已上机包逐字节一致，记录于 `final-search-hardware-equivalence.json`、`final-m5-hardware-equivalence.json`。硬件保留其原始编译器/ELF 身份，没有重新标记为最后编译器的额外设备执行。

## 复现与证据

在本机已安装 LLVM/AIR/AIE/XRT、现有 Python venv 和原 CGRA 模型目录的干净 shell：

```bash
cd /home/shiran/project/orbit
bash scripts/run_taskflow_data_organization_v8.sh cpu /tmp/orbit-data-v8-cpu-NEW
bash scripts/run_taskflow_data_organization_v8.sh all /tmp/orbit-data-v8-all-NEW
```

输出目录必须不存在。`cpu` 完整重建并验证；`all` 追加双候选和完整 M5 上机。模型文件默认来自既有 `/home/shiran/project/aie-toy/vendor/orbit-artifact/reference/input0-neighborhood/models`，缺失或哈希变化明确失败，不以合成值替代。原 M128 长验收入口仍是 `run_taskflow_program_mapping_v7.sh all NEW_DIRECTORY`，可能耗时较长。

本机完整证据：`/home/shiran/project/orbit-data-v8/build/data-organization-v8/`。含初始差异、隔离身份、全部命令和失败尝试、最终 CPU 结果、搜索 archive、AIR/AIE/LLVM/linker/runtime 产物、原始样本、独立数值审计、冻结工具和后续等价性检查。原 v6/v7 目录未覆盖。源码提交不包含生成二进制、大张量或硬件输出。

[Dato 对照](taskflow-dato-overlap-v7.md) 保留论文 §3/4/6 与源码确认内容的区别、固定 revision、用户已给定的 stream/layout/grid 与编译器自动 bundle/chain 映射搜索。ORBIT 的公共计划主要由计算/访存分析和联合搜索生成；Dato graph-chain 与 XRT NPUChain 名称和语义分开。
