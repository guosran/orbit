# Taskflow 后端解耦：实现与复现

本轮实现位于 ORBIT `93bfea39833ca88511ed7c7dc4e94bbffb943eee` 之上的工作树。
LLVM/MLIR 固定为 `6146a88f60492b520a36f8f8f3231e15f3cc6082`。
没有修改 LLVM、已安装 AIR/AIE wheel 或原 Qwen 实验权重。

## 可执行入口

从干净 shell 使用本机已有工具链：

```bash
cd /home/shiran/project/orbit
# 独立 core 构建、测试；生成全部候选；CPU 数值/非法输入校验
bash scripts/run_taskflow_backends.sh cpu

# 两后端构建、全部公共候选、CPU/Amoeba/NPU验证和实测排序
bash scripts/run_taskflow_backends.sh all

# 已有候选时，可单独执行以下阶段
bash scripts/run_taskflow_backends.sh amoeba
bash scripts/run_taskflow_backends.sh npu
bash scripts/run_taskflow_backends.sh report
```

默认输出为 `build/backend-decoupling/`。重跑会替换该目录同名结果；要保留旧批次，
所有阶段统一加 `--output /absolute/new-run-directory`。
`LLVM_BUILD`、`AIE_TOY_ROOT` 可覆盖工具链路径。
脚本从 LLVM CMake cache 读取编译器；本机 LLVM 使用 GCC 15.2 C++ ABI，
用系统 GCC 13 编译出的工具曾在 MLIRContext 初始化时崩溃，不能混用。
无需安装 Python bindings 即可构建完整 Amoeba C++ 工具。

独立构建证明命令：

```bash
cmake --fresh -S . -B build-core -G Ninja \
  -DORBIT_ENABLE_NEURA=OFF -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/home/shiran/project/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/shiran/project/llvm-project/build/lib/cmake/llvm \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/home/shiran/bin/g++ \
  -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so -DZLIB_INCLUDE_DIR=/usr/include
cmake --build build-core --target taskflow-opt check-taskflow-core -j2
```

独立 core 的默认完整构建也通过；`build-core/core-isolation.json` 记录实际编译源、
include 路径和链接符号检查，均不依赖 Neura/AIR 后端。

## 架构与语义边界

* `include/Taskflow/StructuredProgram.h`、`lib/Taskflow/StructuredProgram.cpp`：
  受限结构化计划、语义验证和真实图/计算体物化。
* `tools/taskflow-opt/`：独立 core 工具。源 JSON 是受限交换入口，输出是原 Taskflow
  dialect 中带 affine/arith/math 计算体的真实任务图。重新读取 MLIR 时重建 witness
  对应程序，逐操作验证结构等价；修改计算体、切片、存储或依赖都会被拒绝。
* `OutputCoordinateProof`：将 legacy Neura proof 放在后端注册的服务后面。
  原有 replica/join verifier 检查保留；独立 core 未注册服务时明确拒绝需要该 proof
  的旧程序。普通结构化路径不要求 hardware counter、hyperblock 或 replica 属性。
* `scripts/taskflow_amoeba.py` 和 Neura backend 中的结构化准备 pass：
  消费同一候选，做目标侧阶段划分，接入旧编译及 CPU replay。
* `python/orbit_npu/`：消费相同已认证计划，产生独立目标计划，调用由每个任务的
  实际尺寸、切片和 K 区间参数化的矩阵 recipe。没有按 workload 名跳转固定 demo。
* `scripts/taskflow_experiment.py`：确定性枚举、物化、验证、全图实测和反馈排序。
  NPU 分数单位是毫秒；不改写旧 Neura II、scheduler cycles 或模型 provenance。

交换契约详见 [taskflow-backend-contract.md](taskflow-backend-contract.md)。
旧 `JointNeighborhoodActions` 及 Neura 特殊 materializer 仍属于旧后端，公共结构化
变换不调用它们。该版本不尝试将整个旧 Neura 搜索改成通用编译器。

M/N 分块生成多个任务与真实半开访问区间；K 分块在任务内生成按原顺序累加的多个
循环段，只有一次零初始化。顺序 K 变换不重结合公共 FP32 运算顺序。
completion dependency 只排序，不执行矩阵求和。融合将 GEMM 和 ReLU 两任务的计算
体合入一个任务；受限 fission 再恢复为两个依赖任务。它不承诺消除中间存储。
NPU 当前仍将 ReLU 放 CPU，因此不能称为设备融合 kernel。

## 验证与证据

小尺寸使用 3 个随机种子 `0,1,2`，输入含负数、显式零和非对称权重。
GEMM 为 `(5×23)@(23×19)`，GEMV 为 `(1×23)@(23×19)`；M/N tile 为 `3/11`，
K tile 为 `7`，均实际覆盖非整除尾块。小 FFN 为 `M=5,D=16,I=24`。
大 FFN 使用合成权重、seed=0，真实 `D=2048,I=11008`，分别运行 M=5 和 M=128，
没有用 toy 尺寸代替模型维度，也没有运行完整 Qwen 模型。

独立参考直接以 float64 整矩阵公式计算，不读取候选切片或变换历史：

```text
G = X @ W_gate
U = X @ W_up
H = G * sigmoid(G) * U
Y = H @ W_down
```

CPU 候选解释器以 FP32 顺序执行真实 K 切片，14 个小候选 × 3 seeds 全部通过。
11 项集成非法输入测试全部拒绝；独立 core 测试另外覆盖 WAR/WAW、未知字段、
未注册 legacy proof、极大 tile 溢出和带 witness 的 `--verify-only`。
Amoeba 的 LLVM 编译 CPU replay 独立执行真实 MLIR，避免只检验 JSON 解释器。

Amoeba 的 34 项旧 lit 加 1 项新阶段划分测试共 **35/35 通过**，包含 mapper、
production scheduler、trace、cache 和 verifier 负例。14 个公共候选各 3 seeds 的
LLVM CPU replay 全通过，最大绝对误差约 `1.20e-7`；13 个候选成功构造 Neura kernels。
多输出融合候选的旧 Neura lowering 明确不支持，但原公共融合计算体的 CPU replay 通过。

公共 GEMM 的 base 和 M/N 候选还实际经过旧资源枚举、mapper 和 production scheduler：
base 两个目标阶段 II=`[1,3]` cycles、makespan=`6653` cycles；M/N 八个阶段
II=`[1,3,1,3,1,3,1,3]` cycles、makespan=`6662` cycles。资源来自原枚举器的有界策略
`max-cgras-per-task=1`，没有手填合法 shape。这些是 mapper 与调度估计，不是实体
CGRA 实测。其他候选保持 unscored，不从 CPU 或 NPU 时间伪造 Neura II。

身份审计确认 **14 个相同已物化候选被两个后端消费**；计划与 MLIR 的磁盘 SHA、
manifest、NPU 认证产物及 Amoeba 反馈一致。`search-feedback.json` 保存逐项对应。

NPU 公共语义仍为 f32，目标实现显式采用 BF16 输入、BFP16 EBS8 MAC、FP32 累加/输出。
固定验收同时要求：所有元素 `atol=.015, rtol=.06`；RMS(error)/RMS(reference) ≤3%；
max(abs(error))/max(abs(reference)) ≤8%；形状一致且无 NaN/Inf。全零参考要求严格零。
后两项避免小值 FFN 被绝对容差掩盖。保留最大绝对误差及全输出，不只看近零相对误差。
这是记录精度契约下的近似 NPU 实现，不宣称与原 FP32 位级等价。

正式 NPU 验收 16 个结构候选、44 个 candidate/seed、132 个热图全部通过。
各组先首次加载与运行，再额外预热，之后保留全部 3 次完整图样本。
加载阶段通过有版本守卫的生成 launcher 局部 instrumentation 单独测量；没有改安装包。
编译、首次权重准备、加载和完整图热运行分别保留。完整图包括主机节点、
输入/中间值转换、BO staging、XRT wait、读回和输出复制；没有用算子时间相加代替。
不提供未经测量的 device-only 时间。共享桌面机器、未锁频，小差异不能视为加速证明。

| 工作负载 | 基础候选完整图中位数 ms | 最大绝对误差 | 最大归一化 RMS |
|---|---:|---:|---:|
| GEMM | 0.712 | 0.006652 | 1.001% |
| GEMV，M=1 | 0.518 | 0.004336 | 1.308% |
| GEMM→ReLU | 0.543 | 0.006652 | 1.018% |
| 小标准 FFN | 1.346 | 0.001254 | 1.985% |
| 完整 FFN，M=5 | 21.609 | 0.000961 | 1.735% |
| 完整 FFN，M=128 | 39.805 | 0.001123 | 1.717% |

完整原始证据在 `build/backend-decoupling/`：

```text
manifest.json                  全部源/候选身份与物化命令
search-feedback.json           全部状态、毫秒样本、排名、两后端共享身份
RESULTS.md                     16候选结果表
negative.json                  集成拒绝测试
<case>/source.json              原始程序
<case>/inputs-N.npz             每seed输入
<case>/reference-N.npz          独立CPU参考
<case>/<candidate>/candidate.mlir  真实Taskflow和结构化body
<case>/<candidate>/plan.json    版本化语义执行计划
<case>/<candidate>/cpu.json    CPU候选数值检查
<case>/<candidate>/amoeba/      编译命令、后端IR、shared.so、CPU replay和反馈
<case>/<candidate>/npu-seed-N/  目标计划、身份、AIR/AIE/ELF产物、输入输出和计时
```

## 有限搜索与后续优化

同一 GEMM 的基础、M/N 分块、K 分块分别实测约 `0.712 / 2.108 / 2.318 ms`；
每图 NPU dispatch 分别为 `1 / 4 / 4`。K 候选仍是一个逻辑 task，但包含四个顺序
K 计算段，目标执行没有忽略这个结构。GEMM→ReLU 的 task 数依次为基础2、融合1、
恢复拆分2；三个候选都经过实际后端消费。

排序使用全部保留样本的完整图中位数，失败或缺失结果标为 unscored。
当前变换没有性能收益，不能把更多分块称为优化成功。下一步最有价值的是减少
小任务 dispatch/重新配置、让权重 BO 常驻、合并顺序 K 在设备上累加；之后再考虑
驻留布局与更精细候选。没有训练 NPU 预测器，也没有捏造成本常数。

## 支持范围与未完成项

| 能力 | 本轮状态 |
|---|---|
| 独立 Taskflow core，无 Neura/AIR 链接 | 已构建、验证 |
| 静态二维 f32、显式独立存储、结构化生成的 MLIR | 支持，计划与计算体绑定 |
| M/N 图分块、非整除尾块 | 支持 |
| 顺序 K 分块、初始化与依赖验证 | 支持；NPU partial sum 明确在 CPU |
| GEMM→ReLU 融合/受限 fission | 支持公共真实任务体变换；NPU ReLU 仍在 CPU |
| 标准 SwiGLU FFN | 3个GEMM在NPU，SiLU和乘法在CPU |
| 公共 GEMM→原 Neura mapper/scheduler | base、M/N 两候选完成，cycles 单独记录 |
| 多输出融合图→Neura kernel lowering | 旧路径不支持，公共图 CPU replay 成功 |
| Taskflow→直接 AIR frontend | 尚未实现；当前成熟 recipe 路径为 Taskflow→Triton→AIR→AIE→XRT |
| NPU ReLU 融合 kernel | 尝试失败，Peano vector compare legalization，错误保留 |
| 动态形状、任意 MLIR/控制流/别名 | 明确不支持 |
| 并行 K reduction、复杂 replica、通用 fission | 未实现 |
| 实体 Amoeba CGRA 测量 | 未执行；仅编译、回归及 CPU replay 等明确标注证据 |
| Attention、GCN、完整 Qwen 重跑 | 不在本轮范围 |

NPU 具体实现和失败记录见 [taskflow-npu.md](taskflow-npu.md)。
Amoeba 编译、资源策略和回归命令见 [taskflow-amoeba.md](taskflow-amoeba.md)。
