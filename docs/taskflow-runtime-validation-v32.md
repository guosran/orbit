# v32 typed-region runtime 与局部成本校准审查

CPU 侧的身份、区域费用所有权及失败传播测试通过。随后冻结 CLI r3 与执行器完成了 16、32、64 方阵 GEMM+ReLU 的正式 CPU/NPU 编译、运行时检查和板上数值验证：每个实现三个新进程，每进程 3 次 warmup 与 15 次计时调用，全部输出通过独立参考检查。该结果不提供 CP-SAT 的硬件加速比或硬件最优解；真实校准和联合搜索比较仍在推进。

## 同一 assignment 的执行与计价

组合搜索把选中的 parent graph、原始 region IR、checked native plan 和 projection 一起交给 exporter。exporter 使用这份 IR/plan；重新创建一个文字相同的子程序不能替代指针绑定的 source proof。matrix CPU / relu NPU 与 matrix NPU / relu CPU 两种情况均有独立测试。只在 child 没有任何经完整算术、索引、数值 contract 证明的 matrix task 时，已知全局 matrix 请求才投影为 inactive `orbit.parent_implementation`；eligible child 仍保留实际 matrix 实现。父请求和 source numerical contract 保留。

完整测量边界为：

```text
reset
  + Σ CPU region body
  + Σ NPU region (sync_to + Σ configuration submit + Σ configuration wait + sync_from)
  + control + output copy
```

同一区域内的配置串行 `start/wait2`。设备 setup、body、teardown 和完成等待共同包含在实际 wait span 内；不能另加旧 analytical setup/body fee。submit 与 wait 各自累计，循环记账间隙进入 global control，所有 span 与 wall 的差由 control 显式承接。准备 context/BO、输入类型检查、NaN poisoning 和 evidence I/O 在计时外。这里没有把完整调用或阻塞 wait 称为 pure launch/device cycles。

host/device scratch 在配置之间保持设备端状态；整个序列只在首尾同步。CPU 测试使用分开的 host/device 数组：若在第二配置前错误 TO-sync，host 的 NaN 会覆盖第一配置写出的 device scratch，完整输出检查会失败。第二配置失败时不得执行后续配置或输出 readback。

## ABI、投影与未知成本

- public graph 的真实物理 ABI 可能含额外 temporary BO。测试包含 3 个公共参数、5 个实际 BO 的例子，确认 320 B 的真实 ABI，而非只计公共参数。
- `order_only` 只表示无 actual access、非 declared-writable 的 readonly memory-state 身份。它保留原 host root 和 `done_reads` 顺序，copies 为 -1，不创建 DMA；若实际 NPU ABI保留该参数，仍 set_arg，但不 sync。真实读写保持普通 payload ABI。legacy resident 模式的拒绝条件保持。
- typed dtype/shape/bytes、负 storage index、configuration 数量、ordered AIR identity、物理 ABI、runtime-contract/proof 和 ELF SHA 均有失败反例。配置验证在开启设备/分配 BO 之前执行。
- backend 已确认实际 selection 字段与验证 helper 一致：`begin_event`、`end_event_exclusive`、完整 union coverage 及每个 child 的完整 ABI。CPU 检查不能证明安装的 AIR 编译器已经正确实现了这些事件。
- 缺模型、phase、系数 provenance、range、匹配 runtime/contract，或 feature 越出训练范围时，成本保持 unknown。不能当成零或物理 infeasible。
- 多配置区域的 submit / wait 都必须具有显式 `configuration_count` 价格项。5-config 反例验证两个 phase 各自计价、任一遗漏都返回 unknown。application 同时保留配置总数，供 control 模型使用。

局部 source family 排除 tile/mapping/implementation 等决策元数据，保留实际算术和 source numerical contract。测试确认更换 tile/mapping 不生成一行新的“程序价格”，删除 numerical contract 则会改变 family。

## 离线拟合与验收边界

`scripts/calibrate-heterogeneous.py` 分为 `fit` 和 `validate`。输入 index 显式列出 train/holdout、可比较的 `ranking_group`、固定 phase feature sets，以及 features/identity/samples/result 的 path+SHA。可追加 process receipt。训练阶段不打开 holdout evidence；测试用读取拦截验证了这一点。路径别名或 symlink 不能把同一测量同时塞进两个 split。

拟合按 `(device, source_family, implementation set)` 分桶，各 phase 做非负 NNLS；reset/control/copy 独立拟合。系数和训练 min/max range 输出为 C++ v32 cost schema；没有完整 plan-ID 价格表。共线系数仅用于预测，不解释成可辨识的物理 launch、DMA 或 setup 参数。model-manifest 冻结模型、脚本、index、训练证据 SHA、依赖版本及验收阈值。模型、index 或原训练文件变化后，留出验证拒绝继续。

validation 会验证 identity→compiled SHA→manifest SHA→parent/ordered-child graph→source/ELF/runtime-proof 的证据链、producer numeric assertions 和逐 call exclusive phase sum。它不会独立重算 FP64 数值，也不能替代 NPU raw output/reference 审计。

模型冻结前固定的组验收条件是：完整 typed witness `(runtime contract, parent graph, ordered child graphs)` 至少有 3 个有证据的 fresh measurement-Python process，全部 heldout 组成本 known，组相对误差的中位数 ≤20%。每个 process 先取 hot-call median，再按 witness 聚合；重复 process 不凑次数。原 measurement rows 和所有 unknown 都保留，known-only 误差不能冒称完整覆盖率。

Top-1 只在 measured process-median min/max 区间严格分离、模型非 tie 且所有选择满足进程/覆盖条件时判 pass/fail；重叠或不足则 unknown。supervisor PID 被明确记录为 supervisor，不冒称测量 Python PID，也不满足正式 fresh-process gate。没有合格 process receipt 时只输出 diagnostic；该工具不自动宣称 CP speedup。

```sh
PYTHON scripts/calibrate-heterogeneous.py fit --index INDEX.json --output NEW_MODEL_DIR
PYTHON scripts/calibrate-heterogeneous.py validate --index INDEX.json \
  --model-dir MODEL_DIR --output NEW_VALIDATION_DIR
```

示例 index 位于 `/tmp/heterogeneous-calibration-index-example.json`；其中 feature sets 是待预注册的结构示例，不是已验证的硬件模型。`PYTHON` 需要 NumPy/SciPy；本轮离线测试使用 `/tmp/orbit-v29-fit-env/bin/python`。

## 本轮 CPU 证据

| 检查 | 结果 | 日志 |
|---|---:|---|
| typed cost、source family、order-only、5-config | 52 checks PASS | `/tmp/edge-cost-build.log` |
| matrix request 区域投影 | 27 checks PASS | `/tmp/implementation-projection-build.log` |
| typed executor、真实 CPU ctypes ABI、multi-config scratch/失败传播 | 18 tests PASS | `/tmp/typed-executor-cpu-tests.log` |
| train-only NNLS、hash/graph/range、进程聚合与不确定性负例 | 12 tests PASS | `/tmp/local-phase-calibration-tests.log` |

上述测试中的 AIR/ELF receipt 是明确标注的 synthetic CPU fixture，没有执行设备。新 C++ 测试通过 LLVM clang-format 检查。没有使用 root shared build 或运行 Git 操作。

审查时冻结的 executor SHA-256：`49aacf99bacdf9127e447b8a94cc1704da182496a35c06b72f9f59b388efe446`。此后任何源码改动都需要新的 runtime contract、编译产物和板上验证。

64×64 的实际硬件诊断已经给出了不能忽略的数值反证。`/home/shiran/project/orbit-artifact/results/taskflow/full-joint-v32-r1/smoke64-config-single-r1` 使用 A=0.5I、B 为 BF16 gradient。配置 0 返回的完整 32×32 C tile 与 `B[:32,:32] @ B[32:64,:32]` 逐元素完全相等，FP32 max/mean absolute error 均为 0，而不是应有的 `0.5*B[:32,:32]`。这是错误数据分组的强证据；它不是仅由未执行后续配置造成的 Y 不完整，也不是可以用数值容差解释的近似误差。

对应 runtime 中，每个 A/B shim BD 被聚合为一个 2048-element packet，而同一个 core S2MM channel 的 A/B buffer BD 交替消费 1024 elements。观察结果与“一个 B packet 的前半、后半分别被当成 A/B，相乘得到 B0@B1；A0@A1=0”严格一致。这是依据数值和静态 runtime 的原因推断，尚无独立硬件 packet trace。pack/unpack 的 A(4×8)、B(8×8)、C(4×8)索引与已安装 vendor row-major kernel 一致，生成 wrapper 的 DIM_M/K/N 均为 32。

`smoke32-single-r1` 的同一 32 kernel 使用非对易随机 A/B，3/3 板上检查通过；其 `max|A@B-B@A|=0.019563623`，因此不是 scaled-identity 无法区分 A/B 交换的弱测试。但这只覆盖 single-K 数据包，不能证明 64 two-K 的合并 packet 正确。

因此，host 地址序列的 ordered-concat 覆盖是必要条件之一，**不是充分的 runtime 实现证明**。新 codegen/checker 还必须证明 host packet 边界与 core BD/lock-buffer 消费一致，以及 A/B 从不同 shim channel 汇合时的身份和到达顺序。仅保证总字节数、地址集合或地址遍历顺序相同会错报完成。此项尚未验收；修复后须重新运行非对易随机 64 数据和定位 pattern，再完成全配置、完整输出与独立 reference 的上板复验。冻结模型后的 heldout accuracy/ranking 也仍是独立待验收项。不得将 compiler/checker success、CPU tests 或 synthetic 成本最优等同于硬件 oracle 成立。

## 正式冻结执行路径的板上检查点

原始证据位于 `orbit-artifact/results/taskflow/full-joint-v32-r1/preparation-r5`。
工具为 `/tmp/orbit-v32-runtime-cli-frozen-r3/taskflow-opt`；执行器 SHA-256 为
`49aacf99bacdf9127e447b8a94cc1704da182496a35c06b72f9f59b388efe446`。
每次测量保留 compiled、identity、逐 call 独占 phase spans、result、实际 Python
PID/start/end 的 process receipt 和数值输出。冻结工具检查五个配置的最后 AIE IR；
其 SHA 与先前随机、结构化输入验证通过的程序一致。

| 方阵边长 | CPU 各进程中位 ms | NPU 各进程中位 ms | 数值检查 |
|---|---|---|---|
| 16 | 0.002070 / 0.002050 / 0.002080 | 0.207525 / 0.184216 / 0.213385 | 两设备各 54/54 |
| 32 | 0.005170 / 0.005240 / 0.004999 | 0.232645 / 0.222485 / 0.219535 | 两设备各 54/54 |
| 64 | 0.131317 / 0.131407 / 0.223414 | 1.273290 / 1.247850 / 1.248630 | 两设备各 54/54 |

这些是尚未完成性能稳定性验收的原始测量。并行构建期间 CPU 波动明显，
不得把最佳一次或不同负载下的比例叫作 CP-SAT 加速比。当前 NPU 路径的
成本显著高于 CPU；增加规模并不自动解决配置提交和等待开销。
48×48、32×32×32 分块的 NPU 导出被 static selected allocation sum / SRAM
检查拒绝，保留 `export48-npu.log`；不能跳过容量检查或归类成数值通过。

运行时静态证书覆盖 source/graph、事件与 DMA 恰覆盖、ABI、packet 边界、
路由、物理存储和 FIFO。`core_lock_protocol_checked=false` 与
`kernel_arithmetic_checked=false` 保持显式；数值通过只为已测工作负载提供
运行证据，不能推导所有输入的算术证明。
