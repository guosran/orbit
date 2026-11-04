# 原生 C++ 静态规划

静态生产路径使用 `MLIRTaskflowStructured`，与独立 Core 一起构建，不链接
Neura、AIR 或 XRT。`backend=static` 且未指定 evaluator（或 `evaluator=native`）
时，structured 搜索直接调用 C++ evaluator，不启动 Python。

## 调用路径与职责

迁移前：`runStructuredSearch` → 外部 `taskflow_static_evaluate.py` → Python
`contract` → `mapper` → `communication` → scheduler → `checker`。

迁移后：

```text
Taskflow MLIR
  → StaticAnalysis（实际 task body、SSA、affine 访问和依赖）
  → StaticContract（类型化硬件、实现、成本）
  → StaticMapper（完整放置候选）
  → StaticCommunication（区域版本、副本、搬运、转换、可见性）
  → scheduleStaticPlan（资源日历、生命周期及容量）
  → StaticChecker（生产计划合法性检查）
  → JSON 计划、诊断与预计 makespan
```

共享类型定义在 `include/Taskflow/StaticMapping.h`。分析步骤保留实际 MLIR
operation 指针；导出使用稳定任务、操作和存储身份。JSON 是配置和跨进程记录，
不是另一份绕过 body 的计算语义。受限格式的 witness 用于来源身份和一致性
核验，实际分析必须验证循环边界、访存索引、算术结构和依赖。

`StaticEvaluation.cpp` 为原搜索提供原生入口。每个结构候选物化成 Taskflow
MLIR 后重新分析和规划。原 Neura legacy 搜索不改变；显式外部测量 evaluator
仍可使用。静态结果单位 ms，标记 `static_estimate`、`executed=false`。

## 直接复现

完整复现入口会重建 Core/原 ORBIT 工具、从源生成四个工作负载，运行差分、
原生搜索、进程跟踪与回归，不依赖 v3 的原始硬件资产：

```bash
cd /home/shiran/project/orbit
bash ../orbit-artifact/taskflow/scripts/run_taskflow_static_cpp.sh "$PWD/build/my-static-cpp-run"
```

已有静态夹具时，也可直接调用 C++：

```bash
cd /home/shiran/project/orbit
cmake --build build-core --target taskflow-opt -j2
mkdir -p build/my-native-static
build-core/tools/taskflow-opt/taskflow-opt \
  build/static-mapping-v3/workloads/ffn_m128/base/taskflow.mlir \
  --static-map \
  --target-config=build/static-mapping-v3/workloads/ffn_m128/config.json \
  --mapping-output=build/my-native-static/mapping.json \
  --analysis-output=build/my-native-static/analysis.json

build-core/tools/taskflow-opt/taskflow-opt \
  build/static-mapping-v3/workloads/gemm_relu/base/taskflow.mlir \
  --search --backend=static \
  --target-config=build/static-mapping-v3/workloads/gemm_relu/config.json \
  --search-output=build/my-native-static/search \
  --max-rounds=2 --max-candidates=9 --beam-width=2
```

`--check-mapping=PLAN` 检查裸 `best_plan` 对象，并将检查结果写入
`--mapping-output`。失败返回非零状态。原 ORBIT 驱动仍使用
`--search-joint-neighborhood='decision-flow=structured backend=static ...'`，
无需提供 evaluator。`taskflow_static_experiment.py` 仅调用 C++。

## 当前 Python 边界与历史报告

v5 开始时实际存在 `python/orbit_mapping` 和旧 evaluator，它们是在 v4 后续工作中
恢复的。因此 `build/remove-python-reference/full/RESULTS.md` 的“已删除”和
v4 `COMPLETED-RESULTS.md` 的“保留”描述的是不同时点，不能视作当前状态证明。
本轮先保存实际代码快照及 SHA256 到 `build/direct-air-v5/audit/retired-python-reference`，
再将重复 Python mapping、通信与语义 checker 移出活动代码；未恢复任何已删除实现。
跨语言差分结果保留为历史证据，当前测试不再导入 Python mapper。
Python 仅负责实验编排、独立数值参考、工具链调用与测试。

`GenericAnalysis` 现在直接分析原生函数参数、alloc 与实际 body，无需 `orbit.plan`；
`GenericMapping` 为直接 AIR 提供保守驻留/传输/顺序执行计划。
旧 `StaticAnalysis` 固定模式识别保留在旧结构接口及专用已标定实现路径，
不是通用分析或直接 AIR 的入口。详见 [直接 AIR](taskflow-direct-air-lowering.md)。

原生计划使用 `identity_scheme=llvm-json-sorted-17g-sha256`。原生搜索绑定配置及其
引用文件、实现版本、工具二进制和搜索预算；恢复时再次执行 C++ 评价核验已有分数。
原生路径没有复用 Python evaluator 的磁盘评分缓存。

迁移不改变旧理论配置的来源。CPU/NPU 吞吐、链路带宽和 workspace 参数仍是显式
敏感性假设。新增 `StaticRegion.cpp` 与独立 `StaticRegionChecker.cpp` 支持整段实现
精确覆盖及热执行 host-to-host 实测 wall 成本；内部通信不重复收费。
详见 [整段契约](taskflow-static-regions-v4.md) 和
[标定、留出与真实执行](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-region-calibration-v4.md)。
标定范围外（包括当前 CPU 激活和真实 FFN 整图成本）明确返回 unknown/unsupported。

迁移证据在 `build/static-mapping-v4/cpp-migration`，最终回归在
`build/static-mapping-v4/complete-regression`，独立差分在
`build/static-mapping-v4/reference-differential-final`。直接运行与原生搜索的
`strace -f -e trace=process` 记录用于核对没有 Python 决策子进程。
