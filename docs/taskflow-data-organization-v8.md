# 公共数据组织与副本计划（v8）

本轮在原 Taskflow 分析、mapping 与联合搜索中加入公共数据组织选择。计算输入仍是实际 MLIR task body、循环、访存和依赖；JSON 是目标配置、候选选择与结果报告，不承载另一套计算图。原 v7 FFN 运行产物保留其原二进制、计划和日志身份，本轮源码与构建位于独立工作树。

## 既有类型与公共接口

`ProgramMapping.h` 复用 `StaticMapping.h` 的 `StaticMemory`、`StaticResource`、`StaticTransfer` 目标契约；没有给 SRAM/DRAM、链路或计算资源另起同义类型。`GenericAnalysis` 提供实际任务、存储、访问、依赖和 body fingerprint。`ProgramCommunicationPlan` 是两种适配器共用的区域、版本、分配、副本、事件和消费边表。

| 公共对象 | 含义 | 既有接口对应 |
|---|---|---|
| `MappingRegion` | 原存储的精确逻辑区域，任意 rank | Static 的区域 cell；Generic 实际访存 |
| `MappingVersion` | 某区域一次任务实例完成后的值 | Static version；不将每次标量 store 都展开成节点 |
| `MappingAllocation` | 某 memory 中的物理存储对象及大小 | Static allocation；native target pool |
| `MappingCopy` | 区域版本在 allocation 中的有效表示 | Static location copy；含创建事件、全部用户、最后用户和释放界限 |
| `MappingDataEdge` | 值生产事件到实际读取事件，绑定 region/version/copy 和消费要求 | 实际 load 的数据依赖 |
| `MappingOrderEdge` | 不要求消费值的 source task 顺序 | Taskflow 原依赖中没有实际值消费的部分 |
| `MappingEvent` | compute/initialize/transfer，读写 copy、依赖及资源 | Static event；native phase 实例 |
| `MappingTaskPlacement` | 任务到 resource/memory/implementation 的候选赋值 | 公共 whole-task CPU 适配器；native 另有受检 execution groups |

版本的生产事件与副本的创建事件不同。一次 DMA 保留版本身份，创建一个新位置副本；它不成为该值的逻辑计算生产者。多个数据边可引用同一 copy，其 allocation、创建传输和字节只计一次。`consumerEdges` 记录逻辑消费边，`users` 还包括搬运读取；释放必须覆盖全部实际使用者。

Neura 的 `TaskEdgeContract` 有 Value/RAW/WAR/WAW/Control 与 Taskflow/MemoryOrder/Control 来源。公共 Core 不包含或链接该后端头。实际值读取可产生公共 data edge；纯 memory-order RAW 以及 WAR/WAW/Control 不因此产生副本或传输。本轮 Generic 适配器从实际读取与 source task 依赖生成边，未实现通用 Neura edge importer。

## 事实、选择与机制

事实来自 `GenericAnalysis`：访问哪些存储和区域、哪些循环轴不改变读区域、写区域是否独立、源 task 依赖和源 body 身份。数据组织候选使用同一个 `ProgramDataPolicy`：`phase-local` 或 `retain-across-users`。任务放置、tile 尺寸、驻留范围和执行分组是候选选择。AIR 的 DMA、FIFO、memory space 与 pool 则是目标实现机制。

native adapter 将有限的 retain 选择实现为已证明的 readonly panel scope，循环重排、完整 K panel 的读区域、容量和分组均显式进入计划。旧 `residency=readonly-panel` 与新公共 retain 请求经同一个规范化入口得到相同属性；矛盾组合拒绝。该实现保留同一 execution group 内跨 phase 用户的只读区域。这里没有宣称 AIR 支持任意跨 task/stage 的 SRAM 驻留。

## CPU fanout：从实际 body 到共享 copy

`test/TaskflowGeneric/data-organization.cpp` 内嵌可解析 Taskflow MLIR：任务 A 读取输入并产生临时 X，B/C 实际读取 X 并产生输出。C 还通过声明接口依赖 B，但 body 不读取 B；因此生成一条纯排序边，不搬运 B。

测试调用：

```text
taskflow-data-organization-test
```

最终公共 CPU 检查为 **PASS 39**（`build/data-organization-v8/final-shared-copy-checks.json`），native mapping 检查为 **PASS 36**（`final-native-mapping-checks.json`）。费用归属、独立 checker 溢出边界、显式 clock 转换和 native adapter 伪造字段负例均已包含；统一增量构建与 Core 检查日志为 `final-checker-build-core-r1.log`。较早 PASS 34 的运行命令、测试二进制 SHA 和计划保存在 `shared-copy-checks-r2/`，与最终结果分开保留。这里仅报告 CPU 验证，不代表硬件执行验收。

目标是显式 CPU synthetic 架构：A 在 DRAM 可访问的 CPU 上，B/C 在 SRAM 可访问的计算资源上；DRAM→SRAM 链路参数来自测试常量，不能解释为 NPU 或 CGRA 测量。

- phase-local：B/C 各导入 64 B X，创建两个副本，流量 128 B。
- retain-across-users：B/C 引用同一个 64 B X 副本，创建一次传输，流量 64 B。
- B/C 输出和 X 共同存活时 SRAM 事件峰值为 192 B；数据边数不会重复增加容量。
- 每 task 的显式 synthetic compute estimate 为 1 ms；链路为 0.1 ms 启动与 64 B/ms。串行全图估计分别为 5.2 ms 与 4.1 ms，均为未测量的测试参数结果。

该 whole-task 适配器保守要求每个实际访问是可证明精确的整存储区域，选择显式 task placement，使用直接链路，拒绝需额外临时空间的传输。它不是新增后端调度器；事件按源拓扑顺序串行安排。有限范围外明确拒绝：真实 `splitNativeOuterLoops` 后出现半区域访问不会沿用旧副本。另一个 shape 16→8 的真实 MLIR 改动经重新分析可重建 32 B 共享副本，旧 fingerprint 的计划被拒绝。

## 检查与会计

whole-task checker 独立重放候选事件，对照实际 MLIR 读写集合验证版本生产、copy 创建、读前可用、最新版本、消费位置/布局/dtype、创建依赖、数据边及纯排序边。它不通过重新生成整份 producer JSON 判等。target 表、事件资源、明确的 compute provider 条件以及每项费用也逐项核对。

whole-task capacity 按 `availableFrom <= event < releasedAfter` 的实际串行事件区间计算，同一 allocation 只计一次；报告 `memory_peak_bytes`。allocation 的 phase 起止只是保守的可读摘要。native 的固定 pool 容量仍按 target hardware stage 的实际 pool 分配与寿命检查，其外层 mapping 报告给出 payload 峰值；公共 copy 表给出事件有效期。二者的坐标和会计范围在报告中区分。

负例包括：错误 region/version/consumer requirement、缺创建依赖、提前释放、缺用户/边、重复分配/收费、源 fingerprint 不符、容量不足、非法目标资源、伪造全图成本以及 creator/version/读写/allocation 索引越界。所有索引在解引用前验证。独立 checker 还在容量和流量算术前拒绝超出 int64 的源 storage 字节数及非正 allocation 字节；超大静态 MLIR metadata 负例只做 parse/analyze/check，不实际分配或执行循环。

native 仍使用既有独立 cell/version/copy checker，并新增数据/顺序边、所有 copy 用户和消费边 incidence 检查。大计划超过展开阈值时省略逐项 JSON，但完整受检表参与 semantic digest；不会由 Python 重建副本策略。

## 成本契约与当前边界

`ComputeEstimate.h` 是纯值接口。后端提交既有任务 duration、单位、模型及版本、provenance、计费边界、资源数量/形状、包含/排除费用和供数条件；转换函数要求正有限 duration 和已验证条件。cycles 必须有显式 MHz，不能将 II 当 duration，也不默认每任务 1 ms。`transportScope` 是机器可检查的费用归属：公共 combiner 只接受明确 `ExcludesInterTaskTransfers` 的任务 duration；未说明或已包含 inter-task transfers 的复合估计拒绝组合，不能只凭自由文本推测排除关系。测试另验证 1,000,000 cycles / 1,000 MHz = 1 ms，与原毫秒输入得到相同全图费用。缺任务估计或未标定链路使完整估计为 null。

公共 whole-task 估计以已有 compute duration 加显式直接链路费用，串行消费，避免再次计算模型已包含的部分。synthetic 参数与真实 CGRA 模型分开；未建立真实 CGRA 数据位置变化下的校准模型、任意链路多跳路由或计算/搬运重叠优化。native AIR 的 scalar 操作理论成本仍不能当实测 wall time。硬件能力与数值验收由各后端实验独立记录。

联合搜索、规范化身份、cache/checkpoint 与容量耦合两轮候选由原搜索入口处理；本公共层提供语义事实、数据组织选择和受检计划，不另建搜索器。
