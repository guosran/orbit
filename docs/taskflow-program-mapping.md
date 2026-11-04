# 完整程序候选与公共 mapping

ORBIT 的候选是一个完整程序的实现方案，不是一个 kernel 配置。计算语义由 Taskflow body、循环、访存与依赖保存；公共搜索改变任务结构、逻辑实例组织、资源分配、调度和数据组织；后端消费被检查的方案，提供目标约束及成本反馈。

## 三层职责

| 层 | 信息和职责 | 本轮实现位置 |
|---|---|---|
| 计算 | 真实计算、访问、依赖、归约更新；分析推导 region 和共享事实 | Taskflow dialect、GenericAnalysis、NativeTransforms |
| 公共候选与 mapping | 结构动作链、实例遍历与分组、资源位置、区域版本、副本、驻留区间、搬运和依赖、完整图成本 | Search、GenericMappingPlan、ProgramMapping.h |
| 目标实现 | 能力约束、具体实现选择、layout/scratch 合法化、目标资源绑定及反馈 | Amoeba mapper/scheduler；AIRBackend 的 hierarchy/memspace/DMA/channel/token |

Taskflow dialect 最初服务 Neura；GenericAnalysis/GenericMappingPlan 是后加的原生 body 路径。当前并未把旧 StaticPlan、GenericMappingPlan 和 Neura mapper 合成一个已经覆盖所有目标的调度器。共同部分先复用目标 memory/link/resource 契约，并统一原生路径的数据副本和事件语义；旧 structured/Neura 路径的能力和限制继续独立说明。

## 一个候选包含什么

- **结构**：实际 Taskflow MLIR、初始化分离等公共变换、计算 tile 和归约 tile；原 Neura/structured 路径仍保留各自已有 shape、replica、fusion/fission 动作。
- **资源与实例组织**：逻辑 phase 对应源 task 的受限迭代域，execution group 将实例分配到目标 compute resource 和 memory；依赖给出顺序。资源标识由目标配置赋值，公共数据结构不将其解释为 AIR tile。
- **数据组织**：`ProgramMappingChoice` 选择逐 phase 数据搬运或证明合法的只读面板驻留；`taskLoopOrders` 指定源循环维度对应的实例遍历顺序。readonly-panel 的 storage/reuse-loop/panel-loop 不等于算子类型。
- **实现**：每任务 implementation 记录 scalar 或目标支持的 independent-lane-f32、选中独立维和 lane 数。它不是改变 FP32 归约次序的许可。
- **推导结果**：受限访问 region、版本和副本、实际传输、生命周期、容量/端口需求、成本状态以及 AIR 专属资源绑定。

这些选择进入实际 MLIR 属性或 lowering 消费的计划；候选保存父 ID、动作链、评估和选择理由。默认策略按规范形式存储，避免仅多写一个默认属性就生成重复执行方案。

## 从数据关系到副本和通信

公共通信计划复用 `StaticMemory`、`StaticResource`、`StaticTransfer` 的 memory/resource/link 概念，采用任意 rank 的 `MappingRegion`；旧 StaticAccess 的二维 Range 不作为原生分析限制。精确逻辑 cell 只划分数据区域，不另建一套计算图。

`MappingVersion` 指向区域和产生事件；`MappingCopy` 指向版本、allocation、创建、可用和释放区间及使用者。事件读取具体位置的副本，计算产生新版本，传输在两个位置创建同一版本的副本。系统内存 BO 与 stage-local pool 是不同 allocation；BO 常驻不推导成片上常驻。旧 GenericTransfer 的后端方向通过公共事件和副本绑定解释，避免只有一份无法影响代码的报告。

通信推导和检查的顺序为：源事实 → 结构与 mapping 选择 → 实例和驻留区间 → 副本/事件 → 容量、版本、依赖和资源检查 → 完整图评价 → 后端。AIR 若引入新布局、padding、scratch 或搬运，应记录并重新检查；最终编译器物理行为仍需单独审计。

## 可执行搜索实验

`test/TaskflowAIR/fixtures/gemm-scale.mlir` 只给计算、存储接口和任务依赖。分析发现 B 的读取不使用 M 输出轴，也发现 A 的读取不使用 N 输出轴；独立输出投影证明决定可否改变实例遍历。候选生成不依赖 `gemm`、`weight` 或 `B` 名字。

`test/TaskflowGeneric/program_search_v7.py` 复用原 C++ 搜索驱动，组合计算分块与自动发现的驻留动作。目标示例中，保留完整 K 面板减少重复输入读取，但扩大局部池；过大的面板被容量检查拒绝。评价涵盖后继 scale 的中间量搬运和输出，不只计算第一任务的权重流量。

硬件执行状态、最终候选身份、传输对应和测量以 `build/direct-air-v7/RESULTS.md` 为准。静态合法不等于硬件通过；首轮面板的真实等待失败也必须保留。简单理论成本是显式串行敏感性模型，未知或未标定部分不冒充测量。后续扩展任务合并、跨资源复制/广播、更多 shape 和重叠策略时，继续使用同一个完整程序候选和动作链。

Dato 对照见 [taskflow-dato-overlap-v7.md](taskflow-dato-overlap-v7.md)：借鉴其通信契约与虚实映射思想，同时区分用户已声明的 stream/layout 决策与 ORBIT 从计算访问推导并搜索的方案。

## 整程序命令分片也是结构与调度选择

`ProgramPartitionPlan` 是受限的整程序 fission：所有任务必须沿同一独立外轴写出互不重叠的区域，共享存储在各任务间必须使用相同轴；只有只读输入允许不随该轴变化。C++ 检查拒绝跨片写、索引轴作为数值参与运算、非投影访问、输出覆盖不足和不一致 witness。分片模板克隆真实 body 并收缩已证明独立的轴，K 归约及算术保持原顺序。非整除轴产生完整块和尾块两种模板，而非展开整个大程序。

选择通过 `orbit.program_partition_extent` 留在候选 MLIR，并由原搜索的 `native_program_partition_candidates` 邻域改变。公共计划声明每次完整任务图调用、依赖、输入/输出区域和只读复制；AIR 后端编译每种模板，既有执行器按计划切片和拼接。JSON manifest 是 C++/MLIR 计划的执行报告，不是 Python 提出的新程序图。

该选择同时改变工作集、单次命令工作量、命令次数和主机准备成本。M128 FFN 的单命令标量实现触发已安装驱动的约 60 秒 watchdog；M8 完整图实测约 36 秒，因此选择 16 次 M8 程序片。跨调用复用权重 BO 避免重复主机上传；它不减少各次命令中的片上权重搬运，也不表示片上常驻。当前理论评价累加子计划计算/搬运/固定项，主机切片、配置和 watchdog 时间预测仍未标定；不能仅据该敏感性模型自动保证命令截止时间。

这种受限整程序 fission 与 tile/residency 联合邻域均由原搜索承载。本轮尚未把所有 Neura shape/replica/scheduling 动作迁移到 native AIR，也未完成异构任务迁移、广播、双缓冲或任意 DAG 分割。公共类型为这些后续动作提供区域版本、副本和资源契约，不把当前验证范围写成已实现的全搜索空间。
