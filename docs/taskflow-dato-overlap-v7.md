# ORBIT 与 Dato：决策来源和公共 mapping 的边界

ORBIT 的优化对象是完整 program 候选。候选共同携带计算结构、逻辑实例划分、资源分配、执行顺序与数据组织选择；Amoeba/NPU 提供能力、合法化与成本反馈。通信计划是这些选择的推导结果，不要求用户先写出 AIR stream graph。

## 论文描述

Dato 用户可声明任务虚拟网格、Stream put/get 和 Layout 分片/复制；默认 layout 为复制。编译器检查通信与归约义务，并将虚拟图映射到有限物理资源。bundle 将同构实例放到同一资源多次执行；graph-chain 合并相连生产者/消费者。两者既可手工指定，也能由 §4.3 的搜索自动组合，不能将 Dato 描述成只有人工 mapping。§6 采用保守生命周期、epoch 和端口预算组织 DMA，并讨论 multicast、合并与拆分。这里的 graph-chain 是计算图变换，不是 XRT NPUChain 提交接口。[原论文 v1，§3/§4/§6](https://arxiv.org/html/2509.06794v1)

## 源码核查范围

检查仓库 `cornell-zhang/allo` 的 revision `abc270c7a1919c780df2b7d7e99de45e81c175b9`，即检查时 main；不声称它就是论文实验 revision。选定文件和 SHA 保存在本机 `build/direct-air-v7/dato-source/manifest.json`。

- [`allo/backend/aie/mapping.py`](https://github.com/cornell-zhang/allo/blob/abc270c7a1919c780df2b7d7e99de45e81c175b9/allo/backend/aie/mapping.py)：可确认 ComputationGraph、bundle_multi、chain、带 token/first_use/last_use 的 LiveDTensorTile 和按生命周期分组。初始化仍有粗粒度 `[0,9]` 与更精细 liveness 的 TODO，不能将论文机制概括成源码已实现精确跨 tile 活跃分析。
- [`allo/backend/aie/__init__.py`](https://github.com/cornell-zhang/allo/blob/abc270c7a1919c780df2b7d7e99de45e81c175b9/allo/backend/aie/__init__.py)：检查的 build 入口消费显式 mapping_primitives，随后 refactor 并检查逻辑节点数不超过物理节点数。该入口本身不是论文 Algorithm 1 的自动搜索证据；本次未确认该算法在此 revision 的完整实现位置，也不据此否认论文的自动映射能力。
- [`allo/backend/aie/mlir_codegen.py`](https://github.com/cornell-zhang/allo/blob/abc270c7a1919c780df2b7d7e99de45e81c175b9/allo/backend/aie/mlir_codegen.py)：可确认 global DMA task/token、multicast interface、memory access coalescing 和端口分配路径；没有在 ORBIT 机器上运行 Allo 来重现其性能或验证所有合法性声称。
- [`tests/dataflow/aie/test_mapping_gemm.py`](https://github.com/cornell-zhang/allo/blob/abc270c7a1919c780df2b7d7e99de45e81c175b9/tests/dataflow/aie/test_mapping_gemm.py)：测试调用显式 bundle/chain 序列，说明可编程原语接口；不等同于自动搜索覆盖证明。

## 与原 ORBIT 的重合和缺口

| 概念 | 与 ORBIT 已有动作的关系 | 公共层需要补充的内容 |
|---|---|---|
| 虚拟实例到物理资源 | 与 replica、shape、temporal scheduling 的目标重合 | 实例分组与资源映射必须独立于任务 body 和后端 hierarchy |
| 同资源多次执行 | 与 temporal scheduling、任务分组部分重合 | 明确顺序、资源占用及输入副本能否跨实例保留 |
| graph-chain | 与 fusion 的局部性目标重合；并非所有 fusion 都是 chain | 合并前后中间副本、依赖和外部通信变化 |
| shard/layout | 与 tiling、K-tiling 及复制决策耦合 | 任意 rank region、数据版本、部分结果义务 |
| DMA epoch 与生命周期 | 是既有存储/link/resource 模型的具体合法化例子 | 副本的创建、使用、失效、释放以及端口需求契约 |

本轮采用的思想是：用公共实例、存储副本和通信事件表达选择，用独立检查器验证，再由 AIR 实现。候选可以改变块大小与只读面板驻留范围；面板减少重复搬运，同时增加局部容量并限制合法 tile，不能只按节省的权重字节排序。容量拒绝、实际传输与完整程序计时共同约束选择。

后续空间包括 producer/consumer 跨任务分组、广播/多播、跨资源复制、双缓冲、重叠及更多 resource shape。它们应作为原联合搜索动作扩展，不能因为有了公共类型就宣称已实现。

## 三类信息不能混淆

1. **源语义事实**：实际 body 推导访问 region、输入不变性、共享关系、依赖与归约更新顺序。它们不由候选策略属性宣称。
2. **搜索选择**：结构分块、实例分组、执行顺序、驻留位置与区间、实现选择；候选身份及父动作链绑定实际 MLIR 与计划。
3. **后端机制**：AIR hierarchy、memory space、DMA、channel、FIFO 和 token；额外资源或搬运必须回报并检查。

示例输入只包含计算、访存和任务依赖。编译器从某只读输入的访问不使用一个独立输出轴，发现跨该轴实例的复用；搜索可选择逐块搬运，或交换合法的实例遍历顺序并保留一个面板。用户不必先标注“这是权重”或手写 stream。实际可执行候选、计划和测量结果以 v7 实验报告为准。
