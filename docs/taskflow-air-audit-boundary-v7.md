# v7 AIR 端到端审计边界

本次仅检查源码和已有产物，未构建、修改编译源或提交 NPU 作业。检查链是：真实 Taskflow body → 公共版本/位置副本计划 → canonical AIR → 安装工具链 → 最终 runtime/AIE/linker 产物 → 原执行器及独立数值检查。各层证据不能互相替代。

| 检查层 | 已检查 | 不能据此证明 |
|---|---|---|
| 公共计划与 canonical AIR | 真实访问、最新版本、副本生命周期、容量和依赖；实际 DMA 的 event/copy、BO/pool key、位置与字节绑定；canonical checker 对照源和计划重新生成完整 AIR | 任意后续 AIR/AIE pass 的形式化语义等价 |
| 最终 runtime BO 描述符 | [auditor](../test/TaskflowAIR/audit_runtime_transfers_v7.py) 在计划指定的 stage/sequence 中，识别静态 `dma_configure_task_for` 与直接 `%argN` 的 `dma_bd`，按 BD `len × (repeat_count+1) × dtype bytes` 比较各 storage/方向。最终大计划使用 C++ 从受检 transfers 得出的逐 stage summary；长验收拒绝旧的仅全图汇总回退 | 物理 DDR 计数、实际传输完成、BD 地址/offset/stride 正确性、root main 调用这些 sequence 的顺序。它不是任意 IR 的全部 shim BO 字节完备检查：额外 runtime 函数、其它 DMA 形式及未识别语法不在遍历契约内 |
| L2/memtile 与内部 transport | 最终 AIE IR 可检查实际声明；当前 M128 extent8 产物的 12 个 `aie.buffer` 全位于 compute tile、memory space 2，未见 memtile/L2 buffer 声明 | runtime BO ledger 不统计片内额外跳数、memtile staging、数据重放或协议开销；当前观察不能推广为任意候选均无 L2 staging |
| L1 payload 与 stack | [storage audit](../test/TaskflowAIR/audit_storage_v6.py) 检查最终 core linker 脚本中已识别 buffer 区间、相互/stack 不重叠、对应 stage 的实际 payload 与计划预算。当前 M128 模板最大 payload 为 20480B，四 stage 的额外 payload 均为 0；另保留 4096B stack | 24576B tensor payload 不是全部物理占用。对齐空洞、代码/运行时数据、描述符和 stack 另计；reserved stack 不是实测 stack high-water。此工具不提供全芯片 L2/运行时内存峰值，安装编译器的分配/link 检查仍是目标边界 |
| 端口、FIFO 与依赖 | 必要的 tail 路径由计划指定单逻辑 FIFO 与 compute DMA port；C++ 核对 endpoint 的 storage/方向、静态消息长度/次数/顺序，序列化后重检。[port audit](../test/TaskflowAIR/audit_channel_ports_v6.py) 将计划端口关联到实际 compute BD；安装 routing verifier 独立运行 | compute-port 一致不等于全部 shim routing、队列/锁调度或无死锁证明。port audit 不重放 FIFO/lock 执行；公共 `npu.dma` 逻辑资源也不是物理 engine 数。常规 FFN 模板无必要的 channel legalization，不能声称它运行过 tail-port 专项检查 |

当前最终 M128 extent8 的 `npu.input.air.mlir` 共 24 个直接 BO 参数 BD，runtime 报告统计 24 个；panel r2 对应 14/14，检查时未见其它 host DMA 形式。这补充了**这些具体产物**的遍历覆盖，不消除上表的通用解析边界。反例也保留：panel r1 的最终 IR 缺失 A producer，审计报预期 65536B、实际 0；按公共 panel scope 分组后的 r2 先通过逐 stage 审计，再真实执行完整输出通过。该反例说明原生 expected-IR checker 或编译成功本身不足以证明最终 producer 完整。

[后端](../lib/TaskflowAIR/AIRBackend.cpp) 的 scalar 路径仍克隆实际 task 操作，只限制源循环、重映射局部地址并按计划处理驻留；没有根据 GEMM/FFN 名称替换计算公式。超容量 add→scale 使用相同路径；FFN template 仍包括真实 Gate、Up、`SiLU(G)*U`、Down。公共 partition 只缩小受证明独立的轴及对应 memref 维度，保留每个 template 内的源归约顺序。显式 exp 近似是另行记录的数值选择；严格 FP32 vector multiply 在当前 Peano 不可用，未用 BF16 仿真替代。

[数值与执行证据](taskflow-direct-air-v7-validation.md) 另绑定源、计划、ABI、AIR、ELF 和真实 command/输出。旧 add-scale 硬件证据与当前源码/CPU 检查不能冒充新二进制的又一次设备执行。XRT BO 同步复用是系统内存准备策略，也不等于 AIE 权重驻留。M128 的最终多 seed 状态以验证报告及保存输出为准；静态字节账、`xrt-smi` 的 active submission 或上述检查均不提供 core trace、内部进度、实测 DDR 流量或加速结论。
