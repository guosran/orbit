# v7 公共 program mapping 一致性审阅

本次审阅未发现当前 scalar/readonly-panel 路径的重大计划不一致。公共类型位于 `ProgramMapping.h`，原生适配器为 `GenericCommunication.cpp`；任务实例继续使用现有 phase，没有重建计算图。memory/link/resource 直接复用 `StaticMapping.h` 的契约类型，区域 cell、版本、位置副本和事件采用其已有概念的任意 rank 适配。旧 structured Static planner 的语义未改，Amoeba 尚未消费这份原生副本计划。

实际访存、循环和依赖是 facts。`discoverGenericMappingChoices` 从不可写 input 的实际 SSA 地址与输出投影发现复用机会；它返回候选，不承诺容量或目标可执行性。当前 choices 为 phase-local 或 readonly-panel，后者可交换已证明独立的两个外轴，保留源归约顺序。panel 是实际需求的矩形包围区域；可能包含未被计算访问但已初始化的输入元素，搬运和容量均按整个矩形计。显式超容量选择拒绝，不自动改变 tile。

公共 `executionGroups` 记录 phase 范围、资源、位置及组间调度；AIR hardware stage 绑定对应 group。readonly-panel 使用每 panel scope 一个 group，跨 reuse 轴和内部归约复用输入，完整 panel 不适配容量时拒绝。这一有限分组策略也是当前安装工具链的必要适配：早期合并多个 panel 的产物丢失输入 producer，历史失败保留。它不代表任意分组均可执行。

公共 checker 根据源 phase 和受检 DMA 独立重放 cell 的最新版本及每个 allocation 中的副本。它检查创建前使用、陈旧版本、覆盖后使用、最后用户、释放/复用时机、完整输出、传输源目的、位置/容量及任务实例覆盖。版本表示一个实例结束时 cell 的内容，不是每次 scalar store 单独编号。allocation 的生命周期使用 phase ordinal；copy 的可用/释放和最后用户使用 event ordinal，两种坐标不得混用。

事件依赖为真实 partial order：上一 phase 的全部 terminal events、read copy 的创建事件、覆写物理池之前的创建/最后使用事件。event ID 是一种拓扑编号，不要求 AIR 文本按编号排序；同 phase 的 A/B 输入搬运互不依赖，纯本地初始化也不依赖无关输入。没有为每个 event 强加前一个 event 的边。AIR adapter 对实际 DMA 核验 public event/copy IDs、BO/pool key、位置、字节及区域，再克隆真实 body；独立 canonical AIR 检查保护这些绑定。公共 checker 不是任意 AIR/AIE 转换的完整形式证明。

容量仍为 24576B 的 tensor payload 策略预算。每 stage 的 readonly template/panel、writable 和必要 incoming 池全部计入，不能把 source live region 当实际分配峰值。host BO 和 AIE L1 分开；system memory 的 `INT64_MAX` 是无额外限制的会计上界，不是实测可用 RAM。`npu.dma` 的单槽是逻辑 issue witness，不能解释成物理 DMA engine 数或物理搬运完全串行。端口、FIFO、安装工具链 staging 和最终 linker 审计属于 mechanism/实际资源核验，不能由逻辑 DMA 字节推出物理 DDR counter。默认成本仍未标定；没有按 group 配置、host slicing 或同步成本校准完整图时间。

大计划内部仍枚举 phase/events/copies；超过报告阈值时只省略展开 JSON，保留完整语义 digest 和计数。每 hardware stage 的 `transfer_summary` 直接从受检 transfers 过滤统计，包含 storage、方向、次数和字节，供最终 runtime producer 审计使用，不由 Python 重算规划。phase、exact logical cell 和整程序 invocation 分别有 1048576 的 host 展开上限，hardware stage 总数上限为 128；均不是物理 SRAM 容量。

已实跑旧 tiled 48 项、新 program 30 项（`build/direct-air-v7/indirect-lane-guard-r1`），最终父级会计/身份增量的 program partition 56 项另见 `build/direct-air-v7/parent-bo-accounting-r2`；较早 partial-order 版本的 48/28 项证据也保留。新正例由实际访问发现跨外轴输入复用，DMA 从 576B 降到 448B；负例覆盖版本、副本、依赖、释放、组/池、link/resource 和 DMA 绑定。最终 M128 partition manifest 的 native 静态产物另经只读交叉检查：16 个 extent=8 的完整 program 调用覆盖 extent=128；每 template 为 17060 phases、50976 events、67520 copies，四组实际池峰值 20480B，host BO 271720448B；逐 stage DMA 合计 306176000B，与 template 总数一致。此处仅是静态一致性证据，长 NPU 验收进行中，不能据此声称 M128 数值通过。

当前不支持任意并行实例调度、异构自动放置、任意布局/动态域、双缓冲或自动预取。partition 含多个 extent template 时，父级按各 unique template 的 BO 字节求和，生命周期覆盖完整 seed/program，重复调用复用已有 BO，不按 invocation 次数重复计峰值；scope 排除 full host arrays、reference、context 及 runtime 开销。两模板 640+192=832B 的正例及元数据 8e18B/模板的溢出负例通过，未实际分配巨大 BO。传输字节乘加与理论成本乘加也检查非法值和溢出。tiled host feedback 没有可证明本地 initializer 时后端仍拒绝。`independent-lane-f32` 有源选择/覆盖报告，但安装 Peano 的严格 FP32 vector multiply 不可执行，评价返回 unsupported，不进入可执行候选评分；源端已补 fail-closed：`affine.apply` 间接携带 lane IV 的读地址不再被当作 invariant broadcast；永久 stride-2 负例验证 scalar coverage，并拒绝伪造的 vector implementation。当前证明仍只支持直接 identity-last-dimension 或真正 invariant 的地址，不是任意 affine 地址的向量化证明。

最终 guard 编译器 SHA 为 `ad0af8e27877f401c8f159f1b19db769152e4625a7b9795e70f6b72b2a5822f1`，M128 已送长验收的原产物编译器 SHA 为 `3e041bf53ebb3747e347b251dfc431af1a9f7ea95c6cf9ec74dd613fba68945c`。`build/direct-air-v7/final-guard-m128-equivalence-r1/result.json` 记录新工具重新原生生成的七份语义文件逐字节相等（whole Taskflow/config 与 template Taskflow/plan/facts/AIR/ABI），manifest 仅排除 `compiler_invocation` 后完全相等。两个 compiler identity、命令和日志分别保存。此审计没有重新编译 AIR 或触及设备，不能把它称为新编译器的另一次硬件执行。

最后父级会计增量工具 SHA 为 `2444a3b5e41baf3da7a5d8892b64b186d3597b660806a580e2641ab8e5106ac4`。`parent-bo-accounting-r2/result.json` 再次验证上述七份子语义文件逐字节不变；父 manifest 仅增加 BO/展开上限说明与身份字段。原源分片身份保留为 `partition_witness_id`，新的完整映射 `plan_id` 绑定 witness、template extent/count 和各 child mapping ID，不同 target tile 配置不再共享父映射 ID。旧长验收 manifest、原 ID 和 JSON 身份均保留；新增报告及身份没有冒充原硬件执行所用身份。
